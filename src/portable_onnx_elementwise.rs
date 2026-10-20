//! Bounded static arithmetic: exact integer intermediates and explicit symbolic terms.
use crate::portable_onnx_dynamic::{key, known, symbol};
use crate::portable_onnx_node::{exact_dims, group, kd, kernel, set, shape, static_numbers};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
fn bounded(s: &Value) -> Option<usize> {
    let ds = rows(s).iter().map(dv).collect::<Option<Vec<_>>>()?;
    if ds.iter().any(|n| *n < 0) {
        return None;
    }
    if ds.contains(&0) {
        return Some(0);
    }
    ds.iter().try_fold(1usize, |a, n| {
        usize::try_from(*n)
            .ok()
            .and_then(|n| a.checked_mul(n))
            .filter(|n| *n <= 65536)
    })
}
fn term(op: &str, terms: &[Value]) -> Option<Value> {
    let mut current = terms.first()?.clone();
    for next in &terms[1..] {
        let (a, b) = (dv(&current), dv(next));
        if let (Some(a), Some(b)) = (a, b) {
            let (a, b) = (BigInt::from(a), BigInt::from(b));
            let n = match op {
                "Add" | "Sum" => a + b,
                "Sub" => a - b,
                "Mul" => a * b,
                "Div" if !b.is_zero() => a / b,
                "Max" => a.max(b),
                "Min" => a.min(b),
                _ => return None,
            };
            current = dim(n.to_i64().filter(|n| n.unsigned_abs() <= SAFE as u64)?);
            continue;
        }
        if !known(&current) || !known(next) {
            return None;
        }
        if ((op == "Add" || op == "Sum") && b == Some(0)) || (op == "Mul" && b == Some(1)) {
            continue;
        }
        if ((op == "Add" || op == "Sum") && a == Some(0)) || (op == "Mul" && a == Some(1)) {
            current = next.clone();
            continue;
        }
        if op == "Div" && b == Some(0) {
            return None;
        }
        current = symbol(format!(
            "deepbom_expr:{}({},{})",
            if op == "Sum" {
                "add".into()
            } else {
                op.to_lowercase()
            },
            key(&current),
            key(next)
        ));
    }
    Some(current)
}
fn compare(op: &str, a: f64, b: f64) -> bool {
    match op {
        "Equal" => a == b,
        "Greater" => a > b,
        "GreaterOrEqual" => a >= b,
        "Less" => a < b,
        _ => a <= b,
    }
}
fn compare_terms(op: &str, a: &Value, b: &Value) -> Option<bool> {
    let (x, y) = (dv(a), dv(b));
    if let (Some(x), Some(y)) = (x, y) {
        return Some(compare(op, x as f64, y as f64));
    }
    if a["kind"] == "symbolic" && y.is_some_and(|n| n < 0) {
        return Some(op == "Greater" || op == "GreaterOrEqual");
    }
    if b["kind"] == "symbolic" && x.is_some_and(|n| n < 0) {
        return Some(op == "Less" || op == "LessOrEqual");
    }
    if known(a) && known(b) && key(a) == key(b) {
        Some(compare(op, 0., 0.))
    } else {
        None
    }
}
fn lists(inputs: &[&Value], count: usize, dimensions: bool) -> Option<Vec<Vec<Value>>> {
    let values = inputs
        .iter()
        .map(|t| {
            if dimensions {
                exact_dims(t)
            } else {
                static_numbers(t)
            }
        })
        .collect::<Option<Vec<_>>>()?;
    if values.iter().all(|v| v.len() == 1 || v.len() == count) {
        Some(values)
    } else {
        None
    }
}
fn at(v: &[Value], i: usize) -> &Value {
    &v[if v.len() == 1 { 0 } else { i }]
}
fn propagate(op: &str, inputs: &[&Value], s: &Value) -> Option<Value> {
    let count = bounded(s)?;
    if op == "Where" {
        if kd(inputs[0]) != "BOOL" {
            return None;
        }
        let conditions = static_numbers(inputs[0])?;
        if !(conditions.len() == 1 || conditions.len() == count)
            || conditions
                .iter()
                .any(|v| !matches!(v.as_f64(), Some(0.) | Some(1.)))
        {
            return None;
        }
        for dims in [true, false] {
            if let Some(vals) = lists(&inputs[1..], count, dims) {
                let result = (0..count)
                    .map(|i| {
                        at(
                            &vals[if at(&conditions, i).as_f64() == Some(1.) {
                                0
                            } else {
                                1
                            }],
                            i,
                        )
                        .clone()
                    })
                    .collect::<Vec<_>>();
                if dims && !result.iter().all(known) {
                    continue;
                }
                return Some(if dims {
                    json!({"dimensions":result,"source":op})
                } else {
                    json!({"values":result,"source":op})
                });
            }
        }
        return None;
    }
    if group("BROADCAST_BOOL", op) {
        if inputs.len() != 2 {
            return None;
        }
        if let Some(vals) = lists(inputs, count, true) {
            if let Some(result) = (0..count)
                .map(|i| compare_terms(op, at(&vals[0], i), at(&vals[1], i)))
                .collect::<Option<Vec<_>>>()
            {
                return Some(json!({"values":result,"source":op}));
            }
        }
        let vals = lists(inputs, count, false)?;
        let result = (0..count)
            .map(|i| {
                Some(compare(
                    op,
                    at(&vals[0], i).as_f64()?,
                    at(&vals[1], i).as_f64()?,
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        return Some(json!({"values":result,"source":op}));
    }
    if !["Add", "Sub", "Mul", "Div", "Max", "Min", "Sum"].contains(&op) {
        return None;
    }
    let vals = lists(inputs, count, true)?;
    let result = (0..count)
        .map(|i| {
            term(
                op,
                &vals.iter().map(|v| at(v, i).clone()).collect::<Vec<_>>(),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    Some(json!({"dimensions":result,"source":op}))
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let op = string(&v["node"]["opType"]);
    if !group("BROADCAST_SAME_TYPE", op) && !group("BROADCAST_BOOL", op) && op != "Where" {
        return crate::portable_onnx_special::run(v);
    }
    let inputs = rows(&v["node"]["inputs"])
        .iter()
        .filter(|n| truth(n))
        .map(|n| &v["tensors"][string(n)])
        .collect::<Vec<_>>();
    if inputs.is_empty()
        || inputs.iter().any(|t| shape(t).is_null())
        || op == "Where" && inputs.len() != 3
    {
        return Ok(unresolved(if op == "Where" {
            "where_input_shape_unknown"
        } else {
            "broadcast_input_shape_unknown"
        }));
    }
    let shapes = inputs.iter().map(|t| shape(t)).collect::<Vec<_>>();
    let s = kernel("broadcast", json!({"shapes":shapes}))?;
    if s.is_null() {
        return Ok(unresolved(if op == "Where" {
            "where_broadcast_incompatible"
        } else {
            "broadcast_shapes_incompatible"
        }));
    }
    let od = if op == "Where" {
        if kd(inputs[1]).is_empty() {
            kd(inputs[2])
        } else {
            kd(inputs[1])
        }
    } else if group("BROADCAST_BOOL", op) {
        "BOOL"
    } else {
        inputs
            .iter()
            .take(2)
            .map(|t| kd(t))
            .find(|s| !s.is_empty())
            .unwrap_or("UNKNOWN")
    };
    let vals = propagate(op, &inputs, &s).unwrap_or(Value::Null);
    let mut out = vec![];
    if set(v, &mut out, 0, &s, od, vals) {
        Ok(result(out))
    } else {
        Ok(unresolved(if op == "Where" {
            "where_broadcast_incompatible"
        } else {
            "broadcast_output_unresolved"
        }))
    }
}
