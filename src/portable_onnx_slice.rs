//! Slice controls retain INT64 bounds without rounding through a host number.
use crate::portable_onnx_dynamic::{axes, encode, invalid, key, known, symbol};
use crate::portable_onnx_node::{exact_dims, kd, kernel, set, shape, static_numbers};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde_json::{json, Value};
#[derive(Default)]
struct Control {
    present: bool,
    values: Option<Vec<Value>>,
    terms: Option<Vec<Value>>,
    len: Option<i64>,
    invalid: bool,
}
fn integers(t: &Value, unbounded: bool) -> Option<Vec<Value>> {
    let vals = if let Some(vals) = static_numbers(t) {
        vals.iter()
            .map(|v| safe(v).map(|n| BigInt::from(n)))
            .collect::<Option<Vec<_>>>()?
    } else if t["staticValuesCanonicalTextComplete"] == true {
        t["staticValuesCanonicalTexts"]
            .as_array()?
            .iter()
            .map(|v| {
                let s = v.as_str()?;
                if s.len() > 32768 || s.is_empty() || s.starts_with('+') || s.trim() != s {
                    return None;
                }
                s.parse::<BigInt>().ok()
            })
            .collect::<Option<Vec<_>>>()?
    } else {
        return None;
    };
    vals.iter()
        .map(|n| {
            let text = n.to_string();
            if unbounded {
                let d = kd(t);
                if d == "INT32" && text == "2147483647"
                    || d == "INT64" && text == "9223372036854775807"
                {
                    return Some(json!("+inf"));
                }
                if d == "INT32" && text == "-2147483648"
                    || d == "INT64" && text == "-9223372036854775808"
                {
                    return Some(json!("-inf"));
                }
            }
            if let Some(n) = n.to_i64().filter(|n| n.unsigned_abs() <= SAFE as u64) {
                Some(json!(n))
            } else if unbounded {
                Some(json!(text))
            } else {
                None
            }
        })
        .collect()
}
fn control(v: &Value, i: usize, k: &str, bounds: bool) -> Control {
    let n = &v["node"];
    if truth(&n["inputs"][i]) {
        let t = input(v, i);
        let vals = integers(t, bounds);
        let mut terms = if bounds { exact_dims(t) } else { None };
        let ds = if shape(t).is_null() {
            None
        } else {
            Some(dimensions(t))
        };
        let dt = kd(t);
        let declared = ds
            .as_ref()
            .filter(|d| d.len() == 1)
            .and_then(|d| dv(&d[0]))
            .filter(|n| *n >= 0);
        let invalid = (static_numbers(t).is_some()
            || t["staticValuesCanonicalTextComplete"] == true)
            && vals.is_none()
            || !dt.is_empty() && !matches!(dt, "INT32" | "INT64")
            || ds.as_ref().is_some_and(|d| d.len() != 1)
            || vals
                .as_ref()
                .zip(declared)
                .is_some_and(|(v, n)| v.len() as i64 != n);
        if !invalid
            && terms.is_none()
            && vals.is_none()
            && bounds
            && matches!(dt, "INT32" | "INT64")
            && declared.is_some_and(|n| n <= 4096)
        {
            let name = encode(string(&n["inputs"][i]));
            terms = Some(
                (0..declared.unwrap())
                    .map(|i| symbol(format!("deepbom_runtime:value:{name}:{i}")))
                    .collect(),
            );
        }
        let len = vals
            .as_ref()
            .map(|v| v.len() as i64)
            .or_else(|| terms.as_ref().map(|v| v.len() as i64))
            .or(declared);
        Control {
            present: true,
            values: vals,
            terms,
            len,
            invalid,
        }
    } else {
        let a = attr(n, k);
        if a.is_null() {
            return Control::default();
        }
        let vals = rows(&a["ints"])
            .iter()
            .map(|v| safe(v).map(|n| json!(n)))
            .collect::<Option<Vec<_>>>();
        Control {
            present: true,
            len: vals.as_ref().map(|v| v.len() as i64),
            terms: vals
                .as_ref()
                .map(|v| v.iter().map(|n| dim(safe(n).unwrap())).collect()),
            invalid: vals.is_none(),
            values: vals,
        }
    }
}
fn scalar(v: &Value) -> Option<Value> {
    if v.is_string() || safe(v).is_some() {
        Some(v.clone())
    } else {
        dv(v).map(|n| json!(n))
    }
}
fn termkey(v: &Value) -> String {
    match scalar(v) {
        Some(Value::String(s)) => match s.as_str() {
            "+inf" => "pos_inf".into(),
            "-inf" => "neg_inf".into(),
            _ => format!("i64:{s}"),
        },
        Some(x) => format!("v:{}", safe(&x).unwrap()),
        None => key(v),
    }
}
fn termknown(v: &Value) -> bool {
    scalar(v).is_some() || known(v)
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let x = input(v, 0);
    if shape(x).is_null() {
        return Ok(unresolved("slice_input_shape_unknown"));
    }
    let ds = dimensions(x);
    let a = control(v, 1, "starts", true);
    let b = control(v, 2, "ends", true);
    let c = control(v, 3, "axes", false);
    let d = control(v, 4, "steps", false);
    let bad = [&a, &b, &c, &d].iter().any(|c| c.invalid);
    if bad || !a.present || !b.present {
        return Ok(invalid(
            "slice_control_tensor_contract_invalid",
            json!({"starts_present":a.present,"ends_present":b.present,"invalid_control_present":bad}),
        ));
    }
    let counts = [&a, &b, &c, &d]
        .iter()
        .filter(|c| c.present)
        .filter_map(|c| c.len)
        .collect::<Vec<_>>();
    if counts.iter().any(|n| *n != counts[0]) {
        return Ok(invalid(
            "slice_control_cardinality_mismatch",
            json!({"cardinalities":counts}),
        ));
    }
    let count = counts.first().copied();
    if count.is_some_and(|n| n > ds.len() as i64) {
        return Ok(invalid(
            "slice_control_cardinality_exceeds_rank",
            json!({"control_count":count,"input_rank":ds.len()}),
        ));
    }
    let starts = a.values.as_ref().or(a.terms.as_ref());
    let ends = b.values.as_ref().or(b.terms.as_ref());
    let axisvals = if c.present {
        c.values
            .as_ref()
            .and_then(|v| v.iter().map(safe).collect::<Option<Vec<_>>>())
    } else {
        count.map(|n| (0..n as i64).collect())
    };
    let stepvals = if d.present {
        d.values
            .as_ref()
            .and_then(|v| v.iter().map(safe).collect::<Option<Vec<_>>>())
    } else {
        count.map(|n| vec![1; n as usize])
    };
    let normalized = axisvals.as_ref().and_then(|a| axes(a, ds.len()));
    if axisvals.is_some() && normalized.is_none() {
        return Ok(invalid(
            "slice_axes_invalid",
            json!({"axes":axisvals,"input_rank":ds.len()}),
        ));
    }
    if stepvals.as_ref().is_some_and(|v| v.contains(&0)) {
        return Ok(invalid(
            "slice_step_zero_invalid",
            json!({"steps":stepvals}),
        ));
    }
    let mut out = vec![];
    if starts.is_none_or(|v| !v.iter().all(termknown))
        || ends.is_none_or(|v| !v.iter().all(termknown))
        || stepvals.is_none()
        || normalized.is_none()
    {
        let s = ds
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if normalized.as_ref().is_none_or(|axes| axes.contains(&i)) {
                    unknown()
                } else {
                    d.clone()
                }
            })
            .collect::<Vec<_>>();
        if !set(v, &mut out, 0, &json!(s), kd(x), Value::Null) {
            return Ok(unresolved("slice_partial_output_unresolved"));
        }
        let mut r = result(out);
        r["reason"] = json!("slice_dynamic_controls_preserve_rank_only");
        return Ok(r);
    }
    let (starts, ends, steps, axes) = (
        starts.unwrap(),
        ends.unwrap(),
        stepvals.unwrap(),
        normalized.unwrap(),
    );
    if starts.len() != ends.len() || starts.len() != steps.len() || axes.len() != starts.len() {
        return Ok(invalid(
            "slice_control_cardinality_mismatch",
            json!({"starts":starts.len(),"ends":ends.len(),"steps":steps.len(),"axes":axes.len()}),
        ));
    }
    let mut s = ds.clone();
    let mut ranges = vec![];
    for i in 0..starts.len() {
        let axis = axes[i];
        if !known(&ds[axis]) {
            return Ok(unresolved("slice_range_invalid_or_unsafe"));
        }
        let (start, end) = (scalar(&starts[i]), scalar(&ends[i]));
        if let (Some(size), Some(start), Some(end)) = (dv(&ds[axis]), start.clone(), end.clone()) {
            let r = kernel(
                "slice",
                json!({"size":size,"start":start,"end":end,"step":steps[i]}),
            )?;
            if r.is_null() {
                return Ok(unresolved("slice_range_invalid_or_unsafe"));
            }
            s[axis] = dim(r["length"].as_i64().unwrap());
            ranges.push(r);
        } else {
            if start == Some(json!(0)) && end == Some(json!("+inf")) && steps[i] == 1 {
                s[axis] = ds[axis].clone();
            } else {
                s[axis] = symbol(format!(
                    "deepbom_expr:slice_len({},{},{},{})",
                    key(&ds[axis]),
                    termkey(&starts[i]),
                    termkey(&ends[i]),
                    steps[i]
                ));
            }
            ranges.push(Value::Null);
        }
    }
    let mut values = Value::Null;
    if ds.len() == 1 && axes == vec![0] && !ranges[0].is_null() {
        if let Some(source) = exact_dims(x) {
            let r = &ranges[0];
            let (a, b, step) = (
                r["start"].as_i64().unwrap(),
                r["end"].as_i64().unwrap(),
                steps[0],
            );
            let mut selected = vec![];
            let mut i = a as i128;
            let b = b as i128;
            let mut valid = true;
            while if step > 0 { i < b } else { i > b } {
                if selected.len() >= 65536 {
                    valid = false;
                    break;
                }
                if let Some(d) = usize::try_from(i).ok().and_then(|i| source.get(i)) {
                    selected.push(d.clone());
                } else {
                    valid = false;
                    break;
                }
                i += step as i128;
            }
            if valid {
                values = json!({"dimensions":selected,"source":"Slice"});
            }
        }
    }
    Ok(if set(v, &mut out, 0, &json!(s), kd(x), values) {
        result(out)
    } else {
        unresolved("slice_output_unresolved")
    })
}
