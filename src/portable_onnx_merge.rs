//! Declared versus inferred value contract reconciliation; never silently overwrite conflicts.
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
fn kind(v: &Value, implicit: bool) -> &str {
    let k = v["valueKind"]
        .as_str()
        .or(v["value_kind"].as_str())
        .unwrap_or("");
    if !k.is_empty() && !["undefined", "unresolved"].contains(&k) {
        return k;
    }
    if k.is_empty()
        && (implicit
            || v["shapeDeclared"] == true
            || v["shape_declared"] == true
            || !dtype(v).is_empty()
            || v["typeProto"]["kind"] == "tensor"
            || v["type_proto"]["kind"] == "tensor")
    {
        return "tensor";
    }
    if implicit {
        "tensor"
    } else {
        "unresolved"
    }
}
fn dt(v: &Value) -> &str {
    if v["contractStatus"] == "invalid"
        || !string(&v["valueKind"]).is_empty() && v["valueKind"] != "tensor"
    {
        ""
    } else {
        dtype(v)
    }
}
fn concrete(v: &Value) -> bool {
    v["contractStatus"] != "invalid"
        && v["conditionalShapeContract"]["status"] != "assessed_partial"
        && rank(v).is_some()
        && rows(&v["shape"])
            .iter()
            .all(|d| dimension_integer(d).is_some_and(|n| n >= 0))
}
fn dimension_integer(v: &Value) -> Option<i64> {
    safe(v).or_else(|| {
        v.as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|n| n.unsigned_abs() <= SAFE as u64)
    })
}
fn merge_type(v: &Value, left: &Value, right: &Value) -> R<Value> {
    if left.is_null() {
        return Ok(
            json!({"status":if right.is_null(){"unresolved"}else{"pass"},"type":right,"reason":if right.is_null(){"type_proto_missing"}else{""}}),
        );
    }
    if right.is_null() {
        return Ok(json!({"status":"pass","type":left,"reason":""}));
    }
    crate::portable_onnx_type::run(
        &json!({"operation":"unify","left":left,"right":right,"tensor_type_ids":v["tensor_type_ids"]}),
    )
}
fn conflict(v: &Value, field: &str, a: Value, b: Value) -> Value {
    json!({"conflict":{"node_index":v["node_index"],"op_name":v["op_name"],"tensor_name":v["name"],"field":field,"declared":a,"inferred":b},"descriptor":null})
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let e = &v["existing"];
    let p = &v["patch"];
    if v["operation"] == "concrete_tensor" {
        return Ok(json!(concrete(e) && !dt(e).is_empty()));
    }
    let (ek, pk) = (kind(e, false), kind(p, true));
    let dense = pk == "tensor" && ["tensor", "unresolved"].contains(&ek);
    if !dense {
        if ek != "unresolved" && pk != "unresolved" && ek != pk {
            return Ok(conflict(v, "value_kind", json!(ek), json!(pk)));
        }
        let a = if ek == "unresolved" {
            Value::Null
        } else {
            crate::portable_onnx_container::ty(v, e)
        };
        let b = if pk == "unresolved" {
            Value::Null
        } else {
            crate::portable_onnx_container::ty(v, p)
        };
        let m = merge_type(v, &a, &b)?;
        if m["status"] == "fail" {
            return Ok(conflict(
                v,
                "type_proto",
                m["reason"].clone(),
                json!(b["kind"].as_str().unwrap_or("unresolved")),
            ));
        }
        if m["type"].is_null() {
            return Ok(conflict(v, "type_proto", json!(ek), json!(pk)));
        }
        return Ok(
            json!({"conflict":null,"descriptor":crate::portable_onnx_container::descriptor(&m["type"],json!({}))}),
        );
    }
    let (ed, pd) = (dt(e), dt(p));
    if !ed.is_empty() && !pd.is_empty() && ed != pd {
        return Ok(conflict(v, "dtype", json!(ed), json!(pd)));
    }
    let mut shape = if p["shapeDeclared"] == true {
        p["shape"].clone()
    } else {
        e["shape"].clone()
    };
    if e["shapeDeclared"] == true && p["shapeDeclared"] == true {
        let (es, ps) = (rows(&e["shape"]), rows(&p["shape"]));
        if es.len() != ps.len() {
            return Ok(conflict(v, "rank", json!(es.len()), json!(ps.len())));
        }
        for (i, (a, b)) in es.iter().zip(ps).enumerate() {
            if let (Some(x), Some(y)) = (
                dimension_integer(a).filter(|n| *n >= 0),
                dimension_integer(b).filter(|n| *n >= 0),
            ) {
                if x != y {
                    return Ok(conflict(v, &format!("dimension_{i}"), a.clone(), b.clone()));
                }
            }
        }
        shape = json!(es
            .iter()
            .zip(ps)
            .map(|(a, b)| if dimension_integer(a).is_some_and(|n| n >= 0) {
                a
            } else {
                b
            })
            .collect::<Vec<_>>());
    }
    let dtype = if !pd.is_empty() {
        pd
    } else {
        e["dtype"].as_str().unwrap_or("UNKNOWN")
    };
    let declared = p["shapeDeclared"] == true || e["shapeDeclared"] == true;
    let inferred = if !p["typeProto"].is_null() {
        p["typeProto"].clone()
    } else {
        crate::portable_onnx_container::ty(
            v,
            &json!({"dtype":dtype,"shape":if shape.is_array(){shape}else{e["shape"].clone()},"shapeDeclared":declared}),
        )
    };
    let m = merge_type(v, &crate::portable_onnx_container::ty(v, e), &inferred)?;
    if m["status"] == "fail" {
        return Ok(conflict(
            v,
            "type_proto",
            m["reason"].clone(),
            json!("tensor"),
        ));
    }
    let mut result = crate::portable_onnx_container::descriptor(
        if m["type"].is_null() {
            &inferred
        } else {
            &m["type"]
        },
        json!({}),
    );
    result["valueKind"] = json!("tensor");
    Ok(json!({"conflict":null,"descriptor":result}))
}
