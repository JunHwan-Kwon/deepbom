//! Region boundary contracts for If, Loop and Scan orchestration.
use crate::portable_onnx_conditional::{conditions, unconditional, variants};
use crate::portable_onnx_container::{descriptor, ty};
use crate::portable_onnx_dynamic::axis;
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn known_type(t: &Value) -> bool {
    crate::portable_onnx_type::run(&json!({"operation":"known","value":t})).ok()
        == Some(json!(true))
}
fn dense(t: &Value) -> bool {
    !t.is_null()
        && ["", "tensor", "unresolved", "undefined"].contains(&string(if truth(&t["valueKind"]) {
            &t["valueKind"]
        } else {
            &t["value_kind"]
        }))
}
fn dtype(t: &Value) -> &str {
    if dense(t) {
        super::portable_onnx_semantics::dtype(t)
    } else {
        ""
    }
}
fn support(v: &Value, t: &Value) -> Value {
    let t = ty(v, t);
    let k = string(&t["kind"]);
    let opset = safe(&v["imported_opset"]).unwrap_or(0);
    let (status, reason) = if t.is_null() {
        ("partial", "loop_state_type_unresolved".into())
    } else {
        match k {
            "tensor" => {
                if known_type(&t) {
                    ("pass", "".into())
                } else {
                    (
                        "partial",
                        "loop_state_tensor_element_type_unresolved".into(),
                    )
                }
            }
            "sequence" => {
                if opset < 13 {
                    ("fail", "loop_sequence_state_requires_opset_13".into())
                } else if t["elementType"]["kind"] != "tensor" {
                    ("fail", "loop_sequence_state_element_not_tensor".into())
                } else if known_type(&t) {
                    ("pass", "".into())
                } else {
                    (
                        "partial",
                        "loop_sequence_state_element_type_unresolved".into(),
                    )
                }
            }
            "optional" => {
                if opset < 16 {
                    ("fail", "loop_optional_state_requires_opset_16".into())
                } else if t["elementType"]["kind"] != "tensor"
                    && !(t["elementType"]["kind"] == "sequence"
                        && t["elementType"]["elementType"]["kind"] == "tensor")
                {
                    (
                        "fail",
                        "loop_optional_state_element_not_tensor_or_tensor_sequence".into(),
                    )
                } else if known_type(&t) {
                    ("pass", "".into())
                } else {
                    (
                        "partial",
                        "loop_optional_state_element_type_unresolved".into(),
                    )
                }
            }
            _ => (
                "fail",
                format!(
                    "loop_state_value_kind_not_supported:{}",
                    if k.is_empty() { "undefined" } else { k }
                ),
            ),
        }
    };
    json!({"status":status,"reason":reason})
}
fn clear_type(v: &Value, t: &Value, depth: usize) -> Value {
    if t.is_null() || depth > 128 {
        return Value::Null;
    }
    let mut t = t.clone();
    match string(&t["kind"]) {
        "tensor" => {
            t = tensor_type(
                t["dtype"]
                    .as_str()
                    .or(t["elementTypeName"].as_str())
                    .unwrap_or("UNKNOWN"),
                &[],
                &v["tensor_type_ids"],
            );
            t["shapeDeclared"] = json!(false);
            t["shapeFieldCount"] = json!(0);
        }
        "sequence" | "optional" => {
            t["elementType"] = clear_type(v, &t["elementType"], depth + 1);
        }
        _ => (),
    }
    t
}
fn loop_clear(v: &Value, t: &Value) -> Value {
    let t = clear_type(v, &ty(v, t), 0);
    if t.is_null() {
        return Value::Null;
    }
    let state = match string(&t["kind"]) {
        "sequence" => {
            json!({"sequenceLengthStatus":"not_assessed_loop_variant","sequenceLength":null,"sequenceElementInventoryStatus":"not_assessed_loop_variant","sequenceElementTypes":[]})
        }
        "optional" => {
            json!({"optionalPresenceStatus":"not_assessed_loop_variant","optionalPresence":null})
        }
        _ => json!({}),
    };
    descriptor(&t, state)
}
fn merge(v: &Value, a: &Value, b: &Value) -> R<Value> {
    crate::portable_onnx_type::run(
        &json!({"operation":"union","left":a,"right":b,"tensor_type_ids":v["tensor_type_ids"]}),
    )
}
fn fail() -> Value {
    json!({"status":"fail","patch":null})
}
fn value_copy(v: &Value, t: &Value) -> Value {
    let mut p = json!({"dtype":t["dtype"].as_str().unwrap_or("UNKNOWN"),"shape":if t["shape"].is_array(){t["shape"].clone()}else{json!([])},"shapeDeclared":t["shapeDeclared"]==true||t["shape_declared"]==true});
    if truth(&t["valueKind"]) || truth(&t["value_kind"]) {
        p["valueKind"] = if truth(&t["valueKind"]) {
            t["valueKind"].clone()
        } else {
            t["value_kind"].clone()
        };
    }
    if !t["typeProto"].is_null() || !t["type_proto"].is_null() {
        p["typeProto"] = ty(v, t);
    }
    for k in [
        "sequenceLengthStatus",
        "sequenceLength",
        "sequenceElementInventoryStatus",
        "optionalPresenceStatus",
        "optionalPresence",
        "sequenceElementTypes",
        "runtimeDimensionBounds",
    ] {
        if let Some(x) = t.get(k) {
            p[k] = x.clone();
        }
    }
    copy_static(&mut p, t, false);
    p
}
fn copy_static(p: &mut Value, t: &Value, union: bool) {
    for (prefix, status) in [
        ("staticValues", "complete"),
        (
            "staticDimensionValues",
            "assessed_exact_symbolic_shape_data",
        ),
    ] {
        if t[format!("{prefix}Complete")] == true {
            p[format!("{prefix}Status")] = json!(status);
            p[format!("{prefix}Complete")] = json!(true);
            p[prefix] = t[prefix].clone();
            p[format!("{prefix}Source")] = json!(t[format!("{prefix}Source")]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or(if union { "branch_union" } else { "propagated" }));
        }
    }
}
fn branch_variants(v: &Value, t: &Value, condition: Value) -> Vec<Value> {
    let vs = variants(t, true);
    let source = if vs.is_empty() { vec![t.clone()] } else { vs };
    source
        .into_iter()
        .filter(unconditional)
        .filter_map(|t| {
            let cs = conditions(&t["conditions"], &json!([condition]))?;
            let mut p = value_copy(v, &t);
            p["conditions"] = cs;
            Some(p)
        })
        .collect()
}
// Recognize only an exact symbolic dimension == 1 predicate, not arbitrary
// boolean provenance. The key is shared with Squeeze's conditional contract.
fn region_condition(v: &Value) -> Value {
    let n = &v["node"];
    if n["opType"] != "Equal"
        || !["", "ai.onnx"].contains(&string(&n["domain"]))
        || rows(&n["inputs"]).len() != 2
    {
        return Value::Null;
    }
    let a = &v["tensors"][string(&n["inputs"][0])];
    let b = &v["tensors"][string(&n["inputs"][1])];
    for (d, one) in [(a, b), (b, a)] {
        if !unconditional(d)
            || !unconditional(one)
            || dtype(d) != "INT64"
            || dtype(one) != "INT64"
            || d["staticDimensionValuesComplete"] != true
            || rows(&d["staticDimensionValues"]).len() != 1
            || !rows(&d["shape"]).iter().all(|n| safe(n) == Some(1))
            || scalar(&json!({"kind":"int64","singleton":true}), one) != json!(1)
        {
            continue;
        }
        let dim = &d["staticDimensionValues"][0];
        if dim["kind"] == "symbolic" && !string(&dim["parameter"]).is_empty() {
            return json!({"key":format!("onnx:dimension:{}:equals:1",crate::portable_onnx_dynamic::encode(string(&dim["parameter"]))),"thenValue":"true","elseValue":"false"});
        }
    }
    Value::Null
}
fn branch_failures(t: &Value, condition: Value) -> Vec<Value> {
    let c = &t["conditionalShapeContract"];
    let mut failures = rows(&c["variant_failures"]).to_vec();
    // A partial contract without enumerated failures cannot become complete.
    if c["status"] == "assessed_partial" && failures.is_empty() {
        failures.push(json!({"status":"not_assessed","reason":"branch_partial_contract_without_failure_inventory","conditions":[]}));
    }
    failures
        .into_iter()
        .filter_map(|mut f| {
            f["conditions"] = conditions(&f["conditions"], &json!([condition]))?;
            Some(f)
        })
        .collect()
}
fn union_values(v: &Value) -> R<Value> {
    let (a, b) = (&v["left"], &v["right"]);
    let (at, bt) = (ty(v, a), ty(v, b));
    if at.is_null() || bt.is_null() || at["kind"] != bt["kind"] {
        return Ok(fail());
    }
    let merged = merge(v, &at, &bt)?;
    if merged["status"] != "pass" {
        return Ok(fail());
    }
    let mut p = descriptor(&merged["type"], Value::Null);
    let k = string(&at["kind"]);
    if k == "sequence" {
        let len = if a["sequenceLengthStatus"] == "assessed_exact"
            && b["sequenceLengthStatus"] == "assessed_exact"
            && a["sequenceLength"] == b["sequenceLength"]
        {
            safe(&a["sequenceLength"]).filter(|n| *n >= 0)
        } else {
            None
        };
        p["sequenceLengthStatus"] = json!(if len.is_some() {
            "assessed_exact"
        } else {
            "not_assessed_branch_dependent"
        });
        p["sequenceLength"] = json!(len);
        let mut inventory = None;
        if a["sequenceElementInventoryStatus"] == "assessed_exact"
            && b["sequenceElementInventoryStatus"] == "assessed_exact"
            && a["sequenceElementTypes"].is_array()
            && b["sequenceElementTypes"].is_array()
            && rows(&a["sequenceElementTypes"]).len() == rows(&b["sequenceElementTypes"]).len()
        {
            let mut items = vec![];
            let mut valid = true;
            for (x, y) in rows(&a["sequenceElementTypes"])
                .iter()
                .zip(rows(&b["sequenceElementTypes"]))
            {
                let m = merge(v, x, y)?;
                if m["status"] != "pass" {
                    valid = false;
                    break;
                }
                items.push(m["type"].clone());
            }
            if valid {
                inventory = Some(items);
            }
        }
        p["sequenceElementInventoryStatus"] = json!(if inventory.is_some() {
            "assessed_exact"
        } else {
            "not_assessed_branch_dependent"
        });
        p["sequenceElementTypes"] = json!(inventory.unwrap_or_default());
    } else if k == "optional" {
        let value = if a["optionalPresenceStatus"] == "assessed_exact"
            && b["optionalPresenceStatus"] == "assessed_exact"
            && a["optionalPresence"] == b["optionalPresence"]
        {
            a["optionalPresence"].as_bool()
        } else {
            None
        };
        p["optionalPresenceStatus"] = json!(if value.is_some() {
            "assessed_exact"
        } else {
            "not_assessed_branch_dependent"
        });
        p["optionalPresence"] = json!(value);
    } else if k == "tensor" {
        if !dense(a) || !dense(b) {
            return Ok(fail());
        }
        for prefix in ["staticValues", "staticDimensionValues"] {
            if a[format!("{prefix}Complete")] == true
                && b[format!("{prefix}Complete")] == true
                && a[prefix] == b[prefix]
            {
                let mut source = a.clone();
                source[if prefix == "staticValues" {
                    "staticDimensionValuesComplete"
                } else {
                    "staticValuesComplete"
                }] = json!(false);
                copy_static(&mut p, &source, true);
            }
        }
        let complete = |t: &Value| unconditional(t) || !variants(t, true).is_empty();
        if !v["condition"].is_null() && complete(a) && complete(b) {
            let c = &v["condition"];
            let failures = branch_failures(a, json!({"key":c["key"],"value":c["thenValue"]}))
                .into_iter()
                .chain(branch_failures(
                    b,
                    json!({"key":c["key"],"value":c["elseValue"]}),
                ))
                .collect::<Vec<_>>();
            let all = branch_variants(v, a, json!({"key":c["key"],"value":c["thenValue"]}))
                .into_iter()
                .chain(branch_variants(
                    v,
                    b,
                    json!({"key":c["key"],"value":c["elseValue"]}),
                ));
            let mut seen = BTreeSet::new();
            let vs = all
                .filter(|p| {
                    seen.insert(
                        json!({"conditions":p["conditions"],"dtype":p["dtype"],"type":ty(v,p)})
                            .to_string(),
                    )
                })
                .collect::<Vec<_>>();
            let keys = vs
                .iter()
                .flat_map(|p| {
                    rows(&p["conditions"])
                        .iter()
                        .map(|c| string(&c["key"]).to_owned())
                })
                .collect::<BTreeSet<_>>();
            p["conditionalShapeContract"] = json!({"schema":"deepbom.onnx_conditional_shape_contract.v1","status":if failures.is_empty() && !vs.is_empty(){"assessed_complete"}else{"assessed_partial"},"variant_count":vs.len(),"condition_keys":keys});
            if !failures.is_empty() {
                p["conditionalShapeContract"]["invalid_variant_count"] =
                    json!(failures.iter().filter(|f| f["status"] == "invalid").count());
                p["conditionalShapeContract"]["variant_failures"] = json!(failures);
            }
            p["conditionalShapeVariants"] = json!(vs);
        }
    }
    Ok(json!({"status":"pass","patch":p}))
}
fn merge_tensor(a: &Value, b: &Value) -> Value {
    if a.is_null() {
        return json!({"status":if b.is_null(){"unresolved"}else{"pass"},"patch":b});
    }
    if b.is_null() {
        return json!({"status":"pass","patch":a});
    }
    if !dense(a)
        || !dense(b)
        || !dtype(a).is_empty() && !dtype(b).is_empty() && dtype(a) != dtype(b)
    {
        return fail();
    }
    let (ad, bd) = (a["shapeDeclared"] == true, b["shapeDeclared"] == true);
    if ad && bd && rows(&a["shape"]).len() != rows(&b["shape"]).len() {
        return fail();
    }
    let mut p = json!({"dtype":if !dtype(a).is_empty(){dtype(a)}else if !dtype(b).is_empty(){dtype(b)}else{"UNKNOWN"},"shape":[],"shapeDeclared":ad||bd});
    if ad || bd {
        let x = if ad {
            rows(&a["shape"])
        } else {
            rows(&b["shape"])
        };
        let y = if bd {
            rows(&b["shape"])
        } else {
            rows(&a["shape"])
        };
        let mut s = vec![];
        for (i, d) in x.iter().enumerate() {
            let e = y.get(i).unwrap_or(&Value::Null);
            let (u, w) = (safe(d).filter(|n| *n >= 0), safe(e).filter(|n| *n >= 0));
            if matches!((u,w),(Some(u),Some(w))if u!=w) {
                return fail();
            }
            s.push(if u.is_some() { d.clone() } else { e.clone() });
        }
        p["shape"] = json!(s);
    }
    json!({"status":"pass","patch":p})
}
fn scalar(v: &Value, t: &Value) -> Value {
    let expected = if v["kind"] == "int64" {
        "INT64"
    } else {
        "BOOL"
    };
    if dtype(t) != expected || t["shapeDeclared"] != true {
        return Value::Null;
    }
    let shape = rows(&t["shape"]);
    if v["singleton"] == true {
        if !shape.iter().all(|d| safe(d) == Some(1)) {
            return Value::Null;
        }
    } else if !shape.is_empty() {
        return Value::Null;
    }
    if expected == "INT64" && t["initializerIntegerValuesExactComplete"] == true {
        let values = rows(&t["initializerIntegerValuesExactDecimals"]);
        return if values.len() == 1 {
            values[0]
                .as_str()
                .and_then(|n| n.parse::<i64>().ok())
                .filter(|n| n.unsigned_abs() <= SAFE as u64)
                .map_or(Value::Null, |n| json!(n))
        } else {
            Value::Null
        };
    }
    let values = rows(&t["staticValues"]);
    if t["staticValuesComplete"] != true || values.len() != 1 {
        return Value::Null;
    }
    if expected == "INT64" {
        return safe(&values[0]).map_or(Value::Null, |n| json!(n));
    }
    match &values[0] {
        Value::Bool(b) => json!(b),
        n if safe(n) == Some(0) => json!(false),
        n if safe(n) == Some(1) => json!(true),
        _ => Value::Null,
    }
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let t = &v["value"];
    Ok(match string(&v["operation"]) {
        "region_scalar" => scalar(v, t),
        "region_condition" => region_condition(v),
        "loop_support" => support(v, t),
        "loop_clear" => loop_clear(v, t),
        "loop_merge" => {
            let (a, b) = (&v["left"], &v["right"]);
            let (x, y) = (support(v, a), support(v, b));
            let (at, bt) = (clear_type(v, &ty(v, a), 0), clear_type(v, &ty(v, b), 0));
            let merged = merge(v, &at, &bt)?;
            if x["status"] == "fail"
                || y["status"] == "fail"
                || at.is_null()
                || bt.is_null()
                || merged["status"] != "pass"
            {
                fail()
            } else {
                json!({"status":if x["status"]=="partial"||y["status"]=="partial"{"partial"}else{"pass"},"patch":loop_clear(v,&descriptor(&merged["type"],Value::Null))})
            }
        }
        "union_branch" => union_values(v)?,
        "region_merge_tensor" => merge_tensor(&v["left"], &v["right"]),
        "region_axis" => {
            if !dense(t) {
                json!({"status":"fail","shape":null,"dimension":null})
            } else if t["shapeDeclared"] != true {
                json!({"status":"partial","shape":[],"dimension":-1})
            } else {
                let mut s = rows(&t["shape"]).to_vec();
                let insert = v["insert"] == true;
                let r = s.len() + usize::from(insert);
                if let Some(a) = safe(&v["axis"]).and_then(|a| axis(a, r)) {
                    let d = if insert {
                        s.insert(a, v["dimension"].clone());
                        Value::Null
                    } else {
                        s.remove(a)
                    };
                    json!({"status":"pass","shape":s,"dimension":d})
                } else {
                    json!({"status":"fail","shape":null,"dimension":null})
                }
            }
        }
        _ => return Err("Unknown ONNX region contract".into()),
    })
}
