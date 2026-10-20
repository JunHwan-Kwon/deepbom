//! Portable count/shape aggregation and structural comparison. Rendering and
//! framework collection remain host concerns; all integer arithmetic is exact.
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen::prelude::*;

const SAFE: u64 = 9_007_199_254_740_991;
pub(crate) fn decimal(v: &Value) -> Option<BigUint> {
    match v {
        Value::String(s)
            if !s.is_empty() && s.len() <= 32768 && s.bytes().all(|c| c.is_ascii_digit()) =>
        {
            s.parse().ok()
        }
        Value::Number(n) => n.as_u64().filter(|n| *n <= SAFE).map(BigUint::from),
        Value::Object(m) => decimal(m.get("decimal")?),
        _ => None,
    }
}
pub(crate) fn exact(n: &BigUint) -> Value {
    json!({"decimal":n.to_string(), "number":n.to_u64().filter(|v|*v<=SAFE)})
}
pub(crate) fn count_product(values: &[Value]) -> Result<Option<BigUint>, String> {
    let mut product = BigUint::from(1u32);
    for value in values {
        if let Some(n) = decimal(value) {
            product *= n;
        } else {
            return Ok(None);
        }
        if product.bits() > 131072 {
            return Err("exact product exceeds numerical ABI budget".into());
        }
    }
    Ok(Some(product))
}
fn sum(values: &[Value], assessed_empty: bool) -> Value {
    let mut total = BigUint::zero();
    let mut assessed = 0;
    for value in values {
        if let Some(n) = decimal(value) {
            total += n;
            assessed += 1;
        }
    }
    if assessed == 0 && (!values.is_empty() || !assessed_empty) {
        return json!({"decimal":null,"number":null,"status":"not_assessable","assessed_count":0,"total_count":values.len()});
    }
    let mut result = exact(&total);
    result["status"] = json!(if assessed == values.len() {
        "complete"
    } else {
        "partial"
    });
    result["assessed_count"] = json!(assessed);
    result["total_count"] = json!(values.len());
    result
}
fn shape(value: &Value) -> Option<Vec<u64>> {
    value
        .as_array()?
        .iter()
        .map(|v| v.as_u64().filter(|n| *n <= SAFE))
        .collect()
}
pub(crate) fn broadcast(a: &[u64], b: &[u64]) -> Option<Vec<u64>> {
    let length = a.len().max(b.len());
    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        let x = if i + a.len() >= length {
            a[i + a.len() - length]
        } else {
            1
        };
        let y = if i + b.len() >= length {
            b[i + b.len() - length]
        } else {
            1
        };
        if x != y && x != 1 && y != 1 {
            return None;
        }
        result.push(if x == 1 { y } else { x });
    }
    Some(result)
}
fn utf16(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}
fn node_index(rows: &Value) -> Result<BTreeMap<&str, &Value>, &'static str> {
    let mut map = BTreeMap::new();
    for row in rows.as_array().ok_or("invalid native nodes")? {
        let id = row["id"].as_str().ok_or("invalid native node identity")?;
        if map.insert(id, row).is_some() {
            return Err("duplicate native node");
        }
    }
    Ok(map)
}
fn compare_nodes(a: &Value, b: &Value) -> Result<Value, &'static str> {
    let a = node_index(a)?;
    let b = node_index(b)?;
    let mut ids: Vec<_> = a.keys().chain(b.keys()).copied().collect();
    ids.sort_unstable_by(|a, b| utf16(a, b));
    ids.dedup();
    Ok(
        json!(ids.into_iter().map(|id|json!({"id":id,"status":match (a.get(id),b.get(id)) {
        (None,_)=>"added",(_,None)=>"removed",(Some(x),Some(y)) if x==y=>"unchanged",_=>"changed",
    }})).collect::<Vec<_>>()),
    )
}
fn changes(a: Option<&Value>, b: Option<&Value>, path: &str, out: &mut Vec<Value>) {
    if a.is_some() && a == b {
        return;
    }
    let ao = a.and_then(Value::as_object);
    let bo = b.and_then(Value::as_object);
    let descend = (ao.is_some() && bo.is_some())
        || (a.is_none() && bo.is_some())
        || (ao.is_some() && b.is_none());
    let mut keys: Vec<_> = ao
        .into_iter()
        .flat_map(|m| m.keys())
        .chain(bo.into_iter().flat_map(|m| m.keys()))
        .collect();
    keys.sort_unstable_by(|a, b| utf16(a, b));
    keys.dedup();
    if descend && !keys.is_empty() {
        for key in keys {
            let pointer = key.replace('~', "~0").replace('/', "~1");
            changes(
                ao.and_then(|m| m.get(key)),
                bo.and_then(|m| m.get(key)),
                &format!("{path}/{pointer}"),
                out,
            );
        }
    } else {
        out.push(json!({"path":path,"before_present":a.is_some(),"after_present":b.is_some(),"before":a,"after":b}));
    }
}
fn run(operation: &str, input: &Value) -> Result<Value, String> {
    match operation {
        "features" => Ok(json!([
            "shape",
            "dimension_expression",
            "redesign_validate",
            "evidence_contract",
            "evidence_identity",
            "evidence_workflow",
            "decimal",
            "native_tensor",
            "onnx_type",
            "ir_semantics",
            "coreml_deployment",
            "onnx_semantics",
            "semantic_validation",
            "conversion_receipt",
            "provenance"
        ])),
        "dimension_expression" => Ok(crate::portable_integer::evaluate_dimension(input)),
        "ir_semantics" => crate::portable_ir::run(input),
        "onnx_semantics" => crate::portable_onnx_semantics::run(input),
        "semantic_validation" => crate::portable_validation::run(input),
        "conversion_receipt" => crate::portable_conversion::run(input),
        "provenance" => crate::portable_provenance::run(input),
        "coreml_deployment" => crate::portable_coreml::run(input),
        "native_tensor" => crate::portable_native::run(input),
        "onnx_type" => crate::portable_onnx_type::run(input),
        "evidence_contract" => crate::portable_evidence::run(input),
        "evidence_identity" => crate::portable_identity::run(input),
        "evidence_workflow" => crate::portable_workflow::run(input),
        "decimal" => crate::portable_decimal::run(input),
        "redesign_validate" => Ok(crate::portable_redesign::run(input)),
        "shape" => crate::portable_shape::run(input),
        "integer_expression" => Ok(crate::portable_integer::evaluate(input)),
        "einsum_equation" => Ok(crate::portable_einsum::parse(input)),
        "contraction_cost" => crate::portable_cost::cost(input),
        "graph_partition" => crate::portable_graph::partition(input),
        "graph_boundary" => crate::portable_graph::boundary(input),
        "graph_edges" => crate::portable_graph::edges(input),
        "graph_topology" => crate::portable_graph::topology(input),
        "sum" => Ok(sum(
            input["values"].as_array().ok_or("invalid count list")?,
            input["assessed_empty"].as_bool().unwrap_or(true),
        )),
        "product" => Ok(
            count_product(input["values"].as_array().ok_or("invalid count list")?)?
                .map(|n| exact(&n))
                .unwrap_or(Value::Null),
        ),
        "pack" => {
            let bits = input["bits"].as_u64().filter(|v| *v > 0 && *v <= SAFE);
            Ok(match (decimal(&input["elements"]), bits) {
                (Some(n), Some(bits)) => {
                    exact(&((n * BigUint::from(bits) + BigUint::from(7u32)) / BigUint::from(8u32)))
                }
                _ => Value::Null,
            })
        }
        "broadcast" => Ok(match (shape(&input["left"]), shape(&input["right"])) {
            (Some(a), Some(b)) => json!(broadcast(&a, &b)),
            _ => Value::Null,
        }),
        "nodes" => compare_nodes(&input["before"], &input["after"]).map_err(str::to_owned),
        "changes" => {
            let mut rows = vec![];
            changes(
                if input["before_present"].as_bool().unwrap_or(true) {
                    Some(&input["before"])
                } else {
                    None
                },
                if input["after_present"].as_bool().unwrap_or(true) {
                    Some(&input["after"])
                } else {
                    None
                },
                "",
                &mut rows,
            );
            Ok(json!(rows))
        }
        "deltas" => {
            let a = input["before"]
                .as_object()
                .ok_or("invalid baseline totals")?;
            let b = input["after"]
                .as_object()
                .ok_or("invalid candidate totals")?;
            if a.keys().collect::<BTreeSet<_>>() != b.keys().collect::<BTreeSet<_>>() {
                return Err("totals have different fields".into());
            }
            let mut result = serde_json::Map::new();
            for (key, value) in a {
                let x = decimal(value).ok_or("invalid baseline count")?;
                let y = decimal(&b[key]).ok_or("invalid candidate count")?;
                result.insert(
                    key.clone(),
                    json!((BigInt::from(y) - BigInt::from(x)).to_string()),
                );
            }
            Ok(Value::Object(result))
        }
        _ => Err("unknown portable contract operation".into()),
    }
}
#[wasm_bindgen]
pub fn numerical_contract(operation: &str, input: &str) -> Result<String, JsValue> {
    if input.len() > 128 * 1024 * 1024 {
        return Err(JsValue::from_str("portable contract input exceeds budget"));
    }
    let value: Value =
        serde_json::from_str(input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = run(operation, &value).map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_unknown_empty_and_overflow() {
        assert_eq!(
            sum(&[json!("9007199254740991"), json!(2)], true)["decimal"],
            "9007199254740993"
        );
        assert_eq!(sum(&[json!(null), json!(0)], true)["status"], "partial");
        assert_eq!(sum(&[json!(null)], true)["number"], Value::Null);
        assert_eq!(sum(&[], true)["decimal"], "0");
        assert_eq!(sum(&[], false)["decimal"], Value::Null);
    }
    #[test]
    fn empty_axis_and_packing() {
        assert_eq!(broadcast(&[0, 3], &[1, 3]), Some(vec![0, 3]));
        assert_eq!(broadcast(&[2, 3], &[4, 3]), None);
        assert_eq!(
            run("pack", &json!({"elements":"9007199254740993","bits":4})).unwrap()["decimal"],
            "4503599627370497"
        );
    }
    #[test]
    fn structural_presence_unicode_and_arrays() {
        let rows=run("changes",&json!({"before":{"a/b":null,"array":[1],"😀":1,"\u{e000}":1},"after":{"array":[1,2],"😀":2,"\u{e000}":2}})).unwrap();
        assert_eq!(rows[0]["path"], "/a~1b");
        assert_eq!(rows[0]["after_present"], false);
        assert_eq!(rows[1]["path"], "/array");
        assert_eq!(rows[2]["path"], "/😀");
        assert!(compare_nodes(&json!([{"id":"a"},{"id":"a"}]), &json!([])).is_err());
    }
}
