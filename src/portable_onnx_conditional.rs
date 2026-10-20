//! Finite conditional-shape propagation. Unknown branches remain explicit failures.
use crate::portable_onnx_container::{descriptor, ty};
use crate::portable_onnx_dynamic::{invalid, known};
use crate::portable_onnx_node::{kd, shape};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
pub(crate) fn unconditional(t: &Value) -> bool {
    t["contractStatus"] != "invalid"
        && t["conditionalShapeContract"]["status"] != "assessed_partial"
        && !kd(t).is_empty()
        && !shape(t).is_null()
        && dimensions(t).iter().all(known)
}
pub(crate) fn variants(t: &Value, partial: bool) -> Vec<Value> {
    if t["contractStatus"] == "invalid"
        || !(t["conditionalShapeContract"]["status"] == "assessed_complete"
            || partial && t["conditionalShapeContract"]["status"] == "assessed_partial")
    {
        return vec![];
    }
    let vs = rows(&t["conditionalShapeVariants"]);
    if vs.iter().all(unconditional) {
        vs.to_vec()
    } else {
        vec![]
    }
}
pub(crate) fn conditions(a: &Value, b: &Value) -> Option<Value> {
    let mut values = BTreeMap::new();
    for c in rows(a).iter().chain(rows(b)) {
        let (k, v) = (string(&c["key"]), string(&c["value"]));
        if k.is_empty() || v.is_empty() || values.get(k).is_some_and(|x| *x != v) {
            return None;
        }
        values.insert(k, v);
    }
    Some(json!(values
        .into_iter()
        .map(|(k, v)| json!({"key":k,"value":v}))
        .collect::<Vec<_>>()))
}
fn unique_variants(vs: Vec<Value>) -> Vec<Value> {
    let mut seen = BTreeSet::new();
    vs.into_iter().filter(|v|{let ds=dimensions(v).iter().map(|d|json!({"kind":d["kind"],"value":dv(d),"parameter":if d["kind"]=="symbolic"{d["parameter"].clone()}else{Value::Null}})).collect::<Vec<_>>();seen.insert(json!({"conditions":v["conditions"],"dtype":kd(v),"dimensions":ds}).to_string())}).collect()
}
fn unique_failures(vs: Vec<Value>) -> Vec<Value> {
    let mut seen = BTreeSet::new();
    vs.into_iter().filter_map(|v|{let mut n=json!({"status":v["status"].as_str().filter(|s|!s.is_empty()).unwrap_or("not_assessed"),"reason":v["reason"].as_str().filter(|s|!s.is_empty()).unwrap_or("conditional_shape_variant_not_assessed"),"conditions":if v["conditions"].is_array(){v["conditions"].clone()}else{json!([])},"details":v["details"]});if truth(&v["output_name"]){n["output_name"]=v["output_name"].clone();}if seen.insert(n.to_string()){Some(n)}else{None}}).collect()
}
fn infer(v: &Value) -> R<Value> {
    Ok(crate::portable_onnx_node::run(v)?
        .as_object()
        .map(|m| Value::Object(m.clone()))
        .unwrap_or_else(|| unresolved("implemented_rule_dispatch_missing")))
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let mut base = infer(v)?;
    let inputs = rows(&v["node"]["inputs"])
        .iter()
        .filter(|n| truth(n))
        .map(|n| (string(n), variants(&v["tensors"][string(n)], true)))
        .filter(|(_, vs)| !vs.is_empty())
        .collect::<Vec<_>>();
    if inputs.is_empty() {
        return Ok(base);
    }
    let mut combos = vec![(Vec::<(String, Value)>::new(), json!([]))];
    for (name, vs) in &inputs {
        let mut expanded = vec![];
        for (replacements, cs) in &combos {
            for variant in vs {
                let Some(cs) = conditions(cs, &variant["conditions"]) else {
                    continue;
                };
                let mut next = replacements.clone();
                next.push(((*name).into(), variant.clone()));
                expanded.push((next, cs));
                if expanded.len() > 64 {
                    if !truth(&base["reason"]) {
                        base["reason"] = json!("conditional_shape_variant_limit_exceeded");
                    }
                    return Ok(base);
                }
            }
        }
        combos = expanded;
    }
    if combos.is_empty() {
        return Ok(invalid(
            "conditional_shape_conditions_have_no_compatible_assignment",
            Value::Null,
        ));
    }
    let mut failures = unique_failures(
        inputs
            .iter()
            .flat_map(|(name, _)| {
                rows(&v["tensors"][*name]["conditionalShapeContract"]["variant_failures"]).to_vec()
            })
            .collect(),
    );
    let mut outputs: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (replacements, cs) in combos {
        let mut scoped = v.clone();
        for (name, mut patch) in replacements {
            let m = patch.as_object_mut().unwrap();
            m.remove("conditions");
            m.remove("conditionalShapeVariants");
            let mut existing = scoped["tensors"][&name]
                .as_object()
                .cloned()
                .unwrap_or_default();
            existing.extend(m.clone());
            existing.insert("name".into(), json!(name));
            existing.insert(
                "contractStatus".into(),
                json!("assessed_conditional_variant"),
            );
            scoped["tensors"][&name] = Value::Object(existing);
        }
        let r = infer(&scoped)?;
        if r["status"] == "invalid" {
            failures.push(json!({"status":"invalid","reason":r["reason"].as_str().filter(|s|!s.is_empty()).unwrap_or("conditional_shape_variant_invalid"),"conditions":cs,"details":r["details"]}));
            continue;
        }
        if rows(&r["outputs"]).is_empty() && rows(&v["node"]["outputs"]).iter().any(truth) {
            let why = r["reason"]
                .as_str()
                .filter(|s| !s.is_empty())
                .or(base["reason"].as_str().filter(|s| !s.is_empty()))
                .unwrap_or("conditional_shape_variant_not_inferred");
            failures
                .push(json!({"status":"not_assessed","reason":why,"conditions":cs,"details":null}));
            continue;
        }
        let incomplete = rows(&v["node"]["outputs"])
            .iter()
            .filter(|n| truth(n))
            .find(|n| {
                let p = rows(&r["outputs"])
                    .iter()
                    .find(|p| p[0] == **n)
                    .map(|p| &p[1])
                    .unwrap_or(&Value::Null);
                !unconditional(p) && variants(p, true).is_empty()
            });
        if let Some(name) = incomplete {
            failures.push(json!({"status":"not_assessed","reason":r["reason"].as_str().filter(|s|!s.is_empty()).unwrap_or("conditional_shape_variant_output_incomplete"),"output_name":name,"conditions":cs,"details":null}));
            continue;
        }
        for p in rows(&r["outputs"]) {
            let name = string(&p[0]).to_owned();
            let nested = variants(&p[1], true);
            if !nested.is_empty() {
                for mut inner in nested {
                    if let Some(c) = conditions(&cs, &inner["conditions"]) {
                        inner["conditions"] = c;
                        inner
                            .as_object_mut()
                            .unwrap()
                            .remove("conditionalShapeVariants");
                        inner
                            .as_object_mut()
                            .unwrap()
                            .remove("conditionalShapeContract");
                        outputs.entry(name.clone()).or_default().push(inner);
                    }
                }
                for f in rows(&p[1]["conditionalShapeContract"]["variant_failures"]) {
                    if let Some(c) = conditions(&cs, &f["conditions"]) {
                        let mut f = f.clone();
                        f["conditions"] = c;
                        failures.push(f);
                    }
                }
            } else {
                let mut patch = p[1].clone();
                patch["conditions"] = cs.clone();
                outputs.entry(name).or_default().push(patch);
            }
        }
    }
    if !outputs.values().any(|v| !v.is_empty()) {
        if !failures.is_empty() && failures.iter().all(|r| r["status"] == "invalid") {
            return Ok(invalid(
                "conditional_shape_all_variants_invalid",
                json!({"variant_failures":failures}),
            ));
        }
        base["reason"] = json!("conditional_shape_no_complete_variant");
        base["conditionalVariantFailures"] = json!(failures);
        return Ok(base);
    }
    let mut out = vec![];
    for name in rows(&v["node"]["outputs"]).iter().filter(|n| truth(n)) {
        let vs = unique_variants(outputs.remove(string(name)).unwrap_or_default());
        if vs.is_empty() {
            continue;
        }
        let mut merged = ty(v, &vs[0]);
        for var in &vs[1..] {
            let r = crate::portable_onnx_type::run(
                &json!({"operation":"union","left":merged,"right":ty(v,var),"tensor_type_ids":v["tensor_type_ids"]}),
            )?;
            if r["status"] != "pass" || r["type"].is_null() {
                return Ok(invalid(
                    "conditional_shape_variant_type_conflict",
                    json!({"output_name":name}),
                ));
            }
            merged = r["type"].clone();
        }
        let mut patch = descriptor(&merged, Value::Null);
        let keys = vs
            .iter()
            .chain(failures.iter())
            .flat_map(|v| {
                rows(&v["conditions"])
                    .iter()
                    .map(|c| string(&c["key"]).to_owned())
            })
            .collect::<BTreeSet<_>>();
        patch["conditionalShapeVariants"] = json!(vs);
        patch["conditionalShapeContract"] = json!({"schema":"deepbom.onnx_conditional_shape_contract.v1","status":if failures.is_empty(){"assessed_complete"}else{"assessed_partial"},"variant_count":vs.len(),"unassessed_variant_count":failures.iter().filter(|r|r["status"]!="invalid").count(),"invalid_variant_count":failures.iter().filter(|r|r["status"]=="invalid").count(),"condition_keys":keys,"variant_failures":unique_failures(failures.clone())});
        out.push(json!([name, patch]));
    }
    if out.is_empty() {
        return Ok(base);
    }
    let mut r = result(out);
    r["reason"] = json!(if failures.is_empty() {
        "conditional_shape_contract_propagated"
    } else {
        "conditional_shape_contract_partially_propagated"
    });
    Ok(r)
}
