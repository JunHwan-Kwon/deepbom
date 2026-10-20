//! Core ML deployment-floor semantics; source-pinned tables remain data supplied
//! by the adapter. A load floor is not device placement or execution evidence.
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn rows(v: &Value) -> impl Iterator<Item = &Value> {
    v.as_array().into_iter().flatten()
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    let declared = v["specificationVersion"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 9_007_199_254_740_991)
        .ok_or("Core ML specificationVersion must be a positive integer.")?;
    let model = v["modelType"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    let mut req = vec![];
    fn add(req: &mut Vec<Value>, n: u64, basis: String) {
        req.push(json!({"minimum_specification_version":n,"basis":basis,"status":"source_mapped"}));
    }
    add(
        &mut req,
        v["model_type_floors"][model].as_u64().unwrap_or(1),
        format!("model type {model}"),
    );
    if v["isUpdatable"] == true {
        add(&mut req, 4, "isUpdatable=true".into());
    }
    let d = &v["description"];
    if !d.is_null() && !d.is_object() {
        return Err("Core ML description must be an object.".into());
    }
    for owner in std::iter::once(d).chain(rows(&d["functions"])) {
        if !owner.is_null() && !owner.is_object() {
            return Err("Core ML function description must be an object.".into());
        }
        for key in [
            "inputs",
            "outputs",
            "states",
            "training_inputs",
            "functions",
        ] {
            if let Some(value) = owner.get(key) {
                if !value.is_null() && !value.is_array() {
                    return Err(format!("Core ML {key} must be an array."));
                }
            }
        }
    }
    if !v["mlProgram"].is_null() && !v["mlProgram"].is_object() {
        return Err("Core ML program must be an object.".into());
    }
    if let Some(functions) = v["mlProgram"].get("functions") {
        if !functions.is_null() && !functions.is_object() {
            return Err("Core ML program functions must be an object.".into());
        }
    }
    let functions = rows(&d["functions"]).collect::<Vec<_>>();
    if functions.len() > 1 {
        add(
            &mut req,
            9,
            format!("{} function descriptions", functions.len()),
        );
    }
    let mut features = vec![];
    for k in ["inputs", "outputs", "states", "training_inputs"] {
        features.extend(rows(&d[k]));
    }
    for f in functions {
        for k in ["inputs", "outputs", "states"] {
            features.extend(rows(&f[k]));
        }
    }
    if features.iter().any(|f| !f.is_object()) {
        return Err("Core ML feature descriptions must be objects.".into());
    }
    if let Some(functions) = v["mlProgram"]["functions"].as_object() {
        if functions.values().any(|f| {
            !f.is_object()
                || f.get("opset")
                    .is_some_and(|o| !o.is_null() && !o.is_string())
        }) {
            return Err(
                "Core ML program functions require object descriptions and text opsets.".into(),
            );
        }
    }
    for (matches, n, label) in [
        (
            features
                .iter()
                .any(|f| f["dtype"] == "FLOAT16" || f["dtype"] == "IMAGE_GRAYSCALE_FLOAT16"),
            7,
            "FLOAT16 external feature type",
        ),
        (
            features.iter().any(|f| f["dtype"] == "INT8"),
            10,
            "INT8 external MultiArray feature type",
        ),
        (
            features.iter().any(|f| f["feature_type"] == "state"),
            9,
            "state feature type",
        ),
        (
            features.iter().any(|f| {
                f["constraints"]["flexibility"]["kind"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty())
            }),
            3,
            "flexible image or MultiArray shape",
        ),
    ] {
        if matches {
            add(&mut req, n, label.into());
        }
    }
    let opsets: BTreeSet<_> = v["mlProgram"]["functions"]
        .as_object()
        .into_iter()
        .flat_map(|m| m.values())
        .filter_map(|f| f["opset"].as_str().filter(|s| !s.is_empty()))
        .collect();
    let mut opsets: Vec<_> = opsets.into_iter().collect();
    opsets.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    let mut unmapped = false;
    for op in &opsets {
        if let Some(n) = v["opset_to_spec"][*op].as_u64() {
            add(&mut req, n, format!("MIL opset {op}"));
        } else {
            unmapped = true;
            req.push(json!({"minimum_specification_version":null,"basis":format!("MIL opset {op}"),"status":"not_mapped_by_pinned_source"}));
        }
    }
    let floor = req
        .iter()
        .filter_map(|r| r["minimum_specification_version"].as_u64())
        .max()
        .unwrap_or(1);
    let known = &v["floors"][declared.to_string()];
    let mut contradictions = vec![];
    if declared < floor {
        contradictions.push(format!(
            "declared specification {declared} is below observed feature floor {floor}"
        ));
    }
    if known.is_null() {
        contradictions.push(format!(
            "specification {declared} is newer than the pinned OS availability table"
        ));
    }
    let status = if declared < floor {
        "invalid_declared_version_below_observed_feature_floor"
    } else if known.is_null() {
        "not_assessed_newer_than_pinned_table"
    } else if unmapped {
        "not_assessed_unmapped_feature_floor"
    } else {
        "assessed"
    };
    Ok(
        json!({"schema":"deepbom.coreml.deployment_floor.v1","status":status,"evidence_class":"OBSERVED/SOURCE_PINNED/DERIVED","declared_specification_version":declared,"declared_load_floor":known,"observed_feature_minimum_specification_version":if unmapped{Value::Null}else{json!(floor)},"observed_feature_floor":if unmapped{Value::Null}else{v["floors"][floor.to_string()].clone()},"observed_feature_requirements":req,"mil_opsets":opsets,"contradiction_count":contradictions.len(),"contradictions":contradictions}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_opset_is_not_assessed() {
        let r=run(&json!({"specificationVersion":9,"modelType":"mlProgram","model_type_floors":{"mlProgram":6},"floors":{"9":{}},"mlProgram":{"functions":{"main":{"opset":"Future"}}}})).unwrap();
        assert_eq!(r["status"], "not_assessed_unmapped_feature_floor");
        assert!(r["observed_feature_floor"].is_null());
    }
}
