//! Pinned ai.onnx.ml formal schemas. Shared by every host channel.
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::sync::OnceLock;
fn schemas() -> &'static Value {
    static V: OnceLock<Value> = OnceLock::new();
    V.get_or_init(|| serde_json::from_str(include_str!("onnx_ml_schemas.json")).unwrap())
}
pub(crate) fn version(op: &str, opset: Option<i64>) -> Option<i64> {
    let n = opset.filter(|n| *n >= 1)?;
    Some(match op {
        "LabelEncoder" => {
            if n >= 4 {
                4
            } else if n >= 2 {
                2
            } else {
                1
            }
        }
        "TreeEnsemble" => {
            if n >= 5 {
                5
            } else {
                return None;
            }
        }
        "TreeEnsembleClassifier" | "TreeEnsembleRegressor" => {
            if n >= 5 {
                5
            } else if n >= 3 {
                3
            } else {
                1
            }
        }
        _ => 1,
    })
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let op = n["opType"].as_str().unwrap_or("UNKNOWN");
    let all = schemas();
    if v["operation"] == "ml_registry" {
        return Ok(json!(all["ML_SCHEMAS"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>()));
    }
    if v["operation"] == "ml_version" {
        return Ok(json!(version(op, safe(&v["imported_opset"]))));
    }
    if v["operation"] == "ml_can" {
        return Ok(json!(
            domain(n) == "ai.onnx.ml" && all["ML_SCHEMAS"].get(op).is_some()
        ));
    }
    let version = version(op, safe(&v["imported_opset"]));
    let attrs = match op {
        "LabelEncoder" => {
            &all["LABEL_ENCODER_SCHEMAS"][version.map(|n| n.to_string()).unwrap_or_default()]
        }
        "TreeEnsembleClassifier" => {
            &all[if version == Some(1) {
                "TREE_CLASSIFIER_V1_SCHEMA"
            } else {
                "TREE_CLASSIFIER_V3_SCHEMA"
            }]
        }
        "TreeEnsembleRegressor" => {
            &all[if version == Some(1) {
                "TREE_REGRESSOR_V1_SCHEMA"
            } else {
                "TREE_REGRESSOR_V3_SCHEMA"
            }]
        }
        _ => &all["ML_SCHEMAS"][op],
    };
    let ni = rows(&n["inputs"]).len();
    let no = rows(&n["outputs"]).len();
    let (min, max) = match op {
        "FeatureVectorizer" => (1, None),
        "ArrayFeatureExtractor" => (2, Some(2)),
        _ => (1, Some(1)),
    };
    let out = if [
        "LinearClassifier",
        "SVMClassifier",
        "TreeEnsembleClassifier",
    ]
    .contains(&op)
    {
        2
    } else {
        1
    };
    let required: Vec<&str> = match op {
        "LinearClassifier" => vec!["coefficients"],
        "TreeEnsemble" => vec![
            "nodes_featureids",
            "nodes_splits",
            "nodes_modes",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_trueleafs",
            "nodes_falseleafs",
            "tree_roots",
            "leaf_targetids",
            "leaf_weights",
        ],
        _ => vec![],
    };
    let mut reasons = vec![];
    if version.is_none() {
        reasons.push("operator_not_defined_at_imported_opset".into());
    }
    if ni < min || max.is_some_and(|m| ni > m) {
        reasons.push(format!(
            "input_count_mismatch:{ni}:{min}:{}",
            max.map(|n| n.to_string()).unwrap_or("variadic".into())
        ));
    }
    if no != out {
        reasons.push(format!("output_count_mismatch:{no}:{out}:{out}"));
    }
    for i in 0..ni {
        if !truth(&n["inputs"][i]) {
            reasons.push(format!("required_input_omitted:{i}"));
        }
    }
    for i in 0..out {
        if !truth(&n["outputs"][i]) {
            reasons.push(format!("required_output_omitted:{i}"));
        }
    }
    for name in rows(&n["duplicateAttributeNames"]) {
        let code = format!("duplicate_attribute_name:{}", string(name));
        if !reasons.contains(&code) {
            reasons.push(code);
        }
    }
    for a in rows(&n["attributes"]) {
        let name = string(&a[0]);
        let e = &attrs[name];
        if e.is_null() {
            reasons.push(format!("attribute_not_defined:{name}"));
            continue;
        }
        let r = attribute(&a[1], false);
        if r["status"] != "pass" {
            reasons.push(format!("{}:{name}", string(&r["reason"])));
        } else if &r["type"] != e {
            reasons.push(format!("attribute_type_mismatch:{name}:{e}:{}", r["type"]));
        }
    }
    for name in &required {
        if attr(n, name).is_null() {
            reasons.push(format!("required_attribute_missing:{name}"));
        }
    }
    let mut names = rows(&n["attributes"])
        .iter()
        .map(|a| string(&a[0]))
        .collect::<Vec<_>>();
    names.sort();
    let reasons = crate::portable_onnx_ml::unique(reasons);
    let label = version
        .map(|n| n.to_string())
        .unwrap_or("unresolved".into());
    Ok(
        json!({"op_name":op,"domain":"ai.onnx.ml","imported_opset":v["imported_opset"],"schema_since_version":version,"input_count":ni,"output_count":no,"explicit_attributes":names,"schema_input_range":[min,max],"schema_output_range":[out,out],"schema_input_options":if op=="FeatureVectorizer"{"V"}else{"R"},"schema_output_options":"R","schema_attribute_count":attrs.as_object().map(|a|a.len()).unwrap_or(0),"required_attributes":required,"status":if reasons.is_empty(){"pass"}else{"fail"},"reason_codes":reasons,"detail":if reasons.is_empty(){format!("The NodeProto matches the pinned ai.onnx.ml {op}-{label} formal schema.")}else{format!("The NodeProto violates the pinned ai.onnx.ml {op}-{label} formal schema.")}}),
    )
}
