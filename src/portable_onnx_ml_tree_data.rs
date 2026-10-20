//! ONNX tree attributes retain source dtype, exact integer validity and membership boundaries.
use crate::portable_onnx_ml::{attr_int_exact, attr_string, exact_int, float, ft};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde_json::Value;
#[derive(Default)]
pub(crate) struct Numbers {
    pub present: bool,
    pub values: Vec<f64>,
    pub source: String,
    pub dtype: String,
}
#[derive(Default)]
pub(crate) struct Nodes {
    pub trees: Vec<f64>,
    pub ids: Vec<f64>,
    pub features: Vec<f64>,
    pub modes: Vec<String>,
    pub mode_ids: Numbers,
    pub true_ids: Vec<f64>,
    pub false_ids: Vec<f64>,
    pub true_leaves: Vec<f64>,
    pub false_leaves: Vec<f64>,
    pub missing: Vec<f64>,
    pub values: Numbers,
    pub hitrates: Numbers,
}
#[derive(Default)]
pub(crate) struct Weights {
    pub trees: Vec<f64>,
    pub nodes: Vec<f64>,
    pub ids: Vec<f64>,
    pub values: Numbers,
}
#[derive(Default)]
pub(crate) struct Membership {
    pub sets: Vec<Vec<f64>>,
    pub by_node: Vec<Option<Vec<f64>>>,
    pub nodes: usize,
    pub separators: usize,
    pub count: usize,
    pub duplicates: usize,
    pub zeros: usize,
}
#[derive(Default)]
pub(crate) struct Parsed {
    pub generic: bool,
    pub classifier: bool,
    pub nodes: Nodes,
    pub weights: Weights,
    pub base: Numbers,
    pub labels: Vec<String>,
    pub label_kind: String,
    pub duplicate_labels: usize,
    pub targets: i64,
    pub post: String,
    pub post_source: String,
    pub aggregate: String,
    pub aggregate_source: String,
    pub nonfinite: usize,
    pub roots: Vec<f64>,
    pub leaf_targets: Vec<f64>,
    pub leaf_weights: Numbers,
    pub membership: Membership,
}
pub(crate) fn valid(n: f64) -> Option<i64> {
    if n.is_finite() && n.fract() == 0. && n.abs() <= SAFE as f64 {
        Some(n as i64)
    } else {
        None
    }
}
fn list_ok(a: &Value, kind: i64) -> bool {
    a["type"] == kind
        && rows(&a["valueTypesPresent"])
            .iter()
            .all(|v| v.as_i64() == Some(kind))
}
fn bigints(n: &Value, name: &str, failures: &mut Vec<String>) -> Vec<BigInt> {
    let a = attr(n, name);
    if a.is_null() {
        return vec![];
    }
    if !list_ok(a, 7) || !a["intExactDecimals"].is_array() {
        failures.push(format!(
            "tree_ensemble_integer_list_attribute_invalid:{name}"
        ));
        return vec![];
    }
    rows(&a["intExactDecimals"])
        .iter()
        .map(exact_int)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_else(|| {
            failures.push(format!(
                "tree_ensemble_integer_list_attribute_invalid:{name}"
            ));
            vec![]
        })
}
fn ints(n: &Value, name: &str, failures: &mut Vec<String>) -> Vec<f64> {
    bigints(n, name, failures)
        .iter()
        .map(|n| {
            if let Some(n) = n.to_i64().filter(|n| n.unsigned_abs() <= SAFE as u64) {
                n as f64
            } else {
                failures.push(format!(
                    "tree_ensemble_integer_attribute_out_of_safe_range:{name}"
                ));
                f64::NAN
            }
        })
        .collect()
}
fn floats(n: &Value, name: &str, failures: &mut Vec<String>) -> Vec<f64> {
    let a = attr(n, name);
    if a.is_null() {
        return vec![];
    }
    if !list_ok(a, 6) || !a["floats"].is_array() {
        failures.push(format!("tree_ensemble_float_list_attribute_invalid:{name}"));
        return vec![];
    }
    let values = if a["floatsText"].is_array() {
        &a["floatsText"]
    } else {
        &a["floats"]
    };
    rows(values)
        .iter()
        .map(|n| float(n).unwrap_or(f64::NAN) as f32 as f64)
        .collect()
}
fn strings(n: &Value, name: &str, failures: &mut Vec<String>) -> Vec<String> {
    let a = attr(n, name);
    if a.is_null() {
        return vec![];
    }
    if !list_ok(a, 8) || !a["strings"].is_array() {
        failures.push(format!(
            "tree_ensemble_string_list_attribute_invalid:{name}"
        ));
        return vec![];
    }
    rows(&a["strings"])
        .iter()
        .map(|s| string(s).into())
        .collect()
}
fn integer(n: &Value, name: &str, default: i64, failures: &mut Vec<String>) -> Option<BigInt> {
    let a = attr(n, name);
    if a.is_null() {
        return Some(BigInt::from(default));
    }
    let n = attr_int_exact(a);
    if n.is_none() {
        failures.push(format!("tree_ensemble_integer_attribute_invalid:{name}"));
    }
    n
}
fn text(n: &Value, name: &str, default: &str, failures: &mut Vec<String>) -> String {
    let a = attr(n, name);
    if a.is_null() {
        return default.into();
    }
    attr_string(a).map(str::to_owned).unwrap_or_else(|| {
        failures.push(format!("tree_ensemble_string_attribute_invalid:{name}"));
        "INVALID".into()
    })
}
fn tensor(n: &Value, name: &str, optional: bool, failures: &mut Vec<String>) -> Numbers {
    let a = attr(n, name);
    if a.is_null() {
        if !optional {
            failures.push(format!(
                "tree_ensemble_required_tensor_attribute_missing:{name}"
            ));
        }
        return Numbers {
            source: "absent".into(),
            dtype: "UNKNOWN".into(),
            ..Default::default()
        };
    }
    let t = &a["tensor"];
    if a["type"] != 4
        || t.is_null()
        || !rows(&a["valueTypesPresent"]).contains(&serde_json::json!(4))
    {
        failures.push(format!("tree_ensemble_tensor_attribute_invalid:{name}"));
        return Numbers {
            present: true,
            source: name.into(),
            dtype: "UNKNOWN".into(),
            ..Default::default()
        };
    }
    let shape = if t["shapeDeclared"] == true {
        rows(&t["shape"]).to_vec()
    } else {
        vec![]
    };
    let count = shape.first().and_then(safe).filter(|n| *n >= 0);
    if shape.len() != 1 || count.is_none() {
        failures.push(format!("tree_ensemble_tensor_attribute_not_1d:{name}"));
    }
    let values: Option<Vec<f64>> = if t["staticValuesCanonicalTextComplete"] == true
        && t["staticValuesCanonicalTexts"].is_array()
    {
        rows(&t["staticValuesCanonicalTexts"])
            .iter()
            .map(float)
            .collect()
    } else if t["staticValuesComplete"] == true && t["staticValues"].is_array() {
        let vals = if t["staticValuesText"].is_array() {
            &t["staticValuesText"]
        } else {
            &t["staticValues"]
        };
        rows(vals).iter().map(float).collect()
    } else if t["initializerIntegerValuesExactComplete"] == true
        && t["initializerIntegerValuesExactDecimals"].is_array()
    {
        rows(&t["initializerIntegerValuesExactDecimals"])
            .iter()
            .map(|n| exact_int(n).and_then(|n| n.to_f64()))
            .collect()
    } else {
        None
    };
    if values.is_none()
        || shape.len() == 1
            && count.is_some_and(|n| values.as_ref().is_some_and(|v| v.len() as i64 != n))
    {
        failures.push(format!(
            "tree_ensemble_tensor_attribute_payload_incomplete:{name}"
        ));
    }
    Numbers {
        present: true,
        values: values.unwrap_or_default(),
        source: name.into(),
        dtype: t["dtype"].as_str().unwrap_or("UNKNOWN").into(),
    }
}
fn choice(
    n: &Value,
    list: &str,
    ten: &str,
    version: Option<i64>,
    failures: &mut Vec<String>,
) -> Numbers {
    let (lp, tp) = (!attr(n, list).is_null(), !attr(n, ten).is_null());
    if lp && tp {
        failures.push(format!(
            "tree_ensemble_mutually_exclusive_attributes:{list}:{ten}"
        ));
    }
    if tp && version.unwrap_or(0) < 3 {
        failures.push(format!(
            "tree_ensemble_tensor_attribute_not_defined_at_schema:{ten}"
        ));
    }
    if tp {
        return tensor(n, ten, false, failures);
    }
    let values = floats(n, list, failures);
    Numbers {
        present: lp,
        source: if lp { list } else { "absent" }.into(),
        dtype: "FLOAT32".into(),
        values,
    }
}
fn membership(values: &[f64], modes: &[f64], failures: &mut Vec<String>) -> Membership {
    let mut m = Membership {
        nodes: modes.iter().filter(|n| **n == 6.).count(),
        ..Default::default()
    };
    let mut current = vec![];
    for v in values {
        if v.is_nan() {
            m.sets.push(std::mem::take(&mut current));
            m.separators += 1;
        } else {
            current.push(*v);
            m.count += 1;
            if *v == 0. {
                m.zeros += 1;
            }
        }
    }
    if !current.is_empty() {
        failures.push("tree_ensemble_v5_membership_values_missing_nan_terminator".into());
    }
    if m.separators != m.nodes || m.sets.len() != m.nodes {
        failures.push("tree_ensemble_v5_membership_set_cardinality_mismatch".into());
    }
    m.duplicates = m
        .sets
        .iter()
        .map(|s| {
            s.len()
                - s.iter()
                    .map(|n| ft(*n))
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
        })
        .sum();
    let mut i = 0;
    for n in modes {
        if *n == 6. {
            m.by_node
                .push(Some(m.sets.get(i).cloned().unwrap_or_default()));
            i += 1;
        } else {
            m.by_node.push(None);
        }
    }
    m
}
pub(crate) fn parse(
    v: &Value,
    version: Option<i64>,
    input_dtype: &str,
    failures: &mut Vec<String>,
) -> Parsed {
    let n = &v["node"];
    let generic = n["opType"] == "TreeEnsemble";
    let classifier = n["opType"] == "TreeEnsembleClassifier";
    let mut p = Parsed {
        generic,
        classifier,
        ..Default::default()
    };
    if !generic {
        p.nodes.trees = ints(n, "nodes_treeids", failures);
        p.nodes.ids = ints(n, "nodes_nodeids", failures);
        p.nodes.features = ints(n, "nodes_featureids", failures);
        p.nodes.modes = strings(n, "nodes_modes", failures);
        p.nodes.true_ids = ints(n, "nodes_truenodeids", failures);
        p.nodes.false_ids = ints(n, "nodes_falsenodeids", failures);
        p.nodes.missing = ints(n, "nodes_missing_value_tracks_true", failures);
        p.nodes.values = choice(
            n,
            "nodes_values",
            "nodes_values_as_tensor",
            version,
            failures,
        );
        let prefix = if classifier { "class" } else { "target" };
        p.weights.trees = ints(n, &format!("{prefix}_treeids"), failures);
        p.weights.nodes = ints(n, &format!("{prefix}_nodeids"), failures);
        p.weights.ids = ints(n, &format!("{prefix}_ids"), failures);
        p.weights.values = choice(
            n,
            &format!("{prefix}_weights"),
            &format!("{prefix}_weights_as_tensor"),
            version,
            failures,
        );
        p.base = choice(n, "base_values", "base_values_as_tensor", version, failures);
        if classifier {
            let ints = bigints(n, "classlabels_int64s", failures);
            let texts = strings(n, "classlabels_strings", failures);
            if attr(n, "classlabels_int64s").is_null() == attr(n, "classlabels_strings").is_null() {
                failures.push("tree_classifier_exactly_one_label_attribute_required".into());
            }
            p.label_kind = if !attr(n, "classlabels_strings").is_null() {
                "string"
            } else {
                "int64"
            }
            .into();
            p.labels = if p.label_kind == "string" {
                texts
            } else {
                ints.iter().map(ToString::to_string).collect()
            };
            p.duplicate_labels = p.labels.len()
                - p.labels
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
            p.targets = p.labels.len() as i64;
        } else {
            p.label_kind = "not_applicable".into();
            let raw = integer(n, "n_targets", 0, failures);
            p.targets = raw
                .as_ref()
                .and_then(ToPrimitive::to_i64)
                .filter(|n| *n >= 0 && *n <= SAFE)
                .unwrap_or(0);
            if raw
                .as_ref()
                .and_then(ToPrimitive::to_i64)
                .is_none_or(|n| n <= 0 || n > SAFE)
            {
                failures.push("tree_regressor_n_targets_invalid".into());
            }
        }
        p.post = text(n, "post_transform", "NONE", failures);
        p.post_source = if attr(n, "post_transform").is_null() {
            "onnx_schema_default_NONE"
        } else {
            "explicit_attribute"
        }
        .into();
        p.aggregate = if classifier {
            "SUM".into()
        } else {
            text(n, "aggregate_function", "SUM", failures)
        };
        p.aggregate_source = if classifier {
            "pinned_classifier_sum"
        } else if attr(n, "aggregate_function").is_null() {
            "onnx_schema_default_SUM"
        } else {
            "explicit_attribute"
        }
        .into();
        p.nonfinite = p
            .nodes
            .values
            .values
            .iter()
            .chain(&p.weights.values.values)
            .chain(&p.base.values)
            .filter(|n| !n.is_finite())
            .count();
    } else {
        p.nodes.features = ints(n, "nodes_featureids", failures);
        p.nodes.values = tensor(n, "nodes_splits", false, failures);
        p.nodes.mode_ids = tensor(n, "nodes_modes", false, failures);
        if p.nodes.mode_ids.values.iter().any(|n| valid(*n).is_none()) {
            failures.push("tree_ensemble_tensor_attribute_not_integer:nodes_modes".into());
        }
        p.nodes.true_ids = ints(n, "nodes_truenodeids", failures);
        p.nodes.false_ids = ints(n, "nodes_falsenodeids", failures);
        p.nodes.true_leaves = ints(n, "nodes_trueleafs", failures);
        p.nodes.false_leaves = ints(n, "nodes_falseleafs", failures);
        p.nodes.missing = ints(n, "nodes_missing_value_tracks_true", failures);
        p.nodes.hitrates = tensor(n, "nodes_hitrates", true, failures);
        p.leaf_targets = ints(n, "leaf_targetids", failures);
        p.leaf_weights = tensor(n, "leaf_weights", false, failures);
        p.roots = ints(n, "tree_roots", failures);
        let members = tensor(n, "membership_values", true, failures);
        for (name, t) in [
            ("nodes_splits", &p.nodes.values),
            ("leaf_weights", &p.leaf_weights),
            ("membership_values", &members),
        ] {
            if t.present && t.dtype != input_dtype {
                failures.push(format!(
                    "tree_ensemble_v5_{name}_dtype_mismatch:{}:{input_dtype}",
                    t.dtype
                ));
            }
        }
        if p.nodes.hitrates.present && p.nodes.hitrates.dtype != "FLOAT32" {
            failures.push("tree_ensemble_v5_nodes_hitrates_dtype_not_float32".into());
        }
        if p.nodes.mode_ids.present && p.nodes.mode_ids.dtype != "UINT8" {
            failures.push("tree_ensemble_v5_nodes_modes_dtype_not_uint8".into());
        }
        p.targets = integer(n, "n_targets", 0, failures)
            .and_then(|n| n.to_i64())
            .filter(|n| *n > 0 && *n <= SAFE)
            .unwrap_or(0);
        if p.targets == 0 {
            failures.push("tree_ensemble_v5_n_targets_not_positive".into());
        }
        p.aggregate =
            match integer(n, "aggregate_function", 1, failures).and_then(|n| n.to_i64()) {
                Some(0) => "AVERAGE",
                Some(1) => "SUM",
                Some(2) => "MIN",
                Some(3) => "MAX",
                _ => "INVALID",
            }
            .into();
        p.post = match integer(n, "post_transform", 0, failures).and_then(|n| n.to_i64()) {
            Some(0) => "NONE",
            Some(1) => "SOFTMAX",
            Some(2) => "LOGISTIC",
            Some(3) => "SOFTMAX_ZERO",
            Some(4) => "PROBIT",
            _ => "INVALID",
        }
        .into();
        if p.aggregate == "INVALID" {
            failures.push("tree_ensemble_v5_invalid_aggregate_function".into());
        }
        if p.post == "INVALID" {
            failures.push("tree_ensemble_v5_invalid_post_transform".into());
        }
        p.membership = membership(&members.values, &p.nodes.mode_ids.values, failures);
        p.aggregate_source = if attr(n, "aggregate_function").is_null() {
            "onnx_schema_default_SUM_1"
        } else {
            "explicit_attribute"
        }
        .into();
        p.post_source = if attr(n, "post_transform").is_null() {
            "onnx_schema_default_NONE_0"
        } else {
            "explicit_attribute"
        }
        .into();
        p.nonfinite = p
            .nodes
            .values
            .values
            .iter()
            .chain(&p.leaf_weights.values)
            .chain(p.membership.sets.iter().flatten())
            .filter(|n| !n.is_finite())
            .count();
    }
    p
}
