//! Tree ensemble shape, topology and source-pinned reference contracts.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_ml::{count, ft, templates, unique};
use crate::portable_onnx_ml_mapping::tensor;
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, R};
use serde_json::{json, Value};
pub(crate) struct Input {
    pub dtype: String,
    pub ty: Value,
    pub shape: Vec<Value>,
    pub rank: Option<usize>,
    pub batch: Value,
    pub features: Value,
    pub failures: Vec<String>,
    pub reasons: Vec<String>,
}
fn read_input(v: &Value, generic: bool) -> Input {
    let input = input(v, 0);
    let ty = ty(v, input);
    let dtype = if ty["kind"] == "tensor" {
        ty["dtype"].as_str().unwrap_or("UNKNOWN")
    } else {
        "UNKNOWN"
    }
    .to_owned();
    let declared = ty["kind"] == "tensor" && ty["shapeDeclared"] == true;
    let shape = if declared {
        rows(&ty["shape"]).to_vec()
    } else {
        vec![]
    };
    let rank = if declared { Some(shape.len()) } else { None };
    let (mut failures, mut reasons) = (vec![], vec![]);
    if input.is_null() {
        failures.push("tree_ensemble_input_missing".into());
    } else if ty.is_null() {
        reasons.push("tree_ensemble_input_type_unresolved".into());
    } else if ty["kind"] != "tensor" {
        failures.push(format!(
            "tree_ensemble_input_not_tensor:{}",
            ty["kind"].as_str().unwrap_or("")
        ));
    }
    if dtype == "UNKNOWN" {
        reasons.push("tree_ensemble_input_dtype_unresolved".into());
    } else if !(if generic {
        ["FLOAT16", "FLOAT32", "FLOAT64"].contains(&dtype.as_str())
    } else {
        ["FLOAT32", "FLOAT64", "INT32", "INT64"].contains(&dtype.as_str())
    }) {
        failures.push(format!("tree_ensemble_input_dtype_not_supported:{dtype}"));
    }
    if !declared {
        reasons.push("tree_ensemble_input_shape_unresolved".into());
    }
    let batch = if rank == Some(1) {
        json!(1)
    } else if rank.is_some_and(|n| n >= 2) {
        shape[0].clone()
    } else {
        Value::Null
    };
    let features = if rank == Some(1) {
        shape[0].clone()
    } else if rank.is_some_and(|n| n >= 2) {
        json!(count(&shape[1..]))
    } else {
        Value::Null
    };
    Input {
        dtype,
        ty,
        shape,
        rank,
        batch,
        features,
        failures,
        reasons,
    }
}
pub(crate) fn version(op: &str, opset: &Value) -> Option<i64> {
    crate::portable_onnx_ml_schema::version(op, safe(opset))
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let generic = n["opType"] == "TreeEnsemble";
    let classifier = n["opType"] == "TreeEnsembleClassifier";
    let input = read_input(v, generic);
    let ver = version(n["opType"].as_str().unwrap_or(""), &v["imported_opset"]);
    let (mut failures, mut reasons) = (input.failures.clone(), input.reasons.clone());
    if ver.is_none() {
        failures.push("tree_ensemble_operator_not_defined_at_imported_opset".into());
    }
    if generic {
        if input.rank.is_some_and(|n| n != 2) {
            failures.push("tree_ensemble_v5_input_rank_not_2".into());
        }
    } else if ver.is_some_and(|n| n >= 3) && input.rank.is_some_and(|n| n != 2) {
        failures.push("tree_ensemble_input_rank_not_2".into());
    } else if ver == Some(1) && input.rank.is_some_and(|n| n != 1 && n != 2) {
        reasons.push("tree_ensemble_v1_input_rank_outside_documented_contract".into());
    }
    let p = crate::portable_onnx_ml_tree_data::parse(v, ver, &input.dtype, &mut failures);
    let g = crate::portable_onnx_ml_tree_graph::run(
        &p,
        safe(&input.features),
        &mut failures,
        &mut reasons,
    );
    if !generic && p.targets < 1 {
        failures.push(
            if classifier {
                "tree_classifier_class_count_not_positive"
            } else {
                "tree_regressor_n_targets_not_positive"
            }
            .into(),
        );
    }
    let gap = if generic {
        input.dtype == "FLOAT16"
    } else {
        !classifier && ["INT32", "INT64"].contains(&input.dtype.as_str())
    };
    if gap {
        reasons.push(
            if generic {
                "tree_ensemble_v5_float16_missing_pinned_ort_cpu_kernel"
            } else {
                "tree_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"
            }
            .into(),
        );
    }
    if !generic {
        if classifier && p.duplicate_labels > 0 {
            reasons.push("tree_classifier_duplicate_labels_ambiguous_output_semantics".into());
        }
        if ver == Some(5) {
            reasons.push("tree_ensemble_legacy_operator_deprecated_at_opset_5".into());
        }
        let c = p.base.values.len() as i64;
        if c > 0 && c != p.targets {
            if classifier && p.targets == 2 && c == 1 {
                reasons.push(
                    "tree_classifier_binary_single_base_value_semantics_underspecified".into(),
                );
            } else {
                failures.push("tree_ensemble_base_value_cardinality_mismatch".into());
            }
        }
        if !["NONE", "SOFTMAX", "LOGISTIC", "SOFTMAX_ZERO", "PROBIT"].contains(&p.post.as_str()) {
            failures.push("tree_ensemble_invalid_post_transform".into());
        }
        if !classifier && !["SUM", "AVERAGE", "MIN", "MAX"].contains(&p.aggregate.as_str()) {
            failures.push("tree_regressor_invalid_aggregate_function".into());
        }
    }
    let declared = (input.rank == Some(2) || !generic && ver == Some(1) && input.rank == Some(1))
        && p.targets > 0;
    let shape = if generic && input.rank == Some(2) {
        vec![
            input.batch.clone(),
            if p.targets > 0 {
                json!(p.targets)
            } else {
                Value::Null
            },
        ]
    } else if declared {
        vec![input.batch.clone(), json!(p.targets)]
    } else {
        vec![]
    };
    let labelshape = if declared && classifier {
        vec![input.batch.clone()]
    } else {
        vec![]
    };
    let dtype = if generic {
        input.dtype.as_str()
    } else {
        "FLOAT32"
    };
    let ot = tensor(v, dtype, &shape, declared);
    let lt = tensor(
        v,
        if p.label_kind == "string" {
            "STRING"
        } else {
            "INT64"
        },
        &labelshape,
        declared,
    );
    let reference = if !failures.is_empty() {
        crate::portable_onnx_ml_tree_eval::unresolved("not_assessed_invalid_contract")
    } else if gap {
        crate::portable_onnx_ml_tree_eval::unresolved("not_assessed_pinned_cpu_dtype_gap")
    } else {
        crate::portable_onnx_ml_tree_eval::run(v, &input, &p, &g)
    };
    if reference["status"] == "not_assessed_work_limit" {
        reasons.push("tree_ensemble_reference_work_limit".into());
    }
    let mut risks = Vec::<String>::new();
    macro_rules! risk {
        ($cond:expr,$name:expr) => {
            if $cond {
                risks.push($name.into());
            }
        };
    }
    let positive = |name: &str| g.stats[name].as_u64().unwrap_or(0) > 0;
    if generic {
        risk!(
            gap,
            "tree_ensemble_v5_float16_missing_pinned_ort_cpu_kernel"
        );
        risk!(
            positive("orphanNodeCount"),
            "tree_ensemble_v5_unreachable_serialized_nodes_or_leaves"
        );
        risk!(
            positive("multipleParentNodeCount"),
            "tree_ensemble_v5_shared_subtree_not_strict_tree"
        );
        risk!(
            p.membership.duplicates > 0,
            "tree_ensemble_v5_duplicate_membership_values"
        );
        risk!(
            p.membership.zeros > 0,
            "tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser"
        );
    } else {
        risk!(
            ver == Some(5),
            "tree_ensemble_legacy_operator_deprecated_at_opset_5"
        );
        risk!(
            gap,
            "tree_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"
        );
        risk!(
            classifier && p.duplicate_labels > 0,
            "tree_classifier_duplicate_labels_ambiguous_output_semantics"
        );
        risk!(
            positive("nonLeafWeightCount"),
            "tree_ensemble_nonleaf_weights_ignored_by_pinned_ort"
        );
        risk!(
            positive("singleTargetIgnoredWeightCount"),
            "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort"
        );
        risk!(
            positive("multipleParentNodeCount"),
            "tree_ensemble_shared_subtree_not_strict_tree"
        );
    }
    risk!(
        p.nonfinite > 0 || reference["nonFiniteScoreCount"].as_u64().unwrap_or(0) > 0,
        "tree_ensemble_non_finite_parameter_or_reference_score"
    );
    risk!(
        reference["decisionBoundaryCount"].as_u64().unwrap_or(0) > 0,
        "tree_ensemble_reference_decision_boundary"
    );
    if generic {
        risk!(
            p.targets == 1 && !["NONE", "PROBIT"].contains(&p.post.as_str()),
            "tree_ensemble_v5_single_target_post_transform_noop"
        );
    } else {
        risk!(
            reference["unwrittenScoreCount"].as_u64().unwrap_or(0) > 0,
            "tree_classifier_binary_post_transform_leaves_score_unwritten"
        );
        risk!(
            !classifier && p.targets == 1 && !["NONE", "PROBIT"].contains(&p.post.as_str()),
            "tree_regressor_single_target_post_transform_noop"
        );
        risk!(
            classifier && p.targets == 2 && p.base.values.len() == 1,
            "tree_classifier_binary_single_base_value_semantics_underspecified"
        );
        let represented = p
            .weights
            .ids
            .iter()
            .map(|n| ft(*n))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        risk!(
            classifier
                && p.targets == 2
                && represented > 1
                && p.label_kind == "int64"
                && p.labels != vec!["0", "1"],
            "tree_classifier_pinned_ort_binary_label_index_semantics"
        );
    }
    let status = if !failures.is_empty() {
        "fail"
    } else if !reasons.is_empty() {
        "partial"
    } else {
        "pass"
    };
    let first = failures.first().cloned().unwrap_or_default();
    let onnxstatus = if failures.is_empty() { "pass" } else { "fail" };
    let ortstatus = if failures.is_empty() && !gap {
        "pass"
    } else {
        "fail"
    };
    failures.extend(reasons);
    let codes = unique(failures);
    let mut row = templates()["tree_base"].clone();
    row.as_object_mut().unwrap().extend(
        templates()["tree_reference_extra"]
            .as_object()
            .unwrap()
            .clone(),
    );
    let extra = json!({"scope":v["scope"],"node_index":v["node_index"],"op_name":n["opType"],"contract_kind":if generic{"tree_ensemble_v5"}else if classifier{"tree_ensemble_classifier"}else{"tree_ensemble_regressor"},"imported_opset":v["imported_opset"],"resolved_schema_version":ver,"status":status,"input_name":super::portable_onnx_semantics::input(v,0)["name"].as_str().unwrap_or(""),"output_name":n["outputs"][if classifier{1}else{0}].as_str().unwrap_or(""),"input_dtype":input.dtype,"input_kind":input.ty["kind"].as_str().unwrap_or("unresolved"),"input_rank":input.rank,"input_shape":input.shape,"exact_batch_count":input.batch,"exact_feature_count":input.features,"canonical_output_type":canonical(&ot),"output_dtype":dtype,"exact_output_rank":if declared{Some(shape.len())}else{None},"exact_output_shape":shape,"exact_dense_output_element_count":if declared{count(&shape)}else{None},"output_names":n["outputs"],"canonical_output_types":if classifier{vec![canonical(&lt),canonical(&ot)]}else{vec![canonical(&ot)]},"canonical_output_shapes":if classifier{vec![labelshape.clone(),shape.clone()]}else{vec![shape.clone()]}});
    row.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let extra = json!({"tree_encoding":if generic{"indexed_branch_leaf_v5"}else{"legacy_tuple_v1_v3_v5"},"tree_deprecated_operator":!generic&&ver==Some(5),"tree_onnx_contract_status":onnxstatus,"tree_onnx_contract_reason":first,"tree_pinned_ort_contract_status":ortstatus,"tree_pinned_ort_contract_reason":if first.is_empty()&&gap{"schema_dtype_missing_pinned_ort_cpu_kernel"}else{&first},"tree_aggregate_function":p.aggregate,"tree_aggregate_function_source":p.aggregate_source,"tree_post_transform":p.post,"tree_post_transform_source":p.post_source,"tree_base_value_count":p.base.values.len(),"tree_base_value_source":if generic{"not_applicable"}else{&p.base.source},"tree_class_or_target_count":p.targets,"tree_class_label_kind":if classifier{&p.label_kind}else{"not_applicable"},"tree_class_label_count":p.labels.len(),"tree_duplicate_class_label_count":p.duplicate_labels,"tree_class_label_preview":p.labels.iter().take(64).collect::<Vec<_>>(),"tree_pinned_cpu_dtype_gap":gap,"tree_non_finite_parameter_count":p.nonfinite,"reason_codes":codes,"risk_codes":risks});
    row.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    for (key, source) in templates()["tree_graph_mapping"].as_object().unwrap() {
        row[key] = g.stats[source.as_str().unwrap()].clone();
    }
    for (key, source) in templates()["tree_reference_mapping"].as_object().unwrap() {
        row[key] = reference[source.as_str().unwrap()].clone();
    }
    let wc = if generic {
        p.leaf_weights.values.len()
    } else {
        p.weights.ids.len()
    } as u64;
    let used = g.stats["usedWeightCount"].as_u64().unwrap_or(0);
    let unused = g.stats["unusedWeightCount"].as_u64().unwrap_or(0);
    row["tree_weight_tuple_count"] = json!(wc);
    row["tree_used_weight_count"] = json!(used);
    row["tree_unused_weight_count"] = json!(unused);
    row["tree_ignored_nonleaf_weight_count"] = if generic {
        json!(0)
    } else {
        g.stats["nonLeafWeightCount"].clone()
    };
    row["tree_unresolved_weight_count"] = json!(wc.saturating_sub(used + unused));
    let mut outputs = vec![];
    if status != "fail" && declared {
        if classifier && truth(&n["outputs"][0]) {
            outputs.push(json!([n["outputs"][0], descriptor(&lt, Value::Null)]));
        }
        let index = if classifier { 1 } else { 0 };
        if truth(&n["outputs"][index]) {
            outputs.push(json!([n["outputs"][index], descriptor(&ot, Value::Null)]));
        }
    }
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
