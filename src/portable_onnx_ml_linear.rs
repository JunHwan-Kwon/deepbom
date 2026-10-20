//! Source-pinned linear model contracts and scalar reference calculations.
//! Reference arithmetic is never promoted to observed MLAS execution evidence.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_ml::*;
use crate::portable_onnx_ml_mapping::{dtype, tensor};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
const LIMIT: i64 = 1_000_000;
pub(crate) struct Input {
    pub ty: Value,
    pub dtype: String,
    pub shape: Vec<Value>,
    pub declared: bool,
    pub rank: Option<usize>,
    pub rank_valid: bool,
    pub batches: Option<i64>,
    pub features: Option<i64>,
    pub failures: Vec<String>,
    pub reasons: Vec<String>,
}
pub(crate) fn input_contract(v: &Value, prefix: &str, allowed: &[&str]) -> Input {
    let t = ty(v, input(v, 0));
    let dt = dtype(&t).to_owned();
    let (mut failures, mut reasons) = (vec![], vec![]);
    if t.is_null() {
        reasons.push(format!("{prefix}_input_type_unresolved"));
    } else if t["kind"] != "tensor" {
        failures.push(format!("{prefix}_input_not_tensor:{}", string(&t["kind"])));
    }
    if dt == "UNKNOWN" {
        reasons.push(format!("{prefix}_input_dtype_unresolved"));
    } else if !allowed.contains(&dt.as_str()) {
        failures.push(format!("{prefix}_input_dtype_not_supported:{dt}"));
    }
    let declared = t["kind"] == "tensor" && t["shapeDeclared"] == true;
    let shape = if declared {
        rows(&t["shape"]).to_vec()
    } else {
        vec![]
    };
    let rank = if declared { Some(shape.len()) } else { None };
    let valid = matches!(rank, Some(1) | Some(2));
    if !declared {
        reasons.push(format!("{prefix}_input_shape_unresolved"));
    } else if !valid {
        failures.push(format!(
            "{prefix}_input_rank_not_one_or_two:{}",
            rank.unwrap()
        ));
    }
    let batches = match rank {
        Some(1) => Some(1),
        Some(2) => safe(&shape[0]).filter(|n| *n >= 0),
        _ => None,
    };
    let features = match rank {
        Some(1) => safe(&shape[0]).filter(|n| *n >= 0),
        Some(2) => safe(&shape[1]).filter(|n| *n >= 0),
        _ => None,
    };
    if valid && (batches.is_none() || features.is_none()) {
        reasons.push(format!("{prefix}_dynamic_batch_or_feature_dimension"));
    }
    Input {
        ty: t,
        dtype: dt,
        shape,
        declared,
        rank,
        rank_valid: valid,
        batches,
        features,
        failures,
        reasons,
    }
}
#[derive(Default)]
struct Runtime {
    status: &'static str,
    reason: String,
    expected: Option<i64>,
    used: i64,
    unused: i64,
    intercepts: bool,
    ignored: usize,
}
fn runtime(
    c: &Input,
    coefficients: usize,
    intercepts: usize,
    targets: Option<i64>,
    classifier: bool,
) -> Runtime {
    let prefix = if classifier {
        "linear_classifier"
    } else {
        "linear_regressor"
    };
    let mut r = Runtime {
        status: "fail",
        intercepts: if classifier {
            intercepts > 0
        } else {
            targets == Some(intercepts as i64)
        },
        ignored: if targets == Some(intercepts as i64) {
            0
        } else {
            intercepts
        },
        ..Default::default()
    };
    let reason = if classifier && intercepts == 0 {
        "pinned_ort_requires_nonempty_intercepts"
    } else if !classifier && targets.is_none() {
        "targets_outside_pinned_ort_range"
    } else if coefficients == 0 {
        "pinned_ort_requires_nonempty_coefficients"
    } else if !classifier && c.dtype != "UNKNOWN" && c.dtype != "FLOAT32" {
        r.reason = format!("{prefix}_pinned_ort_cpu_dtype_unsupported:{}", c.dtype);
        return r;
    } else if !c.rank_valid {
        if !c.declared {
            r.status = "partial";
        }
        "input_rank_or_shape_unresolved"
    } else if c.features.is_none() {
        r.status = "partial";
        "feature_count_unresolved"
    } else {
        r.expected = targets
            .and_then(|t| t.checked_mul(c.features.unwrap()))
            .filter(|n| *n <= SAFE);
        if let Some(expected) = r.expected {
            if (coefficients as i64) < expected {
                r.reason = format!("{prefix}_coefficients_undersized:{coefficients}:{expected}");
                r.used = coefficients as i64;
                return r;
            }
            r.status = "pass";
            r.used = expected;
            r.unused = coefficients as i64 - expected;
            if r.unused > 0 {
                "extra_coefficients_ignored"
            } else {
                "runtime_contract_resolved"
            }
        } else {
            r.status = "partial";
            "expected_coefficient_count_overflow"
        }
    };
    r.reason = format!("{prefix}_{reason}");
    r
}
pub(crate) fn logistic(x: f32) -> f32 {
    let positive = (1.0 / (1.0 + (-(x.abs() as f64)).exp()) as f32 as f64) as f32;
    if x < 0. {
        (1.0 - positive as f64) as f32
    } else {
        positive
    }
}
pub(crate) fn softmax(a: &[f32]) -> Vec<f32> {
    let max = if a.iter().any(|n| n.is_nan()) {
        f32::NAN
    } else {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    };
    let exps = a
        .iter()
        .map(|n| ((*n - max) as f64).exp() as f32)
        .collect::<Vec<_>>();
    let sum = exps.iter().fold(0f32, |s, n| s + n);
    exps.into_iter().map(|n| n / sum).collect()
}
fn transform(raw: &[f32], targets: usize, mode: &str, binary: bool) -> Option<Vec<f32>> {
    if mode == "PROBIT" {
        return None;
    }
    if binary {
        return Some(
            raw.iter()
                .flat_map(|n| [(1.0 - *n as f64) as f32, *n])
                .collect(),
        );
    }
    if targets == 1 || mode == "NONE" {
        return Some(raw.to_vec());
    }
    let mut output = vec![];
    for row in raw.chunks(targets.max(1)) {
        match mode {
            "LOGISTIC" => output.extend(row.iter().copied().map(logistic)),
            "SOFTMAX" | "SOFTMAX_ZERO" => output.extend(softmax(row)),
            _ => return None,
        }
    }
    Some(output)
}
fn unresolved(t: &Value, status: Option<&str>) -> Value {
    json!({"status":status.unwrap_or(if t["role"]=="initializer"{"not_assessed_initializer_values"}else{"not_assessed_runtime_values"}),"inputValueCount":null,"rawScoreCount":null,"nonFiniteRawScoreCount":null,"decisionBoundaryCount":null,"rawScorePreview":[],"outputPreview":[],"labelPreview":[]})
}
fn reference(
    v: &Value,
    c: &Input,
    r: &Runtime,
    targets: Option<i64>,
    coefficients: &[f32],
    intercepts: &[f32],
    labels: &[String],
    mode: &str,
    classifier: bool,
    binary: bool,
) -> Value {
    let t = input(v, 0);
    let Some(source) = source(t, &c.dtype) else {
        return unresolved(t, None);
    };
    if r.status != "pass" || c.batches.is_none() || c.features.is_none() || targets.is_none() {
        return unresolved(t, None);
    }
    let (b, f, k) = (c.batches.unwrap(), c.features.unwrap(), targets.unwrap());
    let total = b.checked_mul(f).filter(|n| *n <= SAFE);
    if total != Some(source.len() as i64) || total.is_none_or(|n| n > LIMIT) {
        return unresolved(t, Some("not_assessed_static_input_cardinality_or_limit"));
    }
    // Bound both output materialization and arithmetic, including zero-width rows.
    if b.checked_mul(k).is_none_or(|n| n > LIMIT)
        || b.checked_mul(k)
            .and_then(|n| n.checked_mul(f.max(1)))
            .is_none_or(|n| n > LIMIT)
    {
        return unresolved(t, Some("not_assessed_static_reference_work_limit"));
    }
    let (b, f, k) = (b as usize, f as usize, k as usize);
    let mut raw = Vec::with_capacity(b * k);
    for batch in 0..b {
        for target in 0..k {
            let mut sum = intercepts
                .get(target)
                .copied()
                .filter(|n| !n.is_nan())
                .unwrap_or(0.);
            for feature in 0..f {
                let product =
                    source[batch * f + feature].f32() * coefficients[target * f + feature];
                sum += product;
            }
            raw.push(sum);
        }
    }
    let mut label_values = vec![];
    let mut boundary = 0;
    if classifier {
        for scores in raw.chunks(k.max(1)) {
            let mut selected = 0;
            if k == 1 {
                selected = usize::from(scores[0] > 0.);
                if scores[0] == 0. || !scores[0].is_finite() {
                    boundary += 1;
                }
            } else {
                for i in 1..scores.len() {
                    if scores[i] > scores[selected] {
                        selected = i;
                    }
                }
                let mut sorted = scores.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                if sorted.len() > 1 && (sorted[0] == sorted[1] || !sorted[0].is_finite()) {
                    boundary += 1;
                }
            }
            label_values.push(labels.get(selected).cloned().unwrap_or("undefined".into()));
        }
    }
    let output = transform(&raw, k, mode, binary);
    json!({"status":if output.is_some(){"assessed_scalar_float32_reference_not_runtime_bit_exact"}else{"assessed_raw_scores_post_transform_not_materialized"},"inputValueCount":source.len(),"rawScoreCount":raw.len(),"nonFiniteRawScoreCount":raw.iter().filter(|n|!n.is_finite()).count(),"decisionBoundaryCount":boundary,"rawScorePreview":raw.iter().take(12).map(|n|ft(*n as f64)).collect::<Vec<_>>(),"outputPreview":output.as_ref().map(|a|a.iter().take(12).map(|n|ft(*n as f64)).collect::<Vec<_>>()).unwrap_or_default(),"labelPreview":label_values.into_iter().take(12).collect::<Vec<_>>()})
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let classifier = n["opType"] == "LinearClassifier";
    let prefix = if classifier {
        "linear_classifier"
    } else {
        "linear_regressor"
    };
    let c = input_contract(v, "linear_model", &["FLOAT32", "FLOAT64", "INT32", "INT64"]);
    let (mut failures, mut reasons) = (c.failures.clone(), c.reasons.clone());
    let (ca, ia, pa) = (
        attr(n, "coefficients"),
        attr(n, "intercepts"),
        attr(n, "post_transform"),
    );
    let coefficients = if ca.is_null() {
        if classifier {
            None
        } else {
            Some(vec![])
        }
    } else {
        attr_floats(ca)
    };
    let intercepts = if ia.is_null() {
        Some(vec![])
    } else {
        attr_floats(ia)
    };
    let mode = if pa.is_null() {
        Some("NONE")
    } else {
        attr_string(pa)
    };
    if ca.is_null() && classifier {
        failures.push("linear_classifier_coefficients_required_by_onnx_schema".into());
    } else if !ca.is_null() && coefficients.is_none() {
        failures.push(format!("{prefix}_coefficients_not_float_list"));
    }
    if !ia.is_null() && intercepts.is_none() {
        failures.push(format!("{prefix}_intercepts_not_float_list"));
    }
    let (coefficients, intercepts) = (
        coefficients.unwrap_or_default(),
        intercepts.unwrap_or_default(),
    );
    let invalid_mode = mode
        .is_none_or(|s| !["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"].contains(&s));
    let mut labels = vec![];
    let mut label_kind = "invalid";
    let mut label_status = "pass";
    let mut label_reason = String::new();
    let mut duplicate = 0;
    let mut multi = None;
    let mut target_value = None;
    let targets;
    if classifier {
        let (la, sa, ma) = (
            attr(n, "classlabels_ints"),
            attr(n, "classlabels_strings"),
            attr(n, "multi_class"),
        );
        let ints = if la.is_null() {
            Some(vec![])
        } else {
            attr_ints_exact(la)
        };
        let texts = if sa.is_null() {
            Some(vec![])
        } else {
            attr_strings(sa)
        };
        multi = if ma.is_null() {
            Some(BigInt::zero())
        } else {
            attr_int_exact(ma)
        };
        if !la.is_null() && ints.is_none() {
            failures.push("linear_classifier_int_labels_not_exact_int_list".into());
        }
        if !sa.is_null() && texts.is_none() {
            failures.push("linear_classifier_string_labels_not_string_list".into());
        }
        if multi.is_none() {
            failures.push("linear_classifier_multi_class_not_exact_int_scalar".into());
        }
        if invalid_mode {
            failures.push("linear_classifier_post_transform_invalid".into());
        }
        let (ints, texts) = (ints.unwrap_or_default(), texts.unwrap_or_default());
        targets = Some(intercepts.len() as i64);
        if ints.is_empty() == texts.is_empty() {
            label_status = "fail";
            label_reason = "linear_classifier_requires_exactly_one_nonempty_classlabel_list".into();
        } else {
            if !texts.is_empty() {
                label_kind = "string";
                labels = texts;
            } else {
                label_kind = "int64";
                labels = ints.iter().map(ToString::to_string).collect();
            }
            let expected = if intercepts.len() == 1 {
                2
            } else {
                intercepts.len()
            };
            if intercepts.is_empty() {
                label_status = "fail";
                label_reason = "linear_classifier_pinned_ort_requires_nonempty_intercepts".into();
            } else if labels.len() != expected {
                label_status = "fail";
                label_reason = format!(
                    "linear_classifier_label_count_mismatch:{}:{expected}",
                    labels.len()
                );
            } else {
                label_reason = "linear_classifier_label_contract_resolved".into();
            }
        }
        if label_status == "fail" {
            failures.push(label_reason.clone());
        }
        duplicate = labels.len() - unique(labels.clone()).len();
    } else {
        let a = attr(n, "targets");
        target_value = if a.is_null() {
            Some(BigInt::from(1))
        } else {
            attr_int_exact(a)
        };
        if target_value.is_none() {
            failures.push("linear_regressor_targets_not_exact_int_scalar".into());
        }
        if invalid_mode {
            failures.push("linear_regressor_post_transform_invalid".into());
        }
        targets = target_value
            .as_ref()
            .and_then(ToPrimitive::to_i64)
            .filter(|n| *n > 0 && *n <= i32::MAX as i64);
        if targets.is_none() {
            failures.push("linear_regressor_targets_outside_pinned_ort_range".into());
        }
    }
    let runtime = runtime(
        &c,
        coefficients.len(),
        intercepts.len(),
        targets,
        classifier,
    );
    if runtime.status == "fail" {
        failures.push(runtime.reason.clone());
    } else if runtime.status == "partial" {
        reasons.push(runtime.reason.clone());
    }
    let binary = classifier && targets == Some(1) && labels.len() == 2;
    let scores = if binary { Some(2) } else { targets };
    let declared = c.rank_valid && c.batches.is_some() && scores.is_some_and(|n| n > 0);
    let shape = if declared {
        vec![json!(c.batches), json!(scores)]
    } else {
        vec![]
    };
    let label_shape = if declared {
        vec![json!(c.batches)]
    } else {
        vec![]
    };
    let lt = tensor(
        v,
        if label_kind == "string" {
            "STRING"
        } else {
            "INT64"
        },
        &label_shape,
        declared,
    );
    let ot = tensor(v, "FLOAT32", &shape, declared);
    let nonfinite = coefficients
        .iter()
        .chain(&intercepts)
        .filter(|n| !n.is_finite())
        .count();
    let ref_intercepts = if classifier || runtime.intercepts {
        intercepts.as_slice()
    } else {
        &[]
    };
    let reference = reference(
        v,
        &c,
        &runtime,
        targets,
        &coefficients,
        ref_intercepts,
        &labels,
        mode.unwrap_or("NONE"),
        classifier,
        binary,
    );
    let mut risks = vec![];
    let mode = mode.unwrap_or("unresolved");
    let noop = ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].contains(&mode);
    if classifier {
        if label_status == "fail" {
            risks.push("linear_classifier_onnx_label_contract_invalid".into());
        }
        if runtime.status == "fail" {
            risks.push("linear_classifier_pinned_ort_runtime_contract_invalid".into());
        }
        if runtime.unused > 0 {
            risks.push("linear_classifier_unused_coefficients_ignored".into());
        }
        if duplicate > 0 {
            risks.push("linear_classifier_duplicate_labels_ambiguous_output_semantics".into());
        }
        if multi.as_ref().is_some_and(|n| !n.is_zero()) {
            risks.push("linear_classifier_multi_class_nonzero_ignored_by_pinned_ort".into());
        }
        if targets == Some(1) && !binary && noop {
            risks.push("linear_classifier_single_score_post_transform_noop".into());
        }
        if binary && mode == "PROBIT" {
            risks.push("linear_classifier_binary_probit_second_score_unwritten".into());
        }
        if binary && noop {
            risks.push(
                "linear_classifier_binary_post_transform_ignored_for_complement_expansion".into(),
            );
        }
    } else {
        if runtime.status == "fail" {
            risks.push("linear_regressor_pinned_ort_runtime_contract_invalid".into());
        }
        if ["FLOAT64", "INT32", "INT64"].contains(&c.dtype.as_str()) {
            risks.push("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel".into());
        }
        if runtime.unused > 0 {
            risks.push("linear_regressor_unused_coefficients_ignored".into());
        }
        if runtime.ignored > 0 {
            risks.push("linear_regressor_mismatched_intercepts_ignored".into());
        }
        if targets == Some(1) && noop {
            risks.push("linear_regressor_single_target_post_transform_noop".into());
        }
        if mode == "PROBIT" {
            risks.push("linear_regressor_probit_may_emit_non_finite".into());
        }
    }
    if nonfinite > 0 || reference["nonFiniteRawScoreCount"].as_u64().unwrap_or(0) > 0 {
        risks.push(format!("{prefix}_non_finite_parameter_or_reference_score"));
    }
    if classifier && reference["decisionBoundaryCount"].as_u64().unwrap_or(0) > 0 {
        risks.push("linear_classifier_reference_decision_boundary".into());
    }
    let status = if !failures.is_empty() {
        "fail"
    } else if !reasons.is_empty() {
        "partial"
    } else {
        "pass"
    };
    failures.extend(reasons);
    let codes = unique(failures);
    let mut row = templates()["linear_base"].clone();
    row.as_object_mut().unwrap().extend(
        templates()[if classifier {
            "linear_classifier_extra"
        } else {
            "linear_regressor_extra"
        }]
        .as_object()
        .unwrap()
        .clone(),
    );
    let fields = json!({"scope":v["scope"],"node_index":v["node_index"],"op_name":n["opType"],"contract_kind":prefix,"imported_opset":v["imported_opset"],"status":status,"input_name":input(v,0)["name"].as_str().unwrap_or(""),"output_name":n["outputs"][if classifier{1}else{0}].as_str().unwrap_or(""),"input_dtype":c.dtype,"input_kind":c.ty["kind"].as_str().unwrap_or("unresolved"),"input_rank":c.rank,"input_shape":c.shape,"exact_batch_count":c.batches,"exact_feature_count":c.features,"canonical_output_type":canonical(&ot),"output_dtype":"FLOAT32","exact_output_rank":if declared{Some(shape.len())}else{None},"exact_output_shape":shape,"exact_dense_output_element_count":if declared{count(&shape)}else{None},"output_names":n["outputs"],"canonical_output_types":if classifier{vec![canonical(&lt),canonical(&ot)]}else{vec![canonical(&ot)]},"canonical_output_shapes":if classifier{vec![label_shape.clone(),shape.clone()]}else{vec![shape.clone()]},"linear_onnx_contract_status":if classifier{label_status}else if targets.is_none(){"fail"}else{"pass"},"linear_onnx_contract_reason":if classifier{label_reason}else if targets.is_none(){"linear_regressor_targets_must_be_positive".into()}else{"linear_regressor_schema_contract_resolved".into()},"linear_pinned_ort_contract_status":runtime.status,"linear_pinned_ort_contract_reason":runtime.reason,"linear_class_or_target_count":targets,"linear_expected_coefficient_count":runtime.expected,"linear_coefficient_count":coefficients.len(),"linear_used_coefficient_count":runtime.used,"linear_unused_coefficient_count":runtime.unused,"linear_intercept_count":intercepts.len(),"linear_intercepts_used":runtime.intercepts,"linear_post_transform":mode,"linear_post_transform_source":if pa.is_null(){"onnx_schema_default_NONE"}else{"explicit_attribute"},"linear_non_finite_parameter_count":nonfinite,"reason_codes":codes,"risk_codes":risks});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    for (dst, src) in [
        ("assessment_status", "status"),
        ("input_value_count", "inputValueCount"),
        ("raw_score_count", "rawScoreCount"),
        ("non_finite_raw_score_count", "nonFiniteRawScoreCount"),
        ("decision_boundary_count", "decisionBoundaryCount"),
        ("raw_score_preview", "rawScorePreview"),
        ("output_preview", "outputPreview"),
        ("label_preview", "labelPreview"),
    ] {
        row[format!("linear_reference_{dst}")] = reference[src].clone();
    }
    if !classifier {
        row["linear_reference_decision_boundary_count"] = Value::Null;
        row["linear_ignored_intercept_count"] = json!(runtime.ignored);
        row["linear_targets_value"] = json!(target_value
            .map(|n| n.to_string())
            .unwrap_or("unresolved".into()));
        row["linear_targets_source"] = json!(if attr(n, "targets").is_null() {
            "onnx_schema_default_1_materialized_by_ort_schema_resolution"
        } else {
            "explicit_attribute"
        });
    } else {
        let extra = json!({"classifier_label_output_name":n["outputs"][0].as_str().unwrap_or(""),"classifier_score_output_name":n["outputs"][1].as_str().unwrap_or(""),"classifier_label_output_dtype":if label_kind=="string"{"STRING"}else{"INT64"},"classifier_label_output_shape":label_shape,"classifier_score_output_shape":shape,"classifier_score_class_count":scores,"classifier_binary_score_expansion":binary,"linear_label_kind":label_kind,"linear_label_count":labels.len(),"linear_label_values":labels,"linear_duplicate_label_count":duplicate,"linear_multi_class_value":multi.map(|n|n.to_string()).unwrap_or("unresolved".into()),"linear_multi_class_source":if attr(n,"multi_class").is_null(){"onnx_schema_default_0"}else{"explicit_attribute"}});
        row.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
    }
    let mut outputs = vec![];
    if status != "fail" && declared {
        if classifier && truth(&n["outputs"][0]) {
            outputs.push(json!([n["outputs"][0], descriptor(&lt, Value::Null)]));
        }
        let i = if classifier { 1 } else { 0 };
        if truth(&n["outputs"][i]) {
            outputs.push(json!([n["outputs"][i], descriptor(&ot, Value::Null)]));
        }
    }
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
