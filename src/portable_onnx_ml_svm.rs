//! Source-pinned SVM schema/runtime layout contracts. See independent conformance oracle.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_ml::{
    attr_int_exact, attr_string, count, exact_int, float, ft, templates, unique,
};
use crate::portable_onnx_ml_mapping::tensor;
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, R};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
#[derive(Default)]
pub(crate) struct Attributes {
    pub classifier: bool,
    pub kernel: String,
    pub params: Vec<f32>,
    pub params_present: bool,
    pub vectors: Vec<BigInt>,
    pub support: Vec<f32>,
    pub coefficients: Vec<f32>,
    pub prob_a: Vec<f32>,
    pub prob_b: Vec<f32>,
    pub rho: Vec<f32>,
    pub labels: Vec<String>,
    pub label_kind: &'static str,
    pub post: String,
    pub n_supports: Option<BigInt>,
    pub one_class: BigInt,
}
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
fn read_input(v: &Value) -> Input {
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
        failures.push("svm_input_missing".into());
    } else if ty["kind"] != "tensor" {
        failures.push("svm_input_not_dense_tensor".into());
    } else if !["FLOAT32", "FLOAT64", "INT32", "INT64"].contains(&dtype.as_str()) {
        failures.push(format!("svm_input_dtype_not_supported_by_schema:{dtype}"));
    }
    if rank.is_some_and(|n| n != 1 && n != 2) {
        failures.push(format!("svm_input_rank_not_one_or_two:{}", rank.unwrap()));
    }
    if rank.is_none() {
        reasons.push("svm_input_rank_unresolved".into());
    }
    let batch = match rank {
        Some(1) => json!(1),
        Some(2) => shape[0].clone(),
        _ => Value::Null,
    };
    let features = match rank {
        Some(1) => shape[0].clone(),
        Some(2) => shape[1].clone(),
        _ => Value::Null,
    };
    if rank.is_some() && safe(&features).is_none_or(|n| n < 0) {
        reasons.push("svm_feature_count_unresolved".into());
    }
    if rank == Some(2) && safe(&batch).is_none_or(|n| n < 0) {
        reasons.push("svm_batch_count_unresolved".into());
    }
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
fn list_type(a: &Value, kind: i64) -> bool {
    a["type"] == kind
        && a["valueTypesPresent"].is_array()
        && (rows(&a["valueTypesPresent"]).is_empty() || a["valueTypesPresent"] == json!([kind]))
}
fn floats(n: &Value, name: &str, prefix: &str, failures: &mut Vec<String>) -> Vec<f32> {
    let a = attr(n, name);
    if a.is_null() {
        return vec![];
    }
    let values = if a["floatsText"].is_array() {
        &a["floatsText"]
    } else {
        &a["floats"]
    };
    let values = if list_type(a, 6) && a["floats"].is_array() {
        values
            .as_array()
            .and_then(|v| v.iter().map(|n| float(n).map(|n| n as f32)).collect())
    } else {
        None
    };
    values.unwrap_or_else(|| {
        failures.push(format!("{prefix}_{name}_not_float_list"));
        vec![]
    })
}
fn ints(n: &Value, name: &str, prefix: &str, failures: &mut Vec<String>) -> Vec<BigInt> {
    let a = attr(n, name);
    if a.is_null() {
        return vec![];
    }
    let values = if a["intExactDecimals"].is_array()
        && rows(&a["intExactDecimals"]).len() == rows(&a["ints"]).len()
    {
        &a["intExactDecimals"]
    } else {
        &a["ints"]
    };
    let values = if list_type(a, 7) && a["ints"].is_array() {
        values
            .as_array()
            .and_then(|v| v.iter().map(exact_int).collect())
    } else {
        None
    };
    values.unwrap_or_else(|| {
        failures.push(format!("{prefix}_{name}_not_exact_int_list"));
        vec![]
    })
}
fn attributes(v: &Value, failures: &mut Vec<String>) -> Attributes {
    let n = &v["node"];
    let classifier = n["opType"] == "SVMClassifier";
    let prefix = if classifier {
        "svm_classifier"
    } else {
        "svm_regressor"
    };
    let kernel = if attr(n, "kernel_type").is_null() {
        Some("LINEAR")
    } else {
        attr_string(attr(n, "kernel_type"))
    };
    if kernel.is_none_or(|s| !["LINEAR", "POLY", "RBF", "SIGMOID"].contains(&s)) {
        failures.push(format!("{prefix}_kernel_type_invalid"));
    }
    let post = if attr(n, "post_transform").is_null() {
        Some("NONE")
    } else {
        attr_string(attr(n, "post_transform"))
    };
    if post.is_none_or(|s| !["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"].contains(&s))
    {
        failures.push(format!("{prefix}_post_transform_invalid"));
    }
    let params = floats(n, "kernel_params", prefix, failures);
    if !params.is_empty() && params.len() != 3 {
        failures.push(format!(
            "{prefix}_kernel_params_cardinality_not_zero_or_three"
        ));
    }
    let mut a = Attributes {
        classifier,
        kernel: kernel.unwrap_or("UNRESOLVED").into(),
        params,
        params_present: !attr(n, "kernel_params").is_null(),
        post: post.unwrap_or("UNRESOLVED").into(),
        ..Default::default()
    };
    if classifier {
        a.vectors = ints(n, "vectors_per_class", prefix, failures);
        a.support = floats(n, "support_vectors", prefix, failures);
        a.coefficients = floats(n, "coefficients", prefix, failures);
        a.prob_a = floats(n, "prob_a", prefix, failures);
        a.prob_b = floats(n, "prob_b", prefix, failures);
        a.rho = floats(n, "rho", prefix, failures);
        let integers = ints(n, "classlabels_ints", prefix, failures);
        let sa = attr(n, "classlabels_strings");
        let text: Option<Vec<String>> = if sa.is_null() {
            Some(vec![])
        } else if list_type(sa, 8) {
            sa["strings"]
                .as_array()
                .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
        } else {
            None
        };
        if text.is_none() {
            failures.push("svm_classifier_classlabels_strings_not_string_list".into());
        }
        a.label_kind = if !sa.is_null() { "string" } else { "int64" };
        a.labels = if !sa.is_null() {
            text.unwrap_or_default()
        } else {
            integers.iter().map(ToString::to_string).collect()
        };
        if attr(n, "classlabels_ints").is_null() == sa.is_null() {
            failures.push("svm_classifier_onnx_exactly_one_label_attribute_required".into());
        } else if a.labels.is_empty() {
            failures.push("svm_classifier_onnx_class_labels_empty".into());
        } else if a.labels.len() < 2 {
            failures.push("svm_classifier_onnx_requires_at_least_two_classes".into());
        }
    } else {
        a.n_supports = if attr(n, "n_supports").is_null() {
            Some(BigInt::zero())
        } else {
            attr_int_exact(attr(n, "n_supports"))
        };
        if a.n_supports
            .as_ref()
            .and_then(ToPrimitive::to_i64)
            .is_none_or(|n| n < 0 || n > SAFE)
        {
            failures.push("svm_regressor_onnx_n_supports_not_nonnegative_safe_integer".into());
        }
        let one = if attr(n, "one_class").is_null() {
            Some(BigInt::zero())
        } else {
            attr_int_exact(attr(n, "one_class"))
        };
        if one.is_none() {
            failures.push("svm_regressor_onnx_one_class_not_exact_int_scalar".into());
        }
        a.one_class = one.unwrap_or_default();
        a.support = floats(n, "support_vectors", prefix, failures);
        a.coefficients = floats(n, "coefficients", prefix, failures);
        a.rho = floats(n, "rho", prefix, failures);
    }
    a
}
fn mul(a: i64, b: i64) -> Option<i64> {
    a.checked_mul(b).filter(|n| *n <= SAFE)
}
#[derive(Default)]
pub(crate) struct Runtime {
    pub status: &'static str,
    pub reason: String,
    pub mode: &'static str,
    pub vectors: i64,
    pub pairs: i64,
    pub features: Option<i64>,
    pub width: Option<i64>,
    pub probability: bool,
    pub expected_coeff: Option<i64>,
    pub used_coeff: i64,
    pub unused_coeff: i64,
    pub expected_support: Option<i64>,
    pub used_support: i64,
    pub unused_support: i64,
    pub expected_rho: Option<i64>,
    pub used_rho: i64,
    pub unused_rho: i64,
    pub expected_probability: Option<i64>,
    pub used_probability: i64,
    pub unused_probability: i64,
}
fn failure_context(a: &Attributes) -> Runtime {
    let k = a.labels.len() as i64;
    let vc = if a.classifier {
        a.vectors.iter().try_fold(0i64, |s, n| {
            n.to_i64()
                .filter(|n| *n >= 0 && *n <= SAFE)
                .and_then(|n| s.checked_add(n))
                .filter(|n| *n <= SAFE)
        })
    } else {
        a.n_supports
            .as_ref()
            .and_then(ToPrimitive::to_i64)
            .filter(|n| *n >= 0 && *n <= SAFE)
    };
    let mode = if vc.is_none() {
        "unresolved"
    } else if vc.unwrap() > 0 {
        "svc"
    } else {
        "linear"
    };
    let features = if mode == "svc" {
        vc.filter(|n| {
            *n > 0 && a.support.len() as i64 % n == 0 && (a.classifier || !a.support.is_empty())
        })
        .map(|n| a.support.len() as i64 / n)
    } else if mode == "linear" {
        if a.classifier {
            if k > 0 {
                Some(a.coefficients.len() as i64 / k)
            } else {
                None
            }
        } else {
            Some(a.coefficients.len() as i64)
        }
    } else {
        None
    };
    let pairs = if a.classifier && k >= 2 {
        mul(k, k - 1).map(|n| n / 2).unwrap_or(0)
    } else {
        0
    };
    Runtime {
        status: "fail",
        mode,
        vectors: vc.unwrap_or(0),
        pairs,
        features,
        probability: !a.prob_a.is_empty(),
        expected_coeff: if mode == "svc" {
            if a.classifier {
                vc.filter(|_| k >= 2).and_then(|n| mul(k - 1, n))
            } else {
                vc
            }
        } else if mode == "linear" {
            if a.classifier {
                features.and_then(|n| mul(k, n))
            } else {
                features
            }
        } else {
            None
        },
        expected_support: if mode == "linear" {
            Some(0)
        } else {
            features.zip(vc).and_then(|(f, v)| mul(f, v))
        },
        expected_rho: if !a.classifier {
            Some(1)
        } else if mode == "svc" {
            Some(pairs)
        } else if mode == "linear" {
            Some(1)
        } else {
            None
        },
        expected_probability: if a.classifier {
            Some(if mode == "svc" && !a.prob_a.is_empty() {
                pairs
            } else {
                0
            })
        } else {
            None
        },
        ..Default::default()
    }
}
fn runtime(a: &Attributes, input: &Input) -> Runtime {
    let mut r = failure_context(a);
    macro_rules! fail {
        ($reason:expr) => {{
            r.reason = $reason.into();
            return r;
        }};
    }
    if !a.params_present {
        fail!("kernel_params_attribute_missing_in_pinned_ort_constructor");
    }
    if a.classifier {
        if a.vectors
            .iter()
            .any(|n| n.to_i64().is_none_or(|n| n < 0 || n > SAFE))
        {
            fail!("vectors_per_class_must_be_nonnegative_safe_integers");
        }
        if r.mode == "unresolved" {
            fail!("vectors_per_class_sum_overflow");
        }
        if a.labels.len() < 2 || a.labels.len() >= 65536 {
            fail!("class_count_outside_pinned_ort_range_2_to_65535");
        }
    }
    if a.coefficients.is_empty() {
        fail!("coefficients_empty");
    }
    if a.rho.is_empty() {
        fail!("rho_empty");
    }
    if a.classifier && a.prob_a.len() != a.prob_b.len() {
        fail!("prob_a_prob_b_size_mismatch");
    }
    if !a.classifier && r.mode == "unresolved" {
        fail!("n_supports_not_nonnegative_safe_integer");
    }
    if r.mode == "svc" {
        if a.classifier {
            if a.vectors.len() != a.labels.len() {
                fail!("vectors_per_class_size_must_equal_class_count");
            }
            if a.support.len() as i64 % r.vectors != 0 {
                fail!("support_vectors_size_must_be_divisible_by_vector_count");
            }
            if r.features.is_none_or(|n| n < 1) {
                fail!("support_vector_feature_count_must_be_positive");
            }
            if r.expected_coeff
                .is_some_and(|n| (a.coefficients.len() as i64) < n)
            {
                fail!("coefficients_smaller_than_svc_layout");
            }
            if r.expected_rho.is_some_and(|n| (a.rho.len() as i64) < n) {
                fail!("rho_smaller_than_pairwise_classifier_count");
            }
            if !a.prob_a.is_empty() && (a.prob_a.len() as i64) < r.pairs {
                fail!("probability_parameters_smaller_than_pairwise_classifier_count");
            }
        } else {
            if (a.coefficients.len() as i64) < r.vectors {
                fail!("coefficients_smaller_than_n_supports");
            }
            if a.support.is_empty() || a.support.len() as i64 % r.vectors != 0 {
                fail!("support_vectors_must_be_nonempty_multiple_of_n_supports");
            }
        }
    } else if a.classifier && r.features.is_none_or(|n| n < 1) {
        fail!("linear_feature_count_must_be_positive");
    }
    let known = safe(&input.features).filter(|n| *n >= 0);
    if known.is_some() && known != r.features {
        fail!("artifact_feature_count_differs_from_pinned_ort_parameter_layout");
    }
    r.status = if known.is_some() { "pass" } else { "partial" };
    r.reason = if known.is_some() {
        ""
    } else {
        "artifact_feature_count_unresolved"
    }
    .into();
    r.used_coeff = r
        .expected_coeff
        .unwrap_or(0)
        .min(a.coefficients.len() as i64);
    r.unused_coeff = a.coefficients.len() as i64 - r.used_coeff;
    r.used_support = r.expected_support.unwrap_or(0).min(a.support.len() as i64);
    r.unused_support = a.support.len() as i64 - r.used_support;
    r.used_rho = r.expected_rho.unwrap_or(0).min(a.rho.len() as i64);
    r.unused_rho = a.rho.len() as i64 - r.used_rho;
    if a.classifier {
        r.width = Some(if r.mode == "svc" && !r.probability {
            if a.labels.len() > 2 {
                r.pairs
            } else {
                2
            }
        } else {
            a.labels.len() as i64
        });
        r.used_probability = r.expected_probability.unwrap_or(0) * 2;
        r.unused_probability = (a.prob_a.len() + a.prob_b.len()) as i64 - r.used_probability;
    }
    r
}
fn source(n: &Value, name: &str, default: &str) -> &'static str {
    if !attr(n, name).is_null() {
        "explicit_attribute"
    } else {
        match default {
            "kernel" => "onnx_schema_default_LINEAR",
            "params" => "omitted_schema_optional_but_pinned_ort_required",
            "one" => "onnx_schema_default_0",
            _ => "onnx_schema_default_NONE",
        }
    }
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let input = read_input(v);
    let (mut failures, mut reasons) = (input.failures.clone(), input.reasons.clone());
    let a = attributes(v, &mut failures);
    let initial = failures.first().cloned().unwrap_or_default();
    let r = runtime(&a, &input);
    if r.status == "fail" {
        failures.push(r.reason.clone());
    } else if r.status == "partial" {
        reasons.push(r.reason.clone());
    }
    let prefix = if a.classifier {
        "svm_classifier"
    } else {
        "svm_regressor"
    };
    let gap = !a.classifier && ["FLOAT64", "INT32", "INT64"].contains(&input.dtype.as_str());
    if gap {
        reasons.push("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel".into());
    }
    let schema_width = if a.classifier {
        if a.labels.is_empty() {
            None
        } else {
            Some(a.labels.len() as i64)
        }
    } else {
        Some(1)
    };
    let mismatch = a.classifier
        && r.mode == "svc"
        && !r.probability
        && a.labels.len() > 3
        && r.width != schema_width;
    if mismatch {
        failures.push("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch".into());
    }
    let declared = matches!(input.rank, Some(1) | Some(2)) && schema_width.is_some();
    let batch = if input.rank == Some(1) {
        json!(1)
    } else {
        input.batch.clone()
    };
    let labels_shape = if declared {
        vec![batch.clone()]
    } else {
        vec![]
    };
    let schema_shape = if declared {
        vec![batch, json!(schema_width)]
    } else {
        vec![]
    };
    let shape = if mismatch {
        vec![]
    } else {
        schema_shape.clone()
    };
    let lt = tensor(
        v,
        if a.label_kind == "string" {
            "STRING"
        } else {
            "INT64"
        },
        &labels_shape,
        declared,
    );
    let ot = tensor(v, "FLOAT32", &schema_shape, declared && !mismatch);
    let nonfinite = a
        .params
        .iter()
        .chain(&a.support)
        .chain(&a.coefficients)
        .chain(&a.prob_a)
        .chain(&a.prob_b)
        .chain(&a.rho)
        .filter(|n| !n.is_finite())
        .count();
    let reference = crate::portable_onnx_ml_svm_eval::run(v, &input, &a, &r, gap);
    let dup = a.labels.len() - unique(a.labels.clone()).len();
    let mut risks = Vec::<String>::new();
    if r.status == "fail" {
        risks.push(format!("{prefix}_pinned_ort_runtime_contract_invalid"));
    }
    if a.classifier {
        if mismatch {
            risks.push("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch".into());
        }
        if dup > 0 {
            risks.push("svm_classifier_duplicate_labels_ambiguous_output_semantics".into());
        }
    } else {
        if gap {
            risks.push("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel".into());
        }
        if a.post != "NONE" {
            risks.push("svm_regressor_post_transform_ignored_by_pinned_ort".into());
        }
        if a.one_class != BigInt::zero() && a.one_class != BigInt::from(1) {
            risks.push("svm_regressor_noncanonical_one_class_flag".into());
        }
    }
    if r.unused_coeff > 0 || r.unused_support > 0 || r.unused_rho > 0 || r.unused_probability > 0 {
        risks.push(format!(
            "{prefix}_serialized_parameters_ignored_by_pinned_ort"
        ));
    }
    if r.mode == "linear" && a.kernel != "LINEAR" {
        risks.push(format!("{prefix}_linear_mode_forces_linear_kernel"));
    }
    if a.classifier {
        if r.mode == "svc" && r.probability && a.post != "NONE" {
            risks
                .push("svm_classifier_probability_scores_receive_additional_post_transform".into());
        }
        if r.mode == "svc" && !r.probability && a.labels.len() == 2 {
            if a.post == "PROBIT" {
                risks.push("svm_classifier_binary_probit_second_score_unwritten".into());
            } else if ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].contains(&a.post.as_str()) {
                risks.push("svm_classifier_binary_post_transform_uses_complement_expansion".into());
            }
        }
    }
    if nonfinite > 0 || reference["nonFiniteScoreCount"].as_u64().unwrap_or(0) > 0 {
        risks.push(format!("{prefix}_non_finite_parameter_or_reference_score"));
    }
    if reference["decisionBoundaryCount"].as_u64().unwrap_or(0) > 0 {
        risks.push(format!("{prefix}_reference_decision_boundary"));
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
    let mut row = templates()["svm_base"].clone();
    row.as_object_mut().unwrap().extend(
        templates()[if a.classifier {
            "svm_classifier_extra"
        } else {
            "svm_regressor_extra"
        }]
        .as_object()
        .unwrap()
        .clone(),
    );
    let fields = json!({"scope":v["scope"],"node_index":v["node_index"],"op_name":n["opType"],"contract_kind":prefix,"imported_opset":v["imported_opset"],"status":status,"input_name":super::portable_onnx_semantics::input(v,0)["name"].as_str().unwrap_or(""),"output_name":n["outputs"][if a.classifier{1}else{0}].as_str().unwrap_or(""),"input_dtype":input.dtype,"input_kind":input.ty["kind"].as_str().unwrap_or("unresolved"),"input_rank":input.rank,"input_shape":input.shape,"exact_batch_count":input.batch,"exact_feature_count":input.features,"canonical_output_type":canonical(&tensor(v,"FLOAT32",&shape,declared&&!mismatch)),"output_dtype":"FLOAT32","exact_output_rank":if declared&&!mismatch{Some(shape.len())}else{None},"exact_output_shape":shape,"exact_dense_output_element_count":if declared&&!mismatch{count(&shape)}else{None},"svm_onnx_contract_status":if initial.is_empty(){"pass"}else{"fail"},"svm_onnx_contract_reason":initial,"svm_pinned_ort_contract_status":if gap{"fail"}else{r.status},"svm_pinned_ort_contract_reason":if gap{"schema_dtype_missing_pinned_ort_cpu_kernel"}else{&r.reason},"svm_mode":r.mode,"svm_kernel_type":a.kernel,"svm_kernel_type_source":source(n,"kernel_type","kernel"),"svm_kernel_params":a.params.iter().map(|n|ft(*n as f64)).collect::<Vec<_>>(),"svm_kernel_params_source":source(n,"kernel_params","params"),"svm_post_transform":a.post,"svm_post_transform_source":source(n,"post_transform","post"),"reason_codes":codes,"risk_codes":risks});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let fields = json!({"svm_vector_count":r.vectors,"svm_pairwise_classifier_count":r.pairs,"svm_schema_score_width":schema_width,"svm_pinned_ort_score_width":if a.classifier{r.width}else{Some(1)},"svm_schema_runtime_score_width_mismatch":mismatch,"svm_support_vector_value_count":a.support.len(),"svm_expected_support_vector_value_count":r.expected_support,"svm_used_support_vector_value_count":r.used_support,"svm_unused_support_vector_value_count":r.unused_support,"svm_coefficient_count":a.coefficients.len(),"svm_expected_coefficient_count":r.expected_coeff,"svm_used_coefficient_count":r.used_coeff,"svm_unused_coefficient_count":r.unused_coeff,"svm_rho_count":a.rho.len(),"svm_expected_rho_count":r.expected_rho,"svm_used_rho_count":r.used_rho,"svm_unused_rho_count":r.unused_rho,"svm_prob_a_count":a.prob_a.len(),"svm_prob_b_count":a.prob_b.len(),"svm_probability_enabled":r.probability,"svm_expected_probability_parameter_count_per_array":if a.classifier{r.expected_probability}else{Some(0)},"svm_used_probability_parameter_count":r.used_probability,"svm_unused_probability_parameter_count":r.unused_probability,"svm_non_finite_parameter_count":nonfinite,"svm_linear_mode_forced_kernel":r.mode=="linear"&&a.kernel!="LINEAR"});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    for (dst, src) in [
        ("assessment_status", "status"),
        ("input_value_count", "inputValueCount"),
        ("raw_score_count", "rawScoreCount"),
        ("non_finite_score_count", "nonFiniteScoreCount"),
        ("decision_boundary_count", "decisionBoundaryCount"),
        ("raw_score_preview", "rawScorePreview"),
        ("output_score_preview", "outputScorePreview"),
        ("label_preview", "labelPreview"),
    ] {
        row[format!("svm_reference_{dst}")] = reference[src].clone();
    }
    if a.classifier {
        let fields = json!({"output_names":n["outputs"],"canonical_output_types":[canonical(&lt),canonical(&ot)],"canonical_output_shapes":[labels_shape,shape],"svm_class_label_kind":a.label_kind,"svm_class_label_count":a.labels.len(),"svm_class_label_values":a.labels,"svm_duplicate_label_count":dup,"svm_vectors_per_class":a.vectors.iter().map(ToString::to_string).collect::<Vec<_>>()});
        row.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
    } else {
        row["svm_one_class_value"] = json!(a.one_class.to_string());
        row["svm_one_class_source"] = json!(source(n, "one_class", "one"));
        row["svm_n_supports"] = json!(a.n_supports.as_ref().and_then(ToPrimitive::to_f64));
    }
    let mut outputs = vec![];
    if status != "fail" && declared && !mismatch {
        if a.classifier && truth(&n["outputs"][0]) {
            outputs.push(json!([n["outputs"][0], descriptor(&lt, Value::Null)]));
        }
        let i = if a.classifier { 1 } else { 0 };
        if truth(&n["outputs"][i]) {
            outputs.push(json!([n["outputs"][i], descriptor(&ot, Value::Null)]));
        }
    }
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
