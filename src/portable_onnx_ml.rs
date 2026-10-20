//! Source-pinned traditional ML numerics and contracts. No host numeric fallback.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::{FromPrimitive, Signed, ToPrimitive};
use serde_json::{json, Value};
use std::sync::OnceLock;
pub(crate) fn templates() -> &'static Value {
    static V: OnceLock<Value> = OnceLock::new();
    V.get_or_init(|| serde_json::from_str(include_str!("onnx_ml_row_templates.json")).unwrap())
}
pub(crate) fn float(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() {
        return Some(n);
    }
    match v.as_str()? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        s => s.parse().ok(),
    }
}
pub(crate) fn ft(n: f64) -> String {
    if n.is_nan() {
        "NaN".into()
    } else if n == f64::INFINITY {
        "Infinity".into()
    } else if n == f64::NEG_INFINITY {
        "-Infinity".into()
    } else if n == 0. && n.is_sign_negative() {
        "-0".into()
    } else {
        ryu_js::Buffer::new().format_finite(n).into()
    }
}
#[derive(Clone)]
pub(crate) enum Numeric {
    Int(BigInt),
    Float(f64),
}
impl Numeric {
    pub(crate) fn text(&self) -> String {
        match self {
            Self::Int(n) => n.to_string(),
            Self::Float(n) => ft(*n),
        }
    }
    pub(crate) fn f32(&self) -> f32 {
        match self {
            Self::Int(n) => n.to_f32().unwrap_or(if n.is_negative() {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }),
            Self::Float(n) => *n as f32,
        }
    }
    pub(crate) fn f64(&self) -> f64 {
        match self {
            Self::Int(n) => n.to_f64().unwrap_or(if n.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }),
            Self::Float(n) => *n,
        }
    }
    fn changed(&self) -> bool {
        match self {
            Self::Int(n) => BigInt::from_f32(self.f32()).as_ref() != Some(n),
            Self::Float(n) => self.f32() as f64 != *n,
        }
    }
    fn negative(&self) -> bool {
        match self {
            Self::Int(n) => n.is_negative(),
            Self::Float(n) => *n < 0.,
        }
    }
    fn square(&self, dt: &str) -> f32 {
        match self {
            Self::Int(n) => (n * n).to_f32().unwrap_or(f32::INFINITY),
            Self::Float(n) => {
                let n = if dt == "FLOAT32" {
                    *n as f32 as f64
                } else {
                    *n
                };
                (n * n) as f32
            }
        }
    }
    fn abs_f32(&self) -> f32 {
        match self {
            Self::Int(n) => n.abs().to_f32().unwrap_or(f32::INFINITY),
            Self::Float(n) => n.abs() as f32,
        }
    }
}
pub(crate) fn source(t: &Value, dt: &str) -> Option<Vec<Numeric>> {
    if dt == "INT64" && t["initializerIntegerValuesExactComplete"] == true {
        return t["initializerIntegerValuesExactDecimals"]
            .as_array()?
            .iter()
            .map(|v| {
                let s = v.as_str()?;
                if s.len() > 32768 {
                    return None;
                }
                Some(Numeric::Int(s.parse().ok()?))
            })
            .collect();
    }
    if ["FLOAT32", "FLOAT64", "INT32"].contains(&dt) && t["staticValuesComplete"] == true {
        let vals = if t["staticValuesText"].is_array() {
            &t["staticValuesText"]
        } else {
            &t["staticValues"]
        };
        return vals
            .as_array()?
            .iter()
            .map(|v| float(v).map(Numeric::Float))
            .collect();
    }
    None
}
pub(crate) fn count(s: &[Value]) -> Option<i64> {
    let ds = s.iter().map(safe).collect::<Option<Vec<_>>>()?;
    if ds.iter().any(|n| *n < 0) {
        return None;
    }
    if ds.contains(&0) {
        return Some(0);
    }
    ds.iter()
        .try_fold(1i64, |n, d| n.checked_mul(*d).filter(|n| *n <= SAFE))
}
pub(crate) fn attr_float(a: &Value) -> Option<f64> {
    if a["type"] != 1 || rows(&a["valueTypesPresent"]).len() != 1 {
        return None;
    }
    float(if a.get("fText").is_some() {
        &a["fText"]
    } else {
        &a["f"]
    })
}
pub(crate) fn attr_floats(a: &Value) -> Option<Vec<f32>> {
    if a["type"] != 6 || a["valueTypesPresent"] != json!([6]) || !a["floats"].is_array() {
        return None;
    }
    let vals = if a["floatsText"].is_array() {
        &a["floatsText"]
    } else {
        &a["floats"]
    };
    rows(vals)
        .iter()
        .map(|v| float(v).map(|n| n as f32))
        .collect()
}
pub(crate) fn attr_string(a: &Value) -> Option<&str> {
    if a["type"] == 3 && rows(&a["valueTypesPresent"]).len() == 1 {
        a["s"].as_str()
    } else {
        None
    }
}
pub(crate) fn unique(v: Vec<String>) -> Vec<String> {
    let mut out = vec![];
    for s in v {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}
fn unresolved(t: &Value) -> Value {
    json!({"status":if t["role"]=="initializer"{t["staticValuesStatus"].as_str().filter(|s|!s.is_empty()).unwrap_or("not_assessed_initializer_values")}else{"not_assessed_runtime_values"},"values":null,"inputValueCount":null,"batchCount":null,"rowWidth":null,"divisorPreview":[],"zeroDivisorRowCount":null,"negativeMaxDivisorRowCount":null,"integerFloat32RoundingCount":null,"overflowValueCount":null,"nonFiniteOutputCount":null,"signedZeroOutputCount":null,"outputPreview":[],"aboveCount":null,"atOrBelowCount":null,"equalCount":null})
}
fn normalize(src: &[Numeric], dt: &str, s: &[Value], mode: &str) -> Value {
    let batches = if s.len() == 1 {
        1
    } else {
        safe(&s[0]).unwrap() as u64
    };
    let width = safe(&s[if s.len() == 1 { 0 } else { 1 }]).unwrap() as u64;
    let integer = matches!(dt, "INT32" | "INT64");
    let overflow = if integer && mode.starts_with('L') {
        src.iter()
            .filter(|n| {
                if mode == "L1" {
                    match n {
                        Numeric::Int(n) => n == &BigInt::from(i64::MIN),
                        Numeric::Float(n) => *n == i32::MIN as f64,
                    }
                } else {
                    match n {
                        Numeric::Int(n) => n.abs() > BigInt::from(3_037_000_499i64),
                        Numeric::Float(n) => n.abs() > 46_340.,
                    }
                }
            })
            .count()
    } else {
        0
    };
    let rounding = if integer {
        src.iter().filter(|n| n.changed()).count()
    } else {
        0
    };
    if overflow > 0 {
        return json!({"status":"not_assessed_ort_signed_integer_overflow","values":null,"inputValueCount":src.len(),"batchCount":batches,"rowWidth":width,"divisorPreview":[],"zeroDivisorRowCount":null,"negativeMaxDivisorRowCount":null,"integerFloat32RoundingCount":rounding,"overflowValueCount":overflow,"nonFiniteOutputCount":null,"outputPreview":[],"signedZeroOutputCount":null});
    }
    if width == 0 {
        return json!({"status":"assessed_pinned_ort_float32","values":[],"inputValueCount":0,"batchCount":batches,"rowWidth":0,"divisorPreview":vec![if mode=="MAX"{ft(f32::MIN as f64)}else{"0".into()};batches.min(8) as usize],"zeroDivisorRowCount":if mode=="MAX"{0}else{batches},"negativeMaxDivisorRowCount":0,"integerFloat32RoundingCount":0,"overflowValueCount":0,"nonFiniteOutputCount":0,"signedZeroOutputCount":0,"outputPreview":[]});
    }
    let materialize = src.len() <= 1_000_000;
    let (mut values, mut preview, mut divisors) = (vec![], vec![], vec![]);
    let (mut zeros, mut negative, mut nonfinite, mut signedzero) = (0, 0, 0, 0);
    for b in 0..batches {
        let row = &src[(b * width) as usize..((b + 1) * width) as usize];
        let mut divisor = if mode == "MAX" { f32::MIN } else { 0. };
        for n in row {
            if mode == "MAX" {
                let c = n.f32();
                if c > divisor {
                    divisor = c;
                }
            } else {
                let term = if mode == "L1" {
                    n.abs_f32()
                } else {
                    n.square(dt)
                };
                divisor += term;
            }
        }
        if divisors.len() < 8 {
            divisors.push(ft(divisor as f64));
        }
        if divisor == 0. {
            zeros += 1;
        }
        if mode == "MAX" && width > 0 && divisor < 0. {
            negative += 1;
        }
        for n in row {
            let y = if divisor == 0. {
                n.f32()
            } else if mode == "L2" {
                let q = n.square(dt) / divisor;
                let m = (q as f64).sqrt() as f32;
                if n.negative() {
                    -m
                } else {
                    m
                }
            } else {
                n.f32() / divisor
            };
            if materialize {
                values.push(json!(y as f64));
            }
            if preview.len() < 8 {
                preview.push(ft(y as f64));
            }
            if !y.is_finite() {
                nonfinite += 1;
            }
            if y == 0. && y.is_sign_negative() {
                signedzero += 1;
            }
        }
    }
    json!({"status":if materialize&&nonfinite==0{"assessed_pinned_ort_float32"}else if nonfinite>0{"assessed_non_finite_output_not_propagated"}else{"assessed_counts_output_not_materialized_limit"},"values":if materialize&&nonfinite==0{json!(values)}else{Value::Null},"inputValueCount":src.len(),"batchCount":batches,"rowWidth":width,"divisorPreview":divisors,"zeroDivisorRowCount":zeros,"negativeMaxDivisorRowCount":negative,"integerFloat32RoundingCount":rounding,"overflowValueCount":0,"nonFiniteOutputCount":nonfinite,"signedZeroOutputCount":signedzero,"outputPreview":preview})
}
fn scale(src: &[Numeric], dt: &str, scales: &[f32], offsets: &[f32], stride: usize) -> Value {
    let materialize = src.len() <= 1_000_000;
    let (mut values, mut preview) = (vec![], vec![]);
    let (mut rounding, mut nonfinite, mut signedzero) = (0, 0, 0);
    for (i, n) in src.iter().enumerate() {
        let j = if scales.len() == 1 { 0 } else { i % stride };
        let y = if dt == "FLOAT64" {
            ((n.f64() - offsets[j] as f64) * scales[j] as f64) as f32
        } else {
            if matches!(dt, "INT32" | "INT64") && n.changed() {
                rounding += 1;
            }
            (n.f32() - offsets[j]) * scales[j]
        };
        if materialize {
            values.push(json!(y as f64));
        }
        if preview.len() < 8 {
            preview.push(ft(y as f64));
        }
        if !y.is_finite() {
            nonfinite += 1;
        }
        if y == 0. && y.is_sign_negative() {
            signedzero += 1;
        }
    }
    json!({"status":if materialize&&nonfinite==0{"assessed_pinned_ort_float32"}else if nonfinite>0{"assessed_non_finite_output_not_propagated"}else{"assessed_counts_output_not_materialized_limit"},"values":if materialize&&nonfinite==0{json!(values)}else{Value::Null},"inputValueCount":src.len(),"integerFloat32RoundingCount":rounding,"nonFiniteOutputCount":nonfinite,"signedZeroOutputCount":signedzero,"outputPreview":preview})
}
fn binarize(src: &[Numeric], threshold: f64) -> Value {
    let (mut above, mut equal) = (0, 0);
    let vals = src
        .iter()
        .map(|v| {
            let (a, e) = match v {
                Numeric::Float(n) => (*n > threshold, *n == threshold),
                Numeric::Int(n) => {
                    let above = if threshold.is_nan() || threshold == f64::INFINITY {
                        false
                    } else if threshold == f64::NEG_INFINITY {
                        true
                    } else {
                        n > &BigInt::from_f64(threshold.floor()).unwrap()
                    };
                    let equal = threshold.is_finite()
                        && threshold.fract() == 0.
                        && BigInt::from_f64(threshold).as_ref() == Some(n);
                    (above, equal)
                }
            };
            if a {
                above += 1;
            }
            if e {
                equal += 1;
            }
            if a {
                1
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    json!({"status":"assessed_exact","values":vals,"inputValueCount":src.len(),"aboveCount":above,"atOrBelowCount":src.len()-above,"equalCount":equal})
}
fn main(v: &Value) -> R<Value> {
    let n = &v["node"];
    let op = string(&n["opType"]);
    let prefix = op.to_ascii_lowercase();
    let t = input(v, 0);
    let it = ty(v, t);
    let kind = it["kind"].as_str().unwrap_or("unresolved");
    let dt = if kind == "tensor" {
        it["dtype"]
            .as_str()
            .or(it["elementTypeName"].as_str())
            .unwrap_or("UNKNOWN")
    } else {
        "UNKNOWN"
    };
    let (mut reasons, mut failures, mut risks) = (
        Vec::<String>::new(),
        Vec::<String>::new(),
        Vec::<String>::new(),
    );
    if it.is_null() {
        reasons.push(format!("{prefix}_input_type_unresolved"));
    } else if kind != "tensor" {
        failures.push(format!("{prefix}_input_not_tensor:{kind}"));
    }
    if dt == "UNKNOWN" {
        reasons.push(format!("{prefix}_input_dtype_unresolved"));
    } else if !["FLOAT32", "FLOAT64", "INT32", "INT64"].contains(&dt) {
        failures.push(format!("{prefix}_input_dtype_not_supported:{dt}"));
    }
    let declared = kind == "tensor" && it["shapeDeclared"] == true;
    let s = if declared {
        rows(&it["shape"]).to_vec()
    } else {
        vec![]
    };
    let rank = if declared { Some(s.len()) } else { None };
    let concrete = declared && s.iter().all(|v| safe(v).is_some_and(|n| n >= 0));
    let elements = if concrete { count(&s) } else { None };
    if concrete && elements.is_none() {
        reasons.push(format!("{prefix}_output_element_count_overflow"));
    }
    let od = if matches!(op, "Binarizer" | "Imputer") {
        dt
    } else {
        "FLOAT32"
    };
    let ot = if kind == "tensor" && (op != "Binarizer" || dt != "UNKNOWN") {
        let mut o = tensor_type(od, &dimensions(&json!({"shape":s})), &v["tensor_type_ids"]);
        if !declared {
            o["shapeDeclared"] = json!(false);
            o["shapeFieldCount"] = json!(0);
        }
        o
    } else {
        Value::Null
    };
    let mut patch = if !ot.is_null() {
        descriptor(&ot, Value::Null)
    } else {
        Value::Null
    };
    let mut row = templates()[op].clone();
    for (k, value) in [
        ("scope", v["scope"].clone()),
        ("node_index", v["node_index"].clone()),
        ("imported_opset", v["imported_opset"].clone()),
        ("input_name", n["inputs"][0].clone()),
        ("output_name", n["outputs"][0].clone()),
        ("input_dtype", json!(dt)),
        ("input_kind", json!(kind)),
        ("input_rank", json!(rank)),
        ("input_shape", json!(s)),
        ("exact_output_rank", json!(rank)),
        ("exact_output_shape", json!(s)),
        ("exact_dense_output_element_count", json!(elements)),
        (
            "canonical_output_type",
            json!(if ot.is_null() {
                "unresolved".into()
            } else {
                canonical(&ot)
            }),
        ),
        ("output_dtype", json!(od)),
    ] {
        row[k] = value;
    }
    let src = source(t, dt);
    let mut measured = unresolved(t);
    let mut can = true;
    if op == "Normalizer" {
        let a = attr(n, "norm");
        let explicit = attr_string(a);
        if !a.is_null() && explicit.is_none() {
            failures.push("normalizer_norm_not_string_scalar".into());
        }
        let mode = explicit.unwrap_or("MAX");
        let valid = ["MAX", "L1", "L2"].contains(&mode);
        if explicit.is_some() && !valid {
            failures.push(format!("normalizer_norm_invalid:{mode}"));
        }
        if !declared {
            reasons.push("normalizer_input_shape_unresolved".into());
        } else if !matches!(rank, Some(1) | Some(2)) {
            failures.push(format!(
                "normalizer_input_rank_not_1_or_2:{}",
                rank.unwrap()
            ));
        } else if !concrete {
            reasons.push("normalizer_dynamic_shape_preserved".into());
        }
        let batch = if rank == Some(1) {
            Some(1)
        } else if rank == Some(2) {
            safe(&s[0]).filter(|n| *n >= 0)
        } else {
            None
        };
        let width = if rank == Some(1) {
            safe(&s[0])
        } else if rank == Some(2) {
            safe(&s[1])
        } else {
            None
        }
        .filter(|n| *n >= 0);
        if let (Some(src), Some(total)) = (src.as_ref(), elements) {
            if matches!(rank, Some(1) | Some(2)) && valid {
                if src.len() as i64 != total {
                    failures.push(format!(
                        "normalizer_static_value_count_mismatch:{}:{total}",
                        src.len()
                    ));
                } else {
                    measured = normalize(src, dt, &s, mode);
                    if measured["overflowValueCount"].as_u64().unwrap_or(0) > 0 {
                        reasons
                            .push("normalizer_static_output_not_assessed_signed_overflow".into());
                    }
                }
            }
        }
        for (k, code) in [
            (
                "overflowValueCount",
                "normalizer_signed_integer_abs_or_square_overflow",
            ),
            (
                "negativeMaxDivisorRowCount",
                "normalizer_negative_signed_max_divisor",
            ),
            (
                "integerFloat32RoundingCount",
                "normalizer_integer_to_float32_precision_loss",
            ),
            (
                "nonFiniteOutputCount",
                "normalizer_non_finite_float32_projection",
            ),
        ] {
            if measured[k].as_u64().unwrap_or(0) > 0 {
                risks.push(code.into());
            }
        }
        if t["staticValuesStatus"] == "not_assessed_non_finite_or_unsafe_value"
            && ["FLOAT32", "FLOAT64"].contains(&dt)
        {
            risks.push("normalizer_static_input_contains_non_finite_or_unsafe_value".into());
        }
        row["exact_batch_count"] = json!(batch);
        row["exact_feature_count"] = json!(width);
        row["normalizer_mode"] = json!(mode);
        row["normalizer_mode_source"] = json!(if a.is_null() {
            "onnx_schema_default_MAX"
        } else {
            "explicit_attribute"
        });
        row["normalizer_batch_count"] = json!(batch);
        row["normalizer_row_width"] = json!(width);
        row["normalizer_divisor_kind"] = json!(match mode {
            "MAX" => "signed_max",
            "L1" => "absolute_sum",
            _ => "square_sum_before_sqrt",
        });
        for (dst, src) in [
            ("divisor_preview", "divisorPreview"),
            ("zero_divisor_row_count", "zeroDivisorRowCount"),
            (
                "negative_max_divisor_row_count",
                "negativeMaxDivisorRowCount",
            ),
            ("signed_overflow_value_count", "overflowValueCount"),
        ] {
            row[format!("normalizer_{dst}")] = measured[src].clone();
        }
    } else if op == "Scaler" {
        let (a, b) = (attr(n, "scale"), attr(n, "offset"));
        let sv = if a.is_null() {
            Some(vec![])
        } else {
            attr_floats(a)
        };
        let ov = if b.is_null() {
            Some(vec![])
        } else {
            attr_floats(b)
        };
        if !a.is_null() && sv.is_none() {
            failures.push("scaler_scale_not_float_list".into());
        }
        if !b.is_null() && ov.is_none() {
            failures.push("scaler_offset_not_float_list".into());
        }
        let (scales, offsets) = (sv.unwrap_or_default(), ov.unwrap_or_default());
        let stride = if declared && !s.is_empty() {
            safe(&s[if s.len() == 1 { 0 } else { 1 }]).filter(|n| *n >= 0)
        } else {
            None
        };
        let (state, why, mode) = if a.is_null() || b.is_null() {
            (
                "fail",
                "scaler_pinned_ort_requires_explicit_scale_and_offset",
                "invalid",
            )
        } else if scales.is_empty() {
            ("fail", "scaler_pinned_ort_rejects_empty_scale", "invalid")
        } else if scales.len() != offsets.len() {
            (
                "fail",
                "scaler_pinned_ort_scale_offset_length_mismatch",
                "invalid",
            )
        } else if !declared {
            (
                "not_assessed",
                "scaler_input_rank_unresolved",
                if scales.len() == 1 {
                    "scalar"
                } else {
                    "unresolved"
                },
            )
        } else if s.is_empty() {
            (
                "fail",
                "scaler_pinned_ort_rejects_rank_zero_input",
                "invalid",
            )
        } else if scales.len() == 1 {
            ("pass", "scaler_scalar_parameters", "scalar")
        } else if stride.is_none() {
            (
                "not_assessed",
                "scaler_feature_stride_unresolved",
                "per_feature",
            )
        } else if Some(scales.len() as i64) != stride {
            (
                "fail",
                "scaler_pinned_ort_parameter_length_not_one_or_feature_stride",
                "invalid",
            )
        } else {
            ("pass", "scaler_per_feature_parameters", "per_feature")
        };
        if state != "pass" {
            reasons.push(why.into());
        }
        if declared && !concrete {
            reasons.push("scaler_dynamic_shape_preserved".into());
        }
        if state == "pass" {
            if let (Some(src), Some(total)) = (src.as_ref(), elements) {
                if src.len() as i64 != total {
                    failures.push(format!(
                        "scaler_static_value_count_mismatch:{}:{total}",
                        src.len()
                    ));
                } else {
                    measured = scale(src, dt, &scales, &offsets, stride.unwrap_or(0) as usize);
                }
            }
        }
        can = state != "fail";
        let nonfinite = scales
            .iter()
            .chain(&offsets)
            .filter(|v| !v.is_finite())
            .count();
        if state == "fail" {
            risks.push("scaler_pinned_ort_attribute_or_shape_contract_invalid".into());
        }
        if measured["integerFloat32RoundingCount"]
            .as_u64()
            .unwrap_or(0)
            > 0
        {
            risks.push("scaler_integer_to_float32_precision_loss".into());
        }
        if nonfinite > 0
            || measured["nonFiniteOutputCount"].as_u64().unwrap_or(0) > 0
            || t["staticValuesStatus"] == "not_assessed_non_finite_or_unsafe_value"
        {
            risks.push("scaler_non_finite_parameter_input_or_output".into());
        }
        row["exact_batch_count"] = json!(if rank == Some(1) {
            Some(1)
        } else if rank.is_some_and(|r| r > 1) {
            safe(&s[0]).filter(|n| *n >= 0)
        } else {
            None
        });
        row["exact_feature_count"] = json!(stride);
        for (k, value) in [
            ("parameter_contract_status", json!(state)),
            ("parameter_contract_reason", json!(why)),
            ("parameter_mode", json!(mode)),
            ("feature_stride", json!(stride)),
            ("scale_count", json!(scales.len())),
            ("offset_count", json!(offsets.len())),
            (
                "scale_values",
                json!(scales.iter().map(|n| ft(*n as f64)).collect::<Vec<_>>()),
            ),
            (
                "offset_values",
                json!(offsets.iter().map(|n| ft(*n as f64)).collect::<Vec<_>>()),
            ),
            (
                "zero_scale_count",
                json!(scales.iter().filter(|v| **v == 0.).count()),
            ),
            ("non_finite_parameter_count", json!(nonfinite)),
        ] {
            row[format!("scaler_{k}")] = value;
        }
    } else if op == "Imputer" {
        let (fa, ia, rf, ri) = (
            attr(n, "imputed_value_floats"),
            attr(n, "imputed_value_int64s"),
            attr(n, "replaced_value_float"),
            attr(n, "replaced_value_int64"),
        );
        let floats = if fa.is_null() {
            Some(vec![])
        } else {
            attr_floats(fa)
        };
        let ints = if ia.is_null() {
            Some(vec![])
        } else {
            attr_ints_exact(ia)
        };
        let replaced_float = if rf.is_null() {
            Some(0.)
        } else {
            attr_float(rf).map(|n| n as f32 as f64)
        };
        let replaced_int = if ri.is_null() {
            Some(BigInt::from(0))
        } else {
            attr_int_exact(ri)
        };
        for (missing, code) in [
            (
                !fa.is_null() && floats.is_none(),
                "imputer_imputed_value_floats_not_float_list",
            ),
            (
                !ia.is_null() && ints.is_none(),
                "imputer_imputed_value_int64s_not_exact_int_list",
            ),
            (
                !rf.is_null() && replaced_float.is_none(),
                "imputer_replaced_value_float_not_float_scalar",
            ),
            (
                !ri.is_null() && replaced_int.is_none(),
                "imputer_replaced_value_int64_not_exact_int_scalar",
            ),
        ] {
            if missing {
                failures.push(code.into());
            }
        }
        let (floats, ints) = (floats.unwrap_or_default(), ints.unwrap_or_default());
        let stride = if declared && !s.is_empty() {
            safe(&s[if s.len() == 1 { 0 } else { 1 }]).filter(|n| *n >= 0)
        } else {
            None
        };
        let kind = if floats.is_empty() == ints.is_empty() {
            "invalid"
        } else if !floats.is_empty() {
            "float"
        } else {
            "int64"
        };
        let expected = if matches!(dt, "FLOAT32" | "FLOAT64") {
            "float"
        } else if matches!(dt, "INT32" | "INT64") {
            "int64"
        } else {
            "invalid"
        };
        let active = if kind == "float" {
            floats
                .iter()
                .map(|n| Numeric::Float(*n as f64))
                .collect::<Vec<_>>()
        } else {
            ints.iter().cloned().map(Numeric::Int).collect::<Vec<_>>()
        };
        let (state, why, mode) = if kind == "invalid" {
            (
                "fail",
                "imputer_pinned_ort_requires_exactly_one_nonempty_imputed_value_list",
                "invalid",
            )
        } else if kind != expected {
            (
                "fail",
                "imputer_imputed_value_type_does_not_match_input_dtype",
                "invalid",
            )
        } else if rank == Some(0) {
            (
                "fail",
                "imputer_pinned_ort_rejects_rank_zero_input",
                "invalid",
            )
        } else if rank.is_none() || stride.is_none() {
            if active.len() == 1 {
                ("pass", "imputer_scalar_value", "scalar")
            } else {
                (
                    "not_assessed",
                    "imputer_feature_stride_unresolved",
                    "unresolved",
                )
            }
        } else if stride == Some(active.len() as i64) {
            ("pass", "imputer_per_feature_values", "per_feature")
        } else if active.len() == 1 {
            ("pass", "imputer_scalar_value", "scalar")
        } else {
            (
                "pass",
                "imputer_pinned_ort_scalar_first_fallback",
                "scalar_first_fallback",
            )
        };
        if state != "pass" {
            reasons.push(why.into());
        }
        if declared && !concrete {
            reasons.push("imputer_dynamic_shape_preserved".into());
        }
        let replaced = if kind == "float" {
            replaced_float.map(Numeric::Float)
        } else {
            replaced_int.map(Numeric::Int)
        };
        if state == "pass" {
            if let (Some(src), Some(total)) = (src.as_ref(), elements) {
                if src.len() as i64 != total {
                    failures.push(format!(
                        "imputer_static_value_count_mismatch:{}:{total}",
                        src.len()
                    ));
                } else {
                    measured = impute(src, dt, &active, replaced.as_ref(), stride);
                }
            }
        }
        can = state == "pass";
        let nf = if kind == "float" {
            floats.iter().filter(|n| !n.is_finite()).count()
        } else {
            0
        };
        if state == "fail" {
            risks.push("imputer_pinned_ort_attribute_or_shape_contract_invalid".into());
        }
        if mode == "scalar_first_fallback" {
            risks.push("imputer_attribute_length_outside_onnx_one_or_feature_count".into());
        }
        if matches!(dt, "FLOAT64" | "INT32") {
            risks.push("imputer_schema_dtype_missing_pinned_ort_cpu_kernel".into());
        }
        if nf > 0 || measured["nonFiniteOutputCount"].as_u64().unwrap_or(0) > 0 {
            risks.push("imputer_non_finite_imputed_or_output".into());
        }
        row["exact_batch_count"] = json!(if rank == Some(1) {
            Some(1)
        } else if rank.is_some_and(|r| r > 1) {
            safe(&s[0]).filter(|n| *n >= 0)
        } else {
            None
        });
        row["exact_feature_count"] = json!(stride);
        for (k, val) in [
            ("parameter_contract_status", json!(state)),
            ("parameter_contract_reason", json!(why)),
            ("parameter_mode", json!(mode)),
            ("attribute_kind", json!(kind)),
            ("feature_stride", json!(stride)),
            ("imputed_value_count", json!(active.len())),
            (
                "imputed_values",
                json!(active.iter().map(Numeric::text).collect::<Vec<_>>()),
            ),
            (
                "replaced_value",
                json!(replaced
                    .as_ref()
                    .map(Numeric::text)
                    .unwrap_or("null".into())),
            ),
            (
                "replaced_value_source",
                json!(if if kind == "float" {
                    rf.is_null()
                } else {
                    ri.is_null()
                } {
                    "onnx_schema_default_0"
                } else {
                    "explicit_attribute"
                }),
            ),
            (
                "ignored_imputed_value_count",
                json!(if mode == "scalar_first_fallback" {
                    active.len().saturating_sub(1)
                } else {
                    0
                }),
            ),
            ("non_finite_imputed_value_count", json!(nf)),
            (
                "exact_replacement_count",
                measured["replacementCount"].clone(),
            ),
            (
                "exact_nan_replacement_count",
                measured["nanReplacementCount"].clone(),
            ),
            ("exact_unchanged_count", measured["unchangedCount"].clone()),
        ] {
            row[format!("imputer_{k}")] = val;
        }
    } else {
        let a = attr(n, "threshold");
        let explicit = attr_float(a);
        if !a.is_null() && explicit.is_none() {
            failures.push("binarizer_threshold_not_float_scalar".into());
        }
        let threshold = explicit.unwrap_or(0.);
        if !declared {
            reasons.push("binarizer_input_shape_unresolved".into());
        } else if !concrete {
            reasons.push("binarizer_dynamic_shape_preserved".into());
        }
        if let Some(src) = src.as_ref() {
            measured = binarize(src, threshold);
        }
        if !threshold.is_finite() {
            risks.push("binarizer_non_finite_threshold".into());
        }
        if ["FLOAT64", "INT32", "INT64"].contains(&dt) {
            risks.push("binarizer_dtype_unsupported_by_pinned_ort_cpu".into());
        }
        if t["staticValuesStatus"] == "not_assessed_non_finite_or_unsafe_value"
            && ["FLOAT32", "FLOAT64"].contains(&dt)
        {
            risks.push("binarizer_static_input_contains_non_finite_or_unsafe_value".into());
        }
        row["exact_batch_count"] = Value::Null;
        row["exact_feature_count"] = Value::Null;
        row["threshold_value"] = json!(threshold);
        row["threshold_value_text"] = json!(ft(threshold));
        row["threshold_source"] = json!(if a.is_null() {
            "onnx_schema_default_0"
        } else {
            "explicit_attribute"
        });
        row["threshold_finite"] = json!(threshold.is_finite());
        row["static_value_assessment_status"] = measured["status"].clone();
        row["exact_static_input_value_count"] = measured["inputValueCount"].clone();
        for (dst, src) in [
            ("exact_above_threshold_count", "aboveCount"),
            ("exact_at_or_below_threshold_count", "atOrBelowCount"),
            ("exact_equal_threshold_count", "equalCount"),
            ("exact_output_zero_count", "atOrBelowCount"),
            ("exact_output_one_count", "aboveCount"),
        ] {
            row[dst] = measured[src].clone();
        }
    }
    if op != "Binarizer" {
        for (dst, src) in [
            ("static_assessment_status", "status"),
            ("exact_input_value_count", "inputValueCount"),
            (
                "integer_float32_rounding_count",
                "integerFloat32RoundingCount",
            ),
            ("non_finite_output_count", "nonFiniteOutputCount"),
            ("signed_zero_output_count", "signedZeroOutputCount"),
            ("output_preview", "outputPreview"),
        ] {
            if op != "Imputer" || dst != "integer_float32_rounding_count" {
                row[format!("{prefix}_{dst}")] = measured[src].clone();
            }
        }
        row[format!("{prefix}_output_materialized")] = json!(measured["values"].is_array());
    }
    if measured["values"].is_array() && !patch.is_null() {
        patch["staticValuesStatus"] = json!("complete");
        patch["staticValuesComplete"] = json!(true);
        patch["staticValues"] = measured["values"].clone();
        patch["staticValuesSource"] = json!(if op == "Binarizer" {
            "binarizer_exact_initializer_evaluation".into()
        } else if op == "Imputer" {
            "imputer_pinned_ort_cpu_semantics".into()
        } else {
            format!("{prefix}_pinned_ort_float32_emulation")
        });
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
    row["status"] = json!(status);
    row["reason_codes"] = json!(codes);
    row["risk_codes"] = json!(risks);
    let outputs = if status != "fail" && can && !patch.is_null() && truth(&n["outputs"][0]) {
        vec![json!([n["outputs"][0], patch])]
    } else {
        vec![]
    };
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["node"]["opType"]) {
        "TreeEnsemble" | "TreeEnsembleClassifier" | "TreeEnsembleRegressor" => {
            crate::portable_onnx_ml_tree::run(v)
        }
        "Normalizer" | "Scaler" | "Binarizer" | "Imputer" => main(v),
        "OneHotEncoder" => onehot(v),
        "SVMClassifier" | "SVMRegressor" => crate::portable_onnx_ml_svm::run(v),
        "LabelEncoder" => crate::portable_onnx_ml_label::run(v),
        "LinearClassifier" | "LinearRegressor" => crate::portable_onnx_ml_linear::run(v),
        "ZipMap"
        | "CastMap"
        | "DictVectorizer"
        | "CategoryMapper"
        | "FeatureVectorizer"
        | "ArrayFeatureExtractor" => crate::portable_onnx_ml_mapping::run(v),
        _ => Ok(Value::Null),
    }
}
pub(crate) fn exact_int(v: &Value) -> Option<BigInt> {
    if let Some(s) = v.as_str() {
        if s.len() > 32768 {
            return None;
        }
        s.parse().ok()
    } else {
        safe(v).map(BigInt::from)
    }
}
pub(crate) fn attr_ints_exact(a: &Value) -> Option<Vec<BigInt>> {
    if a["type"] != 7 || a["valueTypesPresent"] != json!([7]) || !a["ints"].is_array() {
        return None;
    }
    let vals = if a["intExactDecimals"].is_array()
        && rows(&a["intExactDecimals"]).len() == rows(&a["ints"]).len()
    {
        &a["intExactDecimals"]
    } else {
        &a["ints"]
    };
    rows(vals).iter().map(exact_int).collect()
}
pub(crate) fn attr_int_exact(a: &Value) -> Option<BigInt> {
    if a["type"] != 2 || a["valueTypesPresent"] != json!([2]) {
        return None;
    }
    exact_int(if truth(&a["iExactDecimal"]) {
        &a["iExactDecimal"]
    } else {
        &a["i"]
    })
}
fn impute(
    src: &[Numeric],
    dt: &str,
    values: &[Numeric],
    replaced: Option<&Numeric>,
    stride: Option<i64>,
) -> Value {
    let materialize = src.len() <= 1_000_000;
    let (mut out, mut preview) = (vec![], vec![]);
    let (mut replacements, mut nanmatches, mut nonfinite, mut signedzero, mut unsafeint) =
        (0, 0, 0, 0, false);
    for (i, n) in src.iter().enumerate() {
        let nan = matches!((n,replaced),(Numeric::Float(x),Some(Numeric::Float(y)))if x.is_nan()&&y.is_nan());
        let equal = nan
            || match (n, replaced) {
                (Numeric::Int(a), Some(Numeric::Int(b))) => a == b,
                (Numeric::Float(a), Some(Numeric::Float(b))) => a == b,
                _ => false,
            };
        let value = if equal {
            &values[if stride == Some(values.len() as i64) {
                i % values.len()
            } else {
                0
            }]
        } else {
            n
        };
        if equal {
            replacements += 1;
        }
        if nan {
            nanmatches += 1;
        }
        if let Numeric::Float(n) = value {
            if !n.is_finite() {
                nonfinite += 1;
            }
            if *n == 0. && n.is_sign_negative() {
                signedzero += 1;
            }
        }
        if dt == "INT64" {
            if let Numeric::Int(n) = value {
                if n.abs() > BigInt::from(SAFE) {
                    unsafeint = true;
                }
            }
        }
        if materialize {
            out.push(json!(value.f64()));
        }
        if preview.len() < 8 {
            preview.push(value.text());
        }
    }
    let valid = materialize && nonfinite == 0 && !unsafeint;
    json!({"status":if valid{"assessed_exact_pinned_ort_semantics"}else if nonfinite>0{"assessed_non_finite_output_not_propagated"}else if unsafeint{"assessed_exact_int64_output_not_materialized_unsafe_json_integer"}else{"assessed_counts_output_not_materialized_limit"},"values":if valid{json!(out)}else{Value::Null},"inputValueCount":src.len(),"replacementCount":replacements,"nanReplacementCount":nanmatches,"unchangedCount":src.len()-replacements,"nonFiniteOutputCount":nonfinite,"signedZeroOutputCount":signedzero,"outputPreview":preview})
}
pub(crate) fn attr_strings(a: &Value) -> Option<Vec<String>> {
    if a["type"] != 8 || a["valueTypesPresent"] != json!([8]) {
        return None;
    }
    a["strings"]
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_owned))
        .collect()
}
fn onehot(v: &Value) -> R<Value> {
    let n = &v["node"];
    let t = input(v, 0);
    let it = ty(v, t);
    let kind = it["kind"].as_str().unwrap_or("unresolved");
    let dt = if kind == "tensor" {
        it["dtype"]
            .as_str()
            .or(it["elementTypeName"].as_str())
            .unwrap_or("UNKNOWN")
    } else {
        "UNKNOWN"
    };
    let (mut reasons, mut failures, mut risks) = (
        Vec::<String>::new(),
        Vec::<String>::new(),
        Vec::<String>::new(),
    );
    if it.is_null() {
        reasons.push("onehot_input_type_unresolved".into());
    } else if kind != "tensor" {
        failures.push(format!("onehot_input_not_tensor:{kind}"));
    }
    let allowed = ["FLOAT32", "FLOAT64", "INT32", "INT64", "STRING"].contains(&dt);
    if dt == "UNKNOWN" {
        reasons.push("onehot_input_dtype_unresolved".into());
    } else if !allowed {
        failures.push(format!("onehot_input_dtype_not_supported:{dt}"));
    }
    let (ia, sa, za) = (
        attr(n, "cats_int64s"),
        attr(n, "cats_strings"),
        attr(n, "zeros"),
    );
    let ints = if ia.is_null() {
        Some(vec![])
    } else {
        attr_ints_exact(ia)
    };
    let strings = if sa.is_null() {
        Some(vec![])
    } else {
        attr_strings(sa)
    };
    let zeros = if za.is_null() {
        Some(BigInt::from(1))
    } else {
        attr_int_exact(za)
    };
    for (bad, code) in [
        (
            !ia.is_null() && ints.is_none(),
            "onehot_cats_int64s_not_exact_int_list",
        ),
        (
            !sa.is_null() && strings.is_none(),
            "onehot_cats_strings_not_string_list",
        ),
        (
            !za.is_null() && zeros.is_none(),
            "onehot_zeros_not_exact_int_scalar",
        ),
    ] {
        if bad {
            failures.push(code.into());
        }
    }
    let (ints, strings) = (ints.unwrap_or_default(), strings.unwrap_or_default());
    let ck = if ints.is_empty() == strings.is_empty() {
        "invalid"
    } else if !ints.is_empty() {
        "int64"
    } else {
        "string"
    };
    let expected = if dt == "STRING" {
        "string"
    } else if allowed {
        "int64"
    } else {
        "invalid"
    };
    let (state, why) = if ck == "invalid" {
        ("fail", "onehot_requires_exactly_one_nonempty_category_list")
    } else if dt == "UNKNOWN" {
        ("not_assessed", "onehot_input_dtype_unresolved")
    } else if ck != expected {
        (
            "fail",
            "onehot_category_list_type_does_not_match_input_dtype",
        )
    } else {
        ("pass", "onehot_single_nonempty_category_list")
    };
    if state != "pass" {
        reasons.push(why.into());
    }
    let categories = if ck == "int64" {
        ints.iter().map(ToString::to_string).collect::<Vec<_>>()
    } else {
        strings
    };
    let mut lookup = std::collections::BTreeMap::new();
    for (i, c) in categories.iter().enumerate() {
        lookup.insert(c.clone(), i);
    }
    let unreachable = categories
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if lookup.get(c) != Some(&i) {
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let duplicates = categories.len() - lookup.len();
    let enabled = zeros.as_ref().map(|n| n != &BigInt::from(0));
    let zeros_canonical = zeros
        .as_ref()
        .is_none_or(|n| n == &BigInt::from(0) || n == &BigInt::from(1));
    let declared = kind == "tensor" && it["shapeDeclared"] == true;
    let s = if declared {
        rows(&it["shape"]).to_vec()
    } else {
        vec![]
    };
    let concrete = declared && s.iter().all(|v| safe(v).is_some_and(|n| n >= 0));
    if declared && !concrete {
        reasons.push("onehot_dynamic_input_shape_preserved".into());
    }
    let output_declared = declared && state == "pass";
    let mut os = if output_declared { s.clone() } else { vec![] };
    if output_declared {
        os.push(json!(categories.len()));
    }
    let mut ot = if state == "pass" {
        tensor_type(
            "FLOAT32",
            &dimensions(&json!({"shape":os})),
            &v["tensor_type_ids"],
        )
    } else {
        Value::Null
    };
    if !ot.is_null() && !output_declared {
        ot["shapeDeclared"] = json!(false);
        ot["shapeFieldCount"] = json!(0);
    }
    let mut patch = if ot.is_null() {
        Value::Null
    } else {
        descriptor(&ot, Value::Null)
    };
    let ie = if concrete { count(&s) } else { None };
    let oe = if state == "pass" {
        ie.and_then(|n| n.checked_mul(categories.len() as i64))
            .filter(|n| *n <= SAFE)
    } else {
        None
    };
    if ie.is_some() && state == "pass" && oe.is_none() {
        reasons.push("onehot_output_element_count_overflow".into());
    }
    let source_keys: Option<Vec<(Option<String>, bool, String)>> = if dt == "STRING"
        && t["staticValuesComplete"] == true
    {
        t["staticValues"].as_array().map(|a| {
            a.iter()
                .map(|n| {
                    (
                        n.as_str().map(str::to_owned),
                        false,
                        n.as_str().map(str::to_owned).unwrap_or(n.to_string()),
                    )
                })
                .collect()
        })
    } else if dt == "INT64" {
        let values = if t["initializerIntegerValuesExactComplete"] == true {
            t["initializerIntegerValuesExactDecimals"].as_array()
        } else if t["staticValuesComplete"] == true {
            t["staticValues"].as_array()
        } else {
            None
        };
        values.map(|a| {
            a.iter()
                .map(|n| {
                    let parsed = exact_int(n);
                    let text = parsed
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| n.to_string());
                    (parsed.map(|n| n.to_string()), false, text)
                })
                .collect()
        })
    } else {
        source(t, dt).map(|a| {
            a.iter()
                .map(|n| {
                    let value = n.f64();
                    let converted = if dt == "INT32" {
                        if value.is_finite() && value.fract() == 0. && value.abs() <= SAFE as f64 {
                            BigInt::from_f64(value)
                        } else {
                            None
                        }
                    } else if ["FLOAT32", "FLOAT64"].contains(&dt)
                        && value.is_finite()
                        && value >= i64::MIN as f64
                        && value < 9_223_372_036_854_775_808f64
                    {
                        BigInt::from_f64(value.trunc())
                    } else {
                        None
                    };
                    (
                        converted.map(|n| n.to_string()),
                        value.trunc() != value,
                        ft(value),
                    )
                })
                .collect()
        })
    };
    let mut measured = unresolved(t);
    measured["guaranteedRuntimeFailure"] = json!(false);
    measured["unknownPreview"] = json!([]);
    if state == "pass" && enabled.is_some() {
        if let (Some(src), Some(total)) = (source_keys.as_ref(), ie) {
            if src.len() as i64 != total {
                failures.push(format!(
                    "onehot_static_value_count_mismatch:{}:{total}",
                    src.len()
                ));
            } else {
                let mut out = oe.filter(|n| *n <= 1_000_000).map(|n| vec![0; n as usize]);
                let (mut matched, mut unknown, mut changed, mut invalid) = (0, 0, 0, 0);
                let mut preview = vec![];
                for (i, (key, cast, text)) in src.iter().enumerate() {
                    let Some(key) = key else {
                        invalid += 1;
                        if preview.len() < 8 {
                            preview.push(text.clone());
                        }
                        continue;
                    };
                    if *cast {
                        changed += 1;
                    }
                    if let Some(j) = lookup.get(key) {
                        matched += 1;
                        if let Some(out) = out.as_mut() {
                            out[i * categories.len() + j] = 1;
                        }
                    } else {
                        unknown += 1;
                        if preview.len() < 8 {
                            preview.push(text.clone());
                        }
                    }
                }
                let runtime = !enabled.unwrap() && unknown > 0;
                let valid = invalid == 0 && !runtime;
                if !valid {
                    out = None;
                }
                let output_preview = out
                    .as_ref()
                    .map(|out| {
                        out.iter()
                            .take(16)
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                measured = json!({"status":if invalid>0{"assessed_unrepresentable_numeric_cast_not_propagated"}else if runtime{"assessed_guaranteed_runtime_failure_unknown_category"}else if out.is_some(){"assessed_exact_pinned_ort_semantics"}else{"assessed_counts_output_not_materialized_limit"},"values":out,"inputValueCount":src.len(),"matchedCount":matched,"unknownCount":unknown,"numericCastChangedCount":changed,"invalidNumericCastCount":invalid,"guaranteedRuntimeFailure":runtime,"outputOneCount":if valid{Some(matched)}else{None},"outputZeroCount":if valid{oe.map(|n|n-matched as i64)}else{None},"unknownPreview":preview,"outputPreview":output_preview});
            }
        }
    }
    if measured["values"].is_array() && !patch.is_null() {
        patch["staticValuesStatus"] = json!("complete");
        patch["staticValuesComplete"] = json!(true);
        patch["staticValues"] = measured["values"].clone();
        patch["staticValuesSource"] = json!("onehot_pinned_ort_cpu_semantics");
    }
    if state == "fail" {
        risks.push("onehot_pinned_ort_attribute_contract_invalid".into());
    }
    if duplicates > 0 {
        risks.push("onehot_duplicate_categories_last_write_wins".into());
    }
    if !zeros_canonical {
        risks.push("onehot_noncanonical_zeros_boolean".into());
    }
    if measured["unknownCount"].as_u64().unwrap_or(0) > 0 && enabled == Some(true) {
        risks.push("onehot_unknown_categories_all_zero_encoding".into());
    }
    if measured["guaranteedRuntimeFailure"] == true {
        risks.push("onehot_unknown_category_guaranteed_runtime_failure".into());
    }
    if measured["invalidNumericCastCount"].as_u64().unwrap_or(0) > 0 {
        risks.push("onehot_numeric_to_int64_cast_not_representable".into());
    }
    if dt == "INT32" {
        risks.push("onehot_schema_dtype_missing_pinned_ort_cpu_kernel".into());
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
    let mut row = templates()["OneHotEncoder"].clone();
    let fields = json!({"scope":v["scope"],"node_index":v["node_index"],"imported_opset":v["imported_opset"],"status":status,"input_name":n["inputs"][0],"output_name":n["outputs"][0],"input_dtype":dt,"input_kind":kind,"input_rank":if declared{Some(s.len())}else{None},"input_shape":s,"canonical_output_type":if ot.is_null(){"unresolved".into()}else{canonical(&ot)},"output_dtype":if state=="pass"{"FLOAT32"}else{"UNKNOWN"},"exact_output_rank":if output_declared{Some(os.len())}else{None},"exact_output_shape":os,"exact_dense_output_element_count":oe,"onehot_parameter_contract_status":state,"onehot_parameter_contract_reason":why,"onehot_category_kind":ck,"onehot_category_count":categories.len(),"onehot_category_values":categories,"onehot_duplicate_category_count":duplicates,"onehot_unreachable_duplicate_column_count":unreachable.len(),"onehot_unreachable_duplicate_column_indices":unreachable,"onehot_zeros_value":zeros.as_ref().map(ToString::to_string).unwrap_or("unresolved".into()),"onehot_zeros_source":if za.is_null(){"onnx_schema_default_1"}else{"explicit_attribute"},"onehot_zeros_enabled":enabled,"onehot_zeros_canonical_boolean":zeros_canonical,"onehot_output_materialized":measured["values"].is_array(),"vocabulary_type":match ck{"string"=>"STRING","int64"=>"INT64",_=>"UNDEFINED"},"vocabulary_count":categories.len(),"duplicate_vocabulary_count":duplicates,"vocabulary_preview":categories.iter().take(8).collect::<Vec<_>>(),"reason_codes":codes,"risk_codes":risks});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    for (dst, src) in [
        ("static_assessment_status", "status"),
        ("exact_input_value_count", "inputValueCount"),
        ("exact_matched_input_count", "matchedCount"),
        ("exact_unknown_input_count", "unknownCount"),
        ("numeric_to_int64_changed_count", "numericCastChangedCount"),
        ("numeric_to_int64_invalid_count", "invalidNumericCastCount"),
        ("guaranteed_runtime_failure", "guaranteedRuntimeFailure"),
        ("exact_output_one_count", "outputOneCount"),
        ("exact_output_zero_count", "outputZeroCount"),
        ("unknown_input_preview", "unknownPreview"),
        ("output_preview", "outputPreview"),
    ] {
        row[format!("onehot_{dst}")] = measured[src].clone();
    }
    let outputs = if status != "fail"
        && state == "pass"
        && measured["guaranteedRuntimeFailure"] != true
        && !patch.is_null()
        && truth(&n["outputs"][0])
    {
        vec![json!([n["outputs"][0], patch])]
    } else {
        vec![]
    };
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
