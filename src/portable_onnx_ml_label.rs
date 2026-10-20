//! Versioned LabelEncoder mappings. Preserve exact INT64 and non-finite tokens.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_ml::{count, exact_int, float, ft, templates, unique};
use crate::portable_onnx_ml_mapping::{dtype, tensor};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde_json::{json, Value};
use std::collections::BTreeMap;
const DTYPES: &[&str] = &["STRING", "INT64", "FLOAT32", "INT32", "INT16", "FLOAT64"];
#[derive(Clone)]
pub(crate) enum Atom {
    Text(String),
    Int(BigInt),
    Num(f64),
}
impl Atom {
    pub(crate) fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Int(n) => n.to_string(),
            Self::Num(n) => ft(*n),
        }
    }
    fn key(&self) -> String {
        match self {
            Self::Text(s) => format!("s:{s}"),
            Self::Int(n) => format!("i:{n}"),
            Self::Num(n) => format!("n:{}", if *n == 0. { "0".into() } else { ft(*n) }),
        }
    }
    fn finite(&self) -> bool {
        !matches!(self,Self::Num(n)if !n.is_finite())
    }
    fn nan(&self) -> bool {
        matches!(self,Self::Num(n)if n.is_nan())
    }
    fn json(&self) -> Value {
        match self {
            Self::Text(s) => json!(s),
            Self::Int(n) => json!(n.to_f64()),
            Self::Num(n) => json!(n),
        }
    }
    fn equals(&self, b: &Self) -> bool {
        match (self, b) {
            (Self::Num(a), Self::Num(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            _ => false,
        }
    }
}
struct Decoded {
    ok: bool,
    reason: String,
    dtype: String,
    source: String,
    values: Vec<Atom>,
}
fn bad(reason: String, dtype: &str, source: &str) -> Decoded {
    Decoded {
        ok: false,
        reason,
        dtype: dtype.into(),
        source: source.into(),
        values: vec![],
    }
}
fn good(values: Vec<Atom>, dt: &str, source: &str) -> Decoded {
    Decoded {
        ok: true,
        reason: String::new(),
        dtype: dt.into(),
        source: source.into(),
        values,
    }
}
fn repeated(a: &Value, dt: &str, kind: i64, source: &str, optional: bool) -> Decoded {
    if a.is_null() {
        return if optional {
            good(vec![], dt, "onnx_schema_default_empty_list")
        } else {
            bad(
                format!("label_encoder_attribute_missing:{source}"),
                dt,
                source,
            )
        };
    }
    if a["type"] != kind || a["valueTypesPresent"] != json!([kind]) {
        return bad(
            format!("label_encoder_attribute_type_invalid:{source}"),
            dt,
            source,
        );
    }
    let values: Option<Vec<Atom>> = if dt == "STRING" {
        a["strings"].as_array().and_then(|a| {
            a.iter()
                .map(|v| v.as_str().map(|s| Atom::Text(s.into())))
                .collect()
        })
    } else if dt == "FLOAT32" {
        let list = if a["floatsText"].is_array() {
            &a["floatsText"]
        } else {
            &a["floats"]
        };
        list.as_array()
            .and_then(|a| a.iter().map(|n| float(n).map(Atom::Num)).collect())
    } else {
        let exact = rows(&a["intExactDecimals"]);
        let result = exact
            .iter()
            .map(|n| exact_int(n).map(Atom::Int))
            .collect::<Option<Vec<_>>>();
        if let Some(values) = result {
            if exact.len() != rows(&a["ints"]).len() {
                return Decoded {
                    ok: false,
                    values,
                    reason: format!("label_encoder_int64_attribute_exact_value_gap:{source}"),
                    dtype: dt.into(),
                    source: source.into(),
                };
            }
            Some(values)
        } else {
            None
        }
    };
    values.map(|v| good(v, dt, source)).unwrap_or_else(|| {
        bad(
            format!("label_encoder_int64_attribute_decode_failed:{source}"),
            dt,
            source,
        )
    })
}
pub(crate) fn tensor_values(t: &Value) -> Result<Vec<Atom>, String> {
    let dt = t["dtype"].as_str().unwrap_or("UNKNOWN");
    let count = safe(&t["shape"][0])
        .filter(|n| *n >= 0)
        .ok_or("label_encoder_tensor_payload_not_exactly_decoded")?;
    if dt == "STRING" {
        let values = rows(&t["stringValues"])
            .iter()
            .filter_map(|s| s.as_str().map(|s| Atom::Text(s.into())))
            .collect::<Vec<_>>();
        return if values.len() as i64 == count {
            Ok(values)
        } else {
            Err("label_encoder_string_tensor_cardinality_mismatch".into())
        };
    }
    let values = if let Some(raw) = t["rawData"].as_array() {
        let width = if ["INT16", "INT32", "FLOAT32", "INT64", "FLOAT64"].contains(&dt) {
            crate::scalar_types_generated::scalar_dtype_bits(dt).unwrap_or(0) as i64 / 8
        } else {
            0
        };
        if width == 0 || count.checked_mul(width) != Some(raw.len() as i64) {
            return Err("label_encoder_tensor_raw_data_size_mismatch".into());
        }
        let bytes = raw
            .iter()
            .map(|n| n.as_u64().filter(|n| *n <= 255).map(|n| n as u8))
            .collect::<Option<Vec<_>>>()
            .ok_or("label_encoder_tensor_raw_data_size_mismatch")?;
        bytes
            .chunks_exact(width as usize)
            .map(|b| match dt {
                "INT16" => Atom::Num(i16::from_le_bytes(b.try_into().unwrap()) as f64),
                "INT32" => Atom::Num(i32::from_le_bytes(b.try_into().unwrap()) as f64),
                "INT64" => Atom::Int(BigInt::from(i64::from_le_bytes(b.try_into().unwrap()))),
                "FLOAT32" => Atom::Num(f32::from_le_bytes(b.try_into().unwrap()) as f64),
                _ => Atom::Num(f64::from_le_bytes(b.try_into().unwrap())),
            })
            .collect::<Vec<_>>()
    } else if let Some(values) = t["typedValues"].as_array() {
        values
            .iter()
            .map(|n| {
                if dt == "INT64" {
                    exact_int(n).map(Atom::Int)
                } else {
                    float(n).map(Atom::Num)
                }
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("label_encoder_tensor_payload_not_exactly_decoded")?
    } else {
        vec![]
    };
    if values.len() as i64 != count {
        Err("label_encoder_tensor_payload_not_exactly_decoded".into())
    } else {
        Ok(values)
    }
}
fn tensor_attr(a: &Value, source: &str) -> Decoded {
    let t = &a["tensor"];
    if a["type"] != 4 || t.is_null() || a["valueTypesPresent"] != json!([4]) {
        return bad(
            format!("label_encoder_tensor_attribute_invalid:{source}"),
            "UNKNOWN",
            source,
        );
    }
    let dt = t["dtype"].as_str().unwrap_or("UNKNOWN");
    if !t["shape"].is_array()
        || rows(&t["shape"]).len() != 1
        || safe(&t["shape"][0]).is_none_or(|n| n < 0)
    {
        return bad(
            format!("label_encoder_tensor_attribute_not_1d:{source}"),
            dt,
            source,
        );
    }
    if !DTYPES.contains(&dt) {
        return bad(
            format!("label_encoder_tensor_attribute_dtype_invalid:{source}:{dt}"),
            dt,
            source,
        );
    }
    if t["externalDataEntries"].as_f64().unwrap_or(0.) > 0.
        || t["dataLocation"].as_f64().unwrap_or(0.) != 0.
    {
        return bad(
            format!("label_encoder_tensor_attribute_external_payload_not_executable:{source}"),
            dt,
            source,
        );
    }
    match tensor_values(t) {
        Ok(values) => good(values, dt, source),
        Err(reason) => bad(reason, dt, source),
    }
}
fn select(n: &Value, role: &str, version: Option<i64>) -> Decoded {
    let candidates = if version.is_some_and(|n| n >= 4) {
        vec![
            (format!("{role}_tensor"), "TENSOR", 4),
            (format!("{role}_strings"), "STRING", 8),
            (format!("{role}_int64s"), "INT64", 7),
            (format!("{role}_floats"), "FLOAT32", 6),
        ]
    } else {
        vec![
            (format!("{role}_strings"), "STRING", 8),
            (format!("{role}_int64s"), "INT64", 7),
            (format!("{role}_floats"), "FLOAT32", 6),
        ]
    };
    let present = candidates
        .iter()
        .filter(|(name, _, _)| !attr(n, name).is_null())
        .collect::<Vec<_>>();
    if present.len() != 1 {
        return bad(
            format!(
                "label_encoder_requires_exactly_one_{role}_attribute:{}",
                present.len()
            ),
            "UNKNOWN",
            "unresolved",
        );
    }
    let (name, dt, kind) = present[0];
    if *dt == "TENSOR" {
        tensor_attr(attr(n, name), name)
    } else {
        repeated(attr(n, name), dt, *kind, name, false)
    }
}
fn default_value(n: &Value, dt: &str, version: Option<i64>) -> Decoded {
    if version.is_some_and(|n| n >= 4) && !attr(n, "default_tensor").is_null() {
        let result = tensor_attr(attr(n, "default_tensor"), "default_tensor");
        if !result.ok {
            return result;
        }
        if result.values.len() != 1 {
            return bad(
                "label_encoder_default_tensor_not_singleton".into(),
                dt,
                "default_tensor",
            );
        }
        if result.dtype != dt {
            return bad(
                format!(
                    "label_encoder_default_tensor_dtype_mismatch:{}:{dt}",
                    result.dtype
                ),
                dt,
                "default_tensor",
            );
        }
        return good(result.values, dt, "explicit_default_tensor");
    }
    let (name, kind, fallback) = match dt {
        "STRING" => ("default_string", 3, Atom::Text("_Unused".into())),
        "INT64" => ("default_int64", 2, Atom::Int(BigInt::from(-1))),
        "FLOAT32" => ("default_float", 1, Atom::Num(-0.)),
        "INT16" | "INT32" => ("", 0, Atom::Num(-1.)),
        "FLOAT64" => ("", 0, Atom::Num(-0.)),
        _ => {
            return bad(
                "label_encoder_default_output_dtype_unresolved".into(),
                dt,
                "unresolved",
            )
        }
    };
    let a = attr(n, name);
    if name.is_empty() || a.is_null() {
        let source = format!("onnx_schema_default_{}", fallback.text());
        return good(vec![fallback], dt, &source);
    }
    if a["type"] != kind || a["valueTypesPresent"] != json!([kind]) {
        return bad(
            format!("label_encoder_default_attribute_invalid:{name}"),
            dt,
            name,
        );
    }
    let value = match kind {
        3 => a["s"].as_str().map(|s| Atom::Text(s.into())),
        2 => exact_int(if truth(&a["iExactDecimal"]) {
            &a["iExactDecimal"]
        } else {
            &a["i"]
        })
        .map(Atom::Int),
        _ => float(if a.get("fText").is_some() {
            &a["fText"]
        } else {
            &a["f"]
        })
        .map(Atom::Num),
    };
    value
        .map(|v| good(vec![v], dt, &format!("explicit_{name}")))
        .unwrap_or_else(|| {
            bad(
                format!("label_encoder_default_attribute_decode_failed:{name}"),
                dt,
                name,
            )
        })
}
fn runtime(
    version: Option<i64>,
    dt: &str,
    od: &str,
    failures: &[String],
    keys: usize,
    values: usize,
) -> (&'static str, String) {
    if version.is_none() || dt == "UNKNOWN" || od == "UNKNOWN" {
        return ("not_assessed", "dtype_or_schema_unresolved".into());
    }
    if !failures.is_empty() {
        return ("fail", failures[0].clone());
    }
    if keys != values {
        return ("fail", "pinned_ort_key_value_count_mismatch".into());
    }
    let pair = format!("{dt}:{od}");
    let (allowed, yes, no) = match version {
        Some(1) => (
            ["STRING:INT64", "INT64:STRING"].contains(&pair.as_str()),
            "pinned_ort_v1_bidirectional_kernel",
            "pinned_ort_v1_dtype_pair_missing",
        ),
        Some(2) => (
            ["STRING", "INT64", "FLOAT32"].contains(&dt)
                && ["STRING", "INT64", "FLOAT32"].contains(&od),
            "pinned_ort_v2_versioned_dtype_pair_kernel",
            "pinned_ort_v2_dtype_pair_missing",
        ),
        _ => (
            [
                "INT64:INT64",
                "INT64:STRING",
                "INT64:FLOAT32",
                "INT64:FLOAT64",
                "FLOAT32:FLOAT32",
                "FLOAT32:STRING",
                "FLOAT32:INT64",
                "STRING:INT64",
                "STRING:FLOAT32",
                "STRING:STRING",
                "STRING:INT16",
                "STRING:FLOAT64",
                "FLOAT64:FLOAT64",
                "FLOAT64:STRING",
                "FLOAT64:INT64",
            ]
            .contains(&pair.as_str()),
            "pinned_ort_v4_typed_dtype_pair_kernel",
            "",
        ),
    };
    (
        if allowed { "pass" } else { "fail" },
        if allowed {
            yes.into()
        } else if no.is_empty() {
            format!("pinned_ort_v4_dtype_pair_missing:{pair}")
        } else {
            no.into()
        },
    )
}
fn static_input(t: &Value, dt: &str) -> Option<Vec<Atom>> {
    if dt == "INT64" && t["initializerIntegerValuesExactComplete"] == true {
        return t["initializerIntegerValuesExactDecimals"]
            .as_array()?
            .iter()
            .map(|n| exact_int(n).map(Atom::Int))
            .collect();
    }
    if DTYPES.contains(&dt) && t["staticValuesComplete"] == true {
        let values = if dt != "STRING" && t["staticValuesText"].is_array() {
            &t["staticValuesText"]
        } else {
            &t["staticValues"]
        };
        return values
            .as_array()?
            .iter()
            .map(|n| {
                if dt == "STRING" {
                    n.as_str().map(|s| Atom::Text(s.into()))
                } else {
                    float(n).map(Atom::Num)
                }
            })
            .collect();
    }
    if ["FLOAT32", "FLOAT64"].contains(&dt) && t["staticValuesCanonicalTextComplete"] == true {
        return t["staticValuesCanonicalTexts"]
            .as_array()?
            .iter()
            .map(|n| float(n).map(Atom::Num))
            .collect();
    }
    None
}
struct Exact {
    row: Value,
    output: Option<Vec<Atom>>,
}
fn unresolved(t: &Value) -> Exact {
    Exact {
        output: None,
        row: json!({"status":if t["role"]=="initializer"{t["staticValuesStatus"].as_str().filter(|s|!s.is_empty()).unwrap_or("not_assessed_initializer_values")}else{"not_assessed_runtime_values"},"inputCount":null,"matchCount":null,"defaultCount":null,"duplicateKeyHitCount":null,"schemaRuntimeMismatchCount":null,"outputPreview":[],"schemaOutputPreview":[],"mismatchInputPreview":[]}),
    }
}
fn evaluate(
    source: &[Atom],
    keys: &[Atom],
    values: &[Atom],
    default: &Atom,
    last: bool,
    version: Option<i64>,
    count: Option<i64>,
) -> Exact {
    let (mut first, mut final_map, mut occurrences) = (
        BTreeMap::<String, usize>::new(),
        BTreeMap::new(),
        BTreeMap::<String, usize>::new(),
    );
    for (i, key) in keys.iter().enumerate() {
        let key = key.key();
        first.entry(key.clone()).or_insert(i);
        final_map.insert(key.clone(), i);
        *occurrences.entry(key).or_default() += 1;
    }
    let active = if last { &final_map } else { &first };
    let mut output = if count.is_none_or(|n| n <= 1_000_000) {
        Some(vec![])
    } else {
        None
    };
    let (mut preview, mut schema_preview, mut mismatch_preview) = (vec![], vec![], vec![]);
    let (mut matches, mut defaults, mut duphits, mut mismatch) = (0, 0, 0, 0);
    for value in source {
        let key = value.key();
        let index = if version == Some(2) && value.nan() {
            None
        } else {
            active.get(&key)
        };
        let runtime = index.map(|i| &values[*i]).unwrap_or(default);
        if index.is_some() {
            matches += 1;
        } else {
            defaults += 1;
        }
        if occurrences.get(&key).is_some_and(|n| *n > 1) {
            duphits += 1;
        }
        if let Some(out) = output.as_mut() {
            out.push(runtime.clone());
        }
        if preview.len() < 16 {
            preview.push(runtime.text());
        }
        if version == Some(4) {
            let schema = final_map.get(&key).map(|i| &values[*i]).unwrap_or(default);
            if schema_preview.len() < 16 {
                schema_preview.push(schema.text());
            }
            if !runtime.equals(schema) {
                mismatch += 1;
                if mismatch_preview.len() < 8 {
                    mismatch_preview.push(value.text());
                }
            }
        }
    }
    Exact {
        row: json!({"status":if output.is_some(){"assessed_exact_pinned_ort_semantics"}else{"assessed_counts_output_not_materialized_limit"},"inputCount":source.len(),"matchCount":matches,"defaultCount":defaults,"duplicateKeyHitCount":duphits,"schemaRuntimeMismatchCount":if version==Some(4){Some(mismatch)}else{None},"outputPreview":preview,"schemaOutputPreview":schema_preview,"mismatchInputPreview":mismatch_preview}),
        output,
    }
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let input = input(v, 0);
    let t = ty(v, input);
    let dt = dtype(&t);
    let version =
        crate::portable_onnx_ml_schema::version("LabelEncoder", safe(&v["imported_opset"]));
    let (mut failures, mut reasons) = (vec![], vec![]);
    if t.is_null() {
        reasons.push("label_encoder_input_type_unresolved".into());
    } else if t["kind"] != "tensor" {
        failures.push(format!(
            "label_encoder_input_not_tensor:{}",
            string(&t["kind"])
        ));
    }
    if let Some(ver) = version {
        if dt == "UNKNOWN" {
            reasons.push("label_encoder_input_dtype_unresolved".into());
        } else if !(if ver == 1 {
            &DTYPES[..2]
        } else if ver == 2 {
            &DTYPES[..3]
        } else {
            DTYPES
        })
        .contains(&dt)
        {
            failures.push(format!(
                "label_encoder_input_dtype_not_supported_by_schema_v{ver}:{dt}"
            ));
        }
    } else {
        failures.push("label_encoder_operator_not_defined_at_imported_opset".into());
    }
    let mut schema_failures = vec![];
    let (keys, values, kdt, od, ks, vs, mode, rpolicy, spolicy, default) = if version == Some(1) {
        let classes = repeated(
            attr(n, "classes_strings"),
            "STRING",
            8,
            "classes_strings",
            true,
        );
        if !classes.ok {
            schema_failures.push(classes.reason.clone());
        }
        let od = match dt {
            "STRING" => "INT64",
            "INT64" => "STRING",
            _ => "UNKNOWN",
        };
        let indices = (0..classes.values.len())
            .map(|i| Atom::Int(BigInt::from(i)))
            .collect::<Vec<_>>();
        let (keys, values) = if dt == "STRING" {
            (classes.values, indices)
        } else {
            (indices, classes.values)
        };
        let default = default_value(n, od, version);
        if !default.ok {
            schema_failures.push(default.reason.clone());
        }
        (
            keys,
            values,
            dt.to_owned(),
            od.to_owned(),
            "classes_strings_or_implicit_indices".into(),
            "classes_strings_or_implicit_indices".into(),
            "v1_classes_strings_bidirectional".into(),
            if dt == "STRING" {
                "last_key_wins"
            } else {
                "unique_implicit_indices"
            },
            if dt == "STRING" {
                "unspecified_list_lookup_index"
            } else {
                "unique_implicit_indices"
            },
            default,
        )
    } else {
        let keys = select(n, "keys", version);
        let values = select(n, "values", version);
        if !keys.ok {
            schema_failures.push(keys.reason.clone());
        }
        if !values.ok {
            schema_failures.push(values.reason.clone());
        }
        if keys.ok && dt != "UNKNOWN" && keys.dtype != dt {
            schema_failures.push(format!(
                "label_encoder_key_dtype_mismatch:{}:{dt}",
                keys.dtype
            ));
        }
        if keys.ok
            && values.ok
            && version.is_some_and(|n| n >= 4)
            && keys.values.len() != values.values.len()
        {
            schema_failures.push(format!(
                "label_encoder_schema_v4_key_value_count_mismatch:{}:{}",
                keys.values.len(),
                values.values.len()
            ));
        }
        let od = if values.ok {
            values.dtype
        } else {
            "UNKNOWN".into()
        };
        let default = default_value(n, &od, version);
        if !default.ok {
            schema_failures.push(default.reason.clone());
        }
        (
            if keys.ok { keys.values } else { vec![] },
            if values.ok { values.values } else { vec![] },
            if keys.ok {
                keys.dtype
            } else {
                "UNKNOWN".into()
            },
            od,
            keys.source,
            values.source,
            format!(
                "v{}_parallel_key_value_attributes",
                version.map(|n| n.to_string()).unwrap_or("null".into())
            ),
            "first_key_wins",
            if version == Some(4) {
                "last_key_wins"
            } else {
                "unspecified_except_bitwise_float_lookup"
            },
            default,
        )
    };
    failures.extend(schema_failures.clone());
    let declared = t["kind"] == "tensor" && t["shapeDeclared"] == true;
    let shape = if declared {
        rows(&t["shape"]).to_vec()
    } else {
        vec![]
    };
    let concrete = declared && shape.iter().all(|d| safe(d).is_some_and(|n| n >= 0));
    let total = if concrete { count(&shape) } else { None };
    if !declared {
        reasons.push("label_encoder_input_shape_unresolved".into());
    } else if !concrete {
        reasons.push("label_encoder_dynamic_input_shape_preserved".into());
    } else if total.is_none() {
        reasons.push("label_encoder_output_element_count_overflow".into());
    }
    let ot = if od != "UNKNOWN" {
        tensor(v, &od, &shape, declared)
    } else {
        Value::Null
    };
    let mut patch = if ot.is_null() {
        Value::Null
    } else {
        descriptor(&ot, Value::Null)
    };
    let (runtime_status, runtime_reason) =
        runtime(version, dt, &od, &schema_failures, keys.len(), values.len());
    let mut exact = unresolved(input);
    if let Some(source) = static_input(input, dt) {
        if total.is_some_and(|n| n != source.len() as i64) {
            failures.push(format!(
                "label_encoder_static_value_count_mismatch:{}:{}",
                source.len(),
                total.unwrap()
            ));
        } else if runtime_status == "pass"
            && keys.len() == values.len()
            && schema_failures.is_empty()
            && default.ok
        {
            exact = evaluate(
                &source,
                &keys,
                &values,
                &default.values[0],
                rpolicy == "last_key_wins",
                version,
                total,
            );
        }
    }
    let agree = exact.row["schemaRuntimeMismatchCount"]
        .as_u64()
        .unwrap_or(0)
        == 0;
    if !patch.is_null() && agree {
        if let Some(out) = &exact.output {
            if od == "INT64" {
                let decimal = out.iter().map(Atom::text).collect::<Vec<_>>();
                let safe=out.iter().all(|n|matches!(n,Atom::Int(n)if n.to_i64().is_some_and(|n|n.unsigned_abs()<=SAFE as u64)));
                patch["initializerIntegerValuesStatus"] = json!(if safe {
                    "complete"
                } else {
                    "not_assessed_outside_safe_integer"
                });
                patch["initializerIntegerValuesComplete"] = json!(safe);
                patch["initializerIntegerValues"] = json!(if safe {
                    out.iter().map(Atom::json).collect::<Vec<_>>()
                } else {
                    vec![]
                });
                patch["initializerIntegerValuesExactComplete"] = json!(true);
                patch["initializerIntegerValuesExactDecimals"] = json!(decimal);
                if safe {
                    patch["staticValuesStatus"] = json!("complete");
                    patch["staticValuesComplete"] = json!(true);
                    patch["staticValues"] = json!(out.iter().map(Atom::json).collect::<Vec<_>>());
                } else {
                    patch["staticValuesStatus"] = json!("complete_exact_int64_decimal_only");
                }
            } else if out.iter().all(Atom::finite) {
                patch["staticValuesStatus"] = json!("complete");
                patch["staticValuesComplete"] = json!(true);
                patch["staticValues"] = json!(out.iter().map(Atom::json).collect::<Vec<_>>());
            } else {
                patch["staticValuesStatus"] = json!("complete_canonical_numeric_text_only");
                patch["staticValuesCanonicalTextComplete"] = json!(true);
                patch["staticValuesCanonicalTexts"] =
                    json!(out.iter().map(Atom::text).collect::<Vec<_>>());
            }
            patch["staticValuesSource"] = json!(format!(
                "label_encoder_v{}_pinned_ort_cpu_semantics",
                version.unwrap_or(0)
            ));
        }
    }
    let dup = keys.len()
        - keys
            .iter()
            .map(Atom::key)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
    let nan = keys.iter().filter(|n| n.nan()).count();
    let nfk = keys.iter().filter(|n| !n.finite()).count();
    let nfv = values.iter().filter(|n| !n.finite()).count();
    let mut risks = Vec::<String>::new();
    if runtime_status == "fail"
        && runtime_reason.contains("dtype_pair_missing")
        && schema_failures.is_empty()
    {
        risks.push("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel".into());
    }
    if runtime_status == "fail" && runtime_reason == "pinned_ort_key_value_count_mismatch" {
        risks.push("label_encoder_pinned_ort_runtime_contract_invalid".into());
    }
    if version == Some(4) && dup > 0 {
        risks.push("label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict".into());
    }
    if version == Some(2) && nan > 0 {
        risks.push("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched".into());
    }
    if version == Some(1) && dup > 0 {
        risks.push("label_encoder_v1_duplicate_class_runtime_last_index".into());
    }
    if exact.row["defaultCount"].as_u64().unwrap_or(0) > 0 {
        risks.push("label_encoder_artifact_known_default_path_reached".into());
    }
    if !agree {
        risks.push("label_encoder_artifact_known_schema_runtime_output_mismatch".into());
    }
    if nfk > 0 || nfv > 0 || default.values.iter().any(|n| !n.finite()) {
        risks.push("label_encoder_non_finite_mapping_state".into());
    }
    let status = if !failures.is_empty() {
        "fail"
    } else if !reasons.is_empty() {
        "partial"
    } else {
        "pass"
    };
    let onnx_status = if !failures.is_empty() {
        "fail"
    } else if od == "UNKNOWN" {
        "not_assessed"
    } else {
        "pass"
    };
    failures.extend(reasons);
    let codes = unique(failures);
    let mut row = templates()["LabelEncoder"].clone();
    let default_text = default
        .values
        .first()
        .filter(|_| default.ok)
        .map(Atom::text);
    let fields = json!({"scope":v["scope"],"node_index":v["node_index"],"imported_opset":v["imported_opset"],"resolved_schema_version":version,"status":status,"input_name":n["inputs"][0].as_str().unwrap_or(""),"output_name":n["outputs"][0].as_str().unwrap_or(""),"input_dtype":dt,"input_kind":t["kind"].as_str().unwrap_or("unresolved"),"input_rank":if declared{Some(shape.len())}else{None},"input_shape":shape,"canonical_output_type":if ot.is_null(){"unresolved".into()}else{canonical(&ot)},"output_dtype":od,"exact_output_rank":if declared&&!ot.is_null(){Some(shape.len())}else{None},"exact_output_shape":if declared&&!ot.is_null(){shape}else{vec![]},"exact_dense_output_element_count":if ot.is_null(){None}else{total},"output_shape_basis":format!("pinned_onnx_label_encoder_v{}_same_shape_mapping",version.map(|n|n.to_string()).unwrap_or("unresolved".into())),"attribute_mode":mode});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let fields = json!({"label_encoder_onnx_contract_status":onnx_status,"label_encoder_pinned_ort_contract_status":runtime_status,"label_encoder_pinned_ort_contract_reason":runtime_reason,"label_encoder_key_dtype":kdt,"label_encoder_value_dtype":od,"label_encoder_key_source":ks,"label_encoder_value_source":vs,"label_encoder_default_source":default.source,"label_encoder_default_value":default_text.clone().unwrap_or("unresolved".into()),"label_encoder_key_count":keys.len(),"label_encoder_value_count":values.len(),"label_encoder_key_values":keys.iter().map(Atom::text).collect::<Vec<_>>(),"label_encoder_value_values":values.iter().map(Atom::text).collect::<Vec<_>>(),"label_encoder_duplicate_key_count":dup,"label_encoder_nan_key_count":nan,"label_encoder_non_finite_key_count":nfk,"label_encoder_non_finite_value_count":nfv,"label_encoder_runtime_duplicate_policy":rpolicy,"label_encoder_schema_duplicate_policy":spolicy,"label_encoder_output_materialized":exact.output.is_some()&&agree,"vocabulary_type":kdt,"vocabulary_count":keys.len(),"duplicate_vocabulary_count":dup,"vocabulary_preview":keys.iter().take(8).map(Atom::text).collect::<Vec<_>>(),"mapping_direction":format!("{kdt}_TO_{od}"),"active_default_type":od,"active_default_value":default_text.unwrap_or_default(),"reason_codes":codes,"risk_codes":risks});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    for (dst, src) in [
        ("static_assessment_status", "status"),
        ("exact_input_value_count", "inputCount"),
        ("exact_match_count", "matchCount"),
        ("exact_default_count", "defaultCount"),
        ("exact_duplicate_key_hit_count", "duplicateKeyHitCount"),
        (
            "schema_runtime_mismatch_count",
            "schemaRuntimeMismatchCount",
        ),
        ("runtime_output_preview", "outputPreview"),
        ("schema_output_preview", "schemaOutputPreview"),
        ("mismatch_input_preview", "mismatchInputPreview"),
    ] {
        row[format!("label_encoder_{dst}")] = exact.row[src].clone();
    }
    let outputs = if status != "fail" && !patch.is_null() && truth(&n["outputs"][0]) {
        vec![json!([n["outputs"][0], patch])]
    } else {
        vec![]
    };
    Ok(
        json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row}),
    )
}
