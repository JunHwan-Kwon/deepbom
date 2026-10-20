//! Shared semantic validation. All frontends use this owner; adapters only
//! transport admitted JSON and retain I/O, immutable-object caching and rendering.
use num_bigint::BigUint;
use num_traits::Zero;
use serde_json::{json, Value};
use std::collections::BTreeMap;
pub(crate) type R<T> = Result<T, String>;
pub(crate) fn need(ok: bool, message: &str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
pub(crate) fn rows(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
pub(crate) fn array<'a>(v: &'a Value, message: &str) -> R<&'a [Value]> {
    v.as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| message.into())
}
pub(crate) fn text(v: &Value, max: usize) -> bool {
    v.as_str().is_some_and(|s| {
        !s.trim_matches(crate::portable_evidence::js_space)
            .is_empty()
            && s.trim_matches(crate::portable_evidence::js_space)
                .encode_utf16()
                .count()
                <= max
    })
}
pub(crate) fn sha(v: &Value) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
pub(crate) fn string(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
pub(crate) fn keys(v: &Value, fields: &[&str], message: &str) -> R<()> {
    need(
        v.as_object()
            .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|f| m.contains_key(*f))),
        message,
    )
}
pub(crate) fn index<'a>(v: &'a Value, message: &str) -> R<BTreeMap<&'a str, &'a Value>> {
    let mut m = BTreeMap::new();
    for row in array(v, message)? {
        let id = row["id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or(message)?;
        need(m.insert(id, row).is_none(), message)?;
    }
    Ok(m)
}
pub(crate) fn member(v: &Value, s: &str) -> bool {
    rows(v).iter().any(|x| x == s)
}
pub(crate) fn count(v: &Value) -> R<()> {
    crate::portable_ir::run(&json!({"operation":"validate_exact_integer","value":v})).map(|_| ())
}
pub(crate) fn integer(v: &Value) -> R<BigUint> {
    crate::portable_contracts::decimal(v).ok_or_else(|| "Invalid exact integer.".into())
}
pub(crate) fn decimal(v: &Value) -> R<BigUint> {
    let s = v
        .as_str()
        .ok_or("count must be a canonical nonnegative decimal string")?;
    need(
        !s.is_empty() && (s == "0" || !s.starts_with('0')),
        "count must be a canonical nonnegative decimal string",
    )?;
    integer(v)
}
pub(crate) fn sum(rows: &[Value], field: impl Fn(&Value) -> &Value) -> R<BigUint> {
    let mut total = BigUint::zero();
    for row in rows {
        let v = field(row);
        if !v.is_null() {
            total += integer(v)?;
        }
    }
    Ok(total)
}
pub(crate) fn hash(v: &Value) -> R<String> {
    crate::portable_identity::digest(v)
}
pub(crate) fn digest(v: &Value, field: &str, message: &str) -> R<()> {
    let mut body = v.as_object().ok_or(message)?.clone();
    let actual = body.remove(field).ok_or(message)?;
    need(
        sha(&actual) && actual == hash(&Value::Object(body))?,
        message,
    )
}
pub(crate) fn ir(operation: &str, args: Value) -> R<Value> {
    let mut args = args;
    args["operation"] = json!(operation);
    crate::portable_ir::run(&args)
}
pub(crate) fn shape_count(v: &Value) -> R<BigUint> {
    let dims = array(v, "numeric shape must have exact nonnegative dimensions")?;
    need(
        dims.iter()
            .all(|n| n.as_u64().is_some_and(|n| n <= 9_007_199_254_740_991)),
        "numeric shape must have exact nonnegative dimensions",
    )?;
    crate::portable_contracts::count_product(dims)?
        .ok_or_else(|| "numeric shape must have exact nonnegative dimensions".into())
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["operation"]) {
        "activation_capture" => {
            crate::portable_activation_validation::capture(
                &v["value"],
                &v["model"],
                &v["max_values"],
            )
            .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "activation" => {
            crate::portable_activation_validation::validate(&v["value"], &v["model"])
                .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "activation_run" => {
            crate::portable_activation_validation::run_contract(
                &v["value"],
                &v["source"],
                &v["model"],
            )
            .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "activation_tensor_binding" => {
            crate::portable_activation_validation::tensor_binding(
                &v["value"],
                &v["model"],
                v["input"] == true,
                &v["entry"],
            )
            .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "activation_values" => {
            crate::portable_activation_validation::values(&v["values"], string(&v["dtype"]))
                .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "activation_coverage" => Ok(crate::portable_activation_validation::coverage(
            &v["requested"],
            &v["tensors"],
            &v["missing"],
            &v["model"],
            &v["entry"],
        )),
        "activation_boundary" => Ok(json!(crate::portable_activation_validation::BOUNDARY)),
        "numerical_source" => {
            crate::portable_numerical_validation::source(&v["value"], &v["model"])
                .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "statistics" => {
            crate::portable_numerical_validation::statistics(&v["value"])
                .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "weight" => {
            crate::portable_numerical_validation::weight(&v["value"], &v["model"])
                .map_err(|e| format!("Numerical IR: {e}"))?;
            Ok(Value::Null)
        }
        "weight_coverage" => {
            crate::portable_numerical_validation::weight_coverage(&v["tensors"], &v["model"])
        }
        "weight_boundary" => Ok(json!(crate::portable_numerical_validation::WEIGHT_BOUNDARY)),
        "model" => {
            crate::portable_model_validation::validate(&v["value"])?;
            Ok(Value::Null)
        }
        "program_links" => {
            crate::portable_model_validation::program_links(&v["value"])?;
            Ok(Value::Null)
        }
        "model_orders" => crate::portable_model_validation::orders(v),
        "model_bindings" => {
            crate::portable_model_validation::bindings(&v["program"], &v["storage"])
        }
        "native_facts" => {
            crate::portable_model_validation::native_facts(&v["value"], &v["artifact_ir_sha256"])
        }
        "artifact" => {
            crate::portable_artifact_validation::validate(&v["value"])?;
            Ok(Value::Null)
        }
        "runtime_overlay" => crate::portable_runtime_overlay::run(v),
        "runtime_reconciliation" => {
            crate::portable_artifact_validation::runtime(
                &v["value"],
                &rows(&v["subjects"])
                    .iter()
                    .filter_map(Value::as_str)
                    .collect(),
            )?;
            Ok(Value::Null)
        }
        _ => Err("Unknown portable semantic validator.".into()),
    }
}
