//! Standard-domain tensor inference. Unknown data is never copied as an exact value.
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::sync::OnceLock;
fn groups() -> &'static Value {
    static GROUPS: OnceLock<Value> = OnceLock::new();
    GROUPS.get_or_init(|| {
        serde_json::from_str(include_str!("onnx_shape_groups.json"))
            .expect("checked shape registry")
    })
}
pub(crate) fn group(name: &str, op: &str) -> bool {
    rows(&groups()[name]).iter().any(|x| x == op)
}
pub(crate) fn shape(t: &Value) -> Value {
    if t["contractStatus"] == "invalid" || rank(t).is_none() {
        Value::Null
    } else {
        json!(dimensions(t))
    }
}
pub(crate) fn kd(t: &Value) -> &str {
    if t["contractStatus"] == "invalid"
        || !string(&t["valueKind"]).is_empty() && t["valueKind"] != "tensor"
    {
        ""
    } else {
        dtype(t)
    }
}
pub(crate) fn attr_ints(n: &Value, k: &str) -> Vec<Value> {
    let a = &attr(n, k)["ints"];
    if rows(a).iter().all(|n| safe(n).is_some()) {
        rows(a).to_vec()
    } else {
        vec![]
    }
}
pub(crate) fn ai_exact(n: &Value, k: &str, default: i64) -> i64 {
    safe(&attr(n, k)["i"])
        .or_else(|| safe(&attr(n, k)["ints"][0]))
        .or_else(|| {
            attr(n, k)["iExactDecimal"]
                .as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .filter(|n| n.unsigned_abs() <= SAFE as u64)
        })
        .unwrap_or(default)
}
pub(crate) fn static_numbers(t: &Value) -> Option<Vec<Value>> {
    if t["staticValuesComplete"] != true {
        return None;
    }
    let vals = t["staticValues"].as_array()?;
    if vals
        .iter()
        .all(|v| v.is_boolean() || v.as_f64().is_some_and(f64::is_finite))
    {
        Some(
            vals.iter()
                .map(|v| {
                    v.as_bool()
                        .map(|b| json!(if b { 1 } else { 0 }))
                        .unwrap_or_else(|| v.clone())
                })
                .collect(),
        )
    } else {
        None
    }
}
pub(crate) fn exact_dims(t: &Value) -> Option<Vec<Value>> {
    if t["staticDimensionValuesComplete"] == true {
        let vals = t["staticDimensionValues"].as_array()?;
        return Some(
            vals.iter()
                .map(|d| {
                    if d["kind"] == "value" {
                        safe(&d["value"]).map(dim).unwrap_or_else(unknown)
                    } else {
                        normalize_dimension(d)
                    }
                })
                .collect(),
        );
    }
    static_numbers(t)?
        .iter()
        .map(|v| safe(v).map(dim))
        .collect()
}
pub(crate) fn inherited(t: &Value, op: &str) -> Value {
    if t["initializerIntegerValuesExactComplete"] == true
        && t["initializerIntegerValuesExactDecimals"].is_array()
    {
        return json!({"exactDecimals":t["initializerIntegerValuesExactDecimals"],"source":op});
    }
    if let Some(d) = exact_dims(t) {
        json!({"dimensions":d,"source":op})
    } else if let Some(n) = static_numbers(t) {
        json!({"values":n,"source":op})
    } else {
        Value::Null
    }
}
pub(crate) fn attach_static(p: &mut Value, values: &Value, op: &str) {
    if let Some(exact) = values["exactDecimals"].as_array() {
        let parsed = exact
            .iter()
            .map(|v| v.as_str().and_then(|s| s.parse::<i64>().ok()))
            .collect::<Option<Vec<_>>>();
        if let Some(parsed) = parsed {
            p["initializerIntegerValuesExactComplete"] = json!(true);
            p["initializerIntegerValuesExactDecimals"] =
                json!(parsed.iter().map(ToString::to_string).collect::<Vec<_>>());
            let safe = parsed.iter().all(|n| n.unsigned_abs() <= SAFE as u64);
            p["staticValuesSource"] = json!(values["source"].as_str().unwrap_or(op));
            p["staticValuesStatus"] = json!(if safe {
                "assessed_exact_static_data"
            } else {
                "complete_exact_int64_decimal_only"
            });
            p["staticValuesComplete"] = json!(safe);
            p["staticValues"] = if safe { json!(parsed) } else { json!([]) };
        }
    }
    if values["values"].is_array() {
        p["staticValuesStatus"] = json!("assessed_exact_static_data");
        p["staticValuesComplete"] = json!(true);
        p["staticValues"] = values["values"].clone();
        p["staticValuesSource"] = json!(values["source"].as_str().unwrap_or(op));
    }
    if let Some(dims) = values["dimensions"].as_array() {
        let nums = dims.iter().map(dv).collect::<Option<Vec<_>>>();
        if let Some(nums) = nums {
            p["staticValuesStatus"] = json!("assessed_exact_static_data");
            p["staticValuesComplete"] = json!(true);
            p["staticValues"] = json!(nums);
            p["staticValuesSource"] = json!(values["source"].as_str().unwrap_or(op));
        }
        p["staticDimensionValuesStatus"] = json!("assessed_exact_symbolic_shape_data");
        p["staticDimensionValuesComplete"] = json!(true);
        p["staticDimensionValues"] = json!(dims);
        p["staticDimensionValuesSource"] = json!(values["source"].as_str().unwrap_or(op));
    }
    if values["canonicalTexts"].is_array() {
        p["staticValuesStatus"] = json!("complete_canonical_text_only_non_finite_or_unsafe_value");
        p["staticValuesComplete"] = json!(false);
        p["staticValues"] = json!([]);
        p["staticValuesCanonicalTextComplete"] = json!(true);
        p["staticValuesCanonicalTexts"] = values["canonicalTexts"].clone();
        p["staticValuesSource"] = json!(values["source"].as_str().unwrap_or(op));
    }
}
pub(crate) fn set(
    v: &Value,
    out: &mut Vec<Value>,
    i: usize,
    s: &Value,
    dt: &str,
    values: Value,
) -> bool {
    let Some(dims) = s.as_array() else {
        return false;
    };
    if !truth(&v["node"]["outputs"][i]) {
        return false;
    }
    let t = tensor_type(dt, dims, &v["tensor_type_ids"]);
    let mut p = json!({"dtype":dt,"shape":t["shape"],"shapeDeclared":true,"typeProto":t});
    attach_static(&mut p, &values, string(&v["node"]["opType"]));
    out.push(json!([v["node"]["outputs"][i], p]));
    true
}
pub(crate) fn kernel(op: &str, mut args: Value) -> R<Value> {
    args["operation"] = json!(op);
    crate::portable_shape::run(&args)
}
pub(crate) fn spatial(n: &Value) -> Value {
    let mut a = json!({"auto_pad":astr(n,"auto_pad","NOTSET"),"ceil_mode":ai_exact(n,"ceil_mode",0),"group":ai_exact(n,"group",1)});
    for k in [
        "kernel_shape",
        "strides",
        "dilations",
        "pads",
        "output_padding",
        "output_shape",
    ] {
        if !attr(n, k).is_null() {
            a[k] = json!(attr_ints(n, k));
        }
    }
    a
}
fn single(v: &Value, s: Value, dt: &str, vals: Value, reason: &str) -> Value {
    let mut out = vec![];
    if set(v, &mut out, 0, &s, dt, vals) {
        result(out)
    } else {
        unresolved(reason)
    }
}
fn type_name(v: &Value, id: i64) -> String {
    v["tensor_type_ids"]
        .as_object()
        .and_then(|m| {
            m.iter()
                .find(|(_, n)| n.as_i64() == Some(id))
                .map(|(k, _)| k.clone())
        })
        .unwrap_or_else(|| {
            if id == 0 {
                "UNDEFINED".into()
            } else {
                format!("TYPE_{id}")
            }
        })
}
fn known(d: &Value) -> bool {
    dv(d).is_some_and(|n| n >= 0) || d["kind"] == "symbolic" && !string(&d["parameter"]).is_empty()
}
fn cast_values(t: &Value, dt: &str, op: &str) -> Value {
    let source = kd(t);
    if dt == source {
        return inherited(t, op);
    }
    let bounds = match dt {
        "BOOL" => Some((0i128, 1i128)),
        "INT8" => Some((-128, 127)),
        "UINT8" => Some((0, 255)),
        "INT16" => Some((-32768, 32767)),
        "UINT16" => Some((0, 65535)),
        "INT32" => Some((i32::MIN as i128, i32::MAX as i128)),
        "UINT32" => Some((0, u32::MAX as i128)),
        "INT64" => Some((i64::MIN as i128, i64::MAX as i128)),
        "UINT64" => Some((0, u64::MAX as i128)),
        _ => None,
    };
    let Some((min, max)) = bounds else {
        return Value::Null;
    };
    if let Some(values) = static_numbers(t) {
        if values
            .iter()
            .all(|n| safe(n).is_some_and(|n| (n as i128) >= min && (n as i128) <= max))
        {
            return json!({"values":values,"source":op});
        }
    }
    if dt == "INT64" && ["INT32", "INT64"].contains(&source) && exact_dims(t).is_some() {
        return inherited(t, op);
    }
    Value::Null
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let op = string(&n["opType"]);
    let (x, y) = (input(v, 0), input(v, 1));
    let dt = if !kd(x).is_empty() {
        kd(x)
    } else if !kd(y).is_empty() {
        kd(y)
    } else {
        "UNKNOWN"
    };
    let mut out = vec![];
    if group("SAME_SHAPE", op) {
        let values = if op == "Identity" {
            inherited(x, op)
        } else {
            Value::Null
        };
        if !set(v, &mut out, 0, &shape(x), dt, values) {
            return Ok(unresolved("input_shape_unknown"));
        }
        if op == "Dropout" {
            set(v, &mut out, 1, &shape(x), "BOOL", Value::Null);
        }
        return Ok(result(out));
    }
    if ["CumSum", "ScatterElements", "ScatterND"].contains(&op) {
        return Ok(single(
            v,
            shape(x),
            dt,
            Value::Null,
            match op {
                "CumSum" => "cumsum_input_shape_unknown",
                "ScatterElements" => "scatter_elements_input_shape_unknown",
                _ => "scatter_nd_data_shape_unknown",
            },
        ));
    }
    if group("BOOL_SAME_SHAPE", op) {
        return Ok(single(
            v,
            shape(x),
            "BOOL",
            Value::Null,
            "input_shape_unknown",
        ));
    }
    if op == "Cast" || op == "CastLike" {
        let od = if op == "Cast" {
            type_name(v, ai_exact(n, "to", 0))
        } else {
            kd(y).into()
        };
        return Ok(single(
            v,
            shape(x),
            &od,
            cast_values(x, &od, op),
            "cast_shape_or_dtype_unknown",
        ));
    }
    if [
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
    ]
    .contains(&op)
    {
        let declared = ai_exact(n, "output_dtype", 0);
        let od = if op == "DequantizeLinear" {
            if declared > 0 {
                type_name(v, declared)
            } else {
                "FLOAT32".into()
            }
        } else if op == "DynamicQuantizeLinear" {
            "UINT8".into()
        } else if !kd(input(v, 2)).is_empty() {
            kd(input(v, 2)).into()
        } else if declared > 0 {
            type_name(v, declared)
        } else {
            "UINT8".into()
        };
        if !set(v, &mut out, 0, &shape(x), &od, Value::Null) {
            return Ok(unresolved(match op {
                "QuantizeLinear" => "quantize_input_shape_unknown",
                "DequantizeLinear" => "dequantize_input_shape_unknown",
                _ => "dynamic_quantize_input_shape_unknown",
            }));
        }
        if op == "DynamicQuantizeLinear" {
            set(v, &mut out, 1, &json!([]), "FLOAT32", Value::Null);
            set(v, &mut out, 2, &json!([]), "UINT8", Value::Null);
        }
        return Ok(result(out));
    }
    if ["Conv", "ConvInteger", "QLinearConv", "ConvTranspose"].contains(&op) {
        let w = if op == "QLinearConv" { input(v, 3) } else { y };
        if op != "ConvTranspose" && rank(x).is_some() && rank(w).is_some() && rank(x) != rank(w) {
            return Ok(
                json!({"status":"invalid","outputs":[],"reason":"conv_input_weight_rank_mismatch","details":{"input_rank":rank(x),"weight_rank":rank(w)}}),
            );
        }
        let s = if rank(w).is_none() {
            Value::Null
        } else {
            kernel(
                if op == "ConvTranspose" {
                    "conv_transpose"
                } else {
                    "conv"
                },
                json!({"input":shape(x),"weight":w["shape"],"attributes":spatial(n)}),
            )?
        };
        let od = if op == "ConvInteger" {
            "INT32"
        } else if op == "QLinearConv" {
            if !kd(input(v, 7)).is_empty() {
                kd(input(v, 7))
            } else {
                "UINT8"
            }
        } else {
            dt
        };
        let r = if op == "ConvTranspose" {
            "conv_transpose"
        } else {
            "conv"
        };
        if !set(v, &mut out, 0, &s, od, Value::Null) {
            return Ok(unresolved(&format!("{r}_shape_contract_unresolved")));
        }
        let mut value = result(out);
        if rows(&s).iter().any(|d| !known(d)) {
            value["reason"] = json!(format!(
                "{r}_runtime_dimensions_unbound_rank_and_output_channel_inferred"
            ));
        }
        return Ok(value);
    }
    if ["MaxPool", "AveragePool"].contains(&op) {
        let k = attr_ints(n, "kernel_shape");
        if k.is_empty() || k.iter().any(|n| safe(n).is_none_or(|n| n <= 0)) {
            return Ok(unresolved("pool_shape_contract_unresolved"));
        }
        let ds = if rank(x).is_some() {
            dimensions(x)
        } else {
            vec![unknown(); k.len() + 2]
        };
        if ds.len() != k.len() + 2 {
            return Ok(unresolved("pool_shape_contract_unresolved"));
        }
        let spatial = kernel(
            "spatial",
            json!({"input":&ds[2..],"kernels":k,"attributes":spatial(n)}),
        )?;
        if spatial.is_null() {
            return Ok(unresolved("pool_shape_contract_unresolved"));
        }
        let mut ds = ds[..2].to_vec();
        ds.extend_from_slice(rows(&spatial));
        if !set(v, &mut out, 0, &json!(ds), dt, Value::Null) {
            return Ok(unresolved("pool_shape_contract_unresolved"));
        }
        if op == "MaxPool" {
            set(v, &mut out, 1, &json!(ds), "INT64", Value::Null);
        }
        let mut result = result(out);
        if ds.iter().any(|d| !known(d)) {
            result["reason"] =
                json!("pool_runtime_dimensions_unbound_rank_and_nonspatial_axes_inferred");
        }
        return Ok(result);
    }
    if ["GlobalAveragePool", "GlobalMaxPool"].contains(&op) {
        if rank(x).is_none_or(|n| n < 3) {
            return Ok(unresolved("global_pool_input_rank_unknown"));
        }
        let mut s = dimensions(x);
        for d in &mut s[2..] {
            *d = dim(1);
        }
        return Ok(single(
            v,
            json!(s),
            dt,
            Value::Null,
            "global_pool_output_unresolved",
        ));
    }
    if op == "Flatten" {
        return Ok(single(
            v,
            kernel(
                "flatten",
                json!({"input":shape(x),"axis":ai_exact(n,"axis",1)}),
            )?,
            dt,
            inherited(x, op),
            "flatten_input_shape_unknown",
        ));
    }
    if op == "Gemm" {
        return Ok(single(
            v,
            kernel(
                "gemm",
                json!({"left":shape(x),"right":shape(y),"trans_a":ai_exact(n,"transA",0)==1,"trans_b":ai_exact(n,"transB",0)==1}),
            )?,
            dt,
            Value::Null,
            "gemm_shape_contract_unresolved",
        ));
    }
    if ["MatMul", "MatMulInteger", "QLinearMatMul"].contains(&op) {
        let right = if op == "QLinearMatMul" {
            input(v, 3)
        } else {
            y
        };
        let od = if op == "MatMulInteger" {
            "INT32"
        } else if op == "QLinearMatMul" {
            if kd(input(v, 7)).is_empty() {
                "UINT8"
            } else {
                kd(input(v, 7))
            }
        } else {
            dt
        };
        return Ok(single(
            v,
            kernel("matmul", json!({"left":shape(x),"right":shape(right)}))?,
            od,
            Value::Null,
            "matmul_shape_contract_unresolved",
        ));
    }
    if op == "LayerNormalization" {
        if rank(x).is_none() {
            return Ok(unresolved("layer_normalization_input_rank_unknown"));
        }
        let ds = dimensions(x);
        let raw = ai_exact(n, "axis", -1);
        let axis = if raw < 0 { raw + ds.len() as i64 } else { raw };
        if axis < 0 || axis >= ds.len() as i64 {
            return Ok(unresolved("layer_normalization_axis_out_of_range"));
        }
        if !set(v, &mut out, 0, &json!(ds), dt, Value::Null) {
            return Ok(unresolved("layer_normalization_output_unresolved"));
        }
        let stash = type_name(v, ai_exact(n, "stash_type", 1));
        let reduced = ds
            .iter()
            .enumerate()
            .map(|(i, d)| if i < axis as usize { d.clone() } else { dim(1) })
            .collect::<Vec<_>>();
        for (i, error) in [
            (1, "layer_normalization_mean_unresolved"),
            (2, "layer_normalization_inv_std_dev_unresolved"),
        ] {
            if truth(&n["outputs"][i]) && !set(v, &mut out, i, &json!(reduced), &stash, Value::Null)
            {
                return Ok(unresolved(error));
            }
        }
        return Ok(result(out));
    }
    if ["RandomNormalLike", "RandomUniformLike"].contains(&op) {
        let od = if !attr(n, "dtype").is_null() {
            type_name(v, ai_exact(n, "dtype", 0))
        } else {
            kd(x).into()
        };
        if od.is_empty() {
            return Ok(unresolved("random_like_shape_or_dtype_unknown"));
        }
        return Ok(single(
            v,
            shape(x),
            &od,
            Value::Null,
            "random_like_shape_or_dtype_unknown",
        ));
    }
    if op == "Transpose" {
        if rank(x).is_none() {
            return Ok(unresolved("transpose_input_shape_unknown"));
        }
        let ds = dimensions(x);
        let raw = attr_ints(n, "perm");
        let perm = if raw.is_empty() {
            Some((0..ds.len()).rev().collect::<Vec<_>>())
        } else {
            raw.iter()
                .map(|v| {
                    safe(v)
                        .filter(|n| *n >= 0 && *n < ds.len() as i64)
                        .map(|n| n as usize)
                })
                .collect::<Option<Vec<_>>>()
        };
        let Some(perm) = perm.filter(|p| {
            p.len() == ds.len()
                && p.iter().collect::<std::collections::BTreeSet<_>>().len() == p.len()
        }) else {
            return Ok(unresolved("transpose_perm_invalid"));
        };
        return Ok(single(
            v,
            json!(perm.iter().map(|i| ds[*i].clone()).collect::<Vec<_>>()),
            dt,
            Value::Null,
            "transpose_output_unresolved",
        ));
    }
    crate::portable_onnx_tensor::run(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unary(op: &str) -> Value {
        json!({"node":{"opType":op,"inputs":["x"],"outputs":["y"],"attributes":[]},"tensors":{"x":{"dtype":"INT32","shape":[2],"shapeDeclared":true,"staticValuesComplete":true,"staticValues":[-2,5]}},"tensor_type_ids":{"INT32":6,"INT64":7,"FLOAT32":1}})
    }
    #[test]
    fn unary_shape_equality_is_not_value_identity() {
        for op in ["Relu", "Sigmoid", "Abs", "Neg", "Dropout"] {
            let r = run(&unary(op)).unwrap();
            assert_eq!(r["outputs"][0][1]["shape"], json!([2]));
            assert!(
                r["outputs"][0][1].get("staticValues").is_none(),
                "{op} cannot inherit unchanged values"
            );
        }
        assert_eq!(
            run(&unary("Identity")).unwrap()["outputs"][0][1]["staticValues"],
            json!([-2, 5])
        );
    }
    #[test]
    fn narrowing_cast_does_not_copy_out_of_range_values() {
        let mut v = unary("Cast");
        v["node"]["attributes"] = json!([["to",{"i":3,"type":2}]]);
        v["tensor_type_ids"]["INT8"] = json!(3);
        v["tensors"]["x"]["staticValues"] = json!([1000, -2]);
        let r = run(&v).unwrap();
        assert_eq!(r["outputs"][0][1]["dtype"], "INT8");
        assert!(r["outputs"][0][1].get("staticValues").is_none());
        v["node"]["attributes"][0][1]["i"] = json!(7);
        assert_eq!(
            run(&v).unwrap()["outputs"][0][1]["staticValues"],
            json!([1000, -2])
        );
    }
}
