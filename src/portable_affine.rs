//! Complete scalar affine reconstruction. Source discovery remains in format
//! adapters; the contract, indexing, range and reconstruction have one owner.
use serde::Serialize;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

fn numbers(value: Option<&Value>) -> Option<Vec<f64>> {
    value?.as_array()?.iter().map(Value::as_f64).collect()
}
fn integer(value: f64) -> bool {
    value.is_finite() && value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_991.0
}
fn reconstruct(
    values: &[f64],
    shape: &[f64],
    dtype: &str,
    contract: &Value,
    order: &str,
) -> Result<(Vec<f64>, Value), &'static str> {
    let unsigned = if dtype.starts_with("UINT") {
        true
    } else if dtype.starts_with("INT") {
        false
    } else {
        return Err("invalid_or_unsupported_affine_contract");
    };
    let bits = crate::scalar_types_generated::scalar_dtype_bits(dtype)
        .filter(|b| matches!(b, 2 | 4 | 8 | 16 | 32))
        .ok_or("invalid_or_unsupported_affine_contract")? as i32;
    let scales = numbers(contract.get("scales")).ok_or("invalid_or_unsupported_affine_contract")?;
    let zeros =
        numbers(contract.get("zero_points")).ok_or("invalid_or_unsupported_affine_contract")?;
    if scales.is_empty()
        || scales.len() > 10_000_000
        || scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || !zeros.is_empty() && zeros.len() != scales.len()
        || zeros.iter().any(|v| !integer(*v))
    {
        return Err("invalid_or_unsupported_affine_contract");
    }
    let minimum = if unsigned {
        0.0
    } else {
        -(2_f64.powi(bits - 1))
    };
    let maximum = 2_f64.powi(if unsigned { bits } else { bits - 1 }) - 1.0;
    if zeros.iter().any(|v| *v < minimum || *v > maximum) {
        return Err("zero_point_outside_encoding_range");
    }
    let parameterization = contract
        .get("parameterization")
        .and_then(Value::as_str)
        .unwrap_or("");
    let axis = contract.get("axis").and_then(Value::as_f64);
    if parameterization != "per_tensor"
        && !axis.is_some_and(|v| integer(v) && v >= 0.0 && v < (shape.len() as f64))
    {
        return Err("invalid_quantization_axis");
    }
    let axis = axis.unwrap_or(0.0) as usize;
    if parameterization == "per_axis" && scales.len() as f64 != shape[axis]
        || parameterization == "per_tensor" && scales.len() != 1
        || !["per_tensor", "per_axis", "blocked"].contains(&parameterization)
    {
        return Err("quantization_cardinality_mismatch");
    }
    if !["first_axis_fastest", "last_axis_fastest"].contains(&order)
        || shape.len() > 65536
        || values.len() > 10_000_000
        || shape.iter().any(|v| !integer(*v) || *v < 0.0)
    {
        return Err("invalid_tensor_shape_or_storage_order");
    }
    let dimensions: Vec<usize> = shape.iter().map(|v| *v as usize).collect();
    let length = if dimensions.contains(&0) {
        Some(0)
    } else {
        dimensions.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
    };
    if length != Some(values.len()) {
        return Err("invalid_tensor_shape_or_storage_order");
    }
    let block = contract.get("block_size").and_then(Value::as_f64);
    let scale_shape = numbers(contract.get("scale_shape"));
    if parameterization == "blocked" {
        let valid = block.is_some_and(|v| integer(v) && v > 0.0)
            && scale_shape.as_ref().is_some_and(|s| {
                s.len() == shape.len()
                    && s.iter().enumerate().all(|(i, d)| {
                        *d == if i == axis {
                            (shape[i] / block.unwrap()).ceil()
                        } else {
                            shape[i]
                        }
                    })
                    && s.iter().product::<f64>() == scales.len() as f64
            });
        if !valid {
            return Err("blocked_quantization_cardinality_mismatch");
        }
    }
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    let axes: Vec<_> = if order == "first_axis_fastest" {
        (0..shape.len()).collect()
    } else {
        (0..shape.len()).rev().collect()
    };
    for i in axes {
        strides[i] = stride;
        stride = stride
            .checked_mul(dimensions[i])
            .ok_or("invalid_tensor_shape_or_storage_order")?;
    }
    let mut output = Vec::with_capacity(values.len());
    let (mut min, mut max, mut lower, mut upper) =
        (f64::INFINITY, f64::NEG_INFINITY, 0usize, 0usize);
    for (i, &code) in values.iter().enumerate() {
        if !integer(code) || code < minimum || code > maximum {
            return Err("stored_code_outside_encoding_range");
        }
        let mut index = 0;
        if parameterization == "per_axis" {
            index = (i / strides[axis]) % dimensions[axis];
        } else if parameterization == "blocked" {
            for dim in 0..shape.len() {
                let coordinate = (i / strides[dim]) % dimensions[dim];
                let coordinate = if dim == axis {
                    (coordinate as f64 / block.unwrap()).floor() as usize
                } else {
                    coordinate
                };
                index = index * (scale_shape.as_ref().unwrap()[dim] as usize) + coordinate;
            }
        }
        let real = (code - zeros.get(index).copied().unwrap_or(0.0)) * scales[index];
        if !real.is_finite() {
            return Err("dequantization_overflow");
        }
        output.push(real);
        min = min.min(code);
        max = max.max(code);
        lower += usize::from(code == minimum);
        upper += usize::from(code == maximum);
    }
    let zero_points = if zeros.is_empty() {
        vec![0.0; scales.len()]
    } else {
        zeros
    };
    Ok((
        output,
        json!({"status":"assessed","method":"affine_scale_times_code_minus_zero_point",
        "parameterization":parameterization,"axis":if parameterization=="per_tensor"{None}else{Some(axis)},
        "block_size":contract.get("block_size").unwrap_or(&Value::Null),"scales":scales,"zero_points":zero_points,
        "encoding_minimum":minimum,"encoding_maximum":maximum,
        "observed_code_minimum":if values.is_empty(){None}else{Some(min)},"observed_code_maximum":if values.is_empty(){None}else{Some(max)},
        "range_utilization":if values.is_empty(){None}else{Some((max-min)/(maximum-minimum))},
        "lower_endpoint_count":lower,"upper_endpoint_count":upper,
        "endpoint_interpretation":"observed_endpoint_occupancy_not_proof_of_clipping",
        "reconstruction_error":{"status":"not_assessed","reason":"original_reference_required"}}),
    ))
}

#[wasm_bindgen]
pub fn numerical_affine(
    values: &[f64],
    shape: &[f64],
    dtype: &str,
    contract: JsValue,
    order: &str,
) -> Result<JsValue, JsValue> {
    let contract: Value = serde_wasm_bindgen::from_value(contract).unwrap_or(Value::Null);
    let result = reconstruct(values, shape, dtype, &contract, order);
    let object = js_sys::Object::new();
    let (data, report) = match result {
        Ok((data, report)) => (Some(data), report),
        Err(reason) => (None, json!({"status":"not_assessed","reason":reason})),
    };
    let report = report
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    js_sys::Reflect::set(&object, &JsValue::from_str("report"), &report)?;
    let array = data
        .map(|data| JsValue::from(js_sys::Float64Array::from(data.as_slice())))
        .unwrap_or(JsValue::NULL);
    js_sys::Reflect::set(&object, &JsValue::from_str("values"), &array)?;
    Ok(object.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_axis_and_overflow() {
        let contract = json!({"scales":[0.5,2.0],"zero_points":[-1,1],"axis":0,"parameterization":"per_axis","block_size":null});
        let (data, report) = reconstruct(
            &[0.0, 2.0, -1.0, 3.0],
            &[2.0, 2.0],
            "INT8",
            &contract,
            "last_axis_fastest",
        )
        .unwrap();
        assert_eq!(data, vec![0.5, 1.5, -4.0, 4.0]);
        assert_eq!(report["lower_endpoint_count"], 0);
        let huge = json!({"scales":[f64::MAX],"zero_points":[],"parameterization":"per_tensor"});
        assert_eq!(
            reconstruct(&[2.0], &[1.0], "INT8", &huge, "last_axis_fastest").unwrap_err(),
            "dequantization_overflow"
        );
    }
    #[test]
    fn bad_shapes_are_not_partial_reconstructions() {
        let c = json!({"scales":[0.5],"zero_points":[],"parameterization":"per_tensor"});
        assert_eq!(
            reconstruct(&[1.0], &[2.0], "UINT8", &c, "last_axis_fastest").unwrap_err(),
            "invalid_tensor_shape_or_storage_order"
        );
    }
}
