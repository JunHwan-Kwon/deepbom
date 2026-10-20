//! Deterministic magnitude-mask preview. Preserves the complete tensor, stable
//! unfolded-position tie breaks, and the existing compensated energy method.
use crate::portable_weight::max_abs as magnitude;
use serde::Serialize;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let y = value - self.correction;
        let t = self.value + y;
        self.correction = (t - self.value) - y;
        self.value = t;
    }
}
fn checked(values: &[f64], rows: u32, columns: u32) -> Result<(), &'static str> {
    if rows == 0
        || rows > 4096
        || columns == 0
        || u64::from(rows) * u64::from(columns) != values.len() as u64
        || values.len() > 2_000_000
    {
        return Err(
            "pruning requires a complete matrix of at most 2,000,000 values and 4,096 channels",
        );
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("pruning requires finite values");
    }
    Ok(())
}
fn removed_count(length: usize, zero: usize, percent: u32) -> usize {
    (length * percent as usize / 100).saturating_sub(zero)
}
fn ranked(values: &[f64]) -> Vec<u32> {
    let mut order: Vec<_> = (0..values.len() as u32)
        .filter(|i| values[*i as usize] != 0.0)
        .collect();
    order.sort_unstable_by(|a, b| {
        values[*a as usize]
            .abs()
            .partial_cmp(&values[*b as usize].abs())
            .unwrap()
            .then(a.cmp(b))
    });
    order
}
fn prepare(values: &[f64], rows: u32, columns: u32) -> Result<(Vec<u32>, Value), &'static str> {
    checked(values, rows, columns)?;
    let order = ranked(values);
    let scale = magnitude(values);
    let zero = values.len() - order.len();
    let mut retained = vec![0.0; order.len() + 1];
    let mut sum = Sum::default();
    for i in (0..order.len()).rev() {
        let v = values[order[i] as usize] / scale;
        sum.add(v * v);
        retained[i] = sum.value;
    }
    let curve:Vec<_>=(0..=100).map(|target|{let k=removed_count(values.len(),zero,target);
        json!({"target_percent":target,"zero_count":zero+k,
            "retained_energy_fraction":if retained[0]!=0.0{Some(retained[k]/retained[0])}else{None}})}).collect();
    Ok((order, json!({"scale":scale,"zeroCount":zero,"curve":curve})))
}
fn simulate(
    values: &[f64],
    rows: u32,
    columns: u32,
    order: &[u32],
    percent: u32,
) -> Result<(Vec<f64>, Vec<u8>, Value), &'static str> {
    checked(values, rows, columns)?;
    if percent > 100 {
        return Err("pruning target must be an integer percentage from 0 to 100");
    }
    if order.len() > values.len() {
        return Err("invalid pruning ranking");
    }
    let zero = values.iter().filter(|v| **v == 0.0).count();
    if zero + order.len() != values.len() {
        return Err("invalid pruning ranking");
    }
    // Verify the prepared rank on each call so callers cannot supply duplicate,
    // stale or arbitrary positions while labeling the preview magnitude-based.
    let mut seen = vec![false; values.len()];
    let mut previous: Option<u32> = None;
    for &i in order {
        if i as usize >= values.len() || seen[i as usize] || values[i as usize] == 0.0 {
            return Err("invalid pruning ranking");
        }
        if let Some(p) = previous {
            let a = values[p as usize].abs();
            let b = values[i as usize].abs();
            if a > b || a == b && p >= i {
                return Err("invalid pruning ranking");
            }
        }
        seen[i as usize] = true;
        previous = Some(i);
    }
    let k = removed_count(values.len(), zero, percent);
    let mut candidate = values.to_vec();
    let mut mask = vec![0u8; values.len()];
    for &position in &order[..k] {
        candidate[position as usize] = 0.0;
        mask[position as usize] = 1;
    }
    let channels:Vec<_>=(0..rows as usize).map(|row|{
        let start=row*columns as usize;let end=start+columns as usize;
        let a=&values[start..end];let b=&candidate[start..end];let scale=magnitude(a);
        let mut original=Sum::default();let mut kept=Sum::default();let mut existing=0;let mut removed=0;
        for j in 0..a.len(){existing+=usize::from(a[j]==0.0);removed+=mask[start+j] as usize;
            if scale!=0.0{let v=a[j]/scale;original.add(v*v);let v=b[j]/scale;kept.add(v*v);}}
        json!({"index":row,"count":a.len(),"original_zero_count":existing,"newly_zeroed_count":removed,
            "candidate_zero_count":existing+removed,"retained_energy_fraction":if original.value!=0.0{Some(kept.value/original.value)}else{None}})
    }).collect();
    Ok((
        candidate,
        mask,
        json!({"k":k,"channels":channels,"target_zero_count":values.len()*percent as usize/100,
        "achieved_zero_fraction":(zero+k) as f64/values.len() as f64,
        "last_removed_absolute_value":if k>0{Some(values[order[k-1] as usize].abs())}else{None}}),
    ))
}
fn encode(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn numerical_pruning_prepare(
    values: &[f64],
    rows: u32,
    columns: u32,
) -> Result<JsValue, JsValue> {
    let (order, body) = prepare(values, rows, columns).map_err(JsValue::from_str)?;
    let object = encode(&body)?;
    js_sys::Reflect::set(
        &object,
        &JsValue::from_str("order"),
        &js_sys::Uint32Array::from(order.as_slice()),
    )?;
    Ok(object)
}
#[wasm_bindgen]
pub fn numerical_pruning_simulate(
    values: &[f64],
    rows: u32,
    columns: u32,
    order: &[u32],
    percent: u32,
) -> Result<JsValue, JsValue> {
    let (candidate, mask, body) =
        simulate(values, rows, columns, order, percent).map_err(JsValue::from_str)?;
    let object = encode(&body)?;
    js_sys::Reflect::set(
        &object,
        &JsValue::from_str("candidate"),
        &js_sys::Float64Array::from(candidate.as_slice()),
    )?;
    js_sys::Reflect::set(
        &object,
        &JsValue::from_str("mask"),
        &js_sys::Uint8Array::from(mask.as_slice()),
    )?;
    Ok(object)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_and_existing_zeros() {
        let values = [0.0, -0.0, 1.0, -1.0, 2.0, -3.0, 4.0, -5.0];
        let (order, body) = prepare(&values, 2, 4).unwrap();
        assert_eq!(order, vec![2, 3, 4, 5, 6, 7]);
        let (candidate, mask, result) = simulate(&values, 2, 4, &order, 50).unwrap();
        assert_eq!(candidate, vec![0.0, -0.0, 0.0, 0.0, 2.0, -3.0, 4.0, -5.0]);
        assert_eq!(mask, vec![0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(result["k"], 2);
        assert_eq!(body["curve"][100]["retained_energy_fraction"], 0.0);
        assert!(simulate(&values, 2, 4, &[3, 2, 4, 5, 6, 7], 50).is_err());
    }
}
