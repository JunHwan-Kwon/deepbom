use crate::ensure_runtime_allowed;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn preprocess_rgba_to_float32(
    rgba: &[u8],
    channels: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    std_r: f32,
    std_g: f32,
    std_b: f32,
    bgr: bool,
) -> Result<js_sys::Float32Array, JsValue> {
    ensure_runtime_allowed()?;
    let out = pack_rgba_float32(
        rgba,
        channels,
        [mean_r, mean_g, mean_b],
        [std_r, std_g, std_b],
        bgr,
    );
    Ok(js_sys::Float32Array::from(out.as_slice()))
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn preprocess_rgba_to_int8(
    rgba: &[u8],
    channels: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    std_r: f32,
    std_g: f32,
    std_b: f32,
    scale: f32,
    zero_point: i32,
    bgr: bool,
) -> Result<js_sys::Int8Array, JsValue> {
    ensure_runtime_allowed()?;
    let floats = pack_rgba_float32(
        rgba,
        channels,
        [mean_r, mean_g, mean_b],
        [std_r, std_g, std_b],
        bgr,
    );
    let safe_scale = if scale.abs() < f32::EPSILON {
        1.0
    } else {
        scale
    };
    let out: Vec<i8> = floats
        .iter()
        .map(|value| ((*value / safe_scale).round() as i32 + zero_point).clamp(-128, 127) as i8)
        .collect();
    Ok(js_sys::Int8Array::from(out.as_slice()))
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn preprocess_rgba_to_uint8(
    rgba: &[u8],
    channels: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    std_r: f32,
    std_g: f32,
    std_b: f32,
    scale: f32,
    zero_point: i32,
    bgr: bool,
) -> Result<js_sys::Uint8Array, JsValue> {
    ensure_runtime_allowed()?;
    let floats = pack_rgba_float32(
        rgba,
        channels,
        [mean_r, mean_g, mean_b],
        [std_r, std_g, std_b],
        bgr,
    );
    let safe_scale = if scale.abs() < f32::EPSILON {
        1.0
    } else {
        scale
    };
    let out: Vec<u8> = floats
        .iter()
        .map(|value| ((*value / safe_scale).round() as i32 + zero_point).clamp(0, 255) as u8)
        .collect();
    Ok(js_sys::Uint8Array::from(out.as_slice()))
}

pub(crate) fn pack_rgba_float32(
    rgba: &[u8],
    channels: u32,
    mean: [f32; 3],
    std: [f32; 3],
    bgr: bool,
) -> Vec<f32> {
    let channels = channels.clamp(1, 4) as usize;
    let pixels = rgba.len() / 4;
    let mut out = Vec::with_capacity(pixels * channels);
    for px in rgba.chunks_exact(4) {
        let rgb = [
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
            px[3] as f32 / 255.0,
        ];
        if channels == 1 {
            let luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
            out.push((luma - mean[0]) / safe_std(std[0]));
            continue;
        }
        for c in 0..channels {
            let src = match c {
                0 if bgr => 2,
                2 if bgr => 0,
                0..=2 => c,
                _ => 3,
            };
            // The public API has three normalization channels; alpha reuses the third.
            let norm_c = c.min(2);
            out.push((rgb[src] - mean[norm_c]) / safe_std(std[norm_c]));
        }
    }
    out
}

fn safe_std(value: f32) -> f32 {
    if value.abs() < f32::EPSILON {
        1.0
    } else {
        value
    }
}
