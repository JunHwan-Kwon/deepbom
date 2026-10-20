//! Explicit per-sample quality metrics. No model execution or task inference.
//! Binary64 method; the pinned external adapter remains a separate oracle.
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

fn fail(message: &str) -> JsValue {
    JsValue::from_str(message)
}
fn positive(v: &Value, key: &str) -> Result<f64, JsValue> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite() && *x > 0.0)
        .ok_or_else(|| fail("metric requires positive finite options"))
}
fn reflection(mut x: isize, size: usize) -> usize {
    let end = size as isize - 1;
    while x < 0 || x > end {
        if x < 0 {
            x = -x;
        }
        if x > end {
            x = 2 * end - x;
        }
    }
    x as usize
}
fn filter_axis(
    values: &[f64],
    shape: &[usize],
    axis: usize,
    kernel: &[f64],
    pad: usize,
) -> (Vec<f64>, Vec<usize>) {
    let stride: usize = shape[axis + 1..].iter().product();
    let inner = shape[axis];
    let outer = values.len() / (inner * stride);
    let output = inner + 2 * pad + 1 - kernel.len();
    let mut result = vec![0.0; outer * output * stride];
    for o in 0..outer {
        for p in 0..output {
            for s in 0..stride {
                let mut total = 0.0;
                for (k, w) in kernel.iter().enumerate() {
                    let q = reflection((p + k) as isize - pad as isize, inner);
                    total += values[(o * inner + q) * stride + s] * w;
                }
                result[(o * output + p) * stride + s] = total;
            }
        }
    }
    let mut dimensions = shape.to_vec();
    dimensions[axis] = output;
    (result, dimensions)
}

#[wasm_bindgen]
pub fn numerical_quality_metric(
    metric: &str,
    prediction: &[f64],
    target: &[f64],
    shape: &[u32],
    settings: &str,
) -> Result<f64, JsValue> {
    if prediction.len() != target.len() || prediction.len() > 8_388_608 || prediction.is_empty() {
        return Err(fail("metric input length/budget mismatch"));
    }
    if prediction.iter().chain(target).any(|v| !v.is_finite()) {
        return Err(fail("metric inputs must be finite"));
    }
    let shape: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
    let count = shape.iter().try_fold(
        1usize,
        |n, d| if *d == 0 { None } else { n.checked_mul(*d) },
    );
    if count != Some(prediction.len()) {
        return Err(fail("metric shape does not match payload"));
    }
    if settings.len() > 4096 {
        return Err(fail("metric settings exceed budget"));
    }
    let options: Value =
        serde_json::from_str(settings).map_err(|_| fail("invalid metric settings"))?;
    match metric {
        "generalized_dice" => dice(prediction, target, &options),
        "ssim" => ssim(prediction, target, &shape, &options),
        _ => Err(fail("unsupported built-in metric")),
    }
}
fn dice(p: &[f64], t: &[f64], o: &Value) -> Result<f64, JsValue> {
    let classes = o["num_classes"]
        .as_u64()
        .filter(|n| *n >= 2 && *n <= 65536)
        .ok_or_else(|| fail("invalid Dice classes"))? as usize;
    let background = o["include_background"]
        .as_bool()
        .ok_or_else(|| fail("missing background policy"))?;
    if o["input_format"] != "index" || o["empty_class_policy"] != "torchmetrics_1.9.0" {
        return Err(fail("unsupported Dice input/empty-class policy"));
    }
    let (mut predictions, mut targets, mut intersections) = (
        vec![0u64; classes],
        vec![0u64; classes],
        vec![0u64; classes],
    );
    for (&a, &b) in p.iter().zip(t) {
        if a < 0.0
            || b < 0.0
            || a.fract() != 0.0
            || b.fract() != 0.0
            || a >= classes as f64
            || b >= classes as f64
        {
            return Err(fail("Dice class outside vocabulary"));
        }
        predictions[a as usize] += 1;
        targets[b as usize] += 1;
        if a == b {
            intersections[a as usize] += 1;
        }
    }
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for c in if background { 0 } else { 1 }..classes {
        let n = targets[c] as f64;
        // Qualified adapter computes one sample at a time. Thus the maximum
        // finite weight for an absent class is zero (not another class's max).
        let w = match o["weight_type"].as_str() {
            Some("linear") => 1.0,
            Some("simple") => {
                if n == 0.0 {
                    0.0
                } else {
                    1.0 / n
                }
            }
            Some("square") => {
                if n == 0.0 {
                    0.0
                } else {
                    1.0 / (n * n)
                }
            }
            _ => return Err(fail("invalid Dice weighting")),
        };
        numerator += 2.0 * intersections[c] as f64 * w;
        denominator += (predictions[c] + targets[c]) as f64 * w;
    }
    Ok(if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    })
}
fn ssim(p: &[f64], t: &[f64], shape: &[usize], o: &Value) -> Result<f64, JsValue> {
    if shape.len() != 3 && shape.len() != 4 {
        return Err(fail("SSIM sample must have CHW or CDHW shape"));
    }
    let (range, sigma, k1, k2) = (
        positive(o, "data_range")?,
        positive(o, "sigma")?,
        positive(o, "k1")?,
        positive(o, "k2")?,
    );
    let gaussian = o["gaussian_kernel"]
        .as_bool()
        .ok_or_else(|| fail("missing kernel policy"))?;
    let kernel_size = o["kernel_size"]
        .as_u64()
        .filter(|x| *x > 0 && x % 2 == 1 && *x <= 255)
        .ok_or_else(|| fail("invalid SSIM kernel size"))? as usize;
    let radius = if gaussian {
        (3.5 * sigma + 0.5).floor() as usize
    } else {
        kernel_size / 2
    };
    if radius > 127 {
        return Err(fail("SSIM kernel exceeds work budget"));
    }
    let length = 2 * radius + 1;
    let mut kernel: Vec<f64> = (0..length)
        .map(|i| {
            if gaussian {
                let x = (i as f64 - radius as f64) / sigma;
                (-x * x / 2.0).exp()
            } else {
                1.0
            }
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for x in &mut kernel {
        *x /= sum;
    }
    let pads: Vec<usize> = (1..shape.len())
        .map(|axis| {
            if shape.len() == 4 && axis == 3 {
                kernel_size / 2
            } else {
                radius
            }
        })
        .collect();
    if pads
        .iter()
        .enumerate()
        .any(|(i, pad)| *pad >= shape[i + 1] || shape[i + 1] + 2 * pad < length)
    {
        return Err(fail(
            "SSIM reflection/kernel incompatible with spatial extent",
        ));
    }
    if p.len()
        .saturating_mul(length)
        .saturating_mul(shape.len() - 1)
        .saturating_mul(5)
        > 200_000_000
    {
        return Err(fail("SSIM convolution work budget exceeded"));
    }
    // Padding may increase the last spatial extent in the qualified 3-D
    // convention. Bound every intermediate allocation, not only input bytes.
    let mut dimensions = shape.to_vec();
    for (i, pad) in pads.iter().enumerate() {
        dimensions[i + 1] = dimensions[i + 1] + 2 * pad + 1 - length;
        let count = dimensions
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| fail("SSIM intermediate size overflow"))?;
        if count > 8_388_608 || count.saturating_mul(length).saturating_mul(5) > 200_000_000 {
            return Err(fail("SSIM intermediate work/memory budget exceeded"));
        }
    }
    let source = [
        p.to_vec(),
        t.to_vec(),
        p.iter().map(|x| x * x).collect(),
        t.iter().map(|x| x * x).collect(),
        p.iter().zip(t).map(|(x, y)| x * y).collect(),
    ];
    let mut filtered = Vec::new();
    for mut values in source {
        let mut dimensions = shape.to_vec();
        for (i, pad) in pads.iter().enumerate() {
            (values, dimensions) = filter_axis(&values, &dimensions, i + 1, &kernel, *pad);
        }
        filtered.push(values);
    }
    let c1 = (k1 * range).powi(2);
    let c2 = (k2 * range).powi(2);
    let mut total = 0.0;
    for i in 0..filtered[0].len() {
        let mp = filtered[0][i];
        let mt = filtered[1][i];
        let vp = (filtered[2][i] - mp * mp).max(0.0);
        let vt = (filtered[3][i] - mt * mt).max(0.0);
        let cov = filtered[4][i] - mp * mt;
        total +=
            ((2.0 * mp * mt + c1) * (2.0 * cov + c2)) / ((mp * mp + mt * mt + c1) * (vp + vt + c2));
    }
    Ok(total / filtered[0].len() as f64)
}

#[wasm_bindgen]
pub fn numerical_metric_methods() -> String {
    json!({"ssim":"separable_reflect_binary64_v1","generalized_dice":"per_sample_weighted_counts_binary64_v1"}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dice_empty_and_identity() {
        let o = json!({"num_classes":2,"include_background":false,"input_format":"index","empty_class_policy":"torchmetrics_1.9.0","weight_type":"square"});
        assert_eq!(dice(&[1.0, 0.0], &[1.0, 0.0], &o).unwrap(), 1.0);
        assert_eq!(dice(&[0.0], &[0.0], &o).unwrap(), 0.0);
    }
    #[test]
    fn ssim_constant_identity() {
        let o = json!({"data_range":1,"sigma":1.5,"kernel_size":3,"k1":0.01,"k2":0.03,"gaussian_kernel":false});
        assert_eq!(ssim(&[1.0; 16], &[1.0; 16], &[1, 4, 4], &o).unwrap(), 1.0);
    }
}
