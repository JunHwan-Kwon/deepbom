//! Bounded scalar FLOAT32 SVM reference; not a model executor or latency estimate.
use crate::portable_onnx_ml::{float, ft, source, Numeric};
use crate::portable_onnx_ml_svm::{Attributes, Input, Runtime};
use crate::portable_onnx_semantics::{input, safe};
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
const LIMIT: i64 = 1_000_000;
fn unresolved(status: &str) -> Value {
    json!({"status":status,"inputValueCount":null,"rawScoreCount":null,"nonFiniteScoreCount":null,"decisionBoundaryCount":null,"rawScorePreview":[],"outputScorePreview":[],"labelPreview":[]})
}
pub(crate) fn logistic(x: f32) -> f32 {
    let p = (1.0 / (1.0 + (-(x.abs() as f64)).exp())) as f32;
    if x < 0. {
        (1.0 - p as f64) as f32
    } else {
        p
    }
}
pub(crate) fn probit(x: f32) -> f32 {
    let x = x as f64;
    let sign = if 2. * x - 1. < 0. { -1. } else { 1. };
    let a = ((1. - (2. * x - 1.)) * (1. + (2. * x - 1.))) as f32;
    let log = (a as f64).ln() as f32;
    let v = (2. / (std::f64::consts::PI * 0.147) + 0.5 * log as f64) as f32;
    let v2 = (log as f64 / 0.147) as f32;
    (std::f64::consts::SQRT_2
        * sign
        * (-(v as f64) + ((v as f64) * (v as f64) - v2 as f64).sqrt()).sqrt()) as f32
}
pub(crate) fn transform(values: &[f32], mode: &str) -> Vec<f32> {
    match mode {
        "LOGISTIC" => values.iter().copied().map(logistic).collect(),
        "PROBIT" => values.iter().copied().map(probit).collect(),
        "SOFTMAX" | "SOFTMAX_ZERO" => {
            let max = if values.iter().any(|n| n.is_nan()) {
                f32::NAN
            } else {
                values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };
            let output = values
                .iter()
                .map(|n| {
                    if mode == "SOFTMAX_ZERO" && (*n as f64).abs() <= 1e-7 {
                        (*n as f64 * (-(max as f64)).exp()) as f32
                    } else {
                        (*n as f64 - max as f64).exp() as f32
                    }
                })
                .collect::<Vec<_>>();
            let sum = output.iter().fold(0f32, |s, n| s + n);
            output.iter().map(|n| n / sum).collect()
        }
        _ => values.to_vec(),
    }
}
fn kernel(input: &[f32], weights: &[f32], kind: &str, params: &[f32], rho: f32) -> f32 {
    let gamma = params.first().copied().unwrap_or(0.);
    let coef = params.get(1).copied().unwrap_or(0.);
    let degree = params.get(2).copied().unwrap_or(0.);
    if kind == "RBF" {
        let mut sum = 0f32;
        for (i, x) in input.iter().enumerate() {
            let diff = *x - weights.get(i).copied().unwrap_or(f32::NAN);
            let product = diff * diff;
            sum += product;
        }
        return ((-gamma * sum) as f64).exp() as f32;
    }
    let mut dot = 0f32;
    for (i, x) in input.iter().enumerate() {
        let product = *x * weights.get(i).copied().unwrap_or(f32::NAN);
        dot += product;
    }
    if kind == "LINEAR" {
        return dot + if rho.is_nan() { 0. } else { rho };
    }
    let product = gamma * dot;
    let affine = product + coef;
    if kind == "POLY" {
        (affine as f64).powf(degree as f64) as f32
    } else {
        (affine as f64).tanh() as f32
    }
}
fn argmax(a: &[f32]) -> usize {
    let mut i = 0;
    for j in 1..a.len() {
        if a[j] > a[i] {
            i = j;
        }
    }
    i
}
fn probabilities(scores: &[f32], a: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let mut matrix = vec![0f32; k * k];
    let mut pair = 0;
    for i in 0..k.saturating_sub(1) {
        for j in i + 1..k {
            let product = scores[pair] * a[pair];
            let sig = 1.0 - logistic(product + b[pair]);
            let value = (sig as f64).clamp(1e-7, 1. - 1e-7);
            matrix[i * k + j] = value as f32;
            matrix[j * k + i] = (1. - value) as f32;
            pair += 1;
        }
    }
    let mut q = vec![0f32; k * k];
    let mut qp = vec![0f32; k];
    let mut p = vec![(1. / k as f64) as f32; k];
    let epsilon = (0.005 / k as f64) as f32;
    for i in 0..k {
        for j in 0..i {
            let square = (matrix[j * k + i] as f64).powi(2) as f32;
            q[i * k + i] += square;
            q[i * k + j] = q[j * k + i];
        }
        for j in i + 1..k {
            let square = (matrix[j * k + i] as f64).powi(2) as f32;
            q[i * k + i] += square;
            q[i * k + j] = -matrix[j * k + i] * matrix[i * k + j];
        }
    }
    for _ in 0..100 {
        let mut pqp = 0f32;
        for i in 0..k {
            qp[i] = 0.;
            for j in 0..k {
                let product = q[i * k + j] * p[j];
                qp[i] += product;
            }
            let product = p[i] * qp[i];
            pqp += product;
        }
        let max_error = qp
            .iter()
            .map(|n| (*n as f64 - pqp as f64).abs())
            .fold(0f64, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        if max_error < epsilon as f64 {
            break;
        }
        for i in 0..k {
            let difference = (-(qp[i] as f64) + pqp as f64) / q[i * k + i] as f64;
            let difference = difference as f32;
            p[i] += difference;
            let d = difference as f64;
            let denom = ((1. + d) * (1. + d)) as f32;
            pqp = ((pqp as f64 + d * (d * q[i * k + i] as f64 + 2. * qp[i] as f64)) / denom as f64)
                as f32;
            for j in 0..k {
                qp[j] = ((qp[j] as f64 + d * q[i * k + j] as f64) / (1. + d)) as f32;
                p[j] = (p[j] as f64 / (1. + d)) as f32;
            }
        }
    }
    p
}
pub(crate) fn run(v: &Value, c: &Input, a: &Attributes, r: &Runtime, gap: bool) -> Value {
    if r.status == "fail" || gap {
        return unresolved(if gap {
            "not_assessed_pinned_ort_cpu_dtype_gap"
        } else {
            "not_assessed_invalid_pinned_ort_contract"
        });
    }
    let k = a.labels.len();
    if a.classifier {
        let work = r.vectors.max(k as i64).checked_add(r.pairs.max(k as i64));
        let cells = if r.probability {
            (k as i64).checked_mul(k as i64)
        } else {
            Some(0)
        };
        if work.is_none_or(|n| n > LIMIT) || cells.is_none_or(|n| n > LIMIT) {
            return unresolved("not_assessed_reference_work_limit");
        }
    }
    let t = input(v, 0);
    let src = if ["FLOAT32", "FLOAT64"].contains(&c.dtype.as_str())
        && t["staticValuesCanonicalTextComplete"] == true
    {
        t["staticValuesCanonicalTexts"]
            .as_array()
            .and_then(|v| v.iter().map(|n| float(n).map(Numeric::Float)).collect())
    } else {
        source(t, &c.dtype)
    };
    let Some(src) = src else {
        return unresolved(if t["role"] == "initializer" {
            "not_assessed_initializer_values"
        } else {
            "not_assessed_runtime_values"
        });
    };
    let b = safe(&c.batch).filter(|n| *n >= 0);
    let f = safe(&c.features).filter(|n| *n >= 0);
    let expected = b.zip(f).and_then(|(b, f)| b.checked_mul(f));
    if expected != Some(src.len() as i64) || expected.is_none_or(|n| n > LIMIT) {
        return unresolved("not_assessed_static_input_cardinality_or_limit");
    }
    let (b, f) = (b.unwrap(), f.unwrap());
    let kernel_work = r
        .vectors
        .max(if a.classifier { k as i64 } else { 1 })
        .checked_mul(f.max(1));
    let pair_work = r.pairs.checked_mul(r.vectors.max(1));
    let probability_work = if r.probability {
        (k as i64)
            .checked_mul(k as i64)
            .and_then(|n| n.checked_mul(100))
    } else {
        Some(0)
    };
    let work = kernel_work
        .zip(pair_work)
        .zip(probability_work)
        .and_then(|((x, y), z)| x.checked_add(y)?.checked_add(z)?.checked_mul(b));
    if work.is_none_or(|n| n > LIMIT) {
        return unresolved("not_assessed_reference_work_limit");
    }
    let (b, f) = (b as usize, f as usize);
    let values = src.iter().map(Numeric::f32).collect::<Vec<_>>();
    let mut raw = vec![];
    let mut out = vec![];
    let mut labels = vec![];
    let mut boundary = 0;
    let features = usize::try_from(r.features.unwrap_or(0)).unwrap_or(0);
    let vc = usize::try_from(r.vectors).unwrap_or(0);
    for batch in 0..b {
        let row = &values[batch * f..(batch + 1) * f];
        if !a.classifier {
            let mut score = if r.mode == "linear" {
                kernel(row, &a.coefficients[..features], "LINEAR", &[], a.rho[0])
            } else {
                let kernels = (0..vc)
                    .map(|i| {
                        kernel(
                            row,
                            &a.support[i * features..(i + 1) * features],
                            &a.kernel,
                            &a.params,
                            0.,
                        )
                    })
                    .collect::<Vec<_>>();
                kernel(&kernels, &a.coefficients[..vc], "LINEAR", &[], a.rho[0])
            };
            if !a.one_class.is_zero() {
                score = if score > 0. { 1. } else { -1. };
            }
            raw.push(score);
            if score == 0. {
                boundary += 1;
            }
            continue;
        }
        if r.mode == "linear" {
            let scores = (0..k)
                .map(|i| {
                    kernel(
                        row,
                        &a.coefficients[i * features..(i + 1) * features],
                        "LINEAR",
                        &[],
                        a.rho[0],
                    )
                })
                .collect::<Vec<_>>();
            raw.extend_from_slice(&scores);
            let max = argmax(&scores);
            let mut selected = max;
            if k == 2 {
                let positive = a.coefficients.iter().all(|n| *n >= 0.);
                if !r.probability
                    && ((positive && scores[max] >= 0.5) || (!positive && scores[max] > 0.))
                {
                    selected = 1;
                }
                if scores[max] == 0. || positive && scores[max] == 0.5 {
                    boundary += 1;
                }
            }
            labels.push(a.labels[selected].clone());
            out.extend(transform(&scores, &a.post));
        } else {
            let kernels = (0..vc)
                .map(|i| {
                    kernel(
                        row,
                        &a.support[i * features..(i + 1) * features],
                        &a.kernel,
                        &a.params,
                        0.,
                    )
                })
                .collect::<Vec<_>>();
            let counts = a
                .vectors
                .iter()
                .map(|n| n.to_usize().unwrap())
                .collect::<Vec<_>>();
            let mut starts = vec![];
            let mut offset = 0;
            for n in &counts {
                starts.push(offset);
                offset += n;
            }
            let mut votes = vec![0f32; k];
            let mut pair_scores = vec![];
            let mut pair = 0;
            for i in 0..k - 1 {
                for j in i + 1..k {
                    let mut sum = 0f64;
                    for m in 0..counts[i] {
                        sum += (a.coefficients[vc * (j - 1) + starts[i] + m]
                            * kernels[starts[i] + m]) as f64;
                    }
                    for m in 0..counts[j] {
                        sum += (a.coefficients[vc * i + starts[j] + m] * kernels[starts[j] + m])
                            as f64;
                    }
                    let score = (sum + a.rho[pair] as f64) as f32;
                    pair_scores.push(score);
                    votes[if score > 0. { i } else { j }] += 1.;
                    if score == 0. {
                        boundary += 1;
                    }
                    pair += 1;
                }
            }
            raw.extend_from_slice(&pair_scores);
            labels.push(a.labels[argmax(&votes)].clone());
            if r.probability {
                out.extend(transform(
                    &probabilities(&pair_scores, &a.prob_a, &a.prob_b, k),
                    &a.post,
                ));
            } else if k == 2 {
                let score = pair_scores[0];
                if a.post == "PROBIT" {
                    out.extend([probit(score), f32::NAN]);
                } else if ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].contains(&a.post.as_str()) {
                    out.extend([(1. - score as f64) as f32, score]);
                } else {
                    out.extend([-score, score]);
                }
            } else {
                out.extend(transform(&pair_scores, &a.post));
            }
        }
    }
    if !a.classifier {
        out = raw.clone();
    }
    let nonfinite = raw.iter().filter(|n| !n.is_finite()).count()
        + if a.classifier {
            out.iter().filter(|n| !n.is_finite()).count()
        } else {
            0
        };
    json!({"status":"assessed_scalar_float32_reference_not_runtime_bit_exact","inputValueCount":src.len(),"rawScoreCount":raw.len(),"nonFiniteScoreCount":nonfinite,"decisionBoundaryCount":boundary,"rawScorePreview":raw.iter().take(12).map(|n|ft(*n as f64)).collect::<Vec<_>>(),"outputScorePreview":out.iter().take(12).map(|n|ft(*n as f64)).collect::<Vec<_>>(),"labelPreview":labels.iter().take(12).collect::<Vec<_>>()})
}
