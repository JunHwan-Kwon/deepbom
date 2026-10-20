//! Bounded binary64 one-sided Jacobi spectrum; shared calculation/validation owner.
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;
fn missing(reason: &str) -> Value {
    json!({"status":"not_assessed","reason":reason})
}
fn power(e: i32) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else if e >= -1074 {
        f64::from_bits(1u64 << (e + 1074))
    } else {
        0.0
    }
}
fn split(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let e = x.log2().floor().min(1023.0) as i32;
    (x / power(e), e)
}
fn scaled(x: f64, e: i32) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let (m, s) = split(x);
    let e = e + s;
    if e > 1023 {
        return f64::INFINITY;
    }
    if e < -1074 {
        m * power(e + 1074) * f64::from_bits(1)
    } else {
        m * power(e)
    }
}
fn summary(s: &[f64], scale: f64, rows: usize, columns: usize) -> Value {
    let n = s.len();
    let total: f64 = s.iter().map(|x| x * x).sum();
    let cutoff = rows.max(columns) as f64 * f64::EPSILON * s.first().copied().unwrap_or(0.0);
    let rank = s.iter().filter(|&&x| x > cutoff).count();
    let mut cumulative = 0.0;
    let retained: Vec<Option<f64>> = s
        .iter()
        .map(|x| {
            cumulative += x * x;
            if total != 0.0 {
                Some((cumulative / total).min(1.0))
            } else {
                None
            }
        })
        .collect();
    let mut errors = vec![Some(0.0); n + 1];
    let mut tail = 0.0_f64;
    for k in (0..n).rev() {
        tail = tail.hypot(s[k]);
        errors[k] = if total != 0.0 {
            Some(tail / total.sqrt())
        } else {
            None
        };
    }
    if total == 0.0 {
        errors[n] = None;
    }
    let ratio = if n > 0 { s[0] / s[n - 1] } else { f64::NAN };
    json!({"numerical_rank":rank,"rank_tolerance":cutoff*scale,"rank_tolerance_normalized":cutoff,"condition_number":if rank==n&&n>0&&s[n-1]>0.0{Some(ratio)}else{None},"condition_status":if rank<n{"numerically_rank_deficient"}else if ratio.is_finite(){"finite"}else{"overflow"},"cumulative_energy":retained,"relative_frobenius_error_by_rank":errors})
}
fn spectrum(data: &[f64], rows: usize, columns: usize, max_work: u64) -> Value {
    let (n, m) = (rows.min(columns), rows.max(columns));
    if n == 0 || m == 0 {
        return missing("empty_matrix");
    }
    let diagonal = data
        .iter()
        .enumerate()
        .all(|(i, x)| *x == 0.0 || i / columns == i % columns);
    let initial_work = if diagonal {
        data.len() as u64
    } else {
        m as u64 * n as u64 * (n - 1) as u64 / 2
    };
    if initial_work > max_work {
        return missing("full_svd_exceeds_budget");
    }
    let scale = data.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let mut a = vec![0.0; data.len()];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = if scale != 0.0 {
                data[if rows >= columns {
                    i * columns + j
                } else {
                    j * columns + i
                }] / scale
            } else {
                0.0
            };
        }
    }
    if !diagonal && data.iter().any(|x| *x != 0.0 && x / scale == 0.0) {
        return missing("svd_normalization_underflow");
    }
    let initial_energy: f64 = a.iter().map(|x| x * x).sum();
    let smallest = a
        .iter()
        .filter(|x| **x != 0.0)
        .map(|x| split(x.abs()).1)
        .min()
        .unwrap_or(0);
    let deflation = smallest - 104 - ((m as f64).log2() / 2.0).ceil() as i32;
    let mut magnitudes = vec![0.0; n];
    let mut exponents = vec![0i32; n];
    if !diagonal {
        for j in 0..n {
            let maximum = (0..m).fold(0.0_f64, |v, i| v.max(a[i * n + j].abs()));
            (magnitudes[j], exponents[j]) = split(maximum);
            if maximum != 0.0 {
                for i in 0..m {
                    a[i * n + j] /= maximum;
                }
            }
        }
    }
    let tolerance = 32.0 * f64::EPSILON;
    let mut work = if diagonal { data.len() as u64 } else { 0 };
    let mut sweeps = 0;
    let mut converged = diagonal;
    let mut residual = 0.0_f64;
    while !diagonal && sweeps < 64 {
        let mut changed = false;
        residual = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                work += m as u64;
                if work > max_work {
                    return missing("svd_iteration_budget_exceeded");
                }
                let (sa, sb) = (magnitudes[p], magnitudes[q]);
                if sa == 0.0 || sb == 0.0 {
                    continue;
                }
                let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (x, y) = (a[i * n + p], a[i * n + q]);
                    aa += x * x;
                    bb += y * y;
                    ab += x * y;
                }
                let correlation = ab.abs() / aa.sqrt() / bb.sqrt();
                residual = residual.max(correlation);
                if correlation <= tolerance {
                    continue;
                }
                let pair = exponents[p].max(exponents[q]);
                let ra = sa * power(exponents[p] - pair);
                let rb = sb * power(exponents[q] - pair);
                let delta = (bb * rb * rb - aa * ra * ra) / 2.0;
                let product = ab * ra * rb;
                let denominator =
                    delta + if delta < 0.0 { -1.0 } else { 1.0 } * delta.hypot(product);
                if denominator == 0.0 {
                    return missing("svd_rotation_underflow");
                }
                let t = product / denominator;
                let c = 1.0 / 1.0_f64.hypot(t);
                let pm = ab * rb * rb / denominator;
                let qm = ab * ra * ra / denominator;
                let (mut mp, mut mq) = (0.0_f64, 0.0_f64);
                for i in 0..m {
                    let (x, y) = (a[i * n + p], a[i * n + q]);
                    a[i * n + p] = c * (x - pm * y);
                    a[i * n + q] = c * (qm * x + y);
                    mp = mp.max(a[i * n + p].abs());
                    mq = mq.max(a[i * n + q].abs());
                }
                for (j, maximum) in [(p, mp), (q, mq)] {
                    let (mantissa, shift) = split(magnitudes[j] * maximum);
                    magnitudes[j] = mantissa;
                    exponents[j] += shift;
                    if exponents[j] < deflation {
                        magnitudes[j] = 0.0
                    }
                    if maximum != 0.0 {
                        for i in 0..m {
                            a[i * n + j] /= maximum;
                        }
                    }
                }
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
        sweeps += 1;
    }
    if !converged {
        return missing("svd_not_converged");
    }
    let (sigma, singular): (Vec<f64>, Vec<f64>) = if diagonal {
        let mut values: Vec<f64> = (0..n).map(|i| data[i * columns + i].abs()).collect();
        values.sort_by(|a, b| b.total_cmp(a));
        (
            values
                .iter()
                .map(|x| if scale != 0.0 { x / scale } else { 0.0 })
                .collect(),
            values,
        )
    } else {
        let mut factors: Vec<(f64, i32)> = (0..n)
            .map(|j| {
                let mut norm = 0.0_f64;
                for i in 0..m {
                    norm = norm.hypot(a[i * n + j]);
                }
                let (mantissa, shift) = split(norm * magnitudes[j]);
                (mantissa, exponents[j] + shift)
            })
            .collect();
        factors.sort_by(|a, b| {
            (b.0 > 0.0)
                .cmp(&(a.0 > 0.0))
                .then(b.1.cmp(&a.1))
                .then(b.0.total_cmp(&a.0))
        });
        let (sm, se) = split(scale);
        (
            factors.iter().map(|(m, e)| scaled(*m, *e)).collect(),
            factors
                .iter()
                .map(|(m, e)| scaled(m * sm, e + se))
                .collect(),
        )
    };
    let total: f64 = sigma.iter().map(|x| x * x).sum();
    let energy_error = if initial_energy != 0.0 {
        (total - initial_energy).abs() / initial_energy
    } else {
        0.0
    };
    if energy_error > 1e-10 {
        return missing("svd_energy_residual_exceeded");
    }
    let mut result = json!({"status":"assessed","method":if diagonal{"diagonal_closed_form_binary64"}else{"scaled_one_sided_jacobi_binary64"},"rows":rows,"columns":columns,"singular_values":singular,"normalized_singular_values":sigma,"overflowed_singular_value_indices":singular.iter().enumerate().filter_map(|(i,x)|if x.is_finite(){None}else{Some(i)}).collect::<Vec<usize>>(),"scale":scale,"sweeps":sweeps,"orthogonality_residual":residual,"energy_relative_residual":energy_error,"work_count":work,"scope":"declared_axis_unfolding_not_full_convolution_operator"});
    result.as_object_mut().unwrap().extend(
        summary(&sigma, scale, rows, columns)
            .as_object()
            .unwrap()
            .clone(),
    );
    result
}
#[wasm_bindgen]
pub fn numerical_spectrum(
    data: &[f64],
    rows: u32,
    columns: u32,
    max_work: f64,
) -> Result<String, JsValue> {
    if data.len() > 10_000_000
        || u64::from(rows) * u64::from(columns) != data.len() as u64
        || data.iter().any(|x| !x.is_finite())
        || !max_work.is_finite()
        || max_work < 0.0
        || max_work > 1_000_000_000.0
    {
        return Err(JsValue::from_str("Invalid spectrum input or budget"));
    }
    Ok(spectrum(data, rows as usize, columns as usize, max_work as u64).to_string())
}
#[wasm_bindgen]
pub fn numerical_spectrum_summary(
    sigma: &[f64],
    scale: f64,
    rows: u32,
    columns: u32,
) -> Result<String, JsValue> {
    if sigma.len() > 65536
        || sigma.iter().any(|x| !x.is_finite() || *x < 0.0)
        || !scale.is_finite()
        || scale < 0.0
    {
        return Err(JsValue::from_str("Invalid spectrum summary"));
    }
    Ok(summary(sigma, scale, rows as usize, columns as usize).to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_and_rank() {
        let a = spectrum(&[3.0, 0.0, 0.0, 2.0], 2, 2, 10000);
        assert_eq!(a["singular_values"], json!([3.0, 2.0]));
        let b = spectrum(&[1.0, 2.0, 2.0, 4.0], 2, 2, 10000);
        assert_eq!(b["numerical_rank"], 1);
        assert!((b["singular_values"][0].as_f64().unwrap() - 5.0).abs() < 1e-12);
    }
    #[test]
    fn subnormal_triangular() {
        let t = f64::from_bits(1);
        assert_eq!(power(-1074), t);
        let a = spectrum(&[1.0, t, 0.0, t], 2, 2, 10000);
        assert_eq!(a["status"], "assessed");
        assert_eq!(a["singular_values"][1].as_f64(), Some(t));
    }
    #[test]
    fn subnormal_diagonal() {
        let a = spectrum(&[1.0, 0.0, 0.0, f64::from_bits(1)], 2, 2, 1000);
        assert_eq!(a["singular_values"][1].as_f64(), Some(f64::from_bits(1)));
    }
}
