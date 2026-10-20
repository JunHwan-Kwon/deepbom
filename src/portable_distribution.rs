//! Exact-count distribution projections. Coarse bins provide bounds, not
//! interpolated raw percentiles. The host validates the public Statistics IR.
use crate::portable_numerics::{integer, rounded_ratio};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct Histogram {
    edges: Vec<f64>,
    counts: Vec<String>,
}
#[derive(Deserialize)]
struct Statistics {
    value_count: String,
    finite_count: String,
    zero_count: String,
    unsafe_integer_count: String,
    minimum: Option<f64>,
    maximum: Option<f64>,
    mean: Option<f64>,
    rms: Option<f64>,
    population_stddev: Option<f64>,
    histogram: Option<Histogram>,
}
struct Counts {
    total: BigInt,
    finite: BigInt,
    zero: BigInt,
    unsafe_count: BigInt,
    bins: Vec<BigInt>,
}
fn error(message: &str) -> JsValue {
    JsValue::from_str(message)
}
fn encode<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| error(&e.to_string()))
}
fn nonnegative(value: &str) -> Result<BigInt, JsValue> {
    let n = integer(value)?;
    if n.is_negative() {
        return Err(error("distribution counts must be nonnegative"));
    }
    Ok(n)
}
fn counts(s: &Statistics) -> Result<Counts, JsValue> {
    let total = nonnegative(&s.value_count)?;
    let finite = nonnegative(&s.finite_count)?;
    let zero = nonnegative(&s.zero_count)?;
    let unsafe_count = nonnegative(&s.unsafe_integer_count)?;
    if finite > total || &zero + &unsafe_count > finite {
        return Err(error("distribution counts do not conserve"));
    }
    for value in [s.minimum, s.maximum, s.mean, s.rms, s.population_stddev]
        .into_iter()
        .flatten()
    {
        if !value.is_finite() {
            return Err(error("nonfinite distribution statistic"));
        }
    }
    let bins = if let Some(h) = &s.histogram {
        if !unsafe_count.is_zero()
            || h.edges != crate::numerical_histogram_edges()
            || h.counts.len() + 1 != h.edges.len()
        {
            return Err(error("distribution histogram contract mismatch"));
        }
        let bins = h
            .counts
            .iter()
            .map(|v| nonnegative(v))
            .collect::<Result<Vec<_>, _>>()?;
        if bins.iter().sum::<BigInt>() != finite {
            return Err(error("histogram count conservation failed"));
        }
        if !finite.is_zero() && !matches!((s.minimum, s.maximum), (Some(a),Some(b)) if a<=b) {
            return Err(error("distribution extrema unavailable"));
        }
        bins
    } else {
        vec![]
    };
    Ok(Counts {
        total,
        finite,
        zero,
        unsafe_count,
        bins,
    })
}
fn ratio(a: &BigInt, b: &BigInt) -> Value {
    json!({"numerator":a.to_string(), "denominator":b.to_string(),
        "value":if b.is_zero() { None } else { Some(rounded_ratio(a,b,0)) }})
}
fn describe(s: Statistics) -> Result<Value, JsValue> {
    let c = counts(&s)?;
    let (negative, positive) = if let Some(h) = &s.histogram {
        let negative: BigInt = c
            .bins
            .iter()
            .zip(&h.edges[1..])
            .filter(|(_, e)| **e <= 0.0)
            .map(|(n, _)| n)
            .sum();
        let positive = &c.finite - &negative - &c.zero;
        if positive.is_negative() {
            return Err(error("histogram signs contradict zero count"));
        }
        let zero_bin = h.edges.iter().position(|e| *e == 0.0).unwrap();
        if c.zero > c.bins[zero_bin] {
            return Err(error("zero count exceeds zero histogram bin"));
        }
        (Some(negative), Some(positive))
    } else {
        (None, None)
    };
    let mut intervals = Vec::new();
    if let Some(h) = &s.histogram {
        if !c.finite.is_zero() {
            for percent in [1, 5, 25, 50, 75, 95, 99] {
                let rank: BigInt = (BigInt::from(percent) * &c.finite + 99) / 100;
                let mut before = BigInt::zero();
                let mut index = 0;
                while index < c.bins.len() - 1 && &before + &c.bins[index] < rank {
                    before += &c.bins[index];
                    index += 1;
                }
                let minimum = s.minimum.unwrap();
                let maximum = s.maximum.unwrap();
                let mut lower = minimum.max(h.edges[index]);
                let mut upper = maximum.min(h.edges[index + 1]);
                let mut upper_closed = index == c.bins.len() - 1 || maximum < h.edges[index + 1];
                if rank == BigInt::from(1) {
                    lower = minimum;
                    upper = minimum;
                    upper_closed = true;
                } else if rank == c.finite {
                    lower = maximum;
                    upper = maximum;
                    upper_closed = true;
                } else if rank > *negative.as_ref().unwrap()
                    && rank <= negative.as_ref().unwrap() + &c.zero
                {
                    lower = 0.0;
                    upper = 0.0;
                    upper_closed = true;
                } else if minimum == maximum {
                    lower = minimum;
                    upper = minimum;
                    upper_closed = true;
                }
                if lower > upper {
                    return Err(error("quantile interval contradicts extrema"));
                }
                intervals.push(
                    json!({"percent":percent,"rank":rank.to_string(),"lower":lower,"upper":upper,
                    "upper_closed":upper_closed,"exact":lower==upper&&upper_closed}),
                );
            }
        }
    }
    Ok(json!({
        "finite_fraction":ratio(&c.finite,&c.total),"nonfinite_fraction":ratio(&(&c.total-&c.finite),&c.total),
        "zero_fraction_of_finite":ratio(&c.zero,&c.finite),
        "sign_counts":{"negative":negative.map(|n|n.to_string()),"zero":c.zero.to_string(),"positive":positive.map(|n|n.to_string())},
        "constant_finite_value":if !c.finite.is_zero()&&c.unsafe_count.is_zero()&&s.minimum==s.maximum {s.minimum}else{None},
        "all_values_zero":if c.total.is_zero(){None}else{Some(c.zero==c.total)},
        "all_values_finite":if c.total.is_zero(){None}else{Some(c.total==c.finite)},
        "quantiles":{"method":"finite_nearest_rank_bounds_from_fixed_bins_v1",
            "status":if intervals.is_empty(){"not_assessed"}else{"assessed"},
            "reason":if !intervals.is_empty(){None}else if !c.unsafe_count.is_zero(){Some("unsafe_integer_histogram_unavailable")}else{Some("no_finite_values")},
            "intervals":intervals}
    }))
}
fn compare(a: Statistics, b: Statistics) -> Result<Value, JsValue> {
    let ca = counts(&a)?;
    let cb = counts(&b)?;
    if a.histogram.is_none() || b.histogram.is_none() {
        return Ok(json!({"status":"not_assessed","reason":"histogram_unavailable"}));
    }
    if ca.finite.is_zero() || cb.finite.is_zero() {
        return Ok(json!({"status":"not_assessed","reason":"no_finite_values"}));
    }
    let distance: BigInt = ca
        .bins
        .iter()
        .zip(&cb.bins)
        .map(|(x, y)| (x * &cb.finite - y * &ca.finite).abs())
        .sum();
    let delta =
        |x: Option<f64>, y: Option<f64>| x.zip(y).map(|(x, y)| y - x).filter(|n| n.is_finite());
    Ok(json!({"status":"assessed","reason":null,
        "histogram_total_variation":ratio(&distance,&(2*&ca.finite*&cb.finite)),
        "baseline_finite_count":ca.finite.to_string(),"candidate_finite_count":cb.finite.to_string(),
        "mean_delta":delta(a.mean,b.mean),"rms_delta":delta(a.rms,b.rms),"population_stddev_delta":delta(a.population_stddev,b.population_stddev),
        "boundary":"Finite-value histogram mass only; differences inside the same bin are unresolved. Zero distance does not establish equal values. Metric deltas are candidate minus baseline; null means unavailable or overflow. Not a paired-value error or a causal effect."}))
}
#[wasm_bindgen]
pub fn numerical_distribution(statistics: JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value(statistics).map_err(|e| error(&e.to_string()))?;
    encode(&describe(input)?)
}
#[wasm_bindgen]
pub fn numerical_distribution_comparison(
    baseline: JsValue,
    candidate: JsValue,
) -> Result<JsValue, JsValue> {
    let a = serde_wasm_bindgen::from_value(baseline).map_err(|e| error(&e.to_string()))?;
    let b = serde_wasm_bindgen::from_value(candidate).map_err(|e| error(&e.to_string()))?;
    encode(&compare(a, b)?)
}
