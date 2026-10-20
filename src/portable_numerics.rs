//! Portable exact binary64 reductions. All accumulation is integer arithmetic
//! in units of 2^-1074; rounding occurs only at the public result boundary.
//! This module owns the arithmetic used by every host/channel adapter.
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::Serialize;
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

const MAX_COUNT: u64 = 9_007_199_254_740_991;
const MAX_DECIMAL_DIGITS: usize = 32768;
const MAX_CHUNK: usize = 65536;

fn error(message: &str) -> JsValue {
    JsValue::from_str(message)
}
fn serialize<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| error(&e.to_string()))
}
pub(crate) fn integer(value: &str) -> Result<BigInt, JsValue> {
    if value.len() > MAX_DECIMAL_DIGITS {
        return Err(error("integer exceeds numerical ABI budget"));
    }
    value.parse().map_err(|_| error("invalid exact integer"))
}
fn count(value: f64) -> Result<u64, JsValue> {
    if !value.is_finite() || value < 0.0 || value > MAX_COUNT as f64 || value.fract() != 0.0 {
        return Err(error("invalid exact count"));
    }
    Ok(value as u64)
}
fn components(value: f64) -> (usize, BigInt) {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 2047;
    let mantissa = (bits & ((1u64 << 52) - 1)) + if exponent > 0 { 1u64 << 52 } else { 0 };
    let integer = BigInt::from(mantissa);
    (
        if exponent > 0 {
            exponent as usize - 1
        } else {
            0
        },
        if bits >> 63 == 1 { -integer } else { integer },
    )
}
fn units(value: f64) -> BigInt {
    let (shift, integer) = components(value);
    integer << shift
}
fn ratio_exponent(a: &BigInt, b: &BigInt) -> i32 {
    let mut exponent = a.bits() as i32 - b.bits() as i32;
    if if exponent >= 0 {
        a < &(b << exponent as usize)
    } else {
        (a << -exponent as usize) < *b
    } {
        exponent -= 1;
    }
    exponent
}
fn scaled(a: &BigInt, b: &BigInt, shift: i32) -> (BigInt, BigInt) {
    if shift >= 0 {
        (a << shift as usize, b.clone())
    } else {
        (a.clone(), b << -shift as usize)
    }
}
fn power_two(exponent: i32) -> f64 {
    if exponent > 1023 {
        f64::INFINITY
    } else if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else if exponent >= -1074 {
        f64::from_bits(1u64 << (exponent + 1074))
    } else {
        0.0
    }
}
pub(crate) fn rounded_ratio(numerator: &BigInt, denominator: &BigInt, power: i32) -> f64 {
    if numerator.is_zero() {
        return 0.0;
    }
    let a = numerator.abs();
    let quantum = (-1074).max(ratio_exponent(&a, denominator) + power - 52);
    let (a, b) = scaled(&a, denominator, power - quantum);
    let mut q = &a / &b;
    let remainder = (&a % &b) << 1usize;
    if remainder > b || remainder == b && q.bit(0) {
        q += 1;
    }
    let result = q.to_f64().unwrap_or(f64::INFINITY) * power_two(quantum);
    if numerator.is_negative() {
        -result
    } else {
        result
    }
}
fn rounded_sqrt(numerator: &BigInt, denominator: &BigInt, power: i32) -> f64 {
    if numerator.is_zero() {
        return 0.0;
    }
    let quantum = (-1074).max((ratio_exponent(numerator, denominator) + power).div_euclid(2) - 52);
    let (a, b) = scaled(numerator, denominator, power - 2 * quantum);
    let mut q = (&a / &b).sqrt();
    let midpoint = &b * ((&q << 1usize) + BigInt::one()).pow(2);
    let four = a << 2usize;
    if four > midpoint || four == midpoint && q.bit(0) {
        q += 1;
    }
    q.to_f64().unwrap_or(f64::INFINITY) * power_two(quantum)
}
type Bins = BTreeMap<usize, BigInt>;
fn add_bin(bins: &mut Bins, shift: usize, value: BigInt) {
    if !value.is_zero() {
        *bins.entry(shift).or_default() += value;
    }
}
fn total(bins: &Bins) -> BigInt {
    bins.iter().map(|(shift, value)| value << *shift).sum()
}
fn finite_chunk(values: &[f64]) -> Result<(), JsValue> {
    if values.len() > MAX_CHUNK {
        return Err(error("numerical ABI chunk exceeds 65536 values"));
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(error("Exact moments require finite values."));
    }
    Ok(())
}

#[wasm_bindgen]
pub fn numerical_abi_version() -> String {
    "deepbom.numerical_abi.v1:exact_binary64_v1".into()
}

#[wasm_bindgen]
pub fn numerical_count_ratio(a: &str, b: &str) -> Result<Option<f64>, JsValue> {
    let (a, b) = (integer(a)?, integer(b)?);
    if a.is_negative() || b.is_negative() {
        return Err(error("Count ratios require nonnegative counts."));
    }
    Ok(if b.is_zero() {
        None
    } else {
        Some(rounded_ratio(&a, &b, 0))
    })
}

#[wasm_bindgen]
pub fn numerical_shape_count(shape: &[f64]) -> Result<String, JsValue> {
    if shape.len() > MAX_CHUNK {
        return Err(error("shape rank exceeds numerical ABI budget"));
    }
    let values = shape
        .iter()
        .map(|x| count(*x))
        .collect::<Result<Vec<_>, _>>()?;
    if values.contains(&0) {
        return Ok("0".into());
    }
    let mut result = BigInt::one();
    for value in values {
        result *= value;
        if result.bits() > 100_000 {
            return Err(error("shape product exceeds numerical ABI budget"));
        }
    }
    Ok(result.to_string())
}

#[wasm_bindgen]
pub struct NumericalMoments {
    sums: Bins,
    squares: Bins,
    with_squares: bool,
    added: u64,
}
#[derive(Serialize)]
struct Moments {
    mean: Option<f64>,
    std: Option<f64>,
    rms: Option<f64>,
    l2: Option<f64>,
}
impl NumericalMoments {
    fn push(&mut self, value: f64) {
        let (shift, integer) = components(value);
        if self.with_squares {
            add_bin(&mut self.squares, 2 * shift, &integer * &integer);
        }
        add_bin(&mut self.sums, shift, integer);
        self.added += 1;
    }
    fn values(&self, count: u64) -> Moments {
        if count == 0 {
            return Moments {
                mean: None,
                std: None,
                rms: None,
                l2: None,
            };
        }
        let (sum, squares, n) = (total(&self.sums), total(&self.squares), BigInt::from(count));
        Moments {
            mean: Some(rounded_ratio(&sum, &n, -1074)),
            std: Some(rounded_sqrt(
                &(&squares * &n - &sum * &sum),
                &(&n * &n),
                -2148,
            )),
            rms: Some(rounded_sqrt(&squares, &n, -2148)),
            l2: Some(rounded_sqrt(&squares, &BigInt::one(), -2148)),
        }
    }
}
#[wasm_bindgen]
impl NumericalMoments {
    #[wasm_bindgen(constructor)]
    pub fn new(squares: bool) -> Self {
        Self {
            sums: Bins::new(),
            squares: Bins::new(),
            with_squares: squares,
            added: 0,
        }
    }
    pub fn add_chunk(&mut self, values: &[f64]) -> Result<(), JsValue> {
        finite_chunk(values)?;
        if self.added + values.len() as u64 > MAX_COUNT {
            return Err(error("exact count budget exceeded"));
        }
        for &value in values {
            self.push(value);
        }
        Ok(())
    }
    pub fn totals(&self) -> Vec<String> {
        vec![
            total(&self.sums).to_string(),
            total(&self.squares).to_string(),
        ]
    }
    pub fn mean(&self, size: f64) -> Result<Option<f64>, JsValue> {
        let n = count(size)?;
        Ok(if n == 0 {
            None
        } else {
            Some(rounded_ratio(&total(&self.sums), &BigInt::from(n), -1074))
        })
    }
    pub fn divided_by(&self, divisor: f64, factor: &str) -> Result<f64, JsValue> {
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(error("Exact sum divisor must be positive."));
        }
        let factor = integer(factor)?;
        if factor <= BigInt::zero() {
            return Err(error("Exact sum factor must be positive."));
        }
        let (shift, significand) = components(divisor);
        Ok(rounded_ratio(
            &total(&self.sums),
            &(significand * factor),
            -(shift as i32),
        ))
    }
    pub fn coefficient_of_variation(&self, size: f64) -> Result<Option<f64>, JsValue> {
        let n = count(size)?;
        if !self.with_squares {
            return Err(error("square moments were not collected"));
        }
        let (sum, squares) = (total(&self.sums), total(&self.squares));
        if n == 0 || sum.is_zero() {
            return Ok(None);
        }
        let variance = squares * BigInt::from(n) - &sum * &sum;
        if variance.is_negative() {
            return Err(error("count contradicts observed moments"));
        }
        Ok(Some(rounded_sqrt(&variance, &(&sum * &sum), 0)))
    }
    pub fn finish(&self, size: f64) -> Result<JsValue, JsValue> {
        let n = count(size)?;
        if !self.with_squares {
            return Err(error("square moments were not collected"));
        }
        if n > 0 && total(&self.squares) * BigInt::from(n) < total(&self.sums).pow(2) {
            return Err(error("count contradicts observed moments"));
        }
        serialize(&self.values(n))
    }
}

fn cosine_distance(left: &BigInt, right: &BigInt, dot: &BigInt) -> Option<f64> {
    if left.is_zero() || right.is_zero() {
        return None;
    }
    let product = left * right;
    if dot.is_zero() {
        return Some(1.0);
    }
    if dot * dot == product {
        return Some(if dot.is_negative() { 2.0 } else { 0.0 });
    }
    let scale = BigInt::one() << 128usize;
    let root = (&product << 256usize).sqrt();
    let positive = dot.is_positive();
    let mut value = if positive {
        rounded_ratio(
            &((&product - dot * dot) * &scale),
            &(&product * &scale + dot * &root),
            0,
        )
    } else {
        rounded_ratio(&(&root - dot * &scale), &root, 0)
    };
    let denominator = BigInt::one() << 1075usize;
    let compare = |numerator: BigInt| {
        let complement = if positive {
            &denominator - numerator
        } else {
            numerator - &denominator
        };
        if complement.is_negative() {
            return if positive { -1 } else { 1 };
        }
        let diff = dot * dot * &denominator * &denominator - &product * &complement * &complement;
        let sign = if diff.is_negative() {
            -1
        } else if diff.is_zero() {
            0
        } else {
            1
        };
        if positive {
            -sign
        } else {
            sign
        }
    };
    loop {
        let code = value.to_bits();
        let odd = code & 1 == 1;
        if code > 0 {
            let previous = f64::from_bits(code - 1);
            let cmp = compare(units(previous) + units(value));
            if cmp < 0 || cmp == 0 && odd {
                value = previous;
                continue;
            }
        }
        let next = f64::from_bits(code + 1);
        let cmp = compare(units(value) + units(next));
        if cmp > 0 || cmp == 0 && odd {
            value = next;
            continue;
        }
        return Some(value);
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct NumericalComparison {
    count: u64,
    absolute: Bins,
    difference: Bins,
    left: Bins,
    right: Bins,
    dot: Bins,
    maximum: BigInt,
}
#[wasm_bindgen]
impl NumericalComparison {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            count: 0,
            absolute: Bins::new(),
            difference: Bins::new(),
            left: Bins::new(),
            right: Bins::new(),
            dot: Bins::new(),
            maximum: BigInt::zero(),
        }
    }
    pub fn add_chunk(&mut self, left: &[f64], right: &[f64]) -> Result<(), JsValue> {
        finite_chunk(left)?;
        finite_chunk(right)?;
        if left.len() != right.len() {
            return Err(error("Vector products require equal value counts."));
        }
        if self.count + left.len() as u64 > MAX_COUNT {
            return Err(error("exact count budget exceeded"));
        }
        for (&a, &b) in left.iter().zip(right) {
            let (sx, x) = components(a);
            let (sy, y) = components(b);
            let shift = sx.min(sy);
            let delta = (&y << (sy - shift)) - (&x << (sx - shift));
            let absolute = delta.abs();
            add_bin(&mut self.absolute, shift, absolute.clone());
            add_bin(&mut self.difference, 2 * shift, &delta * &delta);
            add_bin(&mut self.left, 2 * sx, &x * &x);
            add_bin(&mut self.right, 2 * sy, &y * &y);
            add_bin(&mut self.dot, sx + sy, x * y);
            self.maximum = self.maximum.clone().max(absolute << shift);
        }
        self.count += left.len() as u64;
        Ok(())
    }
    pub fn merge(&mut self, other: &NumericalComparison) -> Result<(), JsValue> {
        if self.count + other.count > MAX_COUNT {
            return Err(error("exact count budget exceeded"));
        }
        for (target, source) in [
            (&mut self.absolute, &other.absolute),
            (&mut self.difference, &other.difference),
            (&mut self.left, &other.left),
            (&mut self.right, &other.right),
            (&mut self.dot, &other.dot),
        ] {
            for (shift, value) in source {
                add_bin(target, *shift, value.clone());
            }
        }
        self.count += other.count;
        self.maximum = self.maximum.clone().max(other.maximum.clone());
        Ok(())
    }
    pub fn duplicate(&mut self) -> Result<(), JsValue> {
        let other = self.clone();
        self.merge(&other)
    }
    pub fn regression_gain(&self) -> Option<f64> {
        let left = total(&self.left);
        if left.is_zero() {
            None
        } else {
            let v = rounded_ratio(&total(&self.dot), &left, 0);
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        }
    }
    pub fn cosine(&self) -> Option<f64> {
        let (left, right, dot) = (total(&self.left), total(&self.right), total(&self.dot));
        if left.is_zero() || right.is_zero() {
            None
        } else {
            let v = rounded_sqrt(&(&dot * &dot), &(left * right), 0);
            Some(if dot.is_negative() { -v } else { v })
        }
    }
    pub fn finish(&self) -> Result<JsValue, JsValue> {
        let n = BigInt::from(self.count.max(1));
        let one = BigInt::one();
        let (absolute, diff, left, right, dot) = (
            total(&self.absolute),
            total(&self.difference),
            total(&self.left),
            total(&self.right),
            total(&self.dot),
        );
        let metrics = [
            (
                "total_absolute_difference",
                Some(rounded_ratio(&absolute, &one, -1074)),
            ),
            (
                "mean_absolute_difference",
                Some(rounded_ratio(&absolute, &n, -1074)),
            ),
            (
                "root_mean_square_difference",
                Some(rounded_sqrt(&diff, &n, -2148)),
            ),
            (
                "maximum_absolute_difference",
                Some(rounded_ratio(&self.maximum, &one, -1074)),
            ),
            (
                "relative_l2_difference",
                if left.is_zero() {
                    None
                } else {
                    Some(rounded_sqrt(&diff, &left, 0))
                },
            ),
            ("cosine_distance", cosine_distance(&left, &right, &dot)),
        ];
        let mut result = serde_json::Map::new();
        let mut overflow = Vec::new();
        for (name, value) in metrics {
            if value.is_some_and(|v| !v.is_finite()) {
                overflow.push(name);
            }
            result.insert(
                name.into(),
                serde_json::json!(value.filter(|v| v.is_finite())),
            );
        }
        result.insert(
            "unrepresentable_metrics".into(),
            serde_json::json!(overflow),
        );
        serialize(&result)
    }
}

fn histogram_edges() -> Vec<f64> {
    let exponents = [
        -1074, -32, -24, -16, -12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128,
        256, 512,
    ];
    let mut edges = vec![-f64::MAX];
    edges.extend(exponents.iter().rev().map(|e| -power_two(*e)));
    edges.push(0.0);
    edges.extend(exponents.iter().map(|e| power_two(*e)));
    edges.push(f64::MAX);
    edges
}
#[wasm_bindgen]
pub fn numerical_histogram_edges() -> Vec<f64> {
    histogram_edges()
}

#[wasm_bindgen]
pub struct NumericalStatistics {
    moments: NumericalMoments,
    count: u64,
    finite: u64,
    nan: u64,
    positive_inf: u64,
    negative_inf: u64,
    zero: u64,
    negative_zero: u64,
    unsafe_integer: u64,
    minimum: Option<f64>,
    maximum: Option<f64>,
    integer_minimum: Option<BigInt>,
    integer_maximum: Option<BigInt>,
    bins: Vec<u64>,
    edges: Vec<f64>,
}
impl NumericalStatistics {
    fn push(&mut self, value: f64) {
        self.count += 1;
        if value.is_nan() {
            self.nan += 1;
            return;
        }
        if value == f64::INFINITY {
            self.positive_inf += 1;
            return;
        }
        if value == f64::NEG_INFINITY {
            self.negative_inf += 1;
            return;
        }
        self.finite += 1;
        self.minimum = Some(self.minimum.map_or(value, |v| {
            if v == 0.0 && value == 0.0 {
                if v.is_sign_negative() || value.is_sign_negative() {
                    -0.0
                } else {
                    0.0
                }
            } else {
                v.min(value)
            }
        }));
        self.maximum = Some(self.maximum.map_or(value, |v| {
            if v == 0.0 && value == 0.0 {
                if v.is_sign_positive() || value.is_sign_positive() {
                    0.0
                } else {
                    -0.0
                }
            } else {
                v.max(value)
            }
        }));
        if value == 0.0 {
            self.zero += 1;
            if value.is_sign_negative() {
                self.negative_zero += 1;
            }
        }
        self.moments.push(value);
        let index = self
            .edges
            .partition_point(|edge| *edge <= value)
            .saturating_sub(1)
            .min(self.bins.len() - 1);
        self.bins[index] += 1;
    }
    fn budget(&self, size: usize) -> Result<(), JsValue> {
        if size > MAX_CHUNK || self.count + size as u64 > MAX_COUNT {
            Err(error("statistics input exceeds exact count/chunk budget"))
        } else {
            Ok(())
        }
    }
}
#[wasm_bindgen]
impl NumericalStatistics {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let edges = histogram_edges();
        Self {
            moments: NumericalMoments::new(true),
            count: 0,
            finite: 0,
            nan: 0,
            positive_inf: 0,
            negative_inf: 0,
            zero: 0,
            negative_zero: 0,
            unsafe_integer: 0,
            minimum: None,
            maximum: None,
            integer_minimum: None,
            integer_maximum: None,
            bins: vec![0; edges.len() - 1],
            edges,
        }
    }
    pub fn add_chunk(&mut self, values: &[f64]) -> Result<(), JsValue> {
        self.budget(values.len())?;
        for &v in values {
            self.push(v);
        }
        Ok(())
    }
    pub fn add_integer(&mut self, value: &str) -> Result<(), JsValue> {
        self.budget(1)?;
        let value = integer(value)?;
        self.integer_minimum = Some(
            self.integer_minimum
                .as_ref()
                .map_or(value.clone(), |v| v.min(&value).clone()),
        );
        self.integer_maximum = Some(
            self.integer_maximum
                .as_ref()
                .map_or(value.clone(), |v| v.max(&value).clone()),
        );
        if value.abs() > BigInt::from(MAX_COUNT) {
            self.count += 1;
            self.unsafe_integer += 1;
        } else {
            self.push(value.to_f64().unwrap());
        }
        Ok(())
    }
    pub fn finish(&self) -> Result<JsValue, JsValue> {
        let complete = self.unsafe_integer == 0;
        let m = self.moments.values(self.finite);
        let values = [
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("mean", m.mean),
            ("population_stddev", m.std),
            ("rms", m.rms),
            ("l2_norm", m.l2),
        ];
        // Serialize directly to JS to preserve negative zero in extrema.
        #[derive(Serialize)]
        struct Statistics<'a> {
            value_count: String,
            finite_count: String,
            nonfinite: std::collections::BTreeMap<&'a str, String>,
            zero_count: String,
            negative_zero_count: String,
            precision: &'a str,
            unsafe_integer_count: String,
            integer_minimum: Option<String>,
            integer_maximum: Option<String>,
            minimum: Option<f64>,
            maximum: Option<f64>,
            mean: Option<f64>,
            population_stddev: Option<f64>,
            rms: Option<f64>,
            l2_norm: Option<f64>,
            overflowed_metrics: Vec<&'a str>,
            histogram: Option<Histogram<'a>>,
        }
        #[derive(Serialize)]
        struct Histogram<'a> {
            method: &'a str,
            interval: &'a str,
            edges: &'a [f64],
            counts: Vec<String>,
        }
        let v = |i: usize| {
            if complete {
                values[i].1.filter(|v| v.is_finite())
            } else {
                None
            }
        };
        serialize(&Statistics {
            value_count: self.count.to_string(),
            finite_count: (self.finite + self.unsafe_integer).to_string(),
            nonfinite: [
                ("nan", self.nan.to_string()),
                ("positive_infinity", self.positive_inf.to_string()),
                ("negative_infinity", self.negative_inf.to_string()),
            ]
            .into(),
            zero_count: self.zero.to_string(),
            negative_zero_count: self.negative_zero.to_string(),
            precision: if complete {
                "binary64_rounded_statistics_exact_counts"
            } else {
                "not_assessed_unsafe_integer_arithmetic"
            },
            unsafe_integer_count: self.unsafe_integer.to_string(),
            integer_minimum: self.integer_minimum.as_ref().map(|v| v.to_string()),
            integer_maximum: self.integer_maximum.as_ref().map(|v| v.to_string()),
            minimum: v(0),
            maximum: v(1),
            mean: v(2),
            population_stddev: v(3),
            rms: v(4),
            l2_norm: v(5),
            overflowed_metrics: if complete {
                values[2..]
                    .iter()
                    .filter(|(_, v)| v.is_some_and(|n| !n.is_finite()))
                    .map(|(k, _)| *k)
                    .collect()
            } else {
                vec![]
            },
            histogram: if complete {
                Some(Histogram {
                    method: "fixed_log2_edges_v1",
                    interval: "left_closed_right_open_last_closed",
                    edges: &self.edges,
                    counts: self.bins.iter().map(|x| x.to_string()).collect(),
                })
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_and_extremes() {
        let mut moments = NumericalMoments::new(true);
        for v in [1e308, 1.0, -1e308] {
            moments.push(v);
        }
        assert_eq!(moments.values(3).mean, Some(1.0 / 3.0));
        let mut tiny = NumericalMoments::new(true);
        tiny.push(f64::from_bits(1));
        assert_eq!(tiny.values(1).rms, Some(f64::from_bits(1)));
    }
    #[test]
    fn beyond_u128_shape() {
        assert_eq!(
            numerical_shape_count(&[67108864.0; 6]).unwrap(),
            (BigInt::one() << 156usize).to_string()
        );
    }
    #[test]
    fn rounding_ties() {
        assert_eq!(
            rounded_ratio(&BigInt::one(), &BigInt::from(2), -1074).to_bits(),
            0
        );
    }
}
