const Q31_SCALE: f64 = 2_147_483_648.0;

#[derive(Clone, Copy)]
pub(super) struct MultiplierEncoding {
    pub(super) multiplier: i32,
    pub(super) shift: i32,
    pub(super) represented: f64,
}

pub(super) fn quantize_multiplier(
    real_multiplier: f64,
    single_rounding: bool,
) -> MultiplierEncoding {
    if real_multiplier == 0.0 {
        return MultiplierEncoding {
            multiplier: 0,
            shift: 0,
            represented: 0.0,
        };
    }
    let (fraction, mut shift) = frexp_positive(real_multiplier);
    let mut fixed = (fraction * Q31_SCALE).round() as i64;
    if fixed == 1i64 << 31 {
        fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        fixed = 0;
        shift = 0;
    }
    if single_rounding && shift > 30 {
        fixed = (1i64 << 31) - 1;
        shift = 30;
    }
    let multiplier = fixed as i32;
    let represented = multiplier as f64 * 2.0_f64.powi(shift - 31);
    MultiplierEncoding {
        multiplier,
        shift,
        represented,
    }
}

fn frexp_positive(value: f64) -> (f64, i32) {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (fraction, shift) = frexp_positive(value * 2.0_f64.powi(54));
        return (fraction, shift - 54);
    }
    let fraction_bits = (bits & ((1u64 << 52) - 1)) | (1022u64 << 52);
    (f64::from_bits(fraction_bits), exponent - 1022)
}

pub(super) fn round_ties_away_from_zero(value: f64) -> i64 {
    if value >= 0.0 {
        (value + 0.5).floor() as i64
    } else {
        (value - 0.5).ceil() as i64
    }
}

pub(super) fn multiply_by_quantized_multiplier_default(
    value: i32,
    multiplier: i32,
    shift: i32,
) -> Option<i32> {
    if multiplier < 0 || !(-31..=30).contains(&shift) {
        return None;
    }
    let left_shift = shift.max(0) as u32;
    let right_shift = (-shift).max(0) as u32;
    let shifted = (value as i64).checked_mul(1i64.checked_shl(left_shift)?)?;
    let shifted = i32::try_from(shifted).ok()?;
    Some(rounding_divide_by_power_of_two(
        saturating_rounding_doubling_high_mul(shifted, multiplier),
        right_shift,
    ))
}

pub(super) fn multiply_by_quantized_multiplier_single_rounding(
    value: i32,
    multiplier: i32,
    shift: i32,
) -> Option<i32> {
    if multiplier < 0 || !(-31..=30).contains(&shift) {
        return None;
    }
    let total_shift = u32::try_from(31 - shift).ok()?;
    let round = 1i64.checked_shl(total_shift.checked_sub(1)?)?;
    let product = (value as i64).checked_mul(multiplier as i64)?;
    let rounded = product.checked_add(round)?;
    i32::try_from(rounded >> total_shift).ok()
}

fn saturating_rounding_doubling_high_mul(left: i32, right: i32) -> i32 {
    if left == i32::MIN && right == i32::MIN {
        return i32::MAX;
    }
    let product = left as i64 * right as i64;
    let nudge = if product >= 0 {
        1i64 << 30
    } else {
        1 - (1i64 << 30)
    };
    ((product + nudge) / (1i64 << 31)) as i32
}

fn rounding_divide_by_power_of_two(value: i32, exponent: u32) -> i32 {
    if exponent == 0 {
        return value;
    }
    let mask = ((1u64 << exponent) - 1) as i32;
    let remainder = value & mask;
    let threshold = (mask >> 1) + i32::from(value < 0);
    (value >> exponent) + i32::from(remainder > threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_are_rounded_away_from_zero() {
        assert_eq!(round_ties_away_from_zero(1.5), 2);
        assert_eq!(round_ties_away_from_zero(-1.5), -2);
        assert_eq!(round_ties_away_from_zero(1.49), 1);
        assert_eq!(round_ties_away_from_zero(-1.49), -1);
    }

    #[test]
    fn q31_encoding_round_trips_represented_value() {
        let encoding = quantize_multiplier(0.75, false);
        assert_eq!(encoding.multiplier, 1_610_612_736);
        assert_eq!(encoding.shift, 0);
        assert_eq!(encoding.represented, 0.75);
    }

    #[test]
    fn default_multiplier_matches_gemmlowp_rounding_edges() {
        assert_eq!(
            multiply_by_quantized_multiplier_default(3, 1 << 30, 0),
            Some(2)
        );
        assert_eq!(
            multiply_by_quantized_multiplier_default(-3, 1 << 30, 0),
            Some(-1)
        );
        assert_eq!(
            multiply_by_quantized_multiplier_default(7, 1 << 30, -1),
            Some(2)
        );
    }

    #[test]
    fn single_rounding_matches_pinned_common_cc_equation() {
        assert_eq!(
            multiply_by_quantized_multiplier_single_rounding(3, 1 << 30, 0),
            Some(2)
        );
        assert_eq!(
            multiply_by_quantized_multiplier_single_rounding(-3, 1 << 30, 0),
            Some(-1)
        );
        assert_eq!(
            multiply_by_quantized_multiplier_single_rounding(7, 1 << 30, -1),
            Some(2)
        );
    }
}
