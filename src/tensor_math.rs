pub(crate) use crate::scalar_types_generated::scalar_dtype_bits;

pub(crate) const MAX_SAFE_INTEGER: u128 = 9_007_199_254_740_991;

pub(crate) fn safe_count_number(value: u128) -> Option<f64> {
    (value <= MAX_SAFE_INTEGER).then_some(value as f64)
}

pub(crate) fn exact_safe_count(value: f64) -> Option<u128> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= MAX_SAFE_INTEGER as f64)
        .then_some(value as u128)
}

pub(crate) fn scalar_dtype_bytes(dtype: &str) -> Option<usize> {
    let bits = scalar_dtype_bits(dtype)?;
    (bits % 8 == 0).then_some((bits / 8) as usize)
}

pub(crate) fn shape_elements(shape: &[i32]) -> Option<u128> {
    if shape.iter().any(|dimension| *dimension < 0) { return None; }
    element_product(&shape.iter().map(|dimension| *dimension as u128).collect::<Vec<_>>())
}

pub(crate) fn element_product(dimensions: &[u128]) -> Option<u128> {
    if dimensions.contains(&0) { return Some(0); }
    dimensions.iter().try_fold(1u128, |product, dimension| product.checked_mul(*dimension))
}

pub(crate) fn logical_bytes(dtype: &str, elements: u128) -> Option<u128> {
    let bits = scalar_dtype_bits(dtype)? as u128;
    // Avoid overflowing the intermediate bit count when the byte count fits.
    (elements / 8).checked_mul(bits)?.checked_add(((elements % 8) * bits).div_ceil(8))
}

pub(crate) fn shape_payload_bytes(dtype: &str, shape: &[i32]) -> Option<usize> {
    usize::try_from(logical_bytes(dtype, shape_elements(shape)?)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_empty_unknown_and_overflow_are_distinct() {
        assert_eq!(shape_elements(&[]), Some(1));
        assert_eq!(shape_elements(&[i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX, 0]), Some(0));
        assert_eq!(shape_elements(&[0, -1]), None);
        assert_eq!(shape_payload_bytes("INT4", &[3]), Some(2));
        assert_eq!(shape_payload_bytes("INT2", &[3]), Some(1));
        assert_eq!(shape_payload_bytes("BFLOAT16", &[3]), Some(6));
        assert_eq!(shape_payload_bytes("Q4_0", &[32]), None);
        assert_eq!(scalar_dtype_bytes("INT4"), None);
        assert_eq!(scalar_dtype_bytes("COMPLEX128"), Some(16));
        assert_eq!(logical_bytes("BOOL", u128::MAX), Some(u128::MAX));
        assert_eq!(logical_bytes("FLOAT64", u128::MAX), None);
        assert_eq!(safe_count_number(MAX_SAFE_INTEGER + 1), None);
        assert_eq!(exact_safe_count(f64::NAN), None);
        assert_eq!(exact_safe_count(0.5), None);
        assert_eq!(exact_safe_count(MAX_SAFE_INTEGER as f64), Some(MAX_SAFE_INTEGER));
    }
}
