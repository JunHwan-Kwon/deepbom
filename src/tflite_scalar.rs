pub(crate) fn tensor_type_name(code: i8) -> String {
    match code {
        0 => "FLOAT32",
        1 => "FLOAT16",
        2 => "INT32",
        3 => "UINT8",
        4 => "INT64",
        5 => "STRING",
        6 => "BOOL",
        7 => "INT16",
        8 => "COMPLEX64",
        9 => "INT8",
        10 => "FLOAT64",
        11 => "COMPLEX128",
        12 => "UINT64",
        13 => "RESOURCE",
        14 => "VARIANT",
        15 => "UINT32",
        16 => "UINT16",
        17 => "INT4",
        18 => "BFLOAT16",
        _ => return format!("TYPE_{}", code),
    }
    .to_string()
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let word = bits as u32;
    let sign = (word & 0x8000) << 16;
    let exponent = (word >> 10) & 0x1f;
    let mantissa = word & 0x03ff;
    f32::from_bits(
        sign | if exponent == 0 {
            if mantissa == 0 {
                0
            } else {
                let mut normalized = mantissa;
                let mut adjusted_exponent = 127u32 - 15 + 1;
                while normalized & 0x0400 == 0 {
                    normalized <<= 1;
                    adjusted_exponent -= 1;
                }
                (adjusted_exponent << 23) | ((normalized & 0x03ff) << 13)
            }
        } else if exponent == 31 {
            (255 << 23) | (mantissa << 13)
        } else {
            ((exponent + 127 - 15) << 23) | (mantissa << 13)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_tensor_types_without_inventing_unknown_names() {
        assert_eq!(tensor_type_name(9), "INT8");
        assert_eq!(tensor_type_name(99), "TYPE_99");
    }

    #[test]
    fn converts_binary16_edge_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
