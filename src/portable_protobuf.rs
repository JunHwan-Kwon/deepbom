//! Bounded protobuf wire framing shared by supported format readers. This is framing, not
//! a format-specific semantic schema decoder. Unknown supported-wire fields are retained.
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Debug)]
struct Field {
    no: u32,
    wire: u8,
    offset: usize,
    length: usize,
    value: Option<String>,
}
fn varint(bytes: &[u8], offset: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for i in 0..10 {
        let byte = *bytes.get(*offset).ok_or("Unterminated protobuf varint")?;
        *offset += 1;
        if i == 9 && byte > 1 {
            return Err("Malformed protobuf: varint exceeds uint64".into());
        }
        value |= u64::from(byte & 0x7f) << (i * 7);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("Malformed protobuf: varint exceeds ten bytes".into())
}
fn fields(bytes: &[u8]) -> Result<Vec<Field>, String> {
    let mut offset = 0;
    let mut result = vec![];
    while offset < bytes.len() {
        if result.len() >= 2_000_000 {
            return Err("protobuf field budget exceeded".into());
        }
        let tag = varint(bytes, &mut offset)?;
        let number = tag >> 3;
        let wire = (tag & 7) as u8;
        if number == 0 || number > 0x1fffffff {
            return Err("Malformed protobuf: invalid field number".into());
        }
        let no = number as u32;
        if wire == 0 {
            let start = offset;
            let value = varint(bytes, &mut offset)?;
            result.push(Field {
                no,
                wire,
                offset: start,
                length: offset - start,
                value: Some(value.to_string()),
            });
        } else {
            let length = match wire {
                1 => 8,
                5 => 4,
                2 => usize::try_from(varint(bytes, &mut offset)?)
                    .map_err(|_| "protobuf length exceeds address space")?,
                _ => return Err(format!("Unsupported protobuf wire type {wire}")),
            };
            let end=offset.checked_add(length).filter(|end|*end<=bytes.len()).ok_or_else(||format!("Malformed protobuf: field {no} wire {wire} exceeds message bounds (exceeds the source byte range)"))?;
            result.push(Field {
                no,
                wire,
                offset,
                length,
                value: None,
            });
            offset = end;
        }
    }
    Ok(result)
}
#[wasm_bindgen]
pub fn protobuf_fields(bytes: &[u8]) -> Result<JsValue, JsValue> {
    fields(bytes)
        .map_err(|e| JsValue::from_str(&e))?
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
#[wasm_bindgen]
pub fn protobuf_packed_varints(bytes: &[u8]) -> Result<Vec<u64>, JsValue> {
    let mut offset = 0;
    let mut result = vec![];
    while offset < bytes.len() {
        result.push(varint(bytes, &mut offset).map_err(|e| JsValue::from_str(&e))?);
    }
    Ok(result)
}
#[wasm_bindgen]
pub fn protobuf_varint(bytes: &[u8]) -> Result<JsValue, JsValue> {
    let mut offset = 0;
    let value = varint(bytes, &mut offset).map_err(|e| JsValue::from_str(&e))?;
    serde_json::json!({"value":value.to_string(),"length":offset})
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
#[wasm_bindgen]
pub fn protobuf_packed_floats(bytes: &[u8], width: u32) -> Result<Vec<f64>, JsValue> {
    if ![4, 8].contains(&width) || bytes.len() % width as usize != 0 {
        return Err(JsValue::from_str(
            "Malformed protobuf: packed floating-point payload has an incomplete element",
        ));
    }
    Ok(bytes
        .chunks_exact(width as usize)
        .map(|row| {
            if width == 4 {
                f32::from_le_bytes(row.try_into().unwrap()) as f64
            } else {
                f64::from_le_bytes(row.try_into().unwrap())
            }
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn valid_wire_and_exact_negative_encoding() {
        let rows = fields(&[8, 150, 1, 18, 3, 97, 98, 99]).unwrap();
        assert_eq!(rows[0].value, Some("150".into()));
        assert_eq!(rows[1].offset, 5);
        assert_eq!(rows[1].length, 3);
        let mut offset = 0;
        assert_eq!(
            varint(
                &[255, 255, 255, 255, 255, 255, 255, 255, 255, 1],
                &mut offset
            )
            .unwrap(),
            u64::MAX
        );
    }
    #[test]
    fn malformed_is_bounded() {
        for data in [
            vec![0, 0],
            vec![8, 128],
            vec![18, 5, 1],
            vec![13, 0],
            vec![8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 2],
            vec![8, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 0],
        ] {
            assert!(fields(&data).is_err(), "{data:?}");
        }
    }
}
