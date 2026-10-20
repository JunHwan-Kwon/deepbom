//! Evidence identity: RFC 8785 UTF-16 ordering, ECMAScript binary64 spelling,
//! UTF-8 JSON strings and SHA-256. Hosts only reject non-JSON host objects.
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) fn canonical(value: &Value) -> Result<String, String> {
    fn write(v: &Value, out: &mut String) -> Result<(), String> {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                let f = n
                    .as_f64()
                    .filter(|f| f.is_finite())
                    .ok_or("non-finite canonical number")?;
                out.push_str(ryu_js::Buffer::new().format_finite(f));
            }
            Value::String(s) => out.push_str(&serde_json::to_string(s).map_err(|e| e.to_string())?),
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    write(v, out)?;
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                let mut keys: Vec<_> = m.keys().collect();
                keys.sort_unstable_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
                for (i, k) in keys.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k).map_err(|e| e.to_string())?);
                    out.push(':');
                    write(&m[*k], out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    let mut out = String::new();
    write(value, &mut out)?;
    Ok(out)
}
pub(crate) fn digest(value: &Value) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical(value)?.as_bytes())
    ))
}
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn evidence_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    Ok(Value::String(match v["operation"].as_str().unwrap_or("") {
        "canonical" => canonical(&v["value"])?,
        "digest" => digest(&v["value"])?,
        _ => return Err("unknown evidence identity operation".into()),
    }))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn ecmascript_and_utf16() {
        assert_eq!(
            canonical(&json!([-0.0, 5e-324, 1e-7, 1e-6, 1e20, 1e21])).unwrap(),
            "[0,5e-324,1e-7,0.000001,100000000000000000000,1e+21]"
        );
        assert_eq!(
            canonical(&json!({"\u{e000}":1,"😀":2,"\n":"\u{0000}\"\\"})).unwrap(),
            "{\"\\n\":\"\\u0000\\\"\\\\\",\"😀\":2,\"\u{e000}\":1}"
        );
        assert_eq!(
            digest(&json!({})).unwrap(),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }
}
