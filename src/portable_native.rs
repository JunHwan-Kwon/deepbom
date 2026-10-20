//! Native tensor identity admission shared by collection and imported IR.
use num_bigint::BigUint;
use serde_json::{json, Value};
use std::collections::BTreeSet;
const SAFE: u64 = 9_007_199_254_740_991;
fn need(ok: bool, msg: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg.into())
    }
}
fn text(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| !s.is_empty() && s.encode_utf16().count() <= 4096)
}
fn sha(v: &Value) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    let row = &v["value"];
    let identity = v["operation"] == "identity";
    need(
        identity || v["operation"] == "descriptor",
        "unknown native tensor contract",
    )?;
    let map = row.as_object().ok_or("native tensor must be an object")?;
    let mut fields = vec![
        "id",
        "name",
        "role",
        "trainable",
        "aliases",
        "dtype",
        "shape",
        "byte_order",
    ];
    if identity {
        fields.extend(["element_count", "byte_length", "payload_sha256"]);
    }
    need(
        map.len() == fields.len() && fields.iter().all(|k| map.contains_key(*k)),
        "native tensor fields do not match contract",
    )?;
    need(text(&row["id"]), "invalid tensor id")?;
    need(text(&row["name"]), "invalid tensor name")?;
    need(
        ["parameter", "buffer", "variable", "activation", "gradient"]
            .contains(&row["role"].as_str().unwrap_or(""))
            && row["trainable"].is_boolean(),
        "invalid native role",
    )?;
    let aliases = row["aliases"].as_array().ok_or("invalid aliases")?;
    let mut seen = BTreeSet::new();
    for a in aliases {
        let a = a.as_str().ok_or("invalid aliases")?;
        need(
            a != row["id"].as_str().unwrap() && seen.insert(a),
            "invalid aliases",
        )?;
    }
    let dtype = row["dtype"].as_str().unwrap_or("");
    need(
        [
            "F16", "BF16", "F32", "F64", "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64",
            "BOOL",
        ]
        .contains(&dtype)
            && row["byte_order"] == "little",
        "invalid tensor layout",
    )?;
    let shape = row["shape"].as_array().ok_or("invalid native shape")?;
    // Validate every dimension before multiplication: zero cannot hide an unknown.
    need(
        shape.iter().all(|x| {
            x.as_f64()
                .is_some_and(|n| n >= 0.0 && n <= SAFE as f64 && n.fract() == 0.0)
        }),
        "invalid native shape",
    )?;
    let dims: Vec<Value> = shape
        .iter()
        .map(|x| json!(x.as_f64().unwrap() as u64))
        .collect();
    let count = if dims.iter().any(|x| x == &json!(0)) {
        BigUint::from(0u32)
    } else {
        crate::portable_contracts::count_product(&dims)?.ok_or("invalid native shape")?
    };
    let bytes = &count
        * BigUint::from(crate::scalar_types_generated::scalar_dtype_bits(dtype).unwrap() / 8);
    if identity {
        need(
            row["element_count"] == count.to_string() && row["byte_length"] == bytes.to_string(),
            "identity counts do not conserve",
        )?;
        need(sha(&row["payload_sha256"]), "invalid native digest")?;
    } else {
        need(
            count <= BigUint::from(100_000_000u64),
            "native tensor exceeds transport budget",
        )?;
    }
    Ok(json!({"element_count":count.to_string(),"byte_length":bytes.to_string()}))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row() -> Value {
        json!({"id":"p","name":"p","role":"parameter","trainable":true,"aliases":[],"dtype":"F32","shape":[2,3],"byte_order":"little"})
    }
    #[test]
    fn counts_and_invalid_zero() {
        let mut x = row();
        assert_eq!(
            run(&json!({"operation":"descriptor","value":x})).unwrap()["byte_length"],
            "24"
        );
        x["shape"] = json!([0, null]);
        assert!(run(&json!({"operation":"descriptor","value":x})).is_err());
        x["shape"] = json!([]);
        assert_eq!(
            run(&json!({"operation":"descriptor","value":x})).unwrap()["element_count"],
            "1"
        );
    }
    #[test]
    fn aliases_and_exact_identity() {
        let mut x = row();
        x["aliases"] = json!(["p"]);
        assert!(run(&json!({"operation":"descriptor","value":x})).is_err());
        x["aliases"] = json!([]);
        x["element_count"] = json!("6");
        x["byte_length"] = json!("24");
        x["payload_sha256"] = json!("a".repeat(64));
        assert!(run(&json!({"operation":"identity","value":x})).is_ok());
        x["byte_length"] = json!(24);
        assert!(run(&json!({"operation":"identity","value":x})).is_err());
    }
}
