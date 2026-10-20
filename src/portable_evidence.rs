//! Shared leaf evidence contracts. Document identity is owned by portable_identity.
use serde_json::{json, Value};

pub(crate) fn js_space(c: char) -> bool {
    matches!(c,'\u{0009}'..='\u{000d}'|'\u{0020}'|'\u{00a0}'|'\u{1680}'|'\u{2000}'..='\u{200a}'|'\u{2028}'|'\u{2029}'|'\u{202f}'|'\u{205f}'|'\u{3000}'|'\u{feff}')
}
pub(crate) fn text(v: &Value, max: usize) -> bool {
    v.as_str()
        .is_some_and(|s| s.encode_utf16().count() <= max && !s.trim_matches(js_space).is_empty())
}
pub(crate) fn subject(v: &Value) -> Option<&'static str> {
    let Some(m) = v.as_object() else {
        return Some("subject reference must be an object");
    };
    if m.len() != 5
        || !["kind", "schema", "sha256", "digest_scope", "subject_ref"]
            .iter()
            .all(|k| m.contains_key(*k))
    {
        return Some("subject reference fields do not match its contract");
    }
    let kind = v["kind"].as_str().unwrap_or("");
    if ![
        "artifact_file",
        "artifact_set",
        "native_state",
        "snapshot",
        "evidence_document",
    ]
    .contains(&kind)
    {
        return Some("unknown subject kind");
    }
    if !v["sha256"].as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }) {
        return Some("subject SHA-256 is invalid");
    }
    if v["digest_scope"]
        != if kind == "artifact_file" {
            "file_bytes"
        } else {
            "canonical_document"
        }
    {
        return Some("subject digest scope disagrees with kind");
    }
    if kind == "artifact_file" {
        if !v["schema"].is_null() || !v["subject_ref"].is_null() {
            return Some("file bytes have no document schema or internal IR subject");
        }
    } else if !text(&v["schema"], 4096) {
        return Some("subject schema must be nonempty bounded text");
    }
    for (k, s, error) in [
        (
            "artifact_set",
            "deepbom.artifact_set.v1",
            "artifact set schema mismatch",
        ),
        (
            "native_state",
            "deepbom.model_state_snapshot.v1",
            "native state schema mismatch",
        ),
        (
            "snapshot",
            "deepbom.snapshot_ir.v1",
            "snapshot schema mismatch",
        ),
    ] {
        if kind == k && v["schema"] != s {
            return Some(error);
        }
    }
    if !v["subject_ref"].is_null() && !text(&v["subject_ref"], 4096) {
        return Some("internal subject reference must be nonempty bounded text");
    }
    None
}
fn number(s: &[u8]) -> Option<i64> {
    if !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(s.iter().fold(0, |n, c| n * 10 + (c - b'0') as i64))
}
pub(crate) fn timestamp(v: &Value) -> Result<i64, &'static str> {
    let s = v.as_str().ok_or("timestamp must be text")?.as_bytes();
    let format = "timestamp requires seconds, explicit timezone, and at most millisecond precision";
    if s.len() < 20
        || s[4] != b'-'
        || s[7] != b'-'
        || s[10] != b'T'
        || s[13] != b':'
        || s[16] != b':'
    {
        return Err(format);
    }
    let y = number(&s[0..4]).ok_or(format)?;
    let m = number(&s[5..7]).ok_or(format)?;
    let d = number(&s[8..10]).ok_or(format)?;
    let h = number(&s[11..13]).ok_or(format)?;
    let min = number(&s[14..16]).ok_or(format)?;
    let sec = number(&s[17..19]).ok_or(format)?;
    let mut i = 19;
    let mut ms = 0;
    if s[i] == b'.' {
        i += 1;
        let begin = i;
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
        }
        let n = i - begin;
        if !(1..=3).contains(&n) {
            return Err(format);
        }
        ms = number(&s[begin..i]).ok_or(format)? * 10i64.pow((3 - n) as u32);
    }
    let zone = &s[i..];
    let offset = if zone == b"Z" {
        0
    } else {
        if zone.len() != 6 || !matches!(zone[0], b'+' | b'-') || zone[3] != b':' {
            return Err(format);
        }
        let zh = number(&zone[1..3]).ok_or(format)?;
        let zm = number(&zone[4..6]).ok_or(format)?;
        (zh * 60 + zm) * if zone[0] == b'-' { -1 } else { 1 }
    };
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&m)
        || d < 1
        || d > days[(m - 1) as usize]
        || h >= 24
        || min >= 60
        || sec >= 60
    {
        return Err("invalid calendar timestamp");
    }
    if zone != b"Z" && (number(&zone[1..3]).unwrap() >= 24 || number(&zone[4..6]).unwrap() >= 60) {
        return Err("invalid timezone offset");
    }
    // Proleptic Gregorian civil date to days since 1970-01-01; year zero is
    // supported, matching the existing millisecond-resolution contract.
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yo = y - era * 400;
    let mo = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mo + 2) / 5 + d - 1;
    let doe = yo * 365 + yo / 4 - yo / 100 + doy;
    Ok(
        ((era * 146097 + doe - 719468) * 86400 + h * 3600 + min * 60 + sec - offset * 60) * 1000
            + ms,
    )
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    Ok(match v["operation"].as_str().unwrap_or("") {
        "exact_keys" => {
            let label = v["label"].as_str().unwrap_or("record");
            let keys = v["keys"].as_array().ok_or("invalid field contract")?;
            let allowed: std::collections::BTreeSet<&str> = keys
                .iter()
                .map(|k| k.as_str().ok_or("invalid field name"))
                .collect::<Result<_, _>>()?;
            json!(match v["value"].as_object() {
                None => Some(format!("{label} must be an object")),
                Some(m) if m.keys().any(|k| !allowed.contains(k.as_str())) =>
                    Some(format!("{label} contains an unknown field")),
                Some(m) if allowed.iter().any(|k| !m.contains_key(*k)) =>
                    Some(format!("{label} is missing a field")),
                _ => None,
            })
        }
        "decimal_count" => json!(v["value"].as_str().is_some_and(|s| !s.is_empty()
            && (s == "0" || !s.starts_with('0'))
            && s.bytes().all(|b| b.is_ascii_digit()))),
        "subject" => json!(subject(&v["value"])),
        "timestamp" => match timestamp(&v["value"]) {
            Ok(n) => json!({"value":n,"error":null}),
            Err(e) => json!({"value":null,"error":e}),
        },
        "unique_ids" => {
            let label = v["label"].as_str().unwrap_or("records");
            match v["value"].as_array() {
                Some(a) if a.len() <= 10000 => {
                    let mut ids = std::collections::BTreeSet::new();
                    let mut error = None;
                    // Text errors precede duplicate errors, as in the existing API.
                    for r in a {
                        if !text(&r["id"], 4096) {
                            error = Some(format!("{label} id must be nonempty bounded text"));
                            break;
                        }
                        ids.insert(r["id"].as_str().unwrap());
                    }
                    if error.is_none() && ids.len() != a.len() {
                        error = Some(format!("{label} IDs must be unique"));
                    }
                    json!(error)
                }
                _ => json!(format!("{label} must be a bounded array")),
            }
        }
        _ => return Err("unknown portable evidence contract".into()),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_calendar() {
        assert_eq!(timestamp(&json!("1970-01-01T00:00:00Z")), Ok(0));
        assert_eq!(
            timestamp(&json!("2000-02-29T00:00:00.1+09:00")),
            Ok(951750000100)
        );
        assert!(timestamp(&json!("1900-02-29T00:00:00Z")).is_err());
        assert!(timestamp(&json!("2000-02-29T24:00:00Z")).is_err());
        assert!(timestamp(&json!("2000-02-29T00:00:00.0000Z")).is_err());
        assert!(timestamp(&json!("2000-02-29T00:00:00+24:00")).is_err());
    }
    #[test]
    fn reference_and_utf16_contract() {
        let mut r = json!({"kind":"artifact_file","schema":null,"sha256":"a".repeat(64),"digest_scope":"file_bytes","subject_ref":null});
        assert_eq!(subject(&r), None);
        r["schema"] = json!("x");
        assert!(subject(&r).is_some());
        assert!(!text(&json!("\u{feff}"), 4096));
        assert!(!text(&json!("😀".repeat(2049)), 4096));
    }
}
