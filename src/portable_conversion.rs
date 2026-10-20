//! Canonical conversion receipts and output binding. Pure data only: no command
//! execution, credential lookup, or claims about conversion correctness.
use crate::portable_validation::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;
const SCHEMA: &str = "deepbom.conversion_receipt.v1";
const BOUND: &str = "deepbom.bound_conversion_receipt.v1";
const BOUNDARY:&str="The active deployment artifact hash and format are observed and exactly bound. Source artifacts, converter identity, invocation, and environment are declarations carried by the receipt and are not re-executed or independently observed by this audit.";
fn allowed(v: &Value, names: &[&str], label: &str) -> R<()> {
    let m = v
        .as_object()
        .ok_or_else(|| format!("{label} must be an object."))?;
    for k in m.keys() {
        need(
            names.contains(&k.as_str()),
            &format!("{label} contains unsupported field {k}."),
        )?;
    }
    Ok(())
}
fn txt(v: &Value, label: &str, max: usize) -> R<String> {
    let s = if let Some(s) = v.as_str() {
        s.to_owned()
    } else if v.is_null() {
        String::new()
    } else {
        v.to_string()
    };
    let s = s
        .trim_matches(crate::portable_evidence::js_space)
        .to_owned();
    need(
        !s.is_empty() && s.encode_utf16().count() <= max,
        &format!("{label} must be a non-empty string no longer than {max} characters."),
    )?;
    Ok(s)
}
fn digest_text(v: &Value, label: &str) -> R<String> {
    let s = string(v).to_ascii_lowercase();
    need(
        sha(&json!(s)),
        &format!("{label} must be a lowercase SHA-256 digest."),
    )?;
    Ok(s)
}
fn optional_digest(v: &Value, label: &str) -> R<Value> {
    if v.is_null() || *v == "" {
        Ok(Value::Null)
    } else {
        Ok(json!(digest_text(v, label)?))
    }
}
fn length(v: &Value, label: &str) -> R<Value> {
    if v.is_null() || *v == "" {
        return Ok(Value::Null);
    }
    let s = v
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| v.to_string());
    decimal(&json!(s))
        .map_err(|_| format!("{label} must be an unsigned decimal integer string."))?;
    Ok(json!(s))
}
fn format(v: &Value, label: &str) -> R<String> {
    let s = string(v)
        .trim_matches(crate::portable_evidence::js_space)
        .to_ascii_lowercase();
    need(
        !s.is_empty()
            && s.len() <= 64
            && s.bytes()
                .next()
                .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
            && s.bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || b"_.+-".contains(&c)),
        &format!("{label} is invalid."),
    )?;
    Ok(match s.as_str() {
        "mlmodel" | "mlpackage" => "coreml".into(),
        "pte" | "ptd" => "executorch".into(),
        _ => s,
    })
}
fn artifact(v: &Value, label: &str) -> R<Value> {
    allowed(
        v,
        &["filename", "format", "sha256", "byte_length_decimal"],
        label,
    )?;
    Ok(
        json!({"filename":txt(&v["filename"],&format!("{label}.filename"),1024)?,"format":format(&v["format"],&format!("{label}.format"))?,"sha256":digest_text(&v["sha256"],&format!("{label}.sha256"))?,"byte_length_decimal":length(&v["byte_length_decimal"],&format!("{label}.byte_length_decimal"))?}),
    )
}
fn secret(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    let b = s.as_bytes();
    for (i, _) in s.char_indices() {
        if i == 0 || matches!(b[i - 1], b'-' | b'_') {
            for key in [
                "token",
                "password",
                "passwd",
                "secret",
                "apikey",
                "api_key",
                "api-key",
                "authorization",
            ] {
                if s[i..].starts_with(key) {
                    let end = i + key.len();
                    if end == b.len() || b[end] == b'=' {
                        return true;
                    }
                }
            }
        }
        if (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'))
            && s[i..].starts_with("bearer")
            && s[i + 6..].chars().next().is_some_and(char::is_whitespace)
        {
            return true;
        }
    }
    false
}
fn converter(v: &Value) -> R<Value> {
    allowed(
        v,
        &[
            "name",
            "version",
            "executable_sha256",
            "invocation",
            "environment",
        ],
        "converter",
    )?;
    allowed(&v["invocation"], &["argv"], "converter.invocation")?;
    let args = array(
        &v["invocation"]["argv"],
        "converter.invocation.argv must contain between 1 and 128 arguments.",
    )?;
    need(
        (1..=128).contains(&args.len()),
        "converter.invocation.argv must contain between 1 and 128 arguments.",
    )?;
    let mut argv = vec![];
    for (i, arg) in args.iter().enumerate() {
        let s = txt(arg, &format!("converter.invocation.argv[{i}]"), 4096)?;
        need(
            !secret(&s),
            &format!(
                "converter.invocation.argv[{i}] appears to contain a credential-bearing argument."
            ),
        )?;
        argv.push(s);
    }
    let e = &v["environment"];
    allowed(
        e,
        &[
            "manifest_sha256",
            "os",
            "architecture",
            "runtime",
            "container_image_digest",
        ],
        "converter.environment",
    )?;
    let image = if e["container_image_digest"].is_null() || e["container_image_digest"] == "" {
        Value::Null
    } else {
        let s = txt(
            &e["container_image_digest"],
            "converter.environment.container_image_digest",
            256,
        )?;
        need(
            s.strip_prefix("sha256:").is_some_and(|x| sha(&json!(x))),
            "converter.environment.container_image_digest must use the sha256:<digest> form.",
        )?;
        json!(s)
    };
    let mut environment = json!({"manifest_sha256":digest_text(&e["manifest_sha256"],"converter.environment.manifest_sha256")?,"container_image_digest":image});
    for (key, max) in [("os", 120), ("architecture", 120), ("runtime", 256)] {
        environment[key] = if e[key].is_null() {
            Value::Null
        } else {
            json!(txt(&e[key], &format!("converter.environment.{key}"), max)?)
        };
    }
    Ok(
        json!({"name":txt(&v["name"],"converter.name",256)?,"version":txt(&v["version"],"converter.version",256)?,"executable_sha256":optional_digest(&v["executable_sha256"],"converter.executable_sha256")?,"invocation":{"argv":argv},"environment":environment}),
    )
}
fn body(v: &Value) -> R<Value> {
    allowed(
        v,
        &[
            "schema",
            "hash_contract",
            "created_at",
            "source_artifacts",
            "converter",
            "supporting_inputs",
            "output_artifact",
            "interpretation_boundary",
            "receipt_sha256",
        ],
        "conversion receipt",
    )?;
    need(
        v["schema"] == SCHEMA,
        "Conversion receipt schema identity is invalid.",
    )?;
    let h = &v["hash_contract"];
    allowed(
        h,
        &[
            "algorithm",
            "canonicalization",
            "source_encoding",
            "excluded_pointers",
        ],
        "hash_contract",
    )?;
    need(
        h["algorithm"] == "SHA-256"
            && h["canonicalization"] == "RFC8785-JCS"
            && h["source_encoding"] == "UTF-8"
            && h["excluded_pointers"] == json!(["/receipt_sha256"]),
        "Conversion receipt hash contract is invalid.",
    )?;
    let created = txt(&v["created_at"], "created_at", 64)?;
    need(
        created.len() == 24
            && created.as_bytes()[19] == b'.'
            && created.ends_with('Z')
            && crate::portable_evidence::timestamp(&json!(created)).is_ok(),
        "created_at must be a canonical ISO-8601 timestamp.",
    )?;
    let inputs = array(
        &v["source_artifacts"],
        "source_artifacts must contain between 1 and 128 hash-identified inputs.",
    )?;
    need(
        (1..=128).contains(&inputs.len()),
        "source_artifacts must contain between 1 and 128 hash-identified inputs.",
    )?;
    let mut sources = vec![];
    let mut supporting = vec![];
    let mut ids = BTreeSet::new();
    for (i, input) in inputs.iter().enumerate() {
        let r = artifact(input, &format!("source_artifacts[{i}]"))?;
        need(
            ids.insert((
                string(&r["sha256"]).to_owned(),
                string(&r["filename"]).to_owned(),
            )),
            "Conversion receipt input identities must be unique.",
        )?;
        sources.push(r);
    }
    if !v["supporting_inputs"].is_null() {
        let inputs = array(
            &v["supporting_inputs"],
            "supporting_inputs must be an array with no more than 256 entries.",
        )?;
        need(
            inputs.len() <= 256,
            "supporting_inputs must be an array with no more than 256 entries.",
        )?;
        for (i, row) in inputs.iter().enumerate() {
            let label = format!("supporting_inputs[{i}]");
            allowed(
                row,
                &["role", "filename", "sha256", "byte_length_decimal"],
                &label,
            )?;
            let r = json!({"role":txt(&row["role"],&format!("{label}.role"),120)?,"filename":txt(&row["filename"],&format!("{label}.filename"),1024)?,"sha256":digest_text(&row["sha256"],&format!("{label}.sha256"))?,"byte_length_decimal":length(&row["byte_length_decimal"],&format!("{label}.byte_length_decimal"))?});
            need(
                ids.insert((
                    string(&r["sha256"]).to_owned(),
                    string(&r["filename"]).to_owned(),
                )),
                "Conversion receipt input identities must be unique.",
            )?;
            supporting.push(r);
        }
    }
    Ok(
        json!({"schema":SCHEMA,"hash_contract":h,"created_at":created,"source_artifacts":sources,"converter":converter(&v["converter"])?,"supporting_inputs":supporting,"output_artifact":artifact(&v["output_artifact"],"output_artifact")?,"interpretation_boundary":txt(&v["interpretation_boundary"],"interpretation_boundary",2400)?}),
    )
}
pub(crate) fn validate(v: &Value) -> R<Value> {
    let mut body = body(v)?;
    let d = digest_text(&v["receipt_sha256"], "receipt_sha256")?;
    need(d == hash(&body)?, "Conversion receipt SHA-256 is invalid.")?;
    body["receipt_sha256"] = json!(d);
    Ok(body)
}
pub(crate) fn bound(v: &Value, active: &Value) -> R<Value> {
    allowed(
        v,
        &[
            "schema",
            "receipt_sha256",
            "receipt_file_sha256",
            "active_artifact",
            "status",
            "evidence_class",
            "output_binding_evidence_class",
            "receipt",
            "interpretation_boundary",
            "binding_sha256",
        ],
        "bound conversion receipt",
    )?;
    need(
        v["schema"] == BOUND
            && v["status"] == "exact_output_hash_and_format_match"
            && v["evidence_class"] == "DECLARED_UNVERIFIED"
            && v["output_binding_evidence_class"] == "OBSERVED",
        "Bound conversion receipt classification is invalid.",
    )?;
    let receipt = validate(&v["receipt"])?;
    need(
        json!(digest_text(&v["receipt_sha256"], "receipt_sha256")?) == receipt["receipt_sha256"],
        "Bound receipt identity is inconsistent.",
    )?;
    let own = artifact(&v["active_artifact"], "active_artifact")?;
    need(
        receipt["output_artifact"]["sha256"] == own["sha256"]
            && receipt["output_artifact"]["format"] == own["format"],
        "Bound conversion receipt output identity is inconsistent.",
    )?;
    if !active.is_null() {
        let expected = artifact(active, "expected_active_artifact")?;
        need(
            own["sha256"] == expected["sha256"] && own["format"] == expected["format"],
            "Bound conversion receipt does not match the expected active artifact.",
        )?;
    }
    optional_digest(&v["receipt_file_sha256"], "receipt_file_sha256")?;
    txt(
        &v["interpretation_boundary"],
        "interpretation_boundary",
        2400,
    )?;
    let mut body = v.clone();
    let d = digest_text(&v["binding_sha256"], "binding_sha256")?;
    body.as_object_mut().unwrap().remove("binding_sha256");
    need(
        d == hash(&body)?,
        "Bound conversion receipt SHA-256 is invalid.",
    )?;
    body["binding_sha256"] = json!(d);
    Ok(body)
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["operation"]) {
        "validate" => validate(&v["value"]),
        "bound" => bound(&v["value"], &v["active"]),
        "build" => {
            let mut input = v["value"].clone();
            need(input.is_object(), "conversion receipt must be an object.")?;
            input["schema"] = json!(SCHEMA);
            input["hash_contract"] = json!({"algorithm":"SHA-256","canonicalization":"RFC8785-JCS","source_encoding":"UTF-8","excluded_pointers":["/receipt_sha256"]});
            let mut body = body(&input)?;
            body["receipt_sha256"] = json!(hash(&body)?);
            Ok(body)
        }
        "bind" => {
            let receipt = validate(&v["value"])?;
            let active = artifact(&v["active"], "active_artifact")?;
            need(
                receipt["output_artifact"]["sha256"] == active["sha256"],
                "Conversion receipt output SHA-256 is not bound to the active artifact.",
            )?;
            need(
                receipt["output_artifact"]["format"] == active["format"],
                "Conversion receipt output format is not bound to the active artifact format.",
            )?;
            let mut body = json!({"schema":BOUND,"receipt_sha256":receipt["receipt_sha256"],"receipt_file_sha256":optional_digest(&v["receipt_file_sha256"],"receipt_file_sha256")?,"active_artifact":active,"status":"exact_output_hash_and_format_match","evidence_class":"DECLARED_UNVERIFIED","output_binding_evidence_class":"OBSERVED","receipt":receipt,"interpretation_boundary":BOUNDARY});
            body["binding_sha256"] = json!(hash(&body)?);
            Ok(body)
        }
        "lineage" => {
            if v["value"].is_null() {
                Ok(
                    json!({"status":"not_provided","evidence_class":"NOT_ASSESSABLE","conversion":null,"interpretation_boundary":"No hash-bound conversion receipt was provided. Converter, source checkpoint, command, and environment lineage are not inferred from the deployment artifact."}),
                )
            } else {
                let conversion = bound(&v["value"], &v["active"])?;
                Ok(
                    json!({"status":"output_bound_source_declared","evidence_class":"DECLARED_UNVERIFIED","interpretation_boundary":conversion["interpretation_boundary"],"conversion":conversion}),
                )
            }
        }
        _ => Err("Unknown conversion receipt operation.".into()),
    }
}
