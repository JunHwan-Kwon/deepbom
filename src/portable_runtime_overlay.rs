//! Imported runtime reconciliation, with explicit source identity only.
use crate::portable_validation::{need, rows, string, R};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn text(v: &Value) -> Option<String> {
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Null => return None,
        _ => v.to_string(),
    };
    let s = s
        .trim_matches(crate::portable_evidence::js_space)
        .to_owned();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
fn first<'a>(row: &'a Value, keys: &[&str]) -> &'a Value {
    keys.iter()
        .map(|k| &row[*k])
        .find(|v| crate::portable_onnx_semantics::truth(v))
        .unwrap_or(&Value::Null)
}
fn number(v: &Value) -> Option<i64> {
    let n = match v {
        Value::Null => Some(0),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s)
            if s.trim_matches(crate::portable_evidence::js_space)
                .is_empty() =>
        {
            Some(0)
        }
        Value::String(s) => s
            .trim_matches(crate::portable_evidence::js_space)
            .parse::<f64>()
            .ok()
            .and_then(|n| crate::portable_onnx_semantics::safe(&json!(n))),
        _ => crate::portable_onnx_semantics::safe(v),
    };
    n.filter(|n| *n >= 0)
}
fn identity(row: &Value) -> Option<String> {
    text(first(
        row,
        &[
            "runtime_node_ref",
            "node_ref",
            "runtime_node_name",
            "node_name",
            "id",
        ],
    ))
    .or_else(|| {
        row.get("runtime_node_index").and_then(number).map(|i| {
            format!(
                "runtime-node:{}:{i}",
                text(first(row, &["backend", "provider", "delegate"]))
                    .unwrap_or_else(|| "unknown".into())
            )
        })
    })
}
fn strings(v: &Value) -> Vec<String> {
    let values = if v.is_array() {
        rows(v).to_vec()
    } else {
        vec![v.clone()]
    };
    values
        .iter()
        .filter(|v| crate::portable_onnx_semantics::truth(v))
        .filter_map(text)
        .collect()
}
fn binding(row: &Value, ops: &BTreeMap<i64, String>) -> R<(Vec<String>, Option<&'static str>)> {
    let explicit = strings(if !rows(&row["source_subject_refs"]).is_empty() {
        &row["source_subject_refs"]
    } else {
        &row["subject_ref"]
    });
    let raw = if !rows(&row["source_op_indices"]).is_empty() {
        &row["source_op_indices"]
    } else if !rows(&row["op_indices"]).is_empty() {
        &row["op_indices"]
    } else {
        &row["op_index"]
    };
    let values = if raw.is_array() {
        rows(raw).to_vec()
    } else if raw.is_null() {
        vec![]
    } else {
        vec![raw.clone()]
    };
    let indices = values.iter().filter_map(number).collect::<BTreeSet<_>>();
    let mut refs = explicit.clone();
    for i in &indices {
        refs.push(
            ops.get(i)
                .ok_or_else(|| {
                    format!("Runtime overlay references unknown primary-scope operator index {i}.")
                })?
                .clone(),
        );
    }
    let mut seen = BTreeSet::new();
    refs.retain(|s| seen.insert(s.clone()));
    let basis = match (!explicit.is_empty(), !indices.is_empty()) {
        (true, true) => Some("explicit_subject_ref_and_native_index_import"),
        (true, false) => Some("explicit_subject_ref_import"),
        (false, true) => Some("explicit_primary_scope_native_index_import"),
        _ => None,
    };
    Ok((refs, basis))
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let evidence = &v["value"];
    if !evidence.is_object() {
        return Ok(json!([]));
    }
    let graph = &v["graph"];
    let architecture = &v["architecture"];
    let subjects = rows(&graph["operators"])
        .iter()
        .chain(rows(&architecture["nodes"]))
        .map(|r| string(&r["id"]).to_owned())
        .collect::<BTreeSet<_>>();
    let ops = rows(&graph["operators"])
        .iter()
        .filter(|r| r["scope_ref"] == graph["primary_scope_ref"])
        .filter_map(|r| {
            r["native_index"]
                .as_i64()
                .map(|i| (i, string(&r["id"]).into()))
        })
        .collect::<BTreeMap<_, _>>();
    let declared = rows(&evidence["runtime_nodes"]);
    let candidate = if !declared.is_empty() {
        declared
    } else {
        rows(first(evidence, &["rows", "assignments"]))
    };
    let mut inputs = vec![];
    for row in candidate {
        if !declared.is_empty() {
            inputs.push(row);
        } else {
            let has_id = identity(row).is_some();
            let has_binding = !binding(row, &ops)?.0.is_empty();
            if has_id && has_binding {
                inputs.push(row);
            }
        }
    }
    if inputs.is_empty() {
        return Ok(json!([]));
    }
    let sha = text(first(evidence, &["artifact_sha256"]))
        .or_else(|| text(&evidence["artifact"]["sha256"]))
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    need(
        crate::portable_validation::sha(&json!(sha)) && sha == string(&v["artifact_sha256"]),
        "Runtime overlay is not bound to the active artifact SHA-256.",
    )?;
    let mut nodes: BTreeMap<String, (Value, BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
    for (i, row) in inputs.iter().enumerate() {
        let id = identity(row).ok_or_else(|| {
            format!("Runtime overlay row {i} does not declare a runtime node identity.")
        })?;
        let (refs, basis) = binding(row, &ops)?;
        need(
            refs.iter().all(|s| subjects.contains(s)),
            "Runtime overlay references an unknown canonical subject.",
        )?;
        let backend = text(first(row, &["backend", "provider", "delegate"]));
        let class = text(&row["evidence_class"]).unwrap_or_else(|| "MEASURED_IMPORTED".into());
        if let Some((existing, _, _)) = nodes.get(&id) {
            need(
                existing["backend"] == json!(backend),
                &format!("Runtime overlay node {id} declares conflicting backends."),
            )?;
            need(
                existing["evidence_class"] == class,
                &format!("Runtime overlay node {id} declares conflicting evidence classes."),
            )?;
        }
        let entry=nodes.entry(id.clone()).or_insert_with(||(json!({"runtime_node_ref":id,"backend":backend,"runtime_node_kind":text(first(row,&["runtime_node_kind","kind"])),"evidence_class":class}),BTreeSet::new(),BTreeSet::new()));
        entry.1.extend(refs);
        if let Some(b) = basis {
            entry.2.insert(b.into());
        }
    }
    let mut normalized = vec![];
    let mut flat = vec![];
    for (_, (mut row, refs, bases)) in nodes {
        let mut refs = refs.into_iter().collect::<Vec<_>>();
        refs.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        row["mapping_cardinality"] = json!(match refs.len() {
            0 => "unmapped",
            1 => "one_to_one",
            _ => "fused_one_to_many",
        });
        row["mapping_basis"] = json!(if refs.is_empty() {
            "explicit_unmapped_runtime_node"
        } else if bases.len() == 1 {
            bases.iter().next().unwrap()
        } else {
            "explicit_subject_ref_and_native_index_import"
        });
        row["source_subject_refs"] = json!(refs);
        normalized.push(row);
    }
    normalized.sort_by(|a, b| {
        string(&a["runtime_node_ref"])
            .encode_utf16()
            .cmp(string(&b["runtime_node_ref"]).encode_utf16())
    });
    for row in &normalized {
        for subject in rows(&row["source_subject_refs"]) {
            flat.push(json!({"subject_ref":subject,"runtime_node_ref":row["runtime_node_ref"],"backend":row["backend"],"evidence_class":row["evidence_class"],"mapping_cardinality":row["mapping_cardinality"]}));
        }
    }
    let mapped = flat
        .iter()
        .map(|r| string(&r["subject_ref"]))
        .collect::<BTreeSet<_>>()
        .len();
    let summary = json!({"runtime_node_count":normalized.len(),"mapped_runtime_node_count":normalized.iter().filter(|r|!rows(&r["source_subject_refs"]).is_empty()).count(),"unmapped_runtime_node_count":normalized.iter().filter(|r|rows(&r["source_subject_refs"]).is_empty()).count(),"one_to_one_runtime_node_count":normalized.iter().filter(|r|r["mapping_cardinality"]=="one_to_one").count(),"fused_runtime_node_count":normalized.iter().filter(|r|r["mapping_cardinality"]=="fused_one_to_many").count(),"source_subject_reference_count":flat.len(),"distinct_source_subject_count":mapped,"canonical_subject_count":subjects.len(),"canonical_subject_coverage_status":if mapped==subjects.len(){"complete_explicit_mapping"}else if mapped>0{"partial_explicit_mapping"}else{"not_mapped"},"name_similarity_mapping_used":false});
    Ok(
        json!([{"id":"overlay:runtime:0","kind":"runtime_assignment","evidence_class":"IMPORTED_IDENTITY_BOUND_RUNTIME_EVIDENCE","summary":summary,"runtime_nodes":normalized,"rows":flat,"interpretation_boundary":"Imported runtime nodes preserve artifact-bound explicit source_subject_refs or primary-scope native op indices, including one-to-one and fused one-to-many mappings. Generated or unresolved runtime nodes remain explicitly unmapped; DEEPBOM never reconciles them by name similarity."}]),
    )
}
