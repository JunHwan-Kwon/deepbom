use crate::portable_validation::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub(crate) fn runtime(overlay: &Value, subjects: &BTreeSet<&str>) -> R<()> {
    let mut ids = BTreeSet::new();
    let mut flattened = 0;
    let mut mapped = 0;
    let mut expected = Vec::new();
    let mut singles = 0;
    let mut fused = 0;
    for node in rows(&overlay["runtime_nodes"]) {
        let id = string(&node["runtime_node_ref"]);
        need(
            text(&node["runtime_node_ref"], 1000) && ids.insert(id),
            "Artifact IR runtime node identity is invalid or duplicated.",
        )?;
        let refs: Vec<_> = rows(&node["source_subject_refs"])
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        need(
            refs.iter().all(|r| subjects.contains(r)),
            "Artifact IR runtime node references an unknown canonical subject.",
        )?;
        need(
            node["mapping_cardinality"]
                == match refs.len() {
                    0 => "unmapped",
                    1 => "one_to_one",
                    _ => "fused_one_to_many",
                },
            "Artifact IR runtime mapping cardinality is inconsistent.",
        )?;
        need(
            refs.len() == rows(&node["source_subject_refs"]).len()
                && refs.iter().collect::<BTreeSet<_>>().len() == refs.len(),
            "Artifact IR runtime source references are invalid or duplicated.",
        )?;
        if refs.len() == 1 {
            singles += 1;
        } else if refs.len() > 1 {
            fused += 1;
        }
        for subject in &refs {
            expected.push(json!({"subject_ref":subject,"runtime_node_ref":node["runtime_node_ref"],"backend":node["backend"],"evidence_class":node["evidence_class"],"mapping_cardinality":node["mapping_cardinality"]}));
        }
        flattened += refs.len();
        if !refs.is_empty() {
            mapped += 1;
        }
    }
    let s = &overlay["summary"];
    let assignments = array(
        &overlay["rows"],
        "Artifact IR runtime assignment ledger is invalid.",
    )?;
    need(
        s["runtime_node_count"] == ids.len()
            && s["source_subject_reference_count"] == flattened
            && assignments.len() == flattened,
        "Artifact IR runtime reconciliation count conservation failed.",
    )?;
    let distinct: BTreeSet<_> = assignments
        .iter()
        .map(|r| string(&r["subject_ref"]))
        .collect();
    need(
        s["distinct_source_subject_count"] == distinct.len()
            && s["mapped_runtime_node_count"] == mapped
            && s["unmapped_runtime_node_count"] == rows(&overlay["runtime_nodes"]).len() - mapped
            && s["name_similarity_mapping_used"] == false,
        "Artifact IR runtime reconciliation summary is inconsistent.",
    )?;
    let mut seen = BTreeSet::new();
    for row in assignments {
        need(
            ids.contains(string(&row["runtime_node_ref"])),
            "Artifact IR runtime assignment references an unknown runtime node.",
        )?;
        need(
            seen.insert((
                string(&row["runtime_node_ref"]),
                string(&row["subject_ref"]),
            )) && expected.iter().any(|e| {
                [
                    "subject_ref",
                    "runtime_node_ref",
                    "backend",
                    "evidence_class",
                    "mapping_cardinality",
                ]
                .iter()
                .all(|k| e[k] == row[k])
            }),
            "Artifact IR runtime assignment ledger disagrees with explicit node bindings.",
        )?;
    }
    need(
        s["one_to_one_runtime_node_count"] == singles
            && s["fused_runtime_node_count"] == fused
            && s["canonical_subject_count"] == subjects.len(),
        "Artifact IR runtime binding coverage is inconsistent.",
    )?;
    let coverage = if distinct.len() == subjects.len() {
        "complete_explicit_mapping"
    } else if !distinct.is_empty() {
        "partial_explicit_mapping"
    } else {
        "not_mapped"
    };
    need(
        s["canonical_subject_coverage_status"] == coverage,
        "Artifact IR runtime binding coverage status is inconsistent.",
    )?;
    Ok(())
}
pub(crate) fn validate(v: &Value) -> R<()> {
    need(
        v["schema"] == "deepbom.artifact_ir.v2"
            && [
                "2.2.0", "2.2.1", "2.3.0", "2.3.1", "2.3.2", "2.3.3", "2.3.4",
            ]
            .contains(&string(&v["method_version"])),
        "Artifact IR schema identity is invalid.",
    )?;
    let h = &v["hash_contract"];
    need(
        h["algorithm"] == "SHA-256"
            && h["canonicalization"] == "RFC8785-JCS"
            && h["source_encoding"] == "UTF-8"
            && h["excluded_pointers"] == json!(["/artifact_ir_sha256"]),
        "Artifact IR hash contract is invalid.",
    )?;
    let a = &v["artifact"];
    need(
        sha(&a["sha256"]) && text(&a["filename"], 1000) && text(&a["format"], 40),
        "Artifact IR artifact identity is invalid.",
    )?;
    count(&a["byte_length"])?;
    let g = &v["graph"];
    for key in ["scopes", "scope_relationships", "operators", "values"] {
        array(&g[key], "Artifact IR graph ledger is invalid.")?;
    }
    for op in rows(&g["operators"]) {
        for key in ["macs", "logical_io_bytes"] {
            count(&op["metrics"][key])?;
        }
    }
    for value in rows(&g["values"]) {
        count(&value["logical_byte_length"])?;
    }
    for key in ["macs", "assessed_macs", "serialized_scope_assessed_macs"] {
        count(&g["totals"][key])?;
    }
    let storage = &v["storage_topology"];
    for row in rows(&storage["objects"]) {
        count(&row["serialized_byte_length"])?;
        count(&row["native_source"]["external_data"]["file_byte_length"])?;
    }
    count(&storage["totals"]["serialized_object_bytes_sum"])?;
    let scopes = index(
        &g["scopes"],
        "Artifact IR graph scope identity is invalid or duplicated.",
    )?;
    let ops = index(
        &g["operators"],
        "Artifact IR operator identity is invalid or duplicated.",
    )?;
    let values = index(
        &g["values"],
        "Artifact IR value identity is invalid or duplicated.",
    )?;
    let gt = &g["totals"];
    need(
        g["status"] != "not_serialized"
            || (scopes.is_empty()
                && ops.is_empty()
                && values.is_empty()
                && gt["relationship_count"] == 0),
        "Artifact IR fabricated an executable graph for a graphless format.",
    )?;
    need(
        gt["scope_count"] == scopes.len()
            && gt["materialized_scope_count"]
                == scopes
                    .values()
                    .filter(|s| s["materialization_status"] == "materialized")
                    .count()
            && gt["scope_relationship_count"] == rows(&g["scope_relationships"]).len()
            && gt["operator_count"] == ops.len()
            && gt["value_count"] == values.len(),
        "Artifact IR graph count conservation failed.",
    )?;
    let primary = string(&g["primary_scope_ref"]);
    need(
        g["status"] != "serialized" || scopes.contains_key(primary),
        "Artifact IR primary graph scope reference is invalid.",
    )?;
    for (id, scope) in &scopes {
        let parent = &scope["parent_scope_ref"];
        need(
            parent.is_null() || (scopes.contains_key(string(parent)) && parent != *id),
            "Artifact IR graph scope parent reference is invalid.",
        )?;
        if scope["materialization_status"] == "materialized" {
            for (field, counted) in [
                (
                    "declared_operator_count",
                    ops.values().filter(|r| r["scope_ref"] == *id).count(),
                ),
                (
                    "declared_value_count",
                    values.values().filter(|r| r["scope_ref"] == *id).count(),
                ),
            ] {
                need(
                    scope[field].is_null() || scope[field] == counted,
                    &format!(
                        "Artifact IR materialized scope {} count is inconsistent.",
                        if field == "declared_operator_count" {
                            "operator"
                        } else {
                            "value"
                        }
                    ),
                )?;
            }
        }
    }
    for op in ops.values() {
        need(
            scopes.contains_key(string(&op["scope_ref"])),
            "Artifact IR operator scope reference is invalid.",
        )?;
        for p in rows(&op["inputs"]).iter().chain(rows(&op["outputs"])) {
            let target = values
                .get(string(&p["value_ref"]))
                .ok_or("Artifact IR operator port references an unknown value.")?;
            need(
                target["scope_ref"] == op["scope_ref"],
                "Artifact IR operator port crosses graph scopes.",
            )?;
        }
    }
    for edge in rows(&g["scope_relationships"]) {
        need(
            scopes.contains_key(string(&edge["source_scope_ref"]))
                && scopes.contains_key(string(&edge["target_scope_ref"]))
                && (edge["source_operator_ref"].is_null()
                    || string(&edge["source_operator_ref"]).is_empty()
                    || ops.contains_key(string(&edge["source_operator_ref"]))),
            "Artifact IR scope relationship reference is invalid.",
        )?;
        if let Some(op) = ops.get(string(&edge["source_operator_ref"])) {
            need(
                op["scope_ref"] == edge["source_scope_ref"],
                "Artifact IR scope relationship source operator belongs to a different scope.",
            )?;
        }
    }
    let mut relationships = 0;
    for value in values.values() {
        need(
            scopes.contains_key(string(&value["scope_ref"])),
            "Artifact IR value scope reference is invalid.",
        )?;
        if !value["producer"].is_null() {
            let op = ops
                .get(string(&value["producer"]["operator_ref"]))
                .ok_or("Artifact IR value producer reference is invalid.")?;
            need(
                op["scope_ref"] == value["scope_ref"],
                "Artifact IR value producer crosses graph scopes.",
            )?;
        }
        for c in rows(&value["consumers"]) {
            let op = ops
                .get(string(&c["operator_ref"]))
                .ok_or("Artifact IR value consumer reference is invalid.")?;
            need(
                op["scope_ref"] == value["scope_ref"],
                "Artifact IR value consumer crosses graph scopes.",
            )?;
            relationships += 1;
        }
    }
    need(
        gt["relationship_count"] == relationships,
        "Artifact IR relationship conservation failed.",
    )?;
    for (key, role, error) in [
        (
            "inputs",
            "graph_input",
            "Artifact IR graph input reference is invalid.",
        ),
        (
            "outputs",
            "graph_output",
            "Artifact IR graph output reference is invalid.",
        ),
    ] {
        for r in rows(&g[key]) {
            let value = values.get(string(r)).ok_or(error)?;
            need(
                value["scope_ref"] == primary && member(&value["roles"], role),
                error,
            )?;
        }
    }
    let primary_ops: Vec<_> = rows(&g["operators"])
        .iter()
        .filter(|r| r["scope_ref"] == primary)
        .cloned()
        .collect();
    if g["status"] == "serialized" {
        need(
            string(&gt["assessed_macs"]["decimal"])
                == sum(&primary_ops, |r| &r["metrics"]["macs"])?.to_string()
                && string(&gt["serialized_scope_assessed_macs"]["decimal"])
                    == sum(rows(&g["operators"]), |r| &r["metrics"]["macs"])?.to_string(),
            "Artifact IR MAC conservation failed.",
        )?;
    }
    let objects = index(
        &storage["objects"],
        "Artifact IR storage identity is invalid or duplicated.",
    )?;
    need(
        storage["totals"]["object_count"] == objects.len(),
        "Artifact IR storage count conservation failed.",
    )?;
    need(
        string(&storage["totals"]["serialized_object_bytes_sum"]["decimal"])
            == sum(rows(&storage["objects"]), |r| &r["serialized_byte_length"])?.to_string(),
        "Artifact IR storage byte conservation failed.",
    )?;
    need(
        storage["totals"]["exact_range_count"]
            == objects
                .values()
                .filter(|r| r["byte_range"]["status"] == "exact")
                .count()
            && storage["totals"]["payload_digest_count"]
                == objects
                    .values()
                    .filter(|r| sha(&r["payload_sha256"]))
                    .count(),
        "Artifact IR storage evidence count conservation failed.",
    )?;
    for value in values.values() {
        for r in rows(&value["storage_refs"]) {
            need(
                objects.contains_key(string(r)),
                "Artifact IR value references an unknown storage object.",
            )?;
        }
    }
    if string(&v["method_version"]).starts_with("2.3.") {
        ir(
            "logical_inventory",
            json!({"inventory":v["logical_inventory"],"graph_values":g["values"],"objects":storage["objects"]}),
        )?;
        ir(
            "members",
            json!({"ledger":v["artifact_members"],"identity":a,"storage":storage}),
        )?;
        for value in values.values() {
            ir("value_type", json!({"value":value["type_contract"]}))?;
        }
        for op in ops.values() {
            for (key, unit) in [("macs", "MAC"), ("logical_io_bytes", "byte")] {
                let m = &op["metric_contracts"][key];
                ir("metric", json!({"value":m}))?;
                need(
                    m["unit"] == unit
                        && m["value"] == op["metrics"][key]
                        && m["source_refs"] == json!([op["id"]]),
                    "IR operator metric contradicts native projection.",
                )?;
            }
            for p in rows(&op["inputs"]) {
                need(
                    rows(&values[string(&p["value_ref"])]["consumers"])
                        .iter()
                        .any(|c| c["operator_ref"] == op["id"] && c["port"] == p["port"]),
                    "IR input port lacks a reciprocal consumer.",
                )?;
            }
            for p in rows(&op["outputs"]) {
                let producer = &values[string(&p["value_ref"])]["producer"];
                need(
                    ops.get(string(&producer["operator_ref"])).is_some_and(|o| {
                        rows(&o["outputs"]).iter().any(|r| {
                            r["value_ref"] == p["value_ref"] && r["port"] == producer["port"]
                        })
                    }),
                    "IR output port lacks a valid producer.",
                )?;
            }
        }
        for value in values.values() {
            for c in rows(&value["consumers"]) {
                need(
                    rows(&ops[string(&c["operator_ref"])]["inputs"])
                        .iter()
                        .any(|p| p["value_ref"] == value["id"] && p["port"] == c["port"]),
                    "IR consumer contradicts its input port.",
                )?;
            }
            let p = &value["producer"];
            if !p.is_null() {
                need(
                    rows(&ops[string(&p["operator_ref"])]["outputs"])
                        .iter()
                        .any(|r| r["value_ref"] == value["id"] && r["port"] == p["port"]),
                    "IR producer contradicts its output port.",
                )?;
            }
        }
        need(
            gt["macs"].is_null() || gt["macs"]["decimal"] == gt["assessed_macs"]["decimal"],
            "IR complete MAC total contradicts assessed subtotal.",
        )?;
    }
    let arch = &v["architecture_projection"];
    array(
        &arch["relationships"],
        "Artifact IR architecture projection is invalid.",
    )?;
    let nodes = index(
        &arch["nodes"],
        "Artifact IR architecture node identity is invalid or duplicated.",
    )?;
    need(
        arch["totals"]["node_count"] == nodes.len()
            && arch["totals"]["relationship_count"] == rows(&arch["relationships"]).len(),
        "Artifact IR architecture count conservation failed.",
    )?;
    need(
        g["status"] != "not_serialized" || rows(&arch["relationships"]).is_empty(),
        "Artifact IR graphless architecture projection contains fabricated relationships.",
    )?;
    for node in nodes.values() {
        for r in rows(&node["storage_object_refs"]) {
            need(
                objects.contains_key(string(r)),
                "Artifact IR architecture node references an unknown storage object.",
            )?;
        }
    }
    let quant = &v["quantization_contracts"];
    let records = index(
        &quant["records"],
        "Artifact IR quantization record identity is invalid or duplicated.",
    )?;
    let qt = &quant["totals"];
    need(
        qt["record_count"] == records.len(),
        "Artifact IR quantization count conservation failed.",
    )?;
    need(
        qt["affine_record_count"]
            == records
                .values()
                .filter(|r| r["mapping"]["family"] == "affine")
                .count()
            && qt["block_encoding_record_count"]
                == records
                    .values()
                    .filter(|r| r["mapping"]["family"] == "format_defined_block_encoding")
                    .count()
            && qt["complete_record_count"]
                == records
                    .values()
                    .filter(|r| r["completeness"] == "complete_for_serialized_contract")
                    .count()
            && qt["partial_record_count"]
                == records
                    .values()
                    .filter(|r| r["completeness"] != "complete_for_serialized_contract")
                    .count(),
        "Artifact IR quantization classification count conservation failed.",
    )?;
    for row in records.values() {
        let s = string(&row["subject_ref"]);
        need(
            values.contains_key(s) || objects.contains_key(s),
            "Artifact IR quantization subject reference is invalid.",
        )?;
        for r in rows(&row["related_storage_refs"]) {
            need(
                objects.contains_key(string(r)),
                "Artifact IR quantization record references an unknown related storage object.",
            )?;
        }
    }
    let lineage = &v["lineage_evidence"];
    need(
        ["not_provided", "output_bound_source_declared"].contains(&string(&lineage["status"]))
            && text(&lineage["interpretation_boundary"], 2400),
        "Artifact IR lineage evidence is invalid.",
    )?;
    if lineage["status"] == "not_provided" {
        need(
            lineage["evidence_class"] == "NOT_ASSESSABLE" && lineage["conversion"].is_null(),
            "Artifact IR absent lineage classification is invalid.",
        )?;
    } else {
        need(
            lineage["evidence_class"] == "DECLARED_UNVERIFIED" && lineage["conversion"].is_object(),
            "Artifact IR conversion lineage classification is invalid.",
        )?;
        crate::portable_conversion::bound(
            &lineage["conversion"],
            &json!({"filename":a["filename"],"format":a["format"],"sha256":a["sha256"],"byte_length_decimal":a["byte_length"]["decimal"]}),
        )?;
    }
    let overlays = &v["overlays"];
    let st = array(
        &overlays["static"],
        "Artifact IR overlay ledger is invalid.",
    )?;
    let rt = array(
        &overlays["runtime"],
        "Artifact IR overlay ledger is invalid.",
    )?;
    let subjects: BTreeSet<_> = ops.keys().chain(nodes.keys()).copied().collect();
    let mut ids = BTreeSet::new();
    for overlay in st.iter().chain(rt) {
        need(
            text(&overlay["id"], usize::MAX) && ids.insert(string(&overlay["id"])),
            "Artifact IR overlay identity is invalid or duplicated.",
        )?;
        for row in rows(&overlay["rows"]) {
            need(
                subjects.contains(string(&row["subject_ref"])),
                "Artifact IR overlay subject reference is invalid.",
            )?;
        }
        if overlay["kind"] == "runtime_assignment" {
            runtime(overlay, &subjects)?;
        }
    }
    need(
        v["completeness"]["static_overlay_count"] == st.len()
            && v["completeness"]["runtime_overlay_count"] == rt.len()
            && v["completeness"]["unknown_is_zero"] == false,
        "Artifact IR completeness summary is inconsistent.",
    )?;
    need(
        text(&v["interpretation_boundary"], 2400),
        "Artifact IR interpretation boundary is missing.",
    )
}
