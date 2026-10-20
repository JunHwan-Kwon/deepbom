use crate::portable_validation::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub(crate) fn native_facts(v: &Value, artifact_hash: &Value) -> R<Value> {
    need(
        v["schema"] == "deepbom.model_ir_native_fact_ledger.v1",
        "Model IR native fact ledger schema is invalid.",
    )?;
    let mut body = v.clone();
    let d = string(&v["ledger_sha256"]).to_ascii_lowercase();
    body.as_object_mut()
        .ok_or("Model IR native fact ledger schema is invalid.")?
        .remove("ledger_sha256");
    need(
        v["artifact_ir_sha256"] == *artifact_hash && d == hash(&body)?,
        "Model IR native fact ledger binding is invalid.",
    )?;
    for row in rows(&v["facts"]["control_dependencies"]) {
        need(
            text(&row["id"], usize::MAX)
                && ["resolved", "unresolved"].contains(&string(&row["status"])),
            "Model IR native control dependency is invalid.",
        )?;
    }
    body["ledger_sha256"] = json!(d);
    Ok(body)
}
pub(crate) fn program_links(p: &Value) -> R<()> {
    let ports = index(
        &p["ports"],
        "Model IR port identity is invalid or duplicated.",
    )?;
    let values = index(
        &p["values"],
        "Model IR value identity is invalid or duplicated.",
    )?;
    let ops = index(
        &p["operations"],
        "Model IR operation identity is invalid or duplicated.",
    )?;
    let regions = index(
        &p["regions"],
        "Model IR region identity is invalid or duplicated.",
    )?;
    let mut used = BTreeSet::new();
    for v in values.values() {
        need(
            regions.contains_key(string(&v["region_ref"])),
            "Model IR value region reference is invalid.",
        )?;
    }
    for op in ops.values() {
        for direction in ["input", "output"] {
            let mut positions = BTreeSet::new();
            for r in array(
                &op[format!("{direction}_port_refs")],
                "Model IR operation port ledger is invalid.",
            )? {
                let id = string(r);
                let port = ports
                    .get(id)
                    .ok_or("Model IR port contradicts its operation or value region.")?;
                let value = values.get(string(&port["value_ref"]));
                need(
                    used.insert(id)
                        && port["operation_ref"] == op["id"]
                        && port["direction"] == direction
                        && port["position"]
                            .as_u64()
                            .is_some_and(|n| n <= 9_007_199_254_740_991 && positions.insert(n))
                        && value.is_some_and(|v| v["region_ref"] == op["region_ref"]),
                    "Model IR port contradicts its operation or value region.",
                )?;
            }
        }
        if !op["metric_contracts"].is_null() {
            for (key, unit) in [("macs", "MAC"), ("logical_io_bytes", "byte")] {
                let m = &op["metric_contracts"][key];
                ir("metric", json!({"value":m}))?;
                need(
                    m["unit"] == unit
                        && m["value"] == op["metrics"][key]
                        && m["source_refs"] == json!([op["id"]]),
                    "Model IR metric contradicts its operation projection.",
                )?;
            }
        }
    }
    need(
        used.len() == ports.len(),
        "Model IR operation omitted a port.",
    )?;
    for block in rows(&p["blocks"]) {
        for r in rows(&block["member_refs"]) {
            need(
                ops.get(string(r))
                    .is_some_and(|op| op["region_ref"] == block["region_ref"]),
                "Model IR block member belongs to a different region.",
            )?;
        }
    }
    for edge in rows(&p["relationships"]) {
        if edge["kind"] == "data_dependency" {
            for (end, direction) in [
                ("from_ref", "output_port_refs"),
                ("to_ref", "input_port_refs"),
            ] {
                need(
                    ops.get(string(&edge[end])).is_some_and(|op| {
                        rows(&op[direction]).iter().any(|r| {
                            ports
                                .get(string(r))
                                .is_some_and(|p| p["value_ref"] == edge["value_ref"])
                        })
                    }),
                    "Model IR data relationship contradicts operation ports.",
                )?;
            }
        } else if edge["kind"] == "storage_binding" {
            need(
                values.get(string(&edge["from_ref"])).is_some_and(|v| {
                    edge["value_ref"] == v["id"]
                        && rows(&v["storage_refs"]).contains(&edge["to_ref"])
                }),
                "Model IR storage relationship contradicts its value.",
            )?;
        }
    }
    Ok(())
}
pub(crate) fn bindings(p: &Value, storage: &Value) -> R<Value> {
    let ids = index(
        &storage["storage_objects"],
        "Model IR storage identity is invalid or duplicated.",
    )?;
    let values = index(
        &p["values"],
        "Model IR value identity is invalid or duplicated.",
    )?;
    let mut out = vec![];
    for port in rows(&p["ports"])
        .iter()
        .filter(|r| r["direction"] == "input")
    {
        if let Some(value) = values.get(string(&port["value_ref"])) {
            for r in rows(&value["storage_refs"]) {
                if !ids.contains_key(string(r)) {
                    continue;
                }
                out.push(json!({"id":format!("binding:{}:{}",string(&port["id"]),string(r)),"operation_ref":port["operation_ref"],"port_ref":port["id"],"value_ref":port["value_ref"],"storage_ref":r,"parameter_role":format!("input_port_{}",port["position"]),"semantic_role":null,"binding_status":"observed_from_graph","binding_basis":"serialized_operation_port_to_value_and_value_to_storage","evidence_class":"OBSERVED_SERIALIZED_ARTIFACT"}));
            }
        }
    }
    Ok(
        json!({"status":if !out.is_empty(){"materialized"}else if p["status"]=="not_serialized"{"not_assessable_graph_not_serialized"}else{"no_bound_serialized_parameters"},"totals":{"binding_count":out.len(),"unresolved_count":0},"bindings":out,"interpretation_boundary":"A weight binding is emitted only when a serialized operation port resolves through a logical value to a serialized storage object. Input-port position is structural; semantic roles are not inferred from names."}),
    )
}
pub(crate) fn validate(v: &Value) -> R<()> {
    need(
        v["schema"] == "deepbom.model_ir.v1"
            && [
                "1.0.0", "1.0.1", "1.1.0", "1.1.1", "1.1.2", "1.1.3", "1.1.4",
            ]
            .contains(&string(&v["method_version"])),
        "Model IR schema identity is invalid.",
    )?;
    let modern = string(&v["method_version"]).starts_with("1.1.");
    let source = &v["source_contract"];
    need(
        source["schema"] == "deepbom.artifact_ir.v2" && sha(&source["sha256"]),
        "Model IR source contract binding is invalid.",
    )?;
    native_facts(&source["native_fact_ledger"], &source["sha256"])?;
    let h = &v["hash_contract"];
    need(
        h["algorithm"] == "SHA-256"
            && h["canonicalization"] == "RFC8785-JCS"
            && h["excluded_pointers"] == json!(["/model_ir_sha256"]),
        "Model IR hash contract is invalid.",
    )?;
    need(
        sha(&v["artifact"]["sha256"]),
        "Model IR artifact identity is invalid.",
    )?;
    count(&v["artifact"]["byte_length"])?;
    let p = &v["program"];
    let storage = &v["tensors_and_storage"];
    let capabilities = json!({"serialized_program_graph":p["status"]=="serialized","tensor_shapes":!rows(&p["values"]).is_empty()||!rows(&storage["logical_values"]).is_empty()||!rows(&storage["storage_objects"]).is_empty(),"serialized_storage":!rows(&storage["storage_objects"]).is_empty(),"weight_bindings":!rows(&v["weight_bindings"]["bindings"]).is_empty(),"quantization_contracts":!rows(&v["quantization"]["records"]).is_empty(),"static_runtime_projection":!rows(&v["static_runtime"]["projections"]).is_empty(),"observed_runtime_overlay":!rows(&v["observed_runtime"]["overlays"]).is_empty()});
    need(
        v["capabilities"] == capabilities,
        "Model IR capability applicability is inconsistent with its ledgers.",
    )?;
    let objects = index(
        &storage["storage_objects"],
        "Model IR storage identity is invalid or duplicated.",
    )?;
    for row in objects.values() {
        for key in [
            "element_count",
            "serialized_byte_length",
            "logical_byte_length",
        ] {
            count(&row[key])?;
        }
        count(&row["external_storage"]["source"]["file_byte_length"])?;
        if modern {
            let ranked = row["shape_rank_status"] == "ranked"
                || (row["shape_rank_status"].is_null() && !rows(&row["shape"]).is_empty());
            let expected = if ranked {
                if rows(&row["shape"]).iter().any(|v| {
                    crate::portable_contracts::decimal(v)
                        .is_some_and(|n| n == num_bigint::BigUint::from(0u32))
                }) {
                    Some(num_bigint::BigUint::from(0u32))
                } else {
                    crate::portable_contracts::count_product(rows(&row["shape"]))?
                }
                .map(|n| crate::portable_contracts::exact(&n))
                .unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            need(
                row["element_count"] == expected,
                "Model IR storage element count contradicts its declared shape.",
            )?;
        }
    }
    for op in rows(&p["operations"]) {
        for k in ["macs", "logical_io_bytes"] {
            count(&op["metrics"][k])?;
        }
    }
    let profiles = index(
        &v["profiles"],
        "Model IR mandatory profile contract is invalid.",
    )?;
    need(
        profiles.get("identity").is_some_and(|p| {
            p["requirement"] == "core_mandatory" && p["applicability"] == "applicable"
        }),
        "Model IR mandatory profile contract is invalid.",
    )?;
    for (id, cap) in [
        ("program_graph", "serialized_program_graph"),
        ("tensor_storage", "serialized_storage"),
        ("quantization", "quantization_contracts"),
        ("static_runtime_projection", "static_runtime_projection"),
        ("observed_runtime", "observed_runtime_overlay"),
    ] {
        need(
            profiles
                .get(id)
                .is_some_and(|p| json!(p["applicability"] == "applicable") == capabilities[cap]),
            &format!("Model IR {id} profile applicability is inconsistent."),
        )?;
    }
    let mut ids = BTreeSet::new();
    for (key, label) in [
        ("programs", "program"),
        ("functions", "function"),
        ("regions", "region"),
        ("blocks", "block"),
        ("operations", "operation"),
        ("values", "value"),
        ("ports", "port"),
    ] {
        for row in array(
            &p[key],
            &format!("Model IR program {key} ledger is invalid."),
        )? {
            need(
                text(&row["id"], usize::MAX) && ids.insert(string(&row["id"])),
                &format!("Model IR {label} identity is invalid or duplicated."),
            )?;
        }
    }
    array(
        &p["relationships"],
        "Model IR program relationships ledger is invalid.",
    )?;
    let totals = &storage["totals"];
    count(&totals["serialized_object_bytes_sum"])?;
    need(
        totals["storage_object_count"] == objects.len()
            && totals["logical_value_count"] == rows(&storage["logical_values"]).len()
            && string(&totals["serialized_object_bytes_sum"]["decimal"])
                == sum(rows(&storage["storage_objects"]), |r| {
                    &r["serialized_byte_length"]
                })?
                .to_string()
            && totals["exact_range_count"]
                == objects
                    .values()
                    .filter(|r| r["byte_range"]["status"] == "exact")
                    .count()
            && totals["payload_digest_count"]
                == objects
                    .values()
                    .filter(|r| sha(&r["payload_sha256"]))
                    .count(),
        "Model IR storage count or byte conservation failed.",
    )?;
    for id in objects.keys() {
        need(
            ids.insert(*id),
            "Model IR storage identity collides with a program subject.",
        )?;
    }
    if modern && !v["logical_inventory"].is_null() {
        ir(
            "logical_inventory",
            json!({"inventory":v["logical_inventory"],"graph_values":p["values"],"objects":storage["storage_objects"]}),
        )?;
        let inventory = index(
            &v["logical_inventory"]["values"],
            "Model IR logical inventory count is inconsistent.",
        )?;
        need(
            inventory.len() == rows(&storage["logical_values"]).len(),
            "Model IR logical inventory count is inconsistent.",
        )?;
        let mut seen = BTreeSet::new();
        for row in rows(&storage["logical_values"]) {
            let id = string(&row["id"]);
            let original = inventory
                .get(id)
                .ok_or("Model IR logical value contradicts inventory.")?;
            need(
                seen.insert(id)
                    && ["dtype", "shape", "type_contract", "storage_refs"]
                        .iter()
                        .all(|k| row[*k] == original[*k]),
                "Model IR logical value contradicts inventory.",
            )?;
        }
        ir(
            "members",
            json!({"ledger":v["artifact_members"],"identity":v["artifact"],"storage":{"objects":storage["storage_objects"]}}),
        )?;
        need(
            v["artifact_set"]["member_refs"]
                == json!(rows(&v["artifact_members"]["members"])
                    .iter()
                    .map(|r| r["id"].clone())
                    .collect::<Vec<_>>()),
            "Model IR artifact member count is inconsistent.",
        )?;
    }
    let mut relations = BTreeSet::new();
    for row in rows(&p["relationships"]) {
        need(
            text(&row["id"], usize::MAX)
                && relations.insert(string(&row["id"]))
                && [
                    "data_dependency",
                    "control_dependency",
                    "call",
                    "region_ownership",
                    "state_dependency",
                    "alias",
                    "storage_binding",
                    "parameter_binding",
                ]
                .contains(&string(&row["kind"])),
            "Model IR relationship identity or kind is invalid.",
        )?;
        need(
            ids.contains(string(&row["from_ref"]))
                && ids.contains(string(&row["to_ref"]))
                && (row["value_ref"].is_null()
                    || string(&row["value_ref"]).is_empty()
                    || ids.contains(string(&row["value_ref"]))),
            "Model IR relationship reference is invalid.",
        )?;
    }
    let regions = index(
        &p["regions"],
        "Model IR region identity is invalid or duplicated.",
    )?;
    let ops = index(
        &p["operations"],
        "Model IR operation identity is invalid or duplicated.",
    )?;
    let values = index(
        &p["values"],
        "Model IR value identity is invalid or duplicated.",
    )?;
    for op in ops.values() {
        need(
            regions.contains_key(string(&op["region_ref"])),
            "Model IR operation region reference is invalid.",
        )?;
    }
    let mut members = BTreeSet::new();
    let mut n = 0;
    for block in rows(&p["blocks"]) {
        for r in rows(&block["member_refs"]) {
            n += 1;
            need(
                members.insert(string(r)) && ops.contains_key(string(r)),
                "Model IR structural block operation conservation is invalid.",
            )?;
        }
        need(
            regions.contains_key(string(&block["region_ref"]))
                && !rows(&block["member_refs"]).is_empty(),
            "Model IR structural block reference is invalid.",
        )?;
    }
    need(
        n == ops.len(),
        "Model IR structural block operation conservation is invalid.",
    )?;
    for port in rows(&p["ports"]) {
        need(
            ops.contains_key(string(&port["operation_ref"]))
                && values.contains_key(string(&port["value_ref"])),
            "Model IR port reference is invalid.",
        )?;
    }
    need(
        p["status"] != "not_serialized"
            || (ops.is_empty() && relations.is_empty() && regions.is_empty()),
        "Model IR fabricated a program for a graphless artifact.",
    )?;
    for b in rows(&v["weight_bindings"]["bindings"]) {
        need(
            ops.contains_key(string(&b["operation_ref"]))
                && values.contains_key(string(&b["value_ref"]))
                && objects.contains_key(string(&b["storage_ref"])),
            "Model IR weight binding reference is invalid.",
        )?;
    }
    if modern {
        program_links(p)?;
        need(
            v["weight_bindings"] == bindings(p, storage)?,
            "Model IR weight binding contradicts operation ports and storage.",
        )?;
    }
    for row in rows(&v["quantization"]["records"]) {
        need(
            values.contains_key(string(&row["subject_ref"]))
                || objects.contains_key(string(&row["subject_ref"])),
            "Model IR quantization subject reference is invalid.",
        )?;
    }
    let model_profiles = index(
        &v["architecture"]["model_profiles"],
        "Model IR model-profile identity is invalid or duplicated.",
    )?;
    for profile in model_profiles.values() {
        for a in rows(&profile["assignments"]) {
            need(
                ids.contains(string(&a["subject_ref"])),
                "Model IR model-profile assignment reference is invalid.",
            )?;
        }
        for g in rows(&profile["groups"]) {
            need(
                rows(&g["member_refs"])
                    .iter()
                    .all(|r| ids.contains(string(r))),
                "Model IR model-profile group member reference is invalid.",
            )?;
        }
    }
    let hierarchy: BTreeSet<_> = rows(&v["architecture"]["hierarchy"])
        .iter()
        .map(|r| string(&r["id"]))
        .collect();
    for projection in rows(&v["static_runtime"]["projections"]) {
        for segment in rows(&projection["segments"]) {
            need(
                rows(&segment["source_subject_refs"])
                    .iter()
                    .all(|r| ops.contains_key(string(r)) || hierarchy.contains(string(r))),
                "Model IR static runtime subject reference is invalid.",
            )?;
            need(
                segment["actual_runtime_assignment_claim"] == false,
                "Model IR static projection was promoted to an actual runtime assignment.",
            )?;
        }
    }
    let generic = &v["generic_analysis"];
    need(
        generic["pass_contract"]["format_specific_branch_count"] == 0
            && generic["pass_contract"]["native_ledger_access"] == false,
        "Model IR generic analysis crossed the format-neutral import boundary.",
    )?;
    need(
        generic["pass_count"] == rows(&generic["passes"]).len(),
        "Model IR generic analysis pass conservation is invalid.",
    )?;
    for pass in rows(&generic["passes"]) {
        need(
            pass["subject_refs"].is_array()
                && text(&pass["status"], usize::MAX)
                && text(&pass["completeness"], usize::MAX),
            "Model IR generic analysis pass contract is invalid.",
        )?;
    }
    need(
        v["completeness"]["unknown_is_zero"] == false
            && v["loss_ledger"]["entries"].is_array()
            && v["loss_ledger"]["count"] == rows(&v["loss_ledger"]["entries"]).len(),
        "Model IR completeness or loss ledger is inconsistent.",
    )?;
    need(
        text(&v["interpretation_boundary"], usize::MAX),
        "Model IR interpretation boundary is missing.",
    )
}

// Source/display order is derived once, independently of a host's Array.sort.
pub(crate) fn orders(v: &Value) -> R<Value> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut ops = rows(&v["operations"]).to_vec();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, row) in ops.iter().enumerate() {
        groups
            .entry(string(&row["region_ref"]).into())
            .or_default()
            .push(i);
    }
    for indices in groups.values() {
        let positions = indices
            .iter()
            .map(|i| (string(&ops[*i]["id"]).to_owned(), *i))
            .collect::<BTreeMap<_, _>>();
        let mut predecessors = indices
            .iter()
            .map(|i| (*i, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let mut outgoing = predecessors.clone();
        for edge in rows(&v["relationships"]) {
            if !["data_dependency", "control_dependency"].contains(&string(&edge["kind"])) {
                continue;
            }
            if let (Some(from), Some(to)) = (
                positions.get(string(&edge["from_ref"])),
                positions.get(string(&edge["to_ref"])),
            ) {
                if from != to {
                    predecessors.get_mut(to).unwrap().insert(*from);
                    outgoing.get_mut(from).unwrap().insert(*to);
                }
            }
        }
        let mut incoming = predecessors.clone();
        let sort = |a: &usize, b: &usize| {
            let left = ops[*a]["native_index"]
                .as_f64()
                .unwrap_or(9_007_199_254_740_991.);
            let right = ops[*b]["native_index"]
                .as_f64()
                .unwrap_or(9_007_199_254_740_991.);
            left.partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    string(&ops[*a]["id"])
                        .encode_utf16()
                        .cmp(string(&ops[*b]["id"]).encode_utf16())
                })
        };
        let mut ready = indices
            .iter()
            .copied()
            .filter(|i| incoming[i].is_empty())
            .collect::<Vec<_>>();
        ready.sort_by(sort);
        let mut ordered = vec![];
        let mut depths: BTreeMap<usize, usize> = BTreeMap::new();
        while !ready.is_empty() {
            let current = ready.remove(0);
            ordered.push(current);
            let depth = predecessors[&current]
                .iter()
                .map(|i| depths[i] + 1)
                .max()
                .unwrap_or(0);
            depths.insert(current, depth);
            for next in &outgoing[&current] {
                incoming.get_mut(next).unwrap().remove(&current);
                if incoming[next].is_empty() {
                    ready.push(*next);
                    ready.sort_by(sort);
                }
            }
        }
        let visited = ordered.iter().copied().collect::<BTreeSet<_>>();
        let mut remaining = indices
            .iter()
            .copied()
            .filter(|i| !visited.contains(i))
            .collect::<Vec<_>>();
        remaining.sort_by(sort);
        ordered.extend(remaining);
        for (display, i) in ordered.into_iter().enumerate() {
            ops[i]["display_order"] = json!(display);
            ops[i]["dependency_order"] = json!(depths.get(&i));
        }
    }
    let complete = |field: &str| {
        ops.iter()
            .all(|r| crate::portable_onnx_semantics::safe(&r[field]).is_some())
    };
    let data = rows(&v["relationships"])
        .iter()
        .any(|r| ["data_dependency", "control_dependency"].contains(&string(&r["kind"])));
    let summary = json!({"source_order_status":if complete("source_order"){"complete"}else{"partial"},"dependency_order_status":if complete("dependency_order"){"complete_acyclic_projection"}else if data{"partial_or_cyclic"}else{"complete_no_inter_operation_dependencies"},"display_order_status":if complete("display_order"){"complete_deterministic"}else{"partial"},"runtime_order_status":"not_observed"});
    Ok(json!({"operations":ops,"summary":summary}))
}
