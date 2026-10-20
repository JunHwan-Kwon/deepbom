//! Declared metadata relationships, OMOP source mapping and bounded BOM resolution.
//! No file or network access and no promotion of declarations to attestations.
use crate::portable_validation::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
const INPUT: &str = "deepbom.provenance_input.v1";
const OMOP: &str = "deepbom.omop_metadata_input.v1";
const PRIMARY: &str = "artifact:primary";
const KINDS: &[&str] = &[
    "model_artifact",
    "dataset",
    "data_release",
    "manifest",
    "cohort_definition",
    "feature_contract",
    "software",
    "environment",
    "run",
    "evaluation_report",
    "quality_report",
    "protocol",
    "bom",
    "document",
];
const ROLES: &[&str] = &[
    "associated_with",
    "uses_training_data",
    "uses_evaluation_data",
    "uses_calibration_data",
    "derived_from",
    "generated_by",
    "uses_code",
    "uses_environment",
    "defined_by",
    "uses_features",
    "evaluated_by",
    "has_quality_report",
    "references_bom",
    "documented_by",
];
const REPORTS: &[&str] = &[
    "manifest",
    "cohort_definition",
    "feature_contract",
    "run",
    "evaluation_report",
    "quality_report",
    "protocol",
    "bom",
    "document",
];
const FIELDS: &[&str] = &[
    "cdm_source_name",
    "cdm_source_abbreviation",
    "cdm_holder",
    "source_description",
    "source_documentation_reference",
    "cdm_etl_reference",
    "source_release_date",
    "cdm_release_date",
    "cdm_version",
    "cdm_version_concept_id",
    "vocabulary_version",
];
const REQUIRED: &[&str] = &[
    "cdm_source_name",
    "cdm_source_abbreviation",
    "cdm_holder",
    "source_release_date",
    "cdm_release_date",
    "cdm_version_concept_id",
    "vocabulary_version",
];
const BOUNDARY:&str="Metadata relationships are supplied declarations, not proof of training, deployment, execution, clinical suitability or FAIR conformance. Matching hashes bind only the bytes actually provided. Manifest identity is not database identity. Schema, reference resolution, content integrity, attribute comparison and publisher authenticity are separate checks; signatures are not verified.";
fn object(v: &Value, required: &[&str], optional: &[&str], label: &str) -> R<()> {
    let m = v
        .as_object()
        .ok_or_else(|| format!("{label} must be an object."))?;
    need(
        m.keys()
            .all(|k| required.contains(&k.as_str()) || optional.contains(&k.as_str())),
        &format!("{label} contains an unknown field."),
    )?;
    need(
        required.iter().all(|k| m.contains_key(*k)),
        &format!("{label} is missing a required field."),
    )
}
fn txt(v: &Value, label: &str, max: usize, empty: bool) -> R<()> {
    need(
        v.as_str().is_some_and(|s| {
            s.encode_utf16().count() <= max
                && (empty
                    || !s
                        .trim_matches(crate::portable_evidence::js_space)
                        .is_empty())
                && !s
                    .chars()
                    .any(|c| (c as u32) < 32 && !['\t', '\n', '\r'].contains(&c))
        }),
        &format!("{label} must be bounded text."),
    )
}
fn id(v: &Value, label: &str) -> R<()> {
    txt(v, label, 256, false)?;
    let s = string(v);
    need(
        s.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:/#-".contains(&b)),
        &format!("{label} contains unsupported identifier characters."),
    )
}
fn path(v: &Value) -> R<()> {
    txt(v, "file", 1024, false)?;
    let s = string(v);
    need(
        !s.starts_with('/')
            && !s.chars().any(|c| ['\\', ':', '\0'].contains(&c))
            && s.split('/').all(|p| !p.is_empty() && p != "." && p != ".."),
        "file must be a relative path without traversal or a URL.",
    )
}
fn hash_text(v: &Value, label: &str) -> R<()> {
    need(
        sha(v),
        &format!("{label} must be a lowercase full SHA-256."),
    )
}
fn bounded<'a>(v: &'a Value, limit: usize, label: &str) -> R<&'a [Value]> {
    let r = array(
        v,
        &format!("{label} exceeds its array bound or is not an array."),
    )?;
    need(
        r.len() <= limit,
        &format!("{label} exceeds its array bound or is not an array."),
    )?;
    Ok(r)
}
fn primitive(v: &Value, label: &str) -> R<()> {
    need(
        v.is_null()
            || v.is_boolean()
            || v.is_string()
            || v.as_f64().is_some_and(|n| {
                n.is_finite() && (n.fract() != 0. || n.abs() <= 9_007_199_254_740_991.)
            }),
        &format!("{label} must be a JSON scalar; encode large integers as decimal strings."),
    )?;
    if v.is_string() {
        txt(v, label, 4096, true)?;
    }
    Ok(())
}
fn sorted(rows: &mut [Value], field: &str) {
    rows.sort_by(|a, b| {
        string(&a[field])
            .encode_utf16()
            .cmp(string(&b[field]).encode_utf16())
    });
}
pub(crate) fn input(v: &Value) -> R<Value> {
    need(
        crate::portable_identity::canonical(v)?.len() <= 2 * 1024 * 1024,
        "metadata input exceeds 2 MiB.",
    )?;
    object(
        v,
        &["schema", "subject", "nodes", "relationships"],
        if v["schema"] == OMOP { &["omop"] } else { &[] },
        "input",
    )?;
    need(
        [INPUT, OMOP].contains(&string(&v["schema"])),
        "unsupported metadata input schema.",
    )?;
    if v["schema"] == OMOP {
        need(
            v.get("omop").is_some(),
            "OMOP input requires omop metadata.",
        )?;
    }
    let s = &v["subject"];
    object(
        s,
        &["artifact_sha256", "artifact_set_sha256"],
        &["model_ir_sha256"],
        "subject",
    )?;
    hash_text(&s["artifact_sha256"], "subject.artifact_sha256")?;
    if !s["artifact_set_sha256"].is_null() {
        hash_text(&s["artifact_set_sha256"], "subject.artifact_set_sha256")?;
    }
    if let Some(h) = s.get("model_ir_sha256") {
        hash_text(h, "subject.model_ir_sha256")?;
    }
    let mut seen = BTreeSet::from([PRIMARY, "omop:release", "omop:context"]);
    for (i, node) in bounded(&v["nodes"], 256, "nodes")?.iter().enumerate() {
        let label = format!("nodes[{i}]");
        object(
            node,
            &["id", "kind", "name"],
            &["version", "uri", "sha256", "file", "attributes", "bom"],
            &label,
        )?;
        id(&node["id"], &format!("{label}.id"))?;
        need(
            seen.insert(string(&node["id"])),
            &format!("{label}.id is duplicated or reserved."),
        )?;
        need(
            KINDS.contains(&string(&node["kind"])),
            &format!("{label}.kind is unsupported."),
        )?;
        txt(&node["name"], &format!("{label}.name"), 4096, false)?;
        for key in ["version", "uri"] {
            if let Some(val) = node.get(key) {
                txt(val, &format!("{label}.{key}"), 4096, false)?;
            }
        }
        if let Some(h) = node.get("sha256") {
            hash_text(h, &format!("{label}.sha256"))?;
        }
        if let Some(f) = node.get("file") {
            path(f)?;
        }
        let mut attributes = BTreeSet::new();
        let empty = json!([]);
        for (at, a) in bounded(
            node.get("attributes")
                .filter(|v| !v.is_null())
                .unwrap_or(&empty),
            128,
            &format!("{label}.attributes"),
        )?
        .iter()
        .enumerate()
        {
            object(
                a,
                &["name", "value"],
                &[],
                &format!("{label}.attributes[{at}]"),
            )?;
            txt(&a["name"], "attribute name", 256, false)?;
            primitive(&a["value"], "attribute value")?;
            need(
                attributes.insert(string(&a["name"])),
                &format!("{label} repeats an attribute name."),
            )?;
        }
        if let Some(bom) = node.get("bom") {
            need(
                node["kind"] == "bom",
                &format!("{label}.bom requires kind=bom."),
            )?;
            object(
                bom,
                &[
                    "format",
                    "spec_version",
                    "document_id",
                    "document_version",
                    "element_ref",
                ],
                &[],
                &format!("{label}.bom"),
            )?;
            for key in ["format", "spec_version", "document_id", "element_ref"] {
                txt(&bom[key], &format!("{label}.bom.{key}"), 4096, false)?;
            }
            need(
                bom["document_version"].is_null()
                    || bom["document_version"]
                        .as_u64()
                        .is_some_and(|n| (1..=9_007_199_254_740_991).contains(&n)),
                "BOM document_version must be a positive safe integer or null.",
            )?;
        }
    }
    let mut edge_ids = BTreeSet::new();
    let mut reference_count = 0;
    for (i, edge) in bounded(&v["relationships"], 1024, "relationships")?
        .iter()
        .enumerate()
    {
        object(
            edge,
            &["id", "from", "to", "role"],
            &["evidence_refs"],
            &format!("relationships[{i}]"),
        )?;
        for key in ["id", "from", "to"] {
            id(&edge[key], &format!("relationship.{key}"))?;
        }
        need(
            !seen.contains(string(&edge["id"])) && edge_ids.insert(string(&edge["id"])),
            "relationship id is duplicated, reserved or collides with a node.",
        )?;
        need(
            ROLES.contains(&string(&edge["role"])),
            "relationship role is unsupported.",
        )?;
        let empty = json!([]);
        let refs = bounded(
            edge.get("evidence_refs")
                .filter(|v| !v.is_null())
                .unwrap_or(&empty),
            256,
            "evidence_refs",
        )?;
        reference_count += refs.len();
        need(
            reference_count <= 4096,
            "relationship evidence references exceed the total bound.",
        )?;
        let mut seen = BTreeSet::new();
        for r in refs {
            id(r, "evidence_ref")?;
            need(seen.insert(string(r)), "evidence_refs contains duplicates.")?;
        }
    }
    Ok(v.clone())
}
pub(crate) fn cdm_source(value: &Value) -> R<Value> {
    let row = if let Some(a) = value.as_array() {
        need(
            a.len() == 1,
            "CDM_SOURCE import requires exactly one explicitly selected row.",
        )?;
        &a[0]
    } else {
        value
    };
    let map = row
        .as_object()
        .ok_or("CDM_SOURCE must be a JSON object or a one-row JSON array.")?;
    need(map.len() <= 128, "CDM_SOURCE has too many fields.")?;
    let mut normalized = serde_json::Map::new();
    for (k, v) in map {
        txt(&json!(k), "CDM_SOURCE column", 128, false)?;
        primitive(v, &format!("CDM_SOURCE.{k}"))?;
        let name = k.to_lowercase();
        need(
            name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                && name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                && !normalized.contains_key(&name),
            "CDM_SOURCE contains invalid or case-colliding column names.",
        )?;
        normalized.insert(name, v.clone());
    }
    Ok(Value::Object(normalized))
}
pub(crate) fn omop_profile(input: &Value) -> R<Value> {
    let p = &input["omop"];
    object(
        p,
        &["cdm_version", "instance_id", "release_id", "cdm_source"],
        &[],
        "omop",
    )?;
    for k in ["cdm_version", "instance_id", "release_id"] {
        txt(&p[k], &format!("omop.{k}"), 256, false)?;
    }
    let fields = cdm_source(&p["cdm_source"])?;
    let version = string(&p["cdm_version"])
        .strip_prefix('v')
        .unwrap_or(string(&p["cdm_version"]));
    let pieces: Vec<_> = version.split('.').collect();
    need(
        (2..=3).contains(&pieces.len())
            && pieces
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit())),
        "omop.cdm_version must be an explicit version.",
    )?;
    if !fields["cdm_version"].is_null() && fields["cdm_version"] != "" {
        need(
            string(&fields["cdm_version"])
                .strip_prefix('v')
                .unwrap_or(string(&fields["cdm_version"]))
                == version,
            "CDM_SOURCE.cdm_version contradicts omop.cdm_version.",
        )?;
    }
    for k in ["source_release_date", "cdm_release_date"] {
        let date = &fields[k];
        if date.is_null() || *date == "" {
            continue;
        }
        need(
            string(date).len() == 10
                && crate::portable_evidence::timestamp(&json!(format!(
                    "{}T00:00:00Z",
                    string(date)
                )))
                .is_ok(),
            &format!("CDM_SOURCE.{k} must be a real ISO calendar date."),
        )?;
    }
    let concept = &fields["cdm_version_concept_id"];
    if !concept.is_null() && *concept != "" {
        need(
            concept.as_u64().is_some_and(|n| n <= 9_007_199_254_740_991)
                || (concept.is_string() && decimal(concept).is_ok()),
            "CDM_SOURCE.cdm_version_concept_id must be an exact nonnegative integer.",
        )?;
    }
    let supported = ["5.4", "5.5"].contains(&version);
    let mut known: BTreeSet<_> = FIELDS.iter().copied().collect();
    if version == "5.5" {
        known.insert("cdm_release_identifier");
    }
    if supported {
        for k in &known {
            let value = &fields[*k];
            if *k == "cdm_version_concept_id" || value.is_null() || *value == "" {
                continue;
            }
            need(
                value.is_string(),
                &format!("CDM_SOURCE.{k} must retain its text or date representation."),
            )?;
            let max = match *k {
                "source_description" => 4096,
                "cdm_source_abbreviation" => 25,
                "vocabulary_version" => 20,
                "cdm_version" => 10,
                _ => 255,
            };
            need(
                string(value).chars().count() <= max,
                &format!("CDM_SOURCE.{k} exceeds the import profile length bound."),
            )?;
        }
    }
    if version == "5.5"
        && !fields["cdm_release_identifier"].is_null()
        && fields["cdm_release_identifier"] != ""
    {
        need(
            fields["cdm_release_identifier"] == p["release_id"],
            "CDM_SOURCE.cdm_release_identifier contradicts omop.release_id.",
        )?;
    }
    let is_array = p["cdm_source"].is_array();
    let original = if is_array {
        &p["cdm_source"][0]
    } else {
        &p["cdm_source"]
    };
    let mut ledger = vec![];
    for column in original.as_object().unwrap().keys() {
        let name = column.to_lowercase();
        ledger.push(json!({"source_pointer":format!("/omop/cdm_source/{}{column}",if is_array{"0/"}else{""}),"name":format!("omop:cdm_source:{name}"),"value":fields[&name],"status":if supported&&known.contains(name.as_str()){"mapped_declared"}else{"preserved_unsupported_mapping"}}));
    }
    let missing: Vec<_> = REQUIRED
        .iter()
        .filter(|k| {
            fields[**k].is_null()
                || fields[**k].as_str().is_some_and(|s| {
                    s.trim_matches(crate::portable_evidence::js_space)
                        .is_empty()
                })
        })
        .copied()
        .collect();
    let mut attributes = vec![
        json!({"name":"omop:instance_id","value":p["instance_id"]}),
        json!({"name":"omop:release_id","value":p["release_id"]}),
        json!({"name":"omop:cdm_version","value":version}),
    ];
    attributes.extend(
        ledger
            .iter()
            .map(|r| json!({"name":r["name"],"value":r["value"]})),
    );
    Ok(
        json!({"node":{"id":"omop:release","kind":"data_release","name":if text(&fields["cdm_source_name"],usize::MAX){&fields["cdm_source_name"]}else{&p["instance_id"]},"version":p["release_id"],"attributes":attributes},"field_ledger":ledger,"missing_fields":missing,"status":if supported{"supported_import_profile"}else{"unsupported_cdm_version"},"cdm_version":version}),
    )
}
pub(crate) fn normalize(v: &Value) -> R<Value> {
    let input = input(v)?;
    let profile = if input["schema"] == OMOP {
        omop_profile(&input)?
    } else {
        Value::Null
    };
    let mut nodes = vec![];
    let mut relationships = vec![];
    if !profile.is_null() {
        nodes.push(profile["node"].clone());
        relationships.push(json!({"id":"omop:context","from":PRIMARY,"to":"omop:release","role":"associated_with","evidence_refs":[]}));
    }
    nodes.extend_from_slice(rows(&input["nodes"]));
    relationships.extend_from_slice(rows(&input["relationships"]));
    Ok(json!({"input":input,"profile":profile,"nodes":nodes,"relationships":relationships}))
}
fn check(checks: &mut Vec<Value>, subject: &Value, kind: &str, status: &str, message: &str) {
    checks.push(json!({"subject_ref":subject,"kind":kind,"status":status,"message":message}));
}
pub(crate) fn bom(
    node: &Value,
    observation: &Value,
    model_hash: &Value,
    current: bool,
    observation_index: i64,
) -> R<Value> {
    let mut checks = vec![];
    let mut fields = vec![];
    let id = &node["id"];
    let reference = &node["bom"];
    let doc = &observation["document"];
    if reference.is_null() {
        check(
            &mut checks,
            id,
            "bom_reference",
            "not_assessed",
            "BOM document/element identity was not supplied.",
        );
        return Ok(json!({"checks":checks,"fields":fields}));
    }
    if doc.is_null() {
        check(
            &mut checks,
            id,
            "bom_reference",
            "unresolved",
            "BOM bytes were not supplied; a URI is not fetched automatically.",
        );
        return Ok(json!({"checks":checks,"fields":fields}));
    }
    need(
        crate::portable_identity::canonical(doc)?.len() <= 2 * 1024 * 1024,
        "parsed BOM exceeds 2 MiB.",
    )?;
    if reference["format"] != "CycloneDX" || reference["spec_version"] != "1.7" {
        check(&mut checks,id,"bom_reference","unsupported","This resolver supports CycloneDX 1.7 reference identity; other formats remain preserved and unresolved.");
        return Ok(json!({"checks":checks,"fields":fields}));
    }
    if doc["bomFormat"] != reference["format"]
        || doc["specVersion"] != reference["spec_version"]
        || doc["serialNumber"] != reference["document_id"]
        || doc["version"] != reference["document_version"]
    {
        check(&mut checks,id,"bom_document_identity","mismatch","BOM format, specification version, document identity or revision differs from the reference.");
        return Ok(json!({"checks":checks,"fields":fields}));
    }
    check(
        &mut checks,
        id,
        "bom_document_identity",
        "match",
        "Supplied BOM document identity matches; this is not full schema or signature validation.",
    );
    let mut pending = vec![];
    if !doc["metadata"]["component"].is_null() {
        pending.push((
            &doc["metadata"]["component"],
            "/metadata/component".to_owned(),
        ));
    }
    for (i, c) in rows(&doc["components"]).iter().enumerate() {
        pending.push((c, format!("/components/{i}")));
    }
    let mut inventory = vec![];
    while let Some((component, pointer)) = pending.pop() {
        need(
            inventory.len() < 4096,
            "BOM component reference inventory exceeds 4096 entries.",
        )?;
        need(component.is_object(), "BOM component must be an object.")?;
        if let Some(children) = component.get("components") {
            for (i, c) in array(children, "BOM components must be an array.")?
                .iter()
                .enumerate()
            {
                pending.push((c, format!("{pointer}/components/{i}")));
            }
        }
        inventory.push((component, pointer));
    }
    let mut refs = BTreeSet::new();
    if inventory
        .iter()
        .filter_map(|(c, _)| c["bom-ref"].as_str())
        .any(|r| !refs.insert(r))
    {
        check(
            &mut checks,
            id,
            "bom_element_identity",
            "mismatch",
            "BOM contains duplicate component references.",
        );
    }
    let selected: Vec<_> = inventory
        .iter()
        .filter(|(c, _)| c["bom-ref"] == reference["element_ref"])
        .collect();
    if selected.len() != 1 {
        check(
            &mut checks,
            id,
            "bom_element_identity",
            if selected.is_empty() {
                "unresolved"
            } else {
                "mismatch"
            },
            "The BOM element does not resolve to exactly one component.",
        );
        return Ok(json!({"checks":checks,"fields":fields}));
    }
    check(
        &mut checks,
        id,
        "bom_element_identity",
        "match",
        "Referenced component was found in the supplied document.",
    );
    let (component, pointer) = selected[0];
    if current {
        check(
            &mut checks,
            id,
            "bom_subject_kind",
            if component["type"] == "machine-learning-model" {
                "match"
            } else {
                "mismatch"
            },
            "A model BOM reference must resolve to a machine-learning-model component.",
        );
        let hashes: Vec<_> = rows(&component["hashes"])
            .iter()
            .filter(|r| r["alg"] == "SHA-256")
            .collect();
        check(
            &mut checks,
            id,
            "bom_model_sha256",
            if hashes.is_empty() {
                "not_assessed"
            } else if hashes.len() == 1
                && json!(string(&hashes[0]["content"]).to_ascii_lowercase()) == *model_hash
            {
                "match"
            } else {
                "mismatch"
            },
            if hashes.is_empty() {
                "The referenced component does not declare SHA-256."
            } else {
                "Compare the referenced component SHA-256 with the active model bytes."
            },
        );
    }
    let empty = json!([]);
    let properties = bounded(
        component
            .get("properties")
            .filter(|v| !v.is_null())
            .unwrap_or(&empty),
        4096,
        "BOM property inventory",
    )?;
    for (i, p) in properties.iter().enumerate() {
        fields.push(json!({"source_document":"provenance_ir","source_pointer":format!("/observations/{observation_index}/document{pointer}/properties/{i}"),"subject_ref":id,"name":p["name"].as_str().unwrap_or("<invalid property>"),"value":p["value"],"status":"preserved_unsupported_mapping"}));
    }
    if !properties.is_empty() {
        check(&mut checks,id,"bom_attribute_reconciliation","unsupported","Component properties are inventoried, not verified by reference resolution. Use verify --bom for supported artifact-attribute reconciliation.");
    }
    Ok(json!({"checks":checks,"fields":fields}))
}
fn role(role: &str, from: &str, to: &str) -> bool {
    match role {
        "uses_training_data" | "uses_evaluation_data" | "uses_calibration_data" => {
            ["model_artifact", "run"].contains(&from) && ["dataset", "data_release"].contains(&to)
        }
        "generated_by" => {
            [
                "model_artifact",
                "dataset",
                "data_release",
                "evaluation_report",
                "quality_report",
            ]
            .contains(&from)
                && to == "run"
        }
        "uses_code" => to == "software",
        "uses_environment" => to == "environment",
        "defined_by" => ["cohort_definition", "protocol"].contains(&to),
        "uses_features" => to == "feature_contract",
        "evaluated_by" => from == "model_artifact" && to == "evaluation_report",
        "has_quality_report" => to == "quality_report",
        "references_bom" => to == "bom",
        "documented_by" => REPORTS.contains(&to),
        "derived_from" => {
            ["model_artifact", "dataset", "data_release"].contains(&from)
                && ["model_artifact", "dataset", "data_release"].contains(&to)
        }
        _ => true,
    }
}
fn cycle(edges: &[Value]) -> bool {
    let mut incoming = BTreeMap::new();
    let mut out: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in edges.iter().filter(|e| e["role"] == "derived_from") {
        let a = string(&edge["from"]);
        let b = string(&edge["to"]);
        incoming.entry(a).or_insert(0usize);
        *incoming.entry(b).or_insert(0) += 1;
        out.entry(a).or_default().push(b);
    }
    let mut ready: VecDeque<_> = incoming
        .iter()
        .filter(|(_, n)| **n == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut seen = 0;
    while let Some(id) = ready.pop_front() {
        seen += 1;
        for next in out.get(id).into_iter().flatten() {
            let n = incoming.get_mut(next).unwrap();
            *n -= 1;
            if *n == 0 {
                ready.push_back(*next);
            }
        }
    }
    seen != incoming.len()
}
pub(crate) fn build(model: &Value, metadata: &Value, observations: &Value) -> R<Value> {
    let source = json!({"model_ir_sha256":model["model_ir_sha256"],"artifact_sha256":model["artifact"]["sha256"],"artifact_set_sha256":model["artifact_set"]["artifact_set_sha256"]});
    let normalized = normalize(metadata)?;
    let input = &normalized["input"];
    let profile = &normalized["profile"];
    need(
        input["subject"]["artifact_sha256"] == source["artifact_sha256"]
            && input["subject"]["artifact_set_sha256"] == source["artifact_set_sha256"],
        "metadata subject does not match the active artifact and artifact-set identity.",
    )?;
    if let Some(h) = input["subject"].get("model_ir_sha256") {
        need(
            *h == source["model_ir_sha256"],
            "metadata references a different Model IR.",
        )?;
    }
    let mut nodes = vec![
        json!({"id":PRIMARY,"kind":"model_artifact","name":model["artifact"]["filename"],"sha256":source["artifact_sha256"],"evidence_class":"OBSERVED_SERIALIZED_ARTIFACT","attributes":[]}),
    ];
    for n in rows(&normalized["nodes"]) {
        let mut n = n.clone();
        if n["attributes"].is_null() {
            n["attributes"] = json!([]);
        }
        n["evidence_class"] = json!("DECLARED_UNVERIFIED");
        nodes.push(n);
    }
    sorted(&mut nodes, "id");
    let node_index: BTreeMap<_, _> = nodes.iter().map(|n| (string(&n["id"]), n)).collect();
    let mut seen = BTreeSet::new();
    let mut bytes = 0u64;
    let mut observed = vec![];
    for row in bounded(observations, 256, "observations")? {
        object(
            row,
            &["node_ref", "sha256", "byte_length"],
            &["document"],
            "observation",
        )?;
        let node = node_index.get(string(&row["node_ref"]));
        need(
            node.is_some() && row["node_ref"] != PRIMARY && seen.insert(string(&row["node_ref"])),
            "observation must refer to one unique metadata node.",
        )?;
        hash_text(&row["sha256"], "observed SHA-256")?;
        let size = row["byte_length"]
            .as_u64()
            .filter(|n| *n <= 32 * 1024 * 1024)
            .ok_or("observed file length exceeds its bound.")?;
        bytes += size;
        need(
            bytes <= 64 * 1024 * 1024,
            "evidence files exceed the 64 MiB total bound.",
        )?;
        if let Some(doc) = row.get("document") {
            need(
                node.unwrap()["kind"] == "bom" && doc.is_object(),
                "only BOM observations may include a parsed document.",
            )?;
        }
        observed.push(row.clone());
    }
    sorted(&mut observed, "node_ref");
    let by_observation: BTreeMap<_, _> = observed
        .iter()
        .enumerate()
        .map(|(i, r)| (string(&r["node_ref"]), (i, r)))
        .collect();
    let mut checks = vec![];
    let mut ledger = rows(&profile["field_ledger"]).to_vec();
    for r in &mut ledger {
        r["source_document"] = json!("input_document");
        r["subject_ref"] = json!("omop:release");
    }
    check(
        &mut checks,
        &json!(PRIMARY),
        "model_binding",
        "match",
        "Metadata is bound to the current artifact identity.",
    );
    check(
        &mut checks,
        &json!(PRIMARY),
        "artifact_set_binding",
        if source["artifact_set_sha256"].is_null() {
            "not_assessed"
        } else {
            "match"
        },
        if source["artifact_set_sha256"].is_null() {
            "Artifact-set identity is not declared in this Model IR."
        } else {
            "The declared artifact-set digest matches the current Model IR source."
        },
    );
    if !profile.is_null() {
        check(
            &mut checks,
            &json!("omop:release"),
            "omop_profile",
            if profile["status"] == "supported_import_profile" {
                "match"
            } else {
                "unsupported"
            },
            &format!(
                "OMOP {} metadata import profile; this is not CDM conformance validation.",
                string(&profile["cdm_version"])
            ),
        );
        for name in rows(&profile["missing_fields"]) {
            check(
                &mut checks,
                &json!("omop:release"),
                &format!("missing:{}", string(name)),
                "not_assessed",
                &format!("CDM_SOURCE.{} was not supplied.", string(name)),
            );
        }
        if ledger
            .iter()
            .any(|r| r["status"] == "preserved_unsupported_mapping")
        {
            check(
                &mut checks,
                &json!("omop:release"),
                "omop_field_mapping",
                "unsupported",
                "Unmapped CDM_SOURCE fields are preserved in the field ledger.",
            );
        }
    }
    for (i, node) in rows(&input["nodes"]).iter().enumerate() {
        for (j, a) in rows(&node["attributes"]).iter().enumerate() {
            ledger.push(json!({"source_document":"input_document","source_pointer":format!("/nodes/{i}/attributes/{j}"),"subject_ref":node["id"],"name":a["name"],"value":a["value"],"status":"preserved_unsupported_mapping"}));
        }
    }
    for node in &nodes {
        let id = &node["id"];
        if *id == PRIMARY {
            continue;
        }
        if *id != "omop:release" && !rows(&node["attributes"]).is_empty() {
            check(&mut checks,id,"attribute_mapping","unsupported","Supplied attributes are preserved as declarations; their meaning and truth have not been verified.");
        }
        let observation = by_observation.get(string(id));
        if !node["sha256"].is_null() || !node["file"].is_null() || observation.is_some() {
            let status = if observation.is_none() || node["sha256"].is_null() {
                "not_assessed"
            } else if observation.unwrap().1["sha256"] == node["sha256"] {
                "match"
            } else {
                "mismatch"
            };
            let message = if observation.is_none() {
                "Referenced bytes were not provided."
            } else if node["sha256"].is_null() {
                "File SHA-256 was measured, but no independent expected digest was declared."
            } else {
                "Compare the provided file bytes with the declared digest; relationship truth remains unverified."
            };
            check(&mut checks, id, "file_sha256", status, message);
        }
        if node["kind"] == "bom" {
            let current = rows(&normalized["relationships"])
                .iter()
                .any(|e| e["from"] == PRIMARY && e["to"] == *id && e["role"] == "references_bom");
            let result = bom(
                node,
                observation.map(|r| r.1).unwrap_or(&Value::Null),
                &source["artifact_sha256"],
                current,
                observation.map(|r| r.0 as i64).unwrap_or(-1),
            )?;
            checks.extend_from_slice(rows(&result["checks"]));
            ledger.extend_from_slice(rows(&result["fields"]));
        }
    }
    let mut relationships = vec![];
    for edge in rows(&normalized["relationships"]) {
        let from = node_index.get(string(&edge["from"]));
        let to = node_index.get(string(&edge["to"]));
        let resolved = from.is_some() && to.is_some();
        let kinds = resolved
            && edge["from"] != edge["to"]
            && role(
                string(&edge["role"]),
                string(&from.unwrap()["kind"]),
                string(&to.unwrap()["kind"]),
            );
        check(
            &mut checks,
            &edge["id"],
            "relationship_reference",
            if !resolved {
                "unresolved"
            } else if kinds {
                "match"
            } else {
                "mismatch"
            },
            if !resolved {
                "Relationship endpoint is absent from the node inventory."
            } else if kinds {
                "Endpoints and relationship kinds are consistent; the relationship remains declared."
            } else {
                "Relationship endpoints contradict the role or form a self-reference."
            },
        );
        for r in rows(&edge["evidence_refs"]) {
            let n = node_index.get(string(r));
            check(&mut checks,&edge["id"],&format!("evidence_reference:{}",string(r)),if n.is_none(){"unresolved"}else if REPORTS.contains(&string(&n.unwrap()["kind"])){"match"}else{"mismatch"},"Evidence references must resolve to a supporting document, manifest or run record.");
        }
        let mut e = edge.clone();
        e["evidence_refs"] = json!(rows(&edge["evidence_refs"]));
        e["evidence_class"] = json!("DECLARED_UNVERIFIED");
        e["resolution"] = json!(if !resolved {
            "unresolved"
        } else if kinds {
            "resolved"
        } else {
            "kind_mismatch"
        });
        e["claim_verification"] = json!("not_assessed");
        relationships.push(e);
    }
    sorted(&mut relationships, "id");
    if cycle(&relationships) {
        check(
            &mut checks,
            &json!("relationships"),
            "derivation_cycle",
            "mismatch",
            "Derived-from relationships contain a cycle.",
        );
    }
    checks.sort_by(|a, b| {
        format!("{}\0{}", string(&a["subject_ref"]), string(&a["kind"]))
            .encode_utf16()
            .cmp(format!("{}\0{}", string(&b["subject_ref"]), string(&b["kind"])).encode_utf16())
    });
    let mut counts = json!({});
    for status in [
        "match",
        "mismatch",
        "not_assessed",
        "unresolved",
        "unsupported",
    ] {
        counts[status] = json!(checks.iter().filter(|r| r["status"] == status).count());
    }
    let status = if counts["mismatch"] != 0 {
        "contradiction_observed"
    } else if ["unresolved", "not_assessed", "unsupported"]
        .iter()
        .any(|s| counts[*s] != 0)
    {
        "incomplete"
    } else {
        "no_contradiction_observed"
    };
    let mut body = json!({"schema":"deepbom.provenance_ir.v1","method_version":"1.0.0","hash_contract":{"algorithm":"SHA-256","canonicalization":"RFC8785-JCS","source_encoding":"UTF-8","excluded_pointers":["/provenance_ir_sha256"]},"source":source,"input_document":input,"input_sha256":hash(input)?,"profile":if profile.is_null(){json!({"id":"deepbom.generic_metadata_profile.v1","status":"supported_import_profile"})}else{json!({"id":"deepbom.omop_metadata_profile.v1","cdm_version":profile["cdm_version"],"status":profile["status"],"missing_source_fields":profile["missing_fields"]})},"coverage":{"node_count":nodes.len(),"relationship_count":relationships.len(),"observed_file_count":observed.len(),"check_count":checks.len(),"check_counts":counts,"field_count":ledger.len(),"mapped_declared_field_count":ledger.iter().filter(|r|r["status"]=="mapped_declared").count(),"unsupported_field_count":ledger.iter().filter(|r|r["status"]=="preserved_unsupported_mapping").count(),"attested_relationship_count":0},"nodes":nodes,"relationships":relationships,"observations":observed,"checks":checks,"field_ledger":ledger,"verdict":{"status":status,"scope":"binding_reference_and_supplied_file_consistency","metadata_truth_verified":false,"publisher_authenticity":"not_verified"},"interpretation_boundary":BOUNDARY});
    body["provenance_ir_sha256"] = json!(hash(&body)?);
    Ok(body)
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let result = (|| match string(&v["operation"]) {
        "input" => input(&v["value"]),
        "cdm_source" => cdm_source(&v["value"]),
        "omop_profile" => omop_profile(&v["value"]),
        "normalize" => normalize(&v["value"]),
        "bom" => bom(
            &v["node"],
            &v["observation"],
            &v["model_sha256"],
            v["current"] == true,
            v["observation_index"].as_i64().unwrap_or(-1),
        ),
        "build" => build(&v["model"], &v["value"], &v["observations"]),
        "validate" => {
            let d = &v["value"];
            need(
                d["schema"] == "deepbom.provenance_ir.v1",
                "unsupported Provenance IR schema; regenerate evidence with the current producer.",
            )?;
            digest(d, "provenance_ir_sha256", "provenance_ir_sha256 mismatch.")?;
            let expected = build(&v["model"], &d["input_document"], &d["observations"])?;
            need(
                *d == expected,
                "IR does not reproduce from its bound source, input and observed-file records.",
            )?;
            Ok(expected)
        }
        _ => Err("Unknown metadata operation.".into()),
    })();
    result.map_err(|e| format!("Provenance: {e}"))
}
