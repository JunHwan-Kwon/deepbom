//! Portable IR type, metric and storage relationships. Hosts only collect source
//! facts and transport JSON. Unknown counts are never promoted to complete zero.
use crate::portable_contracts::{decimal, exact};
use num_bigint::BigUint;
use num_traits::Zero;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
type R<T> = Result<T, String>;
fn need(ok: bool, message: &str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn list<'a>(v: &'a Value, message: &str) -> R<&'a Vec<Value>> {
    v.as_array().ok_or_else(|| message.into())
}
fn nonempty(v: &Value) -> bool {
    v.as_str().is_some_and(|s| !s.is_empty())
}
fn safe(v: &Value) -> Option<u64> {
    v.as_u64().filter(|n| *n <= 9_007_199_254_740_991)
}
fn sha(v: &Value) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
fn count(v: &Value) -> R<Option<BigUint>> {
    if v.is_null() {
        return Ok(None);
    }
    let s = v["decimal"]
        .as_str()
        .ok_or("IR exact integer decimal is invalid.")?;
    need(
        !s.is_empty() && (s == "0" || !s.starts_with('0')),
        "IR exact integer decimal is invalid.",
    )?;
    let n = decimal(&v["decimal"]).ok_or("IR exact integer decimal is invalid.")?;
    need(
        v.get("number") == Some(&exact(&n)["number"]),
        "IR exact integer numeric mirror contradicts its decimal value.",
    )?;
    Ok(Some(n))
}
fn same_keys(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)))
}
fn index<'a>(rows: &'a Value, message: &str) -> R<BTreeMap<&'a str, &'a Value>> {
    let mut result = BTreeMap::new();
    for row in list(rows, message)? {
        let id = row["id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or(message)?;
        need(result.insert(id, row).is_none(), message)?;
    }
    Ok(result)
}
fn visit_type(v: &Value, depth: usize) -> R<()> {
    need(
        depth <= 64 && v.is_object(),
        "IR value type tree is invalid.",
    )?;
    match v["kind"].as_str().unwrap_or("") {
        "tensor" | "sparse_tensor" => {
            let ranked = v["rank_status"] == "ranked";
            need(
                nonempty(&v["dtype"])
                    && (ranked || v["rank_status"] == "unknown_rank")
                    && if ranked {
                        v["dimensions"].is_array()
                    } else {
                        v.get("dimensions") == Some(&Value::Null)
                    },
                "IR tensor rank contract is invalid.",
            )?;
            if let Some(dims) = v["dimensions"].as_array() {
                for dim in dims {
                    need(dim.is_object(), "IR dimension contract is invalid.")?;
                    match dim["kind"].as_str().unwrap_or("") {
                        "constant" => {
                            let n = count(&dim["value"])
                                .map_err(|_| {
                                    "IR dimension exact count is inconsistent.".to_string()
                                })?
                                .ok_or("IR dimension exact count is inconsistent.")?;
                            need(
                                dim["value"] == exact(&n),
                                "IR dimension exact count is inconsistent.",
                            )?;
                        }
                        "symbol" => need(
                            nonempty(&dim["name"]) && nonempty(&dim["scope_ref"]),
                            "IR dimension symbol scope is invalid.",
                        )?,
                        "unknown" => {}
                        _ => return Err("IR dimension kind is invalid.".into()),
                    }
                }
            }
        }
        "sequence" | "optional" => visit_type(&v["element_type"], depth + 1)?,
        "map" => {
            need(nonempty(&v["key_dtype"]), "IR map key type is missing.")?;
            visit_type(&v["value_type"], depth + 1)?;
        }
        "unknown" | "opaque" => {}
        _ => return Err("IR value kind is invalid.".into()),
    }
    Ok(())
}
fn value_type(v: &Value) -> R<()> {
    need(
        v["schema"] == "deepbom.value_type.v1",
        "IR value type schema is invalid.",
    )?;
    visit_type(&v["root"], 0)
}
fn elements(v: &Value) -> R<Value> {
    value_type(v)?;
    let t = &v["root"];
    if !["tensor", "sparse_tensor"].contains(&t["kind"].as_str().unwrap_or(""))
        || t["rank_status"] != "ranked"
    {
        return Ok(Value::Null);
    }
    let dims = t["dimensions"].as_array().unwrap(); // validated before any zero shortcut
    if dims
        .iter()
        .any(|d| d["kind"] == "constant" && d["value"]["decimal"] == "0")
    {
        return Ok(exact(&BigUint::zero()));
    }
    if dims.iter().any(|d| d["kind"] != "constant") {
        return Ok(Value::Null);
    }
    Ok(crate::portable_contracts::count_product(
        &dims.iter().map(|d| d["value"].clone()).collect::<Vec<_>>(),
    )?
    .map(|n| exact(&n))
    .unwrap_or(Value::Null))
}
fn metric(v: &Value) -> R<()> {
    need(
        v["schema"] == "deepbom.metric.v1"
            && ["MAC", "byte", "element", "count"].contains(&v["unit"].as_str().unwrap_or(""))
            && v.get("denominator") == Some(&Value::Null),
        "IR metric unit or denominator is invalid.",
    )?;
    need(
        v.get("value").is_some(),
        "IR exact integer decimal is invalid.",
    )?;
    let n = count(&v["value"])?;
    let a = safe(&v["coverage"]["assessed"]);
    let e = safe(&v["coverage"]["eligible"]);
    need(
        a.is_some() && e.is_some() && a <= e,
        "IR metric coverage is invalid.",
    )?;
    need(
        n.is_none() || a == e,
        "IR metric complete value contradicts partial coverage.",
    )?;
    need(
        v["exactness"]
            == if n.is_none() {
                "not_assessable"
            } else {
                "exact_integer"
            }
            && (n.is_some() || nonempty(&v["reason"])),
        "IR metric exactness is inconsistent.",
    )?;
    let refs = list(
        &v["source_refs"],
        "IR metric source references are invalid.",
    )?;
    let mut seen = BTreeSet::new();
    need(
        refs.iter()
            .all(|r| nonempty(r) && seen.insert(r.as_str().unwrap())),
        "IR metric source references are invalid.",
    )
}
fn new_metric(v: &Value) -> R<Value> {
    let value = &v["value"];
    let m = json!({"schema":"deepbom.metric.v1","unit":v["unit"],"value":value,"exactness":if value.is_null(){"not_assessable"}else{"exact_integer"},"denominator":null,"coverage":{"assessed":v["assessed"],"eligible":v["eligible"]},"source_refs":v["source_refs"],"assumptions":v["assumptions"],"reason":if value.is_null(){v["reason"].as_str().filter(|s|!s.is_empty()).map(json_string).unwrap_or(json!("source_did_not_establish_a_complete_value"))}else{Value::Null},"evidence_class":"DERIVED"});
    metric(&m)?;
    Ok(m)
}
fn json_string(s: &str) -> Value {
    json!(s)
}
fn aggregate(v: &Value) -> R<Value> {
    let rows = list(&v["rows"], "IR metric rows are invalid.")?;
    let mut refs = vec![];
    let mut seen = BTreeSet::new();
    let mut total = BigUint::zero();
    let mut assessed = 0;
    for row in rows {
        metric(row)?;
        need(
            row["unit"] == v["unit"],
            "IR metric aggregation mixes units or ratios.",
        )?;
        for r in row["source_refs"].as_array().unwrap() {
            need(
                seen.insert(r.as_str().unwrap()),
                "IR metric aggregation would count a source twice.",
            )?;
            refs.push(r.clone());
        }
        if let Some(n) = count(&row["value"])? {
            total += n;
            assessed += 1;
        }
    }
    let subtotal = exact(&total);
    let total = new_metric(
        &json!({"value":if assessed==rows.len(){subtotal.clone()}else{Value::Null},"unit":v["unit"],"source_refs":refs,"assessed":assessed,"eligible":rows.len(),"reason":null,"assumptions":[]}),
    )?;
    Ok(json!({"total":total,"assessed_subtotal":subtotal}))
}
fn logical(v: &Value) -> R<()> {
    let inv = &v["inventory"];
    let rows = list(&inv["values"], "IR logical inventory count is invalid.")?;
    need(
        inv["schema"] == "deepbom.logical_inventory.v1"
            && safe(&inv["count"]) == Some(rows.len() as u64),
        "IR logical inventory count is invalid.",
    )?;
    let graph = index(
        &v["graph_values"],
        "IR logical graph identities are invalid.",
    )?;
    let objects = index(&v["objects"], "IR logical storage identities are invalid.")?;
    need(
        graph.is_empty() || rows.len() == graph.len(),
        "IR logical inventory count contradicts its graph values.",
    )?;
    let mut seen = BTreeSet::new();
    for row in rows {
        let id = row["id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("IR logical inventory has duplicate identities.")?;
        need(
            seen.insert(id),
            "IR logical inventory has duplicate identities.",
        )?;
        need(
            row["element_count"] == elements(&row["type_contract"])?,
            "IR logical element count contradicts the type.",
        )?;
        let mut storage_seen = BTreeSet::new();
        need(
            list(
                &row["storage_refs"],
                "IR logical storage binding is invalid.",
            )?
            .iter()
            .all(|r| {
                r.as_str()
                    .is_some_and(|r| objects.contains_key(r) && storage_seen.insert(r))
            }),
            "IR logical storage binding is invalid.",
        )?;
        if row.get("graph_value_ref") != Some(&Value::Null) || graph.contains_key(id) {
            let source = row["graph_value_ref"]
                .as_str()
                .and_then(|s| graph.get(s))
                .ok_or("IR logical inventory contradicts its graph value.")?;
            need(
                [
                    "id",
                    "name",
                    "dtype",
                    "shape",
                    "type_contract",
                    "storage_refs",
                ]
                .iter()
                .all(|k| row[*k] == source[*k]),
                "IR logical inventory contradicts its graph value.",
            )?;
        }
    }
    need(
        graph.keys().all(|id| seen.contains(id)),
        "IR logical inventory omitted a graph value.",
    )
}
fn range(v: &Value) -> R<(BigUint, BigUint)> {
    let start = exact_input(&v["start"]).ok_or("IR storage range is invalid.")?;
    let end = exact_input(&v["end_exclusive"]).ok_or("IR storage range is invalid.")?;
    need(end >= start, "IR storage range is reversed.")?;
    Ok((start, end))
}
fn bounds(row: &Value, member: &Value) -> R<Value> {
    if !member.is_object()
        || row["range"]["status"] != "exact"
        || row["range"]["offset_basis"] != "artifact_absolute"
        || member["kind"] == "bundle"
        || member["byte_length"].is_null()
    {
        return Ok(json!("not_assessable"));
    }
    let (_, end) = range(&row["range"])?;
    let size = count(&member["byte_length"])?.ok_or("IR member size is missing.")?;
    Ok(json!(if end > size {
        "outside_artifact_member"
    } else {
        "within_artifact_member"
    }))
}
fn overlaps(rows: &Value) -> R<Value> {
    let mut groups: BTreeMap<String, Vec<(&Value, BigUint, BigUint)>> = BTreeMap::new();
    for row in list(rows, "IR storage bindings are invalid.")? {
        if !nonempty(&row["member_ref"]) || row["range"]["status"] != "exact" {
            continue;
        }
        let (a, b) = range(&row["range"])?;
        if a == b {
            continue;
        } // empty half-open ranges occupy no storage and cannot overlap
        let key = serde_json::to_string(&json!([row["member_ref"], row["range"]["offset_basis"]]))
            .map_err(|e| e.to_string())?;
        groups.entry(key).or_default().push((row, a, b));
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
    let mut out = vec![];
    for (_, mut rows) in groups {
        rows.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| {
                a.0["storage_ref"]
                    .as_str()
                    .unwrap_or("")
                    .encode_utf16()
                    .cmp(b.0["storage_ref"].as_str().unwrap_or("").encode_utf16())
            })
        });
        let mut group: Vec<&Value> = vec![];
        let mut end = BigUint::zero();
        let flush = |group: &Vec<&Value>, out: &mut Vec<Value>| {
            if group.len() > 1 {
                out.push(json!({"member_ref":group[0]["member_ref"],"offset_basis":group[0]["range"]["offset_basis"],"storage_refs":group.iter().map(|r|r["storage_ref"].clone()).collect::<Vec<_>>(),"classification":"overlapping_serialized_ranges_alias_semantics_not_established"}));
            }
        };
        for (row, a, b) in rows {
            if a >= end {
                flush(&group, &mut out);
                group.clear();
                end = b.clone();
            }
            group.push(row);
            if b > end {
                end = b;
            }
        }
        flush(&group, &mut out);
    }
    Ok(json!(out))
}
fn members(v: &Value) -> R<()> {
    let l = &v["ledger"];
    need(
        l["schema"] == "deepbom.artifact_members.v1",
        "IR artifact member ledger is invalid.",
    )?;
    let members = index(&l["members"], "IR artifact member identity is invalid.")?;
    let objects = index(
        &v["storage"]["objects"],
        "IR storage object identities are invalid.",
    )?;
    for m in members.values() {
        need(
            (m.get("sha256") == Some(&Value::Null) || sha(&m["sha256"]))
                && (m["verification"] != "identity_bound" || sha(&m["sha256"])),
            "IR artifact member identity is invalid.",
        )?;
        count(&m["byte_length"])?;
    }
    let primary = members
        .get("artifact:primary")
        .ok_or("Artifact IR primary member binding is invalid.")?;
    need(
        primary["sha256"] == v["identity"]["sha256"]
            && primary["byte_length"] == v["identity"]["byte_length"]
            && primary["path"] == v["identity"]["filename"],
        "Artifact IR primary member binding is invalid.",
    )?;
    let rows = list(
        &l["storage_bindings"],
        "IR artifact member ledger is invalid.",
    )?;
    let mut seen = BTreeSet::new();
    for row in rows {
        let id = row["storage_ref"]
            .as_str()
            .ok_or("IR storage member reference or range is invalid.")?;
        let object = objects
            .get(id)
            .ok_or("IR storage member reference or range is invalid.")?;
        let m = row["member_ref"]
            .as_str()
            .and_then(|r| members.get(r))
            .copied()
            .unwrap_or(&Value::Null);
        need(
            seen.insert(id)
                && (row.get("member_ref") == Some(&Value::Null) || !m.is_null())
                && row["range"] == object["byte_range"],
            "IR storage member reference or range is invalid.",
        )?;
        if row["range"]["status"] == "exact" {
            let (a, b) = range(&row["range"])?;
            need(
                Some(b - a) == count(&object["serialized_byte_length"])?,
                "IR storage range length contradicts serialized bytes.",
            )?;
        }
        need(
            row["bounds_assessment"] == bounds(row, m)?,
            "IR storage member bounds assessment is inconsistent.",
        )?;
    }
    need(
        seen.len() == objects.len(),
        "IR artifact member ledger omitted storage objects.",
    )?;
    need(
        l["overlap_groups"] == overlaps(&l["storage_bindings"])?,
        "IR storage overlap accounting is inconsistent.",
    )?;
    need(
        l["completeness"]
            == if rows.iter().any(|r| r["member_ref"].is_null()) {
                "partial_member_binding"
            } else {
                "available_members_bound"
            },
        "IR artifact member completeness is inconsistent.",
    )
}
fn input_manifest(v: &Value) -> R<String> {
    let mut rows = vec![];
    for row in list(v, "Runtime captured inputs must be an array.")? {
        let mut m = serde_json::Map::new();
        for k in [
            "value_ref",
            "native_locator",
            "dtype",
            "shape",
            "values_sha256",
        ] {
            m.insert(
                k.into(),
                row.get(k)
                    .ok_or("Runtime captured input identity is incomplete.")?
                    .clone(),
            );
        }
        rows.push(Value::Object(m));
    }
    crate::portable_identity::digest(&json!(rows))
}
fn runtime(v: &Value) -> R<()> {
    let p = &v["value"];
    let run = &v["run"];
    need(
        same_keys(
            p,
            &[
                "schema",
                "environment",
                "environment_sha256",
                "input_manifest_sha256",
                "code_sha256",
                "trace_sha256",
                "signature",
                "trust",
            ],
        ) && p["schema"] == "deepbom.runtime_provenance.v1"
            && p["trust"] == "declared_not_attested",
        "Runtime provenance contract is invalid.",
    )?;
    need(
        p["environment"].is_object()
            && p["environment_sha256"] == crate::portable_identity::digest(&p["environment"])?,
        "Runtime environment digest is inconsistent.",
    )?;
    need(
        p["input_manifest_sha256"] == input_manifest(&v["inputs"])?,
        "Runtime provenance input digest does not bind the captured inputs.",
    )?;
    for k in ["code_sha256", "trace_sha256"] {
        need(
            p[k].is_null() || sha(&p[k]),
            &format!("Runtime provenance {k} is invalid."),
        )?;
    }
    if !run["runtime_evidence"].is_null() {
        need(
            p["trace_sha256"] == run["runtime_evidence"]["sha256"],
            "Runtime provenance trace conflicts with imported evidence.",
        )?;
    }
    let s = &p["signature"];
    if !s.is_null() {
        let base = s["value_base64"].as_str().unwrap_or("").as_bytes();
        need(
            same_keys(
                s,
                &[
                    "algorithm",
                    "key_id",
                    "payload_sha256",
                    "value_base64",
                    "verification",
                ],
            ) && s["algorithm"] == "Ed25519"
                && nonempty(&s["key_id"])
                && s["verification"] == "not_verified"
                && base.len() == 88
                && base[86..] == *b"=="
                && base[..86]
                    .iter()
                    .all(|c| c.is_ascii_alphanumeric() || *c == b'+' || *c == b'/'),
            "Runtime provenance detached signature is invalid.",
        )?;
        let mut statement = p.clone();
        statement.as_object_mut().unwrap().remove("signature");
        for value in [
            &run["execution"]["artifact_sha256"],
            &run["execution"]["configuration_sha256"],
            &run["collector"]["sha256"],
        ] {
            need(
                sha(value),
                "Runtime provenance signature requires complete run identity.",
            )?;
        }
        let signed = json!({"artifact_sha256":run["execution"]["artifact_sha256"],"configuration_sha256":run["execution"]["configuration_sha256"],"collector_sha256":run["collector"]["sha256"],"provenance":statement});
        need(
            s["payload_sha256"] == crate::portable_identity::digest(&signed)?,
            "Runtime provenance signature payload does not bind this run.",
        )?;
    }
    Ok(())
}
fn exact_input(v: &Value) -> Option<BigUint> {
    if !v.is_string() && !v.is_number() {
        return None;
    }
    decimal(v)
}
fn count_value(v: &Value) -> Value {
    let raw = if v.is_object() { &v["decimal"] } else { v };
    if raw
        .as_str()
        .is_some_and(|s| s.is_empty() || (s != "0" && s.starts_with('0')))
    {
        return Value::Null;
    }
    let Some(n) = exact_input(raw) else {
        return Value::Null;
    };
    let record = exact(&n);
    if v.is_object()
        && ["number", "value"]
            .iter()
            .any(|key| v.get(key).is_some_and(|x| x != &record["number"]))
    {
        return Value::Null;
    }
    json!(n.to_string())
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match v["operation"].as_str().unwrap_or("") {
        "exact_integer" => Ok(exact_input(&v["value"])
            .map(|n| exact(&n))
            .unwrap_or(Value::Null)),
        "exact_count_value" => Ok(count_value(&v["value"])),
        "validate_exact_integer" => {
            count(&v["value"])?;
            Ok(Value::Null)
        }
        "safe_bigint_number" => {
            use num_traits::ToPrimitive;
            let n = v["value"]
                .as_str()
                .and_then(|s| s.parse::<num_bigint::BigInt>().ok())
                .and_then(|n| n.to_i64())
                .filter(|n| (-9_007_199_254_740_991..=9_007_199_254_740_991).contains(n));
            Ok(json!(n))
        }
        "value_type" => {
            value_type(&v["value"])?;
            Ok(Value::Null)
        }
        "elements" => elements(&v["value"]),
        "metric" => {
            metric(&v["value"])?;
            Ok(Value::Null)
        }
        "new_metric" => new_metric(v),
        "aggregate_metrics" => aggregate(v),
        "logical_inventory" => {
            logical(v)?;
            Ok(Value::Null)
        }
        "member_bounds" => bounds(&v["row"], &v["member"]),
        "member_overlaps" => overlaps(&v["bindings"]),
        "members" => {
            members(v)?;
            Ok(Value::Null)
        }
        "input_manifest" => Ok(json!(input_manifest(&v["inputs"])?)),
        "runtime_provenance" => {
            runtime(v)?;
            Ok(Value::Null)
        }
        _ => Err("unknown portable IR operation".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_half_open_ranges_do_not_overlap() {
        let row = |id: &str, a: &str, b: &str| json!({"storage_ref":id,"member_ref":"f","range":{"status":"exact","offset_basis":"artifact_absolute","start":a,"end_exclusive":b}});
        assert_eq!(
            overlaps(&json!([
                row("a", "0", "10"),
                row("empty", "5", "5"),
                row("next", "10", "12")
            ]))
            .unwrap(),
            json!([])
        );
        let r = overlaps(&json!([
            row("a", "9007199254740993", "9007199254740999"),
            row("b", "9007199254740998", "9007199254741000")
        ]))
        .unwrap();
        assert_eq!(r[0]["storage_refs"], json!(["a", "b"]));
        assert!(overlaps(&json!([row("bad", "2", "1")])).is_err());
    }
    #[test]
    fn partial_metric_never_becomes_complete_zero() {
        let m = |value: Value, id: &str| {
            new_metric(&json!({"value":value,"unit":"MAC","assessed":if value.is_null(){0}else{1},"eligible":1,"source_refs":[id],"assumptions":[]})).unwrap()
        };
        let r = aggregate(
            &json!({"rows":[m(json!(null),"a"),m(exact(&BigUint::zero()),"b")],"unit":"MAC"}),
        )
        .unwrap();
        assert_eq!(r["total"]["value"], Value::Null);
        assert_eq!(r["assessed_subtotal"]["decimal"], "0");
        assert!(
            aggregate(&json!({"rows":[m(json!(null),"a"),m(json!(null),"a")],"unit":"MAC"}))
                .is_err()
        );
    }
    #[test]
    fn zero_axis_does_not_hide_invalid_dimension() {
        let mut t = json!({"schema":"deepbom.value_type.v1","root":{"kind":"tensor","dtype":"F32","rank_status":"ranked","dimensions":[{"kind":"constant","value":{"decimal":"0","number":0}},{"kind":"symbol","name":"N","scope_ref":"s"}]}});
        assert_eq!(elements(&t).unwrap()["decimal"], "0");
        t["root"]["dimensions"][1] = json!({"kind":"constant","value":{"decimal":"2","number":3}});
        assert!(elements(&t).is_err());
    }
}
