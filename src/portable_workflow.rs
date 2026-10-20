//! Portable Snapshot/Evaluation/Deployment contracts. Context I/O and framework
//! collection stay in hosts; validation never promotes a declaration to a fact.
use crate::portable_evidence::{subject, text, timestamp};
use serde_json::{json, Value};
use std::collections::BTreeSet;
type Check = Result<(), String>;
fn need(ok: bool, msg: &str) -> Check {
    if ok {
        Ok(())
    } else {
        Err(msg.into())
    }
}
fn fields(v: &Value, keys: &[&str], optional: &[&str], label: &str) -> Check {
    let m = v
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    need(
        keys.iter().all(|k| m.contains_key(*k))
            && m.keys()
                .all(|k| keys.contains(&k.as_str()) || optional.contains(&k.as_str())),
        &format!("{label} fields do not match its contract"),
    )
}
fn txt(v: &Value, label: &str) -> Check {
    need(
        text(v, 4096),
        &format!("{label} must be nonempty bounded text"),
    )
}
fn one(v: &Value, choices: &[&str]) -> bool {
    v.as_str().is_some_and(|s| choices.contains(&s))
}
fn reference(v: &Value) -> Check {
    match subject(v) {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}
fn ids(v: &Value, label: &str) -> Check {
    let error =
        crate::portable_evidence::run(&json!({"operation":"unique_ids","value":v,"label":label}))?;
    if let Some(e) = error.as_str() {
        Err(e.into())
    } else {
        Ok(())
    }
}
fn finite(v: &Value, label: &str) -> Result<f64, String> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("{label} must be a finite number"))
}
fn count(v: &Value, label: &str, positive: bool) -> Check {
    let valid = v.as_str().is_some_and(|s| {
        !s.is_empty()
            && s.len() <= 100
            && s.bytes().all(|b| b.is_ascii_digit())
            && (s == "0" || !s.starts_with('0'))
    });
    need(valid, &format!("{label} must be an exact decimal count"))?;
    need(!positive || v != "0", &format!("{label} must be positive"))
}
fn digest(v: &Value, key: &str) -> Check {
    let mut body = v.clone();
    body.as_object_mut()
        .ok_or("evidence document must be an object")?
        .remove(key);
    need(
        v[key].as_str() == Some(crate::portable_identity::digest(&body)?.as_str()),
        &format!("{key} mismatch"),
    )
}
const SNAPSHOT_BOUNDARY: &str="Fixed state only within the declared scope. Identity is not authenticity, complete data capture, execution equivalence or clinical validation.";
fn snapshot(v: &Value) -> Check {
    fields(
        v,
        &[
            "schema",
            "method_version",
            "kind",
            "identity",
            "scope",
            "contents",
            "completeness",
            "boundary",
            "snapshot_ir_sha256",
        ],
        &[],
        "Snapshot IR",
    )?;
    need(
        v["schema"] == "deepbom.snapshot_ir.v1"
            && v["method_version"] == "1.0.0"
            && one(
                &v["kind"],
                &["model_state", "dataset_release", "configuration"],
            ),
        "unsupported Snapshot IR contract",
    )?;
    fields(
        &v["identity"],
        &["namespace", "name", "version"],
        &[],
        "snapshot identity",
    )?;
    for k in ["namespace", "name", "version"] {
        txt(&v["identity"][k], k)?;
    }
    let scope = &v["scope"];
    fields(
        scope,
        &["fixed_aspects", "excluded_aspects", "consistency"],
        &[],
        "snapshot scope",
    )?;
    for k in ["fixed_aspects", "excluded_aspects"] {
        let a = scope[k].as_array().ok_or("invalid snapshot scope list")?;
        need(a.len() <= 256, "invalid snapshot scope list")?;
        let mut seen = BTreeSet::new();
        for x in a {
            txt(x, k)?;
            need(
                seen.insert(x.as_str().unwrap()),
                "invalid snapshot scope list",
            )?;
        }
    }
    let fixed = scope["fixed_aspects"].as_array().unwrap();
    let excluded = scope["excluded_aspects"].as_array().unwrap();
    need(
        !fixed.is_empty() && !fixed.iter().any(|x| excluded.contains(x)),
        "snapshot fixed/excluded aspects conflict",
    )?;
    need(
        one(
            &scope["consistency"],
            &["adapter_declared_scope", "manifest_declared_scope"],
        ),
        "unsupported capture consistency claim",
    )?;
    need(
        v["completeness"] == "declared_scope_only" && v["boundary"] == SNAPSHOT_BOUNDARY,
        "snapshot completeness/boundary contradiction",
    )?;
    ids(&v["contents"], "snapshot contents")?;
    let contents = v["contents"].as_array().unwrap();
    need(
        !contents.is_empty(),
        "snapshot needs fixed content references",
    )?;
    let mut paths = BTreeSet::new();
    let mut unique = true;
    for row in contents {
        fields(row, &["id", "role", "ref"], &["path"], "snapshot content")?;
        txt(&row["role"], "content role")?;
        reference(&row["ref"])?;
        need(
            row["ref"]["subject_ref"].is_null(),
            "snapshot contents must reference whole content documents or files",
        )?;
        if let Some(p) = row.get("path") {
            txt(p, "relative content path")?;
            let p = p.as_str().unwrap();
            let b = p.as_bytes();
            need(
                !p.contains(['\\', '\0'])
                    && !p.starts_with('/')
                    && !(b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
                    && p.split('/').all(|s| !s.is_empty() && s != "." && s != ".."),
                "unsafe manifest content path",
            )?;
            unique &= paths.insert(p);
        }
    }
    need(unique, "duplicate manifest content path")?;
    if v["kind"] == "model_state" {
        need(
            contents.len() == 1
                && contents[0]["id"] == "state"
                && contents[0]["role"] == "native_model_state"
                && contents[0]["ref"]["kind"] == "native_state",
            "model-state snapshot source mismatch",
        )?;
        need(
            *scope
                == json!({"fixed_aspects":["adapter_definition","named_tensor_identities"],"excluded_aspects":["optimizer_state","rng_state","external_python_state"],"consistency":"adapter_declared_scope"}),
            "model-state scope exceeds the native adapter contract",
        )?;
        need(
            v["identity"]["namespace"] == "deepbom.native"
                && v["identity"]["name"] == "model-state"
                && v["identity"]["version"] == contents[0]["ref"]["sha256"],
            "native snapshot identity mismatch",
        )?;
    } else {
        need(
            scope["consistency"] == "manifest_declared_scope",
            "manifest cannot assert atomic state capture",
        )?;
    }
    digest(v, "snapshot_ir_sha256")
}
fn protocol(v: &Value, sealed: bool) -> Check {
    let mut keys = vec![
        "schema",
        "id",
        "comparison",
        "data_policy",
        "metrics",
        "required_populations",
        "boundary",
    ];
    if sealed {
        keys.push("protocol_sha256");
    }
    fields(
        v,
        &keys,
        &[],
        if sealed { "protocol" } else { "protocol input" },
    )?;
    need(
        v["schema"] == "deepbom.evaluation_protocol.v1",
        "unsupported evaluation protocol",
    )?;
    txt(&v["id"], "protocol ID")?;
    need(
        one(&v["comparison"], &["paired", "unpaired", "descriptive"]),
        "unknown comparison mode",
    )?;
    need(
        v["data_policy"] == "same_release",
        "only explicit same-release comparisons are supported",
    )?;
    ids(&v["metrics"], "protocol metrics")?;
    let metrics = v["metrics"].as_array().unwrap();
    need(
        !metrics.is_empty() && metrics.len() <= 64,
        "protocol needs at least one metric",
    )?;
    for m in metrics {
        fields(
            m,
            &[
                "id",
                "unit",
                "analysis_unit",
                "higher_is_better",
                "max_regression",
                "range",
                "aggregation",
                "missing_policy",
                "repeated_observations",
            ],
            &[],
            "metric definition",
        )?;
        for k in ["aggregation", "missing_policy", "repeated_observations"] {
            txt(&m[k], k)?;
        }
        txt(&m["unit"], "metric unit")?;
        txt(&m["analysis_unit"], "analysis unit")?;
        need(
            m["higher_is_better"].is_boolean(),
            "metric direction missing",
        )?;
        if !m["max_regression"].is_null() {
            need(
                finite(&m["max_regression"], "maximum regression")? >= 0.,
                "negative regression threshold",
            )?;
        }
        if !m["range"].is_null() {
            let a = m["range"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("metric range invalid")?;
            need(
                finite(&a[0], "metric range")? <= finite(&a[1], "metric range")?,
                "metric range reversed",
            )?;
        }
    }
    let populations = v["required_populations"]
        .as_array()
        .ok_or("protocol populations invalid")?;
    need(
        !populations.is_empty() && populations.len() <= 64,
        "protocol populations invalid",
    )?;
    let mut seen = BTreeSet::new();
    for p in populations {
        txt(p, "population")?;
        need(
            seen.insert(p.as_str().unwrap()),
            "protocol populations invalid",
        )?;
    }
    txt(&v["boundary"], "protocol boundary")?;
    if sealed {
        digest(v, "protocol_sha256")?;
    }
    Ok(())
}
fn evaluation(v: &Value, sealed: bool) -> Check {
    let mut keys = vec![
        "schema",
        "subject",
        "protocol",
        "dataset",
        "cohort",
        "run",
        "code",
        "environment",
        "configuration",
        "population",
        "origin",
        "comparison",
        "pair_set",
        "metrics",
        "boundary",
    ];
    if sealed {
        keys.push("evaluation_sha256");
    }
    fields(
        v,
        &keys,
        &[],
        if sealed {
            "evaluation record"
        } else {
            "evaluation input"
        },
    )?;
    need(
        v["schema"] == "deepbom.evaluation_record.v1",
        "unsupported evaluation record",
    )?;
    for k in [
        "subject",
        "protocol",
        "dataset",
        "cohort",
        "run",
        "code",
        "environment",
        "configuration",
    ] {
        reference(&v[k])?;
    }
    need(
        v["protocol"]["schema"] == "deepbom.evaluation_protocol.v1"
            && v["protocol"]["kind"] == "evidence_document",
        "evaluation protocol reference mismatch",
    )?;
    txt(&v["population"], "population")?;
    need(
        v["origin"] == "externally_reported",
        "import cannot assert local recomputation",
    )?;
    need(
        one(&v["comparison"], &["paired", "unpaired", "descriptive"]),
        "unknown evaluation comparison mode",
    )?;
    if !v["pair_set"].is_null() {
        reference(&v["pair_set"])?;
    }
    if v["comparison"] != "paired" {
        need(
            v["pair_set"].is_null(),
            "non-paired record cannot claim a pair set",
        )?;
    }
    ids(&v["metrics"], "evaluation metrics")?;
    let metrics = v["metrics"].as_array().unwrap();
    need(
        !metrics.is_empty() && metrics.len() <= 64,
        "evaluation has no metrics",
    )?;
    for m in metrics {
        fields(
            m,
            &[
                "id",
                "value",
                "unit",
                "analysis_unit",
                "denominator",
                "missing_count",
                "resolution",
                "ci",
            ],
            &[],
            "measurement",
        )?;
        finite(&m["value"], "estimate")?;
        txt(&m["unit"], "measurement unit")?;
        txt(&m["analysis_unit"], "measurement analysis unit")?;
        count(&m["denominator"], "denominator", true)?;
        count(&m["missing_count"], "missing count", false)?;
        if !m["resolution"].is_null() {
            need(
                finite(&m["resolution"], "measurement resolution")? > 0.,
                "measurement resolution must be positive",
            )?;
        }
        let ci = &m["ci"];
        if !ci.is_null() {
            fields(
                ci,
                &[
                    "low",
                    "high",
                    "level",
                    "method",
                    "resampling_unit",
                    "repetitions",
                ],
                &[],
                "confidence interval",
            )?;
            let lo = finite(&ci["low"], "CI low")?;
            let hi = finite(&ci["high"], "CI high")?;
            let level = finite(&ci["level"], "CI level")?;
            need(
                lo <= hi && level > 0. && level < 1.,
                "invalid CI bounds/level",
            )?;
            txt(&ci["method"], "CI method")?;
            if !ci["resampling_unit"].is_null() {
                txt(&ci["resampling_unit"], "resampling unit")?;
            }
            if !ci["repetitions"].is_null() {
                count(&ci["repetitions"], "resampling repetitions", true)?;
            }
        }
    }
    txt(&v["boundary"], "evaluation boundary")?;
    if sealed {
        digest(v, "evaluation_sha256")?;
    }
    Ok(())
}
fn measurement(record: &Value, p: &Value) -> Check {
    evaluation(record, true)?;
    protocol(p, true)?;
    need(
        record["protocol"]["sha256"] == p["protocol_sha256"],
        "evaluation refers to another protocol",
    )?;
    need(
        record["comparison"] == p["comparison"],
        "comparison mode disagrees with protocol",
    )?;
    need(
        p["required_populations"]
            .as_array()
            .unwrap()
            .contains(&record["population"]),
        "evaluation population outside protocol",
    )?;
    for m in record["metrics"].as_array().unwrap() {
        let rule = p["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == m["id"])
            .ok_or("metric definition/unit/analysis unit mismatch")?;
        need(
            rule["unit"] == m["unit"] && rule["analysis_unit"] == m["analysis_unit"],
            "metric definition/unit/analysis unit mismatch",
        )?;
        if !rule["range"].is_null() {
            let value = m["value"].as_f64().unwrap();
            need(
                value >= rule["range"][0].as_f64().unwrap()
                    && value <= rule["range"][1].as_f64().unwrap(),
                "estimate outside protocol range",
            )?;
        }
    }
    Ok(())
}
fn review_input(v: &Value) -> Check {
    fields(
        v,
        &[
            "schema",
            "before",
            "after",
            "change_kind",
            "protocol",
            "evaluations",
            "evidence",
        ],
        &[],
        "change review input",
    )?;
    need(
        v["schema"] == "deepbom.change_review_input.v1",
        "unsupported review input",
    )?;
    for k in ["before", "after", "protocol"] {
        reference(&v[k])?;
    }
    need(
        v["before"] != v["after"],
        "before and after must be distinct subjects",
    )?;
    need(
        one(
            &v["change_kind"],
            &[
                "preserving_transformation",
                "approximate_transformation",
                "model_change",
                "configuration_change",
            ],
        ),
        "unknown change kind",
    )?;
    for k in ["evaluations", "evidence"] {
        let a = v[k]
            .as_array()
            .filter(|a| a.len() <= 512)
            .ok_or("review reference limit exceeded")?;
        let mut seen = BTreeSet::new();
        let mut unique = true;
        for r in a {
            reference(r)?;
            unique &= seen.insert(crate::portable_identity::canonical(r)?);
        }
        need(unique, "duplicate review evidence reference")?;
    }
    Ok(())
}
fn configuration(v: &Value) -> Check {
    reference(v)?;
    need(
        v["kind"] == "snapshot",
        "configuration must reference a Snapshot IR",
    )
}
fn deployment(v: &Value, matched: bool) -> Result<Value, String> {
    fields(
        v,
        &[
            "schema",
            "evaluated_configuration",
            "deployments",
            "execution",
        ],
        &[],
        "deployment context input",
    )?;
    need(
        v["schema"] == "deepbom.deployment_context_input.v1",
        "unsupported deployment context",
    )?;
    configuration(&v["evaluated_configuration"])?;
    ids(&v["deployments"], "deployment records")?;
    let mut refs = vec![v["evaluated_configuration"].clone()];
    let mut intervals = Vec::new();
    for d in v["deployments"].as_array().unwrap() {
        fields(
            d,
            &["id", "configuration", "site", "start", "end", "source"],
            &[],
            "deployment record",
        )?;
        configuration(&d["configuration"])?;
        reference(&d["source"])?;
        refs.extend([d["configuration"].clone(), d["source"].clone()]);
        txt(&d["site"], "site")?;
        let start = timestamp(&d["start"])?;
        let end = if d["end"].is_null() {
            None
        } else {
            Some(timestamp(&d["end"])?)
        };
        need(
            end.is_none_or(|end| end > start),
            "deployment interval must be positive",
        )?;
        intervals.push((d, start, end));
    }
    let e = &v["execution"];
    let mut candidates = Vec::new();
    let mut exact = false;
    if !e.is_null() {
        fields(
            e,
            &[
                "event_id",
                "site",
                "at",
                "clock_uncertainty_ms",
                "configuration",
                "output_state",
                "source",
            ],
            &[],
            "execution record",
        )?;
        configuration(&e["configuration"])?;
        reference(&e["source"])?;
        refs.extend([e["configuration"].clone(), e["source"].clone()]);
        txt(&e["event_id"], "execution event")?;
        txt(&e["site"], "execution site")?;
        let at = timestamp(&e["at"])?;
        let uncertainty = e["clock_uncertainty_ms"]
            .as_i64()
            .filter(|x| *x >= 0 && *x <= 9_007_199_254_740_991)
            .ok_or("invalid clock uncertainty")?;
        let lo = at - uncertainty;
        let hi = at + uncertainty;
        need(
            lo.abs() <= 9_007_199_254_740_991 && hi.abs() <= 9_007_199_254_740_991,
            "clock uncertainty exceeds exact time range",
        )?;
        need(
            one(&e["output_state"], &["generated", "displayed", "used"]),
            "unknown output observation",
        )?;
        for (d, start, end) in intervals {
            if d["site"] == e["site"] && start <= hi && end.is_none_or(|end| end > lo) {
                candidates.push(d["id"].clone());
                exact = d["configuration"] == e["configuration"]
                    && start <= lo
                    && end.is_none_or(|end| end > hi);
            }
        }
        exact &= candidates.len() == 1;
    }
    Ok(
        json!({"references":refs,"candidate_deployments":candidates,"status":if e.is_null(){"execution_not_provided"}else if !matched{"unresolved_evidence"}else if !exact{"ambiguous_or_conflicting_deployment"}else{"consistent_declared_execution"},"same_as_evaluated":if e.is_null(){Value::Null}else{json!(v["evaluated_configuration"]==e["configuration"])}}),
    )
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    let value = &v["value"];
    let result = match v["operation"].as_str().unwrap_or("") {
        "snapshot" => snapshot(value),
        "protocol" => protocol(value, v["sealed"].as_bool().unwrap_or(false)),
        "evaluation" => evaluation(value, v["sealed"].as_bool().unwrap_or(false)),
        "measurement" => measurement(value, &v["protocol"]),
        "review_input" => review_input(value),
        "count" => count(
            value,
            v["label"].as_str().unwrap_or("count"),
            v["positive"].as_bool().unwrap_or(false),
        ),
        "deployment" => {
            return deployment(value, v["matched"].as_bool().unwrap_or(false))
                .map_err(|e| format!("Evidence: {e}"))
        }
        _ => return Err("unknown portable evidence workflow operation".into()),
    };
    Ok(json!(result.err()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_decimal_counts() {
        assert!(count(&json!("9007199254740993"), "n", true).is_ok());
        for v in [json!("01"), json!("-1"), json!(1), json!(null)] {
            assert!(count(&v, "n", false).is_err());
        }
        assert!(count(&json!("0"), "n", true).is_err());
    }
}
