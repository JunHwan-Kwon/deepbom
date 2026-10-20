use crate::portable_validation::*;
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use serde_json::{json, Value};
use std::collections::BTreeSet;
pub(crate) const WEIGHT_BOUNDARY:&str="Statistics describe serialized numeric payloads, not trainability, accuracy or execution. Integer codes are not dequantized unless representation explicitly says dequantized_real. Shared storage may have multiple uses; tensor views are not unique physical allocations. Unmapped payloads do not establish layers. Floating moments are rounded binary64; counts are exact decimal integers. No sampling is performed.";
pub(crate) fn statistics(s: &Value) -> R<()> {
    keys(
        s,
        &[
            "value_count",
            "finite_count",
            "nonfinite",
            "zero_count",
            "negative_zero_count",
            "precision",
            "unsafe_integer_count",
            "integer_minimum",
            "integer_maximum",
            "minimum",
            "maximum",
            "mean",
            "population_stddev",
            "rms",
            "l2_norm",
            "overflowed_metrics",
            "histogram",
        ],
        "invalid statistics fields",
    )?;
    keys(
        &s["nonfinite"],
        &["nan", "positive_infinity", "negative_infinity"],
        "invalid nonfinite counts fields",
    )?;
    let total = decimal(&s["value_count"])?;
    let finite = decimal(&s["finite_count"])?;
    let zero = decimal(&s["zero_count"])?;
    let negative_zero = decimal(&s["negative_zero_count"])?;
    let unsafe_n = decimal(&s["unsafe_integer_count"])?;
    let mut nonfinite = BigUint::zero();
    for k in ["nan", "positive_infinity", "negative_infinity"] {
        nonfinite += decimal(&s["nonfinite"][k])?;
    }
    need(total == &finite + nonfinite, "value counts do not conserve")?;
    need(
        negative_zero <= zero && zero <= finite,
        "zero counts do not conserve",
    )?;
    need(
        unsafe_n <= finite,
        "unsafe integer count exceeds finite count",
    )?;
    need(
        s["precision"]
            == if unsafe_n.is_zero() {
                "binary64_rounded_statistics_exact_counts"
            } else {
                "not_assessed_unsafe_integer_arithmetic"
            },
        "precision status contradicts counts",
    )?;
    let metrics = [
        "minimum",
        "maximum",
        "mean",
        "population_stddev",
        "rms",
        "l2_norm",
    ];
    for k in metrics {
        need(
            s[k].is_null() || s[k].as_f64().is_some_and(f64::is_finite),
            &format!("invalid {k}"),
        )?;
    }
    let overflow = array(&s["overflowed_metrics"], "invalid overflow ledger")?;
    let mut seen = BTreeSet::new();
    for k in overflow {
        need(
            ["mean", "population_stddev", "rms", "l2_norm"].contains(&string(k))
                && s[string(k)].is_null(),
            "invalid overflow ledger",
        )?;
        need(seen.insert(string(k)), "duplicate overflow metric")?;
    }
    if !unsafe_n.is_zero() || finite.is_zero() {
        for k in metrics {
            need(s[k].is_null(), "unassessed statistic is not null")?;
        }
    }
    need(
        zero <= &finite - &unsafe_n,
        "zero count includes undecoded unsafe integers",
    )?;
    fn token(v: &Value) -> Option<BigInt> {
        let s = v.as_str()?;
        if s == "-0" {
            return None;
        }
        let abs = s.strip_prefix('-').unwrap_or(s);
        if abs.is_empty()
            || (abs != "0" && abs.starts_with('0'))
            || !abs.bytes().all(|c| c.is_ascii_digit())
            || s.len() > 32768
        {
            return None;
        }
        s.parse().ok()
    }
    need(
        (s["integer_minimum"].is_null() && s["integer_maximum"].is_null() && unsafe_n.is_zero())
            || matches!((token(&s["integer_minimum"]),token(&s["integer_maximum"])),(Some(a),Some(b)) if a<=b),
        "invalid exact integer extrema",
    )?;
    if unsafe_n.is_zero() && !finite.is_zero() {
        need(
            matches!((s["minimum"].as_f64(),s["maximum"].as_f64()),(Some(a),Some(b)) if a<=b),
            "invalid numeric extrema",
        )?;
        for k in ["mean", "population_stddev", "rms", "l2_norm"] {
            need(
                !s[k].is_null() || seen.contains(k),
                "missing metric without overflow reason",
            )?;
        }
        for k in ["population_stddev", "rms", "l2_norm"] {
            need(
                s[k].is_null() || s[k].as_f64().is_some_and(|n| n >= 0.),
                "negative magnitude statistic",
            )?;
        }
    }
    let histogram = &s["histogram"];
    if !histogram.is_null() {
        keys(
            histogram,
            &["method", "interval", "edges", "counts"],
            "invalid histogram fields",
        )?;
        need(
            unsafe_n.is_zero()
                && histogram["method"] == "fixed_log2_edges_v1"
                && histogram["interval"] == "left_closed_right_open_last_closed",
            "histogram method mismatch",
        )?;
        let edges = crate::portable_numerics::numerical_histogram_edges();
        need(
            histogram["edges"].as_array().is_some_and(|input| {
                input.len() == edges.len()
                    && input
                        .iter()
                        .zip(&edges)
                        .all(|(a, b)| a.as_f64() == Some(*b))
            }),
            "histogram edges mismatch",
        )?;
        let counts = array(&histogram["counts"], "histogram count conservation failed")?;
        let mut total = BigUint::zero();
        for n in counts {
            total += decimal(n)?;
        }
        need(
            counts.len() + 1 == edges.len() && total == finite,
            "histogram count conservation failed",
        )?;
    } else {
        need(!unsafe_n.is_zero(), "missing histogram")?;
    }
    Ok(())
}
pub(crate) fn source(source: &Value, model: &Value) -> R<()> {
    keys(
        source,
        &["model_ir_sha256", "artifact_sha256", "artifact_set_sha256"],
        "invalid source fields",
    )?;
    need(
        *source
            == json!({"model_ir_sha256":model["model_ir_sha256"],"artifact_sha256":model["artifact"]["sha256"],"artifact_set_sha256":model["artifact_set"]["artifact_set_sha256"]}),
        "source does not match Model IR and artifact identity",
    )
}
pub(crate) fn weight_coverage(tensors: &Value, model: &Value) -> R<Value> {
    let tensors = array(tensors, "missing weight inventory")?;
    let mut total = BigUint::zero();
    for r in tensors {
        if !r["statistics"]["value_count"].is_null() {
            total += decimal(&r["statistics"]["value_count"])?;
        }
    }
    Ok(
        json!({"model_storage_count":rows(&model["tensors_and_storage"]["storage_objects"]).len(),"inventory_count":tensors.len(),"assessed_count":tensors.iter().filter(|r|r["status"]=="assessed").count(),"not_assessed_count":tensors.iter().filter(|r|r["status"]=="not_assessed").count(),"unlinked_count":tensors.iter().filter(|r|r["storage_ref"].is_null()).count(),"decoded_value_count":total.to_string(),"scope":"all_model_storage_objects_plus_decoder_exposed_payloads","completeness":if tensors.iter().all(|r|r["status"]=="assessed"){"complete_within_inventory"}else{"partial_within_inventory"}}),
    )
}
pub(crate) fn weight(v: &Value, model: &Value) -> R<()> {
    keys(
        v,
        &[
            "schema",
            "method_version",
            "source",
            "budget",
            "coverage",
            "tensors",
            "interpretation_boundary",
            "weight_ir_sha256",
        ],
        "invalid Weight IR fields",
    )?;
    need(
        v["schema"] == "deepbom.weight_ir.v1"
            && v["method_version"] == "1.0.0"
            && v["interpretation_boundary"] == WEIGHT_BOUNDARY,
        "unsupported Weight IR contract",
    )?;
    source(&v["source"], model)?;
    digest(v, "weight_ir_sha256", "weight_ir_sha256 mismatch")?;
    keys(
        &v["budget"],
        &["maximum_decoded_values"],
        "invalid weight budget fields",
    )?;
    let budget = decimal(&v["budget"]["maximum_decoded_values"])?;
    let stores = index(
        &model["tensors_and_storage"]["storage_objects"],
        "invalid Model IR storage identities",
    )?;
    let mut refs = BTreeSet::new();
    for (i, row) in array(&v["tensors"], "missing weight inventory")?
        .iter()
        .enumerate()
    {
        keys(
            row,
            &[
                "id",
                "storage_ref",
                "native_locator",
                "name",
                "dtype",
                "shape",
                "status",
                "reason",
                "representation",
                "payload_sha256",
                "invalid_encoding_count",
                "binding_refs",
                "statistics",
            ],
            "invalid weight tensor fields",
        )?;
        need(
            row["id"] == format!("weight:{i}")
                && text(&row["native_locator"], usize::MAX)
                && row["name"].is_string()
                && row["dtype"].is_string(),
            "invalid weight identity",
        )?;
        if !row["storage_ref"].is_null() {
            let id = string(&row["storage_ref"]);
            let stored = stores
                .get(id)
                .ok_or("invalid or duplicate storage reference")?;
            need(refs.insert(id), "invalid or duplicate storage reference")?;
            need(
                row["dtype"] == stored["dtype"],
                "weight dtype contradicts Model IR",
            )?;
            if stored["shape_rank_status"] == "ranked" || !rows(&stored["shape"]).is_empty() {
                need(
                    row["shape"] == stored["shape"],
                    "weight shape contradicts Model IR",
                )?;
            }
        }
        let binding_refs: Vec<_> = rows(&model["weight_bindings"]["bindings"])
            .iter()
            .filter(|b| b["storage_ref"] == row["storage_ref"])
            .map(|b| b["id"].clone())
            .collect();
        need(
            row["binding_refs"] == json!(binding_refs),
            "weight bindings do not match Model IR",
        )?;
        need(
            row["payload_sha256"].is_null() || sha(&row["payload_sha256"]),
            "invalid payload hash",
        )?;
        let expected = if row["shape"].is_null() {
            None
        } else {
            Some(shape_count(&row["shape"])?)
        };
        if row["status"] == "assessed" {
            need(
                row["reason"].is_null()
                    && ["stored_scalar", "dequantized_real", "complex_magnitude"]
                        .contains(&string(&row["representation"])),
                "invalid assessed representation",
            )?;
            statistics(&row["statistics"])?;
            need(
                Some(decimal(&row["statistics"]["value_count"])?) == expected,
                "weight shape/count mismatch",
            )?;
            decimal(&row["invalid_encoding_count"])?;
        } else {
            need(
                row["status"] == "not_assessed"
                    && text(&row["reason"], usize::MAX)
                    && row["statistics"].is_null()
                    && row["representation"].is_null()
                    && row["invalid_encoding_count"].is_null(),
                "unassessed weight has unsupported statistics",
            )?;
        }
    }
    need(
        refs.len() == stores.len(),
        "Weight IR omitted Model IR storage objects",
    )?;
    need(
        v["coverage"] == weight_coverage(&v["tensors"], model)?,
        "weight coverage ledger mismatch",
    )?;
    need(
        decimal(&v["coverage"]["decoded_value_count"])? <= budget,
        "decoded values exceed budget",
    )
}
