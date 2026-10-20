use crate::portable_numerical_validation::{source, statistics};
use crate::portable_validation::*;
use num_bigint::BigInt;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
pub(crate) const BOUNDARY:&str="Imported execution evidence is consistency-checked, not remotely attested. Statistics describe only the recorded inputs, run and captured values. Synthetic probes do not establish task accuracy, real workload distributions or causal relationships between layers. Configured providers do not establish per-node placement. Instrumented runs do not establish production latency. Raw inputs and activations are not embedded in this IR.";
fn nonempty(v: &Value, label: &str) -> R<()> {
    need(
        v.as_str()
            .is_some_and(|s| !s.is_empty() && s.encode_utf16().count() <= 4096),
        &format!("invalid {label}"),
    )
}
pub(crate) fn run_contract(run: &Value, source: &Value, model: &Value) -> R<()> {
    let mut fields = vec![
        "id",
        "started_at",
        "entry_region_ref",
        "runtime",
        "collector",
        "execution",
        "probe",
        "runtime_evidence",
    ];
    if run.get("provenance").is_some() {
        fields.push("provenance");
    }
    keys(run, &fields, "invalid execution run fields")?;
    let entries: Vec<_> = rows(&model["program"]["programs"])
        .iter()
        .flat_map(|p| rows(&p["entry_region_refs"]))
        .collect();
    need(
        if entries.is_empty() {
            run["entry_region_ref"].is_null()
        } else {
            entries.contains(&&run["entry_region_ref"])
        },
        "run entry region does not match Model IR",
    )?;
    nonempty(&run["id"], "run id")?;
    need(
        string(&run["started_at"]).ends_with('Z')
            && crate::portable_evidence::timestamp(&run["started_at"]).is_ok(),
        "invalid run timestamp",
    )?;
    let r = &run["runtime"];
    keys(
        r,
        &["name", "version", "configured_providers", "device"],
        "invalid runtime fields",
    )?;
    nonempty(&r["name"], "runtime name")?;
    nonempty(&r["version"], "runtime version")?;
    need(
        array(&r["configured_providers"], "invalid configured providers")?
            .iter()
            .all(|p| p.as_str().is_some_and(|p| !p.is_empty())),
        "invalid configured providers",
    )?;
    need(
        r["device"].is_null() || r["device"].is_string(),
        "invalid device",
    )?;
    let c = &run["collector"];
    keys(
        c,
        &["name", "version", "sha256"],
        "invalid collector fields",
    )?;
    nonempty(&c["name"], "collector name")?;
    nonempty(&c["version"], "collector version")?;
    need(sha(&c["sha256"]), "missing collector digest")?;
    let e = &run["execution"];
    keys(
        e,
        &[
            "artifact_sha256",
            "instrumented_artifact_sha256",
            "configuration_sha256",
            "configuration",
        ],
        "invalid execution identity fields",
    )?;
    need(
        e["artifact_sha256"] == source["artifact_sha256"] && sha(&e["configuration_sha256"]),
        "execution identity mismatch",
    )?;
    need(
        e["configuration"].is_object() && e["configuration_sha256"] == hash(&e["configuration"])?,
        "execution configuration digest mismatch",
    )?;
    need(
        e["instrumented_artifact_sha256"].is_null() || sha(&e["instrumented_artifact_sha256"]),
        "invalid instrumented identity",
    )?;
    keys(
        &run["probe"],
        &["kind", "description"],
        "invalid probe fields",
    )?;
    need(
        [
            "representative_data",
            "synthetic_ones",
            "synthetic_zeros",
            "synthetic_identity",
            "synthetic_basis",
            "synthetic_random",
            "custom",
        ]
        .contains(&string(&run["probe"]["kind"])),
        "unknown input probe kind",
    )?;
    nonempty(&run["probe"]["description"], "probe description")?;
    let link = &run["runtime_evidence"];
    if !link.is_null() {
        keys(
            link,
            &["schema", "sha256"],
            "invalid runtime evidence link fields",
        )?;
        nonempty(&link["schema"], "runtime evidence schema")?;
        need(sha(&link["sha256"]), "invalid runtime evidence digest")?;
    }
    Ok(())
}
pub(crate) fn tensor_binding(row: &Value, model: &Value, input: bool, entry: &Value) -> R<()> {
    need(
        row["native_locator"]
            .as_str()
            .is_some_and(|s| s.encode_utf16().count() <= 4096),
        "invalid native locator",
    )?;
    need(
        row["dtype"].as_str().is_some_and(|s| !s.is_empty()),
        "missing tensor dtype",
    )?;
    shape_count(&row["shape"])?;
    if row["value_ref"].is_null() {
        need(
            !input || entry.is_null(),
            "inputs must bind to the selected Model IR entry",
        )?;
        return nonempty(&row["native_locator"], "unmapped capture locator");
    }
    let value = rows(&model["program"]["values"])
        .iter()
        .find(|v| v["id"] == row["value_ref"])
        .ok_or("capture refers to unknown Model IR value")?;
    need(
        value["dtype"] == row["dtype"] || value["dtype"] == "UNKNOWN",
        "capture dtype contradicts Model IR",
    )?;
    let t = &value["type_contract"]["root"];
    need(
        value["type_contract"].is_null()
            || ["tensor", "sparse_tensor"].contains(&string(&t["kind"])),
        "capture requires a tensor value type",
    )?;
    let shape = rows(&row["shape"]);
    let compatible = if t["rank_status"] == "unknown_rank" {
        true
    } else if t["rank_status"] == "ranked" {
        let dims = rows(&t["dimensions"]);
        shape.len() == dims.len()
            && dims.iter().zip(shape).all(|(d, n)| {
                d["kind"] != "constant"
                    || crate::portable_contracts::decimal(&d["value"])
                        == crate::portable_contracts::decimal(n)
            })
    } else {
        let expected = rows(&value["shape"]);
        shape.len() == expected.len()
            && expected.iter().zip(shape).all(|(d, n)| match d {
                Value::String(s)
                    if !s.is_empty()
                        && (s == "0" || !s.starts_with('0'))
                        && s.bytes().all(|c| c.is_ascii_digit()) =>
                {
                    crate::portable_contracts::decimal(d) == crate::portable_contracts::decimal(n)
                }
                Value::Number(x) if x.as_u64().is_some_and(|n| n <= 9_007_199_254_740_991) => {
                    d == n
                }
                _ => true,
            })
    };
    need(compatible, "capture shape contradicts Model IR")?;
    need(
        !input || (member(&value["roles"], "graph_input") && value["region_ref"] == *entry),
        "input is not a serialized entry graph input",
    )
}
pub(crate) fn values(values: &Value, dtype: &str) -> R<()> {
    enum Number {
        Integer(BigInt),
        Float(f64),
    }
    for raw in array(values, "tensor values must be a flat array")? {
        let value = if let Some(n) = raw.as_f64() {
            Number::Float(n)
        } else {
            match string(raw) {
                "NaN" => Number::Float(f64::NAN),
                "+Infinity" => Number::Float(f64::INFINITY),
                "-Infinity" => Number::Float(f64::NEG_INFINITY),
                "-0" => Number::Float(-0.),
                s => {
                    let t = s.strip_prefix('-').unwrap_or(s);
                    need(
                        !t.is_empty()
                            && (t == "0" || !t.starts_with('0'))
                            && t.bytes().all(|b| b.is_ascii_digit())
                            && s.len() <= 32768,
                        "value is not a number or canonical numeric token",
                    )?;
                    Number::Integer(
                        s.parse()
                            .map_err(|_| "value is not a number or canonical numeric token")?,
                    )
                }
            }
        };
        let integer = dtype
            .strip_prefix("UINT")
            .map(|b| (true, b))
            .or_else(|| dtype.strip_prefix("INT").map(|b| (false, b)))
            .and_then(|(u, b)| {
                b.parse::<u32>()
                    .ok()
                    .filter(|b| [8, 16, 32, 64].contains(b))
                    .map(|b| (u, b))
            });
        if let Some((unsigned, bits)) = integer {
            let n = match value {
                Number::Integer(n) => n,
                Number::Float(f) => {
                    need(
                        f.to_bits() != (-0f64).to_bits(),
                        "integer activations cannot encode negative zero",
                    )?;
                    need(f.is_finite()&&f.fract()==0.&&f.abs()<=9_007_199_254_740_991.,"integer activation requires exact integer values; use decimal strings for int64")?;
                    BigInt::from(f as i64)
                }
            };
            let min = if unsigned {
                BigInt::from(0)
            } else {
                -(BigInt::from(1) << (bits - 1))
            };
            let max = (BigInt::from(1) << (bits - if unsigned { 0 } else { 1 })) - 1;
            need(
                n >= min && n <= max,
                "integer activation is outside dtype range",
            )?;
        } else if dtype == "BOOL" {
            need(
                match value {
                    Number::Integer(n) => n == BigInt::from(0) || n == BigInt::from(1),
                    Number::Float(f) => f.to_bits() != (-0f64).to_bits() && (f == 0. || f == 1.),
                },
                "BOOL activation must be 0 or 1",
            )?;
        } else {
            need(
                ["FLOAT16", "BFLOAT16", "FLOAT32", "FLOAT64"].contains(&dtype)
                    && matches!(value, Number::Float(_)),
                "unsupported activation numeric dtype",
            )?;
            let Number::Float(f) = value else {
                unreachable!()
            };
            if !f.is_finite() {
                continue;
            }
            if dtype == "FLOAT16" {
                let magnitude = f.abs();
                let step = if magnitude < 2f64.powi(-14) {
                    2f64.powi(-24)
                } else {
                    2f64.powi(magnitude.log2().floor() as i32 - 10)
                };
                need(
                    magnitude <= 65504. && (magnitude == 0. || (magnitude / step).fract() == 0.),
                    "FLOAT16 value is not exactly representable",
                )?;
            }
            if dtype == "BFLOAT16" {
                need(
                    (f as f32 as f64).to_bits() == f.to_bits() && (f as f32).to_bits() & 65535 == 0,
                    "BFLOAT16 value is not exactly representable",
                )?;
            }
            if dtype == "FLOAT32" {
                need(
                    (f as f32 as f64).to_bits() == f.to_bits(),
                    "FLOAT32 value is not exactly representable",
                )?;
            }
        }
    }
    Ok(())
}
fn eligible(model: &Value, entry: &Value) -> Vec<Value> {
    rows(&model["program"]["values"])
        .iter()
        .filter(|v| {
            rows(&v["storage_refs"]).is_empty()
                && !(member(&v["roles"], "graph_input") && v["region_ref"] == *entry)
        })
        .map(|v| v["id"].clone())
        .collect()
}
pub(crate) fn coverage(
    requested: &Value,
    tensors: &Value,
    missing: &Value,
    model: &Value,
    entry: &Value,
) -> Value {
    let requested = rows(requested);
    let tensors = rows(tensors);
    let eligible = eligible(model, entry);
    let captured: BTreeSet<_> = tensors
        .iter()
        .filter_map(|r| r["value_ref"].as_str())
        .collect();
    json!({"requested_count":requested.len(),"captured_count":captured.len(),"missing_count":rows(missing).len(),"unmapped_capture_count":tensors.iter().filter(|r|r["value_ref"].is_null()).count(),"eligible_graph_value_count":eligible.len(),"not_requested_value_refs":eligible.iter().filter(|id|!requested.contains(id)).collect::<Vec<_>>(),"scope":if entry.is_null(){"unmapped_runtime_values"}else if requested.len()==eligible.len()&&eligible.iter().all(|id|requested.contains(id)){"all_eligible_serialized_values"}else{"explicit_selected_values"},"completeness":if rows(missing).is_empty(){"complete_requested_capture"}else{"partial_requested_capture"}})
}
pub(crate) fn validate(v: &Value, model: &Value) -> R<()> {
    keys(
        v,
        &[
            "schema",
            "method_version",
            "source",
            "capture_sha256",
            "run",
            "inputs",
            "requested_value_refs",
            "tensors",
            "missing",
            "coverage",
            "interpretation_boundary",
            "activation_ir_sha256",
        ],
        "invalid Activation IR fields",
    )?;
    need(
        v["schema"] == "deepbom.activation_ir.v1"
            && v["method_version"] == "1.0.0"
            && v["interpretation_boundary"] == BOUNDARY,
        "unsupported Activation IR contract",
    )?;
    source(&v["source"], model)?;
    digest(v, "activation_ir_sha256", "activation_ir_sha256 mismatch")?;
    run_contract(&v["run"], &v["source"], model)?;
    need(sha(&v["capture_sha256"]), "invalid capture digest")?;
    let entry = &v["run"]["entry_region_ref"];
    let eligible = eligible(model, entry);
    let requested = array(
        &v["requested_value_refs"],
        "requested capture values are unknown, static, input or duplicated",
    )?;
    let mut request_ids = BTreeSet::new();
    for r in requested {
        need(
            eligible.contains(r) && request_ids.insert(string(r)),
            "requested capture values are unknown, static, input or duplicated",
        )?;
    }
    let mut accounted = BTreeSet::new();
    let mut input_refs = BTreeSet::new();
    let mut symbols = BTreeMap::new();
    let values_index = index(
        &model["program"]["values"],
        "invalid Model IR value identities",
    )?;
    for (kind, tensors) in [("input", &v["inputs"]), ("activation", &v["tensors"])] {
        for (i, row) in array(tensors, "invalid activation tensors")?
            .iter()
            .enumerate()
        {
            keys(
                row,
                &[
                    "id",
                    "value_ref",
                    "native_locator",
                    "dtype",
                    "shape",
                    "values_sha256",
                    "statistics",
                ],
                "invalid activation tensor fields",
            )?;
            need(
                row["id"] == format!("{kind}:{i}") && sha(&row["values_sha256"]),
                "invalid activation identity",
            )?;
            tensor_binding(row, model, kind == "input", entry)?;
            statistics(&row["statistics"])?;
            need(
                shape_count(&row["shape"])? == decimal(&row["statistics"]["value_count"])?,
                "activation count mismatch",
            )?;
            if kind == "input" {
                let key = if row["value_ref"].is_null() {
                    format!("unmapped:{}", string(&row["native_locator"]))
                } else {
                    string(&row["value_ref"]).into()
                };
                need(input_refs.insert(key), "duplicate input")?;
            } else if !row["value_ref"].is_null() {
                need(
                    requested.contains(&row["value_ref"])
                        && accounted.insert(string(&row["value_ref"])),
                    "unrequested or duplicate capture",
                )?;
            }
            if let Some(value) = values_index.get(string(&row["value_ref"])) {
                for (axis, dim) in rows(&value["type_contract"]["root"]["dimensions"])
                    .iter()
                    .enumerate()
                {
                    if dim["kind"] == "symbol" {
                        let key = (string(&dim["scope_ref"]), string(&dim["name"]));
                        let size = &row["shape"][axis];
                        need(
                            symbols.get(&key).is_none_or(|old| *old == size),
                            "runtime shapes contradict a shared symbolic dimension",
                        )?;
                        symbols.insert(key, size);
                    }
                }
            }
        }
    }
    if !v["run"]["provenance"].is_null() {
        ir(
            "runtime_provenance",
            json!({"value":v["run"]["provenance"],"inputs":v["inputs"],"run":v["run"]}),
        )?;
    }
    for value in values_index.values().filter(|value| {
        member(&value["roles"], "graph_input")
            && value["region_ref"] == *entry
            && rows(&value["storage_refs"]).is_empty()
    }) {
        need(
            input_refs.contains(string(&value["id"])),
            "run is missing graph input evidence",
        )?;
    }
    for row in array(&v["missing"], "invalid missing capture ledger")? {
        keys(
            row,
            &["value_ref", "reason"],
            "invalid missing capture fields",
        )?;
        nonempty(&row["reason"], "missing capture reason")?;
        need(
            requested.contains(&row["value_ref"]) && accounted.insert(string(&row["value_ref"])),
            "duplicate or unrequested missing value",
        )?;
    }
    need(
        accounted.len() == requested.len(),
        "requested value silently omitted",
    )?;
    need(
        v["coverage"]
            == coverage(
                &v["requested_value_refs"],
                &v["tensors"],
                &v["missing"],
                model,
                entry,
            ),
        "activation coverage mismatch",
    )
}
pub(crate) fn capture(v: &Value, model: &Value, budget: &Value) -> R<()> {
    keys(
        v,
        &[
            "schema",
            "source",
            "run",
            "inputs",
            "requested_value_refs",
            "captures",
            "missing",
        ],
        "invalid activation capture fields",
    )?;
    need(
        v["schema"] == "deepbom.activation_capture.v1",
        "unsupported activation capture schema",
    )?;
    source(&v["source"], model)?;
    run_contract(&v["run"], &v["source"], model)?;
    for key in ["inputs", "captures", "missing", "requested_value_refs"] {
        array(&v[key], "invalid capture arrays")?;
    }
    let budget = budget
        .as_u64()
        .filter(|n| *n <= 9_007_199_254_740_991)
        .ok_or("activation value budget exceeded")?;
    let mut count = 0u64;
    for (key, input) in [("inputs", true), ("captures", false)] {
        for row in rows(&v[key]) {
            keys(
                row,
                &["value_ref", "native_locator", "dtype", "shape", "values"],
                "invalid captured tensor fields",
            )?;
            tensor_binding(row, model, input, &v["run"]["entry_region_ref"])?;
            values(&row["values"], string(&row["dtype"]))?;
            let n = rows(&row["values"]).len();
            need(
                shape_count(&row["shape"])? == num_bigint::BigUint::from(n),
                "captured tensor shape/value count mismatch",
            )?;
            count = count
                .checked_add(n as u64)
                .ok_or("activation value budget exceeded")?;
            need(count <= budget, "activation value budget exceeded")?;
        }
    }
    Ok(())
}
