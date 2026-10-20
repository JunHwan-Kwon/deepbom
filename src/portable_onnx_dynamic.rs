//! Shape rules with explicit conditional/ranged dimension evidence.
use crate::portable_onnx_node::{
    ai_exact, attr_ints, exact_dims, inherited, kd, kernel, set, shape, static_numbers,
};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeSet;
pub(crate) fn axis(n: i64, r: usize) -> Option<usize> {
    let x = if n < 0 { n + r as i64 } else { n };
    if x >= 0 && x < r as i64 {
        Some(x as usize)
    } else {
        None
    }
}
pub(crate) fn axes(values: &[i64], r: usize) -> Option<Vec<usize>> {
    let v = values
        .iter()
        .map(|n| axis(*n, r))
        .collect::<Option<Vec<_>>>()?;
    if v.iter().collect::<BTreeSet<_>>().len() != v.len() {
        return None;
    }
    Some(v)
}
fn ints(t: &Value) -> Option<Vec<i64>> {
    static_numbers(t)?.iter().map(safe).collect()
}
fn list(n: &Value, k: &str) -> Vec<i64> {
    attr_ints(n, k)
        .iter()
        .map(safe)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}
pub(crate) fn symbol(s: String) -> Value {
    json!({"kind":"symbolic","value":null,"parameter":s,"denotation":"","valueFieldCount":1})
}
pub(crate) fn key(d: &Value) -> String {
    if let Some(n) = dv(d) {
        format!("v:{n}")
    } else if d["kind"] == "symbolic" {
        format!("s:{}", string(&d["parameter"]))
    } else {
        "?".into()
    }
}
pub(crate) fn known(d: &Value) -> bool {
    dv(d).is_some_and(|n| n >= 0) || d["kind"] == "symbolic" && !string(&d["parameter"]).is_empty()
}
pub(crate) fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
pub(crate) fn invalid(reason: &str, details: Value) -> Value {
    json!({"status":"invalid","outputs":[],"reason":reason,"details":details})
}
fn one(v: &Value, s: Value, dt: &str, values: Value, error: &str) -> Value {
    let mut out = vec![];
    if set(v, &mut out, 0, &s, dt, values) {
        result(out)
    } else {
        unresolved(error)
    }
}
fn squeeze(v: &Value) -> R<Value> {
    let n = &v["node"];
    let x = input(v, 0);
    if shape(x).is_null() {
        return Ok(unresolved("squeeze_input_shape_unknown"));
    }
    let ds = dimensions(x);
    let mut ax = ints(input(v, 1));
    if truth(&n["inputs"][1]) && ax.is_none() {
        return Ok(unresolved("squeeze_axes_runtime_unknown"));
    }
    if ax.is_none() {
        ax = Some(list(n, "axes"));
    }
    let mut ax = ax.unwrap();
    let implicit = ax.is_empty();
    if implicit {
        ax = ds
            .iter()
            .enumerate()
            .filter(|(_, d)| dv(d) == Some(1))
            .map(|(i, _)| i as i64)
            .collect();
    }
    let Some(mut normalized) = axes(&ax, ds.len()) else {
        return Ok(invalid(
            "squeeze_axes_invalid",
            json!({"axes":ax,"input_rank":ds.len()}),
        ));
    };
    normalized.sort();
    if let Some(i) = normalized
        .iter()
        .copied()
        .find(|i| dv(&ds[*i]).is_some_and(|n| n != 1))
    {
        return Ok(invalid(
            "squeeze_axis_dimension_not_one",
            json!({"axis":i,"dimension":dv(&ds[i])}),
        ));
    }
    let conditional = if implicit {
        ds.iter()
            .enumerate()
            .filter(|(_, d)| dv(d).is_none())
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    } else {
        normalized
            .iter()
            .copied()
            .filter(|i| dv(&ds[*i]).is_none())
            .collect()
    };
    if conditional.is_empty() {
        let s = ds
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !normalized.contains(i))
            .map(|(_, d)| d)
            .collect::<Vec<_>>();
        return Ok(one(
            v,
            json!(s),
            kd(x),
            inherited(x, "Squeeze"),
            "squeeze_output_unresolved",
        ));
    }
    if conditional.len() > 6 {
        return Ok(unresolved(if implicit {
            "squeeze_implicit_axes_conditional_variant_limit_exceeded"
        } else {
            "squeeze_explicit_axes_conditional_variant_limit_exceeded"
        }));
    }
    let keys = conditional
        .iter()
        .map(|i| {
            if ds[*i]["kind"] == "symbolic" {
                format!(
                    "onnx:dimension:{}:equals:1",
                    encode(string(&ds[*i]["parameter"]))
                )
            } else {
                format!(
                    "onnx:tensor:{}:axis:{i}:equals:1",
                    encode(n["inputs"][0].as_str().unwrap_or("input"))
                )
            }
        })
        .collect::<Vec<_>>();
    let mut patches = vec![];
    let mut failures = vec![];
    let count = 1usize << conditional.len();
    for bits in 0..count {
        let assignments = conditional
            .iter()
            .enumerate()
            .map(|(j, i)| (*i, (bits & (1 << (conditional.len() - 1 - j))) == 0))
            .collect::<Vec<_>>();
        let conditions = assignments
            .iter()
            .enumerate()
            .map(|(j, (_, yes))| json!({"key":keys[j],"value":yes.to_string()}))
            .collect::<Vec<_>>();
        if !implicit && assignments.iter().any(|(_, yes)| !yes) {
            failures.push(json!({"status":"invalid","reason":"squeeze_axis_dimension_not_one","conditions":conditions,"details":{"axes":assignments.iter().filter(|(_,yes)|!yes).map(|(i,_)|*i).collect::<Vec<_>>()}}));
            continue;
        }
        let remove = normalized
            .iter()
            .copied()
            .chain(assignments.iter().filter(|(_, yes)| *yes).map(|(i, _)| *i))
            .collect::<BTreeSet<_>>();
        let s = ds
            .iter()
            .enumerate()
            .filter(|(i, _)| !remove.contains(i))
            .map(|(_, d)| d.clone())
            .collect::<Vec<_>>();
        let t = tensor_type(kd(x), &s, &v["tensor_type_ids"]);
        patches.push(json!({"dtype":kd(x),"shape":t["shape"],"shapeDeclared":true,"typeProto":t,"conditions":conditions}));
    }
    let mut merged = patches[0]["typeProto"].clone();
    for p in &patches[1..] {
        let r = crate::portable_onnx_type::run(
            &json!({"operation":"union","left":merged,"right":p["typeProto"],"tensor_type_ids":v["tensor_type_ids"]}),
        )?;
        if r["status"] != "pass" {
            return Ok(unresolved(
                "squeeze_implicit_axes_conditional_type_union_failed",
            ));
        }
        merged = r["type"].clone();
    }
    let mut patch = crate::portable_onnx_container::descriptor(&merged, json!({}));
    let mut sorted = keys;
    sorted.sort();
    if implicit {
        sorted.dedup();
    }
    patch["conditionalShapeContract"] = json!({"schema":"deepbom.onnx_conditional_shape_contract.v1","status":if implicit{"assessed_complete"}else{"assessed_partial"},"variant_count":patches.len(),"unassessed_variant_count":0,"invalid_variant_count":failures.len(),"condition_keys":sorted,"variant_failures":failures});
    patch["conditionalShapeVariants"] = json!(patches);
    if !truth(&n["outputs"][0]) {
        return Ok(unresolved("squeeze_output_unresolved"));
    }
    Ok(
        json!({"outputs":[[n["outputs"][0],patch]],"reason":if implicit{"squeeze_implicit_axes_finite_conditional_contract"}else{"squeeze_explicit_axes_guarded_contract"}}),
    )
}
fn pad(v: &Value) -> R<Value> {
    let n = &v["node"];
    let x = input(v, 0);
    if shape(x).is_null() {
        return Ok(unresolved("pad_input_shape_unknown"));
    }
    let ds = dimensions(x);
    let pp = truth(&n["inputs"][1]);
    let pa = !attr(n, "pads").is_null();
    let ap = truth(&n["inputs"][3]);
    let pads = if pp {
        ints(input(v, 1))
    } else if pa {
        Some(list(n, "pads"))
    } else {
        None
    };
    let ax = if ap {
        ints(input(v, 3))
    } else {
        Some((0..ds.len() as i64).collect())
    };
    let normalized = ax.as_ref().and_then(|a| axes(a, ds.len()));
    let vector = |t: &Value| {
        if rank(t) == Some(1) {
            dv(&dimensions(t)[0])
        } else {
            None
        }
    };
    let pc =
        pads.as_ref()
            .map(|v| v.len() as i64)
            .or_else(|| if pp { vector(input(v, 1)) } else { None });
    let ac = normalized.as_ref().map(|v| v.len() as i64).or_else(|| {
        if ap {
            vector(input(v, 3))
        } else {
            Some(ds.len() as i64)
        }
    });
    if !pp && !pa {
        return Ok(unresolved("pad_values_missing"));
    }
    if pp && rank(input(v, 1)).is_some_and(|r| r != 1) {
        return Ok(unresolved("pad_values_tensor_rank_invalid"));
    }
    if ap && rank(input(v, 3)).is_some_and(|r| r != 1) {
        return Ok(unresolved("pad_axes_tensor_rank_invalid"));
    }
    if ax.is_some() && normalized.is_none() {
        return Ok(unresolved("pad_axes_invalid"));
    }
    if let (Some(p), Some(a)) = (pc, ac) {
        if p != 2 * a {
            return Ok(unresolved("pad_axes_or_cardinality_invalid"));
        }
    }
    if pads.is_none() || normalized.is_none() {
        let s = ds
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if normalized.as_ref().is_some_and(|a| !a.contains(&i)) {
                    d.clone()
                } else {
                    unknown()
                }
            })
            .collect::<Vec<_>>();
        let mut r = one(
            v,
            json!(s),
            kd(x),
            Value::Null,
            "pad_partial_output_unresolved",
        );
        if !rows(&r["outputs"]).is_empty() {
            r["reason"] = json!(if pads.is_none() {
                "pad_values_runtime_bound_preserve_rank_only"
            } else {
                "pad_axes_runtime_bound_preserve_rank_only"
            });
        }
        return Ok(r);
    }
    let s = kernel("pad", json!({"input":ds,"pads":pads,"axes":normalized}))?;
    if s.is_null() {
        return Ok(unresolved("pad_crop_exceeds_input_dimension"));
    }
    Ok(one(v, s, kd(x), Value::Null, "pad_output_unresolved"))
}
fn range(v: &Value) -> Value {
    let ts = [input(v, 0), input(v, 1), input(v, 2)];
    let numbers = ts
        .iter()
        .map(|t| static_numbers(t).and_then(|n| n.first().and_then(Value::as_f64)))
        .collect::<Option<Vec<_>>>();
    if let Some(n) = numbers {
        let (a, b, d) = (n[0], n[1], n[2]);
        if d == 0. {
            return unresolved("range_delta_zero_invalid");
        }
        let size = ((b - a) / d).ceil().max(0.);
        if !size.is_finite() || size > SAFE as f64 {
            return unresolved("range_output_cardinality_unsafe");
        }
        let vals = if size <= 65536. {
            let vals = (0..size as usize)
                .map(|i| a + i as f64 * d)
                .collect::<Vec<_>>();
            if vals.iter().all(|n| n.is_finite()) {
                json!({"values":vals,"source":"Range"})
            } else {
                Value::Null
            }
        } else {
            Value::Null
        };
        return one(
            v,
            json!([dim(size as i64)]),
            kd(ts[0]),
            vals,
            "range_output_unresolved",
        );
    }
    let terms = ts
        .iter()
        .map(|t| {
            if let Some(n) = static_numbers(t).filter(|n| n.len() == 1) {
                return Some((n[0].to_string(), n[0].as_f64()));
            }
            exact_dims(t)
                .filter(|n| n.len() == 1 && (dv(&n[0]).is_some() || n[0]["kind"] == "symbolic"))
                .map(|ds| (key(&ds[0]), dv(&ds[0]).map(|n| n as f64)))
        })
        .collect::<Option<Vec<_>>>();
    let Some(terms) = terms else {
        return unresolved("range_controls_not_static_finite_scalars");
    };
    let Some(delta) = terms[2].1 else {
        return unresolved("range_delta_not_artifact_bound");
    };
    if delta == 0. {
        return unresolved("range_delta_zero_invalid");
    }
    let keys = terms
        .iter()
        .map(|(s, n)| {
            if let Some(n) = n {
                format!(
                    "v:{}",
                    crate::portable_identity::canonical(&json!(n)).unwrap_or_default()
                )
            } else {
                s.clone()
            }
        })
        .collect::<Vec<_>>();
    one(
        v,
        json!([symbol(format!(
            "deepbom_expr:range_len({})",
            keys.join(",")
        ))]),
        kd(ts[0]),
        Value::Null,
        "range_output_unresolved",
    )
}
fn lstm(v: &Value) -> Value {
    let n = &v["node"];
    let x = input(v, 0);
    if rank(x) != Some(3) {
        return unresolved("lstm_input_rank_unknown_or_invalid");
    }
    let ds = dimensions(x);
    let layout = ai_exact(n, "layout", 0);
    if ![0, 1].contains(&layout) {
        return unresolved("lstm_layout_invalid");
    }
    let direction = match astr(n, "direction", "forward") {
        "bidirectional" => dim(2),
        "forward" | "reverse" => dim(1),
        _ => unknown(),
    };
    let hidden = ai_exact(n, "hidden_size", -1);
    let hidden = if hidden > 0 { dim(hidden) } else { unknown() };
    let sequence = ds[if layout == 0 { 0 } else { 1 }].clone();
    let batch = ds[if layout == 0 { 1 } else { 0 }].clone();
    let dt = kd(x);
    if dt.is_empty() {
        return unresolved("lstm_input_dtype_unknown");
    }
    let shape = if layout == 0 {
        vec![sequence, direction.clone(), batch.clone(), hidden.clone()]
    } else {
        vec![batch.clone(), sequence, direction.clone(), hidden.clone()]
    };
    let state = if layout == 0 {
        vec![direction, batch, hidden]
    } else {
        vec![batch, direction, hidden]
    };
    let mut out = vec![];
    for (i, s) in [(0, shape), (1, state.clone()), (2, state)] {
        set(v, &mut out, i, &json!(s), dt, Value::Null);
    }
    if rows(&n["outputs"]).iter().any(truth) {
        result(out)
    } else {
        unresolved("lstm_output_omitted")
    }
}
fn stft(v: &Value) -> Value {
    let n = &v["node"];
    let x = input(v, 0);
    if rank(x).is_none_or(|r| !(2..=3).contains(&r)) {
        return unresolved("stft_signal_shape_unknown_or_invalid");
    }
    let ds = dimensions(x);
    let step = ints(input(v, 1)).and_then(|n| n.first().copied());
    let declared = ints(input(v, 3))
        .and_then(|n| n.first().copied())
        .filter(|n| *n != 0);
    let window = input(v, 2);
    let window = if rank(window) == Some(1) {
        dv(&dimensions(window)[0])
    } else {
        None
    };
    let frame = declared.or(window);
    if step.is_none_or(|n| n <= 0)
        || frame.is_none_or(|n| n <= 0)
        || !known(&ds[1])
        || dv(&ds[1]).is_some_and(|n| n < frame.unwrap())
    {
        return unresolved("stft_frame_contract_not_static_or_invalid");
    }
    let complex = if ds.len() == 3 { dv(&ds[2]) } else { None };
    if ds.len() == 3 && complex.is_none_or(|n| ![1, 2].contains(&n)) {
        return unresolved("stft_signal_complex_dimension_invalid");
    }
    let onesided = ai_exact(n, "onesided", 1);
    if complex == Some(2) && onesided != 0 {
        return unresolved("stft_onesided_complex_input_invalid");
    }
    let (step, frame) = (step.unwrap(), frame.unwrap());
    let frames = if let Some(n) = dv(&ds[1]) {
        dim(1 + (n - frame) / step)
    } else {
        symbol(format!(
            "deepbom_expr:add(floor_div(deepbom_expr:add({},-{frame}),{step}),1)",
            key(&ds[1])
        ))
    };
    let bins = if onesided == 0 { frame } else { frame / 2 + 1 };
    one(
        v,
        json!([ds[0], frames, dim(bins), dim(2)]),
        kd(x),
        Value::Null,
        "stft_output_unresolved",
    )
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["node"]["opType"]) {
        "Squeeze" => squeeze(v),
        "Pad" => pad(v),
        "Range" => Ok(range(v)),
        "LSTM" => Ok(lstm(v)),
        "STFT" => Ok(stft(v)),
        "NonZero" => {
            let x = input(v, 0);
            if shape(x).is_null() {
                return Ok(unresolved("nonzero_input_rank_unknown"));
            }
            let name = v["node"]["inputs"][0].as_str().unwrap_or("input");
            let sym = format!("deepbom_runtime:nnz:{}", encode(name));
            let ds = dimensions(x);
            let mut r = one(
                v,
                json!([dim(ds.len() as i64), symbol(sym.clone())]),
                "INT64",
                Value::Null,
                "nonzero_output_unresolved",
            );
            if rows(&r["outputs"]).is_empty() {
                return Ok(r);
            }
            r["outputs"][0][1]["runtimeDimensionBounds"] = json!([{"axis":1,"symbol":sym,"lower_bound_decimal":"0","upper_bound_expression":if ds.iter().all(known){Some(format!("deepbom_expr:mul({})",ds.iter().map(key).collect::<Vec<_>>().join(",")))}else{None},"basis":"ONNX NonZero output cardinality is bounded by the logical input element count"}]);
            r["reason"] = json!("nonzero_runtime_nnz_symbolically_bounded");
            Ok(r)
        }
        _ => crate::portable_onnx_elementwise::run(v),
    }
}
