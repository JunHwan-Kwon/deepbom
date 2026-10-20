//! Standard ONNX contraction and attention contracts, independent of the host.
use crate::portable_onnx_dynamic::{invalid, key, known};
use crate::portable_onnx_node::{ai_exact, kd, kernel, set, shape, spatial};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeMap;
fn mismatch(a: &Value, b: &Value) -> bool {
    matches!((dv(a),dv(b)),(Some(x),Some(y))if x!=y)
}
fn fail(s: &str) -> Value {
    invalid(s, Value::Null)
}
fn reason(mut r: Value, s: &str) -> Value {
    if !s.is_empty() {
        r["reason"] = json!(s);
    }
    r
}
fn einsum(v: &Value) -> R<Value> {
    let n = &v["node"];
    let inputs = rows(&n["inputs"])
        .iter()
        .filter(|x| truth(x))
        .map(|x| &v["tensors"][string(x)])
        .collect::<Vec<_>>();
    if inputs.is_empty() || inputs.iter().any(|t| shape(t).is_null()) {
        return Ok(unresolved("einsum_input_rank_unknown"));
    }
    let parsed = crate::portable_einsum::parse(
        &json!({"source":astr(n,"equation","").split_whitespace().collect::<String>(),"ranks":inputs.iter().map(|t|dimensions(t).len()).collect::<Vec<_>>()}),
    );
    if parsed["status"] != "assessed" {
        return Ok(fail(string(&parsed["reason"])));
    }
    let dtypes = inputs.iter().map(|t| kd(t)).collect::<Vec<_>>();
    if dtypes.iter().any(|s| s.is_empty()) {
        return Ok(unresolved("einsum_input_dtype_unknown"));
    }
    if dtypes.iter().any(|s| *s != dtypes[0]) {
        return Ok(invalid(
            "einsum_input_dtype_mismatch",
            json!({"input_dtypes":dtypes}),
        ));
    }
    let mut labels: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (i, t) in inputs.iter().enumerate() {
        let ds = dimensions(t);
        for (j, l) in rows(&parsed["operands"][i]).iter().enumerate() {
            labels
                .entry(string(l).into())
                .or_default()
                .push(ds[j].clone());
        }
    }
    let mut merged = BTreeMap::new();
    let mut guard = false;
    for (label, ds) in labels {
        if label.starts_with("@ellipsis:") {
            let b = kernel(
                "broadcast",
                json!({"shapes":ds.iter().map(|d|vec![d]).collect::<Vec<_>>()}),
            )?;
            if b.is_null() {
                return Ok(invalid(
                    "einsum_ellipsis_dimensions_not_broadcast_compatible",
                    json!({"label":label}),
                ));
            }
            merged.insert(label, b[0].clone());
            continue;
        }
        let mut current = ds[0].clone();
        for d in &ds[1..] {
            if mismatch(&current, d) {
                return Ok(invalid(
                    "einsum_label_dimension_mismatch",
                    json!({"label":label,"left":dv(&current),"right":dv(d)}),
                ));
            }
            if key(&current) == key(d) {
                continue;
            }
            if !known(&current) {
                current = d.clone();
            } else if known(d) {
                current = unknown();
                guard = true;
            }
        }
        merged.insert(label, current);
    }
    let s = rows(&parsed["output"])
        .iter()
        .map(|l| merged.get(string(l)).cloned().unwrap_or_else(unknown))
        .collect::<Vec<_>>();
    let mut out = vec![];
    Ok(if set(v, &mut out, 0, &json!(s), dtypes[0], Value::Null) {
        reason(
            result(out),
            if guard {
                "einsum_symbolic_label_equality_runtime_guard"
            } else {
                ""
            },
        )
    } else {
        unresolved("einsum_output_unresolved")
    })
}
fn attention(v: &Value) -> R<Value> {
    let n = &v["node"];
    let (qt, kt, vt) = (input(v, 0), input(v, 1), input(v, 2));
    if [qt, kt, vt].iter().any(|t| shape(t).is_null()) {
        return Ok(unresolved("attention_qkv_rank_unknown"));
    }
    let (q, k, w) = (dimensions(qt), dimensions(kt), dimensions(vt));
    let ranks = vec![q.len(), k.len(), w.len()];
    if ![3, 4].contains(&q.len()) || k.len() != q.len() || w.len() != q.len() {
        return Ok(invalid(
            "attention_qkv_rank_mismatch",
            json!({"ranks":ranks}),
        ));
    }
    let dtypes = [kd(qt), kd(kt), kd(vt)];
    if dtypes.iter().any(|s| s.is_empty()) {
        return Ok(unresolved("attention_qkv_dtype_unknown"));
    }
    if dtypes[0] != dtypes[1] {
        return Ok(fail("attention_query_key_dtype_mismatch"));
    }
    if mismatch(&q[0], &k[0]) || mismatch(&q[0], &w[0]) {
        return Ok(fail("attention_batch_dimension_mismatch"));
    }
    let (pk, pv, mask, np) = (input(v, 4), input(v, 5), input(v, 3), input(v, 6));
    let past = !pk.is_null() || !pv.is_null();
    if past && (pk.is_null() || pv.is_null()) {
        return Ok(fail("attention_past_key_value_pair_incomplete"));
    }
    if !np.is_null() && past {
        return Ok(fail("attention_nonpad_length_with_past_cache"));
    }
    let (qh, kh, qs, ks, qsize, vsize, output) = if q.len() == 4 {
        if mismatch(&k[2], &w[2]) || mismatch(&k[1], &w[1]) || mismatch(&q[3], &k[3]) {
            return Ok(fail("attention_4d_head_or_sequence_contract_mismatch"));
        }
        if let (Some(a), Some(b)) = (dv(&q[1]), dv(&k[1])) {
            if a <= 0 || b <= 0 || a < b || a % b != 0 {
                return Ok(fail("attention_4d_head_count_contract_invalid"));
            }
        }
        (
            q[1].clone(),
            k[1].clone(),
            q[2].clone(),
            k[2].clone(),
            q[3].clone(),
            w[3].clone(),
            vec![q[0].clone(), q[1].clone(), q[2].clone(), w[3].clone()],
        )
    } else {
        let (a, b) = (
            ai_exact(n, "q_num_heads", 0),
            ai_exact(n, "kv_num_heads", 0),
        );
        if a <= 0 || b <= 0 || a < b || a % b != 0 {
            return Ok(fail("attention_3d_head_attributes_invalid"));
        }
        let (x, y, z) = (dv(&q[2]), dv(&k[2]), dv(&w[2]));
        if x.is_some_and(|n| n % a != 0)
            || y.is_some_and(|n| n % b != 0)
            || z.is_some_and(|n| n % b != 0)
        {
            return Ok(fail("attention_hidden_size_not_divisible_by_head_count"));
        }
        let (qsize, ksize, vsize) = (
            x.map(|n| dim(n / a)).unwrap_or_else(unknown),
            y.map(|n| dim(n / b)).unwrap_or_else(unknown),
            z.map(|n| dim(n / b)).unwrap_or_else(unknown),
        );
        if mismatch(&qsize, &ksize) || mismatch(&k[1], &w[1]) {
            return Ok(fail("attention_3d_head_or_sequence_contract_mismatch"));
        }
        let hidden = kernel("multiply", json!({"left":vsize,"right":dim(a)}))?;
        (
            dim(a),
            dim(b),
            q[1].clone(),
            k[1].clone(),
            qsize,
            vsize,
            vec![q[0].clone(), q[1].clone(), hidden],
        )
    };
    let present = truth(&n["outputs"][1]) || truth(&n["outputs"][2]);
    if present && !(truth(&n["outputs"][1]) && truth(&n["outputs"][2])) {
        return Ok(fail("attention_present_key_value_pair_incomplete"));
    }
    if past != present {
        return Ok(fail("attention_past_present_cache_contract_incomplete"));
    }
    if !np.is_null() && present {
        return Ok(fail("attention_nonpad_length_with_present_cache"));
    }
    let mut ps = dim(0);
    if past {
        if rank(pk) != Some(4) || rank(pv) != Some(4) {
            return Ok(fail("attention_past_cache_rank_mismatch"));
        }
        if !kd(pk).is_empty() && kd(pk) != dtypes[0] {
            return Ok(fail("attention_past_key_dtype_mismatch"));
        }
        if !kd(pv).is_empty() && kd(pv) != dtypes[2] {
            return Ok(fail("attention_past_value_dtype_mismatch"));
        }
        let (a, b) = (dimensions(pk), dimensions(pv));
        if [
            (&a[0], &q[0]),
            (&a[1], &kh),
            (&a[2], &b[2]),
            (&a[3], &qsize),
            (&b[0], &q[0]),
            (&b[1], &kh),
            (&b[3], &vsize),
        ]
        .iter()
        .any(|(a, b)| mismatch(a, b))
        {
            return Ok(fail("attention_past_cache_contract_mismatch"));
        }
        ps = a[2].clone();
    }
    let total = kernel("add", json!({"left":ps,"right":ks}))?;
    let mut guard = !known(&qsize) || !known(&vsize);
    if !mask.is_null() {
        if shape(mask).is_null() {
            return Ok(unresolved("attention_mask_rank_unknown"));
        }
        let ms = dimensions(mask);
        if ms.len() > 4 {
            return Ok(fail("attention_mask_rank_invalid"));
        }
        let mut padded = vec![dim(1); 4 - ms.len()];
        padded.extend(ms);
        let target = [q[0].clone(), qh.clone(), qs.clone(), total.clone()];
        for i in 0..4 {
            let (a, b) = (dv(&padded[i]), dv(&target[i]));
            if let (Some(observed), Some(expected)) = (a, b) {
                if observed != 1 {
                    if i == 3 && observed > expected {
                        return Ok(fail("attention_mask_last_dimension_exceeds_total_sequence"));
                    }
                    if i != 3 && observed != expected {
                        return Ok(invalid(
                            "attention_mask_not_broadcastable",
                            json!({"axis":i,"observed":observed,"expected":expected}),
                        ));
                    }
                }
            } else {
                guard = true;
            }
        }
    }
    let mut out = vec![];
    if !set(v, &mut out, 0, &json!(output), dtypes[0], Value::Null) {
        return Ok(unresolved("attention_output_unresolved"));
    }
    for (i, s, dt, err) in [
        (
            1,
            vec![q[0].clone(), kh.clone(), total.clone(), qsize],
            dtypes[0],
            "attention_present_key_unresolved",
        ),
        (
            2,
            vec![q[0].clone(), kh, total.clone(), vsize],
            dtypes[2],
            "attention_present_value_unresolved",
        ),
        (
            3,
            vec![q[0].clone(), qh, qs, total],
            dtypes[0],
            "attention_qk_output_unresolved",
        ),
    ] {
        if truth(&n["outputs"][i]) && !set(v, &mut out, i, &json!(s), dt, Value::Null) {
            return Ok(unresolved(err));
        }
    }
    Ok(reason(
        result(out),
        if guard {
            "attention_symbolic_dimension_runtime_guard"
        } else {
            ""
        },
    ))
}
fn deform(v: &Value) -> R<Value> {
    let (x, w, offset, bias, mask) = (
        input(v, 0),
        input(v, 1),
        input(v, 2),
        input(v, 3),
        input(v, 4),
    );
    if rank(x).is_some() && rank(w).is_some() && rank(x) != rank(w) {
        return Ok(invalid(
            "deform_conv_input_weight_rank_mismatch",
            json!({"input_rank":rank(x),"weight_rank":rank(w)}),
        ));
    }
    let s = kernel(
        "conv",
        json!({"input":shape(x),"weight":w["shape"],"attributes":spatial(&v["node"])}),
    )?;
    let types = [x, w, offset, bias, mask]
        .iter()
        .map(|t| kd(t))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if types.iter().any(|t| *t != types[0]) {
        return Ok(invalid(
            "deform_conv_input_dtype_mismatch",
            json!({"input_dtypes":types}),
        ));
    }
    let mut why = "";
    if [x, w, offset].iter().any(|t| shape(t).is_null()) || s.is_null() {
        why = "deform_conv_auxiliary_contract_runtime_unbound";
    } else {
        let (a, b, o) = (dimensions(x), dimensions(w), dimensions(offset));
        if a.len() < 3 || a.len() != b.len() || a.len() != o.len() {
            return Ok(fail("deform_conv_required_rank_mismatch"));
        }
        let (g, og) = (
            ai_exact(&v["node"], "group", 1),
            ai_exact(&v["node"], "offset_group", 1),
        );
        let volume = b[2..].iter().try_fold(1i128, |a, d| {
            dv(d)
                .filter(|n| *n > 0)
                .and_then(|n| a.checked_mul(n as i128))
        });
        if g <= 0
            || og <= 0
            || volume.is_none()
            || dv(&b[0]).is_none_or(|n| n <= 0)
            || dv(&b[1]).is_none_or(|n| n <= 0)
        {
            return Ok(fail("deform_conv_group_or_kernel_contract_invalid"));
        }
        let bc = dv(&b[1]).unwrap();
        if dv(&a[1]).is_some_and(|n| n > 0 && (n as i128 != bc as i128 * g as i128 || n % og != 0))
            || dv(&b[0]).unwrap() % g != 0
        {
            return Ok(fail("deform_conv_channel_group_contract_invalid"));
        }
        let count = volume
            .unwrap()
            .checked_mul(og as i128)
            .filter(|n| *n <= SAFE as i128);
        let Some(count) = count else {
            return Ok(unresolved(
                "deform_conv_auxiliary_size_exceeds_safe_integer",
            ));
        };
        let offset_count = count
            .checked_mul((a.len() - 2) as i128)
            .filter(|n| *n <= SAFE as i128);
        let Some(offset_count) = offset_count else {
            return Ok(unresolved(
                "deform_conv_auxiliary_size_exceeds_safe_integer",
            ));
        };
        let mut expected = vec![a[0].clone(), dim(offset_count as i64)];
        expected.extend_from_slice(&rows(&s)[2..]);
        if o.iter().zip(&expected).any(|(a, b)| mismatch(a, b)) {
            return Ok(invalid(
                "deform_conv_offset_shape_mismatch",
                json!({"expected_shape":expected.iter().map(dv).collect::<Vec<_>>(),"observed_shape":offset["shape"]}),
            ));
        }
        if !bias.is_null() && rank(bias).is_some() {
            let ds = dimensions(bias);
            if ds.len() != 1 || dv(&ds[0]).is_some_and(|n| n > 0 && Some(n) != dv(&b[0])) {
                return Ok(fail("deform_conv_bias_shape_mismatch"));
            }
        }
        if !mask.is_null() && rank(mask).is_some() {
            expected[1] = dim(count as i64);
            let ds = dimensions(mask);
            if ds.len() != a.len() || ds.iter().zip(&expected).any(|(a, b)| mismatch(a, b)) {
                return Ok(fail("deform_conv_mask_shape_mismatch"));
            }
        }
        if [offset, bias, mask]
            .iter()
            .filter(|t| !t.is_null())
            .any(|t| {
                shape(t).is_null() || dimensions(t).iter().any(|d| dv(d).is_none_or(|n| n < 0))
            })
        {
            why = "deform_conv_auxiliary_dimensions_runtime_unbound";
        }
    }
    let mut out = vec![];
    let dt = if kd(x).is_empty() { kd(w) } else { kd(x) };
    Ok(if set(v, &mut out, 0, &s, dt, Value::Null) {
        reason(result(out), why)
    } else {
        unresolved("deform_conv_shape_contract_unresolved")
    })
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["node"]["opType"]) {
        "Attention" => attention(v),
        "Einsum" => einsum(v),
        "DeformConv" => deform(v),
        "Slice" => crate::portable_onnx_slice::run(v),
        _ => Ok(Value::Null),
    }
}
