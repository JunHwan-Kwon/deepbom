//! Exact nominal contraction cost. Source classification and format transport
//! remain adapters; dimensions, compatibility and arithmetic have one owner.
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde_json::{json, Value};
const SAFE: u64 = 9_007_199_254_740_991;
fn no(reason: impl Into<String>) -> Value {
    json!({"value":null,"value_decimal":null,"status":"not_assessed","reason":reason.into()})
}
fn yes(value: BigUint, reason: &str) -> Value {
    json!({"value":value.to_u64().filter(|n|*n<=SAFE),"value_decimal":value.to_string(),"status":"assessed","reason":reason})
}
fn product(values: impl IntoIterator<Item = u64>) -> BigUint {
    values
        .into_iter()
        .fold(BigUint::from(1u32), |a, b| a * BigUint::from(b))
}
fn shape(t: &Value) -> Option<Vec<u64>> {
    if t["known"] != true {
        return None;
    }
    t["shape"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().filter(|n| *n <= SAFE))
        .collect()
}
fn int(a: &Value, k: &str, default: i64) -> Option<i64> {
    if a.get(k).is_none() {
        Some(default)
    } else {
        a[k].as_i64().filter(|n| n.unsigned_abs() <= SAFE)
    }
}
fn spatial(a: &Value, k: &str, rank: usize, default: u64, zero: bool) -> Option<Vec<u64>> {
    if a.get(k).is_none() {
        return Some(vec![default; rank]);
    }
    let list = a[k].as_array()?;
    if list.len() != rank {
        return None;
    }
    list.iter()
        .map(|n| n.as_u64().filter(|n| *n <= SAFE && (zero || *n > 0)))
        .collect()
}
use crate::portable_contracts::broadcast;
fn pairs(
    input: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
    pad: u64,
    output: u64,
) -> Option<u128> {
    crate::portable_integer::pairs(
        &[input, kernel, stride, dilation, pad, output].map(BigInt::from),
    )?
    .to_u128()
}
fn transpose_pads(
    a: &Value,
    x: &[u64],
    w: &[u64],
    y: &[u64],
    s: &[u64],
    d: &[u64],
    op: &[u64],
) -> Option<Vec<u64>> {
    let rank = x.len();
    if op.iter().enumerate().any(|(i, p)| *p >= s[i] && *p >= d[i]) {
        return None;
    }
    let mode = a["auto_pad"].as_str().unwrap_or("NOTSET");
    if !["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"].contains(&mode) {
        return None;
    }
    if a.get("kernel_shape").is_some() && spatial(a, "kernel_shape", rank, 0, true)?.as_slice() != w
    {
        return None;
    }
    let has_output = a.get("output_shape").is_some();
    if has_output && spatial(a, "output_shape", rank, 0, true)?.as_slice() != y {
        return None;
    }
    let mut pads;
    if has_output || mode.starts_with("SAME_") {
        if !has_output
            && y.iter()
                .enumerate()
                .any(|(i, n)| *n as u128 != x[i] as u128 * s[i] as u128)
        {
            return None;
        }
        pads = vec![];
        let mut ends = vec![];
        for i in 0..rank {
            let total = s[i] as i128 * (x[i] as i128 - 1)
                + op[i] as i128
                + (w[i] as i128 - 1) * d[i] as i128
                + 1
                - y[i] as i128;
            if total < 0 || total > SAFE as i128 {
                return None;
            }
            let first = if mode == "SAME_UPPER" {
                total / 2
            } else {
                total - total / 2
            };
            pads.push(first as u64);
            ends.push((total - first) as u64);
        }
        pads.extend(ends);
    } else if mode == "VALID" {
        if a.get("pads").is_some() {
            return None;
        }
        pads = vec![0; rank * 2];
    } else {
        pads = spatial(a, "pads", rank * 2, 0, true)?;
    }
    for i in 0..rank {
        let expected = s[i] as i128 * (x[i] as i128 - 1)
            + op[i] as i128
            + (w[i] as i128 - 1) * d[i] as i128
            + 1
            - pads[i] as i128
            - pads[i + rank] as i128;
        if expected != y[i] as i128 {
            return None;
        }
    }
    Some(pads)
}
pub(crate) fn cost(v: &Value) -> Result<Value, String> {
    let op = v["op"].as_str().ok_or("Missing cost operation")?;
    let a = &v["attributes"];
    if matches!(
        op,
        "Attention" | "DeformConv" | "RNN" | "GRU" | "LSTM" | "Einsum"
    ) {
        return Ok(extended(v, op, a));
    }
    let input = &v["input"];
    let weight = &v["weight"];
    let output = &v["output"];
    if input.is_null() || weight.is_null() || output.is_null() {
        return Ok(no(format!(
            "{op} input, weight, or output tensor metadata is missing."
        )));
    }
    let shapes = (shape(input), shape(weight), shape(output));
    match op {
        "Conv" | "QLinearConv" | "ConvInteger" => {
            let (Some(x), Some(w), Some(y)) = shapes else {
                return Ok(no(format!("{op} MACs require known equal-rank N,C,D1...Dn input/output and M,C/group,k1...kn weight shapes.")));
            };
            if x.len() < 3 || w.len() != x.len() || y.len() != x.len() {
                return Ok(no(format!("{op} MACs require known equal-rank N,C,D1...Dn input/output and M,C/group,k1...kn weight shapes.")));
            }
            let g = int(a, "group", 1).unwrap_or(0);
            if g <= 0
                || w.contains(&0)
                || x[0] != y[0]
                || y[1] != w[0]
                || x[1] as u128 != w[1] as u128 * g as u128
                || w[0] % g as u64 != 0
            {
                return Ok(no(format!(
                    "{op} tensor channels, batch, or group dimensions are incompatible."
                )));
            }
            Ok(yes(product(y.into_iter().chain(w[1..].iter().copied())),"product(output shape) * product(weight shape excluding output-channel axis) from compatible ONNX N,C,D1...Dn and M,C/group,k1...kn tensors."))
        }
        "Gemm" => {
            let (Some(x), Some(w), Some(y)) = shapes else {
                return Ok(no("Gemm MACs require known rank-2 A and B shapes."));
            };
            if x.len() != 2 || w.len() != 2 {
                return Ok(no("Gemm MACs require known rank-2 A and B shapes."));
            }
            let (ta, tb) = (int(a, "transA", 0), int(a, "transB", 0));
            if !matches!(ta, Some(0 | 1)) || !matches!(tb, Some(0 | 1)) {
                return Ok(no("Gemm transA/transB must be 0 or 1."));
            }
            let ta = ta.unwrap() as usize;
            let tb = tb.unwrap() as usize;
            let (m, k, n) = (x[ta], x[1 - ta], w[1 - tb]);
            if k != w[tb] || y != [m, n] {
                return Ok(no("Gemm inner dimensions or output shape are incompatible after transpose attributes."));
            }
            Ok(yes(
                product([m, n, k]),
                "M*N*K from compatible Gemm A/B/output shapes after transA/transB.",
            ))
        }
        "MatMul" | "QLinearMatMul" | "MatMulInteger" => {
            let (Some(mut x), Some(mut w), Some(y)) = shapes else {
                return Ok(no(format!(
                    "{op} MACs require known rank-1-or-greater input and output shapes."
                )));
            };
            if x.is_empty() || w.is_empty() {
                return Ok(no(format!(
                    "{op} MACs require known rank-1-or-greater input and output shapes."
                )));
            }
            let (vx, vw) = (x.len() == 1, w.len() == 1);
            if vx {
                x.insert(0, 1);
            }
            if vw {
                w.push(1);
            }
            let (m, k, n) = (x[x.len() - 2], x[x.len() - 1], w[w.len() - 1]);
            if k != w[w.len() - 2] {
                return Ok(no(format!("{op} inner dimensions are incompatible.")));
            }
            let Some(batch) = broadcast(&x[..x.len() - 2], &w[..w.len() - 2]) else {
                return Ok(no(format!(
                    "{op} batch dimensions are not broadcast-compatible."
                )));
            };
            let mut expected = batch.clone();
            if !vx {
                expected.push(m);
            }
            if !vw {
                expected.push(n);
            }
            if y != expected {
                return Ok(no(format!("{op} output shape is incompatible with ONNX MatMul rank promotion and batch broadcasting.")));
            }
            Ok(yes(product(batch.into_iter().chain([m,n,k])),"broadcast_batch*M*N*K after ONNX rank-1 promotion, batch broadcasting, and output-shape validation."))
        }
        "ConvTranspose" => {
            let invalid = || {
                no("ConvTranspose requires compatible known N,C,D1...Dn, C,M/group,k1...kn, and N,M,O1...On tensors with a positive group.")
            };
            let (Some(x), Some(w), Some(y)) = shapes else {
                return Ok(invalid());
            };
            let g = int(a, "group", 1).unwrap_or(0);
            if x.len() < 3
                || w.len() != x.len()
                || y.len() != x.len()
                || g <= 0
                || x[0] != y[0]
                || x[1] != w[0]
                || x[1] % g as u64 != 0
                || y[1] as u128 != w[1] as u128 * g as u128
                || w[1..].contains(&0)
            {
                return Ok(invalid());
            }
            let rank = x.len() - 2;
            let attrs = || {
                no("ConvTranspose spatial attributes or serialized output shape are inconsistent with the ONNX contract.")
            };
            let (Some(s), Some(d), Some(op)) = (
                spatial(a, "strides", rank, 1, false),
                spatial(a, "dilations", rank, 1, false),
                spatial(a, "output_padding", rank, 0, true),
            ) else {
                return Ok(attrs());
            };
            let Some(p) = transpose_pads(a, &x[2..], &w[2..], &y[2..], &s, &d, &op) else {
                return Ok(attrs());
            };
            let mut value = product([x[0], x[1], w[1]]);
            for i in 0..rank {
                let Some(count) = pairs(x[i + 2], w[i + 2], s[i], d[i], p[i], y[i + 2]) else {
                    return Ok(no("ConvTranspose contributing-pair count exceeds the bounded exact arithmetic path."));
                };
                value *= BigUint::from(count);
            }
            Ok(yes(value,"Exact contributing input/kernel pairs per spatial axis times N*C*(M/group), after ONNX pad, stride, dilation, output-padding, output-shape, and group validation."))
        }
        _ => Err("Unsupported portable contraction operation".into()),
    }
}
fn extended(v: &Value, op: &str, a: &Value) -> Value {
    let input = &v["input"];
    let weight = &v["weight"];
    let output = &v["output"];
    let inputs = &v["inputs"];
    if op == "Attention" {
        let (Some(q), Some(k), Some(val), Some(y)) = (
            shape(input),
            shape(&inputs[1]),
            shape(&inputs[2]),
            shape(output),
        ) else {
            return no("Attention nominal MACs require static Q, K, V, and Y shapes.");
        };
        let rank = q.len();
        if ![3, 4].contains(&rank) || k.len() != rank || val.len() != rank {
            return no("Attention Q, K, and V must share source-compatible rank 3 or 4.");
        }
        let (b, h, kh, qs, ks, qsize, vsize, expected) = if rank == 4 {
            if k[0] != q[0]
                || val[0] != q[0]
                || k[1] != val[1]
                || k[2] != val[2]
                || k[3] != q[3]
                || k[1] == 0
                || q[1] < k[1]
                || q[1] % k[1] != 0
            {
                return no("Attention 4D Q/K/V head or sequence dimensions are incompatible.");
            }
            (
                q[0],
                q[1],
                k[1],
                q[2],
                k[2],
                q[3],
                val[3],
                vec![q[0], q[1], q[2], val[3]],
            )
        } else {
            let (h, kh) = (
                int(a, "q_num_heads", 0).unwrap_or(0),
                int(a, "kv_num_heads", 0).unwrap_or(0),
            );
            if h <= 0
                || kh <= 0
                || h < kh
                || h % kh != 0
                || q[2] % h as u64 != 0
                || k[2] % kh as u64 != 0
                || val[2] % kh as u64 != 0
            {
                return no(
                    "Attention 3D hidden widths and q_num_heads/kv_num_heads are incompatible.",
                );
            }
            let (h, kh) = (h as u64, kh as u64);
            let qsize = q[2] / h;
            let vsize = val[2] / kh;
            if k[0] != q[0] || val[0] != q[0] || k[1] != val[1] || k[2] / kh != qsize {
                return no(
                    "Attention 3D Q/K/V batch, sequence, or head dimensions are incompatible.",
                );
            }
            let Some(width) = h.checked_mul(vsize).filter(|n| *n <= SAFE) else {
                return no(
                    "Attention Y shape is incompatible with the source-defined Q and V contract.",
                );
            };
            (
                q[0],
                h,
                kh,
                q[1],
                k[1],
                qsize,
                vsize,
                vec![q[0], q[1], width],
            )
        };
        if y != expected {
            return no(
                "Attention Y shape is incompatible with the source-defined Q and V contract.",
            );
        }
        let mut past = 0;
        if !inputs[4].is_null() || !inputs[5].is_null() {
            let invalid =
                || no("Attention past key/value cache contracts are incomplete or incompatible.");
            let (Some(pk), Some(pv)) = (shape(&inputs[4]), shape(&inputs[5])) else {
                return invalid();
            };
            if pk.len() != 4
                || pv.len() != 4
                || pk[0] != b
                || pv[0] != b
                || pk[1] != kh
                || pv[1] != kh
                || pk[2] != pv[2]
                || pk[3] != qsize
                || pv[3] != vsize
            {
                return invalid();
            }
            past = pk[2];
        }
        return yes(product([b,h,qs])*(BigUint::from(past)+BigUint::from(ks))*(BigUint::from(qsize)+BigUint::from(vsize)),"B*q_num_heads*q_sequence_length*total_sequence_length*(qk_head_size+v_head_size), summing the two source-defined dense Attention MatMul contractions; mask, softmax, and backend pruning are outside nominal MACs.");
    }
    if op == "DeformConv" {
        let (Some(x), Some(w), Some(offset), Some(y)) = (
            shape(input),
            shape(weight),
            shape(&inputs[2]),
            shape(output),
        ) else {
            return no("DeformConv requires static X, W, offset, and Y shapes.");
        };
        let rank = x.len();
        let (g, og) = (
            int(a, "group", 1).unwrap_or(0),
            int(a, "offset_group", 1).unwrap_or(0),
        );
        if rank < 3
            || w.len() != rank
            || y.len() != rank
            || offset.len() != rank
            || g <= 0
            || og <= 0
            || x[0] != y[0]
            || x[1] as u128 != w[1] as u128 * g as u128
            || x[1] % og as u64 != 0
            || y[1] != w[0]
            || w[0] % g as u64 != 0
            || w[2..].contains(&0)
        {
            return no(
                "DeformConv X/W/Y ranks, channels, groups, or kernel dimensions are incompatible.",
            );
        }
        let volume = product(w[2..].iter().copied());
        let channels = (&volume * BigUint::from(og as u64) * BigUint::from(rank - 2))
            .to_u64()
            .filter(|n| *n <= SAFE);
        if channels != Some(offset[1]) || offset[0] != x[0] || offset[2..] != y[2..] {
            return no("DeformConv offset shape is incompatible with offset_group, kernel volume, spatial rank, and Y.");
        }
        if !inputs[3].is_null() && shape(&inputs[3]) != Some(vec![w[0]]) {
            return no("DeformConv optional bias shape is incompatible with output channels.");
        }
        if !inputs[4].is_null() {
            let invalid = || {
                no("DeformConv optional mask shape is incompatible with offset_group, kernel volume, and Y.")
            };
            let Some(mask) = shape(&inputs[4]) else {
                return invalid();
            };
            if mask.len() != rank
                || mask[0] != x[0]
                || mask[2..] != y[2..]
                || (&volume * BigUint::from(og as u64))
                    .to_u64()
                    .filter(|n| *n <= SAFE)
                    != Some(mask[1])
            {
                return invalid();
            }
        }
        return yes(product(y.into_iter().chain(w[1..].iter().copied())),"product(Y shape)*product(W shape excluding output-channel axis), counting the source-defined sampled-value/weight contraction only; interpolation arithmetic is excluded from nominal tensor-contraction MACs.");
    }
    if matches!(op, "RNN" | "GRU" | "LSTM") {
        let invalid = || {
            no(format!("{op} MACs require source-compatible X, W, R, hidden_size, direction, layout, and gate dimensions."))
        };
        let (Some(x), Some(w), Some(r)) = (shape(input), shape(weight), shape(&inputs[2])) else {
            return invalid();
        };
        if [&v["input"], &v["weight"], &inputs[2]]
            .iter()
            .any(|t| t["dense"] != true)
        {
            return invalid();
        }
        let gates = match op {
            "LSTM" => 4,
            "GRU" => 3,
            _ => 1,
        };
        let hidden = int(a, "hidden_size", 0).unwrap_or(0);
        let layout = int(a, "layout", 0).unwrap_or(-1);
        let directions = match a["direction"].as_str().unwrap_or("forward") {
            "bidirectional" => 2,
            "forward" | "reverse" => 1,
            _ => 0,
        };
        if x.len() != 3
            || w.len() != 3
            || r.len() != 3
            || ![0, 1].contains(&layout)
            || directions == 0
            || hidden <= 0
            || w[0] != directions
            || r[0] != directions
            || w[1] as u128 != gates as u128 * hidden as u128
            || r[1] != w[1]
            || w[2] != x[2]
            || r[2] != hidden as u64
        {
            return invalid();
        }
        let (sequence, batch) = (x[layout as usize], x[1 - layout as usize]);
        let hidden = hidden as u64;
        if let Some(y) = shape(output) {
            let expected = if layout == 1 {
                vec![batch, sequence, directions, hidden]
            } else {
                vec![sequence, directions, batch, hidden]
            };
            if y != expected {
                return no(format!(
                    "{op} Y shape is incompatible with direction, layout, and hidden_size."
                ));
            }
        }
        return yes(product([sequence,batch,directions,gates,hidden])*(BigUint::from(x[2])+BigUint::from(hidden)),&format!("sequence*batch*directions*{gates}*hidden_size*(input_size+hidden_size), counting source-defined input and recurrent matrix contractions."));
    }
    // The shared equation parser also serves shape and dynamic-cost consumers.
    let Some(rows) = inputs.as_array() else {
        return no("Einsum nominal MACs require every operand shape to be static.");
    };
    let Some(shapes) = rows.iter().map(shape).collect::<Option<Vec<_>>>() else {
        return no("Einsum nominal MACs require every operand shape to be static.");
    };
    if shapes.is_empty() {
        return no("Einsum nominal MACs require every operand shape to be static.");
    }
    let parsed = &v["equation"];
    if parsed["status"] != "assessed" {
        return no(format!(
            "Einsum equation contract is invalid: {}.",
            parsed["reason"].as_str().unwrap_or("invalid")
        ));
    }
    if shapes.len() == 1 {
        return json!({"value":0,"value_decimal":"0","status":"not_applicable","reason":"A one-input Einsum has no binary tensor contraction under the nominal MAC metric."});
    }
    if shapes.len() != 2 {
        return no("Einsum with more than two operands has contraction-order-dependent multiplication work; ONNX does not serialize an evaluation path.");
    }
    let mut dimensions = std::collections::BTreeMap::new();
    for (i, shape) in shapes.iter().enumerate() {
        let Some(labels) = parsed["operands"][i].as_array() else {
            return no("Einsum equation contract is invalid: missing operand labels.");
        };
        if labels.len() != shape.len() {
            return no("Einsum equation contract is invalid: operand rank mismatch.");
        }
        for (axis, label) in labels.iter().enumerate() {
            let Some(label) = label.as_str() else {
                return no("Einsum equation contract is invalid: non-string label.");
            };
            let value = shape[axis];
            let previous = dimensions.get(label).copied();
            if label.starts_with("@ellipsis:") {
                if previous.is_some_and(|p| p != 1 && value != 1 && p != value) {
                    return no("Einsum ellipsis dimensions are not broadcast-compatible.");
                }
                dimensions.insert(label, previous.filter(|p| *p != 1).unwrap_or(value));
            } else if previous.is_some_and(|p| p != value) {
                return no(format!("Einsum label {label} has incompatible dimensions."));
            } else {
                dimensions.insert(label, value);
            }
        }
    }
    let expected = parsed["output"].as_array().and_then(|labels| {
        labels
            .iter()
            .map(|label| dimensions.get(label.as_str()?).copied())
            .collect::<Option<Vec<_>>>()
    });
    if expected.is_none() || shape(output) != expected {
        return no("Einsum output shape is incompatible with the parsed equation and operands.");
    }
    yes(product(dimensions.into_values()),"Product of every unique index-domain extent for a two-input Einsum, equal to one pairwise product/accumulation term per complete Einstein index assignment.")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pair_counts_match_enumeration() {
        for i in 0..8 {
            for k in 1..8 {
                for s in 1..4 {
                    for d in 1..4 {
                        for p in 0..6 {
                            for o in 0..9 {
                                let expected = (0..i)
                                    .flat_map(|x| (0..k).map(move |y| x * s + y * d))
                                    .filter(|v| *v >= p && *v < p + o)
                                    .count() as u128;
                                assert_eq!(pairs(i, k, s, d, p, o), Some(expected));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn exact_large_and_scalar_matmul() {
        let tensor = |s: Vec<u64>| json!({"known":true,"shape":s});
        let result=cost(&json!({"op":"MatMul","attributes":{},"input":tensor(vec![3]),"weight":tensor(vec![3]),"output":tensor(vec![])})).unwrap();
        assert_eq!(result["value_decimal"], "3");
        assert_eq!(
            product([SAFE, SAFE]).to_string(),
            "81129638414606663681390495662081"
        );
    }
}
