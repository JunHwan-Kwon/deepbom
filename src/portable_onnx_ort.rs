//! Source-pinned ORT contrib schema and shape inference, shared by every host.
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::OnceLock;
fn specs() -> &'static Value {
    static SPECS: OnceLock<Value> = OnceLock::new();
    SPECS.get_or_init(|| {
        serde_json::from_str(include_str!("onnx_ort_specs.json")).expect("checked ORT registry")
    })
}
fn standard(n: &Value, opset: &Value) -> bool {
    domain(n) == "ai.onnx"
        && safe(opset).is_some_and(|s| {
            s >= 1
                && (n["opType"] == "SimplifiedLayerNormalization"
                    || n["opType"] == "LayerNormalization" && s < 17)
        })
}
fn family(n: &Value) -> &'static str {
    let op = string(&n["opType"]);
    if domain(n) != "com.microsoft" {
        return "extension";
    }
    if specs().get(op).is_some() {
        "transformer"
    } else if op == "FusedMatMul" {
        "matmul"
    } else if op == "DynamicQuantizeLSTM" {
        "recurrent"
    } else {
        "extension"
    }
}
fn can(n: &Value, opset: &Value, which: &str) -> bool {
    let actual = family(n);
    if which != "extension" {
        return actual == which;
    }
    standard(n, opset)
        || domain(n) == "ai.onnx" && n["opType"] == "SimplifiedLayerNormalization"
        || domain(n) == "com.microsoft"
            && (actual != "extension"
                || ["MatMulNBits", "MatMulBnb4"].contains(&string(&n["opType"])))
}
fn definition(n: &Value) -> Value {
    match family(n) {
        "transformer" => specs()[string(&n["opType"])].clone(),
        "matmul" => {
            json!({"inputMinimum":2,"inputMaximum":2,"outputMinimum":1,"outputMaximum":1,"attributes":{"alpha":1,"transA":2,"transB":2,"transBatchA":2,"transBatchB":2},"requiredAttributes":[],"requiredInputs":[0,1]})
        }
        "recurrent" => {
            json!({"inputMinimum":12,"inputMaximum":12,"outputMinimum":0,"outputMaximum":3,"attributes":{"direction":3,"hidden_size":2,"activation_alpha":6,"activation_beta":6,"clip":1,"activations":8,"input_forget":2},"requiredAttributes":[],"requiredInputs":[0,1,2,8,9,10,11]})
        }
        _ => match string(&n["opType"]) {
            "MatMulNBits" => {
                json!({"inputMinimum":3,"inputMaximum":6,"outputMinimum":1,"outputMaximum":1,"attributes":{"K":2,"N":2,"bits":2,"block_size":2,"accuracy_level":2},"requiredAttributes":["K","N","block_size"],"requiredInputs":[0,1,2]})
            }
            "MatMulBnb4" => {
                json!({"inputMinimum":3,"inputMaximum":3,"outputMinimum":1,"outputMaximum":1,"attributes":{"K":2,"N":2,"block_size":2,"quant_type":2,"training_mode":2,"transB":2},"requiredAttributes":["K","N","block_size","quant_type"],"requiredInputs":[0,1,2]})
            }
            op => {
                json!({"inputMinimum":2,"inputMaximum":if op=="SimplifiedLayerNormalization"{2}else{3},"outputMinimum":1,"outputMaximum":if op=="SimplifiedLayerNormalization"{2}else{3},"attributes":{"axis":2,"epsilon":1,"stash_type":2},"requiredAttributes":[],"requiredInputs":[0,1]})
            }
        },
    }
}
fn value_reasons(n: &Value) -> Vec<String> {
    let mut r = vec![];
    let op = string(&n["opType"]);
    match op {
        "FusedMatMul" => {
            for name in ["transA", "transB", "transBatchA", "transBatchB"] {
                if ![0, 1].contains(&ai(n, name, 0)) {
                    r.push(format!("attribute_value_invalid:{name}:0_or_1"));
                }
            }
        }
        "DynamicQuantizeLSTM" => {
            if !["forward", "reverse", "bidirectional"].contains(&astr(n, "direction", "forward")) {
                r.push("attribute_value_invalid:direction".into());
            }
            if !attr(n, "hidden_size").is_null() && ai(n, "hidden_size", -1) <= 0 {
                r.push("attribute_value_not_positive:hidden_size".into());
            }
            if ![0, 1].contains(&ai(n, "input_forget", 0)) {
                r.push("attribute_value_invalid:input_forget:0_or_1".into());
            }
        }
        "MatMulNBits" | "MatMulBnb4" => {
            for name in ["K", "N"] {
                if !attr(n, name).is_null() && ai(n, name, -1) <= 0 {
                    r.push(format!("attribute_value_not_positive:{name}"));
                }
            }
            if !attr(n, "block_size").is_null() {
                let x = ai(n, "block_size", -1);
                if x < 16 || (x as u64).count_ones() != 1 {
                    r.push("attribute_value_invalid:block_size:power_of_two_at_least_16".into());
                }
            }
            if op == "MatMulNBits" {
                if !(2..=8).contains(&ai(n, "bits", 4)) {
                    r.push("attribute_value_out_of_range:bits:2:8".into());
                }
                if !(0..=4).contains(&ai(n, "accuracy_level", 0)) {
                    r.push("attribute_value_out_of_range:accuracy_level:0:4".into());
                }
            } else {
                if ![0, 1].contains(&ai(n, "quant_type", -1)) {
                    r.push("attribute_value_invalid:quant_type:0_or_1".into());
                }
                for name in ["training_mode", "transB"] {
                    if ![0, 1].contains(&ai(n, name, if name == "transB" { 1 } else { 0 })) {
                        r.push(format!("attribute_value_invalid:{name}:0_or_1"));
                    }
                }
            }
        }
        "MultiHeadAttention" | "GroupQueryAttention" => {
            if ai(n, "num_heads", 0) <= 0 {
                r.push("attribute_value_not_positive:num_heads".into());
            }
            if op == "MultiHeadAttention" {
                if ![0, 1].contains(&ai(n, "unidirectional", 0)) {
                    r.push("attribute_value_invalid:unidirectional:0_or_1".into());
                }
            } else {
                if ai(n, "kv_num_heads", 0) <= 0 {
                    r.push("attribute_value_not_positive:kv_num_heads".into());
                }
                for name in ["do_rotary", "rotary_interleaved"] {
                    if ![0, 1].contains(&ai(n, name, 0)) {
                        r.push(format!("attribute_value_invalid:{name}:0_or_1"));
                    }
                }
                if ![0, 1, 2].contains(&ai(n, "qk_output", 0)) {
                    r.push("attribute_value_invalid:qk_output:0_1_or_2".into());
                }
                if ![4, 8].contains(&ai(n, "kv_cache_bit_width", 8)) {
                    r.push("attribute_value_invalid:kv_cache_bit_width:4_or_8".into());
                }
                for name in ["k_quant_type", "v_quant_type"] {
                    if !["NONE", "PER_TENSOR", "PER_CHANNEL"].contains(&astr(n, name, "NONE")) {
                        r.push(format!(
                            "attribute_value_invalid:{name}:NONE_PER_TENSOR_OR_PER_CHANNEL"
                        ));
                    }
                }
            }
        }
        _ => (),
    }
    r
}
fn schema(v: &Value) -> Value {
    let n = &v["node"];
    let op = n["opType"].as_str().unwrap_or("UNKNOWN");
    let which = v["family"].as_str().unwrap_or("extension");
    let actual = if which == "extension" {
        family(n)
    } else {
        which
    };
    let pinned = standard(n, &v["imported_opset"]);
    let source = match actual {
        "transformer" => "ort_contrib_transformer_schema",
        "matmul" => "ort_contrib_fused_matmul_schema",
        "recurrent" => "ort_contrib_quantized_recurrent_schema",
        _ => {
            if pinned {
                "ort_contrib_standard_domain_extension"
            } else {
                "ort_contrib_schema"
            }
        }
    };
    let mut names = rows(&n["attributes"])
        .iter()
        .map(|r| r[0].clone())
        .collect::<Vec<_>>();
    names.sort_by(|a, b| string(a).encode_utf16().cmp(string(b).encode_utf16()));
    let mut b = json!({"op_name":op,"imported_opset":v["imported_opset"],"schema_since_version":1,"schema_source":source,"input_count":rows(&n["inputs"]).len(),"output_count":rows(&n["outputs"]).len(),"explicit_attributes":names});
    if !can(n, &v["imported_opset"], which) {
        let (reason, detail) = match which {
            "matmul" => (
                "ort_fused_matmul_shape_rule_not_registered",
                "The node is not the pinned com.microsoft:FusedMatMul-1 schema.".into(),
            ),
            "recurrent" => (
                "ort_recurrent_shape_rule_not_registered",
                "The node is not the pinned com.microsoft:DynamicQuantizeLSTM-1 schema.".into(),
            ),
            "transformer" => (
                "ort_transformer_shape_rule_not_registered",
                format!(
                    "{}:{op} is not a pinned ORT transformer shape rule.",
                    domain(n)
                ),
            ),
            _ => (
                "ort_extension_shape_rule_not_registered",
                format!(
                    "{}:{op} is not a pinned local ORT shape rule at opset {}.",
                    domain(n),
                    v["imported_opset"]
                ),
            ),
        };
        b["status"] = json!("fail");
        b["reason_codes"] = json!([reason]);
        b["detail"] = json!(detail);
        return b;
    }
    let d = definition(n);
    let mut reasons = vec![];
    let mut unresolved = vec![];
    for (label, min, max) in [
        (
            "input",
            d["inputMinimum"].as_u64().unwrap() as usize,
            d["inputMaximum"].as_u64().unwrap() as usize,
        ),
        (
            "output",
            d["outputMinimum"].as_u64().unwrap() as usize,
            d["outputMaximum"].as_u64().unwrap() as usize,
        ),
    ] {
        let len = rows(&n[format!("{label}s")]).len();
        if actual == "matmul" || actual == "recurrent" && label == "input" {
            if len != min {
                reasons.push(format!("{label}_count_mismatch:{len}:{min}"));
            }
        } else {
            if len < min {
                reasons.push(format!("{label}_count_below_minimum:{len}:{min}"));
            }
            if len > max {
                reasons.push(format!("{label}_count_above_maximum:{len}:{max}"));
            }
        }
    }
    for index in rows(&d["requiredInputs"]) {
        if !truth(&n["inputs"][index.as_u64().unwrap() as usize]) {
            reasons.push(format!("required_input_omitted:{index}"));
        }
    }
    for i in 0..d["outputMinimum"].as_u64().unwrap() as usize {
        if !truth(&n["outputs"][i]) {
            reasons.push(format!("required_output_omitted:{i}"));
        }
    }
    let mut seen = BTreeSet::new();
    for name in rows(&n["duplicateAttributeNames"]) {
        if seen.insert(string(name)) {
            reasons.push(format!("duplicate_attribute_name:{}", string(name)));
        }
    }
    for a in rows(&n["attributes"]) {
        let name = string(&a[0]);
        let expected = &d["attributes"][name];
        if expected.is_null() {
            if !pinned {
                reasons.push(format!("attribute_not_defined:{name}"));
            }
            continue;
        }
        let a = attribute(&a[1], false);
        if a["status"] == "fail" {
            reasons.push(format!("{}:{name}", string(&a["reason"])));
        } else if a["status"] == "unresolved" {
            unresolved.push(format!("{}:{name}", string(&a["reason"])));
        } else if a["type"] != *expected {
            reasons.push(format!(
                "attribute_type_mismatch:{name}:{expected}:{}",
                a["type"]
            ));
        }
    }
    for name in rows(&d["requiredAttributes"]) {
        if attr(n, string(name)).is_null() {
            reasons.push(format!("required_attribute_missing:{}", string(name)));
        }
    }
    reasons.extend(value_reasons(n));
    b["schema_input_range"] = json!([d["inputMinimum"], d["inputMaximum"]]);
    b["schema_output_range"] = json!([d["outputMinimum"], d["outputMaximum"]]);
    b["schema_attribute_count"] = json!(d["attributes"].as_object().unwrap().len());
    b["required_attributes"] = d["requiredAttributes"].clone();
    if actual == "extension" {
        b["unchecked_attributes_allowed"] = json!(pinned);
    }
    let status = if !reasons.is_empty() {
        "fail"
    } else if !unresolved.is_empty() {
        "unresolved"
    } else {
        "pass"
    };
    b["status"] = json!(status);
    b["reason_codes"] = json!(if !reasons.is_empty() {
        reasons
    } else {
        unresolved
    });
    b["detail"] = json!(match status {
        "fail" => format!("The NodeProto violates the pinned ORT {op}-1 contrib schema."),
        "unresolved" =>
            "The NodeProto attribute type cannot be proven from the parsed AttributeProto.".into(),
        _ => format!(
            "The NodeProto matches the pinned ORT {}:{op}-1 contrib schema.",
            domain(n)
        ),
    });
    b
}
fn one(v: &Value, shape: Vec<Value>, dt: &str) -> Value {
    result(output(v, 0, shape, dt).into_iter().collect())
}
fn transform(raw: &[Value], matrix: bool, batch: bool) -> Vec<Value> {
    if raw.len() == 1 {
        return raw.to_vec();
    }
    let r = raw.len();
    let mut s = raw[if batch { 1 } else { 0 }..if batch { r - 1 } else { r - 2 }].to_vec();
    s.push(
        raw[if matrix {
            r - 1
        } else if batch {
            0
        } else {
            r - 2
        }]
        .clone(),
    );
    s.push(
        raw[if matrix {
            if batch {
                0
            } else {
                r - 2
            }
        } else {
            r - 1
        }]
        .clone(),
    );
    s
}
fn fused(v: &Value) -> R<Value> {
    let n = &v["node"];
    let (a, b) = (input(v, 0), input(v, 1));
    if rank(a).is_none() || rank(b).is_none() {
        return Ok(unresolved("ort_fused_matmul_input_rank_unknown"));
    }
    let (a, b) = (dimensions(a), dimensions(b));
    if a.is_empty() || b.is_empty() {
        return Ok(unresolved("ort_fused_matmul_input_rank_zero"));
    }
    let (av, bv) = (a.len() == 1, b.len() == 1);
    let mut a = transform(&a, ai(n, "transA", 0) != 0, ai(n, "transBatchA", 0) != 0);
    let mut b = transform(&b, ai(n, "transB", 0) != 0, ai(n, "transBatchB", 0) != 0);
    if av {
        a.insert(0, dim(1));
    }
    if bv {
        b.push(dim(1));
    }
    if matches!((dv(a.last().unwrap()),dv(&b[b.len()-2])),(Some(a),Some(b)) if a!=b) {
        return Ok(unresolved(
            "ort_fused_matmul_contraction_dimension_mismatch",
        ));
    }
    let prefix = crate::portable_shape::run(
        &json!({"operation":"broadcast","shapes":[&a[..a.len()-2],&b[..b.len()-2]]}),
    )?;
    let Some(prefix) = prefix.as_array() else {
        return Ok(unresolved("ort_fused_matmul_batch_broadcast_incompatible"));
    };
    let mut s = prefix.clone();
    s.push(a[a.len() - 2].clone());
    s.push(b.last().unwrap().clone());
    if av {
        s.remove(s.len() - 2);
    }
    if bv {
        s.pop();
    }
    let dt = dtype(input(v, 0));
    if dt.is_empty() {
        return Ok(unresolved("ort_fused_matmul_input_dtype_unknown"));
    }
    Ok(one(v, s, dt))
}
fn recurrent(v: &Value) -> Value {
    let n = &v["node"];
    let x = input(v, 0);
    if rank(x) != Some(3) {
        return unresolved("ort_dynamic_quantize_lstm_input_rank_unknown_or_invalid");
    }
    let dt = dtype(x);
    if dt.is_empty() {
        return unresolved("ort_dynamic_quantize_lstm_input_dtype_unknown");
    }
    let shape = dimensions(x);
    let direction = match astr(n, "direction", "forward") {
        "bidirectional" => dim(2),
        "forward" | "reverse" => dim(1),
        _ => unknown(),
    };
    let hidden = ai(n, "hidden_size", -1);
    let hidden = if hidden > 0 { dim(hidden) } else { unknown() };
    let short = vec![direction.clone(), shape[1].clone(), hidden.clone()];
    let mut out = vec![];
    for (i, s) in [
        (
            0,
            vec![shape[0].clone(), direction, shape[1].clone(), hidden],
        ),
        (1, short.clone()),
        (2, short),
    ] {
        out.extend(output(v, i, s, dt));
    }
    if out.is_empty() {
        unresolved("ort_dynamic_quantize_lstm_output_omitted")
    } else {
        result(out)
    }
}
fn quantized(v: &Value) -> Value {
    let n = &v["node"];
    let x = input(v, 0);
    if rank(x).is_none() {
        return unresolved("ort_quantized_matmul_input_rank_unknown");
    }
    let mut s = dimensions(x);
    if s.is_empty() {
        return unresolved("ort_quantized_matmul_input_rank_zero");
    }
    let (k, b) = (ai(n, "K", -1), ai(n, "N", -1));
    if k <= 0 || b <= 0 {
        return unresolved("ort_quantized_matmul_feature_attributes_invalid");
    }
    let trans = n["opType"] == "MatMulNBits" || ai(n, "transB", 1) != 0;
    let (expected, feature) = if trans { (k, b) } else { (b, k) };
    if dv(s.last().unwrap()).is_some_and(|n| n != expected) {
        return unresolved("ort_quantized_matmul_input_feature_mismatch");
    }
    if n["opType"] == "MatMulNBits" && truth(&n["inputs"][5]) {
        let bias = input(v, 5);
        if rank(bias) != Some(1) {
            return unresolved("ort_matmul_nbits_bias_shape_unknown_or_rank_invalid");
        }
        if dv(&dimensions(bias)[0]) != Some(b) {
            return unresolved("ort_matmul_nbits_bias_shape_mismatch");
        }
    }
    let dt = dtype(x);
    if dt.is_empty() {
        return unresolved("ort_quantized_matmul_input_dtype_unknown");
    }
    *s.last_mut().unwrap() = dim(feature);
    one(v, s, dt)
}
fn normalization(v: &Value) -> Value {
    let n = &v["node"];
    let x = input(v, 0);
    let Some(rank) = rank(x) else {
        return unresolved("ort_layer_normalization_input_rank_unknown");
    };
    let mut axis = ai(n, "axis", -1);
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis >= rank as i64 {
        return unresolved("ort_layer_normalization_axis_out_of_range");
    }
    let dt = dtype(input(v, 1));
    if dt.is_empty() {
        return unresolved("ort_layer_normalization_scale_dtype_unknown");
    }
    let stash = string(&v["stash_dtype"]);
    let s = dimensions(x);
    let mut out = vec![];
    out.extend(output(v, 0, s.clone(), dt));
    if n["opType"] == "SimplifiedLayerNormalization" {
        let mut s = s;
        s[axis as usize] = dim(1);
        out.extend(output(v, 1, s, stash));
    } else {
        let s = s
            .into_iter()
            .enumerate()
            .map(|(i, d)| if i < (axis as usize) { d } else { dim(1) })
            .collect::<Vec<_>>();
        out.extend(output(v, 1, s.clone(), stash));
        out.extend(output(v, 2, s, stash));
    }
    result(out)
}
fn mul(a: &Value, b: &Value) -> Value {
    dv(a)
        .zip(dv(b))
        .and_then(|(a, b)| a.checked_mul(b))
        .filter(|n| *n <= SAFE)
        .map(dim)
        .unwrap_or_else(unknown)
}
fn transformer(v: &Value) -> Value {
    let n = &v["node"];
    let op = string(&n["opType"]);
    let q = input(v, 0);
    let dt = dtype(q);
    if ["FastGelu", "RotaryEmbedding"].contains(&op) {
        if rank(q).is_none() {
            return unresolved("ort_transformer_input_rank_unknown");
        }
        if dt.is_empty() {
            return unresolved("ort_transformer_input_dtype_unknown");
        }
        return one(v, dimensions(q), dt);
    }
    if ["SkipLayerNormalization", "SkipSimplifiedLayerNormalization"].contains(&op) {
        if rank(q).is_none_or(|n| n == 0) {
            return unresolved("ort_skip_layer_normalization_input_rank_unknown_or_zero");
        }
        if dt.is_empty() {
            return unresolved("ort_skip_layer_normalization_input_dtype_unknown");
        }
        let s = dimensions(q);
        let mut stash = s.clone();
        *stash.last_mut().unwrap() = dim(1);
        let mut out = vec![];
        for (i, s, dt) in [
            (0, s.clone(), dt),
            (1, stash.clone(), "FLOAT32"),
            (2, stash, "FLOAT32"),
            (3, s, dt),
        ] {
            out.extend(output(v, i, s, dt));
        }
        return result(out);
    }
    if op == "MultiHeadAttention" {
        if rank(q).is_none() {
            return unresolved("ort_multi_head_attention_query_rank_unknown");
        }
        let qs = dimensions(q);
        if ![3, 5].contains(&qs.len()) {
            return unresolved("ort_multi_head_attention_query_rank_invalid");
        }
        if dt.is_empty() {
            return unresolved("ort_multi_head_attention_query_dtype_unknown");
        }
        let mut vl = None;
        let shape = if qs.len() == 5 {
            Some(vec![qs[0].clone(), qs[1].clone(), mul(&qs[2], &qs[4])])
        } else {
            let (value, key) = (input(v, 2), input(v, 1));
            if let Some(r) = rank(value) {
                let vs = dimensions(value);
                if ![3, 4].contains(&r) {
                    return unresolved("ort_multi_head_attention_value_rank_invalid");
                }
                if r == 3 {
                    vl = dv(&vs[1]);
                }
                Some(vec![
                    qs[0].clone(),
                    qs[1].clone(),
                    if r == 3 {
                        vs[2].clone()
                    } else {
                        mul(&vs[1], &vs[3])
                    },
                ])
            } else if rank(key) == Some(5) {
                Some(qs.clone())
            } else {
                None
            }
        };
        let Some(shape) = shape else {
            return unresolved("ort_multi_head_attention_output_shape_inputs_unknown");
        };
        let mut out = output(v, 0, shape, dt).into_iter().collect::<Vec<_>>();
        let (pk, pv) = (input(v, 6), input(v, 7));
        let sharing = rank(pk).is_some() && rank(input(v, 8)).is_some();
        if !sharing && vl.is_some_and(|x| x > 0) && rank(pk) == Some(4) {
            let mut ps = dimensions(pk);
            if let Some(length) = dv(&ps[2])
                .and_then(|p| p.checked_add(vl.unwrap()))
                .filter(|n| *n <= SAFE)
            {
                ps[2] = dim(length);
                out.extend(output(v, 1, ps.clone(), dtype(pk)));
                out.extend(output(v, 2, ps, dtype(pv)));
            }
        }
        return result(out);
    }
    if op != "GroupQueryAttention" {
        return unresolved("ort_transformer_shape_rule_not_registered");
    }
    if rank(q) != Some(3) {
        return unresolved("ort_group_query_attention_query_rank_unknown_or_invalid");
    }
    let qs = dimensions(q);
    if dt.is_empty() {
        return unresolved("ort_group_query_attention_query_dtype_unknown");
    }
    let (h, kv) = (ai(n, "num_heads", 0), ai(n, "kv_num_heads", 0));
    if h <= 0 || kv <= 0 {
        return unresolved("ort_group_query_attention_head_attributes_invalid");
    }
    let value = input(v, 2);
    let (shape, length, head) = if truth(&n["inputs"][2]) {
        (
            qs.clone(),
            if rank(value) == Some(3) {
                dv(&dimensions(value)[1])
            } else {
                None
            },
            dv(&qs[2]).filter(|x| x % h == 0).map(|x| x / h),
        )
    } else {
        let divisor = h + 2 * kv;
        let Some(hidden) = dv(&qs[2]).filter(|x| x % divisor == 0) else {
            return unresolved("ort_group_query_attention_packed_hidden_size_unresolved");
        };
        let head = hidden / divisor;
        (
            vec![qs[0].clone(), qs[1].clone(), dim(head * h)],
            dv(&qs[1]),
            Some(head),
        )
    };
    let mut out = output(v, 0, shape, dt).into_iter().collect::<Vec<_>>();
    let total = if rows(&input(v, 6)["staticValues"]).len() == 1 {
        safe(&input(v, 6)["staticValues"][0])
    } else {
        None
    };
    let (pk, pv) = (input(v, 3), input(v, 4));
    if rank(pk) == Some(4) {
        let mut ps = dimensions(pk);
        let past = dv(&ps[2]);
        if let (Some(total), Some(past)) = (total.filter(|n| *n > 0), past) {
            ps[2] = dim(total.max(past));
        } else if past == Some(0) && truth(&n["inputs"][6]) {
            ps[2] = unknown();
        }
        out.extend(output(v, 1, ps.clone(), dtype(pk)));
        out.extend(output(v, 2, ps, dtype(pv)));
        if ai(n, "qk_output", 0) != 0
            && total.is_some_and(|n| n > 0)
            && qs[..2].iter().all(|d| dv(d).is_some())
        {
            out.extend(output(
                v,
                3,
                vec![qs[0].clone(), dim(h), qs[1].clone(), dim(total.unwrap())],
                dt,
            ));
        }
    } else if let Some(head) = head {
        let seq = total
            .filter(|n| *n > 0)
            .or(length.filter(|n| *n > 0))
            .map(dim)
            .unwrap_or_else(|| qs[1].clone());
        let ps = vec![qs[0].clone(), dim(kv), seq, dim(head)];
        out.extend(output(v, 1, ps.clone(), dt));
        out.extend(output(v, 2, ps, dt));
    }
    result(out)
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    match string(&v["operation"]) {
        "ort_registry" => Ok(json!(specs()
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>())),
        "ort_can" => Ok(json!(can(
            n,
            &v["imported_opset"],
            v["family"].as_str().unwrap_or("extension")
        ))),
        "ort_standard" => Ok(json!(standard(n, &v["imported_opset"]))),
        "ort_schema" => Ok(schema(v)),
        "ort_infer" => {
            let which = v["family"].as_str().unwrap_or("extension");
            let which = if which == "extension" {
                family(n)
            } else {
                which
            };
            Ok(match which {
                "matmul" => return fused(v),
                "recurrent" => recurrent(v),
                "transformer" => transformer(v),
                _ => {
                    if ["MatMulNBits", "MatMulBnb4"].contains(&string(&n["opType"])) {
                        quantized(v)
                    } else {
                        normalization(v)
                    }
                }
            })
        }
        _ => Err("Unknown ORT semantic operation.".into()),
    }
}
