//! ONNX operator semantic contracts; source tables are declarative, execution is portable.
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeSet;
pub(crate) const SAFE: i64 = 9_007_199_254_740_991;
pub(crate) fn safe(v: &Value) -> Option<i64> {
    v.as_i64()
        .filter(|n| n.unsigned_abs() <= SAFE as u64)
        .or_else(|| {
            v.as_f64()
                .filter(|n| n.is_finite() && n.fract() == 0. && n.abs() <= SAFE as f64)
                .map(|n| n as i64)
        })
}
pub(crate) fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.),
        _ => true,
    }
}
pub(crate) fn domain(n: &Value) -> &str {
    let s = string(&n["domain"]).trim();
    if s.is_empty() {
        "ai.onnx"
    } else {
        s
    }
}
pub(crate) fn attr<'a>(n: &'a Value, k: &str) -> &'a Value {
    rows(&n["attributes"])
        .iter()
        .find(|r| r[0] == k)
        .map(|r| &r[1])
        .unwrap_or(&Value::Null)
}
pub(crate) fn ai(n: &Value, k: &str, d: i64) -> i64 {
    safe(&attr(n, k)["i"]).unwrap_or(d)
}
pub(crate) fn astr<'a>(n: &'a Value, k: &str, d: &'a str) -> &'a str {
    attr(n, k)["s"].as_str().unwrap_or(d)
}
pub(crate) fn dim(n: i64) -> Value {
    if n.unsigned_abs() > SAFE as u64 {
        return unknown();
    }
    json!({"kind":"value","value":n,"parameter":"","denotation":"","valueFieldCount":1})
}
pub(crate) fn unknown() -> Value {
    json!({"kind":"unknown","value":null,"parameter":"","denotation":"","valueFieldCount":0})
}
pub(crate) fn dv(v: &Value) -> Option<i64> {
    if v["kind"] == "value" {
        safe(&v["value"])
    } else {
        None
    }
}
pub(crate) fn dtype(v: &Value) -> &str {
    let s = string(&v["dtype"]);
    if s == "UNKNOWN" {
        ""
    } else {
        s
    }
}
pub(crate) fn input<'a>(v: &'a Value, i: usize) -> &'a Value {
    let n = &v["node"]["inputs"][i];
    &v["tensors"][string(n)]
}
pub(crate) fn rank(v: &Value) -> Option<usize> {
    if v["shapeDeclared"] == true
        && (string(&v["valueKind"]).is_empty() || v["valueKind"] == "tensor")
    {
        v["shape"].as_array().map(Vec::len)
    } else {
        None
    }
}
pub(crate) fn dimensions(v: &Value) -> Vec<Value> {
    let t = if v.get("typeProto").is_some() {
        &v["typeProto"]
    } else {
        &v["type_proto"]
    };
    let ds = if t["shapeDimensions"].is_array()
        && rows(&t["shapeDimensions"]).len() == rows(&t["shape"]).len()
    {
        rows(&t["shapeDimensions"])
    } else {
        rows(&v["shape"])
    };
    ds.iter().map(normalize_dimension).collect()
}
pub(crate) fn normalize_dimension(v: &Value) -> Value {
    let note = string(&v["denotation"]);
    let mut d = if v["kind"] == "value" {
        safe(&v["value"])
            .or_else(|| {
                v["value"]
                    .as_str()
                    .and_then(|s| s.parse::<i64>().ok())
                    .filter(|n| n.unsigned_abs() <= SAFE as u64)
            })
            .filter(|n| *n >= 0)
            .map(dim)
            .unwrap_or_else(unknown)
    } else if v["kind"] == "symbolic" && !string(&v["parameter"]).is_empty() {
        json!({"kind":"symbolic","value":null,"parameter":v["parameter"],"denotation":"","valueFieldCount":1})
    } else {
        safe(v).filter(|n| *n >= 0).map(dim).unwrap_or_else(unknown)
    };
    d["denotation"] = json!(note);
    d
}

pub(crate) fn tensor_type(dtype: &str, dims: &[Value], ids: &Value) -> Value {
    let dims = dims.iter().map(normalize_dimension).collect::<Vec<_>>();
    json!({"kind":"tensor","dtype":dtype,"shape":dims.iter().map(|d|dv(d).unwrap_or(-1)).collect::<Vec<_>>(),"shapeDeclared":true,"shapeDimensions":dims,"shapeFieldCount":1,"elementTypeId":ids[dtype].as_u64().unwrap_or(0),"elementTypeName":dtype,"elementTypeFieldCount":if dtype=="UNKNOWN"{0}else{1},"valueFieldsPresent":[1],"denotation":""})
}
pub(crate) fn output(v: &Value, index: usize, dims: Vec<Value>, dt: &str) -> Option<Value> {
    let n = &v["node"]["outputs"][index];
    if !truth(n) || dt.is_empty() {
        return None;
    }
    let t = tensor_type(dt, &dims, &v["tensor_type_ids"]);
    Some(json!([n,{"dtype":dt,"shape":t["shape"],"shapeDeclared":true,"typeProto":t}]))
}
pub(crate) fn result(outputs: Vec<Value>) -> Value {
    json!({"outputs":outputs,"reason":""})
}
pub(crate) fn unresolved(s: &str) -> Value {
    json!({"outputs":[],"reason":s})
}
const ATTR_NAMES: [&str; 15] = [
    "",
    "FLOAT",
    "INT",
    "STRING",
    "TENSOR",
    "GRAPH",
    "FLOATS",
    "INTS",
    "STRINGS",
    "TENSORS",
    "GRAPHS",
    "SPARSE_TENSOR",
    "SPARSE_TENSORS",
    "TYPE_PROTO",
    "TYPE_PROTOS",
];
pub(crate) fn attribute(a: &Value, allow_reference: bool) -> Value {
    let declared = safe(&a["type"]).filter(|n| (1..=14).contains(n));
    if truth(&a["refAttrName"]) {
        if allow_reference {
            return json!({"status":if declared.is_some(){"pass"}else{"fail"},"reason":if declared.is_some(){""}else{"attribute_reference_type_missing"},"type":declared,"reference":a["refAttrName"]});
        }
        return json!({"status":"fail","reason":"attribute_reference_not_allowed_in_main_graph","type":null});
    }
    let mut present = vec![];
    let mut add = |i: i64| {
        if !present.contains(&i) {
            present.push(i);
        }
    };
    if a["valueTypesPresent"].is_array() {
        for p in rows(&a["valueTypesPresent"]) {
            if let Some(p) = safe(p).filter(|n| (1..=14).contains(n)) {
                add(p);
            }
        }
    } else {
        for (k, i) in [("tensor", 4), ("graph", 5)] {
            if truth(&a[k]) {
                add(i);
            }
        }
        for (k, i) in [("tensors", 9), ("graphs", 10)] {
            if !rows(&a[k]).is_empty() {
                add(i);
            }
        }
        if truth(&a["iExactDecimal"]) || safe(&a["i"]).is_some() {
            add(2);
        }
        if a["f"].as_f64().is_some_and(f64::is_finite) {
            add(1);
        }
        if !a["s"].is_null() {
            add(3);
        }
        if !rows(&a["intExactDecimals"]).is_empty() || !rows(&a["ints"]).is_empty() {
            add(7);
        }
        for (k, i) in [("floats", 6), ("strings", 8)] {
            if !rows(&a[k]).is_empty() {
                add(i);
            }
        }
    }
    if present.len() > 1 {
        return json!({"status":"fail","reason":format!("attribute_multiple_value_fields:{}",present.iter().map(|p|ATTR_NAMES[*p as usize]).collect::<Vec<_>>().join("+")),"type":declared});
    }
    if let (Some(d), Some(p)) = (declared, present.first()) {
        if d != *p {
            return json!({"status":"fail","reason":format!("attribute_discriminator_payload_mismatch:{}:{}",ATTR_NAMES[d as usize],ATTR_NAMES[*p as usize]),"type":d});
        }
    }
    let t = declared.or(present.first().copied());
    let Some(t) = t else {
        return json!({"status":"unresolved","reason":"attribute_type_not_representable","type":null});
    };
    if present.is_empty() && ![6, 7, 8, 9, 10, 12, 14].contains(&t) {
        return json!({"status":"fail","reason":format!("attribute_value_missing:{}",ATTR_NAMES[t as usize]),"type":t});
    }
    json!({"status":"pass","reason":"","type":t})
}
fn formal(label: &str, values: &[Value], min: usize, max: usize, options: &str) -> Vec<String> {
    let mut reasons = vec![];
    let opts = options.as_bytes();
    let variadic = opts.iter().position(|c| *c == b'V');
    if values.len() < min {
        reasons.push(format!(
            "{label}_count_below_minimum:{}:{min}",
            values.len()
        ));
    }
    if variadic.is_none() && values.len() > max {
        reasons.push(format!(
            "{label}_count_above_maximum:{}:{max}",
            values.len()
        ));
    }
    for (i, v) in values.iter().enumerate() {
        let op = opts
            .get(i)
            .copied()
            .or_else(|| variadic.filter(|p| i >= *p).map(|_| b'V'));
        match op {
            None => reasons.push(format!("{label}_position_not_defined:{i}")),
            Some(b'O') => (),
            _ => {
                if !truth(v) {
                    reasons.push(format!("required_{label}_omitted:{i}"));
                }
            }
        }
    }
    for (i, opt) in opts.iter().enumerate() {
        if *opt == b'R' && !values.get(i).is_some_and(truth) {
            let s = format!("required_{label}_omitted:{i}");
            if !reasons.contains(&s) {
                reasons.push(s);
            }
        }
    }
    reasons
}
pub(crate) fn schema(v: &Value) -> Value {
    let n = &v["node"];
    let op = n["opType"].as_str().unwrap_or("UNKNOWN");
    let opset = &v["imported_opset"];
    let form = rows(&v["forms"])
        .iter()
        .filter(|f| f[0].as_i64().unwrap_or(i64::MAX) <= opset.as_i64().unwrap_or(0))
        .last();
    let mut names = rows(&n["attributes"])
        .iter()
        .map(|a| a[0].clone())
        .collect::<Vec<_>>();
    names.sort_by(|a, b| string(a).encode_utf16().cmp(string(b).encode_utf16()));
    let mut b = json!({"op_name":op,"imported_opset":opset,"schema_since_version":form.map(|f|&f[0]),"input_count":rows(&n["inputs"]).len(),"output_count":rows(&n["outputs"]).len(),"explicit_attributes":names});
    let Some(f) = form else {
        b["status"] = json!("fail");
        b["reason_codes"] = json!(["operator_not_defined_at_imported_opset"]);
        b["detail"] = json!(format!(
            "{op} has no pinned standard-domain schema at opset {opset}."
        ));
        return b;
    };
    let mut reasons = formal(
        "input",
        rows(&n["inputs"]),
        f[1].as_u64().unwrap_or(0) as usize,
        f[2].as_u64().unwrap_or(0) as usize,
        string(&f[3]),
    );
    reasons.extend(formal(
        "output",
        rows(&n["outputs"]),
        f[4].as_u64().unwrap_or(0) as usize,
        f[5].as_u64().unwrap_or(0) as usize,
        string(&f[6]),
    ));
    let mut seen = BTreeSet::new();
    for name in rows(&n["duplicateAttributeNames"]) {
        if seen.insert(string(name)) {
            reasons.push(format!("duplicate_attribute_name:{}", string(name)));
        }
    }
    let definitions = rows(&f[7])
        .iter()
        .map(|a| {
            (
                &v["attribute_names"][a[0].as_u64().unwrap_or(0) as usize],
                &a[1],
                a[2] == 1,
            )
        })
        .collect::<Vec<_>>();
    let mut unresolved = vec![];
    for entry in rows(&n["attributes"]) {
        let name = string(&entry[0]);
        let Some((_, expected, _)) = definitions.iter().find(|d| d.0 == &entry[0]) else {
            reasons.push(format!("attribute_not_defined:{name}"));
            continue;
        };
        let actual = attribute(&entry[1], false);
        if actual["status"] == "fail" {
            reasons.push(format!("{}:{name}", string(&actual["reason"])));
        } else if actual["status"] == "unresolved" {
            unresolved.push(format!("{}:{name}", string(&actual["reason"])));
        } else if actual["type"] != **expected {
            reasons.push(format!(
                "attribute_type_mismatch:{name}:{}:{}",
                ATTR_NAMES[expected.as_u64().unwrap_or(0) as usize],
                ATTR_NAMES[actual["type"].as_u64().unwrap_or(0) as usize]
            ));
        }
    }
    let required = definitions
        .iter()
        .filter(|d| d.2)
        .map(|d| d.0.clone())
        .collect::<Vec<_>>();
    for name in &required {
        if attr(n, string(name)).is_null() {
            reasons.push(format!("required_attribute_missing:{}", string(name)));
        }
    }
    b["schema_input_range"] = json!([
        f[1],
        if string(&f[3]).contains('V') {
            Value::Null
        } else {
            f[2].clone()
        }
    ]);
    b["schema_output_range"] = json!([
        f[4],
        if string(&f[6]).contains('V') {
            Value::Null
        } else {
            f[5].clone()
        }
    ]);
    b["schema_input_options"] = f[3].clone();
    b["schema_output_options"] = f[6].clone();
    b["schema_attribute_count"] = json!(definitions.len());
    b["required_attributes"] = json!(required);
    let state = if !reasons.is_empty() {
        "fail"
    } else if !unresolved.is_empty() {
        "unresolved"
    } else {
        "pass"
    };
    b["status"] = json!(state);
    b["reason_codes"] = json!(if !reasons.is_empty() {
        reasons
    } else {
        unresolved
    });
    b["detail"] = json!(if state == "unresolved" {
        "The NodeProto attribute type cannot be proven from the parsed AttributeProto.".into()
    } else {
        format!(
            "The NodeProto {} the pinned {op}-{} formal schema.",
            if state == "fail" {
                "violates"
            } else {
                "matches"
            },
            f[0]
        )
    });
    b
}
pub(crate) fn run(v: &Value) -> R<Value> {
    match string(&v["operation"]) {
        "conditional_node" => crate::portable_onnx_conditional::run(v),
        "ml" => crate::portable_onnx_ml::run(v),
        "tfidf" => crate::portable_onnx_tfidf::run(v),
        "ml_registry" | "ml_can" | "ml_version" | "ml_schema" => {
            crate::portable_onnx_ml_schema::run(v)
        }
        "loop_support"
        | "loop_clear"
        | "loop_merge"
        | "union_branch"
        | "region_merge_tensor"
        | "region_scalar"
        | "region_condition"
        | "region_axis" => crate::portable_onnx_regions::run(v),
        "node" => crate::portable_onnx_node::run(v),
        "merge_tensor" | "concrete_tensor" => crate::portable_onnx_merge::run(v),
        "container" => crate::portable_onnx_container::run(v),
        "attribute" => Ok(attribute(&v["value"], v["allow_reference"] == true)),
        "schema" => Ok(schema(v)),
        "ort_registry" | "ort_can" | "ort_standard" | "ort_schema" | "ort_infer" => {
            crate::portable_onnx_ort::run(v)
        }
        _ => Err("Unknown ONNX semantic operation.".into()),
    }
}
