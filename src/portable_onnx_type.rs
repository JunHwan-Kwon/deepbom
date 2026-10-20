//! ONNX TypeProto semantics. Hosts decode protobuf and preserve source evidence;
//! this owner validates and joins/unifies the decoded type tree. Enum identities
//! arrive from the pinned ONNX registry, not a second hand-maintained table.
use serde_json::{json, Value};
use std::collections::BTreeSet;
const SAFE: f64 = 9_007_199_254_740_991.0;
fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or("")
}
fn kind(v: &Value) -> &str {
    v["kind"].as_str().unwrap_or("undefined")
}
fn number(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or(0.0)
}
fn valid_int(v: &Value) -> bool {
    v.as_f64()
        .is_some_and(|n| n >= 0.0 && n <= SAFE && n.fract() == 0.0)
}
fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn child(v: &Value) -> Option<&Value> {
    match kind(v) {
        "sequence" | "optional" => v.get("elementType").filter(|v| !v.is_null()),
        "map" => v.get("valueType").filter(|v| !v.is_null()),
        _ => None,
    }
}
fn known(v: &Value) -> bool {
    match kind(v) {
        "tensor" | "sparse_tensor" => {
            !s(v, "dtype").is_empty()
                && s(v, "dtype") != "UNKNOWN"
                && !s(v, "dtype").starts_with("TYPE_")
        }
        "sequence" | "optional" => child(v).is_some_and(known),
        "map" => map_key(s(v, "keyTypeName")) && child(v).is_some_and(known),
        "opaque" => !s(v, "domain").is_empty() || !s(v, "name").is_empty(),
        _ => false,
    }
}
fn map_key(s: &str) -> bool {
    [
        "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "UINT32", "UINT64",
    ]
    .contains(&s)
}
#[derive(Default)]
struct Assessment {
    reasons: BTreeSet<String>,
    nodes: u64,
    depth: u64,
    bytes: usize,
    symbolic: u64,
    unknown: u64,
}
fn assess(v: &Value, depth: u64, a: &mut Assessment) -> Result<(), String> {
    if depth > 64 {
        return Err("ONNX type depth exceeds portable contract budget".into());
    }
    a.nodes += 1;
    a.depth = a.depth.max(depth);
    a.bytes += s(v, "denotation").len() + s(v, "domain").len() + s(v, "name").len();
    let fields = array(&v["valueFieldsPresent"]);
    if fields.len() != 1 {
        a.reasons.insert(
            if fields.is_empty() {
                "type_proto_value_missing"
            } else {
                "type_proto_oneof_has_multiple_values"
            }
            .into(),
        );
    }
    let k = kind(v);
    match k {
        "tensor" | "sparse_tensor" => {
            let id = &v["elementTypeId"];
            let name = s(v, "elementTypeName");
            if number(&v["elementTypeFieldCount"]) != 1.0
                || !valid_int(id)
                || number(id) <= 0.0
                || name.is_empty()
                || name == "UNDEFINED"
                || name.starts_with("TYPE_")
            {
                a.reasons
                    .insert(format!("{k}_element_type_missing_invalid_or_duplicate"));
            }
            if number(&v["shapeFieldCount"]) > 1.0 {
                a.reasons.insert(format!("{k}_shape_field_duplicate"));
            }
            for d in array(&v["shapeDimensions"]) {
                a.bytes += s(d, "parameter").len() + s(d, "denotation").len();
                if number(&d["valueFieldCount"]) > 1.0 {
                    a.reasons
                        .insert("tensor_dimension_oneof_has_multiple_values".into());
                }
                match s(d, "kind") {
                    "value" => {
                        if !valid_int(&d["value"]) {
                            a.reasons.insert("tensor_dimension_value_invalid".into());
                        }
                    }
                    "symbolic" => {
                        a.symbolic += 1;
                        if s(d, "parameter").is_empty() {
                            a.reasons
                                .insert("tensor_dimension_symbolic_name_empty".into());
                        }
                    }
                    "unknown" => a.unknown += 1,
                    _ => (),
                }
            }
        }
        "sequence" | "optional" => {
            if number(&v["childFieldCount"]) != 1.0 || child(v).is_none() {
                a.reasons
                    .insert(format!("{k}_element_type_missing_or_duplicate"));
            }
        }
        "map" => {
            if number(&v["keyFieldCount"]) != 1.0 || !map_key(s(v, "keyTypeName")) {
                a.reasons.insert("map_key_type_invalid_or_duplicate".into());
            }
            if number(&v["childFieldCount"]) != 1.0 || child(v).is_none() {
                a.reasons
                    .insert("map_value_type_missing_or_duplicate".into());
            }
        }
        "opaque" => (),
        _ => {
            a.reasons
                .insert("type_proto_kind_unsupported_or_missing".into());
        }
    }
    if let Some(c) = child(v) {
        assess(c, depth + 1, a)?;
    }
    if a.nodes > 1_000_000 || a.bytes > 4_194_304 {
        return Err("ONNX type exceeds portable contract budget".into());
    }
    Ok(())
}
fn unknown() -> Value {
    json!({"kind":"unknown","value":null,"parameter":"","denotation":"","valueFieldCount":0})
}
fn dtype(v: &Value) -> &str {
    if !s(v, "dtype").is_empty() {
        s(v, "dtype")
    } else if !s(v, "elementTypeName").is_empty() {
        s(v, "elementTypeName")
    } else {
        "UNKNOWN"
    }
}
fn tensor(dtype: &str, k: &str, registry: &Value) -> Value {
    json!({"kind":k,"dtype":dtype,"shape":[],"shapeDeclared":false,"shapeDimensions":[],"shapeFieldCount":0,"elementTypeId":registry[dtype].as_u64().unwrap_or(0),"elementTypeName":dtype,"elementTypeFieldCount":if dtype=="UNKNOWN"{0}else{1},"valueFieldsPresent":[if k=="sparse_tensor"{8}else{1}],"denotation":""})
}
fn set_dims(v: &mut Value, dims: Vec<Value>) {
    v["shapeDeclared"] = json!(true);
    v["shapeFieldCount"] = json!(1);
    v["shape"] = json!(dims
        .iter()
        .map(|d| if d["kind"] == "value" {
            d["value"].clone()
        } else {
            json!(-1)
        })
        .collect::<Vec<_>>());
    v["shapeDimensions"] = json!(dims);
}
fn merge_dim(a: &Value, b: &Value, union: bool) -> Option<Value> {
    if union {
        return Some(
            if a["kind"] == "value" && b["kind"] == "value" && a["value"] == b["value"]
                || a["kind"] == "symbolic"
                    && b["kind"] == "symbolic"
                    && a["parameter"] == b["parameter"]
            {
                a.clone()
            } else {
                unknown()
            },
        );
    }
    if a["kind"] == "value" && b["kind"] == "value" {
        return if a["value"] == b["value"] {
            Some(a.clone())
        } else {
            None
        };
    }
    Some(if b["kind"] == "value" {
        b.clone()
    } else if a["kind"] == "value" || a["kind"] == "symbolic" {
        a.clone()
    } else if b["kind"] == "symbolic" {
        b.clone()
    } else {
        unknown()
    })
}
fn merge(
    a: &Value,
    b: &Value,
    union: bool,
    registry: &Value,
    depth: usize,
) -> Result<Value, String> {
    if depth > 64 {
        return Err("type_proto_depth_exceeded".into());
    }
    let k = kind(a);
    if k != kind(b) {
        return Err(format!("type_kind_mismatch:{k}:{}", kind(b)));
    }
    match k {
        "tensor" | "sparse_tensor" => {
            let x = dtype(a);
            let y = dtype(b);
            if x != y && (union || x != "UNKNOWN" && y != "UNKNOWN") {
                return Err(format!("tensor_element_type_mismatch:{x}:{y}"));
            }
            let mut out = tensor(if !union && x == "UNKNOWN" { y } else { x }, k, registry);
            let da = array(&a["shapeDimensions"]);
            let db = array(&b["shapeDimensions"]);
            let ad = a["shapeDeclared"] == true;
            let bd = b["shapeDeclared"] == true;
            if ad && bd {
                if da.len() != db.len() {
                    if !union {
                        return Err("tensor_rank_mismatch".into());
                    }
                } else {
                    let mut dims = vec![];
                    for (i, (a, b)) in da.iter().zip(db).enumerate() {
                        dims.push(
                            merge_dim(a, b, union)
                                .ok_or_else(|| format!("tensor_dimension_mismatch:{i}"))?,
                        );
                    }
                    set_dims(&mut out, dims);
                }
            } else if !union && (ad || bd) {
                set_dims(&mut out, if ad { da } else { db }.to_vec());
            }
            Ok(out)
        }
        "sequence" | "optional" => {
            let c = merge(
                child(a).unwrap_or(&Value::Null),
                child(b).unwrap_or(&Value::Null),
                union,
                registry,
                depth + 1,
            )?;
            Ok(
                json!({"kind":k,"valueFieldsPresent":[if k=="sequence"{4}else{9}],"elementType":c,"childFieldCount":1,"dtype":"UNKNOWN","shape":[],"shapeDeclared":false,"shapeDimensions":[],"denotation":""}),
            )
        }
        "map" => {
            if a["keyTypeName"] != b["keyTypeName"] {
                return Err("map_key_type_mismatch".into());
            }
            let c = merge(
                child(a).unwrap_or(&Value::Null),
                child(b).unwrap_or(&Value::Null),
                union,
                registry,
                depth + 1,
            )?;
            let mut out = a.clone();
            out["valueType"] = c;
            Ok(out)
        }
        "opaque" => {
            if a["domain"] == b["domain"] && a["name"] == b["name"] {
                Ok(a.clone())
            } else {
                Err("opaque_identity_mismatch".into())
            }
        }
        _ => Err(format!("unsupported_type_kind:{k}")),
    }
}
pub(crate) fn run(v: &Value) -> Result<Value, String> {
    match s(v, "operation") {
        "assess" => {
            let mut a = Assessment::default();
            assess(&v["value"], v["depth"].as_u64().unwrap_or(1), &mut a)?;
            Ok(
                json!({"reasons":a.reasons,"nodeCount":a.nodes,"maximumDepth":a.depth,"textBytes":a.bytes,"symbolicDimensions":a.symbolic,"unknownDimensions":a.unknown}),
            )
        }
        "known" => Ok(json!(known(&v["value"]))),
        "union" | "unify" => Ok(
            match merge(
                &v["left"],
                &v["right"],
                v["operation"] == "union",
                &v["tensor_type_ids"],
                0,
            ) {
                Ok(t) => json!({"status":"pass","type":t,"reason":""}),
                Err(e) => json!({"status":"fail","type":null,"reason":e}),
            },
        ),
        _ => Err("unknown ONNX type contract".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symbolic_union_is_not_unification() {
        let a = json!({"kind":"symbolic","parameter":"N"});
        let b = json!({"kind":"value","value":3});
        assert_eq!(merge_dim(&a, &b, true).unwrap()["kind"], "unknown");
        assert_eq!(merge_dim(&a, &b, false), Some(b));
    }
    #[test]
    fn rank_and_dtype() {
        let registry = json!({"FLOAT32":1});
        let mut a = tensor("FLOAT32", "tensor", &registry);
        let mut b = a.clone();
        set_dims(&mut a, vec![unknown()]);
        set_dims(&mut b, vec![]);
        assert_eq!(
            merge(&a, &b, false, &registry, 0).unwrap_err(),
            "tensor_rank_mismatch"
        );
        assert_eq!(
            merge(&a, &b, true, &registry, 0).unwrap()["shapeDeclared"],
            false
        );
    }
    #[test]
    fn malformed_and_nested() {
        let v = json!({"kind":"sequence","valueFieldsPresent":[4],"childFieldCount":1,"elementType":{"kind":"map","valueFieldsPresent":[5],"keyFieldCount":1,"keyTypeName":"FLOAT32"}});
        let a = run(&json!({"operation":"assess","value":v,"depth":1})).unwrap();
        assert_eq!(a["nodeCount"], 2);
        assert_eq!(a["reasons"].as_array().unwrap().len(), 2);
    }
}
