//! ML map/vector contracts, exact integer keys and bounded sequence inventory.
use crate::portable_onnx_container::{canonical, descriptor, ty};
use crate::portable_onnx_ml::{count, exact_int, templates, unique};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde_json::{json, Value};
pub(crate) fn dtype(t: &Value) -> &str {
    if t["kind"] == "tensor" {
        t["dtype"]
            .as_str()
            .or(t["elementTypeName"].as_str())
            .unwrap_or("UNKNOWN")
    } else {
        "UNKNOWN"
    }
}
pub(crate) fn tensor(v: &Value, dt: &str, s: &[Value], declared: bool) -> Value {
    let mut t = tensor_type(dt, &dimensions(&json!({"shape":s})), &v["tensor_type_ids"]);
    if !declared {
        t["shapeDeclared"] = json!(false);
        t["shapeFieldCount"] = json!(0);
    }
    t
}
fn ints(a: &Value) -> Option<Vec<BigInt>> {
    let list = if a["intExactDecimals"].is_array()
        && !rows(&a["intExactDecimals"]).is_empty()
        && rows(&a["intExactDecimals"]).len() == rows(&a["ints"]).len()
    {
        &a["intExactDecimals"]
    } else {
        &a["ints"]
    };
    list.as_array()?.iter().map(exact_int).collect()
}
fn strings(a: &Value) -> Vec<Value> {
    rows(&a["strings"]).to_vec()
}
fn strattr<'a>(n: &'a Value, k: &str, default: &'a str) -> &'a str {
    attr(n, k)["s"].as_str().unwrap_or(default)
}
fn integer(a: &Value) -> Option<BigInt> {
    exact_int(if truth(&a["iExactDecimal"]) {
        &a["iExactDecimal"]
    } else {
        &a["i"]
    })
}
fn nonnegative(n: &BigInt) -> Option<i64> {
    n.to_i64().filter(|n| *n >= 0 && *n <= SAFE)
}
fn duplicate(v: &[Value]) -> usize {
    let mut seen = vec![];
    for x in v {
        if !seen.contains(x) {
            seen.push(x.clone());
        }
    }
    v.len() - seen.len()
}
fn decimals(v: &[BigInt]) -> Vec<Value> {
    v.iter().map(|n| json!(n.to_string())).collect()
}
fn exact_tensor(t: &Value) -> Option<Vec<BigInt>> {
    let vals = if t["initializerIntegerValuesExactComplete"] == true {
        &t["initializerIntegerValuesExactDecimals"]
    } else if t["initializerIntegerValuesComplete"] == true {
        &t["initializerIntegerValues"]
    } else if t["staticValuesComplete"] == true {
        &t["staticValues"]
    } else {
        return None;
    };
    vals.as_array()?.iter().map(exact_int).collect()
}
fn set(row: &mut Value, fields: Value) {
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
}
fn base(v: &Value, op: &str, t: &Value) -> Value {
    let mut r = templates()[op].clone();
    set(
        &mut r,
        json!({"scope":v["scope"],"node_index":v["node_index"],"imported_opset":v["imported_opset"],"input_name":v["node"]["inputs"][0].as_str().unwrap_or(""),"output_name":v["node"]["outputs"][0].as_str().unwrap_or(""),"input_kind":t["kind"].as_str().unwrap_or("unresolved")}),
    );
    r
}
fn finish(
    v: &Value,
    mut row: Value,
    ot: Value,
    state: Value,
    mut failures: Vec<String>,
    reasons: Vec<String>,
    risks: Vec<String>,
) -> Value {
    let status = if !failures.is_empty() {
        "fail"
    } else if !reasons.is_empty() {
        "partial"
    } else {
        "pass"
    };
    failures.extend(reasons);
    let codes = unique(failures);
    row["status"] = json!(status);
    row["reason_codes"] = json!(codes);
    row["risk_codes"] = json!(risks);
    row["canonical_output_type"] = json!(if ot.is_null() {
        "unresolved".into()
    } else {
        canonical(&ot)
    });
    let outputs = if status != "fail" && !ot.is_null() && truth(&v["node"]["outputs"][0]) {
        vec![json!([v["node"]["outputs"][0], descriptor(&ot, state)])]
    } else {
        vec![]
    };
    json!({"status":status,"reason":codes.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row})
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let op = string(&v["node"]["opType"]);
    Ok(match op {
        "ZipMap" => zip(v),
        "CastMap" => cast_map(v),
        "DictVectorizer" => dictionary(v),
        "CategoryMapper" => category(v),
        "FeatureVectorizer" => vector(v),
        "ArrayFeatureExtractor" => extract(v),
        _ => Value::Null,
    })
}
fn zip(v: &Value) -> Value {
    let n = &v["node"];
    let t = ty(v, input(v, 0));
    let dt = dtype(&t);
    let (mut failures, mut reasons, mut risks) = (vec![], vec![], vec![]);
    if t["kind"] != "tensor" {
        failures.push("zip_map_input_not_tensor".into());
    }
    if dt != "UNKNOWN" && dt != "FLOAT32" {
        failures.push(format!("zip_map_input_dtype_not_float:{dt}"));
    }
    if dt == "UNKNOWN" {
        reasons.push("zip_map_input_dtype_unresolved".into());
    }
    let sk = strings(attr(n, "classlabels_strings"));
    let ik = ints(attr(n, "classlabels_int64s"))
        .map(|a| decimals(&a))
        .unwrap_or_default();
    if sk.is_empty() == ik.is_empty() {
        failures.push("zip_map_requires_exactly_one_nonempty_classlabels_attribute".into());
    }
    let (kt, keys) = if !sk.is_empty() {
        ("STRING", sk)
    } else if !ik.is_empty() {
        ("INT64", ik)
    } else {
        ("UNDEFINED", vec![])
    };
    let dup = duplicate(&keys);
    if dup > 0 {
        risks.push("zip_map_duplicate_class_keys_information_loss_risk".into());
    }
    let rank = if t["shapeDeclared"] == true {
        Some(rows(&t["shape"]).len())
    } else {
        None
    };
    let (batch, width) = match rank {
        Some(1) => (Some(1), safe(&t["shape"][0]).filter(|n| *n >= 0)),
        Some(2) => (
            safe(&t["shape"][0]).filter(|n| *n >= 0),
            safe(&t["shape"][1]).filter(|n| *n >= 0),
        ),
        _ => (None, None),
    };
    match rank {
        None => reasons.push("zip_map_input_rank_unresolved".into()),
        Some(1 | 2) => {
            if batch.is_none() {
                reasons.push("zip_map_batch_dimension_runtime_unknown".into());
            }
            if width.is_none() {
                reasons.push("zip_map_feature_dimension_runtime_unknown".into());
            }
        }
        Some(r) => failures.push(format!("zip_map_input_rank_not_one_or_two:{r}")),
    }
    if let Some(w) = width {
        if !keys.is_empty() && w != keys.len() as i64 {
            failures.push(format!(
                "zip_map_feature_count_key_count_mismatch:{w}:{}",
                keys.len()
            ));
        }
    }
    let map = json!({"kind":"map","valueFieldsPresent":[5],"keyTypeId":v["tensor_type_ids"][kt],"keyTypeName":kt,"keyFieldCount":if kt=="UNDEFINED"{0}else{1},"valueType":tensor(v,"FLOAT32",&[],true),"childFieldCount":1,"dtype":"UNKNOWN","shape":[],"shapeDeclared":false,"shapeDimensions":[],"denotation":""});
    let seq = json!({"kind":"sequence","valueFieldsPresent":[4],"elementType":map,"childFieldCount":1,"dtype":"UNKNOWN","shape":[],"shapeDeclared":false,"shapeDimensions":[],"denotation":""});
    let inv = batch.filter(|n| *n <= 4096).map(|n| vec![map; n as usize]);
    if batch.is_some() && inv.is_none() {
        reasons.push("zip_map_sequence_inventory_not_materialized_over_limit".into());
    }
    let state = json!({"sequenceLengthStatus":if batch.is_some(){"assessed_exact"}else{"not_assessed_runtime_length"},"sequenceLength":batch,"sequenceElementInventoryStatus":if inv.is_some(){"assessed_exact"}else{"not_assessed"},"sequenceElementTypes":inv.unwrap_or_default(),"mapKeyType":kt,"mapKeyCount":keys.len(),"mapDuplicateKeyCount":dup});
    let mut r = base(v, "ZipMap", &t);
    set(
        &mut r,
        json!({"input_dtype":dt,"input_rank":rank,"input_shape":if rank.is_some(){t["shape"].clone()}else{json!([])},"exact_batch_count":batch,"exact_feature_count":width,"class_key_type":kt,"class_key_count":keys.len(),"duplicate_key_count":dup,"class_key_preview":keys.iter().take(8).collect::<Vec<_>>(),"exact_output_sequence_length":batch}),
    );
    finish(
        v,
        r,
        if kt == "UNDEFINED" { Value::Null } else { seq },
        state,
        failures,
        reasons,
        risks,
    )
}
fn cast_map(v: &Value) -> Value {
    let n = &v["node"];
    let input = input(v, 0);
    let t = ty(v, input);
    let (mut failures, mut reasons) = (vec![], vec![]);
    if t.is_null() {
        reasons.push("cast_map_input_type_unresolved".into());
    } else if t["kind"] != "map" {
        failures.push(format!("cast_map_input_not_map:{}", string(&t["kind"])));
    }
    let kt = if t["kind"] == "map" {
        t["keyTypeName"].as_str().unwrap_or("UNDEFINED")
    } else {
        "UNDEFINED"
    };
    let dt = if t["kind"] == "map" {
        dtype(&t["valueType"])
    } else {
        "UNKNOWN"
    };
    if kt != "UNDEFINED" && kt != "INT64" {
        failures.push(format!("cast_map_key_type_not_int64:{kt}"));
    }
    if dt != "UNKNOWN" && !["FLOAT32", "STRING"].contains(&dt) {
        failures.push(format!("cast_map_value_type_not_float_or_string:{dt}"));
    }
    if kt == "UNDEFINED" {
        reasons.push("cast_map_key_type_unresolved".into());
    }
    if dt == "UNKNOWN" {
        reasons.push("cast_map_value_type_unresolved".into());
    }
    let to = strattr(n, "cast_to", "TO_FLOAT");
    let form = strattr(n, "map_form", "DENSE");
    let od = match to {
        "TO_FLOAT" => "FLOAT32",
        "TO_STRING" => "STRING",
        "TO_INT64" => "INT64",
        _ => "UNKNOWN",
    };
    if od == "UNKNOWN" {
        failures.push(format!("cast_map_cast_to_invalid:{to}"));
    }
    if !["DENSE", "SPARSE"].contains(&form) {
        failures.push(format!("cast_map_form_invalid:{form}"));
    }
    let max = if attr(n, "max_map").is_null() {
        Some(1)
    } else {
        integer(attr(n, "max_map"))
            .and_then(|n| n.to_i64())
            .filter(|n| n.unsigned_abs() <= SAFE as u64)
    };
    if form == "SPARSE" {
        if max.is_none() {
            reasons.push("cast_map_max_map_not_safe_integer".into());
        }
        if max.is_some_and(|n| n < 0) {
            failures.push(format!("cast_map_max_map_negative:{}", max.unwrap()));
        }
    }
    let keys = input["mapKeysExact"]
        .as_array()
        .and_then(|a| a.iter().map(exact_int).collect::<Option<Vec<_>>>());
    let dense = safe(&input["mapKeyCount"])
        .filter(|n| *n >= 0)
        .or_else(|| keys.as_ref().map(|a| a.len() as i64));
    let mut bounds = "not_applicable";
    if form == "SPARSE" {
        if let Some(max) = max.filter(|n| *n >= 0) {
            if let Some(keys) = &keys {
                let invalid = keys
                    .iter()
                    .filter(|n| n < &&BigInt::zero() || n >= &&BigInt::from(max))
                    .count();
                bounds = if invalid > 0 { "fail" } else { "assessed_pass" };
                if invalid > 0 {
                    failures.push(format!("cast_map_sparse_key_out_of_bounds:{invalid}"));
                }
            } else {
                bounds = "not_assessed_runtime_keys";
                reasons.push("cast_map_sparse_key_bounds_runtime_unknown".into());
            }
        } else {
            bounds = "not_assessed_invalid_or_unresolved_max_map";
        }
    }
    let length = if form == "SPARSE" { max } else { dense };
    if form == "DENSE" && length.is_none() {
        reasons.push("cast_map_dense_length_runtime_unknown".into());
    }
    let shape = json!([length.filter(|n| *n >= 0).unwrap_or(-1)]);
    let ot = if od == "UNKNOWN" {
        Value::Null
    } else {
        tensor(v, od, rows(&shape), true)
    };
    let mut r = base(v, "CastMap", &t);
    set(
        &mut r,
        json!({"input_map_key_type":kt,"input_map_value_dtype":dt,"exact_input_map_key_count":dense,"sparse_key_bounds_status":bounds,"output_dtype":od,"exact_output_shape":shape,"exact_dense_output_element_count":length.filter(|n|*n>=0),"output_shape_basis":if form=="SPARSE"{"pinned_onnx_schema_sparse_max_map"}else if dense.is_none(){"rank_only_length_runtime_unknown"}else{"artifact_exact_input_map_cardinality"},"attribute_mode":form,"cast_to":to,"map_form":form,"max_map":max}),
    );
    finish(v, r, ot, Value::Null, failures, reasons, vec![])
}
fn dictionary(v: &Value) -> Value {
    let n = &v["node"];
    let input = input(v, 0);
    let t = ty(v, input);
    let (mut failures, mut reasons, mut risks) = (vec![], vec![], vec![]);
    if t.is_null() {
        reasons.push("dict_vectorizer_input_type_unresolved".into());
    } else if t["kind"] != "map" {
        failures.push(format!(
            "dict_vectorizer_input_not_map:{}",
            string(&t["kind"])
        ));
    }
    let kt = if t["kind"] == "map" {
        t["keyTypeName"].as_str().unwrap_or("UNDEFINED")
    } else {
        "UNDEFINED"
    };
    let dt = if t["kind"] == "map" {
        dtype(&t["valueType"])
    } else {
        "UNKNOWN"
    };
    let (sa, ia) = (attr(n, "string_vocabulary"), attr(n, "int64_vocabulary"));
    if sa.is_null() == ia.is_null() {
        failures.push("dict_vectorizer_requires_exactly_one_vocabulary_attribute".into());
    }
    let ints = ints(ia);
    if !ia.is_null() && ints.is_none() {
        failures.push("dict_vectorizer_int64_vocabulary_not_exactly_decoded".into());
    }
    let (vt, vocab) = if !sa.is_null() {
        ("STRING", strings(sa))
    } else if !ia.is_null() {
        ("INT64", decimals(&ints.unwrap_or_default()))
    } else {
        ("UNDEFINED", vec![])
    };
    let dup = duplicate(&vocab);
    if dup > 0 {
        risks.push("dict_vectorizer_duplicate_vocabulary_columns".into());
    }
    if kt == "UNDEFINED" {
        reasons.push("dict_vectorizer_map_key_type_unresolved".into());
    } else if vt != "UNDEFINED" && kt != vt {
        failures.push(format!(
            "dict_vectorizer_map_key_vocabulary_type_mismatch:{kt}:{vt}"
        ));
    }
    if dt == "UNKNOWN" {
        reasons.push("dict_vectorizer_map_value_type_unresolved".into());
    }
    let allowed = kt == "STRING" && ["INT64", "FLOAT32", "FLOAT64"].contains(&dt)
        || kt == "INT64" && ["STRING", "FLOAT32", "FLOAT64"].contains(&dt);
    if kt != "UNDEFINED" && dt != "UNKNOWN" && !allowed {
        failures.push(format!("dict_vectorizer_map_type_not_allowed:{kt}:{dt}"));
    }
    let s = json!([1, vocab.len()]);
    let ot = if dt == "UNKNOWN" {
        Value::Null
    } else {
        tensor(v, dt, rows(&s), true)
    };
    let mut r = base(v, "DictVectorizer", &t);
    set(
        &mut r,
        json!({"input_map_key_type":kt,"input_map_value_dtype":dt,"exact_input_map_key_count":safe(&input["mapKeyCount"]).filter(|n|*n>=0),"exact_feature_count":vocab.len(),"output_dtype":dt,"exact_output_shape":s,"exact_dense_output_element_count":vocab.len(),"vocabulary_type":vt,"vocabulary_count":vocab.len(),"duplicate_vocabulary_count":dup,"vocabulary_preview":vocab.iter().take(8).collect::<Vec<_>>()}),
    );
    finish(v, r, ot, Value::Null, failures, reasons, risks)
}
fn category(v: &Value) -> Value {
    let n = &v["node"];
    let t = ty(v, input(v, 0));
    let dt = dtype(&t);
    let (mut failures, mut reasons, mut risks) = (vec![], vec![], vec![]);
    if t.is_null() {
        reasons.push("category_mapper_input_type_unresolved".into());
    } else if t["kind"] != "tensor" {
        failures.push(format!(
            "category_mapper_input_not_tensor:{}",
            string(&t["kind"])
        ));
    }
    if dt == "UNKNOWN" {
        reasons.push("category_mapper_input_dtype_unresolved".into());
    } else if !["STRING", "INT64"].contains(&dt) {
        failures.push(format!(
            "category_mapper_input_dtype_not_string_or_int64:{dt}"
        ));
    }
    let (sa, ia) = (attr(n, "cats_strings"), attr(n, "cats_int64s"));
    if sa.is_null() {
        failures.push("category_mapper_cats_strings_required".into());
    }
    if ia.is_null() {
        failures.push("category_mapper_cats_int64s_required".into());
    }
    let ss = strings(sa);
    let ints = ints(ia);
    if !ia.is_null() && ints.is_none() {
        failures.push("category_mapper_cats_int64s_not_exactly_decoded".into());
    }
    let valid = ints.is_some();
    let ii = decimals(&ints.unwrap_or_default());
    let aligned = !sa.is_null() && !ia.is_null() && valid && ss.len() == ii.len();
    if !sa.is_null() && !ia.is_null() && ss.len() != ii.len() {
        failures.push(format!(
            "category_mapper_category_count_mismatch:{}:{}",
            ss.len(),
            ii.len()
        ));
    }
    let (ds, di) = (duplicate(&ss), duplicate(&ii));
    let (dir, od) = match dt {
        "STRING" => ("STRING_TO_INT64", "INT64"),
        "INT64" => ("INT64_TO_STRING", "STRING"),
        _ => ("UNRESOLVED", "UNKNOWN"),
    };
    let dup = if !aligned {
        0
    } else if dt == "STRING" {
        ds
    } else if dt == "INT64" {
        di
    } else {
        0
    };
    if dup > 0 {
        risks.push("category_mapper_duplicate_active_keys_last_write_wins".into());
    }
    let default = if dt == "STRING" {
        integer(attr(n, "default_int64"))
            .map(|n| n.to_string())
            .unwrap_or("-1".into())
    } else if dt == "INT64" {
        strattr(n, "default_string", "_Unused").into()
    } else {
        String::new()
    };
    let declared = t["kind"] == "tensor" && t["shapeDeclared"] == true;
    let s = if declared {
        rows(&t["shape"]).to_vec()
    } else {
        vec![]
    };
    let concrete = declared && s.iter().all(|d| safe(d).is_some_and(|n| n >= 0));
    let total = if concrete { count(&s) } else { None };
    if !declared {
        reasons.push("category_mapper_input_shape_unresolved".into());
    } else if !concrete {
        reasons.push("category_mapper_dynamic_shape_preserved".into());
    } else if total.is_none() {
        reasons.push("category_mapper_output_element_count_overflow".into());
    }
    let ot = if od == "UNKNOWN" {
        Value::Null
    } else {
        tensor(v, od, &s, declared)
    };
    let mut r = base(v, "CategoryMapper", &t);
    set(
        &mut r,
        json!({"input_dtype":dt,"input_rank":if declared{Some(s.len())}else{None},"input_shape":s,"output_dtype":od,"exact_output_rank":if declared{Some(s.len())}else{None},"exact_output_shape":s,"exact_dense_output_element_count":total,"mapping_direction":dir,"category_pair_count":if aligned{ss.len()}else{0},"category_string_count":ss.len(),"category_int64_count":ii.len(),"duplicate_string_key_count":ds,"duplicate_int64_key_count":di,"active_duplicate_key_count":dup,"active_default_type":if od=="UNKNOWN"{"UNDEFINED"}else{od},"active_default_value":default,"category_string_preview":ss.iter().take(8).collect::<Vec<_>>(),"category_int64_preview":ii.iter().take(8).collect::<Vec<_>>()}),
    );
    finish(v, r, ot, Value::Null, failures, reasons, risks)
}
fn vector(v: &Value) -> Value {
    let n = &v["node"];
    let types = rows(&n["inputs"])
        .iter()
        .enumerate()
        .map(|(i, _)| ty(v, input(v, i)))
        .collect::<Vec<_>>();
    let dts = types.iter().map(dtype).collect::<Vec<_>>();
    let shapes = types
        .iter()
        .map(|t| {
            if t["kind"] == "tensor" && t["shapeDeclared"] == true {
                rows(&t["shape"]).to_vec()
            } else {
                vec![]
            }
        })
        .collect::<Vec<_>>();
    let ranks = types
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if t["kind"] == "tensor" && t["shapeDeclared"] == true {
                Some(shapes[i].len())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let (mut failures, mut reasons, mut risks) = (vec![], vec![], vec![]);
    if types.is_empty() {
        failures.push("feature_vectorizer_requires_at_least_one_input".into());
    }
    for (i, t) in types.iter().enumerate() {
        if t.is_null() {
            reasons.push(format!("feature_vectorizer_input_type_unresolved:{i}"));
        } else if t["kind"] != "tensor" {
            failures.push(format!(
                "feature_vectorizer_input_not_tensor:{i}:{}",
                string(&t["kind"])
            ));
        }
        let dt = dts[i];
        if dt == "UNKNOWN" {
            reasons.push(format!("feature_vectorizer_input_dtype_unresolved:{i}"));
        } else if !["INT32", "INT64", "FLOAT32", "FLOAT64"].contains(&dt) {
            failures.push(format!(
                "feature_vectorizer_input_dtype_not_supported:{i}:{dt}"
            ));
        }
        if ranks[i].is_none() {
            reasons.push(format!("feature_vectorizer_input_rank_unresolved:{i}"));
        } else if ranks[i] == Some(0) {
            failures.push(format!("feature_vectorizer_scalar_input_not_supported:{i}"));
        }
    }
    let known_dts = unique(
        dts.iter()
            .filter(|d| **d != "UNKNOWN")
            .map(|d| d.to_string())
            .collect(),
    );
    if known_dts.len() > 1 {
        failures.push(format!(
            "feature_vectorizer_input_dtype_mismatch:{}",
            known_dts.join(":")
        ));
    }
    let a = attr(n, "inputdimensions");
    if a.is_null() {
        failures.push("feature_vectorizer_inputdimensions_required_by_pinned_ort".into());
    }
    let configured = ints(a);
    if !a.is_null() && configured.is_none() {
        failures.push("feature_vectorizer_inputdimensions_not_exactly_decoded".into());
    }
    let configured = configured.unwrap_or_default();
    if !a.is_null() && configured.is_empty() {
        failures.push("feature_vectorizer_inputdimensions_must_be_nonempty".into());
    }
    if !a.is_null() && configured.len() != types.len() {
        failures.push(format!(
            "feature_vectorizer_input_count_dimension_count_mismatch:{}:{}",
            types.len(),
            configured.len()
        ));
    }
    for n in &configured {
        if n < &BigInt::zero() {
            failures.push(format!("feature_vectorizer_negative_inputdimension:{n}"));
        }
    }
    let dimensions = configured.iter().map(nonnegative).collect::<Vec<_>>();
    if dimensions.iter().any(Option::is_none) && !configured.iter().any(|n| n < &BigInt::zero()) {
        reasons.push("feature_vectorizer_inputdimension_exceeds_safe_static_arithmetic".into());
    }
    let total_big: BigInt = configured.iter().sum();
    let total = nonnegative(&total_big);
    if total.is_none() && total_big >= BigInt::zero() {
        reasons.push("feature_vectorizer_total_dimension_exceeds_safe_static_arithmetic".into());
    }
    let batches = ranks
        .iter()
        .enumerate()
        .map(|(i, r)| match r {
            None | Some(0) => None,
            Some(1) => Some(1),
            _ => safe(&shapes[i][0]).filter(|n| *n >= 0),
        })
        .collect::<Vec<_>>();
    let batch = batches.first().copied().flatten();
    if batches.iter().any(Option::is_none) {
        reasons.push("feature_vectorizer_batch_consistency_runtime_unknown".into());
    }
    let known_batches = unique(batches.iter().flatten().map(ToString::to_string).collect());
    if known_batches.len() > 1 {
        failures.push(format!(
            "feature_vectorizer_batch_size_mismatch:{}",
            known_batches.join(":")
        ));
    }
    let widths = ranks
        .iter()
        .enumerate()
        .map(|(i, r)| match r {
            None | Some(0) => None,
            Some(1) => count(&shapes[i]),
            _ => count(&shapes[i][1..]),
        })
        .collect::<Vec<_>>();
    if widths.iter().any(Option::is_none) {
        reasons.push("feature_vectorizer_input_row_width_runtime_unknown".into());
    }
    let mut copied = vec![];
    let mut padded = vec![];
    let mut truncated = vec![];
    for (i, w) in widths.iter().enumerate() {
        let pair = w.zip(dimensions.get(i).copied().flatten());
        copied.push(pair.map(|(a, b)| a.min(b)));
        padded.push(pair.map(|(a, b)| (b - a).max(0)));
        truncated.push(pair.map(|(a, b)| (a - b).max(0)));
    }
    let sum = |a: &[Option<i64>]| {
        a.iter()
            .try_fold(0i64, |sum, n| sum.checked_add((*n)?).filter(|n| *n <= SAFE))
    };
    let (cs, ps, ts) = (sum(&copied), sum(&padded), sum(&truncated));
    if ts.is_some_and(|n| n > 0) {
        risks.push("feature_vectorizer_truncates_input_features".into());
    }
    let s = json!([batch.unwrap_or(-1), total.unwrap_or(-1)]);
    let elements = count(rows(&s));
    if batch.is_some() && total.is_some() && elements.is_none() {
        reasons.push("feature_vectorizer_output_element_count_overflow".into());
    }
    let ot = tensor(v, "FLOAT32", rows(&s), true);
    let mut r = base(v, "FeatureVectorizer", &Value::Null);
    set(
        &mut r,
        json!({"input_names":n["inputs"],"input_dtype":if known_dts.len()==1{known_dts[0].as_str()}else if known_dts.len()>1{"MIXED"}else{"UNKNOWN"},"input_dtypes":dts,"input_kind":"tensor_list","input_rank":ranks.first().copied().flatten(),"input_ranks":ranks,"input_shape":shapes.first().cloned().unwrap_or_default(),"input_shapes":shapes,"exact_batch_count":batch,"exact_batch_counts":batches,"exact_feature_count":total,"exact_input_row_feature_counts":widths,"exact_output_shape":s,"exact_dense_output_element_count":elements,"configured_feature_dimensions":decimals(&configured),"configured_feature_dimension_count":configured.len(),"total_configured_feature_count":total,"copied_feature_counts_per_input":copied,"padded_feature_counts_per_input":padded,"truncated_feature_counts_per_input":truncated,"exact_copied_feature_count_per_batch":cs,"exact_padded_feature_count_per_batch":ps,"exact_truncated_feature_count_per_batch":ts,"padded_input_count":padded.iter().filter(|n|n.is_some_and(|n|n>0)).count(),"truncated_input_count":truncated.iter().filter(|n|n.is_some_and(|n|n>0)).count()}),
    );
    finish(v, r, ot, Value::Null, failures, reasons, risks)
}
fn extract(v: &Value) -> Value {
    let n = &v["node"];
    let (t, it) = (ty(v, input(v, 0)), ty(v, input(v, 1)));
    let (dt, idt) = (dtype(&t), dtype(&it));
    let (mut failures, mut reasons) = (vec![], vec![]);
    for (ty, prefix, not_tensor) in [
        (&t, "data", "data_not_tensor"),
        (&it, "index", "indices_not_tensor"),
    ] {
        if ty.is_null() {
            reasons.push(format!("array_feature_extractor_{prefix}_type_unresolved"));
        } else if ty["kind"] != "tensor" {
            failures.push(format!(
                "array_feature_extractor_{not_tensor}:{}",
                string(&ty["kind"])
            ));
        }
    }
    if dt == "UNKNOWN" {
        reasons.push("array_feature_extractor_data_dtype_unresolved".into());
    } else if !["FLOAT32", "FLOAT64", "INT32", "INT64", "STRING"].contains(&dt) {
        failures.push(format!(
            "array_feature_extractor_data_dtype_not_supported:{dt}"
        ));
    }
    if idt == "UNKNOWN" {
        reasons.push("array_feature_extractor_index_dtype_unresolved".into());
    } else if idt != "INT64" {
        failures.push(format!(
            "array_feature_extractor_index_dtype_not_int64:{idt}"
        ));
    }
    let declared = t["kind"] == "tensor" && t["shapeDeclared"] == true;
    let ideclared = it["kind"] == "tensor" && it["shapeDeclared"] == true;
    let s = if declared {
        rows(&t["shape"]).to_vec()
    } else {
        vec![]
    };
    let is = if ideclared {
        rows(&it["shape"]).to_vec()
    } else {
        vec![]
    };
    let rank = if declared { Some(s.len()) } else { None };
    let irank = if ideclared { Some(is.len()) } else { None };
    if rank.is_none() {
        reasons.push("array_feature_extractor_data_rank_unresolved".into());
    } else if rank == Some(0) {
        failures.push("array_feature_extractor_scalar_data_not_supported".into());
    }
    let concrete = ideclared && is.iter().all(|n| safe(n).is_some_and(|n| n >= 0));
    let ic = if concrete { count(&is) } else { None };
    if !concrete {
        reasons.push("array_feature_extractor_index_count_runtime_unknown".into());
    } else if ic.is_none() {
        reasons.push("array_feature_extractor_index_count_overflow".into());
    } else if ic == Some(0) {
        failures.push("array_feature_extractor_empty_indices_not_supported".into());
    }
    let indices = exact_tensor(input(v, 1));
    if let (Some(vals), Some(count)) = (&indices, ic) {
        if vals.len() as i64 != count {
            failures.push(format!(
                "array_feature_extractor_index_value_count_mismatch:{}:{count}",
                vals.len()
            ));
        }
    }
    let last = s.last().and_then(safe).filter(|n| *n >= 0);
    let (mut invalid, mut bounds) = (0i64, "not_assessed_runtime_values");
    if let Some(vals) = &indices {
        invalid = vals
            .iter()
            .filter(|n| n < &&BigInt::zero() || last.is_some_and(|max| *n >= &BigInt::from(max)))
            .count() as i64;
        if invalid > 0 {
            failures.push(format!(
                "array_feature_extractor_index_out_of_bounds:{invalid}"
            ));
            bounds = "fail";
        } else if last.is_none() {
            reasons.push("array_feature_extractor_last_axis_bound_runtime_unknown".into());
            bounds = "not_assessed_dynamic_axis";
        } else {
            bounds = "assessed_pass";
        }
    } else {
        if ic.is_some_and(|n| n > 0) && last == Some(0) {
            invalid = ic.unwrap();
            failures.push(format!(
                "array_feature_extractor_index_out_of_bounds:{invalid}"
            ));
            bounds = "fail";
        } else if ic.is_none() {
            bounds = "not_assessed_dynamic_cardinality";
        }
        if invalid == 0 {
            reasons.push("array_feature_extractor_index_values_runtime_unknown".into());
        }
    }
    let outdeclared = rank.is_some_and(|n| n > 0);
    let os = if outdeclared {
        let mut os = if rank == Some(1) {
            vec![json!(1)]
        } else {
            s[..s.len() - 1].to_vec()
        };
        os.push(json!(ic.unwrap_or(-1)));
        os
    } else {
        vec![]
    };
    let total = if outdeclared { count(&os) } else { None };
    if outdeclared && os.iter().all(|n| safe(n).is_some_and(|n| n >= 0)) && total.is_none() {
        reasons.push("array_feature_extractor_output_element_count_overflow".into());
    }
    let ot = if dt == "UNKNOWN" {
        Value::Null
    } else {
        tensor(v, dt, &os, outdeclared)
    };
    let vals = indices.as_ref().map(|a| decimals(a)).unwrap_or_default();
    let mut r = base(v, "ArrayFeatureExtractor", &t);
    set(
        &mut r,
        json!({"input_names":n["inputs"],"input_dtype":dt,"input_dtypes":[dt,idt],"input_kind":"tensor","input_rank":rank,"input_ranks":[rank,irank],"input_shape":s,"input_shapes":[s,is],"exact_feature_count":ic,"output_dtype":dt,"exact_output_rank":if outdeclared{Some(os.len())}else{None},"exact_output_shape":os,"exact_dense_output_element_count":total,"index_input_name":n["inputs"][1].as_str().unwrap_or(""),"index_input_dtype":idt,"index_input_rank":irank,"index_input_shape":is,"exact_index_count":ic,"exact_index_values_status":if indices.is_some(){"assessed_exact"}else{"not_assessed_runtime_values"},"exact_index_values":vals,"exact_index_preview":vals.iter().take(16).collect::<Vec<_>>(),"duplicate_index_count":duplicate(&vals),"index_bounds_status":bounds,"out_of_bounds_index_count":invalid}),
    );
    finish(v, r, ot, Value::Null, failures, reasons, vec![])
}
