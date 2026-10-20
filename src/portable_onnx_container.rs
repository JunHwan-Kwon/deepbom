//! Lossless sequence/optional semantic inference with bounded materialization.
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
const LIMIT: usize = 4096;
fn known(t: &Value) -> bool {
    crate::portable_onnx_type::run(&json!({"operation":"known","value":t})).ok()
        == Some(json!(true))
}
pub(crate) fn ty(v: &Value, t: &Value) -> Value {
    if t.is_null() {
        return Value::Null;
    }
    if !t["typeProto"].is_null() {
        return t["typeProto"].clone();
    }
    if !t["type_proto"].is_null() {
        return t["type_proto"].clone();
    }
    if !["", "tensor", "unresolved", "undefined"].contains(&string(&t["valueKind"])) {
        return Value::Null;
    }
    let dtype = t["dtype"].as_str().unwrap_or("UNKNOWN");
    let mut r = tensor_type(dtype, &dimensions(t), &v["tensor_type_ids"]);
    if t["shapeDeclared"] != true {
        unrank(&mut r);
    }
    r
}
fn unrank(t: &mut Value) {
    t["shape"] = json!([]);
    t["shapeDimensions"] = json!([]);
    t["shapeDeclared"] = json!(false);
    t["shapeFieldCount"] = json!(0);
}
fn wrapper(t: &Value, kind: &str) -> Value {
    json!({"kind":kind,"valueFieldsPresent":[if kind=="sequence"{4}else{9}],"elementType":t,"childFieldCount":1,"dtype":"UNKNOWN","shape":[],"shapeDeclared":false,"shapeDimensions":[],"denotation":""})
}
pub(crate) fn descriptor(t: &Value, state: Value) -> Value {
    let dense = ["tensor", "sparse_tensor"].contains(&string(&t["kind"]));
    let mut v = json!({"valueKind":t["kind"],"typeProto":t,"dtype":if dense{t["dtype"].as_str().or(t["elementTypeName"].as_str()).unwrap_or("UNKNOWN")}else{"UNKNOWN"},"shape":if dense{t["shape"].clone()}else{json!([])},"shapeDeclared":dense&&t["shapeDeclared"]==true});
    if let Some(m) = state.as_object() {
        v.as_object_mut().unwrap().extend(m.clone());
    }
    v
}
fn length(t: &Value) -> Option<i64> {
    if t["sequenceLengthStatus"] == "assessed_exact" {
        safe(&t["sequenceLength"]).filter(|n| *n >= 0)
    } else {
        None
    }
}
fn inventory(t: &Value) -> Option<Vec<Value>> {
    if t["sequenceElementInventoryStatus"] == "assessed_exact" {
        t["sequenceElementTypes"].as_array().cloned()
    } else {
        None
    }
}
fn presence(t: &Value) -> Option<bool> {
    if t["optionalPresenceStatus"] == "assessed_exact" {
        t["optionalPresence"].as_bool()
    } else {
        None
    }
}
fn state(length: Option<i64>, inventory: Option<Vec<Value>>) -> Value {
    let length = length.filter(|n| *n >= 0 && *n <= SAFE);
    json!({"sequenceLengthStatus":if length.is_some(){"assessed_exact"}else{"not_assessed_runtime_length"},"sequenceLength":length,"sequenceElementInventoryStatus":if inventory.is_some(){"assessed_exact"}else{"not_assessed"},"sequenceElementTypes":inventory.unwrap_or_default()})
}
fn copy(t: &Value) -> Value {
    let mut m = serde_json::Map::new();
    for k in [
        "sequenceLengthStatus",
        "sequenceLength",
        "sequenceElementInventoryStatus",
        "optionalPresenceStatus",
        "optionalPresence",
        "sequenceElementTypes",
    ] {
        if let Some(x) = t.get(k) {
            m.insert(k.into(), x.clone());
        }
    }
    Value::Object(m)
}
fn scalar(t: i64, source: &str) -> Value {
    json!({"staticValuesStatus":"assessed_exact_static_data","staticValuesComplete":true,"staticValues":[t],"staticValuesSource":source})
}
fn ints(t: &Value) -> Option<Vec<i64>> {
    if t["staticValuesComplete"] != true {
        return None;
    }
    t["staticValues"].as_array()?.iter().map(safe).collect()
}
fn integer(n: &Value, k: &str) -> Option<i64> {
    safe(&attr(n, k)["i"]).or_else(|| {
        attr(n, k)["iExactDecimal"]
            .as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|n| n.unsigned_abs() <= SAFE as u64)
    })
}
fn seq(t: &Value) -> bool {
    t["kind"] == "sequence" && t["elementType"]["kind"] == "tensor" && known(&t["elementType"])
}
fn optional(t: &Value) -> bool {
    match string(&t["kind"]) {
        "tensor" => known(t),
        "sequence" => seq(t),
        "optional" => {
            if t["elementType"]["kind"] == "tensor" {
                known(&t["elementType"])
            } else {
                seq(&t["elementType"])
            }
        }
        _ => false,
    }
}
pub(crate) fn canonical(t: &Value) -> String {
    let kind = string(&t["kind"]);
    match kind {
        "tensor" | "sparse_tensor" => {
            let shape = if t["shapeDeclared"] == true {
                format!(
                    "[{}]",
                    rows(&t["shapeDimensions"])
                        .iter()
                        .map(|d| match string(&d["kind"]) {
                            "value" => d["value"].to_string(),
                            "symbolic" => d["parameter"].as_str().unwrap_or("symbol?").into(),
                            _ => "?".into(),
                        })
                        .collect::<Vec<String>>()
                        .join(",")
                )
            } else {
                "[rank?]".into()
            };
            format!(
                "{kind}<{}{shape}>",
                t["elementTypeName"].as_str().unwrap_or("UNDEFINED")
            )
        }
        "sequence" | "optional" => format!("{kind}<{}>", canonical(&t["elementType"])),
        "map" => format!(
            "map<{},{}>",
            t["keyTypeName"].as_str().unwrap_or("UNDEFINED"),
            canonical(&t["valueType"])
        ),
        "opaque" => format!("opaque<{}:{}>", string(&t["domain"]), string(&t["name"])),
        _ => {
            if t.is_null() {
                "not declared".into()
            } else {
                "undefined".into()
            }
        }
    }
}
fn finish(v: &Value, status: &str, patch: Option<Value>, reasons: Vec<String>) -> Value {
    let n = &v["node"];
    let outputs = patch
        .filter(|_| truth(&n["outputs"][0]))
        .map(|p| vec![json!([n["outputs"][0], p])])
        .unwrap_or_default();
    let row = json!({"scope":v["scope"],"node_index":v["node_index"],"op_name":n["opType"],"imported_opset":v["imported_opset"],"status":status,"input_names":rows(&n["inputs"]).iter().filter(|n|truth(n)).collect::<Vec<_>>(),"output_names":rows(&n["outputs"]).iter().filter(|n|truth(n)).collect::<Vec<_>>(),"output_kinds":outputs.iter().map(|p|p[1]["valueKind"].clone()).collect::<Vec<_>>(),"canonical_output_types":outputs.iter().map(|p|canonical(&p[1]["typeProto"])).collect::<Vec<_>>(),"sequence_lengths":outputs.iter().map(|p|length(&p[1])).collect::<Vec<_>>(),"optional_presence":outputs.iter().map(|p|presence(&p[1])).collect::<Vec<_>>(),"reason_codes":reasons});
    json!({"status":status,"reason":reasons.first().map(String::as_str).unwrap_or(""),"result":{"outputs":outputs},"row":row})
}
fn fail(v: &Value, reason: &str) -> Value {
    finish(v, "fail", None, vec![reason.into()])
}
fn emit(v: &Value, t: Value, state: Value, reason: &str) -> Value {
    finish(
        v,
        if reason.is_empty() { "pass" } else { "partial" },
        Some(descriptor(&t, state)),
        if reason.is_empty() {
            vec![]
        } else {
            vec![reason.into()]
        },
    )
}
fn position(
    v: &Value,
    i: usize,
    len: Option<i64>,
    insert: bool,
    required: bool,
) -> Result<(Option<i64>, &'static str), &'static str> {
    if !truth(&v["node"]["inputs"][i]) {
        if required {
            return Err("sequence_position_missing");
        }
        return Ok(match len {
            None => (None, "sequence_length_runtime_unknown"),
            Some(l) => (Some(if insert { l } else { l - 1 }), ""),
        });
    }
    let t = input(v, i);
    if !string(&t["dtype"]).is_empty() && !["INT32", "INT64"].contains(&string(&t["dtype"])) {
        return Err("sequence_position_dtype_invalid");
    }
    if t["shapeDeclared"] == true && !rows(&t["shape"]).is_empty() {
        return Err("sequence_position_not_scalar");
    }
    let values = ints(t);
    let Some(values) = values else {
        return Ok((None, "sequence_position_runtime_unknown"));
    };
    if values.len() != 1 || len.is_none() {
        return Ok((None, "sequence_length_runtime_unknown"));
    }
    let (raw, len) = (values[0], len.unwrap());
    if raw < -len || raw > if insert { len } else { len - 1 } {
        return Err("sequence_position_out_of_bounds");
    }
    Ok((Some(if raw < 0 { raw + len } else { raw }), ""))
}
fn union(v: &Value, types: &[Value]) -> R<Value> {
    let Some(first) = types.first() else {
        return Ok(json!({"status":"unresolved","type":null,"reason":"type_proto_missing"}));
    };
    let mut t = first.clone();
    for next in &types[1..] {
        let merged = crate::portable_onnx_type::run(
            &json!({"operation":"union","left":t,"right":next,"tensor_type_ids":v["tensor_type_ids"]}),
        )?;
        if merged["status"] != "pass" {
            return Ok(merged);
        }
        t = merged["type"].clone();
    }
    Ok(json!({"status":"pass","type":t,"reason":""}))
}
fn axis(raw: i64, rank: usize) -> Option<usize> {
    let x = if raw < 0 { raw + rank as i64 } else { raw };
    if x >= 0 && x < rank as i64 {
        Some(x as usize)
    } else {
        None
    }
}
fn same(a: &Value, b: &Value) -> bool {
    a["kind"] == b["kind"]
        && match string(&a["kind"]) {
            "value" => a["value"] == b["value"],
            "symbolic" => a["parameter"] == b["parameter"],
            _ => true,
        }
}
fn concat(v: &Value, t: &Value, sequence: &Value) -> Value {
    let elem = &t["elementType"];
    let len = length(sequence);
    if len == Some(0) {
        return fail(v, "concat_from_sequence_provably_empty");
    }
    let inv = inventory(sequence);
    if let (Some(inv), Some(len)) = (&inv, len) {
        if inv.len() as i64 != len {
            return fail(v, "concat_from_sequence_inventory_length_mismatch");
        }
    }
    if inv
        .as_ref()
        .is_some_and(|a| a.iter().any(|t| t["kind"] != "tensor"))
    {
        return fail(v, "concat_from_sequence_inventory_element_not_tensor");
    }
    let dt = elem["dtype"]
        .as_str()
        .or(elem["elementTypeName"].as_str())
        .unwrap_or("UNKNOWN");
    let dts = inv
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|t| {
            t["dtype"]
                .as_str()
                .or(t["elementTypeName"].as_str())
                .unwrap_or("UNKNOWN")
        })
        .collect::<std::collections::BTreeSet<_>>();
    if dts.len() > 1 {
        return fail(v, "concat_from_sequence_inventory_dtype_mismatch");
    }
    if dts.len() == 1 && !dts.contains(dt) {
        return fail(v, "concat_from_sequence_inventory_aggregate_dtype_mismatch");
    }
    let n = &v["node"];
    let Some(raw) = integer(n, "axis") else {
        return fail(v, "concat_from_sequence_axis_missing");
    };
    let new = integer(n, "new_axis").unwrap_or(0);
    if ![0, 1].contains(&new) {
        return fail(v, "concat_from_sequence_new_axis_invalid");
    }
    let source = inv
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .find(|t| t["shapeDeclared"] == true)
        .unwrap_or(elem);
    if source["shapeDeclared"] != true {
        let mut t = tensor_type(dt, &[], &v["tensor_type_ids"]);
        unrank(&mut t);
        return emit(v, t, json!({}), "concat_from_sequence_element_rank_unknown");
    }
    let r = rows(&source["shape"]).len();
    let Some(ax) = axis(raw, r + new as usize) else {
        return fail(v, "concat_from_sequence_axis_out_of_range");
    };
    let mut dims = rows(&source["shapeDimensions"]).to_vec();
    if new == 1 {
        dims.insert(ax, unknown());
    } else {
        dims[ax] = unknown();
    }
    let mut reason = "concat_from_sequence_axis_extent_unknown";
    if let (Some(inv), Some(_)) = (inv, len) {
        if inv.is_empty() {
            return fail(v, "concat_from_sequence_provably_empty");
        }
        if inv.iter().all(|t| t["shapeDeclared"] == true) {
            let shapes = inv
                .iter()
                .map(|t| rows(&t["shapeDimensions"]))
                .collect::<Vec<_>>();
            if shapes.iter().any(|s| s.len() != shapes[0].len()) {
                return fail(v, "concat_from_sequence_inventory_rank_mismatch");
            }
            for i in 0..shapes[0].len() {
                if (new == 1 || i != ax) && shapes.iter().any(|s| !same(&s[i], &shapes[0][i])) {
                    return fail(
                        v,
                        if new == 1 {
                            "concat_from_sequence_stack_shape_mismatch"
                        } else {
                            "concat_from_sequence_non_axis_shape_mismatch"
                        },
                    );
                }
            }
            dims = shapes[0].to_vec();
            if new == 1 {
                dims.insert(ax, dim(shapes.len() as i64));
            } else {
                let sum = shapes
                    .iter()
                    .try_fold(0i64, |sum, s| sum.checked_add(dv(&s[ax])?))
                    .filter(|n| *n <= SAFE);
                dims[ax] = sum.map(dim).unwrap_or_else(unknown);
            }
            reason = "";
        }
    }
    emit(
        v,
        tensor_type(dt, &dims, &v["tensor_type_ids"]),
        json!({}),
        reason,
    )
}
fn split(v: &Value, t: &Value) -> Value {
    if t["kind"] != "tensor" || !known(t) {
        return fail(v, "split_to_sequence_input_not_typed_tensor");
    }
    let n = &v["node"];
    let r = if t["shapeDeclared"] == true {
        Some(rows(&t["shape"]).len())
    } else {
        None
    };
    let ax = r.and_then(|r| axis(integer(n, "axis").unwrap_or(0), r));
    if r.is_some() && ax.is_none() {
        return fail(v, "split_to_sequence_axis_out_of_range");
    }
    let s = input(v, 1);
    let supplied = truth(&n["inputs"][1]);
    let vals = if supplied { ints(s) } else { None };
    if supplied && !s.is_null() && !["INT32", "INT64"].contains(&string(&s["dtype"])) {
        return fail(v, "split_to_sequence_split_dtype_invalid");
    }
    if supplied && s["shapeDeclared"] == true && rows(&s["shape"]).len() > 1 {
        return fail(v, "split_to_sequence_split_rank_invalid");
    }
    if vals
        .as_ref()
        .is_some_and(|v| v.is_empty() || v.iter().any(|n| *n <= 0))
    {
        return fail(v, "split_to_sequence_split_values_not_positive");
    }
    let original = if r.is_some() {
        rows(&t["shapeDimensions"]).to_vec()
    } else {
        vec![]
    };
    let extent = ax
        .and_then(|a| original.get(a))
        .and_then(dv)
        .filter(|n| *n >= 0);
    let mut dims = original.clone();
    let (mut len, mut inv, mut reason) = (None, None, "");
    let dtype = string(&t["dtype"]);
    if !supplied {
        let keep = integer(n, "keepdims").unwrap_or(1);
        if ![0, 1].contains(&keep) {
            return fail(v, "split_to_sequence_keepdims_invalid");
        }
        if let Some(ax) = ax {
            if keep == 0 {
                dims.remove(ax);
            } else {
                dims[ax] = dim(1);
            }
        }
        len = extent;
        if let Some(l) = len {
            if l <= LIMIT as i64 {
                inv = Some(vec![
                    tensor_type(dtype, &dims, &v["tensor_type_ids"]);
                    l as usize
                ]);
            }
        } else {
            reason = "split_to_sequence_axis_extent_unknown";
        }
    } else if vals.is_none() {
        if let Some(ax) = ax {
            dims[ax] = unknown();
        }
        reason = "split_to_sequence_split_values_runtime_unknown";
    } else if s["shapeDeclared"] == true && rows(&s["shape"]).is_empty() {
        let chunk = vals.as_ref().unwrap()[0];
        if let Some(size) = extent {
            let count = size / chunk + i64::from(size % chunk != 0);
            len = Some(count);
            if let Some(ax) = ax {
                dims[ax] = if size % chunk == 0 {
                    dim(chunk)
                } else if count == 1 {
                    dim(size)
                } else {
                    unknown()
                };
            }
            if count <= LIMIT as i64 {
                let mut items = vec![];
                for i in 0..count {
                    let mut ds = original.clone();
                    ds[ax.unwrap()] = dim(chunk.min(size - i * chunk));
                    items.push(tensor_type(dtype, &ds, &v["tensor_type_ids"]));
                }
                inv = Some(items);
            }
        } else {
            reason = "split_to_sequence_axis_extent_unknown";
            if let Some(ax) = ax {
                dims[ax] = dim(chunk);
            }
        }
    } else {
        let vals = vals.unwrap();
        len = Some(vals.len() as i64);
        if let Some(size) = extent {
            if vals.iter().map(|n| *n as i128).sum::<i128>() != size as i128 {
                return fail(v, "split_to_sequence_split_sum_mismatch");
            }
        } else {
            reason = "split_to_sequence_axis_extent_unknown";
        }
        if let Some(ax) = ax {
            dims[ax] = if vals.iter().all(|n| *n == vals[0]) {
                dim(vals[0])
            } else {
                unknown()
            };
            if vals.len() <= LIMIT {
                inv = Some(
                    vals.iter()
                        .map(|value| {
                            let mut ds = original.clone();
                            ds[ax] = dim(*value);
                            tensor_type(dtype, &ds, &v["tensor_type_ids"])
                        })
                        .collect(),
                );
            }
        }
    }
    let mut elem = tensor_type(dtype, &dims, &v["tensor_type_ids"]);
    if r.is_none() {
        unrank(&mut elem);
    }
    emit(v, wrapper(&elem, "sequence"), state(len, inv), reason)
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let op = string(&n["opType"]);
    let x = input(v, 0);
    let t = ty(v, x);
    if op == "Identity" {
        if t.is_null() || t["kind"] == "tensor" {
            return Ok(Value::Null);
        }
        if !(seq(&t) || t["kind"] == "optional" && optional(&t)) {
            return Ok(fail(v, "identity_non_dense_type_not_supported"));
        }
        return Ok(emit(v, t, copy(x), ""));
    }
    let result = match op {
        "SequenceEmpty" => {
            let dt = string(&v["empty_dtype"]);
            if dt.is_empty() || dt == "UNDEFINED" || dt.starts_with("TYPE_") {
                fail(v, "sequence_empty_dtype_invalid")
            } else {
                let mut t = tensor_type(dt, &[], &v["tensor_type_ids"]);
                unrank(&mut t);
                emit(v, wrapper(&t, "sequence"), state(Some(0), Some(vec![])), "")
            }
        }
        "SequenceConstruct" => {
            let types = rows(&n["inputs"])
                .iter()
                .enumerate()
                .filter(|(_, n)| truth(n))
                .map(|(i, _)| ty(v, input(v, i)))
                .collect::<Vec<_>>();
            if types.is_empty() {
                fail(v, "sequence_construct_input_missing")
            } else if types.iter().any(|t| t["kind"] != "tensor") {
                fail(v, "sequence_construct_input_not_tensor")
            } else {
                let u = union(v, &types)?;
                if u["status"] != "pass" {
                    fail(v, &format!("sequence_construct_{}", string(&u["reason"])))
                } else {
                    let mut s = state(
                        Some(types.len() as i64),
                        if types.len() <= LIMIT {
                            Some(types)
                        } else {
                            None
                        },
                    );
                    if s["sequenceElementInventoryStatus"] != "assessed_exact" {
                        s["sequenceElementInventoryStatus"] = json!("not_materialized_over_limit");
                    }
                    emit(v, wrapper(&u["type"], "sequence"), s, "")
                }
            }
        }
        "SequenceLength" | "SequenceInsert" | "SequenceErase" | "SequenceAt"
        | "ConcatFromSequence" => {
            if !seq(&t) {
                let snake = match op {
                    "SequenceLength" => "sequence_length",
                    "SequenceInsert" => "sequence_insert",
                    "SequenceErase" => "sequence_erase",
                    "SequenceAt" => "sequence_at",
                    _ => "concat_from_sequence",
                };
                return Ok(fail(v, &format!("{snake}_input_not_typed_tensor_sequence")));
            }
            let len = length(x);
            if op == "SequenceLength" {
                emit(
                    v,
                    tensor_type("INT64", &[], &v["tensor_type_ids"]),
                    len.map(|l| scalar(l, "SequenceLength"))
                        .unwrap_or(json!({})),
                    if len.is_some() {
                        ""
                    } else {
                        "sequence_length_runtime_unknown"
                    },
                )
            } else if op == "ConcatFromSequence" {
                concat(v, &t, x)
            } else {
                let insert = op == "SequenceInsert";
                let new = ty(v, input(v, 1));
                if insert && new["kind"] != "tensor" {
                    return Ok(fail(v, "sequence_insert_value_not_tensor"));
                }
                let u = if insert {
                    union(v, &[t["elementType"].clone(), new.clone()])?
                } else {
                    json!({"status":"pass","type":t["elementType"]})
                };
                if u["status"] != "pass" {
                    return Ok(fail(
                        v,
                        &format!("sequence_insert_{}", string(&u["reason"])),
                    ));
                }
                if op == "SequenceErase" && len == Some(0) {
                    return Ok(fail(v, "sequence_erase_empty_sequence"));
                }
                let (pos, reason) = match position(
                    v,
                    if insert { 2 } else { 1 },
                    len,
                    insert,
                    op == "SequenceAt",
                ) {
                    Ok(p) => p,
                    Err(e) => return Ok(fail(v, e)),
                };
                let inv = inventory(x);
                if op == "SequenceAt" {
                    let t = inv
                        .as_ref()
                        .and_then(|a| {
                            pos.and_then(|p| usize::try_from(p).ok())
                                .and_then(|p| a.get(p))
                        })
                        .unwrap_or(&t["elementType"])
                        .clone();
                    emit(v, t, json!({}), reason)
                } else {
                    let inv = if let (Some(mut inv), Some(pos)) = (inv, pos) {
                        if pos < 0 || pos > inv.len() as i64 {
                            None
                        } else if insert && inv.len() < LIMIT {
                            inv.insert(pos as usize, new);
                            Some(inv)
                        } else if !insert && pos < inv.len() as i64 {
                            inv.remove(pos as usize);
                            Some(inv)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let t = if insert {
                        wrapper(&u["type"], "sequence")
                    } else {
                        t
                    };
                    emit(
                        v,
                        t,
                        state(
                            len.and_then(|l| {
                                if insert {
                                    l.checked_add(1)
                                } else {
                                    l.checked_sub(1)
                                }
                            }),
                            inv,
                        ),
                        if !reason.is_empty() {
                            reason
                        } else if len.is_none() {
                            "sequence_length_runtime_unknown"
                        } else {
                            ""
                        },
                    )
                }
            }
        }
        "SplitToSequence" => split(v, &t),
        "Optional" => {
            let count = rows(&n["inputs"]).iter().filter(|n| truth(n)).count();
            let e = if count == 1 {
                t
            } else if count == 0 {
                attr(n, "type")["typeProto"].clone()
            } else {
                Value::Null
            };
            if !["tensor", "sequence"].contains(&string(&e["kind"])) || !known(&e) {
                fail(v, "optional_element_type_missing_or_unsupported")
            } else {
                emit(
                    v,
                    wrapper(&e, "optional"),
                    json!({"optionalPresenceStatus":"assessed_exact","optionalPresence":count==1}),
                    "",
                )
            }
        }
        "OptionalHasElement" => {
            let count = rows(&n["inputs"]).iter().filter(|n| truth(n)).count();
            if count > 1 {
                return Ok(fail(v, "optional_has_element_input_cardinality_invalid"));
            }
            if count == 1 && !optional(&t) {
                return Ok(fail(v, "optional_has_element_input_type_invalid"));
            }
            let present = if count == 0 {
                Some(false)
            } else if t["kind"] == "optional" {
                presence(x)
            } else {
                Some(true)
            };
            emit(
                v,
                tensor_type("BOOL", &[], &v["tensor_type_ids"]),
                present
                    .map(|p| scalar(i64::from(p), "OptionalHasElement"))
                    .unwrap_or(json!({})),
                if present.is_some() {
                    ""
                } else {
                    "optional_presence_runtime_unknown"
                },
            )
        }
        "OptionalGetElement" => {
            if !optional(&t) {
                fail(v, "optional_get_element_input_type_invalid")
            } else if t["kind"] != "optional" {
                emit(v, t, copy(x), "")
            } else if presence(x) == Some(false) {
                fail(v, "optional_get_element_provably_empty")
            } else {
                emit(
                    v,
                    t["elementType"].clone(),
                    json!({}),
                    if presence(x).is_some() {
                        ""
                    } else {
                        "optional_presence_runtime_unknown"
                    },
                )
            }
        }
        _ => Value::Null,
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn huge_scalar_split_has_bounded_inventory() {
        let v = json!({"operation":"container","scope":"main_graph","node_index":0,"imported_opset":18,"node":{"opType":"SplitToSequence","inputs":["x","split"],"outputs":["out"],"attributes":[]},"tensor_type_ids":{"FLOAT32":1,"INT64":7},"tensors":{"x":{"dtype":"FLOAT32","shape":[9007199254740991i64],"shapeDeclared":true},"split":{"dtype":"INT64","shape":[],"shapeDeclared":true,"staticValuesComplete":true,"staticValues":[1]}}});
        let r = run(&v).unwrap();
        assert_eq!(r["status"], "pass");
        let p = &r["result"]["outputs"][0][1];
        assert_eq!(p["sequenceLength"], 9007199254740991i64);
        assert_eq!(p["sequenceElementInventoryStatus"], "not_assessed");
        assert_eq!(p["sequenceElementTypes"], json!([]));
    }
}
