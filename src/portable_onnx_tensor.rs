//! Tensor shape-transform rules with explicit dynamic-control boundaries.
use crate::portable_onnx_node::{
    ai_exact, attr_ints, exact_dims, inherited, kd, kernel, set, shape, static_numbers,
};
use crate::portable_onnx_semantics::*;
use crate::portable_validation::{rows, string, R};
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn integers(t: &Value) -> Option<Vec<i64>> {
    static_numbers(t)?.iter().map(safe).collect()
}
fn axis(n: i64, r: usize) -> Option<usize> {
    let x = if n < 0 { n + r as i64 } else { n };
    if x >= 0 && x < r as i64 {
        Some(x as usize)
    } else {
        None
    }
}
fn axes(values: &[i64], r: usize, sort: bool) -> Option<Vec<usize>> {
    let mut v = values
        .iter()
        .map(|n| axis(*n, r))
        .collect::<Option<Vec<_>>>()?;
    if v.iter().collect::<BTreeSet<_>>().len() != v.len() {
        return None;
    }
    if sort {
        v.sort();
    }
    Some(v)
}
fn ai_list(n: &Value, k: &str) -> Vec<i64> {
    attr_ints(n, k)
        .iter()
        .map(safe)
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}
fn one(v: &Value, s: Value, dt: &str, values: Value, error: &str) -> Value {
    let mut out = vec![];
    if set(v, &mut out, 0, &s, dt, values) {
        result(out)
    } else {
        unresolved(error)
    }
}
fn invalid(reason: &str, details: Value) -> Value {
    json!({"status":"invalid","outputs":[],"reason":reason,"details":details})
}
fn concrete(d: &Value) -> bool {
    dv(d).is_some_and(|n| n >= 0)
}
fn known(d: &Value) -> bool {
    concrete(d) || d["kind"] == "symbolic" && !string(&d["parameter"]).is_empty()
}
fn count(d: &[Value]) -> Option<i64> {
    let nums = d.iter().map(dv).collect::<Option<Vec<_>>>()?;
    if nums.iter().any(|n| *n < 0) {
        return None;
    }
    if nums.contains(&0) {
        return Some(0);
    }
    nums.iter()
        .try_fold(1i64, |a, b| a.checked_mul(*b))
        .filter(|n| *n <= SAFE)
}
fn repr(ds: &[Value]) -> Vec<Value> {
    ds.iter().map(|d|json!({"kind":d["kind"].as_str().unwrap_or("unknown"),"value":dv(d),"parameter":if d["kind"]=="symbolic"{Some(string(&d["parameter"]))}else{None}})).collect()
}

pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let op = string(&n["opType"]);
    let (x, y) = (input(v, 0), input(v, 1));
    let dt = kd(x);
    let mut out = vec![];
    match op {
        "Concat" => {
            let inputs = rows(&n["inputs"])
                .iter()
                .filter(|n| truth(n))
                .map(|name| &v["tensors"][string(name)])
                .collect::<Vec<_>>();
            let dtypes = inputs
                .iter()
                .map(|t| kd(t))
                .filter(|d| !d.is_empty())
                .collect::<Vec<_>>();
            if dtypes.len() == inputs.len() && dtypes.iter().collect::<BTreeSet<_>>().len() > 1 {
                return Ok(invalid(
                    "concat_input_dtype_mismatch",
                    json!({"input_dtypes":dtypes}),
                ));
            }
            if inputs.is_empty() || inputs.iter().any(|t| shape(t).is_null()) {
                return Ok(unresolved("concat_input_shape_unknown"));
            }
            let shapes = inputs.iter().map(|t| dimensions(t)).collect::<Vec<_>>();
            let rank = shapes[0].len();
            if shapes.iter().any(|s| s.len() != rank) {
                return Ok(unresolved("concat_rank_mismatch"));
            }
            let Some(ax) = axis(ai_exact(n, "axis", 0), rank) else {
                return Ok(unresolved("concat_axis_invalid"));
            };
            let s = kernel("concat", json!({"shapes":shapes,"axis":ax}))?;
            if s.is_null() {
                return Ok(unresolved("concat_non_axis_dimension_mismatch"));
            }
            let vals = inputs
                .iter()
                .map(|t| exact_dims(t))
                .collect::<Option<Vec<_>>>();
            let vals = if rank == 1 && ax == 0 {
                vals.map(|vals|json!({"dimensions":vals.into_iter().flatten().collect::<Vec<_>>(),"source":"Concat"})).unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            Ok(one(v, s, kd(inputs[0]), vals, "concat_output_unresolved"))
        }
        "Reshape" => {
            if shape(x).is_null() {
                return Ok(unresolved("reshape_data_shape_unknown"));
            }
            let r = kernel(
                "reshape",
                json!({"input":shape(x),"requested":exact_dims(y),"allow_zero":ai_exact(n,"allowzero",0)==1}),
            )?;
            if r["shape"].is_null() {
                return Ok(unresolved(string(&r["reason"])));
            }
            Ok(one(
                v,
                r["shape"].clone(),
                dt,
                inherited(x, op),
                "reshape_output_unresolved",
            ))
        }
        "Unsqueeze" => {
            if shape(x).is_null() {
                return Ok(unresolved("unsqueeze_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let ax = integers(y).unwrap_or_else(|| ai_list(n, "axes"));
            if ax.is_empty() {
                return Ok(unresolved("unsqueeze_axes_not_static"));
            }
            let r = ds.len() + ax.len();
            let Some(ax) = axes(&ax, r, true) else {
                return Ok(unresolved("unsqueeze_axes_invalid"));
            };
            let mut result = vec![];
            let mut i = 0;
            for j in 0..r {
                if ax.contains(&j) {
                    result.push(dim(1));
                } else {
                    result.push(ds[i].clone());
                    i += 1;
                }
            }
            Ok(one(
                v,
                json!(result),
                dt,
                inherited(x, op),
                "unsqueeze_output_unresolved",
            ))
        }
        "Gather" => {
            if shape(x).is_null() || shape(y).is_null() {
                return Ok(unresolved("gather_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let ids = dimensions(y);
            let raw = ai_exact(n, "axis", 0);
            let Some(ax) = axis(raw, ds.len()) else {
                return Ok(invalid(
                    "gather_axis_out_of_range",
                    json!({"axis":raw,"data_rank":ds.len()}),
                ));
            };
            let mut s = ds[..ax].to_vec();
            s.extend(ids);
            s.extend_from_slice(&ds[ax + 1..]);
            let mut vals = Value::Null;
            if let (Some(source), Some(indices)) = (exact_dims(x), integers(y)) {
                if ds.len() == 1 && ax == 0 {
                    let mut selected = vec![];
                    for raw in indices {
                        let i = if raw < 0 {
                            raw + source.len() as i64
                        } else {
                            raw
                        };
                        if i < 0 || i >= source.len() as i64 {
                            return Ok(invalid(
                                "gather_index_out_of_range",
                                json!({"index":i,"axis_extent":source.len()}),
                            ));
                        }
                        selected.push(source[i as usize].clone());
                    }
                    vals = json!({"dimensions":selected,"source":"Gather"});
                }
            }
            Ok(one(v, json!(s), dt, vals, "gather_output_unresolved"))
        }
        "GatherElements" => {
            let (a, b) = (dimensions(x), dimensions(y));
            if shape(x).is_null()
                || shape(y).is_null()
                || a.iter().chain(&b).any(|d| !concrete(d))
                || a.len() != b.len()
            {
                return Ok(unresolved("gather_elements_rank_unknown_or_mismatch"));
            }
            Ok(one(
                v,
                json!(b),
                dt,
                Value::Null,
                "gather_elements_output_unresolved",
            ))
        }
        "GatherND" => {
            if shape(x).is_null() || shape(y).is_null() || rank(y) == Some(0) {
                return Ok(unresolved("gather_nd_input_rank_unknown"));
            }
            let (a, b) = (dimensions(x), dimensions(y));
            let batch = ai_exact(n, "batch_dims", 0);
            if batch < 0 || batch >= a.len().min(b.len()) as i64 {
                return Ok(invalid(
                    "gather_nd_batch_dims_invalid",
                    json!({"batch_dims":batch,"data_rank":a.len(),"indices_rank":b.len()}),
                ));
            }
            let Some(k) = dv(b.last().unwrap()) else {
                return Ok(unresolved("gather_nd_index_tuple_length_runtime_unknown"));
            };
            if k < 0 || k > a.len() as i64 - batch {
                return Ok(invalid(
                    "gather_nd_index_tuple_length_invalid",
                    json!({"index_tuple_length":k,"maximum":a.len()as i64-batch}),
                ));
            }
            for i in 0..batch as usize {
                if let (Some(a), Some(b)) = (dv(&a[i]), dv(&b[i])) {
                    if a != b {
                        return Ok(invalid(
                            "gather_nd_batch_dimension_mismatch",
                            json!({"axis":i,"data_dimension":a,"indices_dimension":b}),
                        ));
                    }
                }
            }
            let mut s = b[..b.len() - 1].to_vec();
            s.extend_from_slice(&a[(batch + k) as usize..]);
            Ok(one(
                v,
                json!(s),
                dt,
                Value::Null,
                "gather_nd_output_unresolved",
            ))
        }
        "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceMax"
        | "ReduceMean" | "ReduceMin" | "ReduceProd" | "ReduceSum" | "ReduceSumSquare" => {
            if shape(x).is_null() {
                return Ok(unresolved("reduce_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let explicit = truth(&n["inputs"][1]);
            let mut ax = integers(y);
            if explicit && ax.is_none() {
                return Ok(unresolved("reduce_axes_runtime_unknown"));
            }
            if ax.is_none() {
                ax = Some(ai_list(n, "axes"));
            }
            let mut ax = ax.unwrap();
            if ax.is_empty() {
                if ai_exact(n, "noop_with_empty_axes", 0) == 1
                    && (explicit || !attr(n, "axes").is_null())
                {
                    return Ok(one(
                        v,
                        json!(ds),
                        dt,
                        inherited(x, op),
                        "reduce_noop_output_unresolved",
                    ));
                }
                ax = (0..ds.len() as i64).collect();
            }
            let Some(ax) = axes(&ax, ds.len(), true) else {
                return Ok(unresolved("reduce_axes_invalid"));
            };
            let keep = ai_exact(n, "keepdims", 1) != 0;
            let s = ds
                .into_iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    if ax.contains(&i) {
                        if keep {
                            Some(dim(1))
                        } else {
                            None
                        }
                    } else {
                        Some(d)
                    }
                })
                .collect::<Vec<_>>();
            Ok(one(
                v,
                json!(s),
                dt,
                Value::Null,
                "reduce_output_unresolved",
            ))
        }
        "ArgMax" | "ArgMin" => {
            if shape(x).is_null() {
                return Ok(unresolved("arg_reduce_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let Some(ax) = axis(ai_exact(n, "axis", 0), ds.len()) else {
                return Ok(unresolved("arg_reduce_axis_invalid"));
            };
            let keep = ai_exact(n, "keepdims", 1) != 0;
            let s = ds
                .into_iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    if i == ax {
                        if keep {
                            Some(dim(1))
                        } else {
                            None
                        }
                    } else {
                        Some(d)
                    }
                })
                .collect::<Vec<_>>();
            Ok(one(
                v,
                json!(s),
                "INT64",
                Value::Null,
                "arg_reduce_output_unresolved",
            ))
        }
        "Tile" => {
            if shape(x).is_null() {
                return Ok(unresolved("tile_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let Some(repeats) =
                integers(y).filter(|a| a.len() == ds.len() && a.iter().all(|n| *n >= 0))
            else {
                return Ok(unresolved("tile_repeats_not_static_or_invalid"));
            };
            let mut s = vec![];
            for (d, r) in ds.iter().zip(repeats) {
                s.push(kernel("factor", json!({"left":d,"factor":r}))?);
            }
            Ok(one(v, json!(s), dt, Value::Null, "tile_output_unresolved"))
        }
        "Split" => {
            if shape(x).is_null() || rows(&n["outputs"]).is_empty() {
                return Ok(unresolved("split_input_shape_unknown"));
            }
            let ds = dimensions(x);
            let Some(ax) = axis(ai_exact(n, "axis", 0), ds.len()) else {
                return Ok(unresolved("split_axis_invalid"));
            };
            let explicit = truth(&n["inputs"][1]);
            let split = integers(y);
            if explicit && split.is_none() {
                return Ok(unresolved("split_sizes_runtime_unknown"));
            }
            let mut split = split.unwrap_or_else(|| ai_list(n, "split"));
            let output_count = rows(&n["outputs"]).iter().filter(|n| truth(n)).count();
            let size = dv(&ds[ax]);
            if split.is_empty() {
                let count = ai_exact(n, "num_outputs", output_count as i64);
                if count <= 0 || size.is_none() || size.unwrap() % count != 0 {
                    return Ok(unresolved("split_equal_partition_not_integral"));
                }
                if count != output_count as i64 {
                    return Ok(unresolved("split_sizes_do_not_conserve_axis"));
                }
                split = vec![size.unwrap() / count; output_count];
            }
            if split.len() != output_count
                || split.iter().any(|n| *n < 0)
                || size.is_some_and(|s| split.iter().map(|n| *n as i128).sum::<i128>() != s as i128)
            {
                return Ok(unresolved("split_sizes_do_not_conserve_axis"));
            }
            let source = if ds.len() == 1 && ax == 0 {
                exact_dims(x)
            } else {
                None
            };
            let mut offset = 0usize;
            let mut part = 0;
            for (i, name) in rows(&n["outputs"]).iter().enumerate() {
                if !truth(name) {
                    continue;
                }
                let extent = split[part];
                part += 1;
                let mut s = ds.clone();
                s[ax] = dim(extent);
                let vals = source
                    .as_ref()
                    .map(|d| {
                        let end = offset
                            .saturating_add(usize::try_from(extent).unwrap_or(usize::MAX))
                            .min(d.len());
                        let vals = d[offset.min(d.len())..end].to_vec();
                        offset = end;
                        json!({"dimensions":vals,"source":"Split"})
                    })
                    .unwrap_or(Value::Null);
                set(v, &mut out, i, &json!(s), dt, vals);
            }
            Ok(result(out))
        }
        "Shape" => {
            if shape(x).is_null() {
                return Ok(unresolved("shape_input_rank_or_dimensions_unknown"));
            }
            let ds = dimensions(x);
            let r = ds.len() as i64;
            let bound = |n: i64| {
                if n < 0 {
                    (n + r).clamp(0, r)
                } else {
                    n.clamp(0, r)
                }
            };
            let start = bound(ai_exact(n, "start", 0));
            let end = bound(ai_exact(n, "end", r));
            if start > end {
                return Ok(unresolved("shape_start_exceeds_end"));
            }
            let vals = &ds[start as usize..end as usize];
            Ok(one(
                v,
                json!([dim(vals.len() as i64)]),
                "INT64",
                json!({"dimensions":vals,"source":"Shape"}),
                "shape_output_unresolved",
            ))
        }
        "Size" => {
            if shape(x).is_null()
                || dimensions(x).iter().any(|d| !concrete(d))
                || x["conditionalShapeContract"]["status"] == "assessed_partial"
            {
                return Ok(unresolved("size_input_shape_unknown"));
            }
            if let Some(n) = count(&dimensions(x)) {
                Ok(one(
                    v,
                    json!([]),
                    "INT64",
                    json!({"values":[n],"source":"Size"}),
                    "size_output_outside_exact_static_integer_range",
                ))
            } else {
                Ok(unresolved("size_output_outside_exact_static_integer_range"))
            }
        }
        "Expand" => {
            let target = exact_dims(y);
            if shape(x).is_null() || target.as_ref().is_none_or(|t| t.iter().any(|d| !known(d))) {
                return Ok(unresolved("expand_shape_input_not_static"));
            }
            let target = target.unwrap();
            let s = kernel("broadcast", json!({"shapes":[dimensions(x),target]}))?;
            if s.is_null() {
                return Ok(invalid(
                    "expand_target_not_broadcast_compatible",
                    json!({"input_shape":repr(&dimensions(x)),"target_shape":repr(&target)}),
                ));
            }
            Ok(one(
                v,
                s,
                dt,
                Value::Null,
                "expand_target_not_broadcast_compatible",
            ))
        }
        "Resize" => {
            if shape(x).is_null() {
                return Ok(unresolved("resize_input_rank_unknown"));
            }
            let ds = dimensions(x);
            let mut s =
                exact_dims(input(v, 3)).filter(|t| t.len() == ds.len() && t.iter().all(known));
            if s.is_none() {
                if let (Some(scales), Some(shape)) = (
                    static_numbers(input(v, 2)),
                    ds.iter().map(dv).collect::<Option<Vec<_>>>(),
                ) {
                    if scales.len() == ds.len()
                        && scales.iter().all(|v| v.as_f64().is_some_and(|n| n > 0.))
                    {
                        s = scales
                            .iter()
                            .zip(shape)
                            .map(|(scale, d)| {
                                let n = (d as f64 * scale.as_f64().unwrap()).floor();
                                if n.is_finite() && n >= 0. && n <= SAFE as f64 {
                                    Some(dim(n as i64))
                                } else {
                                    None
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                    }
                }
            }
            Ok(one(
                v,
                s.map(|s| json!(s)).unwrap_or(Value::Null),
                dt,
                Value::Null,
                "resize_sizes_or_scales_not_static",
            ))
        }
        "TopK" => {
            if shape(x).is_null() || dimensions(x).iter().any(|d| !concrete(d)) {
                return Ok(unresolved("topk_input_shape_unknown"));
            }
            let vals = integers(y);
            let k = vals.as_ref().filter(|v| v.len() == 1).map(|v| v[0]);
            let ds = dimensions(x);
            let ax = axis(ai_exact(n, "axis", -1), ds.len());
            if k.is_none()
                || k.unwrap() < 0
                || ax.is_none()
                || k.unwrap() > dv(&ds[ax.unwrap()]).unwrap()
            {
                return Ok(unresolved("topk_k_or_axis_invalid"));
            }
            let mut ds = ds;
            ds[ax.unwrap()] = dim(k.unwrap());
            set(v, &mut out, 0, &json!(ds), dt, Value::Null);
            set(v, &mut out, 1, &json!(ds), "INT64", Value::Null);
            Ok(result(out))
        }
        "DepthToSpace" => {
            if rank(x) != Some(4) {
                return Ok(unresolved("depth_to_space_input_rank_not_four"));
            }
            let ds = dimensions(x);
            let block = ai_exact(n, "blocksize", 0);
            let channels = dv(&ds[1]);
            let square = block.checked_mul(block);
            if block <= 0
                || channels.is_none()
                || square.is_none_or(|n| n == 0 || channels.unwrap() % n != 0)
            {
                return Ok(unresolved(
                    "depth_to_space_block_or_channel_contract_invalid",
                ));
            }
            let h = kernel("factor", json!({"left":ds[2],"factor":block}))?;
            let w = kernel("factor", json!({"left":ds[3],"factor":block}))?;
            Ok(one(
                v,
                json!([ds[0], dim(channels.unwrap() / square.unwrap()), h, w]),
                dt,
                Value::Null,
                "depth_to_space_output_unresolved",
            ))
        }
        "ConstantOfShape" => {
            let s = exact_dims(x);
            if s.as_ref().is_none_or(|s| s.iter().any(|d| !known(d))) {
                return Ok(unresolved("constant_of_shape_input_not_static"));
            }
            let s = s.unwrap();
            let t = &attr(n, "value")["tensor"];
            let dtype = t["dtype"].as_str().unwrap_or("FLOAT32");
            let fill = if t.is_null() {
                Some(vec![json!(0)])
            } else {
                static_numbers(t)
            };
            let vals = if let (Some(size), Some(fill)) = (
                count(&s).filter(|n| *n <= 65536),
                fill.filter(|f| f.len() == 1),
            ) {
                json!({"values":vec![fill[0].clone();size as usize],"source":"ConstantOfShape"})
            } else {
                Value::Null
            };
            Ok(one(
                v,
                json!(s),
                dtype,
                vals,
                "constant_of_shape_output_unresolved",
            ))
        }
        "Constant" => {
            let t = &attr(n, "value")["tensor"];
            if !t.is_null() {
                let vals = if t["initializerIntegerValuesExactComplete"] == true {
                    json!({"exactDecimals":t["initializerIntegerValuesExactDecimals"],"source":"Constant"})
                } else if let Some(values) = static_numbers(t) {
                    json!({"values":values,"source":"Constant"})
                } else if t["staticValuesCanonicalTextComplete"] == true {
                    json!({"canonicalTexts":t["staticValuesCanonicalTexts"],"source":"Constant"})
                } else {
                    Value::Null
                };
                return Ok(one(
                    v,
                    json!(dimensions(
                        &json!({"shape":t["shape"],"shapeDeclared":true})
                    )),
                    t["dtype"].as_str().unwrap_or("UNKNOWN"),
                    vals,
                    "constant_tensor_contract_invalid",
                ));
            }
            for (k, dt, field, list) in [
                ("value_int", "INT64", "i", false),
                ("value_ints", "INT64", "ints", true),
                ("value_float", "FLOAT32", "f", false),
                ("value_floats", "FLOAT32", "floats", true),
            ] {
                let a = attr(n, k);
                if a.is_null() {
                    continue;
                }
                let values = if list {
                    rows(&a[field]).to_vec()
                } else {
                    vec![a[field].clone()]
                };
                let s = if list {
                    json!([dim(values.len() as i64)])
                } else {
                    json!([])
                };
                let vals = if dt == "INT64" {
                    let exact = if list {
                        if a["intExactDecimals"].is_array() {
                            rows(&a["intExactDecimals"]).to_vec()
                        } else {
                            values
                                .iter()
                                .filter_map(|v| safe(v).map(|n| json!(n.to_string())))
                                .collect()
                        }
                    } else {
                        vec![if a["iExactDecimal"].is_string() {
                            a["iExactDecimal"].clone()
                        } else {
                            safe(&a["i"])
                                .map(|n| json!(n.to_string()))
                                .unwrap_or(Value::Null)
                        }]
                    };
                    if exact.len() != values.len()
                        || exact
                            .iter()
                            .any(|v| v.as_str().and_then(|s| s.parse::<i64>().ok()).is_none())
                    {
                        return Ok(unresolved("constant_attribute_integer_not_exact"));
                    }
                    json!({"exactDecimals":exact,"source":"Constant"})
                } else {
                    let text = if list {
                        rows(&a["floatsText"]).to_vec()
                    } else {
                        vec![a["fText"].clone()]
                    };
                    if values
                        .iter()
                        .all(|n| n.as_f64().is_some_and(f64::is_finite))
                    {
                        json!({"values":values,"source":"Constant"})
                    } else if text.len() == values.len()
                        && text
                            .iter()
                            .all(|n| crate::portable_onnx_ml::float(n).is_some())
                    {
                        json!({"canonicalTexts":text,"source":"Constant"})
                    } else {
                        Value::Null
                    }
                };
                return Ok(one(v, s, dt, vals, "constant_attribute_invalid"));
            }
            Ok(unresolved(
                "constant_value_attribute_unsupported_or_missing",
            ))
        }
        _ => crate::portable_onnx_dynamic::run(v),
    }
}
