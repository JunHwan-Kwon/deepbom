//! TfIdfVectorizer-9: bounded, source-pinned ORT CPU accumulation order.
use crate::portable_onnx_ml::{float, templates};
use crate::portable_onnx_semantics::{attr, input, safe, SAFE};
use crate::portable_validation::{rows, R};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
const OUTPUT_LIMIT: u64 = 65_536;
const WORK_LIMIT: u64 = 5_000_000;
fn unique(v: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    v.into_iter().filter(|x| seen.insert(x.clone())).collect()
}
fn list(n: &Value, name: &str) -> (Option<Vec<i64>>, String) {
    let a = attr(n, name);
    if a.is_null() {
        return (None, format!("tfidf_{name}_missing"));
    }
    let x: Option<Vec<_>> = rows(&a["ints"]).iter().map(safe).collect();
    let reason = if x.is_none() {
        format!("tfidf_{name}_outside_javascript_safe_integer")
    } else {
        String::new()
    };
    (x, reason)
}
fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array()?
        .iter()
        .map(|s| s.as_str().map(str::to_owned))
        .collect()
}
fn decimals(v: &Value) -> Option<Vec<String>> {
    strings(v)?
        .into_iter()
        .map(|s| s.parse::<i64>().ok().map(|n| n.to_string()))
        .collect()
}
fn product(shape: &[u64]) -> Option<u64> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .filter(|x| *x <= SAFE as u64)
}
#[derive(Default)]
struct Definitions {
    total: usize,
    active: Vec<(usize, Vec<String>)>,
    failures: Vec<String>,
    prefix: i64,
    active_dupes: usize,
    inactive_dupes: usize,
}
fn definitions(pool: &[String], counts: &[i64], minimum: i64, maximum: i64) -> Definitions {
    let mut d = Definitions {
        prefix: counts.first().copied().unwrap_or(0).max(0),
        ..Default::default()
    };
    let (mut active, mut inactive) = (BTreeSet::new(), BTreeSet::new());
    for (level, &start) in counts.iter().enumerate() {
        let size = level + 1;
        let end = counts.get(level + 1).copied().unwrap_or(pool.len() as i64);
        if start < 0 || end < start || end > pool.len() as i64 {
            d.failures
                .push("tfidf_ngram_counts_out_of_pool_bounds".into());
            continue;
        }
        if (end - start) as usize % size != 0 {
            d.failures
                .push("tfidf_pool_level_does_not_contain_whole_ngrams".into());
            continue;
        }
        for offset in (start as usize..end as usize).step_by(size) {
            let tokens = pool[offset..offset + size].to_vec();
            let relevant = size as i64 >= minimum && size as i64 <= maximum;
            let fresh = if relevant {
                active.insert(tokens.clone())
            } else {
                inactive.insert(tokens.clone())
            };
            if !fresh {
                if relevant {
                    d.active_dupes += 1;
                } else {
                    d.inactive_dupes += 1;
                }
            }
            if relevant {
                d.active.push((d.total, tokens));
            }
            d.total += 1;
        }
    }
    if d.active_dupes > 0 {
        d.failures
            .push("tfidf_duplicate_active_ngram_rejected_by_pinned_ort".into());
    }
    d.failures = unique(d.failures);
    d
}
#[derive(Default)]
struct Branch {
    children: BTreeMap<String, usize>,
    definition: Option<usize>,
}
struct Execution {
    frequency: Vec<u64>,
    output: Vec<f32>,
    reference: Vec<f32>,
    divergent: Vec<usize>,
    negative: Vec<usize>,
    matches: u64,
}
fn execute(
    values: &[String],
    shape: &[u64],
    defs: &Definitions,
    indexes: &[i64],
    weights: &[f32],
    present: bool,
    mode: &str,
    skip: u64,
    gram: u64,
    width: usize,
) -> Execution {
    let batches = if shape.len() == 1 {
        1
    } else {
        shape[0] as usize
    };
    let columns = *shape.last().unwrap() as usize;
    let mut tree = vec![Branch::default()];
    for (index, tokens) in &defs.active {
        let mut branch = 0;
        for token in tokens {
            let next = if let Some(&i) = tree[branch].children.get(token) {
                i
            } else {
                let i = tree.len();
                tree.push(Branch::default());
                tree[branch].children.insert(token.clone(), i);
                i
            };
            branch = next;
        }
        tree[branch].definition = Some(*index);
    }
    let mut e = Execution {
        frequency: vec![0; batches * width],
        output: vec![0.; batches * width],
        reference: vec![],
        divergent: vec![],
        negative: vec![],
        matches: 0,
    };
    // Empty rows and inactive vocabularies have zero work. Do not iterate a huge
    // skip bound, nor allocate an unused batches × definition-count matrix.
    if columns > 0 && !defs.active.is_empty() {
        for row in 0..batches {
            let mut minimum = defs
                .active
                .iter()
                .map(|(_, t)| t.len() as u64)
                .min()
                .unwrap();
            for distance in 1..=skip + 1 {
                if minimum > gram {
                    break;
                }
                for start in 0..columns {
                    if start as u64 + distance * (minimum - 1) >= columns as u64 {
                        break;
                    }
                    let mut branch = 0;
                    let mut position = start as u64;
                    for size in 1..=gram {
                        if position >= columns as u64 {
                            break;
                        }
                        let Some(&next) = tree[branch]
                            .children
                            .get(&values[row * columns + position as usize])
                        else {
                            break;
                        };
                        branch = next;
                        if size >= minimum {
                            if let Some(index) = tree[branch].definition {
                                let coordinate = indexes[index] as usize;
                                let flat = row * width + coordinate;
                                e.frequency[flat] += 1;
                                e.matches += 1;
                                let w = if present { weights[coordinate] } else { 1. };
                                e.output[flat] = match mode {
                                    "TF" => e.output[flat] + 1.,
                                    "IDF" => w,
                                    _ => e.output[flat] + w,
                                };
                            }
                        }
                        position += distance;
                    }
                }
                if minimum == 1 {
                    minimum = 2;
                }
            }
        }
    }
    e.reference = e
        .frequency
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let w = if present { weights[i % width] } else { 1. };
            match mode {
                "TF" => f as f32,
                "IDF" => {
                    if f > 0 {
                        w
                    } else {
                        0.
                    }
                }
                _ => (f as f64 * w as f64) as f32,
            }
        })
        .collect();
    for (i, &x) in e.output.iter().enumerate() {
        if x.to_bits() != e.reference[i].to_bits() && !(x.is_nan() && e.reference[i].is_nan()) {
            e.divergent.push(i);
        }
        if x == 0. && x.is_sign_negative() {
            e.negative.push(i);
        }
    }
    e
}
pub(crate) fn run(v: &Value) -> R<Value> {
    let n = &v["node"];
    let t = input(v, 0);
    let mode = attr(n, "mode")["s"].as_str().unwrap_or("");
    let minimum = safe(&attr(n, "min_gram_length")["i"]);
    let maximum = safe(&attr(n, "max_gram_length")["i"]);
    let skip = safe(&attr(n, "max_skip_count")["i"]);
    let (counts, counts_reason) = list(n, "ngram_counts");
    let (indexes, indexes_reason) = list(n, "ngram_indexes");
    let sp = attr(n, "pool_strings");
    let ip = attr(n, "pool_int64s");
    let has_sp = !sp.is_null();
    let has_ip = !ip.is_null();
    let pool_kind = if has_sp {
        "string"
    } else if has_ip {
        "int64"
    } else {
        "missing"
    };
    let (pool, pool_reason) = if has_sp {
        (strings(&sp["strings"]), "tfidf_string_pool_invalid")
    } else if has_ip {
        let exact = rows(&ip["intExactDecimals"]);
        if exact.is_empty() || exact.len() != rows(&ip["ints"]).len() {
            (None, "tfidf_integer_pool_exact_values_unavailable")
        } else {
            (
                decimals(&ip["intExactDecimals"]),
                "tfidf_integer_pool_exact_values_invalid",
            )
        }
    } else {
        (None, "pool_attribute_missing")
    };
    let wa = attr(n, "weights");
    let present = !wa.is_null();
    let wf = if wa["floatsText"].is_array() {
        &wa["floatsText"]
    } else {
        &wa["floats"]
    };
    let weights: Option<Vec<f32>> = rows(wf)
        .iter()
        .map(|x| float(x).filter(|f| f.is_finite()).map(|f| f as f32))
        .collect();
    let shape: Option<Vec<u64>> = if t["shapeDeclared"] == true {
        t["shape"].as_array().and_then(|x| {
            x.iter()
                .map(|d| safe(d).filter(|x| *x >= 0).map(|x| x as u64))
                .collect()
        })
    } else {
        None
    };
    let dtype = t["dtype"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or("UNKNOWN");
    let cs = counts.as_deref().unwrap_or(&[]);
    let ix = indexes.as_deref().unwrap_or(&[]);
    let ws = weights.as_deref().unwrap_or(&[]);
    let mut reasons: Vec<String> = vec![];
    macro_rules! fail {
        ($cond:expr,$text:expr) => {
            if $cond {
                reasons.push($text.into());
            }
        };
    }
    fail!(
        safe(&v["imported_opset"]).map_or(true, |x| x < 9),
        "tfidf_opset_9_not_imported"
    );
    fail!(
        shape.as_ref().map_or(true, |s| ![1, 2].contains(&s.len())),
        "tfidf_input_rank_must_be_one_or_two"
    );
    fail!(
        !["STRING", "INT32", "INT64"].contains(&dtype),
        "tfidf_input_dtype_not_supported"
    );
    fail!(
        !["TF", "IDF", "TFIDF"].contains(&mode),
        "tfidf_mode_invalid"
    );
    fail!(
        minimum.map_or(true, |x| x <= 0),
        "tfidf_min_gram_length_not_positive"
    );
    fail!(
        maximum.zip(minimum).map_or(true, |(x, y)| x < y),
        "tfidf_max_gram_length_below_minimum"
    );
    fail!(
        skip.map_or(true, |x| x < 0),
        "tfidf_max_skip_count_negative_or_unsafe"
    );
    fail!(
        counts.is_none() || cs.is_empty(),
        if counts_reason.is_empty() {
            "tfidf_ngram_counts_empty_or_unsafe"
        } else {
            &counts_reason
        }
    );
    fail!(
        indexes.is_none() || ix.is_empty() || ix.iter().any(|x| *x < 0),
        if indexes_reason.is_empty() {
            "tfidf_ngram_indexes_empty_negative_or_unsafe"
        } else {
            &indexes_reason
        }
    );
    fail!(
        has_sp == has_ip,
        if has_sp {
            "tfidf_both_pool_attributes_present"
        } else {
            "tfidf_pool_attribute_missing"
        }
    );
    fail!(
        pool.as_ref().map_or(true, Vec::is_empty),
        if pool.is_none() {
            pool_reason
        } else {
            "tfidf_pool_empty_or_unsafe"
        }
    );
    fail!(
        pool_kind == "string" && dtype != "STRING",
        "tfidf_string_pool_input_dtype_mismatch"
    );
    fail!(
        pool_kind == "int64" && !["INT32", "INT64"].contains(&dtype),
        "tfidf_integer_pool_input_dtype_mismatch"
    );
    fail!(weights.is_none(), "tfidf_weights_non_finite");
    fail!(
        present && weights.is_some() && indexes.is_some() && ws.len() != ix.len(),
        "tfidf_weights_ngram_indexes_length_mismatch"
    );
    fail!(
        shape.as_ref().is_some_and(|s| s.len() == 2 && s[0] == 0),
        "tfidf_pinned_ort_rejects_zero_batch"
    );
    fail!(
        counts.is_some() && minimum.is_some_and(|x| x > cs.len() as i64),
        "tfidf_min_gram_length_outside_ngram_counts"
    );
    fail!(
        counts.is_some() && maximum.is_some_and(|x| x > cs.len() as i64),
        "tfidf_max_gram_length_outside_ngram_counts"
    );
    let width = ix
        .iter()
        .max()
        .map(|x| (*x).max(0))
        .filter(|x| *x < SAFE)
        .map(|x| x as u64 + 1);
    fail!(width.is_none(), "tfidf_output_width_unsafe");
    let defs = match (&pool, &counts, minimum, maximum) {
        (Some(p), Some(c), Some(a), Some(b)) => definitions(p, c, a, b),
        _ => Definitions::default(),
    };
    reasons.extend(defs.failures.clone());
    fail!(
        indexes.is_some() && defs.total != ix.len(),
        "tfidf_definition_index_cardinality_mismatch"
    );
    fail!(
        indexes.is_some()
            && present
            && weights.is_some()
            && ix.iter().any(|x| *x >= ws.len() as i64),
        "tfidf_pinned_ort_weight_coordinate_out_of_bounds"
    );
    reasons = unique(reasons);
    let output_shape = shape.as_ref().zip(width).map(|(s, w)| {
        if s.len() == 1 {
            vec![w]
        } else {
            vec![s.first().copied().unwrap_or(0), w]
        }
    });
    let mut input_status = "not_assessed_input_values_unavailable".to_owned();
    let expected = shape.as_ref().and_then(|s| product(s));
    let mut tokens = None;
    if shape.is_none() {
        input_status = "not_assessed_input_shape_unknown".into();
    } else if dtype == "STRING"
        && t["staticValuesComplete"] == true
        && Some(rows(&t["staticValues"]).len() as u64) == expected
    {
        tokens = strings(&t["staticValues"]);
        if tokens.is_some() {
            input_status = "assessed_exact_string_values".into();
        }
    } else if dtype == "INT64"
        && t["initializerIntegerValuesExactComplete"] == true
        && Some(rows(&t["initializerIntegerValuesExactDecimals"]).len() as u64) == expected
    {
        tokens = decimals(&t["initializerIntegerValuesExactDecimals"]);
        if tokens.is_some() {
            input_status = "assessed_exact_int64_decimals".into();
        }
    } else if ["INT32", "INT64"].contains(&dtype)
        && t["staticValuesComplete"] == true
        && t["staticValues"].is_array()
        && Some(rows(&t["staticValues"]).len() as u64) == expected
    {
        tokens = rows(&t["staticValues"])
            .iter()
            .map(|x| safe(x).map(|x| x.to_string()))
            .collect();
        if tokens.is_some() {
            input_status = format!("assessed_exact_{}_values", dtype.to_lowercase());
        }
    }
    let work = shape.as_ref().and_then(|s| {
        let skip = skip.filter(|x| *x >= 0)? as u64;
        let gram = maximum.filter(|x| *x >= 0)? as u64;
        let batches = if s.len() == 1 { 1 } else { *s.first()? };
        product(&[batches, *s.last()?, skip + 1, gram])
    });
    let mut execution_status = "not_assessed_input_values_unavailable";
    let mut execution = None;
    if reasons.is_empty() && tokens.is_some() && output_shape.is_some() {
        if output_shape
            .as_ref()
            .and_then(|s| product(s))
            .map_or(true, |x| x > OUTPUT_LIMIT)
        {
            execution_status = "not_assessed_output_element_limit";
        } else if work.map_or(true, |x| x > WORK_LIMIT) {
            execution_status = "not_assessed_work_limit";
        } else {
            execution_status = "assessed_exact";
            execution = Some(execute(
                tokens.as_ref().unwrap(),
                shape.as_ref().unwrap(),
                &defs,
                ix,
                ws,
                present,
                mode,
                skip.unwrap() as u64,
                maximum.unwrap() as u64,
                width.unwrap() as usize,
            ));
        }
    }
    let mut coordinate_disagreements = 0;
    let mut value_disagreements = 0;
    if present && mode != "TF" && !ix.is_empty() && !ws.is_empty() {
        for (i, &c) in ix.iter().enumerate() {
            if c != i as i64 {
                coordinate_disagreements += 1;
            }
            if c < ws.len() as i64
                && usize::try_from(c)
                    .ok()
                    .and_then(|index| ws.get(index))
                    .map(|x| x.to_bits())
                    != ws.get(i).map(|x| x.to_bits())
            {
                value_disagreements += 1;
            }
        }
    }
    let duplicates = ix.len() - ix.iter().collect::<BTreeSet<_>>().len();
    let mut risks = reasons.clone();
    if defs.prefix > 0 {
        risks.push("tfidf_ngram_counts_ignore_pool_prefix".into());
    }
    if defs.inactive_dupes > 0 {
        risks.push("tfidf_duplicate_ngram_outside_active_length_range".into());
    }
    if indexes.is_some() && duplicates > 0 {
        risks.push("tfidf_multiple_ngrams_share_output_coordinate".into());
    }
    if coordinate_disagreements > 0 {
        risks.push("tfidf_weight_coordinate_semantics_divergence".into());
    }
    if execution.as_ref().is_some_and(|e| !e.divergent.is_empty()) {
        risks.push("tfidf_ort_repeated_addition_differs_from_onnx_reference".into());
    }
    if [
        "not_assessed_output_element_limit",
        "not_assessed_work_limit",
    ]
    .contains(&execution_status)
    {
        risks.push(format!("tfidf_{execution_status}"));
    }
    let status = if !reasons.is_empty() {
        "fail"
    } else if execution.is_some() {
        "pass"
    } else {
        "partial"
    };
    let reason = reasons.first().cloned().unwrap_or_else(|| {
        if status == "partial" {
            execution_status.into()
        } else {
            String::new()
        }
    });
    let result = if reasons.is_empty() && output_shape.is_some() {
        let mut patch = json!({"dtype":"FLOAT32","shape":output_shape,"shapeDeclared":true});
        if let Some(e) = &execution {
            patch["staticValuesStatus"] = json!("assessed_exact_static_data");
            patch["staticValuesComplete"] = json!(true);
            patch["staticValues"] = json!(e.output);
            patch["staticValuesSource"] = json!("TfIdfVectorizer-9 pinned ORT CPU order");
        }
        json!({"outputs":[[n["outputs"][0],patch]],"reason":if status=="pass"{""}else{execution_status}})
    } else {
        json!({"outputs":[],"reason":reasons.first().map(String::as_str).unwrap_or("tfidf_output_shape_unresolved")})
    };
    let mut row = templates()["TfIdfVectorizer"].clone();
    let fields = json!({"scope":v["scope"],"node_index":v["node_index"],"node_name":v["node_name"].as_str().unwrap_or(""),"imported_opset":v["imported_opset"],"status":status,"reason_codes":reasons,"risk_codes":unique(risks),"input_name":n["inputs"][0].as_str().unwrap_or(""),"input_dtype":dtype,"input_shape":shape.clone().unwrap_or_default(),"output_name":n["outputs"][0].as_str().unwrap_or(""),"exact_output_shape":output_shape.unwrap_or_default(),"exact_output_width":width,"mode":mode,"minimum_gram_length":minimum,"maximum_gram_length":maximum,"maximum_skip_count":skip,"pool_kind":pool_kind,"exact_pool_item_count":pool.as_ref().map(Vec::len),"exact_ngram_level_count":counts.as_ref().map(Vec::len),"exact_ngram_definition_count":defs.total,"exact_active_ngram_definition_count":defs.active.len(),"exact_unused_pool_prefix_item_count":defs.prefix,"exact_duplicate_active_ngram_count":defs.active_dupes,"exact_duplicate_inactive_ngram_count":defs.inactive_dupes,"exact_ngram_index_count":indexes.as_ref().map(Vec::len),"exact_duplicate_output_coordinate_count":indexes.as_ref().map(|_|duplicates),"exact_unaddressed_output_coordinate_count":width.zip(indexes.as_ref()).map(|(w,_)|w as i64-(ix.len()-duplicates) as i64),"weights_present":present,"exact_weight_count":weights.as_ref().map(Vec::len),"exact_weight_coordinate_disagreement_count":coordinate_disagreements,"exact_weight_coordinate_value_disagreement_count":value_disagreements,"static_input_status":input_status,"exact_static_input_value_count":tokens.as_ref().map(Vec::len),"static_execution_status":execution_status,"exact_static_work_units":work});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    if let Some(e) = execution {
        let normalized = |xs: &[f32]| {
            xs.iter()
                .map(|&x| if x == 0. { 0. } else { x })
                .collect::<Vec<_>>()
        };
        let fields = json!({"exact_match_count":e.matches,"exact_nonzero_frequency_count":e.frequency.iter().filter(|&&x|x!=0).count(),"exact_frequency_values":e.frequency,"exact_output_value_count":e.output.len(),"exact_nonzero_output_count":e.output.iter().filter(|&&x|x!=0.).count(),"exact_negative_zero_output_count":e.negative.len(),"exact_output_values":normalized(&e.output),"exact_output_negative_zero_indices":e.negative,"onnx_reference_output_values":normalized(&e.reference),"exact_ort_reference_divergent_output_count":e.divergent.len(),"exact_ort_reference_divergent_output_indices":e.divergent});
        row.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
    }
    Ok(json!({"status":status,"reason":reason,"result":result,"row":row}))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_tokens_do_not_loop_over_huge_skip() {
        let v = json!({"scope":"main_graph","node_index":0,"imported_opset":9,"node":{"inputs":["X"],"outputs":["Y"],"attributes":[["mode",{"s":"TF"}],["min_gram_length",{"i":1}],["max_gram_length",{"i":1}],["max_skip_count",{"i":9007199254740991i64}],["ngram_counts",{"ints":[0]}],["ngram_indexes",{"ints":[0]}],["pool_strings",{"strings":["a"]}]]},"tensors":{"X":{"dtype":"STRING","shapeDeclared":true,"shape":[0],"staticValuesComplete":true,"staticValues":[]}}});
        let r = run(&v).unwrap();
        assert_eq!(r["status"], "pass");
        assert_eq!(r["row"]["exact_output_values"], json!([0.]));
    }
}
