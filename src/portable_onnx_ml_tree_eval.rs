//! Bounded tree scalar references; never runtime-bit-exact observations.
use crate::portable_onnx_ml::{exact_int, float, ft};
use crate::portable_onnx_ml_tree::Input;
use crate::portable_onnx_ml_tree_data::Parsed;
use crate::portable_onnx_ml_tree_graph::{index, Graph};
use crate::portable_onnx_semantics::{input, safe};
use num_traits::ToPrimitive;
use serde_json::{json, Value};
use std::collections::BTreeSet;
const LIMIT: u64 = 1_000_000;
pub(crate) fn unresolved(status: &str) -> Value {
    json!({"status":status,"inputValueCount":null,"rowCount":null,"pathStepCount":null,"rawScoreCount":null,"outputScoreCount":null,"nonFiniteScoreCount":null,"decisionBoundaryCount":null,"unwrittenScoreCount":null,"rawScorePreview":[],"outputScorePreview":[],"labelPreview":[]})
}
fn cast(n: f64, double: bool) -> f64 {
    if double {
        n
    } else {
        n as f32 as f64
    }
}
fn add(a: f64, b: f64, d: bool) -> f64 {
    cast(cast(a, d) + cast(b, d), d)
}
fn logistic(n: f64) -> f64 {
    let n = n as f32 as f64;
    let r = (1. / (1. + (-n.abs()).exp())) as f32 as f64;
    if n < 0. {
        (1. - r) as f32 as f64
    } else {
        r
    }
}
fn probit(n: f64) -> f64 {
    let n = n as f32 as f64;
    let c = (2. * n - 1.) as f32 as f64;
    let sign = if c < 0. { -1. } else { 1. };
    let x = ((1. - c) * (1. + c)) as f32 as f64;
    let log = x.ln() as f32 as f64;
    let v = (2. / (std::f64::consts::PI * 0.147) + 0.5 * log) as f32 as f64;
    let v2 = (log / 0.147) as f32 as f64;
    (std::f64::consts::SQRT_2 * sign * (-v + (v * v - v2).sqrt()).sqrt()) as f32 as f64
}
fn transform(values: &[f64], post: &str, d: bool, regressor: bool) -> Vec<Option<f64>> {
    if regressor && values.len() == 1 {
        return vec![Some(cast(
            if post == "PROBIT" {
                probit(values[0])
            } else {
                values[0]
            },
            d,
        ))];
    }
    if ["SOFTMAX", "SOFTMAX_ZERO"].contains(&post) {
        let mut maximum = -f64::MAX;
        for n in values {
            let n = *n as f32 as f64;
            if n > maximum {
                maximum = n;
            }
        }
        let mut sum = 0.;
        let values = values
            .iter()
            .map(|n| {
                let n = *n as f32 as f64;
                let item = if post == "SOFTMAX_ZERO" && n > -1e-7 && n < 1e-7 {
                    n * (-maximum).exp()
                } else {
                    (n - maximum).exp()
                };
                sum = (sum + item as f32 as f64) as f32 as f64;
                item
            })
            .collect::<Vec<_>>();
        return values
            .iter()
            .map(|n| Some(cast((n / sum) as f32 as f64, d)))
            .collect();
    }
    values
        .iter()
        .map(|n| {
            Some(cast(
                match post {
                    "PROBIT" => probit(*n),
                    "LOGISTIC" => logistic(*n),
                    _ => *n,
                },
                d,
            ))
        })
        .collect()
}
fn branch(v: f64, t: f64, mode: i64, missing: bool, members: Option<&Vec<f64>>) -> bool {
    if v.is_nan() {
        return missing;
    }
    match mode {
        0 => v <= t,
        1 => v < t,
        2 => v >= t,
        3 => v > t,
        4 => v == t,
        5 => v != t,
        6 => members.is_some_and(|a| a.contains(&v)),
        _ => false,
    }
}
fn traverse(row: &[f64], root: usize, p: &Parsed, g: &Graph) -> Result<(usize, u64), &'static str> {
    let mut i = root;
    let mut seen = BTreeSet::new();
    let mut steps = 0;
    if p.generic
        && p.nodes.true_leaves.get(i) == Some(&1.)
        && p.nodes.false_leaves.get(i) == Some(&1.)
        && p.nodes.true_ids.get(i) == p.nodes.false_ids.get(i)
    {
        return Ok((p.nodes.true_ids[i] as usize, 1));
    }
    loop {
        if !seen.insert(i) {
            return Err("not_assessed_runtime_cycle");
        }
        steps += 1;
        if p.generic {
            let n = &p.nodes;
            let feature = index(n.features.get(i).copied(), row.len())
                .and_then(|j| row.get(j))
                .copied()
                .unwrap_or(f64::NAN);
            let yes = branch(
                feature,
                n.values.values[i],
                n.mode_ids.values[i] as i64,
                n.missing.get(i) == Some(&1.),
                p.membership.by_node.get(i).and_then(Option::as_ref),
            );
            let (ids, flags) = if yes {
                (&n.true_ids, &n.true_leaves)
            } else {
                (&n.false_ids, &n.false_leaves)
            };
            let child = ids[i] as usize;
            if flags[i] != 0. {
                return Ok((child, steps));
            }
            i = child;
        } else {
            let n = g.records.get(i).ok_or("not_assessed_missing_node")?;
            if n.mode == "LEAF" {
                return Ok((i, steps));
            }
            let mode = [
                "BRANCH_LEQ",
                "BRANCH_LT",
                "BRANCH_GTE",
                "BRANCH_GT",
                "BRANCH_EQ",
                "BRANCH_NEQ",
            ]
            .iter()
            .position(|s| *s == n.mode)
            .unwrap_or(99) as i64;
            let feature = index(n.feature, row.len())
                .and_then(|j| row.get(j))
                .copied()
                .unwrap_or(f64::NAN);
            let child = if branch(feature, n.value, mode, n.missing, None) {
                n.yes
            } else {
                n.no
            };
            i = *g
                .keys
                .get(&format!(
                    "{}:{}",
                    n.tree,
                    child.map(ft).unwrap_or_else(|| "undefined".into())
                ))
                .ok_or("not_assessed_missing_child")?;
        }
    }
}
fn regressor(leaves: &[usize], p: &Parsed, g: &Graph, d: bool) -> Vec<f64> {
    let mut s = vec![0.; p.targets as usize];
    let mut has = vec![false; s.len()];
    let mut accumulate = |id: usize, value: f64| {
        if id >= s.len() {
            return;
        }
        s[id] = match p.aggregate.as_str() {
            "SUM" | "AVERAGE" => add(s[id], value, d),
            "MIN" => {
                if !has[id] || value < s[id] {
                    value
                } else {
                    s[id]
                }
            }
            "MAX" => {
                if !has[id] || value > s[id] {
                    value
                } else {
                    s[id]
                }
            }
            _ => s[id],
        };
        has[id] = true;
    };
    for leaf in leaves {
        if p.generic {
            accumulate(p.leaf_targets[*leaf] as usize, p.leaf_weights.values[*leaf]);
        } else {
            let entries = &g.weights[*leaf];
            for e in entries
                .iter()
                .take(if p.targets == 1 { 1 } else { entries.len() })
            {
                accumulate(e.id, e.value);
            }
        }
    }
    for i in 0..s.len() {
        if p.aggregate == "AVERAGE" {
            s[i] = cast(cast(s[i], d) / leaves.len() as f64, d);
        }
        if !p.generic && p.base.values.len() == s.len() {
            s[i] = add(s[i], p.base.values[i], d);
        } else if !has[i] && ["MIN", "MAX"].contains(&p.aggregate.as_str()) {
            s[i] = 0.;
        }
    }
    s
}
struct Scores {
    raw: Vec<f64>,
    output: Vec<Option<f64>>,
    label: String,
    boundary: bool,
    unwritten: usize,
}
fn classifier(leaves: &[usize], p: &Parsed, g: &Graph, d: bool) -> Result<Scores, &'static str> {
    let count = p.labels.len();
    if count < 2 {
        return Err("not_assessed_classifier_less_than_two_classes");
    }
    let mut s = vec![0.; count];
    let mut has = vec![false; count];
    for leaf in leaves {
        for e in &g.weights[*leaf] {
            s[e.id] = add(s[e.id], e.value, d);
            has[e.id] = true;
        }
    }
    let b = &p.base.values;
    if count > 2 {
        for (i, b) in b.iter().enumerate() {
            s[i] = if has[i] { add(s[i], *b, d) } else { *b };
            has[i] = true;
        }
        let mut max = None;
        for i in 0..count {
            if has[i] && max.is_none_or(|j| s[i] > s[j]) {
                max = Some(i);
            }
        }
        let max = max.ok_or("not_assessed_classifier_has_no_score")?;
        let boundary = s
            .iter()
            .enumerate()
            .filter(|(i, n)| has[*i] && **n == s[max])
            .count()
            > 1;
        return Ok(Scores {
            output: transform(&s, &p.post, d, false),
            raw: s,
            label: p.labels[max].clone(),
            boundary,
            unwritten: 0,
        });
    }
    let mut expansion = -1;
    if b.len() == 2 {
        if has[1] {
            s[1] = add(b[1], s[0], d);
            s[0] = cast(-s[1], d);
            has[1] = true;
        } else {
            s[1] = add(s[1], b[1], d);
            s[0] = add(s[0], b[0], d);
        }
    } else if b.len() == 1 {
        s[0] = add(s[0], b[0], d);
        if !has[1] {
            s.truncate(1);
        }
    } else if b.is_empty() {
        expansion = 3;
        if !has[1] {
            s.truncate(1);
        }
    }
    let binary = g
        .weights
        .iter()
        .flatten()
        .map(|e| e.id)
        .collect::<BTreeSet<_>>()
        .len()
        == 1;
    let positive = if s.len() == 2 && has[1] {
        s[1]
    } else if has[0] {
        s[0]
    } else {
        0.
    };
    let label = if positive > 0. { 1 } else { 0 };
    if binary {
        expansion = if positive > 0. { 2 } else { 3 };
    }
    let output = if s.len() >= 2 {
        transform(&s, &p.post, d, false)
    } else {
        let v = s[0];
        if p.post == "PROBIT" {
            vec![Some(cast(probit(v), d)), None]
        } else if expansion == 0 || expansion == 1 {
            vec![Some(cast(1. - v, d)), Some(cast(v, d))]
        } else if expansion == 2 || expansion == 3 {
            if p.post == "LOGISTIC" {
                vec![Some(cast(logistic(-v), d)), Some(cast(logistic(v), d))]
            } else {
                vec![Some(cast(-v, d)), Some(cast(v, d))]
            }
        } else {
            vec![Some(cast(v, d)), None]
        }
    };
    let unwritten = output.iter().filter(|n| n.is_none()).count();
    Ok(Scores {
        raw: s,
        output,
        label: if !binary && p.label_kind == "int64" {
            label.to_string()
        } else {
            p.labels[label].clone()
        },
        boundary: positive == 0.,
        unwritten,
    })
}
pub(crate) fn run(v: &Value, i: &Input, p: &Parsed, g: &Graph) -> Value {
    let t = input(v, 0);
    let d = i.dtype == "FLOAT64";
    let values: Option<Vec<f64>> =
        if i.dtype == "INT64" && t["initializerIntegerValuesExactComplete"] == true {
            t["initializerIntegerValuesExactDecimals"]
                .as_array()
                .and_then(|a| {
                    a.iter()
                        .map(|v| {
                            exact_int(v).and_then(|v| {
                                if d {
                                    v.to_f64()
                                } else {
                                    v.to_f32().map(f64::from)
                                }
                            })
                        })
                        .collect()
                })
        } else if ["FLOAT16", "FLOAT32", "FLOAT64"].contains(&i.dtype.as_str())
            && t["staticValuesCanonicalTextComplete"] == true
        {
            t["staticValuesCanonicalTexts"]
                .as_array()
                .and_then(|a| a.iter().map(float).collect())
        } else if ["FLOAT16", "FLOAT32", "FLOAT64", "INT32"].contains(&i.dtype.as_str())
            && t["staticValuesComplete"] == true
        {
            let x = if t["staticValuesText"].is_array() {
                &t["staticValuesText"]
            } else {
                &t["staticValues"]
            };
            x.as_array().and_then(|a| a.iter().map(float).collect())
        } else {
            None
        };
    let Some(values) = values else {
        return unresolved("not_assessed_dynamic_input");
    };
    let Some((batch, features)) = safe(&i.batch)
        .filter(|n| *n >= 0)
        .zip(safe(&i.features).filter(|n| *n >= 0))
        .filter(|(b, f)| b.checked_mul(*f) == Some(values.len() as i64))
    else {
        return unresolved("not_assessed_input_shape_or_payload_mismatch");
    };
    let work = (batch as u64)
        .checked_mul(g.roots.len().max(1) as u64)
        .and_then(|n| n.checked_mul(g.stats["maxDepth"].as_u64().unwrap_or(0).max(1)));
    let outputs = (batch as u64).checked_mul(p.targets.max(1) as u64);
    if work.is_none_or(|n| n > LIMIT) || outputs.is_none_or(|n| n > LIMIT) {
        return unresolved("not_assessed_work_limit");
    }
    let (mut steps, mut boundary, mut unwritten) = (0u64, 0usize, 0usize);
    let (mut raw, mut out, mut labels) = (vec![], vec![], vec![]);
    for batch in 0..batch as usize {
        let row = values[batch * features as usize..(batch + 1) * features as usize]
            .iter()
            .map(|n| cast(*n, d))
            .collect::<Vec<_>>();
        let mut leaves = vec![];
        for root in &g.roots {
            match traverse(&row, *root, p, g) {
                Ok((leaf, n)) => {
                    leaves.push(leaf);
                    steps += n;
                    if steps > LIMIT {
                        return unresolved("not_assessed_work_limit");
                    }
                }
                Err(s) => return unresolved(s),
            }
        }
        if p.classifier {
            let s = match classifier(&leaves, p, g, d) {
                Ok(s) => s,
                Err(s) => return unresolved(s),
            };
            raw.extend(s.raw);
            out.extend(s.output);
            labels.push(s.label);
            boundary += usize::from(s.boundary);
            unwritten += s.unwritten;
        } else {
            let r = regressor(&leaves, p, g, d);
            boundary += r.iter().filter(|n| **n == 0.).count();
            out.extend(transform(&r, &p.post, d, true));
            raw.extend(r);
        }
    }
    json!({"status":"assessed_scalar_reference","inputValueCount":values.len(),"rowCount":batch,"pathStepCount":steps,"rawScoreCount":raw.len(),"outputScoreCount":out.len(),"nonFiniteScoreCount":out.iter().filter(|n|n.is_some_and(|n|!n.is_finite())).count(),"decisionBoundaryCount":boundary,"unwrittenScoreCount":unwritten,"rawScorePreview":raw.iter().take(64).map(|n|ft(*n)).collect::<Vec<_>>(),"outputScorePreview":out.iter().take(64).map(|n|n.map(ft).unwrap_or_else(||"UNWRITTEN".into())).collect::<Vec<_>>(),"labelPreview":labels.into_iter().take(64).collect::<Vec<_>>()})
}
