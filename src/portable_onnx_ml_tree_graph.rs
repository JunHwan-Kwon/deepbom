//! Tree topology validation. Iterative traversal avoids host/WASM stack-depth divergence.
use crate::portable_onnx_ml::ft;
use crate::portable_onnx_ml_tree_data::{valid, Parsed};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone)]
pub(crate) struct Record {
    pub tree: String,
    pub key: String,
    pub feature: Option<f64>,
    pub mode: String,
    pub value: f64,
    pub yes: Option<f64>,
    pub no: Option<f64>,
    pub missing: bool,
}
#[derive(Clone)]
pub(crate) struct Entry {
    pub id: usize,
    pub value: f64,
}
pub(crate) struct Graph {
    pub stats: Value,
    pub records: Vec<Record>,
    pub roots: Vec<usize>,
    pub keys: BTreeMap<String, usize>,
    pub weights: Vec<Vec<Entry>>,
}
pub(crate) fn component(n: Option<f64>) -> String {
    n.map(ft).unwrap_or("undefined".into())
}
fn key(a: Option<f64>, b: Option<f64>) -> String {
    format!("{}:{}", component(a), component(b))
}
pub(crate) fn index(n: Option<f64>, limit: usize) -> Option<usize> {
    n.and_then(valid)
        .filter(|n| *n >= 0 && (*n as u64) < limit as u64)
        .map(|n| n as usize)
}
fn stats() -> Value {
    let mut s = json!({});
    for name in [
        "treeCount",
        "nodeCount",
        "branchNodeCount",
        "leafCount",
        "reachableNodeCount",
        "reachableLeafCount",
        "orphanNodeCount",
        "maxDepth",
        "cycleCount",
        "duplicateNodeCount",
        "invalidChildReferenceCount",
        "invalidFeatureCount",
        "rootMismatchCount",
        "multipleParentNodeCount",
        "invalidWeightReferenceCount",
        "invalidWeightIdCount",
        "nonLeafWeightCount",
        "singleTargetIgnoredWeightCount",
        "usedWeightCount",
        "unusedWeightCount",
        "membershipNodeCount",
        "membershipSetCount",
        "membershipValueCount",
        "membershipDuplicateValueCount",
        "membershipSeparatorCount",
    ] {
        s[name] = json!(0);
    }
    s
}
fn graph() -> Graph {
    Graph {
        stats: stats(),
        records: vec![],
        roots: vec![],
        keys: BTreeMap::new(),
        weights: vec![],
    }
}
pub(crate) fn run(
    p: &Parsed,
    features: Option<i64>,
    failures: &mut Vec<String>,
    reasons: &mut Vec<String>,
) -> Graph {
    if p.generic {
        generic(p, features, failures, reasons)
    } else {
        legacy(p, features, failures, reasons)
    }
}
fn legacy(
    p: &Parsed,
    features: Option<i64>,
    failures: &mut Vec<String>,
    reasons: &mut Vec<String>,
) -> Graph {
    let (n, w) = (&p.nodes, &p.weights);
    let count = n.trees.len();
    let wc = w.ids.len();
    if count == 0
        || [
            n.ids.len(),
            n.features.len(),
            n.values.values.len(),
            n.modes.len(),
            n.true_ids.len(),
            n.false_ids.len(),
        ]
        .iter()
        .any(|n| *n != count)
    {
        failures.push("tree_ensemble_node_tuple_cardinality_mismatch".into());
    }
    if !n.missing.is_empty() && n.missing.len() != count {
        failures.push("tree_ensemble_missing_track_cardinality_mismatch".into());
    }
    if n.modes.iter().any(|n| {
        ![
            "BRANCH_LEQ",
            "BRANCH_LT",
            "BRANCH_GTE",
            "BRANCH_GT",
            "BRANCH_EQ",
            "BRANCH_NEQ",
            "LEAF",
        ]
        .contains(&n.as_str())
    }) {
        failures.push("tree_ensemble_invalid_node_mode".into());
    }
    if n.missing.iter().any(|n| *n != 0. && *n != 1.) {
        failures.push("tree_ensemble_missing_track_not_boolean".into());
    }
    if wc == 0
        || [w.trees.len(), w.nodes.len(), w.values.values.len()]
            .iter()
            .any(|n| *n != wc)
    {
        failures.push("tree_ensemble_weight_tuple_cardinality_mismatch".into());
    }
    let mut g = graph();
    let (mut duplicate, mut invalid_feature) = (0, 0);
    for i in 0..count {
        let tree = component(n.trees.get(i).copied());
        let key = key(n.trees.get(i).copied(), n.ids.get(i).copied());
        if g.keys.contains_key(&key) {
            duplicate += 1;
        } else {
            g.keys.insert(key.clone(), i);
        }
        let mode = n.modes.get(i).cloned().unwrap_or_default();
        let feature = n.features.get(i).copied();
        if mode != "LEAF"
            && feature
                .and_then(valid)
                .is_none_or(|n| n < 0 || features.is_some_and(|f| n >= f))
        {
            invalid_feature += 1;
        }
        g.records.push(Record {
            tree,
            key,
            mode,
            feature,
            value: n.values.values.get(i).copied().unwrap_or(f64::NAN),
            yes: n.true_ids.get(i).copied(),
            no: n.false_ids.get(i).copied(),
            missing: n.missing.get(i) == Some(&1.),
        });
    }
    if duplicate > 0 {
        failures.push("tree_ensemble_duplicate_tree_node_identity".into());
    }
    if invalid_feature > 0 {
        failures.push("tree_ensemble_feature_id_out_of_bounds".into());
    }
    let mut tree_order = vec![];
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    let mut previous = None::<String>;
    let mut closed = BTreeSet::new();
    let mut noncontiguous = 0;
    for (i, r) in g.records.iter().enumerate() {
        if previous.as_ref() != Some(&r.tree) {
            if let Some(old) = previous.as_ref() {
                closed.insert(old.clone());
            }
            if closed.contains(&r.tree) {
                noncontiguous += 1;
            }
            if !groups.contains_key(&r.tree) {
                tree_order.push(r.tree.clone());
            }
            previous = Some(r.tree.clone());
        }
        groups.entry(r.tree.clone()).or_default().push(i);
    }
    if noncontiguous > 0 {
        failures.push("tree_ensemble_noncontiguous_tree_serialization".into());
    }
    let mut incoming = g
        .records
        .iter()
        .map(|r| (r.key.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let (mut invalid_child, mut self_refs) = (0, 0);
    for r in &g.records {
        if r.mode == "LEAF" {
            continue;
        }
        for child in [r.yes, r.no] {
            let key = format!("{}:{}", r.tree, component(child));
            if index(child, count).is_none() || !g.keys.contains_key(&key) {
                invalid_child += 1;
            } else {
                *incoming.entry(key.clone()).or_default() += 1;
                if key == r.key {
                    self_refs += 1;
                }
            }
        }
    }
    if invalid_child > 0 {
        failures.push("tree_ensemble_invalid_child_reference".into());
    }
    if self_refs > 0 {
        failures.push("tree_ensemble_self_reference".into());
    }
    let mut root_mismatch = 0;
    for tree in &tree_order {
        let group = &groups[tree];
        let structural = group
            .iter()
            .filter(|i| incoming[&g.records[**i].key] == 0)
            .copied()
            .collect::<Vec<_>>();
        let root = group[0];
        g.roots.push(root);
        if structural.len() != 1 || structural[0] != root {
            root_mismatch += 1;
        }
        if !p.classifier {
            let mut ids = group
                .iter()
                .map(|i| n.ids.get(*i).copied().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            ids.sort_by(f64::total_cmp);
            if ids.iter().enumerate().any(|(i, n)| *n != i as f64) {
                failures.push("tree_regressor_node_ids_not_zero_based_sequential".into());
            }
        }
    }
    if root_mismatch > 0 {
        failures.push("tree_ensemble_serialized_root_mismatch".into());
    }
    let (mut reachable, mut leaves, mut active) =
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let (mut cycles, mut depth) = (0, 0);
    if invalid_child == 0 {
        for root in &g.roots {
            let mut stack = vec![(*root, 1usize, false)];
            while let Some((i, d, exit)) = stack.pop() {
                if exit {
                    active.remove(&i);
                    continue;
                }
                if i >= g.records.len() {
                    continue;
                }
                depth = depth.max(d);
                if active.contains(&i) {
                    cycles += 1;
                    continue;
                }
                if !reachable.insert(i) {
                    continue;
                }
                active.insert(i);
                stack.push((i, d, true));
                let r = &g.records[i];
                if r.mode == "LEAF" {
                    leaves.insert(i);
                } else {
                    for child in [r.yes, r.no] {
                        if let Some(child) = g.keys.get(&format!("{}:{}", r.tree, component(child)))
                        {
                            stack.push((*child, d + 1, false));
                        }
                    }
                }
            }
        }
    }
    if cycles > 0 {
        failures.push("tree_ensemble_cycle_detected".into());
    }
    let orphans = count - reachable.len();
    if orphans > 0 {
        failures.push("tree_ensemble_orphan_node".into());
    }
    let multiple = incoming.values().filter(|n| **n > 1).count();
    if multiple > 0 {
        reasons.push("tree_ensemble_shared_subtree_not_strict_tree".into());
    }
    g.weights = vec![vec![]; count];
    let (mut invalid_weight, mut nonleaf, mut invalid_id, mut unused) = (0, 0, 0, 0);
    for i in 0..wc {
        let id = w
            .ids
            .get(i)
            .copied()
            .and_then(valid)
            .filter(|n| *n >= 0 && *n < p.targets);
        if id.is_none() {
            invalid_id += 1;
        }
        let key = key(w.trees.get(i).copied(), w.nodes.get(i).copied());
        let Some(node) = g.keys.get(&key).copied() else {
            invalid_weight += 1;
            unused += 1;
            continue;
        };
        if g.records[node].mode != "LEAF" {
            nonleaf += 1;
            unused += 1;
            continue;
        }
        if !reachable.contains(&node) {
            unused += 1;
            continue;
        }
        g.weights[node].push(Entry {
            id: id.unwrap_or(0) as usize,
            value: w.values.values.get(i).copied().unwrap_or(f64::NAN),
        });
    }
    if invalid_weight > 0 {
        failures.push("tree_ensemble_weight_references_missing_node".into());
    }
    if invalid_id > 0 {
        failures.push("tree_ensemble_weight_target_or_class_id_out_of_bounds".into());
    }
    if nonleaf > 0 {
        reasons.push("tree_ensemble_nonleaf_weights_ignored_by_pinned_ort".into());
    }
    let ignored = if p.targets == 1 {
        g.weights
            .iter()
            .map(|v| v.len().saturating_sub(1))
            .sum::<usize>()
    } else {
        0
    };
    if ignored > 0 {
        reasons.push(
            "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort".into(),
        );
    }
    let used = g.weights.iter().map(Vec::len).sum::<usize>() - ignored;
    unused += ignored;
    let s = json!({"treeCount":g.roots.len(),"nodeCount":count,"branchNodeCount":g.records.iter().filter(|r|r.mode!="LEAF").count(),"leafCount":g.records.iter().filter(|r|r.mode=="LEAF").count(),"reachableNodeCount":reachable.len(),"reachableLeafCount":leaves.len(),"orphanNodeCount":orphans,"maxDepth":depth,"cycleCount":cycles,"duplicateNodeCount":duplicate,"invalidChildReferenceCount":invalid_child,"invalidFeatureCount":invalid_feature,"rootMismatchCount":root_mismatch,"multipleParentNodeCount":multiple,"invalidWeightReferenceCount":invalid_weight,"invalidWeightIdCount":invalid_id,"nonLeafWeightCount":nonleaf,"singleTargetIgnoredWeightCount":ignored,"usedWeightCount":used,"unusedWeightCount":unused});
    g.stats
        .as_object_mut()
        .unwrap()
        .extend(s.as_object().unwrap().clone());
    g
}
fn generic(
    p: &Parsed,
    features: Option<i64>,
    failures: &mut Vec<String>,
    reasons: &mut Vec<String>,
) -> Graph {
    let n = &p.nodes;
    let count = n.values.values.len();
    let leaves = p.leaf_weights.values.len();
    if count == 0
        || [
            n.features.len(),
            n.mode_ids.values.len(),
            n.true_ids.len(),
            n.false_ids.len(),
            n.true_leaves.len(),
            n.false_leaves.len(),
        ]
        .iter()
        .any(|n| *n != count)
    {
        failures.push("tree_ensemble_v5_node_tuple_cardinality_mismatch".into());
    }
    if !n.missing.is_empty() && n.missing.len() != count {
        failures.push("tree_ensemble_v5_missing_track_cardinality_mismatch".into());
    }
    if n.hitrates.present && n.hitrates.values.len() != count {
        failures.push("tree_ensemble_v5_hitrate_cardinality_mismatch".into());
    }
    if n.mode_ids
        .values
        .iter()
        .any(|n| valid(*n).is_none_or(|n| !(0..=6).contains(&n)))
    {
        failures.push("tree_ensemble_v5_invalid_node_mode".into());
    }
    if n.true_leaves
        .iter()
        .chain(&n.false_leaves)
        .chain(&n.missing)
        .any(|n| *n != 0. && *n != 1.)
    {
        failures.push("tree_ensemble_v5_branch_flag_not_boolean".into());
    }
    if p.leaf_targets.len() != leaves {
        failures.push("tree_ensemble_v5_leaf_tuple_cardinality_mismatch".into());
    }
    if p.leaf_targets
        .iter()
        .any(|n| valid(*n).is_none_or(|n| n < 0 || n >= p.targets))
    {
        failures.push("tree_ensemble_v5_leaf_target_id_out_of_bounds".into());
    }
    let invalid_feature = n
        .features
        .iter()
        .filter(|n| valid(**n).is_none_or(|n| n < 0 || features.is_some_and(|f| n >= f)))
        .count();
    if invalid_feature > 0 {
        failures.push("tree_ensemble_v5_feature_id_out_of_bounds".into());
    }
    let (mut invalid_child, mut invalid_root) = (0, 0);
    for i in 0..count {
        for (ids, flags) in [
            (&n.true_ids, &n.true_leaves),
            (&n.false_ids, &n.false_leaves),
        ] {
            let is_leaf = flags.get(i).is_some_and(|n| *n != 0. && !n.is_nan());
            if index(ids.get(i).copied(), if is_leaf { leaves } else { count }).is_none() {
                invalid_child += 1;
            }
        }
    }
    for root in &p.roots {
        if index(Some(*root), count).is_none() {
            invalid_root += 1;
        }
    }
    if p.roots.is_empty() || invalid_root > 0 {
        failures.push("tree_ensemble_v5_invalid_tree_root".into());
    }
    if invalid_child > 0 {
        failures.push("tree_ensemble_v5_invalid_child_reference".into());
    }
    let (mut reachable, mut reached_leaves, mut active) =
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let mut incoming = vec![0usize; count];
    let (mut cycles, mut depth) = (0, 0);
    if invalid_root == 0 && invalid_child == 0 {
        for root in &p.roots {
            let mut stack = vec![(*root as usize, 1usize, false, false)];
            while let Some((i, d, exit, edge)) = stack.pop() {
                if exit {
                    active.remove(&i);
                    continue;
                }
                if edge {
                    incoming[i] += 1;
                }
                depth = depth.max(d);
                if active.contains(&i) {
                    cycles += 1;
                    continue;
                }
                if !reachable.insert(i) {
                    continue;
                }
                active.insert(i);
                stack.push((i, d, true, false));
                let degenerate = n.true_leaves.get(i) == Some(&1.)
                    && n.false_leaves.get(i) == Some(&1.)
                    && n.true_ids.get(i) == n.false_ids.get(i);
                if degenerate {
                    reached_leaves.insert(n.true_ids[i] as usize);
                } else {
                    for (ids, flags) in [
                        (&n.true_ids, &n.true_leaves),
                        (&n.false_ids, &n.false_leaves),
                    ] {
                        let child = ids[i] as usize;
                        if flags.get(i).copied().unwrap_or(0.) != 0. {
                            reached_leaves.insert(child);
                        } else {
                            stack.push((child, d + 1, false, true));
                        }
                    }
                }
            }
        }
    }
    if cycles > 0 {
        failures.push("tree_ensemble_v5_cycle_detected".into());
    }
    let orphans = count - reachable.len() + leaves - reached_leaves.len();
    if orphans > 0 {
        reasons.push("tree_ensemble_v5_unreachable_serialized_nodes_or_leaves".into());
    }
    let multiple = incoming.iter().filter(|n| **n > 1).count();
    if multiple > 0 {
        reasons.push("tree_ensemble_v5_shared_subtree_not_strict_tree".into());
    }
    if p.roots.len()
        != p.roots
            .iter()
            .map(|n| ft(*n))
            .collect::<BTreeSet<_>>()
            .len()
    {
        reasons.push("tree_ensemble_v5_duplicate_root_repeats_tree_contribution".into());
    }
    let mut g = graph();
    g.roots = p
        .roots
        .iter()
        .filter_map(|r| index(Some(*r), count))
        .collect();
    let s = json!({"treeCount":p.roots.len(),"nodeCount":count,"branchNodeCount":count,"leafCount":leaves,"reachableNodeCount":reachable.len(),"reachableLeafCount":reached_leaves.len(),"orphanNodeCount":orphans,"maxDepth":depth,"cycleCount":cycles,"invalidChildReferenceCount":invalid_child,"invalidFeatureCount":invalid_feature,"rootMismatchCount":invalid_root,"multipleParentNodeCount":multiple,"usedWeightCount":reached_leaves.len(),"unusedWeightCount":leaves-reached_leaves.len(),"membershipNodeCount":p.membership.nodes,"membershipSetCount":p.membership.sets.len(),"membershipValueCount":p.membership.count,"membershipDuplicateValueCount":p.membership.duplicates,"membershipSeparatorCount":p.membership.separators});
    g.stats
        .as_object_mut()
        .unwrap()
        .extend(s.as_object().unwrap().clone());
    g
}
