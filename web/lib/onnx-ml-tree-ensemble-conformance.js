const TREE_OPS = new Set(["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"]);
const LEGACY_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const GENERIC_DTYPES = new Set(["FLOAT16", "FLOAT32", "FLOAT64"]);
const LEGACY_MODES = new Set(["BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "BRANCH_GT", "BRANCH_EQ", "BRANCH_NEQ", "LEAF"]);

export function validateTreeEnsembleRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || !TREE_OPS.has(row.op_name)) return false;
  const op = ops.find((candidate) => candidate.index === row.node_index && candidate.name === row.op_name
    && candidate.domain === "ai.onnx.ml");
  if (!op) return false;
  const attrs = new Map((op.onnx_attributes || []).map((attribute) => [attribute.name, attribute]));
  const input = tensors.find((tensor) => tensor.name === op.input_names?.[0]);
  const inputFacts = inputContract(input);
  const facts = row.op_name === "TreeEnsemble"
    ? genericFacts(attrs, inputFacts) : legacyFacts(attrs, inputFacts, row.op_name === "TreeEnsembleClassifier", row.resolved_schema_version);
  if (!facts.valid) return row.status === "fail" && Array.isArray(row.reason_codes) && row.reason_codes.length > 0;
  const expectedStatus = facts.fail ? "fail" : facts.partial ? "partial" : "pass";
  const outputShape = facts.outputShape;
  const outputDeclared = outputShape.length > 0;
  const exactElements = outputDeclared && outputShape.every(knownDimension) ? product(outputShape) : null;
  const referenceAssessed = String(row.tree_reference_assessment_status || "").startsWith("assessed_");
  const expectedInputValues = referenceAssessed ? staticValueCount(input) : null;
  const expectedRows = referenceAssessed ? inputFacts.batch : null;
  const expectedRisks = structuralRisks(row, facts);
  return row.status === expectedStatus
    && row.input_name === input?.name && row.input_dtype === inputFacts.dtype
    && row.input_rank === inputFacts.rank && JSON.stringify(row.input_shape) === JSON.stringify(inputFacts.shape)
    && row.exact_batch_count === inputFacts.batch && row.exact_feature_count === inputFacts.features
    && row.resolved_schema_version === facts.schemaVersion
    && row.tree_encoding === facts.encoding
    && row.tree_deprecated_operator === facts.deprecated
    && row.tree_onnx_contract_status === (facts.fail ? "fail" : "pass")
    && row.tree_pinned_ort_contract_status === (facts.fail || facts.cpuGap ? "fail" : "pass")
    && row.tree_aggregate_function === facts.aggregate
    && row.tree_post_transform === facts.postTransform
    && row.tree_base_value_count === facts.baseCount
    && row.tree_class_or_target_count === facts.targetCount
    && row.tree_class_label_count === facts.classCount
    && row.tree_duplicate_class_label_count === facts.duplicateLabels
    && row.tree_exact_tree_count === facts.treeCount
    && row.tree_exact_root_count === facts.treeCount
    && row.tree_exact_node_count === facts.nodeCount
    && row.tree_exact_branch_node_count === facts.branchCount
    && row.tree_exact_leaf_count === facts.leafCount
    && row.tree_reachable_node_count === facts.reachableNodeCount
    && row.tree_reachable_leaf_count === facts.reachableLeafCount
    && row.tree_orphan_node_or_leaf_count === facts.orphanCount
    && row.tree_max_depth === facts.maxDepth
    && row.tree_cycle_count === facts.cycleCount
    && row.tree_duplicate_node_identity_count === facts.duplicateNodeCount
    && row.tree_invalid_child_reference_count === facts.invalidChildCount
    && row.tree_invalid_feature_id_count === facts.invalidFeatureCount
    && row.tree_root_mismatch_count === facts.rootMismatchCount
    && row.tree_multiple_parent_node_count === facts.multipleParentCount
    && row.tree_invalid_weight_reference_count === facts.invalidWeightReferenceCount
    && row.tree_invalid_weight_id_count === facts.invalidWeightIdCount
    && row.tree_single_target_ignored_weight_count === facts.singleTargetIgnoredWeightCount
    && row.tree_weight_tuple_count === facts.weightCount
    && row.tree_used_weight_count === facts.usedWeightCount
    && row.tree_unused_weight_count === facts.unusedWeightCount
    && row.tree_unresolved_weight_count === facts.weightCount - facts.usedWeightCount - facts.unusedWeightCount
    && row.tree_membership_node_count === facts.memberNodeCount
    && row.tree_membership_set_count === facts.memberSetCount
    && row.tree_membership_value_count === facts.memberValueCount
    && row.tree_membership_duplicate_value_count === facts.memberDuplicateCount
    && row.tree_membership_separator_count === facts.memberSeparatorCount
    && row.tree_non_finite_parameter_count === facts.nonFiniteCount
    && row.tree_pinned_cpu_dtype_gap === facts.cpuGap
    && row.exact_output_rank === (outputDeclared ? outputShape.length : null)
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputShape)
    && row.exact_dense_output_element_count === exactElements
    && (!referenceAssessed || row.tree_reference_input_value_count === expectedInputValues
      && row.tree_reference_row_count === expectedRows
      && Number.isSafeInteger(row.tree_reference_path_step_count)
      && row.tree_reference_path_step_count >= expectedRows * facts.treeCount)
    && Number.isSafeInteger(row.tree_reference_non_finite_score_count ?? 0)
    && Number.isSafeInteger(row.tree_reference_decision_boundary_count ?? 0)
    && Number.isSafeInteger(row.tree_reference_unwritten_score_count ?? 0)
    && Array.isArray(row.tree_reference_raw_score_preview)
    && Array.isArray(row.tree_reference_output_score_preview)
    && Array.isArray(row.tree_reference_label_preview)
    && expectedRisks.every((risk) => row.risk_codes.includes(risk))
    && row.risk_codes.every((risk) => expectedRisks.includes(risk));
}

function legacyFacts(attrs, input, classifier, schemaVersion) {
  const prefix = classifier ? "class" : "target";
  const treeIds = ints(attrs.get("nodes_treeids"));
  const nodeIds = ints(attrs.get("nodes_nodeids"));
  const featureIds = ints(attrs.get("nodes_featureids"));
  const modes = strings(attrs.get("nodes_modes"));
  const trueIds = ints(attrs.get("nodes_truenodeids"));
  const falseIds = ints(attrs.get("nodes_falsenodeids"));
  const missing = ints(attrs.get("nodes_missing_value_tracks_true"));
  const nodeValues = numericChoice(attrs, "nodes_values", "nodes_values_as_tensor");
  const weightTreeIds = ints(attrs.get(`${prefix}_treeids`));
  const weightNodeIds = ints(attrs.get(`${prefix}_nodeids`));
  const weightIds = ints(attrs.get(`${prefix}_ids`));
  const weights = numericChoice(attrs, `${prefix}_weights`, `${prefix}_weights_as_tensor`);
  const base = numericChoice(attrs, "base_values", "base_values_as_tensor");
  const intLabels = exactInts(attrs.get("classlabels_int64s"));
  const stringLabels = strings(attrs.get("classlabels_strings"));
  const labelSources = Number(attrs.has("classlabels_int64s")) + Number(attrs.has("classlabels_strings"));
  const labels = classifier ? attrs.has("classlabels_strings") ? stringLabels : intLabels : [];
  const nTargets = classifier ? labels?.length : safeInteger(integer(attrs.get("n_targets"), 0n));
  const postTransform = string(attrs.get("post_transform"), "NONE");
  const aggregate = classifier ? "SUM" : string(attrs.get("aggregate_function"), "SUM");
  const arrays = [treeIds, nodeIds, featureIds, modes, trueIds, falseIds, missing, nodeValues,
    weightTreeIds, weightNodeIds, weightIds, weights, base, intLabels, stringLabels];
  if (arrays.some((value) => value == null) || nTargets == null || postTransform == null || aggregate == null) return invalid();
  const nodeCount = treeIds.length;
  const nodeCardinalityFail = !nodeCount || [nodeIds, featureIds, modes, trueIds, falseIds, nodeValues]
    .some((values) => values.length !== nodeCount) || missing.length > 0 && missing.length !== nodeCount;
  const weightCount = weightIds.length;
  const weightCardinalityFail = !weightCount || [weightTreeIds, weightNodeIds, weights].some((values) => values.length !== weightCount);
  const records = [];
  const keyMap = new Map();
  let duplicateNodeCount = 0;
  let invalidFeatureCount = 0;
  for (let index = 0; index < nodeCount; index += 1) {
    const key = `${treeIds[index]}:${nodeIds[index]}`;
    if (keyMap.has(key)) duplicateNodeCount += 1; else keyMap.set(key, index);
    if (modes[index] !== "LEAF" && (!safeNonnegative(featureIds[index]) || input.features != null && featureIds[index] >= input.features)) invalidFeatureCount += 1;
    records.push({ treeId: treeIds[index], nodeId: nodeIds[index], mode: modes[index], trueId: trueIds[index], falseId: falseIds[index] });
  }
  const groups = new Map();
  const treeOrder = [];
  let previous = null;
  const closed = new Set();
  let nonContiguous = 0;
  records.forEach((record, index) => {
    if (record.treeId !== previous) {
      if (previous != null) closed.add(previous);
      if (closed.has(record.treeId)) nonContiguous += 1;
      if (!groups.has(record.treeId)) treeOrder.push(record.treeId);
      previous = record.treeId;
    }
    const group = groups.get(record.treeId) || [];
    group.push(index);
    groups.set(record.treeId, group);
  });
  const incoming = new Map(records.map((record) => [`${record.treeId}:${record.nodeId}`, 0]));
  let invalidChildCount = 0;
  let selfCount = 0;
  for (const record of records) {
    if (record.mode === "LEAF") continue;
    for (const child of [record.trueId, record.falseId]) {
      const key = `${record.treeId}:${child}`;
      if (!safeNonnegative(child) || child >= nodeCount || !keyMap.has(key)) invalidChildCount += 1;
      else {
        incoming.set(key, (incoming.get(key) || 0) + 1);
        if (key === `${record.treeId}:${record.nodeId}`) selfCount += 1;
      }
    }
  }
  const roots = [];
  let rootMismatchCount = 0;
  let sequentialFail = false;
  for (const treeId of treeOrder) {
    const group = groups.get(treeId) || [];
    if (!group.length) continue;
    roots.push(group[0]);
    const structural = group.filter((index) => incoming.get(`${records[index].treeId}:${records[index].nodeId}`) === 0);
    if (structural.length !== 1 || structural[0] !== group[0]) rootMismatchCount += 1;
    if (!classifier && group.map((index) => records[index].nodeId).sort((a, b) => a - b).some((value, index) => value !== index)) sequentialFail = true;
  }
  const traversal = traverseLegacy(records, roots, keyMap, invalidChildCount === 0);
  const multipleParentCount = [...incoming.values()].filter((count) => count > 1).length;
  const targetCount = Number(nTargets || 0);
  const weightsByLeaf = new Map();
  let invalidWeightReferenceCount = 0;
  let invalidWeightIdCount = 0;
  let nonLeafWeightCount = 0;
  let unusedWeightCount = 0;
  for (let index = 0; index < weightCount; index += 1) {
    const key = `${weightTreeIds[index]}:${weightNodeIds[index]}`;
    const nodeIndex = keyMap.get(key);
    if (!safeNonnegative(weightIds[index]) || weightIds[index] >= targetCount) invalidWeightIdCount += 1;
    if (nodeIndex == null) { invalidWeightReferenceCount += 1; unusedWeightCount += 1; continue; }
    if (records[nodeIndex].mode !== "LEAF") { nonLeafWeightCount += 1; unusedWeightCount += 1; continue; }
    if (!traversal.reachable.has(nodeIndex)) { unusedWeightCount += 1; continue; }
    const entries = weightsByLeaf.get(nodeIndex) || [];
    entries.push(index);
    weightsByLeaf.set(nodeIndex, entries);
  }
  let singleTargetIgnoredWeightCount = 0;
  if (targetCount === 1) for (const entries of weightsByLeaf.values()) singleTargetIgnoredWeightCount += Math.max(0, entries.length - 1);
  unusedWeightCount += singleTargetIgnoredWeightCount;
  const usedWeightCount = [...weightsByLeaf.values()].reduce((sum, entries) => sum + entries.length, 0) - singleTargetIgnoredWeightCount;
  const cpuGap = !classifier && LEGACY_DTYPES.has(input.dtype) && !["FLOAT32", "FLOAT64"].includes(input.dtype);
  const baseFail = base.length > 0 && base.length !== targetCount && !(classifier && targetCount === 2 && base.length === 1);
  const rankFail = schemaVersion >= 3 && input.rank != null && input.rank !== 2;
  const labelFail = classifier && (labelSources !== 1 || targetCount < 1);
  const fail = !input.validLegacy || ![1, 3, 5].includes(schemaVersion) || nodeCardinalityFail || weightCardinalityFail
    || modes.some((mode) => !LEGACY_MODES.has(mode)) || missing.some((value) => ![0, 1].includes(value))
    || duplicateNodeCount > 0 || invalidFeatureCount > 0 || nonContiguous > 0 || invalidChildCount > 0 || selfCount > 0
    || rootMismatchCount > 0 || traversal.cycleCount > 0 || traversal.reachable.size !== nodeCount || sequentialFail
    || invalidWeightReferenceCount > 0 || invalidWeightIdCount > 0 || targetCount < 1 || baseFail || rankFail || labelFail;
  const duplicateLabels = classifier ? labels.length - new Set(labels.map(text)).size : 0;
  const partial = !fail && (cpuGap || schemaVersion === 5 || duplicateLabels > 0 || nonLeafWeightCount > 0
    || singleTargetIgnoredWeightCount > 0 || multipleParentCount > 0 || classifier && targetCount === 2 && base.length === 1
    || schemaVersion === 1 && input.rank != null && ![1, 2].includes(input.rank));
  const batch = input.rank === 1 ? 1 : input.rank === 2 ? input.batch : null;
  const outputShape = batch != null && targetCount > 0 && (input.rank === 2 || schemaVersion === 1 && input.rank === 1)
    ? [batch, targetCount] : [];
  const nonFiniteCount = [...nodeValues, ...weights, ...base].filter((value) => !Number.isFinite(value)).length;
  const represented = new Set(weightIds);
  return {
    valid: true, fail, partial, schemaVersion, encoding: "legacy_tuple_v1_v3_v5", deprecated: schemaVersion === 5,
    aggregate, postTransform, baseCount: base.length, targetCount, classCount: classifier ? targetCount : 0,
    duplicateLabels, treeCount: roots.length, nodeCount, branchCount: modes.filter((mode) => mode !== "LEAF").length,
    leafCount: modes.filter((mode) => mode === "LEAF").length, reachableNodeCount: traversal.reachable.size,
    reachableLeafCount: traversal.leaves.size, orphanCount: Math.max(0, nodeCount - traversal.reachable.size),
    maxDepth: traversal.maxDepth, cycleCount: traversal.cycleCount, duplicateNodeCount, invalidChildCount,
    invalidFeatureCount, rootMismatchCount, multipleParentCount, invalidWeightReferenceCount, invalidWeightIdCount,
    singleTargetIgnoredWeightCount, nonLeafWeightCount, weightCount, usedWeightCount, unusedWeightCount,
    memberNodeCount: 0, memberSetCount: 0, memberValueCount: 0, memberDuplicateCount: 0, memberSeparatorCount: 0,
    nonFiniteCount, cpuGap, outputShape, classifier, labels, representedCount: represented.size,
  };
}

function genericFacts(attrs, input) {
  const featureIds = ints(attrs.get("nodes_featureids"));
  const splits = tensorNumbers(attrs.get("nodes_splits"));
  const modes = tensorNumbers(attrs.get("nodes_modes"));
  const trueIds = ints(attrs.get("nodes_truenodeids"));
  const falseIds = ints(attrs.get("nodes_falsenodeids"));
  const trueLeafs = ints(attrs.get("nodes_trueleafs"));
  const falseLeafs = ints(attrs.get("nodes_falseleafs"));
  const missing = ints(attrs.get("nodes_missing_value_tracks_true"));
  const roots = ints(attrs.get("tree_roots"));
  const leafTargetIds = ints(attrs.get("leaf_targetids"));
  const leafWeights = tensorNumbers(attrs.get("leaf_weights"));
  const membershipValues = tensorNumbers(attrs.get("membership_values"));
  const nTargets = safeInteger(integer(attrs.get("n_targets"), 0n));
  const aggregateValue = integer(attrs.get("aggregate_function"), 1n);
  const postValue = integer(attrs.get("post_transform"), 0n);
  const arrays = [featureIds, splits, modes, trueIds, falseIds, trueLeafs, falseLeafs, missing, roots,
    leafTargetIds, leafWeights, membershipValues];
  if (arrays.some((value) => value == null) || nTargets == null || aggregateValue == null || postValue == null) return invalid();
  const aggregate = new Map([[0n, "AVERAGE"], [1n, "SUM"], [2n, "MIN"], [3n, "MAX"]]).get(aggregateValue) || "INVALID";
  const postTransform = new Map([[0n, "NONE"], [1n, "SOFTMAX"], [2n, "LOGISTIC"], [3n, "SOFTMAX_ZERO"], [4n, "PROBIT"]]).get(postValue) || "INVALID";
  const nodeCount = splits.length;
  const nodeCardinalityFail = !nodeCount || [featureIds, modes, trueIds, falseIds, trueLeafs, falseLeafs]
    .some((values) => values.length !== nodeCount) || missing.length > 0 && missing.length !== nodeCount;
  const leafCount = leafWeights.length;
  const member = membershipFacts(membershipValues, modes);
  let invalidFeatureCount = 0;
  featureIds.forEach((value) => { if (!safeNonnegative(value) || input.features != null && value >= input.features) invalidFeatureCount += 1; });
  let invalidChildCount = 0;
  for (let index = 0; index < nodeCount; index += 1) {
    for (const [child, leaf] of [[trueIds[index], trueLeafs[index]], [falseIds[index], falseLeafs[index]]]) {
      if (!safeNonnegative(child) || child >= (leaf ? leafCount : nodeCount)) invalidChildCount += 1;
    }
  }
  const rootMismatchCount = roots.filter((value) => !safeNonnegative(value) || value >= nodeCount).length;
  const traversal = traverseGeneric({ trueIds, falseIds, trueLeafs, falseLeafs, roots }, nodeCount, leafCount,
    invalidChildCount === 0 && rootMismatchCount === 0);
  const incoming = new Array(nodeCount).fill(0);
  for (let index = 0; index < nodeCount; index += 1) {
    if (!trueLeafs[index] && safeNonnegative(trueIds[index]) && trueIds[index] < nodeCount) incoming[trueIds[index]] += 1;
    if (!falseLeafs[index] && safeNonnegative(falseIds[index]) && falseIds[index] < nodeCount) incoming[falseIds[index]] += 1;
  }
  const multipleParentCount = incoming.filter((count) => count > 1).length;
  const orphanNodes = Math.max(0, nodeCount - traversal.nodes.size);
  const orphanLeaves = Math.max(0, leafCount - traversal.leaves.size);
  const cpuGap = input.dtype === "FLOAT16";
  const fail = !input.validGeneric || input.rank != null && input.rank !== 2 || nTargets < 1
    || aggregate === "INVALID" || postTransform === "INVALID" || nodeCardinalityFail
    || modes.some((value) => !safeNonnegative(value) || value > 6)
    || [...trueLeafs, ...falseLeafs, ...missing].some((value) => ![0, 1].includes(value))
    || leafTargetIds.length !== leafCount || leafTargetIds.some((value) => !safeNonnegative(value) || value >= nTargets)
    || invalidFeatureCount > 0 || invalidChildCount > 0 || !roots.length || rootMismatchCount > 0
    || traversal.cycleCount > 0 || member.invalid;
  const duplicateRoots = roots.length - new Set(roots).size;
  const partial = !fail && (cpuGap || orphanNodes + orphanLeaves > 0 || multipleParentCount > 0 || duplicateRoots > 0);
  const outputShape = input.rank === 2 && nTargets > 0 ? [input.batch, nTargets] : [];
  const nonFiniteCount = [...splits, ...leafWeights, ...member.sets.flat()].filter((value) => !Number.isFinite(value)).length;
  return {
    valid: true, fail, partial, schemaVersion: 5, encoding: "indexed_branch_leaf_v5", deprecated: false,
    aggregate, postTransform, baseCount: 0, targetCount: nTargets, classCount: 0, duplicateLabels: 0,
    treeCount: roots.length, nodeCount, branchCount: nodeCount, leafCount,
    reachableNodeCount: traversal.nodes.size, reachableLeafCount: traversal.leaves.size,
    orphanCount: orphanNodes + orphanLeaves, maxDepth: traversal.maxDepth, cycleCount: traversal.cycleCount,
    duplicateNodeCount: 0, invalidChildCount, invalidFeatureCount, rootMismatchCount,
    multipleParentCount, invalidWeightReferenceCount: 0, invalidWeightIdCount: 0, singleTargetIgnoredWeightCount: 0,
    nonLeafWeightCount: 0, weightCount: leafCount, usedWeightCount: traversal.leaves.size, unusedWeightCount: orphanLeaves,
    memberNodeCount: member.nodeCount, memberSetCount: member.sets.length, memberValueCount: member.valueCount,
    memberDuplicateCount: member.duplicateCount, memberSeparatorCount: member.separatorCount,
    memberZeroCount: member.zeroCount, nonFiniteCount, cpuGap, outputShape, classifier: false,
  };
}

function traverseLegacy(records, roots, keyMap, enabled) {
  const reachable = new Set();
  const leaves = new Set();
  const active = new Set();
  let cycleCount = 0;
  let maxDepth = 0;
  function visit(index, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (active.has(index)) { cycleCount += 1; return; }
    if (reachable.has(index) || index == null) return;
    active.add(index); reachable.add(index);
    const record = records[index];
    if (record.mode === "LEAF") leaves.add(index);
    else {
      visit(keyMap.get(`${record.treeId}:${record.falseId}`), depth + 1);
      visit(keyMap.get(`${record.treeId}:${record.trueId}`), depth + 1);
    }
    active.delete(index);
  }
  if (enabled) roots.forEach((root) => visit(root, 1));
  return { reachable, leaves, cycleCount, maxDepth };
}

function traverseGeneric(graph, nodeCount, leafCount, enabled) {
  const nodes = new Set();
  const leaves = new Set();
  const active = new Set();
  let cycleCount = 0;
  let maxDepth = 0;
  function visit(index, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (active.has(index)) { cycleCount += 1; return; }
    if (nodes.has(index) || index < 0 || index >= nodeCount) return;
    active.add(index); nodes.add(index);
    const degenerate = graph.trueLeafs[index] === 1 && graph.falseLeafs[index] === 1
      && graph.trueIds[index] === graph.falseIds[index];
    if (degenerate) leaves.add(graph.trueIds[index]);
    else {
      for (const [child, leaf] of [[graph.falseIds[index], graph.falseLeafs[index]], [graph.trueIds[index], graph.trueLeafs[index]]]) {
        if (leaf && child >= 0 && child < leafCount) leaves.add(child); else if (!leaf) visit(child, depth + 1);
      }
    }
    active.delete(index);
  }
  if (enabled) graph.roots.forEach((root) => visit(root, 1));
  return { nodes, leaves, cycleCount, maxDepth };
}

function membershipFacts(values, modes) {
  const nodeCount = modes.filter((mode) => mode === 6).length;
  const sets = [];
  let current = [];
  let separatorCount = 0;
  let valueCount = 0;
  let zeroCount = 0;
  values.forEach((value) => {
    if (Number.isNaN(value)) { sets.push(current); current = []; separatorCount += 1; }
    else { current.push(value); valueCount += 1; if (value === 0) zeroCount += 1; }
  });
  const duplicateCount = sets.reduce((sum, set) => sum + set.length - new Set(set.map(text)).size, 0);
  return { nodeCount, sets, separatorCount, valueCount, zeroCount, duplicateCount,
    invalid: current.length > 0 || separatorCount !== nodeCount || sets.length !== nodeCount };
}

function structuralRisks(row, facts) {
  const risks = [];
  if (facts.deprecated) risks.push("tree_ensemble_legacy_operator_deprecated_at_opset_5");
  if (facts.cpuGap) risks.push(row.op_name === "TreeEnsemble"
    ? "tree_ensemble_v5_float16_missing_pinned_ort_cpu_kernel" : "tree_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  if (facts.classifier && facts.duplicateLabels) risks.push("tree_classifier_duplicate_labels_ambiguous_output_semantics");
  if (facts.nonLeafWeightCount) risks.push("tree_ensemble_nonleaf_weights_ignored_by_pinned_ort");
  if (facts.singleTargetIgnoredWeightCount) risks.push("tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort");
  if (facts.multipleParentCount) risks.push(row.op_name === "TreeEnsemble"
    ? "tree_ensemble_v5_shared_subtree_not_strict_tree" : "tree_ensemble_shared_subtree_not_strict_tree");
  if (row.op_name === "TreeEnsemble" && facts.orphanCount) risks.push("tree_ensemble_v5_unreachable_serialized_nodes_or_leaves");
  if (row.op_name === "TreeEnsemble" && facts.memberDuplicateCount) risks.push("tree_ensemble_v5_duplicate_membership_values");
  if (row.op_name === "TreeEnsemble" && facts.memberZeroCount) risks.push("tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser");
  if (facts.nonFiniteCount || Number(row.tree_reference_non_finite_score_count || 0)) risks.push("tree_ensemble_non_finite_parameter_or_reference_score");
  if (Number(row.tree_reference_decision_boundary_count || 0)) risks.push("tree_ensemble_reference_decision_boundary");
  if (Number(row.tree_reference_unwritten_score_count || 0)) risks.push("tree_classifier_binary_post_transform_leaves_score_unwritten");
  if (row.op_name === "TreeEnsembleRegressor" && facts.targetCount === 1 && !["NONE", "PROBIT"].includes(facts.postTransform)) risks.push("tree_regressor_single_target_post_transform_noop");
  if (row.op_name === "TreeEnsemble" && facts.targetCount === 1 && !["NONE", "PROBIT"].includes(facts.postTransform)) risks.push("tree_ensemble_v5_single_target_post_transform_noop");
  if (facts.classifier && facts.targetCount === 2 && facts.baseCount === 1) risks.push("tree_classifier_binary_single_base_value_semantics_underspecified");
  if (facts.classifier && facts.targetCount === 2 && facts.representedCount > 1) {
    const canonical = facts.labels.length === 2 && facts.labels[0] === 0n && facts.labels[1] === 1n;
    if (facts.labels.some((value) => typeof value !== "bigint") || !canonical) risks.push("tree_classifier_pinned_ort_binary_label_index_semantics");
  }
  return risks;
}

function inputContract(input) {
  const shape = input?.shape_declared === true ? [...(input.shape || [])] : [];
  const rank = input?.shape_declared === true ? shape.length : null;
  const dtype = input?.dtype || "UNKNOWN";
  const batch = rank === 1 ? 1 : rank != null && rank >= 2 ? shape[0] : null;
  const features = rank === 1 ? shape[0] : rank != null && rank >= 2 && shape.slice(1).every(knownDimension)
    ? product(shape.slice(1)) : null;
  return {
    dtype, shape, rank, batch, features,
    validLegacy: Boolean(input) && input.value_kind === "tensor" && LEGACY_DTYPES.has(dtype),
    validGeneric: Boolean(input) && input.value_kind === "tensor" && GENERIC_DTYPES.has(dtype),
  };
}

function numericChoice(attrs, listName, tensorName) {
  if (attrs.has(listName) && attrs.has(tensorName)) return null;
  return attrs.has(tensorName) ? tensorNumbers(attrs.get(tensorName)) : floats(attrs.get(listName));
}

function ints(attribute) {
  const exact = exactInts(attribute);
  if (exact == null) return null;
  return exact.map(safeInteger);
}

function exactInts(attribute) {
  if (!attribute) return [];
  if (attribute.type !== 7 || !Array.isArray(attribute.int_values_exact_decimal)) return null;
  try { return attribute.int_values_exact_decimal.map((value) => BigInt(value)); } catch { return null; }
}

function floats(attribute) {
  if (!attribute) return [];
  if (attribute.type !== 6 || !Array.isArray(attribute.float_values_text)) return null;
  return attribute.float_values_text.map(number);
}

function strings(attribute) {
  if (!attribute) return [];
  return attribute.type === 8 && Array.isArray(attribute.string_values) ? [...attribute.string_values] : null;
}

function tensorNumbers(attribute) {
  if (!attribute) return [];
  const tensor = attribute.tensor_value;
  if (attribute.type !== 4 || !tensor || tensor.exact_values_complete !== true || !Array.isArray(tensor.exact_values_text)) return null;
  return tensor.exact_values_text.map(number);
}

function integer(attribute, fallback) {
  if (!attribute) return fallback;
  try { return attribute.type === 2 ? BigInt(attribute.int_value_exact_decimal) : null; } catch { return null; }
}

function string(attribute, fallback) {
  if (!attribute) return fallback;
  return attribute.type === 3 && typeof attribute.string_value === "string" ? attribute.string_value : null;
}

function staticValueCount(input) {
  if (input?.static_values_complete === true && Array.isArray(input.static_values)) return input.static_values.length;
  if (input?.static_values_canonical_text_complete === true && Array.isArray(input.static_values_canonical_texts)) return input.static_values_canonical_texts.length;
  if (input?.initializer_integer_values_exact_complete === true && Array.isArray(input.initializer_integer_values_exact_decimals)) return input.initializer_integer_values_exact_decimals.length;
  return null;
}

function number(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

function safeInteger(value) {
  if (typeof value === "number") return Number.isSafeInteger(value) ? value : null;
  return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function safeNonnegative(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function knownDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function product(values) {
  let result = 1;
  for (const value of values) {
    if (!knownDimension(value) || !Number.isSafeInteger(result * value)) return null;
    result *= value;
  }
  return result;
}

function text(value) {
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function invalid() {
  return { valid: false };
}
