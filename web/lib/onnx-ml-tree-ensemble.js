import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

const LEGACY_INPUT_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const GENERIC_INPUT_DTYPES = new Set(["FLOAT16", "FLOAT32", "FLOAT64"]);
const LEGACY_MODES = new Set([
  "BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "BRANCH_GT", "BRANCH_EQ", "BRANCH_NEQ", "LEAF",
]);
const LEGACY_POST_TRANSFORMS = new Set(["NONE", "SOFTMAX", "LOGISTIC", "SOFTMAX_ZERO", "PROBIT"]);
const LEGACY_AGGREGATES = new Set(["AVERAGE", "SUM", "MIN", "MAX"]);
const MAX_REFERENCE_WORK = 1_000_000;
const PREVIEW_LIMIT = 64;

export function resolveOnnxMlTreeEnsembleVersion(opName, importedOpset) {
  if (!Number.isSafeInteger(importedOpset) || importedOpset < 1) return null;
  if (opName === "TreeEnsemble") return importedOpset >= 5 ? 5 : null;
  if (!["TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(opName)) return null;
  return importedOpset >= 5 ? 5 : importedOpset >= 3 ? 3 : 1;
}

export function inferOnnxMlTreeEnsemble(context) {
  return inferGenericTreeEnsemble(context);
}

export function inferOnnxMlTreeEnsembleClassifier(context) {
  return inferLegacyTreeEnsemble({ ...context, classifier: true });
}

export function inferOnnxMlTreeEnsembleRegressor(context) {
  return inferLegacyTreeEnsemble({ ...context, classifier: false });
}

function inferLegacyTreeEnsemble({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph", classifier }) {
  const opName = classifier ? "TreeEnsembleClassifier" : "TreeEnsembleRegressor";
  const schemaVersion = resolveOnnxMlTreeEnsembleVersion(opName, importedOpset);
  const input = inputContract(node, tensorMap, LEGACY_INPUT_DTYPES);
  const failures = [...input.failures];
  const reasons = [...input.reasons];
  if (schemaVersion == null) failures.push("tree_ensemble_operator_not_defined_at_imported_opset");
  if (schemaVersion >= 3 && input.rank != null && input.rank !== 2) failures.push("tree_ensemble_input_rank_not_2");
  if (schemaVersion === 1 && input.rank != null && ![1, 2].includes(input.rank)) reasons.push("tree_ensemble_v1_input_rank_outside_documented_contract");

  const parsed = parseLegacyAttributes(node, schemaVersion, classifier, failures);
  const graph = validateLegacyGraph(parsed, input.featureCount, classifier, failures, reasons);
  const targetCount = classifier ? parsed.labels.values.length : parsed.nTargets;
  if (targetCount < 1) failures.push(classifier ? "tree_classifier_class_count_not_positive" : "tree_regressor_n_targets_not_positive");
  const cpuDtypeGap = !classifier && LEGACY_INPUT_DTYPES.has(input.dtype) && !["FLOAT32", "FLOAT64"].includes(input.dtype);
  if (cpuDtypeGap) reasons.push("tree_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  if (classifier && parsed.labels.duplicateCount > 0) reasons.push("tree_classifier_duplicate_labels_ambiguous_output_semantics");
  if (schemaVersion === 5) reasons.push("tree_ensemble_legacy_operator_deprecated_at_opset_5");

  validateLegacyBaseValues(parsed.baseValues, targetCount, classifier, failures, reasons);
  if (!LEGACY_POST_TRANSFORMS.has(parsed.postTransform)) failures.push("tree_ensemble_invalid_post_transform");
  if (!classifier && !LEGACY_AGGREGATES.has(parsed.aggregateFunction)) failures.push("tree_regressor_invalid_aggregate_function");

  const outputShapes = legacyOutputShapes(input, targetCount, classifier, schemaVersion);
  const labelDtype = parsed.labels.kind === "string" ? "STRING" : "INT64";
  const outputDtype = classifier ? "FLOAT32" : "FLOAT32";
  const outputType = makeOnnxTensorType(outputDtype, outputShapes.value, outputShapes.declared);
  const labelType = classifier ? makeOnnxTensorType(labelDtype, outputShapes.label, outputShapes.declared) : null;
  const reference = failures.length || cpuDtypeGap
    ? unresolvedReference(failures.length ? "not_assessed_invalid_contract" : "not_assessed_pinned_cpu_dtype_gap")
    : evaluateLegacyReference({ input, parsed, graph, classifier, targetCount });
  if (reference.status === "not_assessed_work_limit") reasons.push("tree_ensemble_reference_work_limit");

  const risks = legacyRisks({ parsed, graph, classifier, schemaVersion, cpuDtypeGap, reference, targetCount });
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = treeBaseRow({
    scope, nodeIndex, importedOpset, schemaVersion, opName,
    contractKind: classifier ? "tree_ensemble_classifier" : "tree_ensemble_regressor",
    status, input, outputName: node.outputs?.[classifier ? 1 : 0] || "", outputDtype,
    outputShape: outputShapes.value, outputShapeDeclared: outputShapes.declared,
    failures, reasons,
  });
  Object.assign(row, {
    output_names: [...(node.outputs || [])],
    canonical_output_types: classifier
      ? [canonicalOnnxTypeProto(labelType), canonicalOnnxTypeProto(outputType)]
      : [canonicalOnnxTypeProto(outputType)],
    canonical_output_shapes: classifier ? [outputShapes.label, outputShapes.value] : [outputShapes.value],
    tree_encoding: "legacy_tuple_v1_v3_v5",
    tree_deprecated_operator: schemaVersion === 5,
    tree_onnx_contract_status: failures.length ? "fail" : "pass",
    tree_onnx_contract_reason: failures[0] || "",
    tree_pinned_ort_contract_status: failures.length || cpuDtypeGap ? "fail" : "pass",
    tree_pinned_ort_contract_reason: failures[0] || (cpuDtypeGap ? "schema_dtype_missing_pinned_ort_cpu_kernel" : ""),
    tree_aggregate_function: classifier ? "SUM" : parsed.aggregateFunction,
    tree_aggregate_function_source: classifier ? "pinned_classifier_sum" : parsed.aggregateSource,
    tree_post_transform: parsed.postTransform,
    tree_post_transform_source: parsed.postTransformSource,
    tree_base_value_count: parsed.baseValues.values.length,
    tree_base_value_source: parsed.baseValues.source,
    tree_class_or_target_count: targetCount,
    tree_class_label_kind: classifier ? parsed.labels.kind : "not_applicable",
    tree_class_label_count: classifier ? parsed.labels.values.length : 0,
    tree_duplicate_class_label_count: classifier ? parsed.labels.duplicateCount : 0,
    tree_class_label_preview: classifier ? parsed.labels.values.slice(0, PREVIEW_LIMIT).map(valueText) : [],
    ...graphRow(graph),
    tree_weight_tuple_count: parsed.weights.count,
    tree_used_weight_count: graph.usedWeightCount,
    tree_ignored_nonleaf_weight_count: graph.nonLeafWeightCount,
    tree_unresolved_weight_count: Math.max(0, parsed.weights.count - graph.usedWeightCount - graph.unusedWeightCount),
    tree_unused_weight_count: graph.unusedWeightCount,
    tree_non_finite_parameter_count: parsed.nonFiniteParameterCount,
    tree_pinned_cpu_dtype_gap: cpuDtypeGap,
    ...referenceRow(reference),
    risk_codes: risks,
  });

  const canPropagate = !failures.length && outputShapes.declared;
  const outputs = [];
  if (canPropagate && classifier && node.outputs?.[0]) outputs.push([node.outputs[0], onnxValueDescriptorFromType(labelType)]);
  if (canPropagate && node.outputs?.[classifier ? 1 : 0]) {
    outputs.push([node.outputs[classifier ? 1 : 0], onnxValueDescriptorFromType(outputType)]);
  }
  return { status, reason: row.reason_codes[0] || "", result: { outputs }, row };
}

function inferGenericTreeEnsemble({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const schemaVersion = resolveOnnxMlTreeEnsembleVersion("TreeEnsemble", importedOpset);
  const input = inputContract(node, tensorMap, GENERIC_INPUT_DTYPES);
  const failures = [...input.failures];
  const reasons = [...input.reasons];
  if (schemaVersion == null) failures.push("tree_ensemble_operator_not_defined_at_imported_opset");
  if (input.rank != null && input.rank !== 2) failures.push("tree_ensemble_v5_input_rank_not_2");
  const parsed = parseGenericAttributes(node, input.dtype, failures);
  const graph = validateGenericGraph(parsed, input.featureCount, failures, reasons);
  const cpuDtypeGap = input.dtype === "FLOAT16";
  if (cpuDtypeGap) reasons.push("tree_ensemble_v5_float16_missing_pinned_ort_cpu_kernel");
  const outputShape = input.rank === 2 ? [input.batchCount, parsed.nTargets > 0 ? parsed.nTargets : null] : [];
  const outputShapeDeclared = input.rank === 2 && parsed.nTargets > 0;
  const outputType = makeOnnxTensorType(input.dtype, outputShape, outputShapeDeclared);
  const reference = failures.length || cpuDtypeGap
    ? unresolvedReference(failures.length ? "not_assessed_invalid_contract" : "not_assessed_pinned_cpu_dtype_gap")
    : evaluateGenericReference({ input, parsed, graph });
  if (reference.status === "not_assessed_work_limit") reasons.push("tree_ensemble_reference_work_limit");
  const risks = genericRisks({ parsed, graph, cpuDtypeGap, reference });
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = treeBaseRow({
    scope, nodeIndex, importedOpset, schemaVersion, opName: "TreeEnsemble",
    contractKind: "tree_ensemble_v5", status, input,
    outputName: node.outputs?.[0] || "", outputDtype: input.dtype,
    outputShape, outputShapeDeclared, failures, reasons,
  });
  Object.assign(row, {
    output_names: [...(node.outputs || [])],
    canonical_output_types: [canonicalOnnxTypeProto(outputType)],
    canonical_output_shapes: [outputShape],
    tree_encoding: "indexed_branch_leaf_v5",
    tree_deprecated_operator: false,
    tree_onnx_contract_status: failures.length ? "fail" : "pass",
    tree_onnx_contract_reason: failures[0] || "",
    tree_pinned_ort_contract_status: failures.length || cpuDtypeGap ? "fail" : "pass",
    tree_pinned_ort_contract_reason: failures[0] || (cpuDtypeGap ? "schema_dtype_missing_pinned_ort_cpu_kernel" : ""),
    tree_aggregate_function: parsed.aggregateFunction,
    tree_aggregate_function_source: parsed.aggregateSource,
    tree_post_transform: parsed.postTransform,
    tree_post_transform_source: parsed.postTransformSource,
    tree_base_value_count: 0,
    tree_base_value_source: "not_applicable",
    tree_class_or_target_count: parsed.nTargets,
    tree_class_label_kind: "not_applicable",
    tree_class_label_count: 0,
    tree_duplicate_class_label_count: 0,
    tree_class_label_preview: [],
    ...graphRow(graph),
    tree_weight_tuple_count: parsed.leafWeights.values.length,
    tree_used_weight_count: graph.usedWeightCount,
    tree_ignored_nonleaf_weight_count: 0,
    tree_unresolved_weight_count: Math.max(0, parsed.leafWeights.values.length - graph.usedWeightCount - graph.unusedWeightCount),
    tree_unused_weight_count: graph.unusedWeightCount,
    tree_membership_node_count: parsed.membership.memberNodeCount,
    tree_membership_set_count: parsed.membership.sets.length,
    tree_membership_value_count: parsed.membership.valueCount,
    tree_membership_duplicate_value_count: parsed.membership.duplicateCount,
    tree_membership_separator_count: parsed.membership.separatorCount,
    tree_non_finite_parameter_count: parsed.nonFiniteParameterCount,
    tree_pinned_cpu_dtype_gap: cpuDtypeGap,
    ...referenceRow(reference),
    risk_codes: risks,
  });
  const outputs = !failures.length && outputShapeDeclared && node.outputs?.[0]
    ? [[node.outputs[0], onnxValueDescriptorFromType(outputType)]] : [];
  return { status, reason: row.reason_codes[0] || "", result: { outputs }, row };
}

function inputContract(node, tensorMap, allowedDtypes) {
  const input = tensorMap.get(node.inputs?.[0]);
  const type = onnxTypeProtoFromValue(input);
  const failures = [];
  const reasons = [];
  const dtype = type?.kind === "tensor" ? type.dtype || "UNKNOWN" : "UNKNOWN";
  const shapeDeclared = type?.kind === "tensor" && type.shapeDeclared === true;
  const shape = shapeDeclared ? [...(type.shape || [])] : [];
  const rank = shapeDeclared ? shape.length : null;
  if (!input) failures.push("tree_ensemble_input_missing");
  else if (!type) reasons.push("tree_ensemble_input_type_unresolved");
  else if (type.kind !== "tensor") failures.push(`tree_ensemble_input_not_tensor:${type.kind}`);
  if (dtype === "UNKNOWN") reasons.push("tree_ensemble_input_dtype_unresolved");
  else if (!allowedDtypes.has(dtype)) failures.push(`tree_ensemble_input_dtype_not_supported:${dtype}`);
  if (!shapeDeclared) reasons.push("tree_ensemble_input_shape_unresolved");
  const batchCount = rank === 1 ? 1 : rank != null && rank >= 2 ? shape[0] : null;
  const featureCount = rank === 1 ? shape[0] : rank != null && rank >= 2
    ? shape.slice(1).every(knownDimension) ? safeShapeElementCount(shape.slice(1)) : null : null;
  return { input, type, dtype, shapeDeclared, shape, rank, batchCount, featureCount, failures, reasons };
}

function parseLegacyAttributes(node, schemaVersion, classifier, failures) {
  const nodes = {
    treeIds: intList(node, "nodes_treeids", failures),
    nodeIds: intList(node, "nodes_nodeids", failures),
    featureIds: intList(node, "nodes_featureids", failures),
    modes: stringList(node, "nodes_modes", failures),
    trueNodeIds: intList(node, "nodes_truenodeids", failures),
    falseNodeIds: intList(node, "nodes_falsenodeids", failures),
    missingTracksTrue: intList(node, "nodes_missing_value_tracks_true", failures),
  };
  nodes.values = numericChoice(node, "nodes_values", "nodes_values_as_tensor", schemaVersion, failures);
  const prefix = classifier ? "class" : "target";
  const weights = {
    treeIds: intList(node, `${prefix}_treeids`, failures),
    nodeIds: intList(node, `${prefix}_nodeids`, failures),
    ids: intList(node, `${prefix}_ids`, failures),
    values: numericChoice(node, `${prefix}_weights`, `${prefix}_weights_as_tensor`, schemaVersion, failures),
  };
  weights.count = weights.ids.length;
  const baseValues = numericChoice(node, "base_values", "base_values_as_tensor", schemaVersion, failures);
  const labels = classifier ? classifierLabels(node, failures) : { kind: "not_applicable", values: [], duplicateCount: 0 };
  const nTargetsRaw = classifier ? null : intScalar(node, "n_targets", 0n, failures);
  const nTargets = nTargetsRaw != null && nTargetsRaw >= 0n && nTargetsRaw <= BigInt(Number.MAX_SAFE_INTEGER)
    ? Number(nTargetsRaw) : 0;
  if (!classifier && (nTargetsRaw == null || nTargetsRaw <= 0n || nTargetsRaw > BigInt(Number.MAX_SAFE_INTEGER))) {
    failures.push("tree_regressor_n_targets_invalid");
  }
  const postTransform = stringScalar(node, "post_transform", "NONE", failures);
  const aggregateFunction = classifier ? "SUM" : stringScalar(node, "aggregate_function", "SUM", failures);
  const parameterValues = [...nodes.values.values, ...weights.values.values, ...baseValues.values];
  return {
    nodes, weights, baseValues, labels, nTargets,
    postTransform, postTransformSource: node.attributes?.has("post_transform") ? "explicit_attribute" : "onnx_schema_default_NONE",
    aggregateFunction, aggregateSource: classifier ? "pinned_classifier_sum"
      : node.attributes?.has("aggregate_function") ? "explicit_attribute" : "onnx_schema_default_SUM",
    nonFiniteParameterCount: parameterValues.filter((value) => !Number.isFinite(value)).length,
  };
}

function parseGenericAttributes(node, inputDtype, failures) {
  const nodes = {
    featureIds: intList(node, "nodes_featureids", failures),
    splits: tensorNumeric(node, "nodes_splits", failures),
    modes: tensorInteger(node, "nodes_modes", failures),
    trueNodeIds: intList(node, "nodes_truenodeids", failures),
    falseNodeIds: intList(node, "nodes_falsenodeids", failures),
    trueLeafs: intList(node, "nodes_trueleafs", failures),
    falseLeafs: intList(node, "nodes_falseleafs", failures),
    missingTracksTrue: intList(node, "nodes_missing_value_tracks_true", failures),
    hitrates: tensorNumeric(node, "nodes_hitrates", failures, true),
  };
  const leafTargetIds = intList(node, "leaf_targetids", failures);
  const leafWeights = tensorNumeric(node, "leaf_weights", failures);
  const treeRoots = intList(node, "tree_roots", failures);
  const membershipTensor = tensorNumeric(node, "membership_values", failures, true);
  for (const [name, tensor] of [["nodes_splits", nodes.splits], ["leaf_weights", leafWeights], ["membership_values", membershipTensor]]) {
    if (tensor.present && tensor.dtype !== inputDtype) failures.push(`tree_ensemble_v5_${name}_dtype_mismatch:${tensor.dtype}:${inputDtype}`);
  }
  if (nodes.hitrates.present && nodes.hitrates.dtype !== "FLOAT32") failures.push("tree_ensemble_v5_nodes_hitrates_dtype_not_float32");
  if (nodes.modes.present && nodes.modes.dtype !== "UINT8") failures.push("tree_ensemble_v5_nodes_modes_dtype_not_uint8");
  const nTargetsRaw = intScalar(node, "n_targets", 0n, failures);
  const nTargets = nTargetsRaw != null && nTargetsRaw > 0n && nTargetsRaw <= BigInt(Number.MAX_SAFE_INTEGER)
    ? Number(nTargetsRaw) : 0;
  if (nTargets === 0) failures.push("tree_ensemble_v5_n_targets_not_positive");
  const aggregateRaw = intScalar(node, "aggregate_function", 1n, failures);
  const postRaw = intScalar(node, "post_transform", 0n, failures);
  const aggregateMap = new Map([[0n, "AVERAGE"], [1n, "SUM"], [2n, "MIN"], [3n, "MAX"]]);
  const postMap = new Map([[0n, "NONE"], [1n, "SOFTMAX"], [2n, "LOGISTIC"], [3n, "SOFTMAX_ZERO"], [4n, "PROBIT"]]);
  const aggregateFunction = aggregateMap.get(aggregateRaw) || "INVALID";
  const postTransform = postMap.get(postRaw) || "INVALID";
  if (aggregateFunction === "INVALID") failures.push("tree_ensemble_v5_invalid_aggregate_function");
  if (postTransform === "INVALID") failures.push("tree_ensemble_v5_invalid_post_transform");
  const membership = splitMembershipValues(membershipTensor.values, nodes.modes.values, failures);
  const nonFiniteParameterCount = [...nodes.splits.values, ...leafWeights.values]
    .filter((value) => !Number.isFinite(value)).length
    + membership.sets.flat().filter((value) => !Number.isFinite(value)).length;
  return {
    nodes, leafTargetIds, leafWeights, treeRoots, membership, nTargets,
    aggregateFunction, aggregateSource: node.attributes?.has("aggregate_function") ? "explicit_attribute" : "onnx_schema_default_SUM_1",
    postTransform, postTransformSource: node.attributes?.has("post_transform") ? "explicit_attribute" : "onnx_schema_default_NONE_0",
    nonFiniteParameterCount,
  };
}

function validateLegacyGraph(parsed, featureCount, classifier, failures, reasons) {
  const { nodes, weights } = parsed;
  const nodeCount = nodes.treeIds.length;
  const tupleLengths = [nodes.treeIds.length, nodes.nodeIds.length, nodes.featureIds.length, nodes.values.values.length,
    nodes.modes.length, nodes.trueNodeIds.length, nodes.falseNodeIds.length];
  if (!nodeCount || tupleLengths.some((length) => length !== nodeCount)) failures.push("tree_ensemble_node_tuple_cardinality_mismatch");
  if (nodes.missingTracksTrue.length && nodes.missingTracksTrue.length !== nodeCount) failures.push("tree_ensemble_missing_track_cardinality_mismatch");
  if (nodes.modes.some((mode) => !LEGACY_MODES.has(mode))) failures.push("tree_ensemble_invalid_node_mode");
  if (nodes.missingTracksTrue.some((value) => ![0, 1].includes(value))) failures.push("tree_ensemble_missing_track_not_boolean");
  const weightLengths = [weights.treeIds.length, weights.nodeIds.length, weights.ids.length, weights.values.values.length];
  if (!weights.count || weightLengths.some((length) => length !== weights.count)) failures.push("tree_ensemble_weight_tuple_cardinality_mismatch");

  const records = [];
  const keyToIndex = new Map();
  let duplicateNodeCount = 0;
  let invalidFeatureCount = 0;
  for (let index = 0; index < nodeCount; index += 1) {
    const treeId = nodes.treeIds[index];
    const nodeId = nodes.nodeIds[index];
    const key = `${treeId}:${nodeId}`;
    if (keyToIndex.has(key)) duplicateNodeCount += 1;
    else keyToIndex.set(key, index);
    const branch = nodes.modes[index] !== "LEAF";
    if (branch && (!Number.isSafeInteger(nodes.featureIds[index]) || nodes.featureIds[index] < 0
      || featureCount != null && nodes.featureIds[index] >= featureCount)) invalidFeatureCount += 1;
    records.push({
      index, treeId, nodeId, key, featureId: nodes.featureIds[index], mode: nodes.modes[index],
      value: nodes.values.values[index], trueNodeId: nodes.trueNodeIds[index], falseNodeId: nodes.falseNodeIds[index],
      missingTrue: nodes.missingTracksTrue[index] === 1,
    });
  }
  if (duplicateNodeCount) failures.push("tree_ensemble_duplicate_tree_node_identity");
  if (invalidFeatureCount) failures.push("tree_ensemble_feature_id_out_of_bounds");

  const treeIds = [];
  const treeGroups = new Map();
  let nonContiguousTreeCount = 0;
  let previous = null;
  const closed = new Set();
  for (const record of records) {
    if (record.treeId !== previous) {
      if (previous != null) closed.add(previous);
      if (closed.has(record.treeId)) nonContiguousTreeCount += 1;
      if (!treeGroups.has(record.treeId)) treeIds.push(record.treeId);
      previous = record.treeId;
    }
    const group = treeGroups.get(record.treeId) || [];
    group.push(record.index);
    treeGroups.set(record.treeId, group);
  }
  if (nonContiguousTreeCount) failures.push("tree_ensemble_noncontiguous_tree_serialization");

  let invalidChildReferenceCount = 0;
  let selfReferenceCount = 0;
  const incoming = new Map(records.map((record) => [record.key, 0]));
  for (const record of records) {
    if (record.mode === "LEAF") continue;
    for (const childId of [record.trueNodeId, record.falseNodeId]) {
      const childKey = `${record.treeId}:${childId}`;
      if (!Number.isSafeInteger(childId) || childId < 0 || childId >= nodeCount || !keyToIndex.has(childKey)) {
        invalidChildReferenceCount += 1;
      } else {
        incoming.set(childKey, (incoming.get(childKey) || 0) + 1);
        if (childKey === record.key) selfReferenceCount += 1;
      }
    }
  }
  if (invalidChildReferenceCount) failures.push("tree_ensemble_invalid_child_reference");
  if (selfReferenceCount) failures.push("tree_ensemble_self_reference");

  let rootMismatchCount = 0;
  const roots = [];
  for (const treeId of treeIds) {
    const group = treeGroups.get(treeId) || [];
    if (!group.length) continue;
    const structuralRoots = group.filter((index) => incoming.get(records[index].key) === 0);
    const serializedRoot = group[0];
    roots.push(serializedRoot);
    if (structuralRoots.length !== 1 || structuralRoots[0] !== serializedRoot) rootMismatchCount += 1;
    if (!classifier) {
      const sorted = group.map((index) => records[index].nodeId).sort((a, b) => a - b);
      if (sorted.some((value, index) => value !== index)) failures.push("tree_regressor_node_ids_not_zero_based_sequential");
    }
  }
  if (rootMismatchCount) failures.push("tree_ensemble_serialized_root_mismatch");

  const reachable = new Set();
  const leafIndices = new Set();
  const active = new Set();
  let cycleCount = 0;
  let maxDepth = 0;
  function visit(index, depth) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= records.length) return;
    maxDepth = Math.max(maxDepth, depth);
    if (active.has(index)) { cycleCount += 1; return; }
    if (reachable.has(index)) return;
    active.add(index);
    reachable.add(index);
    const record = records[index];
    if (record.mode === "LEAF") leafIndices.add(index);
    else {
      visit(keyToIndex.get(`${record.treeId}:${record.falseNodeId}`), depth + 1);
      visit(keyToIndex.get(`${record.treeId}:${record.trueNodeId}`), depth + 1);
    }
    active.delete(index);
  }
  if (!invalidChildReferenceCount) for (const root of roots) visit(root, 1);
  if (cycleCount) failures.push("tree_ensemble_cycle_detected");
  const orphanNodeCount = Math.max(0, nodeCount - reachable.size);
  if (orphanNodeCount) failures.push("tree_ensemble_orphan_node");
  const multipleParentNodeCount = [...incoming.values()].filter((count) => count > 1).length;
  if (multipleParentNodeCount) reasons.push("tree_ensemble_shared_subtree_not_strict_tree");

  const weightsByLeaf = new Map();
  let invalidWeightReferenceCount = 0;
  let nonLeafWeightCount = 0;
  let invalidWeightIdCount = 0;
  let unusedWeightCount = 0;
  for (let index = 0; index < weights.count; index += 1) {
    const key = `${weights.treeIds[index]}:${weights.nodeIds[index]}`;
    const nodeIndex = keyToIndex.get(key);
    const targetCount = classifier ? parsed.labels.values.length : parsed.nTargets;
    if (!Number.isSafeInteger(weights.ids[index]) || weights.ids[index] < 0 || weights.ids[index] >= targetCount) invalidWeightIdCount += 1;
    if (nodeIndex == null) { invalidWeightReferenceCount += 1; unusedWeightCount += 1; continue; }
    if (records[nodeIndex].mode !== "LEAF") { nonLeafWeightCount += 1; unusedWeightCount += 1; continue; }
    if (!reachable.has(nodeIndex)) { unusedWeightCount += 1; continue; }
    const entries = weightsByLeaf.get(nodeIndex) || [];
    entries.push({ index, id: weights.ids[index], value: weights.values.values[index] });
    weightsByLeaf.set(nodeIndex, entries);
  }
  if (invalidWeightReferenceCount) failures.push("tree_ensemble_weight_references_missing_node");
  if (invalidWeightIdCount) failures.push("tree_ensemble_weight_target_or_class_id_out_of_bounds");
  if (nonLeafWeightCount) reasons.push("tree_ensemble_nonleaf_weights_ignored_by_pinned_ort");
  let singleTargetIgnoredWeightCount = 0;
  const targetCount = classifier ? parsed.labels.values.length : parsed.nTargets;
  if (targetCount === 1) {
    for (const entries of weightsByLeaf.values()) singleTargetIgnoredWeightCount += Math.max(0, entries.length - 1);
    if (singleTargetIgnoredWeightCount) reasons.push("tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort");
  }
  const structurallyUsable = [...weightsByLeaf.values()].reduce((sum, entries) => sum + entries.length, 0);
  const usedWeightCount = structurallyUsable - singleTargetIgnoredWeightCount;
  unusedWeightCount += singleTargetIgnoredWeightCount;
  return {
    records, roots, keyToIndex, weightsByLeaf,
    treeCount: roots.length, nodeCount, branchNodeCount: records.filter((record) => record.mode !== "LEAF").length,
    leafCount: records.filter((record) => record.mode === "LEAF").length,
    reachableNodeCount: reachable.size, reachableLeafCount: leafIndices.size, orphanNodeCount,
    maxDepth, cycleCount, duplicateNodeCount, invalidChildReferenceCount, invalidFeatureCount,
    rootMismatchCount, multipleParentNodeCount, invalidWeightReferenceCount, invalidWeightIdCount,
    nonLeafWeightCount, singleTargetIgnoredWeightCount, usedWeightCount, unusedWeightCount,
    membershipNodeCount: 0, membershipSetCount: 0, membershipValueCount: 0,
    membershipDuplicateValueCount: 0, membershipSeparatorCount: 0,
  };
}

function validateGenericGraph(parsed, featureCount, failures, reasons) {
  const { nodes } = parsed;
  const nodeCount = nodes.splits.values.length;
  const lengths = [nodes.featureIds.length, nodes.splits.values.length, nodes.modes.values.length,
    nodes.trueNodeIds.length, nodes.falseNodeIds.length, nodes.trueLeafs.length, nodes.falseLeafs.length];
  if (!nodeCount || lengths.some((length) => length !== nodeCount)) failures.push("tree_ensemble_v5_node_tuple_cardinality_mismatch");
  if (nodes.missingTracksTrue.length && nodes.missingTracksTrue.length !== nodeCount) failures.push("tree_ensemble_v5_missing_track_cardinality_mismatch");
  if (nodes.hitrates.present && nodes.hitrates.values.length !== nodeCount) failures.push("tree_ensemble_v5_hitrate_cardinality_mismatch");
  if (nodes.modes.values.some((mode) => !Number.isSafeInteger(mode) || mode < 0 || mode > 6)) failures.push("tree_ensemble_v5_invalid_node_mode");
  if ([...nodes.trueLeafs, ...nodes.falseLeafs, ...nodes.missingTracksTrue].some((value) => ![0, 1].includes(value))) {
    failures.push("tree_ensemble_v5_branch_flag_not_boolean");
  }
  if (parsed.leafTargetIds.length !== parsed.leafWeights.values.length) failures.push("tree_ensemble_v5_leaf_tuple_cardinality_mismatch");
  if (parsed.leafTargetIds.some((value) => !Number.isSafeInteger(value) || value < 0 || value >= parsed.nTargets)) {
    failures.push("tree_ensemble_v5_leaf_target_id_out_of_bounds");
  }
  let invalidFeatureCount = 0;
  for (const featureId of nodes.featureIds) {
    if (!Number.isSafeInteger(featureId) || featureId < 0 || featureCount != null && featureId >= featureCount) invalidFeatureCount += 1;
  }
  if (invalidFeatureCount) failures.push("tree_ensemble_v5_feature_id_out_of_bounds");

  const leafCount = parsed.leafWeights.values.length;
  let invalidChildReferenceCount = 0;
  let invalidRootCount = 0;
  for (let index = 0; index < nodeCount; index += 1) {
    for (const [child, isLeaf] of [[nodes.trueNodeIds[index], nodes.trueLeafs[index]], [nodes.falseNodeIds[index], nodes.falseLeafs[index]]]) {
      const limit = isLeaf ? leafCount : nodeCount;
      if (!Number.isSafeInteger(child) || child < 0 || child >= limit) invalidChildReferenceCount += 1;
    }
  }
  for (const root of parsed.treeRoots) if (!Number.isSafeInteger(root) || root < 0 || root >= nodeCount) invalidRootCount += 1;
  if (!parsed.treeRoots.length || invalidRootCount) failures.push("tree_ensemble_v5_invalid_tree_root");
  if (invalidChildReferenceCount) failures.push("tree_ensemble_v5_invalid_child_reference");

  const reachableNodes = new Set();
  const reachableLeaves = new Set();
  const active = new Set();
  const incoming = new Array(nodeCount).fill(0);
  let cycleCount = 0;
  let maxDepth = 0;
  function visitNode(index, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (active.has(index)) { cycleCount += 1; return; }
    if (reachableNodes.has(index)) return;
    active.add(index);
    reachableNodes.add(index);
    const degenerateLeaf = nodes.trueLeafs[index] === 1 && nodes.falseLeafs[index] === 1
      && nodes.trueNodeIds[index] === nodes.falseNodeIds[index];
    if (degenerateLeaf) reachableLeaves.add(nodes.trueNodeIds[index]);
    else {
      for (const [child, isLeaf] of [[nodes.falseNodeIds[index], nodes.falseLeafs[index]], [nodes.trueNodeIds[index], nodes.trueLeafs[index]]]) {
        if (isLeaf) reachableLeaves.add(child);
        else { incoming[child] += 1; visitNode(child, depth + 1); }
      }
    }
    active.delete(index);
  }
  if (!invalidRootCount && !invalidChildReferenceCount) for (const root of parsed.treeRoots) visitNode(root, 1);
  if (cycleCount) failures.push("tree_ensemble_v5_cycle_detected");
  const orphanNodeCount = Math.max(0, nodeCount - reachableNodes.size);
  const orphanLeafCount = Math.max(0, leafCount - reachableLeaves.size);
  if (orphanNodeCount || orphanLeafCount) reasons.push("tree_ensemble_v5_unreachable_serialized_nodes_or_leaves");
  const multipleParentNodeCount = incoming.filter((count) => count > 1).length;
  if (multipleParentNodeCount) reasons.push("tree_ensemble_v5_shared_subtree_not_strict_tree");
  const duplicateRootCount = parsed.treeRoots.length - new Set(parsed.treeRoots).size;
  if (duplicateRootCount) reasons.push("tree_ensemble_v5_duplicate_root_repeats_tree_contribution");
  return {
    treeCount: parsed.treeRoots.length, nodeCount, branchNodeCount: nodeCount,
    leafCount, reachableNodeCount: reachableNodes.size, reachableLeafCount: reachableLeaves.size,
    orphanNodeCount: orphanNodeCount + orphanLeafCount, maxDepth, cycleCount,
    duplicateNodeCount: 0, invalidChildReferenceCount, invalidFeatureCount, rootMismatchCount: invalidRootCount,
    multipleParentNodeCount, invalidWeightReferenceCount: 0, invalidWeightIdCount: 0,
    nonLeafWeightCount: 0, singleTargetIgnoredWeightCount: 0,
    usedWeightCount: reachableLeaves.size, unusedWeightCount: orphanLeafCount,
    membershipNodeCount: parsed.membership.memberNodeCount, membershipSetCount: parsed.membership.sets.length,
    membershipValueCount: parsed.membership.valueCount, membershipDuplicateValueCount: parsed.membership.duplicateCount,
    membershipSeparatorCount: parsed.membership.separatorCount,
    reachableNodes, reachableLeaves,
  };
}

function evaluateLegacyReference({ input, parsed, graph, classifier, targetCount }) {
  const values = staticInputValues(input);
  if (!values) return unresolvedReference("not_assessed_dynamic_input");
  const rows = inputRows(input, values);
  if (!rows) return unresolvedReference("not_assessed_input_shape_or_payload_mismatch");
  const estimated = safeProduct(rows.length, safeProduct(Math.max(1, graph.treeCount), Math.max(1, graph.maxDepth)));
  if (estimated == null || estimated > MAX_REFERENCE_WORK) return unresolvedReference("not_assessed_work_limit");
  const precision = input.dtype === "FLOAT64" ? "f64" : "f32";
  let pathSteps = 0;
  let nonFiniteScores = 0;
  let decisionBoundaryCount = 0;
  let unwrittenScoreCount = 0;
  const rawScores = [];
  const outputScores = [];
  const labels = [];
  for (const row of rows) {
    const leafs = [];
    for (const root of graph.roots) {
      const result = traverseLegacy(row, root, graph, input.dtype, precision);
      if (!result.ok) return unresolvedReference(result.reason);
      leafs.push(result.leaf);
      pathSteps += result.steps;
      if (pathSteps > MAX_REFERENCE_WORK) return unresolvedReference("not_assessed_work_limit");
    }
    if (classifier) {
      const finalized = classifierScores(leafs, graph.weightsByLeaf, parsed.baseValues.values,
        parsed.postTransform, parsed.labels.values, parsed.labels.kind, precision);
      if (!finalized.ok) return unresolvedReference(finalized.reason);
      rawScores.push(...finalized.raw);
      outputScores.push(...finalized.output);
      labels.push(finalized.label);
      decisionBoundaryCount += finalized.boundary ? 1 : 0;
      unwrittenScoreCount += finalized.unwritten;
    } else {
      const raw = regressorScores(leafs, graph.weightsByLeaf, parsed.baseValues.values,
        parsed.aggregateFunction, targetCount, precision);
      const output = applyRegressorTransform(raw, parsed.postTransform, precision);
      rawScores.push(...raw);
      outputScores.push(...output);
      decisionBoundaryCount += raw.filter((value) => value === 0).length;
    }
  }
  nonFiniteScores = outputScores.filter((value) => value != null && !Number.isFinite(value)).length;
  return {
    status: "assessed_scalar_reference", inputValueCount: values.length, rowCount: rows.length,
    pathStepCount: pathSteps, rawScoreCount: rawScores.length, outputScoreCount: outputScores.length,
    nonFiniteScoreCount: nonFiniteScores, decisionBoundaryCount, unwrittenScoreCount,
    rawScorePreview: rawScores.slice(0, PREVIEW_LIMIT).map(valueText),
    outputScorePreview: outputScores.slice(0, PREVIEW_LIMIT).map(valueText),
    labelPreview: labels.slice(0, PREVIEW_LIMIT).map(valueText),
  };
}

function evaluateGenericReference({ input, parsed, graph }) {
  const values = staticInputValues(input);
  if (!values) return unresolvedReference("not_assessed_dynamic_input");
  const rows = inputRows(input, values);
  if (!rows) return unresolvedReference("not_assessed_input_shape_or_payload_mismatch");
  const estimated = safeProduct(rows.length, safeProduct(Math.max(1, parsed.treeRoots.length), Math.max(1, graph.maxDepth)));
  if (estimated == null || estimated > MAX_REFERENCE_WORK) return unresolvedReference("not_assessed_work_limit");
  const precision = input.dtype === "FLOAT64" ? "f64" : "f32";
  let pathSteps = 0;
  const rawScores = [];
  const outputScores = [];
  for (const row of rows) {
    const leafs = [];
    for (const root of parsed.treeRoots) {
      const result = traverseGeneric(row, root, parsed, input.dtype);
      if (!result.ok) return unresolvedReference(result.reason);
      leafs.push(result.leaf);
      pathSteps += result.steps;
      if (pathSteps > MAX_REFERENCE_WORK) return unresolvedReference("not_assessed_work_limit");
    }
    const raw = genericScores(leafs, parsed, precision);
    const output = applyRegressorTransform(raw, parsed.postTransform, precision);
    rawScores.push(...raw);
    outputScores.push(...output);
  }
  return {
    status: "assessed_scalar_reference", inputValueCount: values.length, rowCount: rows.length,
    pathStepCount: pathSteps, rawScoreCount: rawScores.length, outputScoreCount: outputScores.length,
    nonFiniteScoreCount: outputScores.filter((value) => !Number.isFinite(value)).length,
    decisionBoundaryCount: rawScores.filter((value) => value === 0).length, unwrittenScoreCount: 0,
    rawScorePreview: rawScores.slice(0, PREVIEW_LIMIT).map(valueText),
    outputScorePreview: outputScores.slice(0, PREVIEW_LIMIT).map(valueText), labelPreview: [],
  };
}

function traverseLegacy(row, root, graph, dtype, precision) {
  let index = root;
  let steps = 0;
  const seen = new Set();
  while (true) {
    if (seen.has(index)) return { ok: false, reason: "not_assessed_runtime_cycle" };
    seen.add(index);
    const node = graph.records[index];
    if (!node) return { ok: false, reason: "not_assessed_missing_node" };
    steps += 1;
    if (node.mode === "LEAF") return { ok: true, leaf: index, steps };
    const feature = runtimeFeature(row[node.featureId], dtype, precision);
    const result = branchResult(feature, node.value, node.mode, node.missingTrue);
    const childId = result ? node.trueNodeId : node.falseNodeId;
    index = graph.keyToIndex.get(`${node.treeId}:${childId}`);
    if (index == null) return { ok: false, reason: "not_assessed_missing_child" };
  }
}

function traverseGeneric(row, root, parsed, dtype) {
  const { nodes, membership } = parsed;
  let index = root;
  let steps = 0;
  const seen = new Set();
  const degenerate = nodes.trueLeafs[index] === 1 && nodes.falseLeafs[index] === 1
    && nodes.trueNodeIds[index] === nodes.falseNodeIds[index];
  if (degenerate) return { ok: true, leaf: nodes.trueNodeIds[index], steps: 1 };
  while (true) {
    if (seen.has(index)) return { ok: false, reason: "not_assessed_runtime_cycle" };
    seen.add(index);
    steps += 1;
    const value = runtimeFeature(row[nodes.featureIds[index]], dtype, dtype === "FLOAT64" ? "f64" : "f32");
    const mode = nodes.modes.values[index];
    const missingTrue = nodes.missingTracksTrue[index] === 1;
    const result = genericBranchResult(value, nodes.splits.values[index], mode, missingTrue, membership.byNode[index]);
    const child = result ? nodes.trueNodeIds[index] : nodes.falseNodeIds[index];
    const isLeaf = result ? nodes.trueLeafs[index] : nodes.falseLeafs[index];
    if (isLeaf) return { ok: true, leaf: child, steps };
    index = child;
  }
}

function regressorScores(leafs, weightsByLeaf, baseValues, aggregate, targetCount, precision) {
  const scores = new Array(targetCount).fill(0);
  const has = new Array(targetCount).fill(false);
  for (const leaf of leafs) {
    const entries = weightsByLeaf.get(leaf) || [];
    const selected = targetCount === 1 ? entries.slice(0, 1) : entries;
    for (const entry of selected) {
      const id = entry.id;
      if (aggregate === "SUM" || aggregate === "AVERAGE") scores[id] = add(scores[id], entry.value, precision);
      else if (aggregate === "MIN") scores[id] = !has[id] || entry.value < scores[id] ? entry.value : scores[id];
      else if (aggregate === "MAX") scores[id] = !has[id] || entry.value > scores[id] ? entry.value : scores[id];
      has[id] = true;
    }
  }
  for (let index = 0; index < targetCount; index += 1) {
    if (aggregate === "AVERAGE") scores[index] = divide(scores[index], leafs.length, precision);
    if (baseValues.length === targetCount) scores[index] = add(scores[index], baseValues[index], precision);
    else if (!has[index] && !["SUM", "AVERAGE"].includes(aggregate)) scores[index] = 0;
  }
  return scores;
}

function genericScores(leafs, parsed, precision) {
  const scores = new Array(parsed.nTargets).fill(0);
  const has = new Array(parsed.nTargets).fill(false);
  for (const leaf of leafs) {
    const id = parsed.leafTargetIds[leaf];
    const value = parsed.leafWeights.values[leaf];
    if (parsed.aggregateFunction === "SUM" || parsed.aggregateFunction === "AVERAGE") scores[id] = add(scores[id], value, precision);
    else if (parsed.aggregateFunction === "MIN") scores[id] = !has[id] || value < scores[id] ? value : scores[id];
    else if (parsed.aggregateFunction === "MAX") scores[id] = !has[id] || value > scores[id] ? value : scores[id];
    has[id] = true;
  }
  for (let index = 0; index < scores.length; index += 1) {
    if (parsed.aggregateFunction === "AVERAGE") scores[index] = divide(scores[index], leafs.length, precision);
    if (!has[index] && ["MIN", "MAX"].includes(parsed.aggregateFunction)) scores[index] = 0;
  }
  return scores;
}

function classifierScores(leafs, weightsByLeaf, baseValues, postTransform, labels, labelKind, precision) {
  const count = labels.length;
  if (count < 2) return { ok: false, reason: "not_assessed_classifier_less_than_two_classes" };
  const scores = new Array(count).fill(0);
  const has = new Array(count).fill(false);
  for (const leaf of leafs) {
    for (const entry of weightsByLeaf.get(leaf) || []) {
      scores[entry.id] = add(scores[entry.id], entry.value, precision);
      has[entry.id] = true;
    }
  }
  if (count > 2) {
    for (let index = 0; index < baseValues.length; index += 1) {
      scores[index] = has[index] ? add(scores[index], baseValues[index], precision) : baseValues[index];
      has[index] = true;
    }
    let maximum = -1;
    for (let index = 0; index < count; index += 1) if (has[index] && (maximum < 0 || scores[index] > scores[maximum])) maximum = index;
    if (maximum < 0) return { ok: false, reason: "not_assessed_classifier_has_no_score" };
    return {
      ok: true, raw: [...scores], output: applyPostTransform(scores, postTransform, precision),
      label: labels[maximum], boundary: scores.filter((value, index) => has[index] && value === scores[maximum]).length > 1,
      unwritten: 0,
    };
  }

  let working = scores.map((score, index) => ({ score, has: has[index] }));
  let expansion = -1;
  if (baseValues.length === 2) {
    if (working[1].has) {
      working[1].score = add(baseValues[1], working[0].score, precision);
      working[0].score = cast(-working[1].score, precision);
      working[1].has = true;
    } else {
      working[1].score = add(working[1].score, baseValues[1], precision);
      working[0].score = add(working[0].score, baseValues[0], precision);
    }
  } else if (baseValues.length === 1) {
    working[0].score = add(working[0].score, baseValues[0], precision);
    if (!working[1].has) working = working.slice(0, 1);
  } else if (baseValues.length === 0) {
    expansion = 3;
    if (!working[1].has) working = working.slice(0, 1);
  }
  const represented = new Set();
  for (const entries of weightsByLeaf.values()) for (const entry of entries) represented.add(entry.id);
  const binaryCase = represented.size === 1;
  const positive = working.length === 2 && working[1].has ? working[1].score : working[0].has ? working[0].score : 0;
  let labelIndex;
  if (binaryCase) {
    expansion = positive > 0 ? 2 : 3;
    labelIndex = positive > 0 ? 1 : 0;
  } else {
    labelIndex = positive > 0 ? 1 : 0;
  }
  const raw = working.map((entry) => entry.score);
  const transformed = writeBinaryScores(raw, postTransform, expansion, precision);
  return {
    ok: true, raw, output: transformed.values,
    label: !binaryCase && labelKind === "int64" ? BigInt(labelIndex) : labels[labelIndex],
    boundary: positive === 0, unwritten: transformed.unwritten,
  };
}

function applyRegressorTransform(values, transform, precision) {
  if (values.length === 1) return transform === "PROBIT" ? [cast(probit(values[0]), precision)] : values.map((value) => cast(value, precision));
  return applyPostTransform(values, transform, precision);
}

function applyPostTransform(values, transform, precision) {
  if (transform === "LOGISTIC") return values.map((value) => cast(logistic(value), precision));
  if (transform === "PROBIT") return values.map((value) => cast(probit(value), precision));
  if (transform === "SOFTMAX" || transform === "SOFTMAX_ZERO") return softmax(values, transform === "SOFTMAX_ZERO").map((value) => cast(value, precision));
  return values.map((value) => cast(value, precision));
}

function writeBinaryScores(values, transform, expansion, precision) {
  if (values.length >= 2) return { values: applyPostTransform(values, transform, precision), unwritten: 0 };
  const score = values[0];
  if (transform === "PROBIT") return { values: [cast(probit(score), precision), null], unwritten: 1 };
  if ([0, 1].includes(expansion)) return { values: [cast(1 - score, precision), cast(score, precision)], unwritten: 0 };
  if ([2, 3].includes(expansion)) {
    if (transform === "LOGISTIC") return { values: [cast(logistic(-score), precision), cast(logistic(score), precision)], unwritten: 0 };
    return { values: [cast(-score, precision), cast(score, precision)], unwritten: 0 };
  }
  return { values: [cast(score, precision), null], unwritten: 1 };
}

function branchResult(value, threshold, mode, missingTrue) {
  if (Number.isNaN(value)) return missingTrue;
  if (mode === "BRANCH_LEQ") return value <= threshold;
  if (mode === "BRANCH_LT") return value < threshold;
  if (mode === "BRANCH_GTE") return value >= threshold;
  if (mode === "BRANCH_GT") return value > threshold;
  if (mode === "BRANCH_EQ") return value === threshold;
  if (mode === "BRANCH_NEQ") return value !== threshold;
  return false;
}

function genericBranchResult(value, threshold, mode, missingTrue, members) {
  if (Number.isNaN(value)) return missingTrue;
  if (mode === 0) return value <= threshold;
  if (mode === 1) return value < threshold;
  if (mode === 2) return value >= threshold;
  if (mode === 3) return value > threshold;
  if (mode === 4) return value === threshold;
  if (mode === 5) return value !== threshold;
  if (mode === 6) return (members || []).some((member) => Object.is(member, value) || member === value);
  return false;
}

function legacyRisks({ parsed, graph, classifier, schemaVersion, cpuDtypeGap, reference, targetCount }) {
  const risks = [];
  if (schemaVersion === 5) risks.push("tree_ensemble_legacy_operator_deprecated_at_opset_5");
  if (cpuDtypeGap) risks.push("tree_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  if (classifier && parsed.labels.duplicateCount) risks.push("tree_classifier_duplicate_labels_ambiguous_output_semantics");
  if (graph.nonLeafWeightCount) risks.push("tree_ensemble_nonleaf_weights_ignored_by_pinned_ort");
  if (graph.singleTargetIgnoredWeightCount) risks.push("tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort");
  if (graph.multipleParentNodeCount) risks.push("tree_ensemble_shared_subtree_not_strict_tree");
  if (parsed.nonFiniteParameterCount || Number(reference.nonFiniteScoreCount || 0)) risks.push("tree_ensemble_non_finite_parameter_or_reference_score");
  if (Number(reference.decisionBoundaryCount || 0)) risks.push("tree_ensemble_reference_decision_boundary");
  if (Number(reference.unwrittenScoreCount || 0)) risks.push("tree_classifier_binary_post_transform_leaves_score_unwritten");
  if (!classifier && targetCount === 1 && !["NONE", "PROBIT"].includes(parsed.postTransform)) risks.push("tree_regressor_single_target_post_transform_noop");
  if (classifier && targetCount === 2 && parsed.baseValues.values.length === 1) risks.push("tree_classifier_binary_single_base_value_semantics_underspecified");
  if (classifier && targetCount === 2) {
    const represented = new Set(parsed.weights.ids);
    const labelsCanonical = parsed.labels.kind !== "int64" || parsed.labels.values.length === 2
      && parsed.labels.values[0] === 0n && parsed.labels.values[1] === 1n;
    if (represented.size > 1 && !labelsCanonical) risks.push("tree_classifier_pinned_ort_binary_label_index_semantics");
  }
  return risks;
}

function genericRisks({ parsed, graph, cpuDtypeGap, reference }) {
  const risks = [];
  if (cpuDtypeGap) risks.push("tree_ensemble_v5_float16_missing_pinned_ort_cpu_kernel");
  if (graph.orphanNodeCount) risks.push("tree_ensemble_v5_unreachable_serialized_nodes_or_leaves");
  if (graph.multipleParentNodeCount) risks.push("tree_ensemble_v5_shared_subtree_not_strict_tree");
  if (parsed.membership.duplicateCount) risks.push("tree_ensemble_v5_duplicate_membership_values");
  if (parsed.membership.zeroValueCount) risks.push("tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser");
  if (parsed.nonFiniteParameterCount || Number(reference.nonFiniteScoreCount || 0)) risks.push("tree_ensemble_non_finite_parameter_or_reference_score");
  if (Number(reference.decisionBoundaryCount || 0)) risks.push("tree_ensemble_reference_decision_boundary");
  if (parsed.nTargets === 1 && !["NONE", "PROBIT"].includes(parsed.postTransform)) risks.push("tree_ensemble_v5_single_target_post_transform_noop");
  return risks;
}

function validateLegacyBaseValues(baseValues, targetCount, classifier, failures, reasons) {
  const count = baseValues.values.length;
  if (!count) return;
  if (count === targetCount) return;
  if (classifier && targetCount === 2 && count === 1) {
    reasons.push("tree_classifier_binary_single_base_value_semantics_underspecified");
    return;
  }
  failures.push("tree_ensemble_base_value_cardinality_mismatch");
}

function legacyOutputShapes(input, targetCount, classifier, schemaVersion) {
  const rankSupported = input.rank === 2 || schemaVersion === 1 && input.rank === 1;
  if (!rankSupported || targetCount < 1) return { declared: false, label: [], value: [] };
  const batch = input.rank === 1 ? 1 : input.batchCount;
  return { declared: true, label: classifier ? [batch] : [], value: [batch, targetCount] };
}

function treeBaseRow({ scope, nodeIndex, importedOpset, schemaVersion, opName, contractKind, status, input,
  outputName, outputDtype, outputShape, outputShapeDeclared, failures, reasons }) {
  return {
    scope, node_index: nodeIndex, op_name: opName, contract_kind: contractKind,
    imported_opset: importedOpset, resolved_schema_version: schemaVersion, status,
    input_name: input.input?.name || "", output_name: outputName,
    input_dtype: input.dtype, input_kind: input.type?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: input.rank, input_shape: input.shape,
    exact_batch_count: input.batchCount, exact_feature_count: input.featureCount,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: canonicalOnnxTypeProto(makeOnnxTensorType(outputDtype, outputShape, outputShapeDeclared)),
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: outputShapeDeclared ? outputShape.length : null, exact_output_shape: outputShape,
    exact_dense_output_element_count: outputShapeDeclared && outputShape.every(knownDimension) ? safeShapeElementCount(outputShape) : null,
    output_shape_basis: "pinned_onnx_tree_ensemble_schema_and_pinned_ort_cpu_runtime_contract",
    runtime_reference_status: "pinned_ort_cpu_tree_ensemble_kernel_common_attribute_aggregator",
    attribute_mode: "tree_topology_leaf_weight_aggregate_post_transform_contract",
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null,
    exact_truncated_feature_count_per_batch: null, padded_input_count: 0, truncated_input_count: 0,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_values: [], exact_index_preview: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: [],
  };
}

function graphRow(graph) {
  return {
    tree_exact_tree_count: graph.treeCount,
    tree_exact_root_count: graph.treeCount,
    tree_exact_node_count: graph.nodeCount,
    tree_exact_branch_node_count: graph.branchNodeCount,
    tree_exact_leaf_count: graph.leafCount,
    tree_reachable_node_count: graph.reachableNodeCount,
    tree_reachable_leaf_count: graph.reachableLeafCount,
    tree_orphan_node_or_leaf_count: graph.orphanNodeCount,
    tree_max_depth: graph.maxDepth,
    tree_cycle_count: graph.cycleCount,
    tree_duplicate_node_identity_count: graph.duplicateNodeCount,
    tree_invalid_child_reference_count: graph.invalidChildReferenceCount,
    tree_invalid_feature_id_count: graph.invalidFeatureCount,
    tree_root_mismatch_count: graph.rootMismatchCount,
    tree_multiple_parent_node_count: graph.multipleParentNodeCount,
    tree_invalid_weight_reference_count: graph.invalidWeightReferenceCount,
    tree_invalid_weight_id_count: graph.invalidWeightIdCount,
    tree_single_target_ignored_weight_count: graph.singleTargetIgnoredWeightCount,
    tree_membership_node_count: graph.membershipNodeCount,
    tree_membership_set_count: graph.membershipSetCount,
    tree_membership_value_count: graph.membershipValueCount,
    tree_membership_duplicate_value_count: graph.membershipDuplicateValueCount,
    tree_membership_separator_count: graph.membershipSeparatorCount,
  };
}

function referenceRow(reference) {
  return {
    tree_reference_assessment_status: reference.status,
    tree_reference_input_value_count: reference.inputValueCount,
    tree_reference_row_count: reference.rowCount,
    tree_reference_path_step_count: reference.pathStepCount,
    tree_reference_raw_score_count: reference.rawScoreCount,
    tree_reference_output_score_count: reference.outputScoreCount,
    tree_reference_non_finite_score_count: reference.nonFiniteScoreCount,
    tree_reference_decision_boundary_count: reference.decisionBoundaryCount,
    tree_reference_unwritten_score_count: reference.unwrittenScoreCount,
    tree_reference_raw_score_preview: reference.rawScorePreview,
    tree_reference_output_score_preview: reference.outputScorePreview,
    tree_reference_label_preview: reference.labelPreview,
    tree_reference_boundary: "Deterministic scalar source-order reference only. The executed ORT thread partition, floating reduction order, platform libm path, optimized graph, and selected execution provider are not observed, so score previews are not runtime-bit-exact evidence and are not propagated as static tensors.",
  };
}

function numericChoice(node, listName, tensorName, schemaVersion, failures) {
  const listPresent = node.attributes?.has(listName);
  const tensorPresent = node.attributes?.has(tensorName);
  if (listPresent && tensorPresent) failures.push(`tree_ensemble_mutually_exclusive_attributes:${listName}:${tensorName}`);
  if (tensorPresent && schemaVersion < 3) failures.push(`tree_ensemble_tensor_attribute_not_defined_at_schema:${tensorName}`);
  if (tensorPresent) return tensorNumeric(node, tensorName, failures);
  const values = floatList(node, listName, failures);
  return { present: listPresent, values, source: listPresent ? listName : "absent", dtype: "FLOAT32", shape: [values.length] };
}

function tensorNumeric(node, name, failures, optional = false) {
  const attribute = node.attributes?.get(name);
  if (!attribute) {
    if (!optional) failures.push(`tree_ensemble_required_tensor_attribute_missing:${name}`);
    return { present: false, values: [], source: "absent", dtype: "UNKNOWN", shape: [] };
  }
  if (attribute.type !== 4 || !attribute.tensor || !attribute.valueTypesPresent?.includes(4)) {
    failures.push(`tree_ensemble_tensor_attribute_invalid:${name}`);
    return { present: true, values: [], source: name, dtype: "UNKNOWN", shape: [] };
  }
  const tensor = attribute.tensor;
  const shape = tensor.shapeDeclared === true ? [...(tensor.shape || [])] : [];
  if (shape.length !== 1 || !knownDimension(shape[0])) failures.push(`tree_ensemble_tensor_attribute_not_1d:${name}`);
  const values = tensorValues(tensor);
  if (values == null || shape.length === 1 && knownDimension(shape[0]) && values.length !== shape[0]) {
    failures.push(`tree_ensemble_tensor_attribute_payload_incomplete:${name}`);
  }
  return { present: true, values: values || [], source: name, dtype: tensor.dtype || "UNKNOWN", shape };
}

function tensorInteger(node, name, failures) {
  const result = tensorNumeric(node, name, failures);
  const values = result.values.map((value) => Number(value));
  if (values.some((value) => !Number.isSafeInteger(value))) failures.push(`tree_ensemble_tensor_attribute_not_integer:${name}`);
  return { ...result, values };
}

function tensorValues(tensor) {
  if (tensor.staticValuesCanonicalTextComplete === true && Array.isArray(tensor.staticValuesCanonicalTexts)) {
    return tensor.staticValuesCanonicalTexts.map(parseNumber);
  }
  if (tensor.staticValuesComplete === true && Array.isArray(tensor.staticValues)) return tensor.staticValues.map((value) => Number(value));
  if (tensor.initializerIntegerValuesExactComplete === true && Array.isArray(tensor.initializerIntegerValuesExactDecimals)) {
    try { return tensor.initializerIntegerValuesExactDecimals.map((value) => Number(BigInt(value))); } catch { return null; }
  }
  return null;
}

function splitMembershipValues(values, modes, failures) {
  const memberNodeCount = modes.filter((mode) => mode === 6).length;
  const sets = [];
  let current = [];
  let separatorCount = 0;
  let valueCount = 0;
  let zeroValueCount = 0;
  for (const value of values) {
    if (Number.isNaN(value)) {
      sets.push(current);
      current = [];
      separatorCount += 1;
    } else {
      current.push(value);
      valueCount += 1;
      if (value === 0) zeroValueCount += 1;
    }
  }
  if (current.length) failures.push("tree_ensemble_v5_membership_values_missing_nan_terminator");
  if (separatorCount !== memberNodeCount || sets.length !== memberNodeCount) failures.push("tree_ensemble_v5_membership_set_cardinality_mismatch");
  const duplicateCount = sets.reduce((sum, set) => sum + set.length - new Set(set.map(valueText)).size, 0);
  let setIndex = 0;
  const byNode = modes.map((mode) => mode === 6 ? sets[setIndex++] || [] : null);
  return {
    sets, memberNodeCount, separatorCount, valueCount, duplicateCount, zeroValueCount,
    byNode,
  };
}

function classifierLabels(node, failures) {
  const intsPresent = node.attributes?.has("classlabels_int64s");
  const stringsPresent = node.attributes?.has("classlabels_strings");
  const ints = bigIntList(node, "classlabels_int64s", failures);
  const strings = stringList(node, "classlabels_strings", failures);
  if (Number(intsPresent) + Number(stringsPresent) !== 1) failures.push("tree_classifier_exactly_one_label_attribute_required");
  const kind = stringsPresent ? "string" : "int64";
  const values = stringsPresent ? strings : ints;
  return { kind, values, duplicateCount: values.length - new Set(values.map(valueText)).size };
}

function intList(node, name, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return [];
  if (attribute.type !== 7 || !Array.isArray(attribute.intExactDecimals)
    || attribute.valueTypesPresent?.some((value) => value !== 7)) {
    failures.push(`tree_ensemble_integer_list_attribute_invalid:${name}`);
    return [];
  }
  try {
    return attribute.intExactDecimals.map((value) => {
      const integer = BigInt(value);
      if (integer < BigInt(Number.MIN_SAFE_INTEGER) || integer > BigInt(Number.MAX_SAFE_INTEGER)) {
        failures.push(`tree_ensemble_integer_attribute_out_of_safe_range:${name}`);
        return Number.NaN;
      }
      return Number(integer);
    });
  } catch {
    failures.push(`tree_ensemble_integer_list_attribute_invalid:${name}`);
    return [];
  }
}

function bigIntList(node, name, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return [];
  if (attribute.type !== 7 || !Array.isArray(attribute.intExactDecimals)
    || attribute.valueTypesPresent?.some((value) => value !== 7)) {
    failures.push(`tree_ensemble_integer_list_attribute_invalid:${name}`);
    return [];
  }
  try { return attribute.intExactDecimals.map((value) => BigInt(value)); }
  catch { failures.push(`tree_ensemble_integer_list_attribute_invalid:${name}`); return []; }
}

function floatList(node, name, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return [];
  if (attribute.type !== 6 || !Array.isArray(attribute.floats)
    || attribute.valueTypesPresent?.some((value) => value !== 6)) {
    failures.push(`tree_ensemble_float_list_attribute_invalid:${name}`);
    return [];
  }
  return attribute.floats.map((value) => Math.fround(value));
}

function stringList(node, name, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return [];
  if (attribute.type !== 8 || !Array.isArray(attribute.strings)
    || attribute.valueTypesPresent?.some((value) => value !== 8)) {
    failures.push(`tree_ensemble_string_list_attribute_invalid:${name}`);
    return [];
  }
  return [...attribute.strings];
}

function intScalar(node, name, fallback, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return fallback;
  if (attribute.type !== 2 || attribute.valueTypesPresent?.length !== 1 || attribute.valueTypesPresent[0] !== 2) {
    failures.push(`tree_ensemble_integer_attribute_invalid:${name}`);
    return null;
  }
  try { return BigInt(attribute.iExactDecimal || attribute.i); } catch { failures.push(`tree_ensemble_integer_attribute_invalid:${name}`); return null; }
}

function stringScalar(node, name, fallback, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return fallback;
  if (attribute.type !== 3 || typeof attribute.s !== "string"
    || attribute.valueTypesPresent?.length !== 1 || attribute.valueTypesPresent[0] !== 3) {
    failures.push(`tree_ensemble_string_attribute_invalid:${name}`);
    return "INVALID";
  }
  return attribute.s;
}

function staticInputValues(input) {
  const tensor = input.input;
  if (input.dtype === "INT64" && tensor?.initializerIntegerValuesExactComplete === true
    && Array.isArray(tensor.initializerIntegerValuesExactDecimals)) {
    try { return tensor.initializerIntegerValuesExactDecimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (["FLOAT16", "FLOAT32", "FLOAT64"].includes(input.dtype)
    && tensor?.staticValuesCanonicalTextComplete === true && Array.isArray(tensor.staticValuesCanonicalTexts)) {
    return tensor.staticValuesCanonicalTexts.map(parseNumber);
  }
  if (["FLOAT16", "FLOAT32", "FLOAT64", "INT32"].includes(input.dtype)
    && tensor?.staticValuesComplete === true && Array.isArray(tensor.staticValues)) return [...tensor.staticValues];
  return null;
}

function inputRows(input, values) {
  if (input.rank === 1 && knownDimension(input.featureCount) && values.length === input.featureCount) return [values];
  if (input.rank != null && input.rank >= 2 && knownDimension(input.batchCount) && knownDimension(input.featureCount)
    && values.length === input.batchCount * input.featureCount) {
    return Array.from({ length: input.batchCount }, (_, index) => values.slice(index * input.featureCount, (index + 1) * input.featureCount));
  }
  return null;
}

function runtimeFeature(value, dtype, precision) {
  if (dtype === "INT64" && typeof value === "bigint") return precision === "f64" ? Number(value) : bigintToFloat32(value);
  return precision === "f32" ? Math.fround(Number(value)) : Number(value);
}

function bigintToFloat32(value) {
  if (value === 0n) return 0;
  const negative = value < 0n;
  const magnitude = negative ? -value : value;
  let exponent = magnitude.toString(2).length - 1;
  if (exponent <= 23) return Math.fround(Number(value));
  const shift = BigInt(exponent - 23);
  let significand = magnitude >> shift;
  const remainder = magnitude - (significand << shift);
  const half = 1n << (shift - 1n);
  if (remainder > half || remainder === half && (significand & 1n) === 1n) significand += 1n;
  if (significand === (1n << 24n)) { significand >>= 1n; exponent += 1; }
  return Math.fround((negative ? -1 : 1) * Number(significand) * (2 ** (exponent - 23)));
}

function add(left, right, precision) {
  return precision === "f32" ? Math.fround(Math.fround(left) + Math.fround(right)) : left + right;
}

function divide(value, divisor, precision) {
  return precision === "f32" ? Math.fround(Math.fround(value) / divisor) : value / divisor;
}

function cast(value, precision) {
  return precision === "f32" ? Math.fround(value) : Number(value);
}

function logistic(value) {
  const projected = Math.fround(value);
  const result = Math.fround(1 / (1 + Math.exp(-Math.abs(projected))));
  return projected < 0 ? Math.fround(1 - result) : result;
}

function probit(value) {
  const projected = Math.fround(value);
  const centered = Math.fround(2 * projected - 1);
  const sign = centered < 0 ? -1 : 1;
  const x = Math.fround((1 - centered) * (1 + centered));
  const log = Math.fround(Math.log(x));
  const v = Math.fround(2 / (Math.PI * 0.147) + 0.5 * log);
  const v2 = Math.fround(log / 0.147);
  return Math.fround(Math.SQRT2 * sign * Math.sqrt(-v + Math.sqrt(v * v - v2)));
}

function softmax(values, zeroAware) {
  let maximum = -Number.MAX_VALUE;
  for (const value of values) if (Math.fround(value) > maximum) maximum = Math.fround(value);
  let sum = 0;
  const output = values.map((value) => {
    const projected = Math.fround(value);
    const item = zeroAware && (projected > -1e-7 && projected < 1e-7)
      ? projected * Math.exp(-maximum) : Math.exp(projected - maximum);
    sum = Math.fround(sum + Math.fround(item));
    return item;
  });
  return output.map((value) => Math.fround(value / sum));
}

function parseNumber(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

function unresolvedReference(status) {
  return {
    status, inputValueCount: null, rowCount: null, pathStepCount: null,
    rawScoreCount: null, outputScoreCount: null, nonFiniteScoreCount: null,
    decisionBoundaryCount: null, unwrittenScoreCount: null,
    rawScorePreview: [], outputScorePreview: [], labelPreview: [],
  };
}

function valueText(value) {
  if (value == null) return "UNWRITTEN";
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function knownDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function safeShapeElementCount(shape) {
  let product = 1;
  for (const dimension of shape) {
    product = safeProduct(product, dimension);
    if (product == null) return null;
  }
  return product;
}

function safeProduct(left, right) {
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0) return null;
  const value = left * right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}
