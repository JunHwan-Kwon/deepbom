export const GRAPH_DIFF_SNAPSHOT_SCHEMA = "deepbom.graph_diff_snapshot.v1";
export const GRAPH_DIFF_SCHEMA = "deepbom.artifact_graph_diff.v1";

export function buildGraphDiffSnapshot(analysis) {
  const graphRows = snapshotGraphRows(analysis);
  const { ops, tensors } = graphRows;
  const tensorByIndex = new Map(tensors.map((tensor) => [Number(tensor?.index), tensor]));
  const producer = new Map();
  const consumers = new Map();
  for (const op of ops) {
    for (const tensorIndex of integers(op.outputs)) if (!producer.has(tensorIndex)) producer.set(tensorIndex, Number(op.index));
    for (const tensorIndex of integers(op.inputs)) {
      if (!consumers.has(tensorIndex)) consumers.set(tensorIndex, []);
      consumers.get(tensorIndex).push(Number(op.index));
    }
  }
  const edges = [];
  for (const tensor of tensors) {
    const tensorIndex = Number(tensor?.index);
    if (!Number.isSafeInteger(tensorIndex)) continue;
    const from = producer.get(tensorIndex);
    for (const to of [...new Set(consumers.get(tensorIndex) || [])]) {
      if (from == null || to == null) continue;
      edges.push({ from, to, contract: tensorContract(tensor) });
    }
  }
  const incoming = adjacency(edges, "to", "from");
  const outgoing = adjacency(edges, "from", "to");
  const byIndex = new Map(ops.map((op) => [Number(op.index), op]));
  const nodes = ops.map((op) => {
    const index = Number(op.index);
    const node = {
      index,
      name: String(op.name || `OP_${index}`),
      domain: String(op.domain || graphRows.format || ""),
      source_identity: clean(op.graph_node_name || op.coreml_layer_name || op.source_layer_name),
      stage: clean(op.stage_key ?? op.stage_index),
      inputs: integers(op.inputs).map((tensorIndex) => tensorContract(tensorByIndex.get(tensorIndex))),
      outputs: integers(op.outputs).map((tensorIndex) => tensorContract(tensorByIndex.get(tensorIndex))),
      quantization: String(op.quantization_state || (op.quantized_compute_path ? "8bit_compute" : "none")),
      quant_risk: String(op.quant_risk || "none"),
      placement: placementContract(op, graphRows.format),
      macs: exactText(op.macs_decimal ?? op.macs),
    };
    node.contract_signature = contractSignature(node);
    node.topology_signature = topologySignature(node,
      (incoming.get(index) || []).map((neighbor) => byIndex.get(neighbor)),
      (outgoing.get(index) || []).map((neighbor) => byIndex.get(neighbor)));
    return node;
  });
  return {
    schema: GRAPH_DIFF_SNAPSHOT_SCHEMA,
    artifact_sha256: graphRows.artifactSha256,
    format: graphRows.format,
    node_count: nodes.length,
    edge_count: edges.length,
    nodes,
    edges,
    interpretation_boundary: "This compact snapshot stores operator and tensor contracts for local topology comparison. It contains no model payload bytes or tensor values.",
  };
}

function snapshotGraphRows(analysis) {
  const artifactIr = analysis?.artifact_ir;
  const format = String(artifactIr?.artifact?.format || analysis?.format || "").toLowerCase();
  const artifactSha256 = String(artifactIr?.artifact?.sha256 || analysis?.model_sha256 || "");
  if (artifactIr?.graph?.status !== "serialized") {
    return {
      ops: Array.isArray(analysis?.ops) ? analysis.ops : [],
      tensors: Array.isArray(analysis?.tensors) ? analysis.tensors : [],
      artifactSha256,
      format,
    };
  }
  const primaryScopeRef = artifactIr.graph.primary_scope_ref;
  const canonicalValues = artifactIr.graph.values.filter((row) => row.scope_ref === primaryScopeRef);
  const valueByRef = new Map(canonicalValues.map((row) => [row.id, row]));
  const nativeOps = indexedRows(analysis?.ops);
  const nativeTensors = indexedRows(analysis?.tensors);
  const tensors = canonicalValues.map((value) => ({
    ...(nativeTensors.get(value.native_index) || {}),
    index: value.native_index,
    name: value.name,
    dtype: value.dtype,
    shape: value.shape,
    artifact_ir_contract: value,
  }));
  const ops = artifactIr.graph.operators.filter((row) => row.scope_ref === primaryScopeRef).map((operator) => ({
    ...(nativeOps.get(operator.native_index) || {}),
    index: operator.native_index,
    name: operator.op_type,
    domain: operator.domain,
    inputs: operator.inputs.map((port) => valueByRef.get(port.value_ref)?.native_index).filter(Number.isSafeInteger),
    outputs: operator.outputs.map((port) => valueByRef.get(port.value_ref)?.native_index).filter(Number.isSafeInteger),
    macs: operator.metrics?.macs?.number,
    macs_decimal: operator.metrics?.macs?.decimal,
    quantization_state: operator.quantization_summary?.state,
    quant_risk: operator.quantization_summary?.risk,
    artifact_ir_contract: operator,
  }));
  return { ops, tensors, artifactSha256, format };
}

function indexedRows(rows) {
  return new Map((Array.isArray(rows) ? rows : []).map((row, position) => {
    const index = Number(row?.index);
    return [Number.isSafeInteger(index) && index >= 0 ? index : position, row];
  }));
}

export function compareGraphDiffSnapshots(left, right) {
  validateSnapshot(left, "left");
  validateSnapshot(right, "right");
  if (left.format !== right.format) throw new Error("Graph diff requires the same artifact format.");
  const matches = [];
  const unmatchedLeft = new Set(left.nodes.map((node) => node.index));
  const unmatchedRight = new Set(right.nodes.map((node) => node.index));
  matchUnique(left.nodes, right.nodes, "topology_signature", "exact_topology_and_tensor_contract", 1, matches, unmatchedLeft, unmatchedRight);
  matchUnique(
    left.nodes.filter((node) => node.source_identity),
    right.nodes.filter((node) => node.source_identity),
    "source_identity",
    "unique_explicit_source_identity",
    0.9,
    matches,
    unmatchedLeft,
    unmatchedRight,
  );
  matchUnique(left.nodes, right.nodes, "contract_signature", "unique_operator_tensor_contract", 0.8, matches, unmatchedLeft, unmatchedRight);
  const ambiguous = ambiguousCandidates(left.nodes, right.nodes, unmatchedLeft, unmatchedRight);
  const rows = matches.sort((a, b) => a.left.index - b.left.index).map((match) => ({
    ...match,
    changes: nodeChanges(match.left, match.right),
  }));
  return {
    schema: GRAPH_DIFF_SCHEMA,
    left_artifact_sha256: left.artifact_sha256,
    right_artifact_sha256: right.artifact_sha256,
    format: left.format,
    matched_count: rows.length,
    changed_match_count: rows.filter((row) => row.changes.length).length,
    unchanged_match_count: rows.filter((row) => !row.changes.length).length,
    unmatched_left_indices: [...unmatchedLeft].sort((a, b) => a - b),
    unmatched_right_indices: [...unmatchedRight].sort((a, b) => a - b),
    ambiguous,
    matches: rows,
    interpretation_boundary: "Nodes are matched only by unique topology plus tensor contracts, unique explicit source identity, or unique operator/tensor contracts. Name or operator index alone is never treated as identity; ambiguous candidates remain unresolved.",
  };
}

function matchUnique(leftNodes, rightNodes, keySpec, method, confidence, matches, unmatchedLeft, unmatchedRight) {
  const key = typeof keySpec === "function" ? keySpec : (node) => node[keySpec];
  const leftGroups = grouped(leftNodes.filter((node) => unmatchedLeft.has(node.index)), key);
  const rightGroups = grouped(rightNodes.filter((node) => unmatchedRight.has(node.index)), key);
  for (const [value, leftRows] of leftGroups) {
    const rightRows = rightGroups.get(value) || [];
    if (!value || leftRows.length !== 1 || rightRows.length !== 1) continue;
    const left = leftRows[0], right = rightRows[0];
    matches.push({ left, right, method, confidence });
    unmatchedLeft.delete(left.index);
    unmatchedRight.delete(right.index);
  }
}

function ambiguousCandidates(leftNodes, rightNodes, leftSet, rightSet) {
  const rightByContract = grouped(rightNodes.filter((node) => rightSet.has(node.index)), (node) => node.contract_signature);
  return leftNodes.filter((node) => leftSet.has(node.index)).map((node) => ({
    left_index: node.index,
    right_candidate_indices: (rightByContract.get(node.contract_signature) || []).map((candidate) => candidate.index).sort((a, b) => a - b),
    reason: "contract_signature_not_unique",
  })).filter((row) => row.right_candidate_indices.length > 0);
}

function nodeChanges(left, right) {
  const changes = [];
  for (const [field, label] of [["inputs", "input tensor contract"], ["outputs", "output tensor contract"], ["quantization", "quantization state"], ["quant_risk", "quantization risk"], ["placement", "placement evidence"], ["macs", "MAC contract"]]) {
    if (JSON.stringify(left[field]) !== JSON.stringify(right[field])) changes.push(label);
  }
  return changes;
}

function topologySignature(node, predecessors, successors) {
  const neighbor = (rows) => rows.filter(Boolean).map((op) => `${op.domain || ""}:${op.name || ""}`).sort().join(",");
  return `${node.contract_signature}|pred:${neighbor(predecessors)}|succ:${neighbor(successors)}`;
}
function contractSignature(node) { return `${node.domain}:${node.name}|in:${node.inputs.join(";")}|out:${node.outputs.join(";")}`; }
function tensorContract(tensor) {
  if (!tensor) return "missing";
  const canonical = tensor.artifact_ir_contract;
  const shapeSource = canonical?.shape ?? tensor.shape;
  const shape = Array.isArray(shapeSource) ? shapeSource.map((value) => String(value)).join("x") : "?";
  const scaleCount = quantizationCardinality(tensor.scales, tensor.scale_count, tensor.quantization_scale_count);
  const zeroCount = quantizationCardinality(tensor.zero_points, tensor.zero_point_count, tensor.quantization_zero_point_count);
  return `${String(canonical?.dtype || tensor.dtype || "UNKNOWN")}[${shape}]|q:${scaleCount}:${zeroCount}`;
}
function quantizationCardinality(values, ...candidates) {
  if (Array.isArray(values)) return values.length;
  const value = candidates.find((candidate) => candidate != null && candidate !== "");
  if (value == null) return 0;
  const numeric = Number(value);
  return Number.isSafeInteger(numeric) && numeric >= 0 ? numeric : "?";
}
function placementContract(op, format) {
  if (String(format).toLowerCase() === "tflite") return Number(op.xnnpack_chain_id) >= 0 ? "CONDITIONALLY_DELEGATABLE_XNNPACK" : op.xnnpack_chain_break ? "PREDICTED_BREAK" : "CPU_OR_UNRESOLVED";
  return String(op.ort_provider || op.execution_provider || op.placement_status || "NOT_OBSERVED");
}
function adjacency(edges, key, value) {
  const map = new Map();
  for (const edge of edges) { if (!map.has(edge[key])) map.set(edge[key], []); map.get(edge[key]).push(edge[value]); }
  return map;
}
function grouped(rows, key) {
  const map = new Map();
  for (const row of rows) { const value = key(row); if (!map.has(value)) map.set(value, []); map.get(value).push(row); }
  return map;
}
function validateSnapshot(value, label) {
  if (value?.schema !== GRAPH_DIFF_SNAPSHOT_SCHEMA || !Array.isArray(value.nodes) || value.node_count !== value.nodes.length) throw new Error(`${label} graph diff snapshot is invalid.`);
}
function integers(value) { return Array.isArray(value) ? value.map(Number).filter((item) => Number.isSafeInteger(item) && item >= 0) : []; }
function clean(value) { const text = String(value ?? "").trim(); return text || null; }
function exactText(value) { if (typeof value === "string" && /^\d+$/.test(value)) return value; const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? String(number) : null; }
