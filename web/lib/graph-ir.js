import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const GRAPH_IR_SCHEMA = "deepbom.graph_ir.v1";
const SHA256 = /^[a-f0-9]{64}$/;

export function buildCanonicalGraphIr(analysis, artifact) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const body = ops.length
    ? buildSerializedGraph(analysis, artifact, ops, tensors)
    : buildContainerProjection(analysis, artifact, tensors);
  validateGraphBody(body);
  return Object.freeze({ ...body, graph_ir_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateCanonicalGraphIr(document) {
  const body = clone(document);
  const digest = String(body.graph_ir_sha256 || "").toLowerCase();
  delete body.graph_ir_sha256;
  validateGraphBody(body);
  if (!SHA256.test(digest) || digest !== sha256TextHex(canonicalJson(body))) throw new Error("Graph IR SHA-256 is invalid.");
  return Object.freeze({ ...body, graph_ir_sha256: digest });
}

function buildSerializedGraph(analysis, artifact, ops, tensors) {
  const placementContext = serializedPlacementContext(analysis, ops);
  const tensorByIndex = new Map(tensors.map((tensor) => [Number(tensor.index), tensor]));
  const producer = new Map();
  const consumers = new Map();
  for (const op of ops) {
    for (const index of integerArray(op.outputs)) if (!producer.has(index)) producer.set(index, Number(op.index));
    for (const index of integerArray(op.inputs)) {
      if (!consumers.has(index)) consumers.set(index, []);
      consumers.get(index).push(Number(op.index));
    }
  }
  const nodes = ops.map((op) => ({
    id: `op:${op.index}`,
    index: Number(op.index),
    kind: "operator",
    label: String(op.name || `OP_${op.index}`),
    secondary_label: String(op.graph_node_name || op.coreml_layer_name || op.stage_key || "") || null,
    domain: String(op.domain || analysis.format || artifact.format),
    stage: String(op.stage_key ?? op.stage_index ?? "") || null,
    topo_depth: nonNegative(op.topo_depth),
    input_tensor_indices: integerArray(op.inputs),
    output_tensor_indices: integerArray(op.outputs),
    output_shapes: shapeList(op.output_shapes),
    macs: exactInteger(op.macs_decimal ?? op.macs),
    estimated_bytes: exactInteger(op.estimated_bytes),
    quantization: {
      state: String(op.quantization_state || (op.quantized_compute_path ? "8bit_compute" : "none")),
      risk: String(op.quant_risk || "none"),
    },
    placement: placement(op, analysis.format, placementContext),
  })).sort((left, right) => left.index - right.index);
  const edges = [];
  for (const tensor of [...tensors].sort((left, right) => Number(left.index) - Number(right.index))) {
    const index = Number(tensor.index);
    const from = producer.has(index) ? `op:${producer.get(index)}` : null;
    const targets = [...new Set(consumers.get(index) || [])].sort((a, b) => a - b);
    if (!targets.length) {
      if (from && integerArray(analysis.output_tensor_indices).includes(index)) edges.push(edge(tensor, from, null, "model_output"));
      continue;
    }
    for (const target of targets) edges.push(edge(tensor, from, `op:${target}`, from ? "operator_tensor" : "model_input_or_constant"));
  }
  const macTotals = graphMacTotals(analysis, nodes);
  return {
    schema: GRAPH_IR_SCHEMA,
    evidence_class: "OBSERVED_SERIALIZED_GRAPH_WITH_DERIVED_RELATIONSHIPS",
    format: String(analysis.format || artifact.format).toLowerCase(),
    artifact: identity(artifact),
    projection: {
      kind: "serialized_executable_graph",
      executable_dag_claim: true,
      placement_evidence: placementContext?.summary || null,
    },
    totals: {
      node_count: nodes.length,
      tensor_count: tensors.length,
      edge_count: edges.length,
      macs: macTotals.total,
      assessed_macs: macTotals.assessed,
      mac_assessment: macTotals.assessment,
    },
    inputs: integerArray(analysis.input_tensor_indices),
    outputs: integerArray(analysis.output_tensor_indices),
    nodes,
    edges,
    interpretation_boundary: "Operator nodes and tensor relationships are projected from the serialized artifact. Placement labels retain their source evidence class and do not imply observed accelerator assignment unless explicitly marked observed.",
  };
}

function buildContainerProjection(analysis, artifact, tensors) {
  const layerStorage = analysis?.on_device_llm?.storage?.layer_storage;
  const layers = Array.isArray(layerStorage?.layers) ? layerStorage.layers : [];
  let nodes;
  let kind;
  let evidence;
  const placementScenario = selectLlmPlacementScenario(analysis?.accelerator_profile_binding);
  const acceleratorLayers = new Set(placementScenario?.serialized_layer_offload?.selected_layer_indices || []);
  if (layers.length) {
    nodes = layers.map((layer) => ({
      id: `layer:${layer.layer_index}`,
      index: Number(layer.layer_index),
      kind: "architecture_layer",
      label: `Decoder layer ${layer.layer_index}`,
      secondary_label: `${layer.tensor_count} tensors`,
      domain: String(analysis?.on_device_llm?.architecture?.family || analysis.format || artifact.format),
      stage: "decoder",
      topo_depth: Number(layer.layer_index),
      input_tensor_indices: [], output_tensor_indices: [], output_shapes: [], macs: null,
      estimated_bytes: normalizeExact(layer.serialized_bytes),
      quantization: { state: "container_tensor_storage", risk: "not_assessed_at_layer_projection" },
      placement: placementScenario
        ? acceleratorLayers.has(Number(layer.layer_index))
          ? { status: "CONDITIONAL_SERIALIZED_LAYER_RESIDENCY_CANDIDATE", backend: "nvidia_accelerator", evidence_class: "DERIVED_CONDITIONAL_STATIC_LOWER_BOUND" }
          : { status: "CONDITIONAL_OTHER_POOL_RESIDENCY_CANDIDATE", backend: "cpu_or_other_pool", evidence_class: "DERIVED_CONDITIONAL_STATIC_LOWER_BOUND" }
        : { status: "NOT_ASSESSABLE", backend: null, evidence_class: "NOT_ASSESSABLE" },
    }));
    kind = "llm_layer_storage_architecture_projection";
    evidence = "OBSERVED_TENSOR_NAMESPACE_WITH_DERIVED_LAYER_ORDER";
  } else {
    const groups = new Map();
    for (const tensor of tensors) {
      const label = String(tensor.name || "unnamed").split(".").slice(0, 2).join(".") || "unnamed";
      if (!groups.has(label)) groups.set(label, { count: 0, bytes: 0n });
      const group = groups.get(label);
      group.count += 1;
      group.bytes += BigInt(exactInteger(tensor.byte_length)?.decimal || "0");
    }
    nodes = [...groups].sort(([left], [right]) => left.localeCompare(right)).map(([label, group], index) => ({
      id: `storage:${index}`, index, kind: "storage_namespace", label, secondary_label: `${group.count} tensors`,
      domain: String(analysis.format || artifact.format), stage: "storage", topo_depth: index,
      input_tensor_indices: [], output_tensor_indices: [], output_shapes: [], macs: null,
      estimated_bytes: exact(group.bytes), quantization: { state: "container_tensor_storage", risk: "not_assessed_at_namespace_projection" },
      placement: { status: "NOT_APPLICABLE", backend: null, evidence_class: "NOT_APPLICABLE" },
    }));
    kind = "tensor_storage_namespace_projection";
    evidence = "OBSERVED_TENSOR_DIRECTORY_WITH_DERIVED_NAMESPACE_GROUPING";
  }
  const edges = nodes.slice(1).map((node, index) => ({
    id: `projection:${index}:${index + 1}`, tensor_index: null, tensor_name: null, from: nodes[index].id, to: node.id,
    relation: "derived_architecture_order", dtype: null, shape: [], byte_length: null,
  }));
  return {
    schema: GRAPH_IR_SCHEMA, evidence_class: evidence, format: String(analysis.format || artifact.format).toLowerCase(), artifact: identity(artifact),
    projection: {
      kind,
      executable_dag_claim: false,
      placement_scenario: placementScenario ? {
        source: placementScenario.scenario_source,
        context_length: placementScenario.context_length,
        batch_size: placementScenario.batch_size,
        storage_bits: placementScenario.storage_bits,
        status: placementScenario.status,
        fit_claim: placementScenario.fit_claim,
      } : null,
    },
    totals: { node_count: nodes.length, tensor_count: tensors.length, edge_count: edges.length, macs: null },
    inputs: [], outputs: [], nodes, edges,
    interpretation_boundary: "GGUF and SafeTensors do not serialize an executable operator DAG. This view is an architecture or tensor-storage projection and must not be read as runtime lowering, kernel order, placement, or execution flow.",
  };
}

function selectLlmPlacementScenario(binding) {
  const rows = binding?.llm_accelerator_residency?.scenarios;
  if (!Array.isArray(rows)) return null;
  const selected = rows.filter((row) => row.scenario_source === "cli_declared");
  return selected.length === 1 && selected[0]?.serialized_layer_offload ? selected[0] : null;
}

function graphMacTotals(analysis, nodes) {
  const ledger = analysis?.mac_assessment;
  const compute = Number(ledger?.compute_ops);
  const assessedCount = Number(ledger?.assessed_compute_ops);
  const unassessedCount = Number(ledger?.not_assessed_compute_ops);
  const assessed = exactInteger(ledger?.total_assessed_macs_decimal ?? ledger?.total_assessed_macs);
  if (Number.isSafeInteger(compute) && compute >= 0 && Number.isSafeInteger(assessedCount) && assessedCount >= 0
    && Number.isSafeInteger(unassessedCount) && unassessedCount >= 0 && assessedCount + unassessedCount === compute && assessed) {
    return {
      total: unassessedCount === 0 ? assessed : null,
      assessed,
      assessment: {
        status: unassessedCount === 0 ? "complete" : assessedCount ? "partial" : "not_assessed",
        compute_op_count: compute,
        assessed_compute_op_count: assessedCount,
        unassessed_compute_op_count: unassessedCount,
        scope: String(ledger.metric_scope || "nominal tensor-contraction MACs"),
      },
    };
  }
  const sum = nodes.reduce((total, node) => total + BigInt(node.macs?.decimal || "0"), 0n);
  return {
    total: exact(sum),
    assessed: exact(sum),
    assessment: {
      status: "complete_without_separate_compute_denominator",
      compute_op_count: null,
      assessed_compute_op_count: null,
      unassessed_compute_op_count: null,
      scope: "sum_of_node_macs_present_in_canonical_graph_ir",
    },
  };
}

function edge(tensor, from, to, relation) {
  const index = Number(tensor.index);
  return {
    id: `tensor:${index}:${from || "external"}:${to || "external"}`,
    tensor_index: index,
    tensor_name: String(tensor.name || `tensor_${index}`),
    from, to, relation,
    dtype: String(tensor.dtype || "UNKNOWN"),
    shape: shape(tensor.shape),
    byte_length: exactInteger(tensor.byte_length ?? tensor.buffer_data_length),
  };
}

function serializedPlacementContext(analysis, ops) {
  const tensorRt = analysis?.tensorrt_static_preflight;
  const projection = tensorRt?.projection;
  if (String(analysis?.format || "").toLowerCase() !== "onnx" || !projection) return null;
  const rows = Array.isArray(projection.rows) ? projection.rows : [];
  const expected = new Set(ops.map((op) => Number(op.index)));
  const byIndex = new Map();
  for (const row of rows) {
    const index = Number(row?.op_index);
    if (!Number.isSafeInteger(index) || !expected.has(index) || byIndex.has(index)) {
      throw new Error(`TensorRT placement projection has an invalid or duplicate op #${String(row?.op_index)}.`);
    }
    byIndex.set(index, row);
  }
  if (byIndex.size !== expected.size) throw new Error("TensorRT placement projection does not cover every serialized ONNX operator.");
  const engine = tensorRt.engine_inspector_evidence;
  return {
    byIndex,
    profileId: String(projection.profile_id || "tensorrt"),
    evidenceClass: String(projection.evidence_class || tensorRt.evidence_class || "NOT_ASSESSABLE"),
    summary: {
      backend: String(projection.label || projection.profile_id || "TensorRT"),
      profile_id: String(projection.profile_id || "tensorrt"),
      projection_schema: String(projection.schema || ""),
      evidence_class: String(projection.evidence_class || tensorRt.evidence_class || "NOT_ASSESSABLE"),
      state_counts: clone(projection.state_counts || {}),
      parser_observation_status: tensorRt.parser_observation?.coverage_status || "not_observed",
      engine_inspector_status: engine?.status || "not_observed",
      engine_sha256: engine?.engine?.sha256 || null,
      engine_source_mapping_status: engine?.source_mapping_status || "not_exposed",
      original_op_engine_selection_claim: false,
      interpretation_boundary: "TensorRT parser rows may establish configuration-bound artifact eligibility. Engine inspector layers are not projected onto original ONNX operators unless an explicit identity mapping is established; this graph makes no such mapping claim.",
    },
  };
}

function placement(op, format, context = null) {
  if (String(format).toLowerCase() === "tflite") {
    return op.xnnpack_supported === true
      ? { status: "CONDITIONALLY_DELEGATABLE", backend: "xnnpack", evidence_state: "ARTIFACT_ELIGIBLE", evidence_class: "PREDICTED_SOURCE_AND_ARTIFACT_ELIGIBILITY" }
      : { status: op.xnnpack_supported === false ? "PREDICTED_FALLBACK_OR_BREAK" : "NOT_ASSESSABLE", backend: "cpu", evidence_state: op.xnnpack_supported === false ? "SOURCE_REGISTERED" : "NOT_ASSESSABLE", evidence_class: op.xnnpack_supported === false ? "PREDICTED" : "NOT_ASSESSABLE" };
  }
  if (String(format).toLowerCase() === "onnx" && context) {
    const row = context.byIndex.get(Number(op.index));
    if (row.state === "CONDITIONALLY_ELIGIBLE") {
      return {
        status: "CONDITIONALLY_ELIGIBLE",
        backend: context.profileId,
        evidence_state: "ARTIFACT_ELIGIBLE",
        evidence_class: context.evidenceClass,
        reason_codes: clone(row.reason_codes || []),
        unresolved_predicates: clone(row.unresolved_predicates || []),
      };
    }
    if (row.state === "DEFINITE_EXCLUSION") {
      return {
        status: "DEFINITE_EXCLUSION",
        backend: context.profileId,
        evidence_state: "BUILD_INCLUDED",
        evidence_class: context.evidenceClass,
        reason_codes: clone(row.reason_codes || []),
        unresolved_predicates: clone(row.unresolved_predicates || []),
      };
    }
    return {
      status: "NOT_ASSESSABLE",
      backend: context.profileId,
      evidence_state: "NOT_ASSESSABLE",
      evidence_class: context.evidenceClass,
      reason_codes: clone(row.reason_codes || []),
      unresolved_predicates: clone(row.unresolved_predicates || []),
    };
  }
  return { status: "NOT_ASSESSABLE", backend: null, evidence_state: "NOT_ASSESSABLE", evidence_class: "NOT_ASSESSABLE" };
}

function validateGraphBody(value) {
  if (value?.schema !== GRAPH_IR_SCHEMA || !text(value.format, 40) || !SHA256.test(String(value.artifact?.sha256 || ""))) throw new Error("Graph IR identity is invalid.");
  if (!Array.isArray(value.nodes) || !Array.isArray(value.edges) || value.totals?.node_count !== value.nodes.length
    || value.totals?.edge_count !== value.edges.length) throw new Error("Graph IR count conservation failed.");
  if (value.projection?.executable_dag_claim === true) {
    const nodeMacs = value.nodes.reduce((sum, node) => sum + BigInt(node.macs?.decimal || "0"), 0n);
    if (String(value.totals?.assessed_macs?.decimal ?? "") !== String(nodeMacs)
      || (value.totals?.mac_assessment?.unassessed_compute_op_count > 0 && value.totals?.macs != null)
      || (value.totals?.mac_assessment?.unassessed_compute_op_count === 0
        && String(value.totals?.macs?.decimal ?? "") !== String(nodeMacs))) {
      throw new Error("Graph IR MAC assessment does not conserve assessed node MACs.");
    }
  }
  const ids = new Set();
  for (const node of value.nodes) {
    if (!text(node.id, 160) || ids.has(node.id) || !text(node.label, 500)) throw new Error("Graph IR node identity is invalid.");
    ids.add(node.id);
  }
  const edgeIds = new Set();
  for (const row of value.edges) {
    if (!text(row.id, 600) || edgeIds.has(row.id) || (row.from && !ids.has(row.from)) || (row.to && !ids.has(row.to))) throw new Error("Graph IR edge identity is invalid.");
    edgeIds.add(row.id);
  }
  if (value.projection?.executable_dag_claim === true && value.totals.tensor_count < 0) throw new Error("Graph IR tensor count is invalid.");
  if (!text(value.interpretation_boundary, 1600)) throw new Error("Graph IR interpretation boundary is missing.");
}

function identity(artifact) { return { filename: String(artifact.filename), sha256: String(artifact.sha256), byte_length: { decimal: String(artifact.size), number: artifact.size }, artifact_set_sha256: artifact.artifact_set_sha256 || null }; }
function integerArray(value) { return Array.isArray(value) ? value.map(Number).filter(Number.isSafeInteger) : []; }
function shape(value) { return Array.isArray(value) ? value.map((item) => Number.isSafeInteger(Number(item)) ? Number(item) : String(item)) : []; }
function shapeList(value) { return Array.isArray(value) ? value.map(shape) : []; }
function nonNegative(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : null; }
function exactInteger(value) {
  if (typeof value === "string" && /^\d+$/.test(value)) return exact(BigInt(value));
  if (typeof value === "bigint" && value >= 0n) return exact(value);
  if (Number.isSafeInteger(Number(value)) && Number(value) >= 0) return exact(BigInt(Number(value)));
  return null;
}
function normalizeExact(value) { return exactInteger(value?.decimal ?? value?.value ?? value); }
function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }
function text(value, maximum) { const normalized = String(value || "").trim(); return normalized.length > 0 && normalized.length <= maximum; }
function clone(value) { return JSON.parse(JSON.stringify(value)); }
