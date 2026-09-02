import { buildArtifactEvidenceIr, validateArtifactEvidenceIr } from "./artifact-ir.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const GRAPH_IR_SCHEMA = "deepbom.graph_ir.v1";
const SHA256 = /^[a-f0-9]{64}$/;

export function buildCanonicalGraphIr(analysis, artifact) {
  return projectArtifactIrToCanonicalGraph(buildArtifactEvidenceIr(analysis, artifact));
}

export function projectArtifactIrToCanonicalGraph(artifactIrDocument) {
  const artifactIr = validateArtifactEvidenceIr(artifactIrDocument);
  const body = artifactIr.graph.status === "serialized"
    ? projectSerializedGraph(artifactIr)
    : projectContainerEvidence(artifactIr);
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

function projectSerializedGraph(artifactIr) {
  const graph = artifactIr.graph;
  const primaryScopeRef = graph.primary_scope_ref;
  const primaryOperators = graph.operators.filter((operator) => operator.scope_ref === primaryScopeRef);
  const primaryValues = graph.values.filter((value) => value.scope_ref === primaryScopeRef);
  const valueById = new Map(primaryValues.map((value) => [value.id, value]));
  const staticOverlay = artifactIr.overlays.static.find((row) => row.kind === "static_placement") || null;
  const placementBySubject = new Map((staticOverlay?.rows || []).map((row) => [row.subject_ref, legacyPlacement(row)]));
  const nodes = primaryOperators.map((operator) => ({
    id: operator.legacy_graph_node_id,
    index: operator.native_index,
    kind: "operator",
    label: operator.op_type,
    secondary_label: operator.name && operator.name !== operator.op_type ? operator.name : null,
    domain: operator.domain,
    stage: operator.topology.stage,
    topo_depth: operator.topology.depth,
    input_tensor_indices: operator.inputs.map((port) => valueById.get(port.value_ref)?.native_index).filter(Number.isSafeInteger),
    output_tensor_indices: operator.outputs.map((port) => valueById.get(port.value_ref)?.native_index).filter(Number.isSafeInteger),
    output_shapes: operator.outputs.map((port) => valueById.get(port.value_ref)?.shape || []),
    macs: clone(operator.metrics.macs),
    estimated_bytes: clone(operator.metrics.logical_io_bytes),
    quantization: clone(operator.quantization_summary),
    placement: placementBySubject.get(operator.id) || notAssessablePlacement(),
    artifact_ir_subject_ref: operator.id,
  }));
  const legacyNodeId = new Map(primaryOperators.map((operator) => [operator.id, operator.legacy_graph_node_id]));
  const graphOutputIds = new Set(graph.outputs);
  const edges = [];
  for (const value of primaryValues) {
    const from = value.producer ? legacyNodeId.get(value.producer.operator_ref) || null : null;
    if (!value.consumers.length) {
      if (from && graphOutputIds.has(value.id)) edges.push(legacyEdge(value, from, null, "model_output"));
      continue;
    }
    for (const consumer of value.consumers) {
      const to = legacyNodeId.get(consumer.operator_ref) || null;
      if (to) edges.push(legacyEdge(value, from, to, from ? "operator_tensor" : "model_input_or_constant"));
    }
  }
  return {
    schema: GRAPH_IR_SCHEMA,
    evidence_class: "OBSERVED_SERIALIZED_GRAPH_WITH_DERIVED_RELATIONSHIPS",
    format: artifactIr.artifact.format,
    artifact: legacyArtifactIdentity(artifactIr.artifact),
    artifact_ir_schema: artifactIr.schema,
    artifact_ir_sha256: artifactIr.artifact_ir_sha256,
    projection: {
      kind: "serialized_executable_graph",
      executable_dag_claim: true,
      source: "deepbom.artifact_ir.v2.graph",
      compatibility_status: "primary_scope_projection_only",
      source_scope_ref: primaryScopeRef,
      omitted_materialized_scope_count: graph.scopes.filter((scope) => scope.id !== primaryScopeRef && scope.materialization_status === "materialized").length,
      placement_evidence: clone(staticOverlay?.summary || null),
    },
    totals: {
      node_count: nodes.length,
      tensor_count: primaryValues.length,
      edge_count: edges.length,
      macs: clone(graph.totals.macs),
      assessed_macs: clone(graph.totals.assessed_macs),
      mac_assessment: clone(graph.totals.mac_assessment),
    },
    inputs: graph.inputs.map((id) => valueById.get(id)?.native_index).filter(Number.isSafeInteger),
    outputs: graph.outputs.map((id) => valueById.get(id)?.native_index).filter(Number.isSafeInteger),
    nodes,
    edges,
    interpretation_boundary: "Deprecated compatibility projection from deepbom.artifact_ir.v2. graph_ir.v1 projects only the primary serialized scope so nested or conditional scopes are never flattened into a false execution DAG. New consumers must use artifact_ir.v2 for complete scoped evidence. Placement remains a separate overlay and does not imply observed accelerator assignment unless its evidence class explicitly says so.",
  };
}

function projectContainerEvidence(artifactIr) {
  const architecture = artifactIr.architecture_projection;
  const sourceNodes = architecture.nodes.length ? architecture.nodes : artifactIr.storage_topology.objects;
  const staticOverlay = artifactIr.overlays.static.find((row) => row.kind === "static_placement") || null;
  const placementBySubject = new Map((staticOverlay?.rows || []).map((row) => [row.subject_ref, legacyPlacement(row)]));
  const nodes = sourceNodes.map((row, position) => {
    const layer = row.kind === "decoder_layer_storage_group";
    const index = Number.isSafeInteger(Number(row.native_index)) ? Number(row.native_index) : position;
    const subjectRef = row.id;
    return {
      id: layer ? `layer:${index}` : `storage:${index}`,
      index,
      kind: layer ? "architecture_layer" : "storage_namespace",
      label: String(row.label || row.name || `Storage ${index}`),
      secondary_label: row.tensor_count != null ? `${row.tensor_count} tensors` : null,
      domain: artifactIr.artifact.format,
      stage: layer ? "decoder" : "storage",
      topo_depth: index,
      input_tensor_indices: [], output_tensor_indices: [], output_shapes: [], macs: null,
      estimated_bytes: clone(row.serialized_bytes || row.serialized_byte_length || null),
      quantization: { state: "container_tensor_storage", risk: "not_assessed_at_non_graph_projection" },
      placement: placementBySubject.get(subjectRef) || { status: "NOT_APPLICABLE", backend: null, evidence_state: "NOT_APPLICABLE", evidence_class: "NOT_APPLICABLE" },
      artifact_ir_subject_ref: subjectRef,
    };
  });
  const layerProjection = architecture.kind === "llm_layer_storage";
  return {
    schema: GRAPH_IR_SCHEMA,
    evidence_class: layerProjection ? "OBSERVED_TENSOR_NAMESPACE_WITH_DERIVED_LAYER_GROUPING" : "OBSERVED_TENSOR_DIRECTORY_WITH_DERIVED_NAMESPACE_GROUPING",
    format: artifactIr.artifact.format,
    artifact: legacyArtifactIdentity(artifactIr.artifact),
    artifact_ir_schema: artifactIr.schema,
    artifact_ir_sha256: artifactIr.artifact_ir_sha256,
    projection: {
      kind: layerProjection ? "llm_layer_storage_architecture_projection" : "tensor_storage_namespace_projection",
      executable_dag_claim: false,
      source: "deepbom.artifact_ir.v2.architecture_projection",
      placement_scenario: clone(staticOverlay?.summary || null),
    },
    totals: { node_count: nodes.length, tensor_count: artifactIr.storage_topology.objects.length, edge_count: 0, macs: null, assessed_macs: null, mac_assessment: null },
    inputs: [], outputs: [], nodes, edges: [],
    interpretation_boundary: "Compatibility projection from deepbom.artifact_ir.v2. GGUF and SafeTensors do not serialize an executable operator DAG. Architecture and storage nodes have no synthesized execution edges and must not be read as runtime lowering, kernel order, placement, or execution flow.",
  };
}

function legacyEdge(value, from, to, relation) {
  return {
    id: `tensor:${value.native_index}:${from || "external"}:${to || "external"}`,
    tensor_index: value.native_index,
    tensor_name: value.name,
    from, to, relation,
    dtype: value.dtype,
    shape: clone(value.shape),
    byte_length: clone(value.logical_byte_length),
    artifact_ir_value_ref: value.id,
  };
}

function legacyPlacement(row) {
  const state = String(row.state || "NOT_ASSESSABLE");
  return {
    status: state === "UNRESOLVED" ? "NOT_ASSESSABLE" : state,
    backend: row.backend || null,
    evidence_state: row.evidence_state || "NOT_ASSESSABLE",
    evidence_class: row.evidence_class || "NOT_ASSESSABLE",
    reason_codes: clone(row.reason_codes || []),
    unresolved_predicates: clone(row.unresolved_predicates || []),
  };
}

function notAssessablePlacement() {
  return { status: "NOT_ASSESSABLE", backend: null, evidence_state: "NOT_ASSESSABLE", evidence_class: "NOT_ASSESSABLE", reason_codes: [], unresolved_predicates: [] };
}

function validateGraphBody(value) {
  if (value?.schema !== GRAPH_IR_SCHEMA || !text(value.format, 40) || !SHA256.test(String(value.artifact?.sha256 || ""))) throw new Error("Graph IR identity is invalid.");
  if (value.artifact_ir_schema !== "deepbom.artifact_ir.v2" || !SHA256.test(String(value.artifact_ir_sha256 || ""))) throw new Error("Graph IR Artifact IR binding is invalid.");
  if (!Array.isArray(value.nodes) || !Array.isArray(value.edges) || value.totals?.node_count !== value.nodes.length || value.totals?.edge_count !== value.edges.length) throw new Error("Graph IR count conservation failed.");
  if (value.projection?.executable_dag_claim === true) {
    const nodeMacs = value.nodes.reduce((sum, node) => sum + BigInt(node.macs?.decimal || "0"), 0n);
    if (String(value.totals?.assessed_macs?.decimal ?? "") !== String(nodeMacs)
      || (value.totals?.mac_assessment?.unassessed_compute_op_count > 0 && value.totals?.macs != null)
      || (value.totals?.mac_assessment?.unassessed_compute_op_count === 0 && String(value.totals?.macs?.decimal ?? "") !== String(nodeMacs))) {
      throw new Error("Graph IR MAC assessment does not conserve assessed node MACs.");
    }
  } else if (value.edges.length) {
    throw new Error("Graphless Graph IR projection contains synthesized edges.");
  }
  const ids = new Set();
  for (const node of value.nodes) {
    if (!text(node.id, 160) || ids.has(node.id) || !text(node.label, 500) || !text(node.artifact_ir_subject_ref, 600)) throw new Error("Graph IR node identity is invalid.");
    ids.add(node.id);
  }
  const edgeIds = new Set();
  for (const row of value.edges) {
    if (!text(row.id, 600) || edgeIds.has(row.id) || (row.from && !ids.has(row.from)) || (row.to && !ids.has(row.to)) || !text(row.artifact_ir_value_ref, 600)) throw new Error("Graph IR edge identity is invalid.");
    edgeIds.add(row.id);
  }
  if (!text(value.interpretation_boundary, 1800)) throw new Error("Graph IR interpretation boundary is missing.");
}

function legacyArtifactIdentity(artifact) {
  return {
    filename: artifact.filename,
    sha256: artifact.sha256,
    byte_length: clone(artifact.byte_length),
    artifact_set_sha256: artifact.artifact_set_sha256,
  };
}

function text(value, maximum) { const normalized = String(value || "").trim(); return normalized.length > 0 && normalized.length <= maximum; }
function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }
