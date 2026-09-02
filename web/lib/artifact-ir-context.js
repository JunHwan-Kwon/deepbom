import { buildArtifactEvidenceIr } from "./artifact-ir.js";
import { projectArtifactIrToCanonicalGraph } from "./graph-ir.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

const CACHE = new WeakMap();

export function getArtifactIrContext(analysis, artifact = {}, { runtimeEvidence = null } = {}) {
  if (!analysis || typeof analysis !== "object") return null;
  const sha256 = String(artifact.sha256 || analysis.model_sha256 || analysis.artifact_sha256 || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(sha256)) return null;
  const normalizedArtifact = {
    filename: String(artifact.filename || analysis.filename || "model"),
    format: String(artifact.format || analysis.format || "unknown"),
    sha256,
    size: artifact.size ?? artifact.byte_length ?? analysis.file_size_bytes ?? analysis.file_size ?? null,
    artifact_set_sha256: artifact.artifact_set_sha256 || analysis?.artifact_set?.artifact_set_sha256 || null,
  };
  const runtimeSignature = sha256TextHex(canonicalJson(runtimeEvidence || null));
  const cacheKey = canonicalJson({ artifact: normalizedArtifact, runtime_signature: runtimeSignature });
  const cached = CACHE.get(analysis);
  if (cached?.cache_key === cacheKey) return cached.context;

  const artifactIr = buildArtifactEvidenceIr(analysis, normalizedArtifact, { runtimeEvidence });
  const graphIr = projectArtifactIrToCanonicalGraph(artifactIr);
  const primaryView = buildPrimaryScopeAnalysisView(analysis, artifactIr);
  const context = Object.freeze({
    artifact_ir: artifactIr,
    graph_ir: graphIr,
    primary_view: primaryView,
    operator_by_subject_ref: new Map(artifactIr.graph.operators.map((row) => [row.id, row])),
    value_by_subject_ref: new Map(artifactIr.graph.values.map((row) => [row.id, row])),
    scope_by_ref: new Map(artifactIr.graph.scopes.map((row) => [row.id, row])),
  });
  CACHE.set(analysis, { cache_key: cacheKey, context });
  return context;
}

export function buildPrimaryScopeAnalysisView(analysis, artifactIr) {
  if (artifactIr?.graph?.status !== "serialized") return analysis;
  const primaryScopeRef = artifactIr.graph.primary_scope_ref;
  const nativeOps = new Map((Array.isArray(analysis.ops) ? analysis.ops : []).map((row, index) => [nativeIndex(row, index), row]));
  const nativeTensors = new Map((Array.isArray(analysis.tensors) ? analysis.tensors : []).map((row, index) => [nativeIndex(row, index), row]));
  const ops = artifactIr.graph.operators.filter((row) => row.scope_ref === primaryScopeRef).map((canonical) => ({
    ...(nativeOps.get(canonical.native_index) || canonicalOperatorFallback(canonical, artifactIr.graph.values)),
    index: canonical.native_index,
    artifact_ir_subject_ref: canonical.id,
    artifact_ir_scope_ref: canonical.scope_ref,
    artifact_ir_contract: canonical,
  }));
  const tensors = artifactIr.graph.values.filter((row) => row.scope_ref === primaryScopeRef).map((canonical) => ({
    ...(nativeTensors.get(canonical.native_index) || canonicalValueFallback(canonical)),
    index: canonical.native_index,
    artifact_ir_subject_ref: canonical.id,
    artifact_ir_scope_ref: canonical.scope_ref,
    artifact_ir_contract: canonical,
  }));
  const view = {};
  const canonicalKeys = new Set(["ops", "tensors", "artifact_ir", "artifact_ir_primary_scope_ref", "artifact_ir_nested_scope_count"]);
  const passthroughKeys = new Set([
    ...Object.keys(analysis),
    "_markdown",
    "_reportGeneratedAt",
    "deployment_delta",
    "deployment_delta_error",
    "external_node_edge_evidence_overlay",
    "findings",
    "on_device_llm",
  ]);
  for (const key of passthroughKeys) {
    if (canonicalKeys.has(key)) continue;
    Object.defineProperty(view, key, {
      enumerable: true,
      configurable: false,
      get: () => analysis[key],
      set: (value) => { analysis[key] = value; },
    });
  }
  Object.assign(view, {
    ops,
    tensors,
    artifact_ir: artifactIr,
    artifact_ir_primary_scope_ref: primaryScopeRef,
    artifact_ir_nested_scope_count: artifactIr.graph.scopes.filter((row) => row.id !== primaryScopeRef).length,
  });
  return view;
}

function canonicalOperatorFallback(operator, values) {
  const valueById = new Map(values.map((row) => [row.id, row]));
  return {
    name: operator.op_type,
    domain: operator.domain,
    version: operator.version,
    inputs: operator.inputs.map((row) => valueById.get(row.value_ref)?.native_index).filter(Number.isSafeInteger),
    outputs: operator.outputs.map((row) => valueById.get(row.value_ref)?.native_index).filter(Number.isSafeInteger),
    macs: operator.metrics?.macs?.number,
    macs_decimal: operator.metrics?.macs?.decimal,
    macs_status: operator.metrics?.mac_assessment_status,
    estimated_bytes: operator.metrics?.logical_io_bytes?.number,
    topo_depth: operator.topology?.depth,
    topo_role: operator.topology?.role,
    stage_key: operator.topology?.stage,
  };
}

function canonicalValueFallback(value) {
  return {
    name: value.name,
    dtype: value.dtype,
    shape: [...value.shape],
    shape_signature: value.shape_signature ? [...value.shape_signature] : [],
    role: value.roles.includes("graph_input") ? "input" : value.roles.includes("graph_output") ? "output" : "",
    constant_buffer: value.roles.includes("serialized_constant_or_storage"),
    byte_length: value.logical_byte_length?.number,
  };
}

function nativeIndex(row, fallback) {
  const value = Number(row?.index);
  return Number.isSafeInteger(value) && value >= 0 ? value : fallback;
}
