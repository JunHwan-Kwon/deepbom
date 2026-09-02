import { buildArtifactEvidenceIrUnchecked, validateArtifactEvidenceIr } from "./artifact-ir.js";
import { normalizeArtifactIrRuntimeOverlay } from "./artifact-ir-runtime.js";
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

  const artifactIr = buildArtifactEvidenceIrUnchecked(analysis, normalizedArtifact, { runtimeEvidence });
  const context = contextFromArtifactIr(analysis, artifactIr, { runtimeSignature });
  CACHE.set(analysis, { cache_key: cacheKey, context });
  return context;
}

export function resolveArtifactIrContext(analysis, artifact = {}, {
  artifactIrContext = null,
  artifactIr = null,
  runtimeEvidence = null,
} = {}) {
  if (!analysis || typeof analysis !== "object") return null;
  if (artifactIrContext) {
    const validated = validateArtifactEvidenceIr(artifactIrContext.artifact_ir);
    requireArtifactIdentity(validated, analysis, artifact);
    if (validated.artifact_ir_sha256 !== artifactIrContext.graph_ir?.artifact_ir_sha256) {
      throw new Error("Artifact IR context graph binding is inconsistent.");
    }
    requireRuntimeIdentity(validated, runtimeEvidence, artifactIrContext.runtime_signature);
    return artifactIrContext;
  }
  const supplied = artifactIr || analysis.artifact_ir || null;
  if (supplied) {
    const validated = validateArtifactEvidenceIr(supplied);
    requireArtifactIdentity(validated, analysis, artifact);
    requireRuntimeIdentity(validated, runtimeEvidence);
    return contextFromArtifactIr(analysis, validated, {
      runtimeSignature: runtimeEvidence == null ? null : sha256TextHex(canonicalJson(runtimeEvidence)),
    });
  }
  return getArtifactIrContext(analysis, artifact, { runtimeEvidence });
}

function contextFromArtifactIr(analysis, artifactIr, { runtimeSignature = null } = {}) {
  const graphIr = projectArtifactIrToCanonicalGraph(artifactIr);
  const primaryView = buildPrimaryScopeAnalysisView(analysis, artifactIr);
  return Object.freeze({
    artifact_ir: artifactIr,
    runtime_signature: runtimeSignature,
    graph_ir: graphIr,
    primary_view: primaryView,
    operator_by_subject_ref: new Map(artifactIr.graph.operators.map((row) => [row.id, row])),
    value_by_subject_ref: new Map(artifactIr.graph.values.map((row) => [row.id, row])),
    scope_by_ref: new Map(artifactIr.graph.scopes.map((row) => [row.id, row])),
  });
}

function requireRuntimeIdentity(artifactIr, runtimeEvidence, contextRuntimeSignature = null) {
  if (!hasMaterialRuntimeEvidence(runtimeEvidence)) return;
  const expectedSignature = sha256TextHex(canonicalJson(runtimeEvidence));
  if (contextRuntimeSignature != null && contextRuntimeSignature !== expectedSignature) {
    throw new Error("Artifact IR context is stale for the supplied runtime evidence.");
  }
  const expected = normalizeArtifactIrRuntimeOverlay(
    runtimeEvidence,
    artifactIr.graph,
    artifactIr.architecture_projection,
    artifactIr.artifact.sha256,
  );
  if (canonicalJson(expected) !== canonicalJson(artifactIr.overlays.runtime)) {
    throw new Error("Artifact IR runtime overlay is stale for the supplied runtime evidence.");
  }
}

function hasMaterialRuntimeEvidence(runtimeEvidence) {
  return runtimeEvidence && typeof runtimeEvidence === "object"
    && [runtimeEvidence.runtime_nodes, runtimeEvidence.rows, runtimeEvidence.assignments]
      .some((rows) => Array.isArray(rows) && rows.length > 0);
}

function requireArtifactIdentity(artifactIr, analysis, artifact) {
  const expected = String(artifact.sha256 || analysis.model_sha256 || analysis.artifact_sha256 || "").toLowerCase();
  if (/^[a-f0-9]{64}$/.test(expected) && artifactIr.artifact.sha256 !== expected) {
    throw new Error("Artifact IR is not bound to the active artifact SHA-256.");
  }
}

export function buildPrimaryScopeAnalysisView(analysis, artifactIr) {
  const graphSerialized = artifactIr?.graph?.status === "serialized";
  const primaryScopeRef = graphSerialized ? artifactIr.graph.primary_scope_ref : null;
  const nativeOps = Array.isArray(analysis.ops) ? analysis.ops : [];
  const nativeTensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const opByNativeIndex = new Map(nativeOps.map((row, index) => [nativeIndex(row, index), row]));
  const tensorByNativeIndex = new Map(nativeTensors.map((row, index) => [nativeIndex(row, index), row]));
  const ops = graphSerialized
    ? artifactIr.graph.operators.filter((row) => row.scope_ref === primaryScopeRef).map((canonical) => ({
      ...(opByNativeIndex.get(canonical.native_index) || canonicalOperatorFallback(canonical, artifactIr.graph.values)),
      index: canonical.native_index,
      artifact_ir_subject_ref: canonical.id,
      artifact_ir_scope_ref: canonical.scope_ref,
      artifact_ir_contract: canonical,
    }))
    : nativeOps;
  const tensors = graphSerialized
    ? artifactIr.graph.values.filter((row) => row.scope_ref === primaryScopeRef).map((canonical) => ({
      ...(tensorByNativeIndex.get(canonical.native_index) || canonicalValueFallback(canonical)),
      index: canonical.native_index,
      artifact_ir_subject_ref: canonical.id,
      artifact_ir_scope_ref: canonical.scope_ref,
      artifact_ir_contract: canonical,
    }))
    : nativeTensors;
  const view = {};
  const canonicalKeys = new Set(["ops", "tensors", "artifact_ir", "artifact_ir_consumer_view", "artifact_ir_primary_scope_ref", "artifact_ir_nested_scope_count"]);
  const passthroughKeys = new Set([
    ...Object.keys(analysis),
    "_markdown",
    "_reportGeneratedAt",
    "external_node_edge_evidence_overlay",
    "findings",
    "on_device_llm",
  ]);
  if (String(analysis.format || "").toLowerCase() === "tflite") {
    passthroughKeys.add("deployment_delta");
    passthroughKeys.add("deployment_delta_error");
  }
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
    artifact_ir_consumer_view: true,
    artifact_ir_primary_scope_ref: primaryScopeRef,
    artifact_ir_nested_scope_count: graphSerialized
      ? artifactIr.graph.scopes.filter((row) => row.id !== primaryScopeRef).length
      : 0,
  });
  return view;
}

export function isArtifactIrConsumerView(analysis) {
  return analysis?.artifact_ir_consumer_view === true
    && analysis?.artifact_ir?.schema === "deepbom.artifact_ir.v2";
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
