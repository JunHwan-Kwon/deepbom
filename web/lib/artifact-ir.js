import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
// Internal construction boundary. Consumers must resolve a shared context through
// artifact-ir-context.js so runtime overlays and scoped identities cannot diverge.
import { ARTIFACT_IR_METHOD_VERSION, ARTIFACT_IR_SCHEMA, GRAPH_FORMATS, SHA256 } from "./artifact-ir/internal/constants.js";
import { buildArchitectureProjection } from "./artifact-ir/internal/architecture.js";
import { buildSerializedGraph, notSerializedGraph } from "./artifact-ir/internal/graph.js";
import { artifactIdentity } from "./artifact-ir/internal/identity.js";
import { buildOverlays } from "./artifact-ir/internal/overlays.js";
import { buildQuantizationContracts } from "./artifact-ir/internal/quantization.js";
import { clone, deepFreeze, list, normalizeFormat } from "./artifact-ir/internal/shared.js";
import { buildStorageTopology } from "./artifact-ir/internal/storage.js";
import { validateArtifactIrBody } from "./artifact-ir/internal/validation.js";

export { ARTIFACT_IR_METHOD_VERSION, ARTIFACT_IR_SCHEMA };

export function buildArtifactEvidenceIrUnchecked(analysis, artifact = {}, { runtimeEvidence = null } = {}) {
  if (!analysis || typeof analysis !== "object") throw new Error("Artifact IR requires analyzed artifact evidence.");
  const format = normalizeFormat(analysis.format || artifact.format);
  const identity = artifactIdentity(analysis, artifact, format);
  const tensors = list(analysis.tensors);
  const graph = GRAPH_FORMATS.has(format)
    ? buildSerializedGraph(analysis, format, tensors)
    : notSerializedGraph(format);
  const storage = buildStorageTopology(analysis, format, tensors);
  const architecture = buildArchitectureProjection(analysis, format, tensors);
  const quantization = buildQuantizationContracts(analysis, format, graph, storage, tensors);
  const overlays = buildOverlays(analysis, format, graph, architecture, runtimeEvidence, identity.sha256);
  const body = {
    schema: ARTIFACT_IR_SCHEMA,
    method_version: ARTIFACT_IR_METHOD_VERSION,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      source_encoding: "UTF-8",
      excluded_pointers: ["/artifact_ir_sha256"],
    },
    artifact: identity,
    evidence_layers: {
      native_evidence: "artifact_serialized_or_hash_bound_sidecar_facts",
      canonical_ir: "format_neutral_identity_and_relationship_projection",
      static_projection: "separate_overlay_not_graph_fact",
      runtime_evidence: overlays.runtime.length ? "imported_identity_bound_overlay" : "not_imported",
    },
    graph,
    storage_topology: storage,
    architecture_projection: architecture,
    quantization_contracts: quantization,
    overlays,
    completeness: completeness(graph, storage, architecture, quantization, overlays),
    interpretation_boundary: "The canonical graph, storage, architecture, and quantization ledgers preserve distinct evidence scopes. Static backend eligibility and imported runtime observations are overlays and never rewrite serialized artifact facts. A missing executable graph is represented as not_serialized; no execution edge is synthesized from tensor names or architecture order.",
  };
  validateArtifactIrBody(body);
  return deepFreeze({ ...body, artifact_ir_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateArtifactEvidenceIr(document) {
  const body = clone(document);
  const digest = String(body.artifact_ir_sha256 || "").toLowerCase();
  delete body.artifact_ir_sha256;
  validateArtifactIrBody(body);
  if (!SHA256.test(digest) || digest !== sha256TextHex(canonicalJson(body))) {
    throw new Error("Artifact IR SHA-256 is invalid.");
  }
  return deepFreeze({ ...body, artifact_ir_sha256: digest });
}

function completeness(graph, storage, architecture, quantization, overlays) {
  return {
    graph: graph.completeness,
    storage: storage.status,
    architecture: architecture.status,
    quantization: quantization.status,
    static_overlay_count: overlays.static.length,
    runtime_overlay_count: overlays.runtime.length,
    unknown_is_zero: false,
  };
}
