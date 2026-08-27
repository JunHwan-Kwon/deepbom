import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const RUNTIME_EVIDENCE_SIDECAR_SCHEMA = "deepbom.runtime_evidence_sidecar.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const ASSIGNMENT_SCHEMA = /^deepbom\.runtime_assignment\.v1(?:\.\d+)?$/;

export function buildRuntimeEvidenceSidecar(analysis, sourceEvidence) {
  if (!sourceEvidence) return null;
  const format = requiredFormat(analysis?.format);
  const artifactSha256 = requiredSha(analysis?.model_sha256, "active artifact SHA-256");
  const sourceSchema = requiredText(sourceEvidence.schema, "source evidence schema");
  const normalized = ASSIGNMENT_SCHEMA.test(sourceSchema)
    ? normalizeAssignment(format, artifactSha256, analysis, sourceEvidence)
    : sourceSchema === "deepbom.gguf_runtime_environment.v2"
      ? normalizeGguf(format, artifactSha256, sourceEvidence)
      : sourceSchema === "deepbom.coreml_compute_plan.v1"
        ? normalizeCoreMl(format, artifactSha256, sourceEvidence)
        : null;
  if (!normalized) throw new Error(`Runtime evidence schema ${sourceSchema} has no common sidecar adapter.`);
  const sourceEvidenceSha256 = sha256TextHex(canonicalJson(sourceEvidence));
  const body = {
    schema: RUNTIME_EVIDENCE_SIDECAR_SCHEMA,
    artifact: {
      format,
      sha256: artifactSha256,
      content_set_sha256: optionalSha(normalized.contentSetSha256, "artifact content-set SHA-256"),
    },
    runtime: normalized.runtime,
    build: normalized.build,
    capture: normalized.capture,
    configuration: normalized.configuration,
    observations: normalized.observations,
    source_contract: {
      schema: sourceSchema,
      evidence_sha256: sourceEvidenceSha256,
      normalization: "RFC8785-JCS source digest followed by schema-specific field projection",
      original_document_required_for_full_verification: true,
    },
    claim_boundary: "This sidecar is a format-neutral evidence index. It preserves build, placement, timing, and memory claim states without promoting requirements, compatibility candidates, or compute-plan estimates to observed execution. The source document identified by source_contract.evidence_sha256 remains authoritative.",
  };
  return { ...body, sidecar_sha256: sha256TextHex(canonicalJson(body)) };
}

export function verifyRuntimeEvidenceSidecar(sidecar, analysis, sourceEvidence = null) {
  if (!sidecar || sidecar.schema !== RUNTIME_EVIDENCE_SIDECAR_SCHEMA) {
    throw new Error(`Runtime evidence sidecar must use ${RUNTIME_EVIDENCE_SIDECAR_SCHEMA}.`);
  }
  const { sidecar_sha256: recorded, ...body } = sidecar;
  if (!SHA256.test(String(recorded || "")) || sha256TextHex(canonicalJson(body)) !== recorded) {
    throw new Error("Runtime evidence sidecar canonical SHA-256 is invalid.");
  }
  if (analysis) {
    if (sidecar.artifact?.format !== requiredFormat(analysis.format)
      || sidecar.artifact?.sha256 !== requiredSha(analysis.model_sha256, "active artifact SHA-256")) {
      throw new Error("Runtime evidence sidecar is not bound to the active artifact.");
    }
  }
  if (sourceEvidence) {
    const expected = buildRuntimeEvidenceSidecar(analysis, sourceEvidence);
    if (canonicalJson(expected) !== canonicalJson(sidecar)) {
      throw new Error("Runtime evidence sidecar does not reconstruct from its source evidence.");
    }
  }
  validateClaims(sidecar.observations);
  return sidecar;
}

function normalizeAssignment(format, artifactSha256, analysis, evidence) {
  if (!["tflite", "onnx"].includes(format)) throw new Error("Runtime assignment evidence is valid only for TFLite or ONNX artifacts.");
  if (requiredSha(evidence.artifact_sha256, "runtime assignment artifact SHA-256") !== artifactSha256) {
    throw new Error("Runtime assignment evidence does not match the active artifact SHA-256.");
  }
  const assignments = Array.isArray(evidence.assignments) ? evidence.assignments : [];
  const graphOpCount = nonNegativeInteger(evidence.graph_op_count ?? analysis?.ops?.length ?? 0, "graph op count");
  const observedAssignments = assignments.length;
  if (observedAssignments > graphOpCount) throw new Error("Runtime assignment coverage exceeds the active graph op count.");
  const timed = assignments.filter((row) => finiteNonNegative(row?.duration_us)).length;
  const collector = evidence.source?.collector || null;
  const adapter = evidence.source?.adapter || null;
  const nativeCapture = adapter?.native_capture || null;
  const selector = evidence.selector_context || null;
  const binarySha256 = optionalSha(
    evidence.runtime?.binary_sha256 || selector?.build?.runtime_binary_sha256 || nativeCapture?.runtime?.primary_binary_sha256,
    "runtime binary SHA-256",
  );
  const sourceCommit = optionalCommit(
    selector?.build?.xnnpack_source_commit || nativeCapture?.runtime?.source_commit || adapter?.source_commit,
  );
  const memory = evidence.runtime_memory || null;
  const collectedAt = optionalTimestamp(evidence.source?.collected_at || nativeCapture?.collection?.collected_at);
  return {
    contentSetSha256: nativeCapture?.artifact?.content_set_sha256 || null,
    runtime: {
      family: format === "onnx" ? "onnxruntime" : "tensorflow_lite",
      name: requiredText(evidence.runtime?.name || (format === "onnx" ? "ONNX Runtime" : "TensorFlow Lite"), "runtime name"),
      version: optionalText(evidence.runtime?.version),
      backend: optionalText(evidence.runtime?.backend),
      binary_sha256: binarySha256,
    },
    build: buildManifest({
      repository: format === "onnx" ? "https://github.com/microsoft/onnxruntime" : "https://github.com/tensorflow/tensorflow",
      sourceCommit,
      binarySha256,
      configurationSha256: selector?.build?.microkernel_build_identifier_sha256
        || nativeCapture?.runtime?.binary_inventory_sha256
        || evidence.tflite_delegate_build_inventory?.ledger_sha256
        || null,
      status: binarySha256 && sourceCommit ? "source_and_binary_bound" : binarySha256 ? "binary_bound_source_unbound" : "runtime_identity_declared_binary_unbound",
    }),
    capture: {
      capture_id: optionalText(evidence.source?.capture_id || nativeCapture?.capture_id),
      collected_at: collectedAt,
      collector_name: optionalText(collector?.name || (nativeCapture ? "deepbom-ort-native-capture" : adapter?.schema)),
      collector_version: optionalText(collector?.version),
    },
    configuration: {
      target_profile_id: optionalText(evidence.target_profile_id),
      target_profile_sha256: optionalSha(evidence.target_profile_sha256, "target-profile SHA-256"),
      execution_mode: optionalText(evidence.runtime?.execution_mode),
      graph_optimization_level: optionalText(evidence.runtime?.graph_optimization_level),
      requested_backend: optionalText(evidence.runtime?.backend),
    },
    observations: {
      build_identity: claim(binarySha256 ? "bound" : "unbound", binarySha256 ? "OBSERVED_RUNTIME" : "DECLARED", binarySha256 ? 1 : 0, 1),
      placement: claim(observedAssignments === graphOpCount && graphOpCount ? "complete" : observedAssignments ? "partial" : "not_observed", "OBSERVED_RUNTIME", observedAssignments, graphOpCount),
      timing: claim(timed === graphOpCount && graphOpCount ? "complete" : timed ? "partial" : "not_collected", timed ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", timed, graphOpCount),
      memory: claim(memory?.status === "assessed" ? "observed" : "not_collected", memory?.status === "assessed" ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", memory?.status === "assessed" ? Number(memory.snapshot_count || 1) : 0, null),
      execution: claim(observedAssignments ? "observed_assignment" : "not_observed", observedAssignments ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", observedAssignments, graphOpCount),
    },
  };
}

function normalizeGguf(format, artifactSha256, evidence) {
  if (format !== "gguf" || requiredSha(evidence.artifact?.sha256, "GGUF runtime artifact SHA-256") !== artifactSha256) {
    throw new Error("GGUF runtime evidence does not match the active GGUF artifact.");
  }
  const graphCount = nonNegativeInteger(evidence.compute_graph?.graph_count ?? 0, "GGUF generated graph count");
  const scheduled = nonNegativeInteger(evidence.compute_graph?.scheduled_node_count ?? 0, "GGUF scheduled node count");
  const dispatched = nonNegativeInteger(evidence.compute_graph?.dispatched_graph_count ?? 0, "GGUF dispatched graph count");
  const binarySha256 = requiredSha(evidence.runtime?.binary_sha256, "GGUF runtime binary SHA-256");
  const sourceCommit = optionalCommit(evidence.runtime?.source_commit);
  return {
    contentSetSha256: null,
    runtime: { family: "llama_cpp", name: "llama.cpp", version: optionalText(evidence.runtime?.version_output), backend: optionalText(evidence.selection?.requested_backend_profile_id), binary_sha256: binarySha256 },
    build: buildManifest({ repository: evidence.runtime?.repository, sourceCommit, binarySha256, configurationSha256: evidence.build?.attestation?.file_sha256 || evidence.build?.cmake_cache_sha256, status: "source_binary_and_build_bound" }),
    capture: { capture_id: optionalText(evidence.capture?.capture_id), collected_at: optionalTimestamp(evidence.capture?.collected_at), collector_name: optionalText(evidence.capture?.collector?.name), collector_version: optionalText(evidence.capture?.collector?.version) },
    configuration: { target_profile_id: null, target_profile_sha256: null, execution_mode: "generated_scheduler_graph", graph_optimization_level: null, requested_backend: optionalText(evidence.selection?.requested_backend_profile_id) },
    observations: {
      build_identity: claim("bound", "OBSERVED_RUNTIME", 1, 1),
      placement: claim(graphCount && scheduled ? "complete_captured_graphs" : "not_observed", graphCount ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", scheduled, scheduled || null),
      timing: claim(finiteNonNegative(evidence.observations?.elapsed_ms) ? "observed_aggregate" : "not_collected", finiteNonNegative(evidence.observations?.elapsed_ms) ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", finiteNonNegative(evidence.observations?.elapsed_ms) ? 1 : 0, 1),
      memory: claim("not_collected", "NOT_ASSESSED", 0, null),
      execution: claim(dispatched ? "observed_dispatch" : evidence.observations?.model_load_status === "observed_success" ? "observed_load_only" : "not_observed", dispatched ? "OBSERVED_RUNTIME" : "NOT_ASSESSED", dispatched, graphCount || null),
    },
  };
}

function normalizeCoreMl(format, artifactSha256, evidence) {
  if (format !== "coreml" || requiredSha(evidence.artifact?.sha256, "Core ML compute-plan artifact SHA-256") !== artifactSha256) {
    throw new Error("Core ML compute-plan evidence does not match the active Core ML artifact.");
  }
  const rows = Array.isArray(evidence.structure?.rows) ? evidence.structure.rows : [];
  if (!rows.length || evidence.execution_status !== "not_observed_compute_plan_only") throw new Error("Core ML compute-plan sidecar input must contain estimate-only operation rows.");
  const binarySha256 = requiredSha(evidence.runtime?.compiled_model_content_sha256, "compiled Core ML model SHA-256");
  const planSourceSha256 = requiredSha(evidence.runtime?.coremltools_compute_plan_source_sha256, "Core ML compute-plan source SHA-256");
  const collectorSourceSha256 = requiredSha(evidence.capture?.collector?.source_sha256, "Core ML collector source SHA-256");
  const computeDevices = Array.isArray(evidence.runtime?.available_compute_devices) ? evidence.runtime.available_compute_devices : [];
  if (evidence.runtime?.platform_system !== "Darwin" || !computeDevices.length
    || new Set(computeDevices.map((device) => device?.type)).size !== computeDevices.length
    || computeDevices.some((device) => !Number.isSafeInteger(Number(device?.instance_count)) || Number(device.instance_count) <= 0)) {
    throw new Error("Core ML compute-plan sidecar input requires a macOS host and unique compute-device inventory.");
  }
  const costRows = rows.filter((row) => finiteNonNegative(row?.estimated_cost_weight)).length;
  return {
    contentSetSha256: null,
    runtime: { family: "coreml", name: "Core ML", version: optionalText(evidence.runtime?.coremltools_version), backend: optionalText(evidence.configuration?.compute_units), binary_sha256: binarySha256 },
    build: buildManifest({ repository: "https://github.com/apple/coremltools", sourceCommit: null, binarySha256, configurationSha256: planSourceSha256, status: "compiled_model_and_plan_source_bound" }),
    capture: {
      capture_id: optionalText(evidence.capture?.capture_id),
      collected_at: optionalTimestamp(evidence.capture?.collected_at),
      collector_name: optionalText(evidence.capture?.collector?.name),
      collector_version: optionalText(evidence.capture?.collector?.version),
      collector_source_sha256: collectorSourceSha256,
      host: {
        platform: requiredText(evidence.runtime?.platform_system, "Core ML host platform"),
        os_version: requiredText(evidence.runtime?.macos_version, "Core ML host OS version"),
        os_build: requiredText(evidence.runtime?.os_build, "Core ML host OS build"),
        hardware_model: requiredText(evidence.runtime?.hardware_model, "Core ML hardware model"),
        architecture: requiredText(evidence.runtime?.architecture, "Core ML host architecture"),
        available_compute_devices: computeDevices,
      },
    },
    configuration: { target_profile_id: null, target_profile_sha256: null, execution_mode: "compute_plan_estimate", graph_optimization_level: null, requested_backend: optionalText(evidence.configuration?.compute_units) },
    observations: {
      build_identity: claim("bound", "OBSERVED_RUNTIME", 1, 1),
      placement: claim(rows.length ? "estimated_plan" : "not_assessed", rows.length ? "RUNTIME_PLAN_ESTIMATE" : "NOT_ASSESSED", rows.length, rows.length || null),
      timing: claim(costRows ? "relative_cost_estimate" : "not_assessed", costRows ? "RUNTIME_PLAN_ESTIMATE" : "NOT_ASSESSED", costRows, rows.length || null),
      memory: claim("not_collected", "NOT_ASSESSED", 0, null),
      execution: claim("not_observed", "NOT_ASSESSED", 0, rows.length || null),
    },
  };
}

function buildManifest({ repository, sourceCommit, binarySha256, configurationSha256, status }) {
  return {
    status,
    source_repository: optionalText(repository),
    source_commit: optionalCommit(sourceCommit),
    runtime_binary_sha256: optionalSha(binarySha256, "runtime binary SHA-256"),
    configuration_sha256: optionalSha(configurationSha256, "build configuration SHA-256"),
  };
}

function claim(status, evidenceClass, numerator, denominator) {
  return { status, evidence_class: evidenceClass, assessed_count: numerator, eligible_count: denominator };
}

function validateClaims(value) {
  for (const key of ["build_identity", "placement", "timing", "memory", "execution"]) {
    const row = value?.[key];
    if (!row || !requiredText(row.status, `${key} status`) || !requiredText(row.evidence_class, `${key} evidence class`)) throw new Error(`Runtime evidence sidecar ${key} claim is invalid.`);
    nonNegativeInteger(row.assessed_count, `${key} assessed count`);
    if (row.eligible_count != null) {
      const denominator = nonNegativeInteger(row.eligible_count, `${key} eligible count`);
      if (row.assessed_count > denominator) throw new Error(`Runtime evidence sidecar ${key} count exceeds its denominator.`);
    }
  }
}

function requiredFormat(value) { const format = String(value || "").toLowerCase(); if (!["tflite", "onnx", "gguf", "coreml"].includes(format)) throw new Error("Runtime evidence sidecars require TFLite, ONNX, GGUF, or Core ML."); return format; }
function requiredSha(value, label) { const text = String(value || "").toLowerCase(); if (!SHA256.test(text)) throw new Error(`${label} is required.`); return text; }
function optionalSha(value, label) { return value == null || value === "" ? null : requiredSha(value, label); }
function requiredText(value, label) { const text = String(value || "").trim(); if (!text || text.length > 8192) throw new Error(`${label} is required and bounded.`); return text; }
function optionalText(value) { const text = String(value ?? "").trim(); return text ? text.slice(0, 8192) : null; }
function optionalCommit(value) { const text = optionalText(value); if (!text) return null; const commit = text.includes("@") ? text.split("@").at(-1) : text; return /^[a-f0-9]{40}$/i.test(commit) ? commit.toLowerCase() : null; }
function optionalTimestamp(value) { const text = optionalText(value); if (!text) return null; const time = Date.parse(text); if (!Number.isFinite(time)) throw new Error("Runtime evidence capture timestamp is invalid."); return new Date(time).toISOString(); }
function nonNegativeInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${label} must be a non-negative safe integer.`); return number; }
function finiteNonNegative(value) { return value != null && Number.isFinite(Number(value)) && Number(value) >= 0; }
