import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { BACKEND_PLACEMENT_STATES, buildBackendPlacementProjection } from "./backend-placement-projection.js";

export const EDGETPU_COMPILER_EVIDENCE_SCHEMA = "deepbom.edgetpu_compiler_evidence.v1";
const SHA256 = /^[a-f0-9]{64}$/;
const MAPPING = new Set(["mapped", "unmapped"]);

export function parseEdgeTpuCompilerEvidence(source, analysis, { fileSha256 = null } = {}) {
  if (String(analysis?.format || "").toLowerCase() !== "tflite" || !Array.isArray(analysis?.ops)) {
    throw new Error("Edge TPU compiler evidence requires a decoded TFLite artifact.");
  }
  const value = typeof source === "string" ? JSON.parse(source) : source;
  if (!value || value.schema !== EDGETPU_COMPILER_EVIDENCE_SCHEMA) {
    throw new Error(`Edge TPU compiler evidence must use ${EDGETPU_COMPILER_EVIDENCE_SCHEMA}.`);
  }
  const artifactSha256 = requireSha(value.artifact_sha256, "artifact SHA-256");
  if (artifactSha256 !== requireSha(analysis.model_sha256, "active artifact SHA-256")) {
    throw new Error("Edge TPU compiler evidence is not bound to the active artifact.");
  }
  const compiler = {
    name: requiredText(value.compiler?.name, "compiler name"),
    version: requiredText(value.compiler?.version, "compiler version"),
    binary_sha256: requireSha(value.compiler?.binary_sha256, "compiler binary SHA-256"),
  };
  const options = validateOptions(value.invocation?.options);
  const invocationSha256 = sha256TextHex(canonicalJson({ options }));
  if (value.invocation?.sha256 != null && requireSha(value.invocation.sha256, "invocation SHA-256") !== invocationSha256) {
    throw new Error("Edge TPU compiler invocation SHA-256 does not reproduce from normalized options.");
  }
  const operations = validateOperations(value.operations, analysis.ops);
  const mapped = operations.filter((row) => row.mapping === "mapped").length;
  const summary = {
    operation_count: operations.length,
    mapped_operation_count: mapped,
    unmapped_operation_count: operations.length - mapped,
    mapping_coverage_ratio: operations.length ? mapped / operations.length : null,
  };
  if (value.summary && canonicalJson(value.summary) !== canonicalJson(summary)) {
    throw new Error("Edge TPU compiler evidence summary does not reproduce from operation rows.");
  }
  const body = {
    schema: EDGETPU_COMPILER_EVIDENCE_SCHEMA,
    evidence_class: "COMPILER_OBSERVED_COMPILED_ARTIFACT_BOUND",
    artifact_sha256: artifactSha256,
    compiler,
    invocation: { options },
    invocation_sha256: invocationSha256,
    compiled_artifact_sha256: requireSha(value.compiled_artifact_sha256, "compiled artifact SHA-256"),
    compiler_report_sha256: requireSha(value.compiler_report_sha256 || fileSha256, "compiler report SHA-256"),
    operations,
    summary,
    execution_status: "not_observed_compiler_evidence_only",
    interpretation_boundary: "The compiler report establishes mapped and unmapped serialized TFLite operations for one compiler binary, invocation, and compiled artifact. It does not establish device execution, physical transfer, latency, task accuracy, or release readiness.",
  };
  return Object.freeze({ ...body, evidence_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateEdgeTpuCompilerEvidence(value, analysis) {
  const copy = JSON.parse(JSON.stringify(value || null));
  const declared = String(copy?.evidence_sha256 || "").toLowerCase();
  if (copy) delete copy.evidence_sha256;
  const reconstructed = parseEdgeTpuCompilerEvidence({
    schema: copy?.schema,
    artifact_sha256: copy?.artifact_sha256,
    compiler: copy?.compiler,
    invocation: { options: copy?.invocation?.options, sha256: copy?.invocation_sha256 },
    compiled_artifact_sha256: copy?.compiled_artifact_sha256,
    compiler_report_sha256: copy?.compiler_report_sha256,
    operations: copy?.operations,
    summary: copy?.summary,
  }, analysis);
  if (!SHA256.test(declared) || reconstructed.evidence_sha256 !== declared) {
    throw new Error("Edge TPU compiler evidence SHA-256 is invalid.");
  }
  return value;
}

export function buildEdgeTpuCompilerProjection(analysis, evidence) {
  validateEdgeTpuCompilerEvidence(evidence, analysis);
  return buildBackendPlacementProjection({
    analysis,
    profileId: "google_edgetpu_compiled_plan",
    label: "Google Edge TPU compiler result",
    evidenceClass: evidence.evidence_class,
    rows: evidence.operations.map((row) => ({
      op_index: row.op_index,
      state: row.mapping === "mapped"
        ? BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE
        : BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION,
      reason_codes: row.mapping === "mapped" ? [] : [row.reason],
      unresolved_predicates: row.mapping === "mapped" ? ["device_execution_not_observed"] : [],
    })),
    source: {
      compiler_name: evidence.compiler.name,
      compiler_version: evidence.compiler.version,
      compiler_binary_sha256: evidence.compiler.binary_sha256,
      compiled_artifact_sha256: evidence.compiled_artifact_sha256,
      compiler_report_sha256: evidence.compiler_report_sha256,
    },
    interpretationBoundary: evidence.interpretation_boundary,
  });
}

function validateOperations(value, graphOps) {
  if (!Array.isArray(value) || value.length !== graphOps.length) {
    throw new Error("Edge TPU compiler evidence must classify every serialized graph operation exactly once.");
  }
  return value.map((row, position) => {
    const opIndex = Number(row?.op_index);
    const graphOp = graphOps[position];
    const expectedIndex = Number.isSafeInteger(Number(graphOp?.index)) ? Number(graphOp.index) : position;
    if (!Number.isSafeInteger(opIndex) || opIndex !== expectedIndex) {
      throw new Error(`Edge TPU compiler operation index mismatch at row ${position}.`);
    }
    const opName = requiredText(row?.op_name, `operation name ${opIndex}`);
    if (opName !== String(graphOp?.name || "")) throw new Error(`Edge TPU compiler operation name mismatch at #${opIndex}.`);
    const mapping = String(row?.mapping || "");
    if (!MAPPING.has(mapping)) throw new Error(`Edge TPU compiler mapping is invalid at #${opIndex}.`);
    const reason = row?.reason == null ? null : requiredText(row.reason, `operation reason ${opIndex}`);
    if (mapping === "unmapped" && !reason) throw new Error(`Unmapped Edge TPU operation #${opIndex} requires a reason.`);
    return { op_index: opIndex, op_name: opName, mapping, reason };
  });
}

function validateOptions(value) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || !item.trim() || item.length > 1024)) {
    throw new Error("Edge TPU compiler options must be a bounded string array.");
  }
  return value.map((item) => item.trim());
}

function requiredText(value, label) {
  const text = String(value || "").trim();
  if (!text || text.length > 1024) throw new Error(`Edge TPU ${label} is invalid.`);
  return text;
}

function requireSha(value, label) {
  const text = String(value || "").toLowerCase();
  if (!SHA256.test(text)) throw new Error(`Edge TPU ${label} is invalid.`);
  return text;
}
