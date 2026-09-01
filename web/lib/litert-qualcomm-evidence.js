import { BACKEND_PLACEMENT_STATES, buildBackendPlacementProjection } from "./backend-placement-projection.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { tfliteAcceleratorSourceManifest } from "./tflite-accelerator-source-profiles.js";

export const LITERT_QUALCOMM_EVIDENCE_SCHEMA = "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1";
const SHA256 = /^[a-f0-9]{64}$/;
const COMPILE_STATES = new Set(["compiled", "not_compiled"]);
const DISPATCH_STATES = new Set(["not_observed", "loaded", "invoked"]);

export function parseLiteRtQualcommEvidence(source, analysis, { fileSha256 = null } = {}) {
  if (String(analysis?.format || "").toLowerCase() !== "tflite" || !Array.isArray(analysis?.ops)) {
    throw new Error("LiteRT Qualcomm evidence requires a decoded TFLite artifact.");
  }
  const value = typeof source === "string" ? JSON.parse(source) : source;
  if (value?.schema !== LITERT_QUALCOMM_EVIDENCE_SCHEMA) {
    throw new Error(`LiteRT Qualcomm evidence must use ${LITERT_QUALCOMM_EVIDENCE_SCHEMA}.`);
  }
  const artifactSha256 = requireSha(value.artifact_sha256, "artifact SHA-256");
  if (artifactSha256 !== requireSha(analysis.model_sha256, "active artifact SHA-256")) {
    throw new Error("LiteRT Qualcomm evidence is not bound to the active artifact.");
  }
  const sourceProfile = tfliteAcceleratorSourceManifest().profiles.find((row) => row.id === "litert_qualcomm_qnn");
  const sourceCommit = requiredCommit(value.source?.litert_commit, "LiteRT source commit");
  if (sourceCommit !== sourceProfile.source.commit) throw new Error("LiteRT Qualcomm evidence source commit differs from the analyzer source profile.");
  const sourceRulepackSha256 = requireSha(value.source?.rulepack_sha256, "source rulepack SHA-256");
  if (sourceRulepackSha256 !== sourceProfile.rulepack_sha256) throw new Error("LiteRT Qualcomm evidence rulepack SHA-256 differs from the analyzer source profile.");
  const compiler = {
    name: requiredText(value.compiler?.name, "compiler name"),
    version: requiredText(value.compiler?.version, "compiler version"),
    binary_sha256: requireSha(value.compiler?.binary_sha256, "compiler binary SHA-256"),
  };
  const options = stringArray(value.invocation?.options, "compiler options");
  const invocationSha256 = sha256TextHex(canonicalJson({ options }));
  if (value.invocation?.sha256 != null && requireSha(value.invocation.sha256, "invocation SHA-256") !== invocationSha256) {
    throw new Error("LiteRT Qualcomm invocation SHA-256 does not reproduce from normalized options.");
  }
  const operations = operationRows(value.operations, analysis.ops);
  const compiledCount = operations.filter((row) => row.compile_status === "compiled").length;
  const dispatchStatus = String(value.dispatch?.status || "not_observed");
  if (!DISPATCH_STATES.has(dispatchStatus)) throw new Error("LiteRT Qualcomm dispatch status is invalid.");
  const runtimeBinarySha256 = value.dispatch?.runtime_binary_sha256 == null
    ? null : requireSha(value.dispatch.runtime_binary_sha256, "dispatch runtime binary SHA-256");
  const dispatchEvidenceSha256 = value.dispatch?.evidence_sha256 == null
    ? null : requireSha(value.dispatch.evidence_sha256, "dispatch evidence SHA-256");
  if (dispatchStatus !== "not_observed" && (!runtimeBinarySha256 || !dispatchEvidenceSha256)) {
    throw new Error("Observed LiteRT Qualcomm dispatch state requires runtime binary and dispatch evidence SHA-256 values.");
  }
  const summary = {
    operation_count: operations.length,
    compiled_operation_count: compiledCount,
    not_compiled_operation_count: operations.length - compiledCount,
    dispatch_status: dispatchStatus,
  };
  if (value.summary && canonicalJson(value.summary) !== canonicalJson(summary)) {
    throw new Error("LiteRT Qualcomm summary does not reproduce from operation and dispatch evidence.");
  }
  const body = {
    schema: LITERT_QUALCOMM_EVIDENCE_SCHEMA,
    evidence_class: "COMPILER_OBSERVED_COMPILED_ARTIFACT_BOUND",
    artifact_sha256: artifactSha256,
    source: { litert_commit: sourceCommit, rulepack_sha256: sourceRulepackSha256 },
    compiler,
    invocation: { options },
    invocation_sha256: invocationSha256,
    compiled_plan_sha256: requireSha(value.compiled_plan_sha256, "compiled QNN plan SHA-256"),
    source_file_sha256: requireSha(value.source_file_sha256 || fileSha256, "evidence source-file SHA-256"),
    operations,
    dispatch: {
      status: dispatchStatus,
      runtime_binary_sha256: runtimeBinarySha256,
      evidence_sha256: dispatchEvidenceSha256,
    },
    summary,
    interpretation_boundary: dispatchStatus === "not_observed"
      ? "Compiler evidence binds source, compiler binary, options, operation legalization results, and compiled QNN plan. Dispatch, device assignment, execution, physical transfer, latency, and task correctness are not observed."
      : "Compiler and dispatch evidence bind one QNN plan and runtime binary. Per-source-op execution, physical transfer, latency, and task correctness are not inferred unless separately measured and identity-bound.",
  };
  return Object.freeze({ ...body, evidence_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateLiteRtQualcommEvidence(value, analysis) {
  const copy = JSON.parse(JSON.stringify(value || null));
  const declared = String(copy?.evidence_sha256 || "").toLowerCase();
  if (copy) delete copy.evidence_sha256;
  const rebuilt = parseLiteRtQualcommEvidence({
    ...copy,
    invocation: { ...copy?.invocation, sha256: copy?.invocation_sha256 },
  }, analysis, { fileSha256: copy?.source_file_sha256 });
  if (!SHA256.test(declared) || declared !== rebuilt.evidence_sha256) throw new Error("LiteRT Qualcomm evidence SHA-256 is invalid.");
  return value;
}

export function buildLiteRtQualcommCompilerProjection(analysis, evidence) {
  validateLiteRtQualcommEvidence(evidence, analysis);
  return buildBackendPlacementProjection({
    analysis,
    profileId: "litert_qualcomm_qnn_compiled_plan",
    label: "LiteRT Qualcomm compiled QNN plan",
    evidenceClass: evidence.evidence_class,
    rows: evidence.operations.map((row) => ({
      op_index: row.op_index,
      state: row.compile_status === "compiled"
        ? BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE
        : BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION,
      reason_codes: row.compile_status === "compiled" ? [] : [row.reason],
      unresolved_predicates: row.compile_status === "compiled"
        ? [evidence.dispatch.status === "not_observed" ? "qnn_dispatch_not_observed" : "per_source_op_execution_not_observed"] : [],
    })),
    source: {
      ...evidence.source,
      compiler_binary_sha256: evidence.compiler.binary_sha256,
      compiled_plan_sha256: evidence.compiled_plan_sha256,
      dispatch_status: evidence.dispatch.status,
      dispatch_runtime_binary_sha256: evidence.dispatch.runtime_binary_sha256,
    },
    interpretationBoundary: evidence.interpretation_boundary,
  });
}

function operationRows(value, graphOps) {
  if (!Array.isArray(value) || value.length !== graphOps.length) {
    throw new Error("LiteRT Qualcomm evidence must classify every serialized graph operation exactly once.");
  }
  return value.map((row, position) => {
    const expected = Number.isSafeInteger(Number(graphOps[position]?.index)) ? Number(graphOps[position].index) : position;
    const opIndex = Number(row?.op_index);
    if (!Number.isSafeInteger(opIndex) || opIndex !== expected) throw new Error(`LiteRT Qualcomm operation index mismatch at row ${position}.`);
    const opName = requiredText(row?.op_name, `operation name ${opIndex}`);
    if (opName !== String(graphOps[position]?.name || "")) throw new Error(`LiteRT Qualcomm operation name mismatch at #${opIndex}.`);
    const compileStatus = String(row?.compile_status || "");
    if (!COMPILE_STATES.has(compileStatus)) throw new Error(`LiteRT Qualcomm compile status is invalid at #${opIndex}.`);
    const reason = row?.reason == null ? null : requiredText(row.reason, `operation reason ${opIndex}`);
    if (compileStatus === "not_compiled" && !reason) throw new Error(`LiteRT Qualcomm non-compiled operation #${opIndex} requires a reason.`);
    return { op_index: opIndex, op_name: opName, compile_status: compileStatus, reason };
  });
}

function stringArray(value, label) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || !item.trim() || item.length > 1024)) {
    throw new Error(`LiteRT Qualcomm ${label} must be a bounded string array.`);
  }
  return value.map((item) => item.trim());
}

function requiredText(value, label) {
  const text = String(value || "").trim();
  if (!text || text.length > 1024) throw new Error(`LiteRT Qualcomm ${label} is invalid.`);
  return text;
}

function requiredCommit(value, label) {
  const text = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{40}$/.test(text)) throw new Error(`LiteRT Qualcomm ${label} is invalid.`);
  return text;
}

function requireSha(value, label) {
  const text = String(value || "").toLowerCase();
  if (!SHA256.test(text)) throw new Error(`LiteRT Qualcomm ${label} is invalid.`);
  return text;
}
