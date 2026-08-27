import { TFLITE_DELEGATE_RULEPACK_METADATA } from "./tflite-delegate-rulepack-metadata.js";

const SHA256 = /^[a-f0-9]{64}$/;
const SOURCE_COMMIT = /^[a-f0-9]{40}$/;
const PROFILE_IDS = ["tflite_gpu", "tflite_nnapi"];
const ROW_STATUSES = new Set(["source_candidate_partial", "definite_exclusion"]);
const REQUIREMENT_IDS = new Set([
  "tflite_gpu_delegate_compiled",
  "tflite_gpu_allow_quant_ops",
  "tflite_nnapi_delegate_compiled",
  "tflite_nnapi_feature_level",
]);

export function applyProtectedTfliteDelegateCompatibilityEvidence(analysis, evidence) {
  validateProtectedTfliteDelegateCompatibilityEvidence(analysis, evidence);
  analysis.tflite_delegate_compatibility_evidence = clone(evidence);
  return analysis;
}

export function validateProtectedTfliteDelegateCompatibilityEvidence(analysis, evidence) {
  if (!analysis || !Array.isArray(analysis.ops)) {
    throw new Error("TFLite delegate compatibility evidence requires a current static analysis.");
  }
  if (!evidence || evidence.schema !== TFLITE_DELEGATE_RULEPACK_METADATA.schema
    || evidence.rulepack_sha256 !== TFLITE_DELEGATE_RULEPACK_METADATA.manifestSha256
    || evidence.tensorflow_source_commit !== TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit
    || !SOURCE_COMMIT.test(String(evidence.tensorflow_source_commit || ""))) {
    throw new Error("TFLite delegate compatibility rulepack identity is invalid.");
  }
  validateSourceFiles(evidence.source_files);

  const isTflite = String(analysis.format || "").toLowerCase() === "tflite";
  if (!isTflite) {
    if (evidence.assessment_status !== "not_applicable_non_tflite"
      || evidence.evidence_class !== "NOT_APPLICABLE"
      || Number(evidence.graph_op_count) !== analysis.ops.length
      || (evidence.profiles || []).length !== 0
      || (evidence.build_requirements || []).length !== 0) {
      throw new Error("Non-TFLite delegate evidence must be explicitly not applicable and empty.");
    }
    return true;
  }

  if (evidence.assessment_status !== "assessed_source_candidates_with_unresolved_runtime_predicates"
    || evidence.evidence_class !== "SOURCE_PINNED/DERIVED_PARTIAL"
    || Number(evidence.graph_op_count) !== analysis.ops.length
    || !String(evidence.interpretation_boundary || "").includes("not selected-build availability")) {
    throw new Error("TFLite delegate compatibility evidence boundary is invalid.");
  }

  const profiles = evidence.profiles;
  if (!Array.isArray(profiles) || profiles.length !== PROFILE_IDS.length
    || profiles.map((profile) => profile.id).join("|") !== PROFILE_IDS.join("|")) {
    throw new Error("TFLite delegate profile coverage is invalid.");
  }
  for (const profile of profiles) validateProfile(analysis, profile);
  validateBuildRequirements(profiles, evidence.build_requirements);
  return true;
}

function validateSourceFiles(sourceFiles) {
  const expected = TFLITE_DELEGATE_RULEPACK_METADATA.sources;
  if (!Array.isArray(sourceFiles) || sourceFiles.length !== expected.length) {
    throw new Error("TFLite delegate source-file ledger is incomplete.");
  }
  for (const source of expected) {
    const row = sourceFiles.find((candidate) => candidate.id === source.id);
    if (!row || row.sha256 !== source.sha256 || !SHA256.test(String(row.sha256 || ""))
      || !String(row.scope || "").trim()
      || !sourceRefMatches(row.source_ref, source.path)) {
      throw new Error(`TFLite delegate source-file identity is invalid for ${source.id}.`);
    }
  }
}

function validateProfile(analysis, profile) {
  const metadata = TFLITE_DELEGATE_RULEPACK_METADATA.profiles.find((item) => item.id === profile.id);
  if (!metadata || profile.label !== metadata.label
    || profile.assessment_status !== "assessed_artifact_visible_subset"
    || profile.evidence_class !== "SOURCE_PINNED/DERIVED_PARTIAL"
    || Number(profile.registered_source_op_count) !== metadata.registeredOpCount
    || Number(profile.assessed_graph_op_count) !== analysis.ops.length
    || profile.selected_build_status !== "not_bound"
    || profile.runtime_assignment_status !== "not_observed"
    || profile.op_count_conservation_status !== "complete"
    || !Array.isArray(profile.rows) || profile.rows.length !== analysis.ops.length) {
    throw new Error(`TFLite ${profile.id} profile identity is invalid.`);
  }

  let candidates = 0;
  let exclusions = 0;
  const seen = new Set();
  for (const [position, row] of profile.rows.entries()) {
    const op = analysis.ops[position];
    const opIndex = Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position;
    const opVersion = Number.isSafeInteger(Number(op?.version)) ? Number(op.version) : 1;
    if (!row || seen.has(Number(row.op_index)) || Number(row.op_index) !== opIndex
      || row.op_name !== op?.name || Number(row.op_version) !== opVersion
      || !ROW_STATUSES.has(row.artifact_precheck_status)
      || !["registered", "not_registered"].includes(row.source_registration_status)
      || !Array.isArray(row.definite_exclusion_reasons)
      || !Array.isArray(row.unresolved_predicates)
      || !Array.isArray(row.source_maximum_op_version_candidates)
      || !Array.isArray(row.source_feature_level_tokens)
      || !row.artifact_facts || !sourceRefMatches(row.source_ref, profile.id === "tflite_gpu"
        ? "tensorflow/lite/delegates/gpu/common/model_builder.cc"
        : "tensorflow/lite/delegates/nnapi/nnapi_delegate.cc")) {
      throw new Error(`TFLite ${profile.id} op identity is invalid at #${opIndex}.`);
    }
    seen.add(Number(row.op_index));
    if (row.source_registration_status === "registered") {
      if (!SHA256.test(String(row.source_text_sha256 || ""))) {
        throw new Error(`TFLite ${profile.id} registered source fragment is unpinned at #${opIndex}.`);
      }
    } else if (row.source_text_sha256 != null) {
      throw new Error(`TFLite ${profile.id} unregistered op has a source-fragment hash at #${opIndex}.`);
    }
    if (row.artifact_precheck_status === "source_candidate_partial") {
      candidates += 1;
      if (row.definite_exclusion_reasons.length || !row.unresolved_predicates.length) {
        throw new Error(`TFLite ${profile.id} candidate boundary is invalid at #${opIndex}.`);
      }
    } else {
      exclusions += 1;
      if (!row.definite_exclusion_reasons.length || row.unresolved_predicates.length) {
        throw new Error(`TFLite ${profile.id} exclusion boundary is invalid at #${opIndex}.`);
      }
    }
  }
  if (candidates + exclusions !== analysis.ops.length
    || Number(profile.source_candidate_after_artifact_precheck_count) !== candidates
    || Number(profile.definite_exclusion_count) !== exclusions
    || Number(profile.unresolved_candidate_count) !== candidates) {
    throw new Error(`TFLite ${profile.id} profile counts do not conserve graph ops.`);
  }
}

function validateBuildRequirements(profiles, requirements) {
  if (!Array.isArray(requirements) || requirements.length !== REQUIREMENT_IDS.size
    || new Set(requirements.map((row) => row.id)).size !== requirements.length) {
    throw new Error("TFLite delegate build-requirement coverage is invalid.");
  }
  for (const row of requirements) {
    const profile = profiles.find((candidate) => candidate.id === row.profile);
    const expectedAffected = row.id === "tflite_gpu_allow_quant_ops"
      ? (profile?.rows || []).filter((candidate) => candidate.artifact_precheck_status === "source_candidate_partial"
        && Number(candidate.artifact_facts?.quantized_nonconstant_tensor_count || 0) > 0).length
      : Number(profile?.source_candidate_after_artifact_precheck_count || 0);
    const applicable = expectedAffected > 0;
    if (!profile || !REQUIREMENT_IDS.has(row.id)
      || row.binding_status !== (applicable ? "pending_runtime_build_evidence" : "not_applicable_no_affected_source_candidates")
      || row.evidence_class !== (applicable ? "REQUIREMENT" : "NOT_APPLICABLE")
      || Number(row.affected_source_candidate_op_count) !== expectedAffected
      || !String(row.required_configuration || "").trim()) {
      throw new Error(`TFLite delegate build requirement is invalid for ${row?.id || "unknown"}.`);
    }
  }
}

function sourceRefMatches(value, path) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:"
      && url.hostname === "github.com"
      && url.pathname.includes(`/${TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit}/${path}`);
  } catch {
    return false;
  }
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}
