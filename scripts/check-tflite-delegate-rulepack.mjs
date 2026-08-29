import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { TFLITE_DELEGATE_RULEPACK_METADATA } from "../web/lib/tflite-delegate-rulepack-metadata.js";
import {
  applyProtectedTfliteDelegateCompatibilityEvidence,
  validateProtectedTfliteDelegateCompatibilityEvidence,
} from "../web/lib/tflite-delegate-compatibility.js";
import {
  bindTfliteDelegateRequirement,
  summarizeTfliteDelegateBuildBinding,
} from "../web/lib/tflite-build-configuration-binding.js";

const manifestPath = new URL("../reference/tflite-delegates/rule-manifest.json", import.meta.url);
const manifestPresent = existsSync(manifestPath);
const manifest = manifestPresent
  ? JSON.parse(await readFile(manifestPath, "utf8"))
  : {
      schema: TFLITE_DELEGATE_RULEPACK_METADATA.schema,
      manifest_sha256: TFLITE_DELEGATE_RULEPACK_METADATA.manifestSha256,
      tensorflow: { commit: TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit },
      sources: TFLITE_DELEGATE_RULEPACK_METADATA.sources,
      profiles: TFLITE_DELEGATE_RULEPACK_METADATA.profiles,
    };
const protectedRustPath = new URL("../protected/deepbom_wasm/src/tflite_delegate_rulepack_generated.rs", import.meta.url);
const protectedConsumerPath = new URL("../protected/deepbom_wasm/src/tflite_delegate_rulepack.rs", import.meta.url);
const protectedSourcesPresent = existsSync(protectedRustPath) && existsSync(protectedConsumerPath);
const rust = protectedSourcesPresent ? await readFile(protectedRustPath, "utf8") : "";
const rustConsumer = protectedSourcesPresent ? await readFile(protectedConsumerPath, "utf8") : "";
assert(manifest.schema === "deepbom.tflite_delegate_source_rulepack.v1", "Unexpected TFLite delegate rulepack schema.");
if (manifestPresent) {
  const hashBasis = { ...manifest };
  delete hashBasis.manifest_sha256;
  delete hashBasis.manifest_hash_basis;
  const reproducedHash = sha256(stableJson(hashBasis));
  assert(manifest.manifest_sha256 === reproducedHash, "TFLite delegate manifest canonical hash does not reproduce.");
}
assert(manifest.manifest_sha256 === TFLITE_DELEGATE_RULEPACK_METADATA.manifestSha256, "Web metadata manifest hash is stale.");
assert(manifest.tensorflow.commit === TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit, "Web metadata TensorFlow commit is stale.");
assert(profileRuleCount(manifest.profiles.find((row) => row.id === "tflite_gpu")) === 80, "GPU registration count is not 80.");
assert(profileRuleCount(manifest.profiles.find((row) => row.id === "tflite_nnapi")) === 94, "NNAPI registration count is not 94.");
if (manifestPresent) {
  assert(new Set(manifest.profiles.flatMap((profile) => profile.rules.map((rule) => `${profile.id}:${rule.op}`))).size === 174, "Delegate registration identities are not unique within profiles.");
}
for (const source of manifest.sources) {
  assert(/^[a-f0-9]{64}$/.test(source.sha256), `Source SHA-256 is invalid for ${source.id}.`);
  assert(source.source_ref.includes(manifest.tensorflow.commit), `Source reference is not commit-pinned for ${source.id}.`);
  if (protectedSourcesPresent) assert(`${rust}\n${rustConsumer}`.includes(source.sha256), `Protected Rust does not contain source digest ${source.id}.`);
}
if (protectedSourcesPresent) {
  assert(rust.includes(manifest.manifest_sha256) && rust.includes(manifest.tensorflow.commit), "Generated Rust rulepack identity is stale.");
}

const analysis = {
  format: "tflite",
  ops: [{ index: 0, name: "ADD", version: 2, inputs: [0, 1], outputs: [2] }],
  tensors: [],
};
const evidence = fixtureEvidence(analysis);
assert(validateProtectedTfliteDelegateCompatibilityEvidence(analysis, evidence), "Valid delegate evidence was rejected.");
const merged = structuredClone(analysis);
applyProtectedTfliteDelegateCompatibilityEvidence(merged, evidence);
assert(merged.tflite_delegate_compatibility_evidence?.rulepack_sha256 === manifest.manifest_sha256, "Validated delegate evidence was not attached.");

for (const mutate of [
  (value) => { value.rulepack_sha256 = "0".repeat(64); },
  (value) => { value.profiles[0].rows[0].op_version = 3; },
  (value) => { value.profiles[0].source_candidate_after_artifact_precheck_count = 0; },
  (value) => { value.profiles[1].rows[0].definite_exclusion_reasons = ["fabricated"]; },
  (value) => { value.build_requirements.pop(); },
  (value) => { value.source_files[0].sha256 = "f".repeat(64); },
]) {
  const invalid = structuredClone(evidence);
  mutate(invalid);
  assertThrows(() => validateProtectedTfliteDelegateCompatibilityEvidence(analysis, invalid));
}

const nonTflite = { format: "onnx", ops: [] };
const notApplicable = fixtureEvidence(nonTflite);
notApplicable.assessment_status = "not_applicable_non_tflite";
notApplicable.evidence_class = "NOT_APPLICABLE";
notApplicable.profiles = [];
notApplicable.build_requirements = [];
notApplicable.graph_op_count = 0;
assert(validateProtectedTfliteDelegateCompatibilityEvidence(nonTflite, notApplicable), "Non-TFLite not-applicable envelope was rejected.");

const selectedBuild = {
  tflite_delegate_build_inventory: {
    evidence_class: "DECLARED_BUILD_AND_RUNTIME_OPTION_INVENTORY",
    gpu: {
      compiled_status: "enabled_by_declared_cmake_option",
      experimental_flags: 1,
      quantized_model_flag_status: "enabled_by_declared_runtime_option",
    },
    nnapi: {
      compiled_status: "enabled_by_declared_cmake_option_and_android_gate",
      runtime_feature_level: 31,
      accelerator_identity: "android-nnapi-default-device",
      capability_source: "android_nnapi_runtime_query",
    },
  },
};
const selectedSummary = summarizeTfliteDelegateBuildBinding(selectedBuild, evidence.build_requirements);
assert(selectedSummary.applicable_requirement_count === 4, "Selected-build summary lost an applicable delegate requirement.");
assert(selectedSummary.satisfied_requirement_count === 3, "Selected-build summary must close only the quant option and two NNAPI requirements.");
assert(selectedSummary.pending_or_partial_count === 1 && selectedSummary.contradiction_count === 0,
  "GPU target-backend uncertainty must remain partial rather than support or contradiction.");
assert(!selectedSummary.all_applicable_requirements_bound,
  "TFLITE_ENABLE_GPU alone must not establish target-backend inclusion or runtime assignment.");
assert(selectedSummary.rows.find((row) => row.id === "tflite_gpu_delegate_compiled")?.binding.status
  === "partial_gpu_delegate_compiled_target_backend_unobserved", "GPU selected-build boundary was widened.");
assert(selectedSummary.rows.find((row) => row.id === "tflite_nnapi_feature_level")?.binding.status
  === "observed_runtime_capability_present", "Observed NNAPI capability query was not recognized.");

const declaredNnapi = structuredClone(selectedBuild);
declaredNnapi.tflite_delegate_build_inventory.nnapi.capability_source = "declared_capture_configuration";
assert(bindTfliteDelegateRequirement(declaredNnapi, evidence.build_requirements[3]).status
  === "partial_declared_runtime_capability_not_observed", "Declared NNAPI capability must not be promoted to an observation.");

const disabledBuild = structuredClone(selectedBuild);
disabledBuild.tflite_delegate_build_inventory.gpu.compiled_status = "disabled_by_declared_cmake_option";
disabledBuild.tflite_delegate_build_inventory.gpu.experimental_flags = 0;
disabledBuild.tflite_delegate_build_inventory.gpu.quantized_model_flag_status = "disabled_by_declared_runtime_option";
disabledBuild.tflite_delegate_build_inventory.nnapi.compiled_status = "disabled_by_non_android_cmake_gate";
const disabledSummary = summarizeTfliteDelegateBuildBinding(disabledBuild, evidence.build_requirements);
assert(disabledSummary.contradiction_count === 3,
  "Disabled GPU compile, GPU quant option, and NNAPI Android gate must remain three explicit contradictions.");

const noAffected = { ...evidence.build_requirements[0], affected_source_candidate_op_count: 0 };
assert(bindTfliteDelegateRequirement(selectedBuild, noAffected).status
  === "not_applicable_no_affected_source_candidates", "Zero-affected requirements must remain not applicable.");

console.log(`TFLite delegate rulepack checks passed: 174 source registrations, ${manifestPresent ? "canonical manifest reproduction" : "public metadata identity"}, strict browser merge, fail-closed mutations, selected-build requirement binding, and ${protectedSourcesPresent ? "private generated-source parity" : "an explicit public-source boundary"}.`);

function profileRuleCount(profile) {
  return profile?.rules?.length ?? profile?.registeredOpCount ?? null;
}

function fixtureEvidence(value) {
  const rows = TFLITE_DELEGATE_RULEPACK_METADATA.profiles.map((profile) => ({
    id: profile.id,
    label: profile.label,
    assessment_status: "assessed_artifact_visible_subset",
    evidence_class: "SOURCE_PINNED/DERIVED_PARTIAL",
    registered_source_op_count: profile.registeredOpCount,
    assessed_graph_op_count: value.ops.length,
    source_candidate_after_artifact_precheck_count: value.ops.length,
    definite_exclusion_count: 0,
    unresolved_candidate_count: value.ops.length,
    op_count_conservation_status: "complete",
    selected_build_status: "not_bound",
    runtime_assignment_status: "not_observed",
    rows: value.ops.map((op, position) => ({
      op_index: op.index ?? position,
      op_name: op.name,
      op_version: op.version ?? 1,
      source_registration_status: "registered",
      artifact_precheck_status: "source_candidate_partial",
      definite_exclusion_reasons: [],
      unresolved_predicates: ["selected_build"],
      source_ref: githubSource(profile.id),
      source_text_sha256: "a".repeat(64),
      source_maximum_op_version_candidates: [],
      source_definite_maximum_op_version: null,
      source_feature_level_tokens: [],
      artifact_facts: {
        input_tensor_count: op.inputs?.length || 0,
        output_tensor_count: op.outputs?.length || 0,
        nonconstant_tensor_count: 0,
        maximum_nonconstant_tensor_rank: null,
        nonconstant_dtypes: [],
        quantized_nonconstant_tensor_count: 1,
      },
    })),
  }));
  return {
    schema: TFLITE_DELEGATE_RULEPACK_METADATA.schema,
    assessment_status: "assessed_source_candidates_with_unresolved_runtime_predicates",
    evidence_class: "SOURCE_PINNED/DERIVED_PARTIAL",
    rulepack_sha256: TFLITE_DELEGATE_RULEPACK_METADATA.manifestSha256,
    tensorflow_source_commit: TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit,
    source_files: TFLITE_DELEGATE_RULEPACK_METADATA.sources.map((source) => ({
      id: source.id,
      source_ref: `https://github.com/tensorflow/tensorflow/blob/${TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit}/${source.path}`,
      sha256: source.sha256,
      scope: "pinned test source",
    })),
    graph_op_count: value.ops.length,
    profiles: rows,
    build_requirements: [
      ["tflite_gpu_delegate_compiled", "tflite_gpu"],
      ["tflite_gpu_allow_quant_ops", "tflite_gpu"],
      ["tflite_nnapi_delegate_compiled", "tflite_nnapi"],
      ["tflite_nnapi_feature_level", "tflite_nnapi"],
    ].map(([id, profile]) => ({
      id,
      profile,
      required_configuration: "test build requirement",
      binding_status: "pending_runtime_build_evidence",
      affected_source_candidate_op_count: value.ops.length,
      evidence_class: "REQUIREMENT",
    })),
    interpretation_boundary: "A source candidate is not selected-build availability, support, assignment, or runtime evidence.",
  };
}

function githubSource(profile) {
  const path = profile === "tflite_gpu"
    ? "tensorflow/lite/delegates/gpu/common/model_builder.cc"
    : "tensorflow/lite/delegates/nnapi/nnapi_delegate.cc";
  return `https://github.com/tensorflow/tensorflow/blob/${TFLITE_DELEGATE_RULEPACK_METADATA.tensorflowCommit}/${path}#L1-L2`;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertThrows(callback) {
  let threw = false;
  try {
    callback();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error("Invalid delegate evidence was accepted.");
}
