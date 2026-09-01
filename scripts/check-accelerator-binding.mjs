import assert from "node:assert/strict";

import {
  ACCELERATOR_BINDING_SCHEMA,
  buildAcceleratorBinding,
  buildCoreMlAcceleratorBinding,
  buildEdgeTpuAcceleratorBinding,
  buildLiteRtQualcommAcceleratorBinding,
  buildNvidiaHostAcceleratorBinding,
  buildOrtAcceleratorBindings,
  buildRuntimeAcceleratorBindings,
  buildTfliteAdditionalAcceleratorBindings,
  collectAcceleratorBindings,
  mergeAcceleratorBindings,
  validateAcceleratorBinding,
} from "../web/lib/accelerator-binding.js";
import { parseEdgeTpuCompilerEvidence } from "../web/lib/edgetpu-compiler-evidence.js";
import { parseLiteRtQualcommEvidence } from "../web/lib/litert-qualcomm-evidence.js";
import { tfliteAcceleratorSourceManifest } from "../web/lib/tflite-accelerator-source-profiles.js";

const sha = (value) => String(value).repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: sha("a"),
  ops: [{ index: 0, name: "CONV_2D" }, { index: 1, name: "RESHAPE" }],
  tensors: [],
};

const source = buildAcceleratorBinding({
  profile_id: "tflite_gpu",
  provider: "tensorflow",
  backend: "tflite_gpu",
  device_class: "gpu",
  binding_source: "source_rulepack",
  evidence_stage: "source_eligibility",
  artifact_sha256: analysis.model_sha256,
  source_rulepack_sha256: sha("b"),
  configuration: {},
  coverage: { candidate_operation_count: 1 },
  claims: { source_eligibility: true, selected_build: false, compiled_plan: false, observed_assignment: false, measured_execution: false },
  interpretation_boundary: "Source eligibility is not selected-build inclusion or runtime assignment.",
});
assert.equal(source.schema, ACCELERATOR_BINDING_SCHEMA);
assert.equal(validateAcceleratorBinding(source).binding_sha256, source.binding_sha256);
assert.throws(() => validateAcceleratorBinding({ ...source, evidence_stage: "observed_assignment" }), /claim|runtime trace|SHA-256/i);

const edgeSource = {
  schema: "deepbom.edgetpu_compiler_evidence.v1",
  artifact_sha256: analysis.model_sha256,
  compiler: { name: "edgetpu_compiler", version: "test", binary_sha256: sha("c") },
  invocation: { options: ["--delegate_search_step=1"] },
  compiled_artifact_sha256: sha("d"),
  compiler_report_sha256: sha("e"),
  operations: [
    { op_index: 0, op_name: "CONV_2D", mapping: "mapped" },
    { op_index: 1, op_name: "RESHAPE", mapping: "unmapped", reason: "compiler_reported_cpu" },
  ],
};
const edgeEvidence = parseEdgeTpuCompilerEvidence(edgeSource, analysis);
const edge = buildEdgeTpuAcceleratorBinding(analysis, edgeEvidence);
assert.equal(edge.evidence_stage, "compiled_plan");
assert.equal(edge.coverage.mapped_operation_count, 1);
assert.equal(edge.claims.observed_assignment, false);
assert.throws(() => parseEdgeTpuCompilerEvidence({ ...edgeSource, operations: edgeSource.operations.slice(0, 1) }, analysis), /every serialized graph operation/i);

const sourceBindings = buildTfliteAdditionalAcceleratorBindings(analysis);
assert.deepEqual(sourceBindings.map((row) => row.profile_id), ["tflite_coreml_delegate", "litert_qualcomm_qnn"]);
assert(sourceBindings.every((row) => row.evidence_stage === "source_eligibility"));
const qualcommSource = tfliteAcceleratorSourceManifest().profiles.find((row) => row.id === "litert_qualcomm_qnn");
const qualcommEvidence = parseLiteRtQualcommEvidence({
  schema: "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1",
  artifact_sha256: analysis.model_sha256,
  source: { litert_commit: qualcommSource.source.commit, rulepack_sha256: qualcommSource.rulepack_sha256 },
  compiler: { name: "fixture", version: "1", binary_sha256: sha("1") },
  invocation: { options: [] },
  compiled_plan_sha256: sha("2"),
  source_file_sha256: sha("3"),
  operations: [
    { op_index: 0, op_name: "CONV_2D", compile_status: "compiled" },
    { op_index: 1, op_name: "RESHAPE", compile_status: "not_compiled", reason: "fixture" },
  ],
  dispatch: { status: "not_observed" },
}, analysis);
const qualcommBinding = buildLiteRtQualcommAcceleratorBinding(analysis, qualcommEvidence);
assert.equal(qualcommBinding.evidence_stage, "compiled_plan");
assert.equal(qualcommBinding.provider, "qualcomm");
assert.equal(qualcommBinding.claims.observed_assignment, false);

const coreMlAnalysis = { model_sha256: analysis.model_sha256 };
const coreMl = buildCoreMlAcceleratorBinding(coreMlAnalysis, {
  schema: "deepbom.coreml_compute_plan.v1",
  artifact: { sha256: analysis.model_sha256 },
  normalized_manifest_sha256: sha("f"),
  runtime: {
    coremltools_compute_plan_source_sha256: sha("1"),
    compiled_model_content_sha256: sha("2"),
    macos_version: "15.0",
    os_build: "24A",
    hardware_model: "Mac",
  },
  configuration: { compute_units: "ALL", function_name: "main" },
  summary: { mapped_operation_count: 2, preferred_compute_device_counts: { GPU: 1, NEURAL_ENGINE: 1 }, unresolved_device_usage_count: 0 },
});
assert.equal(coreMl.claims.compiled_plan, true);
assert.equal(coreMl.claims.observed_assignment, false);

const nvidia = buildNvidiaHostAcceleratorBinding(analysis, {
  schema: "deepbom.accelerator_profile_binding.v1",
  binding_sha256: sha("3"),
  profile_sha256: sha("4"),
  selected_device: { index: 0, compute_capability: "8.9", driver_version: "1", memory_total_bytes: { decimal: "1", number: 1 } },
});
assert.equal(nvidia.evidence_stage, "serialized_artifact");
assert.equal(nvidia.claims.selected_build, false);

const runtime = buildRuntimeAcceleratorBindings(analysis, {
  artifact_sha256: analysis.model_sha256,
  runtime_binary_sha256: sha("5"),
  assignments: [
    { op_index: 0, provider: "QNNExecutionProvider" },
    { op_index: 1, provider: "QNNExecutionProvider", duration_sum_us: 4 },
  ],
});
assert.equal(runtime.length, 1);
assert.equal(runtime[0].evidence_stage, "measured_execution");
assert.equal(runtime[0].provider, "qualcomm");
const assignmentOnly = buildRuntimeAcceleratorBindings(analysis, {
  artifact_sha256: analysis.model_sha256,
  assignments: [{ op_index: 0, provider: "QNNExecutionProvider", duration_us: null }],
});
assert.equal(assignmentOnly[0].evidence_stage, "observed_assignment", "Null timing must not become a measured zero.");

const ortAnalysis = {
  format: "onnx",
  model_sha256: analysis.model_sha256,
  ops: analysis.ops,
  ort_compatibility_evidence: {
    assessment_status: "complete",
    source_commit: "8c546c37",
    execution_providers: [
      {
        execution_provider: "qnn",
        source_id: "ort_qnn",
        source_sha256: sha("6"),
        assessed_op_count: 2,
        source_candidate_after_artifact_precheck_count: 1,
        artifact_precheck_definite_fail_op_count: 1,
        artifact_precheck_unresolved_op_count: 0,
      },
      { execution_provider: "wasm_cpu", source_sha256: sha("7"), assessed_op_count: 2 },
    ],
  },
};
const ort = buildOrtAcceleratorBindings(ortAnalysis, {
  selected_build_provider_binding: {
    source_build_attestation: { attestation_sha256: sha("8") },
    supported_backends_sha256: sha("9"),
    bindings: [{ source_profile: "qnn", backend_name: "qnn", bundled: true }],
  },
});
assert.equal(ort.length, 2, "ORT accelerator source and selected-build stages should both be preserved.");
assert.deepEqual(ort.map((row) => row.evidence_stage), ["source_eligibility", "selected_build"]);
assert(ort.every((row) => row.provider === "qualcomm" && row.backend === "ort_qnn"));
assert.equal(ort.some((row) => row.profile_id.includes("wasm_cpu")), false, "CPU providers must not enter accelerator bindings.");
assert.deepEqual(
  collectAcceleratorBindings({ ...ortAnalysis, model_sha256: null }),
  [],
  "Source eligibility must not be emitted when the artifact identity is unbound.",
);
assert.equal(
  collectAcceleratorBindings({ ...ortAnalysis, model_sha256: null }, null, analysis.model_sha256).length,
  1,
  "A caller-bound artifact identity should enable the source-eligibility projection.",
);

const merged = mergeAcceleratorBindings(source, edge, coreMl, nvidia, runtime, qualcommBinding, source);
assert.equal(merged.length, 6);
assert.deepEqual([...merged].sort((a, b) => a.profile_id.localeCompare(b.profile_id)), merged);

console.log("Accelerator binding checks passed: lifecycle stages, Core ML, Edge TPU, NVIDIA host, ORT source/build adapters, and imported assignment boundaries.");
