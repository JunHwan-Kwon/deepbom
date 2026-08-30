import assert from "node:assert/strict";

import { buildRuntimeEvidence } from "../web/lib/report-evidence.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";
import {
  RUNTIME_EVIDENCE_SIDECAR_SCHEMA,
  buildRuntimeEvidenceSidecar,
  verifyRuntimeEvidenceSidecar,
} from "../web/lib/runtime-evidence-sidecar.js";

const artifactSha256 = "a".repeat(64);
const targetSha256 = "b".repeat(64);
const binarySha256 = "c".repeat(64);
const sourceCommit = "d".repeat(40);
const analysis = {
  format: "tflite",
  filename: "model.tflite",
  model_sha256: artifactSha256,
  ops: [{ index: 0 }, { index: 1 }],
  target_profile: { id: "android_mid_a55", profile_sha256: targetSha256 },
};
const assignment = {
  schema: "deepbom.runtime_assignment.v1.10",
  artifact_sha256: artifactSha256,
  target_profile_id: "android_mid_a55",
  target_profile_sha256: targetSha256,
  graph_op_count: 2,
  runtime: { name: "TensorFlow Lite", version: "2.21.0", backend: "XNNPACK", binary_sha256: binarySha256 },
  source: { kind: "deepbom_native_runtime_capture", collected_at: "2026-08-18T00:00:00.000Z", capture_id: "capture-1", collector: { name: "deepbom-runtime-collector", version: "1.1" } },
  selector_context: {
    device: { architecture: "aarch64", cpu_features: ["neon"] },
    build: { xnnpack_source_commit: sourceCommit, runtime_binary_sha256: binarySha256, microkernel_build_identifier_sha256: "e".repeat(64) },
  },
  selector_observation: {
    collector_attestation_status: "not_attested",
    lowering_observed_op_count: 0,
    microkernel_observed_op_count: 0,
    selector_ambiguity_closed_op_count: 0,
    graph_op_count: 2,
    status: "partial",
  },
  assignments: [
    { op_index: 0, duration_us: 10 },
    { op_index: 1, duration_us: 20 },
  ],
  runtime_memory: { status: "assessed", snapshot_count: 2 },
};

const normalized = buildRuntimeEvidenceSidecar(analysis, assignment);
assert.equal(normalized.schema, RUNTIME_EVIDENCE_SIDECAR_SCHEMA);
assert.equal(normalized.artifact.sha256, artifactSha256);
assert.equal(normalized.runtime.family, "tensorflow_lite");
assert.equal(normalized.build.status, "source_and_binary_bound");
assert.equal(normalized.observations.placement.status, "complete");
assert.equal(normalized.observations.timing.status, "complete");
assert.equal(normalized.observations.memory.status, "observed");
assert.equal(verifyRuntimeEvidenceSidecar(normalized, analysis, assignment), normalized);

const tampered = structuredClone(normalized);
tampered.observations.placement.assessed_count = 1;
assert.throws(() => verifyRuntimeEvidenceSidecar(tampered, analysis, assignment), /SHA-256|reconstruct/);
assert.throws(() => buildRuntimeEvidenceSidecar({ ...analysis, model_sha256: "f".repeat(64) }, assignment), /does not match/);

const runtime = buildRuntimeEvidence({ analysis, runtimeAssignmentEvidence: assignment });
assert.equal(runtime.runtime_evidence_sidecar.sidecar_sha256, normalized.sidecar_sha256);
assert.equal(runtime.assessments.runtime_evidence_sidecar.status, "assessed");
const report = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: assignment }, [], analysis);
assert.match(report, /Cross-format runtime sidecar/);
assert.match(report, new RegExp(normalized.sidecar_sha256));

const onnx = buildRuntimeEvidenceSidecar({ ...analysis, format: "onnx" }, {
  ...assignment,
  runtime: { ...assignment.runtime, name: "ONNX Runtime", backend: "CPUExecutionProvider" },
  source: { ...assignment.source, adapter: { schema: "deepbom.ort_profile_adapter.v2.1", source_commit: `microsoft/onnxruntime@${sourceCommit}` } },
});
assert.equal(onnx.runtime.family, "onnxruntime");

const gguf = buildRuntimeEvidenceSidecar({ format: "gguf", model_sha256: artifactSha256 }, {
  schema: "deepbom.gguf_runtime_environment.v2",
  artifact: { sha256: artifactSha256 },
  runtime: { repository: "https://github.com/ggml-org/llama.cpp", source_commit: sourceCommit, binary_sha256: binarySha256, version_output: "llama-cli" },
  build: { cmake_cache_sha256: "e".repeat(64) },
  selection: { requested_backend_profile_id: "cpu" },
  capture: { capture_id: "gguf-1", collected_at: "2026-08-18T00:00:00.000Z", collector: { name: "deepbom-gguf-runtime-collector", version: "2" } },
  observations: { model_load_status: "observed_success", elapsed_ms: 42 },
  compute_graph: { graph_count: 1, scheduled_node_count: 3, dispatched_graph_count: 1 },
});
assert.equal(gguf.runtime.family, "llama_cpp");
assert.equal(gguf.observations.execution.status, "observed_dispatch");

const coreml = buildRuntimeEvidenceSidecar({ format: "coreml", model_sha256: artifactSha256 }, {
  schema: "deepbom.coreml_compute_plan.v1",
  artifact: { sha256: artifactSha256 },
  runtime: {
    compiled_model_content_sha256: binarySha256,
    coremltools_version: "9.0",
    coremltools_compute_plan_source_sha256: "e".repeat(64),
    platform_system: "Darwin",
    macos_version: "15.6",
    os_build: "24G84",
    hardware_model: "Mac15,7",
    architecture: "arm64",
    available_compute_devices: [{ type: "CPU", source_class: "MLCPUComputeDevice", instance_count: 1 }],
  },
  configuration: { compute_units: "ALL" },
  capture: { capture_id: "coreml-1", collected_at: "2026-08-18T00:00:00.000Z", collector: { name: "deepbom-coreml-plan", version: "2", source_sha256: "f".repeat(64) } },
  structure: { rows: [{ estimated_cost_weight: 1 }] },
  execution_status: "not_observed_compute_plan_only",
});
assert.equal(coreml.runtime.family, "coreml");
assert.equal(coreml.observations.placement.evidence_class, "RUNTIME_PLAN_ESTIMATE");
assert.equal(coreml.observations.execution.status, "not_observed");
assert.equal(coreml.capture.host.os_build, "24G84");
assert.equal(coreml.capture.collector_source_sha256, "f".repeat(64));
assert.throws(() => buildRuntimeEvidenceSidecar({ format: "coreml", model_sha256: artifactSha256 }, {
  schema: "deepbom.coreml_compute_plan.v1", artifact: { sha256: artifactSha256 }, runtime: { compiled_model_content_sha256: binarySha256 },
}), /estimate-only operation rows|compute-plan source/);

assert.equal(buildRuntimeEvidenceSidecar(analysis, null), null);
console.log("Runtime evidence sidecar check passed (TFLite, ONNX, GGUF, Core ML normalization, hash binding, report, and tamper rejection).");
