import assert from "node:assert/strict";

import { deriveCurrentArtifactCapabilityRow } from "../web/lib/format-capability-view.js";
import {
  buildCoreMlComputePlanTemplate,
  parseCoreMlComputePlanDocument,
} from "../web/lib/coreml-compute-plan.js";
import { COREML_DEPLOYMENT_SOURCE } from "../web/lib/coreml-deployment-contract.js";
import { buildEngineeringEvidenceDocument, buildRuntimeEvidence } from "../web/lib/report-evidence.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";
import { readCoreMlPerChannelLinearAnalysis } from "./coreml-legacy-quantization-corpus-lib.mjs";

const SHA = "a".repeat(64);
const analysis = {
  format: "coreml",
  filename: "model.mlpackage",
  model_sha256: SHA,
  coreml: { model_type: "mlProgram", ml_program: { functions: { main: {} } }, description: { default_function_name: "main" } },
  ops: [
    { index: 0, name: "CONV", mil_operation_type: "conv", coreml_layer_name: "conv_out" },
    { index: 1, name: "RELU", mil_operation_type: "relu", coreml_layer_name: "relu_out" },
  ],
  tensor_inventory: { status: "assessed" },
  weight_integrity: { status: "assessed" },
};
const template = buildCoreMlComputePlanTemplate(analysis);
assert.equal(template.artifact.sha256, SHA);
assert.equal(template.structure.kind, "program");
const source = structuredClone(template);
source.runtime.coremltools_version = "9.0-test";
source.runtime.coremltools_compute_plan_source_sha256 = COREML_DEPLOYMENT_SOURCE.compute_plan_sha256;
source.runtime.compiled_model_content_sha256 = "b".repeat(64);
source.runtime.platform = "macOS 15 test";
source.runtime.architecture = "arm64";
source.runtime.platform_system = "Darwin";
source.runtime.macos_version = "15.6";
source.runtime.os_build = "24G84";
source.runtime.hardware_model = "Mac15,7";
source.runtime.python_version = "3.12.7";
source.runtime.available_compute_devices = [
  { type: "CPU", source_class: "MLCPUComputeDevice", instance_count: 1 },
  { type: "GPU", source_class: "MLGPUComputeDevice", instance_count: 1 },
  { type: "NEURAL_ENGINE", source_class: "MLNeuralEngineComputeDevice", instance_count: 1, total_core_count: 16 },
];
source.configuration.function_name = "main";
source.capture.capture_id = "compute-plan-test";
source.capture.collected_at = "2026-08-11T00:00:00.000Z";
source.capture.collector.source_sha256 = "d".repeat(64);
source.structure.rows = [
  { op_index: 0, operator_type: "conv", identity: "conv_out", preferred_compute_device: "NEURAL_ENGINE", supported_compute_devices: ["CPU", "NEURAL_ENGINE"], estimated_cost_weight: 0.8 },
  { op_index: 1, operator_type: "relu", identity: "relu_out", preferred_compute_device: "GPU", supported_compute_devices: ["CPU", "GPU"], estimated_cost_weight: 0.2 },
];

const parsed = parseCoreMlComputePlanDocument(source, analysis, { fileSha256: "c".repeat(64) });
assert.equal(parsed.runtime.pinned_source_alignment, "exact_pinned_compute_plan_source");
assert.equal(parsed.execution_status, "not_observed_compute_plan_only");
assert.equal(parsed.summary.estimated_cost_weight_sum, 1);
assert.deepEqual(parsed.summary.preferred_compute_device_counts, { NEURAL_ENGINE: 1, GPU: 1 });
assert.deepEqual(parsed.runtime.available_compute_devices.map((device) => device.type), ["CPU", "GPU", "NEURAL_ENGINE"]);
assert.match(parsed.normalized_manifest_sha256, /^[a-f0-9]{64}$/);

const runtime = buildRuntimeEvidence({ analysis, runtimeAssignmentEvidence: parsed });
assert.equal(runtime.runtime_assignment, null);
assert.equal(runtime.coreml_compute_plan.structure.operation_count, 2);
assert.equal(runtime.assessments.coreml_compute_plan.status, "assessed");
const markdown = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: parsed });
assert.match(markdown, /Core ML MLComputePlan Estimate/);
assert.match(markdown, /not_observed_compute_plan_only/);
assert.match(markdown, /NEURAL_ENGINE/);
const capability = deriveCurrentArtifactCapabilityRow("coreml", analysis, parsed);
assert.equal(capability.cells[4].id, "plan_bound");

assert.throws(() => parseCoreMlComputePlanDocument({ ...source, artifact: { ...source.artifact, sha256: "0".repeat(64) } }, analysis), /active artifact/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, structure: { ...source.structure, rows: source.structure.rows.map((row, index) => index ? row : { ...row, operator_type: "matmul" }) } }, analysis), /operator type mismatch/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, structure: { ...source.structure, rows: source.structure.rows.map((row, index) => index ? row : { ...row, identity: "wrong" }) } }, analysis), /identity mismatch/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, structure: { ...source.structure, rows: source.structure.rows.map((row, index) => index ? row : { ...row, preferred_compute_device: "GPU", supported_compute_devices: ["CPU"] }) } }, analysis), /not in the supported-device set/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, configuration: { ...source.configuration, function_name: "predict" } }, analysis), /does not match decoded function/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, runtime: { ...source.runtime, os_build: null } }, analysis), /macOS build/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, runtime: { ...source.runtime, available_compute_devices: source.runtime.available_compute_devices.filter((device) => device.type !== "GPU") } }, analysis), /absent from the captured host inventory/);
assert.throws(() => parseCoreMlComputePlanDocument({ ...source, capture: { ...source.capture, collector: { ...source.capture.collector, source_sha256: null } } }, analysis), /collector source SHA-256/);

const cpuOnly = structuredClone(source);
cpuOnly.configuration.compute_units = "CPU_ONLY";
cpuOnly.structure.rows = cpuOnly.structure.rows.map((row) => ({ ...row, preferred_compute_device: "CPU", supported_compute_devices: ["CPU"] }));
assert.equal(parseCoreMlComputePlanDocument(cpuOnly, analysis).configuration.compute_units, "CPU_ONLY");
assert.throws(() => parseCoreMlComputePlanDocument({ ...cpuOnly, structure: { ...cpuOnly.structure, rows: cpuOnly.structure.rows.map((row, index) => index ? row : { ...row, preferred_compute_device: "GPU", supported_compute_devices: ["CPU", "GPU"] }) } }, analysis), /violates compute-units CPU_ONLY/);

const completeAnalysis = await readCoreMlPerChannelLinearAnalysis();
const completeSource = buildCoreMlComputePlanTemplate(completeAnalysis);
completeSource.runtime = structuredClone(source.runtime);
completeSource.capture = structuredClone(source.capture);
completeSource.configuration.compute_units = "CPU_ONLY";
completeSource.structure.rows = completeAnalysis.ops.map((op, index) => ({
  op_index: index,
  operator_type: op.name,
  identity: op.coreml_layer_name,
  preferred_compute_device: "CPU",
  supported_compute_devices: ["CPU"],
  estimated_cost_weight: null,
}));
const completePlan = parseCoreMlComputePlanDocument(completeSource, completeAnalysis);
const identity = { filename: completeAnalysis.filename, format: "coreml", sha256: completeAnalysis.model_sha256 };
const evidence = buildEngineeringEvidenceDocument(completeAnalysis, {
  reportContext: { identity, generatedAt: "2026-08-11T00:00:00.000Z", runtimeEvidence: { runtimeAssignmentEvidence: completePlan } },
  rawEvidenceContext: { identity, generatedAt: "2026-08-11T00:00:00.000Z", runtimeEvidence: { runtimeAssignmentEvidence: completePlan } },
});
for (const id of ["CF-RUNTIME-CML-001", "CF-RUNTIME-CML-002"]) {
  assert.equal(evidence.evidence.conformance_report.checks.find((row) => row.id === id)?.status, "pass", `${id} must independently validate the compute-plan evidence and report projection.`);
}

console.log("Core ML compute-plan checks passed (compiled identity, static op binding, device/cost estimates, and execution boundary).");
