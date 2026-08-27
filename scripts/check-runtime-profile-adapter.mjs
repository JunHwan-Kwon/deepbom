import { createCheck } from "./check-assert.mjs";
import { readFileSync } from "node:fs";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import {
  buildOrtRuntimeAssignmentDocument,
  parseRuntimeProfileSource,
  previewOrtProfileMapping,
} from "../web/lib/runtime-profile-adapter.js";

const { done, expect, expectEqual, expectThrows } = createCheck("Runtime profile adapter check");
const analysis = {
  format: "onnx",
  model_sha256: "a".repeat(64),
  target_profile: { id: "wasm_simd", profile_sha256: "b".repeat(64) },
  tensors: [
    { index: 0, name: "x", dtype: "FLOAT32", shape: [1, 3], shape_declared: true, value_kind: "tensor" },
    { index: 1, name: "w", dtype: "FLOAT32", shape: [3, 4], shape_declared: true, value_kind: "tensor" },
    { index: 2, name: "y", dtype: "FLOAT32", shape: [1, 4], shape_declared: true, value_kind: "tensor" },
  ],
  ops: [
    { index: 0, name: "Conv", graph_node_name: "conv0", inputs: [0, 1], outputs: [2], input_names: ["x", "w"], output_names: ["y"], macs: 100, estimated_bytes: 40 },
    { index: 1, name: "Relu", graph_node_name: "", inputs: [], outputs: [], macs: 0, estimated_bytes: 20 },
    { index: 2, name: "Add", graph_node_name: "residual", inputs: [], outputs: [], macs: 20, estimated_bytes: 20 },
  ],
};
const events = [
  nodeEvent("conv0_kernel_time", 8, "Conv", "XnnpackExecutionProvider", 12, floatContract([[1, 3], [3, 4]], [[1, 4]])),
  nodeEvent("conv0_kernel_time", 8, "Conv", "XnnpackExecutionProvider", 14, floatContract([[1, 3], [3, 4]], [[1, 4]])),
  nodeEvent("Relu_1_kernel_time", 1, "Relu", "CPUExecutionProvider", 3),
  nodeEvent("Relu_1_kernel_time", 1, "Relu", "CPUExecutionProvider", 5),
  nodeEvent("fused_kernel_time", 11, "FusedConv", "XnnpackExecutionProvider", 10),
  { cat: "Session", name: "model_run", dur: 100, args: {} },
];
const parsedProfile = parseRuntimeProfileSource(JSON.stringify(events), analysis);
expectEqual(parsedProfile.kind, "onnxruntime_profile", "Raw ORT trace should be detected.");
expectEqual(parsedProfile.kernel_event_count, 5, "Only complete ORT node kernel events should enter mapping.");

const unknownOptimization = previewOrtProfileMapping(parsedProfile, analysis, {
  graphOptimizationLevel: "unknown",
  executionMode: "sequential",
});
expectEqual(unknownOptimization.assignment_count, 1, "Unknown optimization mode should permit only exact named-node mapping.");
expectEqual(unknownOptimization.duration_semantics, "per_original_op_exclusive", "A single sequential mapped op with equal sample counts has additive row semantics.");

const preview = previewOrtProfileMapping(parsedProfile, analysis, {
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
});
expectEqual(preview.assignment_count, 2, "Optimization-disabled mode should add a safe unnamed index/type mapping.");
expectEqual(preview.unresolved_runtime_node_count, 1, "Fused runtime nodes should remain unresolved.");
expectEqual(preview.assignments[0].mapping_method, "exact_graph_node_name_and_op_type", "Named ONNX nodes should map by exact name and op type.");
expectEqual(preview.assignments[0].duration_us, 13, "ORT event duration should be the arithmetic mean across profile events.");
expectEqual(preview.assignments[0].duration_sum_us, 26, "ORT duration sum should remain available for independent recomputation.");
expectEqual(preview.assignments[0].sample_count, 2, "ORT duration sample count should remain explicit.");
expectEqual(preview.assignments[1].mapping_method, "optimization_disabled_unnamed_node_index_and_op_type", "Unnamed nodes should require the strict optimization-disabled fallback.");
expectEqual(preview.assignments[0].delegated, true, "Non-CPU ORT providers should be classified as delegated/provider-assigned.");
expectEqual(preview.assignments[1].delegated, false, "CPUExecutionProvider should remain fallback CPU placement.");
expectEqual(preview.duration_semantics, "per_original_op_exclusive", "Sequential equal-count events should preserve additive per-op semantics.");
expectEqual(preview.runtime_tensor_observation_count, 1, "Repeated identical ORT type-shape rows should become one consistent mapped observation.");
expectEqual(preview.runtime_tensor_observation_conflict_count, 0, "Identical repeated ORT type-shape rows should not conflict.");

const document = buildOrtRuntimeAssignmentDocument(parsedProfile, analysis, {
  runtimeVersion: "1.26.0",
  runtimeBuild: "release; graph_optimization_level=disabled; execution_mode=sequential",
  backend: "XnnpackExecutionProvider + CPUExecutionProvider",
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
  collectedAt: "2026-07-16T00:00:00.000Z",
  profileSha256: "c".repeat(64),
});
const assignment = parseRuntimeAssignmentDocument(JSON.stringify(document), analysis);
expectEqual(assignment.schema, "deepbom.runtime_assignment.v1.9", "Adapted evidence should normalize to runtime assignment schema v1.9.");
expectEqual(assignment.source.adapter.schema, "deepbom.ort_profile_adapter.v2.2", "Adapted evidence should preserve the internal type-shape adapter schema.");
expectEqual(assignment.source.adapter.runtime_tensor_observation_count, 1, "Canonical parsing should retain validated ORT tensor observations.");
expectEqual(assignment.source.adapter.runtime_tensor_observations[0].output_size_bytes, 16, "ORT output size should retain its exact safe-integer mirror.");
expectEqual(assignment.source.profile_sha256, "c".repeat(64), "Adapted evidence should bind the raw profile digest.");
expectEqual(assignment.source.adapter.mapped_kernel_event_count, 4, "Validated adapter provenance should preserve mapped event count.");
expectEqual(assignment.source.adapter.mapping_coverage_ratio, 2 / 3, "Validated adapter coverage should be recomputed from assignment rows.");
expectEqual(assignment.runtime.graph_optimization_level, "disabled", "Runtime optimization mode should remain in evidence.");
expectEqual(assignment.runtime.execution_mode, "sequential", "Runtime execution mode should remain in evidence.");
expectEqual(assignment.assignments[0].kernel, null, "ORT node events must not be mislabeled as executed microkernel symbols.");
expectEqual(assignment.assignments[0].partition_id, null, "ORT node events must not fabricate provider partition IDs.");

const unequalEvents = [...events, nodeEvent("conv0_kernel_time", 8, "Conv", "XnnpackExecutionProvider", 16)];
const unequal = previewOrtProfileMapping(parseRuntimeProfileSource(JSON.stringify(unequalEvents), analysis), analysis, {
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
});
expectEqual(unequal.duration_semantics, "unspecified", "Unequal per-op sample counts must disable additive duration semantics.");
const parallel = previewOrtProfileMapping(parsedProfile, analysis, {
  graphOptimizationLevel: "disabled",
  executionMode: "parallel",
});
expectEqual(parallel.duration_semantics, "unspecified", "Parallel execution must disable additive duration semantics.");

const conflictEvents = [...events, nodeEvent("conv0_kernel_time", 8, "Conv", "CPUExecutionProvider", 9)];
const conflict = previewOrtProfileMapping(parseRuntimeProfileSource(JSON.stringify(conflictEvents), analysis), analysis, {
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
});
expectEqual(conflict.assignment_count, 1, "Conflicting provider identities for one original op must be excluded.");
expectEqual(conflict.conflict_count, 1, "Provider conflicts should remain explicit adapter diagnostics.");

const shapeConflictEvents = [...events, nodeEvent("conv0_kernel_time", 8, "Conv", "XnnpackExecutionProvider", 9, floatContract([[2, 3], [3, 4]], [[2, 4]]))];
const shapeConflict = previewOrtProfileMapping(parseRuntimeProfileSource(JSON.stringify(shapeConflictEvents), analysis), analysis, {
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
});
expectEqual(shapeConflict.runtime_tensor_observation_conflict_count, 1, "Multiple executed shapes for one runtime node must remain an explicit conflict.");
expectThrows(() => parseRuntimeProfileSource(JSON.stringify([
  nodeEvent("conv0_kernel_time", 8, "Conv", "CPUExecutionProvider", 1, { input_type_shape: [{ float: [1, 3] }] }),
]), analysis), "incomplete input/output type-shape pair", "A partial ORT type-shape pair must fail closed.");
expectThrows(() => parseRuntimeProfileSource(JSON.stringify([
  nodeEvent("conv0_kernel_time", 8, "Conv", "CPUExecutionProvider", 1, { input_type_shape: [{ float: [-1, 3] }], output_type_shape: [{ float: [1, 4] }] }),
]), analysis), "invalid concrete tensor shape", "A negative executed dimension must fail closed.");

const tamperedMean = structuredClone(document);
tamperedMean.assignments[0].duration_us = 99;
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(tamperedMean), analysis), "duration_us must equal", "Canonical parser should reject a tampered duration mean.");
const tamperedMapping = structuredClone(document);
tamperedMapping.assignments[0].runtime_node_name = "other";
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(tamperedMapping), analysis), "does not match", "Canonical parser should reject a tampered graph-node mapping.");
const tamperedCount = structuredClone(document);
tamperedCount.source.adapter.mapped_kernel_event_count += 1;
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(tamperedCount), analysis), "mapped event count", "Canonical parser should reject a tampered mapped-event aggregate.");
expectThrows(() => parseRuntimeProfileSource("[]", analysis), "no valid Node", "Empty traces should fail closed.");
expectThrows(() => parseRuntimeProfileSource(JSON.stringify(events), { ...analysis, format: "tflite" }), "active ONNX", "ORT traces must not cross the artifact-format boundary.");
expect(isFinite(preview.mapping_coverage_ratio), "Mapping coverage should remain finite.");

const sampleAnalysis = analyzeOnnxModel(
  new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx")),
  "sample_cnn_float.onnx",
  { id: "wasm_simd", label: "WebAssembly SIMD", profile_sha256: "d".repeat(64), l1_data_bytes: 32768 },
);
sampleAnalysis.model_sha256 = "e".repeat(64);
const sampleEvents = sampleAnalysis.ops.flatMap((op) => [
  nodeEvent(`${op.name}_${op.index}_kernel_time`, op.index, op.name, op.index % 2 ? "CPUExecutionProvider" : "XnnpackExecutionProvider", op.index + 1),
  nodeEvent(`${op.name}_${op.index}_kernel_time`, op.index, op.name, op.index % 2 ? "CPUExecutionProvider" : "XnnpackExecutionProvider", op.index + 3),
]);
const sampleProfile = parseRuntimeProfileSource(JSON.stringify(sampleEvents), sampleAnalysis);
const sampleDocument = buildOrtRuntimeAssignmentDocument(sampleProfile, sampleAnalysis, {
  runtimeVersion: "1.26.0",
  runtimeBuild: "release; graph_optimization_level=disabled; execution_mode=sequential",
  graphOptimizationLevel: "disabled",
  executionMode: "sequential",
  collectedAt: "2026-07-16T01:00:00.000Z",
  profileSha256: "f".repeat(64),
});
const sampleAssignment = parseRuntimeAssignmentDocument(JSON.stringify(sampleDocument), sampleAnalysis);
const runtimeContext = { runtimeAssignmentEvidence: sampleAssignment };
const reportContext = {
  identity: { filename: sampleAnalysis.filename, format: "onnx", hash: sampleAnalysis.model_sha256 },
  runtimeEvidence: runtimeContext,
  generatedAt: "2026-07-16T02:00:00.000Z",
};
const sampleMlBom = buildMlBomDocument(sampleAnalysis, { hash: sampleAnalysis.model_sha256 });
const sampleBundle = buildEngineeringBundleArtifactFiles(sampleAnalysis, {
  reportContext,
  rawEvidenceContext: { identity: reportContext.identity, runtimeEvidence: runtimeContext },
  mlBomDocument: sampleMlBom,
});
const sampleReport = sampleBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const sampleEvidence = JSON.parse(sampleBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
const sampleComparison = sampleEvidence.evidence?.runtime_results?.runtime_assignment?.comparison || {};
expectEqual(sampleEvidence.evidence?.conformance_report?.status, "pass", "Adapted ORT evidence should pass independent report conformance on the reference ONNX artifact.");
expectEqual(sampleComparison.prediction_applicability, "not_applicable_for_onnx_execution_provider_assignment", "ONNX provider observations must not be compared with TFLite XNNPACK prediction.");
expectEqual(sampleComparison.placement_assessment?.observed_assignment_count, 9, "ORT evidence should report original-op provider coverage independently of prediction coverage.");
expectEqual(sampleComparison.placement_assessment?.assessed_op_count, 0, "ONNX static provider prediction must remain not applicable.");
expectEqual(sampleComparison.placement_assessment?.mismatch_count, 0, "ONNX provider rows must not become false static-placement mismatches.");
expectEqual(sampleComparison.predicted_boundary_inventory?.status, "not_applicable", "ONNX must suppress the TFLite predicted-boundary inventory.");
expectEqual(sampleComparison.observed_partitions?.partition_count, 0, "ORT node events without partition IDs must not fabricate runtime partitions.");
expectEqual(sampleComparison.observed_partitions?.provider_segment_count, 9, "Contiguous provider segments should remain available without being mislabeled as partitions.");
expectEqual(sampleComparison.mac_comparison?.mismatch_macs, null, "MAC mismatch must remain null when static prediction is not applicable.");
expectEqual(sampleComparison.duration_comparison?.mismatch_duration_us, null, "Duration mismatch must remain null when static prediction is not applicable.");
expect(sampleReport.includes(`profile SHA-256 ${"f".repeat(64)}`), "Engineering report should disclose the raw ORT profile digest.");
expect(sampleReport.includes("mapped 18/18 kernel event(s)"), "Engineering report should disclose ORT profile event mapping coverage.");
expect(sampleReport.includes("microsoft/onnxruntime@8c546c37b43caaca1fa25db430dab94b901cf277"), "Engineering report should disclose the pinned ORT event parser basis.");
expect(sampleReport.includes("runtime version/build/preparation are DECLARED; provider/node/duration rows are OBSERVED_RUNTIME"), "Engineering report should separate declared runtime metadata from observed profile rows.");
expect(sampleReport.includes("### Observed Runtime Provider Assignment"), "ONNX report should use a provider-evidence heading instead of prediction-agreement wording.");
expect(!sampleReport.includes("underpredicted delegation"), "ONNX report must not emit false TFLite delegation mismatch language.");

done("Runtime profile adapter check passed (strict ORT identity mapping, timing semantics, digest/provenance, and tamper rejection). ");

function nodeEvent(name, nodeIndex, opName, provider, durationUs, contract = null) {
  return {
    cat: "Node",
    name,
    dur: durationUs,
    args: { node_index: String(nodeIndex), op_name: opName, provider, ...(contract || {}) },
  };
}

function floatContract(inputs, outputs) {
  const outputBytes = outputs.reduce((sum, shape) => sum + shape.reduce((product, dim) => product * dim, 1) * 4, 0);
  return {
    input_type_shape: inputs.map((shape) => ({ float: shape })),
    output_type_shape: outputs.map((shape) => ({ float: shape })),
    activation_size: "12",
    parameter_size: "48",
    output_size: String(outputBytes),
  };
}
