import {
  buildModelIdentity,
  detectModelFormat,
  estimateModelAnalysis,
  FULL_AUDIT_ESTIMATE_METHOD,
  formatEstimate,
  formatMeasuredAudit,
  inferModelFormat,
  modelFormatGate,
  modelSupportsCapability,
  stagedModelCopy,
} from "../web/lib/model-file.js";
import { formatAuditButtonLabel, formatEvidenceScope, formatWorkflowApplicability } from "../web/lib/format-evidence-scope.js";
import { readFileSync } from "node:fs";
import { latencyStats } from "../web/lib/format.js";
import { p99EvidenceForSampleCount } from "../web/lib/benchmark-ui.js";
import { buildBenchmarkTelemetryPayload, buildStructureTelemetryPayload } from "../web/lib/telemetry.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildStaticAuditMarkdown } from "../web/lib/markdown-report.js";
import { buildGraphEvidenceMaps } from "../web/lib/graph-ui.js";
import { quantSummaryEvidence } from "../web/lib/performance-visuals.js";
import {
  buildCanonicalPackageDigest,
  buildEngineeringBundleArtifactFiles,
  buildEngineeringBundleManifest,
  buildFindingsRegister,
  buildMlBomDocument,
} from "../web/lib/report.js";
import { initSync, analyze_tflite_for_target, compute_deployment_frontier, compute_static_runtime_calibration } from "../pkg/tflite_wasm_audit.js";
import { initSync as initDeepBom, analyze_deepbom } from "../web/protected/deepbom/pkg/deepbom_wasm.js";
import { applyProtectedXnnpackSelectorEvidence } from "../web/lib/xnnpack-selector-evidence.js";
import { applyProtectedOrtCompatibilityEvidence } from "../web/lib/ort-compatibility-evidence.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Model file contract check");

const tfliteFile = { name: "edge_model.tflite", size: 1024 * 1024 };
const onnxFile = { name: "edge_model.onnx", size: 24 * 1024 * 1024 };
const unknownTfliteHeader = new TextEncoder().encode("xxxxTFL3yyyy");
const unknownOnnxHeader = new TextEncoder().encode("ir_version: 9\nproducer_name: onnx");
const safeTensorsHeader = new Uint8Array(10);
new DataView(safeTensorsHeader.buffer).setUint32(0, 2, true);
safeTensorsHeader[8] = 0x7b;

for (const [actual, expected, label] of [
  [inferModelFormat(tfliteFile, null), "TFLite", "infer tflite extension"],
  [inferModelFormat(onnxFile, null), "ONNX", "infer onnx extension"],
  [inferModelFormat({ name: "artifact.bin", size: 1 }, unknownTfliteHeader), "TFLite", "infer TFL3 header"],
  [inferModelFormat({ name: "artifact.bin", size: 1 }, unknownOnnxHeader), "Unsupported", "text labels must not be treated as ONNX protobuf"],
  [detectModelFormat("model.onnx", new Uint8Array()), "onnx", "detect onnx filename"],
  [detectModelFormat("model.tflite", new Uint8Array()), "tflite", "detect tflite filename"],
  [detectModelFormat("model.bin", new Uint8Array([0x08, 0x01, 0x12, 0x03])), "onnx", "detect onnx header"],
  [detectModelFormat("model.gguf", new Uint8Array()), "gguf", "detect GGUF filename"],
  [detectModelFormat("model.bin", new Uint8Array([0x47, 0x47, 0x55, 0x46])), "gguf", "detect GGUF magic"],
  [detectModelFormat("model.safetensors", new Uint8Array()), "safetensors", "detect SafeTensors filename"],
  [detectModelFormat("model.bin", safeTensorsHeader), "safetensors", "detect SafeTensors header"],
  [detectModelFormat("model.mlmodel", new Uint8Array()), "coreml", "detect Core ML filename"],
  [detectModelFormat("model.pte", new Uint8Array()), "executorch", "detect ExecuTorch filename"],
  [detectModelFormat("model.bin", new Uint8Array([0, 0, 0, 0, 0x45, 0x54, 0x31, 0x32])), "executorch", "detect ExecuTorch ET12 header"],
  [detectModelFormat("model.bin", new Uint8Array([0, 0, 0, 0, 0x46, 0x54, 0x30, 0x31])), "executorch", "detect ExecuTorch FT01 header"],
  [detectModelFormat("model.pth", new Uint8Array()), "pytorch_pickle", "detect rejected PyTorch pickle filename"],
  [detectModelFormat("model.bin", new Uint8Array([1, 2, 3, 4])), "unsupported", "unknown format must fail closed"],
]) {
  expectEqual(actual, expected, label);
}
expectThrows(
  () => analyzeOnnxModel(new Uint8Array([0x0a, 0x05, 0x01]), "malformed.onnx"),
  "exceeds message bounds",
  "Malformed ONNX length-delimited fields should fail closed.",
);
const sharedComputeDenominatorReport = buildStaticAuditMarkdown({
  filename: "compute-denominator.tflite",
  format: "tflite",
  operator_count: 3,
  tensor_count: 4,
  total_macs: 1_000,
  ops: [
    { index: 0, name: "CONV_2D", xnnpack_chain_id: 0 },
    { index: 1, name: "TRANSPOSE_CONV", xnnpack_chain_id: 0 },
    { index: 2, name: "ADD", xnnpack_chain_id: 0 },
  ],
  stages: [],
  quantization_status: {
    compute_ops: 2,
    quantized_compute_ops: 1,
    compute_macs: 100,
    quantized_compute_macs: 64,
    op_state_counts: [{ name: "quantized_compute", count: 1 }],
  },
});
expect(sharedComputeDenominatorReport.includes("Quantized MAC-bearing compute ops: 1/2 (50.0%)"), "Markdown compute-op count must use the shared analyzer denominator.");
expect(sharedComputeDenominatorReport.includes("Quantized compute MACs: 64 / 100 (64.0%)"), "Markdown compute-MAC ratio must use the analyzer-emitted compute-only denominator rather than total graph MACs or a local op-name list.");
expect(modelSupportsCapability("tflite", "experimental_tflite_research"), "TFLite research capability must be explicit");
expect(modelSupportsCapability("onnx", "runtime_execution"), "ONNX runtime capability must be explicit");
expect(!modelSupportsCapability("coreml", "runtime_execution"), "Core ML metadata adapter must not imply runtime execution");
expect(modelSupportsCapability("coreml", "serialized_graph_analysis"), "Core ML legacy serialized-graph analysis capability must be explicit");
expect(modelSupportsCapability("coreml", "weight_encoding_analysis"), "Core ML WeightParams encoding analysis capability must be explicit");
expect(modelSupportsCapability("executorch", "static_analysis"), "ExecuTorch ET12/FT01 static analysis capability must be explicit");
expect(modelSupportsCapability("executorch", "serialized_graph_analysis"), "ExecuTorch ET12 serialized execution-plan capability must be explicit");
expectEqual(formatEvidenceScope("gguf").depth, "Container and tensor-payload audit", "GGUF UI scope should describe container evidence without claiming a graph audit.");
expect(formatEvidenceScope("safetensors").runtimeBoundary.includes("does not serialize the executable forward graph"), "SafeTensors scope should mark graph/runtime evidence outside the checkpoint-plus-configuration contract.");
const placementSha = "a".repeat(64);
const ggufPlacementAnalysis = { format: "gguf", model_sha256: placementSha, tensor_count: 1, gguf: { backend_compatibility: { status: "source_candidate" } } };
const ggufConfiguration = {
  schema: "deepbom.gguf_runtime_environment.v2",
  artifact: { sha256: placementSha },
  selection: { requested_backend_label: "CPU", gpu_layers: 0, context_size: 128, batch_size: 1 },
  compute_graph: { graph_count: 0, graphs: [] },
};
expectEqual(formatEvidenceScope("gguf", { analysis: ggufPlacementAnalysis, runtimeEvidence: ggufConfiguration }).runtimeObserved, false, "A bound GGUF build/configuration must not be promoted to observed execution without a scheduler graph.");
expectEqual(formatEvidenceScope("gguf", { analysis: ggufPlacementAnalysis, runtimeEvidence: {
  ...ggufConfiguration,
  compute_graph: { graph_count: 1, split_count: 1, successful_dispatch_count: 1, dispatch_count: 1, graphs: [{ scheduled_nodes: [{ scheduled_index: 0, backend: "CPU" }] }] },
} }).runtimeObserved, true, "Backend-assigned GGUF scheduler nodes should establish captured runtime placement.");
const coreMlAnalysis = { format: "coreml", model_sha256: placementSha, ops: [{ index: 0, name: "conv" }], coreml: { deployment_floor: { status: "assessed" } } };
const coreMlPlanScope = formatEvidenceScope("coreml", { analysis: coreMlAnalysis, runtimeEvidence: {
  schema: "deepbom.coreml_compute_plan.v1",
  artifact: { sha256: placementSha },
  configuration: { compute_units: "ALL" },
  structure: { rows: [{ op_index: 0, preferred_compute_device: "CPU" }] },
  summary: { preferred_compute_device_counts: { CPU: 1 } },
} });
expectEqual(coreMlPlanScope.placementEstimateBound, true, "An imported Core ML compute plan should be represented as a bound placement estimate.");
expectEqual(coreMlPlanScope.runtimeObserved, false, "MLComputePlan must not be promoted to observed execution.");
expectEqual(formatEvidenceScope("coreml", { runtimeEvidence: { schema: "deepbom.coreml_compute_plan.v1" } }).placementEstimateBound, false, "A compute plan without an active artifact must not be described as bound.");
expectEqual(formatAuditButtonLabel("coreml"), "Audit Core ML Artifact", "Core ML should have a format-specific audit action.");
expectEqual(formatAuditButtonLabel("executorch"), "Audit ExecuTorch Artifact", "ExecuTorch should have a format-specific audit action.");
expect(!formatWorkflowApplicability("gguf").graph && !formatWorkflowApplicability("gguf").runtime, "GGUF should not expose graph or browser-runtime workflow tabs.");
expect(formatWorkflowApplicability("coreml", { ops: [{ index: 0 }] }).graph, "Core ML should expose Graph only when a serialized graph was decoded.");
expect(formatWorkflowApplicability("executorch", { ops: [{ index: 0 }] }).graph, "ExecuTorch should expose Graph when ET12 instructions were decoded.");
const stagedSafeTensors = stagedModelCopy(
  { name: "weights.safetensors", size: 1024 },
  { format: "SafeTensors", formatId: "safetensors", timingSampleCount: 0, lowMs: 100, highMs: 300 },
  "unused",
);
expect(stagedSafeTensors.meta.includes("checkpoint inventory") && !stagedSafeTensors.meta.includes("metadata-only"), "SafeTensors staging copy should state the actual audit depth.");
expect(!modelSupportsCapability("safetensors", "experimental_tflite_research"), "SafeTensors metadata adapter must not imply TFLite research support");
expectEqual(modelFormatGate("tflite").blocked, false, "TFLite should pass the production format gate.");
expectEqual(modelFormatGate("pytorch_pickle").reason, "unsafe_serialized_code", "Executable pickle formats must fail the production gate with a stable reason.");
expectEqual(modelFormatGate("executorch").blocked, false, "ExecuTorch ET12/FT01 should pass the production format gate.");
expectEqual(modelFormatGate("unsupported").reason, "unsupported_format", "Unknown formats must fail the production gate explicitly.");

const sampleOnnx = analyzeOnnxModel(new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx")), "sample_cnn_float.onnx");
const mnistOnnx = analyzeOnnxModel(new Uint8Array(readFileSync("web/samples/mnist-8.onnx")), "mnist-8.onnx");
expectEqual(mnistOnnx.weight_integrity.constant_tensors_scanned, 8, "ONNX constant integrity should retain every decoded initializer.");
expectEqual(mnistOnnx.weight_integrity.weight_tensors_scanned, 2, "ONNX weight statistics should include only confirmed learned parameters.");
expectEqual(mnistOnnx.weight_integrity.constant_role_counts.control_constant, 2, "ONNX Reshape shape tensors should be classified as control constants.");
expect(Number(mnistOnnx.weight_integrity.max_abs_weight) < 2, "ONNX max_abs_weight must exclude 256-valued Reshape control tensors.");
expect(mnistOnnx.weight_integrity.tensor_results.some((row) => row.constant_role === "control_constant" && row.max_abs_value === 256), "ONNX control constants should remain independently inspectable.");
const scopedOnnxTarget = analyzeOnnxModel(new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx")), "sample_cnn_float.onnx", {
  id: "android_mid_a55", label: "Android mid", profile_sha256: "a".repeat(64), l1_data_bytes: 65536,
  architecture: "AArch64", effective_peak_gops: 5600, xnnpack_kernel_family: "NEON",
});
expectEqual(Object.keys(scopedOnnxTarget.target_profile).sort().join(","), "applicability,id,l1_data_bytes,label,profile_sha256", "ONNX analysis must retain only the target fields used by its static L1 contract.");
expect(!Object.hasOwn(scopedOnnxTarget.target_profile, "xnnpack_kernel_family"), "TFLite XNNPACK target assumptions must not enter ONNX evidence.");
const sampleGemm = sampleOnnx.ops.find((op) => op.name === "Gemm");
expectEqual(sampleGemm?.macs, 320, "ONNX Gemm MACs should honor transA/transB and weight shape.");
expectEqual(sampleOnnx.total_macs, sampleOnnx.ops.reduce((sum, op) => sum + Number(op.macs || 0), 0), "ONNX total MACs should equal sum of op MACs.");
expectEqual(sampleOnnx.size_breakdown.stored_scalar_elements, 5418, "ONNX initializer scalar count should match decoded TensorProto shapes.");
expectEqual(sampleOnnx.size_breakdown.constant_bytes, 21672, "ONNX embedded FLOAT32 initializer payload should retain exact stored bytes.");
expectEqual(sampleOnnx.size_breakdown.theoretical_fp16_constant_bytes, 10836, "ONNX FP16 initializer projection should use two bytes per FLOAT element.");
expectEqual(sampleOnnx.size_breakdown.theoretical_int8_constant_bytes, 5418, "ONNX INT8 initializer projection should use one byte per FLOAT element.");
expectEqual(sampleOnnx.size_breakdown.metrics?.theoretical_fp16_constant_bytes?.status, "assessed", "ONNX FP16 size projection should carry assessed status and method.");
expectEqual(sampleOnnx.size_breakdown.metrics?.theoretical_int8_constant_bytes?.status, "assessed", "ONNX INT8 size projection should carry assessed status and method.");
expectEqual(sampleOnnx.size_breakdown.metadata_bytes, null, "Unseparated ONNX metadata bytes must be null rather than a false zero.");
expectEqual(sampleOnnx.onnx_shape_inference?.schema, "deepbom.onnx_shape_inference.v1.30", "ONNX shape inference should expose a versioned source-bound coverage ledger with source-defined effective opset imports, formal-schema and ORT extension rules, symbolic and conditional shape contracts, guarded Squeeze rank unions, conditionally invalid branch accounting, Slice sentinel, bounded NonZero cardinality, nested semantic-conflict propagation, ConvTranspose, and Reshape quotient propagation, recursively assessed SequenceMap/Loop scope, direct Sequence/Optional, TfIdfVectorizer, and ONNX-ML value inference including TreeEnsemble topology, and dense/non-dense output conservation.");
expectEqual(sampleOnnx.onnx_shape_inference?.status, "assessed", "The sample ONNX shape ledger should be fully assessed.");
expectEqual(sampleOnnx.onnx_shape_inference?.rule_supported_nodes, 9, "All sample ONNX nodes should have a declared local shape rule.");
expectEqual(sampleOnnx.onnx_shape_inference?.known_node_output_count, 9, "All sample ONNX node outputs should have known shape and dtype.");
expectEqual(sampleOnnx.onnx_shape_inference?.inferred_outputs, 8, "Eight sample ONNX outputs should be newly inferred rather than predeclared.");
expectEqual(sampleOnnx.onnx_shape_inference?.schema_form_valid_node_count, 9, "Every sample node should match the greatest pinned OpSchema version allowed by opset 13.");
expectEqual(sampleOnnx.onnx_shape_inference?.schema_form_invalid_node_count, 0, "The sample should have no formal OpSchema violation.");
expectEqual(sampleOnnx.onnx_shape_inference?.shape_scope?.unassessed_reachable_node_count, 0, "The sample should have no reachable nested graph or local-function body outside the shape pass.");
expectEqual(sampleOnnx.weight_integrity.status, "assessed", "ONNX embedded initializer integrity should be assessed.");
expectEqual(sampleOnnx.weight_integrity.weight_tensors_scanned, 6, "ONNX initializer decoder should scan every embedded numeric initializer.");
expectEqual(sampleOnnx.weight_integrity.elements_scanned, 5418, "ONNX initializer decoder should scan every stored scalar element.");
expectEqual(sampleOnnx.weight_integrity.nan_tensors, 0, "ONNX sample should contain no NaN initializer tensor.");
expectEqual(sampleOnnx.weight_integrity.inf_tensors, 0, "ONNX sample should contain no infinite initializer tensor.");
expectEqual(sampleOnnx.weight_integrity.all_zero_tensors, 0, "ONNX sample should contain no all-zero initializer tensor.");
expectEqual(sampleOnnx.weight_integrity.zero_kernel_slice_count, 1, "ONNX sample should retain one intentional zero Conv kernel slice.");
expectEqual(sampleOnnx.metadata_presence?.schema, "deepbom.artifact_metadata.v1.4", "ONNX metadata evidence should expose its parser schema.");
expectEqual(sampleOnnx.metadata_presence?.status, "assessed", "ONNX ModelProto and GraphProto metadata should be assessed.");
expectEqual(sampleOnnx.metadata_presence?.metadata_property_count, 2, "ONNX metadata property count should be exact.");
expectEqual(sampleOnnx.metadata_presence?.producer_version, "1.0", "ONNX producer version should be parsed from ModelProto.");
expectEqual(sampleOnnx.metadata_presence?.model_domain, "org.deepbom.samples", "ONNX model domain should be parsed from ModelProto.");
expectEqual(sampleOnnx.metadata_presence?.model_version, 1, "ONNX model version should be parsed as an exact safe integer.");
expectEqual(sampleOnnx.metadata_presence?.documented_preprocessing, false, "Untyped ONNX metadata properties must not be promoted to a machine-verifiable preprocessing contract.");
expectEqual(sampleOnnx.metadata_presence?.preprocessing_contract_status, "not_assessed_untyped_metadata_properties", "ONNX metadata should expose the untyped-property evidence boundary.");
expectEqual(sampleOnnx.metadata_presence?.output_semantics_documented, false, "Untyped ONNX metadata properties must not be promoted to an output-label contract.");
expectEqual(sampleOnnx.weight_integrity.zero_kernel_slice_details?.[0]?.bias_value_sample?.[0], 0.125, "ONNX zero-kernel-slice evidence should decode the corresponding non-zero bias.");
const onnxZeroSlice = sampleOnnx.weight_integrity.zero_kernel_slice_details?.[0];
expectEqual(JSON.stringify(onnxZeroSlice?.shape), JSON.stringify([32, 16, 3, 3]), "ONNX zero-kernel-slice evidence should preserve the initializer shape.");
expectEqual(onnxZeroSlice?.bias_nonzero_for_flagged_channels, true, "ONNX zero-kernel-slice evidence should classify the decoded non-zero bias.");
expectEqual(onnxZeroSlice?.consumer_ops?.[0], "#3 Conv", "ONNX zero-kernel-slice evidence should identify its direct consumer.");
const secondConv = sampleOnnx.ops.find((op) => op.index === 3);
expect(Math.abs(Number(onnxZeroSlice?.consumer_mac_percent) - Number(secondConv?.macs) / sampleOnnx.total_macs) < 1e-15, "ONNX zero-kernel-slice evidence should retain the direct consumer MAC share as an upper-bound scope.");
expect(Math.abs(Number(onnxZeroSlice?.zero_slice_arithmetic_share) - Number(secondConv?.macs) / 32 / sampleOnnx.total_macs) < 1e-15, "ONNX zero-kernel-slice arithmetic proxy should scale consumer MACs by 1/32 output channels.");
expectEqual(sampleOnnx.size_breakdown.duplicate_initializer_analysis.status, "assessed", "ONNX duplicate initializer analysis should be assessed.");
expectEqual(sampleOnnx.size_breakdown.duplicate_constant_bytes, 0, "ONNX sample should have no exact duplicate initializer payload.");
expect(!sampleOnnx.markdown.includes("[object Object]"), "ONNX raw appendix should not stringify objects.");
expect(sampleOnnx.markdown.includes("Op quantization states: none detected"), "ONNX raw appendix should label all-none quantization states plainly.");
initDeepBom({ module: readFileSync("web/protected/deepbom/pkg/deepbom_wasm_bg.wasm") });

function protectedAnalysis(analysis, bytes = Uint8Array.of(0)) {
  return analyze_deepbom(bytes, JSON.stringify(analysis));
}

const protectedOrtCompatibility = protectedAnalysis(sampleOnnx).ort_compatibility_evidence;
applyProtectedOrtCompatibilityEvidence(sampleOnnx, protectedOrtCompatibility);
expectEqual(sampleOnnx.ort_compatibility_assessment_status, "complete", "Protected ORT source compatibility assessment should be complete.");
expectEqual(sampleOnnx.runtime_compat.derived_min_runtime_version, "1.9", "IR 8 / ai.onnx opset 13 should derive ORT 1.9 as the necessary parser floor.");
expectEqual(sampleOnnx.runtime_compat.effective_min_runtime_version, "1.9", "A standard-domain-only model should retain the derived necessary floor as its complete artifact-side parser floor.");
expectEqual(protectedOrtCompatibility.execution_providers.length, 9, "Protected ORT rulepack should assess WASM CPU, CUDA, WebGPU, WebNN, DirectML, QNN, CoreML, NNAPI, and XNNPACK source profiles.");
expect(protectedOrtCompatibility.execution_providers.every((ep) => ep.assignment_evidence_class === "NOT_OBSERVED"), "Source EP version matches must not claim actual runtime assignment.");
expect(protectedOrtCompatibility.execution_providers.every((ep) => ep.ops[0].imported_opset === 13 && ep.ops[0].resolved_schema_version === 11), "Conv must resolve imported opset 13 to pinned ONNX schema since_version 11 before ORT kernel-range comparison.");
expect(protectedOrtCompatibility.execution_providers.every((ep) => ep.ops[0].schema_source_sha256 === "d1c94c1b4b890350a5ff8cc8bf24bd062b09b7a0689293afb1fdc1f7e987b479"), "Resolved standard-domain schema rows must bind the pinned ONNX schema source hash.");
expectEqual(protectedOrtCompatibility.runtime_floor.source_documents.length, 1, "Legacy ORT floor should bind the hashed compatibility matrix source.");
const modernOnnxIdentity = {
  format: "onnx",
  onnx_ir_version: 13,
  opsets: [{ domain: "", version: 25 }],
  ops: [{ index: 0, name: "Abs", domain: "ai.onnx" }],
};
const modernOrtCompatibility = protectedAnalysis(modernOnnxIdentity).ort_compatibility_evidence;
applyProtectedOrtCompatibilityEvidence(modernOnnxIdentity, modernOrtCompatibility);
expectEqual(modernOnnxIdentity.runtime_compat.derived_min_runtime_version, "1.24.1", "IR 13 / ai.onnx opset 25 should derive the first source-backed ORT 1.24.1 release, not ORT 1.25.");
expectEqual(modernOrtCompatibility.execution_providers[0].ops[0].resolved_schema_version, 13, "Abs imported at opset 25 must resolve to its pinned ONNX schema since_version 13.");
expectEqual(modernOrtCompatibility.runtime_floor.source_documents.length, 4, "Modern ORT floor should bind its dependency manifest plus ONNX, ONNX-ML, and IR source files.");
expect(modernOrtCompatibility.runtime_floor.source_documents.every((source) => /^[a-f0-9]{64}$/.test(source.sha256)), "Every modern ORT floor source document should carry a pinned SHA-256.");
const customOnnxIdentity = {
  format: "onnx",
  onnx_ir_version: 8,
  opsets: [{ domain: "", version: 13 }, { domain: "com.microsoft", version: 1 }],
  ops: [{ index: 0, name: "Attention", domain: "com.microsoft" }],
};
const customOrtCompatibility = protectedAnalysis(customOnnxIdentity).ort_compatibility_evidence;
applyProtectedOrtCompatibilityEvidence(customOnnxIdentity, customOrtCompatibility);
const customCpuRow = customOrtCompatibility.execution_providers.find((ep) => ep.execution_provider === "wasm_cpu")?.ops?.[0];
expectEqual(customOrtCompatibility.runtime_floor.status, "assessed_onnx_model_local_and_source_backed_contrib_domains", "A pinned com.microsoft schema with complete release history should contribute to a complete parser/schema floor.");
expectEqual(customOrtCompatibility.runtime_floor.minimum_ort_version, "1.9", "The standard ONNX 1.9 floor should dominate Attention's source-backed ORT 1.3.0 first-release floor.");
expectEqual(customOrtCompatibility.runtime_floor.standard_minimum_ort_version, "1.9", "The standard-domain floor component should remain explicit.");
expectEqual(customOrtCompatibility.runtime_floor.contrib_minimum_ort_version, "1.3.0", "Attention should bind the earliest pinned official ORT release inventory that contains it.");
expectEqual(customOrtCompatibility.runtime_floor.contrib_operator_floors?.[0]?.op_name, "Attention", "The contrib floor should remain bound to the used operator identity.");
expectEqual(customOnnxIdentity.runtime_compat.effective_min_runtime_version, "1.9", "A fully source-backed contrib domain should retain the combined floor as effective.");
const tamperedCustomFloor = structuredClone(customOrtCompatibility);
tamperedCustomFloor.runtime_floor.contrib_operator_floors[0].minimum_ort_version = "1.0.0";
expectThrows(() => applyProtectedOrtCompatibilityEvidence(customOnnxIdentity, tamperedCustomFloor), "runtime-floor arithmetic is inconsistent", "Browser validation should reject a tampered contrib first-release value.");
expectEqual(customCpuRow?.resolved_schema_version, 1, "com.microsoft Attention should resolve to pinned contrib schema version 1.");
expect(customCpuRow?.schema_kernel_version_match === true, "Pinned CPU registration should match com.microsoft Attention version 1.");
expectEqual(customCpuRow?.status, "SOURCE_SCHEMA_KERNEL_VERSION_MATCH_ARTIFACT_PRECHECK_UNRESOLVED", "Missing artifact tensor facts must keep the com.microsoft Attention precheck unresolved without claiming assignment.");
expectEqual(customCpuRow?.schema_source_sha256, "a00b931b8df0db12e03c3346ef9d1abc84200156e1910acaa40dd622711978c9", "Custom-domain schema evidence should bind the exact ContribOperators digest.");
const sampleOnnxMlBom = buildMlBomDocument(sampleOnnx, { hash: sampleOnnx.model_sha256 || "" });
expectEqual(sampleOnnxMlBom.properties.find((item) => item.name === "ondevice:executionProviderAssignment")?.value, "not_assessable_without_runtime_evidence", "ONNX ML-BOM should state the exact runtime-evidence boundary for execution-provider assignment.");
expect(!sampleOnnxMlBom.properties.some((item) => item.name === "ondevice:delegateSuspects" || item.name === "ondevice:predictedNonDelegatedOps"), "ONNX ML-BOM should not emit TFLite delegate predictions.");
const sampleOnnxBundle = buildEngineeringBundleArtifactFiles(sampleOnnx, {
  reportContext: { identity: { filename: sampleOnnx.filename, format: "onnx" }, generatedAt: "2026-07-15T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: sampleOnnx.filename, format: "onnx" } },
  mlBomDocument: sampleOnnxMlBom,
});
const sampleOnnxReport = sampleOnnxBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const sampleOnnxEvidence = JSON.parse(sampleOnnxBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
const sampleOnnxInputContract = sampleOnnx.input_contracts?.[0];
expectEqual(sampleOnnxInputContract?.schema, "deepbom.input_tensor_contract.v1", "ONNX input contract should carry a versioned evidence schema.");
expectEqual(sampleOnnxInputContract?.layout, "NCHW", "A direct standard-domain Conv input should derive NCHW from ONNX tensor semantics.");
expectEqual(sampleOnnxInputContract?.layout_evidence_class, "DERIVED", "ONNX direct-consumer layout should be derived rather than predicted by convention.");
expectEqual(sampleOnnxInputContract?.layout_source_op_name, "Conv", "ONNX input layout should identify the exact source operator.");
expectEqual(sampleOnnxInputContract?.channel_axis, 1, "ONNX NCHW input should bind channel axis 1.");
expectEqual(sampleOnnxInputContract?.expected_range_low, null, "FLOAT32 ONNX input must not receive a synthetic zero-valued numerical range.");
expectEqual(sampleOnnxInputContract?.expected_range_high, null, "FLOAT32 ONNX input must keep the unknown range upper bound null.");
expect(sampleOnnxReport.includes("### Input Tensor Contract Evidence") && sampleOnnxReport.includes("derived_nchw_from_direct_consumer_semantics"), "ONNX Engineering Report should render the source-derived input layout contract.");
expectEqual(sampleOnnxMlBom.metadata.component.properties.find((item) => item.name === "deepbom:model:inputLayoutDerivedCount")?.value, "1", "ONNX ML-BOM should retain the derived layout count in the compact component summary.");
expect(sampleOnnxReport.includes("Initializer-value integrity assessed for 6 constant tensor(s) / 5,418 logical scalar element(s) (5,418 stored value(s) decoded, 0 sparse implicit zero(s))")
  && sampleOnnxReport.includes("weight magnitude and sparsity cover 6 confirmed learned-parameter tensor(s)"), "ONNX static conclusion should separate constant-wide integrity from learned-parameter statistics.");
expect(sampleOnnxReport.includes("Initializer value decoding | Implemented for 6 available dense TensorProto or validated SparseTensorProto initializer(s); 5,418 stored value(s) decoded plus 0 exact sparse implicit zero(s); 0 incomplete-external/invalid/unsupported tensor(s) not assessed"), "ONNX completeness table should render dense+sparse initializer decoding and unavailable-payload coverage.");
expect(!sampleOnnxReport.includes("All-zero quantized kernel slices"), "FLOAT ONNX report must not use quantized-only finding wording.");
expect(sampleOnnxReport.includes("zero-slice arithmetic proxy 2.3%"), "ONNX finding should disclose the 1/32 arithmetic-waste proxy.");
expectEqual(sampleOnnxEvidence.evidence?.conformance_report?.status, "pass", "ONNX engineering evidence should pass semantic conformance.");
expectEqual(sampleOnnxEvidence.evidence?.conformance_report?.critical_failed_invariants, 0, "ONNX semantic conformance should report zero critical failures.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.unregistered_computation_object_keys?.length, 0, "ONNX evidence should contain no unregistered structured calculation.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.unregistered_analysis_object_keys?.length, 0, "Every ONNX top-level analysis key should belong to exactly one metric family.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.multiply_registered_analysis_object_keys?.length, 0, "No ONNX top-level analysis key should have ambiguous metric ownership.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.unbound_assessed_metric_ids?.length, 0, "ONNX assessed metrics should all bind to structured evidence.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.field_coverage?.missing_required_report_field_count, 0, "ONNX Engineering Report should consume every emitted decision-critical field.");
expectEqual(sampleOnnxEvidence.evidence?.metric_coverage_manifest?.field_coverage?.required_report_field_consumed_count, sampleOnnxEvidence.evidence?.metric_coverage_manifest?.field_coverage?.required_report_field_pattern_count, "ONNX required-report field arithmetic should be complete.");
const sampleOnnxIoCoverage = sampleOnnxEvidence.evidence?.metric_coverage_manifest?.field_coverage?.metric_family_ledger?.find((row) => row.metric_id === "contract.io");
expect(Number(sampleOnnxIoCoverage?.required_report_field_pattern_count || 0) > 0, "ONNX input/output contract should contribute explicit required Engineering Report fields.");
expectEqual(sampleOnnxIoCoverage?.required_report_field_consumed_count, sampleOnnxIoCoverage?.required_report_field_pattern_count, "ONNX Engineering Report should consume every emitted input/output contract decision field.");
expectEqual(sampleOnnxIoCoverage?.missing_required_report_field_count, 0, "ONNX input/output contract should have no missing required report field.");
const sampleOnnxDecisionCoverage = sampleOnnxEvidence.evidence?.metric_coverage_manifest?.decision_coverage;
expectEqual(sampleOnnxDecisionCoverage?.assigned_metric_count, sampleOnnxDecisionCoverage?.applicable_metric_count, "Every ONNX metric family should be conserved into one decision domain.");
expectEqual(sampleOnnxDecisionCoverage?.rows?.find((row) => row.domain_id === "runtime_observation")?.status, "not_assessed", "Static ONNX evidence must not claim observed EP placement or execution.");
expectEqual(sampleOnnxDecisionCoverage?.rows?.find((row) => row.domain_id === "product_validation")?.status, "not_assessed", "Static ONNX evidence must not claim task validation.");
expect(sampleOnnxReport.includes("## Decision Coverage At A Glance (DERIVED)"), "ONNX report should front-load the conserved decision boundary.");
expect(sampleOnnxReport.includes("Rulepack provenance"), "ONNX report should bind protected ORT source-rule provenance.");
expect(sampleOnnxReport.includes("Exported assignment-capture envelope"), "ONNX report should render the canonical assignment-capture capability envelope.");
expect(sampleOnnxReport.includes("## Metric Coverage Manifest (DERIVED)"), "ONNX report should render the metric coverage manifest.");
expect(sampleOnnxReport.includes("## ONNX Shape Inference Coverage (SOURCE_PINNED_AND_DERIVED)"), "ONNX report should render the source-pinned, conserved shape-inference ledger.");
expect(sampleOnnxReport.includes("deepbom.artifact_metadata.v1.4"), "ONNX report should render the artifact-metadata parser schema.");
expect(sampleOnnxReport.includes("not_assessed_untyped_metadata_properties"), "ONNX report should disclose that metadata_props are not typed preprocessing evidence.");
expect(sampleOnnxReport.includes("deepbom.sample.input_basis"), "ONNX report should preserve parsed metadata property keys.");
const sampleOnnxFindingIds = buildFindingsRegister(sampleOnnx).map((item) => item.finding_id);
expect(sampleOnnxFindingIds.includes("EA-IOC-0001"), "Generic ONNX metadata must not suppress the missing preprocessing-contract finding.");
expect(sampleOnnxFindingIds.includes("EA-OUT-0001"), "Generic ONNX metadata must not suppress the missing output-contract finding.");

const invalidSchemaOnnx = structuredClone(sampleOnnx);
const invalidShape = invalidSchemaOnnx.onnx_shape_inference;
invalidShape.status = "fail";
const originalImport = invalidShape.opset_import_contract.rows[0];
Object.assign(originalImport, { version: 0, status: "fail", reason_codes: ["opset_version_not_positive_safe_integer"], diagnostic_codes: [], selected_effective_import: false });
Object.assign(invalidShape.opset_import_contract, {
  status: "fail",
  import_count: 1,
  valid_import_count: 0,
  invalid_import_count: 1,
  effective_domain_count: 0,
  duplicate_domain_count: 0,
  duplicate_identical_domain_count: 0,
  duplicate_version_variant_domain_count: 0,
  invalid_version_count: 1,
  invalid_domains: [originalImport.domain],
  unresolvable_domains: [originalImport.domain],
  effective_imports: [],
});
invalidShape.schema_form_assessment_status = "fail";
invalidShape.schema_form_valid_node_count -= 1;
invalidShape.schema_form_invalid_node_count = 1;
invalidShape.schema_form_rows[0].status = "fail";
invalidShape.schema_form_rows[0].reason_codes = ["attribute_not_defined:invented"];
invalidShape.schema_form_rows[0].detail = "Injected fail-closed OpSchema fixture.";
invalidShape.rule_unresolved_node_count = 1;
invalidShape.rule_unresolved_node_indices = [0];
invalidShape.rule_unresolved_nodes = [{ node_index: 0, op_name: invalidShape.schema_form_rows[0].op_name, reason: "opset_schema_form_invalid" }];
Object.assign(invalidShape.shape_scope, {
  status: "partial",
  nested_graph_count: 1,
  nested_graph_node_count: 2,
  reachable_scope_count: 1,
  reachable_nested_graph_count: 1,
  reachable_nested_graph_node_count: 2,
  executed_reachable_scope_count: 0,
  fully_assessed_reachable_scope_count: 0,
  partially_assessed_reachable_scope_count: 0,
  failed_reachable_scope_count: 0,
  reachable_scope_unresolved_output_count: 0,
  scope_execution_rows: [{
    scope_class: "nested_graph",
    scope: "main_graph/node:0/attribute:then_branch",
    owner: "main_graph",
    status: "not_assessed",
    node_count: 2,
    execution_count: 0,
    assessed_node_count: 0,
    unassessed_node_count: 2,
    unresolved_output_count: 0,
    reason_codes: ["reachable_scope_shape_inference_not_executed"],
  }],
  reachable_exclusion_count: 1,
  unassessed_reachable_node_count: 2,
  exclusions: [{
    scope_class: "nested_graph",
    scope: "main_graph/node:0/attribute:then_branch",
    owner: "main_graph",
    node_count: 2,
    reason_code: "nested_graph_shape_inference_not_executed",
  }],
});
const invalidFindings = buildFindingsRegister(invalidSchemaOnnx);
expectEqual(invalidFindings.find((item) => item.finding_id === "EA-ONX-0007")?.technical_priority, "High", "Formal OpSchema violation should enter the High action queue.");
expectEqual(invalidFindings.find((item) => item.finding_id === "EA-ONX-0008")?.technical_priority, "Medium", "Fully declared outer outputs should keep the exact extended-scope exclusion at Medium.");
const invalidMlBom = buildMlBomDocument(invalidSchemaOnnx, { hash: invalidSchemaOnnx.model_sha256 || "" });
const invalidBundle = buildEngineeringBundleArtifactFiles(invalidSchemaOnnx, {
  reportContext: { identity: { filename: invalidSchemaOnnx.filename, format: "onnx" }, generatedAt: "2026-07-15T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: invalidSchemaOnnx.filename, format: "onnx" } },
  mlBomDocument: invalidMlBom,
});
const invalidReport = invalidBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const invalidEvidence = JSON.parse(invalidBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(invalidReport.includes("### OpSchema Formal Contract Failures"), "Engineering Report should render formal schema failures.");
expect(invalidReport.includes("### OperatorSet Import Contract Failures"), "Engineering Report should render invalid opset imports.");
expect(invalidReport.includes("### Extended Shape Scope Exclusions"), "Engineering Report should render reachable excluded scopes.");
expectEqual(invalidEvidence.evidence?.conformance_report?.status, "pass", "Schema/scope failure fixture should remain internally conformant while reporting the artifact failure.");

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const sampleFloatBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v1_025_224_float.tflite"));
const sampleFloat = analyze_tflite_for_target(sampleFloatBytes, "mobilenet_v1_025_224_float.tflite", "android_mid_a55");
expectEqual(Object.is(sampleFloat.quantization_status.quantized_compute_mac_percent, -0), false, "FLOAT quantized-compute ratio must serialize as canonical positive zero rather than IEEE negative zero.");
expectEqual(Object.is(sampleFloat.quantization_status.quantized_compute_macs, -0), false, "FLOAT quantized-compute MAC total must serialize as canonical positive zero rather than IEEE negative zero.");
sampleFloat.model_sha256 = "f".repeat(64);
const protectedFloatAggregate = analyze_deepbom(sampleFloatBytes, JSON.stringify(sampleFloat));
applyProtectedXnnpackSelectorEvidence(sampleFloat, protectedFloatAggregate.xnnpack_selector_evidence);
applyProtectedOrtCompatibilityEvidence(sampleFloat, protectedFloatAggregate.ort_compatibility_evidence);
const sampleFloatMlBom = buildMlBomDocument(sampleFloat, { hash: sampleFloat.model_sha256, targetId: sampleFloat.target_profile.id });
const sampleFloatBundle = buildEngineeringBundleArtifactFiles(sampleFloat, {
  reportContext: { identity: { filename: sampleFloat.filename, format: "tflite", sha256: sampleFloat.model_sha256 }, deepBomResult: protectedFloatAggregate },
  rawEvidenceContext: { identity: { filename: sampleFloat.filename, format: "tflite", sha256: sampleFloat.model_sha256 } },
  mlBomDocument: sampleFloatMlBom,
});
const sampleFloatReport = sampleFloatBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const sampleFloatEvidence = JSON.parse(sampleFloatBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expectEqual(sampleFloat.input_counterexample.status, "not_applicable", "FLOAT sample should emit a valid zero-source input-counterexample portfolio.");
expect(sampleFloatReport.includes("## Quantization Research Coverage (DERIVED)")
  && sampleFloatReport.includes("Floating-point (float)")
  && sampleFloatReport.includes("Artifact-applicable labs | 0")
  && !sampleFloatReport.includes("## Model Input Tensor ABI Witness (DERIVED)")
  && !sampleFloatReport.includes("constructively realized"),
"FLOAT report should use the shared zero-applicable denominator instead of claiming a constructive input witness certificate.");
expectEqual(sampleFloatEvidence.evidence?.static_analysis?.input_counterexample, null, "FLOAT raw export should suppress class-excluded input-witness detail.");
expectEqual(sampleFloatEvidence.evidence?.conformance_report?.status, "pass", "A zero-source, zero-witness input-counterexample portfolio should pass bundle conformance.");
const sampleTfliteBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
const sampleTflite = analyze_tflite_for_target(
  sampleTfliteBytes,
  "mobilenet_v2_1.0_224_quant.tflite",
  "wasm_simd",
);
const sampleNonMacOps = sampleTflite.total_ops - 2 * sampleTflite.total_macs;
expectEqual(sampleNonMacOps, 279104, "TFLite non-MAC arithmetic operation count.");
expectEqual(2 * sampleTflite.total_macs + sampleNonMacOps, sampleTflite.total_ops, "Arithmetic operation mix must reconcile exactly to total ops.");
expectEqual(270082314 - 2 * 133736512, 2609290, "Requested Overview example should resolve its exact non-MAC operation count.");
sampleTflite.deployment_frontier = compute_deployment_frontier(sampleTfliteBytes, sampleTflite.filename, JSON.stringify(["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"]));
sampleTflite.model_sha256 = sampleTflite.deployment_frontier.artifact_sha256;
expectEqual(sampleTflite.xnnpack_selector_assessment_status, "not_loaded", "Public TFLite analysis should not contain the protected exact selector result.");
const protectedSelector = protectedAnalysis(sampleTflite).xnnpack_selector_evidence;
const selectorEligibleOpCount = sampleTflite.ops.filter((op) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name)).length;
const lowNormAssessedOps = sampleTflite.ops.filter((op) => Number.isInteger(op.low_norm_filter_count) && Number.isInteger(op.low_norm_filter_total));
expectEqual(lowNormAssessedOps.length, selectorEligibleOpCount, "TFLite low-norm op coverage.");
expect(lowNormAssessedOps.every((op) => op.low_norm_filter_count >= 0 && op.low_norm_filter_count <= op.low_norm_filter_total), "TFLite low-norm count bounds.");
const graphEvidenceMaps = buildGraphEvidenceMaps(sampleTflite);
expectEqual(graphEvidenceMaps.opAnnotations.size, sampleTflite.ops.length, "Graph op evidence coverage.");
expectEqual(graphEvidenceMaps.lowNormStats.size, lowNormAssessedOps.length, "Graph low-norm evidence coverage.");
expect(!readFileSync("web/app.js", "utf8").includes("compute_quick_low_norm_stat("), "No per-op analyzer re-entry.");
expectEqual(protectedSelector.assessed_op_count, selectorEligibleOpCount, "Protected selector should assess every eligible TFLite compute op.");
applyProtectedXnnpackSelectorEvidence(sampleTflite, protectedSelector);
expectEqual(sampleTflite.xnnpack_selector_assessment_status, "complete", "Protected selector merge should make the sample assessment complete.");
expectEqual(sampleTflite.xnnpack_selector_evidence_access, "research", "Protected selector merge should retain its access scope.");
const protectedTfliteAggregate = analyze_deepbom(new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite")), JSON.stringify(sampleTflite));
applyProtectedOrtCompatibilityEvidence(sampleTflite, protectedTfliteAggregate.ort_compatibility_evidence);
expectEqual(sampleTflite.ort_compatibility_assessment_status, "not_applicable", "Protected aggregate analysis should preserve an explicitly bounded non-ONNX ORT result.");
expect(sampleTflite.ops.filter((op) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name)).every((op) => String(op.xnnpack_kernel_evidence_class || "").startsWith("SOURCE_ENUMERATED_")), "Every eligible sample op should receive a source-enumerated candidate or explicit no-match result.");
const sampleTfliteRaw = buildStaticAuditMarkdown(sampleTflite, sampleTflite.model_sha256 || "");
const triageBreakdown = sampleTflite.insights.score_breakdown;
const triagePenalty = Object.entries(triageBreakdown)
  .filter(([key]) => key.endsWith("_penalty"))
  .reduce((sum, [, value]) => sum + Number(value || 0), 0);
expectEqual(triageBreakdown.final_score, Math.max(0, triageBreakdown.base - triagePenalty), "Heuristic triage index should equal its emitted deterministic penalty formula.");
expectEqual(sampleTflite.insights.score, triageBreakdown.final_score, "Heuristic triage index and breakdown final score should match.");
expectEqual(sampleTflite.insights.score_evidence_class, "HEURISTIC", "Composite triage index must not be labeled DERIVED or OBSERVED.");
expectEqual(triageBreakdown.exact_zero_kernel_signal_points, 16, "Scheme-independent exact-zero kernel evidence should contribute to triage for per-tensor quantized artifacts.");
expectEqual(triageBreakdown.quantization_risk_penalty, 16, "Quantization triage should take the maximum of op-level and exact-zero kernel signals.");
expect(sampleTflite.insights.score <= 67, "The sample triage score should retain the exact-zero kernel penalty instead of overstating posture.");
expect(!String(sampleTflite.insights.score_label || "").toLowerCase().includes("strong"), "A sample with 11 exact-zero kernel slices must not receive a strong-posture label.");
expectEqual(sampleTflite.metadata_presence?.schema, "deepbom.artifact_metadata.v1.4", "TFLite metadata evidence should expose its parser schema.");
expectEqual(sampleTflite.metadata_presence?.status, "assessed_no_model_metadata", "The quantized sample should explicitly report absent TFLite Model Metadata.");
expectEqual(sampleTflite.metadata_presence?.conversion_metadata_status, "not_present", "The quantized sample should explicitly report absent conversion metadata.");
expectEqual(sampleTflite.metadata_presence?.converter_tensorflow_version, "", "An absent conversion-metadata entry must not synthesize a TensorFlow converter version.");
expectEqual(sampleTflite.metadata_presence?.conversion_metadata_schema_sha256, "2464449e30bfa6032c0218b53a1a83b224c6eda9b5cfd9f12211c4c0017dc20e", "Conversion-metadata decoding should bind its pinned schema digest.");
expectEqual(sampleTflite.metadata_presence?.documented_preprocessing, false, "A TFLite model description alone must not be promoted to a preprocessing contract.");
expectEqual(sampleTflite.metadata_presence?.preprocessing_contract_status, "absent_no_model_metadata", "The quantized sample should expose why preprocessing is not documented.");
expectEqual(sampleTflite.metadata_presence?.output_semantics_documented, false, "The quantized sample should not claim output semantics without a verified packed label mapping.");
expectEqual(sampleTflite.metadata_presence?.associated_file_archive_status, "not_present", "The quantized sample should explicitly report that no packed associated-file archive exists.");
expectEqual(sampleTflite.metadata_presence?.packed_associated_file_count, 0, "Absent associated-file archives should retain an exact zero file count.");
expect(sampleTflite.insights.score_method.includes("score_breakdown"), "Heuristic triage index should disclose where each formula component is emitted.");
const calibration = compute_static_runtime_calibration(
  new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite")),
  "mobilenet_v2_1.0_224_quant.tflite",
  "wasm_simd",
  0,
);
const expectedColdStaticEstimateMs = sampleTflite.ops.reduce((sum, op) => sum + Number(op.bottleneck_total_us || 0), 0) / 1000;
const expectedPackingMs = sampleTflite.ops.reduce((sum, op) => sum + Number(op.bottleneck_packing_us || 0), 0) / 1000;
const expectedBoundarySetupMs = sampleTflite.ops.reduce((sum, op) => sum + Number(op.bottleneck_break_us || 0), 0) / 1000;
const expectedSteadyStaticEstimateMs = expectedColdStaticEstimateMs - expectedPackingMs - expectedBoundarySetupMs;
expectEqual(Number(calibration.static_estimate_ms.toFixed(9)), Number(expectedSteadyStaticEstimateMs.toFixed(9)), "Static runtime calibration should compare post-warmup runtime with the steady-state op ledger.");
expectEqual(Number(calibration.cold_start_static_estimate_ms.toFixed(9)), Number(expectedColdStaticEstimateMs.toFixed(9)), "Static runtime calibration should preserve the cold-start op ledger.");
expectEqual(Number(calibration.one_time_packing_ms.toFixed(9)), Number(expectedPackingMs.toFixed(9)), "Static runtime calibration should expose one-time packing separately.");
expectEqual(Number(calibration.boundary_setup_ms.toFixed(9)), Number(expectedBoundarySetupMs.toFixed(9)), "Static runtime calibration should expose the setup-only partition-planning profile separately.");
expectEqual(calibration.assessed_op_count, sampleTflite.ops.length, "Static runtime calibration should report full op coverage for the sample.");
expect(calibration.method.includes("op.bottleneck_total_us - op.bottleneck_packing_us - op.bottleneck_break_us"), "Static runtime calibration should disclose its steady/cold deterministic formula.");
expect(sampleTflite.quantization_status.quantized_compute_ops <= sampleTflite.quantization_status.compute_ops, "TFLite quantized compute op count must not exceed MAC-bearing compute op denominator.");
expect(sampleTfliteRaw.includes("Quantized MAC-bearing compute ops: 53/53 (100.0%)"), "TFLite raw appendix should report quantized compute ops with the MAC-bearing compute denominator.");
expect(sampleTfliteRaw.includes("Quantized compute MACs: 300,775,552 / 300,775,552 (100.0%)"), "TFLite raw appendix should report quantized compute MACs with a checkable numerator and denominator.");
expect(sampleTfliteRaw.includes("Quantization consistency check: ok"), "TFLite raw appendix should include a quantization consistency invariant.");
expect(!sampleTfliteRaw.includes("64/53"), "TFLite raw appendix should not mix graph-op quant states with compute-op denominator.");
expect(!sampleTfliteRaw.includes("120.8%"), "TFLite raw appendix should not emit ratios above 100% for quantized compute coverage.");
const quantSummary = quantSummaryEvidence(sampleTflite);
expectEqual(quantSummary.quantizedConstants.length, 53, "Quant summary should count all decoded 8-bit constant tensors.");
expectEqual(quantSummary.perAxisConstants.length, 0, "Legacy sample should expose zero per-axis constants, not missing scale analysis.");
expectEqual(quantSummary.opSignals.size, 53, "All 53 asymmetric UINT8 kernel consumers should carry a review overlay.");
expect(quantSummary.categories.find((row) => row.label === "Asymmetric UINT8 kernels")?.evidence.startsWith("53 "), "Quant summary should expose 53 asymmetric UINT8 kernels.");
expect(quantSummary.categories.find((row) => row.label === "Per-tensor depthwise weights")?.evidence.startsWith("17 "), "Quant summary should expose 17 per-tensor depthwise ops.");
expect(quantSummary.categories.find((row) => row.label === "Stored kernel-channel integrity")?.evidence.includes("11 exact-zero"), "Quant summary should expose 11 exact-zero stored slices.");
expect(quantSummary.categories.find((row) => row.label === "8-bit constant grid")?.evidence.startsWith("3/53"), "Quant summary should expose the 3/53 low-grid result.");
const averagePool = sampleTflite.ops.find((op) => op.name === "AVERAGE_POOL_2D");
expectEqual(averagePool?.xnnpack_break_class, "high-adjacent-mac-exposure", "AVERAGE_POOL_2D should retain its adjacent delegated-MAC exposure class without being called structural.");
expectEqual(sampleTflite.xnnpack_structural_chain_breaks, 0, "MobileNetV2 sample should not report its pooling boundary as structural/view.");
expectEqual(sampleTflite.xnnpack_zero_mac_chain_breaks, 1, "MobileNetV2 sample should count one zero-modeled-MAC boundary.");
expectEqual(sampleTflite.xnnpack_effective_chain_breaks, 1, "MobileNetV2 sample pooling boundary should remain an effective non-structural boundary.");
const structuralBreakOps = sampleTflite.ops.filter((op) => op.xnnpack_chain_break && ["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"].includes(op.name));
const zeroMacNonstructuralBreakOps = sampleTflite.ops.filter((op) => op.xnnpack_chain_break && Number(op.macs || 0) === 0 && !["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"].includes(op.name));
expectEqual(sampleTflite.xnnpack_structural_chain_breaks, structuralBreakOps.length, "Structural break aggregate must be recomputable from operator anatomy.");
expectEqual(sampleTflite.xnnpack_zero_mac_chain_breaks, zeroMacNonstructuralBreakOps.length, "Zero-MAC non-structural break aggregate must be recomputable from operator anatomy.");
expectEqual(sampleTflite.xnnpack_effective_chain_breaks + sampleTflite.xnnpack_structural_chain_breaks, sampleTflite.xnnpack_chain_breaks, "Structural and non-structural break anatomy must partition all break ops exactly.");
expectEqual(sampleTflite.predicted_partition_boundaries?.edge_count, 2, "MobileNetV2 should expose the exact delegate-to-CPU and CPU-to-delegate graph edges.");
expectEqual(sampleTflite.predicted_partition_boundaries?.summed_edge_payload_bytes, 64000, "MobileNetV2 predicted internal boundary edges should total 62,720 + 1,280 bytes.");
expectEqual(sampleTflite.tensor_liveness?.peak_bytes, 1505280, "MobileNetV2 declared-shape live payload should remain exactly reproducible.");
expectEqual(sampleTflite.tensor_liveness?.peak_at_op, 4, "MobileNetV2 live-payload peak should remain bound to op #4.");
expectEqual(sampleTflite.tensor_arena_plan?.status, "assessed", "MobileNetV2 ArenaPlanner projection should be fully assessed.");
expectEqual(sampleTflite.tensor_arena_plan?.non_persistent_arena_bytes, 1655808, "MobileNetV2 non-persistent ArenaPlanner high-water mark should remain exact.");
expectEqual(sampleTflite.tensor_arena_plan?.persistent_arena_bytes, 0, "MobileNetV2 sample should have no persistent arena allocation.");
expectEqual(sampleTflite.tensor_arena_plan?.combined_arena_bytes, 1655808, "MobileNetV2 combined ArenaPlanner projection should remain exact.");
expectEqual(sampleTflite.tensor_arena_plan?.root_allocation_count, 56, "MobileNetV2 ArenaPlanner projection should retain 56 root allocations.");
expectEqual(sampleTflite.tensor_arena_plan?.shared_tensor_count, 10, "MobileNetV2 residual ADD registrations should produce 10 deterministic in-place aliases.");
expectEqual(sampleTflite.tensor_arena_plan?.source_comparator_tie_group_count, 0, "MobileNetV2 ArenaPlanner source comparator should fully order every non-full-lifetime root allocation.");
expectEqual(sampleTflite.tensor_arena_plan?.source_comparator_fully_orders_projection, true, "MobileNetV2 projection should not require the analyzer tie-break extension.");
expectEqual(sampleTflite.tensor_arena_plan?.source_commit, "87bbf65b8d23d3f06912b1b2183587e1884bc45c", "ArenaPlanner projection must remain bound to the pinned TensorFlow source commit.");
expect(sampleTfliteRaw.includes("Boundary anatomy: total = non-structural 1 + structural/view 0; zero-MAC non-structural 1 is a subset of non-structural"), "TFLite raw appendix should expose independently checkable exposure and operator-anatomy categories.");
expect(sampleTfliteRaw.includes("Boundary operator categories: pooling/reduction 1 · structural/view 0 · other non-structural 0"), "TFLite raw appendix should classify the zero-modeled-MAC boundary as pooling/reduction.");
expect(!sampleTfliteRaw.includes("zero-MAC shape/structural ops among boundaries"), "TFLite raw appendix should not use the retired zero-MAC-equals-structural wording.");

const expectedNonDelegated = Object.fromEntries(Object.entries((sampleTflite.ops || []).reduce((counts, op) => {
  if (op.xnnpack_chain_break || op.xnnpack_supported === false) counts[op.name] = (counts[op.name] || 0) + 1;
  return counts;
}, {})).sort(([a], [b]) => a.localeCompare(b)));
const sampleMlBom = buildMlBomDocument(sampleTflite, { hash: sampleTflite.model_sha256 || "" });
expect(sampleMlBom.metadata.component.properties.length <= 120, "Compact ML-BOM component summary must remain bounded to 120 properties.");
expect(sampleMlBom.properties.length <= 20, "Compact ML-BOM document summary must remain bounded to 20 properties.");
expect(Buffer.byteLength(JSON.stringify(sampleMlBom)) < 128 * 1024, "Compact ML-BOM should remain below 128 KiB for the verified MobileNetV2 artifact.");
const mlBomNonDelegated = JSON.parse(sampleMlBom.properties.find((item) => item.name === "ondevice:predictedNonDelegatedOps")?.value || "{}");
expectEqual(JSON.stringify(Object.fromEntries(Object.entries(mlBomNonDelegated).sort(([a], [b]) => a.localeCompare(b)))), JSON.stringify(expectedNonDelegated), "ML-BOM predicted non-delegated ops should match the analyzer/report source of truth.");
expectEqual(mlBomNonDelegated.AVERAGE_POOL_2D, 1, "ML-BOM should include the predicted AVERAGE_POOL_2D non-delegated op.");
expectEqual(sampleMlBom.metadata.component.properties.find((item) => item.name === "deepbom:model:predictedPartitionBoundaryLogicalBytes")?.value, "64000", "ML-BOM should carry the graph-derived internal boundary payload in its compact summary.");
expectEqual(sampleMlBom.metadata.component.properties.find((item) => item.name === "deepbom:compatibility:detailLocation")?.value, "engineering_evidence.json#/evidence/static_analysis", "ML-BOM should bind omitted detailed evidence to the canonical engineering-evidence pointer.");
expect(!sampleMlBom.metadata.component.properties.some((item) => item.name === "deepbom:model:arenaCombinedBytes" || item.name === "deepbom:model:arenaInPlaceAliases"), "Compact ML-BOM should not duplicate the detailed ArenaPlanner ledger.");

const sampleFindings = buildFindingsRegister(sampleTflite);
const legacyQuantFinding = sampleFindings.find((item) => item.finding_id === "EA-QNT-0104");
expect(legacyQuantFinding?.observation.includes('description is "TOCO Converted."')
  && legacyQuantFinding?.observation.includes("53 asymmetric UINT8 kernel tensor(s)")
  && legacyQuantFinding?.observation.includes("17 per-tensor depthwise op(s)")
  && legacyQuantFinding?.observation.includes("runtime floor is 1.5.0"), "The legacy quantization finding should synthesize the independently observed TOCO lineage evidence.");
const channelFinding = sampleFindings.find((item) => item.finding_id === "EA-CHN-0001");
const channelRecommendation = sampleTflite.recommendations.find((item) => String(item.title || "").includes("channel tail not aligned"));
expectEqual(Boolean(channelFinding), Boolean(channelRecommendation), "Channel-tail finding and native recommendation should be emitted or suppressed together.");
if (channelFinding && channelRecommendation) {
  expectEqual(channelFinding.affected_operator, `#${String(channelRecommendation.op_index).padStart(3, "0")} ${sampleTflite.ops[channelRecommendation.op_index]?.name || ""}`, "Channel-tail finding and native recommendation should select the same representative op.");
}

const sampleIdentity = {
  filename: sampleTflite.filename,
  format: sampleTflite.format,
  sha256: sampleTflite.model_sha256 || "",
  target_id: sampleTflite.target_profile.id,
  target_label: sampleTflite.target_profile.label,
  target_profile_sha256: sampleTflite.target_profile.profile_sha256,
  operator_count: sampleTflite.operator_count,
  tensor_count: sampleTflite.tensor_count,
  total_macs: sampleTflite.total_macs,
};
const sampleBundleFiles = buildEngineeringBundleArtifactFiles(sampleTflite, {
  reportContext: { identity: sampleIdentity, generatedAt: "2026-07-15T00:00:00.000Z" },
  rawEvidenceContext: { identity: sampleIdentity },
  mlBomDocument: sampleMlBom,
});
const sampleEvidence = JSON.parse(sampleBundleFiles.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
const sampleEngineeringReport = sampleBundleFiles.find((file) => file.name === "engineering_report.md")?.data || "";
const sampleInputContract = sampleTflite.input_contracts?.[0];
const sampleInputTensor = sampleTflite.inputs?.[0];
const sampleInputScale = sampleInputTensor?.scale_sample?.[0];
const sampleInputZeroPoint = sampleInputTensor?.zero_point_sample?.[0];
expectEqual(sampleInputContract?.schema, "deepbom.input_tensor_contract.v1", "TFLite input contract should carry a versioned evidence schema.");
expectEqual(sampleInputContract?.layout, "NHWC", "A direct TFLite CONV_2D activation input should derive NHWC.");
expectEqual(sampleInputContract?.layout_evidence_class, "DERIVED", "TFLite input layout should be derived from operator semantics.");
expectEqual(sampleInputContract?.layout_source_op_index, 0, "TFLite input layout should bind the exact direct consumer op.");
expectEqual(sampleInputContract?.channel_axis, 3, "TFLite NHWC input should bind channel axis 3.");
expectEqual(sampleInputContract?.channels, 3, "TFLite input contract should derive three channels from the semantic channel axis.");
expectEqual(sampleInputContract?.expected_range_low, sampleInputScale * (0 - sampleInputZeroPoint), "UINT8 input lower real bound should derive from qmin, scale, and zero point.");
expectEqual(sampleInputContract?.expected_range_high, sampleInputScale * (255 - sampleInputZeroPoint), "UINT8 input upper real bound should derive from qmax, scale, and zero point.");
expect(sampleEngineeringReport.includes("### Input Tensor Contract Evidence") && sampleEngineeringReport.includes("derived_nhwc_from_direct_consumer_semantics"), "TFLite Engineering Report should render the exact input tensor contract.");
expect(sampleFindings.find((item) => item.finding_id === "EA-IOC-0001")?.observation.includes("tensor numerical status is known_from_artifact_quantization_metadata"), "Input finding should distinguish the known tensor numerical range from undocumented source preprocessing.");
expectEqual(sampleMlBom.metadata.component.properties.find((item) => item.name === "deepbom:model:inputLayoutDerivedCount")?.value, "1", "TFLite ML-BOM should retain the derived input-layout count in the compact component summary.");
const tamperedInputContractAnalysis = structuredClone(sampleTflite);
tamperedInputContractAnalysis.input_contracts[0].layout = "NCHW";
expectThrows(() => buildEngineeringBundleArtifactFiles(tamperedInputContractAnalysis, {
  reportContext: { identity: sampleIdentity, generatedAt: "2026-07-15T00:00:00.000Z" },
  rawEvidenceContext: { identity: sampleIdentity },
  mlBomDocument: buildMlBomDocument(tamperedInputContractAnalysis, { hash: sampleTflite.model_sha256 || "" }),
}), "CF-IO-002", "Input-layout evidence tampering should fail independent conformance.");
const quantContracts = sampleEvidence.evidence?.quantization?.quantization_contract_checks;
expectEqual(quantContracts?.status, "pass", "Top-level quantization contract status should represent deterministic contract integrity only.");
expectEqual(quantContracts?.contract_integrity_status, "pass", "MobileNetV2 deterministic quantization contracts should pass.");
expectEqual(quantContracts?.quantization_design_review_status, "review", "MobileNetV2 design-review signals should remain separate from contract integrity.");
expectEqual(quantContracts?.io_dequantization?.inputs?.[0]?.tensor_numerical_contract_status, "known_from_artifact_quantization_metadata", "Quantized input numerical range should be known from artifact metadata.");
expectEqual(quantContracts?.io_dequantization?.inputs?.[0]?.source_data_to_tensor_preprocessing_status, "not_embedded_in_artifact", "Source-data preprocessing must remain unclaimed by the tensor range.");
expectEqual(quantContracts?.bias_scale?.checked_groups, 53, "Structured quantization evidence should include all 53 bias-scale groups.");
expectEqual(quantContracts?.bias_scale?.mismatch_groups, 0, "Structured bias-scale contract should pass the MobileNetV2 sample.");
expectEqual(quantContracts?.residual_add?.checked_ops, 10, "Structured quantization evidence should include all 10 residual ADD ops.");
expect(Math.abs(Number(quantContracts?.residual_add?.maximum_input_scale_ratio) - 1.455558202640278) < 1e-12, "Structured residual scale ratio should remain exactly reproducible.");
expect(Math.abs(Number(quantContracts?.accumulator_bound?.maximum_int32_ratio) - 0.0020873942422156198) < 1e-15, "Structured exact INT32 accumulator envelope should remain exactly reproducible.");
expectEqual(quantContracts?.accumulator_bound?.checked_channels, 18057, "Structured exact INT32 accumulator evidence should cover every assessed output channel.");
expectEqual(quantContracts?.accumulator_bound?.minimum_int32_headroom_bits, 8, "Structured exact INT32 accumulator evidence should preserve the minimum signed-bit headroom.");
expect(sampleEngineeringReport.includes("53 group(s), 53 channel(s) checked; 0 mismatch group(s)"), "Engineering report should render the structured bias-scale result.");
expect(sampleEngineeringReport.includes("max ratio 1.46x"), "Engineering report should render the structured residual ratio.");
expect(sampleEngineeringReport.includes("Contract integrity status | PASS"), "Engineering report should separate deterministic quantization contract integrity.");
expect(sampleEngineeringReport.includes("Quantization design review status | REVIEW"), "Engineering report should separate quantization design review posture.");
expect(sampleEngineeringReport.includes("NOT_APPLICABLE_TO_QUANTIZATION_SCHEME")
  && sampleEngineeringReport.includes("decoded exact-zero/near-zero stored-slice and Channel Vitality checks remain independently applicable"), "Per-axis scale-vector checks should be classed as scheme-inapplicable without suppressing scheme-independent kernel integrity checks.");
expect(sampleEngineeringReport.includes("artifact contains 0 B of FLOAT constant payload")
  && sampleEngineeringReport.includes("FP32-to-FP16 storage counterfactual is a no-op")
  && sampleEngineeringReport.includes("FP32-to-INT8 storage counterfactual is a no-op"), "A model with no FLOAT constants should not emit meaningful-looking FP16/INT8 storage-floor values.");
expect(sampleEngineeringReport.includes("Source-data-to-tensor preprocessing | NOT_EMBEDDED_IN_ARTIFACT"), "Engineering report should not treat dequantized tensor range as source preprocessing evidence.");
expect(sampleEngineeringReport.includes("All external parameters | fully_affine_quantized")
  && sampleEngineeringReport.includes("Inputs | fully_affine_quantized")
  && sampleEngineeringReport.includes("Outputs | fully_affine_quantized"), "Engineering report should render aggregate external-boundary storage contracts.");
expect(sampleEngineeringReport.includes("channel_order, source_value_domain, mean_standard_deviation_normalization, resize_interpolation"), "Engineering report should state what the boundary contract does not establish.");
expect(sampleEngineeringReport.includes("T4 `MobilenetV2/Conv_1/Relu6`; #061 CONV_2D -> #062 AVERAGE_POOL_2D"), "Engineering report should render the exact delegate-to-CPU boundary tensor edge.");
expect(sampleEngineeringReport.includes("62.5 KiB across 2 internal edge(s)"), "Engineering report should render the exact summed logical boundary-edge payload.");
expect(sampleEngineeringReport.includes("Combined arena projection | 1.6 MiB (1,655,808 B)"), "Engineering report should render the exact ArenaPlanner projection.");
expect(sampleEngineeringReport.includes("Root allocations / in-place aliases | 56 / 10"), "Engineering report should render exact ArenaPlanner root and alias counts.");
expectEqual(sampleEvidence.evidence?.static_analysis?.tensor_arena_plan?.combined_arena_bytes, 1655808, "Structured evidence should retain the exact ArenaPlanner projection.");
expectEqual(sampleEvidence.evidence?.conformance_report?.status, "pass", "Sample engineering evidence should pass self-conformance.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.unregistered_computation_object_keys?.length, 0, "TFLite evidence should contain no unregistered structured calculation.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.unregistered_analysis_object_keys?.length, 0, "Every TFLite top-level analysis key should belong to exactly one metric family.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.multiply_registered_analysis_object_keys?.length, 0, "No TFLite top-level analysis key should have ambiguous metric ownership.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.unbound_assessed_metric_ids?.length, 0, "TFLite assessed metrics should all bind to structured evidence.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.field_coverage?.missing_required_report_field_count, 0, "TFLite Engineering Report should consume every emitted decision-critical field.");
expectEqual(sampleEvidence.evidence?.metric_coverage_manifest?.field_coverage?.required_report_field_consumed_count, sampleEvidence.evidence?.metric_coverage_manifest?.field_coverage?.required_report_field_pattern_count, "TFLite required-report field arithmetic should be complete.");
const sampleTfliteIoCoverage = sampleEvidence.evidence?.metric_coverage_manifest?.field_coverage?.metric_family_ledger?.find((row) => row.metric_id === "contract.io");
expect(Number(sampleTfliteIoCoverage?.required_report_field_pattern_count || 0) > 0, "TFLite input/output contract should contribute explicit required Engineering Report fields.");
expectEqual(sampleTfliteIoCoverage?.required_report_field_consumed_count, sampleTfliteIoCoverage?.required_report_field_pattern_count, "TFLite Engineering Report should consume every emitted input/output contract decision field.");
expectEqual(sampleTfliteIoCoverage?.missing_required_report_field_count, 0, "TFLite input/output contract should have no missing required report field.");
const sampleDecisionCoverage = sampleEvidence.evidence?.metric_coverage_manifest?.decision_coverage;
expectEqual(sampleDecisionCoverage?.assigned_metric_count, sampleDecisionCoverage?.applicable_metric_count, "Every TFLite metric family should be conserved into one decision domain.");
expectEqual(sampleDecisionCoverage?.rows?.find((row) => row.domain_id === "runtime_observation")?.status, "not_assessed", "Static TFLite evidence must not claim observed delegate placement or execution.");
expectEqual(sampleDecisionCoverage?.rows?.find((row) => row.domain_id === "product_validation")?.status, "not_assessed", "Static TFLite evidence must not claim task validation.");
expect(sampleEngineeringReport.includes("## Decision Coverage At A Glance (DERIVED)"), "TFLite report should front-load the conserved decision boundary.");
expect(sampleEngineeringReport.includes("## Advanced Static Proof Coverage (OBSERVED/DERIVED)"), "TFLite report should expose every advanced proof module's status and schema.");
expect(sampleEngineeringReport.includes("Pinned XNNPACK source commit"), "TFLite report should expose the selector source commit.");
expect(sampleEngineeringReport.includes("## Metric Coverage Manifest (DERIVED)"), "TFLite report should render the metric coverage manifest.");
expect(sampleEngineeringReport.includes("## Pinned XNNPACK Kernel Candidates"), "Engineering report should render the protected selector section for the real sample.");
expect(sampleEngineeringReport.includes("deepbom.artifact_metadata.v1.4"), "TFLite report should render the artifact-metadata parser schema.");
expect(sampleEngineeringReport.includes("TFLite conversion metadata") && sampleEngineeringReport.includes("2464449e30bfa6032c0218b53a1a83b224c6eda9b5cfd9f12211c4c0017dc20e"), "TFLite report should disclose conversion-version availability and its pinned schema source.");
expect(sampleEngineeringReport.includes("absent_no_model_metadata"), "TFLite report should render the exact preprocessing evidence boundary.");
expect(sampleEngineeringReport.includes("No terminal ZIP end-of-central-directory record was found"), "TFLite report should render the associated-file archive assessment.");
expect(sampleEngineeringReport.includes(protectedSelector.xnnpack_source_commit), "Engineering report should bind protected candidates to the pinned XNNPACK commit.");
expectEqual(sampleEvidence.evidence?.static_analysis?.xnnpack_selector_assessment_status, "complete", "Structured evidence should retain the complete protected selector status.");
expect(!sampleMlBom.metadata.component.properties.some((item) => item.name === "deepbom:model:xnnpackSelectorEvidenceAccess"), "Compact ML-BOM should not duplicate protected selector detail.");
expect((sampleMlBom.metadata.component.externalReferences || []).some((item) => item.type === "evidence" && item.url === "engineering_evidence.json"), "Compact ML-BOM should link the protected selector and other detailed evidence through the canonical evidence document.");
const sampleManifest = buildEngineeringBundleManifest({ analysis: sampleTflite, model: sampleIdentity, files: sampleBundleFiles, generatedAt: "2026-07-15T00:00:00.000Z" });
const sampleManifestFile = { name: "manifest.json", data: JSON.stringify(sampleManifest, null, 2) };
const samplePackageDigest = await buildCanonicalPackageDigest([...sampleBundleFiles, sampleManifestFile]);
const sampleMlBomProperties = Object.fromEntries(sampleMlBom.metadata.component.properties.map((item) => [item.name, item.value]));
for (const [actual, expected, label] of [
  [sampleEvidence.evidence.static_analysis.operator_count, sampleTflite.operator_count, "evidence static-analysis op count"],
  [sampleEvidence.evidence.model_structure.graph_counts.operators, sampleTflite.operator_count, "evidence model-structure op count"],
  [Number(sampleMlBomProperties["mlbom:model:operatorCount"]), sampleTflite.operator_count, "ML-BOM op count"],
  [sampleManifest.model.operator_count, sampleTflite.operator_count, "manifest op count"],
  [Number(sampleMlBomProperties["mlbom:model:tensorCount"]), sampleTflite.tensor_count, "ML-BOM tensor count"],
  [Number(sampleMlBomProperties["mlbom:model:totalMacs"]), sampleTflite.total_macs, "ML-BOM MAC total"],
  [sampleManifest.model.target_profile_sha256, sampleTflite.target_profile.profile_sha256, "manifest target-profile hash"],
  [samplePackageDigest.files.length, 4, "unsigned package member digest count"],
]) {
  expectEqual(actual, expected, `Cross-output conformance: ${label}.`);
}
expectEqual(JSON.stringify(sampleEvidence.evidence.mlbom_cyclonedx), JSON.stringify(sampleMlBom), "Cross-output conformance: consolidated evidence should embed the exact ML-BOM document.");
expectEqual(JSON.stringify(sampleManifest.payload_files), JSON.stringify(["engineering_report.md", "engineering_evidence.json", "input_counterexample_input.bin"]), "Cross-output conformance: manifest payload list should match the compact bundle files.");
expectEqual(JSON.stringify(samplePackageDigest.files.map((item) => item.name)), JSON.stringify(["engineering_evidence.json", "engineering_report.md", "input_counterexample_input.bin", "manifest.json"]), "Cross-output conformance: canonical digest should cover every unsigned package member in sorted order.");

const estimate = estimateModelAnalysis(tfliteFile, { header: unknownTfliteHeader, readBytes: 64 * 1024, elapsed: 4 });
for (const [actual, expected, label] of [
  [estimate.format, "TFLite", "estimate format"],
  [estimate.readBytes, 64 * 1024, "estimate read bytes"],
  [formatEstimate({ highMs: 999 }), "<1 sec", "sub-second estimate text"],
]) {
  expectEqual(actual, expected, label);
}
expect(estimate.lowMs >= 120, "estimate low bound");
expect(estimate.highMs > estimate.lowMs, "estimate high bound");

const quantSampleFile = { name: "mobilenet_v2_1.0_224_quant.tflite", size: 3577760 };
const quantEstimate = estimateModelAnalysis(quantSampleFile, { readBytes: 64 * 1024, elapsed: 4 }, { comparisonTargetCount: 4 });
const flagshipEstimate = estimateModelAnalysis(quantSampleFile, { readBytes: 64 * 1024, elapsed: 4 }, { comparisonTargetCount: 5 });
const locallyCalibratedEstimate = estimateModelAnalysis(quantSampleFile, null, {
  comparisonTargetCount: 4,
  auditTimings: [{ format: "TFLITE", sizeBytes: quantSampleFile.size, comparisonTargetCount: 4, durationMs: 23000 }],
});
for (const [actual, expected, label] of [
  [formatEstimate(quantEstimate), "about 10-30 sec", "calibrated four-target full audit range"],
  [quantEstimate.estimateMethod, FULL_AUDIT_ESTIMATE_METHOD, "full-audit estimate method version"],
  [formatEstimate(flagshipEstimate), "about 10-35 sec", "calibrated five-target full audit range"],
  [formatEstimate(locallyCalibratedEstimate), "about 15-40 sec on this device", "local-browser full audit range"],
  [locallyCalibratedEstimate.timingSampleCount, 1, "local-browser timing sample count"],
  [formatMeasuredAudit(23822), "23.8 sec measured", "measured full audit text"],
  [formatMeasuredAudit(253), "253 ms measured", "measured sub-second audit text"],
]) {
  expectEqual(actual, expected, label);
}

const identity = buildModelIdentity({
  analysis: {
    filename: "analyzed.tflite",
    format: "tflite",
    model_sha256: "abc123",
    target_profile: { id: "a55", label: "Android A55" },
    operator_count: 7,
    tensor_count: 11,
    total_macs: 13,
  },
  filename: "fallback.onnx",
  modelBytes: new Uint8Array([0x08]),
  selectedTargetId: "wasm_simd",
  selectedTargetLabel: "Browser WASM",
});
for (const [field, expected, label] of [
  ["filename", "analyzed.tflite", "filename"],
  ["format", "tflite", "format"],
  ["sha256", "abc123", "hash"],
  ["target_id", "a55", "target id"],
  ["target_label", "Android A55", "target label"],
  ["operator_count", 7, "op count"],
  ["tensor_count", 11, "tensor count"],
  ["total_macs", 13, "macs"],
]) {
  expectEqual(identity[field], expected, `identity ${label}`);
}

const fallbackIdentity = buildModelIdentity({
  filename: "fallback.onnx",
  modelBytes: new Uint8Array([0x08]),
  selectedTargetId: "wasm_simd",
  selectedTargetLabel: "Browser WASM",
});
for (const [field, expected, label] of [
  ["format", "onnx", "fallback format"],
  ["target_id", "wasm_simd", "fallback target id"],
  ["target_label", "Browser WASM", "fallback target label"],
]) {
  expectEqual(fallbackIdentity[field], expected, `identity ${label}`);
}

const stats = latencyStats([4, 5, 6, 9]);
for (const [actual, expected, label] of [
  [stats.min, 4, "latency min"],
  [stats.max, 9, "latency max"],
  [Number(stats.stddev.toFixed(3)), 1.871, "latency stddev"],
  [Number(stats.cv.toFixed(3)), 0.312, "latency cv"],
]) {
  expectEqual(actual, expected, label);
}

const telemetryPayload = await buildStructureTelemetryPayload({
  format: "tflite",
  histogram: [{ name: "CONV_2D", count: 2 }],
  stages: [{ index: 0, key: "1x8x8x16", op_count: 2, mac_percent: 100, patterns: ["MBConv"] }],
  inputs: [{ dtype: "INT8", shape: [1, 8, 8, 3] }],
  outputs: [{ dtype: "INT8", shape: [1, 8, 8, 2] }],
  target_profile: { label: "Android A55" },
  xnnpack_chain_breaks: 1,
  total_macs: 1024,
  tensor_count: 6,
}, {
  targetId: "android_mid_a55",
  targetProfileLabel: "Android A55",
  browserBucket: "Chrome",
});
for (const [field, expected, label] of [
  ["target", "android_mid_a55", "target id"],
  ["target_profile", "Android A55", "target profile"],
  ["browser_bucket", "Chrome", "browser bucket"],
]) {
  expectEqual(telemetryPayload[field], expected, `telemetry ${label}`);
}
expect(telemetryPayload.fingerprint.startsWith("sf_"), "telemetry fingerprint prefix");
expectEqual(telemetryPayload.fingerprint.length, 67, "telemetry fingerprint length");

const benchmarkPayload = buildBenchmarkTelemetryPayload(telemetryPayload, {
  backend: "wasm",
  compileMs: 1.25,
  firstRunMs: 5.1,
  stats: { min: 4.0, max: 6.8, p50: 4.5, p90: 5.5, p95: 6.0, p99: 6.5, mean: 4.75, stddev: 0.42, cv: 0.0884 },
  steadyStats: { min: 4.0, max: 5.9, p50: 4.4, p90: 5.2, p95: 5.4, p99: 5.9, mean: 4.6, stddev: 0.31, cv: 0.0674 },
  warmup: 3,
  runs: 20,
  outputCount: 2,
  outputDigest: "abcdef1234567890",
  timings: [4.1, 4.5, 5.2],
  inputContracts: [{ input_name: "input" }],
  outputContracts: [{ output_name: "output" }],
  noiseMethod: "private-local-method",
  noiseDiagnostics: { trimmedP50: 4.5 },
}, {
  targetId: "android_mid_a55",
  browserBucket: "Chrome / Windows",
  preparedInput: true,
  runtimeStatus: "Ready",
});
for (const [actual, expected, label] of [
  [benchmarkPayload.consent, true, "consent"],
  [benchmarkPayload.structure.fingerprint, telemetryPayload.fingerprint, "structure fingerprint"],
  [benchmarkPayload.run.backend, "wasm", "backend"],
  [benchmarkPayload.run.p50_ms, 4.5, "p50"],
  [benchmarkPayload.run.p95_ms, 6.0, "p95"],
  [benchmarkPayload.run.metadata.output_digest, "abcdef1234567890", "output digest"],
  [benchmarkPayload.run.metadata.first_run_ms, 5.1, "first run"],
  [benchmarkPayload.run.metadata.steady_p50_ms, 4.4, "steady p50"],
  [benchmarkPayload.run.metadata.steady_mean_ms, 4.6, "steady mean"],
  [benchmarkPayload.run.metadata.steady_cv, 0.0674, "steady cv"],
  [benchmarkPayload.run.metadata.min_ms, 4.0, "min"],
  [benchmarkPayload.run.metadata.max_ms, 6.8, "max"],
  [benchmarkPayload.run.metadata.stddev_ms, 0.42, "stddev"],
  [benchmarkPayload.run.metadata.cv, 0.0884, "cv"],
  [benchmarkPayload.run.metadata.prepared_input, true, "prepared input"],
  [benchmarkPayload.run.metadata.runtime_status, "Ready", "runtime status"],
]) {
  expectEqual(actual, expected, `benchmark ${label}`);
}
const benchmarkTelemetryText = JSON.stringify(benchmarkPayload.run);
expect(!benchmarkTelemetryText.includes("timings") && !benchmarkTelemetryText.includes("inputContracts") && !benchmarkTelemetryText.includes("outputContracts") && !benchmarkTelemetryText.includes("noiseDiagnostics") && !benchmarkTelemetryText.includes("noiseMethod"), "benchmark telemetry must exclude raw latency samples, local noise diagnostics, and tensor contracts.");
expectEqual(p99EvidenceForSampleCount(99).status, "underpowered", "p99 evidence should remain underpowered below 100 measured runs");
expectEqual(p99EvidenceForSampleCount(100).status, "descriptive", "p99 evidence should expose the 100-position empirical threshold without claiming confidence");

done("Model file contract passed (format detection, analysis estimate, and identity fallback).");
