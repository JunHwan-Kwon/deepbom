import assert from "node:assert/strict";
import {
  buildTensorRtStaticPreflight,
  createTensorRtBuildProfile,
  TENSORRT_PARSER_OBSERVATION_SCHEMA,
} from "../web/lib/tensorrt-static-preflight.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { buildExecutionPlacementEvidence } from "../web/lib/execution-placement-evidence.js";
import { buildPublicShareAnalysis } from "../web/lib/public-export.js";
import { buildOnnxDynamicShapeCostContract } from "../web/lib/dynamic-shape-cost.js";
import {
  TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
  tensorRtParserObservationIdentity,
} from "../web/lib/tensorrt-engine-inspector.js";
import { tensorRtCycloneDxPropertyEntries } from "../web/lib/tensorrt-cyclonedx-properties.js";
import { executionPlacementMarkdown } from "../web/lib/report-execution-placement.js";
import { buildArtifactEvidenceEnvelope, validateArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";

const sha = "a".repeat(64);
const analysis = {
  format: "onnx",
  model_sha256: sha,
  quantized_tensors: 0,
  inputs: [{ index: 0, name: "tokens", dtype: "FLOAT16", shape: [1, 16], shape_signature: [-1, 16] }],
  tensors: [
    { index: 0, name: "tokens", dtype: "FLOAT16", shape: [1, 16], shape_signature: [-1, 16] },
    { index: 1, name: "a", dtype: "FLOAT16", shape: [1, 16] },
    { index: 2, name: "b", dtype: "FLOAT16", shape: [1, 16] },
    { index: 3, name: "c", dtype: "FLOAT16", shape: [1, 16] },
    { index: 4, name: "out", dtype: "FLOAT16", shape: [1, 16] },
  ],
  ops: [
    { index: 0, name: "MatMul", inputs: [0], outputs: [1], macs: 256, macs_status: "assessed", estimated_bytes: 64, estimated_bytes_status: "assessed" },
    { index: 1, name: "Relu", inputs: [1], outputs: [2], macs: 0, macs_status: "not_applicable", estimated_bytes: 64, estimated_bytes_status: "assessed" },
    { index: 2, name: "Custom", inputs: [2], outputs: [3], macs: null, macs_status: "not_assessed", estimated_bytes: 64, estimated_bytes_status: "assessed" },
    { index: 3, name: "Add", inputs: [3], outputs: [4], macs: 0, macs_status: "not_applicable", estimated_bytes: 64, estimated_bytes_status: "assessed" },
  ],
  onnx_external_data: { incomplete_tensor_count: 0, verification_failed_tensor_count: 0 },
  onnx_domain_analysis: { registry_issue_count: 0, external_registry_count: 0 },
};

const config = {
  execution_path: "native_tensorrt",
  expected_tensorrt_version: "10.14.1",
  expected_cuda_version: "13.0",
  device_id: 0,
  device_compute_capability: "8.7",
  precision: { tf32: true, fp16: true, bf16: false, int8: false, fp8: false },
  workspace_limit_bytes: 1_073_741_824,
  builder_optimization_level: 3,
  dla_core: null,
  allow_gpu_fallback: false,
  calibration_cache_sha256: null,
  plugins: [],
  optimization_profiles: [{ id: "batch", inputs: [{ name: "tokens", min: [1, 16], opt: [2, 16], max: [8, 16] }] }],
};

const unbound = buildTensorRtStaticPreflight(analysis);
assert.equal(unbound.status, "blocked");
assert.equal(unbound.blocking_issue_count, 1);
assert.equal(unbound.projection.state_counts.UNRESOLVED, 4);
assert.equal(unbound.trust_boundary.browser_engine_deserialization, "prohibited");

const profile = createTensorRtBuildProfile(config);
assert.match(profile.profile_sha256, /^[a-f0-9]{64}$/);
assert.equal(profile.optimization_profiles[0].inputs[0].max[0], 8);
const staticPreflight = buildTensorRtStaticPreflight(analysis, config);
assert.equal(staticPreflight.status, "configuration_valid_parser_observation_required");
assert.equal(staticPreflight.blocking_issue_count, 0);
assert.equal(staticPreflight.projection.state_counts.UNRESOLVED, 4);
assert.equal(staticPreflight.projection.workload_envelope.total.complete_macs_decimal, null);

const dynamicTensors = [
  { index: 0, name: "x", dtype: "FLOAT32", shape: [-1, 3, 32, 32], shape_signature: [-1, 3, 32, 32], shape_declared: true, value_kind: "tensor", type_proto: { shapeDimensions: [{ kind: "symbolic", parameter: "N" }, { kind: "value", value: 3 }, { kind: "value", value: 32 }, { kind: "value", value: 32 }] }, role: "input" },
  { index: 1, name: "w", dtype: "FLOAT32", shape: [16, 3, 3, 3], shape_declared: true, value_kind: "tensor", constant_buffer: true },
  { index: 2, name: "y", dtype: "FLOAT32", shape: [-1, 16, 30, 30], shape_signature: [-1, 16, 30, 30], shape_declared: true, value_kind: "tensor", type_proto: { shapeDimensions: [{ kind: "symbolic", parameter: "N" }, { kind: "value", value: 16 }, { kind: "value", value: 30 }, { kind: "value", value: 30 }] }, role: "output" },
];
const dynamicOps = [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs: null, macs_status: "not_assessed", estimated_bytes: null }];
const dynamicAnalysis = {
  ...analysis,
  inputs: [dynamicTensors[0]], tensors: dynamicTensors, ops: dynamicOps,
  dynamic_shape_cost_contract: buildOnnxDynamicShapeCostContract(dynamicTensors, dynamicOps),
};
const dynamicConfig = structuredClone(config);
dynamicConfig.optimization_profiles = [{ id: "image_batch", inputs: [{ name: "x", min: [1, 3, 32, 32], opt: [2, 3, 32, 32], max: [8, 3, 32, 32] }] }];
const dynamicPreflight = buildTensorRtStaticPreflight(dynamicAnalysis, dynamicConfig);
assert.equal(dynamicPreflight.optimization_profile_cost.status, "assessed");
assert.deepEqual(dynamicPreflight.optimization_profile_cost.scenarios.map((row) => row.total_macs_decimal), ["388800", "777600", "3110400"]);
assert.deepEqual(dynamicPreflight.optimization_profile_cost.scenarios.map((row) => row.peak_live_payload_bytes_decimal), ["69888", "139776", "559104"]);
assert(dynamicPreflight.optimization_profile_cost.scenarios.every((row) => row.status === "exact_conditional"));

const conflictingAnalysis = structuredClone(dynamicAnalysis);
conflictingAnalysis.inputs.push({ ...structuredClone(dynamicTensors[0]), index: 3, name: "mask" });
conflictingAnalysis.tensors.push(conflictingAnalysis.inputs[1]);
conflictingAnalysis.dynamic_shape_cost_contract.symbols[0].occurrences.push({ tensor_index: 3, tensor_name: "mask", axis: 0 });
const conflictingConfig = structuredClone(dynamicConfig);
conflictingConfig.optimization_profiles[0].inputs.push({ name: "mask", min: [2, 3, 32, 32], opt: [2, 3, 32, 32], max: [2, 3, 32, 32] });
const conflictingPreflight = buildTensorRtStaticPreflight(conflictingAnalysis, conflictingConfig);
assert(conflictingPreflight.issues.some((row) => row.id === "profile_symbol_binding_conflict" && row.severity === "BLOCKING"));
const publicAnalysis = buildPublicShareAnalysis({
  ...analysis,
  ort_compatibility_evidence: { execution_providers: [] },
  tensorrt_static_preflight: staticPreflight,
});
assert(!JSON.stringify(publicAnalysis).includes(sha), "Public analysis must remove every nested copy of the source artifact digest.");
assert.doesNotThrow(() => buildExecutionPlacementEvidence(publicAnalysis),
  "Public redaction must preserve TensorRT projection identity against the report-local blank artifact binding.");

const observation = {
  schema: TENSORRT_PARSER_OBSERVATION_SCHEMA,
  artifact_sha256: sha,
  build_profile_sha256: profile.profile_sha256,
  build_profile_file_sha256: sha256TextHex(`${canonicalJson(profile)}\n`),
  build_profile: profile,
  execution_path: "native_tensorrt",
  tensorrt_version: "10.14.1",
  cuda_version: "13.0",
  device_id: 0,
  device_compute_capability: "8.7",
  device_identity: "Jetson test fixture / CC 8.7",
  api_method: "supportsModelV2",
  subgraph_support_semantics: "per_subgraph_api_flag",
  parser_returned: false,
  collector: {
    binary_sha256: "b".repeat(64),
    source_set_sha256: "c".repeat(64),
    git_commit: "fixture-commit",
    git_state: "clean",
  },
  plugins: [],
  subgraphs: [
    { subgraph_index: 0, supported: true, sdk_reported_flag: true, node_indices: [0, 1] },
    { subgraph_index: 1, supported: false, sdk_reported_flag: false, node_indices: [2] },
  ],
  errors: [{ code: 7, message: "Fixture unsupported custom node" }],
};
const observed = buildTensorRtStaticPreflight(analysis, config, observation);
assert.equal(observed.status, "parser_observed_partial");
assert.equal(observed.parser_observation.observed_node_count, 3);
assert.equal(observed.parser_observation.unobserved_node_count, 1);
assert.deepEqual(observed.projection.state_counts, {
  CONDITIONALLY_ELIGIBLE: 2,
  DEFINITE_EXCLUSION: 1,
  UNRESOLVED: 1,
});
assert.equal(observed.projection.boundary_edge_count, 2);
assert.equal(observed.projection.boundary_payload.summed_edge_payload_bytes, 64);
assert.equal(observed.projection.evidence_class, "PARSER_OBSERVED_CONFIGURATION_BOUND");

const engineInformation10 = {
  Layers: [
    {
      Name: "fused_gemm_relu",
      LayerType: "CaskConvolution",
      Inputs: [{ Name: "tokens", Dimensions: [-1, 16], "Format/Datatype": "Row major linear FP16" }],
      Outputs: [{ Name: "b", Dimensions: [-1, 16], "Format/Datatype": "Row major linear FP16" }],
      TacticName: "sm87_fixture_tactic",
      TacticValue: "0x1234",
      StreamId: 0,
      Metadata: "[ONNX Layer: MatMul][ONNX Layer: Relu]",
    },
    {
      Name: "output_add [ONNX Layer: Add]",
      LayerType: "ElementWise",
      Inputs: [{ Name: "c", Dimensions: [1, 16], "Format/Datatype": "Row major linear FP16" }],
      Outputs: [{ Name: "out", Dimensions: [1, 16], "Format/Datatype": "Row major linear FP16" }],
    },
  ],
  Bindings: ["tokens", "out"],
};
const engineEvidence10 = {
  schema: TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
  artifact_sha256: sha,
  build_profile_sha256: profile.profile_sha256,
  parser_observation_sha256: tensorRtParserObservationIdentity(observation),
  engine: { sha256: "d".repeat(64), byte_length: 4096 },
  runtime: {
    tensorrt_version: "10.14.1", cuda_version: "13.0", device_id: 0,
    device_compute_capability: "8.7", device_identity: "Jetson test fixture / CC 8.7",
  },
  build_capture: {
    evidence_class: "DECLARED_BUILD_CAPTURE",
    binding_method: "fixture_files_with_independent_sha256",
    tool_name: "trtexec",
    tool_binary_sha256: "e".repeat(64),
    invocation_sha256: "f".repeat(64),
    collector_source_set_sha256: null,
    model_input_sha256: sha,
    serialized_engine_sha256: "d".repeat(64),
  },
  inspector: {
    source: "trtexec_exportLayerInfo",
    profiling_verbosity: "detailed",
    schema_generation: "tensorrt_10x",
    execution_context_bound: false,
    source_file_sha256: "1".repeat(64),
    source_file_byte_length: 2048,
    canonical_json_sha256: sha256TextHex(canonicalJson(engineInformation10)),
    engine_information: engineInformation10,
  },
};
const engineObserved = buildTensorRtStaticPreflight(analysis, config, observation, engineEvidence10);
assert.equal(engineObserved.status, "engine_inspected_parser_observed_partial");
assert.equal(engineObserved.engine_inspector_evidence.engine_layer_count, 2);
assert.equal(engineObserved.engine_inspector_evidence.io_tensor_count, 2);
assert.equal(engineObserved.engine_inspector_evidence.tactic_annotated_layer_count, 1);
assert.equal(engineObserved.engine_inspector_evidence.multi_source_metadata_layer_count, 1);
assert.equal(engineObserved.engine_inspector_evidence.dynamic_dimension_tensor_count, 2);
assert.equal(engineObserved.engine_inspector_evidence.data_type_inventory.length, 0,
  "TensorRT 10.x combined Format/Datatype text must not be misreported as a separately exposed dtype.");
assert.equal(engineObserved.engine_inspector_evidence.source_mapping_status,
  "source_name_tokens_observed_original_op_identity_not_established");
assert.equal(engineObserved.engine_inspector_evidence.artifact_engine_relation, "declared_build_capture_binding");
const integratedAnalysis = { ...analysis, filename: "fixture.onnx", file_size_bytes: 1024, tensorrt_static_preflight: engineObserved };
const integratedPlacement = buildExecutionPlacementEvidence(integratedAnalysis);
assert.equal(integratedPlacement.configuration_preflights[0].engine_inspector.engine_sha256, "d".repeat(64));
assert.match(executionPlacementMarkdown(integratedPlacement), /Optimized Engine Inspector Evidence/);
assert.match(executionPlacementMarkdown(integratedPlacement), /selected engine metadata, not tactic timings/i);
const propertyMap = new Map(tensorRtCycloneDxPropertyEntries(integratedAnalysis));
assert.equal(propertyMap.get("deepbom:model:tensorRtEngineSha256"), "d".repeat(64));
assert.equal(propertyMap.get("deepbom:model:tensorRtTacticAnnotatedLayerCount"), 1);
assert.equal(propertyMap.get("deepbom:model:tensorRtInspectorProfilingVerbosity"), "detailed");
assert.equal(propertyMap.get("deepbom:model:tensorRtEngineBuildCaptureClass"), "DECLARED_BUILD_CAPTURE");
const envelope = buildArtifactEvidenceEnvelope(integratedAnalysis, { generatedAt: "2026-08-29T00:00:00.000Z" });
assert.equal(validateArtifactEvidenceEnvelope(envelope).valid, true);
assert.equal(envelope.format_extensions.onnx.tensorrt_static_preflight.engine_inspector_evidence.engine.sha256, "d".repeat(64));

const config11 = { ...structuredClone(config), expected_tensorrt_version: "11.0.0" };
const profile11 = createTensorRtBuildProfile(config11);
const engineInformation11 = {
  Layers: [{
    Name: "gemm_0", LayerType: "MatrixMultiply",
    Inputs: [{ Name: "tokens", Dimensions: [-1, 16], Format: "LINEAR", DataType: "FP16" }],
    Outputs: [{ Name: "out", Dimensions: [-1, 16], Format: "LINEAR", Datatype: "FP16" }],
    TacticName: "sm87_fixture_tactic_11",
  }],
  "I/O Tensors": [
    { Name: "tokens", IOMode: "INPUT", DataType: "FP16", Dimensions: [-1, 16], Location: "DEVICE", IsShapeInferenceIO: false, ProfileInfo: [{ MinShape: [1, 16], OptShape: [2, 16], MaxShape: [8, 16], Format: "LINEAR" }] },
    { Name: "out", IOMode: "OUTPUT", Datatype: "FP16", Dimensions: [-1, 16], Location: "DEVICE", IsShapeInferenceIO: false, ProfileInfo: [] },
  ],
};
const engineEvidence11 = structuredClone(engineEvidence10);
engineEvidence11.build_profile_sha256 = profile11.profile_sha256;
engineEvidence11.parser_observation_sha256 = null;
engineEvidence11.runtime.tensorrt_version = "11.0.0";
engineEvidence11.inspector.schema_generation = "tensorrt_11x";
engineEvidence11.inspector.canonical_json_sha256 = sha256TextHex(canonicalJson(engineInformation11));
engineEvidence11.inspector.engine_information = engineInformation11;
const engineOnly = buildTensorRtStaticPreflight(analysis, config11, null, engineEvidence11);
assert.equal(engineOnly.status, "engine_inspected_parser_observation_unbound");
assert.equal(engineOnly.engine_inspector_evidence.engine_layer_count, 1);
assert.equal(engineOnly.engine_inspector_evidence.io_tensor_count, 2);
assert.deepEqual(engineOnly.engine_inspector_evidence.data_type_inventory, [{ name: "FP16", count: 2 }]);
assert.equal(engineOnly.engine_inspector_evidence.io_tensors[0].profile_info[0].max_shape[0], 8);

const badGeneration = structuredClone(engineEvidence10);
badGeneration.inspector.engine_information["I/O Tensors"] = [];
badGeneration.inspector.canonical_json_sha256 = sha256TextHex(canonicalJson(badGeneration.inspector.engine_information));
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, observation, badGeneration), /schema generation/);
const badDigest = structuredClone(engineEvidence10);
badDigest.inspector.canonical_json_sha256 = "2".repeat(64);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, observation, badDigest), /digest/);
const badParserBinding = structuredClone(engineEvidence10);
badParserBinding.parser_observation_sha256 = null;
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, observation, badParserBinding), /parser-observation binding/);
const falseObservedBuild = structuredClone(engineEvidence10);
falseObservedBuild.build_capture.evidence_class = "COLLECTOR_OBSERVED_BUILD_CAPTURE";
falseObservedBuild.build_capture.collector_source_set_sha256 = "3".repeat(64);
falseObservedBuild.build_capture.serialized_engine_sha256 = "4".repeat(64);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, observation, falseObservedBuild), /does not bind/);

const missingProfile = structuredClone(config);
missingProfile.optimization_profiles = [];
assert(buildTensorRtStaticPreflight(analysis, missingProfile).issues.some((row) => row.id === "dynamic_profile_missing" && row.severity === "BLOCKING"));
const badStatic = structuredClone(config);
badStatic.optimization_profiles[0].inputs[0].max[1] = 32;
assert(buildTensorRtStaticPreflight(analysis, badStatic).issues.some((row) => row.id === "profile_static_dimension_changed"));
const int8 = structuredClone(config);
int8.precision.int8 = true;
assert(buildTensorRtStaticPreflight(analysis, int8).issues.some((row) => row.id === "int8_calibration_unbound"));

assert.throws(() => buildTensorRtStaticPreflight(analysis, config, { ...observation, artifact_sha256: "b".repeat(64) }), /identity/);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, {
  ...observation,
  collector: { ...observation.collector, source_set_sha256: null },
}), /identity/);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, {
  ...observation,
  subgraphs: [...observation.subgraphs, { subgraph_index: 2, supported: true, sdk_reported_flag: true, node_indices: [1] }],
}), /duplicate node/);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, { ...observation, tensorrt_version: "10.13.0" }), /runtime version/);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, { ...observation, device_compute_capability: "9.0" }), /identity/);
assert.throws(() => buildTensorRtStaticPreflight(analysis, config, { ...observation, build_profile_file_sha256: "d".repeat(64) }), /identity/);
const dirtyObserved = buildTensorRtStaticPreflight(analysis, config, {
  ...observation,
  collector: { ...observation.collector, git_state: "dirty" },
});
assert(dirtyObserved.issues.some((row) => row.id === "collector_source_dirty" && row.severity === "UNRESOLVED"));

const ortConfig = {
  ...config,
  execution_path: "ort_tensorrt_ep",
  ort_ep_options: {
    provider_priority: 1,
    max_partition_iterations: 1000,
    min_subgraph_size: 1,
    engine_cache_enable: true,
    timing_cache_enable: true,
    context_memory_sharing_enable: false,
  },
};
const ort = buildTensorRtStaticPreflight(analysis, ortConfig);
assert.equal(ort.projection.profile_id, "ort_tensorrt_ep");
assert.equal(ort.projection.state_counts.UNRESOLVED, 4,
  "Native TensorRT parser assumptions must not be projected as ORT TensorRT EP assignment.");

const tflite = buildTensorRtStaticPreflight({ format: "tflite", model_sha256: sha });
assert.equal(tflite.status, "not_applicable_non_onnx");
assert.equal(tflite.projection, null);

console.log("TensorRT static preflight passed (dynamic profiles, precision/calibration, native-vs-ORT separation, identity-bound parser observation, trust boundary, and fail-closed mutations).");
