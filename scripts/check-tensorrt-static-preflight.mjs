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
