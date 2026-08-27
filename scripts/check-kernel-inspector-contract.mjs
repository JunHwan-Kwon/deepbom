import { readFileSync } from "node:fs";
import { buildRuntimeAssignmentTemplate, parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { opPrecisionLabel } from "../web/lib/analysis.js";
import { buildRuntimeEvidence } from "../web/lib/report.js";
import { createCheck } from "./check-assert.mjs";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const { done, expect, expectEqual, expectThrows } = createCheck("Kernel inspector contract check");
const sha = "a".repeat(64);
const profileSha = "c".repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: sha,
  target_profile: { id: "android_mid_a55", profile_sha256: profileSha },
  tensors: [
    { index: 0, name: "input", shape: [1, 4], shape_signature: [1, 4], dtype: "FLOAT32" },
    { index: 1, name: "t1", shape: [1, 4], shape_signature: [1, 4], dtype: "FLOAT32" },
    { index: 2, name: "t2", shape: [1, 4], shape_signature: [1, 4], dtype: "FLOAT32" },
    { index: 3, name: "t3", shape: [1, 4], shape_signature: [1, 4], dtype: "FLOAT32" },
  ],
  inputs: [{ index: 0, name: "input", shape: [1, 4], shape_signature: [1, 4], dtype: "FLOAT32" }],
  ops: [
    { index: 0, name: "CONV_2D", inputs: [0], outputs: [1], xnnpack_chain_id: 0, macs: 100, estimated_bytes: 32 },
    { index: 1, name: "ADD", inputs: [1], outputs: [2], xnnpack_chain_id: 0, macs: 20, estimated_bytes: 32 },
    { index: 2, name: "SQUEEZE", inputs: [2], outputs: [3], xnnpack_chain_id: -1, macs: 0, estimated_bytes: 32 },
    { index: 3, name: "CONV_2D", inputs: [3], outputs: [], xnnpack_chain_id: -1, macs: 80, estimated_bytes: 32 },
  ],
  tensor_arena_plan: {
    schema: "deepbom.tensor_arena_plan.v1",
    status: "assessed",
    combined_arena_bytes: 32,
    allocations: [
      { tensor_index: 0, tensor_name: "input", allocation_status: "allocated", arena: "kTfLiteArenaRw", size_bytes: 16, offset_bytes: 0, first_node: 0, last_node: 0 },
      { tensor_index: 1, tensor_name: "t1", allocation_status: "allocated", arena: "kTfLiteArenaRw", size_bytes: 16, offset_bytes: 16, first_node: 0, last_node: 1 },
      { tensor_index: 2, tensor_name: "t2", allocation_status: "allocated", arena: "kTfLiteArenaRw", size_bytes: 16, offset_bytes: 0, first_node: 1, last_node: 2 },
      { tensor_index: 3, tensor_name: "t3", allocation_status: "allocated", arena: "kTfLiteArenaRw", size_bytes: 16, offset_bytes: 16, first_node: 2, last_node: 3 },
    ],
    aliases: [],
  },
};
const precisionAnalysis = {
  tensors: [
    { index: 0, dtype: "FLOAT16" },
    { index: 1, dtype: "UINT8" },
  ],
};
expectEqual(opPrecisionLabel({ inputs: [0], xnnpack_kernel_candidate: "XNNPACK FP16 AArch64 NEON FP16 arithmetic GEMM" }, precisionAnalysis), "FP16", "FP16 source candidates must not be labeled FP32.");
expectEqual(opPrecisionLabel({ inputs: [1], xnnpack_kernel_candidate: "XNNPACK QU8 AArch64 NEON MLAL GEMM" }, precisionAnalysis), "UINT8", "QU8 source candidates must retain unsigned precision semantics.");
const source = {
  schema: "deepbom.runtime_assignment.v1.1",
  artifact_sha256: sha,
  target_profile_id: "android_mid_a55",
  target_profile_sha256: profileSha,
  runtime: { name: "LiteRT", version: "2.0", backend: "XNNPACK", build: "release" },
  source: {
    kind: "interpreter_plan_export",
    collected_at: "2026-07-15T00:00:00.000Z",
    assignment_semantics: "original_graph_op_assignment",
    partition_semantics: "partition_id_identifies_runtime_partition_when_present",
    duration_semantics: "per_original_op_exclusive",
  },
  assignments: [
    { op_index: 0, op_name: "CONV_2D", provider: "XNNPACK", delegated: true, partition_id: 0, duration_us: 12.5 },
    { op_index: 1, op_name: "ADD", provider: "TFLite CPU", delegated: false, duration_us: 3 },
    { op_index: 2, op_name: "SQUEEZE", provider: "TFLite CPU", delegated: false, duration_us: 1 },
    { op_index: 3, op_name: "CONV_2D", provider: "XNNPACK", delegated: true, partition_id: 1, duration_us: 9.5 },
  ],
};

const parsed = parseRuntimeAssignmentDocument(JSON.stringify(source), analysis);
expectEqual(parsed.evidence_class, "OBSERVED_RUNTIME", "Imported assignment should use the runtime-observed evidence class.");
expectEqual(parsed.assignment_count, 4, "Imported assignment should preserve op coverage.");
expectEqual(parsed.coverage_ratio, 1, "Imported assignment should derive exact graph coverage.");
expectEqual(parsed.target_profile_sha256, profileSha, "Imported assignment should bind the exact target-profile content digest.");
expectEqual(parsed.assignments[0].kernel, null, "Generic runtime assignments must not fabricate an executed microkernel.");
expectEqual(parsed.comparison.placement_assessment.match_count, 2, "Runtime comparison should count matching delegate and CPU placements.");
expectEqual(parsed.comparison.placement_assessment.overpredicted_delegation_count, 1, "Runtime comparison should count static delegation overprediction.");
expectEqual(parsed.comparison.placement_assessment.underpredicted_delegation_count, 1, "Runtime comparison should count static delegation underprediction.");
expectEqual(parsed.comparison.placement_assessment.match_ratio, 0.5, "Runtime comparison placement agreement should be deterministic.");
expectEqual(parsed.comparison.mac_comparison.mismatch_macs, 100, "Runtime comparison should MAC-weight placement mismatches.");
expectEqual(parsed.comparison.observed_partitions.partition_count, 2, "Separated observed delegate regions should form two partitions.");
expectEqual(parsed.comparison.boundary_comparison.mismatch_count, 3, "Runtime comparison should detect under/overpredicted graph-edge boundaries.");
expectEqual(parsed.comparison.observed_boundary_inventory.edge_count, 2, "Observed delegate/CPU transitions should produce two confirmed boundary edges.");
expectEqual(parsed.comparison.observed_boundary_inventory.summed_edge_payload_bytes, 32, "Observed boundary payload should sum exact declared tensor bytes.");
expectEqual(parsed.comparison.duration_comparison.total_duration_us, 26, "Additive per-original-op durations should sum at full coverage.");
const reusedPartition = parseRuntimeAssignmentDocument(JSON.stringify({
  ...source,
  assignments: [
    source.assignments[0],
    source.assignments[1],
    source.assignments[2],
    { ...source.assignments[3], partition_id: 0 },
  ],
}), analysis);
expectEqual(reusedPartition.comparison.observed_partitions.partition_count, 1, "One explicit runtime partition ID reused across noncontiguous original-op indices must remain one partition.");
expectEqual(reusedPartition.comparison.observed_partitions.noncontiguous_partition_id_count, 1, "Noncontiguous reuse of an explicit runtime partition ID should be diagnosed separately.");
expectEqual(reusedPartition.comparison.observed_partitions.partitions[0].contiguous_segment_count, 2, "Explicit partition inventory should retain its graph-index segment count.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, artifact_sha256: "b".repeat(64) }), analysis), "artifact_sha256", "Importer should reject a different artifact.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, target_profile_id: "wasm_simd" }), analysis), "target_profile_id", "Importer should reject a different target profile.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, target_profile_sha256: "d".repeat(64) }), analysis), "target_profile_sha256", "Importer should reject different target-profile content.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, assignments: [...source.assignments, source.assignments[0]] }), analysis), "duplicate op_index", "Importer should reject duplicate op assignments.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, assignments: [{ ...source.assignments[0], delegated: "false" }] }), analysis), "must be boolean", "Importer should reject string booleans.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, assignments: [{ ...source.assignments[0], duration_us: -1 }] }), analysis), "non-negative", "Importer should reject negative durations.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, source: { ...source.source, duration_semantics: "per_node_guess" } }), analysis), "duration_semantics", "Importer should reject unknown duration semantics.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, runtime: { ...source.runtime, name: "REPLACE_WITH_RUNTIME_NAME" } }), analysis), "template placeholder", "Importer should reject unresolved runtime identity placeholders.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, source: { ...source.source, collected_at: null } }), analysis), "collected_at is required", "Current runtime evidence must include a collection timestamp.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, source: { ...source.source, collected_at: "2026-07-15T00:00:00" } }), analysis), "with a timezone", "Current runtime evidence timestamps must be timezone-qualified.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...source, source: { ...source.source, collected_at: "2026-02-31T00:00:00Z" } }), analysis), "with a timezone", "Current runtime evidence timestamps must contain a valid calendar date.");

const xnnpackCommit = "23a67314f7afdbb76191589ae090d82bf55afbfa";
const buildIdentifierSha = "e".repeat(64);
const runtimeMemorySnapshots = [{
  memory_snapshot_id: 0,
  non_persistent_arena_bytes: 64,
  persistent_arena_bytes: 0,
  combined_arena_bytes: 64,
  tensor_count: 5,
  execution_node_count: 4,
  allocation_count: 5,
  alias_count: 0,
  allocated_interval_bytes: 80,
  allocations: [
    { tensor_index: 0, arena: "kTfLiteArenaRw", offset_bytes: 0, size_bytes: 16, first_node: 0, last_node: 0 },
    { tensor_index: 1, arena: "kTfLiteArenaRw", offset_bytes: 16, size_bytes: 16, first_node: 0, last_node: 1 },
    { tensor_index: 2, arena: "kTfLiteArenaRw", offset_bytes: 0, size_bytes: 16, first_node: 1, last_node: 2 },
    { tensor_index: 3, arena: "kTfLiteArenaRw", offset_bytes: 16, size_bytes: 16, first_node: 2, last_node: 3 },
    { tensor_index: 4, arena: "kTfLiteArenaRw", offset_bytes: 32, size_bytes: 16, first_node: 1, last_node: 2 },
  ],
  aliases: [],
}];
const runtimeMemory = {
  schema: "deepbom.runtime_memory.v1",
  status: "assessed",
  evidence_class: "OBSERVED_RUNTIME",
  tensorflow_source_commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  snapshot_count: 1,
  peak_non_persistent_arena_bytes: 64,
  peak_persistent_arena_bytes: 0,
  peak_combined_arena_bytes: 64,
  final_non_persistent_arena_bytes: 64,
  final_persistent_arena_bytes: 0,
  final_combined_arena_bytes: 64,
  allocation_ledger_sha256: sha256TextHex(JSON.stringify(runtimeMemorySnapshots)),
  snapshots: runtimeMemorySnapshots,
  method: "fixture",
  interpretation_boundary: "TFLite arena buffers only.",
};
const nativeSource = {
  ...source,
  schema: "deepbom.runtime_assignment.v1.9",
  runtime: { ...source.runtime, build: "instrumented release", binary_sha256: "b".repeat(64) },
  source: {
    ...source.source,
    kind: "deepbom_native_runtime_capture",
    capture_id: "device-run-001",
    dispatch_sample_semantics: "unique_context_function_selection_per_process",
    collector: {
      schema: "deepbom.native_runtime_collector.v1.1",
      name: "deepbom-runtime-collector",
      version: "0.1.0",
      source_commit: `deepbom@${"d".repeat(40)}`,
      binary_sha256: "c".repeat(64),
      attestation_status: "not_attested",
      instrumentation: { lowering_ids: true, microkernel_ids: true, arena_allocations: true },
    },
  },
  selector_context: {
    schema: "deepbom.runtime_selector_context.v1.1",
    backend_library: "XNNPACK",
    device: {
      architecture: "aarch64",
      identity: "test-device",
      cpu_feature_source: "native_os_api",
      cpu_features: ["asimd", "fp", "neon"],
    },
    build: {
      runtime_binary_sha256: "b".repeat(64),
      xnnpack_source_commit: xnnpackCommit,
      microkernel_build_identifier_sha256: buildIdentifierSha,
      build_manifest_sha256: "a".repeat(64),
      compile_definitions: [
        { name: "XNN_BUILD_ALL_MICROKERNELS", value: "OFF" },
        { name: "XNN_ENABLE_ASSEMBLY", value: "ON" },
      ],
    },
    invocation: {
      inputs: [{ tensor_index: 0, name: "input", shape: [1, 4] }],
      thread_count: 1,
      runtime_options_sha256: "9".repeat(64),
    },
  },
  runtime_memory: runtimeMemory,
  assignments: source.assignments.map((item, index) => index === 0 ? {
    ...item,
    mapping_method: "native_runtime_original_op_instrumentation",
    lowering_id: "convolution_to_igemm",
    kernel_id: "f32-igemm-4x8-neonfma",
    kernel: "xnn_f32_igemm_minmax_ukernel_4x8__neonfma",
    kernel_source_ref: `google/XNNPACK@${xnnpackCommit}/src/f32-igemm/f32-igemm-4x8-neonfma.c`,
    kernel_build_identifier_sha256: buildIdentifierSha,
  } : item),
};
const nativeParsed = parseRuntimeAssignmentDocument(JSON.stringify(nativeSource), analysis, { fileSha256: "f".repeat(64) });
expectEqual(nativeParsed.source.dispatch_sample_semantics, "unique_context_function_selection_per_process", "Native selector evidence must disclose de-duplicated dispatch selection semantics.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, source: { ...nativeSource.source, dispatch_sample_semantics: null } }), analysis, { fileSha256: "f".repeat(64) }), "dispatch_sample_semantics", "Current native dispatch inventories must declare their sample semantics.");
expectEqual(nativeParsed.selector_observation.status, "partial_graph", "One fully instrumented op should produce partial-graph selector closure.");
expectEqual(nativeParsed.selector_observation.selector_ambiguity_closed_op_count, 1, "Only the op with all four observed dimensions should close selector ambiguity.");
expectEqual(nativeParsed.assignments[0].selector_evidence_class, "OBSERVED_MICROKERNEL", "Complete native kernel identity should be labeled observed microkernel evidence.");
expectEqual(JSON.stringify(nativeParsed.assignments[0].resolved_selector_dimensions), JSON.stringify(["runtime_architecture_identity", "compile_configuration", "lowering_shape", "runtime_dispatch"]), "Native selector evidence should close exactly four named dimensions.");
expectEqual(nativeParsed.selector_observation.collector_attestation_status, "not_attested", "Unsigned native collector output must remain explicitly unattested.");
expectEqual(nativeParsed.selector_observation.context_evidence_class, "OBSERVED_RUNTIME_UNATTESTED_EXPORT", "Native context must disclose that the imported collector export is unattested.");
const resourcePartition = {
  schema: "deepbom.resource_partition_observation.v1",
  evidence_class: "OBSERVED_OS_RESOURCE_PARTITION",
  observation_sha256: "6".repeat(64),
  requested_cpu_ids: [2, 3],
  affinity_mode: "taskset_process_and_descendants",
  isolation_expectation: "exclusive_cpuset",
  affinity_status: "observed_all_sampled_threads_within_requested_set",
  exclusive_isolation_status: "observed_cgroup_v2_isolated_partition",
  sample_count: 4,
  maximum_observed_thread_count: 2,
  sampled_threads: [
    { tid: 100, allowed_cpu_ids: [2, 3] },
    { tid: 101, allowed_cpu_ids: [2, 3] },
  ],
  observed_allowed_cpu_ids_union: [2, 3],
  observed_processor_ids: [2, 3],
  observed_effective_cpu_ids: [2, 3],
  cgroup_v2_path: "/deepbom.slice/capture.scope",
  cgroup_v2_partition_state: "isolated",
  online_cpu_ids: [0, 1, 2, 3],
  kernel_command_line: "quiet nohz_full=2-3 rcu_nocbs=2-3",
  kernel_isolation_parameters: { isolcpus: null, nohz_full: "2-3", irqaffinity: null, rcu_nocbs: "2-3" },
  cpu_frequency_policy: [],
  cache_shared_cpu_lists: [],
  interpretation_boundary: "Observed affinity and cgroup v2 partition identity do not establish runtime latency or absence of every source of interference.",
};
const partitionSource = structuredClone(nativeSource);
partitionSource.selector_context.invocation.resource_partition = resourcePartition;
const partitionParsed = parseRuntimeAssignmentDocument(JSON.stringify(partitionSource), analysis, { fileSha256: "6".repeat(64) });
expectEqual(partitionParsed.selector_context.invocation.resource_partition.requested_cpu_ids.join(","), "2,3", "Runtime CPU-set evidence should preserve the exact requested mask.");
expectEqual(partitionParsed.selector_context.invocation.resource_partition.exclusive_isolation_status, "observed_cgroup_v2_isolated_partition", "An exact isolated cpuset should remain distinct from affinity-only evidence.");
const exclusiveNotObserved = structuredClone(partitionSource);
exclusiveNotObserved.selector_context.invocation.resource_partition.exclusive_isolation_status = "not_observed_affinity_only";
exclusiveNotObserved.selector_context.invocation.resource_partition.cgroup_v2_partition_state = "member";
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(exclusiveNotObserved), analysis, { fileSha256: "6".repeat(64) }), "exclusive_cpuset request", "An unmet exclusive cpuset request must fail closed during import.");
const incompleteMask = structuredClone(partitionSource);
incompleteMask.selector_context.invocation.resource_partition.sampled_threads = [{ tid: 100, allowed_cpu_ids: [2] }];
incompleteMask.selector_context.invocation.resource_partition.observed_allowed_cpu_ids_union = [2];
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(incompleteMask), analysis, { fileSha256: "6".repeat(64) }), "exactly reproduce", "Affinity evidence must cover the complete requested CPU set rather than an unreported subset.");
const offlineRequest = structuredClone(partitionSource);
offlineRequest.selector_context.invocation.resource_partition.online_cpu_ids = [0, 1, 2];
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(offlineRequest), analysis, { fileSha256: "6".repeat(64) }), "online CPU set", "A requested offline CPU must invalidate the partition observation.");
const tensorflowCommit = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const delegateBuildSource = structuredClone(nativeSource);
delegateBuildSource.selector_context.build.tensorflow_source_commit = tensorflowCommit;
delegateBuildSource.selector_context.build.compile_definitions = [
  { name: "TFLITE_ENABLE_GPU", value: "ON" },
  { name: "TFLITE_ENABLE_NNAPI", value: "ON" },
  ...delegateBuildSource.selector_context.build.compile_definitions,
];
delegateBuildSource.tflite_delegate_build_inventory = {
  schema: "deepbom.tflite_delegate_build_inventory.v1",
  evidence_class: "DECLARED_BUILD_AND_RUNTIME_OPTION_INVENTORY",
  artifact_sha256: sha,
  tensorflow_source_commit: tensorflowCommit,
  runtime_binary_sha256: "b".repeat(64),
  build_manifest_sha256: "a".repeat(64),
  cmake_system_name: "Android",
  build_options: [
    { name: "TFLITE_ENABLE_GPU", declared_value: "ON", normalized_enabled: true, effective_status: "enabled_by_declared_cmake_option" },
    { name: "TFLITE_ENABLE_NNAPI", declared_value: "ON", normalized_enabled: true, effective_status: "enabled_by_declared_cmake_option_and_android_gate" },
  ],
  gpu: {
    compiled_status: "enabled_by_declared_cmake_option",
    experimental_flags: 1,
    quantized_model_flag_bit: 1,
    quantized_model_flag_status: "enabled_by_declared_runtime_option",
    max_delegated_partitions: 2,
    option_source: "capture runtime-options.json",
  },
  nnapi: {
    compiled_status: "enabled_by_declared_cmake_option_and_android_gate",
    runtime_feature_level: 31,
    accelerator_identity: "android-nnapi-default-device",
    capability_source: "android_nnapi_runtime_query",
  },
  source_files: [
    {
      id: "tflite_cmake_build_options",
      source_ref: `https://github.com/tensorflow/tensorflow/blob/${tensorflowCommit}/tensorflow/lite/CMakeLists.txt`,
      sha256: "bc8574b999dc15f8ce34939303afa70fa026ba2085f290e4f73c9a73163b7694",
    },
    {
      id: "tflite_gpu_delegate_options",
      source_ref: `https://github.com/tensorflow/tensorflow/blob/${tensorflowCommit}/tensorflow/lite/delegates/gpu/delegate_options.h`,
      sha256: "8db9e012233f6ca9f58de9acc5f8e351fbef4d29b4b852ca887a4e0c364abde1",
    },
  ],
  interpretation_boundary: "Declared build prerequisites remain runtime observations for assignment and execution.",
};
const delegateBuildParsed = parseRuntimeAssignmentDocument(JSON.stringify(delegateBuildSource), analysis, { fileSha256: "2".repeat(64) });
expectEqual(delegateBuildParsed.tflite_delegate_build_inventory.gpu.quantized_model_flag_status, "enabled_by_declared_runtime_option", "GPU quantized-model runtime option should remain bound to the imported build inventory.");
expectEqual(delegateBuildParsed.tflite_delegate_build_inventory.nnapi.compiled_status, "enabled_by_declared_cmake_option_and_android_gate", "NNAPI effective CMake gate should require Android and the enabled option.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({
  ...delegateBuildSource,
  tflite_delegate_build_inventory: {
    ...delegateBuildSource.tflite_delegate_build_inventory,
    runtime_binary_sha256: "0".repeat(64),
  },
}), analysis, { fileSha256: "2".repeat(64) }), "source binding", "TFLite delegate build inventory must fail closed on a runtime-binary mismatch.");
const multiLoweringDispatch = {
  lowering_id: "convolution_to_igemm",
  runtime_node_id: 0,
  compute_invocation_id: 0,
  kernel_id: "f32-igemm-4x8-neonfma",
  kernel: "xnn_f32_igemm_minmax_ukernel_4x8__neonfma",
  kernel_source_ref: `google/XNNPACK@${xnnpackCommit}/src/f32-igemm/f32-igemm-4x8-neonfma.c`,
  kernel_build_identifier_sha256: buildIdentifierSha,
  duration_us: null,
  duration_sum_us: null,
  sample_count: 1,
};
const multiLoweringSource = {
  ...nativeSource,
  assignments: nativeSource.assignments.map((item, index) => index === 0 ? {
    ...item,
    lowering_id: null,
    kernel_id: null,
    kernel: null,
    kernel_source_ref: null,
    kernel_build_identifier_sha256: null,
    lowerings: [
      { lowering_id: "convolution_to_igemm", runtime_node_id: 0, observation_count: 1 },
      { lowering_id: "static_reshape", runtime_node_id: 1, observation_count: 1 },
    ],
    dispatches: [multiLoweringDispatch],
  } : item),
};
const multiLoweringParsed = parseRuntimeAssignmentDocument(JSON.stringify(multiLoweringSource), analysis, { fileSha256: "1".repeat(64) });
expectEqual(multiLoweringParsed.assignments[0].lowerings.length, 2, "One original op should preserve both observed lowering rows.");
expectEqual(multiLoweringParsed.assignments[0].dispatches.length, 1, "One executed dispatch should remain available when an op has multiple lowering rows.");
expectEqual(multiLoweringParsed.assignments[0].selector_evidence_class, "OBSERVED_MICROKERNEL", "Dispatch-level lowering identity should close microkernel evidence without a fabricated singular top-level lowering.");
const rustDetectedFeatures = parseRuntimeAssignmentDocument(JSON.stringify({
  ...multiLoweringSource,
  selector_context: {
    ...multiLoweringSource.selector_context,
    device: {
      ...multiLoweringSource.selector_context.device,
      cpu_feature_source: "rust_std_runtime_detection",
      cpu_features: ["avx2", "avxvnni", "fma"],
    },
  },
}), analysis, { fileSha256: "3".repeat(64) });
expectEqual(rustDetectedFeatures.selector_context.device.cpu_feature_source, "rust_std_runtime_detection", "Native collector feature evidence should disclose Rust standard-library runtime detection.");
expectEqual(rustDetectedFeatures.selector_context.device.cpu_features.includes("avxvnni"), true, "AVX-VNNI should remain available as an explicit selector feature.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({
  ...multiLoweringSource,
  assignments: multiLoweringSource.assignments.map((item, index) => index === 0 ? {
    ...item,
    kernel_id: multiLoweringDispatch.kernel_id,
    kernel: multiLoweringDispatch.kernel,
    kernel_source_ref: multiLoweringDispatch.kernel_source_ref,
    kernel_build_identifier_sha256: multiLoweringDispatch.kernel_build_identifier_sha256,
  } : item),
}), analysis, { fileSha256: "1".repeat(64) }), "singular kernel_id", "Multiple lowering rows must not fabricate top-level singular microkernel fields.");
expectEqual(nativeParsed.selector_context.build.compile_definitions[0].value, "0", "XNNPACK false build switches should normalize to canonical zero.");
expectEqual(nativeParsed.selector_context.build.compile_definitions[1].value, "1", "XNNPACK true build switches should normalize to canonical one.");
expectEqual(nativeParsed.runtime_memory.peak_combined_arena_bytes, 64, "Native arena evidence should preserve the independently checked runtime peak.");
expectEqual(nativeParsed.arena_reconciliation.peak_delta_bytes, 32, "Runtime arena reconciliation should subtract the exact declared-shape projection.");
expectEqual(nativeParsed.arena_reconciliation.runtime_temporary_allocation_count, 1, "Runtime-only Prepare tensor indices should remain explicit runtime temporaries.");
expectEqual(nativeParsed.arena_reconciliation.runtime_temporary_interval_bytes, 16, "Runtime temporary interval bytes should sum exact observed allocation sizes.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, runtime_memory: { ...runtimeMemory, allocation_ledger_sha256: "0".repeat(64) } }), analysis, { fileSha256: "f".repeat(64) }), "ledger SHA-256", "Runtime arena import must recompute the canonical allocation ledger digest.");
const overlappingSnapshots = structuredClone(runtimeMemorySnapshots);
overlappingSnapshots[0].allocations[4].offset_bytes = 0;
const overlappingMemory = { ...runtimeMemory, snapshots: overlappingSnapshots, allocation_ledger_sha256: sha256TextHex(JSON.stringify(overlappingSnapshots)) };
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, runtime_memory: overlappingMemory }), analysis, { fileSha256: "f".repeat(64) }), "overlapping live allocations", "Runtime arena import must reject overlapping byte ranges with overlapping lifetimes.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, runtime_memory: null }), analysis, { fileSha256: "f".repeat(64) }), "emitted no runtime_memory evidence", "Declared arena instrumentation must fail closed when memory evidence is absent.");
const multiSelectorSource = {
  ...nativeSource,
  assignments: nativeSource.assignments.map((item, index) => index === 0 ? {
    ...item,
    lowering_id: null,
    kernel_id: null,
    kernel: null,
    kernel_source_ref: null,
    kernel_build_identifier_sha256: null,
    lowerings: [2, 10].map((runtimeNodeId) => ({ lowering_id: "convolution_to_igemm", runtime_node_id: runtimeNodeId, observation_count: 1 })),
    dispatches: [2, 10].map((runtimeNodeId) => ({
      lowering_id: "convolution_to_igemm",
      runtime_node_id: runtimeNodeId,
      compute_invocation_id: 0,
      kernel_id: `f32-igemm-node-${runtimeNodeId}`,
      kernel: `xnn_f32_igemm_minmax_ukernel_${runtimeNodeId}x8__neonfma`,
      kernel_source_ref: `google/XNNPACK@${xnnpackCommit}/src/f32-igemm/node-${runtimeNodeId}.c`,
      kernel_build_identifier_sha256: buildIdentifierSha,
      duration_us: null,
      duration_sum_us: null,
      sample_count: 1,
    })),
  } : item),
};
const multiSelectorParsed = parseRuntimeAssignmentDocument(JSON.stringify(multiSelectorSource), analysis, { fileSha256: "7".repeat(64) });
expectEqual(multiSelectorParsed.assignments[0].lowerings[1].runtime_node_id, 10, "Canonical runtime-node ordering must be numeric rather than lexicographic.");
expectEqual(multiSelectorParsed.assignments[0].dispatches.length, 2, "Multiple runtime dispatches must remain an inventory instead of fabricating one singular kernel.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, selector_context: null }), analysis, { fileSha256: "f".repeat(64) }), "selector_context", "Microkernel claims without a native selector context must be rejected.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, assignments: nativeSource.assignments.map((item, index) => index === 0 ? { ...item, kernel_build_identifier_sha256: "8".repeat(64) } : item) }), analysis, { fileSha256: "f".repeat(64) }), "kernel/build identifier mismatch", "Microkernel evidence must be bound to the captured XNNPACK build identifier.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, selector_context: { ...nativeSource.selector_context, device: { ...nativeSource.selector_context.device, architecture: "x86_64" } } }), analysis, { fileSha256: "f".repeat(64) }), "must be aarch64", "Native selector context architecture must agree with the selected target-profile family.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, selector_context: { ...nativeSource.selector_context, build: { ...nativeSource.selector_context.build, compile_definitions: nativeSource.selector_context.build.compile_definitions.map((item) => item.name === "XNN_ENABLE_ASSEMBLY" ? { ...item, value: "unknown" } : item) } } }), analysis, { fileSha256: "f".repeat(64) }), "explicit boolean build value", "Mandatory XNNPACK build switches must reject ambiguous values.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify({ ...nativeSource, assignments: nativeSource.assignments.map((item, index) => index === 0 ? { ...item, kernel: "not_a_microkernel" } : item) }), analysis, { fileSha256: "f".repeat(64) }), "concrete XNNPACK microkernel symbol", "Native selector context must reject non-XNNPACK kernel symbols.");
expectThrows(() => parseRuntimeAssignmentDocument(JSON.stringify(nativeSource), analysis), "capture file SHA-256", "Native selector context must be bound to the imported canonical capture bytes.");

const legacy = parseRuntimeAssignmentDocument(JSON.stringify({
  ...source,
  schema: "deepbom.runtime_assignment.v1",
  source: { kind: "legacy_interpreter_plan_export", collected_at: null },
}), analysis);
expectEqual(legacy.schema, "deepbom.runtime_assignment.v1.9", "Legacy runtime evidence should normalize to the current in-memory schema.");
expectEqual(legacy.source_schema, "deepbom.runtime_assignment.v1", "Normalized runtime evidence should retain its original schema identity.");
expectEqual(legacy.source.duration_semantics, "unspecified", "Legacy runtime evidence must not imply additive timing semantics.");
expectEqual(legacy.comparison.duration_comparison.total_duration_us, null, "Legacy runtime durations must not be summed without explicit semantics.");

const template = buildRuntimeAssignmentTemplate(analysis);
expectEqual(template.artifact_sha256, sha, "Runtime template should bind the active artifact.");
expectEqual(template.target_profile_sha256, profileSha, "Runtime template should bind exact target-profile content.");
expectEqual(template.schema, "deepbom.runtime_assignment.v1.9", "Runtime template should use the current schema with explicit timing and adapter semantics.");
expectEqual(template.source.duration_semantics, "not_collected", "Runtime template must not imply additive timing before measurements exist.");
expectEqual(template.graph_ops.length, analysis.ops.length, "Runtime template should include a non-evidentiary op reference inventory.");
expect(template.graph_ops.every((item) => item.reference_only === true && typeof item.predicted_delegated === "boolean"), "Runtime template op references must be explicitly non-evidentiary and expose the static placement prediction.");
expect(template.graph_ops.every((item) => Array.isArray(item.input_tensor_ids) && Array.isArray(item.output_tensor_ids)), "Runtime template op references should expose exact tensor topology needed to bind ModelRuntimeDetails exports.");
expectEqual(template.assignments.length, 0, "Runtime template must not contain placeholder rows that could be imported as observed evidence.");

const runtimeEvidence = buildRuntimeEvidence({ analysis, runtimeAssignmentEvidence: parsed });
expectEqual(runtimeEvidence.runtime_execution_status, "evidence_imported", "Assignment-only runtime evidence should not claim benchmark execution.");
expectEqual(runtimeEvidence.assessments.runtime_assignment.status, "assessed", "Assignment evidence should carry assessed status.");
expectEqual(runtimeEvidence.runtime_assignment.artifact_sha256, sha, "Runtime evidence should retain artifact binding.");

const sourceText = readFileSync("web/lib/kernel-inspector.js", "utf8");
expect(!sourceText.includes("innerHTML"), "Kernel inspector should render imported values with textContent only.");
done("Kernel inspector contract passed (artifact/target binding, op validation, and runtime evidence). ");
