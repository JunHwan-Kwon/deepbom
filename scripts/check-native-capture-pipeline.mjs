import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { CAPTURE_MANIFEST_SCHEMA, CAPTURE_PACKAGE_SCHEMA, CAPTURE_RUN_SCHEMA, RUNTIME_ASSIGNMENT_SCHEMA, TENSORFLOW_SOURCE_COMMIT, XNNPACK_SOURCE_COMMIT, runNativeCapture, sha256File, stableJson, validateResourcePartitionObservation, validateRunConfig, verifyNativeCapturePackage } from "./native-capture-lib.mjs";
import { runCommand } from "./run-utils.mjs";

await runCommand("cargo", ["build", "--manifest-path", "native/runtime_collector/Cargo.toml"]);
await runCommand("cargo", ["build", "--manifest-path", "native/runtime_probe/Cargo.toml"]);

const pins = JSON.parse(await readFile("native/pins.json", "utf8"));
const runSchema = JSON.parse(await readFile("native/capture-run.schema.json", "utf8"));
assert.equal(pins.tensorflow.commit, TENSORFLOW_SOURCE_COMMIT);
assert.equal(pins.xnnpack.commit, XNNPACK_SOURCE_COMMIT);
assert.equal(pins.capture_contract.run_schema, CAPTURE_RUN_SCHEMA);
assert.equal(pins.capture_contract.manifest_schema, CAPTURE_MANIFEST_SCHEMA);
assert.equal(pins.capture_contract.package_schema, CAPTURE_PACKAGE_SCHEMA);
assert.equal(pins.capture_contract.assignment_schema, RUNTIME_ASSIGNMENT_SCHEMA);
assert.equal(runSchema.properties.schema.const, CAPTURE_RUN_SCHEMA);

const root = await mkdtemp(path.join(os.tmpdir(), "deepbom-native-capture-"));
try {
  const artifactPath = path.resolve("web/samples/mobilenet_v2_1.0_224_quant.tflite");
  const artifactSha256 = await sha256File(artifactPath);
  const buildIdPath = path.join(root, "microkernel-build-id.txt");
  const planPath = path.join(root, "probe-plan.json");
  const configPath = path.join(root, "capture-run.json");
  await writeFile(buildIdPath, "deepbom-contract-probe/xnnpack-build/v1\n");
  const baseEvent = {
    op_index: 0,
    op_name: "CONV_2D",
    provider: "XNNPACK",
    delegated: true,
    partition_id: "xnn-0",
    lowering_id: "convolution_to_igemm",
    kernel_id: "f32-igemm-4x8-scalar",
    kernel: "xnn_f32_igemm_minmax_ukernel_4x8__scalar",
    kernel_source_ref: `google/XNNPACK@${XNNPACK_SOURCE_COMMIT}/src/f32-igemm/gen/f32-igemm-4x8-minmax.c`,
    duration_us_samples: [10, 14],
  };
  await writeFile(planPath, stableJson({
    schema: "deepbom.native_contract_probe_plan.v1.1",
    artifact_sha256: artifactSha256,
    iterations: 3,
    events: [
      baseEvent,
      {
        op_index: 1,
        op_name: "SQUEEZE",
        provider: "TFLite CPU",
        delegated: false,
        partition_id: null,
        lowering_id: null,
        kernel_id: null,
        kernel: null,
        kernel_source_ref: null,
        duration_us_samples: [2, 3, 4],
      },
    ],
    memory_snapshots: [{
      non_persistent_arena_bytes: 4096,
      persistent_arena_bytes: 256,
      tensor_count: 3,
      execution_node_count: 2,
      allocations: [
        { tensor_index: 0, arena: "kTfLiteArenaRw", offset_bytes: 0, size_bytes: 1024, first_node: 0, last_node: 1 },
        { tensor_index: 1, arena: "kTfLiteArenaRwPersistent", offset_bytes: 0, size_bytes: 256, first_node: 0, last_node: null },
      ],
      aliases: [{ tensor_index: 2, shared_with_tensor_index: 0 }],
    }],
  }));
  const probeExecutable = process.platform === "win32" ? "deepbom-runtime-contract-probe.exe" : "deepbom-runtime-contract-probe";
  const collectorExecutable = process.platform === "win32" ? "deepbom-runtime-collector.exe" : "deepbom-runtime-collector";
  const collectorPath = path.resolve("native", "runtime_collector", "target", "debug", collectorExecutable);
  const probePath = path.resolve("native", "runtime_probe", "target", "debug", probeExecutable);
  const config = {
    schema: CAPTURE_RUN_SCHEMA,
    capture_mode: "synthetic_contract_probe",
    artifact_path: artifactPath,
    output_dir: "unused-output",
    target_profile: { id: "x86_avx2", sha256: "b".repeat(64) },
    runtime: {
      name: "DeepBOM Runtime Contract Probe",
      version: "0.1.0",
      backend: "Synthetic XNNPACK contract",
      build: "release",
      binary_path: probePath,
      arguments: [planPath, "${artifact_path}", "${events_path}"],
      environment: {},
    },
    source: {
      collected_at: "2026-07-16T00:00:00.000Z",
      capture_id: "deterministic-native-contract-probe",
      collector_source_commit: "e1c1529-test-fixture",
    },
    device: { identity: "local-contract-validation-host" },
    build: {
      source: { tensorflow_commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c", validation_kind: "synthetic_contract_probe" },
      xnnpack_source_commit: XNNPACK_SOURCE_COMMIT,
      microkernel_build_identifier_path: buildIdPath,
      compile_definitions: [
        { name: "XNN_BUILD_ALL_MICROKERNELS", value: "1" },
        { name: "XNN_ENABLE_ASSEMBLY", value: "1" },
      ],
      toolchain: { compiler: "rustc", purpose: "collector contract validation only" },
    },
    invocation: {
      inputs: [{ tensor_index: 0, name: "input", shape: [1, 224, 224, 3] }],
      thread_count: 1,
      runtime_options: { iterations: 3, synthetic: true },
    },
    instrumentation: { lowering_ids: true, microkernel_ids: true, arena_allocations: true },
  };
  const affinityConfig = structuredClone(config);
  affinityConfig.invocation.resource_partition = {
    requested_cpu_ids: [2, 3],
    affinity_mode: "taskset_process_and_descendants",
    isolation_expectation: "affinity_only",
  };
  validateRunConfig(affinityConfig);
  const unsortedCpuConfig = structuredClone(affinityConfig);
  unsortedCpuConfig.invocation.resource_partition.requested_cpu_ids = [3, 2];
  assert.throws(() => validateRunConfig(unsortedCpuConfig), /unique, ascending/);
  const oversubscribedConfig = structuredClone(affinityConfig);
  oversubscribedConfig.invocation.thread_count = 3;
  assert.throws(() => validateRunConfig(oversubscribedConfig), /must not exceed/);
  const invalidExpectationConfig = structuredClone(affinityConfig);
  invalidExpectationConfig.invocation.resource_partition.isolation_expectation = "isolcpus_assumed";
  assert.throws(() => validateRunConfig(invalidExpectationConfig), /affinity_only or exclusive_cpuset/);
  const resourceObservation = resourcePartitionFixture();
  assert.equal(validateResourcePartitionObservation(
    resourceObservation,
    affinityConfig.invocation.resource_partition,
    2,
  ), true);
  const tamperedSampleLedger = structuredClone(resourceObservation);
  tamperedSampleLedger.affinity_samples[1].threads[1].allowed_cpu_ids = [2];
  assert.throws(
    () => validateResourcePartitionObservation(tamperedSampleLedger, affinityConfig.invocation.resource_partition, 2),
    /does not reproduce the affinity sample ledger/,
    "raw affinity samples must independently reproduce the emitted summaries",
  );
  await writeFile(configPath, stableJson(config));

  const captureOptions = { collectorPath, skipCollectorBuild: true };
  const first = await runNativeCapture(configPath, { ...captureOptions, outputDir: path.join(root, "capture-a") });
  const second = await runNativeCapture(configPath, { ...captureOptions, outputDir: path.join(root, "capture-b") });
  await verifyNativeCapturePackage(first.outputDir);
  await verifyNativeCapturePackage(second.outputDir);
  assert.deepEqual(first.assignment, second.assignment, "repeated contract captures must produce identical evidence");
  assert.equal(first.index.importable_runtime_evidence, false);
  assert.equal(first.index.schema, CAPTURE_PACKAGE_SCHEMA);
  assert.equal(first.assignment.source.kind, "deepbom_native_runtime_contract_probe");
  assert.equal(first.assignment.source.validation_scope, "collector_contract_only_not_runtime_evidence");
  assert.equal(first.assignment.assignments[0].sample_count, 3);
  assert.equal(first.assignment.assignments[0].duration_sum_us, 34);
  assert.ok(Math.abs(first.assignment.assignments[0].duration_us - (34 / 3)) < 1e-12);
  assert.equal(first.assignment.assignments[1].duration_sum_us, 9);
  assert.equal(first.assignment.runtime_memory.schema, "deepbom.runtime_memory.v1");
  assert.equal(first.assignment.runtime_memory.peak_combined_arena_bytes, 4352);
  assert.equal(first.assignment.runtime_memory.snapshots[0].allocation_count, 2);
  assert.equal(first.assignment.runtime_memory.snapshots[0].alias_count, 1);

  const assignmentText = await readFile(path.join(first.outputDir, "runtime-assignment.json"), "utf8");
  const assignmentSha = await sha256File(path.join(first.outputDir, "runtime-assignment.json"));
  const analysis = {
    model_sha256: artifactSha256,
    target_profile: { id: "x86_avx2", profile_sha256: "b".repeat(64) },
    inputs: [{ index: 0, name: "input", shape: [1, 224, 224, 3] }],
    tensors: [{ index: 0, name: "input", shape: [1, 224, 224, 3], dtype: "UINT8" }],
    ops: [
      { index: 0, name: "CONV_2D", inputs: [0], outputs: [], macs: 1, estimated_bytes: 1 },
      { index: 1, name: "SQUEEZE", inputs: [], outputs: [], macs: 0, estimated_bytes: 0 },
    ],
  };
  assert.throws(
    () => parseRuntimeAssignmentDocument(assignmentText, analysis, { fileSha256: assignmentSha }),
    /source\.collector is valid only for deepbom_native_runtime_capture/,
    "browser importer must reject synthetic contract-probe output as runtime evidence",
  );

  const tamperedAssignmentPath = path.join(second.outputDir, "runtime-assignment.json");
  await writeFile(tamperedAssignmentPath, `${await readFile(tamperedAssignmentPath, "utf8")} `);
  await assert.rejects(
    verifyNativeCapturePackage(second.outputDir),
    /hash mismatch for runtime-assignment\.json/,
    "package verifier must reject post-collection evidence modification",
  );

  const missingNoFusionConfigPath = path.join(root, "missing-no-fusion-run.json");
  await writeFile(missingNoFusionConfigPath, stableJson({ ...config, capture_mode: "runtime_capture" }));
  await assert.rejects(
    runNativeCapture(missingNoFusionConfigPath, { ...captureOptions, outputDir: path.join(root, "capture-missing-no-fusion") }),
    /microkernel attribution capture requires DEEPBOM_XNN_NO_OPERATOR_FUSION=1/,
    "microkernel attribution must fail closed when operator fusion can obscure the original-op mapping",
  );

  const mislabeledConfigPath = path.join(root, "mislabeled-probe-run.json");
  await writeFile(mislabeledConfigPath, stableJson({
    ...config,
    capture_mode: "runtime_capture",
    runtime: { ...config.runtime, environment: { DEEPBOM_XNN_NO_OPERATOR_FUSION: "1" } },
    invocation: {
      ...config.invocation,
      runtime_options: { ...config.invocation.runtime_options, DEEPBOM_XNN_NO_OPERATOR_FUSION: "1" },
    },
  }));
  await assert.rejects(
    runNativeCapture(mislabeledConfigPath, { ...captureOptions, outputDir: path.join(root, "capture-mislabeled") }),
    /synthetic contract probe cannot produce runtime_capture evidence/,
    "official orchestrator must not label the contract probe as observed runtime evidence",
  );

  const conflictingPlanPath = path.join(root, "conflicting-plan.json");
  await writeFile(conflictingPlanPath, stableJson({
    schema: "deepbom.native_contract_probe_plan.v1.1",
    artifact_sha256: artifactSha256,
    iterations: 1,
    events: [baseEvent, { ...baseEvent, lowering_id: "conflicting_lowering" }],
  }));
  const conflictingConfigPath = path.join(root, "conflicting-run.json");
  await writeFile(conflictingConfigPath, stableJson({
    ...config,
    runtime: { ...config.runtime, arguments: [conflictingPlanPath, "${artifact_path}", "${events_path}"] },
  }));
  await assert.rejects(
    runNativeCapture(conflictingConfigPath, { ...captureOptions, outputDir: path.join(root, "capture-conflict") }),
    /runtime evidence collector failed with exit code 1/,
    "collector must reject conflicting selector identity for one original op",
  );

  console.log("Native capture pipeline check passed (deterministic replay, exact aggregation, package tamper rejection, browser isolation, conflict rejection).");
} finally {
  const resolvedRoot = path.resolve(root);
  const resolvedTemp = path.resolve(os.tmpdir());
  if (!resolvedRoot.startsWith(`${resolvedTemp}${path.sep}`) || !path.basename(resolvedRoot).startsWith("deepbom-native-capture-")) {
    throw new Error(`refusing to clean unexpected path ${resolvedRoot}`);
  }
  await rm(resolvedRoot, { recursive: true, force: true });
}

function resourcePartitionFixture() {
  const threads = (processorA, processorB) => [
    { tid: 100, allowed_cpu_ids: [2, 3], processor: processorA },
    { tid: 101, allowed_cpu_ids: [2, 3], processor: processorB },
  ];
  return {
    schema: "deepbom.resource_partition_observation.v1",
    evidence_class: "OBSERVED_OS_RESOURCE_PARTITION",
    requested_cpu_ids: [2, 3],
    affinity_mode: "taskset_process_and_descendants",
    isolation_expectation: "affinity_only",
    affinity_status: "observed_all_sampled_threads_within_requested_set",
    exclusive_isolation_status: "not_observed_affinity_only",
    sample_count: 2,
    maximum_observed_thread_count: 2,
    sampled_threads: [
      { tid: 100, allowed_cpu_ids: [2, 3] },
      { tid: 101, allowed_cpu_ids: [2, 3] },
    ],
    observed_allowed_cpu_ids_union: [2, 3],
    observed_processor_ids: [2, 3],
    observed_effective_cpu_ids: [0, 1, 2, 3],
    cgroup_v2_path: "/user.slice/test.scope",
    cgroup_v2_partition_state: "member",
    online_cpu_ids: [0, 1, 2, 3],
    kernel_command_line: "quiet",
    kernel_isolation_parameters: {},
    cpu_frequency_policy: [],
    cache_shared_cpu_lists: [],
    affinity_samples: [
      { observed_at_unix_ms: 1000, threads: threads(2, 3) },
      { observed_at_unix_ms: 1020, threads: threads(3, 2) },
    ],
    interpretation_boundary: "fixture",
  };
}
