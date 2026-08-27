import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  captureTensorRtParser,
  REPOSITORY_ROOT,
  tensorRtCollectorSourceIdentity,
} from "./tensorrt-capture-lib.mjs";

const source = await readFile(path.join(REPOSITORY_ROOT, "native/tensorrt_collector/src/main.cc"), "utf8");
const cmake = await readFile(path.join(REPOSITORY_ROOT, "native/tensorrt_collector/CMakeLists.txt"), "utf8");
for (const required of ["supportsModelV2", "getNbSubgraphs", "isSubgraphSupported", "getSubgraphNodes", "getNbErrors", "cudaSetDevice", "device-id", "LoadLibraryW", "dlopen"]) {
  assert(source.includes(required), `native collector must use ${required}`);
}
for (const prohibited of ["supportsOperator", "deserializeCudaEngine", "buildSerializedNetwork", "enqueueV3", "executeV2"]) {
  assert(!source.includes(prohibited), `parser-only collector must not use ${prohibited}`);
}
assert(cmake.includes("message(FATAL_ERROR") && cmake.includes("TENSORRT_ROOT"), "TensorRT SDK selection must be explicit");
assert(cmake.includes("OpenSSL REQUIRED") && cmake.includes("CUDAToolkit REQUIRED"), "collector dependencies must fail closed");
assert(cmake.includes("/WX") && cmake.includes("-Werror"), "collector warnings must fail the build");
assert.match((await tensorRtCollectorSourceIdentity()).source_set_sha256, /^[a-f0-9]{64}$/);

const root = await mkdtemp(path.join(os.tmpdir(), "deepbom-trt-test-"));
try {
  const modelPath = path.join(root, "fixture.onnx");
  const outputPath = path.join(root, "evidence.json");
  const mockPath = path.join(root, "mock-collector.mjs");
  await writeFile(modelPath, Buffer.from([0x08, 0x09, 0x12, 0x07, 0x66, 0x69, 0x78, 0x74, 0x75, 0x72, 0x65]));
  await writeFile(mockPath, `
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
const args = process.argv.slice(2); const values = new Map();
for (let i = 0; i < args.length; i += 2) values.set(args[i].slice(2), args[i + 1]);
const bytes = await readFile(values.get("model"));
const profileBytes = await readFile(values.get("profile"));
const sha = (value) => createHash("sha256").update(value).digest("hex");
process.stdout.write(JSON.stringify({
  schema: "deepbom.tensorrt_parser_observation.v1",
  artifact_sha256: sha(bytes),
  build_profile_sha256: values.get("profile-sha256"),
  build_profile_file_sha256: sha(profileBytes),
  build_profile: JSON.parse(profileBytes),
  execution_path: "native_tensorrt",
  tensorrt_version: "fixture-10.14.1",
  onnx_parser_version: 101401,
  cuda_version: "fixture-13.0",
  device_id: Number(values.get("device-id")),
  device_compute_capability: "8.7",
  device_identity: "fixture-device / CC 8.7",
  api_method: "supportsModelV2",
  subgraph_support_semantics: "per_subgraph_api_flag",
  parser_returned: true,
  collector: { binary_sha256: values.get("collector-binary-sha256"), source_set_sha256: values.get("collector-source-set-sha256"), git_commit: values.get("collector-git-commit"), git_state: values.get("collector-git-state") },
  subgraphs: [{ subgraph_index: 0, supported: true, sdk_reported_flag: true, node_indices: [0, 1] }],
  errors: [], collector_log: []
}));
`, "utf8");
  const profile = {
    execution_path: "native_tensorrt",
    expected_tensorrt_version: "10.14.1",
    expected_cuda_version: "13.0",
    device_id: 0,
    device_compute_capability: "8.7",
    precision: { tf32: true, fp16: true, bf16: false, int8: false, fp8: false },
    workspace_limit_bytes: 1073741824,
    builder_optimization_level: 3,
    dla_core: null,
    allow_gpu_fallback: false,
    calibration_cache_sha256: null,
    plugins: [],
    optimization_profiles: [],
  };
  const observation = await captureTensorRtParser({
    modelPath,
    profile,
    collectorPath: process.execPath,
    collectorCommand: process.execPath,
    collectorArgumentPrefix: [mockPath],
    outputPath,
  });
  assert.equal(observation.api_method, "supportsModelV2");
  assert.equal(observation.artifact_components.main.byte_length, 11);
  assert.equal(observation.artifact_components.external.length, 0);
  assert.equal(observation.subgraphs[0].node_indices.length, 2);
  assert.equal(JSON.parse(await readFile(outputPath, "utf8")).collector.source_files.length, 2);

  await assert.rejects(() => captureTensorRtParser({
    modelPath,
    profile: { ...profile, execution_path: "ort_tensorrt_ep", ort_ep_options: { provider_priority: 1, max_partition_iterations: 1000, min_subgraph_size: 1, engine_cache_enable: false, timing_cache_enable: false, context_memory_sharing_enable: false } },
    collectorPath: process.execPath,
    collectorCommand: process.execPath,
    collectorArgumentPrefix: [mockPath],
  }), /cannot stand in for ORT/);
  await assert.rejects(() => captureTensorRtParser({
    modelPath,
    profile,
    collectorPath: process.execPath,
    collectorCommand: process.execPath,
    collectorArgumentPrefix: [mockPath],
    externalComponents: [{ relative_path: "../escape.bin", source_path: modelPath }],
  }), /unsafe/);
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("TensorRT native collector pipeline passed (pinned API surface, prohibited execution APIs, source/binary/Git identity, isolated staging, profile verification, and mock capture/import)." );
