import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { loadCliInput, loadOnnxExternalData } from "../bin/deepbom-input.mjs";
import { createTensorRtBuildProfile, TENSORRT_PARSER_OBSERVATION_SCHEMA } from "../web/lib/tensorrt-static-preflight.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const cases = [
  ["web/samples/mobilenet_v2_1.0_224_quant.tflite", "tflite"],
  ["web/samples/sample_cnn_float.onnx", "onnx"],
  ["web/samples/tinymqa1m.Q4_0.gguf", "gguf"],
  ["web/samples/nanofable-1m-fp16.safetensors", "safetensors"],
  ["web/samples/MNISTClassifier.mlmodel", "coreml"],
];

const emptyExecuTorchPtd = Buffer.from(
  "FAAAAEZUMDEAAAoAEAAEAAgADAAKAAAAAQAAAAgAAAAUAAAAAQAAAAgAAAAEAAQABAAAAAEAAAAMAAAAAAAGAAgABAAGAAAABAAAAAsAAAB3ZWlnaHRzLmJpbgA=",
  "base64",
);

const streamedGgufInput = await loadCliInput(path.resolve(cases[2][0]));
assert.equal(streamedGgufInput.kind, "file", "GGUF CLI input kind");
assert.equal(Object.hasOwn(streamedGgufInput, "bytes"), false, "GGUF input must remain disk-backed before analysis");
assert.equal(streamedGgufInput.prefix.byteLength <= 4096, true, "CLI format sniff is bounded");

for (const [artifact, expectedFormat] of cases) {
  const result = run(["audit", artifact, "--compact"]);
  const document = JSON.parse(result.stdout);
  assert.equal(document.format, expectedFormat, `${artifact} format`);
  assert.equal(document.filename, path.basename(artifact), `${artifact} filename`);
  assert.match(document.model_sha256, /^[a-f0-9]{64}$/, `${artifact} SHA-256`);
  assert.equal(document.file_size_bytes > 0, true, `${artifact} byte size`);
}

const gguf = JSON.parse(run(["gguf", cases[2][0], "--compact"]).stdout);
assert.equal(gguf.gguf?.tensor_count > 0, true, "GGUF command tensor inventory");
const ggufScenario = JSON.parse(run(["gguf", cases[2][0], "--context", "8192", "--batch", "2", "--state-bits", "8", "--memory-mib", "1", "--compact"]).stdout);
assert.equal(ggufScenario.cli_context_scenario?.context_length, 8192, "GGUF context scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.batch_size, 2, "GGUF batch scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.state_storage_bits, 8, "GGUF state-width scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.context_source, "cli_argument", "GGUF context scenario source");
assert.match(ggufScenario.cli_context_scenario?.memory_feasibility?.status, /^lower_bound_(exceeds_capacity|at_or_below_capacity_fit_unresolved)$/, "GGUF memory-capacity lower-bound classification");
assert.match(ggufScenario.cli_context_scenario?.memory_feasibility?.residency_assumption, /simultaneously resident/, "GGUF memory scenario emits its residency assumption");
assert.equal(ggufScenario.cli_context_scenario?.memory_feasibility?.fit_claim, "not_emitted", "GGUF memory scenario must not emit a fit claim");
assert.equal(BigInt(ggufScenario.cli_context_scenario.memory_feasibility.static_lower_bound_bytes.decimal),
  BigInt(ggufScenario.cli_context_scenario.memory_feasibility.serialized_weight_floor_bytes.decimal)
    + BigInt(ggufScenario.cli_context_scenario.memory_feasibility.logical_kv_state_bytes.decimal), "GGUF memory lower-bound conservation");
assert.equal(ggufScenario.gguf?.semantic_contract?.context_length, gguf.gguf?.semantic_contract?.context_length, "GGUF scenario must not mutate serialized context");
assert.match(run(["gguf", "--help"]).stdout, /--context <tokens>/, "subcommand help");
assert.match(run(["gguf", "--help"]).stdout, /--memory-mib <MiB>/, "GGUF memory scenario help");
assert.match(run(["gguf", "--help"]).stdout, /--llm-memory-profile <json>/, "LLM static pool profile help");

const ggufScenarioCycloneDx = JSON.parse(run(["gguf", cases[2][0], "--context", "8192", "--batch", "2", "--state-bits", "8", "--memory-mib", "1", "--format", "cyclonedx", "--timestamp", "2026-08-18T00:00:00.000Z", "--compact"]).stdout);
const ggufScenarioProperties = new Map(ggufScenarioCycloneDx.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioContextLength"), "8192", "CycloneDX retains CLI context scenario");
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioBatchSize"), "2", "CycloneDX retains CLI batch scenario");
assert.match(ggufScenarioProperties.get("deepbom:model:llmCliScenarioResidencyAssumption"), /simultaneously resident/, "CycloneDX retains CLI residency assumption");
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioMemoryFitClaim"), "not_emitted", "CycloneDX retains lower-bound-only memory claim boundary");

const timestamp = "2026-08-18T00:00:00.000Z";
const cyclonedx = JSON.parse(run([
  "audit",
  cases[1][0],
  "--format",
  "cyclonedx",
  "--timestamp",
  timestamp,
  "--compact",
]).stdout);
assert.equal(cyclonedx.bomFormat, "CycloneDX");
assert.equal(cyclonedx.specVersion, "1.7");
assert.equal(cyclonedx.metadata.timestamp, timestamp);

const temp = await mkdtemp(path.join(tmpdir(), "deepbom-cli-tensorrt-"));
try {
  assert.equal(gguf.on_device_llm?.storage?.layer_storage?.status, "assessed_exact_serialized_layer_storage", "GGUF CLI fixture exact layer ledger");
  const ggufSerializedBytes = BigInt(gguf.on_device_llm.storage.serialized_tensor_bytes_decimal);
  const memoryProfilePath = path.join(temp, "memory-profile.json");
  await writeFile(memoryProfilePath, JSON.stringify({
    schema: "deepbom.llm_static_memory_profile.v1",
    artifact: { format: "gguf", sha256: gguf.model_sha256 },
    capacities: { cpu_bytes: String(ggufSerializedBytes * 4n), accelerator_bytes: String(ggufSerializedBytes * 4n) },
    reserves: { cpu_bytes: "0", accelerator_bytes: "0" },
    policy: {
      layer_order: "highest_index_first", non_layer_pool: "cpu", state_pool: "accelerator",
      context_length: gguf.on_device_llm.architecture.context_length, batch_size: 1, state_storage_bits: 16,
    },
  }), "utf8");
  const placedGguf = JSON.parse(run(["gguf", cases[2][0], "--llm-memory-profile", memoryProfilePath, "--compact"]).stdout);
  assert.equal(placedGguf.on_device_llm.static_memory_placement.status, "assessed_lower_bound_candidates", "GGUF conditional static pool placement");
  assert.equal(placedGguf.on_device_llm.static_memory_placement.maximum_accelerator_layer_count_not_disproven,
    placedGguf.on_device_llm.architecture.layer_count, "GGUF static pool candidate enumeration");
  assert.equal(placedGguf.on_device_llm.static_memory_placement.fit_claim, "not_emitted", "GGUF static pool placement must not claim fit");

  const ptdPath = path.join(temp, "weights.ptd");
  await writeFile(ptdPath, emptyExecuTorchPtd);
  const ptd = JSON.parse(run(["audit", ptdPath, "--compact"]).stdout);
  assert.equal(ptd.format, "executorch", "ExecuTorch PTD CLI format");
  assert.equal(ptd.executorch_container, "ptd", "ExecuTorch PTD container identity");
  assert.equal(ptd.tensor_count, 1, "ExecuTorch PTD tensor inventory");
  const ptdCycloneDx = JSON.parse(run(["audit", ptdPath, "--format", "cyclonedx", "--compact"]).stdout);
  assert.equal(ptdCycloneDx.bomFormat, "CycloneDX", "ExecuTorch PTD CycloneDX export");
  assert.match(ptdCycloneDx.serialNumber, /^urn:uuid:/, "ExecuTorch PTD CycloneDX serial number");

  const externalDataPath = path.join(temp, "weights.bin");
  await writeFile(externalDataPath, "external-weight-bytes", "utf8");
  const externalData = await loadOnnxExternalData(path.join(temp, "model.onnx"), {
    onnx_external_data: { tensors: [{ normalized_location: "weights.bin" }] },
  });
  assert.equal(externalData.length, 1, "ONNX external-data discovery");
  assert.equal(externalData[0].path, "weights.bin", "ONNX external-data canonical model-relative path");
  assert.equal(externalData[0].bytes.byteLength, 21, "ONNX external-data byte length");
  assert.equal(externalData[0].sha256, sha256TextHex("external-weight-bytes"), "ONNX external-data SHA-256");
  await assert.rejects(
    loadOnnxExternalData(path.join(temp, "model.onnx"), {
      onnx_external_data: { tensors: [{ normalized_location: "../weights.bin" }] },
    }),
    /not a safe relative path/,
    "ONNX external-data traversal must fail closed",
  );

  const onnxAnalysis = JSON.parse(run(["audit", cases[1][0], "--compact"]).stdout);
  const trtConfig = {
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
    optimization_profiles: [],
  };
  const trtProfile = createTensorRtBuildProfile(trtConfig);
  const profilePath = path.join(temp, "profile.json");
  await writeFile(profilePath, JSON.stringify(trtConfig), "utf8");
  const configured = JSON.parse(run(["audit", cases[1][0], "--tensorrt-profile", profilePath, "--compact"]).stdout);
  assert.equal(configured.tensorrt_static_preflight.status, "configuration_valid_parser_observation_required");
  assert.equal(configured.tensorrt_static_preflight.build_profile.profile_sha256, trtProfile.profile_sha256);
  const evidencePath = path.join(temp, "parser.json");
  await writeFile(evidencePath, JSON.stringify({
    schema: TENSORRT_PARSER_OBSERVATION_SCHEMA,
    artifact_sha256: onnxAnalysis.model_sha256,
    build_profile_sha256: trtProfile.profile_sha256,
    build_profile_file_sha256: sha256TextHex(`${canonicalJson(trtProfile)}\n`),
    build_profile: trtProfile,
    execution_path: "native_tensorrt",
    tensorrt_version: "10.14.1",
    cuda_version: "13.0",
    device_id: 0,
    device_compute_capability: "8.7",
    device_identity: "CLI fixture CC 8.7",
    api_method: "supportsModelV2",
    subgraph_support_semantics: "per_subgraph_api_flag",
    parser_returned: true,
    collector: {
      binary_sha256: "b".repeat(64),
      source_set_sha256: "c".repeat(64),
      git_commit: "cli-fixture",
      git_state: "clean",
    },
    plugins: [],
    subgraphs: [{ subgraph_index: 0, supported: true, sdk_reported_flag: true, node_indices: Array.from({ length: onnxAnalysis.ops.length }, (_, index) => index) }],
    errors: [],
  }), "utf8");
  const observed = JSON.parse(run(["audit", cases[1][0], "--tensorrt-parser-evidence", evidencePath, "--compact"]).stdout);
  assert.equal(observed.tensorrt_static_preflight.status, "parser_observed_all_supported");
  assert.equal(observed.tensorrt_static_preflight.projection.state_counts.CONDITIONALLY_ELIGIBLE, onnxAnalysis.ops.length);
  const wrongFormatTrt = run(["audit", cases[0][0], "--tensorrt-profile", profilePath], false);
  assert.notEqual(wrongFormatTrt.status, 0);
  assert.match(wrongFormatTrt.stderr, /apply only to ONNX/);
  const trtLlmConfigPath = path.join(temp, "tensorrt-llm.json");
  await writeFile(trtLlmConfigPath, JSON.stringify({
    version: "1.2.0",
    pretrained_config: {
      architecture: "LlamaForCausalLM", dtype: "float16", hidden_size: 8, intermediate_size: 16,
      num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 1, head_size: 4,
      vocab_size: 32, max_position_embeddings: 64,
      mapping: { world_size: 1, tp_size: 1, pp_size: 1, cp_size: 1 },
      quantization: { quant_algo: null, kv_cache_quant_algo: null, group_size: 128, has_zero_point: false, exclude_modules: [] },
    },
    build_config: {
      max_input_len: 32, max_seq_len: 64, max_batch_size: 1, max_beam_width: 1,
      max_num_tokens: 64, opt_num_tokens: 32, kv_cache_type: "PAGED", strongly_typed: true,
      weight_streaming: false, plugin_config: { paged_kv_cache: true },
    },
  }), "utf8");
  const trtLlm = JSON.parse(run(["audit", cases[3][0], "--tensorrt-llm-config", trtLlmConfigPath, "--compact"]).stdout);
  assert.equal(trtLlm.on_device_llm.tensorrt_llm.status, "candidate_configuration_unbound");
  assert.equal(trtLlm.on_device_llm.tensorrt_llm.kv_cache_scenario.logical_bytes.decimal, "2048");
  const wrongTrtLlmFormat = run(["audit", cases[1][0], "--tensorrt-llm-config", trtLlmConfigPath], false);
  assert.notEqual(wrongTrtLlmFormat.status, 0);
  assert.match(wrongTrtLlmFormat.stderr, /only to a SafeTensors artifact/);
  const wrongExternalDataFormat = run(["audit", cases[0][0], "--external-data-dir", temp], false);
  assert.notEqual(wrongExternalDataFormat.status, 0);
  assert.match(wrongExternalDataFormat.stderr, /applies only to an ONNX or ExecuTorch PTE file/);
} finally {
  await rm(temp, { recursive: true, force: true });
}

const first = run(["audit", cases[0][0], "--compact"]).stdout;
const second = run(["audit", cases[0][0], "--compact"]).stdout;
assert.equal(first, second, "TFLite analysis JSON must be deterministic");

const wrongCommand = run(["gguf", cases[1][0]], false);
assert.notEqual(wrongCommand.status, 0);
assert.match(wrongCommand.stderr, /requires a GGUF artifact/);
const invalidContext = run(["audit", cases[1][0], "--context", "8192"], false);
assert.notEqual(invalidContext.status, 0);
assert.match(invalidContext.stderr, /valid only for GGUF/);
const invalidStateBits = run(["gguf", cases[2][0], "--context", "8192", "--state-bits", "12"], false);
assert.notEqual(invalidStateBits.status, 0);
assert.match(invalidStateBits.stderr, /must be 8, 16, or 32/);
const orphanMemoryBudget = run(["gguf", cases[2][0], "--memory-mib", "1024"], false);
assert.notEqual(orphanMemoryBudget.status, 0);
assert.match(orphanMemoryBudget.stderr, /require --context/);

console.log("CLI checks passed (six formats, ONNX/ExecuTorch external-data contracts, GGUF context/memory scenario, TensorRT profile/parser evidence, CycloneDX projection, deterministic TFLite output, fail-closed routing).");

function run(args, expectSuccess = true) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  if (expectSuccess && result.status !== 0) {
    throw new Error(`CLI failed: ${args.join(" ")}\n${result.stderr}`);
  }
  return result;
}
