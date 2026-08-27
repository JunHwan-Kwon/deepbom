import assert from "node:assert/strict";
import { buildTensorRtLlmContract, validateTensorRtLlmContract } from "../web/lib/tensorrt-llm-contract.js";
import { TENSORRT_LLM_SOURCE_METADATA } from "../web/lib/tensorrt-llm-source-metadata.js";

const artifactSha = "a".repeat(64);
const configSha = "b".repeat(64);
const planSha = "c".repeat(64);
const bindingSha = "d".repeat(64);
const analysis = {
  format: "safetensors",
  model_sha256: "e".repeat(64),
  artifact_bundle: {
    bundle_sha256: artifactSha,
    model_source_sha256: artifactSha,
    files: [
      { path: "tensorrt_llm_engine_config.json", sha256: configSha, required: true },
      { path: "rank0.engine", sha256: planSha, required: false },
      { path: "deepbom.tensorrt-llm.json", sha256: bindingSha, required: true },
    ],
  },
};
const llm = {
  architecture: {
    family: "mixtral",
    hidden_size: 8,
    intermediate_size: 12,
    layer_count: 5,
    attention_head_count: 2,
    kv_head_count: 1,
    head_width: 4,
    vocabulary_size: 16,
    context_length: 32,
  },
};
const config = {
  version: "1.2.0",
  pretrained_config: {
    architecture: "MixtralForCausalLM",
    dtype: "float16",
    hidden_size: 8,
    intermediate_size: 12,
    num_hidden_layers: 5,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_size: 4,
    vocab_size: 16,
    max_position_embeddings: 32,
    mapping: { world_size: 4, tp_size: 2, pp_size: 2, cp_size: 1, pp_partition: null },
    quantization: { quant_algo: "W4A16_AWQ", kv_cache_quant_algo: "FP8", group_size: 128, has_zero_point: false, exclude_modules: [] },
  },
  build_config: {
    max_input_len: 16,
    max_seq_len: 32,
    max_batch_size: 2,
    max_beam_width: 1,
    max_num_tokens: 64,
    opt_num_tokens: 32,
    kv_cache_type: "PAGED",
    strongly_typed: true,
    weight_streaming: true,
    plugin_config: { gpt_attention_plugin: "float16", paged_kv_cache: true },
  },
};
const engineConfig = { document: config, path: "tensorrt_llm_engine_config.json", byte_length: 2048, sha256: configSha };
const candidate = buildTensorRtLlmContract(analysis, llm, { engineConfig });
assert.equal(candidate.status, "candidate_configuration_unbound");
assert.deepEqual(candidate.parallelism.layer_partition_per_pipeline_rank, [3, 2]);
assert.equal(candidate.parallelism.layer_partition_conservation, "complete");
assert.equal(candidate.parallelism.weight_bytes_per_rank, null);
assert.equal(candidate.kv_cache_scenario.logical_elements_decimal, "2560");
assert.equal(candidate.kv_cache_scenario.logical_bytes.decimal, "2560");
assert.equal(candidate.build_limits.weight_streaming, true);
assert.match(candidate.contract_sha256, /^[a-f0-9]{64}$/);

const binding = {
  document: {
    schema: "deepbom.tensorrt_llm_artifact_binding.v1",
    source_artifact_sha256: artifactSha,
    engine_config_sha256: configSha,
    engine_files: [{ path: "rank0.engine", sha256: planSha }],
  },
  path: "deepbom.tensorrt-llm.json",
  sha256: bindingSha,
};
const bound = buildTensorRtLlmContract(analysis, llm, { engineConfig, binding });
assert.equal(bound.status, "artifact_bound_configuration");
assert.equal(bound.artifact_binding.engine_file_count, 1);
assert.deepEqual(validateTensorRtLlmContract(analysis, { tensorrt_llm: bound }), []);

const badParallel = structuredClone(config);
badParallel.pretrained_config.mapping.world_size = 8;
const invalidParallel = buildTensorRtLlmContract(analysis, llm, { engineConfig: { ...engineConfig, document: badParallel } });
assert.equal(invalidParallel.status, "invalid");
assert(invalidParallel.issues.some((row) => row.code === "parallel_world_size_mismatch"));
const badArchitecture = structuredClone(config);
badArchitecture.pretrained_config.hidden_size = 16;
const invalidArchitecture = buildTensorRtLlmContract(analysis, llm, { engineConfig: { ...engineConfig, document: badArchitecture } });
assert(invalidArchitecture.issues.some((row) => row.code === "source_architecture_mismatch"));
const badBinding = structuredClone(binding);
badBinding.document.source_artifact_sha256 = "f".repeat(64);
const invalidBinding = buildTensorRtLlmContract(analysis, llm, { engineConfig, binding: badBinding });
assert(invalidBinding.issues.some((row) => row.code === "binding_artifact_digest_mismatch"));

assert.equal(TENSORRT_LLM_SOURCE_METADATA.release, "v1.2.0");
assert.equal(TENSORRT_LLM_SOURCE_METADATA.files.length, 4);
for (const row of TENSORRT_LLM_SOURCE_METADATA.files) assert.match(row.sha256, /^[a-f0-9]{64}$/);

console.log("TensorRT-LLM static deployment contract passed (config parsing, TP/PP/CP and layer conservation, quant/KV scenario, artifact binding, source pins, and fail-closed mutations)." );
