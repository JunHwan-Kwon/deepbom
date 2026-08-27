import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { TENSORRT_LLM_SOURCE_METADATA } from "./tensorrt-llm-source-metadata.js";

export const TENSORRT_LLM_CONTRACT_SCHEMA = "deepbom.tensorrt_llm_static_deployment_contract.v1";
export const TENSORRT_LLM_BINDING_SCHEMA = "deepbom.tensorrt_llm_artifact_binding.v1";
const SHA256 = /^[a-f0-9]{64}$/;

export function buildTensorRtLlmContract(analysis, llmContract, { engineConfig = null, binding = null } = {}) {
  if (!engineConfig?.document) return notSelected();
  const issues = [];
  const document = engineConfig.document;
  if (!document || typeof document !== "object" || Array.isArray(document)) {
    return invalid("engine_config_not_object", engineConfig, issues);
  }
  const pretrained = object(document.pretrained_config, "pretrained_config", issues);
  const build = object(document.build_config, "build_config", issues);
  const mapping = object(pretrained.mapping, "pretrained_config.mapping", issues);
  const quantization = object(pretrained.quantization, "pretrained_config.quantization", issues);
  const worldSize = positiveInteger(mapping.world_size, "mapping.world_size", issues);
  const tpSize = positiveInteger(mapping.tp_size, "mapping.tp_size", issues);
  const ppSize = positiveInteger(mapping.pp_size, "mapping.pp_size", issues);
  const cpSize = positiveInteger(mapping.cp_size ?? 1, "mapping.cp_size", issues);
  if ([worldSize, tpSize, ppSize, cpSize].every((value) => value != null) && worldSize !== tpSize * ppSize * cpSize) {
    issues.push(issue("parallel_world_size_mismatch", `${worldSize} != ${tpSize} x ${ppSize} x ${cpSize}`));
  }
  const layerCount = positiveInteger(pretrained.num_hidden_layers, "pretrained_config.num_hidden_layers", issues);
  const ppPartition = normalizePipelinePartition(mapping.pp_partition, ppSize, layerCount, issues);
  const maxInput = positiveInteger(build.max_input_len, "build_config.max_input_len", issues);
  const maxSequence = positiveInteger(build.max_seq_len, "build_config.max_seq_len", issues);
  const maxBatch = positiveInteger(build.max_batch_size, "build_config.max_batch_size", issues);
  const maxTokens = positiveInteger(build.max_num_tokens, "build_config.max_num_tokens", issues);
  const optTokens = nullablePositiveInteger(build.opt_num_tokens, "build_config.opt_num_tokens", issues);
  if (maxInput != null && maxSequence != null && maxInput > maxSequence) issues.push(issue("input_exceeds_sequence_limit", `${maxInput} > ${maxSequence}`));
  if (optTokens != null && maxTokens != null && optTokens > maxTokens) issues.push(issue("optimal_tokens_exceed_maximum", `${optTokens} > ${maxTokens}`));
  const architecture = {
    name: requiredText(pretrained.architecture, "pretrained_config.architecture", issues),
    dtype: requiredText(pretrained.dtype, "pretrained_config.dtype", issues),
    hidden_size: positiveInteger(pretrained.hidden_size, "pretrained_config.hidden_size", issues),
    intermediate_size: positiveInteger(pretrained.intermediate_size, "pretrained_config.intermediate_size", issues),
    layer_count: layerCount,
    attention_head_count: positiveInteger(pretrained.num_attention_heads, "pretrained_config.num_attention_heads", issues),
    kv_head_count: positiveInteger(pretrained.num_key_value_heads, "pretrained_config.num_key_value_heads", issues),
    head_size: positiveInteger(pretrained.head_size, "pretrained_config.head_size", issues),
    vocabulary_size: nullablePositiveInteger(pretrained.vocab_size, "pretrained_config.vocab_size", issues),
    maximum_position_embeddings: nullablePositiveInteger(pretrained.max_position_embeddings, "pretrained_config.max_position_embeddings", issues),
  };
  const architectureComparisons = compareArchitecture(architecture, llmContract?.architecture || {});
  if (architectureComparisons.some((row) => row.status === "mismatch")) issues.push(issue("source_architecture_mismatch", "TensorRT-LLM engine config differs from the artifact-bound architecture contract."));
  const sourceArtifactSha = tensorRtLlmSourceArtifactSha(analysis);
  const bindingResult = validateBinding(binding, engineConfig, sourceArtifactSha, analysis?.artifact_bundle?.files || [], issues);
  const kvBits = kvCacheStorageBits(quantization.kv_cache_quant_algo, architecture.dtype);
  const kvScenario = deriveKvScenario(architecture, maxSequence, maxBatch, kvBits);
  const maxBeam = positiveInteger(build.max_beam_width, "build_config.max_beam_width", issues);
  const kvCacheType = requiredText(build.kv_cache_type, "build_config.kv_cache_type", issues);
  const stronglyTyped = explicitBoolean(build.strongly_typed, "build_config.strongly_typed", issues);
  const weightStreaming = explicitBoolean(build.weight_streaming, "build_config.weight_streaming", issues);
  const groupSize = nullablePositiveInteger(quantization.group_size, "quantization.group_size", issues);
  const hasZeroPoint = nullableBoolean(quantization.has_zero_point, "quantization.has_zero_point", issues);
  const normalized = {
    schema: TENSORRT_LLM_CONTRACT_SCHEMA,
    method_version: "1.0.0",
    status: issues.length ? "invalid" : bindingResult.status === "artifact_bound" ? "artifact_bound_configuration" : "candidate_configuration_unbound",
    evidence_class: bindingResult.status === "artifact_bound" ? "OBSERVED_CONFIG/SOURCE_PINNED/ARTIFACT_BOUND/DERIVED" : "OBSERVED_CONFIG/SOURCE_PINNED/UNBOUND_CANDIDATE",
    engine_config: {
      path: engineConfig.path || null,
      byte_length: engineConfig.byte_length ?? null,
      sha256: engineConfig.sha256 || null,
      version: document.version ?? null,
      canonical_content_sha256: sha256TextHex(canonicalJson(document)),
    },
    artifact_binding: bindingResult,
    architecture,
    architecture_comparison: architectureComparisons,
    parallelism: {
      world_size: worldSize,
      tensor_parallel_size: tpSize,
      pipeline_parallel_size: ppSize,
      context_parallel_size: cpSize,
      layer_partition_per_pipeline_rank: ppPartition,
      layer_partition_conservation: ppPartition && layerCount != null && ppPartition.reduce((sum, value) => sum + value, 0) === layerCount ? "complete" : "not_assessable",
      weight_bytes_per_rank: null,
      weight_bytes_per_rank_reason: "Tensor parallelism and pipeline parallelism do not imply uniform byte sharding for embeddings, LM heads, replicated parameters, plugins, quantized packing, or managed weights. Per-rank engine or builder evidence is required.",
    },
    build_limits: {
      max_input_length: maxInput,
      max_sequence_length: maxSequence,
      maximum_batch_size: maxBatch,
      maximum_beam_width: maxBeam,
      maximum_batched_tokens: maxTokens,
      optimal_batched_tokens: optTokens,
      kv_cache_type: kvCacheType,
      strongly_typed: stronglyTyped,
      weight_streaming: weightStreaming,
      plugin_config_sha256: build.plugin_config && typeof build.plugin_config === "object" ? sha256TextHex(canonicalJson(build.plugin_config)) : null,
    },
    quantization: {
      weight_activation_algorithm: quantization.quant_algo ?? null,
      kv_cache_algorithm: quantization.kv_cache_quant_algo ?? null,
      group_size: groupSize,
      has_zero_point: hasZeroPoint,
      excluded_module_count: Array.isArray(quantization.exclude_modules) ? quantization.exclude_modules.length : 0,
    },
    kv_cache_scenario: kvScenario,
    issues,
    issue_count: issues.length,
    source_basis: TENSORRT_LLM_SOURCE_METADATA,
    trust_boundary: {
      browser_engine_deserialization: "prohibited",
      browser_engine_execution: "prohibited",
      accepted_input: "Hash-bound UTF-8 JSON engine configuration and optional artifact-binding manifest only.",
    },
    interpretation_boundary: "This contract validates serialized TensorRT-LLM configuration, parallelism conservation, artifact binding, and a conditional logical KV-state scenario where dimensions are complete. It does not build or deserialize an engine, establish per-rank weight residency, tactic/kernel selection, workspace or allocator memory, GPU occupancy, throughput, latency, accuracy, or device feasibility.",
  };
  normalized.contract_sha256 = sha256TextHex(canonicalJson(normalized));
  return normalized;
}

export function validateTensorRtLlmContract(analysis, llmContract) {
  const contract = llmContract?.tensorrt_llm;
  if (!contract || contract.status === "not_selected") return [];
  const errors = [];
  if (contract.schema !== TENSORRT_LLM_CONTRACT_SCHEMA || contract.issue_count !== (contract.issues || []).length) errors.push("tensorrt_llm_contract_shape_invalid");
  const clone = JSON.parse(JSON.stringify(contract));
  const digest = clone.contract_sha256;
  delete clone.contract_sha256;
  if (digest !== sha256TextHex(canonicalJson(clone))) errors.push("tensorrt_llm_contract_sha256_mismatch");
  const parallel = contract.parallelism || {};
  if (parallel.world_size !== parallel.tensor_parallel_size * parallel.pipeline_parallel_size * parallel.context_parallel_size) errors.push("tensorrt_llm_parallelism_conservation_failed");
  if (parallel.layer_partition_per_pipeline_rank && parallel.layer_partition_per_pipeline_rank.reduce((sum, value) => sum + value, 0) !== contract.architecture?.layer_count) errors.push("tensorrt_llm_layer_partition_conservation_failed");
  if (contract.status === "artifact_bound_configuration") {
    const sourceArtifactSha = tensorRtLlmSourceArtifactSha(analysis);
    if (contract.artifact_binding?.source_artifact_sha256 !== sourceArtifactSha) errors.push("tensorrt_llm_artifact_binding_mismatch");
  }
  if (contract.issue_count) errors.push("tensorrt_llm_contract_contains_issues");
  return errors;
}

function notSelected() {
  return { schema: TENSORRT_LLM_CONTRACT_SCHEMA, status: "not_selected", evidence_class: "NOT_ASSESSABLE", source_basis: TENSORRT_LLM_SOURCE_METADATA };
}

function invalid(code, engineConfig, issues) {
  issues.push(issue(code, "TensorRT-LLM engine configuration cannot be assessed."));
  const value = { schema: TENSORRT_LLM_CONTRACT_SCHEMA, status: "invalid", evidence_class: "OBSERVED_INVALID", engine_config: { path: engineConfig?.path || null, sha256: engineConfig?.sha256 || null }, issues, issue_count: issues.length, source_basis: TENSORRT_LLM_SOURCE_METADATA };
  value.contract_sha256 = sha256TextHex(canonicalJson(value));
  return value;
}

function validateBinding(binding, engineConfig, sourceArtifactSha, bundleFiles, issues) {
  if (!binding?.document) return { status: "not_selected", source_artifact_sha256: null, engine_config_sha256: null, engine_file_count: 0 };
  const document = binding.document;
  if (document?.schema !== TENSORRT_LLM_BINDING_SCHEMA || !SHA256.test(String(document.source_artifact_sha256 || "")) || !SHA256.test(String(document.engine_config_sha256 || ""))) {
    issues.push(issue("binding_manifest_invalid", "TensorRT-LLM binding manifest schema or digest is invalid."));
    return { status: "invalid", source_artifact_sha256: document?.source_artifact_sha256 || null, engine_config_sha256: document?.engine_config_sha256 || null, engine_file_count: 0 };
  }
  if (document.source_artifact_sha256 !== sourceArtifactSha) issues.push(issue("binding_artifact_digest_mismatch", "Binding manifest does not identify the selected model-source artifact."));
  if (document.engine_config_sha256 !== engineConfig.sha256) issues.push(issue("binding_engine_config_digest_mismatch", "Binding manifest does not identify the selected TensorRT-LLM engine config."));
  const fileMap = new Map(bundleFiles.map((row) => [row.path, row]));
  const engineFiles = Array.isArray(document.engine_files) ? document.engine_files : [];
  for (const row of engineFiles) {
    if (!row || typeof row.path !== "string" || !SHA256.test(String(row.sha256 || "")) || fileMap.get(row.path)?.sha256 !== row.sha256) {
      issues.push(issue("binding_engine_file_mismatch", `Engine component ${row?.path || "<unknown>"} is absent or hash-divergent in the selected bundle.`));
    }
  }
  return { status: "artifact_bound", source_artifact_sha256: document.source_artifact_sha256, engine_config_sha256: document.engine_config_sha256, engine_file_count: engineFiles.length, manifest_path: binding.path || null, manifest_sha256: binding.sha256 || null };
}

function tensorRtLlmSourceArtifactSha(analysis) {
  return String(analysis?.artifact_bundle?.model_source_sha256 || analysis?.model_sha256 || "").toLowerCase();
}

function compareArchitecture(engine, artifact) {
  const mappings = [
    ["hidden_size", "hidden_size"], ["intermediate_size", "intermediate_size"], ["layer_count", "layer_count"],
    ["attention_head_count", "attention_head_count"], ["kv_head_count", "kv_head_count"], ["head_size", "head_width"],
    ["vocabulary_size", "vocabulary_size"], ["maximum_position_embeddings", "context_length"],
  ];
  return mappings.map(([engineKey, artifactKey]) => {
    const left = engine[engineKey]; const right = artifact[artifactKey];
    return { engine_field: engineKey, artifact_field: artifactKey, engine_value: left, artifact_value: right, status: left == null || right == null ? "not_comparable" : left === right ? "match" : "mismatch" };
  });
}

function normalizePipelinePartition(value, ppSize, layers, issues) {
  if (value != null) {
    if (!Array.isArray(value) || value.some((item) => !Number.isSafeInteger(Number(item)) || Number(item) < 0)) { issues.push(issue("pipeline_partition_invalid", "mapping.pp_partition must contain non-negative integer layer counts.")); return null; }
    const rows = value.map(Number);
    if (ppSize != null && rows.length !== ppSize) issues.push(issue("pipeline_partition_rank_count_mismatch", `${rows.length} partition entries for pp_size ${ppSize}.`));
    if (layers != null && rows.reduce((sum, item) => sum + item, 0) !== layers) issues.push(issue("pipeline_partition_layer_count_mismatch", "Pipeline partition does not conserve layer count."));
    return rows;
  }
  if (ppSize == null || layers == null) return null;
  const base = Math.floor(layers / ppSize); const remainder = layers % ppSize;
  return Array.from({ length: ppSize }, (_, index) => base + (index < remainder ? 1 : 0));
}

function kvCacheStorageBits(algorithm, dtype) {
  const value = String(algorithm || "").toUpperCase();
  if (value.includes("FP4") || value.includes("INT4")) return 4;
  if (value.includes("FP8") || value.includes("INT8")) return 8;
  const type = String(dtype || "").toLowerCase();
  if (["float16", "bfloat16", "half"].includes(type)) return 16;
  if (["float32", "float"].includes(type)) return 32;
  return null;
}

function deriveKvScenario(architecture, context, batch, bits) {
  const values = [architecture.layer_count, architecture.kv_head_count, architecture.head_size, context, batch, bits];
  if (!values.every((value) => Number.isSafeInteger(value) && value > 0)) return { status: "not_assessable_incomplete_dimensions", logical_bytes: null };
  const elements = 2n * BigInt(architecture.layer_count) * BigInt(architecture.kv_head_count) * BigInt(architecture.head_size) * BigInt(context) * BigInt(batch);
  const bytes = (elements * BigInt(bits) + 7n) / 8n;
  return { status: "derived_conditional_logical_state", evidence_class: "OBSERVED_CONFIG/DERIVED_CONDITIONAL", context_length: context, batch_size: batch, storage_bits: bits, logical_elements_decimal: String(elements), logical_bytes: { value: bytes <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(bytes) : null, decimal: String(bytes) }, boundary: "Logical full-context KV cardinality from serialized TensorRT-LLM dimensions. Runtime block allocation, reserved capacity, beam sharing, eviction, reuse, fragmentation, and residency are not included." };
}

function object(value, field, issues) { if (!value || typeof value !== "object" || Array.isArray(value)) { issues.push(issue("required_object_missing", field)); return {}; } return value; }
function requiredText(value, field, issues) { const result = String(value ?? "").trim(); if (!result) issues.push(issue("required_text_missing", field)); return result || null; }
function positiveInteger(value, field, issues) { const result = Number(value); if (!Number.isSafeInteger(result) || result <= 0) { issues.push(issue("positive_integer_invalid", field)); return null; } return result; }
function nullablePositiveInteger(value, field, issues) { return value == null ? null : positiveInteger(value, field, issues); }
function explicitBoolean(value, field, issues) { if (typeof value !== "boolean") { issues.push(issue("boolean_not_explicit", field)); return null; } return value; }
function nullableBoolean(value, field, issues) { return value == null ? null : explicitBoolean(value, field, issues); }
function issue(code, detail) { return { code, detail }; }
