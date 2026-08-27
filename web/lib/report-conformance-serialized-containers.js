import { HF_SAFETENSORS_ARCHITECTURE_SOURCE } from "./hf-safetensors-contract.js";
import { isCanonicalGgufDecoderArchitecture } from "./metadata-model-adapters.js";
import {
  SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES,
  SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA,
  SAFETENSORS_QUANTIZATION_SOURCES,
} from "./safetensors-quantization-contract.js";

function exactDecimalInteger(value) {
  try {
    const text = String(value);
    return /^(0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
  } catch {
    return null;
  }
}

function exactShapeCardinality(shape) {
  if (!Array.isArray(shape) || shape.some((value) => !Number.isSafeInteger(value) || value < 0)) return null;
  return shape.reduce((product, value) => product * BigInt(value), 1n);
}

function specializedTensorLedgerValid(tensors, tensorContract) {
  const checkpoint = tensors.reduce((sum, tensor) => {
    const cardinality = exactShapeCardinality(tensor.shape);
    return cardinality == null || sum == null ? null : sum + cardinality;
  }, 0n);
  const expected = (tensorContract.expected_rows || []).reduce((sum, row) => {
    const cardinality = exactShapeCardinality(row.expected_shape);
    return cardinality == null || sum == null ? null : sum + cardinality;
  }, 0n);
  const byName = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  const observed = (tensorContract.expected_rows || []).reduce((sum, row) => {
    const cardinality = exactShapeCardinality(byName.get(row.tensor_name)?.shape);
    return cardinality == null || sum == null ? null : sum + cardinality;
  }, 0n);
  return exactDecimalInteger(tensorContract.checkpoint_parameter_count_decimal) === checkpoint
    && exactDecimalInteger(tensorContract.canonical_expected_parameter_count_decimal) === expected
    && exactDecimalInteger(tensorContract.canonical_observed_parameter_count_decimal) === observed;
}

function specializedSourceValid(hf, configSource, modelingSource, expectedConfigSource, expectedModelingSource, reportText) {
  return hf.source?.repository === HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository
    && hf.source?.source_commit === HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit
    && configSource?.path === expectedConfigSource?.path && configSource?.sha256 === expectedConfigSource?.sha256
    && modelingSource?.path === expectedModelingSource?.path && modelingSource?.sha256 === expectedModelingSource?.sha256
    && reportText.includes(String(configSource?.sha256)) && reportText.includes(String(modelingSource?.sha256));
}

function validateMixtralProjection(hf, fields, projection, kv, tensors, tensorContract, sources, reportText) {
  const h = BigInt(fields.hidden_size), i = BigInt(fields.intermediate_size), layers = BigInt(fields.num_hidden_layers);
  const q = BigInt(fields.num_attention_heads) * BigInt(fields.head_dim);
  const kvWidth = BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim);
  const context = BigInt(fields.max_position_embeddings), experts = BigInt(fields.num_local_experts), active = BigInt(fields.num_experts_per_tok);
  const attention = 2n * h * q + 2n * h * kvWidth;
  const router = h * experts, perExpert = 3n * h * i, activeExpert = perExpert * active;
  const activePerLayer = attention + router + activeExpert, activeAllLayers = activePerLayer * layers;
  const causalPairs = context * (context + 1n) / 2n;
  const expectedKv = layers * BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim) * 2n;
  return projection?.schema === "deepbom.sparse_moe_decoder_projection.v1"
    && exactDecimalInteger(projection.router_macs_per_layer_per_token?.decimal) === router
    && exactDecimalInteger(projection.expert_matrix_parameters_per_expert_per_layer?.decimal) === perExpert
    && exactDecimalInteger(projection.total_expert_matrix_parameters_all_layers?.decimal) === perExpert * experts * layers
    && exactDecimalInteger(projection.active_expert_matrix_macs_per_layer_per_token?.decimal) === activeExpert
    && exactDecimalInteger(projection.active_projection_macs_per_layer_per_token?.decimal) === activePerLayer
    && exactDecimalInteger(projection.active_projection_macs_all_layers_per_token?.decimal) === activeAllLayers
    && exactDecimalInteger(projection.prefill_active_core_macs_at_declared_context?.decimal) === activeAllLayers * context + 2n * q * causalPairs * layers
    && exactDecimalInteger(projection.decode_active_core_macs_at_declared_context?.decimal) === activeAllLayers + 2n * q * context * layers
    && kv?.schema === "deepbom.transformer_kv_state_projection.v1"
    && exactDecimalInteger(kv.elements_per_token_per_batch?.decimal) === expectedKv
    && exactDecimalInteger(kv.elements_at_context_batch_one?.decimal) === expectedKv * context
    && specializedTensorLedgerValid(tensors, tensorContract)
    && specializedSourceValid(hf, ...sources, reportText)
    && reportText.includes("Total expert matrix parameters") && reportText.includes("Active expert matrix MACs");
}

function validateMambaProjection(hf, fields, projection, tensors, tensorContract, sources, reportText) {
  const h = BigInt(fields.hidden_size), intermediate = BigInt(fields.intermediate_size), layers = BigInt(fields.num_hidden_layers);
  const state = BigInt(fields.state_size), rank = BigInt(fields.time_step_rank), kernel = BigInt(fields.conv_kernel);
  const recurrent = hf.recurrent_state_projection;
  const recurrentElements = layers * intermediate * (state + kernel);
  const accounted = h * intermediate * 2n + intermediate * (rank + state * 2n) + rank * intermediate + intermediate * h + intermediate * kernel;
  return projection?.schema === "deepbom.ssm_linear_compute_projection.v1"
    && exactDecimalInteger(projection.accounted_macs_per_layer_per_token?.decimal) === accounted
    && exactDecimalInteger(projection.accounted_macs_all_layers_per_token?.decimal) === accounted * layers
    && projection.accounted_macs_all_layers_at_declared_context == null
    && recurrent?.schema === "deepbom.ssm_recurrent_state_projection.v1"
    && exactDecimalInteger(recurrent.convolution_state_elements_per_layer_per_batch?.decimal) === intermediate * kernel
    && exactDecimalInteger(recurrent.ssm_state_elements_per_layer_per_batch?.decimal) === intermediate * state
    && exactDecimalInteger(recurrent.recurrent_state_elements_all_layers_per_batch?.decimal) === recurrentElements
    && specializedTensorLedgerValid(tensors, tensorContract)
    && specializedSourceValid(hf, ...sources, reportText)
    && reportText.includes("Selective-scan arithmetic") && reportText.includes("Recurrent-state cardinality");
}

function validateJambaProjection(hf, fields, projection, tensors, tensorContract, sources, reportText) {
  const h = BigInt(fields.hidden_size), intermediate = BigInt(fields.intermediate_size);
  const attentionLayers = BigInt(fields.attention_layer_count), mambaLayers = BigInt(fields.mamba_layer_count);
  const denseLayers = BigInt(fields.dense_feed_forward_layer_count), expertLayers = BigInt(fields.expert_feed_forward_layer_count);
  const q = BigInt(fields.num_attention_heads) * BigInt(fields.head_dim);
  const kvWidth = BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim);
  const context = BigInt(fields.max_position_embeddings), experts = BigInt(fields.num_experts), active = BigInt(fields.num_experts_per_tok);
  const mambaIntermediate = BigInt(fields.mamba_intermediate_size), state = BigInt(fields.mamba_d_state);
  const rank = BigInt(fields.mamba_dt_rank), kernel = BigInt(fields.mamba_d_conv);
  const attentionProjectionAll = (2n * h * q + 2n * h * kvWidth) * attentionLayers;
  const mambaPerLayer = 2n * h * mambaIntermediate + mambaIntermediate * (rank + 2n * state)
    + rank * mambaIntermediate + mambaIntermediate * h + mambaIntermediate * kernel;
  const denseFfPerLayer = 3n * h * intermediate;
  const activeMoeAll = (h * experts + denseFfPerLayer * active) * expertLayers;
  const accounted = attentionProjectionAll + mambaPerLayer * mambaLayers + denseFfPerLayer * denseLayers + activeMoeAll;
  const causalPairs = context * (context + 1n) / 2n;
  const decodeAttention = 2n * q * context * attentionLayers;
  const expectedKv = attentionLayers * BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim) * 2n;
  const expectedRecurrent = mambaLayers * mambaIntermediate * (state + kernel);
  return projection?.schema === "deepbom.hybrid_jamba_compute_projection.v1"
    && exactDecimalInteger(projection.attention_projection_macs_all_attention_layers_per_token?.decimal) === attentionProjectionAll
    && exactDecimalInteger(projection.mamba_projection_and_depthwise_macs_all_mamba_layers_per_token?.decimal) === mambaPerLayer * mambaLayers
    && exactDecimalInteger(projection.active_moe_macs_all_expert_layers_per_token?.decimal) === activeMoeAll
    && exactDecimalInteger(projection.accounted_projection_macs_all_layers_per_token?.decimal) === accounted
    && exactDecimalInteger(projection.prefill_accounted_core_macs_at_declared_context?.decimal) === accounted * context + 2n * q * causalPairs * attentionLayers
    && exactDecimalInteger(projection.decode_accounted_core_macs_at_declared_context?.decimal) === accounted + decodeAttention
    && hf.kv_state_projection?.schema === "deepbom.transformer_kv_state_projection.v1"
    && exactDecimalInteger(hf.kv_state_projection.elements_per_token_per_batch?.decimal) === expectedKv
    && exactDecimalInteger(hf.kv_state_projection.elements_at_context_batch_one?.decimal) === expectedKv * context
    && hf.recurrent_state_projection?.schema === "deepbom.ssm_recurrent_state_projection.v1"
    && exactDecimalInteger(hf.recurrent_state_projection.recurrent_state_elements_all_layers_per_batch?.decimal) === expectedRecurrent
    && specializedTensorLedgerValid(tensors, tensorContract)
    && specializedSourceValid(hf, ...sources, reportText)
    && reportText.includes("Hybrid decode core") && reportText.includes("Selective-scan arithmetic");
}

export function registerGgufSerializedConformance({ staticAnalysis, tensors, ops, reportText, check }) {
  const gguf = staticAnalysis.gguf || {};
  const ordered = [...tensors].sort((left, right) => Number(left.data_offset) - Number(right.data_offset));
  const storedBytes = tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
  check("CF-GGUF-RANGE-001", gguf.payload_coverage_status === "complete_without_gaps_or_overlaps"
    && storedBytes === Number(gguf.declared_tensor_byte_length)
    && ordered.every((tensor, index) => Number(tensor.data_end) - Number(tensor.data_offset) === Number(tensor.byte_length)
      && (index === 0 || Number(tensor.data_offset) >= Number(ordered[index - 1].data_end))),
  "GGUF tensor byte ranges do not conserve the declared tensor payload.", ["/evidence/static_analysis/gguf", "/evidence/static_analysis/tensors"]);
  const source = gguf.type_traits_source || {};
  check("CF-GGUF-SOURCE-001", /^[0-9a-f]{40}$/i.test(String(source.source_commit || ""))
    && /^[0-9a-f]{64}$/i.test(String(source.type_traits_source_sha256 || ""))
    && /^[0-9a-f]{64}$/i.test(String(source.block_layout_source_sha256 || ""))
    && reportText.includes(String(source.source_commit || "")),
  "GGUF block cardinality and layout interpretation is not source-commit/content-digest bound.", ["/evidence/static_analysis/gguf/type_traits_source", "/engineering_report.md"]);
  const numerical = staticAnalysis.tensor_numerical_integrity || {};
  const numericalSource = numerical.decoder_source || {};
  const assessedNumericalRows = (numerical.tensor_records || []).filter((row) => row.status === "assessed_full_payload");
  check("CF-GGUF-ENDIAN-001", !numerical.schema || (["little", "big"].includes(gguf.endianness)
    && (gguf.endianness !== "big" || Number(gguf.version) === 3)
    && assessedNumericalRows.every((row) => row.serialized_endianness === gguf.endianness)
    && /^[0-9a-f]{40}$/i.test(String(numericalSource.format_specification_commit || ""))
    && /^[0-9a-f]{64}$/i.test(String(numericalSource.format_specification_source_sha256 || ""))
    && reportText.includes(String(numericalSource.format_specification_commit || ""))
    && reportText.includes(String(numericalSource.format_specification_source_sha256 || ""))
    && reportText.includes("Source-defined multi-byte fields follow file endian")),
  "GGUF scalar/block numerical evidence is not bound to the v3 endian contract or per-tensor serialized-endian ledger.", ["/evidence/static_analysis/gguf/endianness", "/evidence/static_analysis/tensor_numerical_integrity", "/engineering_report.md"]);
  check("CF-GGUF-GRAPH-001", ops.length === 0 && staticAnalysis.total_macs == null
    && reportText.includes("does not serialize an execution-operator DAG"),
  "GGUF report must preserve the absent execution graph and MAC assessment as non-numeric evidence.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/total_macs", "/engineering_report.md"]);

  const semantic = gguf.semantic_contract || {};
  const storage = staticAnalysis.tensor_storage_summary || {};
  const keyWidth = Number(semantic.attention_key_length || semantic.derived_attention_head_width || 0);
  const valueWidth = Number(semantic.attention_value_length || semantic.derived_attention_head_width || 0);
  const kvEligible = [semantic.block_count, semantic.attention_head_count_kv, keyWidth, valueWidth, semantic.context_length]
    .every((value) => Number.isSafeInteger(Number(value)) && Number(value) > 0);
  const kv = semantic.kv_state_projection || null;
  const expectedKvPerToken = kvEligible
    ? BigInt(semantic.block_count) * BigInt(semantic.attention_head_count_kv) * BigInt(keyWidth + valueWidth) : null;
  const expectedKvAtContext = expectedKvPerToken == null ? null : expectedKvPerToken * BigInt(semantic.context_length);
  const compute = semantic.compute_projection || null;
  const computeEligible = isCanonicalGgufDecoderArchitecture(semantic.architecture)
    && [semantic.tokenizer?.vocabulary_count, semantic.embedding_length, semantic.feed_forward_length,
      semantic.block_count, semantic.attention_head_count, semantic.attention_head_count_kv,
      keyWidth, valueWidth, semantic.context_length]
      .every((value) => Number.isSafeInteger(Number(value)) && Number(value) > 0)
    && keyWidth === valueWidth;
  let computeValid = !computeEligible && compute == null;
  if (computeEligible) {
    const vocabulary = BigInt(semantic.tokenizer.vocabulary_count);
    const hidden = BigInt(semantic.embedding_length);
    const intermediate = BigInt(semantic.feed_forward_length);
    const layers = BigInt(semantic.block_count);
    const queryWidth = BigInt(semantic.attention_head_count) * BigInt(keyWidth);
    const kvProjectionWidth = BigInt(semantic.attention_head_count_kv) * BigInt(keyWidth);
    const context = BigInt(semantic.context_length);
    const attentionProjection = 2n * hidden * queryWidth + 2n * hidden * kvProjectionWidth;
    const mlpProjection = 3n * hidden * intermediate;
    const denseToken = (attentionProjection + mlpProjection) * layers;
    const causalPairs = context * (context + 1n) / 2n;
    const prefillCore = denseToken * context + 2n * queryWidth * causalPairs * layers;
    const decodeCore = denseToken + 2n * queryWidth * context * layers;
    const logits = hidden * vocabulary;
    computeValid = compute?.schema === "deepbom.canonical_gated_decoder_compute_projection.v1"
      && exactDecimalInteger(compute.attention_projection_macs_per_layer_per_token?.decimal) === attentionProjection
      && exactDecimalInteger(compute.gated_mlp_projection_macs_per_layer_per_token?.decimal) === mlpProjection
      && exactDecimalInteger(compute.dense_projection_macs_all_layers_per_token?.decimal) === denseToken
      && exactDecimalInteger(compute.declared_context_causal_attention_pair_count?.decimal) === causalPairs
      && exactDecimalInteger(compute.prefill_transformer_core_macs_at_declared_context?.decimal) === prefillCore
      && exactDecimalInteger(compute.decode_transformer_core_macs_at_declared_context?.decimal) === decodeCore
      && exactDecimalInteger(compute.output_projection_macs_per_logit_position?.decimal) === logits
      && exactDecimalInteger(compute.decode_with_one_logit_position_macs?.decimal) === decodeCore + logits
      && reportText.includes("Declared-context prefill core MACs");
  }
  check("CF-GGUF-SEMANTIC-001", semantic.schema === "deepbom.gguf_semantic_contract.v1"
    && String(semantic.serialized_parameter_count_decimal || "") === String(storage.element_count_decimal || "")
    && String(semantic.serialized_tensor_bytes_decimal || "") === String(storage.byte_length_decimal || "")
    && String(semantic.effective_bits_per_parameter || "") === String(storage.effective_bits_per_element || "")
    && Boolean(kv) === kvEligible
    && (!kv || kv.schema === "deepbom.transformer_kv_state_projection.v1"
      && exactDecimalInteger(kv.elements_per_token_per_batch?.decimal) === expectedKvPerToken
      && exactDecimalInteger(kv.elements_at_context_batch_one?.decimal) === expectedKvAtContext
      && kv.bytes_per_token_per_batch_if_storage_width_matches == null
      && reportText.includes("KV-state cardinality"))
    && computeValid,
  "GGUF parameter storage, KV-state cardinality, or registered canonical decoder compute scenario does not reconstruct from tensor ranges and architecture metadata.", ["/evidence/static_analysis/gguf/semantic_contract", "/evidence/static_analysis/tensor_storage_summary", "/engineering_report.md"]);
}

export function registerSafeTensorsSerializedConformance({ staticAnalysis, tensors, ops, reportText, check }) {
  const safe = staticAnalysis.safetensors || {};
  const storedBytes = tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
  const shardGroups = new Map();
  for (const tensor of tensors) {
    const key = tensor.shard_path || "__single_file__";
    if (!shardGroups.has(key)) shardGroups.set(key, []);
    shardGroups.get(key).push(tensor);
  }
  const rangesConservePerFile = [...shardGroups.values()].every((group) => {
    const ordered = [...group].sort((left, right) => Number(left.data_offset) - Number(right.data_offset));
    return ordered.every((tensor, index) => Number(tensor.data_end) - Number(tensor.data_offset) === Number(tensor.byte_length)
      && (index === 0 ? Number(tensor.data_offset) === 0 : Number(tensor.data_offset) === Number(ordered[index - 1].data_end)));
  });
  const shardFiles = (staticAnalysis.artifact_bundle?.files || []).filter((file) => file.role === "tensor_shard");
  const shardContract = safe.sharded
    ? Number(safe.shard_count) === shardGroups.size
      && Number(safe.shard_count) === shardFiles.length
      && Number(safe.index_tensor_count) === tensors.length
      && safe.index_binding_status === "complete_bidirectional"
      && [...shardGroups.keys()].every((path) => shardFiles.some((file) => file.path === path))
    : shardGroups.size === 1
      && Number(safe.shard_count || 1) === 1
      && (!staticAnalysis.artifact_bundle || shardFiles.length === 1);
  check("CF-SAFETENSORS-RANGE-001", safe.payload_coverage_status === "complete_without_gaps_or_overlaps"
    && storedBytes === Number(safe.payload_byte_length)
    && rangesConservePerFile
    && shardContract,
  "SafeTensors tensor byte ranges do not exactly partition the serialized payload.", ["/evidence/static_analysis/safetensors", "/evidence/static_analysis/tensors"]);
  const source = safe.reference_implementation || {};
  check("CF-SAFETENSORS-SOURCE-001", /^[0-9a-f]{40}$/i.test(String(source.commit || ""))
    && /^[0-9a-f]{64}$/i.test(String(source.tensor_rs_sha256 || ""))
    && reportText.includes(String(source.commit || "")),
  "SafeTensors dtype and range interpretation is not source-commit/content-digest bound.", ["/evidence/static_analysis/safetensors/reference_implementation", "/engineering_report.md"]);
  check("CF-SAFETENSORS-QUANT-001", safeTensorsQuantizationContractValid(safe.quantization_contract, reportText),
  "SafeTensors packed-weight declaration, module conservation, pinned source, or report projection is inconsistent.", ["/evidence/static_analysis/safetensors/quantization_contract", "/engineering_report.md"]);
  check("CF-SAFETENSORS-GRAPH-001", ops.length === 0 && staticAnalysis.total_macs == null
    && reportText.includes("does not serialize an execution-operator DAG"),
  "SafeTensors report must preserve the absent execution graph and MAC assessment as non-numeric evidence.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/total_macs", "/engineering_report.md"]);

  const hf = safe.hf_architecture_contract || {};
  const fields = hf.fields || {};
  const tensorContract = hf.tensor_contract || {};
  const projection = hf.compute_projection || null;
  const kv = hf.kv_state_projection || null;
  const configSource = hf.source?.configuration_sources?.[hf.model_type] || null;
  const modelingSource = hf.source?.modeling_sources?.[hf.model_type] || null;
  const expectedConfigSource = HF_SAFETENSORS_ARCHITECTURE_SOURCE.configuration_sources[hf.model_type] || null;
  const expectedModelingSource = HF_SAFETENSORS_ARCHITECTURE_SOURCE.modeling_sources[hf.model_type] || null;
  const registeredContract = hf.status && !String(hf.status).startsWith("not_assessed") && hf.status !== "invalid_config";
  let projectionValid = !registeredContract;
  if (registeredContract) {
    const specializedSources = [configSource, modelingSource, expectedConfigSource, expectedModelingSource];
    if (hf.architecture_kind === "sparse_moe_decoder") {
      projectionValid = validateMixtralProjection(hf, fields, projection, kv, tensors, tensorContract, specializedSources, reportText);
    } else if (hf.architecture_kind === "hybrid_attention_ssm_moe") {
      projectionValid = validateJambaProjection(hf, fields, projection, tensors, tensorContract, specializedSources, reportText);
    } else if (hf.architecture_kind === "ssm_recurrent") {
      projectionValid = validateMambaProjection(hf, fields, projection, tensors, tensorContract, specializedSources, reportText);
    } else {
    const h = BigInt(fields.hidden_size);
    const i = BigInt(fields.intermediate_size);
    const layers = BigInt(fields.num_hidden_layers);
    const q = BigInt(fields.num_attention_heads) * BigInt(fields.head_dim);
    const kvWidth = BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim);
    const context = BigInt(fields.max_position_embeddings);
    const vocabulary = BigInt(fields.vocab_size);
    const attentionProjection = 2n * h * q + 2n * h * kvWidth;
    const mlpMatrixCount = Number.isSafeInteger(hf.mlp_projection_matrix_count)
      ? BigInt(hf.mlp_projection_matrix_count) : 0n;
    const mlpProjection = mlpMatrixCount * h * i;
    const densePerLayer = attentionProjection + mlpProjection;
    const denseAllLayersToken = densePerLayer * layers;
    const causalPairs = context * (context + 1n) / 2n;
    const prefillAttention = 2n * q * causalPairs * layers;
    const decodeAttention = 2n * q * context * layers;
    const prefillCore = denseAllLayersToken * context + prefillAttention;
    const decodeCore = denseAllLayersToken + decodeAttention;
    const logits = h * vocabulary;
    const expectedCheckpointParameters = tensors.reduce((sum, tensor) => {
      const cardinality = exactShapeCardinality(tensor.shape);
      return cardinality == null || sum == null ? null : sum + cardinality;
    }, 0n);
    const expectedCanonical = (tensorContract.expected_rows || []).reduce((sum, row) => {
      const cardinality = exactShapeCardinality(row.expected_shape);
      return cardinality == null || sum == null ? null : sum + cardinality;
    }, 0n);
    const observedCanonical = (tensorContract.check_rows || []).reduce((sum, row) => {
      const cardinality = exactShapeCardinality(row.observed_shape);
      return cardinality == null || sum == null ? null : sum + cardinality;
    }, 0n);
    const expectedKvPerToken = layers * BigInt(fields.num_key_value_heads) * BigInt(fields.head_dim) * 2n;
    const expectedProjectionSchema = mlpMatrixCount === 3n
      ? "deepbom.canonical_gated_decoder_compute_projection.v1"
      : "deepbom.canonical_decoder_compute_projection.v1";
    const observedMlpProjection = mlpMatrixCount === 3n
      ? projection?.gated_mlp_projection_macs_per_layer_per_token
      : projection?.mlp_projection_macs_per_layer_per_token;
    projectionValid = (mlpMatrixCount === 2n || mlpMatrixCount === 3n)
      && projection?.schema === expectedProjectionSchema
      && projection?.mlp_projection_matrix_count === Number(mlpMatrixCount)
      && exactDecimalInteger(projection.attention_projection_macs_per_layer_per_token?.decimal) === attentionProjection
      && exactDecimalInteger(observedMlpProjection?.decimal) === mlpProjection
      && exactDecimalInteger(projection.dense_projection_macs_per_layer_per_token?.decimal) === densePerLayer
      && exactDecimalInteger(projection.dense_projection_macs_all_layers_per_token?.decimal) === denseAllLayersToken
      && exactDecimalInteger(projection.declared_context_causal_attention_pair_count?.decimal) === causalPairs
      && exactDecimalInteger(projection.prefill_transformer_core_macs_at_declared_context?.decimal) === prefillCore
      && exactDecimalInteger(projection.decode_transformer_core_macs_at_declared_context?.decimal) === decodeCore
      && exactDecimalInteger(projection.output_projection_macs_per_logit_position?.decimal) === logits
      && exactDecimalInteger(projection.decode_with_one_logit_position_macs?.decimal) === decodeCore + logits
      && kv?.schema === "deepbom.transformer_kv_state_projection.v1"
      && exactDecimalInteger(kv.elements_per_token_per_batch?.decimal) === expectedKvPerToken
      && exactDecimalInteger(kv.elements_at_context_batch_one?.decimal) === expectedKvPerToken * context
      && exactDecimalInteger(tensorContract.checkpoint_parameter_count_decimal) === expectedCheckpointParameters
      && exactDecimalInteger(tensorContract.canonical_expected_parameter_count_decimal) === expectedCanonical
      && exactDecimalInteger(tensorContract.canonical_observed_parameter_count_decimal) === observedCanonical
      && typeof tensorContract.tensor_layout_id === "string" && tensorContract.tensor_layout_id === hf.tensor_layout_id
      && ["none", "qkv", "all"].includes(hf.attention_bias_scope)
      && tensorContract.attention_bias_scope === hf.attention_bias_scope
      && hf.source?.repository === HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository
      && hf.source?.source_commit === HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit
      && configSource?.path === expectedConfigSource?.path && configSource?.sha256 === expectedConfigSource?.sha256
      && modelingSource?.path === expectedModelingSource?.path && modelingSource?.sha256 === expectedModelingSource?.sha256
      && reportText.includes(String(configSource.sha256)) && reportText.includes(String(modelingSource.sha256))
      && reportText.includes("Declared-context prefill core MACs") && reportText.includes("Canonical / checkpoint parameters");
    }
  }
  check("CF-SAFETENSORS-HF-001", projectionValid,
  "SafeTensors registered dense, sparse-MoE, or SSM tensor/state/compute projection does not independently reconstruct.", ["/evidence/static_analysis/safetensors/hf_architecture_contract", "/evidence/static_analysis/tensors", "/engineering_report.md"]);
}

function safeTensorsQuantizationContractValid(contract, reportText) {
  if (contract?.schema !== SAFETENSORS_QUANTIZATION_CONTRACT_SCHEMA
    || !reportText.includes("SafeTensors Packed-weight Quantization Contract")) return false;
  if (!["assessed", "fail"].includes(contract.status)) {
    return contract.evidence_class === "NOT_ASSESSED"
      && String(contract.status || "").startsWith("not_")
      && reportText.includes("NOT ASSESSED");
  }
  const expectedSource = SAFETENSORS_QUANTIZATION_SOURCES[contract.method];
  if (!expectedSource || Object.entries(expectedSource).some(([key, value]) => contract.source?.[key] !== value)
    || !reportText.includes(expectedSource.sha256)) return false;
  const expectedAuxiliary = SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES[contract.method] || [];
  const sourceFiles = Array.isArray(contract.source_files) ? contract.source_files : [];
  if (expectedAuxiliary.some((expected) => !sourceFiles.some((observed) => Object.entries(expected).every(([key, value]) => observed?.[key] === value)))) return false;
  const modules = Array.isArray(contract.modules) ? contract.modules : [];
  let logical = 0n;
  let packed = 0n;
  let logicalBits = 0n;
  let storageBits = 0n;
  let paddingBits = 0n;
  let scales = 0n;
  let zeroes = 0n;
  let bytes = 0;
  let valid = 0;
  let issueCount = (contract.declaration_conflicts || []).length + (contract.config_issues || []).length;
  for (const module of modules) {
    const issues = Array.isArray(module.issues) ? module.issues : [];
    issueCount += issues.length;
    if ((module.status === "pass") !== (issues.length === 0)) return false;
    if (module.status === "pass") valid += 1;
    const moduleLogical = exactProduct([module.input_features, module.output_features]);
    const bitWidth = module.bits === 1.58 ? 2n : exactDecimalInteger(module.bits);
    const weightTensor = module.tensors?.qweight || module.tensors?.W_q || module.tensors?.weight_packed;
    const scaleTensor = module.tensors?.scales || module.tensors?.scale || module.tensors?.weight_scale;
    if (moduleLogical == null || bitWidth == null || !weightTensor || !scaleTensor) return false;
    const expectedLogicalBits = moduleLogical * bitWidth;
    const expectedStorageBits = BigInt(Number(weightTensor.byte_length || 0)) * 8n;
    const expectedPaddingBits = expectedStorageBits - expectedLogicalBits;
    const packedCapacity = exactDecimalInteger(module.packed_weight_code_capacity);
    const moduleScales = tensorCardinality(scaleTensor);
    const zeroCapacity = quantizedZeroCapacity(contract, module);
    if (packedCapacity == null || moduleScales == null || zeroCapacity == null || expectedPaddingBits < 0n) return false;
    if (exactDecimalInteger(module.logical_weight_element_count) !== moduleLogical
      || exactDecimalInteger(module.packed_weight_code_capacity) !== packedCapacity
      || exactDecimalInteger(module.logical_weight_bits) !== expectedLogicalBits
      || exactDecimalInteger(module.packed_weight_storage_bits) !== expectedStorageBits
      || exactDecimalInteger(module.packing_padding_bits) !== expectedPaddingBits
      || exactDecimalInteger(module.scale_element_count) !== moduleScales
      || exactDecimalInteger(module.zero_point_code_capacity) !== zeroCapacity) return false;
    logical += moduleLogical;
    packed += packedCapacity;
    logicalBits += expectedLogicalBits;
    storageBits += expectedStorageBits;
    paddingBits += expectedPaddingBits;
    scales += moduleScales;
    zeroes += zeroCapacity;
    const moduleBytes = Object.values(module.tensors || {}).reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
    if (!Number.isSafeInteger(moduleBytes) || moduleBytes !== Number(module.packed_tensor_bytes)) return false;
    bytes += moduleBytes;
  }
  const status = issueCount ? "fail" : "assessed";
  const activations = Array.isArray(contract.activation_quantization_contracts) ? contract.activation_quantization_contracts : [];
  if (Number(contract.activation_quantization_contract_count || 0) !== activations.length
    || activations.some((row) => (row.status === "fail") !== Boolean(row.issues?.length))) return false;
  if (!quantizationShardOwnershipValid(contract, modules, activations)) return false;
  return contract.status === status
    && Number(contract.module_count) === modules.length
    && Number(contract.valid_module_count) === valid
    && Number(contract.invalid_module_count) === modules.length - valid
    && exactDecimalInteger(contract.logical_weight_element_count) === logical
    && exactDecimalInteger(contract.packed_weight_code_capacity) === packed
    && exactDecimalInteger(contract.logical_weight_bits) === logicalBits
    && exactDecimalInteger(contract.packed_weight_storage_bits) === storageBits
    && exactDecimalInteger(contract.packing_padding_bits) === paddingBits
    && exactDecimalInteger(contract.scale_element_count) === scales
    && exactDecimalInteger(contract.zero_point_code_capacity) === zeroes
    && Number(contract.packed_tensor_bytes) === bytes
    && contract.packing_conservation_status === (paddingBits === 0n ? "exact_no_padding" : "exact_with_source_defined_padding");
}

function quantizedZeroCapacity(contract, module) {
  const tensor = module.tensors?.qzeros || module.tensors?.zero || module.tensors?.weight_zero_point;
  if (!tensor) return 0n;
  const cardinality = tensorCardinality(tensor);
  if (cardinality == null) return null;
  if (contract.method === "hqq") return cardinality;
  const bits = Number(module.bits);
  const pack = Number(contract.codes_per_storage_word ?? contract.pack_factor);
  if (contract.method === "compressed-tensors") return Number.isSafeInteger(bits) && bits > 0 ? cardinality * BigInt(Math.floor(32 / bits)) : null;
  return Number.isSafeInteger(pack) && pack > 0 ? cardinality * BigInt(pack) : null;
}

function quantizationShardOwnershipValid(contract, modules, activations) {
  const ownership = contract.shard_ownership;
  if (!ownership) return contract.method !== "hqq" && contract.method !== "compressed-tensors";
  const expected = [];
  for (const module of modules) for (const [role, tensor] of Object.entries(module.tensors || {})) {
    expected.push([module.name, role, tensor.tensor_name, tensor.shard_path || null]);
  }
  for (const row of activations) for (const role of ["scale_tensor", "zero_point_tensor", "k_scale_tensor", "v_scale_tensor"]) {
    const tensor = row[role];
    if (tensor) expected.push([row.module_name, `${row.kind}:${role}`, tensor.tensor_name, tensor.shard_path || null]);
  }
  const deduped = [...new Map(expected.map((entry) => [JSON.stringify(entry.slice(0, 3)), entry])).values()];
  const observed = Array.isArray(ownership.rows) ? ownership.rows.map((row) => [row.module_name, row.tensor_role, row.tensor_name, row.shard_path || null]) : [];
  if (JSON.stringify(observed) !== JSON.stringify(deduped)) return false;
  const bound = deduped.filter((row) => row[3]).length;
  const status = bound ? bound === deduped.length ? "assessed_all_quantization_tensors_shard_bound" : "fail_partial_shard_ownership" : "not_applicable_single_file_or_unannotated_input";
  return Number(ownership.tensor_count) === deduped.length
    && Number(ownership.shard_bound_tensor_count) === bound
    && ownership.status === status;
}

function tensorCardinality(tensor) {
  return tensor ? exactProduct(tensor.shape) : null;
}

function exactProduct(values) {
  if (!Array.isArray(values) || values.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) < 1)) return null;
  return values.reduce((product, value) => product * BigInt(value), 1n);
}
