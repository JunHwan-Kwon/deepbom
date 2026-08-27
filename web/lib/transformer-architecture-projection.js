function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive safe integer`);
  return BigInt(value);
}

function safeNumber(value) {
  return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function exact(value) {
  return { value: safeNumber(value), decimal: String(value) };
}

export function buildKvStateProjection({
  layerCount,
  kvHeadCount,
  keyHeadWidth,
  valueHeadWidth,
  contextLength,
  storageBits = null,
}) {
  const layers = positiveInteger(layerCount, "layer count");
  const kvHeads = positiveInteger(kvHeadCount, "KV head count");
  const keyWidth = positiveInteger(keyHeadWidth, "key head width");
  const valueWidth = positiveInteger(valueHeadWidth, "value head width");
  const context = positiveInteger(contextLength, "context length");
  if (storageBits != null && (!Number.isSafeInteger(storageBits) || storageBits <= 0 || storageBits % 8 !== 0)) {
    throw new Error("KV storage width must be a positive whole-byte bit width");
  }
  const elementsPerTokenBatch = layers * kvHeads * (keyWidth + valueWidth);
  const elementsAtContextBatchOne = elementsPerTokenBatch * context;
  const bytesPerTokenBatch = storageBits == null ? null : elementsPerTokenBatch * BigInt(storageBits / 8);
  const bytesAtContextBatchOne = bytesPerTokenBatch == null ? null : bytesPerTokenBatch * context;
  return {
    schema: "deepbom.transformer_kv_state_projection.v1",
    status: "assessed_element_cardinality",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED",
    layer_count: Number(layers),
    kv_head_count: Number(kvHeads),
    key_head_width: Number(keyWidth),
    value_head_width: Number(valueWidth),
    context_length: Number(context),
    elements_per_token_per_batch: exact(elementsPerTokenBatch),
    elements_at_context_batch_one: exact(elementsAtContextBatchOne),
    conditional_storage_bits: storageBits,
    bytes_per_token_per_batch_if_storage_width_matches: bytesPerTokenBatch == null ? null : exact(bytesPerTokenBatch),
    bytes_at_context_batch_one_if_storage_width_matches: bytesAtContextBatchOne == null ? null : exact(bytesAtContextBatchOne),
    formula: "layers * kv_heads * (key_head_width + value_head_width) elements per token per batch",
    boundary: "Element cardinality follows the declared attention-state dimensions. Byte values apply only when the runtime cache uses the supplied storage width and exclude allocator layout, paging, alignment, residency, and backend-private state.",
  };
}

function buildDecoderProjection({
  vocabularySize,
  hiddenSize,
  intermediateSize,
  layerCount,
  attentionHeadCount,
  kvHeadCount,
  headWidth,
  contextLength,
  mlpProjectionMatrixCount,
  schema,
  mlpField,
  mlpLabel,
}) {
  const vocabulary = positiveInteger(vocabularySize, "vocabulary size");
  const hidden = positiveInteger(hiddenSize, "hidden size");
  const intermediate = positiveInteger(intermediateSize, "intermediate size");
  const layers = positiveInteger(layerCount, "layer count");
  const attentionHeads = positiveInteger(attentionHeadCount, "attention head count");
  positiveInteger(kvHeadCount, "KV head count");
  const width = positiveInteger(headWidth, "attention head width");
  const context = positiveInteger(contextLength, "context length");
  const mlpMatrices = positiveInteger(mlpProjectionMatrixCount, "MLP projection matrix count");
  const queryWidth = attentionHeads * width;
  const kvWidth = BigInt(kvHeadCount) * width;
  const attentionProjectionPerLayerToken = hidden * queryWidth * 2n + hidden * kvWidth * 2n;
  const mlpProjectionPerLayerToken = hidden * intermediate * mlpMatrices;
  const denseProjectionPerLayerToken = attentionProjectionPerLayerToken + mlpProjectionPerLayerToken;
  const denseProjectionAllLayersToken = denseProjectionPerLayerToken * layers;
  const causalPairs = context * (context + 1n) / 2n;
  const prefillAttentionAllLayers = queryWidth * causalPairs * 2n * layers;
  const decodeAttentionAllLayers = queryWidth * context * 2n * layers;
  const prefillDenseAllLayers = denseProjectionAllLayersToken * context;
  const prefillCore = prefillDenseAllLayers + prefillAttentionAllLayers;
  const decodeCore = denseProjectionAllLayersToken + decodeAttentionAllLayers;
  const logitsPerPosition = hidden * vocabulary;
  const result = {
    schema,
    status: "assessed_declared_all_global_scenario",
    evidence_class: "SOURCE_PINNED/DERIVED_SCENARIO",
    query_projection_width: exact(queryWidth),
    kv_projection_width: exact(kvWidth),
    attention_projection_macs_per_layer_per_token: exact(attentionProjectionPerLayerToken),
    mlp_projection_matrix_count: Number(mlpMatrices),
    dense_projection_macs_per_layer_per_token: exact(denseProjectionPerLayerToken),
    dense_projection_macs_all_layers_per_token: exact(denseProjectionAllLayersToken),
    declared_context_causal_attention_pair_count: exact(causalPairs),
    prefill_attention_macs_all_layers_at_declared_context: exact(prefillAttentionAllLayers),
    decode_attention_macs_all_layers_at_declared_context: exact(decodeAttentionAllLayers),
    prefill_dense_projection_macs_all_layers_at_declared_context: exact(prefillDenseAllLayers),
    prefill_transformer_core_macs_at_declared_context: exact(prefillCore),
    decode_transformer_core_macs_at_declared_context: exact(decodeCore),
    output_projection_macs_per_logit_position: exact(logitsPerPosition),
    prefill_with_logits_at_every_position_macs: exact(prefillCore + logitsPerPosition * context),
    decode_with_one_logit_position_macs: exact(decodeCore + logitsPerPosition),
    mac_definition: `One matrix multiply-accumulate is one MAC. Dense projection terms include Q/K/V/O and ${Number(mlpMatrices)} ${mlpLabel} MLP matrices. Attention terms include QK and probability-value products over an all-global causal scenario.`,
    boundary: "This is a source-bound architecture scenario, not a reconstructed execution DAG or runtime count. It excludes embedding lookup, normalization, RoPE, activation, softmax, masking, bias, cache movement, local/sliding attention reductions, sparsity, fusion, and backend lowering. The all-global prefill row is an explicit scenario and may exceed models with local or sliding attention.",
  };
  result[mlpField] = exact(mlpProjectionPerLayerToken);
  return result;
}

export function buildCanonicalDecoderProjection(options) {
  return buildDecoderProjection({
    ...options,
    schema: "deepbom.canonical_decoder_compute_projection.v1",
    mlpField: "mlp_projection_macs_per_layer_per_token",
    mlpLabel: "declared",
  });
}

export function buildCanonicalGatedDecoderProjection(options) {
  return buildDecoderProjection({
    ...options,
    mlpProjectionMatrixCount: 3,
    schema: "deepbom.canonical_gated_decoder_compute_projection.v1",
    mlpField: "gated_mlp_projection_macs_per_layer_per_token",
    mlpLabel: "gated",
  });
}
