function positive(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive safe integer`);
  return BigInt(value);
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

export function buildSparseMoeDecoderProjection({
  hiddenSize, intermediateSize, layerCount, attentionHeadCount, kvHeadCount,
  headWidth, contextLength, expertCount, activeExpertCount,
}) {
  const hidden = positive(hiddenSize, "hidden size");
  const intermediate = positive(intermediateSize, "intermediate size");
  const layers = positive(layerCount, "layer count");
  const attentionHeads = positive(attentionHeadCount, "attention head count");
  const kvHeads = positive(kvHeadCount, "KV head count");
  const width = positive(headWidth, "head width");
  const context = positive(contextLength, "context length");
  const experts = positive(expertCount, "expert count");
  const activeExperts = positive(activeExpertCount, "active expert count");
  if (activeExperts > experts) throw new Error("active expert count cannot exceed expert count");
  const queryWidth = attentionHeads * width;
  const kvWidth = kvHeads * width;
  const attentionProjectionPerLayerToken = hidden * queryWidth * 2n + hidden * kvWidth * 2n;
  const routerPerLayerToken = hidden * experts;
  const expertMatrixPerExpert = hidden * intermediate * 3n;
  const totalExpertMatrixParameters = expertMatrixPerExpert * experts * layers;
  const activeExpertMatrixPerLayerToken = expertMatrixPerExpert * activeExperts;
  const activeProjectionPerLayerToken = attentionProjectionPerLayerToken + routerPerLayerToken + activeExpertMatrixPerLayerToken;
  const activeProjectionAllLayersToken = activeProjectionPerLayerToken * layers;
  const causalPairs = context * (context + 1n) / 2n;
  const prefillAttention = queryWidth * causalPairs * 2n * layers;
  const decodeAttention = queryWidth * context * 2n * layers;
  return {
    schema: "deepbom.sparse_moe_decoder_projection.v1",
    status: "assessed_source_bound_active_path_scenario",
    evidence_class: "SOURCE_PINNED/DERIVED_SCENARIO",
    expert_count: Number(experts),
    active_expert_count_per_token: Number(activeExperts),
    routing_fraction_of_experts: Number(activeExperts) / Number(experts),
    router_macs_per_layer_per_token: exact(routerPerLayerToken),
    expert_matrix_parameters_per_expert_per_layer: exact(expertMatrixPerExpert),
    total_expert_matrix_parameters_all_layers: exact(totalExpertMatrixParameters),
    active_expert_matrix_macs_per_layer_per_token: exact(activeExpertMatrixPerLayerToken),
    attention_projection_macs_per_layer_per_token: exact(attentionProjectionPerLayerToken),
    active_projection_macs_per_layer_per_token: exact(activeProjectionPerLayerToken),
    active_projection_macs_all_layers_per_token: exact(activeProjectionAllLayersToken),
    prefill_active_core_macs_at_declared_context: exact(activeProjectionAllLayersToken * context + prefillAttention),
    decode_active_core_macs_at_declared_context: exact(activeProjectionAllLayersToken + decodeAttention),
    formula: "router H*E + top_k*(3*H*I) + Q/K/V/O projections; attention QK/PV uses the declared all-global causal scenario",
    boundary: "Total expert parameters describe serialized expert-matrix residency. Active MAC rows assume exactly top-k experts are evaluated for each token and exclude routing implementation overhead, token sorting, capacity policy, expert imbalance, communication, fusion, activation, normalization, and backend lowering.",
  };
}

export function buildSsmRecurrentStateProjection({
  layerCount, intermediateSize, stateSize, convolutionKernel,
}) {
  const layers = positive(layerCount, "layer count");
  const intermediate = positive(intermediateSize, "intermediate size");
  const state = positive(stateSize, "state size");
  const kernel = positive(convolutionKernel, "convolution kernel");
  const convPerLayerBatch = intermediate * kernel;
  const ssmPerLayerBatch = intermediate * state;
  const totalPerBatch = layers * (convPerLayerBatch + ssmPerLayerBatch);
  return {
    schema: "deepbom.ssm_recurrent_state_projection.v1",
    status: "assessed_element_cardinality",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED",
    layer_count: Number(layers),
    intermediate_size: Number(intermediate),
    state_size: Number(state),
    convolution_kernel: Number(kernel),
    convolution_state_elements_per_layer_per_batch: exact(convPerLayerBatch),
    ssm_state_elements_per_layer_per_batch: exact(ssmPerLayerBatch),
    recurrent_state_elements_all_layers_per_batch: exact(totalPerBatch),
    context_dependency: "constant_per_batch_after_initialization",
    formula: "layers * intermediate_size * (conv_kernel + state_size) elements per batch",
    boundary: "This is recurrent cache cardinality from the pinned MambaCache tensor shapes, not transformer KV state. It excludes allocator alignment, temporary selective-scan tensors, paging, residency, and backend-private workspaces.",
  };
}

export function buildSsmLinearComputeProjection({
  hiddenSize, intermediateSize, layerCount, stateSize, timeStepRank, convolutionKernel, contextLength = null,
}) {
  const hidden = positive(hiddenSize, "hidden size");
  const intermediate = positive(intermediateSize, "intermediate size");
  const layers = positive(layerCount, "layer count");
  const state = positive(stateSize, "state size");
  const rank = positive(timeStepRank, "time-step rank");
  const kernel = positive(convolutionKernel, "convolution kernel");
  const inProjection = hidden * intermediate * 2n;
  const xProjection = intermediate * (rank + state * 2n);
  const dtProjection = rank * intermediate;
  const outProjection = intermediate * hidden;
  const depthwiseConvolution = intermediate * kernel;
  const perLayerToken = inProjection + xProjection + dtProjection + outProjection + depthwiseConvolution;
  const allLayersToken = perLayerToken * layers;
  const context = contextLength == null ? null : positive(contextLength, "context length");
  return {
    schema: "deepbom.ssm_linear_compute_projection.v1",
    status: "assessed_source_bound_projection_scenario",
    evidence_class: "SOURCE_PINNED/DERIVED_SCENARIO",
    input_projection_macs_per_layer_per_token: exact(inProjection),
    state_parameter_projection_macs_per_layer_per_token: exact(xProjection),
    time_step_projection_macs_per_layer_per_token: exact(dtProjection),
    output_projection_macs_per_layer_per_token: exact(outProjection),
    depthwise_convolution_macs_per_layer_per_token_steady_state: exact(depthwiseConvolution),
    accounted_macs_per_layer_per_token: exact(perLayerToken),
    accounted_macs_all_layers_per_token: exact(allLayersToken),
    accounted_macs_all_layers_at_declared_context: context == null ? null : exact(allLayersToken * context),
    formula: "H*(2I) + I*(R+2S) + R*I + I*H + I*K MACs per layer per token",
    boundary: "The row counts pinned linear projections and steady-state depthwise convolution only. Selective-scan discretization, recurrent elementwise updates, activation, normalization, residual addition, optimized parallel scan, temporary tensors, and backend fusion are intentionally excluded rather than estimated.",
  };
}

export function buildHybridJambaComputeProjection({
  vocabularySize, hiddenSize, intermediateSize, attentionLayerCount, mambaLayerCount,
  denseFeedForwardLayerCount, expertFeedForwardLayerCount, attentionHeadCount,
  kvHeadCount, headWidth, contextLength, expertCount, activeExpertCount,
  stateSize, timeStepRank, convolutionKernel, mambaIntermediateSize,
}) {
  const hidden = positive(hiddenSize, "hidden size");
  const intermediate = positive(intermediateSize, "feed-forward intermediate size");
  const attentionLayers = positive(attentionLayerCount, "attention layer count");
  const mambaLayers = positive(mambaLayerCount, "Mamba layer count");
  const denseLayers = BigInt(denseFeedForwardLayerCount);
  const expertLayers = BigInt(expertFeedForwardLayerCount);
  if (denseLayers < 0n || expertLayers < 0n || denseLayers + expertLayers !== attentionLayers + mambaLayers) {
    throw new Error("Jamba feed-forward layer counts must be non-negative and conserve total layers");
  }
  const attentionHeads = positive(attentionHeadCount, "attention head count");
  const kvHeads = positive(kvHeadCount, "KV head count");
  const width = positive(headWidth, "head width");
  const context = positive(contextLength, "context length");
  const experts = positive(expertCount, "expert count");
  const activeExperts = positive(activeExpertCount, "active expert count");
  if (activeExperts > experts) throw new Error("active expert count cannot exceed expert count");
  const mambaIntermediate = positive(mambaIntermediateSize, "Mamba intermediate size");
  const state = positive(stateSize, "state size");
  const rank = positive(timeStepRank, "time-step rank");
  const kernel = positive(convolutionKernel, "convolution kernel");
  const queryWidth = attentionHeads * width;
  const kvWidth = kvHeads * width;
  const attentionProjection = hidden * queryWidth * 2n + hidden * kvWidth * 2n;
  const attentionProjectionAll = attentionProjection * attentionLayers;
  const mambaPerLayer = hidden * mambaIntermediate * 2n
    + mambaIntermediate * (rank + state * 2n)
    + rank * mambaIntermediate + mambaIntermediate * hidden + mambaIntermediate * kernel;
  const mambaAll = mambaPerLayer * mambaLayers;
  const denseFfPerLayer = hidden * intermediate * 3n;
  const denseFfAll = denseFfPerLayer * denseLayers;
  const routerPerExpertLayer = hidden * experts;
  const activeExpertPerLayer = denseFfPerLayer * activeExperts;
  const activeExpertFfAll = (routerPerExpertLayer + activeExpertPerLayer) * expertLayers;
  const totalExpertParameters = denseFfPerLayer * experts * expertLayers;
  const accountedAllLayersToken = attentionProjectionAll + mambaAll + denseFfAll + activeExpertFfAll;
  const causalPairs = context * (context + 1n) / 2n;
  const prefillAttention = queryWidth * causalPairs * 2n * attentionLayers;
  const decodeAttention = queryWidth * context * 2n * attentionLayers;
  const logits = hidden * positive(vocabularySize, "vocabulary size");
  return {
    schema: "deepbom.hybrid_jamba_compute_projection.v1",
    status: "assessed_source_bound_component_scenario",
    evidence_class: "SOURCE_PINNED/DERIVED_SCENARIO",
    attention_layer_count: Number(attentionLayers),
    mamba_layer_count: Number(mambaLayers),
    dense_feed_forward_layer_count: Number(denseLayers),
    expert_feed_forward_layer_count: Number(expertLayers),
    attention_projection_macs_all_attention_layers_per_token: exact(attentionProjectionAll),
    mamba_projection_and_depthwise_macs_all_mamba_layers_per_token: exact(mambaAll),
    dense_feed_forward_macs_all_dense_layers_per_token: exact(denseFfAll),
    active_moe_macs_all_expert_layers_per_token: exact(activeExpertFfAll),
    total_expert_matrix_parameters_all_expert_layers: exact(totalExpertParameters),
    accounted_projection_macs_all_layers_per_token: exact(accountedAllLayersToken),
    prefill_accounted_core_macs_at_declared_context: exact(accountedAllLayersToken * context + prefillAttention),
    decode_accounted_core_macs_at_declared_context: exact(accountedAllLayersToken + decodeAttention),
    output_projection_macs_per_logit_position: exact(logits),
    decode_with_one_logit_position_macs: exact(accountedAllLayersToken + decodeAttention + logits),
    formula: "attention Q/K/V/O on attention layers + Mamba pinned linear/depthwise terms on Mamba layers + dense FFN or router/top-k expert FFN according to the configured layer schedules",
    boundary: "The component scenario follows pinned Jamba layer schedules. Attention uses an all-global causal scenario. Selective-scan arithmetic, activations, normalization, routing implementation, temporary tensors, fusion, communication, and backend lowering are excluded rather than estimated.",
  };
}

export function buildStateStorageScenarios(projection, { batches = [1, 2, 4], storageBits = [8, 16, 32] } = {}) {
  const decimal = projection?.recurrent_state_elements_all_layers_per_batch?.decimal;
  if (!/^\d+$/.test(String(decimal || ""))) return [];
  const elements = BigInt(decimal);
  return batches.flatMap((batch) => storageBits.map((bits) => {
    if (!Number.isSafeInteger(batch) || batch <= 0 || !Number.isSafeInteger(bits) || bits <= 0 || bits % 8 !== 0) throw new Error("state scenario dimensions must be positive whole-byte integers");
    return {
      batch_size: batch,
      storage_bits: bits,
      logical_bytes: exact(elements * BigInt(batch) * BigInt(bits / 8)),
      evidence_class: "DERIVED_CONDITIONAL_SCENARIO",
    };
  }));
}
