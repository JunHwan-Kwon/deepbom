import { buildCanonicalDecoderProjection, buildCanonicalGatedDecoderProjection, buildKvStateProjection } from "./transformer-architecture-projection.js";
import { buildHybridJambaComputeProjection, buildSparseMoeDecoderProjection, buildSsmLinearComputeProjection, buildSsmRecurrentStateProjection } from "./llm-specialized-projection.js";

export const HF_SAFETENSORS_ARCHITECTURE_SOURCE = Object.freeze({
  repository: "huggingface/transformers",
  source_commit: "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc",
  release_tag: "v4.57.1",
  configuration_sources: Object.freeze({
    llama: Object.freeze({ path: "src/transformers/models/llama/configuration_llama.py", sha256: "9becb04559331628af1b4b72f44d68a07265bd0c95e383d11d4c5da5ed3a408e" }),
    mistral: Object.freeze({ path: "src/transformers/models/mistral/configuration_mistral.py", sha256: "be9b1b8c07689c5e30daf314e88d4efc664838fa5c369f7e3c4b48574a472782" }),
    qwen2: Object.freeze({ path: "src/transformers/models/qwen2/configuration_qwen2.py", sha256: "2b9d0a03feeef734ae451eee381ad15deefc02ab4b8b957d87803fb5894e9bd3" }),
    qwen3: Object.freeze({ path: "src/transformers/models/qwen3/configuration_qwen3.py", sha256: "27863e9718fdbc899f2d0e567621e4d3d36d8dc500c1d54b49dba4242d08d2bd" }),
    gemma: Object.freeze({ path: "src/transformers/models/gemma/configuration_gemma.py", sha256: "2d69e8686cf9df1c4f21b653d56cd012e17b46dfcca653d543d36f3f9e5713d2" }),
    gemma2: Object.freeze({ path: "src/transformers/models/gemma2/configuration_gemma2.py", sha256: "41f35c2f2d81a4902405a7561e658fce1e00296d5b30754e40cb22ccdf25520b" }),
    olmo2: Object.freeze({ path: "src/transformers/models/olmo2/configuration_olmo2.py", sha256: "92a0639c0b3493ea59f98f3334c45643eea06aff98a4ea60ea8cbd2d9f808691" }),
    granite: Object.freeze({ path: "src/transformers/models/granite/configuration_granite.py", sha256: "535090da0bd3606c7be77517d2de4839f70b9658a40d4ec9ba98fb365397dc39" }),
    phi3: Object.freeze({ path: "src/transformers/models/phi3/configuration_phi3.py", sha256: "61a99df279323dd0bf02d08f03dc34b634848da066c6018abf10349f41482007" }),
    cohere: Object.freeze({ path: "src/transformers/models/cohere/configuration_cohere.py", sha256: "72e34ea45166961f9aa966d8eccefbd60af511fa3849ca21cc9ec05cdd5c5c0b" }),
    cohere2: Object.freeze({ path: "src/transformers/models/cohere2/configuration_cohere2.py", sha256: "5194652c0eab78a8e7637d135b11407550a84bb51f1c0013d8431ce5b6c2b0db" }),
    nemotron: Object.freeze({ path: "src/transformers/models/nemotron/configuration_nemotron.py", sha256: "40c1ff330e3c6600afa2f7c4dccb5433a5bfdf80d551beee9e9af6ec96954c88" }),
    ministral: Object.freeze({ path: "src/transformers/models/ministral/configuration_ministral.py", sha256: "4ef894f622a4bdea11f151729fb07fe2ac2cee3338d15237844d1952c1f083c1" }),
    smollm3: Object.freeze({ path: "src/transformers/models/smollm3/configuration_smollm3.py", sha256: "4fec68b228fec767ddc8f8cd4c40f8eaf374c3c44bf38c0cf1618382c7731cdf" }),
    exaone4: Object.freeze({ path: "src/transformers/models/exaone4/configuration_exaone4.py", sha256: "e011e8f1d04ea443eda7ee27f9957dad207e26d906765218ad6b6013ac398b01" }),
    olmo: Object.freeze({ path: "src/transformers/models/olmo/configuration_olmo.py", sha256: "b3f628b68bd4b69f95981eeaf515780754ca823869de013a51caa0ef62371897" }),
    mixtral: Object.freeze({ path: "src/transformers/models/mixtral/configuration_mixtral.py", sha256: "ece31b83ca5f694d167d89c6208aaae670f3c7df31f0d3ca8be34e6f7a1201d9" }),
    mamba: Object.freeze({ path: "src/transformers/models/mamba/configuration_mamba.py", sha256: "92b86ded08fed727d8c5d32bdf307d5a1055a942a220db76a3906f0c2f2ff972" }),
    jamba: Object.freeze({ path: "src/transformers/models/jamba/configuration_jamba.py", sha256: "fed02758507b0f11afe0c17776adab4a6cc9dd8b3ef236a6904b4debb9933d64" }),
  }),
  modeling_sources: Object.freeze({
    llama: Object.freeze({ path: "src/transformers/models/llama/modeling_llama.py", sha256: "31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26" }),
    mistral: Object.freeze({ path: "src/transformers/models/mistral/modeling_mistral.py", sha256: "3fbb6202376a1fdf57166d199dd4d43f81226b1e0634ec8b61dcb1ade7de153e" }),
    qwen2: Object.freeze({ path: "src/transformers/models/qwen2/modeling_qwen2.py", sha256: "a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb" }),
    qwen3: Object.freeze({ path: "src/transformers/models/qwen3/modeling_qwen3.py", sha256: "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b" }),
    gemma: Object.freeze({ path: "src/transformers/models/gemma/modeling_gemma.py", sha256: "79acf192010319dee40e33b9e65bc72f73056034a67f5af8553bd232a9e7008d" }),
    gemma2: Object.freeze({ path: "src/transformers/models/gemma2/modeling_gemma2.py", sha256: "2c56ff2202842de31d2eab3bed6599b6d0f64facff12ab93d8543f57bd4698b0" }),
    olmo2: Object.freeze({ path: "src/transformers/models/olmo2/modeling_olmo2.py", sha256: "9524d330f6d36722eaf29dcf6deb29a9e812b4ce5298b30d94f6b616a4fcb6d1" }),
    granite: Object.freeze({ path: "src/transformers/models/granite/modeling_granite.py", sha256: "920678d503bcb6795ba46c1b9579c28aad208a3ff0b73e7e02754e7cd9e3c19c" }),
    phi3: Object.freeze({ path: "src/transformers/models/phi3/modeling_phi3.py", sha256: "3b2e19b8396d472c6bad2cc86d10950c34aa5a13debcd9288daa8ee026f7c3bd" }),
    cohere: Object.freeze({ path: "src/transformers/models/cohere/modeling_cohere.py", sha256: "95461410624af84f8ffefbbe5b3756c444b504388d6894efd8a068ce740a1b7b" }),
    cohere2: Object.freeze({ path: "src/transformers/models/cohere2/modeling_cohere2.py", sha256: "aaa7a1c6fbd9df2d5d0289e16f3b11ffe2dce80a70474041b1a26cad71a534cc" }),
    nemotron: Object.freeze({ path: "src/transformers/models/nemotron/modeling_nemotron.py", sha256: "a7d2f1a96cd637a6b165535f887b672e4230d53914f6d92c897fe48305df9a1e" }),
    ministral: Object.freeze({ path: "src/transformers/models/ministral/modeling_ministral.py", sha256: "db704bf8835dfe246ea4c530068b576a1cc75acd6c72180c435ced4400ce44ad" }),
    smollm3: Object.freeze({ path: "src/transformers/models/smollm3/modeling_smollm3.py", sha256: "06816c41869a75e66dc78468987c75f64349450c44c980b786a20b22041596ec" }),
    exaone4: Object.freeze({ path: "src/transformers/models/exaone4/modeling_exaone4.py", sha256: "24e041fed3376840f2e2dc669479875793139e91902800627029709f8e014209" }),
    olmo: Object.freeze({ path: "src/transformers/models/olmo/modeling_olmo.py", sha256: "d2e98af440964388f95c3ea0301b75715e2ae6d475708b006337213faac0fe4a" }),
    mixtral: Object.freeze({ path: "src/transformers/models/mixtral/modeling_mixtral.py", sha256: "26078dc562c0538d86d0ef5fb92c5260de8bc1f250c756410c29e2e2e19c072d" }),
    mamba: Object.freeze({ path: "src/transformers/models/mamba/modeling_mamba.py", sha256: "4a5deb1cc9c22757826ac5dda343f5a0c3b928c50ac5d9c7c860e247bce06cdf" }),
    jamba: Object.freeze({ path: "src/transformers/models/jamba/modeling_jamba.py", sha256: "e5979bd84d076e56f4eb0d04d5cebad4a4205ce84d7e631a930489c5d19c4883" }),
  }),
  interpretation_boundary: "Pinned configuration classes define architecture fields/defaults and pinned modeling classes define the registered canonical state-dict modules. Tensor checks do not reconstruct an executable graph or establish runtime cache allocation.",
});

const STANDARD_LAYOUT = Object.freeze({
  id: "split_qkv_split_gated_mlp",
  attention: "split_qkv",
  mlp: "split_gated",
  norms: Object.freeze([
    Object.freeze({ suffix: "input_layernorm", width: "hidden" }),
    Object.freeze({ suffix: "post_attention_layernorm", width: "hidden" }),
  ]),
  mlp_projection_matrix_count: 3,
  config_flags: Object.freeze({}),
  attention_bias_scope: "none",
});

const PARALLEL_RESIDUAL_LAYOUT = Object.freeze({
  ...STANDARD_LAYOUT,
  id: "split_qkv_parallel_residual_split_gated_mlp",
  norms: Object.freeze([Object.freeze({ suffix: "input_layernorm", shape: Object.freeze(["hidden"]) })]),
});

const UNGATED_LAYOUT = Object.freeze({
  ...STANDARD_LAYOUT,
  id: "split_qkv_split_ungated_mlp",
  mlp: "split_ungated",
  mlp_projection_matrix_count: 2,
});

const ARCHITECTURES = Object.freeze({
  llama: Object.freeze({
    defaults: Object.freeze({ vocab_size: 32000, hidden_size: 4096, intermediate_size: 11008, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: null, max_position_embeddings: 2048, head_dim: null, tie_word_embeddings: false, attention_bias: false, mlp_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention", mlp_bias: "mlp" }) }),
  }),
  mistral: Object.freeze({
    defaults: Object.freeze({ vocab_size: 32000, hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, max_position_embeddings: 131072, head_dim: null, tie_word_embeddings: false, sliding_window: 4096 }),
    tensor_layout: STANDARD_LAYOUT,
  }),
  qwen2: Object.freeze({
    defaults: Object.freeze({ vocab_size: 151936, hidden_size: 4096, intermediate_size: 22016, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 32, max_position_embeddings: 32768, head_dim: null, tie_word_embeddings: false, attention_bias: true, sliding_window: null }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "qkv" }),
  }),
  qwen3: Object.freeze({
    defaults: Object.freeze({ vocab_size: 151936, hidden_size: 4096, intermediate_size: 22016, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 32, max_position_embeddings: 32768, head_dim: 128, tie_word_embeddings: false, attention_bias: false, sliding_window: 4096 }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, id: "split_qkv_qk_head_norm_split_gated_mlp", attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }), norms: Object.freeze([
      ...STANDARD_LAYOUT.norms,
      Object.freeze({ suffix: "self_attn.q_norm", width: "head" }),
      Object.freeze({ suffix: "self_attn.k_norm", width: "head" }),
    ]) }),
  }),
  gemma: Object.freeze({
    defaults: Object.freeze({ vocab_size: 256000, hidden_size: 3072, intermediate_size: 24576, num_hidden_layers: 28, num_attention_heads: 16, num_key_value_heads: 16, max_position_embeddings: 8192, head_dim: 256, tie_word_embeddings: true, attention_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }) }),
  }),
  gemma2: Object.freeze({
    defaults: Object.freeze({ vocab_size: 256000, hidden_size: 2304, intermediate_size: 9216, num_hidden_layers: 26, num_attention_heads: 8, num_key_value_heads: 4, max_position_embeddings: 8192, head_dim: 256, tie_word_embeddings: true, attention_bias: false, sliding_window: 4096 }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, id: "split_qkv_four_norm_split_gated_mlp", attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }), norms: Object.freeze([
      ...STANDARD_LAYOUT.norms,
      Object.freeze({ suffix: "pre_feedforward_layernorm", width: "hidden" }),
      Object.freeze({ suffix: "post_feedforward_layernorm", width: "hidden" }),
    ]) }),
  }),
  olmo2: Object.freeze({
    defaults: Object.freeze({ vocab_size: 50304, hidden_size: 4096, intermediate_size: 11008, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: null, max_position_embeddings: 2048, head_dim: null, tie_word_embeddings: false, attention_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, id: "split_qkv_full_width_qk_norm_post_norms", attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }), norms: Object.freeze([
      Object.freeze({ suffix: "post_attention_layernorm", width: "hidden" }),
      Object.freeze({ suffix: "post_feedforward_layernorm", width: "hidden" }),
      Object.freeze({ suffix: "self_attn.q_norm", width: "query" }),
      Object.freeze({ suffix: "self_attn.k_norm", width: "kv" }),
    ]) }),
  }),
  granite: Object.freeze({
    defaults: Object.freeze({ vocab_size: 32000, hidden_size: 4096, intermediate_size: 11008, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: null, max_position_embeddings: 2048, head_dim: null, tie_word_embeddings: false, attention_bias: false, mlp_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention", mlp_bias: "mlp" }) }),
  }),
  phi3: Object.freeze({
    defaults: Object.freeze({ vocab_size: 32064, hidden_size: 3072, intermediate_size: 8192, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: null, max_position_embeddings: 4096, head_dim: null, tie_word_embeddings: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, id: "fused_qkv_fused_gate_up_mlp", attention: "fused_qkv", mlp: "fused_gate_up" }),
  }),
  cohere: Object.freeze({
    defaults: Object.freeze({ vocab_size: 256000, hidden_size: 8192, intermediate_size: 22528, num_hidden_layers: 40, num_attention_heads: 64, num_key_value_heads: null, max_position_embeddings: 8192, head_dim: null, tie_word_embeddings: true, attention_bias: false, use_qk_norm: false }),
    tensor_layout: Object.freeze({
      ...PARALLEL_RESIDUAL_LAYOUT,
      id: "split_qkv_parallel_residual_optional_qk_norm_split_gated_mlp",
      attention_bias_scope: "config_all",
      config_flags: Object.freeze({ attention_bias: "attention", use_qk_norm: "cohere_qk_norm" }),
    }),
  }),
  cohere2: Object.freeze({
    defaults: Object.freeze({ vocab_size: 256000, hidden_size: 8192, intermediate_size: 22528, num_hidden_layers: 40, num_attention_heads: 64, num_key_value_heads: null, max_position_embeddings: 8192, head_dim: null, tie_word_embeddings: true, attention_bias: false, sliding_window: 4096 }),
    tensor_layout: Object.freeze({ ...PARALLEL_RESIDUAL_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }) }),
  }),
  nemotron: Object.freeze({
    defaults: Object.freeze({ vocab_size: 256000, hidden_size: 6144, intermediate_size: 24576, num_hidden_layers: 32, num_attention_heads: 48, num_key_value_heads: null, max_position_embeddings: 4096, head_dim: null, tie_word_embeddings: false, attention_bias: false, mlp_bias: false }),
    tensor_layout: Object.freeze({ ...UNGATED_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention", mlp_bias: "mlp" }) }),
  }),
  ministral: Object.freeze({
    defaults: Object.freeze({ vocab_size: 32000, hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, max_position_embeddings: 131072, head_dim: null, tie_word_embeddings: false, sliding_window: 4096 }),
    tensor_layout: STANDARD_LAYOUT,
  }),
  smollm3: Object.freeze({
    defaults: Object.freeze({ vocab_size: 128256, hidden_size: 2048, intermediate_size: 11008, num_hidden_layers: 36, num_attention_heads: 16, num_key_value_heads: 4, max_position_embeddings: 32768, head_dim: null, tie_word_embeddings: true, attention_bias: false, mlp_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention", mlp_bias: "mlp" }) }),
  }),
  exaone4: Object.freeze({
    defaults: Object.freeze({ vocab_size: 102400, hidden_size: 4096, intermediate_size: 16384, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 32, max_position_embeddings: 2048, head_dim: null, tie_word_embeddings: false, sliding_window: 4096 }),
    tensor_layout: Object.freeze({
      ...STANDARD_LAYOUT,
      id: "split_qkv_head_qk_norm_post_norms_split_gated_mlp",
      norms: Object.freeze([
        Object.freeze({ suffix: "post_attention_layernorm", shape: Object.freeze(["hidden"]) }),
        Object.freeze({ suffix: "post_feedforward_layernorm", shape: Object.freeze(["hidden"]) }),
        Object.freeze({ suffix: "self_attn.q_norm", shape: Object.freeze(["head"]) }),
        Object.freeze({ suffix: "self_attn.k_norm", shape: Object.freeze(["head"]) }),
      ]),
    }),
  }),
  olmo: Object.freeze({
    defaults: Object.freeze({ vocab_size: 50304, hidden_size: 4096, intermediate_size: 11008, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: null, max_position_embeddings: 2048, head_dim: null, tie_word_embeddings: false, attention_bias: false }),
    tensor_layout: Object.freeze({ ...STANDARD_LAYOUT, attention_bias_scope: "config_all", config_flags: Object.freeze({ attention_bias: "attention" }) }),
  }),
  mixtral: Object.freeze({
    kind: "sparse_moe_decoder",
    defaults: Object.freeze({ vocab_size: 32000, hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, max_position_embeddings: 131072, head_dim: null, num_experts_per_tok: 2, num_local_experts: 8, tie_word_embeddings: false }),
  }),
  mamba: Object.freeze({
    kind: "ssm_recurrent",
    defaults: Object.freeze({ vocab_size: 50280, hidden_size: 768, state_size: 16, num_hidden_layers: 32, expand: 2, conv_kernel: 4, time_step_rank: "auto", use_bias: false, use_conv_bias: true, tie_word_embeddings: false }),
  }),
  jamba: Object.freeze({
    kind: "hybrid_attention_ssm_moe",
    defaults: Object.freeze({
      vocab_size: 65536, hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32,
      num_attention_heads: 32, num_key_value_heads: 8, max_position_embeddings: 262144,
      num_experts_per_tok: 2, num_experts: 16, expert_layer_period: 2, expert_layer_offset: 1,
      attn_layer_period: 8, attn_layer_offset: 4, mamba_d_state: 16, mamba_d_conv: 4,
      mamba_expand: 2, mamba_dt_rank: "auto", mamba_conv_bias: true, mamba_proj_bias: false,
      tie_word_embeddings: false,
    }),
  }),
});

const CACHE_DTYPE_BITS = Object.freeze({
  float64: 64, double: 64, float32: 32, float: 32, float16: 16, half: 16, bfloat16: 16,
});

const MAX_CANONICAL_DECODER_LAYERS = 4096;

function sourceEvidence() {
  return {
    ...HF_SAFETENSORS_ARCHITECTURE_SOURCE,
    configuration_sources: Object.fromEntries(Object.entries(HF_SAFETENSORS_ARCHITECTURE_SOURCE.configuration_sources).map(([key, value]) => [key, { ...value }])),
    modeling_sources: Object.fromEntries(Object.entries(HF_SAFETENSORS_ARCHITECTURE_SOURCE.modeling_sources).map(([key, value]) => [key, { ...value }])),
  };
}

function exactProduct(values) {
  return values.reduce((product, value) => product * BigInt(value), 1n);
}

function safeNumber(value) {
  return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function shapeEquals(actual, expected) {
  return Array.isArray(actual) && actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function shapeText(shape) {
  return Array.isArray(shape) ? shape.join("x") || "scalar" : "unknown";
}

function issue(code, severity, detail) {
  return Object.freeze({ code, severity, ...detail });
}

function canonicalTensorChecks(tensors, fields, layout) {
  const byName = new Map((tensors || []).map((tensor) => [tensor.name, tensor]));
  const checks = [];
  const expectedRows = [];
  const mismatches = [];
  const expectedNames = [];
  const missingNames = [];
  let expectedCanonicalParameterCount = 0n;
  let observedCanonicalParameterCount = 0n;
  const check = (name, expectedShape, role, required = true) => {
    const expectedElements = exactProduct(expectedShape);
    if (required) {
      expectedNames.push(name);
      expectedCanonicalParameterCount += expectedElements;
      expectedRows.push({ tensor_name: name, role, expected_shape: expectedShape, expected_parameter_count_decimal: String(expectedElements) });
    }
    const tensor = byName.get(name);
    if (!tensor) {
      if (required) missingNames.push(name);
      return false;
    }
    const observedElements = exactProduct(tensor.shape);
    observedCanonicalParameterCount += observedElements;
    const matched = shapeEquals(tensor.shape, expectedShape);
    const row = {
      tensor_name: name,
      role,
      required,
      expected_shape: expectedShape,
      observed_shape: tensor.shape,
      expected_parameter_count_decimal: String(expectedElements),
      observed_parameter_count_decimal: String(observedElements),
      status: matched ? "matched" : "mismatch",
    };
    checks.push(row);
    if (!matched) mismatches.push(row);
    return true;
  };

  check("model.embed_tokens.weight", [fields.vocab_size, fields.hidden_size], "token_embedding");
  check("model.norm.weight", [fields.hidden_size], "final_normalization");
  const lmHeadPresent = check("lm_head.weight", [fields.vocab_size, fields.hidden_size], "language_model_head", !fields.tie_word_embeddings);
  const layerIndices = new Set();
  for (const tensor of tensors || []) {
    const match = /^model\.layers\.(\d+)\./.exec(String(tensor.name || ""));
    if (match) layerIndices.add(Number(match[1]));
  }
  const expectedLayerIndices = Array.from({ length: fields.num_hidden_layers }, (_, index) => index);
  const missingLayerIndices = expectedLayerIndices.filter((index) => !layerIndices.has(index));
  const excessLayerIndices = [...layerIndices].filter((index) => index >= fields.num_hidden_layers).sort((a, b) => a - b);
  const queryWidth = fields.num_attention_heads * fields.head_dim;
  const kvWidth = fields.num_key_value_heads * fields.head_dim;
  const widths = {
    hidden: fields.hidden_size,
    head: fields.head_dim,
    query: queryWidth,
    kv: kvWidth,
    attention_heads: fields.num_attention_heads,
    kv_heads: fields.num_key_value_heads,
    intermediate: fields.intermediate_size,
  };
  const resolvedShape = (shape) => shape.map((dimension) => typeof dimension === "string" ? widths[dimension] : dimension);
  const attentionBiasScope = layout.attention_bias_scope === "qkv"
    ? "qkv"
    : layout.attention_bias_scope === "config_all" && fields.attention_bias ? "all" : "none";
  const qkvBias = attentionBiasScope === "qkv" || attentionBiasScope === "all";
  const outputBias = attentionBiasScope === "all";
  const checkLinear = (prefix, suffix, weightShape, outputWidth, role, bias) => {
    check(`${prefix}.${suffix}.weight`, weightShape, role);
    if (bias) check(`${prefix}.${suffix}.bias`, [outputWidth], `${role}_bias`);
  };
  for (const index of expectedLayerIndices) {
    const prefix = `model.layers.${index}`;
    if (layout.attention === "fused_qkv") {
      checkLinear(prefix, "self_attn.qkv_proj", [queryWidth + 2 * kvWidth, fields.hidden_size], queryWidth + 2 * kvWidth, "fused_query_key_value_projection", qkvBias);
    } else {
      checkLinear(prefix, "self_attn.q_proj", [queryWidth, fields.hidden_size], queryWidth, "query_projection", qkvBias);
      checkLinear(prefix, "self_attn.k_proj", [kvWidth, fields.hidden_size], kvWidth, "key_projection", qkvBias);
      checkLinear(prefix, "self_attn.v_proj", [kvWidth, fields.hidden_size], kvWidth, "value_projection", qkvBias);
    }
    checkLinear(prefix, "self_attn.o_proj", [fields.hidden_size, queryWidth], fields.hidden_size, "attention_output_projection", outputBias);
    if (layout.mlp === "fused_gate_up") {
      checkLinear(prefix, "mlp.gate_up_proj", [2 * fields.intermediate_size, fields.hidden_size], 2 * fields.intermediate_size, "fused_mlp_gate_up_projection", fields.mlp_bias);
    } else if (layout.mlp === "split_gated") {
      checkLinear(prefix, "mlp.gate_proj", [fields.intermediate_size, fields.hidden_size], fields.intermediate_size, "mlp_gate_projection", fields.mlp_bias);
      checkLinear(prefix, "mlp.up_proj", [fields.intermediate_size, fields.hidden_size], fields.intermediate_size, "mlp_up_projection", fields.mlp_bias);
    } else if (layout.mlp === "split_ungated") {
      checkLinear(prefix, "mlp.up_proj", [fields.intermediate_size, fields.hidden_size], fields.intermediate_size, "mlp_up_projection", fields.mlp_bias);
    } else {
      throw new Error(`Unsupported registered SafeTensors MLP layout ${layout.mlp}`);
    }
    checkLinear(prefix, "mlp.down_proj", [fields.hidden_size, fields.intermediate_size], fields.hidden_size, "mlp_down_projection", fields.mlp_bias);
    for (const norm of layout.norms) {
      const shape = norm.shape ? resolvedShape(norm.shape) : [widths[norm.width]];
      check(`${prefix}.${norm.suffix}.weight`, shape, `${norm.suffix}_normalization`);
    }
    if (fields.use_qk_norm) {
      check(`${prefix}.self_attn.q_norm.weight`, [fields.num_attention_heads, fields.head_dim], "query_head_normalization");
      check(`${prefix}.self_attn.k_norm.weight`, [fields.num_key_value_heads, fields.head_dim], "key_head_normalization");
    }
  }

  const embedding = byName.get("model.embed_tokens.weight");
  const lmHead = byName.get("lm_head.weight");
  const embeddingSha = embedding?.numerical_integrity?.payload_sha256 || null;
  const lmHeadSha = lmHead?.numerical_integrity?.payload_sha256 || null;
  const tiedWeightBinding = !fields.tie_word_embeddings
    ? "not_declared_tied"
    : embedding && !lmHeadPresent
      ? "single_serialized_embedding_candidate"
      : embeddingSha && lmHeadSha
        ? embeddingSha === lmHeadSha ? "content_identical_candidate" : "serialized_payloads_differ"
        : embedding && lmHead ? "payload_digest_unavailable" : "canonical_tensor_names_not_present";

  const canonicalNamingObserved = checks.length > 0 || layerIndices.size > 0;
  const observedTotalParameterCount = (tensors || []).reduce((sum, tensor) => sum + exactProduct(tensor.shape), 0n);
  return {
    status: mismatches.length || excessLayerIndices.length ? "invalid"
      : canonicalNamingObserved && !missingLayerIndices.length && !missingNames.length ? "assessed_canonical_layout"
        : canonicalNamingObserved ? "partial_canonical_layout" : "not_assessed_noncanonical_tensor_names",
    canonical_tensor_expected_count: expectedNames.length,
    canonical_tensor_check_count: checks.length,
    canonical_tensor_missing_count: missingNames.length,
    canonical_tensor_missing_names: missingNames.slice(0, 256),
    canonical_tensor_shape_match_count: checks.length - mismatches.length,
    canonical_tensor_shape_mismatch_count: mismatches.length,
    canonical_expected_parameter_count: safeNumber(expectedCanonicalParameterCount),
    canonical_expected_parameter_count_decimal: String(expectedCanonicalParameterCount),
    canonical_observed_parameter_count: safeNumber(observedCanonicalParameterCount),
    canonical_observed_parameter_count_decimal: String(observedCanonicalParameterCount),
    checkpoint_parameter_count: safeNumber(observedTotalParameterCount),
    checkpoint_parameter_count_decimal: String(observedTotalParameterCount),
    canonical_layer_index_count: layerIndices.size,
    missing_layer_index_count: missingLayerIndices.length,
    missing_layer_indices: missingLayerIndices.slice(0, 256),
    excess_layer_index_count: excessLayerIndices.length,
    excess_layer_indices: excessLayerIndices.slice(0, 256),
    check_rows: checks.slice(0, 512),
    expected_rows: expectedRows,
    mismatch_rows: mismatches.slice(0, 128),
    tied_weight_binding_status: tiedWeightBinding,
    tensor_layout_id: layout.id,
    attention_bias_scope: attentionBiasScope,
    tied_weight_interpretation: "Matching payload digests are a content-addressed candidate only; runtime Parameter aliasing is not established by SafeTensors.",
  };
}

function specializedTensorChecks(tensors, expectedRows, layoutId) {
  const byName = new Map((tensors || []).map((tensor) => [tensor.name, tensor]));
  const checks = [];
  const missing = [];
  const mismatches = [];
  let expectedParameters = 0n;
  let observedParameters = 0n;
  for (const row of expectedRows) {
    const expectedElements = exactProduct(row.shape);
    if (row.required !== false) expectedParameters += expectedElements;
    const tensor = byName.get(row.name);
    if (!tensor) {
      if (row.required !== false) missing.push(row.name);
      continue;
    }
    const observedElements = exactProduct(tensor.shape);
    if (row.required !== false) observedParameters += observedElements;
    const status = shapeEquals(tensor.shape, row.shape) ? "matched" : "mismatch";
    const check = {
      tensor_name: row.name,
      role: row.role,
      required: row.required !== false,
      expected_shape: row.shape,
      observed_shape: tensor.shape,
      expected_parameter_count_decimal: String(expectedElements),
      observed_parameter_count_decimal: String(observedElements),
      status,
    };
    checks.push(check);
    if (status === "mismatch") mismatches.push(check);
  }
  const canonicalNamesObserved = checks.length > 0;
  const checkpointParameters = (tensors || []).reduce((sum, tensor) => sum + exactProduct(tensor.shape), 0n);
  return {
    status: mismatches.length ? "invalid" : canonicalNamesObserved && !missing.length ? "assessed_canonical_layout"
      : canonicalNamesObserved ? "partial_canonical_layout" : "not_assessed_noncanonical_tensor_names",
    tensor_layout_id: layoutId,
    canonical_tensor_expected_count: expectedRows.filter((row) => row.required !== false).length,
    canonical_tensor_check_count: checks.length,
    canonical_tensor_missing_count: missing.length,
    canonical_tensor_missing_names: missing.slice(0, 256),
    canonical_tensor_shape_match_count: checks.length - mismatches.length,
    canonical_tensor_shape_mismatch_count: mismatches.length,
    canonical_expected_parameter_count: safeNumber(expectedParameters),
    canonical_expected_parameter_count_decimal: String(expectedParameters),
    canonical_observed_parameter_count: safeNumber(observedParameters),
    canonical_observed_parameter_count_decimal: String(observedParameters),
    checkpoint_parameter_count: safeNumber(checkpointParameters),
    checkpoint_parameter_count_decimal: String(checkpointParameters),
    expected_rows: expectedRows.filter((row) => row.required !== false).map((row) => ({ tensor_name: row.name, role: row.role, expected_shape: row.shape })),
    check_rows: checks.slice(0, 512),
    mismatch_rows: mismatches.slice(0, 128),
  };
}

function specializedConfigReader(config, defaults, issues, provenance) {
  return {
    integer(key, { required = true } = {}) {
      const declared = Object.hasOwn(config, key);
      const value = declared ? config[key] : defaults[key];
      provenance[key] = declared ? "declared_config" : "pinned_transformers_default";
      if (value == null && !required) return null;
      if (!Number.isSafeInteger(value) || value <= 0) {
        issues.push(issue("HF_CONFIG_POSITIVE_INTEGER_INVALID", "error", { field: key, value }));
        return null;
      }
      return value;
    },
    boolean(key, fallback = null) {
      const declared = Object.hasOwn(config, key);
      const value = declared ? config[key] : fallback;
      provenance[key] = declared ? "declared_config" : fallback == null ? "not_declared" : "pinned_transformers_default";
      if (value == null) return null;
      if (typeof value !== "boolean") {
        issues.push(issue("HF_CONFIG_BOOLEAN_INVALID", "error", { field: key, value }));
        return null;
      }
      return value;
    },
  };
}

function buildMixtralContract(config, tensors, registry) {
  const issues = [];
  const provenance = {};
  const read = specializedConfigReader(config, registry.defaults, issues, provenance);
  const fields = {
    vocab_size: read.integer("vocab_size"), hidden_size: read.integer("hidden_size"), intermediate_size: read.integer("intermediate_size"),
    num_hidden_layers: read.integer("num_hidden_layers"), num_attention_heads: read.integer("num_attention_heads"),
    num_key_value_heads: read.integer("num_key_value_heads"), max_position_embeddings: read.integer("max_position_embeddings"),
    head_dim: read.integer("head_dim", { required: false }), num_experts_per_tok: read.integer("num_experts_per_tok"),
    num_local_experts: read.integer("num_local_experts"), tie_word_embeddings: read.boolean("tie_word_embeddings", registry.defaults.tie_word_embeddings),
  };
  if (fields.head_dim == null && fields.hidden_size && fields.num_attention_heads && config.head_dim == null) {
    if (fields.hidden_size % fields.num_attention_heads === 0) {
      fields.head_dim = fields.hidden_size / fields.num_attention_heads;
      provenance.head_dim = "pinned_modeling_hidden_size_integer_div_num_attention_heads";
    } else issues.push(issue("HF_CONFIG_HEAD_DIM_DIVISION_HAS_REMAINDER", "error", { hidden_size: fields.hidden_size, num_attention_heads: fields.num_attention_heads }));
  }
  if (fields.num_experts_per_tok && fields.num_local_experts && fields.num_experts_per_tok > fields.num_local_experts) {
    issues.push(issue("HF_MOE_ACTIVE_EXPERT_COUNT_EXCEEDS_TOTAL", "error", { active: fields.num_experts_per_tok, total: fields.num_local_experts }));
  }
  if (fields.num_attention_heads && fields.num_key_value_heads && fields.num_attention_heads % fields.num_key_value_heads !== 0) {
    issues.push(issue("HF_CONFIG_GQA_GROUP_NONINTEGRAL", "error", { num_attention_heads: fields.num_attention_heads, num_key_value_heads: fields.num_key_value_heads }));
  }
  if (fields.num_hidden_layers && fields.num_local_experts && fields.num_hidden_layers * fields.num_local_experts * 3 > 32768) {
    issues.push(issue("HF_MOE_CANONICAL_TENSOR_CHECK_LIMIT_EXCEEDED", "error", { layers: fields.num_hidden_layers, experts: fields.num_local_experts, maximum_expert_matrix_checks: 32768 }));
  }
  if (Object.entries(fields).some(([key, value]) => key !== "tie_word_embeddings" && value == null) || issues.some((row) => row.severity === "error")) {
    return specializedFailure("mixtral", fields, provenance, issues);
  }
  const queryWidth = fields.num_attention_heads * fields.head_dim;
  const kvWidth = fields.num_key_value_heads * fields.head_dim;
  const rows = [
    { name: "model.embed_tokens.weight", shape: [fields.vocab_size, fields.hidden_size], role: "token_embedding" },
    { name: "model.norm.weight", shape: [fields.hidden_size], role: "final_normalization" },
    { name: "lm_head.weight", shape: [fields.vocab_size, fields.hidden_size], role: "language_model_head", required: !fields.tie_word_embeddings },
  ];
  for (let layer = 0; layer < fields.num_hidden_layers; layer += 1) {
    const prefix = `model.layers.${layer}`;
    rows.push(
      { name: `${prefix}.input_layernorm.weight`, shape: [fields.hidden_size], role: "input_normalization" },
      { name: `${prefix}.post_attention_layernorm.weight`, shape: [fields.hidden_size], role: "post_attention_normalization" },
      { name: `${prefix}.self_attn.q_proj.weight`, shape: [queryWidth, fields.hidden_size], role: "query_projection" },
      { name: `${prefix}.self_attn.k_proj.weight`, shape: [kvWidth, fields.hidden_size], role: "key_projection" },
      { name: `${prefix}.self_attn.v_proj.weight`, shape: [kvWidth, fields.hidden_size], role: "value_projection" },
      { name: `${prefix}.self_attn.o_proj.weight`, shape: [fields.hidden_size, queryWidth], role: "attention_output_projection" },
      { name: `${prefix}.block_sparse_moe.gate.weight`, shape: [fields.num_local_experts, fields.hidden_size], role: "expert_router" },
    );
    for (let expert = 0; expert < fields.num_local_experts; expert += 1) rows.push(
      { name: `${prefix}.block_sparse_moe.experts.${expert}.w1.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "expert_gate_projection" },
      { name: `${prefix}.block_sparse_moe.experts.${expert}.w2.weight`, shape: [fields.hidden_size, fields.intermediate_size], role: "expert_down_projection" },
      { name: `${prefix}.block_sparse_moe.experts.${expert}.w3.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "expert_up_projection" },
    );
  }
  const tensorContract = specializedTensorChecks(tensors, rows, "mixtral_sparse_topk_experts");
  if (tensorContract.canonical_tensor_shape_mismatch_count) issues.push(issue("HF_CANONICAL_TENSOR_SHAPE_MISMATCH", "error", { mismatch_count: tensorContract.canonical_tensor_shape_mismatch_count }));
  const kvStateProjection = buildKvStateProjection({ layerCount: fields.num_hidden_layers, kvHeadCount: fields.num_key_value_heads, keyHeadWidth: fields.head_dim, valueHeadWidth: fields.head_dim, contextLength: fields.max_position_embeddings });
  return specializedSuccess("mixtral", "sparse_moe_decoder", fields, provenance, issues, tensorContract, {
    kv_state_projection: kvStateProjection,
    compute_projection: buildSparseMoeDecoderProjection({ hiddenSize: fields.hidden_size, intermediateSize: fields.intermediate_size, layerCount: fields.num_hidden_layers, attentionHeadCount: fields.num_attention_heads, kvHeadCount: fields.num_key_value_heads, headWidth: fields.head_dim, contextLength: fields.max_position_embeddings, expertCount: fields.num_local_experts, activeExpertCount: fields.num_experts_per_tok }),
    gqa_query_heads_per_kv_head: fields.num_attention_heads / fields.num_key_value_heads,
    moe_projection: { expert_count: fields.num_local_experts, active_expert_count_per_token: fields.num_experts_per_tok, router_evidence: "SOURCE_PINNED_CONFIG_AND_MODELING" },
  });
}

function buildMambaContract(config, tensors, registry) {
  const issues = [];
  const provenance = {};
  const read = specializedConfigReader(config, registry.defaults, issues, provenance);
  const hidden = read.integer("hidden_size");
  const expandDeclared = Object.hasOwn(config, "expand");
  const expand = expandDeclared ? config.expand : registry.defaults.expand;
  provenance.expand = expandDeclared ? "declared_config" : "pinned_transformers_default";
  if (!Number.isFinite(expand) || expand <= 0 || !Number.isSafeInteger(expand * (hidden || 0))) issues.push(issue("HF_MAMBA_EXPANSION_INVALID", "error", { expand, hidden_size: hidden }));
  const rankRaw = Object.hasOwn(config, "time_step_rank") ? config.time_step_rank : registry.defaults.time_step_rank;
  const timeStepRank = rankRaw === "auto" && hidden ? Math.ceil(hidden / 16) : rankRaw;
  provenance.time_step_rank = Object.hasOwn(config, "time_step_rank") ? "declared_config" : "pinned_transformers_default_auto_ceil_hidden_div_16";
  if (!Number.isSafeInteger(timeStepRank) || timeStepRank <= 0) issues.push(issue("HF_MAMBA_TIME_STEP_RANK_INVALID", "error", { value: rankRaw, resolved: timeStepRank }));
  const fields = {
    vocab_size: read.integer("vocab_size"), hidden_size: hidden, intermediate_size: hidden && Number.isFinite(expand) ? expand * hidden : null,
    state_size: read.integer("state_size"), num_hidden_layers: read.integer("num_hidden_layers"), conv_kernel: read.integer("conv_kernel"),
    expand, time_step_rank: Number.isSafeInteger(timeStepRank) ? timeStepRank : null,
    use_bias: read.boolean("use_bias", registry.defaults.use_bias), use_conv_bias: read.boolean("use_conv_bias", registry.defaults.use_conv_bias),
    tie_word_embeddings: read.boolean("tie_word_embeddings", null),
  };
  if (Object.entries(fields).some(([key, value]) => !["tie_word_embeddings"].includes(key) && value == null) || issues.some((row) => row.severity === "error")) {
    return specializedFailure("mamba", fields, provenance, issues);
  }
  const rows = [
    { name: "backbone.embeddings.weight", shape: [fields.vocab_size, fields.hidden_size], role: "token_embedding" },
    { name: "backbone.norm_f.weight", shape: [fields.hidden_size], role: "final_normalization" },
    { name: "lm_head.weight", shape: [fields.vocab_size, fields.hidden_size], role: "language_model_head", required: fields.tie_word_embeddings === false },
  ];
  for (let layer = 0; layer < fields.num_hidden_layers; layer += 1) {
    const prefix = `backbone.layers.${layer}`;
    rows.push(
      { name: `${prefix}.norm.weight`, shape: [fields.hidden_size], role: "input_normalization" },
      { name: `${prefix}.mixer.in_proj.weight`, shape: [2 * fields.intermediate_size, fields.hidden_size], role: "mixer_input_projection" },
      { name: `${prefix}.mixer.conv1d.weight`, shape: [fields.intermediate_size, 1, fields.conv_kernel], role: "depthwise_convolution" },
      { name: `${prefix}.mixer.x_proj.weight`, shape: [fields.time_step_rank + 2 * fields.state_size, fields.intermediate_size], role: "state_parameter_projection" },
      { name: `${prefix}.mixer.dt_proj.weight`, shape: [fields.intermediate_size, fields.time_step_rank], role: "time_step_projection" },
      { name: `${prefix}.mixer.dt_proj.bias`, shape: [fields.intermediate_size], role: "time_step_bias" },
      { name: `${prefix}.mixer.A_log`, shape: [fields.intermediate_size, fields.state_size], role: "state_transition_log" },
      { name: `${prefix}.mixer.D`, shape: [fields.intermediate_size], role: "skip_parameter" },
      { name: `${prefix}.mixer.out_proj.weight`, shape: [fields.hidden_size, fields.intermediate_size], role: "mixer_output_projection" },
    );
    if (fields.use_conv_bias) rows.push({ name: `${prefix}.mixer.conv1d.bias`, shape: [fields.intermediate_size], role: "convolution_bias" });
    if (fields.use_bias) rows.push(
      { name: `${prefix}.mixer.in_proj.bias`, shape: [2 * fields.intermediate_size], role: "mixer_input_bias" },
      { name: `${prefix}.mixer.out_proj.bias`, shape: [fields.hidden_size], role: "mixer_output_bias" },
    );
  }
  const tensorContract = specializedTensorChecks(tensors, rows, "mamba_recurrent_ssm");
  if (tensorContract.canonical_tensor_shape_mismatch_count) issues.push(issue("HF_CANONICAL_TENSOR_SHAPE_MISMATCH", "error", { mismatch_count: tensorContract.canonical_tensor_shape_mismatch_count }));
  return specializedSuccess("mamba", "ssm_recurrent", fields, provenance, issues, tensorContract, {
    recurrent_state_projection: buildSsmRecurrentStateProjection({ layerCount: fields.num_hidden_layers, intermediateSize: fields.intermediate_size, stateSize: fields.state_size, convolutionKernel: fields.conv_kernel }),
    compute_projection: buildSsmLinearComputeProjection({ hiddenSize: fields.hidden_size, intermediateSize: fields.intermediate_size, layerCount: fields.num_hidden_layers, stateSize: fields.state_size, timeStepRank: fields.time_step_rank, convolutionKernel: fields.conv_kernel }),
  });
}

function buildJambaContract(config, tensors, registry) {
  const issues = [];
  const provenance = {};
  const read = specializedConfigReader(config, registry.defaults, issues, provenance);
  const hidden = read.integer("hidden_size");
  const offset = (key) => {
    const declared = Object.hasOwn(config, key);
    const value = declared ? config[key] : registry.defaults[key];
    provenance[key] = declared ? "declared_config" : "pinned_transformers_default";
    if (!Number.isSafeInteger(value) || value < 0) {
      issues.push(issue("HF_CONFIG_NONNEGATIVE_INTEGER_INVALID", "error", { field: key, value }));
      return null;
    }
    return value;
  };
  const mambaExpand = Object.hasOwn(config, "mamba_expand") ? config.mamba_expand : registry.defaults.mamba_expand;
  provenance.mamba_expand = Object.hasOwn(config, "mamba_expand") ? "declared_config" : "pinned_transformers_default";
  if (!Number.isFinite(mambaExpand) || mambaExpand <= 0 || !Number.isSafeInteger(mambaExpand * (hidden || 0))) {
    issues.push(issue("HF_JAMBA_MAMBA_EXPANSION_INVALID", "error", { mamba_expand: mambaExpand, hidden_size: hidden }));
  }
  const rankRaw = Object.hasOwn(config, "mamba_dt_rank") ? config.mamba_dt_rank : registry.defaults.mamba_dt_rank;
  const mambaDtRank = rankRaw === "auto" && hidden ? Math.ceil(hidden / 16) : rankRaw;
  provenance.mamba_dt_rank = Object.hasOwn(config, "mamba_dt_rank") ? "declared_config" : "pinned_transformers_default_auto_ceil_hidden_div_16";
  if (!Number.isSafeInteger(mambaDtRank) || mambaDtRank <= 0) {
    issues.push(issue("HF_JAMBA_MAMBA_DT_RANK_INVALID", "error", { value: rankRaw, resolved: mambaDtRank }));
  }
  const fields = {
    vocab_size: read.integer("vocab_size"), hidden_size: hidden, intermediate_size: read.integer("intermediate_size"),
    num_hidden_layers: read.integer("num_hidden_layers"), num_attention_heads: read.integer("num_attention_heads"),
    num_key_value_heads: read.integer("num_key_value_heads"), max_position_embeddings: read.integer("max_position_embeddings"),
    num_experts_per_tok: read.integer("num_experts_per_tok"), num_experts: read.integer("num_experts"),
    expert_layer_period: read.integer("expert_layer_period"), expert_layer_offset: offset("expert_layer_offset"),
    attn_layer_period: read.integer("attn_layer_period"), attn_layer_offset: offset("attn_layer_offset"),
    mamba_d_state: read.integer("mamba_d_state"), mamba_d_conv: read.integer("mamba_d_conv"),
    mamba_expand: mambaExpand, mamba_intermediate_size: Number.isFinite(mambaExpand) ? mambaExpand * (hidden || 0) : null,
    mamba_dt_rank: Number.isSafeInteger(mambaDtRank) ? mambaDtRank : null,
    mamba_conv_bias: read.boolean("mamba_conv_bias", registry.defaults.mamba_conv_bias),
    mamba_proj_bias: read.boolean("mamba_proj_bias", registry.defaults.mamba_proj_bias),
    tie_word_embeddings: read.boolean("tie_word_embeddings", registry.defaults.tie_word_embeddings),
  };
  if (fields.num_attention_heads && hidden && hidden % fields.num_attention_heads !== 0) {
    issues.push(issue("HF_CONFIG_HEAD_DIM_DIVISION_HAS_REMAINDER", "error", { hidden_size: hidden, num_attention_heads: fields.num_attention_heads }));
  }
  fields.head_dim = fields.num_attention_heads && hidden ? hidden / fields.num_attention_heads : null;
  provenance.head_dim = "pinned_modeling_hidden_size_integer_div_num_attention_heads";
  if (fields.num_attention_heads && fields.num_key_value_heads && fields.num_attention_heads % fields.num_key_value_heads !== 0) {
    issues.push(issue("HF_CONFIG_GQA_GROUP_NONINTEGRAL", "error", { num_attention_heads: fields.num_attention_heads, num_key_value_heads: fields.num_key_value_heads }));
  }
  if (fields.num_experts_per_tok && fields.num_experts && fields.num_experts_per_tok > fields.num_experts) {
    issues.push(issue("HF_MOE_ACTIVE_EXPERT_COUNT_EXCEEDS_TOTAL", "error", { active: fields.num_experts_per_tok, total: fields.num_experts }));
  }
  for (const [kind, period, layerOffset] of [
    ["attention", fields.attn_layer_period, fields.attn_layer_offset],
    ["expert", fields.expert_layer_period, fields.expert_layer_offset],
  ]) if (period != null && layerOffset != null && layerOffset >= period) {
    issues.push(issue("HF_JAMBA_LAYER_OFFSET_NOT_BELOW_PERIOD", "error", { kind, period, offset: layerOffset }));
  }
  if (fields.num_hidden_layers && fields.num_experts && fields.num_hidden_layers * fields.num_experts * 3 > 32768) {
    issues.push(issue("HF_MOE_CANONICAL_TENSOR_CHECK_LIMIT_EXCEEDED", "error", { layers: fields.num_hidden_layers, experts: fields.num_experts, maximum_expert_matrix_checks: 32768 }));
  }
  if (Object.entries(fields).some(([key, value]) => key !== "tie_word_embeddings" && value == null)
    || issues.some((row) => row.severity === "error")) {
    return specializedFailure("jamba", fields, provenance, issues);
  }

  const attentionLayers = [];
  const mambaLayers = [];
  const expertLayers = [];
  const denseLayers = [];
  for (let layer = 0; layer < fields.num_hidden_layers; layer += 1) {
    (layer % fields.attn_layer_period === fields.attn_layer_offset ? attentionLayers : mambaLayers).push(layer);
    (layer % fields.expert_layer_period === fields.expert_layer_offset ? expertLayers : denseLayers).push(layer);
  }
  if (!attentionLayers.length || !mambaLayers.length) {
    issues.push(issue("HF_JAMBA_HYBRID_LAYER_SET_EMPTY", "error", { attention_layer_count: attentionLayers.length, mamba_layer_count: mambaLayers.length }));
    return specializedFailure("jamba", fields, provenance, issues);
  }
  Object.assign(fields, {
    attention_layer_count: attentionLayers.length, mamba_layer_count: mambaLayers.length,
    expert_feed_forward_layer_count: expertLayers.length, dense_feed_forward_layer_count: denseLayers.length,
    state_size: fields.mamba_d_state, conv_kernel: fields.mamba_d_conv,
    expand: fields.mamba_expand, time_step_rank: fields.mamba_dt_rank,
  });
  const queryWidth = fields.num_attention_heads * fields.head_dim;
  const kvWidth = fields.num_key_value_heads * fields.head_dim;
  const rows = [
    { name: "model.embed_tokens.weight", shape: [fields.vocab_size, fields.hidden_size], role: "token_embedding" },
    { name: "model.final_layernorm.weight", shape: [fields.hidden_size], role: "final_normalization" },
    { name: "lm_head.weight", shape: [fields.vocab_size, fields.hidden_size], role: "language_model_head", required: !fields.tie_word_embeddings },
  ];
  for (let layer = 0; layer < fields.num_hidden_layers; layer += 1) {
    const prefix = `model.layers.${layer}`;
    rows.push(
      { name: `${prefix}.input_layernorm.weight`, shape: [fields.hidden_size], role: "input_normalization" },
      { name: `${prefix}.pre_ff_layernorm.weight`, shape: [fields.hidden_size], role: "pre_feed_forward_normalization" },
    );
    if (attentionLayers.includes(layer)) rows.push(
      { name: `${prefix}.self_attn.q_proj.weight`, shape: [queryWidth, fields.hidden_size], role: "query_projection" },
      { name: `${prefix}.self_attn.k_proj.weight`, shape: [kvWidth, fields.hidden_size], role: "key_projection" },
      { name: `${prefix}.self_attn.v_proj.weight`, shape: [kvWidth, fields.hidden_size], role: "value_projection" },
      { name: `${prefix}.self_attn.o_proj.weight`, shape: [fields.hidden_size, queryWidth], role: "attention_output_projection" },
    );
    else {
      const mamba = `${prefix}.mamba`;
      rows.push(
        { name: `${mamba}.in_proj.weight`, shape: [2 * fields.mamba_intermediate_size, fields.hidden_size], role: "mamba_input_projection" },
        { name: `${mamba}.conv1d.weight`, shape: [fields.mamba_intermediate_size, 1, fields.mamba_d_conv], role: "mamba_depthwise_convolution" },
        { name: `${mamba}.x_proj.weight`, shape: [fields.mamba_dt_rank + 2 * fields.mamba_d_state, fields.mamba_intermediate_size], role: "mamba_state_parameter_projection" },
        { name: `${mamba}.dt_proj.weight`, shape: [fields.mamba_intermediate_size, fields.mamba_dt_rank], role: "mamba_time_step_projection" },
        { name: `${mamba}.dt_proj.bias`, shape: [fields.mamba_intermediate_size], role: "mamba_time_step_bias" },
        { name: `${mamba}.A_log`, shape: [fields.mamba_intermediate_size, fields.mamba_d_state], role: "mamba_state_transition_log" },
        { name: `${mamba}.D`, shape: [fields.mamba_intermediate_size], role: "mamba_skip_parameter" },
        { name: `${mamba}.out_proj.weight`, shape: [fields.hidden_size, fields.mamba_intermediate_size], role: "mamba_output_projection" },
        { name: `${mamba}.dt_layernorm.weight`, shape: [fields.mamba_dt_rank], role: "mamba_time_step_normalization" },
        { name: `${mamba}.b_layernorm.weight`, shape: [fields.mamba_d_state], role: "mamba_b_normalization" },
        { name: `${mamba}.c_layernorm.weight`, shape: [fields.mamba_d_state], role: "mamba_c_normalization" },
      );
      if (fields.mamba_conv_bias) rows.push({ name: `${mamba}.conv1d.bias`, shape: [fields.mamba_intermediate_size], role: "mamba_convolution_bias" });
      if (fields.mamba_proj_bias) rows.push(
        { name: `${mamba}.in_proj.bias`, shape: [2 * fields.mamba_intermediate_size], role: "mamba_input_bias" },
        { name: `${mamba}.out_proj.bias`, shape: [fields.hidden_size], role: "mamba_output_bias" },
      );
    }
    const feedForward = `${prefix}.feed_forward`;
    if (expertLayers.includes(layer)) {
      rows.push({ name: `${feedForward}.router.weight`, shape: [fields.num_experts, fields.hidden_size], role: "expert_router" });
      for (let expert = 0; expert < fields.num_experts; expert += 1) rows.push(
        { name: `${feedForward}.experts.${expert}.gate_proj.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "expert_gate_projection" },
        { name: `${feedForward}.experts.${expert}.up_proj.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "expert_up_projection" },
        { name: `${feedForward}.experts.${expert}.down_proj.weight`, shape: [fields.hidden_size, fields.intermediate_size], role: "expert_down_projection" },
      );
    } else rows.push(
      { name: `${feedForward}.gate_proj.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "dense_gate_projection" },
      { name: `${feedForward}.up_proj.weight`, shape: [fields.intermediate_size, fields.hidden_size], role: "dense_up_projection" },
      { name: `${feedForward}.down_proj.weight`, shape: [fields.hidden_size, fields.intermediate_size], role: "dense_down_projection" },
    );
  }
  const tensorContract = specializedTensorChecks(tensors, rows, "jamba_hybrid_attention_ssm_moe");
  if (tensorContract.canonical_tensor_shape_mismatch_count) issues.push(issue("HF_CANONICAL_TENSOR_SHAPE_MISMATCH", "error", { mismatch_count: tensorContract.canonical_tensor_shape_mismatch_count }));
  const kv = buildKvStateProjection({ layerCount: attentionLayers.length, kvHeadCount: fields.num_key_value_heads, keyHeadWidth: fields.head_dim, valueHeadWidth: fields.head_dim, contextLength: fields.max_position_embeddings });
  const recurrent = buildSsmRecurrentStateProjection({ layerCount: mambaLayers.length, intermediateSize: fields.mamba_intermediate_size, stateSize: fields.mamba_d_state, convolutionKernel: fields.mamba_d_conv });
  return specializedSuccess("jamba", "hybrid_attention_ssm_moe", fields, provenance, issues, tensorContract, {
    kv_state_projection: kv,
    recurrent_state_projection: recurrent,
    compute_projection: buildHybridJambaComputeProjection({
      vocabularySize: fields.vocab_size, hiddenSize: fields.hidden_size, intermediateSize: fields.intermediate_size,
      attentionLayerCount: attentionLayers.length, mambaLayerCount: mambaLayers.length,
      denseFeedForwardLayerCount: denseLayers.length, expertFeedForwardLayerCount: expertLayers.length,
      attentionHeadCount: fields.num_attention_heads, kvHeadCount: fields.num_key_value_heads,
      headWidth: fields.head_dim, contextLength: fields.max_position_embeddings,
      expertCount: fields.num_experts, activeExpertCount: fields.num_experts_per_tok,
      stateSize: fields.mamba_d_state, timeStepRank: fields.mamba_dt_rank,
      convolutionKernel: fields.mamba_d_conv, mambaIntermediateSize: fields.mamba_intermediate_size,
    }),
    gqa_query_heads_per_kv_head: fields.num_attention_heads / fields.num_key_value_heads,
    moe_projection: {
      expert_count: fields.num_experts, active_expert_count_per_token: fields.num_experts_per_tok,
      expert_feed_forward_layer_count: expertLayers.length, dense_feed_forward_layer_count: denseLayers.length,
      router_evidence: "SOURCE_PINNED_CONFIG_AND_MODELING",
    },
    layer_schedule: {
      attention_layer_indices: attentionLayers, mamba_layer_indices: mambaLayers,
      expert_feed_forward_layer_indices: expertLayers, dense_feed_forward_layer_indices: denseLayers,
    },
  });
}

function specializedFailure(modelType, fields, provenance, issues) {
  return {
    schema: "deepbom.hf_safetensors_architecture_contract.v1",
    status: "invalid_config", evidence_class: "OBSERVED/SOURCE_PINNED", model_type: modelType,
    fields, field_provenance: provenance, issue_count: issues.length, error_count: issues.filter((row) => row.severity === "error").length,
    issues, source: sourceEvidence(), boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
  };
}

function specializedSuccess(modelType, architectureKind, fields, provenance, issues, tensorContract, extra) {
  const errorCount = issues.filter((row) => row.severity === "error").length;
  return {
    schema: "deepbom.hf_safetensors_architecture_contract.v1",
    status: errorCount ? "invalid" : tensorContract.status.startsWith("assessed") ? "assessed" : "partial",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", model_type: modelType, architecture_kind: architectureKind,
    architectures: [], fields, field_provenance: provenance, tensor_layout_id: tensorContract.tensor_layout_id,
    tensor_contract: tensorContract, issue_count: issues.length, error_count: errorCount, issues, ...extra,
    source: sourceEvidence(), boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
  };
}

export function buildHfSafeTensorsContract(config, tensors) {
  if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("Hugging Face config.json must be an object");
  const modelType = typeof config.model_type === "string" ? config.model_type : null;
  const registry = modelType ? ARCHITECTURES[modelType] : null;
  if (!registry) {
    return {
      schema: "deepbom.hf_safetensors_architecture_contract.v1",
      status: "not_assessed_unregistered_architecture",
      evidence_class: "OBSERVED",
      model_type: modelType,
      registered_model_types: Object.keys(ARCHITECTURES),
      source: sourceEvidence(),
      reason: modelType ? `model_type ${modelType} is not registered for deterministic architecture reconstruction` : "config.json does not declare model_type",
      boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
    };
  }

  if (registry.kind === "sparse_moe_decoder") return buildMixtralContract(config, tensors, registry);
  if (registry.kind === "ssm_recurrent") return buildMambaContract(config, tensors, registry);
  if (registry.kind === "hybrid_attention_ssm_moe") return buildJambaContract(config, tensors, registry);

  const issues = [];
  const provenance = {};
  const integer = (key) => {
    let value;
    if (!Object.hasOwn(config, key)) {
      value = registry.defaults[key];
      provenance[key] = "pinned_transformers_default";
    } else {
      value = config[key];
      provenance[key] = "declared_config";
    }
    if (!Number.isSafeInteger(value) || value <= 0) {
      issues.push(issue("HF_CONFIG_POSITIVE_INTEGER_INVALID", "error", { field: key, value }));
      return null;
    }
    return value;
  };
  const fields = {
    vocab_size: integer("vocab_size"),
    hidden_size: integer("hidden_size"),
    intermediate_size: integer("intermediate_size"),
    num_hidden_layers: integer("num_hidden_layers"),
    num_attention_heads: integer("num_attention_heads"),
    num_key_value_heads: integer("num_key_value_heads"),
    max_position_embeddings: integer("max_position_embeddings"),
    head_dim: integer("head_dim"),
    tie_word_embeddings: typeof config.tie_word_embeddings === "boolean" ? config.tie_word_embeddings : registry.defaults.tie_word_embeddings,
    attention_bias: registry.defaults.attention_bias === true,
    mlp_bias: false,
    use_qk_norm: false,
  };
  for (const key of Object.keys(registry.tensor_layout.config_flags || {})) {
    const value = Object.hasOwn(config, key) ? config[key] : registry.defaults[key];
    if (typeof value !== "boolean") {
      issues.push(issue("HF_CONFIG_BOOLEAN_INVALID", "error", { field: key, value }));
    } else {
      fields[key] = value;
      provenance[key] = Object.hasOwn(config, key) ? "declared_config" : "pinned_transformers_default";
    }
  }
  if (fields.num_hidden_layers != null && fields.num_hidden_layers > MAX_CANONICAL_DECODER_LAYERS) {
    issues.push(issue("HF_CONFIG_LAYER_COUNT_EXCEEDS_ANALYSIS_LIMIT", "error", {
      field: "num_hidden_layers",
      value: fields.num_hidden_layers,
      maximum: MAX_CANONICAL_DECODER_LAYERS,
    }));
  }
  provenance.tie_word_embeddings = typeof config.tie_word_embeddings === "boolean" ? "declared_config" : "pinned_transformers_default";
  if (registry.tensor_layout.attention_bias_scope === "qkv") {
    provenance.attention_bias = "pinned_transformers_fixed_qkv_projection_bias";
  }
  if (fields.num_key_value_heads == null && fields.num_attention_heads != null && config.num_key_value_heads == null) {
    fields.num_key_value_heads = fields.num_attention_heads;
    provenance.num_key_value_heads = "pinned_transformers_num_attention_heads_fallback";
    const at = issues.findIndex((row) => row.field === "num_key_value_heads");
    if (at >= 0) issues.splice(at, 1);
  }
  if (fields.head_dim == null && fields.hidden_size != null && fields.num_attention_heads != null && config.head_dim == null) {
    fields.head_dim = Math.floor(fields.hidden_size / fields.num_attention_heads);
    provenance.head_dim = "pinned_transformers_hidden_size_integer_div_num_attention_heads";
    const at = issues.findIndex((row) => row.field === "head_dim");
    if (at >= 0) issues.splice(at, 1);
    if (fields.hidden_size % fields.num_attention_heads !== 0) {
      issues.push(issue("HF_CONFIG_HEAD_DIM_DIVISION_HAS_REMAINDER", "error", { hidden_size: fields.hidden_size, num_attention_heads: fields.num_attention_heads, derived_head_dim: fields.head_dim }));
    }
  }
  if (Object.values(fields).some((value) => value == null)) {
    return {
      schema: "deepbom.hf_safetensors_architecture_contract.v1",
      status: "invalid_config",
      evidence_class: "OBSERVED/SOURCE_PINNED",
      model_type: modelType,
      fields,
      field_provenance: provenance,
      issue_count: issues.length,
      issues,
      source: sourceEvidence(),
      boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
    };
  }
  if (fields.num_attention_heads % fields.num_key_value_heads !== 0) {
    issues.push(issue("HF_CONFIG_GQA_GROUP_NONINTEGRAL", "error", { num_attention_heads: fields.num_attention_heads, num_key_value_heads: fields.num_key_value_heads }));
  }
  const layerTypes = config.layer_types;
  if (layerTypes != null && (!Array.isArray(layerTypes) || layerTypes.length !== fields.num_hidden_layers || layerTypes.some((value) => typeof value !== "string"))) {
    issues.push(issue("HF_CONFIG_LAYER_TYPES_CARDINALITY_INVALID", "error", { expected: fields.num_hidden_layers, observed: Array.isArray(layerTypes) ? layerTypes.length : null }));
  }
  for (const key of ["bos_token_id", "eos_token_id", "pad_token_id"]) {
    const raw = config[key];
    const values = Array.isArray(raw) ? raw : raw == null ? [] : [raw];
    for (const value of values) if (!Number.isSafeInteger(value) || value < 0 || value >= fields.vocab_size) {
      issues.push(issue("HF_CONFIG_SPECIAL_TOKEN_ID_OUT_OF_RANGE", "error", { field: key, value, vocab_size: fields.vocab_size }));
    }
  }
  const queryProjectionWidth = fields.num_attention_heads * fields.head_dim;
  const kvProjectionWidth = fields.num_key_value_heads * fields.head_dim;
  const fusedQkvProjectionWidth = queryProjectionWidth + 2 * kvProjectionWidth;
  const fusedGateUpProjectionWidth = 2 * fields.intermediate_size;
  if (!Number.isSafeInteger(queryProjectionWidth) || !Number.isSafeInteger(kvProjectionWidth)
    || (registry.tensor_layout.attention === "fused_qkv" && !Number.isSafeInteger(fusedQkvProjectionWidth))
    || (registry.tensor_layout.mlp === "fused_gate_up" && !Number.isSafeInteger(fusedGateUpProjectionWidth))) {
    issues.push(issue("HF_CONFIG_DERIVED_PROJECTION_WIDTH_UNSAFE", "error", {
      num_attention_heads: fields.num_attention_heads,
      num_key_value_heads: fields.num_key_value_heads,
      head_dim: fields.head_dim,
      intermediate_size: fields.intermediate_size,
      tensor_layout_id: registry.tensor_layout.id,
    }));
  }
  if (issues.some((row) => row.severity === "error")) {
    return {
      schema: "deepbom.hf_safetensors_architecture_contract.v1",
      status: "invalid_config",
      evidence_class: "OBSERVED/SOURCE_PINNED",
      model_type: modelType,
      fields,
      field_provenance: provenance,
      issue_count: issues.length,
      error_count: issues.filter((row) => row.severity === "error").length,
      issues,
      source: sourceEvidence(),
      boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
    };
  }

  const declaredTorchDtype = typeof config.torch_dtype === "string" ? config.torch_dtype.toLowerCase() : null;
  const cacheBits = declaredTorchDtype ? CACHE_DTYPE_BITS[declaredTorchDtype] || null : null;
  const kvStateProjection = buildKvStateProjection({
    layerCount: fields.num_hidden_layers,
    kvHeadCount: fields.num_key_value_heads,
    keyHeadWidth: fields.head_dim,
    valueHeadWidth: fields.head_dim,
    contextLength: fields.max_position_embeddings,
    storageBits: cacheBits,
  });
  const projectionInputs = {
    vocabularySize: fields.vocab_size,
    hiddenSize: fields.hidden_size,
    intermediateSize: fields.intermediate_size,
    layerCount: fields.num_hidden_layers,
    attentionHeadCount: fields.num_attention_heads,
    kvHeadCount: fields.num_key_value_heads,
    headWidth: fields.head_dim,
    contextLength: fields.max_position_embeddings,
  };
  const mlpProjectionMatrixCount = registry.tensor_layout.mlp_projection_matrix_count;
  const computeProjection = mlpProjectionMatrixCount === 3
    ? buildCanonicalGatedDecoderProjection(projectionInputs)
    : buildCanonicalDecoderProjection({ ...projectionInputs, mlpProjectionMatrixCount });
  const tensorContract = canonicalTensorChecks(tensors, fields, registry.tensor_layout);
  if (tensorContract.canonical_tensor_shape_mismatch_count) {
    issues.push(issue("HF_CANONICAL_TENSOR_SHAPE_MISMATCH", "error", { mismatch_count: tensorContract.canonical_tensor_shape_mismatch_count }));
  }
  if (tensorContract.excess_layer_index_count) {
    issues.push(issue("HF_CANONICAL_LAYER_INDEX_OUT_OF_RANGE", "error", { excess_layer_indices: tensorContract.excess_layer_indices }));
  }
  const errorCount = issues.filter((row) => row.severity === "error").length;
  return {
    schema: "deepbom.hf_safetensors_architecture_contract.v1",
    status: errorCount ? "invalid" : tensorContract.status.startsWith("assessed") ? "assessed" : "partial",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED",
    model_type: modelType,
    tensor_layout_id: registry.tensor_layout.id,
    attention_bias_scope: registry.tensor_layout.attention_bias_scope === "qkv"
      ? "qkv"
      : fields.attention_bias ? "all" : "none",
    mlp_projection_matrix_count: mlpProjectionMatrixCount,
    architectures: Array.isArray(config.architectures) && config.architectures.every((value) => typeof value === "string") ? [...config.architectures] : [],
    fields,
    field_provenance: provenance,
    gqa_query_heads_per_kv_head: fields.num_attention_heads % fields.num_key_value_heads === 0 ? fields.num_attention_heads / fields.num_key_value_heads : null,
    query_projection_width: queryProjectionWidth,
    kv_projection_width: kvProjectionWidth,
    kv_cache_elements_per_token_per_batch: kvStateProjection.elements_per_token_per_batch.value,
    kv_cache_elements_per_token_per_batch_decimal: kvStateProjection.elements_per_token_per_batch.decimal,
    kv_cache_elements_at_declared_max_context_batch_1: kvStateProjection.elements_at_context_batch_one.value,
    kv_cache_elements_at_declared_max_context_batch_1_decimal: kvStateProjection.elements_at_context_batch_one.decimal,
    declared_torch_dtype: declaredTorchDtype,
    declared_torch_dtype_bits: cacheBits,
    kv_cache_bytes_per_token_per_batch_if_cache_dtype_matches_declared: kvStateProjection.bytes_per_token_per_batch_if_storage_width_matches?.value ?? null,
    kv_cache_bytes_per_token_per_batch_if_cache_dtype_matches_declared_decimal: kvStateProjection.bytes_per_token_per_batch_if_storage_width_matches?.decimal ?? null,
    kv_cache_bytes_at_declared_max_context_batch_1_if_cache_dtype_matches_declared: kvStateProjection.bytes_at_context_batch_one_if_storage_width_matches?.value ?? null,
    kv_cache_bytes_at_declared_max_context_batch_1_if_cache_dtype_matches_declared_decimal: kvStateProjection.bytes_at_context_batch_one_if_storage_width_matches?.decimal ?? null,
    cache_dtype_boundary: "Element cardinality is architecture-derived. Byte projections apply only if the runtime KV-cache dtype equals config.torch_dtype; allocation layout, paging, alignment, and cache policy remain runtime evidence.",
    kv_state_projection: kvStateProjection,
    compute_projection: computeProjection,
    tensor_contract: tensorContract,
    issue_count: issues.length,
    error_count: errorCount,
    issues,
    source: sourceEvidence(),
    boundary: HF_SAFETENSORS_ARCHITECTURE_SOURCE.interpretation_boundary,
  };
}

export function supportedHfSafeTensorsModelTypes() {
  return Object.freeze(Object.keys(ARCHITECTURES));
}
