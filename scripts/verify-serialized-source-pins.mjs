import { createHash } from "node:crypto";
import { GGML_TYPE_TRAITS_SOURCE, SAFETENSORS_REFERENCE } from "../web/lib/metadata-model-adapters.js";
import { GGUF_DEQUANTIZATION_SOURCE, SAFETENSORS_NUMERICAL_SOURCE } from "../web/lib/tensor-numerical-integrity.js";
import { ggufCodebookBytesForVerification, ggufCodebookManifest } from "../web/lib/gguf-codebooks.generated.js";
import { HF_SAFETENSORS_ARCHITECTURE_SOURCE } from "../web/lib/hf-safetensors-contract.js";
import { GGUF_BACKEND_PROFILES, GGUF_BACKEND_SOURCE, GGUF_LLAMA_ARCHITECTURES } from "../web/lib/gguf-backend-contract.generated.js";
import { ONNX_OPERATION_COST_SOURCE } from "../web/onnx.js";
import { SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES, SAFETENSORS_QUANTIZATION_SOURCES } from "../web/lib/safetensors-quantization-contract.js";

const TFLITE_CONTROL_FLOW_SOURCE = Object.freeze({
  repository: "tensorflow/tensorflow",
  commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  files: Object.freeze([
    Object.freeze({ path: "tensorflow/compiler/mlir/lite/schema/schema.fbs", expected: "3bfa613428459de18db5d70d8581e7b6afd127c4522bb18ff59c8e589c3b75a1" }),
    Object.freeze({ path: "tensorflow/lite/kernels/if.cc", expected: "e23c06a2a3984ae704153c667f891be2bbd31d03be523a85cc976ea6ca75a428" }),
    Object.freeze({ path: "tensorflow/lite/kernels/while.cc", expected: "6ac2e230d01317af3c39bb2a730c8d1aa632cc707f9043b3d8bb6e2123389005" }),
    Object.freeze({ path: "tensorflow/lite/kernels/call_once.cc", expected: "21bf79b3ba4c47bf63090b5cf3b7734591f79ddcecc8b4364b18997ca159a44d" }),
    Object.freeze({ path: "tensorflow/lite/kernels/control_flow_common.h", expected: "acce5ae7657930db1eac7a7548c8e74ff91314fa2ad03681d3968125f6665a46" }),
    Object.freeze({ path: "tensorflow/lite/kernels/conv.cc", expected: "66a2fef9a8e7fe81b7bdd9d18bd099cc589546ac29cca7665711de890fba9281" }),
    Object.freeze({ path: "tensorflow/lite/kernels/depthwise_conv.cc", expected: "343f85c01e6adf2b21dbcd7e610ae04acf78f4ba1fea912e2fb02e33c92f6629" }),
    Object.freeze({ path: "tensorflow/lite/kernels/fully_connected.cc", expected: "a2667242af7d0d933d31408a0393974718e82da221248db9cb25aac2a8d3c585" }),
    Object.freeze({ path: "tensorflow/lite/kernels/conv3d.cc", expected: "7dfd75d047b7d22f76c365d48ecb1facad4656897ed3d58a661afcb0ad503b36" }),
  ]),
});

if (GGUF_DEQUANTIZATION_SOURCE.block_layout_source !== GGML_TYPE_TRAITS_SOURCE.block_layout_source
  || GGUF_DEQUANTIZATION_SOURCE.block_layout_source_sha256 !== GGML_TYPE_TRAITS_SOURCE.block_layout_source_sha256) {
  throw new Error("GGUF storage and dequantization provenance disagree on the pinned block-layout source.");
}

const rows = [
  ...ONNX_OPERATION_COST_SOURCE.documents.map((source) => ({
    repository: ONNX_OPERATION_COST_SOURCE.repository,
    commit: ONNX_OPERATION_COST_SOURCE.commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...TFLITE_CONTROL_FLOW_SOURCE.files.map((source) => ({
    repository: TFLITE_CONTROL_FLOW_SOURCE.repository,
    commit: TFLITE_CONTROL_FLOW_SOURCE.commit,
    path: source.path,
    expected: source.expected,
  })),
  {
    repository: GGML_TYPE_TRAITS_SOURCE.repository,
    commit: GGML_TYPE_TRAITS_SOURCE.source_commit,
    path: GGML_TYPE_TRAITS_SOURCE.type_traits_source,
    expected: GGML_TYPE_TRAITS_SOURCE.type_traits_source_sha256,
  },
  {
    repository: GGML_TYPE_TRAITS_SOURCE.repository,
    commit: GGML_TYPE_TRAITS_SOURCE.source_commit,
    path: GGML_TYPE_TRAITS_SOURCE.block_layout_source,
    expected: GGML_TYPE_TRAITS_SOURCE.block_layout_source_sha256,
  },
  {
    repository: GGUF_DEQUANTIZATION_SOURCE.repository,
    commit: GGUF_DEQUANTIZATION_SOURCE.source_commit,
    path: GGUF_DEQUANTIZATION_SOURCE.source,
    expected: GGUF_DEQUANTIZATION_SOURCE.source_sha256,
  },
  {
    repository: GGUF_DEQUANTIZATION_SOURCE.repository,
    commit: GGUF_DEQUANTIZATION_SOURCE.source_commit,
    path: GGUF_DEQUANTIZATION_SOURCE.numeric_format_source,
    expected: GGUF_DEQUANTIZATION_SOURCE.numeric_format_source_sha256,
  },
  {
    repository: GGUF_DEQUANTIZATION_SOURCE.format_specification_repository,
    commit: GGUF_DEQUANTIZATION_SOURCE.format_specification_commit,
    path: GGUF_DEQUANTIZATION_SOURCE.format_specification_source,
    expected: GGUF_DEQUANTIZATION_SOURCE.format_specification_source_sha256,
  },
  {
    repository: SAFETENSORS_REFERENCE.repository,
    commit: SAFETENSORS_REFERENCE.commit,
    path: SAFETENSORS_REFERENCE.tensor_source,
    expected: SAFETENSORS_REFERENCE.tensor_rs_sha256,
  },
  {
    repository: SAFETENSORS_REFERENCE.repository,
    commit: SAFETENSORS_REFERENCE.commit,
    path: SAFETENSORS_REFERENCE.torch_binding_source,
    expected: SAFETENSORS_REFERENCE.torch_binding_source_sha256,
  },
  ...SAFETENSORS_NUMERICAL_SOURCE.pytorch_sources.map((source) => ({
    repository: SAFETENSORS_NUMERICAL_SOURCE.pytorch_repository,
    commit: SAFETENSORS_NUMERICAL_SOURCE.pytorch_commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...Object.values(HF_SAFETENSORS_ARCHITECTURE_SOURCE.configuration_sources).map((source) => ({
    repository: HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository,
    commit: HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...Object.values(HF_SAFETENSORS_ARCHITECTURE_SOURCE.modeling_sources).map((source) => ({
    repository: HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository,
    commit: HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...Object.values(GGUF_BACKEND_SOURCE.files).map((source) => ({
    repository: GGUF_BACKEND_SOURCE.repository,
    commit: GGUF_BACKEND_SOURCE.source_commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...Object.entries(SAFETENSORS_QUANTIZATION_SOURCES).map(([method, source]) => ({
    method,
    repository: source.repository,
    commit: source.commit,
    path: source.path,
    expected: source.sha256,
  })),
  ...Object.entries(SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES).flatMap(([method, sources]) => sources.map((source) => ({
    method,
    repository: source.repository || SAFETENSORS_QUANTIZATION_SOURCES[method].repository,
    commit: source.commit || SAFETENSORS_QUANTIZATION_SOURCES[method].commit,
    path: source.path,
    expected: source.sha256,
  }))),
];

const fetched = new Map();
for (const row of rows) {
  const url = `https://raw.githubusercontent.com/${row.repository}/${row.commit}/${row.path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${row.repository}/${row.path}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== row.expected) throw new Error(`${row.repository}@${row.commit}/${row.path}: expected ${row.expected}, got ${actual}`);
  fetched.set(`${row.repository}@${row.commit}/${row.path}`, bytes);
  console.log(`${row.repository}@${row.commit}/${row.path}: ${bytes.length} B ${actual}`);
}

const commonKey = `${GGML_TYPE_TRAITS_SOURCE.repository}@${GGML_TYPE_TRAITS_SOURCE.source_commit}/${GGML_TYPE_TRAITS_SOURCE.block_layout_source}`;
const commonSource = fetched.get(commonKey)?.toString("utf8");
if (!commonSource) throw new Error("Pinned GGUF common source was not retained for codebook verification.");
const ggufSpecKey = `${GGUF_DEQUANTIZATION_SOURCE.format_specification_repository}@${GGUF_DEQUANTIZATION_SOURCE.format_specification_commit}/${GGUF_DEQUANTIZATION_SOURCE.format_specification_source}`;
const ggufSpecSource = fetched.get(ggufSpecKey)?.toString("utf8") || "";
const ggufEndianMarkers = [
  "Models are little-endian by default.",
  "all values (including metadata values and tensors) will also be big-endian",
  "Adds big-endian support.",
];
if (!ggufEndianMarkers.every((marker) => ggufSpecSource.includes(marker))) {
  throw new Error("Pinned GGUF v3 endian semantics no longer reproduce from the format specification.");
}
const ggufTypedFieldMarkers = [
  "uint8_t qh[4]",
  "uint16_t qs[QK_K/8]",
  "int16_t bsums[QK_K/16]",
  "uint16_t scales_h",
];
if (!ggufTypedFieldMarkers.every((marker) => commonSource.includes(marker))) {
  throw new Error("Pinned GGUF block field types no longer reproduce the source-field endian contract.");
}
for (const [name, metadata] of Object.entries(ggufCodebookManifest())) {
  const match = commonSource.match(new RegExp(`GGML_TABLE_BEGIN\\([^,]+,\\s*${name},[^\\n]*\\)\\s*([\\s\\S]*?)GGML_TABLE_END\\(\\)`));
  if (!match) throw new Error(`Pinned GGUF common source is missing generated codebook ${name}.`);
  const tokens = match[1].replace(/\/\/.*$/gm, "").match(/0x[0-9a-fA-F]+|\b\d+\b/g) || [];
  if (tokens.length !== metadata.entries) throw new Error(`GGUF codebook ${name} entry count does not reproduce.`);
  const expected = Buffer.alloc(metadata.byte_length);
  tokens.forEach((token, index) => {
    let value = BigInt(token);
    for (let offset = 0; offset < metadata.width; offset += 1) {
      expected[index * metadata.width + offset] = Number(value & 255n);
      value >>= 8n;
    }
    if (value !== 0n) throw new Error(`GGUF codebook ${name} entry ${index} exceeds its declared width.`);
  });
  const generated = Buffer.from(ggufCodebookBytesForVerification(name));
  if (!generated.equals(expected)) throw new Error(`Generated GGUF codebook ${name} differs from the pinned source bytes.`);
  const digest = createHash("sha256").update(generated).digest("hex");
  if (digest !== metadata.sha256) throw new Error(`Generated GGUF codebook ${name} digest does not reproduce.`);
}

const tfliteSource = (path) => fetched.get(`${TFLITE_CONTROL_FLOW_SOURCE.repository}@${TFLITE_CONTROL_FLOW_SOURCE.commit}/${path}`)?.toString("utf8") || "";
const onnxCostSource = (path) => fetched.get(`${ONNX_OPERATION_COST_SOURCE.repository}@${ONNX_OPERATION_COST_SOURCE.commit}/${path}`)?.toString("utf8") || "";
const onnxCostMarkers = new Map([
  ["onnx/defs/nn/defs.cc", [
    "ONNX_OPERATOR_SET_SCHEMA(Conv,", "QLinearConv", "ConvInteger",
    "ONNX_API void convTransposeShapeInference", "strides[i] * (input_shape.dim(i + 2).dim_value() - 1)",
    "effective_kernel_shape[i] - pads[i] - pads[i + n_input_dims]",
    "DeformConv_ver22_doc", "offset_group", "Attention_ver24_doc",
    "KTranspose = Transpose", "YPreReshape = MatMul(SoftmaxOut, VAttentionInput)",
  ]],
  ["onnx/defs/math/defs.cc", [
    "Gemm,", "MatMul,", "QLinearMatMul", "MatMulInteger",
    "Einsum_ver12_doc", "The Einsum operator evaluates algebraic tensor operations",
  ]],
]);
for (const [path, markers] of onnxCostMarkers) {
  const source = onnxCostSource(path);
  if (!markers.every((marker) => source.includes(marker))) throw new Error(`Pinned ONNX operation-cost semantics no longer reproduce for ${path}.`);
}
const tfliteMarkers = new Map([
  ["tensorflow/compiler/mlir/lite/schema/schema.fbs", ["table IfOptions", "table WhileOptions", "table CallOnceOptions", "union BuiltinOptions"]],
  ["tensorflow/lite/kernels/if.cc", ["TF_LITE_ENSURE_EQ(context, cond->type, kTfLiteBool)", "TF_LITE_ENSURE_EQ(context, NumElements(cond), 1)", "num_inputs, subgraph->inputs().size()"]],
  ["tensorflow/lite/kernels/while.cc", ["The condition output must be a single boolean value", "cond_output->dims->size, 1", "body_input->type, body_output->type"]],
  ["tensorflow/lite/kernels/call_once.cc", ["node->inputs->size, 0", "node->outputs->size, 0", "init_subgraph->inputs().size(), 0", "init_subgraph->outputs().size(), 0"]],
  ["tensorflow/lite/kernels/control_flow_common.h", ["CopyTensorsShapeAndType", "dst_tensor->type = src_tensor->type"]],
  ["tensorflow/lite/kernels/conv.cc", ["[filter_count, filter_height, filter_width, input_depth]", "int channels_in = filter->dims->data[3]", "int channels_out = filter->dims->data[0]"]],
  ["tensorflow/lite/kernels/depthwise_conv.cc", ["SizeOfDimension(filter, 0), 1", "int channels_out = SizeOfDimension(filter, 3)", "int filter_width = SizeOfDimension(filter, 2)"]],
  ["tensorflow/lite/kernels/fully_connected.cc", ["const int num_units = filter->dims->data[0]", "const int batch_size = input_size / filter->dims->data[1]"]],
  ["tensorflow/lite/kernels/conv3d.cc", ["input->dims->data[4], filter->dims->data[3]", "[filter_depth, filter_height, filter_width,", "int channels_out = filter->dims->data[4]"]],
]);
for (const [path, markers] of tfliteMarkers) {
  const source = tfliteSource(path);
  if (!markers.every((marker) => source.includes(marker))) throw new Error(`Pinned TFLite serialized-contract semantics no longer reproduce for ${path}.`);
}

const safeTensorSource = fetched.get(`${SAFETENSORS_REFERENCE.repository}@${SAFETENSORS_REFERENCE.commit}/${SAFETENSORS_REFERENCE.tensor_source}`)?.toString("utf8") || "";
const safeTorchSource = fetched.get(`${SAFETENSORS_REFERENCE.repository}@${SAFETENSORS_REFERENCE.commit}/${SAFETENSORS_REFERENCE.torch_binding_source}`)?.toString("utf8") || "";
if (!["F4 => 4", "F6_E3M2 => 6", "F6_E2M3 => 6", "F8_E8M0 => 8"].every((token) => safeTensorSource.includes(token))) {
  throw new Error("Pinned SafeTensors dtype cardinalities no longer reproduce.");
}
if (!["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn_x2"].every((token) => safeTorchSource.includes(token))) {
  throw new Error("Pinned SafeTensors PyTorch dtype bindings no longer reproduce.");
}
const pytorchSource = (path) => fetched.get(`${SAFETENSORS_NUMERICAL_SOURCE.pytorch_repository}@${SAFETENSORS_NUMERICAL_SOURCE.pytorch_commit}/${path}`)?.toString("utf8") || "";
if (!pytorchSource("torch/headeronly/util/Float4_e2m1fn_x2.h").includes("original value             | val1 : val0")) {
  throw new Error("Pinned PyTorch F4 packed element order no longer reproduces.");
}
const semanticMarkers = new Map([
  ["torch/headeronly/util/Float8_e4m3fn.h", ["bias = 7", "(x & 0b01111111) == 0b01111111"]],
  ["torch/headeronly/util/Float8_e5m2.h", ["bias = 15", "(x & 0b01111111) == 0b01111100"]],
  ["torch/headeronly/util/Float8_e4m3fnuz.h", ["bias = 8", "x == 0b10000000"]],
  ["torch/headeronly/util/Float8_e5m2fnuz.h", ["bias = 16", "x == 0b10000000"]],
  ["torch/headeronly/util/Float8_e8m0fnu.h", ["x == 0b11111111", "0x00400000"]],
]);
for (const [path, markers] of semanticMarkers) {
  const source = pytorchSource(path);
  if (!markers.every((marker) => source.includes(marker))) throw new Error(`Pinned PyTorch low-precision semantics no longer reproduce for ${path}.`);
}

const quantizationSource = (method, path) => {
  const primary = SAFETENSORS_QUANTIZATION_SOURCES[method];
  const auxiliary = (SAFETENSORS_QUANTIZATION_AUXILIARY_SOURCES[method] || []).find((row) => row.path === path);
  const repository = auxiliary?.repository || primary.repository;
  const commit = auxiliary?.commit || primary.commit;
  return fetched.get(`${repository}@${commit}/${path}`)?.toString("utf8") || "";
};
const quantizationMarkers = new Map([
  ["gptqmodel/nn_modules/qlinear/__init__.py", ["class GPTQQuantLinear", '"qweight"', '"qzeros"', '"scales"', '"g_idx"', "self.pack_factor = self.pack_dtype_bits // self.bits"]],
  ["hqq/core/quantize.py", ["SUPPORTED_BITS", "group_size", "axis", "packing"]],
  ["hqq/models/base.py", ["name_to_linear_tag", "model.linear_tags", "patch_params.update(quant_config)"]],
  ["src/compressed_tensors/compressors/quantized_compressors/pack_quantized.py", ["weight_packed", "weight_scale", "weight_shape"]],
  ["src/compressed_tensors/quantization/lifecycle/apply.py", ["find_name_or_class_matches", "re.match(pattern, value)", "_merge_schemes"]],
  ["src/compressed_tensors/quantization/quant_config.py", ["config_groups", "kv_cache_scheme", "ignore"]],
]);
for (const [path, markers] of quantizationMarkers) {
  const method = path.startsWith("hqq/") ? "hqq" : path.startsWith("gptqmodel/") ? "gptq" : "compressed-tensors";
  const source = quantizationSource(method, path);
  if (!markers.every((marker) => source.includes(marker))) throw new Error(`Pinned ${method} quantization semantics no longer reproduce for ${path}.`);
}

for (const [modelType, sourceRow] of Object.entries(HF_SAFETENSORS_ARCHITECTURE_SOURCE.configuration_sources)) {
  const source = fetched.get(`${HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository}@${HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit}/${sourceRow.path}`)?.toString("utf8") || "";
  const commonMarkers = modelType === "jamba"
    ? [`model_type = "${modelType}"`, "layers_block_type", "layers_num_experts", "attn_layer_period", "attn_layer_offset", "expert_layer_period", "expert_layer_offset", "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_dt_rank"]
    : modelType === "mamba"
    ? [`model_type = "${modelType}"`, "num_hidden_layers", "hidden_size", "state_size", "conv_kernel", "expand", "time_step_rank"]
    : modelType === "mixtral"
      ? [`model_type = "${modelType}"`, "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "max_position_embeddings", "num_experts_per_tok", "num_local_experts"]
      : [`model_type = "${modelType}"`, "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "max_position_embeddings"];
  if (!commonMarkers.every((marker) => source.includes(marker))) throw new Error(`Pinned Transformers architecture semantics no longer reproduce for ${modelType}.`);
}
const hfModelingMarkers = Object.freeze({
  llama: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "config.attention_bias", "config.mlp_bias"],
  mistral: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "self.up_proj =", "self.down_proj ="],
  qwen2: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "bias=True", "self.o_proj =", "bias=False"],
  qwen3: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.q_norm =", "self.k_norm =", "config.attention_bias"],
  gemma: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "config.attention_bias"],
  gemma2: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.pre_feedforward_layernorm =", "self.post_feedforward_layernorm =", "config.attention_bias"],
  olmo2: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.q_norm =", "self.k_norm =", "config.attention_bias"],
  granite: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "config.attention_bias", "config.mlp_bias"],
  phi3: ["self.qkv_proj =", "self.gate_up_proj =", "self.down_proj =", "self.input_layernorm =", "self.post_attention_layernorm ="],
  cohere: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "self.input_layernorm =", "self.use_qk_norm ="],
  cohere2: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "self.input_layernorm =", "config.layer_types"],
  nemotron: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.up_proj =", "self.down_proj =", "self.post_attention_layernorm ="],
  ministral: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "self.up_proj =", "self.down_proj ="],
  smollm3: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "config.mlp_bias", "config.layer_types"],
  exaone4: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.q_norm =", "self.k_norm =", "self.post_feedforward_layernorm ="],
  olmo: ["self.q_proj =", "self.k_proj =", "self.v_proj =", "self.gate_proj =", "self.up_proj =", "self.down_proj ="],
  mixtral: ["self.gate =", "self.experts =", "self.top_k =", "self.w1 =", "self.w2 =", "self.w3 =", "torch.topk"],
  mamba: ["self.in_proj =", "self.conv1d =", "self.x_proj =", "self.dt_proj =", "self.A_log", "self.D =", "self.out_proj =", "self.ssm_states", "self.conv_states"],
  jamba: ["class HybridMambaAttentionDynamicCache", "self.q_proj =", "self.k_proj =", "self.v_proj =", "self.in_proj =", "self.conv1d =", "self.x_proj =", "self.dt_proj =", "self.A_log", "self.router =", "self.experts =", "self.input_layernorm =", "self.pre_ff_layernorm ="],
});
for (const [modelType, sourceRow] of Object.entries(HF_SAFETENSORS_ARCHITECTURE_SOURCE.modeling_sources)) {
  const source = fetched.get(`${HF_SAFETENSORS_ARCHITECTURE_SOURCE.repository}@${HF_SAFETENSORS_ARCHITECTURE_SOURCE.source_commit}/${sourceRow.path}`)?.toString("utf8") || "";
  const markers = hfModelingMarkers[modelType] || [];
  if (!markers.length || !markers.every((marker) => source.includes(marker))) throw new Error(`Pinned Transformers state-dict module semantics no longer reproduce for ${modelType}.`);
}

const ggufBackendSource = (id) => {
  const row = GGUF_BACKEND_SOURCE.files[id];
  return fetched.get(`${GGUF_BACKEND_SOURCE.repository}@${GGUF_BACKEND_SOURCE.source_commit}/${row.path}`)?.toString("utf8") || "";
};
const architectureSource = ggufBackendSource("architecture_registry");
if (!GGUF_LLAMA_ARCHITECTURES.every((name) => architectureSource.includes(`"${name}"`))) throw new Error("Generated GGUF architecture registry no longer reproduces from pinned llama.cpp source.");
const buildOptionSource = ggufBackendSource("build_options");
const backendRegistrationSource = ggufBackendSource("backend_registration");
for (const profile of GGUF_BACKEND_PROFILES) {
  if (!buildOptionSource.includes(`option(${profile.cmake_option}`)
    || !backendRegistrationSource.includes(profile.compiled_registration_macro)
    || !backendRegistrationSource.includes(profile.registration_function)) {
    throw new Error(`Generated GGUF backend profile ${profile.id} no longer reproduces from pinned llama.cpp source.`);
  }
}

console.log(`Serialized-format source pins verified (${rows.length} files and ${Object.keys(ggufCodebookManifest()).length} generated GGUF codebooks).`);
