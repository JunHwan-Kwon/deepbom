import { assessOnnxAttributeProto } from "./onnx-schema-legality.js";
import {
  assessOnnxOrtTransformerSchemaForm,
  canInferOnnxOrtTransformerNode,
  inferOnnxOrtTransformerNode,
} from "./onnx-ort-transformer-shape-inference.js";
import {
  assessOnnxOrtRecurrentSchemaForm,
  canInferOnnxOrtRecurrentNode,
  inferOnnxOrtRecurrentNode,
} from "./onnx-ort-recurrent-shape-inference.js";
import {
  assessOnnxOrtMatMulSchemaForm,
  canInferOnnxOrtMatMulNode,
  inferOnnxOrtMatMulNode,
} from "./onnx-ort-matmul-shape-inference.js";
import { makeOnnxTensorTypeFromDimensions, onnxShapeDimensionsFromValue } from "./onnx-type-proto.js";

const ORT_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
const NORMALIZATION_ATTRIBUTES = Object.freeze(new Map([
  ["axis", 2],
  ["epsilon", 1],
  ["stash_type", 2],
]));
const MATMUL_NBITS_ATTRIBUTES = Object.freeze(new Map([
  ["K", 2], ["N", 2], ["bits", 2], ["block_size", 2], ["accuracy_level", 2],
]));
const MATMUL_BNB4_ATTRIBUTES = Object.freeze(new Map([
  ["K", 2], ["N", 2], ["block_size", 2], ["quant_type", 2], ["training_mode", 2], ["transB", 2],
]));

export const ONNX_ORT_EXTENSION_SHAPE_SOURCE = Object.freeze({
  release: "v1.23.0-source-pin",
  commit: ORT_COMMIT,
  documents: Object.freeze([
    Object.freeze({
      role: "ort_standard_domain_extension_schemas",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/contrib_defs.cc`,
      sha256: "e313a9ec5b8c11620445961c1f36da5ed894f70765ad94771e141de13b3e45ca",
    }),
    Object.freeze({
      role: "ort_transformer_contrib_schemas",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/contrib_ops/transformers/bert/bert_defs.cc`,
      sha256: "5783df1b2d56a6af0ba8a8b228ce165526fc3fcc5d162b00aa047709da621258",
    }),
    Object.freeze({
      role: "ort_transformer_shared_shape_functions",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/shape_inference_functions.cc`,
      sha256: "de6b66a75150dd390f52f9eaf5affe8a846ff5ff8b6641cf353d25e89fdd59db",
    }),
    Object.freeze({
      role: "ort_quantized_recurrent_contrib_schema",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/quantization_defs.cc`,
      sha256: "de215366d115e5b49fbc7ac0bbd09af19738cb05b9278469dd8a0a2537d28d6c",
    }),
  ]),
  interpretation_boundary: "These rules cover ORT contrib schemas registered in ai.onnx at the pinned ORT source commit. They do not promote those extensions to ONNX-standard operators or prove runtime kernel assignment.",
});

export function isOnnxOrtStandardDomainExtension(node, importedOpset) {
  if (normalizeDomain(node?.domain) !== "ai.onnx" || !Number.isSafeInteger(importedOpset) || importedOpset < 1) return false;
  if (node?.opType === "SimplifiedLayerNormalization") return true;
  return node?.opType === "LayerNormalization" && importedOpset < 17;
}

export function canInferOnnxOrtExtensionNode(node, importedOpset) {
  const domain = normalizeDomain(node?.domain);
  if (domain === "ai.onnx" && node?.opType === "SimplifiedLayerNormalization") return true;
  if (isOnnxOrtStandardDomainExtension(node, importedOpset)) return true;
  return domain === "com.microsoft" && (["MatMulNBits", "MatMulBnb4"].includes(node?.opType)
    || canInferOnnxOrtTransformerNode(node)
    || canInferOnnxOrtRecurrentNode(node)
    || canInferOnnxOrtMatMulNode(node));
}

export function assessOnnxOrtExtensionSchemaForm(node, importedOpset) {
  if (canInferOnnxOrtTransformerNode(node)) return assessOnnxOrtTransformerSchemaForm(node, importedOpset);
  if (canInferOnnxOrtRecurrentNode(node)) return assessOnnxOrtRecurrentSchemaForm(node, importedOpset);
  if (canInferOnnxOrtMatMulNode(node)) return assessOnnxOrtMatMulSchemaForm(node, importedOpset);
  const op = String(node?.opType || "UNKNOWN");
  const simplified = op === "SimplifiedLayerNormalization";
  const matMulNBits = op === "MatMulNBits";
  const matMulBnb4 = op === "MatMulBnb4";
  const standardDomainExtension = isOnnxOrtStandardDomainExtension(node, importedOpset);
  const attributes = matMulNBits ? MATMUL_NBITS_ATTRIBUTES : matMulBnb4 ? MATMUL_BNB4_ATTRIBUTES : NORMALIZATION_ATTRIBUTES;
  const requiredAttributes = matMulNBits
    ? ["K", "N", "block_size"]
    : matMulBnb4 ? ["K", "N", "block_size", "quant_type"] : [];
  const inputMinimum = matMulNBits || matMulBnb4 ? 3 : 2;
  const inputMaximum = matMulNBits ? 6 : matMulBnb4 ? 3 : simplified ? 2 : 3;
  const outputMaximum = matMulNBits || matMulBnb4 ? 1 : simplified ? 2 : 3;
  const uncheckedAttributesAllowed = standardDomainExtension;
  const base = {
    op_name: op,
    imported_opset: importedOpset,
    schema_since_version: 1,
    schema_source: standardDomainExtension ? "ort_contrib_standard_domain_extension" : "ort_contrib_schema",
    input_count: node?.inputs?.length || 0,
    output_count: node?.outputs?.length || 0,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
  };
  if (!canInferOnnxOrtExtensionNode(node, importedOpset)) {
    return { ...base, status: "fail", reason_codes: ["ort_extension_shape_rule_not_registered"], detail: `${normalizeDomain(node?.domain)}:${op} is not a pinned local ORT shape rule at opset ${importedOpset}.` };
  }

  const reasons = [];
  const unresolved = [];
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  if (inputs.length < inputMinimum) reasons.push(`input_count_below_minimum:${inputs.length}:${inputMinimum}`);
  if (inputs.length > inputMaximum) reasons.push(`input_count_above_maximum:${inputs.length}:${inputMaximum}`);
  if (outputs.length < 1) reasons.push(`output_count_below_minimum:${outputs.length}:1`);
  if (outputs.length > outputMaximum) reasons.push(`output_count_above_maximum:${outputs.length}:${outputMaximum}`);
  if (!inputs[0]) reasons.push("required_input_omitted:0");
  if (!inputs[1]) reasons.push("required_input_omitted:1");
  if ((matMulNBits || matMulBnb4) && !inputs[2]) reasons.push("required_input_omitted:2");
  if (!outputs[0]) reasons.push("required_output_omitted:0");
  for (const name of new Set(node.duplicateAttributeNames || [])) reasons.push(`duplicate_attribute_name:${name}`);
  for (const [name, attribute] of node.attributes || []) {
    const expected = attributes.get(name);
    if (expected == null) {
      if (!uncheckedAttributesAllowed) reasons.push(`attribute_not_defined:${name}`);
      continue;
    }
    const actual = assessOnnxAttributeProto(attribute);
    if (actual.status === "fail") reasons.push(`${actual.reason}:${name}`);
    else if (actual.status === "unresolved") unresolved.push(`${actual.reason}:${name}`);
    else if (actual.type !== expected) reasons.push(`attribute_type_mismatch:${name}:${expected}:${actual.type}`);
  }
  for (const name of requiredAttributes) if (!node.attributes?.has(name)) reasons.push(`required_attribute_missing:${name}`);
  reasons.push(...attributeValueReasons(node, op));
  return {
    ...base,
    schema_input_range: [inputMinimum, inputMaximum],
    schema_output_range: [1, outputMaximum],
    schema_attribute_count: attributes.size,
    required_attributes: requiredAttributes,
    unchecked_attributes_allowed: uncheckedAttributesAllowed,
    status: reasons.length ? "fail" : unresolved.length ? "unresolved" : "pass",
    reason_codes: reasons.length ? reasons : unresolved,
    detail: reasons.length
      ? `The NodeProto violates the pinned ORT ${op}-1 contrib schema.`
      : unresolved.length
        ? `The NodeProto attribute type cannot be proven from the parsed AttributeProto.`
        : `The NodeProto matches the pinned ORT ${normalizeDomain(node?.domain)}:${op}-1 contrib schema.`,
  };
}

export function inferOnnxOrtExtensionNode(node, tensors, tensorTypeName) {
  if (canInferOnnxOrtTransformerNode(node)) return inferOnnxOrtTransformerNode(node, tensors);
  if (canInferOnnxOrtRecurrentNode(node)) return inferOnnxOrtRecurrentNode(node, tensors);
  if (canInferOnnxOrtMatMulNode(node)) return inferOnnxOrtMatMulNode(node, tensors);
  if (node.opType === "MatMulNBits" || node.opType === "MatMulBnb4") return inferQuantizedWeightMatMul(node, tensors);
  const data = inputTensor(tensors, node.inputs?.[0]);
  const scale = inputTensor(tensors, node.inputs?.[1]);
  if (!tensorRankKnown(data)) return unresolved("ort_layer_normalization_input_rank_unknown");
  const dimensions = onnxShapeDimensionsFromValue(data);
  const axis = normalizeAxis(attributeInt(node, "axis", -1), dimensions.length);
  if (axis == null) return unresolved("ort_layer_normalization_axis_out_of_range");
  const outputDtype = knownDtype(scale);
  if (!outputDtype) return unresolved("ort_layer_normalization_scale_dtype_unknown");
  const stashDtype = tensorTypeName(attributeInt(node, "stash_type", 1));
  const outputs = [];
  const set = (index, shape, dtype) => {
    const name = node.outputs?.[index];
    if (!name) return true;
    const typeProto = makeOnnxTensorTypeFromDimensions(dtype, shape, true);
    outputs.push([name, { dtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto }]);
    return true;
  };
  set(0, dimensions, outputDtype);
  if (node.opType === "SimplifiedLayerNormalization") {
    const invStdShape = dimensions.map((dimension, index) => index === axis ? valueDimension(1) : cloneDimension(dimension));
    set(1, invStdShape, stashDtype);
  } else {
    const stashShape = dimensions.map((dimension, index) => index < axis ? cloneDimension(dimension) : valueDimension(1));
    set(1, stashShape, stashDtype);
    set(2, stashShape, stashDtype);
  }
  return { outputs, reason: "" };
}

function inferQuantizedWeightMatMul(node, tensors) {
  const data = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(data)) return unresolved("ort_quantized_matmul_input_rank_unknown");
  const dimensions = onnxShapeDimensionsFromValue(data);
  if (!dimensions.length) return unresolved("ort_quantized_matmul_input_rank_zero");
  const K = attributeInt(node, "K", -1);
  const N = attributeInt(node, "N", -1);
  if (!Number.isSafeInteger(K) || K <= 0 || !Number.isSafeInteger(N) || N <= 0) return unresolved("ort_quantized_matmul_feature_attributes_invalid");
  const transB = node.opType === "MatMulNBits" || attributeInt(node, "transB", 1) !== 0;
  const expectedInputFeature = transB ? K : N;
  const outputFeature = transB ? N : K;
  const actualInputFeature = dimensionValue(dimensions.at(-1));
  if (actualInputFeature != null && actualInputFeature !== expectedInputFeature) return unresolved("ort_quantized_matmul_input_feature_mismatch");
  if (node.opType === "MatMulNBits" && node.inputs?.[5]) {
    const bias = inputTensor(tensors, node.inputs[5]);
    if (!tensorRankKnown(bias) || bias.shape.length !== 1) return unresolved("ort_matmul_nbits_bias_shape_unknown_or_rank_invalid");
    const biasDimensions = onnxShapeDimensionsFromValue(bias);
    if (dimensionValue(biasDimensions[0]) !== N) return unresolved("ort_matmul_nbits_bias_shape_mismatch");
  }
  const dtype = knownDtype(data);
  if (!dtype) return unresolved("ort_quantized_matmul_input_dtype_unknown");
  const shape = [...dimensions.slice(0, -1).map(cloneDimension), valueDimension(outputFeature)];
  const typeProto = makeOnnxTensorTypeFromDimensions(dtype, shape, true);
  return { outputs: [[node.outputs[0], { dtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto }]], reason: "" };
}

function attributeValueReasons(node, op) {
  const reasons = [];
  const explicit = (name) => node.attributes?.has(name);
  const value = (name, fallback) => attributeInt(node, name, fallback);
  if (["MatMulNBits", "MatMulBnb4"].includes(op)) {
    for (const name of ["K", "N"]) if (explicit(name) && value(name, -1) <= 0) reasons.push(`attribute_value_not_positive:${name}`);
    if (explicit("block_size")) {
      const blockSize = value("block_size", -1);
      if (blockSize < 16 || (blockSize & (blockSize - 1)) !== 0) reasons.push("attribute_value_invalid:block_size:power_of_two_at_least_16");
    }
  }
  if (op === "MatMulNBits") {
    const bits = value("bits", 4);
    if (bits < 2 || bits > 8) reasons.push("attribute_value_out_of_range:bits:2:8");
    const accuracy = value("accuracy_level", 0);
    if (accuracy < 0 || accuracy > 4) reasons.push("attribute_value_out_of_range:accuracy_level:0:4");
  }
  if (op === "MatMulBnb4") {
    const quantType = value("quant_type", -1);
    if (quantType !== 0 && quantType !== 1) reasons.push("attribute_value_invalid:quant_type:0_or_1");
    for (const name of ["training_mode", "transB"]) {
      const current = value(name, name === "transB" ? 1 : 0);
      if (current !== 0 && current !== 1) reasons.push(`attribute_value_invalid:${name}:0_or_1`);
    }
  }
  return reasons;
}

function inputTensor(tensors, name) {
  const tensor = tensors.get(name);
  if (tensor?.valueKind && tensor.valueKind !== "tensor") return { ...tensor, dtype: "UNKNOWN", shape: null, shapeDeclared: false };
  return tensor && tensor.shapeDeclared !== true ? { ...tensor, shape: null } : tensor;
}
function tensorRankKnown(tensor) { return tensor?.shapeDeclared === true && Array.isArray(tensor.shape); }
function knownDtype(tensor) { const value = String(tensor?.dtype || ""); return value && value !== "UNKNOWN" ? value : ""; }
function attributeInt(node, name, fallback) { const value = node.attributes?.get(name)?.i; return Number.isSafeInteger(value) ? value : fallback; }
function normalizeAxis(axis, rank) { const value = axis < 0 ? axis + rank : axis; return Number.isSafeInteger(value) && value >= 0 && value < rank ? value : null; }
function normalizeDomain(value) { const domain = String(value || "").trim(); return domain || "ai.onnx"; }
function valueDimension(value) { return { kind: "value", value, parameter: "", denotation: "", valueFieldCount: 1 }; }
function cloneDimension(dimension) { return { ...dimension }; }
function dimensionValue(dimension) { return dimension?.kind === "value" && Number.isSafeInteger(dimension.value) ? dimension.value : null; }
function unresolved(reason) { return { outputs: [], reason }; }
