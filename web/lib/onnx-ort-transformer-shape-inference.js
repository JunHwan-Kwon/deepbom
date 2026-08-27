import { assessOnnxAttributeProto } from "./onnx-schema-legality.js";
import { makeOnnxTensorTypeFromDimensions, onnxShapeDimensionsFromValue } from "./onnx-type-proto.js";

const FLOAT = 1;
const INT = 2;
const STRING = 3;

const SPECS = Object.freeze(new Map([
  ["FastGelu", spec(1, 2, 1, 1, {}, [], [0])],
  ["RotaryEmbedding", spec(4, 4, 1, 1, {
    scale: FLOAT,
    interleaved: INT,
    rotary_embedding_dim: INT,
    num_heads: INT,
    is_packed_batching: INT,
  }, [], [0, 1, 2, 3])],
  ["SkipLayerNormalization", spec(3, 5, 1, 4, { epsilon: FLOAT }, [], [0, 1, 2])],
  ["SkipSimplifiedLayerNormalization", spec(3, 4, 1, 4, { epsilon: FLOAT }, [], [0, 1, 2])],
  ["MultiHeadAttention", spec(1, 10, 1, 4, {
    num_heads: INT,
    mask_filter_value: FLOAT,
    scale: FLOAT,
    unidirectional: INT,
  }, ["num_heads"], [0])],
  ["GroupQueryAttention", spec(7, 14, 3, 4, {
    num_heads: INT,
    kv_num_heads: INT,
    scale: FLOAT,
    softcap: FLOAT,
    local_window_size: INT,
    do_rotary: INT,
    rotary_interleaved: INT,
    smooth_softmax: INT,
    qk_output: INT,
    k_quant_type: STRING,
    v_quant_type: STRING,
    kv_cache_bit_width: INT,
  }, ["num_heads", "kv_num_heads"], [0, 5, 6])],
]));

export const ORT_TRANSFORMER_SHAPE_OPS = Object.freeze(new Set(SPECS.keys()));

export function canInferOnnxOrtTransformerNode(node) {
  return normalizeDomain(node?.domain) === "com.microsoft" && ORT_TRANSFORMER_SHAPE_OPS.has(node?.opType);
}

export function assessOnnxOrtTransformerSchemaForm(node, importedOpset) {
  const op = String(node?.opType || "UNKNOWN");
  const definition = SPECS.get(op);
  const base = {
    op_name: op,
    imported_opset: importedOpset,
    schema_since_version: 1,
    schema_source: "ort_contrib_transformer_schema",
    input_count: node?.inputs?.length || 0,
    output_count: node?.outputs?.length || 0,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
  };
  if (!definition || !canInferOnnxOrtTransformerNode(node)) {
    return { ...base, status: "fail", reason_codes: ["ort_transformer_shape_rule_not_registered"], detail: `${normalizeDomain(node?.domain)}:${op} is not a pinned ORT transformer shape rule.` };
  }

  const reasons = [];
  const unresolved = [];
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  if (inputs.length < definition.inputMinimum) reasons.push(`input_count_below_minimum:${inputs.length}:${definition.inputMinimum}`);
  if (inputs.length > definition.inputMaximum) reasons.push(`input_count_above_maximum:${inputs.length}:${definition.inputMaximum}`);
  if (outputs.length < definition.outputMinimum) reasons.push(`output_count_below_minimum:${outputs.length}:${definition.outputMinimum}`);
  if (outputs.length > definition.outputMaximum) reasons.push(`output_count_above_maximum:${outputs.length}:${definition.outputMaximum}`);
  for (const index of definition.requiredInputs) if (!inputs[index]) reasons.push(`required_input_omitted:${index}`);
  for (let index = 0; index < definition.outputMinimum; index += 1) if (!outputs[index]) reasons.push(`required_output_omitted:${index}`);
  for (const name of new Set(node.duplicateAttributeNames || [])) reasons.push(`duplicate_attribute_name:${name}`);
  for (const [name, attribute] of node.attributes || []) {
    const expected = definition.attributes.get(name);
    if (expected == null) {
      reasons.push(`attribute_not_defined:${name}`);
      continue;
    }
    const actual = assessOnnxAttributeProto(attribute);
    if (actual.status === "fail") reasons.push(`${actual.reason}:${name}`);
    else if (actual.status === "unresolved") unresolved.push(`${actual.reason}:${name}`);
    else if (actual.type !== expected) reasons.push(`attribute_type_mismatch:${name}:${expected}:${actual.type}`);
  }
  for (const name of definition.requiredAttributes) if (!node.attributes?.has(name)) reasons.push(`required_attribute_missing:${name}`);
  reasons.push(...attributeValueReasons(node, op));
  return {
    ...base,
    schema_input_range: [definition.inputMinimum, definition.inputMaximum],
    schema_output_range: [definition.outputMinimum, definition.outputMaximum],
    schema_attribute_count: definition.attributes.size,
    required_attributes: [...definition.requiredAttributes],
    status: reasons.length ? "fail" : unresolved.length ? "unresolved" : "pass",
    reason_codes: reasons.length ? reasons : unresolved,
    detail: reasons.length
      ? `The NodeProto violates the pinned ORT ${op}-1 contrib schema.`
      : unresolved.length
        ? "The NodeProto attribute type cannot be proven from the parsed AttributeProto."
        : `The NodeProto matches the pinned ORT com.microsoft:${op}-1 contrib schema.`,
  };
}

export function inferOnnxOrtTransformerNode(node, tensors) {
  switch (node?.opType) {
    case "FastGelu":
    case "RotaryEmbedding":
      return inferSameShape(node, tensors);
    case "SkipLayerNormalization":
    case "SkipSimplifiedLayerNormalization":
      return inferSkipLayerNormalization(node, tensors);
    case "MultiHeadAttention":
      return inferMultiHeadAttention(node, tensors);
    case "GroupQueryAttention":
      return inferGroupQueryAttention(node, tensors);
    default:
      return unresolved("ort_transformer_shape_rule_not_registered");
  }
}

function inferSameShape(node, tensors) {
  const input = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(input)) return unresolved("ort_transformer_input_rank_unknown");
  const dtype = knownDtype(input);
  if (!dtype) return unresolved("ort_transformer_input_dtype_unknown");
  return resultWith(node, [[0, onnxShapeDimensionsFromValue(input), dtype]]);
}

function inferSkipLayerNormalization(node, tensors) {
  const input = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(input) || input.shape.length < 1) return unresolved("ort_skip_layer_normalization_input_rank_unknown_or_zero");
  const dtype = knownDtype(input);
  if (!dtype) return unresolved("ort_skip_layer_normalization_input_dtype_unknown");
  const shape = onnxShapeDimensionsFromValue(input);
  const stashShape = shape.map((dimension, index) => index === shape.length - 1 ? valueDimension(1) : cloneDimension(dimension));
  return resultWith(node, [
    [0, shape, dtype],
    [1, stashShape, "FLOAT32"],
    [2, stashShape, "FLOAT32"],
    [3, shape, dtype],
  ]);
}

function inferMultiHeadAttention(node, tensors) {
  const query = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(query)) return unresolved("ort_multi_head_attention_query_rank_unknown");
  const queryShape = onnxShapeDimensionsFromValue(query);
  if (![3, 5].includes(queryShape.length)) return unresolved("ort_multi_head_attention_query_rank_invalid");
  const queryDtype = knownDtype(query);
  if (!queryDtype) return unresolved("ort_multi_head_attention_query_dtype_unknown");

  let outputShape = null;
  let valueSequenceLength = null;
  if (queryShape.length === 5) {
    outputShape = [cloneDimension(queryShape[0]), cloneDimension(queryShape[1]), multiplyDimensions(queryShape[2], queryShape[4])];
  } else {
    const value = inputTensor(tensors, node.inputs?.[2]);
    const key = inputTensor(tensors, node.inputs?.[1]);
    if (tensorRankKnown(value)) {
      const valueShape = onnxShapeDimensionsFromValue(value);
      if (![3, 4].includes(valueShape.length)) return unresolved("ort_multi_head_attention_value_rank_invalid");
      if (valueShape.length === 3) valueSequenceLength = dimensionValue(valueShape[1]);
      const hidden = valueShape.length === 3
        ? cloneDimension(valueShape[2])
        : multiplyDimensions(valueShape[1], valueShape[3]);
      outputShape = [cloneDimension(queryShape[0]), cloneDimension(queryShape[1]), hidden];
    } else if (tensorRankKnown(key) && key.shape.length === 5) {
      outputShape = queryShape.map(cloneDimension);
    }
  }
  if (!outputShape) return unresolved("ort_multi_head_attention_output_shape_inputs_unknown");

  const entries = [[0, outputShape, queryDtype]];
  const pastKey = inputTensor(tensors, node.inputs?.[6]);
  const pastValue = inputTensor(tensors, node.inputs?.[7]);
  const sharing = tensorRankKnown(pastKey) && tensorRankKnown(inputTensor(tensors, node.inputs?.[8]));
  if (!sharing && valueSequenceLength > 0 && tensorRankKnown(pastKey) && pastKey.shape.length === 4) {
    const pastShape = onnxShapeDimensionsFromValue(pastKey);
    const pastLength = dimensionValue(pastShape[2]);
    if (pastLength != null) {
      const presentShape = pastShape.map(cloneDimension);
      presentShape[2] = valueDimension(pastLength + valueSequenceLength);
      const pastKeyDtype = knownDtype(pastKey);
      const pastValueDtype = knownDtype(pastValue);
      if (pastKeyDtype) entries.push([1, presentShape, pastKeyDtype]);
      if (pastValueDtype) entries.push([2, presentShape, pastValueDtype]);
    }
  }
  return resultWith(node, entries);
}

function inferGroupQueryAttention(node, tensors) {
  const query = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(query) || query.shape.length !== 3) return unresolved("ort_group_query_attention_query_rank_unknown_or_invalid");
  const queryShape = onnxShapeDimensionsFromValue(query);
  const queryDtype = knownDtype(query);
  if (!queryDtype) return unresolved("ort_group_query_attention_query_dtype_unknown");
  const numHeads = attributeInt(node, "num_heads", 0);
  const kvNumHeads = attributeInt(node, "kv_num_heads", 0);
  if (numHeads <= 0 || kvNumHeads <= 0) return unresolved("ort_group_query_attention_head_attributes_invalid");

  const value = inputTensor(tensors, node.inputs?.[2]);
  const valuePresent = Boolean(node.inputs?.[2]);
  let outputShape;
  let kvSequenceLength = null;
  let headSize = null;
  if (valuePresent) {
    if (tensorRankKnown(value) && value.shape.length === 3) kvSequenceLength = dimensionValue(onnxShapeDimensionsFromValue(value)[1]);
    outputShape = queryShape.map(cloneDimension);
    const hidden = dimensionValue(queryShape[2]);
    if (hidden != null && hidden % numHeads === 0) headSize = hidden / numHeads;
  } else {
    const packedHidden = dimensionValue(queryShape[2]);
    const divisor = numHeads + 2 * kvNumHeads;
    if (packedHidden == null || packedHidden % divisor !== 0) return unresolved("ort_group_query_attention_packed_hidden_size_unresolved");
    headSize = packedHidden / divisor;
    outputShape = [cloneDimension(queryShape[0]), cloneDimension(queryShape[1]), valueDimension(headSize * numHeads)];
    kvSequenceLength = dimensionValue(queryShape[1]);
  }

  const entries = [[0, outputShape, queryDtype]];
  const totalSequenceLength = staticScalar(inputTensor(tensors, node.inputs?.[6]));
  const pastKey = inputTensor(tensors, node.inputs?.[3]);
  const pastValue = inputTensor(tensors, node.inputs?.[4]);
  if (tensorRankKnown(pastKey) && pastKey.shape.length === 4) {
    const pastShape = onnxShapeDimensionsFromValue(pastKey);
    const presentShape = pastShape.map(cloneDimension);
    const pastLength = dimensionValue(pastShape[2]);
    if (totalSequenceLength > 0 && pastLength != null) presentShape[2] = valueDimension(Math.max(totalSequenceLength, pastLength));
    else if (pastLength === 0 && node.inputs?.[6]) presentShape[2] = unknownDimension();
    const pastKeyDtype = knownDtype(pastKey);
    const pastValueDtype = knownDtype(pastValue);
    if (pastKeyDtype) entries.push([1, presentShape, pastKeyDtype]);
    if (pastValueDtype) entries.push([2, presentShape, pastValueDtype]);
    const qkOutput = attributeInt(node, "qk_output", 0);
    if (node.outputs?.[3] && qkOutput !== 0 && totalSequenceLength > 0 && dimensionsConcrete(queryShape.slice(0, 2))) {
      entries.push([3, [cloneDimension(queryShape[0]), valueDimension(numHeads), cloneDimension(queryShape[1]), valueDimension(totalSequenceLength)], queryDtype]);
    }
  } else {
    if (headSize == null) return resultWith(node, entries);
    const sequence = totalSequenceLength > 0
      ? valueDimension(totalSequenceLength)
      : kvSequenceLength > 0 ? valueDimension(kvSequenceLength) : cloneDimension(queryShape[1]);
    const presentShape = [cloneDimension(queryShape[0]), valueDimension(kvNumHeads), sequence, valueDimension(headSize)];
    entries.push([1, presentShape, queryDtype], [2, presentShape, queryDtype]);
  }
  return resultWith(node, entries);
}

function attributeValueReasons(node, op) {
  const reasons = [];
  if (["MultiHeadAttention", "GroupQueryAttention"].includes(op) && attributeInt(node, "num_heads", 0) <= 0) {
    reasons.push("attribute_value_not_positive:num_heads");
  }
  if (op === "MultiHeadAttention") {
    const unidirectional = attributeInt(node, "unidirectional", 0);
    if (![0, 1].includes(unidirectional)) reasons.push("attribute_value_invalid:unidirectional:0_or_1");
  }
  if (op === "GroupQueryAttention") {
    if (attributeInt(node, "kv_num_heads", 0) <= 0) reasons.push("attribute_value_not_positive:kv_num_heads");
    for (const name of ["do_rotary", "rotary_interleaved"]) {
      const value = attributeInt(node, name, 0);
      if (![0, 1].includes(value)) reasons.push(`attribute_value_invalid:${name}:0_or_1`);
    }
    const qkOutput = attributeInt(node, "qk_output", 0);
    if (![0, 1, 2].includes(qkOutput)) reasons.push("attribute_value_invalid:qk_output:0_1_or_2");
    const bitWidth = attributeInt(node, "kv_cache_bit_width", 8);
    if (![4, 8].includes(bitWidth)) reasons.push("attribute_value_invalid:kv_cache_bit_width:4_or_8");
    for (const name of ["k_quant_type", "v_quant_type"]) {
      const value = attributeString(node, name, "NONE");
      if (!["NONE", "PER_TENSOR", "PER_CHANNEL"].includes(value)) reasons.push(`attribute_value_invalid:${name}:NONE_PER_TENSOR_OR_PER_CHANNEL`);
    }
  }
  return reasons;
}

function spec(inputMinimum, inputMaximum, outputMinimum, outputMaximum, attributes, requiredAttributes, requiredInputs) {
  return Object.freeze({
    inputMinimum, inputMaximum, outputMinimum, outputMaximum,
    attributes: Object.freeze(new Map(Object.entries(attributes))),
    requiredAttributes: Object.freeze([...requiredAttributes]),
    requiredInputs: Object.freeze([...requiredInputs]),
  });
}

function resultWith(node, entries) {
  const outputs = [];
  for (const [index, shape, dtype] of entries) {
    const name = node.outputs?.[index];
    if (!name || !dtype || !Array.isArray(shape)) continue;
    const typeProto = makeOnnxTensorTypeFromDimensions(dtype, shape, true);
    outputs.push([name, { dtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto }]);
  }
  return { outputs, reason: outputs.length ? "" : "ort_transformer_output_not_materialized" };
}

function inputTensor(tensors, name) {
  const tensor = tensors.get(name);
  if (tensor?.valueKind && tensor.valueKind !== "tensor") return { ...tensor, dtype: "UNKNOWN", shape: null, shapeDeclared: false };
  return tensor && tensor.shapeDeclared !== true ? { ...tensor, shape: null } : tensor;
}
function tensorRankKnown(tensor) { return tensor?.shapeDeclared === true && Array.isArray(tensor.shape); }
function knownDtype(tensor) { const value = String(tensor?.dtype || ""); return value && value !== "UNKNOWN" ? value : ""; }
function normalizeDomain(value) { const domain = String(value || "").trim(); return domain || "ai.onnx"; }
function attributeInt(node, name, fallback) { const value = node.attributes?.get(name)?.i; return Number.isSafeInteger(value) ? value : fallback; }
function attributeString(node, name, fallback) { const value = node.attributes?.get(name)?.s; return typeof value === "string" ? value : fallback; }
function staticScalar(tensor) {
  if (!Array.isArray(tensor?.staticValues) || tensor.staticValues.length !== 1) return null;
  const value = tensor.staticValues[0];
  return Number.isSafeInteger(value) ? value : null;
}
function dimensionsConcrete(dimensions) { return dimensions.every((dimension) => dimensionValue(dimension) != null); }
function multiplyDimensions(left, right) {
  const a = dimensionValue(left);
  const b = dimensionValue(right);
  return a != null && b != null ? valueDimension(a * b) : unknownDimension();
}
function valueDimension(value) { return { kind: "value", value, parameter: "", denotation: "", valueFieldCount: 1 }; }
function unknownDimension() { return { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 }; }
function cloneDimension(dimension) { return { ...dimension }; }
function dimensionValue(dimension) { return dimension?.kind === "value" && Number.isSafeInteger(dimension.value) ? dimension.value : null; }
function unresolved(reason) { return { outputs: [], reason }; }
