import { assessOnnxAttributeProto } from "./onnx-schema-legality.js";
import { makeOnnxTensorTypeFromDimensions, onnxShapeDimensionsFromValue } from "./onnx-type-proto.js";

const ATTRIBUTES = Object.freeze(new Map([
  ["direction", 3],
  ["hidden_size", 2],
  ["activation_alpha", 6],
  ["activation_beta", 6],
  ["clip", 1],
  ["activations", 8],
  ["input_forget", 2],
]));
const REQUIRED_INPUTS = Object.freeze([0, 1, 2, 8, 9, 10, 11]);

export function canInferOnnxOrtRecurrentNode(node) {
  return normalizeDomain(node?.domain) === "com.microsoft" && node?.opType === "DynamicQuantizeLSTM";
}

export function assessOnnxOrtRecurrentSchemaForm(node, importedOpset) {
  const base = {
    op_name: String(node?.opType || "UNKNOWN"),
    imported_opset: importedOpset,
    schema_since_version: 1,
    schema_source: "ort_contrib_quantized_recurrent_schema",
    input_count: node?.inputs?.length || 0,
    output_count: node?.outputs?.length || 0,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
  };
  if (!canInferOnnxOrtRecurrentNode(node)) {
    return { ...base, status: "fail", reason_codes: ["ort_recurrent_shape_rule_not_registered"], detail: "The node is not the pinned com.microsoft:DynamicQuantizeLSTM-1 schema." };
  }
  const reasons = [];
  const unresolved = [];
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  if (inputs.length !== 12) reasons.push(`input_count_mismatch:${inputs.length}:12`);
  if (outputs.length > 3) reasons.push(`output_count_above_maximum:${outputs.length}:3`);
  for (const index of REQUIRED_INPUTS) if (!inputs[index]) reasons.push(`required_input_omitted:${index}`);
  for (const name of new Set(node.duplicateAttributeNames || [])) reasons.push(`duplicate_attribute_name:${name}`);
  for (const [name, attribute] of node.attributes || []) {
    const expected = ATTRIBUTES.get(name);
    if (expected == null) {
      reasons.push(`attribute_not_defined:${name}`);
      continue;
    }
    const actual = assessOnnxAttributeProto(attribute);
    if (actual.status === "fail") reasons.push(`${actual.reason}:${name}`);
    else if (actual.status === "unresolved") unresolved.push(`${actual.reason}:${name}`);
    else if (actual.type !== expected) reasons.push(`attribute_type_mismatch:${name}:${expected}:${actual.type}`);
  }
  const direction = attributeString(node, "direction", "forward");
  if (!["forward", "reverse", "bidirectional"].includes(direction)) reasons.push("attribute_value_invalid:direction");
  const hiddenSize = attributeInt(node, "hidden_size", -1);
  if (node.attributes?.has("hidden_size") && hiddenSize <= 0) reasons.push("attribute_value_not_positive:hidden_size");
  const inputForget = attributeInt(node, "input_forget", 0);
  if (![0, 1].includes(inputForget)) reasons.push("attribute_value_invalid:input_forget:0_or_1");
  return {
    ...base,
    schema_input_range: [12, 12],
    schema_output_range: [0, 3],
    schema_attribute_count: ATTRIBUTES.size,
    required_attributes: [],
    status: reasons.length ? "fail" : unresolved.length ? "unresolved" : "pass",
    reason_codes: reasons.length ? reasons : unresolved,
    detail: reasons.length
      ? "The NodeProto violates the pinned ORT DynamicQuantizeLSTM-1 contrib schema."
      : unresolved.length
        ? "The NodeProto attribute type cannot be proven from the parsed AttributeProto."
        : "The NodeProto matches the pinned ORT com.microsoft:DynamicQuantizeLSTM-1 contrib schema.",
  };
}

export function inferOnnxOrtRecurrentNode(node, tensors) {
  const input = inputTensor(tensors, node.inputs?.[0]);
  if (!tensorRankKnown(input) || input.shape.length !== 3) return unresolved("ort_dynamic_quantize_lstm_input_rank_unknown_or_invalid");
  const dtype = knownDtype(input);
  if (!dtype) return unresolved("ort_dynamic_quantize_lstm_input_dtype_unknown");
  const inputShape = onnxShapeDimensionsFromValue(input);
  const direction = attributeString(node, "direction", "forward");
  const directionCount = direction === "bidirectional" ? valueDimension(2)
    : ["forward", "reverse"].includes(direction) ? valueDimension(1) : unknownDimension();
  const hiddenSizeValue = attributeInt(node, "hidden_size", -1);
  const hiddenSize = hiddenSizeValue > 0 ? valueDimension(hiddenSizeValue) : unknownDimension();
  const sequenceLength = cloneDimension(inputShape[0]);
  const batchSize = cloneDimension(inputShape[1]);
  const entries = [
    [0, [sequenceLength, directionCount, batchSize, hiddenSize]],
    [1, [directionCount, batchSize, hiddenSize]],
    [2, [directionCount, batchSize, hiddenSize]],
  ];
  const outputs = [];
  for (const [index, shape] of entries) {
    const name = node.outputs?.[index];
    if (!name) continue;
    const typeProto = makeOnnxTensorTypeFromDimensions(dtype, shape, true);
    outputs.push([name, { dtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto }]);
  }
  return outputs.length ? { outputs, reason: "" } : unresolved("ort_dynamic_quantize_lstm_output_omitted");
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
function valueDimension(value) { return { kind: "value", value, parameter: "", denotation: "", valueFieldCount: 1 }; }
function unknownDimension() { return { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 }; }
function cloneDimension(dimension) { return { ...dimension }; }
function unresolved(reason) { return { outputs: [], reason }; }
