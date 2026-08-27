import { assessOnnxAttributeProto } from "./onnx-schema-legality.js";
import { makeOnnxTensorTypeFromDimensions, onnxShapeDimensionsFromValue } from "./onnx-type-proto.js";

const ATTRIBUTES = Object.freeze(new Map([
  ["alpha", 1],
  ["transA", 2],
  ["transB", 2],
  ["transBatchA", 2],
  ["transBatchB", 2],
]));

export function canInferOnnxOrtMatMulNode(node) {
  return normalizeDomain(node?.domain) === "com.microsoft" && node?.opType === "FusedMatMul";
}

export function assessOnnxOrtMatMulSchemaForm(node, importedOpset) {
  const base = {
    op_name: String(node?.opType || "UNKNOWN"),
    imported_opset: importedOpset,
    schema_since_version: 1,
    schema_source: "ort_contrib_fused_matmul_schema",
    input_count: node?.inputs?.length || 0,
    output_count: node?.outputs?.length || 0,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
  };
  if (!canInferOnnxOrtMatMulNode(node)) {
    return { ...base, status: "fail", reason_codes: ["ort_fused_matmul_shape_rule_not_registered"], detail: "The node is not the pinned com.microsoft:FusedMatMul-1 schema." };
  }
  const reasons = [];
  const unresolved = [];
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  if (inputs.length !== 2) reasons.push(`input_count_mismatch:${inputs.length}:2`);
  if (outputs.length !== 1) reasons.push(`output_count_mismatch:${outputs.length}:1`);
  if (!inputs[0]) reasons.push("required_input_omitted:0");
  if (!inputs[1]) reasons.push("required_input_omitted:1");
  if (!outputs[0]) reasons.push("required_output_omitted:0");
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
  for (const name of ["transA", "transB", "transBatchA", "transBatchB"]) {
    const value = attributeInt(node, name, 0);
    if (![0, 1].includes(value)) reasons.push(`attribute_value_invalid:${name}:0_or_1`);
  }
  return {
    ...base,
    schema_input_range: [2, 2],
    schema_output_range: [1, 1],
    schema_attribute_count: ATTRIBUTES.size,
    required_attributes: [],
    status: reasons.length ? "fail" : unresolved.length ? "unresolved" : "pass",
    reason_codes: reasons.length ? reasons : unresolved,
    detail: reasons.length
      ? "The NodeProto violates the pinned ORT FusedMatMul-1 contrib schema."
      : unresolved.length
        ? "The NodeProto attribute type cannot be proven from the parsed AttributeProto."
        : "The NodeProto matches the pinned ORT com.microsoft:FusedMatMul-1 contrib schema.",
  };
}

export function inferOnnxOrtMatMulNode(node, tensors) {
  const left = inputTensor(tensors, node.inputs?.[0]);
  const right = inputTensor(tensors, node.inputs?.[1]);
  if (!tensorRankKnown(left) || !tensorRankKnown(right)) return unresolved("ort_fused_matmul_input_rank_unknown");
  const leftRaw = onnxShapeDimensionsFromValue(left);
  const rightRaw = onnxShapeDimensionsFromValue(right);
  if (!leftRaw.length || !rightRaw.length) return unresolved("ort_fused_matmul_input_rank_zero");

  const leftShape = transformShape(leftRaw, attributeInt(node, "transA", 0) !== 0, attributeInt(node, "transBatchA", 0) !== 0);
  const rightShape = transformShape(rightRaw, attributeInt(node, "transB", 0) !== 0, attributeInt(node, "transBatchB", 0) !== 0);
  const leftVector = leftRaw.length === 1;
  const rightVector = rightRaw.length === 1;
  const promotedLeft = leftVector ? [valueDimension(1), cloneDimension(leftShape[0])] : leftShape;
  const promotedRight = rightVector ? [cloneDimension(rightShape[0]), valueDimension(1)] : rightShape;
  if (!contractionDimensionsCanMatch(promotedLeft.at(-1), promotedRight.at(-2))) return unresolved("ort_fused_matmul_contraction_dimension_mismatch");
  const prefix = broadcastManyDimensions([promotedLeft.slice(0, -2), promotedRight.slice(0, -2)]);
  if (!prefix) return unresolved("ort_fused_matmul_batch_broadcast_incompatible");
  const shape = [...prefix, cloneDimension(promotedLeft.at(-2)), cloneDimension(promotedRight.at(-1))];
  if (leftVector) shape.splice(shape.length - 2, 1);
  if (rightVector) shape.pop();
  const dtype = knownDtype(left);
  if (!dtype) return unresolved("ort_fused_matmul_input_dtype_unknown");
  const typeProto = makeOnnxTensorTypeFromDimensions(dtype, shape, true);
  return { outputs: [[node.outputs[0], { dtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto }]], reason: "" };
}

function transformShape(raw, transposeMatrix, transposeBatch) {
  if (raw.length === 1) return raw.map(cloneDimension);
  const rank = raw.length;
  const shape = [];
  const start = transposeBatch ? 1 : 0;
  const end = transposeBatch ? rank - 1 : rank - 2;
  for (let index = start; index < end; index += 1) shape.push(cloneDimension(raw[index]));
  shape.push(cloneDimension(raw[transposeMatrix ? rank - 1 : transposeBatch ? 0 : rank - 2]));
  shape.push(cloneDimension(raw[transposeMatrix ? transposeBatch ? 0 : rank - 2 : rank - 1]));
  return shape;
}

function broadcastManyDimensions(shapes) {
  const rank = Math.max(0, ...shapes.map((shape) => shape.length));
  const output = [];
  for (let offset = 1; offset <= rank; offset += 1) {
    const dimensions = shapes.map((shape) => shape.at(-offset) || valueDimension(1));
    const nonOne = dimensions.filter((dimension) => dimensionValue(dimension) !== 1);
    if (!nonOne.length) {
      output.unshift(valueDimension(1));
      continue;
    }
    const concrete = [...new Set(nonOne.map(dimensionValue).filter((value) => value != null))];
    if (concrete.length > 1) return null;
    if (concrete.length === 1) {
      output.unshift(valueDimension(concrete[0]));
      continue;
    }
    const symbols = [...new Set(nonOne.filter((dimension) => dimension?.kind === "symbolic").map((dimension) => dimension.parameter))];
    const hasUnknown = nonOne.some((dimension) => !dimensionKnown(dimension));
    output.unshift(!hasUnknown && symbols.length === 1 ? symbolicDimension(symbols[0]) : unknownDimension());
  }
  return output;
}

function inputTensor(tensors, name) {
  const tensor = tensors.get(name);
  if (tensor?.valueKind && tensor.valueKind !== "tensor") return { ...tensor, dtype: "UNKNOWN", shape: null, shapeDeclared: false };
  return tensor && tensor.shapeDeclared !== true ? { ...tensor, shape: null } : tensor;
}
function tensorRankKnown(tensor) { return tensor?.shapeDeclared === true && Array.isArray(tensor.shape); }
function knownDtype(tensor) { const value = String(tensor?.dtype || ""); return value && value !== "UNKNOWN" ? value : ""; }
function attributeInt(node, name, fallback) { const value = node.attributes?.get(name)?.i; return Number.isSafeInteger(value) ? value : fallback; }
function normalizeDomain(value) { const domain = String(value || "").trim(); return domain || "ai.onnx"; }
function valueDimension(value) { return { kind: "value", value, parameter: "", denotation: "", valueFieldCount: 1 }; }
function symbolicDimension(parameter) { return { kind: "symbolic", value: null, parameter, denotation: "", valueFieldCount: 1 }; }
function unknownDimension() { return { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 }; }
function cloneDimension(dimension) { return { ...dimension }; }
function dimensionValue(dimension) { return dimension?.kind === "value" && Number.isSafeInteger(dimension.value) ? dimension.value : null; }
function dimensionKnown(dimension) { return dimensionValue(dimension) != null && dimensionValue(dimension) >= 0 || dimension?.kind === "symbolic" && Boolean(dimension.parameter); }
function contractionDimensionsCanMatch(left, right) { const a = dimensionValue(left); const b = dimensionValue(right); return a == null || b == null || a === b; }
function unresolved(reason) { return { outputs: [], reason }; }
