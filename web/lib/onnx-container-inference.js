import {
  canonicalOnnxTypeProto,
  cloneOnnxTypeProto,
  makeOnnxOptionalType,
  makeOnnxSequenceType,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxTypeProtoKnown,
  onnxValueDescriptorFromType,
  unionOnnxTypeProtos,
} from "./onnx-type-proto.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const MAX_EXACT_SEQUENCE_ELEMENTS = 4_096;

export const ONNX_CONTAINER_VALUE_OPS = new Set([
  "ConcatFromSequence", "Optional", "OptionalGetElement", "OptionalHasElement", "SequenceAt",
  "SequenceConstruct", "SequenceEmpty", "SequenceErase", "SequenceInsert", "SequenceLength",
  "SplitToSequence",
]);

export const ONNX_CONTAINER_VALUE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  documents: Object.freeze([
    Object.freeze({ role: "current_sequence_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/defs.cc`, sha256: "a57eb7cd7d58b70f26c561aad318c7998a11ffcf7bf03675fcec1a7cc7efabaa" }),
    Object.freeze({ role: "historical_sequence_schema", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/old.cc`, sha256: "f0815ecb7ce2ba994826e521bcbedb012229025f46e80b80e1e7edbee9a13643" }),
    Object.freeze({ role: "split_to_sequence_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/utils.cc`, sha256: "7c3ec816bc676a5d7c0213ae1d4ec225446608cf87ab930e39987a6c087dfce8" }),
    Object.freeze({ role: "optional_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/optional/defs.cc`, sha256: "d9e8bacff27ab70539a9cd5c84646bb7b384e6116c3f529379814f4f88e9effb" }),
  ]),
});

export function inferOnnxContainerNode({ node, tensorMap, tensorTypeName, nodeIndex, importedOpset, scope = "main_graph" }) {
  const op = String(node?.opType || "");
  if (op === "Identity") {
    const input = tensorMap.get(node.inputs?.[0]);
    const type = onnxTypeProtoFromValue(input);
    if (!type || type.kind === "tensor") return null;
    if (!validIdentityNonDenseType(type)) {
      return finish(node, nodeIndex, importedOpset, scope, "fail", [], ["identity_non_dense_type_not_supported"]);
    }
    return finish(node, nodeIndex, importedOpset, scope, "pass", [[node.outputs?.[0], descriptor(type, copyState(input))]], []);
  }
  if (!ONNX_CONTAINER_VALUE_OPS.has(op)) return null;
  const input = (index) => tensorMap.get(node.inputs?.[index]);
  const type = (index) => onnxTypeProtoFromValue(input(index));
  const fail = (...reasons) => finish(node, nodeIndex, importedOpset, scope, "fail", [], reasons);

  if (op === "SequenceEmpty") {
    const dtypeId = attributeInteger(node, "dtype") ?? 1;
    const dtype = tensorTypeName(dtypeId);
    if (!dtype || dtype === "UNDEFINED" || String(dtype).startsWith("TYPE_")) return fail("sequence_empty_dtype_invalid");
    const outputType = makeOnnxSequenceType(makeOnnxTensorType(dtype));
    return finish(node, nodeIndex, importedOpset, scope, "pass", [[node.outputs[0], descriptor(outputType, sequenceState(0, []))]], []);
  }

  if (op === "SequenceConstruct") {
    const inputTypes = (node.inputs || []).filter(Boolean).map((_, index) => type(index));
    if (!inputTypes.length) return fail("sequence_construct_input_missing");
    if (inputTypes.some((item) => item?.kind !== "tensor")) return fail("sequence_construct_input_not_tensor");
    const union = unionOnnxTypeProtos(inputTypes);
    if (union.status !== "pass") return fail(`sequence_construct_${union.reason}`);
    const inventory = inputTypes.length <= MAX_EXACT_SEQUENCE_ELEMENTS ? inputTypes : null;
    const state = sequenceState(inputTypes.length, inventory);
    if (!inventory) state.sequenceElementInventoryStatus = "not_materialized_over_limit";
    return finish(node, nodeIndex, importedOpset, scope, "pass", [[node.outputs[0], descriptor(makeOnnxSequenceType(union.type), state)]], []);
  }

  if (["SequenceInsert", "SequenceErase", "SequenceAt", "SequenceLength", "ConcatFromSequence"].includes(op)) {
    const sequence = input(0);
    const sequenceType = type(0);
    if (!validTensorSequenceType(sequenceType)) return fail(`${snake(op)}_input_not_typed_tensor_sequence`);

    if (op === "SequenceLength") {
      const output = descriptor(makeOnnxTensorType("INT64", [], true));
      const length = exactSequenceLength(sequence);
      if (length != null) Object.assign(output, staticScalar(length, "SequenceLength"));
      return finish(node, nodeIndex, importedOpset, scope, length == null ? "partial" : "pass", [[node.outputs[0], output]], length == null ? ["sequence_length_runtime_unknown"] : []);
    }

    if (op === "SequenceInsert") {
      const tensorType = type(1);
      if (tensorType?.kind !== "tensor") return fail("sequence_insert_value_not_tensor");
      const union = unionOnnxTypeProtos([sequenceType.elementType, tensorType]);
      if (union.status !== "pass") return fail(`sequence_insert_${union.reason}`);
      const length = exactSequenceLength(sequence);
      const position = sequencePosition(node, tensorMap, 2, length, true);
      if (position.status === "fail") return fail(position.reason);
      const inventory = exactSequenceInventory(sequence);
      let outputInventory = null;
      if (inventory && position.value != null && inventory.length + 1 <= MAX_EXACT_SEQUENCE_ELEMENTS) {
        outputInventory = inventory.map(cloneOnnxTypeProto);
        outputInventory.splice(position.value, 0, cloneOnnxTypeProto(tensorType));
      }
      const state = sequenceState(length == null ? null : length + 1, outputInventory);
      const status = position.status === "partial" || length == null ? "partial" : "pass";
      return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], descriptor(makeOnnxSequenceType(union.type), state)]], status === "partial" ? [position.reason || "sequence_length_runtime_unknown"] : []);
    }

    if (op === "SequenceErase") {
      const length = exactSequenceLength(sequence);
      if (length === 0) return fail("sequence_erase_empty_sequence");
      const position = sequencePosition(node, tensorMap, 1, length, false);
      if (position.status === "fail") return fail(position.reason);
      const inventory = exactSequenceInventory(sequence);
      let outputInventory = null;
      if (inventory && position.value != null) {
        outputInventory = inventory.map(cloneOnnxTypeProto);
        outputInventory.splice(position.value, 1);
      }
      const state = sequenceState(length == null ? null : length - 1, outputInventory);
      const status = position.status === "partial" || length == null ? "partial" : "pass";
      return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], descriptor(sequenceType, state)]], status === "partial" ? [position.reason || "sequence_length_runtime_unknown"] : []);
    }

    if (op === "SequenceAt") {
      const length = exactSequenceLength(sequence);
      const position = sequencePosition(node, tensorMap, 1, length, false, true);
      if (position.status === "fail") return fail(position.reason);
      const inventory = exactSequenceInventory(sequence);
      const outputType = inventory && position.value != null ? inventory[position.value] : sequenceType.elementType;
      const status = position.status === "partial" ? "partial" : "pass";
      return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], descriptor(outputType)]], status === "partial" ? [position.reason] : []);
    }

    return inferConcatFromSequence(node, nodeIndex, importedOpset, scope, sequence, sequenceType, fail);
  }

  if (op === "SplitToSequence") return inferSplitToSequence(node, nodeIndex, importedOpset, scope, input(0), input(1), type(0), fail);

  if (op === "Optional") {
    const suppliedInputs = (node.inputs || []).filter(Boolean);
    let elementType = null;
    let presence = null;
    if (suppliedInputs.length === 1) {
      elementType = type(0);
      presence = true;
    } else if (suppliedInputs.length === 0) {
      elementType = cloneOnnxTypeProto(node.attributes?.get("type")?.typeProto);
      presence = false;
    }
    if (!elementType || !["tensor", "sequence"].includes(elementType.kind) || !onnxTypeProtoKnown(elementType)) {
      return fail("optional_element_type_missing_or_unsupported");
    }
    const state = { optionalPresenceStatus: "assessed_exact", optionalPresence: presence };
    return finish(node, nodeIndex, importedOpset, scope, "pass", [[node.outputs[0], descriptor(makeOnnxOptionalType(elementType), state)]], []);
  }

  if (op === "OptionalHasElement") {
    const suppliedInputs = (node.inputs || []).filter(Boolean);
    if (suppliedInputs.length > 1) return fail("optional_has_element_input_cardinality_invalid");
    let presence = false;
    let status = "pass";
    const reasons = [];
    if (suppliedInputs.length === 1) {
      const inputType = type(0);
      if (!validOptionalOperandType(inputType)) return fail("optional_has_element_input_type_invalid");
      if (inputType.kind === "optional") {
        presence = exactOptionalPresence(input(0));
        if (presence == null) {
          status = "partial";
          reasons.push("optional_presence_runtime_unknown");
        }
      } else presence = true;
    }
    const output = descriptor(makeOnnxTensorType("BOOL", [], true));
    if (presence != null) Object.assign(output, staticScalar(presence ? 1 : 0, "OptionalHasElement"));
    return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], output]], reasons);
  }

  if (op === "OptionalGetElement") {
    const inputType = type(0);
    if (!validOptionalOperandType(inputType)) return fail("optional_get_element_input_type_invalid");
    if (inputType.kind !== "optional") return finish(node, nodeIndex, importedOpset, scope, "pass", [[node.outputs[0], descriptor(inputType, copyState(input(0)))]], []);
    const presence = exactOptionalPresence(input(0));
    if (presence === false) return fail("optional_get_element_provably_empty");
    return finish(node, nodeIndex, importedOpset, scope, presence == null ? "partial" : "pass", [[node.outputs[0], descriptor(inputType.elementType)]], presence == null ? ["optional_presence_runtime_unknown"] : []);
  }

  return null;
}

function inferSplitToSequence(node, nodeIndex, importedOpset, scope, tensor, split, tensorType, fail) {
  if (tensorType?.kind !== "tensor" || !onnxTypeProtoKnown(tensorType)) return fail("split_to_sequence_input_not_typed_tensor");
  const rank = tensorType.shapeDeclared === true ? tensorType.shape.length : null;
  const rawAxis = attributeInteger(node, "axis") ?? 0;
  const axis = rank == null ? null : normalizeAxis(rawAxis, rank);
  if (rank != null && axis == null) return fail("split_to_sequence_axis_out_of_range");
  const splitName = node.inputs?.[1] || "";
  const splitValues = splitName ? exactIntegers(split) : null;
  if (splitName && split && !["INT32", "INT64"].includes(split.dtype)) return fail("split_to_sequence_split_dtype_invalid");
  if (splitName && split?.shapeDeclared && ![0, 1].includes(split.shape.length)) return fail("split_to_sequence_split_rank_invalid");
  if (splitValues?.some((value) => value <= 0) || splitValues?.length === 0) return fail("split_to_sequence_split_values_not_positive");

  const dimensions = tensorType.shapeDeclared === true ? tensorType.shapeDimensions.map((dimension) => ({ ...dimension })) : [];
  const axisExtent = axis == null ? null : concreteDimension(dimensions[axis]);
  let length = null;
  let inventory = null;
  let elementDimensions = dimensions.map((dimension) => ({ ...dimension }));
  let status = "pass";
  const reasons = [];
  if (!splitName) {
    const keepdims = attributeInteger(node, "keepdims") ?? 1;
    if (![0, 1].includes(keepdims)) return fail("split_to_sequence_keepdims_invalid");
    if (axis != null && keepdims === 0) elementDimensions.splice(axis, 1);
    else if (axis != null) elementDimensions[axis] = valueDimension(1);
    length = axisExtent;
    if (length == null) {
      status = "partial";
      reasons.push("split_to_sequence_axis_extent_unknown");
    } else if (length <= MAX_EXACT_SEQUENCE_ELEMENTS) {
      inventory = Array.from({ length }, () => tensorTypeWithDimensions(tensorType.dtype, elementDimensions));
    }
  } else if (!splitValues) {
    if (axis != null) elementDimensions[axis] = unknownDimension();
    status = "partial";
    reasons.push("split_to_sequence_split_values_runtime_unknown");
  } else if (split?.shapeDeclared && split.shape.length === 0) {
    const chunk = splitValues[0];
    if (axisExtent == null) {
      status = "partial";
      reasons.push("split_to_sequence_axis_extent_unknown");
      if (axis != null) elementDimensions[axis] = valueDimension(chunk);
    } else {
      length = Math.ceil(axisExtent / chunk);
      const chunks = Array.from({ length }, (_, index) => Math.min(chunk, axisExtent - index * chunk));
      if (axis != null) elementDimensions[axis] = chunks.every((value) => value === chunks[0]) ? valueDimension(chunks[0]) : unknownDimension();
      if (length <= MAX_EXACT_SEQUENCE_ELEMENTS) inventory = chunks.map((value) => {
        const itemDimensions = dimensions.map((dimension) => ({ ...dimension }));
        itemDimensions[axis] = valueDimension(value);
        return tensorTypeWithDimensions(tensorType.dtype, itemDimensions);
      });
    }
  } else {
    length = splitValues.length;
    if (axisExtent == null) {
      status = "partial";
      reasons.push("split_to_sequence_axis_extent_unknown");
    } else if (splitValues.reduce((sum, value) => sum + value, 0) !== axisExtent) return fail("split_to_sequence_split_sum_mismatch");
    if (axis != null) elementDimensions[axis] = splitValues.every((value) => value === splitValues[0]) ? valueDimension(splitValues[0]) : unknownDimension();
    if (length <= MAX_EXACT_SEQUENCE_ELEMENTS && axis != null) inventory = splitValues.map((value) => {
      const itemDimensions = dimensions.map((dimension) => ({ ...dimension }));
      itemDimensions[axis] = valueDimension(value);
      return tensorTypeWithDimensions(tensorType.dtype, itemDimensions);
    });
  }
  const elementType = tensorType.shapeDeclared === true ? tensorTypeWithDimensions(tensorType.dtype, elementDimensions) : makeOnnxTensorType(tensorType.dtype);
  return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], descriptor(makeOnnxSequenceType(elementType), sequenceState(length, inventory))]], reasons);
}

function inferConcatFromSequence(node, nodeIndex, importedOpset, scope, sequence, sequenceType, fail) {
  const elementType = sequenceType.elementType;
  if (elementType?.kind !== "tensor") return fail("concat_from_sequence_element_not_tensor");
  const length = exactSequenceLength(sequence);
  if (length === 0) return fail("concat_from_sequence_provably_empty");
  const inventory = exactSequenceInventory(sequence);
  if (inventory && length != null && inventory.length !== length) return fail("concat_from_sequence_inventory_length_mismatch");
  if (inventory?.some((type) => type?.kind !== "tensor")) return fail("concat_from_sequence_inventory_element_not_tensor");
  const inventoryDtypes = [...new Set((inventory || []).map((type) => type.dtype || type.elementTypeName || "UNKNOWN"))];
  if (inventoryDtypes.length > 1) return fail("concat_from_sequence_inventory_dtype_mismatch");
  const elementDtype = elementType.dtype || elementType.elementTypeName || "UNKNOWN";
  if (inventoryDtypes.length === 1 && inventoryDtypes[0] !== elementDtype) return fail("concat_from_sequence_inventory_aggregate_dtype_mismatch");
  const rawAxis = attributeInteger(node, "axis");
  if (rawAxis == null) return fail("concat_from_sequence_axis_missing");
  const newAxis = attributeInteger(node, "new_axis") ?? 0;
  if (![0, 1].includes(newAxis)) return fail("concat_from_sequence_new_axis_invalid");
  const rankSource = inventory?.find((type) => type?.shapeDeclared === true) || elementType;
  if (rankSource.shapeDeclared !== true) {
    return finish(node, nodeIndex, importedOpset, scope, "partial", [[node.outputs[0], descriptor(makeOnnxTensorType(elementType.dtype))]], ["concat_from_sequence_element_rank_unknown"]);
  }
  const inputRank = rankSource.shape.length;
  const outputRank = inputRank + newAxis;
  const axis = normalizeAxis(rawAxis, outputRank);
  if (axis == null) return fail("concat_from_sequence_axis_out_of_range");
  const outputDimensions = rankSource.shapeDimensions.map((dimension) => ({ ...dimension }));
  if (newAxis) outputDimensions.splice(axis, 0, unknownDimension());
  else outputDimensions[axis] = unknownDimension();
  let status = "partial";
  const reasons = ["concat_from_sequence_axis_extent_unknown"];
  if (inventory && length != null) {
    const exact = concatInventoryShape(inventory, axis, newAxis);
    if (exact.status === "fail") return fail(exact.reason);
    if (exact.dimensions) {
      outputDimensions.splice(0, outputDimensions.length, ...exact.dimensions);
      status = "pass";
      reasons.length = 0;
    }
  }
  const outputType = tensorTypeWithDimensions(elementDtype, outputDimensions);
  return finish(node, nodeIndex, importedOpset, scope, status, [[node.outputs[0], descriptor(outputType)]], reasons);
}

function concatInventoryShape(inventory, axis, newAxis) {
  if (!inventory.length) return { status: "fail", dimensions: null, reason: "concat_from_sequence_provably_empty" };
  if (inventory.some((type) => type?.kind !== "tensor" || type.shapeDeclared !== true)) return { status: "partial", dimensions: null, reason: "concat_from_sequence_inventory_shape_unknown" };
  const shapes = inventory.map((type) => type.shapeDimensions);
  const rank = shapes[0].length;
  if (shapes.some((shape) => shape.length !== rank)) return { status: "fail", reason: "concat_from_sequence_inventory_rank_mismatch" };
  if (newAxis) {
    for (let dimension = 0; dimension < rank; dimension += 1) {
      if (!shapes.every((shape) => sameDimension(shape[dimension], shapes[0][dimension]))) return { status: "fail", reason: "concat_from_sequence_stack_shape_mismatch" };
    }
    const output = shapes[0].map((dimension) => ({ ...dimension }));
    output.splice(axis, 0, valueDimension(shapes.length));
    return { status: "pass", dimensions: output };
  }
  for (let dimension = 0; dimension < rank; dimension += 1) {
    if (dimension !== axis && !shapes.every((shape) => sameDimension(shape[dimension], shapes[0][dimension]))) return { status: "fail", reason: "concat_from_sequence_non_axis_shape_mismatch" };
  }
  const axisValues = shapes.map((shape) => concreteDimension(shape[axis]));
  const output = shapes[0].map((dimension) => ({ ...dimension }));
  output[axis] = axisValues.every((value) => value != null) ? valueDimension(axisValues.reduce((sum, value) => sum + value, 0)) : unknownDimension();
  return { status: "pass", dimensions: output };
}

function finish(node, nodeIndex, importedOpset, scope, status, outputs, reasons) {
  const cleanOutputs = outputs.filter(([name, patch]) => Boolean(name) && Boolean(patch));
  const row = {
    scope,
    node_index: nodeIndex,
    op_name: node.opType,
    imported_opset: importedOpset,
    status,
    input_names: (node.inputs || []).filter(Boolean),
    output_names: (node.outputs || []).filter(Boolean),
    output_kinds: cleanOutputs.map(([, patch]) => patch.valueKind || "tensor"),
    canonical_output_types: cleanOutputs.map(([, patch]) => canonicalOnnxTypeProto(patch.typeProto || onnxTypeProtoFromValue(patch))),
    sequence_lengths: cleanOutputs.map(([, patch]) => patch.sequenceLengthStatus === "assessed_exact" ? patch.sequenceLength : null),
    optional_presence: cleanOutputs.map(([, patch]) => patch.optionalPresenceStatus === "assessed_exact" ? patch.optionalPresence : null),
    reason_codes: [...new Set(reasons.filter(Boolean))],
  };
  return { status, reason: row.reason_codes[0] || "", result: { outputs: cleanOutputs }, row };
}

function descriptor(type, state = {}) {
  return onnxValueDescriptorFromType(type, state);
}

function sequenceState(length, inventory) {
  return {
    sequenceLengthStatus: Number.isSafeInteger(length) && length >= 0 ? "assessed_exact" : "not_assessed_runtime_length",
    sequenceLength: Number.isSafeInteger(length) && length >= 0 ? length : null,
    sequenceElementInventoryStatus: Array.isArray(inventory) ? "assessed_exact" : "not_assessed",
    sequenceElementTypes: Array.isArray(inventory) ? inventory.map(cloneOnnxTypeProto) : [],
  };
}

function exactSequenceLength(value) {
  return value?.sequenceLengthStatus === "assessed_exact" && Number.isSafeInteger(value.sequenceLength) && value.sequenceLength >= 0 ? value.sequenceLength : null;
}

function exactSequenceInventory(value) {
  return value?.sequenceElementInventoryStatus === "assessed_exact" && Array.isArray(value.sequenceElementTypes)
    ? value.sequenceElementTypes.map(cloneOnnxTypeProto) : null;
}

function exactOptionalPresence(value) {
  return value?.optionalPresenceStatus === "assessed_exact" && typeof value.optionalPresence === "boolean" ? value.optionalPresence : null;
}

function validTensorSequenceType(type) {
  return type?.kind === "sequence" && type.elementType?.kind === "tensor" && onnxTypeProtoKnown(type.elementType);
}

function validOptionalOperandType(type) {
  if (type?.kind === "tensor") return onnxTypeProtoKnown(type);
  if (type?.kind === "sequence") return validTensorSequenceType(type);
  if (type?.kind !== "optional") return false;
  return type.elementType?.kind === "tensor"
    ? onnxTypeProtoKnown(type.elementType)
    : validTensorSequenceType(type.elementType);
}

function validIdentityNonDenseType(type) {
  return type?.kind === "sequence" ? validTensorSequenceType(type)
    : type?.kind === "optional" && validOptionalOperandType(type);
}

function copyState(value) {
  const state = {};
  for (const key of ["sequenceLengthStatus", "sequenceLength", "sequenceElementInventoryStatus", "optionalPresenceStatus", "optionalPresence"]) {
    if (value && key in value) state[key] = value[key];
  }
  if (Array.isArray(value?.sequenceElementTypes)) state.sequenceElementTypes = value.sequenceElementTypes.map(cloneOnnxTypeProto);
  return state;
}

function sequencePosition(node, tensorMap, inputIndex, length, insertion, required = false) {
  const name = node.inputs?.[inputIndex] || "";
  if (!name) {
    if (required) return { status: "fail", value: null, reason: "sequence_position_missing" };
    if (length == null) return { status: "partial", value: null, reason: "sequence_length_runtime_unknown" };
    return { status: "pass", value: insertion ? length : length - 1, reason: "" };
  }
  const positionTensor = tensorMap.get(name);
  if (positionTensor?.dtype && !["INT32", "INT64"].includes(positionTensor.dtype)) return { status: "fail", value: null, reason: "sequence_position_dtype_invalid" };
  if (positionTensor?.shapeDeclared === true && positionTensor.shape.length !== 0) return { status: "fail", value: null, reason: "sequence_position_not_scalar" };
  const values = exactIntegers(positionTensor);
  if (!values || values.length !== 1 || length == null) return { status: "partial", value: null, reason: values ? "sequence_length_runtime_unknown" : "sequence_position_runtime_unknown" };
  const raw = values[0];
  const minimum = -length;
  const maximum = insertion ? length : length - 1;
  if (raw < minimum || raw > maximum) return { status: "fail", value: null, reason: "sequence_position_out_of_bounds" };
  return { status: "pass", value: raw < 0 ? raw + length : raw, reason: "" };
}

function exactIntegers(tensor) {
  if (tensor?.staticValuesComplete !== true || !Array.isArray(tensor.staticValues)) return null;
  return tensor.staticValues.every(Number.isSafeInteger) ? tensor.staticValues.map(Number) : null;
}

function staticScalar(value, source) {
  return { staticValuesStatus: "assessed_exact_static_data", staticValuesComplete: true, staticValues: [value], staticValuesSource: source };
}

function tensorTypeWithDimensions(dtype, dimensions) {
  const type = makeOnnxTensorType(dtype, [], true);
  type.shapeDimensions = dimensions.map((dimension) => ({ ...dimension }));
  type.shape = type.shapeDimensions.map((dimension) => dimension.kind === "value" ? dimension.value : -1);
  return type;
}

function valueDimension(value) { return { kind: "value", value, parameter: "", denotation: "", valueFieldCount: 1 }; }
function unknownDimension() { return { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 }; }
function concreteDimension(dimension) { return dimension?.kind === "value" && Number.isSafeInteger(dimension.value) && dimension.value >= 0 ? dimension.value : null; }
function sameDimension(left, right) { return left?.kind === right?.kind && (left?.kind === "value" ? left.value === right.value : left?.kind === "symbolic" ? left.parameter === right.parameter : true); }

function normalizeAxis(axis, rank) {
  const value = Number(axis);
  if (!Number.isSafeInteger(value) || !Number.isSafeInteger(rank) || rank < 0 || value < -rank || value >= rank) return null;
  return value < 0 ? value + rank : value;
}

function attributeInteger(node, name) {
  const attribute = node.attributes?.get(name);
  if (Number.isSafeInteger(attribute?.i)) return attribute.i;
  if (/^-?\d+$/.test(attribute?.iExactDecimal || "")) {
    const value = Number(attribute.iExactDecimal);
    if (Number.isSafeInteger(value)) return value;
  }
  return null;
}

function snake(value) { return String(value || "").replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase(); }
