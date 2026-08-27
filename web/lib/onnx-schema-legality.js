import {
  ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES,
  ONNX_SHAPE_SCHEMA_FORMS,
  ONNX_SHAPE_SCHEMA_SOURCE,
} from "./onnx-shape-schema-generated.js";

const ATTRIBUTE_TYPE_NAMES = Object.freeze({
  1: "FLOAT", 2: "INT", 3: "STRING", 4: "TENSOR", 5: "GRAPH",
  6: "FLOATS", 7: "INTS", 8: "STRINGS", 9: "TENSORS", 10: "GRAPHS",
  11: "SPARSE_TENSOR", 12: "SPARSE_TENSORS", 13: "TYPE_PROTO", 14: "TYPE_PROTOS",
});

export { ONNX_SHAPE_SCHEMA_SOURCE };

export function resolveOnnxSchemaSinceVersion(opName, importedOpset) {
  const forms = ONNX_SHAPE_SCHEMA_FORMS.get(opName) || [];
  return forms.filter((candidate) => candidate[0] <= importedOpset).at(-1)?.[0] ?? null;
}

export function assessOnnxAttributeProto(attribute, { allowReference = false } = {}) {
  if (attribute?.refAttrName && allowReference) {
    const declared = Number(attribute?.type || 0);
    return Number.isSafeInteger(declared) && declared >= 1 && declared <= 14
      ? { status: "pass", reason: "", type: declared, reference: attribute.refAttrName }
      : { status: "fail", reason: "attribute_reference_type_missing", type: null, reference: attribute.refAttrName };
  }
  return assessAttributeProto(attribute);
}

export function assessOnnxNodeSchemaForm(node, importedOpset) {
  const forms = ONNX_SHAPE_SCHEMA_FORMS.get(node?.opType) || [];
  const form = forms.filter((candidate) => candidate[0] <= importedOpset).at(-1) || null;
  const base = {
    op_name: node?.opType || "UNKNOWN",
    imported_opset: importedOpset,
    schema_since_version: form?.[0] ?? null,
    input_count: node?.inputs?.length || 0,
    output_count: node?.outputs?.length || 0,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
  };
  if (!form) {
    return {
      ...base,
      status: "fail",
      reason_codes: ["operator_not_defined_at_imported_opset"],
      detail: `${base.op_name} has no pinned standard-domain schema at opset ${importedOpset}.`,
    };
  }

  const [, inputMin, inputMax, inputOptions, outputMin, outputMax, outputOptions, encodedAttributes] = form;
  const reasons = [
    ...assessFormalList("input", node.inputs || [], inputMin, inputMax, inputOptions),
    ...assessFormalList("output", node.outputs || [], outputMin, outputMax, outputOptions),
  ];
  for (const name of new Set(node.duplicateAttributeNames || [])) reasons.push(`duplicate_attribute_name:${name}`);
  const unresolved = [];
  const allowed = new Map(encodedAttributes.map(([nameIndex, type, required]) => [
    ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES[nameIndex], { type, required: required === 1 },
  ]));
  for (const [name, attribute] of node.attributes || []) {
    const contract = allowed.get(name);
    if (!contract) {
      reasons.push(`attribute_not_defined:${name}`);
      continue;
    }
    const actual = assessAttributeProto(attribute);
    if (actual.status === "fail") reasons.push(`${actual.reason}:${name}`);
    else if (actual.status === "unresolved") unresolved.push(`${actual.reason}:${name}`);
    else if (actual.type !== contract.type) reasons.push(`attribute_type_mismatch:${name}:${ATTRIBUTE_TYPE_NAMES[contract.type]}:${ATTRIBUTE_TYPE_NAMES[actual.type] || actual.type}`);
  }
  for (const [name, contract] of allowed) {
    if (contract.required && !node.attributes?.has(name)) reasons.push(`required_attribute_missing:${name}`);
  }
  return {
    ...base,
    schema_input_range: [inputMin, inputOptions.includes("V") ? null : inputMax],
    schema_output_range: [outputMin, outputOptions.includes("V") ? null : outputMax],
    schema_input_options: inputOptions,
    schema_output_options: outputOptions,
    schema_attribute_count: allowed.size,
    required_attributes: [...allowed].filter(([, contract]) => contract.required).map(([name]) => name),
    status: reasons.length ? "fail" : unresolved.length ? "unresolved" : "pass",
    reason_codes: reasons.length ? reasons : unresolved,
    detail: reasons.length
      ? `The NodeProto violates the pinned ${base.op_name}-${form[0]} formal schema.`
      : unresolved.length
        ? `The NodeProto attribute type cannot be proven from the parsed AttributeProto.`
        : `The NodeProto matches the pinned ${base.op_name}-${form[0]} formal schema.`,
  };
}

function assessFormalList(label, values, min, max, options) {
  const reasons = [];
  const variadic = options.indexOf("V");
  if (values.length < min) reasons.push(`${label}_count_below_minimum:${values.length}:${min}`);
  if (variadic < 0 && values.length > max) reasons.push(`${label}_count_above_maximum:${values.length}:${max}`);
  for (let index = 0; index < values.length; index += 1) {
    const option = index < options.length ? options[index] : variadic >= 0 && index >= variadic ? "V" : "";
    if (!option) {
      reasons.push(`${label}_position_not_defined:${index}`);
      continue;
    }
    if (!values[index] && option !== "O") reasons.push(`required_${label}_omitted:${index}`);
  }
  for (let index = 0; index < options.length; index += 1) {
    if (options[index] === "R" && !values[index]) reasons.push(`required_${label}_omitted:${index}`);
  }
  return [...new Set(reasons)];
}

function assessAttributeProto(attribute) {
  if (attribute?.refAttrName) return { status: "fail", reason: "attribute_reference_not_allowed_in_main_graph", type: null };
  const declared = Number(attribute?.type || 0);
  const declaredType = Number.isSafeInteger(declared) && declared >= 1 && declared <= 14 ? declared : null;
  const present = Array.isArray(attribute?.valueTypesPresent)
    ? [...new Set(attribute.valueTypesPresent.filter((type) => Number.isSafeInteger(type) && type >= 1 && type <= 14))]
    : inferredAttributeValueTypes(attribute);
  if (present.length > 1) return { status: "fail", reason: `attribute_multiple_value_fields:${present.map((type) => ATTRIBUTE_TYPE_NAMES[type]).join("+")}`, type: declaredType };
  if (declaredType != null && present.length === 1 && present[0] !== declaredType) {
    return { status: "fail", reason: `attribute_discriminator_payload_mismatch:${ATTRIBUTE_TYPE_NAMES[declaredType]}:${ATTRIBUTE_TYPE_NAMES[present[0]]}`, type: declaredType };
  }
  const type = declaredType ?? present[0] ?? null;
  if (type == null) return { status: "unresolved", reason: "attribute_type_not_representable", type: null };
  if (!present.length && ![6, 7, 8, 9, 10, 12, 14].includes(type)) {
    return { status: "fail", reason: `attribute_value_missing:${ATTRIBUTE_TYPE_NAMES[type]}`, type };
  }
  return { status: "pass", reason: "", type };
}

function inferredAttributeValueTypes(attribute) {
  const types = [];
  if (attribute?.tensor) types.push(4);
  if (attribute?.graph) types.push(5);
  if (attribute?.tensors?.length) types.push(9);
  if (attribute?.graphs?.length) types.push(10);
  if (attribute?.iExactDecimal || Number.isSafeInteger(attribute?.i)) types.push(2);
  if (typeof attribute?.f === "number" && Number.isFinite(attribute.f)) types.push(1);
  if (attribute?.s != null) types.push(3);
  if (attribute?.intExactDecimals?.length || attribute?.ints?.length) types.push(7);
  if (attribute?.floats?.length) types.push(6);
  if (attribute?.strings?.length) types.push(8);
  return [...new Set(types)];
}
