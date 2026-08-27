import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

export const ONE_HOT_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;

const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64", "STRING"]);
const PINNED_ORT_CPU_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT64", "STRING"]);
const INT64_MIN = -(1n << 63n);
const INT64_LIMIT = 1n << 63n;

export function inferOnnxMlOneHotEncoder({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("onehot_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`onehot_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype === "UNKNOWN") reasons.push("onehot_input_dtype_unresolved");
  else if (!ALLOWED_DTYPES.has(inputDtype)) failures.push(`onehot_input_dtype_not_supported:${inputDtype}`);

  const intCategoriesAttribute = node.attributes?.get("cats_int64s");
  const stringCategoriesAttribute = node.attributes?.get("cats_strings");
  const zerosAttribute = node.attributes?.get("zeros");
  const intCategories = intCategoriesAttribute ? intListAttribute(intCategoriesAttribute) : [];
  const stringCategories = stringCategoriesAttribute ? stringListAttribute(stringCategoriesAttribute) : [];
  const zerosValue = zerosAttribute ? intScalarAttribute(zerosAttribute) : 1n;
  if (intCategoriesAttribute && intCategories == null) failures.push("onehot_cats_int64s_not_exact_int_list");
  if (stringCategoriesAttribute && stringCategories == null) failures.push("onehot_cats_strings_not_string_list");
  if (zerosAttribute && zerosValue == null) failures.push("onehot_zeros_not_exact_int_scalar");

  const contract = oneHotParameterContract(inputDtype, intCategories || [], stringCategories || []);
  if (contract.status !== "pass") reasons.push(contract.reason);
  const activeCategories = contract.categoryKind === "int64" ? intCategories || [] : stringCategories || [];
  const categoryLedger = duplicateCategoryLedger(activeCategories);
  const zerosEnabled = zerosValue == null ? null : zerosValue !== 0n;
  const zerosCanonical = zerosValue == null || zerosValue === 0n || zerosValue === 1n;

  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  if (shapeDeclared && inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("onehot_dynamic_input_shape_preserved");
  const outputShapeDeclared = shapeDeclared && contract.status === "pass";
  const outputShape = outputShapeDeclared ? [...inputShape, activeCategories.length] : [];
  const outputType = contract.status === "pass" ? makeOnnxTensorType("FLOAT32", outputShape, outputShapeDeclared) : null;
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactInputElements = shapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  const exactOutputElements = exactInputElements == null || contract.status !== "pass"
    ? null : safeProduct(exactInputElements, activeCategories.length);
  if (exactInputElements != null && contract.status === "pass" && exactOutputElements == null) reasons.push("onehot_output_element_count_overflow");

  const source = staticInput(input, inputDtype);
  let staticResult = unresolvedStaticResult(input);
  if (contract.status === "pass" && source && exactInputElements != null && zerosEnabled != null) {
    if (source.length !== exactInputElements) failures.push(`onehot_static_value_count_mismatch:${source.length}:${exactInputElements}`);
    else staticResult = evaluateOneHot(source, inputDtype, activeCategories, zerosEnabled, exactOutputElements);
  }
  if (staticResult.values && patch) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = staticResult.values;
    patch.staticValuesSource = "onehot_pinned_ort_cpu_semantics";
  }

  const riskCodes = [];
  if (contract.status === "fail") riskCodes.push("onehot_pinned_ort_attribute_contract_invalid");
  if (categoryLedger.duplicateCount > 0) riskCodes.push("onehot_duplicate_categories_last_write_wins");
  if (!zerosCanonical) riskCodes.push("onehot_noncanonical_zeros_boolean");
  if (staticResult.unknownCount > 0 && zerosEnabled === true) riskCodes.push("onehot_unknown_categories_all_zero_encoding");
  if (staticResult.guaranteedRuntimeFailure) riskCodes.push("onehot_unknown_category_guaranteed_runtime_failure");
  if (staticResult.invalidNumericCastCount > 0) riskCodes.push("onehot_numeric_to_int64_cast_not_representable");
  if (ALLOWED_DTYPES.has(inputDtype) && !PINNED_ORT_CPU_DTYPES.has(inputDtype)) {
    riskCodes.push("onehot_schema_dtype_missing_pinned_ort_cpu_kernel");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "OneHotEncoder", contract_kind: "tensor_encoder",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: shapeDeclared ? inputShape.length : null, input_shape: inputShape,
    exact_batch_count: null, exact_feature_count: null,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: contract.status === "pass" ? "FLOAT32" : "UNKNOWN",
    exact_output_rank: outputShapeDeclared ? outputShape.length : null, exact_output_shape: outputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_append_category_axis_and_ort_cpu_lookup",
    runtime_reference_status: "pinned_ort_cpu_int64_float_double_string_kernels",
    attribute_mode: "single_category_vocabulary_and_zeros_policy",
    onehot_parameter_contract_status: contract.status,
    onehot_parameter_contract_reason: contract.reason,
    onehot_category_kind: contract.categoryKind,
    onehot_category_count: activeCategories.length,
    onehot_category_values: activeCategories.map(exactValueText),
    onehot_duplicate_category_count: categoryLedger.duplicateCount,
    onehot_unreachable_duplicate_column_count: categoryLedger.unreachableColumnIndices.length,
    onehot_unreachable_duplicate_column_indices: categoryLedger.unreachableColumnIndices,
    onehot_zeros_value: zerosValue?.toString() ?? "unresolved",
    onehot_zeros_source: zerosAttribute ? "explicit_attribute" : "onnx_schema_default_1",
    onehot_zeros_enabled: zerosEnabled,
    onehot_zeros_canonical_boolean: zerosCanonical,
    onehot_static_assessment_status: staticResult.status,
    onehot_exact_input_value_count: staticResult.inputValueCount,
    onehot_exact_matched_input_count: staticResult.matchedCount,
    onehot_exact_unknown_input_count: staticResult.unknownCount,
    onehot_numeric_to_int64_changed_count: staticResult.numericCastChangedCount,
    onehot_numeric_to_int64_invalid_count: staticResult.invalidNumericCastCount,
    onehot_guaranteed_runtime_failure: staticResult.guaranteedRuntimeFailure,
    onehot_exact_output_one_count: staticResult.outputOneCount,
    onehot_exact_output_zero_count: staticResult.outputZeroCount,
    onehot_output_materialized: Boolean(staticResult.values),
    onehot_unknown_input_preview: staticResult.unknownPreview,
    onehot_output_preview: staticResult.outputPreview,
    vocabulary_type: contract.categoryKind === "string" ? "STRING" : contract.categoryKind === "int64" ? "INT64" : "UNDEFINED",
    vocabulary_count: activeCategories.length, duplicate_vocabulary_count: categoryLedger.duplicateCount,
    vocabulary_preview: activeCategories.slice(0, 8).map(exactValueText),
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null, exact_truncated_feature_count_per_batch: null,
    padded_input_count: 0, truncated_input_count: 0,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_values: [], exact_index_preview: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  const canPropagate = status !== "fail" && contract.status === "pass" && !staticResult.guaranteedRuntimeFailure;
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: canPropagate && patch && node.outputs?.[0] ? [[node.outputs[0], patch]] : [] }, row,
  };
}

function oneHotParameterContract(dtype, intCategories, stringCategories) {
  const intActive = intCategories.length > 0;
  const stringActive = stringCategories.length > 0;
  if (intActive === stringActive) {
    return { status: "fail", reason: "onehot_requires_exactly_one_nonempty_category_list", categoryKind: "invalid" };
  }
  const categoryKind = intActive ? "int64" : "string";
  if (dtype === "UNKNOWN") return { status: "not_assessed", reason: "onehot_input_dtype_unresolved", categoryKind };
  const expectedKind = dtype === "STRING" ? "string" : ALLOWED_DTYPES.has(dtype) ? "int64" : "invalid";
  if (categoryKind !== expectedKind) {
    return { status: "fail", reason: "onehot_category_list_type_does_not_match_input_dtype", categoryKind };
  }
  return { status: "pass", reason: "onehot_single_nonempty_category_list", categoryKind };
}

function evaluateOneHot(source, dtype, categories, zerosEnabled, outputElementCount) {
  const lookup = new Map();
  categories.forEach((value, index) => lookup.set(categoryKey(value), index));
  const materialize = outputElementCount != null && outputElementCount <= ONE_HOT_MAX_PROPAGATED_STATIC_VALUES;
  const output = materialize ? new Array(outputElementCount).fill(0) : null;
  const unknownPreview = [];
  let matchedCount = 0;
  let unknownCount = 0;
  let numericCastChangedCount = 0;
  let invalidNumericCastCount = 0;
  for (let index = 0; index < source.length; index += 1) {
    const converted = convertInputCategory(source[index], dtype);
    if (!converted.ok) {
      invalidNumericCastCount += 1;
      if (unknownPreview.length < 8) unknownPreview.push(exactValueText(source[index]));
      continue;
    }
    if (converted.changed) numericCastChangedCount += 1;
    const categoryIndex = lookup.get(categoryKey(converted.value));
    if (categoryIndex == null) {
      unknownCount += 1;
      if (unknownPreview.length < 8) unknownPreview.push(exactValueText(source[index]));
    } else {
      matchedCount += 1;
      if (output) output[index * categories.length + categoryIndex] = 1;
    }
  }
  const guaranteedRuntimeFailure = !zerosEnabled && unknownCount > 0;
  const validOutput = invalidNumericCastCount === 0 && !guaranteedRuntimeFailure;
  const values = validOutput ? output : null;
  const outputOneCount = validOutput ? matchedCount : null;
  const outputZeroCount = validOutput && outputElementCount != null ? outputElementCount - matchedCount : null;
  return {
    status: invalidNumericCastCount > 0 ? "assessed_unrepresentable_numeric_cast_not_propagated"
      : guaranteedRuntimeFailure ? "assessed_guaranteed_runtime_failure_unknown_category"
        : values ? "assessed_exact_pinned_ort_semantics"
          : "assessed_counts_output_not_materialized_limit",
    values,
    inputValueCount: source.length,
    matchedCount,
    unknownCount,
    numericCastChangedCount,
    invalidNumericCastCount,
    guaranteedRuntimeFailure,
    outputOneCount,
    outputZeroCount,
    unknownPreview,
    outputPreview: values ? values.slice(0, 16).map(String) : [],
  };
}

function convertInputCategory(value, dtype) {
  if (dtype === "STRING") return typeof value === "string" ? { ok: true, value, changed: false } : { ok: false };
  if (dtype === "INT64") {
    try { return { ok: true, value: typeof value === "bigint" ? value : BigInt(value), changed: false }; } catch { return { ok: false }; }
  }
  if (dtype === "INT32") {
    return Number.isSafeInteger(value) ? { ok: true, value: BigInt(value), changed: false } : { ok: false };
  }
  if (!["FLOAT32", "FLOAT64"].includes(dtype) || typeof value !== "number" || !Number.isFinite(value)) return { ok: false };
  if (value < Number(INT64_MIN) || value >= Number(INT64_LIMIT)) return { ok: false };
  const truncated = Math.trunc(value);
  try { return { ok: true, value: BigInt(truncated), changed: !Object.is(value, truncated) }; } catch { return { ok: false }; }
}

function duplicateCategoryLedger(categories) {
  const lastIndex = new Map();
  categories.forEach((value, index) => lastIndex.set(categoryKey(value), index));
  const unreachableColumnIndices = [];
  categories.forEach((value, index) => {
    if (lastIndex.get(categoryKey(value)) !== index) unreachableColumnIndices.push(index);
  });
  return { duplicateCount: categories.length - lastIndex.size, unreachableColumnIndices };
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializerIntegerValuesExactComplete === true
    && Array.isArray(input.initializerIntegerValuesExactDecimals)) {
    try { return input.initializerIntegerValuesExactDecimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (ALLOWED_DTYPES.has(dtype) && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) {
    return input.staticValues;
  }
  return null;
}

function unresolvedStaticResult(input) {
  return {
    status: input?.role === "initializer" ? input.staticValuesStatus || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null, inputValueCount: null, matchedCount: null, unknownCount: null,
    numericCastChangedCount: null, invalidNumericCastCount: null, guaranteedRuntimeFailure: false,
    outputOneCount: null, outputZeroCount: null, unknownPreview: [], outputPreview: [],
  };
}

function intListAttribute(attribute) {
  if (attribute?.type !== 7 || !Array.isArray(attribute.ints)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 7) return null;
  const exact = Array.isArray(attribute.intExactDecimals) ? attribute.intExactDecimals : [];
  const source = exact.length === attribute.ints.length ? exact : attribute.ints;
  try { return source.map((value) => BigInt(value)); } catch { return null; }
}

function stringListAttribute(attribute) {
  if (attribute?.type !== 8 || !Array.isArray(attribute.strings)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 8) return null;
  return [...attribute.strings];
}

function intScalarAttribute(attribute) {
  if (attribute?.type !== 2 || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== 2) return null;
  try { return BigInt(attribute.iExactDecimal || attribute.i); } catch { return null; }
}

function categoryKey(value) {
  return typeof value === "bigint" ? `i:${value}` : `s:${value}`;
}

function exactValueText(value) {
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function knownDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function safeShapeElementCount(shape) {
  let product = 1;
  for (const dimension of shape) {
    product = safeProduct(product, dimension);
    if (product == null) return null;
  }
  return product;
}

function safeProduct(left, right) {
  const value = left * right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}
