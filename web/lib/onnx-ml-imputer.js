import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

export const IMPUTER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;

const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const PINNED_ORT_CPU_DTYPES = new Set(["FLOAT32", "INT64"]);

export function inferOnnxMlImputer({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("imputer_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`imputer_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype === "UNKNOWN") reasons.push("imputer_input_dtype_unresolved");
  else if (!ALLOWED_DTYPES.has(inputDtype)) failures.push(`imputer_input_dtype_not_supported:${inputDtype}`);

  const floatValuesAttribute = node.attributes?.get("imputed_value_floats");
  const intValuesAttribute = node.attributes?.get("imputed_value_int64s");
  const replacedFloatAttribute = node.attributes?.get("replaced_value_float");
  const replacedIntAttribute = node.attributes?.get("replaced_value_int64");
  const floatValues = floatValuesAttribute ? floatListAttribute(floatValuesAttribute) : [];
  const intValues = intValuesAttribute ? intListAttribute(intValuesAttribute) : [];
  const replacedFloat = replacedFloatAttribute ? floatScalarAttribute(replacedFloatAttribute) : 0;
  const replacedInt = replacedIntAttribute ? intScalarAttribute(replacedIntAttribute) : 0n;
  if (floatValuesAttribute && floatValues == null) failures.push("imputer_imputed_value_floats_not_float_list");
  if (intValuesAttribute && intValues == null) failures.push("imputer_imputed_value_int64s_not_exact_int_list");
  if (replacedFloatAttribute && replacedFloat == null) failures.push("imputer_replaced_value_float_not_float_scalar");
  if (replacedIntAttribute && replacedInt == null) failures.push("imputer_replaced_value_int64_not_exact_int_scalar");

  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  const rank = shapeDeclared ? inputShape.length : null;
  const featureStride = imputerFeatureStride(inputShape, shapeDeclared);
  const parameterContract = imputerParameterContract({
    dtype: inputDtype,
    floatValues: floatValues || [],
    intValues: intValues || [],
    rank,
    featureStride,
  });
  if (parameterContract.status !== "pass") reasons.push(parameterContract.reason);
  if (shapeDeclared && inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("imputer_dynamic_shape_preserved");

  const outputType = inputType?.kind === "tensor" ? makeOnnxTensorType(inputDtype, inputShape, shapeDeclared) : null;
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = shapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  if (shapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("imputer_output_element_count_overflow");

  const activeValues = parameterContract.attributeKind === "float" ? floatValues || [] : intValues || [];
  const replacedValue = parameterContract.attributeKind === "float" ? replacedFloat : replacedInt;
  const source = staticNumericInput(input, inputDtype);
  let staticResult = unresolvedStaticResult(input);
  if (parameterContract.status === "pass" && source && exactOutputElements != null) {
    if (source.length !== exactOutputElements) failures.push(`imputer_static_value_count_mismatch:${source.length}:${exactOutputElements}`);
    else staticResult = evaluateImputer(source, inputDtype, activeValues, replacedValue, featureStride);
  }
  if (staticResult.values && patch) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = staticResult.values;
    patch.staticValuesSource = "imputer_pinned_ort_cpu_semantics";
  }

  const nonFiniteImputedCount = parameterContract.attributeKind === "float"
    ? activeValues.filter((value) => !Number.isFinite(value)).length : 0;
  const ignoredImputedValueCount = parameterContract.mode === "scalar_first_fallback"
    ? Math.max(0, activeValues.length - 1) : 0;
  const riskCodes = [];
  if (parameterContract.status === "fail") riskCodes.push("imputer_pinned_ort_attribute_or_shape_contract_invalid");
  if (parameterContract.mode === "scalar_first_fallback") riskCodes.push("imputer_attribute_length_outside_onnx_one_or_feature_count");
  if (ALLOWED_DTYPES.has(inputDtype) && !PINNED_ORT_CPU_DTYPES.has(inputDtype)) {
    riskCodes.push("imputer_schema_dtype_missing_pinned_ort_cpu_kernel");
  }
  if (nonFiniteImputedCount > 0 || staticResult.nonFiniteOutputCount > 0) {
    riskCodes.push("imputer_non_finite_imputed_or_output");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "Imputer", contract_kind: "tensor_imputation",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: rank, input_shape: inputShape,
    exact_batch_count: rank === 1 ? 1 : rank != null && rank > 1 && knownDimension(inputShape[0]) ? inputShape[0] : null,
    exact_feature_count: featureStride,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: inputDtype,
    exact_output_rank: rank, exact_output_shape: inputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_same_dtype_same_shape_and_ort_second_dimension_feature_stride",
    runtime_reference_status: "pinned_ort_cpu_imputer_kernel",
    attribute_mode: "replace_equal_or_nan_marker",
    imputer_parameter_contract_status: parameterContract.status,
    imputer_parameter_contract_reason: parameterContract.reason,
    imputer_parameter_mode: parameterContract.mode,
    imputer_attribute_kind: parameterContract.attributeKind,
    imputer_feature_stride: featureStride,
    imputer_imputed_value_count: activeValues.length,
    imputer_imputed_values: activeValues.map(exactValueText),
    imputer_replaced_value: exactValueText(replacedValue),
    imputer_replaced_value_source: parameterContract.attributeKind === "float"
      ? replacedFloatAttribute ? "explicit_attribute" : "onnx_schema_default_0"
      : replacedIntAttribute ? "explicit_attribute" : "onnx_schema_default_0",
    imputer_ignored_imputed_value_count: ignoredImputedValueCount,
    imputer_non_finite_imputed_value_count: nonFiniteImputedCount,
    imputer_static_assessment_status: staticResult.status,
    imputer_exact_input_value_count: staticResult.inputValueCount,
    imputer_exact_replacement_count: staticResult.replacementCount,
    imputer_exact_nan_replacement_count: staticResult.nanReplacementCount,
    imputer_exact_unchanged_count: staticResult.unchangedCount,
    imputer_non_finite_output_count: staticResult.nonFiniteOutputCount,
    imputer_signed_zero_output_count: staticResult.signedZeroOutputCount,
    imputer_output_materialized: Boolean(staticResult.values),
    imputer_output_preview: staticResult.outputPreview,
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
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
  const canPropagate = status !== "fail" && parameterContract.status === "pass";
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: canPropagate && patch && node.outputs?.[0] ? [[node.outputs[0], patch]] : [] }, row,
  };
}

function imputerFeatureStride(shape, shapeDeclared) {
  if (!shapeDeclared || shape.length === 0) return null;
  const candidate = shape.length === 1 ? shape[0] : shape[1];
  return knownDimension(candidate) ? candidate : null;
}

function imputerParameterContract({ dtype, floatValues, intValues, rank, featureStride }) {
  const floatActive = floatValues.length > 0;
  const intActive = intValues.length > 0;
  if (floatActive === intActive) {
    return { status: "fail", reason: "imputer_pinned_ort_requires_exactly_one_nonempty_imputed_value_list", mode: "invalid", attributeKind: "invalid" };
  }
  const attributeKind = floatActive ? "float" : "int64";
  const expectedKind = ["FLOAT32", "FLOAT64"].includes(dtype) ? "float"
    : ["INT32", "INT64"].includes(dtype) ? "int64" : "invalid";
  if (attributeKind !== expectedKind) {
    return { status: "fail", reason: "imputer_imputed_value_type_does_not_match_input_dtype", mode: "invalid", attributeKind };
  }
  if (rank === 0) return { status: "fail", reason: "imputer_pinned_ort_rejects_rank_zero_input", mode: "invalid", attributeKind };
  const count = floatActive ? floatValues.length : intValues.length;
  if (rank == null || featureStride == null) {
    if (count === 1) return { status: "pass", reason: "imputer_scalar_value", mode: "scalar", attributeKind };
    return { status: "not_assessed", reason: "imputer_feature_stride_unresolved", mode: "unresolved", attributeKind };
  }
  if (featureStride != null && count === featureStride) {
    return { status: "pass", reason: "imputer_per_feature_values", mode: "per_feature", attributeKind };
  }
  if (count === 1) return { status: "pass", reason: "imputer_scalar_value", mode: "scalar", attributeKind };
  return { status: "pass", reason: "imputer_pinned_ort_scalar_first_fallback", mode: "scalar_first_fallback", attributeKind };
}

function evaluateImputer(source, dtype, imputedValues, replacedValue, stride) {
  const materialize = source.length <= IMPUTER_MAX_PROPAGATED_STATIC_VALUES;
  const output = materialize ? new Array(source.length) : null;
  const outputPreview = [];
  let replacementCount = 0;
  let nanReplacementCount = 0;
  let nonFiniteOutputCount = 0;
  let signedZeroOutputCount = 0;
  let unsafeInt64Output = false;
  const perFeature = imputedValues.length === stride;
  for (let index = 0; index < source.length; index += 1) {
    const inputValue = source[index];
    const nanMatch = typeof inputValue === "number" && typeof replacedValue === "number"
      && Number.isNaN(inputValue) && Number.isNaN(replacedValue);
    const match = nanMatch || inputValue === replacedValue;
    const value = match ? imputedValues[perFeature ? index % stride : 0] : inputValue;
    if (match) replacementCount += 1;
    if (nanMatch) nanReplacementCount += 1;
    if (typeof value === "number" && !Number.isFinite(value)) nonFiniteOutputCount += 1;
    if (typeof value === "number" && Object.is(value, -0)) signedZeroOutputCount += 1;
    if (dtype === "INT64" && typeof value === "bigint"
      && (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER))) unsafeInt64Output = true;
    if (output) output[index] = typeof value === "bigint" ? Number(value) : value;
    if (outputPreview.length < 8) outputPreview.push(exactValueText(value));
  }
  const safeValues = output && nonFiniteOutputCount === 0 && !unsafeInt64Output ? output : null;
  return {
    status: safeValues ? "assessed_exact_pinned_ort_semantics"
      : nonFiniteOutputCount > 0 ? "assessed_non_finite_output_not_propagated"
        : unsafeInt64Output ? "assessed_exact_int64_output_not_materialized_unsafe_json_integer"
          : "assessed_counts_output_not_materialized_limit",
    values: safeValues,
    inputValueCount: source.length,
    replacementCount,
    nanReplacementCount,
    unchangedCount: source.length - replacementCount,
    nonFiniteOutputCount,
    signedZeroOutputCount,
    outputPreview,
  };
}

function staticNumericInput(input, dtype) {
  if (dtype === "INT64" && input?.initializerIntegerValuesExactComplete === true
    && Array.isArray(input.initializerIntegerValuesExactDecimals)) {
    try {
      return input.initializerIntegerValuesExactDecimals.map((value) => BigInt(value));
    } catch {
      return null;
    }
  }
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) return input.staticValues;
  return null;
}

function unresolvedStaticResult(input) {
  return {
    status: input?.role === "initializer" ? input.staticValuesStatus || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null,
    inputValueCount: null,
    replacementCount: null,
    nanReplacementCount: null,
    unchangedCount: null,
    nonFiniteOutputCount: null,
    signedZeroOutputCount: null,
    outputPreview: [],
  };
}

function floatListAttribute(attribute) {
  if (attribute?.type !== 6 || !Array.isArray(attribute.floats)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 6) return null;
  return attribute.floats.map((value) => Math.fround(value));
}

function floatScalarAttribute(attribute) {
  if (attribute?.type !== 1 || typeof attribute.f !== "number"
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 1) return null;
  return Math.fround(attribute.f);
}

function intListAttribute(attribute) {
  if (attribute?.type !== 7 || !Array.isArray(attribute.ints)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 7) return null;
  const exact = Array.isArray(attribute.intExactDecimals) ? attribute.intExactDecimals : [];
  const source = exact.length === attribute.ints.length ? exact : attribute.ints;
  try {
    return source.map((value) => {
      if (typeof value === "number" && !Number.isSafeInteger(value)) throw new Error("unsafe integer");
      return BigInt(value);
    });
  } catch {
    return null;
  }
}

function intScalarAttribute(attribute) {
  if (attribute?.type !== 2 || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== 2) return null;
  const source = String(attribute.iExactDecimal || "") || attribute.i;
  try {
    if (typeof source === "number" && !Number.isSafeInteger(source)) throw new Error("unsafe integer");
    return BigInt(source);
  } catch {
    return null;
  }
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
    product *= dimension;
    if (!Number.isSafeInteger(product) || product < 0) return null;
  }
  return product;
}
