import {
  numericArraysExactlyEqual,
  parseCanonicalFloatText,
  staticValuesWithSignedZeros,
} from "./onnx-static-value-evidence.js";

const MATERIALIZATION_LIMIT = 1_000_000;
const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const PINNED_ORT_CPU_DTYPES = new Set(["FLOAT32", "INT64"]);

export function validateImputerRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || row.op_name !== "Imputer") return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const output = tensors.find((tensor) => tensor.name === row.output_name);
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === "Imputer" && candidate.domain === "ai.onnx.ml");
  const floatValues = publicFloatListAttribute(op, "imputed_value_floats");
  const intValues = publicIntListAttribute(op, "imputed_value_int64s");
  const replacedFloat = publicFloatScalarAttribute(op, "replaced_value_float", 0);
  const replacedInt = publicIntScalarAttribute(op, "replaced_value_int64", 0n);
  if (!floatValues.ok || !intValues.ok || !replacedFloat.ok || !replacedInt.ok) {
    return row.status === "fail" && row.reason_codes.length > 0;
  }
  const shape = Array.isArray(row.input_shape) ? row.input_shape : [];
  const shapeDeclared = Number.isSafeInteger(row.input_rank) && row.input_rank >= 0;
  const featureStride = shapeDeclared && shape.length > 0
    ? knownDimension(shape.length === 1 ? shape[0] : shape[1]) ? (shape.length === 1 ? shape[0] : shape[1]) : null
    : null;
  const elements = shapeDeclared && shape.every(knownDimension) ? safeShapeProduct(shape) : null;
  const contract = parameterContract(row.input_dtype, floatValues.values, intValues.values, row.input_rank, featureStride);
  const activeValues = contract.attributeKind === "float" ? floatValues.values : intValues.values;
  const replacement = contract.attributeKind === "float" ? replacedFloat.value : replacedInt.value;
  const source = staticInput(input, row.input_dtype);
  const reconstructed = contract.status === "pass" && source && elements != null && source.length === elements
    ? reconstruct(source, row.input_dtype, activeValues, replacement, featureStride)
    : unresolvedResult(input);
  const nonfiniteImputed = contract.attributeKind === "float"
    ? activeValues.filter((value) => !Number.isFinite(value)).length : 0;
  const ignoredValues = contract.mode === "scalar_first_fallback" ? Math.max(0, activeValues.length - 1) : 0;
  const expectedRisks = [];
  if (contract.status === "fail") expectedRisks.push("imputer_pinned_ort_attribute_or_shape_contract_invalid");
  if (contract.mode === "scalar_first_fallback") expectedRisks.push("imputer_attribute_length_outside_onnx_one_or_feature_count");
  if (ALLOWED_DTYPES.has(row.input_dtype) && !PINNED_ORT_CPU_DTYPES.has(row.input_dtype)) {
    expectedRisks.push("imputer_schema_dtype_missing_pinned_ort_cpu_kernel");
  }
  if (nonfiniteImputed > 0 || reconstructed.nonfiniteCount > 0) expectedRisks.push("imputer_non_finite_imputed_or_output");
  const outputStatic = staticValuesWithSignedZeros(output);
  const outputMatches = reconstructed.values
    ? outputStatic != null && numericArraysExactlyEqual(outputStatic, reconstructed.values)
    : output?.static_values_complete !== true;
  return Boolean(op)
    && row.input_kind === "tensor" && ALLOWED_DTYPES.has(row.input_dtype)
    && row.output_kind === "tensor" && row.output_dtype === row.input_dtype
    && row.exact_output_rank === row.input_rank
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(shape)
    && row.exact_dense_output_element_count === elements
    && row.output_shape_basis === "pinned_onnx_same_dtype_same_shape_and_ort_second_dimension_feature_stride"
    && row.runtime_reference_status === "pinned_ort_cpu_imputer_kernel"
    && row.attribute_mode === "replace_equal_or_nan_marker"
    && row.imputer_parameter_contract_status === contract.status
    && row.imputer_parameter_contract_reason === contract.reason
    && row.imputer_parameter_mode === contract.mode
    && row.imputer_attribute_kind === contract.attributeKind
    && row.imputer_feature_stride === featureStride
    && row.imputer_imputed_value_count === activeValues.length
    && JSON.stringify(row.imputer_imputed_values) === JSON.stringify(activeValues.map(exactText))
    && row.imputer_replaced_value === exactText(replacement)
    && row.imputer_replaced_value_source === (contract.attributeKind === "float"
      ? replacedFloat.present ? "explicit_attribute" : "onnx_schema_default_0"
      : replacedInt.present ? "explicit_attribute" : "onnx_schema_default_0")
    && row.imputer_ignored_imputed_value_count === ignoredValues
    && row.imputer_non_finite_imputed_value_count === nonfiniteImputed
    && row.imputer_static_assessment_status === reconstructed.status
    && row.imputer_exact_input_value_count === reconstructed.inputCount
    && row.imputer_exact_replacement_count === reconstructed.replacementCount
    && row.imputer_exact_nan_replacement_count === reconstructed.nanReplacementCount
    && row.imputer_exact_unchanged_count === reconstructed.unchangedCount
    && row.imputer_non_finite_output_count === reconstructed.nonfiniteCount
    && row.imputer_signed_zero_output_count === reconstructed.signedZeroCount
    && row.imputer_output_materialized === Boolean(reconstructed.values)
    && JSON.stringify(row.imputer_output_preview) === JSON.stringify(reconstructed.preview)
    && outputMatches
    && JSON.stringify([...row.risk_codes].sort()) === JSON.stringify(expectedRisks.sort());
}

function parameterContract(dtype, floatValues, intValues, rank, stride) {
  const floatActive = floatValues.length > 0;
  const intActive = intValues.length > 0;
  if (floatActive === intActive) return { status: "fail", reason: "imputer_pinned_ort_requires_exactly_one_nonempty_imputed_value_list", mode: "invalid", attributeKind: "invalid" };
  const attributeKind = floatActive ? "float" : "int64";
  const expectedKind = ["FLOAT32", "FLOAT64"].includes(dtype) ? "float"
    : ["INT32", "INT64"].includes(dtype) ? "int64" : "invalid";
  if (attributeKind !== expectedKind) return { status: "fail", reason: "imputer_imputed_value_type_does_not_match_input_dtype", mode: "invalid", attributeKind };
  if (rank === 0) return { status: "fail", reason: "imputer_pinned_ort_rejects_rank_zero_input", mode: "invalid", attributeKind };
  const count = activeLength(floatValues, intValues);
  if (rank == null || stride == null) {
    if (count === 1) return { status: "pass", reason: "imputer_scalar_value", mode: "scalar", attributeKind };
    return { status: "not_assessed", reason: "imputer_feature_stride_unresolved", mode: "unresolved", attributeKind };
  }
  if (count === stride) return { status: "pass", reason: "imputer_per_feature_values", mode: "per_feature", attributeKind };
  if (count === 1) return { status: "pass", reason: "imputer_scalar_value", mode: "scalar", attributeKind };
  return { status: "pass", reason: "imputer_pinned_ort_scalar_first_fallback", mode: "scalar_first_fallback", attributeKind };
}

function reconstruct(source, dtype, imputedValues, replacement, stride) {
  const output = source.length <= MATERIALIZATION_LIMIT ? new Array(source.length) : null;
  const preview = [];
  let replacementCount = 0;
  let nanReplacementCount = 0;
  let nonfiniteCount = 0;
  let signedZeroCount = 0;
  let unsafeInt64 = false;
  const perFeature = imputedValues.length === stride;
  for (let index = 0; index < source.length; index += 1) {
    const inputValue = source[index];
    const nanMatch = typeof inputValue === "number" && typeof replacement === "number"
      && Number.isNaN(inputValue) && Number.isNaN(replacement);
    const match = nanMatch || inputValue === replacement;
    const value = match ? imputedValues[perFeature ? index % stride : 0] : inputValue;
    if (match) replacementCount += 1;
    if (nanMatch) nanReplacementCount += 1;
    if (typeof value === "number" && !Number.isFinite(value)) nonfiniteCount += 1;
    if (typeof value === "number" && Object.is(value, -0)) signedZeroCount += 1;
    if (typeof value === "bigint" && (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER))) unsafeInt64 = true;
    if (output) output[index] = typeof value === "bigint" ? Number(value) : value;
    if (preview.length < 8) preview.push(exactText(value));
  }
  const values = output && nonfiniteCount === 0 && !unsafeInt64 ? output : null;
  return {
    status: values ? "assessed_exact_pinned_ort_semantics"
      : nonfiniteCount > 0 ? "assessed_non_finite_output_not_propagated"
        : unsafeInt64 ? "assessed_exact_int64_output_not_materialized_unsafe_json_integer"
          : "assessed_counts_output_not_materialized_limit",
    values, inputCount: source.length, replacementCount, nanReplacementCount,
    unchangedCount: source.length - replacementCount, nonfiniteCount, signedZeroCount, preview,
  };
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializer_integer_values_exact_complete === true
    && Array.isArray(input.initializer_integer_values_exact_decimals)) {
    try { return input.initializer_integer_values_exact_decimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)) return staticValuesWithSignedZeros(input);
  return null;
}

function unresolvedResult(input) {
  return {
    status: input?.initializer_storage_kind ? input.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null, inputCount: null, replacementCount: null, nanReplacementCount: null,
    unchangedCount: null, nonfiniteCount: null, signedZeroCount: null, preview: [],
  };
}

function publicFloatListAttribute(op, name) {
  const attribute = findAttribute(op, name);
  if (!attribute) return { ok: true, present: false, values: [] };
  if (attribute.type !== 6 || !Array.isArray(attribute.float_values) || !Array.isArray(attribute.float_values_text)
    || attribute.float_values.length !== attribute.float_values_text.length) return { ok: false, present: true, values: [] };
  const values = [];
  for (let index = 0; index < attribute.float_values_text.length; index += 1) {
    const parsed = parseCanonicalFloatText(attribute.float_values_text[index], { float32: true });
    if (!parsed.ok || Number.isFinite(parsed.value) && attribute.float_values[index] !== parsed.value
      || !Number.isFinite(parsed.value) && attribute.float_values[index] !== null) return { ok: false, present: true, values: [] };
    values.push(parsed.value);
  }
  return { ok: true, present: true, values };
}

function publicIntListAttribute(op, name) {
  const attribute = findAttribute(op, name);
  if (!attribute) return { ok: true, present: false, values: [] };
  if (attribute.type !== 7 || !Array.isArray(attribute.int_values_exact_decimal)) return { ok: false, present: true, values: [] };
  try { return { ok: true, present: true, values: attribute.int_values_exact_decimal.map((value) => BigInt(value)) }; }
  catch { return { ok: false, present: true, values: [] }; }
}

function publicFloatScalarAttribute(op, name, defaultValue) {
  const attribute = findAttribute(op, name);
  if (!attribute) return { ok: true, present: false, value: defaultValue };
  if (attribute.type !== 1) return { ok: false, present: true, value: defaultValue };
  const parsed = parseCanonicalFloatText(attribute.float_value_text, { float32: true });
  if (!parsed.ok || Number.isFinite(parsed.value) && attribute.float_value !== parsed.value
    || !Number.isFinite(parsed.value) && attribute.float_value !== null) return { ok: false, present: true, value: defaultValue };
  return { ok: true, present: true, value: parsed.value };
}

function publicIntScalarAttribute(op, name, defaultValue) {
  const attribute = findAttribute(op, name);
  if (!attribute) return { ok: true, present: false, value: defaultValue };
  if (attribute.type !== 2) return { ok: false, present: true, value: defaultValue };
  try { return { ok: true, present: true, value: BigInt(attribute.int_value_exact_decimal) }; }
  catch { return { ok: false, present: true, value: defaultValue }; }
}

function findAttribute(op, name) {
  return (op?.onnx_attributes || []).find((attribute) => attribute.name === name);
}

function activeLength(floatValues, intValues) {
  return floatValues.length || intValues.length;
}

function exactText(value) {
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

function safeShapeProduct(shape) {
  let product = 1;
  for (const dimension of shape) {
    product *= dimension;
    if (!Number.isSafeInteger(product) || product < 0) return null;
  }
  return product;
}
