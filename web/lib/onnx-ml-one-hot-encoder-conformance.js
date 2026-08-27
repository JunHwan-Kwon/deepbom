import { numericArraysExactlyEqual, staticValuesWithSignedZeros } from "./onnx-static-value-evidence.js";

const LIMIT = 1_000_000;
const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64", "STRING"]);
const PINNED_ORT_CPU_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT64", "STRING"]);
const INT64_MIN = -(1n << 63n);
const INT64_LIMIT = 1n << 63n;

export function validateOneHotEncoderRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || row.op_name !== "OneHotEncoder") return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const output = tensors.find((tensor) => tensor.name === row.output_name);
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === "OneHotEncoder" && candidate.domain === "ai.onnx.ml");
  const intCategories = publicIntListAttribute(op, "cats_int64s");
  const stringCategories = publicStringListAttribute(op, "cats_strings");
  const zeros = publicIntScalarAttribute(op, "zeros", 1n);
  if (!intCategories.ok || !stringCategories.ok || !zeros.ok) {
    return row.status === "fail" && row.reason_codes.length > 0;
  }
  const contract = parameterContract(row.input_dtype, intCategories.values, stringCategories.values);
  const categories = contract.categoryKind === "int64" ? intCategories.values : stringCategories.values;
  const duplicateLedger = duplicates(categories);
  const shape = Array.isArray(row.input_shape) ? row.input_shape : [];
  const shapeDeclared = Number.isSafeInteger(row.input_rank) && row.input_rank >= 0;
  const inputElements = shapeDeclared && shape.every(knownDimension) ? shapeProduct(shape) : null;
  const outputElements = inputElements == null || contract.status !== "pass" ? null : safeProduct(inputElements, categories.length);
  const outputShape = shapeDeclared && contract.status === "pass" ? [...shape, categories.length] : [];
  const zerosEnabled = zeros.value !== 0n;
  const source = staticInput(input, row.input_dtype);
  const reconstructed = contract.status === "pass" && source && inputElements != null && source.length === inputElements
    ? reconstruct(source, row.input_dtype, categories, zerosEnabled, outputElements)
    : unresolved(input);
  const expectedRisks = [];
  if (contract.status === "fail") expectedRisks.push("onehot_pinned_ort_attribute_contract_invalid");
  if (duplicateLedger.indices.length > 0) expectedRisks.push("onehot_duplicate_categories_last_write_wins");
  if (zeros.value !== 0n && zeros.value !== 1n) expectedRisks.push("onehot_noncanonical_zeros_boolean");
  if (reconstructed.unknownCount > 0 && zerosEnabled) expectedRisks.push("onehot_unknown_categories_all_zero_encoding");
  if (reconstructed.guaranteedRuntimeFailure) expectedRisks.push("onehot_unknown_category_guaranteed_runtime_failure");
  if (reconstructed.invalidCastCount > 0) expectedRisks.push("onehot_numeric_to_int64_cast_not_representable");
  if (ALLOWED_DTYPES.has(row.input_dtype) && !PINNED_ORT_CPU_DTYPES.has(row.input_dtype)) {
    expectedRisks.push("onehot_schema_dtype_missing_pinned_ort_cpu_kernel");
  }
  const outputStatic = output?.static_values_complete === true ? output.static_values : null;
  const outputMatches = reconstructed.values
    ? numericArraysExactlyEqual(outputStatic, reconstructed.values)
    : output?.static_values_complete !== true;
  return Boolean(op)
    && row.input_kind === "tensor" && ALLOWED_DTYPES.has(row.input_dtype)
    && row.output_kind === "tensor" && row.output_dtype === (contract.status === "pass" ? "FLOAT32" : "UNKNOWN")
    && row.exact_output_rank === (shapeDeclared && contract.status === "pass" ? outputShape.length : null)
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputShape)
    && row.exact_dense_output_element_count === outputElements
    && row.output_shape_basis === "pinned_onnx_append_category_axis_and_ort_cpu_lookup"
    && row.runtime_reference_status === "pinned_ort_cpu_int64_float_double_string_kernels"
    && row.attribute_mode === "single_category_vocabulary_and_zeros_policy"
    && row.onehot_parameter_contract_status === contract.status
    && row.onehot_parameter_contract_reason === contract.reason
    && row.onehot_category_kind === contract.categoryKind
    && row.onehot_category_count === categories.length
    && JSON.stringify(row.onehot_category_values) === JSON.stringify(categories.map(textValue))
    && row.onehot_duplicate_category_count === duplicateLedger.indices.length
    && row.onehot_unreachable_duplicate_column_count === duplicateLedger.indices.length
    && JSON.stringify(row.onehot_unreachable_duplicate_column_indices) === JSON.stringify(duplicateLedger.indices)
    && row.onehot_zeros_value === zeros.value.toString()
    && row.onehot_zeros_source === (zeros.present ? "explicit_attribute" : "onnx_schema_default_1")
    && row.onehot_zeros_enabled === zerosEnabled
    && row.onehot_zeros_canonical_boolean === (zeros.value === 0n || zeros.value === 1n)
    && row.onehot_static_assessment_status === reconstructed.status
    && row.onehot_exact_input_value_count === reconstructed.inputCount
    && row.onehot_exact_matched_input_count === reconstructed.matchedCount
    && row.onehot_exact_unknown_input_count === reconstructed.unknownCount
    && row.onehot_numeric_to_int64_changed_count === reconstructed.changedCastCount
    && row.onehot_numeric_to_int64_invalid_count === reconstructed.invalidCastCount
    && row.onehot_guaranteed_runtime_failure === reconstructed.guaranteedRuntimeFailure
    && row.onehot_exact_output_one_count === reconstructed.oneCount
    && row.onehot_exact_output_zero_count === reconstructed.zeroCount
    && row.onehot_output_materialized === Boolean(reconstructed.values)
    && JSON.stringify(row.onehot_unknown_input_preview) === JSON.stringify(reconstructed.unknownPreview)
    && JSON.stringify(row.onehot_output_preview) === JSON.stringify(reconstructed.outputPreview)
    && outputMatches
    && JSON.stringify([...row.risk_codes].sort()) === JSON.stringify(expectedRisks.sort());
}

function parameterContract(dtype, ints, strings) {
  const intActive = ints.length > 0;
  const stringActive = strings.length > 0;
  if (intActive === stringActive) return { status: "fail", reason: "onehot_requires_exactly_one_nonempty_category_list", categoryKind: "invalid" };
  const categoryKind = intActive ? "int64" : "string";
  if (dtype === "UNKNOWN") return { status: "not_assessed", reason: "onehot_input_dtype_unresolved", categoryKind };
  const expected = dtype === "STRING" ? "string" : ALLOWED_DTYPES.has(dtype) ? "int64" : "invalid";
  if (categoryKind !== expected) return { status: "fail", reason: "onehot_category_list_type_does_not_match_input_dtype", categoryKind };
  return { status: "pass", reason: "onehot_single_nonempty_category_list", categoryKind };
}

function reconstruct(source, dtype, categories, zerosEnabled, outputElements) {
  const lookup = new Map();
  categories.forEach((value, index) => lookup.set(key(value), index));
  const output = outputElements != null && outputElements <= LIMIT ? new Array(outputElements).fill(0) : null;
  const unknownPreview = [];
  let matchedCount = 0;
  let unknownCount = 0;
  let changedCastCount = 0;
  let invalidCastCount = 0;
  source.forEach((value, index) => {
    const converted = convert(value, dtype);
    if (!converted.ok) {
      invalidCastCount += 1;
      if (unknownPreview.length < 8) unknownPreview.push(textValue(value));
      return;
    }
    if (converted.changed) changedCastCount += 1;
    const categoryIndex = lookup.get(key(converted.value));
    if (categoryIndex == null) {
      unknownCount += 1;
      if (unknownPreview.length < 8) unknownPreview.push(textValue(value));
    } else {
      matchedCount += 1;
      if (output) output[index * categories.length + categoryIndex] = 1;
    }
  });
  const guaranteedRuntimeFailure = !zerosEnabled && unknownCount > 0;
  const valid = invalidCastCount === 0 && !guaranteedRuntimeFailure;
  const values = valid ? output : null;
  return {
    status: invalidCastCount > 0 ? "assessed_unrepresentable_numeric_cast_not_propagated"
      : guaranteedRuntimeFailure ? "assessed_guaranteed_runtime_failure_unknown_category"
        : values ? "assessed_exact_pinned_ort_semantics" : "assessed_counts_output_not_materialized_limit",
    values, inputCount: source.length, matchedCount, unknownCount, changedCastCount, invalidCastCount,
    guaranteedRuntimeFailure, oneCount: valid ? matchedCount : null,
    zeroCount: valid && outputElements != null ? outputElements - matchedCount : null,
    unknownPreview, outputPreview: values ? values.slice(0, 16).map(String) : [],
  };
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializer_integer_values_exact_complete === true
    && Array.isArray(input.initializer_integer_values_exact_decimals)) {
    try { return input.initializer_integer_values_exact_decimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (dtype === "STRING") {
    return input?.static_values_complete === true && Array.isArray(input.static_values)
      && input.static_values.every((value) => typeof value === "string") ? [...input.static_values] : null;
  }
  return staticValuesWithSignedZeros(input);
}

function unresolved(input) {
  return {
    status: input?.initializer_storage_kind ? input.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null, inputCount: null, matchedCount: null, unknownCount: null, changedCastCount: null,
    invalidCastCount: null, guaranteedRuntimeFailure: false, oneCount: null, zeroCount: null,
    unknownPreview: [], outputPreview: [],
  };
}

function publicIntListAttribute(op, name) {
  const attribute = attr(op, name);
  if (!attribute) return { ok: true, present: false, values: [] };
  if (attribute.type !== 7 || !Array.isArray(attribute.int_values_exact_decimal)) return { ok: false, present: true, values: [] };
  try { return { ok: true, present: true, values: attribute.int_values_exact_decimal.map((value) => BigInt(value)) }; }
  catch { return { ok: false, present: true, values: [] }; }
}

function publicStringListAttribute(op, name) {
  const attribute = attr(op, name);
  return !attribute ? { ok: true, present: false, values: [] }
    : attribute.type === 8 && Array.isArray(attribute.string_values)
      ? { ok: true, present: true, values: [...attribute.string_values] }
      : { ok: false, present: true, values: [] };
}

function publicIntScalarAttribute(op, name, fallback) {
  const attribute = attr(op, name);
  if (!attribute) return { ok: true, present: false, value: fallback };
  if (attribute.type !== 2) return { ok: false, present: true, value: fallback };
  try { return { ok: true, present: true, value: BigInt(attribute.int_value_exact_decimal) }; }
  catch { return { ok: false, present: true, value: fallback }; }
}

function attr(op, name) {
  return (op?.onnx_attributes || []).find((attribute) => attribute.name === name);
}

function duplicates(categories) {
  const last = new Map();
  categories.forEach((value, index) => last.set(key(value), index));
  const indices = [];
  categories.forEach((value, index) => { if (last.get(key(value)) !== index) indices.push(index); });
  return { indices };
}

function convert(value, dtype) {
  if (dtype === "STRING") return typeof value === "string" ? { ok: true, value, changed: false } : { ok: false };
  if (dtype === "INT64") {
    try { return { ok: true, value: typeof value === "bigint" ? value : BigInt(value), changed: false }; } catch { return { ok: false }; }
  }
  if (dtype === "INT32") return Number.isSafeInteger(value) ? { ok: true, value: BigInt(value), changed: false } : { ok: false };
  if (!["FLOAT32", "FLOAT64"].includes(dtype) || typeof value !== "number" || !Number.isFinite(value)
    || value < Number(INT64_MIN) || value >= Number(INT64_LIMIT)) return { ok: false };
  const truncated = Math.trunc(value);
  try { return { ok: true, value: BigInt(truncated), changed: !Object.is(value, truncated) }; } catch { return { ok: false }; }
}

function key(value) {
  return typeof value === "bigint" ? `i:${value}` : `s:${value}`;
}

function textValue(value) {
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

function shapeProduct(shape) {
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
