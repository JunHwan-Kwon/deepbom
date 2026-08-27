import { canonicalFloatText, numericArraysExactlyEqual, staticValuesWithSignedZeros } from "./onnx-static-value-evidence.js";

const MATERIALIZATION_LIMIT = 1_000_000;
const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const INT32_MIN = -2_147_483_648;
const INT32_L2_LIMIT = 46_340;
const INT64_MIN = -9_223_372_036_854_775_808n;
const INT64_L2_LIMIT = 3_037_000_499n;
const FLOAT32_LOWEST = -Math.fround(3.4028234663852886e38);

// Deliberately independent from onnx-ml-normalizer.js: exported evidence is
// reconstructed from public tensor facts so self-consistent row tampering fails.
export function validateNormalizerRowAgainstEvidence(row, tensors = []) {
  if (!row || row.op_name !== "Normalizer") return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const output = tensors.find((tensor) => tensor.name === row.output_name);
  const shape = Array.isArray(row.input_shape) ? row.input_shape : [];
  const rank = row.input_rank;
  const validRank = rank === 1 || rank === 2;
  const validMode = ["MAX", "L1", "L2"].includes(row.normalizer_mode);
  const knownShape = validRank && shape.every(knownDimension);
  const elements = knownShape ? safeShapeProduct(shape) : null;
  const batchCount = rank === 1 ? 1 : rank === 2 && knownDimension(shape[0]) ? shape[0] : null;
  const rowWidth = rank === 1 && knownDimension(shape[0]) ? shape[0]
    : rank === 2 && knownDimension(shape[1]) ? shape[1] : null;
  const source = staticInput(input, row.input_dtype);
  const canEvaluate = source && validRank && validMode && elements != null && source.length === elements;
  const reconstructed = canEvaluate
    ? reconstruct(source, row.input_dtype, shape, row.normalizer_mode)
    : unresolvedResult(input);
  const expectedRisks = [];
  if (reconstructed.overflowCount > 0) expectedRisks.push("normalizer_signed_integer_abs_or_square_overflow");
  if (reconstructed.negativeMaxRows > 0) expectedRisks.push("normalizer_negative_signed_max_divisor");
  if (reconstructed.integerRoundingCount > 0) expectedRisks.push("normalizer_integer_to_float32_precision_loss");
  if (reconstructed.nonfiniteCount > 0) expectedRisks.push("normalizer_non_finite_float32_projection");
  if (input?.static_values_status === "not_assessed_non_finite_or_unsafe_value"
    && ["FLOAT32", "FLOAT64"].includes(row.input_dtype)) {
    expectedRisks.push("normalizer_static_input_contains_non_finite_or_unsafe_value");
  }
  const outputStaticValues = staticValuesWithSignedZeros(output);
  const outputValuesMatch = reconstructed.values
    ? outputStaticValues != null
      && numericArraysExactlyEqual(outputStaticValues, reconstructed.values)
    : output?.static_values_complete !== true;
  const structurallyValid = ["tensor", "unresolved"].includes(row.input_kind)
    && ["FLOAT32", "FLOAT64", "INT32", "INT64", "UNKNOWN"].includes(row.input_dtype)
    && row.output_kind === "tensor" && row.output_dtype === "FLOAT32"
    && row.exact_output_rank === rank
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(shape)
    && row.exact_dense_output_element_count === elements
    && row.exact_batch_count === batchCount && row.exact_feature_count === rowWidth
    && row.normalizer_batch_count === batchCount && row.normalizer_row_width === rowWidth
    && row.output_shape_basis === "pinned_onnx_float_output_same_shape_and_ort_row_semantics"
    && row.runtime_reference_status === "pinned_ort_cpu_float32_output_kernel"
    && ["explicit_attribute", "onnx_schema_default_MAX"].includes(row.normalizer_mode_source)
    && (row.normalizer_mode_source !== "onnx_schema_default_MAX" || row.normalizer_mode === "MAX")
    && row.normalizer_divisor_kind === (row.normalizer_mode === "MAX" ? "signed_max"
      : row.normalizer_mode === "L1" ? "absolute_sum" : "square_sum_before_sqrt")
    && Array.isArray(row.normalizer_divisor_preview)
    && Array.isArray(row.normalizer_output_preview)
    && row.normalizer_divisor_preview.length <= 8
    && row.normalizer_output_preview.length <= 8;
  if (!structurallyValid) return false;
  if (row.status === "fail" || !ALLOWED_DTYPES.has(row.input_dtype) || !validRank || !validMode) {
    return row.status === "fail" && row.reason_codes.length > 0;
  }
  return row.normalizer_static_assessment_status === reconstructed.status
    && row.normalizer_exact_input_value_count === reconstructed.inputCount
    && row.normalizer_zero_divisor_row_count === reconstructed.zeroRows
    && row.normalizer_negative_max_divisor_row_count === reconstructed.negativeMaxRows
    && row.normalizer_integer_float32_rounding_count === reconstructed.integerRoundingCount
    && row.normalizer_signed_overflow_value_count === reconstructed.overflowCount
    && row.normalizer_non_finite_output_count === reconstructed.nonfiniteCount
    && row.normalizer_signed_zero_output_count === reconstructed.signedZeroCount
    && row.normalizer_output_materialized === Boolean(reconstructed.values)
    && JSON.stringify(row.normalizer_divisor_preview) === JSON.stringify(reconstructed.divisorPreview)
    && JSON.stringify(row.normalizer_output_preview) === JSON.stringify(reconstructed.outputPreview)
    && outputValuesMatch
    && JSON.stringify([...row.risk_codes].sort()) === JSON.stringify(expectedRisks.sort())
    && (reconstructed.overflowCount > 0)
      === row.reason_codes.includes("normalizer_static_output_not_assessed_signed_overflow");
}

function reconstruct(source, dtype, shape, mode) {
  const batchCount = shape.length === 1 ? 1 : shape[0];
  const rowWidth = shape.length === 1 ? shape[0] : shape[1];
  const overflowCount = countOverflow(source, dtype, mode);
  const integerRoundingCount = countIntegerRounding(source, dtype);
  if (overflowCount > 0) {
    return {
      status: "not_assessed_ort_signed_integer_overflow", values: null, inputCount: source.length,
      divisorPreview: [], outputPreview: [], zeroRows: null, negativeMaxRows: null,
      integerRoundingCount, overflowCount, nonfiniteCount: null, signedZeroCount: null,
    };
  }
  const values = source.length <= MATERIALIZATION_LIMIT ? new Array(source.length) : null;
  const divisorPreview = [];
  const outputPreview = [];
  let zeroRows = 0;
  let negativeMaxRows = 0;
  let nonfiniteCount = 0;
  let signedZeroCount = 0;
  for (let batch = 0; batch < batchCount; batch += 1) {
    const offset = batch * rowWidth;
    const divisor = divisorFor(source, dtype, offset, rowWidth, mode);
    if (divisorPreview.length < 8) divisorPreview.push(canonicalFloatText(divisor));
    if (divisor === 0) zeroRows += 1;
    if (mode === "MAX" && rowWidth > 0 && divisor < 0) negativeMaxRows += 1;
    for (let index = 0; index < rowWidth; index += 1) {
      const sourceIndex = offset + index;
      const inputFloat = toFloat32(source[sourceIndex]);
      let result;
      if (divisor === 0) result = inputFloat;
      else if (mode === "L2") {
        const quotient = Math.fround(squareToFloat32(source[sourceIndex], dtype) / divisor);
        const magnitude = Math.fround(Math.sqrt(quotient));
        result = source[sourceIndex] < 0 ? Math.fround(-magnitude) : magnitude;
      } else result = Math.fround(inputFloat / divisor);
      if (values) values[sourceIndex] = result;
      if (outputPreview.length < 8) outputPreview.push(canonicalFloatText(result));
      if (!Number.isFinite(result)) nonfiniteCount += 1;
      if (Object.is(result, -0)) signedZeroCount += 1;
    }
  }
  const safeValues = values && nonfiniteCount === 0 ? values : null;
  return {
    status: safeValues ? "assessed_pinned_ort_float32"
      : nonfiniteCount > 0 ? "assessed_non_finite_output_not_propagated"
        : "assessed_counts_output_not_materialized_limit",
    values: safeValues, inputCount: source.length, divisorPreview, outputPreview,
    zeroRows, negativeMaxRows, integerRoundingCount, overflowCount, nonfiniteCount, signedZeroCount,
  };
}

function divisorFor(source, dtype, offset, width, mode) {
  if (mode === "MAX") {
    let maximum = FLOAT32_LOWEST;
    for (let index = 0; index < width; index += 1) {
      const candidate = toFloat32(source[offset + index]);
      if (candidate > maximum) maximum = candidate;
    }
    return maximum;
  }
  let sum = 0;
  for (let index = 0; index < width; index += 1) {
    const value = source[offset + index];
    const term = mode === "L1" ? toFloat32(integerAbs(value)) : squareToFloat32(value, dtype);
    sum = Math.fround(sum + term);
  }
  return sum;
}

function countOverflow(source, dtype, mode) {
  if (!mode.startsWith("L") || !["INT32", "INT64"].includes(dtype)) return 0;
  return source.reduce((count, value) => {
    if (mode === "L1") return count + ((dtype === "INT64" ? value === INT64_MIN : value === INT32_MIN) ? 1 : 0);
    const magnitude = typeof value === "bigint" ? (value < 0n ? -value : value) : Math.abs(value);
    return count + ((dtype === "INT64" ? magnitude > INT64_L2_LIMIT : magnitude > INT32_L2_LIMIT) ? 1 : 0);
  }, 0);
}

function countIntegerRounding(source, dtype) {
  if (!["INT32", "INT64"].includes(dtype)) return 0;
  return source.reduce((count, value) => {
    const rounded = toFloat32(value);
    return count + ((typeof value === "bigint" ? BigInt(rounded) !== value : rounded !== value) ? 1 : 0);
  }, 0);
}

function squareToFloat32(value, dtype) {
  if (dtype === "INT64") return bigintToFloat32(value * value);
  if (dtype === "INT32") return Math.fround(value * value);
  const numeric = dtype === "FLOAT32" ? Math.fround(value) : Number(value);
  return Math.fround(numeric * numeric);
}

function integerAbs(value) {
  return typeof value === "bigint" ? (value < 0n ? -value : value) : Math.abs(value);
}

function toFloat32(value) {
  return typeof value === "bigint" ? bigintToFloat32(value) : Math.fround(Number(value));
}

function bigintToFloat32(value) {
  if (value === 0n) return 0;
  const negative = value < 0n;
  const magnitude = negative ? -value : value;
  let exponent = magnitude.toString(2).length - 1;
  if (exponent <= 23) return Math.fround(Number(value));
  const shift = BigInt(exponent - 23);
  let significand = magnitude >> shift;
  const remainder = magnitude - (significand << shift);
  const half = 1n << (shift - 1n);
  if (remainder > half || remainder === half && (significand & 1n) === 1n) significand += 1n;
  if (significand === (1n << 24n)) {
    significand >>= 1n;
    exponent += 1;
  }
  const rounded = Number(significand) * (2 ** (exponent - 23));
  return Math.fround(negative ? -rounded : rounded);
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializer_integer_values_exact_complete === true
    && Array.isArray(input.initializer_integer_values_exact_decimals)) {
    try {
      return input.initializer_integer_values_exact_decimals.map((value) => BigInt(value));
    } catch {
      return null;
    }
  }
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.static_values_complete === true) return staticValuesWithSignedZeros(input);
  return null;
}

function unresolvedResult(input) {
  return {
    status: input?.role === "initializer" ? input.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null, inputCount: null, divisorPreview: [], outputPreview: [], zeroRows: null,
    negativeMaxRows: null, integerRoundingCount: null, overflowCount: null, nonfiniteCount: null, signedZeroCount: null,
  };
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
