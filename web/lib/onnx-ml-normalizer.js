import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

export const NORMALIZER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;

const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const MODES = new Set(["MAX", "L1", "L2"]);
const INT32_MIN = -2_147_483_648;
const INT32_L2_LIMIT = 46_340;
const INT64_MIN = -9_223_372_036_854_775_808n;
const INT64_L2_LIMIT = 3_037_000_499n;
const FLOAT32_LOWEST = -Math.fround(3.4028234663852886e38);

export function inferOnnxMlNormalizer({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("normalizer_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`normalizer_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype === "UNKNOWN") reasons.push("normalizer_input_dtype_unresolved");
  else if (!ALLOWED_DTYPES.has(inputDtype)) failures.push(`normalizer_input_dtype_not_supported:${inputDtype}`);

  const normAttribute = node.attributes?.get("norm");
  const explicitMode = normAttribute ? stringScalarAttribute(normAttribute) : null;
  if (normAttribute && explicitMode == null) failures.push("normalizer_norm_not_string_scalar");
  const mode = explicitMode ?? "MAX";
  if (explicitMode != null && !MODES.has(explicitMode)) failures.push(`normalizer_norm_invalid:${explicitMode}`);

  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  const rank = shapeDeclared ? inputShape.length : null;
  if (!shapeDeclared) reasons.push("normalizer_input_shape_unresolved");
  else if (![1, 2].includes(rank)) failures.push(`normalizer_input_rank_not_1_or_2:${rank}`);
  else if (inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("normalizer_dynamic_shape_preserved");

  const outputType = inputType?.kind === "tensor" ? makeOnnxTensorType("FLOAT32", inputShape, shapeDeclared) : null;
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = shapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  const derivedBatchCount = rank === 1 ? 1 : rank === 2 && knownDimension(inputShape[0]) ? inputShape[0] : null;
  const derivedRowWidth = rank === 1 && knownDimension(inputShape[0]) ? inputShape[0]
    : rank === 2 && knownDimension(inputShape[1]) ? inputShape[1] : null;
  if (shapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("normalizer_output_element_count_overflow");

  const source = staticNumericInput(input, inputDtype);
  let staticResult = unresolvedStaticResult(input);
  if (source && [1, 2].includes(rank) && exactOutputElements != null && MODES.has(mode)) {
    if (source.length !== exactOutputElements) {
      failures.push(`normalizer_static_value_count_mismatch:${source.length}:${exactOutputElements}`);
    } else {
      staticResult = evaluateNormalizer(source, inputDtype, inputShape, mode);
      if (staticResult.overflowValueCount > 0) reasons.push("normalizer_static_output_not_assessed_signed_overflow");
    }
  }

  if (staticResult.values && patch) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = staticResult.values;
    patch.staticValuesSource = "normalizer_pinned_ort_float32_emulation";
  }
  const riskCodes = [];
  if (staticResult.overflowValueCount > 0) riskCodes.push("normalizer_signed_integer_abs_or_square_overflow");
  if (staticResult.negativeMaxDivisorRowCount > 0) riskCodes.push("normalizer_negative_signed_max_divisor");
  if (staticResult.integerFloat32RoundingCount > 0) riskCodes.push("normalizer_integer_to_float32_precision_loss");
  if (staticResult.nonFiniteOutputCount > 0) riskCodes.push("normalizer_non_finite_float32_projection");
  if (input?.staticValuesStatus === "not_assessed_non_finite_or_unsafe_value" && ["FLOAT32", "FLOAT64"].includes(inputDtype)) {
    riskCodes.push("normalizer_static_input_contains_non_finite_or_unsafe_value");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "Normalizer", contract_kind: "tensor_normalization",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: rank, input_shape: inputShape,
    exact_batch_count: derivedBatchCount, exact_feature_count: derivedRowWidth,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: "FLOAT32",
    exact_output_rank: rank, exact_output_shape: inputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_float_output_same_shape_and_ort_row_semantics",
    runtime_reference_status: "pinned_ort_cpu_float32_output_kernel",
    attribute_mode: "normalization_mode",
    normalizer_mode: mode,
    normalizer_mode_source: normAttribute ? "explicit_attribute" : "onnx_schema_default_MAX",
    normalizer_static_assessment_status: staticResult.status,
    normalizer_exact_input_value_count: staticResult.inputValueCount,
    normalizer_batch_count: derivedBatchCount,
    normalizer_row_width: derivedRowWidth,
    normalizer_divisor_kind: mode === "MAX" ? "signed_max" : mode === "L1" ? "absolute_sum" : "square_sum_before_sqrt",
    normalizer_divisor_preview: staticResult.divisorPreview,
    normalizer_zero_divisor_row_count: staticResult.zeroDivisorRowCount,
    normalizer_negative_max_divisor_row_count: staticResult.negativeMaxDivisorRowCount,
    normalizer_integer_float32_rounding_count: staticResult.integerFloat32RoundingCount,
    normalizer_signed_overflow_value_count: staticResult.overflowValueCount,
    normalizer_non_finite_output_count: staticResult.nonFiniteOutputCount,
    normalizer_signed_zero_output_count: staticResult.signedZeroOutputCount,
    normalizer_output_materialized: Boolean(staticResult.values),
    normalizer_output_preview: staticResult.outputPreview,
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null,
    exact_truncated_feature_count_per_batch: null, padded_input_count: 0, truncated_input_count: 0,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_values: [], exact_index_preview: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function evaluateNormalizer(source, dtype, shape, mode) {
  const batchCount = shape.length === 1 ? 1 : shape[0];
  const rowWidth = shape.length === 1 ? shape[0] : shape[1];
  const overflowValueCount = countSignedOverflow(source, dtype, mode);
  const integerFloat32RoundingCount = countIntegerFloat32Rounding(source, dtype);
  if (overflowValueCount > 0) {
    return {
      status: "not_assessed_ort_signed_integer_overflow", values: null, inputValueCount: source.length,
      batchCount, rowWidth, divisorPreview: [], zeroDivisorRowCount: null, negativeMaxDivisorRowCount: null,
      integerFloat32RoundingCount, overflowValueCount, nonFiniteOutputCount: null, outputPreview: [],
      signedZeroOutputCount: null,
    };
  }

  const values = source.length <= NORMALIZER_MAX_PROPAGATED_STATIC_VALUES ? new Array(source.length) : null;
  const divisorPreview = [];
  const outputPreview = [];
  let zeroDivisorRowCount = 0;
  let negativeMaxDivisorRowCount = 0;
  let nonFiniteOutputCount = 0;
  let signedZeroOutputCount = 0;
  for (let batch = 0; batch < batchCount; batch += 1) {
    const offset = batch * rowWidth;
    const divisor = normalizerDivisor(source, dtype, offset, rowWidth, mode, values);
    if (divisorPreview.length < 8) divisorPreview.push(canonicalFloatText(divisor));
    if (divisor === 0) zeroDivisorRowCount += 1;
    if (mode === "MAX" && rowWidth > 0 && divisor < 0) negativeMaxDivisorRowCount += 1;
    for (let index = 0; index < rowWidth; index += 1) {
      const sourceIndex = offset + index;
      const inputFloat = toFloat32(source[sourceIndex]);
      let output;
      if (divisor === 0) output = inputFloat;
      else if (mode === "L2") {
        const square = values ? values[sourceIndex] : squareToFloat32(source[sourceIndex], dtype);
        const quotient = Math.fround(square / divisor);
        const magnitude = Math.fround(Math.sqrt(quotient));
        output = source[sourceIndex] < 0 ? Math.fround(-magnitude) : magnitude;
      } else output = Math.fround(inputFloat / divisor);
      if (values) values[sourceIndex] = output;
      if (outputPreview.length < 8) outputPreview.push(canonicalFloatText(output));
      if (!Number.isFinite(output)) nonFiniteOutputCount += 1;
      if (Object.is(output, -0)) signedZeroOutputCount += 1;
    }
  }
  const safeValues = values && nonFiniteOutputCount === 0 ? values : null;
  return {
    status: safeValues ? "assessed_pinned_ort_float32"
      : nonFiniteOutputCount > 0 ? "assessed_non_finite_output_not_propagated"
        : "assessed_counts_output_not_materialized_limit",
    values: safeValues,
    inputValueCount: source.length,
    batchCount, rowWidth, divisorPreview, zeroDivisorRowCount, negativeMaxDivisorRowCount,
    integerFloat32RoundingCount, overflowValueCount, nonFiniteOutputCount,
    signedZeroOutputCount,
    outputPreview,
  };
}

function normalizerDivisor(source, dtype, offset, width, mode, scratch) {
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
    if (scratch && mode === "L2") scratch[offset + index] = term;
    sum = Math.fround(sum + term);
  }
  return sum;
}

function countSignedOverflow(source, dtype, mode) {
  if (!mode.startsWith("L") || !["INT32", "INT64"].includes(dtype)) return 0;
  return source.reduce((count, value) => {
    if (mode === "L1") return count + ((dtype === "INT64" ? value === INT64_MIN : value === INT32_MIN) ? 1 : 0);
    const magnitude = typeof value === "bigint" ? (value < 0n ? -value : value) : Math.abs(value);
    return count + ((dtype === "INT64" ? magnitude > INT64_L2_LIMIT : magnitude > INT32_L2_LIMIT) ? 1 : 0);
  }, 0);
}

function countIntegerFloat32Rounding(source, dtype) {
  if (!['INT32', 'INT64'].includes(dtype)) return 0;
  return source.reduce((count, value) => {
    const rounded = toFloat32(value);
    const changed = typeof value === "bigint" ? BigInt(rounded) !== value : rounded !== value;
    return count + (changed ? 1 : 0);
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
  let magnitude = negative ? -value : value;
  let exponent = magnitude.toString(2).length - 1;
  if (exponent <= 23) return Math.fround(Number(value));
  let shift = BigInt(exponent - 23);
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
    values: null, inputValueCount: null, batchCount: null, rowWidth: null, divisorPreview: [],
    zeroDivisorRowCount: null, negativeMaxDivisorRowCount: null, integerFloat32RoundingCount: null,
    overflowValueCount: null, nonFiniteOutputCount: null, signedZeroOutputCount: null, outputPreview: [],
  };
}

function stringScalarAttribute(attribute) {
  return attribute?.type === 3 && typeof attribute.s === "string"
    && Array.isArray(attribute.valueTypesPresent) && attribute.valueTypesPresent.length === 1 ? attribute.s : null;
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

function canonicalFloatText(value) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}
