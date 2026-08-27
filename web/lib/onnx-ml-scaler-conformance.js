import {
  canonicalFloatText,
  numericArraysExactlyEqual,
  parseCanonicalFloatText,
  staticValuesWithSignedZeros,
} from "./onnx-static-value-evidence.js";

const MATERIALIZATION_LIMIT = 1_000_000;
const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);

export function validateScalerRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || row.op_name !== "Scaler") return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const output = tensors.find((tensor) => tensor.name === row.output_name);
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === "Scaler" && candidate.domain === "ai.onnx.ml");
  const scale = publicFloatListAttribute(op, "scale");
  const offset = publicFloatListAttribute(op, "offset");
  const shape = Array.isArray(row.input_shape) ? row.input_shape : [];
  const shapeDeclared = Number.isSafeInteger(row.input_rank) && row.input_rank >= 0;
  const featureStride = shapeDeclared && shape.length > 0
    ? knownDimension(shape.length === 1 ? shape[0] : shape[1]) ? (shape.length === 1 ? shape[0] : shape[1]) : null
    : null;
  const elements = shapeDeclared && shape.every(knownDimension) ? safeShapeProduct(shape) : null;
  const structurallyValid = op && row.input_kind === "tensor" && ALLOWED_DTYPES.has(row.input_dtype)
    && row.output_kind === "tensor" && row.output_dtype === "FLOAT32"
    && row.exact_output_rank === row.input_rank
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(shape)
    && row.exact_dense_output_element_count === elements
    && row.output_shape_basis === "pinned_ort_cpu_float_output_same_shape_and_second_dimension_feature_stride"
    && row.runtime_reference_status === "pinned_ort_cpu_scaler_kernel"
    && row.attribute_mode === "offset_then_scale"
    && Array.isArray(row.scaler_scale_values) && Array.isArray(row.scaler_offset_values)
    && Array.isArray(row.scaler_output_preview);
  if (!structurallyValid) return false;
  if (!scale.ok || !offset.ok) return row.status === "fail" && row.reason_codes.length > 0;

  const contract = parameterContract(scale.present, offset.present, scale.values, offset.values, shapeDeclared, row.input_rank, featureStride);
  const source = staticInput(input, row.input_dtype);
  const reconstructed = contract.status === "pass" && source && elements != null && source.length === elements
    ? reconstruct(source, row.input_dtype, scale.values, offset.values, featureStride)
    : unresolvedResult(input);
  const expectedRisks = [];
  if (contract.status === "fail") expectedRisks.push("scaler_pinned_ort_attribute_or_shape_contract_invalid");
  if (reconstructed.integerRoundingCount > 0) expectedRisks.push("scaler_integer_to_float32_precision_loss");
  const nonfiniteParameters = [...scale.values, ...offset.values].filter((value) => !Number.isFinite(value)).length;
  if (nonfiniteParameters > 0 || reconstructed.nonfiniteCount > 0
    || input?.static_values_status === "not_assessed_non_finite_or_unsafe_value") {
    expectedRisks.push("scaler_non_finite_parameter_input_or_output");
  }
  const outputStatic = staticValuesWithSignedZeros(output);
  const outputMatches = reconstructed.values
    ? outputStatic != null && numericArraysExactlyEqual(outputStatic, reconstructed.values)
    : output?.static_values_complete !== true;
  return row.scaler_parameter_contract_status === contract.status
    && row.scaler_parameter_contract_reason === contract.reason
    && row.scaler_parameter_mode === contract.mode
    && row.scaler_feature_stride === featureStride
    && row.scaler_scale_count === scale.values.length
    && row.scaler_offset_count === offset.values.length
    && JSON.stringify(row.scaler_scale_values) === JSON.stringify(scale.texts)
    && JSON.stringify(row.scaler_offset_values) === JSON.stringify(offset.texts)
    && row.scaler_zero_scale_count === scale.values.filter((value) => value === 0).length
    && row.scaler_non_finite_parameter_count === nonfiniteParameters
    && row.scaler_static_assessment_status === reconstructed.status
    && row.scaler_exact_input_value_count === reconstructed.inputCount
    && row.scaler_integer_float32_rounding_count === reconstructed.integerRoundingCount
    && row.scaler_non_finite_output_count === reconstructed.nonfiniteCount
    && row.scaler_signed_zero_output_count === reconstructed.signedZeroCount
    && row.scaler_output_materialized === Boolean(reconstructed.values)
    && JSON.stringify(row.scaler_output_preview) === JSON.stringify(reconstructed.outputPreview)
    && outputMatches
    && JSON.stringify([...row.risk_codes].sort()) === JSON.stringify(expectedRisks.sort());
}

function publicFloatListAttribute(op, name) {
  const attribute = (op?.onnx_attributes || []).find((item) => item.name === name);
  if (!attribute) return { ok: true, present: false, values: [], texts: [] };
  if (attribute.type !== 6 || !Array.isArray(attribute.float_values) || !Array.isArray(attribute.float_values_text)
    || attribute.float_values.length !== attribute.float_values_text.length) {
    return { ok: false, present: true, values: [], texts: [] };
  }
  const values = [];
  for (let index = 0; index < attribute.float_values_text.length; index += 1) {
    const parsed = parseCanonicalFloatText(attribute.float_values_text[index], { float32: true });
    if (!parsed.ok) return { ok: false, present: true, values: [], texts: [] };
    const jsonValue = attribute.float_values[index];
    if (Number.isFinite(parsed.value)) {
      if (!Number.isFinite(jsonValue) || Object.is(jsonValue, -0)
        || !(Object.is(jsonValue, parsed.value) || jsonValue === parsed.value)) return { ok: false, present: true, values: [], texts: [] };
    } else if (jsonValue !== null) return { ok: false, present: true, values: [], texts: [] };
    values.push(parsed.value);
  }
  return { ok: true, present: true, values, texts: [...attribute.float_values_text] };
}

function parameterContract(scalePresent, offsetPresent, scales, offsets, shapeDeclared, rank, featureStride) {
  if (!scalePresent || !offsetPresent) return { status: "fail", reason: "scaler_pinned_ort_requires_explicit_scale_and_offset", mode: "invalid" };
  if (scales.length === 0) return { status: "fail", reason: "scaler_pinned_ort_rejects_empty_scale", mode: "invalid" };
  if (scales.length !== offsets.length) return { status: "fail", reason: "scaler_pinned_ort_scale_offset_length_mismatch", mode: "invalid" };
  if (!shapeDeclared) return { status: "not_assessed", reason: "scaler_input_rank_unresolved", mode: scales.length === 1 ? "scalar" : "unresolved" };
  if (rank === 0) return { status: "fail", reason: "scaler_pinned_ort_rejects_rank_zero_input", mode: "invalid" };
  if (scales.length === 1) return { status: "pass", reason: "scaler_scalar_parameters", mode: "scalar" };
  if (featureStride == null) return { status: "not_assessed", reason: "scaler_feature_stride_unresolved", mode: "per_feature" };
  if (scales.length !== featureStride) return { status: "fail", reason: "scaler_pinned_ort_parameter_length_not_one_or_feature_stride", mode: "invalid" };
  return { status: "pass", reason: "scaler_per_feature_parameters", mode: "per_feature" };
}

function reconstruct(source, dtype, scales, offsets, stride) {
  const values = source.length <= MATERIALIZATION_LIMIT ? new Array(source.length) : null;
  const outputPreview = [];
  let integerRoundingCount = 0;
  let nonfiniteCount = 0;
  let signedZeroCount = 0;
  for (let index = 0; index < source.length; index += 1) {
    const parameterIndex = scales.length === 1 ? 0 : index % stride;
    let output;
    if (dtype === "FLOAT64") output = Math.fround((Number(source[index]) - offsets[parameterIndex]) * scales[parameterIndex]);
    else {
      const projected = toFloat32(source[index]);
      if (["INT32", "INT64"].includes(dtype)) {
        const changed = typeof source[index] === "bigint" ? BigInt(projected) !== source[index] : projected !== source[index];
        if (changed) integerRoundingCount += 1;
      }
      output = Math.fround(Math.fround(projected - offsets[parameterIndex]) * scales[parameterIndex]);
    }
    if (values) values[index] = output;
    if (outputPreview.length < 8) outputPreview.push(canonicalFloatText(output));
    if (!Number.isFinite(output)) nonfiniteCount += 1;
    if (Object.is(output, -0)) signedZeroCount += 1;
  }
  const safeValues = values && nonfiniteCount === 0 ? values : null;
  return {
    status: safeValues ? "assessed_pinned_ort_float32"
      : nonfiniteCount > 0 ? "assessed_non_finite_output_not_propagated"
        : "assessed_counts_output_not_materialized_limit",
    values: safeValues, inputCount: source.length, integerRoundingCount,
    nonfiniteCount, signedZeroCount, outputPreview,
  };
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
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)) return staticValuesWithSignedZeros(input);
  return null;
}

function unresolvedResult(input) {
  return {
    status: input?.role === "initializer" ? input.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    values: null, inputCount: null, integerRoundingCount: null,
    nonfiniteCount: null, signedZeroCount: null, outputPreview: [],
  };
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
