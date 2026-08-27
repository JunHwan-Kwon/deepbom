import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

export const SCALER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;

const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);

export function inferOnnxMlScaler({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("scaler_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`scaler_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype === "UNKNOWN") reasons.push("scaler_input_dtype_unresolved");
  else if (!ALLOWED_DTYPES.has(inputDtype)) failures.push(`scaler_input_dtype_not_supported:${inputDtype}`);

  const scaleAttribute = node.attributes?.get("scale");
  const offsetAttribute = node.attributes?.get("offset");
  const scaleValues = scaleAttribute ? floatListAttribute(scaleAttribute) : [];
  const offsetValues = offsetAttribute ? floatListAttribute(offsetAttribute) : [];
  if (scaleAttribute && scaleValues == null) failures.push("scaler_scale_not_float_list");
  if (offsetAttribute && offsetValues == null) failures.push("scaler_offset_not_float_list");
  const scales = scaleValues || [];
  const offsets = offsetValues || [];

  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  const rank = shapeDeclared ? inputShape.length : null;
  const featureStride = scalerFeatureStride(inputShape, shapeDeclared);
  const parameterContract = scalerParameterContract({
    scaleAttribute: Boolean(scaleAttribute), offsetAttribute: Boolean(offsetAttribute),
    scales, offsets, shapeDeclared, rank, featureStride,
  });
  if (parameterContract.status !== "pass") reasons.push(parameterContract.reason);
  if (shapeDeclared && inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("scaler_dynamic_shape_preserved");

  const outputType = inputType?.kind === "tensor" ? makeOnnxTensorType("FLOAT32", inputShape, shapeDeclared) : null;
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = shapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  if (shapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("scaler_output_element_count_overflow");

  const source = staticNumericInput(input, inputDtype);
  let staticResult = unresolvedStaticResult(input);
  if (parameterContract.status === "pass" && source && exactOutputElements != null) {
    if (source.length !== exactOutputElements) failures.push(`scaler_static_value_count_mismatch:${source.length}:${exactOutputElements}`);
    else staticResult = evaluateScaler(source, inputDtype, scales, offsets, featureStride);
  }
  if (staticResult.values && patch) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = staticResult.values;
    patch.staticValuesSource = "scaler_pinned_ort_float32_emulation";
  }

  const nonFiniteParameterCount = [...scales, ...offsets].filter((value) => !Number.isFinite(value)).length;
  const zeroScaleCount = scales.filter((value) => value === 0).length;
  const riskCodes = [];
  if (parameterContract.status === "fail") riskCodes.push("scaler_pinned_ort_attribute_or_shape_contract_invalid");
  if (staticResult.integerFloat32RoundingCount > 0) riskCodes.push("scaler_integer_to_float32_precision_loss");
  if (nonFiniteParameterCount > 0 || staticResult.nonFiniteOutputCount > 0
    || input?.staticValuesStatus === "not_assessed_non_finite_or_unsafe_value") {
    riskCodes.push("scaler_non_finite_parameter_input_or_output");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "Scaler", contract_kind: "tensor_affine_scaler",
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
    output_kind: "tensor", output_dtype: "FLOAT32",
    exact_output_rank: rank, exact_output_shape: inputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_ort_cpu_float_output_same_shape_and_second_dimension_feature_stride",
    runtime_reference_status: "pinned_ort_cpu_scaler_kernel",
    attribute_mode: "offset_then_scale",
    scaler_parameter_contract_status: parameterContract.status,
    scaler_parameter_contract_reason: parameterContract.reason,
    scaler_parameter_mode: parameterContract.mode,
    scaler_feature_stride: featureStride,
    scaler_scale_count: scales.length,
    scaler_offset_count: offsets.length,
    scaler_scale_values: scales.map(canonicalFloatText),
    scaler_offset_values: offsets.map(canonicalFloatText),
    scaler_zero_scale_count: zeroScaleCount,
    scaler_non_finite_parameter_count: nonFiniteParameterCount,
    scaler_static_assessment_status: staticResult.status,
    scaler_exact_input_value_count: staticResult.inputValueCount,
    scaler_integer_float32_rounding_count: staticResult.integerFloat32RoundingCount,
    scaler_non_finite_output_count: staticResult.nonFiniteOutputCount,
    scaler_signed_zero_output_count: staticResult.signedZeroOutputCount,
    scaler_output_materialized: Boolean(staticResult.values),
    scaler_output_preview: staticResult.outputPreview,
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
  const canPropagate = status !== "fail" && parameterContract.status !== "fail";
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: canPropagate && patch && node.outputs?.[0] ? [[node.outputs[0], patch]] : [] }, row,
  };
}

function scalerFeatureStride(shape, shapeDeclared) {
  if (!shapeDeclared || shape.length === 0) return null;
  const candidate = shape.length === 1 ? shape[0] : shape[1];
  return knownDimension(candidate) ? candidate : null;
}

function scalerParameterContract({ scaleAttribute, offsetAttribute, scales, offsets, shapeDeclared, rank, featureStride }) {
  if (!scaleAttribute || !offsetAttribute) {
    return { status: "fail", reason: "scaler_pinned_ort_requires_explicit_scale_and_offset", mode: "invalid" };
  }
  if (scales.length === 0) return { status: "fail", reason: "scaler_pinned_ort_rejects_empty_scale", mode: "invalid" };
  if (scales.length !== offsets.length) return { status: "fail", reason: "scaler_pinned_ort_scale_offset_length_mismatch", mode: "invalid" };
  if (!shapeDeclared) return { status: "not_assessed", reason: "scaler_input_rank_unresolved", mode: scales.length === 1 ? "scalar" : "unresolved" };
  if (rank === 0) return { status: "fail", reason: "scaler_pinned_ort_rejects_rank_zero_input", mode: "invalid" };
  if (scales.length === 1) return { status: "pass", reason: "scaler_scalar_parameters", mode: "scalar" };
  if (featureStride == null) return { status: "not_assessed", reason: "scaler_feature_stride_unresolved", mode: "per_feature" };
  if (scales.length !== featureStride) return { status: "fail", reason: "scaler_pinned_ort_parameter_length_not_one_or_feature_stride", mode: "invalid" };
  return { status: "pass", reason: "scaler_per_feature_parameters", mode: "per_feature" };
}

function evaluateScaler(source, dtype, scales, offsets, stride) {
  const values = source.length <= SCALER_MAX_PROPAGATED_STATIC_VALUES ? new Array(source.length) : null;
  const outputPreview = [];
  let integerFloat32RoundingCount = 0;
  let nonFiniteOutputCount = 0;
  let signedZeroOutputCount = 0;
  const scalar = scales.length === 1;
  for (let index = 0; index < source.length; index += 1) {
    const parameterIndex = scalar ? 0 : index % stride;
    const scale = scales[parameterIndex];
    const offset = offsets[parameterIndex];
    let output;
    if (dtype === "FLOAT64") output = Math.fround((Number(source[index]) - offset) * scale);
    else {
      const inputFloat = toFloat32(source[index]);
      if (["INT32", "INT64"].includes(dtype)) {
        const changed = typeof source[index] === "bigint" ? BigInt(inputFloat) !== source[index] : inputFloat !== source[index];
        if (changed) integerFloat32RoundingCount += 1;
      }
      output = Math.fround(Math.fround(inputFloat - offset) * scale);
    }
    if (values) values[index] = output;
    if (outputPreview.length < 8) outputPreview.push(canonicalFloatText(output));
    if (!Number.isFinite(output)) nonFiniteOutputCount += 1;
    if (Object.is(output, -0)) signedZeroOutputCount += 1;
  }
  const safeValues = values && nonFiniteOutputCount === 0 ? values : null;
  return {
    status: safeValues ? "assessed_pinned_ort_float32"
      : nonFiniteOutputCount > 0 ? "assessed_non_finite_output_not_propagated"
        : "assessed_counts_output_not_materialized_limit",
    values: safeValues, inputValueCount: source.length, integerFloat32RoundingCount,
    nonFiniteOutputCount, signedZeroOutputCount, outputPreview,
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
    values: null, inputValueCount: null, integerFloat32RoundingCount: null,
    nonFiniteOutputCount: null, signedZeroOutputCount: null, outputPreview: [],
  };
}

function floatListAttribute(attribute) {
  if (attribute?.type !== 6 || !Array.isArray(attribute.floats)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 6) return null;
  return attribute.floats.map((value) => Math.fround(value));
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
