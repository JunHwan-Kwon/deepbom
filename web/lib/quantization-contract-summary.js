import {
  canonicalContractSha256,
  contractHashDescriptor,
  interfaceContractPayload,
  INTERFACE_CONTRACT_SCHEMA,
  legacyContractSha256,
  quantizationContractPayload,
} from "./interface-contract.js";

const CONV_LIKE = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]);
const BIAS_RELATIVE_TOLERANCE = 1e-5;
const NEAR_ZERO_REPRESENTABLE_ABS_THRESHOLD = 1e-6;
const SCALE_OUTLIER_RATIO_THRESHOLD = 1e6;
const COMMON_INPUT_CONVENTIONS = [
  { id: "normalized_0_1", label: "[0,1] normalized", range: [0, 1] },
  { id: "normalized_minus1_1", label: "[-1,1] normalized", range: [-1, 1] },
  { id: "pixel_0_255", label: "[0,255] pixel domain", range: [0, 255] },
  { id: "signed_code_identity", label: "[-128,127] signed-code identity", range: [-128, 127], dtype: "INT8" },
];

export function buildBiasScaleCheck(analysis) {
  const details = [];
  let checkedChannels = 0;
  for (const op of analysis?.ops || []) {
    if (!CONV_LIKE.has(opName(op))) continue;
    const input = tensorAt(analysis, op.inputs?.[0]);
    const weight = tensorAt(analysis, op.inputs?.[1]);
    const bias = tensorAt(analysis, op.inputs?.[2]);
    const inputScale = firstScale(input);
    const weightScales = numericScales(weight);
    const biasScales = numericScales(bias);
    if (!(inputScale > 0) || !weightScales.length || !biasScales.length) continue;
    const channels = Math.max(weightScales.length, biasScales.length);
    let maxRelativeError = 0;
    let worst = null;
    const derivedInputScales = [];
    for (let channel = 0; channel < channels; channel += 1) {
      const weightScale = weightScales[Math.min(channel, weightScales.length - 1)];
      const actual = biasScales[Math.min(channel, biasScales.length - 1)];
      const expected = inputScale * weightScale;
      if (!(expected > 0) || !(actual > 0)) continue;
      checkedChannels += 1;
      derivedInputScales.push(actual / weightScale);
      const relativeError = Math.abs(actual - expected) / expected;
      if (!worst || relativeError > maxRelativeError) {
        maxRelativeError = relativeError;
        worst = { channel, actual, expected, relative_error: relativeError };
      }
    }
    details.push({
      op_index: op.index,
      op_name: op.name,
      input_tensor_index: tensorIndex(input),
      weight_tensor_index: tensorIndex(weight),
      bias_tensor_index: tensorIndex(bias),
      checked_channels: channels,
      declared_input_scale: inputScale,
      derived_input_scale_min: Math.min(...derivedInputScales),
      derived_input_scale_max: Math.max(...derivedInputScales),
      maximum_relative_error: maxRelativeError,
      status: maxRelativeError > BIAS_RELATIVE_TOLERANCE ? "fail" : "pass",
      worst_channel: worst,
      formula: "bias_scale[channel] == input_scale * weight_scale[channel]",
      evidence_json_pointer: `/evidence/static_analysis/ops/${op.index}`,
    });
  }
  const mismatchGroups = details.filter((item) => item.status === "fail");
  return {
    status: details.length ? (mismatchGroups.length ? "fail" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_groups: details.length,
    checked_channels: checkedChannels,
    mismatch_groups: mismatchGroups.length,
    maximum_relative_error: maxOrZero(details.map((item) => item.maximum_relative_error)),
    relative_tolerance: BIAS_RELATIVE_TOLERANCE,
    method: "bias_scale[channel] == input_scale * weight_scale[channel]",
    details,
  };
}

export function buildRepresentableKernelChannelCheck(analysis) {
  const kernels = new Map();
  for (const op of analysis?.ops || []) {
    if (!CONV_LIKE.has(opName(op))) continue;
    const weight = tensorAt(analysis, op.inputs?.[1]);
    const index = tensorIndex(weight);
    if (index == null || kernels.has(index)) continue;
    kernels.set(index, { tensor: weight, consumers: [] });
  }
  for (const op of analysis?.ops || []) {
    const index = Number(op.inputs?.[1]);
    if (kernels.has(index)) kernels.get(index).consumers.push(op);
  }

  const details = [];
  let assessedChannels = 0;
  let flaggedChannels = 0;
  for (const [index, { tensor, consumers }] of kernels) {
    const dtype = String(tensor?.dtype || "").toUpperCase();
    const scaleEntries = (tensor?.scale_sample || [])
      .map((value, channel) => ({ channel, scale: Number(value) }))
      .filter((row) => row.scale > 0 && Number.isFinite(row.scale));
    if (!["INT8", "UINT8"].includes(dtype) || scaleEntries.length < 2) continue;
    const [qmin, qmax] = dtype === "INT8" ? [-127, 127] : [0, 255];
    const zeroPoints = (tensor?.zero_point_sample || []).map(Number);
    const maximumScale = Math.max(...scaleEntries.map((row) => row.scale));
    const minimumScale = Math.min(...scaleEntries.map((row) => row.scale));
    const flagged = [];
    assessedChannels += scaleEntries.length;
    for (const { channel, scale } of scaleEntries) {
      const zeroPoint = Number.isFinite(zeroPoints[channel])
        ? zeroPoints[channel]
        : Number.isFinite(zeroPoints[0]) ? zeroPoints[0] : 0;
      const maximumCodeDistance = Math.max(Math.abs(qmin - zeroPoint), Math.abs(qmax - zeroPoint));
      const maximumRepresentableAbs = scale * maximumCodeDistance;
      const maximumToChannelScaleRatio = maximumScale / scale;
      if (maximumRepresentableAbs <= NEAR_ZERO_REPRESENTABLE_ABS_THRESHOLD
        && maximumToChannelScaleRatio >= SCALE_OUTLIER_RATIO_THRESHOLD) {
        flagged.push({
          channel,
          scale,
          zero_point: zeroPoint,
          maximum_code_distance: maximumCodeDistance,
          maximum_representable_abs: maximumRepresentableAbs,
          maximum_to_channel_scale_ratio: maximumToChannelScaleRatio,
        });
      }
    }
    flaggedChannels += flagged.length;
    details.push({
      tensor_index: index,
      tensor_name: tensor?.name || `T${index}`,
      dtype,
      quantized_dimension: Number(tensor?.quantized_dimension || 0),
      declared_scale_count: Number(tensor?.quant_scales || tensor?.scale_sample?.length || 0),
      assessed_channel_count: scaleEntries.length,
      flagged_channel_count: flagged.length,
      minimum_scale: minimumScale,
      maximum_scale: maximumScale,
      maximum_to_minimum_scale_ratio: maximumScale / minimumScale,
      flagged_channels: flagged.slice(0, 64),
      flagged_channel_sample_truncated: flagged.length > 64,
      consumer_ops: consumers.slice(0, 8).map((op) => `#${op.index} ${op.name}`),
      evidence_json_pointer: `/evidence/static_analysis/tensors/${index}`,
    });
  }
  const flaggedTensors = details.filter((row) => row.flagged_channel_count > 0);
  return {
    status: flaggedChannels ? "review" : details.length ? "pass" : "not_applicable",
    evidence_class: flaggedChannels ? "DERIVED_WITH_HEURISTIC_THRESHOLD" : "DERIVED",
    assessed_kernel_tensors: details.length,
    assessed_channels: assessedChannels,
    flagged_kernel_tensors: flaggedTensors.length,
    flagged_channels: flaggedChannels,
    maximum_representable_abs_threshold: NEAR_ZERO_REPRESENTABLE_ABS_THRESHOLD,
    scale_outlier_ratio_threshold: SCALE_OUTLIER_RATIO_THRESHOLD,
    method: "For each per-axis kernel channel, compute scale[channel] * max(|qmin-zp[channel]|, |qmax-zp[channel]|), using the TFLite symmetric INT8 weight domain [-127,127] and UINT8 [0,255]. Review only channels at or below 1e-6 whose scale is at least 1e6 below the tensor maximum.",
    interpretation_boundary: "The representable range and scale ratio are exact artifact-derived quantities. Thresholds are methodology heuristics; functional inactivity, QAT failure, and task-accuracy impact are not proven.",
    details,
  };
}

export function buildIoDequantizationCheck(analysis) {
  const ledger = buildInterfaceQuantizationContractLedger(analysis);
  const details = ledger.parameters.flatMap((parameter) => {
    const domain = parameter.quantization.scalar_real_code_domain;
    if (parameter.quantization.status !== "complete" || !domain || !["INT8", "UINT8"].includes(parameter.dtype)) return [];
    const [scale] = parameter.quantization.scales;
    const [zeroPoint] = parameter.quantization.zero_points;
    return [{
      role: parameter.direction,
      ordinal: parameter.ordinal,
      tensor_index: parameter.tensor_index,
      tensor_name: parameter.tensor_name,
      dtype: parameter.dtype,
      scale,
      zero_point: zeroPoint,
      real_range: [domain.real_min, domain.real_max],
      tensor_numerical_contract_status: "known_from_artifact_quantization_metadata",
      source_data_to_tensor_preprocessing_status: "not_embedded_in_artifact",
      equation: `real_value = ${scale} * (quantized_value - ${zeroPoint})`,
      evidence_json_pointer: `/evidence/static_analysis/${parameter.direction}s/${parameter.ordinal}`,
    }];
  });
  const inputs = details.filter((row) => row.role === "input");
  const outputs = details.filter((row) => row.role === "output");
  return {
    status: inputs.length || outputs.length ? "pass" : "not_applicable",
    evidence_class: "DERIVED",
    formula: "real_value = scale * (quantized_value - zero_point)",
    inputs,
    outputs,
  };
}

export function buildInterfaceQuantizationContractLedger(analysis = {}) {
  const parameters = [
    ...(analysis.inputs || []).map((tensor, ordinal) => interfaceParameter(analysis, tensor, "input", ordinal)),
    ...(analysis.outputs || []).map((tensor, ordinal) => interfaceParameter(analysis, tensor, "output", ordinal)),
  ];
  const complete = parameters.filter((row) => row.quantization.status === "complete");
  const boundaryContract = buildInterfaceBoundaryContract(parameters);
  const value = {
    schema: INTERFACE_CONTRACT_SCHEMA,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      method: "SHA-256 over UTF-8 RFC8785-JCS canonical JSON",
      selection: "containing_object_after_exclusions",
      excluded_json_pointers: ["/ledger_sha256", "/ledger_hash", "/generated_at", "/subject", "/provenance"],
      interpretation: "Remove ledger_sha256, ledger_hash, and any document-envelope generated_at, subject, and provenance members from the ledger object; canonicalize the remaining contract with RFC8785-JCS, encode it as UTF-8, and hash it with SHA-256.",
      compatibility: "Per-parameter v1.1 JSON.stringify hashes remain inside each hash descriptor during schema migration.",
    },
    parameter_count: parameters.length,
    quantized_parameter_count: complete.length,
    unquantized_parameter_count: parameters.filter((row) => row.quantization.status === "not_quantized").length,
    invalid_or_incomplete_parameter_count: parameters.filter((row) => row.quantization.status === "invalid_or_incomplete").length,
    per_tensor_parameter_count: complete.filter((row) => row.quantization.granularity === "per_tensor").length,
    per_axis_parameter_count: complete.filter((row) => row.quantization.granularity === "per_axis").length,
    blocked_parameter_count: complete.filter((row) => row.quantization.granularity === "blocked").length,
    distinct_complete_quantization_contract_count: new Set(complete.map((row) => row.quantization.contract_sha256)).size,
    multiple_complete_quantization_contracts_within_artifact: new Set(complete.map((row) => row.quantization.contract_sha256)).size > 1,
    source_preprocessing_contract_status: preprocessingContractStatus(analysis),
    boundary_contract: boundaryContract,
    parameters,
    interpretation_boundary: "The ledger records serialized boundary dtype, shape, and affine quantization facts and validates affine value/cardinality consistency. A FLOAT32 boundary is an explicit unquantized storage contract, not missing quantization metadata. This ledger alone does not establish RGB/BGR order, source-value normalization, mean/standard-deviation transforms, resize interpolation, application tensor layout, the production preprocessing implementation, mismatch prevalence, or task accuracy.",
  };
  const ledgerSha256 = canonicalContractSha256(value);
  return {
    ...value,
    ledger_sha256: ledgerSha256,
    ledger_hash: contractHashDescriptor(ledgerSha256, null, {
      selection: "containing_object_after_exclusions",
      excludedJsonPointers: ["/ledger_sha256", "/ledger_hash", "/generated_at", "/subject", "/provenance"],
      interpretation: "Remove ledger_sha256, ledger_hash, and any document-envelope generated_at, subject, and provenance members from the ledger object; canonicalize the remaining contract with RFC8785-JCS, encode it as UTF-8, and hash it with SHA-256.",
    }),
  };
}

export function buildInterfaceBoundaryContract(parameters = []) {
  const summarize = (rows) => {
    const complete = rows.filter((row) => row.quantization?.status === "complete").length;
    const unquantized = rows.filter(
      (row) => row.quantization?.status === "not_quantized",
    ).length;
    const invalid = rows.length - complete - unquantized;
    let status = "not_declared";
    if (invalid > 0) status = "invalid_or_incomplete";
    else if (rows.length > 0 && complete === rows.length) status = "fully_affine_quantized";
    else if (rows.length > 0 && unquantized === rows.length) status = "fully_unquantized";
    else if (rows.length > 0) status = "mixed_quantized_unquantized";
    return {
      status,
      parameter_count: rows.length,
      affine_quantized_parameter_count: complete,
      unquantized_parameter_count: unquantized,
      invalid_or_incomplete_parameter_count: invalid,
      float32_parameter_count: rows.filter((row) => row.dtype === "FLOAT32").length,
      integer_parameter_count: rows.filter((row) =>
        ["INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32"].includes(row.dtype)).length,
    };
  };
  const inputs = parameters.filter((row) => row.direction === "input");
  const outputs = parameters.filter((row) => row.direction === "output");
  return {
    schema: "deepbom.interface_boundary_contract.v1",
    ...summarize(parameters),
    inputs: summarize(inputs),
    outputs: summarize(outputs),
    recorded_facts: ["direction", "ordinal", "dtype", "shape", "shape_signature", "affine_quantization"],
    not_established_by_this_contract: [
      "channel_order",
      "source_value_domain",
      "mean_standard_deviation_normalization",
      "resize_interpolation",
      "application_tensor_layout",
      "semantic_labels",
    ],
  };
}

export function buildInputQuantizationConventionCheck(analysis) {
  const details = buildIoDequantizationCheck(analysis).inputs.map((input) => {
    const candidates = COMMON_INPUT_CONVENTIONS
      .filter((candidate) => !candidate.dtype || candidate.dtype === input.dtype)
      .map((candidate) => {
        const expectedStep = (candidate.range[1] - candidate.range[0]) / 255;
        const endpointError = Math.max(
          Math.abs(input.real_range[0] - candidate.range[0]),
          Math.abs(input.real_range[1] - candidate.range[1]),
        );
        const tolerance = Math.max(input.scale, expectedStep) * 1.01;
        return {
          id: candidate.id,
          label: candidate.label,
          expected_real_range: candidate.range,
          maximum_endpoint_error: endpointError,
          tolerance,
          matches_within_one_quantization_step: endpointError <= tolerance,
        };
      })
      .sort((left, right) => left.maximum_endpoint_error - right.maximum_endpoint_error
        || left.id.localeCompare(right.id));
    const matches = candidates.filter((candidate) => candidate.matches_within_one_quantization_step);
    return {
      ...input,
      status: matches.length ? "matched_common_convention" : "no_common_full_domain_match",
      matched_convention_ids: matches.map((candidate) => candidate.id),
      closest_convention: candidates[0] || null,
      convention_candidates: candidates,
    };
  });
  const unmatched = details.filter((detail) => detail.status === "no_common_full_domain_match");
  return {
    status: unmatched.length ? "review" : details.length ? "pass" : "not_applicable",
    evidence_class: "DERIVED_RANGE_WITH_REFERENCE_HEURISTIC",
    assessed_inputs: details.length,
    unmatched_inputs: unmatched.length,
    method: "Compute the exact full quantized-code real range from artifact scale/zero-point metadata. Compare its two endpoints with common [0,1], [-1,1], [0,255], and applicable signed-code identity conventions using a tolerance of one quantization step.",
    interpretation_boundary: "The real range is artifact-derived. Convention matching is a reference heuristic: an unmatched custom range can be intentional and does not prove bad calibration or an application bug. The production decoder, channel order, resize, normalization, and quantizer remain unobserved until separately bound.",
    details,
  };
}

function interfaceParameter(analysis, tensor, direction, ordinal) {
  const rawScales = numericContractValues(tensor?.interface_scale_values, tensor?.scale_sample);
  const rawZeroPoints = numericContractValues(tensor?.interface_zero_point_values, tensor?.zero_point_sample);
  const scaleCount = declaredCount(tensor?.quant_scales, rawScales.length);
  const zeroPointCount = declaredCount(tensor?.quant_zero_points, rawZeroPoints.length);
  const scales = rawScales.slice(0, scaleCount);
  const zeroPoints = rawZeroPoints.slice(0, zeroPointCount);
  const shape = Array.isArray(tensor?.shape) ? tensor.shape.map(Number) : [];
  const quantizedDimension = Number.isSafeInteger(Number(tensor?.quantized_dimension))
    ? Number(tensor.quantized_dimension) : null;
  const noValues = scaleCount === 0 && zeroPointCount === 0 && !rawScales.length && !rawZeroPoints.length;
  const serializedParameterization = String(tensor?.quantization_parameterization || "").toLowerCase().replaceAll("-", "_");
  const granularity = noValues ? "not_quantized"
    : serializedParameterization === "blocked" ? "blocked"
    : scaleCount === 1 && zeroPointCount === 1 ? "per_tensor"
      : scaleCount > 1 && zeroPointCount > 0 ? "per_axis" : "invalid_or_incomplete";
  const blockSize = positiveIntegerOrNull(tensor?.quantization_block_size);
  const cardinality = interfaceCardinality(
    granularity,
    scaleCount,
    zeroPointCount,
    quantizedDimension,
    shape,
    blockSize,
    tensor?.quantization_scale_tensor_shape,
    tensor?.quantization_cardinality_status,
    tensor?.quantization_cardinality_detail,
  );
  const codeRange = dtypeCodeRange(tensor?.dtype);
  const valuesValid = rawScales.length === scaleCount && rawZeroPoints.length === zeroPointCount
    && scales.every((value) => Number.isFinite(value) && value > 0)
    && zeroPoints.every((value) => Number.isSafeInteger(value)
      && codeRange && value >= codeRange[0] && value <= codeRange[1]);
  const status = noValues ? "not_quantized"
    : valuesValid && cardinality.status === "valid" ? "complete" : "invalid_or_incomplete";
  const semantics = quantizationContractPayload({
    status,
    scheme: status === "not_quantized" ? "none" : "affine",
    granularity,
    parameterization: granularity,
    scales,
    zeroPoints,
    axis: ["per_axis", "blocked"].includes(granularity) ? quantizedDimension : null,
    blockSize,
  });
  const legacyGranularity = noValues ? "not_quantized"
    : scaleCount === 1 && zeroPointCount === 1 ? "per_tensor"
      : scaleCount > 1 && zeroPointCount > 0 ? "per_axis" : "invalid_or_incomplete";
  const legacySemantics = {
    scheme: status === "not_quantized" ? "none" : "affine",
    granularity: legacyGranularity,
    scales,
    zero_points: zeroPoints,
    axis: legacyGranularity === "per_axis" ? quantizedDimension : null,
  };
  const quantizationSha256 = canonicalContractSha256(semantics);
  const serializedPayload = {
    ...semantics,
    quantized_dimension: quantizedDimension,
    axis_source: tensor?.quantization_axis_source || null,
    scale_tensor_shape: quantizationValueShape(tensor?.quantization_scale_tensor_shape, scaleCount),
    zero_point_tensor_shape: quantizationValueShape(tensor?.quantization_zero_point_tensor_shape, zeroPointCount),
  };
  const serializedSha256 = canonicalContractSha256(serializedPayload);
  const quantization = {
    status,
    ...semantics,
    affine_mapping_status: status === "not_quantized" ? "no_affine_mapping_declared"
      : status === "complete" ? "complete_affine_mapping" : "invalid_or_incomplete_affine_mapping",
    scheme_evidence: status === "not_quantized" ? "not_applicable"
      : "generic_affine_equation_observed_symmetry_not_encoded",
    symmetry_classification: status === "not_quantized" ? "not_applicable" : "not_encoded",
    scale_count: scaleCount,
    zero_point_count: zeroPointCount,
    scale_values_complete: rawScales.length === scaleCount,
    zero_point_values_complete: rawZeroPoints.length === zeroPointCount,
    quantized_dimension: quantizedDimension,
    axis_source: tensor?.quantization_axis_source || null,
    axis_applicable: ["per_axis", "blocked"].includes(granularity),
    quantized_dimension_role: ["per_axis", "blocked"].includes(granularity)
      ? "semantic_affine_axis" : quantizedDimension == null ? "not_serialized" : "serialized_default_not_semantic_axis",
    block_size: blockSize,
    scale_tensor_shape: serializedPayload.scale_tensor_shape,
    zero_point_tensor_shape: serializedPayload.zero_point_tensor_shape,
    value_vector_shape_basis: [tensor?.quantization_scale_tensor_shape, tensor?.quantization_zero_point_tensor_shape]
      .some((candidate) => Array.isArray(candidate) && candidate.length)
      ? "serialized_or_source_inferred_tensor_shape" : status === "not_quantized" ? "not_applicable" : "derived_flat_affine_value_vector",
    cardinality_status: cardinality.status,
    cardinality_reason: cardinality.reason,
    scalar_real_code_domain: scalarCodeDomain(codeRange, scales, zeroPoints),
    contract_sha256: quantizationSha256,
    contract_hash: contractHashDescriptor(quantizationSha256, legacyContractSha256(legacySemantics), {
      includedJsonPointers: ["/status", "/scheme", "/granularity", "/parameterization", "/scales", "/zero_points", "/axis", "/block_size"],
      legacyOrderedPayloadFields: ["scheme", "granularity", "scales", "zero_points", "axis"],
    }),
    serialized_quantization_sha256: serializedSha256,
    serialized_quantization_hash: contractHashDescriptor(
      serializedSha256,
      legacyContractSha256({ ...legacySemantics, quantized_dimension: quantizedDimension }),
      {
        includedJsonPointers: ["/status", "/scheme", "/granularity", "/parameterization", "/scales", "/zero_points", "/axis", "/block_size", "/quantized_dimension", "/axis_source", "/scale_tensor_shape", "/zero_point_tensor_shape"],
        legacyOrderedPayloadFields: ["scheme", "granularity", "scales", "zero_points", "axis", "quantized_dimension"],
      },
    ),
  };
  const inputContract = direction === "input"
    ? (analysis.input_contracts || []).find((row) => Number(row.tensor_index) === Number(tensor?.index)) : null;
  const contract = {
    parameter_id: `${direction}:${ordinal}:T${tensorIndex(tensor) ?? ordinal}`,
    direction,
    ordinal,
    tensor_index: tensorIndex(tensor),
    tensor_name: tensor?.name || "",
    dtype: String(tensor?.dtype || "UNKNOWN").toUpperCase(),
    shape,
    shape_signature: Array.isArray(tensor?.shape_signature) && tensor.shape_signature.length
      ? tensor.shape_signature.map(Number) : null,
    source_data_to_tensor_preprocessing_status: direction === "output"
      ? "not_applicable"
      : normalizePreprocessingStatus(inputContract?.source_data_to_tensor_preprocessing_status),
    quantization,
  };
  const legacyInterfacePayload = {
    direction, ordinal, name: contract.tensor_name, dtype: contract.dtype,
    shape, shape_signature: contract.shape_signature, quantization: legacySemantics,
  };
  contract.interface_contract_sha256 = canonicalContractSha256(interfaceContractPayload(contract));
  contract.interface_contract_hash = contractHashDescriptor(
    contract.interface_contract_sha256,
    legacyContractSha256(legacyInterfacePayload),
    {
      includedJsonPointers: ["/direction", "/ordinal", "/tensor_name", "/dtype", "/shape", "/shape_signature", "/quantization/status", "/quantization/scheme", "/quantization/granularity", "/quantization/parameterization", "/quantization/scales", "/quantization/zero_points", "/quantization/axis", "/quantization/block_size"],
      legacyOrderedPayloadFields: ["direction", "ordinal", "name", "dtype", "shape", "shape_signature", "quantization"],
    },
  );
  return contract;
}

function interfaceCardinality(granularity, scales, zeroPoints, axis, shape, blockSize, scaleShape, boundStatus, boundDetail) {
  if (granularity === "not_quantized") return { status: "valid", reason: "No affine parameters are declared." };
  if (granularity === "per_tensor") return { status: "valid", reason: "One scale and one zero-point are declared." };
  if (granularity === "blocked") {
    if (boundStatus === "fail") return { status: "invalid", reason: boundDetail || "ONNX blocked-quantization cardinality is invalid." };
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length || !(blockSize > 0)
      || !Array.isArray(scaleShape) || scaleShape.length !== shape.length || scales !== zeroPoints) {
      return { status: "invalid", reason: "Blocked quantization requires a valid axis, positive block size, rank-matched scale shape, and matching scale/zero-point counts." };
    }
    const expected = shape.map((dimension, index) => index === axis ? Math.ceil(Number(dimension) / blockSize) : Number(dimension));
    const geometryValid = expected.every((dimension, index) => Number.isSafeInteger(dimension)
      && dimension >= 0 && dimension === Number(scaleShape[index]));
    const expectedCount = geometryValid ? expected.reduce((product, dimension) => product * dimension, 1) : null;
    return geometryValid && Number.isSafeInteger(expectedCount) && scales === expectedCount
      ? { status: "valid", reason: `Scale tensor shape and ${scales} affine value(s) match blocked axis ${axis} with block size ${blockSize}.` }
      : { status: "invalid", reason: `Blocked quantization expected shape ${JSON.stringify(expected)} and ${expectedCount ?? "unresolved"} affine value(s); observed scale shape ${JSON.stringify(scaleShape)} and ${scales}/${zeroPoints} scale/zero-point value(s).` };
  }
  if (granularity !== "per_axis" || !Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length || scales !== zeroPoints) {
    return { status: "invalid", reason: "Per-axis scale and zero-point cardinality requires a valid serialized axis." };
  }
  const dimension = Number(shape[axis]);
  if (!Number.isSafeInteger(dimension) || dimension < 0) {
    return { status: "not_assessable", reason: `Shape dimension ${axis} is dynamic or unbound; serialized vectors remain preserved.` };
  }
  return dimension === scales
    ? { status: "valid", reason: `${scales} scale(s) and zero-point(s) index shape dimension ${axis}.` }
    : { status: "invalid", reason: `Expected ${dimension} scale(s) and zero-point(s) for shape dimension ${axis}; found ${scales}/${zeroPoints}.` };
}

function numericContractValues(primary, fallback) {
  const values = Array.isArray(primary) ? primary : Array.isArray(fallback) ? fallback : [];
  return values.map(Number);
}

function quantizationValueShape(values, valueCount) {
  if (Array.isArray(values) && values.length) {
    return values.map((value) => value == null ? null : Number(value));
  }
  return Number.isSafeInteger(valueCount) && valueCount > 0 ? [valueCount] : null;
}

function normalizePreprocessingStatus(value) {
  const status = String(value || "").trim().toLowerCase();
  if (["embedded_in_artifact", "declared_in_artifact", "present"].includes(status)) return "embedded_in_artifact";
  if (["not_applicable"].includes(status)) return "not_applicable";
  return "not_embedded_in_artifact";
}

function preprocessingContractStatus(analysis) {
  const inputs = Array.isArray(analysis?.inputs) ? analysis.inputs : [];
  if (!inputs.length) return "not_applicable";
  const declared = analysis?.metadata_presence?.preprocessing_contract_status;
  return normalizePreprocessingStatus(declared);
}

function positiveIntegerOrNull(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number > 0 ? number : null;
}

function scalarCodeDomain(codeRange, scales, zeroPoints) {
  if (!codeRange || scales.length !== 1 || zeroPoints.length !== 1) return null;
  return {
    qmin: codeRange[0], qmax: codeRange[1],
    real_min: (codeRange[0] - zeroPoints[0]) * scales[0],
    real_max: (codeRange[1] - zeroPoints[0]) * scales[0],
    formula: "real=(q-zero_point)*scale",
  };
}

function dtypeCodeRange(dtype) {
  return ({ UINT8: [0, 255], INT8: [-128, 127], UINT16: [0, 65535], INT16: [-32768, 32767], UINT32: [0, 4294967295], INT32: [-2147483648, 2147483647] })[String(dtype || "").toUpperCase()] || null;
}

function declaredCount(value, fallback) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : fallback;
}

function tensorAt(analysis, index) {
  const numeric = Number(index);
  return Number.isInteger(numeric) && numeric >= 0 ? (analysis?.tensors || [])[numeric] : null;
}

function tensorIndex(tensor) {
  const index = Number.isInteger(tensor?.index) ? tensor.index : Number(tensor?.tensor_index);
  return Number.isInteger(index) && index >= 0 ? index : null;
}

function firstScale(tensor) {
  return Number(tensor?.scale_sample?.[0] || 0);
}

function numericScales(tensor) {
  return (tensor?.scale_sample || []).map(Number).filter((value) => value > 0 && Number.isFinite(value));
}

function opName(op) {
  return String(op?.name || "").toUpperCase();
}

function maxOrZero(values) {
  return values.length ? Math.max(...values.map((value) => Number(value || 0))) : 0;
}
