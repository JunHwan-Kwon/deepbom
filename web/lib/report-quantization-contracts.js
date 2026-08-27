import { ANALYZER_METADATA } from "./report-metadata.js";
import {
  buildBiasScaleCheck,
  buildInputQuantizationConventionCheck,
  buildIoDequantizationCheck,
  buildRepresentableKernelChannelCheck,
} from "./quantization-contract-summary.js";
import { quantResearchLabCoverage } from "./quant-research-applicability.js";

const CONV_LIKE = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]);
const RESIDUAL_SCALE_REVIEW_RATIO = 2;
const INT32_MAX = 2_147_483_647;

export function buildQuantizationContractChecks(analysis) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  if (format === "onnx") return buildOnnxQuantizationContractChecks(analysis);
  if (format !== "tflite") return buildSerializedQuantizationContractChecks(analysis, format);

  const biasScale = buildBiasScaleCheck(analysis);
  const representableKernelChannels = buildRepresentableKernelChannelCheck(analysis);
  const residualAdd = buildResidualAddCheck(analysis);
  const weightZeroPoint = buildWeightZeroPointCheck(analysis);
  const accumulatorBound = gatedTfliteResearchCheck(analysis, "accumulator_atlas", () => buildAccumulatorBoundCheck(analysis));
  const requantizationFidelity = gatedTfliteResearchCheck(analysis, "requantization_fidelity", () => buildRequantizationFidelityCheck(analysis));
  const kernelExtremumWitness = gatedTfliteResearchCheck(analysis, "kernel_extremum_witness", () => buildKernelExtremumWitnessCheck(analysis));
  const channelVitality = gatedTfliteResearchCheck(analysis, "channel_vitality", () => buildChannelVitalityCheck(analysis));
  const roundingEquivalence = gatedTfliteResearchCheck(analysis, "rounding_equivalence", () => buildRoundingEquivalenceCheck(analysis));
  const accumulatorReachability = gatedTfliteResearchCheck(analysis, "accumulator_reachability", () => buildAccumulatorReachabilityCheck(analysis));
  const numericalAbiPropagation = gatedTfliteResearchCheck(analysis, "numerical_abi_propagation", () => buildNumericalAbiPropagationCheck(analysis));
  const inputCounterexample = gatedTfliteResearchCheck(analysis, "input_counterexample", () => buildInputCounterexampleCheck(analysis));
  const preprocessingRealizability = gatedTfliteResearchCheck(analysis, "preprocessing_realizability", () => buildPreprocessingRealizabilityCheck(analysis));
  const contractMigration = gatedTfliteResearchCheck(analysis, "contract_migration", () => buildContractMigrationCheck(analysis));
  const residualStepResponse = gatedTfliteResearchCheck(analysis, "residual_step_response", () => buildResidualStepResponseCheck(analysis));
  const residualContractDistortion = gatedTfliteResearchCheck(analysis, "residual_contract_distortion", () => buildResidualContractDistortionCheck(analysis));
  const ioDequantization = buildIoDequantizationCheck(analysis);
  const inputConvention = buildInputQuantizationConventionCheck(analysis);
  const missingMetadata = buildMissingMetadataCheck(analysis);
  const qdqBoundaries = buildQdqBoundaryCheck(analysis);
  const kernelQuantization = buildKernelQuantizationCheck(analysis);
  const checks = [biasScale, residualAdd, weightZeroPoint, accumulatorBound, requantizationFidelity, kernelExtremumWitness, channelVitality, roundingEquivalence, accumulatorReachability, numericalAbiPropagation, inputCounterexample, preprocessingRealizability, contractMigration, residualStepResponse, residualContractDistortion, ioDequantization, missingMetadata, qdqBoundaries, kernelQuantization];
  const contractIntegrityStatus = checks.some((check) => check.status === "fail")
    ? "fail"
    : checks.some((check) => check.status === "pass")
      ? "pass"
      : "not_applicable";
  const designReviewChecks = [representableKernelChannels, inputConvention, residualAdd, weightZeroPoint, requantizationFidelity, kernelExtremumWitness, channelVitality, roundingEquivalence, accumulatorReachability, numericalAbiPropagation, inputCounterexample, preprocessingRealizability, contractMigration, residualStepResponse, residualContractDistortion, missingMetadata, kernelQuantization];
  const designReviewStatus = designReviewChecks.some((check) => check.status === "review")
    ? "review"
    : designReviewChecks.some((check) => check.status === "pass")
      ? "clear"
      : "not_applicable";

  return {
    schema: ANALYZER_METADATA.schemas.quantizationContractChecks,
    status: contractIntegrityStatus,
    contract_integrity_status: contractIntegrityStatus,
    quantization_design_review_status: designReviewStatus,
    status_semantics: "Contract integrity reports deterministic artifact-contract violations only. Design review reports threshold/configuration signals without changing contract pass/fail.",
    evidence_class: "DERIVED",
    method: "Deterministic graph-edge and quantization-metadata checks over the deployment artifact.",
    bias_scale: biasScale,
    representable_kernel_channels: representableKernelChannels,
    residual_add: residualAdd,
    weight_zero_point: weightZeroPoint,
    accumulator_bound: accumulatorBound,
    requantization_fidelity: requantizationFidelity,
    kernel_extremum_witness: kernelExtremumWitness,
    channel_vitality: channelVitality,
    rounding_equivalence: roundingEquivalence,
    accumulator_reachability: accumulatorReachability,
    numerical_abi_propagation: numericalAbiPropagation,
    input_counterexample: inputCounterexample,
    preprocessing_realizability: preprocessingRealizability,
    contract_migration: contractMigration,
    residual_step_response: residualStepResponse,
    residual_contract_distortion: residualContractDistortion,
    io_dequantization: ioDequantization,
    input_quantization_convention: inputConvention,
    missing_quantization_metadata: missingMetadata,
    qdq_boundaries: qdqBoundaries,
    kernel_quantization: kernelQuantization,
  };
}

function buildSerializedQuantizationContractChecks(analysis, format) {
  const status = analysis?.quantization_status || null;
  const formatContract = format === "safetensors"
    ? analysis?.safetensors?.quantization_contract || null
    : format === "gguf"
      ? analysis?.gguf?.type_traits_source || analysis?.gguf?.backend_compatibility || null
      : format === "coreml"
        ? analysis?.coreml?.quantization_contract || status
        : null;
  const assessment = String(formatContract?.assessment_status || formatContract?.status || status?.assessment_status || status?.status || "not_assessed").toLowerCase();
  const failed = ["fail", "failed", "invalid"].some((token) => assessment.includes(token));
  const ggufAssessed = format === "gguf"
    && analysis?.gguf?.payload_coverage_status === "complete_without_gaps_or_overlaps"
    && Number(status?.unsupported_encoding_tensor_count || 0) === 0
    && /^[0-9a-f]{64}$/.test(String(analysis?.gguf?.type_traits_source?.type_traits_source_sha256 || ""))
    && /^[0-9a-f]{64}$/.test(String(analysis?.gguf?.type_traits_source?.block_layout_source_sha256 || ""));
  const assessed = ggufAssessed || assessment === "assessed" || assessment.startsWith("assessed_") || assessment === "complete" || assessment === "pass";
  const integrityStatus = failed ? "fail" : assessed ? "pass" : "not_assessed";
  return {
    schema: ANALYZER_METADATA.schemas.quantizationContractChecks,
    format_contract: `${format || "serialized"} serialized quantization/storage contract`,
    status: integrityStatus,
    contract_integrity_status: integrityStatus,
    quantization_design_review_status: "not_assessed",
    status_semantics: "Only a normalized format-specific serialized quantization contract can establish storage or numerical quantization state. Missing metadata is NOT_ASSESSED and is not relabeled as floating-point or zero quantization.",
    evidence_class: assessed || failed ? "OBSERVED/DERIVED" : "NOT_ASSESSED",
    method: "Preserve the analyzer-emitted format-specific storage/quantization contract without applying TFLite fixed-point or ONNX Q/DQ rules across formats.",
    serialized_contract: {
      status: ggufAssessed ? "assessed_source_pinned_storage_contract" : assessment,
      quantization_status: status,
      format_contract: formatContract,
      interpretation_boundary: "This contract does not establish executed kernel precision, runtime dequantization/fusion, task accuracy, or deployment performance.",
    },
  };
}

function gatedTfliteResearchCheck(analysis, labId, build) {
  const coverage = quantResearchLabCoverage(analysis, labId);
  return coverage?.class_supported
    ? build()
    : notApplicableCheck(`${coverage?.reason || "This lab is outside the normalized artifact class"} (${coverage?.reason_code || "QR-CLASS-NOT-APPLICABLE"}).`);
}

function buildOnnxQuantizationContractChecks(analysis) {
  const parameterIntegrity = buildOnnxParameterIntegrityCheck(analysis);
  const biasScale = buildOnnxBiasScaleCheck(analysis);
  const representableKernelChannels = notApplicableCheck("TFLite tensor scale vectors are not substituted for ONNX Q/DQ parameter bindings.");
  const residualAdd = notApplicableCheck("Residual ADD scale checks require quantized tensor contracts on both ONNX Add inputs; no QLinearAdd standard operator is assumed.");
  const weightZeroPoint = buildOnnxWeightZeroPointCheck(analysis);
  const accumulatorBound = buildOnnxAccumulatorBoundCheck(analysis);
  const requantizationFidelity = notApplicableCheck("Pinned TFLite Q0.31 requantization encoding does not apply to ONNX execution-provider kernels.");
  const kernelExtremumWitness = notApplicableCheck("Pinned TFLite per-channel integer extremum witnesses do not apply to ONNX execution-provider kernels.");
  const channelVitality = notApplicableCheck("Pinned TFLite fixed-point channel-vitality proofs do not apply to ONNX execution-provider kernels.");
  const roundingEquivalence = notApplicableCheck("Pinned TFLite default/single-rounding interval-hull equivalence proofs do not apply to ONNX execution-provider kernels.");
  const accumulatorReachability = notApplicableCheck("Pinned TFLite bounded-sum accumulator reachability proofs do not apply to ONNX execution-provider kernels.");
  const numericalAbiPropagation = notApplicableCheck("TFLite fixed-point build-mode divergence propagation does not apply to ONNX execution-provider kernels.");
  const inputCounterexample = notApplicableCheck("TFLite fixed-point model-input tensor counterexamples do not apply to ONNX execution-provider kernels.");
  const preprocessingRealizability = notApplicableCheck("TFLite constructive input-witness preprocessing counterfactuals do not apply to ONNX execution-provider kernels.");
  const contractMigration = notApplicableCheck("TFLite residual containment-contract migration does not apply to ONNX execution-provider kernels.");
  const residualStepResponse = notApplicableCheck("TFLite residual local code-step response geometry does not apply to ONNX execution-provider kernels.");
  const residualContractDistortion = notApplicableCheck("TFLite residual containment-contract distortion geometry does not apply to ONNX execution-provider kernels.");
  const ioDequantization = buildIoDequantizationCheck(analysis);
  const missingMetadata = buildOnnxMissingMetadataCheck(analysis);
  const qdqBoundaries = buildOnnxQdqBoundaryCheck(analysis);
  const kernelQuantization = buildOnnxKernelQuantizationCheck(analysis);
  const integrityChecks = [parameterIntegrity, biasScale, weightZeroPoint, accumulatorBound];
  const contractIntegrityStatus = integrityChecks.some((check) => check.status === "fail")
    ? "fail"
    : integrityChecks.some((check) => check.status === "pass") ? "pass" : "not_applicable";
  const designReviewStatus = [weightZeroPoint, kernelQuantization, missingMetadata].some((check) => check.status === "review")
    ? "review"
    : "clear";
  return {
    schema: ANALYZER_METADATA.schemas.quantizationContractChecks,
    format_contract: "ONNX standard-domain QuantizeLinear/DequantizeLinear/QLinearConv/QLinearMatMul",
    status: contractIntegrityStatus,
    contract_integrity_status: contractIntegrityStatus,
    quantization_design_review_status: designReviewStatus,
    status_semantics: "Contract integrity reports deterministic ONNX parameter/signature violations. Runtime QDQ fusion and execution-provider lowering are not inferred from graph syntax.",
    evidence_class: "DERIVED",
    method: "Deterministic binding of embedded ONNX TensorProto scale/zero-point parameters to standard-domain Q/DQ and QLinear signatures.",
    parameter_integrity: parameterIntegrity,
    bias_scale: biasScale,
    representable_kernel_channels: representableKernelChannels,
    residual_add: residualAdd,
    weight_zero_point: weightZeroPoint,
    accumulator_bound: accumulatorBound,
    requantization_fidelity: requantizationFidelity,
    kernel_extremum_witness: kernelExtremumWitness,
    channel_vitality: channelVitality,
    rounding_equivalence: roundingEquivalence,
    accumulator_reachability: accumulatorReachability,
    numerical_abi_propagation: numericalAbiPropagation,
    input_counterexample: inputCounterexample,
    preprocessing_realizability: preprocessingRealizability,
    contract_migration: contractMigration,
    residual_step_response: residualStepResponse,
    residual_contract_distortion: residualContractDistortion,
    io_dequantization: ioDequantization,
    missing_quantization_metadata: missingMetadata,
    qdq_boundaries: qdqBoundaries,
    kernel_quantization: kernelQuantization,
  };
}

function buildOnnxParameterIntegrityCheck(analysis) {
  const binding = analysis?.onnx_quantization_binding || {};
  const details = binding.bindings || [];
  const failures = details.filter((item) => item.status === "fail");
  const unresolved = details.filter((item) => String(item.status || "").startsWith("not_assessed"));
  return {
    status: failures.length ? "fail" : details.length && !unresolved.length ? "pass" : "not_applicable",
    evidence_class: "DERIVED",
    checked_bindings: details.length,
    valid_bindings: details.filter((item) => item.status === "pass").length,
    invalid_bindings: failures.length,
    unresolved_bindings: unresolved.length,
    integer_compute_without_real_scale_count: Number(binding.integer_compute_without_real_scale_count || 0),
    method: binding.method || "ONNX quantization parameter binding was not emitted.",
    details,
  };
}

function buildOnnxBiasScaleCheck(analysis) {
  const details = [];
  let checkedChannels = 0;
  for (const op of analysis?.ops || []) {
    if (op.name !== "QLinearConv" || Number(op.inputs?.[8]) < 0) continue;
    const input = tensorAt(analysis, op.inputs[0]);
    const weight = tensorAt(analysis, op.inputs[3]);
    const bias = tensorAt(analysis, op.inputs[8]);
    const inputScale = onnxQuantScales(analysis, input)[0] || 0;
    const weightScales = onnxQuantScales(analysis, weight);
    if (!(inputScale > 0) || !weightScales.length || !bias) continue;
    const expectedScales = weightScales.map((scale) => inputScale * scale);
    const biasElements = elementCount(bias.shape);
    const outputChannels = Number(weight?.shape?.[0] || expectedScales.length);
    const dtypeValid = String(bias.dtype || "").toUpperCase() === "INT32";
    const cardinalityValid = biasElements === outputChannels;
    checkedChannels += expectedScales.length;
    details.push({
      op_index: op.index,
      op_name: op.name,
      input_tensor_index: tensorIndex(input),
      weight_tensor_index: tensorIndex(weight),
      bias_tensor_index: tensorIndex(bias),
      checked_channels: expectedScales.length,
      expected_bias_scales: expectedScales.slice(0, 256),
      expected_scale_count: expectedScales.length,
      bias_dtype: bias.dtype,
      bias_element_count: biasElements,
      output_channel_count: outputChannels,
      status: dtypeValid && cardinalityValid ? "pass" : "fail",
      reasons: [!dtypeValid ? "QLinearConv bias must be INT32" : "", !cardinalityValid ? "bias element count must equal output channels" : ""].filter(Boolean),
      formula: "implicit_bias_scale[channel] = x_scale * w_scale[channel]",
      contract_note: "ONNX QLinearConv stores INT32 bias values without a separate bias-scale TensorProto; the real-value scale is implicit in the operator contract.",
      evidence_json_pointer: `/evidence/static_analysis/ops/${op.index}`,
    });
  }
  const failures = details.filter((item) => item.status === "fail");
  return {
    status: details.length ? (failures.length ? "fail" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    contract_kind: "onnx_implicit_qlinearconv_bias_scale",
    checked_groups: details.length,
    checked_channels: checkedChannels,
    mismatch_groups: failures.length,
    maximum_relative_error: null,
    relative_tolerance: null,
    method: "implicit_bias_scale[channel] = x_scale * w_scale[channel]; verify INT32 dtype and output-channel cardinality",
    details,
  };
}

function buildOnnxWeightZeroPointCheck(analysis) {
  const details = [];
  for (const weight of onnxQuantizedKernelTensors(analysis)) {
    const zeroPoints = onnxQuantZeroPoints(analysis, weight);
    const dtype = String(weight.dtype || "").toUpperCase();
    const legalRange = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : null;
    const outOfRange = !legalRange || zeroPoints.some((value) => value < legalRange[0] || value > legalRange[1] || !Number.isInteger(value));
    const symmetricInt8Violation = false;
    const asymmetricUint8 = dtype === "UINT8" && zeroPoints.some((value) => value !== 0);
    details.push({
      tensor_index: tensorIndex(weight), tensor_name: weight.name || "", dtype, zero_points: zeroPoints,
      legal_range: legalRange, out_of_range: outOfRange, symmetric_int8_violation: symmetricInt8Violation,
      asymmetric_uint8_observed: asymmetricUint8,
      status: outOfRange ? "fail" : "pass",
      formula: "qmin <= zero_point <= qmax; ONNX QLinear signed weights are not assumed symmetric",
      evidence_json_pointer: `/evidence/static_analysis/tensors/${tensorIndex(weight)}`,
    });
  }
  const failures = details.filter((item) => item.status === "fail");
  const reviews = details.filter((item) => item.status === "review");
  return {
    status: details.length ? (failures.length ? "fail" : reviews.length ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED", checked_tensors: details.length, violation_tensors: failures.length,
    asymmetric_uint8_tensors: details.filter((item) => item.asymmetric_uint8_observed).length, converter_lineage_status: "not_assessable_from_artifact", details,
  };
}

function buildOnnxAccumulatorBoundCheck(analysis) {
  const details = [];
  for (const op of analysis?.ops || []) {
    if (!["QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger"].includes(op.name)) continue;
    const input = tensorAt(analysis, op.inputs?.[0]);
    const weight = tensorAt(analysis, op.inputs?.[["QLinearConv", "QLinearMatMul"].includes(op.name) ? 3 : 1]);
    const terms = onnxAccumulationTerms(op, weight);
    const inputBound = onnxQuantMagnitudeBound(analysis, input);
    const weightBound = onnxQuantMagnitudeBound(analysis, weight);
    if (!(terms > 0) || !(inputBound > 0) || !(weightBound > 0)) continue;
    const worst = terms * inputBound * weightBound;
    const ratio = worst / INT32_MAX;
    details.push({
      op_index: op.index, op_name: op.name, input_tensor_index: tensorIndex(input), weight_tensor_index: tensorIndex(weight),
      accumulation_terms: terms, input_integer_magnitude_bound: inputBound, weight_integer_magnitude_bound: weightBound,
      worst_case_absolute_accumulator: worst, int32_max: INT32_MAX, int32_ratio: ratio,
      status: ratio >= 1 ? "fail" : "pass",
      formula: "accumulation_terms * max_abs(q_input-zp_input) * max_abs(q_weight-zp_weight) / INT32_MAX",
      evidence_json_pointer: `/evidence/static_analysis/ops/${op.index}`,
    });
  }
  const failures = details.filter((item) => item.status === "fail");
  return { status: details.length ? (failures.length ? "fail" : "pass") : "not_applicable", evidence_class: "DERIVED", checked_ops: details.length, overflow_risk_ops: failures.length, maximum_int32_ratio: maxOrZero(details.map((item) => item.int32_ratio)), accumulator_dtype: "INT32", details };
}

function buildOnnxMissingMetadataCheck(analysis) {
  const binding = analysis?.onnx_quantization_binding || {};
  const details = (binding.bindings || []).filter((item) => item.status !== "pass").map((item) => ({
    op_index: item.op_index, op_name: item.op_name, tensor_name: item.tensor_name, role: item.role,
    classification: item.status === "fail" ? "invalid_quantization_parameter_contract" : "quantization_parameter_not_assessed",
    status: item.status === "fail" ? "fail" : "review", reasons: item.reasons || [],
    evidence_json_pointer: `/evidence/static_analysis/onnx_quantization_binding/bindings/${(binding.bindings || []).indexOf(item)}`,
  }));
  return {
    status: details.some((item) => item.status === "fail") ? "fail" : details.length ? "review" : (binding.binding_count ? "pass" : "not_applicable"),
    evidence_class: "DERIVED", tensors_without_metadata: details.length, review_tensors: details.filter((item) => item.status === "review").length,
    not_applicable_tensors: 0, details,
  };
}

function buildOnnxQdqBoundaryCheck(analysis) {
  const binding = analysis?.onnx_quantization_binding || {};
  return {
    status: "pass",
    evidence_class: "OBSERVED",
    explicit_qdq_ops: Number(binding.explicit_qdq_boundary_count || 0),
    mid_graph_quantization_holes: null,
    runtime_materialization_status: "not_assessed_static_graph_only",
    details: binding.boundary_edges || [],
  };
}

function buildOnnxKernelQuantizationCheck(analysis) {
  const entries = onnxQuantizedKernelTensors(analysis);
  const scales = entries.flatMap((weight) => onnxQuantScales(analysis, weight));
  return {
    status: entries.length ? "pass" : "not_applicable", evidence_class: "DERIVED", kernel_tensors: entries.length,
    per_tensor_kernel_tensors: entries.filter((weight) => Number(weight.quant_scales || 0) === 1).length,
    per_channel_kernel_tensors: entries.filter((weight) => Number(weight.quant_scales || 0) > 1).length,
    depthwise_per_tensor_kernel_tensors: 0,
    minimum_weight_scale: scales.length ? Math.min(...scales) : null,
    maximum_weight_scale: scales.length ? Math.max(...scales) : null,
    maximum_to_minimum_scale_ratio: scales.length ? Math.max(...scales) / Math.min(...scales) : null,
  };
}

function onnxQuantizedKernelTensors(analysis) {
  const indices = new Set();
  const producerByTensor = new Map();
  for (const candidate of analysis?.ops || []) for (const output of candidate.outputs || []) if (Number(output) >= 0) producerByTensor.set(Number(output), candidate);
  for (const op of analysis?.ops || []) {
    if (["QLinearConv", "QLinearMatMul"].includes(op.name) && Number(op.inputs?.[3]) >= 0) indices.add(Number(op.inputs[3]));
    if (!["Conv", "Gemm", "MatMul"].includes(op.name)) continue;
    const candidate = producerByTensor.get(Number(op.inputs?.[1]));
    const producer = candidate?.name === "DequantizeLinear" ? candidate : null;
    if (producer && Number(producer.inputs?.[0]) >= 0) indices.add(Number(producer.inputs[0]));
  }
  return [...indices].map((index) => tensorAt(analysis, index)).filter((tensor) => tensor && Number(tensor.quant_scales || 0) > 0);
}

function onnxAccumulationTerms(op, weight) {
  const shape = (weight?.shape || []).map((dim) => Math.max(1, Number(dim) || 1));
  if (["QLinearConv", "ConvInteger"].includes(op.name) && shape.length >= 4) return shape.slice(1).reduce((product, dim) => product * dim, 1);
  return shape.length >= 2 ? shape.at(-2) : 0;
}

function onnxQuantBinding(analysis, tensor) {
  return (analysis?.onnx_quantization_binding?.bindings || []).find((item) => item.tensor_name === tensor?.name && item.scale_values?.length) || null;
}

function onnxQuantScales(analysis, tensor) {
  return (onnxQuantBinding(analysis, tensor)?.scale_values || tensor?.scale_sample || []).map(Number).filter((value) => value > 0 && Number.isFinite(value));
}

function onnxQuantZeroPoints(analysis, tensor) {
  return (onnxQuantBinding(analysis, tensor)?.zero_point_values || tensor?.zero_point_sample || []).map(Number);
}

function onnxQuantMagnitudeBound(analysis, tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const [lower, upper] = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : [0, 0];
  if (lower === 0 && upper === 0) return 0;
  const zeroPoints = onnxQuantZeroPoints(analysis, tensor);
  const values = zeroPoints.length ? zeroPoints : [0];
  return Math.max(...values.map((zeroPoint) => Math.max(Math.abs(lower - zeroPoint), Math.abs(upper - zeroPoint))));
}

function notApplicableCheck(reason) {
  return { status: "not_applicable", evidence_class: "NOT_APPLICABLE", reason, checked_ops: 0, review_ops: 0, maximum_input_scale_ratio: 0, review_threshold_ratio: RESIDUAL_SCALE_REVIEW_RATIO, details: [] };
}

function buildResidualAddCheck(analysis) {
  const details = [];
  const latticeRows = analysis?.quantization_lattice?.residual_adds || [];
  const latticeByOp = new Map(latticeRows.map((row, index) => [Number(row.op_index), { row, index }]));
  for (const op of analysis?.ops || []) {
    if (opName(op) !== "ADD") continue;
    const input0 = tensorAt(analysis, op.inputs?.[0]);
    const input1 = tensorAt(analysis, op.inputs?.[1]);
    const output = tensorAt(analysis, op.outputs?.[0]);
    const input0Scale = firstScale(input0);
    const input1Scale = firstScale(input1);
    if (!(input0Scale > 0) || !(input1Scale > 0)) continue;
    const ratio = Math.max(input0Scale, input1Scale) / Math.min(input0Scale, input1Scale);
    const lattice = latticeByOp.get(Number(op.index)) || null;
    const latticeRow = lattice?.row?.assessment_status === "assessed" ? lattice.row : null;
    details.push({
      op_index: op.index,
      op_name: op.name,
      input_tensor_indices: [tensorIndex(input0), tensorIndex(input1)],
      output_tensor_index: tensorIndex(output),
      input_scales: [input0Scale, input1Scale],
      output_scale: firstScale(output) || null,
      input_scale_ratio: ratio,
      status: ratio >= RESIDUAL_SCALE_REVIEW_RATIO ? "review" : "pass",
      review_threshold_ratio: RESIDUAL_SCALE_REVIEW_RATIO,
      exhaustive_legal_code_pair_count: latticeRow?.enumerated_code_pair_count ?? null,
      legal_domain_escape_pair_count: latticeRow?.range_escape_pair_count ?? null,
      legal_domain_escape_pair_ratio: latticeRow?.range_escape_pair_ratio ?? null,
      fixed_zero_point_containment_scale_ratio: latticeRow?.fixed_zero_point_containment?.scale_ratio_to_current ?? null,
      globally_finest_containment_scale_ratio: latticeRow?.globally_finest_containment?.scale_ratio_to_current ?? null,
      globally_finest_containment_zero_point_delta: latticeRow?.globally_finest_containment?.signed_zero_point_delta ?? null,
      globally_finest_containment_clamp_pair_count: latticeRow?.globally_finest_containment?.rounded_projection_clamp_pair_count ?? null,
      formula: "max(input_scale_0,input_scale_1) / min(input_scale_0,input_scale_1)",
      evidence_json_pointer: lattice
        ? `/evidence/static_analysis/quantization_lattice/residual_adds/${lattice.index}`
        : `/evidence/static_analysis/ops/${op.index}`,
    });
  }
  const reviewOps = details.filter((item) => item.status === "review");
  return {
    status: details.length ? (reviewOps.length ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_ops: details.length,
    review_ops: reviewOps.length,
    maximum_input_scale_ratio: maxOrZero(details.map((item) => item.input_scale_ratio)),
    containment_design_ops: details.filter((item) => item.globally_finest_containment_scale_ratio != null).length,
    maximum_global_finest_containment_scale_ratio: maxNullable(details.map((item) => item.globally_finest_containment_scale_ratio)),
    review_threshold_ratio: RESIDUAL_SCALE_REVIEW_RATIO,
    threshold_class: "HEURISTIC",
    details,
  };
}

function buildContractMigrationCheck(analysis) {
  const migration = analysis?.contract_migration;
  if (!migration) return notApplicableCheck("Residual containment-contract migration analysis was not emitted.");
  const details = (migration.migrations || []).flatMap((row, migrationIndex) =>
    (row.scenarios || []).map((scenario, scenarioIndex) => ({
      source_add_op_index: row.source_add_op_index,
      output_tensor_index: row.output_tensor_index,
      design: scenario.design,
      candidate_output_scale: scenario.candidate_output_scale,
      candidate_output_zero_point: scenario.candidate_output_zero_point,
      scale_ratio_to_current: scenario.scale_ratio_to_current,
      signed_zero_point_delta: scenario.signed_zero_point_delta,
      direct_consumer_count: row.direct_consumer_count,
      assessed_consumer_count: scenario.assessed_consumer_count,
      unassessed_consumer_count: scenario.unassessed_consumer_count,
      assessed_kernel_channels: scenario.assessed_kernel_channel_count,
      changed_multiplier_encodings: scenario.multiplier_encoding_changed_channel_count,
      changed_multiplier_shifts: scenario.multiplier_shift_changed_channel_count,
      changed_bias_codes: scenario.bias_code_changed_channel_count,
      bias_int32_overflow_channels: scenario.bias_int32_overflow_channel_count,
      changed_add_parameter_encodings: scenario.add_parameter_encoding_changed_count,
      reachable_downstream_ops: row.reachable_downstream_op_count,
      maximum_downstream_edge_depth: row.maximum_downstream_edge_depth,
      status: scenario.unassessed_consumer_count || scenario.bias_int32_overflow_channel_count ? "review" : "pass",
      evidence_json_pointer: `/evidence/static_analysis/contract_migration/migrations/${migrationIndex}/scenarios/${scenarioIndex}`,
    })));
  const review = details.filter((item) => item.status === "review");
  return {
    status: details.length ? (review.length ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_residual_contracts: migration.residual_contract_count,
    checked_candidate_scenarios: details.length,
    direct_consumers: migration.direct_consumer_count,
    assessed_consumer_scenarios: migration.assessed_consumer_scenario_count,
    unassessed_consumer_scenarios: migration.unassessed_consumer_scenario_count,
    assessed_kernel_channel_scenarios: migration.assessed_kernel_channel_scenario_count,
    changed_multiplier_encoding_channel_scenarios: migration.multiplier_encoding_changed_channel_scenario_count,
    changed_multiplier_shift_channel_scenarios: migration.multiplier_shift_changed_channel_scenario_count,
    changed_bias_code_channel_scenarios: migration.bias_code_changed_channel_scenario_count,
    bias_int32_overflow_channel_scenarios: migration.bias_int32_overflow_channel_scenario_count,
    changed_add_parameter_encoding_scenarios: migration.add_parameter_encoding_changed_scenario_count,
    structural_downstream_op_union: migration.reachable_downstream_op_union_count,
    maximum_downstream_edge_depth: migration.maximum_downstream_edge_depth,
    bound_class: "counterfactual_reexport_parameter_regeneration",
    method: migration.method,
    interpretation_boundary: migration.interpretation_boundary,
    details,
  };
}

function buildResidualStepResponseCheck(analysis) {
  const response = analysis?.residual_step_response;
  if (!response) return notApplicableCheck("Residual step-response analysis was not emitted.");
  const details = (response.residual_adds || []).flatMap((row, rowIndex) =>
    (row.contracts || []).map((contract, contractIndex) => ({
      op_index: row.op_index,
      design: contract.design,
      output_scale: contract.output_scale,
      output_zero_point: contract.output_zero_point,
      rounded_projection_clamp_pair_count: contract.rounded_projection_clamp_pair_count,
      visible_transition_count: contract.visible_transition_count,
      silent_transition_count: contract.silent_transition_count,
      both_branches_visible_cell_count: contract.both_branches_visible_cell_count,
      neither_branch_visible_cell_count: contract.neither_branch_visible_cell_count,
      removed_rounded_clamp_pairs_vs_current: contract.removed_rounded_clamp_pairs_vs_current,
      additional_silent_transitions_vs_current: contract.additional_silent_transitions_vs_current,
      transition_ledger_sha256: contract.transition_ledger_sha256,
      evidence_json_pointer: `/evidence/static_analysis/residual_step_response/residual_adds/${rowIndex}/contracts/${contractIndex}`,
    })));
  return {
    status: response.assessed_add_count ? (response.unassessed_add_count ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_residual_adds: response.assessed_add_count,
    checked_contracts: response.contract_response_count,
    exact_branch_transitions: response.total_transition_count,
    exact_joint_interior_cells: response.total_joint_interior_cell_count,
    current_silent_transitions: response.current_silent_transition_count,
    containment_additional_silent_transitions: response.containment_additional_silent_transition_count,
    containment_removed_rounded_clamp_pairs: response.containment_removed_rounded_clamp_pair_count,
    maximum_containment_silent_ratio_increase: response.maximum_containment_silent_ratio_increase,
    bound_class: "uniform_legal_code_local_distinguishability",
    method: response.method,
    interpretation_boundary: response.interpretation_boundary,
    details,
  };
}

function buildResidualContractDistortionCheck(analysis) {
  const distortion = analysis?.residual_contract_distortion;
  if (!distortion) return notApplicableCheck("Residual contract-distortion analysis was not emitted.");
  const details = (distortion.residual_adds || []).flatMap((row, rowIndex) =>
    (row.scenarios || []).map((scenario, scenarioIndex) => ({
      op_index: row.op_index,
      design: scenario.design,
      candidate_output_scale: scenario.candidate_output_scale,
      candidate_output_zero_point: scenario.candidate_output_zero_point,
      enumerated_pair_count: scenario.enumerated_pair_count,
      rescued_current_clamp_pair_count: scenario.rescued_current_clamp_pair_count,
      changed_represented_value_pair_count: scenario.changed_represented_value_pair_count,
      ideal_error_improved_pair_count: scenario.ideal_error_improved_pair_count,
      ideal_error_worsened_pair_count: scenario.ideal_error_worsened_pair_count,
      sign_class_changed_pair_count: scenario.sign_class_changed_pair_count,
      root_mean_square_contract_delta_current_steps: scenario.root_mean_square_contract_delta_current_steps,
      p99_absolute_contract_delta_current_steps: scenario.p99_absolute_contract_delta_current_steps,
      pair_ledger_sha256: scenario.pair_ledger_sha256,
      evidence_json_pointer: `/evidence/static_analysis/residual_contract_distortion/residual_adds/${rowIndex}/scenarios/${scenarioIndex}`,
    })));
  return {
    status: distortion.assessed_add_count ? (distortion.unassessed_add_count ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_residual_adds: distortion.assessed_add_count,
    checked_candidate_scenarios: distortion.scenario_count,
    exact_pair_comparisons: distortion.total_enumerated_pair_count,
    rescued_current_clamp_pair_instances: distortion.rescued_current_clamp_pair_instance_count,
    candidate_clamped_pairs: distortion.candidate_clamped_pair_count,
    changed_represented_value_pairs: distortion.changed_represented_value_pair_count,
    ideal_error_improved_pairs: distortion.ideal_error_improved_pair_count,
    ideal_error_worsened_pairs: distortion.ideal_error_worsened_pair_count,
    ideal_error_equal_within_tolerance_pairs: distortion.ideal_error_equal_within_tolerance_pair_count,
    sign_class_changed_pairs: distortion.sign_class_changed_pair_count,
    maximum_rms_contract_delta_current_steps: distortion.maximum_rms_contract_delta_current_steps,
    maximum_p99_contract_delta_current_steps: distortion.maximum_p99_contract_delta_current_steps,
    bound_class: "uniform_legal_code_contract_counterfactual",
    method: distortion.method,
    interpretation_boundary: distortion.interpretation_boundary,
    details,
  };
}

function buildWeightZeroPointCheck(analysis) {
  const details = [];
  const seen = new Set();
  for (const op of analysis?.ops || []) {
    if (!CONV_LIKE.has(opName(op))) continue;
    const weight = tensorAt(analysis, op.inputs?.[1]);
    const index = tensorIndex(weight);
    if (index == null || seen.has(index) || !numericScales(weight).length) continue;
    seen.add(index);
    const dtype = String(weight?.dtype || "").toUpperCase();
    const zeroPoints = (weight?.zero_point_sample || []).map(Number);
    const legalRange = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : null;
    const outOfRange = legalRange ? zeroPoints.some((value) => value < legalRange[0] || value > legalRange[1]) : false;
    const symmetricInt8Violation = dtype === "INT8" && zeroPoints.some((value) => value !== 0);
    const asymmetricUint8 = dtype === "UINT8";
    details.push({
      tensor_index: index,
      tensor_name: weight?.name || "",
      dtype,
      zero_points: zeroPoints,
      legal_range: legalRange,
      out_of_range: outOfRange,
      symmetric_int8_violation: symmetricInt8Violation,
      asymmetric_uint8_observed: asymmetricUint8,
      status: outOfRange || symmetricInt8Violation ? "fail" : asymmetricUint8 ? "review" : "pass",
      formula: dtype === "INT8" ? "weight_zero_point[channel] == 0" : "qmin <= zero_point <= qmax",
      evidence_json_pointer: `/evidence/static_analysis/tensors/${index}`,
    });
  }
  const violations = details.filter((item) => item.status === "fail");
  const asymmetric = details.filter((item) => item.asymmetric_uint8_observed);
  return {
    status: details.length ? (violations.length ? "fail" : asymmetric.length ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_tensors: details.length,
    violation_tensors: violations.length,
    asymmetric_uint8_tensors: asymmetric.length,
    converter_lineage_status: asymmetric.length ? "not_assessable_from_artifact" : "not_applicable",
    details,
  };
}

function buildAccumulatorBoundCheck(analysis) {
  const atlas = String(analysis?.format || "tflite").toLowerCase() === "tflite"
    ? analysis?.accumulator_atlas
    : null;
  if (atlas?.assessed_op_count) {
    const details = (atlas.ops || [])
      .map((row, rowIndex) => row.assessment_status === "assessed" ? ({
        op_index: row.op_index,
        op_name: row.op_name,
        input_tensor_index: row.input_tensor_index,
        weight_tensor_index: row.weight_tensor_index,
        weight_tensor_name: row.weight_tensor_name,
        accumulation_terms: row.accumulation_terms_per_channel,
        assessed_channels: row.assessed_channel_count,
        overflow_channels: row.int32_overflow_channel_count,
        overflow_channel_indices: [...(row.overflow_channel_indices || [])],
        maximum_absolute_accumulator_decimal: row.maximum_absolute_accumulator_decimal,
        int32_max: INT32_MAX,
        int32_ratio: row.maximum_int32_ratio,
        required_signed_bits: row.maximum_required_signed_bits,
        int32_headroom_bits: row.minimum_int32_headroom_bits,
        metadata_only_magnitude_bound_decimal: row.metadata_only_magnitude_bound_decimal,
        metadata_only_int32_ratio: row.metadata_only_int32_ratio,
        exact_tightening_factor: row.exact_tightening_factor,
        status: row.int32_overflow_channel_count ? "fail" : "pass",
        formula: row.formula,
        channel_ledger_sha256: row.channel_ledger_sha256,
        evidence_json_pointer: `/evidence/static_analysis/accumulator_atlas/ops/${rowIndex}`,
      }) : null)
      .filter(Boolean);
    return {
      status: atlas.int32_overflow_channel_count ? "fail" : "pass",
      evidence_class: "DERIVED",
      bound_class: "exact_stored_weight_channel_integer_domain",
      checked_ops: atlas.assessed_op_count,
      checked_channels: atlas.assessed_channel_count,
      overflow_risk_ops: atlas.overflow_op_count,
      overflow_risk_channels: atlas.int32_overflow_channel_count,
      maximum_int32_ratio: atlas.maximum_int32_ratio,
      maximum_absolute_accumulator_decimal: atlas.maximum_absolute_accumulator_decimal,
      maximum_required_signed_bits: atlas.maximum_required_signed_bits,
      minimum_int32_headroom_bits: atlas.minimum_int32_headroom_bits,
      maximum_metadata_only_int32_ratio: maxOrZero(details.map((item) => item.metadata_only_int32_ratio)),
      accumulator_dtype: "INT32",
      source_commit: atlas.source_commit,
      method: atlas.method,
      interpretation_boundary: atlas.interpretation_boundary,
      details,
    };
  }
  const details = [];
  for (const op of analysis?.ops || []) {
    if (!CONV_LIKE.has(opName(op))) continue;
    const input = tensorAt(analysis, op.inputs?.[0]);
    const weight = tensorAt(analysis, op.inputs?.[1]);
    const terms = kernelAccumulationTerms(op, weight);
    const inputMagnitudeBound = quantMagnitudeBound(input);
    const weightMagnitudeBound = quantMagnitudeBound(weight);
    if (!(terms > 0) || !(inputMagnitudeBound > 0) || !(weightMagnitudeBound > 0)) continue;
    const worstCaseAbsoluteAccumulator = terms * inputMagnitudeBound * weightMagnitudeBound;
    const ratio = worstCaseAbsoluteAccumulator / INT32_MAX;
    details.push({
      op_index: op.index,
      op_name: op.name,
      input_tensor_index: tensorIndex(input),
      weight_tensor_index: tensorIndex(weight),
      accumulation_terms: terms,
      input_integer_magnitude_bound: inputMagnitudeBound,
      weight_integer_magnitude_bound: weightMagnitudeBound,
      worst_case_absolute_accumulator: worstCaseAbsoluteAccumulator,
      int32_max: INT32_MAX,
      int32_ratio: ratio,
      status: ratio >= 1 ? "fail" : "pass",
      formula: "accumulation_terms * max_abs(q_input-zp_input) * max_abs(q_weight-zp_weight) / INT32_MAX",
      evidence_json_pointer: `/evidence/static_analysis/ops/${op.index}`,
    });
  }
  const risks = details.filter((item) => item.status === "fail");
  return {
    status: details.length ? (risks.length ? "fail" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    checked_ops: details.length,
    overflow_risk_ops: risks.length,
    maximum_int32_ratio: maxOrZero(details.map((item) => item.int32_ratio)),
    bound_class: "metadata_only_legal_code_magnitude_bound",
    accumulator_dtype: "INT32",
    details,
  };
}

function buildRequantizationFidelityCheck(analysis) {
  const fidelity = analysis?.requantization_fidelity;
  if (!fidelity?.assessed_op_count) {
    return notApplicableCheck("No accumulator-bounded quantized TFLite Conv/Depthwise/FC channel exposed a complete input, weight, and output scale contract.");
  }
  const details = (fidelity.ops || []).map((row, rowIndex) => row.assessment_status === "assessed" ? ({
    op_index: row.op_index,
    op_name: row.op_name,
    input_tensor_index: row.input_tensor_index,
    weight_tensor_index: row.weight_tensor_index,
    output_tensor_index: row.output_tensor_index,
    weight_scale_mode: row.weight_scale_mode,
    assessed_channels: row.assessed_channel_count,
    shift_range: [row.minimum_shift, row.maximum_shift],
    maximum_relative_multiplier_error: row.maximum_relative_multiplier_error,
    maximum_multiplier_error_ppm: row.maximum_multiplier_error_ppm,
    maximum_encoding_drift_bound_codes: row.maximum_encoding_drift_bound_codes,
    maximum_default_double_rounding_bound_codes: row.maximum_default_double_rounding_bound_codes,
    maximum_single_rounding_bound_codes: row.maximum_single_rounding_bound_codes,
    default_pre_shift_overflow_channels: row.default_pre_shift_overflow_channel_count,
    single_rounding_encoding_divergence_channels: row.single_rounding_encoding_divergence_channel_count,
    channel_ledger_sha256: row.channel_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/requantization_fidelity/ops/${rowIndex}`,
  }) : null).filter(Boolean);
  const review = fidelity.default_pre_shift_overflow_channel_count > 0
    || fidelity.single_rounding_encoding_divergence_channel_count > 0;
  return {
    status: review ? "review" : "pass",
    evidence_class: "DERIVED",
    bound_class: "pinned_q0_31_encoding_and_conservative_pre_clamp_rounding",
    checked_ops: fidelity.assessed_op_count,
    checked_channels: fidelity.assessed_channel_count,
    fixed_point_bound_channels: fidelity.fixed_point_bound_channel_count,
    build_flag_status: "TFLITE_SINGLE_ROUNDING_not_embedded_in_artifact",
    default_pre_shift_overflow_channels: fidelity.default_pre_shift_overflow_channel_count,
    single_rounding_encoding_divergence_channels: fidelity.single_rounding_encoding_divergence_channel_count,
    half_code_encoding_drift_channels: fidelity.half_code_encoding_drift_channel_count,
    one_code_encoding_drift_channels: fidelity.one_code_encoding_drift_channel_count,
    shift_range: [fidelity.minimum_shift, fidelity.maximum_shift],
    maximum_relative_multiplier_error: fidelity.maximum_relative_multiplier_error,
    maximum_multiplier_error_ppm: fidelity.maximum_multiplier_error_ppm,
    maximum_encoding_drift_bound_codes: fidelity.maximum_encoding_drift_bound_codes,
    maximum_default_double_rounding_bound_codes: fidelity.maximum_default_double_rounding_bound_codes,
    maximum_single_rounding_bound_codes: fidelity.maximum_single_rounding_bound_codes,
    source_commit: fidelity.source_commit,
    source_references: fidelity.source_references,
    formula: fidelity.rounding_bound_formula,
    interpretation_boundary: fidelity.interpretation_boundary,
    details,
  };
}

function buildKernelExtremumWitnessCheck(analysis) {
  const witness = analysis?.kernel_extremum_witness;
  if (!witness?.assessed_op_count) {
    return notApplicableCheck("No quantized convolution-family channel exposed a complete stored-weight extremum and pinned fixed-point projection contract.");
  }
  const details = (witness.ops || []).map((row, rowIndex) => row.assessment_status === "assessed" ? ({
    op_index: row.op_index,
    op_name: row.op_name,
    assessed_channels: row.assessed_channel_count,
    terms_per_channel: row.accumulation_terms_per_channel,
    witness_assignments: row.witness_assignment_count,
    fixed_point_endpoint_evaluations: row.fixed_point_endpoint_evaluation_count,
    default_ideal_mismatch_endpoints: row.default_ideal_mismatch_endpoint_count,
    single_ideal_mismatch_endpoints: row.single_ideal_mismatch_endpoint_count,
    build_mode_divergent_endpoints: row.build_mode_divergent_endpoint_count,
    default_activation_clamped_endpoints: row.default_activation_clamped_endpoint_count,
    collapsed_extrema_channels: row.default_collapsed_extrema_channel_count,
    maximum_default_ideal_delta_codes: row.maximum_default_ideal_delta_codes,
    witness_ledger_sha256: row.witness_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/kernel_extremum_witness/ops/${rowIndex}`,
  }) : null).filter(Boolean);
  return {
    status: witness.build_mode_divergent_endpoint_count || witness.unassessed_op_count ? "review" : "pass",
    evidence_class: "DERIVED",
    bound_class: "canonical_full_valid_receptive_field_legal_code_extrema",
    checked_ops: witness.assessed_op_count,
    checked_channels: witness.assessed_channel_count,
    fixed_point_checked_channels: witness.fixed_point_assessed_channel_count,
    canonical_witness_assignments: witness.witness_assignment_count,
    fixed_point_endpoint_evaluations: witness.fixed_point_endpoint_evaluation_count,
    default_ideal_mismatch_endpoints: witness.default_ideal_mismatch_endpoint_count,
    single_ideal_mismatch_endpoints: witness.single_ideal_mismatch_endpoint_count,
    build_mode_divergent_endpoints: witness.build_mode_divergent_endpoint_count,
    default_activation_clamped_endpoints: witness.default_activation_clamped_endpoint_count,
    collapsed_extrema_channels: witness.default_collapsed_extrema_channel_count,
    maximum_default_ideal_delta_codes: witness.maximum_default_ideal_delta_codes,
    build_flag_status: "TFLITE_SINGLE_ROUNDING_not_embedded_in_artifact",
    source_commit: witness.source_commit,
    source_references: witness.source_references,
    method: witness.method,
    interpretation_boundary: witness.interpretation_boundary,
    details,
  };
}

function buildChannelVitalityCheck(analysis) {
  const vitality = analysis?.channel_vitality;
  if (!vitality?.assessed_op_count) {
    return notApplicableCheck("No quantized convolution-family channel exposed complete exact endpoints under both pinned fixed-point build paths.");
  }
  const details = (vitality.ops || []).map((row, rowIndex) => row.assessment_status === "assessed" ? ({
    op_index: row.op_index,
    op_name: row.op_name,
    assessed_channels: row.assessed_channel_count,
    constant_accumulator_channels: row.constant_accumulator_channel_count,
    dual_mode_constant_output_channels: row.dual_mode_constant_output_channel_count,
    nonconstant_accumulator_dual_mode_constant_channels: row.nonconstant_accumulator_dual_mode_constant_channel_count,
    mode_dependent_constant_output_channels: row.mode_dependent_constant_output_channel_count,
    minimum_default_inclusive_code_span: row.minimum_default_inclusive_code_span,
    minimum_single_inclusive_code_span: row.minimum_single_inclusive_code_span,
    default_constant_channel_indices: row.default_constant_channel_indices,
    single_constant_channel_indices: row.single_constant_channel_indices,
    mode_dependent_constant_channel_indices: row.mode_dependent_constant_channel_indices,
    source_witness_ledger_sha256: row.source_witness_ledger_sha256,
    vitality_ledger_sha256: row.vitality_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/channel_vitality/ops/${rowIndex}`,
  }) : null).filter(Boolean);
  return {
    status: vitality.nonconstant_accumulator_dual_mode_constant_channel_count
      || vitality.mode_dependent_constant_output_channel_count
      || vitality.unassessed_op_count ? "review" : "pass",
    evidence_class: "DERIVED",
    proof_domain: "full_valid_receptive_field_legal_input_code_domain",
    checked_ops: vitality.assessed_op_count,
    checked_channels: vitality.assessed_channel_count,
    fixed_point_checked_channels: vitality.fixed_point_assessed_channel_count,
    constant_accumulator_channels: vitality.constant_accumulator_channel_count,
    dual_mode_constant_output_channels: vitality.dual_mode_constant_output_channel_count,
    nonconstant_accumulator_dual_mode_constant_channels: vitality.nonconstant_accumulator_dual_mode_constant_channel_count,
    mode_dependent_constant_output_channels: vitality.mode_dependent_constant_output_channel_count,
    post_bias_negative_locked_channels: vitality.post_bias_negative_locked_channel_count,
    post_bias_positive_locked_channels: vitality.post_bias_positive_locked_channel_count,
    default_full_activation_span_channels: vitality.default_full_activation_span_channel_count,
    single_full_activation_span_channels: vitality.single_full_activation_span_channel_count,
    span_histogram: vitality.span_histogram,
    source_commit: vitality.source_commit,
    source_references: vitality.source_references,
    constant_proof: vitality.constant_proof,
    span_definition: vitality.span_definition,
    method: vitality.method,
    interpretation_boundary: vitality.interpretation_boundary,
    details,
  };
}

function buildRoundingEquivalenceCheck(analysis) {
  const equivalence = analysis?.rounding_equivalence;
  if (!equivalence?.assessed_op_count) {
    return notApplicableCheck("No quantized convolution-family channel exposed a complete int32 post-bias interval under both pinned TFLite fixed-point build paths.");
  }
  const details = (equivalence.ops || []).map((row, rowIndex) => row.assessment_status === "assessed" ? ({
    op_index: row.op_index,
    op_name: row.op_name,
    assessed_channels: row.assessed_channel_count,
    equivalent_channels: row.equivalent_channel_count,
    divergent_channels: row.divergent_channel_count,
    interval_state_count_decimal: row.interval_state_count_decimal,
    divergent_state_count_decimal: row.divergent_state_count_decimal,
    divergent_state_ratio: row.divergent_state_ratio,
    default_lower_state_count_decimal: row.default_lower_state_count_decimal,
    default_higher_state_count_decimal: row.default_higher_state_count_decimal,
    maximum_absolute_output_delta: row.maximum_absolute_output_delta,
    maximum_pair_segment_count: row.maximum_pair_segment_count,
    maximum_divergent_region_count: row.maximum_divergent_region_count,
    source_witness_ledger_sha256: row.source_witness_ledger_sha256,
    source_requantization_ledger_sha256: row.source_requantization_ledger_sha256,
    equivalence_ledger_sha256: row.equivalence_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/rounding_equivalence/ops/${rowIndex}`,
  }) : null).filter(Boolean);
  return {
    status: equivalence.divergent_channel_count || equivalence.unassessed_op_count ? "review" : "pass",
    evidence_class: "DERIVED",
    proof_domain: "closed_post_bias_int32_interval_hull",
    checked_ops: equivalence.assessed_op_count,
    checked_channels: equivalence.assessed_channel_count,
    equivalent_channels: equivalence.equivalent_channel_count,
    divergent_channels: equivalence.divergent_channel_count,
    divergent_ops: equivalence.divergent_op_count,
    interval_state_count_decimal: equivalence.interval_state_count_decimal,
    divergent_state_count_decimal: equivalence.divergent_state_count_decimal,
    divergent_state_ratio: equivalence.divergent_state_ratio,
    default_lower_state_count_decimal: equivalence.default_lower_state_count_decimal,
    default_higher_state_count_decimal: equivalence.default_higher_state_count_decimal,
    maximum_absolute_output_delta: equivalence.maximum_absolute_output_delta,
    exact_pair_segments: equivalence.pair_segment_count,
    divergent_regions: equivalence.divergent_region_count,
    divergence_histogram: equivalence.divergence_histogram,
    build_flag_status: "TFLITE_SINGLE_ROUNDING_not_embedded_in_artifact",
    source_commit: equivalence.source_commit,
    source_references: equivalence.source_references,
    equivalence_proof: equivalence.equivalence_proof,
    segmentation_bound: equivalence.segmentation_bound,
    method: equivalence.method,
    interpretation_boundary: equivalence.interpretation_boundary,
    details,
  };
}

function buildAccumulatorReachabilityCheck(analysis) {
  const reachability = analysis?.accumulator_reachability;
  if (!reachability?.candidate_op_count) {
    return notApplicableCheck("No TFLite quantized accumulator channel was available for bounded-sum reachability analysis.");
  }
  const details = (reachability.ops || []).map((row, rowIndex) => ({
    op_index: row.op_index,
    op_name: row.op_name,
    assessment_status: row.assessment_status,
    assessed_channels: row.assessed_channel_count,
    complete_integer_interval_channels: row.complete_integer_interval_channel_count,
    complete_modular_lattice_channels: row.complete_modular_lattice_channel_count,
    partial_band_channels: row.partial_band_channel_count,
    singleton_channels: row.singleton_channel_count,
    exact_reachable_divergent_channels: row.exact_reachable_divergent_channel_count,
    unresolved_divergent_channels: row.unresolved_divergent_channel_count,
    interval_divergent_state_count_decimal: row.interval_divergent_state_count_decimal,
    exact_reachable_divergent_state_count_decimal: row.exact_reachable_divergent_state_count_decimal,
    provably_unreachable_divergent_state_count_decimal: row.provably_unreachable_divergent_state_count_decimal,
    unresolved_divergent_state_count_decimal: row.unresolved_divergent_state_count_decimal,
    maximum_lattice_gcd: row.maximum_lattice_gcd ?? null,
    source_witness_ledger_sha256: row.source_witness_ledger_sha256,
    source_rounding_equivalence_ledger_sha256: row.source_rounding_equivalence_ledger_sha256,
    reachability_ledger_sha256: row.reachability_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/accumulator_reachability/ops/${rowIndex}`,
  }));
  return {
    status: reachability.exact_reachable_divergent_channel_count || reachability.unresolved_divergent_channel_count || reachability.unassessed_op_count ? "review" : "pass",
    evidence_class: "DERIVED",
    proof_domain: "full_valid_kernel_local_independent_legal_input_codes",
    candidate_ops: reachability.candidate_op_count,
    assessed_ops: reachability.assessed_op_count,
    unassessed_ops: reachability.unassessed_op_count,
    assessed_channels: reachability.assessed_channel_count,
    complete_integer_interval_channels: reachability.complete_integer_interval_channel_count,
    complete_modular_lattice_channels: reachability.complete_modular_lattice_channel_count,
    partial_band_channels: reachability.partial_band_channel_count,
    singleton_channels: reachability.singleton_channel_count,
    exact_reachable_divergent_channels: reachability.exact_reachable_divergent_channel_count,
    unresolved_divergent_channels: reachability.unresolved_divergent_channel_count,
    interval_only_divergent_channels: reachability.interval_only_divergent_channel_count,
    interval_state_count_decimal: reachability.interval_state_count_decimal,
    certified_reachable_state_count_decimal: reachability.certified_reachable_state_count_decimal,
    provably_unreachable_state_count_decimal: reachability.provably_unreachable_state_count_decimal,
    unresolved_state_count_decimal: reachability.unresolved_state_count_decimal,
    interval_divergent_state_count_decimal: reachability.interval_divergent_state_count_decimal,
    exact_reachable_divergent_state_count_decimal: reachability.exact_reachable_divergent_state_count_decimal,
    provably_unreachable_divergent_state_count_decimal: reachability.provably_unreachable_divergent_state_count_decimal,
    unresolved_divergent_state_count_decimal: reachability.unresolved_divergent_state_count_decimal,
    exact_reachable_divergent_ratio: reachability.exact_reachable_divergent_ratio,
    maximum_lattice_gcd: reachability.maximum_lattice_gcd,
    source_commit: reachability.source_commit,
    source_references: reachability.source_references,
    bounded_sum_proof: reachability.bounded_sum_proof,
    state_conservation: reachability.state_conservation,
    method: reachability.method,
    interpretation_boundary: reachability.interpretation_boundary,
    details,
  };
}

function buildNumericalAbiPropagationCheck(analysis) {
  const propagation = analysis?.numerical_abi_propagation;
  if (!propagation?.candidate_source_op_count) {
    return notApplicableCheck("No TFLite rounding-equivalence source op was available for structural output propagation analysis.");
  }
  const details = (propagation.sources || []).map((source, sourceIndex) => ({
    op_index: source.op_index,
    op_name: source.op_name,
    assessment_status: source.assessment_status,
    divergent_channels: source.divergent_channel_count,
    divergent_state_count_decimal: source.divergent_state_count_decimal,
    local_reachability_status: source.local_reachability_status,
    exact_reachable_divergent_channels: source.exact_reachable_divergent_channel_count,
    exact_reachable_divergent_state_count_decimal: source.exact_reachable_divergent_state_count_decimal,
    provably_unreachable_divergent_state_count_decimal: source.provably_unreachable_divergent_state_count_decimal,
    unresolved_divergent_state_count_decimal: source.unresolved_divergent_state_count_decimal,
    reachable_ops: source.reachable_op_count,
    reachable_model_outputs: source.reachable_model_output_tensor_count,
    exact_model_output_graph_route_count_decimal: source.exact_model_output_graph_route_count_decimal ?? null,
    reconvergence_ops: source.reconvergence_op_count,
    single_branch_merges: source.single_branch_merge_op_count,
    predicted_boundary_edges: source.predicted_boundary_edge_count,
    assessed_boundary_logical_payload_bytes: source.assessed_boundary_logical_payload_bytes,
    source_equivalence_ledger_sha256: source.source_equivalence_ledger_sha256,
    source_reachability_ledger_sha256: source.source_reachability_ledger_sha256,
    propagation_ledger_sha256: source.propagation_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/numerical_abi_propagation/sources/${sourceIndex}`,
  }));
  return {
    status: propagation.output_reachable_source_op_count || propagation.unassessed_source_op_count ? "review" : "pass",
    evidence_class: "DERIVED",
    propagation_semantics: "tensor_level_structural_potential_not_observed_numerical_effect",
    candidate_sources: propagation.candidate_source_op_count,
    divergent_sources: propagation.divergent_source_op_count,
    equivalent_sources: propagation.equivalent_source_op_count,
    unassessed_sources: propagation.unassessed_source_op_count,
    local_reachability_unassessed_sources: propagation.local_reachability_unassessed_source_op_count,
    output_reachable_sources: propagation.output_reachable_source_op_count,
    output_isolated_sources: propagation.output_isolated_source_op_count,
    exact_local_counterexample_sources: propagation.exact_local_counterexample_source_op_count,
    residue_excluded_divergence_sources: propagation.residue_excluded_divergence_source_op_count,
    unresolved_divergence_sources: propagation.unresolved_divergence_source_op_count,
    exact_output_reachable_sources: propagation.exact_output_reachable_source_op_count,
    interval_divergent_state_count_decimal: propagation.interval_divergent_state_count_decimal,
    exact_local_divergent_state_count_decimal: propagation.exact_local_divergent_state_count_decimal,
    residue_excluded_divergent_state_count_decimal: propagation.residue_excluded_divergent_state_count_decimal,
    unresolved_divergent_state_count_decimal: propagation.unresolved_divergent_state_count_decimal,
    graph_edges: propagation.graph_edge_count,
    source_corridor_edge_instances: propagation.source_corridor_edge_instance_count,
    unique_reachable_ops: propagation.unique_reachable_op_count,
    unique_reachable_tensors: propagation.unique_reachable_tensor_count,
    unique_predicted_boundary_edges: propagation.unique_predicted_boundary_edge_count,
    unique_predicted_boundary_logical_payload_bytes: propagation.unique_predicted_boundary_logical_payload_bytes ?? null,
    exact_source_corridor_edge_instances: propagation.exact_source_corridor_edge_instance_count,
    exact_unique_reachable_ops: propagation.exact_unique_reachable_op_count,
    exact_unique_reachable_tensors: propagation.exact_unique_reachable_tensor_count,
    exact_unique_predicted_boundary_edges: propagation.exact_unique_predicted_boundary_edge_count,
    exact_unique_predicted_boundary_logical_payload_bytes: propagation.exact_unique_predicted_boundary_logical_payload_bytes ?? null,
    maximum_model_output_op_hops: propagation.maximum_model_output_op_hops ?? null,
    maximum_model_output_graph_route_count_decimal: propagation.maximum_model_output_graph_route_count_decimal ?? null,
    graph_cycle_status: propagation.graph_cycle_status,
    graph_ledger_sha256: propagation.graph_ledger_sha256,
    route_definition: propagation.route_definition,
    ranking_definition: propagation.ranking_definition,
    method: propagation.method,
    interpretation_boundary: propagation.interpretation_boundary,
    details,
  };
}

function buildInputCounterexampleCheck(analysis) {
  const evidence = analysis?.input_counterexample;
  if (!evidence || evidence.status === "not_computed_internal_scope") {
    return notApplicableCheck("The model-input tensor counterexample constructor was not available in this analysis scope.");
  }
  const details = (evidence.sources || []).map((source, sourceIndex) => ({
    op_index: source.op_index,
    op_name: source.op_name,
    input_tensor_index: source.input_tensor_index ?? null,
    input_origin: source.input_origin,
    classification: source.classification,
    exact_reachable_divergent_channels: source.exact_reachable_divergent_channel_count,
    exact_reachable_divergent_state_count_decimal: source.exact_reachable_divergent_state_count_decimal,
    reachable_model_outputs: source.reachable_model_output_tensor_count,
    exact_model_output_graph_route_count_decimal: source.exact_model_output_graph_route_count_decimal ?? null,
    representative_witness_index: source.representative_witness_index ?? null,
    reachability_ledger_sha256: source.source_reachability_ledger_sha256,
    propagation_ledger_sha256: source.source_propagation_ledger_sha256,
    witness_ledger_sha256: source.representative_witness_ledger_sha256,
    evidence_json_pointer: `/evidence/static_analysis/input_counterexample/sources/${sourceIndex}`,
  }));
  return {
    status: evidence.tensor_abi_constructive_source_op_count > 0 || evidence.upstream_activation_unresolved_source_op_count > 0 ? "review" : "pass",
    evidence_class: "DERIVED",
    proof_semantics: "complete_quantized_model_input_tensor_exists_for_exact_source_output_code_divergence",
    exact_local_sources: evidence.exact_local_source_op_count,
    direct_model_input_sources: evidence.direct_model_input_source_op_count,
    tensor_abi_constructive_sources: evidence.tensor_abi_constructive_source_op_count,
    upstream_activation_unresolved_sources: evidence.upstream_activation_unresolved_source_op_count,
    not_assessed_sources: evidence.not_assessed_source_op_count,
    tensor_abi_constructive_channels: evidence.tensor_abi_constructive_channel_count,
    tensor_abi_constructive_divergent_state_count_decimal: evidence.tensor_abi_constructive_divergent_state_count_decimal,
    output_reachable_constructive_sources: evidence.output_reachable_constructive_source_op_count,
    representative_witnesses: evidence.representative_witness_count,
    source_classification_conservation: evidence.source_classification_conservation,
    portfolio_ledger_sha256: evidence.portfolio_ledger_sha256,
    method: evidence.method,
    proof_scope: evidence.proof_scope,
    interpretation_boundary: evidence.interpretation_boundary,
    details,
  };
}

function buildPreprocessingRealizabilityCheck(analysis) {
  const evidence = analysis?.preprocessing_realizability;
  if (!evidence || evidence.status === "not_computed_internal_scope" || !evidence.candidate_evaluation_count) {
    return notApplicableCheck("No eligible constructive NHWC RGB input witness was available for preprocessing counterfactual analysis.");
  }
  return {
    status: evidence.exact_tensor_realization_candidate_count > 0 && evidence.non_exact_candidate_count > 0 ? "review" : "pass",
    evidence_class: "DERIVED",
    assessment_kind: evidence.assessment_kind,
    proof_semantics: "finite_256_code_source_domain_inverse_under_explicit_candidate_contract",
    source_witnesses: evidence.source_witness_count,
    eligible_image_witnesses: evidence.eligible_image_witness_count,
    candidate_contracts: evidence.candidate_contract_count,
    candidate_evaluations: evidence.candidate_evaluation_count,
    assessed_candidates: evidence.assessed_candidate_count,
    exact_tensor_realization_candidates: evidence.exact_tensor_realization_candidate_count,
    non_exact_candidates: evidence.non_exact_candidate_count,
    exact_contract_ids: evidence.exact_contract_ids,
    best_non_exact_contract_id: evidence.best_non_exact_contract_id,
    best_non_exact_unrealizable_elements: evidence.best_non_exact_unrealizable_element_count ?? null,
    candidate_conservation: evidence.candidate_conservation,
    portfolio_ledger_sha256: evidence.portfolio_ledger_sha256,
    method: evidence.method,
    proof_scope: evidence.proof_scope,
    interpretation_boundary: evidence.interpretation_boundary,
    details: (evidence.candidates || []).map((candidate, candidateIndex) => ({
      contract_id: candidate.contract_id,
      contract_label: candidate.contract_label,
      tensor_channel_order: candidate.tensor_channel_order,
      exact_tensor_realization: candidate.exact_tensor_realization,
      exact_tensor_elements: candidate.exact_tensor_element_count,
      unrealizable_tensor_elements: candidate.unrealizable_tensor_element_count,
      minimum_total_absolute_tensor_code_error_decimal: candidate.minimum_total_absolute_tensor_code_error_decimal,
      maximum_absolute_tensor_code_error: candidate.maximum_absolute_tensor_code_error,
      reachable_tensor_codes_by_channel: (candidate.channel_maps || []).map((row) => row.reachable_tensor_code_count),
      tensor_code_holes_by_channel: (candidate.channel_maps || []).map((row) => row.tensor_code_hole_count),
      nearest_rgb_fixture_sha256: candidate.nearest_rgb_fixture_sha256,
      candidate_ledger_sha256: candidate.candidate_ledger_sha256,
      evidence_json_pointer: `/evidence/static_analysis/preprocessing_realizability/candidates/${candidateIndex}`,
    })),
  };
}

function buildMissingMetadataCheck(analysis) {
  const quantizedArtifact = Number(analysis?.quantized_tensors || 0) > 0;
  const details = [];
  if (quantizedArtifact) {
    for (const tensor of analysis?.tensors || []) {
      if (Number(tensor?.quant_scales || 0) > 0) continue;
      const classification = missingMetadataRole(analysis, tensor);
      details.push({
        tensor_index: tensorIndex(tensor),
        tensor_name: tensor?.name || "",
        dtype: tensor?.dtype || "UNKNOWN",
        classification,
        metadata_expected: classification !== "shape_or_configuration",
        status: classification === "shape_or_configuration" ? "not_applicable" : "review",
        evidence_json_pointer: `/evidence/static_analysis/tensors/${tensorIndex(tensor)}`,
      });
    }
  }
  const review = details.filter((item) => item.status === "review");
  return {
    status: !quantizedArtifact ? "not_applicable" : review.length ? "review" : "pass",
    evidence_class: "DERIVED",
    tensors_without_metadata: details.length,
    review_tensors: review.length,
    not_applicable_tensors: details.length - review.length,
    details,
  };
}

function buildQdqBoundaryCheck(analysis) {
  const details = (analysis?.ops || []).filter((op) => ["QUANTIZE", "DEQUANTIZE"].includes(opName(op))).map((op) => ({
    op_index: op.index,
    op_name: op.name,
    input_tensor_indices: (op.inputs || []).map(Number),
    output_tensor_indices: (op.outputs || []).map(Number),
    estimated_bytes: Number(op.estimated_bytes || 0),
    evidence_json_pointer: `/evidence/static_analysis/ops/${op.index}`,
  }));
  return {
    status: "pass",
    evidence_class: "OBSERVED",
    explicit_qdq_ops: details.length,
    mid_graph_quantization_holes: Number(analysis?.quant_hole_count || 0),
    details,
  };
}

function buildKernelQuantizationCheck(analysis) {
  const kernels = new Map();
  for (const op of analysis?.ops || []) {
    if (!CONV_LIKE.has(opName(op))) continue;
    const weight = tensorAt(analysis, op.inputs?.[1]);
    const index = tensorIndex(weight);
    if (index != null && numericScales(weight).length) kernels.set(index, { op, weight });
  }
  const entries = [...kernels.values()];
  const scales = entries.flatMap(({ weight }) => numericScales(weight));
  const depthwisePerTensor = entries.filter(({ op, weight }) => opName(op) === "DEPTHWISE_CONV_2D" && Number(weight?.quant_scales || 0) === 1).length;
  return {
    status: entries.length ? (depthwisePerTensor ? "review" : "pass") : "not_applicable",
    evidence_class: "DERIVED",
    kernel_tensors: entries.length,
    per_tensor_kernel_tensors: entries.filter(({ weight }) => Number(weight?.quant_scales || 0) === 1).length,
    per_channel_kernel_tensors: entries.filter(({ weight }) => Number(weight?.quant_scales || 0) > 1).length,
    depthwise_per_tensor_kernel_tensors: depthwisePerTensor,
    minimum_weight_scale: scales.length ? Math.min(...scales) : null,
    maximum_weight_scale: scales.length ? Math.max(...scales) : null,
    maximum_to_minimum_scale_ratio: scales.length ? Math.max(...scales) / Math.min(...scales) : null,
  };
}

function missingMetadataRole(analysis, tensor) {
  const consumers = (analysis?.ops || []).filter((op) => (op.inputs || []).includes(Number(tensor?.index)));
  const names = consumers.map(opName);
  const dtype = String(tensor?.dtype || "").toUpperCase();
  if (["RESHAPE", "SHAPE", "STRIDED_SLICE", "PACK", "CONCATENATION"].some((name) => names.includes(name))) return "shape_or_configuration";
  if (["INT32", "INT64"].includes(dtype) && (tensor?.constant_buffer || elementCount(tensor?.shape) <= 8)) return "shape_or_configuration";
  return tensor?.constant_buffer ? "constant" : "activation";
}

function quantMagnitudeBound(tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const [lower, upper] = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : [0, 0];
  if (lower === 0 && upper === 0) return 0;
  const zeroPoints = (tensor?.zero_point_sample || [0]).map(Number);
  return Math.max(...zeroPoints.map((zeroPoint) => Math.max(Math.abs(lower - zeroPoint), Math.abs(upper - zeroPoint))));
}

function kernelAccumulationTerms(op, weight) {
  const shape = (weight?.shape || []).map((dim) => Math.max(1, Number(dim) || 1));
  if (opName(op) === "DEPTHWISE_CONV_2D" && shape.length >= 4) return shape[1] * shape[2];
  return shape.length >= 2 ? shape.slice(1).reduce((product, dim) => product * dim, 1) : 0;
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

function maxNullable(values) {
  const assessed = values.filter((value) => value != null).map(Number).filter(Number.isFinite);
  return assessed.length ? Math.max(...assessed) : null;
}

function elementCount(shape) {
  return Array.isArray(shape) && shape.length && shape.every((dim) => Number(dim) > 0)
    ? shape.reduce((product, dim) => product * Number(dim), 1)
    : 0;
}
