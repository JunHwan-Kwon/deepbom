import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

const INPUT_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const KERNEL_TYPES = new Set(["LINEAR", "POLY", "RBF", "SIGMOID"]);
const POST_TRANSFORMS = new Set(["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"]);
const MAX_REFERENCE_VALUES = 1_000_000;

export function inferOnnxMlSvmClassifier(context) {
  return inferSvmClassifier(context);
}

export function inferOnnxMlSvmRegressor(context) {
  return inferSvmRegressor(context);
}

function inferSvmClassifier({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const input = svmInputContract(node, tensorMap);
  const failures = [...input.failures];
  const reasons = [...input.reasons];
  const attributes = classifierAttributes(node, failures);
  const labels = classifierLabels(attributes, failures);
  const onnxContractFailure = failures[0] || "";
  const runtime = classifierRuntimeContract(input, attributes, labels);
  if (runtime.status === "fail") failures.push(runtime.reason);
  else if (runtime.status === "partial") reasons.push(runtime.reason);

  const schemaScoreWidth = labels.count > 0 ? labels.count : null;
  const runtimeScoreWidth = runtime.scoreWidth;
  const shapeMismatch = runtime.mode === "svc" && !runtime.haveProbability
    && labels.count > 3 && runtimeScoreWidth !== schemaScoreWidth;
  if (shapeMismatch) failures.push("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch");

  const labelDtype = labels.kind === "string" ? "STRING" : "INT64";
  const schemaShapes = classifierOutputShapes(input, schemaScoreWidth);
  const runtimeShapes = classifierOutputShapes(input, runtimeScoreWidth);
  const labelType = makeOnnxTensorType(labelDtype, schemaShapes.label, schemaShapes.declared);
  const scoreType = makeOnnxTensorType("FLOAT32", schemaShapes.score, schemaShapes.declared && !shapeMismatch);
  const exactScoreElements = schemaShapes.declared && !shapeMismatch && schemaShapes.score.every(knownDimension)
    ? safeShapeElementCount(schemaShapes.score) : null;
  const duplicateLabels = duplicateValueCount(labels.values);
  const allParameters = [
    ...attributes.kernelParams, ...attributes.supportVectors, ...attributes.coefficients,
    ...attributes.probA, ...attributes.probB, ...attributes.rho,
  ];
  const nonFiniteParameters = allParameters.filter((value) => !Number.isFinite(value)).length;
  const reference = evaluateClassifierReference({ input, attributes, labels, runtime });

  const riskCodes = [];
  if (runtime.status === "fail") riskCodes.push("svm_classifier_pinned_ort_runtime_contract_invalid");
  if (shapeMismatch) riskCodes.push("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch");
  if (duplicateLabels > 0) riskCodes.push("svm_classifier_duplicate_labels_ambiguous_output_semantics");
  if (runtime.unusedCoefficientCount > 0 || runtime.unusedSupportVectorCount > 0
    || runtime.unusedRhoCount > 0 || runtime.unusedProbabilityCount > 0) {
    riskCodes.push("svm_classifier_serialized_parameters_ignored_by_pinned_ort");
  }
  if (runtime.mode === "linear" && attributes.kernelType !== "LINEAR") {
    riskCodes.push("svm_classifier_linear_mode_forces_linear_kernel");
  }
  if (runtime.mode === "svc" && runtime.haveProbability && attributes.postTransform !== "NONE") {
    riskCodes.push("svm_classifier_probability_scores_receive_additional_post_transform");
  }
  if (runtime.mode === "svc" && !runtime.haveProbability && labels.count === 2
    && attributes.postTransform === "PROBIT") {
    riskCodes.push("svm_classifier_binary_probit_second_score_unwritten");
  }
  if (runtime.mode === "svc" && !runtime.haveProbability && labels.count === 2
    && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(attributes.postTransform)) {
    riskCodes.push("svm_classifier_binary_post_transform_uses_complement_expansion");
  }
  if (nonFiniteParameters > 0 || Number(reference.nonFiniteScoreCount || 0) > 0) {
    riskCodes.push("svm_classifier_non_finite_parameter_or_reference_score");
  }
  if (Number(reference.decisionBoundaryCount || 0) > 0) riskCodes.push("svm_classifier_reference_decision_boundary");

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = svmBaseRow({
    scope, nodeIndex, importedOpset, opName: "SVMClassifier", contractKind: "svm_classifier",
    status, input, outputName: node.outputs?.[1] || "", outputDtype: "FLOAT32",
    outputShape: shapeMismatch ? [] : schemaShapes.score,
    outputShapeDeclared: schemaShapes.declared && !shapeMismatch,
    exactOutputElements: exactScoreElements, failures, reasons,
  });
  Object.assign(row, {
    output_names: [...(node.outputs || [])],
    canonical_output_types: [canonicalOnnxTypeProto(labelType), canonicalOnnxTypeProto(scoreType)],
    canonical_output_shapes: [schemaShapes.label, shapeMismatch ? [] : schemaShapes.score],
    svm_onnx_contract_status: onnxContractFailure ? "fail" : "pass",
    svm_onnx_contract_reason: onnxContractFailure,
    svm_pinned_ort_contract_status: runtime.status,
    svm_pinned_ort_contract_reason: runtime.reason,
    svm_mode: runtime.mode,
    svm_kernel_type: attributes.kernelType,
    svm_kernel_type_source: attributes.kernelTypeSource,
    svm_kernel_params: attributes.kernelParams.map(valueText),
    svm_kernel_params_source: attributes.kernelParamsSource,
    svm_post_transform: attributes.postTransform,
    svm_post_transform_source: attributes.postTransformSource,
    svm_class_label_kind: labels.kind,
    svm_class_label_count: labels.count,
    svm_class_label_values: labels.values.map(valueText),
    svm_duplicate_label_count: duplicateLabels,
    svm_vectors_per_class: attributes.vectorsPerClass.map((value) => value.toString()),
    svm_vector_count: runtime.vectorCount,
    svm_pairwise_classifier_count: runtime.pairCount,
    svm_schema_score_width: schemaScoreWidth,
    svm_pinned_ort_score_width: runtimeScoreWidth,
    svm_schema_runtime_score_width_mismatch: shapeMismatch,
    svm_support_vector_value_count: attributes.supportVectors.length,
    svm_expected_support_vector_value_count: runtime.expectedSupportVectorCount,
    svm_used_support_vector_value_count: runtime.usedSupportVectorCount,
    svm_unused_support_vector_value_count: runtime.unusedSupportVectorCount,
    svm_coefficient_count: attributes.coefficients.length,
    svm_expected_coefficient_count: runtime.expectedCoefficientCount,
    svm_used_coefficient_count: runtime.usedCoefficientCount,
    svm_unused_coefficient_count: runtime.unusedCoefficientCount,
    svm_rho_count: attributes.rho.length,
    svm_expected_rho_count: runtime.expectedRhoCount,
    svm_used_rho_count: runtime.usedRhoCount,
    svm_unused_rho_count: runtime.unusedRhoCount,
    svm_prob_a_count: attributes.probA.length,
    svm_prob_b_count: attributes.probB.length,
    svm_probability_enabled: runtime.haveProbability,
    svm_expected_probability_parameter_count_per_array: runtime.expectedProbabilityCount,
    svm_used_probability_parameter_count: runtime.usedProbabilityCount,
    svm_unused_probability_parameter_count: runtime.unusedProbabilityCount,
    svm_non_finite_parameter_count: nonFiniteParameters,
    svm_linear_mode_forced_kernel: runtime.mode === "linear" && attributes.kernelType !== "LINEAR",
    svm_reference_assessment_status: reference.status,
    svm_reference_input_value_count: reference.inputValueCount,
    svm_reference_raw_score_count: reference.rawScoreCount,
    svm_reference_non_finite_score_count: reference.nonFiniteScoreCount,
    svm_reference_decision_boundary_count: reference.decisionBoundaryCount,
    svm_reference_raw_score_preview: reference.rawScorePreview,
    svm_reference_output_score_preview: reference.outputScorePreview,
    svm_reference_label_preview: reference.labelPreview,
    svm_reference_boundary: "Deterministic scalar FLOAT32 reference only; pinned ORT uses MLAS GEMM or platform libm paths and the executed accumulation/transcendental implementation is not observed, so values are not propagated as runtime-bit-exact tensor evidence.",
    risk_codes: riskCodes,
  });
  const outputs = [];
  const canPropagate = status !== "fail" && schemaShapes.declared && !shapeMismatch;
  if (canPropagate && node.outputs?.[0]) outputs.push([node.outputs[0], onnxValueDescriptorFromType(labelType)]);
  if (canPropagate && node.outputs?.[1]) outputs.push([node.outputs[1], onnxValueDescriptorFromType(scoreType)]);
  return { status, reason: row.reason_codes[0] || "", result: { outputs }, row };
}

function inferSvmRegressor({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const input = svmInputContract(node, tensorMap);
  const failures = [...input.failures];
  const reasons = [...input.reasons];
  const attributes = regressorAttributes(node, failures);
  const onnxContractFailure = failures[0] || "";
  const runtime = regressorRuntimeContract(input, attributes);
  if (runtime.status === "fail") failures.push(runtime.reason);
  else if (runtime.status === "partial") reasons.push(runtime.reason);
  const cpuDtypeGap = INPUT_DTYPES.has(input.dtype) && input.dtype !== "FLOAT32";
  if (cpuDtypeGap) reasons.push("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");

  const outputShape = input.rank === 1 ? [1, 1]
    : input.rank === 2 ? [input.batchCount, 1] : [];
  const shapeDeclared = input.rank === 1 || input.rank === 2;
  const outputType = makeOnnxTensorType("FLOAT32", outputShape, shapeDeclared);
  const exactOutputElements = shapeDeclared && outputShape.every(knownDimension)
    ? safeShapeElementCount(outputShape) : null;
  const allParameters = [
    ...attributes.kernelParams, ...attributes.supportVectors, ...attributes.coefficients, ...attributes.rho,
  ];
  const nonFiniteParameters = allParameters.filter((value) => !Number.isFinite(value)).length;
  const reference = evaluateRegressorReference({ input, attributes, runtime, cpuDtypeGap });
  const ignoredPostTransform = attributes.postTransform !== "NONE";

  const riskCodes = [];
  if (runtime.status === "fail") riskCodes.push("svm_regressor_pinned_ort_runtime_contract_invalid");
  if (cpuDtypeGap) riskCodes.push("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  if (ignoredPostTransform) riskCodes.push("svm_regressor_post_transform_ignored_by_pinned_ort");
  if (![0n, 1n].includes(attributes.oneClass)) riskCodes.push("svm_regressor_noncanonical_one_class_flag");
  if (runtime.unusedCoefficientCount > 0 || runtime.unusedSupportVectorCount > 0 || runtime.unusedRhoCount > 0) {
    riskCodes.push("svm_regressor_serialized_parameters_ignored_by_pinned_ort");
  }
  if (runtime.mode === "linear" && attributes.kernelType !== "LINEAR") {
    riskCodes.push("svm_regressor_linear_mode_forces_linear_kernel");
  }
  if (nonFiniteParameters > 0 || Number(reference.nonFiniteScoreCount || 0) > 0) {
    riskCodes.push("svm_regressor_non_finite_parameter_or_reference_score");
  }
  if (Number(reference.decisionBoundaryCount || 0) > 0) riskCodes.push("svm_regressor_reference_decision_boundary");

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = svmBaseRow({
    scope, nodeIndex, importedOpset, opName: "SVMRegressor", contractKind: "svm_regressor",
    status, input, outputName: node.outputs?.[0] || "", outputDtype: "FLOAT32",
    outputShape, outputShapeDeclared: shapeDeclared,
    exactOutputElements, failures, reasons,
  });
  Object.assign(row, {
    svm_onnx_contract_status: onnxContractFailure ? "fail" : "pass",
    svm_onnx_contract_reason: onnxContractFailure,
    svm_pinned_ort_contract_status: cpuDtypeGap ? "fail" : runtime.status,
    svm_pinned_ort_contract_reason: cpuDtypeGap ? "schema_dtype_missing_pinned_ort_cpu_kernel" : runtime.reason,
    svm_mode: runtime.mode,
    svm_kernel_type: attributes.kernelType,
    svm_kernel_type_source: attributes.kernelTypeSource,
    svm_kernel_params: attributes.kernelParams.map(valueText),
    svm_kernel_params_source: attributes.kernelParamsSource,
    svm_post_transform: attributes.postTransform,
    svm_post_transform_source: attributes.postTransformSource,
    svm_post_transform_applied_by_pinned_ort: false,
    svm_one_class_value: attributes.oneClass.toString(),
    svm_one_class_source: attributes.oneClassSource,
    svm_n_supports: attributes.nSupports == null ? null : Number(attributes.nSupports),
    svm_vector_count: runtime.vectorCount,
    svm_pairwise_classifier_count: 0,
    svm_schema_score_width: 1,
    svm_pinned_ort_score_width: 1,
    svm_schema_runtime_score_width_mismatch: false,
    svm_support_vector_value_count: attributes.supportVectors.length,
    svm_expected_support_vector_value_count: runtime.expectedSupportVectorCount,
    svm_used_support_vector_value_count: runtime.usedSupportVectorCount,
    svm_unused_support_vector_value_count: runtime.unusedSupportVectorCount,
    svm_coefficient_count: attributes.coefficients.length,
    svm_expected_coefficient_count: runtime.expectedCoefficientCount,
    svm_used_coefficient_count: runtime.usedCoefficientCount,
    svm_unused_coefficient_count: runtime.unusedCoefficientCount,
    svm_rho_count: attributes.rho.length,
    svm_expected_rho_count: runtime.expectedRhoCount,
    svm_used_rho_count: runtime.usedRhoCount,
    svm_unused_rho_count: runtime.unusedRhoCount,
    svm_prob_a_count: 0, svm_prob_b_count: 0, svm_probability_enabled: false,
    svm_expected_probability_parameter_count_per_array: 0,
    svm_used_probability_parameter_count: 0, svm_unused_probability_parameter_count: 0,
    svm_non_finite_parameter_count: nonFiniteParameters,
    svm_linear_mode_forced_kernel: runtime.mode === "linear" && attributes.kernelType !== "LINEAR",
    svm_reference_assessment_status: reference.status,
    svm_reference_input_value_count: reference.inputValueCount,
    svm_reference_raw_score_count: reference.rawScoreCount,
    svm_reference_non_finite_score_count: reference.nonFiniteScoreCount,
    svm_reference_decision_boundary_count: reference.decisionBoundaryCount,
    svm_reference_raw_score_preview: reference.rawScorePreview,
    svm_reference_output_score_preview: reference.outputScorePreview,
    svm_reference_label_preview: [],
    svm_reference_boundary: "Deterministic scalar FLOAT32 reference only; pinned ORT CPU supports FLOAT32 and uses MLAS GEMM or platform libm paths. The post_transform attribute is not applied by the pinned regressor implementation.",
    risk_codes: riskCodes,
  });
  const canPropagate = !failures.length && shapeDeclared;
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: canPropagate && node.outputs?.[0] ? [[node.outputs[0], onnxValueDescriptorFromType(outputType)]] : [] },
    row,
  };
}

function svmInputContract(node, tensorMap) {
  const input = tensorMap.get(node.inputs?.[0]);
  const type = onnxTypeProtoFromValue(input);
  const failures = [];
  const reasons = [];
  const dtype = type?.kind === "tensor" ? type.dtype || "UNKNOWN" : "UNKNOWN";
  const shapeDeclared = type?.kind === "tensor" && type.shapeDeclared === true;
  const shape = shapeDeclared ? [...(type.shape || [])] : [];
  const rank = shapeDeclared ? shape.length : null;
  if (!input) failures.push("svm_input_missing");
  else if (type?.kind !== "tensor") failures.push("svm_input_not_dense_tensor");
  else if (!INPUT_DTYPES.has(dtype)) failures.push(`svm_input_dtype_not_supported_by_schema:${dtype}`);
  if (rank != null && ![1, 2].includes(rank)) failures.push(`svm_input_rank_not_one_or_two:${rank}`);
  if (rank == null) reasons.push("svm_input_rank_unresolved");
  const batchCount = rank === 1 ? 1 : rank === 2 ? shape[0] : null;
  const featureCount = rank === 1 ? shape[0] : rank === 2 ? shape[1] : null;
  if (rank != null && !knownDimension(featureCount)) reasons.push("svm_feature_count_unresolved");
  if (rank === 2 && !knownDimension(batchCount)) reasons.push("svm_batch_count_unresolved");
  return { input, type, dtype, shapeDeclared, shape, rank, batchCount, featureCount, failures, reasons };
}

function classifierAttributes(node, failures) {
  const readFloats = (name, fallback = []) => {
    const attribute = node.attributes?.get(name);
    if (!attribute) return fallback;
    const values = floatListAttribute(attribute);
    if (values == null) failures.push(`svm_classifier_${name}_not_float_list`);
    return values || [];
  };
  const readInts = (name) => {
    const attribute = node.attributes?.get(name);
    if (!attribute) return [];
    const values = intListAttribute(attribute);
    if (values == null) failures.push(`svm_classifier_${name}_not_exact_int_list`);
    return values || [];
  };
  const kernelAttribute = node.attributes?.get("kernel_type");
  const kernelType = kernelAttribute ? stringScalarAttribute(kernelAttribute) : "LINEAR";
  if (!KERNEL_TYPES.has(kernelType)) failures.push("svm_classifier_kernel_type_invalid");
  const postAttribute = node.attributes?.get("post_transform");
  const postTransform = postAttribute ? stringScalarAttribute(postAttribute) : "NONE";
  if (!POST_TRANSFORMS.has(postTransform)) failures.push("svm_classifier_post_transform_invalid");
  const kernelParams = readFloats("kernel_params");
  if (kernelParams.length !== 0 && kernelParams.length !== 3) failures.push("svm_classifier_kernel_params_cardinality_not_zero_or_three");
  return {
    kernelType: kernelType || "UNRESOLVED", kernelTypeSource: kernelAttribute ? "explicit_attribute" : "onnx_schema_default_LINEAR",
    kernelParams, kernelParamsPresent: node.attributes?.has("kernel_params") === true,
    kernelParamsSource: node.attributes?.has("kernel_params") ? "explicit_attribute" : "omitted_schema_optional_but_pinned_ort_required",
    vectorsPerClass: readInts("vectors_per_class"), supportVectors: readFloats("support_vectors"),
    coefficients: readFloats("coefficients"), probA: readFloats("prob_a"), probB: readFloats("prob_b"), rho: readFloats("rho"),
    intLabels: readInts("classlabels_ints"), stringLabels: stringList(node, "classlabels_strings", failures),
    intLabelsPresent: node.attributes?.has("classlabels_ints") === true,
    stringLabelsPresent: node.attributes?.has("classlabels_strings") === true,
    postTransform: postTransform || "UNRESOLVED", postTransformSource: postAttribute ? "explicit_attribute" : "onnx_schema_default_NONE",
  };
}

function regressorAttributes(node, failures) {
  const readFloats = (name) => {
    const attribute = node.attributes?.get(name);
    if (!attribute) return [];
    const values = floatListAttribute(attribute);
    if (values == null) failures.push(`svm_regressor_${name}_not_float_list`);
    return values || [];
  };
  const kernelAttribute = node.attributes?.get("kernel_type");
  const kernelType = kernelAttribute ? stringScalarAttribute(kernelAttribute) : "LINEAR";
  if (!KERNEL_TYPES.has(kernelType)) failures.push("svm_regressor_kernel_type_invalid");
  const postAttribute = node.attributes?.get("post_transform");
  const postTransform = postAttribute ? stringScalarAttribute(postAttribute) : "NONE";
  if (!POST_TRANSFORMS.has(postTransform)) failures.push("svm_regressor_post_transform_invalid");
  const kernelParams = readFloats("kernel_params");
  if (kernelParams.length !== 0 && kernelParams.length !== 3) failures.push("svm_regressor_kernel_params_cardinality_not_zero_or_three");
  const nSupportsAttribute = node.attributes?.get("n_supports");
  const nSupports = nSupportsAttribute ? intScalarAttribute(nSupportsAttribute) : 0n;
  if (nSupports == null || nSupports < 0n || nSupports > BigInt(Number.MAX_SAFE_INTEGER)) failures.push("svm_regressor_onnx_n_supports_not_nonnegative_safe_integer");
  const oneClassAttribute = node.attributes?.get("one_class");
  const oneClass = oneClassAttribute ? intScalarAttribute(oneClassAttribute) : 0n;
  if (oneClass == null) failures.push("svm_regressor_onnx_one_class_not_exact_int_scalar");
  return {
    kernelType: kernelType || "UNRESOLVED", kernelTypeSource: kernelAttribute ? "explicit_attribute" : "onnx_schema_default_LINEAR",
    kernelParams, kernelParamsPresent: node.attributes?.has("kernel_params") === true,
    kernelParamsSource: node.attributes?.has("kernel_params") ? "explicit_attribute" : "omitted_schema_optional_but_pinned_ort_required",
    supportVectors: readFloats("support_vectors"), coefficients: readFloats("coefficients"), rho: readFloats("rho"),
    nSupports, oneClass: oneClass ?? 0n, oneClassSource: oneClassAttribute ? "explicit_attribute" : "onnx_schema_default_0",
    postTransform: postTransform || "UNRESOLVED", postTransformSource: postAttribute ? "explicit_attribute" : "onnx_schema_default_NONE",
  };
}

function classifierLabels(attributes, failures) {
  const exactlyOne = Number(attributes.intLabelsPresent) + Number(attributes.stringLabelsPresent) === 1;
  const values = attributes.stringLabelsPresent ? attributes.stringLabels : attributes.intLabels;
  const kind = attributes.stringLabelsPresent ? "string" : "int64";
  let reason = "";
  if (!exactlyOne) reason = "svm_classifier_onnx_exactly_one_label_attribute_required";
  else if (!values.length) reason = "svm_classifier_onnx_class_labels_empty";
  else if (values.length < 2) reason = "svm_classifier_onnx_requires_at_least_two_classes";
  if (reason) failures.push(reason);
  return { status: reason ? "fail" : "pass", reason, kind, values, count: values.length };
}

function classifierRuntimeContract(input, attributes, labels) {
  const failureContext = classifierRuntimeFailureContext(attributes, labels);
  const fail = (reason, extra = {}) => runtimeFailure(reason, { ...failureContext, ...extra });
  if (!attributes.kernelParamsPresent) return fail("kernel_params_attribute_missing_in_pinned_ort_constructor");
  const vectorCounts = [];
  for (const value of attributes.vectorsPerClass) {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) return fail("vectors_per_class_must_be_nonnegative_safe_integers");
    vectorCounts.push(Number(value));
  }
  const vectorCount = vectorCounts.reduce((sum, value) => safeSum(sum, value), 0);
  if (vectorCount == null) return fail("vectors_per_class_sum_overflow");
  const classCount = labels.count;
  if (classCount < 2 || classCount >= 65_536) return fail("class_count_outside_pinned_ort_range_2_to_65535");
  if (!attributes.coefficients.length) return fail("coefficients_empty");
  if (!attributes.rho.length) return fail("rho_empty");
  if (attributes.probA.length !== attributes.probB.length) return fail("prob_a_prob_b_size_mismatch");
  const pairCount = safeProduct(classCount, classCount - 1) / 2;
  const mode = vectorCount > 0 ? "svc" : "linear";
  let featureCount = null;
  let expectedCoefficientCount = null;
  let expectedSupportVectorCount = 0;
  let expectedRhoCount = mode === "svc" ? pairCount : 1;
  let expectedProbabilityCount = mode === "svc" && attributes.probA.length ? pairCount : 0;
  if (mode === "svc") {
    if (vectorCounts.length !== classCount) return fail("vectors_per_class_size_must_equal_class_count", { mode, vectorCount, pairCount });
    if (attributes.supportVectors.length % vectorCount !== 0) return fail("support_vectors_size_must_be_divisible_by_vector_count", { mode, vectorCount, pairCount });
    featureCount = attributes.supportVectors.length / vectorCount;
    if (featureCount < 1) return fail("support_vector_feature_count_must_be_positive", { mode, vectorCount, pairCount });
    expectedSupportVectorCount = safeProduct(vectorCount, featureCount);
    expectedCoefficientCount = safeProduct(classCount - 1, vectorCount);
    if (attributes.coefficients.length < expectedCoefficientCount) return fail("coefficients_smaller_than_svc_layout", { mode, vectorCount, pairCount, expectedCoefficientCount });
    if (attributes.rho.length < expectedRhoCount) return fail("rho_smaller_than_pairwise_classifier_count", { mode, vectorCount, pairCount, expectedRhoCount });
    if (attributes.probA.length && attributes.probA.length < pairCount) return fail("probability_parameters_smaller_than_pairwise_classifier_count", { mode, vectorCount, pairCount, expectedProbabilityCount });
  } else {
    featureCount = Math.floor(attributes.coefficients.length / classCount);
    if (featureCount < 1) return fail("linear_feature_count_must_be_positive", { mode, vectorCount, pairCount });
    expectedCoefficientCount = safeProduct(classCount, featureCount);
  }
  if (knownDimension(input.featureCount) && input.featureCount !== featureCount) {
    return fail("artifact_feature_count_differs_from_pinned_ort_parameter_layout", { mode, vectorCount, pairCount, featureCount, expectedCoefficientCount, expectedSupportVectorCount, expectedRhoCount, expectedProbabilityCount });
  }
  const partial = !knownDimension(input.featureCount);
  const usedCoefficientCount = Math.min(attributes.coefficients.length, expectedCoefficientCount);
  const usedSupportVectorCount = Math.min(attributes.supportVectors.length, expectedSupportVectorCount);
  const usedRhoCount = Math.min(attributes.rho.length, expectedRhoCount);
  const usedProbabilityCount = expectedProbabilityCount ? expectedProbabilityCount * 2 : 0;
  return {
    status: partial ? "partial" : "pass", reason: partial ? "artifact_feature_count_unresolved" : "",
    mode, vectorCount, pairCount, featureCount,
    scoreWidth: mode === "svc" && !attributes.probA.length ? (classCount > 2 ? pairCount : 2) : classCount,
    haveProbability: attributes.probA.length > 0,
    expectedCoefficientCount, usedCoefficientCount, unusedCoefficientCount: attributes.coefficients.length - usedCoefficientCount,
    expectedSupportVectorCount, usedSupportVectorCount, unusedSupportVectorCount: attributes.supportVectors.length - usedSupportVectorCount,
    expectedRhoCount, usedRhoCount, unusedRhoCount: attributes.rho.length - usedRhoCount,
    expectedProbabilityCount, usedProbabilityCount,
    unusedProbabilityCount: attributes.probA.length + attributes.probB.length - usedProbabilityCount,
  };
}

function regressorRuntimeContract(input, attributes) {
  const failureContext = regressorRuntimeFailureContext(attributes);
  const fail = (reason, extra = {}) => runtimeFailure(reason, { ...failureContext, ...extra });
  if (!attributes.kernelParamsPresent) return fail("kernel_params_attribute_missing_in_pinned_ort_constructor");
  if (!attributes.coefficients.length) return fail("coefficients_empty");
  if (!attributes.rho.length) return fail("rho_empty");
  if (attributes.nSupports == null || attributes.nSupports < 0n || attributes.nSupports > BigInt(Number.MAX_SAFE_INTEGER)) {
    return fail("n_supports_not_nonnegative_safe_integer");
  }
  const vectorCount = Number(attributes.nSupports);
  const mode = vectorCount > 0 ? "svc" : "linear";
  let featureCount;
  let expectedCoefficientCount;
  let expectedSupportVectorCount = 0;
  if (mode === "svc") {
    if (attributes.coefficients.length < vectorCount) return fail("coefficients_smaller_than_n_supports", { mode, vectorCount });
    if (!attributes.supportVectors.length || attributes.supportVectors.length % vectorCount !== 0) {
      return fail("support_vectors_must_be_nonempty_multiple_of_n_supports", { mode, vectorCount });
    }
    featureCount = attributes.supportVectors.length / vectorCount;
    expectedCoefficientCount = vectorCount;
    expectedSupportVectorCount = safeProduct(vectorCount, featureCount);
  } else {
    featureCount = attributes.coefficients.length;
    expectedCoefficientCount = featureCount;
  }
  if (knownDimension(input.featureCount) && input.featureCount !== featureCount) {
    return fail("artifact_feature_count_differs_from_pinned_ort_parameter_layout", { mode, vectorCount, featureCount, expectedCoefficientCount, expectedSupportVectorCount });
  }
  const usedCoefficientCount = Math.min(attributes.coefficients.length, expectedCoefficientCount);
  const usedSupportVectorCount = Math.min(attributes.supportVectors.length, expectedSupportVectorCount);
  return {
    status: knownDimension(input.featureCount) ? "pass" : "partial",
    reason: knownDimension(input.featureCount) ? "" : "artifact_feature_count_unresolved",
    mode, vectorCount, featureCount,
    expectedCoefficientCount, usedCoefficientCount, unusedCoefficientCount: attributes.coefficients.length - usedCoefficientCount,
    expectedSupportVectorCount, usedSupportVectorCount, unusedSupportVectorCount: attributes.supportVectors.length - usedSupportVectorCount,
    expectedRhoCount: 1, usedRhoCount: 1, unusedRhoCount: Math.max(0, attributes.rho.length - 1),
  };
}

function classifierRuntimeFailureContext(attributes, labels) {
  const vectorCounts = attributes.vectorsPerClass.every((value) => value >= 0n && value <= BigInt(Number.MAX_SAFE_INTEGER))
    ? attributes.vectorsPerClass.map(Number) : null;
  const vectorCount = vectorCounts?.reduce((sum, value) => safeSum(sum, value), 0) ?? null;
  const classCount = labels.count;
  const pairCount = classCount >= 2 ? safeProduct(classCount, classCount - 1) / 2 : 0;
  const mode = vectorCount == null ? "unresolved" : vectorCount > 0 ? "svc" : "linear";
  const featureCount = mode === "svc"
    ? vectorCount > 0 && attributes.supportVectors.length % vectorCount === 0
      ? attributes.supportVectors.length / vectorCount : null
    : mode === "linear" && classCount > 0 ? Math.floor(attributes.coefficients.length / classCount) : null;
  return {
    mode, vectorCount: vectorCount ?? 0, pairCount, featureCount,
    haveProbability: attributes.probA.length > 0,
    expectedCoefficientCount: mode === "svc" && classCount >= 2 && vectorCount != null
      ? safeProduct(classCount - 1, vectorCount)
      : mode === "linear" && featureCount != null ? safeProduct(classCount, featureCount) : null,
    expectedSupportVectorCount: mode === "linear" ? 0
      : featureCount == null || vectorCount == null ? null : safeProduct(vectorCount, featureCount),
    expectedRhoCount: mode === "svc" ? pairCount : mode === "linear" ? 1 : null,
    expectedProbabilityCount: mode === "svc" && attributes.probA.length ? pairCount : 0,
  };
}

function regressorRuntimeFailureContext(attributes) {
  const vectorCount = attributes.nSupports != null && attributes.nSupports >= 0n
    && attributes.nSupports <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(attributes.nSupports) : null;
  const mode = vectorCount == null ? "unresolved" : vectorCount > 0 ? "svc" : "linear";
  const featureCount = mode === "svc"
    ? vectorCount > 0 && attributes.supportVectors.length > 0 && attributes.supportVectors.length % vectorCount === 0
      ? attributes.supportVectors.length / vectorCount : null
    : mode === "linear" ? attributes.coefficients.length : null;
  return {
    mode, vectorCount: vectorCount ?? 0, featureCount,
    expectedCoefficientCount: mode === "svc" ? vectorCount
      : mode === "linear" ? featureCount : null,
    expectedSupportVectorCount: mode === "linear" ? 0
      : featureCount == null || vectorCount == null ? null : safeProduct(vectorCount, featureCount),
    expectedRhoCount: 1,
  };
}

function runtimeFailure(reason, extra = {}) {
  return {
    status: "fail", reason, mode: extra.mode || "unresolved", vectorCount: extra.vectorCount ?? 0,
    pairCount: extra.pairCount ?? 0, featureCount: extra.featureCount ?? null, scoreWidth: null,
    haveProbability: extra.haveProbability ?? false,
    expectedCoefficientCount: extra.expectedCoefficientCount ?? null, usedCoefficientCount: 0, unusedCoefficientCount: 0,
    expectedSupportVectorCount: extra.expectedSupportVectorCount ?? null, usedSupportVectorCount: 0, unusedSupportVectorCount: 0,
    expectedRhoCount: extra.expectedRhoCount ?? null, usedRhoCount: 0, unusedRhoCount: 0,
    expectedProbabilityCount: extra.expectedProbabilityCount ?? null, usedProbabilityCount: 0, unusedProbabilityCount: 0,
  };
}

function evaluateClassifierReference({ input, attributes, labels, runtime }) {
  if (runtime.status === "fail") return unresolvedReference("not_assessed_invalid_pinned_ort_contract");
  const referenceWork = safeSum(runtime.vectorCount || labels.count, runtime.pairCount || labels.count);
  const probabilityCells = runtime.haveProbability ? safeProduct(labels.count, labels.count) : 0;
  if (referenceWork == null || referenceWork > MAX_REFERENCE_VALUES
    || probabilityCells == null || probabilityCells > MAX_REFERENCE_VALUES) {
    return unresolvedReference("not_assessed_reference_work_limit");
  }
  const values = staticNumericInput(input.input, input.dtype);
  if (!values) return unresolvedReference(input.input?.role === "initializer" ? "not_assessed_initializer_values" : "not_assessed_runtime_values");
  const expected = safeProduct(input.batchCount, input.featureCount);
  if (expected == null || values.length !== expected || expected > MAX_REFERENCE_VALUES) return unresolvedReference("not_assessed_static_input_cardinality_or_limit");
  const converted = values.map((value) => toFloat32(value, input.dtype));
  const rawScores = [];
  const outputScores = [];
  const outputLabels = [];
  let decisionBoundaryCount = 0;
  for (let batch = 0; batch < input.batchCount; batch += 1) {
    const row = converted.slice(batch * input.featureCount, (batch + 1) * input.featureCount);
    if (runtime.mode === "linear") {
      const scores = [];
      for (let cls = 0; cls < labels.count; cls += 1) {
        const weights = attributes.coefficients.slice(cls * runtime.featureCount, (cls + 1) * runtime.featureCount);
        scores.push(kernelReference(row, weights, "LINEAR", [], attributes.rho[0]));
      }
      rawScores.push(...scores);
      const max = argMax(scores);
      if (labels.count === 2) {
        const allPositive = attributes.coefficients.every((value) => value >= 0);
        let selected = max.index;
        if (!runtime.haveProbability) {
          if (allPositive && max.value >= 0.5) selected = 1;
          else if (!allPositive && max.value > 0) selected = 1;
        }
        outputLabels.push(labels.values[selected]);
        if (max.value === 0 || allPositive && max.value === 0.5) decisionBoundaryCount += 1;
      } else {
        outputLabels.push(labels.values[max.index]);
      }
      outputScores.push(...applyPostTransform(scores, attributes.postTransform));
    } else {
      const kernels = [];
      for (let vector = 0; vector < runtime.vectorCount; vector += 1) {
        const support = attributes.supportVectors.slice(vector * runtime.featureCount, (vector + 1) * runtime.featureCount);
        kernels.push(kernelReference(row, support, attributes.kernelType, attributes.kernelParams, 0));
      }
      const pairScores = [];
      const votes = new Array(labels.count).fill(0);
      const starts = [];
      let offset = 0;
      for (const count of attributes.vectorsPerClass) { starts.push(offset); offset += Number(count); }
      let pair = 0;
      for (let i = 0; i < labels.count - 1; i += 1) {
        for (let j = i + 1; j < labels.count; j += 1) {
          let sum = 0;
          for (let m = 0; m < Number(attributes.vectorsPerClass[i]); m += 1) {
            const coefficient = attributes.coefficients[runtime.vectorCount * (j - 1) + starts[i] + m];
            sum += Math.fround(coefficient * kernels[starts[i] + m]);
          }
          for (let m = 0; m < Number(attributes.vectorsPerClass[j]); m += 1) {
            const coefficient = attributes.coefficients[runtime.vectorCount * i + starts[j] + m];
            sum += Math.fround(coefficient * kernels[starts[j] + m]);
          }
          const score = Math.fround(sum + attributes.rho[pair]);
          pairScores.push(score);
          votes[score > 0 ? i : j] += 1;
          if (score === 0) decisionBoundaryCount += 1;
          pair += 1;
        }
      }
      rawScores.push(...pairScores);
      outputLabels.push(labels.values[argMax(votes).index]);
      if (runtime.haveProbability) {
        const probabilities = svmProbabilities(pairScores, attributes.probA, attributes.probB, labels.count);
        outputScores.push(...applyPostTransform(probabilities, attributes.postTransform));
      } else if (labels.count === 2) {
        const score = pairScores[0];
        if (attributes.postTransform === "PROBIT") outputScores.push(probit(score), Number.NaN);
        else if (["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(attributes.postTransform)) outputScores.push(Math.fround(1 - score), score);
        else outputScores.push(Math.fround(-score), score);
      } else {
        outputScores.push(...applyPostTransform(pairScores, attributes.postTransform));
      }
    }
  }
  const nonFiniteScoreCount = [...rawScores, ...outputScores].filter((value) => !Number.isFinite(value)).length;
  return {
    status: "assessed_scalar_float32_reference_not_runtime_bit_exact",
    inputValueCount: values.length, rawScoreCount: rawScores.length, nonFiniteScoreCount, decisionBoundaryCount,
    rawScorePreview: rawScores.slice(0, 12).map(valueText),
    outputScorePreview: outputScores.slice(0, 12).map(valueText),
    labelPreview: outputLabels.slice(0, 12).map(valueText),
  };
}

function evaluateRegressorReference({ input, attributes, runtime, cpuDtypeGap }) {
  if (runtime.status === "fail" || cpuDtypeGap) return unresolvedReference(cpuDtypeGap ? "not_assessed_pinned_ort_cpu_dtype_gap" : "not_assessed_invalid_pinned_ort_contract");
  const values = staticNumericInput(input.input, input.dtype);
  if (!values) return unresolvedReference(input.input?.role === "initializer" ? "not_assessed_initializer_values" : "not_assessed_runtime_values");
  const expected = safeProduct(input.batchCount, input.featureCount);
  if (expected == null || values.length !== expected || expected > MAX_REFERENCE_VALUES) return unresolvedReference("not_assessed_static_input_cardinality_or_limit");
  const converted = values.map((value) => toFloat32(value, input.dtype));
  const rawScores = [];
  for (let batch = 0; batch < input.batchCount; batch += 1) {
    const row = converted.slice(batch * input.featureCount, (batch + 1) * input.featureCount);
    let score;
    if (runtime.mode === "linear") {
      score = kernelReference(row, attributes.coefficients.slice(0, runtime.featureCount), "LINEAR", [], attributes.rho[0]);
    } else {
      const kernels = [];
      for (let vector = 0; vector < runtime.vectorCount; vector += 1) {
        const support = attributes.supportVectors.slice(vector * runtime.featureCount, (vector + 1) * runtime.featureCount);
        kernels.push(kernelReference(row, support, attributes.kernelType, attributes.kernelParams, 0));
      }
      score = kernelReference(kernels, attributes.coefficients.slice(0, runtime.vectorCount), "LINEAR", [], attributes.rho[0]);
    }
    if (attributes.oneClass !== 0n) score = score > 0 ? 1 : -1;
    rawScores.push(Math.fround(score));
  }
  return {
    status: "assessed_scalar_float32_reference_not_runtime_bit_exact",
    inputValueCount: values.length, rawScoreCount: rawScores.length,
    nonFiniteScoreCount: rawScores.filter((value) => !Number.isFinite(value)).length,
    decisionBoundaryCount: rawScores.filter((value) => value === 0).length,
    rawScorePreview: rawScores.slice(0, 12).map(valueText),
    outputScorePreview: rawScores.slice(0, 12).map(valueText), labelPreview: [],
  };
}

function kernelReference(input, weights, kernelType, params, rho) {
  const gamma = params[0] ?? 0;
  const coef0 = params[1] ?? 0;
  const degree = params[2] ?? 0;
  if (kernelType === "RBF") {
    let sum = 0;
    for (let index = 0; index < input.length; index += 1) {
      const difference = Math.fround(input[index] - weights[index]);
      sum = Math.fround(sum + Math.fround(difference * difference));
    }
    return Math.fround(Math.exp(Math.fround(-gamma * sum)));
  }
  let dot = 0;
  for (let index = 0; index < input.length; index += 1) dot = Math.fround(dot + Math.fround(input[index] * weights[index]));
  if (kernelType === "LINEAR") return Math.fround(dot + (rho || 0));
  const affine = Math.fround(Math.fround(gamma * dot) + coef0);
  if (kernelType === "POLY") return Math.fround(Math.pow(affine, degree));
  return Math.fround(Math.tanh(affine));
}

function svmProbabilities(scores, probA, probB, classCount) {
  const matrix = new Array(classCount * classCount).fill(0);
  let pair = 0;
  for (let i = 0; i < classCount - 1; i += 1) {
    for (let j = i + 1; j < classCount; j += 1) {
      const value = Math.min(Math.max(sigmoidProbability(scores[pair], probA[pair], probB[pair]), 1e-7), 1 - 1e-7);
      matrix[i * classCount + j] = Math.fround(value);
      matrix[j * classCount + i] = Math.fround(1 - value);
      pair += 1;
    }
  }
  return multiclassProbability(matrix, classCount);
}

function multiclassProbability(matrix, classCount) {
  const q = new Array(classCount * classCount).fill(0);
  const qp = new Array(classCount).fill(0);
  const p = new Array(classCount).fill(Math.fround(1 / classCount));
  const epsilon = Math.fround(0.005 / classCount);
  for (let i = 0; i < classCount; i += 1) {
    for (let j = 0; j < i; j += 1) {
      q[i * classCount + i] = Math.fround(q[i * classCount + i] + Math.fround(matrix[j * classCount + i] ** 2));
      q[i * classCount + j] = q[j * classCount + i];
    }
    for (let j = i + 1; j < classCount; j += 1) {
      q[i * classCount + i] = Math.fround(q[i * classCount + i] + Math.fround(matrix[j * classCount + i] ** 2));
      q[i * classCount + j] = Math.fround(-matrix[j * classCount + i] * matrix[i * classCount + j]);
    }
  }
  for (let loop = 0; loop < 100; loop += 1) {
    let pQp = 0;
    let maxError = 0;
    for (let i = 0; i < classCount; i += 1) {
      qp[i] = 0;
      for (let j = 0; j < classCount; j += 1) qp[i] = Math.fround(qp[i] + Math.fround(q[i * classCount + j] * p[j]));
      pQp = Math.fround(pQp + Math.fround(p[i] * qp[i]));
    }
    for (let i = 0; i < classCount; i += 1) maxError = Math.max(maxError, Math.abs(qp[i] - pQp));
    if (maxError < epsilon) break;
    for (let i = 0; i < classCount; i += 1) {
      const difference = Math.fround((-qp[i] + pQp) / q[i * classCount + i]);
      p[i] = Math.fround(p[i] + difference);
      const denominator = Math.fround((1 + difference) * (1 + difference));
      pQp = Math.fround((pQp + difference * (difference * q[i * classCount + i] + 2 * qp[i])) / denominator);
      for (let j = 0; j < classCount; j += 1) {
        qp[j] = Math.fround((qp[j] + difference * q[i * classCount + j]) / (1 + difference));
        p[j] = Math.fround(p[j] / (1 + difference));
      }
    }
  }
  return p;
}

function sigmoidProbability(score, a, b) {
  return Math.fround(1 - logistic(Math.fround(Math.fround(score * a) + b)));
}

function applyPostTransform(values, transform) {
  if (transform === "LOGISTIC") return values.map(logistic);
  if (transform === "PROBIT") return values.map(probit);
  if (transform === "SOFTMAX") return softmax(values, false);
  if (transform === "SOFTMAX_ZERO") return softmax(values, true);
  return [...values];
}

function logistic(value) {
  const v = Math.fround(1 / (1 + Math.exp(-Math.abs(value))));
  return value < 0 ? Math.fround(1 - v) : v;
}

function probit(value) {
  const sign = 2 * value - 1 < 0 ? -1 : 1;
  const x = Math.fround((1 - (2 * value - 1)) * (1 + (2 * value - 1)));
  const log = Math.fround(Math.log(x));
  const v = Math.fround(2 / (Math.PI * 0.147) + 0.5 * log);
  const v2 = Math.fround(log / 0.147);
  return Math.fround(Math.SQRT2 * sign * Math.sqrt(-v + Math.sqrt(v * v - v2)));
}

function softmax(values, zeroAware) {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) maximum = Math.max(maximum, value);
  const output = values.map((value) => zeroAware && Math.abs(value) <= 1e-7
    ? Math.fround(value * Math.exp(-maximum)) : Math.fround(Math.exp(value - maximum)));
  const sum = output.reduce((total, value) => Math.fround(total + value), 0);
  return output.map((value) => Math.fround(value / sum));
}

function classifierOutputShapes(input, width) {
  if (![1, 2].includes(input.rank) || !Number.isSafeInteger(width) || width < 0) return { declared: false, label: [], score: [] };
  const batch = input.rank === 1 ? 1 : input.batchCount;
  return { declared: true, label: [batch], score: [batch, width] };
}

function svmBaseRow({ scope, nodeIndex, importedOpset, opName, contractKind, status, input, outputName, outputDtype, outputShape, outputShapeDeclared, exactOutputElements, failures, reasons }) {
  return {
    scope, node_index: nodeIndex, op_name: opName, contract_kind: contractKind,
    imported_opset: importedOpset, resolved_schema_version: 1, status,
    input_name: input.input?.name || "", output_name: outputName,
    input_dtype: input.dtype, input_kind: input.type?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: input.rank, input_shape: input.shape,
    exact_batch_count: input.batchCount, exact_feature_count: input.featureCount,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: canonicalOnnxTypeProto(makeOnnxTensorType(outputDtype, outputShape, outputShapeDeclared)),
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: outputShapeDeclared ? outputShape.length : null, exact_output_shape: outputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_svm_schema_and_pinned_ort_cpu_runtime_contract",
    runtime_reference_status: "pinned_ort_cpu_svm_kernel_and_ml_common",
    attribute_mode: "svm_kernel_support_vector_coefficient_rho_probability_label_contract",
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
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: [],
  };
}

function staticNumericInput(input, dtype) {
  if (dtype === "INT64" && input?.initializerIntegerValuesExactComplete === true
    && Array.isArray(input.initializerIntegerValuesExactDecimals)) {
    try { return input.initializerIntegerValuesExactDecimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (["FLOAT32", "FLOAT64"].includes(dtype) && input?.staticValuesCanonicalTextComplete === true
    && Array.isArray(input.staticValuesCanonicalTexts)) return input.staticValuesCanonicalTexts.map(parseCanonicalNumber);
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) return input.staticValues;
  return null;
}

function parseCanonicalNumber(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

function toFloat32(value, dtype) {
  if (dtype === "INT64" && typeof value === "bigint") return bigintToFloat32(value);
  return Math.fround(Number(value));
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
  if (significand === (1n << 24n)) { significand >>= 1n; exponent += 1; }
  return Math.fround((negative ? -1 : 1) * Number(significand) * (2 ** (exponent - 23)));
}

function floatListAttribute(attribute) {
  if (attribute?.type !== 6 || !Array.isArray(attribute.floats)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length > 1
    || attribute.valueTypesPresent.length === 1 && attribute.valueTypesPresent[0] !== 6) return null;
  return attribute.floats.map((value) => Math.fround(value));
}

function intListAttribute(attribute) {
  if (attribute?.type !== 7 || !Array.isArray(attribute.ints)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length > 1
    || attribute.valueTypesPresent.length === 1 && attribute.valueTypesPresent[0] !== 7) return null;
  const exact = Array.isArray(attribute.intExactDecimals) && attribute.intExactDecimals.length === attribute.ints.length
    ? attribute.intExactDecimals : attribute.ints;
  try { return exact.map((value) => BigInt(value)); } catch { return null; }
}

function intScalarAttribute(attribute) {
  if (attribute?.type !== 2 || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== 2) return null;
  try { return BigInt(attribute.iExactDecimal || attribute.i); } catch { return null; }
}

function stringScalarAttribute(attribute) {
  if (attribute?.type !== 3 || typeof attribute.s !== "string"
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 3) return null;
  return attribute.s;
}

function stringList(node, name, failures) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return [];
  if (attribute.type !== 8 || !Array.isArray(attribute.strings)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length > 1
    || attribute.valueTypesPresent.length === 1 && attribute.valueTypesPresent[0] !== 8) {
    failures.push(`svm_classifier_${name}_not_string_list`);
    return [];
  }
  return [...attribute.strings];
}

function unresolvedReference(status) {
  return {
    status, inputValueCount: null, rawScoreCount: null, nonFiniteScoreCount: null, decisionBoundaryCount: null,
    rawScorePreview: [], outputScorePreview: [], labelPreview: [],
  };
}

function argMax(values) {
  let index = 0;
  for (let candidate = 1; candidate < values.length; candidate += 1) if (values[candidate] > values[index]) index = candidate;
  return { index, value: values[index] };
}

function duplicateValueCount(values) {
  return values.length - new Set(values.map((value) => typeof value === "bigint" ? `i:${value}` : `s:${value}`)).size;
}

function valueText(value) {
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
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0) return null;
  const value = left * right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function safeSum(left, right) {
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0) return null;
  const value = left + right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}
