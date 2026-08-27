import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

const ALLOWED_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const POST_TRANSFORMS = new Set(["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"]);
const MAX_REFERENCE_VALUES = 1_000_000;

export function inferOnnxMlLinearClassifier(context) {
  return inferLinearClassifier(context);
}

export function inferOnnxMlLinearRegressor(context) {
  return inferLinearRegressor(context);
}

function inferLinearClassifier({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const common = linearInputContract(node, tensorMap);
  const failures = [...common.failures];
  const reasons = [...common.reasons];
  const coefficientsAttribute = node.attributes?.get("coefficients");
  const interceptsAttribute = node.attributes?.get("intercepts");
  const intLabelsAttribute = node.attributes?.get("classlabels_ints");
  const stringLabelsAttribute = node.attributes?.get("classlabels_strings");
  const multiClassAttribute = node.attributes?.get("multi_class");
  const postTransformAttribute = node.attributes?.get("post_transform");
  const coefficients = coefficientsAttribute ? floatListAttribute(coefficientsAttribute) : null;
  const intercepts = interceptsAttribute ? floatListAttribute(interceptsAttribute) : [];
  const intLabels = intLabelsAttribute ? intListAttribute(intLabelsAttribute) : [];
  const stringLabels = stringLabelsAttribute ? stringListAttribute(stringLabelsAttribute) : [];
  const multiClass = multiClassAttribute ? intScalarAttribute(multiClassAttribute) : 0n;
  const postTransform = postTransformAttribute ? stringScalarAttribute(postTransformAttribute) : "NONE";

  if (!coefficientsAttribute) failures.push("linear_classifier_coefficients_required_by_onnx_schema");
  else if (coefficients == null) failures.push("linear_classifier_coefficients_not_float_list");
  if (interceptsAttribute && intercepts == null) failures.push("linear_classifier_intercepts_not_float_list");
  if (intLabelsAttribute && intLabels == null) failures.push("linear_classifier_int_labels_not_exact_int_list");
  if (stringLabelsAttribute && stringLabels == null) failures.push("linear_classifier_string_labels_not_string_list");
  if (multiClass == null) failures.push("linear_classifier_multi_class_not_exact_int_scalar");
  if (postTransform == null || !POST_TRANSFORMS.has(postTransform)) failures.push("linear_classifier_post_transform_invalid");

  const coefficientValues = coefficients || [];
  const interceptValues = intercepts || [];
  const integerLabels = intLabels || [];
  const textLabels = stringLabels || [];
  const labelContract = classifierLabelContract(integerLabels, textLabels, interceptValues.length);
  if (labelContract.status === "fail") failures.push(labelContract.reason);
  const runtimeContract = classifierRuntimeContract(common, coefficientValues, interceptValues);
  if (runtimeContract.status === "fail") failures.push(runtimeContract.reason);
  else if (runtimeContract.status === "partial") reasons.push(runtimeContract.reason);

  const classCount = interceptValues.length;
  const expandedBinary = classCount === 1 && labelContract.labelCount === 2;
  const scoreClassCount = expandedBinary ? 2 : classCount;
  const labelOutputType = labelContract.labelKind === "string" ? "STRING" : "INT64";
  const outputShapes = classifierOutputShapes(common, scoreClassCount);
  const labelType = makeOnnxTensorType(labelOutputType, outputShapes.labelShape, outputShapes.shapeDeclared);
  const scoreType = makeOnnxTensorType("FLOAT32", outputShapes.scoreShape, outputShapes.shapeDeclared);
  const labelPatch = onnxValueDescriptorFromType(labelType);
  const scorePatch = onnxValueDescriptorFromType(scoreType);
  const exactScoreElements = outputShapes.shapeDeclared && outputShapes.scoreShape.every(knownDimension)
    ? safeShapeElementCount(outputShapes.scoreShape) : null;
  const duplicateLabels = duplicateValueCount(labelContract.labels);
  const nonFiniteParameterCount = [...coefficientValues, ...interceptValues].filter((value) => !Number.isFinite(value)).length;
  const reference = evaluateClassifierReference({
    input: common.input,
    dtype: common.inputDtype,
    batchCount: common.batchCount,
    featureCount: common.featureCount,
    classCount,
    scoreClassCount,
    coefficients: coefficientValues,
    intercepts: interceptValues,
    labels: labelContract.labels,
    postTransform: postTransform || "NONE",
    expandedBinary,
    runtimeContract,
  });

  const riskCodes = [];
  if (labelContract.status === "fail") riskCodes.push("linear_classifier_onnx_label_contract_invalid");
  if (runtimeContract.status === "fail") riskCodes.push("linear_classifier_pinned_ort_runtime_contract_invalid");
  if (runtimeContract.unusedCoefficientCount > 0) riskCodes.push("linear_classifier_unused_coefficients_ignored");
  if (duplicateLabels > 0) riskCodes.push("linear_classifier_duplicate_labels_ambiguous_output_semantics");
  if (multiClass != null && multiClass !== 0n) riskCodes.push("linear_classifier_multi_class_nonzero_ignored_by_pinned_ort");
  if (classCount === 1 && !expandedBinary && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform)) {
    riskCodes.push("linear_classifier_single_score_post_transform_noop");
  }
  if (expandedBinary && postTransform === "PROBIT") riskCodes.push("linear_classifier_binary_probit_second_score_unwritten");
  if (expandedBinary && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform)) {
    riskCodes.push("linear_classifier_binary_post_transform_ignored_for_complement_expansion");
  }
  if (nonFiniteParameterCount > 0 || reference.nonFiniteRawScoreCount > 0) {
    riskCodes.push("linear_classifier_non_finite_parameter_or_reference_score");
  }
  if (reference.decisionBoundaryCount > 0) riskCodes.push("linear_classifier_reference_decision_boundary");

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = linearBaseRow({
    scope, nodeIndex, importedOpset, opName: "LinearClassifier", contractKind: "linear_classifier",
    status, common, outputName: node.outputs?.[1] || "", outputDtype: "FLOAT32",
    outputShape: outputShapes.scoreShape, outputShapeDeclared: outputShapes.shapeDeclared,
    exactOutputElements: exactScoreElements, failures, reasons,
  });
  Object.assign(row, {
    output_names: [...(node.outputs || [])],
    canonical_output_types: [canonicalOnnxTypeProto(labelType), canonicalOnnxTypeProto(scoreType)],
    canonical_output_shapes: [outputShapes.labelShape, outputShapes.scoreShape],
    classifier_label_output_name: node.outputs?.[0] || "",
    classifier_score_output_name: node.outputs?.[1] || "",
    classifier_label_output_dtype: labelOutputType,
    classifier_label_output_shape: outputShapes.labelShape,
    classifier_score_output_shape: outputShapes.scoreShape,
    classifier_score_class_count: scoreClassCount,
    classifier_binary_score_expansion: expandedBinary,
    linear_onnx_contract_status: labelContract.status,
    linear_onnx_contract_reason: labelContract.reason,
    linear_pinned_ort_contract_status: runtimeContract.status,
    linear_pinned_ort_contract_reason: runtimeContract.reason,
    linear_class_or_target_count: classCount,
    linear_expected_coefficient_count: runtimeContract.expectedCoefficientCount,
    linear_coefficient_count: coefficientValues.length,
    linear_used_coefficient_count: runtimeContract.usedCoefficientCount,
    linear_unused_coefficient_count: runtimeContract.unusedCoefficientCount,
    linear_intercept_count: interceptValues.length,
    linear_intercepts_used: interceptValues.length > 0,
    linear_label_kind: labelContract.labelKind,
    linear_label_count: labelContract.labelCount,
    linear_label_values: labelContract.labels.map(exactValueText),
    linear_duplicate_label_count: duplicateLabels,
    linear_multi_class_value: multiClass?.toString() ?? "unresolved",
    linear_multi_class_source: multiClassAttribute ? "explicit_attribute" : "onnx_schema_default_0",
    linear_multi_class_used_by_pinned_ort: false,
    linear_post_transform: postTransform || "unresolved",
    linear_post_transform_source: postTransformAttribute ? "explicit_attribute" : "onnx_schema_default_NONE",
    linear_non_finite_parameter_count: nonFiniteParameterCount,
    linear_reference_assessment_status: reference.status,
    linear_reference_input_value_count: reference.inputValueCount,
    linear_reference_raw_score_count: reference.rawScoreCount,
    linear_reference_non_finite_raw_score_count: reference.nonFiniteRawScoreCount,
    linear_reference_decision_boundary_count: reference.decisionBoundaryCount,
    linear_reference_raw_score_preview: reference.rawScorePreview,
    linear_reference_output_preview: reference.outputPreview,
    linear_reference_label_preview: reference.labelPreview,
    linear_reference_boundary: "Deterministic scalar FLOAT32 reference only; pinned ORT uses MLAS GEMM and selected microkernel accumulation order is not observed, so values are not promoted as runtime-bit-exact tensor evidence.",
    risk_codes: riskCodes,
  });
  const canPropagate = status !== "fail" && outputShapes.shapeDeclared;
  const outputs = [];
  if (canPropagate && node.outputs?.[0]) outputs.push([node.outputs[0], labelPatch]);
  if (canPropagate && node.outputs?.[1]) outputs.push([node.outputs[1], scorePatch]);
  return { status, reason: row.reason_codes[0] || "", result: { outputs }, row };
}

function inferLinearRegressor({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const common = linearInputContract(node, tensorMap);
  const failures = [...common.failures];
  const reasons = [...common.reasons];
  const coefficientsAttribute = node.attributes?.get("coefficients");
  const interceptsAttribute = node.attributes?.get("intercepts");
  const targetsAttribute = node.attributes?.get("targets");
  const postTransformAttribute = node.attributes?.get("post_transform");
  const coefficients = coefficientsAttribute ? floatListAttribute(coefficientsAttribute) : [];
  const intercepts = interceptsAttribute ? floatListAttribute(interceptsAttribute) : [];
  const targets = targetsAttribute ? intScalarAttribute(targetsAttribute) : 1n;
  const postTransform = postTransformAttribute ? stringScalarAttribute(postTransformAttribute) : "NONE";

  if (coefficientsAttribute && coefficients == null) failures.push("linear_regressor_coefficients_not_float_list");
  if (interceptsAttribute && intercepts == null) failures.push("linear_regressor_intercepts_not_float_list");
  if (targets == null) failures.push("linear_regressor_targets_not_exact_int_scalar");
  if (postTransform == null || !POST_TRANSFORMS.has(postTransform)) failures.push("linear_regressor_post_transform_invalid");
  const targetCount = targets != null && targets > 0n && targets <= 2_147_483_647n ? Number(targets) : null;
  if (targetCount == null) failures.push("linear_regressor_targets_outside_pinned_ort_range");

  const coefficientValues = coefficients || [];
  const interceptValues = intercepts || [];
  const runtimeContract = regressorRuntimeContract(common, coefficientValues, interceptValues, targetCount);
  if (runtimeContract.status === "fail") failures.push(runtimeContract.reason);
  else if (runtimeContract.status === "partial") reasons.push(runtimeContract.reason);
  const outputShapeDeclared = common.rankValid && common.batchCount != null && targetCount != null;
  const outputShape = outputShapeDeclared ? [common.batchCount, targetCount] : [];
  const outputType = makeOnnxTensorType("FLOAT32", outputShape, outputShapeDeclared);
  const outputPatch = onnxValueDescriptorFromType(outputType);
  const exactOutputElements = outputShapeDeclared ? safeShapeElementCount(outputShape) : null;
  const nonFiniteParameterCount = [...coefficientValues, ...interceptValues].filter((value) => !Number.isFinite(value)).length;
  const reference = evaluateRegressorReference({
    input: common.input,
    dtype: common.inputDtype,
    batchCount: common.batchCount,
    featureCount: common.featureCount,
    targetCount,
    coefficients: coefficientValues,
    intercepts: runtimeContract.interceptsUsed ? interceptValues : [],
    postTransform: postTransform || "NONE",
    runtimeContract,
  });

  const riskCodes = [];
  if (runtimeContract.status === "fail") riskCodes.push("linear_regressor_pinned_ort_runtime_contract_invalid");
  if (ALLOWED_DTYPES.has(common.inputDtype) && common.inputDtype !== "FLOAT32") {
    riskCodes.push("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  }
  if (runtimeContract.unusedCoefficientCount > 0) riskCodes.push("linear_regressor_unused_coefficients_ignored");
  if (runtimeContract.interceptsIgnoredCount > 0) riskCodes.push("linear_regressor_mismatched_intercepts_ignored");
  if (targetCount === 1 && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform)) {
    riskCodes.push("linear_regressor_single_target_post_transform_noop");
  }
  if (postTransform === "PROBIT") riskCodes.push("linear_regressor_probit_may_emit_non_finite");
  if (nonFiniteParameterCount > 0 || reference.nonFiniteRawScoreCount > 0) {
    riskCodes.push("linear_regressor_non_finite_parameter_or_reference_score");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = linearBaseRow({
    scope, nodeIndex, importedOpset, opName: "LinearRegressor", contractKind: "linear_regressor",
    status, common, outputName: node.outputs?.[0] || "", outputDtype: "FLOAT32",
    outputShape, outputShapeDeclared, exactOutputElements, failures, reasons,
  });
  Object.assign(row, {
    output_names: [...(node.outputs || [])],
    canonical_output_types: [canonicalOnnxTypeProto(outputType)],
    canonical_output_shapes: [outputShape],
    linear_onnx_contract_status: targetCount == null ? "fail" : "pass",
    linear_onnx_contract_reason: targetCount == null ? "linear_regressor_targets_must_be_positive" : "linear_regressor_schema_contract_resolved",
    linear_pinned_ort_contract_status: runtimeContract.status,
    linear_pinned_ort_contract_reason: runtimeContract.reason,
    linear_class_or_target_count: targetCount,
    linear_expected_coefficient_count: runtimeContract.expectedCoefficientCount,
    linear_coefficient_count: coefficientValues.length,
    linear_used_coefficient_count: runtimeContract.usedCoefficientCount,
    linear_unused_coefficient_count: runtimeContract.unusedCoefficientCount,
    linear_intercept_count: interceptValues.length,
    linear_intercepts_used: runtimeContract.interceptsUsed,
    linear_ignored_intercept_count: runtimeContract.interceptsIgnoredCount,
    linear_targets_value: targets?.toString() ?? "unresolved",
    linear_targets_source: targetsAttribute ? "explicit_attribute" : "onnx_schema_default_1_materialized_by_ort_schema_resolution",
    linear_post_transform: postTransform || "unresolved",
    linear_post_transform_source: postTransformAttribute ? "explicit_attribute" : "onnx_schema_default_NONE",
    linear_non_finite_parameter_count: nonFiniteParameterCount,
    linear_reference_assessment_status: reference.status,
    linear_reference_input_value_count: reference.inputValueCount,
    linear_reference_raw_score_count: reference.rawScoreCount,
    linear_reference_non_finite_raw_score_count: reference.nonFiniteRawScoreCount,
    linear_reference_decision_boundary_count: null,
    linear_reference_raw_score_preview: reference.rawScorePreview,
    linear_reference_output_preview: reference.outputPreview,
    linear_reference_label_preview: [],
    linear_reference_boundary: "Deterministic scalar FLOAT32 reference only; pinned ORT uses MLAS GEMM and selected microkernel accumulation order is not observed, so values are not promoted as runtime-bit-exact tensor evidence.",
    risk_codes: riskCodes,
  });
  const outputs = status !== "fail" && outputShapeDeclared && node.outputs?.[0] ? [[node.outputs[0], outputPatch]] : [];
  return { status, reason: row.reason_codes[0] || "", result: { outputs }, row };
}

function linearInputContract(node, tensorMap) {
  const failures = [];
  const reasons = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("linear_model_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`linear_model_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype === "UNKNOWN") reasons.push("linear_model_input_dtype_unresolved");
  else if (!ALLOWED_DTYPES.has(inputDtype)) failures.push(`linear_model_input_dtype_not_supported:${inputDtype}`);
  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  const rank = shapeDeclared ? inputShape.length : null;
  const rankValid = rank === 1 || rank === 2;
  if (!shapeDeclared) reasons.push("linear_model_input_shape_unresolved");
  else if (!rankValid) failures.push(`linear_model_input_rank_not_one_or_two:${rank}`);
  const batchCount = rank === 1 ? 1 : rank === 2 && knownDimension(inputShape[0]) ? inputShape[0] : null;
  const featureCount = rank === 1 && knownDimension(inputShape[0]) ? inputShape[0]
    : rank === 2 && knownDimension(inputShape[1]) ? inputShape[1] : null;
  if (rankValid && (batchCount == null || featureCount == null)) reasons.push("linear_model_dynamic_batch_or_feature_dimension");
  return { input, inputType, inputDtype, inputShape, shapeDeclared, rank, rankValid, batchCount, featureCount, failures, reasons };
}

function classifierLabelContract(intLabels, stringLabels, classCount) {
  const intActive = intLabels.length > 0;
  const stringActive = stringLabels.length > 0;
  if (intActive === stringActive) {
    return { status: "fail", reason: "linear_classifier_requires_exactly_one_nonempty_classlabel_list", labelKind: "invalid", labelCount: 0, labels: [] };
  }
  const labels = stringActive ? stringLabels : intLabels;
  const labelKind = stringActive ? "string" : "int64";
  const expected = classCount === 1 ? 2 : classCount;
  if (classCount <= 0) return { status: "fail", reason: "linear_classifier_pinned_ort_requires_nonempty_intercepts", labelKind, labelCount: labels.length, labels };
  if (labels.length !== expected) {
    return { status: "fail", reason: `linear_classifier_label_count_mismatch:${labels.length}:${expected}`, labelKind, labelCount: labels.length, labels };
  }
  return { status: "pass", reason: "linear_classifier_label_contract_resolved", labelKind, labelCount: labels.length, labels };
}

function classifierRuntimeContract(common, coefficients, intercepts) {
  if (intercepts.length === 0) return runtimeResult("fail", "linear_classifier_pinned_ort_requires_nonempty_intercepts");
  if (coefficients.length === 0) return runtimeResult("fail", "linear_classifier_pinned_ort_requires_nonempty_coefficients");
  if (!common.rankValid) return runtimeResult(common.shapeDeclared ? "fail" : "partial", "linear_classifier_input_rank_or_shape_unresolved");
  if (common.featureCount == null) return runtimeResult("partial", "linear_classifier_feature_count_unresolved");
  const expected = safeProduct(intercepts.length, common.featureCount);
  if (expected == null) return runtimeResult("partial", "linear_classifier_expected_coefficient_count_overflow");
  if (coefficients.length < expected) return runtimeResult("fail", `linear_classifier_coefficients_undersized:${coefficients.length}:${expected}`, expected, coefficients.length, 0);
  return runtimeResult("pass", coefficients.length > expected ? "linear_classifier_extra_coefficients_ignored" : "linear_classifier_runtime_contract_resolved", expected, expected, coefficients.length - expected);
}

function regressorRuntimeContract(common, coefficients, intercepts, targetCount) {
  if (targetCount == null) return { ...runtimeResult("fail", "linear_regressor_targets_outside_pinned_ort_range"), interceptsUsed: false, interceptsIgnoredCount: intercepts.length };
  if (coefficients.length === 0) return { ...runtimeResult("fail", "linear_regressor_pinned_ort_requires_nonempty_coefficients"), interceptsUsed: intercepts.length === targetCount, interceptsIgnoredCount: intercepts.length === targetCount ? 0 : intercepts.length };
  if (common.inputDtype !== "UNKNOWN" && common.inputDtype !== "FLOAT32") {
    return { ...runtimeResult("fail", `linear_regressor_pinned_ort_cpu_dtype_unsupported:${common.inputDtype}`), interceptsUsed: intercepts.length === targetCount, interceptsIgnoredCount: intercepts.length === targetCount ? 0 : intercepts.length };
  }
  if (!common.rankValid) return { ...runtimeResult(common.shapeDeclared ? "fail" : "partial", "linear_regressor_input_rank_or_shape_unresolved"), interceptsUsed: intercepts.length === targetCount, interceptsIgnoredCount: intercepts.length === targetCount ? 0 : intercepts.length };
  if (common.featureCount == null) return { ...runtimeResult("partial", "linear_regressor_feature_count_unresolved"), interceptsUsed: intercepts.length === targetCount, interceptsIgnoredCount: intercepts.length === targetCount ? 0 : intercepts.length };
  const expected = safeProduct(targetCount, common.featureCount);
  const interceptsUsed = intercepts.length === targetCount;
  const interceptsIgnoredCount = interceptsUsed ? 0 : intercepts.length;
  if (expected == null) return { ...runtimeResult("partial", "linear_regressor_expected_coefficient_count_overflow"), interceptsUsed, interceptsIgnoredCount };
  if (coefficients.length < expected) return { ...runtimeResult("fail", `linear_regressor_coefficients_undersized:${coefficients.length}:${expected}`, expected, coefficients.length, 0), interceptsUsed, interceptsIgnoredCount };
  return {
    ...runtimeResult("pass", coefficients.length > expected ? "linear_regressor_extra_coefficients_ignored" : "linear_regressor_runtime_contract_resolved", expected, expected, coefficients.length - expected),
    interceptsUsed, interceptsIgnoredCount,
  };
}

function runtimeResult(status, reason, expectedCoefficientCount = null, usedCoefficientCount = 0, unusedCoefficientCount = 0) {
  return { status, reason, expectedCoefficientCount, usedCoefficientCount, unusedCoefficientCount };
}

function classifierOutputShapes(common, scoreClassCount) {
  const shapeDeclared = common.rankValid && common.batchCount != null && Number.isSafeInteger(scoreClassCount) && scoreClassCount > 0;
  return { shapeDeclared, labelShape: shapeDeclared ? [common.batchCount] : [], scoreShape: shapeDeclared ? [common.batchCount, scoreClassCount] : [] };
}

function evaluateClassifierReference(args) {
  const source = staticNumericInput(args.input, args.dtype);
  if (!source || args.runtimeContract.status !== "pass" || args.batchCount == null || args.featureCount == null) return unresolvedReference(args.input);
  const expectedInputs = safeProduct(args.batchCount, args.featureCount);
  if (expectedInputs == null || source.length !== expectedInputs || expectedInputs > MAX_REFERENCE_VALUES) return unresolvedReference(args.input, "not_assessed_static_input_cardinality_or_limit");
  const raw = evaluateLinearScores(source, args.dtype, args.batchCount, args.featureCount, args.classCount, args.coefficients, args.intercepts);
  const labels = [];
  let decisionBoundaryCount = 0;
  for (let batch = 0; batch < args.batchCount; batch += 1) {
    const scores = raw.slice(batch * args.classCount, (batch + 1) * args.classCount);
    let selected = 0;
    if (args.classCount === 1) {
      selected = scores[0] > 0 ? 1 : 0;
      if (scores[0] === 0 || !Number.isFinite(scores[0])) decisionBoundaryCount += 1;
    } else {
      for (let index = 1; index < scores.length; index += 1) if (scores[index] > scores[selected]) selected = index;
      const sorted = [...scores].sort((left, right) => right - left);
      if (sorted.length > 1 && (sorted[0] === sorted[1] || !Number.isFinite(sorted[0]))) decisionBoundaryCount += 1;
    }
    labels.push(args.labels[selected]);
  }
  const transformed = classifierPostTransform(raw, args.classCount, args.postTransform, args.expandedBinary);
  return referenceResult(source.length, raw, transformed, labels, decisionBoundaryCount);
}

function evaluateRegressorReference(args) {
  const source = staticNumericInput(args.input, args.dtype);
  if (!source || args.runtimeContract.status !== "pass" || args.batchCount == null || args.featureCount == null || args.targetCount == null) return unresolvedReference(args.input);
  const expectedInputs = safeProduct(args.batchCount, args.featureCount);
  if (expectedInputs == null || source.length !== expectedInputs || expectedInputs > MAX_REFERENCE_VALUES) return unresolvedReference(args.input, "not_assessed_static_input_cardinality_or_limit");
  const intercepts = args.intercepts.length ? args.intercepts : new Array(args.targetCount).fill(0);
  const raw = evaluateLinearScores(source, args.dtype, args.batchCount, args.featureCount, args.targetCount, args.coefficients, intercepts);
  const transformed = regressorPostTransform(raw, args.targetCount, args.postTransform);
  return referenceResult(source.length, raw, transformed, [], 0);
}

function evaluateLinearScores(source, dtype, batches, features, targets, coefficients, intercepts) {
  const values = new Array(batches * targets);
  for (let batch = 0; batch < batches; batch += 1) {
    for (let target = 0; target < targets; target += 1) {
      let sum = Math.fround(intercepts[target] || 0);
      for (let feature = 0; feature < features; feature += 1) {
        const input = toFloat32(source[batch * features + feature], dtype);
        const product = Math.fround(input * coefficients[target * features + feature]);
        sum = Math.fround(sum + product);
      }
      values[batch * targets + target] = sum;
    }
  }
  return values;
}

function classifierPostTransform(raw, classCount, postTransform, expandedBinary) {
  if (expandedBinary) {
    if (postTransform === "PROBIT") return null;
    return raw.flatMap((score) => [Math.fround(1 - score), score]);
  }
  if (classCount === 1) return postTransform === "PROBIT" ? null : [...raw];
  return transformBatches(raw, classCount, postTransform);
}

function regressorPostTransform(raw, targetCount, postTransform) {
  if (postTransform === "PROBIT") return null;
  if (targetCount === 1) return [...raw];
  return transformBatches(raw, targetCount, postTransform);
}

function transformBatches(raw, width, postTransform) {
  if (postTransform === "NONE") return [...raw];
  const output = [];
  for (let offset = 0; offset < raw.length; offset += width) {
    const row = raw.slice(offset, offset + width);
    if (postTransform === "LOGISTIC") output.push(...row.map(logisticReference));
    else if (["SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform)) output.push(...softmaxReference(row));
    else return null;
  }
  return output;
}

function logisticReference(value) {
  const magnitude = Math.abs(value);
  const positive = Math.fround(1 / Math.fround(1 + Math.exp(-magnitude)));
  return Math.fround(value < 0 ? 1 - positive : positive);
}

function softmaxReference(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.fround(Math.exp(Math.fround(value - maximum))));
  const sum = exponentials.reduce((total, value) => Math.fround(total + value), 0);
  return exponentials.map((value) => Math.fround(value / sum));
}

function referenceResult(inputValueCount, raw, output, labels, decisionBoundaryCount) {
  const nonFiniteRawScoreCount = raw.filter((value) => !Number.isFinite(value)).length;
  return {
    status: output ? "assessed_scalar_float32_reference_not_runtime_bit_exact" : "assessed_raw_scores_post_transform_not_materialized",
    inputValueCount,
    rawScoreCount: raw.length,
    nonFiniteRawScoreCount,
    decisionBoundaryCount,
    rawScorePreview: raw.slice(0, 12).map(exactValueText),
    outputPreview: output ? output.slice(0, 12).map(exactValueText) : [],
    labelPreview: labels.slice(0, 12).map(exactValueText),
  };
}

function unresolvedReference(input, status = null) {
  return {
    status: status || (input?.role === "initializer" ? "not_assessed_initializer_values" : "not_assessed_runtime_values"),
    inputValueCount: null, rawScoreCount: null, nonFiniteRawScoreCount: null, decisionBoundaryCount: null,
    rawScorePreview: [], outputPreview: [], labelPreview: [],
  };
}

function linearBaseRow({ scope, nodeIndex, importedOpset, opName, contractKind, status, common, outputName, outputDtype, outputShape, outputShapeDeclared, exactOutputElements, failures, reasons }) {
  return {
    scope, node_index: nodeIndex, op_name: opName, contract_kind: contractKind,
    imported_opset: importedOpset, status,
    input_name: common.input?.name || "", output_name: outputName,
    input_dtype: common.inputDtype, input_kind: common.inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: common.rank, input_shape: common.inputShape,
    exact_batch_count: common.batchCount, exact_feature_count: common.featureCount,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: canonicalOnnxTypeProto(makeOnnxTensorType(outputDtype, outputShape, outputShapeDeclared)),
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: outputShapeDeclared ? outputShape.length : null, exact_output_shape: outputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_linear_model_rank_and_pinned_ort_attribute_contract",
    runtime_reference_status: "pinned_ort_cpu_linear_model_kernel_and_ml_common",
    attribute_mode: "linear_coefficients_intercepts_labels_targets_and_post_transform",
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
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) return input.staticValues;
  return null;
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
  const rounded = Number(significand) * (2 ** (exponent - 23));
  return Math.fround(negative ? -rounded : rounded);
}

function floatListAttribute(attribute) {
  if (attribute?.type !== 6 || !Array.isArray(attribute.floats)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 6) return null;
  return attribute.floats.map((value) => Math.fround(value));
}

function intListAttribute(attribute) {
  if (attribute?.type !== 7 || !Array.isArray(attribute.ints)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 7) return null;
  const exact = Array.isArray(attribute.intExactDecimals) && attribute.intExactDecimals.length === attribute.ints.length
    ? attribute.intExactDecimals : attribute.ints;
  try { return exact.map((value) => BigInt(value)); } catch { return null; }
}

function stringListAttribute(attribute) {
  if (attribute?.type !== 8 || !Array.isArray(attribute.strings)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 8) return null;
  return [...attribute.strings];
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

function duplicateValueCount(values) {
  return values.length - new Set(values.map((value) => typeof value === "bigint" ? `i:${value}` : `s:${value}`)).size;
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
    product = safeProduct(product, dimension);
    if (product == null) return null;
  }
  return product;
}

function safeProduct(left, right) {
  const value = left * right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}
