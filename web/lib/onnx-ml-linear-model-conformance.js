import { staticValuesWithSignedZeros } from "./onnx-static-value-evidence.js";

const POST_TRANSFORMS = new Set(["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"]);

export function validateLinearModelRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || !["LinearClassifier", "LinearRegressor"].includes(row.op_name)) return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === row.op_name && candidate.domain === "ai.onnx.ml");
  if (!op) return false;
  return row.op_name === "LinearClassifier"
    ? validateClassifier(row, input, op)
    : validateRegressor(row, input, op);
}

function validateClassifier(row, input, op) {
  const coefficients = floatList(op, "coefficients");
  const intercepts = floatList(op, "intercepts", []);
  const intLabels = intList(op, "classlabels_ints", []);
  const stringLabels = stringList(op, "classlabels_strings", []);
  const multiClass = intScalar(op, "multi_class", 0n);
  const postTransform = stringScalar(op, "post_transform", "NONE");
  if (![coefficients, intercepts, intLabels, stringLabels, multiClass, postTransform].every((value) => value.ok)) {
    return row.status === "fail" && row.reason_codes.length > 0;
  }
  const shape = inputShape(row);
  const rankValid = shape.rank === 1 || shape.rank === 2;
  const batch = shape.rank === 1 ? 1 : shape.rank === 2 && knownDimension(shape.values[0]) ? shape.values[0] : null;
  const features = shape.rank === 1 && knownDimension(shape.values[0]) ? shape.values[0]
    : shape.rank === 2 && knownDimension(shape.values[1]) ? shape.values[1] : null;
  const classCount = intercepts.value.length;
  const intActive = intLabels.value.length > 0;
  const stringActive = stringLabels.value.length > 0;
  const labels = stringActive ? stringLabels.value : intLabels.value;
  const labelKind = stringActive ? "string" : intActive ? "int64" : "invalid";
  const expectedLabelCount = classCount === 1 ? 2 : classCount;
  const onnxPass = intActive !== stringActive && classCount > 0 && labels.length === expectedLabelCount;
  const calculatedExpectedCoefficients = features == null ? null : safeProduct(classCount, features);
  const runtimeContract = classifierRuntimeContract({ classCount, coefficientCount: coefficients.value.length,
    rank: shape.rank, rankValid, features, expectedCoefficients: calculatedExpectedCoefficients });
  const runtimePass = runtimeContract.status === "pass";
  const expanded = classCount === 1 && labels.length === 2;
  const scoreClasses = expanded ? 2 : classCount;
  const labelShape = batch == null || scoreClasses <= 0 ? [] : [batch];
  const scoreShape = batch == null || scoreClasses <= 0 ? [] : [batch, scoreClasses];
  const expectedCoefficients = runtimeContract.expectedCoefficientCount;
  const used = runtimeContract.usedCoefficientCount;
  const unused = runtimeContract.unusedCoefficientCount;
  const duplicates = duplicateCount(labels);
  const reference = runtimePass && onnxPass
    ? reconstructClassifierReference(input, row.input_dtype, batch, features, classCount, coefficients.value, intercepts.value, labels, postTransform.value, expanded)
    : unresolvedReference(input);
  const risks = [];
  if (!onnxPass) risks.push("linear_classifier_onnx_label_contract_invalid");
  if (!runtimePass) risks.push("linear_classifier_pinned_ort_runtime_contract_invalid");
  if (unused > 0) risks.push("linear_classifier_unused_coefficients_ignored");
  if (duplicates > 0) risks.push("linear_classifier_duplicate_labels_ambiguous_output_semantics");
  if (multiClass.value !== 0n) risks.push("linear_classifier_multi_class_nonzero_ignored_by_pinned_ort");
  if (classCount === 1 && !expanded && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform.value)) risks.push("linear_classifier_single_score_post_transform_noop");
  if (expanded && postTransform.value === "PROBIT") risks.push("linear_classifier_binary_probit_second_score_unwritten");
  if (expanded && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform.value)) risks.push("linear_classifier_binary_post_transform_ignored_for_complement_expansion");
  const nonFiniteParameters = [...coefficients.value, ...intercepts.value].filter((value) => !Number.isFinite(value)).length;
  if (nonFiniteParameters > 0 || reference.nonFiniteRawScoreCount > 0) risks.push("linear_classifier_non_finite_parameter_or_reference_score");
  if (reference.decisionBoundaryCount > 0) risks.push("linear_classifier_reference_decision_boundary");
  return row.contract_kind === "linear_classifier"
    && row.input_kind === "tensor"
    && row.linear_onnx_contract_status === (onnxPass ? "pass" : "fail")
    && row.linear_pinned_ort_contract_status === runtimeContract.status
    && row.linear_pinned_ort_contract_reason === runtimeContract.reason
    && row.linear_class_or_target_count === classCount
    && row.linear_expected_coefficient_count === expectedCoefficients
    && row.linear_coefficient_count === coefficients.value.length
    && row.linear_used_coefficient_count === used
    && row.linear_unused_coefficient_count === unused
    && row.linear_intercept_count === intercepts.value.length
    && row.linear_label_kind === labelKind
    && row.linear_label_count === labels.length
    && JSON.stringify(row.linear_label_values) === JSON.stringify(labels.map(text))
    && row.linear_duplicate_label_count === duplicates
    && row.linear_multi_class_value === multiClass.value.toString()
    && row.linear_multi_class_used_by_pinned_ort === false
    && row.linear_post_transform === postTransform.value
    && row.classifier_binary_score_expansion === expanded
    && row.classifier_score_class_count === scoreClasses
    && JSON.stringify(row.classifier_label_output_shape) === JSON.stringify(labelShape)
    && JSON.stringify(row.classifier_score_output_shape) === JSON.stringify(scoreShape)
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(scoreShape)
    && row.exact_dense_output_element_count === (scoreShape.length ? safeProduct(batch, scoreClasses) : null)
    && row.linear_non_finite_parameter_count === nonFiniteParameters
    && referenceMatches(row, reference)
    && sameSet(row.risk_codes, risks);
}

function validateRegressor(row, input, op) {
  const coefficients = floatList(op, "coefficients", []);
  const intercepts = floatList(op, "intercepts", []);
  const targets = intScalar(op, "targets", 1n);
  const postTransform = stringScalar(op, "post_transform", "NONE");
  if (![coefficients, intercepts, targets, postTransform].every((value) => value.ok)) {
    return row.status === "fail" && row.reason_codes.length > 0;
  }
  const shape = inputShape(row);
  const rankValid = shape.rank === 1 || shape.rank === 2;
  const batch = shape.rank === 1 ? 1 : shape.rank === 2 && knownDimension(shape.values[0]) ? shape.values[0] : null;
  const features = shape.rank === 1 && knownDimension(shape.values[0]) ? shape.values[0]
    : shape.rank === 2 && knownDimension(shape.values[1]) ? shape.values[1] : null;
  const targetCount = targets.value > 0n && targets.value <= 2_147_483_647n ? Number(targets.value) : null;
  const calculatedExpectedCoefficients = targetCount == null || features == null ? null : safeProduct(targetCount, features);
  const runtimeContract = regressorRuntimeContract({ targetCount,
    coefficientCount: coefficients.value.length, dtype: row.input_dtype, rank: shape.rank, rankValid,
    features, expectedCoefficients: calculatedExpectedCoefficients, interceptCount: intercepts.value.length });
  const runtimePass = runtimeContract.status === "pass";
  const expectedCoefficients = runtimeContract.expectedCoefficientCount;
  const used = runtimeContract.usedCoefficientCount;
  const unused = runtimeContract.unusedCoefficientCount;
  const interceptsUsed = runtimeContract.interceptsUsed;
  const ignoredIntercepts = runtimeContract.interceptsIgnoredCount;
  const outputShape = batch == null || targetCount == null ? [] : [batch, targetCount];
  const reference = runtimePass
    ? reconstructRegressorReference(input, row.input_dtype, batch, features, targetCount, coefficients.value, interceptsUsed ? intercepts.value : [], postTransform.value)
    : unresolvedReference(input);
  const risks = [];
  if (!runtimePass) risks.push("linear_regressor_pinned_ort_runtime_contract_invalid");
  if (["FLOAT64", "INT32", "INT64"].includes(row.input_dtype)) risks.push("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
  if (unused > 0) risks.push("linear_regressor_unused_coefficients_ignored");
  if (ignoredIntercepts > 0) risks.push("linear_regressor_mismatched_intercepts_ignored");
  if (targetCount === 1 && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(postTransform.value)) risks.push("linear_regressor_single_target_post_transform_noop");
  if (postTransform.value === "PROBIT") risks.push("linear_regressor_probit_may_emit_non_finite");
  const nonFiniteParameters = [...coefficients.value, ...intercepts.value].filter((value) => !Number.isFinite(value)).length;
  if (nonFiniteParameters > 0 || reference.nonFiniteRawScoreCount > 0) risks.push("linear_regressor_non_finite_parameter_or_reference_score");
  return row.contract_kind === "linear_regressor"
    && row.input_kind === "tensor"
    && row.linear_onnx_contract_status === (targetCount == null ? "fail" : "pass")
    && row.linear_pinned_ort_contract_status === runtimeContract.status
    && row.linear_pinned_ort_contract_reason === runtimeContract.reason
    && row.linear_class_or_target_count === targetCount
    && row.linear_expected_coefficient_count === expectedCoefficients
    && row.linear_coefficient_count === coefficients.value.length
    && row.linear_used_coefficient_count === used
    && row.linear_unused_coefficient_count === unused
    && row.linear_intercept_count === intercepts.value.length
    && row.linear_intercepts_used === interceptsUsed
    && row.linear_ignored_intercept_count === ignoredIntercepts
    && row.linear_targets_value === targets.value.toString()
    && row.linear_targets_source === (targets.present ? "explicit_attribute" : "onnx_schema_default_1_materialized_by_ort_schema_resolution")
    && row.linear_post_transform === postTransform.value
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputShape)
    && row.exact_dense_output_element_count === (outputShape.length ? safeProduct(batch, targetCount) : null)
    && row.linear_non_finite_parameter_count === nonFiniteParameters
    && referenceMatches(row, reference)
    && sameSet(row.risk_codes, risks);
}

function classifierRuntimeContract({ classCount, coefficientCount, rank, rankValid, features, expectedCoefficients }) {
  if (classCount === 0) return runtimeResult("fail", "linear_classifier_pinned_ort_requires_nonempty_intercepts");
  if (coefficientCount === 0) return runtimeResult("fail", "linear_classifier_pinned_ort_requires_nonempty_coefficients");
  if (!rankValid) return runtimeResult(rank == null ? "partial" : "fail", "linear_classifier_input_rank_or_shape_unresolved");
  if (features == null) return runtimeResult("partial", "linear_classifier_feature_count_unresolved");
  if (expectedCoefficients == null) return runtimeResult("partial", "linear_classifier_expected_coefficient_count_overflow");
  if (coefficientCount < expectedCoefficients) {
    return runtimeResult("fail", `linear_classifier_coefficients_undersized:${coefficientCount}:${expectedCoefficients}`, expectedCoefficients, coefficientCount, 0);
  }
  return runtimeResult("pass", coefficientCount > expectedCoefficients
    ? "linear_classifier_extra_coefficients_ignored" : "linear_classifier_runtime_contract_resolved",
  expectedCoefficients, expectedCoefficients, coefficientCount - expectedCoefficients);
}

function regressorRuntimeContract({ targetCount, coefficientCount, dtype, rank, rankValid, features, expectedCoefficients, interceptCount }) {
  const interceptState = (available) => ({ interceptsUsed: available && interceptCount === targetCount,
    interceptsIgnoredCount: available && interceptCount === targetCount ? 0 : interceptCount });
  if (targetCount == null) return { ...runtimeResult("fail", "linear_regressor_targets_outside_pinned_ort_range"), ...interceptState(false) };
  if (coefficientCount === 0) return { ...runtimeResult("fail", "linear_regressor_pinned_ort_requires_nonempty_coefficients"), ...interceptState(true) };
  if (dtype !== "FLOAT32") return { ...runtimeResult("fail", `linear_regressor_pinned_ort_cpu_dtype_unsupported:${dtype}`), ...interceptState(true) };
  if (!rankValid) return { ...runtimeResult(rank == null ? "partial" : "fail", "linear_regressor_input_rank_or_shape_unresolved"), ...interceptState(true) };
  if (features == null) return { ...runtimeResult("partial", "linear_regressor_feature_count_unresolved"), ...interceptState(true) };
  if (expectedCoefficients == null) return { ...runtimeResult("partial", "linear_regressor_expected_coefficient_count_overflow"), ...interceptState(true) };
  if (coefficientCount < expectedCoefficients) {
    return { ...runtimeResult("fail", `linear_regressor_coefficients_undersized:${coefficientCount}:${expectedCoefficients}`, expectedCoefficients, coefficientCount, 0), ...interceptState(true) };
  }
  return { ...runtimeResult("pass", coefficientCount > expectedCoefficients
    ? "linear_regressor_extra_coefficients_ignored" : "linear_regressor_runtime_contract_resolved",
  expectedCoefficients, expectedCoefficients, coefficientCount - expectedCoefficients), ...interceptState(true) };
}

function runtimeResult(status, reason, expectedCoefficientCount = null, usedCoefficientCount = 0, unusedCoefficientCount = 0) {
  return { status, reason, expectedCoefficientCount, usedCoefficientCount, unusedCoefficientCount };
}

function reconstructClassifierReference(input, dtype, batches, features, classes, coefficients, intercepts, labels, postTransform, expanded) {
  const source = staticInput(input, dtype);
  if (!source || batches == null || features == null || source.length !== batches * features) return unresolvedReference(input);
  const raw = scalarScores(source, dtype, batches, features, classes, coefficients, intercepts);
  const labelPreview = [];
  let decisionBoundaryCount = 0;
  for (let batch = 0; batch < batches; batch += 1) {
    const scores = raw.slice(batch * classes, (batch + 1) * classes);
    let selected = 0;
    if (classes === 1) {
      selected = scores[0] > 0 ? 1 : 0;
      if (scores[0] === 0 || !Number.isFinite(scores[0])) decisionBoundaryCount += 1;
    } else {
      for (let index = 1; index < scores.length; index += 1) if (scores[index] > scores[selected]) selected = index;
      const sorted = [...scores].sort((left, right) => right - left);
      if (sorted[0] === sorted[1] || !Number.isFinite(sorted[0])) decisionBoundaryCount += 1;
    }
    if (labelPreview.length < 12) labelPreview.push(text(labels[selected]));
  }
  const output = expanded
    ? postTransform === "PROBIT" ? null : raw.flatMap((score) => [Math.fround(1 - score), score])
    : classes === 1 ? postTransform === "PROBIT" ? null : raw
      : transformed(raw, classes, postTransform);
  return reference(source.length, raw, output, labelPreview, decisionBoundaryCount);
}

function reconstructRegressorReference(input, dtype, batches, features, targets, coefficients, intercepts, postTransform) {
  const source = staticInput(input, dtype);
  if (!source || source.length !== batches * features) return unresolvedReference(input);
  const bias = intercepts.length ? intercepts : new Array(targets).fill(0);
  const raw = scalarScores(source, dtype, batches, features, targets, coefficients, bias);
  const output = postTransform === "PROBIT" ? null : targets === 1 ? raw : transformed(raw, targets, postTransform);
  return reference(source.length, raw, output, [], null);
}

function scalarScores(source, dtype, batches, features, targets, coefficients, intercepts) {
  const output = [];
  for (let batch = 0; batch < batches; batch += 1) {
    for (let target = 0; target < targets; target += 1) {
      let sum = Math.fround(intercepts[target] || 0);
      for (let feature = 0; feature < features; feature += 1) {
        const input = dtype === "INT64" && typeof source[batch * features + feature] === "bigint"
          ? bigintFloat32(source[batch * features + feature]) : Math.fround(Number(source[batch * features + feature]));
        sum = Math.fround(sum + Math.fround(input * coefficients[target * features + feature]));
      }
      output.push(sum);
    }
  }
  return output;
}

function transformed(raw, width, mode) {
  if (mode === "NONE") return [...raw];
  if (mode === "PROBIT") return null;
  const output = [];
  for (let offset = 0; offset < raw.length; offset += width) {
    const row = raw.slice(offset, offset + width);
    if (mode === "LOGISTIC") output.push(...row.map((value) => {
      const positive = Math.fround(1 / Math.fround(1 + Math.exp(-Math.abs(value))));
      return Math.fround(value < 0 ? 1 - positive : positive);
    }));
    else {
      const maximum = Math.max(...row);
      const exps = row.map((value) => Math.fround(Math.exp(Math.fround(value - maximum))));
      const sum = exps.reduce((total, value) => Math.fround(total + value), 0);
      output.push(...exps.map((value) => Math.fround(value / sum)));
    }
  }
  return output;
}

function reference(inputCount, raw, output, labels, decisionBoundaryCount) {
  return {
    status: output ? "assessed_scalar_float32_reference_not_runtime_bit_exact" : "assessed_raw_scores_post_transform_not_materialized",
    inputCount, rawCount: raw.length, nonFiniteRawScoreCount: raw.filter((value) => !Number.isFinite(value)).length,
    decisionBoundaryCount, rawPreview: raw.slice(0, 12).map(text),
    outputPreview: output ? output.slice(0, 12).map(text) : [], labelPreview: labels,
  };
}

function unresolvedReference(input) {
  return {
    status: input?.initializer_storage_kind ? "not_assessed_initializer_values" : "not_assessed_runtime_values",
    inputCount: null, rawCount: null, nonFiniteRawScoreCount: null, decisionBoundaryCount: null,
    rawPreview: [], outputPreview: [], labelPreview: [],
  };
}

function referenceMatches(row, value) {
  return row.linear_reference_assessment_status === value.status
    && row.linear_reference_input_value_count === value.inputCount
    && row.linear_reference_raw_score_count === value.rawCount
    && row.linear_reference_non_finite_raw_score_count === value.nonFiniteRawScoreCount
    && row.linear_reference_decision_boundary_count === value.decisionBoundaryCount
    && JSON.stringify(row.linear_reference_raw_score_preview) === JSON.stringify(value.rawPreview)
    && JSON.stringify(row.linear_reference_output_preview) === JSON.stringify(value.outputPreview)
    && JSON.stringify(row.linear_reference_label_preview) === JSON.stringify(value.labelPreview);
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializer_integer_values_exact_complete === true
    && Array.isArray(input.initializer_integer_values_exact_decimals)) {
    try { return input.initializer_integer_values_exact_decimals.map((value) => BigInt(value)); } catch { return null; }
  }
  return staticValuesWithSignedZeros(input);
}

function floatList(op, name, fallback = null) {
  const value = attr(op, name);
  if (!value) return fallback == null ? { ok: false, present: false, value: [] } : { ok: true, present: false, value: fallback };
  if (value.type !== 6 || !Array.isArray(value.float_values_text)) return { ok: false, present: true, value: [] };
  return { ok: true, present: true, value: value.float_values_text.map(parseFloatText) };
}

function intList(op, name, fallback) {
  const value = attr(op, name);
  if (!value) return { ok: true, present: false, value: fallback };
  if (value.type !== 7 || !Array.isArray(value.int_values_exact_decimal)) return { ok: false, present: true, value: [] };
  try { return { ok: true, present: true, value: value.int_values_exact_decimal.map((item) => BigInt(item)) }; }
  catch { return { ok: false, present: true, value: [] }; }
}

function stringList(op, name, fallback) {
  const value = attr(op, name);
  return !value ? { ok: true, present: false, value: fallback }
    : value.type === 8 && Array.isArray(value.string_values)
      ? { ok: true, present: true, value: [...value.string_values] }
      : { ok: false, present: true, value: [] };
}

function intScalar(op, name, fallback) {
  const value = attr(op, name);
  if (!value) return { ok: true, present: false, value: fallback };
  if (value.type !== 2) return { ok: false, present: true, value: fallback };
  try { return { ok: true, present: true, value: BigInt(value.int_value_exact_decimal) }; }
  catch { return { ok: false, present: true, value: fallback }; }
}

function stringScalar(op, name, fallback) {
  const value = attr(op, name);
  if (!value) return { ok: true, present: false, value: fallback };
  return value.type === 3 && POST_TRANSFORMS.has(value.string_value)
    ? { ok: true, present: true, value: value.string_value }
    : { ok: false, present: true, value: fallback };
}

function attr(op, name) {
  return (op?.onnx_attributes || []).find((value) => value.name === name);
}

function inputShape(row) {
  return { rank: row.input_rank, values: Array.isArray(row.input_shape) ? row.input_shape : [] };
}

function duplicateCount(values) {
  return values.length - new Set(values.map((value) => typeof value === "bigint" ? `i:${value}` : `s:${value}`)).size;
}

function parseFloatText(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Math.fround(Number(value));
}

function bigintFloat32(value) {
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

function text(value) {
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

function safeProduct(left, right) {
  const value = left * right;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function sameSet(left, right) {
  return JSON.stringify([...(left || [])].sort()) === JSON.stringify([...right].sort());
}
