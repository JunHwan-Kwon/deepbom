const INPUT_DTYPES = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
const KERNEL_TYPES = new Set(["LINEAR", "POLY", "RBF", "SIGMOID"]);
const POST_TRANSFORMS = new Set(["NONE", "LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO", "PROBIT"]);

export function validateSvmRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || !["SVMClassifier", "SVMRegressor"].includes(row.op_name)) return false;
  const op = ops.find((candidate) => candidate.index === row.node_index && candidate.name === row.op_name
    && candidate.domain === "ai.onnx.ml");
  if (!op) return false;
  const input = tensors.find((tensor) => tensor.name === op.input_names?.[0]);
  const attrs = new Map((op.onnx_attributes || []).map((attribute) => [attribute.name, attribute]));
  const common = inputContract(input);
  const facts = row.op_name === "SVMClassifier"
    ? classifierFacts(attrs, common) : regressorFacts(attrs, common);
  if (!facts.valid) return row.status === "fail" && Array.isArray(row.reason_codes) && row.reason_codes.length > 0;
  const expectedStatus = facts.fail ? "fail" : facts.partial ? "partial" : "pass";
  const outputDeclared = !facts.widthMismatch && facts.outputShape.length > 0;
  const exactOutputElements = outputDeclared && facts.outputShape.every(knownDimension)
    ? shapeProduct(facts.outputShape) : null;
  const referenceAssessed = String(row.svm_reference_assessment_status || "").startsWith("assessed_");
  const expectedReferenceInputs = referenceAssessed && common.batch != null && common.features != null
    ? common.batch * common.features : null;
  const expectedRawScores = referenceAssessed
    ? row.op_name === "SVMClassifier"
      ? common.batch * (facts.mode === "svc" ? facts.pairs : facts.classCount)
      : common.batch
    : null;
  const expectedRisks = structuralRisks(row, facts, common);
  return row.status === expectedStatus
    && row.input_name === input?.name && row.input_dtype === common.dtype
    && row.input_rank === common.rank && JSON.stringify(row.input_shape) === JSON.stringify(common.shape)
    && row.exact_batch_count === common.batch && row.exact_feature_count === common.features
    && row.svm_mode === facts.mode && row.svm_kernel_type === facts.kernel
    && row.svm_onnx_contract_status === (facts.onnxFail ? "fail" : "pass")
    && row.svm_pinned_ort_contract_status === (row.op_name === "SVMRegressor" && facts.cpuGap
      ? "fail" : facts.runtimeFail ? "fail" : facts.partial ? "partial" : "pass")
    && row.svm_post_transform === facts.postTransform
    && JSON.stringify(row.svm_kernel_params) === JSON.stringify(facts.kernelParams.map(text))
    && row.svm_vector_count === facts.vectorCount
    && row.svm_pairwise_classifier_count === facts.pairs
    && row.svm_schema_score_width === facts.schemaWidth
    && row.svm_pinned_ort_score_width === facts.runtimeWidth
    && row.svm_schema_runtime_score_width_mismatch === facts.widthMismatch
    && row.svm_support_vector_value_count === facts.support.length
    && row.svm_expected_support_vector_value_count === facts.expectedSupport
    && row.svm_used_support_vector_value_count === facts.usedSupport
    && row.svm_unused_support_vector_value_count === (facts.fail ? 0 : facts.support.length - facts.usedSupport)
    && row.svm_coefficient_count === facts.coefficients.length
    && row.svm_expected_coefficient_count === facts.expectedCoefficients
    && row.svm_used_coefficient_count === facts.usedCoefficients
    && row.svm_unused_coefficient_count === (facts.fail ? 0 : facts.coefficients.length - facts.usedCoefficients)
    && row.svm_rho_count === facts.rho.length
    && row.svm_expected_rho_count === facts.expectedRho
    && row.svm_used_rho_count === facts.usedRho
    && row.svm_unused_rho_count === (facts.fail ? 0 : facts.rho.length - facts.usedRho)
    && row.svm_non_finite_parameter_count === facts.nonFiniteParameters
    && row.exact_output_rank === (outputDeclared ? facts.outputShape.length : null)
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputDeclared ? facts.outputShape : [])
    && row.exact_dense_output_element_count === exactOutputElements
    && (!referenceAssessed || row.svm_reference_input_value_count === expectedReferenceInputs
      && row.svm_reference_raw_score_count === expectedRawScores)
    && Number.isSafeInteger(row.svm_reference_non_finite_score_count ?? 0)
    && Number.isSafeInteger(row.svm_reference_decision_boundary_count ?? 0)
    && Array.isArray(row.svm_reference_raw_score_preview)
    && Array.isArray(row.svm_reference_output_score_preview)
    && Array.isArray(row.svm_reference_label_preview)
    && expectedRisks.every((risk) => row.risk_codes.includes(risk))
    && row.risk_codes.every((risk) => expectedRisks.includes(risk));
}

function classifierFacts(attrs, common) {
  const base = sharedAttributes(attrs);
  const intLabelsPresent = attrs.has("classlabels_ints");
  const stringLabelsPresent = attrs.has("classlabels_strings");
  const intLabels = ints(attrs.get("classlabels_ints"));
  const stringLabels = strings(attrs.get("classlabels_strings"));
  const vectors = ints(attrs.get("vectors_per_class"));
  const probabilityA = floats(attrs.get("prob_a"));
  const probabilityB = floats(attrs.get("prob_b"));
  if (intLabels == null || stringLabels == null || vectors == null || probabilityA == null || probabilityB == null) return invalid();
  const labels = stringLabelsPresent ? stringLabels : intLabels;
  const classCount = labels.length;
  const vectorNumbers = vectors.map(safeInteger);
  const vectorCount = vectorNumbers.every((value) => value != null && value >= 0)
    ? vectorNumbers.reduce((sum, value) => sum + value, 0) : null;
  const pairs = classCount >= 0 ? classCount * (classCount - 1) / 2 : null;
  const mode = vectorCount > 0 ? "svc" : "linear";
  const onnxFail = !common.valid || !base.valid || Number(intLabelsPresent) + Number(stringLabelsPresent) !== 1
    || classCount < 2;
  let fail = onnxFail
    || classCount < 2 || classCount >= 65_536 || vectorCount == null
    || !base.kernelParamsPresent || !base.coefficients.length || !base.rho.length
    || probabilityA.length !== probabilityB.length;
  let features = mode === "linear" && classCount > 0 ? Math.floor(base.coefficients.length / classCount) : null;
  let expectedSupport = mode === "linear" ? 0 : null;
  let expectedCoefficients = mode === "linear" && features != null ? classCount * features : null;
  let expectedRho = mode === "svc" ? pairs : 1;
  if (mode === "svc") {
    const supportLayoutValid = vectorCount > 0 && base.support.length % vectorCount === 0;
    features = supportLayoutValid ? base.support.length / vectorCount : null;
    expectedSupport = features == null ? null : vectorCount * features;
    expectedCoefficients = (classCount - 1) * vectorCount;
    fail = fail || vectors.length !== classCount || !supportLayoutValid || features < 1
      || base.coefficients.length < expectedCoefficients || base.rho.length < expectedRho
      || probabilityA.length > 0 && probabilityA.length < pairs;
  } else {
    fail = fail || features < 1;
  }
  fail = fail || common.features != null && features != null && common.features !== features;
  const haveProbability = probabilityA.length > 0;
  const runtimeWidth = fail ? null : mode === "svc" && !haveProbability ? (classCount > 2 ? pairs : 2) : classCount;
  const widthMismatch = !fail && mode === "svc" && !haveProbability && classCount > 3 && runtimeWidth !== classCount;
  const runtimeFail = fail;
  const partial = !fail && !widthMismatch && common.features == null;
  return {
    ...base, valid: true, fail: fail || widthMismatch, runtimeFail, onnxFail, partial, mode, classCount, vectorCount: vectorCount ?? 0,
    pairs: pairs ?? 0, schemaWidth: classCount, runtimeWidth, widthMismatch,
    expectedSupport, usedSupport: fail ? 0 : Math.min(base.support.length, expectedSupport),
    expectedCoefficients, usedCoefficients: fail ? 0 : Math.min(base.coefficients.length, expectedCoefficients),
    expectedRho, usedRho: fail ? 0 : Math.min(base.rho.length, expectedRho),
    outputShape: common.rank == null ? [] : [common.batch, classCount],
    labels, duplicateLabels: duplicateCount(labels), haveProbability,
    probabilityA, probabilityB,
    expectedProbability: mode === "svc" && haveProbability ? pairs : 0,
  };
}

function regressorFacts(attrs, common) {
  const base = sharedAttributes(attrs);
  const nSupports = integer(attrs.get("n_supports"), 0n);
  const oneClass = integer(attrs.get("one_class"), 0n);
  if (nSupports == null || oneClass == null) return invalid();
  const vectorCount = safeInteger(nSupports);
  const mode = vectorCount > 0 ? "svc" : "linear";
  const onnxFail = !common.valid || !base.valid || vectorCount == null || vectorCount < 0 || oneClass == null;
  let fail = onnxFail
    || !base.kernelParamsPresent || !base.coefficients.length || !base.rho.length;
  let features = mode === "linear" ? base.coefficients.length : null;
  let expectedSupport = mode === "linear" ? 0 : null;
  let expectedCoefficients = mode === "linear" ? features : vectorCount;
  if (mode === "svc") {
    const supportLayoutValid = vectorCount > 0 && base.support.length > 0 && base.support.length % vectorCount === 0;
    features = supportLayoutValid ? base.support.length / vectorCount : null;
    expectedSupport = features == null ? null : vectorCount * features;
    fail = fail || base.coefficients.length < vectorCount || !supportLayoutValid;
  }
  fail = fail || common.features != null && features != null && common.features !== features;
  const cpuGap = INPUT_DTYPES.has(common.dtype) && common.dtype !== "FLOAT32";
  const runtimeFail = fail;
  const partial = !fail && (common.features == null || cpuGap);
  return {
    ...base, valid: true, fail, runtimeFail, onnxFail, partial, mode, classCount: 0, vectorCount: vectorCount ?? 0, pairs: 0,
    schemaWidth: 1, runtimeWidth: 1, widthMismatch: false,
    expectedSupport, usedSupport: fail ? 0 : Math.min(base.support.length, expectedSupport),
    expectedCoefficients, usedCoefficients: fail ? 0 : Math.min(base.coefficients.length, expectedCoefficients),
    expectedRho: 1, usedRho: fail ? 0 : 1,
    outputShape: common.rank == null ? [] : [common.batch, 1],
    oneClass, cpuGap,
  };
}

function sharedAttributes(attrs) {
  const kernel = string(attrs.get("kernel_type"), "LINEAR");
  const postTransform = string(attrs.get("post_transform"), "NONE");
  const kernelParams = floats(attrs.get("kernel_params"));
  const support = floats(attrs.get("support_vectors"));
  const coefficients = floats(attrs.get("coefficients"));
  const rho = floats(attrs.get("rho"));
  const valid = KERNEL_TYPES.has(kernel) && POST_TRANSFORMS.has(postTransform)
    && kernelParams != null && [0, 3].includes(kernelParams.length)
    && support != null && coefficients != null && rho != null;
  const values = [...(kernelParams || []), ...(support || []), ...(coefficients || []), ...(rho || [])];
  return {
    valid, kernel, postTransform, kernelParams: kernelParams || [], kernelParamsPresent: attrs.has("kernel_params"),
    support: support || [], coefficients: coefficients || [], rho: rho || [],
    nonFiniteParameters: values.filter((value) => !Number.isFinite(value)).length,
  };
}

function structuralRisks(row, facts, common) {
  const risks = [];
  if (row.op_name === "SVMClassifier") {
    if (facts.fail && !facts.widthMismatch) risks.push("svm_classifier_pinned_ort_runtime_contract_invalid");
    if (facts.widthMismatch) risks.push("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch");
    if (facts.duplicateLabels > 0) risks.push("svm_classifier_duplicate_labels_ambiguous_output_semantics");
    if (!facts.fail && (facts.coefficients.length > facts.usedCoefficients || facts.support.length > facts.usedSupport
      || facts.rho.length > facts.usedRho
      || facts.probabilityA.length + facts.probabilityB.length > 2 * facts.expectedProbability)) {
      risks.push("svm_classifier_serialized_parameters_ignored_by_pinned_ort");
    }
    if (facts.mode === "linear" && facts.kernel !== "LINEAR") risks.push("svm_classifier_linear_mode_forces_linear_kernel");
    if (facts.mode === "svc" && facts.haveProbability && facts.postTransform !== "NONE") risks.push("svm_classifier_probability_scores_receive_additional_post_transform");
    if (facts.mode === "svc" && !facts.haveProbability && facts.classCount === 2 && facts.postTransform === "PROBIT") risks.push("svm_classifier_binary_probit_second_score_unwritten");
    if (facts.mode === "svc" && !facts.haveProbability && facts.classCount === 2
      && ["LOGISTIC", "SOFTMAX", "SOFTMAX_ZERO"].includes(facts.postTransform)) risks.push("svm_classifier_binary_post_transform_uses_complement_expansion");
  } else {
    if (facts.fail) risks.push("svm_regressor_pinned_ort_runtime_contract_invalid");
    if (facts.cpuGap) risks.push("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel");
    if (facts.postTransform !== "NONE") risks.push("svm_regressor_post_transform_ignored_by_pinned_ort");
    if (![0n, 1n].includes(facts.oneClass)) risks.push("svm_regressor_noncanonical_one_class_flag");
    if (!facts.fail && (facts.coefficients.length > facts.usedCoefficients || facts.support.length > facts.usedSupport || facts.rho.length > facts.usedRho)) risks.push("svm_regressor_serialized_parameters_ignored_by_pinned_ort");
    if (facts.mode === "linear" && facts.kernel !== "LINEAR") risks.push("svm_regressor_linear_mode_forces_linear_kernel");
  }
  if (facts.nonFiniteParameters > 0 || Number(row.svm_reference_non_finite_score_count || 0) > 0) risks.push(`${row.op_name === "SVMClassifier" ? "svm_classifier" : "svm_regressor"}_non_finite_parameter_or_reference_score`);
  if (Number(row.svm_reference_decision_boundary_count || 0) > 0) risks.push(`${row.op_name === "SVMClassifier" ? "svm_classifier" : "svm_regressor"}_reference_decision_boundary`);
  return risks;
}

function inputContract(input) {
  const shape = input?.shape_declared === true ? [...(input.shape || [])] : [];
  const rank = input?.shape_declared === true ? shape.length : null;
  const dtype = input?.dtype || "UNKNOWN";
  return {
    valid: Boolean(input) && input.value_kind === "tensor" && INPUT_DTYPES.has(dtype) && [1, 2].includes(rank),
    dtype, shape, rank, batch: rank === 1 ? 1 : rank === 2 ? shape[0] : null,
    features: rank === 1 ? shape[0] : rank === 2 ? shape[1] : null,
  };
}

function floats(attribute) {
  if (!attribute) return [];
  if (attribute.type !== 6 || !Array.isArray(attribute.float_values_text)) return null;
  return attribute.float_values_text.map(number);
}

function ints(attribute) {
  if (!attribute) return [];
  if (attribute.type !== 7 || !Array.isArray(attribute.int_values_exact_decimal)) return null;
  try { return attribute.int_values_exact_decimal.map((value) => BigInt(value)); } catch { return null; }
}

function strings(attribute) {
  if (!attribute) return [];
  return attribute.type === 8 && Array.isArray(attribute.string_values) ? [...attribute.string_values] : null;
}

function integer(attribute, fallback) {
  if (!attribute) return fallback;
  try { return attribute.type === 2 ? BigInt(attribute.int_value_exact_decimal) : null; } catch { return null; }
}

function string(attribute, fallback) {
  return attribute ? attribute.type === 3 ? attribute.string_value : null : fallback;
}

function number(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

function safeInteger(value) {
  const numeric = Number(value);
  return Number.isSafeInteger(numeric) ? numeric : null;
}

function shapeProduct(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function knownDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function duplicateCount(values) {
  return values.length - new Set(values.map((value) => typeof value === "bigint" ? `i:${value}` : `s:${value}`)).size;
}

function text(value) {
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function invalid() {
  return { valid: false };
}
