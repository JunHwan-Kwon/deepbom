import { numericArraysExactlyEqual, staticValuesWithSignedZeros } from "./onnx-static-value-evidence.js";

const VERSION_DTYPES = Object.freeze({
  1: new Set(["STRING", "INT64"]),
  2: new Set(["STRING", "INT64", "FLOAT32"]),
  4: new Set(["STRING", "INT64", "FLOAT32", "INT32", "INT16", "FLOAT64"]),
});
const V4_ORT_PAIRS = new Set([
  "INT64:INT64", "INT64:STRING", "INT64:FLOAT32", "INT64:FLOAT64",
  "FLOAT32:FLOAT32", "FLOAT32:STRING", "FLOAT32:INT64",
  "STRING:INT64", "STRING:FLOAT32", "STRING:STRING", "STRING:INT16", "STRING:FLOAT64",
  "FLOAT64:FLOAT64", "FLOAT64:STRING", "FLOAT64:INT64",
]);
const LIMIT = 1_000_000;

export function validateLabelEncoderRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || row.op_name !== "LabelEncoder") return false;
  const input = tensors.find((tensor) => tensor.name === row.input_name);
  const output = tensors.find((tensor) => tensor.name === row.output_name);
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === "LabelEncoder" && candidate.domain === "ai.onnx.ml");
  if (!op) return false;
  const version = resolveVersion(row.imported_opset);
  const parameters = version === 1 ? versionOne(op, row.input_dtype) : versioned(op, row.input_dtype, version);
  if (!parameters.ok) return row.status === "fail" && row.reason_codes.length > 0;
  const shape = Array.isArray(row.input_shape) ? row.input_shape : [];
  const shapeDeclared = Number.isSafeInteger(row.input_rank) && row.input_rank >= 0;
  const elements = shapeDeclared && shape.every(knownDimension) ? shapeProduct(shape) : null;
  const runtime = runtimeContract(version, row.input_dtype, parameters.outputDtype, parameters.keys.length === parameters.values.length);
  const source = staticInput(input, row.input_dtype);
  const exact = source && elements != null && source.length === elements && runtime.status === "pass"
    ? evaluate(source, parameters, version, elements) : unresolved(input);
  const duplicateKeyCount = duplicateCount(parameters.keys);
  const nanKeyCount = parameters.keys.filter((value) => typeof value === "number" && Number.isNaN(value)).length;
  const nonFiniteKeyCount = parameters.keys.filter(nonFinite).length;
  const nonFiniteValueCount = parameters.values.filter(nonFinite).length;
  const expectedRisks = [];
  if (runtime.status === "fail" && runtime.reason.includes("dtype_pair_missing")) {
    expectedRisks.push("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel");
  }
  if (runtime.status === "fail" && runtime.reason === "pinned_ort_key_value_count_mismatch") {
    expectedRisks.push("label_encoder_pinned_ort_runtime_contract_invalid");
  }
  if (version === 4 && duplicateKeyCount > 0) expectedRisks.push("label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict");
  if (version === 2 && nanKeyCount > 0) expectedRisks.push("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched");
  if (version === 1 && duplicateKeyCount > 0) expectedRisks.push("label_encoder_v1_duplicate_class_runtime_last_index");
  if (Number(exact.defaultCount || 0) > 0) expectedRisks.push("label_encoder_artifact_known_default_path_reached");
  if (Number(exact.mismatchCount || 0) > 0) expectedRisks.push("label_encoder_artifact_known_schema_runtime_output_mismatch");
  if (nonFiniteKeyCount > 0 || nonFiniteValueCount > 0 || nonFinite(parameters.defaultValue)) expectedRisks.push("label_encoder_non_finite_mapping_state");
  const semanticsAgree = exact.mismatchCount == null || exact.mismatchCount === 0;
  let outputStatic = null;
  if (row.output_dtype === "INT64" && output?.initializer_integer_values_exact_complete === true) {
    try { outputStatic = output.initializer_integer_values_exact_decimals.map(BigInt); } catch { outputStatic = null; }
  } else if (output?.static_values_complete === true) {
    outputStatic = row.output_dtype === "STRING" ? output.static_values : staticValuesWithSignedZeros(output);
  }
  const expectedOutput = exact.values && semanticsAgree ? exact.values : null;
  const outputHasExactValues = output?.static_values_complete === true || output?.initializer_integer_values_exact_complete === true;
  const outputMatches = expectedOutput ? arraysEqual(outputStatic, expectedOutput) : !outputHasExactValues;
  const valid = row.resolved_schema_version === version
    && row.input_kind === "tensor" && VERSION_DTYPES[version]?.has(row.input_dtype)
    && row.output_kind === "tensor" && row.output_dtype === parameters.outputDtype
    && row.exact_output_rank === (shapeDeclared ? shape.length : null)
    && JSON.stringify(row.exact_output_shape) === JSON.stringify(shapeDeclared ? shape : [])
    && row.exact_dense_output_element_count === elements
    && row.output_shape_basis === `pinned_onnx_label_encoder_v${version}_same_shape_mapping`
    && row.runtime_reference_status === "pinned_ort_cpu_label_encoder_versioned_kernels"
    && row.attribute_mode === parameters.attributeMode
    && row.label_encoder_onnx_contract_status === "pass"
    && row.label_encoder_pinned_ort_contract_status === runtime.status
    && row.label_encoder_pinned_ort_contract_reason === runtime.reason
    && row.label_encoder_key_dtype === parameters.keyDtype
    && row.label_encoder_value_dtype === parameters.outputDtype
    && row.label_encoder_key_source === parameters.keySource
    && row.label_encoder_value_source === parameters.valueSource
    && row.label_encoder_default_source === parameters.defaultSource
    && row.label_encoder_default_value === text(parameters.defaultValue)
    && row.label_encoder_key_count === parameters.keys.length
    && row.label_encoder_value_count === parameters.values.length
    && JSON.stringify(row.label_encoder_key_values) === JSON.stringify(parameters.keys.map(text))
    && JSON.stringify(row.label_encoder_value_values) === JSON.stringify(parameters.values.map(text))
    && row.label_encoder_duplicate_key_count === duplicateKeyCount
    && row.label_encoder_nan_key_count === nanKeyCount
    && row.label_encoder_non_finite_key_count === nonFiniteKeyCount
    && row.label_encoder_non_finite_value_count === nonFiniteValueCount
    && row.label_encoder_runtime_duplicate_policy === parameters.runtimePolicy
    && row.label_encoder_schema_duplicate_policy === parameters.schemaPolicy
    && row.label_encoder_static_assessment_status === exact.status
    && row.label_encoder_exact_input_value_count === exact.inputCount
    && row.label_encoder_exact_match_count === exact.matchCount
    && row.label_encoder_exact_default_count === exact.defaultCount
    && row.label_encoder_exact_duplicate_key_hit_count === exact.duplicateHitCount
    && row.label_encoder_schema_runtime_mismatch_count === exact.mismatchCount
    && row.label_encoder_output_materialized === Boolean(expectedOutput)
    && JSON.stringify(row.label_encoder_runtime_output_preview) === JSON.stringify(exact.outputPreview)
    && JSON.stringify(row.label_encoder_schema_output_preview) === JSON.stringify(exact.schemaPreview)
    && JSON.stringify(row.label_encoder_mismatch_input_preview) === JSON.stringify(exact.mismatchPreview)
    && row.vocabulary_type === parameters.keyDtype
    && row.vocabulary_count === parameters.keys.length
    && row.duplicate_vocabulary_count === duplicateKeyCount
    && row.category_pair_count === 0
    && row.active_duplicate_key_count === 0
    && row.active_default_type === parameters.outputDtype
    && row.active_default_value === text(parameters.defaultValue)
    && outputMatches
    && JSON.stringify([...row.risk_codes].sort()) === JSON.stringify(expectedRisks.sort());
  return valid;
}

function versionOne(op, inputDtype) {
  const classes = repeated(op, "classes_strings", "STRING", 8, true);
  if (!classes.ok) return classes;
  const outputDtype = inputDtype === "STRING" ? "INT64" : inputDtype === "INT64" ? "STRING" : "UNKNOWN";
  const keys = inputDtype === "STRING" ? classes.values : classes.values.map((_, index) => BigInt(index));
  const values = inputDtype === "STRING" ? classes.values.map((_, index) => BigInt(index)) : classes.values;
  const defaultValue = defaultFor(op, outputDtype, 1);
  if (!defaultValue.ok) return defaultValue;
  return {
    ok: true, keys, values, keyDtype: inputDtype, outputDtype,
    keySource: "classes_strings_or_implicit_indices", valueSource: "classes_strings_or_implicit_indices",
    defaultValue: defaultValue.value, defaultSource: defaultValue.source,
    attributeMode: "v1_classes_strings_bidirectional",
    runtimePolicy: inputDtype === "STRING" ? "last_key_wins" : "unique_implicit_indices",
    schemaPolicy: inputDtype === "STRING" ? "unspecified_list_lookup_index" : "unique_implicit_indices",
  };
}

function versioned(op, inputDtype, version) {
  if (!version) return { ok: false };
  const keyCandidates = version >= 4
    ? [["keys_tensor", "TENSOR", 4], ["keys_strings", "STRING", 8], ["keys_int64s", "INT64", 7], ["keys_floats", "FLOAT32", 6]]
    : [["keys_strings", "STRING", 8], ["keys_int64s", "INT64", 7], ["keys_floats", "FLOAT32", 6]];
  const valueCandidates = version >= 4
    ? [["values_tensor", "TENSOR", 4], ["values_strings", "STRING", 8], ["values_int64s", "INT64", 7], ["values_floats", "FLOAT32", 6]]
    : [["values_strings", "STRING", 8], ["values_int64s", "INT64", 7], ["values_floats", "FLOAT32", 6]];
  const keys = select(op, keyCandidates);
  const values = select(op, valueCandidates);
  if (!keys.ok || !values.ok || keys.dtype !== inputDtype || version >= 4 && keys.values.length !== values.values.length) return { ok: false };
  const defaultValue = defaultFor(op, values.dtype, version);
  if (!defaultValue.ok) return defaultValue;
  return {
    ok: true, keys: keys.values, values: values.values, keyDtype: keys.dtype, outputDtype: values.dtype,
    keySource: keys.source, valueSource: values.source, defaultValue: defaultValue.value, defaultSource: defaultValue.source,
    attributeMode: `v${version}_parallel_key_value_attributes`, runtimePolicy: "first_key_wins",
    schemaPolicy: version === 4 ? "last_key_wins" : "unspecified_except_bitwise_float_lookup",
  };
}

function select(op, candidates) {
  const present = candidates.filter(([name]) => Boolean(attr(op, name)));
  if (present.length !== 1) return { ok: false };
  const [name, dtype, type] = present[0];
  const value = attr(op, name);
  return dtype === "TENSOR" ? tensorAttribute(value, name) : repeated(op, name, dtype, type);
}

function repeated(op, name, dtype, type, optional = false) {
  const value = attr(op, name);
  if (!value) return optional ? { ok: true, values: [], dtype, source: "onnx_schema_default_empty_list" } : { ok: false };
  if (value.type !== type) return { ok: false };
  if (dtype === "STRING") return { ok: Array.isArray(value.string_values), values: value.string_values || [], dtype, source: name };
  if (dtype === "FLOAT32") {
    if (!Array.isArray(value.float_values_text)) return { ok: false };
    return { ok: true, values: value.float_values_text.map(parseNumber), dtype, source: name };
  }
  try {
    return { ok: Array.isArray(value.int_values_exact_decimal), values: (value.int_values_exact_decimal || []).map(BigInt), dtype, source: name };
  } catch { return { ok: false }; }
}

function tensorAttribute(attribute, source) {
  const tensor = attribute?.tensor_value;
  if (attribute?.type !== 4 || !tensor || tensor.shape_declared !== true || !Array.isArray(tensor.shape)
    || tensor.shape.length !== 1 || !knownDimension(tensor.shape[0]) || tensor.exact_values_complete !== true
    || !Array.isArray(tensor.exact_values_text) || tensor.exact_values_text.length !== tensor.shape[0]
    || !VERSION_DTYPES[4].has(tensor.dtype) || Number(tensor.external_data_entries || 0) !== 0 || Number(tensor.data_location || 0) !== 0) return { ok: false };
  try {
    const values = tensor.exact_values_text.map((value) => parseTyped(value, tensor.dtype));
    return { ok: true, values, dtype: tensor.dtype, source };
  } catch { return { ok: false }; }
}

function defaultFor(op, dtype, version) {
  const defaultTensor = version >= 4 ? attr(op, "default_tensor") : null;
  if (defaultTensor) {
    const decoded = tensorAttribute(defaultTensor, "default_tensor");
    return decoded.ok && decoded.dtype === dtype && decoded.values.length === 1
      ? { ok: true, value: decoded.values[0], source: "explicit_default_tensor" } : { ok: false };
  }
  const definition = dtype === "STRING" ? ["default_string", "_Unused"]
    : dtype === "INT64" ? ["default_int64", -1n]
      : dtype === "FLOAT32" ? ["default_float", -0]
        : dtype === "INT16" || dtype === "INT32" ? ["", -1]
          : dtype === "FLOAT64" ? ["", -0] : null;
  if (!definition) return { ok: false };
  const [name, fallback] = definition;
  const value = name ? attr(op, name) : null;
  if (!value) return { ok: true, value: fallback, source: `onnx_schema_default_${text(fallback)}` };
  try {
    return { ok: true, value: dtype === "STRING" ? value.string_value : dtype === "INT64" ? BigInt(value.int_value_exact_decimal) : parseNumber(value.float_value_text), source: `explicit_${name}` };
  } catch { return { ok: false }; }
}

function runtimeContract(version, inputDtype, outputDtype, countsEqual) {
  if (!countsEqual) return { status: "fail", reason: "pinned_ort_key_value_count_mismatch" };
  if (version === 1) return ["STRING:INT64", "INT64:STRING"].includes(`${inputDtype}:${outputDtype}`)
    ? { status: "pass", reason: "pinned_ort_v1_bidirectional_kernel" } : { status: "fail", reason: "pinned_ort_v1_dtype_pair_missing" };
  if (version === 2) return VERSION_DTYPES[2].has(inputDtype) && VERSION_DTYPES[2].has(outputDtype)
    ? { status: "pass", reason: "pinned_ort_v2_versioned_dtype_pair_kernel" } : { status: "fail", reason: "pinned_ort_v2_dtype_pair_missing" };
  return V4_ORT_PAIRS.has(`${inputDtype}:${outputDtype}`)
    ? { status: "pass", reason: "pinned_ort_v4_typed_dtype_pair_kernel" }
    : { status: "fail", reason: `pinned_ort_v4_dtype_pair_missing:${inputDtype}:${outputDtype}` };
}

function evaluate(source, parameters, version, elements) {
  const runtimeMap = map(parameters.keys, parameters.values, parameters.runtimePolicy);
  const schemaMap = version === 4 ? map(parameters.keys, parameters.values, "last_key_wins") : null;
  const duplicateKeys = duplicateSet(parameters.keys);
  const values = elements <= LIMIT ? [] : null;
  const outputPreview = [];
  const schemaPreview = [];
  const mismatchPreview = [];
  let matchCount = 0;
  let defaultCount = 0;
  let duplicateHitCount = 0;
  let mismatchCount = version === 4 ? 0 : null;
  source.forEach((input) => {
    const key = identity(input);
    const matched = !(version === 2 && typeof input === "number" && Number.isNaN(input)) && runtimeMap.has(key);
    const output = matched ? runtimeMap.get(key) : parameters.defaultValue;
    if (matched) matchCount += 1; else defaultCount += 1;
    if (duplicateKeys.has(key)) duplicateHitCount += 1;
    if (values) values.push(output);
    if (outputPreview.length < 16) outputPreview.push(text(output));
    if (schemaMap) {
      const schemaOutput = schemaMap.has(key) ? schemaMap.get(key) : parameters.defaultValue;
      if (schemaPreview.length < 16) schemaPreview.push(text(schemaOutput));
      if (!same(output, schemaOutput)) {
        mismatchCount += 1;
        if (mismatchPreview.length < 8) mismatchPreview.push(text(input));
      }
    }
  });
  return {
    status: values ? "assessed_exact_pinned_ort_semantics" : "assessed_counts_output_not_materialized_limit",
    inputCount: source.length, matchCount, defaultCount, duplicateHitCount, mismatchCount,
    values, outputPreview, schemaPreview, mismatchPreview,
  };
}

function unresolved(input) {
  return {
    status: input?.initializer_storage_kind ? input.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    inputCount: null, matchCount: null, defaultCount: null, duplicateHitCount: null, mismatchCount: null,
    values: null, outputPreview: [], schemaPreview: [], mismatchPreview: [],
  };
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializer_integer_values_exact_complete === true) {
    try { return input.initializer_integer_values_exact_decimals.map(BigInt); } catch { return null; }
  }
  if (dtype === "STRING") return input?.static_values_complete === true ? input.static_values : null;
  if (["FLOAT32", "FLOAT64"].includes(dtype) && input?.static_values_canonical_text_complete === true
    && Array.isArray(input.static_values_canonical_texts)) {
    return input.static_values_canonical_texts.map(parseNumber);
  }
  return staticValuesWithSignedZeros(input);
}

function map(keys, values, policy) {
  const result = new Map();
  keys.forEach((key, index) => {
    const id = identity(key);
    if (policy === "last_key_wins" || !result.has(id)) result.set(id, values[index]);
  });
  return result;
}

function duplicateCount(values) { return values.length - new Set(values.map(identity)).size; }
function duplicateSet(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(identity(value), (counts.get(identity(value)) || 0) + 1));
  return new Set([...counts].filter(([, count]) => count > 1).map(([key]) => key));
}
function identity(value) {
  if (typeof value === "bigint") return `i:${value}`;
  if (typeof value === "string") return `s:${value}`;
  if (Number.isNaN(value)) return "n:NaN";
  if (value === Infinity) return "n:Infinity";
  if (value === -Infinity) return "n:-Infinity";
  return `n:${Object.is(value, -0) ? 0 : value}`;
}
function parseTyped(value, dtype) {
  if (dtype === "STRING") return value;
  if (dtype === "INT64") return BigInt(value);
  const parsed = parseNumber(value);
  if (["INT16", "INT32"].includes(dtype) && !Number.isSafeInteger(parsed)) throw new Error("invalid integer");
  return parsed;
}
function parseNumber(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}
function text(value) {
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}
function nonFinite(value) { return typeof value === "number" && !Number.isFinite(value); }
function same(left, right) { return typeof left === "number" && typeof right === "number" ? Object.is(left, right) || left === right : left === right; }
function arraysEqual(left, right) {
  if (right.some((value) => typeof value === "string" || typeof value === "bigint")) {
    return Array.isArray(left) && left.length === right.length && left.every((value, index) => String(value) === String(right[index]));
  }
  return numericArraysExactlyEqual(left, right);
}
function resolveVersion(opset) { return opset >= 4 ? 4 : opset >= 2 ? 2 : opset >= 1 ? 1 : null; }
function attr(op, name) { return (op?.onnx_attributes || []).find((value) => value.name === name); }
function knownDimension(value) { return Number.isSafeInteger(value) && value >= 0; }
function shapeProduct(shape) {
  let product = 1;
  for (const dimension of shape) {
    product *= dimension;
    if (!Number.isSafeInteger(product) || product < 0) return null;
  }
  return product;
}
