import {
  canonicalOnnxTypeProto,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";

export const LABEL_ENCODER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;

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

export function resolveOnnxMlLabelEncoderVersion(importedOpset) {
  return importedOpset >= 4 ? 4 : importedOpset >= 2 ? 2 : importedOpset >= 1 ? 1 : null;
}

export function inferOnnxMlLabelEncoder({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const reasons = [];
  const failures = [];
  const schemaVersion = resolveOnnxMlLabelEncoderVersion(importedOpset);
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("label_encoder_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`label_encoder_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (schemaVersion == null) failures.push("label_encoder_operator_not_defined_at_imported_opset");
  else if (inputDtype === "UNKNOWN") reasons.push("label_encoder_input_dtype_unresolved");
  else if (!VERSION_DTYPES[schemaVersion].has(inputDtype)) failures.push(`label_encoder_input_dtype_not_supported_by_schema_v${schemaVersion}:${inputDtype}`);

  const parameters = schemaVersion === 1
    ? versionOneParameters(node, inputDtype)
    : versionedKeyValueParameters(node, inputDtype, schemaVersion);
  failures.push(...parameters.schemaFailures);
  const outputDtype = parameters.outputDtype;
  const shapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  const inputShape = shapeDeclared ? [...inputType.shape] : [];
  if (!shapeDeclared) reasons.push("label_encoder_input_shape_unresolved");
  else if (inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("label_encoder_dynamic_input_shape_preserved");
  const outputType = outputDtype !== "UNKNOWN" ? makeOnnxTensorType(outputDtype, inputShape, shapeDeclared) : null;
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = shapeDeclared && inputShape.every(knownDimension) ? shapeProduct(inputShape) : null;
  if (shapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("label_encoder_output_element_count_overflow");

  const runtime = pinnedOrtContract(schemaVersion, inputDtype, outputDtype, parameters);
  const source = staticInput(input, inputDtype);
  let exact = unresolvedStaticResult(input);
  if (source && exactOutputElements != null && source.length !== exactOutputElements) {
    failures.push(`label_encoder_static_value_count_mismatch:${source.length}:${exactOutputElements}`);
  } else if (source && runtime.status === "pass" && parameters.keyValues && parameters.defaultValue?.ok) {
    exact = evaluateStatic(source, parameters, schemaVersion, exactOutputElements);
  }
  const semanticsAgree = exact.schemaRuntimeMismatchCount == null || exact.schemaRuntimeMismatchCount === 0;
  if (patch && exact.outputValues && semanticsAgree) {
    if (outputDtype === "INT64") {
      const decimals = exact.outputValues.map((value) => BigInt(value).toString());
      const safeValues = decimals.map((value) => Number(value));
      patch.initializerIntegerValuesStatus = safeValues.every(Number.isSafeInteger) ? "complete" : "not_assessed_outside_safe_integer";
      patch.initializerIntegerValuesComplete = safeValues.every(Number.isSafeInteger);
      patch.initializerIntegerValues = safeValues.every(Number.isSafeInteger) ? safeValues : [];
      patch.initializerIntegerValuesExactComplete = true;
      patch.initializerIntegerValuesExactDecimals = decimals;
      if (safeValues.every(Number.isSafeInteger)) {
        patch.staticValuesStatus = "complete";
        patch.staticValuesComplete = true;
        patch.staticValues = safeValues;
      } else {
        patch.staticValuesStatus = "complete_exact_int64_decimal_only";
      }
    } else {
      patch.staticValuesStatus = "complete";
      patch.staticValuesComplete = true;
      patch.staticValues = exact.outputValues;
    }
    patch.staticValuesSource = `label_encoder_v${schemaVersion}_pinned_ort_cpu_semantics`;
  }

  const riskCodes = [];
  if (runtime.status === "fail" && runtime.reason.includes("dtype_pair_missing") && parameters.schemaFailures.length === 0) {
    riskCodes.push("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel");
  }
  if (runtime.status === "fail" && runtime.reason === "pinned_ort_key_value_count_mismatch") {
    riskCodes.push("label_encoder_pinned_ort_runtime_contract_invalid");
  }
  if (schemaVersion === 4 && parameters.duplicateKeyCount > 0) {
    riskCodes.push("label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict");
  }
  if (schemaVersion === 2 && parameters.nanKeyCount > 0) {
    riskCodes.push("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched");
  }
  if (schemaVersion === 1 && parameters.duplicateKeyCount > 0) {
    riskCodes.push("label_encoder_v1_duplicate_class_runtime_last_index");
  }
  if (Number(exact.defaultCount || 0) > 0) riskCodes.push("label_encoder_artifact_known_default_path_reached");
  if (Number(exact.schemaRuntimeMismatchCount || 0) > 0) riskCodes.push("label_encoder_artifact_known_schema_runtime_output_mismatch");
  if (parameters.nonFiniteKeyCount > 0 || parameters.nonFiniteValueCount > 0 || parameters.defaultValue?.nonFinite) {
    riskCodes.push("label_encoder_non_finite_mapping_state");
  }

  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "LabelEncoder", contract_kind: "tensor_label_mapping",
    imported_opset: importedOpset, resolved_schema_version: schemaVersion, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: shapeDeclared ? inputShape.length : null, input_shape: inputShape,
    exact_batch_count: null, exact_feature_count: null,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: shapeDeclared && outputType ? inputShape.length : null,
    exact_output_shape: shapeDeclared && outputType ? inputShape : [],
    exact_dense_output_element_count: outputType ? exactOutputElements : null,
    output_shape_basis: `pinned_onnx_label_encoder_v${schemaVersion || "unresolved"}_same_shape_mapping`,
    runtime_reference_status: "pinned_ort_cpu_label_encoder_versioned_kernels",
    attribute_mode: parameters.attributeMode,
    label_encoder_onnx_contract_status: failures.length ? "fail" : outputDtype === "UNKNOWN" ? "not_assessed" : "pass",
    label_encoder_pinned_ort_contract_status: runtime.status,
    label_encoder_pinned_ort_contract_reason: runtime.reason,
    label_encoder_key_dtype: parameters.keyDtype,
    label_encoder_value_dtype: outputDtype,
    label_encoder_key_source: parameters.keySource,
    label_encoder_value_source: parameters.valueSource,
    label_encoder_default_source: parameters.defaultValue?.source || "unresolved",
    label_encoder_default_value: parameters.defaultValue?.ok ? valueText(parameters.defaultValue.value) : "unresolved",
    label_encoder_key_count: parameters.keys.length,
    label_encoder_value_count: parameters.values.length,
    label_encoder_key_values: parameters.keys.map(valueText),
    label_encoder_value_values: parameters.values.map(valueText),
    label_encoder_duplicate_key_count: parameters.duplicateKeyCount,
    label_encoder_nan_key_count: parameters.nanKeyCount,
    label_encoder_non_finite_key_count: parameters.nonFiniteKeyCount,
    label_encoder_non_finite_value_count: parameters.nonFiniteValueCount,
    label_encoder_runtime_duplicate_policy: parameters.runtimeDuplicatePolicy,
    label_encoder_schema_duplicate_policy: parameters.schemaDuplicatePolicy,
    label_encoder_static_assessment_status: exact.status,
    label_encoder_exact_input_value_count: exact.inputCount,
    label_encoder_exact_match_count: exact.matchCount,
    label_encoder_exact_default_count: exact.defaultCount,
    label_encoder_exact_duplicate_key_hit_count: exact.duplicateKeyHitCount,
    label_encoder_schema_runtime_mismatch_count: exact.schemaRuntimeMismatchCount,
    label_encoder_output_materialized: Boolean(exact.outputValues && semanticsAgree),
    label_encoder_runtime_output_preview: exact.outputPreview,
    label_encoder_schema_output_preview: exact.schemaOutputPreview,
    label_encoder_mismatch_input_preview: exact.mismatchInputPreview,
    vocabulary_type: parameters.keyDtype,
    vocabulary_count: parameters.keys.length,
    duplicate_vocabulary_count: parameters.duplicateKeyCount,
    vocabulary_preview: parameters.keys.slice(0, 8).map(valueText),
    mapping_direction: `${parameters.keyDtype}_TO_${outputDtype}`,
    // Category pair fields belong to CategoryMapper's bidirectional contract.
    // LabelEncoder owns a separate versioned key/value ledger above.
    category_pair_count: 0,
    category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: outputDtype, active_default_value: parameters.defaultValue?.ok ? valueText(parameters.defaultValue.value) : "",
    category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null, exact_truncated_feature_count_per_batch: null,
    padded_input_count: 0, truncated_input_count: 0,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_values: [], exact_index_preview: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  // ONNX same-shape/type inference remains valid even when the pinned ORT CPU
  // build has no executable kernel for the serialized contract.
  const canPropagate = status !== "fail" && outputType;
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: canPropagate && patch && node.outputs?.[0] ? [[node.outputs[0], patch]] : [] }, row,
  };
}

function versionOneParameters(node, inputDtype) {
  const schemaFailures = [];
  const classes = decodeRepeatedAttribute(node.attributes?.get("classes_strings"), "STRING", 8, "classes_strings", { optional: true });
  if (!classes.ok) schemaFailures.push(classes.reason);
  const outputDtype = inputDtype === "STRING" ? "INT64" : inputDtype === "INT64" ? "STRING" : "UNKNOWN";
  const keys = inputDtype === "STRING" ? classes.values : classes.values.map((_, index) => BigInt(index));
  const values = inputDtype === "STRING" ? classes.values.map((_, index) => BigInt(index)) : classes.values;
  const defaultValue = scalarDefault(node, outputDtype, 1);
  if (!defaultValue.ok) schemaFailures.push(defaultValue.reason);
  return finishParameters({
    schemaFailures, outputDtype, keys, values, keyDtype: inputDtype, keySource: "classes_strings_or_implicit_indices",
    valueSource: "classes_strings_or_implicit_indices", defaultValue, attributeMode: "v1_classes_strings_bidirectional",
    runtimeDuplicatePolicy: inputDtype === "STRING" ? "last_key_wins" : "unique_implicit_indices",
    schemaDuplicatePolicy: inputDtype === "STRING" ? "unspecified_list_lookup_index" : "unique_implicit_indices",
  });
}

function versionedKeyValueParameters(node, inputDtype, schemaVersion) {
  const schemaFailures = [];
  const keyCandidates = schemaVersion >= 4
    ? [["keys_tensor", "TENSOR", 4], ["keys_strings", "STRING", 8], ["keys_int64s", "INT64", 7], ["keys_floats", "FLOAT32", 6]]
    : [["keys_strings", "STRING", 8], ["keys_int64s", "INT64", 7], ["keys_floats", "FLOAT32", 6]];
  const valueCandidates = schemaVersion >= 4
    ? [["values_tensor", "TENSOR", 4], ["values_strings", "STRING", 8], ["values_int64s", "INT64", 7], ["values_floats", "FLOAT32", 6]]
    : [["values_strings", "STRING", 8], ["values_int64s", "INT64", 7], ["values_floats", "FLOAT32", 6]];
  const keys = selectMappingAttribute(node, keyCandidates, "keys");
  const values = selectMappingAttribute(node, valueCandidates, "values");
  if (!keys.ok) schemaFailures.push(keys.reason);
  if (!values.ok) schemaFailures.push(values.reason);
  if (keys.ok && inputDtype !== "UNKNOWN" && keys.dtype !== inputDtype) schemaFailures.push(`label_encoder_key_dtype_mismatch:${keys.dtype}:${inputDtype}`);
  if (keys.ok && values.ok && schemaVersion >= 4 && keys.values.length !== values.values.length) {
    schemaFailures.push(`label_encoder_schema_v4_key_value_count_mismatch:${keys.values.length}:${values.values.length}`);
  }
  const defaultValue = scalarDefault(node, values.ok ? values.dtype : "UNKNOWN", schemaVersion);
  if (!defaultValue.ok) schemaFailures.push(defaultValue.reason);
  return finishParameters({
    schemaFailures, outputDtype: values.ok ? values.dtype : "UNKNOWN",
    keys: keys.ok ? keys.values : [], values: values.ok ? values.values : [],
    keyDtype: keys.ok ? keys.dtype : "UNKNOWN", keySource: keys.source, valueSource: values.source,
    defaultValue, attributeMode: `v${schemaVersion}_parallel_key_value_attributes`,
    runtimeDuplicatePolicy: "first_key_wins",
    schemaDuplicatePolicy: schemaVersion === 4 ? "last_key_wins" : "unspecified_except_bitwise_float_lookup",
  });
}

function finishParameters(parameters) {
  const duplicateKeyCount = duplicateCount(parameters.keys);
  return {
    ...parameters,
    keyValues: parameters.keys.length === parameters.values.length && parameters.schemaFailures.length === 0,
    duplicateKeyCount,
    nanKeyCount: parameters.keys.filter((value) => typeof value === "number" && Number.isNaN(value)).length,
    nonFiniteKeyCount: parameters.keys.filter((value) => typeof value === "number" && !Number.isFinite(value)).length,
    nonFiniteValueCount: parameters.values.filter((value) => typeof value === "number" && !Number.isFinite(value)).length,
  };
}

function selectMappingAttribute(node, candidates, role) {
  const present = candidates.filter(([name]) => node.attributes?.has(name));
  if (present.length !== 1) return { ok: false, reason: `label_encoder_requires_exactly_one_${role}_attribute:${present.length}`, values: [], dtype: "UNKNOWN", source: "unresolved" };
  const [name, dtype, type] = present[0];
  const attribute = node.attributes.get(name);
  if (dtype === "TENSOR") return decodeTensorAttribute(attribute, name);
  return decodeRepeatedAttribute(attribute, dtype, type, name);
}

function decodeRepeatedAttribute(attribute, dtype, type, source, { optional = false } = {}) {
  if (!attribute) return optional ? { ok: true, values: [], dtype, source: "onnx_schema_default_empty_list" }
    : { ok: false, reason: `label_encoder_attribute_missing:${source}`, values: [], dtype, source };
  if (attribute.type !== type || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== type) {
    return { ok: false, reason: `label_encoder_attribute_type_invalid:${source}`, values: [], dtype, source };
  }
  if (dtype === "STRING") return { ok: true, values: [...attribute.strings], dtype, source };
  if (dtype === "FLOAT32") return { ok: true, values: [...attribute.floats], dtype, source };
  const exact = Array.isArray(attribute.intExactDecimals) ? attribute.intExactDecimals : [];
  try {
    return { ok: exact.length === attribute.ints.length, values: exact.map((value) => BigInt(value)), dtype, source,
      reason: exact.length === attribute.ints.length ? "" : `label_encoder_int64_attribute_exact_value_gap:${source}` };
  } catch {
    return { ok: false, reason: `label_encoder_int64_attribute_decode_failed:${source}`, values: [], dtype, source };
  }
}

function decodeTensorAttribute(attribute, source) {
  const tensor = attribute?.tensor;
  if (attribute?.type !== 4 || !tensor || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== 4) {
    return { ok: false, reason: `label_encoder_tensor_attribute_invalid:${source}`, values: [], dtype: "UNKNOWN", source };
  }
  if (!Array.isArray(tensor.shape) || tensor.shape.length !== 1 || !knownDimension(tensor.shape[0])) {
    return { ok: false, reason: `label_encoder_tensor_attribute_not_1d:${source}`, values: [], dtype: tensor.dtype || "UNKNOWN", source };
  }
  if (!VERSION_DTYPES[4].has(tensor.dtype)) {
    return { ok: false, reason: `label_encoder_tensor_attribute_dtype_invalid:${source}:${tensor.dtype || "UNKNOWN"}`, values: [], dtype: tensor.dtype || "UNKNOWN", source };
  }
  if (Number(tensor.externalDataEntries || 0) > 0 || Number(tensor.dataLocation || 0) !== 0) {
    return { ok: false, reason: `label_encoder_tensor_attribute_external_payload_not_executable:${source}`, values: [], dtype: tensor.dtype, source };
  }
  const decoded = decodeTensorValues(tensor);
  if (!decoded.ok || decoded.values.length !== tensor.shape[0]) {
    return { ok: false, reason: decoded.reason || `label_encoder_tensor_attribute_cardinality_mismatch:${source}`, values: [], dtype: tensor.dtype, source };
  }
  return { ok: true, values: decoded.values, dtype: tensor.dtype, source };
}

function decodeTensorValues(tensor) {
  const count = tensor.shape[0];
  if (tensor.dtype === "STRING") {
    const values = Array.isArray(tensor.stringValues) ? [...tensor.stringValues] : [];
    return { ok: values.length === count, values, reason: values.length === count ? "" : "label_encoder_string_tensor_cardinality_mismatch" };
  }
  let values = [];
  if (tensor.rawData instanceof Uint8Array) {
    const bytes = { INT16: 2, INT32: 4, INT64: 8, FLOAT32: 4, FLOAT64: 8 }[tensor.dtype];
    if (!bytes || tensor.rawData.byteLength !== count * bytes) return { ok: false, values: [], reason: "label_encoder_tensor_raw_data_size_mismatch" };
    const view = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
    for (let index = 0; index < count; index += 1) {
      const offset = index * bytes;
      values.push(tensor.dtype === "INT16" ? view.getInt16(offset, true)
        : tensor.dtype === "INT32" ? view.getInt32(offset, true)
          : tensor.dtype === "INT64" ? view.getBigInt64(offset, true)
            : tensor.dtype === "FLOAT32" ? view.getFloat32(offset, true) : view.getFloat64(offset, true));
    }
  } else if (Array.isArray(tensor.typedValues)) {
    values = tensor.typedValues.map((value) => tensor.dtype === "INT64" ? BigInt(value) : Number(value));
  }
  return { ok: values.length === count, values, reason: values.length === count ? "" : "label_encoder_tensor_payload_not_exactly_decoded" };
}

function scalarDefault(node, outputDtype, schemaVersion) {
  if (schemaVersion >= 4 && node.attributes?.has("default_tensor")) {
    const decoded = decodeTensorAttribute(node.attributes.get("default_tensor"), "default_tensor");
    if (!decoded.ok) return { ok: false, reason: decoded.reason, source: "default_tensor", value: null };
    if (decoded.values.length !== 1) return { ok: false, reason: "label_encoder_default_tensor_not_singleton", source: "default_tensor", value: null };
    if (decoded.dtype !== outputDtype) return { ok: false, reason: `label_encoder_default_tensor_dtype_mismatch:${decoded.dtype}:${outputDtype}`, source: "default_tensor", value: null };
    return { ok: true, source: "explicit_default_tensor", value: decoded.values[0], nonFinite: isNonFinite(decoded.values[0]) };
  }
  const definition = outputDtype === "STRING" ? ["default_string", 3, "_Unused"]
    : outputDtype === "INT64" ? ["default_int64", 2, -1n]
      : outputDtype === "FLOAT32" ? ["default_float", 1, -0]
        : outputDtype === "INT16" || outputDtype === "INT32" ? ["", 0, -1]
          : outputDtype === "FLOAT64" ? ["", 0, -0] : null;
  if (!definition) return { ok: false, reason: "label_encoder_default_output_dtype_unresolved", source: "unresolved", value: null };
  const [name, type, fallback] = definition;
  const attribute = name ? node.attributes?.get(name) : null;
  if (!attribute) return { ok: true, source: `onnx_schema_default_${valueText(fallback)}`, value: fallback, nonFinite: false };
  if (attribute.type !== type || !Array.isArray(attribute.valueTypesPresent)
    || attribute.valueTypesPresent.length !== 1 || attribute.valueTypesPresent[0] !== type) {
    return { ok: false, reason: `label_encoder_default_attribute_invalid:${name}`, source: name, value: null };
  }
  let value;
  try {
    value = type === 3 ? attribute.s : type === 2 ? BigInt(attribute.iExactDecimal || attribute.i) : attribute.f;
  } catch {
    return { ok: false, reason: `label_encoder_default_attribute_decode_failed:${name}`, source: name, value: null };
  }
  return { ok: true, source: `explicit_${name}`, value, nonFinite: isNonFinite(value) };
}

function pinnedOrtContract(version, inputDtype, outputDtype, parameters) {
  if (!version || inputDtype === "UNKNOWN" || outputDtype === "UNKNOWN") return { status: "not_assessed", reason: "dtype_or_schema_unresolved" };
  if (parameters.schemaFailures.length) return { status: "fail", reason: parameters.schemaFailures[0] };
  if (parameters.keys.length !== parameters.values.length) return { status: "fail", reason: "pinned_ort_key_value_count_mismatch" };
  if (version === 1) return ["STRING:INT64", "INT64:STRING"].includes(`${inputDtype}:${outputDtype}`)
    ? { status: "pass", reason: "pinned_ort_v1_bidirectional_kernel" }
    : { status: "fail", reason: "pinned_ort_v1_dtype_pair_missing" };
  if (version === 2) return VERSION_DTYPES[2].has(inputDtype) && VERSION_DTYPES[2].has(outputDtype)
    ? { status: "pass", reason: "pinned_ort_v2_versioned_dtype_pair_kernel" }
    : { status: "fail", reason: "pinned_ort_v2_dtype_pair_missing" };
  return V4_ORT_PAIRS.has(`${inputDtype}:${outputDtype}`)
    ? { status: "pass", reason: "pinned_ort_v4_typed_dtype_pair_kernel" }
    : { status: "fail", reason: `pinned_ort_v4_dtype_pair_missing:${inputDtype}:${outputDtype}` };
}

function evaluateStatic(source, parameters, version, outputElementCount) {
  const runtimePolicy = parameters.runtimeDuplicatePolicy;
  const runtimeMap = mapping(parameters.keys, parameters.values, runtimePolicy);
  const schemaMap = version === 4 ? mapping(parameters.keys, parameters.values, "last_key_wins") : null;
  const duplicateKeys = duplicateKeysSet(parameters.keys);
  const outputValues = outputElementCount <= LABEL_ENCODER_MAX_PROPAGATED_STATIC_VALUES ? [] : null;
  const schemaValues = schemaMap && outputElementCount <= LABEL_ENCODER_MAX_PROPAGATED_STATIC_VALUES ? [] : null;
  const outputPreview = [];
  const schemaOutputPreview = [];
  const mismatchInputPreview = [];
  let matchCount = 0;
  let defaultCount = 0;
  let duplicateKeyHitCount = 0;
  let schemaRuntimeMismatchCount = version === 4 ? 0 : null;
  source.forEach((value) => {
    const key = lookupKey(value);
    const v2NanMiss = version === 2 && typeof value === "number" && Number.isNaN(value);
    const matched = !v2NanMiss && runtimeMap.has(key);
    const runtimeValue = matched ? runtimeMap.get(key) : parameters.defaultValue.value;
    if (matched) matchCount += 1; else defaultCount += 1;
    if (duplicateKeys.has(key)) duplicateKeyHitCount += 1;
    if (outputValues) outputValues.push(runtimeValue);
    if (outputPreview.length < 16) outputPreview.push(valueText(runtimeValue));
    if (schemaMap) {
      const schemaValue = schemaMap.has(key) ? schemaMap.get(key) : parameters.defaultValue.value;
      if (schemaValues) schemaValues.push(schemaValue);
      if (schemaOutputPreview.length < 16) schemaOutputPreview.push(valueText(schemaValue));
      if (!sameValue(runtimeValue, schemaValue)) {
        schemaRuntimeMismatchCount += 1;
        if (mismatchInputPreview.length < 8) mismatchInputPreview.push(valueText(value));
      }
    }
  });
  return {
    status: outputValues ? "assessed_exact_pinned_ort_semantics" : "assessed_counts_output_not_materialized_limit",
    inputCount: source.length, matchCount, defaultCount, duplicateKeyHitCount, schemaRuntimeMismatchCount,
    outputValues, outputPreview, schemaOutputPreview, mismatchInputPreview,
  };
}

function unresolvedStaticResult(input) {
  return {
    status: input?.role === "initializer" ? input.staticValuesStatus || "not_assessed_initializer_values" : "not_assessed_runtime_values",
    inputCount: null, matchCount: null, defaultCount: null, duplicateKeyHitCount: null,
    schemaRuntimeMismatchCount: null, outputValues: null, outputPreview: [], schemaOutputPreview: [], mismatchInputPreview: [],
  };
}

function staticInput(input, dtype) {
  if (dtype === "INT64" && input?.initializerIntegerValuesExactComplete === true
    && Array.isArray(input.initializerIntegerValuesExactDecimals)) {
    try { return input.initializerIntegerValuesExactDecimals.map((value) => BigInt(value)); } catch { return null; }
  }
  if (VERSION_DTYPES[4].has(dtype) && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) return input.staticValues;
  if (["FLOAT32", "FLOAT64"].includes(dtype) && input?.staticValuesCanonicalTextComplete === true
    && Array.isArray(input.staticValuesCanonicalTexts)) {
    return input.staticValuesCanonicalTexts.map(parseCanonicalNumber);
  }
  return null;
}

function parseCanonicalNumber(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

function mapping(keys, values, policy) {
  const result = new Map();
  keys.forEach((key, index) => {
    const identity = lookupKey(key);
    if (policy === "last_key_wins" || !result.has(identity)) result.set(identity, values[index]);
  });
  return result;
}

function duplicateCount(values) {
  return values.length - new Set(values.map(lookupKey)).size;
}

function duplicateKeysSet(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(lookupKey(value), (counts.get(lookupKey(value)) || 0) + 1));
  return new Set([...counts].filter(([, count]) => count > 1).map(([key]) => key));
}

function lookupKey(value) {
  if (typeof value === "bigint") return `i:${value}`;
  if (typeof value === "string") return `s:${value}`;
  if (Number.isNaN(value)) return "n:NaN";
  if (value === Number.POSITIVE_INFINITY) return "n:Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "n:-Infinity";
  return `n:${Object.is(value, -0) ? 0 : value}`;
}

function sameValue(left, right) {
  return typeof left === "number" && typeof right === "number" ? Object.is(left, right) || left === right : left === right;
}

function valueText(value) {
  if (typeof value === "bigint") return value.toString();
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function isNonFinite(value) {
  return typeof value === "number" && !Number.isFinite(value);
}

function knownDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function shapeProduct(shape) {
  let product = 1;
  for (const dimension of shape) {
    product *= dimension;
    if (!Number.isSafeInteger(product) || product < 0) return null;
  }
  return product;
}
