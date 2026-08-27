const ONNX_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const ORT_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";

const MAX_STATIC_OUTPUT_ELEMENTS = 65_536;
const MAX_STATIC_WORK_UNITS = 5_000_000;

export const ONNX_TFIDF_VECTORIZER_SOURCE = Object.freeze({
  schema: "deepbom.onnx_tfidf_vectorizer_source.v1",
  onnx_release: "v1.21.0",
  onnx_commit: ONNX_COMMIT,
  ort_release: "v1.26.0",
  ort_commit: ORT_COMMIT,
  documents: Object.freeze([
    source("onnx_schema_and_shape", "onnx/defs/nn/defs.cc", "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02", "onnx"),
    source("onnx_reference", "onnx/reference/ops/op_tfidf_vectorizer.py", "ac8e8495a50a0b85fb0f4adf5de2284efa2f5e14a999d4e1ac12e20c1079e69d", "onnx"),
    source("onnx_backend_tests", "onnx/backend/test/case/node/tfidfvectorizer.py", "9b8e1d00174a66727864baf40964c6f3da5d4e3a72858f0ee34589990b14e9dc", "onnx"),
    source("ort_cpu_kernel_header", "onnxruntime/core/providers/cpu/nn/tfidfvectorizer.h", "76308c5d7cf403eec02d9819fa919f24cd5a8567a42b8e4cf9695592d25f5645", "ort"),
    source("ort_cpu_kernel", "onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc", "8d3494b5d9344d49d97fae6a0ca2ed41cb97be2a5bfa26d598c34c96de7321b8", "ort"),
    source("ort_cpu_tests", "onnxruntime/test/providers/cpu/nn/tfidfvectorizer_test.cc", "21e16e4382809769a9cd8857b9a57c16a1e37348248378fc5f8cb28c5ae46687", "ort"),
  ]),
});

export function canInferOnnxTfIdfVectorizer(node) {
  return normalizeDomain(node?.domain) === "ai.onnx" && node?.opType === "TfIdfVectorizer";
}

export function inferOnnxTfIdfVectorizerNode({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  const input = tensorMap.get(node.inputs?.[0]);
  const mode = scalarString(node, "mode");
  const minimum = scalarInteger(node, "min_gram_length");
  const maximum = scalarInteger(node, "max_gram_length");
  const maximumSkip = scalarInteger(node, "max_skip_count");
  const counts = integerList(node, "ngram_counts");
  const indexes = integerList(node, "ngram_indexes");
  const stringPoolAttribute = node.attributes?.get("pool_strings");
  const integerPoolAttribute = node.attributes?.get("pool_int64s");
  const hasStringPool = Boolean(stringPoolAttribute);
  const hasIntegerPool = Boolean(integerPoolAttribute);
  const poolKind = hasStringPool ? "string" : hasIntegerPool ? "int64" : "missing";
  const pool = hasStringPool
    ? exactStringList(stringPoolAttribute)
    : hasIntegerPool ? exactIntegerTokens(integerPoolAttribute) : invalidList("pool_attribute_missing");
  const weightsAttribute = node.attributes?.get("weights");
  const weights = weightsAttribute ? exactFloat32List(weightsAttribute) : { ok: true, values: [], texts: [], reason: "" };
  const inputShape = concreteShape(input);
  const inputDtype = String(input?.dtype || "UNKNOWN");
  const reasons = [];

  if (!Number.isSafeInteger(importedOpset) || importedOpset < 9) reasons.push("tfidf_opset_9_not_imported");
  if (!inputShape || ![1, 2].includes(inputShape.length)) reasons.push("tfidf_input_rank_must_be_one_or_two");
  if (!["STRING", "INT32", "INT64"].includes(inputDtype)) reasons.push("tfidf_input_dtype_not_supported");
  if (!mode.ok || !["TF", "IDF", "TFIDF"].includes(mode.value)) reasons.push("tfidf_mode_invalid");
  if (!minimum.ok || minimum.value <= 0) reasons.push("tfidf_min_gram_length_not_positive");
  if (!maximum.ok || !minimum.ok || maximum.value < minimum.value) reasons.push("tfidf_max_gram_length_below_minimum");
  if (!maximumSkip.ok || maximumSkip.value < 0) reasons.push("tfidf_max_skip_count_negative_or_unsafe");
  if (!counts.ok || !counts.values.length) reasons.push(counts.reason || "tfidf_ngram_counts_empty_or_unsafe");
  if (!indexes.ok || !indexes.values.length || indexes.values.some((value) => value < 0)) reasons.push(indexes.reason || "tfidf_ngram_indexes_empty_negative_or_unsafe");
  if (hasStringPool === hasIntegerPool) reasons.push(hasStringPool ? "tfidf_both_pool_attributes_present" : "tfidf_pool_attribute_missing");
  if (!pool.ok || !pool.values.length) reasons.push(pool.reason || "tfidf_pool_empty_or_unsafe");
  if (poolKind === "string" && inputDtype !== "STRING") reasons.push("tfidf_string_pool_input_dtype_mismatch");
  if (poolKind === "int64" && !["INT32", "INT64"].includes(inputDtype)) reasons.push("tfidf_integer_pool_input_dtype_mismatch");
  if (!weights.ok) reasons.push(weights.reason || "tfidf_weights_invalid");
  if (weightsAttribute && weights.ok && indexes.ok && weights.values.length !== indexes.values.length) reasons.push("tfidf_weights_ngram_indexes_length_mismatch");
  if (inputShape?.length === 2 && inputShape[0] === 0) reasons.push("tfidf_pinned_ort_rejects_zero_batch");
  if (minimum.ok && counts.ok && minimum.value > counts.values.length) reasons.push("tfidf_min_gram_length_outside_ngram_counts");
  if (maximum.ok && counts.ok && maximum.value > counts.values.length) reasons.push("tfidf_max_gram_length_outside_ngram_counts");

  const outputWidth = indexes.ok && indexes.values.length
    ? safeIncrement(indexes.values.reduce((maximumValue, value) => Math.max(maximumValue, value), 0)) : null;
  if (outputWidth == null) reasons.push("tfidf_output_width_unsafe");

  const parsed = pool.ok && counts.ok && minimum.ok && maximum.ok
    ? parseDefinitions(pool.values, counts.values, minimum.value, maximum.value)
    : emptyDefinitions();
  reasons.push(...parsed.failure_reasons);
  if (indexes.ok && parsed.total_definition_count !== indexes.values.length) reasons.push("tfidf_definition_index_cardinality_mismatch");
  if (indexes.ok && weightsAttribute && weights.ok) {
    const outOfBounds = indexes.values.filter((index) => index >= weights.values.length).length;
    if (outOfBounds) reasons.push("tfidf_pinned_ort_weight_coordinate_out_of_bounds");
  }

  const uniqueReasons = [...new Set(reasons)];
  const outputShape = outputWidth == null || !inputShape ? null
    : inputShape.length === 1 ? [outputWidth] : [inputShape[0], outputWidth];
  const inputValues = exactInputTokens(input, inputDtype, inputShape);
  const workUnits = estimateWork(inputShape, maximumSkip.value, maximum.value);
  let execution = notExecuted("not_assessed_input_values_unavailable");
  if (!uniqueReasons.length && inputValues.ok && outputShape) {
    if (elementCount(outputShape) > MAX_STATIC_OUTPUT_ELEMENTS) {
      execution = notExecuted("not_assessed_output_element_limit");
    } else if (workUnits == null || workUnits > MAX_STATIC_WORK_UNITS) {
      execution = notExecuted("not_assessed_work_limit");
    } else {
      execution = executePinnedOrt({
        rows: rowTokens(inputValues.values, inputShape), definitions: parsed.relevant_definitions,
        indexes: indexes.values, weights: weights.values, weightsPresent: Boolean(weightsAttribute),
        mode: mode.value, maximumSkip: maximumSkip.value, maximumGram: maximum.value, outputWidth,
      });
    }
  }

  const mapping = assessWeightMapping(indexes.values, weights.values, Boolean(weightsAttribute), mode.value);
  const riskCodes = [...uniqueReasons];
  if (parsed.unused_pool_prefix_item_count) riskCodes.push("tfidf_ngram_counts_ignore_pool_prefix");
  if (parsed.irrelevant_duplicate_ngram_count) riskCodes.push("tfidf_duplicate_ngram_outside_active_length_range");
  if (indexes.ok && duplicateCount(indexes.values)) riskCodes.push("tfidf_multiple_ngrams_share_output_coordinate");
  if (mapping.coordinate_disagreement_count) riskCodes.push("tfidf_weight_coordinate_semantics_divergence");
  if (execution.reference_divergent_output_count) riskCodes.push("tfidf_ort_repeated_addition_differs_from_onnx_reference");
  if (["not_assessed_output_element_limit", "not_assessed_work_limit"].includes(execution.status)) riskCodes.push(`tfidf_${execution.status}`);

  const status = uniqueReasons.length ? "fail" : execution.status === "assessed_exact" ? "pass" : "partial";
  const result = !uniqueReasons.length && outputShape ? {
    outputs: [[node.outputs?.[0], {
      dtype: "FLOAT32", shape: outputShape, shapeDeclared: true,
      ...(execution.status === "assessed_exact" ? staticValuePatch(execution.runtime_output_values) : {}),
    }]],
    reason: status === "pass" ? "" : execution.status,
  } : { outputs: [], reason: uniqueReasons[0] || "tfidf_output_shape_unresolved" };

  return {
    status,
    reason: uniqueReasons[0] || (status === "partial" ? execution.status : ""),
    result,
    row: {
      schema: "deepbom.onnx_tfidf_vectorizer_node.v1",
      evidence_class: "SOURCE_PINNED_AND_DERIVED",
      scope,
      node_index: nodeIndex,
      node_name: node.name || "",
      op_name: "TfIdfVectorizer",
      imported_opset: importedOpset,
      resolved_schema_version: 9,
      status,
      reason_codes: uniqueReasons,
      risk_codes: [...new Set(riskCodes)],
      input_name: node.inputs?.[0] || "",
      input_dtype: inputDtype,
      input_shape: inputShape || [],
      output_name: node.outputs?.[0] || "",
      output_dtype: "FLOAT32",
      exact_output_shape: outputShape || [],
      exact_output_width: outputWidth,
      mode: mode.ok ? mode.value : "",
      minimum_gram_length: minimum.ok ? minimum.value : null,
      maximum_gram_length: maximum.ok ? maximum.value : null,
      maximum_skip_count: maximumSkip.ok ? maximumSkip.value : null,
      pool_kind: poolKind,
      exact_pool_item_count: pool.ok ? pool.values.length : null,
      exact_ngram_level_count: counts.ok ? counts.values.length : null,
      exact_ngram_definition_count: parsed.total_definition_count,
      exact_active_ngram_definition_count: parsed.relevant_definitions.length,
      exact_unused_pool_prefix_item_count: parsed.unused_pool_prefix_item_count,
      exact_duplicate_active_ngram_count: parsed.active_duplicate_ngram_count,
      exact_duplicate_inactive_ngram_count: parsed.irrelevant_duplicate_ngram_count,
      exact_ngram_index_count: indexes.ok ? indexes.values.length : null,
      exact_duplicate_output_coordinate_count: indexes.ok ? duplicateCount(indexes.values) : null,
      exact_unaddressed_output_coordinate_count: outputWidth == null || !indexes.ok ? null : outputWidth - new Set(indexes.values).size,
      weights_present: Boolean(weightsAttribute),
      exact_weight_count: weights.ok ? weights.values.length : null,
      exact_weight_coordinate_disagreement_count: mapping.coordinate_disagreement_count,
      exact_weight_coordinate_value_disagreement_count: mapping.value_disagreement_count,
      weight_application_contract: "ONNX prose associates weights[i] with pool n-gram i; pinned ONNX reference and ORT CPU apply weights[ngram_indexes[i]] at the output coordinate.",
      static_input_status: inputValues.status,
      exact_static_input_value_count: inputValues.ok ? inputValues.values.length : null,
      static_execution_status: execution.status,
      exact_static_work_units: workUnits,
      exact_match_count: execution.match_count,
      exact_nonzero_frequency_count: execution.nonzero_frequency_count,
      exact_frequency_values: execution.frequency_values,
      exact_output_value_count: execution.output_values?.length ?? null,
      exact_nonzero_output_count: execution.nonzero_output_count,
      exact_negative_zero_output_count: execution.negative_zero_output_count,
      exact_output_values: execution.output_values,
      exact_output_negative_zero_indices: execution.output_negative_zero_indices,
      onnx_reference_output_values: execution.reference_output_values,
      exact_ort_reference_divergent_output_count: execution.reference_divergent_output_count,
      exact_ort_reference_divergent_output_indices: execution.reference_divergent_output_indices,
      static_limits: { maximum_output_elements: MAX_STATIC_OUTPUT_ELEMENTS, maximum_work_units: MAX_STATIC_WORK_UNITS },
      method: "Validate TfIdfVectorizer-9 attributes and pool/index cardinality, build the pinned trie, enumerate every start and skip distance up to max_skip_count, and reproduce ORT CPU FLOAT32 assignment or repeated-addition order for complete static inputs.",
      interpretation_boundary: "Exact for artifact-known inputs under the emitted bounds. It is not token-distribution frequency, selected-EP evidence, an optimized-graph trace, or proof that a reduced ORT Web/WASM build includes the CPU text kernel.",
    },
  };
}

function executePinnedOrt({ rows, definitions, indexes, weights, weightsPresent, mode, maximumSkip, maximumGram, outputWidth }) {
  const root = trie(definitions);
  const frequencies = new Array(rows.length * outputWidth).fill(0);
  const definitionFrequencies = new Array(rows.length * indexes.length).fill(0);
  const output = new Array(rows.length * outputWidth).fill(0);
  let matches = 0;
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    let minimum = definitions.reduce((value, item) => Math.min(value, item.size), Number.POSITIVE_INFINITY);
    if (!Number.isFinite(minimum)) minimum = maximumGram + 1;
    for (let distance = 1; distance <= maximumSkip + 1; distance += 1) {
      for (let start = 0; start < row.length; start += 1) {
        if (start + distance * (minimum - 1) >= row.length) break;
        let branch = root;
        for (let size = 1, position = start; size <= maximumGram && position < row.length; size += 1, position += distance) {
          branch = branch.children.get(row[position]);
          if (!branch) break;
          if (size >= minimum && branch.definition_index != null) {
            const definitionIndex = branch.definition_index;
            const coordinate = indexes[definitionIndex];
            const flat = rowIndex * outputWidth + coordinate;
            frequencies[flat] += 1;
            definitionFrequencies[rowIndex * indexes.length + definitionIndex] += 1;
            matches += 1;
            if (mode === "TF") output[flat] = Math.fround(output[flat] + 1);
            else if (mode === "IDF") output[flat] = weightsPresent ? weights[coordinate] : 1;
            else output[flat] = Math.fround(output[flat] + (weightsPresent ? weights[coordinate] : 1));
          }
        }
      }
      if (minimum === 1) {
        minimum = 2;
        if (minimum > maximumGram) break;
      }
    }
  }
  const reference = frequencies.map((frequency, index) => {
    const coordinate = index % outputWidth;
    if (mode === "TF") return Math.fround(frequency);
    if (mode === "IDF") return frequency > 0 ? (weightsPresent ? weights[coordinate] : 1) : 0;
    return Math.fround(frequency * (weightsPresent ? weights[coordinate] : 1));
  });
  const divergent = output.flatMap((value, index) => Object.is(value, reference[index]) ? [] : [index]);
  const negativeZeros = output.flatMap((value, index) => Object.is(value, -0) ? [index] : []);
  return {
    status: "assessed_exact",
    match_count: matches,
    nonzero_frequency_count: frequencies.filter((value) => value !== 0).length,
    frequency_values: frequencies,
    definition_frequency_values: definitionFrequencies,
    runtime_output_values: output,
    output_values: output.map((value) => Object.is(value, -0) ? 0 : value),
    output_negative_zero_indices: negativeZeros,
    negative_zero_output_count: negativeZeros.length,
    nonzero_output_count: output.filter((value) => value !== 0).length,
    reference_output_values: reference.map((value) => Object.is(value, -0) ? 0 : value),
    reference_divergent_output_count: divergent.length,
    reference_divergent_output_indices: divergent,
  };
}

function parseDefinitions(pool, counts, minimum, maximum) {
  const definitions = [];
  const relevant = [];
  const failures = [];
  const activeKeys = new Set();
  const inactiveKeys = new Set();
  let activeDuplicates = 0;
  let inactiveDuplicates = 0;
  for (let level = 0; level < counts.length; level += 1) {
    const size = level + 1;
    const start = counts[level];
    const end = level + 1 < counts.length ? counts[level + 1] : pool.length;
    if (![start, end].every(Number.isSafeInteger) || start < 0 || end < start || end > pool.length) {
      failures.push("tfidf_ngram_counts_out_of_pool_bounds");
      continue;
    }
    const items = end - start;
    if (items % size !== 0) {
      failures.push("tfidf_pool_level_does_not_contain_whole_ngrams");
      continue;
    }
    for (let offset = start; offset < end; offset += size) {
      const tokens = pool.slice(offset, offset + size);
      const definition = { definition_index: definitions.length, size, tokens };
      definitions.push(definition);
      const active = size >= minimum && size <= maximum;
      const key = tokens.map((token) => `${String(token).length}:${token}`).join("|");
      const keys = active ? activeKeys : inactiveKeys;
      if (keys.has(key)) {
        if (active) activeDuplicates += 1;
        else inactiveDuplicates += 1;
      } else keys.add(key);
      if (active) relevant.push(definition);
    }
  }
  if (activeDuplicates) failures.push("tfidf_duplicate_active_ngram_rejected_by_pinned_ort");
  return {
    total_definition_count: definitions.length,
    relevant_definitions: relevant,
    failure_reasons: [...new Set(failures)],
    unused_pool_prefix_item_count: counts.length && counts[0] > 0 ? counts[0] : 0,
    active_duplicate_ngram_count: activeDuplicates,
    irrelevant_duplicate_ngram_count: inactiveDuplicates,
  };
}

function trie(definitions) {
  const root = { children: new Map(), definition_index: null };
  for (const definition of definitions) {
    let node = root;
    for (const token of definition.tokens) {
      if (!node.children.has(token)) node.children.set(token, { children: new Map(), definition_index: null });
      node = node.children.get(token);
    }
    node.definition_index = definition.definition_index;
  }
  return root;
}

function exactInputTokens(tensor, dtype, shape) {
  if (!shape) return { ok: false, values: [], status: "not_assessed_input_shape_unknown" };
  const expected = elementCount(shape);
  if (dtype === "STRING" && tensor?.staticValuesComplete === true && Array.isArray(tensor.staticValues)
    && tensor.staticValues.length === expected && tensor.staticValues.every((value) => typeof value === "string")) {
    return { ok: true, values: [...tensor.staticValues], status: "assessed_exact_string_values" };
  }
  if (dtype === "INT64" && tensor?.initializerIntegerValuesExactComplete === true
    && Array.isArray(tensor.initializerIntegerValuesExactDecimals) && tensor.initializerIntegerValuesExactDecimals.length === expected) {
    return { ok: true, values: [...tensor.initializerIntegerValuesExactDecimals], status: "assessed_exact_int64_decimals" };
  }
  if (["INT32", "INT64"].includes(dtype) && tensor?.staticValuesComplete === true
    && Array.isArray(tensor.staticValues) && tensor.staticValues.length === expected && tensor.staticValues.every(Number.isSafeInteger)) {
    return { ok: true, values: tensor.staticValues.map(String), status: `assessed_exact_${dtype.toLowerCase()}_values` };
  }
  return { ok: false, values: [], status: "not_assessed_input_values_unavailable" };
}

function exactIntegerTokens(attribute) {
  const exact = attribute?.intExactDecimals || [];
  if (!exact.length || exact.length !== (attribute?.ints || []).length) return invalidList("tfidf_integer_pool_exact_values_unavailable");
  try {
    exact.forEach((value) => BigInt(value));
    return { ok: true, values: [...exact], reason: "" };
  } catch {
    return invalidList("tfidf_integer_pool_exact_values_invalid");
  }
}

function integerList(node, name) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return invalidList(`tfidf_${name}_missing`);
  const values = attribute.ints || [];
  return values.every(Number.isSafeInteger)
    ? { ok: true, values: [...values], reason: "" }
    : invalidList(`tfidf_${name}_outside_javascript_safe_integer`);
}

function exactStringList(attribute) {
  return Array.isArray(attribute?.strings) ? { ok: true, values: [...attribute.strings], reason: "" } : invalidList("tfidf_string_pool_invalid");
}

function exactFloat32List(attribute) {
  const values = attribute?.floats || [];
  if (!values.every(Number.isFinite)) return { ok: false, values: [], texts: values.map(String), reason: "tfidf_weights_non_finite" };
  return { ok: true, values: values.map(Math.fround), texts: values.map(String), reason: "" };
}

function scalarInteger(node, name) {
  const attribute = node.attributes?.get(name);
  return Number.isSafeInteger(attribute?.i)
    ? { ok: true, value: attribute.i }
    : { ok: false, value: null };
}

function scalarString(node, name) {
  const attribute = node.attributes?.get(name);
  return typeof attribute?.s === "string" ? { ok: true, value: attribute.s } : { ok: false, value: "" };
}

function rowTokens(values, shape) {
  return shape.length === 1 ? [values] : Array.from({ length: shape[0] }, (_, row) => values.slice(row * shape[1], (row + 1) * shape[1]));
}

function assessWeightMapping(indexes, weights, present, mode) {
  if (!present || mode === "TF" || !indexes.length || !weights.length) return { coordinate_disagreement_count: 0, value_disagreement_count: 0 };
  let coordinates = 0;
  let values = 0;
  indexes.forEach((coordinate, definition) => {
    if (coordinate !== definition) coordinates += 1;
    if (coordinate < weights.length && !Object.is(weights[coordinate], weights[definition])) values += 1;
  });
  return { coordinate_disagreement_count: coordinates, value_disagreement_count: values };
}

function estimateWork(shape, maximumSkip, maximumGram) {
  if (!shape || ![maximumSkip, maximumGram].every((value) => Number.isSafeInteger(value) && value >= 0)) return null;
  const rows = shape.length === 1 ? 1 : shape[0];
  const columns = shape.at(-1);
  const value = rows * columns * (maximumSkip + 1) * maximumGram;
  return Number.isSafeInteger(value) ? value : null;
}

function staticValuePatch(values) {
  return { staticValuesStatus: "assessed_exact_static_data", staticValuesComplete: true, staticValues: [...values], staticValuesSource: "TfIdfVectorizer-9 pinned ORT CPU order" };
}

function notExecuted(status) {
  return {
    status, match_count: null, nonzero_frequency_count: null, frequency_values: [], runtime_output_values: null, output_values: null,
    output_negative_zero_indices: [], negative_zero_output_count: null, nonzero_output_count: null,
    reference_output_values: null, reference_divergent_output_count: null, reference_divergent_output_indices: [],
  };
}

function emptyDefinitions() {
  return { total_definition_count: 0, relevant_definitions: [], failure_reasons: [], unused_pool_prefix_item_count: 0, active_duplicate_ngram_count: 0, irrelevant_duplicate_ngram_count: 0 };
}

function source(role, path, sha256, repo) {
  const root = repo === "onnx" ? `https://raw.githubusercontent.com/onnx/onnx/${ONNX_COMMIT}` : `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}`;
  return Object.freeze({ role, source_ref: `${root}/${path}`, sha256 });
}

function invalidList(reason) { return { ok: false, values: [], reason }; }
function normalizeDomain(domain) { return !domain || domain === "ai.onnx" ? "ai.onnx" : String(domain); }
function concreteShape(tensor) { return tensor?.shapeDeclared === true && Array.isArray(tensor.shape) && tensor.shape.every((dim) => Number.isSafeInteger(dim) && dim >= 0) ? [...tensor.shape] : null; }
function elementCount(shape) { return shape.reduce((product, dimension) => product * dimension, 1); }
function safeIncrement(value) { return Number.isSafeInteger(value) && value >= 0 && value < Number.MAX_SAFE_INTEGER ? value + 1 : null; }
function duplicateCount(values) { return values.length - new Set(values).size; }
