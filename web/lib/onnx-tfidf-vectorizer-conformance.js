const MAX_OUTPUT_ELEMENTS = 65_536;
const MAX_WORK_UNITS = 5_000_000;

export function validateTfIdfVectorizerRowAgainstEvidence(row, tensors = [], ops = []) {
  if (!row || row.op_name !== "TfIdfVectorizer") return false;
  const op = ops.find((candidate) => candidate.index === row.node_index
    && candidate.name === "TfIdfVectorizer" && candidate.domain === "ai.onnx");
  if (!op) return false;
  const facts = reconstruct(op, tensors);
  const output = tensors.find((tensor) => tensor.name === op.output_names?.[0]);
  return row.schema === "deepbom.onnx_tfidf_vectorizer_node.v1"
    && row.evidence_class === "SOURCE_PINNED_AND_DERIVED"
    && row.status === facts.status
    && sameSet(row.reason_codes, facts.reasons)
    && sameSet(row.risk_codes, facts.risks)
    && row.node_name === (op.graph_node_name || "")
    && row.imported_opset === 9
    && row.resolved_schema_version === 9
    && row.input_name === facts.inputName
    && row.input_dtype === facts.inputDtype
    && same(row.input_shape, facts.inputShape)
    && row.output_name === facts.outputName
    && row.output_dtype === "FLOAT32"
    && same(row.exact_output_shape, facts.outputShape)
    && row.exact_output_width === facts.outputWidth
    && row.mode === facts.mode
    && row.minimum_gram_length === facts.minimum
    && row.maximum_gram_length === facts.maximum
    && row.maximum_skip_count === facts.maximumSkip
    && row.pool_kind === facts.poolKind
    && row.exact_pool_item_count === facts.poolItemCount
    && row.exact_ngram_level_count === facts.levelCount
    && row.exact_ngram_definition_count === facts.definitionCount
    && row.exact_active_ngram_definition_count === facts.activeDefinitionCount
    && row.exact_unused_pool_prefix_item_count === facts.unusedPoolPrefix
    && row.exact_duplicate_active_ngram_count === facts.activeDuplicates
    && row.exact_duplicate_inactive_ngram_count === facts.inactiveDuplicates
    && row.exact_ngram_index_count === facts.indexCount
    && row.exact_duplicate_output_coordinate_count === facts.duplicateCoordinates
    && row.exact_unaddressed_output_coordinate_count === facts.unaddressedCoordinates
    && row.weights_present === facts.weightsPresent
    && row.exact_weight_count === facts.weightCount
    && row.exact_weight_coordinate_disagreement_count === facts.coordinateDisagreements
    && row.exact_weight_coordinate_value_disagreement_count === facts.weightValueDisagreements
    && row.static_input_status === facts.staticInputStatus
    && row.exact_static_input_value_count === facts.staticInputCount
    && row.static_execution_status === facts.executionStatus
    && row.exact_static_work_units === facts.workUnits
    && row.exact_match_count === facts.matchCount
    && row.exact_nonzero_frequency_count === facts.nonzeroFrequencyCount
    && same(row.exact_frequency_values, facts.frequencies)
    && row.exact_output_value_count === facts.outputValueCount
    && row.exact_nonzero_output_count === facts.nonzeroOutputCount
    && row.exact_negative_zero_output_count === facts.negativeZeroCount
    && same(row.exact_output_values, normalized(facts.runtimeOutput))
    && same(row.exact_output_negative_zero_indices, facts.negativeZeroIndices)
    && same(row.onnx_reference_output_values, normalized(facts.referenceOutput))
    && row.exact_ort_reference_divergent_output_count === facts.referenceDivergenceCount
    && same(row.exact_ort_reference_divergent_output_indices, facts.referenceDivergentIndices)
    && row.static_limits?.maximum_output_elements === MAX_OUTPUT_ELEMENTS
    && row.static_limits?.maximum_work_units === MAX_WORK_UNITS
    && outputEvidenceConserves(output, facts);
}

export function reconstructTfIdfVectorizerEvidence(op, tensors = []) {
  return reconstruct(op, tensors);
}

function reconstruct(op, tensors) {
  const attributes = new Map((op.onnx_attributes || []).map((attribute) => [attribute.name, attribute]));
  const input = tensors.find((tensor) => tensor.name === op.input_names?.[0]);
  const inputShape = concreteShape(input);
  const inputDtype = String(input?.dtype || "UNKNOWN");
  const mode = stringScalar(attributes, "mode");
  const minimum = integerScalar(attributes, "min_gram_length");
  const maximum = integerScalar(attributes, "max_gram_length");
  const maximumSkip = integerScalar(attributes, "max_skip_count");
  const counts = safeIntegerList(attributes, "ngram_counts");
  const indexes = safeIntegerList(attributes, "ngram_indexes");
  const hasStrings = attributes.has("pool_strings");
  const hasIntegers = attributes.has("pool_int64s");
  const poolKind = hasStrings ? "string" : hasIntegers ? "int64" : "missing";
  const pool = hasStrings ? stringList(attributes.get("pool_strings"))
    : hasIntegers ? exactIntegerTokens(attributes.get("pool_int64s")) : invalid("tfidf_pool_attribute_missing");
  const weightsPresent = attributes.has("weights");
  const weights = weightsPresent ? float32List(attributes.get("weights")) : valid([]);
  const reasons = [];
  const importedOpset = Number(op.imported_opset ?? op.opset ?? 9);

  if (!Number.isSafeInteger(importedOpset) || importedOpset < 9) reasons.push("tfidf_opset_9_not_imported");
  if (!inputShape || ![1, 2].includes(inputShape.length)) reasons.push("tfidf_input_rank_must_be_one_or_two");
  if (!["STRING", "INT32", "INT64"].includes(inputDtype)) reasons.push("tfidf_input_dtype_not_supported");
  if (!mode.ok || !["TF", "IDF", "TFIDF"].includes(mode.value)) reasons.push("tfidf_mode_invalid");
  if (!minimum.ok || minimum.value <= 0) reasons.push("tfidf_min_gram_length_not_positive");
  if (!maximum.ok || !minimum.ok || maximum.value < minimum.value) reasons.push("tfidf_max_gram_length_below_minimum");
  if (!maximumSkip.ok || maximumSkip.value < 0) reasons.push("tfidf_max_skip_count_negative_or_unsafe");
  if (!counts.ok || !counts.values.length) reasons.push(counts.reason || "tfidf_ngram_counts_empty_or_unsafe");
  if (!indexes.ok || !indexes.values.length || indexes.values.some((value) => value < 0)) reasons.push(indexes.reason || "tfidf_ngram_indexes_empty_negative_or_unsafe");
  if (hasStrings === hasIntegers) reasons.push(hasStrings ? "tfidf_both_pool_attributes_present" : "tfidf_pool_attribute_missing");
  if (!pool.ok || !pool.values.length) reasons.push(pool.reason || "tfidf_pool_empty_or_unsafe");
  if (poolKind === "string" && inputDtype !== "STRING") reasons.push("tfidf_string_pool_input_dtype_mismatch");
  if (poolKind === "int64" && !["INT32", "INT64"].includes(inputDtype)) reasons.push("tfidf_integer_pool_input_dtype_mismatch");
  if (!weights.ok) reasons.push(weights.reason || "tfidf_weights_invalid");
  if (weightsPresent && weights.ok && indexes.ok && weights.values.length !== indexes.values.length) reasons.push("tfidf_weights_ngram_indexes_length_mismatch");
  if (inputShape?.length === 2 && inputShape[0] === 0) reasons.push("tfidf_pinned_ort_rejects_zero_batch");
  if (minimum.ok && counts.ok && minimum.value > counts.values.length) reasons.push("tfidf_min_gram_length_outside_ngram_counts");
  if (maximum.ok && counts.ok && maximum.value > counts.values.length) reasons.push("tfidf_max_gram_length_outside_ngram_counts");

  const outputWidth = indexes.ok && indexes.values.length
    ? safeAddOne(indexes.values.reduce((maximumValue, value) => Math.max(maximumValue, value), 0)) : null;
  if (outputWidth == null) reasons.push("tfidf_output_width_unsafe");
  const definitions = pool.ok && counts.ok && minimum.ok && maximum.ok
    ? definitionLedger(pool.values, counts.values, minimum.value, maximum.value) : emptyDefinitions();
  reasons.push(...definitions.failures);
  if (indexes.ok && definitions.all.length !== indexes.values.length) reasons.push("tfidf_definition_index_cardinality_mismatch");
  if (indexes.ok && weightsPresent && weights.ok
    && indexes.values.some((coordinate) => coordinate >= weights.values.length)) reasons.push("tfidf_pinned_ort_weight_coordinate_out_of_bounds");
  const uniqueReasons = unique(reasons);
  const outputShape = outputWidth == null || !inputShape ? []
    : inputShape.length === 1 ? [outputWidth] : [inputShape[0], outputWidth];
  const inputValues = staticTokens(input, inputDtype, inputShape);
  const workUnits = estimatedWork(inputShape, maximumSkip.value, maximum.value);
  let execution = notExecuted("not_assessed_input_values_unavailable");
  if (!uniqueReasons.length && inputValues.ok && outputShape.length) {
    if (product(outputShape) > MAX_OUTPUT_ELEMENTS) execution = notExecuted("not_assessed_output_element_limit");
    else if (workUnits == null || workUnits > MAX_WORK_UNITS) execution = notExecuted("not_assessed_work_limit");
    else execution = execute({
      rows: splitRows(inputValues.values, inputShape), definitions: definitions.active,
      indexes: indexes.values, weights: weights.values, weightsPresent,
      mode: mode.value, maximumSkip: maximumSkip.value, maximumGram: maximum.value, outputWidth,
    });
  }
  const mapping = weightMapping(indexes.values, weights.values, weightsPresent, mode.value);
  const risks = [...uniqueReasons];
  if (definitions.unusedPrefix) risks.push("tfidf_ngram_counts_ignore_pool_prefix");
  if (definitions.inactiveDuplicates) risks.push("tfidf_duplicate_ngram_outside_active_length_range");
  const duplicateCoordinates = indexes.ok ? indexes.values.length - new Set(indexes.values).size : null;
  if (duplicateCoordinates) risks.push("tfidf_multiple_ngrams_share_output_coordinate");
  if (mapping.coordinateDisagreements) risks.push("tfidf_weight_coordinate_semantics_divergence");
  if (execution.referenceDivergentIndices.length) risks.push("tfidf_ort_repeated_addition_differs_from_onnx_reference");
  if (["not_assessed_output_element_limit", "not_assessed_work_limit"].includes(execution.status)) risks.push(`tfidf_${execution.status}`);
  const status = uniqueReasons.length ? "fail" : execution.status === "assessed_exact" ? "pass" : "partial";

  return {
    status, reason: uniqueReasons[0] || (status === "partial" ? execution.status : ""), reasons: uniqueReasons, risks: unique(risks),
    inputName: op.input_names?.[0] || "", inputDtype, inputShape: inputShape || [], outputName: op.output_names?.[0] || "",
    outputShape, outputWidth, mode: mode.ok ? mode.value : "", minimum: minimum.ok ? minimum.value : null,
    maximum: maximum.ok ? maximum.value : null, maximumSkip: maximumSkip.ok ? maximumSkip.value : null, poolKind,
    poolItemCount: pool.ok ? pool.values.length : null, levelCount: counts.ok ? counts.values.length : null,
    definitionCount: definitions.all.length, activeDefinitionCount: definitions.active.length,
    unusedPoolPrefix: definitions.unusedPrefix, activeDuplicates: definitions.activeDuplicates,
    inactiveDuplicates: definitions.inactiveDuplicates, indexCount: indexes.ok ? indexes.values.length : null,
    duplicateCoordinates, unaddressedCoordinates: outputWidth == null || !indexes.ok ? null : outputWidth - new Set(indexes.values).size,
    weightsPresent, weightCount: weights.ok ? weights.values.length : null,
    coordinateDisagreements: mapping.coordinateDisagreements, weightValueDisagreements: mapping.valueDisagreements,
    staticInputStatus: inputValues.status, staticInputCount: inputValues.ok ? inputValues.values.length : null,
    executionStatus: execution.status, workUnits, matchCount: execution.matchCount,
    nonzeroFrequencyCount: execution.nonzeroFrequencyCount, frequencies: execution.frequencies,
    outputValueCount: execution.runtimeOutput?.length ?? null, nonzeroOutputCount: execution.nonzeroOutputCount,
    runtimeOutput: execution.runtimeOutput, negativeZeroIndices: execution.negativeZeroIndices,
    negativeZeroCount: execution.negativeZeroCount, referenceOutput: execution.referenceOutput,
    referenceDivergentIndices: execution.referenceDivergentIndices, referenceDivergenceCount: execution.referenceDivergenceCount,
  };
}

function execute({ rows, definitions, indexes, weights, weightsPresent, mode, maximumSkip, maximumGram, outputWidth }) {
  const root = buildTrie(definitions);
  const frequencies = new Array(rows.length * outputWidth).fill(0);
  const output = new Array(rows.length * outputWidth).fill(0);
  let matchCount = 0;
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const tokens = rows[rowIndex];
    let minimum = definitions.reduce((value, definition) => Math.min(value, definition.size), Number.POSITIVE_INFINITY);
    if (!Number.isFinite(minimum)) minimum = maximumGram + 1;
    for (let distance = 1; distance <= maximumSkip + 1; distance += 1) {
      for (let start = 0; start < tokens.length; start += 1) {
        if (start + distance * (minimum - 1) >= tokens.length) break;
        let branch = root;
        for (let size = 1, position = start; size <= maximumGram && position < tokens.length; size += 1, position += distance) {
          branch = branch.children.get(tokens[position]);
          if (!branch) break;
          if (size >= minimum && branch.definitionIndex != null) {
            const coordinate = indexes[branch.definitionIndex];
            const flat = rowIndex * outputWidth + coordinate;
            frequencies[flat] += 1;
            matchCount += 1;
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
  const referenceOutput = frequencies.map((frequency, index) => {
    const coordinate = index % outputWidth;
    if (mode === "TF") return Math.fround(frequency);
    if (mode === "IDF") return frequency > 0 ? (weightsPresent ? weights[coordinate] : 1) : 0;
    return Math.fround(frequency * (weightsPresent ? weights[coordinate] : 1));
  });
  const negativeZeroIndices = indices(output, (value) => Object.is(value, -0));
  const referenceDivergentIndices = indices(output, (value, index) => !Object.is(value, referenceOutput[index]));
  return {
    status: "assessed_exact", matchCount, frequencies,
    nonzeroFrequencyCount: frequencies.filter((value) => value !== 0).length,
    runtimeOutput: output, nonzeroOutputCount: output.filter((value) => value !== 0).length,
    negativeZeroIndices, negativeZeroCount: negativeZeroIndices.length,
    referenceOutput,
    referenceDivergentIndices, referenceDivergenceCount: referenceDivergentIndices.length,
  };
}

function definitionLedger(pool, counts, minimum, maximum) {
  const all = [];
  const active = [];
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
    if ((end - start) % size !== 0) {
      failures.push("tfidf_pool_level_does_not_contain_whole_ngrams");
      continue;
    }
    for (let offset = start; offset < end; offset += size) {
      const tokens = pool.slice(offset, offset + size);
      const definition = { definitionIndex: all.length, size, tokens };
      all.push(definition);
      const isActive = size >= minimum && size <= maximum;
      const key = tokens.map((token) => `${String(token).length}:${token}`).join("|");
      const keys = isActive ? activeKeys : inactiveKeys;
      if (keys.has(key)) {
        if (isActive) activeDuplicates += 1; else inactiveDuplicates += 1;
      } else keys.add(key);
      if (isActive) active.push(definition);
    }
  }
  if (activeDuplicates) failures.push("tfidf_duplicate_active_ngram_rejected_by_pinned_ort");
  return { all, active, failures: unique(failures), unusedPrefix: counts[0] > 0 ? counts[0] : 0, activeDuplicates, inactiveDuplicates };
}

function buildTrie(definitions) {
  const root = { children: new Map(), definitionIndex: null };
  for (const definition of definitions) {
    let branch = root;
    for (const token of definition.tokens) {
      if (!branch.children.has(token)) branch.children.set(token, { children: new Map(), definitionIndex: null });
      branch = branch.children.get(token);
    }
    branch.definitionIndex = definition.definitionIndex;
  }
  return root;
}

function staticTokens(tensor, dtype, shape) {
  if (!shape) return { ok: false, values: [], status: "not_assessed_input_shape_unknown" };
  const expected = product(shape);
  if (dtype === "STRING" && tensor?.static_values_complete === true && Array.isArray(tensor.static_values)
    && tensor.static_values.length === expected && tensor.static_values.every((value) => typeof value === "string")) {
    return { ok: true, values: [...tensor.static_values], status: "assessed_exact_string_values" };
  }
  if (dtype === "INT64" && tensor?.initializer_integer_values_exact_complete === true
    && Array.isArray(tensor.initializer_integer_values_exact_decimals)
    && tensor.initializer_integer_values_exact_decimals.length === expected) {
    try {
      tensor.initializer_integer_values_exact_decimals.forEach((value) => BigInt(value));
      return { ok: true, values: [...tensor.initializer_integer_values_exact_decimals], status: "assessed_exact_int64_decimals" };
    } catch { return { ok: false, values: [], status: "not_assessed_input_values_unavailable" }; }
  }
  const values = signedStaticValues(tensor);
  if (["INT32", "INT64"].includes(dtype) && values && values.length === expected && values.every(Number.isSafeInteger)) {
    return { ok: true, values: values.map(String), status: `assessed_exact_${dtype.toLowerCase()}_values` };
  }
  return { ok: false, values: [], status: "not_assessed_input_values_unavailable" };
}

function outputEvidenceConserves(output, facts) {
  if (facts.executionStatus !== "assessed_exact") return output?.static_values_complete !== true;
  const values = signedStaticValues(output);
  return Boolean(values) && sameNumbers(values, facts.runtimeOutput);
}

function safeIntegerList(attributes, name) {
  const attribute = attributes.get(name);
  if (!attribute) return invalid(`tfidf_${name}_missing`);
  const exact = attribute.int_values_exact_decimal || [];
  try {
    const values = exact.map((value) => Number(BigInt(value)));
    return values.every(Number.isSafeInteger) ? valid(values) : invalid(`tfidf_${name}_outside_javascript_safe_integer`);
  } catch { return invalid(`tfidf_${name}_outside_javascript_safe_integer`); }
}

function exactIntegerTokens(attribute) {
  const exact = attribute?.int_values_exact_decimal || [];
  if (!exact.length) return invalid("tfidf_integer_pool_exact_values_unavailable");
  try { exact.forEach((value) => BigInt(value)); return valid([...exact]); }
  catch { return invalid("tfidf_integer_pool_exact_values_invalid"); }
}

function stringList(attribute) {
  return Array.isArray(attribute?.string_values) ? valid([...attribute.string_values]) : invalid("tfidf_string_pool_invalid");
}

function float32List(attribute) {
  const texts = attribute?.float_values_text || [];
  const values = texts.map((value) => Number(value));
  return values.every(Number.isFinite) ? valid(values.map(Math.fround)) : invalid("tfidf_weights_non_finite");
}

function integerScalar(attributes, name) {
  const attribute = attributes.get(name);
  try {
    const value = Number(BigInt(attribute?.int_value_exact_decimal));
    return Number.isSafeInteger(value) ? { ok: true, value } : { ok: false, value: null };
  } catch { return { ok: false, value: null }; }
}

function stringScalar(attributes, name) {
  const value = attributes.get(name)?.string_value;
  return typeof value === "string" ? { ok: true, value } : { ok: false, value: "" };
}

function weightMapping(indexes, weights, present, mode) {
  if (!present || mode === "TF" || !indexes.length || !weights.length) return { coordinateDisagreements: 0, valueDisagreements: 0 };
  let coordinateDisagreements = 0;
  let valueDisagreements = 0;
  indexes.forEach((coordinate, definition) => {
    if (coordinate !== definition) coordinateDisagreements += 1;
    if (coordinate < weights.length && !Object.is(weights[coordinate], weights[definition])) valueDisagreements += 1;
  });
  return { coordinateDisagreements, valueDisagreements };
}

function notExecuted(status) {
  return { status, matchCount: null, frequencies: [], nonzeroFrequencyCount: null, runtimeOutput: null,
    nonzeroOutputCount: null, negativeZeroIndices: [], negativeZeroCount: null, referenceOutput: null,
    referenceDivergentIndices: [], referenceDivergenceCount: null };
}

function concreteShape(tensor) {
  return tensor?.shape_declared === true && Array.isArray(tensor.shape)
    && tensor.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0) ? [...tensor.shape] : null;
}

function signedStaticValues(tensor) {
  if (tensor?.static_values_complete !== true || !Array.isArray(tensor.static_values)) return null;
  const values = [...tensor.static_values];
  for (const index of tensor.static_values_negative_zero_indices || []) if (index >= 0 && index < values.length) values[index] = -0;
  return values;
}

function estimatedWork(shape, maximumSkip, maximumGram) {
  if (!shape || ![maximumSkip, maximumGram].every((value) => Number.isSafeInteger(value) && value >= 0)) return null;
  const value = (shape.length === 1 ? 1 : shape[0]) * shape.at(-1) * (maximumSkip + 1) * maximumGram;
  return Number.isSafeInteger(value) ? value : null;
}

function splitRows(values, shape) {
  return shape.length === 1 ? [values]
    : Array.from({ length: shape[0] }, (_, row) => values.slice(row * shape[1], (row + 1) * shape[1]));
}

function emptyDefinitions() { return { all: [], active: [], failures: [], unusedPrefix: 0, activeDuplicates: 0, inactiveDuplicates: 0 }; }
function valid(values) { return { ok: true, values, reason: "" }; }
function invalid(reason) { return { ok: false, values: [], reason }; }
function safeAddOne(value) { return Number.isSafeInteger(value) && value >= 0 && value < Number.MAX_SAFE_INTEGER ? value + 1 : null; }
function product(values) { return values.reduce((value, item) => value * item, 1); }
function unique(values) { return [...new Set(values)]; }
function normalized(values) { return values == null ? null : values.map((value) => Object.is(value, -0) ? 0 : value); }
function indices(values, predicate) { return values.flatMap((value, index) => predicate(value, index) ? [index] : []); }
function same(left, right) { return JSON.stringify(left) === JSON.stringify(right); }
function sameSet(left = [], right = []) { return same([...left].sort(), [...right].sort()); }
function sameNumbers(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => Object.is(value, right[index])); }
