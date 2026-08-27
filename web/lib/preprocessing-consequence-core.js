import { sha256Hex, sha256TypedArrayListHex } from "./hash.js";
import { buildInputWitnessTensor } from "./input-counterexample.js";
import { buildCandidateRgbFixture } from "./preprocessing-realizability.js";

export const PREPROCESSING_CONSEQUENCE_SCHEMA = "deepbom.preprocessing_consequence_atlas.v1";
export const PREPROCESSING_CONSEQUENCE_METHOD_VERSION = "2026-07-18.1";

const CANDIDATE_PREFIX = "deepbom.preprocessing_consequence_atlas.candidate.v1\0";
const PORTFOLIO_PREFIX = "deepbom.preprocessing_consequence_atlas.portfolio.v1\0";

export async function buildCandidateReplayInput(candidate, witness) {
  const fixture = buildCandidateRgbFixture(candidate, witness);
  const data = allocateInput(witness.model_input_dtype, witness.model_input_element_count);
  for (let linear = 0; linear < data.length; linear += 1) {
    const tensorChannel = linear % 3;
    const pixelBase = linear - tensorChannel;
    const map = candidate.channel_maps[tensorChannel];
    const pixel = fixture.rgb[pixelBase + map.source_pixel_channel];
    data[linear] = map.pixel_to_tensor_codes[pixel];
  }
  return { data, fixture };
}

export function buildCanonicalWitnessInput(witness) {
  const tensor = buildInputWitnessTensor(witness);
  const data = allocateInput(witness.model_input_dtype, tensor.bytes.length);
  new Uint8Array(data.buffer, data.byteOffset, data.byteLength).set(tensor.bytes);
  return data;
}

export async function buildPreprocessingConsequenceEvidence({
  analysis,
  artifactSha256,
  runtime,
  preprocessingValidation,
  baselineCapture,
  candidateCaptures,
  outputDetails,
  otherInputs = [],
}) {
  const preprocessing = preprocessingValidation.evidence;
  const witness = preprocessingValidation.input.evidence.witnesses[preprocessing.candidates[0]?.witness_index ?? 0];
  assert(witness, "Preprocessing consequence witness is unavailable.");
  assert(candidateCaptures.length === preprocessing.candidates.length, "Preprocessing consequence capture matrix is incomplete.");
  const baselineInputSha256 = await rawTypedArraySha256(baselineCapture.input);
  assert(baselineInputSha256 === witness.full_model_input_tensor_sha256, "Preprocessing consequence baseline input SHA-256 mismatch.");
  const baselineOutputSha256 = await sha256TypedArrayListHex(baselineCapture.outputs);
  const baselineOutputs = await summarizeOutputSet(baselineCapture.outputs, baselineCapture.outputs, outputDetails);

  const candidates = [];
  for (let index = 0; index < preprocessing.candidates.length; index += 1) {
    const source = preprocessing.candidates[index];
    const capture = candidateCaptures[index];
    const inputSha256 = await rawTypedArraySha256(capture.input);
    const outputSha256 = await sha256TypedArrayListHex(capture.outputs);
    const inputDifference = summarizeArrayDifference(baselineCapture.input, capture.input);
    const outputDifference = await summarizeOutputSet(baselineCapture.outputs, capture.outputs, outputDetails);
    const row = {
      candidate_index: index,
      witness_index: source.witness_index,
      contract_id: source.contract_id,
      contract_label: source.contract_label,
      source_candidate_ledger_sha256: source.candidate_ledger_sha256,
      source_exact_tensor_realization: source.exact_tensor_realization,
      source_unrealizable_tensor_element_count: source.unrealizable_tensor_element_count,
      input_tensor_sha256: inputSha256,
      input_identical_to_witness: inputSha256 === baselineInputSha256,
      input_changed_element_count: inputDifference.changed_element_count,
      input_total_absolute_code_difference_decimal: inputDifference.total_absolute_difference_decimal,
      input_maximum_absolute_code_difference: inputDifference.maximum_absolute_difference,
      output_tensor_set_sha256: outputSha256,
      output_identical_to_witness_replay: outputSha256 === baselineOutputSha256,
      output_changed_element_count: outputDifference.changed_element_count,
      output_total_element_count: outputDifference.total_element_count,
      output_total_absolute_difference_decimal: outputDifference.total_absolute_difference_decimal,
      output_mean_absolute_difference: outputDifference.mean_absolute_difference,
      output_root_mean_square_difference: outputDifference.root_mean_square_difference,
      output_maximum_absolute_difference: outputDifference.maximum_absolute_difference,
      first_output_top1_index: outputDifference.candidate_top1_index,
      baseline_first_output_top1_index: outputDifference.baseline_top1_index,
      first_output_top1_changed: outputDifference.top1_changed,
      deterministic_replay: capture.deterministic_replay === true,
      replay_output_tensor_count: capture.outputs.length,
      output_tensors: outputDifference.outputs,
      first_output_difference: outputDifference.first_difference,
      candidate_ledger_sha256: "",
    };
    row.candidate_ledger_sha256 = await sha256Text(candidateCanonical(row));
    candidates.push(row);
  }

  const inputClasses = equivalenceClasses(candidates, "input_tensor_sha256", "input_class");
  const outputClasses = equivalenceClasses(candidates, "output_tensor_set_sha256", "output_class");
  const exactRows = candidates.filter((row) => row.source_exact_tensor_realization);
  const nonExactRows = candidates.filter((row) => !row.source_exact_tensor_realization);
  const changedRows = candidates.filter((row) => !row.output_identical_to_witness_replay);
  const top1Rows = candidates.filter((row) => row.first_output_top1_changed);
  const mostSensitive = [...candidates].sort(compareSensitivity)[0] || null;
  const evidence = {
    schema: PREPROCESSING_CONSEQUENCE_SCHEMA,
    method_version: PREPROCESSING_CONSEQUENCE_METHOD_VERSION,
    evidence_class: "MEASURED_SYNTHETIC",
    assessment_kind: "LOCAL_COUNTERFACTUAL_TENSOR_REPLAY",
    status: "assessed",
    artifact_sha256: artifactSha256,
    source_preprocessing_schema: preprocessing.schema,
    source_preprocessing_portfolio_sha256: preprocessing.portfolio_ledger_sha256,
    source_input_witness_ledger_sha256: witness.witness_ledger_sha256,
    source_input_witness_tensor_sha256: witness.full_model_input_tensor_sha256,
    runtime: {
      name: String(runtime?.name || "LiteRT.js"),
      version: String(runtime?.version || "not_declared"),
      backend: String(runtime?.backend || "wasm"),
      execution_scope: "browser_local",
    },
    execution_contract: {
      candidate_count: candidates.length,
      captured_repetitions_per_input: 2,
      deterministic_replay_required: true,
      model_compilation_reused_across_candidates: true,
      stateful_variable_tensor_count: (analysis?.tensors || []).filter((tensor) => tensor.is_variable).length,
      primary_input_tensor_index: witness.model_input_tensor_index,
      primary_input_dtype: witness.model_input_dtype,
      primary_input_shape: witness.model_input_shape,
      other_inputs: otherInputs,
      output_tensor_count: outputDetails.length,
    },
    baseline: {
      input_tensor_sha256: baselineInputSha256,
      output_tensor_set_sha256: baselineOutputSha256,
      output_tensor_count: baselineCapture.outputs.length,
      output_total_element_count: baselineOutputs.total_element_count,
      first_output_top1_index: baselineOutputs.baseline_top1_index,
      output_tensors: baselineOutputs.outputs.map((row) => ({
        output_index: row.output_index,
        output_name: row.output_name,
        output_dtype: row.output_dtype,
        output_shape: row.output_shape,
        element_count: row.element_count,
        baseline_sha256: row.baseline_sha256,
      })),
    },
    candidate_count: candidates.length,
    exact_source_contract_count: exactRows.length,
    non_exact_source_contract_count: nonExactRows.length,
    unique_input_tensor_count: inputClasses.length,
    unique_output_tensor_set_count: outputClasses.length,
    output_changed_candidate_count: changedRows.length,
    non_exact_output_changed_candidate_count: nonExactRows.filter((row) => !row.output_identical_to_witness_replay).length,
    top1_changed_candidate_count: top1Rows.length,
    exact_contract_output_conservation: exactRows.every((row) => row.input_identical_to_witness && row.output_identical_to_witness_replay),
    maximum_output_changed_element_count: Math.max(0, ...candidates.map((row) => row.output_changed_element_count)),
    maximum_output_absolute_difference: Math.max(0, ...candidates.map((row) => finiteNumber(row.output_maximum_absolute_difference))),
    most_output_sensitive_contract_id: mostSensitive?.contract_id || null,
    candidates,
    input_equivalence_classes: inputClasses,
    output_equivalence_classes: outputClasses,
    portfolio_ledger_sha256: "",
    interpretation_boundary: "MEASURED_SYNTHETIC counterfactual replay of analyzer-generated quantized tensors through the declared browser LiteRT.js WASM runtime. The replay observes model outputs for these constructed tensors; it does not observe the production decoder, resize/interpolation path, channel order, normalization code, device runtime, label semantics, representative-data frequency, task accuracy, or user impact. Equal output digests prove equality only for the captured runtime, artifact, inputs, and output bytes bound by this ledger.",
  };
  evidence.portfolio_ledger_sha256 = await sha256Text(portfolioCanonical(evidence));
  return evidence;
}

export function candidateCanonical(row) {
  let value = CANDIDATE_PREFIX;
  value += `candidate=${row.candidate_index};contract=${row.contract_id};source=${row.source_candidate_ledger_sha256};exact=${row.source_exact_tensor_realization};unrealizable=${row.source_unrealizable_tensor_element_count}\n`;
  value += `input=${row.input_tensor_sha256};same=${row.input_identical_to_witness};changed=${row.input_changed_element_count};total_abs=${row.input_total_absolute_code_difference_decimal};max=${row.input_maximum_absolute_code_difference}\n`;
  value += `output=${row.output_tensor_set_sha256};same=${row.output_identical_to_witness_replay};changed=${row.output_changed_element_count};total=${row.output_total_element_count};total_abs=${nullable(row.output_total_absolute_difference_decimal)};mean=${numberText(row.output_mean_absolute_difference)};rms=${numberText(row.output_root_mean_square_difference)};max=${numberText(row.output_maximum_absolute_difference)};top1=${nullable(row.first_output_top1_index)};baseline_top1=${nullable(row.baseline_first_output_top1_index)};flip=${row.first_output_top1_changed};deterministic=${row.deterministic_replay}\n`;
  for (const output of row.output_tensors) {
    value += `tensor=${output.output_index};name=${output.output_name};dtype=${output.output_dtype};shape=${output.output_shape.join("x")};elements=${output.element_count};baseline=${output.baseline_sha256};candidate=${output.candidate_sha256};changed=${output.changed_element_count};total_abs=${nullable(output.total_absolute_difference_decimal)};mean=${numberText(output.mean_absolute_difference)};rms=${numberText(output.root_mean_square_difference)};max=${numberText(output.maximum_absolute_difference)}\n`;
  }
  return value;
}

export function portfolioCanonical(evidence) {
  let value = PORTFOLIO_PREFIX;
  value += `artifact=${evidence.artifact_sha256};preprocessing=${evidence.source_preprocessing_portfolio_sha256};witness=${evidence.source_input_witness_ledger_sha256};runtime=${evidence.runtime.name}@${evidence.runtime.version}/${evidence.runtime.backend};baseline_input=${evidence.baseline.input_tensor_sha256};baseline_output=${evidence.baseline.output_tensor_set_sha256}\n`;
  for (const row of evidence.candidates) value += `candidate=${row.contract_id};ledger=${row.candidate_ledger_sha256}\n`;
  for (const row of evidence.input_equivalence_classes) value += `input_class=${row.class_id};sha=${row.sha256};contracts=${row.contract_ids.join(",")}\n`;
  for (const row of evidence.output_equivalence_classes) value += `output_class=${row.class_id};sha=${row.sha256};contracts=${row.contract_ids.join(",")}\n`;
  return value;
}

export function summarizeArrayDifference(baseline, candidate) {
  assert(ArrayBuffer.isView(baseline) && ArrayBuffer.isView(candidate), "Difference operands must be typed arrays.");
  assert(baseline.constructor === candidate.constructor && baseline.length === candidate.length, "Difference operands must have identical typed-array contracts.");
  const integer = isIntegerTypedArray(baseline);
  let changed = 0;
  let totalAbsInteger = 0n;
  let totalAbs = 0;
  let sumSq = 0;
  let maximum = 0;
  let first = null;
  let nonFinite = 0;
  for (let index = 0; index < baseline.length; index += 1) {
    const left = Number(baseline[index]);
    const right = Number(candidate[index]);
    if (!Object.is(left, right)) {
      changed += 1;
      if (!first) first = { linear_index: index, baseline_value: left, candidate_value: right };
    }
    if (!Number.isFinite(left) || !Number.isFinite(right)) {
      if (!Object.is(left, right)) nonFinite += 1;
      continue;
    }
    const delta = right - left;
    const absolute = Math.abs(delta);
    if (integer) totalAbsInteger += BigInt(Math.trunc(absolute));
    totalAbs += absolute;
    sumSq += delta * delta;
    maximum = Math.max(maximum, absolute);
  }
  return {
    difference_domain: integer ? "exact_storage_code_integer" : "numeric_runtime_value",
    total_element_count: baseline.length,
    changed_element_count: changed,
    total_absolute_difference_decimal: integer ? totalAbsInteger.toString() : null,
    total_absolute_difference: totalAbs,
    mean_absolute_difference: baseline.length ? totalAbs / baseline.length : 0,
    root_mean_square_difference: baseline.length ? Math.sqrt(sumSq / baseline.length) : 0,
    maximum_absolute_difference: maximum,
    non_finite_mismatch_count: nonFinite,
    first_difference: first,
  };
}

async function summarizeOutputSet(baseline, candidate, details) {
  assert(baseline.length === candidate.length && baseline.length === details.length, "Output tensor cardinality mismatch.");
  const outputs = [];
  let totalElements = 0;
  let changed = 0;
  let totalAbs = 0;
  let sumSq = 0;
  let maximum = 0;
  let allInteger = true;
  let totalAbsInteger = 0n;
  let first = null;
  for (let index = 0; index < baseline.length; index += 1) {
    const difference = summarizeArrayDifference(baseline[index], candidate[index]);
    const baselineSha256 = await sha256TypedArrayListHex([baseline[index]]);
    const candidateSha256 = await sha256TypedArrayListHex([candidate[index]]);
    totalElements += difference.total_element_count;
    changed += difference.changed_element_count;
    totalAbs += difference.total_absolute_difference;
    sumSq += difference.root_mean_square_difference ** 2 * difference.total_element_count;
    maximum = Math.max(maximum, difference.maximum_absolute_difference);
    if (difference.total_absolute_difference_decimal == null) allInteger = false;
    else totalAbsInteger += BigInt(difference.total_absolute_difference_decimal);
    if (!first && difference.first_difference) first = { output_index: index, ...difference.first_difference };
    outputs.push({
      output_index: index,
      output_name: String(details[index]?.name || `output_${index}`),
      output_dtype: String(details[index]?.dtype || candidate[index].constructor.name),
      output_shape: Array.from(details[index]?.shape || [candidate[index].length]),
      element_count: difference.total_element_count,
      baseline_sha256: baselineSha256,
      candidate_sha256: candidateSha256,
      changed_element_count: difference.changed_element_count,
      total_absolute_difference_decimal: difference.total_absolute_difference_decimal,
      mean_absolute_difference: difference.mean_absolute_difference,
      root_mean_square_difference: difference.root_mean_square_difference,
      maximum_absolute_difference: difference.maximum_absolute_difference,
      non_finite_mismatch_count: difference.non_finite_mismatch_count,
      first_difference: difference.first_difference,
    });
  }
  const baselineTop1 = baseline[0]?.length ? stableArgmax(baseline[0]) : null;
  const candidateTop1 = candidate[0]?.length ? stableArgmax(candidate[0]) : null;
  return {
    outputs,
    total_element_count: totalElements,
    changed_element_count: changed,
    total_absolute_difference_decimal: allInteger ? totalAbsInteger.toString() : null,
    total_absolute_difference: totalAbs,
    mean_absolute_difference: totalElements ? totalAbs / totalElements : 0,
    root_mean_square_difference: totalElements ? Math.sqrt(sumSq / totalElements) : 0,
    maximum_absolute_difference: maximum,
    baseline_top1_index: baselineTop1,
    candidate_top1_index: candidateTop1,
    top1_changed: baselineTop1 !== candidateTop1,
    first_difference: first,
  };
}

function equivalenceClasses(rows, key, prefix) {
  const classes = [];
  const byDigest = new Map();
  for (const row of rows) {
    const digest = row[key];
    let item = byDigest.get(digest);
    if (!item) {
      item = { class_id: `${prefix}_${classes.length}`, sha256: digest, contract_ids: [], candidate_count: 0 };
      byDigest.set(digest, item);
      classes.push(item);
    }
    item.contract_ids.push(row.contract_id);
    item.candidate_count += 1;
  }
  return classes;
}

function compareSensitivity(left, right) {
  return Number(right.output_changed_element_count) - Number(left.output_changed_element_count)
    || finiteNumber(right.output_maximum_absolute_difference) - finiteNumber(left.output_maximum_absolute_difference)
    || String(left.contract_id).localeCompare(String(right.contract_id));
}

function allocateInput(dtype, length) {
  if (dtype === "INT8") return new Int8Array(length);
  if (dtype === "UINT8") return new Uint8Array(length);
  throw new Error(`Preprocessing consequence requires INT8 or UINT8 input, received ${dtype}.`);
}

function stableArgmax(values) {
  let index = 0;
  let maximum = Number(values[0]);
  for (let cursor = 1; cursor < values.length; cursor += 1) {
    const value = Number(values[cursor]);
    if (value > maximum) {
      maximum = value;
      index = cursor;
    }
  }
  return index;
}

function isIntegerTypedArray(value) {
  return value instanceof Int8Array || value instanceof Uint8Array || value instanceof Int16Array
    || value instanceof Uint16Array || value instanceof Int32Array || value instanceof Uint32Array
    || typeof BigInt64Array !== "undefined" && (value instanceof BigInt64Array || value instanceof BigUint64Array);
}

async function rawTypedArraySha256(value) {
  return sha256Hex(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
}

async function sha256Text(value) {
  return sha256Hex(new TextEncoder().encode(value));
}

function nullable(value) {
  return value == null ? "none" : String(value);
}

function numberText(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toPrecision(17) : "non_finite";
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
