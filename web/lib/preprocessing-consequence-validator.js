import { sha256Hex, sha256TypedArrayListHex } from "./hash.js";
import { buildInputWitnessTensor } from "./input-counterexample.js";
import {
  candidateCanonical,
  portfolioCanonical,
  PREPROCESSING_CONSEQUENCE_METHOD_VERSION,
  PREPROCESSING_CONSEQUENCE_SCHEMA,
} from "./preprocessing-consequence-core.js";
import { validatePreprocessingRealizabilityAnalysis } from "./preprocessing-realizability.js";

export async function validatePreprocessingConsequenceCapture({
  analysis,
  evidence,
  baselineCapture,
  candidateCaptures,
  outputDetails,
}) {
  assert(evidence?.schema === PREPROCESSING_CONSEQUENCE_SCHEMA, "Preprocessing consequence schema mismatch.");
  assert(evidence.method_version === PREPROCESSING_CONSEQUENCE_METHOD_VERSION, "Preprocessing consequence method mismatch.");
  assert(evidence.evidence_class === "MEASURED_SYNTHETIC" && evidence.status === "assessed", "Preprocessing consequence evidence class mismatch.");
  assert(evidence.artifact_sha256 === analysis.model_sha256, "Preprocessing consequence artifact binding mismatch.");
  const preprocessing = await validatePreprocessingRealizabilityAnalysis(analysis);
  assert(evidence.source_preprocessing_portfolio_sha256 === preprocessing.evidence.portfolio_ledger_sha256, "Preprocessing consequence source portfolio mismatch.");
  assert(candidateCaptures.length === preprocessing.evidence.candidates.length && evidence.candidates.length === candidateCaptures.length, "Preprocessing consequence candidate count mismatch.");
  const witness = preprocessing.input.evidence.witnesses[preprocessing.evidence.candidates[0].witness_index];
  const canonical = buildInputWitnessTensor(witness);
  assert(rawBytesEqual(canonical.bytes, baselineCapture.input), "Preprocessing consequence baseline tensor reconstruction mismatch.");
  await verifyReplay(baselineCapture, evidence.baseline.output_tensor_set_sha256, "baseline");

  for (let index = 0; index < evidence.candidates.length; index += 1) {
    const source = preprocessing.evidence.candidates[index];
    const row = evidence.candidates[index];
    const capture = candidateCaptures[index];
    const reconstructed = independentlyReconstructInput(source, witness, canonical.bytes);
    assert(rawBytesEqual(reconstructed, capture.input), `Preprocessing consequence input reconstruction mismatch for ${source.contract_id}.`);
    const inputSha256 = await rawSha256(capture.input);
    const outputSha256 = await sha256TypedArrayListHex(capture.outputs);
    assert(row.contract_id === source.contract_id && row.source_candidate_ledger_sha256 === source.candidate_ledger_sha256, `Preprocessing consequence source join mismatch for ${source.contract_id}.`);
    assert(row.input_tensor_sha256 === inputSha256 && row.output_tensor_set_sha256 === outputSha256, `Preprocessing consequence digest mismatch for ${source.contract_id}.`);
    await verifyReplay(capture, outputSha256, source.contract_id);
    const inputDiff = independentDifference(baselineCapture.input, capture.input);
    assert(row.input_changed_element_count === inputDiff.changed && row.input_total_absolute_code_difference_decimal === inputDiff.totalAbs.toString() && row.input_maximum_absolute_code_difference === inputDiff.maximum, `Preprocessing consequence input difference mismatch for ${source.contract_id}.`);
    const aggregate = await independentOutputDifference(baselineCapture.outputs, capture.outputs, outputDetails);
    assert(row.output_changed_element_count === aggregate.changed
      && row.output_total_element_count === aggregate.total
      && row.output_total_absolute_difference_decimal === aggregate.totalAbsDecimal
      && close(row.output_mean_absolute_difference, aggregate.mean)
      && close(row.output_root_mean_square_difference, aggregate.rms)
      && close(row.output_maximum_absolute_difference, aggregate.maximum)
      && row.first_output_top1_index === aggregate.candidateTop1
      && row.baseline_first_output_top1_index === aggregate.baselineTop1
      && row.first_output_top1_changed === (aggregate.candidateTop1 !== aggregate.baselineTop1), `Preprocessing consequence output difference mismatch for ${source.contract_id}.`);
    assert(row.output_tensors.length === aggregate.outputs.length, `Preprocessing consequence output row count mismatch for ${source.contract_id}.`);
    aggregate.outputs.forEach((output, outputIndex) => {
      const actual = row.output_tensors[outputIndex];
      assert(actual.baseline_sha256 === output.baselineSha256
        && actual.candidate_sha256 === output.candidateSha256
        && actual.changed_element_count === output.changed
        && actual.total_absolute_difference_decimal === output.totalAbsDecimal
        && close(actual.mean_absolute_difference, output.mean)
        && close(actual.root_mean_square_difference, output.rms)
        && close(actual.maximum_absolute_difference, output.maximum), `Preprocessing consequence output tensor mismatch for ${source.contract_id}/${outputIndex}.`);
    });
    const ledgerSha256 = await textSha256(candidateCanonical(row));
    assert(row.candidate_ledger_sha256 === ledgerSha256, `Preprocessing consequence candidate ledger mismatch for ${source.contract_id}.`);
  }

  verifyClasses(evidence.input_equivalence_classes, evidence.candidates, "input_tensor_sha256", "input_class");
  verifyClasses(evidence.output_equivalence_classes, evidence.candidates, "output_tensor_set_sha256", "output_class");
  const portfolioSha256 = await textSha256(portfolioCanonical(evidence));
  assert(evidence.portfolio_ledger_sha256 === portfolioSha256, "Preprocessing consequence portfolio ledger mismatch.");
  return {
    status: "independently_verified",
    candidate_count: evidence.candidates.length,
    captured_repetitions_per_input: 2,
    portfolio_ledger_sha256: portfolioSha256,
  };
}

function independentlyReconstructInput(candidate, witness, witnessBytes) {
  const output = witness.model_input_dtype === "INT8" ? new Int8Array(witnessBytes.length) : new Uint8Array(witnessBytes.length);
  for (let linear = 0; linear < output.length; linear += 1) {
    const channel = linear % 3;
    const targetByte = witnessBytes[linear];
    const target = witness.model_input_dtype === "INT8" && targetByte >= 128 ? targetByte - 256 : targetByte;
    const codes = candidate.channel_maps[channel].pixel_to_tensor_codes;
    let selectedCode = codes[0];
    let selectedPixel = 0;
    let minimum = Math.abs(selectedCode - target);
    for (let pixel = 1; pixel < 256; pixel += 1) {
      const error = Math.abs(codes[pixel] - target);
      if (error < minimum) {
        selectedPixel = pixel;
        selectedCode = codes[pixel];
        minimum = error;
      }
    }
    void selectedPixel;
    output[linear] = selectedCode;
  }
  return output;
}

async function independentOutputDifference(baseline, candidate, details) {
  let total = 0;
  let changed = 0;
  let totalAbs = 0;
  let totalAbsInteger = 0n;
  let allInteger = true;
  let sumSq = 0;
  let maximum = 0;
  const outputs = [];
  for (let index = 0; index < baseline.length; index += 1) {
    const diff = independentDifference(baseline[index], candidate[index]);
    total += baseline[index].length;
    changed += diff.changed;
    totalAbs += diff.totalAbsNumber;
    sumSq += diff.sumSq;
    maximum = Math.max(maximum, diff.maximum);
    if (!diff.integer) allInteger = false;
    else totalAbsInteger += diff.totalAbs;
    outputs.push({
      baselineSha256: await sha256TypedArrayListHex([baseline[index]]),
      candidateSha256: await sha256TypedArrayListHex([candidate[index]]),
      changed: diff.changed,
      totalAbsDecimal: diff.integer ? diff.totalAbs.toString() : null,
      mean: baseline[index].length ? diff.totalAbsNumber / baseline[index].length : 0,
      rms: baseline[index].length ? Math.sqrt(diff.sumSq / baseline[index].length) : 0,
      maximum: diff.maximum,
      details: details[index],
    });
  }
  return {
    total,
    changed,
    totalAbsDecimal: allInteger ? totalAbsInteger.toString() : null,
    mean: total ? totalAbs / total : 0,
    rms: total ? Math.sqrt(sumSq / total) : 0,
    maximum,
    baselineTop1: baseline[0]?.length ? stableArgmax(baseline[0]) : null,
    candidateTop1: candidate[0]?.length ? stableArgmax(candidate[0]) : null,
    outputs,
  };
}

function independentDifference(left, right) {
  assert(left.constructor === right.constructor && left.length === right.length, "Preprocessing consequence typed-array contract mismatch.");
  const integer = isInteger(left);
  let changed = 0;
  let totalAbs = 0n;
  let totalAbsNumber = 0;
  let sumSq = 0;
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = Number(left[index]);
    const b = Number(right[index]);
    if (!Object.is(a, b)) changed += 1;
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const absolute = Math.abs(b - a);
    if (integer) totalAbs += BigInt(Math.trunc(absolute));
    totalAbsNumber += absolute;
    sumSq += (b - a) ** 2;
    maximum = Math.max(maximum, absolute);
  }
  return { integer, changed, totalAbs, totalAbsNumber, sumSq, maximum };
}

async function verifyReplay(capture, digest, label) {
  assert(capture.deterministic_replay === true && Array.isArray(capture.repeat_outputs), `Preprocessing consequence repeat capture missing for ${label}.`);
  const first = await sha256TypedArrayListHex(capture.outputs);
  const second = await sha256TypedArrayListHex(capture.repeat_outputs);
  assert(first === digest && second === digest && outputSetsEqual(capture.outputs, capture.repeat_outputs), `Preprocessing consequence replay is non-deterministic for ${label}.`);
}

function verifyClasses(classes, rows, key, prefix) {
  const expected = [];
  const map = new Map();
  rows.forEach((row) => {
    if (!map.has(row[key])) {
      const item = { class_id: `${prefix}_${expected.length}`, sha256: row[key], contract_ids: [] };
      map.set(row[key], item);
      expected.push(item);
    }
    map.get(row[key]).contract_ids.push(row.contract_id);
  });
  assert(JSON.stringify(classes.map(({ class_id, sha256, contract_ids }) => ({ class_id, sha256, contract_ids }))) === JSON.stringify(expected), `Preprocessing consequence ${prefix} equivalence classes mismatch.`);
}

function outputSetsEqual(left, right) {
  return left.length === right.length && left.every((array, index) => rawBytesEqual(array, right[index]));
}

function rawBytesEqual(left, right) {
  if (!ArrayBuffer.isView(left) || !ArrayBuffer.isView(right) || left.byteLength !== right.byteLength) return false;
  const a = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const b = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  return a.every((value, index) => value === b[index]);
}

function isInteger(value) {
  return value instanceof Int8Array || value instanceof Uint8Array || value instanceof Int16Array
    || value instanceof Uint16Array || value instanceof Int32Array || value instanceof Uint32Array;
}

function stableArgmax(values) {
  let selected = 0;
  for (let index = 1; index < values.length; index += 1) if (Number(values[index]) > Number(values[selected])) selected = index;
  return selected;
}

async function rawSha256(value) {
  return sha256Hex(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
}

async function textSha256(value) {
  return sha256Hex(new TextEncoder().encode(value));
}

function close(left, right) {
  return Math.abs(Number(left) - Number(right)) <= 1e-12 * Math.max(1, Math.abs(Number(left)), Math.abs(Number(right)));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
