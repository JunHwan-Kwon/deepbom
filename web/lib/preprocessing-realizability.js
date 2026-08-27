import { sha256Hex } from "./hash.js";
import { buildInputWitnessTensor, validateInputCounterexampleAnalysis } from "./input-counterexample.js";
import { decodeStoredRgbPng, encodeRgbPng } from "./rgb-png.js";

export const PREPROCESSING_REALIZABILITY_SCHEMA = "deepbom.preprocessing_realizability.v1";
export const PREPROCESSING_REALIZABILITY_METHOD_VERSION = "2026-07-18.1";

const CANDIDATE_PREFIX = "deepbom.preprocessing_realizability.candidate.v1\0";
const PORTFOLIO_PREFIX = "deepbom.preprocessing_realizability.portfolio.v1\0";
const CONTRACTS = new Map([
  ["raw_storage_rgb", { transform: "raw", order: [0, 1, 2], rounding: "not_applicable_direct_storage" }],
  ["raw_storage_bgr", { transform: "raw", order: [2, 1, 0], rounding: "not_applicable_direct_storage" }],
  ["artifact_affine_rgb", { transform: "artifact", order: [0, 1, 2] }],
  ["center_128_div_128_rgb", { transform: "center128", order: [0, 1, 2] }],
  ["minus_one_to_one_rgb", { transform: "minus_one_one", order: [0, 1, 2] }],
  ["unit_interval_rgb", { transform: "unit", order: [0, 1, 2] }],
  ["imagenet_mean_std_rgb", { transform: "imagenet", order: [0, 1, 2], mean: [485n, 456n, 406n], std: [229n, 224n, 225n] }],
  ["imagenet_mean_std_bgr", { transform: "imagenet", order: [2, 1, 0], mean: [406n, 456n, 485n], std: [225n, 224n, 229n] }],
]);

export async function validatePreprocessingRealizabilityAnalysis(analysis, inputValidation = null) {
  const evidence = analysis?.preprocessing_realizability;
  assertShape(evidence);
  const input = inputValidation || await validateInputCounterexampleAnalysis(analysis);
  assert(evidence.source_input_counterexample_portfolio_sha256 === input.evidence.portfolio_ledger_sha256, "Preprocessing source portfolio join mismatch.");
  assert(evidence.source_witness_count === input.evidence.witnesses.length, "Preprocessing source witness count mismatch.");
  const eligible = input.evidence.witnesses.filter(eligibleWitness);
  assert(evidence.eligible_image_witness_count === eligible.length, "Eligible image witness count mismatch.");
  assert(evidence.ineligible_witness_count === input.evidence.witnesses.length - eligible.length, "Ineligible image witness count mismatch.");
  assert(evidence.candidate_contract_count === CONTRACTS.size, "Preprocessing contract count mismatch.");
  assert(evidence.candidate_evaluation_count === evidence.candidates.length, "Preprocessing candidate evaluation count mismatch.");
  assert(evidence.candidates.length === eligible.length * CONTRACTS.size, "Preprocessing candidate matrix is incomplete.");

  const reconstructed = [];
  for (const candidate of evidence.candidates) {
    const witness = input.evidence.witnesses[integer(candidate.witness_index, "candidate witness index")];
    assert(witness && eligibleWitness(witness), `Preprocessing candidate ${candidate.contract_id} references an ineligible witness.`);
    const contract = CONTRACTS.get(candidate.contract_id);
    assert(contract, `Unknown preprocessing contract ${candidate.contract_id}.`);
    assert(candidate.status === "assessed", `Preprocessing candidate ${candidate.contract_id} is not assessed in the independently verifiable matrix.`);
    reconstructed.push(await reconstructCandidate(candidate, witness, contract));
  }
  const assessed = evidence.candidates.filter((candidate) => candidate.status === "assessed");
  const exact = assessed.filter((candidate) => candidate.exact_tensor_realization);
  const nonExact = assessed.filter((candidate) => !candidate.exact_tensor_realization);
  assert(evidence.assessed_candidate_count === assessed.length, "Assessed preprocessing candidate count mismatch.");
  assert(evidence.exact_tensor_realization_candidate_count === exact.length, "Exact preprocessing candidate count mismatch.");
  assert(evidence.non_exact_candidate_count === nonExact.length, "Non-exact preprocessing candidate count mismatch.");
  assert(sameArray(evidence.exact_contract_ids, exact.map((candidate) => candidate.contract_id)), "Exact preprocessing contract ids mismatch.");
  const best = [...nonExact].sort((left, right) => left.unrealizable_tensor_element_count - right.unrealizable_tensor_element_count
    || compareBigInt(BigInt(left.minimum_total_absolute_tensor_code_error_decimal), BigInt(right.minimum_total_absolute_tensor_code_error_decimal))
    || left.contract_id.localeCompare(right.contract_id))[0];
  assert(evidence.best_non_exact_contract_id === (best?.contract_id || ""), "Best non-exact preprocessing contract mismatch.");
  assert(evidence.best_non_exact_unrealizable_element_count === (best?.unrealizable_tensor_element_count ?? null), "Best non-exact preprocessing mismatch count is inconsistent.");
  const notAssessed = evidence.candidates.length - assessed.length;
  const conservation = `${evidence.candidates.length} evaluations = ${assessed.length} assessed (${exact.length} exact + ${nonExact.length} non-exact) + ${notAssessed} not-assessed`;
  assert(evidence.candidate_conservation === conservation, "Preprocessing candidate conservation mismatch.");
  const portfolioSha256 = await sha256Hex(new TextEncoder().encode(portfolioCanonical(evidence)));
  assert(portfolioSha256 === evidence.portfolio_ledger_sha256, "Preprocessing portfolio ledger SHA-256 mismatch.");
  return { evidence, candidates: reconstructed, input, portfolioSha256 };
}

export function buildCandidateRgbFixture(candidate, witness) {
  const contract = CONTRACTS.get(candidate?.contract_id);
  assert(contract, "Preprocessing fixture contract is unknown.");
  const tensor = buildInputWitnessTensor(witness);
  const [batch, height, width, channels] = tensor.shape;
  assert(batch === 1 && channels === 3, "Preprocessing RGB fixture requires NHWC [1,H,W,3].");
  assert(Array.isArray(candidate.channel_maps) && candidate.channel_maps.length === 3, "Preprocessing channel maps are missing.");
  const rgb = new Uint8Array(height * width * 3);
  let totalError = 0n;
  let exactElements = 0;
  let maximumError = 0;
  for (let linear = 0; linear < tensor.elementCount; linear += 1) {
    const tensorChannel = linear % 3;
    const target = decodeTensorCode(tensor.bytes[linear], witness.model_input_dtype);
    const selection = choosePixel(candidate.channel_maps[tensorChannel].pixel_to_tensor_codes, target);
    rgb[(linear - tensorChannel) + contract.order[tensorChannel]] = selection.pixel;
    totalError += BigInt(selection.error);
    maximumError = Math.max(maximumError, selection.error);
    if (selection.error === 0) exactElements += 1;
  }
  const png = encodeRgbPng(rgb, width, height);
  const decoded = decodeStoredRgbPng(png);
  assert(decoded.width === width && decoded.height === height && sameArray(decoded.rgb, rgb), "Generated RGB PNG does not round-trip byte-for-byte.");
  return { rgb, png, width, height, exactElements, totalError, maximumError };
}

async function reconstructCandidate(candidate, witness, contract) {
  assert(candidate.source_op_index === witness.source_op_index, `Preprocessing source op mismatch for ${candidate.contract_id}.`);
  assert(candidate.source_input_witness_ledger_sha256 === witness.witness_ledger_sha256, `Preprocessing witness ledger join mismatch for ${candidate.contract_id}.`);
  assert(candidate.source_pixel_order === "RGB" && candidate.tensor_channel_order === (contract.order.join("") === "012" ? "RGB" : "BGR"), `Preprocessing channel order mismatch for ${candidate.contract_id}.`);
  assert(candidate.rounding_mode === (contract.rounding || "nearest_ties_away_from_zero"), `Preprocessing rounding mode mismatch for ${candidate.contract_id}.`);
  const scale = exactPositiveF64Ratio(witness.model_input_scale);
  const expectedMaps = [0, 1, 2].map((channel) => {
    const codes = Array.from({ length: 256 }, (_, pixel) => mapPixel(pixel, channel, witness, contract, scale));
    return summarizeMap(channel, contract.order[channel], codes);
  });
  assert(candidate.channel_maps.length === 3, `Preprocessing channel-map count mismatch for ${candidate.contract_id}.`);
  for (let channel = 0; channel < 3; channel += 1) assertMap(candidate.channel_maps[channel], expectedMaps[channel], candidate.contract_id);
  const fixture = buildCandidateRgbFixture(candidate, witness);
  const fixtureSha256 = await sha256Hex(fixture.rgb);
  assert(fixtureSha256 === candidate.nearest_rgb_fixture_sha256, `Nearest RGB fixture SHA-256 mismatch for ${candidate.contract_id}.`);
  const exactTensor = fixture.exactElements === witness.model_input_element_count;
  assert(candidate.exact_tensor_realization === exactTensor, `Exact tensor realizability mismatch for ${candidate.contract_id}.`);
  assert(candidate.exact_tensor_element_count === fixture.exactElements, `Exact tensor element count mismatch for ${candidate.contract_id}.`);
  assert(candidate.unrealizable_tensor_element_count === witness.model_input_element_count - fixture.exactElements, `Unrealizable tensor element count mismatch for ${candidate.contract_id}.`);
  assert(BigInt(candidate.minimum_total_absolute_tensor_code_error_decimal) === fixture.totalError, `Minimum tensor-code error mismatch for ${candidate.contract_id}.`);
  assert(candidate.maximum_absolute_tensor_code_error === fixture.maximumError, `Maximum tensor-code error mismatch for ${candidate.contract_id}.`);
  assert((candidate.exact_rgb_fixture_sha256 ?? null) === (exactTensor ? fixtureSha256 : null), `Exact RGB fixture digest mismatch for ${candidate.contract_id}.`);
  validateCodeRows(candidate, witness, expectedMaps);
  const ledgerSha256 = await sha256Hex(new TextEncoder().encode(candidateCanonical(candidate)));
  assert(ledgerSha256 === candidate.candidate_ledger_sha256, `Preprocessing candidate ledger SHA-256 mismatch for ${candidate.contract_id}.`);
  return { candidate, fixture, fixtureSha256, ledgerSha256 };
}

function validateCodeRows(candidate, witness, maps) {
  const tensor = buildInputWitnessTensor(witness);
  const counts = new Map();
  let firstMismatch = null;
  for (let linear = 0; linear < tensor.elementCount; linear += 1) {
    const channel = linear % 3;
    const target = decodeTensorCode(tensor.bytes[linear], witness.model_input_dtype);
    const key = `${channel}:${target}`;
    counts.set(key, (counts.get(key) || 0) + 1);
    const selection = choosePixel(maps[channel].pixel_to_tensor_codes, target);
    if (!firstMismatch && selection.error > 0) {
      firstMismatch = { linear, channel, target, ...selection };
    }
  }
  assert(candidate.witness_code_realizations.length === counts.size, `Witness code-row count mismatch for ${candidate.contract_id}.`);
  let exactPairs = 0;
  for (const row of candidate.witness_code_realizations) {
    const key = `${row.tensor_channel}:${row.target_tensor_code}`;
    assert(counts.get(key) === row.tensor_element_count, `Witness code frequency mismatch for ${candidate.contract_id}/${key}.`);
    const selection = choosePixel(maps[row.tensor_channel].pixel_to_tensor_codes, row.target_tensor_code);
    assert(row.source_pixel_channel === maps[row.tensor_channel].source_pixel_channel, `Witness source channel mismatch for ${candidate.contract_id}/${key}.`);
    assert(sameArray(row.exact_source_pixel_codes, selection.exactPixels), `Witness exact inverse mismatch for ${candidate.contract_id}/${key}.`);
    assert(row.selected_source_pixel_code === selection.pixel && row.roundtrip_tensor_code === selection.mapped && row.absolute_tensor_code_error === selection.error, `Witness nearest inverse mismatch for ${candidate.contract_id}/${key}.`);
    if (selection.error === 0) exactPairs += 1;
  }
  assert(candidate.distinct_witness_channel_code_pair_count === counts.size && candidate.exact_witness_channel_code_pair_count === exactPairs, `Witness pair summary mismatch for ${candidate.contract_id}.`);
  if (!firstMismatch) {
    assert(candidate.first_unrealizable_element == null, `Unexpected first mismatch for ${candidate.contract_id}.`);
  } else {
    const row = candidate.first_unrealizable_element;
    const width = witness.model_input_shape[2];
    assert(row.tensor_linear_index === firstMismatch.linear
      && sameArray(row.tensor_coordinate_nhwc, [0, Math.floor(firstMismatch.linear / 3 / width), Math.floor(firstMismatch.linear / 3) % width, firstMismatch.channel])
      && row.tensor_channel === firstMismatch.channel
      && row.source_pixel_channel === maps[firstMismatch.channel].source_pixel_channel
      && row.target_tensor_code === firstMismatch.target
      && row.selected_source_pixel_code === firstMismatch.pixel
      && row.roundtrip_tensor_code === firstMismatch.mapped
      && row.absolute_tensor_code_error === firstMismatch.error, `First unrealizable witness element mismatch for ${candidate.contract_id}.`);
  }
}

function mapPixel(pixel, channel, witness, contract, scale) {
  const [qmin, qmax] = witness.model_input_code_range;
  if (contract.transform === "raw") return witness.model_input_dtype === "INT8" && pixel >= 128 ? pixel - 256 : pixel;
  if (contract.transform === "artifact") return qmin + pixel;
  let numerator;
  let denominator;
  if (contract.transform === "center128") [numerator, denominator] = [BigInt(pixel) - 128n, 128n];
  else if (contract.transform === "minus_one_one") [numerator, denominator] = [2n * BigInt(pixel) - 255n, 255n];
  else if (contract.transform === "unit") [numerator, denominator] = [BigInt(pixel), 255n];
  else if (contract.transform === "imagenet") [numerator, denominator] = [1000n * BigInt(pixel) - 255n * contract.mean[channel], 255n * contract.std[channel]];
  else throw new Error(`Unsupported preprocessing transform ${contract.transform}.`);
  const rounded = roundRatioAway(numerator * scale.denominator, denominator * scale.numerator);
  const shifted = rounded + BigInt(witness.model_input_zero_point);
  return Number(clampBigInt(shifted, BigInt(qmin), BigInt(qmax)));
}

function exactPositiveF64Ratio(value) {
  assert(Number.isFinite(value) && value > 0, "Preprocessing input scale must be positive and finite.");
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  const bits = view.getBigUint64(0, false);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  let numerator = exponentBits === 0 ? fraction : (1n << 52n) | fraction;
  let exponent = exponentBits === 0 ? -1074 : exponentBits - 1023 - 52;
  assert(numerator > 0n, "Preprocessing input scale mantissa is zero.");
  while (exponent < 0 && (numerator & 1n) === 0n) {
    numerator >>= 1n;
    exponent += 1;
  }
  return exponent >= 0
    ? { numerator: numerator << BigInt(exponent), denominator: 1n }
    : { numerator, denominator: 1n << BigInt(-exponent) };
}

function roundRatioAway(numerator, denominator) {
  assert(denominator > 0n, "Preprocessing rational denominator must be positive.");
  const negative = numerator < 0n;
  const magnitude = negative ? -numerator : numerator;
  const quotient = magnitude / denominator;
  const remainder = magnitude % denominator;
  const rounded = quotient + (2n * remainder >= denominator ? 1n : 0n);
  return negative ? -rounded : rounded;
}

function summarizeMap(tensorChannel, sourceChannel, codes) {
  const counts = new Map();
  for (const value of codes) counts.set(value, (counts.get(value) || 0) + 1);
  const keys = [...counts.keys()].sort((left, right) => left - right);
  return {
    tensor_channel: tensorChannel,
    source_pixel_channel: sourceChannel,
    source_pixel_channel_name: ["R", "G", "B"][sourceChannel],
    reachable_tensor_code_count: counts.size,
    tensor_code_hole_count: 256 - counts.size,
    collision_tensor_code_count: [...counts.values()].filter((count) => count > 1).length,
    maximum_preimage_multiplicity: Math.max(...counts.values()),
    reachable_tensor_code_min: keys[0],
    reachable_tensor_code_max: keys[keys.length - 1],
    pixel_to_tensor_codes: codes,
  };
}

function assertMap(actual, expected, contractId) {
  for (const key of ["tensor_channel", "source_pixel_channel", "source_pixel_channel_name", "reachable_tensor_code_count", "tensor_code_hole_count", "collision_tensor_code_count", "maximum_preimage_multiplicity", "reachable_tensor_code_min", "reachable_tensor_code_max"]) {
    assert(actual[key] === expected[key], `Preprocessing ${key} mismatch for ${contractId}/channel ${expected.tensor_channel}.`);
  }
  assert(sameArray(actual.pixel_to_tensor_codes, expected.pixel_to_tensor_codes), `Preprocessing LUT mismatch for ${contractId}/channel ${expected.tensor_channel}.`);
}

function choosePixel(codes, target) {
  const exactPixels = [];
  let best = { pixel: 0, mapped: codes[0], error: Math.abs(codes[0] - target) };
  for (let pixel = 0; pixel < codes.length; pixel += 1) {
    const mapped = codes[pixel];
    const error = Math.abs(mapped - target);
    if (mapped === target) exactPixels.push(pixel);
    if (error < best.error || (error === best.error && pixel < best.pixel)) best = { pixel, mapped, error };
  }
  if (exactPixels.length) return { pixel: exactPixels[0], mapped: target, error: 0, exactPixels };
  return { ...best, exactPixels };
}

function candidateCanonical(candidate) {
  let value = CANDIDATE_PREFIX;
  value += `witness=${candidate.witness_index};source=${candidate.source_op_index};contract=${candidate.contract_id};order=${candidate.tensor_channel_order};status=${candidate.status};input_witness=${candidate.source_input_witness_ledger_sha256}\n`;
  for (const map of candidate.channel_maps) value += `map=${map.tensor_channel};source_channel=${map.source_pixel_channel};reachable=${map.reachable_tensor_code_count};holes=${map.tensor_code_hole_count};collisions=${map.collision_tensor_code_count};multiplicity=${map.maximum_preimage_multiplicity};codes=${map.pixel_to_tensor_codes.join(",")}\n`;
  for (const row of candidate.witness_code_realizations) value += `code=${row.tensor_channel};target=${row.target_tensor_code};source_channel=${row.source_pixel_channel};count=${row.tensor_element_count};exact_pixels=${row.exact_source_pixel_codes.join(",")};selected=${row.selected_source_pixel_code};roundtrip=${row.roundtrip_tensor_code};error=${row.absolute_tensor_code_error}\n`;
  value += `summary=${candidate.witness_tensor_element_count};exact_elements=${candidate.exact_tensor_element_count};unrealizable=${candidate.unrealizable_tensor_element_count};total_error=${candidate.minimum_total_absolute_tensor_code_error_decimal};max_error=${candidate.maximum_absolute_tensor_code_error};nearest_sha=${candidate.nearest_rgb_fixture_sha256};exact_sha=${candidate.exact_rgb_fixture_sha256 ?? "none"}\n`;
  return value;
}

function portfolioCanonical(evidence) {
  let value = PORTFOLIO_PREFIX;
  value += `input_counterexample=${evidence.source_input_counterexample_portfolio_sha256}\n`;
  for (const candidate of evidence.candidates) value += `candidate=${candidate.contract_id};witness=${candidate.witness_index};status=${candidate.status};ledger=${candidate.candidate_ledger_sha256}\n`;
  return value;
}

function eligibleWitness(witness) {
  return Array.isArray(witness?.model_input_shape)
    && witness.model_input_shape.length === 4
    && witness.model_input_shape[0] === 1
    && witness.model_input_shape[1] > 0
    && witness.model_input_shape[2] > 0
    && witness.model_input_shape[3] === 3
    && ["UINT8", "INT8"].includes(witness.model_input_dtype)
    && witness.model_input_shape.reduce((product, value) => product * value, 1) === witness.model_input_element_count;
}

function assertShape(evidence) {
  assert(evidence?.schema === PREPROCESSING_REALIZABILITY_SCHEMA, "Preprocessing realizability schema mismatch.");
  assert(evidence.method_version === PREPROCESSING_REALIZABILITY_METHOD_VERSION, "Preprocessing realizability method mismatch.");
  assert(evidence.evidence_class === "DERIVED" && evidence.assessment_kind === "EXPLICIT_PREPROCESSING_COUNTERFACTUALS", "Preprocessing realizability evidence class mismatch.");
  assert(["assessed", "partial", "not_applicable", "not_applicable_no_image_witness", "not_computed_internal_scope"].includes(evidence.status), "Preprocessing realizability status is invalid.");
  assert(evidence.source_input_counterexample_schema === "deepbom.input_counterexample.v1", "Preprocessing source schema mismatch.");
  assert(Array.isArray(evidence.candidates) && Array.isArray(evidence.exact_contract_ids), "Preprocessing candidate portfolio is missing.");
  assertDigest(evidence.portfolio_ledger_sha256, "preprocessing portfolio ledger");
}

function decodeTensorCode(byte, dtype) {
  return dtype === "INT8" && byte >= 128 ? byte - 256 : byte;
}

function clampBigInt(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

function compareBigInt(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function integer(value, label) {
  assert(Number.isSafeInteger(value), `${label} must be a safe integer.`);
  return value;
}

function assertDigest(value, label) {
  assert(typeof value === "string" && /^[a-f0-9]{64}$/.test(value), `${label} must be SHA-256.`);
}

function sameArray(left, right) {
  return Array.isArray(left) || left instanceof Uint8Array
    ? left.length === right.length && left.every((value, index) => value === right[index])
    : false;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
