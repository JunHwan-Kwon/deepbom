import { sha256Hex } from "./hash.js";
import {
  reconstructKernelChannel,
  reconstructKernelOpChannelsWithTerms,
} from "./kernel-witness.js";
import { reconstructRoundingEquivalenceIntervalChannel } from "./rounding-equivalence.js";

export const ACCUMULATOR_REACHABILITY_SCHEMA = "deepbom.accumulator_reachability.v1";
const METHOD_VERSION = "2026-07-18.1";
const LEDGER_PREFIX = new TextEncoder().encode("deepbom.accumulator_reachability.v1\0");
const TOP_LIMIT = 16;
const MISSING_U64 = (1n << 64n) - 1n;
const MISSING_I64 = -(1n << 63n);

export function validateAccumulatorReachabilityShape(evidence) {
  assert(evidence?.schema === ACCUMULATOR_REACHABILITY_SCHEMA, "Accumulator-reachability schema mismatch.");
  assert(evidence.method_version === METHOD_VERSION, "Accumulator-reachability method mismatch.");
  assert(["assessed", "partial", "not_assessed", "not_applicable", "not_computed_internal_scope"].includes(evidence.status), "Accumulator-reachability status is invalid.");
  assert(evidence.evidence_class === "DERIVED", "Accumulator-reachability evidence class must be DERIVED.");
  assert(Array.isArray(evidence.ops) && evidence.ops.length === Number(evidence.candidate_op_count), "Accumulator-reachability op cardinality mismatch.");
  assert(Number(evidence.assessed_op_count) + Number(evidence.unassessed_op_count) === evidence.ops.length, "Accumulator-reachability op conservation failed.");
  const totals = sumRows(evidence.ops.filter((row) => row.assessment_status === "assessed"));
  compareDecimal(evidence.interval_state_count_decimal, totals.interval, "Global interval states");
  compareDecimal(evidence.lattice_compatible_state_count_decimal, totals.compatible, "Global compatible states");
  compareDecimal(evidence.certified_reachable_state_count_decimal, totals.certified, "Global certified states");
  compareDecimal(evidence.provably_unreachable_state_count_decimal, totals.excluded, "Global excluded states");
  compareDecimal(evidence.unresolved_state_count_decimal, totals.unresolved, "Global unresolved states");
  compareDecimal(evidence.interval_divergent_state_count_decimal, totals.intervalDivergent, "Global divergent states");
  compareDecimal(evidence.exact_reachable_divergent_state_count_decimal, totals.exactDivergent, "Global exact divergent states");
  compareDecimal(evidence.provably_unreachable_divergent_state_count_decimal, totals.excludedDivergent, "Global excluded divergent states");
  compareDecimal(evidence.unresolved_divergent_state_count_decimal, totals.unresolvedDivergent, "Global unresolved divergent states");
  assert(totals.interval === totals.certified + totals.excluded + totals.unresolved, "Global reachability state partition failed.");
  assert(totals.intervalDivergent === totals.exactDivergent + totals.excludedDivergent + totals.unresolvedDivergent, "Global divergent state partition failed.");
  for (const row of evidence.ops) validateOpShape(row);
  return evidence;
}

export function reconstructAccumulatorReachabilityAnalysis(analysis, modelBytes) {
  const evidence = analysis?.accumulator_reachability;
  validateAccumulatorReachabilityShape(evidence);
  const rows = evidence.ops.map((source) => reconstructOp(analysis, modelBytes, source));
  const assessed = rows.filter((row) => row.assessment_status === "assessed");
  const totals = sumRows(assessed);
  const ranking = [...assessed].sort(compareOps).map((row) => row.op_index);
  return {
    status: !rows.length ? "not_applicable" : assessed.length === rows.length ? "assessed" : assessed.length ? "partial" : "not_assessed",
    candidate_op_count: rows.length,
    assessed_op_count: assessed.length,
    unassessed_op_count: rows.length - assessed.length,
    assessed_channel_count: sumNumber(assessed, "assessed_channel_count"),
    complete_integer_interval_channel_count: sumNumber(assessed, "complete_integer_interval_channel_count"),
    complete_modular_lattice_channel_count: sumNumber(assessed, "complete_modular_lattice_channel_count"),
    partial_band_channel_count: sumNumber(assessed, "partial_band_channel_count"),
    singleton_channel_count: sumNumber(assessed, "singleton_channel_count"),
    exact_reachable_divergent_channel_count: sumNumber(assessed, "exact_reachable_divergent_channel_count"),
    unresolved_divergent_channel_count: sumNumber(assessed, "unresolved_divergent_channel_count"),
    interval_only_divergent_channel_count: sumNumber(assessed, "interval_only_divergent_channel_count"),
    interval_state_count_decimal: String(totals.interval),
    lattice_compatible_state_count_decimal: String(totals.compatible),
    certified_reachable_state_count_decimal: String(totals.certified),
    provably_unreachable_state_count_decimal: String(totals.excluded),
    unresolved_state_count_decimal: String(totals.unresolved),
    interval_divergent_state_count_decimal: String(totals.intervalDivergent),
    exact_reachable_divergent_state_count_decimal: String(totals.exactDivergent),
    provably_unreachable_divergent_state_count_decimal: String(totals.excludedDivergent),
    unresolved_divergent_state_count_decimal: String(totals.unresolvedDivergent),
    exact_reachable_divergent_ratio: ratio(totals.exactDivergent, totals.intervalDivergent),
    maximum_lattice_gcd: maxOptional(assessed, "maximum_lattice_gcd"),
    reachability_ranking_op_indices: ranking,
    ops: rows,
  };
}

export function validateAccumulatorReachabilityAgainstReconstruction(evidence, expected) {
  const actual = validateAccumulatorReachabilityShape(evidence);
  for (const key of [
    "status", "candidate_op_count", "assessed_op_count", "unassessed_op_count", "assessed_channel_count",
    "complete_integer_interval_channel_count", "complete_modular_lattice_channel_count", "partial_band_channel_count",
    "singleton_channel_count", "exact_reachable_divergent_channel_count", "unresolved_divergent_channel_count",
    "interval_only_divergent_channel_count", "interval_state_count_decimal", "lattice_compatible_state_count_decimal",
    "certified_reachable_state_count_decimal", "provably_unreachable_state_count_decimal", "unresolved_state_count_decimal",
    "interval_divergent_state_count_decimal", "exact_reachable_divergent_state_count_decimal",
    "provably_unreachable_divergent_state_count_decimal", "unresolved_divergent_state_count_decimal", "maximum_lattice_gcd",
  ]) assert(actual[key] === expected[key], `Accumulator-reachability ${key} differs from reconstruction.`);
  assert(close(actual.exact_reachable_divergent_ratio, expected.exact_reachable_divergent_ratio), "Accumulator-reachability ratio differs from reconstruction.");
  assert(JSON.stringify(actual.reachability_ranking_op_indices) === JSON.stringify(expected.reachability_ranking_op_indices), "Accumulator-reachability ranking differs from reconstruction.");
  assert(actual.ops.length === expected.ops.length, "Accumulator-reachability reconstructed op count mismatch.");
  actual.ops.forEach((row, index) => compareOpRow(row, expected.ops[index]));
  return expected;
}

export async function validateAccumulatorReachabilityDigests(analysis, modelBytes) {
  const expected = reconstructAccumulatorReachabilityAnalysis(analysis, modelBytes);
  validateAccumulatorReachabilityAgainstReconstruction(analysis.accumulator_reachability, expected);
  for (let index = 0; index < expected.ops.length; index += 1) {
    const row = expected.ops[index];
    if (row.assessment_status !== "assessed") continue;
    const digest = await sha256Hex(row.ledger_bytes);
    assert(digest === analysis.accumulator_reachability.ops[index].reachability_ledger_sha256, `Accumulator-reachability SHA-256 mismatch at op #${row.op_index}.`);
  }
  return expected;
}

export function reconstructAccumulatorReachabilityChannel(analysis, modelBytes, opIndex, channelIndex) {
  const kernel = reconstructKernelChannel(analysis, modelBytes, opIndex, channelIndex);
  const rounding = reconstructRoundingEquivalenceIntervalChannel(analysis, opIndex, channelIndex, true);
  const denominations = denominationsFromTerms(kernel.term_rows);
  const selected = analyzeChannel(
    Number(channelIndex),
    Number(kernel.term_count),
    Number(kernel.input_code_range[1]) - Number(kernel.input_code_range[0]),
    rounding.post_bias_minimum_decimal,
    rounding.post_bias_maximum_decimal,
    denominations,
    rounding.segments,
  );
  const firstWitness = selected.first_exact_reachable_divergent_accumulator_decimal == null
    ? []
    : aggregateWitness(selected, BigInt(selected.first_exact_reachable_divergent_accumulator_decimal));
  return {
    ...publicChannel(selected),
    post_bias_minimum_decimal: rounding.post_bias_minimum_decimal,
    post_bias_maximum_decimal: rounding.post_bias_maximum_decimal,
    denomination_coverage_steps: selected.denomination_coverage_steps,
    first_exact_reachable_aggregate_coefficient_witness: firstWitness,
    rounding_segments: rounding.segments,
  };
}

function reconstructOp(analysis, modelBytes, source) {
  const witness = analysis?.kernel_extremum_witness?.ops?.find((row) => Number(row.op_index) === Number(source.op_index));
  const equivalence = analysis?.rounding_equivalence?.ops?.find((row) => Number(row.op_index) === Number(source.op_index));
  if (source.assessment_status !== "assessed") return { ...source };
  assert(witness?.assessment_status === "assessed" && equivalence?.assessment_status === "assessed", `Reachability source evidence is unavailable at op #${source.op_index}.`);
  const kernel = reconstructKernelOpChannelsWithTerms(analysis, modelBytes, source.op_index);
  const channels = kernel.channels.map((channel, index) => {
    const rounding = reconstructRoundingEquivalenceIntervalChannel(analysis, source.op_index, index, true);
    return analyzeChannel(
      index,
      kernel.term_count,
      kernel.input_code_range[1] - kernel.input_code_range[0],
      rounding.post_bias_minimum_decimal,
      rounding.post_bias_maximum_decimal,
      denominationsFromTerms(channel.term_rows),
      rounding.segments,
    );
  });
  const totals = sumChannels(channels);
  const topChannels = [...channels].sort(compareChannels).slice(0, TOP_LIMIT).map(publicChannel);
  const ledgerBytes = reachabilityLedger(witness.witness_ledger_sha256, equivalence.equivalence_ledger_sha256, source.op_index, channels);
  return {
    op_index: Number(source.op_index),
    op_name: String(source.op_name),
    assessment_status: "assessed",
    not_assessed_reason: "",
    assessed_channel_count: channels.length,
    complete_integer_interval_channel_count: channels.filter((row) => row.proof_status === "complete_integer_interval").length,
    complete_modular_lattice_channel_count: channels.filter((row) => row.proof_status === "complete_modular_lattice").length,
    partial_band_channel_count: channels.filter((row) => row.proof_status === "partial_endpoint_bands").length,
    singleton_channel_count: channels.filter((row) => row.proof_status === "singleton").length,
    exact_reachable_divergent_channel_count: channels.filter((row) => row.exact_reachable_divergent_state_count > 0n).length,
    unresolved_divergent_channel_count: channels.filter((row) => row.unresolved_divergent_state_count > 0n).length,
    interval_only_divergent_channel_count: channels.filter((row) => row.interval_divergent_state_count > 0n && row.exact_reachable_divergent_state_count === 0n).length,
    interval_state_count_decimal: String(totals.interval),
    lattice_compatible_state_count_decimal: String(totals.compatible),
    certified_reachable_state_count_decimal: String(totals.certified),
    provably_unreachable_state_count_decimal: String(totals.excluded),
    unresolved_state_count_decimal: String(totals.unresolved),
    interval_divergent_state_count_decimal: String(totals.intervalDivergent),
    exact_reachable_divergent_state_count_decimal: String(totals.exactDivergent),
    provably_unreachable_divergent_state_count_decimal: String(totals.excludedDivergent),
    unresolved_divergent_state_count_decimal: String(totals.unresolvedDivergent),
    exact_reachable_divergent_ratio: ratio(totals.exactDivergent, totals.intervalDivergent),
    maximum_lattice_gcd: channels.reduce((maximum, row) => Math.max(maximum, row.lattice_gcd), 0),
    channel_lattice_gcds: channels.map((row) => row.lattice_gcd),
    channel_proof_statuses: channels.map((row) => row.proof_status),
    channel_certified_reachable_state_counts_decimal: channels.map((row) => String(row.certified_reachable_state_count)),
    channel_provably_unreachable_state_counts_decimal: channels.map((row) => String(row.provably_unreachable_state_count)),
    channel_unresolved_state_counts_decimal: channels.map((row) => String(row.unresolved_state_count)),
    channel_exact_reachable_divergent_state_counts_decimal: channels.map((row) => String(row.exact_reachable_divergent_state_count)),
    channel_provably_unreachable_divergent_state_counts_decimal: channels.map((row) => String(row.provably_unreachable_divergent_state_count)),
    channel_unresolved_divergent_state_counts_decimal: channels.map((row) => String(row.unresolved_divergent_state_count)),
    channel_first_exact_reachable_divergent_accumulators_decimal: channels.map((row) => row.first_exact_reachable_divergent_accumulator_decimal),
    top_channels: topChannels,
    source_witness_ledger_sha256: witness.witness_ledger_sha256,
    source_rounding_equivalence_ledger_sha256: equivalence.equivalence_ledger_sha256,
    ledger_bytes: ledgerBytes,
    channels,
  };
}

function analyzeChannel(channelIndex, termCount, inputSpan, minimumDecimal, maximumDecimal, denominations, segments) {
  const minimum = BigInt(minimumDecimal);
  const maximum = BigInt(maximumDecimal);
  assert(minimum <= maximum, "Reachability accumulator interval is non-monotone.");
  const intervalStateCount = maximum - minimum + 1n;
  const latticeGcd = denominations.reduce((value, row) => gcd(value, BigInt(row.absolute_centered_weight)), 0n);
  const expectedSpan = denominations.reduce((sum, row) => sum + BigInt(row.absolute_centered_weight) * BigInt(inputSpan) * BigInt(row.term_count), 0n);
  assert(expectedSpan === maximum - minimum, "Stored-weight bounded-sum span does not reproduce the accumulator interval.");
  const proof = coverageProof(denominations, inputSpan, latticeGcd);
  const compatible = latticeGcd === 0n ? 1n : proof.total_lattice_step_count + 1n;
  const certified = proof.proof_status === "partial_endpoint_bands" ? 2n * (proof.certified_prefix_lattice_step_count + 1n) : compatible;
  const excluded = intervalStateCount - compatible;
  const unresolved = compatible - certified;
  assert(certified >= 0n && excluded >= 0n && unresolved >= 0n && certified + excluded + unresolved === intervalStateCount, "Reachability state partition failed.");
  const divergence = intersectDivergence(minimum, maximum, latticeGcd, proof, segments);
  assert(divergence.interval === divergence.exact + divergence.excluded + divergence.unresolved, "Reachability divergent-state partition failed.");
  return {
    channel_index: channelIndex,
    term_count: termCount,
    nonzero_term_count: denominations.reduce((sum, row) => sum + row.term_count, 0),
    input_code_span: inputSpan,
    lattice_gcd: Number(latticeGcd),
    proof_status: proof.proof_status,
    coverage_failure_step_index: proof.coverage_failure_step_index,
    total_lattice_step_count: proof.total_lattice_step_count,
    certified_prefix_lattice_step_count: proof.certified_prefix_lattice_step_count,
    interval_state_count: intervalStateCount,
    lattice_compatible_state_count: compatible,
    certified_reachable_state_count: certified,
    provably_unreachable_state_count: excluded,
    unresolved_state_count: unresolved,
    interval_divergent_state_count: divergence.interval,
    exact_reachable_divergent_state_count: divergence.exact,
    provably_unreachable_divergent_state_count: divergence.excluded,
    unresolved_divergent_state_count: divergence.unresolved,
    first_exact_reachable_divergent_accumulator_decimal: divergence.first ? String(divergence.first.accumulator) : null,
    first_default_output_code: divergence.first?.default ?? null,
    first_single_output_code: divergence.first?.single ?? null,
    last_exact_reachable_divergent_accumulator_decimal: divergence.last == null ? null : String(divergence.last),
    denomination_coverage_steps: proof.steps,
    minimum,
    maximum,
  };
}

function coverageProof(denominations, inputSpan, latticeGcd) {
  if (!denominations.length) return { proof_status: "singleton", coverage_failure_step_index: null, total_lattice_step_count: 0n, certified_prefix_lattice_step_count: 0n, steps: [] };
  assert(latticeGcd > 0n, "Nonzero denominations have zero gcd.");
  let reachable = 0n;
  let total = 0n;
  let failure = null;
  const steps = denominations.map((row, index) => {
    const denomination = BigInt(row.absolute_centered_weight) / latticeGcd;
    const capacity = BigInt(inputSpan) * BigInt(row.term_count);
    const contribution = denomination * capacity;
    total += contribution;
    const before = reachable;
    let status;
    if (failure != null) status = "after_gap_not_used";
    else if (denomination <= reachable + 1n) { reachable += contribution; status = "extends_prefix"; }
    else { failure = index; status = "first_gap"; }
    return {
      absolute_centered_weight: row.absolute_centered_weight,
      normalized_denomination: Number(denomination),
      term_count: row.term_count,
      aggregate_coefficient_capacity_decimal: String(capacity),
      reachable_prefix_before_decimal: String(before),
      reachable_prefix_after_decimal: String(reachable),
      coverage_status: status,
    };
  });
  const proofStatus = failure == null ? latticeGcd === 1n ? "complete_integer_interval" : "complete_modular_lattice" : "partial_endpoint_bands";
  assert(failure != null || reachable === total, "Complete bounded coverage does not span the lattice.");
  return { proof_status: proofStatus, coverage_failure_step_index: failure, total_lattice_step_count: total, certified_prefix_lattice_step_count: reachable, steps };
}

function intersectDivergence(minimum, maximum, latticeGcd, proof, segments) {
  const modulus = latticeGcd || 1n;
  const prefixMaximum = minimum + modulus * proof.certified_prefix_lattice_step_count;
  const suffixMinimum = maximum - modulus * proof.certified_prefix_lattice_step_count;
  const result = { interval: 0n, exact: 0n, excluded: 0n, unresolved: 0n, first: null, last: null };
  for (const segment of segments.filter((row) => row.divergent)) {
    const start = BigInt(segment.accumulator_minimum_decimal);
    const end = BigInt(segment.accumulator_maximum_decimal);
    const interval = end - start + 1n;
    const compatible = countProgression(start, end, minimum, modulus);
    let exact = compatible;
    if (proof.proof_status === "partial_endpoint_bands") {
      exact = countProgression(start, minBigInt(end, prefixMaximum), minimum, modulus)
        + countProgression(maxBigInt(start, suffixMinimum), end, minimum, modulus);
    }
    const excluded = interval - compatible;
    const unresolved = compatible - exact;
    result.interval += interval;
    result.exact += exact;
    result.excluded += excluded;
    result.unresolved += unresolved;
    if (exact > 0n) {
      const first = firstCertified(start, end, minimum, modulus, proof.proof_status, prefixMaximum, suffixMinimum);
      const last = lastCertified(start, end, minimum, modulus, proof.proof_status, prefixMaximum, suffixMinimum);
      assert(first != null && last != null, "Exact divergent witness is missing.");
      if (!result.first) result.first = { accumulator: first, default: segment.default_output_code, single: segment.single_output_code };
      result.last = last;
    }
  }
  return result;
}

function aggregateWitness(row, accumulator) {
  const gcdValue = BigInt(row.lattice_gcd);
  if (!gcdValue) return [];
  const target = (accumulator - row.minimum) / gcdValue;
  const partial = row.proof_status === "partial_endpoint_bands";
  const complement = partial && target > row.certified_prefix_lattice_step_count;
  let remaining = complement ? row.total_lattice_step_count - target : target;
  const usable = partial ? row.denomination_coverage_steps.findIndex((step) => step.coverage_status !== "extends_prefix") : row.denomination_coverage_steps.length;
  const limit = usable < 0 ? row.denomination_coverage_steps.length : usable;
  const coefficients = Array(row.denomination_coverage_steps.length).fill(0n);
  for (let index = limit - 1; index >= 0; index -= 1) {
    const step = row.denomination_coverage_steps[index];
    const denomination = BigInt(step.normalized_denomination);
    const capacity = BigInt(step.aggregate_coefficient_capacity_decimal);
    coefficients[index] = minBigInt(capacity, remaining / denomination);
    remaining -= coefficients[index] * denomination;
  }
  assert(remaining === 0n, "Bounded coverage proof did not construct the selected witness.");
  if (complement) row.denomination_coverage_steps.forEach((step, index) => { coefficients[index] = BigInt(step.aggregate_coefficient_capacity_decimal) - coefficients[index]; });
  return row.denomination_coverage_steps.flatMap((step, index) => coefficients[index] > 0n ? [{
    absolute_centered_weight: step.absolute_centered_weight,
    normalized_denomination: step.normalized_denomination,
    term_count: step.term_count,
    aggregate_input_code_delta_decimal: String(coefficients[index]),
    aggregate_capacity_decimal: step.aggregate_coefficient_capacity_decimal,
  }] : []);
}

function publicChannel(row) {
  const witnessCount = row.first_exact_reachable_divergent_accumulator_decimal == null ? 0 : aggregateWitness(row, BigInt(row.first_exact_reachable_divergent_accumulator_decimal)).length;
  return {
    channel_index: row.channel_index,
    term_count: row.term_count,
    nonzero_term_count: row.nonzero_term_count,
    input_code_span: row.input_code_span,
    lattice_gcd: row.lattice_gcd,
    proof_status: row.proof_status,
    ...(row.coverage_failure_step_index == null ? {} : { coverage_failure_step_index: row.coverage_failure_step_index }),
    total_lattice_step_count_decimal: String(row.total_lattice_step_count),
    certified_prefix_lattice_step_count_decimal: String(row.certified_prefix_lattice_step_count),
    interval_state_count_decimal: String(row.interval_state_count),
    lattice_compatible_state_count_decimal: String(row.lattice_compatible_state_count),
    certified_reachable_state_count_decimal: String(row.certified_reachable_state_count),
    provably_unreachable_state_count_decimal: String(row.provably_unreachable_state_count),
    unresolved_state_count_decimal: String(row.unresolved_state_count),
    interval_divergent_state_count_decimal: String(row.interval_divergent_state_count),
    exact_reachable_divergent_state_count_decimal: String(row.exact_reachable_divergent_state_count),
    provably_unreachable_divergent_state_count_decimal: String(row.provably_unreachable_divergent_state_count),
    unresolved_divergent_state_count_decimal: String(row.unresolved_divergent_state_count),
    ...(row.first_exact_reachable_divergent_accumulator_decimal == null ? {} : {
      first_exact_reachable_divergent_accumulator_decimal: row.first_exact_reachable_divergent_accumulator_decimal,
      first_default_output_code: row.first_default_output_code,
      first_single_output_code: row.first_single_output_code,
    }),
    ...(row.last_exact_reachable_divergent_accumulator_decimal == null ? {} : {
      last_exact_reachable_divergent_accumulator_decimal: row.last_exact_reachable_divergent_accumulator_decimal,
    }),
    denomination_group_count: row.denomination_coverage_steps.length,
    first_exact_reachable_witness_group_count: witnessCount,
  };
}

function denominationsFromTerms(termRows) {
  assert(termRows instanceof Int16Array && termRows.length % 3 === 0, "Kernel term rows are unavailable for reachability reconstruction.");
  const counts = new Map();
  for (let index = 0; index < termRows.length; index += 3) {
    const absolute = Math.abs(termRows[index]);
    if (absolute) counts.set(absolute, (counts.get(absolute) || 0) + 1);
  }
  return [...counts].sort((left, right) => left[0] - right[0]).map(([absolute_centered_weight, term_count]) => ({ absolute_centered_weight, term_count }));
}

function reachabilityLedger(witnessDigest, equivalenceDigest, opIndex, channels) {
  assert(/^[a-f0-9]{64}$/.test(witnessDigest || "") && /^[a-f0-9]{64}$/.test(equivalenceDigest || ""), "Reachability source digest identity is invalid.");
  const stepCount = channels.reduce((sum, row) => sum + row.denomination_coverage_steps.length, 0);
  const writer = new BinaryWriter(LEDGER_PREFIX.length + 128 + channels.length * 24 * 8 + stepCount * 7 * 8);
  writer.writeBytes(LEDGER_PREFIX);
  writer.writeBytes(new TextEncoder().encode(witnessDigest));
  writer.writeBytes(new TextEncoder().encode(equivalenceDigest));
  for (const row of channels) {
    for (const value of [opIndex, row.channel_index, row.term_count, row.nonzero_term_count, row.input_code_span, row.lattice_gcd, proofStatusCode(row.proof_status), row.coverage_failure_step_index ?? MISSING_U64]) writer.writeU64(value);
    for (const value of [row.total_lattice_step_count, row.certified_prefix_lattice_step_count, row.interval_state_count, row.lattice_compatible_state_count, row.certified_reachable_state_count, row.provably_unreachable_state_count, row.unresolved_state_count, row.interval_divergent_state_count, row.exact_reachable_divergent_state_count, row.provably_unreachable_divergent_state_count, row.unresolved_divergent_state_count]) writer.writeU64(value);
    writer.writeI64(row.first_exact_reachable_divergent_accumulator_decimal ?? MISSING_I64);
    writer.writeI64(row.first_default_output_code ?? MISSING_I64);
    writer.writeI64(row.first_single_output_code ?? MISSING_I64);
    writer.writeI64(row.last_exact_reachable_divergent_accumulator_decimal ?? MISSING_I64);
    writer.writeU64(row.denomination_coverage_steps.length);
    for (const step of row.denomination_coverage_steps) {
      for (const value of [step.absolute_centered_weight, step.normalized_denomination, step.term_count, step.aggregate_coefficient_capacity_decimal, step.reachable_prefix_before_decimal, step.reachable_prefix_after_decimal, coverageStatusCode(step.coverage_status)]) writer.writeU64(value);
    }
  }
  return writer.finish();
}

class BinaryWriter {
  constructor(length) { this.bytes = new Uint8Array(length); this.view = new DataView(this.bytes.buffer); this.offset = 0; }
  writeBytes(value) { this.bytes.set(value, this.offset); this.offset += value.length; }
  writeU64(value) { this.view.setBigUint64(this.offset, BigInt(value), true); this.offset += 8; }
  writeI64(value) { this.view.setBigInt64(this.offset, BigInt(value), true); this.offset += 8; }
  finish() { assert(this.offset === this.bytes.length, "Reachability ledger length mismatch."); return this.bytes; }
}

function validateOpShape(row) {
  assert(Number.isInteger(row.op_index) && row.op_index >= 0, "Accumulator-reachability op index is invalid.");
  assert(["assessed", "not_assessed"].includes(row.assessment_status), `Accumulator-reachability op #${row.op_index} status is invalid.`);
  if (row.assessment_status !== "assessed") return;
  const count = Number(row.assessed_channel_count);
  for (const key of ["channel_lattice_gcds", "channel_proof_statuses", "channel_certified_reachable_state_counts_decimal", "channel_provably_unreachable_state_counts_decimal", "channel_unresolved_state_counts_decimal", "channel_exact_reachable_divergent_state_counts_decimal", "channel_provably_unreachable_divergent_state_counts_decimal", "channel_unresolved_divergent_state_counts_decimal", "channel_first_exact_reachable_divergent_accumulators_decimal"]) assert(Array.isArray(row[key]) && row[key].length === count, `Accumulator-reachability ${key} cardinality mismatch at op #${row.op_index}.`);
  assert(row.channel_proof_statuses.every((status) => ["singleton", "complete_integer_interval", "complete_modular_lattice", "partial_endpoint_bands"].includes(status)), `Accumulator-reachability proof status is invalid at op #${row.op_index}.`);
  assert(/^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || "") && /^[a-f0-9]{64}$/.test(row.source_rounding_equivalence_ledger_sha256 || "") && /^[a-f0-9]{64}$/.test(row.reachability_ledger_sha256 || ""), `Accumulator-reachability digest is invalid at op #${row.op_index}.`);
  const interval = BigInt(row.interval_state_count_decimal);
  const certified = BigInt(row.certified_reachable_state_count_decimal);
  const excluded = BigInt(row.provably_unreachable_state_count_decimal);
  const unresolved = BigInt(row.unresolved_state_count_decimal);
  assert(interval === certified + excluded + unresolved, `Accumulator-reachability state conservation failed at op #${row.op_index}.`);
  const divergent = BigInt(row.interval_divergent_state_count_decimal);
  const exact = BigInt(row.exact_reachable_divergent_state_count_decimal);
  const divergentExcluded = BigInt(row.provably_unreachable_divergent_state_count_decimal);
  const divergentUnresolved = BigInt(row.unresolved_divergent_state_count_decimal);
  assert(divergent === exact + divergentExcluded + divergentUnresolved, `Accumulator-reachability divergent-state conservation failed at op #${row.op_index}.`);
  assert(Array.isArray(row.top_channels) && row.top_channels.length <= TOP_LIMIT, `Accumulator-reachability top channels are invalid at op #${row.op_index}.`);
}

function compareOpRow(actual, expected) {
  const scalarKeys = ["op_index", "op_name", "assessment_status", "not_assessed_reason", "assessed_channel_count", "complete_integer_interval_channel_count", "complete_modular_lattice_channel_count", "partial_band_channel_count", "singleton_channel_count", "exact_reachable_divergent_channel_count", "unresolved_divergent_channel_count", "interval_only_divergent_channel_count", "interval_state_count_decimal", "lattice_compatible_state_count_decimal", "certified_reachable_state_count_decimal", "provably_unreachable_state_count_decimal", "unresolved_state_count_decimal", "interval_divergent_state_count_decimal", "exact_reachable_divergent_state_count_decimal", "provably_unreachable_divergent_state_count_decimal", "unresolved_divergent_state_count_decimal", "maximum_lattice_gcd", "source_witness_ledger_sha256", "source_rounding_equivalence_ledger_sha256"];
  for (const key of scalarKeys) assert(actual[key] === expected[key], `Accumulator-reachability ${key} mismatch at op #${actual.op_index}.`);
  assert(close(actual.exact_reachable_divergent_ratio, expected.exact_reachable_divergent_ratio), `Accumulator-reachability ratio mismatch at op #${actual.op_index}.`);
  for (const key of ["channel_lattice_gcds", "channel_proof_statuses", "channel_certified_reachable_state_counts_decimal", "channel_provably_unreachable_state_counts_decimal", "channel_unresolved_state_counts_decimal", "channel_exact_reachable_divergent_state_counts_decimal", "channel_provably_unreachable_divergent_state_counts_decimal", "channel_unresolved_divergent_state_counts_decimal", "channel_first_exact_reachable_divergent_accumulators_decimal", "top_channels"]) assert(JSON.stringify(actual[key]) === JSON.stringify(expected[key]), `Accumulator-reachability ${key} mismatch at op #${actual.op_index}.`);
}

function sumChannels(rows) {
  return rows.reduce((sum, row) => ({
    interval: sum.interval + row.interval_state_count,
    compatible: sum.compatible + row.lattice_compatible_state_count,
    certified: sum.certified + row.certified_reachable_state_count,
    excluded: sum.excluded + row.provably_unreachable_state_count,
    unresolved: sum.unresolved + row.unresolved_state_count,
    intervalDivergent: sum.intervalDivergent + row.interval_divergent_state_count,
    exactDivergent: sum.exactDivergent + row.exact_reachable_divergent_state_count,
    excludedDivergent: sum.excludedDivergent + row.provably_unreachable_divergent_state_count,
    unresolvedDivergent: sum.unresolvedDivergent + row.unresolved_divergent_state_count,
  }), emptySums());
}

function sumRows(rows) {
  return rows.reduce((sum, row) => ({
    interval: sum.interval + BigInt(row.interval_state_count_decimal),
    compatible: sum.compatible + BigInt(row.lattice_compatible_state_count_decimal),
    certified: sum.certified + BigInt(row.certified_reachable_state_count_decimal),
    excluded: sum.excluded + BigInt(row.provably_unreachable_state_count_decimal),
    unresolved: sum.unresolved + BigInt(row.unresolved_state_count_decimal),
    intervalDivergent: sum.intervalDivergent + BigInt(row.interval_divergent_state_count_decimal),
    exactDivergent: sum.exactDivergent + BigInt(row.exact_reachable_divergent_state_count_decimal),
    excludedDivergent: sum.excludedDivergent + BigInt(row.provably_unreachable_divergent_state_count_decimal),
    unresolvedDivergent: sum.unresolvedDivergent + BigInt(row.unresolved_divergent_state_count_decimal),
  }), emptySums());
}

function emptySums() { return { interval: 0n, compatible: 0n, certified: 0n, excluded: 0n, unresolved: 0n, intervalDivergent: 0n, exactDivergent: 0n, excludedDivergent: 0n, unresolvedDivergent: 0n }; }
function compareChannels(left, right) { return compareBig(right.exact_reachable_divergent_state_count, left.exact_reachable_divergent_state_count) || compareBig(right.unresolved_divergent_state_count, left.unresolved_divergent_state_count) || right.lattice_gcd - left.lattice_gcd || left.channel_index - right.channel_index; }
function compareOps(left, right) { return compareBig(BigInt(right.exact_reachable_divergent_state_count_decimal), BigInt(left.exact_reachable_divergent_state_count_decimal)) || compareBig(BigInt(right.unresolved_divergent_state_count_decimal), BigInt(left.unresolved_divergent_state_count_decimal)) || left.op_index - right.op_index; }
function compareBig(left, right) { return left < right ? -1 : left > right ? 1 : 0; }
function countProgression(start, end, anchor, modulus) { const first = firstProgression(start, end, anchor, modulus); if (first == null) return 0n; const last = lastProgression(start, end, anchor, modulus); return (last - first) / modulus + 1n; }
function firstProgression(start, end, anchor, modulus) { if (start > end) return null; const residue = mod(start - anchor, modulus); const value = start + (residue === 0n ? 0n : modulus - residue); return value <= end ? value : null; }
function lastProgression(start, end, anchor, modulus) { if (start > end) return null; const value = end - mod(end - anchor, modulus); return value >= start ? value : null; }
function firstCertified(start, end, anchor, modulus, status, prefixMaximum, suffixMinimum) { if (status !== "partial_endpoint_bands") return firstProgression(start, end, anchor, modulus); return firstProgression(start, minBigInt(end, prefixMaximum), anchor, modulus) ?? firstProgression(maxBigInt(start, suffixMinimum), end, anchor, modulus); }
function lastCertified(start, end, anchor, modulus, status, prefixMaximum, suffixMinimum) { if (status !== "partial_endpoint_bands") return lastProgression(start, end, anchor, modulus); return lastProgression(maxBigInt(start, suffixMinimum), end, anchor, modulus) ?? lastProgression(start, minBigInt(end, prefixMaximum), anchor, modulus); }
function gcd(left, right) { while (right) [left, right] = [right, left % right]; return left; }
function mod(value, modulus) { const result = value % modulus; return result < 0n ? result + modulus : result; }
function minBigInt(left, right) { return left < right ? left : right; }
function maxBigInt(left, right) { return left > right ? left : right; }
function proofStatusCode(status) { return { singleton: 0, complete_integer_interval: 1, complete_modular_lattice: 2, partial_endpoint_bands: 3 }[status] ?? MISSING_U64; }
function coverageStatusCode(status) { return { extends_prefix: 0, first_gap: 1, after_gap_not_used: 2 }[status] ?? MISSING_U64; }
function ratio(numerator, denominator) { return denominator ? Number(numerator) / Number(denominator) : 0; }
function sumNumber(rows, key) { return rows.reduce((sum, row) => sum + Number(row[key] || 0), 0); }
function maxOptional(rows, key) { return rows.reduce((maximum, row) => row[key] == null ? maximum : maximum == null ? Number(row[key]) : Math.max(maximum, Number(row[key])), null); }
function compareDecimal(actual, expected, label) { assert(BigInt(actual) === expected, `${label} mismatch.`); }
function close(left, right) { return Math.abs(Number(left) - Number(right)) <= 1e-12; }
function assert(condition, message) { if (!condition) throw new Error(message); }
