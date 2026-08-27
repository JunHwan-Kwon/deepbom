import { sha256Hex } from "./hash.js";
import {
  reconstructKernelChannel,
  reconstructKernelWitnessAnalysis,
  validateKernelWitnessAnalysis,
} from "./kernel-witness.js";
import {
  multiplyByQuantizedMultiplierDefault,
  multiplyByQuantizedMultiplierSingleRounding,
} from "./quantization-math.js";

export const ROUNDING_EQUIVALENCE_SCHEMA = "deepbom.rounding_equivalence.v1";
const METHOD_VERSION = "2026-07-17.1";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LEDGER_PREFIX = new TextEncoder().encode("deepbom.rounding_equivalence.v1\0");
const MISSING_I64 = -(1n << 63n);
const TOP_LIMIT = 16;
const HISTOGRAM_LABELS = ["0%", "(0,0.01%]", "(0.01%,0.1%]", "(0.1%,1%]", "(1%,10%]", "(10%,50%]", "(50%,100%]"];

export function createRoundingEquivalenceController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getContext,
  jumpToGraphOp,
  onDownload,
}) {
  let analysis = null;
  let modelBytes = null;
  let evidence = null;
  let selectedOpIndex = null;
  let selectedChannelIndex = null;
  let field = "exposure";
  let worker = null;
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "rounding_equivalence.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    worker?.terminate();
    worker = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    const context = getContext?.() || {};
    analysis = explicitAnalysis || context.analysis || null;
    modelBytes = context.modelBytes || null;
    evidence = analysis?.rounding_equivalence || null;
    if (!evidence || !modelBytes || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedOpIndex = null;
      selectedChannelIndex = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      validateRoundingEquivalenceShape(evidence);
      const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = evidence.equivalence_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
        selectedChannelIndex = null;
      }
      renderSummary(summary, evidence);
      renderBody();
      setStatus(status, evidence.assessed_op_count ? "source arithmetic / verification pending" : humanize(evidence.status), evidence.divergent_channel_count ? "watch" : "ok");
      if (evidence.assessed_op_count && typeof Worker === "function") {
        worker = new Worker(new URL("./rounding-equivalence-worker.js", import.meta.url), { type: "module" });
        worker.onmessage = (event) => {
          if (token !== renderToken) return;
          setStatus(status, event.data?.ok ? "independently verified" : `integrity error: ${event.data?.error || "verification failed"}`, event.data?.ok ? evidence.divergent_channel_count ? "watch" : "ok" : "risk");
          worker?.terminate();
          worker = null;
        };
        worker.onerror = (event) => {
          if (token !== renderToken) return;
          setStatus(status, `integrity error: ${event.message || "worker failed"}`, "risk");
          worker?.terminate();
          worker = null;
        };
        worker.postMessage({ analysis, modelBytes });
      }
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Rounding-equivalence evidence rejected: ${error.message}`, "risk"));
      setStatus(status, "evidence rejected", "risk");
    }
  }

  function renderBody() {
    if (!body || !evidence) return;
    resizeObserver?.disconnect();
    const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
    if (!assessed.length) {
      body.replaceChildren(messageNode("No fixed-point channel interval was assessable."), unassessedTable(evidence.ops));
      return;
    }
    const row = assessed.find((candidate) => candidate.op_index === selectedOpIndex) || assessed[0];
    selectedOpIndex = row.op_index;
    const suggested = row.top_channels?.[0]?.channel_index ?? firstDivergentChannel(row) ?? 0;
    if (!Number.isInteger(selectedChannelIndex) || selectedChannelIndex < 0 || selectedChannelIndex >= row.assessed_channel_count) selectedChannelIndex = suggested;

    const toolbar = element("div", "rounding-equivalence-toolbar");
    const opSelect = element("select", "rounding-equivalence-op-select");
    opSelect.setAttribute("aria-label", "Rounding-equivalence operator");
    for (const candidate of assessed) {
      const option = new Option(`#${candidate.op_index} ${candidate.op_name} / ${formatNumber(candidate.divergent_channel_count)} divergent`, String(candidate.op_index));
      option.selected = candidate.op_index === row.op_index;
      opSelect.append(option);
    }
    opSelect.addEventListener("change", () => {
      selectedOpIndex = Number(opSelect.value);
      selectedChannelIndex = null;
      renderBody();
    });
    const channelLabel = element("label", "rounding-equivalence-channel-control", "Channel");
    const channelInput = element("input", "");
    channelInput.type = "number";
    channelInput.min = "0";
    channelInput.max = String(row.assessed_channel_count - 1);
    channelInput.step = "1";
    channelInput.value = String(selectedChannelIndex);
    channelInput.setAttribute("aria-label", `Output channel 0 through ${row.assessed_channel_count - 1}`);
    channelInput.addEventListener("change", () => {
      selectedChannelIndex = clampInteger(channelInput.value, 0, row.assessed_channel_count - 1);
      renderBody();
    });
    channelLabel.append(channelInput);
    toolbar.append(
      opSelect,
      channelLabel,
      segmentedControl("Rounding-equivalence field", [["exposure", "Exposure"], ["delta", "Max delta"], ["regions", "Regions"]], field, (value) => {
        field = value;
        renderBody();
      }),
    );

    const selected = reconstructRoundingEquivalenceChannel(analysis, modelBytes, row.op_index, selectedChannelIndex, true);
    compareSelected(row, selected);
    const main = element("div", "rounding-equivalence-main");
    const plots = element("div", "rounding-equivalence-plots");
    const heatWrap = element("div", "rounding-equivalence-canvas-wrap");
    const heatCanvas = element("canvas", "rounding-equivalence-canvas");
    heatCanvas.tabIndex = 0;
    heatCanvas.setAttribute("role", "img");
    heatCanvas.setAttribute("aria-label", `${field} atlas for operator ${row.op_index}`);
    const heatTooltip = element("div", "rounding-equivalence-tooltip");
    heatTooltip.hidden = true;
    heatWrap.append(heatCanvas, heatTooltip);
    installHeatInteraction(heatCanvas, heatTooltip, row, () => field, (channel) => {
      selectedChannelIndex = channel;
      renderBody();
    });
    const traceWrap = element("div", "rounding-equivalence-trace-wrap");
    const traceCanvas = element("canvas", "rounding-equivalence-trace");
    traceCanvas.tabIndex = 0;
    traceCanvas.setAttribute("role", "img");
    traceCanvas.setAttribute("aria-label", `Exact accumulator-output trace for operator ${row.op_index} channel ${selectedChannelIndex}`);
    const traceTooltip = element("div", "rounding-equivalence-tooltip trace");
    traceTooltip.hidden = true;
    traceWrap.append(traceCanvas, traceTooltip);
    installTraceInteraction(traceCanvas, traceTooltip, selected);
    plots.append(heatWrap, traceWrap);

    const details = element("div", "rounding-equivalence-details");
    const selectedDownload = commandButton("Certificate JSON", "Download the selected channel equivalence certificate", () => onDownload?.(
      selectedChannelExport(analysis, evidence, row, selected),
      `rounding_equivalence_op_${row.op_index}_channel_${selectedChannelIndex}.json`,
    ));
    const graphButton = commandButton("Graph op", "Open the selected operator in the graph workspace", () => jumpToGraphOp?.(row.op_index));
    const actions = element("div", "rounding-equivalence-actions");
    actions.append(selectedDownload, graphButton);
    const firstWitness = selected.first_divergent_accumulator_decimal == null
      ? "none; outputs are equal for the complete interval hull"
      : `${formatInteger(selected.first_divergent_accumulator_decimal)} -> default ${selected.first_default_output_code}, single ${selected.first_single_output_code}`;
    details.append(
      definitionTable([
        ["Coordinate", `#${row.op_index} ${row.op_name}, output channel ${selectedChannelIndex}`],
        ["Accumulator interval", `${formatInteger(selected.post_bias_minimum_decimal)} .. ${formatInteger(selected.post_bias_maximum_decimal)} (${formatInteger(selected.interval_state_count_decimal)} states)`],
        ["Certificate", selected.divergent_state_count === 0n ? "bit-exact equivalent over the complete interval hull" : `${formatInteger(selected.divergent_state_count_decimal)} divergent states / ${formatPercent(selected.divergent_state_ratio)}`],
        ["Direction", `${formatInteger(selected.default_lower_state_count_decimal)} default lower / ${formatInteger(selected.default_higher_state_count_decimal)} default higher`],
        ["Partition", `${formatNumber(selected.pair_segment_count)} exact pair segments / ${formatNumber(selected.divergent_region_count)} divergent regions`],
        ["First counterexample", firstWitness],
        ["Maximum output delta", `${selected.maximum_absolute_output_delta} code`],
        ["Encodings", `default q=${selected.default_quantized_multiplier}, shift=${selected.default_shift}; single q=${selected.single_quantized_multiplier}, shift=${selected.single_shift}`],
        ["Certificate ledger", row.equivalence_ledger_sha256],
      ]),
      actions,
    );
    main.append(plots, details);

    const proof = element("p", "rounding-equivalence-proof", evidence.equivalence_proof);
    const boundary = element("p", "rounding-equivalence-boundary", evidence.interpretation_boundary);
    body.replaceChildren(
      toolbar,
      main,
      histogramTable(evidence.divergence_histogram),
      rankingTable(evidence, row.op_index, (opIndex) => {
        selectedOpIndex = opIndex;
        selectedChannelIndex = null;
        renderBody();
      }, jumpToGraphOp),
      proof,
      boundary,
      unassessedTable(evidence.ops),
    );
    requestAnimationFrame(() => {
      drawRoundingEquivalenceCanvas(heatCanvas, row, field, selectedChannelIndex);
      drawRoundingTraceCanvas(traceCanvas, selected);
    });
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => {
        drawRoundingEquivalenceCanvas(heatCanvas, row, field, selectedChannelIndex);
        drawRoundingTraceCanvas(traceCanvas, selected);
      });
      resizeObserver.observe(plots);
    }
  }

  return { render };
}

export function validateRoundingEquivalenceShape(evidence) {
  assert(evidence?.schema === ROUNDING_EQUIVALENCE_SCHEMA, "Rounding-equivalence schema mismatch.");
  assert(evidence.method_version === METHOD_VERSION, "Rounding-equivalence method mismatch.");
  assert(evidence.source_commit === SOURCE_COMMIT, "Rounding-equivalence source commit mismatch.");
  assert(Array.isArray(evidence.ops) && evidence.ops.length === Number(evidence.candidate_op_count), "Rounding-equivalence op rows are incomplete.");
  assert(Array.isArray(evidence.equivalence_ranking_op_indices), "Rounding-equivalence ranking is unavailable.");
  assert(Array.isArray(evidence.divergence_histogram) && evidence.divergence_histogram.map((bin) => bin.label).join("|") === HISTOGRAM_LABELS.join("|"), "Rounding-equivalence histogram is invalid.");
  for (const row of evidence.ops) {
    assert(Number.isInteger(row.op_index) && typeof row.op_name === "string", "Rounding-equivalence op identity is invalid.");
    if (row.assessment_status !== "assessed") continue;
    const count = Number(row.assessed_channel_count);
    for (const key of [
      "channel_interval_state_counts_decimal",
      "channel_divergent_state_counts_decimal",
      "channel_default_lower_state_counts_decimal",
      "channel_default_higher_state_counts_decimal",
      "channel_pair_segment_counts",
      "channel_divergent_region_counts",
      "channel_maximum_absolute_output_deltas",
      "channel_first_divergent_accumulators_decimal",
      "channel_first_default_output_codes",
      "channel_first_single_output_codes",
    ]) assert(Array.isArray(row[key]) && row[key].length === count, `Rounding-equivalence ${key} cardinality mismatch at op #${row.op_index}.`);
    assert(row.channel_pair_segment_counts.every((value) => Number.isInteger(value) && value >= 1 && value <= 511), `Pair-segment bound failed at op #${row.op_index}.`);
    assert(row.channel_maximum_absolute_output_deltas.every((value) => Number.isSafeInteger(value) && value >= 0), `Output delta is invalid at op #${row.op_index}.`);
    assert(/^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || "") && /^[a-f0-9]{64}$/.test(row.source_requantization_ledger_sha256 || "") && /^[a-f0-9]{64}$/.test(row.equivalence_ledger_sha256 || ""), `Rounding-equivalence digest is invalid at op #${row.op_index}.`);
    assert(Array.isArray(row.top_channels) && row.top_channels.length <= TOP_LIMIT, `Top-channel set is invalid at op #${row.op_index}.`);
  }
  return evidence;
}

export function validateRoundingEquivalenceAnalysis(analysis, modelBytes) {
  validateKernelWitnessAnalysis(analysis, modelBytes);
  const expected = reconstructRoundingEquivalenceAnalysis(analysis, modelBytes);
  validateRoundingEquivalenceAgainstReconstruction(analysis?.rounding_equivalence, expected);
  return expected;
}

export function validateRoundingEquivalenceAgainstReconstruction(evidence, expected) {
  const actual = validateRoundingEquivalenceShape(evidence);
  compareSummary(actual, expected);
  assert(actual.ops.length === expected.ops.length, "Rounding-equivalence op count mismatch.");
  actual.ops.forEach((row, index) => compareOpRow(row, expected.ops[index]));
  return expected;
}

export async function validateRoundingEquivalenceDigests(analysis, modelBytes) {
  const reconstructed = validateRoundingEquivalenceAnalysis(analysis, modelBytes);
  await validateRoundingEquivalenceDigestsAgainstReconstruction(analysis.rounding_equivalence, reconstructed);
  return reconstructed;
}

export async function validateRoundingEquivalenceDigestsAgainstReconstruction(evidence, reconstructed) {
  validateRoundingEquivalenceAgainstReconstruction(evidence, reconstructed);
  for (let index = 0; index < reconstructed.ops.length; index += 1) {
    const row = reconstructed.ops[index];
    if (row.assessment_status !== "assessed") continue;
    const digest = await sha256Hex(row.ledger_bytes);
    assert(digest === evidence.ops[index].equivalence_ledger_sha256, `Rounding-equivalence SHA-256 mismatch at op #${row.op_index}.`);
  }
  return reconstructed;
}

export function reconstructRoundingEquivalenceAnalysis(analysis, modelBytes) {
  const witness = reconstructKernelWitnessAnalysis(analysis, modelBytes);
  const sourceWitness = new Map((analysis?.kernel_extremum_witness?.ops || []).map((row) => [Number(row.op_index), row]));
  const requant = new Map((analysis?.requantization_fidelity?.ops || []).map((row) => [Number(row.op_index), row]));
  const ops = witness.ops.map((row) => {
    const source = sourceWitness.get(row.op_index);
    const requantRow = requant.get(row.op_index);
    if (row.assessment_status !== "assessed" || requantRow?.assessment_status !== "assessed") return notAssessedRow(row, source, requantRow);
    return reconstructOp(row, source, requantRow);
  });
  const assessed = ops.filter((row) => row.assessment_status === "assessed");
  const total = sumBigInt(assessed, "interval_state_count_decimal");
  const divergent = sumBigInt(assessed, "divergent_state_count_decimal");
  const allChannels = assessed.flatMap((row) => row.channels);
  return {
    status: !ops.length ? "not_applicable" : assessed.length === ops.length ? "assessed" : assessed.length ? "partial" : "not_assessed",
    candidate_op_count: ops.length,
    assessed_op_count: assessed.length,
    unassessed_op_count: ops.length - assessed.length,
    assessed_channel_count: sumNumber(assessed, "assessed_channel_count"),
    equivalent_channel_count: sumNumber(assessed, "equivalent_channel_count"),
    divergent_channel_count: sumNumber(assessed, "divergent_channel_count"),
    divergent_op_count: assessed.filter((row) => row.divergent_channel_count > 0).length,
    interval_state_count_decimal: String(total),
    divergent_state_count_decimal: String(divergent),
    divergent_state_ratio: ratio(divergent, total),
    default_lower_state_count_decimal: String(sumBigInt(assessed, "default_lower_state_count_decimal")),
    default_higher_state_count_decimal: String(sumBigInt(assessed, "default_higher_state_count_decimal")),
    maximum_absolute_output_delta: maxOptional(assessed, "maximum_absolute_output_delta"),
    pair_segment_count: assessed.reduce((sum, row) => sum + row.channel_pair_segment_counts.reduce((subtotal, value) => subtotal + value, 0), 0),
    divergent_region_count: assessed.reduce((sum, row) => sum + row.channel_divergent_region_counts.reduce((subtotal, value) => subtotal + value, 0), 0),
    divergence_histogram: divergenceHistogram(allChannels),
    equivalence_ranking_op_indices: [...assessed].sort(compareOps).map((row) => row.op_index),
    ops,
  };
}

export function reconstructRoundingEquivalenceChannel(analysis, modelBytes, opIndex, channelIndex, includeSegments = true) {
  const witness = reconstructKernelChannel(analysis, modelBytes, opIndex, channelIndex);
  const requant = analysis?.requantization_fidelity?.ops?.find((row) => Number(row.op_index) === Number(opIndex));
  assert(requant?.assessment_status === "assessed", `Requantization evidence is unavailable at op #${opIndex}.`);
  const sourceWitness = analysis?.kernel_extremum_witness?.ops?.find((row) => Number(row.op_index) === Number(opIndex));
  const activationRange = sourceWitness?.activation_code_range?.map(Number);
  assert(activationRange?.length === 2, `Activation range is unavailable at op #${opIndex}.`);
  return analyzeChannel(
    channelIndex,
    witness.minimum.post_bias_accumulator_decimal,
    witness.maximum.post_bias_accumulator_decimal,
    projectionParameters(requant, activationRange, channelIndex),
    includeSegments,
  );
}

export function reconstructRoundingEquivalenceIntervalChannel(analysis, opIndex, channelIndex, includeSegments = true) {
  const accumulator = analysis?.accumulator_atlas?.ops?.find((row) => Number(row.op_index) === Number(opIndex));
  const requant = analysis?.requantization_fidelity?.ops?.find((row) => Number(row.op_index) === Number(opIndex));
  const sourceWitness = analysis?.kernel_extremum_witness?.ops?.find((row) => Number(row.op_index) === Number(opIndex));
  assert(accumulator?.assessment_status === "assessed", `Accumulator evidence is unavailable at op #${opIndex}.`);
  assert(requant?.assessment_status === "assessed", `Requantization evidence is unavailable at op #${opIndex}.`);
  const channel = Number(channelIndex);
  assert(Number.isInteger(channel) && channel >= 0 && channel < Number(accumulator.assessed_channel_count), `Rounding-equivalence channel ${channelIndex} is outside the op range.`);
  const activationRange = sourceWitness?.activation_code_range?.map(Number);
  assert(activationRange?.length === 2, `Activation range is unavailable at op #${opIndex}.`);
  return analyzeChannel(
    channel,
    accumulator.channel_post_bias_min_decimals[channel],
    accumulator.channel_post_bias_max_decimals[channel],
    projectionParameters(requant, activationRange, channel),
    includeSegments,
  );
}

function reconstructOp(witness, sourceWitness, requant) {
  const count = Number(witness.assessed_channel_count);
  assert(witness.channels?.length === count, `Witness channels are incomplete at op #${witness.op_index}.`);
  const activationRange = sourceWitness?.activation_code_range?.map(Number);
  assert(activationRange?.length === 2 && Number.isSafeInteger(Number(requant.output_zero_point)), `Output contract is incomplete at op #${witness.op_index}.`);
  for (const key of ["channel_quantized_multipliers", "channel_shifts", "channel_single_rounding_quantized_multipliers", "channel_single_rounding_shifts"]) assert(requant[key]?.length === count, `Requantization ${key} is incomplete at op #${witness.op_index}.`);
  const channels = witness.channels.map((channel, index) => analyzeChannel(
    index,
    channel.minimum.post_bias_accumulator_decimal,
    channel.maximum.post_bias_accumulator_decimal,
    projectionParameters(requant, activationRange, index),
    false,
  ));
  const total = sumBigInt(channels, "interval_state_count_decimal");
  const divergent = sumBigInt(channels, "divergent_state_count_decimal");
  const ledger = equivalenceLedger(sourceWitness.witness_ledger_sha256, requant.channel_ledger_sha256, witness.op_index, channels);
  const ranked = [...channels].sort(compareChannels);
  return {
    op_index: witness.op_index,
    op_name: witness.op_name,
    assessment_status: "assessed",
    not_assessed_reason: "",
    assessed_channel_count: count,
    equivalent_channel_count: channels.filter((channel) => channel.divergent_state_count === 0n).length,
    divergent_channel_count: channels.filter((channel) => channel.divergent_state_count > 0n).length,
    interval_state_count_decimal: String(total),
    divergent_state_count_decimal: String(divergent),
    divergent_state_ratio: ratio(divergent, total),
    default_lower_state_count_decimal: String(sumBigInt(channels, "default_lower_state_count_decimal")),
    default_higher_state_count_decimal: String(sumBigInt(channels, "default_higher_state_count_decimal")),
    maximum_absolute_output_delta: maxOptional(channels, "maximum_absolute_output_delta"),
    maximum_pair_segment_count: maxOptional(channels, "pair_segment_count"),
    maximum_divergent_region_count: maxOptional(channels, "divergent_region_count"),
    activation_code_range: activationRange,
    output_zero_point: Number(requant.output_zero_point),
    channel_interval_state_counts_decimal: channels.map((channel) => channel.interval_state_count_decimal),
    channel_divergent_state_counts_decimal: channels.map((channel) => channel.divergent_state_count_decimal),
    channel_default_lower_state_counts_decimal: channels.map((channel) => channel.default_lower_state_count_decimal),
    channel_default_higher_state_counts_decimal: channels.map((channel) => channel.default_higher_state_count_decimal),
    channel_pair_segment_counts: channels.map((channel) => channel.pair_segment_count),
    channel_divergent_region_counts: channels.map((channel) => channel.divergent_region_count),
    channel_maximum_absolute_output_deltas: channels.map((channel) => channel.maximum_absolute_output_delta),
    channel_first_divergent_accumulators_decimal: channels.map((channel) => channel.first_divergent_accumulator_decimal),
    channel_first_default_output_codes: channels.map((channel) => channel.first_default_output_code),
    channel_first_single_output_codes: channels.map((channel) => channel.first_single_output_code),
    top_channels: ranked.slice(0, TOP_LIMIT).map(witnessRow),
    source_witness_ledger_sha256: sourceWitness.witness_ledger_sha256,
    source_requantization_ledger_sha256: requant.channel_ledger_sha256,
    channels,
    ledger_bytes: ledger,
  };
}

function analyzeChannel(channelIndex, minimumDecimal, maximumDecimal, parameters, includeSegments) {
  const minimum = checkedInt32BigInt(minimumDecimal, "Accumulator minimum");
  const maximum = checkedInt32BigInt(maximumDecimal, "Accumulator maximum");
  assert(minimum <= maximum, "Accumulator interval is non-monotone.");
  const segments = includeSegments ? [] : null;
  let cursor = minimum;
  let pairSegmentCount = 0;
  let divergentStateCount = 0n;
  let defaultLowerStateCount = 0n;
  let defaultHigherStateCount = 0n;
  let divergentRegionCount = 0;
  let maximumAbsoluteOutputDelta = 0;
  let previousDivergent = false;
  let first = null;
  let last = null;
  while (true) {
    const pair = projectPair(cursor, parameters);
    const end = findPairRunEnd(cursor, maximum, pair, parameters);
    pairSegmentCount += 1;
    const count = end - cursor + 1n;
    const delta = pair.default_output_code - pair.single_output_code;
    const divergent = delta !== 0;
    if (divergent) {
      divergentStateCount += count;
      if (delta < 0) defaultLowerStateCount += count;
      else defaultHigherStateCount += count;
      maximumAbsoluteOutputDelta = Math.max(maximumAbsoluteOutputDelta, Math.abs(delta));
      if (!previousDivergent) divergentRegionCount += 1;
      if (!first) first = { accumulator: cursor, default: pair.default_output_code, single: pair.single_output_code };
      last = { accumulator: end, default: pair.default_output_code, single: pair.single_output_code };
    }
    if (segments) segments.push({
      accumulator_minimum_decimal: String(cursor),
      accumulator_maximum_decimal: String(end),
      state_count_decimal: String(count),
      default_output_code: pair.default_output_code,
      single_output_code: pair.single_output_code,
      divergent,
    });
    previousDivergent = divergent;
    if (end === maximum) break;
    cursor = end + 1n;
    assert(pairSegmentCount <= 511, "Ordered output-pair partition exceeds the 8-bit bound.");
  }
  const intervalStateCount = maximum - minimum + 1n;
  return {
    channel_index: channelIndex,
    post_bias_minimum_decimal: String(minimum),
    post_bias_maximum_decimal: String(maximum),
    interval_state_count_decimal: String(intervalStateCount),
    interval_state_count: intervalStateCount,
    divergent_state_count_decimal: String(divergentStateCount),
    divergent_state_count: divergentStateCount,
    divergent_state_ratio: ratio(divergentStateCount, intervalStateCount),
    default_lower_state_count_decimal: String(defaultLowerStateCount),
    default_lower_state_count: defaultLowerStateCount,
    default_higher_state_count_decimal: String(defaultHigherStateCount),
    default_higher_state_count: defaultHigherStateCount,
    pair_segment_count: pairSegmentCount,
    divergent_region_count: divergentRegionCount,
    maximum_absolute_output_delta: maximumAbsoluteOutputDelta,
    first_divergent_accumulator_decimal: first ? String(first.accumulator) : null,
    first_default_output_code: first?.default ?? null,
    first_single_output_code: first?.single ?? null,
    last_divergent_accumulator_decimal: last ? String(last.accumulator) : null,
    last_default_output_code: last?.default ?? null,
    last_single_output_code: last?.single ?? null,
    default_quantized_multiplier: parameters.defaultMultiplier,
    default_shift: parameters.defaultShift,
    single_quantized_multiplier: parameters.singleMultiplier,
    single_shift: parameters.singleShift,
    activation_code_range: parameters.activationRange,
    output_zero_point: parameters.outputZeroPoint,
    segments,
  };
}

function projectionParameters(requant, activationRange, channel) {
  return {
    defaultMultiplier: Number(requant.channel_quantized_multipliers[channel]),
    defaultShift: Number(requant.channel_shifts[channel]),
    singleMultiplier: Number(requant.channel_single_rounding_quantized_multipliers[channel]),
    singleShift: Number(requant.channel_single_rounding_shifts[channel]),
    outputZeroPoint: Number(requant.output_zero_point),
    activationRange,
  };
}

function projectPair(accumulator, parameters) {
  const value = Number(accumulator);
  const defaultScaled = multiplyByQuantizedMultiplierDefault(value, parameters.defaultMultiplier, parameters.defaultShift);
  const singleScaled = multiplyByQuantizedMultiplierSingleRounding(value, parameters.singleMultiplier, parameters.singleShift);
  assert(defaultScaled != null && singleScaled != null, "Fixed-point projection is outside its source contract.");
  return {
    default_output_code: clamp(defaultScaled + parameters.outputZeroPoint, ...parameters.activationRange),
    single_output_code: clamp(singleScaled + parameters.outputZeroPoint, ...parameters.activationRange),
  };
}

function findPairRunEnd(start, maximum, expected, parameters) {
  if (samePair(projectPair(maximum, parameters), expected)) return maximum;
  let same = start;
  let different = maximum;
  while (same + 1n < different) {
    const middle = same + ((different - same) >> 1n);
    if (samePair(projectPair(middle, parameters), expected)) same = middle;
    else different = middle;
  }
  return same;
}

function samePair(left, right) {
  return left.default_output_code === right.default_output_code && left.single_output_code === right.single_output_code;
}

function equivalenceLedger(witnessDigest, requantDigest, opIndex, channels) {
  assert(/^[a-f0-9]{64}$/.test(witnessDigest || "") && /^[a-f0-9]{64}$/.test(requantDigest || ""), "Source digest identity is invalid.");
  const writer = new BinaryWriter(LEDGER_PREFIX.length + 128 + channels.length * 17 * 8);
  writer.writeBytes(LEDGER_PREFIX);
  writer.writeBytes(new TextEncoder().encode(witnessDigest));
  writer.writeBytes(new TextEncoder().encode(requantDigest));
  for (const channel of channels) {
    writer.writeU64(opIndex);
    writer.writeU64(channel.channel_index);
    writer.writeI64(channel.post_bias_minimum_decimal);
    writer.writeI64(channel.post_bias_maximum_decimal);
    writer.writeU64(channel.interval_state_count);
    writer.writeU64(channel.divergent_state_count);
    writer.writeU64(channel.default_lower_state_count);
    writer.writeU64(channel.default_higher_state_count);
    writer.writeU64(channel.pair_segment_count);
    writer.writeU64(channel.divergent_region_count);
    writer.writeI64(channel.maximum_absolute_output_delta);
    writer.writeI64(channel.first_divergent_accumulator_decimal ?? MISSING_I64);
    writer.writeI64(channel.first_default_output_code ?? MISSING_I64);
    writer.writeI64(channel.first_single_output_code ?? MISSING_I64);
    writer.writeI64(channel.last_divergent_accumulator_decimal ?? MISSING_I64);
    writer.writeI64(channel.last_default_output_code ?? MISSING_I64);
    writer.writeI64(channel.last_single_output_code ?? MISSING_I64);
  }
  return writer.finish();
}

function witnessRow(channel) {
  const row = {
    channel_index: channel.channel_index,
    post_bias_minimum_decimal: channel.post_bias_minimum_decimal,
    post_bias_maximum_decimal: channel.post_bias_maximum_decimal,
    interval_state_count_decimal: channel.interval_state_count_decimal,
    divergent_state_count_decimal: channel.divergent_state_count_decimal,
    divergent_state_ratio: channel.divergent_state_ratio,
    default_lower_state_count_decimal: channel.default_lower_state_count_decimal,
    default_higher_state_count_decimal: channel.default_higher_state_count_decimal,
    pair_segment_count: channel.pair_segment_count,
    divergent_region_count: channel.divergent_region_count,
    maximum_absolute_output_delta: channel.maximum_absolute_output_delta,
    default_quantized_multiplier: channel.default_quantized_multiplier,
    default_shift: channel.default_shift,
    single_quantized_multiplier: channel.single_quantized_multiplier,
    single_shift: channel.single_shift,
  };
  if (channel.first_divergent_accumulator_decimal != null) {
    row.first_divergent_accumulator_decimal = channel.first_divergent_accumulator_decimal;
    row.first_default_output_code = channel.first_default_output_code;
    row.first_single_output_code = channel.first_single_output_code;
    row.last_divergent_accumulator_decimal = channel.last_divergent_accumulator_decimal;
    row.last_default_output_code = channel.last_default_output_code;
    row.last_single_output_code = channel.last_single_output_code;
  }
  return row;
}

function notAssessedRow(witness, sourceWitness, requant) {
  return {
    op_index: witness.op_index,
    op_name: witness.op_name,
    assessment_status: "not_assessed",
    not_assessed_reason: witness.not_assessed_reason || "Source fixed-point evidence is not assessed.",
    source_witness_ledger_sha256: sourceWitness?.witness_ledger_sha256 || "",
    source_requantization_ledger_sha256: requant?.channel_ledger_sha256 || "",
  };
}

function compareSummary(actual, expected) {
  for (const key of [
    "status", "candidate_op_count", "assessed_op_count", "unassessed_op_count", "assessed_channel_count",
    "equivalent_channel_count", "divergent_channel_count", "divergent_op_count", "interval_state_count_decimal",
    "divergent_state_count_decimal", "default_lower_state_count_decimal", "default_higher_state_count_decimal",
    "maximum_absolute_output_delta", "pair_segment_count", "divergent_region_count",
  ]) assert(String(actual[key]) === String(expected[key]), `Rounding-equivalence ${key} mismatch.`);
  assert(nearlyEqual(actual.divergent_state_ratio, expected.divergent_state_ratio), "Rounding-equivalence ratio mismatch.");
  assert(JSON.stringify(actual.equivalence_ranking_op_indices) === JSON.stringify(expected.equivalence_ranking_op_indices), "Rounding-equivalence ranking mismatch.");
  assert(JSON.stringify(actual.divergence_histogram) === JSON.stringify(expected.divergence_histogram), "Rounding-equivalence histogram mismatch.");
}

function compareOpRow(actual, expected) {
  for (const key of ["op_index", "op_name", "assessment_status", "not_assessed_reason", "source_witness_ledger_sha256", "source_requantization_ledger_sha256"]) assert(String(actual[key] ?? "") === String(expected[key] ?? ""), `Rounding-equivalence op ${key} mismatch.`);
  if (actual.assessment_status !== "assessed") return;
  for (const key of [
    "assessed_channel_count", "equivalent_channel_count", "divergent_channel_count", "interval_state_count_decimal",
    "divergent_state_count_decimal", "default_lower_state_count_decimal", "default_higher_state_count_decimal",
    "maximum_absolute_output_delta", "maximum_pair_segment_count", "maximum_divergent_region_count", "output_zero_point",
  ]) assert(String(actual[key]) === String(expected[key]), `Rounding-equivalence op #${actual.op_index} ${key} mismatch.`);
  assert(nearlyEqual(actual.divergent_state_ratio, expected.divergent_state_ratio), `Rounding-equivalence op #${actual.op_index} ratio mismatch.`);
  for (const key of [
    "activation_code_range", "channel_interval_state_counts_decimal", "channel_divergent_state_counts_decimal",
    "channel_default_lower_state_counts_decimal", "channel_default_higher_state_counts_decimal", "channel_pair_segment_counts",
    "channel_divergent_region_counts", "channel_maximum_absolute_output_deltas", "channel_first_divergent_accumulators_decimal",
    "channel_first_default_output_codes", "channel_first_single_output_codes",
  ]) assert(JSON.stringify(actual[key]) === JSON.stringify(expected[key]), `Rounding-equivalence op #${actual.op_index} ${key} mismatch.`);
  compareTopChannels(actual.op_index, actual.top_channels, expected.top_channels);
}

function compareTopChannels(opIndex, actual, expected) {
  assert(actual.length === expected.length, `Rounding-equivalence op #${opIndex} top-channel count mismatch.`);
  for (let index = 0; index < actual.length; index += 1) {
    const left = actual[index];
    const right = expected[index];
    const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
    for (const key of keys) {
      const matches = key === "divergent_state_ratio"
        ? nearlyEqual(left[key], right[key])
        : String(left[key] ?? "") === String(right[key] ?? "");
      assert(matches, `Rounding-equivalence op #${opIndex} top channel ${index} ${key} mismatch.`);
    }
  }
}

function compareSelected(row, selected) {
  const channel = selected.channel_index;
  const fields = [
    ["interval_state_count_decimal", "channel_interval_state_counts_decimal"],
    ["divergent_state_count_decimal", "channel_divergent_state_counts_decimal"],
    ["default_lower_state_count_decimal", "channel_default_lower_state_counts_decimal"],
    ["default_higher_state_count_decimal", "channel_default_higher_state_counts_decimal"],
    ["pair_segment_count", "channel_pair_segment_counts"],
    ["divergent_region_count", "channel_divergent_region_counts"],
    ["maximum_absolute_output_delta", "channel_maximum_absolute_output_deltas"],
    ["first_divergent_accumulator_decimal", "channel_first_divergent_accumulators_decimal"],
    ["first_default_output_code", "channel_first_default_output_codes"],
    ["first_single_output_code", "channel_first_single_output_codes"],
  ];
  for (const [selectedKey, rowKey] of fields) assert(String(selected[selectedKey] ?? "") === String(row[rowKey][channel] ?? ""), `Selected channel ${selectedKey} mismatch.`);
}

function divergenceHistogram(channels) {
  return HISTOGRAM_LABELS.map((label, index) => {
    const selected = channels.filter((channel) => ratioBin(channel.divergent_state_count, channel.interval_state_count) === index);
    return {
      label,
      channel_count: selected.length,
      interval_state_count_decimal: String(selected.reduce((sum, channel) => sum + channel.interval_state_count, 0n)),
      divergent_state_count_decimal: String(selected.reduce((sum, channel) => sum + channel.divergent_state_count, 0n)),
    };
  });
}

function ratioBin(numerator, denominator) {
  if (numerator === 0n) return 0;
  if (numerator * 10_000n <= denominator) return 1;
  if (numerator * 1_000n <= denominator) return 2;
  if (numerator * 100n <= denominator) return 3;
  if (numerator * 10n <= denominator) return 4;
  if (numerator * 2n <= denominator) return 5;
  return 6;
}

function compareChannels(left, right) {
  return right.maximum_absolute_output_delta - left.maximum_absolute_output_delta
    || compareBigRatio(left.divergent_state_count, left.interval_state_count, right.divergent_state_count, right.interval_state_count)
    || right.divergent_region_count - left.divergent_region_count
    || left.channel_index - right.channel_index;
}

function compareOps(left, right) {
  return (right.maximum_absolute_output_delta || 0) - (left.maximum_absolute_output_delta || 0)
    || compareBigRatio(BigInt(left.divergent_state_count_decimal), BigInt(left.interval_state_count_decimal), BigInt(right.divergent_state_count_decimal), BigInt(right.interval_state_count_decimal))
    || right.divergent_channel_count - left.divergent_channel_count
    || left.op_index - right.op_index;
}

function compareBigRatio(leftNumerator, leftDenominator, rightNumerator, rightDenominator) {
  const left = leftNumerator * rightDenominator;
  const right = rightNumerator * leftDenominator;
  return left === right ? 0 : left > right ? -1 : 1;
}

export function drawRoundingEquivalenceCanvas(canvas, row, field = "exposure", selectedChannel = null, logicalWidth = null, logicalHeight = null) {
  if (!canvas || !row?.assessed_channel_count) return;
  const width = Math.max(320, Math.round(logicalWidth || canvas.clientWidth || 760));
  const columns = Math.max(16, Math.min(64, Math.floor(width / 13)));
  const rows = Math.ceil(row.assessed_channel_count / columns);
  const cell = Math.max(7, Math.min(14, Math.floor((width - 32) / columns)));
  const height = Math.max(126, logicalHeight || rows * cell + 34);
  const ratioScale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratioScale);
  canvas.height = Math.round(height * ratioScale);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratioScale, 0, 0, ratioScale, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111820";
  context.fillRect(0, 0, width, height);
  const offsetX = Math.floor((width - columns * cell) / 2);
  const offsetY = 16;
  for (let channel = 0; channel < row.assessed_channel_count; channel += 1) {
    const x = offsetX + (channel % columns) * cell;
    const y = offsetY + Math.floor(channel / columns) * cell;
    context.fillStyle = channelColor(row, channel, field);
    context.fillRect(x + 1, y + 1, Math.max(1, cell - 2), Math.max(1, cell - 2));
    if (channel === selectedChannel) {
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.strokeRect(x, y, cell, cell);
    }
  }
  canvas.dataset.columns = String(columns);
  canvas.dataset.cell = String(cell);
  canvas.dataset.offsetX = String(offsetX);
  canvas.dataset.offsetY = String(offsetY);
}

export function drawRoundingTraceCanvas(canvas, selected, logicalWidth = null, logicalHeight = 244) {
  if (!canvas || !selected?.segments) return;
  const width = Math.max(320, Math.round(logicalWidth || canvas.clientWidth || 760));
  const height = logicalHeight;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.fillStyle = "#111820";
  context.fillRect(0, 0, width, height);
  const left = 42;
  const right = width - 18;
  const top = 20;
  const bottom = height - 36;
  const minimum = BigInt(selected.post_bias_minimum_decimal);
  const total = BigInt(selected.interval_state_count_decimal);
  const [qmin, qmax] = selected.activation_code_range;
  const xFor = (value) => left + Number((BigInt(value) - minimum) * 1_000_000n / (total > 1n ? total - 1n : 1n)) / 1_000_000 * (right - left);
  const yFor = (value) => bottom - ((value - qmin) / Math.max(1, qmax - qmin)) * (bottom - top);
  context.strokeStyle = "#394552";
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = top + (bottom - top) * index / 4;
    context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
  }
  for (const segment of selected.segments) {
    if (!segment.divergent) continue;
    const x1 = xFor(segment.accumulator_minimum_decimal);
    const x2 = xFor(segment.accumulator_maximum_decimal);
    context.fillStyle = "rgba(229, 91, 75, 0.48)";
    context.fillRect(x1, top, Math.max(1, x2 - x1 + 1), bottom - top);
  }
  drawStepPath(context, selected.segments, "default_output_code", xFor, yFor, "#f4bf4f", left, right);
  drawStepPath(context, selected.segments, "single_output_code", xFor, yFor, "#54c7b1", left, right);
  context.fillStyle = "#aeb8c2";
  context.font = "11px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText(String(qmax), 8, top + 4);
  context.fillText(String(qmin), 8, bottom + 4);
  context.fillText(formatInteger(selected.post_bias_minimum_decimal), left, height - 13);
  const maximumLabel = formatInteger(selected.post_bias_maximum_decimal);
  context.fillText(maximumLabel, right - context.measureText(maximumLabel).width, height - 13);
  context.fillStyle = "#f4bf4f";
  context.fillRect(left, 7, 12, 2);
  context.fillStyle = "#aeb8c2";
  context.fillText("default", left + 17, 11);
  context.fillStyle = "#54c7b1";
  context.fillRect(left + 78, 7, 12, 2);
  context.fillStyle = "#aeb8c2";
  context.fillText("single", left + 95, 11);
  canvas.dataset.plotLeft = String(left);
  canvas.dataset.plotRight = String(right);
}

function drawStepPath(context, segments, key, xFor, yFor, color, left, right) {
  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  let started = false;
  for (const segment of segments) {
    const x1 = Math.max(left, xFor(segment.accumulator_minimum_decimal));
    const x2 = Math.min(right, xFor(segment.accumulator_maximum_decimal));
    const y = yFor(segment[key]);
    if (!started) { context.moveTo(x1, y); started = true; }
    else context.lineTo(x1, y);
    context.lineTo(x2, y);
  }
  context.stroke();
}

function channelColor(row, channel, field) {
  const divergent = BigInt(row.channel_divergent_state_counts_decimal[channel]);
  const total = BigInt(row.channel_interval_state_counts_decimal[channel]);
  if (field === "delta") return row.channel_maximum_absolute_output_deltas[channel] ? "#e55b4b" : "#2f7f73";
  if (field === "regions") {
    const value = row.channel_divergent_region_counts[channel];
    const light = 30 + Math.min(38, Math.log2(value + 1) * 5);
    return value ? `hsl(42 78% ${light}%)` : "#2f7f73";
  }
  if (divergent === 0n) return "#2f7f73";
  const ratio = Number(divergent * 1_000_000n / total) / 1_000_000;
  const light = 34 + Math.min(30, Math.sqrt(ratio) * 90);
  return `hsl(7 72% ${light}%)`;
}

function installHeatInteraction(canvas, tooltip, row, getField, onSelect) {
  const channelAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const columns = Number(canvas.dataset.columns || 1);
    const cell = Number(canvas.dataset.cell || 1);
    const x = event.clientX - rect.left - Number(canvas.dataset.offsetX || 0);
    const y = event.clientY - rect.top - Number(canvas.dataset.offsetY || 0);
    const column = Math.floor(x / cell);
    const line = Math.floor(y / cell);
    const channel = line * columns + column;
    return column >= 0 && column < columns && line >= 0 && channel < row.assessed_channel_count ? channel : null;
  };
  canvas.addEventListener("pointermove", (event) => {
    const channel = channelAt(event);
    if (channel == null) { tooltip.hidden = true; return; }
    const divergent = BigInt(row.channel_divergent_state_counts_decimal[channel]);
    const total = BigInt(row.channel_interval_state_counts_decimal[channel]);
    tooltip.textContent = `ch ${channel} / ${getField()} / ${formatInteger(divergent)} of ${formatInteger(total)} / ${formatPercent(ratio(divergent, total))} / ${row.channel_divergent_region_counts[channel]} regions`;
    tooltip.hidden = false;
    positionTooltip(tooltip, event, canvas);
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  canvas.addEventListener("click", (event) => { const channel = channelAt(event); if (channel != null) onSelect(channel); });
}

function installTraceInteraction(canvas, tooltip, selected) {
  const segmentAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const left = Number(canvas.dataset.plotLeft || 42);
    const right = Number(canvas.dataset.plotRight || rect.width - 18);
    const x = Math.max(left, Math.min(right, event.clientX - rect.left));
    const fraction = (x - left) / Math.max(1, right - left);
    const minimum = BigInt(selected.post_bias_minimum_decimal);
    const width = BigInt(selected.interval_state_count_decimal) - 1n;
    const accumulator = minimum + BigInt(Math.round(fraction * 1_000_000)) * width / 1_000_000n;
    return binaryFindSegment(selected.segments, accumulator);
  };
  canvas.addEventListener("pointermove", (event) => {
    const segment = segmentAt(event);
    if (!segment) { tooltip.hidden = true; return; }
    tooltip.textContent = `${formatInteger(segment.accumulator_minimum_decimal)} .. ${formatInteger(segment.accumulator_maximum_decimal)} / ${formatInteger(segment.state_count_decimal)} states / default ${segment.default_output_code} / single ${segment.single_output_code}${segment.divergent ? " / divergent" : " / equivalent"}`;
    tooltip.hidden = false;
    positionTooltip(tooltip, event, canvas);
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

function binaryFindSegment(segments, accumulator) {
  let low = 0;
  let high = segments.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    const segment = segments[middle];
    if (accumulator < BigInt(segment.accumulator_minimum_decimal)) high = middle - 1;
    else if (accumulator > BigInt(segment.accumulator_maximum_decimal)) low = middle + 1;
    else return segment;
  }
  return null;
}

function renderSummary(container, evidence) {
  if (!container) return;
  const metrics = [
    ["Certified channels", formatNumber(evidence.assessed_channel_count), `${formatNumber(evidence.assessed_op_count)} ops`],
    ["Equivalent", formatNumber(evidence.equivalent_channel_count), "complete interval hull"],
    ["Divergent", formatNumber(evidence.divergent_channel_count), `${formatNumber(evidence.divergent_op_count)} ops`],
    ["Divergent states", formatInteger(evidence.divergent_state_count_decimal), `${formatPercent(evidence.divergent_state_ratio)} of ${formatInteger(evidence.interval_state_count_decimal)}`],
    ["Maximum delta", `${evidence.maximum_absolute_output_delta ?? 0} code`, `${formatInteger(evidence.default_lower_state_count_decimal)} lower / ${formatInteger(evidence.default_higher_state_count_decimal)} higher`],
  ];
  container.replaceChildren(...metrics.map(([label, value, detail]) => {
    const metric = element("div", "rounding-equivalence-metric");
    metric.append(element("span", "", label), element("strong", "", value), element("small", "", detail));
    return metric;
  }));
}

function histogramTable(histogram) {
  return tableBlock("Interval-hull divergence distribution", ["Divergence ratio", "Channels", "Interval states", "Divergent states"], histogram.map((bin) => [bin.label, formatNumber(bin.channel_count), formatInteger(bin.interval_state_count_decimal), formatInteger(bin.divergent_state_count_decimal)]), "rounding-equivalence-histogram");
}

function rankingTable(evidence, selectedOpIndex, onSelect, jumpToGraphOp) {
  const rows = evidence.equivalence_ranking_op_indices.slice(0, 16).map((opIndex) => evidence.ops.find((row) => row.op_index === opIndex)).filter(Boolean);
  const block = tableBlock("Build-mode exposure ranking", ["Op", "Channels", "Divergent states", "Ratio", "Max delta"], rows.map((row) => [
    `#${row.op_index} ${row.op_name}`,
    `${formatNumber(row.divergent_channel_count)} / ${formatNumber(row.assessed_channel_count)}`,
    formatInteger(row.divergent_state_count_decimal),
    formatPercent(row.divergent_state_ratio),
    String(row.maximum_absolute_output_delta ?? 0),
  ]), "rounding-equivalence-ranking");
  block.querySelectorAll("tbody tr").forEach((tableRow, index) => {
    const row = rows[index];
    tableRow.tabIndex = 0;
    tableRow.dataset.selected = String(row.op_index === selectedOpIndex);
    tableRow.addEventListener("click", () => onSelect(row.op_index));
    tableRow.addEventListener("dblclick", () => jumpToGraphOp?.(row.op_index));
    tableRow.addEventListener("keydown", (event) => { if (event.key === "Enter") onSelect(row.op_index); });
  });
  return block;
}

function selectedChannelExport(analysis, evidence, row, selected) {
  const certificate = JSON.parse(JSON.stringify(selected, (_key, value) => typeof value === "bigint" ? value.toString() : value));
  return {
    schema: "deepbom.rounding_equivalence_selected_channel.v1",
    generated_from: {
      model_sha256: analysis.model_sha256,
      target_profile_id: analysis.target_profile?.id,
      evidence_schema: evidence.schema,
      method_version: evidence.method_version,
      source_commit: evidence.source_commit,
      source_witness_ledger_sha256: row.source_witness_ledger_sha256,
      source_requantization_ledger_sha256: row.source_requantization_ledger_sha256,
      equivalence_ledger_sha256: row.equivalence_ledger_sha256,
    },
    op_index: row.op_index,
    op_name: row.op_name,
    ...certificate,
    interval_hull_boundary: evidence.interpretation_boundary,
  };
}

function tableBlock(title, headers, rows, className) {
  const block = element("section", `rounding-equivalence-table ${className}`);
  block.append(element("h4", "", title));
  const scroll = element("div", "rounding-equivalence-table-scroll");
  const table = element("table", "");
  const head = element("thead", "");
  const headRow = element("tr", "");
  headers.forEach((header) => headRow.append(element("th", "", header)));
  head.append(headRow);
  const tbody = element("tbody", "");
  rows.forEach((row) => { const tr = element("tr", ""); row.forEach((value) => tr.append(element("td", "", value))); tbody.append(tr); });
  table.append(head, tbody);
  scroll.append(table);
  block.append(scroll);
  return block;
}

function unassessedTable(rows) {
  const unassessed = (rows || []).filter((row) => row.assessment_status !== "assessed");
  if (!unassessed.length) return element("div", "rounding-equivalence-unassessed", "All candidate operators were assessed.");
  return tableBlock("Unassessed operators", ["Op", "Reason"], unassessed.map((row) => [`#${row.op_index} ${row.op_name}`, row.not_assessed_reason]), "rounding-equivalence-unassessed");
}

function definitionTable(rows) {
  const dl = element("dl", "rounding-equivalence-definition");
  for (const [term, detail] of rows) dl.append(element("dt", "", term), element("dd", "", detail));
  return dl;
}

function segmentedControl(label, options, selected, onSelect) {
  const group = element("div", "rounding-equivalence-segments");
  group.setAttribute("role", "tablist");
  group.setAttribute("aria-label", label);
  for (const [value, text] of options) {
    const button = element("button", "", text);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(value === selected));
    button.addEventListener("click", () => onSelect(value));
    group.append(button);
  }
  return group;
}

function commandButton(text, title, action) {
  const button = element("button", "secondary-action", text);
  button.type = "button";
  button.title = title;
  button.addEventListener("click", action);
  return button;
}

function messageNode(text, tone = "") {
  const node = element("p", "rounding-equivalence-message", text);
  if (tone) node.dataset.tone = tone;
  return node;
}

function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  return node;
}

function positionTooltip(tooltip, event, canvas) {
  const rect = canvas.getBoundingClientRect();
  tooltip.style.left = `${Math.min(rect.width - 16, Math.max(8, event.clientX - rect.left + 12))}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - rect.top - 30)}px`;
}

function setStatus(status, text, tone) {
  if (!status) return;
  status.textContent = text;
  status.dataset.tone = tone;
}

function firstDivergentChannel(row) {
  const index = row.channel_divergent_state_counts_decimal.findIndex((value) => BigInt(value) > 0n);
  return index >= 0 ? index : null;
}

function checkedInt32BigInt(value, label) {
  const integer = BigInt(value);
  assert(integer >= -2147483648n && integer <= 2147483647n, `${label} is outside int32.`);
  return integer;
}

function clamp(value, minimum, maximum) { return Math.max(minimum, Math.min(maximum, value)); }
function clampInteger(value, minimum, maximum) { const number = Number.parseInt(value, 10); return Math.max(minimum, Math.min(maximum, Number.isFinite(number) ? number : minimum)); }
function ratio(numerator, denominator) { return denominator === 0n ? 0 : Number(numerator) / Number(denominator); }
function sumBigInt(rows, key) { return rows.reduce((sum, row) => sum + BigInt(row[key]), 0n); }
function sumNumber(rows, key) { return rows.reduce((sum, row) => sum + Number(row[key] || 0), 0); }
function maxOptional(rows, key) { const values = rows.map((row) => row[key]).filter((value) => value != null); return values.length ? Math.max(...values) : null; }
function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatInteger(value) { try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "not assessed"); } }
function formatPercent(value) { const number = Number(value || 0) * 100; return `${number < 0.001 && number > 0 ? number.toFixed(6) : number < 0.1 ? number.toFixed(4) : number.toFixed(2)}%`; }
function humanize(value) { return String(value || "not assessed").replaceAll("_", " "); }
function nearlyEqual(left, right) { return Math.abs(Number(left) - Number(right)) <= 1e-12 * Math.max(1, Math.abs(Number(left)), Math.abs(Number(right))); }
function assert(condition, message) { if (!condition) throw new Error(message); }

class BinaryWriter {
  constructor(capacity) { this.bytes = new Uint8Array(capacity); this.view = new DataView(this.bytes.buffer); this.offset = 0; }
  writeBytes(value) { this.bytes.set(value, this.offset); this.offset += value.length; }
  writeI64(value) { this.view.setBigInt64(this.offset, BigInt(value), true); this.offset += 8; }
  writeU64(value) { this.view.setBigUint64(this.offset, BigInt(value), true); this.offset += 8; }
  finish() { assert(this.offset === this.bytes.length, "Rounding-equivalence ledger length mismatch."); return this.bytes; }
}
