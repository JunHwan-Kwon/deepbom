import { validateAccumulatorAtlas } from "./accumulator-atlas.js";
import { browserAssetUrl } from "./browser-asset-url.js";
import { sha256Hex } from "./hash.js";
import {
  multiplyByQuantizedMultiplierDefault,
  multiplyByQuantizedMultiplierSingleRounding,
  roundTiesAway,
} from "./quantization-math.js";
import { validateRequantizationFidelity } from "./requantization-fidelity.js";

export const KERNEL_WITNESS_SCHEMA = "deepbom.kernel_extremum_witness.v1";
const METHOD_VERSION = "2026-07-17.1";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const CANDIDATE_OPS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]);
const LEDGER_PREFIX = new TextEncoder().encode("deepbom.kernel_extremum_witness.v1\0");
const PATTERN_PREFIX = new TextEncoder().encode("deepbom.kernel_extremum_pattern.v1\0");
const MISSING_I64 = -(1n << 63n);
const TOP_LIMIT = 8;

export function createKernelWitnessController({
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
  let bytes = null;
  let evidence = null;
  let selectedOpIndex = null;
  let selectedChannelIndex = null;
  let selectedEndpoint = "maximum";
  let mode = "pattern";
  let worker = null;
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "kernel_extremum_witness.json");
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
    bytes = context.modelBytes || null;
    evidence = analysis?.kernel_extremum_witness || null;
    if (!evidence || !bytes || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedOpIndex = null;
      selectedChannelIndex = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      validateKernelWitnessShape(evidence);
      const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = evidence.witness_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
        selectedChannelIndex = null;
      }
      renderKernelWitnessSummary(summary, evidence);
      renderBody();
      if (status) {
        status.textContent = evidence.assessed_op_count ? "source arithmetic / verification pending" : evidence.status.replaceAll("_", " ");
        status.dataset.tone = evidence.build_mode_divergent_endpoint_count ? "watch" : evidence.assessed_op_count ? "ok" : "muted";
      }
      if (evidence.assessed_op_count && typeof Worker === "function") {
        worker = new Worker(browserAssetUrl("./lib/kernel-witness-worker.js", "./kernel-witness-worker.js", import.meta.url), { type: "module" });
        worker.onmessage = (event) => {
          if (token !== renderToken || !status) return;
          if (event.data?.ok) {
            status.textContent = "independently verified";
            status.dataset.tone = evidence.build_mode_divergent_endpoint_count ? "watch" : "ok";
          } else {
            status.textContent = `integrity error: ${event.data?.error || "verification failed"}`;
            status.dataset.tone = "risk";
          }
          worker?.terminate();
          worker = null;
        };
        worker.onerror = (event) => {
          if (token !== renderToken || !status) return;
          status.textContent = `integrity error: ${event.message || "worker failed"}`;
          status.dataset.tone = "risk";
          worker?.terminate();
          worker = null;
        };
        worker.postMessage({ analysis, modelBytes: bytes });
      }
    } catch (error) {
      if (summary) summary.replaceChildren();
      if (body) body.replaceChildren(messageNode(`Kernel witness evidence rejected: ${error.message}`, "risk"));
      if (status) {
        status.textContent = "evidence rejected";
        status.dataset.tone = "risk";
      }
    }
  }

  function renderBody() {
    if (!body || !evidence) return;
    resizeObserver?.disconnect();
    const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
    if (!assessed.length) {
      body.replaceChildren(messageNode("No constant 8-bit convolution-family kernel witness was assessable."), unassessedTable(evidence.ops));
      return;
    }
    const row = assessed.find((candidate) => candidate.op_index === selectedOpIndex) || assessed[0];
    selectedOpIndex = row.op_index;
    const defaultChannel = row.worst_channel?.channel_index ?? 0;
    if (!Number.isInteger(selectedChannelIndex) || selectedChannelIndex < 0 || selectedChannelIndex >= row.assessed_channel_count) {
      selectedChannelIndex = defaultChannel;
    }
    const selected = reconstructKernelChannel(analysis, bytes, row.op_index, selectedChannelIndex);
    compareSelectedTopWitness(row, selected);

    const toolbar = document.createElement("div");
    toolbar.className = "kernel-witness-toolbar";
    const opSelect = document.createElement("select");
    opSelect.className = "kernel-witness-op-select";
    opSelect.setAttribute("aria-label", "Kernel witness operator");
    for (const candidate of assessed) {
      const option = new Option(
        `#${candidate.op_index} ${candidate.op_name} / ${formatNumber(candidate.assessed_channel_count)} ch / ${formatNumber(candidate.build_mode_divergent_endpoint_count)} mode deltas`,
        String(candidate.op_index),
      );
      option.selected = candidate.op_index === row.op_index;
      opSelect.append(option);
    }
    opSelect.addEventListener("change", () => {
      selectedOpIndex = Number(opSelect.value);
      selectedChannelIndex = null;
      renderBody();
    });
    const channelLabel = document.createElement("label");
    channelLabel.className = "kernel-witness-channel-control";
    channelLabel.append(document.createTextNode("Channel"));
    const channelInput = document.createElement("input");
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
    toolbar.append(opSelect, channelLabel, segmentedControl("Witness endpoint", [["minimum", "Minimum"], ["maximum", "Maximum"]], selectedEndpoint, (value) => {
      selectedEndpoint = value;
      renderBody();
    }), segmentedControl("Witness field", [["pattern", "Input pattern"], ["contribution", "Contribution"]], mode, (value) => {
      mode = value;
      renderBody();
    }));

    const main = document.createElement("div");
    main.className = "kernel-witness-main";
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "kernel-witness-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "kernel-witness-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${selectedEndpoint} canonical input witness for operator ${row.op_index} channel ${selectedChannelIndex}`);
    const tooltip = document.createElement("div");
    tooltip.className = "kernel-witness-tooltip";
    tooltip.hidden = true;
    canvasWrap.append(canvas, tooltip);
    installCanvasInteraction(canvas, tooltip);

    const details = document.createElement("div");
    details.className = "kernel-witness-details";
    const endpoint = selected[selectedEndpoint];
    const witnessDownload = document.createElement("button");
    witnessDownload.type = "button";
    witnessDownload.className = "secondary-action";
    witnessDownload.textContent = "Witness JSON";
    witnessDownload.title = "Download the selected canonical receptive-field input codes and pinned expected outputs";
    witnessDownload.addEventListener("click", async () => {
      selected.witness_pattern_sha256 ||= await sha256Hex(selected.pattern_bytes);
      onDownload?.(
        selectedWitnessExport(analysis, row, selected, selectedEndpoint),
        `kernel_witness_op_${row.op_index}_channel_${selectedChannelIndex}_${selectedEndpoint}.json`,
      );
    });
    const graphButton = document.createElement("button");
    graphButton.type = "button";
    graphButton.className = "secondary-action";
    graphButton.textContent = "Graph op";
    graphButton.addEventListener("click", () => jumpToGraphOp?.(row.op_index));
    const actions = document.createElement("div");
    actions.className = "kernel-witness-actions";
    actions.append(witnessDownload, graphButton);
    details.append(
      definitionTable([
        ["Coordinate", `#${row.op_index} ${row.op_name}, output channel ${selectedChannelIndex}`],
        ["Receptive field", `${formatNumber(selected.term_count)} terms, ${shapeText(row.weight_shape)} kernel`],
        ["Input contract", `${row.input_dtype} [${row.input_code_range.join(", ")}], zp ${row.input_zero_point}`],
        ["Output contract", `${row.output_dtype} [${row.output_code_range.join(", ")}], zp ${row.output_zero_point}; activation ${row.fused_activation}`],
        ["Term signs", `${formatNumber(selected.positive_centered_weight_count)} positive / ${formatNumber(selected.negative_centered_weight_count)} negative / ${formatNumber(selected.zero_centered_weight_count)} zero`],
        ["Pattern digest", selected.witness_pattern_sha256 || "computed on selected-witness export"],
      ]),
      actions,
      endpointPath(endpoint),
    );
    main.append(canvasWrap, details);

    const comparison = endpointComparison(selected);
    const ranking = rankingTable(evidence, row.op_index, (opIndex) => {
      selectedOpIndex = opIndex;
      selectedChannelIndex = null;
      renderBody();
    }, jumpToGraphOp);
    const boundary = document.createElement("p");
    boundary.className = "kernel-witness-boundary";
    boundary.textContent = evidence.interpretation_boundary;
    body.replaceChildren(toolbar, main, comparison, ranking, boundary, unassessedTable(evidence.ops));
    requestAnimationFrame(() => drawKernelWitnessCanvas(canvas, selected, selectedEndpoint, mode));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawKernelWitnessCanvas(canvas, selected, selectedEndpoint, mode));
      resizeObserver.observe(canvasWrap);
    }
  }

  return { render };
}

export function validateKernelWitnessShape(evidence) {
  assert(evidence && evidence.schema === KERNEL_WITNESS_SCHEMA, "Kernel witness schema mismatch.");
  assert(evidence.method_version === METHOD_VERSION, "Kernel witness method mismatch.");
  assert(evidence.source_commit === SOURCE_COMMIT, "Kernel witness source commit mismatch.");
  assert(Array.isArray(evidence.ops) && Array.isArray(evidence.witness_ranking_op_indices), "Kernel witness rows are unavailable.");
  assert(evidence.ops.length === Number(evidence.candidate_op_count), "Kernel witness candidate count mismatch.");
  for (const row of evidence.ops) {
    assert(Number.isInteger(row.op_index) && CANDIDATE_OPS.has(row.op_name), "Kernel witness op identity is invalid.");
    if (row.assessment_status !== "assessed") continue;
    assert(/^[a-f0-9]{64}$/.test(row.witness_ledger_sha256 || ""), `Kernel witness ledger digest is invalid at op #${row.op_index}.`);
    assert(Array.isArray(row.top_channels) && row.top_channels.length <= TOP_LIMIT, `Kernel witness top-channel set is invalid at op #${row.op_index}.`);
    for (const witness of row.top_channels) assert(/^[a-f0-9]{64}$/.test(witness.witness_pattern_sha256 || ""), `Kernel witness pattern digest is invalid at op #${row.op_index}.`);
  }
  return evidence;
}

export function validateKernelWitnessAnalysis(analysis, modelBytes) {
  validateAccumulatorAtlas(analysis, modelBytes);
  validateRequantizationFidelity(analysis, modelBytes);
  const actual = validateKernelWitnessShape(analysis?.kernel_extremum_witness);
  const expected = reconstructKernelWitnessAnalysis(analysis, modelBytes);
  compareSummary(actual, expected);
  assert(actual.ops.length === expected.ops.length, "Kernel witness op count mismatch.");
  actual.ops.forEach((row, index) => compareOpRow(row, expected.ops[index]));
  return expected;
}

export async function validateKernelWitnessDigests(analysis, modelBytes) {
  const reconstructed = validateKernelWitnessAnalysis(analysis, modelBytes);
  const actualRows = analysis.kernel_extremum_witness.ops;
  for (let index = 0; index < reconstructed.ops.length; index += 1) {
    const expected = reconstructed.ops[index];
    const actual = actualRows[index];
    if (expected.assessment_status !== "assessed") continue;
    const digest = await sha256Hex(expected.ledger_bytes);
    assert(digest === actual.witness_ledger_sha256, `Kernel witness ledger SHA-256 mismatch at op #${actual.op_index}.`);
    for (const witness of actual.top_channels) {
      const selected = reconstructKernelChannel(analysis, modelBytes, actual.op_index, witness.channel_index);
      const patternDigest = await sha256Hex(selected.pattern_bytes);
      assert(patternDigest === witness.witness_pattern_sha256, `Kernel witness pattern SHA-256 mismatch at op #${actual.op_index} channel ${witness.channel_index}.`);
    }
  }
  return reconstructed;
}

export function reconstructKernelWitnessAnalysis(analysis, modelBytes) {
  const bytes = asBytes(modelBytes);
  const tensors = tensorMap(analysis);
  const accumulatorRows = new Map((analysis?.accumulator_atlas?.ops || []).map((row) => [Number(row.op_index), row]));
  const requantRows = new Map((analysis?.requantization_fidelity?.ops || []).map((row) => [Number(row.op_index), row]));
  const candidates = (analysis?.ops || []).filter((op) => CANDIDATE_OPS.has(String(op.name)));
  const ops = candidates.map((op) => {
    try {
      return reconstructOp(op, tensors, bytes, accumulatorRows.get(Number(op.index)), requantRows.get(Number(op.index)));
    } catch (error) {
      return { op_index: Number(op.index), op_name: String(op.name), assessment_status: "not_assessed", not_assessed_reason: error.message };
    }
  });
  const assessed = ops.filter((row) => row.assessment_status === "assessed");
  const ranking = [...assessed].sort(compareOpRanking).map((row) => row.op_index);
  return {
    status: !ops.length ? "not_applicable" : assessed.length === ops.length ? "assessed" : assessed.length ? "partial" : "not_assessed",
    candidate_op_count: ops.length,
    assessed_op_count: assessed.length,
    unassessed_op_count: ops.length - assessed.length,
    assessed_channel_count: sum(assessed, "assessed_channel_count"),
    fixed_point_assessed_channel_count: sum(assessed, "fixed_point_assessed_channel_count"),
    witness_assignment_count: sum(assessed, "witness_assignment_count"),
    fixed_point_endpoint_evaluation_count: sum(assessed, "fixed_point_endpoint_evaluation_count"),
    default_ideal_mismatch_endpoint_count: sum(assessed, "default_ideal_mismatch_endpoint_count"),
    single_ideal_mismatch_endpoint_count: sum(assessed, "single_ideal_mismatch_endpoint_count"),
    build_mode_divergent_endpoint_count: sum(assessed, "build_mode_divergent_endpoint_count"),
    default_activation_clamped_endpoint_count: sum(assessed, "default_activation_clamped_endpoint_count"),
    single_activation_clamped_endpoint_count: sum(assessed, "single_activation_clamped_endpoint_count"),
    default_collapsed_extrema_channel_count: sum(assessed, "default_collapsed_extrema_channel_count"),
    single_collapsed_extrema_channel_count: sum(assessed, "single_collapsed_extrema_channel_count"),
    maximum_default_ideal_delta_codes: maxOptional(assessed, "maximum_default_ideal_delta_codes"),
    maximum_single_ideal_delta_codes: maxOptional(assessed, "maximum_single_ideal_delta_codes"),
    witness_ranking_op_indices: ranking,
    ops,
  };
}

export function reconstructKernelChannel(analysis, modelBytes, opIndex, channelIndex) {
  const bytes = asBytes(modelBytes);
  const tensors = tensorMap(analysis);
  const op = (analysis?.ops || []).find((candidate) => Number(candidate.index) === Number(opIndex));
  assert(op && CANDIDATE_OPS.has(String(op.name)), `Kernel witness op #${opIndex} is unavailable.`);
  const accumulator = (analysis?.accumulator_atlas?.ops || []).find((row) => Number(row.op_index) === Number(opIndex));
  const requant = (analysis?.requantization_fidelity?.ops || []).find((row) => Number(row.op_index) === Number(opIndex));
  const context = kernelContext(op, tensors, bytes, accumulator, requant);
  const channel = Number(channelIndex);
  assert(Number.isInteger(channel) && channel >= 0 && channel < context.channels, `Kernel witness channel ${channelIndex} is outside the op range.`);
  return reconstructChannel(context, channel, true);
}

export function reconstructKernelOpChannelsWithTerms(analysis, modelBytes, opIndex) {
  const bytes = asBytes(modelBytes);
  const tensors = tensorMap(analysis);
  const op = (analysis?.ops || []).find((candidate) => Number(candidate.index) === Number(opIndex));
  assert(op && CANDIDATE_OPS.has(String(op.name)), `Kernel witness op #${opIndex} is unavailable.`);
  const accumulator = (analysis?.accumulator_atlas?.ops || []).find((row) => Number(row.op_index) === Number(opIndex));
  const requant = (analysis?.requantization_fidelity?.ops || []).find((row) => Number(row.op_index) === Number(opIndex));
  const context = kernelContext(op, tensors, bytes, accumulator, requant);
  return {
    op_index: Number(op.index),
    op_name: String(op.name),
    input_code_range: [...context.inputRange],
    input_zero_point: context.inputZeroPoint,
    term_count: context.terms,
    channel_count: context.channels,
    channels: Array.from({ length: context.channels }, (_, channel) => reconstructChannel(context, channel, true)),
  };
}

function reconstructOp(op, tensors, bytes, accumulator, requant) {
  const context = kernelContext(op, tensors, bytes, accumulator, requant);
  const rowBytes = 9 * 8 + 18 * 8 + context.terms * 6;
  const ledger = new ByteWriter(LEDGER_PREFIX.byteLength + context.channels * rowBytes);
  ledger.writeBytes(LEDGER_PREFIX);
  const channels = [];
  for (let channel = 0; channel < context.channels; channel += 1) {
    const witness = reconstructChannel(context, channel, true);
    writeChannelLedger(ledger, context, witness);
    channels.push(stripTermRows(witness));
  }
  const ranked = [...channels].sort(compareChannelRanking);
  const fixed = channels.filter((channel) => channel.fixed_point_assessed);
  return {
    op_index: Number(op.index),
    op_name: String(op.name),
    assessment_status: "assessed",
    not_assessed_reason: "",
    input_code_range: context.inputRange,
    input_zero_point: context.inputZeroPoint,
    output_channel_axis: context.layout.axis,
    output_scale: context.outputScale,
    output_zero_point: context.outputZeroPoint,
    output_code_range: context.outputRange,
    fused_activation: String(op.fused_activation),
    activation_code_range: context.activationRange,
    assessed_channel_count: context.channels,
    fixed_point_assessed_channel_count: fixed.length,
    accumulation_terms_per_channel: context.terms,
    witness_assignment_count: context.channels * context.terms * 2,
    fixed_point_endpoint_evaluation_count: fixed.length * 4,
    default_ideal_mismatch_endpoint_count: countEndpoints(channels, (endpoint) => endpoint.default_output_code != null && endpoint.default_output_code !== endpoint.ideal_output_code),
    single_ideal_mismatch_endpoint_count: countEndpoints(channels, (endpoint) => endpoint.single_output_code != null && endpoint.single_output_code !== endpoint.ideal_output_code),
    build_mode_divergent_endpoint_count: countEndpoints(channels, (endpoint) => endpoint.default_output_code != null && endpoint.default_output_code !== endpoint.single_output_code),
    default_activation_clamped_endpoint_count: countEndpoints(channels, (endpoint) => endpoint.default_activation_clamped === true),
    single_activation_clamped_endpoint_count: countEndpoints(channels, (endpoint) => endpoint.single_activation_clamped === true),
    default_collapsed_extrema_channel_count: channels.filter((channel) => channel.minimum.default_output_code != null && channel.minimum.default_output_code === channel.maximum.default_output_code).length,
    single_collapsed_extrema_channel_count: channels.filter((channel) => channel.minimum.single_output_code != null && channel.minimum.single_output_code === channel.maximum.single_output_code).length,
    maximum_default_ideal_delta_codes: maxOptional(channels, "maximum_default_ideal_delta_codes"),
    maximum_single_ideal_delta_codes: maxOptional(channels, "maximum_single_ideal_delta_codes"),
    top_channels: ranked.slice(0, TOP_LIMIT),
    worst_channel: ranked[0] || null,
    channels,
    ledger_bytes: ledger.finish(),
  };
}

function kernelContext(op, tensors, bytes, accumulator, requant) {
  const input = requiredTensor(tensors, op.inputs?.[0], "Input tensor is unavailable.");
  const weight = requiredTensor(tensors, op.inputs?.[1], "Weight tensor is unavailable.");
  const output = requiredTensor(tensors, op.outputs?.[0], "Output tensor is unavailable.");
  const inputRange = codeRange(input.dtype, `Input tensor ${input.index} is not INT8 or UINT8.`);
  assert(input.zero_point_sample?.length === 1, `Input tensor ${input.index} does not expose one zero-point.`);
  const inputZeroPoint = Number(input.zero_point_sample[0]);
  assert(inputZeroPoint >= inputRange[0] && inputZeroPoint <= inputRange[1], `Input zero-point ${inputZeroPoint} lies outside its code range.`);
  const weightRange = codeRange(weight.dtype, `Weight tensor ${weight.index} is not INT8 or UINT8.`);
  const layout = weightLayout(op, weight);
  const zeroPoints = expandedZeroPoints(weight, layout.channels, weightRange);
  const rawWeights = tensorBytes(bytes, weight, `Weight tensor ${weight.index} constant bytes are unavailable.`);
  assert(rawWeights.length === layout.channels * layout.terms, `Weight tensor ${weight.index} byte count does not match its shape.`);
  const biases = decodeBias(op, tensors, bytes, layout.channels);
  assert(accumulator?.assessment_status === "assessed" && Number(accumulator.assessed_channel_count) === layout.channels, `Accumulator evidence is incomplete at op #${op.index}.`);
  assert(requant?.assessment_status === "assessed" && Number(requant.assessed_channel_count) === layout.channels, `Requantization evidence is incomplete at op #${op.index}.`);
  for (const key of ["channel_real_multipliers", "channel_quantized_multipliers", "channel_shifts", "channel_single_rounding_quantized_multipliers", "channel_single_rounding_shifts"]) {
    assert(requant[key]?.length === layout.channels, `Requantization ${key} cardinality mismatch at op #${op.index}.`);
  }
  const outputScale = Number(requant.output_scale);
  const outputZeroPoint = Number(requant.output_zero_point);
  const outputRange = requant.output_code_range?.map(Number);
  assert(Number.isFinite(outputScale) && outputScale > 0 && outputRange?.length === 2, `Output contract is incomplete at op #${op.index}.`);
  return {
    op,
    input,
    weight,
    output,
    inputRange,
    inputZeroPoint,
    layout,
    channels: layout.channels,
    terms: layout.terms,
    zeroPoints,
    rawWeights,
    biases,
    accumulator,
    requant,
    outputScale,
    outputZeroPoint,
    outputRange,
    activationRange: activationCodeRange(String(op.fused_activation), outputScale, outputZeroPoint, outputRange),
  };
}

function reconstructChannel(context, channel, includeTerms) {
  const termRows = includeTerms ? new Int16Array(context.terms * 3) : null;
  let positive = 0;
  let negative = 0;
  let zero = 0;
  let maxWeight = 0;
  let maxContribution = 0;
  let dotMinimum = 0n;
  let dotMaximum = 0n;
  const minimumHistogram = new Map();
  const maximumHistogram = new Map();
  const weightZeroPoint = BigInt(context.zeroPoints[channel]);
  for (let term = 0; term < context.terms; term += 1) {
    const centered = BigInt(rawCode(context.rawWeights[context.layout.rawIndex(channel, term)], context.weight.dtype)) - weightZeroPoint;
    const centeredNumber = Number(centered);
    const [minimumCode, maximumCode] = canonicalCodes(centeredNumber, context.inputRange, context.inputZeroPoint);
    if (centered > 0n) positive += 1;
    else if (centered < 0n) negative += 1;
    else zero += 1;
    increment(minimumHistogram, minimumCode);
    increment(maximumHistogram, maximumCode);
    const minimumContribution = centered * BigInt(minimumCode - context.inputZeroPoint);
    const maximumContribution = centered * BigInt(maximumCode - context.inputZeroPoint);
    dotMinimum += minimumContribution;
    dotMaximum += maximumContribution;
    maxWeight = Math.max(maxWeight, Math.abs(centeredNumber));
    maxContribution = Math.max(maxContribution, Number(maxBigInt(absBigInt(minimumContribution), absBigInt(maximumContribution))));
    if (termRows) {
      termRows[term * 3] = centeredNumber;
      termRows[term * 3 + 1] = minimumCode;
      termRows[term * 3 + 2] = maximumCode;
    }
  }
  const bias = BigInt(context.biases[channel]);
  const postMinimum = dotMinimum + bias;
  const postMaximum = dotMaximum + bias;
  assert(String(postMinimum) === context.accumulator.channel_post_bias_min_decimals[channel], `Minimum witness does not reproduce op #${context.op.index} channel ${channel}.`);
  assert(String(postMaximum) === context.accumulator.channel_post_bias_max_decimals[channel], `Maximum witness does not reproduce op #${context.op.index} channel ${channel}.`);
  const minimum = endpointProjection("minimum", minimumHistogram, dotMinimum, bias, postMinimum, context, channel);
  const maximum = endpointProjection("maximum", maximumHistogram, dotMaximum, bias, postMaximum, context, channel);
  const patternBytes = termRows ? patternLedger(context.op.index, channel, context.terms, termRows) : null;
  return {
    channel_index: channel,
    term_count: context.terms,
    input_code_range: context.inputRange,
    input_zero_point: context.inputZeroPoint,
    positive_centered_weight_count: positive,
    negative_centered_weight_count: negative,
    zero_centered_weight_count: zero,
    maximum_absolute_centered_weight: maxWeight,
    maximum_absolute_term_contribution: maxContribution,
    minimum,
    maximum,
    maximum_default_ideal_delta_codes: maxEndpointDelta(minimum.default_ideal_delta_codes, maximum.default_ideal_delta_codes),
    maximum_single_ideal_delta_codes: maxEndpointDelta(minimum.single_ideal_delta_codes, maximum.single_ideal_delta_codes),
    build_mode_divergent_endpoint_count: [minimum, maximum].filter((endpoint) => endpoint.default_output_code != null && endpoint.default_output_code !== endpoint.single_output_code).length,
    fixed_point_assessed: minimum.default_output_code != null && maximum.default_output_code != null && minimum.single_output_code != null && maximum.single_output_code != null,
    term_rows: termRows,
    pattern_bytes: patternBytes,
  };
}

function endpointProjection(endpoint, histogram, dot, bias, postBias, context, channel) {
  const realMultiplier = Number(context.requant.channel_real_multipliers[channel]);
  const idealPreclamp = absBigInt(postBias) <= (1n << 53n) ? roundTiesAway(Number(postBias) * realMultiplier) + context.outputZeroPoint : null;
  const idealOutput = idealPreclamp == null ? null : clamp(idealPreclamp, ...context.activationRange);
  const postInt32 = postBias >= -2147483648n && postBias <= 2147483647n ? Number(postBias) : null;
  const defaultScaled = postInt32 == null ? null : multiplyByQuantizedMultiplierDefault(postInt32, context.requant.channel_quantized_multipliers[channel], context.requant.channel_shifts[channel]);
  const singleScaled = postInt32 == null ? null : multiplyByQuantizedMultiplierSingleRounding(postInt32, context.requant.channel_single_rounding_quantized_multipliers[channel], context.requant.channel_single_rounding_shifts[channel]);
  const defaultPreclamp = defaultScaled == null ? null : defaultScaled + context.outputZeroPoint;
  const singlePreclamp = singleScaled == null ? null : singleScaled + context.outputZeroPoint;
  const defaultOutput = defaultPreclamp == null ? null : clamp(defaultPreclamp, ...context.activationRange);
  const singleOutput = singlePreclamp == null ? null : clamp(singlePreclamp, ...context.activationRange);
  return {
    endpoint,
    witness_code_histogram: [...histogram]
      .sort((left, right) => left[0] - right[0])
      .map(([code, count]) => ({ code, count })),
    dot_product_decimal: String(dot),
    bias_decimal: String(bias),
    post_bias_accumulator_decimal: String(postBias),
    ideal_preclamp_code: idealPreclamp,
    ideal_output_code: idealOutput,
    default_scaled_accumulator: defaultScaled,
    default_preclamp_code: defaultPreclamp,
    default_output_code: defaultOutput,
    default_activation_clamped: defaultPreclamp == null ? null : defaultPreclamp < context.activationRange[0] || defaultPreclamp > context.activationRange[1],
    default_ideal_delta_codes: delta(defaultOutput, idealOutput),
    single_scaled_accumulator: singleScaled,
    single_preclamp_code: singlePreclamp,
    single_output_code: singleOutput,
    single_activation_clamped: singlePreclamp == null ? null : singlePreclamp < context.activationRange[0] || singlePreclamp > context.activationRange[1],
    single_ideal_delta_codes: delta(singleOutput, idealOutput),
    build_mode_output_delta_codes: delta(defaultOutput, singleOutput),
  };
}

function writeChannelLedger(writer, context, witness) {
  for (const value of [context.op.index, witness.channel_index, context.terms, witness.positive_centered_weight_count, witness.negative_centered_weight_count, witness.zero_centered_weight_count, witness.minimum.bias_decimal, witness.minimum.post_bias_accumulator_decimal, witness.maximum.post_bias_accumulator_decimal]) writer.writeI64(value);
  for (const endpoint of [witness.minimum, witness.maximum]) {
    for (const value of [endpoint.ideal_preclamp_code, endpoint.ideal_output_code, endpoint.default_scaled_accumulator, endpoint.default_preclamp_code, endpoint.default_output_code, optionalBool(endpoint.default_activation_clamped), endpoint.single_scaled_accumulator, endpoint.single_preclamp_code, endpoint.single_output_code]) writer.writeI64(value ?? MISSING_I64);
  }
  for (let offset = 0; offset < witness.term_rows.length; offset += 1) writer.writeI16(witness.term_rows[offset]);
}

function patternLedger(opIndex, channel, terms, termRows) {
  const writer = new ByteWriter(PATTERN_PREFIX.byteLength + 24 + termRows.length * 2);
  writer.writeBytes(PATTERN_PREFIX);
  writer.writeI64(opIndex);
  writer.writeI64(channel);
  writer.writeI64(terms);
  for (const value of termRows) writer.writeI16(value);
  return writer.finish();
}

function stripTermRows(witness) {
  const { term_rows, pattern_bytes, ...summary } = witness;
  return summary;
}

function compareSummary(actual, expected) {
  for (const key of ["status", "candidate_op_count", "assessed_op_count", "unassessed_op_count", "assessed_channel_count", "fixed_point_assessed_channel_count", "witness_assignment_count", "fixed_point_endpoint_evaluation_count", "default_ideal_mismatch_endpoint_count", "single_ideal_mismatch_endpoint_count", "build_mode_divergent_endpoint_count", "default_activation_clamped_endpoint_count", "single_activation_clamped_endpoint_count", "default_collapsed_extrema_channel_count", "single_collapsed_extrema_channel_count", "maximum_default_ideal_delta_codes", "maximum_single_ideal_delta_codes"]) {
    assert(actual[key] === expected[key], `Kernel witness ${key} mismatch.`);
  }
  assert(equalArray(actual.witness_ranking_op_indices, expected.witness_ranking_op_indices), "Kernel witness ranking mismatch.");
}

function compareOpRow(actual, expected) {
  assert(Number(actual.op_index) === expected.op_index && actual.op_name === expected.op_name, "Kernel witness op binding mismatch.");
  assert(actual.assessment_status === expected.assessment_status, `Kernel witness assessment mismatch at op #${expected.op_index}.`);
  assert(actual.not_assessed_reason === expected.not_assessed_reason, `Kernel witness reason mismatch at op #${expected.op_index}.`);
  if (expected.assessment_status !== "assessed") return;
  for (const key of ["input_zero_point", "output_channel_axis", "output_scale", "output_zero_point", "fused_activation", "assessed_channel_count", "fixed_point_assessed_channel_count", "accumulation_terms_per_channel", "witness_assignment_count", "fixed_point_endpoint_evaluation_count", "default_ideal_mismatch_endpoint_count", "single_ideal_mismatch_endpoint_count", "build_mode_divergent_endpoint_count", "default_activation_clamped_endpoint_count", "single_activation_clamped_endpoint_count", "default_collapsed_extrema_channel_count", "single_collapsed_extrema_channel_count", "maximum_default_ideal_delta_codes", "maximum_single_ideal_delta_codes"]) {
    assert(actual[key] === expected[key], `Kernel witness ${key} mismatch at op #${expected.op_index}.`);
  }
  for (const key of ["input_code_range", "output_code_range", "activation_code_range"]) assert(equalArray(actual[key], expected[key]), `Kernel witness ${key} mismatch at op #${expected.op_index}.`);
  assert(actual.top_channels.length === expected.top_channels.length, `Kernel witness top-channel count mismatch at op #${expected.op_index}.`);
  actual.top_channels.forEach((witness, index) => compareWitness(witness, expected.top_channels[index], expected.op_index));
  compareWitness(actual.worst_channel, expected.worst_channel, expected.op_index);
}

function compareWitness(actual, expected, opIndex) {
  assert(actual && expected, `Kernel witness is missing at op #${opIndex}.`);
  for (const key of ["channel_index", "term_count", "positive_centered_weight_count", "negative_centered_weight_count", "zero_centered_weight_count", "maximum_absolute_centered_weight", "maximum_absolute_term_contribution", "maximum_default_ideal_delta_codes", "maximum_single_ideal_delta_codes", "build_mode_divergent_endpoint_count"]) assert(actual[key] === expected[key], `Kernel witness ${key} mismatch at op #${opIndex}.`);
  compareEndpoint(actual.minimum, expected.minimum, opIndex);
  compareEndpoint(actual.maximum, expected.maximum, opIndex);
  assert(/^[a-f0-9]{64}$/.test(actual.witness_pattern_sha256 || ""), `Kernel witness pattern digest syntax mismatch at op #${opIndex}.`);
}

function compareEndpoint(actual, expected, opIndex) {
  for (const key of ["endpoint", "dot_product_decimal", "bias_decimal", "post_bias_accumulator_decimal", "ideal_preclamp_code", "ideal_output_code", "default_scaled_accumulator", "default_preclamp_code", "default_output_code", "default_activation_clamped", "default_ideal_delta_codes", "single_scaled_accumulator", "single_preclamp_code", "single_output_code", "single_activation_clamped", "single_ideal_delta_codes", "build_mode_output_delta_codes"]) assert(actual[key] === expected[key], `Kernel witness endpoint ${key} mismatch at op #${opIndex}.`);
  assert(JSON.stringify(actual.witness_code_histogram) === JSON.stringify(expected.witness_code_histogram), `Kernel witness code histogram mismatch at op #${opIndex}.`);
}

function compareSelectedTopWitness(row, selected) {
  const stored = row.top_channels.find((witness) => witness.channel_index === selected.channel_index);
  if (!stored) return;
  compareWitness(stored, stripTermRows(selected), row.op_index);
  selected.witness_pattern_sha256 = stored.witness_pattern_sha256;
}

export function drawKernelWitnessCanvas(canvas, selected, endpoint = "maximum", mode = "pattern", logicalWidth = null, logicalHeight = null) {
  if (!canvas || !selected?.term_rows) return null;
  const width = logicalWidth || Math.max(300, Math.floor(canvas.parentElement?.clientWidth || canvas.clientWidth || 720));
  const height = logicalHeight || Math.max(220, Math.round(width * 0.54));
  const dpr = Math.max(1, Math.min(2, globalThis.devicePixelRatio || 1));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.fillStyle = "#0d1518";
  context.fillRect(0, 0, width, height);
  const count = selected.term_count;
  const columns = Math.min(64, Math.max(3, Math.ceil(Math.sqrt(count * 1.7))));
  const rows = Math.ceil(count / columns);
  const gap = count > 512 ? 1 : 2;
  const padX = 20;
  const padTop = 32;
  const padBottom = 32;
  const cellWidth = (width - padX * 2 - gap * (columns - 1)) / columns;
  const cellHeight = (height - padTop - padBottom - gap * (rows - 1)) / rows;
  let maxContribution = 1;
  for (let term = 0; term < count; term += 1) {
    const weight = selected.term_rows[term * 3];
    const code = selected.term_rows[term * 3 + (endpoint === "minimum" ? 1 : 2)];
    maxContribution = Math.max(maxContribution, Math.abs(weight * (code - selected.input_zero_point)));
  }
  for (let term = 0; term < count; term += 1) {
    const column = term % columns;
    const row = Math.floor(term / columns);
    const x = padX + column * (cellWidth + gap);
    const y = padTop + row * (cellHeight + gap);
    const weight = selected.term_rows[term * 3];
    const code = selected.term_rows[term * 3 + (endpoint === "minimum" ? 1 : 2)];
    const contribution = weight * (code - selected.input_zero_point);
    context.fillStyle = mode === "contribution"
      ? contributionColor(contribution, maxContribution)
      : codeColor(code, selected.input_code_range, selected.input_zero_point);
    context.fillRect(x, y, Math.max(1, cellWidth), Math.max(1, cellHeight));
  }
  context.fillStyle = "#d8e5e7";
  context.font = "600 12px system-ui";
  context.fillText(`${endpoint.toUpperCase()} / ${mode === "contribution" ? "SIGNED TERM CONTRIBUTION" : "CANONICAL INPUT CODE"}`, padX, 19);
  context.fillStyle = "#8fa4a8";
  context.font = "11px system-ui";
  context.fillText(`${formatNumber(count)} ordered terms`, padX, height - 10);
  canvas.__kernelWitnessState = { selected, endpoint, columns, rows, gap, padX, padTop, cellWidth, cellHeight, width, height };
  return canvas;
}

function installCanvasInteraction(canvas, tooltip) {
  canvas.addEventListener("pointermove", (event) => {
    const state = canvas.__kernelWitnessState;
    if (!state) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const column = Math.floor((x - state.padX) / (state.cellWidth + state.gap));
    const row = Math.floor((y - state.padTop) / (state.cellHeight + state.gap));
    const term = row * state.columns + column;
    if (column < 0 || row < 0 || column >= state.columns || term >= state.selected.term_count) {
      tooltip.hidden = true;
      return;
    }
    const cellX = state.padX + column * (state.cellWidth + state.gap);
    const cellY = state.padTop + row * (state.cellHeight + state.gap);
    if (x > cellX + state.cellWidth || y > cellY + state.cellHeight) {
      tooltip.hidden = true;
      return;
    }
    const weight = state.selected.term_rows[term * 3];
    const code = state.selected.term_rows[term * 3 + (state.endpoint === "minimum" ? 1 : 2)];
    const contribution = weight * (code - state.selected.input_zero_point);
    tooltip.innerHTML = `<strong>term ${term}</strong><span>input code ${code}</span><span>centered weight ${signed(weight)}</span><span>contribution ${signed(contribution)}</span>`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(state.width - 190, Math.max(8, x + 12))}px`;
    tooltip.style.top = `${Math.min(state.height - 90, Math.max(8, y + 12))}px`;
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

function renderKernelWitnessSummary(root, evidence) {
  if (!root) return;
  const endpointCount = evidence.fixed_point_assessed_channel_count * 2;
  root.replaceChildren(
    metric("Canonical assignments", formatNumber(evidence.witness_assignment_count), "two exact endpoint patterns"),
    metric("Fixed-point executions", formatNumber(evidence.fixed_point_endpoint_evaluation_count), "default plus single rounding"),
    metric("Default vs ideal", formatNumber(evidence.default_ideal_mismatch_endpoint_count), `${formatPercent(evidence.default_ideal_mismatch_endpoint_count / Math.max(1, endpointCount))} of endpoint witnesses`),
    metric("Build-mode deltas", formatNumber(evidence.build_mode_divergent_endpoint_count), `max ${evidence.maximum_default_ideal_delta_codes ?? "N/A"} code`),
    metric("Default clamps", formatNumber(evidence.default_activation_clamped_endpoint_count), `${formatPercent(evidence.default_activation_clamped_endpoint_count / Math.max(1, endpointCount))} of legal-domain extrema`),
  );
}

function endpointPath(endpoint) {
  const root = document.createElement("div");
  root.className = "kernel-witness-path";
  for (const [label, value, tone] of [
    ["Dot", formatInteger(endpoint.dot_product_decimal), "neutral"],
    ["+ bias", formatInteger(endpoint.post_bias_accumulator_decimal), "neutral"],
    ["Ideal", formatOptional(endpoint.ideal_output_code), "ideal"],
    ["Default", formatOptional(endpoint.default_output_code), endpoint.default_ideal_delta_codes ? "watch" : "ok"],
    ["Single", formatOptional(endpoint.single_output_code), endpoint.single_ideal_delta_codes ? "watch" : "ok"],
  ]) {
    const item = document.createElement("div");
    item.className = `kernel-witness-stage ${tone}`;
    const name = document.createElement("span");
    name.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    item.append(name, result);
    root.append(item);
  }
  return root;
}

function endpointComparison(selected) {
  const wrap = document.createElement("div");
  wrap.className = "kernel-witness-table-wrap";
  const table = document.createElement("table");
  table.className = "kernel-witness-table";
  table.innerHTML = "<thead><tr><th>Endpoint</th><th>Input codes</th><th>Dot</th><th>Post-bias</th><th>Ideal</th><th>Default</th><th>Single</th><th>Mode delta</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const endpoint of [selected.minimum, selected.maximum]) {
    const tr = document.createElement("tr");
    for (const value of [
      endpoint.endpoint,
      endpoint.witness_code_histogram.map((item) => `${item.code} x ${formatNumber(item.count)}`).join(", "),
      formatInteger(endpoint.dot_product_decimal),
      formatInteger(endpoint.post_bias_accumulator_decimal),
      formatOptional(endpoint.ideal_output_code),
      `${formatOptional(endpoint.default_output_code)}${endpoint.default_activation_clamped ? " clipped" : ""}`,
      `${formatOptional(endpoint.single_output_code)}${endpoint.single_activation_clamped ? " clipped" : ""}`,
      signed(endpoint.build_mode_output_delta_codes || 0),
    ]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

function rankingTable(evidence, selectedOpIndex, onSelect, jumpToGraphOp) {
  const wrap = document.createElement("div");
  wrap.className = "kernel-witness-table-wrap";
  const table = document.createElement("table");
  table.className = "kernel-witness-table";
  table.innerHTML = "<thead><tr><th>Op</th><th>Channels x terms</th><th>Build-mode endpoints</th><th>Default clamps</th><th>Collapsed spans</th><th>Ledger</th><th></th></tr></thead>";
  const rows = evidence.witness_ranking_op_indices.map((index) => evidence.ops.find((row) => row.op_index === index)).filter(Boolean).slice(0, 16);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.classList.toggle("selected", row.op_index === selectedOpIndex);
    const coordinate = document.createElement("button");
    coordinate.type = "button";
    coordinate.className = "table-link";
    coordinate.textContent = `#${row.op_index} ${row.op_name}`;
    coordinate.addEventListener("click", () => onSelect(row.op_index));
    const graph = document.createElement("button");
    graph.type = "button";
    graph.className = "icon-action";
    graph.textContent = "->";
    graph.title = `Open op #${row.op_index} in Graph Explorer`;
    graph.addEventListener("click", () => jumpToGraphOp?.(row.op_index));
    for (const value of [coordinate, `${formatNumber(row.assessed_channel_count)} x ${formatNumber(row.accumulation_terms_per_channel)}`, formatNumber(row.build_mode_divergent_endpoint_count), formatNumber(row.default_activation_clamped_endpoint_count), formatNumber(row.default_collapsed_extrema_channel_count), row.witness_ledger_sha256.slice(0, 12), graph]) {
      const td = document.createElement("td");
      if (value instanceof Node) td.append(value); else td.textContent = value;
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

function selectedWitnessExport(analysis, row, selected, endpoint) {
  const column = endpoint === "minimum" ? 1 : 2;
  const terms = [];
  for (let term = 0; term < selected.term_count; term += 1) {
    const centeredWeight = selected.term_rows[term * 3];
    const inputCode = selected.term_rows[term * 3 + column];
    terms.push({
      term_index: term,
      input_code: inputCode,
      centered_input: inputCode - selected.input_zero_point,
      centered_weight: centeredWeight,
      integer_contribution: centeredWeight * (inputCode - selected.input_zero_point),
    });
  }
  return {
    schema: "deepbom.kernel_extremum_selected_witness.v1",
    artifact_sha256: analysis.model_sha256,
    source_commit: SOURCE_COMMIT,
    op_index: row.op_index,
    op_name: row.op_name,
    output_channel: selected.channel_index,
    endpoint,
    term_order: row.op_name === "DEPTHWISE_CONV_2D" ? "filter [H,W,O] for selected O" : row.op_name === "CONV_2D" ? "filter OHWI within selected O" : "filter output-major input-feature order",
    full_valid_receptive_field: true,
    input_tensor_index: row.input_tensor_index,
    input_code_range: row.input_code_range,
    input_zero_point: row.input_zero_point,
    output_tensor_index: row.output_tensor_index,
    output_code_range: row.output_code_range,
    output_zero_point: row.output_zero_point,
    fused_activation: row.fused_activation,
    projection: selected[endpoint],
    pattern_sha256: selected.witness_pattern_sha256,
    terms,
    interpretation_boundary: analysis.kernel_extremum_witness.interpretation_boundary,
  };
}

class ByteWriter {
  constructor(size) {
    this.bytes = new Uint8Array(size);
    this.view = new DataView(this.bytes.buffer);
    this.offset = 0;
  }
  writeBytes(bytes) { this.bytes.set(bytes, this.offset); this.offset += bytes.byteLength; }
  writeI64(value) { this.view.setBigInt64(this.offset, BigInt(value), true); this.offset += 8; }
  writeI16(value) { this.view.setInt16(this.offset, Number(value), true); this.offset += 2; }
  finish() { assert(this.offset === this.bytes.byteLength, "Kernel witness ledger byte count mismatch."); return this.bytes; }
}

function segmentedControl(label, options, value, onChange) {
  const root = document.createElement("div");
  root.className = "kernel-witness-segments";
  root.setAttribute("role", "tablist");
  root.setAttribute("aria-label", label);
  for (const [id, text] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.classList.toggle("active", id === value);
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(id === value));
    button.addEventListener("click", () => onChange(id));
    root.append(button);
  }
  return root;
}

function metric(label, value, detail) {
  const node = document.createElement("div");
  node.className = "kernel-witness-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  const note = document.createElement("small");
  note.textContent = detail;
  node.append(name, number, note);
  return node;
}

function definitionTable(rows) {
  const dl = document.createElement("dl");
  dl.className = "kernel-witness-definitions";
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.append(dt, dd);
  }
  return dl;
}

function unassessedTable(rows) {
  const unassessed = (rows || []).filter((row) => row.assessment_status !== "assessed");
  if (!unassessed.length) return document.createDocumentFragment();
  const root = document.createElement("div");
  root.className = "kernel-witness-unassessed";
  for (const row of unassessed) {
    const item = document.createElement("p");
    item.textContent = `#${row.op_index} ${row.op_name}: ${row.not_assessed_reason}`;
    root.append(item);
  }
  return root;
}

function messageNode(text, tone = "muted") {
  const node = document.createElement("p");
  node.className = `kernel-witness-message ${tone}`;
  node.textContent = text;
  return node;
}

function tensorMap(analysis) { return new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor])); }
function requiredTensor(tensors, index, message) { const tensor = tensors.get(Number(index)); assert(tensor, message); return tensor; }
function tensorBytes(bytes, tensor, message) { const offset = Number(tensor.buffer_data_offset); const length = Number(tensor.buffer_data_length); assert(Number.isSafeInteger(offset) && Number.isSafeInteger(length) && length > 0 && offset >= 0 && offset + length <= bytes.length, message); return bytes.subarray(offset, offset + length); }
function codeRange(dtype, message) { if (dtype === "INT8") return [-128, 127]; if (dtype === "UINT8") return [0, 255]; throw new Error(message); }
function rawCode(byte, dtype) { return dtype === "INT8" ? (byte << 24) >> 24 : byte; }
function expandedZeroPoints(tensor, channels, range) { const values = (tensor.zero_point_sample || []).map(Number); assert(values.length === 1 || values.length === channels, `Weight tensor ${tensor.index} zero-point cardinality mismatch.`); assert(values.every((value) => value >= range[0] && value <= range[1]), `Weight tensor ${tensor.index} zero-point range mismatch.`); return values.length === 1 ? Array(channels).fill(values[0]) : values; }
function weightLayout(op, weight) { const dimensions = (weight.shape || []).map(Number); assert(dimensions.every((value) => Number.isSafeInteger(value) && value > 0), `Weight tensor ${weight.index} has an invalid shape.`); if (op.name === "CONV_2D" && dimensions.length === 4) return { channels: dimensions[0], terms: product(dimensions.slice(1)), axis: 0, rawIndex: (channel, term) => channel * product(dimensions.slice(1)) + term }; if (op.name === "DEPTHWISE_CONV_2D" && dimensions.length === 4 && dimensions[0] === 1) return { channels: dimensions[3], terms: dimensions[1] * dimensions[2], axis: 3, rawIndex: (channel, term) => term * dimensions[3] + channel }; if (op.name === "FULLY_CONNECTED" && dimensions.length === 2) return { channels: dimensions[0], terms: dimensions[1], axis: 0, rawIndex: (channel, term) => channel * dimensions[1] + term }; throw new Error(`${op.name} weight layout is outside the witness contract.`); }
function decodeBias(op, tensors, bytes, channels) { const index = Number(op.inputs?.[2]); if (!Number.isInteger(index) || index < 0) return Array(channels).fill(0); const tensor = requiredTensor(tensors, index, `Bias tensor ${index} is unavailable.`); assert(tensor.dtype === "INT32", `Bias tensor ${index} is not INT32.`); const raw = tensorBytes(bytes, tensor, `Bias tensor ${index} bytes are unavailable.`); assert(raw.length === channels * 4, `Bias tensor ${index} cardinality mismatch.`); const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength); return Array.from({ length: channels }, (_, channel) => view.getInt32(channel * 4, true)); }
function activationCodeRange(activation, scale, zeroPoint, range) { const quantize = (real) => clamp(roundTiesAway(real / scale) + zeroPoint, ...range); if (activation === "NONE") return range; if (activation === "RELU") return [quantize(0), range[1]]; if (activation === "RELU_N1_TO_1") return [quantize(-1), quantize(1)]; if (activation === "RELU6") return [quantize(0), quantize(6)]; throw new Error(`Fused activation ${activation} is outside the pinned integer witness contract.`); }
function canonicalCodes(weight, range, zeroPoint) { return weight > 0 ? [range[0], range[1]] : weight < 0 ? [range[1], range[0]] : [zeroPoint, zeroPoint]; }
function optionalBool(value) { return value == null ? null : value ? 1 : 0; }
function increment(map, key) { map.set(key, (map.get(key) || 0) + 1); }
function countEndpoints(channels, predicate) { return channels.reduce((count, channel) => count + [channel.minimum, channel.maximum].filter(predicate).length, 0); }
function compareChannelRanking(left, right) { return (right.maximum_default_ideal_delta_codes ?? -1) - (left.maximum_default_ideal_delta_codes ?? -1) || right.build_mode_divergent_endpoint_count - left.build_mode_divergent_endpoint_count || compareBigIntDesc(endpointAccumulatorAbs(left), endpointAccumulatorAbs(right)) || left.channel_index - right.channel_index; }
function compareOpRanking(left, right) { return (right.maximum_default_ideal_delta_codes ?? -1) - (left.maximum_default_ideal_delta_codes ?? -1) || right.default_activation_clamped_endpoint_count - left.default_activation_clamped_endpoint_count || left.op_index - right.op_index; }
function endpointAccumulatorAbs(witness) { return maxBigInt(absBigInt(BigInt(witness.minimum.post_bias_accumulator_decimal)), absBigInt(BigInt(witness.maximum.post_bias_accumulator_decimal))); }
function compareBigIntDesc(left, right) { return left === right ? 0 : left > right ? -1 : 1; }
function maxEndpointDelta(...values) { const filtered = values.filter((value) => value != null).map(Math.abs); return filtered.length ? Math.max(...filtered) : null; }
function maxOptional(rows, key) { const values = rows.map((row) => row[key]).filter((value) => value != null); return values.length ? Math.max(...values) : null; }
function delta(left, right) { return left == null || right == null ? null : left - right; }
function clamp(value, minimum, maximum) { return Math.min(maximum, Math.max(minimum, value)); }
function clampInteger(value, minimum, maximum) { const number = Number(value); return clamp(Number.isInteger(number) ? number : minimum, minimum, maximum); }
function absBigInt(value) { return value < 0n ? -value : value; }
function maxBigInt(left, right) { return left > right ? left : right; }
function sum(rows, key) { return rows.reduce((total, row) => total + Number(row[key] || 0), 0); }
function product(values) { return values.reduce((total, value) => total * value, 1); }
function equalArray(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function asBytes(value) { if (value instanceof Uint8Array) return value; if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength); if (value instanceof ArrayBuffer) return new Uint8Array(value); throw new Error("Kernel witness verification requires model bytes."); }
function codeColor(code, range, zeroPoint) { if (code === zeroPoint) return "#7f9296"; const t = (code - range[0]) / Math.max(1, range[1] - range[0]); const low = [33, 181, 177]; const high = [239, 104, 80]; return `rgb(${Math.round(low[0] + (high[0] - low[0]) * t)},${Math.round(low[1] + (high[1] - low[1]) * t)},${Math.round(low[2] + (high[2] - low[2]) * t)})`; }
function contributionColor(value, maximum) { const intensity = Math.log2(Math.abs(value) + 1) / Math.log2(maximum + 1); const base = value < 0 ? [47, 154, 187] : value > 0 ? [235, 116, 75] : [99, 116, 120]; return `rgb(${Math.round(20 + base[0] * intensity * 0.85)},${Math.round(25 + base[1] * intensity * 0.85)},${Math.round(28 + base[2] * intensity * 0.85)})`; }
function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatPercent(value) { return `${(Number(value || 0) * 100).toFixed(2)}%`; }
function formatInteger(value) { try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "N/A"); } }
function formatOptional(value) { return value == null ? "N/A" : formatNumber(value); }
function signed(value) { const number = Number(value || 0); return `${number > 0 ? "+" : ""}${number.toLocaleString("en-US")}`; }
function shapeText(shape) { return `[${(shape || []).join(" x ")}]`; }
function assert(condition, message) { if (!condition) throw new Error(message); }
