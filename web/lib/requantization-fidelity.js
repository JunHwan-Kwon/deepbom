import { validateAccumulatorAtlas } from "./accumulator-atlas.js";
import { sha256Hex } from "./hash.js";
import { quantizeMultiplier } from "./quantization-math.js";

export const REQUANTIZATION_FIDELITY_SCHEMA = "deepbom.requantization_fidelity.v1";
const METHOD_VERSION = "2026-07-17.2";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const SOURCE_SHA256 = Object.freeze({
  "tensorflow/lite/kernels/internal/quantization_util.cc": "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
  "tensorflow/lite/kernels/kernel_util.cc": "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
  "tensorflow/lite/kernels/internal/common.cc": "ba5308bf76383d600d033c948fe0659710939e6f1f15a800b5413e5fc822ddfa",
});

export function createRequantizationFidelityController({
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
  let fidelity = null;
  let reconstructed = null;
  let selectedOpIndex = null;
  let mode = "encoding";
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (fidelity) onDownload?.(fidelity, "requantization_fidelity.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    const context = getContext?.() || {};
    analysis = explicitAnalysis || context.analysis || null;
    fidelity = analysis?.requantization_fidelity || null;
    const bytes = context.modelBytes || null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (!fidelity || !bytes || String(analysis?.format || "").toLowerCase() !== "tflite") {
      selectedOpIndex = null;
      reconstructed = null;
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      reconstructed = validateRequantizationFidelity(analysis, bytes);
      const assessed = fidelity.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = fidelity.fidelity_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
      }
      renderSummary(summary, fidelity);
      renderBody();
      if (status) {
        status.textContent = fidelity.assessed_op_count ? "arithmetic verified / digest pending" : fidelity.status.replaceAll("_", " ");
        status.dataset.tone = fidelity.default_pre_shift_overflow_channel_count ? "risk" : fidelity.assessed_op_count ? "ok" : "muted";
      }
      if (fidelity.assessed_op_count) {
        void validateRequantizationFidelityDigests(analysis, bytes).then(() => {
          if (token !== renderToken || !status) return;
          status.textContent = "independently verified";
          status.dataset.tone = fidelity.default_pre_shift_overflow_channel_count ? "risk" : "ok";
        }).catch((error) => {
          if (token !== renderToken || !status) return;
          status.textContent = `integrity error: ${error.message}`;
          status.dataset.tone = "risk";
        });
      }
    } catch (error) {
      reconstructed = null;
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Requantization evidence rejected: ${error.message}`, "risk"));
      if (status) {
        status.textContent = "evidence rejected";
        status.dataset.tone = "risk";
      }
    }
  }

  function renderBody() {
    if (!body || !fidelity) return;
    resizeObserver?.disconnect();
    const assessed = fidelity.ops.filter((row) => row.assessment_status === "assessed");
    if (!assessed.length) {
      body.replaceChildren(messageNode("No quantized convolution-family output scale was assessable."), unassessedTable(fidelity.ops));
      return;
    }
    const row = assessed.find((candidate) => candidate.op_index === selectedOpIndex) || assessed[0];
    selectedOpIndex = row.op_index;
    const toolbar = document.createElement("div");
    toolbar.className = "requant-toolbar";
    const select = document.createElement("select");
    select.className = "requant-op-select";
    select.setAttribute("aria-label", "Requantization operator");
    for (const candidate of assessed) {
      const option = new Option(
        `#${candidate.op_index} ${candidate.op_name} / ${formatNumber(candidate.assessed_channel_count)} ch / shifts ${signed(candidate.minimum_shift)}..${signed(candidate.maximum_shift)}`,
        String(candidate.op_index),
      );
      option.selected = candidate.op_index === row.op_index;
      select.append(option);
    }
    select.addEventListener("change", () => {
      selectedOpIndex = Number(select.value);
      renderBody();
    });
    toolbar.append(select, modeControl());

    const main = document.createElement("div");
    main.className = "requant-main";
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "requant-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "requant-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Per-channel requantization map for operator ${row.op_index}`);
    const tooltip = document.createElement("div");
    tooltip.className = "requant-tooltip";
    tooltip.hidden = true;
    canvasWrap.append(canvas, tooltip);
    installCanvasInteraction(canvas, tooltip, row);
    const details = document.createElement("div");
    details.className = "requant-details";
    details.append(
      definitionTable([
        ["Scale contract", `${formatScale(row.input_scale)} x ${row.weight_scale_mode.replaceAll("_", " ")} / ${formatScale(row.output_scale)}`],
        ["Output contract", `${row.output_dtype} [${row.output_code_range.join(", ")}], zp ${row.output_zero_point}`],
        ["Build modes", `default double rounding + TFLITE_SINGLE_ROUNDING; artifact flag unresolved`],
        ["Pinned sources", sourceLinks(fidelity)],
        ["Channel ledger", row.channel_ledger_sha256],
      ]),
      witnessTable(row.worst_channel),
    );
    main.append(canvasWrap, details);
    const boundary = document.createElement("p");
    boundary.className = "requant-boundary";
    boundary.textContent = fidelity.interpretation_boundary;
    body.replaceChildren(toolbar, main, shiftHistogram(row), rankingTable(fidelity, row.op_index, jumpToGraphOp, (opIndex) => {
      selectedOpIndex = opIndex;
      renderBody();
    }), boundary, unassessedTable(fidelity.ops));
    requestAnimationFrame(() => drawRequantizationFidelityCanvas(canvas, row, mode));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawRequantizationFidelityCanvas(canvas, row, mode));
      resizeObserver.observe(canvasWrap);
    }
  }

  function modeControl() {
    const control = document.createElement("div");
    control.className = "requant-mode-control";
    control.setAttribute("role", "tablist");
    control.setAttribute("aria-label", "Requantization map mode");
    for (const [id, label] of [["encoding", "Encoding drift"], ["rounding", "Rounding bound"], ["shift", "Exponent shift"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.dataset.requantMode = id;
      button.classList.toggle("active", mode === id);
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(mode === id));
      button.addEventListener("click", () => {
        mode = id;
        renderBody();
      });
      control.append(button);
    }
    return control;
  }

  return { render, getReconstructed: () => reconstructed };
}

export function validateRequantizationFidelity(analysis, modelBytes) {
  validateAccumulatorAtlas(analysis, modelBytes);
  const actual = analysis?.requantization_fidelity;
  assert(actual?.schema === REQUANTIZATION_FIDELITY_SCHEMA, "Requantization schema mismatch.");
  assert(actual.method_version === METHOD_VERSION, "Requantization method version mismatch.");
  assert(actual.evidence_class === "DERIVED", "Requantization evidence class must be DERIVED.");
  assert(actual.source_commit === SOURCE_COMMIT, "Requantization source commit mismatch.");
  validateSources(actual.source_references);
  assert(String(actual.interpretation_boundary || "").includes("does not identify the runtime's TFLITE_SINGLE_ROUNDING compile flag"), "Requantization build boundary is incomplete.");
  assert(String(actual.interpretation_boundary || "").includes("estimate model accuracy"), "Requantization accuracy boundary is incomplete.");
  const expected = reconstructRequantizationFidelity(analysis);
  compareAnalysis(actual, expected);
  return expected;
}

export async function validateRequantizationFidelityDigests(analysis, modelBytes) {
  const reconstructed = validateRequantizationFidelity(analysis, modelBytes);
  const rows = analysis.requantization_fidelity.ops;
  for (let index = 0; index < rows.length; index += 1) {
    if (rows[index].assessment_status !== "assessed") continue;
    const digest = await sha256Hex(new TextEncoder().encode(reconstructed.ops[index].ledger_text));
    assert(digest === rows[index].channel_ledger_sha256, `Requantization ledger SHA-256 mismatch at op #${rows[index].op_index}.`);
  }
  return reconstructed;
}

export function reconstructRequantizationFidelity(analysis) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const ops = new Map((analysis?.ops || []).map((op) => [Number(op.index), op]));
  const accumulatorRows = analysis?.accumulator_atlas?.ops || [];
  const rows = accumulatorRows.map((accumulator) => {
    const op = ops.get(Number(accumulator.op_index));
    if (!op) return null;
    try {
      return reconstructOp(op, tensors, accumulator);
    } catch (error) {
      return notAssessedRow(op, tensors, error.message);
    }
  }).filter(Boolean);
  const assessed = rows.filter((row) => row.assessment_status === "assessed");
  const ranking = [...assessed].sort((left, right) =>
    right.maximum_encoding_drift_bound_codes - left.maximum_encoding_drift_bound_codes
    || right.maximum_relative_multiplier_error - left.maximum_relative_multiplier_error
    || left.op_index - right.op_index);
  const shiftCounts = new Map();
  for (const row of assessed) {
    for (const bin of row.shift_histogram) shiftCounts.set(bin.shift, (shiftCounts.get(bin.shift) || 0) + bin.channel_count);
  }
  const candidateOpCount = rows.length;
  const unassessedOpCount = candidateOpCount - assessed.length;
  return {
    status: candidateOpCount === 0 ? "not_applicable" : unassessedOpCount ? "partial" : "assessed",
    candidate_op_count: candidateOpCount,
    assessed_op_count: assessed.length,
    unassessed_op_count: unassessedOpCount,
    assessed_channel_count: sum(assessed, "assessed_channel_count"),
    fixed_point_bound_channel_count: sum(assessed, "fixed_point_bound_channel_count"),
    per_tensor_weight_op_count: assessed.filter((row) => row.weight_scale_mode === "per_tensor").length,
    per_axis_weight_op_count: assessed.filter((row) => row.weight_scale_mode === "per_output_channel").length,
    default_pre_shift_overflow_channel_count: sum(assessed, "default_pre_shift_overflow_channel_count"),
    single_rounding_encoding_divergence_channel_count: sum(assessed, "single_rounding_encoding_divergence_channel_count"),
    half_code_encoding_drift_channel_count: sum(assessed, "half_code_encoding_drift_channel_count"),
    one_code_encoding_drift_channel_count: sum(assessed, "one_code_encoding_drift_channel_count"),
    minimum_shift: optionalMin(assessed.map((row) => row.minimum_shift)),
    maximum_shift: optionalMax(assessed.map((row) => row.maximum_shift)),
    shift_histogram: [...shiftCounts.entries()].sort((left, right) => left[0] - right[0]).map(([shift, channel_count]) => ({ shift, channel_count })),
    maximum_relative_multiplier_error: optionalMax(assessed.map((row) => row.maximum_relative_multiplier_error)),
    maximum_multiplier_error_ppm: optionalMax(assessed.map((row) => row.maximum_multiplier_error_ppm)),
    maximum_encoding_drift_bound_codes: optionalMax(assessed.map((row) => row.maximum_encoding_drift_bound_codes)),
    maximum_default_double_rounding_bound_codes: optionalMax(assessed.map((row) => row.maximum_default_double_rounding_bound_codes)),
    maximum_single_rounding_bound_codes: optionalMax(assessed.map((row) => row.maximum_single_rounding_bound_codes)),
    fidelity_ranking_op_indices: ranking.map((row) => row.op_index),
    ops: rows,
  };
}

function reconstructOp(op, tensors, accumulator) {
  if (accumulator.assessment_status !== "assessed") throw new Error(`Accumulator atlas row is ${accumulator.assessment_status}.`);
  const input = requiredTensor(tensors, op.inputs?.[0], "Input");
  const weight = requiredTensor(tensors, op.inputs?.[1], "Weight");
  const output = requiredTensor(tensors, op.outputs?.[0], "Output");
  const channels = Number(accumulator.output_channel_count || 0);
  assert(channels > 0 && Number(accumulator.assessed_channel_count) === channels, "Accumulator output-channel cardinality is unavailable.");
  assert(input.scale_sample?.length === 1 && output.scale_sample?.length === 1, "Input and output require one per-tensor scale.");
  const inputScale = Number(input.scale_sample[0]);
  const outputScale = Number(output.scale_sample[0]);
  assert(validScale(inputScale) && validScale(outputScale), "Input or output scale is non-finite or non-positive.");
  assert((weight.scale_sample?.length === 1 || weight.scale_sample?.length === channels)
    && weight.scale_sample.every((scale) => validScale(Number(scale))), `Weight scale cardinality must be 1 or ${channels}.`);
  const outputCodeRange = quantizedCodeRange(output.dtype);
  assert(outputCodeRange, `Output dtype ${output.dtype} is not INT8 or UINT8.`);
  assert(output.zero_point_sample?.length === 1
    && Number(output.zero_point_sample[0]) >= outputCodeRange[0]
    && Number(output.zero_point_sample[0]) <= outputCodeRange[1], "Output requires one in-range zero-point.");
  assert(accumulator.channel_accumulator_envelope_min_decimals?.length === channels
    && accumulator.channel_accumulator_envelope_max_decimals?.length === channels
    && accumulator.channel_post_bias_min_decimals?.length === channels
    && accumulator.channel_post_bias_max_decimals?.length === channels, "Accumulator channel arrays do not match output-channel count.");

  const witnesses = [];
  const realMultipliers = [];
  const quantizedMultipliers = [];
  const shifts = [];
  const representedMultipliers = [];
  const absoluteErrors = [];
  const relativeErrors = [];
  const encodingDrifts = [];
  const defaultBounds = [];
  const singleBounds = [];
  const preShiftSafety = [];
  const singleMultipliers = [];
  const singleShifts = [];
  const shiftCounts = new Map();
  let ledgerText = "";
  for (let channel = 0; channel < channels; channel += 1) {
    const weightScale = Number(weight.scale_sample[weight.scale_sample.length === 1 ? 0 : channel]);
    const realMultiplier = inputScale * weightScale / outputScale;
    assert(validScale(realMultiplier), `Channel ${channel} effective multiplier is invalid.`);
    const defaultEncoding = quantizeMultiplier(realMultiplier, false);
    const singleEncoding = quantizeMultiplier(realMultiplier, true);
    const envelopeMinimum = parseDecimal(accumulator.channel_accumulator_envelope_min_decimals[channel], `Channel ${channel} accumulator minimum is invalid.`);
    const envelopeMaximum = parseDecimal(accumulator.channel_accumulator_envelope_max_decimals[channel], `Channel ${channel} accumulator maximum is invalid.`);
    const postBiasMinimum = parseDecimal(accumulator.channel_post_bias_min_decimals[channel], `Channel ${channel} post-bias accumulator minimum is invalid.`);
    const postBiasMaximum = parseDecimal(accumulator.channel_post_bias_max_decimals[channel], `Channel ${channel} post-bias accumulator maximum is invalid.`);
    const maximumAbsolute = maxBigInt(absBigInt(postBiasMinimum), absBigInt(postBiasMaximum));
    const absoluteError = Math.abs(realMultiplier - defaultEncoding.represented);
    const relativeError = absoluteError / realMultiplier;
    const encodingDrift = Number(maximumAbsolute) * absoluteError;
    const singleEncodingDrift = Number(maximumAbsolute) * Math.abs(realMultiplier - singleEncoding.represented);
    const accumulatorFitsInt32 = envelopeMinimum >= -2147483648n && envelopeMaximum <= 2147483647n;
    const defaultPreShiftSafe = accumulatorFitsInt32 && preShiftFitsInt32(postBiasMinimum, postBiasMaximum, defaultEncoding.shift);
    const defaultBound = defaultPreShiftSafe ? encodingDrift + defaultRoundingOnlyBound(defaultEncoding.shift) : null;
    const singleBound = accumulatorFitsInt32 ? singleEncodingDrift + 0.5 : null;
    const singleDiverges = defaultEncoding.multiplier !== singleEncoding.multiplier || defaultEncoding.shift !== singleEncoding.shift;
    const witness = {
      channel_index: channel,
      post_bias_accumulator_min_decimal: postBiasMinimum.toString(),
      post_bias_accumulator_max_decimal: postBiasMaximum.toString(),
      maximum_absolute_post_bias_accumulator_decimal: maximumAbsolute.toString(),
      input_scale: inputScale,
      weight_scale: weightScale,
      output_scale: outputScale,
      real_multiplier: realMultiplier,
      quantized_multiplier: defaultEncoding.multiplier,
      shift: defaultEncoding.shift,
      represented_multiplier: defaultEncoding.represented,
      absolute_multiplier_error: absoluteError,
      relative_multiplier_error: relativeError,
      multiplier_error_ppm: relativeError * 1_000_000,
      encoding_drift_bound_codes: encodingDrift,
      encoding_drift_ceil_codes_decimal: Math.ceil(encodingDrift).toFixed(0),
      default_double_rounding_bound_codes: defaultBound,
      single_rounding_bound_codes: singleBound,
      default_pre_shift_int32_safe: defaultPreShiftSafe,
      single_rounding_quantized_multiplier: singleEncoding.multiplier,
      single_rounding_shift: singleEncoding.shift,
      single_rounding_represented_multiplier: singleEncoding.represented,
      single_rounding_encoding_diverges: singleDiverges,
    };
    ledgerText += ledgerRow(Number(op.index), witness);
    shiftCounts.set(defaultEncoding.shift, (shiftCounts.get(defaultEncoding.shift) || 0) + 1);
    witnesses.push(witness);
    realMultipliers.push(realMultiplier);
    quantizedMultipliers.push(defaultEncoding.multiplier);
    shifts.push(defaultEncoding.shift);
    representedMultipliers.push(defaultEncoding.represented);
    absoluteErrors.push(absoluteError);
    relativeErrors.push(relativeError);
    encodingDrifts.push(encodingDrift);
    defaultBounds.push(defaultBound);
    singleBounds.push(singleBound);
    preShiftSafety.push(defaultPreShiftSafe);
    singleMultipliers.push(singleEncoding.multiplier);
    singleShifts.push(singleEncoding.shift);
  }
  witnesses.sort((left, right) => right.encoding_drift_bound_codes - left.encoding_drift_bound_codes
    || right.relative_multiplier_error - left.relative_multiplier_error || left.channel_index - right.channel_index);
  return {
    op_index: Number(op.index),
    op_name: String(op.name),
    assessment_status: "assessed",
    not_assessed_reason: "",
    input_tensor_index: Number(input.index),
    input_tensor_name: String(input.name || ""),
    weight_tensor_index: Number(weight.index),
    weight_tensor_name: String(weight.name || ""),
    output_tensor_index: Number(output.index),
    output_tensor_name: String(output.name || ""),
    input_dtype: String(input.dtype),
    weight_dtype: String(weight.dtype),
    output_dtype: String(output.dtype),
    input_scale: inputScale,
    output_scale: outputScale,
    output_zero_point: Number(output.zero_point_sample[0]),
    output_code_range: outputCodeRange,
    weight_scale_mode: weight.scale_sample.length === 1 ? "per_tensor" : "per_output_channel",
    assessed_channel_count: channels,
    fixed_point_bound_channel_count: defaultBounds.filter((value) => value != null).length,
    default_pre_shift_overflow_channel_count: preShiftSafety.filter((safe) => !safe).length,
    single_rounding_encoding_divergence_channel_count: witnesses.filter((witness) => witness.single_rounding_encoding_diverges).length,
    half_code_encoding_drift_channel_count: encodingDrifts.filter((value) => value >= 0.5).length,
    one_code_encoding_drift_channel_count: encodingDrifts.filter((value) => value >= 1).length,
    minimum_shift: Math.min(...shifts),
    maximum_shift: Math.max(...shifts),
    shift_histogram: [...shiftCounts.entries()].sort((left, right) => left[0] - right[0]).map(([shift, channel_count]) => ({ shift, channel_count })),
    maximum_relative_multiplier_error: Math.max(...relativeErrors),
    maximum_multiplier_error_ppm: Math.max(...relativeErrors) * 1_000_000,
    maximum_encoding_drift_bound_codes: Math.max(...encodingDrifts),
    maximum_default_double_rounding_bound_codes: optionalMax(defaultBounds),
    maximum_single_rounding_bound_codes: optionalMax(singleBounds),
    channel_real_multipliers: realMultipliers,
    channel_quantized_multipliers: quantizedMultipliers,
    channel_shifts: shifts,
    channel_represented_multipliers: representedMultipliers,
    channel_absolute_multiplier_errors: absoluteErrors,
    channel_relative_multiplier_errors: relativeErrors,
    channel_encoding_drift_bound_codes: encodingDrifts,
    channel_default_double_rounding_bound_codes: defaultBounds,
    channel_single_rounding_bound_codes: singleBounds,
    channel_default_pre_shift_int32_safe: preShiftSafety,
    channel_single_rounding_quantized_multipliers: singleMultipliers,
    channel_single_rounding_shifts: singleShifts,
    worst_channel: witnesses[0] || null,
    ledger_text: ledgerText,
  };
}

function notAssessedRow(op, tensors, reason) {
  const input = tensors.get(Number(op.inputs?.[0]));
  const weight = tensors.get(Number(op.inputs?.[1]));
  const output = tensors.get(Number(op.outputs?.[0]));
  return {
    op_index: Number(op.index),
    op_name: String(op.name),
    assessment_status: "not_assessed",
    not_assessed_reason: reason,
    input_tensor_index: op.inputs?.[0] ?? null,
    input_tensor_name: String(input?.name || ""),
    weight_tensor_index: op.inputs?.[1] ?? null,
    weight_tensor_name: String(weight?.name || ""),
    output_tensor_index: op.outputs?.[0] ?? null,
    output_tensor_name: String(output?.name || ""),
  };
}

function compareAnalysis(actual, expected) {
  for (const key of [
    "status", "candidate_op_count", "assessed_op_count", "unassessed_op_count", "assessed_channel_count",
    "fixed_point_bound_channel_count", "per_tensor_weight_op_count", "per_axis_weight_op_count",
    "default_pre_shift_overflow_channel_count", "single_rounding_encoding_divergence_channel_count",
    "half_code_encoding_drift_channel_count", "one_code_encoding_drift_channel_count", "minimum_shift", "maximum_shift",
  ]) assert(actual[key] === expected[key], `Requantization ${key} mismatch.`);
  for (const key of [
    "maximum_relative_multiplier_error", "maximum_multiplier_error_ppm", "maximum_encoding_drift_bound_codes",
    "maximum_default_double_rounding_bound_codes", "maximum_single_rounding_bound_codes",
  ]) assertSameNumber(actual[key], expected[key], `Requantization ${key} mismatch.`);
  assert(JSON.stringify(actual.shift_histogram) === JSON.stringify(expected.shift_histogram), "Requantization shift histogram mismatch.");
  assert(JSON.stringify(actual.fidelity_ranking_op_indices) === JSON.stringify(expected.fidelity_ranking_op_indices), "Requantization ranking mismatch.");
  assert(actual.ops.length === expected.ops.length, "Requantization op row count mismatch.");
  for (let index = 0; index < actual.ops.length; index += 1) compareRow(actual.ops[index], expected.ops[index]);
}

function compareRow(actual, expected) {
  for (const key of [
    "op_index", "op_name", "assessment_status", "not_assessed_reason", "input_tensor_index", "input_tensor_name",
    "weight_tensor_index", "weight_tensor_name", "output_tensor_index", "output_tensor_name",
  ]) assert((actual[key] ?? null) === (expected[key] ?? null), `Requantization ${key} mismatch at op #${expected.op_index}.`);
  if (expected.assessment_status !== "assessed") return;
  for (const key of [
    "input_dtype", "weight_dtype", "output_dtype", "output_zero_point", "weight_scale_mode", "assessed_channel_count",
    "fixed_point_bound_channel_count", "default_pre_shift_overflow_channel_count", "single_rounding_encoding_divergence_channel_count",
    "half_code_encoding_drift_channel_count", "one_code_encoding_drift_channel_count", "minimum_shift", "maximum_shift",
  ]) assert(actual[key] === expected[key], `Requantization ${key} mismatch at op #${expected.op_index}.`);
  for (const key of [
    "input_scale", "output_scale", "maximum_relative_multiplier_error", "maximum_multiplier_error_ppm",
    "maximum_encoding_drift_bound_codes", "maximum_default_double_rounding_bound_codes", "maximum_single_rounding_bound_codes",
  ]) assertSameNumber(actual[key], expected[key], `Requantization ${key} mismatch at op #${expected.op_index}.`);
  assert(JSON.stringify(actual.output_code_range) === JSON.stringify(expected.output_code_range), `Requantization output range mismatch at op #${expected.op_index}.`);
  assert(JSON.stringify(actual.shift_histogram) === JSON.stringify(expected.shift_histogram), `Requantization shift histogram mismatch at op #${expected.op_index}.`);
  for (const key of [
    "channel_quantized_multipliers", "channel_shifts", "channel_default_pre_shift_int32_safe",
    "channel_single_rounding_quantized_multipliers", "channel_single_rounding_shifts",
  ]) assert(JSON.stringify(actual[key]) === JSON.stringify(expected[key]), `Requantization ${key} mismatch at op #${expected.op_index}.`);
  for (const key of [
    "channel_real_multipliers", "channel_represented_multipliers", "channel_absolute_multiplier_errors",
    "channel_relative_multiplier_errors", "channel_encoding_drift_bound_codes", "channel_default_double_rounding_bound_codes",
    "channel_single_rounding_bound_codes",
  ]) compareNumberArray(actual[key], expected[key], `Requantization ${key} mismatch at op #${expected.op_index}.`);
  compareWitness(actual.worst_channel, expected.worst_channel, expected.op_index);
  assert(/^[a-f0-9]{64}$/.test(actual.channel_ledger_sha256 || ""), `Requantization ledger digest is invalid at op #${expected.op_index}.`);
}

function compareWitness(actual, expected, opIndex) {
  assert(actual && expected, `Requantization witness missing at op #${opIndex}.`);
  for (const key of [
    "channel_index", "post_bias_accumulator_min_decimal", "post_bias_accumulator_max_decimal",
    "maximum_absolute_post_bias_accumulator_decimal", "quantized_multiplier", "shift", "encoding_drift_ceil_codes_decimal",
    "default_pre_shift_int32_safe", "single_rounding_quantized_multiplier", "single_rounding_shift",
    "single_rounding_encoding_diverges",
  ]) assert(actual[key] === expected[key], `Requantization witness ${key} mismatch at op #${opIndex}.`);
  for (const key of [
    "input_scale", "weight_scale", "output_scale", "real_multiplier", "represented_multiplier",
    "absolute_multiplier_error", "relative_multiplier_error", "multiplier_error_ppm", "encoding_drift_bound_codes",
    "default_double_rounding_bound_codes", "single_rounding_bound_codes", "single_rounding_represented_multiplier",
  ]) assertSameNumber(actual[key], expected[key], `Requantization witness ${key} mismatch at op #${opIndex}.`);
}

function ledgerRow(opIndex, witness) {
  return `op=${opIndex};channel=${witness.channel_index};real=${f64Bits(witness.real_multiplier)};q=${witness.quantized_multiplier};shift=${witness.shift};represented=${f64Bits(witness.represented_multiplier)};abs_error=${f64Bits(witness.absolute_multiplier_error)};relative_error=${f64Bits(witness.relative_multiplier_error)};encoding_drift=${f64Bits(witness.encoding_drift_bound_codes)};default_bound=${optionalF64Bits(witness.default_double_rounding_bound_codes)};pre_shift_safe=${Number(witness.default_pre_shift_int32_safe)};single_q=${witness.single_rounding_quantized_multiplier};single_shift=${witness.single_rounding_shift};single_represented=${f64Bits(witness.single_rounding_represented_multiplier)};single_bound=${optionalF64Bits(witness.single_rounding_bound_codes)};single_diverges=${Number(witness.single_rounding_encoding_diverges)}\n`;
}

function f64Bits(value) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  return view.getBigUint64(0, false).toString(16).padStart(16, "0");
}

function optionalF64Bits(value) {
  return value == null ? "na" : f64Bits(value);
}

function preShiftFitsInt32(minimum, maximum, shift) {
  if (shift <= 0) return true;
  const factor = 1n << BigInt(shift);
  return minimum * factor >= -2147483648n && maximum * factor <= 2147483647n;
}

function defaultRoundingOnlyBound(shift) {
  return shift < 0 ? 0.5 + (2 ** (shift - 1)) : 0.5;
}

export function drawRequantizationFidelityCanvas(canvas, row, mode = "encoding", logicalWidth = null, logicalHeight = null) {
  if (!canvas || !row || row.assessment_status !== "assessed") return null;
  const width = logicalWidth || Math.max(260, Math.floor(canvas.parentElement?.clientWidth || canvas.clientWidth || 720));
  const height = logicalHeight || Math.max(160, Math.round(width * 0.56));
  const dpr = Math.max(1, Math.min(2, globalThis.devicePixelRatio || 1));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#10171a";
  context.fillRect(0, 0, width, height);
  const count = row.assessed_channel_count;
  const columns = Math.min(64, Math.max(4, Math.ceil(Math.sqrt(count * 1.75))));
  const rows = Math.ceil(count / columns);
  const gap = count > 512 ? 1 : 2;
  const pad = 18;
  const cellWidth = (width - pad * 2 - gap * (columns - 1)) / columns;
  const cellHeight = (height - pad * 2 - gap * (rows - 1)) / rows;
  for (let channel = 0; channel < count; channel += 1) {
    const column = channel % columns;
    const rowIndex = Math.floor(channel / columns);
    context.fillStyle = requantColor(row, channel, mode);
    context.fillRect(
      pad + column * (cellWidth + gap),
      pad + rowIndex * (cellHeight + gap),
      Math.max(1, cellWidth),
      Math.max(1, cellHeight),
    );
  }
  context.strokeStyle = "rgba(255,255,255,.15)";
  context.strokeRect(pad - 1, pad - 1, width - pad * 2 + 2, height - pad * 2 + 2);
  canvas._requantGeometry = { width, height, pad, gap, columns, rows, cellWidth, cellHeight, count };
  canvas.dataset.pixelSignature = `${mode}:${row.op_index}:${row.maximum_encoding_drift_bound_codes}`;
  return canvas._requantGeometry;
}

function requantColor(row, channel, mode) {
  if (!row.channel_default_pre_shift_int32_safe[channel]) return "#d63c52";
  if (mode === "shift") {
    const shift = row.channel_shifts[channel];
    const span = Math.max(1, row.maximum_shift - row.minimum_shift);
    return mixColor([42, 117, 166], [225, 177, 54], (shift - row.minimum_shift) / span);
  }
  if (mode === "rounding") {
    const value = row.channel_default_double_rounding_bound_codes[channel] ?? 1;
    return mixColor([36, 140, 120], [218, 92, 60], Math.max(0, Math.min(1, value)));
  }
  const value = row.channel_encoding_drift_bound_codes[channel];
  const normalized = Math.max(0, Math.min(1, (Math.log10(value + 1e-15) + 15) / 15));
  return mixColor([39, 126, 153], [226, 166, 52], normalized);
}

function installCanvasInteraction(canvas, tooltip, row) {
  canvas.addEventListener("pointermove", (event) => {
    const geometry = canvas._requantGeometry;
    if (!geometry) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * geometry.width / rect.width - geometry.pad;
    const y = (event.clientY - rect.top) * geometry.height / rect.height - geometry.pad;
    const column = Math.floor(x / (geometry.cellWidth + geometry.gap));
    const rowIndex = Math.floor(y / (geometry.cellHeight + geometry.gap));
    const channel = rowIndex * geometry.columns + column;
    const inside = column >= 0 && column < geometry.columns && rowIndex >= 0 && rowIndex < geometry.rows
      && x % (geometry.cellWidth + geometry.gap) <= geometry.cellWidth
      && y % (geometry.cellHeight + geometry.gap) <= geometry.cellHeight && channel < geometry.count;
    if (!inside) { tooltip.hidden = true; return; }
    tooltip.textContent = `ch ${channel} / q ${row.channel_quantized_multipliers[channel]} / shift ${signed(row.channel_shifts[channel])} / encoding ${formatScientific(row.channel_encoding_drift_bound_codes[channel])} codes / default <= ${formatBound(row.channel_default_double_rounding_bound_codes[channel])}`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(rect.width - 24, Math.max(12, event.clientX - rect.left))}px`;
    tooltip.style.top = `${Math.max(12, event.clientY - rect.top)}px`;
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
}

function renderSummary(container, fidelity) {
  if (!container) return;
  container.replaceChildren(
    summaryMetric(formatNumber(fidelity.assessed_channel_count), "verified channels"),
    summaryMetric(formatScientific(fidelity.maximum_encoding_drift_bound_codes), "max encoding drift", fidelity.half_code_encoding_drift_channel_count ? "risk" : "ok"),
    summaryMetric(`${formatBound(fidelity.maximum_default_double_rounding_bound_codes)} / ${formatBound(fidelity.maximum_single_rounding_bound_codes)}`, "default / single bound"),
    summaryMetric(`${signed(fidelity.minimum_shift)}..${signed(fidelity.maximum_shift)}`, "Q0.31 shifts", fidelity.default_pre_shift_overflow_channel_count ? "risk" : "ok"),
  );
}

function summaryMetric(value, label, tone = "") {
  const node = document.createElement("div");
  node.className = "requant-summary-metric";
  if (tone) node.dataset.tone = tone;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  node.append(strong, span);
  return node;
}

function definitionTable(rows) {
  const section = document.createElement("section");
  section.className = "requant-definition";
  const heading = document.createElement("h4");
  heading.textContent = "Artifact and source contract";
  const list = document.createElement("dl");
  for (const [term, detail] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    if (detail instanceof Node) dd.append(detail); else dd.textContent = String(detail);
    list.append(dt, dd);
  }
  section.append(heading, list);
  return section;
}

function witnessTable(witness) {
  const section = document.createElement("section");
  section.className = "requant-definition";
  const heading = document.createElement("h4");
  heading.textContent = "Worst encoding witness";
  const list = document.createElement("dl");
  const rows = witness ? [
    ["Channel / post-bias |acc|", `${witness.channel_index} / ${formatInteger(witness.maximum_absolute_post_bias_accumulator_decimal)}`],
    ["Real / represented", `${formatScale(witness.real_multiplier)} / ${formatScale(witness.represented_multiplier)}`],
    ["Q0.31 encoding", `${witness.quantized_multiplier} x 2^${signed(witness.shift - 31)}`],
    ["Relative error", `${formatScientific(witness.relative_multiplier_error)} (${formatScientific(witness.multiplier_error_ppm)} ppm)`],
    ["Encoding drift", `${formatScientific(witness.encoding_drift_bound_codes)} output codes`],
    ["Execution bounds", `default <= ${formatBound(witness.default_double_rounding_bound_codes)} / single <= ${formatBound(witness.single_rounding_bound_codes)} codes`],
  ] : [["Status", "not assessed"]];
  for (const [term, detail] of rows) {
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = detail;
    list.append(dt, dd);
  }
  section.append(heading, list);
  return section;
}

function shiftHistogram(row) {
  const section = document.createElement("section");
  section.className = "requant-histogram-section";
  const heading = document.createElement("h4");
  heading.textContent = "Q0.31 exponent inventory";
  const chart = document.createElement("div");
  chart.className = "requant-shift-histogram";
  const maximum = Math.max(1, ...row.shift_histogram.map((bin) => bin.channel_count));
  for (const bin of row.shift_histogram) {
    const item = document.createElement("div");
    item.className = "requant-shift-bar";
    item.style.setProperty("--bar", `${Math.max(4, bin.channel_count / maximum * 100)}%`);
    const label = document.createElement("span"); label.textContent = signed(bin.shift);
    const value = document.createElement("b"); value.textContent = formatNumber(bin.channel_count);
    item.append(label, value);
    chart.append(item);
  }
  section.append(heading, chart);
  return section;
}

function rankingTable(fidelity, selected, jumpToGraphOp, selectOp) {
  const section = document.createElement("section");
  section.className = "requant-ranking-section";
  const heading = document.createElement("h4");
  heading.textContent = "Operator fidelity portfolio";
  const wrap = document.createElement("div");
  wrap.className = "requant-ranking-wrap";
  const table = document.createElement("table");
  table.className = "requant-ranking-table";
  table.innerHTML = "<thead><tr><th>Rank</th><th>Op</th><th>Channels</th><th>Shift</th><th>Error ppm</th><th>Encoding drift</th><th>Default bound</th><th>Ledger</th><th></th></tr></thead>";
  const tbody = document.createElement("tbody");
  const byIndex = new Map(fidelity.ops.filter((row) => row.assessment_status === "assessed").map((row) => [row.op_index, row]));
  fidelity.fidelity_ranking_op_indices.forEach((opIndex, rank) => {
    const row = byIndex.get(opIndex);
    if (!row) return;
    const tr = document.createElement("tr");
    tr.classList.toggle("selected", row.op_index === selected);
    tr.tabIndex = 0;
    tr.addEventListener("click", () => selectOp(row.op_index));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectOp(row.op_index); }
    });
    const values = [
      `#${rank + 1}`, `#${row.op_index} ${row.op_name}`, formatNumber(row.assessed_channel_count),
      `${signed(row.minimum_shift)}..${signed(row.maximum_shift)}`, formatScientific(row.maximum_multiplier_error_ppm),
      formatScientific(row.maximum_encoding_drift_bound_codes), formatBound(row.maximum_default_double_rounding_bound_codes),
      row.channel_ledger_sha256.slice(0, 12),
    ];
    for (const value of values) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    const action = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "requant-graph-button";
    button.textContent = "Graph";
    button.addEventListener("click", (event) => { event.stopPropagation(); jumpToGraphOp?.(row.op_index); });
    action.append(button); tr.append(action); tbody.append(tr);
  });
  table.append(tbody); wrap.append(table); section.append(heading, wrap); return section;
}

function unassessedTable(rows) {
  const unassessed = (rows || []).filter((row) => row.assessment_status !== "assessed");
  if (!unassessed.length) return document.createDocumentFragment();
  const node = document.createElement("div");
  node.className = "requant-message";
  node.textContent = unassessed.map((row) => `#${row.op_index} ${row.op_name}: ${row.not_assessed_reason}`).join(" / ");
  return node;
}

function sourceLinks(fidelity) {
  const span = document.createElement("span");
  fidelity.source_references.forEach((source, index) => {
    if (index) span.append(" / ");
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = source.role.replaceAll("_", " ");
    link.title = `${source.file} / SHA-256 ${source.sha256}`;
    span.append(link);
  });
  return span;
}

function validateSources(sources) {
  assert(Array.isArray(sources) && sources.length === 3, "Requantization source inventory mismatch.");
  for (const source of sources) {
    assert(SOURCE_SHA256[source.file] === source.sha256, `Requantization source SHA-256 mismatch for ${source.file}.`);
    assert(source.url === `https://github.com/tensorflow/tensorflow/blob/${SOURCE_COMMIT}/${source.file}`, `Requantization source URL mismatch for ${source.file}.`);
  }
}

function requiredTensor(tensors, index, role) {
  const tensor = tensors.get(Number(index));
  assert(tensor, `${role} tensor is unavailable.`);
  return tensor;
}

function quantizedCodeRange(dtype) {
  if (dtype === "INT8") return [-128, 127];
  if (dtype === "UINT8") return [0, 255];
  return null;
}

function parseDecimal(value, message) {
  try { return BigInt(value); } catch { throw new Error(message); }
}

function compareNumberArray(actual, expected, message) {
  assert(Array.isArray(actual) && actual.length === expected.length, message);
  for (let index = 0; index < expected.length; index += 1) assertSameNumber(actual[index], expected[index], `${message} channel ${index}.`);
}

function assertSameNumber(actual, expected, message) {
  if (actual == null || expected == null) { assert(actual == null && expected == null, message); return; }
  assert(Object.is(Number(actual), Number(expected)), message);
}

function optionalMin(values) {
  const assessed = values.filter((value) => value != null);
  return assessed.length ? Math.min(...assessed) : null;
}

function optionalMax(values) {
  const assessed = values.filter((value) => value != null);
  return assessed.length ? Math.max(...assessed) : null;
}

function sum(rows, field) {
  return rows.reduce((total, row) => total + Number(row[field] || 0), 0);
}

function validScale(value) {
  return Number.isFinite(value) && value > 0;
}

function absBigInt(value) { return value < 0n ? -value : value; }
function maxBigInt(left, right) { return left > right ? left : right; }

function mixColor(left, right, amount) {
  const values = left.map((value, index) => Math.round(value + (right[index] - value) * amount));
  return `rgb(${values.join(",")})`;
}

function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatScale(value) { return Number(value).toPrecision(9); }
function formatScientific(value) { return value == null ? "N/A" : Number(value).toExponential(3); }
function formatBound(value) { return value == null ? "N/A" : Number(value).toFixed(6); }
function signed(value) { return value == null ? "N/A" : `${Number(value) >= 0 ? "+" : ""}${value}`; }
function formatInteger(value) { try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value); } }

function messageNode(text, tone = "muted") {
  const node = document.createElement("div");
  node.className = "requant-message";
  node.dataset.tone = tone;
  node.textContent = text;
  return node;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
