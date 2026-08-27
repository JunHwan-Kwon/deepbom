import { sha256Hex } from "./hash.js";
import {
  reconstructKernelChannel,
  reconstructKernelWitnessAnalysis,
  validateKernelWitnessAnalysis,
} from "./kernel-witness.js";

export const CHANNEL_VITALITY_SCHEMA = "deepbom.channel_vitality.v1";
const METHOD_VERSION = "2026-07-17.1";
const SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LEDGER_PREFIX = new TextEncoder().encode("deepbom.channel_vitality.v1\0");
const MISSING_I64 = -(1n << 63n);
const TOP_LIMIT = 16;
const REASON_LABELS = [
  "nonconstant",
  "constant_accumulator",
  "lower_code_clamp",
  "upper_code_clamp",
  "fixed_point_projection_collapse",
];
const SIGN_LABELS = {
  "-1": "post_bias_negative_locked",
  0: "post_bias_zero_containing",
  1: "post_bias_positive_locked",
};
const HISTOGRAM_BINS = [
  ["1", 1, 1],
  ["2-3", 2, 3],
  ["4-15", 4, 15],
  ["16-63", 16, 63],
  ["64-127", 64, 127],
  ["128-255", 128, 255],
  ["256", 256, 256],
];

export function createChannelVitalityController({
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
  let buildMode = "default";
  let field = "span";
  let worker = null;
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "channel_vitality.json");
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
    evidence = analysis?.channel_vitality || null;
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
      validateChannelVitalityShape(evidence);
      const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = evidence.vitality_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
        selectedChannelIndex = null;
      }
      renderSummary(summary, evidence);
      renderBody();
      if (status) {
        status.textContent = evidence.assessed_op_count ? "source arithmetic / verification pending" : humanize(evidence.status);
        status.dataset.tone = evidence.nonconstant_accumulator_dual_mode_constant_channel_count ? "risk" : evidence.mode_dependent_constant_output_channel_count ? "watch" : "ok";
      }
      if (evidence.assessed_op_count && typeof Worker === "function") {
        worker = new Worker(new URL("./channel-vitality-worker.js", import.meta.url), { type: "module" });
        worker.onmessage = (event) => {
          if (token !== renderToken || !status) return;
          status.textContent = event.data?.ok ? "independently verified" : `integrity error: ${event.data?.error || "verification failed"}`;
          status.dataset.tone = event.data?.ok
            ? evidence.nonconstant_accumulator_dual_mode_constant_channel_count ? "risk" : evidence.mode_dependent_constant_output_channel_count ? "watch" : "ok"
            : "risk";
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
        worker.postMessage({ analysis, modelBytes });
      }
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(messageNode(`Channel vitality evidence rejected: ${error.message}`, "risk"));
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
      body.replaceChildren(messageNode("No constant 8-bit convolution-family channel was assessable."), unassessedTable(evidence.ops));
      return;
    }
    const row = assessed.find((candidate) => candidate.op_index === selectedOpIndex) || assessed[0];
    selectedOpIndex = row.op_index;
    const suggestedChannel = row.top_channels?.[0]?.channel_index ?? 0;
    if (!Number.isInteger(selectedChannelIndex) || selectedChannelIndex < 0 || selectedChannelIndex >= row.assessed_channel_count) {
      selectedChannelIndex = suggestedChannel;
    }

    const toolbar = document.createElement("div");
    toolbar.className = "channel-vitality-toolbar";
    const opSelect = document.createElement("select");
    opSelect.className = "channel-vitality-op-select";
    opSelect.setAttribute("aria-label", "Channel vitality operator");
    for (const candidate of assessed) {
      const option = new Option(
        `#${candidate.op_index} ${candidate.op_name} / ${formatNumber(candidate.assessed_channel_count)} ch / ${formatNumber(candidate.dual_mode_constant_output_channel_count)} constant`,
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
    channelLabel.className = "channel-vitality-channel-control";
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
    toolbar.append(
      opSelect,
      channelLabel,
      segmentedControl("TFLite fixed-point build path", [["default", "Default"], ["single", "Single rounding"]], buildMode, (value) => {
        buildMode = value;
        renderBody();
      }),
      segmentedControl("Channel vitality field", [["span", "Code span"], ["reason", "Cause"], ["sign", "Sign"]], field, (value) => {
        field = value;
        renderBody();
      }),
    );

    const selected = selectedChannelEvidence(row, selectedChannelIndex);
    const reconstructed = reconstructKernelChannel(analysis, modelBytes, row.op_index, selectedChannelIndex);
    compareSelectedWitness(row, selected, reconstructed);
    const main = document.createElement("div");
    main.className = "channel-vitality-main";
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "channel-vitality-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "channel-vitality-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${field} atlas for operator ${row.op_index} under ${buildMode} fixed-point mode`);
    const tooltip = document.createElement("div");
    tooltip.className = "channel-vitality-tooltip";
    tooltip.hidden = true;
    canvasWrap.append(canvas, tooltip);
    installCanvasInteraction(canvas, tooltip, (channel) => {
      selectedChannelIndex = channel;
      renderBody();
    });

    const details = document.createElement("div");
    details.className = "channel-vitality-details";
    const selectedDownload = document.createElement("button");
    selectedDownload.type = "button";
    selectedDownload.className = "secondary-action";
    selectedDownload.textContent = "Channel JSON";
    selectedDownload.title = "Download the selected channel vitality proof";
    selectedDownload.addEventListener("click", () => onDownload?.(
      selectedChannelExport(analysis, evidence, row, selected, reconstructed),
      `channel_vitality_op_${row.op_index}_channel_${selectedChannelIndex}.json`,
    ));
    const graphButton = document.createElement("button");
    graphButton.type = "button";
    graphButton.className = "secondary-action";
    graphButton.textContent = "Graph op";
    graphButton.addEventListener("click", () => jumpToGraphOp?.(row.op_index));
    const actions = document.createElement("div");
    actions.className = "channel-vitality-actions";
    actions.append(selectedDownload, graphButton);
    details.append(
      definitionTable([
        ["Coordinate", `#${row.op_index} ${row.op_name}, output channel ${selectedChannelIndex}`],
        ["Post-bias domain", `${formatInteger(reconstructed.minimum.post_bias_accumulator_decimal)} .. ${formatInteger(reconstructed.maximum.post_bias_accumulator_decimal)}`],
        ["Sign class", humanize(selected.sign_label)],
        ["Default", pathText(selected.default, reconstructed.minimum.default_preclamp_code, reconstructed.maximum.default_preclamp_code)],
        ["Single rounding", pathText(selected.single, reconstructed.minimum.single_preclamp_code, reconstructed.maximum.single_preclamp_code)],
        ["Mode status", selected.mode_dependent_constant ? "constant classification changes by build flag" : selected.dual_mode_constant ? "constant under both pinned build paths" : "nonconstant under both pinned build paths"],
        ["Vitality ledger", row.vitality_ledger_sha256],
      ]),
      actions,
      proofPath(selected, reconstructed),
    );
    main.append(canvasWrap, details);

    const histogram = histogramTable(evidence.span_histogram);
    const ranking = rankingTable(evidence, row.op_index, (opIndex) => {
      selectedOpIndex = opIndex;
      selectedChannelIndex = null;
      renderBody();
    }, jumpToGraphOp);
    const proof = document.createElement("p");
    proof.className = "channel-vitality-proof";
    proof.textContent = evidence.constant_proof;
    const boundary = document.createElement("p");
    boundary.className = "channel-vitality-boundary";
    boundary.textContent = evidence.interpretation_boundary;
    body.replaceChildren(toolbar, main, histogram, ranking, proof, boundary, unassessedTable(evidence.ops));
    requestAnimationFrame(() => drawChannelVitalityCanvas(canvas, row, buildMode, field, selectedChannelIndex));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawChannelVitalityCanvas(canvas, row, buildMode, field, selectedChannelIndex));
      resizeObserver.observe(canvasWrap);
    }
  }

  return { render };
}

export function validateChannelVitalityShape(evidence) {
  assert(evidence?.schema === CHANNEL_VITALITY_SCHEMA, "Channel vitality schema mismatch.");
  assert(evidence.method_version === METHOD_VERSION, "Channel vitality method mismatch.");
  assert(evidence.source_commit === SOURCE_COMMIT, "Channel vitality source commit mismatch.");
  assert(evidence.source_evidence_schema === "deepbom.kernel_extremum_witness.v1", "Channel vitality source schema mismatch.");
  assert(Array.isArray(evidence.ops) && evidence.ops.length === Number(evidence.candidate_op_count), "Channel vitality op rows are incomplete.");
  assert(Array.isArray(evidence.vitality_ranking_op_indices), "Channel vitality ranking is unavailable.");
  for (const row of evidence.ops) {
    assert(Number.isInteger(row.op_index) && typeof row.op_name === "string", "Channel vitality op identity is invalid.");
    if (row.assessment_status !== "assessed") continue;
    const count = Number(row.assessed_channel_count);
    for (const key of [
      "default_minimum_output_codes",
      "default_maximum_output_codes",
      "single_minimum_output_codes",
      "single_maximum_output_codes",
      "post_bias_sign_codes",
      "default_constant_reason_codes",
      "single_constant_reason_codes",
    ]) assert(Array.isArray(row[key]) && row[key].length === count, `Channel vitality ${key} cardinality mismatch at op #${row.op_index}.`);
    assert(row.post_bias_sign_codes.every((code) => code === -1 || code === 0 || code === 1), `Channel vitality sign code is invalid at op #${row.op_index}.`);
    assert(row.default_constant_reason_codes.every(validReasonCode) && row.single_constant_reason_codes.every(validReasonCode), `Channel vitality reason code is invalid at op #${row.op_index}.`);
    assert(/^[a-f0-9]{64}$/.test(row.source_witness_ledger_sha256 || ""), `Channel vitality source digest is invalid at op #${row.op_index}.`);
    assert(/^[a-f0-9]{64}$/.test(row.vitality_ledger_sha256 || ""), `Channel vitality digest is invalid at op #${row.op_index}.`);
    assert(Array.isArray(row.top_channels) && row.top_channels.length <= TOP_LIMIT, `Channel vitality top-channel set is invalid at op #${row.op_index}.`);
  }
  return evidence;
}

export function validateChannelVitalityAnalysis(analysis, modelBytes) {
  validateKernelWitnessAnalysis(analysis, modelBytes);
  const actual = validateChannelVitalityShape(analysis?.channel_vitality);
  const expected = reconstructChannelVitalityAnalysis(analysis, modelBytes);
  compareSummary(actual, expected);
  assert(actual.ops.length === expected.ops.length, "Channel vitality op count mismatch.");
  actual.ops.forEach((row, index) => compareOpRow(row, expected.ops[index]));
  return expected;
}

export async function validateChannelVitalityDigests(analysis, modelBytes) {
  const reconstructed = validateChannelVitalityAnalysis(analysis, modelBytes);
  for (let index = 0; index < reconstructed.ops.length; index += 1) {
    const row = reconstructed.ops[index];
    if (row.assessment_status !== "assessed") continue;
    const digest = await sha256Hex(row.ledger_bytes);
    assert(digest === analysis.channel_vitality.ops[index].vitality_ledger_sha256, `Channel vitality ledger SHA-256 mismatch at op #${row.op_index}.`);
  }
  return reconstructed;
}

export function reconstructChannelVitalityAnalysis(analysis, modelBytes) {
  const witness = reconstructKernelWitnessAnalysis(analysis, modelBytes);
  const sourceRows = new Map((analysis?.kernel_extremum_witness?.ops || []).map((row) => [Number(row.op_index), row]));
  const ops = witness.ops.map((row) => row.assessment_status === "assessed"
    ? reconstructOp(row, sourceRows.get(row.op_index))
    : notAssessedRow(row, sourceRows.get(row.op_index)));
  const assessed = ops.filter((row) => row.assessment_status === "assessed");
  const defaultSpans = assessed.flatMap((row) => spans(row.default_minimum_output_codes, row.default_maximum_output_codes));
  const singleSpans = assessed.flatMap((row) => spans(row.single_minimum_output_codes, row.single_maximum_output_codes));
  return {
    status: !ops.length ? "not_applicable" : assessed.length === ops.length ? "assessed" : assessed.length ? "partial" : "not_assessed",
    candidate_op_count: ops.length,
    assessed_op_count: assessed.length,
    unassessed_op_count: ops.length - assessed.length,
    assessed_channel_count: sum(assessed, "assessed_channel_count"),
    fixed_point_assessed_channel_count: sum(assessed, "fixed_point_assessed_channel_count"),
    constant_accumulator_channel_count: sum(assessed, "constant_accumulator_channel_count"),
    post_bias_negative_locked_channel_count: sum(assessed, "post_bias_negative_locked_channel_count"),
    post_bias_positive_locked_channel_count: sum(assessed, "post_bias_positive_locked_channel_count"),
    post_bias_zero_containing_channel_count: sum(assessed, "post_bias_zero_containing_channel_count"),
    default_constant_output_channel_count: sum(assessed, "default_constant_output_channel_count"),
    single_constant_output_channel_count: sum(assessed, "single_constant_output_channel_count"),
    dual_mode_constant_output_channel_count: sum(assessed, "dual_mode_constant_output_channel_count"),
    nonconstant_accumulator_dual_mode_constant_channel_count: sum(assessed, "nonconstant_accumulator_dual_mode_constant_channel_count"),
    mode_dependent_constant_output_channel_count: sum(assessed, "mode_dependent_constant_output_channel_count"),
    default_severely_constrained_channel_count: sum(assessed, "default_severely_constrained_channel_count"),
    single_severely_constrained_channel_count: sum(assessed, "single_severely_constrained_channel_count"),
    default_full_activation_span_channel_count: sum(assessed, "default_full_activation_span_channel_count"),
    single_full_activation_span_channel_count: sum(assessed, "single_full_activation_span_channel_count"),
    minimum_default_inclusive_code_span: minimum(defaultSpans),
    minimum_single_inclusive_code_span: minimum(singleSpans),
    span_histogram: buildHistogram(defaultSpans, singleSpans),
    vitality_ranking_op_indices: [...assessed].sort(compareOpRanking).map((row) => row.op_index),
    ops,
  };
}

function reconstructOp(row, sourceRow) {
  assert(sourceRow?.witness_ledger_sha256, `Source witness ledger is unavailable at op #${row.op_index}.`);
  assert(row.channels.length === Number(row.assessed_channel_count), `Source witness channel count mismatch at op #${row.op_index}.`);
  const writer = new ByteWriter(LEDGER_PREFIX.byteLength + 64 + row.channels.length * 13 * 8);
  writer.writeBytes(LEDGER_PREFIX);
  writer.writeAscii(sourceRow.witness_ledger_sha256);
  const channels = row.channels.map((channel, expectedIndex) => {
    assert(channel.channel_index === expectedIndex, `Source witness channel order mismatch at op #${row.op_index}.`);
    const result = classifyChannel(row, channel);
    for (const value of [
      row.op_index,
      channel.channel_index,
      channel.minimum.default_preclamp_code,
      channel.maximum.default_preclamp_code,
      channel.minimum.default_output_code,
      channel.maximum.default_output_code,
      channel.minimum.single_preclamp_code,
      channel.maximum.single_preclamp_code,
      channel.minimum.single_output_code,
      channel.maximum.single_output_code,
      result.sign_code,
      result.default.reason_code,
      result.single.reason_code,
    ]) writer.writeI64(value ?? MISSING_I64);
    return result;
  });
  const defaultMinimum = channels.map((channel) => channel.default.minimum_output_code);
  const defaultMaximum = channels.map((channel) => channel.default.maximum_output_code);
  const singleMinimum = channels.map((channel) => channel.single.minimum_output_code);
  const singleMaximum = channels.map((channel) => channel.single.maximum_output_code);
  const defaultSpans = channels.map((channel) => channel.default.inclusive_code_span).filter(isNumber);
  const singleSpans = channels.map((channel) => channel.single.inclusive_code_span).filter(isNumber);
  const activationSpan = optionalSpan(row.activation_code_range?.[0], row.activation_code_range?.[1]);
  const defaultConstantIndices = channels.filter((channel) => channel.default.reason_code !== 0).map((channel) => channel.channel_index);
  const singleConstantIndices = channels.filter((channel) => channel.single.reason_code !== 0).map((channel) => channel.channel_index);
  const modeDependentIndices = channels.filter((channel) => channel.mode_dependent_constant).map((channel) => channel.channel_index);
  const fixed = channels.filter((channel) => channel.default.inclusive_code_span != null && channel.single.inclusive_code_span != null);
  return {
    op_index: row.op_index,
    op_name: row.op_name,
    assessment_status: "assessed",
    not_assessed_reason: "",
    output_code_range: row.output_code_range,
    activation_code_range: row.activation_code_range,
    assessed_channel_count: channels.length,
    fixed_point_assessed_channel_count: fixed.length,
    constant_accumulator_channel_count: channels.filter((channel) => channel.constant_accumulator).length,
    post_bias_negative_locked_channel_count: channels.filter((channel) => channel.sign_code === -1).length,
    post_bias_positive_locked_channel_count: channels.filter((channel) => channel.sign_code === 1).length,
    post_bias_zero_containing_channel_count: channels.filter((channel) => channel.sign_code === 0).length,
    default_constant_output_channel_count: defaultConstantIndices.length,
    single_constant_output_channel_count: singleConstantIndices.length,
    dual_mode_constant_output_channel_count: channels.filter((channel) => channel.dual_mode_constant).length,
    nonconstant_accumulator_dual_mode_constant_channel_count: channels.filter((channel) => !channel.constant_accumulator && channel.dual_mode_constant).length,
    mode_dependent_constant_output_channel_count: modeDependentIndices.length,
    default_severely_constrained_channel_count: defaultSpans.filter((span) => span <= 15).length,
    single_severely_constrained_channel_count: singleSpans.filter((span) => span <= 15).length,
    default_full_activation_span_channel_count: defaultSpans.filter((span) => span === activationSpan).length,
    single_full_activation_span_channel_count: singleSpans.filter((span) => span === activationSpan).length,
    minimum_default_inclusive_code_span: minimum(defaultSpans),
    minimum_single_inclusive_code_span: minimum(singleSpans),
    default_minimum_output_codes: defaultMinimum,
    default_maximum_output_codes: defaultMaximum,
    single_minimum_output_codes: singleMinimum,
    single_maximum_output_codes: singleMaximum,
    post_bias_sign_codes: channels.map((channel) => channel.sign_code),
    default_constant_reason_codes: channels.map((channel) => channel.default.reason_code),
    single_constant_reason_codes: channels.map((channel) => channel.single.reason_code),
    default_constant_channel_indices: defaultConstantIndices,
    single_constant_channel_indices: singleConstantIndices,
    mode_dependent_constant_channel_indices: modeDependentIndices,
    top_channels: [...channels].sort(compareChannelRanking).slice(0, TOP_LIMIT).map(publicChannel),
    source_witness_ledger_sha256: sourceRow.witness_ledger_sha256,
    channels,
    ledger_bytes: writer.finish(),
  };
}

function classifyChannel(row, channel) {
  const postMinimum = BigInt(channel.minimum.post_bias_accumulator_decimal);
  const postMaximum = BigInt(channel.maximum.post_bias_accumulator_decimal);
  assert(postMaximum >= postMinimum, `Post-bias interval is nonmonotone at op #${row.op_index} channel ${channel.channel_index}.`);
  const signCode = postMaximum < 0n ? -1 : postMinimum > 0n ? 1 : 0;
  const constantAccumulator = postMinimum === postMaximum;
  const defaultPath = classifyPath(
    channel.minimum.default_preclamp_code,
    channel.maximum.default_preclamp_code,
    channel.minimum.default_output_code,
    channel.maximum.default_output_code,
    row.activation_code_range,
    constantAccumulator,
  );
  const singlePath = classifyPath(
    channel.minimum.single_preclamp_code,
    channel.maximum.single_preclamp_code,
    channel.minimum.single_output_code,
    channel.maximum.single_output_code,
    row.activation_code_range,
    constantAccumulator,
  );
  return {
    channel_index: channel.channel_index,
    post_bias_minimum_decimal: String(postMinimum),
    post_bias_maximum_decimal: String(postMaximum),
    accumulator_span_decimal: String(postMaximum - postMinimum),
    sign_code: signCode,
    sign_label: SIGN_LABELS[signCode],
    constant_accumulator: constantAccumulator,
    default: defaultPath,
    single: singlePath,
    dual_mode_constant: defaultPath.reason_code !== 0 && singlePath.reason_code !== 0,
    mode_dependent_constant: (defaultPath.reason_code === 0) !== (singlePath.reason_code === 0),
  };
}

function classifyPath(minimumPreclamp, maximumPreclamp, minimumOutput, maximumOutput, activationRange, constantAccumulator) {
  const span = optionalSpan(minimumOutput, maximumOutput);
  let reasonCode = 0;
  if (span === 1) {
    if (constantAccumulator) reasonCode = 1;
    else if (maximumPreclamp != null && maximumPreclamp <= activationRange[0]) reasonCode = 2;
    else if (minimumPreclamp != null && minimumPreclamp >= activationRange[1]) reasonCode = 3;
    else reasonCode = 4;
  }
  return {
    minimum_preclamp_code: minimumPreclamp,
    maximum_preclamp_code: maximumPreclamp,
    minimum_output_code: minimumOutput,
    maximum_output_code: maximumOutput,
    inclusive_code_span: span,
    reason_code: reasonCode,
    reason_label: REASON_LABELS[reasonCode],
  };
}

function publicChannel(channel) {
  return {
    channel_index: channel.channel_index,
    post_bias_minimum_decimal: channel.post_bias_minimum_decimal,
    post_bias_maximum_decimal: channel.post_bias_maximum_decimal,
    accumulator_span_decimal: channel.accumulator_span_decimal,
    post_bias_sign_class: channel.sign_label,
    default_minimum_preclamp_code: channel.default.minimum_preclamp_code,
    default_maximum_preclamp_code: channel.default.maximum_preclamp_code,
    default_minimum_output_code: channel.default.minimum_output_code,
    default_maximum_output_code: channel.default.maximum_output_code,
    default_inclusive_code_span: channel.default.inclusive_code_span,
    default_constant_reason: channel.default.reason_label,
    single_minimum_preclamp_code: channel.single.minimum_preclamp_code,
    single_maximum_preclamp_code: channel.single.maximum_preclamp_code,
    single_minimum_output_code: channel.single.minimum_output_code,
    single_maximum_output_code: channel.single.maximum_output_code,
    single_inclusive_code_span: channel.single.inclusive_code_span,
    single_constant_reason: channel.single.reason_label,
    dual_mode_constant: channel.dual_mode_constant,
    mode_dependent_constant: channel.mode_dependent_constant,
  };
}

function notAssessedRow(row, sourceRow) {
  return {
    op_index: row.op_index,
    op_name: row.op_name,
    assessment_status: "not_assessed",
    not_assessed_reason: row.not_assessed_reason,
    source_witness_ledger_sha256: sourceRow?.witness_ledger_sha256 || "",
  };
}

function compareSummary(actual, expected) {
  for (const key of [
    "status",
    "candidate_op_count",
    "assessed_op_count",
    "unassessed_op_count",
    "assessed_channel_count",
    "fixed_point_assessed_channel_count",
    "constant_accumulator_channel_count",
    "post_bias_negative_locked_channel_count",
    "post_bias_positive_locked_channel_count",
    "post_bias_zero_containing_channel_count",
    "default_constant_output_channel_count",
    "single_constant_output_channel_count",
    "dual_mode_constant_output_channel_count",
    "nonconstant_accumulator_dual_mode_constant_channel_count",
    "mode_dependent_constant_output_channel_count",
    "default_severely_constrained_channel_count",
    "single_severely_constrained_channel_count",
    "default_full_activation_span_channel_count",
    "single_full_activation_span_channel_count",
    "minimum_default_inclusive_code_span",
    "minimum_single_inclusive_code_span",
  ]) assert(actual[key] === expected[key], `Channel vitality ${key} mismatch.`);
  assert(equalArray(actual.vitality_ranking_op_indices, expected.vitality_ranking_op_indices), "Channel vitality ranking mismatch.");
  assert(JSON.stringify(actual.span_histogram) === JSON.stringify(expected.span_histogram), "Channel vitality histogram mismatch.");
}

function compareOpRow(actual, expected) {
  assert(Number(actual.op_index) === expected.op_index && actual.op_name === expected.op_name, "Channel vitality op binding mismatch.");
  assert(actual.assessment_status === expected.assessment_status, `Channel vitality assessment mismatch at op #${expected.op_index}.`);
  assert(actual.not_assessed_reason === expected.not_assessed_reason, `Channel vitality reason mismatch at op #${expected.op_index}.`);
  if (expected.assessment_status !== "assessed") return;
  for (const key of [
    "assessed_channel_count",
    "fixed_point_assessed_channel_count",
    "constant_accumulator_channel_count",
    "post_bias_negative_locked_channel_count",
    "post_bias_positive_locked_channel_count",
    "post_bias_zero_containing_channel_count",
    "default_constant_output_channel_count",
    "single_constant_output_channel_count",
    "dual_mode_constant_output_channel_count",
    "nonconstant_accumulator_dual_mode_constant_channel_count",
    "mode_dependent_constant_output_channel_count",
    "default_severely_constrained_channel_count",
    "single_severely_constrained_channel_count",
    "default_full_activation_span_channel_count",
    "single_full_activation_span_channel_count",
    "minimum_default_inclusive_code_span",
    "minimum_single_inclusive_code_span",
  ]) assert(actual[key] === expected[key], `Channel vitality ${key} mismatch at op #${expected.op_index}.`);
  for (const key of [
    "output_code_range",
    "activation_code_range",
    "default_minimum_output_codes",
    "default_maximum_output_codes",
    "single_minimum_output_codes",
    "single_maximum_output_codes",
    "post_bias_sign_codes",
    "default_constant_reason_codes",
    "single_constant_reason_codes",
    "default_constant_channel_indices",
    "single_constant_channel_indices",
    "mode_dependent_constant_channel_indices",
  ]) assert(equalArray(actual[key], expected[key]), `Channel vitality ${key} mismatch at op #${expected.op_index}.`);
  assert(actual.source_witness_ledger_sha256 === expected.source_witness_ledger_sha256, `Channel vitality source digest mismatch at op #${expected.op_index}.`);
  assert(actual.top_channels.length === expected.top_channels.length, `Channel vitality top-channel count mismatch at op #${expected.op_index}.`);
  actual.top_channels.forEach((channel, index) => assert(JSON.stringify(channel) === JSON.stringify(expected.top_channels[index]), `Channel vitality top channel mismatch at op #${expected.op_index}.`));
}

export function drawChannelVitalityCanvas(canvas, row, buildMode = "default", field = "span", selectedChannelIndex = null, logicalWidth = null, logicalHeight = null) {
  if (!canvas || !row?.assessed_channel_count) return null;
  const width = logicalWidth || Math.max(300, Math.floor(canvas.parentElement?.clientWidth || canvas.clientWidth || 720));
  const height = logicalHeight || Math.max(240, Math.round(width * 0.52));
  const dpr = Math.max(1, Math.min(2, globalThis.devicePixelRatio || 1));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.fillStyle = "#0c1518";
  context.fillRect(0, 0, width, height);
  const count = Number(row.assessed_channel_count);
  const columns = Math.min(64, Math.max(4, Math.ceil(Math.sqrt(count * 1.8))));
  const rows = Math.ceil(count / columns);
  const gap = count > 512 ? 1 : 2;
  const padX = 20;
  const padTop = 34;
  const padBottom = 35;
  const cellWidth = (width - padX * 2 - gap * (columns - 1)) / columns;
  const cellHeight = (height - padTop - padBottom - gap * (rows - 1)) / rows;
  for (let channel = 0; channel < count; channel += 1) {
    const column = channel % columns;
    const gridRow = Math.floor(channel / columns);
    const x = padX + column * (cellWidth + gap);
    const y = padTop + gridRow * (cellHeight + gap);
    context.fillStyle = vitalityColor(row, channel, buildMode, field);
    context.fillRect(x, y, Math.max(1, cellWidth), Math.max(1, cellHeight));
    if (channel === selectedChannelIndex && cellWidth >= 3 && cellHeight >= 3) {
      context.strokeStyle = "#f5f8f2";
      context.lineWidth = 1.5;
      context.strokeRect(x + 0.75, y + 0.75, Math.max(1, cellWidth - 1.5), Math.max(1, cellHeight - 1.5));
    }
  }
  context.fillStyle = "#d8e5e7";
  context.font = "600 12px system-ui";
  context.fillText(`${buildMode === "single" ? "SINGLE ROUNDING" : "DEFAULT"} / ${field.toUpperCase()}`, padX, 20);
  context.fillStyle = "#8fa4a8";
  context.font = "11px system-ui";
  context.fillText(`${formatNumber(count)} output channels`, padX, height - 11);
  canvas.__channelVitalityState = { row, buildMode, field, columns, rows, gap, padX, padTop, cellWidth, cellHeight, width, height };
  return canvas;
}

function vitalityColor(row, channel, buildMode, field) {
  if (field === "reason") {
    const code = row[`${buildMode}_constant_reason_codes`]?.[channel];
    return ["#2f7774", "#8665ab", "#d45549", "#397db5", "#d39a34"][code] || "#4e6064";
  }
  if (field === "sign") return { "-1": "#3f82bd", 0: "#3d8b7b", 1: "#dc7b3c" }[row.post_bias_sign_codes?.[channel]] || "#4e6064";
  const minimum = row[`${buildMode}_minimum_output_codes`]?.[channel];
  const maximum = row[`${buildMode}_maximum_output_codes`]?.[channel];
  const span = optionalSpan(minimum, maximum);
  if (span == null) return "#4e6064";
  if (span === 1) return "#d45549";
  if (span <= 3) return "#e58943";
  if (span <= 15) return "#d5ad45";
  if (span <= 63) return "#43a29b";
  if (span <= 127) return "#4c89b8";
  if (span <= 255) return "#6fa780";
  return "#294f4e";
}

function installCanvasInteraction(canvas, tooltip, onSelect) {
  canvas.addEventListener("pointermove", (event) => {
    const state = canvas.__channelVitalityState;
    if (!state) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const column = Math.floor((x - state.padX) / (state.cellWidth + state.gap));
    const rowIndex = Math.floor((y - state.padTop) / (state.cellHeight + state.gap));
    const channel = rowIndex * state.columns + column;
    if (column < 0 || rowIndex < 0 || column >= state.columns || channel >= state.row.assessed_channel_count) {
      tooltip.hidden = true;
      return;
    }
    const cellX = state.padX + column * (state.cellWidth + state.gap);
    const cellY = state.padTop + rowIndex * (state.cellHeight + state.gap);
    if (x > cellX + state.cellWidth || y > cellY + state.cellHeight) {
      tooltip.hidden = true;
      return;
    }
    const minimum = state.row[`${state.buildMode}_minimum_output_codes`][channel];
    const maximum = state.row[`${state.buildMode}_maximum_output_codes`][channel];
    const span = optionalSpan(minimum, maximum);
    const reason = REASON_LABELS[state.row[`${state.buildMode}_constant_reason_codes`][channel]];
    const sign = SIGN_LABELS[state.row.post_bias_sign_codes[channel]];
    tooltip.innerHTML = `<strong>channel ${channel}</strong><span>output ${formatOptional(minimum)} .. ${formatOptional(maximum)}</span><span>inclusive span ${formatOptional(span)}</span><span>${humanize(reason)}</span><span>${humanize(sign)}</span>`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(state.width - 210, Math.max(8, x + 12))}px`;
    tooltip.style.top = `${Math.min(state.height - 112, Math.max(8, y + 12))}px`;
    canvas.__channelVitalityHover = channel;
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; canvas.__channelVitalityHover = null; });
  canvas.addEventListener("click", () => {
    if (Number.isInteger(canvas.__channelVitalityHover)) onSelect(canvas.__channelVitalityHover);
  });
}

function selectedChannelEvidence(row, channel) {
  const defaultMinimum = row.default_minimum_output_codes[channel];
  const defaultMaximum = row.default_maximum_output_codes[channel];
  const singleMinimum = row.single_minimum_output_codes[channel];
  const singleMaximum = row.single_maximum_output_codes[channel];
  const defaultReason = row.default_constant_reason_codes[channel];
  const singleReason = row.single_constant_reason_codes[channel];
  return {
    channel_index: channel,
    sign_code: row.post_bias_sign_codes[channel],
    sign_label: SIGN_LABELS[row.post_bias_sign_codes[channel]],
    default: {
      minimum_output_code: defaultMinimum,
      maximum_output_code: defaultMaximum,
      inclusive_code_span: optionalSpan(defaultMinimum, defaultMaximum),
      reason_code: defaultReason,
      reason_label: REASON_LABELS[defaultReason],
    },
    single: {
      minimum_output_code: singleMinimum,
      maximum_output_code: singleMaximum,
      inclusive_code_span: optionalSpan(singleMinimum, singleMaximum),
      reason_code: singleReason,
      reason_label: REASON_LABELS[singleReason],
    },
    dual_mode_constant: defaultReason !== 0 && singleReason !== 0,
    mode_dependent_constant: (defaultReason === 0) !== (singleReason === 0),
  };
}

function compareSelectedWitness(row, selected, reconstructed) {
  assert(reconstructed.channel_index === selected.channel_index, `Selected channel binding mismatch at op #${row.op_index}.`);
  assert(reconstructed.minimum.default_output_code === selected.default.minimum_output_code && reconstructed.maximum.default_output_code === selected.default.maximum_output_code, `Selected default output interval mismatch at op #${row.op_index}.`);
  assert(reconstructed.minimum.single_output_code === selected.single.minimum_output_code && reconstructed.maximum.single_output_code === selected.single.maximum_output_code, `Selected single output interval mismatch at op #${row.op_index}.`);
  const minimum = BigInt(reconstructed.minimum.post_bias_accumulator_decimal);
  const maximum = BigInt(reconstructed.maximum.post_bias_accumulator_decimal);
  const signCode = maximum < 0n ? -1 : minimum > 0n ? 1 : 0;
  assert(signCode === selected.sign_code, `Selected post-bias sign mismatch at op #${row.op_index}.`);
}

function renderSummary(root, evidence) {
  if (!root) return;
  root.replaceChildren(
    metric("Dual-mode constant", formatNumber(evidence.dual_mode_constant_output_channel_count), `${formatPercent(evidence.dual_mode_constant_output_channel_count / Math.max(1, evidence.assessed_channel_count))} of ${formatNumber(evidence.assessed_channel_count)} assessed channels`),
    metric("Variable but constant", formatNumber(evidence.nonconstant_accumulator_dual_mode_constant_channel_count), "constant after fixed-point projection"),
    metric("Mode-dependent", formatNumber(evidence.mode_dependent_constant_output_channel_count), "default versus single rounding"),
    metric("Sign-locked", formatNumber(evidence.post_bias_negative_locked_channel_count + evidence.post_bias_positive_locked_channel_count), `${formatNumber(evidence.post_bias_negative_locked_channel_count)} negative / ${formatNumber(evidence.post_bias_positive_locked_channel_count)} positive`),
    metric("Full default span", formatNumber(evidence.default_full_activation_span_channel_count), `${formatPercent(evidence.default_full_activation_span_channel_count / Math.max(1, evidence.fixed_point_assessed_channel_count))} of projected channels`),
  );
}

function proofPath(selected, reconstructed) {
  const root = document.createElement("div");
  root.className = "channel-vitality-path";
  for (const [label, value, tone] of [
    ["Accumulator", `${formatInteger(reconstructed.minimum.post_bias_accumulator_decimal)} .. ${formatInteger(reconstructed.maximum.post_bias_accumulator_decimal)}`, selected.sign_code ? "watch" : "neutral"],
    ["Default", `${formatOptional(selected.default.minimum_output_code)} .. ${formatOptional(selected.default.maximum_output_code)}`, selected.default.reason_code ? "risk" : "ok"],
    ["Single", `${formatOptional(selected.single.minimum_output_code)} .. ${formatOptional(selected.single.maximum_output_code)}`, selected.single.reason_code ? "risk" : "ok"],
  ]) {
    const item = document.createElement("div");
    item.className = `channel-vitality-stage ${tone}`;
    const name = document.createElement("span");
    name.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    item.append(name, result);
    root.append(item);
  }
  return root;
}

function histogramTable(histogram) {
  const wrap = document.createElement("div");
  wrap.className = "channel-vitality-table-wrap";
  const table = document.createElement("table");
  table.className = "channel-vitality-table channel-vitality-histogram";
  table.innerHTML = "<thead><tr><th>Inclusive code span</th><th>Default channels</th><th>Single-rounding channels</th><th>Delta</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const bin of histogram || []) {
    const tr = document.createElement("tr");
    const delta = Number(bin.single_rounding_channel_count) - Number(bin.default_channel_count);
    for (const value of [bin.label, formatNumber(bin.default_channel_count), formatNumber(bin.single_rounding_channel_count), signed(delta)]) {
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
  wrap.className = "channel-vitality-table-wrap";
  const table = document.createElement("table");
  table.className = "channel-vitality-table";
  table.innerHTML = "<thead><tr><th>Op</th><th>Channels</th><th>Dual constant</th><th>Variable constant</th><th>Mode-dependent</th><th>Minimum span D / S</th><th>Ledger</th><th></th></tr></thead>";
  const tbody = document.createElement("tbody");
  const rows = evidence.vitality_ranking_op_indices.map((index) => evidence.ops.find((row) => row.op_index === index)).filter(Boolean).slice(0, 16);
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
    for (const value of [
      coordinate,
      formatNumber(row.assessed_channel_count),
      formatNumber(row.dual_mode_constant_output_channel_count),
      formatNumber(row.nonconstant_accumulator_dual_mode_constant_channel_count),
      formatNumber(row.mode_dependent_constant_output_channel_count),
      `${formatOptional(row.minimum_default_inclusive_code_span)} / ${formatOptional(row.minimum_single_inclusive_code_span)}`,
      row.vitality_ledger_sha256.slice(0, 12),
      graph,
    ]) {
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

function selectedChannelExport(analysis, evidence, row, selected, reconstructed) {
  return {
    schema: "deepbom.channel_vitality_selected_channel.v1",
    artifact_sha256: analysis.model_sha256,
    source_commit: evidence.source_commit,
    source_evidence_schema: evidence.source_evidence_schema,
    op_index: row.op_index,
    op_name: row.op_name,
    output_channel: selected.channel_index,
    output_code_range: row.output_code_range,
    activation_code_range: row.activation_code_range,
    post_bias_minimum_decimal: reconstructed.minimum.post_bias_accumulator_decimal,
    post_bias_maximum_decimal: reconstructed.maximum.post_bias_accumulator_decimal,
    accumulator_span_decimal: String(BigInt(reconstructed.maximum.post_bias_accumulator_decimal) - BigInt(reconstructed.minimum.post_bias_accumulator_decimal)),
    post_bias_sign_class: selected.sign_label,
    default: {
      ...selected.default,
      minimum_preclamp_code: reconstructed.minimum.default_preclamp_code,
      maximum_preclamp_code: reconstructed.maximum.default_preclamp_code,
    },
    single_rounding: {
      ...selected.single,
      minimum_preclamp_code: reconstructed.minimum.single_preclamp_code,
      maximum_preclamp_code: reconstructed.maximum.single_preclamp_code,
    },
    dual_mode_constant: selected.dual_mode_constant,
    mode_dependent_constant: selected.mode_dependent_constant,
    source_witness_ledger_sha256: row.source_witness_ledger_sha256,
    vitality_ledger_sha256: row.vitality_ledger_sha256,
    constant_proof: evidence.constant_proof,
    span_definition: evidence.span_definition,
    interpretation_boundary: evidence.interpretation_boundary,
  };
}

function buildHistogram(defaultSpans, singleSpans) {
  return HISTOGRAM_BINS.map(([label, minimumValue, maximumValue]) => ({
    label,
    minimum_inclusive_span: minimumValue,
    maximum_inclusive_span: maximumValue,
    default_channel_count: defaultSpans.filter((span) => span >= minimumValue && span <= maximumValue).length,
    single_rounding_channel_count: singleSpans.filter((span) => span >= minimumValue && span <= maximumValue).length,
  }));
}

function compareOpRanking(left, right) {
  return right.nonconstant_accumulator_dual_mode_constant_channel_count - left.nonconstant_accumulator_dual_mode_constant_channel_count
    || right.dual_mode_constant_output_channel_count - left.dual_mode_constant_output_channel_count
    || right.mode_dependent_constant_output_channel_count - left.mode_dependent_constant_output_channel_count
    || right.default_severely_constrained_channel_count - left.default_severely_constrained_channel_count
    || optionalCompare(left.minimum_default_inclusive_code_span, right.minimum_default_inclusive_code_span)
    || left.op_index - right.op_index;
}

function compareChannelRanking(left, right) {
  return Number(right.dual_mode_constant) - Number(left.dual_mode_constant)
    || Number(right.mode_dependent_constant) - Number(left.mode_dependent_constant)
    || Number(!right.constant_accumulator) - Number(!left.constant_accumulator)
    || optionalCompare(left.default.inclusive_code_span, right.default.inclusive_code_span)
    || left.channel_index - right.channel_index;
}

class ByteWriter {
  constructor(size) {
    this.bytes = new Uint8Array(size);
    this.view = new DataView(this.bytes.buffer);
    this.offset = 0;
  }
  writeBytes(bytes) { this.bytes.set(bytes, this.offset); this.offset += bytes.byteLength; }
  writeAscii(value) {
    const bytes = new TextEncoder().encode(String(value));
    assert(bytes.byteLength === 64, "Channel vitality source digest byte count mismatch.");
    this.writeBytes(bytes);
  }
  writeI64(value) { this.view.setBigInt64(this.offset, BigInt(value), true); this.offset += 8; }
  finish() { assert(this.offset === this.bytes.byteLength, "Channel vitality ledger byte count mismatch."); return this.bytes; }
}

function segmentedControl(label, options, value, onChange) {
  const root = document.createElement("div");
  root.className = "channel-vitality-segments";
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
  node.className = "channel-vitality-metric";
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
  dl.className = "channel-vitality-definitions";
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
  root.className = "channel-vitality-unassessed";
  for (const row of unassessed) {
    const item = document.createElement("p");
    item.textContent = `#${row.op_index} ${row.op_name}: ${row.not_assessed_reason}`;
    root.append(item);
  }
  return root;
}

function messageNode(text, tone = "muted") {
  const node = document.createElement("p");
  node.className = `channel-vitality-message ${tone}`;
  node.textContent = text;
  return node;
}

function pathText(path, minimumPreclamp, maximumPreclamp) {
  return `preclamp ${formatOptional(minimumPreclamp)} .. ${formatOptional(maximumPreclamp)} -> output ${formatOptional(path.minimum_output_code)} .. ${formatOptional(path.maximum_output_code)}; span ${formatOptional(path.inclusive_code_span)}; ${humanize(path.reason_label)}`;
}
function optionalSpan(minimumValue, maximumValue) { if (minimumValue == null && maximumValue == null) return null; assert(minimumValue != null && maximumValue != null && maximumValue >= minimumValue, "Channel vitality endpoint span is invalid."); return maximumValue - minimumValue + 1; }
function spans(minimumValues, maximumValues) { return minimumValues.map((value, index) => optionalSpan(value, maximumValues[index])).filter(isNumber); }
function minimum(values) { return values.length ? Math.min(...values) : null; }
function optionalCompare(left, right) { if (left == null && right == null) return 0; if (left == null) return -1; if (right == null) return 1; return left - right; }
function sum(rows, key) { return rows.reduce((total, row) => total + Number(row[key] || 0), 0); }
function equalArray(left, right) { return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function validReasonCode(value) { return Number.isInteger(value) && value >= 0 && value < REASON_LABELS.length; }
function isNumber(value) { return typeof value === "number" && Number.isFinite(value); }
function clampInteger(value, minimumValue, maximumValue) { const number = Number(value); return Math.min(maximumValue, Math.max(minimumValue, Number.isInteger(number) ? number : minimumValue)); }
function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatPercent(value) { return `${(Number(value || 0) * 100).toFixed(2)}%`; }
function formatInteger(value) { try { return BigInt(value).toLocaleString("en-US"); } catch { return String(value ?? "N/A"); } }
function formatOptional(value) { return value == null ? "N/A" : formatNumber(value); }
function signed(value) { const number = Number(value || 0); return `${number > 0 ? "+" : ""}${number.toLocaleString("en-US")}`; }
function humanize(value) { return String(value || "not assessed").replaceAll("_", " "); }
function assert(condition, message) { if (!condition) throw new Error(message); }
