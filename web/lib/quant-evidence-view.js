import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
import {
  buildBiasScaleCheck,
  buildRepresentableKernelChannelCheck,
} from "./quantization-contract-summary.js";
import { classifyTensorRoles } from "./tensor-inventory.js";
import { downloadText } from "./download.js";
import { formatNumber, padOp } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const PAGE_SIZE = 256;

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function button(label, className = "") {
  const node = element("button", className, label);
  node.type = "button";
  return node;
}

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function scientific(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  if (number === 0) return "0";
  return number.toExponential(4);
}

function tensorIndex(tensor, fallback = null) {
  return Number.isInteger(tensor?.index) ? tensor.index : fallback;
}

function consumersFor(analysis, tensorId) {
  return (artifactIrOperators(analysis) || []).filter((op) => (op.inputs || []).includes(tensorId));
}

function lowNormForConsumers(consumers) {
  return consumers.reduce((summary, op) => {
    const count = Number(op.low_norm_filter_count);
    const total = Number(op.low_norm_filter_total);
    if (Number.isInteger(count) && Number.isInteger(total) && total > 0) {
      summary.count += count;
      summary.total += total;
      summary.rows.push({ op_index: op.index, op_name: op.name, count, total });
    }
    return summary;
  }, { count: 0, total: 0, rows: [] });
}

function quantDomain(dtype) {
  return String(dtype || "").toUpperCase() === "UINT8" ? [0, 255] : [-127, 127];
}

function scaleRows(tensor, {
  maximumRepresentableAbsThreshold = 1e-6,
  scaleOutlierRatioThreshold = 1e6,
} = {}) {
  const scales = (tensor?.scale_sample || []).map(Number);
  const zeroPoints = (tensor?.zero_point_sample || []).map(Number);
  const maximumScale = scales.reduce((maximum, value) =>
    Number.isFinite(value) && value > 0 ? Math.max(maximum, value) : maximum, 0);
  const [qmin, qmax] = quantDomain(tensor?.dtype);
  return scales.map((scale, channel) => {
    const zeroPoint = Number.isFinite(zeroPoints[channel])
      ? zeroPoints[channel]
      : Number.isFinite(zeroPoints[0]) ? zeroPoints[0] : 0;
    const codeDistance = Math.max(Math.abs(qmin - zeroPoint), Math.abs(qmax - zeroPoint));
    const maximumRepresentableAbs = scale * codeDistance;
    const maximumToChannelScaleRatio = scale > 0 ? maximumScale / scale : Number.POSITIVE_INFINITY;
    const flagged = maximumRepresentableAbs <= maximumRepresentableAbsThreshold
      && maximumToChannelScaleRatio >= scaleOutlierRatioThreshold;
    const nearThreshold = !flagged
      && maximumRepresentableAbs <= maximumRepresentableAbsThreshold * 2
      && maximumToChannelScaleRatio >= scaleOutlierRatioThreshold / 2;
    return {
      channel,
      scale,
      zeroPoint,
      maximumRepresentableAbs,
      maximumToChannelScaleRatio,
      flagged,
      nearThreshold,
    };
  });
}

function metric(label, value, detail = "", tone = "") {
  const item = element("div", `qe-metric${tone ? ` qe-${tone}` : ""}`);
  item.append(element("span", "", label), element("strong", "", value));
  if (detail) {
    const note = element("small", "", detail);
    note.title = detail;
    item.append(note);
  }
  return item;
}

function renderScaleChart(rows, selectedChannel, onSelectChannel) {
  const panel = element("div", "qe-chart-panel");
  const header = element("div", "qe-chart-head");
  header.append(
    element("strong", "", "Per-axis scale vector"),
    element("span", "", `${formatNumber(rows.length)} channel values / logarithmic y-axis`),
  );
  const svg = svgElement("svg", {
    class: "qe-scale-chart",
    viewBox: "0 0 1000 260",
    role: "img",
    "aria-label": "Complete per-axis quantization scale vector",
    preserveAspectRatio: "none",
  });
  const valid = rows.filter((row) => Number.isFinite(row.scale) && row.scale > 0);
  if (!valid.length) {
    panel.append(header, element("p", "qe-empty", "No positive scale values were emitted."));
    return panel;
  }
  const minimumLog = Math.floor(Math.min(...valid.map((row) => Math.log10(row.scale))));
  const maximumLog = Math.ceil(Math.max(...valid.map((row) => Math.log10(row.scale))));
  const span = Math.max(1e-9, maximumLog - minimumLog);
  const left = 54;
  const right = 18;
  const top = 18;
  const bottom = 32;
  const width = 1000 - left - right;
  const height = 260 - top - bottom;
  const xFor = (channel) => left + channel / Math.max(1, rows.length - 1) * width;
  const yFor = (scale) => top + (maximumLog - Math.log10(Math.max(scale, Number.MIN_VALUE))) / span * height;

  const exponentStep = Math.max(1, Math.ceil(span / 5));
  const exponents = [];
  for (let exponent = maximumLog; exponent >= minimumLog; exponent -= exponentStep) exponents.push(exponent);
  if (exponents.at(-1) !== minimumLog) exponents.push(minimumLog);
  for (const logValue of exponents) {
    const y = top + (maximumLog - logValue) / span * height;
    svg.append(
      svgElement("line", { x1: left, x2: 1000 - right, y1: y, y2: y, class: "qe-grid-line" }),
      svgElement("text", { x: left - 8, y: y + 4, class: "qe-axis-label", "text-anchor": "end" }),
    );
    svg.lastChild.textContent = `1e${logValue}`;
  }

  const points = valid.map((row) => `${xFor(row.channel)},${yFor(row.scale)}`).join(" ");
  svg.append(svgElement("polyline", { points, class: "qe-scale-line" }));
  for (const row of rows.filter((item) => item.flagged)) {
    svg.append(svgElement("circle", {
      cx: xFor(row.channel),
      cy: yFor(row.scale),
      r: row.channel === selectedChannel ? 5 : 3.5,
      class: "qe-scale-flag",
      "data-channel": row.channel,
    }));
  }
  if (Number.isInteger(selectedChannel) && rows[selectedChannel]) {
    const row = rows[selectedChannel];
    svg.append(svgElement("line", {
      x1: xFor(row.channel),
      x2: xFor(row.channel),
      y1: top,
      y2: top + height,
      class: "qe-selection-line",
    }));
  }
  const xStart = svgElement("text", { x: left, y: 250, class: "qe-axis-label", "text-anchor": "start" });
  xStart.textContent = "channel 0";
  const xEnd = svgElement("text", { x: 1000 - right, y: 250, class: "qe-axis-label", "text-anchor": "end" });
  xEnd.textContent = `channel ${Math.max(0, rows.length - 1)}`;
  svg.append(xStart, xEnd);
  svg.addEventListener("click", (event) => {
    const bounds = svg.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / Math.max(1, bounds.width) * 1000;
    const channel = Math.max(0, Math.min(rows.length - 1, Math.round((x - left) / width * Math.max(1, rows.length - 1))));
    onSelectChannel(channel);
  });
  panel.append(header, svg);
  return panel;
}

function renderChannelLedger(rows, page, selectedChannel, onPage, onSelectChannel) {
  const panel = element("div", "qe-channel-ledger");
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.max(0, Math.min(pages - 1, page));
  const toolbar = element("div", "qe-ledger-toolbar");
  const previous = button("Previous", "secondary-action");
  previous.disabled = safePage === 0;
  previous.addEventListener("click", () => onPage(safePage - 1));
  const next = button("Next", "secondary-action");
  next.disabled = safePage >= pages - 1;
  next.addEventListener("click", () => onPage(safePage + 1));
  toolbar.append(
    element("strong", "", "Channel ledger"),
    element("span", "", `${safePage * PAGE_SIZE}-${Math.min(rows.length - 1, (safePage + 1) * PAGE_SIZE - 1)} / ${formatNumber(rows.length)} channels`),
    previous,
    next,
  );
  const wrap = element("div", "qe-table-wrap");
  const table = element("table", "qe-channel-table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Channel", "Scale", "Zero point", "Max |real|", "Layer max scale / channel scale", "Bias code", "INT32 use", "Status"]) {
    headRow.append(element("th", "", label));
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const row of rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE)) {
    const tr = document.createElement("tr");
    if (row.flagged || row.exactZero || row.dualModeConstant) tr.classList.add("qe-flagged-row");
    else if (row.nearThreshold) tr.classList.add("qe-threshold-row");
    if (row.channel === selectedChannel) tr.classList.add("selected-row");
    const values = [
      row.channel,
      scientific(row.scale),
      row.zeroPoint,
      scientific(row.maximumRepresentableAbs),
      scientific(row.maximumToChannelScaleRatio),
      row.biasCode == null ? "N/A" : formatNumber(row.biasCode),
      row.biasInt32Utilization == null ? "N/A" : `${(row.biasInt32Utilization * 100).toFixed(2)}%`,
      row.exactZero && row.dualModeConstant
        ? "EXACT-ZERO + CONSTANT OUTPUT"
        : row.exactZero ? "EXACT-ZERO STORED"
          : row.flagged ? "NEAR-ZERO RANGE"
            : row.nearThreshold ? "NEAR REVIEW THRESHOLD" : "within threshold",
    ];
    for (const value of values) tr.append(element("td", "", String(value)));
    tr.addEventListener("click", () => onSelectChannel(row.channel));
    body.append(tr);
  }
  table.append(head, body);
  wrap.append(table);
  panel.append(toolbar, wrap);
  return panel;
}

function csvForTensor(tensor, rows) {
  const lines = [
    "tensor_index,tensor_name,channel,scale,zero_point,maximum_representable_abs,layer_maximum_scale_to_channel_scale_ratio,near_zero_representable_flag,near_review_threshold_flag,exact_zero_stored_flag,bias_int32_code,bias_int32_utilization,dual_mode_constant_output_flag",
  ];
  const escapedName = `"${String(tensor?.name || "").replaceAll("\"", "\"\"")}"`;
  for (const row of rows) {
    lines.push([
      tensorIndex(tensor),
      escapedName,
      row.channel,
      row.scale,
      row.zeroPoint,
      row.maximumRepresentableAbs,
      row.maximumToChannelScaleRatio,
      row.flagged,
      row.nearThreshold,
      row.exactZero,
      row.biasCode ?? "",
      row.biasInt32Utilization ?? "",
      row.dualModeConstant,
    ].join(","));
  }
  return `${lines.join("\n")}\n`;
}

function renderEmpty(root) {
  const empty = element("div", "qe-empty-state");
  empty.append(
    element("h3", "", "No per-axis INT8 kernel scale vectors"),
    element("p", "", "This artifact does not expose a per-axis INT8/UINT8 kernel tensor that can be assessed by the scale-vector contract."),
  );
  root.replaceChildren(empty);
}

function kernelCandidates(analysis, check = buildRepresentableKernelChannelCheck(analysis)) {
  const roles = new Map(classifyTensorRoles(analysis).map(({ index, role }) => [index, role]));
  return (check.details || [])
    .filter((row) => roles.get(row.tensor_index) === "kernel")
    .sort((left, right) => right.flagged_channel_count - left.flagged_channel_count
      || right.maximum_to_minimum_scale_ratio - left.maximum_to_minimum_scale_ratio
      || left.tensor_index - right.tensor_index);
}

export function createQuantEvidenceController({
  root,
  onOpenOp = () => {},
  onOpenNode = () => {},
} = {}) {
  const state = {
    analysis: null,
    selectedTensorIndex: null,
    selectedChannel: null,
    page: 0,
  };

  function selectTensor(index) {
    state.selectedTensorIndex = Number(index);
    state.selectedChannel = null;
    state.page = 0;
    render();
  }

  function selectChannel(channel) {
    state.selectedChannel = Number(channel);
    state.page = Math.floor(state.selectedChannel / PAGE_SIZE);
    render();
  }

  function selectOp(opIndex) {
    const op = artifactIrOperators(state.analysis).find((item) => item.index === Number(opIndex));
    if (!op) return false;
    const candidateIndices = new Set(kernelCandidates(state.analysis).map((row) => Number(row.tensor_index)));
    const kernelIndex = (op.inputs || []).map(Number).find((index) => candidateIndices.has(index));
    if (!Number.isInteger(kernelIndex)) return false;
    selectTensor(kernelIndex);
    return true;
  }

  function hasEvidenceForOp(opIndex) {
    const op = artifactIrOperators(state.analysis).find((item) => item.index === Number(opIndex));
    if (!op) return false;
    const candidateIndices = new Set(kernelCandidates(state.analysis).map((row) => Number(row.tensor_index)));
    return (op.inputs || []).map(Number).some((index) => candidateIndices.has(index));
  }

  function setAnalysis(analysis) {
    state.analysis = analysis;
    const check = buildRepresentableKernelChannelCheck(analysis);
    const candidates = kernelCandidates(analysis, check);
    if (!candidates.some((row) => row.tensor_index === state.selectedTensorIndex)) {
      state.selectedTensorIndex = (candidates.find((row) => row.flagged_channel_count > 0) || candidates[0])?.tensor_index ?? null;
      state.selectedChannel = null;
      state.page = 0;
    }
    render();
  }

  function render() {
    if (!root || !state.analysis) return;
    const analysis = state.analysis;
    const check = buildRepresentableKernelChannelCheck(analysis);
    const candidates = kernelCandidates(analysis, check);
    if (!candidates.length) {
      renderEmpty(root);
      return;
    }
    const selectedDetail = candidates.find((row) => row.tensor_index === state.selectedTensorIndex) || candidates[0];
    state.selectedTensorIndex = selectedDetail.tensor_index;
    const tensor = (artifactIrValues(analysis) || []).find((item, index) => tensorIndex(item, index) === selectedDetail.tensor_index);
    const consumers = consumersFor(analysis, selectedDetail.tensor_index);
    const lowNorm = lowNormForConsumers(consumers);
    const rows = scaleRows(tensor, {
      maximumRepresentableAbsThreshold: check.maximum_representable_abs_threshold,
      scaleOutlierRatioThreshold: check.scale_outlier_ratio_threshold,
    });
    const zeroSliceDetail = (analysis.weight_integrity?.zero_kernel_slice_details || [])
      .find((item) => Number(item.tensor_index) === Number(selectedDetail.tensor_index));
    const exactZeroChannels = new Set((zeroSliceDetail?.exact_zero_channels || []).map(Number));
    const sampledChannelIndex = new Map((zeroSliceDetail?.channels || []).map((channel, index) => [Number(channel), index]));
    const constantOutputChannels = new Set();
    for (const consumer of consumers) {
      const vitality = (analysis.channel_vitality?.ops || []).find((item) => Number(item.op_index) === Number(consumer.index));
      if (!vitality || vitality.assessment_status !== "assessed") continue;
      const single = new Set((vitality.single_constant_channel_indices || []).map(Number));
      for (const channel of vitality.default_constant_channel_indices || []) {
        if (single.has(Number(channel))) constantOutputChannels.add(Number(channel));
      }
    }
    for (const row of rows) {
      const sampleIndex = sampledChannelIndex.get(row.channel);
      row.exactZero = exactZeroChannels.has(row.channel);
      row.biasCode = sampleIndex == null ? null : zeroSliceDetail?.bias_code_sample?.[sampleIndex] ?? null;
      row.biasInt32Utilization = sampleIndex == null ? null : zeroSliceDetail?.bias_int32_utilization_sample?.[sampleIndex] ?? null;
      row.dualModeConstant = constantOutputChannels.has(row.channel);
    }
    if (state.selectedChannel == null) {
      state.selectedChannel = selectedDetail.flagged_channels?.[0]?.channel ?? 0;
      state.page = Math.floor(state.selectedChannel / PAGE_SIZE);
    }
    const biasCheck = buildBiasScaleCheck(analysis);
    const biasRows = biasCheck.details.filter((row) => row.weight_tensor_index === selectedDetail.tensor_index);

    const head = element("div", "qe-head");
    const title = element("div");
    title.append(
      element("span", "qe-kicker", "QUANTIZATION EVIDENCE"),
      element("h3", "", "Scale vectors and channel-level numerical contracts"),
      element("p", "", "Artifact-derived quantities remain separate from methodology thresholds. Select a tensor or channel to inspect the exact values behind a finding."),
    );
    const exportButton = button("Download scale CSV", "secondary-action");
    exportButton.addEventListener("click", () => downloadText(
      `tensor_${selectedDetail.tensor_index}_scale_vector.csv`,
      csvForTensor(tensor, rows),
      "text/csv",
    ));
    head.append(title, exportButton);

    const summary = element("div", "qe-summary");
    summary.append(
      metric("Assessed scale channels", formatNumber(check.assessed_channels), `${check.assessed_kernel_tensors} per-axis kernel tensor(s)`),
      metric(
        "Near-zero representable",
        formatNumber(selectedDetail.flagged_channel_count),
        `Selected T${selectedDetail.tensor_index}; ${formatNumber(check.flagged_channels)} across all kernels. Exact range <= ${scientific(check.maximum_representable_abs_threshold)} and scale outlier >= ${scientific(check.scale_outlier_ratio_threshold)}x`,
        selectedDetail.flagged_channel_count ? "risk" : "ok",
      ),
      metric(
        "Decoded low-norm filters",
        `${formatNumber(lowNorm.count)} / ${formatNumber(lowNorm.total)}`,
        "Separate value-domain heuristic: decoded filter L2 norm < 2% of the layer maximum. It is not the scale-vector count.",
        lowNorm.count ? "warn" : "",
      ),
      metric(
        "Exact-zero stored",
        formatNumber(exactZeroChannels.size),
        `${[...exactZeroChannels].filter((channel) => constantOutputChannels.has(channel)).length} also have a constant output code under both pinned fixed-point paths`,
        exactZeroChannels.size ? "risk" : "ok",
      ),
      metric(
        "Bias-scale consistency",
        biasRows.length ? (biasRows.some((row) => row.status === "fail") ? "Mismatch" : "Pass") : "Not assessed",
        biasRows.length ? "bias_scale[channel] = input_scale × weight_scale[channel]" : "No linked bias-scale group",
        biasRows.some((row) => row.status === "fail") ? "risk" : "",
      ),
    );

    const layout = element("div", "qe-layout");
    const tensorList = element("aside", "qe-tensor-list");
    tensorList.append(element("h4", "", "Per-axis kernel tensors"));
    const appendCandidate = (candidate, parent = tensorList) => {
      const item = button("", `qe-tensor-item${candidate.tensor_index === selectedDetail.tensor_index ? " active" : ""}`);
      item.append(
        element("strong", "", `T${candidate.tensor_index}`),
        element("span", "", candidate.tensor_name || "unnamed kernel"),
        element(
          "small",
          candidate.flagged_channel_count ? "qe-risk-text" : "",
          `${formatNumber(candidate.assessed_channel_count)} scales / ${formatNumber(candidate.flagged_channel_count)} near-zero / ${scientific(candidate.maximum_to_minimum_scale_ratio)}x spread`,
        ),
      );
      item.addEventListener("click", () => selectTensor(candidate.tensor_index));
      parent.append(item);
    };
    const signaledCandidates = candidates.filter((candidate) => candidate.flagged_channel_count > 0);
    const quietCandidates = candidates.filter((candidate) => candidate.flagged_channel_count === 0);
    for (const candidate of signaledCandidates) appendCandidate(candidate);
    if (quietCandidates.length) {
      const quiet = document.createElement("details");
      quiet.className = "qe-quiet-tensors";
      quiet.append(element("summary", "", `${formatNumber(quietCandidates.length)} tensors without a near-zero range signal`));
      for (const candidate of quietCandidates) appendCandidate(candidate, quiet);
      tensorList.append(quiet);
    }

    const detail = element("section", "qe-detail");
    const selectedHead = element("div", "qe-selected-head");
    const selectedTitle = element("div");
    selectedTitle.append(
      element("span", "qe-kicker", `T${selectedDetail.tensor_index} / axis ${selectedDetail.quantized_dimension}`),
      element("h4", "", selectedDetail.tensor_name || "Unnamed kernel tensor"),
      element("p", "", `${tensor?.dtype || selectedDetail.dtype} / ${(tensor?.shape || []).join("x") || "shape unavailable"} / ${formatNumber(rows.length)} emitted scale values`),
    );
    const opButtons = element("div", "qe-op-buttons");
    for (const op of consumers.slice(0, 8)) {
      const opButton = button(`#${padOp(op.index)} ${op.name}`, "secondary-action");
      opButton.title = "Open this consumer in Node View";
      opButton.addEventListener("click", () => onOpenNode(op.index));
      opButtons.append(opButton);
    }
    selectedHead.append(selectedTitle, opButtons);

    const definitions = element("div", "qe-definition-ledger");
    const selectedScale = rows[state.selectedChannel] || null;
    definitions.append(
      selectedScale
        ? element(
            "p",
            selectedScale.flagged ? "qe-risk-text" : "",
            `Selected channel ${selectedScale.channel}: scale ${scientific(selectedScale.scale)}, zero point ${selectedScale.zeroPoint}, max representable |real weight| ${scientific(selectedScale.maximumRepresentableAbs)}, layer max scale / channel scale ${scientific(selectedScale.maximumToChannelScaleRatio)}x${selectedScale.biasCode == null ? "" : `, bias code ${formatNumber(selectedScale.biasCode)} (${(selectedScale.biasInt32Utilization * 100).toFixed(2)}% INT32)`}${selectedScale.dualModeConstant ? ", constant output under both pinned fixed-point paths" : ""}.`,
          )
        : element("p", "", "No channel is selected."),
      element("p", "", `Near-zero representable channels: ${selectedDetail.flagged_channel_count}. This uses only scale, zero point, and the quantized code domain; range and ratio are exact, while the two thresholds are declared heuristics.`),
      element("p", "", `The scale ratio is the layer maximum scale divided by this channel's scale. It is not max |real| / scale and can exceed the INT8 code limit. Exact-zero stored channels use centered integer codes and are reported separately from the representable-range heuristic.`),
      element("p", "", `Decoded low-norm filters: ${lowNorm.count}/${lowNorm.total}. This decodes stored weight values and compares each filter's L2 norm with 2% of its layer maximum. A channel may satisfy either, both, or neither test.`),
    );
    for (const row of lowNorm.rows) {
      const open = button(`Inspect #${padOp(row.op_index)} ${row.op_name}`, "text-action");
      open.addEventListener("click", () => onOpenOp(row.op_index));
      definitions.append(open);
    }
    for (const row of biasRows) {
      definitions.append(element(
        "p",
        row.status === "fail" ? "qe-risk-text" : "qe-pass-text",
        `Bias T${row.bias_tensor_index}: ${row.status.toUpperCase()} across ${row.checked_channels} channel(s); max relative error ${scientific(row.maximum_relative_error)}.`,
      ));
    }

    detail.append(
      selectedHead,
      definitions,
      renderScaleChart(rows, state.selectedChannel, selectChannel),
      renderChannelLedger(rows, state.page, state.selectedChannel, (page) => {
        state.page = page;
        render();
      }, selectChannel),
    );
    layout.append(tensorList, detail);
    root.replaceChildren(head, summary, layout);
  }

  function resetInteractionState() {
    state.analysis = null;
    state.selectedTensorIndex = null;
    state.selectedChannel = null;
    state.page = 0;
    root?.replaceChildren();
  }

  return { setAnalysis, render, selectTensor, selectOp, hasEvidenceForOp, resetInteractionState };
}
