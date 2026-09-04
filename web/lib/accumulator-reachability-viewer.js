import { formatNumber, padOp } from "./format.js";
import { browserAssetUrl } from "./browser-asset-url.js";
import {
  reconstructAccumulatorReachabilityChannel,
  validateAccumulatorReachabilityShape,
} from "./accumulator-reachability.js";

export function createAccumulatorReachabilityController({
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
  let field = "exact";
  let worker = null;
  let verificationObserver = null;
  let resizeObserver = null;
  let renderToken = 0;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "accumulator_reachability.json");
  });

  function render(explicitAnalysis = null) {
    renderToken += 1;
    const token = renderToken;
    worker?.terminate();
    worker = null;
    verificationObserver?.disconnect();
    verificationObserver = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    const context = getContext?.() || {};
    analysis = explicitAnalysis || context.analysis || null;
    modelBytes = context.modelBytes || null;
    evidence = analysis?.accumulator_reachability || null;
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
      validateAccumulatorReachabilityShape(evidence);
      const assessed = evidence.ops.filter((row) => row.assessment_status === "assessed");
      if (!assessed.some((row) => row.op_index === selectedOpIndex)) {
        selectedOpIndex = evidence.reachability_ranking_op_indices?.[0] ?? assessed[0]?.op_index ?? null;
        selectedChannelIndex = null;
      }
      renderSummary(summary, evidence);
      renderBody();
      setStatus(status, assessed.length ? "source bound / verification queued" : humanize(evidence.status), assessed.length ? "watch" : "ok");
      if (assessed.length) scheduleVerification(token);
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(message(`Accumulator reachability evidence rejected: ${error.message}`, "risk"));
      setStatus(status, "evidence rejected", "risk");
    }
  }

  function scheduleVerification(token) {
    const start = () => {
      if (token !== renderToken || worker || typeof Worker !== "function") return;
      setStatus(status, "independent reconstruction running", "watch");
      worker = new Worker(browserAssetUrl("./lib/accumulator-reachability-worker.js", "./accumulator-reachability-worker.js", import.meta.url), { type: "module" });
      worker.onmessage = (event) => {
        if (token !== renderToken) return;
        setStatus(status, event.data?.ok ? "independently verified" : `integrity error: ${event.data?.error || "verification failed"}`, event.data?.ok ? "ok" : "risk");
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
    };
    if (typeof IntersectionObserver !== "function" || !root) {
      start();
      return;
    }
    verificationObserver = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      verificationObserver?.disconnect();
      verificationObserver = null;
      start();
    }, { rootMargin: "240px", threshold: 0.01 });
    verificationObserver.observe(root);
  }

  function renderBody() {
    if (!body || !evidence) return;
    resizeObserver?.disconnect();
    const rows = evidence.ops.filter((row) => row.assessment_status === "assessed");
    if (!rows.length) {
      body.replaceChildren(message("No quantized accumulator channel was eligible for reachability proof."));
      return;
    }
    const row = rows.find((candidate) => candidate.op_index === selectedOpIndex) || rows[0];
    selectedOpIndex = row.op_index;
    if (!Number.isInteger(selectedChannelIndex) || selectedChannelIndex < 0 || selectedChannelIndex >= row.assessed_channel_count) {
      selectedChannelIndex = row.top_channels?.[0]?.channel_index ?? 0;
    }
    let selected;
    try {
      selected = reconstructAccumulatorReachabilityChannel(analysis, modelBytes, row.op_index, selectedChannelIndex);
    } catch (error) {
      body.replaceChildren(message(`Selected channel reconstruction failed: ${error.message}`, "risk"));
      return;
    }

    const toolbar = element("div", "reachability-toolbar");
    const opLabel = element("label", "reachability-select-label", "Operator");
    const opSelect = element("select", "reachability-op-select");
    opSelect.setAttribute("aria-label", "Accumulator reachability operator");
    rows.forEach((candidate) => {
      const option = new Option(`#${padOp(candidate.op_index)} ${candidate.op_name} / ${formatInteger(candidate.exact_reachable_divergent_state_count_decimal)} exact / ${formatInteger(candidate.unresolved_divergent_state_count_decimal)} unresolved`, String(candidate.op_index));
      option.selected = candidate.op_index === row.op_index;
      opSelect.append(option);
    });
    opSelect.addEventListener("change", () => { selectedOpIndex = Number(opSelect.value); selectedChannelIndex = null; renderBody(); });
    opLabel.append(opSelect);
    const channelLabel = element("label", "reachability-channel-control", "Channel");
    const channelInput = document.createElement("input");
    channelInput.type = "number";
    channelInput.min = "0";
    channelInput.max = String(row.assessed_channel_count - 1);
    channelInput.value = String(selectedChannelIndex);
    channelInput.addEventListener("change", () => {
      selectedChannelIndex = Math.max(0, Math.min(row.assessed_channel_count - 1, Number(channelInput.value) || 0));
      renderBody();
    });
    channelLabel.append(channelInput);
    const modes = element("div", "reachability-modes");
    [["exact", "Exact"], ["excluded", "Excluded"], ["unresolved", "Unresolved"], ["gcd", "GCD"]].forEach(([value, label]) => {
      const button = element("button", "", label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(field === value));
      button.addEventListener("click", () => { field = value; renderBody(); });
      modes.append(button);
    });
    const actions = element("div", "reachability-actions");
    actions.append(
      commandButton("Channel JSON", "Download the selected channel's reconstructed proof and witness", () => onDownload?.(
        selectedChannelExport(analysis, evidence, row, selected),
        `accumulator_reachability_op_${row.op_index}_channel_${selectedChannelIndex}.json`,
      )),
      commandButton("Graph op", "Open this operator in the graph workspace", () => jumpToGraphOp?.(row.op_index)),
    );
    toolbar.append(opLabel, channelLabel, modes, actions);

    const main = element("div", "reachability-main");
    const plots = element("div", "reachability-plots");
    const heatWrap = element("div", "reachability-canvas-wrap");
    const heat = element("canvas", "reachability-heatmap");
    heat.tabIndex = 0;
    heat.setAttribute("role", "img");
    heat.setAttribute("aria-label", `Accumulator reachability channels for operator ${row.op_index}`);
    const heatTooltip = element("div", "reachability-tooltip");
    heatTooltip.hidden = true;
    heatWrap.append(heat, heatTooltip);
    installHeatInteraction(heat, heatTooltip, row, () => field, (channel) => { selectedChannelIndex = channel; renderBody(); });
    const traceWrap = element("div", "reachability-trace-wrap");
    const trace = element("canvas", "reachability-trace");
    trace.setAttribute("role", "img");
    trace.setAttribute("aria-label", `Certified accumulator lattice and rounding divergence for channel ${selectedChannelIndex}`);
    traceWrap.append(trace);
    plots.append(heatWrap, traceWrap);

    const detail = element("div", "reachability-details");
    detail.append(
      definitionTable([
        ["Proof", humanize(selected.proof_status)],
        ["Lattice", selected.lattice_gcd ? `accumulator = minimum + ${formatNumber(selected.lattice_gcd)}k` : "singleton accumulator"],
        ["Interval", `${formatInteger(selected.post_bias_minimum_decimal)} .. ${formatInteger(selected.post_bias_maximum_decimal)} / ${formatInteger(selected.interval_state_count_decimal)} states`],
        ["Certified reachable", formatInteger(selected.certified_reachable_state_count_decimal)],
        ["Residue excluded", formatInteger(selected.provably_unreachable_state_count_decimal)],
        ["Unresolved", formatInteger(selected.unresolved_state_count_decimal)],
        ["Divergent partition", `${formatInteger(selected.exact_reachable_divergent_state_count_decimal)} exact / ${formatInteger(selected.provably_unreachable_divergent_state_count_decimal)} excluded / ${formatInteger(selected.unresolved_divergent_state_count_decimal)} unresolved`],
        ["First exact counterexample", selected.first_exact_reachable_divergent_accumulator_decimal == null ? "none" : `acc ${formatInteger(selected.first_exact_reachable_divergent_accumulator_decimal)} -> default ${selected.first_default_output_code}, single ${selected.first_single_output_code}`],
        ["Source ledger", row.reachability_ledger_sha256],
      ]),
      legend(),
      message("Exact here means kernel-local bounded-sum reachability under independent legal input codes. It is not full-model-input reachability or runtime frequency.", "boundary"),
    );
    main.append(plots, detail);

    body.replaceChildren(
      toolbar,
      main,
      inputWitnessScope(analysis),
      coverageTable(selected),
      witnessTable(selected),
      rankingTable(evidence, row.op_index, (opIndex) => { selectedOpIndex = opIndex; selectedChannelIndex = null; renderBody(); }, jumpToGraphOp),
      message(evidence.interpretation_boundary, "boundary"),
    );
    const redraw = () => {
      drawAccumulatorReachabilityHeatmap(heat, row, field, selectedChannelIndex);
      drawAccumulatorReachabilityTrace(trace, selected);
    };
    requestAnimationFrame(redraw);
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(redraw);
      resizeObserver.observe(main);
    }
  }

  return { render };
}

function inputWitnessScope(analysis) {
  const evidence = analysis?.input_counterexample;
  if (!evidence || evidence.schema !== "deepbom.input_counterexample.v1") {
    return message("Full-model-input witness classification is unavailable.", "boundary");
  }
  const constructive = Number(evidence.tensor_abi_constructive_source_op_count || 0);
  const unresolved = Number(evidence.upstream_activation_unresolved_source_op_count || 0);
  const notAssessed = Number(evidence.not_assessed_source_op_count || 0);
  return message(
    `${formatInteger(evidence.exact_local_source_op_count)} exact-local source ops = ${formatInteger(constructive)} full model-input constructive + ${formatInteger(unresolved)} upstream-activation unresolved + ${formatInteger(notAssessed)} not assessed. A source is lifted to a complete tensor-ABI witness only when its activation input is a declared model input; intermediate activations remain unresolved until exact inversion through upstream nonlinear and quantized operators is established.`,
    constructive ? "review" : "boundary",
  );
}

export function drawAccumulatorReachabilityHeatmap(canvas, row, field = "exact", selectedChannel = null, logicalWidth = null) {
  if (!canvas || !row?.assessed_channel_count) return;
  const width = Math.max(320, Math.round(logicalWidth || canvas.clientWidth || 760));
  const columns = Math.max(16, Math.min(64, Math.floor(width / 13)));
  const lines = Math.ceil(row.assessed_channel_count / columns);
  const cell = Math.max(7, Math.min(14, Math.floor((width - 32) / columns)));
  const height = Math.max(132, lines * cell + 34);
  const scale = globalThis.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.fillStyle = "#0c1518";
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

export function drawAccumulatorReachabilityTrace(canvas, selected, logicalWidth = null, logicalHeight = 250) {
  if (!canvas || !selected) return;
  const width = Math.max(320, Math.round(logicalWidth || canvas.clientWidth || 760));
  const height = logicalHeight;
  const scale = globalThis.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.fillStyle = "#0c1518";
  context.fillRect(0, 0, width, height);
  const left = 44;
  const right = width - 18;
  const minimum = BigInt(selected.post_bias_minimum_decimal);
  const maximum = BigInt(selected.post_bias_maximum_decimal);
  const span = maximum - minimum || 1n;
  const xFor = (value) => left + Number((BigInt(value) - minimum) * 1_000_000n / span) / 1_000_000 * (right - left);
  context.fillStyle = "#273237";
  context.fillRect(left, 56, right - left, 22);
  for (const segment of selected.rounding_segments.filter((row) => row.divergent)) {
    const x1 = xFor(segment.accumulator_minimum_decimal);
    const x2 = xFor(segment.accumulator_maximum_decimal);
    context.fillStyle = "rgba(231, 103, 77, 0.82)";
    context.fillRect(x1, 56, Math.max(1, x2 - x1 + 1), 22);
  }
  const gcd = BigInt(selected.lattice_gcd || 1);
  const prefixSteps = BigInt(selected.certified_prefix_lattice_step_count_decimal);
  const prefixMaximum = minimum + gcd * prefixSteps;
  const suffixMinimum = maximum - gcd * prefixSteps;
  context.fillStyle = "#4fb49f";
  if (selected.proof_status === "partial_endpoint_bands") {
    context.fillRect(left, 102, Math.max(2, xFor(prefixMaximum) - left), 20);
    context.fillRect(xFor(suffixMinimum), 102, Math.max(2, right - xFor(suffixMinimum)), 20);
    context.fillStyle = "#9b7f4d";
    context.fillRect(xFor(prefixMaximum), 102, Math.max(1, xFor(suffixMinimum) - xFor(prefixMaximum)), 20);
  } else {
    context.fillRect(left, 102, right - left, 20);
  }
  if (selected.lattice_gcd > 1 && selected.lattice_gcd <= 64) {
    context.fillStyle = "#d7ece7";
    for (let value = minimum, count = 0; value <= maximum && count < 2200; value += gcd, count += 1) {
      const x = xFor(value);
      context.fillRect(x, 132, 1, 9);
    }
  }
  context.fillStyle = "#91a5aa";
  context.font = "11px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText("rounding divergence", left, 46);
  context.fillText("certified lattice bands", left, 93);
  context.fillText(formatInteger(selected.post_bias_minimum_decimal), left, height - 18);
  const maximumLabel = formatInteger(selected.post_bias_maximum_decimal);
  context.fillText(maximumLabel, right - context.measureText(maximumLabel).width, height - 18);
  context.fillStyle = "#e7674d";
  context.fillRect(left, 158, 12, 8);
  context.fillStyle = "#aebfc2";
  context.fillText("divergent interval", left + 18, 166);
  context.fillStyle = "#4fb49f";
  context.fillRect(left + 142, 158, 12, 8);
  context.fillStyle = "#aebfc2";
  context.fillText("exact reachable", left + 160, 166);
  context.fillStyle = "#9b7f4d";
  context.fillRect(left + 272, 158, 12, 8);
  context.fillStyle = "#aebfc2";
  context.fillText("unresolved compatible", left + 290, 166);
  if (selected.first_exact_reachable_divergent_accumulator_decimal != null) {
    const x = xFor(selected.first_exact_reachable_divergent_accumulator_decimal);
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.5;
    context.beginPath(); context.moveTo(x, 44); context.lineTo(x, 145); context.stroke();
    context.fillStyle = "#ffffff";
    context.fillText("first exact", Math.min(right - 66, x + 4), 190);
  }
}

export function renderAccumulatorReachabilityCanvas(analysis, filename = "") {
  const evidence = analysis?.accumulator_reachability;
  const canvas = document.createElement("canvas");
  const logicalWidth = 1180;
  const logicalHeight = 820;
  canvas.width = logicalWidth * 2;
  canvas.height = logicalHeight * 2;
  const context = canvas.getContext("2d");
  context.scale(2, 2);
  context.fillStyle = "#0b1215";
  context.fillRect(0, 0, logicalWidth, logicalHeight);
  context.fillStyle = "#edf4f4";
  context.font = "700 25px system-ui, sans-serif";
  context.fillText("Accumulator Reachability Lattice", 48, 54);
  context.fillStyle = "#8fa3a7";
  context.font = "12px system-ui, sans-serif";
  context.fillText(filename || analysis?.filename || "TFLite artifact", 48, 78);
  if (!evidence?.assessed_op_count) {
    context.fillText("No assessed quantized accumulator channels.", 48, 130);
    return canvas;
  }
  const metrics = [
    ["Assessed channels", evidence.assessed_channel_count],
    ["Complete integer", evidence.complete_integer_interval_channel_count],
    ["Complete modular", evidence.complete_modular_lattice_channel_count],
    ["Partial bands", evidence.partial_band_channel_count],
    ["Exact divergent", evidence.exact_reachable_divergent_state_count_decimal],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 48 + index * 218;
    context.fillStyle = "#7f9599";
    context.font = "11px system-ui, sans-serif";
    context.fillText(label, x, 128);
    context.fillStyle = index === 4 ? "#efb06d" : "#e5eeee";
    context.font = "700 20px ui-monospace, monospace";
    context.fillText(formatInteger(value), x, 154);
  });
  const rowsByIndex = new Map(evidence.ops.map((row) => [row.op_index, row]));
  const rows = evidence.reachability_ranking_op_indices.slice(0, 18).map((index) => rowsByIndex.get(index)).filter(Boolean);
  context.fillStyle = "#dfeaea";
  context.font = "700 14px system-ui, sans-serif";
  context.fillText("Exact kernel-local rounding divergence by operator", 48, 208);
  const maximum = rows.reduce((value, row) => maxBigInt(value, BigInt(row.exact_reachable_divergent_state_count_decimal)), 1n);
  rows.forEach((row, index) => {
    const y = 238 + index * 29;
    const value = BigInt(row.exact_reachable_divergent_state_count_decimal);
    const unresolved = BigInt(row.unresolved_divergent_state_count_decimal);
    context.fillStyle = "#8fa3a7";
    context.font = "11px ui-monospace, monospace";
    context.fillText(`#${padOp(row.op_index)} ${row.op_name}`, 48, y + 11);
    const width = Number(value * 650n / maximum);
    context.fillStyle = "#4fb49f";
    context.fillRect(240, y, Math.max(value ? 2 : 0, width), 13);
    if (unresolved > 0n) {
      const unresolvedWidth = Math.max(2, Number(unresolved * 650n / maximum));
      context.fillStyle = "#9b7f4d";
      context.fillRect(240 + width, y, unresolvedWidth, 13);
    }
    context.fillStyle = "#cbd9da";
    context.fillText(`${formatInteger(value)} exact / ${formatInteger(unresolved)} unresolved`, 910, y + 11);
  });
  context.fillStyle = "#7f9296";
  context.font = "11px system-ui, sans-serif";
  wrapText(context, evidence.interpretation_boundary, 48, 780, 1080, 16);
  return canvas;
}

function renderSummary(container, evidence) {
  if (!container) return;
  const exact = BigInt(evidence.exact_reachable_divergent_state_count_decimal);
  const divergent = BigInt(evidence.interval_divergent_state_count_decimal);
  const metrics = [
    ["Complete lattice", formatNumber(evidence.complete_integer_interval_channel_count + evidence.complete_modular_lattice_channel_count), `${formatNumber(evidence.assessed_channel_count)} assessed channels`],
    ["Partial bands", formatNumber(evidence.partial_band_channel_count), `${formatNumber(evidence.singleton_channel_count)} singleton`],
    ["Exact divergent", formatInteger(exact), `${formatPercent(ratio(exact, divergent))} of interval divergence`],
    ["Residue excluded", formatInteger(evidence.provably_unreachable_divergent_state_count_decimal), "mathematically incompatible"],
    ["Unresolved", formatInteger(evidence.unresolved_divergent_state_count_decimal), `${formatNumber(evidence.unresolved_divergent_channel_count)} channels`],
  ];
  container.replaceChildren(...metrics.map(([label, value, detail]) => {
    const metric = element("div", "reachability-metric");
    metric.append(element("span", "", label), element("strong", "", value), element("small", "", detail));
    return metric;
  }));
}

function coverageTable(selected) {
  return tableBlock("Bounded denomination coverage", ["|centered weight|", "Normalized d", "Terms", "Coefficient capacity", "R before", "R after", "Proof step"], selected.denomination_coverage_steps.map((step) => [
    formatNumber(step.absolute_centered_weight), formatNumber(step.normalized_denomination), formatNumber(step.term_count), formatInteger(step.aggregate_coefficient_capacity_decimal), formatInteger(step.reachable_prefix_before_decimal), formatInteger(step.reachable_prefix_after_decimal), humanize(step.coverage_status),
  ]), "reachability-coverage-table");
}

function witnessTable(selected) {
  const rows = selected.first_exact_reachable_aggregate_coefficient_witness || [];
  if (!rows.length) return message("No exact divergent state exists for the selected channel, so no counterexample coefficient witness is emitted.");
  return tableBlock("First exact divergent aggregate witness", ["|centered weight|", "Normalized d", "Terms", "Aggregate code delta", "Capacity"], rows.map((row) => [
    formatNumber(row.absolute_centered_weight), formatNumber(row.normalized_denomination), formatNumber(row.term_count), formatInteger(row.aggregate_input_code_delta_decimal), formatInteger(row.aggregate_capacity_decimal),
  ]), "reachability-witness-table");
}

function rankingTable(evidence, selectedOpIndex, onSelect, jumpToGraphOp) {
  const rows = evidence.reachability_ranking_op_indices.slice(0, 16).map((index) => evidence.ops.find((row) => row.op_index === index)).filter(Boolean);
  const block = tableBlock("Reachable numerical ABI ranking", ["Op", "Exact channels", "Exact divergent", "Excluded", "Unresolved", "Max GCD"], rows.map((row) => [
    `#${padOp(row.op_index)} ${row.op_name}`,
    `${formatNumber(row.exact_reachable_divergent_channel_count)} / ${formatNumber(row.assessed_channel_count)}`,
    formatInteger(row.exact_reachable_divergent_state_count_decimal),
    formatInteger(row.provably_unreachable_divergent_state_count_decimal),
    formatInteger(row.unresolved_divergent_state_count_decimal),
    formatNumber(row.maximum_lattice_gcd || 0),
  ]), "reachability-ranking");
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

function installHeatInteraction(canvas, tooltip, row, getField, onSelect) {
  const channelAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const columns = Number(canvas.dataset.columns || 1);
    const cell = Number(canvas.dataset.cell || 1);
    const column = Math.floor((event.clientX - rect.left - Number(canvas.dataset.offsetX || 0)) / cell);
    const line = Math.floor((event.clientY - rect.top - Number(canvas.dataset.offsetY || 0)) / cell);
    const channel = line * columns + column;
    return column >= 0 && column < columns && line >= 0 && channel < row.assessed_channel_count ? channel : null;
  };
  canvas.addEventListener("pointermove", (event) => {
    const channel = channelAt(event);
    if (channel == null) { tooltip.hidden = true; return; }
    tooltip.textContent = `ch ${channel} / ${getField()} / ${humanize(row.channel_proof_statuses[channel])} / gcd ${row.channel_lattice_gcds[channel]} / exact ${formatInteger(row.channel_exact_reachable_divergent_state_counts_decimal[channel])} / excluded ${formatInteger(row.channel_provably_unreachable_divergent_state_counts_decimal[channel])} / unresolved ${formatInteger(row.channel_unresolved_divergent_state_counts_decimal[channel])}`;
    tooltip.hidden = false;
    const rect = canvas.getBoundingClientRect();
    tooltip.style.left = `${Math.min(rect.width - 270, Math.max(8, event.clientX - rect.left + 12))}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - rect.top + 12)}px`;
  });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  canvas.addEventListener("click", (event) => { const channel = channelAt(event); if (channel != null) onSelect(channel); });
}

function channelColor(row, channel, field) {
  if (field === "gcd") {
    const gcd = row.channel_lattice_gcds[channel];
    if (!gcd) return "#5c6570";
    if (gcd === 1) return "#347f70";
    const light = 42 + Math.min(22, Math.log2(gcd) * 6);
    return `hsl(191 48% ${light}%)`;
  }
  const keys = {
    exact: "channel_exact_reachable_divergent_state_counts_decimal",
    excluded: "channel_provably_unreachable_divergent_state_counts_decimal",
    unresolved: "channel_unresolved_divergent_state_counts_decimal",
  };
  const value = BigInt(row[keys[field]][channel]);
  if (!value) return "#243237";
  const magnitude = Math.min(1, Math.log10(Number(value > 1_000_000n ? 1_000_000n : value) + 1) / 6);
  if (field === "exact") return `hsl(168 48% ${34 + magnitude * 28}%)`;
  if (field === "excluded") return `hsl(8 66% ${36 + magnitude * 26}%)`;
  return `hsl(39 56% ${37 + magnitude * 25}%)`;
}

function selectedChannelExport(analysis, evidence, row, selected) {
  return {
    schema: "deepbom.accumulator_reachability_selected_channel.v1",
    generated_from: {
      model_sha256: analysis.model_sha256,
      target_profile_id: analysis.target_profile?.id,
      evidence_schema: evidence.schema,
      method_version: evidence.method_version,
      source_commit: evidence.source_commit,
      source_witness_ledger_sha256: row.source_witness_ledger_sha256,
      source_rounding_equivalence_ledger_sha256: row.source_rounding_equivalence_ledger_sha256,
      reachability_ledger_sha256: row.reachability_ledger_sha256,
    },
    op_index: row.op_index,
    op_name: row.op_name,
    ...selected,
    interpretation_boundary: evidence.interpretation_boundary,
  };
}

function definitionTable(rows) {
  const list = element("dl", "reachability-definition");
  rows.forEach(([term, value]) => { list.append(element("dt", "", term), element("dd", "", value)); });
  return list;
}

function legend() {
  const value = element("div", "reachability-legend");
  [["exact", "Exact reachable"], ["excluded", "Residue excluded"], ["unresolved", "Unresolved compatible"]].forEach(([tone, label]) => {
    const item = element("span", "");
    item.dataset.tone = tone;
    item.append(element("i", ""), document.createTextNode(label));
    value.append(item);
  });
  return value;
}

function tableBlock(title, headers, rows, className) {
  const block = element("section", `reachability-table ${className}`);
  block.append(element("h4", "", title));
  const scroll = element("div", "reachability-table-scroll");
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

function commandButton(label, title, onClick) {
  const button = element("button", "secondary-action", label);
  button.type = "button";
  button.title = title;
  button.addEventListener("click", onClick);
  return button;
}

function message(text, tone = "") {
  const value = element("p", "reachability-message", text);
  if (tone) value.dataset.tone = tone;
  return value;
}

function element(tag, className = "", text = null) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text != null) value.textContent = String(text);
  return value;
}

function setStatus(node, text, tone) {
  if (!node) return;
  node.textContent = text;
  node.dataset.tone = tone || "";
}

function formatInteger(value) {
  try { return BigInt(value || 0).toLocaleString("en-US"); } catch { return String(value ?? "0"); }
}

function formatPercent(value) { return `${(Number(value || 0) * 100).toFixed(2)}%`; }
function ratio(numerator, denominator) { return denominator ? Number(numerator) / Number(denominator) : 0; }
function humanize(value) { return String(value || "").replaceAll("_", " "); }
function maxBigInt(left, right) { return left > right ? left : right; }
function wrapText(context, text, x, y, maxWidth, lineHeight) { let line = ""; for (const word of String(text || "").split(/\s+/)) { const next = line ? `${line} ${word}` : word; if (context.measureText(next).width > maxWidth && line) { context.fillText(line, x, y); y += lineHeight; line = word; } else line = next; } if (line) context.fillText(line, x, y); }
