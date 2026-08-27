import { formatNumber } from "./format.js";
import { runPreprocessingConsequenceAtlas } from "./preprocessing-consequence-runtime.js";

const COLORS = ["#63c9ad", "#e4ad5c", "#e36d63", "#66a7e4", "#b790da", "#58b9c5", "#d887b0", "#9bb85b"];

export function createPreprocessingConsequenceController({
  root,
  status,
  summary,
  body,
  runButton,
  downloadButton,
  getContext,
  ensureRuntime,
  onResult,
  onDownload,
  onDownloadBinary,
}) {
  let analysis = null;
  let result = null;
  let capture = null;
  let selectedIndex = 0;
  let resizeObserver = null;
  let generation = 0;

  runButton?.addEventListener("click", () => run());
  downloadButton?.addEventListener("click", () => {
    if (result) onDownload?.(result, "preprocessing_consequence_atlas.json");
  });

  function render(explicitAnalysis = null) {
    const next = explicitAnalysis || getContext?.()?.analysis || null;
    const nextIdentity = `${next?.model_sha256 || next?.filename || ""}:${next?.preprocessing_realizability?.portfolio_ledger_sha256 || ""}`;
    const currentIdentity = `${analysis?.model_sha256 || analysis?.filename || ""}:${analysis?.preprocessing_realizability?.portfolio_ledger_sha256 || ""}`;
    if (nextIdentity !== currentIdentity) {
      generation += 1;
      result = null;
      capture = null;
      selectedIndex = 0;
    }
    analysis = next;
    const eligible = String(analysis?.format || "").toLowerCase() === "tflite"
      && Number(analysis?.preprocessing_realizability?.candidate_evaluation_count || 0) > 0;
    if (root) root.hidden = !eligible;
    if (!eligible) return;
    if (runButton) {
      runButton.disabled = !getContext?.()?.modelBytes;
      runButton.textContent = result ? "Replay again" : "Run local replay";
    }
    if (downloadButton) downloadButton.disabled = !result;
    if (result) {
      setStatus(status, "independently verified", "ok");
      renderSummary(summary, result);
      renderResult();
    } else {
      setStatus(status, "runtime replay not run", "watch");
      renderReady();
    }
  }

  async function run() {
    const token = ++generation;
    const context = getContext?.() || {};
    if (!context.analysis || !context.modelBytes) return;
    result = null;
    capture = null;
    if (runButton) {
      runButton.disabled = true;
      runButton.textContent = "Replaying 0 / 9";
    }
    if (downloadButton) downloadButton.disabled = true;
    setStatus(status, "compiling local runtime", "watch");
    renderProgress(0, 9, "Compiling LiteRT.js WASM");
    try {
      const completed = await runPreprocessingConsequenceAtlas({
        analysis: context.analysis,
        modelBytes: context.modelBytes,
        ensureRuntime,
        onProgress: ({ completed: count, total, label }) => {
          if (token !== generation) return;
          if (runButton) runButton.textContent = `Replaying ${count} / ${total}`;
          setStatus(status, `${count} / ${total} local replays`, "watch");
          renderProgress(count, total, label);
        },
      });
      if (token !== generation) return;
      result = completed.evidence;
      capture = completed.capture;
      selectedIndex = 0;
      onResult?.(result);
      render(context.analysis);
    } catch (error) {
      if (token !== generation) return;
      setStatus(status, "replay rejected", "risk");
      body?.replaceChildren(message(`Local replay rejected: ${error.message}`, "risk"));
    } finally {
      if (token === generation && runButton) {
        runButton.disabled = !getContext?.()?.modelBytes;
        runButton.textContent = result ? "Replay again" : "Run local replay";
      }
    }
  }

  function renderReady() {
    resizeObserver?.disconnect();
    summary?.replaceChildren(
      metric("Candidate contracts", formatNumber(analysis.preprocessing_realizability.candidate_evaluation_count), "Explicit finite-domain counterfactuals"),
      metric("Exact tensor aliases", formatNumber(analysis.preprocessing_realizability.exact_tensor_realization_candidate_count), "Expected to collapse to one tensor fingerprint"),
      metric("Runtime", "LiteRT.js WASM", "Browser-local execution"),
      metric("Repetitions", "2 per input", "Byte-identical output required"),
      metric("Evidence", "Not run", "No output consequence is asserted yet"),
    );
    body?.replaceChildren(
      message("No output consequence has been measured in this browser session.", "boundary"),
      definitionTable([
        ["Replay baseline", "The complete constructive model-input tensor witness"],
        ["Candidate inputs", "Each contract's exact or stable minimum-code-error roundtrip tensor"],
        ["Runtime path", "@litertjs/core WASM, one compilation, two byte-compared executions per input"],
        ["Retention budget", "64 MiB maximum for duplicate local output captures"],
      ]),
    );
  }

  function renderProgress(completed, total, label) {
    const progress = element("div", "consequence-progress");
    const rail = element("div", "consequence-progress-rail");
    const fill = element("div", "consequence-progress-fill");
    fill.style.width = `${Math.max(0, Math.min(100, total ? completed / total * 100 : 0))}%`;
    rail.append(fill);
    progress.append(element("strong", "", label), rail, element("span", "", `${completed} / ${total}`));
    body?.replaceChildren(progress);
  }

  function renderResult() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    const row = result.candidates[selectedIndex] || result.candidates[0];
    const candidateCapture = capture?.candidates?.[selectedIndex] || null;
    renderSummary(summary, result);
    const toolbar = element("div", "consequence-toolbar");
    const label = element("label", "consequence-select");
    label.append(element("span", "", "Contract replay"));
    const select = element("select");
    result.candidates.forEach((candidate, index) => {
      const option = new Option(`${candidate.output_identical_to_witness_replay ? "SAME" : formatNumber(candidate.output_changed_element_count)} | ${candidate.contract_label}`, String(index));
      option.selected = index === selectedIndex;
      select.append(option);
    });
    select.addEventListener("change", () => {
      selectedIndex = Number(select.value);
      renderResult();
    });
    label.append(select);
    const actions = element("div", "consequence-actions");
    actions.append(
      commandButton("Candidate JSON", "Download the selected measured replay ledger", () => onDownload?.(row, `preprocessing_consequence_${row.contract_id}.json`)),
      commandButton("Output tensor", "Download the selected first output tensor bytes", () => {
        const output = candidateCapture?.outputs?.[0];
        if (output) onDownloadBinary?.(new Uint8Array(output.buffer, output.byteOffset, output.byteLength), `preprocessing_consequence_${row.contract_id}_output0.bin`, "application/octet-stream");
      }, !candidateCapture?.outputs?.[0]),
    );
    toolbar.append(label, actions);

    const visual = element("section", "consequence-band visual");
    visual.append(sectionHead("Contract fingerprint flow", `${result.unique_input_tensor_count} input classes / ${result.unique_output_tensor_set_count} output classes`));
    const canvas = element("canvas", "consequence-canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Preprocessing contracts flowing through unique input and output fingerprint classes with selected output difference spectrum");
    visual.append(canvas);

    const certificate = element("section", "consequence-band");
    certificate.append(sectionHead("Selected replay certificate", row.contract_label), definitionTable([
      ["Input tensor", `${row.input_changed_element_count} changed / total |code delta| ${row.input_total_absolute_code_difference_decimal} / SHA-256 ${row.input_tensor_sha256}`],
      ["Output tensor set", `${formatNumber(row.output_changed_element_count)} changed of ${formatNumber(row.output_total_element_count)} / maximum |delta| ${formatNumber(row.output_maximum_absolute_difference)}`],
      ["Output digest", row.output_tensor_set_sha256],
      ["First-output top-1", `${row.baseline_first_output_top1_index} -> ${row.first_output_top1_index}${row.first_output_top1_changed ? " (changed)" : " (stable)"}`],
      ["Replay determinism", row.deterministic_replay ? "2 / 2 byte-identical" : "rejected"],
      ["Candidate ledger SHA-256", row.candidate_ledger_sha256],
    ]));

    const matrix = element("section", "consequence-band table-band");
    matrix.append(sectionHead("Counterfactual consequence matrix", result.exact_contract_output_conservation ? "Exact source contracts conserve the tensor-ABI replay" : "Exact-contract conservation failed"), table(
      ["Contract", "Input class", "Input changed", "Output class", "Output changed", "Max |delta|", "Top-1", "Replay"],
      result.candidates.map((candidate, index) => [
        selectButton(candidate.contract_label, index, () => { selectedIndex = index; renderResult(); }),
        classFor(result.input_equivalence_classes, candidate.contract_id),
        formatNumber(candidate.input_changed_element_count),
        classFor(result.output_equivalence_classes, candidate.contract_id),
        formatNumber(candidate.output_changed_element_count),
        formatNumber(candidate.output_maximum_absolute_difference),
        candidate.first_output_top1_changed ? `${candidate.baseline_first_output_top1_index} -> ${candidate.first_output_top1_index}` : String(candidate.first_output_top1_index),
        candidate.deterministic_replay ? "2/2" : "rejected",
      ]),
    ));

    const classes = element("section", "consequence-band class-grid");
    classes.append(
      classList("Input equivalence classes", result.input_equivalence_classes),
      classList("Output equivalence classes", result.output_equivalence_classes),
    );
    const mismatch = row.first_output_difference
      ? message(`First output difference: output ${row.first_output_difference.output_index}, linear ${formatNumber(row.first_output_difference.linear_index)}, baseline ${row.first_output_difference.baseline_value}, candidate ${row.first_output_difference.candidate_value}.`, row.first_output_top1_changed ? "risk" : "watch")
      : message("The complete captured output tensor set is byte-identical to the tensor-ABI witness replay.", "exact");
    body?.replaceChildren(toolbar, visual, certificate, matrix, classes, mismatch, message(result.interpretation_boundary, "boundary"));
    requestAnimationFrame(() => drawPreprocessingConsequence(canvas, result, capture, selectedIndex));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawPreprocessingConsequence(canvas, result, capture, selectedIndex));
      resizeObserver.observe(body);
    }
  }

  return { render, run, getResult: () => result, getCapture: () => capture };
}

export function drawPreprocessingConsequence(canvas, result, capture, selectedIndex = 0) {
  if (!canvas || !result) return;
  const compact = Number(canvas.parentElement?.clientWidth || 0) > 0 && Number(canvas.parentElement.clientWidth) < 640;
  const width = compact ? 390 : 1320;
  const height = compact ? 1040 : 690;
  prepareCanvas(canvas, width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#0d1820";
  context.fillRect(0, 0, width, height);
  if (compact) {
    drawFlow(context, result, 24, 54, 342, 560, true);
    drawDeltaSpectrum(context, result, capture, selectedIndex, 24, 700, 342, 250);
  } else {
    drawFlow(context, result, 38, 54, 790, 570, false);
    drawDeltaSpectrum(context, result, capture, selectedIndex, 880, 92, 398, 465);
  }
}

export function renderPreprocessingConsequenceCanvas(result, capture, filename = "") {
  if (!result || typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = 2360;
  canvas.height = 1540;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0b151d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#eaf4f5";
  context.font = "700 58px Inter, system-ui, sans-serif";
  context.fillText("Preprocessing Consequence Atlas", 92, 108);
  context.fillStyle = "#8fa4b0";
  context.font = "25px Inter, system-ui, sans-serif";
  context.fillText(filename || "TFLite artifact / local LiteRT.js WASM replay", 94, 154);
  const metrics = [
    ["CONTRACTS", result.candidate_count],
    ["INPUT CLASSES", result.unique_input_tensor_count],
    ["OUTPUT CLASSES", result.unique_output_tensor_set_count],
    ["OUTPUT CHANGED", result.output_changed_candidate_count],
    ["TOP-1 CHANGED", result.top1_changed_candidate_count],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 94 + index * 445;
    context.fillStyle = "#829aa7";
    context.font = "18px Inter, system-ui, sans-serif";
    context.fillText(label, x, 225);
    context.fillStyle = index >= 3 && Number(value) ? "#efb26b" : "#e5f0f1";
    context.font = "700 34px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.fillText(String(value), x, 270);
  });
  drawFlow(context, result, 92, 350, 1390, 990, false);
  drawDeltaSpectrum(context, result, capture, result.candidates.findIndex((row) => row.contract_id === result.most_output_sensitive_contract_id), 1570, 430, 690, 720);
  context.fillStyle = "#687f8b";
  context.font = "20px Inter, system-ui, sans-serif";
  context.fillText(`portfolio ${result.portfolio_ledger_sha256}`, 94, 1460, 2160);
  return canvas;
}

function drawFlow(context, result, x, y, width, height, compact) {
  const candidateX = x + 8;
  const inputX = x + width * (compact ? 0.54 : 0.51);
  const outputX = x + width - 20;
  context.fillStyle = "#dce9eb";
  context.font = "700 18px Inter, system-ui, sans-serif";
  context.fillText("CONTRACT", candidateX, y - 18);
  context.textAlign = "center";
  context.fillText(compact ? "INPUT" : "INPUT SHA", inputX, y - 18);
  context.textAlign = "right";
  context.fillText(compact ? "OUTPUT" : "OUTPUT SHA", outputX, y - 18);
  context.textAlign = "left";
  const top = y + 26;
  const rowGap = (height - 56) / Math.max(1, result.candidates.length - 1);
  const inputPositions = classPositions(result.input_equivalence_classes, top, height - 30);
  const outputPositions = classPositions(result.output_equivalence_classes, top, height - 30);
  result.candidates.forEach((row, index) => {
    const rowY = top + index * rowGap;
    const inClass = result.input_equivalence_classes.find((item) => item.contract_ids.includes(row.contract_id));
    const outClass = result.output_equivalence_classes.find((item) => item.contract_ids.includes(row.contract_id));
    const color = row.first_output_top1_changed ? "#e36d63" : row.output_identical_to_witness_replay ? "#63c9ad" : COLORS[index % COLORS.length];
    context.strokeStyle = color;
    context.globalAlpha = 0.66;
    context.lineWidth = row.source_exact_tensor_realization ? 3 : 1.8;
    bezier(context, candidateX + (compact ? 104 : 210), rowY, inputX - 14, inputPositions.get(inClass.class_id));
    bezier(context, inputX + 14, inputPositions.get(inClass.class_id), outputX - 92, outputPositions.get(outClass.class_id));
    context.globalAlpha = 1;
    context.fillStyle = color;
    context.beginPath();
    context.arc(candidateX + 2, rowY, 5, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#d7e4e6";
    context.font = `${row.contract_id === result.most_output_sensitive_contract_id ? "700 " : ""}${compact ? 10 : 12}px Inter, system-ui, sans-serif`;
    context.fillText(compact ? shortContract(row.contract_id) : row.contract_label, candidateX + 13, rowY + 4, compact ? 94 : 196);
  });
  drawClassNodes(context, result.input_equivalence_classes, inputPositions, inputX, "I");
  drawClassNodes(context, result.output_equivalence_classes, outputPositions, outputX - 16, "O", true);
}

function drawDeltaSpectrum(context, result, capture, selectedIndex, x, y, width, height) {
  const row = result.candidates[Math.max(0, selectedIndex)] || result.candidates[0];
  const baseline = capture?.baseline?.outputs?.[0];
  const candidate = capture?.candidates?.[Math.max(0, selectedIndex)]?.outputs?.[0];
  context.fillStyle = "#dce9eb";
  context.font = "700 18px Inter, system-ui, sans-serif";
  context.fillText("FIRST-OUTPUT DELTA SPECTRUM", x, y - 24, width);
  context.fillStyle = "#91a6b1";
  context.font = "13px Inter, system-ui, sans-serif";
  context.fillText(shortContract(row.contract_id), x, y, width);
  const plotY = y + 34;
  const plotHeight = height - 94;
  context.fillStyle = "#14242e";
  context.fillRect(x, plotY, width, plotHeight);
  if (!baseline || !candidate || baseline.length !== candidate.length) {
    context.fillStyle = "#758d99";
    context.fillText("Captured output bytes unavailable", x + 18, plotY + 35);
    return;
  }
  const bins = Math.min(72, Math.max(1, baseline.length));
  const values = Array.from({ length: bins }, () => ({ changed: 0, maximum: 0 }));
  for (let index = 0; index < baseline.length; index += 1) {
    const bin = Math.min(bins - 1, Math.floor(index / baseline.length * bins));
    const delta = Math.abs(Number(candidate[index]) - Number(baseline[index]));
    if (!Object.is(Number(candidate[index]), Number(baseline[index]))) values[bin].changed += 1;
    values[bin].maximum = Math.max(values[bin].maximum, Number.isFinite(delta) ? delta : 0);
  }
  const maximumChanged = Math.max(1, ...values.map((item) => item.changed));
  const slot = width / bins;
  values.forEach((item, index) => {
    const barHeight = item.changed / maximumChanged * (plotHeight - 34);
    context.fillStyle = item.changed ? "#e4ad5c" : "#28404c";
    context.fillRect(x + index * slot + 1, plotY + plotHeight - barHeight - 18, Math.max(1, slot - 2), barHeight);
  });
  context.fillStyle = row.first_output_top1_changed ? "#e36d63" : "#9bb0ba";
  context.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText(`${formatNumber(row.output_changed_element_count)} changed / max |delta| ${formatNumber(row.output_maximum_absolute_difference)}`, x + 10, plotY + plotHeight - 4, width - 20);
}

function classPositions(classes, top, span) {
  const positions = new Map();
  classes.forEach((item, index) => positions.set(item.class_id, top + (index + 0.5) * span / classes.length));
  return positions;
}

function drawClassNodes(context, classes, positions, x, prefix, right = false) {
  classes.forEach((item, index) => {
    const y = positions.get(item.class_id);
    context.fillStyle = COLORS[index % COLORS.length];
    context.beginPath();
    context.arc(x, y, 11, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#10202a";
    context.font = "700 9px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.textAlign = "center";
    context.fillText(`${prefix}${index}`, x, y + 3);
    context.fillStyle = "#9eb1ba";
    context.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.textAlign = right ? "right" : "left";
    context.fillText(`${item.candidate_count}x ${item.sha256.slice(0, 8)}`, right ? x - 16 : x + 16, y + 4);
  });
  context.textAlign = "left";
}

function bezier(context, x0, y0, x1, y1) {
  const mid = (x0 + x1) / 2;
  context.beginPath();
  context.moveTo(x0, y0);
  context.bezierCurveTo(mid, y0, mid, y1, x1, y1);
  context.stroke();
}

function renderSummary(root, result) {
  root?.replaceChildren(
    metric("Contract replays", formatNumber(result.candidate_count), "Two byte-compared executions per input"),
    metric("Unique input tensors", formatNumber(result.unique_input_tensor_count), `${result.exact_source_contract_count} exact-source contract aliases`),
    metric("Unique outputs", formatNumber(result.unique_output_tensor_set_count), "Whole output-tensor-set SHA-256 classes"),
    metric("Output changed", formatNumber(result.output_changed_candidate_count), `${result.non_exact_output_changed_candidate_count} non-exact candidate replays`),
    metric("Top-1 changed", formatNumber(result.top1_changed_candidate_count), `Raw first-output rank only; labels ${result.top1_changed_candidate_count ? "not embedded" : "not required for equality"}`),
  );
}

function classFor(classes, contractId) {
  const index = classes.findIndex((item) => item.contract_ids.includes(contractId));
  return index >= 0 ? `${classes[index].class_id} (${classes[index].candidate_count}x)` : "-";
}

function classList(title, classes) {
  const section = element("div", "consequence-class-list");
  section.append(element("strong", "", title));
  classes.forEach((item) => section.append(element("div", "consequence-class-row", `${item.class_id} | ${item.contract_ids.map(shortContract).join(" / ")} | ${item.sha256.slice(0, 16)}`)));
  return section;
}

function shortContract(value) {
  return String(value || "").replace("_rgb", "").replaceAll("_", " ");
}

function metric(label, value, detail) {
  const card = element("div", "consequence-metric");
  card.append(element("span", "", label), element("strong", "", value), element("small", "", detail));
  return card;
}

function definitionTable(rows) {
  return table(["Field", "Value"], rows);
}

function table(headers, rows) {
  const wrap = element("div", "consequence-table-wrap");
  const node = element("table", "consequence-table");
  const head = element("thead");
  const headRow = element("tr");
  headers.forEach((header) => headRow.append(element("th", "", header)));
  head.append(headRow);
  const body = element("tbody");
  rows.forEach((values) => {
    const row = element("tr");
    values.forEach((value) => {
      const cell = element("td");
      if (value instanceof Node) cell.append(value);
      else cell.textContent = String(value ?? "-");
      row.append(cell);
    });
    body.append(row);
  });
  node.append(head, body);
  wrap.append(node);
  return wrap;
}

function sectionHead(title, detail) {
  const head = element("div", "consequence-band-head");
  head.append(element("h4", "", title), element("span", "", detail));
  return head;
}

function selectButton(label, index, handler) {
  const button = element("button", "consequence-row-select", label);
  button.type = "button";
  button.dataset.index = String(index);
  button.addEventListener("click", handler);
  return button;
}

function commandButton(label, title, handler, disabled = false) {
  const button = element("button", "secondary-action", label);
  button.type = "button";
  button.title = title;
  button.disabled = disabled;
  button.addEventListener("click", handler);
  return button;
}

function message(text, tone = "") {
  return element("p", `consequence-message ${tone}`.trim(), text);
}

function prepareCanvas(canvas, width, height) {
  const ratio = Math.min(2, globalThis.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.aspectRatio = `${width} / ${height}`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function setStatus(node, text, tone) {
  if (!node) return;
  node.textContent = text;
  node.dataset.tone = tone;
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}
