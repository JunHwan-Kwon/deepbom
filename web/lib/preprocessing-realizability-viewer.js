import { formatNumber, padOp } from "./format.js";
import {
  buildCandidateRgbFixture,
  validatePreprocessingRealizabilityAnalysis,
} from "./preprocessing-realizability.js";

const CHANNEL_COLORS = ["#e86f61", "#59c49f", "#5b9fe3"];

export function createPreprocessingRealizabilityController({
  root,
  status,
  summary,
  body,
  downloadButton,
  getAnalysis,
  jumpToGraphOp,
  onDownload,
  onDownloadBinary,
}) {
  let analysis = null;
  let evidence = null;
  let verified = null;
  let selectedIndex = 0;
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "preprocessing_realizability.json");
  });

  function render(explicitAnalysis = null) {
    const token = ++renderToken;
    resizeObserver?.disconnect();
    resizeObserver = null;
    analysis = explicitAnalysis || getAnalysis?.() || null;
    evidence = analysis?.preprocessing_realizability || null;
    verified = null;
    selectedIndex = 0;
    if (!evidence || String(analysis?.format || "").toLowerCase() !== "tflite") {
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    renderSummary(summary, evidence);
    renderBody();
    setStatus(status, evidence.candidates?.length ? "LUT / fixture verification pending" : humanize(evidence.status), evidence.candidates?.length ? "watch" : "ok");
    if (!evidence.candidates?.length) return;
    validatePreprocessingRealizabilityAnalysis(analysis).then((result) => {
      if (token !== renderToken) return;
      verified = result;
      renderBody();
      setStatus(status, "independently verified", "ok");
    }).catch((error) => {
      if (token !== renderToken) return;
      body?.replaceChildren(message(`Preprocessing evidence rejected: ${error.message}`, "risk"));
      setStatus(status, "evidence rejected", "risk");
    });
  }

  function renderBody() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (!body || !evidence) return;
    if (!evidence.candidates?.length) {
      body.replaceChildren(message("No eligible constructive NHWC RGB input witness is available."));
      return;
    }
    selectedIndex = Math.min(selectedIndex, evidence.candidates.length - 1);
    const candidate = evidence.candidates[selectedIndex];
    const witness = analysis.input_counterexample.witnesses[candidate.witness_index];
    const fixture = verified?.candidates?.[selectedIndex]?.fixture || safeFixture(candidate, witness);

    const toolbar = element("div", "preprocess-lab-toolbar");
    const selectorLabel = element("label", "preprocess-contract-select");
    selectorLabel.append(element("span", "", "Candidate contract"));
    const selector = element("select");
    evidence.candidates.forEach((row, index) => {
      const option = new Option(`${row.exact_tensor_realization ? "EXACT" : formatNumber(row.unrealizable_tensor_element_count)} | ${row.contract_label}`, String(index));
      option.selected = index === selectedIndex;
      selector.append(option);
    });
    selector.addEventListener("change", () => {
      selectedIndex = Number(selector.value);
      renderBody();
    });
    selectorLabel.append(selector);
    const actions = element("div", "preprocess-lab-actions");
    const candidateButton = commandButton("Candidate JSON", "Download the selected counterfactual contract ledger", () => onDownload?.(candidate, `preprocessing_${candidate.contract_id}.json`));
    const pngButton = commandButton("RGB fixture PNG", "Download the exact or minimum-code-error RGB source fixture", () => {
      const bytes = verified?.candidates?.[selectedIndex]?.fixture?.png;
      if (bytes) onDownloadBinary?.(bytes, `preprocessing_${candidate.contract_id}_${candidate.fixture_kind}.png`, "image/png");
    });
    const rgbButton = commandButton("RGB bytes", "Download the RGB8 source raster without a container", () => {
      const bytes = verified?.candidates?.[selectedIndex]?.fixture?.rgb;
      if (bytes) onDownloadBinary?.(bytes, `preprocessing_${candidate.contract_id}_${candidate.fixture_kind}.rgb`, "application/octet-stream");
    });
    pngButton.disabled = !verified;
    rgbButton.disabled = !verified;
    actions.append(candidateButton, pngButton, rgbButton, commandButton("Graph source", "Open the certified first-layer operator", () => jumpToGraphOp?.(candidate.source_op_index)));
    toolbar.append(selectorLabel, actions);

    const visual = element("section", "preprocess-lab-band visual");
    visual.append(sectionHead("Pixel-code transfer and fixture", `${candidate.tensor_channel_order} tensor / RGB source / ${candidate.fixture_kind.replaceAll("_", " ")}`));
    const canvas = element("canvas", "preprocess-lab-canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Pixel to tensor code transfer curves and reconstructed RGB fixture");
    visual.append(canvas, channelLegend(candidate));

    const contract = element("section", "preprocess-lab-band");
    contract.append(
      sectionHead("Finite-domain certificate", candidate.contract_label),
      definitionTable([
        ["Candidate status", candidate.exact_tensor_realization ? "EXACT COMPLETE TENSOR REALIZATION" : "NON-EXACT COUNTERFACTUAL"],
        ["Pixel-to-real", candidate.pixel_to_real_formula],
        ["Real-to-tensor", candidate.real_to_tensor_formula],
        ["Tensor elements", `${formatNumber(candidate.exact_tensor_element_count)} exact / ${formatNumber(candidate.unrealizable_tensor_element_count)} unrealizable / ${formatNumber(candidate.witness_tensor_element_count)} total`],
        ["Minimum code error", `${formatNumber(candidate.minimum_total_absolute_tensor_code_error_decimal)} total / ${formatNumber(candidate.maximum_absolute_tensor_code_error)} maximum`],
        ["Witness code pairs", `${candidate.exact_witness_channel_code_pair_count} exact / ${candidate.distinct_witness_channel_code_pair_count} distinct channel-code pairs`],
        ["Fixture RGB SHA-256", candidate.nearest_rgb_fixture_sha256],
        ["Candidate ledger SHA-256", candidate.candidate_ledger_sha256],
      ]),
    );

    const codeRows = element("section", "preprocess-lab-band table-band");
    codeRows.append(
      sectionHead("Witness inverse pixels", `${candidate.witness_code_realizations.length} channel-code pairs`),
      table(["Tensor ch", "Source ch", "Target code", "Elements", "Exact pixels", "Selected", "Roundtrip", "|error|"], candidate.witness_code_realizations.map((row) => [
        channelName(row.tensor_channel),
        channelName(row.source_pixel_channel),
        row.target_tensor_code,
        formatNumber(row.tensor_element_count),
        row.exact_source_pixel_codes.length ? row.exact_source_pixel_codes.join(", ") : "none",
        row.selected_source_pixel_code,
        row.roundtrip_tensor_code,
        row.absolute_tensor_code_error,
      ])),
    );
    if (String(candidate.contract_id || "").startsWith("raw_storage")
      && String(witness?.model_input_dtype || "").toUpperCase() === "INT8") {
      codeRows.append(message("Raw-storage witness note: negative signed INT8 tensor codes are represented by their two's-complement UINT8 source bytes (for example, -128 -> 128 and -92 -> 164); non-negative codes retain the same byte value.", "boundary"));
    }

    const matrix = element("section", "preprocess-lab-band table-band");
    matrix.append(
      sectionHead("Candidate matrix", evidence.candidate_conservation),
      table(["Contract", "Result", "Exact elements", "Unrealizable", "Total |error|", "Code coverage R/G/B", "Fixture SHA"], evidence.candidates.map((row, index) => [
        selectButton(row.contract_label, index, () => {
          selectedIndex = index;
          renderBody();
        }),
        row.exact_tensor_realization ? "EXACT" : "NON-EXACT",
        formatNumber(row.exact_tensor_element_count),
        formatNumber(row.unrealizable_tensor_element_count),
        formatNumber(row.minimum_total_absolute_tensor_code_error_decimal),
        row.channel_maps.map((map) => `${map.reachable_tensor_code_count}/256`).join(" / "),
        row.nearest_rgb_fixture_sha256.slice(0, 16),
      ])),
    );
    const mismatch = candidate.first_unrealizable_element
      ? message(`First finite-domain mismatch: tensor [${candidate.first_unrealizable_element.tensor_coordinate_nhwc.join(", ")}] target ${candidate.first_unrealizable_element.target_tensor_code}; source pixel ${candidate.first_unrealizable_element.selected_source_pixel_code} round-trips to ${candidate.first_unrealizable_element.roundtrip_tensor_code}.`, "watch")
      : message(`Complete RGB8 source raster exists for op #${padOp(candidate.source_op_index)} under this exact candidate contract.`, "exact");
    body.replaceChildren(toolbar, visual, contract, codeRows, matrix, mismatch, message(evidence.interpretation_boundary, "boundary"));
    requestAnimationFrame(() => drawPreprocessingRealizability(canvas, candidate, witness, fixture));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawPreprocessingRealizability(canvas, candidate, witness, fixture));
      resizeObserver.observe(body);
    }
  }

  return { render };
}

export function drawPreprocessingRealizability(canvas, candidate, witness, fixture) {
  if (!canvas || !candidate || !witness) return;
  const compact = Number(canvas.parentElement?.clientWidth || 0) > 0 && Number(canvas.parentElement.clientWidth) < 620;
  const width = compact ? 390 : 1320;
  const height = compact ? 850 : 620;
  prepareCanvas(canvas, width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#0e1820";
  context.fillRect(0, 0, width, height);
  if (compact) {
    drawTransfer(context, candidate, witness, 34, 78, 322, 300);
    drawFixture(context, fixture, 75, 465, 240, 240);
    drawOutcome(context, candidate, 28, 760, 334);
  } else {
    drawTransfer(context, candidate, witness, 58, 90, 710, 430);
    drawFixture(context, fixture, 894, 112, 320, 320);
    drawOutcome(context, candidate, 836, 500, 420);
  }
}

export function renderPreprocessingRealizabilityCanvas(analysis, filename = "") {
  const evidence = analysis?.preprocessing_realizability;
  const candidate = evidence?.candidates?.[0];
  const witness = analysis?.input_counterexample?.witnesses?.[candidate?.witness_index];
  if (!candidate || !witness || typeof document === "undefined") return null;
  const fixture = safeFixture(candidate, witness);
  const canvas = document.createElement("canvas");
  canvas.width = 2360;
  canvas.height = 1600;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0c161e";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#edf6f6";
  context.font = "700 58px Inter, system-ui, sans-serif";
  context.fillText("Pixel-to-Tensor Contract Lab", 92, 108);
  context.fillStyle = "#8fa4b0";
  context.font = "25px Inter, system-ui, sans-serif";
  context.fillText(filename || analysis.filename || "TFLite artifact", 94, 154);
  const metrics = [
    ["CANDIDATES", evidence.candidate_evaluation_count],
    ["EXACT", evidence.exact_tensor_realization_candidate_count],
    ["NON-EXACT", evidence.non_exact_candidate_count],
    ["INPUT ELEMENTS", formatNumber(candidate.witness_tensor_element_count)],
    ["PORTFOLIO", evidence.portfolio_ledger_sha256.slice(0, 16)],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 94 + index * 445;
    context.fillStyle = "#829aa7";
    context.font = "18px Inter, system-ui, sans-serif";
    context.fillText(label, x, 225);
    context.fillStyle = index === 1 ? "#64d1b4" : "#e5f0f1";
    context.font = "700 34px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.fillText(String(value), x, 270);
  });
  drawTransfer(context, candidate, witness, 110, 385, 1260, 790);
  drawFixture(context, fixture, 1650, 425, 520, 520);
  drawOutcome(context, candidate, 1510, 1080, 730);
  context.fillStyle = "#627783";
  context.font = "19px Inter, system-ui, sans-serif";
  context.fillText("Explicit preprocessing counterfactuals; production decoder, resize, channel order, and normalization remain external evidence.", 96, 1535);
  return canvas;
}

function drawTransfer(context, candidate, witness, x, y, width, height) {
  const plotLeft = x + 52;
  const plotTop = y + 28;
  const plotWidth = width - 72;
  const plotHeight = height - 74;
  const [qmin, qmax] = witness.model_input_code_range;
  context.fillStyle = "#dce9eb";
  context.font = "700 18px Inter, system-ui, sans-serif";
  context.fillText("256-code transfer", x, y - 12);
  context.strokeStyle = "#2b3b46";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const py = plotTop + tick * plotHeight / 4;
    context.beginPath();
    context.moveTo(plotLeft, py);
    context.lineTo(plotLeft + plotWidth, py);
    context.stroke();
  }
  candidate.channel_maps.forEach((map, channel) => {
    context.strokeStyle = CHANNEL_COLORS[channel];
    context.lineWidth = 2.5;
    context.setLineDash(channel === 0 ? [18, 8] : channel === 1 ? [10, 6] : [3, 6]);
    context.beginPath();
    map.pixel_to_tensor_codes.forEach((code, pixel) => {
      const px = plotLeft + pixel / 255 * plotWidth;
      const py = plotTop + (qmax - code) / (qmax - qmin) * plotHeight;
      if (pixel === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.stroke();
  });
  context.setLineDash([]);
  context.fillStyle = "#8299a6";
  context.font = "13px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText("pixel 0", plotLeft, plotTop + plotHeight + 27);
  context.textAlign = "right";
  context.fillText("pixel 255", plotLeft + plotWidth, plotTop + plotHeight + 27);
  context.fillText(String(qmax), plotLeft - 12, plotTop + 5);
  context.fillText(String(qmin), plotLeft - 12, plotTop + plotHeight + 5);
  context.textAlign = "left";
}

function drawFixture(context, fixture, x, y, width, height) {
  context.fillStyle = "#dce9eb";
  context.font = "700 18px Inter, system-ui, sans-serif";
  context.fillText("RGB source fixture", x, y - 22);
  if (!fixture?.rgb) {
    context.fillStyle = "#263640";
    context.fillRect(x, y, width, height);
    return;
  }
  const source = document.createElement("canvas");
  source.width = fixture.width;
  source.height = fixture.height;
  const sourceContext = source.getContext("2d");
  const rgba = new Uint8ClampedArray(fixture.width * fixture.height * 4);
  for (let pixel = 0; pixel < fixture.width * fixture.height; pixel += 1) {
    rgba[pixel * 4] = fixture.rgb[pixel * 3];
    rgba[pixel * 4 + 1] = fixture.rgb[pixel * 3 + 1];
    rgba[pixel * 4 + 2] = fixture.rgb[pixel * 3 + 2];
    rgba[pixel * 4 + 3] = 255;
  }
  sourceContext.putImageData(new ImageData(rgba, fixture.width, fixture.height), 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(source, x, y, width, height);
  context.strokeStyle = "#5f7682";
  context.strokeRect(x, y, width, height);
  const inset = Math.max(82, Math.min(132, width * 0.38));
  const insetX = x + width - inset - 10;
  const insetY = y + height - inset - 10;
  context.fillStyle = "#0a1218";
  context.fillRect(insetX - 5, insetY - 24, inset + 10, inset + 29);
  context.imageSmoothingEnabled = false;
  context.drawImage(source, 0, 0, 3, 3, insetX, insetY, inset, inset);
  context.strokeStyle = "#d3a95f";
  context.lineWidth = 2;
  context.strokeRect(insetX, insetY, inset, inset);
  context.fillStyle = "#d8e4e5";
  context.font = "11px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText("3x3 origin", insetX, insetY - 8);
}

function drawOutcome(context, candidate, x, y, width) {
  context.fillStyle = candidate.exact_tensor_realization ? "#64d1b4" : "#efb86b";
  context.font = "700 20px Inter, system-ui, sans-serif";
  context.fillText(candidate.exact_tensor_realization ? "EXACT COMPLETE REALIZATION" : "NON-EXACT COUNTERFACTUAL", x, y);
  context.fillStyle = "#94a9b4";
  context.font = "14px ui-monospace, SFMono-Regular, Consolas, monospace";
  const lines = [
    `${formatNumber(candidate.exact_tensor_element_count)} exact / ${formatNumber(candidate.unrealizable_tensor_element_count)} unrealizable`,
    `minimum total |code error| ${formatNumber(candidate.minimum_total_absolute_tensor_code_error_decimal)}`,
    `RGB SHA-256 ${candidate.nearest_rgb_fixture_sha256.slice(0, Math.max(16, Math.floor(width / 12)))}...`,
  ];
  lines.forEach((line, index) => context.fillText(line, x, y + 35 + index * 25));
}

function renderSummary(root, evidence) {
  root?.replaceChildren(
    metric("Candidate contracts", formatNumber(evidence.candidate_evaluation_count), evidence.candidate_conservation),
    metric("Exact RGB realizations", formatNumber(evidence.exact_tensor_realization_candidate_count), evidence.exact_contract_ids.map(humanize).join(" / ")),
    metric("Non-exact contracts", formatNumber(evidence.non_exact_candidate_count), "finite 256-code source domain"),
    metric("Best non-exact", evidence.best_non_exact_contract_id ? humanize(evidence.best_non_exact_contract_id) : "none", evidence.best_non_exact_unrealizable_element_count == null ? "not applicable" : `${formatNumber(evidence.best_non_exact_unrealizable_element_count)} unrealizable elements`),
    metric("Eligible witnesses", `${evidence.eligible_image_witness_count} / ${evidence.source_witness_count}`, `${evidence.ineligible_witness_count} ineligible image witnesses`),
  );
}

function safeFixture(candidate, witness) {
  try {
    return buildCandidateRgbFixture(candidate, witness);
  } catch {
    return null;
  }
}

function channelLegend(candidate) {
  const node = element("div", "preprocess-lab-legend");
  candidate.channel_maps.forEach((map, index) => {
    const item = element("span");
    const swatch = element("i");
    swatch.style.background = CHANNEL_COLORS[index];
    item.append(swatch, document.createTextNode(`tensor ${channelName(map.tensor_channel)} -> source ${channelName(map.source_pixel_channel)} / ${map.reachable_tensor_code_count} codes / ${map.tensor_code_hole_count} holes`));
    node.append(item);
  });
  return node;
}

function definitionTable(rows) {
  const list = element("dl", "preprocess-lab-definition");
  rows.forEach(([label, value]) => list.append(element("dt", "", label), element("dd", "", String(value))));
  return list;
}

function table(headers, rows) {
  const wrap = element("div", "preprocess-lab-table-wrap");
  const node = element("table", "preprocess-lab-table");
  const head = element("thead");
  const headRow = element("tr");
  headers.forEach((header) => headRow.append(element("th", "", header)));
  head.append(headRow);
  const tbody = element("tbody");
  rows.forEach((row) => {
    const tr = element("tr");
    row.forEach((value) => {
      const td = element("td");
      td.append(value instanceof Node ? value : document.createTextNode(String(value)));
      tr.append(td);
    });
    tbody.append(tr);
  });
  node.append(head, tbody);
  wrap.append(node);
  return wrap;
}

function selectButton(label, index, handler) {
  const button = element("button", "preprocess-contract-row", label);
  button.type = "button";
  button.dataset.candidateIndex = String(index);
  button.addEventListener("click", handler);
  return button;
}

function metric(label, value, detail) {
  const node = element("div", "preprocess-lab-metric");
  node.append(element("span", "label", label), element("strong", "", String(value)), element("small", "", detail));
  return node;
}

function sectionHead(title, detail) {
  const head = element("div", "preprocess-lab-section-head");
  head.append(element("h4", "", title), element("span", "", detail));
  return head;
}

function commandButton(label, title, handler) {
  const button = element("button", "secondary-action", label);
  button.type = "button";
  button.title = title;
  button.addEventListener("click", handler);
  return button;
}

function message(text, tone = "") {
  return element("p", `preprocess-lab-message ${tone}`.trim(), text);
}

function setStatus(node, text, tone) {
  if (!node) return;
  node.textContent = text;
  node.dataset.tone = tone;
}

function prepareCanvas(canvas, width, height) {
  const ratio = Math.max(1, Math.min(2, globalThis.devicePixelRatio || 1));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.aspectRatio = `${width} / ${height}`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function channelName(index) {
  return ["R", "G", "B"][index] || String(index);
}

function humanize(value) {
  return String(value || "not assessed").replaceAll("_", " ");
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}
