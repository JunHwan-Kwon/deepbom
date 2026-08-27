import { formatNumber, padOp } from "./format.js";
import {
  validateInputCounterexampleAnalysis,
  validateInputCounterexampleShape,
} from "./input-counterexample.js";

export function createInputCounterexampleController({
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
  let renderToken = 0;
  let resizeObserver = null;

  downloadButton?.addEventListener("click", () => {
    if (evidence) onDownload?.(evidence, "input_counterexample.json");
  });

  function render(explicitAnalysis = null) {
    const token = ++renderToken;
    resizeObserver?.disconnect();
    resizeObserver = null;
    analysis = explicitAnalysis || getAnalysis?.() || null;
    evidence = analysis?.input_counterexample || null;
    verified = null;
    if (!evidence || String(analysis?.format || "").toLowerCase() !== "tflite") {
      if (root) root.hidden = true;
      if (downloadButton) downloadButton.disabled = true;
      return;
    }
    if (root) root.hidden = false;
    if (downloadButton) downloadButton.disabled = false;
    try {
      validateInputCounterexampleShape(evidence);
      renderSummary(summary, evidence);
      renderBody();
      setStatus(status, evidence.witnesses.length ? "tensor reconstruction / arithmetic verification pending" : humanize(evidence.status), evidence.witnesses.length ? "watch" : "ok");
      if (evidence.witnesses.length) {
        validateInputCounterexampleAnalysis(analysis).then((result) => {
          if (token !== renderToken) return;
          verified = result;
          renderBody();
          setStatus(status, "independently verified", "ok");
        }).catch((error) => {
          if (token !== renderToken) return;
          body?.replaceChildren(message(`Input witness evidence rejected: ${error.message}`, "risk"));
          setStatus(status, "evidence rejected", "risk");
        });
      }
    } catch (error) {
      summary?.replaceChildren();
      body?.replaceChildren(message(`Input witness evidence rejected: ${error.message}`, "risk"));
      setStatus(status, "evidence rejected", "risk");
    }
  }

  function renderBody() {
    if (!body || !evidence) return;
    resizeObserver?.disconnect();
    resizeObserver = null;
    const witness = evidence.witnesses[0];
    if (!witness) {
      body.replaceChildren(message(evidence.status === "not_applicable"
        ? "No exact-local source required a model-input realizability assessment."
        : "No complete model-input tensor witness was constructed."));
      return;
    }
    const reconstructed = verified?.witnesses?.[0] || null;
    const actions = element("div", "input-witness-actions");
    const witnessButton = commandButton("Input witness JSON", "Download the constructive input witness and arithmetic ledger", () => onDownload?.(witness, `input_witness_op_${witness.source_op_index}.json`));
    const rawButton = commandButton("Input tensor", "Download the complete raw quantized model-input tensor", () => {
      const bytes = verified?.witnesses?.[0]?.bytes;
      if (bytes) onDownloadBinary?.(bytes, `input_witness_op_${witness.source_op_index}.${witness.model_input_dtype.toLowerCase()}.bin`, "application/octet-stream");
    });
    rawButton.disabled = !reconstructed;
    actions.append(
      witnessButton,
      rawButton,
      commandButton("Graph source", "Open the certified source operator in the graph workspace", () => jumpToGraphOp?.(witness.source_op_index)),
    );

    const visualBand = element("section", "input-witness-band visual");
    visualBand.append(sectionHead("Constructed receptive field", `tensor #${witness.model_input_tensor_index} ${witness.model_input_dtype} / zero-point fill ${witness.full_tensor_fill_code}`));
    const canvas = element("canvas", "input-witness-canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "Constructive model-input patch and exact-local source classification strip");
    visualBand.append(canvas, legend());

    const certificate = element("section", "input-witness-band");
    certificate.append(
      sectionHead("Exact arithmetic certificate", `#${padOp(witness.source_op_index)} ${witness.source_op_name} / channel ${witness.source_channel_index}`),
      definitionTable([
        ["Complete input tensor", `${witness.model_input_shape.join(" x ")} ${witness.model_input_dtype}; ${formatNumber(witness.model_input_element_count)} elements`],
        ["Sparse construction", `${formatNumber(witness.sparse_override_count)} overrides on zero-point code ${witness.full_tensor_fill_code}`],
        ["Certified output coordinate", `[${witness.source_output_coordinate.join(", ")}] from patch origin [${witness.patch_origin_yx.join(", ")}]`],
        ["Kernel geometry", `${witness.kernel_shape.join(" x ")} / stride ${witness.stride_hw.join(" x ")} / dilation ${witness.dilation_hw.join(" x ")} / ${witness.padding}`],
        ["Exact dot + bias", `${formatNumber(witness.dot_product_decimal)} + ${formatNumber(witness.bias_decimal)} = ${formatNumber(witness.post_bias_accumulator_decimal)}`],
        ["Pinned output paths", `default ${witness.default_output_code} / single-rounding ${witness.single_rounding_output_code} / delta ${signed(witness.output_code_delta)}`],
        ["Structural output routes", `${formatNumber(sourceForWitness(evidence, witness)?.exact_model_output_graph_route_count_decimal || 0)} graph routes; declared-output effect not proven`],
        ["Full tensor SHA-256", witness.full_model_input_tensor_sha256],
        ["Witness ledger SHA-256", witness.witness_ledger_sha256],
      ]),
    );

    const terms = element("section", "input-witness-band table-band");
    terms.append(
      sectionHead("Kernel term ledger", `${formatNumber(witness.terms.length)} independently summable terms`),
      table(["Term", "Kernel HWC", "Input NHWC", "Linear", "Code", "Centered", "Weight", "Product"], witness.terms.map((term) => [
        term.term_index,
        term.kernel_coordinate.join(","),
        term.input_coordinate.join(","),
        formatNumber(term.input_linear_index),
        term.input_code,
        term.centered_input_code,
        term.centered_weight,
        formatNumber(term.term_product_decimal),
      ])),
    );

    const portfolio = element("section", "input-witness-band table-band");
    portfolio.append(
      sectionHead("Realizability portfolio", evidence.source_classification_conservation),
      table(["Source", "Input origin", "Classification", "Exact channels", "Exact states", "Output routes", "Witness"], evidence.sources.map((source) => [
        opButton(source, jumpToGraphOp),
        humanize(source.input_origin),
        humanize(source.classification),
        formatNumber(source.exact_reachable_divergent_channel_count),
        formatNumber(source.exact_reachable_divergent_state_count_decimal),
        formatNumber(source.exact_model_output_graph_route_count_decimal || 0),
        source.representative_witness_index == null ? "none" : source.representative_witness_ledger_sha256.slice(0, 12),
      ])),
    );

    body.replaceChildren(actions, visualBand, certificate, terms, portfolio, message(evidence.interpretation_boundary, "boundary"));
    requestAnimationFrame(() => drawInputCounterexample(canvas, evidence, witness));
    if (typeof ResizeObserver === "function") {
      resizeObserver = new ResizeObserver(() => drawInputCounterexample(canvas, evidence, witness));
      resizeObserver.observe(body);
    }
  }

  return { render };
}

export function drawInputCounterexample(canvas, evidence, witness) {
  if (!canvas || !evidence || !witness) return;
  const compact = Number(canvas.parentElement?.clientWidth || 0) > 0 && Number(canvas.parentElement.clientWidth) < 620;
  const width = compact ? 390 : 1320;
  const height = compact ? 740 : 600;
  prepareCanvas(canvas, width, height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#0f1820";
  context.fillRect(0, 0, width, height);
  if (compact) {
    drawPatch(context, witness, 18, 60, 354, 330);
    drawSourceStrip(context, evidence, 18, 470, 354, 72);
  } else {
    drawPatch(context, witness, 44, 74, 500, 430);
    drawSourceStrip(context, evidence, 608, 105, 660, 170);
  }
  context.fillStyle = "#dbe9eb";
  context.font = `700 ${compact ? 17 : 22}px Inter, system-ui, sans-serif`;
  context.fillText("Constructive model-input tensor", compact ? 18 : 608, compact ? 612 : 348);
  context.fillStyle = "#8fa4b0";
  context.font = `${compact ? 12 : 16}px Inter, system-ui, sans-serif`;
  const lines = compact ? [
    `${formatNumber(witness.model_input_element_count)} elements / ${formatNumber(witness.sparse_override_count)} overrides`,
    `dot ${formatNumber(witness.dot_product_decimal)} + bias ${formatNumber(witness.bias_decimal)} = acc ${formatNumber(witness.post_bias_accumulator_decimal)}`,
    `output default ${witness.default_output_code} / single ${witness.single_rounding_output_code}`,
    `SHA-256 ${witness.full_model_input_tensor_sha256.slice(0, 24)}...`,
  ] : [
    `${formatNumber(witness.model_input_element_count)} elements / ${formatNumber(witness.sparse_override_count)} sparse overrides`,
    `dot ${formatNumber(witness.dot_product_decimal)} + bias ${formatNumber(witness.bias_decimal)} = accumulator ${formatNumber(witness.post_bias_accumulator_decimal)}`,
    `default code ${witness.default_output_code} versus single-rounding code ${witness.single_rounding_output_code}`,
    `tensor SHA-256 ${witness.full_model_input_tensor_sha256.slice(0, 24)}...`,
  ];
  lines.forEach((line, index) => context.fillText(line, compact ? 18 : 608, (compact ? 640 : 388) + index * (compact ? 27 : 34)));
}

export function renderInputCounterexampleCanvas(analysis, filename = "") {
  const evidence = analysis?.input_counterexample;
  const witness = evidence?.witnesses?.[0];
  if (!evidence || !witness || typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = 2360;
  canvas.height = 1600;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0d161e";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#eff7f7";
  context.font = "700 58px Inter, system-ui, sans-serif";
  context.fillText("Model Input Tensor ABI Witness", 92, 110);
  context.fillStyle = "#8fa3b1";
  context.font = "25px Inter, system-ui, sans-serif";
  context.fillText(filename || analysis.filename || "TFLite artifact", 94, 154);
  const metrics = [
    ["CONSTRUCTIVE SOURCES", `${evidence.tensor_abi_constructive_source_op_count} / ${evidence.exact_local_source_op_count}`],
    ["EXACT CHANNELS", formatNumber(evidence.tensor_abi_constructive_channel_count)],
    ["REALIZABLE STATES", formatNumber(evidence.tensor_abi_constructive_divergent_state_count_decimal)],
    ["INPUT ELEMENTS", formatNumber(witness.model_input_element_count)],
    ["SPARSE OVERRIDES", formatNumber(witness.sparse_override_count)],
  ];
  metrics.forEach(([label, value], index) => {
    const x = 94 + index * 445;
    context.fillStyle = "#89a0ad";
    context.font = "18px Inter, system-ui, sans-serif";
    context.fillText(label, x, 226);
    context.fillStyle = index === 2 ? "#efbd6c" : "#e9f4f4";
    context.font = "700 34px ui-monospace, SFMono-Regular, Consolas, monospace";
    context.fillText(value, x, 270);
  });
  drawPatch(context, witness, 100, 370, 860, 780);
  drawSourceStrip(context, evidence, 1090, 410, 1160, 260);
  context.fillStyle = "#e5eff1";
  context.font = "700 32px Inter, system-ui, sans-serif";
  context.fillText("Exact arithmetic certificate", 1090, 790);
  context.fillStyle = "#9eb0bb";
  context.font = "24px ui-monospace, SFMono-Regular, Consolas, monospace";
  const lines = [
    `op #${padOp(witness.source_op_index)} / channel ${witness.source_channel_index} / output [${witness.source_output_coordinate.join(",")}]`,
    `dot ${witness.dot_product_decimal} + bias ${witness.bias_decimal} = ${witness.post_bias_accumulator_decimal}`,
    `default ${witness.default_output_code} / single ${witness.single_rounding_output_code} / delta ${signed(witness.output_code_delta)}`,
    `full tensor ${formatNumber(witness.model_input_element_count)} bytes / overrides ${witness.sparse_override_count}`,
    `SHA-256 ${witness.full_model_input_tensor_sha256}`,
    `witness ${witness.witness_ledger_sha256}`,
  ];
  lines.forEach((line, index) => context.fillText(line, 1090, 850 + index * 58));
  context.fillStyle = "#607582";
  context.font = "19px Inter, system-ui, sans-serif";
  context.fillText("Exact at the quantized model tensor ABI; preprocessing-domain realizability and declared-output change remain outside this certificate.", 96, 1535);
  return canvas;
}

function drawPatch(context, witness, x, y, width, height) {
  const [patchH, patchW, channels] = witness.effective_patch_shape;
  const cell = Math.min(width / patchW, height / patchH);
  context.fillStyle = "#dce9eb";
  context.font = "700 20px Inter, system-ui, sans-serif";
  context.fillText(`${patchH} x ${patchW} x ${channels} receptive-field patch`, x, y - 28);
  for (let row = 0; row < patchH; row += 1) {
    for (let column = 0; column < patchW; column += 1) {
      const offset = (row * patchW + column) * channels;
      const values = witness.patch_codes_hwc.slice(offset, offset + channels);
      const rgb = [0, 1, 2].map((channel) => displayCode(values[channel] ?? witness.full_tensor_fill_code, witness.model_input_dtype));
      const px = x + column * cell;
      const py = y + row * cell;
      context.fillStyle = `rgb(${rgb.join(",")})`;
      context.fillRect(px, py, cell - 4, cell - 4);
      context.strokeStyle = "rgba(224,239,241,.34)";
      context.strokeRect(px, py, cell - 4, cell - 4);
      context.fillStyle = luminance(rgb) > 145 ? "#101920" : "#f4f8f8";
      context.font = `${Math.max(12, Math.min(21, cell / 7))}px ui-monospace, SFMono-Regular, Consolas, monospace`;
      context.textAlign = "center";
      context.fillText(values.join("/"), px + (cell - 4) / 2, py + cell / 2);
    }
  }
  context.textAlign = "left";
}

function drawSourceStrip(context, evidence, x, y, width, height) {
  const sources = evidence.sources;
  const cellWidth = width / Math.max(1, sources.length);
  context.fillStyle = "#dce9eb";
  context.font = "700 20px Inter, system-ui, sans-serif";
  context.fillText("Exact-local source realizability", x, y - 28);
  sources.forEach((source, index) => {
    context.fillStyle = source.classification === "tensor_abi_constructive" ? "#58c4b0"
      : source.classification === "upstream_activation_constraint_unresolved" ? "#9a7443" : "#d66353";
    context.fillRect(x + index * cellWidth, y, Math.max(2, cellWidth - 2), height);
  });
  context.fillStyle = "#93a8b4";
  context.font = "15px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.fillText(`#${sources[0]?.op_index ?? 0}`, x, y + height + 26);
  context.textAlign = "right";
  context.fillText(`#${sources.at(-1)?.op_index ?? 0}`, x + width, y + height + 26);
  context.textAlign = "left";
}

function renderSummary(root, evidence) {
  const witness = evidence.witnesses[0];
  const source = witness ? sourceForWitness(evidence, witness) : null;
  root?.replaceChildren(
    metric("Tensor-ABI constructive", `${formatNumber(evidence.tensor_abi_constructive_source_op_count)} / ${formatNumber(evidence.exact_local_source_op_count)}`, `${formatNumber(evidence.upstream_activation_unresolved_source_op_count)} upstream constraints unresolved`),
    metric("Exact channels", formatNumber(evidence.tensor_abi_constructive_channel_count), `${formatNumber(evidence.tensor_abi_constructive_divergent_state_count_decimal)} realizable divergent accumulator states`),
    metric("Complete input tensor", witness ? formatNumber(witness.model_input_element_count) : "none", witness ? `${formatNumber(witness.sparse_override_count)} sparse overrides / ${witness.model_input_dtype}` : "no representative witness"),
    metric("Certified code delta", witness ? signed(witness.output_code_delta) : "n/a", witness ? `accumulator ${formatNumber(witness.post_bias_accumulator_decimal)} / source channel ${witness.source_channel_index}` : "not applicable"),
    metric("Structural output routes", source ? formatNumber(source.exact_model_output_graph_route_count_decimal || 0) : "0", "route existence is not declared-output divergence"),
  );
}

function sourceForWitness(evidence, witness) {
  return evidence.sources.find((source) => source.op_index === witness.source_op_index);
}

function definitionTable(rows) {
  const list = element("dl", "input-witness-definition");
  for (const [label, value] of rows) list.append(element("dt", "", label), element("dd", "", String(value)));
  return list;
}

function table(headers, rows) {
  const wrap = element("div", "input-witness-table-wrap");
  const node = element("table", "input-witness-table");
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

function opButton(source, jumpToGraphOp) {
  const button = element("button", "input-witness-op", `#${padOp(source.op_index)} ${source.op_name}`);
  button.type = "button";
  button.addEventListener("click", () => jumpToGraphOp?.(source.op_index));
  return button;
}

function metric(label, value, detail) {
  const node = element("div", "input-witness-metric");
  node.append(element("span", "label", label), element("strong", "", value), element("small", "", detail));
  return node;
}

function sectionHead(title, detail) {
  const head = element("div", "input-witness-section-head");
  head.append(element("h4", "", title), element("span", "", detail));
  return head;
}

function legend() {
  const node = element("div", "input-witness-legend");
  for (const [className, label] of [["constructive", "tensor-ABI constructive"], ["unresolved", "upstream activation unresolved"], ["not-assessed", "not assessed"]]) {
    const item = element("span", className);
    item.append(element("i"), document.createTextNode(label));
    node.append(item);
  }
  return node;
}

function commandButton(label, title, handler) {
  const button = element("button", "secondary-action", label);
  button.type = "button";
  button.title = title;
  button.addEventListener("click", handler);
  return button;
}

function message(text, tone = "") {
  return element("p", `input-witness-message ${tone}`.trim(), text);
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

function displayCode(value, dtype) {
  return dtype === "INT8" ? Math.max(0, Math.min(255, value + 128)) : Math.max(0, Math.min(255, value));
}

function luminance([red, green, blue]) {
  return red * 0.299 + green * 0.587 + blue * 0.114;
}

function signed(value) {
  return `${value > 0 ? "+" : ""}${value}`;
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
