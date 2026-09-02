import { downloadBlob, downloadText } from "./download.js";
import { formatBytes, formatNumber, padOp } from "./format.js";
import { opSteadyStateUs } from "./analysis.js";
import {
  buildRedesignImplementationFiles,
  buildRedesignScenarioSet,
  canonicalScenario,
  scenarioFingerprint,
} from "./redesign-codegen.js";
import { createZipBlob } from "./zip.js";
import {
  createNodeViewController,
  getRedesignContractState,
  summarizeRedesignContracts,
} from "./node-view.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const CACHE_WATCH_RATIO = 0.9;
const L1_OPTIONS = [
  { label: "Bound", value: null },
  { label: "16 KiB", value: 16 * 1024 },
  { label: "32 KiB", value: 32 * 1024 },
  { label: "64 KiB", value: 64 * 1024 },
  { label: "128 KiB", value: 128 * 1024 },
];

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

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function ratioText(value) {
  return finite(value) == null ? "N/A" : `${Number(value).toFixed(2)}x`;
}

function percentText(value) {
  return finite(value) == null ? "N/A" : `${Number(value).toFixed(1)}%`;
}

function metricValue(value, formatter = formatNumber) {
  return value == null ? "N/A" : formatter(value);
}

function storageBytes(dtype) {
  const normalized = String(dtype || "").toUpperCase();
  if (["FLOAT64", "INT64"].includes(normalized)) return 8;
  if (["FLOAT32", "INT32", "UINT32"].includes(normalized)) return 4;
  if (["FLOAT16", "BFLOAT16", "INT16", "UINT16"].includes(normalized)) return 2;
  if (["INT8", "UINT8", "BOOL"].includes(normalized)) return 1;
  return null;
}

function cacheRows(analysis, state) {
  const l1Bytes = state.l1Bytes ?? Number(analysis?.target_profile?.l1_data_bytes || 0);
  const l2Bytes = Number(analysis?.target_profile?.l2_bytes || 0);
  const blockByOp = new Map();
  for (const block of analysis?.block_inventory?.blocks || []) {
    for (const opIndex of block.op_indices || []) {
      blockByOp.set(opIndex, {
        blockId: block.block_id,
        stageIndex: block.stage_index,
      });
    }
  }
  return (analysis?.ops || [])
    .filter((op) => op.cache_payload?.status === "assessed")
    .map((op) => {
      const payload = op.cache_payload;
      const sourceInputWidth = storageBytes(payload.input_dtype);
      const sourceOutputWidth = storageBytes(payload.output_dtype);
      const requestedWidth = state.dtype === "int8" ? 1 : state.dtype === "float32" ? 4 : null;
      const inputBytes = requestedWidth == null || !sourceInputWidth
        ? Number(payload.input_strip_bytes || 0)
        : Number(payload.input_strip_bytes || 0) / sourceInputWidth * requestedWidth;
      const outputBytes = requestedWidth == null || !sourceOutputWidth
        ? Number(payload.output_row_bytes || 0)
        : Number(payload.output_row_bytes || 0) / sourceOutputWidth * requestedWidth;
      const logicalBytes = inputBytes + outputBytes;
      const l1Ratio = l1Bytes > 0 ? logicalBytes / l1Bytes : null;
      const l2Ratio = l2Bytes > 0 ? logicalBytes / l2Bytes : null;
      const owner = blockByOp.get(op.index) || {};
      return {
        op,
        payload,
        inputBytes,
        outputBytes,
        logicalBytes,
        l1Bytes,
        l2Bytes,
        l1Ratio,
        l2Ratio,
        width: Number(payload.input_width || 0),
        channels: Number(payload.input_channels || 0),
        timeUs: opSteadyStateUs(op),
        blockId: owner.blockId || "",
        stageIndex: owner.stageIndex ?? null,
      };
    })
    .filter((row) => {
      if (state.cacheStage !== "all" && String(row.stageIndex) !== state.cacheStage) return false;
      if (state.filter === "watch") return Number(row.l1Ratio || 0) >= CACHE_WATCH_RATIO;
      if (state.filter === "2x") return Number(row.l1Ratio || 0) >= 2;
      if (state.filter === "5x") return Number(row.l1Ratio || 0) >= 5;
      if (state.filter === "l2") return Number(row.l2Ratio || 0) >= 1;
      return true;
    });
}

function stageLabel(stage) {
  const spatial = stage?.spatial || {};
  const output = spatial.output_h && spatial.output_w ? `${spatial.output_h}x${spatial.output_w}` : "non-spatial";
  return `Stage ${stage.index} / ${output} / ${stage.display_name || "operator group"}`;
}

function blockSummary(block) {
  const aggregate = block.aggregates || {};
  return `${formatNumber(aggregate.op_count || 0)} ops / ${percentText(Number(aggregate.mac_percent || 0) * 100)} MAC / max L1 ${ratioText(aggregate.l1_max_ratio)} / ${formatNumber(aggregate.predicted_break_count || 0)} predicted breaks`;
}

function renderBlocks(root, analysis, state, actions) {
  root.replaceChildren();
  const inventory = analysis?.block_inventory;
  if (!inventory || inventory.status !== "assessed") {
    root.append(emptyPanel(
      "Block Inventory not assessed",
      analysis?.format === "onnx"
        ? "ONNX block semantics are intentionally suppressed until an ONNX-specific graph-pattern contract is implemented."
        : "The analyzer did not emit a block inventory for this artifact.",
    ));
    return;
  }
  const head = element("div", "xr-section-head");
  const title = element("div");
  title.append(
    element("span", "xr-kicker", "BLOCK INVENTORY"),
    element("h3", "", "Architecture blocks and stage pressure"),
    element("p", "", "Graph-semantic motifs are preferred; names only label otherwise-unclaimed groups. Stage cache pressure is the maximum op ratio, never a sum."),
  );
  const counts = element("div", "xr-head-metrics");
  counts.append(
    compactMetric("Stages", inventory.stage_count),
    compactMetric("Blocks", inventory.block_count),
    compactMetric("Semantic", inventory.semantic_block_count),
    compactMetric("Named", inventory.named_block_count),
    compactMetric("Unnamed", inventory.unnamed_block_count),
  );
  head.append(title, counts);
  const toolbar = element("div", "xr-cache-toolbar xr-block-toolbar");
  toolbar.append(segmentGroup(
    "Sort",
    [
      ["execution", "Execution"],
      ["impact", "Impact"],
    ].map(([value, label]) => ({
      label,
      active: state.blockSort === value,
      run: () => {
        state.blockSort = value;
        renderBlocks(root, analysis, state, actions);
      },
    })),
  ));
  const warnings = button(state.blockWarningsOnly ? "Warnings only" : "All blocks", state.blockWarningsOnly ? "active" : "");
  warnings.addEventListener("click", () => {
    state.blockWarningsOnly = !state.blockWarningsOnly;
    renderBlocks(root, analysis, state, actions);
  });
  const collapse = button(state.collapseStages ? "Expand stages" : "Collapse stages");
  collapse.addEventListener("click", () => {
    state.collapseStages = !state.collapseStages;
    renderBlocks(root, analysis, state, actions);
  });
  toolbar.append(warnings, collapse);

  const layout = element("div", "xr-block-layout");
  const list = element("div", "xr-stage-list");
  const detail = element("aside", "xr-block-detail");
  const selectedBlock = inventory.blocks.find((block) => block.block_id === state.selectedBlockId)
    || inventory.blocks[0];
  if (selectedBlock) state.selectedBlockId = selectedBlock.block_id;

  const totalTraffic = (inventory.stages || []).reduce(
    (sum, stage) => sum + Number(stage.aggregates?.logical_traffic_bytes || 0),
    0,
  );
  let stages = [...(inventory.stages || [])];
  if (state.blockWarningsOnly) {
    stages = stages.filter((stage) => (stage.block_ids || []).some((blockId) => {
      const block = inventory.blocks.find((item) => item.block_id === blockId);
      return block && blockHasWarning(block, analysis);
    }));
  }
  if (state.blockSort === "impact") {
    stages.sort((left, right) => Number(right.aggregates?.modeled_time_ms || 0) - Number(left.aggregates?.modeled_time_ms || 0));
  }
  for (const stage of stages) {
    const stageNode = element("section", "xr-stage");
    const stageButton = button("", "xr-stage-head");
    const identity = element("span");
    identity.append(
      element("strong", "", stageLabel(stage)),
      element("small", "", blockSummary(stage)),
    );
    const l1 = Number(stage.aggregates?.l1_max_ratio || 0);
    const badges = element("span", "xr-stage-badges");
    if (l1 >= 5) badges.append(element("em", "danger", `L1 ${l1.toFixed(2)}x`));
    else if (l1 >= CACHE_WATCH_RATIO) badges.append(element("em", "warn", `L1 ${l1.toFixed(2)}x`));
    if (Number(stage.aggregates?.predicted_break_count || 0) > 0) {
      badges.append(element("em", "warn", `${stage.aggregates.predicted_break_count} breaks`));
    }
    if (totalTraffic > 0 && Number(stage.aggregates?.logical_traffic_bytes || 0) / totalTraffic > 0.1) {
      badges.append(element("em", "warn", "traffic >10%"));
    }
    stageButton.append(identity, badges);
    stageButton.addEventListener("click", () => {
      if (!state.collapsedStageIds.has(stage.stage_id)) state.collapsedStageIds.add(stage.stage_id);
      else state.collapsedStageIds.delete(stage.stage_id);
      renderBlocks(root, analysis, state, actions);
    });
    stageNode.append(stageButton);
    const blockList = element("div", "xr-block-list");
    const stageBlocks = (stage.block_ids || [])
      .map((blockId) => inventory.blocks.find((item) => item.block_id === blockId))
      .filter(Boolean)
      .filter((block) => !state.blockWarningsOnly || blockHasWarning(block, analysis))
      .sort((left, right) => state.blockSort === "impact"
        ? Number(right.aggregates?.modeled_time_ms || 0) - Number(left.aggregates?.modeled_time_ms || 0)
        : Number(left.op_indices?.[0] || 0) - Number(right.op_indices?.[0] || 0));
    for (const block of stageBlocks) {
      const item = button("", `xr-block-row${block.block_id === state.selectedBlockId ? " active" : ""}`);
      const left = element("span");
      left.append(
        element("strong", "", block.display_name),
        element("small", "", `${block.block_type.replaceAll("_", " ")} / ${block.extraction.method} ${block.extraction.confidence}`),
      );
      const right = element("span", "xr-block-row-metrics");
      right.append(
        element("b", "", `${formatNumber(block.aggregates?.op_count || 0)} ops`),
        element("b", "", `L1 ${ratioText(block.aggregates?.l1_max_ratio)}`),
      );
      item.append(left, right);
      item.addEventListener("click", () => {
        state.selectedBlockId = block.block_id;
        renderBlocks(root, analysis, state, actions);
      });
      blockList.append(item);
    }
    blockList.hidden = state.collapseStages || state.collapsedStageIds.has(stage.stage_id);
    stageNode.append(blockList);
    list.append(stageNode);
  }
  if (selectedBlock) renderBlockDetail(detail, selectedBlock, analysis, actions);
  layout.append(list, detail);
  root.append(head, toolbar, layout);
}

function blockHasWarning(block, analysis) {
  if (Number(block.aggregates?.l1_max_ratio || 0) >= CACHE_WATCH_RATIO) return true;
  if (Number(block.aggregates?.predicted_break_count || 0) > 0) return true;
  const ops = new Set(block.op_indices || []);
  return (analysis.ops || []).some((op) => ops.has(op.index) && !["none", "low"].includes(String(op.quant_risk || "none").toLowerCase()));
}

function renderBlockDetail(root, block, analysis, actions) {
  root.replaceChildren();
  const top = element("div", "xr-detail-head");
  const title = element("div");
  title.append(
    element("span", "xr-kicker", `STAGE ${block.stage_index}`),
    element("h3", "", block.display_name),
    element("p", "", `${block.extraction.method} / ${block.extraction.confidence} / ${block.extraction.source_pattern}`),
  );
  const clone = button("Clone to Redesign", "primary-action");
  clone.addEventListener("click", () => actions.openRedesign?.(block.block_id));
  top.append(title, clone);

  const structure = element("div", "xr-structure-ledger");
  structure.append(
    ledgerRow("Type", block.block_type.replaceAll("_", " ")),
    ledgerRow("Ops", (block.op_indices || []).map((index) => `#${padOp(index)}`).join(" -> ")),
    ledgerRow("Spatial", `${block.spatial?.input_h ?? "?"}x${block.spatial?.input_w ?? "?"} -> ${block.spatial?.output_h ?? "?"}x${block.spatial?.output_w ?? "?"}`),
    ledgerRow("Channels", `${block.channels?.input ?? "?"} -> ${block.channels?.expand ?? "-"} -> ${block.channels?.output ?? "?"}`),
    ledgerRow("Kernel / stride", `${block.params?.kernel_h ?? "?"}x${block.params?.kernel_w ?? "?"} / ${block.params?.stride_h ?? "?"}x${block.params?.stride_w ?? "?"}`),
    ledgerRow("Residual", block.residual ? "graph edge present" : "not detected"),
    ledgerRow("Liveness", block.residual ? "The skip input remains live through the terminal ADD; model peak uses the graph liveness ledger." : "No block-spanning residual lifetime was detected."),
  );
  const aggregate = block.aggregates || {};
  const metrics = element("div", "xr-detail-metrics");
  metrics.append(
    compactMetric("MACs", formatNumber(aggregate.macs || 0)),
    compactMetric("Modeled steady", `${Number(aggregate.modeled_time_ms || 0).toFixed(3)} ms`),
    compactMetric("Cold start", `${Number(aggregate.modeled_cold_start_time_ms ?? aggregate.modeled_time_ms ?? 0).toFixed(3)} ms`),
    compactMetric("Max L1", ratioText(aggregate.l1_max_ratio)),
    compactMetric("L1 watch", aggregate.l1_watch_count || 0),
    compactMetric("Logical traffic", formatBytes(aggregate.logical_traffic_bytes || 0)),
    compactMetric("Parameters", formatNumber(aggregate.parameter_elements || 0)),
  );
  const ops = element("div", "xr-op-chip-list");
  for (const opIndex of block.op_indices || []) {
    const op = (analysis.ops || []).find((item) => item.index === opIndex);
    const row = button("", "xr-op-ledger");
    row.title = "Open this operator in the Ops view";
    row.append(
      element("strong", "", `#${padOp(opIndex)} ${op?.name || "UNKNOWN"}`),
      element("span", "", op?.cache_payload?.logical_row_payload_bytes == null
        ? "cache N/A"
        : `${formatBytes(op.cache_payload.logical_row_payload_bytes)} / ${ratioText(cacheRatioForOp(op, analysis))} L1`),
    );
    row.addEventListener("click", () => actions.selectOp?.(opIndex));
    ops.append(row);
  }
  root.append(top, structure, metrics, element("h4", "", "Operators"), ops);
}

function cacheRatioForOp(op, analysis) {
  const denominator = Number(analysis.target_profile?.l1_data_bytes || 0);
  return denominator > 0
    ? Number(op.cache_payload?.logical_row_payload_bytes || 0) / denominator
    : null;
}

function renderCache(root, analysis, state, actions) {
  root.replaceChildren();
  const globalRows = cacheRows(analysis, { ...state, filter: "all", cacheStage: "all" });
  const allRows = cacheRows(analysis, { ...state, filter: "all" });
  if (!globalRows.length) {
    root.append(emptyPanel(
      "Cache payload not assessed",
      "No convolution, depthwise convolution, fully connected, or batched-matmul row payload was deterministically available.",
    ));
    return;
  }
  const head = element("div", "xr-section-head");
  const title = element("div");
  title.append(
    element("span", "xr-kicker", "CACHE SURFACE"),
    element("h3", "", "Logical row payload against target cache references"),
    element("p", "", "Points use input width and channels. Color encodes input-strip plus output-row payload divided by the selected L1D denominator. This is not a cache hit-rate or residency claim."),
  );
  const max = allRows.length ? Math.max(...allRows.map((row) => Number(row.l1Ratio || 0))) : null;
  head.append(title, compactMetric("Maximum", max == null ? "N/A in filter" : `${max.toFixed(2)}x L1D`));

  const toolbar = element("div", "xr-cache-toolbar");
  toolbar.append(segmentGroup(
    "L1D",
    L1_OPTIONS.map((option) => ({
      label: option.label,
      active: state.l1Bytes === option.value,
      run: () => {
        state.l1Bytes = option.value;
        renderCache(root, analysis, state, actions);
      },
    })),
  ));
  toolbar.append(segmentGroup(
    "Storage",
    ["source", "int8", "float32"].map((value) => ({
      label: value === "source" ? "Source" : value.toUpperCase(),
      active: state.dtype === value,
      run: () => {
        state.dtype = value;
        renderCache(root, analysis, state, actions);
      },
    })),
  ));
  toolbar.append(segmentGroup(
    "Filter",
    [
      ["all", "All"],
      ["watch", ">=0.9x"],
      ["2x", ">=2x"],
      ["5x", ">=5x"],
      ["l2", "L2 exceed"],
    ].map(([value, label]) => ({
      label,
      active: state.filter === value,
      run: () => {
        state.filter = value;
        renderCache(root, analysis, state, actions);
      },
    })),
  ));
  const stageOptions = [
    ["all", "All stages"],
    ...(analysis.block_inventory?.stages || []).map((stage) => [
      String(stage.index),
      `Stage ${stage.index} / ${stage.spatial?.output_h ?? "?"}x${stage.spatial?.output_w ?? "?"}`,
    ]),
  ];
  const stageFilter = selectField("Block", stageOptions, state.cacheStage);
  stageFilter.root.classList.add("xr-compact-select");
  stageFilter.input.addEventListener("change", () => {
    state.cacheStage = stageFilter.input.value;
    renderCache(root, analysis, state, actions);
  });
  toolbar.append(stageFilter.root);
  const trajectory = button(state.trajectory ? "Trajectory on" : "Trajectory off", state.trajectory ? "active" : "");
  trajectory.addEventListener("click", () => {
    state.trajectory = !state.trajectory;
    renderCache(root, analysis, state, actions);
  });
  toolbar.append(trajectory);
  const blockColor = button(state.blockColor ? "Block color on" : "Block color off", state.blockColor ? "active" : "");
  blockColor.addEventListener("click", () => {
    state.blockColor = !state.blockColor;
    renderCache(root, analysis, state, actions);
  });
  toolbar.append(blockColor);
  const projectionOverlay = button(
    state.projectionOverlay ? "Projection overlay on" : "Projection overlay off",
    state.projectionOverlay ? "active" : "",
  );
  projectionOverlay.disabled = !(state.projection?.cache_points || []).length;
  projectionOverlay.title = projectionOverlay.disabled
    ? "Run a Redesign projection to compare source and projected cache coordinates."
    : "Show source-to-projected W x C movement vectors from the WASM projection.";
  projectionOverlay.addEventListener("click", () => {
    state.projectionOverlay = !state.projectionOverlay;
    renderCache(root, analysis, state, actions);
  });
  toolbar.append(projectionOverlay);

  const rows = cacheRows(analysis, state);
  const projectionPoints = state.projectionOverlay && state.dtype === "source"
    ? state.projection?.cache_points || []
    : [];
  const selected = rows.find((row) => row.op.index === state.selectedCacheOp)
    || [...rows].sort((left, right) => Number(right.l1Ratio || 0) - Number(left.l1Ratio || 0))[0]
    || allRows[0];
  if (selected) state.selectedCacheOp = selected.op.index;
  const body = element("div", "xr-cache-layout");
  const visualColumn = element("div", "xr-cache-visuals");
  const scatterPanel = element("article", "xr-panel");
  const scatterHead = panelHeading(
    "W x C surface",
    `${rows.length}/${allRows.length} visible; boundary guides use #${selected ? padOp(selected.op.index) : "-"}`,
  );
  const scatter = renderCacheScatter(rows, state, (row) => {
    state.selectedCacheOp = row.op.index;
    renderCache(root, analysis, state, actions);
  }, projectionPoints, selected);
  scatterPanel.append(scatterHead, scatter, cacheLegend(projectionPoints.length > 0));
  const timelinePanel = element("article", "xr-panel");
  timelinePanel.append(
    panelHeading("Execution-order pressure", "Click a bar to inspect its payload ledger"),
    renderCacheTimeline(rows, (row) => {
      state.selectedCacheOp = row.op.index;
      renderCache(root, analysis, state, actions);
    }),
  );
  visualColumn.append(scatterPanel, timelinePanel);
  const side = element("aside", "xr-cache-side");
  if (selected) side.append(renderCacheDetail(selected, analysis, state, actions));
  side.append(renderChannelBudget(rows, analysis, state));
  body.append(visualColumn, side);
  root.append(head, toolbar, body);
}

function renderCacheScatter(rows, state, select, projectionPoints = [], guideRow = null) {
  const wrap = element("div", "xr-scatter-wrap");
  if (!rows.length) {
    wrap.append(element("p", "xr-empty", "No rows match the active cache filter."));
    return wrap;
  }
  const width = 760;
  const height = 390;
  const margin = { left: 58, right: 24, top: 24, bottom: 48 };
  const projected = projectionPoints.filter((point) => point.projected_width > 0 && point.projected_channels > 0);
  const xValues = [
    ...rows.map((row) => Math.max(1, row.width)),
    ...projected.map((point) => Math.max(1, point.projected_width)),
  ];
  const yValues = [
    ...rows.map((row) => Math.max(1, row.channels)),
    ...projected.map((point) => Math.max(1, point.projected_channels)),
  ];
  const xMin = Math.log2(Math.max(1, Math.min(...xValues) / 1.25));
  const xMax = Math.log2(Math.max(...xValues) * 1.25);
  const yMin = Math.log2(Math.max(1, Math.min(...yValues) / 1.25));
  const yMax = Math.log2(Math.max(...yValues) * 1.25);
  const x = (value) => margin.left + (Math.log2(Math.max(1, value)) - xMin) / Math.max(1e-9, xMax - xMin) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (Math.log2(Math.max(1, value)) - yMin) / Math.max(1e-9, yMax - yMin) * (height - margin.top - margin.bottom);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Input width by channel cache pressure scatter",
  });
  const defs = svgElement("defs");
  const marker = svgElement("marker", {
    id: "xr-projection-arrow",
    markerWidth: 7,
    markerHeight: 7,
    refX: 6,
    refY: 3.5,
    orient: "auto",
    markerUnits: "strokeWidth",
  });
  marker.append(svgElement("path", { d: "M0,0 L7,3.5 L0,7 Z", class: "xr-projection-arrow-head" }));
  defs.append(marker);
  svg.append(defs);
  for (const value of powersOfTwo(Math.min(...xValues), Math.max(...xValues))) {
    const axisX = x(value);
    svg.append(svgElement("line", { x1: axisX, x2: axisX, y1: margin.top, y2: height - margin.bottom, class: "xr-grid-line" }));
    const label = svgElement("text", { x: axisX, y: height - 17, class: "xr-axis-label", "text-anchor": "middle" });
    label.textContent = String(value);
    svg.append(label);
  }
  for (const value of powersOfTwo(Math.min(...yValues), Math.max(...yValues))) {
    const axisY = y(value);
    svg.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: axisY, y2: axisY, class: "xr-grid-line" }));
    const label = svgElement("text", { x: margin.left - 9, y: axisY + 4, class: "xr-axis-label", "text-anchor": "end" });
    label.textContent = String(value);
    svg.append(label);
  }
  const xTitle = svgElement("text", { x: width / 2, y: height - 3, class: "xr-axis-title", "text-anchor": "middle" });
  xTitle.textContent = "Input width W (log2)";
  const yTitle = svgElement("text", { x: 14, y: height / 2, class: "xr-axis-title", transform: `rotate(-90 14 ${height / 2})`, "text-anchor": "middle" });
  yTitle.textContent = "Input channels C (log2)";
  svg.append(xTitle, yTitle);
  if (guideRow) {
    const guides = [
      ["L1 0.5x", Number(guideRow.l1Bytes || 0) * 0.5, "half"],
      ["L1 1.0x", Number(guideRow.l1Bytes || 0), "one"],
      ["L1 2.0x", Number(guideRow.l1Bytes || 0) * 2, "double"],
      ["L2 1.0x", Number(guideRow.l2Bytes || 0), "l2"],
    ];
    for (const [labelText, budget, tone] of guides) {
      const guide = cacheBoundaryGuide(guideRow, budget, Math.min(...xValues), Math.max(...xValues), state);
      const visible = guide.filter((point) => point.channels > 0 && point.channels >= 2 ** yMin / 1.5 && point.channels <= 2 ** yMax * 1.5);
      if (visible.length < 2) continue;
      svg.append(svgElement("polyline", {
        points: visible.map((point) => `${x(point.width)},${y(point.channels)}`).join(" "),
        class: `xr-cache-guide ${tone}`,
        fill: "none",
      }));
      const last = visible.at(-1);
      const label = svgElement("text", {
        x: x(last.width) - 4,
        y: y(last.channels) - 5,
        class: `xr-guide-label ${tone}`,
        "text-anchor": "end",
      });
      label.textContent = labelText;
      svg.append(label);
    }
  }
  if (state.trajectory && rows.length > 1) {
    const ordered = [...rows].sort((left, right) => left.op.index - right.op.index);
    svg.append(svgElement("polyline", {
      points: ordered.map((row) => `${x(row.width)},${y(row.channels)}`).join(" "),
      class: "xr-trajectory",
      fill: "none",
    }));
  }
  const sourceRows = new Map(rows.map((row) => [row.op.index, row]));
  for (const point of projected) {
    const source = sourceRows.get(point.op_index);
    if (!source?.width || !source?.channels) continue;
    const same = source.width === point.projected_width && source.channels === point.projected_channels;
    if (!same) {
      svg.append(svgElement("line", {
        x1: x(source.width),
        y1: y(source.channels),
        x2: x(point.projected_width),
        y2: y(point.projected_channels),
        class: "xr-projection-vector",
        "marker-end": "url(#xr-projection-arrow)",
      }));
    }
    const projectedPoint = svgElement("circle", {
      cx: x(point.projected_width),
      cy: y(point.projected_channels),
      r: 4.5,
      class: "xr-projected-point",
    });
    const projectedTitle = svgElement("title");
    projectedTitle.textContent = `#${padOp(point.op_index)} projected: W ${point.projected_width}, C ${point.projected_channels}, ${point.projected_logical_row_payload_bytes == null ? "N/A" : formatBytes(point.projected_logical_row_payload_bytes)}, ${ratioText(point.projected_l1_ratio)} L1`;
    projectedPoint.append(projectedTitle);
    svg.append(projectedPoint);
  }
  const maxTime = Math.max(1, ...rows.map((row) => row.timeUs));
  for (const row of rows) {
    if (!row.width || !row.channels) continue;
    const group = svgElement("g", { tabindex: "0", role: "button", class: "xr-cache-point" });
    const radius = 5 + Math.sqrt(Math.max(0, row.timeUs) / maxTime) * 8;
    const circle = svgElement("circle", {
      cx: x(row.width),
      cy: y(row.channels),
      r: radius,
      class: cacheTone(row.l1Ratio),
    });
    if (state.blockColor && row.stageIndex != null) {
      circle.style.fill = stageColor(row.stageIndex);
      circle.style.stroke = cacheStroke(row.l1Ratio);
    }
    if (row.op.xnnpack_supported === false || row.op.chain_break) {
      circle.style.strokeDasharray = "4 3";
    }
    const title = svgElement("title");
    title.textContent = `#${padOp(row.op.index)} ${row.op.name}: W ${row.width}, C ${row.channels}, ${formatBytes(row.logicalBytes)}, ${ratioText(row.l1Ratio)} L1`;
    group.append(circle, title);
    group.addEventListener("click", () => select(row));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") select(row);
    });
    svg.append(group);
  }
  wrap.append(svg);
  return wrap;
}

function cacheBoundaryGuide(row, budget, minimumWidth, maximumWidth, state) {
  if (!["CONV_2D", "DEPTHWISE_CONV_2D"].includes(row.op.name)) return [];
  if (!(budget > 0) || !(row.width > 0) || !(row.channels > 0)) return [];
  const sourceInputWidth = storageBytes(row.payload.input_dtype);
  const sourceOutputWidth = storageBytes(row.payload.output_dtype);
  if (!sourceInputWidth || !sourceOutputWidth) return [];
  const inputBytes = state.dtype === "int8" ? 1 : state.dtype === "float32" ? 4 : sourceInputWidth;
  const outputBytes = state.dtype === "int8" ? 1 : state.dtype === "float32" ? 4 : sourceOutputWidth;
  const effectiveKh = Number(row.payload.effective_kernel_height || 0);
  const outputWidthScale = Number(row.payload.output_width || 0) / Number(row.payload.input_width || 1);
  const outputChannels = Number(row.payload.output_channels || 0);
  if (!(effectiveKh > 0) || !(outputWidthScale > 0) || !(outputChannels > 0)) return [];
  const depthwise = row.op.name === "DEPTHWISE_CONV_2D";
  const depthMultiplier = depthwise ? outputChannels / Math.max(1, row.channels) : 0;
  const steps = 48;
  const logMin = Math.log2(Math.max(1, minimumWidth));
  const logMax = Math.log2(Math.max(1, maximumWidth));
  const points = [];
  for (let index = 0; index < steps; index += 1) {
    const width = 2 ** (logMin + (logMax - logMin) * index / Math.max(1, steps - 1));
    const outputWidth = width * outputWidthScale;
    const inputCoefficient = effectiveKh * width * inputBytes;
    const outputCoefficient = depthwise ? outputWidth * depthMultiplier * outputBytes : 0;
    const fixedOutput = depthwise ? 0 : outputWidth * outputChannels * outputBytes;
    const channels = budget > fixedOutput
      ? (budget - fixedOutput) / Math.max(1e-9, inputCoefficient + outputCoefficient)
      : 0;
    points.push({ width, channels });
  }
  return points;
}

function renderCacheTimeline(rows, select) {
  const list = element("div", "xr-cache-timeline");
  if (!rows.length) {
    list.append(element("p", "xr-empty", "No rows match the active cache filter."));
    return list;
  }
  const maximum = Math.max(1, ...rows.map((row) => Number(row.l1Ratio || 0)));
  for (const row of [...rows].sort((left, right) => left.op.index - right.op.index)) {
    const item = button("", "xr-timeline-row");
    item.title = `#${padOp(row.op.index)} ${row.op.name}`;
    const label = element("span", "", `#${padOp(row.op.index)}`);
    const track = element("span", "xr-timeline-track");
    const fill = element("i", cacheTone(row.l1Ratio));
    fill.style.height = `${Math.max(3, Number(row.l1Ratio || 0) / maximum * 100)}%`;
    track.append(fill);
    item.append(label, track);
    item.addEventListener("click", () => select(row));
    list.append(item);
  }
  return list;
}

function renderCacheDetail(row, analysis, state, actions) {
  const panel = element("article", "xr-panel xr-cache-detail");
  const head = panelHeading(`#${padOp(row.op.index)} ${row.op.name}`, `${formatBytes(row.logicalBytes)} / ${ratioText(row.l1Ratio)} L1D`);
  const composition = element("div", "xr-payload-bars");
  const total = Math.max(1, row.logicalBytes);
  composition.append(
    payloadBar("Input strip", row.inputBytes, total),
    payloadBar("Output row", row.outputBytes, total),
    payloadBar("Serialized kernel", Number(row.payload.serialized_kernel_bytes || 0), null),
    payloadBar("Serialized bias", Number(row.payload.serialized_bias_bytes || 0), null),
  );
  const ledger = element("div", "xr-structure-ledger");
  ledger.append(
    ledgerRow("Geometry", `W ${row.width} / C ${row.channels} / KH_eff ${row.payload.effective_kernel_height || "?"} / KW ${row.payload.kernel_width || "?"}`),
    ledgerRow("Storage", `${state.dtype === "source" ? `${row.payload.input_dtype} -> ${row.payload.output_dtype}` : state.dtype.toUpperCase()} counterfactual`),
    ledgerRow("L1D reference", `${formatBytes(row.l1Bytes)} / ${ratioText(row.l1Ratio)}`),
    ledgerRow("L2 reference", `${formatBytes(row.l2Bytes)} / ${ratioText(row.l2Ratio)}`),
    ledgerRow("Evidence", `${row.payload.evidence_class}; ${row.payload.status}`),
  );
  const whatIf = element("div", "xr-whatif-table");
  const sourceWidths = [
    storageBytes(row.payload.input_dtype),
    storageBytes(row.payload.output_dtype),
  ].filter(Boolean);
  const alreadyByteStorage = sourceWidths.length > 0 && sourceWidths.every((width) => width === 1);
  const scenarios = [
    ["Current view", state.dtype, 1, 1],
    ["Width half", state.dtype, 0.5, 1],
    ...(!alreadyByteStorage && state.dtype !== "int8"
      ? [
          ["INT8 storage", "int8", 1, 1],
          ["INT8 + width half", "int8", 0.5, 1],
        ]
      : []),
  ];
  for (const [label, dtype, widthScale, channelScale] of scenarios) {
    const projected = projectedLogicalPayload(row, dtype, widthScale, channelScale);
    const ratio = row.l1Bytes > 0 ? projected / row.l1Bytes : null;
    const line = element("div");
    line.append(
      element("span", "", label),
      element("strong", cacheTone(ratio), ratioText(ratio)),
      element(
        "small",
        "",
        label === "Current view"
          ? "viewer-only"
          : label.includes("Width")
            ? "shape rewrite + validation"
            : "conversion + validation",
      ),
    );
    whatIf.append(line);
  }
  const open = button("Open in Ops", "secondary-action");
  open.addEventListener("click", () => actions.selectOp?.(row.op.index));
  panel.append(head, composition, ledger, element("h4", "", "Storage-width what-if"), whatIf, open);
  return panel;
}

function projectedLogicalPayload(row, dtype, widthScale, channelScale) {
  const sourceInputWidth = storageBytes(row.payload.input_dtype);
  const sourceOutputWidth = storageBytes(row.payload.output_dtype);
  if (!sourceInputWidth || !sourceOutputWidth) return row.logicalBytes;
  const requestedWidth = dtype === "int8" ? 1 : dtype === "float32" ? 4 : null;
  const inputWidth = requestedWidth ?? sourceInputWidth;
  const outputWidth = requestedWidth ?? sourceOutputWidth;
  const inputElements = Number(row.payload.input_strip_bytes || 0) / sourceInputWidth;
  const outputElements = Number(row.payload.output_row_bytes || 0) / sourceOutputWidth;
  return (
    inputElements * inputWidth * widthScale * channelScale
    + outputElements * outputWidth * widthScale * channelScale
  );
}

function renderChannelBudget(rows, analysis, state) {
  const panel = element("article", "xr-panel");
  const selectedL1 = state.l1Bytes ?? Number(analysis.target_profile?.l1_data_bytes || 0);
  panel.append(panelHeading("Channel budget", `Selected ${formatBytes(selectedL1)} L1D reference`));
  const tableWrap = element("div", "xr-table-wrap");
  const table = element("table", "xr-budget-table");
  const head = element("thead");
  const headRow = element("tr");
  for (const label of ["Op", "W", "Current C", "INT8 max C", "FP32 max C"]) headRow.append(element("th", "", label));
  head.append(headRow);
  const body = element("tbody");
  const unique = [...rows]
    .sort((left, right) => right.width - left.width || left.op.index - right.op.index)
    .slice(0, 16);
  for (const row of unique) {
    const line = element("tr");
    const int8 = channelBudget(row, 1);
    const fp32 = channelBudget(row, 4);
    const values = [
      `#${padOp(row.op.index)}`,
      row.width,
      row.channels,
      int8,
      fp32,
    ];
    values.forEach((value, index) => {
      const cell = element("td", "", value?.label ?? String(value));
      if (index >= 3 && value?.reason) cell.title = value.reason;
      line.append(cell);
    });
    body.append(line);
  }
  table.append(head, body);
  tableWrap.append(table);
  panel.append(tableWrap, element("p", "xr-note", "For depthwise-like rows, the denominator includes both the effective input strip and output row. For fixed-output Conv rows, the output-row payload is reserved before solving the input-channel limit. This is a logical-payload constraint only."));
  return panel;
}

function channelBudget(row, bytesPerElement) {
  const l1 = Number(row.l1Bytes || 0);
  const kh = Number(row.payload.effective_kernel_height || 0);
  const inputWidth = Number(row.payload.input_width || 0);
  const outputWidth = Number(row.payload.output_width || 0);
  const outputChannels = Number(row.payload.output_channels || 0);
  if (!l1 || !kh || !inputWidth || !outputWidth || !outputChannels) {
    return { label: "N/A", reason: "Required geometry or L1D reference is not bound." };
  }
  const depthwiseLike = row.op.name === "DEPTHWISE_CONV_2D";
  const denominator = kh * inputWidth * bytesPerElement + (depthwiseLike ? outputWidth * bytesPerElement : 0);
  const reserved = depthwiseLike ? 0 : outputWidth * outputChannels * bytesPerElement;
  if (denominator <= 0) return { label: "N/A", reason: "The logical payload denominator is not positive." };
  if (l1 <= reserved) {
    return {
      label: "none",
      reason: `The fixed output row alone is ${formatBytes(reserved)}, which meets or exceeds the ${formatBytes(l1)} L1D reference.`,
    };
  }
  const channels = Math.floor((l1 - reserved) / denominator);
  return {
    label: String(channels),
    reason: `Solved after reserving ${formatBytes(reserved)} for the output row. The relationship is not a fixed 4x ratio because this reservation is independent of input channels.`,
  };
}

function renderRedesign(root, analysis, state, actions) {
  root.replaceChildren();
  if (!analysis) {
    root.append(emptyPanel("Redesign unavailable", "Run a TFLite static audit before creating an isolated projection."));
    return;
  }
  if (analysis.format !== "tflite" || !analysis.block_inventory?.blocks?.length) {
    root.append(emptyPanel(
      "Redesign not assessed for this format",
      "The current projection engine is TFLite-only and requires a WASM-derived block inventory. No approximate ONNX projection is emitted.",
    ));
    return;
  }
  ensureRedesignState(analysis, state);
  const banner = element("div", "xr-redesign-banner");
  banner.append(
    element("strong", "", "PROJECTED_UNTRAINED"),
    element("span", "", "Projection only. Source bytes and bound audit evidence remain unchanged; runtime, accuracy, safety, and regulatory equivalence are not asserted."),
  );
  const source = document.createElement("details");
  source.className = "xr-source-binding";
  const sourceSummary = document.createElement("summary");
  sourceSummary.append(
    element("span", "", "Bound source"),
    element("strong", "", analysis.filename || "unnamed"),
    element("em", "", analysis.target_profile?.label || analysis.target_profile?.id || "target unbound"),
  );
  const sourceLedger = element("div", "xr-source-ledger");
  sourceLedger.append(
    ledgerRow("Source SHA-256", analysis.model_sha256 || "hash pending"),
    ledgerRow("Target profile SHA-256", analysis.target_profile?.profile_sha256 || "hash unavailable"),
    ledgerRow("Isolation", "Loaded source bytes, bound analysis, findings, and reports remain unchanged"),
  );
  source.append(sourceSummary, sourceLedger);

  const nodeStage = element("section", "xr-redesign-node-stage");
  const nodeLegend = element("div", "xr-redesign-node-legend");
  const propagationLegend = element("div", "xr-redesign-legend-group");
  propagationLegend.append(element("strong", "", "Border / change scope"));
  for (const [tone, label] of [
    ["direct", "Direct edit"],
    ["propagated", "Auto-propagated contract"],
    ["global", "Global projection"],
    ["unchanged", "Unchanged"],
  ]) {
    const item = element("span");
    item.append(element("i", tone), document.createTextNode(label));
    propagationLegend.append(item);
  }
  const contractLegend = element("div", "xr-redesign-legend-group");
  contractLegend.append(element("strong", "", "Fill / projected status"));
  for (const [tone, label] of [
    ["issue", "Issue remains"],
    ["watch", "Watch remains"],
    ["satisfied", "Satisfied"],
    ["conditional", "Conditional proof"],
    ["unassessed", "Not assessed"],
    ["blocked", "Projection blocked"],
  ]) {
    const item = element("span");
    item.append(element("i", tone), document.createTextNode(label));
    contractLegend.append(item);
  }
  nodeLegend.append(propagationLegend, contractLegend);
  const nodeHost = element("div", "xr-redesign-node-host");
  nodeStage.append(nodeLegend, nodeHost);

  const selectedOp = analysis.ops?.find((op) => op.index === state.selectedRedesignOpIndex);
  const selectedContract = selectedOp
    ? getRedesignContractState(analysis, state.projection, selectedOp.index)
    : null;
  const statusDock = renderRedesignStatusDock(analysis, state, selectedOp, selectedContract);
  const workbench = element("div", "xr-redesign-workbench");
  const editor = element("aside", "xr-redesign-editor");
  const editorHead = element("div", "xr-editor-head");
  const editorTitle = element("div");
  editorTitle.append(
    element("span", "xr-kicker", selectedOp
      ? `SELECTED OPERATOR #${padOp(selectedOp.index)} ${selectedOp.name}`
      : "MODEL PROJECTION"),
    element("h3", "", "Scenario controls"),
  );
  const runState = element("span", `xr-projection-run-state ${projectionRunTone(state)}`, projectionRunLabel(state));
  runState.setAttribute("aria-live", "polite");
  editorHead.append(editorTitle, runState);
  editor.append(
    editorHead,
  );
  const globalGroup = element("fieldset", "xr-control-group");
  globalGroup.append(element("legend", "", "Model-wide projection"));
  const global = element("div", "xr-form-grid");
  const sourceInput = (analysis.inputs || []).find((tensor) => tensor.shape?.length === 4);
  const height = numberField("Input height", state.request.input_height, 1, 8192, 1, {
    key: "input_height",
    source: sourceInput?.shape?.[1],
  });
  const width = numberField("Input width", state.request.input_width, 1, 8192, 1, {
    key: "input_width",
    source: sourceInput?.shape?.[2],
  });
  const multiplier = numberField("Width multiplier", state.request.width_multiplier, 0.25, 2, 0.05, {
    key: "width_multiplier",
    source: 1,
  });
  const dtype = selectField("Storage projection", [
    ["source", "Source storage"],
    ["int8", "INT8 storage target"],
    ["float32", "FLOAT32 storage target"],
  ], state.request.activation_dtype, "activation_dtype");
  for (const [field, key, parser, eventName] of [
    [height.input, "input_height", (value) => Number(value), "input"],
    [width.input, "input_width", (value) => Number(value), "input"],
    [multiplier.input, "width_multiplier", (value) => Number(value), "input"],
    [dtype.input, "activation_dtype", (value) => value, "change"],
  ]) {
    field.addEventListener(eventName, () => {
      if (field instanceof HTMLInputElement && (!field.value || !field.validity.valid)) return;
      state.request[key] = parser(field.value);
      actions.queueProjection?.();
    });
  }
  global.append(height.root, width.root, multiplier.root, dtype.root);
  globalGroup.append(global);
  editor.append(globalGroup);

  const blockSelect = selectField(
    "Block editor",
    analysis.block_inventory.blocks.map((block) => [block.block_id, `${block.block_id} / ${block.display_name}`]),
    state.selectedRedesignBlockId,
    "selected_block",
  );
  blockSelect.input.addEventListener("change", () => {
    state.selectedRedesignBlockId = blockSelect.input.value;
    const selected = analysis.block_inventory.blocks.find((item) => item.block_id === state.selectedRedesignBlockId);
    state.selectedRedesignOpIndex = selected?.op_indices?.[0] ?? state.selectedRedesignOpIndex;
    renderRedesign(root, analysis, state, actions);
  });
  const blockGroup = element("fieldset", "xr-control-group");
  blockGroup.append(element("legend", "", "Scenario block editor"), blockSelect.root);
  const block = analysis.block_inventory.blocks.find((item) => item.block_id === state.selectedRedesignBlockId)
    || analysis.block_inventory.blocks[0];
  const owningBlock = selectedOp ? analysis.block_inventory.blocks.find((item) => item.op_indices?.includes(selectedOp.index)) : null;
  const editContext = element("p", "xr-edit-context");
  if (!selectedOp) {
    editContext.textContent = "No operator is selected. Controls apply to the explicitly selected semantic block.";
  } else if (owningBlock?.block_id === block?.block_id) {
    editContext.textContent = `Operator #${padOp(selectedOp.index)} belongs to ${owningBlock.block_id}. Controls edit that semantic block; the operator is not edited independently.`;
  } else if (owningBlock) {
    editContext.textContent = `Operator #${padOp(selectedOp.index)} belongs to ${owningBlock.block_id}; controls currently apply to ${block?.block_id || "the selected block"}.`;
  } else {
    editContext.textContent = `Operator #${padOp(selectedOp.index)} is not directly editable. Controls apply to ${block?.block_id || "the explicitly selected block"}.`;
  }
  blockGroup.append(editContext);
  if (block) {
    const edit = redesignEditFor(state, block.block_id);
    const fields = element("div", "xr-form-grid");
    const outputChannels = numberField("Output channels", edit.output_channels ?? block.channels?.output ?? 1, 1, 8192, 1, {
      key: "output_channels",
      source: block.channels?.output,
    });
    const expandRatio = numberField("Expand ratio", edit.expand_ratio ?? block.params?.expand_ratio ?? 1, 1, 16, 0.25, {
      key: "expand_ratio",
      source: block.params?.expand_ratio,
    });
    const repeat = numberField("Repeat", edit.repeat ?? 1, 1, 8, 1, {
      key: "repeat",
      source: 1,
    });
    const kernel = numberField("Kernel", edit.kernel_size ?? block.params?.kernel_h ?? 1, 1, 7, 2, {
      key: "kernel_size",
      source: block.params?.kernel_h,
    });
    const bindings = [
      [outputChannels.input, "output_channels", block.channels?.output],
      [expandRatio.input, "expand_ratio", block.params?.expand_ratio],
      [repeat.input, "repeat", 1],
      [kernel.input, "kernel_size", block.params?.kernel_h],
    ];
    for (const [input, key, original] of bindings) {
      input.addEventListener("input", () => {
        if (!input.value || !input.validity.valid) return;
        persistRedesignEdit(state, edit);
        const value = Number(input.value);
        if (Math.abs(value - Number(original ?? value)) < 1e-9) delete edit[key];
        else edit[key] = value;
        compactRedesignEdits(state);
        actions.queueProjection?.();
      });
    }
    fields.append(outputChannels.root, expandRatio.root, repeat.root, kernel.root);
    blockGroup.append(
      fields,
      element("p", "xr-note", `${block.block_type.replaceAll("_", " ")} / ${block.extraction?.method} ${block.extraction?.confidence}. Repeat is constrained to stride-1 residual blocks with equal input/output channels.`),
    );
  }
  editor.append(blockGroup);
  const commands = element("div", "xr-command-row");
  const run = button(state.running ? "Calculating..." : "Recalculate now", "primary-action");
  run.dataset.redesignAction = "run";
  run.disabled = state.running || !state.dirty;
  run.addEventListener("click", actions.runProjection);
  const resetBlock = button("Reset block", "secondary-action");
  resetBlock.disabled = !state.request.block_edits.some((item) => item.block_id === state.selectedRedesignBlockId);
  resetBlock.addEventListener("click", actions.resetSelectedBlock);
  const reset = button("Reset all", "secondary-action");
  reset.dataset.redesignAction = "reset-all";
  reset.disabled = activeScenarioChangeCount(analysis, state) === 0;
  reset.addEventListener("click", actions.resetProjection);
  commands.append(run, resetBlock, reset);
  editor.append(commands);

  const result = element("section", "xr-redesign-result");
  if (state.error) {
    result.append(emptyPanel("Projection rejected", state.error));
  } else if (!state.projection) {
    result.append(emptyPanel("Projection not run", "Change a control or run the deterministic WASM projection."));
  } else {
    result.append(renderProjectionResult(state.projection, analysis, state, actions));
  }
  const scenarioLab = renderScenarioLab(analysis, state, actions);
  workbench.append(editor, nodeStage);
  root.append(banner, source, statusDock, workbench, scenarioLab, result);
  actions.mountRedesignNode?.(nodeHost);
}

function renderScenarioLab(analysis, state, actions) {
  const section = element("section", "xr-scenario-lab");
  const head = element("div", "xr-section-head compact");
  const title = element("div");
  title.append(
    element("span", "xr-kicker", "SCENARIO DECISION SPACE"),
    element("h3", "", "Save, compare, and materialize a structure plan"),
    element("p", "", "Saved scenarios remain bound to this source SHA and target for the current browser session. Pareto retention is a structural proxy, never an accuracy claim."),
  );
  const commands = element("div", "xr-command-row");
  const save = button("Save current", "secondary-action");
  save.disabled = !state.projection || state.dirty || state.running;
  save.addEventListener("click", actions.saveScenario);
  const exportScenarios = button("Export scenarios", "secondary-action");
  exportScenarios.disabled = !state.savedScenarios.length;
  exportScenarios.addEventListener("click", actions.exportScenarios);
  const pareto = button(state.paretoRunning ? "Exploring..." : "Explore Pareto", "secondary-action");
  pareto.disabled = state.paretoRunning || state.running || state.dirty || !state.projection || !actions.paretoAvailable;
  pareto.addEventListener("click", actions.runPareto);
  const implementation = button("Export structure code", "primary-action");
  implementation.disabled = !state.projection || state.dirty || state.running
    || !state.projection?.implementation_plan?.exportable;
  implementation.title = "Export weight-free PyTorch/Keras structure code and the bound implementation ledger.";
  implementation.addEventListener("click", () => actions.exportImplementation?.());
  commands.append(save, exportScenarios, pareto, implementation);
  head.append(title, commands);
  section.append(head);

  const saved = element("div", "xr-scenario-section");
  const savedHead = element("div", "xr-subsection-head");
  savedHead.append(
    element("h4", "", "Scenario comparison"),
    element("span", "", `${formatNumber(state.savedScenarios.length)} saved`),
  );
  saved.append(savedHead);
  const records = [];
  if (state.projection && !state.dirty) {
    records.push({
      scenarioId: scenarioFingerprint(state.request),
      label: "Current",
      request: state.request,
      projection: state.projection,
      current: true,
    });
  }
  records.push(...state.savedScenarios.filter((record) => !records.some((current) => current.scenarioId === record.scenarioId)));
  if (records.length) saved.append(scenarioComparisonTable(records, actions));
  else saved.append(element("p", "xr-empty", "Run and save a projection to compare scenarios without changing the bound source audit."));
  section.append(saved);

  const paretoSection = element("div", "xr-scenario-section");
  const paretoHead = element("div", "xr-subsection-head");
  paretoHead.append(
    element("h4", "", "Pareto candidates"),
    element("span", "", state.pareto
      ? `${formatNumber(state.pareto.frontier_candidate_count)} frontier / ${formatNumber(state.pareto.evaluated_candidate_count)} evaluated`
      : "not explored"),
  );
  paretoSection.append(paretoHead);
  if (state.paretoError) paretoSection.append(element("p", "xr-empty error", state.paretoError));
  else if (state.paretoRunning) paretoSection.append(element("p", "xr-empty", "Projecting the deterministic resolution and width grid in WASM..."));
  else if (state.pareto?.candidates?.length) {
    paretoSection.append(paretoCandidateTable(state.pareto, actions));
    paretoSection.append(element("p", "xr-note", state.pareto.interpretation_boundary || ""));
  } else {
    paretoSection.append(element("p", "xr-empty", "Explore candidates to identify non-dominated structural tradeoffs under the bound target profile."));
  }
  section.append(paretoSection);
  return section;
}

function scenarioComparisonTable(records, actions) {
  const wrap = element("div", "xr-table-wrap");
  const table = element("table", "xr-scenario-table");
  const head = element("thead");
  const headRow = element("tr");
  for (const label of ["Scenario", "Input", "Width", "Storage", "MACs", "Latency", "Parameters", "Status", "Action"]) {
    headRow.append(element("th", "", label));
  }
  head.append(headRow);
  const body = element("tbody");
  for (const record of records) {
    const row = element("tr", record.current ? "current" : "");
    const projected = record.projection?.metrics?.projected || {};
    const actionsCell = element("td", "xr-scenario-actions");
    if (!record.current) {
      const apply = button("Apply", "secondary-action");
      apply.addEventListener("click", () => actions.applyScenario(record.request));
      const remove = button("Remove", "secondary-action");
      remove.addEventListener("click", () => actions.deleteScenario(record.scenarioId));
      actionsCell.append(apply, remove);
    } else {
      actionsCell.textContent = "active";
    }
    row.append(
      element("th", "", `${record.label} / ${record.scenarioId.slice(0, 8)}`),
      element("td", "", `${record.request.input_height}x${record.request.input_width}`),
      element("td", "", `${Number(record.request.width_multiplier).toFixed(3)}x`),
      element("td", "", record.request.activation_dtype),
      element("td", "", metricValue(projected.macs)),
      element("td", "", projected.modeled_latency_ms == null ? "N/A" : `${Number(projected.modeled_latency_ms).toFixed(3)} ms`),
      element("td", "", metricValue(projected.parameter_elements)),
      element("td", "", record.projection?.status || "not assessed"),
      actionsCell,
    );
    body.append(row);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

function paretoCandidateTable(search, actions) {
  const container = element("div", "xr-pareto-results");
  const candidates = (search.candidates || []).filter((candidate) => candidate.pareto_optimal);
  container.append(paretoRowsTable(candidates.slice(0, 8), actions));
  if (candidates.length > 8) {
    const more = document.createElement("details");
    more.className = "xr-more-conditions";
    more.append(element("summary", "", `${formatNumber(candidates.length - 8)} more frontier candidates`));
    more.append(paretoRowsTable(candidates.slice(8), actions));
    container.append(more);
  }
  return container;
}

function paretoRowsTable(candidates, actions) {
  const wrap = element("div", "xr-table-wrap");
  const table = element("table", "xr-scenario-table xr-pareto-table");
  const head = element("thead");
  const headRow = element("tr");
  for (const label of ["Candidate", "Input", "Width", "Retention proxy", "Latency", "MACs", "Parameters", "Peak live", "Action"]) {
    headRow.append(element("th", "", label));
  }
  head.append(headRow);
  const body = element("tbody");
  for (const candidate of candidates) {
    const row = element("tr", "pareto");
    const action = element("td", "xr-scenario-actions");
    const apply = button("Apply", "secondary-action");
    apply.addEventListener("click", () => actions.applyScenario(candidate.request, { preservePareto: true }));
    action.append(apply);
    row.append(
      element("th", "", candidate.candidate_id),
      element("td", "", `${candidate.request.input_height}x${candidate.request.input_width}`),
      element("td", "", `${Number(candidate.request.width_multiplier).toFixed(3)}x`),
      element("td", "", `${(Number(candidate.retained_structure_proxy) * 100).toFixed(1)}%`),
      element("td", "", `${Number(candidate.modeled_latency_ms).toFixed(3)} ms`),
      element("td", "", formatNumber(candidate.macs)),
      element("td", "", formatNumber(candidate.parameter_elements)),
      element("td", "", candidate.peak_live_activation_bytes == null ? "N/A" : formatBytes(candidate.peak_live_activation_bytes)),
      action,
    );
    body.append(row);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

function renderRedesignStatusDock(analysis, state, selectedOp, selectedContract) {
  const dock = element("section", `xr-redesign-status-dock${state.dirty ? " pending" : ""}`);
  dock.setAttribute("aria-label", "Current redesign scenario status");
  const projection = state.projection;
  const source = projection?.metrics?.source || {};
  const projected = projection?.metrics?.projected || {};
  const delta = projection?.metrics?.delta || {};
  const contractCounts = summarizeRedesignContracts(analysis, projection);
  const scenarioChanges = activeScenarioChangeCount(analysis, state);
  const identityCells = [
    statusDockCell(
      selectedOp ? `#${padOp(selectedOp.index)} ${selectedOp.name}` : "No operator selected",
      selectedContract?.label || "Not assessed",
      selectedContract?.id || "unassessed",
      `${contractCounts.issue + contractCounts.blocked} issue / ${contractCounts.watch} watch / ${contractCounts.satisfied} satisfied`,
    ),
    statusDockCell(
      "Active scenario",
      state.dirty ? "Updating" : `${scenarioChanges} change${scenarioChanges === 1 ? "" : "s"}`,
      state.dirty ? "pending" : scenarioChanges ? "changed" : "neutral",
      projection?.status || "Projection not run",
    ),
  ];
  if (projection) {
    dock.append(
      ...identityCells,
      statusDockMetric("MACs", source.macs, projected.macs, delta.mac_percent, formatNumber),
      statusDockMetric("Modeled latency", source.modeled_latency_ms, projected.modeled_latency_ms, delta.modeled_latency_percent, (value) => `${Number(value || 0).toFixed(3)} ms`),
      statusDockMetric("Max logical L1", source.l1_max_ratio, projected.l1_max_ratio, null, ratioText),
      statusDockMetric("Predicted breaks", source.predicted_break_count, projected.predicted_break_count, null, formatNumber),
    );
  } else {
    const pending = statusDockCell("Projection metrics", "Not calculated", "neutral", "Change a control to run the deterministic WASM projection. Source metrics remain in the bound audit.");
    pending.classList.add("projection-placeholder");
    dock.append(...identityCells, pending);
  }
  return dock;
}

function statusDockCell(label, value, tone, detail) {
  const cell = element("div", `xr-status-cell ${tone || "neutral"}`);
  cell.append(
    element("span", "", label),
    element("strong", "", String(value ?? "N/A")),
    element("small", "", detail || ""),
  );
  return cell;
}

function statusDockMetric(label, before, after, delta, formatter) {
  const assessed = before != null && after != null;
  const changed = assessed && Math.abs(Number(after) - Number(before)) > 1e-9;
  const detail = assessed
    ? `${formatter(before)} -> ${formatter(after)}`
    : "Not assessed";
  const value = !assessed
    ? "N/A"
    : delta != null
      ? formatSignedPercent(delta)
      : changed
        ? "Changed"
        : "Same";
  return statusDockCell(label, value, changed ? deltaTone(delta ?? Number(after) - Number(before)) : "neutral", detail);
}

function activeScenarioChangeCount(analysis, state) {
  const sourceInput = (analysis.inputs || []).find((tensor) => tensor.shape?.length === 4);
  let count = state.request.block_edits.reduce(
    (sum, edit) => sum + Object.keys(edit).filter((key) => key !== "block_id").length,
    0,
  );
  if (Number(state.request.input_height) !== Number(sourceInput?.shape?.[1] || 1)) count += 1;
  if (Number(state.request.input_width) !== Number(sourceInput?.shape?.[2] || 1)) count += 1;
  if (Math.abs(Number(state.request.width_multiplier) - 1) > 1e-9) count += 1;
  if (state.request.activation_dtype !== "source") count += 1;
  return count;
}

function formatSignedPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  if (Math.abs(number) < 0.005) return "Same";
  return `${number > 0 ? "+" : ""}${number.toFixed(1)}%`;
}

function projectionRunLabel(state) {
  if (state.running) return "Calculating";
  if (state.dirty) return "Update queued";
  if (state.error) return "Rejected";
  if (state.projection) return "Current";
  return "Not run";
}

function projectionRunTone(state) {
  if (state.running || state.dirty) return "pending";
  if (state.error) return "error";
  if (state.projection) return "current";
  return "idle";
}

function renderProjectionResult(projection, analysis, state, actions) {
  const fragment = document.createDocumentFragment();
  const head = element("div", "xr-section-head compact");
  const title = element("div");
  title.append(
    element("span", "xr-kicker", `${projection.status} / ${projection.projection_coverage?.status || "coverage unknown"}`),
    element("h3", "", "Source vs projected structure"),
    element("p", "", projection.interpretation_boundary || ""),
  );
  const commands = element("div", "xr-command-row");
  const json = button("JSON", "secondary-action");
  json.addEventListener("click", () => actions.exportJson?.(projection));
  const markdown = button("Markdown", "secondary-action");
  markdown.addEventListener("click", () => actions.exportMarkdown?.(projection));
  commands.append(json, markdown);
  head.append(title, commands);
  const impact = projection.impact_summary || {};
  const impactMetrics = element("div", "xr-propagation-summary");
  impactMetrics.append(
    compactMetric("Direct edits", formatNumber(impact.direct_edit_op_count || 0)),
    compactMetric("Auto-propagated", formatNumber(impact.propagated_op_count || 0)),
    compactMetric("Changed edges", formatNumber(impact.changed_edge_count || 0)),
    compactMetric("Unresolved", formatNumber(impact.unresolved_contract_count || 0)),
  );
  const implementation = renderImplementationHandoff(projection, state, actions);

  const tableWrap = element("div", "xr-table-wrap");
  const table = element("table", "xr-projection-table");
  const thead = element("thead");
  const tr = element("tr");
  for (const label of ["Metric", "Source", "Projected", "Delta", "Evidence"]) tr.append(element("th", "", label));
  thead.append(tr);
  const tbody = element("tbody");
  const source = projection.metrics?.source || {};
  const projected = projection.metrics?.projected || {};
  const delta = projection.metrics?.delta || {};
  const rows = [
    ["Operators", source.operator_count, projected.operator_count, null, "DERIVED"],
    ["MACs", source.macs, projected.macs, delta.mac_percent, "DERIVED_FROM_SCENARIO"],
    ["Operations", source.operations, projected.operations, delta.operations_percent, "DERIVED_FROM_SCENARIO"],
    ["Parameter elements", source.parameter_elements, projected.parameter_elements, delta.parameter_percent, "DERIVED_FROM_SCENARIO"],
    ["Modeled latency", source.modeled_latency_ms, projected.modeled_latency_ms, delta.modeled_latency_percent, "ESTIMATED_TARGET_PROFILE", (value) => `${Number(value || 0).toFixed(3)} ms`],
    ["Max logical L1", source.l1_max_ratio, projected.l1_max_ratio, null, "DERIVED_LOGICAL_PAYLOAD", ratioText],
    ["L1 watch count", source.l1_watch_count, projected.l1_watch_count, null, "DERIVED_LOGICAL_PAYLOAD"],
    ["Peak live activation", source.peak_live_activation_bytes, projected.peak_live_activation_bytes, delta.peak_live_activation_percent, "DERIVED_WHEN_TOPOLOGY_UNCHANGED", formatBytes],
    ["Arena projection", source.arena_bytes, projected.arena_bytes, delta.arena_percent, "DERIVED_WHEN_TOPOLOGY_UNCHANGED", formatBytes],
    ["Predicted breaks", source.predicted_break_count, projected.predicted_break_count, null, projected.delegation_evidence_class || "NOT_ASSESSABLE"],
  ];
  for (const [label, before, after, change, evidence, formatter = formatNumber] of rows) {
    const line = element("tr");
    const delta = change == null
      ? before != null && after != null && Math.abs(Number(after) - Number(before)) <= 1e-9
        ? "Same"
        : "N/A"
      : formatSignedPercent(change);
    const evidenceCell = element("td", "xr-evidence-cell", evidenceLabel(evidence));
    evidenceCell.title = evidence;
    line.append(
      element("th", "", label),
      element("td", "", before == null ? "N/A" : formatter(before)),
      element("td", "", after == null ? "N/A" : formatter(after)),
      element("td", deltaTone(change), delta),
      evidenceCell,
    );
    tbody.append(line);
  }
  table.append(thead, tbody);
  tableWrap.append(table);
  const technicalLedger = document.createElement("details");
  technicalLedger.className = "xr-technical-ledger";
  technicalLedger.append(element("summary", "", "Full metric and evidence ledger"), tableWrap);

  const constraints = element("div", "xr-constraint-list");
  const constraintHead = element("div", "xr-subsection-head");
  constraintHead.append(
    element("h4", "", "Projection conditions"),
    element("span", "", `${formatNumber((projection.constraints || []).length)} emitted`),
  );
  constraints.append(constraintHead);
  if (!(projection.constraints || []).length) constraints.append(element("p", "xr-empty", "No structural constraint violation was emitted."));
  const orderedConstraints = [...(projection.constraints || [])].sort(
    (left, right) => constraintSeverityRank(left.severity) - constraintSeverityRank(right.severity)
      || String(left.scope || "").localeCompare(String(right.scope || "")),
  );
  for (const item of orderedConstraints.slice(0, 4)) {
    const row = element("article", `xr-constraint ${constraintTone(item.severity)}`);
    row.append(
      element("strong", "", `${item.code} / ${item.scope}`),
      element("p", "", item.detail),
      element("small", "", item.evidence_class),
    );
    constraints.append(row);
  }
  if (orderedConstraints.length > 4) {
    const remaining = document.createElement("details");
    remaining.className = "xr-more-conditions";
    remaining.append(element("summary", "", `${formatNumber(orderedConstraints.length - 4)} more conditions`));
    for (const item of orderedConstraints.slice(4)) {
      const row = element("article", `xr-constraint ${constraintTone(item.severity)}`);
      row.append(
        element("strong", "", `${item.code} / ${item.scope}`),
        element("p", "", item.detail),
        element("small", "", item.evidence_class),
      );
      remaining.append(row);
    }
    constraints.append(remaining);
  }
  const diff = element("div", "xr-diff-list");
  const diffHead = element("div", "xr-subsection-head");
  diffHead.append(
    element("h4", "", "Block diff"),
    element("span", "", `${formatNumber((projection.block_diffs || []).length)} changed`),
  );
  diff.append(diffHead);
  if (!(projection.block_diffs || []).length) diff.append(element("p", "xr-empty", "No structural change from the source request."));
  for (const item of projection.block_diffs || []) {
    const row = element("article", "xr-diff-row");
    row.append(
      element("strong", "", `${item.block_id} / ${item.display_name}`),
      element("p", "", (item.changes || []).join(" / ")),
      element("small", "", `MAC ${formatNumber(item.source_macs || 0)} -> ${formatNumber(item.projected_macs || 0)} / L1 ${ratioText(item.source_l1_max_ratio)} -> ${ratioText(item.projected_l1_max_ratio)}`),
    );
    diff.append(row);
  }
  const footer = element("p", "xr-note", `Loaded source bytes unchanged: ${projection.source?.loaded_source_bytes_unchanged ? "verified in session" : "FAILED"}. Framework code export is intentionally unavailable because a deployment artifact does not encode a reversible training graph.`);
  fragment.append(head, impactMetrics, implementation, constraints, diff, technicalLedger, footer);
  return fragment;
}

function renderImplementationHandoff(projection, state, actions) {
  const plan = projection.implementation_plan;
  const section = element("section", "xr-implementation-handoff");
  const head = element("div", "xr-subsection-head");
  head.append(
    element("h4", "", "Implementation handoff"),
    element("span", "", plan ? `${formatNumber(plan.mapped_source_layer_count)}/${formatNumber(plan.nodes?.length || 0)} source-like mappings` : "not available"),
  );
  section.append(head);
  if (!plan) {
    section.append(element("p", "xr-empty", "This projection predates the WASM implementation-plan contract."));
    return section;
  }
  const summary = element("div", "xr-propagation-summary");
  summary.append(
    compactMetric("Exact structure", formatNumber(plan.exact_codegen_op_count || 0)),
    compactMetric("Scaffold", formatNumber(plan.scaffold_codegen_op_count || 0)),
    compactMetric("Unsupported / repeat", `${formatNumber(plan.unsupported_codegen_op_count || 0)} / ${formatNumber(plan.non_materialized_repeat_edit_count || 0)}`),
    compactMetric("Frameworks", (plan.framework_targets || []).join(" / ")),
  );
  const commands = element("div", "xr-command-row");
  const exportCode = button("Export structure code", "primary-action");
  exportCode.disabled = !plan.exportable || state.dirty || state.running;
  exportCode.title = plan.exportable
    ? "Export weight-free PyTorch/Keras structure code."
    : "Blocked because repeat edits are not materialized as projected tensor nodes.";
  exportCode.addEventListener("click", () => actions.exportImplementation?.(projection));
  commands.append(exportCode);
  const ledger = document.createElement("details");
  ledger.className = "xr-technical-ledger";
  ledger.append(element("summary", "", "Source-layer and code-generation ledger"));
  const wrap = element("div", "xr-table-wrap");
  const table = element("table", "xr-scenario-table xr-implementation-table");
  const tableHead = element("thead");
  const header = element("tr");
  for (const label of ["Op", "Block", "Generated", "Artifact source-like path", "Evidence", "Codegen"]) {
    header.append(element("th", "", label));
  }
  tableHead.append(header);
  const body = element("tbody");
  for (const node of plan.nodes || []) {
    const row = element("tr");
    row.append(
      element("th", "", `#${padOp(node.op_index)} ${node.op_name}`),
      element("td", "", node.block_id || "unbound"),
      element("td", "", node.generated_symbol),
      element("td", "", node.source_layer_ref || "not available"),
      element("td", "", node.source_layer_evidence_class),
      element("td", "", node.codegen_status),
    );
    body.append(row);
  }
  table.append(tableHead, body);
  wrap.append(table);
  ledger.append(wrap);
  section.append(summary, commands, ledger, element("p", "xr-note", plan.interpretation_boundary || ""));
  return section;
}

function evidenceLabel(value) {
  return {
    DERIVED: "Derived",
    DERIVED_FROM_SCENARIO: "Scenario-derived",
    DERIVED_LOGICAL_PAYLOAD: "Logical payload",
    DERIVED_WHEN_TOPOLOGY_UNCHANGED: "Topology-bound",
    ESTIMATED_TARGET_PROFILE: "Target estimate",
    PREDICTED_SOURCE_ARTIFACT: "Source prediction",
    NOT_ASSESSABLE: "Not assessable",
  }[value] || String(value || "Not assessed").replaceAll("_", " ").toLowerCase();
}

function constraintSeverityRank(value) {
  return { error: 0, warn: 1, warning: 1, info: 2 }[String(value || "").toLowerCase()] ?? 3;
}

function constraintTone(value) {
  const normalized = String(value || "info").toLowerCase();
  return normalized === "warning" ? "warn" : normalized;
}

function ensureRedesignState(analysis, state) {
  const input = (analysis.inputs || []).find((tensor) => tensor.shape?.length === 4);
  const targetProfileSha = analysis.target_profile?.profile_sha256 || analysis.target_profile?.id || "";
  if (!state.boundSha
    || state.boundSha !== analysis.model_sha256
    || state.boundTargetProfileSha !== targetProfileSha) {
    const priorBinding = Boolean(state.boundSha || state.boundTargetProfileSha);
    state.boundSha = analysis.model_sha256;
    state.boundTargetProfileSha = targetProfileSha;
    state.request = {
      schema: "deepbom.redesign_request.v1",
      source_sha256: analysis.model_sha256 || "",
      input_height: Number(input?.shape?.[1] || 1),
      input_width: Number(input?.shape?.[2] || 1),
      width_multiplier: 1,
      activation_dtype: "source",
      block_edits: [],
    };
    state.selectedRedesignBlockId = analysis.block_inventory.blocks[0]?.block_id || "";
    state.selectedRedesignOpIndex = analysis.block_inventory.blocks[0]?.op_indices?.[0]
      ?? analysis.ops?.[0]?.index
      ?? 0;
    state.projection = null;
    state.dirty = false;
    state.requestRevision = Number(state.requestRevision || 0) + 1;
    state.error = "";
    if (priorBinding) {
      state.savedScenarios = [];
      state.pareto = null;
      state.paretoError = "";
      state.paretoRevision = Number(state.paretoRevision || 0) + 1;
    }
  }
}

function redesignEditFor(state, blockId) {
  return state.request.block_edits.find((item) => item.block_id === blockId)
    || { block_id: blockId };
}

function persistRedesignEdit(state, edit) {
  if (!state.request.block_edits.includes(edit)) state.request.block_edits.push(edit);
}

function compactRedesignEdits(state) {
  state.request.block_edits = state.request.block_edits.filter((edit) => Object.keys(edit).some((key) => key !== "block_id"));
}

function projectionMarkdown(projection) {
  const source = projection.metrics?.source || {};
  const projected = projection.metrics?.projected || {};
  const lines = [
    "# DEEPBOM Redesign Projection",
    "",
    `- Schema: \`${projection.schema}\``,
    `- Projection status: **${projection.projection_status}**`,
    `- Assessment status: \`${projection.status}\``,
    `- Source artifact: \`${projection.source?.filename || "unknown"}\``,
    `- Source SHA-256: \`${projection.source?.sha256_before || "unknown"}\``,
    `- Loaded source bytes unchanged in session: ${projection.source?.loaded_source_bytes_unchanged ? "yes" : "NO"}`,
    `- Target: \`${projection.source?.target_id || "unknown"}\``,
    "",
    "## Projected Metrics",
    "",
    "| Metric | Source | Projected |",
    "|---|---:|---:|",
    `| MACs | ${formatNumber(source.macs || 0)} | ${formatNumber(projected.macs || 0)} |`,
    `| Operations | ${formatNumber(source.operations || 0)} | ${formatNumber(projected.operations || 0)} |`,
    `| Parameter elements | ${formatNumber(source.parameter_elements || 0)} | ${formatNumber(projected.parameter_elements || 0)} |`,
    `| Modeled latency | ${Number(source.modeled_latency_ms || 0).toFixed(3)} ms | ${Number(projected.modeled_latency_ms || 0).toFixed(3)} ms |`,
    `| Max logical L1 ratio | ${ratioText(source.l1_max_ratio)} | ${ratioText(projected.l1_max_ratio)} |`,
    `| Peak live activation | ${source.peak_live_activation_bytes == null ? "N/A" : formatBytes(source.peak_live_activation_bytes)} | ${projected.peak_live_activation_bytes == null ? "N/A" : formatBytes(projected.peak_live_activation_bytes)} |`,
    "",
    "## Modifications",
    "",
    ...(projection.block_diffs || []).flatMap((item) => [
      `### ${item.block_id} / ${item.display_name}`,
      "",
      ...(item.changes || []).map((change) => `- ${change}`),
      "",
    ]),
    "## Not Evaluated",
    "",
    ...(projection.not_evaluated || []).map((item) => `- ${item}`),
    "",
    "## Required Next Steps",
    "",
    ...(projection.required_next_steps || []).map((item) => `1. ${item}`),
    "",
    "## Interpretation Boundary",
    "",
    projection.interpretation_boundary || "",
    "",
  ];
  return lines.join("\n");
}

function emptyPanel(title, detail) {
  const panel = element("article", "xr-empty-panel");
  panel.append(element("h3", "", title), element("p", "", detail));
  return panel;
}

function compactMetric(label, value) {
  const metric = element("div", "xr-compact-metric");
  metric.append(element("span", "", label), element("strong", "", String(value ?? "N/A")));
  return metric;
}

function ledgerRow(label, value) {
  const row = element("div", "xr-ledger-row");
  row.append(element("span", "", label), element("strong", "", String(value ?? "N/A")));
  return row;
}

function panelHeading(title, status) {
  const head = element("div", "xr-panel-heading");
  head.append(element("h4", "", title), element("span", "", status));
  return head;
}

function payloadBar(label, value, total) {
  const row = element("div", "xr-payload-row");
  const head = element("div");
  head.append(element("strong", "", label), element("span", "", formatBytes(value || 0)));
  const track = element("span", "xr-payload-track");
  const fill = element("i");
  fill.style.width = `${total == null ? 0 : Math.max(1, Math.min(100, Number(value || 0) / total * 100))}%`;
  track.append(fill);
  row.append(head, track);
  return row;
}

function segmentGroup(label, options) {
  const group = element("div", "xr-segment-group");
  group.append(element("span", "", label));
  const controls = element("div");
  for (const option of options) {
    const control = button(option.label, option.active ? "active" : "");
    control.addEventListener("click", option.run);
    controls.append(control);
  }
  group.append(controls);
  return group;
}

function numberField(label, value, min, max, step, { key = "", source = null } = {}) {
  const root = element("label", "xr-field");
  const input = document.createElement("input");
  input.type = "number";
  input.value = String(value ?? min);
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  if (key) input.dataset.redesignField = key;
  root.append(element("span", "", label), input);
  if (source != null) root.append(element("small", "", `Source ${source}`));
  return { root, input };
}

function selectField(label, options, selected, key = "") {
  const root = element("label", "xr-field");
  const input = document.createElement("select");
  if (key) input.dataset.redesignField = key;
  for (const [value, text] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    option.selected = value === selected;
    input.append(option);
  }
  root.append(element("span", "", label), input);
  return { root, input };
}

function powersOfTwo(minimum, maximum) {
  const values = [];
  let value = 2 ** Math.floor(Math.log2(Math.max(1, minimum)));
  while (value <= maximum * 1.01) {
    values.push(value);
    value *= 2;
  }
  return values;
}

function cacheTone(value) {
  const ratio = Number(value || 0);
  if (ratio >= 5) return "critical";
  if (ratio >= 2) return "danger";
  if (ratio >= CACHE_WATCH_RATIO) return "warn";
  return "good";
}

function cacheStroke(value) {
  const tone = cacheTone(value);
  return {
    good: "#4fb39e",
    warn: "#f3b64e",
    danger: "#ef765f",
    critical: "#c96ee8",
  }[tone];
}

function stageColor(index) {
  const colors = ["#60a5fa", "#66c4b1", "#f3b64e", "#c96ee8", "#ef765f", "#9ab07d"];
  return colors[Math.abs(Number(index || 0)) % colors.length];
}

function cacheLegend(showProjection = false) {
  const legend = element("div", "xr-cache-legend");
  for (const [className, label] of [
    ["good", "<0.9x"],
    ["warn", "0.9-2x"],
    ["danger", "2-5x"],
    ["critical", ">=5x"],
  ]) {
    const item = element("span");
    item.append(element("i", className), document.createTextNode(label));
    legend.append(item);
  }
  if (showProjection) {
    const item = element("span");
    item.append(element("i", "projected"), document.createTextNode("Redesign projection"));
    legend.append(item);
  }
  return legend;
}

function deltaTone(value) {
  if (finite(value) == null) return "";
  if (Number(value) < -0.01) return "good";
  if (Number(value) > 0.01) return "warn";
  return "";
}

function captureRedesignFocus(root) {
  const active = document.activeElement;
  if (!root?.contains(active) || !(active instanceof HTMLInputElement || active instanceof HTMLSelectElement)) {
    return null;
  }
  return {
    key: active.dataset.redesignField || "",
    start: active instanceof HTMLInputElement ? active.selectionStart : null,
    end: active instanceof HTMLInputElement ? active.selectionEnd : null,
  };
}

function restoreRedesignFocus(root, focus) {
  if (!focus?.key) return;
  requestAnimationFrame(() => {
    const next = root?.querySelector(`[data-redesign-field="${CSS.escape(focus.key)}"]`);
    if (!(next instanceof HTMLInputElement || next instanceof HTMLSelectElement)) return;
    next.focus({ preventScroll: true });
    if (next instanceof HTMLInputElement && focus.start != null && focus.end != null) {
      next.setSelectionRange(focus.start, focus.end);
    }
  });
}

function updateProjectionActivity(root, state, analysis) {
  const status = root?.querySelector(".xr-projection-run-state");
  if (status) {
    status.className = `xr-projection-run-state ${projectionRunTone(state)}`;
    status.textContent = projectionRunLabel(state);
  }
  const dock = root?.querySelector(".xr-redesign-status-dock");
  dock?.classList.toggle("pending", state.running || state.dirty);
  const run = root?.querySelector('[data-redesign-action="run"]');
  if (run instanceof HTMLButtonElement) {
    run.disabled = state.running || !state.dirty;
    run.textContent = state.running ? "Calculating..." : "Recalculate now";
  }
  const reset = root?.querySelector('[data-redesign-action="reset-all"]');
  if (reset instanceof HTMLButtonElement) reset.disabled = activeScenarioChangeCount(analysis, state) === 0;
}

export function createExplorerRedesignController({
  blocksRoot,
  cacheRoot,
  redesignRoot,
  project,
  explorePareto = null,
  selectOp,
  openWorkspace,
  filenameForExport = (suffix) => suffix,
}) {
  const state = {
    selectedBlockId: "",
    blockSort: "execution",
    blockWarningsOnly: false,
    collapseStages: false,
    collapsedStageIds: new Set(),
    selectedCacheOp: null,
    l1Bytes: null,
    dtype: "source",
    filter: "all",
    cacheStage: "all",
    trajectory: true,
    blockColor: false,
    projectionOverlay: true,
    selectedRedesignBlockId: "",
    selectedRedesignOpIndex: 0,
    request: null,
    projection: null,
    running: false,
    pendingProjection: false,
    dirty: false,
    requestRevision: 0,
    error: "",
    boundSha: "",
    boundTargetProfileSha: "",
    savedScenarios: [],
    pareto: null,
    paretoRunning: false,
    paretoError: "",
    paretoRevision: 0,
  };
  let analysis = null;
  let projectionTimer = null;
  let redesignNodeController = null;

  const actions = {
    paretoAvailable: typeof explorePareto === "function",
    selectOp,
    openRedesign(blockId) {
      ensureRedesignState(analysis, state);
      state.selectedRedesignBlockId = blockId;
      state.selectedRedesignOpIndex = analysis.block_inventory.blocks
        .find((block) => block.block_id === blockId)?.op_indices?.[0]
        ?? state.selectedRedesignOpIndex;
      openWorkspace?.("redesign");
      renderRedesign(redesignRoot, analysis, state, actions);
      if (!state.projection && !state.running) actions.runProjection();
    },
    queueProjection() {
      state.dirty = true;
      state.pareto = null;
      state.paretoError = "";
      state.requestRevision += 1;
      updateProjectionActivity(redesignRoot, state, analysis);
      clearTimeout(projectionTimer);
      projectionTimer = setTimeout(() => actions.runProjection(), 360);
    },
    async runProjection() {
      if (!analysis || !state.request) return;
      if (state.running) {
        state.pendingProjection = true;
        return;
      }
      const revision = state.requestRevision;
      const request = structuredClone(state.request);
      clearTimeout(projectionTimer);
      state.running = true;
      state.pendingProjection = false;
      state.error = "";
      updateProjectionActivity(redesignRoot, state, analysis);
      let accepted = false;
      try {
        const projection = await project(request);
        if (revision === state.requestRevision) {
          state.projection = projection;
          state.dirty = false;
          accepted = true;
        } else {
          state.pendingProjection = true;
        }
      } catch (error) {
        if (revision === state.requestRevision) {
          state.error = error instanceof Error ? error.message : String(error);
          state.projection = null;
          state.dirty = false;
          accepted = true;
        } else {
          state.pendingProjection = true;
        }
      } finally {
        state.running = false;
        if (state.pendingProjection) {
          state.pendingProjection = false;
          clearTimeout(projectionTimer);
          projectionTimer = setTimeout(() => actions.runProjection(), 0);
          updateProjectionActivity(redesignRoot, state, analysis);
        } else if (accepted) {
          const focus = captureRedesignFocus(redesignRoot);
          renderRedesign(redesignRoot, analysis, state, actions);
          restoreRedesignFocus(redesignRoot, focus);
          if (cacheRoot) renderCache(cacheRoot, analysis, state, actions);
        }
      }
    },
    selectRedesignOp(opIndex, { focusEditor = false } = {}) {
      const numeric = Number(opIndex);
      const block = analysis?.block_inventory?.blocks?.find((item) =>
        item.op_indices?.includes(numeric),
      );
      state.selectedRedesignOpIndex = numeric;
      if (block) state.selectedRedesignBlockId = block.block_id;
      renderRedesign(redesignRoot, analysis, state, actions);
      if (focusEditor) {
        requestAnimationFrame(() => {
          const editor = redesignRoot?.querySelector(".xr-redesign-editor");
          editor?.scrollIntoView({ behavior: "smooth", block: "start" });
          editor?.querySelector("input")?.focus({ preventScroll: true });
        });
      }
    },
    mountRedesignNode(host) {
      if (!redesignNodeController) {
        redesignNodeController = createNodeViewController({
          mode: "redesign",
          onSelectOp: (opIndex) => actions.selectRedesignOp(opIndex),
          onEditNode: (opIndex) => actions.selectRedesignOp(opIndex, { focusEditor: true }),
        });
      }
      redesignNodeController.sync({
        root: host,
        analysis,
        projection: state.projection,
        selectedOpIndex: state.selectedRedesignOpIndex,
      });
    },
    resetProjection() {
      state.boundSha = "";
      state.boundTargetProfileSha = "";
      ensureRedesignState(analysis, state);
      state.pendingProjection = false;
      state.pareto = null;
      state.paretoError = "";
      renderRedesign(redesignRoot, analysis, state, actions);
      actions.runProjection();
    },
    resetSelectedBlock() {
      state.request.block_edits = state.request.block_edits.filter(
        (item) => item.block_id !== state.selectedRedesignBlockId,
      );
      state.dirty = true;
      state.pareto = null;
      state.paretoError = "";
      state.requestRevision += 1;
      renderRedesign(redesignRoot, analysis, state, actions);
      actions.runProjection();
    },
    saveScenario() {
      if (!state.projection || state.dirty || state.running) return;
      const request = canonicalScenario(state.request);
      const scenarioId = scenarioFingerprint(request);
      const existing = state.savedScenarios.findIndex((item) => item.scenarioId === scenarioId);
      const record = {
        scenarioId,
        label: existing >= 0 ? state.savedScenarios[existing].label : `S${state.savedScenarios.length + 1}`,
        request,
        projection: structuredClone(state.projection),
      };
      if (existing >= 0) state.savedScenarios.splice(existing, 1, record);
      else state.savedScenarios.push(record);
      renderRedesign(redesignRoot, analysis, state, actions);
    },
    deleteScenario(scenarioId) {
      state.savedScenarios = state.savedScenarios.filter((item) => item.scenarioId !== scenarioId);
      renderRedesign(redesignRoot, analysis, state, actions);
    },
    exportScenarios() {
      if (!analysis || !state.savedScenarios.length) return;
      const scenarioSet = buildRedesignScenarioSet({
        analysis,
        savedScenarios: state.savedScenarios,
        pareto: state.pareto,
      });
      downloadText(
        filenameForExport("redesign_scenarios.json"),
        `${JSON.stringify(scenarioSet, null, 2)}\n`,
        "application/json",
      );
    },
    applyScenario(request, { preservePareto = false } = {}) {
      state.request = structuredClone(canonicalScenario(request));
      state.projection = null;
      state.error = "";
      state.dirty = true;
      if (!preservePareto) {
        state.pareto = null;
        state.paretoError = "";
      }
      state.requestRevision += 1;
      renderRedesign(redesignRoot, analysis, state, actions);
      actions.runProjection();
    },
    async runPareto() {
      if (!analysis || !state.request || state.paretoRunning || typeof explorePareto !== "function") return;
      const revision = ++state.paretoRevision;
      state.paretoRunning = true;
      state.paretoError = "";
      renderRedesign(redesignRoot, analysis, state, actions);
      try {
        const result = await explorePareto(structuredClone(state.request));
        if (revision === state.paretoRevision) state.pareto = result;
      } catch (error) {
        if (revision === state.paretoRevision) {
          state.pareto = null;
          state.paretoError = error instanceof Error ? error.message : String(error);
        }
      } finally {
        if (revision === state.paretoRevision) {
          state.paretoRunning = false;
          renderRedesign(redesignRoot, analysis, state, actions);
        }
      }
    },
    exportImplementation(projection = state.projection) {
      if (!analysis || !projection || state.dirty || !projection.implementation_plan?.exportable) return;
      const files = buildRedesignImplementationFiles({
        analysis,
        projection,
        request: state.request,
      });
      downloadBlob(
        filenameForExport("redesign_structure_package.zip"),
        createZipBlob(files, { timestamp: new Date(1980, 0, 1, 0, 0, 0) }),
      );
    },
    exportJson(projection) {
      downloadText(
        filenameForExport("redesign-projection.json"),
        `${JSON.stringify(projection, null, 2)}\n`,
        "application/json",
      );
    },
    exportMarkdown(projection) {
      downloadText(
        filenameForExport("redesign-report.md"),
        projectionMarkdown(projection),
        "text/markdown",
      );
    },
  };

  return {
    resetInteractionState() {
      const projectionWasRunning = state.running;
      clearTimeout(projectionTimer);
      redesignNodeController?.resetInteractionState();
      redesignNodeController = null;
      state.requestRevision += 1;
      state.selectedBlockId = "";
      state.blockSort = "execution";
      state.blockWarningsOnly = false;
      state.collapseStages = false;
      state.collapsedStageIds.clear();
      state.selectedCacheOp = null;
      state.l1Bytes = null;
      state.dtype = "source";
      state.filter = "all";
      state.cacheStage = "all";
      state.trajectory = true;
      state.blockColor = false;
      state.projectionOverlay = true;
      state.selectedRedesignBlockId = "";
      state.selectedRedesignOpIndex = 0;
      state.request = null;
      state.projection = null;
      state.running = projectionWasRunning;
      state.pendingProjection = projectionWasRunning;
      state.dirty = false;
      state.error = "";
      state.boundSha = "";
      state.boundTargetProfileSha = "";
      state.savedScenarios = [];
      state.pareto = null;
      state.paretoRunning = false;
      state.paretoError = "";
      state.paretoRevision += 1;
      analysis = null;
      blocksRoot?.replaceChildren();
      cacheRoot?.replaceChildren();
      redesignRoot?.replaceChildren();
    },
    render(nextAnalysis) {
      analysis = nextAnalysis;
      if (!analysis) {
        blocksRoot?.replaceChildren();
        cacheRoot?.replaceChildren();
        redesignRoot?.replaceChildren();
        return;
      }
      if (blocksRoot) renderBlocks(blocksRoot, analysis, state, actions);
      if (cacheRoot) renderCache(cacheRoot, analysis, state, actions);
      if (redesignRoot) renderRedesign(redesignRoot, analysis, state, actions);
    },
    renderBlocks() {
      if (analysis && blocksRoot) renderBlocks(blocksRoot, analysis, state, actions);
    },
    renderCache() {
      if (analysis && cacheRoot) renderCache(cacheRoot, analysis, state, actions);
    },
    renderRedesign() {
      if (analysis && redesignRoot) renderRedesign(redesignRoot, analysis, state, actions);
    },
    selectBlock(blockId) {
      state.selectedBlockId = blockId;
      state.selectedRedesignBlockId = blockId;
      state.selectedRedesignOpIndex = analysis?.block_inventory?.blocks
        ?.find((block) => block.block_id === blockId)?.op_indices?.[0]
        ?? state.selectedRedesignOpIndex;
      if (analysis) {
        renderBlocks(blocksRoot, analysis, state, actions);
        renderRedesign(redesignRoot, analysis, state, actions);
      }
    },
    selectRedesignOp(opIndex) {
      actions.selectRedesignOp(opIndex);
    },
    state,
  };
}
