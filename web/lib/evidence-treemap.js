const MAP_WIDTH = 1000;
const MAP_HEIGHT = 560;

const OP_METRICS = Object.freeze({
  macs: {
    label: "MACs",
    unit: "MACs",
    evidenceClass: "DERIVED",
    value: (op) => finiteNonNegative(op?.macs),
  },
  traffic: {
    label: "Logical traffic",
    unit: "B",
    evidenceClass: "DERIVED",
    value: (op) => finiteNonNegative(op?.estimated_bytes),
  },
  time: {
    label: "Modeled steady time",
    unit: "us",
    evidenceClass: "ESTIMATED_TARGET_PROFILE",
    value: (op) => finiteNonNegative(op?.bottleneck_total_us),
  },
});

const TENSOR_METRICS = Object.freeze({
  serialized_bytes: {
    label: "Serialized payload",
    unit: "B",
    evidenceClass: "OBSERVED",
    value: (tensor) => finiteNonNegative(tensor?.byte_length ?? tensor?.buffer_data_length),
  },
});

function finiteNonNegative(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : null;
}

function stableText(value, fallback) {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function formatInteger(value) {
  return Math.round(Number(value || 0)).toLocaleString("en-US");
}

function formatMetric(value, unit) {
  if (unit === "B") {
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(2)} MiB`;
    if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
    return `${formatInteger(value)} B`;
  }
  if (unit === "us") return value >= 1000 ? `${(value / 1000).toFixed(2)} ms` : `${value.toFixed(2)} us`;
  return `${formatInteger(value)} ${unit}`;
}

function option(id, label) {
  return { id, label };
}

function buildBlockLookup(analysis) {
  const lookup = new Map();
  for (const block of analysis?.block_inventory?.blocks || []) {
    for (const opIndex of block.op_indices || []) {
      lookup.set(Number(opIndex), {
        id: stableText(block.block_id, `block-${opIndex}`),
        label: stableText(block.display_name, stableText(block.block_id, `Block ${opIndex}`)),
      });
    }
  }
  return lookup;
}

function buildStageLookup(analysis) {
  const lookup = new Map();
  const semantic = analysis?.block_inventory?.status === "assessed"
    ? analysis.block_inventory.stages || []
    : [];
  const stages = semantic.length ? semantic : analysis?.stages || [];
  for (const stage of stages) {
    const explicit = Array.isArray(stage.op_indices) ? stage.op_indices : [];
    const first = Number(stage.first_op);
    const last = Number(stage.last_op);
    const indices = explicit.length
      ? explicit
      : Number.isSafeInteger(first) && Number.isSafeInteger(last) && last >= first
        ? (analysis?.ops || []).map((op) => Number(op.index)).filter((index) => index >= first && index <= last)
        : [];
    for (const opIndex of indices) {
      lookup.set(Number(opIndex), {
        id: stableText(stage.stage_id, `stage-${stage.index}`),
        label: stableText(stage.display_name, stableText(stage.key, `Stage ${stage.index}`)),
      });
    }
  }
  return lookup;
}

function groupForOp(op, groupBy, stageLookup, blockLookup) {
  if (groupBy === "block") return blockLookup.get(Number(op.index)) || { id: "unassigned", label: "Unassigned block" };
  if (groupBy === "op_type") {
    const name = stableText(op.name, "UNKNOWN");
    return { id: `op-${name}`, label: name };
  }
  return stageLookup.get(Number(op.index)) || {
    id: `stage-${op.stage_index ?? "unassigned"}`,
    label: op.stage_key ? stableText(op.stage_key, "Unassigned stage") : `Stage ${op.stage_index ?? "unassigned"}`,
  };
}

function tensorNamespace(name) {
  const text = stableText(name, "unnamed");
  const head = text.split(/[/.]/, 1)[0];
  return head.length <= 28 ? head : `${head.slice(0, 25)}...`;
}

function groupForTensor(tensor, groupBy) {
  if (groupBy === "encoding") {
    const encoding = stableText(tensor.encoding_class, stableText(tensor.dtype, "UNKNOWN"));
    return { id: `encoding-${encoding}`, label: encoding.replaceAll("_", " ") };
  }
  if (groupBy === "namespace") {
    const namespace = tensorNamespace(tensor.name);
    return { id: `namespace-${namespace}`, label: namespace };
  }
  const dtype = stableText(tensor.dtype, "UNKNOWN");
  return { id: `dtype-${dtype}`, label: dtype };
}

function quantTone(op, signal) {
  if (signal?.tone === "risk" || op?.quant_risk === "risk") return "risk";
  if (signal || op?.quant_risk === "warn") return "warn";
  if (op?.quantization_state === "quantized_compute") return "good";
  if (op?.quantization_state && op.quantization_state !== "none") return "mixed";
  return "neutral";
}

function opTone(op, colorBy) {
  if (colorBy === "quantization") return quantTone(op, null);
  if (colorBy === "delegation") {
    if (op?.xnnpack_chain_break) return "risk";
    if (op?.xnnpack_chain_id != null && op.xnnpack_chain_id !== "" && Number(op.xnnpack_chain_id) >= 0) return "good";
    return "neutral";
  }
  if (op?.intensity_status === "not_assessed") return "neutral";
  if (op?.static_bound_guess === "compute-bound") return "cool";
  if (op?.static_bound_guess === "mixed") return "warn";
  if (op?.static_bound_guess === "memory-bound") return "violet";
  return "neutral";
}

function resourceLegend(scope, colorBy) {
  if (scope === "tensors") return [
    ["good", "block / integer storage"],
    ["cool", "floating storage"],
    ["warn", "review signal"],
    ["risk", "integrity risk"],
    ["neutral", "other / unassessed"],
  ];
  if (colorBy === "delegation") return [
    ["good", "conditionally delegatable"],
    ["risk", "predicted break"],
    ["neutral", "not conditionally delegated"],
  ];
  if (colorBy === "quantization") return quantLegend();
  return [
    ["cool", "high intensity"],
    ["warn", "mixed intensity"],
    ["violet", "low intensity"],
    ["neutral", "not assessed"],
  ];
}

function quantLegend() {
  return [
    ["good", "8-bit arithmetic"],
    ["mixed", "boundary / weight-only"],
    ["warn", "review signal"],
    ["risk", "risk signal"],
    ["neutral", "unquantized / not assessed"],
  ];
}

function storageTone(tensor) {
  if (tensor.tone) return tensor.tone;
  const integrity = tensor.numerical_integrity || {};
  const invalid = Number(integrity.nan_value_count || 0)
    + Number(integrity.positive_infinity_value_count || 0)
    + Number(integrity.negative_infinity_value_count || 0)
    + Number(integrity.invalid_encoding_value_count || 0);
  if (invalid) return "risk";
  if (integrity.status === "not_assessed" || integrity.all_zero || integrity.constant_finite) return "warn";
  if (String(tensor.encoding_class || "").includes("quantized") || /^(?:U|I)\d+$/.test(String(tensor.dtype || ""))) return "good";
  if (/^(?:F|BF)\d+/.test(String(tensor.dtype || ""))) return "cool";
  return "neutral";
}

function finalizePresentation(base, rawItems, metric) {
  const assessed = rawItems.filter((item) => item.value != null);
  const positive = assessed.filter((item) => item.value > 0);
  const zeroCount = assessed.length - positive.length;
  const unassessedCount = rawItems.length - assessed.length;
  const total = positive.reduce((sum, item) => sum + item.value, 0);
  const itemTotal = positive.reduce((sum, item) => sum + item.value, 0);
  const assessedGroupCount = new Set(assessed.map((item) => item.groupId)).size;
  const mappedGroupCount = new Set(positive.map((item) => item.groupId)).size;
  return {
    ...base,
    items: positive,
    total,
    assessedCount: assessed.length,
    zeroCount,
    unassessedCount,
    assessedGroupCount,
    mappedGroupCount,
    conservationStatus: Number.isFinite(total) && Math.abs(total - itemTotal) <= Math.max(1e-9, total * 1e-12)
      ? "exact"
      : "failed",
    metricLabel: metric.label,
    unit: metric.unit,
    evidenceClass: metric.evidenceClass,
    status: positive.length ? "assessed" : unassessedCount ? "not_assessed" : "zero_only",
  };
}

export function buildResourceMapPresentation(analysis = {}, options = {}) {
  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const tensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const scope = ops.length ? "operators" : "tensors";
  const metricSet = scope === "operators" ? OP_METRICS : TENSOR_METRICS;
  const metricId = metricSet[options.metric] ? options.metric : Object.keys(metricSet)[0];
  const metric = metricSet[metricId];
  const hasBlocks = Array.isArray(analysis?.block_inventory?.blocks) && analysis.block_inventory.blocks.length > 0;
  const groupOptions = scope === "operators"
    ? [option("stage", "Stage"), ...(hasBlocks ? [option("block", "Block")] : []), option("op_type", "Op type")]
    : [option("dtype", "DType"), option("encoding", "Encoding"), option("namespace", "Namespace")];
  const groupBy = groupOptions.some((row) => row.id === options.groupBy) ? options.groupBy : groupOptions[0].id;
  const colorOptions = scope === "operators"
    ? [option("intensity", "Intensity"), option("quantization", "Quantization"), ...(analysis.format === "tflite" ? [option("delegation", "Delegation")] : [])]
    : [option("storage", "Storage / integrity")];
  const colorBy = colorOptions.some((row) => row.id === options.colorBy) ? options.colorBy : colorOptions[0].id;
  const stageLookup = buildStageLookup(analysis);
  const blockLookup = buildBlockLookup(analysis);
  const source = scope === "operators" ? ops : tensors;
  const items = source.map((entry) => {
    const value = metric.value(entry);
    const group = scope === "operators"
      ? groupForOp(entry, groupBy, stageLookup, blockLookup)
      : groupForTensor(entry, groupBy);
    return {
      id: `${scope === "operators" ? "op" : "tensor"}-${entry.index}`,
      index: Number(entry.index),
      kind: scope === "operators" ? "op" : "tensor",
      label: scope === "operators" ? `#${entry.index} ${stableText(entry.name, "UNKNOWN")}` : `T${entry.index} ${stableText(entry.name, "unnamed")}`,
      shortLabel: scope === "operators" ? `#${entry.index} ${stableText(entry.name, "UNKNOWN")}` : `T${entry.index}`,
      detail: scope === "operators" ? stableText(entry.stage_key, group.label) : `${stableText(entry.dtype, "UNKNOWN")} ${Array.isArray(entry.shape) ? entry.shape.join("x") : "shape unknown"}`,
      value,
      groupId: group.id,
      groupLabel: group.label,
      tone: scope === "operators" ? opTone(entry, colorBy) : storageTone(entry),
      source: entry,
    };
  });
  return finalizePresentation({
    kind: "resource",
    title: scope === "operators" ? "Explorer Resource Map" : "Serialized Tensor Resource Map",
    scope,
    metricId,
    groupBy,
    colorBy,
    metricOptions: Object.entries(metricSet).map(([id, value]) => option(id, value.label)),
    groupOptions,
    colorOptions,
    legend: resourceLegend(scope, colorBy),
    boundary: scope === "operators"
      ? "Area encodes the selected artifact-derived or target-modeled quantity. Color is a separate categorical overlay; it never changes area."
      : "Area encodes serialized tensor bytes. Storage encoding does not establish an execution graph or affine input/output contract.",
  }, items, metric);
}

export function buildQuantizationExposurePresentation(analysis = {}, options = {}, serializedPresentation = null, opSignals = new Map()) {
  if (serializedPresentation) {
    const metric = TENSOR_METRICS.serialized_bytes;
    const groupOptions = [option("encoding", "Encoding"), option("dtype", "DType"), option("namespace", "Namespace")];
    const groupBy = groupOptions.some((row) => row.id === options.groupBy) ? options.groupBy : "encoding";
    const items = serializedPresentation.tiles.map((tensor) => {
      const group = groupForTensor(tensor, groupBy);
      return {
        id: `tensor-${tensor.index}`,
        index: Number(tensor.index),
        kind: "tensor",
        label: `T${tensor.index} ${stableText(tensor.name, "unnamed")}`,
        shortLabel: `T${tensor.index} ${stableText(tensor.dtype, "UNKNOWN")}`,
        detail: stableText(tensor.encoding_class, "storage contract").replaceAll("_", " "),
        value: metric.value(tensor),
        groupId: group.id,
        groupLabel: group.label,
        tone: storageTone(tensor),
        source: tensor,
      };
    });
    return finalizePresentation({
      kind: "quantization",
      title: "Quantization Exposure Map",
      scope: "serialized_tensors",
      metricId: "serialized_bytes",
      groupBy,
      colorBy: "storage_contract",
      metricOptions: [option("serialized_bytes", "Serialized payload")],
      groupOptions,
      colorOptions: [option("storage_contract", "Encoding / integrity")],
      legend: serializedPresentation.legend,
      boundary: serializedPresentation.scope,
    }, items, metric);
  }

  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const metricId = ["macs", "traffic"].includes(options.metric) ? options.metric : "macs";
  const metric = OP_METRICS[metricId];
  const hasBlocks = Array.isArray(analysis?.block_inventory?.blocks) && analysis.block_inventory.blocks.length > 0;
  const groupOptions = [option("stage", "Stage"), ...(hasBlocks ? [option("block", "Block")] : []), option("op_type", "Op type")];
  const groupBy = groupOptions.some((row) => row.id === options.groupBy) ? options.groupBy : "stage";
  const stageLookup = buildStageLookup(analysis);
  const blockLookup = buildBlockLookup(analysis);
  const items = ops.map((op) => {
    const group = groupForOp(op, groupBy, stageLookup, blockLookup);
    const signal = opSignals.get(Number(op.index));
    return {
      id: `op-${op.index}`,
      index: Number(op.index),
      kind: "op",
      label: `#${op.index} ${stableText(op.name, "UNKNOWN")}`,
      shortLabel: `#${op.index} ${stableText(op.name, "UNKNOWN")}`,
      detail: signal?.labels?.join(" / ") || stableText(op.quantization_detail, "No serialized quantization signal"),
      value: metric.value(op),
      groupId: group.id,
      groupLabel: group.label,
      tone: quantTone(op, signal),
      source: op,
    };
  });
  return finalizePresentation({
    kind: "quantization",
    title: "Quantization Exposure Map",
    scope: "graph_ops",
    metricId,
    groupBy,
    colorBy: "quantization",
    metricOptions: [option("macs", "MAC exposure"), option("traffic", "Logical traffic")],
    groupOptions,
    colorOptions: [option("quantization", "Quantization state / risk")],
    legend: quantLegend(),
    boundary: "Area encodes assessed MACs or logical traffic, not risk severity. Zero-MAC boundary ops are counted separately and become visible when a byte metric is selected. Runtime fusion, placement, and task accuracy remain unobserved.",
  }, items, metric);
}

function splitIndex(items, total) {
  let cumulative = 0;
  for (let index = 0; index < items.length - 1; index += 1) {
    cumulative += items[index].value;
    if (cumulative >= total / 2) {
      const before = Math.abs(total / 2 - (cumulative - items[index].value));
      const after = Math.abs(total / 2 - cumulative);
      return Math.max(1, after <= before ? index + 1 : index);
    }
  }
  return Math.max(1, Math.floor(items.length / 2));
}

export function layoutTreemap(items, width = MAP_WIDTH, height = MAP_HEIGHT) {
  const source = items
    .filter((item) => Number.isFinite(item.value) && item.value > 0)
    .map((item, order) => ({ ...item, order }))
    .sort((left, right) => right.value - left.value || left.order - right.order);
  const total = source.reduce((sum, item) => sum + item.value, 0);
  if (!source.length || !(total > 0) || !(width > 0) || !(height > 0)) return [];
  const output = [];
  const visit = (rows, x, y, w, h) => {
    if (rows.length === 1) {
      output.push({ ...rows[0], rect: { x, y, w, h } });
      return;
    }
    const rowTotal = rows.reduce((sum, row) => sum + row.value, 0);
    const pivot = splitIndex(rows, rowTotal);
    const left = rows.slice(0, pivot);
    const right = rows.slice(pivot);
    const leftTotal = left.reduce((sum, row) => sum + row.value, 0);
    const ratio = leftTotal / rowTotal;
    if (w >= h) {
      const leftWidth = w * ratio;
      visit(left, x, y, leftWidth, h);
      visit(right, x + leftWidth, y, w - leftWidth, h);
    } else {
      const topHeight = h * ratio;
      visit(left, x, y, w, topHeight);
      visit(right, x, y + topHeight, w, h - topHeight);
    }
  };
  visit(source, 0, 0, width, height);
  return output;
}

export function layoutGroupedTreemap(items, width = MAP_WIDTH, height = MAP_HEIGHT) {
  const groups = new Map();
  for (const item of items) {
    const current = groups.get(item.groupId) || { id: item.groupId, label: item.groupLabel, value: 0, items: [] };
    current.value += item.value;
    current.items.push(item);
    groups.set(item.groupId, current);
  }
  const groupLayout = layoutTreemap([...groups.values()], width, height);
  const tiles = [];
  const groupRects = [];
  for (const group of groupLayout) {
    groupRects.push({ id: group.id, label: group.label, value: group.value, rect: group.rect });
    const children = layoutTreemap(group.items, group.rect.w, group.rect.h);
    for (const child of children) {
      tiles.push({
        ...child,
        rect: {
          x: group.rect.x + child.rect.x,
          y: group.rect.y + child.rect.y,
          w: child.rect.w,
          h: child.rect.h,
        },
      });
    }
  }
  return { groups: groupRects, tiles, width, height };
}

function selectControl(label, value, options, name, onChange) {
  const wrap = document.createElement("label");
  wrap.className = "evidence-map-control";
  const caption = document.createElement("span");
  caption.textContent = label;
  const select = document.createElement("select");
  select.dataset.treemapControl = name;
  for (const row of options) {
    const entry = document.createElement("option");
    entry.value = row.id;
    entry.textContent = row.label;
    entry.selected = row.id === value;
    select.append(entry);
  }
  select.disabled = options.length < 2;
  select.addEventListener("change", () => onChange(name, select.value));
  wrap.append(caption, select);
  return wrap;
}

function tileAccessibleLabel(item, presentation) {
  const share = presentation.total > 0 ? item.value / presentation.total : 0;
  return `${item.label}; ${presentation.metricLabel} ${formatMetric(item.value, presentation.unit)}; ${(share * 100).toFixed(2)} percent; group ${item.groupLabel}; ${item.detail}`;
}

export function renderEvidenceTreemap(container, presentation, { onSelect = null, onControl = null } = {}) {
  if (!container) return;
  container.replaceChildren();
  const head = document.createElement("div");
  head.className = "evidence-map-head";
  const copy = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = presentation.title;
  const subtitle = document.createElement("p");
  subtitle.textContent = `${presentation.metricLabel} by ${presentation.groupOptions.find((row) => row.id === presentation.groupBy)?.label || presentation.groupBy}; ${presentation.evidenceClass}.`;
  copy.append(title, subtitle);
  const controls = document.createElement("div");
  controls.className = "evidence-map-controls";
  const change = (name, value) => onControl?.(name, value);
  controls.append(
    selectControl("Group", presentation.groupBy, presentation.groupOptions, "groupBy", change),
    selectControl("Area", presentation.metricId, presentation.metricOptions, "metric", change),
    selectControl("Color", presentation.colorBy, presentation.colorOptions, "colorBy", change),
  );
  head.append(copy, controls);

  const summary = document.createElement("div");
  summary.className = "evidence-map-summary";
  const summaryRows = [
    ["Mapped total", presentation.total > 0 ? formatMetric(presentation.total, presentation.unit) : "No positive values"],
    ["Positive / assessed", `${presentation.items.length} / ${presentation.assessedCount}`],
    ["Groups mapped", `${presentation.mappedGroupCount} / ${presentation.assessedGroupCount}`],
    ["Exact zero", String(presentation.zeroCount)],
    ["Not assessed", String(presentation.unassessedCount)],
    ["Conservation", presentation.conservationStatus],
  ];
  for (const [label, value] of summaryRows) {
    const row = document.createElement("span");
    const caption = document.createElement("small");
    caption.textContent = label;
    const metricValue = document.createElement("strong");
    metricValue.textContent = value;
    row.append(caption, metricValue);
    summary.append(row);
  }

  const legend = document.createElement("div");
  legend.className = "evidence-map-legend";
  for (const [tone, label] of presentation.legend) {
    const item = document.createElement("span");
    item.className = `evidence-map-key tone-${tone}`;
    item.textContent = label;
    legend.append(item);
  }

  if (presentation.status !== "assessed") {
    const empty = document.createElement("p");
    empty.className = "visual-empty";
    empty.textContent = presentation.status === "zero_only"
      ? `The selected metric was assessed, but all ${presentation.assessedCount} values are exactly zero.`
      : "The selected quantity is not available for this artifact. It is not reported as zero.";
    const boundary = document.createElement("p");
    boundary.className = "evidence-map-boundary";
    boundary.textContent = presentation.boundary;
    container.append(head, summary, legend, empty, boundary);
    return;
  }

  const layout = layoutGroupedTreemap(presentation.items);
  const selection = document.createElement("div");
  selection.className = "evidence-map-selection";
  selection.hidden = true;
  const selectItem = (item, trigger) => {
    container.querySelectorAll(".evidence-map-selected").forEach((node) => node.classList.remove("evidence-map-selected"));
    trigger?.classList.add("evidence-map-selected");
    selection.hidden = false;
    selection.replaceChildren();
    const label = document.createElement("strong");
    label.textContent = item.label;
    const detail = document.createElement("span");
    const share = presentation.total ? item.value / presentation.total : 0;
    detail.textContent = `${item.groupLabel} | ${formatMetric(item.value, presentation.unit)} | ${(share * 100).toFixed(2)}% | ${item.detail}`;
    selection.append(label, detail);
    onSelect?.(item, trigger);
  };
  const canvas = document.createElement("div");
  canvas.className = "evidence-treemap-canvas";
  canvas.setAttribute("role", "group");
  canvas.setAttribute("aria-label", `${presentation.title}; area represents ${presentation.metricLabel}`);
  for (const group of layout.groups) {
    const outline = document.createElement("div");
    outline.className = "evidence-treemap-group";
    outline.style.left = `${group.rect.x / layout.width * 100}%`;
    outline.style.top = `${group.rect.y / layout.height * 100}%`;
    outline.style.width = `${group.rect.w / layout.width * 100}%`;
    outline.style.height = `${group.rect.h / layout.height * 100}%`;
    const label = document.createElement("span");
    label.textContent = group.label;
    outline.append(label);
    canvas.append(outline);
  }
  for (const item of layout.tiles) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = `evidence-treemap-tile tone-${item.tone}`;
    tile.style.left = `${item.rect.x / layout.width * 100}%`;
    tile.style.top = `${item.rect.y / layout.height * 100}%`;
    tile.style.width = `${item.rect.w / layout.width * 100}%`;
    tile.style.height = `${item.rect.h / layout.height * 100}%`;
    tile.setAttribute("aria-label", tileAccessibleLabel(item, presentation));
    tile.title = tileAccessibleLabel(item, presentation);
    const label = document.createElement("strong");
    label.textContent = item.shortLabel;
    const value = document.createElement("span");
    value.textContent = formatMetric(item.value, presentation.unit);
    tile.append(label, value);
    tile.addEventListener("click", () => selectItem(item, tile));
    canvas.append(tile);
  }

  const mobile = document.createElement("div");
  mobile.className = "evidence-treemap-mobile-list";
  for (const item of [...presentation.items].sort((left, right) => right.value - left.value || left.index - right.index)) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `evidence-map-mobile-row tone-${item.tone}`;
    row.setAttribute("aria-label", tileAccessibleLabel(item, presentation));
    const share = presentation.total ? item.value / presentation.total : 0;
    const label = document.createElement("span");
    const labelMain = document.createElement("strong");
    labelMain.textContent = item.label;
    const labelGroup = document.createElement("small");
    labelGroup.textContent = item.groupLabel;
    label.append(labelMain, labelGroup);
    const value = document.createElement("span");
    const valueMain = document.createElement("strong");
    valueMain.textContent = formatMetric(item.value, presentation.unit);
    const valueShare = document.createElement("small");
    valueShare.textContent = `${(share * 100).toFixed(2)}%`;
    value.append(valueMain, valueShare);
    row.append(label, value);
    row.addEventListener("click", () => selectItem(item, row));
    mobile.append(row);
  }

  const boundary = document.createElement("p");
  boundary.className = "evidence-map-boundary";
  boundary.textContent = presentation.boundary;
  container.append(head, summary, legend, canvas, mobile, selection, boundary);
}
