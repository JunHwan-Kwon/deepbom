import { clampNumber, formatNumber, formatPercent, humanizeStageKey, humanStatusLabel, padOp, sumNumbers } from "./format.js";

export function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

export function detailGrid(items) {
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  for (const [label, value] of items) {
    const item = document.createElement("div");
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value || "-";
    item.append(span, strong);
    grid.append(item);
  }
  return grid;
}

export function insightCard(label, value, detail, tone = "neutral", jumpTab = null, jumpFilter = null, evidenceClass = "") {
  const card = document.createElement(jumpTab ? "button" : "div");
  card.className = `insight-card ${tone}${jumpTab ? " jumpable" : ""}`;
  card.dataset.cardLabel = label;
  card.dataset.cardTone = tone;
  card.title = [label, value, evidenceClass, detail].filter(Boolean).join("\n");
  if (String(value || "").length > 28) card.classList.add("compact-value");
  if (jumpTab) {
    card.type = "button";
    card.dataset.jumpTab = jumpTab;
    if (jumpFilter) card.dataset.jumpFilter = jumpFilter;
  }
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value || "-";
  const p = document.createElement("p");
  p.textContent = detail || "";
  card.append(span, strong, p);
  if (evidenceClass) {
    const evidence = document.createElement("small");
    evidence.className = "insight-evidence";
    evidence.textContent = evidenceClass;
    card.append(evidence);
  }
  if (jumpTab) {
    const badge = document.createElement("em");
    badge.className = "card-jump-badge";
    badge.textContent = "→";
    card.append(badge);
  }
  return card;
}

export function evidenceDisclosure(summaryText, content, {
  contentLabel = "Exact evidence",
} = {}) {
  const details = document.createElement("details");
  details.className = "evidence-disclosure";
  const summary = document.createElement("summary");
  summary.textContent = summaryText;
  const code = document.createElement("code");
  code.setAttribute("aria-label", contentLabel);
  code.textContent = content || "Not emitted";
  details.append(summary, code);
  return details;
}

export function rooflineBar(label, count, total, className) {
  const node = document.createElement("div");
  node.className = "roofline-bar";
  const head = document.createElement("div");
  const left = document.createElement("span");
  left.textContent = label;
  const right = document.createElement("strong");
  right.textContent = `${formatNumber(count)} (${formatPercent(count / Math.max(1, total))})`;
  head.append(left, right);
  const track = document.createElement("div");
  track.className = "bar-track";
  const fill = document.createElement("i");
  fill.className = className;
  fill.style.width = `${clampNumber((count / Math.max(1, total)) * 100, 0, 100)}%`;
  track.append(fill);
  node.append(head, track);
  return node;
}

export function signalItem(label, value, tone = "neutral") {
  const item = document.createElement("li");
  item.className = `signal-item ${tone}`;
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value || "-";
  item.append(span, strong);
  return item;
}

export function runtimeSignal(label, value, tone) {
  const node = document.createElement("div");
  node.className = `runtime-signal ${tone}`;
  const dot = document.createElement("i");
  const text = document.createElement("span");
  text.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  node.append(dot, text, strong);
  return node;
}

export function appendDetailList(container, titleText, items) {
  const title = document.createElement("h4");
  title.textContent = titleText;
  container.append(title);
  if (!items.length) {
    const empty = document.createElement("p");
    empty.textContent = "-";
    container.append(empty);
    return;
  }
  const list = document.createElement("ul");
  for (const value of items) {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  }
  container.append(list);
}

export function td(value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) {
    cell.className = className;
  }
  return cell;
}

export function visualListItem(label, value, tone = "neutral") {
  const item = document.createElement("li");
  item.className = `visual-list-item ${tone}`;
  const strong = document.createElement("strong");
  strong.textContent = label;
  const span = document.createElement("span");
  span.textContent = value;
  item.append(strong, span);
  return item;
}

export function renderVisualEmpty(container, text) {
  const node = targetCompareMessage(text);
  container.replaceChildren(node);
}

export function targetCompareMessage(text) {
  const node = document.createElement("div");
  node.className = "visual-empty";
  node.textContent = text;
  return node;
}

export function protocolBlock(titleText, items) {
  const block = document.createElement("div");
  block.className = "deepbom-protocol-block";
  const title = document.createElement("h3");
  title.textContent = titleText;
  const list = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    if (item.status) li.classList.add(`protocol-${item.status}`);
    const strong = document.createElement("strong");
    strong.textContent = item.name || "-";
    if (item.status) {
      const badge = document.createElement("em");
      badge.textContent = humanStatusLabel(item.status);
      strong.append(badge);
    }
    const span = document.createElement("span");
    span.textContent = item.detail || "-";
    li.append(strong, span);
    list.append(li);
  }
  block.append(title, list);
  return block;
}

export function deepBomMetric(label, value, hint, status = {}) {
  const node = document.createElement("div");
  node.className = `deepbom-metric ${status.tone || "neutral"}`;
  const small = document.createElement("span");
  small.textContent = label;
  if (status.label) {
    const badge = document.createElement("em");
    badge.textContent = status.label;
    badge.className = `metric-status ${status.tone || "neutral"}`;
    small.append(badge);
  }
  const strong = document.createElement("strong");
  strong.textContent = value;
  node.append(small, strong);
  // Visual score bar when a 0–1 magnitude is provided
  if (status.score01 != null) {
    const track = document.createElement("div");
    track.className = "metric-bar-track";
    const fill = document.createElement("div");
    fill.className = `metric-bar-fill ${status.tone || "neutral"}`;
    fill.style.width = `${Math.max(2, Math.round(Number(status.score01) * 100))}%`;
    track.append(fill);
    node.append(track);
  }
  const p = document.createElement("p");
  p.textContent = hint;
  node.append(p);
  // Criteria behind a collapsible so cards stay compact by default
  if (status.criteria) {
    const det = document.createElement("details");
    det.className = "metric-detail";
    const sum = document.createElement("summary");
    sum.textContent = "Criteria";
    const criP = document.createElement("p");
    criP.textContent = status.criteria;
    det.append(sum, criP);
    node.append(det);
  }
  return node;
}

export function targetStack(totals) {
  const node = document.createElement("div");
  node.className = "target-stack";
  const total = Math.max(1e-9, sumNumbers(Object.values(totals || {})));
  for (const [key, label] of [
    ["computeUs", "compute"],
    ["memoryUs", "memory"],
    ["packingUs", "packing"],
    ["breakUs", "break"],
    ["fallbackUs", "fallback"],
  ]) {
    const share = Number(totals?.[key] || 0) / total;
    if (share <= 0) continue;
    const segment = document.createElement("i");
    segment.className = label;
    segment.style.flexBasis = `${Math.max(2, share * 100)}%`;
    segment.title = `${label}: ${formatPercent(share)}`;
    node.append(segment);
  }
  return node;
}

export function targetMetric(label, value, detail = "") {
  const item = document.createElement("div");
  if (detail) {
    item.title = detail;
    item.setAttribute("aria-label", `${label}: ${value}. ${detail}`);
  }
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  item.append(span, strong);
  return item;
}

export function createExportCanvasShell(title, subtitle, width = 1180, height = 680, filename = "") {
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  roundRect(ctx, 24, 24, width - 48, height - 48, 10);
  ctx.fill();
  ctx.strokeStyle = "#d9e0ea";
  ctx.lineWidth = 1;
  roundRect(ctx, 24, 24, width - 48, height - 48, 10);
  ctx.stroke();
  ctx.fillStyle = "#2454d6";
  ctx.font = "700 11px Inter, Arial, sans-serif";
  ctx.fillText("DEEPBOM VISUAL", 48, 56);
  ctx.fillStyle = "#151a22";
  ctx.font = "800 24px Inter, Arial, sans-serif";
  ctx.fillText(title, 48, 88);
  ctx.fillStyle = "#657286";
  ctx.font = "13px Inter, Arial, sans-serif";
  ctx.fillText(subtitle, 48, 112);
  ctx.fillStyle = "#657286";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(filename, width - 48, 56);
  ctx.textAlign = "left";
  return { canvas, ctx, width, height, y: 148 };
}

export async function canvasToPngBytes(canvas) {
  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob((result) => result ? resolve(result) : reject(new Error("PNG rendering failed")), "image/png");
  });
  return new Uint8Array(await blob.arrayBuffer());
}

export async function withBusyButton(button, label, task, onSettled = () => {}) {
  const previousLabel = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try {
    return await task();
  } finally {
    button.textContent = previousLabel;
    onSettled();
  }
}

export function roundRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

export function drawFlame(ctx, segments, total, x, y, width, height) {
  let cursor = x;
  for (const segment of segments) {
    const segmentWidth = Math.max(18, (Number(segment.value || 0) / Math.max(1, total)) * width);
    ctx.fillStyle = colorForTone(segment.tone);
    roundRect(ctx, cursor, y, Math.min(segmentWidth, x + width - cursor), height, 7);
    ctx.fill();
    if (segmentWidth > 60) {
      ctx.fillStyle = "#151a22";
      ctx.font = "800 12px Inter, Arial, sans-serif";
      ctx.fillText(segment.label, cursor + 10, y + height - 32);
      ctx.fillStyle = "#657286";
      ctx.font = "11px Inter, Arial, sans-serif";
      ctx.fillText(formatPercent(Number(segment.value || 0) / Math.max(1, total)), cursor + 10, y + height - 14);
    }
    cursor += segmentWidth + 3;
    if (cursor >= x + width) break;
  }
}

export function drawRankList(ctx, rows, x, y, width) {
  rows.forEach(([label, value, tone], index) => {
    const rowY = y + index * 42;
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, x, rowY, width, 32, 6);
    ctx.fill();
    ctx.strokeStyle = "#d9e0ea";
    ctx.stroke();
    ctx.fillStyle = colorForTone(tone, true);
    ctx.fillRect(x, rowY, 4, 32);
    ctx.fillStyle = "#151a22";
    ctx.font = "700 12px Inter, Arial, sans-serif";
    ctx.fillText(trimCanvasText(ctx, label, width * 0.66), x + 14, rowY + 20);
    ctx.fillStyle = "#657286";
    ctx.textAlign = "right";
    ctx.font = "12px Inter, Arial, sans-serif";
    ctx.fillText(value, x + width - 14, rowY + 20);
    ctx.textAlign = "left";
  });
}

export function drawLegend(ctx, x, y) {
  [
    ["compute-bound", "Compute"],
    ["mixed", "Mixed"],
    ["memory-bound", "Low-intensity"],
    ["good", "Quantized"],
    ["neutral", "Neutral"],
  ].forEach(([tone, label], index) => {
    const lx = x + index * 142;
    ctx.fillStyle = colorForTone(tone);
    roundRect(ctx, lx, y, 18, 18, 4);
    ctx.fill();
    ctx.fillStyle = "#657286";
    ctx.font = "12px Inter, Arial, sans-serif";
    ctx.fillText(label, lx + 26, y + 14);
  });
}

export function drawEmptyState(ctx, text, x, y) {
  ctx.fillStyle = "#657286";
  ctx.font = "14px Inter, Arial, sans-serif";
  ctx.fillText(text, x, y + 24);
}

export function drawStack(ctx, totals, x, y, width, height) {
  const entries = [
    ["computeUs", "compute-bound"],
    ["memoryUs", "memory-bound"],
    ["packingUs", "packing"],
    ["breakUs", "break"],
    ["fallbackUs", "fallback"],
  ];
  const total = Math.max(1e-9, sumNumbers(entries.map(([key]) => totals?.[key] || 0)));
  let cursor = x;
  for (const [key, tone] of entries) {
    const share = Number(totals?.[key] || 0) / total;
    if (share <= 0) continue;
    const w = Math.max(2, share * width);
    ctx.fillStyle = colorForTone(tone, true);
    ctx.fillRect(cursor, y, w, height);
    cursor += w;
  }
}

export function drawChainBlock(ctx, chain, x, y, width, height) {
  ctx.fillStyle = "#ffffff";
  roundRect(ctx, x, y, width, height, 8);
  ctx.fill();
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.fillStyle = "#151a22";
  ctx.font = "800 17px Inter, Arial, sans-serif";
  ctx.fillText(`C${chain.id}`, x + 12, y + 28);
  ctx.fillStyle = "#657286";
  ctx.font = "12px Inter, Arial, sans-serif";
  ctx.fillText(`ops ${chain.first_op}-${chain.last_op}`, x + 12, y + 50);
  ctx.fillStyle = "#0f766e";
  ctx.font = "800 12px Inter, Arial, sans-serif";
  ctx.fillText(`${formatPercent(chain.mac_percent || 0)} MACs`, x + 12, y + 68);
}

export function drawBreakPill(ctx, text, x, y) {
  ctx.fillStyle = "#fff7ed";
  roundRect(ctx, x, y, 72, 28, 14);
  ctx.fill();
  ctx.strokeStyle = "#a16207";
  ctx.stroke();
  ctx.fillStyle = "#7a4a04";
  ctx.font = "800 11px Inter, Arial, sans-serif";
  ctx.fillText(text, x + 10, y + 18);
}

export function drawTargetCard(ctx, row, x, y, width, height) {
  ctx.fillStyle = "#ffffff";
  roundRect(ctx, x, y, width, height, 8);
  ctx.fill();
  ctx.strokeStyle = row.error ? "#b42318" : "#d9e0ea";
  ctx.stroke();
  ctx.fillStyle = "#151a22";
  ctx.font = "800 14px Inter, Arial, sans-serif";
  ctx.fillText(trimCanvasText(ctx, row.label || "Target", width - 24), x + 12, y + 26);
  ctx.fillStyle = "#657286";
  ctx.font = "11px Inter, Arial, sans-serif";
  if (row.error) {
    ctx.fillText(trimCanvasText(ctx, row.error, width - 24), x + 12, y + 54);
    return;
  }
  ctx.fillText(`ridge ~${formatNumber(row.ridge)} ops/B`, x + 12, y + 48);
  drawStack(ctx, row.totals, x + 12, y + 64, width - 24, 12);
  const metrics = [
    ["Memory", formatPercent(row.memoryRatio)],
    ["Breaks", formatNumber(row.chainBreaks)],
    ["INT8", `~${row.speedup.toFixed(2)}x`],
    ["Top", row.topOp ? `#${padOp(row.topOp.index)}` : "-"],
  ];
  metrics.forEach(([label, value], index) => {
    const mx = x + 12 + (index % 2) * ((width - 24) / 2);
    const my = y + 100 + Math.floor(index / 2) * 34;
    ctx.fillStyle = "#657286";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.fillText(label, mx, my);
    ctx.fillStyle = "#151a22";
    ctx.font = "800 13px Inter, Arial, sans-serif";
    ctx.fillText(value, mx, my + 18);
  });
}

export function drawStageMixRow(ctx, stage, ops, x, y, width) {
  const labelWidth = 250;
  ctx.fillStyle = "#151a22";
  ctx.font = "800 12px Inter, Arial, sans-serif";
  ctx.fillText(trimCanvasText(ctx, `#${stage.index} ${humanizeStageKey(stage.key)}`, labelWidth - 10), x, y + 20);
  ctx.fillStyle = "#657286";
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillText(`${formatPercent(stage.mac_percent || 0)} MACs / breaks ${stage.xnnpack_chain_breaks || 0}`, x, y + 36);
  const totalMacs = sumNumbers(ops.map((op) => op.macs));
  const denominator = totalMacs || Math.max(1, ops.length);
  const counts = { "compute-bound": 0, mixed: 0, "memory-bound": 0 };
  for (const op of ops) {
    const key = op.static_bound_guess === "compute-bound" || op.static_bound_guess === "mixed" ? op.static_bound_guess : "memory-bound";
    counts[key] += totalMacs ? Number(op.macs || 0) : 1;
  }
  let cursor = x + labelWidth;
  const barWidth = width - labelWidth - 120;
  for (const key of ["compute-bound", "mixed", "memory-bound"]) {
    const share = counts[key] / denominator;
    const w = Math.max(share > 0 ? 3 : 0, share * barWidth);
    ctx.fillStyle = colorForTone(key, true);
    ctx.fillRect(cursor, y + 12, w, 16);
    cursor += w;
  }
  ctx.fillStyle = "#657286";
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillText(`${formatPercent(counts["memory-bound"] / denominator)} low-int`, x + width - 108, y + 25);
}

export function colorForTone(tone, solid = false) {
  const palette = {
    "compute-bound": solid ? "#0f766e" : "#cdebe6",
    mixed: solid ? "#a16207" : "#f7ddb0",
    "memory-bound": solid ? "#b42318" : "#f5c4be",
    risk: solid ? "#b42318" : "#f5c4be",
    warn: solid ? "#a16207" : "#f7ddb0",
    good: solid ? "#0f766e" : "#d7f0ec",
    packing: solid ? "#6046b6" : "#ddd4fa",
    break: solid ? "#2454d6" : "#d8e2ff",
    fallback: solid ? "#a16207" : "#f7ddb0",
    neutral: solid ? "#657286" : "#e5eaf0",
  };
  return palette[tone] || palette.neutral;
}

export function trimCanvasText(ctx, text, maxWidth) {
  const value = String(text || "");
  if (ctx.measureText(value).width <= maxWidth) return value;
  let trimmed = value;
  while (trimmed.length > 4 && ctx.measureText(`${trimmed}...`).width > maxWidth) {
    trimmed = trimmed.slice(0, -1);
  }
  return `${trimmed}...`;
}
