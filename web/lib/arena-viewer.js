import { decorateEvidenceElement } from "./evidence-visual-contract.js";

const DTYPE_COLORS = Object.freeze({
  FLOAT32: "#38bdf8",
  FLOAT16: "#22c55e",
  BFLOAT16: "#14b8a6",
  INT8: "#f59e0b",
  UINT8: "#eab308",
  INT16: "#f97316",
  UINT16: "#fb7185",
  INT32: "#a78bfa",
  UINT32: "#c084fc",
  BOOL: "#94a3b8",
});

const resizeObservers = new WeakMap();

function disconnectResizeObserver(container) {
  resizeObservers.get(container)?.disconnect();
  resizeObservers.delete(container);
}

function element(tag, className, text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function opEnd(allocation, opCount) {
  const last = allocation.last_node == null ? opCount - 1 : Number(allocation.last_node);
  return Math.max(Number(allocation.first_node || 0), Math.min(opCount - 1, last));
}

function peakRootOccupancy(allocations, opCount, arenaBytes) {
  let peakBytes = 0;
  let peakOp = 0;
  for (let op = 0; op < opCount; op += 1) {
    const liveBytes = allocations.reduce((sum, allocation) => {
      const first = Math.max(0, Number(allocation.first_node || 0));
      return op >= first && op <= opEnd(allocation, opCount) ? sum + Number(allocation.size_bytes || 0) : sum;
    }, 0);
    if (liveBytes > peakBytes) {
      peakBytes = liveBytes;
      peakOp = op;
    }
  }
  return { peak_bytes: peakBytes, peak_op: peakOp, ratio: arenaBytes > 0 ? peakBytes / arenaBytes : null };
}

function metric(label, value, title = "") {
  const item = element("span", "arena-metric");
  const key = element("span", "arena-metric-label", label);
  const data = element("strong", "arena-metric-value", value);
  if (title) item.title = title;
  item.append(key, data);
  return item;
}

function signedBytes(formatBytes, value) {
  const number = finiteNumber(value);
  if (number == null) return "Not assessed";
  return `${number > 0 ? "+" : number < 0 ? "-" : ""}${formatBytes(Math.abs(number))}`;
}

function renderRuntimeDifferences(container, reconciliation, formatBytes) {
  const differences = (reconciliation?.allocation_rows || []).filter((row) =>
    !row.projected_present || !row.observed_present
      || row.size_match === false || row.arena_match === false || row.offset_match === false);
  if (!differences.length) {
    container.append(element("div", "arena-runtime-match", "All joined final-snapshot roots match projected arena, size, and offset."));
    return;
  }
  const wrap = element("div", "arena-runtime-table-wrap");
  const table = element("table", "arena-runtime-table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Tensor", "Origin", "Projected", "Observed", "Size delta"]) {
    headRow.append(element("th", "", label));
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const row of differences.slice(0, 40)) {
    const tr = document.createElement("tr");
    const projected = row.projected_present
      ? `${row.projected_arena} | ${formatBytes(row.projected_size_bytes)} @ ${formatBytes(row.projected_offset_bytes)}`
      : "Not projected";
    const observed = row.observed_present
      ? `${row.observed_arena} | ${formatBytes(row.observed_size_bytes)} @ ${formatBytes(row.observed_offset_bytes)}`
      : "Not observed";
    for (const value of [
      `T${row.tensor_index} ${row.tensor_name || ""}`,
      row.artifact_tensor ? "Artifact" : "Runtime temporary",
      projected,
      observed,
      row.size_delta_bytes == null ? "N/A" : signedBytes(formatBytes, row.size_delta_bytes),
    ]) tr.append(element("td", "", value));
    body.append(tr);
  }
  table.append(head, body);
  wrap.append(table);
  if (differences.length > 40) wrap.append(element("div", "arena-runtime-table-note", `40 / ${differences.length} difference rows shown; the complete ledger remains in raw evidence.`));
  container.append(wrap);
}

function renderIssues(container, plan) {
  const issues = Array.isArray(plan?.unassessed_tensors) ? plan.unassessed_tensors : [];
  if (!issues.length) return;
  const row = element("div", "arena-issue-row");
  row.append(element("strong", "arena-issue-label", `${issues.length} calculation issue${issues.length === 1 ? "" : "s"}`));
  for (const issue of issues.slice(0, 4)) {
    const item = element("span", "arena-issue-item", `${issue.tensor_index == null ? "plan" : `T${issue.tensor_index}`}: ${issue.reason || "not assessed"}`);
    item.title = issue.tensor_name || "";
    row.append(item);
  }
  if (issues.length > 4) row.append(element("span", "arena-issue-item", `+${issues.length - 4} more`));
  container.append(row);
}

function renderAliases(container, plan, onSelectOp) {
  const aliases = Array.isArray(plan?.aliases) ? plan.aliases : [];
  if (!aliases.length) return;
  const row = element("div", "arena-alias-row");
  row.append(element("span", "arena-alias-label", "In-place aliases"));
  for (const alias of aliases.slice(0, 10)) {
    const button = element("button", "arena-alias-chip", `T${alias.tensor_index} -> T${alias.shared_with_tensor_index}  #${alias.op_index}`);
    button.type = "button";
    button.title = `${alias.op_name || "operator"}; ${alias.data_unmodified ? "data unmodified" : "buffer contents changed"}; ${alias.source || "source not emitted"}`;
    button.addEventListener("click", () => onSelectOp?.(Number(alias.op_index)));
    row.append(button);
  }
  if (aliases.length > 10) row.append(element("span", "arena-alias-more", `+${aliases.length - 10}`));
  container.append(row);
}

function renderArenaCanvas(container, analysis, plan, formatBytes, onSelectOp) {
  const opCount = Math.max(1, (analysis?.ops || []).length);
  const arenaBytes = finiteNumber(plan?.non_persistent_arena_bytes);
  const allocations = (plan?.allocations || []).filter((allocation) =>
    allocation?.arena === "kTfLiteArenaRw"
      && allocation?.allocation_status === "allocated"
      && finiteNumber(allocation?.offset_bytes) != null
      && finiteNumber(allocation?.size_bytes) != null,
  );
  if (arenaBytes == null || arenaBytes <= 0 || !allocations.length) return;

  const canvas = element("canvas", "arena-map-canvas");
  const cssWidth = Math.max(320, Math.round(container.clientWidth || 960));
  const cssHeight = 190;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.height = `${cssHeight}px`;
  canvas.setAttribute("aria-label", "TFLite ArenaPlanner declared-shape allocation map");
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  const theme = getComputedStyle(document.documentElement);
  const canvasColor = theme.getPropertyValue("--viz-canvas").trim() || "#0b1220";
  const gridColor = theme.getPropertyValue("--viz-grid").trim() || "rgba(148, 163, 184, 0.18)";
  const metaColor = theme.getPropertyValue("--viz-meta").trim() || "#94a3b8";
  const peakColor = theme.getPropertyValue("--risk").trim() || "#dc2626";

  // Reserve enough room for MiB labels; a narrow margin clipped the leading
  // digit and made otherwise monotonic ticks appear to decrease.
  const left = 86;
  const right = 8;
  const top = 12;
  const bottom = 22;
  const plotWidth = cssWidth - left - right;
  const plotHeight = cssHeight - top - bottom;
  const xFor = (op) => left + (Math.max(0, Math.min(opCount - 1, op)) / Math.max(1, opCount - 1)) * plotWidth;
  const yFor = (offset) => top + (Math.max(0, Math.min(arenaBytes, offset)) / arenaBytes) * plotHeight;

  context.fillStyle = canvasColor;
  context.fillRect(0, 0, cssWidth, cssHeight);
  context.font = "10px ui-monospace, SFMono-Regular, Consolas, monospace";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const bytes = (arenaBytes * tick) / 4;
    const y = yFor(bytes);
    context.strokeStyle = gridColor;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(cssWidth - right, y);
    context.stroke();
    context.fillStyle = metaColor;
    context.textAlign = "right";
    context.fillText(formatBytes(bytes), left - 6, y + 3);
  }

  const hitRegions = [];
  for (const allocation of allocations) {
    const first = Math.max(0, Number(allocation.first_node || 0));
    const last = opEnd(allocation, opCount);
    const offset = Number(allocation.offset_bytes);
    const size = Number(allocation.size_bytes);
    const x = xFor(first);
    const width = Math.max(2, xFor(last) - x + Math.max(2, plotWidth / Math.max(1, opCount)));
    const y = yFor(offset);
    const height = Math.max(2, (size / arenaBytes) * plotHeight);
    const color = DTYPE_COLORS[String(allocation.tensor_dtype || "").toUpperCase()] || "#64748b";
    context.globalAlpha = 0.64;
    context.fillStyle = color;
    context.fillRect(x, y, Math.min(width, cssWidth - right - x), Math.min(height, top + plotHeight - y));
    context.globalAlpha = 1;
    context.strokeStyle = color;
    context.strokeRect(x + 0.5, y + 0.5, Math.max(1, Math.min(width, cssWidth - right - x) - 1), Math.max(1, Math.min(height, top + plotHeight - y) - 1));
    hitRegions.push({ allocation, x, y, width: Math.min(width, cssWidth - right - x), height: Math.min(height, top + plotHeight - y) });
  }

  const occupancy = peakRootOccupancy(allocations, opCount, arenaBytes);
  canvas.dataset.peakMarker = "rendered";
  canvas.dataset.peakOp = String(occupancy.peak_op);
  canvas.dataset.peakBytes = String(occupancy.peak_bytes);
  const peakX = xFor(occupancy.peak_op);
  context.save();
  context.strokeStyle = peakColor;
  context.setLineDash([4, 3]);
  context.beginPath();
  context.moveTo(peakX, top);
  context.lineTo(peakX, top + plotHeight);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = peakColor;
  context.textAlign = peakX > cssWidth - 90 ? "right" : "left";
  context.fillText(`peak #${occupancy.peak_op}`, peakX + (peakX > cssWidth - 90 ? -4 : 4), top + 10);
  context.restore();

  context.fillStyle = metaColor;
  context.textAlign = "left";
  context.fillText("op 0", left, cssHeight - 6);
  context.textAlign = "right";
  context.fillText(`op ${opCount - 1}`, cssWidth - right, cssHeight - 6);

  const tooltip = element("div", "layered-tooltip arena-map-tooltip");
  tooltip.hidden = true;
  const regionAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * cssWidth;
    const y = ((event.clientY - rect.top) / rect.height) * cssHeight;
    return hitRegions
      .filter((region) => x >= region.x && x <= region.x + region.width && y >= region.y && y <= region.y + region.height)
      .sort((leftRegion, rightRegion) => leftRegion.width * leftRegion.height - rightRegion.width * rightRegion.height)[0] || null;
  };
  canvas.addEventListener("mousemove", (event) => {
    const region = regionAt(event);
    if (!region) {
      tooltip.hidden = true;
      return;
    }
    const allocation = region.allocation;
    tooltip.textContent = `T${allocation.tensor_index} ${allocation.tensor_name || ""} | ${allocation.tensor_dtype} ${(allocation.tensor_shape || []).join("x") || "scalar"} | ${formatBytes(allocation.size_bytes)} @ ${formatBytes(allocation.offset_bytes)} | #${allocation.first_node}-#${opEnd(allocation, opCount)}`;
    const canvasRect = canvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    tooltip.style.left = `${Math.max(4, Math.min(event.clientX - containerRect.left + 10, containerRect.width - 360))}px`;
    tooltip.style.top = `${canvasRect.top - containerRect.top + 6}px`;
    tooltip.hidden = false;
  });
  canvas.addEventListener("mouseleave", () => {
    tooltip.hidden = true;
  });
  canvas.addEventListener("click", (event) => {
    const region = regionAt(event);
    if (region) onSelectOp?.(Number(region.allocation.first_node || 0));
  });
  const toolbar = element("div", "arena-map-toolbar");
  const zoomOut = element("button", "icon-action", "-");
  const fit = element("button", "secondary-action", "Fit");
  const zoomIn = element("button", "icon-action", "+");
  for (const [button, label] of [[zoomOut, "Zoom out arena map"], [fit, "Fit arena map to width"], [zoomIn, "Zoom in arena map"]]) {
    button.type = "button";
    button.setAttribute("aria-label", label);
    button.title = label;
  }
  const scroll = element("div", "arena-map-scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", "Scrollable ArenaPlanner allocation map");
  let zoom = 1;
  const fittedWidth = () => Math.max(1, Math.min(cssWidth, scroll.clientWidth || cssWidth));
  const applyZoom = () => { canvas.style.width = `${Math.round(fittedWidth() * zoom)}px`; };
  zoomOut.addEventListener("click", () => { zoom = Math.max(1, zoom - 0.25); applyZoom(); });
  fit.addEventListener("click", () => { zoom = 1; applyZoom(); scroll.scrollLeft = 0; });
  zoomIn.addEventListener("click", () => { zoom = Math.min(3, zoom + 0.25); applyZoom(); });
  toolbar.append(fit, zoomOut, zoomIn);
  scroll.append(canvas);
  container.append(toolbar, scroll, tooltip);
  applyZoom();
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => applyZoom());
    observer.observe(scroll);
    resizeObservers.set(container, observer);
  }
}

export function renderTensorArenaViewer(container, analysis, {
  formatBytes = (value) => `${Number(value || 0)} B`,
  onSelectOp = null,
  runtimeEvidence = null,
} = {}) {
  if (!container) return;
  disconnectResizeObserver(container);
  container.replaceChildren();
  const liveness = analysis?.tensor_liveness || {};
  const plan = analysis?.tensor_arena_plan || null;
  if (!plan && liveness.peak_bytes == null) return;

  const heading = element("div", "arena-viewer-heading");
  const status = element("span", `arena-status arena-status-${plan?.status || liveness.status || "not_assessed"}`, String(plan?.evidence_class || (liveness.assessed ? "DERIVED" : "NOT_ASSESSABLE")));
  decorateEvidenceElement(status, plan?.evidence_class || (liveness.assessed ? "DERIVED" : "NOT_ASSESSABLE"), { label: false });
  heading.append(
    element("strong", "arena-viewer-title", plan ? "TFLite ArenaPlanner declared-shape projection" : "Live activation payload"),
    status,
  );
  container.append(heading);

  const metrics = element("div", "arena-metric-strip");
  metrics.append(metric("Live payload", liveness.peak_bytes == null ? "Not assessed" : formatBytes(liveness.peak_bytes), liveness.method || ""));
  if (plan) {
    const artifactBytes = finiteNumber(analysis?.file_size);
    const combinedArenaBytes = finiteNumber(plan.combined_arena_bytes);
    const deploymentFootprint = artifactBytes == null || combinedArenaBytes == null
      ? null
      : artifactBytes + combinedArenaBytes;
    const rootAllocations = (plan.allocations || []).filter((allocation) => allocation?.arena === "kTfLiteArenaRw"
      && allocation?.allocation_status === "allocated" && finiteNumber(allocation?.size_bytes) != null);
    const occupancy = peakRootOccupancy(rootAllocations, Math.max(1, (analysis?.ops || []).length), Number(plan.non_persistent_arena_bytes || 0));
    metrics.append(
      metric("Arena RW", plan.non_persistent_arena_bytes == null ? "Not assessed" : formatBytes(plan.non_persistent_arena_bytes)),
      metric("Persistent", plan.persistent_arena_bytes == null ? "Not assessed" : formatBytes(plan.persistent_arena_bytes)),
      metric(
        "Arena + FlatBuffer",
        deploymentFootprint == null ? "Not assessed" : formatBytes(deploymentFootprint),
        "Declared-shape combined arena projection plus serialized artifact bytes. This excludes runtime code, delegate state, scratch buffers, and allocator overhead.",
      ),
      metric("Root buffers", String(plan.root_allocation_count || 0)),
      metric("Aliases", String(plan.shared_tensor_count || 0)),
      metric("Alignment", `${plan.tensor_alignment_bytes || 0} B`),
      metric("Dynamic signatures", String(plan.dynamic_shape_signature_tensor_count || 0)),
      metric("Sort ties", String(plan.source_comparator_tie_group_count || 0), plan.deterministic_tie_break || ""),
      metric("Peak root occupancy", occupancy.ratio == null ? "Not assessed" : `${(occupancy.ratio * 100).toFixed(1)}% at op #${occupancy.peak_op}`, "Sum of serialized root allocation sizes alive at the peak execution position divided by the projected non-persistent arena. Aliases are not counted as independent roots."),
    );
  }
  container.append(metrics);

  if (plan) {
    renderArenaCanvas(container, analysis, plan, formatBytes, onSelectOp);
    renderAliases(container, plan, onSelectOp);
    renderIssues(container, plan);
    const basis = element("div", "arena-source-basis", `tensorflow/tensorflow@${String(plan.source_commit || "not emitted").slice(0, 12)} | declared shapes | ${plan.tensor_alignment_bytes || 0} B alignment | ${plan.preserve_all_tensors ? "preserve all" : "reuse enabled"}`);
    basis.title = plan.interpretation_boundary || "";
    container.append(basis);
  } else if (!liveness.assessed) {
    const reason = liveness.unassessed_tensors?.[0]?.reason || "Required tensor payloads were not assessable.";
    container.append(element("div", "arena-issue-row", reason));
  }

  const runtimeMemory = runtimeEvidence?.runtime_memory || null;
  const reconciliation = runtimeEvidence?.arena_reconciliation || null;
  if (runtimeMemory && reconciliation) {
    const runtimeHeading = element("div", "arena-viewer-heading arena-runtime-heading");
    const runtimeStatus = element("span", "arena-status arena-status-assessed", runtimeMemory.evidence_class || "MEASURED");
    decorateEvidenceElement(runtimeStatus, runtimeMemory.evidence_class || "MEASURED", { label: false });
    runtimeHeading.append(
      element("strong", "arena-viewer-title", "Observed TFLite arena allocation"),
      runtimeStatus,
    );
    const runtimeMetrics = element("div", "arena-metric-strip arena-runtime-metrics");
    const delta = reconciliation.peak_delta_bytes;
    const deltaLabel = signedBytes(formatBytes, delta);
    runtimeMetrics.append(
      metric("Peak observed", formatBytes(runtimeMemory.peak_combined_arena_bytes), runtimeMemory.interpretation_boundary || ""),
      metric("Final observed", formatBytes(runtimeMemory.final_combined_arena_bytes)),
      metric("Projection delta", deltaLabel),
      metric("Snapshots", String(runtimeMemory.snapshot_count || 0)),
      metric("Runtime-only", String(reconciliation.runtime_only_allocation_count || 0)),
      metric("Prepare temporaries", `${reconciliation.runtime_temporary_allocation_count || 0} / ${formatBytes(reconciliation.runtime_temporary_interval_bytes || 0)}`),
      metric("Size differences", String(reconciliation.size_mismatch_count || 0)),
      metric("Alias differences", String(reconciliation.alias_mismatch_count || 0)),
    );
    const runtimeBasis = element(
      "div",
      "arena-source-basis arena-runtime-basis",
      `tensorflow/tensorflow@${String(runtimeMemory.tensorflow_source_commit || "not emitted").slice(0, 12)} | snapshot ${reconciliation.runtime_snapshot_id} | ledger ${String(runtimeMemory.allocation_ledger_sha256 || "").slice(0, 16)}`,
    );
    runtimeBasis.title = reconciliation.interpretation_boundary || "";
    container.append(runtimeHeading, runtimeMetrics);
    renderRuntimeDifferences(container, reconciliation, formatBytes);
    container.append(runtimeBasis);
  }
}
