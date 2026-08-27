import { runtimeSignal, td } from "./dom.js";
import { backendProfileText } from "./runtime.js";
import { benchmarkNoise, movingAverage } from "./format.js";

export function p99EvidenceForSampleCount(value) {
  const measuredRuns = Math.max(0, Number.parseInt(value, 10) || 0);
  return measuredRuns < 100
    ? {
      status: "underpowered",
      label: `Underpowered for p99 (${measuredRuns}/100 minimum empirical positions)`,
      detail: `Nearest-rank p99 is descriptive only: ${measuredRuns} measured runs provide fewer than 100 empirical positions.`,
      tone: "warn",
    }
    : {
      status: "descriptive",
      label: `Empirical p99 (${measuredRuns} runs)`,
      detail: `Nearest-rank p99 uses ${measuredRuns} measured runs; no confidence interval, independence, or stationarity guarantee is asserted.`,
      tone: "good",
    };
}

export function runtimeReadinessSignals(analysis, {
  backendValue = "auto",
  navigatorLike = globalThis.navigator,
  warmup = 1,
  runs = 50,
} = {}) {
  const unsupported = analysis.inputs
    .filter((tensor) => tensor.value_kind && tensor.value_kind !== "tensor"
      || !["FLOAT32", "INT32", "UINT8", "INT8"].includes(tensor.dtype))
    .map((tensor) => `${tensor.name || `input_${tensor.index}`}=${tensor.value_kind && tensor.value_kind !== "tensor" ? tensor.value_kind : tensor.dtype}`);
  const acceleratorReady = "gpu" in navigatorLike || "ml" in navigatorLike;
  const backendText = backendProfileText(analysis.format, backendValue, navigatorLike);
  const p99Evidence = p99EvidenceForSampleCount(runs);
  return [
    runtimeSignal("Model", "Local file", "good"),
    runtimeSignal(
      "Input",
      unsupported.length ? "Unsupported dtype" : "Synthetic tensor",
      unsupported.length ? "risk" : "warn",
    ),
    runtimeSignal("Backend path", backendText, analysis.format === "onnx" || acceleratorReady ? "good" : "warn"),
    runtimeSignal("Profile", `1 cold / ${warmup} warmup / ${runs} measured`, "good"),
    runtimeSignal("Statistics", "Nearest-rank / population SD", "good"),
    runtimeSignal("p99 evidence", p99Evidence.label, p99Evidence.tone),
  ];
}

export function appendBenchmarkRow(benchmarkBody, backend, status) {
  const tr = document.createElement("tr");
  tr.append(td(backend), td("-"), td("-"), td("-"), td("-"), td("-"), td("-"), td("-"), td("-"), td("-"), td("-"), td(status, "wrap"));
  benchmarkBody.append(tr);
  return tr;
}

export function updateBenchmarkRow(row, result) {
  const stats = result.stats || null;
  const noise = result.timings?.length >= 3 ? result.noiseDiagnostics || benchmarkNoise(result.timings) : null;
  const p99Evidence = p99EvidenceForSampleCount(result.timings?.length || 0);
  const p99Cell = td(stats ? `${stats.p99.toFixed(2)} ms` : "-", `numeric${stats && p99Evidence.status === "underpowered" ? " tail-underpowered" : ""}`);
  if (stats) p99Cell.title = p99Evidence.detail;

  row.replaceChildren(
    td(result.backend),
    td(result.compileMs == null ? "-" : `${result.compileMs.toFixed(2)} ms`, "numeric"),
    td(result.firstRunMs == null ? "-" : `${result.firstRunMs.toFixed(2)} ms`, "numeric"),
    td(stats ? `${stats.p50.toFixed(2)} ms` : "-", "numeric"),
    td(stats ? `${stats.p90.toFixed(2)} ms` : "-", "numeric"),
    td(stats ? `${stats.p95.toFixed(2)} ms` : "-", "numeric"),
    p99Cell,
    td(stats ? `${stats.mean.toFixed(2)} ms` : "-", "numeric"),
    td(noise ? `${noise.trimmedP50.toFixed(2)} ms` : "-", "numeric"),
    td(stats ? formatCv(stats.cv) : "-", "numeric"),
    td(noise ? noise.trendLabel : "-", "numeric bench-trend"),
    td(result.status, "wrap"),
  );

  // Remove stale chart row if present
  const existing = row.nextElementSibling;
  if (existing?.classList.contains("bench-chart-row")) existing.remove();

  // Insert time-series chart row when we have raw timings
  if (result.timings?.length >= 5) {
    const chartRow = document.createElement("tr");
    chartRow.className = "bench-chart-row";
    const cell = document.createElement("td");
    cell.colSpan = 12;
    cell.className = "bench-chart-cell";
    chartRow.append(cell);
    row.after(chartRow);
    renderBenchmarkTimeSeries(cell, result.timings, noise);
  }
}

// ── SVG time-series chart ──────────────────────────────────────────────────

function renderBenchmarkTimeSeries(container, timings, noise) {
  const n = timings.length;
  if (n < 2) return;

  const W = 800;
  const H = 90;
  const PAD = { top: 8, right: 14, bottom: 20, left: 48 };
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top - PAD.bottom;

  const rawMin = Math.min(...timings);
  const rawMax = Math.max(...timings);
  const span = rawMax - rawMin || rawMin * 0.1 || 1;
  const yMin = Math.max(0, rawMin - span * 0.12);
  const yMax = rawMax + span * 0.12;
  const yRange = yMax - yMin;

  const mean = timings.reduce((a, b) => a + b, 0) / n;
  const stddev = Math.sqrt(timings.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const limit = mean + 2.5 * stddev;

  const xScale = (i) => PAD.left + (n > 1 ? (i / (n - 1)) * pw : pw / 2);
  const yScale = (v) => PAD.top + (1 - (v - yMin) / yRange) * ph;

  // Moving average window: larger window for more runs to smooth further
  const maWindow = Math.min(11, Math.max(3, Math.floor(n / 8)));
  const ma = movingAverage(timings, maWindow);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "bench-chart", preserveAspectRatio: "none" });

  // Plot background
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: pw, height: ph, class: "bench-chart-bg" }));

  // Horizontal grid lines at 25/50/75%
  for (const frac of [0.25, 0.5, 0.75]) {
    const y = yScale(yMin + frac * yRange);
    svg.append(svgEl("line", { x1: PAD.left, x2: PAD.left + pw, y1: y, y2: y, class: "bench-chart-grid" }));
  }

  // Mean reference line
  const meanY = yScale(mean);
  svg.append(svgEl("line", { x1: PAD.left, x2: PAD.left + pw, y1: meanY, y2: meanY, class: "bench-chart-mean" }));

  // Outlier threshold line (only when meaningful)
  if (stddev > 0 && limit < yMax * 0.97) {
    const limY = yScale(limit);
    svg.append(svgEl("line", { x1: PAD.left, x2: PAD.left + pw, y1: limY, y2: limY, class: "bench-chart-limit" }));
  }

  // Individual run dots — draw normal first, then outliers on top
  const normalPts = [], outlierPts = [];
  for (let i = 0; i < n; i++) {
    const pt = [xScale(i), yScale(timings[i])];
    (timings[i] > limit ? outlierPts : normalPts).push(pt);
  }
  for (const [cx, cy] of normalPts) {
    svg.append(svgEl("circle", { cx, cy, r: "2", class: "bench-chart-dot" }));
  }
  for (const [cx, cy] of outlierPts) {
    svg.append(svgEl("circle", { cx, cy, r: "3.5", class: "bench-chart-outlier" }));
  }

  // Moving average line
  const maPath = ma
    .map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`)
    .join(" ");
  svg.append(svgEl("path", { d: maPath, class: "bench-chart-ma", fill: "none" }));

  // Trend overlay (dashed line through first and last MA point, only when significant)
  if (noise && Math.abs(noise.trendSlope) >= 0.03 && n >= 10) {
    const x0 = xScale(0);
    const x1 = xScale(n - 1);
    const y0 = yScale(ma[0]);
    const y1 = yScale(ma[n - 1]);
    svg.append(svgEl("line", { x1: x0, y1: y0, x2: x1, y2: y1, class: "bench-chart-trend" }));
  }

  // Y-axis labels (4 ticks)
  for (let i = 0; i <= 3; i++) {
    const v = yMin + (i / 3) * yRange;
    const y = yScale(v);
    const label = svgEl("text", { x: PAD.left - 5, y: y + 3.5, class: "bench-chart-label", "text-anchor": "end" });
    label.textContent = v < 10 ? v.toFixed(1) : v.toFixed(0);
    svg.append(label);
  }

  // X-axis "run" label centred
  const xLabel = svgEl("text", { x: PAD.left + pw / 2, y: H - 3, class: "bench-chart-label", "text-anchor": "middle" });
  xLabel.textContent = `run # (${n} total · MA window ${maWindow})`;
  svg.append(xLabel);

  container.append(svg);

  // Text summary beneath chart
  const summary = document.createElement("div");
  summary.className = "bench-chart-summary";
  const parts = [];
  if (noise) {
    if (noise.gcSpikeCount > 0) parts.push(`${noise.gcSpikeCount} GC spike${noise.gcSpikeCount > 1 ? "s" : ""} (isolated)`);
    if (noise.outlierCount > noise.gcSpikeCount) {
      parts.push(`${noise.outlierCount - noise.gcSpikeCount} sustained outlier${noise.outlierCount - noise.gcSpikeCount > 1 ? "s" : ""}`);
    }
    if (noise.trendLabel !== "stable" && noise.trendLabel !== "—") {
      parts.push(`trend ${noise.trendLabel}${Math.abs(noise.trendSlope) > 0.3 ? " ⚠ possible throttling" : ""}`);
    }
    parts.push(`trimmed p50 ${noise.trimmedP50.toFixed(2)} ms (top+bottom 10% removed)`);
  }
  summary.textContent = parts.length > 0
    ? parts.join(" · ")
    : `clean — no spikes or trend detected · trimmed p50 ${noise?.trimmedP50.toFixed(2) ?? "—"} ms`;
  container.append(summary);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function formatCv(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}
