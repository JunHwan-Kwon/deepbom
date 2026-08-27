// ── Data computation ──────────────────────────────────────────────────────────
// Rendering only. Histogram, filter statistics, and clustering are computed by
// the Rust/WASM analyzer and passed in as structured data.
function fmtN(v, d = 3) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10)   return v.toFixed(1);
  return v.toFixed(d);
}

function dprCanvas(cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const c = document.createElement("canvas");
  c.width = cssW * dpr; c.height = cssH * dpr;
  c.style.width = cssW + "px"; c.style.height = cssH + "px";
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  return { c, ctx };
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ── Histogram canvas with hover ───────────────────────────────────────────────
function mkHistogram(h) {
  const { bins, bin_min, bin_max, mean, p05, p25, p75, p95, element_count } = h;
  const BINS = bins.length;
  const maxBin = Math.max(1, ...bins);
  const range = (bin_max - bin_min) || 1;
  const totalN = element_count || bins.reduce((a, b) => a + b, 0);

  const W = 320, H = 96;
  const { c: canvas, ctx } = dprCanvas(W, H);
  canvas.className = "whist-canvas";

  const PT = 6; // padding top

  function toX(v) { return (v - bin_min) / range * W; }

  function barColor(i, hover) {
    const center = bin_min + (i + 0.5) * range / BINS;
    if (hover) return "#1e40af";
    if (center < -range * 0.01) return "rgba(59,130,246,0.78)";
    if (center >  range * 0.01) return "rgba(249,115,22,0.78)";
    return "rgba(148,163,184,0.55)";
  }

  function draw(hovBin = -1) {
    ctx.clearRect(0, 0, W, H);
    const CH = H - PT;
    const BW = W / BINS;

    // IQR band
    ctx.fillStyle = "rgba(59,130,246,0.07)";
    ctx.fillRect(toX(p25), PT, toX(p75) - toX(p25), CH);

    // Bars
    for (let i = 0; i < BINS; i++) {
      const bh = bins[i] / maxBin * CH;
      ctx.fillStyle = barColor(i, i === hovBin);
      ctx.fillRect(Math.floor(i * BW) + 0.5, PT + CH - bh, Math.max(1, Math.ceil(BW) - 0.5), bh);
    }

    // P5 / P95 dashed
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = "rgba(100,116,139,0.35)";
    ctx.lineWidth = 1;
    for (const v of [p05, p95]) {
      if (!isFinite(v)) continue;
      const x = toX(v);
      ctx.beginPath(); ctx.moveTo(x, PT); ctx.lineTo(x, H); ctx.stroke();
    }
    ctx.restore();

    // Mean
    if (isFinite(mean)) {
      const x = toX(mean);
      ctx.strokeStyle = "#1e3a8a";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, PT); ctx.lineTo(x, H); ctx.stroke();
    }
  }

  draw();

  // Hover tooltip
  const tip = el("div", "whist-hover-tip");
  tip.style.display = "none";

  canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const bi = Math.min(BINS - 1, Math.max(0, Math.floor(relX * BINS)));
    const lo = bin_min + bi / BINS * range;
    const hi = bin_min + (bi + 1) / BINS * range;
    const count = bins[bi];
    const pct = totalN > 0 ? (count / totalN * 100).toFixed(2) : "?";   // W3: guard totalN=0
    tip.textContent = `[${fmtN(lo, 2)},  ${fmtN(hi, 2)})   ·   ${count.toLocaleString()}  (${pct}%)`;
    tip.style.display = "block";
    const tipW = 240;
    const tipLeft = Math.min(W - tipW - 4, Math.max(4, e.clientX - rect.left - tipW / 2));
    tip.style.left = tipLeft + "px";
    draw(bi);
  });
  canvas.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    draw();
  });

  const wrap = el("div", "whist-chart-region");
  wrap.append(canvas, tip);
  return wrap;
}

// ── X-axis labels ─────────────────────────────────────────────────────────────
function mkXAxis(h) {
  const axis = el("div", "whist-xaxis");
  if (h.bin_min === h.bin_max) {               // W9: single-value tensor — show one centred tick
    const only = el("span", null, fmtN(h.bin_min));
    only.style.textAlign = "center";
    axis.append(only);
  } else {
    axis.append(
      el("span", null, fmtN(h.bin_min)),
      el("span", null, fmtN((h.bin_min + h.bin_max) / 2)),
      el("span", null, fmtN(h.bin_max)),
    );
  }
  return axis;
}

// ── Box-plot row ──────────────────────────────────────────────────────────────
function mkBoxPlot(h) {
  // WASM returns val_min/val_max; accept both spellings for resilience
  const min_val = h.val_min ?? h.min_val ?? 0;
  const max_val = h.val_max ?? h.max_val ?? 0;
  const { p25, p50, p75, p05, p95 } = h;
  const range = (max_val - min_val) || 1;
  const toP = v => Math.max(0, Math.min(100, (v - min_val) / range * 100));

  const W = 320, H = 28;
  const { c, ctx } = dprCanvas(W, H);
  c.className = "whist-bp-canvas";

  const midY = H / 2, boxH = 10;

  // Whisker lines: min→P5 and P95→max
  ctx.strokeStyle = "rgba(100,116,139,0.55)";
  ctx.lineWidth = 1;
  for (const [a, b] of [[min_val, p05], [p95, max_val]]) {
    ctx.beginPath();
    ctx.moveTo(toP(a) / 100 * W, midY);
    ctx.lineTo(toP(b) / 100 * W, midY);
    ctx.stroke();
    // cap
    ctx.beginPath();
    ctx.moveTo(toP(a) / 100 * W, midY - 4);
    ctx.lineTo(toP(a) / 100 * W, midY + 4);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toP(b) / 100 * W, midY - 4);
    ctx.lineTo(toP(b) / 100 * W, midY + 4);
    ctx.stroke();
  }

  // IQR box
  const q1x = toP(p25) / 100 * W;
  const q3x = toP(p75) / 100 * W;
  ctx.fillStyle = "rgba(59,130,246,0.18)";
  ctx.fillRect(q1x, midY - boxH / 2, q3x - q1x, boxH);
  ctx.strokeStyle = "rgba(59,130,246,0.6)";
  ctx.lineWidth = 1;
  ctx.strokeRect(q1x, midY - boxH / 2, q3x - q1x, boxH);

  // Median line
  const mx = toP(p50) / 100 * W;
  ctx.strokeStyle = "#1e40af";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mx, midY - boxH / 2);
  ctx.lineTo(mx, midY + boxH / 2);
  ctx.stroke();

  // Hover tooltip for exact values
  const tip = el("div", "whist-hover-tip");
  tip.style.display = "none";

  const LANDMARKS = [
    [min_val, "min"], [p05, "P5"], [p25, "Q1"], [p50, "median"],
    [p75, "Q3"], [p95, "P95"], [max_val, "max"],
  ];

  c.addEventListener("mousemove", e => {
    const rect = c.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const dataVal = min_val + relX * range;
    // Find nearest landmark
    let nearest = null, nearD = Infinity;
    for (const [v, label] of LANDMARKS) {
      const d = Math.abs(toP(v) / 100 - relX);
      if (d < nearD) { nearD = d; nearest = [v, label]; }
    }
    if (nearest && nearD < 0.06) {
      tip.textContent = `${nearest[1]}  =  ${fmtN(nearest[0], 4)}`;
    } else {
      tip.textContent = `≈ ${fmtN(dataVal, 4)}`;
    }
    tip.style.display = "block";
    const tipLeft = Math.min(W - 64 - 4, Math.max(4, e.clientX - rect.left - 60)); // K2: use canvas W not hardcode 270
    tip.style.left = tipLeft + "px";
    tip.style.top = "-24px";
  });
  c.addEventListener("mouseleave", () => { tip.style.display = "none"; });

  const wrap = el("div", "whist-bp-wrap");
  wrap.append(c, tip);
  return wrap;
}

// ── Stats grid ────────────────────────────────────────────────────────────────
function mkStatsGrid(h) {
  const grid = el("div", "whist-stats-grid");

  const items = [
    ["μ",          fmtN(h.mean)],
    ["σ",          fmtN(h.std_dev)],
    ["min",        fmtN(h.min_val)],
    ["max",        fmtN(h.max_val)],
    ["median",     fmtN(h.p50)],
    ["sparsity",   `${(h.sparsity * 100).toFixed(1)}%`],
    ["entropy",    `${(h.entropy_bits ?? 0).toFixed(2)} bits`],   // W2: guard undefined
    ...(h.range_utilization > 0 ? [["range util", `${(h.range_utilization * 100).toFixed(1)}%`]] : []),
  ];

  for (const [k, v] of items) {
    const cell = el("div", "whist-stat-cell");
    cell.append(el("span", "whist-stat-lbl", k), el("span", "whist-stat-val", v));
    grid.append(cell);
  }
  return grid;
}

// ── Kernel grid ───────────────────────────────────────────────────────────────
function divergingRgb(t) {
  const c = Math.round(Math.min(1, Math.max(0, t)) * 255);
  if (t < 0.5) return [c * 2, c * 2, 255];
  const f = Math.round((t - 0.5) * 2 * 255);
  return [255, 255 - f, 255 - f];
}

function drawKernelCanvas(values, outCh, kH, kW, inCh, oc, globalMin, globalMax, scale) {
  const { c, ctx } = dprCanvas(kW * scale, kH * scale);
  const img = ctx.createImageData(kW * scale, kH * scale);
  const range = (globalMax - globalMin) || 1;
  const isDepthwise = outCh === 1;

  for (let ky = 0; ky < kH; ky++) {
    for (let kx = 0; kx < kW; kx++) {
      let r, g, b;
      if (isDepthwise) {
        const v = values[ky * kW * inCh + kx * inCh + oc];
        [r, g, b] = divergingRgb((v - globalMin) / range);
      } else if (inCh === 3) {
        const base = oc * kH * kW * inCh + ky * kW * inCh + kx * inCh;
        r = Math.round((values[base]     - globalMin) / range * 255);
        g = Math.round((values[base + 1] - globalMin) / range * 255);
        b = Math.round((values[base + 2] - globalMin) / range * 255);
      } else {
        const base = oc * kH * kW * inCh + ky * kW * inCh + kx * inCh;
        let sum = 0;
        for (let ic = 0; ic < inCh; ic++) sum += values[base + ic];
        [r, g, b] = divergingRgb((sum / inCh - globalMin) / range);
      }
      for (let py = 0; py < scale; py++) {
        for (let px = 0; px < scale; px++) {
          const i = ((ky * scale + py) * kW * scale + kx * scale + px) * 4;
          img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// 10 visually distinct hues for cluster coloring
const CLUSTER_PALETTE = [
  '#ef4444','#f97316','#eab308','#22c55e','#06b6d4',
  '#6366f1','#a855f7','#ec4899','#14b8a6','#f43f5e',
];

export function renderKernelGrid(h) {
  // All filter arrays are now computed in WASM and returned as part of WeightHistogram.
  // raw_values: dequantized floats for pixel drawing (empty for 1×1 or large tensors)
  const shape = h.shape;
  const rawValues = h.raw_values;
  const valMin = h.min_val ?? h.val_min ?? 0;   // B2: WASM exports min_val/max_val
  const valMax = h.max_val ?? h.val_max ?? 0;
  const isInputLayer = h.isInputLayer ?? false;
  if (!shape || shape.length !== 4 || !rawValues || rawValues.length === 0) return null;
  const [outCh, kH, kW, inCh] = shape;
  if (kH < 2 && kW < 2) return null;

  const isDepthwise = outCh === 1;
  const numFilters = isDepthwise ? inCh : outCh;
  if (numFilters < 1) return null;

  // Read WASM-computed filter arrays (no JS computation needed)
  const norms      = h.filter_norms      ?? new Float32Array(numFilters);
  const sortOrder  = h.filter_sort_order ?? Array.from({ length: numFilters }, (_, i) => i);
  const signRatios = h.filter_sign_ratios ?? new Float32Array(numFilters).fill(0.5);
  const clusterMap = h.cluster_map ?? new Int32Array(numFilters).fill(-1);
  const clusterCount = h.cluster_count ?? 0;

  let maxNorm = 0;
  for (let i = 0; i < norms.length; i++) if (norms[i] > maxNorm) maxNorm = norms[i]; // C2: avoid spread stack-overflow
  if (maxNorm === 0) maxNorm = 1;

  const lowNormCount = h.low_norm_filters ?? 0;
  const effRank   = h.eff_rank ?? numFilters;
  const diversity = h.diversity ?? null;

  const scale = Math.max(3, Math.min(20, Math.floor(60 / Math.max(kH, kW))));
  const INITIAL_CAP = 64;

  const section = el("div", "kgrid-section");

  // ── Header row
  const metaRow = el("div", "kgrid-meta");
  const titleWrap = el("div", "kgrid-title-wrap");
  const titleText = isDepthwise
    ? `Kernel Filters · ${numFilters} ch · ${kH}×${kW}`
    : `Kernel Filters · ${outCh} out · ${kH}×${kW}×${inCh}`;
  titleWrap.append(
    el("span", "kgrid-title", titleText),
    el("span", "kgrid-subtitle", "sorted by L2 norm ↓"),
  );
  if (isInputLayer) {
    const badge = el("span", "kbadge-input-layer", "1st layer");
    badge.title = "First-layer convolutions learn edge/color detectors. Low EffRank is expected here.";
    titleWrap.append(badge);
  }
  metaRow.append(titleWrap);

  const badgeRow = el("div", "kgrid-badges");
  const addBadge = (label, value, cls = "") => {
    const b = el("span", `kgrid-badge ${cls}`);
    b.append(el("span", "kbadge-label", label), el("span", "kbadge-value", value));
    badgeRow.append(b);
  };

  addBadge("Low norm", `${lowNormCount} / ${numFilters}`,
    lowNormCount / numFilters > 0.1 ? "kbadge-warn" : "");
  addBadge("Eff. Rank", `${effRank} / ${numFilters}`,
    isInputLayer && effRank / numFilters < 0.25 ? "kbadge-expected" : "");
  if (diversity != null)
    addBadge("Diversity", diversity.toFixed(3),
      diversity > 0.7 ? "kbadge-good" : diversity < 0.3 ? "kbadge-warn" : "");
  if (clusterCount > 0)
    addBadge("Redundant", `${clusterCount} group${clusterCount > 1 ? "s" : ""}`, "kbadge-warn");

  metaRow.append(badgeRow);
  section.append(metaRow);

  // ── Cluster legend (only when clusters exist)
  if (clusterCount > 0) {
    const cleg = el("div", "kgrid-cluster-legend");
    cleg.append(el("span", "kgrid-cluster-legend-title", "Redundant groups:"));
    for (let ci = 0; ci < Math.min(clusterCount, CLUSTER_PALETTE.length); ci++) {
      const dot = el("span", "kgrid-cluster-dot");
      dot.style.background = CLUSTER_PALETTE[ci];
      const members = [];
      for (let f = 0; f < numFilters; f++) if (clusterMap[f] === ci) members.push(f);
      dot.title = `Cluster ${ci}: filters ${members.join(", ")}`;
      const cnt = el("span", "kgrid-cluster-cnt", `×${members.length}`);
      const grp = el("span", "kgrid-cluster-grp");
      grp.append(dot, cnt);
      cleg.append(grp);
    }
    section.append(cleg);
  }

  // ── Color legend
  const legend = el("div", "kgrid-legend");
  legend.innerHTML = `<span class="kleg kleg-lo"></span><span>neg</span>`
    + `<span class="kleg kleg-mid"></span><span>zero</span>`
    + `<span class="kleg kleg-hi"></span><span>pos</span>`
    + `<span class="kgrid-sign-legend-sep"></span>`
    + `<span class="kgrid-sign-legend-bar" style="background:rgba(249,115,22,0.7);width:14px;height:6px;display:inline-block;border-radius:1px;vertical-align:middle;margin-right:3px"></span>`
    + `<span style="font-size:9px;opacity:0.7">+weight ratio</span>`;
  section.append(legend);

  const grid = el("div", "kgrid");
  section.append(grid);

  function renderRange(limit) {
    grid.replaceChildren();
    const n = Math.min(limit, numFilters);
    for (let fi = 0; fi < n; fi++) {
      const f = sortOrder[fi];
      const norm = norms[f];
      const normFrac = norm / maxNorm;
      const isLowNorm = norm < 0.02 * maxNorm;
      const cid = clusterMap[f];
      const clusterColor = cid >= 0 && cid < CLUSTER_PALETTE.length ? CLUSTER_PALETTE[cid] : null;

      const cell = el("div", "kgrid-cell");
      if (isLowNorm) cell.classList.add("kgrid-cell--dead");
      if (clusterColor) {
        cell.style.setProperty("--cluster-color", clusterColor);
        cell.classList.add("kgrid-cell--clustered");
        cell.title = `filter #${f} — Redundant group ${cid} (cosine_sim ≥ 0.90)`;
      }

      const cvs = drawKernelCanvas(rawValues, outCh, kH, kW, inCh, f, valMin, valMax, scale);
      if (!clusterColor) cvs.title = `filter ${f} · L2 = ${isFinite(norm) ? norm.toFixed(3) : "n/a"}`; // W4
      cell.append(cvs);

      // Norm bar
      const normBar = el("div", "kgrid-norm-bar-wrap");
      const bar = el("div", "kgrid-norm-bar");
      bar.style.width = `${normFrac * 100}%`;
      bar.style.opacity = `${0.4 + normFrac * 0.6}`;
      normBar.append(bar);
      cell.append(normBar);

      // Sign ratio bar: orange = positive fraction, blue = negative fraction
      const sr = signRatios[f] ?? 0.5;
      const signBar = el("div", "kgrid-sign-bar-wrap");
      signBar.title = `+${(sr * 100).toFixed(0)}% pos  /  ${((1 - sr) * 100).toFixed(0)}% neg`;
      const posBar = el("div", "kgrid-sign-bar-pos");
      posBar.style.width = `${sr * 100}%`;
      const negBar = el("div", "kgrid-sign-bar-neg");
      negBar.style.width = `${(1 - sr) * 100}%`;
      signBar.append(posBar, negBar);
      cell.append(signBar);

      // Label
      const lbl = el("div", "kgrid-cell-lbl");
      lbl.textContent = `#${f}`;
      lbl.title = `L2: ${isFinite(norm) ? norm.toFixed(4) : "n/a"} · +${(sr * 100).toFixed(0)}% pos`; // W4
      cell.append(lbl);

      grid.append(cell);
    }
  }

  renderRange(INITIAL_CAP);

  if (numFilters > INITIAL_CAP) {
    const btn = el("button", "kgrid-show-more", `Show all ${numFilters} filters`);
    btn.addEventListener("click", () => { renderRange(numFilters); btn.remove(); });
    section.append(btn);
  }

  return section;
}

// ── Top-level render ──────────────────────────────────────────────────────────
export function renderWeightHistogram(h) {
  const wrap = el("div", "whist-tensor");

  // Header
  const header = el("div", "whist-header");
  const shortName = (h.name || "").split("/").pop() || h.name || "tensor";
  const nameEl = el("span", "whist-name", shortName);
  if (h.name && h.name !== shortName) nameEl.title = h.name;
  const badge = el("span", "whist-dtype-badge", h.dtype);
  header.append(nameEl, badge);

  const meta = el("div", "whist-meta");
  const shapeTxt = Array.isArray(h.shape) ? h.shape.join("×") : "";
  meta.textContent = `${(h.element_count || 0).toLocaleString()} params${shapeTxt ? "  ·  [" + shapeTxt + "]" : ""}`;
  wrap.append(header, meta);

  // Histogram canvas
  if (h.bins?.length) {
    wrap.append(mkHistogram(h));
    wrap.append(mkXAxis(h));
    wrap.append(mkBoxPlot(h));
  }

  wrap.append(mkStatsGrid(h));

  const kg = renderKernelGrid(h);
  if (kg) wrap.append(kg);

  return wrap;
}
