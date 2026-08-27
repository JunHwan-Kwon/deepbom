const LANDSCAPE_COLOR_STOPS = ["#ffffff", "#f6e7cf", "#e0b06a", "#b87936", "#7a4a1e", "#3a2410"];

function themeColor(token, fallback) {
  if (typeof document === "undefined") return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(token).trim() || fallback;
}

export function modelFilterColor(value, extreme) {
  if (extreme < 1e-10) return "#e2e8f0";
  const normalized = Math.max(-1, Math.min(1, value / extreme));
  const white = Math.round(255 * (1 - Math.abs(normalized)));
  return normalized < 0 ? `rgb(${white},${white},255)` : `rgb(255,${white},${white})`;
}

export function drawMvFilter(canvas, entry, outputChannelIndex) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!entry || !entry.spatial_flat?.length) {
    context.fillStyle = themeColor("--viz-node-muted", "#1e293b");
    context.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const { kh, kw, oc } = entry;
  const outputChannel = Math.min(outputChannelIndex, oc - 1);
  const base = outputChannel * kh * kw;
  const slice = entry.spatial_flat.slice(base, base + kh * kw);
  const extreme = Math.max(...slice.map(Math.abs), 1e-10);
  const cellWidth = canvas.width / kw;
  const cellHeight = canvas.height / kh;

  for (let h = 0; h < kh; h += 1) {
    for (let w = 0; w < kw; w += 1) {
      context.fillStyle = modelFilterColor(slice[h * kw + w] ?? 0, extreme);
      context.fillRect(
        Math.floor(w * cellWidth),
        Math.floor(h * cellHeight),
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }

  if (kh <= 7 && kw <= 7) {
    context.strokeStyle = "#0f172a";
    context.lineWidth = 0.8;
    for (let h = 1; h < kh; h += 1) {
      context.beginPath();
      context.moveTo(0, h * cellHeight);
      context.lineTo(canvas.width, h * cellHeight);
      context.stroke();
    }
    for (let w = 1; w < kw; w += 1) {
      context.beginPath();
      context.moveTo(w * cellWidth, 0);
      context.lineTo(w * cellWidth, canvas.height);
      context.stroke();
    }
  }
}

export function modelDepthValues(tomography = [], state = {}) {
  return tomography.map((entry) => {
    switch (state.metric) {
      case "l2": {
        const outputChannel = Math.min(Number(state.ocIdx || 0), entry.oc - 1);
        return entry.l2_per_oc?.[outputChannel] ?? 0;
      }
      case "haar_ll": return entry.haar_ll ?? 0;
      case "haar_lh": return entry.haar_lh ?? 0;
      case "haar_hl": return entry.haar_hl ?? 0;
      case "haar_hh": return entry.haar_hh ?? 0;
      case "snr": return entry.quant_snr_valid ? Math.max(0, 60 - entry.quant_snr_db) : 0;
      default: return 0;
    }
  });
}

export function drawMvDepth(canvas, tomography, state) {
  const context = canvas.getContext("2d");
  const count = tomography.length;
  const width = canvas.width;
  const height = canvas.height;
  const left = 6;
  const right = 6;
  const top = 8;
  const bottom = 18;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  context.clearRect(0, 0, width, height);
  if (!count) return;

  const values = modelDepthValues(tomography, state);
  const maximum = Math.max(...values, 1e-10);
  const toX = (index) => left + (index / (count - 1 || 1)) * innerWidth;
  const toY = (value) => top + innerHeight - (value / maximum) * innerHeight;

  context.strokeStyle = themeColor("--viz-grid", "#1e293b");
  context.lineWidth = 0.5;
  for (let tick = 0; tick <= 3; tick += 1) {
    const y = top + (tick / 3) * innerHeight;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(left + innerWidth, y);
    context.stroke();
  }

  context.beginPath();
  context.moveTo(toX(0), toY(0));
  values.forEach((value, index) => context.lineTo(toX(index), toY(value)));
  context.lineTo(toX(count - 1), toY(0));
  context.closePath();
  context.fillStyle = "rgba(148,163,184,0.12)";
  context.fill();

  context.beginPath();
  values.forEach((value, index) => (index === 0
    ? context.moveTo(toX(index), toY(value))
    : context.lineTo(toX(index), toY(value))));
  context.strokeStyle = themeColor("--viz-meta", "#94a3b8");
  context.lineWidth = 1.5;
  context.stroke();

  const selectedIndex = Math.max(0, Math.min(Number(state.layerIdx || 0), count - 1));
  const selectedX = toX(selectedIndex);
  context.strokeStyle = themeColor("--accent", "#f59e0b");
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(selectedX, top);
  context.lineTo(selectedX, top + innerHeight);
  context.stroke();
  context.beginPath();
  context.arc(selectedX, toY(values[selectedIndex] ?? 0), 3.5, 0, Math.PI * 2);
  context.fillStyle = themeColor("--accent", "#f59e0b");
  context.fill();

  context.fillStyle = themeColor("--viz-meta", "#64748b");
  context.font = "9px monospace";
  context.textAlign = "center";
  context.fillText("layer index  →", left + innerWidth / 2, height - 4);
}

export function landscapeColor(value) {
  const normalized = Math.max(0, Math.min(1, Number.isNaN(value) ? 0 : value));
  const segmentCount = LANDSCAPE_COLOR_STOPS.length - 1;
  const index = Math.min(segmentCount - 1, Math.floor(normalized * segmentCount));
  const fraction = normalized * segmentCount - index;
  const first = LANDSCAPE_COLOR_STOPS[index];
  const second = LANDSCAPE_COLOR_STOPS[index + 1];
  const interpolate = (offset) => {
    const start = Number.parseInt(first.slice(offset, offset + 2), 16);
    const end = Number.parseInt(second.slice(offset, offset + 2), 16);
    return Math.round(start + (end - start) * fraction);
  };
  return `rgb(${interpolate(1)},${interpolate(3)},${interpolate(5)})`;
}

export function drawLandscapeCanvas(canvas, grid, gridSize, maximum) {
  const context = canvas.getContext("2d");
  const cellWidth = canvas.width / gridSize;
  const cellHeight = canvas.height / gridSize;
  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const value = grid[row]?.[column];
      context.fillStyle = landscapeColor(Number.isNaN(value) ? 0 : value / (maximum || 1));
      context.fillRect(
        Math.round(column * cellWidth),
        Math.round((gridSize - 1 - row) * cellHeight),
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }
  const centerX = (gridSize / 2) * cellWidth;
  const centerY = (gridSize / 2) * cellHeight;
  context.beginPath();
  context.arc(centerX, centerY, 4, 0, Math.PI * 2);
  context.fillStyle = "white";
  context.fill();
  context.strokeStyle = "black";
  context.lineWidth = 1.3;
  context.stroke();
}

export function buildDualRadialSvg(firstRadial, secondRadial) {
  const radii = firstRadial?.rc ?? secondRadial?.rc;
  if (!radii?.length) return null;
  const namespace = "http://www.w3.org/2000/svg";
  const width = 380;
  const height = 150;
  const left = 44;
  const right = 14;
  const top = 14;
  const bottom = 32;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const radiusMaximum = Math.max(...radii, 1e-8);
  const values = [...(firstRadial?.mu ?? []), ...(secondRadial?.mu ?? [])].filter((value) => !Number.isNaN(value));
  const valueMaximum = Math.max(...values, 1e-8);
  const toX = (radius) => left + (radius / radiusMaximum) * innerWidth;
  const toY = (value) => top + innerHeight - Math.max(0, Math.min(1, value / valueMaximum)) * innerHeight;

  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.style.cssText = "overflow:visible;display:block";

  for (let index = 0; index <= 4; index += 1) {
    const y = top + (index / 4) * innerHeight;
    const gridLine = document.createElementNS(namespace, "line");
    gridLine.setAttribute("x1", left);
    gridLine.setAttribute("x2", left + innerWidth);
    gridLine.setAttribute("y1", y);
    gridLine.setAttribute("y2", y);
    gridLine.setAttribute("stroke", "#334155");
    gridLine.setAttribute("stroke-width", "0.5");
    svg.append(gridLine);
  }

  const drawLine = (radial, stroke, fillOpacity) => {
    if (!radial?.mu?.length) return;
    const { rc, mu, sem } = radial;
    if (sem?.length) {
      const upper = rc.map((radius, index) => `${toX(radius)},${toY(mu[index] + sem[index])}`);
      const lower = [...rc].reverse().map((radius, index) => `${toX(radius)},${toY(mu[rc.length - 1 - index] - sem[rc.length - 1 - index])}`);
      const band = document.createElementNS(namespace, "polygon");
      band.setAttribute("points", [...upper, ...lower].join(" "));
      band.setAttribute("fill", stroke);
      band.setAttribute("fill-opacity", fillOpacity);
      svg.append(band);
    }
    const line = document.createElementNS(namespace, "polyline");
    line.setAttribute("points", rc.map((radius, index) => `${toX(radius)},${toY(mu[index])}`).join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", stroke);
    line.setAttribute("stroke-width", "1.8");
    svg.append(line);
    rc.forEach((radius, index) => {
      const point = document.createElementNS(namespace, "circle");
      point.setAttribute("cx", toX(radius));
      point.setAttribute("cy", toY(mu[index]));
      point.setAttribute("r", 2.5);
      point.setAttribute("fill", stroke);
      svg.append(point);
    });
  };

  drawLine(firstRadial, "#b87936", "0.18");
  drawLine(secondRadial, "#3b82f6", "0.15");

  const appendText = (x, y, content, attributes = {}) => {
    const text = document.createElementNS(namespace, "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y);
    text.setAttribute("font-size", "9");
    text.setAttribute("fill", "#94a3b8");
    Object.entries(attributes).forEach(([key, value]) => text.setAttribute(key, value));
    text.textContent = content;
    svg.append(text);
  };
  appendText(left + innerWidth / 2, height - 4, "perturbation radius r = √(α²+β²) (× ‖W‖)", { "text-anchor": "middle" });
  appendText(10, top + innerHeight / 2, "Δ drift", { "text-anchor": "middle", transform: `rotate(-90, 10, ${top + innerHeight / 2})` });
  return svg;
}
