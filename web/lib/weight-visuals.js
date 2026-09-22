// Display projections only. The hash-bound numerical IR remains unchanged.
export const escapeXml = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
export const countLabel = value => value == null ? "Not assessed" : BigInt(value).toLocaleString("en-US");
export function numberLabel(value) {
  if (value == null || !Number.isFinite(value)) return "Not assessed";
  if (Object.is(value, -0)) return "−0";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude < 0.001 || magnitude >= 10000
    ? value.toExponential(2).replace("e+", "e")
    : Number(value.toPrecision(5)).toString();
}
export function countRatio(numerator, denominator) {
  const n = BigInt(numerator), d = BigInt(denominator);
  return d === 0n ? null : Number(n * 1_000_000n / d) / 1_000_000;
}
export function zeroLabel(statistics) {
  if (!statistics || statistics.unsafe_integer_count !== "0") return "Not assessed";
  const ratio = countRatio(statistics.zero_count, statistics.finite_count);
  return ratio == null ? "No finite values" : `${(ratio * 100).toFixed(2)}%`;
}
export function histogramWindow(rows) {
  let first = Infinity, last = -1, edges = null;
  for (const row of rows) {
    const histogram = row.statistics?.histogram;
    if (!histogram) continue;
    edges ||= histogram.edges;
    histogram.counts.forEach((count, index) => {
      if (BigInt(count) > 0n) { first = Math.min(first, index); last = Math.max(last, index); }
    });
  }
  return last < 0 ? null : { first: Math.max(0, first - 1), last: Math.min(edges.length - 2, last + 1), edges };
}
export function intervalLabel(histogram, index) {
  const close = index === histogram.counts.length - 1 ? "]" : ")";
  return `[${histogram.edges[index]}, ${histogram.edges[index + 1]}${close}`;
}
export function createConnectionIndex(model) {
  const bindings = new Map((model?.weight_bindings?.bindings || []).map(binding => [binding.id, binding]));
  const operations = new Map((model?.program?.operations || []).map(op => [op.id, op]));
  const portsByValue = new Map(), bindingsByStorage = new Map();
  for (const port of model?.program?.ports || []) {
    if (!portsByValue.has(port.value_ref)) portsByValue.set(port.value_ref, []);
    portsByValue.get(port.value_ref).push(port);
  }
  for (const binding of bindings.values()) {
    if (!bindingsByStorage.has(binding.storage_ref)) bindingsByStorage.set(binding.storage_ref, []);
    bindingsByStorage.get(binding.storage_ref).push(binding.id);
  }
  return { bindings, operations, portsByValue, bindingsByStorage };
}
export function linkedOperations(row, model, lookup = createConnectionIndex(model)) {
  const {bindings, operations, portsByValue} = lookup;
  const links = new Map();
  for (const id of row.binding_refs || []) {
    const binding = bindings.get(id), op = operations.get(binding?.operation_ref);
    if (!op) continue;
    if (!links.has(op.id)) links.set(op.id, { operation: op, ports: [] });
    links.get(op.id).ports.push(binding.parameter_role || binding.port_ref);
  }
  // Activation captures are associated through serialized value references.
  if (row.value_ref) for (const port of portsByValue.get(row.value_ref) || []) {
    const op = operations.get(port.operation_ref);
    if (!op) continue;
    if (!links.has(op.id)) links.set(op.id, { operation: op, ports: [] });
    links.get(op.id).ports.push(`${port.direction || "value"} ${port.position ?? ""}`.trim());
  }
  return [...links.values()];
}
export const operationLabel = op => `${op.name || op.native_op?.name || op.id}${Number.isSafeInteger(op.native_index) ? ` · op ${op.native_index}` : ""}`;

function shell(width, height, title, palette, body, description = "") {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(title)}"><title>${escapeXml(title)}</title><desc>${escapeXml(description)}</desc><rect width="${width}" height="${height}" fill="${palette.surface}"/><g font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" fill="${palette.ink}">${body}</g></svg>`;
}
function axisTicks(window, x, width, y, palette, maxTicks = 5) {
  const n = window.last - window.first + 1;
  const length = Math.min(n + 1, maxTicks);
  const indices = [...new Set(Array.from({ length }, (_, i) => window.first + Math.round(i * n / (length - 1))))];
  return indices.map(index => `<text x="${x + (index - window.first) * width / n}" y="${y}" text-anchor="${index === window.first ? "start" : index === window.last + 1 ? "end" : "middle"}" fill="${palette.muted}">${escapeXml(numberLabel(window.edges[index]))}</text>`).join("");
}
export function histogramSvg(row, palette, { logCount = false } = {}) {
  const histogram = row.statistics?.histogram, window = histogramWindow([row]);
  if (!window) return null;
  const W = 760, H = 298, x = 72, y = 42, width = 664, height = 186;
  const counts = histogram.counts.slice(window.first, window.last + 1);
  const maximum = counts.reduce((max, value) => BigInt(value) > max ? BigInt(value) : max, 1n);
  const scale = count => logCount ? Math.log1p(Number(count)) / Math.log1p(Number(maximum)) : countRatio(count, maximum);
  let body = `<text x="${x}" y="20" fill="${palette.muted}">COUNT${logCount ? " · log(1 + count)" : " · linear"}</text>`;
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const count = logCount ? Math.expm1(fraction * Math.log1p(Number(maximum))) : fraction * Number(maximum);
    const position = y + height * (1 - fraction);
    body += `<path d="M${x} ${position}H${x + width}" stroke="${palette.line}"/><text x="${x - 10}" y="${position + 4}" text-anchor="end" fill="${palette.muted}">${escapeXml(numberLabel(count))}</text>`;
  }
  counts.forEach((count, i) => {
    const index = i + window.first, barHeight = scale(count) * height;
    const color = histogram.edges[index + 1] <= 0 ? palette.blue : palette.accent;
    const label = escapeXml(`${intervalLabel(histogram,index)}: ${countLabel(count)} values`);
    body += `<g data-bin="${index}" tabindex="0" role="button" aria-label="${label}"><title>${label}</title><rect x="${x + i * width / counts.length}" y="${y}" width="${width / counts.length}" height="${height}" fill="transparent"/><rect x="${x + i * width / counts.length + 1}" y="${y + height - barHeight}" width="${Math.max(1, width / counts.length - 2)}" height="${barHeight}" fill="${color}" fill-opacity="0.85"/></g>`;
  });
  body += axisTicks(window, x, width, y + height + 24, palette);
  body += `<text x="${x}" y="282" fill="${palette.muted}">VALUE · fixed, unequal-width log₂ bins · occupied range</text>`;
  return shell(W, H, `${row.name || row.native_locator} — value distribution`, palette, body, "Equal screen widths represent different numeric intervals. Bar height is count, not probability density. Select a bin to inspect its exact interval and count.");
}
export function distributionMapSvg(rows, selected, palette) {
  const window = histogramWindow(rows);
  if (!window) return null;
  const x = 210, width = 528, top = 36, rowHeight = 28, H = top + rows.length * rowHeight + 43;
  const n = window.last - window.first + 1;
  let body = `<text x="12" y="17" fill="${palette.muted}">TENSOR</text><text x="${x}" y="17" fill="${palette.muted}">DISTRIBUTION · row peak = full color</text>`;
  rows.forEach((row, i) => {
    const y = top + i * rowHeight, histogram = row.statistics?.histogram;
    body += `<g data-tensor-id="${escapeXml(row.id)}" role="button" tabindex="0" aria-label="Select ${escapeXml(row.name || row.native_locator)}"><rect x="0" y="${y - 4}" width="760" height="27" fill="${selected === row.id ? palette.selected : palette.surface}"/><text x="12" y="${y + 13}">${escapeXml((row.name || row.native_locator || row.id).slice(0,25))}</text><title>${escapeXml(row.name || row.native_locator || row.id)}</title>`;
    if (histogram && BigInt(row.statistics.finite_count) > 0n) {
      const maximum = histogram.counts.reduce((max, c) => BigInt(c) > max ? BigInt(c) : max, 1n);
      for (let index = window.first; index <= window.last; index++) {
        const count = histogram.counts[index], ratio = countRatio(count, maximum);
        body += `<rect x="${x + (index - window.first) * width / n}" y="${y}" width="${Math.max(1,width / n - 1)}" height="18" fill="${histogram.edges[index+1] <= 0 ? palette.blue : palette.accent}" fill-opacity="${BigInt(count) ? 0.15 + 0.85 * ratio : 0.04}"><title>${escapeXml(`${row.name || row.native_locator}: ${intervalLabel(histogram,index)} · ${countLabel(count)} values`)}</title></rect>`;
      }
    } else body += `<text x="${x}" y="${y + 13}" fill="${palette.muted}">${escapeXml(row.reason?.replaceAll("_", " ") || (row.statistics ? "Histogram not assessable" : "Not analyzed"))}</text>`;
    body += "</g>";
  });
  body += axisTicks(window, x, width, H - 9, palette);
  return shell(760, H, "Tensor distribution map", palette, body, "Rows share bin edges; each row is normalized to its own peak count. Stored integer codes and real values are different representations. Not-assessed tensors have no painted distribution.");
}
export function connectionsSvg(row, links, palette) {
  if (!links.length) return null;
  const shown = links.slice(0, 12), height = Math.max(112, shown.length * 66 + 24), middle = height / 2;
  let body = `<rect x="14" y="${middle - 32}" width="260" height="64" rx="4" stroke="${palette.accent}" fill="${palette.selected}"/><text x="28" y="${middle - 5}">${escapeXml((row.name || row.native_locator || row.id).slice(0,29))}</text><text x="28" y="${middle + 16}" fill="${palette.muted}">${escapeXml(row.dtype)} · ${escapeXml((row.shape || []).join(" × "))}</text>`;
  shown.forEach(({operation:op, ports}, i) => {
    const y = 12 + i * 66, target = y + 26;
    body += `<path d="M274 ${middle}C350 ${middle} 345 ${target} 414 ${target}" fill="none" stroke="${palette.muted}"/><rect x="414" y="${y}" width="328" height="52" rx="4" fill="${palette.surface}" stroke="${palette.line}"/><text x="428" y="${y+21}">${escapeXml(operationLabel(op).slice(0,35))}</text><text x="428" y="${y+39}" fill="${palette.muted}">${escapeXml(`${op.native_op?.name || op.kind || "operation"} · ${ports.join(", ")}`.slice(0,43))}</text>`;
  });
  return shell(760, height, "Serialized tensor and operation connections", palette, body, "Edges are references in Model IR. They do not measure correlation, causal importance, training layers or runtime scheduling. The complete connection list follows this diagram.");
}

export function tensorMetricMapSvg(rows, selected, palette, mode) {
  if (!rows.length) return null;
  const value = row => !row.statistics ? null : mode === 'rms' ? row.statistics.rms : row.statistics.unsafe_integer_count === '0' && BigInt(row.statistics.finite_count) > 0n ? Number(row.statistics.zero_count) / Number(row.statistics.finite_count) : null;
  const maximum = mode === 'rms' ? Math.max(1e-300, ...rows.map(row => value(row) ?? 0)) : 1;
  const height = 42 + rows.length * 29;
  let body = `<text x="218" y="20" fill="${palette.muted}">${mode === 'rms' ? 'RMS · linear scale shared across visible tensors' : 'Exact zero fraction · 0–100%'} · select a tensor</text>`;
  rows.forEach((row, i) => {
    const v = value(row), y = 34 + i * 29;
    body += `<g data-tensor-id="${escapeXml(row.id)}" role="button" tabindex="0" aria-label="Select ${escapeXml(row.name || row.id)}"><rect x="0" y="${y - 3}" width="760" height="28" fill="${selected === row.id ? palette.selected : palette.surface}"/><text x="12" y="${y + 13}">${escapeXml((row.name || row.id).slice(0, 25))}</text>`;
    if (v !== null) body += `<rect x="218" y="${y}" width="${Math.max(0, v / maximum * 386)}" height="18" fill="${palette.accent}"/><text x="620" y="${y + 13}">${escapeXml(mode === 'rms' ? numberLabel(v) : numberLabel(v * 100) + '%')}</text><title>${escapeXml(row.name || row.id)}: ${v}</title>`;
    else body += `<text x="218" y="${y + 13}" fill="${palette.muted}">Not assessed</text>`;
    body += '</g>';
  });
  return shell(760, height, 'Across-tensor metric comparison', palette, body, 'Stored-payload statistics. Compare compatible representations. Unassessed values are not plotted as zero.');
}
