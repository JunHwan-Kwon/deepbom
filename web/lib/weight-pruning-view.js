import { escapeXml as esc, numberLabel as num, countLabel } from "./weight-visuals.js";

const percent = x => x === null ? "Not assessed" : num(x * 100) + "%";
const caption = mode => ({ candidate: "Candidate", difference: "Candidate − original", removed: "Newly removed positions" })[mode];
function heatmap(result, mode, p) {
  const projections = [result.projections.original, result.projections[mode]], height = 360;
  let content = '';
  projections.forEach((projection, side) => {
    const left = 24 + side * 378, top = 60, w = 330 / projection.columns, h = 254 / projection.rows;
    content += `<text x="${left}" y="28">${side ? caption(mode) : "Original"}</text>`;
    content += `<text x="${left}" y="46" font-size="10" fill="${p.muted}">${projection.source_rows} × ${projection.source_columns} positions · ${projection.rows} × ${projection.columns} cells</text>`;
    projection.cells.forEach((cell, i) => {
      const value = side && mode === 'removed' ? cell.mean : result.color_scale ? cell.mean / result.color_scale : 0;
      content += `<rect data-cell="${side}:${i}" tabindex="-1" x="${left + i % projection.columns * w}" y="${top + Math.floor(i / projection.columns) * h}" width="${Math.max(.1, w - .4)}" height="${Math.max(.1, h - .4)}" fill="${side && mode === 'removed' ? p.blue : value < 0 ? p.blue : p.accent}" fill-opacity="${.06 + .94 * Math.min(1, Math.abs(value))}"><title>${esc(`Mean ${cell.mean}; min ${cell.minimum}; max ${cell.maximum}; ${cell.count} positions; ${cell.zero_count} zeros`)}</title></rect>`;
    });
  });
  content += `<text x="24" y="340" font-size="10" fill="${p.muted}">${esc(mode === 'removed' ? 'Right: brighter blue = larger fraction newly removed. Original zeros are not new removals.' : 'Shared fixed color range across target changes; blue negative, green positive; cells show means.')}</text>`;
  content += `<text x="24" y="354" font-size="10" fill="${p.muted}">${esc(`Target ${result.target_percent}% zeros · view: ${result.selected_channel === null ? 'whole tensor' : 'channel ' + result.selected_channel} · axis ${result.axes.channel_axis} · weight color scale ±${num(result.color_scale)}`)}</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 ${height}" role="img" aria-label="Original and pruning preview heatmaps"><title>Pruning preview · ${result.target_percent}% target zeros</title><rect width="760" height="${height}" fill="${p.surface}"/><g font-family="ui-monospace,monospace" font-size="13" fill="${p.ink}">${content}</g></svg>`;
}
function curveSvg(result, p) {
  const defined = result.curve.filter(point => point.retained_energy_fraction !== null);
  let body = '';
  for (const fraction of [0, .25, .5, .75, 1]) body += `<path d="M55 ${170 - fraction * 125}H724" stroke="${p.line}"/><text x="45" y="${174 - fraction * 125}" text-anchor="end">${fraction * 100}%</text>`;
  body += `<path d="${defined.map((point, i) => `${i ? 'L' : 'M'}${55 + point.target_percent * 6.69} ${170 - point.retained_energy_fraction * 125}`).join(' ')}" fill="none" stroke="${p.accent}" stroke-width="2"/>`;
  body += `<path data-curve-marker="" d="M${55 + result.target_percent * 6.69} 36V170" stroke="${p.blue}" stroke-dasharray="4 4"/>`;
  for (let target = 0; target <= 100; target += 25) body += `<text x="${55 + target * 6.69}" y="190" text-anchor="middle">${target}%</text>`;
  body += `<text x="55" y="23">Retained weight energy · not task accuracy</text><text x="55" y="214" fill="${p.muted}">${defined.length ? 'Target total zeros · click the curve to set the target' : 'All-zero tensor: energy fraction is undefined'}</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 230" role="img" aria-label="Target zero fraction versus retained weight energy"><rect width="760" height="230" fill="${p.surface}"/><g font-family="ui-monospace,monospace" font-size="11" fill="${p.ink}">${body}</g></svg>`;
}

// Components retain their controls and SVG cells across updates. Only changed
// attributes move; keyboard focus and the user's zoom are not discarded.
function patchSvg(container, markup) {
  const next = new DOMParser().parseFromString(markup, 'image/svg+xml').documentElement;
  if (next.localName !== 'svg' || next.querySelector('parsererror')) throw new Error('Cannot render an invalid SVG projection');
  const before = container.querySelector('svg');
  if (!before || before.querySelectorAll('[data-cell]').length !== next.querySelectorAll('[data-cell]').length) { container.replaceChildren(document.importNode(next, true)); return; }
  const a = before.querySelectorAll('*'), b = next.querySelectorAll('*');
  if (a.length !== b.length) { container.replaceChildren(document.importNode(next, true)); return; }
  a.forEach((node, i) => {
    for (const attr of b[i].attributes) if (node.getAttribute(attr.name) !== attr.value) node.setAttribute(attr.name, attr.value);
    if (!node.children.length && node.textContent !== b[i].textContent) node.textContent = b[i].textContent;
  });
}

export function installPruningView(host, { context, weight, weightRef, options, colors, state, onResult, onExport, onError, isCurrent }) {
  let worker = null, destroyed = false, ready = false, latest = 0, applied = -1, inFlight = null, timer = null, result = null, mode = state.mode || 'candidate', channelPage = 0;
  let zoom = 1, panX = 0, panY = 0, drag = null, moved = false;
  const target = state.target ?? 50;
  host.innerHTML = `<div class="weight-lab">
    <header><div><span class="weight-eyebrow">Interactive experiment</span><h3>Pruning workbench</h3></div><div class="weight-lab-header-actions"><span class="weight-badge">Numerical simulation</span><button type="button" data-prune-expand>Expand workspace</button><button type="button" data-prune-export="svg" disabled>SVG</button><button type="button" data-prune-export="png" disabled>PNG</button><button type="button" data-prune-export="json" disabled>Evidence JSON</button></div></header>
    <p class="weight-lab-intro">Explore this tensor, then select a channel to inspect it. One magnitude mask applies to the complete tensor; changing the view does not change the experiment.</p>
    <p class="weight-lab-hint" data-lab-basis></p>
    <div class="weight-lab-control"><label for="pruning-target">Target total zeros <output data-target-label>${target}%</output></label><div><input id="pruning-target" data-prune-target type="range" min="0" max="100" step="1" value="${target}" disabled><input data-prune-number aria-label="Target total zeros percentage" type="number" min="0" max="100" step="1" value="${target}" disabled></div><p>Includes existing zeros. Additional values are removed in increasing absolute magnitude.</p></div>
    <p data-lab-status role="status" aria-live="polite">Preparing complete tensor values in a local worker…</p>
    <div data-lab-results hidden>
      <div class="weight-lab-summary"><dl class="weight-lab-metrics"></dl><div data-prune-curve class="weight-prune-curve"></div></div>
      <div class="weight-lab-channel-head"><strong>Channel / axis-slice response</strong><div><button type="button" data-channel-page="prev" aria-label="Previous channels">←</button><span data-channel-page-label></span><button type="button" data-channel-page="next" aria-label="Next channels">→</button></div></div>
      <p class="weight-lab-hint">Each bar shows retained squared weight norm within one channel. Select a channel for its slice; a low norm is not proof of low importance.</p>
      <div data-prune-channels class="weight-prune-channels" role="group" aria-label="Inspect channel response"></div>
      <div class="weight-lab-navigation"><button type="button" data-all-channels>Whole tensor</button><label>Channel <input data-prune-channel type="number" min="0" aria-label="Inspect channel index" placeholder="All"></label><span data-channel-scope></span></div>
      <div class="weight-lab-viewbar"><div role="group" aria-label="Pruning projection">${['candidate', 'removed', 'difference'].map(id => `<button type="button" data-prune-mode="${id}" aria-pressed="${id === mode}">${caption(id)}</button>`).join('')}</div><div role="group" aria-label="Heatmap zoom"><button type="button" data-prune-zoom="out" aria-label="Zoom out">−</button><button type="button" data-prune-zoom="reset">Reset view</button><button type="button" data-prune-zoom="in" aria-label="Zoom in">+</button></div></div>
      <div data-prune-map class="weight-prune-map" tabindex="0" role="group" aria-label="Heatmaps. Use zoom buttons and arrow keys to pan; pointer drag also pans."></div>
      <div class="weight-cell-inspector"><label>Inspect cell <input data-prune-cell type="number" min="0" value="0" aria-label="Inspect heatmap cell index"></label><p data-prune-readout aria-live="polite"></p></div>
      <p class="weight-lab-boundary">Complete values within the reported budget. Aggregated cells expose exact counts and extrema. This preview changes no model file, and measures neither accuracy nor runtime speed.</p>
    </div></div>`;
  const $ = selector => host.querySelector(selector);
  const status = $('[data-lab-status]');
  const expand = $('[data-prune-expand]');
  expand.hidden = !document.fullscreenEnabled || !host.requestFullscreen;
  const fullscreenChanged = () => { if (!destroyed) expand.textContent = document.fullscreenElement === host ? 'Exit expanded view' : 'Expand workspace'; };
  document.addEventListener('fullscreenchange', fullscreenChanged);
  expand.onclick = async () => { try { if (document.fullscreenElement === host) await document.exitFullscreen(); else await host.requestFullscreen(); } catch { setStatus('This browser does not allow expanded view. All controls remain available here.'); } };
  const valid = () => !destroyed && isCurrent();
  const setStatus = text => { status.textContent = text; };
  function request(immediate = false) {
    if (!valid()) return;
    latest++; clearTimeout(timer); onResult(null, null);
    for (const button of host.querySelectorAll('[data-prune-export]')) button.disabled = true;
    $('[data-lab-results]').dataset.pending = 'true';
    setStatus(ready ? 'Updating the complete-tensor mask…' : 'Preparing complete tensor values in a local worker…');
    if (immediate) send(); else timer = setTimeout(send, 90);
  }
  function send() {
    if (!ready || inFlight !== null || !valid()) return;
    inFlight = latest;
    worker.postMessage({ type: 'preview', request: latest, settings: { target_percent: state.target ?? target, channel: state.channel ?? null } });
  }
  function setTarget(value) {
    if (!Number.isInteger(value) || value < 0 || value > 100) { setStatus('Choose a whole percentage from 0 to 100.'); $('[data-prune-number]').value = state.target ?? target; return; }
    state.target = value; $('[data-prune-target]').value = value; $('[data-prune-number]').value = value; $('[data-target-label]').textContent = `${value}%`; request();
  }
  function selectChannel(value) {
    if (!result || value !== null && (!Number.isInteger(value) || value < 0 || value >= result.channels.length)) { setStatus('Choose a channel index within this tensor.'); return; }
    state.channel = value; $('[data-prune-channel]').value = value ?? ''; zoom = 1; panX = panY = 0;
    if (value !== null) channelPage = Math.floor(value / 32);
    request(true);
  }
  function renderChannels() {
    const rows = result.channels.slice(channelPage * 32, (channelPage + 1) * 32), box = $('[data-prune-channels]');
    const html = rows.map(row => `<button type="button" data-lab-channel="${row.index}" aria-pressed="${row.index === (state.channel ?? null)}" aria-label="Inspect channel ${row.index}, ${percent(row.retained_energy_fraction)} energy retained"><span>#${row.index}</span><i><b style="width:${(row.retained_energy_fraction ?? 0) * 100}%"></b></i><span>${percent(row.retained_energy_fraction)}</span></button>`).join('');
    const ids = rows.map(row => row.index).join(',');
    if (box.dataset.ids !== ids) { box.innerHTML = html; box.dataset.ids = ids; }
    else rows.forEach((row, i) => { const button = box.children[i]; button.setAttribute('aria-pressed', String(row.index === (state.channel ?? null))); button.setAttribute('aria-label', `Inspect channel ${row.index}, ${percent(row.retained_energy_fraction)} energy retained`); button.querySelector('b').style.width = `${(row.retained_energy_fraction ?? 0) * 100}%`; button.lastElementChild.textContent = percent(row.retained_energy_fraction); });
    $('[data-channel-page-label]').textContent = `${channelPage * 32 + 1}–${channelPage * 32 + rows.length} / ${result.channels.length}`;
    $('[data-channel-page="prev"]').disabled = channelPage === 0; $('[data-channel-page="next"]').disabled = (channelPage + 1) * 32 >= result.channels.length;
  }
  function readCell(index = Number($('[data-prune-cell]').value) || 0) {
    if (!result) return;
    const left = result.projections.original, right = result.projections[mode];
    index = Math.max(0, Math.min(left.cells.length - 1, Number.isFinite(index) ? Math.floor(index) : 0)); $('[data-prune-cell]').value = index;
    const a = left.cells[index], b = right.cells[index];
    $('[data-prune-readout]').textContent = `Cell ${index} · ${a.count} positions. Original mean ${a.mean}; range [${a.minimum}, ${a.maximum}]. ${caption(mode)} mean ${b.mean}; range [${b.minimum}, ${b.maximum}]. ${mode === 'removed' ? `${b.count - b.zero_count} newly removed positions.` : `${b.zero_count} zeros in the right-hand cell.`}`;
    for (const cell of host.querySelectorAll('[data-cell]')) cell.classList.toggle('weight-cell-selected', Number(cell.dataset.cell.split(':')[1]) === index);
  }
  function applyZoom() {
    const svg = $('[data-prune-map] svg'); if (!svg) return;
    panX = Math.max(0, Math.min(760 - 760 / zoom, panX)); panY = Math.max(0, Math.min(360 - 360 / zoom, panY));
    svg.setAttribute('viewBox', `${panX} ${panY} ${760 / zoom} ${360 / zoom}`);
    $('[data-prune-map]').style.touchAction = zoom > 1 ? 'none' : 'pan-y';
  }
  function renderMap() {
    const markup = heatmap(result, mode, colors()); patchSvg($('[data-prune-map]'), markup); applyZoom();
    for (const button of host.querySelectorAll('[data-prune-mode]')) button.setAttribute('aria-pressed', String(button.dataset.pruneMode === mode));
    $('[data-prune-cell]').max = result.projections.original.cells.length - 1; readCell();
    if (applied === latest && valid()) { onResult(result, markup); for (const button of host.querySelectorAll('[data-prune-export]')) button.disabled = false; }
  }
  function render() {
    const p = colors(); $('[data-lab-results]').hidden = false; $('[data-lab-results]').dataset.pending = String(applied !== latest);
    $('[data-lab-basis]').textContent = `Values: ${result.representation === 'dequantized_real' ? 'restored real values from serialized quantization' : 'stored floating values'} · axis ${result.axes.channel_axis} (${String(result.axes.channel_meaning || 'declared axis slices').replaceAll('_', ' ')}).`;
    $('[data-prune-channel]').max = result.channels.length - 1;
    $('[data-prune-channel]').value = state.channel ?? '';
    $('[data-channel-scope]').textContent = state.channel == null ? `All ${result.channels.length} slices · axis ${result.axes.channel_axis}` : `Channel ${state.channel} of ${result.channels.length} · axis ${result.axes.channel_axis}`;
    $('[data-all-channels]').setAttribute('aria-pressed', String(state.channel == null));
    const rows = [['Existing zeros', countLabel(result.original_zero_count)], ['Newly zeroed', countLabel(result.newly_zeroed_count)], ['Resulting zeros', percent(result.achieved_zero_fraction)], ['Energy retained', percent(result.retained_energy_fraction)], ['Relative L2 change', num(result.metrics.relative_l2)], ['RMSE', num(result.metrics.rmse)]];
    $('[data-lab-results] dl').innerHTML = rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${esc(value)}</dd></div>`).join('');
    patchSvg($('[data-prune-curve]'), curveSvg(result, p)); renderChannels(); renderMap();
    setStatus(`${countLabel(result.value_count)} / ${countLabel(result.value_count)} values evaluated · target ${result.target_percent}% · actual zeros ${percent(result.achieved_zero_fraction)}. ${result.target_zero_count < result.original_zero_count ? 'Existing zeros already exceed the requested target.' : 'Original values remain unchanged.'}`);
  }
  $('[data-prune-target]').oninput = e => setTarget(Number(e.target.value));
  $('[data-prune-number]').onchange = e => setTarget(Number(e.target.value));
  $('[data-prune-channel]').onchange = e => selectChannel(e.target.value === '' ? null : Number(e.target.value));
  $('[data-prune-cell]').onchange = () => readCell();
  host.addEventListener('click', event => {
    const download = event.target.closest('[data-prune-export]'); if (download && applied === latest && valid()) onExport?.(download.dataset.pruneExport);
    const channelButton = event.target.closest('[data-lab-channel]'); if (channelButton) selectChannel(Number(channelButton.dataset.labChannel));
    if (event.target.closest('[data-all-channels]')) selectChannel(null);
    const page = event.target.closest('[data-channel-page]'); if (page && result) { channelPage += page.dataset.channelPage === 'next' ? 1 : -1; renderChannels(); }
    const view = event.target.closest('[data-prune-mode]'); if (view && result) { mode = state.mode = view.dataset.pruneMode; renderMap(); }
    const zoomButton = event.target.closest('[data-prune-zoom]');
    if (zoomButton) { const previous = zoom; zoom = zoomButton.dataset.pruneZoom === 'reset' ? 1 : Math.max(1, Math.min(8, zoom * (zoomButton.dataset.pruneZoom === 'in' ? 1.5 : 1 / 1.5))); panX += 380 / previous - 380 / zoom; panY += 180 / previous - 180 / zoom; applyZoom(); }
    const cell = event.target.closest('[data-cell]'); if (cell && !moved) readCell(Number(cell.dataset.cell.split(':')[1]));
    const curve = event.target.closest('[data-prune-curve] svg');
    if (curve && result) { const box = curve.getBoundingClientRect(); setTarget(Math.max(0, Math.min(100, Math.round(((event.clientX - box.left) * 760 / box.width - 55) / 6.69)))); }
  });
  const map = $('[data-prune-map]');
  map.onpointerdown = event => { moved = false; if (zoom === 1) return; drag = { x: event.clientX, y: event.clientY, panX, panY }; map.setPointerCapture(event.pointerId); };
  map.onpointermove = event => { if (!drag) return; const box = map.getBoundingClientRect(); const dx = event.clientX - drag.x, dy = event.clientY - drag.y; moved ||= Math.abs(dx) + Math.abs(dy) > 4; panX = drag.panX - dx * 760 / box.width / zoom; panY = drag.panY - dy * 360 / box.height / zoom; applyZoom(); };
  map.onpointerup = map.onpointercancel = () => { drag = null; };
  map.onkeydown = event => { const changes = { ArrowLeft: [-30, 0], ArrowRight: [30, 0], ArrowUp: [0, -30], ArrowDown: [0, 30] }; if (changes[event.key]) { event.preventDefault(); panX += changes[event.key][0] / zoom; panY += changes[event.key][1] / zoom; applyZoom(); } };
  try {
    worker = new Worker(new URL('../workers/weight-pruning-worker.js', import.meta.url), { type: 'module' });
    const fail = message => { if (!valid()) return; ready = false; worker?.terminate(); $('[data-lab-results]').hidden = true; for (const input of host.querySelectorAll('[data-prune-target], [data-prune-number]')) input.disabled = true; onResult(null, null); setStatus(`Preview unavailable: ${message.replaceAll('_', ' ')}.`); onError?.(message); };
    worker.onerror = event => fail(event.message || 'Pruning worker failed');
    worker.onmessage = ({ data }) => {
      if (!valid()) return;
      if (data.type === 'error') { fail(data.error); return; }
      if (data.type === 'ready') { ready = true; for (const input of host.querySelectorAll('[data-prune-target], [data-prune-number]')) input.disabled = false; request(true); return; }
      if (data.type === 'preview') { inFlight = null; if (data.request === latest) { try { applied = data.request; result = data.result; render(); } catch (error) { fail(error.message); } } else send(); }
    };
    worker.postMessage({ type: 'initialize', ...context, weight, weightRef, options });
  } catch (error) { setStatus(`Preview unavailable: ${error.message}`); }
  return { destroy() { destroyed = true; clearTimeout(timer); worker?.terminate(); document.removeEventListener('fullscreenchange', fullscreenChanged); if (document.fullscreenElement === host) document.exitFullscreen().catch(() => {}); }, repaint() { if (result) { patchSvg($('[data-prune-curve]'), curveSvg(result, colors())); renderMap(); } } };
}
