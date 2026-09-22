import { buildActivationIr, buildNumericalEvidenceBundle } from "./activation-ir.js";
import { parseStrictJson } from "./strict-json.js";
import { escapeXml as esc, countLabel, numberLabel, zeroLabel, createConnectionIndex, linkedOperations, operationLabel, histogramSvg, distributionMapSvg, connectionsSvg, intervalLabel } from "./weight-visuals.js";

const PAGE_SIZE = 40;
const phrase = value => String(value || "").replaceAll("_", " ");
const shapeLabel = row => Array.isArray(row.shape) ? `[${row.shape.join(" × ")}]` : "Shape not assessed";
const rowName = row => row.name || row.native_locator || row.id;
const rowCount = row => row.statistics?.value_count ?? row.element_count?.decimal ?? null;
const metric = (label, value, note = "") => `<div><dt>${label}</dt><dd>${esc(value)}</dd>${note ? `<small>${esc(note)}</small>` : ""}</div>`;

export function installNumericalPanel(host, getContext) {
  if (!host || host.dataset.numericalPanel != null) return null;
  host.dataset.numericalPanel = "";
  host.classList.add("weight-workspace");
  host.innerHTML = `
    <header class="weight-head">
      <div><span class="weight-eyebrow">Numerical inspection</span><h2>Weight explorer</h2><p>Trace stored tensors through the model. Inspect their distributions and numerical structure.</p></div>
      <span class="weight-local">Local analysis · no model execution</span>
    </header>
    <div class="weight-toolbar">
      <div class="weight-run-actions"><button type="button" data-action="weights" class="weight-primary">Analyze weights</button><button type="button" data-action="cancel" hidden>Cancel</button><button type="button" data-action="import">Import activation capture</button><input type="file" accept="application/json,.json" hidden aria-label="Import activation capture"></div>
      <div class="weight-export-actions"><button type="button" data-action="svg" disabled>Save SVG</button><button type="button" data-action="png" disabled>Save PNG</button><button type="button" data-action="save" disabled>Save evidence JSON</button></div>
    </div>
    <div class="weight-status-line"><p role="status" aria-live="polite">Select and audit a model to begin.</p><progress hidden aria-label="Weight analysis progress"></progress></div>
    <dl class="weight-overview" data-overview></dl>
    <div class="weight-view-controls"><div class="weight-segmented" role="group" aria-label="Numeric evidence source"><button type="button" data-source="weight" aria-pressed="true">Stored weights</button><button type="button" data-source="activation" aria-pressed="false">Imported activations <span data-activation-count>0</span></button></div><span data-subject class="weight-subject"></span></div>
    <div class="weight-layout">
      <aside class="weight-inventory" aria-label="Tensor inventory">
        <header><strong>Tensor inventory</strong><span data-match-count></span></header>
        <label class="weight-search"><span>Find tensor or operation</span><input type="search" placeholder="Name, dtype, or connected op" aria-label="Find tensor or operation"></label>
        <div class="weight-filters"><label>Encoding<select data-filter="dtype"><option value="">All encodings</option></select></label><label>Status<select data-filter="status"><option value="">All tensors</option><option value="assessed">Assessed</option><option value="not_assessed">Not assessed</option><option value="not_analyzed">Not analyzed</option></select></label></div>
        <label class="weight-sort">Order<select data-sort><option value="source">Source inventory</option><option value="count">Value count ↓</option><option value="rms">RMS ↓</option><option value="zero">Zero count ↓</option><option value="name">Name A–Z</option></select></label>
        <div class="weight-tensor-list" data-inventory role="group" aria-label="Select tensor"></div>
        <div class="weight-pagination"><button type="button" data-page="prev" aria-label="Previous tensors">←</button><span data-page-label></span><button type="button" data-page="next" aria-label="Next tensors">→</button></div>
      </aside>
      <div class="weight-canvas">
        <section class="weight-map-section" aria-labelledby="weightMapTitle"><header class="weight-section-head"><div><span class="weight-eyebrow">01 / Across tensors</span><h3 id="weightMapTitle">Distribution map</h3></div><span class="weight-legend"><i class="weight-negative"></i>Negative <i class="weight-positive"></i>Zero / positive</span></header><div data-map class="weight-map"></div><p class="weight-chart-note">Shared value bins; intensity is relative to each tensor’s peak. Select a row to inspect it. Filter by encoding when comparing stored integer codes and real values.</p></section>
        <section class="weight-detail" data-detail aria-label="Selected tensor"></section>
      </div>
    </div>
    <details class="weight-method"><summary>Evidence, coverage and method</summary><div data-method></div><button type="button" data-action="model">Save Model IR</button></details>`;
  const $ = selector => host.querySelector(selector);
  const button = action => $(`[data-action="${action}"]`);
  const status = $('[role="status"]'), fileInput = $('[type="file"]'), search = $('[type="search"]');
  let model = null, weight = null, activation = null, worker = null, generation = 0, source = "weight", selected = null, page = 0;
  let logCount = false, allRows = [], filtered = [], visible = [], index = new Map(), lastSvg = null;

  function palette() {
    const style = getComputedStyle(host);
    return Object.fromEntries(Object.entries({surface:"--surface", ink:"--ink", muted:"--muted", line:"--line", accent:"--accent", blue:"--blue", selected:"--surface-tint"}).map(([name,variable]) => [name,style.getPropertyValue(variable).trim()]));
  }
  function setStatus(text, error = false) { status.textContent = text; status.dataset.error = String(error); }
  function stopWorker() { generation++; worker?.terminate(); worker = null; button("cancel").hidden = true; button("weights").disabled = !model; $("progress").hidden = true; }
  function sync() {
    const next = getContext()?.model || null;
    if (next?.model_ir_sha256 === model?.model_ir_sha256) return false;
    stopWorker(); model = next; weight = null; activation = null; selected = null; page = 0; source = "weight";
    search.value = ""; $('[data-filter="status"]').value = ""; $('[data-sort]').value = "source";
    button("weights").disabled = !model; button("model").disabled = !model; button("import").disabled = !model;
    setStatus(model ? "Structure is ready. Analyze weights to inspect stored numeric values; the model will not run." : "Select and audit a model to begin.");
    rebuild(); return true;
  }
  function requireContext() { sync(); const context = getContext(); if (!model || !context?.model) throw new Error("Analyze a model first."); return context; }
  function rebuild() {
    const connectionIndex = createConnectionIndex(model);
    allRows = source === "activation" ? (activation?.tensors || []).map(row => ({...row,status:"assessed"}))
      : weight?.tensors || (model?.tensors_and_storage.storage_objects || []).map(row => ({...row,storage_ref:row.id,native_locator:row.native_source?.path || row.id,status:"not_analyzed",statistics:null,binding_refs:connectionIndex.bindingsByStorage.get(row.id) || []}));
    index = new Map(allRows.map(row => [row.id,{row,links:linkedOperations(row,model,connectionIndex)}]));
    for (const entry of index.values()) entry.query = [rowName(entry.row),entry.row.id,entry.row.storage_ref,entry.row.dtype,...entry.links.map(link => operationLabel(link.operation))].join(" ").toLowerCase();
    const dtype = $('[data-filter="dtype"]'), before = dtype.value;
    dtype.innerHTML = '<option value="">All encodings</option>' + [...new Set(allRows.map(row => row.dtype))].sort().map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join("");
    if ([...dtype.options].some(option => option.value === before)) dtype.value = before;
    for (const tab of host.querySelectorAll('[data-source]')) tab.setAttribute("aria-pressed",String(tab.dataset.source === source));
    $('[data-activation-count]').textContent = activation?.coverage.captured_count || "0";
    $('[data-subject]').textContent = model?.artifact?.filename || "";
    renderOverview(); renderInventory(); renderMethod();
    button("save").disabled = !weight && !activation;
  }
  function renderOverview() {
    const doc = source === "weight" ? weight : activation, stats = (doc?.tensors || []).filter(row => row.statistics).map(row => row.statistics);
    const sum = get => stats.reduce((total,row) => total + BigInt(get(row)),0n);
    const total = sum(row => row.value_count), zero = sum(row => row.zero_count), unsafe = sum(row => row.unsafe_integer_count);
    const nonfinite = sum(row => Object.values(row.nonfinite).reduce((n,c) => n + BigInt(c),0n));
    const assessed = source === "weight" ? weight?.coverage.assessed_count : activation?.coverage.captured_count;
    const missing = source === "weight" ? weight?.coverage.not_assessed_count : activation?.coverage.missing_count;
    $('[data-overview]').innerHTML = metric(source === "weight" ? "Stored payloads" : "Captured tensors",String(allRows.length),source === "weight" ? "Model IR inventory" : "Imported execution")
      + metric("Assessed values",doc ? countLabel(total) : "Not analyzed",doc ? "Exact count · no sampling" : "Optional payload inspection")
      + metric("Exact zeros",stats.length && !unsafe ? countLabel(zero) : "Not assessed",unsafe ? "Unsafe integer arithmetic" : "Within assessed tensors")
      + metric("Non-finite values",stats.length ? countLabel(nonfinite) : "Not assessed","Within assessed tensors")
      + metric("Coverage",doc ? `${assessed} / ${source === "weight" ? weight.coverage.inventory_count : activation.coverage.requested_count}` : "Pending",doc ? `${missing} ${source === "weight" ? "not assessed" : "missing"}` : "Explicit opt-in");
  }
  function renderInventory() {
    const query = search.value.trim().toLowerCase(), dtype = $('[data-filter="dtype"]').value, state = $('[data-filter="status"]').value, sort = $('[data-sort]').value;
    filtered = allRows.filter(row => (!query || index.get(row.id).query.includes(query)) && (!dtype || row.dtype === dtype) && (!state || row.status === state));
    const sortValue = row => sort === "count" ? rowCount(row) : sort === "zero" ? row.statistics?.zero_count : row.statistics?.rms;
    if (sort === "name") filtered.sort((a,b) => rowName(a).localeCompare(rowName(b)));
    else if (sort !== "source") filtered.sort((a,b) => {const av = sortValue(a),bv = sortValue(b); if (av == null) return bv == null ? 0 : 1; if (bv == null) return -1; const left = sort === "rms" ? av : BigInt(av),right = sort === "rms" ? bv : BigInt(bv); return left > right ? -1 : left < right ? 1 : 0;});
    page = Math.min(page,Math.max(0,Math.ceil(filtered.length / PAGE_SIZE) - 1));
    visible = filtered.slice(page * PAGE_SIZE,(page + 1) * PAGE_SIZE);
    if (!visible.some(row => row.id === selected)) selected = visible[0]?.id || null;
    $('[data-match-count]').textContent = `${filtered.length} / ${allRows.length}`;
    $('[data-page-label]').textContent = filtered.length ? `${page * PAGE_SIZE + 1}–${page * PAGE_SIZE + visible.length} of ${filtered.length}` : "0 tensors";
    $('[data-page="prev"]').disabled = page === 0; $('[data-page="next"]').disabled = (page + 1) * PAGE_SIZE >= filtered.length;
    $('[data-inventory]').innerHTML = visible.length ? visible.map(row => `<button type="button" class="weight-tensor" data-tensor-id="${esc(row.id)}" aria-pressed="${selected === row.id}"><span class="weight-tensor-title">${esc(rowName(row))}<i class="weight-state ${esc(row.status)}" title="${esc(phrase(row.status))}"></i></span><span>${esc(row.dtype)} <b>${esc(shapeLabel(row))}</b></span><small>${rowCount(row) == null ? esc(phrase(row.status)) : `${countLabel(rowCount(row))} values`} · ${index.get(row.id).links.length} ops</small></button>`).join("") : `<p class="weight-list-empty">${allRows.length ? "No tensors match these filters." : source === "activation" ? "Import an activation capture to inspect observed values." : "No stored tensor inventory is available."}</p>`;
    renderCharts();
  }
  function empty(title, description, run = false) { return `<div class="weight-empty"><span class="weight-empty-glyph" aria-hidden="true">▥</span><h4>${esc(title)}</h4><p>${esc(description)}</p>${run ? '<button type="button" data-action="weights-inline">Analyze weights</button>' : ""}</div>`; }
  function renderCharts() {
    const colors = palette(), row = index.get(selected)?.row, map = distributionMapSvg(visible,selected,colors);
    $('[data-map]').innerHTML = map || empty(source === "activation" ? "Execution evidence belongs here" : weight ? "No plottable distributions" : "See the numerical structure",source === "activation" ? "Import a hash-bound capture from an explicit runtime execution. No model is run by this workspace." : weight ? "The current tensors have no assessed finite histograms. Their status and reason remain in the inventory and exported evidence." : "Read stored values to reveal a distribution for each tensor. Structure and connections are already available below.",source === "weight" && !weight && Boolean(model));
    const detail = $('[data-detail]');
    if (!row) {detail.innerHTML = empty("No tensor selected","Select a tensor from the inventory or distribution map."); lastSvg = null; button("svg").disabled = button("png").disabled = true; return;}
    const links = index.get(row.id).links, s = row.statistics;
    lastSvg = histogramSvg(row,colors,{logCount});
    const connections = connectionsSvg(row,links,colors);
    detail.innerHTML = `<header class="weight-section-head"><div><span class="weight-eyebrow">02 / Selected tensor</span><h3>${esc(rowName(row))}</h3><p class="weight-tensor-contract">${esc(row.dtype)} <span>·</span> ${esc(shapeLabel(row))} <span>·</span> ${esc(phrase(row.representation || (source === "activation" ? "runtime capture" : row.status)))}</p></div><span class="weight-badge ${esc(row.status)}">${esc(phrase(row.status))}</span></header>
      <dl class="weight-stat-grid">${metric("Mean",numberLabel(s?.mean))}${metric("Standard deviation",numberLabel(s?.population_stddev),"Population")}${metric("RMS",numberLabel(s?.rms))}${metric("Zero fraction",zeroLabel(s),"Among finite values")}${metric("Minimum",s?.integer_minimum != null && s?.precision === "not_assessed_unsafe_integer_arithmetic" ? s.integer_minimum : numberLabel(s?.minimum))}${metric("Maximum",s?.integer_maximum != null && s?.precision === "not_assessed_unsafe_integer_arithmetic" ? s.integer_maximum : numberLabel(s?.maximum))}</dl>
      <div class="weight-plot-head"><strong>Value distribution</strong><label><input type="checkbox" data-log-count ${logCount ? "checked" : ""}> Log count axis</label></div>
      <div class="weight-histogram" data-histogram>${lastSvg || empty(row.status === "not_analyzed" ? "Numeric values have not been read" : "Histogram unavailable",row.reason ? phrase(row.reason) : s?.unsafe_integer_count !== "0" && s ? "Exact integer extrema are preserved. Floating statistics and histogram are not assessed for unsafe integer arithmetic." : s ? "No finite values are available for a histogram." : "Run optional weight analysis to populate this view.")}</div>
      <p class="weight-bin-readout" data-bin-readout>Hover or select a bin for its exact interval and count. Unequal numeric intervals are drawn with equal screen width; this is a count histogram, not density.</p>
      ${s ? `<div class="weight-integrity"><span>NaN <b>${countLabel(s.nonfinite.nan)}</b></span><span>+∞ <b>${countLabel(s.nonfinite.positive_infinity)}</b></span><span>−∞ <b>${countLabel(s.nonfinite.negative_infinity)}</b></span><span>Negative zero <b>${countLabel(s.negative_zero_count)}</b></span><span>Values <b>${countLabel(s.value_count)}</b></span></div>` : ""}
      <div class="weight-connections"><header class="weight-section-head"><div><span class="weight-eyebrow">03 / Serialized structure</span><h3>Operation connections</h3></div><span>${links.length} connected operation${links.length === 1 ? "" : "s"}</span></header>${connections || `<p class="weight-connection-empty">${source === "activation" ? "No serialized operation port is bound to this captured value." : "No serialized operation binding is available for this tensor. A name alone does not establish a layer relationship."}</p>`}
      ${links.length ? `<details><summary>All ${links.length} operation connections${links.length > 12 ? " · diagram shows first 12" : ""}</summary><ul>${links.map(({operation:op,ports}) => `<li><strong>${esc(operationLabel(op))}</strong> · ${esc(op.native_op?.name || op.kind)}<code>${esc(op.id)}</code><span>${esc(ports.join(", "))}</span></li>`).join("")}</ul></details>` : ""}<p class="weight-chart-note">Connections come from serialized references. They do not measure correlation or causal importance, and do not identify training layers or runtime scheduling.</p></div>
      <details class="weight-raw"><summary>Exact tensor evidence</summary><pre>${esc(JSON.stringify(row,null,2))}</pre></details>`;
    button("svg").disabled = button("png").disabled = !lastSvg;
  }
  function renderMethod() {
    $('[data-method]').innerHTML = `<p>Weight statistics describe stored payloads. Integer codes are not automatically dequantized. Counts are exact; floating moments are rounded binary64. A missing or unassessed statistic is never replaced with zero. Each chart is a projection of the exported IR.</p><p>Analysis reads at most 100,000,000 values per request. Any payload outside the decoding or value budget remains explicitly not assessed.</p>${model ? `<dl><dt>Model IR SHA-256</dt><dd><code>${esc(model.model_ir_sha256)}</code></dd></dl>` : ""}${weight ? `<dl><dt>Weight IR SHA-256</dt><dd><code>${esc(weight.weight_ir_sha256)}</code></dd></dl>` : ""}${activation ? `<h4>Imported execution evidence</h4><p>${activation.coverage.captured_count}/${activation.coverage.requested_count} requested values captured; ${activation.coverage.missing_count} missing. A hash-bound capture is not independent execution attestation.</p><pre>${esc(JSON.stringify({run:activation.run,coverage:activation.coverage,missing:activation.missing},null,2))}</pre>` : ""}`;
  }
  function analyzeWeights() {
    try {
      const context = requireContext(); if (worker) return;
      if (!context.source) throw new Error("Select one serialized artifact file for numeric inspection. Package sidecars need local CLI analysis.");
      const digest = model.model_ir_sha256, ticket = ++generation;
      button("weights").disabled = true; button("cancel").hidden = false; $("progress").hidden = false; $("progress").removeAttribute("value");
      setStatus("Checking artifact identity before reading stored values…");
      const activeWorker = new Worker(new URL("../workers/numerical-ir-worker.js",import.meta.url),{type:"module"}); worker = activeWorker;
      const finish = () => {activeWorker.terminate(); if (worker === activeWorker) {worker = null; button("weights").disabled = !model; button("cancel").hidden = true; $("progress").hidden = true;}};
      activeWorker.onerror = event => {if (ticket !== generation) return; setStatus(event.message || "Numeric analysis worker failed.",true); finish();};
      activeWorker.onmessage = event => {
        if (ticket !== generation) return;
        if (getContext()?.model?.model_ir_sha256 !== digest) {finish(); sync(); return;}
        if (event.data.progress) {const p = event.data.progress; if (p.phase === "weights") {setStatus(`Reading tensors: ${p.completed} / ${p.total}`); $("progress").max = Math.max(1,p.total); $("progress").value = p.completed;} return;}
        finish(); if (event.data.error) {setStatus(event.data.error,true); return;}
        weight = event.data.result; source = "weight"; page = 0;
        setStatus(`${weight.coverage.assessed_count}/${weight.coverage.inventory_count} payloads assessed · ${countLabel(weight.coverage.decoded_value_count)} values · ${weight.coverage.not_assessed_count} not assessed.`);
        rebuild();
      };
      activeWorker.postMessage({model,analysis:context.analysis,source:context.source});
    } catch (error) {stopWorker(); setStatus(error.message,true);}
  }
  function download(name, blob) {
    const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href = url; link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),60_000);
  }
  function saveJson(name, value) {download(name,new Blob([JSON.stringify(value,null,2)+"\n"],{type:"application/json"}));}
  async function exportFigure(format) {
    requireContext(); if (!lastSvg) throw new Error("Select a tensor with an assessed histogram first.");
    const row = index.get(selected).row, colors = palette(), caption = `${rowName(row)} · ${row.dtype} · ${shapeLabel(row)}`;
    const inner = lastSvg.replace(/^<svg[^>]*>/,"").replace(/<\/svg>$/,"");
    const numericalDigest = source === "weight" ? weight.weight_ir_sha256 : activation.activation_ir_sha256;
    const figure = `<svg xmlns="http://www.w3.org/2000/svg" width="1520" height="790" viewBox="0 0 760 395"><rect width="760" height="395" fill="${colors.surface}"/><title>${esc(model.artifact.filename)} · ${esc(caption)}</title><text x="18" y="25" font-family="sans-serif" font-size="15" fill="${colors.ink}">${esc(caption.slice(0,90))}</text><g transform="translate(0 38)">${inner}</g><g font-family="monospace" font-size="8" fill="${colors.muted}"><text x="18" y="357">Model IR SHA-256: ${esc(model.model_ir_sha256)}</text><text x="18" y="373">${source === "weight" ? "Weight" : "Activation"} IR SHA-256: ${esc(numericalDigest)}</text></g></svg>`;
    const blob = new Blob([figure],{type:"image/svg+xml"}), stem = `deepbom-${source}-${rowName(row).replace(/[^a-zA-Z0-9._-]/g,"_").slice(0,80)}`;
    if (format === "svg") {download(stem+".svg",blob); return;}
    const url = URL.createObjectURL(blob);
    try {const image = new Image(); image.src = url; await image.decode(); const canvas = document.createElement("canvas"); canvas.width = 1520; canvas.height = 790; canvas.getContext("2d").drawImage(image,0,0,1520,790); const png = await new Promise(resolve => canvas.toBlob(resolve,"image/png")); if (!png) throw new Error("PNG export failed."); download(stem+".png",png);} finally {URL.revokeObjectURL(url);}
  }
  host.addEventListener("click", event => {
    const tensor = event.target.closest("[data-tensor-id]");
    if (tensor && index.has(tensor.dataset.tensorId)) {
      const section = tensor.closest('[data-inventory]') ? '[data-inventory]' : '[data-map]';
      selected = tensor.dataset.tensorId; renderInventory();
      if (event.detail === 0) [...$(section).querySelectorAll('[data-tensor-id]')].find(item => item.dataset.tensorId === selected)?.focus({preventScroll:true});
      return;
    }
    const tab = event.target.closest("[data-source]");
    if (tab) {sync(); source = tab.dataset.source; selected = null; page = 0; rebuild(); return;}
    const paging = event.target.closest("[data-page]");
    if (paging) {page += paging.dataset.page === "next" ? 1 : -1; page = Math.max(0,page); selected = null; renderInventory(); return;}
    if (event.target.closest('[data-action="weights-inline"]')) analyzeWeights();
    const bin = event.target.closest("[data-bin]"); if (bin) inspectBin(bin);
  });
  function inspectBin(bin) {const histogram = index.get(selected)?.row?.statistics?.histogram; if (!histogram) return; const binIndex = Number(bin.dataset.bin); $('[data-bin-readout]').textContent = `Interval ${intervalLabel(histogram,binIndex)} · ${countLabel(histogram.counts[binIndex])} values. Counts are exact; bin widths are not uniform in value space.`;}
  host.addEventListener("mouseover", event => {const bin = event.target.closest("[data-bin]"); if (bin) inspectBin(bin);});
  host.addEventListener("focusin", event => {const bin = event.target.closest("[data-bin]"); if (bin) inspectBin(bin);});
  host.addEventListener("keydown", event => {const svgButton = event.target.closest('svg [role="button"]'); if (svgButton && ["Enter"," "].includes(event.key)) {event.preventDefault(); svgButton.dispatchEvent(new MouseEvent("click",{bubbles:true}));}});
  host.addEventListener("change", event => {if (event.target.matches("[data-log-count]")) {logCount = event.target.checked; renderCharts();}});
  for (const control of [search,$('[data-filter="dtype"]'),$('[data-filter="status"]'),$('[data-sort]')]) control.addEventListener(control === search ? "input" : "change",() => {sync(); page = 0; renderInventory();});
  button("weights").onclick = analyzeWeights;
  button("cancel").onclick = () => {stopWorker(); setStatus("Weight analysis cancelled. No incomplete result was applied.");};
  button("import").onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    try {
      requireContext(); const file = fileInput.files[0]; if (!file) return;
      if (file.size > 16*1024*1024) throw new Error("Activation capture exceeds 16 MiB.");
      const boundModel = model, digest = model.model_ir_sha256, text = await file.text();
      if (getContext()?.model?.model_ir_sha256 !== digest) throw new Error("Model changed during import. Select a capture for the current artifact.");
      activation = buildActivationIr(boundModel,parseStrictJson(text,"activation capture")); source = "activation"; selected = null; page = 0;
      setStatus(`${activation.coverage.captured_count}/${activation.coverage.requested_count} requested values captured · ${activation.coverage.missing_count} missing. Imported capture; no execution performed here.`);
      rebuild();
    } catch (error) {sync(); setStatus(error.message,true);} finally {fileInput.value = "";}
  };
  button("save").onclick = () => {try {requireContext(); if (!weight && !activation) throw new Error("No numeric evidence is available for this artifact."); saveJson("deepbom-numerical-evidence.json",{bundle:buildNumericalEvidenceBundle(model,weight,activation),model_ir:model,weight_ir:weight,activation_ir:activation});} catch (error) {setStatus(error.message,true);}};
  button("model").onclick = () => {try {requireContext(); saveJson("deepbom-model-ir.json",model);} catch (error) {setStatus(error.message,true);}};
  for (const type of ["svg","png"]) button(type).onclick = () => exportFigure(type).catch(error => setStatus(error.message,true));
  const themeObserver = new MutationObserver(() => renderCharts());
  themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  button("weights").disabled = button("model").disabled = button("import").disabled = true;
  rebuild(); sync();
  return {sync,destroy() {stopWorker(); themeObserver.disconnect();}};
}
