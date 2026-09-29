import { buildProvenanceIr } from "./provenance-ir.js";
import { EVIDENCE_IR_NAME, PROVENANCE_IR } from "./evidence-ir.js";
import { metadataTemplate, fillOmopTemplate } from "./provenance/omop.js";
import { parseMetadataText, observeBrowserEvidence } from "./provenance/files.js";
import { PROVENANCE_LIMITS, PRIMARY_MODEL_REF, NODE_KINDS, RELATION_ROLES } from "./provenance/contracts.js";
import { PROVENANCE_FILENAME, provenanceJson, projectProvenanceToCycloneDx } from "./provenance/cyclonedx.js";
import { buildPublicCycloneDx17ArtifactContract } from "./public-cyclonedx-export.js";
import { parseStrictJson } from "./strict-json.js";

const esc = value => String(value ?? "").replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const words = value => String(value).replaceAll("_", " ");

export function installProvenancePanel(host, getContext, { onResult = () => {}, onDownload = localDownload } = {}) {
  if (!host || host.dataset.evidenceLinksPanel !== undefined) return null;
  if (!document.querySelector("link[data-provenance-style]")) { const style = document.createElement("link"); style.rel = "stylesheet"; style.href = new URL("/web/provenance.css", import.meta.url).href; style.dataset.evidenceLinksStyle = ""; document.head.append(style); }
  host.dataset.evidenceLinksPanel = ""; host.classList.add("provenance-workspace");
  host.innerHTML = `<header class="links-heading"><div><span class="links-eyebrow">${EVIDENCE_IR_NAME}</span><h2>${PROVENANCE_IR.name}</h2><p>Connect this model to data, code and evaluation records. Inspect the basis of each relationship.</p></div><span class="links-local">Local files · optional</span></header>
    <div class="links-toolbar"><button type="button" data-link-action="template">OMOP template</button><button type="button" data-link-action="generic">Generic template</button><button type="button" data-link-action="import">Import metadata JSON</button><input type="file" data-link-file="metadata" accept="application/json,.json" hidden aria-label="Import metadata JSON"><button type="button" data-link-action="files">Choose evidence files</button><input type="file" data-link-file="evidence" multiple hidden aria-label="Choose evidence files"><span data-link-file-count>No evidence files selected</span></div>
    <details class="links-editor" open><summary>Metadata input</summary><div class="links-omop" data-link-omop hidden><p>Use a JSON export of one <code>CDM_SOURCE</code> row. Keep patient records in your institution.</p><label>Instance identifier<input data-link-field="instance_id" autocomplete="off" placeholder="Institution-scoped identifier"></label><label>Data release identifier<input data-link-field="release_id" autocomplete="off" placeholder="Specific release or snapshot"></label><label>CDM version<input data-link-field="cdm_version" placeholder="5.4 or 5.5"></label><button type="button" data-link-action="cdm">Import CDM_SOURCE JSON</button><input type="file" data-link-file="cdm" accept="application/json,.json" hidden aria-label="Import CDM_SOURCE JSON"></div><label class="links-json-label">Model-bound metadata JSON<textarea data-link-editor rows="12" spellcheck="false" aria-label="Metadata JSON"></textarea></label><div class="links-toolbar"><button type="button" class="links-primary" data-link-action="apply">Check connections</button><button type="button" data-link-action="save-input">Save metadata input</button></div></details>
    <details class="links-editor"><summary>Add a connected record</summary><p>Create a declaration linking data, code or an external report. Supporting files are read only when selected separately.</p><div class="links-omop"><label>Record kind<select data-link-new="kind">${NODE_KINDS.map(kind => `<option>${kind}</option>`).join("")}</select></label><label>Name<input data-link-new="name" placeholder="Cohort definition, ETL release, evaluation report…"></label><label>From record<select data-link-new="from"></select></label><label>Relationship<select data-link-new="role">${RELATION_ROLES.map(role => `<option>${role}</option>`).join("")}</select></label><label>Supporting filename (optional)<input data-link-new="file" placeholder="cohort-definition.json"></label><label>Expected SHA-256 (optional)<input data-link-new="sha256" placeholder="Full digest supplied independently"></label><button type="button" data-link-action="add-record">Add record and connection</button></div></details>
    <p class="links-status" role="status" aria-live="polite">Analyze a model to begin.</p><div class="links-metrics" data-link-metrics></div>
    <div class="links-toolbar"><button type="button" data-link-action="save-ir" disabled>Save ${PROVENANCE_IR.name}</button><button type="button" data-link-action="save-bom" disabled>Save linked CycloneDX</button></div>
    <div class="links-layout" data-link-results hidden><aside><label>Find a record<input type="search" data-link-search placeholder="Name, kind or identifier"></label><div class="links-inventory" data-link-inventory></div></aside><div class="links-canvas"><div class="links-map" data-link-map></div><section data-link-detail aria-label="Selected metadata record"></section></div></div>
    <details class="links-checks" data-link-checks hidden><summary>All checks and mapping coverage</summary><div data-link-check-table></div></details>
    <p class="links-boundary">Hash matches identify provided bytes. Relationships remain declarations; training, execution, publisher authenticity and clinical suitability are not established. A manifest hash is not a database hash.</p>`;
  const $ = selector => host.querySelector(selector), action = name => $(`[data-link-action="${name}"]`), inputFile = name => $(`[data-link-file="${name}"]`);
  const editor = $('[data-link-editor]'), status = $('[role="status"]');
  let model = null, result = null, files = [], generation = 0, selected = PRIMARY_MODEL_REF;
  const say = (message, error = false) => { status.textContent = message; status.dataset.error = String(error); };
  function invalidate() { generation++; result = null; action("apply").disabled = !model; action("save-ir").disabled = action("save-bom").disabled = true; $('[data-link-results]').hidden = $('[data-link-checks]').hidden = true; $('[data-link-metrics]').replaceChildren(); onResult(null); }
  function setInput(value) { editor.value = JSON.stringify(value, null, 2); syncFields(value); invalidate(); }
  function syncFields(value) { const options = [{ id: PRIMARY_MODEL_REF, name: "Current model" }, ...(value?.omop ? [{ id: "omop:release", name: "OMOP release" }] : []), ...(value?.nodes || [])]; $('[data-link-new="from"]').innerHTML = options.map(node => `<option value="${esc(node.id)}">${esc(node.name)}</option>`).join(""); $('[data-link-omop]').hidden = !value?.omop; for (const field of host.querySelectorAll('[data-link-field]')) field.value = value?.omop?.[field.dataset.linkField] || ""; }
  function sync() {
    const next = getContext()?.model || null;
    if (next?.model_ir_sha256 === model?.model_ir_sha256) return false;
    model = next; files = []; selected = PRIMARY_MODEL_REF; invalidate();
    $('[data-link-file-count]').textContent = "No evidence files selected";
    for (const name of ["template", "generic", "import", "apply", "files", "cdm", "save-input", "add-record"]) action(name).disabled = !model;
    editor.disabled = !model; editor.value = ""; $('[data-link-omop]').hidden = true;
    say(model ? "Choose an OMOP template or import metadata for this model. Nothing is uploaded." : "Analyze a model to begin.");
    return true;
  }
  const current = () => { sync(); if (!model) throw new Error("Analyze a model first."); return model; };
  async function apply() {
    let token;
    try {
      const activeModel = current(); invalidate(); token = generation;
      const metadata = parseMetadataText(editor.value); syncFields(metadata);
      action("apply").disabled = true; say("Checking references and reading selected evidence files…");
      const observations = await observeBrowserEvidence(metadata, files);
      if (generation !== token || getContext()?.model?.model_ir_sha256 !== activeModel.model_ir_sha256) return;
      result = buildProvenanceIr(activeModel, metadata, { observations });
      onResult(result); render(); say(`${words(result.verdict.status)} · ${result.coverage.check_count} checks · relationship truth is not verified`);
    } catch (error) { if (token === undefined || token === generation) say(error.message, true); }
    finally { if (token === undefined || token === generation) action("apply").disabled = !model; }
  }
  function render() {
    if (!result) return;
    action("save-ir").disabled = action("save-bom").disabled = false; $('[data-link-results]').hidden = $('[data-link-checks]').hidden = false;
    const coverage = result.coverage;
    const node = result.nodes.find(node => node.id === selected) || result.nodes.find(node => node.id === PRIMARY_MODEL_REF); selected = node.id;
    $('[data-link-metrics]').innerHTML = [["Records", coverage.node_count], ["Declared relations", coverage.relationship_count], ["Files read", coverage.observed_file_count], ["Contradictions", coverage.check_counts.mismatch], ["Unmapped fields", coverage.unsupported_field_count]].map(([name, count]) => `<div><span>${esc(name)}</span><strong>${count}</strong></div>`).join("");
    const query = $('[data-link-search]').value.toLowerCase();
    $('[data-link-inventory]').innerHTML = result.nodes.filter(node => `${node.name} ${node.kind} ${node.id}`.toLowerCase().includes(query)).map(node => `<button type="button" data-link-node="${esc(node.id)}" aria-pressed="${selected === node.id}"><strong>${esc(node.name)}</strong><span>${esc(words(node.kind))}</span><code>${esc(node.id)}</code></button>`).join("");
    const edges = result.relationships.filter(edge => edge.from === node.id || edge.to === node.id);
    const relevantChecks = result.checks.filter(row => row.subject_ref === node.id || edges.some(edge => edge.id === row.subject_ref));
    const observed = result.observations.find(row => row.node_ref === node.id);
    $('[data-link-map]').innerHTML = graph(result, selected);
    $('[data-link-detail]').innerHTML = `<header><span class="links-eyebrow">Selected record</span><h3>${esc(node.name)}</h3><p>${esc(words(node.kind))} · ${esc(node.evidence_class)}</p></header><dl class="links-identity"><dt>Identifier</dt><dd>${esc(node.id)}</dd>${node.version ? `<dt>Version</dt><dd>${esc(node.version)}</dd>` : ""}<dt>Declared SHA-256</dt><dd>${esc(node.sha256 || "Not supplied")}</dd><dt>Observed file SHA-256</dt><dd>${esc(observed?.sha256 || (node.id === PRIMARY_MODEL_REF ? model.artifact.sha256 : "File not provided"))}</dd>${node.uri ? `<dt>Location (not fetched)</dt><dd>${esc(node.uri)}</dd>` : ""}</dl><h4>Relationships</h4>${edges.length ? `<ul class="links-relations">${edges.map(edge => `<li><strong>${esc(words(edge.role))}</strong><span>${esc(edge.from)} → ${esc(edge.to)}</span><small>${esc(words(edge.resolution))} · declared, not attested</small></li>`).join("")}</ul>` : '<p>No relationships supplied for this record.</p>'}<h4>Checks</h4>${checksTable(relevantChecks)}${node.attributes.length ? `<h4>Declared metadata</h4><dl class="links-identity">${node.attributes.map(row => `<dt>${esc(row.name)}</dt><dd>${esc(row.value === null ? "Not supplied" : row.value)}</dd>`).join("")}</dl>` : ""}`;
    $('[data-link-check-table]').innerHTML = `<p>${coverage.check_count} checks: ${coverage.check_counts.match} match, ${coverage.check_counts.mismatch} mismatch, ${coverage.check_counts.not_assessed} not assessed, ${coverage.check_counts.unresolved} unresolved, ${coverage.check_counts.unsupported} unsupported.</p><p>Field mapping: ${coverage.mapped_declared_field_count} mapped declarations and ${coverage.unsupported_field_count} preserved unmapped fields, out of ${coverage.field_count} inventoried fields. These are separate denominators; there is no overall verification percentage.</p>${checksTable(result.checks)}<p>${PROVENANCE_IR.name} SHA-256</p><code>${esc(result.provenance_ir_sha256)}</code>`;
  }
  async function fileText(file) { if (!file || file.size > PROVENANCE_LIMITS.input_bytes) throw new Error("Choose a JSON file of at most 2 MiB."); return await file.text(); }
  for (const [name, kind] of [["template", "omop"], ["generic", "generic"]]) action(name).addEventListener("click", () => { try { setInput(metadataTemplate(current(), kind)); say(kind === "omop" ? "Enter the instance and release identifiers, then import CDM_SOURCE or edit the JSON." : "Add metadata nodes and relationships, then check connections."); } catch (error) { say(error.message, true); } });
  for (const [button, file] of [["import", "metadata"], ["files", "evidence"], ["cdm", "cdm"]]) action(button).addEventListener("click", () => inputFile(file).click());
  inputFile("metadata").addEventListener("change", async () => { invalidate(); const token = generation; try { const value = parseMetadataText(await fileText(inputFile("metadata").files[0])); if (token !== generation) return; setInput(value); say("Metadata loaded. Select any supporting files, then check connections."); } catch (error) { if (token === generation) say(error.message, true); } finally { inputFile("metadata").value = ""; } });
  inputFile("cdm").addEventListener("change", async () => { invalidate(); const token = generation; try { const source = parseStrictJson(await fileText(inputFile("cdm").files[0]), "CDM_SOURCE"); if (token !== generation) return; const template = parseStrictJson(editor.value, "metadata"); setInput(fillOmopTemplate(template, source)); say("CDM_SOURCE imported. Supply the institution-scoped instance and release identifiers, then check connections."); } catch (error) { if (token === generation) say(error.message, true); } finally { inputFile("cdm").value = ""; } });
  inputFile("evidence").addEventListener("change", () => { files = [...inputFile("evidence").files]; invalidate(); $('[data-link-file-count]').textContent = `${files.length} evidence files selected`; });
  editor.addEventListener("input", () => { invalidate(); say("Metadata changed. Check connections again to update the evidence."); });
  editor.addEventListener("change", () => { try { syncFields(parseStrictJson(editor.value)); } catch { /* The Apply action reports input errors. */ } });
  for (const field of host.querySelectorAll('[data-link-field]')) field.addEventListener("change", () => { try { const value = parseStrictJson(editor.value); value.omop[field.dataset.linkField] = field.value; setInput(value); } catch (error) { say(error.message, true); } });
  action("add-record").addEventListener("click", () => {
    try {
      current(); const value = parseStrictJson(editor.value, "metadata"), get = key => $(`[data-link-new="${key}"]`).value.trim();
      if (!get("name")) throw new Error("Enter a record name.");
      const usedIds = new Set([...value.nodes, ...value.relationships].map(row => row.id));
      let index = 1; while (usedIds.has(`record:${index}`) || usedIds.has(`link:${index}`)) index++;
      const node = { id: `record:${index}`, kind: get("kind"), name: get("name"), ...(get("file") ? { file: get("file") } : {}), ...(get("sha256") ? { sha256: get("sha256") } : {}) };
      value.nodes.push(node); value.relationships.push({ id: `link:${index}`, from: get("from"), to: node.id, role: get("role") });
      setInput(value); say("Record added as a declaration. Check connections to validate its references and role.");
    } catch (error) { say(error.message, true); }
  });
  action("apply").addEventListener("click", () => { void apply(); });
  action("save-input").addEventListener("click", async () => { try { const value = parseStrictJson(editor.value); await onDownload(new Blob([JSON.stringify(value, null, 2) + "\n"], { type: "application/json" }), "deepbom_metadata_input.json"); } catch (error) { say(error.message, true); } });
  action("save-ir").addEventListener("click", async () => { try { if (sync() || !result) return; await onDownload(new Blob([provenanceJson(result)], { type: "application/json" }), PROVENANCE_FILENAME); } catch (error) { say(error.message, true); } });
  action("save-bom").addEventListener("click", async () => {
    try { if (sync() || !result) return; const context = getContext(); const base = (typeof context.cyclonedx === "function" ? context.cyclonedx() : context.cyclonedx) || buildPublicCycloneDx17ArtifactContract(context.analysis, { hash: model.artifact.sha256 }); const bom = projectProvenanceToCycloneDx(base, result, model); await onDownload(new Blob([JSON.stringify(bom, null, 2) + "\n"], { type: "application/json" }), "deepbom_linked.cdx.json"); } catch (error) { say(error.message, true); }
  });
  $('[data-link-search]').addEventListener("input", render);
  host.addEventListener("click", event => { const button = event.target.closest('[data-link-node]'); if (button) { selected = button.dataset.linkNode; render(); } });
  host.addEventListener("keydown", event => { const node = event.target.closest('g[data-link-node]'); if (node && ["Enter", " "].includes(event.key)) { event.preventDefault(); selected = node.dataset.linkNode; render(); host.querySelector('g[data-link-node][aria-pressed="true"]')?.focus(); } });
  // Initial null context must still disable actions.
  model = { model_ir_sha256: "uninitialized" }; sync();
  return { sync, getResult: () => result, apply };
}

function checksTable(rows) {
  if (!rows.length) return "<p>No independent checks were applicable.</p>";
  return `<div class="links-table-scroll"><table><thead><tr><th>Check</th><th>Status</th><th>Scope</th></tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row.kind)}</td><td><span class="links-check-state" data-state="${esc(row.status)}">${esc(words(row.status))}</span></td><td>${esc(row.message)}</td></tr>`).join("")}</tbody></table></div>`;
}

function graph(ir, selected) {
  const neighborhood = new Set([selected]);
  for (const edge of ir.relationships) if (edge.from === selected || edge.to === selected) { neighborhood.add(edge.from); neighborhood.add(edge.to); }
  const nodes = [ir.nodes.find(node => node.id === selected), ...ir.nodes.filter(node => node.id !== selected && neighborhood.has(node.id)).slice(0, 24)].filter(Boolean), positions = new Map();
  const center = nodes.find(node => node.id === selected); if (center) positions.set(center.id, { x: 270, y: 30 });
  nodes.filter(node => node.id !== selected).forEach((node, index) => positions.set(node.id, { x: index % 2 ? 520 : 20, y: 145 + Math.floor(index / 2) * 100 }));
  const height = Math.max(210, 245 + Math.floor(Math.max(0, nodes.length - 2) / 2) * 100);
  const edges = ir.relationships.filter(edge => positions.has(edge.from) && positions.has(edge.to));
  const svg = `<svg viewBox="0 0 760 ${height}" role="group" aria-label="Selected metadata record and its declared relationships"><defs><marker id="links-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>${edges.map(edge => { const a = positions.get(edge.from), b = positions.get(edge.to); return `<path class="links-edge" d="M${a.x + 105},${a.y + 60} C${a.x + 105},${a.y + 95} ${b.x + 105},${b.y - 35} ${b.x + 105},${b.y}" marker-end="url(#links-arrow)"><title>${esc(words(edge.role))} · ${esc(words(edge.resolution))} · declared</title></path>`; }).join("")}${nodes.map(node => { const point = positions.get(node.id); return `<g tabindex="0" role="button" aria-label="Select ${esc(node.name)}" aria-pressed="${node.id === selected}" data-link-node="${esc(node.id)}" transform="translate(${point.x},${point.y})"><title>${esc(node.name)} · ${esc(node.id)}</title><rect width="210" height="64" rx="8"/><text x="12" y="23">${esc(node.name.length > 25 ? node.name.slice(0, 24) + "…" : node.name)}</text><text class="links-node-kind" x="12" y="45">${esc(words(node.kind))}</text></g>`; }).join("")}</svg>`;
  return `${svg}<p class="links-graph-note">${nodes.length} of ${ir.nodes.filter(node => neighborhood.has(node.id)).length} neighboring records shown. Select a record to navigate. Dashed arrows are declared relationships; direction and role are listed below. Missing endpoints remain in the checks.</p>`;
}

function localDownload(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href = url; link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
}
