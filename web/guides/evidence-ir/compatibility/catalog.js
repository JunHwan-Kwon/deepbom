const $ = id => document.getElementById(id);
const base = "/schemas/compatibility/", github = "https://github.com/JunHwan-Kwon/deepbom/blob/main/";
let catalog, byId, index, selected = "", loadSequence = 0, initialLayer = true;
const params = new URLSearchParams(location.search);
const controls = { q: $("query"), layer: $("layer"), direction: $("direction"), status: $("mapping-status") };
const text = (tag, value, className) => { const node = document.createElement(tag); node.textContent = value; if (className) node.className = className; return node; };
const link = (label, href) => { const node = text("a", label); node.href = href; return node; };
const badge = status => { const node = text("span", status, "badge"); node.dataset.status = status; return node; };
const kind = row => byId.get(row.from).kind === "input" ? "input" : byId.get(row.from).kind === "ir" && byId.get(row.to).kind === "ir" ? "ir" : "output";
function updateUrl() {
  const state = new URLSearchParams({ version: catalog.catalog_version });
  for (const [key, control] of Object.entries(controls)) if (control.value) state.set(key, control.value);
  if (selected) state.set("mapping", selected);
  history.replaceState(null, "", `${location.pathname}?${state}`);
}
function showDetail(row) {
  const box = $("mapping-detail"); box.replaceChildren(); box.scrollTop = 0;
  if (!row) { box.append(text("p", "No mapping is selected. Adjust the search or filters.")); return; }
  const from = byId.get(row.from), to = byId.get(row.to);
  box.append(badge(row.status), text("h3", `${from.name} → ${to.name}`), text("p", row.id, "mapping-id"), text("p", row.boundary, "mapping-boundary"));
  const table = document.createElement("table"); table.className = "field-paths";
  const head = document.createElement("thead"), header = document.createElement("tr");
  header.append(text("th", "Source field group"), text("th", "Destination field group")); head.append(header); table.append(head);
  const body = document.createElement("tbody");
  for (const field of row.fields) { const tr = document.createElement("tr"); for (const value of [field.source, field.target]) { const td = document.createElement("td"); td.append(text("code", value)); tr.append(td); } body.append(tr); }
  table.append(body); if (row.fields.length) box.append(table); else box.append(text("p", "No field mapping is implemented for this proposed path."));
  box.append(text("h4", "Contracts"));
  for (const node of [from, to]) { const p = text("p", `${node.name}: ${node.contract} `); if (node.json_schema) p.append(link("JSON Schema", node.json_schema)); box.append(p); }
  const neighbors = catalog.mappings.filter(other => other.id !== row.id && (other.from === row.to || other.to === row.from));
  if (neighbors.length) {
    box.append(text("h4", "Follow connected mappings")); const nav = text("nav", "", "mapping-neighbors"); nav.setAttribute("aria-label", "Connected mappings");
    for (const other of neighbors) {
      const button = text("button", `${byId.get(other.from).name} → ${byId.get(other.to).name}`); button.type = "button";
      button.addEventListener("click", () => { for (const control of Object.values(controls)) control.value = ""; selected = other.id; render(); $("mapping-detail").focus({ preventScroll: true }); });
      nav.append(button);
    }
    box.append(nav);
  }
  for (const [title, files] of [["Implementation references", row.implementation], ["Related regression checks", row.checks]]) {
    box.append(text("h4", title)); const list = document.createElement("ul");
    for (const file of files) { const li = document.createElement("li"); li.append(link(file, github + file)); list.append(li); }
    box.append(files.length ? list : text("p", "No implementation check claimed."));
  }
  box.append(text("p", "References show where to inspect and test this path; they are not a test-run receipt. Actual coverage is recorded in each artifact's evidence."));
  const deepLink = new URL(location.href); deepLink.search = new URLSearchParams({ version: catalog.catalog_version, mapping: row.id }).toString();
  box.append(link("Permanent catalog selection", deepLink.href));
}
function render() {
  if (!catalog) return;
  const query = controls.q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matches = catalog.mappings.filter(row => {
    const searchable = JSON.stringify([row, byId.get(row.from), byId.get(row.to)]).toLowerCase();
    return query.every(word => searchable.includes(word)) && (!controls.layer.value || [row.from, row.to].includes(controls.layer.value)) && (!controls.direction.value || controls.direction.value === kind(row)) && (!controls.status.value || controls.status.value === row.status);
  });
  if (!matches.some(row => row.id === selected)) selected = matches[0]?.id || "";
  const list = $("mapping-list"); list.replaceChildren();
  for (const row of matches) {
    const button = document.createElement("button"); button.type = "button"; button.className = "mapping-row"; button.dataset.mapping = row.id; button.setAttribute("aria-pressed", String(row.id === selected));
    button.append(text("strong", `${byId.get(row.from).name} → ${byId.get(row.to).name}`), badge(row.status), text("small", row.id));
    button.addEventListener("click", () => { selected = row.id; for (const item of list.children) item.setAttribute("aria-pressed", String(item === button)); showDetail(row); updateUrl(); });
    list.append(button);
  }
  $("empty").hidden = matches.length !== 0;
  $("result-count").textContent = `${matches.length} of ${catalog.mappings.length} documented mappings · catalog ${catalog.catalog_version}. Counts describe this catalog, not artifact coverage.`;
  showDetail(matches.find(row => row.id === selected)); updateUrl();
  const chosen = list.querySelector('[aria-pressed="true"]');
  if (chosen) { const region = list.parentElement; region.scrollTop += chosen.getBoundingClientRect().top - region.getBoundingClientRect().top; }
}
function inventory(kind, target) {
  const root = $(target); root.replaceChildren();
  for (const node of catalog.nodes.filter(node => node.kind === kind)) {
    const card = text("article", "", "schema-card"); card.append(text("h3", node.name), text("code", node.contract), text("p", node.boundary));
    if (node.json_schema) card.append(link("JSON Schema", node.json_schema));
    const first = catalog.mappings.find(row => row.from === node.id || row.to === node.id);
    if (first) { const href = new URL(location.href); href.search = new URLSearchParams({ version: catalog.catalog_version, mapping: first.id }).toString(); card.append(link("Inspect mapping", href.href)); }
    root.append(card);
  }
}
async function load(version) {
  const sequence = ++loadSequence; catalog = null;
  $("mapping-list").replaceChildren(); $("mapping-detail").replaceChildren(); $("inputs").replaceChildren(); $("outputs").replaceChildren();
  $("source-pins").replaceChildren(); $("snapshot-digest").textContent = ""; $("baseline").textContent = "";
  $("load-error").hidden = true; $("download-catalog").hidden = true; $("revision").textContent = "Verifying catalog snapshot…";
  try {
    const entry = index.snapshots.find(item => item.version === version);
    if (!entry || entry.file !== `${version}.json` || !/^\d+\.\d+\.\d+$/.test(version)) throw Error("This catalog version is not available in the published index.");
    const response = await fetch(base + entry.file); if (!response.ok) throw Error(`Catalog request failed (${response.status}).`);
    const bytes = await response.arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(n => n.toString(16).padStart(2, "0")).join("");
    if (hash !== entry.sha256) throw Error("Catalog checksum mismatch. Reload after the deployment completes; this snapshot will not be displayed.");
    const data = JSON.parse(new TextDecoder().decode(bytes));
    if (data.schema !== "deepbom.evidence_compatibility_catalog.v1" || data.catalog_version !== version) throw Error("Unsupported or mismatched catalog identity.");
    if (sequence !== loadSequence) return;
    catalog = data; byId = new Map(catalog.nodes.map(node => [node.id, node]));
    $("version").value = version; $("revision").textContent = `${catalog.status} baseline · snapshot verified · engine release is tracked separately`;
    const download = $("download-catalog"); download.href = base + entry.file; download.download = `deepbom-compatibility-${version}.json`; download.hidden = false;
    $("baseline").textContent = catalog.baseline;
    $("snapshot-digest").textContent = `SHA-256 (complete snapshot bytes): ${hash}`;
    $("source-pins").replaceChildren(...catalog.source_files.map(pin => { const li = document.createElement("li"); li.append(link(pin.path, github + pin.path), text("code", pin.sha256)); return li; }));
    $("input-count").textContent = `${catalog.nodes.filter(n => n.kind === "input").length} input contracts · maturity and limits retained`;
    $("output-count").textContent = `${catalog.nodes.filter(n => n.kind === "output").length} output routes · includes explicitly proposed exchange`;
    const layer = controls.layer, previous = initialLayer ? params.get("layer") : layer.value; initialLayer = false; layer.replaceChildren(new Option("All layers", ""), ...catalog.nodes.filter(n => n.kind === "ir").map(node => new Option(node.name, node.id))); layer.value = previous || "";
    if (!layer.value) layer.value = "";
    inventory("input", "inputs"); inventory("output", "outputs"); render();
  } catch (error) {
    if (sequence !== loadSequence) return;
    $("load-error").textContent = error.message; $("load-error").hidden = false; $("revision").textContent = "Catalog unavailable"; $("result-count").textContent = "No unverified mappings displayed.";
  }
}
$("filters").addEventListener("submit", event => event.preventDefault());
for (const [key, control] of Object.entries(controls)) { control.value = params.get(key) || ""; control.addEventListener(control.tagName === "INPUT" ? "input" : "change", render); }
$("reset").addEventListener("click", () => { for (const control of Object.values(controls)) control.value = ""; render(); });
$("version").addEventListener("change", () => load($("version").value));
try {
  const response = await fetch(base + "index.json"); if (!response.ok) throw Error(`Catalog index request failed (${response.status}).`);
  index = await response.json(); if (index.schema !== "deepbom.evidence_compatibility_index.v1") throw Error("Unsupported catalog index.");
  $("version").replaceChildren(...index.snapshots.map(item => new Option(`${item.version} · ${item.status}`, item.version))); $("version").disabled = false;
  selected = params.get("mapping") || "onnx-artifact";
  await load(params.get("version") || index.current);
} catch (error) { $("load-error").textContent = error.message; $("load-error").hidden = false; $("revision").textContent = "Catalog unavailable"; }
