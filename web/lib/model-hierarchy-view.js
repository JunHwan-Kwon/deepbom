import { projectModelHierarchy, validateModelHierarchy } from "./model-hierarchy.js";
import { validateModelIr } from "./model-ir.js";
import { downloadText } from "./download.js";

const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const exact = value => value?.decimal == null ? "not assessed" : `${value.decimal}${value.status === "partial" ? " (partial)" : ""}`;
const clip = (value, count = 38) => String(value || "").length > count ? `${String(value).slice(0, count - 1)}…` : String(value || "");
const stores = new WeakMap();

export function renderModelHierarchySvg(view, { theme = "light", selected = null, metric = "structure" } = {}) {
  if (view?.schema !== "deepbom.model_hierarchy_view.v1" || !/^[a-f0-9]{64}$/.test(view.view_sha256 || "") || !/^[a-f0-9]{64}$/.test(view.source?.artifact_sha256 || "")) throw new Error("Hierarchy renderer requires a source-bound view projection.");
  if (!["light", "dark", "monochrome"].includes(theme)) throw new Error("Unknown hierarchy theme.");
  if (!["structure", "macs", "storage"].includes(metric)) throw new Error("Unknown hierarchy display metric.");
  if (view.rows.length > 200) throw new Error("Select a smaller module for a readable diagram (maximum 200 visible nodes).");
  const ids = new Set(view.rows.map(row => row.id)), incoming = new Map([...ids].map(id => [id, new Set()]));
  const children = new Map([...ids].map(id => [id, new Set()]));
  for (const edge of view.edges) if (ids.has(edge.from_ref) && ids.has(edge.to_ref)) {
    incoming.get(edge.to_ref).add(edge.from_ref); children.get(edge.from_ref).add(edge.to_ref);
  }
  const depths = new Map([...ids].map(id => [id, 0])), done = new Set();
  const queue = [...ids].filter(id => !incoming.get(id).size);
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i]; done.add(id);
    for (const to of children.get(id)) {
      depths.set(to, Math.max(depths.get(to), depths.get(id) + 1));
      incoming.get(to).delete(id); if (!incoming.get(to).size) queue.push(to);
    }
  }
  // Non-convex source groups can induce cycles. Keep every edge and use a
  // deterministic display row; never pretend that grouping changed the DAG.
  let fallback = Math.max(0, ...depths.values()) + 1;
  for (const id of ids) if (!done.has(id)) depths.set(id, fallback++);
  const layers = new Map(), positions = new Map();
  for (const row of view.rows) {
    const depth = depths.get(row.id);
    if (!layers.has(depth)) layers.set(depth, []);
    layers.get(depth).push(row);
  }
  let lanes = 1;
  [...layers.keys()].sort((a, b) => a - b).forEach((depth, rank) => {
    const rows = layers.get(depth); lanes = Math.max(lanes, rows.length);
    rows.forEach((row, lane) => positions.set(row.id, { x: 70 + lane * 300, y: 104 + rank * 152 }));
  });
  const detours = new Map();
  for (const edge of view.edges) {
    const from = positions.get(edge.from_ref), to = positions.get(edge.to_ref);
    if (from && to && to.y - from.y !== 152) detours.set(edge.id, detours.size);
  }
  const routingStart = lanes * 300 + 80;
  const width = Math.max(560, routingStart + Math.max(1, detours.size) * 12 + 50), height = Math.max(270, layers.size * 152 + 200);
  const palette = theme === "dark" ? ["#111b23", "#edf1f5", "#22313b", "#879ba8", "#80d3be"]
    : theme === "monochrome" ? ["#fff", "#111", "#fff", "#444", "#111"] : ["#f7f9fa", "#192b35", "#fff", "#718591", "#176b59"];
  const [bg, fg, nodeBg, line, accent] = palette;
  const scope = `mh-${view.view_sha256.slice(0, 16)}-${theme}-${metric}`;
  const marker = `${scope}-arrow`;
  const connectors = view.edges.map((edge, edgeIndex) => {
    const from = positions.get(edge.from_ref), to = positions.get(edge.to_ref);
    if (!from && !to) return "";
    let d;
    if (!from) d = `M${to.x - 28} ${to.y + 45}H${to.x}`;
    else if (!to) d = `M${from.x + 260} ${from.y + 45}h28`;
    else {
      const x1 = from.x + 130, y1 = from.y + 96, x2 = to.x + 130, y2 = to.y;
      if (!detours.has(edge.id)) { const mid = (y1 + y2) / 2; d = `M${x1} ${y1}V${mid}H${x2}V${y2}`; }
      else { const lane = routingStart + detours.get(edge.id) * 12; d = `M${x1} ${y1}V${y1 + 12}H${lane}V${y2 - 16}H${x2}V${y2}`; }
    }
    return `<g data-edge-ref="${esc(edge.id)}"><path d="${d}" fill="none" stroke="${line}" stroke-width="1.5" marker-end="url(#${marker})"/><title>${edge.edge_refs.length} exact port edge(s); ${esc(edge.from_ref || "outside / input / storage")} → ${esc(edge.to_ref || "outside / model output")}</title></g>`;
  }).join("");
  const highlighted = new Set(view.highlighted_operation_refs);
  const nodes = view.rows.map(row => {
    const { x, y } = positions.get(row.id);
    const active = row.id === selected || row.member_refs.some(ref => highlighted.has(ref));
    const values = row.aggregation, output = values.outputs[0];
    const shape = output ? `${output.shape === null ? "shape unknown" : output.shape.length ? output.shape.join(" × ") : "scalar"} · ${output.dtype || "dtype unknown"}` : "No output boundary";
    const label = metric === "macs" ? `${exact(values.macs)} MACs` : metric === "storage" ? `${exact(values.serialized_element_count)} stored elements` : shape;
    return `<g data-subject-ref="${esc(row.id)}" tabindex="0" role="button" aria-label="${esc(row.label)}; ${row.member_refs.length} operations" transform="translate(${x} ${y})"><rect width="260" height="96" rx="8" fill="${nodeBg}" stroke="${active ? accent : line}" stroke-width="${active ? 3 : 1}"/><text x="14" y="24" font-weight="600">${esc(clip(row.label, 31))}</text><text x="14" y="45" class="secondary">${esc(clip(row.native_type || (row.kind === "group" ? "Source / structural group" : "Serialized operation"), 38))}</text><text x="14" y="65" class="secondary">${esc(clip(label, 37))}</text><text x="14" y="84" class="secondary">${row.member_refs.length} ops · ${values.inputs.length} inputs · ${values.outputs.length} outputs</text><title>${esc(row.label)}\n${esc(label)}\n${esc(row.native_type)}\n${esc(row.id)}</title></g>`;
  }).join("");
  return `<svg class="${scope}" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Model source hierarchy" data-view-sha256="${view.view_sha256}"><metadata>${esc(JSON.stringify({source:view.source,hierarchy_sha256:view.hierarchy_sha256,view_sha256:view.view_sha256,selection:view.selection,rendering:{theme,metric,layout:"native_axes_not_inferred",static_macs_not_latency:true}}))}</metadata><defs><marker id="${marker}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="${line}"/></marker></defs><style>.${scope} text{font:14px system-ui,sans-serif;fill:${fg}}.${scope} .secondary{font-size:12px}.${scope} g[role=button]{cursor:pointer}.${scope} g[role=button]:focus rect{stroke:${accent};stroke-width:3}</style><rect width="100%" height="100%" fill="${bg}"/><text x="38" y="35" font-weight="600">Model hierarchy · ${view.conservation.represented_operations}/${view.conservation.total_operations} operations represented</text><text x="38" y="59" class="secondary">${esc(view.source.artifact_sha256.slice(0, 16))} · collapsed membership retained · not an execution schedule</text>${connectors}${nodes}<text x="38" y="${height - 46}" class="secondary">Port-derived connections · external boundaries retained</text><text x="38" y="${height - 28}" class="secondary">Static MACs · trainable parameters unknown</text></svg>`;
}

export function renderModelHierarchyTable(view, format = "markdown") {
  const rows = [["Subject", "Source / operation", "Operations", "Serialized elements", "Serialized bytes", "MACs", "MAC status", "Input values", "Output values"],
    ...view.rows.map(row => [row.id, row.label, row.member_refs.length, exact(row.aggregation.serialized_element_count),
      exact(row.aggregation.serialized_byte_length), exact(row.aggregation.macs), row.aggregation.macs.status,
      row.aggregation.boundary.input_value_refs.join("; "), row.aggregation.boundary.output_value_refs.join("; ")])];
  if (format === "csv") return rows.map(row => row.map(value => {
    let text = String(value); if (/^[=+@-]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  }).join(",")).join("\n") + "\n";
  if (format !== "markdown" && format !== "table") throw new Error("Unknown hierarchy table format.");
  const text = rows.map(row => row.map(value => String(value).replaceAll("|", "\\|").replaceAll("\n", " ")));
  if (format === "table") return `DEEPBOM format-neutral model summary — hierarchy\nArtifact SHA-256: ${view.source.artifact_sha256}\nModel IR SHA-256: ${view.source.model_ir_sha256}\nHierarchy view SHA-256: ${view.view_sha256}\nRepresented operations: ${view.conservation.represented_operations} / ${view.conservation.total_operations}\n\n${text.map(row => row.join("\t")).join("\n")}\n\nTrainability: not_assessable_from_serialized_deployment_artifact\n${view.interpretation_boundary}\n`;
  return `Model IR: ${view.source.model_ir_sha256}\n\n| ${text[0].join(" | ")} |\n| ${text[0].map(() => "---").join(" | ")} |\n${text.slice(1).map(row => `| ${row.join(" | ")} |`).join("\n")}\n\n${view.interpretation_boundary}\n`;
}

export function renderModelHierarchyHtml(view) {
  const svg = renderModelHierarchySvg(view, { theme: "monochrome" });
  const records = view.rows.map(row => `<details id="${esc(row.id)}"><summary>${esc(row.label)} · ${row.member_refs.length} operations</summary><pre>${esc(JSON.stringify(row, null, 2))}</pre></details>`).join("");
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model hierarchy — selected view</title><style>body{font:15px system-ui,sans-serif;color:#171717;background:#fff;max-width:1100px;margin:32px auto;padding:0 20px}h1{font-size:24px}.diagram{overflow:auto;max-height:760px;border:1px solid #aaa}.diagram svg{max-width:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;padding:12px;border-bottom:1px solid #ccc}@media print{.diagram{max-height:none;overflow:visible}.diagram svg{max-width:100%;height:auto}details{break-inside:avoid}}</style><h1>Model hierarchy — selected view</h1><p>Artifact SHA-256: <code>${view.source.artifact_sha256}</code></p><p>${view.conservation.represented_operations} of ${view.conservation.total_operations} operations represented. Collapsed members remain in the record. ${view.conservation.omitted_operation_refs.length} operations are outside this view. Select a smaller module before printing when the diagram would be unreadable.</p><div class="diagram">${svg}</div><h2>Source-bound group and operation records</h2>${records}<p>${esc(view.interpretation_boundary)}</p><script>document.querySelectorAll('[data-subject-ref]').forEach(function(n){function open(){var d=document.getElementById(n.getAttribute('data-subject-ref'));if(d){d.open=true;d.scrollIntoView({block:'start'});}}n.addEventListener('click',open);n.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});});</script></html>`;
}

export function mountModelHierarchy(container, model, hierarchy, { onSelect = () => {} } = {}) {
  // Validate imported/worker-cloned values once, then retain only the immutable
  // results. Revalidating a mutable caller object on every click is both slow
  // and vulnerable to changing the source underneath a mounted view.
  try {
    model = validateModelIr(model);
    hierarchy = validateModelHierarchy(hierarchy, model);
  } catch (error) {
    stores.delete(container); container.replaceChildren(); container.textContent = `Hierarchy unavailable: ${error.message}`; throw error;
  }
  let state = stores.get(container);
  if (state?.identity !== hierarchy.hierarchy_sha256) state = { identity: hierarchy.hierarchy_sha256, expanded: [], selected: null, scope: null, focus: null, search: "", theme: "light", metric: "structure" };
  else state = { ...state }; // Preserve the view, invalidate previous mount callbacks.
  stores.set(container, state);
  const el = (tag, text = "") => { const node = document.createElement(tag); node.textContent = text; return node; };
  const button = (label, fn) => { const node = el("button", label); node.type = "button"; node.addEventListener("click", fn); return node; };
  const shell = el("section"); shell.className = "model-hierarchy-panel";
  const style = el("style"); style.textContent = `.model-hierarchy-panel{border:1px solid #aab6bd;border-radius:10px;padding:16px;margin:12px 0;color:inherit;font:14px system-ui,sans-serif}.mh-tools{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.mh-tools button,.mh-tools input,.mh-tools select{font:inherit;padding:7px 10px;max-width:100%}.mh-body{display:grid;grid-template-columns:minmax(0,1fr) minmax(240px,30%);gap:12px}.mh-canvas{overflow:auto;max-height:640px;border:1px solid #aab6bd}.mh-canvas svg{max-width:none;height:auto}.mh-detail{overflow:auto;max-height:640px;white-space:pre-wrap;overflow-wrap:anywhere}.mh-table{overflow:auto;max-height:280px;margin-top:12px}.mh-table table{border-collapse:collapse;width:100%;font:inherit}.mh-table td,.mh-table th{padding:8px;border-bottom:1px solid #aab6bd;text-align:left}.mh-status{font-size:13px}.mh-detail button{margin:4px}@media(max-width:700px){.mh-body{grid-template-columns:1fr}.mh-canvas{max-height:360px}.mh-detail{max-height:320px}}@media(prefers-reduced-motion:reduce){.model-hierarchy-panel *{transition:none}}`;
  const title = el("h3", "Source hierarchy & tensor connections");
  const status = el("p"); status.className = "mh-status";
  const tools = el("div"); tools.className = "mh-tools";
  const search = el("input"); search.type = "search"; search.placeholder = "Search source path or operation"; search.setAttribute("aria-label", "Search source hierarchy"); search.value = state.search;
  const theme = el("select"); theme.setAttribute("aria-label", "Hierarchy theme");
  ["light", "dark", "monochrome"].forEach(value => { const option = el("option", value); option.value = value; theme.append(option); }); theme.value = state.theme;
  const metric = el("select"); metric.setAttribute("aria-label", "Hierarchy display metric");
  ["structure", "macs", "storage"].forEach(value => { const option = el("option", value); option.value = value; metric.append(option); }); metric.value = state.metric;
  const body = el("div"); body.className = "mh-body";
  const canvas = el("div"); canvas.className = "mh-canvas"; canvas.tabIndex = 0;
  const detail = el("div"); detail.className = "mh-detail";
  const table = el("div"); table.className = "mh-table";
  let view, svgText;
  const save = (extension, data, type) => downloadText(`model-hierarchy.${extension}`, data, type);
  const download = async (extension) => {
    if (!view || stores.get(container) !== state) return;
    const capturedView = view, capturedSvg = svgText;
    if (extension === "svg") save(extension, svgText, "image/svg+xml");
    else if (extension === "csv") save(extension, renderModelHierarchyTable(view, "csv"), "text/csv");
    else if (extension === "html") save(extension, renderModelHierarchyHtml(view), "text/html");
    else if (extension === "png") {
      try {
        const svg = canvas.querySelector("svg"), width = Number(svg.getAttribute("width")), height = Number(svg.getAttribute("height"));
        if (width * height * 2.25 > 32_000_000 || Math.max(width, height) * 1.5 > 16_384) throw new Error("Select a smaller module for PNG export; SVG and full JSON retain the larger view.");
        const { rasterizeColorModelView } = await import("./model-ir-browser-export.js");
        const bytes = await rasterizeColorModelView(capturedSvg, { width_mm: width * 25.4 / 96, height_mm: height * 25.4 / 96 }, { dpi: 144 });
        if (stores.get(container) === state && view === capturedView) save("png", bytes, "image/png");
      } catch (error) { if (stores.get(container) === state && view === capturedView) status.textContent = error.message; }
    }
    else save("json", JSON.stringify({ view, hierarchy }, null, 2), "application/json");
  };
  const choose = ref => { if (stores.get(container) !== state) return; state.selected = ref; draw(); const op = model.program.operations.find(row => row.id === ref); if (op) onSelect(op); };
  function draw() {
    if (stores.get(container) !== state) return;
    try {
      const next = projectModelHierarchy(model, hierarchy, { expanded: state.expanded, selected: state.scope, search: state.search, focus: state.focus });
      svgText = renderModelHierarchySvg(next, { theme: state.theme, selected: state.selected, metric: state.metric }); view = next;
      // SVG is generated exclusively by the escaping renderer above.
      const scroll = { left: canvas.scrollLeft, top: canvas.scrollTop };
      canvas.innerHTML = svgText;
      canvas.scrollLeft = scroll.left; canvas.scrollTop = scroll.top;
      canvas.querySelectorAll("[data-subject-ref]").forEach(node => {
        node.addEventListener("click", () => choose(node.getAttribute("data-subject-ref")));
        node.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(node.getAttribute("data-subject-ref")); } });
      });
      status.textContent = `${hierarchy.status} · ${hierarchy.coverage.annotated_operation_count}/${hierarchy.coverage.operation_count} source-annotated · ${view.conservation.represented_operations} represented · ${view.conservation.omitted_operation_refs.length} outside this view. ${hierarchy.coverage.non_data_relationship_count} control/state/call relationships retained separately in JSON. Training parameters and runtime calls are not inferred.`;
      detail.replaceChildren();
      const group = hierarchy.groups.find(row => row.id === state.selected);
      const row = view.rows.find(item => item.id === state.selected);
      if (group) {
        detail.append(el("h4", group.path), button(state.expanded.includes(group.id) ? "Collapse group" : "Expand group", () => {
          state.expanded = state.expanded.includes(group.id) ? state.expanded.filter(ref => ref !== group.id) : [...state.expanded, group.id]; draw();
        }), button("Focus this group", () => { state.scope = group.id; state.expanded = [...new Set([...state.expanded, group.id])]; state.search = ""; search.value = ""; draw(); }));
      }
      if (row?.kind === "operation") detail.append(button("Trace upstream data paths", () => { state.focus = row.id; draw(); }));
      if (row || group) detail.append(el("pre", JSON.stringify({ aggregation: row?.aggregation || group.aggregation,
        edge_details: hierarchy.edges.filter(edge => (row?.member_refs || group.member_refs).includes(edge.from_ref) || (row?.member_refs || group.member_refs).includes(edge.to_ref)) }, null, 2)));
      else detail.append(el("p", "Select a group or operation. Counts, boundary ports and exact edges come from the same Model IR projection as the table and exports."));
      const t = el("table"), head = el("tr"); ["Source / operation", "Ops", "MACs", "Stored elements"].forEach(label => head.append(el("th", label))); t.append(head);
      for (const item of view.rows) { const tr = el("tr"), td = el("td"); td.append(button(item.label, () => choose(item.id))); tr.append(td, el("td", item.member_refs.length), el("td", exact(item.aggregation.macs)), el("td", exact(item.aggregation.serialized_element_count))); t.append(tr); }
      table.replaceChildren(t);
    } catch (error) { view = null; canvas.replaceChildren(); detail.replaceChildren(); table.replaceChildren(); status.textContent = error.message; }
  }
  search.addEventListener("input", () => { state.search = search.value; draw(); });
  theme.addEventListener("change", () => { state.theme = theme.value; draw(); });
  metric.addEventListener("change", () => { state.metric = metric.value; draw(); });
  tools.append(search, theme, metric, button("Whole model", () => { state.scope = null; state.focus = null; state.expanded = []; state.search = ""; search.value = ""; draw(); }),
    button("Export SVG", () => download("svg")), button("Export PNG", () => download("png")), button("Export CSV", () => download("csv")),
    button("Export HTML / print PDF", () => download("html")), button("Export full evidence + view", () => download("json")));
  body.append(canvas, detail); shell.append(style, title, status, tools, body, table); container.replaceChildren(shell); draw();
  return { destroy() { if (stores.get(container) === state) { stores.delete(container); container.replaceChildren(); } } };
}
