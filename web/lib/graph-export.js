import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { buildHierarchicalGraphProjection } from "./graph-hierarchy.js";

export const VISUALIZATION_MANIFEST_SCHEMA = "deepbom.visualization_manifest.v1";
const NODE_WIDTH = 248;
const NODE_HEIGHT = 94;
const GAP_X = 54;
const GAP_Y = 56;
const MARGIN = 48;

export function exportGraphVisualization(graph, { view = "structure", format = "svg", compact = false } = {}) {
  const { projection, scene } = buildGraphVisualizationScene(graph, view);
  const manifestBody = {
    schema: VISUALIZATION_MANIFEST_SCHEMA,
    graph_ir_sha256: graph.graph_ir_sha256,
    view,
    format,
    full_node_count: graph.nodes.length,
    rendered_node_count: projection.nodes.length,
    rendered_edge_count: projection.edges.length,
    level_of_detail: projection.level,
    hierarchy_conservation: projection.hierarchy?.conservation || null,
    executable_dag_claim: graph.projection.executable_dag_claim,
    dimensions: { width: scene.width, height: scene.height },
    determinism: "canonical_graph_ir_stable_order_and_fixed_layout_v1",
    interpretation_boundary: graph.interpretation_boundary,
  };
  const manifest = { ...manifestBody, visualization_manifest_sha256: sha256TextHex(canonicalJson(manifestBody)) };
  if (format === "json") return { text: `${JSON.stringify({ graph_ir: graph, visualization_manifest: manifest }, null, compact ? 0 : 2)}\n`, mediaType: "application/json", manifest };
  const svg = renderSvg(graph, scene, manifest);
  if (format === "svg") return { text: svg, mediaType: "image/svg+xml", manifest };
  if (format === "html") return { text: renderHtml(graph, svg, manifest), mediaType: "text/html", manifest };
  if (format === "mermaid") return { text: renderMermaid(graph, projection, manifest), mediaType: "text/plain", manifest };
  if (format === "dot") return { text: renderDot(graph, projection, manifest), mediaType: "text/vnd.graphviz", manifest };
  throw new Error(`Unsupported graph export format: ${format}.`);
}

export function buildGraphVisualizationScene(graph, view = "structure") {
  const projection = project(graph, view);
  return { projection, scene: layout(projection.nodes, projection.edges) };
}

function project(graph, view) {
  if (!["structure", "placement", "quantization", "architecture"].includes(view)) throw new Error(`Unsupported graph view: ${view}.`);
  if (graph.nodes.length <= 2000) return { nodes: graph.nodes, edges: graph.edges.filter((edge) => edge.from && edge.to), level: "full_graph" };
  const hierarchy = buildHierarchicalGraphProjection(graph);
  return { nodes: hierarchy.nodes, edges: hierarchy.edges, level: `${hierarchy.level}_exact_contraction`, hierarchy };
}

function layout(nodes, edges) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const depth = new Map();
  for (const node of nodes) depth.set(node.id, Number.isSafeInteger(node.topo_depth) ? node.topo_depth : node.index);
  for (const edge of edges) {
    if (!byId.has(edge.from) || !byId.has(edge.to)) continue;
    depth.set(edge.to, Math.max(depth.get(edge.to) || 0, (depth.get(edge.from) || 0) + 1));
  }
  const layers = new Map();
  for (const node of nodes) {
    const key = depth.get(node.id) || 0;
    if (!layers.has(key)) layers.set(key, []);
    layers.get(key).push(node);
  }
  const orderedDepths = [...layers.keys()].sort((a, b) => a - b);
  const positions = new Map();
  let maximumLaneCount = 1;
  orderedDepths.forEach((layerDepth, row) => {
    const layer = layers.get(layerDepth).sort((left, right) => left.index - right.index);
    maximumLaneCount = Math.max(maximumLaneCount, layer.length);
    layer.forEach((node, lane) => positions.set(node.id, { x: MARGIN + lane * (NODE_WIDTH + GAP_X), y: MARGIN + 72 + row * (NODE_HEIGHT + GAP_Y) }));
  });
  return {
    nodes: nodes.map((node) => ({ ...node, ...positions.get(node.id) })), edges,
    width: Math.max(760, MARGIN * 2 + maximumLaneCount * NODE_WIDTH + Math.max(0, maximumLaneCount - 1) * GAP_X),
    height: Math.max(420, MARGIN * 2 + 72 + orderedDepths.length * NODE_HEIGHT + Math.max(0, orderedDepths.length - 1) * GAP_Y),
    positions,
  };
}

function renderSvg(graph, scene, manifest) {
  const edges = scene.edges.map((edge) => {
    const from = scene.positions.get(edge.from), to = scene.positions.get(edge.to);
    if (!from || !to) return "";
    const x1 = from.x + NODE_WIDTH / 2, y1 = from.y + NODE_HEIGHT, x2 = to.x + NODE_WIDTH / 2, y2 = to.y;
    const middle = (y1 + y2) / 2;
    return `<path class="edge" d="M${x1} ${y1} C${x1} ${middle} ${x2} ${middle} ${x2} ${y2}"><title>${xml(edge.tensor_name || edge.id)}${edge.dtype ? ` · ${xml(edge.dtype)}` : ""}${edge.shape?.length ? ` · ${xml(edge.shape.join("×"))}` : ""}</title></path>`;
  }).join("");
  const nodes = scene.nodes.map((node) => {
    const tone = nodeTone(node, manifest.view);
    const detail = node.output_shapes?.[0]?.length ? node.output_shapes[0].join("×")
      : node.estimated_bytes?.decimal ? `${formatBytes(node.estimated_bytes.decimal)} serialized/logical` : node.secondary_label || "";
    const mac = node.macs?.decimal && node.macs.decimal !== "0" ? `${formatInteger(node.macs.decimal)} MACs` : node.placement?.status || node.kind;
    return `<g class="node ${tone}" data-node-id="${xml(node.id)}" data-search="${xml(`${node.index} ${node.label} ${node.secondary_label || ""} ${node.domain} ${node.placement?.backend || ""} ${node.placement?.status || ""}`.toLowerCase())}" transform="translate(${node.x} ${node.y})"><rect width="${NODE_WIDTH}" height="${NODE_HEIGHT}" rx="6"/><text class="index" x="14" y="20">#${String(node.index).padStart(3, "0")} · ${xml(node.domain)}</text><text class="label" x="14" y="45">${xml(clip(node.label, 30))}</text><text class="detail" x="14" y="67">${xml(clip(detail, 36))}</text><text class="metric" x="14" y="84">${xml(clip(mac, 38))}</text><title>${xml(node.label)} · ${xml(node.placement?.status || "")}</title></g>`;
  }).join("");
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${scene.width}" height="${scene.height}" viewBox="0 0 ${scene.width} ${scene.height}" role="img" aria-labelledby="title desc"><title id="title">${xml(graph.artifact.filename)} · ${xml(manifest.view)} graph</title><desc id="desc">${xml(graph.interpretation_boundary)}</desc><style>:root{color-scheme:light}.background{fill:#f5f2ea}.edge{fill:none;stroke:#8d887d;stroke-width:1.5;opacity:.72}.node rect{fill:#fffdfa;stroke:#777064;stroke-width:1.4}.node .index{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#6a6258}.node .label{font:700 15px Inter,Arial,sans-serif;fill:#191714}.node .detail,.node .metric{font:11px Inter,Arial,sans-serif;fill:#514b43}.node.delegated rect{stroke:#177d6b;stroke-width:2}.node.fallback rect{stroke:#b44c3e;stroke-width:2}.node.quantized rect{fill:#edf7f3;stroke:#177d6b}.node.risk rect{fill:#fff1ed;stroke:#b44c3e}.heading{font:700 20px Inter,Arial,sans-serif;fill:#191714}.subheading{font:12px Inter,Arial,sans-serif;fill:#625b51}</style><rect class="background" width="100%" height="100%"/><text class="heading" x="${MARGIN}" y="34">${xml(graph.artifact.filename)}</text><text class="subheading" x="${MARGIN}" y="55">${xml(`${manifest.view} · ${manifest.rendered_node_count}/${manifest.full_node_count} nodes · ${manifest.level_of_detail}`)}</text><g class="edges">${edges}</g><g class="nodes">${nodes}</g></svg>\n`;
}

function renderHtml(graph, svg, manifest) {
  const embedded = svg.replace(/^<\?xml[^>]+>\s*/, "");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${html(graph.artifact.filename)} · DEEPBOM graph</title><style>html,body{margin:0;height:100%;font-family:Inter,Arial,sans-serif;background:#f5f2ea;color:#191714}body{display:flex;flex-direction:column;overflow:hidden}header{min-height:52px;box-sizing:border-box;display:flex;flex:0 0 auto;align-items:center;gap:8px;padding:8px 12px;border-bottom:1px solid #d5d0c5;background:#fffdfa}button,input{box-sizing:border-box;height:34px;border:1px solid #aaa296;background:#fffdfa;color:#191714;border-radius:4px;padding:0 10px}button{min-width:36px}input{width:min(360px,38vw);min-width:120px}#viewport{min-height:0;flex:1 1 auto;overflow:hidden;touch-action:none}#scene{transform-origin:0 0;will-change:transform}svg{display:block;max-width:none}.match rect{stroke:#d17d00!important;stroke-width:3!important}@media(max-width:620px){header{flex-wrap:wrap;gap:6px;padding:6px 8px}header strong{flex:1 1 auto}header input{order:2;flex:1 1 100%;width:100%}header span{font-size:12px;margin-left:auto}}</style></head><body><header><strong>DEEPBOM graph</strong><button id="fit" type="button">Fit</button><button id="minus" type="button" aria-label="Zoom out">−</button><button id="plus" type="button" aria-label="Zoom in">+</button><input id="search" type="search" placeholder="Find op, index, stage, domain"><span>${manifest.rendered_node_count}/${manifest.full_node_count} nodes</span></header><div id="viewport"><div id="scene">${embedded}</div></div><script>const viewport=document.querySelector('#viewport'),scene=document.querySelector('#scene'),svg=scene.querySelector('svg');let scale=1,x=0,y=0,drag=null;function apply(){scene.style.transform='translate('+x+'px,'+y+'px) scale('+scale+')'}function fit(){const r=viewport.getBoundingClientRect(),w=Number(svg.getAttribute('width')),h=Number(svg.getAttribute('height'));scale=Math.min(r.width/w,r.height/h,.98);x=(r.width-w*scale)/2;y=(r.height-h*scale)/2;apply()}function zoom(f,cx=viewport.clientWidth/2,cy=viewport.clientHeight/2){const next=Math.max(.08,Math.min(4,scale*f));x=cx-(cx-x)*next/scale;y=cy-(cy-y)*next/scale;scale=next;apply()}viewport.addEventListener('wheel',e=>{e.preventDefault();const r=viewport.getBoundingClientRect();zoom(e.deltaY<0?1.12:.89,e.clientX-r.left,e.clientY-r.top)},{passive:false});viewport.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,ox:x,oy:y};viewport.setPointerCapture(e.pointerId)});viewport.addEventListener('pointermove',e=>{if(drag){x=drag.ox+e.clientX-drag.x;y=drag.oy+e.clientY-drag.y;apply()}});viewport.addEventListener('pointerup',()=>drag=null);document.querySelector('#fit').onclick=fit;document.querySelector('#minus').onclick=()=>zoom(.8);document.querySelector('#plus').onclick=()=>zoom(1.25);document.querySelector('#search').oninput=e=>{const q=e.target.value.trim().toLowerCase();for(const n of svg.querySelectorAll('.node'))n.classList.toggle('match',!!q&&n.dataset.search.includes(q))};addEventListener('resize',fit);fit();</script></body></html>\n`;
}

function renderMermaid(graph, projection, manifest) {
  const ids = new Map(projection.nodes.map((node, index) => [node.id, `N${index}`]));
  const lines = [`%% ${manifest.schema} ${manifest.visualization_manifest_sha256}`, "flowchart TD"];
  for (const node of projection.nodes) lines.push(`  ${ids.get(node.id)}["${mermaid(`#${node.index} ${node.label}`)}"]`);
  for (const edge of projection.edges) if (ids.has(edge.from) && ids.has(edge.to)) lines.push(`  ${ids.get(edge.from)} -->|"${mermaid(edge.tensor_name || "tensor")}"| ${ids.get(edge.to)}`);
  lines.push(`  %% ${graph.interpretation_boundary}`);
  return `${lines.join("\n")}\n`;
}

function renderDot(graph, projection, manifest) {
  const lines = ["digraph deepbom {", "  rankdir=TB;", `  graph [label="${dot(graph.artifact.filename)} · ${dot(manifest.view)}", labelloc=t];`, "  node [shape=box, style=rounded];"];
  for (const node of projection.nodes) lines.push(`  "${dot(node.id)}" [label="#${node.index} ${dot(node.label)}"];`);
  for (const edge of projection.edges) if (edge.from && edge.to) lines.push(`  "${dot(edge.from)}" -> "${dot(edge.to)}" [label="${dot(edge.tensor_name || "tensor")}"];`);
  lines.push("}");
  return `${lines.join("\n")}\n`;
}

function nodeTone(node, view) {
  if (view === "placement") {
    const status = String(node.placement?.status || "");
    return status.includes("DELEGATABLE") || status.includes("ELIGIBLE") || status.includes("ACCELERATOR") ? "delegated"
      : status.includes("FALLBACK") || status.includes("OTHER_POOL") ? "fallback" : "neutral";
  }
  if (view === "quantization") return node.quantization?.risk && node.quantization.risk !== "none" ? "risk" : node.quantization?.state !== "none" ? "quantized" : "neutral";
  return "neutral";
}

function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }
function clip(value, maximum) { const text = String(value || ""); return text.length <= maximum ? text : `${text.slice(0, maximum - 1)}…`; }
function formatInteger(value) { return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
function formatBytes(value) { let n=Number(value),i=0;const u=["B","KiB","MiB","GiB","TiB"];while(n>=1024&&i<u.length-1){n/=1024;i+=1}return `${n.toFixed(i?1:0)} ${u[i]}`; }
function xml(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[char]); }
function html(value) { return xml(value); }
function mermaid(value) { return String(value || "").replace(/["\n\r]/g, " "); }
function dot(value) { return String(value || "").replaceAll("\\", "\\\\").replaceAll('"', '\\"').replace(/[\n\r]/g, " "); }
