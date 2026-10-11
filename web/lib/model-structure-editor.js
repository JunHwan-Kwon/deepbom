import { createStructureSession, indexStructure, inputsOf, outputsOf, structureMermaid, parseStructureMermaid, structureLayerSnippet, topologicalOrder, STRUCTURE_LAYERS, STRUCTURE_TEMPLATES } from "./model-structure.js";
import { downloadText } from "./download.js";

const NS = "http://www.w3.org/2000/svg";
function el(tag, cls = "", text = "") { const n = document.createElement(tag); n.className = cls; n.textContent = text; return n; }
function button(text, fn, cls = "") { const n = el("button", cls, text); n.type = "button"; n.addEventListener("click", fn); return n; }
function svg(tag, attrs = {}) { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; }
function contract(v) { return v ? `${v.shape == null ? "shape ?" : v.shape.length ? v.shape.join(" × ") : "scalar"} · ${v.dtype || "dtype ?"}` : "Disconnected"; }

export function createModelStructureEditor({ sourceContext = () => null } = {}) {
  let host, session, source, selected = null, region, draft, idx, zoom = 1, positions = new Map();
  let viewport, scene, wireLayer, message = "", textOpen = true, fromValue = null, portElements = new Map(), previews = [];
  let gestureCleanup = null, resizeObserver = null, candidateNumber = 0;
  let layerSearch = "", codeShare = 46, expanded = false, artifactInfo = null;
  const paneOffsets = new Map();
  const rememberPanes = () => { for (const node of host?.querySelectorAll('.se-palette,.se-inspector,.se-text-panel,.se-code-input') || []) paneOffsets.set(node.className, [node.scrollTop, node.scrollLeft]); };
  const textBuffers = new Map();
  const disposeGesture = () => { gestureCleanup?.(); gestureCleanup = null; };
  const command = edit => {
    disposeGesture();
    let applied = false;
    try { const near = positions.get(selected) || { x: 30, y: 110 }; session.apply(edit); message = ""; fromValue = null;
      if (["add", "duplicate"].includes(edit.type)) { selected = session.draft.changes.at(-1).created_ref; positions.set(selected, { x: near.x + 305, y: near.y }); }
      applied = true;
    } catch (error) { message = error.message; }
    render(); return applied;
  };
  const fail = error => { message = error.message; render(); };
  const hasPendingCode = () => [...textBuffers.values()].some(buffer => buffer.text !== buffer.baseText);
  const canSave = () => {
    if (!hasPendingCode()) return true;
    fail(new Error("Code edits are pending. Use Sync to code to apply them, or Sync to graph to replace them, before saving the project or candidate.")); return false;
  };
  const replaceProject = project => {
    if (!canSave()) return;
    if (project) session.load(project); else session.reset();
    textBuffers.clear(); positions.clear(); selected = null; fromValue = null; message = ""; render(); viewport?.scrollTo(0, 0); resetCodeView();
  };
  function resetCodeView() { const text = host?.querySelector('.se-code-input'); if (text) { text.scrollTo(0, 0); text.dispatchEvent(new Event('scroll')); } }
  function focusSelection() {
    const pos = positions.get(selected);
    if (pos) viewport?.scrollTo({ left: Math.max(0, pos.x * zoom - 36), top: Math.max(0, pos.y * zoom - 65), behavior: "smooth" });
  }
  function toggleFocus() {
    expanded = !expanded; render();
    [...host.querySelectorAll('button')].find(node => node.textContent === (expanded ? 'Exit focus' : 'Focus mode'))?.focus({ preventScroll: true });
  }
  function focusSemantics() {
    host.classList.toggle('se-expanded', expanded);
    if (expanded) { host.setAttribute('role', 'dialog'); host.setAttribute('aria-modal', 'true'); host.setAttribute('aria-label', 'Model workspace'); }
    else for (const name of ['role', 'aria-modal', 'aria-label']) host.removeAttribute(name);
  }
  function showLayerCode() {
    textOpen = true; render();
    const buffer = codeState();
    if (buffer.revision !== session.revision) return fail(new Error("Resolve the code conflict before locating a layer by its current ID."));
    const alias = structureMermaid(draft.program, region).names.get(selected), text = host.querySelector('[aria-label="Mermaid structure code"]');
    const match = new RegExp(`^\\s*${alias}\\[.*$`, "m").exec(text.value);
    if (match) { text.focus(); text.setSelectionRange(match.index, match.index + match[0].length); text.scrollTop = text.value.slice(0, match.index).split("\n").length * 20 - 60; text.dispatchEvent(new Event("scroll")); }
  }
  function codeState() {
    let buffer = textBuffers.get(region);
    if (!buffer || buffer.text === buffer.baseText) {
      const generated = structureMermaid(draft.program, region).text;
      buffer = { text: generated, baseText: generated, revision: session.revision };
      textBuffers.set(region, buffer);
    }
    return buffer;
  }
  function layerCode(options, insert) {
    try {
      const buffer = codeState();
      if (buffer.revision !== session.revision) throw new Error("Graph changed while code was being edited. Resolve the code conflict before inserting or copying a layer.");
      const snippet = structureLayerSnippet(draft.program, region, { ...options, source_ref: selected, text: buffer.text });
      textOpen = true; message = "";
      if (insert) {
        buffer.text = `${buffer.text}\n${snippet}`; render();
        const text = host.querySelector('.se-text-panel textarea'); text.focus(); text.setSelectionRange(text.value.length, text.value.length); text.scrollTop = text.scrollHeight; text.scrollLeft = 0; text.dispatchEvent(new Event('scroll'));
      } else {
        render();
        const panel = host.querySelector('.se-text-panel'), box = el("textarea", "se-snippet-preview");
        box.value = snippet; box.readOnly = true; box.setAttribute("aria-label", "Layer code to copy");
        const hint = el("p", "se-hint", "Copy this snippet into Structure code. Use a new ID for each copy; then connect its output.");
        panel.append(hint, box); box.focus(); box.select();
        navigator.clipboard?.writeText(snippet).then(() => { hint.textContent = "Copied. Paste into Structure code, connect the output, then Sync to code."; }).catch(() => { hint.textContent = "Clipboard unavailable. The snippet is selected; use Ctrl+C / Cmd+C to copy."; });
      }
    } catch (error) { fail(error); }
  }
  function arrange() {
    const ops = topologicalOrder(draft.program).filter(op => op.region_ref === region);
    const levels = new Map(), groups = new Map();
    for (const op of ops) {
      const parentLevels = draft.program.relationships.filter(e => e.kind === "data_dependency" && e.to_ref === op.id).map(e => levels.get(e.from_ref) ?? -1);
      const level = parentLevels.length ? Math.max(...parentLevels) + 1 : 0;
      levels.set(op.id, level);
      if (!groups.has(level)) groups.set(level, []); groups.get(level).push(op);
    }
    let y = 110;
    for (const list of groups.values()) {
      let height = 100;
      list.forEach((op, i) => { positions.set(op.id, { x: 70 + i * 305, y }); height = Math.max(height, nodeHeight(op)); });
      y += height + 65;
    }
    // Invalid cycles still remain visible and editable.
    for (const op of draft.program.operations.filter(op => op.region_ref === region)) if (!positions.has(op.id)) { positions.set(op.id, { x: 70, y }); y += nodeHeight(op) + 65; }
  }
  function nodeHeight(op) { return 74 + 28 * Math.max(op.input_port_refs.length, op.output_port_refs.length, 1); }
  function visibleOps() { return draft.program.operations.filter(op => op.region_ref === region); }
  function render() {
    if (!host || !session) return;
    disposeGesture();
    resizeObserver?.disconnect();
    const scroll = { x: viewport?.scrollLeft || 0, y: viewport?.scrollTop || 0 };
    rememberPanes();
    draft = session.draft; idx = indexStructure(draft.program);
    if (!region || !draft.program.regions.some(r => r.id === region)) region = draft.program.programs[0]?.entry_region_refs?.[0] || draft.program.regions[0]?.id;
    codeState();
    if (!positions.size) arrange();
    if (!idx.operations.has(selected)) selected = visibleOps()[0]?.id || null;
    host.replaceChildren(); host.classList.add("structure-ide"); focusSemantics();
    host.style.setProperty('--se-code-share', `${codeShare}fr`); host.style.setProperty('--se-graph-share', `${100 - codeShare}fr`);
    const header = el("div", "se-header");
    const title = el("div"); title.append(el("span", "se-kicker", "DEEPBOM / REDESIGN"), el("h3", "", "Model workspace"),
      el("p", "", "Connect layers, edit structure and review each change."));
    const badge = el("span", `se-badge ${draft.issues.some(i => i.severity === "error") ? "is-error" : ""}`,
      draft.issues.some(i => i.severity === "error") ? "Connections need attention" : draft.diff.length ? "Draft · partial checks" : "Original structure");
    header.append(title, badge); host.append(header);
    const binding = el("div", "se-source-strip");
    const identity = el("code", "", `SHA-256 ${artifactInfo.sha256.slice(0, 12)}…`); identity.title = artifactInfo.sha256;
    binding.append(el("strong", "", artifactInfo.filename || "Local model"), el("span", "", artifactInfo.format.toUpperCase()), identity,
      el("span", "se-source-boundary", "Original preserved · local structural draft")); host.append(binding);
    const tools = el("div", "se-toolbar");
    const undo = button("Undo", () => { session.undo(); message = ""; render(); }); undo.disabled = !session.canUndo;
    const redo = button("Redo", () => { session.redo(); message = ""; render(); }); redo.disabled = !session.canRedo;
    const regions = el("select"); regions.setAttribute("aria-label", "Graph region");
    for (const r of draft.program.regions) { const b = textBuffers.get(r.id), pending = b && b.text !== b.baseText;
      const option = el("option", "", `${r.name || r.id}${pending ? " · code edits" : ""}`); option.value = r.id; regions.append(option); }
    regions.value = region; regions.addEventListener("change", () => { region = regions.value; selected = null; positions.clear(); render(); viewport.scrollTo(0, 0); });
    tools.append(undo, redo, regions, button("Arrange", () => { positions.clear(); arrange(); render(); viewport.scrollTo(0, 0); }),
      button("Focus layer", focusSelection),
      button("−", () => { zoom = Math.max(.35, zoom - .15); render(); }), el("span", "se-zoom", `${Math.round(zoom * 100)}%`),
      button("+", () => { zoom = Math.min(1.8, zoom + .15); render(); }),
      button(textOpen ? "Hide code" : "Show code", () => { textOpen = !textOpen; render(); }),
      button(expanded ? "Exit focus" : "Focus mode", toggleFocus),
      button("Save project", () => { if (canSave()) downloadText("deepbom-structure-project.json", JSON.stringify(session.export(), null, 2), "application/json"); }),
      button("Open project", () => { if (canSave()) fileInput.click(); }),
      button("Reset", () => replaceProject(null)));
    const fileInput = el("input"); fileInput.type = "file"; fileInput.accept = ".json"; fileInput.hidden = true;
    fileInput.addEventListener("change", async () => { try { const file = fileInput.files[0]; if (!file) return;
      if (file.size > 2 * 1024 * 1024) throw new Error("Project exceeds 2 MiB.");
      if (!canSave()) return;
      const expected = session, revision = session.revision; const value = JSON.parse(await file.text());
      if (expected !== session) return;
      if (revision !== session.revision || hasPendingCode()) throw new Error("The workspace changed while this project was opening. Open the project again after finishing your edits.");
      replaceProject(value);
    } catch (error) { fail(error); } });
    tools.append(fileInput); host.append(tools);
    if (message) { const alert = el("p", "se-alert", message); alert.setAttribute("role", "alert"); host.append(alert); }
    const grid = el("div", `se-grid${textOpen ? " se-with-code" : ""}`); grid.append(palette());
    viewport = el("div", "se-viewport"); viewport.tabIndex = 0; viewport.setAttribute("aria-label", "Model structure canvas");
    viewport.addEventListener("keydown", event => { if (event.key === "Escape") { fromValue = null; disposeGesture(); render(); } });
    const canvasLabel = el("div", "se-canvas-label"); canvasLabel.append(el("strong", "", "GRAPH"), el("span", "", `${visibleOps().length} layers · drag ● → ○`)); viewport.append(canvasLabel);
    const space = el("div", "se-space"); scene = el("div", "se-scene"); wireLayer = svg("svg", { class: "se-wires", "aria-label": "Tensor connections" });
    scene.append(wireLayer); portElements = new Map();
    const external = externalValues();
    external.forEach((v, i) => {
      const n = el("div", "se-external"); n.style.left = `${20 + i * 235}px`; n.style.top = "12px";
      const p = portButton("output", v.id, v.name || "Model input"); p.title = contract(v); n.append(p); scene.append(n);
    });
    for (const op of visibleOps()) scene.append(node(op));
    let width = Math.max(760, external.length * 235 + 50), height = 500;
    for (const op of visibleOps()) { const pos = positions.get(op.id) || { x: 70, y: 110 }; width = Math.max(width, pos.x + 330); height = Math.max(height, pos.y + nodeHeight(op) + 100); }
    scene.style.width = `${width}px`; scene.style.height = `${height}px`; scene.style.transform = `scale(${zoom})`;
    space.style.width = `${width * zoom}px`; space.style.height = `${height * zoom}px`; space.append(scene); viewport.append(space);
    viewport.addEventListener("dragover", event => event.preventDefault());
    viewport.addEventListener("drop", event => { event.preventDefault(); const layer = event.dataTransfer.getData("application/x-deepbom-layer");
      if (STRUCTURE_LAYERS.includes(layer)) {
        const pos = { x: Math.max(0, (event.clientX - viewport.getBoundingClientRect().left + viewport.scrollLeft) / zoom),
          y: Math.max(110, (event.clientY - viewport.getBoundingClientRect().top + viewport.scrollTop) / zoom) };
        if (command({ type: "add", layer, region_ref: region })) { positions.set(selected, pos); render(); }
      }
    });
    grid.append(viewport);
    if (textOpen) grid.append(textPanel());
    grid.append(inspector()); host.append(grid);
    host.append(comparison());
    for (const [className, [top, left]] of paneOffsets) {
      const node = host.querySelector(`.${className.split(' ').join('.')}`);
      if (node) { node.scrollTop = top; node.scrollLeft = left; node.dispatchEvent(new Event("scroll")); }
    }
    viewport.scrollTo(scroll.x, scroll.y);
    resizeObserver = new ResizeObserver(() => wires()); resizeObserver.observe(viewport);
    requestAnimationFrame(() => { if (scene?.isConnected) wires(); });
  }
  function externalValues() {
    const produced = new Set(draft.program.ports.filter(p => p.direction === "output").map(p => p.value_ref));
    return draft.program.values.filter(v => v.region_ref === region && !produced.has(v.id) && !v.storage_refs?.length
      && (draft.program.programs.some(p => p.input_value_refs.includes(v.id)) || draft.program.ports.some(p => p.direction === "input" && p.value_ref === v.id)));
  }
  function palette() {
    const panel = el("aside", "se-palette"); panel.append(el("h4", "", "Layer library"), el("p", "se-hint", "Add to graph or insert editable code."));
    for (const template of STRUCTURE_TEMPLATES) {
      const layer = template.name, item = el("div", "se-template-card");
      const b = button(`+ ${layer}`, () => command({ type: "add", layer, region_ref: region }), "se-layer-template");
      b.draggable = true; b.addEventListener("dragstart", event => event.dataTransfer.setData("application/x-deepbom-layer", layer));
      const actions = el("div", "se-template-actions");
      actions.append(button("Insert code", () => layerCode({ layer }, true)), button("Copy code", () => layerCode({ layer }, false)));
      b.title = template.description;
      item.append(b, el("small", "se-hint", `${template.inputs} input${template.inputs > 1 ? 's' : ''} / ${template.outputs} output`), actions); panel.append(item);
    }
    panel.append(el("h4", "", `Model layers · ${visibleOps().length}`));
    const search = el("input"); search.type = "search"; search.placeholder = "Find a layer…"; search.setAttribute("aria-label", "Find a layer");
    search.value = layerSearch;
    const list = el("div", "se-layer-list");
    function fill() { list.replaceChildren(); for (const op of visibleOps().filter(op => `${op.name} ${op.native_op.name}`.toLowerCase().includes(search.value.toLowerCase()))) {
      const b = button(`${op.name || op.native_op.name}`, () => { selected = op.id; render(); focusSelection(); }, op.id === selected ? "is-selected" : "");
      b.append(el("small", "", `${op.native_index == null ? "Draft" : `#${op.native_index}`} · ${op.input_port_refs.length} → ${op.output_port_refs.length}`));
      b.title = op.id; b.setAttribute('aria-pressed', String(op.id === selected)); list.append(b);
    } if (!list.childElementCount) list.append(el('p', 'se-hint', 'No matching layers in this region.')); }
    search.addEventListener("input", () => { layerSearch = search.value; fill(); }); fill(); panel.append(search, list); return panel;
  }
  function node(op) {
    const n = el("article", `se-node ${selected === op.id ? "is-selected" : ""} ${draft.affected.includes(op.id) ? "is-changed" : ""}`);
    n.dataset.nodeRef = op.id;
    const pos = positions.get(op.id) || { x: 70, y: 110 }; positions.set(op.id, pos); n.style.left = `${pos.x}px`; n.style.top = `${pos.y}px`;
    if (draft.issues.some(row => row.subject_ref === op.id && row.severity === "error")) n.classList.add("has-error");
    const head = el("div", "se-node-head"); head.append(el("strong", "", op.name || op.native_op.name), el("span", "", op.draft_template ? "NEW LAYER" : `${op.native_op.name} · ${op.native_index ?? op.source_order ?? op.id}`));
    head.tabIndex = 0; head.setAttribute("role", "button"); head.setAttribute("aria-label", `Select ${op.name}`);
    head.addEventListener("keydown", event => { if (event.key === "Enter") { selected = op.id; render(); } });
    head.addEventListener("pointerdown", event => {
      if (event.button !== 0) return; event.preventDefault(); disposeGesture();
      const oldX = pos.x, oldY = pos.y;
      const move = e => { pos.x = Math.max(0, oldX + (e.clientX - event.clientX) / zoom); pos.y = Math.max(95, oldY + (e.clientY - event.clientY) / zoom);
        n.style.left = `${pos.x}px`; n.style.top = `${pos.y}px`; wires(); };
      const up = () => { disposeGesture(); selected = op.id; render(); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up, { once: true });
      window.addEventListener("pointercancel", up, { once: true });
      gestureCleanup = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
    });
    n.append(head);
    const io = el("div", "se-node-io"), left = el("div"), right = el("div");
    for (const p of inputsOf(op, idx)) { const v = idx.values.get(p.value_ref); const b = portButton("input", p.id, `in ${p.position}${v?.storage_refs?.length ? " · state" : ""}`); b.title = `${v?.name || "Unbound"}\n${contract(v)}`; if (!v) b.classList.add("is-unbound"); left.append(b); }
    for (const p of outputsOf(op, idx)) { const terminal = draft.program.programs.some(g => g.output_value_refs.includes(p.value_ref)); const b = portButton("output", p.value_ref, `out ${p.position}${terminal ? " · model" : ""}`); b.title = contract(idx.values.get(p.value_ref)); right.append(b); }
    io.append(left, right); n.append(io);
    const first = outputsOf(op, idx)[0]; n.append(el("div", "se-node-contract", contract(idx.values.get(first?.value_ref))));
    return n;
  }
  function portButton(direction, ref, label) {
    const b = el("button", `se-port se-${direction}`, `${direction === "input" ? "○ " : ""}${label}${direction === "output" ? " ●" : ""}`);
    b.type = "button"; b.dataset[direction === "input" ? "inputPort" : "outputValue"] = ref;
    b.setAttribute("aria-label", `${direction} ${label}`); portElements.set(`${direction}:${ref}`, b);
    if (direction === "output") {
      b.addEventListener("pointerdown", event => {
        if (event.button !== 0) return; event.preventDefault(); disposeGesture(); fromValue = ref;
        b.classList.add("is-connecting");
        const ghost = svg("path", { class: "se-wire is-changed", "stroke-dasharray": "5 4" }); wireLayer.append(ghost);
        const move = e => { const bounds = scene.getBoundingClientRect(), rect = b.getBoundingClientRect();
          const x = (rect.right - bounds.left) / zoom, y = (rect.top + rect.height / 2 - bounds.top) / zoom;
          const tx = (e.clientX - bounds.left) / zoom, ty = (e.clientY - bounds.top) / zoom;
          ghost.setAttribute("d", `M${x},${y} C${x + 50},${y} ${tx - 50},${ty} ${tx},${ty}`);
        };
        const up = e => { const target = document.elementFromPoint(e.clientX, e.clientY)?.closest("[data-input-port]"); disposeGesture();
          if (target && host.contains(target)) command({ type: "connect", port_ref: target.dataset.inputPort, value_ref: ref }); };
        window.addEventListener("pointerup", up, { once: true });
        window.addEventListener("pointermove", move);
        const cancel = () => { fromValue = null; disposeGesture(); };
        window.addEventListener("pointercancel", cancel, { once: true });
        gestureCleanup = () => { window.removeEventListener("pointerup", up); window.removeEventListener("pointermove", move); window.removeEventListener("pointercancel", cancel); ghost.remove(); b.classList.remove("is-connecting"); };
      });
      b.addEventListener("click", () => { fromValue = ref; b.classList.add("is-connecting"); });
    } else b.addEventListener("click", () => { if (fromValue) command({ type: "connect", port_ref: ref, value_ref: fromValue }); else { selected = idx.ports.get(ref).operation_ref; render(); } });
    return b;
  }
  function wires() {
    if (!wireLayer || !scene) return; wireLayer.replaceChildren(); const bounds = scene.getBoundingClientRect();
    for (const op of visibleOps()) for (const p of inputsOf(op, idx)) {
      const a = portElements.get(`output:${p.value_ref}`), b = portElements.get(`input:${p.id}`); if (!a || !b) continue;
      const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect();
      const x1 = (ar.right - bounds.left) / zoom, y1 = (ar.top + ar.height / 2 - bounds.top) / zoom;
      const x2 = (br.left - bounds.left) / zoom, y2 = (br.top + br.height / 2 - bounds.top) / zoom;
      const d = `M${x1},${y1} C${x1 + 55},${y1} ${x2 - 55},${y2} ${x2},${y2}`;
      const path = svg("path", { d, class: `se-wire ${draft.affected.includes(op.id) ? "is-changed" : ""}` });
      const title = svg("title"); title.textContent = `Input ${p.position}: ${idx.values.get(p.value_ref)?.name || p.value_ref}`; path.append(title); wireLayer.append(path);
    }
  }
  function inspector() {
    const panel = el("aside", "se-inspector"), op = idx.operations.get(selected);
    panel.append(el("h4", "", "Layer inspector")); if (!op) { panel.append(el("p", "", "Add or select a layer.")); return panel; }
    const name = el("input"); name.value = op.name || op.native_op.name; name.setAttribute("aria-label", "Layer name");
    name.addEventListener("change", () => command({ type: "rename", operation_ref: op.id, name: name.value })); panel.append(name, el("code", "se-ref", op.id));
    const actions = el("div", "se-layer-actions"); actions.append(button("Duplicate", () => command({ type: "duplicate", operation_ref: op.id })),
      button("Insert layer code", () => layerCode({ operation_ref: op.id }, true)), button("Copy layer code", () => layerCode({ operation_ref: op.id }, false)),
      button("Locate in code", showLayerCode),
      button("Bypass", () => command({ type: "bypass", operation_ref: op.id })), button("Delete", () => command({ type: "delete", operation_ref: op.id })));
    panel.append(actions, el("p", "se-hint", "Copies retain input and weight references. Bypass reconnects a unary layer's input and can change behavior."));
    const context = sourceContext(op);
    if (context) {
      const section = el("section", "se-source-context");
      section.append(el("span", "se-kicker", "ORIGINAL MODEL"), el("strong", "", context.label),
        el("p", "se-hint", context.detail));
      if (context.open) section.append(button("Open source block scenario", () => context.open(), "se-primary"));
      section.append(el("p", "se-hint", "Source scenarios do not include Structure IDE edits. Your draft is preserved when switching views.")); panel.append(section);
    }
    panel.append(el("h4", "", "Input connections"));
    const produced = new Map(draft.program.ports.filter(p => p.direction === "output").map(p => [p.value_ref, p]));
    for (const p of inputsOf(op, idx)) {
      const field = el("label", "se-field", `Input ${p.position}`), select = el("select"); select.setAttribute("aria-label", `Input ${p.position} connection`);
      const empty = el("option", "", "Disconnected"); empty.value = ""; select.append(empty);
      for (const v of draft.program.values.filter(v => v.region_ref === op.region_ref)) {
        const producer = produced.get(v.id), layer = idx.operations.get(producer?.operation_ref);
        const opt = el("option", "", layer ? `${layer.name} #${layer.native_index ?? layer.source_order ?? layer.id} · out ${producer.position}` : `${v.name || v.id}${v.storage_refs?.length ? " · state" : " · input"}`);
        opt.value = v.id; select.append(opt);
      }
      select.value = p.value_ref || ""; select.addEventListener("change", () => command({ type: "connect", port_ref: p.id, value_ref: select.value || null }));
      field.append(select, el("small", "", contract(idx.values.get(p.value_ref)))); panel.append(field);
      const insert = el("div", "se-insert"); insert.append(button("Insert ReLU", () => command({ type: "add", layer: "ReLU", region_ref: region, insert_at: p.id })),
        button("Insert Identity", () => command({ type: "add", layer: "Identity", region_ref: region, insert_at: p.id }))); panel.append(insert);
    }
    const attrs = (op.attributes?.entries || []).map(entry => {
      const field = ["int_value", "float_value", "string_value", "int_values", "float_values"].find(key => entry[key] != null && (!Array.isArray(entry[key]) || entry[key].length) && entry[key] !== "");
      return field ? { entry, field } : null;
    }).filter(Boolean);
    if (attrs.length) panel.append(el("h4", "", "Native attributes"));
    for (const { entry, field } of attrs) {
      const label = el("label", "se-field", entry.name), input = el("input");
      input.value = Array.isArray(entry[field]) ? entry[field].join(", ") : String(entry[field]); input.setAttribute("aria-label", entry.name);
      input.addEventListener("change", () => {
        const value = field === "string_value" ? input.value : field.endsWith("values") ? input.value.split(",").map(v => Number(v.trim())) : Number(input.value);
        if (field !== "string_value" && (!input.value.trim() || input.value.split(",").some(v => !v.trim()))) { fail(new Error("Enter an explicit numeric value.")); return; }
        command({ type: "attribute", operation_ref: op.id, name: entry.name, field, value });
      }); label.append(input); panel.append(label);
    }
    if (!attrs.length) panel.append(el("p", "se-hint", "No editable scalar attributes were exposed by this source. Weight-defined dimensions remain bound to their tensors."));
    panel.append(el("h4", "", "Outputs"));
    for (const p of outputsOf(op, idx)) { const v = idx.values.get(p.value_ref); panel.append(el("p", "se-output-contract", `out ${p.position}: ${contract(v)}`)); }
    for (const graph of draft.program.programs) graph.output_value_refs.forEach((ref, position) => {
      if (idx.values.get(ref)?.region_ref !== region) return;
      const label = el("label", "se-field", `Model output ${position}`), select = el("select");
      select.setAttribute("aria-label", `Model output ${position} binding`);
      for (const v of draft.program.values.filter(v => v.region_ref === region)) { const option = el("option", "", v.name || v.id); option.value = v.id; select.append(option); }
      select.value = ref; select.addEventListener("change", () => command({ type: "model_output", program_ref: graph.id, position, value_ref: select.value }));
      label.append(select); panel.append(label);
    });
    const ownIssues = draft.issues.filter(row => row.subject_ref === op.id);
    for (const issue of ownIssues) panel.append(el("p", `se-issue ${issue.severity}`, issue.message));
    return panel;
  }
  function textPanel() {
    const panel = el("section", "se-text-panel"), buffer = codeState();
    panel.append(el("h4", "", "Structure code · Mermaid"), el("p", "se-hint", "Insert or paste a layer, connect its ports, then apply. Ctrl / ⌘ + Enter to sync."));
    const status = el("p", "se-sync-status"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
    const text = el("textarea", "se-code-input"); text.value = buffer.text; text.spellcheck = false; text.wrap = "off"; text.setAttribute("aria-label", "Mermaid structure code");
    const editor = el("div", "se-code-editor"), gutter = el("pre", "se-code-lines"); gutter.setAttribute("aria-hidden", "true"); editor.append(gutter, text);
    const stats = el("span");
    const lineNumbers = () => { const count = text.value.split('\n').length; gutter.textContent = Array.from({ length: count }, (_, i) => i + 1).join('\n'); stats.textContent = `${count} lines · Mermaid`; gutter.scrollTop = text.scrollTop; };
    text.addEventListener('scroll', () => { gutter.scrollTop = text.scrollTop; }); lineNumbers();
    const toGraph = button("Sync to graph", () => { textBuffers.delete(region); message = ""; render(); resetCodeView(); });
    toGraph.title = "Graph → Code: replace this code draft with the current graph";
    const toCode = button("Sync to code", () => { try {
      if (buffer.revision !== session.revision) throw new Error("Graph changed while code was being edited. Download your code to keep it, then use Sync to graph and reapply your edits to the current graph.");
      const before = draft.program.operations.length;
      session.apply(parseStructureMermaid(buffer.text, draft.program, region), { rejectErrors: true });
      if (session.draft.program.operations.length > before) { positions.clear(); selected = session.draft.program.operations.at(-1).id; }
      textBuffers.delete(region); message = ""; fromValue = null; render(); resetCodeView();
    } catch (error) { fail(error); } }, "se-primary");
    toCode.title = "Code → Graph: validate and apply this code as one undoable edit";
    const update = () => {
      const dirty = buffer.text !== buffer.baseText, conflict = dirty && buffer.revision !== session.revision;
      status.textContent = conflict ? "Conflict · graph changed; code preserved" : dirty ? "Code edits pending · graph unchanged" : "Live · graph → code";
      status.classList.toggle("is-conflict", conflict); toCode.disabled = !dirty;
    };
    text.addEventListener("input", () => {
      buffer.text = text.value;
      if (buffer.text === buffer.baseText && buffer.revision !== session.revision) {
        buffer.text = buffer.baseText = structureMermaid(draft.program, region).text; buffer.revision = session.revision; text.value = buffer.text;
      }
      lineNumbers(); update();
    }); update();
    text.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); if (!toCode.disabled) toCode.click(); } });
    const actions = el("div", "se-sync-actions");
    const graphAction = el("div"), codeAction = el("div");
    graphAction.append(toGraph, el("small", "", "Graph → Code")); codeAction.append(toCode, el("small", "", "Code → Graph"));
    actions.append(graphAction, codeAction);
    const footer = el("div", "se-code-footer"); footer.append(stats, button("Download Mermaid", () => downloadText("deepbom-structure.mmd", buffer.text, "text/plain")));
    const width = el("label", "se-code-width", "Code width"), range = el("input"); range.type = "range"; range.min = "30"; range.max = "65"; range.value = codeShare; range.setAttribute('aria-label', 'Code panel width');
    range.addEventListener('input', () => { codeShare = Number(range.value); host.style.setProperty('--se-code-share', `${codeShare}fr`); host.style.setProperty('--se-graph-share', `${100 - codeShare}fr`); }); width.append(range);
    panel.append(status, actions, editor, footer, width);
    return panel;
  }
  function comparison() {
    const panel = el("section", "se-comparison");
    const summary = el("div", "se-summary");
    for (const [label, value] of [["Layers · before → after", `${draft.metrics.before_operations} → ${draft.metrics.after_operations}`],
      ["Input bindings", `${draft.metrics.before_connections} → ${draft.metrics.after_connections}`],
      ["Costs requiring reassessment", draft.metrics.invalidated_operations], ["Connection errors", draft.issues.filter(i => i.severity === "error").length]]) {
      const metric = el("div"); metric.append(el("span", "", label), el("strong", "", String(value))); summary.append(metric);
    }
    panel.append(summary, el("h4", "", "Before / after"));
    const diff = el("div", "se-diff");
    if (!draft.diff.length) diff.append(el("p", "se-hint", "No structural changes. Start by connecting a port or selecting a layer."));
    for (const row of draft.diff) {
      const item = button(`${row.kind === "added" ? "+" : row.kind === "removed" ? "−" : "~"} ${row.label} · ${row.kind}`, () => { if (idx.operations.has(row.subject_ref)) { selected = row.subject_ref; render(); } }, `se-diff-row ${row.kind}`);
      item.title = row.subject_ref;
      for (const detail of row.details || []) item.append(el("span", "se-diff-detail", detail));
      diff.append(item);
    }
    panel.append(diff);
    const issues = el("details", "se-diagnostics"); issues.append(el("summary", "", `${draft.issues.length} diagnostics · ${draft.status.replaceAll("_", " ")}`));
    for (const row of draft.issues) issues.append(el("p", `se-issue ${row.severity}`, `${row.subject_ref ? `${idx.operations.get(row.subject_ref)?.name || row.subject_ref}: ` : ""}${row.message}`));
    panel.append(issues, el("p", "se-hint", "Draft structure only. Edited native operations need their source-specific calculation rules; stale costs are not reused. Project export contains edits, not model weights or a trained model."));
    const actions = el("div", "se-layer-actions"); actions.append(button("Save candidate", () => { if (!canSave()) return; if (previews.length >= 8) previews.shift(); previews.push({ label: `Candidate ${++candidateNumber}`, project: session.export() }); render(); }));
    for (const item of previews) actions.append(button(item.label, () => replaceProject(item.project))); panel.append(actions);
    return panel;
  }
  function keyboard(event) {
    if (!session) return;
    if (event.key === 'Escape' && expanded) { event.preventDefault(); toggleFocus(); return; }
    if (event.key === 'Tab' && expanded) {
      const focusable = [...host.querySelectorAll('button,input,select,textarea,[tabindex="0"]')].filter(node => !node.disabled && !node.hidden && node.getClientRects().length);
      const first = focusable[0], last = focusable.at(-1);
      if (event.shiftKey && event.target === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && event.target === last) { event.preventDefault(); first?.focus(); }
    }
    if (event.target.closest('input,textarea,select,[contenteditable="true"]')) return;
    if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(event.key.toLowerCase())) {
      event.preventDefault(); if (event.shiftKey || event.key.toLowerCase() === 'y') session.redo(); else session.undo(); message = ""; render(); viewport?.focus({ preventScroll: true });
    }
  }
  return {
    sync(root, model) {
      rememberPanes(); host?.removeEventListener('keydown', keyboard); host = root; host.addEventListener('keydown', keyboard);
      if (!model) { disposeGesture(); resizeObserver?.disconnect(); source = null; session = null; expanded = false; focusSemantics(); textBuffers.clear(); paneOffsets.clear(); host.replaceChildren(el("p", "se-hint", "Run an artifact analysis to open its structure.")); return; }
      if (source !== model.model_ir_sha256) {
        disposeGesture(); positions.clear(); selected = null; region = null; previews = []; candidateNumber = 0; zoom = 1; fromValue = null; textBuffers.clear(); paneOffsets.clear(); layerSearch = "";
        try { session = createStructureSession(model); source = model.model_ir_sha256; artifactInfo = model.artifact; message = ""; } catch (error) { source = null; session = null; host.replaceChildren(el("p", "se-hint", error.message)); return; }
      }
      if (session) render();
    },
    suspend() { disposeGesture(); resizeObserver?.disconnect(); },
    reset() { disposeGesture(); resizeObserver?.disconnect(); host?.removeEventListener('keydown', keyboard); source = null; session = null; expanded = false; if (host) focusSemantics(); positions.clear(); textBuffers.clear(); paneOffsets.clear(); host?.replaceChildren(); },
  };
}
