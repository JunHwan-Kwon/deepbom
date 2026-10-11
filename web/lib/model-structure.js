import { validateModelIr, validateModelProgramLinks } from "./model-ir.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { broadcastStaticShapes, logicalBytesForShape } from "./tensor-size.js";

// An editing session is a source-bound proposal, never a rewritten audit.
export const STRUCTURE_PROJECT_SCHEMA = "deepbom.structure_project.v1";
const copy = value => structuredClone(value);
const MAX_EDITS = 1000;
const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
export const STRUCTURE_TEMPLATES = Object.freeze([
  Object.freeze({ name: "Identity", inputs: 1, outputs: 1, description: "Pass-through · preserves shape and dtype" }),
  Object.freeze({ name: "ReLU", inputs: 1, outputs: 1, description: "Activation · preserves shape and dtype" }),
  Object.freeze({ name: "Add", inputs: 2, outputs: 1, description: "Elementwise sum · equal dtype, broadcast shapes" }),
]);
export const STRUCTURE_LAYERS = Object.freeze(STRUCTURE_TEMPLATES.map(row => row.name));

export function createStructureSession(model) {
  const baseline = copy(validateModelIr(model));
  if (baseline.program.status !== "serialized" || !baseline.program.operations.length) {
    throw new Error("This artifact has no serialized operation graph to edit.");
  }
  let commands = [], cursor = 0, revision = 0;
  let current = buildStructureDraft(baseline, []);
  const refresh = () => { current = buildStructureDraft(baseline, commands.slice(0, cursor)); revision++; };
  return {
    get model() { return copy(baseline); },
    get draft() { return copy(current); },
    get revision() { return revision; },
    get canUndo() { return cursor > 0; },
    get canRedo() { return cursor < commands.length; },
    apply(command, { rejectErrors = false } = {}) {
      if (cursor >= MAX_EDITS) throw new Error("Save this project before exceeding 1,000 edits.");
      const next = [...commands.slice(0, cursor), copy(command)];
      const draft = buildStructureDraft(baseline, next); // transactional, including failed text imports
      if (rejectErrors && draft.issues.some(issue => issue.severity === "error")) {
        throw new Error(`Code was not applied: ${draft.issues.find(issue => issue.severity === "error").message}`);
      }
      commands = next; cursor = next.length; current = draft; revision++;
    },
    undo() { if (cursor) { cursor--; refresh(); } },
    redo() { if (cursor < commands.length) { cursor++; refresh(); } },
    reset() { commands = []; cursor = 0; refresh(); },
    export() {
      const body = { schema: STRUCTURE_PROJECT_SCHEMA, source_model_ir_sha256: baseline.model_ir_sha256,
        source_artifact_sha256: baseline.artifact.sha256, commands: copy(commands.slice(0, cursor)),
        interpretation: "Structural proposal only; source weights, runtime performance and task quality are not changed or established." };
      return { ...body, project_sha256: sha256TextHex(canonicalJson(body)) };
    },
    load(project) {
      const { project_sha256, ...body } = project || {};
      if (body.schema !== STRUCTURE_PROJECT_SCHEMA || body.source_model_ir_sha256 !== baseline.model_ir_sha256
        || body.source_artifact_sha256 !== baseline.artifact.sha256) throw new Error("Project belongs to a different source Model IR. Open its original model first.");
      if (sha256TextHex(canonicalJson(body)) !== project_sha256) throw new Error("Project digest does not match its contents.");
      if (!Array.isArray(body.commands) || body.commands.length > MAX_EDITS) throw new Error("Project edit limit exceeded.");
      const draft = buildStructureDraft(baseline, body.commands);
      commands = copy(body.commands); cursor = commands.length; current = draft; revision++;
    },
  };
}

export function buildStructureDraft(model, commands) {
  const program = copy(model.program);
  const touched = new Set(), changes = [];
  const operation = id => required(program.operations, id, "Layer");
  const port = id => required(program.ports, id, "Port");
  const value = id => required(program.values, id, "Tensor");
  const bind = (portRef, valueRef) => {
    const p = port(portRef), op = operation(p.operation_ref);
    if (p.direction !== "input") throw new Error("Choose an input port as the connection target.");
    if (valueRef !== null && value(valueRef).region_ref !== op.region_ref) throw new Error("Connections cannot cross graph regions.");
    if (p.value_ref === valueRef) return;
    p.value_ref = valueRef; p.evidence_class = "DECLARED_UNVERIFIED"; touched.add(op.id);
  };
  const bindOutput = command => {
    const graph = required(program.programs, command.program_ref, "Program");
    if (!Number.isInteger(command.position) || command.position < 0 || command.position >= graph.output_value_refs.length) throw new Error("Unknown model output position.");
    const previous = value(graph.output_value_refs[command.position]), next = value(command.value_ref);
    if (previous.region_ref !== next.region_ref) throw new Error("Model outputs cannot cross graph regions.");
    graph.output_value_refs[command.position] = next.id;
  };
  const applyEdit = (command, step) => {
    if (!command || typeof command !== "object") throw new Error("Invalid structure edit.");
    const id = `draft:layer:${step}`;
    switch (command.type) {
      case "connect": bind(command.port_ref, command.value_ref); break;
      case "model_output": bindOutput(command); break;
      case "structure_code": {
        const edits = parseStructureCode(command.text, program, command.region_ref, step);
        edits.forEach((edit, index) => applyEdit(edit, `${step}.${index}`));
        break;
      }
      case "connections": case "structure_connections": {
        if (!Array.isArray(command.bindings) || command.bindings.length > program.ports.length) throw new Error("Invalid connection list.");
        const seen = new Set();
        for (const row of command.bindings) {
          if (seen.has(row.port_ref)) throw new Error("An input port cannot have two bindings.");
          seen.add(row.port_ref); bind(row.port_ref, row.value_ref);
        }
        if (command.type === "connections" && own(command, "output_bindings")) throw new Error("Model output edits require the structure_connections command.");
        const outputs = command.output_bindings ?? [];
        if (!Array.isArray(outputs) || outputs.length > program.programs.reduce((sum, graph) => sum + graph.output_value_refs.length, 0)) throw new Error("Invalid model output bindings.");
        const outputKeys = new Set();
        for (const row of outputs) {
          const key = JSON.stringify([row.program_ref, row.position]);
          if (outputKeys.has(key)) throw new Error("A model output cannot have two bindings.");
          outputKeys.add(key); bindOutput(row);
        }
        break;
      }
      case "attribute": {
        const op = operation(command.operation_ref);
        const entry = op.attributes?.entries?.find(row => row.name === command.name);
        if (!entry || !["int_value", "float_value", "string_value", "int_values", "float_values"].includes(command.field)
          || !own(entry, command.field)) throw new Error("This native attribute is not editable.");
        const values = command.field.endsWith("values") ? command.value : [command.value];
        if (!Array.isArray(values) || values.length > 256 || values.some(item => command.field.includes("int")
          ? !Number.isSafeInteger(item) : command.field.includes("float") ? !Number.isFinite(item) : typeof item !== "string" || item.length > 4096)) {
          throw new Error("Attribute values must have the declared type and a bounded length.");
        }
        entry[command.field] = copy(command.value);
        if (command.field === "int_value" && own(entry, "int_value_exact_decimal")) entry.int_value_exact_decimal = String(command.value);
        if (command.field === "int_values" && own(entry, "int_values_exact_decimal")) entry.int_values_exact_decimal = command.value.map(String);
        if (command.field === "float_value" && own(entry, "float_value_text")) entry.float_value_text = String(command.value);
        if (command.field === "float_values" && own(entry, "float_values_text")) entry.float_values_text = command.value.map(String);
        touched.add(op.id); break;
      }
      case "rename": {
        if (typeof command.name !== "string" || !command.name.trim() || command.name.length > 200) throw new Error("Choose a layer name of 1–200 characters.");
        operation(command.operation_ref).name = command.name; break;
      }
      case "add": case "duplicate": {
        const op = appendStructureLayer(program, command, id);
        if (command.insert_at) {
          if (op.input_port_refs.length !== 1 || op.output_port_refs.length !== 1) throw new Error("Insertion requires one input and one output; connect other templates explicitly.");
          const target = port(command.insert_at);
          bind(op.input_port_refs[0], target.value_ref); bind(target.id, port(op.output_port_refs[0]).value_ref);
        }
        break;
      }
      case "delete": case "bypass": {
        const op = operation(command.operation_ref);
        if (hasProtectedLinks(program, op.id)) {
          throw new Error("This layer has control, state or nested-region relationships. Deletion needs a dedicated rule.");
        }
        const outputs = op.output_port_refs.map(ref => port(ref).value_ref);
        if (command.type === "bypass" && (op.input_port_refs.length !== 1 || outputs.length !== 1 || !port(op.input_port_refs[0]).value_ref)) {
          throw new Error("Bypass needs exactly one connected input and one output.");
        }
        const replacement = command.type === "bypass" ? port(op.input_port_refs[0]).value_ref : null;
        if (!replacement && program.programs.some(row => row.output_value_refs.some(ref => outputs.includes(ref)))) {
          throw new Error("Reconnect the model output before deleting its producer, or bypass a unary layer.");
        }
        for (const p of program.ports) if (p.direction === "input" && outputs.includes(p.value_ref) && p.operation_ref !== op.id) bind(p.id, replacement);
        for (const row of program.programs) row.output_value_refs = row.output_value_refs.map(ref => outputs.includes(ref) ? replacement : ref);
        program.ports = program.ports.filter(p => p.operation_ref !== op.id);
        program.operations = program.operations.filter(row => row.id !== op.id);
        program.values = program.values.filter(row => !outputs.includes(row.id));
        program.blocks.forEach(block => { block.member_refs = block.member_refs.filter(ref => ref !== op.id); });
        program.blocks = program.blocks.filter(block => block.member_refs.length);
        touched.delete(op.id); break;
      }
      default: throw new Error("Unknown structure edit type.");
    }
    changes.push({ ...copy(command), created_ref: ["add", "duplicate"].includes(command.type) ? id : null });
  };
  commands.forEach(applyEdit);
  rebuildRelationships(program);
  const boundaryOutputs = new Set(program.programs.flatMap(row => row.output_value_refs));
  const originalOutputs = new Set(model.program.programs.flatMap(row => row.output_value_refs));
  for (const v of program.values) {
    if (originalOutputs.has(v.id) && !boundaryOutputs.has(v.id)) v.roles = (v.roles || []).filter(role => role !== "graph_output");
    if (boundaryOutputs.has(v.id) && !v.roles?.includes("graph_output")) v.roles = [...(v.roles || []), "graph_output"];
  }
  const indexes = indexStructure(program), issues = [];
  // Invalidation follows the final structure, not abandoned intermediate edits.
  touched.clear();
  const sourceIndex = indexStructure(model.program);
  for (const op of program.operations) {
    const before = sourceIndex.operations.get(op.id);
    if (!before || canonicalJson(before.attributes || {}) !== canonicalJson(op.attributes || {})
      || op.input_port_refs.some(ref => sourceIndex.ports.get(ref)?.value_ref !== indexes.ports.get(ref)?.value_ref)) touched.add(op.id);
  }
  const affected = new Set(touched);
  for (let change = true; change;) {
    change = false;
    for (const row of program.relationships) if (row.kind === "data_dependency" && affected.has(row.from_ref) && !affected.has(row.to_ref)) {
      affected.add(row.to_ref); change = true;
    }
  }
  const order = topologicalOrder(program);
  if (order.length !== program.operations.length) issues.push({ severity: "error", subject_ref: null, message: "A data-flow cycle is present. Remove the feedback connection." });
  for (const p of program.ports) if (p.direction === "input" && !p.value_ref) issues.push({ severity: "error", subject_ref: p.operation_ref, message: `Input ${p.position} is disconnected.` });
  for (const ref of affected) {
    const op = indexes.operations.get(ref);
    op.metrics = { macs: null, logical_io_bytes: null, mac_assessment_status: "not_assessed_after_edit" };
    delete op.metric_contracts; op.evidence_class = "DECLARED_UNVERIFIED";
    for (const p of outputsOf(op, indexes)) {
      const v = indexes.values.get(p.value_ref);
      Object.assign(v, { shape: null, dtype: null, shape_signature: null, type_contract: null, logical_byte_length: null, evidence_class: "DECLARED_UNVERIFIED" });
    }
  }
  for (const op of order) {
    if (!affected.has(op.id)) continue;
    const inputs = inputsOf(op, indexes).map(p => indexes.values.get(p.value_ref));
    const inferred = inferTemplate(op, inputs);
    if (inferred?.error) issues.push({ severity: "error", subject_ref: op.id, message: inferred.error });
    for (const p of outputsOf(op, indexes)) {
      if (inferred && !inferred.error) {
        const v = indexes.values.get(p.value_ref);
        Object.assign(v, inferred, { logical_byte_length: logicalBytesForShape(inferred.dtype, inferred.shape) });
      }
    }
    if (!inferred || inferred.error) issues.push({ severity: "warning", subject_ref: op.id, message: "Output contract and cost need a source-specific rule after this edit." });
  }
  if (!program.ports.some(p => !p.value_ref)) validateModelProgramLinks(program);
  const before = new Map(model.program.operations.map(op => [op.id, op]));
  const diff = [];
  for (const op of program.operations) {
    const old = before.get(op.id);
    if (!old) diff.push({ kind: "added", subject_ref: op.id, label: op.name });
    else if (canonicalJson(old.attributes || {}) !== canonicalJson(op.attributes || {}) || old.name !== op.name
      || old.input_port_refs.some(ref => model.program.ports.find(p => p.id === ref)?.value_ref !== indexes.ports.get(ref)?.value_ref)) {
      const details = [];
      if (old.name !== op.name) details.push(`Name: ${old.name} → ${op.name}`);
      for (const ref of op.input_port_refs) {
        const from = sourceIndex.ports.get(ref), to = indexes.ports.get(ref);
        if (from?.value_ref !== to.value_ref) details.push(`Input ${to.position}: ${from?.value_ref || "disconnected"} → ${to.value_ref || "disconnected"}`);
      }
      for (const entry of op.attributes?.entries || []) {
        const previous = old.attributes?.entries?.find(row => row.name === entry.name);
        for (const key of ["int_value", "float_value", "string_value", "int_values", "float_values"]) {
          if (own(entry, key) && canonicalJson(previous?.[key] ?? null) !== canonicalJson(entry[key])) details.push(`${entry.name}: ${JSON.stringify(previous?.[key])} → ${JSON.stringify(entry[key])}`);
        }
      }
      diff.push({ kind: "changed", subject_ref: op.id, label: op.name, details });
    } else if (affected.has(op.id)) diff.push({ kind: "affected", subject_ref: op.id, label: op.name, details: ["Upstream structure changed; derived contracts and costs require reassessment."] });
  }
  for (const op of model.program.operations) if (!indexes.operations.has(op.id)) diff.push({ kind: "removed", subject_ref: op.id, label: op.name });
  for (const graph of program.programs) if (canonicalJson(graph.output_value_refs) !== canonicalJson(model.program.programs.find(row => row.id === graph.id).output_value_refs)) {
    graph.evidence_class = "DECLARED_UNVERIFIED";
    diff.push({ kind: "changed", subject_ref: graph.id, label: "Model output bindings", details: graph.output_value_refs.map((ref, i) => `Output ${i}: ${model.program.programs.find(row => row.id === graph.id).output_value_refs[i]} → ${ref}`) });
  }
  return { program, issues, changes, diff, affected: [...affected], source_model_ir_sha256: model.model_ir_sha256,
    status: issues.some(row => row.severity === "error") ? "needs_correction" : diff.length ? "draft_checks_partial" : "source_structure",
    evidence_class: "DECLARED_UNVERIFIED", measurements: { latency: null, accuracy: null },
    metrics: { before_operations: before.size, after_operations: program.operations.length,
      before_connections: model.program.ports.filter(p => p.direction === "input" && p.value_ref).length,
      after_connections: program.ports.filter(p => p.direction === "input" && p.value_ref).length,
      invalidated_operations: affected.size } };
}

function inferTemplate(op, inputs) {
  // Only explicit editor templates have these semantics. Never infer native
  // operator behavior from a display name (custom domains may reuse that name).
  if (!op.draft_template || inputs.some(v => !v || !Array.isArray(v.shape) || !v.dtype)) return null;
  if (["Identity", "ReLU"].includes(op.draft_template)) return { shape: copy(inputs[0].shape), dtype: inputs[0].dtype };
  if (op.draft_template === "Add") {
    if (inputs[0].dtype !== inputs[1].dtype) return { error: "Add inputs have different dtypes." };
    if (inputs.some(v => v.shape.some(d => !Number.isSafeInteger(d) || d < 0))) return null;
    const shape = broadcastStaticShapes(inputs[0].shape, inputs[1].shape);
    return shape ? { shape, dtype: inputs[0].dtype } : { error: "Add input shapes cannot broadcast." };
  }
  return null;
}

export function indexStructure(program) {
  return Object.fromEntries(["operations", "ports", "values"].map(key => [key, new Map(program[key].map(row => [row.id, row]))]));
}
export function inputsOf(op, indexes) { return op.input_port_refs.map(ref => indexes.ports.get(ref)).sort((a, b) => a.position - b.position); }
export function outputsOf(op, indexes) { return op.output_port_refs.map(ref => indexes.ports.get(ref)).sort((a, b) => a.position - b.position); }
function required(rows, id, label) { const row = rows.find(item => item.id === id); if (!row) throw new Error(`${label} is not part of this graph.`); return row; }
function hasProtectedLinks(program, id) {
  return program.relationships.some(row => row.kind === "region_ownership" ? row.from_ref === id
    : !["data_dependency", "storage_binding"].includes(row.kind) && [row.from_ref, row.to_ref].includes(id));
}

function appendStructureLayer(program, command, id) {
  const source = command.type === "duplicate" ? required(program.operations, command.operation_ref, "Layer") : null;
  if (source && hasProtectedLinks(program, source.id)) throw new Error("Duplicating control, state or nested-region ownership needs a dedicated rule.");
  const region = source?.region_ref || command.region_ref;
  required(program.regions, region, "Region");
  const template = STRUCTURE_TEMPLATES.find(row => row.name === command.layer);
  if (!source && !template) throw new Error(`Unsupported layer template. Choose ${STRUCTURE_LAYERS.join(", ")}, or copy an imported layer.`);
  const op = source ? copy(source) : { kind: "operation", native_op: { domain: "deepbom.structure", name: template.name, version: 1 },
    name: template.name, attributes: { entries: [] }, semantic_family: null };
  Object.assign(op, { id, region_ref: region, source_order: null, native_index: null, native_subject_ref: null,
    native_source: null, evidence_class: "DECLARED_UNVERIFIED", input_port_refs: [], output_port_refs: [], draft_template: source ? source.draft_template || null : template.name });
  const ports = refs => refs.map(ref => required(program.ports, ref, "Port"));
  const inputs = source ? ports(source.input_port_refs) : Array.from({ length: template.inputs }, (_, position) => ({ position, value_ref: null }));
  for (const input of inputs) {
    const p = { id: `${id}:input:${input.position}`, operation_ref: id, direction: "input", position: input.position,
      value_ref: input.value_ref, evidence_class: "DECLARED_UNVERIFIED" };
    program.ports.push(p); op.input_port_refs.push(p.id);
  }
  const outputs = source ? ports(source.output_port_refs) : Array.from({ length: template.outputs }, (_, position) => ({ position }));
  for (const output of outputs) {
    const v = { id: `${id}:value:${output.position}`, region_ref: region, name: `${op.name}:${output.position}`, kind: "tensor",
      shape: null, dtype: null, storage_refs: [], roles: [], evidence_class: "DECLARED_UNVERIFIED" };
    const p = { id: `${id}:output:${output.position}`, operation_ref: id, direction: "output", position: output.position,
      value_ref: v.id, evidence_class: "DECLARED_UNVERIFIED" };
    program.values.push(v); program.ports.push(p); op.output_port_refs.push(p.id);
  }
  program.operations.push(op); program.blocks.push({ id: `${id}:block`, region_ref: region, member_refs: [id] });
  return op;
}

function rebuildRelationships(program) {
  const operations = new Set(program.operations.map(row => row.id));
  const producers = new Map(program.ports.filter(p => p.direction === "output").map(p => [p.value_ref, p.operation_ref]));
  const regions = new Set(program.regions.map(row => row.id));
  program.relationships = program.relationships.filter(row => row.kind !== "data_dependency"
    && !(row.kind === "region_ownership" && operations.has(row.to_ref))
    && (row.kind !== "storage_binding" || program.values.some(v => v.id === row.from_ref))
    && (row.kind !== "region_ownership" || regions.has(row.to_ref) || operations.has(row.to_ref)));
  for (const op of program.operations) program.relationships.push({ id: `draft:owner:${op.id}`, kind: "region_ownership", from_ref: op.region_ref, to_ref: op.id });
  for (const p of program.ports) if (p.direction === "input" && producers.has(p.value_ref) && operations.has(p.operation_ref)) {
    program.relationships.push({ id: `draft:edge:${p.id}`, kind: "data_dependency", from_ref: producers.get(p.value_ref), to_ref: p.operation_ref, value_ref: p.value_ref });
  }
}

export function topologicalOrder(program) {
  const indexes = indexStructure(program), counts = new Map(program.operations.map(op => [op.id, 0]));
  const next = new Map(program.operations.map(op => [op.id, []]));
  for (const edge of program.relationships.filter(row => row.kind === "data_dependency")) {
    if (!counts.has(edge.from_ref) || !counts.has(edge.to_ref)) continue;
    counts.set(edge.to_ref, counts.get(edge.to_ref) + 1); next.get(edge.from_ref).push(edge.to_ref);
  }
  const queue = program.operations.filter(op => counts.get(op.id) === 0).map(op => op.id), order = [];
  for (let i = 0; i < queue.length; i++) {
    const id = queue[i]; order.push(indexes.operations.get(id));
    for (const target of next.get(id)) { counts.set(target, counts.get(target) - 1); if (!counts.get(target)) queue.push(target); }
  }
  return order;
}

// A bounded Mermaid profile: template_* creates a supported template;
// copy_nN duplicates an existing layer through the same graph edit rules.
// Native attributes/state stay in the source-bound project. No eval or HTML.
export function structureMermaid(program, regionRef) {
  const ops = program.operations.filter(op => op.region_ref === regionRef);
  const idx = indexStructure(program), names = new Map(ops.map((op, i) => [op.id, `n${i}`]));
  const producer = new Map(program.ports.filter(p => p.direction === "output").map(p => [p.value_ref, p]));
  const external = new Map(), outputs = new Map();
  for (const op of ops) for (const p of inputsOf(op, idx)) if (p.value_ref && !producer.has(p.value_ref) && !external.has(p.value_ref)) external.set(p.value_ref, `v${external.size}`);
  for (const graph of program.programs) {
    for (const ref of graph.input_value_refs) if (idx.values.get(ref)?.region_ref === regionRef && !producer.has(ref) && !external.has(ref)) external.set(ref, `v${external.size}`);
    graph.output_value_refs.forEach((ref, position) => {
      if (idx.values.get(ref)?.region_ref !== regionRef) return;
      outputs.set(`o${outputs.size}`, { program_ref: graph.id, position, value_ref: ref });
      if (!producer.has(ref) && !external.has(ref)) external.set(ref, `v${external.size}`);
    });
  }
  const lines = ["flowchart LR", "  %% Keep existing declarations. Add unique IDs with :::template_ReLU or :::copy_n0; connect every input."];
  for (const [id, name] of external) lines.push(`  ${name}["${diagramLabel(idx.values.get(id)?.name || "Input / state")}"]`);
  for (const op of ops) lines.push(`  ${names.get(op.id)}["${diagramLabel(op.name || op.native_op.name)}"]${op.draft_template ? `:::template_${op.draft_template}` : ""}`);
  for (const [alias, output] of outputs) lines.push(`  ${alias}["Model output ${output.position}"]`);
  for (const op of ops) for (const p of inputsOf(op, idx)) {
    if (!p.value_ref) continue;
    const source = producer.get(p.value_ref), from = source ? names.get(source.operation_ref) : external.get(p.value_ref);
    if (from) lines.push(`  ${from} -->|out ${source?.position ?? 0} : in ${p.position}| ${names.get(op.id)}`);
  }
  for (const [alias, output] of outputs) {
    const source = producer.get(output.value_ref), from = source ? names.get(source.operation_ref) : external.get(output.value_ref);
    if (from) lines.push(`  ${from} -->|out ${source?.position ?? 0} : output ${output.position}| ${alias}`);
  }
  return { text: lines.join("\n"), names, external, outputs };
}
function diagramLabel(value) { return String(value).replace(/[^\p{L}\p{N} _.:/()-]/gu, " ").slice(0, 100); }
export function parseStructureMermaid(text, program, regionRef) {
  const edits = parseStructureCode(text, program, regionRef, "preview");
  // Fail closed on older readers, which do not understand node creation in code.
  return edits.length === 1 ? edits[0] : { type: "structure_code", text, region_ref: regionRef };
}
function parseStructureCode(text, original, regionRef, step) {
  if (typeof text !== "string" || text.length > 500000) throw new Error("Diagram text exceeds the editing limit.");
  const program = copy(original);
  required(program.regions, regionRef, "Region");
  const reference = structureMermaid(program, regionRef), idx = indexStructure(program);
  const nodes = new Map([...reference.names].map(([id, alias]) => [alias, idx.operations.get(id)]));
  const external = new Map([...reference.external].map(([id, alias]) => [alias, id]));
  const declared = reference.text.split("\n").filter(line => /^\s*\w+\[/.test(line)).map(line => line.trim());
  const seenDeclarations = new Set(), bindings = new Map(), outputBindings = new Map(), additions = [], edges = [];
  const aliases = new Set([...nodes.keys(), ...external.keys(), ...reference.outputs.keys()]);
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(line => line && !line.startsWith("%%"));
  if (lines.shift() !== "flowchart LR") throw new Error("Keep the generated flowchart LR declaration.");
  for (const line of lines) {
    if (declared.includes(line)) { if (seenDeclarations.has(line)) throw new Error("Duplicate node declaration."); seenDeclarations.add(line); continue; }
    const declaration = /^(\w+)\["([\p{L}\p{N} _.:/()-]{1,100})"\]:::(template|copy)_(\w+)$/u.exec(line);
    if (declaration) {
      const [, alias, label, kind, target] = declaration;
      if (aliases.has(alias)) throw new Error("Duplicate node ID or changed existing declaration. Give the copy a new ID.");
      if (additions.length >= 100) throw new Error("Add at most 100 layers per code sync.");
      const source = kind === "copy" && nodes.get(target);
      if (kind === "copy" && (!source || !reference.names.has(source.id))) throw new Error("Copy an existing graph layer, using its current nN ID.");
      if (kind === "template" && label !== target) throw new Error("New template labels must match their template name. Rename the layer in the inspector after adding it.");
      if (source && label !== diagramLabel(source.name || source.native_op.name)) throw new Error("Keep the copied layer label; rename it in the inspector after adding it.");
      const command = source ? { type: "duplicate", operation_ref: source.id } : { type: "add", layer: target, region_ref: regionRef };
      const op = appendStructureLayer(program, command, `draft:layer:${step}.${additions.length}`);
      additions.push(command); nodes.set(alias, op); aliases.add(alias);
      for (const p of program.ports.filter(p => p.operation_ref === op.id)) idx.ports.set(p.id, p);
      continue;
    }
    if (/^\w+\[/.test(line)) throw new Error("Use a new node ID and a supported :::template_Name or :::copy_nN declaration; keep existing declarations unchanged.");
    edges.push(line);
  }
  for (const line of edges) {
    const match = /^(\w+)\s*-->\|out (\d+) : (in|output) (\d+)\|\s*(\w+)$/.exec(line);
    if (!match) throw new Error("Use the generated node declarations and out N : in N connections. Other Mermaid syntax is not editable here.");
    const [, from, out, kind, input, to] = match, target = nodes.get(to);
    const targetPort = target && inputsOf(target, idx).find(p => p.position === Number(input));
    const source = nodes.get(from), sourceValue = source ? outputsOf(source, idx).find(p => p.position === Number(out))?.value_ref : Number(out) === 0 ? external.get(from) : null;
    if (!sourceValue) throw new Error("A diagram connection references an unknown node or port.");
    if (kind === "output") {
      const target = reference.outputs.get(to);
      if (!target || target.position !== Number(input)) throw new Error("Unknown model output in code.");
      if (outputBindings.has(to)) throw new Error("A model output cannot have two bindings.");
      outputBindings.set(to, { ...target, value_ref: sourceValue }); continue;
    }
    if (!targetPort) throw new Error("A diagram connection references an unknown node or port.");
    if (bindings.has(targetPort.id)) throw new Error("Each input accepts one connection; use Add for a merge.");
    bindings.set(targetPort.id, sourceValue);
  }
  if (seenDeclarations.size !== declared.length) throw new Error("Keep all node declarations; remove layers using the canvas controls.");
  if (outputBindings.size !== reference.outputs.size) throw new Error("Keep every model output connected.");
  // Older project readers reject this new command instead of silently ignoring
  // model-output edits. Their original connections command remains supported.
  return [...additions, { type: "structure_connections", bindings: [...nodes.values()].flatMap(op => inputsOf(op, idx).map(p => ({ port_ref: p.id, value_ref: bindings.get(p.id) ?? null }))), output_bindings: [...outputBindings.values()] }];
}

export function structureLayerSnippet(program, regionRef, { layer, operation_ref, source_ref, text = "" }) {
  const reference = structureMermaid(program, regionRef), idx = indexStructure(program);
  const source = operation_ref ? required(program.operations, operation_ref, "Layer") : null;
  if (source && (!reference.names.has(source.id) || hasProtectedLinks(program, source.id))) throw new Error("This layer cannot be copied here: control, state or nested-region ownership needs a dedicated rule.");
  const template = STRUCTURE_TEMPLATES.find(row => row.name === layer);
  if (!source && !template) throw new Error("Select a supported layer template.");
  const used = new Set([...`${reference.text}\n${text}`.matchAll(/\b(\w+)\s*\[/g)].map(match => match[1]));
  const prefix = source ? "copy" : layer.toLowerCase();
  let number = 1; while (used.has(`${prefix}_${number}`)) number++;
  const alias = `${prefix}_${number}`;
  const lines = [`  ${alias}["${diagramLabel(source ? source.name || source.native_op.name : layer)}"]:::${source ? `copy_${reference.names.get(source.id)}` : `template_${layer}`}`];
  const producers = new Map(program.ports.filter(p => p.direction === "output").map(p => [p.value_ref, p]));
  const selected = idx.operations.get(source_ref), firstOutput = selected && reference.names.has(selected.id) ? outputsOf(selected, idx)[0]?.value_ref : null;
  const ports = source ? inputsOf(source, idx) : Array.from({ length: template.inputs }, (_, position) => ({ position, value_ref: position === 0 ? firstOutput : null }));
  for (const p of ports) {
    const producer = producers.get(p.value_ref), from = producer ? reference.names.get(producer.operation_ref) : reference.external.get(p.value_ref);
    lines.push(from ? `  ${from} -->|out ${producer?.position ?? 0} : in ${p.position}| ${alias}` : `  %% Connect a source to ${alias} input ${p.position} before syncing.`);
  }
  lines.push(`  %% Route ${alias} out 0 to a downstream input or model output to use this branch.`);
  return lines.join("\n");
}
