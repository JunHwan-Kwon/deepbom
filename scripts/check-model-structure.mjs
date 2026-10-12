import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { createStructureSession, structureMermaid, parseStructureMermaid, structureLayerSnippet, STRUCTURE_TEMPLATES } from "../web/lib/model-structure.js";
import { EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";
import { validateModelIr } from "../web/lib/model-ir.js";
import { routeStructureWires, segmentHitsBox } from "../web/lib/structure-wire-routing.js";

const boxes = [0, 200, 400].map(y => ({ x: 80, y: y + 30, width: 240, height: 110 }));
const wireCases = [
  { id: 'trunk', value_ref: 'a', start: { x: 200, y: 149 }, end: { x: 200, y: 221 } },
  { id: 'skip', value_ref: 'b', start: { x: 200, y: 149 }, end: { x: 176, y: 421 } },
  { id: 'skip2', value_ref: 'c', start: { x: 200, y: 149 }, end: { x: 224, y: 421 } },
  { id: 'feedback', value_ref: 'd', start: { x: 200, y: 549 }, end: { x: 200, y: 21 } },
];
const routed = routeStructureWires(wireCases, boxes);
assert.equal(routed[0].points.length, 2, 'Sequential flow is a straight vertical trunk');
assert(routed.slice(1).every(row => row.side && !row.blocked), 'Skip and reverse paths use outside corridors');
for (const row of routed) for (let i = 1; i < row.points.length; i++) {
  assert(row.points[i].x === row.points[i - 1].x || row.points[i].y === row.points[i - 1].y, 'Routes are orthogonal');
  assert(boxes.every(box => !segmentHitsBox(row.points[i - 1], row.points[i], box)), 'A route never cuts through a card');
}
assert.deepEqual(routeStructureWires(wireCases, boxes), routed, 'Wire layout is deterministic');
const staggered = [boxes[0], boxes[1], { ...boxes[2], y: 630 }, { x: 260, y: 150, width: 80, height: 60 }, { x: 0, y: 450, width: 180, height: 80 }];
const [detour] = routeStructureWires([{ id: 'dragged', value_ref: 'x', start: { x: 200, y: 149 }, end: { x: 200, y: 621 } }], staggered);
assert(!detour.blocked, 'Dragged layers can route around staggered obstacles that block simple outside lanes');
assert(detour.points.every((p, i) => !i || staggered.every(b => !segmentHitsBox(detour.points[i - 1], p, b))));
const [overlap] = routeStructureWires([wireCases[0]], [...boxes, { x: 160, y: 140, width: 80, height: 50 }]);
assert(overlap.blocked, 'Overlapping cards are explicitly marked as obstructed');

const out = mkdtempSync(path.join(tmpdir(), "deepbom-structure-"));
function graph(file, i) {
  const output = path.join(out, `${i}.json`);
  const p = spawnSync(process.execPath, ["bin/deepbom.mjs", "graph", file, "--output-format", "json", "-o", output], { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
  assert.equal(p.status, 0, p.stderr);
  return JSON.parse(readFileSync(output)).model_ir;
}
const pte = path.join(out, "add.pte");
writeFileSync(pte, Buffer.from(EXECUTORCH_ADD_PTE_BASE64, "base64"));
const files = [pte, "web/samples/sample_cnn_float.onnx", "web/samples/mobilenet_v2_1.0_224_quant.tflite", "web/samples/MNISTClassifier.mlmodel", "scripts/fixtures/onnx_recursive_scope.onnx"];
for (const [i, file] of files.entries()) {
  const model = graph(file, i), baseline = JSON.stringify(model), s = createStructureSession(model);
  for (const region of model.program.regions) {
    const mmd = structureMermaid(s.draft.program, region.id);
    s.apply(parseStructureMermaid(mmd.text, s.draft.program, region.id));
  }
  assert.equal(s.draft.affected.length, 0, `${file}: text round trip must preserve every binding`);
  assert.equal(s.draft.diff.length, 0);
  for (const edge of model.program.relationships.filter(e => e.kind === "region_ownership" && model.program.regions.some(r => r.id === e.to_ref))) {
    assert(s.draft.program.relationships.some(e => e.id === edge.id), "Nested region relationships survive editing");
    if (model.program.operations.some(op => op.id === edge.from_ref)) {
      assert.throws(() => s.apply({ type: "duplicate", operation_ref: edge.from_ref }), /dedicated rule/);
      assert.throws(() => s.apply({ type: "delete", operation_ref: edge.from_ref }), /dedicated rule/);
      assert.throws(() => structureLayerSnippet(s.draft.program, model.program.operations.find(op => op.id === edge.from_ref).region_ref, { operation_ref: edge.from_ref }), /dedicated rule/);
    }
  }
  const op = model.program.operations.find(op => op.input_port_refs.length);
  const foreign = model.program.values.find(v => v.region_ref !== op.region_ref);
  if (foreign) assert.throws(() => s.apply({ type: "connect", port_ref: op.input_port_refs[0], value_ref: foreign.id }), /cross graph regions/);
  s.apply({ type: "add", layer: "Identity", region_ref: op.region_ref, insert_at: op.input_port_refs[0] });
  const added = s.draft.changes.at(-1).created_ref;
  assert.equal(s.draft.metrics.after_operations, model.program.operations.length + 1);
  assert(s.draft.affected.includes(op.id));
  assert.equal(s.draft.program.operations.find(x => x.id === op.id).metrics.macs, null, "Edited native cost cannot remain a stale number");
  const saved = s.export();
  s.reset(); assert.equal(s.draft.affected.length, 0); s.load(saved);
  assert.deepEqual(s.export(), saved);
  assert.throws(() => s.load({ ...saved, source_artifact_sha256: "0".repeat(64) }), /different source/);
  const tampered = structuredClone(saved); tampered.commands[0].type = "bad";
  assert.throws(() => s.load(tampered), /digest/);
  const current = s.export(); assert.throws(() => s.apply({ type: "connect", port_ref: "missing", value_ref: "missing" }));
  assert.deepEqual(s.export(), current, "Failed edit must be atomic");
  s.apply({ type: "bypass", operation_ref: added });
  assert.equal(s.draft.diff.length, 0, "Insert then bypass restores original structure and metrics");
  s.undo(); assert(s.draft.affected.length); s.redo(); assert.equal(s.draft.affected.length, 0);
  s.reset(); s.apply({ type: "duplicate", operation_ref: op.id });
  const duplicate = s.draft.program.operations.at(-1);
  for (let n = 0; n < op.input_port_refs.length; n++) {
    assert.equal(s.draft.program.ports.find(p => p.id === duplicate.input_port_refs[n]).value_ref,
      model.program.ports.find(p => p.id === op.input_port_refs[n]).value_ref, "Explicit duplicated state and input bindings");
  }
  // Native copy in code uses the same constructor/state bindings as the canvas.
  s.reset();
  const nativeText = structureMermaid(s.draft.program, op.region_ref).text;
  const snippet = structureLayerSnippet(s.draft.program, op.region_ref, { operation_ref: op.id });
  s.apply(parseStructureMermaid(`${nativeText}\n${snippet}`, s.draft.program, op.region_ref), { rejectErrors: true });
  const codeCopy = s.draft.program.operations.at(-1);
  assert.deepEqual(codeCopy.attributes, duplicate.attributes);
  assert.deepEqual(codeCopy.native_op, duplicate.native_op);
  assert.equal(codeCopy.output_port_refs.length, op.output_port_refs.length);
  assert.deepEqual(codeCopy.input_port_refs.map(ref => s.draft.program.ports.find(p => p.id === ref).value_ref),
    op.input_port_refs.map(ref => model.program.ports.find(p => p.id === ref).value_ref));
  assert(codeCopy.output_port_refs.every(ref => !op.output_port_refs.includes(ref)), "Copied outputs have fresh identities");
  const codeProject = s.export(); s.reset(); s.load(codeProject);
  assert.deepEqual(s.export(), codeProject);
  s.undo(); assert.equal(s.draft.diff.length, 0, "Whole code addition undoes in one step");
  s.redo(); assert.equal(s.draft.program.operations.length, model.program.operations.length + 1);
  assert.equal(JSON.stringify(model), baseline, "Editing never mutates source evidence");
  validateModelIr(model);
}
const model = graph("web/samples/sample_cnn_float.onnx", "onnx");
const s = createStructureSession(model), region = model.program.regions[0].id;
const first = model.program.operations[0], input = model.program.ports.find(p => p.id === first.input_port_refs[0]).value_ref;
const output = model.program.ports.find(p => p.id === first.output_port_refs[0]).value_ref;
s.apply({ type: "model_output", program_ref: model.program.programs[0].id, position: 0, value_ref: output });
assert.equal(s.draft.program.programs[0].output_value_refs[0], output);
assert(s.draft.diff.some(row => row.label === "Model output bindings"));
assert(s.draft.program.values.find(v => v.id === output).roles.includes("graph_output"));
assert.throws(() => s.apply({ type: "delete", operation_ref: first.id }), /model output/);
s.undo(); assert.equal(s.draft.diff.length, 0);
s.apply({ type: "add", layer: "Add", region_ref: region });
let node = s.draft.program.operations.at(-1);
s.apply({ type: "connect", port_ref: node.input_port_refs[0], value_ref: input });
s.apply({ type: "connect", port_ref: node.input_port_refs[1], value_ref: output });
assert(s.draft.issues.some(i => i.severity === "error" && /broadcast/.test(i.message)));
s.apply({ type: "connect", port_ref: node.input_port_refs[1], value_ref: input });
assert(!s.draft.issues.some(i => i.severity === "error"));
assert.deepEqual(s.draft.program.values.find(v => v.id === s.draft.program.ports.find(p => p.id === node.output_port_refs[0]).value_ref).shape,
  model.program.values.find(v => v.id === input).shape);
s.reset(); s.apply({ type: "connect", port_ref: first.input_port_refs[0], value_ref: output });
assert(s.draft.issues.some(i => /cycle/.test(i.message)));
s.undo(); assert.equal(s.draft.issues.length, 0);
s.apply({ type: "connect", port_ref: first.input_port_refs[0], value_ref: null });
assert(s.draft.issues.some(i => /disconnected/.test(i.message)));
s.reset();
const text = structureMermaid(s.draft.program, region).text;
const currentProject = s.export(), currentRevision = s.revision;
assert.equal(parseStructureMermaid(text, s.draft.program, region).type, "structure_connections");
assert.throws(() => s.apply({ type: "connections", bindings: [], output_bindings: [] }), /require the structure_connections/);
const cyclicText = text.replace(/v\d+ -->\|out 0 : in 0\| n0/, "n0 -->|out 0 : in 0| n0");
assert.throws(() => s.apply(parseStructureMermaid(cyclicText, s.draft.program, region), { rejectErrors: true }), /cycle/);
assert.deepEqual(s.export(), currentProject); assert.equal(s.revision, currentRevision);
const outputText = text.replace(/n\d+ -->\|out 0 : output 0\| o0/, "v0 -->|out 0 : output 0| o0");
s.apply(parseStructureMermaid(outputText, s.draft.program, region), { rejectErrors: true });
assert.equal(s.draft.program.programs[0].output_value_refs[0], input);
s.undo(); assert.deepEqual(s.export(), currentProject);
assert.throws(() => parseStructureMermaid(text.replace(/^.*: output 0.*$/m, ""), s.draft.program, region), /every model output/);
assert.throws(() => parseStructureMermaid(text + '\nclick n0 "javascript:alert(1)"', s.draft.program, region), /syntax/);
const edge = text.split("\n").find(line => line.includes("-->"));
assert.throws(() => parseStructureMermaid(`${text}\n${edge}`, s.draft.program, region), /one connection/);
assert.throws(() => s.apply(parseStructureMermaid(text.replace(edge, ""), s.draft.program, region), { rejectErrors: true }), /disconnected/);
assert.deepEqual(s.export(), currentProject);
const outputEdge = text.split("\n").find(line => line.includes(": output "));
assert.throws(() => parseStructureMermaid(`${text}\n${outputEdge}`, s.draft.program, region), /two bindings/);
for (const template of STRUCTURE_TEMPLATES) {
  const extra = `\nnew_node["${template.name}"]:::template_${template.name}\n` +
    Array.from({ length: template.inputs }, (_, position) => `v0 -->|out 0 : in ${position}| new_node`).join("\n");
  const code = text.replace(outputEdge, "  new_node -->|out 0 : output 0| o0") + extra;
  s.apply(parseStructureMermaid(code, s.draft.program, region), { rejectErrors: true });
  const newOutput = s.draft.program.values.find(v => v.id === s.draft.program.programs[0].output_value_refs[0]);
  assert.deepEqual(newOutput.shape, model.program.values.find(v => v.id === input).shape);
  assert.equal(newOutput.dtype, model.program.values.find(v => v.id === input).dtype);
  assert(s.draft.diff.some(row => row.label === "Model output bindings"));
  // A copied template must retain its explicit semantics as well as its ports.
  s.apply({ type: "duplicate", operation_ref: s.draft.program.operations.at(-1).id });
  const copied = s.draft.program.operations.at(-1);
  assert.equal(copied.draft_template, template.name);
  const mmd = structureMermaid(s.draft.program, region).text, unchanged = s.draft.program;
  s.apply(parseStructureMermaid(mmd, s.draft.program, region), { rejectErrors: true });
  assert.deepEqual(s.draft.program, unchanged, "Added and copied template declarations round-trip");
  s.reset();
}
const chain = text.replace(outputEdge, "two -->|out 0 : output 0| o0") + '\none["ReLU"]:::template_ReLU\ntwo["Identity"]:::template_Identity\nv0 -->|out 0 : in 0| one\none -->|out 0 : in 0| two';
s.apply(parseStructureMermaid(chain, s.draft.program, region), { rejectErrors: true });
assert.equal(s.draft.program.operations.length, model.program.operations.length + 2);
const chainSaved = s.export(), chainGraph = s.draft.program;
s.reset(); s.load(chainSaved); assert.deepEqual(s.draft.program, chainGraph);
s.undo(); assert.equal(s.draft.diff.length, 0); s.redo(); assert.deepEqual(s.draft.program, chainGraph); s.reset();
const rejected = s.export(), revision = s.revision;
for (const [extra, error] of [
  ['new_node["Conv"]:::template_Conv', /Unsupported layer/],
  ['new_node["ReLU"]', /template_Name/],
  ['n0["ReLU"]:::template_ReLU', /Duplicate node ID/],
  ['x["ReLU"]:::template_ReLU\nx["ReLU"]:::template_ReLU', /Duplicate node ID/],
  ['x["Conv"]:::copy_missing', /existing graph layer/],
  ['x["ReLU"]:::template_ReLU', /disconnected/],
  ['x["ReLU"]:::template_ReLU\nv0 -->|out 1 : in 0| x', /unknown node or port/],
  ['x["Add"]:::template_Add\nv0 -->|out 0 : in 0| x\nn0 -->|out 0 : in 1| x', /broadcast/],
  ['x["ReLU"]:::template_ReLU\nx -->|out 0 : in 0| x', /cycle/],
  [Array.from({ length: 101 }, (_, i) => `x${i}["ReLU"]:::template_ReLU`).join("\n"), /at most 100/],
]) {
  assert.throws(() => s.apply(parseStructureMermaid(`${text}\n${extra}`, s.draft.program, region), { rejectErrors: true }), error);
  assert.deepEqual(s.export(), rejected); assert.equal(s.revision, revision);
}
const oneSnippet = structureLayerSnippet(s.draft.program, region, { layer: "ReLU", source_ref: first.id });
const twoSnippet = structureLayerSnippet(s.draft.program, region, { layer: "ReLU", source_ref: first.id, text: oneSnippet });
assert.match(oneSnippet, /relu_1\[/); assert.match(twoSnippet, /relu_2\[/);
s.apply(parseStructureMermaid(`${text}\n${oneSnippet}\n${twoSnippet}`, s.draft.program, region), { rejectErrors: true });
s.reset();
s.apply({ type: "connections", bindings: [{ port_ref: first.input_port_refs[0], value_ref: input }] });
assert.equal(s.draft.diff.length, 0, "Legacy connection-only projects still replay"); s.undo();
const attr = first.attributes.entries.find(e => e.name === "pads");
assert(attr);
s.apply({ type: "attribute", operation_ref: first.id, name: "pads", field: "int_values", value: [0, 0, 0, 0] });
assert.equal(s.draft.program.operations[0].metrics.macs, null);
assert.deepEqual(s.draft.program.operations[0].attributes.entries.find(e => e.name === "pads").int_values_exact_decimal, ["0", "0", "0", "0"]);
const storage = graph("web/samples/tinymqa1m.Q4_0.gguf", "gguf");
assert.throws(() => createStructureSession(storage), /no serialized/);
console.log("Structure editor core passed: ONNX/TFLite/CoreML/ExecuTorch/nested regions, port edits, shared-state bindings, immutable source, invalidation, broadcast, cycles, undo, code additions/copies, rejected incomplete/invalid code and source-bound project replay.");
