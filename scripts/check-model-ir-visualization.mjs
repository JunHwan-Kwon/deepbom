import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

import {
  MODEL_IR_VIEW_IDS,
  buildModelIrVisualizationBundle,
  modelIrVisualizationFiles,
} from "../web/lib/model-ir-visualization.js";

const graphModel = runModelIr("web/samples/sample_cnn_float.onnx");
const graphlessModel = runModelIr("web/samples/tinymqa1m.Q4_0.gguf");

for (const [label, modelIr] of [["graph", graphModel], ["graphless", graphlessModel]]) {
  const first = buildModelIrVisualizationBundle(modelIr);
  const second = buildModelIrVisualizationBundle(modelIr);
  assert.deepEqual(first, second, `${label} visualization must be deterministic`);
  assert.deepEqual(first.manifest.selected_views, MODEL_IR_VIEW_IDS, `${label} all levels selected`);
  assert.equal(first.manifest.visual_contract.page_size, "ISO_A4");
  assert.equal(first.manifest.visual_contract.color_contract, "black_white_only");
  assert.equal(first.manifest.visual_contract.minimum_text_size_pt, 7);
  assert.equal(first.manifest.conservation.status, "conserved", `${label} exhaustive conservation`);
  assert.equal(first.manifest.conservation.omitted_count, 0, `${label} exhaustive omissions`);
  assert.equal(first.manifest.conservation.duplicate_count, 0, `${label} exhaustive duplicates`);
  assert.equal(first.manifest.conservation.omitted_relationship_count, 0, `${label} exhaustive relationship omissions`);
  assert.equal(first.manifest.conservation.duplicate_relationship_count, 0, `${label} exhaustive relationship duplicates`);
  assert(first.pages.every((page) => /^[a-f0-9]{64}$/.test(page.sha256)), `${label} page digests`);
  assert(first.pages.every((page) => page.svg.includes("data-subject-ref=")), `${label} source trace attributes`);
  assert(first.pages.every((page) => !/(#[a-f0-9]{3,8})/gi.test(page.svg.replaceAll("#000", "").replaceAll("#fff", ""))), `${label} black-white palette only`);
  assert(first.pages.every((page) => page.svg.includes('width="210mm"') && page.svg.includes('height="297mm"')), `${label} A4 portrait`);
  assert(first.pages.every((page) => page.caption.includes(modelIr.artifact.sha256.slice(0, 12))), `${label} caption identity`);
  const packaged = modelIrVisualizationFiles(modelIr, { views: ["identity-boundary", "exhaustive"] });
  assert(packaged.files.some((file) => file.name === "model-views/manifest.json"), `${label} Word insertion manifest`);
}

const graphExhaustive = buildModelIrVisualizationBundle(graphModel, { views: ["exhaustive"], orientation: "landscape" });
assert.equal(graphExhaustive.manifest.conservation.expected_count, graphModel.program.operations.length);
assert.equal(graphExhaustive.manifest.conservation.rendered_count, graphModel.program.operations.length);
assert(graphExhaustive.pages.some((page) => page.svg.includes("data-relationship-ref=")), "graph exhaustive SVG must render relationship references");
assert(graphExhaustive.pages.every((page) => page.svg.includes('width="297mm"') && page.svg.includes('height="210mm"')));

const blockDetail = buildModelIrVisualizationBundle(graphModel, { views: ["block-detail"] });
const blockDetailRows = blockDetail.pages.flatMap((page) => page.render_model.rows);
const renderedBlockOperations = blockDetailRows.filter((row) => row.kind.startsWith("Operation ")).map((row) => row.subject_ref);
assert.deepEqual([...renderedBlockOperations].sort(), graphModel.program.operations.map((row) => row.id).sort(),
  "block detail must retain every serialized operation exactly once");
assert(blockDetailRows.some((row) => row.kind === "Structural block" && /signature [a-f0-9]{12}/.test(row.detail)),
  "block detail must expose its deterministic structural signature");
for (const block of graphModel.program.blocks.filter((row) => row.member_refs.length + 1 <= 22)) {
  const pageIndexes = new Set(blockDetail.pages.flatMap((page, pageIndex) => page.render_model.rows.some((row) => row.subject_ref === block.id || block.member_refs.includes(row.subject_ref)) ? [pageIndex] : []));
  assert.equal(pageIndexes.size, 1, `block ${block.id} that fits on one page must not be split`);
}

const graphless = buildModelIrVisualizationBundle(graphlessModel, { views: ["exhaustive", "architecture-overview"] });
assert.equal(graphless.manifest.conservation.expected_count, graphlessModel.tensors_and_storage.storage_objects.length);
assert.equal(graphless.manifest.conservation.rendered_count, graphlessModel.tensors_and_storage.storage_objects.length);
assert(graphless.pages.some((page) => page.caption.includes("does not serialize a program graph")), "graphless caption boundary");

assert.throws(() => buildModelIrVisualizationBundle(graphModel, { views: ["invented"] }), /Unsupported/);
assert.throws(() => buildModelIrVisualizationBundle(graphModel, { orientation: "letter" }), /portrait or landscape/);

console.log("Model IR visualization checks passed (six views, A4 portrait/landscape, deterministic monochrome, traceability, and exhaustive conservation).\n");

function runModelIr(file) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", "graph", file, "--format", "json", "--compact"], {
    encoding: "utf8", maxBuffer: 512 * 1024 * 1024, timeout: 120_000,
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return JSON.parse(result.stdout).model_ir;
}
