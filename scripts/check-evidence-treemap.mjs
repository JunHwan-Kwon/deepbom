import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  buildQuantizationExposurePresentation,
  buildResourceMapPresentation,
  layoutGroupedTreemap,
  layoutTreemap,
} from "../web/lib/evidence-treemap.js";

const analysis = {
  format: "tflite",
  block_inventory: {
    status: "assessed",
    blocks: [
      { block_id: "b0", display_name: "Stem", op_indices: [0, 1] },
      { block_id: "b1", display_name: "Head", op_indices: [2, 3] },
    ],
    stages: [
      { stage_id: "s0", index: 0, display_name: "Feature extraction", op_indices: [0, 1] },
      { stage_id: "s1", index: 1, display_name: "Prediction", op_indices: [2, 3] },
    ],
  },
  ops: [
    { index: 0, name: "CONV_2D", macs: 100, estimated_bytes: 400, bottleneck_total_us: 8, stage_index: 0, quantization_state: "quantized_compute", quant_risk: "ok", static_bound_guess: "compute-bound", xnnpack_chain_id: 0 },
    { index: 1, name: "RESHAPE", macs: 0, estimated_bytes: 120, bottleneck_total_us: 1, stage_index: 0, quantization_state: "quantized_data_movement", quant_risk: "ok", static_bound_guess: "memory-bound", xnnpack_chain_id: 0 },
    { index: 2, name: "ADD", macs: 50, estimated_bytes: 300, bottleneck_total_us: 4, stage_index: 1, quantization_state: "quantized_compute", quant_risk: "warn", static_bound_guess: "mixed", xnnpack_chain_break: true },
    { index: 3, name: "CUSTOM", macs: null, estimated_bytes: null, bottleneck_total_us: null, stage_index: 1, quantization_state: "none", quant_risk: "none", intensity_status: "not_assessed" },
  ],
  tensors: [],
};

const resource = buildResourceMapPresentation(analysis, { metric: "macs", groupBy: "stage", colorBy: "intensity" });
assert.equal(resource.status, "assessed");
assert.equal(resource.total, 150);
assert.equal(resource.items.length, 2);
assert.equal(resource.assessedCount, 3);
assert.equal(resource.zeroCount, 1);
assert.equal(resource.unassessedCount, 1);
assert.equal(resource.mappedGroupCount, 2);
assert.equal(resource.assessedGroupCount, 2);
assert.equal(resource.conservationStatus, "exact");

const grouped = layoutGroupedTreemap(resource.items, 1000, 500);
assert.equal(grouped.tiles.length, 2);
assert.equal(grouped.groups.length, 2);
const canvasArea = 1000 * 500;
for (const tile of grouped.tiles) {
  const observedShare = tile.rect.w * tile.rect.h / canvasArea;
  assert.ok(Math.abs(observedShare - tile.value / resource.total) < 1e-12, `${tile.id}: area share is not linear.`);
}
assert.deepEqual(grouped, layoutGroupedTreemap(resource.items, 1000, 500), "Treemap layout must be deterministic.");

const traffic = buildResourceMapPresentation(analysis, { metric: "traffic", groupBy: "block", colorBy: "delegation" });
assert.equal(traffic.total, 820);
assert.equal(traffic.zeroCount, 0);
assert.equal(traffic.unassessedCount, 1);
assert.equal(traffic.items.find((row) => row.index === 2).tone, "risk");

const quant = buildQuantizationExposurePresentation(analysis, { metric: "macs", groupBy: "stage" }, null, new Map([
  [2, { tone: "risk", labels: ["requantization contract"] }],
]));
assert.equal(quant.total, 150);
assert.equal(quant.zeroCount, 1, "Zero-MAC quantized movement must remain explicit outside the area map.");
assert.equal(quant.items.find((row) => row.index === 2).tone, "risk");
assert.match(quant.items.find((row) => row.index === 2).detail, /requantization contract/);

const structural = buildResourceMapPresentation({
  format: "onnx",
  stages: [
    { index: 0, key: "stem", first_op: 0, last_op: 1 },
    { index: 1, key: "head", first_op: 2, last_op: 2 },
  ],
  ops: analysis.ops.slice(0, 3).map((op) => ({ ...op, stage_index: undefined, stage_key: undefined })),
  tensors: [],
}, { metric: "traffic", groupBy: "stage" });
assert.equal(structural.mappedGroupCount, 2, "Range-based structural stages must remain distinct when op_indices is absent.");
assert.equal(structural.groupOptions.some((row) => row.id === "block"), false, "Formats without a block inventory must not expose a non-functional Block grouping.");

const serialized = {
  format: "gguf",
  legend: [["good", "block decode passed"], ["risk", "invalid encoding"]],
  scope: "GGUF storage encoding is not an affine interface contract.",
  tiles: [
    { index: 0, name: "a.weight", dtype: "Q4_0", encoding_class: "block_quantized", tone: "good", byte_length: 96 },
    { index: 1, name: "b.weight", dtype: "F32", encoding_class: "scalar", tone: "neutral", byte_length: 32 },
    { index: 2, name: "c.weight", dtype: "Q4_0", encoding_class: "unsupported_or_invalid", tone: "risk", byte_length: 0 },
  ],
};
const storage = buildQuantizationExposurePresentation({ format: "gguf", ops: [], tensors: [] }, {}, serialized);
assert.equal(storage.scope, "serialized_tensors");
assert.equal(storage.total, 128);
assert.equal(storage.zeroCount, 1);
assert.equal(storage.unassessedCount, 0);
assert.match(storage.boundary, /not an affine interface contract/);

assert.deepEqual(layoutTreemap([], 100, 100), []);
assert.deepEqual(layoutTreemap([{ id: "zero", value: 0 }], 100, 100), []);

const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
const elements = await readFile(new URL("../web/lib/elements.js", import.meta.url), "utf8");
const graphWorkspace = await readFile(new URL("../web/lib/app-graph-workspace.js", import.meta.url), "utf8");
const performanceVisuals = await readFile(new URL("../web/lib/performance-visuals.js", import.meta.url), "utf8");
const visualExport = await readFile(new URL("../web/lib/visual-export.js", import.meta.url), "utf8");
assert.match(html, /id="resourceMapPanel"/);
assert.match(html, /data-explorer-tab="resource"/);
assert.match(html, /id="quantExposureMap"/);
assert.match(elements, /resourceMapPanel: doc\.getElementById\("resourceMapPanel"\)/);
assert.match(elements, /quantExposureMap: doc\.getElementById\("quantExposureMap"\)/);
assert.match(graphWorkspace, /renderResourceMap\(analysis\)/);
assert.match(performanceVisuals, /renderQuantizationExposureMap\(analysis\)/);
assert.match(visualExport, /visuals\/explorer_resource_map\.png/);
assert.match(visualExport, /visuals\/quantization_exposure_map\.png/);

console.log("Evidence treemap checks passed: exact totals, zero/unassessed separation, deterministic linear area, format-aware scope, and DOM integration.");
