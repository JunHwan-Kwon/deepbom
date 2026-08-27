import assert from "node:assert/strict";
import {
  aggregateGrids,
  aggregateHessian,
  applyLandscapePatch,
  computeHessian2D,
  computeRadialProfileSEM,
  linspaceArr,
  requantRatio,
  subtractCenter,
} from "../web/lib/research.js";

const model = new Uint8Array(8);
model[0] = 0xff;
model[1] = 0x01;
new DataView(model.buffer).setFloat32(4, 2, true);
const patched = applyLandscapePatch(
  model,
  new Float32Array([-1, 1, 2]),
  new Float32Array(3),
  0.5,
  0,
  [
    { buf_offset: 0, elem_count: 2, dtype: "INT8", oc: 1, filter_size: 2, scales: [1], zps: [0] },
    { buf_offset: 4, elem_count: 1, dtype: "FLOAT32", oc: 1, filter_size: 1, scales: [], zps: [] },
  ],
);
assert.deepEqual(Array.from(patched.slice(0, 2)), [0xfe, 0x02], "INT8 landscape patch must round both half ties away from zero");
assert.equal(new DataView(patched.buffer).getFloat32(4, true), 3, "FLOAT32 landscape patch must use the same direction ledger");
assert.deepEqual(Array.from(model.slice(0, 2)), [0xff, 0x01], "Landscape patch must not mutate the source artifact bytes");
assert.equal(new DataView(model.buffer).getFloat32(4, true), 2, "Landscape patch must preserve source FLOAT32 bytes");

assert.throws(
  () => applyLandscapePatch(model, new Float32Array(2), new Float32Array(2), 1, 0, [
    { buf_offset: 0, elem_count: 3, dtype: "INT8", oc: 1, filter_size: 3, scales: [1], zps: [0] },
  ]),
  /exactly 3 values/,
);
assert.throws(
  () => applyLandscapePatch(model, new Float32Array(1), new Float32Array(1), 1, 0, [
    { buf_offset: 8, elem_count: 1, dtype: "INT8", oc: 1, filter_size: 1, scales: [1], zps: [0] },
  ]),
  /exceeds 8 model bytes/,
);
assert.throws(
  () => applyLandscapePatch(model, new Float32Array(1), new Float32Array(1), 1, 0, [
    { buf_offset: 0, elem_count: 1, dtype: "UINT8", oc: 1, filter_size: 1, scales: [1], zps: [0] },
  ]),
  /Unsupported landscape weight dtype UINT8/,
);

assert.deepEqual(linspaceArr(-1, 1, 5), [-1, -0.5, 0, 0.5, 1]);
const axes = [-1, 0, 1];
const quadratic = axes.map((beta) => axes.map((alpha) => 3 * alpha ** 2 + 2 * alpha * beta + 5 * beta ** 2));
const hessian = computeHessian2D(quadratic, axes, 3);
assert.equal(hessian.Haa, 6);
assert.equal(hessian.Hbb, 10);
assert.equal(hessian.Hab, 2);
assert.equal(hessian.trace, 16);
assert.ok(Math.abs(hessian.lambdaMax - (8 + Math.sqrt(8))) < 1e-12);
assert.equal(computeHessian2D(quadratic, [-1, 0, 2], 3), null, "Non-uniform finite-difference axes must fail closed");

const hessianAggregate = aggregateHessian([
  { lambdaMax: 1, trace: 2 },
  { lambdaMax: 3, trace: 6 },
]);
assert.equal(hessianAggregate.lambdaMax_mean, 2);
assert.equal(hessianAggregate.lambdaMax_sem, 1, "Two-sample lambda SEM must use sample variance");
assert.equal(hessianAggregate.trace_mean, 4);
assert.equal(hessianAggregate.trace_sem, 2, "Two-sample trace SEM must use sample variance");

const grids = aggregateGrids([[[1]], [[3]]], 1);
assert.equal(grids.mean[0][0], 2);
assert.equal(grids.sem[0][0], 1, "Grid SEM must use sample variance");
assert.equal(grids.centerLoss, 2);
assert.deepEqual(subtractCenter([[1, 2], [3, 4]], 2), [[-3, -2], [-1, 0]]);

const radial = computeRadialProfileSEM(
  [quadratic, quadratic.map((row) => row.map((value) => value * 2))],
  axes,
  3,
  2,
);
assert.equal(radial.rc.length, 2);
assert.equal(radial.mu.length, 2);
assert.equal(radial.sem.length, 2);
assert.ok(radial.sem.some((value) => value > 0), "Cross-seed radial SEM must retain observed variation");
assert.equal(requantRatio([[0, 2], [4, 6]], [[0, 1], [2, 3]], 2), 2);
assert.equal(
  requantRatio([[0, 1_000_000], [2, 4]], [[0, 1e-12], [1, 2]], 2),
  2,
  "OLS gain must exclude near-zero floating deltas instead of amplifying pointwise ratios",
);
assert.equal(requantRatio([[0]], [[0]], 1), null, "Undefined OLS gain must remain not-assessed");

console.log("Landscape math contract passed (strict patch spans, ties-away INT8 requantization, analytic Hessian, and sample SEM).");
