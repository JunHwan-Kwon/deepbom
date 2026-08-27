import assert from "node:assert/strict";
import {
  buildDualRadialSvg,
  drawLandscapeCanvas,
  drawMvDepth,
  drawMvFilter,
  landscapeColor,
  modelDepthValues,
  modelFilterColor,
} from "../web/lib/research-visuals.js";

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.attributes = new Map();
    this.children = [];
    this.style = {};
    this.textContent = "";
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  append(...children) {
    this.children.push(...children);
  }
}

assert.equal(modelFilterColor(0, 0), "#e2e8f0");
assert.equal(modelFilterColor(-1, 1), "rgb(0,0,255)");
assert.equal(modelFilterColor(1, 1), "rgb(255,0,0)");
assert.equal(landscapeColor(0), "rgb(255,255,255)");
assert.equal(landscapeColor(1), "rgb(58,36,16)");

const tomography = [
  { oc: 2, l2_per_oc: [1, 4], haar_ll: 7, haar_hh: 11, quant_snr_valid: true, quant_snr_db: 45 },
  { oc: 1, l2_per_oc: [3], haar_ll: 8, haar_hh: 12, quant_snr_valid: false, quant_snr_db: 1 },
];
assert.deepEqual(modelDepthValues(tomography, { metric: "l2", ocIdx: 1 }), [4, 3]);
assert.deepEqual(modelDepthValues(tomography, { metric: "haar_hh", ocIdx: 0 }), [11, 12]);
assert.deepEqual(modelDepthValues(tomography, { metric: "snr", ocIdx: 0 }), [15, 0]);
assert.deepEqual(modelDepthValues(tomography, { metric: "unknown", ocIdx: 0 }), [0, 0]);

const filterCanvas = fakeCanvas(20, 20);
drawMvFilter(filterCanvas, { kh: 2, kw: 2, oc: 1, spatial_flat: [-1, 0, 0.5, 1] }, 0);
assert.equal(filterCanvas.context.calls.filter((call) => call[0] === "fillRect").length, 4);
assert.equal(filterCanvas.context.calls.filter((call) => call[0] === "stroke").length, 2);

const depthCanvas = fakeCanvas(120, 60);
drawMvDepth(depthCanvas, tomography, { metric: "haar_ll", ocIdx: 0, layerIdx: 99 });
assert.equal(depthCanvas.context.calls.filter((call) => call[0] === "arc").length, 1);
assert(depthCanvas.context.calls.some((call) => call[0] === "fillText" && call[1] === "layer index  →"));

const landscapeCanvas = fakeCanvas(20, 20);
drawLandscapeCanvas(landscapeCanvas, [[0, 0.5], [0.75, 1]], 2, 1);
assert.equal(landscapeCanvas.context.calls.filter((call) => call[0] === "fillRect").length, 4);
assert.equal(landscapeCanvas.context.calls.filter((call) => call[0] === "arc").length, 1);

const priorDocument = globalThis.document;
globalThis.document = { createElementNS: (_namespace, tagName) => new FakeElement(tagName) };
try {
  const svg = buildDualRadialSvg(
    { rc: [0, 1], mu: [0, 2], sem: [0.1, 0.2] },
    { rc: [0, 1], mu: [0, 1], sem: [0.05, 0.1] },
  );
  assert.equal(svg.tagName, "svg");
  assert.equal(countTags(svg, "polygon"), 2);
  assert.equal(countTags(svg, "polyline"), 2);
  assert.equal(countTags(svg, "circle"), 4);
  assert.equal(countTags(svg, "text"), 2);
  assert.equal(buildDualRadialSvg(null, null), null);
} finally {
  if (priorDocument === undefined) delete globalThis.document;
  else globalThis.document = priorDocument;
}

console.log("Research visual math passed (state-local depth metrics, canvas grids, and radial SVG geometry).");

function fakeCanvas(width, height) {
  const calls = [];
  const context = { calls };
  for (const name of ["clearRect", "fillRect", "beginPath", "moveTo", "lineTo", "stroke", "closePath", "fill", "arc", "fillText"]) {
    context[name] = (...args) => calls.push([name, ...args]);
  }
  return { width, height, context, getContext: () => context };
}

function countTags(node, tagName) {
  return Number(node.tagName === tagName)
    + node.children.reduce((count, child) => count + countTags(child, tagName), 0);
}
