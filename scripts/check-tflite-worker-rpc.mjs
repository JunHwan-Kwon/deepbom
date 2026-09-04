import assert from "node:assert/strict";

import { STATIC_AUDIT_OPERATION } from "../web/lib/static-audit-worker-protocol.js";
import { createTfliteWorkerRpc } from "../web/lib/tflite-worker-rpc.js";

const calls = [];
const bytes = new Uint8Array([1, 2, 3]);
const client = {
  run(operation, payload) {
    calls.push({ operation, payload });
    return Promise.resolve({ operation, payload });
  },
};
const rpc = createTfliteWorkerRpc(client, {
  resolveTarget: (targetId) => `resolved:${targetId}`,
  getCurrentModel: () => ({ bytes, filename: "current.tflite" }),
});

await rpc.analyzeForTarget(bytes, "explicit.tflite", "a55");
await rpc.projectRedesign(bytes, "explicit.tflite", "a55", { edits: [] });
await rpc.exploreRedesignPareto(bytes, "explicit.tflite", "a55", { limit: 4 });
await rpc.inputInfluence(bytes, "explicit.tflite", 7, "a55");
await rpc.outputInfluence(bytes, "explicit.tflite", 8, "a55");
await rpc.runtimeCalibration(bytes, "explicit.tflite", "a55", 2.5);
await rpc.weightHistogram(bytes, "explicit.tflite", 11, "a55");
await rpc.modelTomography(bytes, "explicit.tflite", "a55");
await rpc.layerLandscape(bytes, 12, 1, 2, 9, 0.4);
await rpc.run(STATIC_AUDIT_OPERATION.TFLITE_DEPLOYMENT_DELTA, {
  baselineBytes: new Uint8Array([4, 5, 6]),
  baselineFilename: "baseline.tflite",
  targetIdsJson: JSON.stringify(["android_mid_a55", "rpi4_a72"]),
});

assert.deepEqual(calls.map((call) => call.operation), [
  STATIC_AUDIT_OPERATION.TFLITE_ANALYZE,
  STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PROJECT,
  STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PARETO,
  STATIC_AUDIT_OPERATION.TFLITE_INPUT_INFLUENCE,
  STATIC_AUDIT_OPERATION.TFLITE_OUTPUT_INFLUENCE,
  STATIC_AUDIT_OPERATION.TFLITE_RUNTIME_CALIBRATION,
  STATIC_AUDIT_OPERATION.TFLITE_WEIGHT_HISTOGRAM,
  STATIC_AUDIT_OPERATION.TFLITE_MODEL_TOMOGRAPHY,
  STATIC_AUDIT_OPERATION.TFLITE_LAYER_LANDSCAPE,
  STATIC_AUDIT_OPERATION.TFLITE_DEPLOYMENT_DELTA,
]);
for (const call of calls.slice(0, 8)) {
  assert.equal(call.payload.filename, "explicit.tflite");
  assert.equal(call.payload.targetId, "resolved:a55");
}
assert.equal(calls[3].payload.opIndex, 7);
assert.equal(calls[4].payload.opIndex, 8);
assert.equal(calls[5].payload.measuredMs, 2.5);
assert.equal(calls[6].payload.tensorIndex, 11);
assert.equal(calls[8].payload.filename, "current.tflite", "filename-free kernels must retain the loaded model identity");
assert.equal(calls[8].payload.opIndex, 12);
assert.equal(calls[8].payload.gridSize, 9);
assert.equal(calls[8].payload.radius, 0.4);
assert.equal(calls[9].payload.filename, "current.tflite");
assert.equal(calls[9].payload.targetIdsJson, '["android_mid_a55","rpi4_a72"]');

console.log(`TFLite Worker RPC facade passed (${calls.length} operation and argument bindings).`);
