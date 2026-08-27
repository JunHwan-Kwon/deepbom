import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { indexQuantRiskOps, quantRiskForOp } from "../web/lib/app-deepbom-workspace.js";
import { analyzeDeepBomInWorker } from "../web/lib/deepbom-worker-client.js";

const duplicateNames = [
  { op_index: 7, name: "CONV_2D", quant_risk: "warn" },
  { op_index: 12, name: "CONV_2D", quant_risk: "risk" },
];
const riskMap = indexQuantRiskOps(duplicateNames);
assert.equal(quantRiskForOp(riskMap, { index: 7, name: "CONV_2D" }, 0)?.quant_risk, "warn");
assert.equal(quantRiskForOp(riskMap, { index: 12, name: "CONV_2D" }, 1)?.quant_risk, "risk");

const originalWorker = globalThis.Worker;
class FakeWorker {
  listeners = new Map();
  terminated = false;
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  postMessage(payload, transfer) {
    assert.equal(payload.analysisJson, '{"operator_count":2}');
    assert.equal(payload.modelBytes.byteLength, 4);
    assert.equal(transfer?.[0], payload.modelBytes.buffer);
    queueMicrotask(() => this.listeners.get("message")?.({
      data: { type: "result", result: { score_assessment: { status: "ASSESSED" } } },
    }));
  }
  terminate() { this.terminated = true; }
}
globalThis.Worker = FakeWorker;
const sourceBytes = new Uint8Array([1, 2, 3, 4]);
const result = await analyzeDeepBomInWorker({
  manifest: {
    module_url: "https://deepbom.org/module.js",
    wasm_url: "https://deepbom.org/module.wasm",
    version: "test",
  },
  modelBytes: sourceBytes,
  analysis: { operator_count: 2 },
});
assert.equal(result.score_assessment.status, "ASSESSED");
assert.deepEqual([...sourceBytes], [1, 2, 3, 4], "Worker transfer must not detach the retained model bytes.");
globalThis.Worker = originalWorker;

const rustSource = await readFile(new URL("../protected/deepbom_wasm/src/lib.rs", import.meta.url), "utf8");
assert(!rustSource.includes("Score confidence band"));
assert(rustSource.includes("Assumption sensitivity envelope"));
assert(rustSource.includes("NOT_ASSESSABLE"));
assert(rustSource.includes("non_finite_count"));

console.log("DEEPBOM isolated-analysis, op-identity, and score-boundary contract checks passed.");
