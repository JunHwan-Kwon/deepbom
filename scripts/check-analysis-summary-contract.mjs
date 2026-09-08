import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

import {
  deriveMacConfidence,
  MAC_CONFIDENCE_VALUES,
  normalizeAnalysisSummaryContract,
} from "../web/lib/analysis-summary-contract.js";

assert.deepEqual(MAC_CONFIDENCE_VALUES, ["exact", "symbolic", "partial", "not_applicable"]);

for (const row of [
  { analysis: { format: "onnx", total_macs: 32, mac_assessment: { status: "assessed", compute_ops: 1, assessed_compute_ops: 1 } }, expected: "exact" },
  { analysis: { format: "tflite", total_macs: null, mac_assessment: { status: "partially_assessed", compute_ops: 1, assessed_compute_ops: 0 }, dynamic_shape_cost_contract: { total_macs_formula_status: "exact_symbolic_integer_polynomial", total_macs_formula: { expression: "32*D0" } } }, expected: "symbolic" },
  { analysis: { format: "coreml", total_macs: null, mac_assessment: { status: "partially_assessed", compute_ops: 2, assessed_compute_ops: 1 } }, expected: "partial" },
  { analysis: { format: "executorch", total_macs: 0, mac_assessment: { status: "assessed", compute_ops: 0, assessed_compute_ops: 0 } }, expected: "not_applicable" },
  { analysis: { format: "gguf" }, expected: "not_applicable" },
  { analysis: { format: "safetensors" }, expected: "not_applicable" },
]) {
  assert.equal(deriveMacConfidence(row.analysis), row.expected);
  const normalized = normalizeAnalysisSummaryContract(row.analysis);
  assert.equal(normalized.mac_confidence, row.expected);
  assert.equal(Object.hasOwn(normalized, "total_macs"), true);
  assert.equal(Object.hasOwn(normalized, "total_macs_decimal"), true);
}

for (const [path, expected] of [
  ["web/samples/mnist-8.onnx", "exact"],
  ["web/samples/MNISTClassifier.mlmodel", "exact"],
  ["web/samples/tinymqa1m.Q4_0.gguf", "not_applicable"],
  ["web/samples/nanofable-1m-fp16.safetensors", "not_applicable"],
]) {
  const run = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", path, "--output-format", "json-compact", "--pointer", "/mac_confidence"], {
    cwd: process.cwd(),
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  assert.equal(run.status, 0, run.stderr);
  assert.equal(JSON.parse(run.stdout).value, expected, `${path} confidence changed`);
}

console.log("Analysis summary contract verified: MAC confidence is explicit and normalized across executable graphs and weight containers.");
