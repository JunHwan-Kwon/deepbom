import assert from "node:assert/strict";

import { installRuntimeEvidenceController } from "../web/lib/runtime-evidence-controller.js";
import { buildExecutionPlacementEvidence } from "../web/lib/execution-placement-evidence.js";
import { tfliteAcceleratorSourceManifest } from "../web/lib/tflite-accelerator-source-profiles.js";

const SHA = "a".repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: SHA,
  ops: [
    { index: 0, name: "CONV_2D", version: 1, inputs: [0], outputs: [1], macs: 8, xnnpack_chain_id: 0 },
    { index: 1, name: "CUSTOM", version: 1, inputs: [1], outputs: [2], macs: 0, xnnpack_chain_id: -1 },
  ],
  tensors: [
    { index: 0, name: "in", dtype: "FLOAT32", shape: [1, 2] },
    { index: 1, name: "mid", dtype: "FLOAT32", shape: [1, 2] },
    { index: 2, name: "out", dtype: "FLOAT32", shape: [1, 2] },
  ],
};

const input = new EventTarget();
input.files = [];
input.value = "";
const clearButton = new EventTarget();
let runtimeEvidence = null;
let changed = 0;
let statusResolver = null;
installRuntimeEvidenceController({
  elements: { input, clearButton },
  modal: { open() {}, close() {} },
  getAnalysis: () => analysis,
  getEvidence: () => runtimeEvidence,
  setEvidence: (value) => { runtimeEvidence = value; },
  getPending: () => null,
  setPending() {},
  ensureArtifactHash: async () => {},
  artifactFilename: (value) => value,
  onChanged: () => { changed += 1; },
  setStatus: (message, tone) => statusResolver?.({ message, tone }),
});

const qualcommSource = tfliteAcceleratorSourceManifest().profiles.find((row) => row.id === "litert_qualcomm_qnn");
await importDocument({
  schema: "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1",
  artifact_sha256: SHA,
  source: { litert_commit: qualcommSource.source.commit, rulepack_sha256: qualcommSource.rulepack_sha256 },
  compiler: { name: "fixture", version: "1", binary_sha256: "b".repeat(64) },
  invocation: { options: [] },
  compiled_plan_sha256: "c".repeat(64),
  operations: [
    { op_index: 0, op_name: "CONV_2D", compile_status: "compiled" },
    { op_index: 1, op_name: "CUSTOM", compile_status: "not_compiled", reason: "compiler_rejected" },
  ],
  dispatch: { status: "not_observed" },
}, "qualcomm.json", /Qualcomm compiler\/dispatch evidence imported/);
assert.equal(analysis.litert_qualcomm_evidence.summary.compiled_operation_count, 1);
assert.equal(buildExecutionPlacementEvidence(analysis).static_profiles.some((row) => row.profile_id === "litert_qualcomm_qnn_compiled_plan"), true);

clearButton.dispatchEvent(new Event("click"));
assert.equal(analysis.litert_qualcomm_evidence, undefined, "clear must remove imported Qualcomm evidence");

await importDocument({
  schema: "deepbom.edgetpu_compiler_evidence.v1",
  artifact_sha256: SHA,
  compiler: { name: "edgetpu_compiler", version: "1", binary_sha256: "d".repeat(64) },
  invocation: { options: [] },
  compiled_artifact_sha256: "e".repeat(64),
  compiler_report_sha256: "f".repeat(64),
  operations: [
    { op_index: 0, op_name: "CONV_2D", mapping: "mapped" },
    { op_index: 1, op_name: "CUSTOM", mapping: "unmapped", reason: "compiler_rejected" },
  ],
}, "edgetpu.json", /Edge TPU compiler evidence imported/);
assert.equal(analysis.edgetpu_compiler_evidence.summary.mapped_operation_count, 1);
assert.equal(buildExecutionPlacementEvidence(analysis).static_profiles.some((row) => row.profile_id === "google_edgetpu_compiled_plan"), true);
assert(changed >= 3, "imports and clear must invalidate dependent Web projections");

console.log("Runtime accelerator import checks passed: Web schema routing, artifact binding, placement projection, and clear semantics.");

async function importDocument(document, name, expectedStatus) {
  const status = new Promise((resolve) => { statusResolver = resolve; });
  input.files = [new File([JSON.stringify(document)], name, { type: "application/json" })];
  input.dispatchEvent(new Event("change"));
  const result = await status;
  assert.equal(result.tone, "ok", result.message);
  assert.match(result.message, expectedStatus);
}
