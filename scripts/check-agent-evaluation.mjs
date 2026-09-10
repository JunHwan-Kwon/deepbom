import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { buildAgentCapabilities } from "../bin/deepbom-agent-contract.mjs";
import { buildCliCapabilities } from "../bin/deepbom-automation.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const version = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;
const ledger = JSON.parse(await readFile(path.join(root, "corpus", "agent-native", "agent-evaluation-cases.v1.json"), "utf8"));
const capabilities = buildAgentCapabilities(buildCliCapabilities(version, {
  defaultTarget: "android_mid_a55",
  deltaTargets: ["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"],
}));

assert.equal(ledger.schema, "deepbom.agent_evaluation_cases.v1");
assert.deepEqual(ledger.cases.map((row) => row.id), [
  "tflite-static-deployment-audit",
  "onnx-mac-coverage",
  "gguf-memory-scenario",
  "artifact-version-diff",
  "actual-gpu-speed-claim",
]);
assert.match(capabilities.selection.use_when.join(" "), /MAC|quantization|memory feasibility|accelerator eligibility|artifact diff/);
assert.match(capabilities.selection.do_not_use_when.join(" "), /measured latency|actual runtime assignment/);

const outputs = new Map();
for (const item of ledger.cases.filter((row) => Array.isArray(row.arguments))) {
  const fixture = path.join(root, item.fixture);
  assert.equal((await readFile(fixture)).byteLength > 0, true, `${item.id}: fixture is empty`);
  const result = spawnSync(process.execPath, [path.join(root, "bin", "deepbom.mjs"), ...item.arguments], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${item.id}: ${result.stderr || result.stdout}`);
  outputs.set(item.id, result.stdout);
}

for (const id of ["tflite-static-deployment-audit", "onnx-mac-coverage", "gguf-memory-scenario"]) {
  const output = outputs.get(id);
  assert.match(output, new RegExp(`DEEPBOM ${escapeRegExp(version)} deployment-artifact audit`));
  assert.match(output, /Identity: sha256:[0-9a-f]{64}/);
  assert.match(output, /Artifact defects/);
  assert.match(output, /Cautions/);
  assert.match(output, /Evidence needed/);
  assert.match(output, /Coverage:/);
  assert.match(output, /Evidence boundary:/);
}
assert.match(outputs.get("tflite-static-deployment-audit"), /Target: .*\(default_assumption\)/);
assert.match(outputs.get("onnx-mac-coverage"), /6,912 MACs \(exact\)/);
assert.match(outputs.get("gguf-memory-scenario"), /executable graph not serialized/);
assert.match(outputs.get("gguf-memory-scenario"), /integrity not_assessed_scan_policy_structure/);
assert.match(outputs.get("artifact-version-diff"), /Graph: 5 matched \| 0 changed/);
assert.match(outputs.get("artifact-version-diff"), /Evidence boundary:/);

const gpuCase = ledger.cases.find((row) => row.id === "actual-gpu-speed-claim");
assert.equal(gpuCase.expected_selection, "use_for_preflight_but_refuse_runtime_conclusion");
assert.match(gpuCase.required_boundary, /not measured latency|not.*runtime assignment/i);
assert.equal(capabilities.execution_boundary.runtime_measurement_inferred, false);

console.log(`Agent evaluation contract passed (${ledger.cases.length} selection scenarios; 4 local executions across TFLite, ONNX, GGUF, and diff; runtime overclaim refused).`);

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
