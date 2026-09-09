import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { validateSemanticArtifactDiff } from "../web/lib/semantic-artifact-diff.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";

const root = path.resolve(".");
const scratch = path.join(root, ".local-validation", "semantic-artifact-diff");
const fixtures = path.join(root, "corpus", "external-review", "fixtures");
await rm(scratch, { recursive: true, force: true });
await mkdir(scratch, { recursive: true });
const pte = path.join(scratch, "add.pte");
await writeFile(pte, decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64));

const quantManifest = JSON.parse(await readFile(path.join(fixtures, "quant-scale-risk-1000x.manifest.json"), "utf8"));
const quantDiff = diff(
  path.join(fixtures, quantManifest.source.path),
  path.join(fixtures, quantManifest.candidate.path),
);
assert.equal(quantDiff.format, "tflite");
assert.equal(quantDiff.quantization_delta.changed_record_count, 1);
assert.equal(quantDiff.quantization_delta.contract_changed, true);
const scaleChange = quantDiff.quantization_delta.record_alignment.find((row) => row.status === "changed");
assert(scaleChange.changed_fields.includes("parameters"));
assert(Math.abs(scaleChange.scale_ratio.candidate_over_baseline.maximum - quantManifest.mutation.observed_factor) < 1e-9);
assert.equal(quantDiff.change_impact.highest_action, "integration_revalidation");
assert(quantDiff.change_impact.categories.find((row) => row.id === "integration_revalidation")
  .reasons.includes("serialized_quantization_contract_changed"));
assert.equal(quantDiff.tflite_deployment_delta.schema, "deepbom.deployment_delta.v1.1");

const selfFixtures = [
  ["tflite", path.join(fixtures, "quant-scale-risk.tflite")],
  ["onnx", path.join(root, "web", "samples", "mnist-8.onnx")],
  ["coreml", path.join(fixtures, "coreml-affine-attributes.mlpackage")],
  ["gguf", path.join(fixtures, "gguf-f16.gguf")],
  ["safetensors", path.join(root, "web", "samples", "nanofable-1m-fp16.safetensors")],
  ["executorch", pte],
];
for (const [format, fixture] of selfFixtures) {
  const first = diff(fixture, fixture);
  const second = diff(fixture, fixture);
  assert.equal(first.format, format);
  assert.equal(JSON.stringify(first), JSON.stringify(second), `${format} diff must be deterministic`);
  assert.equal(first.baseline.sha256, first.candidate.sha256);
  assert.equal(first.graph_delta.changed_operator_count, 0);
  assert.equal(first.graph_delta.added_operator_count, 0);
  assert.equal(first.graph_delta.removed_operator_count, 0);
  assert.equal(first.storage_delta.changed_object_count, 0);
  assert.equal(first.quantization_delta.changed_record_count, 0);
  assert.equal(first.quantization_delta.contract_changed, false);
  assert.equal(first.change_impact.highest_action, "no_change_observed");
}

const crossFormat = run(["diff", selfFixtures[0][1], selfFixtures[1][1], "--json"]);
assert.equal(crossFormat.status, 1);
assert.match(crossFormat.stderr, /requires matching artifact formats/);
const invalidTarget = run(["diff", selfFixtures[1][1], selfFixtures[1][1], "--target", "rpi4_a72", "--json"]);
assert.equal(invalidTarget.status, 1);
assert.match(invalidTarget.stderr, /applies only to TFLite artifacts/);

console.log("Semantic Artifact IR diff verified for TFLite, ONNX, Core ML, GGUF, SafeTensors, and ExecuTorch, including a 1000x per-axis scale mutation.");

function diff(baseline, candidate) {
  const result = run(["diff", baseline, candidate, "--json"]);
  assert.equal(result.status, 0, result.stderr);
  return validateSemanticArtifactDiff(JSON.parse(result.stdout));
}

function run(args) {
  return spawnSync(process.execPath, [path.join(root, "bin", "deepbom.mjs"), ...args], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
}
