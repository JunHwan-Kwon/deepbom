import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(".");
const fixtureRoot = path.join(root, "corpus", "external-review", "fixtures", "onnx-invalid-external-range");
const manifest = JSON.parse(await readFile(path.join(fixtureRoot, "manifest.json"), "utf8"));
assert.equal(manifest.schema, "deepbom.external_review_onnx_external_range_fixture.v1");
for (const row of manifest.files) {
  const bytes = await readFile(path.join(fixtureRoot, row.path));
  assert.equal(bytes.byteLength, row.bytes);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.sha256);
}
const result = spawnSync(process.execPath, [
  path.join(root, "bin", "deepbom.mjs"),
  "audit",
  path.join(fixtureRoot, "model.onnx"),
  "--output-format",
  "json-compact",
], { cwd: root, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
assert.equal(result.status, 0, result.stderr);
const analysis = JSON.parse(result.stdout);
assert.equal(analysis.onnx_external_data.status, manifest.expected.external_data_status);
assert.equal(analysis.onnx_external_data.range_out_of_bounds_count, manifest.expected.range_out_of_bounds_count);
assert.equal(analysis.onnx_external_data.verified_payload_count, manifest.expected.verified_payload_count);
const weight = analysis.tensors.find((row) => row.name === "weight");
assert(weight);
assert.equal(weight.constant_buffer, true, "An external initializer remains a semantic constant even when its payload cannot be verified.");
assert.equal(weight.initializer_bytes, 0);
assert.equal(weight.buffer_data_status, "not_assessed_external_payload_unavailable");
console.log("External-review ONNX correctness verified: a hash-bound out-of-bounds sidecar range fails closed without hydrating tensor data.");
