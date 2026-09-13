import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";

import { canonicalJson } from "../web/lib/report-utils.js";

const manifest = JSON.parse(await readFile("corpus/external-review/measurement-oracles.v1.json", "utf8"));
assert.equal(manifest.schema, "deepbom.measurement_oracles.v1");
assert.equal(manifest.artifacts.length, 3);
for (const row of manifest.artifacts) {
  const bytes = await readFile(row.path);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.sha256, `${row.format} oracle artifact identity drifted`);
}

const ggufOracle = manifest.artifacts.find((row) => row.format === "gguf");
const gguf = cli([ggufOracle.path, "--tensors", "--compact"]);
assert.equal(gguf.tensor_count, ggufOracle.expected.tensor_count);
assert.deepEqual(Object.fromEntries(gguf.encoding_inventory.map((row) => [row.dtype, row.tensor_count])), ggufOracle.expected.encoding_counts);
assert.equal(gguf.encoding_inventory_sha256, ggufOracle.expected.encoding_inventory_sha256);
assert.equal(gguf.tensor_encoding_assignment_sha256, ggufOracle.expected.tensor_encoding_assignment_sha256);
const independentlyProjectedAssignments = gguf.tensors
  .map((row) => ({ index: row.index, name: row.name, dtype: row.encoding, shape: row.shape, byte_length: row.byte_length }))
  .sort((left, right) => left.index - right.index || left.name.localeCompare(right.name));
assert.equal(createHash("sha256").update(canonicalJson(independentlyProjectedAssignments)).digest("hex"), gguf.tensor_encoding_assignment_sha256,
  "The public tensor table must contain enough information to reproduce the assignment signature without DeepBOM internals.");

const onnxOracle = manifest.artifacts.find((row) => row.format === "onnx");
const onnx = cli(["audit", onnxOracle.path, "--compact"]);
assert.equal(onnx.quantization_status.classification, onnxOracle.expected.classification);
assert.equal(onnx.onnx_quantization_binding.status, onnxOracle.expected.binding_status);
assert.equal(onnx.onnx_quantization_binding.binding_count, onnxOracle.expected.binding_count);
for (const binding of onnx.onnx_quantization_binding.bindings) {
  assert.equal(binding.axis, onnxOracle.expected.axis);
  assert.equal(binding.parameterization, onnxOracle.expected.parameterization);
  assert.deepEqual(binding.scale_values, onnxOracle.expected.scale_values);
  assert.deepEqual(binding.zero_point_values, onnxOracle.expected.zero_point_values);
}

const safeOracle = manifest.artifacts.find((row) => row.format === "safetensors");
const safeBytes = await readFile(safeOracle.path);
const headerLength = Number(safeBytes.readBigUInt64LE(0));
const safeHeader = JSON.parse(safeBytes.subarray(8, 8 + headerLength).toString("utf8").trim());
const headerTensors = Object.entries(safeHeader).filter(([name]) => name !== "__metadata__");
assert.equal(headerTensors.length, safeOracle.expected.tensor_count);
assert.equal(headerTensors.every(([, tensor]) => tensor.dtype === "F16"), true);
const safe = cli(["audit", safeOracle.path, "--section", "summary", "--compact"]).sections.summary.storage;
assert.equal(safe.tensor_count, safeOracle.expected.tensor_count);
assert.deepEqual(Object.fromEntries(safe.encodings.map((row) => [row.dtype, row.tensor_count])), safeOracle.expected.encoding_counts);
assert.equal(safe.declared_tensor_bytes, safeOracle.expected.declared_tensor_bytes);
assert.equal(safe.index_binding_status, safeOracle.expected.index_binding_status);
console.log("Measurement oracle gate passed for pinned GGUF, ONNX QDQ, and SafeTensors artifacts, including independently reproducible tensor-assignment hashing.");

function cli(args) {
  const run = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 });
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  return JSON.parse(run.stdout);
}
