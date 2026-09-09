import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";

const root = path.resolve("corpus/external-review/fixtures");
const manifest = JSON.parse(await readFile(path.join(root, "gguf-storage-classes.manifest.json"), "utf8"));
assert.equal(manifest.schema, "deepbom.external_review_gguf_fixtures.v1");
assert.deepEqual(manifest.source, {
  repository: "ggml-org/llama.cpp",
  commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
  path: "ggml/src/ggml.c",
  sha256: "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4",
});

for (const fixture of manifest.fixtures) {
  const filename = path.join(root, fixture.path);
  const bytes = new Uint8Array(await readFile(filename));
  assert.equal(bytes.length, fixture.bytes, `${fixture.path} byte length changed`);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), fixture.sha256, `${fixture.path} digest changed`);

  const direct = parseMetadataModel(bytes, fixture.path, bytes.length, "gguf");
  assert.equal(direct.tensors[0].dtype, fixture.expected.dtype);
  assert.equal(direct.quantization_status.block_quantized_tensor_count, fixture.expected.block_quantized_tensor_count);
  assert.equal(direct.quantization_status.classification, fixture.expected.quantization_classification);
  assert.equal(direct.quantization_status.storage_precision_classification, fixture.expected.storage_precision_classification);
  direct.model_sha256 = fixture.sha256;
  const artifactIr = getArtifactIrContext(direct, {
    filename: fixture.path, format: "gguf", sha256: fixture.sha256, size: bytes.length,
  }).artifact_ir;
  assert.equal(artifactIr.quantization_contracts.records.length, fixture.expected.block_quantized_tensor_count);
  if (fixture.expected.dtype === "F16") {
    assert.equal(artifactIr.quantization_contracts.status, "not_applicable_no_serialized_quantization_contract");
    assert.match(direct.quantization_status.summary, /floating-point storage is not relabeled as quantization/i);
  } else {
    assert.equal(artifactIr.quantization_contracts.records[0].mapping.family, "format_defined_block_encoding");
    assert.equal(artifactIr.quantization_contracts.records[0].mapping.scheme, fixture.expected.dtype);
  }

  const cli = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", filename, "--output-format", "json-compact", "--scan", "full"], {
    cwd: process.cwd(), encoding: "utf8", maxBuffer: 32 * 1024 * 1024,
  });
  assert.equal(cli.status, 0, cli.stderr);
  const output = JSON.parse(cli.stdout);
  assert.equal(output.quantization_status.classification, fixture.expected.quantization_classification);
  assert.equal(output.quantization_status.storage_precision_classification, fixture.expected.storage_precision_classification);

  const envelopeRun = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", filename, "--output-format", "envelope", "--compact", "--scan", "full"], {
    cwd: process.cwd(), encoding: "utf8", maxBuffer: 32 * 1024 * 1024,
  });
  assert.equal(envelopeRun.status, 0, envelopeRun.stderr);
  const envelope = JSON.parse(envelopeRun.stdout);
  assert.equal(envelope.identity.sha256, fixture.sha256);
  assert.equal(envelope.capabilities.conservation.valid, true);
  assert.equal(envelope.capabilities.assessed.includes("block_quantization"), fixture.expected.block_quantized_tensor_count > 0);
}

console.log("External-review GGUF correctness verified: F16 storage remains non-quantized while Q4_0 and Q8_0 retain source-pinned block quantization across native, Artifact IR, CLI, and envelope surfaces.");
