import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";

const root = path.resolve("corpus/external-review/fixtures");
const manifest = JSON.parse(await readFile(path.join(root, "coreml-affine-attributes.manifest.json"), "utf8"));
assert.equal(manifest.schema, "deepbom.external_review_coreml_fixtures.v1");
assert.deepEqual(manifest.generator, {
  name: "coremltools",
  version: "9.0",
  sdist_sha256: "4ff346b29c31c4b45acd19a20e0f0a1ac65180a96776e62f15bd5c46f4926687",
  backend_exporter_sha256: "1a01006e1d87eb37626869c249c3e9df755cbb9a83a9c1e233d68a4b435620ae",
  affine_op_definition_sha256: "2575e57c32f99c18e8e9d32af152ee19f9414835cb1c141ab19a92ec0b659a6d",
});

const [validRow, invalidRow] = manifest.fixtures;
const validPath = path.join(root, validRow.path);
const invalidPath = path.join(root, invalidRow.path);
const validBytes = await readFile(validPath);
const invalidBytes = await readFile(invalidPath);
assertFixture(validBytes, validRow);
assertFixture(invalidBytes, invalidRow);

const valid = (await readCoreMlModelFile(new File([validBytes], "model.mlmodel"))).analysis;
const contract = valid.coreml?.mil_compression_contract?.transforms?.[0];
assert.equal(contract?.status, "assessed_exact_serialized_contract");
assert.equal(contract?.normalized_axis, 1);
assert.equal(contract?.scale_elements, 3);
assert.deepEqual(contract?.serialized_binding_sources, {
  quantized_data: "attribute",
  zero_point: "attribute",
  scale: "attribute",
  axis: "attribute",
});

const packageFiles = [
  packageFile(await readFile(path.join(root, "coreml-affine-attributes.mlpackage", "Manifest.json")), "coreml-affine-attributes.mlpackage/Manifest.json"),
  packageFile(validBytes, "coreml-affine-attributes.mlpackage/Data/com.deepbom.fixture/model.mlmodel"),
];
const bundled = (await readArtifactBundle(packageFiles)).analysis;
assert.equal(bundled.artifact_bundle?.kind, "coreml_mlpackage");
assert.equal(bundled.coreml?.mil_compression_contract?.status, "assessed_exact_serialized_contracts");

await assert.rejects(
  readCoreMlModelFile(new File([invalidBytes], "missing-axis.mlmodel")),
  /missing a serialized quantized_data, zero_point, scale, or scalar axis binding/,
);

const cli = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", validPath, "--output-format", "envelope", "--compact"], {
  cwd: process.cwd(), encoding: "utf8", maxBuffer: 32 * 1024 * 1024,
});
assert.equal(cli.status, 0, cli.stderr);
const envelope = JSON.parse(cli.stdout);
assert.equal(envelope.identity.sha256, validRow.sha256);
assert.equal(envelope.capabilities.assessed.includes("affine_quantization"), true);
assert.equal(envelope.format_extensions.coreml.mil_compression_contract.transforms[0].normalized_axis, 1);

console.log("External-review Core ML correctness verified: official CoreML6 attribute serialization is assessed and a missing required axis fails closed.");

function assertFixture(bytes, row) {
  assert.equal(bytes.length, row.bytes, `${row.path} byte length changed`);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.sha256, `${row.path} digest changed`);
}

function packageFile(payload, relativePath) {
  const file = new File([payload], relativePath.split("/").at(-1));
  Object.defineProperty(file, "webkitRelativePath", { value: relativePath });
  return file;
}
