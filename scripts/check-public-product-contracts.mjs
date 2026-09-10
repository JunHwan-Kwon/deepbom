import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { buildCliCapabilities } from "../bin/deepbom-automation.mjs";
import { MODEL_FORMAT_ADAPTERS, modelFormatAdapter } from "../web/lib/model-file.js";
import { PUBLIC_PRODUCT_CONTRACTS } from "../web/lib/public-product-contracts.js";
import { resolveVersionContract } from "./version-contract.mjs";

assert.equal(PUBLIC_PRODUCT_CONTRACTS.schema, "deepbom.public_product_contracts.v1");
assert.deepEqual(PUBLIC_PRODUCT_CONTRACTS.format_maturity.map((row) => row.format),
  ["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"]);
for (const row of PUBLIC_PRODUCT_CONTRACTS.format_maturity) {
  assert(MODEL_FORMAT_ADAPTERS[row.format], `Missing format adapter for ${row.format}`);
  assert.equal(modelFormatAdapter(row.format).maturity, row);
  assert.match(row.interpretation_boundary, /\S/);
  for (const key of ["overall", "parser", "serialized_graph", "quantization", "execution_placement"]) assert.match(row[key], /^[a-z][a-z0-9_]*$/);
}
for (const format of ["gguf", "safetensors"]) {
  const row = PUBLIC_PRODUCT_CONTRACTS.format_maturity.find((item) => item.format === format);
  assert.equal(row.serialized_graph, "not_applicable");
  assert.equal(row.execution_placement, "not_applicable");
}
assert.equal(PUBLIC_PRODUCT_CONTRACTS.format_maturity.find((row) => row.format === "coreml").overall, "preview");
assert.equal(PUBLIC_PRODUCT_CONTRACTS.format_maturity.find((row) => row.format === "executorch").overall, "preview");

const schemas = new Set(PUBLIC_PRODUCT_CONTRACTS.machine_contracts.map((row) => row.schema));
assert.equal(schemas.size, PUBLIC_PRODUCT_CONTRACTS.machine_contracts.length);
for (const required of ["deepbom.cli_capabilities.v1", "deepbom.artifact_evidence_envelope.v1", "deepbom.artifact_ir.v2", "deepbom.semantic_artifact_diff.v1", "cyclonedx-1.7-json"]) assert(schemas.has(required));
assert.equal([...schemas].some((value) => /2\.0|perspective/i.test(value)), false);

const release = PUBLIC_PRODUCT_CONTRACTS.release_policy;
assert.equal(release.stable_channel.npm_dist_tag, "latest");
assert.equal(release.prerelease_channel.npm_dist_tag, "next");
assert.equal(release.development_channel.publishable, false);
assert(release.stable_channel.target_minimum_interval_days >= 14);
const versionSource = { schema: "deepbom.release_version.v1", base_version: "2.0.0", channel: "release" };
assert.deepEqual(pickVersion(resolveVersionContract(versionSource)), {
  channel: "release", displayVersion: "2.0.0", pythonVersion: "2.0.0", npmDistTag: "latest", publishable: true,
});
assert.deepEqual(pickVersion(resolveVersionContract({ ...versionSource, channel: "prerelease", prerelease: "rc.2" })), {
  channel: "prerelease", displayVersion: "2.0.0-rc.2", pythonVersion: "2.0.0rc2", npmDistTag: "next", publishable: true,
});
assert.deepEqual(pickVersion(resolveVersionContract({ ...versionSource, channel: "dev" })), {
  channel: "dev", displayVersion: "2.0.0-dev", pythonVersion: "2.0.0.dev0", npmDistTag: null, publishable: false,
});
assert.throws(() => resolveVersionContract({ ...versionSource, channel: "prerelease" }), /requires prerelease/);
assert.throws(() => resolveVersionContract({ ...versionSource, channel: "release", prerelease: "rc.1" }), /Only the prerelease/);

const capabilities = buildCliCapabilities("1.96.11", { defaultTarget: "android_mid_a55", deltaTargets: [] });
assert.deepEqual(capabilities.public_product_contracts, JSON.parse(JSON.stringify(PUBLIC_PRODUCT_CONTRACTS)));
const publicReadme = await readFile("docs/PUBLIC_README.md", "utf8");
const automation = await readFile("docs/CLI_AUTOMATION.md", "utf8");
assert.match(publicReadme, /Format maturity/i);
assert.match(automation, /Stable machine-readable contracts/i);
assert.match(`${publicReadme}\n${automation}`, /metadata\.component[\s\S]*components\[\]/i);
assert.doesNotMatch(`${publicReadme}\n${automation}`, /CycloneDX 2\.0|working group|pull request/i);

console.log("Public product contracts verified: six format maturity rows, stable machine surfaces, and separate stable/prerelease release policy.");

function pickVersion(contract) {
  return Object.fromEntries(["channel", "displayVersion", "pythonVersion", "npmDistTag", "publishable"].map((key) => [key, contract[key]]));
}
