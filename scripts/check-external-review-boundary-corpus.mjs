import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const ledger = JSON.parse(await readFile("corpus/external-review/golden-boundary-manifest.v1.json", "utf8"));
const packageJson = JSON.parse(await readFile("package.json", "utf8"));
const tierSource = await readFile("scripts/check-tier.mjs", "utf8");

assert.equal(ledger.schema, "deepbom.external_review_golden_boundary_manifest.v1");
assert.deepEqual(ledger.boundaries.map((row) => row.id), [
  "tflite_dynamic_and_transpose_conv",
  "gguf_f16_q4_q8_storage",
  "coreml_compressed_affine_contract",
  "onnx_invalid_external_byte_range",
]);

for (const boundary of ledger.boundaries) {
  const bytes = await readFile(boundary.manifest);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), boundary.manifest_sha256, `${boundary.id} manifest digest drifted.`);
  const manifest = JSON.parse(bytes.toString("utf8"));
  const paths = new Set([
    ...(manifest.fixtures || []).map((row) => row.path),
    ...(manifest.files || []).map((row) => row.path),
  ]);
  for (const required of boundary.required_cases) assert(paths.has(required), `${boundary.id} is missing ${required}.`);
  assert.match(tierSource, new RegExp(escapeRegex(boundary.verification_script)), `${boundary.verification_script} must remain in a quality tier.`);
  assert(Object.values(packageJson.scripts).some((command) => command.includes(boundary.verification_script)), `${boundary.verification_script} must remain directly invocable from package scripts.`);
}

console.log(`External-review boundary corpus verified: ${ledger.boundaries.length} hash-bound boundary families with format-tier regressions.`);

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
