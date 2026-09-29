import assert from "node:assert/strict";
import { readFile, readdir, access } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import Ajv2020 from "ajv/dist/2020.js";
import { EVIDENCE_IR_NAME, EVIDENCE_IR_LAYERS } from "../web/lib/evidence-ir.js";
import { buildProvenanceIr, validateProvenanceIr, provenanceSummary } from "../web/lib/provenance-ir.js";
import { metadataTemplate } from "../web/lib/provenance/omop.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildWeightIr } from "../web/lib/weight-ir.js";
import { buildActivationIr } from "../web/lib/activation-ir.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { model, node, tensor, valueInfo, float32 } from "./onnx-proto-fixture.mjs";
import { provenanceJson } from "../web/lib/provenance/cyclonedx.js";

// A retired name must not return through another channel or generated schema.
const retiredNames = /evidence_link_ir|evidence_link_input|evidence_link_summary|evidence_links|evidence-link-ir|evidence-links|EvidenceLinkIr|legacy_section/;
async function checkCurrentSources(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory() && !["vendor", "pkg"].includes(entry.name)) await checkCurrentSources(file);
    if (entry.isFile() && /\.(?:js|mjs|json)$/.test(entry.name)) assert(!retiredNames.test(await readFile(file, "utf8")), `Retired provenance contract in ${file}`);
  }
}
for (const directory of ["web", "bin"]) await checkCurrentSources(directory);
try { await access("worker"); await checkCurrentSources("worker"); } catch (error) { if (error.code !== "ENOENT") throw error; }
for (const file of ["web/lib/evidence-link-ir.js", "docs/schemas/deepbom-evidence-links-v1.schema.json", "docs/EVIDENCE_LINK_IR.md"]) {
  await assert.rejects(access(file), { code: "ENOENT" });
}

const generated = spawnSync(process.execPath, ["scripts/generate-evidence-ir-catalog.mjs", "--check"], { encoding: "utf8" });
assert.equal(generated.status, 0, generated.stderr);
const catalog = JSON.parse(await readFile("docs/evidence-ir/catalog.json", "utf8"));
assert.equal(catalog.name, EVIDENCE_IR_NAME);
assert.deepEqual(catalog.layers, EVIDENCE_IR_LAYERS);
assert.equal(new Set(catalog.layers.map(layer => layer.schema)).size, 5);

const ajv = new Ajv2020({ strict: true, allErrors: true });
for (const file of ["artifact-ir-v2", "model-ir-v1", "numerical-ir-v1", "provenance-ir-v1", "evidence-ir-v1"]) {
  ajv.addSchema(JSON.parse(await readFile(`docs/schemas/deepbom-${file}.schema.json`, "utf8")));
}
const validate = ajv.getSchema(catalog.json_schema);
assert(validate, "family entry point must resolve entirely from local files");

const bytes = new Uint8Array(model(
  [node("Add", "first", ["x", "w"], ["y"])],
  [tensor("w", 1, [4], float32([-2, 0, 1, 3]))],
  [valueInfo("x", 1, [4])], [valueInfo("y", 1, [4])], 13,
));
const analysis = analyzeOnnxModel(bytes, "family.onnx");
analysis.model_sha256 = sha256BytesHex(bytes);
const context = getArtifactIrContext(analysis, { filename: "family.onnx", format: "onnx", sha256: analysis.model_sha256, size: bytes.length });
const ir = context.model_ir;
const original = canonicalJson(ir);
const weight = await buildWeightIr(ir, analysis, bytes);
const input = ir.program.values.find(value => value.name === "x");
const output = ir.program.values.find(value => value.name === "y");
const configuration = { optimization: "disabled" };
// Synthetic contract fixture, not a claim that an inference engine ran.
const activation = buildActivationIr(ir, {
  schema: "deepbom.activation_capture.v1", source: weight.source,
  run: {
    id: "synthetic-family-fixture", entry_region_ref: input.region_ref, started_at: "2026-09-22T00:00:00Z",
    runtime: { name: "fixture", version: "1", configured_providers: ["CPU"], device: null },
    collector: { name: "test", version: "1", sha256: "a".repeat(64) },
    execution: { artifact_sha256: weight.source.artifact_sha256, instrumented_artifact_sha256: null, configuration, configuration_sha256: sha256TextHex(canonicalJson(configuration)) },
    probe: { kind: "synthetic_ones", description: "Synthetic contract fixture; no runtime measurement" }, runtime_evidence: null,
  },
  inputs: [{ value_ref: input.id, native_locator: "x", dtype: "FLOAT32", shape: [4], values: [1, 1, 1, 1] }],
  requested_value_refs: [output.id],
  captures: [{ value_ref: output.id, native_locator: "y", dtype: "FLOAT32", shape: [4], values: [-1, 1, 2, 4] }], missing: [],
});
const metadata = metadataTemplate(ir, "generic");
const provenance = buildProvenanceIr(ir, metadata);
assert.equal(buildProvenanceIr(ir, metadata).provenance_ir_sha256, provenance.provenance_ir_sha256);
assert.equal(canonicalJson(ir), original, "optional layers must not alter their common source");
assert.deepEqual(validateProvenanceIr(provenance, ir), provenance);
const deliveryDigest = sha256BytesHex(new TextEncoder().encode(provenanceJson(provenance)));
assert.notEqual(deliveryDigest, provenance.provenance_ir_sha256, "delivery checksum covers complete exported bytes, not the IR exclusion hash");
assert.notEqual(deliveryDigest, provenance.source.artifact_sha256, "delivery checksum is not the model checksum");

const documents = [context.artifact_ir, ir, weight, activation, provenance];
for (const [index, document] of documents.entries()) {
  const layer = catalog.layers[index];
  assert.equal(document.schema, layer.schema);
  assert.match(document[layer.digest_field], /^[a-f0-9]{64}$/);
  assert(validate(document), `${layer.name}: ${JSON.stringify(validate.errors)}`);
  const invalid = structuredClone(document); delete invalid[layer.digest_field];
  assert.equal(validate(invalid), false, `${layer.name}: missing digest rejected`);
}
assert.equal(validate({ schema: "deepbom.unknown_ir.v1" }), false);
assert.equal(validate({ ...provenance, schema: "deepbom.evidence_link_ir.v1" }), false, "retired identities must be rejected");
const wrongDigestField = structuredClone(provenance);
wrongDigestField.evidence_link_ir_sha256 = wrongDigestField.provenance_ir_sha256;
delete wrongDigestField.provenance_ir_sha256;
assert.equal(validate(wrongDigestField), false);
const tampered = structuredClone(provenance); tampered.coverage.node_count++;
assert.throws(() => validateProvenanceIr(tampered, ir), "semantic checks remain required beyond JSON Schema");

// Optional build check: every referenced canonical schema must be served byte-for-byte.
if (process.argv.includes("--dist")) {
  for (const schema of Object.values(ajv.schemas).map(entry => entry.schema).filter(value => value.$id?.startsWith("https://deepbom.org/schemas/"))) {
    const name = new URL(schema.$id).pathname.split("/").pop();
    assert.equal(await readFile(`dist/schemas/${name}`, "utf8"), await readFile(`docs/schemas/${name}`, "utf8"));
  }
  assert.equal(await readFile("dist/schemas/evidence-ir-catalog.json", "utf8"), await readFile("docs/evidence-ir/catalog.json", "utf8"));
  assert.match(await readFile("dist/guides/evidence-ir/index.html", "utf8"), /Provenance IR/);
  assert.match(await readFile("dist/sitemap.xml", "utf8"), /https:\/\/deepbom\.org\/guides\/evidence-ir\//);
}
console.log("Evidence IR: five member schemas, canonical implementation, deterministic digests, no retired aliases, invalid identities and local reference closure passed.");
