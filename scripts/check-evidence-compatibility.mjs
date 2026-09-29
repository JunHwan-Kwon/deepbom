import assert from "node:assert/strict";
import { readFile, access, mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Ajv2020 from "ajv/dist/2020.js";
import { buildSnapshot, generate, DIRECTORY, encode, sha256, compareVersions } from "./generate-evidence-compatibility.mjs";
import { EVIDENCE_IR_LAYERS } from "../web/lib/evidence-ir.js";
import { PUBLIC_PRODUCT_CONTRACTS } from "../web/lib/public-product-contracts.js";
import { AUDIT_OUTPUT_FORMATS } from "../web/lib/audit-output-contracts.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildSpdxArtifactDocument } from "../web/lib/spdx-artifact-export.js";
import { buildProvenanceIr } from "../web/lib/provenance-ir.js";
import { metadataTemplate } from "../web/lib/provenance/omop.js";
import { projectProvenanceToCycloneDx } from "../web/lib/provenance/cyclonedx.js";
import { buildModelSummary } from "../web/lib/model-summary.js";

const catalog = await buildSnapshot(), nodes = new Map(catalog.nodes.map(node => [node.id, node]));
const schema = JSON.parse(await readFile("docs/schemas/deepbom-evidence-compatibility-v1.schema.json", "utf8"));
const validate = new Ajv2020({ strict: true, allErrors: true }).compile(schema);
assert(validate(catalog), JSON.stringify(validate.errors));
assert.equal(nodes.size, catalog.nodes.length, "unique endpoint IDs");
assert.equal(new Set(catalog.mappings.map(row => row.id)).size, catalog.mappings.length, "unique mapping IDs");
assert.equal(new Set(catalog.source_files.map(row => row.path)).size, catalog.source_files.length);
assert.equal(compareVersions("0.9.0", "0.10.0"), -1);
assert.equal(compareVersions("1.0.0", "0.99.99"), 1);
assert.throws(() => compareVersions("01.0.0", "1.0.0"));
const definitions = new Map();
for (const node of catalog.nodes) {
  assert(catalog.mappings.some(row => row.from === node.id || row.to === node.id), `orphan endpoint ${node.id}`);
  if (!node.json_schema) continue;
  const url = new URL(node.json_schema);
  assert.equal(url.origin, "https://deepbom.org");
  const file = `docs${url.pathname}`, document = JSON.parse(await readFile(file, "utf8"));
  const pointer = url.hash || document.$ref || "#";
  let definition = document;
  for (const key of pointer.slice(2).split("/").filter(Boolean)) definition = definition[key];
  assert(definition, `schema reference must resolve: ${node.json_schema}`); definitions.set(node.id, definition);
}
for (const layer of EVIDENCE_IR_LAYERS) {
  assert.equal(nodes.get(layer.id).contract, layer.schema);
  assert.equal(nodes.get(layer.id).digest_field, layer.digest_field);
}
for (const row of PUBLIC_PRODUCT_CONTRACTS.format_maturity) assert.deepEqual(nodes.get(`input-${row.format}`).maturity, row);
for (const output of AUDIT_OUTPUT_FORMATS) assert(nodes.has(`output-${output}`), `all CLI audit formats: ${output}`);
for (const row of catalog.mappings) {
  assert(nodes.has(row.from) && nodes.has(row.to), `closed references: ${row.id}`);
  assert(row.implementation.length > 0);
  if (row.status !== "proposed") assert(row.fields.length && row.checks.length, `implemented route needs fields and regression references: ${row.id}`);
  for (const file of [...row.implementation, ...row.checks]) { assert(!file.includes("..") && !file.startsWith("/")); await access(file); }
  for (const field of row.fields) for (const [endpoint, path] of [[row.from, field.source], [row.to, field.target]]) {
    if (nodes.get(endpoint).kind !== "ir" || path === "(whole document)") continue;
    assert(path.startsWith("/"), `IR selector: ${row.id} ${path}`);
    assert(Object.hasOwn(definitions.get(endpoint).properties, path.slice(1)), `IR field exists in member schema: ${row.id} ${path}`);
  }
}
for (const format of ["gguf", "safetensors", "hdf5", "pytorch_checkpoint"]) assert(!catalog.mappings.find(row => row.id === `${format}-artifact`).fields.some(field => field.target === "/graph"), "no graph promise for containers");
assert.equal(catalog.mappings.find(row => row.id === "provenance-tea").status, "proposed");
assert.equal(nodes.get("output-spdx").contract, "SPDX-2.3");
assert(!nodes.has("output-spdx3"));
const invalid = structuredClone(catalog); invalid.mappings[0].status = "certified"; assert.equal(validate(invalid), false);

// Check concrete identity and storage mappings against a real ONNX artifact.
const bytes = new Uint8Array(await readFile("web/samples/sample_cnn_float.onnx")), digest = sha256(bytes);
const analysis = analyzeOnnxModel(bytes, "sample_cnn_float.onnx"); analysis.model_sha256 = digest;
const context = getArtifactIrContext(analysis, { filename: "sample_cnn_float.onnx", format: "onnx", sha256: digest, size: bytes.length });
assert.equal(context.model_ir.source_contract.sha256, context.artifact_ir.artifact_ir_sha256);
assert.equal(context.model_ir.program.operations.length, context.artifact_ir.graph.operators.length);
assert.equal(context.model_ir.tensors_and_storage.storage_objects.length, context.artifact_ir.storage_topology.objects.length);
const summary = buildModelSummary(context.model_ir);
assert.equal(summary.totals.operation_count, context.model_ir.program.operations.length);
assert.equal(summary.totals.storage_object_count, context.model_ir.tensors_and_storage.storage_objects.length);
const envelope = buildArtifactEvidenceEnvelope(analysis, { filename: "sample_cnn_float.onnx", sha256: digest });
const spdx = buildSpdxArtifactDocument(envelope, { generatedAt: "2026-09-29T00:00:00Z" });
assert.equal(spdx.packages[0].checksums[0].checksumValue, digest);
assert.equal(spdx.spdxVersion, "SPDX-2.3");
assert.equal(spdx.packages[0].licenseConcluded, "NOASSERTION");
const input = metadataTemplate(context.model_ir, "generic");
input.nodes.push({ id: "dataset", kind: "dataset", name: "Declared training data" });
input.relationships.push({ id: "training", from: "artifact:primary", to: "dataset", role: "uses_training_data" });
const provenance = buildProvenanceIr(context.model_ir, input);
const bom = projectProvenanceToCycloneDx({ bomFormat: "CycloneDX", specVersion: "1.7", metadata: { component: { "bom-ref": "model", hashes: [{ alg: "SHA-256", content: digest }] } } }, provenance, context.model_ir);
assert.equal(bom.metadata.component.modelCard.modelParameters.datasets[0].ref, bom.components[0]["bom-ref"]);
assert.equal(bom.properties.find(row => row.name === "deepbom:provenance:irSha256").value, provenance.provenance_ir_sha256);
assert.equal(provenance.verdict.metadata_truth_verified, false);

if (!process.argv.includes("--definition-only")) {
  await generate({ check: true });
  const index = JSON.parse(await readFile(`${DIRECTORY}/index.json`, "utf8"));
  for (const entry of index.snapshots) {
    const content = await readFile(`${DIRECTORY}/${entry.file}`, "utf8"); assert.equal(sha256(content), entry.sha256);
    const document = JSON.parse(content); assert(validate(document)); assert.equal(document.catalog_version, entry.version);
    if (entry.version === index.current) assert.equal(content, encode(catalog));
    if (process.argv.includes("--dist")) assert.equal(await readFile(`dist/schemas/compatibility/${entry.file}`, "utf8"), content);
  }
  if (process.argv.includes("--dist")) assert.equal(await readFile("dist/schemas/compatibility/index.json", "utf8"), await readFile(`${DIRECTORY}/index.json`, "utf8"));

  // Exercise same-version and archive guards in an isolated filesystem copy.
  const original = process.cwd(), temporary = await mkdtemp(path.join(os.tmpdir(), "deepbom-compatibility-"));
  try {
    for (const file of [...catalog.source_files.map(pin => pin.path), `${DIRECTORY}/index.json`, ...index.snapshots.map(entry => `${DIRECTORY}/${entry.file}`), `${DIRECTORY}/README.md`]) {
      await mkdir(path.dirname(path.join(temporary, file)), { recursive: true });
      await writeFile(path.join(temporary, file), await readFile(file));
    }
    process.chdir(temporary);
    const owner = "web/lib/artifact-ir.js", originalOwner = await readFile(owner);
    await writeFile(owner, Buffer.concat([originalOwner, Buffer.from("\n// changed source\n")]));
    await assert.rejects(generate({ check: true }), /immutable/);
    await writeFile(owner, originalOwner);
    const altered = structuredClone(index); altered.snapshots[0].sha256 = "0".repeat(64);
    await writeFile(`${DIRECTORY}/index.json`, encode(altered));
    await assert.rejects(generate({ check: true }), /removed or altered/);
    await writeFile(`${DIRECTORY}/index.json`, encode({ ...index, current: "999.0.0" }));
    await assert.rejects(generate({ check: true }), /cannot move backwards/);
  } finally { process.chdir(original); await rm(temporary, { recursive: true, force: true }); }
}
console.log("Compatibility: schema, endpoint/field closure, canonical capability parity, concrete ONNX/Provenance/SPDX bindings, archive digests and generated views passed.");
