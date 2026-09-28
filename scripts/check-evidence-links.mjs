import Ajv2020 from "ajv/dist/2020.js";
import { evidenceLinkSummary } from "../web/lib/evidence-link-ir.js";
import { validateEvidenceLinkSummary } from "../web/lib/evidence-links/summary.js";
import assert from "node:assert/strict";
import { readFile, mkdtemp, writeFile, mkdir, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildSingleFileArtifactSet } from "../web/lib/artifact-set.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { buildEvidenceLinkIr, validateEvidenceLinkIr } from "../web/lib/evidence-link-ir.js";
import { sealEvidence } from "../web/lib/ir-evidence-contract.js";
import { metadataTemplate, fillOmopTemplate } from "../web/lib/evidence-links/omop.js";
import { observeEvidenceBytes, parseMetadataText } from "../web/lib/evidence-links/files.js";
import { observeLocalEvidence } from "../bin/deepbom-evidence-links.mjs";
import { projectEvidenceLinksToCycloneDx } from "../web/lib/evidence-links/cyclonedx.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

const filename = "sample_cnn_float.onnx", source = `web/samples/${filename}`;
const bytes = new Uint8Array(await readFile(source)), sha = sha256BytesHex(bytes);
const analysis = analyzeOnnxModel(bytes, filename); analysis.model_sha256 = sha;
analysis.artifact_set = buildSingleFileArtifactSet({ filename, format: "onnx", sha256: sha, byteLength: bytes.length });
const context = getArtifactIrContext(analysis, { filename, format: "onnx", sha256: sha, size: bytes.length });
const model = context.model_ir, original = canonicalJson(model);
const sourceRow = { CDM_SOURCE_NAME: "Synthetic metadata only", CDM_SOURCE_ABBREVIATION: "SYN", CDM_HOLDER: "Example institution", SOURCE_RELEASE_DATE: "2026-09-01", CDM_RELEASE_DATE: "2026-09-02", CDM_VERSION: "5.5", CDM_VERSION_CONCEPT_ID: 0, VOCABULARY_VERSION: "synthetic-v1", CDM_RELEASE_IDENTIFIER: "release-1" };
const metadata = fillOmopTemplate(metadataTemplate(model), sourceRow, { instanceId: "example:cdm" });
metadata.nodes = [
  { id: "cohort", kind: "cohort_definition", name: "Cohort definition", file: "cohort.json", sha256: sha256BytesHex(new TextEncoder().encode("{}")) },
  { id: "code", kind: "software", name: "Synthetic training script", version: "example-1" },
  { id: "train", kind: "run", name: "Declared training run" },
  { id: "quality", kind: "quality_report", name: "External quality report" },
];
metadata.relationships = [
  { id: "definition", from: "omop:release", to: "cohort", role: "defined_by" },
  { id: "training", from: "artifact:primary", to: "train", role: "generated_by" },
  { id: "data", from: "train", to: "omop:release", role: "uses_training_data", evidence_refs: ["cohort"] },
  { id: "software", from: "train", to: "code", role: "uses_code" },
  { id: "quality-ref", from: "omop:release", to: "quality", role: "has_quality_report" },
];
const observed = observeEvidenceBytes(metadata.nodes[0], new TextEncoder().encode("{}"));
const ir = buildEvidenceLinkIr(model, metadata, { observations: [observed] });
assert.equal(ir.verdict.status, "no_contradiction_observed");
assert.equal(ir.coverage.attested_relationship_count, 0);
assert.equal(ir.coverage.node_count, 6);
assert.equal(ir.coverage.field_count, Object.keys(sourceRow).length);
assert.equal(ir.coverage.check_count, Object.values(ir.coverage.check_counts).reduce((a, b) => a + b, 0));
assert(Object.isFrozen(ir.nodes[0]));
assert.deepEqual(validateEvidenceLinkIr(ir, model), ir);
const validateSchema = new Ajv2020({ strict: true, allErrors: true }).compile(JSON.parse(await readFile("docs/schemas/deepbom-evidence-links-v1.schema.json", "utf8")));
for (const document of [metadata, ir, evidenceLinkSummary(ir), metadataTemplate(model, "generic")]) assert(validateSchema(document), JSON.stringify(validateSchema.errors));
validateEvidenceLinkSummary(evidenceLinkSummary(ir), sha, model.model_ir_sha256);
assert.throws(() => validateEvidenceLinkSummary({ ...evidenceLinkSummary(ir), check_count: 999 }, sha, model.model_ir_sha256));
assert.throws(() => validateEvidenceLinkSummary({ ...evidenceLinkSummary(ir), patient_rows: [] }, sha, model.model_ir_sha256));
assert.equal(canonicalJson(model), original, "optional evidence never mutates Model IR");
assert.equal(buildEvidenceLinkIr(model, JSON.parse(canonicalJson(metadata)), { observations: [observed] }).evidence_link_ir_sha256, ir.evidence_link_ir_sha256);
assert.equal(buildEvidenceLinkIr(model, metadata).verdict.status, "incomplete");
const change = mutate => { const copy = structuredClone(metadata); mutate(copy); return copy; };
const result = mutate => buildEvidenceLinkIr(model, change(mutate), { observations: [observed] });
assert.equal(result(v => v.nodes[0].sha256 = "a".repeat(64)).verdict.status, "contradiction_observed");
assert.throws(() => result(v => v.subject.artifact_sha256 = "a".repeat(64)), /subject/);
assert.throws(() => result(v => v.subject.artifact_set_sha256 = "a".repeat(64)), /subject/);
assert.throws(() => result(v => v.subject.model_ir_sha256 = "a".repeat(64)), /Model IR/);
assert.throws(() => result(v => v.nodes.push(v.nodes[0])), /duplicated/);
assert.throws(() => result(v => v.relationships[0].id = "cohort"), /collides/);
assert.throws(() => result(v => v.nodes[0].file = "../escape.json"), /traversal/);
assert.throws(() => result(v => v.omop.cdm_source.source_release_date = "2026-02-30"), /real ISO/);
assert.throws(() => result(v => v.omop.cdm_source.CDM_VERSION = "5.4"), /colliding/);
assert.throws(() => result(v => v.omop.cdm_source.cdm_version = "5.4"), /contradicts/);
assert.throws(() => result(v => v.omop.release_id = "other"), /contradicts/);
assert.throws(() => result(v => v.omop.cdm_source.cdm_version_concept_id = 9007199254740992), /large integers/);
assert.throws(() => result(v => v.omop.cdm_source = [v.omop.cdm_source, v.omop.cdm_source]), /exactly one/);
const oneRow = result(v => v.omop.cdm_source = [v.omop.cdm_source]);
assert.equal(oneRow.coverage.field_count, ir.coverage.field_count);
assert(oneRow.field_ledger.every(row => row.source_pointer.startsWith("/omop/cdm_source/0/")));
assert.equal(result(v => v.omop.cdm_source.future_field = "preserve me").coverage.unsupported_field_count, 1);
assert.equal(result(v => v.omop.cdm_source.future_field = "preserve me").verdict.status, "incomplete");
assert.equal(result(v => { v.omop.cdm_version = v.omop.cdm_source.cdm_version = "6.0"; }).profile.status, "unsupported_cdm_version");
const legacy = result(v => { v.omop.cdm_version = v.omop.cdm_source.cdm_version = "5.4"; delete v.omop.cdm_source.cdm_release_identifier; });
assert.equal(legacy.profile.status, "supported_import_profile");
assert.equal(result(v => v.relationships[0].to = "missing").coverage.check_counts.unresolved, 1);
assert.equal(result(v => v.relationships[0].to = "code").verdict.status, "contradiction_observed");
assert.equal(result(v => v.nodes[1].attributes = [{ name: "unverified", value: true }]).verdict.status, "incomplete");
const forged = structuredClone(ir); forged.coverage.node_count++; delete forged.evidence_link_ir_sha256;
assert.throws(() => validateEvidenceLinkIr(sealEvidence(forged, "evidence_link_ir_sha256"), model), /reproduce/);
assert.throws(() => parseMetadataText('{"schema":"a","schema":"b"}'), /Duplicate|duplicate/);

const base = buildPublicCycloneDx17ArtifactContract(analysis, { hash: sha, fileSizeBytes: bytes.length, artifactIr: context.artifact_ir });
const projected = projectEvidenceLinksToCycloneDx(base, ir, model);
assertCycloneDx17(projected, "metadata projection");
const conflicting = projectEvidenceLinksToCycloneDx(base, result(v => v.nodes[0].sha256 = "a".repeat(64)), model);
const observedComponent = conflicting.components.find(node => node.properties?.some(row => row.name === "deepbom:evidenceLinks:nodeId" && row.value === "cohort"));
assert.equal(observedComponent.hashes[0].content, observed.sha256, "export preserves observed bytes over a contradicted declaration");
assert(observedComponent.properties.some(row => row.name === "deepbom:evidenceLinks:expectedSha256" && row.value === "a".repeat(64)));
assert.notEqual(projected.serialNumber, base.serialNumber);
const collisionBase = structuredClone(base);
collisionBase.components ||= [];
collisionBase.components.push({ type: "file", name: "collision", "bom-ref": `deepbom-link:${ir.evidence_link_ir_sha256}:cohort` });
assert.throws(() => projectEvidenceLinksToCycloneDx(collisionBase, ir, model), /duplicate/);
assert.equal(projected.metadata.component.hashes.find(v => v.alg === "SHA-256").content, sha);
assert.equal(projected.metadata.component.modelCard.modelParameters.datasets.length, 1);
assert.equal(projected.formulation[0].workflows[0].inputs.length, 2);
assert.throws(() => projectEvidenceLinksToCycloneDx(projected, ir, model), /already/);
const bomNode = { id: "bom", kind: "bom", name: "External model BOM", file: "bom.json", bom: { format: "CycloneDX", spec_version: "1.7", document_id: base.serialNumber, document_version: base.version, element_ref: base.metadata.component["bom-ref"] } };
const bomInput = metadataTemplate(model, "generic"); bomInput.nodes = [bomNode]; bomInput.relationships = [{ id: "bom-ref", from: "artifact:primary", to: "bom", role: "references_bom" }];
const resolveBom = value => buildEvidenceLinkIr(model, bomInput, { observations: [observeEvidenceBytes(bomNode, new TextEncoder().encode(JSON.stringify(value)))] });
const boundBom = resolveBom(base);
assert(boundBom.checks.some(row => row.kind === "bom_model_sha256" && row.status === "match"));
assert.equal(boundBom.coverage.unsupported_field_count, base.metadata.component.properties.length);
assert(boundBom.field_ledger.every(row => row.source_pointer.startsWith("/observations/0/document/metadata/component/properties/")));
const lying = structuredClone(base); lying.metadata.component.properties.push({ name: "mlbom:quantizationClassification", value: "fabricated" });
assert.equal(resolveBom(lying).verdict.status, "incomplete", "reference resolution must not certify unverified properties");
const other = structuredClone(base); other.version++;
assert.equal(resolveBom(other).verdict.status, "contradiction_observed");
const hashLie = structuredClone(base); hashLie.metadata.component.hashes = [{ alg: "SHA-256", content: "0".repeat(64) }];
assert.equal(resolveBom(hashLie).verdict.status, "contradiction_observed");
const duplicate = structuredClone(base); duplicate.components ||= []; duplicate.components.push(duplicate.metadata.component);
assert.equal(resolveBom(duplicate).verdict.status, "contradiction_observed");

const dir = await mkdtemp(path.join(tmpdir(), "deepbom-evidence-links-"));
try {
  await writeFile(path.join(dir, "cohort.json"), "{}");
  assert.deepEqual(await observeLocalEvidence(metadata, dir), [observed]);
  await mkdir(path.join(dir, "safe")); await symlink(path.join(dir, "cohort.json"), path.join(dir, "safe", "cohort.json"));
  await assert.rejects(() => observeLocalEvidence(metadata, path.join(dir, "safe")), /outside/);
  await writeFile(path.join(dir, "metadata.json"), JSON.stringify(metadata));
  const run = args => spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", source, ...args], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  const ptd = path.join(dir, "example.ptd");
  await writeFile(ptd, Buffer.from(await readFile("scripts/fixtures/numerical-tensor.ptd.base64.txt", "utf8"), "base64"));
  for (const artifact of [source, "web/samples/mobilenet_v2_1.0_224_quant.tflite", "web/samples/tinymqa1m.Q4_0.gguf", "web/samples/nanofable-1m-fp16.safetensors", "web/samples/MNISTClassifier.mlmodel", ptd]) {
    const generated = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", artifact, "--metadata-template", "generic"], { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 });
    assert.equal(generated.status, 0, `${artifact}: ${generated.stderr}`);
    const inputFile = path.join(dir, "format-metadata.json"); await writeFile(inputFile, generated.stdout);
    const tested = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", artifact, "--metadata", inputFile, "--section", "evidence_link_ir", "--output-format", "json-compact"], { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 });
    assert.equal(tested.status, 0, `${artifact}: ${tested.stderr}`);
    assert.equal(JSON.parse(tested.stdout).sections.evidence_link_ir.verdict.status, "no_contradiction_observed");
  }
  const template = run(["--metadata-template", "omop"]); assert.equal(template.status, 0, template.stderr); assert.equal(JSON.parse(template.stdout).subject.artifact_sha256, sha);
  const linked = run(["--metadata", path.join(dir, "metadata.json"), "--evidence-files", dir, "--output-format", "analysis", "--section", "evidence_link_ir"]);
  assert.equal(linked.status, 0, linked.stderr); const emitted = JSON.parse(linked.stdout); assert.equal(emitted.sections.evidence_link_ir.source.artifact_sha256, sha); assert.deepEqual(emitted.sections.evidence_link_ir.nodes, ir.nodes); assert.deepEqual(emitted.sections.evidence_link_ir.coverage, ir.coverage);
  const missing = run(["--metadata", path.join(dir, "metadata.json"), "--summary"]); assert.equal(missing.status, 3, missing.stderr); assert.match(missing.stdout, /Metadata links: incomplete/);
  const cdx = run(["--metadata", path.join(dir, "metadata.json"), "--evidence-files", dir, "--output-format", "cyclonedx"]); assert.equal(cdx.status, 0, cdx.stderr); assertCycloneDx17(JSON.parse(cdx.stdout), "CLI linked export");
} finally { await rm(dir, { recursive: true, force: true }); }
console.log("Evidence links: binding, OMOP profiles, exact counts, tampering, unmapped claims, BOM identity, schema, file isolation and CLI checks passed.");
