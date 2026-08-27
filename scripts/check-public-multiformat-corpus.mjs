import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

import { canonicalJson } from "../web/lib/report-utils.js";
import { primaryArtifactSha256, readPublicMultiformatCorpus } from "./public-multiformat-corpus-lib.mjs";

const manifestPath = "corpus/public-multiformat-corpus.v1.json";
const evidencePath = "corpus/cyclonedx-generalization-evidence.v1.json";
const manifest = await readPublicMultiformatCorpus(manifestPath);
const evidence = JSON.parse(await readFile(evidencePath, "utf8"));
assert(evidence.schema === "deepbom.cyclonedx_generalization_evidence.v1", "Generalization evidence schema changed.");
const { ledger_sha256: ledger, ...body } = evidence;
assert(ledger === sha256(canonicalJson(body)), "Generalization evidence ledger SHA-256 is stale.");
assert(evidence.population.manifest_sha256 === sha256(await readFile(manifestPath)), "Evidence is not bound to the current corpus manifest.");
assert(evidence.population.path_record_count === manifest.summary.path_record_count, "Path-record denominator changed.");
assert(evidence.population.unique_primary_artifact_count === manifest.summary.unique_primary_artifact_count, "Unique-byte denominator changed.");
assert(evidence.records.path === "corpus/cyclonedx-generalization-evidence.records.v1.json.gz", "Per-artifact evidence path changed or escaped its reviewed repository location.");
assert(evidence.records.record_count === manifest.summary.unique_primary_artifact_count, "Declared per-artifact record count changed.");
const compressedRecords = await readFile(evidence.records.path);
assert(compressedRecords.length === evidence.records.compressed_bytes, "Compressed per-artifact evidence byte length changed.");
assert(sha256(compressedRecords) === evidence.records.compressed_sha256, "Compressed per-artifact evidence SHA-256 changed.");
const recordsJson = gunzipSync(compressedRecords);
assert(recordsJson.length === evidence.records.uncompressed_bytes, "Uncompressed per-artifact evidence byte length changed.");
assert(sha256(recordsJson) === evidence.records.uncompressed_sha256, "Uncompressed per-artifact evidence SHA-256 changed.");
const records = JSON.parse(recordsJson.toString("utf8"));
assert(records.schema === evidence.records.schema, "Per-artifact evidence schema changed.");
const { ledger_sha256: recordsLedger, ...recordsBody } = records;
assert(recordsLedger === evidence.records.ledger_sha256 && recordsLedger === sha256(canonicalJson(recordsBody)), "Per-artifact evidence ledger is stale.");
assert(records.manifest_sha256 === evidence.population.manifest_sha256, "Per-artifact evidence is not manifest-bound.");
assert(records.artifacts.length === manifest.summary.unique_primary_artifact_count, "Per-artifact evidence coverage is incomplete.");
assert(evidence.artifact_index.length === records.artifacts.length, "Per-artifact evidence index coverage is incomplete.");
assert(new Set(records.artifacts.map((artifact) => artifact.artifact_sha256)).size === records.artifacts.length, "Detailed per-artifact evidence repeats a primary SHA-256.");
assert(new Set(evidence.artifact_index.map((artifact) => artifact.artifact_sha256)).size === evidence.artifact_index.length, "Per-artifact evidence index repeats a primary SHA-256.");
assert(manifest.summary.format_unique_counts.onnx >= 40 && manifest.summary.format_unique_counts.gguf >= 20
  && manifest.summary.format_unique_counts.safetensors >= 18 && manifest.summary.format_unique_counts.coreml >= 27,
"Public real-file format strata regressed below the reviewed minimum population.");
assert(manifest.summary.bound_onnx_sidecar_record_count >= 10, "Bound ONNX external-data coverage regressed.");
assert(manifest.selection_protocol.coreml.includes("every model asset linked"), "Core ML catalog enumeration boundary changed.");
assert(manifest.artifacts.every((artifact) => artifact.source.kind !== "generated_fixture"), "A generated fixture entered the public real-file population.");
const evidenceBySha = new Map(records.artifacts.map((artifact) => [artifact.artifact_sha256, artifact]));
const indexBySha = new Map(evidence.artifact_index.map((artifact) => [artifact.artifact_sha256, artifact]));
for (const artifact of manifest.artifacts) {
  const observed = evidenceBySha.get(primaryArtifactSha256(artifact));
  assert(observed && observed.format === artifact.format, `${artifact.id}: per-artifact evidence is absent or format-mismatched.`);
  const indexed = indexBySha.get(observed.artifact_sha256);
  assert(indexed?.analysis_sha256 === observed.analysis_sha256 && indexed?.receipt_sha256 === observed.receipt_sha256,
    `${artifact.id}: per-artifact index does not bind the detailed record digests.`);
  assert(indexed?.serialized_contract_status === observed.analysis_summary.serialized_contract_status,
    `${artifact.id}: per-artifact index and detailed contract status diverge.`);
  assert(observed.cyclonedx_observation?.artifact_sha256 === observed.artifact_sha256, `${artifact.id}: observation identity is not byte-bound.`);
  const assessment = observed.cyclonedx_observation?.serialized_contract_assessment;
  assert(assessment?.issue_count === (assessment?.issues || []).reduce((sum, row) => sum + Number(row.count || 0), 0), `${artifact.id}: serialized-contract issue count does not conserve its rows.`);
  assert(observed.analysis_summary.serialized_contract_issue_count === assessment.issue_count, `${artifact.id}: compact and detailed contract assessments diverge.`);
  assert(JSON.stringify(observed.analysis_summary.serialized_contract_issue_codes) === JSON.stringify(assessment.issues.map((row) => row.code)), `${artifact.id}: compact contract issue codes diverge.`);
  const allowedEvidenceClasses = new Set(["OBSERVED", "OBSERVED_DERIVED", "EXTERNAL", "UNAVAILABLE", "NOT_APPLICABLE", "RUNTIME_REQUIRED", "OUT_OF_SCOPE"]);
  assert(observed.cyclonedx_observation.field_evidence.every((row) => allowedEvidenceClasses.has(row.evidence_class)), `${artifact.id}: field evidence contains an undeclared class.`);
  const graphField = observed.cyclonedx_observation.field_evidence.find((row) => row.path === "model.graph");
  if (observed.analysis_summary.graph_presence === "serialized_graph_payload_not_decoded") {
    assert(graphField?.evidence_class === "UNAVAILABLE", `${artifact.id}: an undecoded serialized graph was mislabeled as not applicable.`);
  }
}
for (const [format, population] of Object.entries(evidence.format_populations)) {
  assert(population.unique_artifact_count === manifest.summary.format_unique_counts[format], `${format}: format denominator does not conserve the manifest.`);
  assert(population.affine_external_parameter_count + population.unquantized_or_undeclared_external_parameter_count === population.external_parameter_count,
    `${format}: external parameter classification does not conserve its denominator.`);
  assert(Object.values(population.serialized_contract_status).reduce((sum, count) => sum + count, 0) === population.unique_artifact_count,
    `${format}: serialized-contract status does not conserve the format denominator.`);
}
const evidenceClassFinding = evidence.cross_format_schema_findings.find((row) => row.id === "evidence-class-separation");
const validityFinding = evidence.cross_format_schema_findings.find((row) => row.id === "validity-is-not-availability-or-runtime-evidence");
assert(evidenceClassFinding?.observed_denominator === evidence.population.unique_primary_artifact_count, "Evidence-class finding denominator diverged.");
assert(validityFinding?.observed_denominator === evidence.population.unique_primary_artifact_count, "Validity finding denominator diverged.");
console.log(`Public multiformat corpus evidence passed (${manifest.summary.path_record_count} path / ${manifest.summary.unique_primary_artifact_count} unique artifacts).`);

function assert(value, message) { if (!value) throw new Error(message); }
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
