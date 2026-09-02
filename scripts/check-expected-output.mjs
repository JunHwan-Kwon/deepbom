import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { ANALYZER_BUILD_SOURCE_STATE } from "../web/lib/build-metadata.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const expected = JSON.parse(await readFile("examples/expected-output/gpu-partition-probe.expected.json", "utf8"));
const bytes = await readFile(expected.artifact.path);
assert.equal(bytes.byteLength, expected.artifact.byte_length, "artifact byte length");
assert.equal(createHash("sha256").update(bytes).digest("hex"), expected.artifact.sha256, "artifact SHA-256");

const analysis = run(["audit", expected.artifact.path, "--compact"]);
assert.equal(analysis.format, expected.analysis.format);
assert.equal(analysis.operator_count, expected.analysis.operator_count);
assert.equal(analysis.tensor_count, expected.analysis.tensor_count);
assert.equal(analysis.total_macs, expected.analysis.total_macs);
assert.equal(analysis.mac_assessment?.status, expected.analysis.mac_assessment_status);
assert.equal(analysis.mac_assessment?.total_assessed_macs, expected.analysis.total_assessed_macs);
assert.equal(analysis.mac_assessment?.assessed_compute_ops, expected.analysis.assessed_compute_ops);
assert.equal(analysis.mac_assessment?.compute_ops, expected.analysis.compute_ops);
assert.equal(analysis.onnx_shape_inference?.status, expected.analysis.shape_inference_status);

const envelope = run(["audit", expected.artifact.path, "--format", "envelope", "--timestamp", expected.fixed_timestamp, "--compact"]);
assert.equal(envelope.identity.sha256, expected.artifact.sha256);
assert.equal(envelope.artifact_set?.artifact_set_sha256, expected.artifact_set_sha256);
const envelopeBody = structuredClone(envelope);
delete envelopeBody.envelope_sha256;
assert.equal(envelope.envelope_sha256, sha256TextHex(canonicalJson(envelopeBody)), "evidence envelope canonical SHA-256");
if (ANALYZER_BUILD_SOURCE_STATE === "clean") {
  assert.equal(envelope.envelope_sha256, expected.artifact_evidence_envelope_sha256, "clean release evidence envelope SHA-256");
} else {
  assert.equal(ANALYZER_BUILD_SOURCE_STATE, "working-tree-dirty", "known analyzer source state");
  const provenanceFindings = envelopeBody.findings.filter((finding) => finding.id === "EA-PROV-0001");
  assert.equal(provenanceFindings.length, 1, "dirty build provenance finding count");
  envelopeBody.findings = envelopeBody.findings.filter((finding) => finding.id !== "EA-PROV-0001");
  assert.equal(
    sha256TextHex(canonicalJson(envelopeBody)),
    expected.artifact_evidence_envelope_sha256,
    "dirty build envelope after removing the required provenance finding",
  );
}
console.log(`Expected output verified: ${expected.artifact.sha256.slice(0, 12)}..., ${analysis.operator_count} ops, ${analysis.total_macs} MACs.`);

function run(args) {
  return JSON.parse(execFileSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 }));
}
