import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

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
assert.equal(envelope.envelope_sha256, expected.artifact_evidence_envelope_sha256);
console.log(`Expected output verified: ${expected.artifact.sha256.slice(0, 12)}..., ${analysis.operator_count} ops, ${analysis.total_macs} MACs.`);

function run(args) {
  return JSON.parse(execFileSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 }));
}
