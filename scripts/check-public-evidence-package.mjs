import assert from "node:assert/strict";

import {
  buildPublicEvidencePackageFiles,
  PUBLIC_EVIDENCE_PACKAGE_SCHEMA,
  validatePublicEvidencePackageFiles,
} from "../web/lib/public-evidence-package.js";
import { validateCycloneDx20ParameterContractPreview } from "../web/lib/cyclonedx-20-preview.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

const SHA256 = "a".repeat(64);
const TARGET_SHA256 = "b".repeat(64);
const analysis = {
  filename: "public-package-model.tflite",
  format: "tflite",
  model_sha256: SHA256,
  file_size_bytes: 16,
  operator_count: 1,
  tensor_count: 2,
  total_macs: 4,
  mac_assessment: { status: "assessed", compute_ops: 1, assessed_compute_ops: 1 },
  target_profile: { id: "rpi5_a76", name: "RPi 5 Cortex-A76", profile_sha256: TARGET_SHA256 },
  inputs: [{ index: 0, name: "image", dtype: "UINT8", shape: [1, 1, 1, 1], scale_sample: [0.5], zero_point_sample: [128] }],
  outputs: [{ index: 1, name: "score", dtype: "UINT8", shape: [1, 1], scale_sample: [0.25], zero_point_sample: [0] }],
  quantization_status: {
    classification: "full_integer",
    label: "Full integer",
    full_integer: true,
    summary: "All assessed compute is integer.",
    detail: "Fixture",
    compute_ops: 1,
    quantized_compute_ops: 1,
    quantized_compute_mac_percent: 100,
    quantize_ops: 0,
    dequantize_ops: 0,
  },
  findings: [],
  ops: [{ index: 0, name: "CONV_2D", macs: 4 }],
  tensors: [
    { index: 0, name: "image", dtype: "UINT8", shape: [1, 1, 1, 1] },
    { index: 1, name: "score", dtype: "UINT8", shape: [1, 1] },
  ],
};
const scope = {
  label: "TFLite",
  completion: "TFLite static deployment audit run complete",
  evidenceClass: "STATIC ARTIFACT EVIDENCE",
  depth: "Deep graph and deployment-model audit",
  assessed: "Artifact graph and contracts",
  runtimeStatus: "Runtime execution not observed in this run",
  runtimeBoundary: "Runtime assignment requires imported evidence.",
  releaseStatus: "Release readiness not assessed.",
};
const files = buildPublicEvidencePackageFiles({
  analysis,
  context: { generatedAt: "2026-08-24T00:00:00.000Z" },
  scope,
  origin: "https://deepbom.org",
});

assert.equal(files.length, 8);
assert.equal(new Set(files.map((file) => file.name)).size, files.length);
const byName = new Map(files.map((file) => [file.name, file.data]));
const packageScope = JSON.parse(byName.get("package_scope.json"));
assert.equal(packageScope.schema, PUBLIC_EVIDENCE_PACKAGE_SCHEMA);
assert.equal(packageScope.artifact.sha256, SHA256);
assert(packageScope.excluded_content.includes("model artifact bytes and weights"));
assert.match(byName.get("README.txt"), /local_browser signer as integrity evidence only/i);
assert.match(byName.get("RIGHTS.txt"), /All rights reserved/i);
assert.match(byName.get("RIGHTS.txt"), /removal or concealment of attribution/i);
assert.equal(packageScope.rights.profile, "public");
assert.equal(packageScope.rights.license_grant, "none");
assert.match(byName.get("public_static_evidence_summary.html"), /DEEPBOM PUBLIC COPY/);
assert.equal(validatePublicEvidencePackageFiles(files, packageScope), true);

const cycloneDx17 = JSON.parse(byName.get("cyclonedx_1_7_artifact_evidence.cdx.json"));
assertCycloneDx17(cycloneDx17, "public evidence package CycloneDX 1.7");
const preview = JSON.parse(byName.get("proposal/cyclonedx_2_0_parameter_contract.preview.cdx.json"));
const previewValidation = validateCycloneDx20ParameterContractPreview(preview);
assert.equal(previewValidation.valid, true, previewValidation.errors.join("; "));

const tamperedScope = structuredClone(packageScope);
tamperedScope.report_body_sha256 = "0".repeat(64);
assert.equal(validatePublicEvidencePackageFiles(files, tamperedScope), false);

console.log("Public Evidence Package passed (watermarked summary, manifest, CycloneDX 1.7/2.0 preview, and claim boundary).");
