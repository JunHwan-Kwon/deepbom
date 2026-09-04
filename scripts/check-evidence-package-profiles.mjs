import assert from "node:assert/strict";

import {
  buildEvidencePackageProfileFiles,
  EVIDENCE_PACKAGE_PROFILES,
  validateEvidencePackageProfileFiles,
} from "../web/lib/evidence-package-profiles.js";
import {
  classifyEvidenceClass,
  EVIDENCE_LEVEL_PROFILES,
  validateEvidenceLevelManifest,
} from "../web/lib/evidence-level-report.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const SHA256 = "a".repeat(64);
const analysis = {
  filename: "profile-fixture.tflite",
  format: "tflite",
  model_sha256: SHA256,
  file_size_bytes: 32,
  operator_count: 1,
  tensor_count: 2,
  total_macs: 8,
  mac_assessment: { status: "assessed", compute_ops: 1, assessed_compute_ops: 1 },
  target_profile: { id: "rpi5_a76", name: "RPi 5 Cortex-A76", profile_sha256: "b".repeat(64) },
  inputs: [{ index: 0, name: "image", dtype: "UINT8", shape: [1, 1], scale_sample: [0.5], zero_point_sample: [128] }],
  outputs: [{ index: 1, name: "score", dtype: "UINT8", shape: [1, 1], scale_sample: [0.25], zero_point_sample: [0] }],
  quantization_status: { classification: "full_integer", label: "Full integer", summary: "Fixture", detail: "Fixture" },
  findings: [
    { id: "F-OBS", title: "Observed", priority: "Low", evidence_class: "OBSERVED", observation: "Observed fact." },
    { id: "F-DER", title: "Derived", priority: "Low", evidence_class: "DERIVED", observation: "Derived fact." },
    { id: "F-EST", title: "Estimated", priority: "Low", evidence_class: "ESTIMATED", observation: "Estimated fact." },
    { id: "F-RUN", title: "Runtime", priority: "Low", evidence_class: "OBSERVED_RUNTIME", observation: "Runtime fact." },
    { id: "F-MISSING", title: "Missing class", priority: "Low", observation: "Must fail closed." },
  ],
  ops: [{ index: 0, name: "CONV_2D", macs: 8 }],
  tensors: [
    { index: 0, name: "image", dtype: "UINT8", shape: [1, 1] },
    { index: 1, name: "score", dtype: "UINT8", shape: [1, 1] },
  ],
};
const metricCoverage = {
  schema: "deepbom.metric_coverage_manifest.fixture",
  entries: [
    { metric_id: "artifact.identity", label: "Artifact identity", status: "assessed", evidence_class: "OBSERVED", report_section: "Artifact" },
    { metric_id: "compute.macs", label: "MACs", status: "assessed", evidence_class: "DERIVED", report_section: "Compute" },
    { metric_id: "deployment.prediction", label: "Prediction", status: "assessed", evidence_class: "PREDICTED/DERIVED", report_section: "Deployment" },
    { metric_id: "runtime.assignment", label: "Runtime", status: "assessed", evidence_class: "OBSERVED_RUNTIME", report_section: "Runtime" },
    { metric_id: "validation.task_quality", label: "Task quality", status: "not_assessed", evidence_class: "NOT_ASSESSED", report_section: "Conclusion" },
  ],
};
const context = { generatedAt: "2026-08-24T00:00:00.000Z" };
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

for (const profile of EVIDENCE_PACKAGE_PROFILES) {
  for (const level of EVIDENCE_LEVEL_PROFILES) {
    const label = `${profile.id}/${level.id}`;
    const reportBody = profile.id === "engineering" ? "ENGINEERING REPORT\nExact engineering body."
      : profile.id === "regulatory" ? "REGULATORY SUPPORT REPORT\nExact support body."
        : null;
    const files = buildEvidencePackageProfileFiles({
      profileId: profile.id,
      evidenceLevelId: level.id,
      analysis,
      context,
      scope,
      metricCoverage,
      origin: "https://deepbom.org",
      reportBody,
    });
    assert.equal(validateEvidencePackageProfileFiles(files, { profileId: profile.id }), true, `${label} member contract`);
    assert.equal(files.length, new Set(files.map((file) => file.name)).size, `${label} unique members`);
    assert(files.every((file) => !/\.(tflite|onnx|gguf|safetensors|mlmodel)$/i.test(file.name)), `${label} excludes model payloads`);
    const byName = new Map(files.map((file) => [file.name, String(file.data)]));
    const manifestName = profile.id === "public" ? "public_report_verification_manifest.json" : "verification_manifest.json";
    const manifest = JSON.parse(byName.get(manifestName));
    const packageScope = JSON.parse(byName.get("package_scope.json"));
    const levelManifest = JSON.parse(byName.get("evidence_level_manifest.json"));
    const bodyName = profile.id === "public" ? "public_static_evidence_summary.txt"
      : profile.id === "machine_readable" ? "evidence_body.txt" : `reports/${profile.id}_report.txt`;
    assert.equal(manifest.report.body_sha256, sha256TextHex(byName.get(bodyName)), `${label} exact report-body binding`);
    assert.equal(validateEvidenceLevelManifest(levelManifest), true, `${label} evidence-level manifest`);
    assert.equal(packageScope.evidence_level, level.id, `${label} package level binding`);
    assert(levelManifest.included_metrics.every((row) => classifyEvidenceClass(row.evidence_class).rank <= level.rank), `${label} metric ceiling`);
    assert(levelManifest.included_findings.every((row) => classifyEvidenceClass(row.evidence_class).rank <= level.rank), `${label} finding ceiling`);
    assert(levelManifest.excluded_findings.some((row) => row.finding_id === "F-MISSING"
      && row.exclusion_reason === "unclassified_evidence_class_fail_closed"), `${label} missing finding class fails closed`);
    const standardsEvidence = files.filter((file) => file.name.endsWith(".cdx.json"));
    assert.equal(standardsEvidence.length, level.id === "all_available" ? 1 : 0, `${label} CycloneDX scope`);
    assert(files.every((file) => !file.name.includes("cyclonedx_2_0")), `${label} excludes draft-status artifacts`);
    if (level.id !== "all_available") assert.match(byName.get(bodyName), /^# DEEPBOM Scoped Evidence Report/m, `${label} scoped body`);
    assert.equal(packageScope.login_required, false, `${label} login-free contract`);
    assert.equal(packageScope.rights.profile, profile.id, `${label} rights profile binding`);
    assert.equal(packageScope.rights.license_grant, "none", `${label} no implicit license grant`);
    assert.match(byName.get("RIGHTS.txt"), /All rights reserved/i, `${label} copyright notice`);
    assert.match(byName.get("RIGHTS.txt"), /removal or concealment of attribution/i, `${label} provenance concealment boundary`);
    assert.match(byName.get("RIGHTS.txt"), /applicable law/i, `${label} applicable-law boundary`);
    assert.deepEqual(Object.keys(packageScope.member_roles).sort(), files.map((file) => file.name).sort(), `${label} complete member-role ledger`);
    assert.match(packageScope.integrity_boundary, /(?:cannot|do not) prevent copying or editing/i);
    const htmlMembers = files.filter((file) => file.name.endsWith(".html"));
    assert.equal(htmlMembers.length, profile.id === "machine_readable" ? 0 : 1, `${label} presentation cardinality`);
    if (htmlMembers.length) {
      assert.match(String(htmlMembers[0].data), /Report-body SHA-256/);
      assert.match(String(htmlMembers[0].data), /(?:cannot|do(?:es)? not) prevent copying/i);
    }
  }
}

console.log("Evidence Package profiles passed (four review profiles x four cumulative evidence levels; fail-closed metric/finding ceilings, exact body hashes, scoped CycloneDX omission, and no model payloads)." );
