import { buildPublicCycloneDxDocuments } from "./public-cyclonedx-export.js";
import {
  buildPublicAuditSummaryText,
  buildPublicEngineeringReportHtml,
} from "./session-evidence.js";
import {
  buildPublicVerificationManifest,
  validatePublicVerificationManifest,
} from "./public-verification-manifest.js";
import { jsonForDownload, zipTextFile } from "./report-utils.js";
import {
  buildEvidencePackageRights,
  evidencePackageRightsText,
  validateEvidencePackageRights,
} from "./evidence-package-rights.js";

export const PUBLIC_EVIDENCE_PACKAGE_SCHEMA = "deepbom.public_static_evidence_package.v1";

export function syncPublicEvidencePackageButton(button, { hasAnalysis, reportTargetReady, profileLabel = "Public" } = {}) {
  if (!button) return;
  button.disabled = !hasAnalysis || !reportTargetReady;
  button.classList.remove("account-locked");
  button.textContent = "Download Evidence Package";
  button.title = reportTargetReady
    ? `Download the login-free ${profileLabel} profile with a bound verification manifest and local-browser ES256 package signature.`
    : "Run or load the selected report target analysis first.";
}

const PACKAGE_MEMBERS = Object.freeze([
  ["README.txt", "Human-readable scope, verification steps, and license boundary."],
  ["RIGHTS.txt", "Human-readable copyright, permitted-purpose, provenance, and reuse boundary."],
  ["public_static_evidence_summary.txt", "Exact plain-text body bound by the verification manifest."],
  ["public_static_evidence_summary.html", "Watermarked presentation copy of the exact summary body."],
  ["public_report_verification_manifest.json", "Artifact, report, analyzer, rulepack, target, and runtime identity hashes."],
  ["cyclonedx_1_7_artifact_evidence.cdx.json", "Standalone CycloneDX 1.7 artifact-evidence document."],
  ["proposal/cyclonedx_2_0_parameter_contract.preview.cdx.json", "Commit-pinned CycloneDX 2.0 draft preview; not a conformance claim."],
  ["package_scope.json", "Machine-readable package scope and claim boundary."],
]);

export function buildPublicEvidencePackageFiles({
  analysis,
  context,
  scope,
  runtimeEvidence = null,
  origin = "",
} = {}) {
  if (!analysis || !context?.generatedAt || !scope) {
    throw new Error("Public Evidence Package requires a completed, report-bound audit.");
  }
  const verificationManifest = buildPublicVerificationManifest({
    analysis,
    context,
    scope,
    runtimeEvidence,
    origin,
  });
  if (!validatePublicVerificationManifest(verificationManifest)) {
    throw new Error("Public Evidence Package verification manifest failed self-validation.");
  }
  const reportBody = buildPublicAuditSummaryText(analysis, {
    scope,
    generatedAt: context.generatedAt,
  });
  const reportHtml = buildPublicEngineeringReportHtml(reportBody, {
    generatedAt: context.generatedAt,
    modelName: analysis.filename || "model",
    origin,
    reportFingerprint: verificationManifest.report.body_sha256,
  });
  const cycloneDx = buildPublicCycloneDxDocuments(analysis, {
    hash: verificationManifest.artifact.sha256,
    fileSizeBytes: verificationManifest.artifact.byte_length,
    generatedAt: context.generatedAt,
  });
  const rights = buildEvidencePackageRights({ profile: "public" });
  const packageScope = {
    schema: PUBLIC_EVIDENCE_PACKAGE_SCHEMA,
    generated_at: context.generatedAt,
    artifact: verificationManifest.artifact,
    evidence_class: scope.evidenceClass,
    report_body_sha256: verificationManifest.report.body_sha256,
    manifest_sha256: verificationManifest.manifest_sha256,
    member_roles: Object.fromEntries(PACKAGE_MEMBERS),
    rights,
    excluded_content: [
      "model artifact bytes and weights",
      "raw tensor values",
      "editable Engineering Report",
      "account-bound raw evidence ledger, CSV, graph, and visual exports",
    ],
    detached_signature: {
      member: "deepbom_public_key_signature.json",
      coverage: "SHA-256 member ledger over every package member listed above; the signature member is appended afterward and excluded from its own signed payload",
      expected_signer_class: "local_browser",
      trust_boundary: "The embedded public key permits integrity verification but does not establish DEEPBOM authorship unless an independently trusted official key registry identifies the signer.",
    },
    claim_boundary: [
      scope.runtimeBoundary,
      scope.releaseStatus,
      "Static artifact evidence does not establish task accuracy, clinical performance, observed native placement, or production release approval.",
    ],
  };
  const files = [
    zipTextFile("README.txt", packageReadme(packageScope)),
    zipTextFile("RIGHTS.txt", evidencePackageRightsText(rights)),
    zipTextFile("public_static_evidence_summary.txt", reportBody),
    zipTextFile("public_static_evidence_summary.html", reportHtml),
    zipTextFile("public_report_verification_manifest.json", jsonForDownload(verificationManifest)),
    zipTextFile("cyclonedx_1_7_artifact_evidence.cdx.json", jsonForDownload(cycloneDx.documents.cyclonedx_evidence)),
    zipTextFile("proposal/cyclonedx_2_0_parameter_contract.preview.cdx.json", jsonForDownload(cycloneDx.documents.cyclonedx_2_0_parameter_contract_preview)),
    zipTextFile("package_scope.json", jsonForDownload(packageScope)),
  ];
  if (!validatePublicEvidencePackageFiles(files, packageScope)) {
    throw new Error("Public Evidence Package failed member-contract validation.");
  }
  return files;
}

export function validatePublicEvidencePackageFiles(files, packageScope) {
  if (packageScope?.schema !== PUBLIC_EVIDENCE_PACKAGE_SCHEMA) return false;
  const expected = new Set(PACKAGE_MEMBERS.map(([name]) => name));
  const names = files.map((file) => file?.name);
  if (names.length !== expected.size || new Set(names).size !== names.length) return false;
  if (names.some((name) => !expected.has(name))) return false;
  const manifestFile = files.find((file) => file.name === "public_report_verification_manifest.json");
  const manifest = JSON.parse(String(manifestFile?.data || "null"));
  return validatePublicVerificationManifest(manifest)
    && validateEvidencePackageRights(packageScope.rights, { profile: "public" })
    && manifest.report.body_sha256 === packageScope.report_body_sha256
    && manifest.manifest_sha256 === packageScope.manifest_sha256
    && manifest.artifact.sha256 === packageScope.artifact.sha256;
}

function packageReadme(scope) {
  return [
    "DEEPBOM PUBLIC STATIC EVIDENCE PACKAGE",
    "",
    `Schema: ${scope.schema}`,
    `Generated: ${scope.generated_at}`,
    `Artifact SHA-256: ${scope.artifact.sha256}`,
    "",
    "CONTENTS",
    ...PACKAGE_MEMBERS.map(([name, role]) => `- ${name}: ${role}`),
    "- deepbom_public_key_signature.json: appended after package construction; signs the preceding member ledger with ES256.",
    "",
    "VERIFICATION",
    "1. Verify public_report_verification_manifest.json according to its embedded verification steps.",
    "2. Recompute the artifact SHA-256 from the original artifact bytes and compare it with the manifest.",
    "3. Verify deepbom_public_key_signature.json against its embedded public key and signed package-member digest.",
    "4. Treat a local_browser signer as integrity evidence only. It is not an official DEEPBOM release signature.",
    "",
    "BOUNDARY",
    ...scope.claim_boundary.map((line) => `- ${line}`),
    "",
    "LICENSE AND REUSE",
    "Copyright (C) 2026 Jun-Hwan Kwon. All rights reserved.",
    "This package is provided for review, verification, and citation of this audit result. No source-code, executable, model-weight, redistribution, modification, or implementation license is granted.",
    "Citation: https://doi.org/10.5281/zenodo.21834509",
    "",
  ].join("\n");
}
