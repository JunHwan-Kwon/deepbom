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
import {
  buildEvidenceLevelManifest,
  buildEvidenceLevelReport,
  evidenceLevelProfile,
  validateEvidenceLevelManifest,
} from "./evidence-level-report.js";

export const EVIDENCE_PACKAGE_PROFILE_SCHEMA = "deepbom.evidence_package_profile.v1.1";

export const EVIDENCE_PACKAGE_PROFILES = Object.freeze([
  Object.freeze({ id: "public", label: "Public", detail: "Watermarked summary, verification manifest, and CycloneDX documents" }),
  Object.freeze({ id: "engineering", label: "Engineering review", detail: "Watermarked synthesized Engineering Report without raw model or tensor payloads" }),
  Object.freeze({ id: "regulatory", label: "Regulatory support", detail: "Watermarked regulatory-support report and Engineering Report appendix; not a submission or approval" }),
  Object.freeze({ id: "machine_readable", label: "Machine-readable", detail: "Verification manifest, CycloneDX documents, exact report body, and package scope" }),
]);

const PROFILE_BY_ID = new Map(EVIDENCE_PACKAGE_PROFILES.map((profile) => [profile.id, profile]));

export function evidencePackageProfile(profileId) {
  return PROFILE_BY_ID.get(String(profileId || "")) || PROFILE_BY_ID.get("public");
}

export function buildEvidencePackageProfileFiles({
  profileId = "public",
  evidenceLevelId = "all_available",
  analysis,
  context,
  scope,
  runtimeEvidence = null,
  metricCoverage = null,
  findings = null,
  origin = "",
  reportBody = null,
} = {}) {
  const profile = evidencePackageProfile(profileId);
  if (!analysis || !context?.generatedAt || !scope) {
    throw new Error("Evidence Package requires a completed, report-bound audit.");
  }
  const evidenceLevel = evidenceLevelProfile(evidenceLevelId);
  const evidenceLevelManifest = buildEvidenceLevelManifest({
    levelId: evidenceLevel.id,
    analysis,
    metricCoverage,
    findings,
    runtimeEvidence,
    generatedAt: context.generatedAt,
  });
  const scoped = evidenceLevel.id !== "all_available";
  const body = scoped
    ? buildEvidenceLevelReport(evidenceLevelManifest, { scope })
    : profile.id === "public" || profile.id === "machine_readable"
      ? buildPublicAuditSummaryText(analysis, { scope, generatedAt: context.generatedAt })
      : String(reportBody || "");
  if (!body) throw new Error(`${profile.label} Evidence Package requires a generated report body.`);
  const reportKindBase = profile.id === "engineering"
    ? "engineering-review-report"
    : profile.id === "regulatory"
      ? "regulatory-support-report"
      : profile.id === "public"
        ? "public-static-evidence-summary"
        : "machine-readable-static-evidence-body";
  const reportKind = scoped ? `${reportKindBase}-${evidenceLevel.id}` : reportKindBase;
  const verificationManifest = buildPublicVerificationManifest({
    analysis,
    context,
    scope,
    runtimeEvidence,
    origin,
    reportBody: body,
    reportKind,
    reportPresentation: profile.id === "machine_readable"
      ? "exact UTF-8 text member"
      : "browser-generated watermarked HTML review copy",
  });
  if (!validatePublicVerificationManifest(verificationManifest)) {
    throw new Error(`${profile.label} verification manifest failed self-validation.`);
  }
  const files = [];
  const roles = {};
  const rights = buildEvidencePackageRights({ profile: profile.id });
  add(files, roles, "README.txt", packageReadme(profile, verificationManifest, evidenceLevel), "Profile scope, evidence level, verification steps, and reuse boundary.");
  add(files, roles, "RIGHTS.txt", evidencePackageRightsText(rights), "Copyright, permitted-purpose, provenance, and reuse boundary.");
  const bodyName = profile.id === "public"
    ? "public_static_evidence_summary.txt"
    : profile.id === "machine_readable"
      ? "evidence_body.txt"
      : `reports/${profile.id}_report.txt`;
  add(files, roles, bodyName, body, "Exact UTF-8 report body bound by the verification manifest.");
  if (profile.id !== "machine_readable") {
    const html = buildPublicEngineeringReportHtml(body, {
      generatedAt: context.generatedAt,
      modelName: analysis.filename || "model",
      origin,
      reportFingerprint: verificationManifest.report.body_sha256,
      profile: profile.id,
    });
    const htmlName = profile.id === "public"
      ? "public_static_evidence_summary.html"
      : `reports/${profile.id}_report.html`;
    add(files, roles, htmlName, html, "Watermarked presentation copy of the exact report body.");
  }
  const manifestName = profile.id === "public" ? "public_report_verification_manifest.json" : "verification_manifest.json";
  add(files, roles, manifestName, jsonForDownload(verificationManifest), "Artifact, report, analyzer, target, and imported-runtime identity hashes.");
  add(files, roles, "evidence_level_manifest.json", jsonForDownload(evidenceLevelManifest), "Machine-readable cumulative evidence-class selection and exclusion ledger.");
  if (!scoped) {
    const cycloneDx = buildPublicCycloneDxDocuments(analysis, {
      hash: verificationManifest.artifact.sha256,
      fileSizeBytes: verificationManifest.artifact.byte_length,
      generatedAt: context.generatedAt,
    });
    const cdx17Name = profile.id === "public"
      ? "cyclonedx_1_7_artifact_evidence.cdx.json"
      : "cyclonedx/cyclonedx_1_7_artifact_evidence.cdx.json";
    const cdx20Name = profile.id === "public"
      ? "proposal/cyclonedx_2_0_parameter_contract.preview.cdx.json"
      : "cyclonedx/proposal_2_0_parameter_contract.preview.cdx.json";
    add(files, roles, cdx17Name, jsonForDownload(cycloneDx.documents.cyclonedx_evidence), "Standalone CycloneDX 1.7 artifact-evidence document.");
    add(files, roles, cdx20Name, jsonForDownload(cycloneDx.documents.cyclonedx_2_0_parameter_contract_preview), "Commit-pinned CycloneDX 2.0 draft preview; not a conformance claim.");
  }
  roles["package_scope.json"] = "Machine-readable profile membership, exclusions, and claim boundary.";
  const packageScope = {
    schema: EVIDENCE_PACKAGE_PROFILE_SCHEMA,
    profile: profile.id,
    profile_label: profile.label,
    evidence_level: evidenceLevel.id,
    evidence_level_label: evidenceLevel.label,
    evidence_level_rank: evidenceLevel.rank,
    generated_at: context.generatedAt,
    login_required: false,
    artifact: verificationManifest.artifact,
    report: verificationManifest.report,
    member_roles: roles,
    rights,
    excluded_content: [
      "original model artifact bytes and weights",
      "raw tensor values",
      "research-module execution payloads",
      "task-accuracy or clinical-validation claims not supplied as evidence",
      ...(scoped ? ["standalone unscoped CycloneDX documents", "metric families and findings above the selected evidence level"] : []),
    ],
    integrity_boundary: "The verification manifest and appended detached package signature make changes detectable. They do not prevent copying or editing, prove official DEEPBOM authorship without an independently trusted key registry, or grant redistribution rights.",
    claim_boundary: [
      scope.runtimeBoundary,
      scope.releaseStatus,
      "Static artifact evidence does not establish task accuracy, clinical performance, observed native placement, or production release approval.",
    ],
  };
  files.push(zipTextFile("package_scope.json", jsonForDownload(packageScope)));
  if (!validateEvidencePackageProfileFiles(files, { profileId: profile.id })) {
    throw new Error(`${profile.label} Evidence Package failed member-contract validation.`);
  }
  return files;
}

export function validateEvidencePackageProfileFiles(files, { profileId } = {}) {
  const profile = evidencePackageProfile(profileId);
  const names = files.map((file) => file?.name);
  if (names.some((name) => !name) || names.length !== new Set(names).size) return false;
  const manifestName = profile.id === "public" ? "public_report_verification_manifest.json" : "verification_manifest.json";
  const scopeName = "package_scope.json";
  if (!names.includes(manifestName) || !names.includes(scopeName) || !names.includes("README.txt")
    || !names.includes("evidence_level_manifest.json")) return false;
  const manifest = JSON.parse(String(files.find((file) => file.name === manifestName)?.data || "null"));
  const packageScope = JSON.parse(String(files.find((file) => file.name === scopeName)?.data || "null"));
  const evidenceLevelManifest = JSON.parse(String(files.find((file) => file.name === "evidence_level_manifest.json")?.data || "null"));
  const expectedHtml = profile.id !== "machine_readable";
  const htmlCount = names.filter((name) => name.endsWith(".html")).length;
  const scoped = packageScope.evidence_level !== "all_available";
  const cycloneDxCount = names.filter((name) => name.endsWith(".cdx.json")).length;
  return validatePublicVerificationManifest(manifest)
    && packageScope.schema === EVIDENCE_PACKAGE_PROFILE_SCHEMA
    && packageScope.profile === profile.id
    && packageScope.evidence_level === evidenceLevelManifest.requested_level
    && packageScope.login_required === false
    && validateEvidencePackageRights(packageScope.rights, { profile: profile.id })
    && validateEvidenceLevelManifest(evidenceLevelManifest)
    && packageScope.report.body_sha256 === manifest.report.body_sha256
    && htmlCount === (expectedHtml ? 1 : 0)
    && cycloneDxCount === (scoped ? 0 : 2)
    && names.every((name) => !/\.(tflite|onnx|gguf|safetensors|mlmodel)$/i.test(name));
}

function add(files, roles, name, data, role) {
  files.push(zipTextFile(name, data));
  roles[name] = role;
}

function packageReadme(profile, manifest, evidenceLevel) {
  return [
    `DEEPBOM ${profile.label.toUpperCase()} EVIDENCE PACKAGE`,
    "",
    `Profile: ${profile.id}`,
    `Evidence level: ${evidenceLevel.id} (${evidenceLevel.label})`,
    `Artifact SHA-256: ${manifest.artifact.sha256}`,
    `Report-body SHA-256: ${manifest.report.body_sha256}`,
    "",
    "VERIFY",
    "1. Recompute the SHA-256 of the exact report-body text member and compare it with verification_manifest.json.",
    "2. Recompute the original artifact SHA-256 and compare it with verification_manifest.json.",
    "3. Verify the appended deepbom_public_key_signature.json member ledger.",
    "4. Treat a local-browser signer as integrity evidence only unless an independently trusted registry identifies that public key.",
    "",
    "BOUNDARY",
    "This package excludes original model bytes, weights, raw tensor values, and research execution payloads.",
    "Hashes and signatures make changes detectable. They cannot prevent copying or editing and do not grant redistribution or implementation rights.",
    "Citation: https://doi.org/10.5281/zenodo.21834509",
    "",
  ].join("\n");
}
