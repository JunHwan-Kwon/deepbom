import {
  ANALYZER_BUILD_COMMIT,
  ANALYZER_BUILD_SOURCE_STATE,
  ANALYZER_BUNDLE_CONTENT_HASH_METHOD,
  ANALYZER_BUNDLE_CONTENT_MANIFEST_SHA256,
  ANALYZER_BUNDLE_CONTENT_SHA256,
  RULEPACK_SHA256,
} from "./build-metadata.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { buildPublicAuditSummaryText } from "./session-evidence.js";

const SHA256 = /^[a-f0-9]{64}$/;

export function buildPublicVerificationManifest({
  analysis,
  context,
  scope,
  runtimeEvidence = null,
  origin = "",
  citationDoi = "10.5281/zenodo.21834509",
  reportBody = null,
  reportKind = "public-static-evidence-summary",
  reportPresentation = "browser-generated watermarked HTML print view",
} = {}) {
  if (!analysis || !context?.generatedAt || !scope) throw new Error("Public verification manifest requires a bound audited report context.");
  const artifactSha256 = requiredSha(analysis.model_sha256, "artifact SHA-256");
  const resolvedReportBody = reportBody == null
    ? buildPublicAuditSummaryText(analysis, { scope, generatedAt: context.generatedAt })
    : String(reportBody);
  if (!resolvedReportBody) throw new Error("Public verification manifest requires a non-empty report body.");
  const reportBodySha256 = sha256TextHex(resolvedReportBody);
  const targetSha256 = optionalSha(analysis.target_profile?.profile_sha256);
  const runtimeArtifactSha256 = optionalSha(runtimeEvidence?.artifact_sha256);
  const runtimeTargetSha256 = optionalSha(runtimeEvidence?.target_profile_sha256);
  const runtimeBinarySha256 = optionalSha(runtimeEvidence?.runtime?.binary_sha256 || runtimeEvidence?.source?.collector?.binary_sha256);
  const body = {
    schema: "deepbom.public_report_verification_manifest.v1",
    document_type: "detached-public-report-verification-manifest",
    generated_at: context.generatedAt,
    artifact: {
      filename: analysis.filename || null,
      format: analysis.format || null,
      byte_length: Number(analysis.file_size_bytes ?? analysis.file_size) || null,
      sha256: artifactSha256,
    },
    report: {
      kind: String(reportKind || "public-static-evidence-summary"),
      body_sha256: reportBodySha256,
      body_hash_basis: "SHA-256 over the UTF-8 bytes of the exact plain-text body printed below the report header",
      presentation: String(reportPresentation || "browser-generated watermarked HTML print view"),
    },
    analyzer: {
      source_commit: ANALYZER_BUILD_COMMIT,
      source_state: ANALYZER_BUILD_SOURCE_STATE,
      bundle_content_sha256: requiredSha(ANALYZER_BUNDLE_CONTENT_SHA256, "analyzer bundle content SHA-256"),
      bundle_hash_method: ANALYZER_BUNDLE_CONTENT_HASH_METHOD,
      bundle_manifest_sha256: requiredSha(ANALYZER_BUNDLE_CONTENT_MANIFEST_SHA256, "analyzer bundle manifest SHA-256"),
      rulepack_sha256: requiredSha(RULEPACK_SHA256, "rulepack SHA-256"),
    },
    target_profile: analysis.target_profile ? {
      id: analysis.target_profile.id || null,
      sha256: targetSha256,
      binding_status: targetSha256 ? "hash_bound" : "unbound",
    } : null,
    runtime_evidence: runtimeEvidence ? {
      schema: runtimeEvidence.schema || null,
      artifact_sha256: runtimeArtifactSha256,
      target_profile_sha256: runtimeTargetSha256,
      runtime_binary_sha256: runtimeBinarySha256,
      binding_status: runtimeArtifactSha256 === artifactSha256
        && (!targetSha256 || runtimeTargetSha256 === targetSha256)
        && Boolean(runtimeBinarySha256) ? "identity_bound" : "incomplete",
    } : { binding_status: "not_imported" },
    citation: { doi: citationDoi, url: `https://doi.org/${citationDoi}` },
    origin: String(origin || ""),
    verification: {
      canonicalization: "RFC8785-JCS",
      steps: [
        "Recompute SHA-256 over the exact public report body and compare report.body_sha256.",
        "Recompute SHA-256 over the audited artifact bytes and compare artifact.sha256.",
        "Remove manifest_sha256, canonicalize the remaining object with RFC8785-JCS, and compare its SHA-256.",
      ],
      signature_status: "unsigned",
      boundary: "Hashes make changes detectable. They do not prove authorship, execution, task accuracy, clinical performance, or release approval.",
    },
  };
  return { ...body, manifest_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validatePublicVerificationManifest(manifest) {
  if (manifest?.schema !== "deepbom.public_report_verification_manifest.v1") return false;
  if (!SHA256.test(String(manifest.artifact?.sha256 || "")) || !SHA256.test(String(manifest.report?.body_sha256 || ""))) return false;
  const body = { ...manifest };
  delete body.manifest_sha256;
  return SHA256.test(String(manifest.manifest_sha256 || ""))
    && sha256TextHex(canonicalJson(body)) === manifest.manifest_sha256;
}

function requiredSha(value, label) {
  const digest = optionalSha(value);
  if (!digest) throw new Error(`Public verification manifest requires ${label}.`);
  return digest;
}

function optionalSha(value) {
  const digest = String(value || "").toLowerCase();
  return SHA256.test(digest) ? digest : null;
}
