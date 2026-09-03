import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
import {
  modelQuantizationStatus,
  quantizationScopeExplanation,
  quantizationScopeSummary,
} from "./analysis.js";
import { formatNumber } from "./format.js";
import {
  ANALYZER_METADATA,
  RUNTIME_COMPATIBILITY_EVIDENCE_LABEL,
  buildAnalyzerMetadata,
  scopeAndEvidenceBoundaryMarkdown,
} from "./report-metadata.js";

export function buildBundleModelSummaryLines(analysis, identity = {}, { includeSpeedup = false } = {}) {
  const quant = modelQuantizationStatus(analysis);
  const format = String(identity.format || analysis?.format || "tflite").toLowerCase();
  const weightContainer = ["gguf", "safetensors"].includes(format);
  const operatorCount = identity.operator_count ?? analysis?.operator_count ?? (weightContainer ? null : artifactIrOperators(analysis)?.length);
  const tensorCount = identity.tensor_count ?? analysis?.tensor_count ?? artifactIrValues(analysis)?.length;
  const totalMacs = identity.total_macs ?? analysis?.total_macs;
  const unavailable = weightContainer ? "not serialized by this format" : "not assessed";
  const lines = [
    `- Filename: \`${identity.filename || analysis?.filename || ""}\``,
    `- Format: \`${identity.format || analysis?.format || "tflite"}\``,
    `- SHA-256: \`${identity.sha256 || analysis?.model_sha256 || ""}\``,
    `- Target: ${identity.target_label || analysis?.target_label || "-"}`,
    `- Operators: ${operatorCount == null ? unavailable : formatNumber(operatorCount)}`,
    `- Tensors: ${tensorCount == null ? "not assessed" : formatNumber(tensorCount)}`,
    `- MACs: ${totalMacs == null ? unavailable : formatNumber(totalMacs)}`,
    `- Quantization: ${quant.label || quant.classification || "unknown"}`,
    `- Quantized compute scope: ${quantizationScopeSummary(analysis)}`,
  ];
  if (includeSpeedup && analysis?.estimated_int8_speedup != null && Number.isFinite(Number(analysis.estimated_int8_speedup))) {
    lines.push(`- INT8 speedup estimate: ~${Number(analysis.estimated_int8_speedup).toFixed(2)}x for ${identity.target_label || analysis?.target_label || "-"}`);
    if (analysis?.estimated_int8_speedup_detail) {
      lines.push(`- INT8 speedup method: ${analysis.estimated_int8_speedup_detail}`);
    }
  }
  return lines;
}

export function buildBundlePrivacyMarkdownLines({ includeAdvancedJson = false } = {}) {
  const lines = [
    "- Raw model weights are not included in this ZIP.",
    "- The model file, generated tensors, and outputs are not uploaded by this browser workflow.",
    "- Account identity and authorization details are removed, but model filename and artifact SHA-256 remain for engineering traceability. This is not an artifact-anonymized public-share mode.",
  ];
  if (includeAdvancedJson) {
    lines.push("- Advanced JSON files contain aggregate scores, timings, drift summaries, and metadata only.");
  }
  return lines;
}

function bundleMemberLayout(files = []) {
  const payloadFiles = files.map((file) => file.name);
  return {
    payload_files: payloadFiles,
    unsigned_package_members: [...payloadFiles, "manifest.json"],
    package_manifest_member: "manifest.json",
    attestation_member: "attestation.json",
    attestation_member_excluded_from_package_hash: true,
    member_order: "lexicographic_utf8",
    path_normalization: "relative_posix_nfc",
    duplicate_member_policy: "reject_after_nfc_and_case_fold",
    undeclared_member_policy: "reject",
    attestation_schema: ANALYZER_METADATA.schemas.attestation,
    verification_receipt_location: "attestation.json",
    package_digest_status: "computed and server-registered after all unsigned members, including manifest.json, are assembled",
  };
}

function reportPrivacyScope(files = [], { artifactIdentityEmbedded = true } = {}) {
  const report = String(files.find((file) => file.name === "engineering_report.md")?.data || "");
  const fieldValue = (label) => {
    const prefix = `| ${label} |`;
    const line = report.split(/\r?\n/).find((candidate) => candidate.startsWith(prefix));
    return line ? line.slice(prefix.length).replace(/\|\s*$/, "").trim() : "";
  };
  const embedsSensitiveValue = (label) => {
    const value = fieldValue(label);
    return Boolean(value) && !/not captured|none |redacted from report|unknown/i.test(value);
  };
  const exactBrowserSessionTimestamp = [
    "Consent timestamp",
    "Structural telemetry current report session transmission",
    "Structural telemetry most recent prior transmission",
  ].some((label) => /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(fieldValue(label)));
  return {
    report_present: Boolean(report),
    artifact_identity_embedded: Boolean(report) && artifactIdentityEmbedded,
    filename_embedded: Boolean(report) && artifactIdentityEmbedded,
    artifact_sha256_embedded: Boolean(report) && artifactIdentityEmbedded,
    consent_event_id_embedded: embedsSensitiveValue("Consent event ID"),
    telemetry_fingerprint_embedded: embedsSensitiveValue("Structural telemetry payload fingerprint"),
    exact_browser_session_timestamp_embedded: exactBrowserSessionTimestamp,
    report_generated_timestamp_embedded: /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(fieldValue("Generated")),
  };
}

function bundlePrivacyScopes(files, generatedAt, { publicRedacted = false } = {}) {
  const manifestSelf = {
    artifact_identity_embedded: !publicRedacted,
    filename_embedded: !publicRedacted,
    artifact_sha256_embedded: !publicRedacted,
    internal_target_profile_identifier_embedded: !publicRedacted,
    target_profile_digest_embedded: true,
    exact_export_timestamp_embedded: Boolean(generatedAt),
    account_identity_embedded: false,
    authorization_details_embedded: false,
    consent_event_id_embedded: false,
    telemetry_fingerprint_embedded: false,
    exact_browser_session_timestamp_embedded: false,
  };
  const includedReport = reportPrivacyScope(files, { artifactIdentityEmbedded: !publicRedacted });
  const effective = {
    raw_model_bytes_included: false,
    model_uploaded: false,
    outputs_uploaded: false,
    account_identity_embedded: false,
    authorization_details_embedded: false,
    artifact_identity_embedded: manifestSelf.artifact_identity_embedded || includedReport.artifact_identity_embedded,
    filename_embedded: manifestSelf.filename_embedded || includedReport.filename_embedded,
    artifact_sha256_embedded: manifestSelf.artifact_sha256_embedded || includedReport.artifact_sha256_embedded,
    consent_event_id_embedded: manifestSelf.consent_event_id_embedded || includedReport.consent_event_id_embedded,
    telemetry_fingerprint_embedded: manifestSelf.telemetry_fingerprint_embedded || includedReport.telemetry_fingerprint_embedded,
    exact_browser_session_timestamp_embedded: manifestSelf.exact_browser_session_timestamp_embedded || includedReport.exact_browser_session_timestamp_embedded,
    report_generated_timestamp_embedded: includedReport.report_generated_timestamp_embedded,
    exact_export_timestamp_embedded: manifestSelf.exact_export_timestamp_embedded,
    internal_target_profile_identifier_embedded: manifestSelf.internal_target_profile_identifier_embedded,
    target_profile_digest_embedded: manifestSelf.target_profile_digest_embedded,
    public_share_suitable_without_identity_review: publicRedacted,
  };
  return {
    manifest_self_privacy: manifestSelf,
    included_report_privacy: includedReport,
    bundle_effective_privacy: effective,
    privacy: effective,
  };
}

export function buildEngineeringBundleSummary({
  files = [],
  analysis = null,
  identity = {},
  user = { label: "anonymous" },
  generatedAt = new Date().toISOString(),
} = {}) {
  const included = files.map((file) => `- \`${file.name}\``).join("\n");
  return [
    "# DEEPBOM Engineering Bundle",
    "",
    `Generated: ${generatedAt}`,
    "Export mode: account-redacted external bundle (model identity retained; account identity and authorization details omitted)",
    "",
    "## Analyzer Metadata",
    `- Analyzer: ${ANALYZER_METADATA.name} ${ANALYZER_METADATA.version}`,
    `- Rulepack: ${ANALYZER_METADATA.rulepackVersion}`,
    `- Report schema: ${ANALYZER_METADATA.schemas.engineeringReport}`,
    `- Bundle schema: ${ANALYZER_METADATA.schemas.engineeringBundle}`,
    "",
    "## Scope",
    "- Engineering-facing technical package.",
    "- Includes the Engineering Report and static audit artifacts.",
    "- Does not include the Regulatory Report or Research module execution payloads.",
    "",
    scopeAndEvidenceBoundaryMarkdown("## Evidence Boundary"),
    "",
    "## Model",
    ...buildBundleModelSummaryLines(analysis, identity),
    "",
    "## Included Files",
    included || "- No files.",
    "",
    "## Privacy Boundary",
    ...buildBundlePrivacyMarkdownLines(),
    "",
  ].join("\n");
}

export function buildEvidenceBundleSummary({
  moduleLog = [],
  files = [],
  analysis = null,
  identity = {},
  capabilities = {},
  user = { label: "anonymous" },
  generatedAt = new Date().toISOString(),
} = {}) {
  const included = files.map((file) => `- \`${file.name}\``).join("\n");
  const modules = moduleLog.map((item) => {
    const detail = item.error ? ` (${item.error})` : item.reason ? ` (${item.reason})` : "";
    const prefix = item.status === "complete"
      ? "[complete]"
      : item.status === "failed"
        ? "[failed]"
        : item.status === "blocked"
          ? "[blocked]"
          : item.status === "skipped"
            ? "[skipped]"
            : "[status]";
    const includedText = item.included ? " / file included" : "";
    return `- ${prefix} ${item.label}: ${item.status}${includedText}${detail}`;
  }).join("\n");
  return [
    "# DEEPBOM Regulatory Bundle",
    "",
    "Regulatory Bundle = complete Engineering Bundle + Regulatory Report + enabled Research module evidence.",
    "",
    `Generated: ${generatedAt}`,
    "Export mode: account-redacted external bundle (model identity retained; account identity and authorization details omitted)",
    "",
    "## Analyzer Metadata",
    `- Analyzer: ${ANALYZER_METADATA.name} ${ANALYZER_METADATA.version}`,
    `- Rulepack: ${ANALYZER_METADATA.rulepackVersion}`,
    `- Report schema: ${ANALYZER_METADATA.schemas.regulatoryReport}`,
    `- Bundle schema: ${ANALYZER_METADATA.schemas.regulatoryBundle}`,
    "",
    scopeAndEvidenceBoundaryMarkdown("## Evidence Boundary"),
    "",
    "## Model",
    ...buildBundleModelSummaryLines(analysis, identity, { includeSpeedup: true }),
    "",
    "## Authorization Boundary",
    "- Required access was checked at generation time.",
    "- Account identity, role, access profile, and capability matrix are intentionally not embedded in this external-share package.",
    `- Runtime module evidence, when present, is labeled ${RUNTIME_COMPATIBILITY_EVIDENCE_LABEL}.`,
    "",
    "## Module Run Log",
    modules || "- No Research module was attempted.",
    "",
    "## Included Files",
    included || "- No files.",
    "",
    "## Privacy Boundary",
    ...buildBundlePrivacyMarkdownLines({ includeAdvancedJson: true }),
    "",
    "## Interpretation Notes",
    "- Static speedup, roofline, L1, packing, and delegate severity are target-dependent. Re-run the audit after changing the target profile.",
    "- Byte values in JSON are raw bytes. UI labels use binary units such as KiB/MiB for compact display.",
    "- Quantization op-state counts cover all graph ops; quantized compute counts cover MAC-bearing compute ops only, so the denominators intentionally differ.",
    "",
  ].join("\n");
}

export function buildEngineeringBundleManifest({
  model = {},
  user = { label: "anonymous" },
  files = [],
  generatedAt = new Date().toISOString(),
  analysis = null,
  exportMode = "account_redacted_external",
} = {}) {
  const memberLayout = bundleMemberLayout(files);
  const publicRedacted = exportMode === "artifact_redacted_public";
  const privacyScopes = bundlePrivacyScopes(files, generatedAt, { publicRedacted });
  return {
    schema: ANALYZER_METADATA.schemas.engineeringBundle,
    generated_at: generatedAt,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    model,
    export_mode: exportMode,
    artifact_id: publicRedacted ? "ARTIFACT-001" : null,
    report_scope: "Engineering Report only; Regulatory Report is not included.",
    ...memberLayout,
    ...privacyScopes,
  };
}

export function buildEvidenceBundleManifest({
  analysis = null,
  model = {},
  user = { label: "anonymous" },
  capabilities = {},
  moduleLog = [],
  files = [],
  generatedAt = new Date().toISOString(),
} = {}) {
  const memberLayout = bundleMemberLayout(files);
  const privacyScopes = bundlePrivacyScopes(files, generatedAt);
  return {
    schema: ANALYZER_METADATA.schemas.regulatoryBundle,
    generated_at: generatedAt,
    analyzer_metadata: buildAnalyzerMetadata(analysis),
    model,
    export_mode: "account_redacted_external",
    package_hierarchy: "Regulatory Bundle = Engineering Bundle + Regulatory Report + enabled Research module evidence.",
    target_dependent_static_estimates: {
      target_id: model.target_id,
      target_label: model.target_label,
      note: "Roofline posture, L1 pressure, packing hints, XNNPACK chain severity, and INT8 speedup are recomputed for the selected target. Bundles generated for A55, A72, AVX2, and WASM are not expected to have identical speedup or bottleneck values.",
    },
    quantization_scope: quantizationScopeExplanation(analysis),
    modules: moduleLog,
    ...memberLayout,
    ...privacyScopes,
  };
}
