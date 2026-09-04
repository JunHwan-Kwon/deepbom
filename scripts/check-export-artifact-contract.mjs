import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { TEXT_EXPORT_ARTIFACTS } from "../web/lib/export-artifacts.js";
import {
  ANALYZER_METADATA,
  RUNTIME_COMPATIBILITY_EVIDENCE_LABEL,
  buildCanonicalPackageDigest,
  buildChangeAnalysis,
  buildConformanceReport,
  buildEngineeringBundleArtifactFiles,
  buildEngineeringEvidenceDocument,
  buildEngineeringBundleManifest,
  buildEngineeringBundleSummary,
  buildEvidenceBundleManifest,
  buildEngineeringReport,
  buildEngineeringReportArtifacts,
  buildEvidenceBundleSummary,
  buildFindingsRegister,
  buildMlBomDocument,
  buildDecisionCoverageLedger,
  buildMetricCoverageManifest,
  buildRawDataArtifactFiles,
  buildRawEvidenceFiles,
  buildRuntimeAssignmentComparisonCsv,
  buildRuntimeBoundaryComparisonCsv,
  buildPublicShareAnalysis,
  buildPublicShareIdentity,
  artifactUuidFromSha256,
  buildAccountEngineeringReportHtml,
  buildPublicAuditSummaryText,
  buildPublicEngineeringReportHtml,
  buildSessionPrivacy,
  buildSessionReportContextSet,
  buildStaticAnalysisExport,
  buildReportContextSet,
  buildRuntimeEvidence,
  markdownWithModelSha256,
  validateMetricCoverageManifest,
} from "../web/lib/report.js";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { BROWSER_BENCHMARK_NOISE_METHOD, BROWSER_BENCHMARK_STATISTICS_METHOD, BROWSER_BENCHMARK_TIMING_METHOD } from "../web/lib/runtime.js";
import { benchmarkNoise, latencyStats } from "../web/lib/format.js";
import { decodeRoofReason } from "../web/lib/reason-codes.js";
import { canonicalJson, csvCell } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { analyzerContentVersion } from "../web/lib/cyclonedx-identity.js";
import { createCheck } from "./check-assert.mjs";
import { buildMlBomCompatibilityProjection } from "../web/lib/report-mlbom-compat.js";
import { buildPublicCycloneDx17ArtifactContract } from "../web/lib/public-cyclonedx-export.js";

const {
  done,
  expect,
  expectEqual,
  expectThrows,
} = createCheck("Export artifact contract check");

const failedRuntimeEvidence = buildRuntimeEvidence({ benchmarkResults: [{ backend: "wasm", ok: false, error: "failed" }] });
expectEqual(failedRuntimeEvidence.runtime_execution_status, "attempted_failed", "A failed benchmark attempt must not claim runtime execution.");
expectEqual(failedRuntimeEvidence.assessments.benchmarks.status, "not_assessed", "All-failed benchmark evidence must remain not assessed.");
const mixedRuntimeEvidence = buildRuntimeEvidence({ benchmarkResults: [{ backend: "wasm", ok: true }, { backend: "webgpu", ok: false }] });
expectEqual(mixedRuntimeEvidence.runtime_execution_status, "executed", "At least one completed backend supplies browser runtime evidence.");
expectEqual(mixedRuntimeEvidence.assessments.benchmarks.status, "partial", "Mixed benchmark success must retain partial coverage.");

const appSource = readFileSync("web/app.js", "utf8");
const appGraphWorkspaceSource = readFileSync("web/lib/app-graph-workspace.js", "utf8");
const appSources = `${appSource}\n${appGraphWorkspaceSource}`;
const publicKeySignatureSource = readFileSync("web/lib/public-key-signature.js", "utf8");
const htmlSource = readFileSync("web/index.html", "utf8");
const medicalSurfaceSource = readFileSync("web/lib/app-surface.js", "utf8");
const exportContractSource = readFileSync("web/lib/export-contract-view.js", "utf8");
const entries = Object.entries(TEXT_EXPORT_ARTIFACTS);
const requiredKeys = [
  "engineeringReport",
  "regulatoryReport",
  "rooflineCsv",
  "mermaidStageGraph",
  "cyclonedxEvidence",
  "cyclonedx20DraftStatus",
  "observedFormulation",
  "runtimeRequirementManifest",
  "missingProvenanceFields",
  "graphSvg",
];
const textExportButtons = {
  engineeringReport: "downloadMarkdown",
  regulatoryReport: "downloadRegulatoryReport",
  rooflineCsv: "downloadCsv",
  mermaidStageGraph: "downloadMermaid",
};

expectExactSet("export artifacts", entries.map(([key]) => key), requiredKeys);
expectUnique("export suffixes", entries.map(([, artifact]) => artifact.suffix));

for (const [key, artifact] of entries) {
  expectNonEmpty(artifact.permissionLabel, `${key}.permissionLabel`);
  expectNonEmpty(artifact.suffix, `${key}.suffix`);
  expectNonEmpty(artifact.type, `${key}.type`);
  if (/\s/.test(artifact.suffix || "")) {
    expect(false, `${key}.suffix must not contain whitespace.`);
  }
  if (!appSources.includes(`TEXT_EXPORT_ARTIFACTS.${key}`) && !exportContractSource.includes(`"${key}"`)) {
    expect(false, `No export controller references TEXT_EXPORT_ARTIFACTS.${key}.`);
  }
}

for (const [key, buttonId] of Object.entries(textExportButtons)) {
  const source = key === "regulatoryReport" ? medicalSurfaceSource : htmlSource;
  const htmlLabel = key === "regulatoryReport" ? "web/lib/app-surface.js" : "web/index.html";
  expectHtmlId(source, buttonId, `${key} button`, htmlLabel);
  expectAppSnippet(
    `registerTextExport(${buttonId}, TEXT_EXPORT_ARTIFACTS.${key}`,
    `${key} should register ${buttonId} through registerTextExport.`,
  );
}
expectHtmlId(htmlSource, "downloadGraphSvg", "graphSvg button", "web/index.html");
expectHtmlId(htmlSource, "downloadRawData", "authorized raw data button", "web/index.html");
expectHtmlId(htmlSource, "printPublicReport", "login-free watermarked print button", "web/index.html");
expectNotHtmlId(htmlSource, "downloadRegulatoryReport", "default app should not expose Regulatory Report", "web/index.html");
expectNotHtmlId(htmlSource, "downloadEvidenceBundle", "default app should not expose Regulatory Bundle", "web/index.html");
for (const [snippet, label] of [
  ["downloadGraphSvg.addEventListener", "graphSvg should be wired to the graph SVG download button."],
  ["downloadRawData.addEventListener", "Download Raw Data should be wired to the authorized raw data ZIP button."],
  ["downloadMarkdown.disabled = !current || !reportTargetReady", "Engineering Report button should depend on a completed report binding, not account authorization."],
  ['profile: "engineering"', "Standalone Engineering Report should use the login-free watermarked presentation profile."],
  ['profile: "regulatory"', "Standalone Regulatory Support Report should use the login-free watermarked presentation profile."],
  ["syncPublicPrintButton(printPublicReport, { hasAnalysis: Boolean(current), reportTargetReady })", "Public print should depend on report binding, not account authorization."],
  ["bindPublicAuditPrintButton(printPublicReport", "Public print should use the login-free fingerprinted summary controller."],
  ['appendPublicKeySignature(files, "engineering_bundle")', "Engineering Bundle should include an independently verifiable public-key signature."],
  ['appendPublicKeySignature(files, `evidence_package_${profile.id}`)', "Every Evidence Package profile should include an independently verifiable local-browser public-key signature."],
  ['appendPublicKeySignature(files, "raw_data")', "Download Raw Data should include an independently verifiable public-key signature."],
  ['appendPublicKeySignature(packagedFiles, "regulatory_bundle")', "Regulatory Bundle should include an independently verifiable public-key signature."],
  ['appendPackageAttestation(files, "raw_data")', "Download Raw Data should include a canonical digest attestation."],
  ["artifact: TEXT_EXPORT_ARTIFACTS.graphSvg", "graphSvg should use TEXT_EXPORT_ARTIFACTS.graphSvg."],
  ["registerTextExport,", "text export registration should come from the shared download helper."],
]) {
  expectAppSnippet(snippet, label);
}
expect(publicKeySignatureSource.includes('zipTextFile("deepbom_public_key_signature.json"'), "Public-key signing should emit a stable detached signature member.");
for (const [snippet, label] of [
  ['signPackageDigest(signedDigest, { scope: "deployment_contract_pack" })', "Deployment Contract Pack should sign its canonical member digest with ES256."],
  ['verifyPackageSignature(signature, signedDigest)', "Deployment Contract Pack should self-verify its detached signature before download."],
  ['zipTextFile("deepbom_public_key_signature.json"', "Deployment Contract Pack should emit the detached public-key signature member."],
]) {
  expect(exportContractSource.includes(snippet), label);
}
for (const [key, field, expected] of [
  ["engineeringReport", "ensureHash", true],
  ["engineeringReport", "type", "text/html"],
  ["engineeringReport", "raw", undefined],
  ["regulatoryReport", "regulatory", true],
  ["regulatoryReport", "ensureHash", true],
  ["cyclonedxEvidence", "requireModelBytes", true],
  ["cyclonedxEvidence", "ensureHash", true],
  ["cyclonedxEvidence", "type", "application/vnd.cyclonedx+json; version=1.7"],
  ["cyclonedxEvidence", "raw", true],
  ["cyclonedx20DraftStatus", "requireModelBytes", true],
  ["cyclonedx20DraftStatus", "ensureHash", true],
  ["cyclonedx20DraftStatus", "type", "application/json"],
  ["cyclonedx20DraftStatus", "raw", true],
  ["observedFormulation", "type", "application/vnd.cyclonedx+json; version=1.7"],
  ["observedFormulation", "raw", true],
  ["runtimeRequirementManifest", "raw", true],
  ["missingProvenanceFields", "raw", true],
  ["graphSvg", "type", "image/svg+xml"],
  ["graphSvg", "raw", true],
  ["rooflineCsv", "type", "text/csv"],
  ["rooflineCsv", "raw", true],
  ["mermaidStageGraph", "raw", true],
]) {
  expectEqual(TEXT_EXPORT_ARTIFACTS[key]?.[field], expected, `${key}.${field} mismatch.`);
}
expectEqual(
  markdownWithModelSha256("## Summary\n- Existing", "abc123"),
  "## Summary\n- Model SHA-256: `abc123`\n- Existing",
  "markdownWithModelSha256 should insert SHA under Summary.",
);
expectEqual(
  markdownWithModelSha256("- Model SHA-256: `old`\n\nBody", "new"),
  "- Model SHA-256: `new`\n\nBody",
  "markdownWithModelSha256 should replace an existing SHA line.",
);
expectEqual(
  markdownWithModelSha256("Body", "abc123"),
  "- Model SHA-256: `abc123`\n\nBody",
  "markdownWithModelSha256 should prepend SHA when no Summary exists.",
);

expectEqual(
  artifactUuidFromSha256("00112233445566778899aabbccddeeff"),
  "00112233-4455-8677-8899-aabbccddeeff",
  "Artifact UUID derivation must be deterministic and preserve the SHA-derived payload.",
);
expectThrows(
  () => buildAccountEngineeringReportHtml("# Report"),
  "frozen generatedAt",
  "Engineering Report HTML must reject a non-reproducible timestamp.",
);
const accountHtml = buildAccountEngineeringReportHtml("# Report\n<script>alert('x')</script>", {
  generatedAt: "2026-07-22T00:00:00.000Z",
  owner: "analyst@example.com<script>",
  modelName: "model<&>.onnx",
  origin: "https://deepbom.org/?x=<unsafe>",
});
expect(accountHtml.includes("Generated: 2026-07-22T00:00:00.000Z"), "Engineering Report HTML should preserve the frozen report timestamp.");
expect(accountHtml.includes("model&lt;&amp;&gt;.onnx"), "Engineering Report HTML should escape model metadata.");
expect(accountHtml.includes("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"), "Engineering Report HTML should escape report content.");
expect(!accountHtml.includes("<script>alert"), "Engineering Report HTML must not inject report content as executable markup.");
expect(accountHtml.includes("@media (max-width: 640px)"), "Engineering Report HTML should carry a mobile overflow layout.");
expectThrows(
  () => buildPublicEngineeringReportHtml("# Report", { generatedAt: "2026-07-22T00:00:00.000Z", reportFingerprint: "not-a-hash" }),
  "SHA-256 report fingerprint",
  "Public report HTML must reject an unbound report fingerprint.",
);
const publicHtml = buildPublicEngineeringReportHtml("# Report\n<script>alert('x')</script>", {
  generatedAt: "2026-07-22T00:00:00.000Z",
  modelName: "public<&>.tflite",
  origin: "https://deepbom.org/?x=<unsafe>",
  reportFingerprint: "a".repeat(64),
});
expect(publicHtml.includes("DEEPBOM PUBLIC COPY"), "Public report HTML should carry a visible print watermark.");
expect(publicHtml.includes(`Report-body SHA-256: ${"a".repeat(64)}`), "Public report HTML should expose the exact report-body fingerprint.");
expect(publicHtml.includes("not a digital signature"), "Public report HTML should preserve the verification boundary.");
expect(publicHtml.includes("public&lt;&amp;&gt;.tflite"), "Public report HTML should escape model metadata.");
expect(!publicHtml.includes("<script>alert"), "Public report HTML must not inject report content as executable markup.");
const publicSummary = buildPublicAuditSummaryText({
  filename: "sample<&>.gguf",
  format: "gguf",
  model_sha256: "b".repeat(64),
  file_size_bytes: 128,
  operator_count: 0,
  tensor_count: 2,
  inputs: [],
  outputs: [],
  mac_assessment: { status: "not_applicable_weight_container" },
  quantization_status: {
    label: "GGUF tensor encodings",
    summary: "Two stored tensors assessed.",
    detail: "Storage encoding is not an execution dtype contract.",
  },
}, {
  generatedAt: "2026-07-22T00:00:00.000Z",
  scope: {
    id: "gguf",
    label: "GGUF",
    completion: "GGUF container audit run complete",
    evidenceClass: "OBSERVED / DERIVED",
    depth: "Container and tensor audit",
    assessed: "Header, metadata, tensor descriptors, payload ranges, and storage encodings",
    runtimeStatus: "Not observed",
    runtimeBoundary: "Execution graph and backend dispatch are runtime-defined",
    releaseStatus: "Not assessed",
  },
});
expect(publicSummary.includes("Operators: Not applicable from a tensor container"), "Public GGUF summary must not render a misleading zero-op graph claim.");
expect(publicSummary.includes("MAC assessment: Not applicable from this artifact representation"), "Public GGUF summary must retain MAC applicability semantics.");
const publicPrintHandler = appSource.slice(
  appSource.indexOf("bindPublicAuditPrintButton(printPublicReport"),
  appSource.indexOf('copyReportBtn?.addEventListener("click"'),
);
expect(!publicPrintHandler.includes("loadEngineeringFormatter"), "Login-free public print must not load the controlled Engineering Report formatter.");

const sessionPrivacy = buildSessionPrivacy({
  consentLog: [
    { kind: "consent-restored", at: "2026-07-20T00:00:00.000Z", event_id: "evt-consent" },
    { kind: "structure-shared", at: "2026-07-21T00:00:00.000Z", fingerprint: "fp-current" },
  ],
  agreementRecord: { accepted_at: "2026-07-19T00:00:00.000Z" },
  researchConsent: true,
  structureTelemetryState: { fingerprint: "fp-current" },
  policyVersion: "policy.v1",
  historySaved: true,
});
expectEqual(sessionPrivacy.consentTimestamp, "2026-07-20T00:00:00.000Z", "Session privacy should use the latest explicit consent event.");
expectEqual(sessionPrivacy.telemetryCurrentShareAt, "2026-07-21T00:00:00.000Z", "Session privacy should bind telemetry disclosure to the current fingerprint.");
expectEqual(sessionPrivacy.telemetryFingerprint, "fp-current", "Session privacy should preserve the shared structure fingerprint.");
expectEqual(sessionPrivacy.historySaved, true, "Session privacy should disclose local history persistence.");

const sessionContextSet = buildSessionReportContextSet({
  analysis: { filename: "contract.onnx" },
  identity: { filename: "contract.onnx", sha256: "abc" },
  runtimeBenchmarkResults: [{ backend: "wasm", ok: true }],
  deepBomResult: { status: "complete" },
  perturbationResult: { status: "partial" },
  runtimeBasinResult: { status: "complete" },
  deployCurvatureResult: { status: "complete" },
  preprocessingConsequenceResult: { status: "complete" },
  runtimeAssignmentEvidence: { schema: "deepbom.runtime_assignment.v1" },
  browserRuntime: {
    browserBucket: "chromium-test",
    sharedArrayBufferAvailable: true,
    webgpuAvailable: true,
    webnnAvailable: false,
  },
  fileSizeBytes: 4096,
  generatedAt: "2026-07-22T00:00:00.000Z",
});
expectEqual(sessionContextSet.reportContext.generatedAt, "2026-07-22T00:00:00.000Z", "Session report context should preserve its explicit timestamp.");
expectEqual(sessionContextSet.regulatoryReportContext.fileSizeBytes, 4096, "Session report context should preserve artifact size for regulatory output.");
expectEqual(sessionContextSet.rawEvidenceContext.runtimeEvidence.browserBucket, "chromium-test", "Session report context should preserve the browser evidence bucket.");
expectEqual(sessionContextSet.rawEvidenceContext.runtimeEvidence.runtimeAssignmentEvidence?.schema, "deepbom.runtime_assignment.v1", "Session report context should preserve runtime assignment evidence.");
expectEqual(sessionContextSet.rawEvidenceContext.weightIndicatorEvidence.perturbationResult?.status, "partial", "Session report context should preserve weight-indicator module evidence.");

const reportBoundaryAnalysis = JSON.parse(gunzipSync(readFileSync(new URL("./fixtures/report-boundary-analysis.v1.json.gz", import.meta.url))).toString("utf8"));
const reportFixtureFileSize = reportBoundaryAnalysis.file_size || reportBoundaryAnalysis.file_size_bytes;
reportBoundaryAnalysis.artifact_byte_integrity = {
  schema: "deepbom.tflite_artifact_byte_integrity.v1",
  status: "assessed_clean",
  evidence_class: "DERIVED",
  file_size: reportFixtureFileSize,
  flatbuffer_referenced_range_count: 2,
  flatbuffer_referenced_bytes: reportFixtureFileSize - 2,
  flatbuffer_referenced_end: reportFixtureFileSize,
  flatbuffer_envelope_bytes: reportFixtureFileSize,
  flatbuffer_internal_alignment_or_unreferenced_bytes: 2,
  flatbuffer_referenced_ranges: [
    { offset: 0, end: reportFixtureFileSize - 100, length: reportFixtureFileSize - 100, class: "verified_flatbuffer_reference" },
    { offset: reportFixtureFileSize - 98, end: reportFixtureFileSize, length: 98, class: "verified_flatbuffer_reference" },
  ],
  terminal_zero_alignment_bytes: 0,
  metadata_archive_status: "not_present",
  metadata_archive_start: null,
  metadata_archive_end: null,
  metadata_archive_central_directory_start: null,
  metadata_archive_central_directory_end: undefined,
  metadata_archive_eocd_offset: null,
  metadata_archive_bytes: 0,
  metadata_archive_file_count: 0,
  metadata_archive_case_insensitive_name_collision_count: 0,
  metadata_archive_size_policy_bytes: 134217728,
  metadata_archive_size_policy_exceeded: false,
  unowned_trailing_bytes: 0,
  unowned_trailing_ranges: [],
  exact_shared_buffer_range_count: 0,
  partial_buffer_overlap_count: 0,
  flatbuffer_archive_overlap_bytes: 0,
  classified_bytes: reportFixtureFileSize,
  conservation_status: "exact",
  issue_count: 0,
  issues: [],
  method: "fixture byte-ledger method",
  detail: "fixture byte-ledger detail",
};

const unregisteredMetricCoverage = buildMetricCoverageManifest({
  ...reportBoundaryAnalysis,
  future_calculation: { schema: "deepbom.future_calculation.v1", status: "assessed", evidence_class: "DERIVED" },
  future_scalar: 17,
  future_rows: [{ value: 1 }],
});
expectEqual(unregisteredMetricCoverage.unregistered_computation_object_keys.join(","), "future_calculation", "Metric coverage discovery must reject a new structured calculation until it has report/viewer/export bindings.");
expectEqual(unregisteredMetricCoverage.unregistered_analysis_object_keys.join(","), "future_calculation,future_rows,future_scalar", "Metric coverage must reject unowned scalar, array, and structured top-level analysis fields.");
const reportBoundaryIdentity = {
  filename: "boundary_check.tflite",
  format: "tflite",
  sha256: "abc123",
  target_label: "Boundary target",
};
const engineeringReport = buildEngineeringReport(reportBoundaryAnalysis, { identity: reportBoundaryIdentity });
expectContains(engineeringReport, "Artifact Byte Integrity Ledger (DERIVED)", "Engineering Report should expose the artifact byte-integrity ledger.");
expectContains(engineeringReport, "fixture byte-ledger method", "Engineering Report should preserve the byte-ledger method.");
const publicByteLedgerBom = buildPublicCycloneDx17ArtifactContract(reportBoundaryAnalysis, { generatedAt: "2026-08-15T00:00:00.000Z" });
const publicByteLedgerProperties = new Map(publicByteLedgerBom.metadata.component.properties.map((property) => [property.name, property.value]));
expectEqual(publicByteLedgerProperties.get("deepbom:model:artifactByteIntegritySchema"), reportBoundaryAnalysis.artifact_byte_integrity.schema, "CycloneDX should expose the byte-integrity schema.");
expect(/^[a-f0-9]{64}$/.test(publicByteLedgerProperties.get("deepbom:model:artifactByteIntegrityLedgerSha256") || ""), "CycloneDX should bind the full byte-integrity ledger by SHA-256.");
expectEqual(
  publicByteLedgerProperties.get("deepbom:model:artifactByteIntegrityLedgerSha256"),
  sha256TextHex(canonicalJson({ ...reportBoundaryAnalysis.artifact_byte_integrity, metadata_archive_central_directory_end: null })),
  "CycloneDX should normalize an absent WASM Option value to JSON null before hashing the byte-integrity ledger.",
);
const engineeringReportArtifacts = buildEngineeringReportArtifacts(reportBoundaryAnalysis, { identity: reportBoundaryIdentity });
expectContains(engineeringReport, "**Model artifact:** `boundary_check.tflite`", "Engineering Report should identify the audited model at the top.");
expectContains(engineeringReport, "(37.5% inactive lanes)", "A single SIMD tail estimate should render as one exact percentage.");
expect(!engineeringReport.includes("37.5%..37.5%"), "Equal SIMD tail endpoints must not render as a duplicated percentage range.");
expectContains(engineeringReport, "Modeled candidate-set lane utilization: 62.5%", "SIMD finding should use the same collapsed range formatter as the report summary.");
expect(engineeringReport.indexOf("**Model artifact:** `boundary_check.tflite`") < engineeringReport.indexOf("## Read First"), "Engineering Report model identity should precede every analysis section.");
expectEqual(engineeringReportArtifacts.metricCoverage.schema, "deepbom.metric_coverage_manifest.v1.55", "Engineering Report should emit field-level metric coverage schema v1.55 with canonical Artifact IR ownership, ONNX conflict-capsule traceability, and the existing format-specific static/runtime decision bindings.");
expectEqual(engineeringReportArtifacts.metricCoverage.coverage_status, "partial", "A static-only Engineering Report should preserve not-assessed runtime and product-validation families rather than claiming complete coverage.");
expectEqual(engineeringReportArtifacts.metricCoverage.unregistered_computation_object_keys.length, 0, "Engineering Report should not leave a structured calculation without a report/viewer/export binding.");
expectEqual(engineeringReportArtifacts.metricCoverage.unregistered_analysis_object_keys.length, 0, "Engineering Report should not leave an analysis field without an ownership binding.");
const browserTimedAnalysis = {
  ...reportBoundaryAnalysis,
  static_audit_timing: {
    schema: "deepbom.static_audit_timing.v1",
    evidence_class: "MEASURED_BROWSER_WALL_CLOCK",
    wall_ms: 123.456,
    core_static_analysis_ms: 98.765,
    comparison_target_count: 1,
  },
};
const browserTimedArtifacts = buildEngineeringReportArtifacts(browserTimedAnalysis, { identity: reportBoundaryIdentity });
expectEqual(browserTimedArtifacts.metricCoverage.unregistered_computation_object_keys.length, 0, "Browser audit timing must be owned by a cross-format metric family.");
expectContains(browserTimedArtifacts.report, "Browser static-audit timing", "Engineering Report should expose measured browser audit timing.");
expectContains(browserTimedArtifacts.report, "not inference latency", "Engineering Report should distinguish analyzer workflow timing from inference latency.");
const decisionCoverage = engineeringReportArtifacts.metricCoverage.decision_coverage;
expectEqual(decisionCoverage.schema, "deepbom.decision_coverage.v1.11", "Decision coverage should expose a versioned machine-readable schema with canonical Artifact IR assignment and the existing format-specific deployment/runtime ownership while preserving decision-domain conservation.");
expectEqual(decisionCoverage.assigned_metric_count, decisionCoverage.applicable_metric_count, "Every applicable metric family must belong to exactly one decision domain.");
expectEqual(decisionCoverage.unassigned_metric_ids.length, 0, "Decision coverage must not leave an applicable metric family unassigned.");
expectEqual(decisionCoverage.multiply_assigned_metric_ids.length, 0, "Decision coverage must not count one metric family in multiple decision domains.");
for (const row of decisionCoverage.rows) {
  expectEqual(Object.values(row.status_counts).reduce((sum, count) => sum + count, 0), row.metric_count, `Decision domain ${row.domain_id} must conserve every assigned metric across all five statuses.`);
}
expectEqual(decisionCoverage.rows.find((row) => row.domain_id === "runtime_observation")?.status, "not_assessed", "A static-only report must not imply runtime observation.");
expectEqual(decisionCoverage.rows.find((row) => row.domain_id === "product_validation")?.status, "not_assessed", "Static artifact analysis must not imply task-quality validation.");
expectContains(engineeringReport, "## Decision Coverage At A Glance (DERIVED)", "Engineering Report should front-load the conserved decision boundary.");
expect(engineeringReport.indexOf("## Model At A Glance") > engineeringReport.indexOf("## Read First") && engineeringReport.indexOf("## Model At A Glance") < engineeringReport.indexOf("## Static Audit Conclusion"), "Engineering Report should place practical model metrics before the proof chain.");
expectContains(engineeringReport, "selected 64 KiB L1D / 256 KiB L2 references; watch >=0.90x", "Model At A Glance should bind cache pressure to the selected denominators and threshold.");
expectContains(engineeringReport, reportBoundaryAnalysis.target_profile.cache_assumption, "Engineering Report should preserve the target cache-assumption boundary.");
expectContains(engineeringReport, "selected microkernel", "Decision coverage should state the executed-kernel residual explicitly.");
expectContains(engineeringReport, "Static artifact evidence cannot certify accuracy", "Decision coverage should state the task-quality boundary explicitly.");
const tamperedDecisionCoverage = structuredClone(engineeringReportArtifacts.metricCoverage);
tamperedDecisionCoverage.decision_coverage.rows[0].status = "not_assessed";
expect(validateMetricCoverageManifest(tamperedDecisionCoverage, engineeringReport).some((failure) => failure.includes("decision coverage")), "Metric coverage validation must reject a tampered decision-domain status.");
const tamperedDecisionStatusCount = structuredClone(engineeringReportArtifacts.metricCoverage);
tamperedDecisionStatusCount.decision_coverage.rows[0].status_counts.not_applicable += 1;
expect(validateMetricCoverageManifest(tamperedDecisionStatusCount, engineeringReport).some((failure) => failure.includes("decision coverage")), "Metric coverage validation must reject a non-conserving decision-domain status count.");
expect(engineeringReportArtifacts.metricCoverage.field_coverage.leaf_field_pattern_count > 0, "Field coverage should discover nested analysis leaf patterns.");
expect(engineeringReportArtifacts.metricCoverage.field_coverage.report_consumed_field_pattern_count > 0, "Field coverage should record leaf patterns consumed by the report generator.");
expect(engineeringReportArtifacts.metricCoverage.field_coverage.required_report_field_pattern_count > 0, "Field coverage should register emitted decision-critical report fields.");
expectEqual(engineeringReportArtifacts.metricCoverage.field_coverage.missing_required_report_field_count, 0, "Engineering Report should consume every emitted decision-critical field.");
expect(engineeringReportArtifacts.metricCoverage.field_coverage.raw_evidence_only_field_pattern_count > 0, "Field coverage should disclose analysis fields retained only in machine-readable evidence.");
expectEqual(engineeringReportArtifacts.metricCoverage.field_coverage.unbound_field_pattern_count, 0, "Report-only field coverage should not invent unbound fields before raw export binding is checked.");
expectContains(engineeringReport, "Nested field coverage:", "Engineering Report should summarize field-level report/raw routing.");
const incompleteStaticExport = buildStaticAnalysisExport(reportBoundaryAnalysis);
incompleteStaticExport.target_profile = { ...incompleteStaticExport.target_profile };
delete incompleteStaticExport.target_profile.fp32_lanes;
const incompleteFieldCoverage = buildMetricCoverageManifest(reportBoundaryAnalysis, {
  evidenceRoot: {
    evidence: { static_analysis: incompleteStaticExport, findings_register: { findings: [] } },
    supplemental_sources: { roofline_csv: reportBoundaryAnalysis.roofline_csv, stage_graph_mermaid: reportBoundaryAnalysis.stage_mermaid },
  },
  reportAccessedFieldPaths: engineeringReportArtifacts.reportAccessedFieldPaths,
});
expect(incompleteFieldCoverage.field_coverage.unbound_field_paths.includes("/target_profile/fp32_lanes"), "Field coverage must identify a nested scalar omitted from the machine-readable static export.");
expectEqual(incompleteFieldCoverage.coverage_status, "fail", "An unbound nested field must fail metric coverage.");
const nonJsonSafeCoverage = buildMetricCoverageManifest({ ...reportBoundaryAnalysis, total_macs: Number.POSITIVE_INFINITY });
expect(nonJsonSafeCoverage.field_coverage.non_json_safe_field_paths.includes("/total_macs"), "Field coverage must identify a non-finite number that JSON export would coerce to null.");
expectEqual(nonJsonSafeCoverage.coverage_status, "fail", "A non-JSON-safe analysis value must fail metric coverage.");
const omittedRequiredPath = "/target_profile/fp32_lanes";
const missingRequiredCoverage = buildMetricCoverageManifest(reportBoundaryAnalysis, {
  reportAccessedFieldPaths: engineeringReportArtifacts.reportAccessedFieldPaths.filter((path) => path !== omittedRequiredPath),
});
expect(missingRequiredCoverage.field_coverage.missing_required_report_field_paths.includes(omittedRequiredPath), "Field coverage must identify a decision-critical emitted field omitted from the Engineering Report.");
expectEqual(missingRequiredCoverage.coverage_status, "fail", "An unconsumed decision-critical report field must fail metric coverage.");
expectContains(engineeringReport, "F32 GEMM AArch64 NEON FMA assembly MR {1,4,6} NR8", "Engineering Report should render the assembly selector candidate.");
expectContains(engineeringReport, "F32 GEMM AArch64 NEON FMA intrinsics MR {1,6} NR8", "Engineering Report should render the intrinsics selector candidate.");
expectContains(engineeringReport, ANALYZER_METADATA.rulepackProvenance.xnnpackGemmConfigSha256, "Engineering Report should bind selector candidates to the pinned source-file SHA-256.");
expectContains(engineeringReport, ANALYZER_METADATA.rulepackProvenance.xnnpackReadmeSha256, "Engineering Report should bind delegation predictions to the pinned README SHA-256.");
expectContains(engineeringReport, ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateRuleManifestSha256, "Engineering Report should bind delegation predicates to the semantic manifest SHA-256.");
expectContains(engineeringReport, "133/133 artifact-visible constraints mapped; 0 unmapped; 2 rulepack runtime-only requirement(s), 1 artifact-applicable", "Engineering Report should disclose rulepack-wide and artifact-applicable build-flag coverage separately.");
expectContains(engineeringReport, "~6.25 ops/B", "Engineering Report should calculate the FP32 ridge from the exact fixture peak and bandwidth constants.");
expectContains(engineeringReport, "separate from the configured peak-throughput ratio", "Engineering Report should separate the reduced-precision formula factor from the target peak ratio.");
expectContains(engineeringReport, "structural/view candidate", "Engineering Report should not label a structural view operator as an observed copy.");
expectNotContains(engineeringReport, "copy-like", "Engineering Report should remove the ambiguous copy-like label.");
expectContains(engineeringReport, "Pinned XNNPACK source commit", "Engineering Report should label the selector source revision explicitly.");
expectContains(engineeringReport, "XNN_ENABLE_ASSEMBLY", "Engineering Report should preserve unresolved compile-time selector conditions.");
expectContains(engineeringReport, "candidate configuration does not identify the executed runtime microkernel", "Engineering Report should preserve the selector-candidate evidence boundary.");
const heuristicEngineeringReport = buildEngineeringReport({
  ...reportBoundaryAnalysis,
  xnnpack_selector_assessment_status: "not_loaded",
  xnnpack_selector_evidence_schema: "",
  xnnpack_selector_evidence_access: "research_authorization_required",
  ops: reportBoundaryAnalysis.ops.map((op) => ({
    ...op,
    xnnpack_kernel_candidates: [],
    xnnpack_kernel_evidence_class: "HEURISTIC_PROFILE",
  })),
}, { identity: reportBoundaryIdentity });
expectContains(heuristicEngineeringReport, "output-channel microkernel tile not source-enumerated", "Engineering Report should avoid presenting SIMD lanes as an enumerated output-channel kernel tile.");
expectContains(heuristicEngineeringReport, "SIMD 8-bit integer lanes x16", "Engineering Report should expose dtype-neutral 8-bit register-lane posture.");
expectContains(heuristicEngineeringReport, "register capacity FP16 x8; native vector arithmetic not source-bound", "Engineering Report should separate FP16 register capacity from native arithmetic support.");
expectContains(heuristicEngineeringReport, "SIMD FP32 lanes x4", "Engineering Report should expose FP32 register-lane posture.");
expectContains(heuristicEngineeringReport, "controlled source-backed selector module was not loaded", "Engineering Report should distinguish an unloaded protected selector from a source-enumerated zero result.");
const runtimeBoundAnalysis = {
  ...reportBoundaryAnalysis,
  model_sha256: "d".repeat(64),
  target_profile: { ...reportBoundaryAnalysis.target_profile, profile_sha256: "e".repeat(64) },
  deployment_frontier: {
    schema: "deepbom.deployment_frontier.v1.6",
    evidence_class: "DERIVED_FROM_PINNED_TARGET_PROFILE_STATIC_ESTIMATES",
    artifact_sha256: "d".repeat(64),
    target_count: 2,
    op_count: 1,
    cache_watch_ratio: 0.9,
    targets: [
      { target_id: "a", target_label: "Target A", l1_data_bytes: 65536, l2_bytes: 262144, l1_watch_count: 0, l2_watch_count: 0, max_l1_ratio: 0.02, max_l2_ratio: 0.005, total_us: 1, top_op_index: 0, top_op_name: "CONV_2D", top_op_us: 1 },
      { target_id: "b", target_label: "Target B", l1_data_bytes: 32768, l2_bytes: 131072, l1_watch_count: 0, l2_watch_count: 0, max_l1_ratio: 0.04, max_l2_ratio: 0.01, total_us: 2, top_op_index: 0, top_op_name: "CONV_2D", top_op_us: 2 },
    ],
    robust_coverage: {
      threshold: 0.8,
      selected_op_indices: [0],
      selected_op_count: 1,
      minimum_union_coverage: 1,
      per_target: [
        { target_id: "a", selected_prefix_op_count: 1, union_coverage: 1 },
        { target_id: "b", selected_prefix_op_count: 1, union_coverage: 1 },
      ],
    },
    target_divergence: {
      mean_normalized_jensen_shannon_divergence: 0,
      max_normalized_jensen_shannon_divergence: 0,
      min_coverage_prefix_jaccard: 1,
      pairs: [{
        left_target_id: "a",
        right_target_id: "b",
        normalized_jensen_shannon_divergence: 0,
        coverage_prefix_jaccard: 1,
        attribution_sum: 0,
        attribution_prefix_op_count: 0,
        attribution_prefix_coverage: 1,
        top_driver_op_index: null,
        top_driver_op_name: null,
        top_driver_attribution_share: 0,
        bound_transition_op_count: 0,
        dominant_component_transition_op_count: 0,
        drivers: [{
          op_index: 0,
          op_name: "CONV_2D",
          normalized_js_contribution: 0,
          attribution_share: 0,
          component_contribution_delta: { largest_absolute_component: "memory", memory: 0 },
        }],
      }],
    },
    interventions: [],
    ops: [{ op_index: 0, op_name: "CONV_2D", in_robust_coverage_union: true, min_contribution_share: 1, max_contribution_share: 1, best_rank: 1, worst_rank: 1, rank_span: 0, bound_classes: ["memory-bound"], dominant_components: ["memory"] }],
    method: "Fixture target contribution distributions.",
    interpretation_boundary: "Fixture static estimate; not measured runtime.",
  },
};
const runtimeAssignment = parseRuntimeAssignmentDocument(JSON.stringify({
  schema: "deepbom.runtime_assignment.v1.2",
  artifact_sha256: runtimeBoundAnalysis.model_sha256,
  target_profile_id: runtimeBoundAnalysis.target_profile.id,
  target_profile_sha256: runtimeBoundAnalysis.target_profile.profile_sha256,
  runtime: { name: "LiteRT", version: "2.0", backend: "XNNPACK", build: "fixture-release" },
  source: {
    kind: "interpreter_plan_export",
    collected_at: "2026-07-16T00:00:00.000Z",
    assignment_semantics: "original_graph_op_assignment",
    partition_semantics: "partition_id_identifies_runtime_partition_when_present",
    duration_semantics: "not_collected",
  },
  assignments: [{ op_index: 0, op_name: "CONV_2D", provider: "TFLite CPU", delegated: false }],
}), runtimeBoundAnalysis);
const runtimeBoundIdentity = { ...reportBoundaryIdentity, sha256: runtimeBoundAnalysis.model_sha256 };
const runtimeMlBom = buildMlBomDocument(runtimeBoundAnalysis, { hash: runtimeBoundAnalysis.model_sha256, targetId: runtimeBoundAnalysis.target_profile.id });
const runtimeReportContext = { identity: runtimeBoundIdentity, runtimeEvidence: { runtimeAssignmentEvidence: runtimeAssignment } };
const runtimeEvidenceContext = { identity: runtimeBoundIdentity, runtimeEvidence: { runtimeAssignmentEvidence: runtimeAssignment } };
const runtimeEngineeringEvidence = buildEngineeringEvidenceDocument(runtimeBoundAnalysis, {
  reportContext: runtimeReportContext,
  rawEvidenceContext: runtimeEvidenceContext,
  mlBomDocument: runtimeMlBom,
});
expectEqual(runtimeEngineeringEvidence.evidence?.conformance_report?.status, "pass", "Runtime assignment comparison should pass independent semantic conformance.");
expectEqual(runtimeEngineeringEvidence.evidence?.runtime_results?.runtime_assignment?.comparison?.placement_assessment?.overpredicted_delegation_count, 1, "Runtime evidence should preserve static delegation overprediction.");
expectEqual(runtimeEngineeringEvidence.evidence?.execution_placement?.runtime_observation?.covered_item_count, 1, "Normalized placement evidence should preserve imported original-op assignment coverage.");
expectEqual(runtimeEngineeringEvidence.evidence?.execution_placement?.flow?.evidence_basis, "OBSERVED_RUNTIME_ASSIGNMENT", "Normalized placement flow should switch from static prediction to imported runtime assignment.");
expect(runtimeEngineeringEvidence.evidence?.findings_register?.findings?.some((item) => item.finding_id === "EA-DEL-0003" && item.evidence_class === "DERIVED_FROM_OBSERVED_RUNTIME"), "Observed delegation mismatch should enter the authoritative findings register.");
expect(runtimeEngineeringEvidence.supplemental_sources?.runtime_assignment_comparison_csv?.includes("overpredicted_delegation"), "Consolidated evidence should include runtime assignment comparison CSV.");
const assignmentCsv = buildRuntimeAssignmentComparisonCsv(runtimeAssignment);
const boundaryCsv = buildRuntimeBoundaryComparisonCsv(runtimeAssignment);
expect(assignmentCsv.includes("artifact_sha256,target_profile_id,target_profile_sha256,runtime_name") && assignmentCsv.includes("duration_semantics,evidence_class"), "Runtime assignment CSV should retain standalone provenance and timing semantics.");
expect(boundaryCsv.includes("tensor_dtype,tensor_shape") && boundaryCsv.includes("observed_relation_reason,payload_bytes,materialization_status"), "Runtime boundary CSV should retain tensor and boundary-decision evidence.");
expectEqual(csvCell("=HYPERLINK(\"https://example.invalid\")"), "\"'=HYPERLINK(\"\"https://example.invalid\"\")\"", "CSV string cells should neutralize spreadsheet formulas.");
expectEqual(csvCell(-1), "-1", "Numeric CSV cells should remain numeric after formula-injection hardening.");
const runtimeEngineeringReport = buildEngineeringReport(runtimeBoundAnalysis, runtimeReportContext);
expect(runtimeEngineeringReport.includes("### Predicted Vs Observed Runtime Assignment"), "Engineering Report should render predicted-vs-observed runtime assignment evidence.");
expect(runtimeEngineeringReport.includes("Observed runtime assignment differs from static delegation prediction"), "Engineering Report action queue should include the observed delegation mismatch finding.");
const browserSamples = [1, 2, 3];
const browserStats = latencyStats(browserSamples);
const browserNoise = benchmarkNoise(browserSamples);
const browserBenchmark = {
  backend: "wasm",
  ok: true,
  compile_ms: 4,
  first_run_ms: 3,
  warmup: 2,
  runs: 3,
  timing_method: BROWSER_BENCHMARK_TIMING_METHOD,
  phase_counts: { cold_first_runs: 1, warmup_runs: 2, measured_runs: 3 },
  stats: browserStats,
  steady_stats: { ...browserStats },
  measured_samples_ms: browserSamples,
  statistics_method: BROWSER_BENCHMARK_STATISTICS_METHOD,
  noise_method: BROWSER_BENCHMARK_NOISE_METHOD,
  noise_diagnostics: browserNoise,
  input_basis: "declared_static_shape",
  input_contracts: [{ input_name: "input", artifact_dtype: "FLOAT32", runtime_dtype: "float32", declared_shape: [1, 4], artifact_shape_signature: [-1, 4], runtime_declared_shape: [1, 4], executed_shape: [1, 4], element_count: 4, basis: "declared_static_shape" }],
  output_count: 1,
  output_contracts: [{ output_name: "output", artifact_dtype: "FLOAT32", runtime_dtype: "float32", declared_shape: [1, 2], artifact_shape_signature: [1, 2], runtime_declared_shape: [1, 2], executed_shape: [1, 2], element_count: 2, basis: "observed_runtime_output" }],
  output_digest: "a".repeat(64),
  generated_at: "2026-07-21T00:00:00.000Z",
};
const browserBenchmarkEvidence = buildEngineeringEvidenceDocument(runtimeBoundAnalysis, {
  reportContext: { identity: runtimeBoundIdentity, runtimeBenchmarkResults: [browserBenchmark] },
  rawEvidenceContext: { identity: runtimeBoundIdentity, runtimeEvidence: { benchmarkResults: [browserBenchmark] } },
  mlBomDocument: runtimeMlBom,
});
expectEqual(browserBenchmarkEvidence.evidence?.conformance_report?.status, "pass", "A successful browser benchmark should pass exact timing-phase and executed-input conformance.");
expectEqual(browserBenchmarkEvidence.evidence?.runtime_results?.runtime_execution_status, "executed", "A successful browser benchmark should establish runtime execution evidence.");
expectThrows(() => buildEngineeringEvidenceDocument(runtimeBoundAnalysis, {
  reportContext: { identity: runtimeBoundIdentity, runtimeBenchmarkResults: [{ ...browserBenchmark, input_contracts: [{ ...browserBenchmark.input_contracts[0], element_count: 5 }] }] },
  rawEvidenceContext: { identity: runtimeBoundIdentity, runtimeEvidence: { benchmarkResults: [{ ...browserBenchmark, input_contracts: [{ ...browserBenchmark.input_contracts[0], element_count: 5 }] }] } },
  mlBomDocument: runtimeMlBom,
}), "(CF-RUNTIME-BENCH-001)", "A successful benchmark with non-conserving input element count must fail evidence conformance.");
expectThrows(() => buildEngineeringEvidenceDocument(runtimeBoundAnalysis, {
  reportContext: { identity: runtimeBoundIdentity, runtimeBenchmarkResults: [{ ...browserBenchmark, stats: { ...browserStats, p50: 99 } }] },
  rawEvidenceContext: { identity: runtimeBoundIdentity, runtimeEvidence: { benchmarkResults: [{ ...browserBenchmark, stats: { ...browserStats, p50: 99 } }] } },
  mlBomDocument: runtimeMlBom,
}), "(CF-RUNTIME-BENCH-001)", "A benchmark statistic that cannot be reconstructed from raw samples must fail evidence conformance.");
const browserBenchmarkReport = buildEngineeringReport(runtimeBoundAnalysis, { identity: runtimeBoundIdentity, runtimeBenchmarkResults: [browserBenchmark] });
expect(browserBenchmarkReport.includes(BROWSER_BENCHMARK_TIMING_METHOD), "Engineering Report should render the exact successful benchmark timing method.");
expect(browserBenchmarkReport.includes("artifact [1x4] / signature [-1x4] / runtime [1x4] / executed [1x4] / 4 elements"), "Engineering Report should preserve the complete successful benchmark input contract.");
expect(browserBenchmarkReport.includes("output:float32; artifact [1x2] / signature [1x2] / runtime [1x2] / executed [1x2] / 2 elements"), "Engineering Report should preserve the complete observed output contract.");
expect(browserBenchmarkReport.includes(BROWSER_BENCHMARK_STATISTICS_METHOD) && browserBenchmarkReport.includes("1, 2, 3"), "Engineering Report should preserve the exact measured samples and statistics method.");
expect(browserBenchmarkReport.includes(BROWSER_BENCHMARK_NOISE_METHOD) && browserBenchmarkReport.includes("trimmed p50 2 ms"), "Engineering Report should preserve reconstructible noise diagnostics and method.");
expect(browserBenchmarkReport.includes("| Runtime execution observation | partial | 1 assessed / 0 partial / 3 not assessed / 0 not applicable / 0 suppressed | MEASURED_SYNTHETIC |"), "A completed browser benchmark should promote only the executed runtime metric while leaving assignment, arena, and representative-dataset observations explicitly unobserved.");
const privacyReport = buildEngineeringReport(reportBoundaryAnalysis, {
  identity: reportBoundaryIdentity,
  generatedAt: "2026-07-15T00:00:00.000Z",
  sessionPrivacy: {
    consent: true,
    consentTimestamp: "2026-07-14T12:34:56.000Z",
    consentEventId: "consent-secret-123",
    telemetryCurrentShareAt: "2026-07-15T01:02:03.000Z",
    telemetryLastShareAt: "2026-07-14T01:02:03.000Z",
    telemetryFingerprint: "telemetry-secret-456",
  },
});
expect(!privacyReport.includes("consent-secret-123") && !privacyReport.includes("telemetry-secret-456"), "Engineering report should redact consent IDs and telemetry fingerprints.");
expect(!privacyReport.includes("2026-07-14T12:34:56.000Z") && !privacyReport.includes("2026-07-15T01:02:03.000Z"), "Engineering report should redact browser-session timestamps.");
const privacyManifest = buildEngineeringBundleManifest({
  analysis: reportBoundaryAnalysis,
  model: reportBoundaryIdentity,
  files: [{ name: "engineering_report.md", data: privacyReport }],
  generatedAt: "2026-07-15T00:00:00.000Z",
});
expectEqual(privacyManifest.included_report_privacy?.consent_event_id_embedded, false, "Included-report privacy should confirm consent ID redaction.");
expectEqual(privacyManifest.included_report_privacy?.telemetry_fingerprint_embedded, false, "Included-report privacy should confirm telemetry fingerprint redaction.");
expectEqual(privacyManifest.included_report_privacy?.exact_browser_session_timestamp_embedded, false, "Included-report privacy should confirm browser-session timestamp redaction.");
expectEqual(privacyManifest.included_report_privacy?.report_generated_timestamp_embedded, true, "Included-report privacy should separately disclose the report generation timestamp.");
const filenameOnlyChange = buildChangeAnalysis({ ...reportBoundaryAnalysis, filename: "model_v2.tflite", model_sha256: "new" }, {
  priorSnapshot: { filename: "model_v1.tflite", sha256: "old", format: "tflite", target: reportBoundaryAnalysis.target_profile.id },
  identity: reportBoundaryIdentity,
});
expectEqual(filenameOnlyChange.status, "not_performed", "Filename similarity alone must not establish model lineage.");
expectEqual(filenameOnlyChange.reason_code, "LINEAGE_NOT_ESTABLISHED", "Rejected filename-only comparison should expose a structured reason.");
const explicitLineageChange = buildChangeAnalysis({ ...reportBoundaryAnalysis, model_sha256: "new", model_lineage_id: "lineage-1" }, {
  priorSnapshot: { sha256: "old", format: "tflite", target: reportBoundaryAnalysis.target_profile.id, modelLineageId: "lineage-1" },
  identity: reportBoundaryIdentity,
});
expectEqual(explicitLineageChange.status, "assessed", "Matching model lineage IDs should allow deterministic comparison.");
expectEqual(explicitLineageChange.comparison_basis, "matching_model_lineage_id", "Change analysis should record its lineage basis.");
const incompleteMacChange = buildChangeAnalysis({ ...reportBoundaryAnalysis, model_sha256: "new", model_lineage_id: "lineage-1", total_macs: null }, {
  priorSnapshot: { sha256: "old", format: "tflite", target: reportBoundaryAnalysis.target_profile.id, modelLineageId: "lineage-1", totalMacs: reportBoundaryAnalysis.total_macs },
  identity: reportBoundaryIdentity,
});
expectEqual(incompleteMacChange.deltas.total_macs, null, "Change analysis must not coerce an unassessed MAC total to zero.");
const engineeringBundleSummary = buildEngineeringBundleSummary({ analysis: reportBoundaryAnalysis, identity: reportBoundaryIdentity });
const regulatoryBundleSummary = buildEvidenceBundleSummary({ analysis: reportBoundaryAnalysis, identity: reportBoundaryIdentity });

for (const [snippet, label] of [
  ["## Scope And Evidence Boundary", "open with scope/evidence boundary"],
  ["Not Evaluated From This Artifact Alone", "state artifact-only limitations"],
  ["`MEASURED_SYNTHETIC` values are local browser/runtime measurements", "label synthetic measurements"],
  [ANALYZER_METADATA.schemas.engineeringReport, "include its schema version"],
  [ANALYZER_METADATA.semanticVersion, "include DeepBOM semantic version"],
  [ANALYZER_METADATA.rulepackVersion, "include rulepack version"],
  [ANALYZER_METADATA.buildCommit, "include analyzer build commit"],
  [ANALYZER_METADATA.buildContentSha256, "include analyzer bundle content hash"],
  [ANALYZER_METADATA.rulepackSha256, "include rulepack source hash"],
  ["Analyzer bundle content SHA-256", "label analyzer bundle content hash"],
  ["Analyzer bundle content hash method", "state the deterministic build-content hash method"],
  ["Analyzer build-content manifest SHA-256", "bind the file-level build-content manifest"],
  ["Analyzer release provenance policy", "state clean-tree release policy"],
  ["Rulepack hash basis", "describe what the rulepack hash covers"],
  ["## Artifact Integrity Posture", "include artifact integrity posture"],
  ["Tensor buffers", "include tensor buffer posture"],
  ["3 tensor constant buffer(s), 2 unique region(s)", "summarize TFLite tensor buffer regions"],
  ["## Runtime Environment And Reproducibility", "include runtime environment and reproducibility notes"],
  ["p50/p90/p95/p99", "describe the steady-state latency protocol"],
  [RUNTIME_COMPATIBILITY_EVIDENCE_LABEL, "use report-facing runtime compatibility wording"],
  ["tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c", "pin XNNPACK rule provenance to a reproducible TensorFlow commit"],
  ["## Static Audit Conclusion", "front-load the static conclusion"],
  ["## Engineer Action Queue", "front-load the engineering action queue"],
  ["EA-IOC-0001", "include a preprocessing-contract finding"],
  ["EA-OUT-0001", "include an output-semantics finding"],
  ["EA-RUN-0001", "include a bundled-runtime-version finding"],
  ["EA-PKG-0001", "include a packing warmup finding when static estimates flag one"],
  ["EA-LIN-0001", "include source checkpoint/conversion lineage finding"],
  ["lineage requirement", "class lineage findings separately"],
  ["integration requirement", "class integration findings separately"],
  ["integration verification", "class runtime-version manifest findings separately"],
  ["Weight packing warmup cost watchlist", "surface already-computed packing estimates in action queue"],
  ["Input layout determination", "label layout determination explicitly"],
  ["Input contract derived risks", "surface analyzer-computed input contract risks"],
  ["Evidence: DERIVED", "promote graph-semantic layout when a TFLite consumer op determines it"],
  ["TFLite CONV_2D #000", "show the op-semantic basis for NHWC layout"],
  ["Profile source", "separate target-profile source"],
  ["Profile source basis", "state the exact embedded target-profile derivation basis"],
  ["ISA feature assumptions", "surface dot-product and SVE2 target-profile assumptions"],
  ["Static intensity heuristic bands", "surface the exact low/high intensity thresholds"],
  ["HEURISTIC_PROFILE hardware specification; HEURISTIC performance model; not device-calibrated", "separate target hardware confidence from performance-model confidence"],
  ["Static prioritization only; not a measured performance profile", "label target-profile use"],
  ["Assumed 8-bit:FP32 peak throughput ratio", "distinguish peak throughput ratio from formula factor"],
  ["Reduced-precision formula factor", "show the speedup formula factor explicitly"],
  ["## Peak Live Activation Payload (DERIVED)", "avoid implying a measured runtime arena"],
  ["Estimated peak live activation payload", "name the static liveness quantity precisely"],
  ["actual runtime arena may be lower through storage aliasing", "state arena caveats with runtime aliasing/view mechanisms"],
  ["Decodable constant tensors scanned", "avoid implying only trained weights were scanned"],
  ["Quantized constants decoded", "make quantized constant decoding explicit"],
  ["Kernel tensors evaluated", "split all-zero kernel-slice denominator fields"],
  ["Output channels evaluated", "split all-zero kernel-slice denominator fields"],
  ["Near-zero decoded kernel output slices", "separate decoded near-zero kernel-slice results"],
  ["Exact-zero stored kernel output slices", "separate stored centered-code exact-zero results"],
  ["Max \\|decoded constant\\|", "avoid implying all scanned constants are trained weights"],
  ["not assessed; no eligible Conv/FC/depthwise kernel layout was decoded", "avoid a misleading 0-across-0 dead-channel result"],
  ["near-zero means \\|x\\| < 1e-8", "define weight-integrity thresholds"],
  ["TFLite Model Metadata payload bytes", "narrow metadata-byte wording"],
  ["Residual bytes outside constant and metadata-buffer payloads", "name the residual byte denominator without claiming every byte is graph structure"],
  ["Hypothetical all-FLOAT32-to-INT8 scalar payload floor", "label theoretical INT8 size as an ideal payload floor"],
  ["Raw 0x00 byte ratio in constant buffers", "define zero-byte ratio as raw bytes"],
  ["target\\|pipe", "escape pipe characters in raw appendix table cells"],
  ["MAC-weighted 8-bit integer compute-kernel throughput ceiling", "qualify the reduced-precision opportunity number"],
  ["not an end-to-end model speedup estimate", "avoid presenting the throughput ceiling as a claim"],
  ["Reduced-precision method", "include INT8 speedup method row"],
  ["compute-kernel ceiling = 1 /", "include the INT8 compute-kernel ceiling formula"],
  ["## Artifact-side Runtime Requirements (OBSERVED/DERIVED)", "avoid claiming app runtime compatibility from artifact-only evidence"],
  ["Bundled application runtime version", "state whether the deployed runtime version was supplied"],
  ["Compatibility conclusion", "state app-runtime compatibility as assessable or not assessable"],
  ["NOT_ASSESSABLE; bundled application LiteRT/TFLite runtime version was not provided", "avoid inferring app runtime compatibility"],
  ["Observed op-version necessary floor", "derive a necessary runtime floor from op/version mapping without implying execution compatibility"],
  ["Effective artifact-side runtime floor", "state the effective artifact-side runtime floor"],
  ["Highest observed operator version", "name operator version evidence precisely"],
  ["Derived from pinned TensorFlow runtime-version map", "cite runtime-version derivation basis"],
  ["## Static Structural Triage (HEURISTIC)", "surface WASM-computed insight signals with the correct evidence class"],
  ["Requirement-aware assessment", "state that release criteria were not evaluated without deployment requirements"],
  ["Context-free composite score", "state explicitly that a readiness-looking composite is not reported"],
  ["not reported; component signals remain separate", "preserve individual evidence without a context-free 0-100 score"],
  ["No pass/fail or deployment-readiness meaning", "bound the heuristic score interpretation"],
  ["Component signals", "state the heuristic component signals without a readiness score"],
  ["Packing warn ops", "surface WASM-computed packing warning count"],
  ["Operator type histogram", "surface analyzer-computed op histogram"],
  ["Tensor dtype inventory", "surface analyzer-computed tensor dtype inventory"],
  ["## Computed Analysis Coverage (DERIVED/ESTIMATED)", "surface computed fields that otherwise only feed charts/raw exports"],
  ["## Quantization Research Coverage (DERIVED)", "surface artifact class, the 15-lab applicability denominator, and excluded analysis status"],
  ["Scan denominator policy", "separate not-applicable and not-assessed labs from defect-free scan denominators"],
  ["Operator code table entries", "surface parsed operator-code table count"],
  ["Total arithmetic ops", "surface analyzer-computed arithmetic op total"],
  ["Conditionally delegatable / predicted-fallback MACs", "surface predicted candidate/fallback MAC totals"],
  ["Conditionally delegatable / predicted-fallback logical bytes", "surface predicted candidate/fallback logical-byte totals"],
  ["XNNPACK predicted partition breakdown", "surface non-structural, structural zero-MAC, zero-MAC non-structural, and longest-segment counts"],
  ["Conv/FC weight op inventory", "surface computed weight-op inventory and packing warn counts"],
  ["Fallback traffic by op family", "surface computed fallback traffic family inventory"],
  ["Predicted structural delegate-break watchlist", "surface the TFLite structural delegate-break watchlist without implying ONNX EP support"],
  ["Topology annotations", "surface analyzer-computed topology role/depth/fan-out annotations"],
  ["Bottleneck component totals", "surface per-op bottleneck component totals"],
  ["HEURISTIC profile-derived cold-start composition", "label static cost outputs and separate steady-state from one-time packing"],
  ["display sum 100.0%", "make displayed component percentages sum exactly after rounding"],
  ["sum check ok", "verify bottleneck component arithmetic"],
  ["not target-device latency predictions", "bound static cost interpretation"],
  ["representative highest-ranked op #000 CONV_2D", "identify the representative channel-tail op"],
  ["all misaligned ops: #000 CONV_2D", "list every channel-tail candidate"],
  ["Raw native analyzer signals", "surface native WASM signal count without creating a second authoritative finding engine"],
  ["Raw native optimization hints", "surface native hint count without creating a second authoritative action source"],
  ["Activation precision boundary details", "surface serialized mid-graph 8-bit/FP32 boundary inventory without implying float-island regions"],
  ["Export-only computed artifacts", "identify roofline CSV and Mermaid stage graph computations"],
  ["## Stage And Pattern Summary (DERIVED)", "surface stage and pattern computations in the report body"],
  ["Stage partitioning rule", "state the stage grouping rule in the report body"],
  ["Observed output channels", "separate stage channel buckets from actual channel counts"],
  ["Conditionally delegatable ops", "separate conditionally delegatable op ratio from MAC ratio"],
  ["Pattern participation", "label cross-stage pattern participation explicitly"],
  ["Summed logical op bytes", "name stage bytes as summed logical op bytes"],
  ["## Movement And Packing Estimates (DERIVED/ESTIMATED)", "surface movement and packing estimates in the report body"],
  ["Explicit copy or movement operators", "surface explicit movement-op count"],
  ["Structural or view operators", "surface structural/view op count separately"],
  ["Predicted non-delegated operators", "surface predicted non-delegated op count separately"],
  ["Potential partition-interface logical payload", "surface graph-derived boundary-edge payload separately"],
  ["### Predicted Partition Boundary Edges (PREDICTED assignment / DERIVED payload)", "render the structured internal execution-domain edge inventory"],
  ["Confirmed runtime copy bytes", "state static artifact cannot confirm runtime copy bytes"],
  ["Target packing bandwidth", "surface target packing-bandwidth assumption"],
  ["Fixed packing setup overhead", "surface target packing setup assumption"],
  ["Packing watch threshold", "surface packing threshold assumption"],
  ["Model-content processing", "split local model processing privacy from other services"],
  ["Account and access services", "disclose non-model account/network services separately"],
  ["Telemetry schema version", "state structural telemetry schema"],
  ["Structural telemetry field categories", "list network payload categories"],
  ["Structural telemetry explicit exclusions", "list network payload exclusions"],
  ["Report-fingerprint registration payload", "separate fingerprint registration payload from core telemetry"],
  ["Signed package-attestation payload", "describe the actual attestation v2.1 payload"],
  ["Registered report-body SHA-256", "name the registered report-body digest without calling it a bundle digest"],
  ["Package digest attestation", "direct package verification to the non-self-referential attestation receipt"],
  ["Artifact SHA-256 transmitted", "state when artifact hashes are transmitted"],
  ["Filename transmitted", "state filename transmission behavior"],
  ["Account ID linked to telemetry", "state account linkage for consented telemetry"],
  ["Consent text/policy version", "state consent policy version capture boundary"],
  ["Retention period", "state telemetry retention capture boundary"],
  ["Consent event ID", "state consent event identifier when captured"],
  ["Telemetry endpoint", "state structure telemetry endpoint"],
  ["Structural telemetry architecture warning", "warn that structure can reveal architecture"],
  ["## Checkpoint-to-Artifact Provenance (NOT_ASSESSABLE)", "surface checkpoint-to-artifact lineage gap"],
  ["Source checkpoint SHA-256", "state source checkpoint lineage field"],
  ["Deletion or withdrawal mechanism", "state withdrawal/delete mechanism boundary"],
  ["Browser report history", "name IndexedDB/local history separately"],
  ["Deployment reviewers must obtain that contract separately", "keep preprocessing handoff factual"],
]) {
  expectContains(engineeringReport, snippet, `Engineering Report should ${label}.`);
}
const provenanceGapFindings = buildFindingsRegister({
  ...reportBoundaryAnalysis,
  target_profile: { ...reportBoundaryAnalysis.target_profile, profile_sha256: "" },
});
expect(provenanceGapFindings.some((finding) => finding.finding_id === "EA-PROV-0001"), "A missing target-profile digest should create EA-PROV-0001.");
const multiSubgraphAnalysis = { ...reportBoundaryAnalysis, subgraphs: 3 };
const multiSubgraphFinding = buildFindingsRegister(multiSubgraphAnalysis).find((finding) => finding.finding_id === "EA-GRF-0001");
expect(Boolean(multiSubgraphFinding) && multiSubgraphFinding.technical_priority === "Medium", "A multi-subgraph TFLite artifact must create the Medium EA-GRF-0001 runtime-aggregation finding.");
expectContains(multiSubgraphFinding?.observation || "", "independent per-op MAC", "The multi-subgraph finding must disclose all-scope deep static coverage.");
expectContains(multiSubgraphFinding?.observation || "", "not summed across control flow", "The multi-subgraph finding must preserve the runtime invocation-count boundary.");
expectContains(buildEngineeringReport(multiSubgraphAnalysis, { identity: reportBoundaryIdentity }), "Cross-subgraph execution aggregation requires runtime invocation counts", "Engineering Report action queue must surface the cross-subgraph aggregation boundary.");
expectBefore(engineeringReport, "## Static Audit Conclusion", "## Evidence Coverage", "Static conclusion should be before evidence details.");
expectBefore(engineeringReport, "## Decision Coverage At A Glance (DERIVED)", "## Static Structural Triage (HEURISTIC)", "Decision coverage should appear before heuristic triage.");
expectBefore(engineeringReport, "## Engineer Action Queue", "## Artifact And Target", "Action queue should be near the front.");
expectNotContains(engineeringReport, "Beta Runtime Basin", "Engineering Report should not expose the internal Runtime Basin module label.");
expectNotContains(engineeringReport, "main @ 2026-06-26", "Engineering Report should not cite an unpinned TensorFlow branch/date.");
expectNotContains(engineeringReport, "official XNNPACK delegate docs", "Engineering Report should use the pinned README provenance wording.");
expectNotContains(engineeringReport, "deployment-environment omissions create in regulatory summaries", "Engineering Report handoff note should not use the regulatory analogy.");
expectNotContains(engineeringReport, "Estimated peak activation arena", "Engineering Report should not imply the payload is the runtime arena.");
expectNotContains(engineeringReport, "Float weight tensors scanned", "Engineering Report should use float constant wording.");
expectNotContains(engineeringReport, "Float constant tensors scanned", "Engineering Report should use decodable constant wording.");
expectNotContains(engineeringReport, "buffer reuse", "Engineering Report should not claim ordinary buffer reuse can reduce peak live payload.");
expectNotContains(engineeringReport, "Dead output channels | 0 across 0", "Engineering Report should not show a misleading 0-across-0 channel result.");
expectNotContains(engineeringReport, "Max |weight|", "Engineering Report should not imply all scanned constants are weights.");
expectNotContains(engineeringReport, "Runtime Version Compatibility", "Engineering Report should not claim app-runtime compatibility from artifact-only evidence.");
expectNotContains(engineeringReport, "Metadata bytes", "Engineering Report should narrow metadata-byte wording.");
expectNotContains(engineeringReport, "Theoretical constants @INT8", "Engineering Report should label ideal INT8 payload floors explicitly.");
expectNotContains(engineeringReport, "output interpretation is embedded", "Engineering Report should distinguish mathematical output structure from semantic contracts.");
expectNotContains(engineeringReport, "up to ~3.24x", "Engineering Report should not market the estimated speedup as an upper-bound claim.");
expectNotContains(engineeringReport, "target_INT8_vs_FP32 4.5x", "Engineering Report speedup method should not contradict the target profile peak ratio.");
expectNotContains(engineeringReport, "non_compute_share", "Engineering Report should not model non-compute runtime share without runtime evidence.");
expectNotContains(engineeringReport, "fallback_penalty", "Engineering Report should not model fallback runtime penalty without runtime evidence.");
expectNotContains(engineeringReport, "Heuristic triage index |", "Engineering Report should not expose a context-free 0-100 readiness-looking score.");
expectNotContains(engineeringReport, "Artifact runtime floor (declared/rulepack max)", "Engineering Report should not imply the derived runtime floor is directly artifact-declared.");
expectNotContains(engineeringReport, "Report-bundle SHA-256", "Engineering Report should not mislabel the report-body verification hash as a bundle hash.");
expectNotContains(engineeringReport, "filename SHA-256 hash", "Engineering Report should not claim package attestation transmits a filename hash.");
expectNotContains(engineeringReport, "capability context", "Engineering Report should not claim the attestation embeds account capability context.");
expectEqual(
  decodeRoofReason("ROOF:MEM:0:android_mid_a55"),
  "Zero-MAC shape/structural operation / low-intensity posture (target: android_mid_a55)",
  "Zero-MAC roofline reason should use posture wording instead of the old memory-bound wording.",
);
expectEqual(
  decodeRoofReason("ROOF:F:3.20:4:8:a55:GATHER"),
  "FP32 intensity 3.20 ops/byte; thresholds: low <4, high ≥8; target: a55; depthwise layout-sensitive on x86 SIMD",
  "Roofline reason should use semicolon-separated text so appendix tables remain stable.",
);
for (const [text, snippet, label] of [
  [engineeringBundleSummary, "## Evidence Boundary", "Engineering Bundle summary should include evidence boundary."],
  [regulatoryBundleSummary, "## Evidence Boundary", "Regulatory Bundle summary should include evidence boundary."],
  [engineeringBundleSummary, ANALYZER_METADATA.schemas.engineeringBundle, "Engineering Bundle summary should include bundle schema."],
  [regulatoryBundleSummary, ANALYZER_METADATA.schemas.regulatoryBundle, "Regulatory Bundle summary should include bundle schema."],
  [regulatoryBundleSummary, RUNTIME_COMPATIBILITY_EVIDENCE_LABEL, "Regulatory Bundle summary should use report-facing runtime compatibility wording."],
]) {
  expectContains(text, snippet, label);
}

const mlBom = buildMlBomDocument(reportBoundaryAnalysis, {
  hash: "abc123",
  target: { id: "boundary", label: "Boundary target" },
});
const mlBomProperties = mlBom.metadata.component.properties;
assertCycloneDx17(mlBom, "engineering-bundle ML-BOM");
expectEqual(
  mlBom.metadata.tools.components[0].version,
  analyzerContentVersion(ANALYZER_METADATA.semanticVersion, ANALYZER_METADATA.buildCommit, ANALYZER_METADATA.buildContentSha256),
  "ML-BOM tool version should bind semantic and exact analyzer execution identity.",
);
for (const [name, value, label] of [
  ["deepbom:compatibility:profile", "deepbom.compact_mlbom_compatibility.v2", "compact compatibility profile"],
  ["deepbom:compatibility:detailLocation", "engineering_evidence.json#/evidence/static_analysis", "single detailed-evidence pointer"],
  ["mlbom:model:format", "tflite", "artifact format"],
  ["mlbom:model:operatorCount", String(reportBoundaryAnalysis.operator_count), "operator count"],
  ["mlbom:model:tensorCount", String(reportBoundaryAnalysis.tensor_count), "tensor count"],
  ["mlbom:model:quantizationClassification", null, "quantization classification"],
]) {
  expectProperty(mlBomProperties, name, value, `ML-BOM should include ${label}.`);
}
const compatibilityProjection = buildMlBomCompatibilityProjection(reportBoundaryAnalysis, { fileSizeBytes: reportBoundaryAnalysis.file_size_bytes, target: { id: "boundary", label: "Boundary target" } });
expect(compatibilityProjection.componentProperties.length <= 30 && compatibilityProjection.documentProperties.length <= 15, "ML-BOM compatibility projection should remain compact and bounded.");
expectProperty(mlBom.properties, "ondevice:predictedNonDelegatedOps", "{}", "TFLite ML-BOM should include the predicted non-delegated op inventory.");
expect(!mlBom.metadata.component.properties.some((item) => item.name === "deepbom:model:arenaCombinedBytes" || item.name === "deepbom:model:xnnpackSelectorEvidenceSchema"), "Detailed arena and selector ledgers should remain in structured companion evidence instead of the ML-BOM compatibility projection.");
expect(mlBom.metadata.component.externalReferences.some((item) => item.type === "evidence" && item.url === "engineering_evidence.json"), "Compact ML-BOM should link its detailed structured evidence companion.");
expect(!mlBom.properties.some((item) => item.name === "ondevice:delegateSuspects"), "ML-BOM should not use the ambiguous delegateSuspects property.");
expect(!mlBom.properties.some((item) => item.name === "ondevice:staticSuspectOpFamilies"), "ML-BOM should not use the ambiguous staticSuspectOpFamilies property.");
expectEqual(mlBom.metadata.authors[0]?.name, "DEEPBOM", "ML-BOM should use a non-personal package author.");
expectEqual(mlBom.metadata.authors[0]?.email, undefined, "ML-BOM should not embed a personal email address.");

const onnxMlBom = buildMlBomDocument({ ...reportBoundaryAnalysis, format: "onnx", runtime_review_watchlist: [{ name: "Flatten", count: 1, reason_code: "VIEW_OR_MATERIALIZATION_CANDIDATE" }] }, { hash: "abc123" });
expectProperty(onnxMlBom.properties, "ondevice:executionProviderAssignment", "not_assessable_without_runtime_evidence", "ONNX ML-BOM should state that execution-provider assignment is not assessable.");
expect(!onnxMlBom.properties.some((item) => item.name === "ondevice:predictedNonDelegatedOps" || item.name === "ondevice:delegateSuspects"), "ONNX ML-BOM should not invent delegate or EP fallback assignments.");
expect(!onnxMlBom.properties.some((item) => item.name === "ondevice:runtimeReviewWatchlistCounts"), "ONNX runtime watchlist detail should remain in structured companion evidence.");
expect(!onnxMlBom.metadata.component.properties.some((item) => item.name === "mlbom:target:ridgePointOpsPerByte"), "ONNX ML-BOM should suppress TFLite target-ridge assumptions.");

const partialMacOnnxMlBom = buildMlBomDocument({
  ...reportBoundaryAnalysis,
  format: "onnx",
  quantization_status: {
    ...reportBoundaryAnalysis.quantization_status,
    label: "Quantization signals; MAC coverage partial",
    quantized_compute_mac_percent: null,
    mac_coverage_complete: false,
  },
}, { hash: "abc123" });
const partialMacProperties = partialMacOnnxMlBom.metadata.component.properties;
expect(!partialMacProperties.some((item) => item.name === "mlbom:model:quantizedComputeMacRatio" || item.name === "mlbom:model:quantizedComputeMacPercent"), "ONNX ML-BOM should omit an unassessed quantized-MAC ratio rather than coerce it to zero.");
expectProperty(partialMacProperties, "mlbom:model:quantizedComputeMacAssessment", "not_assessed_mac_coverage_incomplete", "ONNX ML-BOM should state why the quantized-MAC ratio is unassessed.");

const staticAnalysisExport = buildStaticAnalysisExport(reportBoundaryAnalysis, {
  identity: reportBoundaryIdentity,
  hash: reportBoundaryIdentity.sha256,
});
expectEqual(staticAnalysisExport.schema, ANALYZER_METADATA.schemas.staticAnalysis, "Static analysis export should include shared schema.");
expectEqual(staticAnalysisExport.tensor_arena_plan?.combined_arena_bytes, 1003520, "Static analysis export should preserve the deterministic ArenaPlanner projection.");
expectAnalyzerMetadata(staticAnalysisExport.analyzer_metadata, "static analysis export");

const rawEvidenceFiles = buildRawEvidenceFiles(reportBoundaryAnalysis, { identity: reportBoundaryIdentity });
const rawJsonEvidence = new Map(rawEvidenceFiles
  .filter((file) => file.name.endsWith(".json"))
  .map((file) => [file.name, JSON.parse(file.data)]));
expectAnalyzerMetadata(rawJsonEvidence.get("raw-evidence/analyzer_metadata.json"), "raw analyzer metadata");
for (const [path, schema, label] of [
  ["raw-evidence/model_structure.json", ANALYZER_METADATA.schemas.modelStructure, "model structure evidence"],
  ["raw-evidence/quantization.json", ANALYZER_METADATA.schemas.quantizationEvidence, "quantization evidence"],
  ["raw-evidence/runtime_results.json", ANALYZER_METADATA.schemas.runtimeEvidence, "runtime evidence"],
  ["raw-evidence/weight_indicators.json", ANALYZER_METADATA.schemas.weightIndicatorEvidence, "weight indicator evidence"],
  ["raw-evidence/security_posture.json", ANALYZER_METADATA.schemas.securityPosture, "security posture evidence"],
  ["raw-evidence/findings_register.json", ANALYZER_METADATA.schemas.findingsRegister, "findings register evidence"],
]) {
  expectEvidenceSchema(rawJsonEvidence.get(path), schema, label);
}
expectEqual(rawJsonEvidence.get("raw-evidence/execution_placement.json")?.schema,
  ANALYZER_METADATA.schemas.executionPlacementEvidence,
  "Execution placement evidence should use its declared schema without duplicating package-level analyzer metadata.");
expectEqual(
  rawJsonEvidence.get("raw-evidence/security_posture.json")?.integrity?.schema_or_opset,
  "TFLite schema unknown",
  "Security posture should include artifact schema/opset identity.",
);
expectEqual(
  rawJsonEvidence.get("raw-evidence/security_posture.json")?.integrity?.external_data_summary,
  "external data not applicable",
  "Security posture should include external-data posture.",
);
const securityIntegrity = rawJsonEvidence.get("raw-evidence/security_posture.json")?.integrity || {};
for (const [field, value, label] of [
  ["tflite_constant_tensor_count", 3, "TFLite constant tensor buffer count"],
  ["tflite_out_of_bounds_tensor_buffers", 1, "TFLite tensor buffer bounds count"],
]) {
  expectEqual(securityIntegrity[field], value, `Security posture should include ${label}.`);
}

const engineeringManifest = buildEngineeringBundleManifest({
  model: reportBoundaryIdentity,
  files: rawEvidenceFiles,
});
const regulatoryManifest = buildEvidenceBundleManifest({
  analysis: reportBoundaryAnalysis,
  model: reportBoundaryIdentity,
  files: rawEvidenceFiles,
});
for (const [manifest, schema, label] of [
  [engineeringManifest, ANALYZER_METADATA.schemas.engineeringBundle, "Engineering bundle manifest"],
  [regulatoryManifest, ANALYZER_METADATA.schemas.regulatoryBundle, "Regulatory bundle manifest"],
]) {
  expectEqual(manifest.schema, schema, `${label} should use shared schema.`);
}
expectAnalyzerMetadata(engineeringManifest.analyzer_metadata, "engineering bundle manifest");
expectAnalyzerMetadata(regulatoryManifest.analyzer_metadata, "regulatory bundle manifest");
for (const [manifest, label] of [[engineeringManifest, "engineering"], [regulatoryManifest, "regulatory"]]) {
  expectEqual(manifest.export_mode, "account_redacted_external", `${label} manifest should declare account-redacted external export mode.`);
  expectEqual(manifest.attestation_member, "attestation.json", `${label} manifest should name the attestation member.`);
  expectEqual(manifest.attestation_member_excluded_from_package_hash, true, `${label} manifest should define attestation hash exclusion.`);
  expectEqual(manifest.attestation_schema, ANALYZER_METADATA.schemas.attestation, `${label} manifest should declare the attestation schema.`);
  expectEqual(manifest.privacy?.artifact_identity_embedded, true, `${label} manifest should disclose retained artifact identity.`);
  expectEqual(manifest.privacy?.public_share_suitable_without_identity_review, false, `${label} manifest should not describe account redaction as artifact anonymization.`);
  expect(!("user" in manifest), `${label} manifest should not embed account identity.`);
  expect(!("capabilities" in manifest), `${label} manifest should not embed account capability details.`);
  expect(Array.isArray(manifest.payload_files) && manifest.payload_files.length === rawEvidenceFiles.length, `${label} manifest should distinguish payload files.`);
  expect(manifest.unsigned_package_members.includes("manifest.json"), `${label} manifest should declare the unsigned package members.`);
  expectEqual(manifest.path_normalization, "relative_posix_nfc", `${label} manifest should declare safe path normalization.`);
  expectEqual(manifest.duplicate_member_policy, "reject_after_nfc_and_case_fold", `${label} manifest should reject portable filename collisions.`);
  expect(Boolean(manifest.manifest_self_privacy), `${label} manifest should scope its own privacy claims.`);
  expect(Boolean(manifest.included_report_privacy), `${label} manifest should scope included-report privacy claims.`);
  expect(Boolean(manifest.bundle_effective_privacy), `${label} manifest should publish effective bundle privacy.`);
  expectEqual(manifest.bundle_effective_privacy?.consent_event_id_embedded, false, `${label} bundle should not embed consent event identifiers.`);
  expectEqual(manifest.bundle_effective_privacy?.telemetry_fingerprint_embedded, false, `${label} bundle should not embed telemetry fingerprints.`);
  expectEqual(manifest.bundle_effective_privacy?.exact_browser_session_timestamp_embedded, false, `${label} bundle should not embed exact browser-session timestamps.`);
}

const publicManifest = buildEngineeringBundleManifest({
  analysis: reportBoundaryAnalysis,
  model: { artifact_id: "ARTIFACT-001", filename: "ARTIFACT-001", sha256: "redacted", target_id: "PUBLIC-TARGET" },
  files: rawEvidenceFiles,
  exportMode: "artifact_redacted_public",
});
expectEqual(publicManifest.export_mode, "artifact_redacted_public", "Public bundle manifest should declare artifact-redacted export mode.");
expectEqual(publicManifest.artifact_id, "ARTIFACT-001", "Public bundle should use a report-local artifact ID.");
expectEqual(publicManifest.privacy?.filename_embedded, false, "Public bundle should remove the source filename.");
expectEqual(publicManifest.privacy?.artifact_sha256_embedded, false, "Public bundle should remove the private artifact SHA-256.");
expectEqual(publicManifest.privacy?.internal_target_profile_identifier_embedded, false, "Public bundle should remove the internal target-profile ID.");
expectEqual(publicManifest.privacy?.public_share_suitable_without_identity_review, true, "Public bundle should explicitly declare public-share suitability.");

const publicAnalysis = buildPublicShareAnalysis({ ...reportBoundaryAnalysis, model_sha256: "a".repeat(64) });
const publicIdentity = buildPublicShareIdentity(publicAnalysis);
const publicContext = buildReportContextSet({ analysis: publicAnalysis, identity: publicIdentity, generatedAt: "2026-07-15T00:00:00.000Z" });
const publicMlBom = buildMlBomDocument(publicAnalysis, { hash: "", targetId: "PUBLIC-TARGET", timestamp: "2026-07-15T00:00:00.000Z" });
const publicFiles = buildEngineeringBundleArtifactFiles(publicAnalysis, {
  reportContext: publicContext.reportContext,
  rawEvidenceContext: publicContext.rawEvidenceContext,
  mlBomDocument: publicMlBom,
});
const publicPayloadText = publicFiles.map((file) => String(file.data || "")).join("\n");
expect(!publicPayloadText.includes("boundary_check.tflite"), "Public bundle payload should not contain the source filename.");
expect(!publicPayloadText.includes("a".repeat(64)), "Public bundle payload should not contain the source artifact SHA-256.");
expect(!publicPayloadText.includes('"id": "boundary"'), "Public bundle payload should not contain the internal target-profile ID.");
expect(publicPayloadText.includes("ARTIFACT-001") && publicPayloadText.includes("PUBLIC-TARGET"), "Public bundle should use report-local artifact and target identifiers.");

const publicFrontierAnalysis = buildPublicShareAnalysis({
  filename: "secret-model.tflite",
  model_sha256: "b".repeat(64),
  target_profile: { id: "private-current-target", label: "Private current target" },
  deployment_frontier: {
    artifact_sha256: "b".repeat(64),
    artifact_filename: "secret-model.tflite",
    targets: [
      { target_id: "private-a", target_label: "Private A", target_profile_sha256: "1".repeat(64) },
      { target_id: "private-b", target_label: "Private B", target_profile_sha256: "2".repeat(64) },
    ],
    robust_coverage: { per_target: [{ target_id: "private-a" }, { target_id: "private-b" }] },
    target_divergence: { pairs: [{ left_target_id: "private-a", right_target_id: "private-b" }] },
    interventions: [{ per_target: [{ target_id: "private-a" }, { target_id: "private-b" }] }],
    ops: [{ target_estimates: [{ target_id: "private-a" }, { target_id: "private-b" }] }],
  },
});
const publicFrontierText = JSON.stringify(publicFrontierAnalysis);
expect(!publicFrontierText.includes("secret-model.tflite") && !publicFrontierText.includes("b".repeat(64)), "Public frontier should remove the artifact filename and SHA-256.");
expect(!publicFrontierText.includes("private-a") && !publicFrontierText.includes("private-b"), "Public frontier should remove every internal planning-target identifier reference.");
expect(publicFrontierText.includes("PUBLIC-PLANNING-TARGET-01") && publicFrontierText.includes("PUBLIC-PLANNING-TARGET-02"), "Public frontier should retain stable report-local planning-target references.");

const privateBundleSha = "c".repeat(64);
const publicPackageAnalysis = buildPublicShareAnalysis({
  format: "safetensors",
  filename: "private/model.safetensors.index.json",
  model_sha256: privateBundleSha,
  artifact_bundle: {
    bundle_sha256: privateBundleSha,
    model_source_sha256: "d".repeat(64),
    files: [
      { path: "private/model-00001-of-00001.safetensors", byte_length: 16, sha256: "e".repeat(64), role: "tensor_shard", required: true },
    ],
  },
});
expectEqual(publicPackageAnalysis.artifact_bundle.bundle_sha256, "", "Public package projection should remove the canonical private bundle digest.");
expectEqual(publicPackageAnalysis.artifact_bundle.model_source_sha256, "", "Public package projection should remove the private model-source digest.");
expectEqual(publicPackageAnalysis.artifact_bundle.files[0].sha256, "", "Public package projection should remove member content digests.");
expectEqual(publicPackageAnalysis.artifact_bundle.public_redaction?.member_content_sha256_removed, true, "Public package projection should disclose member-digest redaction.");

const digestFiles = [
  { name: "engineering_report.md", data: "report\n" },
  { name: "engineering_evidence.json", data: "{\"ok\":true}\n" },
];
const digestForward = await buildCanonicalPackageDigest(digestFiles);
const digestReverse = await buildCanonicalPackageDigest([...digestFiles].reverse());
const digestChanged = await buildCanonicalPackageDigest([{ ...digestFiles[0], data: "changed\n" }, digestFiles[1]]);
expectEqual(digestForward.package_hash_sha256, digestReverse.package_hash_sha256, "Canonical package hash should not depend on ZIP member order.");
expect(digestForward.package_hash_sha256 !== digestChanged.package_hash_sha256, "Canonical package hash should change when a member changes.");
expectEqual(digestForward.files[0].name, "engineering_evidence.json", "Canonical package members should be sorted by name.");
expectEqual(digestForward.files.find((item) => item.name === "engineering_report.md")?.size, 7, "Package member size should be measured from exact UTF-8 bytes.");
expectEqual(digestForward.canonicalization, "RFC8785-JCS", "Canonical package hash should declare RFC8785-JCS.");

const contextSet = buildReportContextSet({
  analysis: reportBoundaryAnalysis,
  identity: reportBoundaryIdentity,
  user: { label: "Contract User" },
  capabilities: { export: true, regulatory_report: true },
  files: rawEvidenceFiles,
  moduleLog: [{ id: "deepbom", status: "complete" }],
  runtimeBenchmarkResults: [{ backend: "wasm", ok: true }],
  deepBomResult: { status: "complete" },
  perturbationResult: { status: "complete" },
  runtimeBasinResult: { status: "complete" },
  deployCurvatureResult: { status: "complete" },
  runtimeEvidence: {
    browserBucket: "Contract Browser",
    sharedArrayBufferAvailable: true,
    webgpuAvailable: true,
    webnnAvailable: false,
  },
  weightIndicatorEvidence: { deepBomResult: { status: "complete" } },
  fileSizeBytes: 1234,
  securityPosture: { schema: "security" },
  generatedAt: "2026-06-20T00:00:00.000Z",
});
for (const [actual, expected, label] of [
  [contextSet.reportContext.identity.filename, reportBoundaryIdentity.filename, "identity"],
  [contextSet.reportContext.generatedAt, "2026-06-20T00:00:00.000Z", "generatedAt"],
  [contextSet.regulatoryReportContext.fileSizeBytes, 1234, "regulatory file size"],
  [contextSet.bundleSummaryContext.files.length, rawEvidenceFiles.length, "bundle files"],
  [contextSet.bundleManifestContext.model.sha256, reportBoundaryIdentity.sha256, "manifest model identity"],
  [contextSet.rawEvidenceContext.runtimeEvidence.browserBucket, "Contract Browser", "runtime evidence"],
  [contextSet.reportContext.runtimeEvidence.browserBucket, "Contract Browser", "report runtime evidence"],
  [contextSet.rawEvidenceContext.securityPosture.schema, "security", "security posture"],
  [contextSet.findingsContext.deepBomResult.status, "complete", "findings inputs"],
]) {
  expectEqual(actual, expected, `Report context set should preserve ${label}.`);
}

const engineeringArtifactFiles = buildEngineeringBundleArtifactFiles(reportBoundaryAnalysis, {
  reportContext: { identity: reportBoundaryIdentity },
  rawEvidenceContext: { identity: reportBoundaryIdentity },
  mlBomDocument: mlBom,
  graphSvgText: "<svg></svg>",
  visualPngFiles: [{ name: "visuals/performance_overview.png", data: new Uint8Array([1, 2, 3]) }],
});
const engineeringArtifactNames = engineeringArtifactFiles.map((file) => file.name);
for (const requiredName of [
  "engineering_report.md",
  "engineering_evidence.json",
]) {
  expect(engineeringArtifactNames.includes(requiredName), `Engineering bundle artifacts should include ${requiredName}.`);
}
expectEqual(engineeringArtifactNames.length, 2, "Engineering bundle should keep its pre-envelope artifact set compact.");
const engineeringEvidence = JSON.parse(engineeringArtifactFiles.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
const bundledEngineeringReport = engineeringArtifactFiles.find((file) => file.name === "engineering_report.md")?.data || "";
expectEqual(engineeringEvidence.schema, ANALYZER_METADATA.schemas.engineeringEvidence, "Consolidated engineering evidence should use its declared schema.");
expect(bundledEngineeringReport.includes("## Execution Placement Evidence")
  && bundledEngineeringReport.includes(ANALYZER_METADATA.schemas.executionPlacementEvidence),
"Engineering Report should render the normalized execution-placement schema exported in engineering evidence.");
for (const requiredField of [
  "static_analysis",
  "model_structure",
  "quantization",
  "runtime_results",
  "execution_placement",
  "weight_indicators",
  "security_posture",
  "findings_register",
  "conformance_report",
  "change_analysis",
  "analyzer_build_content_manifest",
  "mlbom_cyclonedx",
]) {
  expect(requiredField in (engineeringEvidence.evidence || {}), `Consolidated engineering evidence should include ${requiredField}.`);
}
expectEqual(engineeringEvidence.evidence?.analyzer_build_content_manifest?.content_digest?.sha256, ANALYZER_METADATA.buildContentSha256, "Consolidated engineering evidence should include the exact file-level build-content manifest.");
const synthesizedFindings = engineeringEvidence.evidence?.findings_register?.findings || [];
expect(synthesizedFindings.length > 0 && synthesizedFindings.every((item) => item.origin === "report_synthesis" && item.source_rule_id && item.method_version && Array.isArray(item.evidence_json_pointers)), "Synthesized findings should carry machine-readable origin, rule, method, and evidence pointers.");
const nativeSignals = engineeringEvidence.evidence?.findings_register?.raw_analyzer_signals || [];
expect(nativeSignals.length === reportBoundaryAnalysis.findings.length && nativeSignals.every((item) => item.origin === "native_analyzer" && item.authoritative === false && !("severity" in item) && !("actions" in item)), "Native analyzer output should remain non-authoritative raw signals.");
expectEqual(engineeringEvidence.evidence?.conformance_report?.status, "pass", "Engineering evidence should pass its self-conformance invariants.");
const bundledMetricCoverageFailures = validateMetricCoverageManifest(engineeringEvidence.evidence?.metric_coverage_manifest, bundledEngineeringReport);
expectEqual(bundledMetricCoverageFailures.length, 0, `Bundled Engineering Report metric coverage should validate without missing report bindings, unbound evidence, or non-JSON-safe fields: ${bundledMetricCoverageFailures.join(" / ")}`);
const pointerValidation = engineeringEvidence.evidence?.conformance_report?.finding_evidence_pointer_validation;
expectEqual(pointerValidation?.finding_count, synthesizedFindings.length, "Finding pointer validation should cover every authoritative finding.");
expectEqual(pointerValidation?.resolved_pointer_count, pointerValidation?.pointer_count, "Every authoritative finding pointer should resolve against the exported evidence document.");
expectEqual(pointerValidation?.unresolved_pointer_count, 0, "Release evidence should contain no stale finding pointer.");
expectEqual(pointerValidation?.schema_error_count, 0, "Release findings should satisfy the machine-readable ID, priority, confidence, source, method, and pointer schema.");
const queueSection = bundledEngineeringReport.split("## Engineer Action Queue")[1]?.split("## Evidence Coverage")[0] || "";
const priorityRank = new Map([["High", 0], ["Medium", 1], ["Low", 2], ["Informational", 3]]);
const expectedFindingOrder = synthesizedFindings.map((item, index) => ({ item, index })).sort((left, right) =>
  (priorityRank.get(left.item.technical_priority) ?? 99) - (priorityRank.get(right.item.technical_priority) ?? 99)
  || left.index - right.index).map(({ item }) => item.finding_id);
const renderedFindingOrder = [...queueSection.matchAll(/^### (EA-[A-Z]+-[0-9]+) - /gm)].map((match) => match[1]);
expectEqual(JSON.stringify(renderedFindingOrder), JSON.stringify(expectedFindingOrder), "Engineering Report should preserve every finding exactly once in deterministic priority order.");
expect(queueSection.includes("EA-LIM-0001 - Representative-data validation not assessable"), "Engineering Report must expose the representative-data limitation instead of filtering it from the action queue.");
expect(synthesizedFindings.every((item) => queueSection.includes(item.source_rule_id)
  && item.evidence_json_pointers.every((pointer) => queueSection.includes(pointer))), "Every rendered finding should expose its source rule and complete evidence pointer list.");

const tamperedFindingsRegister = structuredClone(engineeringEvidence.evidence.findings_register);
tamperedFindingsRegister.findings[0].evidence_json_pointers = ["/evidence/static_analysis/does_not_exist"];
const tamperedEvidenceRoot = structuredClone(engineeringEvidence);
tamperedEvidenceRoot.evidence.findings_register = tamperedFindingsRegister;
const tamperedFindingConformance = buildConformanceReport({
  analysis: reportBoundaryAnalysis,
  staticAnalysis: engineeringEvidence.evidence.static_analysis,
  quantization: engineeringEvidence.evidence.quantization,
  findingsRegister: tamperedFindingsRegister,
  runtimeResults: engineeringEvidence.evidence.runtime_results,
  securityPosture: engineeringEvidence.evidence.security_posture,
  mlBomDocument: engineeringEvidence.evidence.mlbom_cyclonedx,
  engineeringReport: bundledEngineeringReport,
  metricCoverage: engineeringEvidence.evidence.metric_coverage_manifest,
  evidenceRoot: tamperedEvidenceRoot,
});
expectEqual(tamperedFindingConformance.status, "fail", "A stale authoritative finding pointer must fail bundle conformance.");
expect(tamperedFindingConformance.failures.some((item) => item.id === "CF-FINDING-005"), "Stale pointer tampering should fail the dedicated finding-evidence invariant.");
expectEqual(tamperedFindingConformance.release_export_allowed, false, "A stale finding pointer must block release export.");

const tamperedMetricCoverage = structuredClone(engineeringEvidence.evidence.metric_coverage_manifest);
const tamperedRuntimeAssignmentEntry = tamperedMetricCoverage.entries.find((entry) => entry.metric_id === "runtime.assignment");
tamperedRuntimeAssignmentEntry.status = "assessed";
tamperedRuntimeAssignmentEntry.evidence_class = "OBSERVED/DERIVED";
tamperedMetricCoverage.status_counts = Object.fromEntries(["assessed", "partial", "not_assessed", "not_applicable", "suppressed"].map((status) => [status, tamperedMetricCoverage.entries.filter((entry) => entry.status === status).length]));
tamperedMetricCoverage.decision_coverage = buildDecisionCoverageLedger(tamperedMetricCoverage.entries, tamperedMetricCoverage.format);
const tamperedMetricEvidenceRoot = structuredClone(engineeringEvidence);
tamperedMetricEvidenceRoot.evidence.metric_coverage_manifest = tamperedMetricCoverage;
const tamperedMetricConformance = buildConformanceReport({
  analysis: reportBoundaryAnalysis,
  staticAnalysis: engineeringEvidence.evidence.static_analysis,
  quantization: engineeringEvidence.evidence.quantization,
  findingsRegister: engineeringEvidence.evidence.findings_register,
  runtimeResults: engineeringEvidence.evidence.runtime_results,
  securityPosture: engineeringEvidence.evidence.security_posture,
  mlBomDocument: engineeringEvidence.evidence.mlbom_cyclonedx,
  engineeringReport: bundledEngineeringReport,
  metricCoverage: tamperedMetricCoverage,
  evidenceRoot: tamperedMetricEvidenceRoot,
});
expect(tamperedMetricConformance.failures.some((item) => item.id === "CF-COVERAGE-002"), "A self-consistent metric/decision status tamper must fail independent reconstruction from analysis and runtime evidence.");

const metadataVerifiedAnalysis = structuredClone(reportBoundaryAnalysis);
metadataVerifiedAnalysis.metadata_presence = {
  ...metadataVerifiedAnalysis.metadata_presence,
  schema: ANALYZER_METADATA.schemas.artifactMetadata,
  status: "parsed",
  has_model_metadata: true,
  metadata_entries: ["TFLITE_METADATA"],
  documented_preprocessing: true,
  preprocessing_contract_status: "assessed_explicit_input_process_units",
  output_semantics_documented: true,
  metadata_schema_identifier: "M001",
  metadata_min_parser_version: "1.5.0",
  metadata_model_name: "Contract fixture",
  model_metadata_entry_count: 1,
  subgraph_metadata_count: 1,
  input_tensor_metadata_count: 1,
  output_tensor_metadata_count: 1,
  described_input_tensor_count: 1,
  described_output_tensor_count: 1,
  input_process_unit_count: 1,
  recognized_input_process_unit_count: 1,
  invalid_input_process_unit_count: 0,
  unrecognized_input_process_unit_count: 0,
  normalization_unit_count: 1,
  input_process_units: [{ scope: "input_tensor", input_ordinal: 0, tensor_name: "input", options_type: "NormalizationOptions", options_type_code: 1, status: "assessed", mean: [127.5], std: [127.5], associated_files: [], detail: "fixture" }],
  output_associated_file_count: 1,
  output_label_file_count: 1,
  verified_output_associated_file_count: 1,
  missing_output_associated_file_count: 0,
  verified_output_label_file_count: 1,
  missing_output_label_file_count: 0,
  invalid_output_label_file_count: 0,
  verified_output0_label_file_count: 1,
  payload_verified_file_count: 1,
  payload_invalid_file_count: 0,
  payload_unsupported_file_count: 0,
  label_cardinality_match_count: 1,
  label_cardinality_mismatch_count: 0,
  label_cardinality_ambiguous_count: 0,
  label_cardinality_unresolved_count: 0,
  output_associated_files: [{
    output_ordinal: 0, tensor_name: "conv_out", name: "labels.txt", description: "fixture labels",
    file_type: "TENSOR_AXIS_LABELS", file_type_code: 2, locale: "", version: "", packed_status: "verified_payload",
    payload_status: "verified", payload_sha256: "c".repeat(64), payload_bytes: 32, crc32_verified: true,
    text_encoding_status: "valid_utf8", label_entry_count: 8, blank_label_entry_count: 0,
    output_shape: [1, 112, 112, 8], cardinality_status: "verified_unique_axis_match", matching_output_axes: [3],
    validation_detail: "Verified fixture label cardinality.",
  }],
  associated_file_archive_status: "assessed",
  associated_file_archive_detail: "Fixture terminal ZIP central directory parsed exactly.",
  packed_associated_file_count: 1,
  packed_associated_files: [{
    name: "labels.txt", compressed_bytes: 32, uncompressed_bytes: 32, compression_method: 0,
    crc32: 0x12345678, local_header_offset: 96, payload_status: "verified", payload_sha256: "c".repeat(64),
    decoded_bytes: 32, crc32_verified: true, detail: "Fixture payload verified.",
  }],
  detail: "Fixture M001 metadata and payload verified.",
};
const metadataVerifiedFiles = buildEngineeringBundleArtifactFiles(metadataVerifiedAnalysis, {
  reportContext: { identity: reportBoundaryIdentity },
  rawEvidenceContext: { identity: reportBoundaryIdentity },
  mlBomDocument: buildMlBomDocument(metadataVerifiedAnalysis, { hash: reportBoundaryIdentity.sha256, targetId: reportBoundaryIdentity.target_id }),
});
const metadataVerifiedReport = metadataVerifiedFiles.find((file) => file.name === "engineering_report.md")?.data || "";
const metadataVerifiedEvidence = JSON.parse(metadataVerifiedFiles.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expectEqual(metadataVerifiedEvidence.evidence?.conformance_report?.status, "pass", "Payload- and cardinality-verified metadata should pass evidence conformance.");
expectContains(metadataVerifiedReport, "8 entries (0 blank)", "Verified metadata report should render exact label cardinality.");
expectContains(metadataVerifiedReport, "verified_unique_axis_match", "Verified metadata report should render axis-cardinality status.");
expectContains(metadataVerifiedReport, "c".repeat(64), "Verified metadata report should render decoded payload SHA-256.");
expect(!metadataVerifiedEvidence.evidence?.findings_register?.findings?.some((item) => item.finding_id === "EA-OUT-0001"), "A payload- and cardinality-verified output label contract should suppress EA-OUT-0001.");
for (const requiredField of [
  "raw_static_audit_markdown",
  "roofline_csv",
  "core_isolation_roofline_csv",
  "memory_cache_csv",
  "arena_plan_csv",
  "runtime_assignment_comparison_csv",
  "runtime_boundary_comparison_csv",
  "stage_graph_mermaid",
  "graph_neighborhood_svg",
]) {
  expect(requiredField in (engineeringEvidence.supplemental_sources || {}), `Consolidated engineering evidence should include ${requiredField}.`);
}

const rawDataArtifactFiles = buildRawDataArtifactFiles(reportBoundaryAnalysis, {
  rawEvidenceContext: { identity: reportBoundaryIdentity },
  mlBomDocument: mlBom,
  graphSvgText: "<svg></svg>",
  visualPngFiles: [{ name: "visuals/performance_overview.png", data: new Uint8Array([1, 2, 3]) }],
});
const rawDataArtifactNames = rawDataArtifactFiles.map((file) => file.name);
for (const requiredName of [
  "static/raw_static_audit.md",
  "static/roofline.csv",
  "static/core_isolation_roofline.csv",
  "static/stage_graph.mmd",
  "static/static_analysis.json",
  "static/arena_plan.csv",
  "static/mlbom_cdx.json",
  "raw-evidence/model_structure.json",
  "raw-evidence/quantization.json",
  "raw-evidence/memory_cache.csv",
  "raw-evidence/arena_plan.csv",
  "raw-evidence/runtime_results.json",
  "raw-evidence/execution_placement.json",
  "raw-evidence/weight_indicators.json",
  "raw-evidence/security_posture.json",
  "raw-evidence/analyzer_metadata.json",
  "raw-evidence/findings_register.json",
  "static/graph_neighborhood.svg",
  "visuals/performance_overview.png",
]) {
  expect(rawDataArtifactNames.includes(requiredName), `Detailed Raw Data ZIP artifacts should preserve ${requiredName}.`);
}
const runtimeRawDataNames = buildRawDataArtifactFiles(runtimeBoundAnalysis, {
  rawEvidenceContext: runtimeEvidenceContext,
  mlBomDocument: runtimeMlBom,
}).map((file) => file.name);
expect(runtimeRawDataNames.includes("runtime/assignment_comparison.csv"), "Raw Data ZIP should include the imported runtime assignment comparison CSV.");
expect(runtimeRawDataNames.includes("runtime/boundary_comparison.csv"), "Raw Data ZIP should include the runtime boundary comparison CSV.");

done(`Export artifact contract passed (${entries.length} artifacts).`);

function expectContains(text, snippet, label) {
  expect(String(text || "").includes(snippet), `${label} Missing ${JSON.stringify(snippet)}.`);
}

function expectNotContains(text, snippet, label) {
  expect(!String(text || "").includes(snippet), `${label} Unexpected ${JSON.stringify(snippet)}.`);
}

function expectBefore(text, first, second, label) {
  const haystack = String(text || "");
  const a = haystack.indexOf(first);
  const b = haystack.indexOf(second);
  expect(a >= 0 && b >= 0 && a < b, `${label} Expected ${JSON.stringify(first)} before ${JSON.stringify(second)}.`);
}

function expectProperty(properties, name, value, label) {
  expect(
    properties.some((item) => item.name === name && (value == null || item.value === value)),
    label,
  );
}

function expectEvidenceSchema(value, schema, label) {
  if (!value || typeof value !== "object") {
    expect(false, `${label} should be a JSON object.`);
    return;
  }
  expectEqual(value.schema, schema, `${label} schema mismatch.`);
  expectAnalyzerMetadata(value.analyzer_metadata, `${label} analyzer metadata`);
}

function expectAnalyzerMetadata(value, label) {
  if (!value || typeof value !== "object") {
    expect(false, `${label} should include analyzer metadata.`);
    return;
  }
  expectEqual(value.analyzer, ANALYZER_METADATA.name, `${label} analyzer mismatch.`);
  expectEqual(value.analyzer_version, ANALYZER_METADATA.version, `${label} analyzer version mismatch.`);
  expectEqual(value.analyzer_semantic_version, ANALYZER_METADATA.semanticVersion, `${label} analyzer semantic version mismatch.`);
  expectEqual(value.analyzer_build_commit, ANALYZER_METADATA.buildCommit, `${label} analyzer build commit mismatch.`);
  expectEqual(value.analyzer_build_source_state, ANALYZER_METADATA.buildSourceState, `${label} analyzer source state mismatch.`);
  expectEqual(value.analyzer_build_content_sha256, ANALYZER_METADATA.buildContentSha256, `${label} analyzer bundle hash mismatch.`);
  expectEqual(value.analyzer_build_content_hash_method, ANALYZER_METADATA.buildContentHashMethod, `${label} analyzer bundle hash method mismatch.`);
  expectEqual(value.analyzer_build_content_manifest_sha256, ANALYZER_METADATA.buildContentManifestSha256, `${label} analyzer build manifest hash mismatch.`);
  expect(Boolean(value.target_source_basis), `${label} target source basis must not be blank.`);
  expect(Boolean(value.target_profile_provenance?.format_applicability), `${label} target profile applicability must be machine-readable.`);
  expectEqual(value.rulepack_version, ANALYZER_METADATA.rulepackVersion, `${label} rulepack mismatch.`);
  expectEqual(value.rulepack_sha256, ANALYZER_METADATA.rulepackSha256, `${label} rulepack SHA mismatch.`);
  expectEqual(value.schemas?.engineeringReport, ANALYZER_METADATA.schemas.engineeringReport, `${label} report schema mismatch.`);
}

function expectNonEmpty(value, label) {
  expect(typeof value === "string" && Boolean(value.trim()), `${label} must be a non-empty string.`);
}

function expectHtmlId(source, id, label, htmlPath) {
  expect(source.includes(`id="${id}"`), `${htmlPath} is missing ${label} id="${id}".`);
}

function expectNotHtmlId(source, id, label, htmlPath) {
  expect(!source.includes(`id="${id}"`), `${htmlPath} ${label}: unexpected id="${id}".`);
}

function expectAppSnippet(snippet, label) {
  expect(appSources.includes(snippet), label);
}

function expectExactSet(label, actual, expected) {
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  for (const value of expectedSet) {
    if (!actualSet.has(value)) {
      expect(false, `${label} is missing ${value}.`);
    }
  }
  for (const value of actualSet) {
    if (!expectedSet.has(value)) {
      expect(false, `${label} has unexpected value ${value}.`);
    }
  }
}

function expectUnique(label, values) {
  const seen = new Set();
  for (const value of values) {
    if (seen.has(value)) {
      expect(false, `${label} contains duplicate ${value}.`);
    }
    seen.add(value);
  }
}
