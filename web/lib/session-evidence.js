import { buildReportContextSet } from "./report-context.js";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[character]);
}

export function artifactUuidFromSha256(hash) {
  const hex = String(hash || "")
    .toLowerCase()
    .replace(/[^a-f0-9]/g, "")
    .padEnd(32, "0")
    .slice(0, 32)
    .split("");
  hex[12] = "8";
  hex[16] = ((Number.parseInt(hex[16], 16) & 0x3) | 0x8).toString(16);
  const value = hex.join("");
  return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20)}`;
}

export function buildAccountEngineeringReportHtml(markdown, {
  generatedAt,
  owner = "account",
  modelName = "model",
  origin = "",
} = {}) {
  if (!generatedAt) throw new Error("Engineering Report HTML requires a frozen generatedAt timestamp.");
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DEEPBOM Engineering Report</title>
    <style>
      :root { color: #111827; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { margin: 0; background: #f8fafc; }
      main { position: relative; max-width: 920px; margin: 0 auto; padding: 48px 40px 72px; background: #fff; min-height: 100vh; box-shadow: 0 20px 80px rgba(15, 23, 42, 0.08); }
      main::before { content: "deepbom.org"; position: fixed; inset: 40% auto auto 50%; transform: translate(-50%, -50%) rotate(-28deg); font-size: 84px; font-weight: 800; color: rgba(20, 83, 45, 0.07); letter-spacing: 0.08em; pointer-events: none; white-space: nowrap; }
      header { border-bottom: 1px solid #dbe3ef; margin-bottom: 28px; padding-bottom: 20px; }
      .eyebrow { color: #14532d; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 8px; }
      h1 { margin: 0; font-size: 30px; }
      .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 18px; margin-top: 16px; color: #475569; font-size: 12px; }
      pre { position: relative; z-index: 1; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font: 12px/1.62 "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
      .print-note { color: #64748b; font-size: 12px; margin-top: 10px; }
      @media (max-width: 640px) {
        main { padding: 28px 20px 48px; }
        .meta { grid-template-columns: minmax(0, 1fr); }
        main::before { font-size: 48px; }
      }
      @media print {
        body { background: #fff; }
        main { max-width: none; box-shadow: none; padding: 20mm 16mm; }
        main::before { color: rgba(20, 83, 45, 0.09); }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <p class="eyebrow">DEEPBOM Engineering Report</p>
        <h1>Engineering Report</h1>
        <div class="meta">
          <span>Model: ${escapeHtml(modelName)}</span>
          <span>Generated: ${escapeHtml(generatedAt)}</span>
          <span>Issued to: ${escapeHtml(owner)}</span>
          <span>Origin: ${escapeHtml(origin)}</span>
        </div>
        <p class="print-note">Use the browser print dialog to save this report as PDF. Raw JSON, CSV, ML-BOM, SVG, PNG, and bundle exports require a verified, authorized account.</p>
      </header>
      <pre>${escapeHtml(markdown)}</pre>
    </main>
  </body>
</html>`;
}

function publicScalar(value, fallback = "Not assessed") {
  if (value == null || value === "") return fallback;
  return String(value);
}

function publicInteger(value, fallback = "Not assessed") {
  if (value == null || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat("en-US").format(number) : fallback;
}

function publicInterfaceRows(items, role) {
  if (!Array.isArray(items) || !items.length) return [`- ${role}: none serialized`];
  return items.map((item, index) => {
    const shape = Array.isArray(item?.shape) && item.shape.length ? `[${item.shape.join(", ")}]` : "shape not declared";
    const scale = Array.isArray(item?.scale_sample) && item.scale_sample.length === 1
      ? `; scale ${item.scale_sample[0]}; zero point ${item.zero_point_sample?.[0] ?? "not declared"}`
      : "";
    return `- ${role} ${index + 1}: ${publicScalar(item?.name, "unnamed")} | ${publicScalar(item?.dtype)} | ${shape}${scale}`;
  });
}

export function buildPublicAuditSummaryText(analysis, {
  scope,
  generatedAt,
} = {}) {
  if (!analysis || typeof analysis !== "object") throw new Error("Public report summary requires completed analysis evidence.");
  if (!scope?.label || !scope?.assessed || !scope?.runtimeBoundary) {
    throw new Error("Public report summary requires a format-specific evidence scope.");
  }
  if (!generatedAt) throw new Error("Public report summary requires a frozen generatedAt timestamp.");

  const macAssessment = analysis.mac_assessment || {};
  const quantization = analysis.quantization_status || {};
  const findings = Array.isArray(analysis.findings) ? analysis.findings : [];
  const format = String(analysis.format || scope.id || "unknown").toUpperCase();
  const target = analysis.target_profile?.name || analysis.target_profile?.id;
  const macLine = String(macAssessment.status || "").startsWith("not_applicable")
    ? `Not applicable from this artifact representation (${macAssessment.status})`
    : Number.isFinite(Number(analysis.total_macs))
      ? `${publicInteger(analysis.total_macs)} total assessed MACs; ${publicInteger(macAssessment.assessed_compute_ops)} of ${publicInteger(macAssessment.compute_ops)} compute ops assessed`
      : publicScalar(macAssessment.detail);
  const findingRows = findings.length
    ? findings.flatMap((finding) => {
      const evidence = (finding.evidence || []).map((item) => `  Evidence - ${publicScalar(item?.source, "Source")}: ${publicScalar(item?.text, "Not stated")}`);
      const actions = (finding.actions || []).map((action) => `  Action - ${action}`);
      return [
        `- [${String(finding.severity || "informational").toUpperCase()}] ${publicScalar(finding.id, "finding")} - ${publicScalar(finding.title, "Static finding")}`,
        ...evidence,
        finding.impact ? `  Possible effect - ${finding.impact}` : null,
        ...actions,
      ].filter(Boolean);
    })
    : ["- No native analyzer finding was emitted for this run. This is not a task-accuracy or release-readiness conclusion."];

  return [
    "DEEPBOM PUBLIC STATIC EVIDENCE SUMMARY",
    "",
    "CLAIM BOUNDARY",
    `Run status: ${scope.completion}`,
    `Evidence class: ${scope.evidenceClass}`,
    `Analysis depth: ${scope.depth}`,
    `Assessed now: ${scope.assessed}`,
    `Runtime evidence: ${scope.runtimeStatus}. ${scope.runtimeBoundary}`,
    `Release readiness: ${scope.releaseStatus}`,
    "",
    "ARTIFACT IDENTITY",
    `Name: ${publicScalar(analysis.filename)}`,
    `Format: ${format}`,
    `Artifact SHA-256: ${publicScalar(analysis.model_sha256, "Not embedded")}`,
    `Serialized bytes: ${publicInteger(analysis.file_size_bytes ?? analysis.file_size)}`,
    `Generated: ${generatedAt}`,
    target ? `Reference target/profile: ${target}` : "Reference target/profile: not applicable to this artifact-only run",
    "",
    "STRUCTURAL INVENTORY",
    `Operators: ${["gguf", "safetensors"].includes(String(analysis.format).toLowerCase()) ? "Not applicable from a tensor container" : publicInteger(analysis.operator_count ?? analysis.total_ops)}`,
    `Tensors: ${publicInteger(analysis.tensor_count ?? analysis.tensors?.length)}`,
    `MAC assessment: ${macLine}`,
    "",
    "INTERFACE CONTRACT",
    ...publicInterfaceRows(analysis.inputs, "Input"),
    ...publicInterfaceRows(analysis.outputs, "Output"),
    "",
    "QUANTIZATION OR STORAGE EVIDENCE",
    `Classification: ${publicScalar(quantization.label || quantization.classification)}`,
    `Summary: ${publicScalar(quantization.summary)}`,
    `Detail: ${publicScalar(quantization.detail)}`,
    "",
    `NATIVE ANALYZER FINDINGS (${findings.length})`,
    ...findingRows,
    "",
    "INTERPRETATION LIMITS",
    `- ${scope.runtimeBoundary}`,
    `- ${scope.releaseStatus}`,
    "- Static artifact evidence does not establish task accuracy, clinical performance, observed native placement, or production release approval.",
    "- The report-body SHA-256 on the print copy detects body changes but is not a digital signature.",
  ].join("\n");
}

export function buildPublicEngineeringReportHtml(reportText, {
  generatedAt,
  modelName = "model",
  origin = "",
  reportFingerprint,
  citationDoi = "10.5281/zenodo.21834509",
  profile = "public",
} = {}) {
  if (!generatedAt) throw new Error("Public report HTML requires a frozen generatedAt timestamp.");
  if (!/^[a-f0-9]{64}$/i.test(String(reportFingerprint || ""))) {
    throw new Error("Public report HTML requires a SHA-256 report fingerprint.");
  }
  const presentation = reportPresentation(profile);
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(presentation.documentTitle)}</title>
    <style>
      :root { color: #111827; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      * { box-sizing: border-box; }
      body { margin: 0; background: #eef2f7; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      main { position: relative; max-width: 920px; margin: 0 auto; padding: 44px 38px 72px; background: #fff; min-height: 100vh; box-shadow: 0 20px 80px rgba(15, 23, 42, 0.08); }
      main::before { content: "${presentation.watermark}"; position: fixed; inset: 42% auto auto 50%; transform: translate(-50%, -50%) rotate(-28deg); z-index: 0; font-size: 72px; font-weight: 800; color: rgba(20, 83, 45, 0.075); letter-spacing: 0; pointer-events: none; white-space: nowrap; }
      header, pre { position: relative; z-index: 1; }
      header { border-bottom: 1px solid #dbe3ef; margin-bottom: 26px; padding-bottom: 18px; }
      .eyebrow { color: #14532d; font-size: 12px; font-weight: 800; text-transform: uppercase; margin: 0 0 8px; }
      h1 { margin: 0; font-size: 28px; }
      .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 18px; margin-top: 16px; color: #475569; font-size: 11px; }
      .meta span { overflow-wrap: anywhere; }
      .fingerprint { grid-column: 1 / -1; font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
      .boundary { margin: 14px 0 0; padding: 10px 12px; border-left: 3px solid #14532d; background: #f0fdf4; color: #334155; font-size: 11px; line-height: 1.5; }
      pre { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font: 11px/1.6 "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
      @media (max-width: 640px) {
        main { padding: 26px 18px 44px; }
        .meta { grid-template-columns: minmax(0, 1fr); }
        .fingerprint { grid-column: auto; }
        main::before { font-size: 40px; }
      }
      @page { margin: 12mm; }
      @media print {
        body { background: #fff; }
        main { max-width: none; box-shadow: none; padding: 8mm 5mm; }
        main::before { color: rgba(20, 83, 45, 0.09); }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <p class="eyebrow">${escapeHtml(presentation.eyebrow)}</p>
        <h1>${escapeHtml(presentation.heading)}</h1>
        <div class="meta">
          <span>Model: ${escapeHtml(modelName)}</span>
          <span>Generated: ${escapeHtml(generatedAt)}</span>
          <span>Origin: ${escapeHtml(origin)}</span>
          <span>DOI: https://doi.org/${escapeHtml(citationDoi)}</span>
          <span class="fingerprint">Report-body SHA-256: ${escapeHtml(reportFingerprint)}</span>
        </div>
        <p class="boundary">${escapeHtml(presentation.boundary)}</p>
      </header>
      <pre>${escapeHtml(reportText)}</pre>
    </main>
    <script>addEventListener("load", () => setTimeout(() => print(), 250), { once: true });</script>
  </body>
</html>`;
}

function reportPresentation(profile) {
  if (profile === "engineering") return {
    documentTitle: "DEEPBOM Engineering Review Report",
    watermark: "DEEPBOM ENGINEERING COPY",
    eyebrow: "DEEPBOM Engineering Review Copy",
    heading: "Engineering Report",
    boundary: "This login-free presentation copy contains the synthesized Engineering Report, but not original model bytes, raw tensor values, or controlled derivative files. Its body SHA-256 makes later body changes detectable. When this report is delivered inside an Evidence Package, that package separately carries a detached member-ledger signature. These integrity mechanisms do not prevent copying, prove official authorship without an independently trusted key, establish runtime measurement, or grant redistribution rights.",
  };
  if (profile === "regulatory") return {
    documentTitle: "DEEPBOM Regulatory Support Report",
    watermark: "DEEPBOM REGULATORY SUPPORT",
    eyebrow: "DEEPBOM Regulatory Support Copy",
    heading: "Model Artifact Regulatory Support Report",
    boundary: "This login-free support copy preserves regulatory-facing evidence classification and the Engineering Report appendix. It is not a regulatory submission, approval, clinical-validation result, or release authorization. Its body SHA-256 makes later body changes detectable. An Evidence Package separately carries a detached member-ledger signature. These integrity mechanisms do not prevent copying or prove official authorship without an independently trusted key.",
  };
  return {
    documentTitle: "DEEPBOM Public Static Evidence Summary",
    watermark: "DEEPBOM PUBLIC COPY",
    eyebrow: "DEEPBOM Public Sharing Copy",
    heading: "Static Evidence Summary",
    boundary: "This login-free presentation copy contains the public static-evidence summary, not the complete Engineering Report or raw evidence ledger. The SHA-256 identifies the exact body and makes later body edits detectable, but it does not prevent copying or editing and is not a digital signature, proof of authorship, runtime measurement, or release approval. The detached verification manifest binds this body hash to artifact, analyzer, rulepack, target, and any imported runtime identity.",
  };
}

export async function openPublicAuditPrintView({
  analysis,
  context,
  scope,
  origin,
  sha256Hex,
  isCurrent = () => true,
  openWindow = () => window.open("about:blank", "_blank"),
} = {}) {
  const printWindow = openWindow();
  if (!printWindow) throw new Error("Allow popups to open the watermarked print view.");
  printWindow.opener = null;
  printWindow.document.title = "Preparing DEEPBOM public report";
  printWindow.document.body.textContent = "Preparing fingerprinted report...";
  try {
    const body = buildPublicAuditSummaryText(analysis, { scope, generatedAt: context?.generatedAt });
    const reportFingerprint = await sha256Hex(new TextEncoder().encode(body));
    if (!isCurrent()) throw new Error("Report binding changed while the public copy was being generated.");
    const html = buildPublicEngineeringReportHtml(body, {
      generatedAt: context.generatedAt,
      modelName: analysis.filename || "model",
      origin,
      reportFingerprint,
    });
    printWindow.document.open();
    printWindow.document.write(html);
    printWindow.document.close();
    return printWindow;
  } catch (error) {
    printWindow.close();
    throw error;
  }
}

export function bindPublicAuditPrintButton(button, {
  getAnalysis,
  getBinding,
  bindingMatches,
  getContext,
  getScope,
  sha256Hex,
  origin,
  setStatus,
  formatError,
} = {}) {
  button?.addEventListener("click", async () => {
    const binding = getBinding();
    const analysis = getAnalysis();
    if (!analysis || !binding.canCopy || !bindingMatches(binding, analysis)) {
      setStatus("Analyze or load the report binding before printing", "error");
      return;
    }
    try {
      await openPublicAuditPrintView({
        analysis,
        context: getContext(),
        scope: getScope(analysis),
        origin,
        sha256Hex,
        isCurrent: () => getAnalysis() === analysis && bindingMatches(binding, getAnalysis()),
      });
      setStatus("Watermarked public report opened for PDF printing", "ok");
    } catch (error) {
      console.error("[printPublicReport]", error);
      setStatus(`Public report failed: ${formatError(error)}`, "error");
    }
  });
}

export function syncPublicPrintButton(button, { hasAnalysis, reportTargetReady } = {}) {
  if (!button) return;
  button.disabled = !hasAnalysis || !reportTargetReady;
  button.classList.remove("account-locked");
  button.title = !hasAnalysis
    ? "Run a static audit before opening the public print view."
    : !reportTargetReady
      ? "Analyze or load the selected report binding before printing."
      : "Open a login-free watermarked presentation copy with a report-body SHA-256, then use the browser print dialog to save PDF.";
}

export function buildSessionPrivacy({
  consentLog = [],
  agreementRecord = {},
  researchConsentRecord = {},
  researchConsent = false,
  structureTelemetryState = null,
  policyVersion = "",
  historySaved = false,
} = {}) {
  const latestConsentEvent = [...consentLog]
    .reverse()
    .find((entry) => entry.kind === "consent-restored" || entry.kind === "consent-withdrawn");
  const latestStructureShare = [...consentLog]
    .reverse()
    .find((entry) => entry.kind === "structure-shared");
  const currentStructureShare = structureTelemetryState?.fingerprint
    ? [...consentLog]
      .reverse()
      .find((entry) => entry.kind === "structure-shared" && entry.fingerprint === structureTelemetryState.fingerprint)
    : null;
  return {
    consent: Boolean(researchConsent),
    consentTimestamp: latestConsentEvent?.at || researchConsentRecord?.updated_at || agreementRecord?.accepted_at || null,
    consentEventId: latestConsentEvent?.event_id || null,
    policyVersion,
    retentionPeriod: "Local consent records remain in this browser until local storage is cleared; server-side research telemetry retention is governed by the host service policy.",
    telemetryEndpoint: "/api/benchmark/structure",
    telemetryPayloadSchema: "deepbom.structure_telemetry.v1.1",
    telemetryPayloadPreview: currentStructureShare
      ? "payload was sent; preview was not retained. Payload class: structure fingerprint and op/stage summary only; raw model bytes, weights, filenames, tensor values, inputs, outputs, and reports are excluded"
      : "none sent in this report session",
    telemetryCurrentShareAt: currentStructureShare?.at || null,
    telemetryLastShareAt: latestStructureShare?.at || null,
    telemetryFingerprint: structureTelemetryState?.fingerprint || null,
    historySaved: Boolean(historySaved),
  };
}

export function buildSessionReportContextSet({
  analysis = null,
  identity = {},
  user = {},
  capabilities = {},
  files = [],
  moduleLog = [],
  runtimeBenchmarkResults = [],
  deepBomResult = null,
  perturbationResult = null,
  runtimeBasinResult = null,
  deployCurvatureResult = null,
  preprocessingConsequenceResult = null,
  calibrationValidationResult = null,
  runtimeAssignmentEvidence = null,
  browserRuntime = {},
  fileSizeBytes = 0,
  generatedAt,
} = {}) {
  const runtimeEvidence = {
    browserBucket: browserRuntime.browserBucket || "",
    benchmarkResults: runtimeBenchmarkResults,
    runtimeBasinResult,
    preprocessingConsequenceResult,
    calibrationValidationResult,
    runtimeAssignmentEvidence,
    sharedArrayBufferAvailable: Boolean(browserRuntime.sharedArrayBufferAvailable),
    webgpuAvailable: Boolean(browserRuntime.webgpuAvailable),
    webnnAvailable: Boolean(browserRuntime.webnnAvailable),
  };
  return buildReportContextSet({
    analysis,
    identity,
    user,
    capabilities,
    files,
    moduleLog,
    runtimeBenchmarkResults,
    deepBomResult,
    perturbationResult,
    runtimeBasinResult,
    deployCurvatureResult,
    runtimeEvidence,
    weightIndicatorEvidence: {
      deepBomResult,
      perturbationResult,
      deployCurvatureResult,
    },
    fileSizeBytes,
    generatedAt,
  });
}
