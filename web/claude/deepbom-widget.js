import { McpAppClient } from "../lib/mcp-app-client.js";
import { analyzeOnnxModel } from "../onnx.js";
import { analyzeExecuTorchModel } from "../executorch.js";
import { readCoreMlModelFile } from "../lib/coreml-metadata-adapter.js";
import { readMetadataModelFile } from "../lib/metadata-model-adapters.js";
import { detectModelFormat } from "../lib/model-file.js";
import { normalizeAnalysisSummaryContract } from "../lib/analysis-summary-contract.js";
import { attachOnnxContractConflictCapsule } from "../lib/onnx-contract-conflict.js";
import { buildOnDeviceLlmContract } from "../lib/on-device-llm-contract.js";
import { getArtifactIrContext } from "../lib/artifact-ir-context.js";
import { buildArtifactEvidenceEnvelope, validateArtifactEvidenceEnvelope } from "../lib/artifact-evidence-envelope.js";
import { buildReviewSummary } from "../lib/review-summary.js";
import { createStaticAuditWorkerClient } from "../lib/static-audit-worker-client.js";
import { STATIC_AUDIT_OPERATION } from "../lib/static-audit-worker-protocol.js";
import { sha256FileHex } from "../lib/hash.js";
import { ANALYZER_SEMANTIC_VERSION } from "../lib/app-config.js";
import { compactModelSummaryForConversation, renderModelSummaryTable } from "../lib/model-summary.js";

const FULL_FILE_LIMIT = 128 * 1024 * 1024;
const RANGE_CHUNK_BYTES = 4 * 1024 * 1024;
const MAX_FINDINGS_PER_KIND = 8;
const SUPPORTED_FORMATS = new Set(["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"]);
const root = document.getElementById("deepbom-claude-root");
const app = new McpAppClient(
  { name: "DEEPBOM Artifact Evidence", version: ANALYZER_SEMANTIC_VERSION },
  { autoResize: true },
);
let analysisDepth = "structure";
let analyzing = false;

// Reuse the self-contained worker bundle served to sandboxed MCP Apps.
// location.origin belongs to the host sandbox, not to the DEEPBOM deployment.
const workerModuleUrl = new URL("../chatgpt/static-audit-worker.js", import.meta.url);
workerModuleUrl.search = new URL(import.meta.url).search;
let sandboxWorkerSource = null;
const staticAuditWorkerClient = createStaticAuditWorkerClient({
  createWorker: () => {
    if (sandboxWorkerSource === null) throw new Error("DEEPBOM analyzer is not loaded.");
    const url = URL.createObjectURL(new Blob([
      `globalThis.__deepbomWorkerModuleUrl = ${JSON.stringify(workerModuleUrl.href)};\n`,
      sandboxWorkerSource,
    ], { type: "text/javascript" }));
    const release = () => URL.revokeObjectURL(url);
    try {
      const worker = new Worker(url, { name: "deepbom-claude-static-audit" });
      worker.addEventListener("message", release, { once: true });
      worker.addEventListener("error", release, { once: true });
      window.addEventListener("pagehide", release, { once: true });
      return worker;
    } catch (error) {
      release();
      throw error;
    }
  },
});

async function prepareSandboxWorker() {
  if (sandboxWorkerSource !== null) return;
  setStatus("Loading isolated TFLite analyzer", "Downloading the DEEPBOM analyzer into this browser sandbox.");
  const response = await fetch(workerModuleUrl, { credentials: "omit", signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`DEEPBOM analyzer download failed (HTTP ${response.status}).`);
  if (!/javascript/i.test(response.headers.get("content-type") || "")) {
    throw new Error("DEEPBOM analyzer download returned an unexpected content type.");
  }
  sandboxWorkerSource = await response.text();
}

app.ontoolinput = ({ arguments: args }) => {
  analysisDepth = args?.analysis_depth === "payload_integrity" ? "payload_integrity" : "structure";
  updateDepthText();
};

renderShell();
void connectApp();

async function connectApp() {
  try {
    await app.connect();
    setStatus("Ready for a local file", "Choose one supported deployment artifact. Selection requires an explicit user action.");
    root.querySelector("#file").disabled = false;
    root.querySelector("#analyze").disabled = false;
  } catch (error) {
    renderError(error, "mcp_app_unavailable");
  }
}

root.querySelector("#analyze").addEventListener("click", () => {
  if (analyzing) return;
  const file = root.querySelector("#file").files?.[0];
  if (!file) {
    renderError(new Error("Choose one supported model artifact first."), "file_selection_missing");
    return;
  }
  void analyzeSelectedFile(file);
});

async function analyzeSelectedFile(file) {
  analyzing = true;
  root.querySelector("#analyze").disabled = true;
  root.querySelector("#result").replaceChildren();
  try {
    const safeName = String(file.name || "model").split(/[\\/]/).pop().slice(0, 512) || "model";
    const prefix = new Uint8Array(await file.slice(0, Math.min(file.size, 64 * 1024)).arrayBuffer());
    const format = detectModelFormat(safeName, prefix);
    if (!SUPPORTED_FORMATS.has(format)) {
      throw new Error(`Unsupported or unsafe serialized format: ${format}. Use TFLite, ONNX, GGUF, SafeTensors, Core ML .mlmodel, or ExecuTorch .pte/.ptd.`);
    }

    setStatus("Computing artifact identity", `${safeName} · ${format.toUpperCase()} · ${formatBytes(file.size)}`);
    const sha256 = await sha256FileHex(file, RANGE_CHUNK_BYTES);
    setStatus("Running browser-local static analysis", analysisDepth === "payload_integrity"
      ? "Inspecting serialized tensor payload evidence where the format supports it."
      : "Reading structure, metadata, interfaces, and tensor encodings without executing the model.");
    let analysis = await analyzeArtifact(file, safeName, format, analysisDepth);
    analysis = normalizeAnalysisSummaryContract(analysis);
    analysis.model_sha256 = sha256;
    analysis.file_size_bytes = file.size;
    analysis.cli_scan_policy = {
      schema: "deepbom.scan_policy.v1",
      requested_mode: analysisDepth === "payload_integrity" ? "integrity" : "structure",
      resolved_mode: analysisDepth === "payload_integrity" ? "integrity" : "structure",
      source: "mcp_app_browser_widget",
    };
    analysis.accelerator_bindings ||= [];
    if (format === "onnx") attachOnnxContractConflictCapsule(analysis);
    if (["onnx", "tflite"].includes(format)) {
      try { analysis.on_device_llm = buildOnDeviceLlmContract(analysis); } catch { /* not every graph is an LLM */ }
    }

    const artifact = { filename: safeName, format, sha256, size: file.size };
    const artifactIrContext = getArtifactIrContext(analysis, artifact);
    if (!artifactIrContext) throw new Error("The canonical Artifact Evidence IR could not be constructed for this file.");
    const analysisView = artifactIrContext.primary_view;
    const envelope = buildArtifactEvidenceEnvelope(analysisView, {
      hash: sha256,
      fileSizeBytes: file.size,
      filename: safeName,
      provenance: {
        analyzer: "DEEPBOM",
        version: ANALYZER_SEMANTIC_VERSION,
        command: "mcp_app_browser_analysis",
        execution_location: "mcp_app_browser_sandbox",
      },
    });
    const validation = validateArtifactEvidenceEnvelope(envelope);
    if (!validation.valid) throw new Error(`Evidence envelope validation failed: ${validation.errors.join(", ")}`);
    const summary = buildReviewSummary({ analysis: analysisView, envelope, artifactIrContext });
    const result = compactForConversation(summary, artifactIrContext.model_summary);

    setStatus("Returning a bounded result", "Only the evidence summary is sent to Claude; the selected model bytes are not sent to the DEEPBOM service.");
    const published = await app.callServerTool({ name: "deepbom_publish_browser_analysis", arguments: { result } });
    const returned = published?.structuredContent || result;
    await app.updateModelContext({
      content: [{ type: "text", text: resultText(returned) }],
      structuredContent: returned,
    });
    renderResult(result, artifactIrContext.model_summary);
  } catch (error) {
    await publishFailure(error, file);
  } finally {
    analyzing = false;
    root.querySelector("#analyze").disabled = false;
  }
}

async function analyzeArtifact(file, name, format, depth) {
  if (["gguf", "safetensors"].includes(format)) {
    return (await readMetadataModelFile(file, format, {
      scanMode: depth === "payload_integrity" ? "integrity" : "structure",
      onProgress: (progress) => setStatus("Reading tensor evidence", progressText(progress)),
    })).analysis;
  }
  if (format === "coreml") return (await readCoreMlModelFile(file)).analysis;
  if (file.size > FULL_FILE_LIMIT) {
    throw new Error(`${format.toUpperCase()} browser analysis is limited to ${formatBytes(FULL_FILE_LIMIT)} because this format requires the complete file. Use the local DEEPBOM desktop extension or CLI for this ${formatBytes(file.size)} artifact.`);
  }
  const bytes = new Uint8Array(await file.slice(0, file.size).arrayBuffer());
  if (format === "onnx") return analyzeOnnxModel(bytes, name);
  if (format === "executorch") return analyzeExecuTorchModel(bytes, name);
  if (format === "tflite") {
    await prepareSandboxWorker();
    return staticAuditWorkerClient.run(STATIC_AUDIT_OPERATION.TFLITE_ANALYZE, {
      bytes,
      filename: name,
      targetId: null,
      onStatus: (phase) => setStatus("Running isolated TFLite analysis", phase),
    });
  }
  throw new Error(`No MCP App browser analyzer is registered for ${format}.`);
}

function compactForConversation(summary, modelSummary) {
  const clip = (rows) => (rows || []).slice(0, MAX_FINDINGS_PER_KIND).map((row) => ({
    id: String(row.id || ""),
    title: String(row.title || ""),
    severity: String(row.severity || "informational"),
    evidence_class: String(row.evidence_class || "DERIVED"),
    recommendation: row.recommendation == null ? null : String(row.recommendation).slice(0, 800),
    ...(Number.isSafeInteger(row.affected_tensor_count) ? { affected_tensor_count: row.affected_tensor_count } : {}),
    ...(Array.isArray(row.affected_tensors) ? { affected_tensors: row.affected_tensors.slice(0, 16).map(String) } : {}),
  }));
  return {
    schema: "deepbom.browser_analysis_result.v1",
    analyzer_version: ANALYZER_SEMANTIC_VERSION,
    analysis_location: "mcp_app_browser_sandbox",
    artifact: {
      filename: String(summary.artifact.filename || "model"),
      format: String(summary.artifact.format || "unknown"),
      sha256: String(summary.artifact.sha256 || ""),
      byte_length: Number(summary.artifact.byte_length || 0),
      artifact_ir_sha256: summary.artifact.artifact_ir_sha256 || null,
    },
    verdict: {
      status: String(summary.verdict.status || "not_assessed"),
      artifact_defect_count: Number(summary.verdict.artifact_defect_count || 0),
      caution_count: Number(summary.verdict.caution_count || 0),
      evidence_needed_count: Number(summary.verdict.evidence_needed_count || 0),
    },
    graph: {
      operator_count: nullableNonnegativeInteger(summary.graph.operator_count),
      tensor_count: nullableNonnegativeInteger(summary.graph.tensor_count),
      total_macs: nullableNonnegativeNumber(summary.graph.total_macs),
      mac_confidence: String(summary.graph.mac_confidence || "partial"),
    },
    storage: summary.storage ? {
      container_kind: summary.storage.container_kind || null,
      tensor_count: summary.storage.tensor_count,
      declared_tensor_bytes: summary.storage.declared_tensor_bytes,
      encodings: (summary.storage.encodings || []).slice(0, 32),
      numerical_integrity_status: summary.storage.numerical_integrity_status,
      assessed_tensor_count: summary.storage.assessed_tensor_count,
      boundary: summary.storage.boundary,
    } : null,
    quantization: {
      classification: summary.quantization.classification,
      max_risk: summary.quantization.max_risk,
      max_risk_op_index: summary.quantization.max_risk_op_index,
      max_risk_op_name: summary.quantization.max_risk_op_name,
    },
    model_summary: compactModelSummaryForConversation(modelSummary),
    findings: {
      artifact_defects: clip(summary.findings.artifact_defects),
      cautions: clip(summary.findings.cautions),
      evidence_needed: clip(summary.findings.evidence_needed),
      truncated: [summary.findings.artifact_defects, summary.findings.cautions, summary.findings.evidence_needed]
        .some((rows) => (rows || []).length > MAX_FINDINGS_PER_KIND),
    },
    evidence_boundary: String(summary.verdict.scope || "Static serialized-artifact evidence only."),
    transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
    reproduction: summary.reproduction || null,
  };
}

function renderShell() {
  root.innerHTML = `<style>
    :root{color-scheme:light dark;font-family:ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}
    body{margin:0;background:transparent;color:CanvasText}.card{border:1px solid color-mix(in srgb,CanvasText 22%,transparent);border-radius:14px;padding:16px;background:Canvas}
    h2{font-size:16px;margin:0 0 8px}.status{font-weight:650;margin:0 0 5px}.detail,.privacy,.depth{font-size:13px;line-height:1.45;margin:0;color:color-mix(in srgb,CanvasText 72%,transparent)}
    .picker{display:flex;gap:8px;align-items:center;margin:14px 0}.picker input{min-width:0;flex:1}.picker button{border:1px solid CanvasText;border-radius:8px;padding:7px 12px;background:Canvas;color:CanvasText;font-weight:650}
    .privacy{border-top:1px solid color-mix(in srgb,CanvasText 15%,transparent);margin-top:12px;padding-top:10px}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
    .metric{border:1px solid color-mix(in srgb,CanvasText 14%,transparent);border-radius:10px;padding:8px}.metric b{display:block;font-size:18px}.metric span{font-size:11px}.error{color:#a02020;white-space:pre-wrap}.model-summary{border-top:1px solid color-mix(in srgb,CanvasText 15%,transparent);margin-top:14px;padding-top:12px}.model-summary h3{font-size:13px;margin:0 0 7px}.model-summary pre{overflow:auto;max-height:360px;padding:9px;border:1px solid color-mix(in srgb,CanvasText 18%,transparent);border-radius:8px;font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre}
  </style><section class="card"><h2>DEEPBOM</h2><p class="status" id="status">Connecting to the MCP App host</p><p class="detail" id="detail">No file has been read.</p>
  <div class="picker"><input id="file" type="file" accept=".tflite,.onnx,.gguf,.safetensors,.mlmodel,.pte,.ptd" disabled><button id="analyze" type="button" disabled>Analyze</button></div>
  <p class="depth" id="depth"></p><div id="result"></div><p class="privacy">You choose the file explicitly. Analysis runs in this browser sandbox; the DEEPBOM service receives the bounded result, not the selected model bytes.</p></section>`;
  updateDepthText();
}

function updateDepthText() {
  const element = root?.querySelector("#depth");
  if (element) element.textContent = analysisDepth === "payload_integrity"
    ? "Mode: structure and serialized payload integrity"
    : "Mode: serialized structure and metadata";
}

function setStatus(status, detail) {
  root.querySelector("#status").textContent = status;
  root.querySelector("#detail").textContent = detail || "";
}

function renderResult(result, modelSummary) {
  setStatus("Static evidence ready", `${result.artifact.filename} · ${result.artifact.format.toUpperCase()} · sha256:${result.artifact.sha256.slice(0, 12)}…`);
  root.querySelector("#result").innerHTML = `<div class="metrics">
    <div class="metric"><b>${result.verdict.artifact_defect_count}</b><span>artifact defects</span></div>
    <div class="metric"><b>${result.verdict.caution_count}</b><span>cautions</span></div>
    <div class="metric"><b>${result.verdict.evidence_needed_count}</b><span>evidence gaps</span></div>
  </div><section class="model-summary"><h3>Format-neutral model summary</h3><pre tabindex="0"></pre></section>`;
  root.querySelector(".model-summary pre").textContent = renderModelSummaryTable(modelSummary);
}

function renderError(error, code = "analysis_failed") {
  setStatus("Analysis could not be completed", "The file was not interpreted as a successful result.");
  root.querySelector("#result").innerHTML = `<p class="detail error"></p>`;
  root.querySelector(".error").textContent = `${code}: ${error?.message || String(error)}`;
}

async function publishFailure(error, file = null) {
  const failure = structuredFailure(error, file);
  renderError(error, failure.code);
  try {
    const published = await app.callServerTool({ name: "deepbom_publish_browser_error", arguments: { error: failure } });
    const returned = published?.structuredContent || failure;
    await app.updateModelContext({
      content: [{ type: "text", text: `DEEPBOM did not complete: ${returned.message} Suggested action: ${returned.suggested_action}` }],
      structuredContent: returned,
    });
  } catch {
    // Preserve the original visible diagnostic if the host bridge is unavailable.
  }
}

function structuredFailure(error, file) {
  const message = String(error?.message || error || "The analysis did not complete.").slice(0, 2048);
  let code = "analysis_failed";
  let suggestedAction = "Retry with a supported single-file deployment artifact. Use the local DEEPBOM desktop extension or CLI if the failure persists.";
  if (/choose one supported|selection/i.test(message)) {
    code = "file_selection_missing";
    suggestedAction = "Choose one supported model file in the MCP App, then select Analyze.";
  } else if (/unsupported or unsafe serialized format|No MCP App browser analyzer/i.test(message)) {
    code = "unsupported_format";
    suggestedAction = "Choose TFLite, ONNX, GGUF, SafeTensors, Core ML .mlmodel, or ExecuTorch .pte/.ptd, or use the local tool for a package.";
  } else if (/limited to|large file|byte ranges/i.test(message)) {
    code = "browser_limit";
    suggestedAction = "Use the local DEEPBOM desktop extension or CLI for this artifact.";
  } else if (/payload|protobuf|flatbuffer|header|truncat|invalid|malformed/i.test(message)) {
    code = "artifact_parse_failed";
    suggestedAction = "Confirm the file is complete and has the expected format. Re-export or re-download it before retrying.";
  }
  return {
    schema: "deepbom.browser_error.v1",
    analyzer_version: ANALYZER_SEMANTIC_VERSION,
    analysis_location: "mcp_app_browser_sandbox",
    code,
    message,
    suggested_action: suggestedAction,
    file_name: file?.name ? String(file.name).slice(0, 512) : null,
    transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
  };
}

function resultText(result) {
  return `DEEPBOM ${result.analyzer_version} analyzed ${result.artifact.filename} (${result.artifact.format}, sha256:${result.artifact.sha256}). Static result: ${result.verdict.artifact_defect_count} artifact defect(s), ${result.verdict.caution_count} caution(s), and ${result.verdict.evidence_needed_count} evidence gap(s). Preserve the attached evidence boundary.`;
}

function nullableNonnegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function nullableNonnegativeNumber(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function progressText(progress) {
  const phase = String(progress?.phase || "Reading serialized payload");
  const index = Number(progress?.index);
  const count = Number(progress?.count);
  return Number.isSafeInteger(index) && Number.isSafeInteger(count) && count > 0
    ? `${phase} ${index + 1}/${count}`
    : phase;
}
