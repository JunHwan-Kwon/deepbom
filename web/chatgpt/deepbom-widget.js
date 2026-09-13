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

const FULL_FILE_LIMIT = 128 * 1024 * 1024;
const RANGE_CHUNK_BYTES = 4 * 1024 * 1024;
const MAX_FINDINGS_PER_KIND = 8;
const SUPPORTED_FORMATS = new Set(["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"]);
const root = document.getElementById("deepbom-chatgpt-root");
const staticAuditWorkerClient = createStaticAuditWorkerClient({
  createWorker: () => new Worker(
    new URL("../workers/static-audit-worker.js", import.meta.url),
    { type: "module", name: "deepbom-chatgpt-static-audit" },
  ),
});

renderShell();
void start().catch((error) => publishFailure(error));

async function start() {
  const openai = await waitForOpenAi();
  const input = openai.toolInput || {};
  const supplied = input.file || window.openai?.toolOutput?.file || null;
  if (!supplied?.file_id && !supplied?.download_url) {
    throw new Error("ChatGPT did not provide an authorized file. Attach one supported model artifact and run DEEPBOM again.");
  }
  const file = await resolveOpenAiFile(openai, supplied);
  const depth = input.analysis_depth === "payload_integrity" ? "payload_integrity" : "structure";

  setStatus("Inspecting the authorized attachment", "Resolving its byte length and format without sending it to the DEEPBOM service.");
  const remote = await RemoteArtifactFile.open(file.download_url, file.file_name || "model");
  const prefix = new Uint8Array(await remote.slice(0, Math.min(remote.size, 64 * 1024)).arrayBuffer());
  const format = detectModelFormat(remote.name, prefix);
  if (!SUPPORTED_FORMATS.has(format)) {
    throw new Error(`Unsupported or unsafe serialized format: ${format}. Use TFLite, ONNX, GGUF, SafeTensors, Core ML .mlmodel, or ExecuTorch .pte/.ptd.`);
  }

  setStatus("Computing artifact identity", `${remote.name} · ${format.toUpperCase()} · ${formatBytes(remote.size)}. SHA-256 may require reading the complete attachment.`);
  const sha256 = await sha256FileHex(remote, RANGE_CHUNK_BYTES);

  setStatus("Running static analysis", depth === "payload_integrity"
    ? "Inspecting serialized tensor payload evidence where the format supports it."
    : "Reading structure, metadata, interfaces, and tensor encodings without executing the model.");
  let analysis = await analyzeRemoteArtifact(remote, format, depth);
  analysis = normalizeAnalysisSummaryContract(analysis);
  analysis.model_sha256 = sha256;
  analysis.file_size_bytes = remote.size;
  analysis.cli_scan_policy = {
    schema: "deepbom.scan_policy.v1",
    requested_mode: depth === "payload_integrity" ? "integrity" : "structure",
    resolved_mode: depth === "payload_integrity" ? "integrity" : "structure",
    source: "chatgpt_browser_widget",
  };
  analysis.accelerator_bindings ||= [];
  if (format === "onnx") attachOnnxContractConflictCapsule(analysis);
  if (["onnx", "tflite"].includes(format)) {
    try { analysis.on_device_llm = buildOnDeviceLlmContract(analysis); } catch { /* not every graph is an LLM */ }
  }

  const artifact = { filename: remote.name, format, sha256, size: remote.size };
  const artifactIrContext = getArtifactIrContext(analysis, artifact);
  if (!artifactIrContext) throw new Error("The canonical Artifact Evidence IR could not be constructed for this attachment.");
  const analysisView = artifactIrContext.primary_view;
  const envelope = buildArtifactEvidenceEnvelope(analysisView, {
    hash: sha256,
    fileSizeBytes: remote.size,
    filename: remote.name,
    provenance: {
      analyzer: "DEEPBOM",
      version: ANALYZER_SEMANTIC_VERSION,
      command: "chatgpt_browser_analysis",
      execution_location: "chatgpt_browser_sandbox",
    },
  });
  const validation = validateArtifactEvidenceEnvelope(envelope);
  if (!validation.valid) throw new Error(`Evidence envelope validation failed: ${validation.errors.join(", ")}`);
  const summary = buildReviewSummary({ analysis: analysisView, envelope, artifactIrContext });
  const result = compactForConversation(summary);

  setStatus("Returning the bounded result", "Only the evidence summary below is sent to the conversation; model bytes remain outside the DEEPBOM service.");
  const published = await openai.callTool("deepbom_publish_analysis", { result });
  renderResult(result);
  const returned = published?.structuredContent || result;
  if (typeof openai.sendFollowUpMessage === "function") {
    await openai.sendFollowUpMessage({
      prompt: `DEEPBOM completed a browser-local static audit. Summarize this result for me, keeping artifact defects, cautions, and evidence gaps separate and preserving the evidence boundary:\n${JSON.stringify(returned)}`,
      scrollToBottom: true,
    });
  }
}

async function analyzeRemoteArtifact(file, format, depth) {
  if (["gguf", "safetensors"].includes(format)) {
    return (await readMetadataModelFile(file, format, {
      scanMode: depth === "payload_integrity" ? "integrity" : "structure",
      onProgress: (progress) => setStatus("Reading tensor evidence", progressText(progress)),
    })).analysis;
  }
  if (format === "coreml") return (await readCoreMlModelFile(file)).analysis;
  if (file.size > FULL_FILE_LIMIT) {
    throw new Error(`${format.toUpperCase()} browser analysis is limited to ${formatBytes(FULL_FILE_LIMIT)} because this format requires the complete file. Use the local DEEPBOM MCP for this ${formatBytes(file.size)} artifact.`);
  }
  const bytes = new Uint8Array(await file.slice(0, file.size).arrayBuffer());
  if (format === "onnx") return analyzeOnnxModel(bytes, file.name);
  if (format === "executorch") return analyzeExecuTorchModel(bytes, file.name);
  if (format === "tflite") {
    return staticAuditWorkerClient.run(STATIC_AUDIT_OPERATION.TFLITE_ANALYZE, {
      bytes,
      filename: file.name,
      targetId: null,
      onStatus: (phase) => setStatus("Running isolated TFLite analysis", phase),
    });
  }
  throw new Error(`No ChatGPT browser analyzer is registered for ${format}.`);
}

function compactForConversation(summary) {
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
    schema: "deepbom.chatgpt_analysis_result.v1",
    analyzer_version: ANALYZER_SEMANTIC_VERSION,
    analysis_location: "chatgpt_browser_sandbox",
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
    findings: {
      artifact_defects: clip(summary.findings.artifact_defects),
      cautions: clip(summary.findings.cautions),
      evidence_needed: clip(summary.findings.evidence_needed),
      truncated: [
        summary.findings.artifact_defects,
        summary.findings.cautions,
        summary.findings.evidence_needed,
      ].some((rows) => (rows || []).length > MAX_FINDINGS_PER_KIND),
    },
    evidence_boundary: String(summary.verdict.scope || "Static serialized-artifact evidence only."),
    transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
    reproduction: summary.reproduction || null,
  };
}

class RemoteArtifactFile {
  constructor(url, name, size, cachedBytes = null) {
    this.url = url;
    this.name = name;
    this.size = size;
    this.cachedBytes = cachedBytes;
  }

  static async open(url, name) {
    const safeName = String(name || "model").split(/[\\/]/).pop().slice(0, 512) || "model";
    let response = null;
    try {
      response = await fetch(url, { method: "HEAD", cache: "no-store" });
    } catch {
      // Some authorized attachment origins do not expose HEAD through CORS.
      // A one-byte range request is the portable size-discovery fallback.
    }
    let size = response ? contentLength(response) : null;
    if (!response?.ok || !Number.isSafeInteger(size)) {
      response = await fetch(url, { headers: { Range: "bytes=0-0" }, cache: "no-store" });
      if (!response.ok) throw new Error(`ChatGPT attachment download failed with HTTP ${response.status}.`);
      size = contentRangeTotal(response) ?? contentLength(response);
      if (!Number.isSafeInteger(size)) throw new Error("The attachment server did not report a safe byte length.");
      if (response.status === 200) {
        if (size > FULL_FILE_LIMIT) throw new Error("The attachment server does not support byte ranges for this large file. Use the local DEEPBOM MCP.");
        return new RemoteArtifactFile(url, safeName, size, new Uint8Array(await response.arrayBuffer()));
      }
      await response.body?.cancel();
    }
    return new RemoteArtifactFile(url, safeName, size);
  }

  slice(start = 0, end = this.size) {
    const boundedStart = Math.max(0, Math.min(this.size, Number(start) || 0));
    const boundedEnd = Math.max(boundedStart, Math.min(this.size, Number(end) || this.size));
    return {
      arrayBuffer: async () => {
        if (this.cachedBytes) return detachedArrayBuffer(this.cachedBytes.slice(boundedStart, boundedEnd));
        if (boundedStart === boundedEnd) return new ArrayBuffer(0);
        const response = await fetch(this.url, {
          headers: { Range: `bytes=${boundedStart}-${boundedEnd - 1}` },
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`Attachment range read failed with HTTP ${response.status}.`);
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (response.status === 206) {
          if (bytes.byteLength !== boundedEnd - boundedStart) throw new Error("Attachment range response length is inconsistent.");
          return detachedArrayBuffer(bytes);
        }
        if (this.size > FULL_FILE_LIMIT) throw new Error("The attachment server stopped honoring byte ranges. Use the local DEEPBOM MCP for this artifact.");
        this.cachedBytes = bytes;
        if (bytes.byteLength !== this.size) throw new Error("Attachment response length is inconsistent with its declared size.");
        return detachedArrayBuffer(bytes.slice(boundedStart, boundedEnd));
      },
    };
  }
}

async function resolveOpenAiFile(openai, supplied) {
  let downloadUrl = supplied.download_url || "";
  if (!downloadUrl && supplied.file_id && typeof openai.getFileDownloadUrl === "function") {
    const resolved = await openai.getFileDownloadUrl({ fileId: supplied.file_id });
    downloadUrl = resolved?.downloadUrl || "";
  }
  if (!/^https:\/\//i.test(downloadUrl)) throw new Error("ChatGPT did not provide an HTTPS download URL for the attachment.");
  return { ...supplied, download_url: downloadUrl };
}

function waitForOpenAi() {
  if (window.openai) return Promise.resolve(window.openai);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("The ChatGPT component bridge did not initialize.")), 10_000);
    const listener = () => {
      if (!window.openai) return;
      clearTimeout(timeout);
      window.removeEventListener("openai:set_globals", listener);
      resolve(window.openai);
    };
    window.addEventListener("openai:set_globals", listener);
  });
}

function renderShell() {
  root.innerHTML = `<style>
    :root{color-scheme:light dark;font-family:ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}
    body{margin:0;background:transparent;color:CanvasText}.card{border:1px solid color-mix(in srgb,CanvasText 22%,transparent);border-radius:14px;padding:16px;background:Canvas}
    h2{font-size:16px;margin:0 0 8px}.status{font-weight:650;margin:0 0 5px}.detail,.privacy{font-size:13px;line-height:1.45;margin:0;color:color-mix(in srgb,CanvasText 72%,transparent)}
    .privacy{border-top:1px solid color-mix(in srgb,CanvasText 15%,transparent);margin-top:12px;padding-top:10px}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
    .metric{border:1px solid color-mix(in srgb,CanvasText 14%,transparent);border-radius:10px;padding:8px}.metric b{display:block;font-size:18px}.metric span{font-size:11px}.error{color:#a02020;white-space:pre-wrap}
  </style><section class="card"><h2>DEEPBOM</h2><p class="status" id="status">Preparing browser-local analysis</p><p class="detail" id="detail">Waiting for the ChatGPT file authorization.</p><div id="result"></div><p class="privacy">The attachment is read in this browser sandbox. DEEPBOM's service receives the bounded result, not the model bytes.</p></section>`;
}

function setStatus(status, detail) {
  root.querySelector("#status").textContent = status;
  root.querySelector("#detail").textContent = detail || "";
}

function renderResult(result) {
  setStatus("Static evidence ready", `${result.artifact.filename} · ${result.artifact.format.toUpperCase()} · sha256:${result.artifact.sha256.slice(0, 12)}…`);
  root.querySelector("#result").innerHTML = `<div class="metrics">
    <div class="metric"><b>${result.verdict.artifact_defect_count}</b><span>artifact defects</span></div>
    <div class="metric"><b>${result.verdict.caution_count}</b><span>cautions</span></div>
    <div class="metric"><b>${result.verdict.evidence_needed_count}</b><span>evidence gaps</span></div>
  </div>`;
}

function renderError(error) {
  setStatus("Analysis could not be completed", "The attachment was not interpreted as a successful result.");
  root.querySelector("#result").innerHTML = `<p class="detail error"></p>`;
  root.querySelector(".error").textContent = error?.message || String(error);
}

async function publishFailure(error) {
  renderError(error);
  const failure = structuredFailure(error);
  try {
    const openai = await waitForOpenAi();
    const published = await openai.callTool("deepbom_publish_error", { error: failure });
    const returned = published?.structuredContent || failure;
    if (typeof openai.sendFollowUpMessage === "function") {
      await openai.sendFollowUpMessage({
        prompt: `DEEPBOM could not complete the browser-local audit. Explain the failure and suggested action without treating it as an artifact defect:\n${JSON.stringify(returned)}`,
        scrollToBottom: true,
      });
    }
  } catch {
    // The visible component error remains available if the ChatGPT bridge itself
    // is unavailable; do not replace the original diagnostic with bridge noise.
  }
}

function structuredFailure(error) {
  const message = String(error?.message || error || "The analysis did not complete.").slice(0, 2048);
  const fileName = String(window.openai?.toolInput?.file?.file_name || "").slice(0, 512) || null;
  let code = "analysis_failed";
  let suggestedAction = "Retry with a supported single-file deployment artifact. Use the local DEEPBOM CLI or MCP if the failure persists.";
  if (/did not provide an authorized file|download URL/i.test(message)) {
    code = "file_authorization_missing";
    suggestedAction = "Attach one supported model file in ChatGPT and run DEEPBOM again.";
  } else if (/unsupported or unsafe serialized format|No ChatGPT browser analyzer/i.test(message)) {
    code = "unsupported_format";
    suggestedAction = "Use TFLite, ONNX, GGUF, SafeTensors, Core ML .mlmodel, or ExecuTorch .pte/.ptd, or use the local tool for a package.";
  } else if (/limited to|large file|byte ranges/i.test(message)) {
    code = "browser_limit";
    suggestedAction = "Use the local DEEPBOM CLI or local MCP for this artifact.";
  } else if (/download failed|range read failed|response length|safe byte length/i.test(message)) {
    code = "attachment_read_failed";
    suggestedAction = "Reattach the file and retry. If the attachment origin cannot provide a complete or ranged response, use the local DEEPBOM CLI or MCP.";
  } else if (/payload|protobuf|flatbuffer|header|truncat|invalid|malformed/i.test(message)) {
    code = "artifact_parse_failed";
    suggestedAction = "Confirm the file is complete and has the expected format. Re-export or re-download it before retrying.";
  }
  return {
    schema: "deepbom.chatgpt_error.v1",
    analyzer_version: ANALYZER_SEMANTIC_VERSION,
    analysis_location: "chatgpt_browser_sandbox",
    code,
    message,
    suggested_action: suggestedAction,
    file_name: fileName,
    transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
  };
}

function contentLength(response) {
  const value = Number(response.headers.get("content-length"));
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function contentRangeTotal(response) {
  const match = String(response.headers.get("content-range") || "").match(/^bytes\s+\d+-\d+\/(\d+)$/i);
  const value = match ? Number(match[1]) : NaN;
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function detachedArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
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
