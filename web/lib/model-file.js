import { formatBytes } from "./format.js";

import { formatEvidenceScope } from "./format-evidence-scope.js";

export const FULL_AUDIT_ESTIMATE_METHOD = "deepbom.full_audit_estimate.2026-07-23.2";
export const MODEL_FORMAT_ADAPTERS = Object.freeze({
  tflite: Object.freeze({ id: "tflite", label: "TFLite", extensions: [".tflite"], load: "full", analyzer: "wasm", executable: false, capabilities: ["static_analysis", "artifact_export", "runtime_execution", "target_profiles", "protected_source_analysis", "experimental_tflite_research", "multi_projection_geometry"] }),
  onnx: Object.freeze({ id: "onnx", label: "ONNX", extensions: [".onnx"], load: "full_or_external_data", analyzer: "javascript", executable: false, capabilities: ["static_analysis", "artifact_export", "runtime_execution", "target_profiles", "protected_source_analysis"] }),
  gguf: Object.freeze({ id: "gguf", label: "GGUF", extensions: [".gguf"], load: "progressive_header", analyzer: "javascript_metadata", executable: false, capabilities: ["metadata_analysis", "artifact_export"] }),
  safetensors: Object.freeze({ id: "safetensors", label: "SafeTensors", extensions: [".safetensors"], load: "declared_header", analyzer: "javascript_metadata", executable: false, capabilities: ["metadata_analysis", "artifact_export"] }),
  coreml: Object.freeze({ id: "coreml", label: "Core ML", extensions: [".mlmodel", ".mlpackage"], load: "file_or_package", analyzer: "javascript_metadata", executable: false, capabilities: ["metadata_analysis", "serialized_graph_analysis", "weight_encoding_analysis", "artifact_export"] }),
  executorch: Object.freeze({ id: "executorch", label: "ExecuTorch", extensions: [".pte", ".ptd"], load: "full_or_external_data", analyzer: "javascript", executable: false, capabilities: ["static_analysis", "serialized_graph_analysis", "artifact_export"] }),
  pytorch_pickle: Object.freeze({ id: "pytorch_pickle", label: "PyTorch pickle", extensions: [".pt", ".pth", ".ckpt"], load: "rejected", analyzer: "none", executable: true, capabilities: [] }),
  unsupported: Object.freeze({ id: "unsupported", label: "Unsupported", extensions: [], load: "rejected", analyzer: "none", executable: false, capabilities: [] }),
});

export function modelFormatAdapter(format) {
  return MODEL_FORMAT_ADAPTERS[String(format || "").toLowerCase()] || MODEL_FORMAT_ADAPTERS.unsupported;
}

export function modelFormatGate(format) {
  const adapter = modelFormatAdapter(format);
  if (adapter.load === "rejected") {
    return {
      adapter,
      blocked: true,
      reason: adapter.executable ? "unsafe_serialized_code" : "unsupported_format",
      message: adapter.executable
        ? "Unsafe serialized-code formats are not deserialized. Use an exported model or SafeTensors."
        : "Unsupported model format.",
    };
  }
  if (adapter.analyzer === "not_implemented") {
    return {
      adapter,
      blocked: true,
      reason: "analyzer_not_implemented",
      message: `${adapter.label} metadata analysis is not available in this build.`,
    };
  }
  return { adapter, blocked: false, reason: null, message: "" };
}

export function modelSupportsCapability(format, capability) {
  return modelFormatAdapter(format).capabilities.includes(String(capability || ""));
}

export function modelReportBindingMatchesAnalysis(binding, analysis) {
  if (!binding || !analysis) return false;
  const targetProfileApplicable = modelSupportsCapability(analysis.format, "target_profiles");
  return binding.bindingScope === "artifact"
    ? !targetProfileApplicable
    : Boolean(binding.targetId && analysis.target_profile?.id === binding.targetId);
}

export function modelExportAvailability(analysis) {
  const format = analysis?.format;
  const staticAnalysis = Boolean(analysis && modelSupportsCapability(format, "static_analysis"));
  return {
    graph: Boolean(staticAnalysis || analysis
      && modelSupportsCapability(format, "serialized_graph_analysis")
      && (analysis.ops || []).length),
    performanceDerivatives: staticAnalysis,
  };
}

export function estimateModelAnalysis(file, probe, options = {}) {
  const sizeMb = file.size / (1024 * 1024);
  const formatId = detectModelFormat(file?.name || "", probe?.header);
  const format = modelFormatAdapter(formatId).label;
  const probeThroughput = probe?.readBytes ? probe.readBytes / Math.max(1, probe.elapsed) : 0;
  const readPenaltyMs = probeThroughput ? Math.min(3000, file.size / probeThroughput) : sizeMb * 18;
  const comparisonTargetCount = Math.max(1, Number(options.comparisonTargetCount || 4));
  const recentDurations = matchingAuditDurations(
    options.auditTimings,
    format,
    file.size,
    comparisonTargetCount,
  );
  const observedMedianMs = median(recentDurations);
  const graphFactor = sizeMb > 50 ? 1.4 : sizeMb > 15 ? 1.18 : 1;
  const targetFactor = 1 + Math.max(0, comparisonTargetCount - 4) * 0.12;
  // Full-workflow calibration basis: the optimized 3.41 MiB quant TFLite audit
  // measured 15.2 s across four targets; the 21.6 KiB ONNX sample remains sub-second.
  const calibratedMs = format === "ONNX"
    ? 220 + readPenaltyMs + sizeMb * 180 * graphFactor
    : format === "ExecuTorch"
      ? 180 + readPenaltyMs + sizeMb * 120 * graphFactor
    : format === "TFLite"
      ? (10000 + readPenaltyMs + sizeMb * 1600 * graphFactor) * targetFactor
      : 180 + readPenaltyMs;
  const estimateMs = observedMedianMs || calibratedMs;
  const low = Math.max(format === "TFLite" ? 1000 : 120, estimateMs * (observedMedianMs ? 0.75 : 0.7));
  const high = Math.max(low + 80, estimateMs * (observedMedianMs ? 1.6 : 1.9));
  return {
    format,
    formatId,
    readBytes: probe?.readBytes || 0,
    probeMs: probe?.elapsed || 0,
    estimateMs,
    lowMs: low,
    highMs: high,
    sizeMb,
    comparisonTargetCount,
    timingSampleCount: recentDurations.length,
    estimateSource: observedMedianMs ? "local_browser_history" : "calibrated_full_audit",
    estimateMethod: FULL_AUDIT_ESTIMATE_METHOD,
  };
}

export async function inspectModelFile(file, options = {}) {
  const readBytes = Math.min(file.size, 64 * 1024);
  const started = performance.now();
  const header = new Uint8Array(await file.slice(0, readBytes).arrayBuffer());
  const elapsed = Math.max(0.1, performance.now() - started);
  return estimateModelAnalysis(file, { readBytes, elapsed, header }, typeof options === "function" ? options(header) : options);
}

export function selectedModelCopy(file) {
  return {
    name: file.name,
    meta: `${formatBytes(file.size)} / local file handle / full path hidden by browser privacy`,
    estimate: "Inspecting",
    note: "Reading a tiny local slice to estimate static audit cost.",
    status: "Preparing",
  };
}

export function stagedModelCopy(file, inspection, targetLabel) {
  const localHistory = Number(inspection.timingSampleCount || 0) > 0;
  const targetBound = inspection.format === "TFLite";
  const scope = formatEvidenceScope(inspection.formatId);
  return {
    name: file.name,
    meta: `${inspection.format} / ${formatBytes(file.size)}${targetBound ? ` / target ${targetLabel}` : ` / ${scope.stagedDescriptor}`}`,
    estimate: formatEstimate(inspection),
    note: localHistory
      ? `Based on ${inspection.timingSampleCount} recent full audit${inspection.timingSampleCount === 1 ? "" : "s"} in this browser for a similar ${inspection.format} size and target count.`
      : inspection.format === "TFLite"
      ? `Cold-run range includes selected-target analysis, ${inspection.comparisonTargetCount}-target frontier work, repair analysis, and rendering; CPU and browser speed vary.`
      : targetBound
        ? "Cold-run range includes full model analysis and rendering; CPU and browser speed vary."
      : inspection.format === "ExecuTorch"
        ? "Cold-run range includes bounded ET12/FT01 parsing, segment conservation, execution-plan projection, and rendering; backend blobs are not executed."
        : "Cold-run range includes bounded metadata parsing, streaming hashing, evidence projection, and rendering; tensor payload values are not loaded.",
    status: "Ready to run",
  };
}

export function inferModelFormat(file, header) {
  return modelFormatAdapter(detectModelFormat(file?.name || "", header)).label;
}

export function formatEstimate(inspection) {
  if (!inspection || inspection.highMs < 1000) return "<1 sec";
  const low = roundedDurationBound(inspection.lowMs, "low");
  const high = Math.max(low + 1, roundedDurationBound(inspection.highMs, "high"));
  if (high <= 2) return "about 1-2 sec";
  return `about ${low}-${high} sec${inspection.estimateSource === "local_browser_history" ? " on this device" : ""}`;
}

export function formatMeasuredAudit(durationMs) {
  const value = Number(durationMs || 0);
  if (!Number.isFinite(value) || value <= 0) return "Not measured";
  if (value < 1000) return `${Math.round(value)} ms measured`;
  return `${(value / 1000).toFixed(1)} sec measured`;
}

function matchingAuditDurations(records, format, sizeBytes, comparisonTargetCount) {
  if (!Array.isArray(records) || !Number.isFinite(sizeBytes) || sizeBytes <= 0) return [];
  return records
    .filter((record) => String(record?.format || "").toUpperCase() === String(format || "").toUpperCase())
    .filter((record) => Number(record?.comparisonTargetCount || 0) === comparisonTargetCount)
    .filter((record) => {
      const recordedSize = Number(record?.sizeBytes || 0);
      const ratio = recordedSize / sizeBytes;
      return Number.isFinite(ratio) && ratio >= 0.5 && ratio <= 2;
    })
    .map((record) => Number(record?.durationMs || 0))
    .filter((value) => Number.isFinite(value) && value >= 50 && value <= 10 * 60 * 1000)
    .slice(-8);
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function roundedDurationBound(valueMs, direction) {
  const seconds = Math.max(1, valueMs / 1000);
  if (seconds < 10) return direction === "low" ? Math.floor(seconds) : Math.ceil(seconds);
  return direction === "low" ? Math.floor(seconds / 5) * 5 : Math.ceil(seconds / 5) * 5;
}

export function detectModelFormat(filename, bytes) {
  const lower = String(filename || "").toLowerCase();
  for (const adapter of Object.values(MODEL_FORMAT_ADAPTERS)) {
    if (adapter.extensions.some((extension) => lower.endsWith(extension))) return adapter.id;
  }
  if (bytes?.length >= 8 && bytes[4] === 0x54 && bytes[5] === 0x46 && bytes[6] === 0x4c && bytes[7] === 0x33) return "tflite";
  if (bytes?.length >= 8 && ((bytes[4] === 0x45 && bytes[5] === 0x54 && bytes[6] === 0x31 && bytes[7] === 0x32)
    || (bytes[4] === 0x46 && bytes[5] === 0x54 && bytes[6] === 0x30 && bytes[7] === 0x31))) return "executorch";
  if (bytes?.length >= 4 && bytes[0] === 0x47 && bytes[1] === 0x47 && bytes[2] === 0x55 && bytes[3] === 0x46) return "gguf";
  if (looksLikeSafeTensorsHeader(bytes)) return "safetensors";
  if (bytes?.length >= 4 && bytes[0] === 0x08) return "onnx";
  return "unsupported";
}

function looksLikeSafeTensorsHeader(bytes) {
  if (!bytes || bytes.length < 9 || bytes[8] !== 0x7b) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const low = view.getUint32(0, true);
  const high = view.getUint32(4, true);
  return high === 0 && low >= 2 && low <= 100 * 1024 * 1024;
}

export function buildModelIdentity({
  analysis = null,
  filename = "",
  modelBytes = [],
  selectedTargetProfile = null,
  selectedTargetId = "",
  selectedTargetLabel = "",
} = {}) {
  const resolvedFilename = analysis?.filename || filename || "";
  const resolvedFormat = analysis?.format || detectModelFormat(resolvedFilename, modelBytes || []);
  const targetBound = modelSupportsCapability(resolvedFormat, "target_profiles");
  const target = targetBound ? analysis?.target_profile || selectedTargetProfile || {} : {};
  return {
    filename: resolvedFilename,
    format: resolvedFormat,
    sha256: analysis?.model_sha256 || "",
    target_id: targetBound ? target.id || selectedTargetId : "",
    target_label: targetBound ? target.label || selectedTargetLabel : "",
    operator_count: analysis?.operator_count == null ? (Array.isArray(analysis?.ops) && !["gguf", "safetensors"].includes(resolvedFormat) ? analysis.ops.length : null) : Number(analysis.operator_count),
    tensor_count: analysis?.tensor_count == null ? (Array.isArray(analysis?.tensors) ? analysis.tensors.length : null) : Number(analysis.tensor_count),
    total_macs: analysis?.total_macs == null ? null : Number(analysis.total_macs),
  };
}
