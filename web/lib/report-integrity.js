import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
export const REPORT_VERIFICATION_SENTINEL = "<!-- deepbom-verification -->";

export function reportBodyForFingerprint(text) {
  const body = String(text || "");
  const idx = body.indexOf(REPORT_VERIFICATION_SENTINEL);
  return (idx >= 0 ? body.slice(0, idx) : body).replace(/\s+$/, "");
}

export function collectArtifactIntegrity(analysis = {}, identity = {}, fileSizeBytes = 0) {
  const format = String(identity.format || analysis.format || "unknown").toLowerCase();
  const customLikeOps = (analysis.histogram || []).filter((item) => /CUSTOM|FLEX|SELECT/i.test(item.name || ""));
  const domainAnalysis = analysis.onnx_domain_analysis || null;
  const externalDomains = domainAnalysis
    ? [...(domainAnalysis.external_custom_domains || [])]
    : [...new Set((artifactIrOperators(analysis) || [])
      .map((op) => String(op.domain || "").trim())
      .filter((domain) => domain && !["ai.onnx", "ai.onnx.ml"].includes(domain)))].sort();
  const externalDataTensorCount = Number(analysis.onnx_external_data_tensor_count || 0);
  const externalDataEvidence = analysis.onnx_external_data || {};
  const externalDataVerifiedCount = Number(externalDataEvidence.verified_payload_count || 0);
  const externalDataVerificationFailedCount = Number(externalDataEvidence.payload_verification_failed_count || 0);
  const externalDataIncompleteCount = Math.max(0, externalDataTensorCount - externalDataVerifiedCount);
  const onnxOpsets = formatOpsets(analysis.opsets);
  const schemaOrOpset = formatArtifactSchema(analysis, format, onnxOpsets);
  const externalDomainSummary = format === "onnx"
    ? (externalDomains.length
      ? `external custom registry domains ${externalDomains.join(" / ")}`
      : `no external custom registry domain; ${domainAnalysis?.model_local_function_call_count || 0} model-local function call(s); ${domainAnalysis?.ort_contrib_node_count || 0} ORT contrib node(s)`)
    : "external domains not applicable";
  const externalDataSummary = format === "onnx"
    ? (externalDataTensorCount
      ? `${externalDataTensorCount} ONNX external-data tensor(s): ${externalDataVerifiedCount} verified, ${externalDataVerificationFailedCount} failed verification, ${externalDataIncompleteCount} incomplete; ${Number(externalDataEvidence.verified_payload_bytes || 0)} verified payload byte(s)`
      : "no ONNX external-data tensor detected")
    : "external data not applicable";
  const tfliteBufferPosture = collectTfliteTensorBufferPosture(analysis, analysis.file_size || fileSizeBytes || 0, format);
  return {
    format,
    sha256: identity.sha256 || analysis.model_sha256 || "pending-browser-export",
    file_size_bytes: analysis.file_size || fileSizeBytes || 0,
    schema_or_opset: schemaOrOpset,
    tflite_version: analysis.version || "",
    onnx_ir_version: analysis.onnx_ir_version || 0,
    onnx_producer: analysis.producer || "",
    onnx_opsets: onnxOpsets,
    graph_name: analysis.graph_name || "",
    subgraph_count: analysis.subgraph_count || 1,
    custom_like_ops: customLikeOps,
    external_domains: externalDomains,
    onnx_domain_analysis_status: domainAnalysis?.status || (format === "onnx" ? "not_assessed" : "not_applicable"),
    external_data_tensor_count: externalDataTensorCount,
    external_data_verified_tensor_count: externalDataVerifiedCount,
    external_data_verification_failed_count: externalDataVerificationFailedCount,
    external_data_incomplete_tensor_count: externalDataIncompleteCount,
    external_data_integrity_status: externalDataTensorCount === 0 || externalDataIncompleteCount === 0
      ? "complete" : externalDataVerificationFailedCount > 0 ? "verification_failed" : "incomplete",
    external_data_summary: externalDataSummary,
    ...tfliteBufferPosture,
    summary: [
      schemaOrOpset,
      customLikeOps.length ? `custom-like ${customLikeOps.map((item) => `${item.name}:${item.count}`).join(" / ")}` : "no custom-like op family",
      externalDomainSummary,
      externalDataSummary,
      tfliteBufferPosture.tflite_buffer_summary,
    ].join("; "),
  };
}

function formatArtifactSchema(analysis, format, onnxOpsets) {
  if (format === "onnx") return `IR ${analysis.onnx_ir_version || "unknown"} / opset ${onnxOpsets || "unknown"}`;
  if (format === "tflite") return `TFLite schema ${analysis.version || "unknown"}`;
  if (format === "gguf") return `GGUF v${analysis.gguf?.version || "unknown"}`;
  if (format === "safetensors") return "SafeTensors format (unversioned)";
  if (format === "coreml") return `Core ML specification ${analysis.coreml?.specification_version || "unknown"}`;
  if (format === "executorch") return `ExecuTorch ${String(analysis.executorch_container || "artifact").toUpperCase()} schema ${analysis.version ?? "unknown"}`;
  return format === "unknown" ? "Unknown artifact format" : `${format} format; version not declared`;
}

export function formatOpsets(opsets = []) {
  return (opsets || [])
    .map((opset) => `${opset.domain || "ai.onnx"}:${opset.version || "unknown"}`)
    .join(" / ");
}

function collectTfliteTensorBufferPosture(analysis = {}, fileSizeBytes = 0, format = "unknown") {
  const notApplicable = {
    tflite_constant_tensor_count: 0,
    tflite_unique_constant_regions: 0,
    tflite_constant_buffer_bytes: 0,
    tflite_largest_constant_buffer_bytes: 0,
    tflite_duplicate_constant_regions: 0,
    tflite_out_of_bounds_tensor_buffers: 0,
    tflite_buffer_summary: "TFLite tensor buffer posture not applicable",
  };
  if (format !== "tflite") return notApplicable;

  const constantTensors = (artifactIrValues(analysis) || []).filter((tensor) => Number(tensor.buffer_data_length || 0) > 0);
  const regionCounts = new Map();
  let outOfBounds = 0;
  for (const tensor of constantTensors) {
    const offset = Number(tensor.buffer_data_offset || 0);
    const length = Number(tensor.buffer_data_length || 0);
    const key = `${offset}:${length}`;
    regionCounts.set(key, (regionCounts.get(key) || 0) + 1);
    if (fileSizeBytes && offset + length > fileSizeBytes) outOfBounds += 1;
  }

  let uniqueBytes = 0;
  let largest = 0;
  let duplicateRegions = 0;
  for (const [key, count] of regionCounts.entries()) {
    const length = Number(key.split(":")[1] || 0);
    uniqueBytes += length;
    largest = Math.max(largest, length);
    if (count > 1) duplicateRegions += count - 1;
  }

  const bounds = outOfBounds
    ? `${outOfBounds} tensor constant buffer reference(s) exceed artifact byte length`
    : "all tensor constant buffers fit within artifact bytes";
  const duplicates = duplicateRegions
    ? `${duplicateRegions} duplicate/shared non-empty tensor buffer region reference(s)`
    : "no duplicate non-empty tensor buffer regions";
  return {
    tflite_constant_tensor_count: constantTensors.length,
    tflite_unique_constant_regions: regionCounts.size,
    tflite_constant_buffer_bytes: uniqueBytes,
    tflite_largest_constant_buffer_bytes: largest,
    tflite_duplicate_constant_regions: duplicateRegions,
    tflite_out_of_bounds_tensor_buffers: outOfBounds,
    tflite_buffer_summary: `${constantTensors.length} tensor constant buffer(s), ${regionCounts.size} unique region(s), ${bounds}, ${duplicates}`,
  };
}
