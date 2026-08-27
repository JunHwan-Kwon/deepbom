import { formatBytes, formatNumber } from "./format.js";

export function renderExternalDataStatus({
  control,
  status,
  format = "",
  files = [],
  evidence = null,
} = {}) {
  if (!control || !status) return;
  const normalized = String(format).toLowerCase();
  const supported = normalized === "onnx" || normalized === "executorch";
  control.hidden = !supported;
  if (!supported) return;
  const totalBytes = files.reduce((total, item) => total + Number(item.bytes?.byteLength || 0), 0);
  if (normalized === "executorch") {
    status.textContent = files.length
      ? `${formatNumber(files.length)} PTD file(s) / ${formatBytes(totalBytes)} hashed${evidence ? ` / ${formatNumber(evidence.resolved_name_count || 0)}/${formatNumber(evidence.required_name_count || 0)} names bound` : ""}`
      : evidence?.required_name_count
        ? `0/${formatNumber(evidence.required_name_count)} external tensor names bound`
        : evidence ? "No external PTD data referenced" : "No PTD sidecars selected";
    return;
  }
  status.textContent = files.length
    ? `${formatNumber(files.length)} file(s) / ${formatBytes(totalBytes)} hashed${evidence ? ` / ${formatNumber(evidence.verified_payload_count || 0)}/${formatNumber(evidence.tensor_count || 0)} range(s) verified` : ""}`
    : evidence?.tensor_count
      ? `0/${formatNumber(evidence.tensor_count)} range(s) verified`
      : evidence ? "No external data referenced" : "No sidecars selected";
}

export const renderOnnxExternalDataStatus = renderExternalDataStatus;
