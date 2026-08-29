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
    const ptdCount = files.filter((item) => {
      const bytes = item?.bytes;
      return bytes instanceof Uint8Array && bytes.byteLength >= 8
        && bytes[4] === 0x46 && bytes[5] === 0x54 && bytes[6] === 0x30 && bytes[7] === 0x31;
    }).length;
    const selectedBuildCount = files.filter((item) => String(item?.path || item?.name || "").replaceAll("\\", "/").split("/").at(-1)?.toLowerCase() === "deepbom.executorch-build.json").length;
    const selectedBuild = evidence?.selected_build_binding?.selected_build_input || null;
    const external = evidence?.external_tensor_data || null;
    status.textContent = files.length
      ? `${formatNumber(ptdCount)} PTD + ${formatNumber(selectedBuildCount)} build attestation / ${formatBytes(totalBytes)} hashed${evidence ? ` / ${formatNumber(external?.resolved_name_count || 0)}/${formatNumber(external?.required_name_count || 0)} external names bound / build ${selectedBuild ? "bound" : "unbound"}` : ""}`
      : external?.required_name_count
        ? `0/${formatNumber(external.required_name_count)} external tensor names bound / selected build ${selectedBuild ? "bound" : "unbound"}`
        : evidence ? `No external PTD data referenced / selected build ${selectedBuild ? "bound" : "unbound"}` : "No PTD or selected-build sidecars selected";
    return;
  }
  status.textContent = files.length
    ? `${formatNumber(files.length)} file(s) / ${formatBytes(totalBytes)} hashed${evidence ? ` / ${formatNumber(evidence.verified_payload_count || 0)}/${formatNumber(evidence.tensor_count || 0)} range(s) verified` : ""}`
    : evidence?.tensor_count
      ? `0/${formatNumber(evidence.tensor_count)} range(s) verified`
      : evidence ? "No external data referenced" : "No sidecars selected";
}

export const renderOnnxExternalDataStatus = renderExternalDataStatus;
