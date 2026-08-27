import { insightCard } from "./dom.js";
import { formatBytes, formatNumber, shapeText, shortError } from "./format.js";
import { quantParams } from "./preprocess.js";

export function preprocessPanelCards(spec, {
  selectedImageFile = null,
  preparedInput = null,
  channelOrderValue = "rgb",
} = {}) {
  if (!spec.ok) {
    return [
      insightCard("Input contract", "Not image-like", spec.reason, "risk"),
      insightCard("Privacy posture", "Still zero upload", "The model can still be statically audited locally.", "good"),
    ];
  }

  const quant = quantParams(spec.tensor);
  const cards = [
    insightCard("Target Tensor", `${spec.width}x${spec.height}x${spec.channels}`, `${spec.tensor.name || `T${spec.tensor.index}`} / ${spec.dtype} / shape ${shapeText(spec.shape)}`, "good"),
    insightCard("Tensor Pack", spec.dtype, `${spec.dtype === "FLOAT32" ? "normalize" : `scale=${quant.scale} / zero_point=${quant.zeroPoint}`} / ${channelOrderValue.toUpperCase()}`, "good"),
    insightCard("Selected Image", selectedImageFile?.name || "None", selectedImageFile ? `${formatBytes(selectedImageFile.size)} / local file` : "Choose an image to produce a real input tensor", selectedImageFile ? "good" : "neutral"),
    insightCard("Privacy Posture", "Local only", "No model upload, no image upload, no stored user data.", "good"),
  ];

  if (preparedInput) {
    cards.push(
      insightCard("Prepared Tensor", `${formatNumber(preparedInput.elements)} values`, `${preparedInput.dtype} / ${formatBytes(preparedInput.bytes)} / shape ${shapeText(preparedInput.shape)}`, "good"),
      insightCard("Local Timing", `${preparedInput.totalMs.toFixed(2)} ms`, `decode ${preparedInput.decodeMs.toFixed(2)} / resize ${preparedInput.resizeMs.toFixed(2)} / pack ${preparedInput.packMs.toFixed(2)}`, "good"),
    );
  }

  return cards;
}

export function preprocessErrorCard(error) {
  return insightCard("Preprocess failed", shortError(error), "Check the input shape, dtype, and image file.", "risk");
}
