export function updateFormatSpecificAuditLabels({
  modelFormat,
  analysis = null,
  auditTabs,
  activeTab,
  selectTab,
}) {
  const onnx = modelFormat === "onnx";
  const coreMl = modelFormat === "coreml";
  const execuTorch = modelFormat === "executorch";
  const llmContainer = ["gguf", "safetensors"].includes(modelFormat);
  const serializedLlm = analysis?.on_device_llm?.serialized_graph;
  const hasSerializedLlmEvidence = ["tflite", "onnx"].includes(modelFormat) && Boolean(serializedLlm) && (
    Number(serializedLlm.explicit_operator_count || 0) > 0
    || Number(serializedLlm.external_state_candidate_count || 0) > 0
    || serializedLlm.transformer_motif_candidate === true
  );
  const available = new Set(modelFormat === "tflite"
    ? ["overview", "xnnpack", "accelerator", "quant", "quant-labs", "roofline", "stage"]
    : onnx ? ["overview", "accelerator", "quant", "quant-labs", "roofline", "stage"]
      : coreMl && analysis?.ops?.length ? ["overview", "accelerator", "quant", "stage"]
        : execuTorch && analysis?.executorch_container === "pte" ? ["overview", "accelerator", "stage"]
          : execuTorch ? ["overview", "quant"]
        : llmContainer ? ["overview", "accelerator", "llm", "quant"] : ["overview", "quant"]);
  if (hasSerializedLlmEvidence) available.add("llm");
  for (const tab of auditTabs) tab.hidden = !available.has(tab.dataset.auditTab);
  const mobileAuditView = auditTabs[0]?.closest(".audit-workbench")?.querySelector("#mobileAuditView");
  for (const option of mobileAuditView?.options || []) {
    const applicable = available.has(option.value);
    option.hidden = !applicable;
    option.disabled = !applicable;
  }
  if (!available.has(activeTab())) selectTab("overview");

  const overview = auditTabs.find((tab) => tab.dataset.auditTab === "overview");
  const quant = auditTabs.find((tab) => tab.dataset.auditTab === "quant");
  const roofline = auditTabs.find((tab) => tab.dataset.auditTab === "roofline");
  if (overview?.querySelector("em")) overview.querySelector("em").textContent = onnx ? "Coverage" : "Triage";
  if (quant) {
    const span = quant.querySelector("span");
    const strong = quant.querySelector("strong");
    const em = quant.querySelector("em");
    const labels = modelFormat === "gguf"
      ? ["Storage", "GGUF Encoding", "Blocks"]
      : modelFormat === "safetensors"
        ? ["Storage", "Tensor Dtypes", "Payload"]
        : coreMl ? ["Numerics", "Core ML", "Evidence"]
          : execuTorch ? ["Storage", "ExecuTorch", "Tensors"] : ["Numerics", "Quant", "Summary"];
    if (span) span.textContent = labels[0];
    if (strong) strong.textContent = labels[1];
    if (em) em.textContent = labels[2];
    const mobileOption = mobileAuditView?.querySelector('option[value="quant"]');
    if (mobileOption) mobileOption.textContent = modelFormat === "gguf"
      ? "GGUF encoding and blocks"
      : modelFormat === "safetensors"
        ? "Tensor dtypes and payload"
        : coreMl ? "Core ML numerics" : execuTorch ? "ExecuTorch tensor storage" : "Quantization summary";
  }
  if (roofline) {
    const span = roofline.querySelector("span");
    const strong = roofline.querySelector("strong");
    const em = roofline.querySelector("em");
    if (span) span.textContent = onnx ? "Static" : "Target";
    if (strong) strong.textContent = onnx ? "Intensity" : "Roofline";
    if (em) em.textContent = onnx ? "Posture" : "Bound";
  }
  const accessibleLabels = {
    overview: `${modelFormat.toUpperCase()} overview and static evidence coverage`,
    xnnpack: "TFLite XNNPACK conditional eligibility and predicted partition segments",
    accelerator: `${modelFormat.toUpperCase()} source-backed accelerator eligibility, partition boundaries, payload exposure, and unresolved runtime conditions`,
    quant: modelFormat === "gguf"
      ? "GGUF encoding and storage blocks"
      : modelFormat === "safetensors"
        ? "SafeTensors dtype and payload distribution"
        : coreMl
          ? "Core ML numerical and serialized evidence"
          : execuTorch
            ? "ExecuTorch tensor shape, layout, storage span, and segment conservation"
          : `${modelFormat.toUpperCase()} quantization state and risk summary`,
    "quant-labs": onnx ? "ONNX Q/DQ and QOperator contracts" : "TFLite quantization arithmetic contracts",
    llm: "On-device LLM architecture, tokenizer, KV or recurrent state, MoE/SSM compute, runtime residency, and medical claim boundary",
    roofline: onnx ? "ONNX static intensity posture" : "TFLite static roofline and target bound posture",
    stage: `${modelFormat.toUpperCase()} stage topology and blocks`,
  };
  for (const tab of auditTabs) tab.setAttribute("aria-label", accessibleLabels[tab.dataset.auditTab] || tab.textContent.trim());
  return available;
}
