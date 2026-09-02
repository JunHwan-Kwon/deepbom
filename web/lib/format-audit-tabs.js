import { APPLICABILITY_STATUS, applicabilityLabel, auditTabApplicability } from "./evidence-applicability.js";

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
  const applicability = auditTabApplicability(modelFormat, analysis);
  const available = new Set(Object.entries(applicability)
    .filter(([, record]) => record.applicability_status === APPLICABILITY_STATUS.APPLICABLE)
    .map(([tabId]) => tabId));
  for (const tab of auditTabs) {
    const record = applicability[tab.dataset.auditTab];
    tab.hidden = false;
    tab.dataset.applicabilityStatus = record?.applicability_status || APPLICABILITY_STATUS.NOT_ASSESSABLE;
    tab.dataset.applicabilityReasonCode = record?.reason_code || "APPLICABILITY_NOT_RESOLVED";
    tab.dataset.applicabilityReason = record?.reason_text || "Applicability was not resolved.";
    tab.dataset.applicabilityRequired = record?.required_evidence || "";
    tab.classList.toggle("not-applicable", tab.dataset.applicabilityStatus !== APPLICABILITY_STATUS.APPLICABLE);
  }
  const mobileAuditView = auditTabs[0]?.closest(".audit-workbench")?.querySelector("#mobileAuditView");
  for (const option of mobileAuditView?.options || []) {
    const record = applicability[option.value];
    option.hidden = false;
    option.disabled = false;
    if (!option.dataset.baseLabel) option.dataset.baseLabel = option.textContent;
    option.textContent = record?.applicability_status !== APPLICABILITY_STATUS.APPLICABLE
      ? `${option.dataset.baseLabel} - ${applicabilityStatusLabel(record?.applicability_status)}`
      : option.dataset.baseLabel;
  }

  const overview = auditTabs.find((tab) => tab.dataset.auditTab === "overview");
  if (overview?.querySelector("em")) overview.querySelector("em").textContent = onnx ? "Coverage" : "Triage";
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
  for (const tab of auditTabs) {
    const record = applicability[tab.dataset.auditTab];
    const label = accessibleLabels[tab.dataset.auditTab] || applicabilityLabel(tab.dataset.auditTab);
    tab.setAttribute("aria-label", record?.applicability_status !== APPLICABILITY_STATUS.APPLICABLE
      ? `${label}. ${applicabilityStatusLabel(record?.applicability_status)}: ${record.reason_text}`
      : label);
    tab.title = record?.reason_text || label;
  }
  return available;
}

function applicabilityStatusLabel(status) {
  if (status === APPLICABILITY_STATUS.NOT_APPLICABLE) return "not applicable";
  if (status === APPLICABILITY_STATUS.NOT_ASSESSABLE) return "not assessable";
  if (status === APPLICABILITY_STATUS.NOT_ASSESSED_YET) return "not assessed yet";
  return "applicable";
}
