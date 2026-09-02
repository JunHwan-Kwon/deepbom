export const APPLICABILITY_SCHEMA = "deepbom.evidence_applicability.v1";

export const APPLICABILITY_STATUS = Object.freeze({
  APPLICABLE: "applicable",
  NOT_APPLICABLE: "not_applicable",
  NOT_ASSESSABLE: "not_assessable",
  NOT_ASSESSED_YET: "not_assessed_yet",
});

const TAB_LABELS = Object.freeze({
  overview: "Overview",
  xnnpack: "XNNPACK",
  accelerator: "Execution Placement",
  quant: "Numerical Contracts",
  "quant-labs": "Quantization Arithmetic",
  llm: "On-device LLM",
  roofline: "Memory & Cost",
  stage: "Architecture",
});

function record(status, reasonCode, reasonText, requiredEvidence = null) {
  return Object.freeze({
    schema: APPLICABILITY_SCHEMA,
    applicability_status: status,
    reason_code: reasonCode,
    reason_text: reasonText,
    required_evidence: requiredEvidence,
  });
}

function available(reasonText = "Evidence is available for this artifact class.") {
  return record(APPLICABILITY_STATUS.APPLICABLE, "ARTIFACT_CLASS_SUPPORTED", reasonText);
}

function notApplicable(code, text) {
  return record(APPLICABILITY_STATUS.NOT_APPLICABLE, code, text);
}

export function auditTabApplicability(format, analysis = null) {
  const id = String(format || analysis?.format || "").toLowerCase();
  const graphRows = Array.isArray(analysis?.ops) ? analysis.ops.length : 0;
  const serializedLlm = analysis?.on_device_llm?.serialized_graph;
  const hasSerializedLlmEvidence = ["tflite", "onnx"].includes(id) && Boolean(serializedLlm) && (
    Number(serializedLlm.explicit_operator_count || 0) > 0
    || Number(serializedLlm.external_state_candidate_count || 0) > 0
    || serializedLlm.transformer_motif_candidate === true
  );
  const llmContainer = ["gguf", "safetensors"].includes(id);
  const graphSerialized = ["tflite", "onnx"].includes(id)
    || id === "coreml" && graphRows > 0
    || id === "executorch" && analysis?.executorch_container === "pte" && graphRows > 0;
  const noDag = `${id.toUpperCase()} does not serialize the executable operator DAG required by this view.`;
  const rows = {
    overview: available(),
    quant: available("Serialized numerical or storage contracts can be inspected for this artifact class."),
    accelerator: ["tflite", "onnx", "coreml", "executorch", "gguf", "safetensors"].includes(id)
      ? available("Placement evidence and its explicit runtime boundary can be inspected.")
      : notApplicable("PLACEMENT_ADAPTER_NOT_DEFINED", "No placement adapter is defined for this artifact class."),
    xnnpack: id === "tflite"
      ? available("Source-pinned XNNPACK eligibility and conditional partitions are available.")
      : notApplicable("XNNPACK_REQUIRES_TFLITE_GRAPH", "XNNPACK source-eligibility analysis is defined for the serialized TFLite graph."),
    "quant-labs": ["tflite", "onnx"].includes(id)
      ? available("Serialized Q/DQ or integer arithmetic contracts are available.")
      : notApplicable("QUANT_ARITHMETIC_GRAPH_NOT_SERIALIZED", `${noDag} Storage encoding remains available under Numerical Contracts.`),
    llm: llmContainer || hasSerializedLlmEvidence
      ? available("An architecture or serialized-graph LLM contract was identified.")
      : notApplicable("LLM_CONTRACT_NOT_IDENTIFIED", "No source-bound LLM architecture or serialized transformer contract was identified."),
    roofline: ["tflite", "onnx"].includes(id)
      ? available("Static memory and cost posture is available under the stated target assumptions.")
      : notApplicable("EXECUTABLE_COST_GRAPH_NOT_SERIALIZED", `${noDag} Runtime memory requirements may still be imported separately.`),
    stage: graphSerialized
      ? available("Serialized graph scopes and architecture stages are available.")
      : notApplicable("EXECUTABLE_GRAPH_NOT_SERIALIZED", llmContainer
        ? `${noDag} Use the On-device LLM lens for source-bound architecture projection.`
        : noDag),
  };
  return Object.freeze(rows);
}

export function applicabilityLabel(tabId) {
  return TAB_LABELS[tabId] || String(tabId || "Evidence");
}
