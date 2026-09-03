import { artifactIrOperators } from "./artifact-ir-selectors.js";
const EVIDENCE_LABELS = Object.freeze({
  operatorCount: "Graph operators",
  tensorCount: "Serialized tensors",
  totalMacs: "Assessed MACs",
  constantBytes: "Constant payload",
  peakLiveBytes: "Peak live payload",
  arenaBytes: "ArenaPlanner projection",
  quantizeOps: "QUANTIZE ops",
  dequantizeOps: "DEQUANTIZE ops",
  xnnpackSourceSupportedOps: "Source-supported XNNPACK ops",
  predictedBreakOps: "Predicted break ops",
  storedScalarElements: "Stored scalar elements",
  peakLiveAtOp: "Peak live at op",
  payloadBytes: "Tensor payload",
  decodedValueCount: "Decoded values",
  nonfiniteValueCount: "Non-finite values",
  inputCount: "External inputs",
  outputCount: "External outputs",
  tensorRtConditionallyEligibleOps: "TensorRT parser-observed eligible ops",
  tensorRtDefiniteExclusionOps: "TensorRT parser-observed exclusions",
  tensorRtUnresolvedOps: "TensorRT parser-unresolved ops",
  llmMatrixMultiplyOps: "LLM graph MatMul ops",
  llmSoftmaxOps: "LLM graph Softmax ops",
  llmNormalizationOps: "LLM graph normalization ops",
  llmExternalStateCandidates: "Serialized state-name candidates",
});

function numeric(value) {
  if (value == null) return null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

export function collectPublicSampleObservedEvidence(format, analysis = {}) {
  const normalized = String(format || analysis.format || "").toLowerCase();
  const container = ["gguf", "safetensors"].includes(normalized);
  const tensorRtStates = analysis.tensorrt_static_preflight?.projection?.state_counts || {};
  const llmGraph = analysis.on_device_llm?.serialized_graph || {};
  return {
    operatorCount: container ? null : numeric(analysis.operator_count),
    tensorCount: numeric(analysis.tensor_count),
    totalMacs: container ? null : numeric(analysis.total_macs),
    constantBytes: numeric(analysis.size_breakdown?.constant_bytes),
    peakLiveBytes: numeric(analysis.tensor_liveness?.peak_bytes),
    arenaBytes: numeric(analysis.tensor_arena_plan?.combined_arena_bytes),
    quantizeOps: numeric(analysis.quantization_status?.quantize_ops),
    dequantizeOps: numeric(analysis.quantization_status?.dequantize_ops),
    xnnpackSourceSupportedOps: (artifactIrOperators(analysis) || []).filter((op) => op.xnnpack_supported).length,
    predictedBreakOps: (artifactIrOperators(analysis) || []).filter((op) => op.xnnpack_chain_break).length,
    storedScalarElements: numeric(analysis.size_breakdown?.stored_scalar_elements),
    peakLiveAtOp: numeric(analysis.tensor_liveness?.peak_at_op),
    payloadBytes: numeric(analysis.gguf?.payload_byte_length ?? analysis.safetensors?.payload_byte_length),
    decodedValueCount: numeric(analysis.tensor_numerical_integrity?.decoded_value_count),
    nonfiniteValueCount: numeric(analysis.tensor_numerical_integrity?.nonfinite_value_count),
    inputCount: Array.isArray(analysis.inputs) ? analysis.inputs.length : null,
    outputCount: Array.isArray(analysis.outputs) ? analysis.outputs.length : null,
    tensorRtConditionallyEligibleOps: numeric(tensorRtStates.CONDITIONALLY_ELIGIBLE),
    tensorRtDefiniteExclusionOps: numeric(tensorRtStates.DEFINITE_EXCLUSION),
    tensorRtUnresolvedOps: numeric(tensorRtStates.UNRESOLVED),
    llmMatrixMultiplyOps: numeric(llmGraph.primitive_counts?.matrix_multiply),
    llmSoftmaxOps: numeric(llmGraph.primitive_counts?.softmax),
    llmNormalizationOps: numeric(llmGraph.primitive_counts?.normalization),
    llmExternalStateCandidates: numeric(llmGraph.external_state_candidate_count),
  };
}

export function comparePublicSampleEvidence(sample, analysis, {
  artifactSha256 = analysis?.model_sha256 || "",
  artifactByteLength = analysis?.file_size_bytes ?? analysis?.file_size ?? null,
} = {}) {
  if (!sample || !analysis) return null;
  const observed = collectPublicSampleObservedEvidence(sample.format, analysis);
  const rows = [
    comparisonRow("Artifact format", sample.format, String(analysis.format || "").toLowerCase()),
    comparisonRow("Artifact SHA-256", sample.sha256, String(artifactSha256 || "").toLowerCase(), "sha256"),
    comparisonRow("Artifact bytes", sample.byteLength, numeric(artifactByteLength), "number"),
  ];
  for (const [key, expected] of Object.entries(sample.expectedEvidence || {})) {
    if (typeof expected !== "number" && expected !== null) continue;
    if (!Object.hasOwn(observed, key)) continue;
    rows.push(comparisonRow(EVIDENCE_LABELS[key] || key, expected, observed[key], "number"));
  }
  const passed = rows.filter((row) => row.status === "pass").length;
  return Object.freeze({
    schema: "deepbom.public_sample_verification.v1",
    sampleId: sample.id,
    artifactSha256: String(artifactSha256 || "").toLowerCase(),
    checks: Object.freeze(rows),
    passed,
    failed: rows.length - passed,
    status: passed === rows.length ? "pass" : "fail",
  });
}

function comparisonRow(label, expected, observed, kind = "value") {
  return Object.freeze({
    label,
    expected,
    observed,
    kind,
    status: Object.is(expected, observed) ? "pass" : "fail",
  });
}
