import {
  contractDynamicDimSummary,
  contractHasDynamicDims,
  modelQuantizationStatus,
  predictedPartitionBoundaryInventory,
} from "./analysis.js";
import { tensorShapeText } from "./format.js";
import { collectArtifactIntegrity } from "./report-integrity.js";

const DETAIL_POINTER = "engineering_evidence.json#/evidence/static_analysis";

function text(value) {
  return String(value ?? "").trim();
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function property(name, value) {
  if (value === null || value === undefined || value === "") return null;
  return { name, value: typeof value === "string" ? value : String(value) };
}

function compact(rows) {
  return rows.filter(Boolean);
}

function predictedNonDelegatedOpCounts(analysis) {
  if (String(analysis?.format || "tflite").toLowerCase() !== "tflite") return null;
  const counts = {};
  for (const op of analysis?.ops || []) {
    if (Number(op.xnnpack_chain_id) >= 0) continue;
    const name = text(op.name) || "UNKNOWN";
    counts[name] = (counts[name] || 0) + 1;
  }
  return counts;
}

function inputContractSummary(analysis) {
  const rows = analysis?.input_contracts || [];
  return {
    count: rows.length,
    derived_layout_count: rows.filter((row) => row.layout_evidence_class === "DERIVED").length,
    unassessed_layout_count: rows.filter((row) => row.layout_evidence_class === "NOT_ASSESSABLE").length,
  };
}

export function buildMlBomCompatibilityProjection(analysis, {
  hash = "",
  fileSizeBytes = 0,
  target = {},
  targetId = "",
  serialNumber = "",
  runtimeAssignmentEvidence = null,
} = {}) {
  const format = text(analysis?.format || "tflite").toLowerCase();
  const quant = modelQuantizationStatus(analysis);
  const integrity = collectArtifactIntegrity(analysis, { sha256: hash }, fileSizeBytes);
  const boundary = predictedPartitionBoundaryInventory(analysis);
  const inputSummary = inputContractSummary(analysis);
  const runtimeObserved = Boolean(runtimeAssignmentEvidence);
  const quantizedRatio = finite(quant.quantized_compute_mac_percent);
  const nonDelegated = predictedNonDelegatedOpCounts(analysis);
  const validSerialNumber = /^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(serialNumber);

  const componentProperties = compact([
    property("deepbom:compatibility:profile", "deepbom.compact_mlbom_compatibility.v2"),
    property("deepbom:compatibility:detailLocation", DETAIL_POINTER),
    property("mlbom:model:format", format),
    property("mlbom:model:schemaOrOpset", integrity.schema_or_opset),
    property("mlbom:model:fileSizeBytes", finite(fileSizeBytes)),
    property("mlbom:model:operatorCount", finite(analysis?.operator_count)),
    property("mlbom:model:tensorCount", finite(analysis?.tensor_count)),
    property("mlbom:model:totalMacs", finite(analysis?.total_macs)),
    property("mlbom:model:macAssessmentStatus", analysis?.mac_assessment?.status || (format === "tflite" ? "assessed" : "not_assessed")),
    property("mlbom:model:macAssessedComputeOps", finite(analysis?.mac_assessment?.assessed_compute_ops)),
    property("mlbom:model:macComputeOps", finite(analysis?.mac_assessment?.compute_ops ?? quant.compute_ops)),
    property("mlbom:model:quantizationClassification", quant.classification),
    property("mlbom:model:fullIntegerQuantized", typeof quant.full_integer === "boolean" ? quant.full_integer : null),
    property("mlbom:model:quantizedComputeMacRatio", quantizedRatio),
    property("mlbom:model:quantizedComputeMacAssessment", quantizedRatio == null ? "not_assessed_mac_coverage_incomplete" : "assessed"),
    property("mlbom:model:quantizedComputeOperators", finite(quant.quantized_compute_ops)),
    property("mlbom:model:computeOperators", finite(quant.compute_ops)),
    property("mlbom:model:quantizedComputeMacs", finite(quant.quantized_compute_macs)),
    property("mlbom:model:computeMacs", finite(quant.compute_macs)),
    property("mlbom:target:id", target.id || targetId),
    property("mlbom:target:label", target.label),
    property("mlbom:target:hardwareEvidenceClass", target.hardware_spec?.evidence_class || (target.id || targetId ? "HEURISTIC_PROFILE" : null)),
    property("mlbom:target:performanceEvidenceClass", target.performance_model_evidence_class || (target.id || targetId ? "HEURISTIC" : null)),
    property("deepbom:model:inputTensorContractCount", inputSummary.count),
    property("deepbom:model:inputLayoutDerivedCount", inputSummary.derived_layout_count),
    property("deepbom:model:inputLayoutUnassessedCount", inputSummary.unassessed_layout_count),
    property("deepbom:model:predictedPartitionBoundaryEdges", boundary?.edge_count),
    property("deepbom:model:predictedPartitionBoundaryLogicalBytes", boundary?.summed_edge_payload_bytes),
    property("deepbom:model:runtimeAssignmentStatus", runtimeObserved ? "imported_observed_runtime_evidence" : "not_observed"),
  ]);

  const documentProperties = compact([
    property("ondevice:generatedBy", "DEEPBOM"),
    property("ondevice:generatedLocally", "true"),
    property("ondevice:inputContract", (analysis?.inputs || []).map((tensor) => `${tensor.name}:${tensor.dtype}${tensorShapeText(tensor)}`).join(" / ")),
    property("ondevice:outputContract", (analysis?.outputs || []).map((tensor) => `${tensor.name}:${tensor.dtype}${tensorShapeText(tensor)}`).join(" / ")),
    property("ondevice:inputHasDynamicDims", contractHasDynamicDims(analysis?.inputs) ? "true" : "false"),
    property("ondevice:outputHasDynamicDims", contractHasDynamicDims(analysis?.outputs) ? "true" : "false"),
    property("ondevice:dynamicInputSignatures", contractDynamicDimSummary(analysis?.inputs)),
    property("ondevice:dynamicOutputSignatures", contractDynamicDimSummary(analysis?.outputs)),
    property("ondevice:predictedNonDelegatedOps", nonDelegated == null ? null : JSON.stringify(nonDelegated)),
    property("ondevice:executionProviderAssignment", format === "onnx" ? "not_assessable_without_runtime_evidence" : null),
    property("ondevice:runtimeAssignmentEvidence", runtimeObserved ? "imported" : "not_observed"),
    property("ondevice:detailEvidencePointer", DETAIL_POINTER),
    property("ondevice:privacy:modelUpload", "false"),
    property("ondevice:privacy:reportUpload", "false"),
  ]);

  return {
    serialNumber: validSerialNumber ? serialNumber : null,
    componentProperties,
    documentProperties,
  };
}
