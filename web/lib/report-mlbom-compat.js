import { artifactIrOperators } from "./artifact-ir-selectors.js";
import {
  contractDynamicDimSummary,
  contractHasDynamicDims,
  modelQuantizationStatus,
  predictedPartitionBoundaryInventory,
} from "./analysis.js";
import { tensorShapeText } from "./format.js";
import { deriveMacCoverage, deriveQuantizedComputeAssessment } from "./mac-coverage.js";
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
  for (const op of artifactIrOperators(analysis) || []) {
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
  const artifactSet = analysis?.artifact_set;
  const accelerator = analysis?.accelerator_profile_binding;
  const conversion = analysis?.conversion_receipt;
  const macCoverage = deriveMacCoverage(analysis, quant);
  const quantizedAssessment = deriveQuantizedComputeAssessment(analysis, quant, macCoverage);
  const computeOperators = finite(quant.compute_ops ?? macCoverage.compute_ops);
  const quantizedComputeOperators = finite(quant.quantized_compute_ops);

  const componentProperties = compact([
    property("deepbom:compatibility:profile", "deepbom.compact_mlbom_compatibility.v2"),
    property("deepbom:compatibility:detailLocation", DETAIL_POINTER),
    property("mlbom:model:format", format),
    property("mlbom:model:schemaOrOpset", integrity.schema_or_opset),
    property("mlbom:model:fileSizeBytes", finite(fileSizeBytes)),
    property("mlbom:model:operatorCount", finite(analysis?.operator_count)),
    property("mlbom:model:tensorCount", finite(analysis?.tensor_count)),
    property("mlbom:model:totalMacs", finite(analysis?.total_macs)),
    property("mlbom:model:macAssessmentStatus", macCoverage.status),
    property("mlbom:model:macAssessmentReason", macCoverage.reason),
    property("mlbom:model:macAssessedComputeOps", finite(macCoverage.assessed_compute_ops)),
    property("mlbom:model:macComputeOps", finite(macCoverage.compute_ops)),
    property("deepbom:model:serializedContractStatus", analysis?.onnx_contract_conflict?.status),
    property("deepbom:model:contractConflictCapsuleSha256", analysis?.onnx_contract_conflict?.capsule_sha256),
    property("deepbom:model:contractConflictRootCount", finite(analysis?.onnx_contract_conflict?.summary?.unconditional_root_conflict_count)),
    property("deepbom:model:contractConflictConditionalVariantCount", finite(analysis?.onnx_contract_conflict?.summary?.condition_bound_invalid_variant_count)),
    property("deepbom:model:contractConflictBlockedMacRows", finite(analysis?.onnx_contract_conflict?.summary?.blocked_mac_row_count)),
    property("mlbom:model:quantizationClassification", quant.classification),
    property("mlbom:model:fullIntegerQuantized", typeof quant.full_integer === "boolean" ? quant.full_integer : null),
    property("mlbom:model:quantizedComputeMacRatio", quantizedRatio),
    property("mlbom:model:quantizedComputeMacAssessment", quantizedAssessment.mac_status),
    property("mlbom:model:quantizedComputeMacAssessmentReason", quantizedAssessment.mac_reason),
    property("mlbom:model:quantizedComputeOperatorAssessment", quantizedAssessment.operator_status),
    property("mlbom:model:quantizedComputeOperatorAssessmentReason", quantizedAssessment.operator_reason),
    property("mlbom:model:quantizedComputeOperators", quantizedComputeOperators),
    property("mlbom:model:computeOperators", computeOperators),
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
    property("deepbom:model:artifactSetSchema", artifactSet?.schema),
    property("deepbom:model:artifactSetSha256", artifactSet?.artifact_set_sha256),
    property("deepbom:model:artifactSourceKind", artifactSet?.source?.kind),
    property("deepbom:model:artifactSourceImmutability", artifactSet?.source?.immutability?.kind),
    property("deepbom:model:conversionReceiptStatus", conversion?.status),
    property("deepbom:model:conversionReceiptSha256", conversion?.receipt_sha256),
    property("deepbom:model:conversionReceiptBindingSha256", conversion?.binding_sha256),
    property("deepbom:model:conversionSourceArtifactCount", conversion?.receipt?.source_artifacts?.length),
    property("deepbom:model:conversionConverterName", conversion?.receipt?.converter?.name),
    property("deepbom:model:conversionConverterVersion", conversion?.receipt?.converter?.version),
    property("deepbom:model:conversionEnvironmentManifestSha256", conversion?.receipt?.converter?.environment?.manifest_sha256),
    property("deepbom:model:conversionEvidenceClass", conversion?.evidence_class),
    property("deepbom:model:conversionOutputBindingEvidenceClass", conversion?.output_binding_evidence_class),
    property("deepbom:model:acceleratorProfileBindingSha256", accelerator?.binding_sha256),
    property("deepbom:model:acceleratorProfileSha256", accelerator?.profile_sha256),
    property("deepbom:model:acceleratorDeviceName", accelerator?.selected_device?.name),
    property("deepbom:model:acceleratorComputeCapability", accelerator?.selected_device?.compute_capability),
    property("deepbom:model:acceleratorPhysicalVramBytes", accelerator?.selected_device?.memory_total_bytes?.decimal),
    property("deepbom:model:acceleratorSelectedBuildStatus", accelerator?.selected_build?.status),
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
    property("deepbom:artifact:sourceLocator", artifactSet?.source?.canonical_locator),
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
