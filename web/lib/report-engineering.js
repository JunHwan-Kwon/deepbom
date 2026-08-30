import { modelQuantizationStatus, predictedPartitionBoundaryInventory, quantizationScopeSummary, quantStateLabel, xnnpackLabel } from "./analysis.js";
import { formatBytes, formatDrift, formatExactInteger, formatNumber, formatPercent, formatPercentRange, formatUs, padOp, score100, sumNumbers } from "./format.js";
import { FINDING_PRIORITIES, sortFindingsByPriority } from "./finding-contract.js";
import { ANALYZER_METADATA, RUNTIME_COMPATIBILITY_EVIDENCE_LABEL, TARGET_PROFILE_SOURCE_BASIS, scopeAndEvidenceBoundaryMarkdown } from "./report-metadata.js";
import { buildFindingsRegister } from "./report-findings.js";
import { collectArtifactIntegrity, reportBodyForFingerprint, REPORT_VERIFICATION_SENTINEL } from "./report-integrity.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { buildQuantizationContractChecks } from "./report-quantization-contracts.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";
import { quantResearchEvidenceMarkdown } from "./report-engineering-quant-research.js";
import {
  onnxShapeInferenceMarkdown,
  onnxSparseTensorContractMarkdown,
  onnxTensorDataTypeContractMarkdown,
  onnxTypeProtoContractMarkdown,
} from "./report-engineering-onnx.js";
import { benchmarkExternalDataBindingText, benchmarkInputContractText, benchmarkOutputContractText, benchmarkSampleLedgerMarkdown, calibrationValidationMarkdown, evidenceClassLegend, preprocessingConsequenceMarkdown, runtimeEnvironmentMarkdown } from "./report-sections.js";
import { code, markdownTable } from "./report-utils.js";
import { buildStaticAuditMarkdown } from "./markdown-report.js";
import { buildMetricCoverageManifest, createAnalysisFieldAccessTracker, decisionCoverageMarkdown, metricCoverageMarkdown } from "./metric-coverage.js";
import { buildModelAtGlance } from "./model-glance.js";
import { coreMlStaticResourceMarkdown, onDeviceLlmEvidenceMarkdown, serializedFormatEvidenceMarkdown } from "./report-engineering-serialized.js";
import { tfliteSubgraphInventoryMarkdown } from "./report-engineering-tflite-subgraphs.js";
import { tfliteAlternateDelegateCompatibilityMarkdown } from "./report-engineering-tflite-delegates.js";
import { analyzerContentVersion } from "./cyclonedx-identity.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { executionPlacementCoverageRows, executionPlacementMarkdown, executionPlacementRuntimeMetricResults } from "./report-execution-placement.js";
import { buildOnnxRuntimeShapeBinding } from "./onnx-runtime-shape-binding.js";
import { delegationRuleBasisText, formatRidge, intensityPosture, peakArenaReconciliation, simdAssumptionsForAnalysis, xnnpackBuildRequirementsSummary } from "./report-engineering-derivations.js";
import { buildChangeAnalysis } from "./report-change-analysis.js";
import { kernelSourceCandidatesMarkdown } from "./report-engineering-kernels.js";
import { dynamicShapeCostMarkdown } from "./report-engineering-dynamic-shapes.js";
import { delegationRepairMarkdown, deploymentDeltaMarkdown, deploymentFrontierMarkdown } from "./report-engineering-deployment.js";
import {
  inputContractEvidenceMarkdown,
  inputContractRiskSummary,
  inputLayoutDetermination,
  inputNumericalContractSummary,
  interfaceQuantizationLedgerMarkdown,
  sourcePreprocessingContractSummary,
} from "./report-engineering-interface.js";

export { buildChangeAnalysis };

const METRIC_COVERAGE_PLACEHOLDER = "<!-- deepbom-metric-coverage-placeholder -->";
const DECISION_COVERAGE_PLACEHOLDER = "<!-- deepbom-decision-coverage-placeholder -->";

function plural(count, singular, pluralForm = `${singular}s`) {
  const n = Number(count || 0);
  return `${formatNumber(n)} ${n === 1 ? singular : pluralForm}`;
}

function formatCompactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? "-");
  return new Intl.NumberFormat("en-US", { maximumSignificantDigits: 8 }).format(number);
}

function packingSetupUs(targetProfile = {}) {
  return targetProfile?.in_order ? 8 : 4;
}

function delegatedOpRatioText(analysis) {
  if (isOnnxAnalysis(analysis)) return "not applicable for ONNX; ORT execution-provider assignment is not modeled";
  const ops = analysis?.ops || [];
  if (!ops.length) return "not emitted";
  const delegated = ops.filter((op) => Number(op.xnnpack_chain_id) >= 0).length;
  return `${formatNumber(delegated)} / ${formatNumber(ops.length)} (${formatPercent(delegated / ops.length)})`;
}

function stageDelegatedMacText(stage) {
  return Number(stage?.macs || 0) > 0
    ? formatPercent(stage.delegated_mac_percent || 0)
    : "N/A, stage has zero modeled MACs";
}

function channelSetText(channels = []) {
  return channels.length ? `{${channels.join(", ")}}` : "-";
}

function compactCountItems(items = [], limit = 10) {
  const rows = (items || []).slice(0, limit).map((item) => `${item.name || "unknown"}:${formatNumber(item.count || 0)}`);
  if ((items || []).length > limit) rows.push(`+${formatNumber(items.length - limit)} more`);
  return rows.join(" / ") || "not emitted";
}

function isOnnxAnalysis(analysis) {
  return String(analysis?.format || "").toLowerCase() === "onnx";
}

function reportModelName(analysis, identity = {}) {
  const value = analysis?.filename || identity?.filename || "unnamed artifact";
  return String(value).replace(/[\u0000-\u001f\u007f]+/g, " ").replace(/\s+/g, " ").trim() || "unnamed artifact";
}

function assessedPercent(value) {
  return value == null ? "N/A; MAC coverage incomplete" : formatPercent(value);
}

function sectionsSuppressedText(analysis) {
  const suppressed = analysis?.onnx_sections_suppressed || [];
  return suppressed.length ? suppressed.join(" / ") : "none";
}

function runtimeAssumptionText(analysis) {
  return isOnnxAnalysis(analysis)
    ? "ONNX artifact static analysis; ONNX Runtime execution-provider assignment is not modeled in this report"
    : `TFLite/LiteRT-style CPU path with XNNPACK delegation rules from the rulepack; GPU and NNAPI rows are pinned-source candidates only when the protected delegate rulepack is present; ${xnnpackBuildRequirementsSummary(analysis)}; effective peak/bandwidth already absorb thread and frequency assumptions`;
}

function structuralViewOpNames(analysis) {
  return isOnnxAnalysis(analysis)
    ? new Set(["Flatten", "Reshape", "Shape", "Squeeze", "Unsqueeze"])
    : new Set(["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"]);
}

function structuralViewOps(analysis) {
  const names = structuralViewOpNames(analysis);
  return (analysis?.ops || []).filter((op) => names.has(String(op.name || "")));
}

function opNameList(ops = [], limit = 4) {
  if (!ops.length) return "none";
  const names = ops.slice(0, limit).map((op) => `#${padOp(op.index)} ${op.name}`);
  return `${names.join(" / ")}${ops.length > limit ? " / ..." : ""}`;
}

function predictedNonDelegatedOps(analysis) {
  if (isOnnxAnalysis(analysis)) return [];
  return (analysis?.ops || []).filter((op) => Number(op.xnnpack_chain_id) < 0);
}

function l1WatchCount(analysis) {
  if (analysis?.insights?.l1_watch_count != null) return Number(analysis.insights.l1_watch_count || 0);
  return (analysis?.ops || []).filter((op) => Number(op.row_working_set_ratio || 0) >= 0.9).length;
}

function maxRowWorkingSetRatio(analysis) {
  return (analysis?.ops || []).reduce((max, op) => Math.max(max, Number(op.row_working_set_ratio || 0)), 0);
}

function cacheRatioText(rowBytes, cacheBytes) {
  if (!Number.isFinite(Number(rowBytes)) || !(Number(cacheBytes) > 0)) return "N/A";
  const ratio = Number(rowBytes) / Number(cacheBytes);
  return `${ratio.toFixed(2)}x / ${ratio >= 0.9 ? ratio > 1 ? "exceeds" : "watch" : "ok"}`;
}

function modelAtGlanceMarkdown(analysis) {
  const onnx = isOnnxAnalysis(analysis);
  const glance = buildModelAtGlance(analysis);
  const cache = glance.memory.cache;
  const cacheText = [
    glance.memory.peakLiveBytes == null ? "peak live N/A" : `peak live ${formatBytes(glance.memory.peakLiveBytes)}`,
    glance.memory.arenaBytes == null ? "arena N/A" : `arena ${formatBytes(glance.memory.arenaBytes)}`,
    glance.memory.artifactPlusArenaBytes == null
      ? "artifact + arena N/A"
      : `artifact + arena ${formatBytes(glance.memory.artifactPlusArenaBytes)} (runtime code, delegate state, and scratch excluded)`,
    cache.l1.maxRatio == null ? "L1 N/A" : `max L1 ${cache.l1.maxRatio.toFixed(2)}x (${cache.l1.watchCount} watch)`,
    cache.l2.maxRatio == null ? `L2 watch N/A (${String(cache.l2.assessmentStatus || "capacity ownership unbound").replaceAll("_", " ")})` : `max L2 ${cache.l2.maxRatio.toFixed(2)}x (${cache.l2.watchCount} watch)`,
  ].join("; ");
  const boundaryPayload = glance.delegation.boundaryPayloadStatus === "partial"
    ? `${formatBytes(glance.delegation.boundaryPayloadBytes || 0)} assessed`
    : formatBytes(glance.delegation.boundaryPayloadBytes || 0);
  const boundarySetup = glance.delegation.boundaryOverhead?.highUs > 0
    ? `; cold-start partition-planning setup range ${formatUs(glance.delegation.boundaryOverhead.lowUs)}-${formatUs(glance.delegation.boundaryOverhead.highUs)}, midpoint ${formatUs(glance.delegation.boundaryOverhead.midpointUs)}`
    : "";
  const motifAttribution = glance.delegation.motifAttribution?.summary
    ? ` ${glance.delegation.motifAttribution.summary}`
    : "";
  const artifactShape = glance.artifact.totalMacs == null
    ? `MAC shape-dependent; ${formatNumber(glance.artifact.opCount)} ops / ${formatBytes(glance.artifact.fileSizeBytes)}`
    : `${formatNumber(glance.artifact.totalMacs)} MAC / ${formatNumber(glance.artifact.opCount)} ops / ${formatBytes(glance.artifact.fileSizeBytes)}`;
  const delegatedMacText = glance.delegation.delegatedMacRatio == null
    ? "MAC share not assessed until shape binding"
    : `${formatPercent(glance.delegation.delegatedMacRatio)} MAC`;
  const quantResearch = glance.quantization.researchCoverage;
  const rangeText = (lowUs, highUs) => Math.abs(Number(highUs || 0) - Number(lowUs || 0)) <= 1e-9
    ? formatUs(lowUs)
    : `${formatUs(lowUs)}-${formatUs(highUs)}`;
  return [
    "## Model At A Glance",
    markdownTable(["Question", "Answer", "Evidence"], [
      ["Artifact shape", artifactShape, onnx ? `DERIVED; MAC coverage ${glance.artifact.macAssessedComputeOps}/${glance.artifact.macComputeOps} compute ops` : glance.artifact.totalMacsEvidenceClass === "NOT_ASSESSED_DYNAMIC_SHAPE" ? "NOT_ASSESSED; bind non-batch dynamic axes" : glance.artifact.totalMacsEvidenceClass === "ASSUMPTION_BOUND_N_EQ_1" ? "ASSUMPTION_BOUND; exact emitted polynomial evaluation at serialized N=1" : "OBSERVED + DERIVED"],
      ...(!quantResearch ? [] : [[
        "Quant research artifact class",
        `${quantResearch.artifact_class_label}; class-supported ${formatNumber(quantResearch.class_supported_lab_count)}/${formatNumber(quantResearch.lab_count)}, artifact-applicable ${formatNumber(quantResearch.artifact_applicable_lab_count)}, assessed ${formatNumber(quantResearch.assessed_lab_count)}, partial ${formatNumber(quantResearch.partial_lab_count)}, not assessed ${formatNumber(quantResearch.not_assessed_lab_count)}`,
        `DERIVED; ${quantResearch.artifact_class_reason_code}. Not-applicable labs are excluded from defect-rate denominators.`,
      ]]),
      ["Quantization", `${glance.quantization.label}; ${formatNumber(glance.quantization.perChannelTensorCount)} per-channel tensor(s) / ${formatPercent(glance.quantization.perChannelTensorRatio)} of quantized tensors`, "OBSERVED + DERIVED"],
      ["Intensity posture", `high ${glance.intensity.highCount} / mixed ${glance.intensity.mixedCount} / low ${glance.intensity.lowCount}; ${glance.intensity.assessedOpCount}/${glance.intensity.opCount} assessed; ridge ${onnx ? "not applied" : `${glance.intensity.ridgeOpsPerByte || 0} ops/B`}`, onnx ? "DERIVED static bands" : "ESTIMATED profile bands"],
      ["Activation/cache pressure", cacheText, `${glance.memory.peakLiveEvidenceClass}; cache ratios DERIVED_FROM_${glance.memory.performanceModelEvidenceClass}_PROFILE against selected ${formatBytes(analysis?.target_profile?.l1_data_bytes || 0)} L1D / ${formatBytes(analysis?.target_profile?.l2_bytes || 0)} L2 references; watch >=0.90x`],
      [onnx ? "EP portability" : "Delegation", onnx ? `${analysis?.ort_ep_portability_frontier?.execution_provider_count || 0} source-backed EP rule set(s); runtime assignment not observed` : `${glance.delegation.delegatedOpCount}/${glance.delegation.opCount} predicted conditionally delegatable ops (${delegatedMacText}); ${glance.delegation.boundaryEdgeCount} predicted boundary edge(s), ${boundaryPayload}${boundarySetup}.${motifAttribution}`, onnx ? "SOURCE-BACKED PRECHECK" : "PREDICTED assignment + HEURISTIC cold-start setup cost"],
      ["Selected-target modeled range", onnx
        ? "NOT_ASSESSED; no cross-host ORT latency model"
        : glance.latency.range?.status === "not_assessed_dynamic_shape"
          ? `NOT_ASSESSED; ${glance.latency.range.reason || "non-batch dynamic dimensions require an explicit shape binding"}`
          : glance.latency.range?.status === "cold_setup_profile_range"
            ? `${rangeText(glance.latency.range.steadyLowUs, glance.latency.range.steadyHighUs)} steady${Math.abs(Number(glance.latency.range.steadyHighUs || 0) - Number(glance.latency.range.steadyLowUs || 0)) <= 1e-9 ? "" : ` (point ${formatUs(glance.latency.range.steadyPointUs)})`}. ${rangeText(glance.latency.range.coldLowUs, glance.latency.range.coldHighUs)} cold; partition setup ${rangeText(glance.delegation.boundaryOverhead.lowUs, glance.delegation.boundaryOverhead.highUs)}; PAD-folding candidate upper bound ${formatUs(glance.latency.range.padFusionRecoverableUpperBoundUs || 0)} across ${formatNumber(glance.delegation.padFusion?.directConvolutionCandidateCount || 0)}/${formatNumber(glance.delegation.padFusion?.candidateCount || 0)} direct-convolution candidate(s)`
            : `${formatUs(glance.latency.totals.steadyStateUs)} steady / ${formatUs(glance.latency.totals.coldStartUs)} cold`, onnx || glance.latency.range?.status === "not_assessed_dynamic_shape" ? "NOT_ASSESSED" : `HEURISTIC MODEL; steady excludes packing and setup; cold low/mid/high adds the corresponding setup profile; low bound removes only direct PAD-to-convolution candidate materialization. ${glance.delegation.padFusion?.interpretationBoundary || "Runtime fusion is not observed."}`],
      ["Cross-target modeled total", onnx ? "NOT_ASSESSED; no cross-host ORT latency model" : glance.frontierTargets.length ? glance.frontierTargets.map((target) => `${target.label}: ${formatUs(target.steadyStateUs)} steady / ${formatUs(target.coldStartUs)} cold`).join(" / ") : "NOT_ASSESSED; deployment frontier unavailable", onnx ? "NOT_ASSESSED" : "HEURISTIC MODEL; cold adds one-time packing and partition-planning setup; no cache-miss penalty inferred"],
    ]),
    `> Cache ratios use the selected profile denominator. ${glance.memory.cacheAssumption || "Cache topology was not observed from the executing device."}`,
  ].join("\n\n");
}

function targetValue(analysis, tfliteValue, onnxValue = "not used in this ONNX report") {
  return isOnnxAnalysis(analysis) ? onnxValue : tfliteValue;
}

function hardwareSourceText(source = {}) {
  const binding = source.sha256
    ? `SHA-256 ${source.sha256}`
    : source.url
      ? `URL ${source.url}; content digest not embedded`
      : "content digest not embedded";
  return `${source.document || "unnamed source"}; ${source.revision || "revision not stated"}; pages ${source.pages || "not stated"}; ${binding}; scope: ${source.scope || "not stated"}`;
}

function targetHardwareSpecRows(target = {}) {
  const spec = target?.hardware_spec;
  if (!spec) return [
    ["Hardware-spec evidence", "HEURISTIC_PROFILE; no source-bound implementation record embedded for this target"],
  ];
  return [
    ["Hardware-spec evidence", spec.evidence_class || "not emitted"],
    ["Hardware-spec scope", spec.scope || "not emitted"],
    ["Configuration context", spec.configuration_context || "not emitted"],
    ["Core configuration", spec.core_configuration || "not emitted"],
    ["Maximum documented clock", spec.max_clock_mhz ? `${formatNumber(spec.max_clock_mhz)} MHz operating target; not an observed active clock` : "not established by the bound source set"],
    ["L1 instruction cache", `${formatBytes(spec.l1_instruction_bytes)}; ${formatNumber(spec.l1_instruction_ways)}-way; ${formatNumber(spec.l1_line_bytes)} B line`],
    ["L1 data cache", `${formatBytes(spec.l1_data_bytes)}; ${formatNumber(spec.l1_data_ways)}-way; ${formatNumber(spec.l1_line_bytes)} B line`],
    ["L2 cache", `${formatBytes(spec.l2_bytes)}; ${formatNumber(spec.l2_ways)}-way; ${formatNumber(spec.l2_line_bytes)} B line`],
    ["Advanced SIMD", spec.advanced_simd ? "documented" : "not documented"],
    ["Native FP16 vector arithmetic", spec.fp16_vector_arithmetic ? "documented" : "not documented; FP16 register-element capacity must not be read as native FP16 arithmetic support"],
    ["Hardware source documents", (spec.sources || []).map(hardwareSourceText).join(" / ") || "not emitted"],
  ];
}

function artifactSizeBreakdownMarkdown(analysis) {
  const size = analysis?.size_breakdown || {};
  if (isOnnxAnalysis(analysis)) {
    const external = analysis?.onnx_external_data || {};
    const structureBinding = analysis?.onnx_external_data_structure_binding || null;
    const externalRows = (external.tensors || []).slice(0, 24).map((tensor) => [
      code(tensor.scope || "unknown_scope"),
      tensor.tensor_role || "TensorProto",
      code(tensor.tensor_name || "(unnamed)"),
      `${tensor.dtype || "UNKNOWN"} [${(tensor.shape || []).join(", ")}]`,
      tensor.normalized_location && tensor.normalized_location !== tensor.location
        ? `${code(tensor.location || "(missing)")} -> ${code(tensor.normalized_location)}`
        : code(tensor.location || "(missing)"),
      tensor.location_status || "not assessed",
      `${tensor.offset == null ? "default 0" : formatNumber(tensor.offset)} / ${tensor.length == null ? "to EOF or undeclared" : `${formatNumber(tensor.length)} B`}`,
      tensor.checksum ? `${tensor.checksum} (${tensor.checksum_status || "not assessed"})` : "not declared",
      tensor.sidecar_sha256 ? `${tensor.sidecar_path || tensor.location}; ${formatBytes(tensor.sidecar_bytes)}; SHA-256 ${tensor.sidecar_sha256}` : "not supplied",
      tensor.reference_status || "not assessed",
      tensor.payload_status === "verified" ? `verified; ${formatBytes(tensor.payload_bytes)} (${formatNumber(tensor.payload_bytes)} B)` : tensor.payload_status || "not supplied",
    ]);
    return [
      "## Artifact Size Breakdown (OBSERVED)",
      markdownTable(["Field", "Value"], [
        ["File size", `${formatBytes(size.file_size || analysis.file_size || 0)} (${formatNumber(size.file_size || analysis.file_size || 0)} B)`],
        ["Main-graph initializer declarations", `${formatNumber(size.constant_tensor_count || 0)} total / ${formatNumber(size.embedded_constant_tensor_count || 0)} fully embedded / ${formatNumber(size.external_data_tensor_count || 0)} external TensorProto component(s)`],
        ["Dense / sparse initializer declarations", `${formatNumber(size.dense_initializer_count || 0)} / ${formatNumber(size.sparse_initializer_count || 0)}`],
        ["Main-graph embedded initializer payload", `${formatBytes(size.constant_bytes || 0)} (${formatNumber(size.constant_bytes || 0)} B), ${formatNumber(size.stored_scalar_elements || 0)} stored values/indices element(s)`],
        ["Main-graph verified external initializer payload", `${formatBytes(size.verified_external_payload_bytes || 0)} (${formatNumber(size.verified_external_payload_bytes || 0)} B), ${formatNumber(size.verified_external_scalar_elements || 0)} stored values/indices element(s)`],
        ["Main-graph available initializer total", `${formatBytes(size.available_initializer_bytes || 0)} (${formatNumber(size.available_initializer_bytes || 0)} B), ${formatNumber(size.available_initializer_scalar_elements || 0)} stored values/indices element(s)`],
        ["Logical initializer elements", `${formatNumber(size.logical_initializer_elements || 0)} dense-semantic element(s); sparse implicit zeros are logical elements, not stored payload`],
        ["FLOAT initializer payload bytes", size.float_constant_bytes == null ? "NOT_ASSESSED" : `${formatBytes(size.float_constant_bytes)} (${formatNumber(size.float_constant_bytes)} B)`],
        ["External-data evidence envelope", `${external.schema || "schema not emitted"} / ${external.status || "not emitted"} / ${external.evidence_class || "evidence class not emitted"}`],
        ["All-scope ONNX external data", `${formatNumber(external.tensor_count || 0)} TensorProto reference(s), ${formatNumber(external.entry_count || 0)} key/value record(s), ${formatBytes(external.verified_payload_bytes || 0)} verified`],
        ["External reference integrity", `${formatNumber(external.malformed_reference_count || 0)} malformed / ${formatNumber(external.unsafe_location_count || 0)} unsafe location / ${formatNumber(external.missing_location_count || 0)} missing location / ${formatNumber(external.duplicate_key_count || 0)} duplicate key / ${formatNumber(external.invalid_range_count || 0)} invalid range / ${formatNumber(external.invalid_checksum_count || 0)} invalid checksum / ${formatNumber(external.embedded_payload_conflict_count || 0)} embedded-payload conflict / ${formatNumber(external.data_location_mismatch_count || 0)} data-location mismatch`],
        ["External payload coverage", `${formatNumber(external.supplied_payload_count || 0)} supplied / ${formatNumber(external.verified_payload_count || 0)} verified / ${formatNumber(external.payload_verification_failed_count || 0)} failed; verified ${formatBytes(external.verified_payload_bytes || 0)}; declared ${external.declared_payload_bytes == null ? "not fully declared" : `${formatBytes(external.declared_payload_bytes)} (${formatNumber(external.declared_payload_bytes)} B)`}`],
        ["External payload failures", `${formatNumber(external.range_out_of_bounds_count || 0)} out-of-bounds / ${formatNumber(external.payload_size_mismatch_count || 0)} dtype-shape byte mismatch / ${formatNumber(external.checksum_mismatch_count || 0)} SHA-1 mismatch`],
        ["External sidecar files", `${formatNumber(external.supplied_file_count || 0)} supplied (${formatBytes(external.supplied_file_bytes || 0)}) / ${formatNumber(external.used_file_count || 0)} used / ${formatNumber(external.unused_file_count || 0)} unused`],
        ["Structure-mode sidecar binding", structureBinding
          ? `${structureBinding.status}; ${formatNumber(structureBinding.file_count)} content-hashed file(s), ${formatNumber(structureBinding.tensor_count)} serialized range(s), ${formatBytes(Number(structureBinding.declared_payload_bytes?.number || 0))} declared; numerical payload decode ${structureBinding.numerical_payload_decode}`
          : "not emitted"],
        ["External-data detail", external.detail || "not emitted"],
        ["Separately attributable metadata bytes", size.metadata_bytes == null ? "NOT_ASSESSABLE; protobuf structure and metadata are not separated" : `${formatBytes(size.metadata_bytes)} (${formatNumber(size.metadata_bytes)} B)`],
        ["Graph protobuf / metadata overhead", `${formatBytes(size.structure_overhead_bytes || 0)} (${formatNumber(size.structure_overhead_bytes || 0)} B)`],
        ["Unique / duplicate embedded initializer bytes", size.duplicate_initializer_analysis?.status === "assessed"
          ? `${formatBytes(size.unique_constant_bytes)} (${formatNumber(size.unique_constant_bytes)} B) / ${formatBytes(size.duplicate_constant_bytes)} (${formatNumber(size.duplicate_constant_bytes)} B)`
          : `NOT_ASSESSED; values withheld as ${size.unique_constant_bytes == null ? "null" : formatNumber(size.unique_constant_bytes)} / ${size.duplicate_constant_bytes == null ? "null" : formatNumber(size.duplicate_constant_bytes)} because ${size.duplicate_initializer_analysis?.reason || "complete duplicate equivalence was not evaluated"}`],
        ["Unique / duplicate available initializer bytes", size.duplicate_initializer_analysis?.status === "assessed"
          ? `${formatBytes(size.available_unique_constant_bytes)} (${formatNumber(size.available_unique_constant_bytes)} B) / ${formatBytes(size.available_duplicate_constant_bytes)} (${formatNumber(size.available_duplicate_constant_bytes)} B)`
          : `NOT_ASSESSED; values withheld as ${size.available_unique_constant_bytes == null ? "null" : formatNumber(size.available_unique_constant_bytes)} / ${size.available_duplicate_constant_bytes == null ? "null" : formatNumber(size.available_duplicate_constant_bytes)} because ${size.duplicate_initializer_analysis?.reason || "complete duplicate equivalence was not evaluated"}`],
        ["Projected embedded initializer payload at FP16", `${formatBytes(size.theoretical_fp16_constant_bytes)} (${formatNumber(size.theoretical_fp16_constant_bytes)} B); FLOAT elements x 2 B, non-FLOAT bytes unchanged`],
        ["Projected embedded initializer payload at INT8", `${formatBytes(size.theoretical_int8_constant_bytes)} (${formatNumber(size.theoretical_int8_constant_bytes)} B); FLOAT elements x 1 B, non-FLOAT bytes unchanged; excludes quantization metadata/bias policy/alignment`],
        ["Raw 0x00 byte ratio across available initializer payloads", size.metrics?.zero_constant_byte_ratio?.status === "assessed"
          ? `${formatPercent(size.zero_constant_byte_ratio)} (compressibility hint, not scalar sparsity)`
          : `NOT_ASSESSED; ${size.metrics?.zero_constant_byte_ratio?.reason || "exact raw-byte coverage unavailable"}`],
      ]),
      `> ${size.detail || "ONNX initializer byte counts are read from TensorProto raw_data or typed scalar fields when present."}`,
      externalRows.length ? "### ONNX External Data Reference Ledger" : "",
      externalRows.length ? markdownTable(["Scope", "Tensor role", "Tensor", "Dtype / shape", "Declared / canonical location", "Path status", "Offset / length", "SHA-1 checksum", "Supplied sidecar", "Reference", "Payload"], externalRows) : "",
      (external.tensors || []).length > externalRows.length ? `> ${formatNumber((external.tensors || []).length - externalRows.length)} additional external-data tensor reference(s) remain in structured evidence.` : "",
      externalRows.length ? `> ${external.detail || "External payload bytes were not supplied with the selected .onnx file."}` : "",
    ].filter(Boolean).join("\n");
  }
  return [
    "## Artifact Size Breakdown (OBSERVED)",
    markdownTable(["Field", "Value"], [
      ["File size", `${formatBytes(size.file_size || analysis.file_size || 0)} (${formatNumber(size.file_size || analysis.file_size || 0)} B)`],
      ["Physical constant buffer bytes", `${formatBytes(size.constant_bytes || 0)} (${formatNumber(size.constant_bytes || 0)} B) across ${formatNumber(size.physical_constant_buffer_count || 0)} unique location(s), referenced by ${formatNumber(size.constant_tensor_count || 0)} tensor(s)`],
      ["Logical constant references", `${formatBytes(size.logical_constant_reference_bytes || 0)} (${formatNumber(size.logical_constant_reference_bytes || 0)} B reference-sum; ${formatNumber(size.stored_scalar_elements || 0)} physically stored scalar elements)`],
      ["FLOAT constant tensor bytes", `${formatBytes(size.float_constant_bytes || 0)} (${formatNumber(size.float_constant_bytes || 0)} B)`],
      ["Unique / duplicate constant bytes", `${formatBytes(size.unique_constant_bytes || 0)} (${formatNumber(size.unique_constant_bytes || 0)} B) / ${formatBytes(size.duplicate_constant_bytes || 0)} (${formatNumber(size.duplicate_constant_bytes || 0)} B)`],
      ["TFLite Model Metadata payload bytes", `${formatBytes(size.metadata_bytes || 0)} (${formatNumber(size.metadata_bytes || 0)} B)`],
      ["Residual bytes outside constant and metadata-buffer payloads", `${formatBytes(size.structure_overhead_bytes || 0)} (${formatNumber(size.structure_overhead_bytes || 0)} B); ownership is resolved separately by the Artifact Byte Integrity Ledger`],
      ["Hypothetical all-FLOAT32-to-FP16 scalar payload floor", Number(size.float_constant_bytes || 0) > 0
        ? `${formatBytes(size.theoretical_fp16_constant_bytes || 0)} (float constants halved; non-float constants unchanged)`
        : "NOT_APPLICABLE; artifact contains 0 B of FLOAT constant payload, so an FP32-to-FP16 storage counterfactual is a no-op"],
      ["Hypothetical all-FLOAT32-to-INT8 scalar payload floor", Number(size.float_constant_bytes || 0) > 0
        ? `${formatBytes(size.theoretical_int8_constant_bytes || 0)} (ideal scalar payload floor only; excludes INT32 bias retention, scales, zero-points, alignment, metadata, runtime packing, and accuracy constraints)`
        : "NOT_APPLICABLE; artifact contains 0 B of FLOAT constant payload, so an FP32-to-INT8 storage counterfactual is a no-op"],
      ["Raw 0x00 byte ratio in constant buffers", `${formatPercent(size.zero_constant_byte_ratio || 0)} (compressibility hint, not scalar sparsity)`],
      ["Analyzer detail", size.detail || "not emitted"],
    ]),
    "> \"Stored constant elements\", not \"trainable parameters\": a deployment artifact cannot establish which constants were trainable.",
  ].join("\n");
}

function artifactByteIntegrityMarkdown(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "tflite") return "";
  const ledger = analysis?.artifact_byte_integrity;
  if (!ledger?.schema) return "";
  const referencedRanges = Array.isArray(ledger.flatbuffer_referenced_ranges)
    ? ledger.flatbuffer_referenced_ranges
    : [];
  const unownedRanges = Array.isArray(ledger.unowned_trailing_ranges)
    ? ledger.unowned_trailing_ranges
    : [];
  const rangeRows = [...referencedRanges.slice(0, 24), ...unownedRanges].map((range) => [
    range.class || "unclassified",
    formatNumber(range.offset),
    formatNumber(range.end),
    `${formatBytes(range.length)} (${formatNumber(range.length)} B)`,
  ]);
  return [
    "## Artifact Byte Integrity Ledger (DERIVED)",
    markdownTable(["Field", "Value"], [
      ["Schema / status", `${ledger.schema} / ${ledger.status || "not emitted"}`],
      ["File-size conservation", `${ledger.conservation_status || "not assessed"}; classified ${ledger.classified_bytes == null ? "not available" : `${formatNumber(ledger.classified_bytes)} B`} / file ${formatNumber(ledger.file_size)} B`],
      ["Verified FlatBuffer references", `${formatBytes(ledger.flatbuffer_referenced_bytes)} across ${formatNumber(ledger.flatbuffer_referenced_range_count)} merged range(s); greatest referenced end ${formatNumber(ledger.flatbuffer_referenced_end)}`],
      ["FlatBuffer envelope", `${formatBytes(ledger.flatbuffer_envelope_bytes)}; ${formatBytes(ledger.flatbuffer_internal_alignment_or_unreferenced_bytes)} internal alignment or unreferenced bytes inside the envelope`],
      ["Terminal zero-alignment candidate", `${formatBytes(ledger.terminal_zero_alignment_bytes)}; only an all-zero gap of at most 16 B receives this classification`],
      ["Terminal metadata ZIP", `${ledger.metadata_archive_status || "not assessed"}; ${formatBytes(ledger.metadata_archive_bytes)}; range ${ledger.metadata_archive_start == null ? "not present" : `[${formatNumber(ledger.metadata_archive_start)}..${formatNumber(ledger.metadata_archive_end)})`}; ${formatNumber(ledger.metadata_archive_file_count)} file(s)`],
      ["ZIP directory / EOCD", ledger.metadata_archive_central_directory_start == null ? "not present" : `central [${formatNumber(ledger.metadata_archive_central_directory_start)}..${formatNumber(ledger.metadata_archive_central_directory_end)}); EOCD ${formatNumber(ledger.metadata_archive_eocd_offset)}`],
      ["Unowned trailing bytes", `${formatBytes(ledger.unowned_trailing_bytes)} across ${formatNumber(unownedRanges.length)} exact range(s)`],
      ["Buffer range ownership", `${formatNumber(ledger.exact_shared_buffer_range_count)} exact shared reference(s); ${formatNumber(ledger.partial_buffer_overlap_count)} partial overlap pair(s)`],
      ["Archive filename collisions", `${formatNumber(ledger.metadata_archive_case_insensitive_name_collision_count)} case-insensitive collision(s)`],
      ["Encoded archive size policy", `${formatBytes(ledger.metadata_archive_size_policy_bytes)} limit; ${ledger.metadata_archive_size_policy_exceeded ? "exceeded" : "within limit"}`],
      ["Issues", ledger.issues?.length ? ledger.issues.join("; ") : "none"],
      ["Method", ledger.method || "not emitted"],
    ]),
    rangeRows.length ? "### Byte Range Ledger" : "",
    rangeRows.length ? markdownTable(["Class", "Start", "End exclusive", "Length"], rangeRows) : "",
    referencedRanges.length > 24 ? `> ${formatNumber(referencedRanges.length - 24)} additional verified FlatBuffer reference range(s) remain in the structured evidence.` : "",
  ].filter(Boolean).join("\n");
}

function tfliteSparseStorageMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const sparse = analysis?.tflite_sparse_storage_contract || {};
  const rows = Array.isArray(sparse.rows) ? sparse.rows : [];
  const renderedRows = rows.slice(0, 64).map((row) => {
    const encoding = row.encoding || {};
    return [
      `S${formatNumber(row.subgraph_index || 0)}/T${formatNumber(row.tensor_index)}`,
      code(row.tensor_name || "(unnamed)"),
      `${row.dtype || "UNKNOWN"} [${(row.shape || []).join(", ")}]`,
      `${formatNumber(encoding.logical_element_count || 0)} = ${formatNumber(encoding.stored_element_count || 0)} + ${formatNumber(encoding.implicit_zero_element_count || 0)}`,
      encoding.stored_value_bytes == null ? "runtime/external" : `${formatNumber(encoding.stored_value_bytes)} B`,
      (encoding.dimensions || []).map((dimension) => `${dimension.format}@d${dimension.expanded_dimension}`).join(" -> ") || "not emitted",
      encoding.status || "not assessed",
      code(encoding.canonical_metadata_sha256 || "not emitted"),
    ];
  });
  return [
    `## TFLite Sparse Storage Contract (${sparse.status === "assessed" ? "DERIVED" : sparse.status === "partial" ? "DERIVED/PARTIAL" : "NOT_APPLICABLE"})`,
    markdownTable(["Field", "Value"], [
      ["Schema / status / evidence", `${sparse.schema || "not emitted"} / ${sparse.status || "not emitted"} / ${sparse.evidence_class || "not emitted"}`],
      ["Sparse tensors", `${formatNumber(sparse.sparse_tensor_count || 0)} total; ${formatNumber(sparse.serialized_value_tensor_count || 0)} serialized-value; ${formatNumber(sparse.fully_decoded_tensor_count || 0)} fully decoded; ${formatNumber(sparse.partial_tensor_count || 0)} partial`],
      ["Element conservation", `${formatNumber(sparse.logical_element_count || 0)} logical = ${formatNumber(sparse.stored_element_count || 0)} stored + ${formatNumber(sparse.implicit_zero_element_count || 0)} implicit raw-zero element(s)`],
      ["Serialized sparse value bytes", `${formatNumber(sparse.serialized_value_bytes || 0)} B`],
      ["Pinned TensorFlow commit", code(sparse.source_commit || "not emitted")],
      ["Schema source SHA-256", `${code(sparse.schema_source || "not emitted")} / ${code(sparse.schema_source_sha256 || "not emitted")}`],
      ["Converter source SHA-256", `${code(sparse.converter_source || "not emitted")} / ${code(sparse.converter_source_sha256 || "not emitted")}`],
      ["Method", sparse.method || "not emitted"],
    ]),
    renderedRows.length ? "### Sparse Tensor Ledger" : "",
    renderedRows.length ? markdownTable(["Tensor", "Name", "Dtype / shape", "Logical = stored + implicit", "Physical values", "Traversal levels", "Status", "Metadata SHA-256"], renderedRows) : "No TFLite SparsityParameters record is serialized in this artifact.",
    rows.length > renderedRows.length ? `> ${formatNumber(rows.length - renderedRows.length)} additional sparse tensor row(s) remain in engineering_evidence.json and static/static_analysis.json.` : "",
    `> ${sparse.interpretation_boundary || "Sparse storage metadata does not establish retained sparse runtime execution."}`,
  ].filter(Boolean).join("\n\n");
}

function peakLiveActivationMarkdown(analysis) {
  const live = analysis?.tensor_liveness || {};
  if (!live.assessed) {
    return [
      "## Peak Live Activation Payload (NOT_ASSESSABLE)",
      markdownTable(["Field", "Value"], [
        ["Assessment status", live.status || "not_assessed"],
        ["Peak live activation payload", live.peak_bytes == null ? "not assessed (null; no zero substitution)" : formatBytes(live.peak_bytes)],
        ["Evidence / peak-byte status", `${live.evidence_class || "NOT_ASSESSABLE"} / ${live.peak_bytes_status || "not_assessed"}`],
        ["Reason", live.unassessed_tensors?.[0]?.reason || live.non_dense_values?.[0]?.reason || (isOnnxAnalysis(analysis) ? "ONNX shape inference did not produce enough activation shapes to compute a liveness payload without inventing defaults" : "A required tensor payload is not deterministically known from the declared shape and dtype")],
        ["Assessed / unassessed activation tensors", `${formatNumber(live.assessed_tensor_count ?? 0)} / ${formatNumber(live.unassessed_tensor_count ?? 0)}`],
        ["Unknown activation tensors", formatNumber(live.unknown_activation_tensors ?? 0)],
        ["Non-dense runtime values excluded", `${formatNumber(live.non_dense_value_count || 0)}; ${(live.non_dense_values || []).slice(0, 8).map((row) => `T${row.tensor_index} ${row.tensor_name || "-"} (${row.value_kind}): ${row.reason}`).join(" / ") || "none"}`],
        ["Method", live.method || "No unknown shape or dtype substitution is permitted"],
      ]),
    ].join("\n");
  }
  const partial = live.status === "partial";
  const arenaReconciliation = peakArenaReconciliation(analysis, live);
  return [
    `## Peak Live Activation Payload (${partial ? "DERIVED/PARTIAL LOWER BOUND" : "DERIVED"})`,
    markdownTable(["Field", "Value"], [
      ["Evidence / peak-byte status", `${live.evidence_class || "DERIVED"} / ${live.peak_bytes_status || live.status || "assessed"}`],
      [partial ? "Assessed-tensor live subtotal lower bound" : "Estimated peak live activation payload", `${formatBytes(live.peak_bytes)}`],
      ["Peak at op", live.peak_at_op == null ? "graph input lifetime" : `#${padOp(live.peak_at_op)} ${live.peak_at_op_name || "-"}`],
      ["Assessed / unassessed activation tensors", `${formatNumber(live.assessed_tensor_count || 0)} / ${formatNumber(live.unassessed_tensor_count || 0)}`],
      ["Unknown activation tensors after shape assessment", formatNumber(live.unknown_activation_tensors ?? live.unassessed_tensor_count ?? 0)],
      ["Non-dense runtime values excluded", `${formatNumber(live.non_dense_value_count || 0)}; ${(live.non_dense_values || []).slice(0, 8).map((row) => `T${row.tensor_index} ${row.tensor_name || "-"} (${row.value_kind}): ${row.reason}`).join(" / ") || "none"}`],
      ...(arenaReconciliation ? [["Peak-live / ArenaPlanner reconciliation", arenaReconciliation]] : []),
      ["Method", live.method || "tensor liveness sweep over op order (producer to last consumer); declared shapes"],
    ]),
    partial
      ? "> PARTIAL LOWER BOUND: the subtotal excludes every dense activation with unknown shape or dtype and every declared non-dense runtime value; none are substituted with zero bytes. It is not a complete peak-live or runtime-arena estimate."
      : "> DERIVED: computed from the graph and declared/inferred shapes as live activation payload. The actual runtime arena may be lower through storage aliasing, view tensors, shape-only operations, or in-place execution, or higher because of alignment, allocator behavior, persistent buffers, delegate boundaries, temporary tensors, operator-specific scratch, and bookkeeping. Weights and runtime scratch are excluded from this payload number.",
  ].join("\n");
}

function tensorArenaPlanMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const plan = analysis?.tensor_arena_plan;
  if (!plan) {
    return [
      "## TFLite ArenaPlanner Declared-Shape Projection (NOT_ASSESSABLE)",
      "The analyzer did not emit a source-bound ArenaPlanner projection.",
    ].join("\n");
  }
  const totalsAssessed = plan.non_persistent_arena_bytes != null
    && plan.persistent_arena_bytes != null
    && plan.combined_arena_bytes != null;
  const roots = (plan.allocations || [])
    .filter((allocation) => allocation.allocation_status === "allocated")
    .sort((left, right) => Number(right.size_bytes || 0) - Number(left.size_bytes || 0) || Number(left.tensor_index) - Number(right.tensor_index));
  const aliases = plan.aliases || [];
  const sourceLink = `[tensorflow/tensorflow@${plan.source_commit}](${plan.planner_source_url})`;
  const arenaLink = `[SimpleMemoryArena](${plan.arena_source_url})`;
  return [
    `## TFLite ArenaPlanner Declared-Shape Projection (${plan.evidence_class || "DERIVED"}${plan.status === "assessed" ? "" : " / PARTIAL"})`,
    markdownTable(["Field", "Value"], [
      ["Projection status", plan.status || "not_assessed"],
      ["Projection schema", plan.schema || "not emitted"],
      ["Non-persistent arena high-water mark", totalsAssessed ? `${formatBytes(plan.non_persistent_arena_bytes)} (${formatNumber(plan.non_persistent_arena_bytes)} B)` : "NOT_ASSESSED"],
      ["Persistent arena high-water mark", totalsAssessed ? `${formatBytes(plan.persistent_arena_bytes)} (${formatNumber(plan.persistent_arena_bytes)} B)` : "NOT_ASSESSED"],
      ["Combined arena projection", totalsAssessed ? `${formatBytes(plan.combined_arena_bytes)} (${formatNumber(plan.combined_arena_bytes)} B)` : "NOT_ASSESSED"],
      ["Root allocations / in-place aliases", `${formatNumber(plan.root_allocation_count || 0)} / ${formatNumber(plan.shared_tensor_count || 0)}`],
      ["Planned tensors / non-persistent / persistent allocations", `${formatNumber(plan.planned_tensor_count || 0)} / ${formatNumber(plan.non_persistent_allocation_count || 0)} / ${formatNumber(plan.persistent_allocation_count || 0)}`],
      ["Preserve-all-tensors mode", plan.preserve_all_tensors ? "enabled" : "disabled"],
      ["Tensor alignment", `${formatNumber(plan.tensor_alignment_bytes || 0)} B`],
      ["Dynamic shape-signature tensors", formatNumber(plan.dynamic_shape_signature_tensor_count || 0)],
      ["Source-comparator equal-key groups", `${formatNumber(plan.source_comparator_tie_group_count || 0)} group(s), ${formatNumber(plan.source_comparator_tied_tensor_count || 0)} tensor(s)`],
      ["Allocation ordering", plan.source_comparator_fully_orders_projection ? "Pinned source comparator fully orders this projection" : `DeepBOM deterministic extension applied: ${plan.deterministic_tie_break || "tensor-index tie-break"}`],
      ["Deterministic equal-key tie-break", plan.deterministic_tie_break || "not emitted"],
      ["Unassessed tensors", formatNumber(plan.unassessed_tensor_count || 0)],
      ["Calculation issues", formatNumber(plan.calculation_issue_count || 0)],
      ["Pinned planner source", `${sourceLink}; ${arenaLink}`],
      ["Method", plan.method || "not emitted"],
    ]),
    "### Root Allocation Inventory",
    roots.length ? markdownTable(["Tensor", "Arena", "Shape / dtype", "Bytes", "Offset", "Lifetime"], roots.slice(0, 32).map((allocation) => [
      `T${allocation.tensor_index} ${code(allocation.tensor_name || "-")}`,
      allocation.arena || "-",
      `${(allocation.tensor_shape || []).join("x") || "scalar"} / ${allocation.tensor_dtype || "-"}`,
      allocation.size_bytes == null ? "NOT_ASSESSED" : `${formatBytes(allocation.size_bytes)} (${formatNumber(allocation.size_bytes)} B)`,
      allocation.offset_bytes == null ? "NOT_ASSESSED" : `${formatBytes(allocation.offset_bytes)} (${formatNumber(allocation.offset_bytes)} B)`,
      `#${allocation.first_node}-${allocation.last_node == null ? "inference end" : `#${allocation.last_node}`}`,
    ])) : "No root arena allocation was emitted.",
    roots.length > 32 ? `> ${formatNumber(roots.length - 32)} additional root allocation(s) remain in structured evidence.` : "",
    aliases.length ? "### In-place Alias Inventory" : "",
    aliases.length ? markdownTable(["Alias", "Root", "Operator", "Mutation", "Pinned registration source"], aliases.slice(0, 32).map((alias) => [
      `T${alias.tensor_index} ${code(alias.tensor_name || "-")}`,
      `T${alias.shared_with_tensor_index} ${code(alias.shared_with_tensor_name || "-")}`,
      `#${padOp(alias.op_index)} ${alias.op_name || "-"}; input ${alias.input_slot}`,
      alias.data_unmodified ? "data unmodified" : "buffer contents changed",
      alias.source || "not emitted",
    ])) : "",
    aliases.length > 32 ? `> ${formatNumber(aliases.length - 32)} additional alias record(s) remain in structured evidence.` : "",
    (plan.unassessed_tensors || []).length ? `> Calculation issues: ${(plan.unassessed_tensors || []).slice(0, 6).map((item) => `${item.tensor_index == null ? "plan" : `T${item.tensor_index}`}: ${item.reason}`).join("; ")}` : "",
    `> DERIVED declared-shape projection: ${plan.interpretation_boundary || "This is not observed runtime memory."}`,
  ].filter(Boolean).join("\n");
}

function runtimeRequirementsMarkdown(analysis, bundledRuntimeVersion, runtimeConclusion) {
  if (isOnnxAnalysis(analysis)) {
    const floor = analysis.ort_compatibility_evidence?.runtime_floor || null;
    const artifactRuntime = analysis.runtime_compat || {};
    const assessed = floor?.evidence_class === "DERIVED_NECESSARY_MINIMUM";
    const complete = [
      "assessed_onnx_and_model_local_domains",
      "assessed_onnx_model_local_and_source_backed_contrib_domains",
    ].includes(floor?.status);
    return [
      `## ONNX Runtime Requirements (${assessed ? "OBSERVED/DERIVED" : "OBSERVED/NOT_ASSESSABLE"})`,
      markdownTable(["Field", "Value"], [
        ["Bundled application runtime version", bundledRuntimeVersion || "not provided"],
        ["Necessary ONNX parser/schema runtime floor", assessed ? `ONNX Runtime ${floor.minimum_ort_version}` : "NOT_ASSESSABLE"],
        ["Standard IR/opset / contrib component floors", `${floor?.standard_minimum_ort_version || "not derived"} / ${floor?.contrib_minimum_ort_version || "not applicable"}`],
        ["ai.onnx / ai.onnx.ml opsets", `${floor?.standard_domain_opset ?? "missing"} / ${floor?.standard_ml_domain_opset ?? "not imported"}`],
        ["Floor status", floor?.status || "protected ORT rulepack not loaded"],
        ["External runtime-registry constraints", (floor?.unresolved_domains || []).join(" / ") || "none"],
        ["Artifact-defined local function domains", (floor?.model_local_function_domains || []).join(" / ") || "none"],
        ["Source-backed external domains", (floor?.source_backed_external_domains || []).join(" / ") || "none"],
        ["Contrib operator first-release floors", (floor?.contrib_operator_floors || []).map((row) => `${row.domain}::${row.op_name} import ${row.imported_opset}: ORT ${row.minimum_ort_version}; ${row.evidence_class}; ${row.source_sha256}; ${row.source_ref}`).join(" / ") || "none"],
        ["Compatibility conclusion", complete
          ? "Necessary parser/schema floor derived; deployment compatibility remains NOT_CONCLUDED until selected EP, build, device, and runtime assignment are bound"
          : "NOT_ASSESSABLE as a complete deployment floor"],
        ["ONNX IR version", analysis.onnx_ir_version || "unknown"],
        ["Opsets", (analysis.opsets || []).map((opset) => `${opset.domain || "ai.onnx"}:${opset.version || "unknown"}`).join(" / ") || "unknown"],
        ["Analyzer-derived / effective floor", `${artifactRuntime.derived_min_runtime_version || "not derived"} / ${artifactRuntime.effective_min_runtime_version || "not determined"}`],
        ["Declared artifact minimum", artifactRuntime.min_runtime_version || "not embedded"],
        ["Artifact runtime-floor status / evidence", `${artifactRuntime.runtime_floor_status || "not emitted"} / ${artifactRuntime.runtime_floor_evidence_class || "not emitted"}`],
        ["Highest version requirement / unmapped versioned ops", `${artifactRuntime.max_op_version ?? "not emitted"} / ${(artifactRuntime.unmapped_versioned_ops || []).join(", ") || "none"}`],
        ["Unresolved artifact runtime-floor domains", (artifactRuntime.unresolved_runtime_floor_domains || []).join(" / ") || "none"],
        ["Artifact-defined local function domains", (artifactRuntime.model_local_function_domains || []).join(" / ") || "none"],
        ["Artifact runtime detail", artifactRuntime.detail || "not emitted"],
        ["Runtime-version basis", artifactRuntime.runtime_version_basis || "not emitted"],
        ["Floor source documents", (floor?.source_documents || []).map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}; ${source.detail}`).join(" / ") || "not emitted"],
      ]),
      assessed
        ? `> DERIVED_NECESSARY_MINIMUM: ${floor.basis} This is not proof that every EP kernel, reduced build, custom domain, device, or session configuration can execute the graph.`
        : "> OBSERVED: IR/opset metadata is parsed from the ONNX artifact. No protected, pinned runtime-floor result was available in this export.",
    ].join("\n");
  }
  const runtime = analysis.runtime_compat || {};
  const mappedFloor = runtime.derived_min_runtime_version || "not derived";
  const effectiveFloor = runtime.effective_min_runtime_version || "";
  const completeFloor = runtime.runtime_floor_status === "complete_for_observed_builtin_op_versions";
  return [
    "## Artifact-side Runtime Requirements (OBSERVED/DERIVED)",
    markdownTable(["Field", "Value"], [
      ["Bundled application runtime version", bundledRuntimeVersion || "not provided"],
      ["Compatibility conclusion", runtimeConclusion],
      ["Declared min_runtime_version", runtime.min_runtime_version || "not embedded"],
      ["Observed op-version necessary floor", mappedFloor],
      ["Runtime-floor evidence class", runtime.runtime_floor_evidence_class || "not emitted"],
      ["Runtime-map coverage", `${runtime.runtime_floor_status || "coverage status not emitted"}; ${formatNumber(runtime.operator_code_count || 0)} operator code(s) total; ${formatNumber(runtime.mapped_operator_code_count || 0)}/${formatNumber(runtime.builtin_operator_code_count || 0)} builtin op code(s) mapped; ${formatNumber(runtime.custom_operator_code_count || 0)} custom op code(s)`],
      ["Effective artifact-side runtime floor", completeFloor ? (effectiveFloor || "not determined") : `NOT_ASSESSABLE; partial mapped-op necessary floor is ${mappedFloor}${effectiveFloor ? `; ignored inconsistent emitted value ${effectiveFloor}` : ""}`],
      ["Highest observed operator version", `${runtime.max_op_version || 1}`],
      ["Version-driving ops", (runtime.version_driving_ops || []).slice(0, 6).join(", ") || "-"],
      ["Unmapped versioned ops", (runtime.unmapped_versioned_ops || []).join(", ") || "none"],
      ["Analyzer detail", runtime.detail || "not emitted"],
      ["Runtime-version basis", runtime.runtime_version_basis || "not emitted by analyzer"],
    ]),
    "> OBSERVED/DERIVED: declared metadata is read from the artifact when present. The mapped-op value is a necessary floor from the pinned TensorFlow op/version table, not an execution guarantee. An effective floor is emitted only when every observed builtin op code/version is mapped; application compatibility additionally requires the exact bundled LiteRT/TFLite runtime version and build flags.",
  ].join("\n");
}

function artifactMetadataMarkdown(analysis) {
  const metadata = analysis.metadata_presence || {};
  if (isOnnxAnalysis(analysis)) {
    const propertyRows = (metadata.metadata_properties || []).slice(0, 24).map((entry) => [
      entry.scope || "model",
      code(entry.key || "(empty key)"),
      entry.value || "(empty value)",
    ]);
    return [
      "## Artifact Metadata & Signatures (OBSERVED)",
      markdownTable(["Item", "Present", "Detail"], [
        ["Metadata parser envelope", `${metadata.format || "onnx"} / ${metadata.status || "not emitted"} / ${metadata.schema || "schema not emitted"}`, `${formatNumber(metadata.graph_input_count ?? (analysis.inputs || []).length)} input(s), ${formatNumber(metadata.graph_output_count ?? (analysis.outputs || []).length)} output(s); ${metadata.detail || "not emitted"}`],
        ["ONNX graph input/output contracts", "yes", `${formatNumber((analysis.inputs || []).length)} input(s), ${formatNumber((analysis.outputs || []).length)} output(s)`],
        ["Model metadata present", metadata.has_model_metadata ? "yes" : "no", `${metadata.description || "no doc_string"}; keys ${(metadata.metadata_entries || []).slice(0, 24).join(", ") || "none"}${(metadata.metadata_entries || []).length > 24 ? `; +${formatNumber((metadata.metadata_entries || []).length - 24)} more` : ""}`],
        ["ONNX producer", metadata.producer_name ? "yes" : "no", `${metadata.producer_name || "none"}${metadata.producer_version ? ` @ ${metadata.producer_version}` : ""}`],
        ["Model domain / version", metadata.model_domain || metadata.model_version ? "yes" : "no", `${metadata.model_domain || "none"} / ${metadata.model_version ?? "not declared"}`],
        ["Model / graph doc_string", metadata.has_description ? "yes" : "no", `${metadata.model_doc_string || "model doc_string absent"} / ${metadata.graph_doc_string || "graph doc_string absent"}`],
        ["ONNX metadata_props", formatNumber(metadata.metadata_property_count || 0), `${(metadata.metadata_entries || []).slice(0, 24).join(", ") || "none"}${(metadata.metadata_entries || []).length > 24 ? `; +${formatNumber((metadata.metadata_entries || []).length - 24)} more` : ""}; decoded metadata text ${formatNumber(metadata.metadata_text_bytes || 0)} B`],
        ["Documented preprocessing contract", metadata.documented_preprocessing ? "yes" : "no", metadata.preprocessing_contract_status || "not assessed"],
        ["Output label/semantic contract", metadata.output_semantics_documented ? "yes" : "no", `${formatNumber(metadata.output_label_file_count || 0)} machine-verifiable output label mapping(s); ONNX metadata_props remain untyped declarations`],
      ]),
      propertyRows.length ? "### ONNX Metadata Properties" : "",
      propertyRows.length ? markdownTable(["Scope", "Key", "Value"], propertyRows) : "",
      (metadata.metadata_properties || []).length > propertyRows.length
        ? `> ${formatNumber((metadata.metadata_properties || []).length - propertyRows.length)} additional metadata property row(s) remain in structured evidence.`
        : "",
      metadata.documented_preprocessing ? "" : "> Handoff note: ONNX ModelProto metadata_props and doc_string are parsed and preserved, but ONNX does not type them as an executable preprocessing contract. Deployment reviewers must obtain that contract separately, including the production decoder, resize, color, normalization, and tensor-packing specification, before treating runtime or accuracy results as reproducible.",
    ].filter(Boolean).join("\n");
  }
  const processRows = (metadata.input_process_units || []).slice(0, 24).map((unit) => [
    unit.scope === "input_tensor" ? `Input ${unit.input_ordinal ?? "?"} ${code(unit.tensor_name || "-")}` : "Subgraph input pipeline",
    `${unit.options_type || "UNKNOWN"} (${formatNumber(unit.options_type_code || 0)})`,
    unit.status || "not assessed",
    unit.options_type === "NormalizationOptions"
      ? `mean [${(unit.mean || []).map((value) => formatCompactNumber(value)).join(", ")}], std [${(unit.std || []).map((value) => formatCompactNumber(value)).join(", ")}]`
      : (unit.associated_files || []).join(", ") || unit.detail || "-",
    unit.detail || "-",
  ]);
  const associatedRows = (metadata.output_associated_files || []).slice(0, 24).map((file) => [
    `Output ${formatNumber(file.output_ordinal)} ${code(file.tensor_name || "-")}`,
    code(file.name || "(unnamed)"),
    `${file.file_type || "UNKNOWN"} (${formatNumber(file.file_type_code || 0)})`,
    file.locale || "-",
    file.packed_status || "not assessed",
    `${file.payload_status || "not assessed"}; ${file.payload_bytes == null ? "bytes unavailable" : `${formatNumber(file.payload_bytes)} B`}; CRC ${file.crc32_verified == null ? "not assessed" : file.crc32_verified ? "verified" : "mismatch"}; SHA-256 ${file.payload_sha256 || "not available"}`,
    file.label_entry_count == null
      ? file.cardinality_status || "not assessed"
      : `${formatNumber(file.label_entry_count)} entries (${formatNumber(file.blank_label_entry_count || 0)} blank); shape [${(file.output_shape || []).join(", ")}]; ${file.cardinality_status || "not assessed"}; axes [${(file.matching_output_axes || []).join(", ")}]`,
    file.validation_detail || file.description || "-",
  ]);
  const packedRows = (metadata.packed_associated_files || []).slice(0, 24).map((file) => [
    code(file.name || "(unnamed)"),
    `${formatNumber(file.uncompressed_bytes || 0)} / ${formatNumber(file.compressed_bytes || 0)} B`,
    formatNumber(file.compression_method || 0),
    `${file.payload_status || "not assessed"}; decoded ${file.decoded_bytes == null ? "not available" : `${formatNumber(file.decoded_bytes)} B`}`,
    `0x${Number(file.crc32 || 0).toString(16).padStart(8, "0")} / ${file.crc32_verified == null ? "not assessed" : file.crc32_verified ? "verified" : "mismatch"}`,
    file.payload_sha256 || "not available",
    formatNumber(file.local_header_offset || 0),
    file.detail || "-",
  ]);
  return [
    "## Artifact Metadata & Signatures (OBSERVED)",
    markdownTable(["Item", "Present", "Detail"], [
      ["Metadata parser envelope", `${metadata.format || "tflite"} / ${metadata.status || "not emitted"}`, `${metadata.schema || "schema not emitted"}; ${metadata.detail || "not emitted"}`],
      ["Signature defs", metadata.has_signature_defs ? "yes" : "no", `${formatNumber(metadata.signature_count || 0)}; keys ${(metadata.signature_keys || []).join(", ") || "none"}`],
      ["TFLite Model Metadata entry", metadata.has_model_metadata ? "yes" : "no", `${formatNumber(metadata.model_metadata_entry_count || 0)} TFLITE_METADATA entry; Model.metadata names ${(metadata.metadata_entries || []).join(", ") || "none"}`],
      ["TFLite conversion metadata", `${formatNumber(metadata.conversion_metadata_entry_count || 0)} / ${metadata.conversion_metadata_status || "not assessed"}`, `TensorFlow ${metadata.converter_tensorflow_version || "not declared"}; converter API ${metadata.converter_api_version ?? "not declared"}; source model ${metadata.converter_model_type || "not declared"}; optimization modes ${(metadata.converter_optimization_modes || []).join(", ") || "not declared"}; schema tensorflow/tensorflow@${metadata.conversion_metadata_schema_source_commit || "not pinned"}/${metadata.conversion_metadata_schema_source_file || "not emitted"}; SHA-256 ${metadata.conversion_metadata_schema_sha256 || "not emitted"}`],
      ["Metadata schema / parser floor", metadata.metadata_schema_identifier || "not present", metadata.metadata_min_parser_version || "not declared"],
      ["Metadata model identity", metadata.metadata_model_name || "not declared", `version ${metadata.metadata_model_version || "not declared"}; author ${metadata.metadata_author || "not declared"}; license ${metadata.metadata_license || "not declared"}`],
      ["Metadata model description", metadata.metadata_model_description ? "yes" : "no", metadata.metadata_model_description || "none"],
      ["TFLite model description", metadata.has_description ? "yes" : "no", String(metadata.description || "none").replace(/\.{2,}/g, ".")],
      ["Subgraph / tensor metadata", formatNumber(metadata.subgraph_metadata_count || 0), `input ${formatNumber(metadata.input_tensor_metadata_count || 0)} (${formatNumber(metadata.described_input_tensor_count || 0)} named/described); output ${formatNumber(metadata.output_tensor_metadata_count || 0)} (${formatNumber(metadata.described_output_tensor_count || 0)} named/described)`],
      ["Explicit input process units", formatNumber(metadata.input_process_unit_count || 0), `${formatNumber(metadata.recognized_input_process_unit_count || 0)} assessed / ${formatNumber(metadata.invalid_input_process_unit_count || 0)} invalid / ${formatNumber(metadata.unrecognized_input_process_unit_count || 0)} unsupported; normalization ${formatNumber(metadata.normalization_unit_count || 0)}`],
      ["Documented preprocessing contract", metadata.documented_preprocessing ? "yes" : "no", metadata.preprocessing_contract_status || "not assessed"],
      ["Packed associated-file archive", metadata.associated_file_archive_status || "not assessed", `${formatNumber(metadata.packed_associated_file_count || 0)} central-directory file(s); payloads ${formatNumber(metadata.payload_verified_file_count || 0)} verified / ${formatNumber(metadata.payload_invalid_file_count || 0)} invalid / ${formatNumber(metadata.payload_unsupported_file_count || 0)} unsupported; ${metadata.associated_file_archive_detail || "no archive detail"}`],
      ["Output associated files", `${formatNumber(metadata.output_associated_file_count || 0)} declared / ${formatNumber(metadata.verified_output_associated_file_count || 0)} payload-verified / ${formatNumber(metadata.missing_output_associated_file_count || 0)} proven missing`, "Verification cross-checks central and local ZIP headers, decodes stored/DEFLATE payloads within bounded memory, recomputes CRC32, and binds a SHA-256 to decoded content."],
      ["Output label mappings", `${formatNumber(metadata.output_label_file_count || 0)} declared / ${formatNumber(metadata.verified_output_label_file_count || 0)} semantically verified / ${formatNumber(metadata.missing_output_label_file_count || 0)} proven missing / ${formatNumber(metadata.invalid_output_label_file_count || 0)} invalid or unresolved`, `${formatNumber(metadata.verified_output0_label_file_count || 0)} verified output-0 label mapping(s); TENSOR_AXIS_LABELS cardinality ${formatNumber(metadata.label_cardinality_match_count || 0)} unique match / ${formatNumber(metadata.label_cardinality_mismatch_count || 0)} mismatch / ${formatNumber(metadata.label_cardinality_ambiguous_count || 0)} ambiguous / ${formatNumber(metadata.label_cardinality_unresolved_count || 0)} unresolved; ${metadata.output_semantics_documented ? "output semantic mapping is documented" : "no payload- and cardinality-verified output-0 label mapping"}`],
    ]),
    processRows.length ? "### Input Process Units" : "",
    processRows.length ? markdownTable(["Scope", "Options", "Status", "Contract", "Validation"], processRows) : "",
    associatedRows.length ? "### Output Associated File Declarations" : "",
    associatedRows.length ? markdownTable(["Tensor", "File", "Type", "Locale", "Binding", "Payload integrity", "Labels / cardinality", "Validation"], associatedRows) : "",
    packedRows.length ? "### Packed Associated-File Central Directory" : "",
    packedRows.length ? markdownTable(["File", "Uncompressed / compressed", "Method", "Payload", "CRC32 declaration / recomputation", "Decoded SHA-256", "Local header offset", "Validation"], packedRows) : "",
    metadata.documented_preprocessing ? "" : "> Handoff note: Model Metadata presence alone is not treated as a preprocessing contract. An explicit, parseable input ProcessUnit is required. Deployment reviewers must obtain that contract separately, including the production decoder, resize, color, normalization, and tensor-packing specification.",
  ].filter(Boolean).join("\n");
}

function deadChannelDetailsMarkdown(weightIntegrity = {}) {
  const details = Array.isArray(weightIntegrity.zero_kernel_slice_details) ? weightIntegrity.zero_kernel_slice_details : [];
  if (!details.length) return "";
  return markdownTable(["Tensor", "Shape", "Near-zero decoded / exact-zero stored channels", "Scale/ZP sample", "Bias real / raw INT32", "Fused activation", "Downstream/residual", "Functional status"], details.slice(0, 8).map((item) => [
    code(item.tensor_name || `T${item.tensor_index}`),
    (item.shape || item.tensor_shape || []).join("x") || "-",
    `${formatNumber(item.channel_count || 0)}: ${(item.channels || []).slice(0, 24).join(", ")}${(item.channel_count || 0) > 24 ? ", ..." : ""} / exact ${formatNumber(item.exact_zero_channel_count || 0)}: ${(item.exact_zero_channels || []).slice(0, 24).join(", ") || "none"}`,
    `${(item.scale_sample || []).slice(0, 4).map((v) => Number(v).toExponential(3)).join(", ") || "-"} / ${(item.zero_point_sample || []).slice(0, 4).join(", ") || "-"}`,
    item.bias_tensor_name
      ? `${code(item.bias_tensor_name)} ${item.bias_dtype || ""}; real ${(item.bias_value_sample || []).slice(0, 4).map((v) => Number(v).toExponential(3)).join(", ") || "not decoded"}; raw ${(item.bias_code_sample || []).slice(0, 4).map((v) => formatNumber(v)).join(", ") || "not decoded"}; INT32 use ${(item.bias_int32_utilization_sample || []).slice(0, 4).map((v) => formatPercent(v)).join(", ") || "not assessed"}`
      : "not decoded or not present",
    item.fused_activation || "-",
    `${(item.next_consumers || []).join(" / ") || "no direct downstream op listed"}; ${item.residual_path || "residual not assessed"}`,
    item.functional_status || "NOT_ASSESSABLE from kernel weights alone",
  ]));
}

function onnxQuantGridDetailsMarkdown(weightIntegrity = {}) {
  const details = Array.isArray(weightIntegrity.quant_grid_details) ? weightIntegrity.quant_grid_details : [];
  if (!details.length) return "";
  return [
    "### Quantized Kernel Grid Ledger",
    markdownTable(["Tensor", "Storage", "Shape", "Logical / stored / implicit", "Used / legal levels", "Grid utilization", "Endpoint elements", "Saturation", "Review"], details.map((item) => [
      code(item.tensor_name),
      item.storage_kind || "tensor_proto",
      (item.shape || []).join("x") || "scalar",
      `${formatNumber(item.elements_scanned)} / ${formatNumber(item.stored_values_decoded ?? item.elements_scanned)} / ${formatNumber(item.implicit_zero_elements || 0)}`,
      `${formatNumber(item.unique_integer_levels)} / ${formatNumber(item.legal_integer_levels)}`,
      formatPercent(item.grid_utilization),
      formatNumber(item.endpoint_elements),
      formatPercent(item.saturation_ratio),
      [item.low_utilization_review ? "low utilization" : "", item.saturation_review ? "endpoint saturation" : ""].filter(Boolean).join(" / ") || "none",
    ])),
    `> ${details[0].formula}`,
  ].join("\n");
}

function onnxWeightIntegrityConclusion(weightIntegrity = {}) {
  const conservedElementCount = weightIntegrity.elements_scanned;
  const logicalElementCount = weightIntegrity.logical_elements_assessed ?? conservedElementCount;
  if (weightIntegrity.status !== "assessed") {
    return "Initializer-value integrity was not assessable because no supported embedded or verified-external numeric initializer payload was decoded.";
  }
  const anomalies = [
    Number(weightIntegrity.nan_tensors || 0) ? `${formatNumber(weightIntegrity.nan_tensors)} NaN tensor(s)` : "",
    Number(weightIntegrity.inf_tensors || 0) ? `${formatNumber(weightIntegrity.inf_tensors)} Inf tensor(s)` : "",
    Number(weightIntegrity.all_zero_tensors || 0) ? `${formatNumber(weightIntegrity.all_zero_tensors)} all-zero tensor(s)` : "",
    Number(weightIntegrity.zero_kernel_slice_count || 0) ? `${formatNumber(weightIntegrity.zero_kernel_slice_count)} all-zero kernel slice(s)` : "",
  ].filter(Boolean);
  const coverage = weightIntegrity.coverage_status === "complete"
    ? "complete for available numeric initializers"
    : `partial; ${formatNumber(weightIntegrity.initializer_tensors_unassessed)} external or unsupported tensor(s) remain unassessed`;
  return `Initializer-value integrity assessed for ${formatNumber(weightIntegrity.weight_tensors_scanned)} tensor(s) / ${formatNumber(logicalElementCount)} logical scalar element(s) (${formatNumber(weightIntegrity.stored_weight_values_decoded ?? conservedElementCount)} stored value(s) decoded, ${formatNumber(weightIntegrity.implicit_zero_elements || 0)} sparse implicit zero(s)); coverage ${coverage}. ${anomalies.length ? `Observed ${anomalies.join(", ")}.` : "No implemented numerical-integrity anomaly was observed."}`;
}

function onnxWeightIntegrityCompletenessRows(weightIntegrity = {}) {
  const assessed = weightIntegrity.status === "assessed";
  return [
    ["Initializer value decoding", assessed
      ? `Implemented for ${formatNumber(weightIntegrity.weight_tensors_scanned)} available dense TensorProto or validated SparseTensorProto initializer(s); ${formatNumber(weightIntegrity.stored_weight_values_decoded ?? weightIntegrity.elements_scanned)} stored value(s) decoded plus ${formatNumber(weightIntegrity.implicit_zero_elements || 0)} exact sparse implicit zero(s); ${formatNumber(weightIntegrity.initializer_tensors_unassessed)} incomplete-external/invalid/unsupported tensor(s) not assessed`
      : "Not assessable; no supported complete embedded or verified-external numeric initializer payload was decoded"],
    ["Weight integrity", assessed
      ? `Assessed over logical initializer values: NaN/Inf, all-zero tensors, sparsity, magnitude, and eligible Conv/Gemm/MatMul output-axis kernel slices; coverage ${weightIntegrity.coverage_status || "unknown"}`
      : "Not assessable for initializer values"],
  ];
}

function tensorAt(analysis, id) {
  return Number.isInteger(id) && id >= 0 ? (analysis?.tensors || [])[id] : null;
}

function tensorScale0(tensor) {
  return Number((tensor?.scale_sample || [])[0] || 0);
}

function quantizationModeText(tensor) {
  const scales = Number(tensor?.quant_scales || 0);
  if (scales > 1) return `per-channel/per-axis (${formatNumber(scales)} scales, qdim=${tensor?.quantized_dimension ?? "?"})`;
  if (scales === 1) return "per-tensor (single scale)";
  return "missing quantization metadata";
}

function quantizedTensorRangeText(tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const scale = tensorScale0(tensor);
  const zp = Number((tensor?.zero_point_sample || [0])[0] ?? 0);
  if (!["INT8", "UINT8"].includes(dtype) || !(scale > 0)) return "not applicable";
  const lower = dtype === "INT8" ? -128 : 0;
  const upper = dtype === "INT8" ? 127 : 255;
  return `[${(scale * (lower - zp)).toExponential(4)}, ${(scale * (upper - zp)).toExponential(4)}]`;
}

function tensorQuantizationText(tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const scale = tensorScale0(tensor);
  if (["INT8", "UINT8"].includes(dtype) && scale > 0) {
    const zp = (tensor?.zero_point_sample || [0])[0] ?? 0;
    return `${quantizationModeText(tensor)}; scale=${scale}; zp=${zp}; range=${quantizedTensorRangeText(tensor)}; real=scale*(q-zp)`;
  }
  return Number(tensor?.quant_scales || 0) ? `${tensor.quant_scales} scale(s)` : "none";
}

function outputProducerSummary(analysis, output) {
  const outputIndex = Number.isInteger(output?.index) ? output.index : Number(output?.tensor_index);
  const producer = Number.isFinite(outputIndex)
    ? (analysis?.ops || []).find((op) => (op.outputs || []).includes(outputIndex))
    : null;
  if (!producer) return "producer not found; probability semantics NOT_ASSESSABLE";
  const name = String(producer.name || "").toUpperCase();
  return `${name} #${padOp(producer.index)}; probability semantics ${name === "SOFTMAX" ? "DERIVED from terminal SOFTMAX" : "NOT_ASSESSABLE, no terminal SOFTMAX observed"}`;
}

function missingQuantMetadataSummary(analysis) {
  const mostlyQuantized = Number(analysis?.quantized_tensors || 0) > 0
    && Number(analysis?.quantized_tensors || 0) < Number(analysis?.tensor_count || (analysis?.tensors || []).length);
  if (!mostlyQuantized) return "none";
  const rows = (analysis?.tensors || [])
    .filter((tensor) => Number(tensor.quant_scales || 0) <= 0)
    .slice(0, 8)
    .map((tensor) => {
      const role = quantizationMissingRole(analysis, tensor);
      return `T${tensor.index} ${tensor.name || "-"} ${tensor.dtype || "-"}; role=${role}; quantization metadata ${role.includes("shape/configuration") ? "not expected" : "absent"}`;
    });
  return rows.length ? rows.join(" / ") : "none";
}

function quantizationMissingRole(analysis, tensor) {
  const consumers = (analysis?.ops || []).filter((op) => (op.inputs || []).includes(Number(tensor?.index)));
  const names = consumers.map((op) => String(op.name || "").toUpperCase());
  const dtype = String(tensor?.dtype || "").toUpperCase();
  if (["RESHAPE", "SHAPE", "STRIDED_SLICE", "PACK", "CONCATENATION"].some((name) => names.includes(name))) {
    return "shape/configuration tensor";
  }
  if (["INT32", "INT64"].includes(dtype) && (tensor?.constant_buffer || (tensor?.shape || []).reduce((p, d) => p * Math.max(1, Number(d) || 1), 1) <= 8)) {
    return "shape/configuration tensor";
  }
  return tensor?.constant_buffer ? "constant tensor" : "activation tensor";
}

function quantMagnitudeBound(tensor) {
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const [lower, upper] = dtype === "INT8" ? [-128, 127] : dtype === "UINT8" ? [0, 255] : [0, 0];
  if (upper === 0 && lower === 0) return 0;
  const rawZps = tensor?.zero_point_sample || [];
  const zps = (rawZps.length ? rawZps : [0]).map((zp) => Number(zp || 0));
  return Math.max(...zps.map((zp) => Math.max(Math.abs(lower - zp), Math.abs(upper - zp))));
}

function kernelAccumulationTerms(op, weight) {
  const shape = (weight?.shape || []).map((dim) => Math.max(1, Number(dim) || 1));
  const name = String(op?.name || "").toUpperCase();
  if (name === "DEPTHWISE_CONV_2D" && shape.length >= 4) return shape[1] * shape[2];
  if (shape.length >= 2) return shape.slice(1).reduce((product, dim) => product * dim, 1);
  return 0;
}

function weightIntegrityMarkdown(analysis, weightIntegrity = {}) {
  const serialized = analysis?.tensor_numerical_integrity;
  if (serialized) {
    const records = Array.isArray(serialized.tensor_records) ? serialized.tensor_records : [];
    const assessedRows = records.filter((row) => row.status === "assessed_full_payload")
      .sort((left, right) => Number(right.byte_length || 0) - Number(left.byte_length || 0)
        || Number(left.tensor_index || 0) - Number(right.tensor_index || 0));
    const encodingRows = assessedRows.filter((row) => row.quantization_code_levels_legal != null || row.encoded_codebook_entries_legal != null);
    const exceptionRows = records.filter((row) => row.status !== "assessed_full_payload"
      || Number(row.nan_value_count || 0) + Number(row.positive_infinity_value_count || 0)
        + Number(row.negative_infinity_value_count || 0) + Number(row.invalid_encoding_value_count || 0) > 0);
    const source = serialized.decoder_source || {};
    return [
      `## Serialized Tensor Numerical Integrity (${serialized.status === "assessed" ? "OBSERVED/DERIVED" : "PARTIAL"})`,
      markdownTable(["Check", "Result"], [
        ["Full payload scan", `${formatNumber(serialized.assessed_tensor_count || 0)}/${formatNumber(serialized.tensor_count || 0)} tensors; ${formatNumber(serialized.assessed_tensor_bytes || 0)}/${formatNumber(serialized.declared_tensor_bytes || 0)} declared tensor bytes`],
        ["Byte conservation", serialized.byte_conservation_status || "not assessed"],
        ["Decoded values", formatNumber(serialized.decoded_value_count || 0)],
        ["Non-finite decoded values", formatNumber(serialized.nonfinite_value_count || 0)],
        ["Invalid encoded values", formatNumber(serialized.invalid_encoding_value_count || 0)],
        ["Blocks with non-finite scales", formatNumber(serialized.nonfinite_scale_block_count || 0)],
        ...(analysis.format === "gguf" ? [["Serialized endian", analysis.gguf?.endianness || "not bound"]] : []),
        ["Exact-zero values", formatNumber(serialized.exact_zero_value_count || 0)],
        ["All-zero / constant tensors", `${formatNumber(serialized.all_zero_tensor_count || 0)} / ${formatNumber(serialized.constant_tensor_count || 0)}`],
        ["Unassessed tensors / bytes", `${formatNumber(serialized.unassessed_tensor_count || 0)} / ${formatNumber(serialized.unassessed_tensor_bytes || 0)} B`],
        ["Decoder provenance", source.source_commit
            ? `${source.repository}@${source.source_commit}; ${source.source} SHA-256 ${source.source_sha256}`
            + `${source.block_layout_source ? `; ${source.block_layout_source} SHA-256 ${source.block_layout_source_sha256}` : ""}`
            + `${source.numeric_format_source ? `; ${source.numeric_format_source} SHA-256 ${source.numeric_format_source_sha256}` : ""}`
            + `${source.format_specification_commit ? `; ${source.format_specification_repository}@${source.format_specification_commit}/${source.format_specification_source} SHA-256 ${source.format_specification_source_sha256}` : ""}`
          : source.commit ? `${source.repository}@${source.commit}; ${source.tensor_source || "safetensors/src/tensor.rs"} SHA-256 ${source.tensor_rs_sha256 || "not emitted"}`
            + `${source.torch_binding_source ? `; ${source.torch_binding_source} SHA-256 ${source.torch_binding_source_sha256 || "not emitted"}` : ""}`
            + `${source.pytorch_commit ? `; ${source.pytorch_repository || "pytorch/pytorch"}@${source.pytorch_commit}; ${(source.pytorch_sources || []).map((row) => `${row.path} SHA-256 ${row.sha256}`).join("; ")}` : ""}`
            + `${source.ocp_mx_version ? `; OCP MX ${source.ocp_mx_version}` : ""}`
            : "declared scalar dtype contract"],
      ]),
      assessedRows.length ? "### Full-Payload Finite Moment Ledger" : "",
      assessedRows.length ? markdownTable(["Tensor", "Dtype / shape", "Bytes / values", "Finite range", "Mean / RMS", "Zero / distinct", "Posture", "Payload SHA-256"], assessedRows.slice(0, 64).map((row) => [
        code(row.tensor_name || `T${row.tensor_index}`),
        `${row.dtype || "UNKNOWN"} / ${(row.shape || []).join("x") || "scalar"}`,
        `${formatNumber(row.byte_length || 0)} / ${formatNumber(row.value_count || 0)}`,
        row.minimum_finite != null || row.minimum_finite_decimal != null ? `${row.minimum_finite ?? row.minimum_finite_decimal} to ${row.maximum_finite ?? row.maximum_finite_decimal}` : "not assessed",
        row.arithmetic_mean_finite == null ? row.moment_status || "not assessed" : `${row.arithmetic_mean_finite} / ${row.rms_finite}`,
        `${formatNumber(row.zero_value_count || 0)} / ${row.distinct_finite_values == null ? row.distinct_value_status || "not assessed" : formatNumber(row.distinct_finite_values)}`,
        row.all_zero ? "all-zero" : row.constant_finite ? "constant" : "variable",
        code(row.payload_sha256 || "not emitted"),
      ])) : "",
      assessedRows.length > 64 ? `> ${formatNumber(assessedRows.length - 64)} additional full-payload tensor row(s) remain in structured evidence.` : "",
      encodingRows.length ? "### Encoded Grid Utilization Ledger" : "",
      encodingRows.length ? markdownTable(["Tensor", "Encoding", "Stored grid", "Utilization", "Blocks / non-finite scales"], encodingRows.slice(0, 64).map((row) => {
        const used = row.encoded_codebook_entries_used ?? row.quantization_code_levels_used;
        const legal = row.encoded_codebook_entries_legal ?? row.quantization_code_levels_legal;
        const utilization = row.encoded_codebook_utilization ?? row.quantization_code_utilization;
        return [
          code(row.tensor_name || `T${row.tensor_index}`),
          row.dtype || "UNKNOWN",
          row.encoded_codebook_name ? `${row.encoded_codebook_name}: ${formatNumber(used)}/${formatNumber(legal)} entries` : `${formatNumber(used)}/${formatNumber(legal)} scalar codes`,
          utilization == null ? "not assessed" : formatPercent(utilization),
          `${formatNumber(row.encoded_block_count || 0)} / ${formatNumber(row.nonfinite_scale_block_count || 0)}`,
        ];
      })) : "",
      encodingRows.length > 64 ? `> ${formatNumber(encodingRows.length - 64)} additional encoded-grid row(s) remain in structured evidence.` : "",
      exceptionRows.length ? "### Payload Exception Ledger" : "",
      exceptionRows.length ? markdownTable(["Tensor", "Dtype / shape", "Status", "Values / zeros", "NaN / Inf / invalid", "Finite range", "SHA-256 or reason"], exceptionRows.slice(0, 48).map((row) => [
        code(row.tensor_name || `T${row.tensor_index}`),
        `${row.dtype || "UNKNOWN"} / ${(row.shape || []).join("x") || "scalar"}`,
        row.status || "not assessed",
        row.value_count == null ? "not assessed" : `${formatNumber(row.value_count)} / ${formatNumber(row.zero_value_count || 0)}`,
        row.nan_value_count == null ? "not assessed" : `${formatNumber(row.nan_value_count || 0)} / ${formatNumber(Number(row.positive_infinity_value_count || 0) + Number(row.negative_infinity_value_count || 0))} / ${formatNumber(row.invalid_encoding_value_count || 0)}`,
        row.minimum_finite != null || row.minimum_finite_decimal != null ? `${row.minimum_finite ?? row.minimum_finite_decimal} to ${row.maximum_finite ?? row.maximum_finite_decimal}` : "not assessed",
        row.payload_sha256 || row.reason || "not emitted",
      ])) : "",
      exceptionRows.length > 48 ? `> ${formatNumber(exceptionRows.length - 48)} additional exception row(s) remain in structured evidence.` : "",
      `> OBSERVED payload bytes are hashed over each exact declared tensor range. Scalar statistics are derived from declared dtype semantics; GGUF block values are DERIVED from the pinned llama.cpp dequantization source and the serialized file endian. Source-defined multi-byte fields follow file endian while byte-packed code lanes retain their declared byte order; Q8_K redundant block sums are checked against decoded quants. SafeTensors F4/FP8 values are DERIVED from the pinned SafeTensors/PyTorch representation contract. ${source.subbyte_boundary || ""} No execution-graph, task-accuracy, or runtime-kernel effect is inferred for weight-only containers.`.trim(),
    ].filter(Boolean).join("\n");
  }
  if (analysis?.format === "coreml" && String(weightIntegrity.schema || "").startsWith("deepbom.coreml.")) {
    const parameters = Array.isArray(weightIntegrity.parameters) ? weightIntegrity.parameters : [];
    const review = parameters.filter((row) => !String(row.numerical_integrity?.status || "").startsWith("assessed")
      || Number(row.numerical_integrity?.nonfinite_count || 0) > 0 || row.numerical_integrity?.all_zero === true);
    return [
      `## Core ML Serialized Constant Numerical Integrity (${weightIntegrity.status === "assessed" ? "OBSERVED/DERIVED" : "PARTIAL"})`,
      markdownTable(["Check", "Result"], [
        ["Parameter coverage", `${formatNumber(weightIntegrity.assessed_parameter_count || 0)}/${formatNumber(weightIntegrity.parameter_count || 0)} serialized constants`],
        ["Payload byte conservation", `${formatNumber(weightIntegrity.assessed_payload_bytes || 0)}/${formatNumber(weightIntegrity.payload_bytes || 0)} B; ${weightIntegrity.payload_byte_conservation ? "complete" : "partial"}`],
        ["Non-finite decoded values", formatNumber(weightIntegrity.nonfinite_value_count || 0)],
        ["Exact all-zero parameters", formatNumber(weightIntegrity.all_zero_parameter_count || 0)],
        ["Binding", "Each payload is bound to a serialized legacy layer/WeightParams role, classical-model coefficient/threshold group, pipeline stage, or ML Program SSA tensor/blob offset; applicable parent cardinalities are validated before reporting"],
      ]),
      review.length ? "### Core ML Parameter Review Ledger" : "",
      review.length ? markdownTable(["Layer / role", "Storage", "Values / bytes", "Status", "NaN / Inf", "Finite range", "Payload SHA-256"], review.slice(0, 64).map((row) => [
        `${code(row.layer_name || row.tensor_name || `tensor_${row.tensor_index ?? row.layer_index ?? "?"}`)} / ${row.role || "constant"}`,
        row.storage || "unknown",
        `${row.value_count == null ? "not assessed" : formatNumber(row.value_count)} / ${formatNumber(row.byte_length || 0)} B`,
        row.numerical_integrity?.status || "not assessed",
        `${formatNumber(row.numerical_integrity?.nan_count || 0)} / ${formatNumber(Number(row.numerical_integrity?.positive_infinity_count || 0) + Number(row.numerical_integrity?.negative_infinity_count || 0))}`,
        row.numerical_integrity?.finite_min == null ? "not assessed" : `${row.numerical_integrity.finite_min} to ${row.numerical_integrity.finite_max}`,
        row.numerical_integrity?.payload_sha256 || "not emitted",
      ])) : "",
      review.length > 64 ? `> ${formatNumber(review.length - 64)} additional rows remain in engineering_evidence.json.` : "",
      "> OBSERVED payload bytes and hashes come from legacy WeightParams, classical-model FLOAT64 arrays, nested pipeline stages, or package-bound ML Program blob v2 ranges. Scalar integrity and packed-code utilization are DERIVED from pinned Core ML schemas and tensor cardinality; runtime placement and task accuracy remain separate evidence classes.",
    ].filter(Boolean).join("\n");
  }
  if (isOnnxAnalysis(analysis)) {
    const assessed = weightIntegrity.status === "assessed";
    if (assessed) {
      return [
        "## Weight Integrity (OBSERVED)",
        markdownTable(["Check", "Result"], [
          ["Evidence class", weightIntegrity.evidence_class || "OBSERVED/DERIVED"],
          ["Initializer inventory present", `${formatNumber(weightIntegrity.initializer_tensors_present ?? weightIntegrity.weight_tensors_scanned)} tensor(s), ${formatNumber(weightIntegrity.initializer_elements_present ?? weightIntegrity.elements_scanned)} scalar element(s); decoder ${weightIntegrity.initializer_value_decoding || "available numeric TensorProto"}`],
          ["Storage kinds", `${formatNumber(weightIntegrity.dense_initializer_tensors || 0)} dense TensorProto / ${formatNumber(weightIntegrity.sparse_initializer_tensors || 0)} SparseTensorProto initializer(s)`],
          ["Coverage", `${formatNumber(weightIntegrity.weight_tensors_scanned)} initializer tensor(s), ${formatNumber(weightIntegrity.logical_elements_assessed ?? weightIntegrity.elements_scanned)} logical scalar element(s) assessed; ${formatNumber(weightIntegrity.stored_weight_values_decoded ?? weightIntegrity.elements_scanned)} stored value(s) decoded + ${formatNumber(weightIntegrity.implicit_zero_elements || 0)} exact sparse implicit zero(s); ${formatNumber(weightIntegrity.initializer_tensors_unassessed)} incomplete-external/invalid/unsupported tensor(s) not assessed`],
          ["NaN / Inf tensors", `${formatNumber(weightIntegrity.nan_tensors)} / ${formatNumber(weightIntegrity.inf_tensors)}`],
          ["All-zero tensors", formatNumber(weightIntegrity.all_zero_tensors)],
          ["Mean near-zero sparsity", formatPercent(weightIntegrity.mean_sparsity)],
          ["Large-magnitude / high-sparsity tensors", `${formatNumber(weightIntegrity.large_magnitude_tensors || 0)} / ${formatNumber(weightIntegrity.high_sparsity_tensors || 0)}`],
          ["Max |decoded initializer|", Number(weightIntegrity.max_abs_weight || 0).toExponential(3)],
          ["Kernel tensors / output channels evaluated", `${formatNumber(weightIntegrity.eligible_kernel_tensors_scanned)} / ${formatNumber(weightIntegrity.output_channels_evaluated)}`],
          ["All-zero kernel output slices", `${formatNumber(weightIntegrity.zero_kernel_slice_count)} across ${formatNumber(weightIntegrity.zero_kernel_slice_tensors)} tensor(s)`],
          ["Quantization grid / endpoint saturation", Number(weightIntegrity.quantized_constant_tensors_scanned || 0) > 0
            ? `${formatNumber(weightIntegrity.quantized_constant_tensors_scanned)} bound 8-bit kernel tensor(s); minimum level utilization ${formatPercent(weightIntegrity.min_grid_utilization)}; maximum qmin/qmax saturation ${formatPercent(weightIntegrity.max_saturation_percent)}`
            : "NOT_APPLICABLE; no available 8-bit kernel initializer had a valid Q/DQ or QLinear parameter binding"],
        ]),
        onnxQuantGridDetailsMarkdown(weightIntegrity),
        deadChannelDetailsMarkdown(weightIntegrity),
        `> OBSERVED for embedded and verified-external numeric TensorProto payloads and ONNX-defined sparse implicit zeros; grid and dequantized-value metrics are DERIVED from bound ONNX scale/zero-point parameters. ${weightIntegrity.detail} Functional channel inactivity remains NOT_ASSESSABLE from kernel values alone.`,
      ].join("\n");
    }
    return [
      "## Weight Integrity (NOT_ASSESSABLE)",
      markdownTable(["Check", "Result"], [
        ["Status", "NOT_ASSESSABLE"],
        ["Evidence class", weightIntegrity.evidence_class || "NOT_ASSESSABLE"],
        ["Initializer tensors present", formatNumber(weightIntegrity.initializer_tensors_present ?? analysis.size_breakdown?.constant_tensor_count ?? 0)],
        ["Initializer scalar elements present", formatNumber(weightIntegrity.initializer_elements_present ?? analysis.size_breakdown?.stored_scalar_elements ?? 0)],
        ["Storage kinds", `${formatNumber(weightIntegrity.dense_initializer_tensors || 0)} dense TensorProto / ${formatNumber(weightIntegrity.sparse_initializer_tensors || 0)} SparseTensorProto initializer(s)`],
        ["Initializer value decoding", weightIntegrity.initializer_value_decoding || "no supported numeric payload was available"],
        ["Coverage status", weightIntegrity.coverage_status || "not assessed"],
        ["Assessed / unassessed initializer coverage", `${formatNumber(weightIntegrity.weight_tensors_scanned || 0)} tensor(s), ${formatNumber(weightIntegrity.logical_elements_assessed || 0)} logical scalar element(s) assessed / ${formatNumber(weightIntegrity.initializer_tensors_unassessed || 0)} tensor(s) unassessed`],
        ["Stored values / sparse implicit zeros assessed", `${weightIntegrity.stored_weight_values_decoded == null ? "not assessed" : formatNumber(weightIntegrity.stored_weight_values_decoded)} / ${weightIntegrity.implicit_zero_elements == null ? "not assessed" : formatNumber(weightIntegrity.implicit_zero_elements)}`],
        ["NaN / Inf / sparsity checks", "not performed"],
        ["All-zero kernel-slice checks", "not performed"],
        ["Interpretation", "No clean integrity result is emitted without a decodable embedded payload"],
      ]),
      `> NOT_ASSESSABLE: ${weightIntegrity.detail || "ONNX initializer values were inventoried, but no supported embedded or verified-external numeric payload was available for numerical integrity checks."}`,
    ].join("\n");
  }
  const eligibleKernelTensors = Number(weightIntegrity.eligible_kernel_tensors_scanned || 0);
  const evaluatedChannels = Number(weightIntegrity.output_channels_evaluated || 0);
  return [
    "## Weight Integrity (OBSERVED)",
    markdownTable(["Check", "Result"], [
      ["Status", String(weightIntegrity.status || "ok").toLowerCase() === "ok" ? "No anomalies detected by the implemented constant checks" : (weightIntegrity.status || "warn").toUpperCase()],
      ["Decodable constant tensors scanned", `${formatNumber(weightIntegrity.weight_tensors_scanned || 0)} (${formatNumber(weightIntegrity.elements_scanned || 0)} elements)`],
      ["Quantized constants decoded", `${plural(weightIntegrity.quantized_constant_tensors_scanned || 0, "INT8/UINT8 tensor")} dequantized with artifact scale/zero-point metadata`],
      ["Quantization grid measured values", weightIntegrity.quant_grid_detail || "not emitted"],
      ["Minimum grid utilization / maximum endpoint saturation", Number(weightIntegrity.quantized_constant_tensors_scanned || 0) > 0
        ? `${formatPercent(weightIntegrity.min_grid_utilization || 0)} / ${(Number(weightIntegrity.max_saturation_percent || 0) * 100).toFixed(2)}%`
        : "NOT_APPLICABLE; no decoded 8-bit constant tensor"],
      ["Threshold-eligible grid minimum", Number(weightIntegrity.threshold_eligible_quantized_constant_tensors || 0) > 0
        ? `${formatPercent(weightIntegrity.min_threshold_eligible_grid_utilization)} across ${formatNumber(weightIntegrity.threshold_eligible_quantized_constant_tensors)} tensor(s) with at least 256 elements`
        : "N/A; no decoded 8-bit constant has at least 256 elements"],
      ["Grid alert threshold", "HEURISTIC: review when an 8-bit constant tensor with at least 256 elements uses <25% of levels, or endpoint saturation exceeds 1%; methodology threshold, not task-accuracy evidence"],
      ["Low grid utilization / endpoint saturation", `${formatNumber(weightIntegrity.low_grid_utilization_tensors || 0)} tensor(s) / ${formatNumber(weightIntegrity.saturated_quantized_tensors || 0)} tensor(s)`],
      ["NaN / Inf tensors", `${weightIntegrity.nan_tensors || 0} / ${weightIntegrity.inf_tensors || 0}`],
      ["All-zero tensors", `${weightIntegrity.all_zero_tensors || 0}`],
      ["Channel-slice zero checks", eligibleKernelTensors > 0 ? "near-zero decoded values and exact-zero stored centered codes evaluated separately for Conv/FC/depthwise kernel layouts" : "not assessed; no eligible Conv/FC/depthwise kernel layout was decoded"],
      ["Kernel tensors evaluated", formatNumber(eligibleKernelTensors)],
      ["Output channels evaluated", formatNumber(evaluatedChannels)],
      ["Near-zero decoded kernel output slices", `${formatNumber(weightIntegrity.zero_kernel_slice_count || 0)} across ${formatNumber(weightIntegrity.zero_kernel_slice_tensors || 0)} tensor(s)`],
      ["Exact-zero stored kernel output slices", `${formatNumber(weightIntegrity.exact_zero_kernel_slice_count || 0)} across ${formatNumber(weightIntegrity.exact_zero_kernel_slice_tensors || 0)} tensor(s)`],
      ["Max |decoded constant|", `${Number(weightIntegrity.max_abs_weight || 0).toExponential(2)}`],
      ["Large-magnitude tensors (|w|>1e4)", `${weightIntegrity.large_magnitude_tensors || 0}`],
      ["Mean near-zero sparsity", formatPercent(weightIntegrity.mean_sparsity || 0)],
      ["High-sparsity tensors (>50%)", `${weightIntegrity.high_sparsity_tensors || 0}`],
      ["Criteria", "near-zero means |x| < 1e-8; high-sparsity means >50% near-zero elements; near-zero decoded slice applies that threshold to every element; exact-zero stored slice requires every centered quantized code or stored float value to equal zero"],
      ["Analyzer detail", weightIntegrity.detail || "not emitted"],
    ]),
    deadChannelDetailsMarkdown(weightIntegrity),
    weightIntegrity.high_sparsity_tensors ? "> High sparsity may enable XNNPACK sparse kernels on supported runtimes." : "",
    "> OBSERVED: read directly from decodable constant bytes in the artifact. FLOAT32/FLOAT16 are read as stored; INT8/UINT8 constants preserve both stored centered-code exact-zero and dequantized near-zero classifications. Functional model-output inactivity remains separate from this kernel-slice evidence and requires the channel-vitality/downstream/representative-output chain.",
  ].join("\n");
}

function quantizationContractChecksMarkdown(analysis) {
  const checks = buildQuantizationContractChecks(analysis);
  if (checks.status === "not_applicable" && !checks.bias_scale) return "";
  const onnx = (analysis?.format || "tflite") === "onnx";
  const quantResearchCoverage = onnx
    ? null
    : analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const advancedInteger = onnx
    || ["full_integer", "mixed_integer"].includes(quantResearchCoverage?.artifact_class);
  const bias = checks.bias_scale;
  const residual = checks.residual_add;
  const zeroPoint = checks.weight_zero_point;
  const accumulator = checks.accumulator_bound;
  const kernel = checks.kernel_quantization;
  const io = checks.io_dequantization;
  const inputConvention = checks.input_quantization_convention;
  const missing = checks.missing_quantization_metadata;
  const qdq = checks.qdq_boundaries;
  const representable = checks.representable_kernel_channels;
  const ioText = (items, label) => items.length
    ? items.map((item) => `${label} ${item.ordinal}: ${item.dtype}, scale=${item.scale}, zp=${item.zero_point}, range=[${item.real_range.map((value) => value.toExponential(4)).join(", ")}], equation ${item.equation}`).join(" / ")
    : "none";
  const scaleRangeText = kernel.minimum_weight_scale != null
    ? `${kernel.minimum_weight_scale.toExponential(4)} to ${kernel.maximum_weight_scale.toExponential(4)} (${kernel.maximum_to_minimum_scale_ratio.toFixed(2)}x); ${formatNumber(kernel.per_tensor_kernel_tensors)} per-tensor / ${formatNumber(kernel.per_channel_kernel_tensors)} per-channel kernel tensor(s)`
    : "no quantized Conv/Depthwise/FC weight scales emitted";
  const parameter = checks.parameter_integrity;
  const binding = onnx ? analysis?.onnx_quantization_binding || null : null;
  const annotationBindings = binding ? (binding.bindings || []).filter((item) => item.binding_source === "graph_quantization_annotation") : [];
  const zeroPointText = onnx
    ? `${formatNumber(zeroPoint.violation_tensors)} out-of-range/type violation(s); ${formatNumber(zeroPoint.asymmetric_uint8_tensors)} legal asymmetric UINT8 weight tensor(s) observed; converter lineage is not inferred from legal ONNX zero-points`
    : `${formatNumber(zeroPoint.violation_tensors)} invalid/symmetric-INT8 violation(s); ${formatNumber(zeroPoint.asymmetric_uint8_tensors)} asymmetric UINT8 weight tensor(s) observed; converter lineage not assessable from artifact`;
  const biasText = bias.contract_kind === "onnx_implicit_qlinearconv_bias_scale"
    ? `${formatNumber(bias.checked_groups)} QLinearConv bias group(s), ${formatNumber(bias.checked_channels)} implicit scale channel(s); ${formatNumber(bias.mismatch_groups)} dtype/cardinality violation(s); formula ${bias.method}`
    : `${formatNumber(bias.checked_groups)} group(s), ${formatNumber(bias.checked_channels)} channel(s) checked; ${formatNumber(bias.mismatch_groups)} mismatch group(s); max relative error ${formatPercent(bias.maximum_relative_error)}; relative tolerance ${bias.relative_tolerance}`;
  const rows = [
      ["Artifact contract", checks.format_contract || "TFLite quantized operator metadata"],
      ...(binding ? [["ONNX binding ledger", `${binding.status || "not_assessed"}; evidence ${binding.evidence_class || "not emitted"}; ${formatNumber(binding.valid_binding_count || 0)}/${formatNumber(binding.binding_count || 0)} valid, ${formatNumber(binding.invalid_binding_count || 0)} invalid, ${formatNumber(binding.unresolved_binding_count || 0)} unresolved; ${formatNumber(binding.explicit_qdq_boundary_count || 0)} explicit Q/DQ boundary edge(s); ${formatNumber(binding.integer_compute_without_real_scale_count || 0)} integer compute op(s) without embedded real scale; schema ${binding.schema || "not emitted"}`]] : []),
      ...(binding ? [["GraphProto quantization annotations", `${formatNumber(binding.main_graph_annotation_count || 0)} main-graph mapping(s): ${formatNumber(binding.valid_annotation_count || 0)} valid / ${formatNumber(binding.invalid_annotation_count || 0)} invalid / ${formatNumber(binding.unresolved_annotation_count || 0)} unresolved; nested ${formatNumber(binding.nested_graph_annotation_count || 0)}; scope ${binding.annotation_scope_status || "not emitted"}; pinned source ${binding.annotation_source_sha256 || "not emitted"}; ${binding.annotation_source_ref || "not emitted"}`]] : []),
      ...(parameter ? [["ONNX parameter integrity", `${formatNumber(parameter.checked_bindings)} binding(s); ${formatNumber(parameter.invalid_bindings)} invalid, ${formatNumber(parameter.unresolved_bindings)} unresolved; integer compute ops without embedded real scale ${formatNumber(parameter.integer_compute_without_real_scale_count)}`]] : []),
      ["Contract integrity status", checks.contract_integrity_status.toUpperCase()],
      ["Quantization design review status", checks.quantization_design_review_status.toUpperCase()],
      ["Bias scale contract", biasText],
      ...(!onnx ? [[
        "Near-zero representable kernel channels",
        representable.status === "not_applicable"
          ? `NOT_APPLICABLE_TO_QUANTIZATION_SCHEME; 0 per-axis kernel channels are present, so this scale-vector detector has no denominator. This is not "0 defects"; decoded exact-zero/near-zero stored-slice and Channel Vitality checks remain independently applicable.`
          : `${formatNumber(representable.flagged_channels || 0)} of ${formatNumber(representable.assessed_channels || 0)} assessed per-axis channel(s) across ${formatNumber(representable.flagged_kernel_tensors || 0)} kernel tensor(s); exact range formula with HEURISTIC thresholds max |real weight| <=${Number(representable.maximum_representable_abs_threshold || 0).toExponential(1)} and tensor scale ratio >=${Number(representable.scale_outlier_ratio_threshold || 0).toExponential(1)}`,
      ]] : []),
      ...(!onnx ? [["Depthwise per-tensor quantization", `${formatNumber(kernel.depthwise_per_tensor_kernel_tensors)} DEPTHWISE_CONV_2D weight tensor(s) use single-scale quantization; per-channel kernel tensors ${formatNumber(kernel.per_channel_kernel_tensors)}`]] : []),
      ["Kernel weight quantization mode", scaleRangeText],
      ...(!onnx && advancedInteger ? [["Residual ADD input scale ratio", `${formatNumber(residual.checked_ops)} ADD op(s) checked; ${formatNumber(residual.review_ops)} op(s) >=${residual.review_threshold_ratio}x; max ratio ${residual.maximum_input_scale_ratio.toFixed(2)}x`]] : []),
      ["Weight zero-point contract", zeroPointText],
      ...(advancedInteger ? [["Accumulator dtype / full-code-domain bound", accumulator.bound_class === "exact_stored_weight_channel_integer_domain"
        ? `INT32; ${formatNumber(accumulator.checked_ops)} op(s) / ${formatNumber(accumulator.checked_channels)} output channel(s) exactly bounded from stored centered weights and bias; max INT32 utilization ${formatPercent(accumulator.maximum_int32_ratio)}; ${formatNumber(accumulator.overflow_risk_channels)} full-domain exceedance channel(s)`
        : `INT32; ${formatNumber(accumulator.checked_ops)} op(s) metadata-only bounded; max worst-case int32 ratio ${Number(accumulator.maximum_int32_ratio || 0).toFixed(4)}x; ${formatNumber(accumulator.overflow_risk_ops)} op(s) >=1.0x`]] : []),
      ...(advancedInteger && accumulator.bound_class === "exact_stored_weight_channel_integer_domain" ? [[
        "Exact accumulator width / headroom",
        `${formatNumber(accumulator.maximum_required_signed_bits)} signed bits required; minimum INT32 headroom ${formatNumber(accumulator.minimum_int32_headroom_bits)} bits; maximum metadata-only comparison ${formatPercent(accumulator.maximum_metadata_only_int32_ratio)}`,
      ]] : []),
      ...(!onnx && advancedInteger ? [["Kernel extremum witnesses", `${formatNumber(checks.kernel_extremum_witness.checked_channels || 0)} channel(s); ${formatNumber(checks.kernel_extremum_witness.canonical_witness_assignments || 0)} canonical term assignment(s); ${formatNumber(checks.kernel_extremum_witness.fixed_point_endpoint_evaluations || 0)} pinned fixed-point endpoint execution(s); ${formatNumber(checks.kernel_extremum_witness.build_mode_divergent_endpoints || 0)} default/single output-code difference(s)`]] : []),
      ...(!onnx && advancedInteger ? [["Channel vitality proof", `${formatNumber(checks.channel_vitality.checked_channels || 0)} channel(s); ${formatNumber(checks.channel_vitality.dual_mode_constant_output_channels || 0)} dual-mode constant; ${formatNumber(checks.channel_vitality.nonconstant_accumulator_dual_mode_constant_channels || 0)} variable-accumulator constant; ${formatNumber(checks.channel_vitality.mode_dependent_constant_output_channels || 0)} build-mode-dependent constant`]] : []),
      ...(!onnx && advancedInteger ? [["Fixed-point build-mode equivalence", `${formatNumber(checks.rounding_equivalence.checked_channels || 0)} channel(s); ${formatNumber(checks.rounding_equivalence.equivalent_channels || 0)} equivalent / ${formatNumber(checks.rounding_equivalence.divergent_channels || 0)} divergent; ${formatNumber(checks.rounding_equivalence.divergent_state_count_decimal || 0)} of ${formatNumber(checks.rounding_equivalence.interval_state_count_decimal || 0)} interval states (${formatPercent(checks.rounding_equivalence.divergent_state_ratio || 0)}); maximum ${formatNumber(checks.rounding_equivalence.maximum_absolute_output_delta || 0)} code`]] : []),
      ...(!onnx && advancedInteger ? [["Accumulator reachability", `${formatNumber(checks.accumulator_reachability.exact_reachable_divergent_state_count_decimal || 0)} exact kernel-local divergent state(s); ${formatNumber(checks.accumulator_reachability.provably_unreachable_divergent_state_count_decimal || 0)} residue-excluded / ${formatNumber(checks.accumulator_reachability.unresolved_divergent_state_count_decimal || 0)} unresolved; complete integer/modular channels ${formatNumber(checks.accumulator_reachability.complete_integer_interval_channels || 0)} / ${formatNumber(checks.accumulator_reachability.complete_modular_lattice_channels || 0)}`]] : []),
      ...(!onnx && advancedInteger ? [["Numerical ABI propagation", `${formatNumber(checks.numerical_abi_propagation.exact_local_counterexample_sources || 0)} exact-local source op(s); ${formatNumber(checks.numerical_abi_propagation.exact_local_divergent_state_count_decimal || 0)} reachable divergent state(s), ${formatNumber(checks.numerical_abi_propagation.residue_excluded_divergent_state_count_decimal || 0)} residue-excluded and ${formatNumber(checks.numerical_abi_propagation.unresolved_divergent_state_count_decimal || 0)} unresolved; ${formatNumber(checks.numerical_abi_propagation.exact_source_corridor_edge_instances || 0)} exact-qualified source-corridor edge instance(s)`]] : []),
      ...(!onnx && advancedInteger ? [["Model-input tensor ABI witness", `${formatNumber(checks.input_counterexample.tensor_abi_constructive_sources || 0)} of ${formatNumber(checks.input_counterexample.exact_local_sources || 0)} exact-local source op(s) constructively realized by complete model-input tensor(s); ${formatNumber(checks.input_counterexample.tensor_abi_constructive_channels || 0)} channels / ${formatNumber(checks.input_counterexample.tensor_abi_constructive_divergent_state_count_decimal || 0)} exact states; ${formatNumber(checks.input_counterexample.upstream_activation_unresolved_sources || 0)} source(s) remain upstream-constrained`]] : []),
      ...(!onnx && advancedInteger ? [["Pixel-to-tensor contract matrix", `${formatNumber(checks.preprocessing_realizability.exact_tensor_realization_candidates || 0)} exact and ${formatNumber(checks.preprocessing_realizability.non_exact_candidates || 0)} non-exact explicit preprocessing counterfactual(s); best non-exact ${checks.preprocessing_realizability.best_non_exact_contract_id || "none"} leaves ${formatNumber(checks.preprocessing_realizability.best_non_exact_unrealizable_elements || 0)} tensor element(s) unrealizable`]] : []),
      ...(!onnx && advancedInteger ? [["Residual step response", `${formatNumber(checks.residual_step_response.checked_contracts || 0)} contract(s); ${formatNumber(checks.residual_step_response.exact_branch_transitions || 0)} exact adjacent branch transition(s); ${signedCount(checks.residual_step_response.containment_additional_silent_transitions || 0)} containment silent-transition delta; ${formatNumber(checks.residual_step_response.containment_removed_rounded_clamp_pairs || 0)} rounded clamp pair(s) removed`]] : []),
      ...(!onnx && advancedInteger ? [["Residual contract distortion", `${formatNumber(checks.residual_contract_distortion.checked_candidate_scenarios || 0)} candidate scenario(s); ${formatNumber(checks.residual_contract_distortion.exact_pair_comparisons || 0)} exact pair comparison(s); ${formatNumber(checks.residual_contract_distortion.rescued_current_clamp_pair_instances || 0)} current clamp instance(s) rescued; ${formatNumber(checks.residual_contract_distortion.ideal_error_improved_pairs || 0)} improve / ${formatNumber(checks.residual_contract_distortion.ideal_error_worsened_pairs || 0)} worsen`]] : []),
      ["Missing quantization metadata", `${formatNumber(missing.tensors_without_metadata)} binding/tensor record(s); ${formatNumber(missing.review_tensors)} require review; ${formatNumber(missing.not_applicable_tensors)} classified as shape/configuration`],
      ["Input tensor numerical contract", ioText(io.inputs || [], "Input")],
      ...(!onnx && inputConvention ? [["Input range convention check", `${formatNumber(inputConvention.unmatched_inputs)} of ${formatNumber(inputConvention.assessed_inputs)} quantized input(s) do not match [0,1], [-1,1], [0,255], or applicable signed-code identity full-domain endpoints within one quantization step; ${inputConvention.evidence_class}`]] : []),
      ["Source-data-to-tensor preprocessing", (io.inputs || []).length ? "NOT_EMBEDDED_IN_ARTIFACT; dequantized tensor range does not establish color order, decoding, resize, interpolation, or source normalization" : "not applicable or not emitted"],
      ["Output dequantization contract", ioText(io.outputs || [], "Output")],
      ["Q/DQ boundary inventory", onnx
        ? `${formatNumber(qdq.explicit_qdq_ops)} explicit QuantizeLinear/DequantizeLinear op(s); logical edge payload is reported per boundary; runtime materialization ${qdq.runtime_materialization_status}`
        : `${formatNumber(qdq.explicit_qdq_ops)} explicit QUANTIZE/DEQUANTIZE op(s); ${formatNumber(qdq.mid_graph_quantization_holes)} serialized mid-graph 8-bit/FP32 boundary op(s), not float-island regions`],
    ];
  return [
    "## Quantization Contract Checks (DERIVED)",
    markdownTable(["Check", "Result"], rows),
    ...(onnx && annotationBindings.length ? [
      "### ONNX GraphProto TensorAnnotation Bindings",
      markdownTable(["Tensor", "Status", "Scale tensor", "Zero-point tensor", "Parameterization", "Axis source", "Operator cross-check", "Reasons"], annotationBindings.map((item) => [
        item.tensor_name || "unnamed",
        item.status,
        `${item.scale_tensor_name || "missing"} (${formatNumber(item.scale_count || 0)})`,
        `${item.zero_point_tensor_name || "missing"} (${formatNumber(item.zero_point_count || 0)})`,
        item.parameterization,
        item.axis_source,
        item.operator_cross_check_status,
        (item.reasons || []).join("; ") || "none",
      ])),
    ] : []),
    ...(onnx && binding?.nested_graph_annotation_count ? [
      "### Nested GraphProto Annotation Inventory",
      markdownTable(["Scope", "Tensor", "Parameter entries", "Assessment"], (binding.nested_graph_annotations || []).map((item) => [item.scope, item.tensor_name || "unnamed", formatNumber(item.parameter_entry_count || 0), "inventoried; scope-local numerical binding not reconstructed"])),
    ] : []),
    `> DERIVED: this table renders the same ${checks.schema} object exported in engineering_evidence.json. Absence of a contract violation is not a task-accuracy claim.`,
  ].join("\n");
}

function delegationPredictionMarkdown(analysis, delegationSegments) {
  if (isOnnxAnalysis(analysis)) {
    const watchlist = analysis?.runtime_review_watchlist || [];
    const domainAnalysis = analysis?.onnx_domain_analysis || null;
    const compatibility = analysis?.ort_compatibility_evidence || null;
    const conditionInventory = compatibility?.source_condition_inventory || null;
    const exportedCaptureCapability = analysis?.ort_assignment_capture_capability || null;
    const captureCapability = exportedCaptureCapability || compatibility?.assignment_capture_capability || null;
    const providers = compatibility?.execution_providers || [];
    const artifactConditionIssues = providers.flatMap((ep) => (ep.ops || []).flatMap((op) => (op.artifact_conditions || [])
      .filter((condition) => condition.status !== "PASS")
      .map((condition) => ({ execution_provider: ep.execution_provider, op, condition }))))
      .sort((left, right) => (left.condition.status === "DEFINITE_FAIL" ? 0 : 1) - (right.condition.status === "DEFINITE_FAIL" ? 0 : 1)
        || Number(left.condition.condition_kind === "source_condition_fragment") - Number(right.condition.condition_kind === "source_condition_fragment")
        || left.op.op_index - right.op.op_index || left.execution_provider.localeCompare(right.execution_provider))
      .slice(0, 32);
    const schemaSources = [...new Map(providers.flatMap((ep) => (ep.ops || []))
      .filter((row) => row.schema_source_ref && row.schema_source_sha256)
      .map((row) => [row.schema_source_ref, { source_ref: row.schema_source_ref, sha256: row.schema_source_sha256 }])).values()];
    const artifactConditionSources = [...new Map(providers.flatMap((ep) => (ep.ops || []))
      .flatMap((row) => row.artifact_conditions || [])
      .filter((condition) => [condition.source_ref, condition.source_sha256].every(Boolean))
      .map((condition) => [condition.source_ref, { source_ref: condition.source_ref, sha256: condition.source_sha256 }])).values()];
    const sourceCompatibility = providers.length ? [
      "## Execution Provider Source Compatibility (SOURCE+ARTIFACT_PRECHECK/NOT_OBSERVED)",
      "> Each model-domain opset import is resolved to the greatest pinned ONNX OpSchema since_version not exceeding that import. CPU parameter/type registrations, WebGPU/WebNN clauses, DirectML registration versions, QNN static-shape/Reshape/Resize gates, CoreML common rank gates, NNAPI static-shape gates, and extracted XNNPACK conditions are evaluated against the artifact. Remaining support-query, builder, build, device, partitioning, graph-transform, and runtime-assignment dimensions stay unresolved. Definite failures exclude a candidate; pass or unresolved never proves support, partitioning, or assignment.",
      markdownTable(["Rulepack provenance", "Value"], [
        ["Export assessment / schema / access", `${analysis?.ort_compatibility_assessment_status || "not_assessed"} / ${analysis?.ort_compatibility_evidence_schema || "not embedded"} / ${analysis?.ort_compatibility_evidence_access || "not reported"}`],
        ["Evidence assessment / schema", `${compatibility.assessment_status || "not_assessed"} / ${compatibility.schema || "not embedded"}`],
        ["Method / access scope", `${compatibility.method_version || "not embedded"} / ${compatibility.access_scope || "not reported"}`],
        ["Pinned ORT source commit", compatibility.source_commit || "not embedded"],
        ["Runtime-floor status / model IR", `${compatibility.runtime_floor?.status || "not_assessed"} / ${compatibility.runtime_floor?.model_ir_version ?? "not emitted"}`],
        ["Source-condition extractor", conditionInventory ? `${conditionInventory.status} / ${conditionInventory.schema}` : "not assessed"],
        ["Source rules / CPU registration variants", conditionInventory ? `${formatNumber(conditionInventory.source_rule_count)} / ${formatNumber(conditionInventory.cpu_registration_variant_count)} (signature ${formatNumber(conditionInventory.cpu_registration_variant_with_signature_count)}, type set ${formatNumber(conditionInventory.cpu_registration_variant_with_type_constraint_count)})` : "not assessed"],
        ["Machine conditions / schema-default bindings / unresolved source fragments", conditionInventory ? `${formatNumber(conditionInventory.machine_condition_count)} / ${formatNumber(conditionInventory.versioned_scalar_schema_default_binding_count)} / ${formatNumber(conditionInventory.unresolved_source_fragment_count)}; informational ${formatNumber(conditionInventory.informational_source_note_count)}` : "not assessed"],
      ]),
      conditionInventory ? markdownTable(["EP source extractor", "Source rules", "Registration variants", "Machine conditions", "Unresolved source fragments", "Informational notes"], (conditionInventory.execution_providers || []).map((row) => [
        row.execution_provider,
        formatNumber(row.source_rule_count),
        formatNumber(row.registration_variant_count),
        formatNumber(row.machine_condition_count),
        formatNumber(row.unresolved_source_fragment_count),
        formatNumber(row.informational_source_note_count),
      ])) : "",
      conditionInventory ? `> ${conditionInventory.evidence_boundary}` : "",
      markdownTable(["EP profile", "Source scope", "Evaluator coverage", "Evidence class", "Source-version matches", "Narrowed candidates", "Precheck pass/fail/unresolved/no-condition", "Conditions total/pass/fail/unresolved", "No source version", "Schema/rule unresolved", "Source SHA-256", "Assignment"], providers.map((ep) => [
        ep.execution_provider,
        ep.source_scope,
        ep.evaluator_coverage,
        ep.support_evidence_class,
        `${formatNumber(ep.schema_kernel_version_match_count)}/${formatNumber(ep.assessed_op_count)} (${formatPercent(ep.schema_kernel_version_match_ratio)})`,
        `${formatNumber(ep.source_candidate_after_artifact_precheck_count)}/${formatNumber(ep.assessed_op_count)} (${formatPercent(ep.source_candidate_after_artifact_precheck_ratio)})`,
        `${formatNumber(ep.artifact_precheck_pass_op_count)}/${formatNumber(ep.artifact_precheck_definite_fail_op_count)}/${formatNumber(ep.artifact_precheck_unresolved_op_count)}/${formatNumber(ep.artifact_precheck_no_condition_op_count)}`,
        `${formatNumber(ep.artifact_condition_count)}/${formatNumber(ep.artifact_condition_pass_count)}/${formatNumber(ep.artifact_condition_fail_count)}/${formatNumber(ep.artifact_condition_unresolved_count)}`,
        formatNumber(ep.schema_kernel_version_no_match_count),
        `${formatNumber(ep.schema_version_unresolved_count)}/${formatNumber(ep.source_rule_missing_count)}; local ${formatNumber(ep.model_local_function_count)}; external ${formatNumber(ep.external_custom_registry_count)}`,
        `${ep.source_sha256}; ${ep.source_ref}`,
        ep.assignment_evidence_class,
      ])),
      artifactConditionIssues.length ? "### Artifact Precheck Failures and Unresolved Clauses" : "",
      artifactConditionIssues.length ? markdownTable(["EP", "Op", "Status", "Condition", "Expected", "Observed", "Reason", "Pinned condition source"], artifactConditionIssues.map(({ execution_provider, op, condition }) => [
        execution_provider,
        `#${padOp(op.op_index)} ${op.op_name}`,
        condition.status,
        `${condition.condition_id}: ${condition.subject}`,
        condition.expected,
        condition.observed || "not observed",
        condition.reason,
        condition.source_ref ? `${condition.source_ref}; SHA-256 ${condition.source_sha256}` : "provider source row",
      ])) : "All emitted artifact-visible source conditions passed; source conditions absent from the machine registry remain covered by the evidence boundary.",
      "### Operator Schema Sources",
      markdownTable(["Pinned schema source", "SHA-256"], schemaSources.map((source) => [source.source_ref, source.sha256])),
      artifactConditionSources.length ? "### Artifact Condition Sources" : "",
      artifactConditionSources.length ? markdownTable(["Pinned predicate source", "SHA-256"], artifactConditionSources.map((source) => [source.source_ref, source.sha256])) : "",
      `> ${compatibility.evidence_boundary}`,
    ].join("\n") : [
      "## Execution Provider Source Compatibility (NOT_ASSESSABLE)",
      `Assessment: ${analysis?.ort_compatibility_assessment_status || "not_assessed"}; schema ${analysis?.ort_compatibility_evidence_schema || "not embedded"}; access ${analysis?.ort_compatibility_evidence_access || "not reported"}. The controlled ORT source rulepack was not loaded for this export.`,
    ].join("\n");
    const domainInventory = domainAnalysis ? [
      "## ONNX Domain and Function Registry (OBSERVED/DERIVED)",
      markdownTable(["Registry field", "Value"], [
        ["Status / schema", `${domainAnalysis.status || "not_assessed"} / ${domainAnalysis.schema || "not emitted"}`],
        ["Analysis scope", domainAnalysis.scope || "not emitted"],
        ["Standard / ORT contrib / external custom nodes", `${formatNumber(domainAnalysis.standard_node_count || 0)} / ${formatNumber(domainAnalysis.ort_contrib_node_count || 0)} / ${formatNumber(domainAnalysis.external_custom_node_count || 0)}`],
      ]),
      markdownTable(["Domain", "Imported opset", "Nodes (main/nested/function)", "Local definitions", "Resolution class", "Operators"], (domainAnalysis.domains || []).map((domain) => [
        domain.domain,
        domain.imported_opset ?? "missing",
        `${domain.node_count} (${domain.main_graph_node_count}/${domain.nested_graph_node_count}/${domain.function_body_node_count})`,
        formatNumber(domain.local_function_definition_count),
        (domain.resolution_classes || []).join(" / ") || (domain.unused_import ? "unused import" : "unresolved"),
        (domain.op_types || []).join(", ") || "-",
      ])),
      (domainAnalysis.functions || []).length
        ? markdownTable(["Local function", "Body nodes", "Dependencies", "Cycle"], domainAnalysis.functions.map((fn) => [
          fn.id,
          formatNumber(fn.body_node_count),
          (fn.local_function_dependencies || []).join(" / ") || "none",
          fn.recursive_cycle ? "yes" : "no",
        ]))
        : "No model-local FunctionProto definition was observed.",
      `> DERIVED classification: ${domainAnalysis.interpretation_boundary}`,
    ].join("\n") : "## ONNX Domain and Function Registry (NOT_ASSESSABLE)\nDomain analysis was not emitted.";
    return [
      domainInventory,
      "",
      sourceCompatibility,
      "",
      "## Actual Execution Provider Assignment (NOT_ASSESSABLE)",
      markdownTable(["Field", "Value"], [
        ["Static source rulepack", providers.length ? `ORT ${compatibility.runtime_version_basis}; ${providers.map((ep) => ep.execution_provider).join(" / ")}` : "not loaded"],
        ["Reason", "Resolved OpSchema and kernel-registration version matches do not establish type/attribute eligibility, GetCapability partitioning, or observed provider placement"],
        ["Pinned browser capture capability", captureCapability
          ? `${captureCapability.status}; ${captureCapability.evidence_class}; ORT Web ${captureCapability.pinned_ort_web_version}`
          : "not assessed"],
        ["Exported assignment-capture envelope", exportedCaptureCapability
          ? `${exportedCaptureCapability.status || "not_assessed"}; ${exportedCaptureCapability.evidence_class || "not emitted"}; ORT Web ${exportedCaptureCapability.pinned_ort_web_version || "not emitted"}`
          : "not emitted"],
        ["Browser profiling API", captureCapability
          ? `start=${captureCapability.start_profiling_status}; end=${captureCapability.end_profiling_status}`
          : "not assessed"],
        ["Assignment capture modes", captureCapability
          ? `automatic browser assignment ${captureCapability.automatic_browser_assignment_capture ? "available" : "unavailable"}; external runtime profile import ${captureCapability.external_runtime_profile_import ? "available" : "unavailable"}`
          : "not assessed"],
        ["Pinned native capture path", captureCapability?.pinned_native_capture_available
          ? `${captureCapability.pinned_native_runtime}; ${captureCapability.native_capture_command}; schemas ${captureCapability.native_capture_schema} / ${captureCapability.native_profile_schema}; roles ${captureCapability.native_profile_roles}`
          : "not available"],
        ["Native capture contract", captureCapability
          ? `available=${captureCapability.pinned_native_capture_available ? "yes" : "no"}; runtime ${captureCapability.pinned_native_runtime || "not emitted"}; command ${captureCapability.native_capture_command || "not emitted"}; envelope ${captureCapability.native_capture_schema || "not emitted"}; profile ${captureCapability.native_profile_schema || "not emitted"}; roles ${captureCapability.native_profile_roles || "not emitted"}`
          : "not emitted"],
        ["Required evidence", captureCapability?.required_observation_path || "Pinned ONNX Runtime version, selected execution providers, and ORT profiling/session assignment logs"],
      ]),
      ...(captureCapability?.source_documents?.length ? [
        "### Assignment Capture Capability Sources",
        markdownTable(["Role", "Pinned source", "SHA-256"], captureCapability.source_documents.map((source) => [source.role, source.source_ref, source.sha256])),
      ] : []),
      "",
      "## Runtime Review Watchlist (OBSERVED/HEURISTIC)",
      watchlist.length
        ? markdownTable(["Operator", "Count", "Reason code", "Review reason"], watchlist.map((item) => [item.name, item.count, item.reason_code, item.reason]))
        : "No format-aware runtime review operator was observed.",
      "> Operator presence is OBSERVED. Review classification is HEURISTIC and does not assert execution-provider assignment, fallback, materialization, or measured cost.",
    ].join("\n");
  }
  const opsAll = analysis.ops || [];
  const isDefaultReject = (r) => r === "XNNP:F:REJECT" || r === "XNNP:Q:REJECT";
  const matched = opsAll.filter((op) => op.xnnpack_supported).length;
  const listedUnsupported = opsAll.filter((op) => !op.xnnpack_supported && String(op.xnnpack_reason || "").startsWith("XNNP:F:REJECT:")).length;
  const conditionFailed = opsAll.filter((op) => !op.xnnpack_supported && !isDefaultReject(op.xnnpack_reason) && !String(op.xnnpack_reason || "").startsWith("XNNP:F:REJECT:")).length;
  const notInTables = opsAll.filter((op) => !op.xnnpack_supported && isDefaultReject(op.xnnpack_reason)).length;
  const boundary = predictedPartitionBoundaryInventory(analysis) || {};
  return [
    "## Delegation Prediction Coverage",
    markdownTable(["Prediction summary", "Value"], [
      ["Conditionally delegatable MAC / predicted-fallback logical-byte share", `${formatPercent(analysis.delegated_mac_percent || 0)} / ${formatPercent(analysis.fallback_byte_percent || 0)}`],
      ["Boundary assessment status", `${boundary.status || "not_assessed"}; payload ${boundary.payload_coverage_status || "not_assessed"}`],
      ["Boundary payload binding", boundary.payload_binding || "not assessed"],
      ["Boundary edges assessed / unassessed", `${formatNumber(boundary.assessed_payload_edge_count || 0)} / ${formatNumber(boundary.unassessed_payload_edge_count || 0)} of ${formatNumber(boundary.edge_count || 0)}`],
      ["Summed edge payload assessed / total", `${formatBytes(boundary.assessed_edge_payload_bytes || 0)} / ${boundary.summed_edge_payload_bytes == null ? "NOT_ASSESSED" : formatBytes(boundary.summed_edge_payload_bytes)}`],
      ["Unique-tensor payload assessed / total", `${formatBytes(boundary.assessed_unique_tensor_payload_bytes || 0)} / ${boundary.unique_tensor_payload_bytes == null ? "NOT_ASSESSED" : formatBytes(boundary.unique_tensor_payload_bytes)}`],
      ["Boundary interpretation", boundary.interpretation_boundary || "not emitted"],
    ]),
    markdownTable(["Prediction basis", "Ops"], [
      ["Rule matched - conditionally delegatable under required build flags", `${matched}`],
      ["Rule matched - condition failed (predicted fallback)", `${conditionFailed}`],
      ["Explicitly listed unsupported in rulepack", `${listedUnsupported}`],
      ["Not in rulepack tables (unknown - treated as fallback)", `${notInTables}`],
    ]),
    "> Coverage shows how strongly each prediction is grounded: \"not in tables\" rows are conservative guesses, not documented incompatibilities.",
    tfliteAlternateDelegateCompatibilityMarkdown(analysis),
    "",
    "## Predicted Delegate Segment View",
    delegationSegments.length
      ? markdownTable(["Segment", "Type", "Length", "MAC share", "Op range", "Operators"], delegationSegments)
      : "No predicted XNNPACK candidate segment data was available.",
    "> All rows are predictions from the rulepack's delegation rules; a partition boundary becomes confirmed only through runtime delegate logs.",
  ].join("\n");
}

function actionQueueMarkdown(findings) {
  if (!findings.length) return "No prioritized engineering finding was generated from the current static and runtime evidence.";
  const ordered = sortFindingsByPriority(findings);
  const counts = FINDING_PRIORITIES.map((priority) => `${priority} ${ordered.filter((item) => item.technical_priority === priority).length}`).join(" / ");
  const summary = markdownTable(["ID", "Priority", "Class", "Evidence", "Confidence", "Title", "Affected component"], ordered.map((finding) => [
    finding.finding_id,
    finding.technical_priority,
    finding.finding_class || finding.category,
    finding.evidence_class,
    finding.confidence,
    finding.title,
    finding.affected_operator || finding.affected_tensor || "-",
  ]));
  const details = ordered.map((finding) => [
    `### ${finding.finding_id} - ${finding.title}`,
    markdownTable(["Finding field", "Value"], [
      ["Priority / class", `${finding.technical_priority} / ${finding.finding_class || finding.category}`],
      ["Evidence / confidence", `${finding.evidence_class} / ${finding.confidence}`],
      ["Origin / source rule / method", `${finding.origin || "report_synthesis"} / ${finding.source_rule_id} / ${finding.method_version}`],
      ["Evidence JSON Pointer(s)", (finding.evidence_json_pointers || []).map((pointer) => code(pointer)).join(" / ") || "not emitted"],
      ["Affected component", finding.affected_operator || finding.affected_tensor || "-"],
      ["Observation", finding.observation || "-"],
      ["Interpretation", finding.interpretation || "-"],
      ["Possible effects", (finding.possible_effects || []).join(" / ") || "-"],
      ["Recommendation", finding.recommendation || "-"],
      ["Evidence limitation", finding.limitations || "-"],
      ["Documentation relevance", finding.regulatory_relevance || "-"],
    ]),
  ].join("\n\n")).join("\n\n");
  return [`Finding conservation: ${ordered.length} emitted = ${counts}. No finding is omitted from the queue. Bounded inline evidence previews may be summarized; the complete machine-readable records remain at the listed evidence pointers and are checked against the exported evidence document by bundle conformance.`, summary, details].join("\n\n");
}

function signedScientific(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${number.toExponential(6)}`;
}

function signedCount(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${formatNumber(number)}`;
}

function analyzerTriageMarkdown(analysis) {
  const insights = analysis?.insights || {};
  const movement = analysis?.movement_analysis || {};
  const epNotModeled = isOnnxAnalysis(analysis);
  const structuralOps = structuralViewOps(analysis);
  const nonDelegated = predictedNonDelegatedOps(analysis);
  const boundaryInventory = epNotModeled ? null : predictedPartitionBoundaryInventory(analysis);
  const quantizationContracts = buildQuantizationContractChecks(analysis);
  const flaggedRepresentableChannels = Number(
    quantizationContracts?.representable_kernel_channels?.flagged_channels || 0,
  );
  const exactZeroChannels = Number(analysis?.weight_integrity?.exact_zero_kernel_slice_count || 0);
  const vitalityByOp = new Map((analysis?.channel_vitality?.ops || []).map((row) => [Number(row.op_index), row]));
  let exactZeroConstantChannels = 0;
  for (const detail of analysis?.weight_integrity?.zero_kernel_slice_details || []) {
    const exact = new Set((detail.exact_zero_channels || []).map(Number));
    const consumers = (detail.consumer_ops || [])
      .map((label) => Number(String(label).match(/^#(\d+)/)?.[1]))
      .filter(Number.isInteger);
    for (const opIndex of consumers) {
      const vitality = vitalityByOp.get(opIndex);
      if (vitality?.assessment_status !== "assessed") continue;
      const defaultConstant = new Set((vitality.default_constant_channel_indices || []).map(Number));
      const singleConstant = new Set((vitality.single_constant_channel_indices || []).map(Number));
      exactZeroConstantChannels += [...exact].filter((channel) => defaultConstant.has(channel) && singleConstant.has(channel)).length;
    }
  }
  const unmatchedInputConventions = Number(
    quantizationContracts?.input_quantization_convention?.unmatched_inputs || 0,
  );
  const triageHeadline = exactZeroConstantChannels > 0
    ? `${formatNumber(exactZeroChannels)} exact-zero stored kernel channel(s) were observed; ${formatNumber(exactZeroConstantChannels)} also have exact constant-output proofs under both pinned fixed-point paths. This is a high-priority export/QAT defect review signal, while declared-output and task impact still require downstream and representative-data evidence.`
    : exactZeroChannels > 0
      ? `${formatNumber(exactZeroChannels)} exact-zero stored kernel channel(s) require bias, fused-activation, downstream, and representative-output review; functional inactivity is not concluded from weights alone.`
      : flaggedRepresentableChannels > 0
    ? `${formatNumber(flaggedRepresentableChannels)} near-zero representable kernel channel(s) require quantization design review. Exact artifact ranges meet disclosed heuristic thresholds; functional inactivity and accuracy impact are not concluded.`
    : unmatchedInputConventions > 0
      ? `${formatNumber(unmatchedInputConventions)} quantized input range(s) do not match the disclosed common full-domain preprocessing references; bind the production preprocessing contract before release.`
      : "Context-free static triage only; no high-severity structural signal is concluded without product requirements, runtime evidence, and reference-data tests.";
  return markdownTable(["Signal", "Value"], [
    ["Triage headline", triageHeadline],
    ["Requirement-aware assessment", "not performed; intended precision, cold-start budget, p95 latency budget, peak-memory budget, no-fallback requirement, artifact-size limit, and dynamic-shape policy were not supplied"],
    ["Context-free composite score", "not reported; component signals remain separate to avoid implying model quality or release readiness"],
    ["Analyzer label / tone", `${insights.label || "not emitted"} / ${insights.tone || "not emitted"}`],
    ["Evidence class", insights.score_evidence_class || "HEURISTIC"],
    ["Interpretation boundary", "No pass/fail or deployment-readiness meaning. Component signals are static triage evidence and are not validated substitutes for measured deployment outcomes."],
    ["Component signals", "quantization coverage, low-intensity op mix, L1 row-working-set watch count, structural/view ops, predicted fallback ops, predicted fallback logical-byte ratio, quant-risk ops, explicit movement ops, packing watch count, channel alignment, and dynamic inputs"],
    ["Rationale", insights.rationale || "not emitted"],
    ["Emitted posture counts", `high-intensity ${formatNumber(insights.bound_compute || 0)} / mixed ${formatNumber(insights.bound_mixed || 0)} / low-intensity ${formatNumber(insights.bound_memory || 0)}`],
    ["Emitted quantization signals", `quantized tensor ratio ${formatPercent(insights.quant_ratio || 0)}; per-channel ratio ${formatPercent(insights.per_channel_ratio || 0)}; quant-risk ops ${formatNumber(insights.quant_risk_ops || 0)}`],
    ["Emitted chain signals", `${formatNumber(insights.chain_breaks || 0)} predicted break op(s), ${formatNumber(insights.effective_chain_breaks || 0)} effective; ${insights.chain_summary || "no summary"}`],
    ["Emitted watchlist signals", `${formatNumber(insights.copy_like_op_count || 0)} structural/view candidate, ${formatNumber(insights.misaligned_ops || 0)} misaligned, ${formatNumber(insights.suspect_total || 0)} unsupported non-structural suspect; max L1 ratio ${Number(insights.max_l1_ratio || 0).toFixed(2)}x; this candidate count is not an observed copy and explicit movement remains a separate metric; structural predicted breaks are counted separately; ${insights.suspect_summary || "no suspect-family summary"}`],
    ["Dynamic input policy", `${formatNumber(insights.dynamic_input_count || 0)} dynamic input(s); ${formatNumber(insights.dynamic_non_batch_input_count || 0)} with non-batch dynamic axes. Serialized batch-only N=1 projections are not penalized.`],
    ["Predicted conditionally delegatable operator ratio", delegatedOpRatioText(analysis)],
    ["Conditionally delegatable MAC ratio", epNotModeled ? "N/A; no ORT EP rulepack or session assignment log" : (insights.delegated_mac_ratio == null ? formatPercent(analysis?.delegated_mac_percent || 0) : formatPercent(insights.delegated_mac_ratio))],
    ["Predicted fallback logical-byte ratio", epNotModeled ? "N/A; no ORT EP rulepack or session assignment log" : (insights.fallback_byte_ratio == null ? formatPercent(analysis?.fallback_byte_percent || 0) : `${formatPercent(insights.fallback_byte_ratio)} (not confirmed runtime copy traffic)`)],
    ["Low-intensity op mix", insights.memory_ratio == null ? "not emitted" : `${formatPercent(insights.memory_ratio)} (${formatNumber(insights.bound_memory || 0)} low-intensity ops)`],
    ["L1 watch count", formatNumber(l1WatchCount(analysis))],
    ["Global maximum row-WS / L1 ratio", `${maxRowWorkingSetRatio(analysis).toFixed(2)}x`],
    ["Packing warn ops", formatNumber(insights.packing_warn_ops || (Number(analysis?.conv_packing_warn_ops || 0) + Number(analysis?.fc_packing_warn_ops || 0)))],
    ["Misaligned channel ops", misalignedChannelOpSummary(analysis)],
    ["Explicit copy or movement operators", epNotModeled
      ? "0 confirmed materializing op(s); see structural/view row"
      : movement.total_movement_bytes == null
        ? `${formatNumber(movement.movement_op_count || 0)} op(s); PARTIAL ${formatBytes(movement.assessed_movement_bytes || 0)} across assessed outputs, ${formatNumber(movement.unassessed_output_tensor_count || 0)} output tensor(s) not assessed`
        : `${formatNumber(movement.movement_op_count || 0)} op(s); ${formatBytes(movement.total_movement_bytes)} static output payload`],
    ["Structural or view operators", `${formatNumber(structuralOps.length)} op(s); ${opNameList(structuralOps)}`],
    ["Predicted non-delegated operators", epNotModeled ? "N/A; ONNX Runtime EP assignment is not modeled" : `${formatNumber(nonDelegated.length)} op(s); ${opNameList(nonDelegated)}`],
    ["Predicted internal partition edges", epNotModeled ? "N/A; provider partitioning is not modeled" : `${formatNumber(boundaryInventory?.edge_count || 0)} edge(s), ${formatNumber(boundaryInventory?.unique_tensor_count || 0)} unique tensor(s)`],
    ["Potential partition-interface logical payload", epNotModeled ? "N/A; provider partition movement is not modeled" : predictedBoundaryPayloadText(analysis)],
    ["Confirmed runtime copy bytes", "NOT_ASSESSABLE"],
    ["Dynamic inputs", formatNumber(insights.dynamic_input_count || 0)],
  ]);
}

function stageAndPatternMarkdown(analysis) {
  const stageRows = (analysis?.stages || []).slice(0, 16).map((stage) => [
    `S${stage.index}`,
    stage.key || "-",
    channelSetText(stage.channels || []),
    `${padOp(stage.first_op)}-${padOp(stage.last_op)}`,
    formatNumber(stage.op_count || 0),
    formatPercent(stage.mac_percent || 0),
    `${formatNumber(stage.delegated_ops || 0)} / ${formatNumber(stage.op_count || 0)} (${formatPercent(stage.delegated_op_percent || 0)})`,
    stageDelegatedMacText(stage),
    `${formatNumber(stage.fallback_ops || 0)} / ${formatNumber(stage.op_count || 0)} (${formatPercent(stage.fallback_op_percent || 0)})`,
    formatBytes(stage.estimated_bytes || 0),
    formatNumber(stage.xnnpack_chain_breaks || 0),
    (stage.patterns || []).join(" / ") || "-",
  ]);
  if (isOnnxAnalysis(analysis)) {
    const onnxRows = (analysis?.stages || []).slice(0, 16).map((stage) => [
      `S${stage.index}`,
      stage.key || "-",
      channelSetText(stage.channels || []),
      `${padOp(stage.first_op)}-${padOp(stage.last_op)}`,
      formatNumber(stage.op_count || 0),
      formatPercent(stage.mac_percent || 0),
      Number(stage.byte_not_assessed_ops || 0) > 0
        ? `${formatBytes(stage.estimated_bytes || 0)} assessed subtotal; ${formatNumber(stage.byte_not_assessed_ops)} op(s) unassessed`
        : formatBytes(stage.estimated_bytes || 0),
    ]);
    const shape = analysis?.onnx_shape_inference || {};
    return [
      "## Stage And Pattern Summary (DERIVED)",
      "ONNX stage note: static stages are contiguous op-order runs by operator family after ONNX shape inference. Execution-provider partitioning is not modeled in this section.",
      `Shape inference: ${formatNumber(shape.inferred_outputs || 0)} output tensor(s) inferred; ${formatNumber(shape.unknown_tensor_count || 0)} tensor(s) remain unknown.`,
      "",
      onnxRows.length
        ? markdownTable(["Stage", "Key", "Observed output channels", "Op range", "Ops", "MAC share", "Summed logical op bytes"], onnxRows)
        : "No ONNX stage summary was emitted.",
      (analysis?.stages || []).length > onnxRows.length ? `Stage table truncated: showing ${formatNumber(onnxRows.length)} of ${formatNumber((analysis.stages || []).length)} stage(s).` : "",
    ].join("\n");
  }
  const patternGroups = groupPatterns(analysis?.patterns || []);
  const semanticStages = analysis?.block_inventory?.status === "assessed"
    ? analysis.block_inventory.stages || []
    : [];
  if (semanticStages.length) {
    const blocksById = new Map((analysis.block_inventory.blocks || []).map((block) => [block.block_id, block]));
    const opByIndex = new Map((analysis.ops || []).map((op) => [op.index, op]));
    const semanticRows = semanticStages.slice(0, 16).map((stage) => {
      const stageOps = (stage.op_indices || []).map((index) => opByIndex.get(index)).filter(Boolean);
      const delegated = stageOps.filter((op) => Number(op.xnnpack_chain_id) >= 0);
      const stageMacs = stageOps.reduce((total, op) => total + Number(op.macs || 0), 0);
      const delegatedMacs = delegated.reduce((total, op) => total + Number(op.macs || 0), 0);
      const blocks = (stage.block_ids || []).map((id) => blocksById.get(id)).filter(Boolean);
      const blockTypes = [...new Set(blocks.map((block) => String(block.block_type || "").replaceAll("_", " ")))];
      const opRange = (stage.op_indices || []).length
        ? `${padOp(Math.min(...stage.op_indices))}-${padOp(Math.max(...stage.op_indices))}`
        : "-";
      return [
        `S${stage.index}`,
        stage.display_name || "-",
        formatNumber(blocks.length),
        opRange,
        formatNumber(stageOps.length),
        formatPercent(stage.aggregates?.mac_percent || 0),
        `${formatNumber(delegated.length)} / ${formatNumber(stageOps.length)}`,
        stageMacs > 0 ? formatPercent(delegatedMacs / stageMacs) : "N/A",
        formatBytes(stage.aggregates?.logical_traffic_bytes || 0),
        stage.aggregates?.l1_max_ratio == null ? "N/A" : `${Number(stage.aggregates.l1_max_ratio).toFixed(2)}x`,
        formatNumber(stage.aggregates?.l1_watch_count || 0),
        formatNumber(stage.aggregates?.predicted_break_count || 0),
        blockTypes.join(" / ") || "-",
      ];
    });
    return [
      "## Stage And Pattern Summary (DERIVED)",
      "Stage partitioning rule: the shared Block Inventory groups graph-semantic blocks by output spatial resolution. PAD/PADV2-only preludes inherit the following compute stage and do not create standalone architecture stages.",
      "",
      markdownTable(["Stage", "Semantic label", "Blocks", "Op range", "Ops", "MAC share", "Conditionally delegatable ops", "Conditionally delegatable MAC", "Logical traffic", "Max L1 ratio", "L1 watch", "Predicted breaks", "Block types"], semanticRows),
      semanticStages.length > semanticRows.length ? `Stage table truncated: showing ${formatNumber(semanticRows.length)} of ${formatNumber(semanticStages.length)} semantic stage(s).` : "",
      "",
      patternGroups.length
        ? markdownTable(["Pattern", "Count", "First ranges", "Summary"], patternGroups.slice(0, 12))
        : "No recognized graph pattern was emitted.",
    ].join("\n");
  }
  return [
    "## Stage And Pattern Summary (DERIVED)",
    "Stage key note: 4D stage keys are HxWxCbucket<=N buckets, not exact channel counts; the Observed output channels column reports the set of actual channel counts seen in the stage.",
    "Stage partitioning rule: contiguous op-order runs with the same primary-output shape key; 4D tensors use H×W×channel-bucket keys, and structural tensors use fixed vector/scalar keys.",
    "",
    stageRows.length
      ? markdownTable(["Stage", "Key", "Observed output channels", "Op range", "Ops", "MAC share", "Conditionally delegatable ops", "Conditionally delegatable MAC", "Predicted fallback ops", "Summed logical op bytes", "Predicted breaks", "Pattern participation"], stageRows)
      : "No stage summary was emitted.",
    (analysis?.stages || []).length > stageRows.length ? `Stage table truncated: showing ${formatNumber(stageRows.length)} of ${formatNumber((analysis.stages || []).length)} stage(s).` : "",
    "",
    patternGroups.length
      ? markdownTable(["Pattern", "Count", "First ranges", "Summary"], patternGroups.slice(0, 12))
      : "No recognized graph pattern was emitted.",
  ].join("\n");
}

function groupPatterns(patterns = []) {
  const groups = new Map();
  for (const pattern of patterns || []) {
    const name = pattern.name || "pattern";
    if (!groups.has(name)) {
      groups.set(name, { name, count: 0, ranges: [], summary: pattern.summary || "-" });
    }
    const group = groups.get(name);
    group.count += 1;
    if (group.ranges.length < 4) group.ranges.push(`${padOp(pattern.first_op)}-${padOp(pattern.last_op)}`);
    if (pattern.summary && group.summary === "-") group.summary = pattern.summary;
  }
  return [...groups.values()].map((group) => [
    group.name,
    formatNumber(group.count),
    `${group.ranges.join(" / ")}${group.count > group.ranges.length ? " / ..." : ""}`,
    group.summary,
  ]);
}

function movementAndPackingMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) {
    const structuralOps = structuralViewOps(analysis);
    return [
      "## Movement And Packing Estimates (NOT_ASSESSABLE)",
      markdownTable(["Field", "Value"], [
        ["Structural reshape or view operations detected", `${formatNumber(structuralOps.length)} op(s); ${opNameList(structuralOps)}`],
        ["Confirmed materializing copy operators", "NOT_ASSESSABLE; ONNX Runtime kernels/providers may materialize or alias these tensors"],
        ["Static copy bytes", "NOT_ASSESSABLE unless materialization is provable"],
        ["Provider partition movement", "NOT_ASSESSABLE; ONNX Runtime execution-provider assignment is not modeled"],
        ["Provider weight-packing estimates", "suppressed until ONNX Runtime execution-provider behavior is modeled"],
        ["Required evidence", "ORT profiling/session execution-provider assignment plus provider-specific packing behavior"],
      ]),
    ].join("\n");
  }
  const movement = analysis?.movement_analysis || {};
  const structuralOps = structuralViewOps(analysis);
  const nonDelegated = predictedNonDelegatedOps(analysis);
  const boundaryInventory = predictedPartitionBoundaryInventory(analysis);
  const boundaryPayload = Number(boundaryInventory?.summed_edge_payload_bytes ?? boundaryInventory?.assessed_edge_payload_bytes ?? 0);
  const totalLogicalBytes = (analysis?.ops || []).reduce((sum, op) => sum + Number(op.estimated_bytes || 0), 0);
  const boundaryPayloadRatio = totalLogicalBytes > 0 ? boundaryPayload / totalLogicalBytes : 0;
  const packingOps = [...(analysis?.ops || [])]
    .filter((op) => op.weight_packing_risk === "warn")
    .sort((a, b) => Number(b.weight_packing_overhead_us || 0) - Number(a.weight_packing_overhead_us || 0))
    .slice(0, 8);
  return [
    `## Movement And Packing Estimates (${movement.status === "partial" ? "DERIVED/PARTIAL" : "DERIVED"}/ESTIMATED)`,
    markdownTable(["Field", "Value"], [
      ["Explicit copy or movement operators", `${formatNumber(movement.movement_op_count || 0)} op(s)`],
      ["Explicit copy/movement op ratio", formatPercent(movement.movement_op_ratio || 0)],
      ["Explicit static movement bytes", movement.total_movement_bytes == null ? `PARTIAL: ${formatBytes(movement.assessed_movement_bytes || 0)} across assessed outputs; ${formatNumber(movement.unassessed_output_tensor_count || 0)} output tensor(s) not assessed` : formatBytes(movement.total_movement_bytes)],
      ["Explicit movement bytes at predicted partition breaks", movement.xnn_break_movement_bytes == null ? `PARTIAL: ${formatBytes(movement.assessed_xnn_break_movement_bytes || 0)} across assessed outputs` : formatBytes(movement.xnn_break_movement_bytes)],
      ["Movement assessment ledger", `${formatBytes(movement.assessed_movement_bytes || 0)} assessed explicit bytes across ${formatNumber(movement.assessed_output_tensor_count || 0)} output tensor(s); ${formatNumber(movement.unassessed_output_tensor_count || 0)} unassessed; ${formatNumber(movement.calculation_issue_count || 0)} calculation issue(s); assessed break-local explicit bytes ${formatBytes(movement.assessed_xnn_break_movement_bytes || 0)}`],
      ["Structural or view operators", `${formatNumber(structuralOps.length)} op(s); ${opNameList(structuralOps)}`],
      ["Predicted non-delegated operators", `${formatNumber(nonDelegated.length)} op(s); ${opNameList(nonDelegated)}`],
      ["Predicted internal partition edges", `${formatNumber(boundaryInventory?.edge_count || 0)} edge(s); ${formatNumber(boundaryInventory?.unique_tensor_count || 0)} unique tensor(s); payload coverage ${boundaryInventory?.payload_coverage_status || "not assessed"}`],
      ["Potential partition-interface logical payload", `${predictedBoundaryPayloadText(analysis)}${boundaryInventory?.summed_edge_payload_bytes != null ? `; ${formatPercent(boundaryPayloadRatio)} of summed logical op bytes` : ""}`],
      ["Confirmed materialized copy operations", "NOT_ASSESSABLE from static artifact evidence"],
      ["Confirmed runtime copy bytes", "NOT_ASSESSABLE from static artifact evidence"],
      ["Conv packing warn ops", formatNumber(analysis?.conv_packing_warn_ops || 0)],
      ["FC packing warn ops", formatNumber(analysis?.fc_packing_warn_ops || 0)],
      ["Target packing bandwidth", analysis?.target_profile?.weight_packing_bandwidth_gbps ? `${analysis.target_profile.weight_packing_bandwidth_gbps} GB/s (rulepack heuristic, decimal GB/s, not device-calibrated)` : "not emitted"],
      ["Fixed packing setup overhead", `${packingSetupUs(analysis?.target_profile).toFixed(1)} us, rulepack heuristic, not device-calibrated (${analysis?.target_profile?.in_order ? "in-order target profile" : "out-of-order/browser target profile"})`],
      ["Packing watch threshold", "10 us single per-op warning threshold; all sub-threshold rows remain in the cold-start ledger"],
    ]),
    "> Explicit static movement bytes count output payloads for explicit copy/movement op families only. Predicted partition edges are exact graph producer/consumer edges under the static assignment, not an input-plus-output approximation around a break op. Their logical payload does not confirm copy materialization or latency. Packing estimates use stored weight bytes and the selected target profile's packing-bandwidth/setup assumptions; they describe possible one-time preparation or first-invocation cost, not steady-state latency.",
    "",
    "### Predicted Partition Boundary Edges (PREDICTED assignment / DERIVED payload)",
    predictedBoundaryEdgesMarkdown(analysis),
    "",
    packingOps.length
      ? markdownTable(["Op", "Name", "Weight bytes", "Illustrative profile formula output", "Risk", "Detail"], packingOps.map((op) => [
        `#${padOp(op.index)}`,
        op.name,
        formatBytes(op.weight_bytes || 0),
        formatUs(op.weight_packing_overhead_us || 0),
        op.weight_packing_risk || "-",
        op.weight_packing_detail || "-",
      ]))
      : "No weight-packing watchlist op was emitted for this target profile.",
  ].join("\n");
}

function coreIsolationRooflineMarkdown(analysis, runtimeEvidence = {}) {
  const result = analysis?.core_isolation_analysis;
  if (!result) return "";
  if (result.status !== "assessed" || !(result.scenarios || []).length) {
    return [
      "## Core Allocation Roofline (NOT_ASSESSED)",
      result?.unavailable_reason || "The artifact/target pair does not bind the core-count denominator needed for resource-partition scenarios.",
      result?.isolation_evidence_boundary ? `> ${result.isolation_evidence_boundary}` : "",
    ].filter(Boolean).join("\n\n");
  }
  const assignment = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || null;
  const partition = assignment?.selector_context?.invocation?.resource_partition || null;
  const requestedCoreCount = partition?.requested_cpu_ids?.length || 0;
  const onlineCoreCount = partition?.online_cpu_ids?.length || 0;
  const matchingScenarios = partition ? result.scenarios.filter((row) => (
    Number(row.ai_assigned_core_count) === requestedCoreCount
      && (!onlineCoreCount || Number(row.system_core_count) === onlineCoreCount)
  )) : [];
  const runtimeStatus = !partition ? "not imported"
    : partition.exclusive_isolation_status === "observed_cgroup_v2_isolated_partition"
      ? `OBSERVED_OS_RESOURCE_PARTITION; isolated cgroup v2 cpuset [${partition.observed_effective_cpu_ids.join(", ")}], sampled ${partition.sample_count} time(s), max ${partition.maximum_observed_thread_count} thread(s)`
      : `OBSERVED_OS_AFFINITY; requested CPUs [${partition.requested_cpu_ids.join(", ")}], sampled ${partition.sample_count} time(s), max ${partition.maximum_observed_thread_count} thread(s); exclusivity not established`;
  const runtimeBinding = !partition ? "not imported"
    : !onlineCoreCount ? `partial: imported ${requestedCoreCount}-CPU mask, but online system CPU count is unavailable`
      : matchingScenarios.length === 1 ? `matches ${matchingScenarios[0].scenario_id}; this binds OS allocation evidence, not the scenario's estimated latency`
        : `mismatch: imported ${onlineCoreCount}-CPU system / ${requestedCoreCount}-CPU mask has no static scenario in this target profile`;
  return [
    "## Core Allocation Roofline (ESTIMATED_STATIC_RESOURCE_PARTITION)",
    markdownTable(["Field", "Value"], [
      ["Target / profile SHA-256", `${result.target_profile_id} / ${result.target_profile_sha256}`],
      ["Peak-throughput reference cores", formatNumber(result.performance_reference_core_count)],
      ["Documented system variants", result.system_core_count_options.map((value) => `${value} cores`).join(" / ")],
      ["Imported affinity/isolation evidence", runtimeStatus],
      ["Runtime-to-static scenario binding", runtimeBinding],
      ["Memory-bandwidth treatment", "Shared target interface ceiling is unchanged across core counts; it is not multiplied by assigned cores."],
    ]),
    markdownTable(["System", "AI", "Housekeeping", "Allocation", "INT8 / FP32 ceiling", "Theoretical roofline floor", "Utilization-adjusted roofline", "Predicted overhead", "Steady estimate", "Cold estimate", "1-core ratio"], result.scenarios.map((row) => [
      formatNumber(row.system_core_count),
      formatNumber(row.ai_assigned_core_count),
      formatNumber(row.housekeeping_core_count),
      row.allocation_class,
      `${Number(row.int8_issue_ceiling_gops).toFixed(3)} / ${Number(row.fp32_issue_ceiling_gops).toFixed(3)} GOPS`,
      formatUs(row.theoretical_roofline_floor_us),
      formatUs(row.utilization_adjusted_roofline_estimate_us),
      formatUs(row.predicted_runtime_overhead_us),
      formatUs(row.steady_state_estimate_us),
      formatUs(row.cold_start_estimate_us),
      `${Number(row.estimate_ratio_vs_one_ai_core).toFixed(3)}x`,
    ])),
    `Method: ${result.method}`,
    `> ${result.isolation_evidence_boundary}`,
  ].join("\n\n");
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function emittedNumber(analysis, key, formatter = formatNumber) {
  return hasOwn(analysis, key) ? formatter(Number(analysis?.[key] || 0)) : "not emitted";
}

function percentOf(numerator, denominator) {
  const den = Number(denominator || 0);
  return den > 0 ? formatPercent(Number(numerator || 0) / den) : "N/A";
}

function nativeItemSummary(items = [], kind = "item") {
  if (!Array.isArray(items) || !items.length) return `0 ${kind}(s) emitted`;
  const top = items.slice(0, 3).map((item) => {
    const title = item.title || item.category || item.tone || kind;
    const prefix = item.id || (Number.isFinite(Number(item.op_index)) && Number(item.op_index) >= 0 ? `#${padOp(item.op_index)}` : "");
    if (prefix && String(title).startsWith(prefix)) return title;
    return [prefix, title].filter(Boolean).join(" ");
  });
  return `${formatNumber(items.length)} ${kind}(s): ${top.join(" / ")}${items.length > top.length ? " / ..." : ""}`;
}

function topologyAnnotationSummary(ops = []) {
  const annotated = ops.filter((op) => op.topo_role || hasOwn(op, "topo_depth") || hasOwn(op, "topo_fan_out_max"));
  if (!annotated.length) return "not emitted";
  const counts = new Map();
  let maxDepth = 0;
  let maxFanOut = 0;
  for (const op of annotated) {
    const role = op.topo_role || "unclassified";
    counts.set(role, (counts.get(role) || 0) + 1);
    maxDepth = Math.max(maxDepth, Number(op.topo_depth || 0));
    maxFanOut = Math.max(maxFanOut, Number(op.topo_fan_out_max || 0));
  }
  const roleText = [...counts.entries()]
    .sort(([a], [b]) => String(a).localeCompare(String(b)))
    .map(([role, count]) => `${role}:${formatNumber(count)}`)
    .join(" / ");
  return `${roleText}; max depth ${formatNumber(maxDepth)}; max fan-out ${formatNumber(maxFanOut)}`;
}

function bottleneckComponentSummary(ops = [], target = {}) {
  const estimated = ops.filter((op) => [
    "bottleneck_compute_us",
    "bottleneck_memory_us",
    "bottleneck_packing_us",
    "bottleneck_break_us",
    "bottleneck_fallback_us",
    "bottleneck_total_us",
  ].some((key) => hasOwn(op, key)));
  if (!estimated.length) return "not emitted";
  const totals = {
    compute: 0,
    memory: 0,
    packing: sumNumbers(estimated.map((op) => op.bottleneck_packing_us)),
    break: sumNumbers(estimated.map((op) => op.bottleneck_break_us)),
    fallback: sumNumbers(estimated.map((op) => op.bottleneck_fallback_us)),
    total: sumNumbers(estimated.map((op) => op.bottleneck_total_us)),
  };
  for (const op of estimated) {
    const compute = Number(op.bottleneck_compute_us || 0);
    const memory = Number(op.bottleneck_memory_us || 0);
    const bound = Math.max(compute, memory);
    if (compute >= memory) totals.compute += bound;
    else totals.memory += bound;
  }
  const componentTotal = totals.compute + totals.memory + totals.packing + totals.break + totals.fallback;
  const displayed = apportionedPercentages([
    totals.compute,
    totals.memory,
    totals.packing,
    totals.break,
    totals.fallback,
  ]);
  const consistency = Math.abs(componentTotal - totals.total) <= Math.max(1e-6, totals.total * 1e-9) ? "sum check ok" : "sum check mismatch";
  const steadyTotal = totals.compute + totals.memory + totals.fallback;
  return `HEURISTIC profile-derived cold-start composition using one common denominator (${formatUs(totals.total)}): compute-dominant steady base ${displayed[0]}, memory-dominant steady base ${displayed[1]}, one-time packing ${displayed[2]}, partition-planning setup ${displayed[3]}, fallback traffic ${displayed[4]} (display sum 100.0%; ${consistency}). Steady-state ${formatUs(steadyTotal)} = compute-dominant base ${formatUs(totals.compute)} + memory-dominant base ${formatUs(totals.memory)} + fallback ${formatUs(totals.fallback)}; cold-start ${formatUs(totals.total)} = steady-state + one-time packing ${formatUs(totals.packing)} + setup ${formatUs(totals.break)}. These compute/memory values are mutually exclusive attribution buckets selected by per-op max(compute,memory), not raw compute and raw memory estimates to add together. The setup term is an unmeasured profile constant, not per-inference latency. Per op base=max(arithmetic ops/(configured effective peak x compute-utilization factor ${Number(target.compute_utilization_factor ?? 1).toFixed(2)}), logical bytes/effective bandwidth). Absolute microsecond outputs are uncalibrated planning-profile quantities, not target-device latency predictions.`;
}

function apportionedPercentages(values) {
  const normalized = values.map((value) => Math.max(0, Number(value || 0)));
  const total = normalized.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return normalized.map(() => "N/A");
  const rawTenths = normalized.map((value) => value / total * 1000);
  const tenths = rawTenths.map(Math.floor);
  let remaining = 1000 - tenths.reduce((sum, value) => sum + value, 0);
  const order = rawTenths
    .map((value, index) => ({ index, remainder: value - tenths[index] }))
    .sort((a, b) => b.remainder - a.remainder || a.index - b.index);
  for (let i = 0; i < remaining; i++) tenths[order[i % order.length].index] += 1;
  return tenths.map((value) => `${(value / 10).toFixed(1)}%`);
}

function misalignedChannelOpSummary(analysis) {
  const ops = (analysis?.ops || [])
    .filter((op) => op.channel_alignment_status === "misaligned")
    .sort((a, b) => Number(b.channel_tail_overhead_percent || 0) - Number(a.channel_tail_overhead_percent || 0) || Number(a.index || 0) - Number(b.index || 0));
  if (!ops.length) return "0";
  const details = ops.map((op) => {
    const multiples = (op.xnnpack_kernel_alignment_multiples || []).length
      ? op.xnnpack_kernel_alignment_multiples.map((value) => `x${value}`).join("/")
      : `x${formatNumber(op.channel_alignment_multiple || 0)}`;
    const min = Number(op.channel_tail_overhead_percent_min ?? op.channel_tail_overhead_percent ?? 0);
    const max = Number(op.channel_tail_overhead_percent_max ?? op.channel_tail_overhead_percent ?? 0);
    const range = formatPercentRange(min, max);
    return `#${padOp(op.index)} ${op.name} C=${formatNumber(op.output_channels || 0)} vs ${multiples} (${range} inactive lanes)`;
  });
  return `${formatNumber(ops.length)} op(s); representative highest-ranked op ${details[0]}; all misaligned ops: ${details.join(" / ")}`;
}

function quantizationHoleSummary(analysis) {
  const holes = Array.isArray(analysis?.quant_holes) ? analysis.quant_holes : [];
  const count = hasOwn(analysis, "quant_hole_count") ? Number(analysis.quant_hole_count || 0) : holes.length;
  const impact = hasOwn(analysis, "quant_hole_mac_impact") ? formatPercent(analysis.quant_hole_mac_impact || 0) : "not emitted";
  if (!holes.length) return `${formatNumber(count)} serialized mid-graph 8-bit/FP32 boundary op(s); maximum graph-neighbor MAC share ${impact}; float-island regions are not inferred`;
  const top = holes.slice(0, 4).map((hole) => `#${padOp(hole.op_index)} ${hole.op_name || "Q/DQ"} ${hole.hole_class || "mixed"} ${hole.from_dtype || "?"}->${hole.to_dtype || "?"} (${formatPercent(hole.adjacent_mac_percent || 0)} adjacent MACs)`);
  return `${formatNumber(count)} serialized mid-graph 8-bit/FP32 boundary op(s); maximum graph-neighbor MAC share ${impact}; ${top.join(" / ")}${holes.length > top.length ? " / ..." : ""}; float-island regions are not inferred`;
}

function trafficFamilySummary(items = []) {
  if (!Array.isArray(items) || !items.length) return "none emitted";
  return items.slice(0, 5).map((item) => `${item.name || "unknown"}:${formatBytes(item.estimated_bytes || 0)} (${formatPercent(item.byte_percent || 0)} bytes, ${formatPercent(item.mac_percent || 0)} MACs)`).join(" / ");
}

function exportOnlyArtifactSummary(analysis) {
  const csv = String(analysis?.roofline_csv || "").trim();
  const csvRows = csv ? Math.max(0, csv.split(/\r?\n/).length - 1) : 0;
  const mermaid = String(analysis?.stage_mermaid || "").trim();
  return `roofline CSV ${csvRows ? `${formatNumber(csvRows)} data row(s)` : "not emitted"}; Mermaid stage graph ${mermaid ? `${formatNumber(mermaid.length)} chars available` : "not emitted"}`;
}

function staticAuditTimingSummary(analysis) {
  const timing = analysis?.static_audit_timing;
  if (!timing || !Number.isFinite(Number(timing.wall_ms))) return "not measured in this report session";
  const core = Number.isFinite(Number(timing.core_static_analysis_ms))
    ? `${Number(timing.core_static_analysis_ms).toFixed(3)} ms core static analysis`
    : "core interval not measured";
  return `${Number(timing.wall_ms).toFixed(3)} ms complete browser workflow / ${core} / ${formatNumber(timing.comparison_target_count || 0)} comparison target(s); ${timing.evidence_class || "MEASURED_BROWSER_WALL_CLOCK"}; not inference latency`;
}

function computedAnalysisCoverageMarkdown(analysis) {
  const ops = analysis?.ops || [];
  const onnx = isOnnxAnalysis(analysis);
  const quantResearchCoverage = onnx
    ? null
    : analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const advancedInteger = onnx
    || ["full_integer", "mixed_integer"].includes(quantResearchCoverage?.artifact_class);
  const totalMacs = Number(analysis?.total_macs || 0);
  const delegatedMacs = Number(analysis?.delegated_macs || 0);
  const fallbackMacs = Number(analysis?.fallback_macs || 0);
  const delegatedBytes = Number(analysis?.delegated_estimated_bytes || 0);
  const fallbackBytes = Number(analysis?.fallback_estimated_bytes || 0);
  const totalBytes = delegatedBytes + fallbackBytes;
  const rows = [
    [onnx ? "Operator/opset entries" : "Operator code table entries", onnx
      ? (analysis?.opsets || []).map((opset) => `${opset.domain || "ai.onnx"}:${opset.version || "unknown"}`).join(" / ") || "not emitted"
      : emittedNumber(analysis, "operator_codes")],
    [onnx ? "Assessed arithmetic ops" : "Total arithmetic ops", emittedNumber(analysis, "total_ops")],
    ["Topology annotations", topologyAnnotationSummary(ops)],
    ["Bottleneck component totals", bottleneckComponentSummary(ops, analysis?.target_profile)],
    ["Raw native analyzer signals", nativeItemSummary(analysis?.findings || [], "signal")],
    ["Raw native optimization hints", nativeItemSummary(analysis?.recommendations || [], "hint")],
    ["Activation precision boundary details", quantizationHoleSummary(analysis)],
    ["Browser static-audit timing", staticAuditTimingSummary(analysis)],
    ["Export-only computed artifacts", exportOnlyArtifactSummary(analysis)],
  ];
  if (!onnx) {
    rows.splice(2, 0,
      ["Conditionally delegatable / predicted-fallback MACs", hasOwn(analysis, "delegated_macs") || hasOwn(analysis, "fallback_macs")
        ? `${formatNumber(delegatedMacs)} / ${formatNumber(fallbackMacs)} (${percentOf(delegatedMacs, totalMacs)} delegated)`
        : "not emitted"],
      ["Conditionally delegatable / predicted-fallback logical bytes", hasOwn(analysis, "delegated_estimated_bytes") || hasOwn(analysis, "fallback_estimated_bytes")
        ? `${formatBytes(delegatedBytes)} / ${formatBytes(fallbackBytes)} (${percentOf(fallbackBytes, totalBytes)} fallback)`
        : "not emitted"],
      ["XNNPACK predicted partition breakdown", `total = non-structural ${emittedNumber(analysis, "xnnpack_effective_chain_breaks")} + structural ${emittedNumber(analysis, "xnnpack_structural_chain_breaks")}; zero-MAC non-structural ${emittedNumber(analysis, "xnnpack_zero_mac_chain_breaks")} is a subset of non-structural; longest candidate segment ${emittedNumber(analysis, "xnnpack_longest_chain")} op(s)`],
      advancedInteger ? ["Kernel extremum witnesses", analysis?.kernel_extremum_witness
        ? `${formatNumber(analysis.kernel_extremum_witness.assessed_channel_count)} channel(s); ${formatNumber(analysis.kernel_extremum_witness.witness_assignment_count)} canonical term assignment(s); ${formatNumber(analysis.kernel_extremum_witness.fixed_point_endpoint_evaluation_count)} fixed-point endpoint execution(s); ${formatNumber(analysis.kernel_extremum_witness.build_mode_divergent_endpoint_count)} default/single code difference(s)`
        : "not emitted"] : null,
      advancedInteger ? ["Channel vitality proof", analysis?.channel_vitality
        ? `${formatNumber(analysis.channel_vitality.assessed_channel_count)} channel(s); ${formatNumber(analysis.channel_vitality.dual_mode_constant_output_channel_count)} dual-mode constant; ${formatNumber(analysis.channel_vitality.nonconstant_accumulator_dual_mode_constant_channel_count)} variable-accumulator constant; ${formatNumber(analysis.channel_vitality.mode_dependent_constant_output_channel_count)} build-mode-dependent constant`
        : "not emitted"] : null,
      advancedInteger ? ["Fixed-point build-mode equivalence", analysis?.rounding_equivalence
        ? `${formatNumber(analysis.rounding_equivalence.assessed_channel_count)} channel(s); ${formatNumber(analysis.rounding_equivalence.equivalent_channel_count)} equivalent / ${formatNumber(analysis.rounding_equivalence.divergent_channel_count)} divergent; ${formatNumber(analysis.rounding_equivalence.divergent_state_count_decimal)} of ${formatNumber(analysis.rounding_equivalence.interval_state_count_decimal)} interval state(s); maximum ${formatNumber(analysis.rounding_equivalence.maximum_absolute_output_delta || 0)} code`
        : "not emitted"] : null,
      advancedInteger ? ["Accumulator reachability", analysis?.accumulator_reachability
        ? `${formatNumber(analysis.accumulator_reachability.exact_reachable_divergent_state_count_decimal)} exact kernel-local divergent state(s); ${formatNumber(analysis.accumulator_reachability.provably_unreachable_divergent_state_count_decimal)} residue-excluded / ${formatNumber(analysis.accumulator_reachability.unresolved_divergent_state_count_decimal)} unresolved; ${formatNumber(analysis.accumulator_reachability.complete_integer_interval_channel_count)} complete integer and ${formatNumber(analysis.accumulator_reachability.complete_modular_lattice_channel_count)} complete modular channel(s)`
        : "not emitted"] : null,
      advancedInteger ? ["Numerical ABI propagation", analysis?.numerical_abi_propagation
        ? `${formatNumber(analysis.numerical_abi_propagation.exact_local_counterexample_source_op_count)} of ${formatNumber(analysis.numerical_abi_propagation.divergent_source_op_count)} divergent source op(s) have exact kernel-local counterexamples; ${formatNumber(analysis.numerical_abi_propagation.exact_local_divergent_state_count_decimal)} exact reachable / ${formatNumber(analysis.numerical_abi_propagation.residue_excluded_divergent_state_count_decimal)} residue-excluded / ${formatNumber(analysis.numerical_abi_propagation.unresolved_divergent_state_count_decimal)} unresolved states; exact-qualified union ${formatNumber(analysis.numerical_abi_propagation.exact_unique_reachable_op_count)} ops / ${formatNumber(analysis.numerical_abi_propagation.exact_unique_predicted_boundary_edge_count)} predicted boundaries`
        : "not emitted"] : null,
      advancedInteger ? ["Model-input tensor ABI witness", analysis?.input_counterexample
        ? `${formatNumber(analysis.input_counterexample.tensor_abi_constructive_source_op_count)} of ${formatNumber(analysis.input_counterexample.exact_local_source_op_count)} exact-local source op(s) constructively realized; ${formatNumber(analysis.input_counterexample.tensor_abi_constructive_channel_count)} channels / ${formatNumber(analysis.input_counterexample.tensor_abi_constructive_divergent_state_count_decimal)} exact states; ${formatNumber(analysis.input_counterexample.upstream_activation_unresolved_source_op_count)} upstream-constrained source(s)`
        : "not emitted"] : null,
      advancedInteger ? ["Pixel-to-tensor preprocessing realizability", analysis?.preprocessing_realizability
        ? `${formatNumber(analysis.preprocessing_realizability.exact_tensor_realization_candidate_count)} exact / ${formatNumber(analysis.preprocessing_realizability.non_exact_candidate_count)} non-exact explicit candidate contract(s); portfolio ${analysis.preprocessing_realizability.portfolio_ledger_sha256}`
        : "not emitted"] : null,
      advancedInteger ? ["Residual contract migration", analysis?.contract_migration
        ? `${formatNumber(analysis.contract_migration.candidate_scenario_count)} candidate scenario(s); ${formatNumber(analysis.contract_migration.direct_consumer_count)} direct parameter consumer(s); ${formatNumber(analysis.contract_migration.assessed_kernel_channel_scenario_count)} kernel channel scenario(s); ${formatNumber(analysis.contract_migration.bias_code_changed_channel_scenario_count)} bias code change(s); ${formatNumber(analysis.contract_migration.add_parameter_encoding_changed_scenario_count)} ADD encoding change(s); ${formatNumber(analysis.contract_migration.reachable_downstream_op_union_count)} structurally reachable op(s)`
        : "not emitted"] : null,
      advancedInteger ? ["Residual step response", analysis?.residual_step_response
        ? `${formatNumber(analysis.residual_step_response.contract_response_count)} contract response(s); ${formatNumber(analysis.residual_step_response.total_transition_count)} exact branch transition(s); ${formatNumber(analysis.residual_step_response.total_joint_interior_cell_count)} joint interior cell(s); ${signedCount(analysis.residual_step_response.containment_additional_silent_transition_count)} containment silent transition delta; ${formatNumber(analysis.residual_step_response.containment_removed_rounded_clamp_pair_count)} rounded clamp pair(s) removed`
        : "not emitted"] : null,
      advancedInteger ? ["Residual contract distortion", analysis?.residual_contract_distortion
        ? `${formatNumber(analysis.residual_contract_distortion.scenario_count)} candidate scenario(s); ${formatNumber(analysis.residual_contract_distortion.total_enumerated_pair_count)} exact current-versus-candidate pair(s); ${formatNumber(analysis.residual_contract_distortion.rescued_current_clamp_pair_instance_count)} current clamp instance(s) rescued; ${formatNumber(analysis.residual_contract_distortion.ideal_error_improved_pair_count)} ideal-error improvement(s); ${formatNumber(analysis.residual_contract_distortion.ideal_error_worsened_pair_count)} worsening(s); maximum p99 ${Number(analysis.residual_contract_distortion.maximum_p99_contract_delta_current_steps).toFixed(6)} current steps`
        : "not emitted"] : null,
      ["Conv/FC weight op inventory", `Conv/depthwise ${emittedNumber(analysis, "conv_weight_ops")} (${emittedNumber(analysis, "conv_packing_warn_ops")} packing warn); FC ${emittedNumber(analysis, "fc_ops")} (${emittedNumber(analysis, "fc_packing_warn_ops")} packing warn)`],
      ["Fallback traffic by op family", trafficFamilySummary(analysis?.fallback_traffic_by_op_family || [])],
      ["Predicted structural delegate-break watchlist", compactCountItems(analysis?.suspects || [])],
    );
  } else {
    rows.splice(2, 0,
      ["MAC assessment coverage", analysis?.mac_assessment?.detail || "not emitted"],
      ["MAC-unassessed compute ops", analysis?.mac_assessment?.not_assessed?.length
        ? analysis.mac_assessment.not_assessed.map((op) => `#${padOp(op.index)} ${op.name}: ${op.reason}`).join(" / ")
        : "none"],
      ["Algorithm-dependent arithmetic", analysis?.mac_assessment?.algorithm_dependent_arithmetic?.length
        ? analysis.mac_assessment.algorithm_dependent_arithmetic.map((op) => `#${padOp(op.index)} ${op.name}: ${op.reason}`).join(" / ")
        : "none"],
      ["Execution-provider computed totals", "not assessable; ONNX Runtime EP assignment was not modeled"],
      ["Initializer value-dependent metrics", analysis?.weight_integrity?.status === "assessed"
        ? `${formatNumber(analysis.weight_integrity.weight_tensors_scanned)} initializer tensor(s) / ${formatNumber(analysis.weight_integrity.logical_elements_assessed ?? analysis.weight_integrity.elements_scanned)} logical scalar element(s) assessed (${formatNumber(analysis.weight_integrity.stored_weight_values_decoded ?? analysis.weight_integrity.elements_scanned)} stored decoded, ${formatNumber(analysis.weight_integrity.implicit_zero_elements || 0)} sparse implicit zeros); coverage ${analysis.weight_integrity.coverage_status || "complete"}`
        : "not assessed; no supported embedded or verified-external numeric initializer payload was available"],
    );
  }
  return [
    "## Computed Analysis Coverage (DERIVED/ESTIMATED)",
    markdownTable(["Computed field group", "Report value"], rows.filter(Boolean)),
    "> Coverage note: this table surfaces analyzer values that are otherwise easy to miss because they also feed charts, CSV/raw exports, or action-queue heuristics. Bottleneck components are estimated signals, and analyzer op-total is the per-op total emitted by the analyzer rather than a sum-of-components claim. Rows marked not emitted are not silently interpreted elsewhere in this report.",
  ].join("\n");
}

function advancedStaticProofCoverageMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const coverage = analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  if (!["full_integer", "mixed_integer"].includes(coverage.artifact_class)) return "";
  const rows = [
    ["Accumulator headroom", "accumulator_atlas"],
    ["Requantization fidelity", "requantization_fidelity"],
    ["Kernel extremum witness", "kernel_extremum_witness"],
    ["Channel vitality", "channel_vitality"],
    ["Rounding equivalence", "rounding_equivalence"],
    ["Accumulator reachability", "accumulator_reachability"],
    ["Numerical ABI propagation", "numerical_abi_propagation"],
    ["Model-input witness", "input_counterexample"],
    ["Preprocessing realizability", "preprocessing_realizability"],
    ["Residual lattice", "quantization_lattice"],
    ["Contract migration", "contract_migration"],
    ["Residual step response", "residual_step_response"],
    ["Residual contract distortion", "residual_contract_distortion"],
  ].map(([label, key]) => {
    const result = analysis?.[key];
    return [
      label,
      result?.status || "not emitted",
      result?.evidence_class || "not emitted",
      result?.schema || "not emitted",
      `${result?.source_evidence_schema || result?.source_input_counterexample_schema || "not emitted"}${result?.source_input_counterexample_portfolio_sha256 ? `; SHA-256 ${result.source_input_counterexample_portfolio_sha256}` : ""}`,
    ];
  });
  return [
    "## Advanced Static Proof Coverage (OBSERVED/DERIVED)",
    markdownTable(["Proof module", "Status", "Evidence class", "Schema", "Source evidence binding"], rows),
    "> This is an execution and schema ledger, not a claim that every module is applicable or complete. Detailed sections retain assessed, excluded, unresolved, and not-applicable denominators.",
  ].join("\n");
}

function checkpointLineageMarkdown(analysis, quant = {}) {
  const description = String(analysis?.metadata_presence?.description || "");
  const converter = /toco/i.test(description)
    ? "TOCO indicated by artifact description; exact converter version not embedded"
    : "not embedded in the artifact";
  const quantConfig = quant.classification === "not_quantized_float"
    ? "not applicable to this FP32 artifact; intended precision still belongs in the release manifest"
    : quant.classification === "float16_weight_storage"
      ? "FP16 constant storage and FP32 expansion are observed; no 8-bit calibration contract is embedded"
      : "quantization parameters are observed in tensors, but converter/calibration recipe is not embedded";
  return [
    "## Checkpoint-to-Artifact Provenance (NOT_ASSESSABLE)",
    markdownTable(["Field", "Value"], [
      ["Source checkpoint SHA-256", "not provided"],
      ["Export framework and version", "not embedded"],
      ["Converter", converter],
      ["Export configuration", "not embedded"],
      ["Quantization configuration", quantConfig],
      ["Representative dataset ID", "not provided"],
      ["Build pipeline ID", "not provided"],
      ["Software release ID", "not provided"],
      ["Model requirement ID", "not provided"],
      ["Previous released artifact SHA-256", "not provided"],
      ["Traceability conclusion", "incomplete; the deployment artifact does not bind source checkpoint, converter, release, and requirement lineage"],
    ]),
    "> Release manifest requirement: bind checkpoint hash, converter/framework version, export options, quantization/calibration configuration, representative dataset identifier where applicable, pipeline ID, software release ID, requirement ID, and previous artifact SHA-256 outside the TFLite/ONNX artifact when they are not embedded.",
  ].join("\n");
}

function predictedBoundaryPayloadText(analysis) {
  const inventory = predictedPartitionBoundaryInventory(analysis);
  if (!inventory) return "NOT_ASSESSED; predicted boundary-edge inventory not emitted";
  if (inventory.summed_edge_payload_bytes != null) {
    return `${formatBytes(inventory.summed_edge_payload_bytes)} across ${formatNumber(inventory.edge_count)} internal edge(s); ${formatBytes(inventory.unique_tensor_payload_bytes || 0)} across ${formatNumber(inventory.unique_tensor_count)} unique tensor(s)`;
  }
  return `PARTIAL: ${formatBytes(inventory.assessed_edge_payload_bytes || 0)} across ${formatNumber(inventory.assessed_payload_edge_count)} assessed edge(s); ${formatNumber(inventory.unassessed_payload_edge_count)} edge payload(s) not assessed`;
}

function predictedBoundaryEdgesMarkdown(analysis) {
  const inventory = predictedPartitionBoundaryInventory(analysis);
  if (!inventory?.edges?.length) return "No internal producer-to-consumer edge crosses the static predicted execution domains.";
  return [
    markdownTable(["Tensor edge", "Shape / dtype", "Predicted transition", "Logical payload / binding", "Materialization"], inventory.edges.slice(0, 24).map((edge) => [
      `T${edge.tensor_index} ${code(edge.tensor_name || "-")}; #${padOp(edge.producer_op_index)} ${edge.producer_op_name} -> #${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`,
      `${(edge.tensor_shape || []).join("x") || "scalar"} / ${edge.tensor_dtype || "-"}`,
      `${edge.producer_domain} -> ${edge.consumer_domain} (${edge.direction})`,
      edge.payload_bytes == null ? `NOT_ASSESSED; ${edge.payload_reason || "unknown static payload"}` : `${formatBytes(edge.payload_bytes)} (${formatNumber(edge.payload_bytes)} B); ${edge.payload_binding || "static"}; ${edge.payload_status || "assessed"}`,
      edge.materialization_status || "NOT_ASSESSABLE_FROM_STATIC_ARTIFACT",
    ])),
    inventory.edges.length > 24 ? `> ${formatNumber(inventory.edges.length - 24)} additional edge(s) remain in structured evidence.` : "",
    `> Assignment evidence ${inventory.assignment_evidence_class}; payload evidence ${inventory.payload_evidence_class}; coverage ${inventory.payload_coverage_status}. Logical payload does not assert a runtime copy, layout conversion, allocator allocation, or latency cost.`,
  ].filter(Boolean).join("\n");
}

function staticAuditConclusionMarkdown(analysis, artifactIntegrity, quant, runtimeBenchmarkResults) {
  const inputCount = (analysis.inputs || []).length;
  const outputCount = (analysis.outputs || []).length;
  const integrityClean = !artifactIntegrity.custom_like_ops.length
    && !artifactIntegrity.tflite_out_of_bounds_tensor_buffers
    && !artifactIntegrity.external_data_incomplete_tensor_count;
  const onnx = isOnnxAnalysis(analysis);
  const successfulBenchmarks = runtimeBenchmarkResults.filter((row) => row?.ok === true);
  const delegateText = onnx
    ? "ONNX Runtime execution-provider assignment is not modeled; provider-specific placement requires ORT session/profiling evidence."
    : Number(analysis.xnnpack_effective_chain_breaks || 0)
    ? `${plural(analysis.xnnpack_effective_chain_breaks, "non-structural predicted partition break")}; runtime confirmation is required.`
    : "No non-structural predicted partition break was reported by the rulepack.";
  const liveText = analysis.tensor_liveness?.assessed === false
    ? "Peak live activation payload not emitted because required tensor shapes were not fully inferred."
    : `Graph-liveness activation payload estimated at ${formatBytes(analysis.tensor_liveness?.peak_bytes)}. Weights, byte alignment, allocator overhead, delegate buffers, and runtime scratch are excluded.`;
  return markdownTable(["Area", "Conclusion"], [
    ["Artifact integrity", onnx
      ? `${integrityClean ? "No graph-level structural defect was observed in the parsed scope." : artifactIntegrity.summary} ${onnxWeightIntegrityConclusion(analysis.weight_integrity)}`
      : integrityClean ? "No structural integrity defect observed by the static artifact checks." : artifactIntegrity.summary],
    ["Interface", `${plural(inputCount, "input tensor")} and ${plural(outputCount, "output tensor")} identified. Preprocessing contract ${analysis.metadata_presence?.documented_preprocessing ? "explicitly parsed from artifact process units with every referenced file payload verified" : "not machine-verifiably embedded"}; output semantic class mapping and label order ${analysis.metadata_presence?.output_semantics_documented ? "declared in parsed output metadata, payload/CRC/SHA-256 verified, and cardinality-bound where axis labels are used; label ontology and task meaning remain release-contract evidence" : "not embedded"}.`],
    ["Quantization", quant.label || quant.classification || "unknown"],
    ["Delegate compatibility", delegateText],
    ["Static memory", liveText],
    ["Runtime performance", successfulBenchmarks.length
      ? `Measured for ${successfulBenchmarks.length}/${runtimeBenchmarkResults.length} attempted browser backend path(s); exact input, output, and timing-phase contracts are listed below.`
      : runtimeBenchmarkResults.length ? `Attempted for ${runtimeBenchmarkResults.length} browser backend path(s), but none completed; no latency evidence was produced.` : "Not measured in this report."],
    ["Task and clinical performance", "Not assessed from the deployment artifact."],
    ["Release decision", "Not assessable from static artifact evidence alone."],
  ]);
}

function blockInventoryMarkdown(analysis) {
  const inventory = analysis.block_inventory;
  if (!inventory || inventory.status !== "assessed") {
    return [
      `Status: ${inventory?.status || "not emitted"}.`,
      "",
      isOnnxAnalysis(analysis)
        ? "ONNX architecture-block semantics are intentionally not inferred by the TFLite block contract."
        : "No deterministic block inventory was emitted for this graph.",
    ].join("\n");
  }
  return [
    `Schema: \`${inventory.schema}\`; evidence ${inventory.evidence_class || "DERIVED"}; method ${inventory.method || "not emitted"}; ${formatNumber(inventory.stage_count)} stage(s), ${formatNumber(inventory.block_count)} block(s), ${formatNumber(inventory.semantic_block_count)} graph-semantic match(es), ${formatNumber(inventory.unnamed_block_count)} shape-grouped fallback(s).`,
    "",
    markdownTable(
      ["Stage", "Block", "Type / extraction", "Ops", "Spatial", "Channels", "MAC share", "Modeled time", "Max logical L1", "Traffic", "Predicted breaks"],
      (inventory.blocks || []).map((block) => [
        String(block.stage_index),
        `${block.block_id} / ${block.display_name}`,
        `${block.block_type}; ${block.extraction?.method || "unknown"} ${block.extraction?.confidence || ""}`.trim(),
        (block.op_indices || []).map((index) => `#${padOp(index)}`).join(", "),
        `${block.spatial?.input_h ?? "?"}x${block.spatial?.input_w ?? "?"} -> ${block.spatial?.output_h ?? "?"}x${block.spatial?.output_w ?? "?"}`,
        `${block.channels?.input ?? "?"} -> ${block.channels?.expand ?? "-"} -> ${block.channels?.output ?? "?"}`,
        formatPercent(block.aggregates?.mac_percent || 0),
        `${Number(block.aggregates?.modeled_time_ms || 0).toFixed(3)} ms (${block.aggregates?.time_evidence_class || "not assessed"})`,
        block.aggregates?.l1_max_ratio == null ? "N/A" : `${Number(block.aggregates.l1_max_ratio).toFixed(2)}x`,
        formatBytes(block.aggregates?.logical_traffic_bytes || 0),
        formatNumber(block.aggregates?.predicted_break_count || 0),
      ]),
    ),
    "",
    `> ${inventory.interpretation_boundary || "Block labels describe serialized graph motifs, not recovered training-framework modules."}`,
  ].join("\n");
}

function serializedArtifactEngineeringReportArtifacts(sourceAnalysis, {
  identity = {},
  findingsContext = {},
  runtimeEvidence = {},
  executionPlacementEvidence = null,
  verification = null,
  generatedAt = new Date().toISOString(),
} = {}) {
  const fieldAccess = createAnalysisFieldAccessTracker(sourceAnalysis);
  const analysis = fieldAccess.analysis;
  const format = String(analysis.format || "").toLowerCase();
  const executionPlacement = executionPlacementEvidence || buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const calibrationValidation = runtimeEvidence?.calibrationValidationResult || runtimeEvidence?.representative_dataset_validation || null;
  const importedRuntimeEvidence = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || null;
  const coreMlComputePlan = importedRuntimeEvidence?.schema === "deepbom.coreml_compute_plan.v1" ? importedRuntimeEvidence : null;
  const label = format === "gguf" ? "GGUF" : format === "safetensors" ? "SafeTensors" : format === "executorch" ? "ExecuTorch" : "Core ML";
  const modelName = reportModelName(analysis, identity);
  const findings = buildFindingsRegister(analysis, { runtimeEvidence, ...findingsContext });
  const findingRows = findings.findings || [];
  const inputs = Array.isArray(analysis.inputs) ? analysis.inputs : [];
  const outputs = Array.isArray(analysis.outputs) ? analysis.outputs : [];
  const tensors = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const tensorLimit = 200;
  const tensorRows = tensors.slice(0, tensorLimit).map((tensor) => [
    tensor.index ?? "-",
    code(tensor.name || `tensor_${tensor.index ?? "?"}`),
    Array.isArray(tensor.shape) ? code(`[${tensor.shape.join(", ")}]`) : "not serialized",
    code(tensor.dtype || "UNKNOWN"),
    tensor.byte_length == null && tensor.buffer_data_length == null ? "not statically bounded" : `${formatNumber(tensor.byte_length ?? tensor.buffer_data_length)} B`,
    tensor.storage_status || tensor.role || (format === "coreml" ? "named graph blob" : "assessed"),
  ]);
  const interfaceRows = [
    ...inputs.map((tensor) => ["input", code(tensor.name || "unnamed"), code(tensor.dtype || "UNKNOWN"), Array.isArray(tensor.shape) ? code(`[${tensor.shape.join(", ")}]`) : "not serialized"]),
    ...outputs.map((tensor) => ["output", code(tensor.name || "unnamed"), code(tensor.dtype || "UNKNOWN"), Array.isArray(tensor.shape) ? code(`[${tensor.shape.join(", ")}]`) : "not serialized"]),
  ];
  const coreMlControlFlowPartial = analysis.mac_assessment?.status === "partial_control_flow_execution_count_not_reconstructed";
  const coreMlMacDetail = coreMlControlFlowPartial
    ? `${formatNumber(analysis.mac_assessment?.nested_block_operation_count || 0)} nested-block operation(s) decoded; per-op MAC formulas are retained, but no total is emitted without branch/loop execution counts`
    : `${formatNumber(analysis.mac_assessment?.assessed_compute_ops || 0)}/${formatNumber(analysis.mac_assessment?.compute_ops || 0)} MAC-bearing operation(s) assessed${analysis.total_macs == null ? "" : ` (${formatNumber(analysis.total_macs)} MACs)`}`;
  const coreMlGraphUnit = analysis.coreml?.model_type === "mlProgram" ? "MIL SSA operation(s)"
    : analysis.coreml?.pipeline ? "pipeline operation(s)"
      : analysis.coreml?.classical_model ? "classical model operation(s)" : "legacy NeuralNetwork layer(s)";
  const execuTorchProgram = format === "executorch" && analysis.executorch_container === "pte";
  const execuTorchMac = analysis.mac_assessment || {};
  const execuTorchMacDetail = execuTorchProgram
    ? `${formatNumber(execuTorchMac.source_bound_kernel_instruction_count || 0)}/${formatNumber(execuTorchMac.kernel_instruction_count || 0)} KernelCall signatures source-bound; ${formatNumber(execuTorchMac.mac_assessed_kernel_instruction_count || 0)} MAC-assessed; ${formatNumber(execuTorchMac.mac_unassessed_kernel_instruction_count || 0)} kernel and ${formatNumber(execuTorchMac.delegate_instruction_count || 0)} delegate call(s) unassessed`
    : "";
  const graphBoundary = format === "coreml"
    ? `${formatNumber(analysis.operator_count ?? (analysis.ops || []).length)} serialized ${coreMlGraphUnit} decoded; ${coreMlMacDetail}. Runtime placement, fusion, and latency require separate execution evidence.`
    : execuTorchProgram
      ? `${formatNumber(analysis.subgraphs || 0)} ET12 execution plan(s) and ${formatNumber(analysis.operator_count || 0)} ordered instruction(s) decoded. ${execuTorchMacDetail}. Matching portable KernelCall direction is source-bound; delegate internals and runtime execution remain external.`
    : `${label} does not serialize an execution-operator DAG. Operator count, MACs, activation liveness, Q/DQ placement, delegation, and runtime latency are not numeric-zero results; they are outside this artifact contract.`;
  const reportParts = [
    `# DEEPBOM ${label} Serialized Artifact Audit`,
    "",
    `**Model artifact:** ${code(modelName)}`,
    "",
    "**Evidence level: static serialized-artifact evidence only.**",
    "",
    "## Read First",
    `- This report is bound to the artifact SHA-256 and does not inherit an execution-target profile from the viewer UI.`,
    `- ${graphBoundary}`,
    "- OBSERVED values come from bounded parsing of the selected local artifact. SOURCE_PINNED interpretations identify the exact source commit and content digest used by the analyzer.",
    "",
    "## Artifact Identity",
    markdownTable(["Field", "Value"], [
      ["Filename", code(modelName)],
      ["Format", code(label)],
      ["Artifact bytes", `${formatBytes(analysis.file_size ?? analysis.file_size_bytes ?? 0)} (${formatNumber(analysis.file_size ?? analysis.file_size_bytes ?? 0)} B)`],
      ["SHA-256", code(analysis.model_sha256 || identity.sha256 || "not computed")],
      ["Report binding", "artifact-only; no CPU, GPU, NPU, execution provider, or target profile selected or inferred"],
      ["Serialized tensors", formatNumber(analysis.tensor_count ?? tensors.length)],
      ["Serialized graph operations", format === "coreml" || execuTorchProgram ? formatNumber(analysis.operator_count ?? (analysis.ops || []).length) : "not serialized by this format"],
      ["Compute MACs", format === "coreml"
        ? analysis.total_macs == null ? `partial; ${analysis.mac_assessment?.status || "reason not emitted"}; ${coreMlMacDetail}` : `${formatNumber(analysis.total_macs)}; ${analysis.mac_assessment?.status || "assessed"}`
        : execuTorchProgram ? analysis.total_macs == null
          ? `partial; ${execuTorchMac.total_assessed_macs_decimal || "0"} assessed MACs; ${execuTorchMac.status || "unassessed rows remain"}`
          : `${analysis.total_macs_decimal ?? formatNumber(analysis.total_macs)}; ${execuTorchMac.status}` : "not applicable; no execution graph serialized"],
      ["Analyzer", `${ANALYZER_METADATA.name} ${ANALYZER_METADATA.version} / ${ANALYZER_METADATA.semanticVersion}`],
      ["Analyzer execution identity", analyzerContentVersion(ANALYZER_METADATA.semanticVersion, ANALYZER_METADATA.buildCommit, ANALYZER_METADATA.buildContentSha256)],
      ["Analyzer build commit", ANALYZER_METADATA.buildCommit || "not embedded"],
      ["Analyzer bundle content SHA-256", ANALYZER_METADATA.buildContentSha256 || "not embedded"],
      ["Report schema", ANALYZER_METADATA.schemas.engineeringReport],
      ["Generated", generatedAt],
    ]),
    "",
    coreIsolationRooflineMarkdown(analysis, runtimeEvidence),
    "",
    serializedFormatEvidenceMarkdown(analysis),
    "",
    executionPlacementMarkdown(executionPlacement),
    "",
    runtimeEnvironmentMarkdown(runtimeEvidence, [], analysis),
    "",
    calibrationValidationMarkdown(calibrationValidation, "##"),
    "",
    weightIntegrityMarkdown(analysis, analysis.weight_integrity || {}),
    "",
    coreMlStaticResourceMarkdown(analysis),
    "",
    "## External Interface Contract",
    interfaceRows.length
      ? markdownTable(["Direction", "Name", "Dtype", "Shape"], interfaceRows)
      : "No schema-addressable external input or output is serialized by this format.",
    "",
    "## Tensor Inventory",
    tensorRows.length
      ? markdownTable(["Index", "Name", "Shape", "Dtype / encoding", "Stored bytes", "Assessment"], tensorRows)
      : "No tensor descriptor was decoded.",
    ...(tensors.length > tensorLimit ? ["", `Tensor table truncated explicitly: ${formatNumber(tensorLimit)}/${formatNumber(tensors.length)} rows shown; complete descriptors remain in engineering_evidence.json and Raw Data.`] : []),
    "",
    "## Engineer Action Queue",
    findingRows.length ? actionQueueMarkdown(findings) : "No authoritative finding was emitted for the assessed serialized-artifact scope.",
    "",
    "## Assessment Boundary",
    markdownTable(["Question", "Status", "Reason"], [
      ["Tensor storage layout", "ASSESSED", "Serialized dtype/encoding, shape, and bounded byte ranges are decoded where the format provides them."],
      ["Affine scale / zero-point", format === "coreml" ? analysis.coreml?.classical_model ? "NOT APPLICABLE" : "PARTIAL" : "NOT SERIALIZED", format === "coreml" ? analysis.coreml?.classical_model ? "Classical GLM/SVM/Tree parameters are serialized FLOAT64 numerical arrays, not affine-quantized tensors." : "Decoded WeightParams encoding and packed-code contracts are reported; activation execution precision and runtime conversions are not inferred." : `${label} does not provide a general executable affine tensor contract.`],
      ["Execution graph", format === "coreml" && (analysis.ops || []).length ? (analysis.total_macs != null ? "ASSESSED_SERIALIZED_SCOPE" : "PARTIAL") : execuTorchProgram ? "ASSESSED_INSTRUCTION_SCOPE" : "NOT SERIALIZED", graphBoundary],
      ["MACs", format === "coreml" ? (analysis.total_macs == null ? "PARTIAL" : "ASSESSED_SERIALIZED_SCOPE") : execuTorchProgram ? (analysis.total_macs == null ? "PARTIAL_SOURCE_BOUND" : "ASSESSED_SOURCE_BOUND") : "NOT SERIALIZED", format === "coreml" ? `${coreMlMacDetail}; ${analysis.mac_assessment?.status || "status not emitted"}.` : execuTorchProgram ? `${execuTorchMacDetail}. The metric is nominal tensor-contraction MACs; no complete total is emitted while any kernel/delegate row is unassessed.` : "No execution graph is serialized by this format."],
      ["Peak live logical activation payload", format === "coreml" ? (analysis.tensor_liveness?.status === "assessed" ? "ASSESSED_SERIALIZED_SCOPE" : String(analysis.tensor_liveness?.status || "NOT ASSESSED").toUpperCase()) : execuTorchProgram ? "NOT ASSESSED / AOT PLAN OBSERVED" : "NOT ASSESSED", format === "coreml" ? analysis.tensor_liveness?.method || "Liveness method was not emitted; no numeric zero is substituted." : execuTorchProgram ? `${analysis.tensor_liveness?.planned_non_const_memory_decimal || "0"} B of serialized planned non-constant buffers is observed. KernelCall direction is bound where signatures match, but alias-aware control-flow liveness is not reconstructed.` : "Requires runtime allocation/liveness semantics; no numeric zero is substituted."],
      ["Runtime placement / latency", coreMlComputePlan ? "ANTICIPATED PLAN / EXECUTION NOT OBSERVED" : "NOT ASSESSED", coreMlComputePlan ? "An identity-bound MLComputePlan supplies preferred/supported devices and relative-cost estimates for the compiled model; executed placement and latency remain unobserved." : "Requires a bound runtime build, device, execution trace, and workload."],
      ["Task accuracy", "NOT ASSESSED", "Requires representative labeled data and an evaluation protocol."],
    ]),
    "",
    METRIC_COVERAGE_PLACEHOLDER,
    "",
    DECISION_COVERAGE_PLACEHOLDER,
    "",
    "## Evidence Class Legend",
    evidenceClassLegend(),
    "",
    "> Sharing note: this document embeds the local filename and artifact SHA-256. The analyzed model bytes remain local unless the user separately uploads or shares an exported report.",
    "",
    verification ? REPORT_VERIFICATION_SENTINEL : "",
    verification ? `## Report Fingerprint Registry\nRegistration code: **${verification.code}**\nRegistered report fingerprint (SHA-256): \`${verification.reportHash}\`` : "",
  ];
  const reportAccessedFieldPaths = fieldAccess.accessedFieldPaths();
  const metricCoverage = buildMetricCoverageManifest(sourceAnalysis, {
    findings: findingRows,
    runtimeResults: executionPlacementRuntimeMetricResults(runtimeEvidence),
    reportAccessedFieldPaths,
  });
  const applicableMetrics = (metricCoverage.entries || []).filter((entry) => entry.applicability === metricCoverage.format);
  reportParts[reportParts.indexOf(METRIC_COVERAGE_PLACEHOLDER)] = [
    "## Metric Coverage",
    markdownTable(["Metric family", "Status", "Evidence", "Report binding", "Method"], applicableMetrics.map((entry) => [
      entry.metric_id,
      entry.status,
      entry.evidence_class,
      entry.report_section.replace(/^#{1,6}\s+/, ""),
      entry.calculation_method,
    ])),
    `> ${formatNumber((metricCoverage.entries || []).length - applicableMetrics.length)} graph/runtime metric families are format-inapplicable and remain listed in engineering_evidence.json; they are not rendered as pseudo-zero findings in this report. Field coverage: ${formatNumber(metricCoverage.field_coverage?.report_consumed_field_pattern_count || 0)} report-consumed / ${formatNumber(metricCoverage.field_coverage?.raw_evidence_only_field_pattern_count || 0)} raw-evidence-only / ${formatNumber(metricCoverage.field_coverage?.unbound_field_pattern_count || 0)} unbound.`,
  ].join("\n\n");
  reportParts[reportParts.indexOf(DECISION_COVERAGE_PLACEHOLDER)] = decisionCoverageMarkdown(metricCoverage);
  return { report: reportParts.join("\n"), metricCoverage, reportAccessedFieldPaths };
}

export function buildEngineeringReportArtifacts(analysis, {
  identity = {},
  runtimeBenchmarkResults = [],
  runtimeEvidence = {},
  executionPlacementEvidence = null,
  deepBomResult = null,
  perturbationResult = null,
  runtimeBasinResult = null,
  deployCurvatureResult = null,
  findingsContext = {},
  sessionPrivacy = null,
  priorSnapshot = null,
  verification = null,
  productionInterfaceContract = null,
  generatedAt = new Date().toISOString(),
} = {}) {
  if (!analysis) return { report: "", metricCoverage: null, reportAccessedFieldPaths: [] };
  if (["gguf", "safetensors", "coreml", "executorch"].includes(String(analysis.format || "").toLowerCase())) {
    return serializedArtifactEngineeringReportArtifacts(analysis, {
      identity,
      findingsContext,
      runtimeEvidence,
      executionPlacementEvidence,
      verification,
      generatedAt,
    });
  }
  const sourceAnalysis = analysis;
  const fieldAccess = createAnalysisFieldAccessTracker(sourceAnalysis);
  analysis = fieldAccess.analysis;
  const executionPlacement = executionPlacementEvidence || buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const quant = modelQuantizationStatus(analysis);
  const batchOneProjection = deriveTfliteBatchOneProjection(analysis);
  const findings = buildFindingsRegister(analysis, {
    runtimeEvidence,
    runtimeBasinResult,
    deepBomResult,
    deployCurvatureResult,
    ...findingsContext,
  });
  const artifactIntegrity = collectArtifactIntegrity(analysis, identity, analysis.file_size || 0);
  const topOps = [...(analysis.ops || [])].sort((a, b) => Number(b.macs || 0) - Number(a.macs || 0)).slice(0, 10);
  const cacheOps = (analysis.ops || [])
    .filter((op) => op.cache_payload?.status === "assessed")
    .sort((a, b) => Number(b.cache_payload?.logical_row_payload_bytes || 0) - Number(a.cache_payload?.logical_row_payload_bytes || 0))
    .slice(0, 10);
  const delegationSegments = (() => {
    const rows = [];
    const opsSeq = analysis.ops || [];
    const totalMacs = Math.max(1, Number(analysis.total_macs || 0));
    let seg = null;
    const flush = () => { if (seg) { rows.push(seg); seg = null; } };
    for (const op of opsSeq) {
      const delegated = Number(op.xnnpack_chain_id) >= 0;
      const kind = delegated ? `C${op.xnnpack_chain_id >= 0 ? op.xnnpack_chain_id : "?"}` : "N";
      if (!seg || seg.kind !== kind || (!delegated && seg.type === "predicted fallback" && seg.end !== op.index - 1)) flush();
      if (!seg) seg = { kind, type: delegated ? "conditionally delegatable candidate segment" : "predicted fallback", start: op.index, end: op.index, count: 0, macs: 0, names: new Set() };
      seg.end = op.index; seg.count += 1; seg.macs += Number(op.macs || 0); seg.names.add(op.name);
    }
    flush();
    let n = 0;
    return rows.map((row) => [
      row.type === "predicted fallback" ? `N${n++}` : row.kind,
      row.type,
      String(row.count),
      `${((row.macs / totalMacs) * 100).toFixed(1)}%`,
      `${row.start}-${row.end}`,
      row.names.size <= 2 ? [...row.names].join(", ") : "mixed",
    ]);
  })();
  const chainRows = (analysis.xnnpack_chains || []).slice(0, 12).map((chain) => [
    `C${chain.id}`,
    `${formatNumber(chain.op_count)} ops`,
    formatPercent(chain.mac_percent || 0),
    `${chain.first_op}-${chain.last_op}`,
  ]);
  const benchmarkRows = runtimeBenchmarkResults.map((row) => [
    row.backend,
    row.ok ? "ok" : "failed",
    row.compile_ms == null ? "-" : `${Number(row.compile_ms).toFixed(2)} ms`,
    row.first_run_ms == null ? "-" : `${Number(row.first_run_ms).toFixed(2)} ms`,
    row.stats?.p50 == null ? "-" : `${Number(row.stats.p50).toFixed(2)} ms`,
    row.stats?.p90 == null ? "-" : `${Number(row.stats.p90).toFixed(2)} ms`,
    row.stats?.p95 == null ? "-" : `${Number(row.stats.p95).toFixed(2)} ms`,
    row.stats?.p99 == null ? "-" : `${Number(row.stats.p99).toFixed(2)} ms`,
    row.stats?.mean == null ? "-" : `${Number(row.stats.mean).toFixed(2)} ms`,
    row.steady_stats?.p50 == null ? "-" : `${Number(row.steady_stats.p50).toFixed(2)} ms`,
    row.stats?.cv == null ? "-" : formatPercent(row.stats.cv),
    row.ok ? [
      `${row.output_count} outputs`,
      ...(row.output_contracts || []).map(benchmarkOutputContractText),
      ...(row.input_contracts || []).map(benchmarkInputContractText),
      row.external_data_runtime_binding ? benchmarkExternalDataBindingText(row.external_data_runtime_binding) : "",
      row.timing_method || "timing method not recorded",
      row.phase_counts ? `phases cold ${row.phase_counts.cold_first_runs} / warmup ${row.phase_counts.warmup_runs} / measured ${row.phase_counts.measured_runs}` : "phase counts not recorded",
      row.p99_evidence?.detail || "p99 sample adequacy not recorded",
      row.output_digest ? `digest ${row.output_digest.slice(0, 16)}` : "",
    ].filter(Boolean).join(" / ") : row.error || "-",
  ]);
  const weightIntegrity = analysis.weight_integrity || {};
  const eligibleKernelTensors = Number(weightIntegrity.eligible_kernel_tensors_scanned || 0);
  const evaluatedChannels = Number(weightIntegrity.output_channels_evaluated || 0);
  const onnx = isOnnxAnalysis(analysis);
  const bundledRuntimeVersion = runtimeEvidence?.bundledRuntimeVersion
    || runtimeEvidence?.litertRuntimeVersion
    || runtimeEvidence?.tfliteRuntimeVersion
    || runtimeEvidence?.runtimeVersion
    || "";
  const runtimeConclusion = bundledRuntimeVersion
    ? "NEEDS_RUNTIME_CONFIRMATION; artifact-side minimum is derived, but compatibility still depends on the exact deployed runtime build and flags"
    : "NOT_ASSESSABLE; bundled application LiteRT/TFLite runtime version was not provided";
  const preprocessingConsequence = runtimeEvidence?.preprocessingConsequenceResult || runtimeEvidence?.preprocessing_consequence_atlas || null;
  const calibrationValidation = runtimeEvidence?.calibrationValidationResult || runtimeEvidence?.representative_dataset_validation || null;
  const hasSuccessfulBenchmark = runtimeBenchmarkResults.some((row) => row?.ok === true)
    || (runtimeEvidence?.benchmarkResults || runtimeEvidence?.benchmark_results || []).some((row) => row?.ok === true);
  const hasBrowserRuntimeEvidence = hasSuccessfulBenchmark || preprocessingConsequence != null;
  const hasImportedRuntimeObservation = executionPlacement.runtime_observation.status === "observed";
  const hasRuntimeEvidence = hasBrowserRuntimeEvidence || hasImportedRuntimeObservation || calibrationValidation != null;
  const modelName = reportModelName(analysis, identity);
  const reportRuntimeMetricResults = executionPlacementRuntimeMetricResults(runtimeEvidence, { browserExecuted: hasBrowserRuntimeEvidence });
  if (onnx && reportRuntimeMetricResults.runtime_assignment) {
    reportRuntimeMetricResults.onnx_runtime_shape_binding = buildOnnxRuntimeShapeBinding(sourceAnalysis, reportRuntimeMetricResults.runtime_assignment);
  }
  const metricCoverageContext = {
    findings: findings.findings || findings,
    runtimeResults: reportRuntimeMetricResults,
  };
  const reportParts = [
    hasRuntimeEvidence
      ? "# DEEPBOM Artifact And Runtime Evidence Audit"
      : "# DEEPBOM Static Artifact Engineering Audit",
    "",
    `**Model artifact:** ${code(modelName)}`,
    "",
    `**Evidence level: ${hasRuntimeEvidence ? "Static + runtime evidence" : "Static only"}** - see Evidence Coverage below.`,
    "",
    "## Read First",
    "- This report is written for model/runtime engineers who need to debug deployment behavior.",
    "- It keeps the full static audit appendix but front-loads action items, target assumptions, and runtime caveats.",
    hasSuccessfulBenchmark
      ? "- Measured runtime values in this report are browser-local synthetic or prepared-input runs unless explicitly stated otherwise."
      : "- No successful runtime measurement is present; runtime values are static planning estimates unless a later section explicitly identifies imported or observed evidence.",
    "",
    modelAtGlanceMarkdown(analysis),
    "",
    scopeAndEvidenceBoundaryMarkdown(),
    "",
    "## Static Audit Conclusion",
    staticAuditConclusionMarkdown(analysis, artifactIntegrity, quant, runtimeBenchmarkResults),
    "",
    "## Engineer Action Queue",
    "Priority basis: context-free heuristic priority because no product requirement or acceptance budget was supplied. Provide cold-start budget, p95 latency budget, peak-memory budget, no-fallback requirement, and maximum artifact size to calibrate priorities.",
    "",
    actionQueueMarkdown(findings),
    "",
    "## Evidence Coverage",
    "What this report's conclusions are based on. Anything marked *not run* or *not provided* is outside the evidence in this document.",
    "",
    markdownTable(["Evidence class", "Status"], [
      ["Static analysis", "completed"],
      ["Browser runtime benchmark", hasSuccessfulBenchmark
        ? `measured (${runtimeBenchmarkResults.filter((row) => row?.ok === true).length}/${runtimeBenchmarkResults.length} backend path(s) completed)`
        : runtimeBenchmarkResults.length ? `attempted (${runtimeBenchmarkResults.length} backend path(s)); no completed measurement` : "not run"],
      ["Preprocessing consequence replay", preprocessingConsequence ? `run (${preprocessingConsequence.candidate_count} candidate contract input(s), two captures each)` : "not run"],
      ["Representative dataset validation", calibrationValidation ? `imported (${calibrationValidation.sample_count} sample(s); ledger ${calibrationValidation.ledger_sha256})` : "not imported"],
      ...executionPlacementCoverageRows(executionPlacement),
      ["Task-level regression test (reference data)", "not provided"],
      ["Clinical validation", "outside scope"],
      ["Research DEEPBOM", deepBomResult ? "run" : "not run"],
      ["Research Perturbation", perturbationResult ? "run" : "not run"],
      [RUNTIME_COMPATIBILITY_EVIDENCE_LABEL, runtimeBasinResult ? "run" : "not run"],
      ["Research Deployment Sensitivity Proxy", deployCurvatureResult ? "run" : "not run"],
    ]),
    "",
    executionPlacementMarkdown(executionPlacement),
    "",
    DECISION_COVERAGE_PLACEHOLDER,
    "",
    "## Static Structural Triage (HEURISTIC)",
    analyzerTriageMarkdown(analysis),
    "",
    "## Artifact And Target",
    markdownTable(["Field", "Value"], [
      ["Filename", code(modelName)],
      ["Format", code(String(analysis.format || identity.format || "unknown").toUpperCase())],
      ["Artifact bytes", hasOwn(analysis, "file_size") ? `${formatBytes(analysis.file_size)} (${formatNumber(analysis.file_size)} B)` : "not emitted"],
      ["SHA-256", code(analysis.model_sha256 || identity.sha256 || "pending-browser-export")],
      ["Target", identity.target_label],
      ["Operators / tensors", `${formatNumber(analysis.operator_count ?? identity.operator_count ?? (analysis.ops || []).length)} / ${formatNumber(analysis.tensor_count ?? identity.tensor_count ?? (analysis.tensors || []).length)}`],
      [onnx ? "Assessed MAC total" : "MACs", onnx
        ? `${analysis.mac_assessment?.total_assessed_macs_decimal ?? formatNumber(analysis.mac_assessment?.total_assessed_macs ?? analysis.total_macs ?? identity.total_macs ?? 0)}; numeric mirror ${analysis.mac_assessment?.total_assessed_macs == null ? "withheld outside safe integer range" : formatNumber(analysis.mac_assessment.total_assessed_macs)}; coverage ${analysis.mac_assessment?.assessed_compute_ops ?? 0}/${analysis.mac_assessment?.compute_ops ?? 0} compute op(s); status ${analysis.mac_assessment?.status || "not_assessed"}`
        : batchOneProjection.status === "assumption_bound_batch_one"
          ? `${formatNumber(batchOneProjection.projected_total_macs)} at serialized N=1 (ASSUMPTION_BOUND; exact polynomial evaluation)`
          : batchOneProjection.status === "requires_explicit_shape_binding"
            ? "shape-dependent; bind non-batch dynamic axes"
            : formatNumber(analysis.total_macs ?? identity.total_macs ?? 0)],
      ...(onnx ? [["MAC assessment exclusions", `${formatNumber(analysis.mac_assessment?.not_assessed_compute_ops || 0)} compute op(s) not assessed; denominator contains assessed MACs only`]] : []),
      ...(onnx ? [["Algorithm-dependent arithmetic", `${formatNumber(analysis.mac_assessment?.algorithm_dependent_arithmetic_ops || 0)} transform op(s) excluded from the nominal tensor-contraction MAC denominator because the serialized operator does not select an implementation algorithm`]] : []),
      ...(onnx ? [["Assessed arithmetic operations", `${analysis.mac_assessment?.total_assessed_ops_decimal ?? formatNumber(analysis.mac_assessment?.total_assessed_ops ?? analysis.total_ops ?? 0)}; numeric mirror ${analysis.mac_assessment?.total_assessed_ops == null ? "withheld outside safe integer range" : formatNumber(analysis.mac_assessment.total_assessed_ops)}; ${analysis.mac_assessment?.safe_number_mirror_status || "mirror status not emitted"}`]] : []),
      ["Target profile version", `${analysis.target_profile?.id || "unknown"} @ rulepack ${ANALYZER_METADATA.rulepackVersion}`],
      ["Profile source", `Rulepack planning assumptions for ${analysis.target_profile?.label || analysis.target_profile?.id || "selected target"}; illustrative profile, not a measured device specification`],
      ["Target architecture", targetValue(analysis, analysis.target_profile?.architecture || "not emitted")],
      ["SIMD width", targetValue(analysis, analysis.target_profile?.simd_width_bits ? `${formatNumber(analysis.target_profile.simd_width_bits)} bits` : "not emitted")],
      ["SIMD register-element capacity", targetValue(analysis, `8-bit integer x${formatNumber(analysis.target_profile?.int8_lanes || 0)} / FP16 x${formatNumber(analysis.target_profile?.fp16_lanes || 0)} / FP32 x${formatNumber(analysis.target_profile?.fp32_lanes || 0)}; lane count is not an executed-kernel or native-arithmetic claim`)],
      ["ISA feature assumptions", targetValue(analysis, `dot-product ${analysis.target_profile?.dot_product ? "enabled" : "disabled"}; SVE2 ${analysis.target_profile?.sve2 ? "enabled" : "disabled"}`)],
      ["Channel-alignment planning multiple", targetValue(analysis, analysis.target_profile?.channel_alignment_multiple ? `x${formatNumber(analysis.target_profile.channel_alignment_multiple)}; 8-bit integer/profile alignment assumption, not an executed FP32 kernel tile` : "not emitted")],
      ...targetHardwareSpecRows(analysis.target_profile),
      ["Profile source basis", TARGET_PROFILE_SOURCE_BASIS],
      ["Measured target identity", "not provided; replace with device model, SoC, active core, thread count, governor, runtime build, and measured profile for product evaluation"],
      ["Profile confidence", `${analysis.target_profile?.hardware_spec?.evidence_class || "HEURISTIC_PROFILE"} hardware specification; ${analysis.target_profile?.performance_model_evidence_class || "HEURISTIC"} performance model; not device-calibrated`],
      ["Profile use", onnx ? "Static L1/L2 row working-set ratio references only; not an ONNX Runtime execution-provider, packing, bandwidth, throughput, or ridge model" : "Static prioritization only; not a measured performance profile"],
      ["Performance-model boundary", analysis.target_profile?.performance_model_assumption || "not emitted"],
      ["Kernel-utilization factor", targetValue(analysis, `${Number(analysis.target_profile?.compute_utilization_factor ?? 1).toFixed(2)} of the configured effective-peak planning constant; compute_us = ops / (effective_peak_gops x utilization_factor). This remains optimistic until calibrated against the bound runtime/device.`)],
      ["Partition setup range", targetValue(analysis, `${Number(analysis.target_profile?.chain_break_overhead_us_low || 0).toFixed(1)}-${Number(analysis.target_profile?.chain_break_overhead_us_high || 0).toFixed(1)} us per predicted break; midpoint enters the heuristic cost ledger independently of logical edge payload`)],
      ["Shared-cache concurrency boundary", targetValue(analysis, /shared/i.test(analysis.target_profile?.cache_assumption || "") ? "L2 is shared; no fabricated per-thread capacity is emitted because active threads, co-runners, and occupancy are unbound" : "L2 profile is not declared shared; runtime occupancy remains unobserved")],
      ...(String(analysis.target_profile?.id || "").includes("zynq") ? [["Zynq execution scope", "Cortex-A53 APU CPU only; PL, DPU/NPU, DMA overlap, and accelerator memory paths are not modeled"]] : []),
      ["Assumed 8-bit:FP32 peak throughput ratio", targetValue(analysis, analysis.target_profile?.fp32_compute_factor ? `${analysis.target_profile.fp32_compute_factor}:1` : "-")],
      ["Reduced-precision formula factor", targetValue(analysis, analysis.target_profile?.int8_speedup_estimate ? `${Math.min(Number(analysis.target_profile.int8_speedup_estimate || 1), Number(analysis.target_profile.fp32_compute_factor || analysis.target_profile.int8_speedup_estimate || 1)).toFixed(2)}x 8-bit vs FP32 effective factor used in the MAC-weighted compute-kernel ceiling; separate from the configured peak-throughput ratio` : "-")],
      ["Assumed 8-bit peak compute", targetValue(analysis, analysis.target_profile?.effective_peak_gops ? `${analysis.target_profile.effective_peak_gops} GOPS (effective)` : "-")],
      ["Assumed FP32 peak compute", targetValue(analysis, analysis.target_profile?.effective_peak_gops && analysis.target_profile?.fp32_compute_factor
        ? `${Math.round(analysis.target_profile.effective_peak_gops / analysis.target_profile.fp32_compute_factor)} GOPS (effective; 8-bit peak divided by FP32 factor ${analysis.target_profile.fp32_compute_factor})` : "-")],
      ["Assumed memory bandwidth", targetValue(analysis, analysis.target_profile?.effective_memory_bandwidth_gbps ? `${analysis.target_profile.effective_memory_bandwidth_gbps} GB/s (effective)` : "-")],
      ["Assumed packing bandwidth", targetValue(analysis, analysis.target_profile?.weight_packing_bandwidth_gbps ? `${analysis.target_profile.weight_packing_bandwidth_gbps} GB/s (rulepack heuristic, decimal GB/s, not device-calibrated)` : "-")],
      ["Fixed packing setup overhead", targetValue(analysis, `${packingSetupUs(analysis.target_profile).toFixed(1)} us, rulepack heuristic, not device-calibrated (${analysis.target_profile?.in_order ? "in-order target profile" : "out-of-order/browser target profile"})`)],
      ["Packing watch threshold", targetValue(analysis, "10 us single per-op warning threshold; sub-threshold rows remain cold-start components, not findings")],
      ["Packing evidence", targetValue(analysis, "ESTIMATED", "suppressed until ONNX Runtime EP behavior is modeled")],
      ["Output-channel kernel tile candidates", targetValue(analysis, simdAssumptionsForAnalysis(analysis))],
      ["XNNPACK build requirements", targetValue(analysis, xnnpackBuildRequirementsSummary(analysis))],
      ["XNNPACK profile kernel family", targetValue(analysis, analysis.target_profile?.xnnpack_kernel_family || "not emitted")],
      ["Predicted partition-break overhead assumption", targetValue(analysis, analysis.target_profile?.chain_break_overhead_us_low != null && analysis.target_profile?.chain_break_overhead_us_high != null
        ? `${formatUs(analysis.target_profile.chain_break_overhead_us_low)} to ${formatUs(analysis.target_profile.chain_break_overhead_us_high)} per modeled break`
        : "not emitted")],
      ["Runtime assumption", runtimeAssumptionText(analysis)],
      ["8-bit theoretical ridge", targetValue(analysis, analysis.target_profile?.effective_peak_gops && analysis.target_profile?.effective_memory_bandwidth_gbps
        ? `~${formatRidge(Number(analysis.target_profile.effective_peak_gops) / Number(analysis.target_profile.effective_memory_bandwidth_gbps))} ops/B` : "-")],
      ["FP32 theoretical ridge", targetValue(analysis, analysis.target_profile?.effective_peak_gops && analysis.target_profile?.fp32_compute_factor && analysis.target_profile?.effective_memory_bandwidth_gbps
        ? `~${formatRidge((Number(analysis.target_profile.effective_peak_gops) / Number(analysis.target_profile.fp32_compute_factor)) / Number(analysis.target_profile.effective_memory_bandwidth_gbps))} ops/B` : "-")],
      ["Posture classes note", onnx ? "low/mixed/high-intensity labels use fixed ONNX static intensity bands in this build; target profile is used for L1 ratio only" : "low/mixed/high-intensity labels come from heuristic bands configured in the rulepack, not from ridge position — see appendix"],
      ["Static intensity heuristic bands", targetValue(analysis, analysis.target_profile?.memory_bound_intensity != null && analysis.target_profile?.compute_bound_intensity != null
        ? `low-intensity below ${analysis.target_profile.memory_bound_intensity} ops/B; high-intensity at or above ${analysis.target_profile.compute_bound_intensity} ops/B; middle is mixed`
        : "not emitted")],
      ["L1D reference", analysis.target_profile?.l1_data_bytes ? `${formatBytes(analysis.target_profile.l1_data_bytes)} selected-profile denominator; near-capacity watch begins at 0.90x` : "-"],
      ["L2 reference", analysis.target_profile?.l2_bytes ? `${formatBytes(analysis.target_profile.l2_bytes)} selected-profile denominator; near-capacity watch begins at 0.90x` : "not emitted"],
      ["Cache assumption", analysis.target_profile?.cache_assumption || "not emitted"],
      ["Cache source URL", analysis.target_profile?.cache_source_url || "host/device-dependent planning default; no single hardware source applies"],
      ["Analyzer", `${ANALYZER_METADATA.name} ${ANALYZER_METADATA.version}`],
      ["DeepBOM semantic version", ANALYZER_METADATA.semanticVersion],
      ["Analyzer execution identity", analyzerContentVersion(ANALYZER_METADATA.semanticVersion, ANALYZER_METADATA.buildCommit, ANALYZER_METADATA.buildContentSha256)],
      ["Analyzer build commit", ANALYZER_METADATA.buildCommit ? `${ANALYZER_METADATA.buildCommit}${ANALYZER_METADATA.buildSourceState && ANALYZER_METADATA.buildSourceState !== "clean" ? ` (${ANALYZER_METADATA.buildSourceState})` : ""}` : "not embedded in this browser bundle"],
      ["Analyzer bundle content SHA-256", ANALYZER_METADATA.buildContentSha256 || "not embedded in this browser bundle"],
      ["Analyzer bundle content hash method", ANALYZER_METADATA.buildContentHashMethod || "not embedded"],
      ["Analyzer bundle hash basis", (ANALYZER_METADATA.buildContentHashBasis || []).join("; ") || "not embedded"],
      ["Analyzer build-content manifest SHA-256", ANALYZER_METADATA.buildContentManifestSha256 || "not embedded"],
      ["Analyzer release provenance policy", ANALYZER_METADATA.buildSourceState === "clean" ? "release-compatible clean tree" : "development/dirty bundle; commit alone is not reproducible, use bundle content SHA-256 and do not treat as clean release provenance"],
      ["Rulepack", ANALYZER_METADATA.rulepackVersion],
      ["Rulepack SHA-256", ANALYZER_METADATA.rulepackSha256 || "not embedded; use rulepack version and pinned provenance above until the build emits a hash"],
      ["Rulepack hash basis", (ANALYZER_METADATA.rulepackHashBasis || []).join("; ") || "not embedded"],
      ["Target-profile SHA-256", analysis.target_profile?.profile_sha256 || "not embedded; profile id/version is reported but profile hash is not bundled"],
      ["Registered report-body SHA-256", verification?.reportHash || "not registered; this field is populated only when fingerprint registration is requested"],
      ["Package digest attestation", "Issued after all unsigned ZIP members are assembled; see attestation.json for the authoritative canonical package-member digest-set SHA-256. It is intentionally not embedded in this report body to avoid self-reference."],
      ["Delegation rule basis", delegationRuleBasisText(analysis)],
      ["XNNPACK delegate README SHA-256", ANALYZER_METADATA.rulepackProvenance.xnnpackReadmeSha256],
      ["XNNPACK semantic rule manifest", `${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateRuleManifestSchema}; SHA-256 ${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateRuleManifestSha256}`],
      ["XNNPACK documentary constraint coverage", `${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateImplementedConstraintCount}/${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateDocumentedConstraintCount} artifact-visible constraints mapped; ${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateUnmappedConstraintCount} unmapped; ${ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateRuntimeOnlyRequirementCount} rulepack runtime-only requirement(s), ${new Set((analysis.ops || []).filter((op) => op.xnnpack_supported).map((op) => op.xnnpack_build_requirement).filter(Boolean)).size} artifact-applicable`],
      ["XNNPACK source conformance", ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateSourceConformanceStatus],
      ["Report schema", ANALYZER_METADATA.schemas.engineeringReport],
      ["Generated", generatedAt],
    ]),
    "",
    serializedFormatEvidenceMarkdown(analysis),
    "",
    onDeviceLlmEvidenceMarkdown(analysis),
    "",
    deploymentFrontierMarkdown(analysis),
    "",
    deploymentDeltaMarkdown(analysis),
    "",
    delegationRepairMarkdown(analysis),
    "",
    "## Pinned XNNPACK Kernel Candidates",
    kernelSourceCandidatesMarkdown(analysis),
    "",
    "## Input/Output Contract",
    markdownTable(["Field", "Value"], onnx ? [
      ["Graph input names", (analysis.inputs || []).map((t) => t.name).filter(Boolean).join(", ") || "none parsed"],
      ["Graph input tensor indices", (analysis.input_tensor_indices || []).map((index) => formatNumber(index)).join(", ") || "none parsed"],
      ["Graph output tensor indices", (analysis.output_tensor_indices || []).map((index) => formatNumber(index)).join(", ") || "none parsed"],
      ["Main graphs", "1"],
      ["Nested subgraphs", `${Math.max(0, Number(analysis.subgraphs || 1) - 1)}`],
      ["Input layout determination", inputLayoutDetermination(analysis)],
      ["Input tensor numerical contract", inputNumericalContractSummary(analysis)],
      ["Source-data-to-tensor preprocessing", sourcePreprocessingContractSummary(analysis)],
      ["Input contract derived risks", inputContractRiskSummary(analysis)],
    ] : [
      ["Signature key(s)", (analysis.metadata_presence?.signature_keys || []).join(", ") || "none embedded"],
      ["Graph input tensor indices", (analysis.input_tensor_indices || []).map((index) => formatNumber(index)).join(", ") || "none parsed"],
      ["Graph output tensor indices", (analysis.output_tensor_indices || []).map((index) => formatNumber(index)).join(", ") || "none parsed"],
      ["Subgraphs", `${analysis.subgraphs || 1}`],
      ["Stateful (variable) tensors", `${(analysis.tensors || []).filter((t) => t.is_variable).length}`],
      ["Input layout determination", inputLayoutDetermination(analysis)],
      ["Input tensor numerical contract", inputNumericalContractSummary(analysis)],
      ["Source-data-to-tensor preprocessing", sourcePreprocessingContractSummary(analysis)],
      ["Input contract derived risks", inputContractRiskSummary(analysis)],
    ]),
    "",
    markdownTable(["Tensor", "Name", "Shape", "Dtype", "Dynamic dims", "Quantization"],
      [...(analysis.inputs || []).map((t, i) => [`Input ${i}`, code(t.name || "-"), (t.shape || []).join("×") || "-", t.dtype || "-",
        Array.isArray(t.shape_signature) && t.shape_signature.some((d) => Number(d) < 0) ? "yes" : "none",
        tensorQuantizationText(t)]),
       ...(analysis.outputs || []).map((t, i) => [`Output ${i}`, code(t.name || "-"), (t.shape || []).join("×") || "-", t.dtype || "-",
        Array.isArray(t.shape_signature) && t.shape_signature.some((d) => Number(d) < 0) ? "yes" : "none",
        tensorQuantizationText(t)])]),
    "",
    "### Input Tensor Contract Evidence",
    inputContractEvidenceMarkdown(analysis),
    "",
    "### Interface Quantization Contract Ledger",
    interfaceQuantizationLedgerMarkdown(analysis, productionInterfaceContract),
    "",
    dynamicShapeCostMarkdown(analysis),
    "",
    "## Analysis Completeness",
    (() => {
      const customCount = artifactIntegrity.custom_like_ops.reduce((n, item) => n + Number(item.count || 0), 0);
      const parsedOps = (analysis.ops || []).length;
      const subgraphInventory = analysis.tflite_subgraph_inventory || {};
      const deepSubgraphs = analysis.tflite_subgraph_deep_analysis || {};
      const allSubgraphsParsed = Number(subgraphInventory.parsed_subgraph_count || 0) === Number(analysis.subgraphs || 1);
      const full = customCount === 0 && allSubgraphsParsed;
      if (onnx) {
        const external = analysis.onnx_external_data || {};
        const structureBinding = analysis.onnx_external_data_structure_binding || null;
        const runtimeAssignment = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || null;
        const ortCompatibility = analysis.ort_compatibility_evidence || null;
        return markdownTable(["Analysis area", "Status"], [
          ["Graph topology", "Complete for parsed main graph"],
          ["Operator parsing", "Complete for parsed main graph"],
          ["Shape inference", analysis.onnx_shape_inference
            ? `${analysis.onnx_shape_inference.status || "partial"}; ${formatNumber(analysis.onnx_shape_inference.known_node_output_count || 0)}/${formatNumber(analysis.onnx_shape_inference.node_output_count || 0)} node output(s) known; ${formatNumber(analysis.onnx_shape_inference.rule_unsupported_nodes || 0)} unsupported-rule node(s); OpSchema ${analysis.onnx_shape_inference.schema_form_assessment_status || "not emitted"}; ${formatNumber(analysis.onnx_shape_inference.shape_scope?.unassessed_reachable_node_count || 0)} reachable extended-scope node(s) unassessed`
            : "Not assessed; no shape-inference ledger emitted"],
          ["Dynamic shape cost", analysis.dynamic_shape_cost_contract?.status === "not_applicable_static_shapes"
            ? "not applicable; all assessed tensor shapes are static"
            : `${analysis.dynamic_shape_cost_contract?.status || "not assessed"}; ${analysis.dynamic_shape_cost_contract?.total_macs_formula_status || "total MAC formula not emitted"}; ${analysis.dynamic_shape_cost_contract?.arena_projection_status || "arena status not emitted"}`],
          ["Initializer inventory", structureBinding
            ? `${formatNumber(analysis.size_breakdown?.constant_tensor_count || 0)} declaration(s); ${formatNumber(structureBinding.tensor_count)} external range(s) location/offset/length/file-SHA bound; numerical payload decode not assessed by structure policy`
            : `${formatNumber(analysis.size_breakdown?.constant_tensor_count || 0)} declaration(s); ${formatNumber(analysis.size_breakdown?.embedded_constant_tensor_count || 0)} embedded, ${formatNumber(external.verified_payload_count || 0)}/${formatNumber(external.tensor_count || 0)} external payload range(s) path/range/cardinality/hash verified; ${external.status || "external-data status not emitted"}`],
          ...onnxWeightIntegrityCompletenessRows(analysis.weight_integrity),
          ["Execution-provider assignment", runtimeAssignment
            ? `${runtimeAssignment.assignment_evidence_class || "OBSERVED_RUNTIME"}; ${formatNumber(runtimeAssignment.mapped_op_count || 0)}/${formatNumber((analysis.ops || []).length)} original op(s) mapped; ${formatNumber(runtimeAssignment.unresolved_op_count || 0)} unresolved`
            : "NOT_ASSESSABLE; no artifact-bound ORT runtime profile was imported"],
          ["ORT source compatibility", ortCompatibility
            ? `${analysis.ort_compatibility_assessment_status || "complete"}; runtime floor ${ortCompatibility.runtime_floor?.status || "not assessed"}; artifact-visible definite exclusions narrow source candidates but remaining candidates do not prove GetCapability eligibility or placement`
            : "NOT_ASSESSED; protected pinned-source ORT rulepack was not loaded"],
          ["Packing estimate", "Suppressed"],
          ["Sections suppressed", sectionsSuppressedText(analysis)],
        ]);
      }
      return markdownTable(["Field", "Value"], [
        ["Static parser coverage", full ? "Complete for serialized structure and independent deep analysis across every subgraph" : `Partial static parser coverage - ${customCount ? `${customCount} custom/opaque primary op(s); ` : ""}${allSubgraphsParsed ? "" : `${subgraphInventory.parsed_subgraph_count || 0}/${analysis.subgraphs || 1} subgraphs parsed`}`.replace(/; $/, "")],
        ["Scope qualifier", "Completeness here means parsed static artifact fields only; it is not runtime, task-level, or clinical completeness."],
        ["Parsed subgraphs", `${subgraphInventory.parsed_subgraph_count || 0}/${analysis.subgraphs || 1} serialized structure; ${deepSubgraphs.assessed_subgraph_count || 0}/${analysis.subgraphs || 1} independently deep-assessed; no cross-control-flow execution total without invocation counts`],
        ["Parsed operators", `${parsedOps} primary / ${subgraphInventory.serialized_operator_count ?? parsedOps} serialized across all subgraphs`],
        ["Operator type histogram", compactCountItems(analysis.histogram || [])],
        ["Tensor dtype inventory", compactCountItems(analysis.tensor_types || [])],
        ["Custom/opaque operators", customCount ? artifactIntegrity.custom_like_ops.map((item) => `${item.name}:${item.count}`).join(" / ") : "0"],
        ["Unresolved external data", artifactIntegrity.external_data_summary?.startsWith("none") || artifactIntegrity.external_data_summary === "-" ? "0" : artifactIntegrity.external_data_summary],
        ["Dynamic shape limitations", analysis.dynamic_shape_cost_contract?.status === "not_applicable_static_shapes"
          ? "none"
          : `${analysis.dynamic_shape_cost_contract?.status || "not assessed"}; ${analysis.dynamic_shape_cost_contract?.total_macs_formula_status || "total MAC formula not emitted"}; ${analysis.dynamic_shape_cost_contract?.arena_projection_status || "arena status not emitted"}`],
        ["Sections suppressed", sectionsSuppressedText(analysis)],
      ]);
    })(),
    "",
    METRIC_COVERAGE_PLACEHOLDER,
    "",
    advancedStaticProofCoverageMarkdown(analysis),
    "",
    computedAnalysisCoverageMarkdown(analysis),
    "",
    stageAndPatternMarkdown(analysis),
    "",
    onnxShapeInferenceMarkdown(analysis),
    "",
    onnxTensorDataTypeContractMarkdown(analysis),
    "",
    onnxTypeProtoContractMarkdown(analysis),
    "",
    onnxSparseTensorContractMarkdown(analysis),
    "",
    tfliteSubgraphInventoryMarkdown(analysis),
    "",
    tfliteSparseStorageMarkdown(analysis),
    "",
    artifactSizeBreakdownMarkdown(analysis),
    artifactByteIntegrityMarkdown(analysis),
    "",
    peakLiveActivationMarkdown(analysis),
    "",
    tensorArenaPlanMarkdown(analysis),
    "",
    weightIntegrityMarkdown(analysis, weightIntegrity),
    "",
    quantizationContractChecksMarkdown(analysis),
    "",
    quantResearchEvidenceMarkdown(analysis),
    "",
    runtimeRequirementsMarkdown(analysis, bundledRuntimeVersion, runtimeConclusion),
    "",
    artifactMetadataMarkdown(analysis),
    "",
    checkpointLineageMarkdown(analysis, quant),
    "",
    (() => {
      const changeAnalysis = buildChangeAnalysis(analysis, { priorSnapshot, identity });
      if (changeAnalysis.status !== "assessed") return `## Change vs Prior Audit\nChange analysis was not performed (${changeAnalysis.reason_code}).\n\n${changeAnalysis.reason}\n`;
      const fmtNum = (n) => new Intl.NumberFormat("en-US").format(Math.round(n || 0));
      const lines = ["## Change vs Prior Audit (DERIVED)",
        `Compared against a locally stored snapshot audited ${priorSnapshot.updatedAt || priorSnapshot.createdAt || "previously"} (${priorSnapshot.analyzerVersion}/${priorSnapshot.rulepackVersion}).`,
        `Comparison basis: \`${changeAnalysis.comparison_basis}\`.`,
        "",
        markdownTable(["Metric", "Prior", "Now"], [
          ["SHA-256", `\`${(priorSnapshot.sha256 || "").slice(0, 12)}…\``, `\`${(analysis.model_sha256 || "").slice(0, 12)}…\``],
          ["Operators", `${priorSnapshot.operatorCount}`, `${analysis.operator_count}`],
          ["Quantized compute MACs", assessedPercent(priorSnapshot.quant?.quantComputeMacPercent), assessedPercent(analysis.quantization_status?.quantized_compute_mac_percent)],
          ["Predicted non-structural partition breaks", `${priorSnapshot.delegation?.effectiveChainBreaks ?? "-"}`, `${analysis.xnnpack_effective_chain_breaks || 0}`],
          ["Total MACs", fmtNum(priorSnapshot.totalMacs), fmtNum(analysis.total_macs)],
        ]),
        "> DERIVED local comparison. Treat it as a change-control record only when the declared lineage basis is valid for the release process."];
      return lines.join("\n");
    })(),
    "",
    "## Artifact Integrity Posture",
    markdownTable(["Item", "Value"], onnx ? [
      ["IR / opset", artifactIntegrity.schema_or_opset],
      ["Graph / producer", [artifactIntegrity.graph_name, artifactIntegrity.onnx_producer].filter(Boolean).join(" / ") || "-"],
      ["Custom-like op families", artifactIntegrity.custom_like_ops.length ? artifactIntegrity.custom_like_ops.map((item) => `${item.name}:${item.count}`).join(" / ") : "none detected"],
      ["External ONNX domains", artifactIntegrity.external_domains.length ? artifactIntegrity.external_domains.join(" / ") : "none detected"],
      ["ONNX external data", artifactIntegrity.external_data_summary],
      ["Integrity note", artifactIntegrity.summary],
    ] : [
      ["Schema", artifactIntegrity.schema_or_opset],
      ["Custom-like op families", artifactIntegrity.custom_like_ops.length ? artifactIntegrity.custom_like_ops.map((item) => `${item.name}:${item.count}`).join(" / ") : "none detected"],
      ["Tensor buffers", artifactIntegrity.tflite_buffer_summary],
      ["Integrity note", artifactIntegrity.summary],
    ]),
    "",
    "## Limitations",
    findings.filter((f) => f.category === "limitation").length
      ? findings.filter((f) => f.category === "limitation").map((f) => `- **${f.title}** — ${f.recommendation}`).join("\n")
      : "- None recorded beyond the standing evidence boundary above.",
    "",
    "## Quantization Debug View",
    markdownTable(["Metric", "Value"], [
      ["Status", quant.label || quant.classification || "unknown"],
      ["Detail", quant.detail || quant.summary || "-"],
      ["Analyzer summary", quant.summary || "not emitted"],
      ["Scope", quantizationScopeSummary(analysis)],
      ["Quantized tensors", `${formatNumber(analysis.quantized_tensors || 0)} / ${formatNumber(onnx ? quant.dense_tensor_count || 0 : analysis.tensor_count || 0)}${onnx ? ` dense tensor(s); ${formatNumber(quant.non_dense_value_count || 0)} non-dense value(s) excluded` : ""}`],
      ["Quantized tensor coverage", formatPercent(quant.quantized_tensor_percent ?? 0)],
      ["Per-channel tensors", formatNumber(analysis.per_channel_tensors || 0)],
      ["Quantized compute MACs", assessedPercent(quant.quantized_compute_mac_percent)],
      ["Quantized compute assessment", `${formatNumber(quant.quantized_compute_ops || 0)}/${formatNumber(quant.compute_ops || 0)} compute op(s); MAC-assessed ${formatNumber(quant.mac_assessed_compute_ops ?? quant.compute_ops ?? 0)}; coverage ${quant.mac_coverage_complete == null ? "not emitted" : quant.mac_coverage_complete ? "complete" : "partial"}`],
      ["Tensor dtype counts", `INT8 ${formatNumber(quant.int8_tensors || 0)} / UINT8 ${formatNumber(quant.uint8_tensors || 0)} / FLOAT16 ${formatNumber(quant.float16_tensors || 0)} / FLOAT total ${formatNumber(quant.float_tensors || 0)}; input ${(quant.input_dtypes || []).join(" / ") || "-"}; output ${(quant.output_dtypes || []).join(" / ") || "-"}`],
      ["Integer-path contract", `${quant.full_integer ? "full integer" : "not full integer"}; serialized Q/DQ ${formatNumber(quant.quantize_ops || 0)} / ${formatNumber(quant.dequantize_ops || 0)}; activation Q/DQ ${formatNumber(quant.activation_quantize_ops || 0)} / ${formatNumber(quant.activation_dequantize_ops || 0)}; 8-bit/float ${formatNumber(quant.activation_8bit_float_boundary_ops || 0)}; integer requantization ${formatNumber(quant.integer_requantization_ops || 0)}`],
      ["Constant precision conversions", `${formatNumber(quant.constant_precision_conversion_ops || 0)} total; FP16-to-FP32 ${formatNumber(quant.float16_constant_expansion_ops || 0)}`],
      [onnx ? "Reduced-precision estimate" : ["not_quantized_float", "float16_weight_storage"].includes(quant.classification) ? "Reduced-precision opportunity" : "Quantized-path planning estimate", onnx
        ? "NOT_ASSESSABLE; no ONNX quantization opportunity model was executed"
        : ["not_quantized_float", "float16_weight_storage"].includes(quant.classification)
          ? `MAC-weighted 8-bit integer compute-kernel throughput ceiling: approximately ${Number(analysis.estimated_int8_speedup || 1).toFixed(1)}x under the configured heuristic profile; this is not an end-to-end model speedup estimate and excludes zero-MAC operator time, runtime overhead, packing, Q/DQ boundaries, unsupported kernels, memory stalls, and task-level regression constraints`
          : `Estimated quantized-path compute-kernel ceiling: approximately ${Number(analysis.estimated_int8_speedup || 1).toFixed(1)}x versus the configured FP32 planning baseline on ${identity.target_label || "selected target"}; this describes the already-quantized path, not additional optimization opportunity`],
      ["Reduced-precision method", onnx ? "No ONNX quantization opportunity model was executed." : String(analysis.estimated_int8_speedup_detail || "No model-level 8-bit integer opportunity method was emitted by the analyzer.").replaceAll("INT8", "8-bit integer")],
    ]),
    "",
    "## Architecture Block Inventory",
    blockInventoryMarkdown(analysis),
    "",
    "## Compute Hotspots",
    onnx
      ? markdownTable(["Op", "Name", "MACs", "MAC share", "Intensity posture", "Quant"], topOps.map((op) => [
        `#${padOp(op.index)}`,
        op.name,
        formatExactInteger(op.macs_decimal, op.macs, "N/A"),
        op.mac_percent == null ? "N/A" : formatPercent(op.mac_percent),
        intensityPosture(op.static_bound_guess),
        quantStateLabel(op),
      ]))
      : markdownTable(["Op", "Name", "MACs", "MAC share", "Intensity posture", "XNNPACK", "Quant"], topOps.map((op) => [
        `#${padOp(op.index)}`,
        op.name,
        formatNumber(op.macs || 0),
        formatPercent(op.mac_percent || 0),
        intensityPosture(op.static_bound_guess),
        xnnpackLabel(op),
        quantStateLabel(op),
      ])),
    "",
    "## Memory And Cache Hotspots",
    markdownTable(["Op", "Name", "Input W x C", "Output W x C", "Kernel", "Storage", "Input strip", "Output row", "Logical row payload", "Serialized kernel", "Serialized bias", `L1 ratio (${formatBytes(analysis.target_profile?.l1_data_bytes || 0)})`, `L2 ratio (${formatBytes(analysis.target_profile?.l2_bytes || 0)})`, "Evidence"], cacheOps.map((op) => [
      `#${padOp(op.index)}`,
      op.name,
      `${formatNumber(op.cache_payload.input_width)} x ${formatNumber(op.cache_payload.input_channels)}`,
      `${formatNumber(op.cache_payload.output_width)} x ${formatNumber(op.cache_payload.output_channels)}`,
      `${formatNumber(op.cache_payload.kernel_height)}x${formatNumber(op.cache_payload.kernel_width)}; effective H ${formatNumber(op.cache_payload.effective_kernel_height)}`,
      `${op.cache_payload.input_dtype || "?"} -> ${op.cache_payload.output_dtype || "?"}`,
      formatBytes(op.cache_payload.input_strip_bytes),
      formatBytes(op.cache_payload.output_row_bytes),
      formatBytes(op.cache_payload.logical_row_payload_bytes),
      op.cache_payload.serialized_kernel_bytes == null ? "N/A" : formatBytes(op.cache_payload.serialized_kernel_bytes),
      op.cache_payload.serialized_bias_bytes == null ? "N/A" : formatBytes(op.cache_payload.serialized_bias_bytes),
      cacheRatioText(op.cache_payload.logical_row_payload_bytes, analysis.target_profile?.l1_data_bytes),
      cacheRatioText(op.cache_payload.logical_row_payload_bytes, analysis.target_profile?.l2_bytes),
      op.cache_payload.evidence_class || "NOT_ASSESSABLE",
    ])),
    cacheOps.length
      ? `> Cache payload schema \`${cacheOps[0].cache_payload.schema || "not emitted"}\`; method ${cacheOps[0].cache_payload.method || "not emitted"}. ${cacheOps[0].cache_payload.interpretation_boundary || ""}`
      : "",
    cacheOps.length
      ? onnx
        ? "> Shared action note: The table reports graph-semantic NCHW logical row payloads for supported ONNX convolution signatures. It is not an ORT kernel, cache-residency, hit-rate, or DRAM-traffic measurement."
        : Number(quant.quantized_compute_mac_percent || 0) >= 0.95
          ? "> Shared action note: The artifact already uses an 8-bit compute path. Logical row payload is input strip plus one output row; serialized kernel and bias are listed separately and are not assumed simultaneously cache-resident."
          : "> Shared action note: Logical row payload is input strip plus one output row. Reduced precision changes storage width only after conversion and representative-output validation; no cache residency is asserted."
      : "> No operator has a complete supported logical-row-payload assessment; the section is empty rather than substituting zero.",
    "",
    coreIsolationRooflineMarkdown(analysis, runtimeEvidence),
    "",
    movementAndPackingMarkdown(analysis),
    "",
    delegationPredictionMarkdown(analysis, delegationSegments),
    "",
    "## Runtime Environment And Reproducibility",
    runtimeEnvironmentMarkdown(runtimeEvidence, runtimeBenchmarkResults, analysis),
    "",
    "## Runtime Benchmark Evidence",
    benchmarkRows.length
      ? markdownTable(["Backend", "Status", "Compile", "First", "p50", "p90", "p95", "p99", "Mean", "Steady p50", "Jitter", "Notes"], benchmarkRows)
      : "No Runtime Benchmark table has been run in this browser session. Use the Runtime tab for p50/p90/p95/p99 latency evidence.",
    benchmarkSampleLedgerMarkdown(runtimeBenchmarkResults),
    "",
    ["full_integer", "mixed_integer"].includes((analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis)).artifact_class)
      ? preprocessingConsequenceMarkdown(preprocessingConsequence, "##")
      : "",
    "",
    calibrationValidationMarkdown(calibrationValidation, "##"),
    "",
    "## Research Module Evidence",
    markdownTable(["Module", "Status", "Key point"], [
      ["Research DEEPBOM", deepBomResult?.status || (deepBomResult ? "complete" : "not run"), deepBomResult ? (deepBomResult.score_assessment?.status === "ASSESSED" ? `basin=${score100(deepBomResult.basin_proxy_score)} topology=${score100(deepBomResult.topology_stress_score)}` : `NOT_ASSESSABLE: ${deepBomResult.score_assessment?.reason || "required score inputs unavailable"}`) : "No deploy-artifact weight/topology proxy result."],
      ["Research Perturbation", perturbationResult?.status || "not run", perturbationResult?.drift ? `max drift=${formatDrift(perturbationResult.drift.max_abs)} ${perturbationResult.output_profile?.unit || ""}` : "No perturbation result."],
      [RUNTIME_COMPATIBILITY_EVIDENCE_LABEL, runtimeBasinResult?.status || "not run", runtimeBasinResult?.max_drift ? `max backend drift=${formatDrift(runtimeBasinResult.max_drift.max_abs)} ${runtimeBasinResult.output_profile?.unit || ""}` : "No backend drift result."],
      ["Research Deployment Sensitivity Proxy", deployCurvatureResult?.status || "not run", deployCurvatureResult?.basin ? `deploy basin=${Number(deployCurvatureResult.basin.score || 0).toFixed(1)} / 100` : "No deployment-sensitivity result."],
    ]),
    "",
    "## Session Privacy Manifest (declared)",
    markdownTable(["Session privacy field", "Value"], [
      ["Model-content processing", "local in browser by the core static-analysis workflow"],
      ["Model bytes uploaded", "no, not transmitted by the core workflow"],
      ["Tensor values uploaded", "no, not transmitted by the core workflow"],
      ["Full report synced to a server", "no report sync is enabled by the core workflow"],
      ["Account and access services", "network service may be used for sign-in, capability authorization, report-fingerprint registration, or hosted app delivery; this is separate from model-content processing"],
      ["Telemetry schema version", ANALYZER_METADATA.schemas.modelStructure],
      ["Structural telemetry consent", sessionPrivacy ? (sessionPrivacy.consent ? "accepted; structural event data may be sent" : "declined/withdrawn") : "unknown (not captured at generation time)"],
      ["Consent timestamp", sessionPrivacy?.consentTimestamp ? "captured in local consent log; exact timestamp redacted from report" : "not captured in this report context"],
      ["Consent event ID", sessionPrivacy?.consentEventId ? "captured in local consent log; identifier redacted from report" : "not captured in this report context"],
      ["Consent text/policy version", sessionPrivacy?.policyVersion || "agreement v3 / research-consent v1 unless overridden by the host service"],
      ["Retention period", sessionPrivacy?.retentionPeriod || "not captured in this report context; governed by host service policy for server-side consent/telemetry records"],
      ["Retention duration", sessionPrivacy?.retentionDuration || sessionPrivacy?.retentionPeriod || "not captured in this report context; production release evidence should state an explicit day count"],
      ["Policy version", sessionPrivacy?.policyVersion || "agreement v3 / research-consent v1 unless overridden by the host service"],
      ["Deletion SLA", sessionPrivacy?.deletionSla || "not captured in this report context"],
      ["Export request method", sessionPrivacy?.exportRequestMethod || "not captured in this report context"],
      ["Deletion or withdrawal mechanism", "local research consent can be withdrawn in the app; browser history can be cleared locally; server-side deletion/export requests depend on the signed-in account and host service policy"],
      ["Telemetry endpoint", sessionPrivacy?.telemetryEndpoint || "/api/benchmark/structure when structure sharing is enabled"],
      ["Structural telemetry current report session transmission", sessionPrivacy?.telemetryCurrentShareAt ? "sent; exact timestamp redacted from report" : "none sent in this report session"],
      ["Structural telemetry most recent prior transmission", sessionPrivacy?.telemetryLastShareAt ? "recorded; exact timestamp redacted from report" : "none recorded in this browser"],
      ["Structural telemetry payload schema", sessionPrivacy?.telemetryPayloadSchema || "deepbom.structure_telemetry.v1.1"],
      ["Structural telemetry payload preview", sessionPrivacy?.telemetryPayloadPreview || "not captured in this report context"],
      ["Structural telemetry field categories", sessionPrivacy?.consent ? "format, target/profile, op histogram, stage structure, bounded op sequence with shapes/dtypes/MACs/bytes/quant/partition-break class, predicted partition-break counts, IO contracts, bound mix, pattern names, browser bucket, and structure fingerprint" : "not sent unless research sharing consent is accepted"],
      ["Structural telemetry architecture warning", "structural metadata can reveal aspects of model architecture; share only if authorized by the model owner or organization"],
      ["Structural telemetry explicit exclusions", "filename, artifact SHA-256, raw model bytes, constant/weight bytes, tensor values, user inputs, model outputs, and full report text"],
      ["Benchmark telemetry field categories", "backend/browser identifiers, compile and latency statistics, run counts, output count/digest, prepared-input flag, and runtime status; sent only when benchmark sharing consent is active"],
      ["Benchmark telemetry explicit exclusions", "ordered raw latency samples, input/output tensor contracts, model bytes, tensor values, and model outputs remain local"],
      ["Report-fingerprint registration payload", verification ? "used: report-body SHA-256, artifact SHA-256, analyzer/rulepack versions, registry authentication tag/code, and signed-in account session; no full report or model bytes" : "not used in this report; if requested, sends report-body SHA-256, artifact SHA-256, analyzer/rulepack versions, and signed-in account session"],
      ["Signed package-attestation payload", "client-submitted model SHA-256; target id, label, and profile SHA-256; canonical package-member names, byte sizes, and SHA-256 digests; canonical digest-set SHA-256; issuance scope/time; and immutable signing-key metadata. No raw model, report body, filename/hash, account identity, or capability matrix is embedded"],
      ["Artifact SHA-256 transmitted", verification ? "yes, for report-fingerprint registration in this report; package attestation also transmits it when requested" : "not by core local analysis or structure telemetry; yes only if report-fingerprint registration or package attestation is requested"],
      ["Filename transmitted", "no raw filename or filename hash in structure telemetry, report-fingerprint registration, or package attestation; package attestation contains only generated ZIP-internal member names"],
      ["Account ID linked to telemetry", "yes when signed in and consented benchmark/structure endpoints are used; anonymous structure records may be stored without a user id"],
      ["Telemetry record linked to account", "yes when signed in and consented benchmark/structure endpoints are used; no for anonymous local-only analysis unless a host service records anonymous telemetry"],
      ["Structural telemetry payload fingerprint", sessionPrivacy?.telemetryFingerprint ? "computed; fingerprint redacted from report" : "none sent this session"],
      ["Browser report history", sessionPrivacy ? (sessionPrivacy.historySaved ? "local IndexedDB in this browser" : "not saved to local history in this session") : "unknown"],
      ["Processing mode", isOnnxAnalysis(analysis) ? "browser-local JavaScript ONNX parsing/static assessment plus ONNX Runtime Web WASM for optional inference" : "browser-local Rust/WASM for TFLite model-content analysis"],
    ]),
    "",
    "> Sharing note: this document embeds the local filename and artifact SHA-256. Core local analysis and structural telemetry do not transmit those identifiers. Report-fingerprint registration sends the report-body hash and artifact SHA-256; package attestation sends the artifact SHA-256, target identity, and generated member digest metadata, but no filename or filename hash. The embedded identifiers travel with this report if it is shared externally.",
    "",
    "## Evidence Class Legend",
    evidenceClassLegend(),
    "",
    "## Raw Static Audit Appendix",
    "",
    analysis._markdown || (isOnnxAnalysis(analysis) ? analysis.markdown : "") || buildStaticAuditMarkdown(analysis, analysis.model_sha256 || "") || "No raw static audit markdown was generated.",
    "",
    verification ? REPORT_VERIFICATION_SENTINEL : "",
    verification ? `## Report Fingerprint Registry\nRegistration code: **${verification.code}**\nCompare this document at ${verification.origin || "https://deepbom.org"}/verify. The page hashes the report body locally and compares it with the registered fingerprint.\n\nA match shows only that the report body matches the value a signed-in user registered. Registration does not prove that DEEPBOM generated or issued the report, and it does not certify analytical correctness, model safety, deployment readiness, regulatory conformity, or clinical performance.${ANALYZER_METADATA.buildSourceState && ANALYZER_METADATA.buildSourceState !== "clean" ? " This report was generated by a development/dirty analyzer build, so the registry record is not release-grade build provenance." : ""}\n\nRegistered report fingerprint (SHA-256, body above this line): \`${verification.reportHash}\`${verification.authenticationTag ? `\nRegistry authentication tag (${verification.authenticationAlgorithm || "HS256"}, server-verifiable only): \`${verification.authenticationTag.slice(0, 32)}…\`` : ""}` : "",
  ];
  const reportAccessedFieldPaths = fieldAccess.accessedFieldPaths();
  const metricCoverage = buildMetricCoverageManifest(sourceAnalysis, {
    ...metricCoverageContext,
    reportAccessedFieldPaths,
  });
  const coverageIndex = reportParts.indexOf(METRIC_COVERAGE_PLACEHOLDER);
  if (coverageIndex < 0) throw new Error("Engineering Report metric coverage placeholder is missing.");
  reportParts[coverageIndex] = metricCoverageMarkdown(metricCoverage);
  const decisionCoverageIndex = reportParts.indexOf(DECISION_COVERAGE_PLACEHOLDER);
  if (decisionCoverageIndex < 0) throw new Error("Engineering Report decision coverage placeholder is missing.");
  reportParts[decisionCoverageIndex] = decisionCoverageMarkdown(metricCoverage);
  return {
    report: reportParts.join("\n"),
    metricCoverage,
    reportAccessedFieldPaths,
  };
}

export function buildEngineeringReport(analysis, options = {}) {
  return buildEngineeringReportArtifacts(analysis, options).report;
}

export { reportBodyForFingerprint, REPORT_VERIFICATION_SENTINEL };
