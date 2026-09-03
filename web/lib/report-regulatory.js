import { artifactIrOperators } from "./artifact-ir-selectors.js";
import {
  contractDynamicDimSummary,
  modelQuantizationStatus,
  quantizationScopeSummary,
} from "./analysis.js";
import { formatBytes, formatNumber, formatPercent, tensorShapeText } from "./format.js";
import { ANALYZER_METADATA } from "./report-metadata.js";
import { buildSecurityPostureEvidence } from "./report-security-posture.js";
import { buildEngineeringReport } from "./report-engineering.js";
import { buildFindingsRegister } from "./report-findings.js";
import { collectArtifactIntegrity } from "./report-integrity.js";
import {
  evidenceClassLegend,
  l1PressureSummary,
  rooflineMixSummary,
  runtimeEvidenceMarkdown,
  weightIndicatorMarkdown,
} from "./report-sections.js";
import { bulletList, code, markdownTable } from "./report-utils.js";

function assessedPercent(value) {
  return value == null ? "N/A; MAC coverage incomplete" : formatPercent(value);
}

export function buildRegulatoryReport(analysis, {
  identity = {},
  fileSizeBytes = 0,
  securityPosture = null,
  runtimeBenchmarkResults = [],
  runtimeEvidence = {},
  deepBomResult = null,
  perturbationResult = null,
  runtimeBasinResult = null,
  deployCurvatureResult = null,
  findingsContext = {},
  generatedAt = new Date().toISOString(),
} = {}) {
  if (!analysis) return "";
  const onnx = String(analysis.format || "tflite").toLowerCase() === "onnx";
  const quant = modelQuantizationStatus(analysis);
  const findings = buildFindingsRegister(analysis, {
    runtimeBasinResult,
    deepBomResult,
    deployCurvatureResult,
    ...findingsContext,
  });
  const security = securityPosture || buildSecurityPostureEvidence({
    analysis,
    model: identity,
    fileSizeBytes: analysis.file_size || fileSizeBytes || 0,
    runtimeBenchmarkResults,
    runtimeBasinResult,
  });
  const artifactIntegrity = collectArtifactIntegrity(analysis, identity, analysis.file_size || fileSizeBytes || 0);
  const reportContext = {
    identity,
    runtimeBenchmarkResults,
    runtimeEvidence,
    deepBomResult,
    perturbationResult,
    runtimeBasinResult,
    deployCurvatureResult,
    findingsContext,
    generatedAt,
  };
  const ortCompatibility = analysis.ort_compatibility_evidence || null;
  const ortFloor = ortCompatibility?.runtime_floor || null;
  const ortProviders = ortCompatibility?.execution_providers || [];
  const executionStructureRow = onnx
    ? ["Execution-provider assignment", "NOT_ASSESSABLE", ortProviders.length
      ? "Pinned per-op ONNX schema versions and ORT kernel-registration ranges were assessed, but type/attribute constraints, actual GetCapability partitioning, and provider placement were not observed."
      : "ONNX Runtime execution-provider assignment was not modeled; no fallback or partition count is asserted."]
    : ["Delegate partitions", "PREDICTED", `${formatNumber(analysis.xnnpack_chains?.length || 0)} predicted delegate segment(s), ${formatNumber(analysis.xnnpack_effective_chain_breaks || 0)} non-structural predicted partition break(s)`];
  const suitabilityRows = onnx ? [
    ["Assessed MAC subtotal", Number(analysis.mac_assessment?.not_assessed_compute_ops || 0) ? "DERIVED/PARTIAL" : "DERIVED", `${formatNumber(analysis.mac_assessment?.total_assessed_macs_decimal || 0)} across ${formatNumber(analysis.mac_assessment?.assessed_compute_ops || 0)}/${formatNumber(analysis.mac_assessment?.compute_ops || 0)} compute op(s); ${formatNumber(analysis.mac_assessment?.not_assessed_compute_ops || 0)} unassessed; complete top-level total ${analysis.total_macs == null ? "withheld" : formatNumber(analysis.total_macs)}`],
    ["Target ridge", "NOT_APPLICABLE", "Suppressed; the selected TFLite planning profile is used only as an L1 working-set reference for ONNX."],
    ["Static intensity posture", "HEURISTIC", rooflineMixSummary(analysis)],
    ["L1 row pressure", "ESTIMATED", l1PressureSummary(analysis)],
    ["Execution-provider traffic", "NOT_ASSESSABLE", "ONNX Runtime execution-provider assignment and partition traffic were not modeled."],
    ["INT8 speedup", "NOT_ASSESSABLE", "No ONNX Runtime execution-provider or target throughput model is applied."],
  ] : [
    ["Total MACs", "DERIVED", formatNumber(identity.total_macs)],
    ["Target ridge", "ESTIMATED", analysis.target_profile?.ridge_point_ops_per_byte ? `~${analysis.target_profile.ridge_point_ops_per_byte} ops/B` : "-"],
    ["Static roofline mix", "ESTIMATED", rooflineMixSummary(analysis)],
    ["L1 row pressure", "ESTIMATED", l1PressureSummary(analysis)],
    ["Fallback tensor traffic", "PREDICTED", `${formatBytes(analysis.fallback_estimated_bytes || 0)} / ${formatPercent(analysis.fallback_byte_percent || 0)}`],
    ["INT8 speedup", "ESTIMATED", `~${Number(analysis.estimated_int8_speedup || 1).toFixed(2)}x for ${identity.target_label}; not measured latency. ${analysis.estimated_int8_speedup_detail || "No method detail emitted."}`],
  ];
  const runtimeCompatibilityRows = onnx ? [
    ["Execution-provider assignment", "NOT_ASSESSABLE", ortProviders.length
      ? `${ortProviders.length} pinned source EP profile(s) assessed after ONNX schema-version resolution; kernel-registration matches are not runtime placement.`
      : "No protected ORT EP source rulepack was loaded."],
    ["Minimum ONNX Runtime version", ortFloor?.evidence_class || "NOT_ASSESSABLE", ortFloor?.minimum_ort_version
      ? `Necessary standard-domain floor ORT ${ortFloor.minimum_ort_version}; status ${ortFloor.status}; not a sufficient EP/build/device guarantee.`
      : "IR/opset identity is observed, but no complete pinned runtime floor was derived."],
    ["Runtime review watchlist", "OBSERVED/HEURISTIC", (analysis.runtime_review_watchlist || []).map((item) => `${item.name}:${item.count} (${item.reason_code})`).join(" / ") || "none"],
    ["Suppressed analyses", "NOT_ASSESSABLE", (analysis.onnx_sections_suppressed || []).join(" / ") || "none declared"],
  ] : [
    ["XNNPACK partition boundaries", "PREDICTED", `${formatNumber(analysis.xnnpack_chain_breaks || 0)} total = ${formatNumber(analysis.xnnpack_effective_chain_breaks || 0)} non-structural + ${formatNumber(analysis.xnnpack_structural_chain_breaks || 0)} structural; ${formatNumber(analysis.xnnpack_zero_mac_chain_breaks || 0)} zero-MAC non-structural is a subset of non-structural`],
    ["Conditionally delegatable MACs", "PREDICTED", formatPercent(analysis.delegated_mac_percent || 0)],
    ["Fallback families", "PREDICTED", (analysis.fallback_traffic_by_op_family || []).slice(0, 6).map((item) => `${item.name}:${formatBytes(item.estimated_bytes)}`).join(" / ") || "none"],
    ["Fusion review", "PREDICTED", `${formatNumber((artifactIrOperators(analysis) || []).filter((op) => String(op.fusion_status || "").includes("review")).length)} op(s)`],
    ["Prediction status", "PREDICTED", `Based on ${ANALYZER_METADATA.rulepackVersion} and local runtime support profile; not a confirmed delegate log.`],
  ];
  return [
    "# DEEPBOM Model Artifact Regulatory Support Report",
    "",
    "## 1. Document Control",
    markdownTable(["Field", "Value"], [
      ["Report schema", ANALYZER_METADATA.schemas.regulatoryReport],
      ["Generated", generatedAt],
      ["Generated by", ANALYZER_METADATA.displayName],
      ["Analyzer version", ANALYZER_METADATA.version],
      ["Rulepack version", ANALYZER_METADATA.rulepackVersion],
      ["Artifact filename", code(identity.filename)],
      ["Artifact SHA-256", code(identity.sha256 || "pending-browser-export")],
      ["Target profile", identity.target_label],
      ["Report boundary", "Deployment artifact technical characterization only"],
    ]),
    "",
    "## 2. Scope And Explicit Limitations",
    "### Input",
    "- One TFLite or ONNX deployment artifact selected through the browser File API.",
    "",
    "### Automatically Evaluated",
    bulletList([
      "artifact structure and integrity signals",
      "computational graph, operator inventory, tensor contract, and dynamic dimensions",
      "quantization configuration and numerical metadata",
      "compute, memory, cache, and target-profile suitability estimates",
      "delegate/runtime compatibility predictions",
      "browser-local synthetic runtime and numerical consistency evidence when executed",
      "deployment-artifact weight/topology and perturbation proxies when executed",
      "artifact-level security-relevant indicators and integrity posture",
    ]),
    "",
    "### Not Evaluated",
    bulletList([
      "training, validation, clinical, or representative dataset quality",
      "clinical accuracy, clinical benefit, intended-use suitability, bias, subgroup performance, or population generalizability",
      "complete medical-device cybersecurity, threat modeling, network architecture, patch process, or total-product lifecycle controls",
      "regulatory conformity, risk acceptability, patient-harm severity, or market authorization readiness",
    ]),
    "",
    "## 3. Evidence Classification",
    evidenceClassLegend(),
    "",
    "## 4. Artifact Identification And Integrity",
    markdownTable(["Evidence", "Class", "Value"], [
      ["Filename", "OBSERVED", code(identity.filename)],
      ["Format", "OBSERVED", code((identity.format || "unknown").toUpperCase())],
      ["SHA-256", "OBSERVED", code(identity.sha256 || "pending-browser-export")],
      ["File size", "OBSERVED", formatBytes(analysis.file_size || fileSizeBytes || 0)],
      ["Schema / opset", "OBSERVED", code(artifactIntegrity.schema_or_opset)],
      ["Graph / producer", "OBSERVED", [artifactIntegrity.graph_name, artifactIntegrity.onnx_producer].filter(Boolean).join(" / ") || "-"],
      ["Subgraphs", "OBSERVED", formatNumber(analysis.subgraph_count || 1)],
      ["Operators", "OBSERVED", formatNumber(identity.operator_count)],
      ["Tensors", "OBSERVED", formatNumber(identity.tensor_count)],
      ["Custom-like op families", "OBSERVED", artifactIntegrity.custom_like_ops.length ? artifactIntegrity.custom_like_ops.map((item) => `${item.name}:${item.count}`).join(" / ") : "none detected"],
      ["External ONNX domains", "OBSERVED", artifactIntegrity.external_domains.length ? artifactIntegrity.external_domains.join(" / ") : "none detected"],
      ["ONNX external data", "OBSERVED", artifactIntegrity.external_data_summary],
      ["TFLite tensor buffers", "OBSERVED", artifactIntegrity.tflite_buffer_summary],
      ["Analyzer/report schema", "OBSERVED", `${ANALYZER_METADATA.schemas.regulatoryReport} / ${ANALYZER_METADATA.schemas.staticAnalysis}`],
    ]),
    "",
    "## 5. Model Architecture And Interface Contract",
    markdownTable(["Item", "Evidence", "Result"], [
      ["Inputs", "OBSERVED", (analysis.inputs || []).map((tensor) => `${tensor.name}:${tensor.dtype}${tensorShapeText(tensor)}`).join(" / ") || "-"],
      ["Outputs", "OBSERVED", (analysis.outputs || []).map((tensor) => `${tensor.name}:${tensor.dtype}${tensorShapeText(tensor)}`).join(" / ") || "-"],
      ["Dynamic input dimensions", "OBSERVED", contractDynamicDimSummary(analysis.inputs) || "none detected"],
      ["Operator inventory", "OBSERVED", (analysis.histogram || []).slice(0, 12).map((item) => `${item.name}:${item.count}`).join(" / ")],
      ["Stage decomposition", "DERIVED", `${formatNumber((analysis.stages || []).length)} stage(s)`],
      ["Recognized patterns", "DERIVED", (analysis.patterns || []).map((item) => `${item.name || "pattern"}:${item.count || 1}`).join(" / ") || "none detected"],
      executionStructureRow,
    ]),
    "",
    "## 6. Quantization And Numerical Contract",
    markdownTable(["Item", "Evidence", "Result"], [
      ["Classification", "DERIVED", quant.label || quant.classification || "unknown"],
      ["Scope", "DERIVED", quantizationScopeSummary(analysis)],
      ["Quantized tensors", "OBSERVED", `${formatNumber(analysis.quantized_tensors || 0)} / ${formatNumber(analysis.tensor_count || 0)}`],
      ["Per-channel/per-axis metadata", "OBSERVED", formatNumber(analysis.per_channel_tensors || 0)],
      ["Quantized compute MACs", "DERIVED", assessedPercent(quant.quantized_compute_mac_percent)],
      ["Zero-point/scale risk", "DERIVED", `${formatNumber((artifactIrOperators(analysis) || []).filter((op) => op.quant_risk === "risk" || op.quant_risk === "warn").length)} op(s) flagged`],
      ["Interpretation", "PROXY", "Quantization metadata can trigger review but cannot prove accuracy degradation without representative data."],
    ]),
    "",
    "## 7. Compute, Memory And Cache Suitability",
    markdownTable(["Item", "Evidence", "Result"], suitabilityRows),
    "",
    onnx ? "## 8. ONNX Execution-Provider And Runtime Compatibility" : "## 8. Fusion, Delegate And Runtime Compatibility",
    markdownTable(["Item", "Evidence", "Result"], runtimeCompatibilityRows),
    "",
    "## 9. Browser Runtime Compatibility And Numerical Consistency",
    runtimeEvidenceMarkdown({
      runtimeBenchmarkResults,
      runtimeBasinResult,
      runtimeEvidence,
    }),
    "",
    "## 10. Weight-Space Stability Indicators",
    weightIndicatorMarkdown({
      deepBomResult,
      perturbationResult,
      deployCurvatureResult,
    }),
    "",
    "## 11. AI Model Artifact Security And Integrity Posture",
    markdownTable(["Posture item", "Evidence", "Result"], [
      ["Raw model upload", "OBSERVED", security.privacy.raw_model_upload],
      ["Generated tensor/output upload", "OBSERVED", security.privacy.generated_tensor_upload],
      ["Model bytes in ZIP", "OBSERVED", security.privacy.raw_model_bytes_included],
      ["Artifact hash binding", "OBSERVED", identity.sha256 ? "present" : "pending"],
      ["Schema/opset posture", "OBSERVED", security.integrity.schema_or_opset || artifactIntegrity.schema_or_opset],
      ["External data posture", "OBSERVED", security.integrity.external_data_summary || artifactIntegrity.external_data_summary],
      ["TFLite buffer posture", "OBSERVED", security.integrity.tflite_buffer_summary || artifactIntegrity.tflite_buffer_summary],
      ["Custom/external execution surface", "PREDICTED", security.execution_surface.summary],
      ["Resource exhaustion indicators", "ESTIMATED", security.resource_exhaustion.summary],
      ["Runtime execution integrity", "MEASURED_SYNTHETIC", security.execution_integrity.summary],
      ["Boundary statement", "NOT_ASSESSABLE", "This is an application-observed local processing record, not a no-egress proof for the entire operating system or network."],
    ]),
    "",
    "## 12. Prioritized Findings Register",
    findings.length
      ? markdownTable(["Finding ID", "Origin", "Category", "Evidence", "Priority", "Affected", "Recommendation"], findings.map((finding) => [
          finding.finding_id,
          finding.origin || "report_synthesis",
          finding.category,
          finding.evidence_class,
          finding.technical_priority,
          [finding.affected_operator, finding.affected_tensor].filter(Boolean).join(" / ") || "-",
          finding.recommendation,
        ]))
      : "No prioritized finding was generated from the current evidence set.",
    "",
    "## 13. Regulatory Relevance Map",
    markdownTable(["Area", "Relevance", "Boundary"], [
      ["Software technical characterization", "Model architecture, I/O contract, target assumptions, and runtime compatibility can support technical documentation.", "Does not establish intended-use suitability or clinical performance."],
      ["Configuration identification", "SHA-256, format, schema/opset, operator inventory, and ML-BOM bind evidence to a specific artifact.", "Does not replace full device software configuration management."],
      ["Cybersecurity/SBOM support", "CycloneDX ML-BOM and artifact integrity posture support model artifact inventory.", "Not a complete medical-device SBOM/CBOM or threat model."],
      ["Risk-management input", "Findings can feed engineering risk analysis as technical observations.", "Technical priority is not patient harm severity or risk acceptability."],
      ["AI Act/MDR-style technical file support", "Evidence can support explainable technical characterization of a deployed AI component.", "No conformity conclusion is produced."],
    ]),
    "",
    "## 14. Methodology, Rulepack And Limitations",
    bulletList([
      "OBSERVED facts are parsed from the selected deployment artifact in browser memory.",
      "DERIVED metrics such as MACs, graph stages, and quantization scope are computed from parsed graph metadata.",
      "ESTIMATED cache/roofline/speedup values depend on the selected target profile and must be rerun after target changes.",
      "PREDICTED delegate partitions are based on local rulepack/runtime support profiles and are not confirmed delegate logs.",
      "MEASURED_SYNTHETIC results use synthetic or prepared browser-local inputs and are not clinical validation.",
      "PROXY weight/topology/perturbation composites are descriptive and experimentally unvalidated; they do not establish prioritization thresholds, accuracy, latency, robustness, or generalization.",
    ]),
    "",
    "## 15. Engineering Evidence Appendix",
    "The Engineering Report is included below as an appendix so the Regulatory Report remains the larger evidence package. The Engineering Report itself does not include this regulatory section.",
    "",
    buildEngineeringReport(analysis, reportContext),
    "",
    "## 16. References For Regulatory Framing",
    "- FDA, Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket",
    "- FDA, Cybersecurity in Medical Devices FAQ, including SBOM discussion: https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity-medical-devices-frequently-asked-questions-faqs",
    "- IMDRF N73, Principles and Practices for Software Bill of Materials for Medical Device Cybersecurity: https://www.imdrf.org/documents/principles-and-practices-software-bill-materials-medical-device-cybersecurity",
    "- Regulation (EU) 2024/1689 Artificial Intelligence Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    "",
  ].join("\n");
}
