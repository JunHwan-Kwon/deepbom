import { artifactIrOperators, artifactIrValues } from "./artifact-ir-selectors.js";
import { alignmentLabel, bottleneckDistributionData, predictedPartitionBoundaryInventory } from "./analysis.js";
import { formatBytes, formatNumber, formatPercent, formatPercentRange, formatScientific, formatUs, maxBy, padOp } from "./format.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { ANALYZER_METADATA } from "./report-metadata.js";
import { buildQuantizationContractChecks } from "./report-quantization-contracts.js";

const FINDING_CLASSES = {
  integration: "integration requirement",
  integration_verification: "integration verification",
  execution_provider_compatibility: "execution-provider compatibility",
  quantization: "deployment-risk (requires confirmation)",
  quantization_design_review: "quantization design review",
  quantization_calibration_review: "quantization calibration review",
  numerical_structure_review: "numerical structure review",
  delegate: "deployment-risk (requires confirmation)",
  packing: "optimization opportunity",
  memory_cache: "deployment-resource verification",
  channel_alignment: "optimization opportunity",
  runtime: "deployment-risk (requires confirmation)",
  runtime_compatibility: "runtime compatibility",
  evidence_reproducibility: "evidence reproducibility",
  lineage: "lineage requirement",
  synthetic_sensitivity: "deployment-risk (requires confirmation)",
  integrity: "defect / integrity concern",
  limitation: "limitation",
};

const FINDING_EVIDENCE_PATHS = Object.freeze({
  "EA-IOC-0001": ["/evidence/static_analysis/inputs/0", "/evidence/static_analysis/metadata_presence/documented_preprocessing"],
  "EA-DYN-0001": ["/evidence/static_analysis/dynamic_shape_cost_contract", "/evidence/static_analysis/inputs"],
  "EA-OUT-0001": ["/evidence/static_analysis/outputs/0", "/evidence/static_analysis/metadata_presence/output_semantics_documented"],
  "EA-LIN-0001": ["/evidence/static_analysis/metadata_presence"],
  "EA-PROV-0001": ["/evidence/static_analysis/target_profile", "/analyzer_metadata"],
  "EA-RUN-0001": ["/evidence/static_analysis/runtime_compat"],
  "EA-GRF-0001": ["/evidence/static_analysis/subgraphs", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0004": ["/evidence/static_analysis/onnx_external_data", "/evidence/static_analysis/size_breakdown/external_data_tensor_count", "/evidence/static_analysis/weight_integrity/tensor_results"],
  "EA-CML-0001": ["/evidence/static_analysis/weight_integrity/parameters"],
  "EA-CML-0002": ["/evidence/static_analysis/weight_integrity/parameters"],
  "EA-CML-0003": ["/evidence/static_analysis/weight_integrity"],
  "EA-WGT-0001": ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details"],
  "EA-QNT-0100": ["/evidence/static_analysis/weight_integrity/quant_grid_detail", "/evidence/static_analysis/weight_integrity/quant_grid_details"],
  "EA-QNT-0101": ["/evidence/quantization/quantization_contract_checks/kernel_quantization"],
  "EA-QNT-0102": ["/evidence/quantization/quantization_contract_checks/bias_scale"],
  "EA-QNT-0103": ["/evidence/quantization/quantization_contract_checks/residual_add"],
  "EA-QNT-0104": ["/evidence/quantization/quantization_contract_checks/weight_zero_point"],
  "EA-QNT-0105": ["/evidence/quantization/quantization_contract_checks/parameter_integrity"],
  "EA-QNT-0106": ["/evidence/quantization/quantization_contract_checks/accumulator_bound", "/evidence/static_analysis/accumulator_atlas"],
  "EA-QNT-0107": ["/evidence/quantization/quantization_contract_checks/requantization_fidelity", "/evidence/static_analysis/requantization_fidelity"],
  "EA-QNT-0108": ["/evidence/static_analysis/quantization_lattice", "/evidence/static_analysis/contract_migration", "/evidence/quantization/quantization_contract_checks/residual_add", "/evidence/quantization/quantization_contract_checks/contract_migration"],
  "EA-QNT-0109": ["/evidence/static_analysis/residual_step_response", "/evidence/quantization/quantization_contract_checks/residual_step_response"],
  "EA-QNT-0110": ["/evidence/static_analysis/residual_contract_distortion", "/evidence/quantization/quantization_contract_checks/residual_contract_distortion"],
  "EA-QNT-0111": ["/evidence/static_analysis/kernel_extremum_witness", "/evidence/quantization/quantization_contract_checks/kernel_extremum_witness"],
  "EA-QNT-0112": ["/evidence/static_analysis/channel_vitality", "/evidence/quantization/quantization_contract_checks/channel_vitality"],
  "EA-QNT-0113": ["/evidence/static_analysis/rounding_equivalence", "/evidence/quantization/quantization_contract_checks/rounding_equivalence"],
  "EA-QNT-0114": ["/evidence/static_analysis/numerical_abi_propagation", "/evidence/quantization/quantization_contract_checks/numerical_abi_propagation"],
  "EA-QNT-0115": ["/evidence/static_analysis/accumulator_reachability", "/evidence/quantization/quantization_contract_checks/accumulator_reachability"],
  "EA-QNT-0116": ["/evidence/static_analysis/input_counterexample", "/evidence/quantization/quantization_contract_checks/input_counterexample"],
  "EA-DEL-0004": ["/evidence/static_analysis/delegation_repair/runtime_build_risks"],
  "EA-QNT-0117": ["/evidence/static_analysis/preprocessing_realizability", "/evidence/quantization/quantization_contract_checks/preprocessing_realizability"],
  "EA-QNT-0118": ["/evidence/quantization/quantization_contract_checks/representable_kernel_channels"],
  "EA-QNT-0119": ["/evidence/quantization/quantization_contract_checks/input_quantization_convention", "/evidence/static_analysis/inputs"],
  "EA-QNT-0001": ["/evidence/static_analysis/quant_holes", "/evidence/static_analysis/quant_hole_count"],
  "EA-DEL-0001": ["/evidence/static_analysis/xnnpack_chains", "/evidence/static_analysis/ops"],
  "EA-QNT-0002": ["/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-DEL-0002": ["/evidence/static_analysis/predicted_partition_boundaries"],
  "EA-DEL-0003": ["/evidence/runtime_results/runtime_assignment/comparison"],
  "EA-MEM-0001": ["/evidence/static_analysis/tensor_liveness"],
  "EA-MEM-0002": [
    "/evidence/static_analysis/tensor_arena_plan",
    "/evidence/runtime_results/runtime_assignment/runtime_memory",
    "/evidence/runtime_results/runtime_assignment/arena_reconciliation",
  ],
  "EA-PKG-0001": ["/evidence/static_analysis/ops"],
  "EA-CHN-0001": ["/evidence/static_analysis/ops", "/evidence/static_analysis/target_profile"],
  "EA-RUN-0002": ["/evidence/runtime_results/runtime_basin"],
  "EA-RUN-0003": ["/evidence/runtime_results/preprocessing_consequence_atlas"],
  "EA-ONX-0001": ["/evidence/static_analysis/onnx_domain_analysis/external_custom_domains"],
  "EA-ONX-0002": ["/evidence/static_analysis/onnx_domain_analysis/recursive_function_cycles", "/evidence/static_analysis/onnx_domain_analysis/duplicate_function_ids"],
  "EA-ONX-0003": ["/evidence/static_analysis/onnx_domain_analysis/functions"],
  "EA-ONX-0005": ["/evidence/static_analysis/ort_compatibility_evidence/execution_providers", "/evidence/static_analysis/ort_ep_portability_frontier"],
  "EA-ONX-0006": ["/evidence/static_analysis/onnx_shape_inference/declaration_conflicts", "/evidence/static_analysis/tensors"],
  "EA-ONX-0071": ["/evidence/static_analysis/onnx_shape_inference/semantic_contract_conflicts", "/evidence/static_analysis/onnx_shape_inference/rule_unresolved_nodes", "/evidence/static_analysis/tensors"],
  "EA-ONX-0072": ["/evidence/static_analysis/onnx_shape_inference/conditionally_invalid_node_output_count", "/evidence/static_analysis/onnx_shape_inference/conditional_invalid_variant_count", "/evidence/static_analysis/tensors"],
  "EA-ONX-0007": ["/evidence/static_analysis/onnx_shape_inference/opset_import_contract", "/evidence/static_analysis/onnx_shape_inference/schema_form_rows", "/evidence/static_analysis/onnx_shape_inference/source_documents"],
  "EA-ONX-0008": ["/evidence/static_analysis/onnx_shape_inference/shape_scope", "/evidence/static_analysis/onnx_shape_inference/extended_scope_inference", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/partial_rows", "/evidence/static_analysis/onnx_domain_analysis/functions"],
  "EA-ONX-0009": ["/evidence/static_analysis/onnx_shape_inference/extended_scope_inference", "/evidence/static_analysis/onnx_shape_inference/shape_scope"],
  "EA-ONX-0010": ["/evidence/static_analysis/onnx_type_proto_contract/invalid_rows", "/evidence/static_analysis/onnx_type_proto_contract/source_sha256"],
  "EA-ONX-0011": ["/evidence/static_analysis/onnx_sparse_tensor_contract/invalid_rows", "/evidence/static_analysis/onnx_sparse_tensor_contract/source_sha256"],
  "EA-ONX-0012": ["/evidence/static_analysis/onnx_shape_inference/container_value_inference/failed_rows", "/evidence/static_analysis/onnx_shape_inference/container_value_inference/source_documents"],
  "EA-ONX-0013": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/failed_rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents"],
  "EA-ONX-0014": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents"],
  "EA-ONX-0015": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0016": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0017": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0018": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0019": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0020": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0021": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0022": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0023": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0024": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0025": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0026": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0027": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0028": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0029": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0030": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0031": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0032": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0033": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0034": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0035": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0036": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0037": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0038": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/failed_rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0039": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0040": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0042": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0043": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0044": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0045": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0046": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0047": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0048": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0049": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/tensors"],
  "EA-ONX-0050": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0051": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0052": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0053": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0054": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0055": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0056": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0057": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0058": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0059": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0060": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0061": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents"],
  "EA-ONX-0062": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0063": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"],
  "EA-ONX-0064": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0065": ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0066": ["/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/failed_rows", "/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/source_documents", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"],
  "EA-ONX-0067": ["/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/rows", "/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/source_documents", "/evidence/static_analysis/ops"],
  "EA-ONX-0068": ["/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/rows", "/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/source_documents", "/evidence/static_analysis/tensors"],
  "EA-ONX-0069": ["/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/rows", "/evidence/static_analysis/ops"],
  "EA-ONX-0070": ["/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/partial_rows", "/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference/rows"],
  "EA-SYN-0001": ["/evidence/weight_indicators/deploy_curvature_basin"],
  "EA-LIM-0001": ["/evidence/static_analysis"],
});

const EVIDENCE_CONFIDENCE = Object.freeze({
  OBSERVED: "high",
  "OBSERVED/DERIVED": "high",
  "OBSERVED/NOT_ASSESSABLE": "high",
  SOURCE_PINNED_AND_DERIVED: "high",
  SOURCE_PINNED_AND_OBSERVED: "high",
  DERIVED: "high",
  DERIVED_FROM_OBSERVED_RUNTIME: "high",
  NOT_ASSESSABLE: "high",
  HEURISTIC: "medium",
  ESTIMATED: "medium",
  PREDICTED: "medium",
  MEASURED_SYNTHETIC: "medium",
  PROXY: "low",
});

export function finding({
  id,
  category,
  title,
  evidence,
  priority,
  op = "",
  tensor = "",
  observation = "",
  interpretation = "",
  recommendation = "",
  relevance = "",
  findingClass = "",
  origin = "report_synthesis",
  sourceRuleId = "",
  evidenceJsonPointers = null,
  methodVersion = "",
  confidence = "",
}) {
  return {
    finding_id: id,
    origin,
    source_rule_id: sourceRuleId || `deepbom.report_findings.${id}.v1`,
    evidence_json_pointers: evidenceJsonPointers || FINDING_EVIDENCE_PATHS[id] || ["/"],
    method_version: methodVersion || ANALYZER_METADATA.schemas.findingsRegister,
    confidence: confidence || EVIDENCE_CONFIDENCE[evidence] || "not_declared",
    confidence_scope: "Confidence describes the evidence extraction or deterministic method, not the likelihood or severity of a product effect.",
    category,
    finding_class: findingClass || FINDING_CLASSES[category] || "engineering review",
    title,
    evidence_class: evidence,
    technical_priority: priority,
    affected_operator: op,
    affected_tensor: tensor,
    observation,
    interpretation,
    possible_effects: possibleEffectsForCategory(category),
    recommendation,
    limitations: evidence === "NOT_ASSESSABLE" ? "Model artifact alone is insufficient." : "Confirm with target runtime and/or representative data where applicable.",
    regulatory_relevance: relevance,
  };
}

export function normalizeNativeAnalyzerFindings(analysis, normalizedFindings = []) {
  const idMap = {
    "input-contract": "EA-IOC-0001",
    "quant-holes": "EA-QNT-0001",
    "peak-activation-memory": "EA-MEM-0001",
    "heavy-data-movement": "EA-MEM-0001",
    "low-intensity-majority": "EA-MEM-0001",
    "weight-integrity-zero-kernel-slices": "EA-WGT-0001",
    "quant-grid-utilization": "EA-QNT-0100",
    "depthwise-per-tensor-weights": "EA-QNT-0101",
    "asymmetric-uint8-weights": "EA-QNT-0104",
  };
  return (analysis?.findings || []).map((item, index) => {
    const opIndex = Number(String(item.id || "").match(/^xnn-break-(\d+)$/)?.[1]);
    const normalized = Number.isInteger(opIndex)
      ? normalizedFindings.find((findingItem) => String(findingItem.affected_operator || "").includes(`#${String(opIndex).padStart(3, "0")}`))
      : normalizedFindings.find((findingItem) => findingItem.finding_id === idMap[item.id]);
    return {
      signal_id: item.id || `signal_${index}`,
      title: item.title || "",
      category: item.category || "",
      observations: Array.isArray(item.evidence) ? item.evidence : [],
      origin: "native_analyzer",
      authoritative: false,
      classification: "raw_analyzer_signal",
      normalized_finding_id: normalized?.finding_id || null,
      source_rule_id: `deepbom.native_analyzer.${item.id || `signal_${index}`}.v1`,
      evidence_json_pointers: [`/evidence/static_analysis/raw_analyzer_signals/${index}`],
      method_version: ANALYZER_METADATA.rulepackVersion,
      confidence: item.confidence || "not_declared",
      confidence_scope: "This is an extraction signal only. Priority, interpretation, and action are authoritative only in the normalized findings register.",
    };
  });
}

export function possibleEffectsForCategory(category) {
  const map = {
    integration: ["preprocessing mismatch", "label-map mismatch", "application contract drift"],
    integration_verification: ["runtime support gap", "release-manifest omission", "deployment reproducibility gap"],
    quantization: ["quantization sensitivity", "calibration review priority", "backend-specific numerical drift"],
    quantization_design_review: ["quantization design review", "converter lineage review", "representative regression planning"],
    quantization_calibration_review: ["calibration range review", "representative histogram review", "quantization sensitivity triage"],
    numerical_structure_review: ["model-content structure review", "representative regression planning", "converter/checkpoint triage"],
    delegate: ["fallback latency", "copy traffic", "runtime partition overhead"],
    packing: ["first-run latency", "weight repacking cost", "benchmark warmup sensitivity"],
    memory_cache: ["reduced cache reuse", "memory-traffic latency", "bandwidth pressure"],
    channel_alignment: ["SIMD lane padding", "scalar tail overhead", "micro-kernel inefficiency"],
    runtime: ["backend-specific latency anomaly", "numeric drift", "silent fallback"],
    evidence_reproducibility: ["report reproducibility gap", "audit-trace ambiguity", "target-profile drift"],
    lineage: ["source-artifact traceability gap", "converter drift", "release-manifest omission"],
    synthetic_sensitivity: ["output plateau", "rank instability under perturbation"],
    limitation: ["not assessable from deployment artifact alone"],
  };
  return map[category] || ["engineering review priority"];
}

export function buildFindingsRegister(analysis, {
  runtimeEvidence = null,
  runtimeBasinResult = null,
  deepBomResult = null,
  deployCurvatureResult = null,
  analyzerMetadata = ANALYZER_METADATA,
} = {}) {
  const metadata = analyzerMetadata && typeof analyzerMetadata === "object" ? analyzerMetadata : ANALYZER_METADATA;
  const findings = [];
  const ops = artifactIrOperators(analysis) || [];
  const isOnnx = (analysis?.format || "tflite") === "onnx";
  const coreMlIntegrity = analysis?.format === "coreml" ? analysis?.weight_integrity : null;
  if (String(coreMlIntegrity?.schema || "").startsWith("deepbom.coreml.")) {
    const coreMlProgramConstants = String(coreMlIntegrity.schema).includes("mlprogram");
    const coreMlClassicalParameters = String(coreMlIntegrity.schema).includes("classical") || String(coreMlIntegrity.schema).includes("pipeline");
    const coreMlConstantLabel = coreMlProgramConstants ? "Core ML MIL constants" : coreMlClassicalParameters ? "Core ML serialized numerical parameters" : "Core ML WeightParams";
    const parameters = Array.isArray(coreMlIntegrity.parameters) ? coreMlIntegrity.parameters : [];
    const parameterName = (row) => `${row.pipeline_model_name ? `${row.pipeline_model_name}/` : ""}${row.layer_name || row.tensor_name || row.pipeline_model_type || `tensor_${row.tensor_index ?? "?"}`}/${row.role || "constant"}`;
    const nonfinite = parameters.filter((row) => Number(row.numerical_integrity?.nonfinite_count || 0) > 0);
    if (nonfinite.length) findings.push(finding({
      id: "EA-CML-0001",
      category: "integrity",
      title: "Core ML serialized constants contain non-finite values",
      evidence: "OBSERVED",
      priority: "High",
      op: coreMlConstantLabel,
      tensor: nonfinite.slice(0, 8).map(parameterName).join(", "),
      observation: `${formatNumber(coreMlIntegrity.nonfinite_value_count || 0)} NaN or infinity value(s) were decoded from ${formatNumber(nonfinite.length)} Core ML constant payload(s): ${nonfinite.slice(0, 8).map((row) => `${parameterName(row)} (${formatNumber(row.numerical_integrity.nonfinite_count || 0)})`).join("; ")}.`,
      interpretation: `This is direct numerical evidence from serialized ${coreMlProgramConstants ? "MIL immediate/package-blob constant bytes" : coreMlClassicalParameters ? "classical coefficient, support-vector, threshold, or nested pipeline parameter bytes" : "WeightParams bytes"}, not a runtime or task-accuracy inference.`,
      recommendation: "Reject the release artifact, trace the affected exporter/checkpoint values, regenerate it, and require a clean payload scan before execution validation.",
      relevance: "serialized numerical integrity; release gating",
    }));
    const allZero = parameters.filter((row) => (coreMlIntegrity.schema.includes("mlprogram") || ["weights", "data", "support_vectors", "coefficients"].includes(row.role)
      || String(row.role || "").startsWith("weights_")) && row.numerical_integrity?.all_zero === true && row.value_count > 0);
    if (allZero.length) findings.push(finding({
      id: "EA-CML-0002",
      category: "numerical_structure_review",
      title: "Core ML serialized constants are exactly all-zero",
      evidence: "DERIVED",
      priority: "Medium",
      op: coreMlConstantLabel,
      tensor: allZero.slice(0, 8).map(parameterName).join(", "),
      observation: `${formatNumber(allZero.length)}/${formatNumber(coreMlIntegrity.assessed_parameter_count || 0)} assessed ${coreMlProgramConstants ? "MIL constant" : coreMlClassicalParameters ? "numerical parameter" : "WeightParams"} payload(s) contain only exact zero values.`,
      interpretation: "Exact all-zero storage is deterministic; whether a zero bias is intentional or a zero kernel is defective depends on its bound layer role.",
      recommendation: "Review zero kernels as release blockers and verify zero bias/offset parameters against the source checkpoint and export manifest.",
      relevance: "weight payload integrity; exporter verification",
    }));
    if (Number(coreMlIntegrity.parameter_count || 0) > Number(coreMlIntegrity.assessed_parameter_count || 0)) findings.push(finding({
      id: "EA-CML-0003",
      category: "limitation",
      title: "Core ML constant numerical coverage is partial",
      evidence: "OBSERVED/NOT_ASSESSABLE",
      priority: "Informational",
      op: coreMlConstantLabel,
      observation: `${formatNumber(coreMlIntegrity.assessed_parameter_count || 0)}/${formatNumber(coreMlIntegrity.parameter_count || 0)} ${coreMlProgramConstants ? "MIL constant" : coreMlClassicalParameters ? "serialized numerical parameter" : "WeightParams"} payload(s) were numerically assessed; ${Number.isSafeInteger(coreMlIntegrity.payload_bytes) ? `${formatNumber(coreMlIntegrity.assessed_payload_bytes || 0)}/${formatNumber(coreMlIntegrity.payload_bytes)} payload bytes assessed` : "the external package payload byte denominator remains unbound"}.`,
      interpretation: "Unimplemented layer cardinality or encoding semantics remain explicit and are not converted to a clean zero-anomaly result.",
      recommendation: "Use a source-pinned decoder for every listed storage contract before treating Core ML payload integrity as complete.",
      relevance: "analysis coverage; evidence completeness",
    }));
  }
  const serializedIntegrity = analysis?.tensor_numerical_integrity;
  if (serializedIntegrity) {
    const records = Array.isArray(serializedIntegrity.tensor_records) ? serializedIntegrity.tensor_records : [];
    const nonfinite = records.filter((row) => Number(row.nan_value_count || 0)
      + Number(row.positive_infinity_value_count || 0) + Number(row.negative_infinity_value_count || 0) > 0);
    const invalid = records.filter((row) => Number(row.invalid_encoding_value_count || 0) > 0
      || Number(row.nonfinite_scale_block_count || 0) > 0);
    const compromised = [...new Map([...nonfinite, ...invalid].map((row) => [row.tensor_index, row])).values()];
    if (compromised.length) {
      findings.push(finding({
        id: "EA-SER-0001",
        category: "numerical_structure_review",
        title: "Serialized tensor payload contains non-finite or invalid numerical values",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: "serialized tensor payload",
        tensor: compromised.slice(0, 8).map((row) => row.tensor_name).join(", "),
        observation: `${formatNumber(serializedIntegrity.nonfinite_value_count || 0)} non-finite decoded value(s), ${formatNumber(serializedIntegrity.invalid_encoding_value_count || 0)} invalid encoded value(s), and ${formatNumber(serializedIntegrity.nonfinite_scale_block_count || 0)} block(s) with non-finite scale metadata were found by a full payload scan. Affected tensors: ${compromised.slice(0, 8).map((row) => `${row.tensor_name} (${formatNumber(row.nan_value_count || 0)} NaN, ${formatNumber(Number(row.positive_infinity_value_count || 0) + Number(row.negative_infinity_value_count || 0))} Inf, ${formatNumber(row.invalid_encoding_value_count || 0)} invalid)`).join("; ")}${compromised.length > 8 ? `; plus ${formatNumber(compromised.length - 8)} more` : ""}.`,
        interpretation: "The values are read from the declared serialized tensor ranges. For GGUF block encodings, numerical values are derived with the pinned llama.cpp dequantization formulas; this is not a task-accuracy claim.",
        recommendation: "Reject the release artifact, inspect the named source tensors and exporter/quantizer path, regenerate the artifact, and require a clean full-payload scan before runtime validation.",
        relevance: "serialized numerical integrity; release gating",
        evidenceJsonPointers: ["/evidence/static_analysis/tensor_numerical_integrity"],
      }));
    }
    const allZero = records.filter((row) => row.status === "assessed_full_payload" && row.all_zero);
    if (allZero.length) {
      findings.push(finding({
        id: "EA-SER-0002",
        category: "numerical_structure_review",
        title: "Serialized tensor payloads are exactly all-zero",
        evidence: "DERIVED",
        priority: "Medium",
        op: "serialized tensor payload",
        tensor: allZero.slice(0, 8).map((row) => row.tensor_name).join(", "),
        observation: `${formatNumber(allZero.length)}/${formatNumber(serializedIntegrity.assessed_tensor_count || 0)} fully decoded tensor payload(s) contain only exact zero values: ${allZero.slice(0, 8).map((row) => `${row.tensor_name} (${formatNumber(row.value_count || 0)} values)`).join("; ")}${allZero.length > 8 ? `; plus ${formatNumber(allZero.length - 8)} more` : ""}.`,
        interpretation: "Exact all-zero storage is deterministic evidence about the serialized tensor, but its functional effect cannot be inferred without an executable graph and consumer semantics.",
        recommendation: "Confirm that each all-zero tensor is intentional in the source checkpoint/export manifest; for executable models, join it to consumer operators and representative-output regression.",
        relevance: "weight payload integrity; export verification",
        evidenceJsonPointers: ["/evidence/static_analysis/tensor_numerical_integrity/tensor_records"],
      }));
    }
    if (serializedIntegrity.status !== "assessed") {
      findings.push(finding({
        id: "EA-SER-0003",
        category: "limitation",
        title: "Some serialized tensor payload semantics are not assessed",
        evidence: "OBSERVED/NOT_ASSESSABLE",
        priority: "Informational",
        op: "serialized tensor payload",
        observation: `${formatNumber(serializedIntegrity.assessed_tensor_count || 0)}/${formatNumber(serializedIntegrity.tensor_count || 0)} tensor payloads and ${formatNumber(serializedIntegrity.assessed_tensor_bytes || 0)}/${formatNumber(serializedIntegrity.declared_tensor_bytes || 0)} declared tensor bytes were numerically decoded. ${formatNumber(serializedIntegrity.unassessed_tensor_count || 0)} tensor(s) remain explicit: ${(serializedIntegrity.limitations || []).slice(0, 6).map((row) => `${row.tensor_name} ${row.dtype}: ${row.reason}`).join("; ") || "reason ledger unavailable"}.`,
        interpretation: "Container byte ranges are still conserved; no numerical result is inferred for an encoding whose source semantics are not implemented.",
        recommendation: "Use a source-pinned decoder for the listed encodings or convert to an assessed dtype before treating payload integrity as complete.",
        relevance: "analysis coverage; evidence completeness",
        evidenceJsonPointers: ["/evidence/static_analysis/tensor_numerical_integrity/limitations"],
      }));
    }
  }
  const quantizationContracts = buildQuantizationContractChecks(analysis);
  const preprocessingConsequence = runtimeEvidence?.preprocessingConsequenceResult || runtimeEvidence?.preprocessing_consequence_atlas || null;
  const input0 = (analysis?.inputs || [])[0];
  const input0Contract = (analysis?.input_contracts || []).find((contract) => contract.tensor_index === input0?.index)
    || (analysis?.input_contracts || [])[0];
  if (input0 && !analysis?.metadata_presence?.documented_preprocessing) {
    const preprocessingEvidencePointer = Object.prototype.hasOwnProperty.call(analysis?.metadata_presence || {}, "documented_preprocessing")
      ? "/evidence/static_analysis/metadata_presence/documented_preprocessing"
      : "/evidence/static_analysis/metadata_presence/preprocessing_contract_status";
    const scalarRange = Number.isFinite(input0Contract?.expected_range_low) && Number.isFinite(input0Contract?.expected_range_high)
      ? `[${Number(input0Contract.expected_range_low).toPrecision(9)}, ${Number(input0Contract.expected_range_high).toPrecision(9)}]`
      : "not emitted";
    findings.push(finding({
      id: "EA-IOC-0001",
      category: "integration",
      title: "Preprocessing contract is not embedded",
      evidence: "OBSERVED",
      priority: "High",
      op: "Input 0",
      observation: `The artifact exposes ${input0.dtype || "unknown"} ${shapeText(input0.shape)} input. Tensor layout is ${input0Contract?.layout || "not determined"} (${input0Contract?.layout_evidence_class || "NOT_ASSESSABLE"}; ${input0Contract?.layout_status || "no structured layout contract"}); tensor numerical status is ${input0Contract?.tensor_numerical_contract_status || "not assessed"} with scalar real range ${scalarRange}. Source decoding, resize policy, interpolation, RGB/BGR order, and normalization remain undocumented.`,
      interpretation: "The model artifact alone does not define the application-side input transformation needed for reproducible runtime or accuracy results.",
      recommendation: "Bind a versioned preprocessing specification and its checksum to the artifact release manifest.",
      relevance: "integration contract; deployment reproducibility",
      evidenceJsonPointers: [
        "/evidence/static_analysis/inputs/0",
        ...(input0Contract ? ["/evidence/static_analysis/input_contracts/0"] : []),
        preprocessingEvidencePointer,
      ],
    }));
  }
  const dynamicShapeCost = analysis?.dynamic_shape_cost_contract;
  const batchOneProjection = deriveTfliteBatchOneProjection(analysis);
  if (Number(dynamicShapeCost?.dynamic_tensor_count || 0) > 0
    && batchOneProjection.status !== "assumption_bound_batch_one") {
    const unresolved = Number(dynamicShapeCost.unresolved_dynamic_compute_op_count || 0);
    const totalBlockers = Number(dynamicShapeCost.total_macs_unresolved_op_count || 0);
    const livenessUnresolved = Number(dynamicShapeCost.liveness?.unresolved_candidate_program_point_count || 0);
    const totalExpression = dynamicShapeCost.total_macs_formula?.expression || dynamicShapeCost.total_macs_formula_status || "not assessed";
    const peakExpression = dynamicShapeCost.liveness?.peak_live_payload_formula?.expression || dynamicShapeCost.liveness?.peak_live_payload_max_formula?.expression || dynamicShapeCost.liveness?.peak_selection_status || "not assessed";
    findings.push(finding({
      id: "EA-DYN-0001",
      category: "integration",
      title: "Dynamic dimensions are not bound to approved deployment scenarios",
      evidence: "DERIVED",
      priority: unresolved > 0 || totalBlockers > 0 || livenessUnresolved > 0 ? "High" : "Medium",
      op: "Dynamic tensor contract",
      observation: `${formatNumber(dynamicShapeCost.dynamic_tensor_count)} dynamic tensor(s) map to ${formatNumber(dynamicShapeCost.symbol_count || 0)} symbol(s). Exact dynamic compute formulas ${formatNumber(dynamicShapeCost.op_formula_count || 0)}/${formatNumber(dynamicShapeCost.dynamic_compute_op_count || 0)}; dynamic-op unresolved ${formatNumber(unresolved)}; total-MAC blockers ${formatNumber(totalBlockers)}. Total MAC expression: ${totalExpression}. Symbolic live-payload peak: ${peakExpression}; unresolved live-set program points ${formatNumber(livenessUnresolved)}. Dimension bounds status: ${dynamicShapeCost.dimension_bounds_status || "not assessed"}.`,
      interpretation: "The emitted polynomials are exact after runtime dimensions are bound, but the artifact contains no approved shape portfolio or numeric bounds. A concrete stored TFLite shape is an example projection, not a worst-case memory or latency claim.",
      recommendation: "Bind a versioned deployment shape portfolio or explicit dimension bounds, evaluate every emitted cost formula for those scenarios, and attach artifact-bound runtime assignment, allocation, and latency captures for the approved extrema.",
      relevance: "capacity planning; dynamic input contract; runtime reproducibility",
      evidenceJsonPointers: FINDING_EVIDENCE_PATHS["EA-DYN-0001"],
    }));
  }
  if (preprocessingConsequence?.status === "assessed" && Number(preprocessingConsequence.output_changed_candidate_count || 0) > 0) {
    findings.push(finding({
      id: "EA-RUN-0003",
      category: "runtime",
      title: "Counterfactual preprocessing contracts diverge at captured model outputs",
      evidence: "MEASURED_SYNTHETIC",
      priority: "Medium",
      op: "Model input to declared outputs",
      observation: `${formatNumber(preprocessingConsequence.candidate_count)} explicit contract input(s) were replayed twice through ${preprocessingConsequence.runtime?.name || "LiteRT.js"} ${preprocessingConsequence.runtime?.version || ""} ${preprocessingConsequence.runtime?.backend || "WASM"}. They collapsed to ${formatNumber(preprocessingConsequence.unique_input_tensor_count)} unique input tensor(s) and ${formatNumber(preprocessingConsequence.unique_output_tensor_set_count)} unique output tensor set(s). ${formatNumber(preprocessingConsequence.output_changed_candidate_count)} candidate(s) changed output values; ${formatNumber(preprocessingConsequence.top1_changed_candidate_count)} changed the raw first-output argmax index. Maximum changed output elements ${formatNumber(preprocessingConsequence.maximum_output_changed_element_count)}; maximum absolute raw output delta ${formatNumber(preprocessingConsequence.maximum_output_absolute_difference)}. Portfolio SHA-256 ${preprocessingConsequence.portfolio_ledger_sha256}.`,
      interpretation: "This establishes output sensitivity for the analyzer-generated counterfactual tensors in the captured browser runtime. It does not identify the production preprocessing path, attach labels to output indices, estimate representative frequency, or establish an accuracy or user-impact change.",
      recommendation: "Replay the versioned production decoder/resize/channel/normalization implementation with representative regression inputs, bind its source and binary hashes, and compare declared output tensors and application decisions against the approved reference pipeline.",
      relevance: "input integration verification; browser-local numerical replay",
    }));
  }
  const output0 = (analysis?.outputs || [])[0];
  if (output0 && !analysis?.metadata_presence?.output_semantics_documented) {
    findings.push(finding({
      id: "EA-OUT-0001",
      category: "integration",
      title: "Output interpretation contract is not embedded",
      evidence: "OBSERVED",
      priority: "Medium",
      op: "Output 0",
      observation: outputSemanticObservation(analysis, output0),
      interpretation: "The graph can expose mathematical tensor structure, but application semantics such as class ordering, label mapping, and consumer-side postprocessing are not established by shape and dtype alone.",
      recommendation: "Attach and hash the output label mapping and application-side interpretation specification.",
      relevance: "integration contract; result interpretation",
    }));
  }
  findings.push(finding({
    id: "EA-LIN-0001",
    category: "lineage",
    title: "Source checkpoint and conversion lineage were not provided",
    evidence: "NOT_ASSESSABLE",
    priority: "High",
    op: "Release manifest",
    observation: "The artifact/report context does not provide source checkpoint SHA-256, export framework/version, converter version, export configuration, quantization/calibration configuration, representative dataset ID, build pipeline ID, software release ID, model requirement ID, or previous released artifact SHA-256.",
    interpretation: "The deployment artifact can be structurally analyzed, but it does not by itself establish traceability from training checkpoint to released runtime artifact.",
    recommendation: "Bind checkpoint-to-artifact lineage in the release manifest and include immutable hashes/IDs for source checkpoint, converter/framework, export options, calibration inputs where applicable, pipeline, software release, requirement, and previous artifact.",
    relevance: "release traceability; configuration management",
  }));
  const bundledRuntimeVersion = bundledRuntimeVersionText(runtimeEvidence);
  const dirtyAnalyzerBuild = metadata.buildSourceState && metadata.buildSourceState !== "clean";
  const targetProfileSha256 = isOnnx ? "" : analysis?.target_profile?.profile_sha256 || "";
  const targetProfileApplicable = String(analysis?.format || "tflite").toLowerCase() === "tflite";
  const missingProvenance = [
    metadata.semanticVersion ? "" : "DeepBOM semantic version",
    metadata.buildCommit ? "" : "analyzer commit",
    dirtyAnalyzerBuild ? "dirty analyzer source state" : "",
    metadata.buildContentSha256 ? "" : "analyzer bundle content SHA-256",
    metadata.rulepackSha256 ? "" : "rulepack SHA-256",
    targetProfileApplicable && !targetProfileSha256 ? "target-profile SHA-256" : "",
  ].filter(Boolean);
  if (missingProvenance.length) {
    findings.push(finding({
      id: "EA-PROV-0001",
      category: "evidence_reproducibility",
      title: isOnnx ? "Analyzer/rulepack immutable identity is incomplete or dirty" : "Analyzer/rulepack/target-profile immutable identity is incomplete or dirty",
      evidence: "OBSERVED",
      priority: "High",
      op: "Report metadata",
      observation: `Reproducibility gap(s): ${missingProvenance.join(", ")}. Analyzer commit ${metadata.buildCommit || "not embedded"}; source state ${metadata.buildSourceState || "not embedded"}; bundle content SHA-256 ${metadata.buildContentSha256 || "not embedded"}; rulepack SHA-256 ${metadata.rulepackSha256 || "not embedded"}; target-profile SHA-256 ${targetProfileApplicable ? targetProfileSha256 || "not embedded" : "not applicable to this artifact-only format assessment"}.`,
      interpretation: `Date/version strings help humans, but a dirty analyzer commit does not identify the executed source; immutable analyzer, bundle-content, and rulepack identifiers${isOnnx ? "" : " plus the selected target-profile identifier"} are needed to reproduce exported evidence exactly.`,
      recommendation: `Build release reports from a clean analyzer tree; for dirty development builds, embed the bundle content SHA-256 and still report the dirty source state explicitly. Embed analyzer and rulepack identifiers${isOnnx ? "" : " plus the selected target-profile identifier"} in every exported report and signed bundle.`,
      relevance: "evidence reproducibility; audit traceability",
      evidenceJsonPointers: targetProfileApplicable ? FINDING_EVIDENCE_PATHS["EA-PROV-0001"] : ["/analyzer_metadata"],
    }));
  }
  if (isOnnx) {
    const domains = analysis?.onnx_domain_analysis || {};
    const externalData = analysis?.onnx_external_data || {};
    const shapeInference = analysis?.onnx_shape_inference || {};
    const typeProtoContract = analysis?.onnx_type_proto_contract || {};
    const sparseTensorContract = analysis?.onnx_sparse_tensor_contract || {};
    const shapeConflicts = shapeInference.declaration_conflicts || [];
    const semanticShapeConflicts = shapeInference.semantic_contract_conflicts || [];
    const conditionalInvalidTensors = (artifactIrValues(analysis) || []).filter((tensor) => (
      (tensor.conditional_shape_contract?.variant_failures || []).some((row) => row?.status === "invalid")
    ));
    const invalidSchemaForms = (shapeInference.schema_form_rows || []).filter((row) => row.status === "fail");
    const invalidOpsetImports = (shapeInference.opset_import_contract?.rows || []).filter((row) => row.status === "fail");
    const shapeScope = shapeInference.shape_scope || {};
    const extendedShape = shapeInference.extended_scope_inference || {};
    const failedFunctionCalls = (extendedShape.function_call_rows || []).filter((row) => row.status === "fail");
    const failedControlFlow = (extendedShape.control_flow_rows || []).filter((row) => row.status === "fail");
    const failedSequenceMaps = (extendedShape.sequence_map_rows || []).filter((row) => row.status === "fail");
    const partialControlFlow = (extendedShape.control_flow_rows || []).filter((row) => row.status === "partial");
    const partialSequenceMaps = (extendedShape.sequence_map_rows || []).filter((row) => row.status === "partial");
    const partialRecursiveScopes = (extendedShape.scope_rows || []).filter((row) => row.status === "partial");
    const invalidTypeProtoRows = typeProtoContract.invalid_rows || [];
    const invalidSparseTensorRows = sparseTensorContract.invalid_rows || [];
    const invalidContainerRows = shapeInference.container_value_inference?.failed_rows || [];
    const mlValueInference = shapeInference.ml_value_inference || {};
    const invalidMlValueRows = mlValueInference.failed_rows || [];
    const partialMlValueRows = mlValueInference.partial_rows || [];
    const duplicateMlKeyRows = (mlValueInference.rows || []).filter((row) => Number(row.duplicate_key_count || 0) > 0);
    const duplicateMlVocabularyRows = (mlValueInference.rows || []).filter((row) => row.op_name === "DictVectorizer"
      && Number(row.duplicate_vocabulary_count || 0) > 0);
    const duplicateMlCategoryRows = (mlValueInference.rows || []).filter((row) => Number(row.active_duplicate_key_count || 0) > 0);
    const truncatingFeatureVectorizerRows = (mlValueInference.rows || []).filter((row) => Number(row.exact_truncated_feature_count_per_batch || 0) > 0);
    const riskyBinarizerRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).some((code) => code === "binarizer_non_finite_threshold" || code === "binarizer_static_input_contains_non_finite_or_unsafe_value"));
    const ortUnsupportedBinarizerRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("binarizer_dtype_unsupported_by_pinned_ort_cpu"));
    const normalizerOverflowRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("normalizer_signed_integer_abs_or_square_overflow"));
    const normalizerNegativeMaxRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("normalizer_negative_signed_max_divisor"));
    const normalizerIntegerRoundingRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("normalizer_integer_to_float32_precision_loss"));
    const normalizerNonfiniteRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).some((code) => code === "normalizer_non_finite_float32_projection" || code === "normalizer_static_input_contains_non_finite_or_unsafe_value"));
    const scalerInvalidContractRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("scaler_pinned_ort_attribute_or_shape_contract_invalid"));
    const scalerIntegerRoundingRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("scaler_integer_to_float32_precision_loss"));
    const scalerNonfiniteRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("scaler_non_finite_parameter_input_or_output"));
    const imputerInvalidContractRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("imputer_pinned_ort_attribute_or_shape_contract_invalid"));
    const imputerScalarFirstRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("imputer_attribute_length_outside_onnx_one_or_feature_count"));
    const imputerDtypeGapRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("imputer_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const imputerNonfiniteRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("imputer_non_finite_imputed_or_output"));
    const oneHotInvalidContractRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_pinned_ort_attribute_contract_invalid"));
    const oneHotDuplicateRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_duplicate_categories_last_write_wins"));
    const oneHotUnknownAllZeroRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_unknown_categories_all_zero_encoding"));
    const oneHotGuaranteedFailureRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_unknown_category_guaranteed_runtime_failure"));
    const oneHotDtypeGapRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const oneHotNoncanonicalZerosRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_noncanonical_zeros_boolean"));
    const oneHotInvalidCastRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("onehot_numeric_to_int64_cast_not_representable"));
    const labelEncoderDtypeGapRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel"));
    const labelEncoderDuplicateRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).some((code) => [
      "label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict",
      "label_encoder_v1_duplicate_class_runtime_last_index",
    ].includes(code)));
    const labelEncoderNanRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched"));
    const labelEncoderDefaultRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("label_encoder_artifact_known_default_path_reached"));
    const labelEncoderNonfiniteRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("label_encoder_non_finite_mapping_state"));
    const labelEncoderRuntimeInvalidRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("label_encoder_pinned_ort_runtime_contract_invalid"));
    const linearRuntimeInvalidRows = (mlValueInference.rows || []).filter((row) => ["LinearClassifier", "LinearRegressor"].includes(row.op_name)
      && (row.risk_codes || []).some((code) => code.endsWith("pinned_ort_runtime_contract_invalid")));
    const linearIgnoredParameterRows = (mlValueInference.rows || []).filter((row) => Number(row.linear_unused_coefficient_count || 0) > 0
      || Number(row.linear_ignored_intercept_count || 0) > 0);
    const linearRegressorDtypeGapRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const linearPostTransformRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).some((code) => [
      "linear_classifier_single_score_post_transform_noop",
      "linear_classifier_binary_probit_second_score_unwritten",
      "linear_classifier_binary_post_transform_ignored_for_complement_expansion",
      "linear_regressor_single_target_post_transform_noop",
      "linear_regressor_probit_may_emit_non_finite",
    ].includes(code)));
    const linearMultiClassRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("linear_classifier_multi_class_nonzero_ignored_by_pinned_ort"));
    const linearDuplicateLabelRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).includes("linear_classifier_duplicate_labels_ambiguous_output_semantics"));
    const linearNumericalRiskRows = (mlValueInference.rows || []).filter((row) => (row.risk_codes || []).some((code) => [
      "linear_classifier_non_finite_parameter_or_reference_score",
      "linear_classifier_reference_decision_boundary",
      "linear_regressor_non_finite_parameter_or_reference_score",
    ].includes(code)));
    const svmRows = (mlValueInference.rows || []).filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name));
    const svmRuntimeInvalidRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_pinned_ort_runtime_contract_invalid",
      "svm_regressor_pinned_ort_runtime_contract_invalid",
    ].includes(code)));
    const svmScoreWidthMismatchRows = svmRows.filter((row) => (row.risk_codes || [])
      .includes("svm_classifier_onnx_vs_pinned_ort_score_width_mismatch"));
    const svmDtypeGapRows = svmRows.filter((row) => (row.risk_codes || [])
      .includes("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const svmIgnoredTransformRows = svmRows.filter((row) => (row.risk_codes || [])
      .includes("svm_regressor_post_transform_ignored_by_pinned_ort"));
    const svmIgnoredOrForcedRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_serialized_parameters_ignored_by_pinned_ort",
      "svm_regressor_serialized_parameters_ignored_by_pinned_ort",
      "svm_classifier_linear_mode_forces_linear_kernel",
      "svm_regressor_linear_mode_forces_linear_kernel",
    ].includes(code)));
    const svmNumericalRiskRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_non_finite_parameter_or_reference_score",
      "svm_classifier_reference_decision_boundary",
      "svm_regressor_non_finite_parameter_or_reference_score",
      "svm_regressor_reference_decision_boundary",
    ].includes(code)));
    const svmSemanticHazardRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_duplicate_labels_ambiguous_output_semantics",
      "svm_classifier_probability_scores_receive_additional_post_transform",
      "svm_classifier_binary_probit_second_score_unwritten",
      "svm_classifier_binary_post_transform_uses_complement_expansion",
      "svm_regressor_noncanonical_one_class_flag",
    ].includes(code)));
    const treeRows = (mlValueInference.rows || []).filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name));
    const treeRuntimeInvalidRows = treeRows.filter((row) => row.tree_pinned_ort_contract_status === "fail"
      && row.tree_pinned_cpu_dtype_gap !== true);
    const treeDtypeGapRows = treeRows.filter((row) => row.tree_pinned_cpu_dtype_gap === true);
    const treeDeprecatedRows = treeRows.filter((row) => row.tree_deprecated_operator === true);
    const treeDeadOrNonTreeStateRows = treeRows.filter((row) => (row.risk_codes || []).some((code) => [
      "tree_ensemble_nonleaf_weights_ignored_by_pinned_ort",
      "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort",
      "tree_ensemble_shared_subtree_not_strict_tree",
      "tree_ensemble_v5_shared_subtree_not_strict_tree",
      "tree_ensemble_v5_unreachable_serialized_nodes_or_leaves",
    ].includes(code)));
    const treeMembershipRows = treeRows.filter((row) => (row.risk_codes || []).some((code) => [
      "tree_ensemble_v5_duplicate_membership_values",
      "tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser",
    ].includes(code)));
    const treeSemanticHazardRows = treeRows.filter((row) => (row.risk_codes || []).some((code) => [
      "tree_classifier_duplicate_labels_ambiguous_output_semantics",
      "tree_classifier_binary_post_transform_leaves_score_unwritten",
      "tree_regressor_single_target_post_transform_noop",
      "tree_ensemble_v5_single_target_post_transform_noop",
      "tree_classifier_binary_single_base_value_semantics_underspecified",
      "tree_classifier_pinned_ort_binary_label_index_semantics",
    ].includes(code)));
    const treeNumericalRiskRows = treeRows.filter((row) => (row.risk_codes || []).some((code) => [
      "tree_ensemble_non_finite_parameter_or_reference_score",
      "tree_ensemble_reference_decision_boundary",
    ].includes(code)));
    const tfidfInference = shapeInference.tfidf_vectorizer_inference || {};
    const tfidfRows = tfidfInference.rows || [];
    const tfidfInvalidRows = tfidfInference.failed_rows || [];
    const tfidfWeightSemanticRows = tfidfRows.filter((row) => (row.risk_codes || []).includes("tfidf_weight_coordinate_semantics_divergence"));
    const tfidfReferenceDivergenceRows = tfidfRows.filter((row) => Number(row.exact_ort_reference_divergent_output_count || 0) > 0);
    const tfidfNoncanonicalRows = tfidfRows.filter((row) => (row.risk_codes || []).some((code) => [
      "tfidf_ngram_counts_ignore_pool_prefix",
      "tfidf_duplicate_ngram_outside_active_length_range",
      "tfidf_multiple_ngrams_share_output_coordinate",
    ].includes(code)));
    const tfidfBoundedResidualRows = (tfidfInference.partial_rows || []).filter((row) => [
      "not_assessed_output_element_limit", "not_assessed_work_limit",
    ].includes(row.static_execution_status));
    if (invalidTypeProtoRows.length) {
      const examples = invalidTypeProtoRows.slice(0, 8).map((row) => `${row.scope} ${row.role} ${row.value_name}: ${(row.reason_codes || []).join("/") || "invalid TypeProto"}`);
      findings.push(finding({
        id: "EA-ONX-0010",
        category: "integrity",
        title: "ONNX TypeProto declaration violates the pinned IR contract",
        evidence: "SOURCE_PINNED_AND_OBSERVED",
        priority: "High",
        op: `${formatNumber(invalidTypeProtoRows.length)} invalid type declaration(s)`,
        observation: `${examples.join("; ")}${invalidTypeProtoRows.length > examples.length ? `; plus ${formatNumber(invalidTypeProtoRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A required ValueInfo name/type/shape, recursive TypeProto oneof, tensor element type, dimension oneof, sequence/optional child, or map key/value contract is malformed under the pinned ONNX IR. Dense tensor shape, MAC, payload, liveness, browser input, and provider conclusions are suppressed where the declared value kind is not a valid dense tensor contract.",
        recommendation: "Reject the artifact for release, run the same pinned ONNX checker, repair the exporter or serialized TypeProto, and require zero TypeProto contract failures before runtime qualification.",
        relevance: "artifact integrity; value type system; downstream metric validity",
      }));
    }
    if (invalidSparseTensorRows.length) {
      const examples = invalidSparseTensorRows.slice(0, 8).map((row) => `${row.scope} ${row.sparse_tensor_name || "unnamed"}: ${(row.reason_codes || []).join("/") || "invalid SparseTensorProto"}`);
      findings.push(finding({
        id: "EA-ONX-0011",
        category: "integrity",
        title: "ONNX SparseTensorProto storage violates the pinned IR contract",
        evidence: "SOURCE_PINNED_AND_OBSERVED",
        priority: "High",
        op: `${formatNumber(invalidSparseTensorRows.length)} invalid sparse tensor record(s)`,
        observation: `${examples.join("; ")}${invalidSparseTensorRows.length > examples.length ? `; plus ${formatNumber(invalidSparseTensorRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The values/indices/dense-shape/NNZ tuple or exact decoded index order, uniqueness, or bounds violates the pinned SparseTensorProto contract. Logical dense size and operator arithmetic must not be trusted until the sparse storage is repaired.",
        recommendation: "Reject the artifact for release, regenerate values and INT64 indices in strictly ascending unique order within the declared dense shape, verify every external component, and require a zero-failure sparse contract report.",
        relevance: "artifact integrity; sparse initializer storage; downstream metric validity",
      }));
    }
    if (invalidContainerRows.length) {
      const examples = invalidContainerRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${(row.reason_codes || []).join("/") || "container value contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0012",
        category: "integrity",
        title: "ONNX Sequence or Optional value contract is deterministically invalid",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(invalidContainerRows.length)} invalid container-value node(s)`,
        observation: `${examples.join("; ")}${invalidContainerRows.length > examples.length ? `; plus ${formatNumber(invalidContainerRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned ONNX Sequence/Optional type inference proves an element-type mismatch, scalar/index/range defect, split inconsistency, concat incompatibility, or access to a provably empty optional. The affected graph path is invalid independently of execution-provider choice.",
        recommendation: "Reject the artifact for release, repair the exporter or graph constants, run the pinned ONNX checker/reference inferencer, and require zero container-value contract failures before runtime qualification.",
        relevance: "artifact integrity; ONNX value type system; runtime validity",
      }));
    }
    if (invalidMlValueRows.length) {
      const examples = invalidMlValueRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name || "ONNX-ML"}: ${(row.reason_codes || []).join("/") || "value-contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0013",
        category: "integrity",
        title: "ONNX-ML value contract is deterministically invalid",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(invalidMlValueRows.length)} invalid supported ONNX-ML value-contract node(s)`,
        observation: `${examples.join("; ")}${invalidMlValueRows.length > examples.length ? `; plus ${formatNumber(invalidMlValueRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned ai.onnx.ml-1 semantics prove an input type/rank defect, invalid Binarizer threshold, invalid Normalizer dtype/rank/mode, invalid CastMap mode or bound, missing/conflicting vocabulary, key/vocabulary mismatch, disallowed map value type, CategoryMapper category-array defect, FeatureVectorizer variadic-width/batch defect, ArrayFeatureExtractor empty or out-of-bounds index contract, or ZipMap feature-to-key mismatch. The affected output contract is invalid independently of execution-provider choice.",
        recommendation: "Reject the artifact for release, repair the exporter attributes and input TypeProto, run the pinned ONNX checker/reference inferencer, and require zero ONNX-ML value-contract failures before runtime qualification.",
        relevance: "artifact integrity; tensor/map value typing; classifier and feature-vector contracts; ONNX-ML runtime validity",
      }));
    }
    if (duplicateMlKeyRows.length) {
      const examples = duplicateMlKeyRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.duplicate_key_count || 0)} duplicate occurrence(s) among ${formatNumber(row.class_key_count || 0)} key(s)`);
      findings.push(finding({
        id: "EA-ONX-0014",
        category: "integration_verification",
        title: "ONNX-ML ZipMap class labels contain duplicate map keys",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(duplicateMlKeyRows.length)} affected ZipMap node(s); ${formatNumber(mlValueInference.duplicate_class_key_count || 0)} duplicate occurrence(s)`,
        observation: `${examples.join("; ")}${duplicateMlKeyRows.length > examples.length ? `; plus ${formatNumber(duplicateMlKeyRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A map cannot preserve separate score columns under the same key. The artifact's class-label list therefore cannot provide a one-to-one externally consumable mapping for every score column, even though duplicate labels are not promoted here to an invented OpSchema rejection.",
        recommendation: "Regenerate the artifact with unique class labels, verify class-key cardinality against the final score dimension, and test the consumer's decoded label-to-score contract before release.",
        relevance: "classifier correctness; output information loss; application integration",
      }));
    }
    if (duplicateMlVocabularyRows.length) {
      const examples = duplicateMlVocabularyRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.duplicate_vocabulary_count || 0)} duplicate occurrence(s) among ${formatNumber(row.vocabulary_count || 0)} vocabulary entries`);
      findings.push(finding({
        id: "EA-ONX-0015",
        category: "integration_verification",
        title: "ONNX-ML DictVectorizer vocabulary repeats output columns",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(duplicateMlVocabularyRows.length)} affected DictVectorizer node(s); ${formatNumber(duplicateMlVocabularyRows.reduce((sum, row) => sum + Number(row.duplicate_vocabulary_count || 0), 0))} duplicate occurrence(s)`,
        observation: `${examples.join("; ")}${duplicateMlVocabularyRows.length > examples.length ? `; plus ${formatNumber(duplicateMlVocabularyRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT CPU implementation intentionally accepts duplicate vocabulary entries and emits one output column per entry. Repeated entries therefore copy the same map value into multiple columns; this is valid runtime behavior, but it changes feature dimensionality and can silently duplicate a downstream feature.",
        recommendation: "Confirm that repeated feature columns are intentional, bind vocabulary order and cardinality to the downstream model contract, and regenerate with unique vocabulary entries when duplicated features are not required.",
        relevance: "feature-vector correctness; output column identity; application integration",
      }));
    }
    if (duplicateMlCategoryRows.length) {
      const examples = duplicateMlCategoryRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.mapping_direction || "UNRESOLVED"}: ${formatNumber(row.active_duplicate_key_count || 0)} active duplicate occurrence(s) among ${formatNumber(row.category_pair_count || 0)} pair(s)`);
      findings.push(finding({
        id: "EA-ONX-0016",
        category: "integration_verification",
        title: "ONNX-ML CategoryMapper active keys overwrite earlier mappings",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(duplicateMlCategoryRows.length)} affected CategoryMapper node(s); ${formatNumber(mlValueInference.duplicate_category_active_key_count || 0)} active duplicate occurrence(s)`,
        observation: `${examples.join("; ")}${duplicateMlCategoryRows.length > examples.length ? `; plus ${formatNumber(duplicateMlCategoryRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT CPU constructor inserts category pairs in source order into an unordered map, so a repeated active-direction key replaces its earlier value. The artifact remains executable, but at least one declared category pair is unreachable and the externally visible mapping is not one-to-one with the attribute arrays.",
        recommendation: "Regenerate the artifact with unique keys for the active input direction, verify the retained last value against the application label contract, and require zero active duplicate categories before release.",
        relevance: "categorical mapping correctness; label identity; application integration",
      }));
    }
    if (truncatingFeatureVectorizerRows.length) {
      const examples = truncatingFeatureVectorizerRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.exact_truncated_feature_count_per_batch || 0)} feature value(s) discarded per batch row across ${formatNumber(row.truncated_input_count || 0)} input(s)`);
      findings.push(finding({
        id: "EA-ONX-0017",
        category: "integration_verification",
        title: "ONNX-ML FeatureVectorizer deterministically truncates input features",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(truncatingFeatureVectorizerRows.length)} affected FeatureVectorizer node(s); ${formatNumber(mlValueInference.exact_feature_vectorizer_truncated_feature_count_per_batch || 0)} discarded feature value(s) per batch row`,
        observation: `${examples.join("; ")}${truncatingFeatureVectorizerRows.length > examples.length ? `; plus ${formatNumber(truncatingFeatureVectorizerRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT CPU implementation copies only min(actual row width, inputdimensions[i]) values from each input. The remaining source values are not represented in the output. This is executable behavior rather than an OpSchema failure, but it is deterministic information loss when the configured width is smaller than the artifact-known input row width.",
        recommendation: "Confirm that every discarded feature is intentional, bind inputdimensions and source tensor shapes to the downstream feature contract, and regenerate the graph with matching widths when truncation is not required.",
        relevance: "feature-vector correctness; deterministic information loss; downstream input identity",
      }));
    }
    if (riskyBinarizerRows.length) {
      const examples = riskyBinarizerRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} threshold ${row.threshold_value_text || "unresolved"}: ${(row.risk_codes || []).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0018",
        category: "integration_verification",
        title: "ONNX-ML Binarizer contains a non-finite numerical contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(riskyBinarizerRows.length)} affected Binarizer node(s)`,
        observation: `${examples.join("; ")}${riskyBinarizerRows.length > examples.length ? `; plus ${formatNumber(riskyBinarizerRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A NaN or infinite threshold produces a degenerate comparison contract, while the pinned ORT CPU Binarizer rejects a non-finite FLOAT32 input at execution. The ONNX artifact may remain schema-valid, but its output can collapse to a constant code or fail on the pinned CPU path.",
        recommendation: "Regenerate the model with a finite threshold and finite initializer values, verify the exact zero/one counts in the Engineering Report, and execute a bound ORT profile before release.",
        relevance: "numerical integrity; deterministic output collapse; runtime compatibility",
      }));
    }
    if (ortUnsupportedBinarizerRows.length) {
      const examples = ortUnsupportedBinarizerRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0019",
        category: "runtime_compatibility",
        title: "ONNX-ML Binarizer dtype has no pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(ortUnsupportedBinarizerRows.length)} affected Binarizer node(s)`,
        observation: `${examples.join("; ")}${ortUnsupportedBinarizerRows.length > examples.length ? `; plus ${formatNumber(ortUnsupportedBinarizerRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The artifact is valid under the pinned ONNX Binarizer-1 schema, but the pinned ORT CPU provider registers this operator only for FLOAT32. A FLOAT64, INT32, or INT64 Binarizer therefore has a deterministic schema/runtime support gap on that execution path.",
        recommendation: "Convert the Binarizer input/output contract to FLOAT32 or bind and capture an execution provider that demonstrably supports the artifact dtype; require successful provider assignment and execution before release.",
        relevance: "runtime compatibility; execution-provider assignment; deployment validity",
      }));
    }
    if (normalizerOverflowRows.length) {
      const examples = normalizerOverflowRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"} ${row.normalizer_mode || "UNKNOWN"}: ${formatNumber(row.normalizer_signed_overflow_value_count || 0)} value(s)`);
      findings.push(finding({
        id: "EA-ONX-0020",
        category: "integrity",
        title: "ONNX-ML Normalizer reaches signed integer overflow in the pinned CPU arithmetic",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(normalizerOverflowRows.length)} affected Normalizer node(s); ${formatNumber(mlValueInference.exact_normalizer_signed_overflow_value_count || 0)} overflowing value(s)`,
        observation: `${examples.join("; ")}${normalizerOverflowRows.length > examples.length ? `; plus ${formatNumber(normalizerOverflowRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT CPU implementation applies signed abs for L1 or signed x*x for L2 before casting to FLOAT32. INT32/INT64 minimum abs and magnitudes above the exact square limit therefore enter C++ signed-overflow territory; DEEPBOM suppresses a fabricated output instead of assigning deterministic numerical semantics to undefined arithmetic.",
        recommendation: "Reject this CPU execution contract, insert a safe FLOAT32 cast before Normalizer or regenerate bounded integer inputs, and verify the corrected graph against a pinned ORT session before release.",
        relevance: "numerical integrity; undefined signed arithmetic; runtime validity",
      }));
    }
    if (normalizerNegativeMaxRows.length) {
      const examples = normalizerNegativeMaxRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.normalizer_negative_max_divisor_row_count || 0)} all-negative row(s); divisors ${(row.normalizer_divisor_preview || []).join("/") || "not previewed"}`);
      findings.push(finding({
        id: "EA-ONX-0021",
        category: "integration_verification",
        title: "ONNX-ML Normalizer MAX uses a negative signed maximum",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(normalizerNegativeMaxRows.length)} affected Normalizer node(s); ${formatNumber(mlValueInference.exact_normalizer_negative_max_divisor_row_count || 0)} row(s)`,
        observation: `${examples.join("; ")}${normalizerNegativeMaxRows.length > examples.length ? `; plus ${formatNumber(normalizerNegativeMaxRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned operator divides by max(x), not max(abs(x)). An all-negative row therefore uses a negative divisor, flips output signs, and sets the least-negative input to +1. This is source-defined behavior but is frequently mistaken for max-absolute normalization.",
        recommendation: "Confirm signed-MAX semantics against the training/export pipeline; use L1/L2 or an explicit absolute-value reduction when max-absolute normalization was intended, then bind a reference output test to the artifact.",
        relevance: "feature preprocessing semantics; sign behavior; application contract",
      }));
    }
    if (normalizerIntegerRoundingRows.length) {
      const examples = normalizerIntegerRoundingRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}: ${formatNumber(row.normalizer_integer_float32_rounding_count || 0)} changed value(s)`);
      findings.push(finding({
        id: "EA-ONX-0022",
        category: "integration_verification",
        title: "ONNX-ML Normalizer integer inputs lose precision at the required FLOAT32 projection",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(normalizerIntegerRoundingRows.length)} affected Normalizer node(s); ${formatNumber(mlValueInference.exact_normalizer_integer_float32_rounding_count || 0)} changed integer value(s)`,
        observation: `${examples.join("; ")}${normalizerIntegerRoundingRows.length > examples.length ? `; plus ${formatNumber(normalizerIntegerRoundingRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Normalizer-1 always emits FLOAT32. Artifact-known INT32/INT64 values outside exact FLOAT32 integer spacing are rounded before or during the pinned row arithmetic, so distinct integer inputs can become numerically indistinguishable without violating the ONNX schema.",
        recommendation: "Quantify whether the changed integers are semantically distinct features, rescale or cast upstream under an explicit contract, and compare pinned ORT outputs against the exporter reference before release.",
        relevance: "numerical precision; feature identity; reference-output verification",
      }));
    }
    if (normalizerNonfiniteRows.length) {
      const examples = normalizerNonfiniteRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.normalizer_mode || "UNKNOWN"}: ${row.normalizer_non_finite_output_count == null ? "input not safely decodable" : `${formatNumber(row.normalizer_non_finite_output_count)} non-finite output(s)`}`);
      findings.push(finding({
        id: "EA-ONX-0023",
        category: "integrity",
        title: "ONNX-ML Normalizer has a statically non-finite numerical path",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(normalizerNonfiniteRows.length)} affected Normalizer node(s); ${formatNumber(mlValueInference.exact_normalizer_non_finite_output_count || 0)} proven non-finite output(s)`,
        observation: `${examples.join("; ")}${normalizerNonfiniteRows.length > examples.length ? `; plus ${formatNumber(normalizerNonfiniteRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Finite source values can still overflow the pinned FLOAT32 square/sum path and produce Infinity/Infinity or another non-finite projection; explicitly non-finite source constants are also unsafe. Such outputs cannot be promoted as valid static tensors or stable downstream features.",
        recommendation: "Reject the affected numerical path, rescale or sanitize the input before normalization, and require finite outputs from the pinned ORT native fixture and representative runtime inputs.",
        relevance: "numerical integrity; non-finite propagation; downstream runtime validity",
      }));
    }
    if (scalerInvalidContractRows.length) {
      const examples = scalerInvalidContractRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.scaler_parameter_contract_reason || "invalid pinned-runtime contract"}`);
      findings.push(finding({
        id: "EA-ONX-0024",
        category: "runtime_compatibility",
        title: "ONNX-ML Scaler is valid by schema but invalid for the pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(scalerInvalidContractRows.length)} affected Scaler node(s)`,
        observation: `${examples.join("; ")}${scalerInvalidContractRows.length > examples.length ? `; plus ${formatNumber(scalerInvalidContractRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Scaler-1 marks scale and offset optional in the ONNX schema, but the pinned ORT CPU implementation requires both arrays to be present, nonempty, equal in length, and either scalar or equal to dimension 0 for rank 1 and dimension 1 otherwise; rank-0 input is rejected. This is a deterministic schema/runtime compatibility gap, not an execution-provider prediction.",
        recommendation: "Emit explicit scale and offset arrays satisfying the pinned feature-stride contract, reject rank-0 input, and require successful construction plus reference-output execution in the pinned ORT session before release.",
        relevance: "runtime compatibility; attribute contract; deployment validity",
      }));
    }
    if (scalerIntegerRoundingRows.length) {
      const examples = scalerIntegerRoundingRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}: ${formatNumber(row.scaler_integer_float32_rounding_count || 0)} changed value(s)`);
      findings.push(finding({
        id: "EA-ONX-0025",
        category: "integration_verification",
        title: "ONNX-ML Scaler integer inputs lose precision at FLOAT32 projection",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(scalerIntegerRoundingRows.length)} affected Scaler node(s); ${formatNumber(mlValueInference.exact_scaler_integer_float32_rounding_count || 0)} changed integer value(s)`,
        observation: `${examples.join("; ")}${scalerIntegerRoundingRows.length > examples.length ? `; plus ${formatNumber(scalerIntegerRoundingRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned Scaler CPU kernel emits FLOAT32. Artifact-known INT32 or INT64 inputs outside exact FLOAT32 integer spacing are rounded before the source-ordered offset-then-scale arithmetic, so distinct feature values may become identical without violating the ONNX schema.",
        recommendation: "Verify that changed integers are not semantically distinct, rescale or cast upstream under an explicit contract, and compare the exact pinned ORT outputs with the exporter reference.",
        relevance: "numerical precision; feature identity; reference-output verification",
      }));
    }
    if (scalerNonfiniteRows.length) {
      const examples = scalerNonfiniteRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.scaler_non_finite_parameter_count || 0)} non-finite parameter(s), ${row.scaler_non_finite_output_count == null ? "output unassessed" : `${formatNumber(row.scaler_non_finite_output_count)} non-finite output(s)`}`);
      findings.push(finding({
        id: "EA-ONX-0026",
        category: "integrity",
        title: "ONNX-ML Scaler has a statically non-finite affine path",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(scalerNonfiniteRows.length)} affected Scaler node(s); ${formatNumber(mlValueInference.exact_scaler_non_finite_parameter_count || 0)} non-finite parameter(s); ${formatNumber(mlValueInference.exact_scaler_non_finite_output_count || 0)} proven non-finite output(s)`,
        observation: `${examples.join("; ")}${scalerNonfiniteRows.length > examples.length ? `; plus ${formatNumber(scalerNonfiniteRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A NaN or infinite scale/offset, an unsafe source constant, or finite affine arithmetic that overflows FLOAT32 can create a non-finite output. DEEPBOM preserves the exact canonical parameter text but does not propagate a non-finite result as a valid downstream static tensor.",
        recommendation: "Reject the affected numerical path, regenerate finite parameters or bounded inputs, and require finite reference outputs from the pinned ORT native fixture and representative runtime data.",
        relevance: "numerical integrity; non-finite propagation; downstream runtime validity",
      }));
    }
    if (imputerInvalidContractRows.length) {
      const examples = imputerInvalidContractRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.imputer_parameter_contract_reason || "invalid pinned-runtime contract"}`);
      findings.push(finding({
        id: "EA-ONX-0027",
        category: "runtime_compatibility",
        title: "ONNX-ML Imputer contract is invalid for the pinned ORT CPU implementation",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(imputerInvalidContractRows.length)} affected Imputer node(s)`,
        observation: `${examples.join("; ")}${imputerInvalidContractRows.length > examples.length ? `; plus ${formatNumber(imputerInvalidContractRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT Imputer constructor and compute path require exactly one nonempty imputed-value list whose numeric family matches the input, and reject rank-0 tensors. A malformed or mismatched contract cannot be treated as a merely unknown runtime value path.",
        recommendation: "Emit one nonempty imputed-value list matching the input dtype family, use rank-1 or higher tensor input, and require successful construction plus exact reference-output execution with the pinned ORT package.",
        relevance: "runtime compatibility; missing-value contract; deployment validity",
      }));
    }
    if (imputerScalarFirstRows.length) {
      const examples = imputerScalarFirstRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.imputer_imputed_value_count || 0)} values for feature stride ${row.imputer_feature_stride == null ? "unresolved" : formatNumber(row.imputer_feature_stride)}; ${formatNumber(row.imputer_ignored_imputed_value_count || 0)} ignored`);
      findings.push(finding({
        id: "EA-ONX-0028",
        category: "integration_verification",
        title: "ONNX-ML Imputer silently ignores trailing imputed values in the pinned ORT kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(imputerScalarFirstRows.length)} affected Imputer node(s); ${formatNumber(mlValueInference.exact_imputer_ignored_imputed_value_count || 0)} ignored trailing value(s)`,
        observation: `${examples.join("; ")}${imputerScalarFirstRows.length > examples.length ? `; plus ${formatNumber(imputerScalarFirstRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "ONNX prose permits an imputed-value length of one or the feature count F. The pinned ORT CPU implementation accepts other nonempty lengths but uses only element zero, so the artifact can run while silently discarding configured values. This is source-observed runtime behavior, not a provider prediction.",
        recommendation: "Regenerate the artifact with exactly one imputed value or exactly F values, reject scalar-first fallback in release policy, and compare the pinned ORT output against exporter expectations.",
        relevance: "feature semantics; silent configuration loss; runtime portability",
      }));
    }
    if (imputerDtypeGapRows.length) {
      const examples = imputerDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0029",
        category: "runtime_compatibility",
        title: "ONNX-ML Imputer dtype is schema-valid but lacks a pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(imputerDtypeGapRows.length)} affected Imputer node(s)`,
        observation: `${examples.join("; ")}${imputerDtypeGapRows.length > examples.length ? `; plus ${formatNumber(imputerDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Imputer-1 permits FLOAT64 and INT32 in the ONNX schema, but the pinned ORT CPU registration provides FLOAT32 and INT64 kernels only. A schema-valid artifact can therefore fail session initialization on the pinned CPU execution path.",
        recommendation: "Convert the Imputer path to FLOAT32 or INT64 under an explicit numerical contract, or bind runtime evidence from a selected provider that implements the declared dtype before release.",
        relevance: "schema/runtime gap; execution-provider compatibility; session initialization",
      }));
    }
    if (imputerNonfiniteRows.length) {
      const examples = imputerNonfiniteRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.imputer_non_finite_imputed_value_count || 0)} non-finite imputed value(s), ${row.imputer_non_finite_output_count == null ? "output unassessed" : `${formatNumber(row.imputer_non_finite_output_count)} non-finite output(s)`}`);
      findings.push(finding({
        id: "EA-ONX-0030",
        category: "integrity",
        title: "ONNX-ML Imputer has a statically non-finite output path",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(imputerNonfiniteRows.length)} affected Imputer node(s); ${formatNumber(mlValueInference.exact_imputer_non_finite_imputed_value_count || 0)} non-finite imputed value(s); ${formatNumber(mlValueInference.exact_imputer_non_finite_output_count || 0)} proven non-finite output(s)`,
        observation: `${examples.join("; ")}${imputerNonfiniteRows.length > examples.length ? `; plus ${formatNumber(imputerNonfiniteRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A non-finite imputed value or retained non-finite source value can leave NaN or infinity in the exact output. A NaN replacement marker is valid and is not itself flagged when it produces finite replacements; the finding is tied to the resulting imputed/output path.",
        recommendation: "Use finite imputed values, define whether retained non-finite inputs are permitted, and require finite pinned-runtime outputs before downstream feature consumption.",
        relevance: "numerical integrity; missing-value sanitization; downstream validity",
      }));
    }
    if (oneHotInvalidContractRows.length) {
      const examples = oneHotInvalidContractRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.onehot_parameter_contract_reason || "invalid category contract"}`);
      findings.push(finding({
        id: "EA-ONX-0031",
        category: "runtime_compatibility",
        title: "ONNX-ML OneHotEncoder category contract is invalid for the pinned ORT CPU implementation",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(oneHotInvalidContractRows.length)} affected OneHotEncoder node(s)`,
        observation: `${examples.join("; ")}${oneHotInvalidContractRows.length > examples.length ? `; plus ${formatNumber(oneHotInvalidContractRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned constructor requires exactly one nonempty cats_int64s or cats_strings list, and the active list must match the input family. A missing, dual, or mismatched vocabulary is a deterministic construction defect rather than an unknown data-dependent path.",
        recommendation: "Emit exactly one nonempty vocabulary attribute matching the input dtype family, then require successful construction and exact output parity with the pinned ORT CPU fixture.",
        relevance: "runtime compatibility; categorical feature contract; session initialization",
      }));
    }
    if (oneHotDuplicateRows.length) {
      const examples = oneHotDuplicateRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.onehot_duplicate_category_count || 0)} duplicate(s); unreachable column(s) ${(row.onehot_unreachable_duplicate_column_indices || []).join(", ") || "none"}`);
      findings.push(finding({
        id: "EA-ONX-0032",
        category: "integration_verification",
        title: "ONNX-ML OneHotEncoder vocabulary contains unreachable duplicate columns",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(oneHotDuplicateRows.length)} affected OneHotEncoder node(s); ${formatNumber(mlValueInference.exact_onehot_unreachable_duplicate_column_count || 0)} unreachable column(s)`,
        observation: `${examples.join("; ")}${oneHotDuplicateRows.length > examples.length ? `; plus ${formatNumber(oneHotDuplicateRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT lookup overwrites earlier indices when a category repeats. The final duplicate owns the active one-hot column, leaving every earlier duplicate column permanently zero and making vocabulary width larger than the reachable encoding space.",
        recommendation: "Deduplicate the vocabulary while preserving the intended category order, retrain or remap downstream feature indices as needed, and verify exact encoded vectors before release.",
        relevance: "categorical feature identity; unreachable dimensions; downstream coefficient alignment",
      }));
    }
    if (oneHotUnknownAllZeroRows.length) {
      const examples = oneHotUnknownAllZeroRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.onehot_exact_unknown_input_count || 0)} unknown value(s) [${(row.onehot_unknown_input_preview || []).join(", ") || "none"}]`);
      findings.push(finding({
        id: "EA-ONX-0033",
        category: "input_contract",
        title: "ONNX-ML OneHotEncoder maps artifact-known unknown categories to all-zero slices",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(oneHotUnknownAllZeroRows.length)} affected OneHotEncoder node(s); ${formatNumber(mlValueInference.exact_onehot_unknown_input_count || 0)} unknown input value(s)`,
        observation: `${examples.join("; ")}${oneHotUnknownAllZeroRows.length > examples.length ? `; plus ${formatNumber(oneHotUnknownAllZeroRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "With zeros enabled, each unknown category is encoded as an all-zero vector. That representation can collide with missing-feature conventions and carries no explicit unknown bucket unless the surrounding feature contract defines one.",
        recommendation: "Document and test the unknown-category policy, add an explicit unknown category when downstream semantics require distinction, and compare representative feature vectors against training-time preprocessing.",
        relevance: "preprocessing parity; unknown-category semantics; feature collision",
      }));
    }
    if (oneHotGuaranteedFailureRows.length) {
      const examples = oneHotGuaranteedFailureRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} zeros=0 with ${formatNumber(row.onehot_exact_unknown_input_count || 0)} artifact-known unknown value(s)`);
      findings.push(finding({
        id: "EA-ONX-0034",
        category: "runtime_compatibility",
        title: "ONNX-ML OneHotEncoder has an artifact-proven unknown-category runtime failure",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(oneHotGuaranteedFailureRows.length)} affected OneHotEncoder node(s)`,
        observation: `${examples.join("; ")}${oneHotGuaranteedFailureRows.length > examples.length ? `; plus ${formatNumber(oneHotGuaranteedFailureRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned ORT CPU kernel returns failure when zeros is zero and a category is absent from the vocabulary. Exact artifact-known input values prove that this branch is reached, so output propagation is suppressed.",
        recommendation: "Repair the vocabulary or enable a reviewed unknown policy, then require successful pinned-runtime execution for the exact fixture and production-representative categorical inputs.",
        relevance: "deterministic runtime failure; preprocessing contract; release blocking",
      }));
    }
    if (oneHotDtypeGapRows.length) {
      const examples = oneHotDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0035",
        category: "runtime_compatibility",
        title: "ONNX-ML OneHotEncoder dtype is schema-valid but lacks a pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(oneHotDtypeGapRows.length)} affected OneHotEncoder node(s)`,
        observation: `${examples.join("; ")}${oneHotDtypeGapRows.length > examples.length ? `; plus ${formatNumber(oneHotDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "OneHotEncoder-1 permits INT32 in the ONNX schema, while the pinned ORT CPU registration provides INT64, FLOAT32, FLOAT64, and STRING kernels. A schema-valid INT32 node can fail provider kernel resolution.",
        recommendation: "Convert the categorical tensor to INT64 or bind measured assignment evidence from a selected provider that implements INT32 before release.",
        relevance: "schema/runtime gap; execution-provider compatibility; session initialization",
      }));
    }
    if (oneHotNoncanonicalZerosRows.length) {
      const examples = oneHotNoncanonicalZerosRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} zeros=${row.onehot_zeros_value}`);
      findings.push(finding({
        id: "EA-ONX-0036",
        category: "integration_verification",
        title: "ONNX-ML OneHotEncoder uses a noncanonical boolean value for zeros",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(oneHotNoncanonicalZerosRows.length)} affected OneHotEncoder node(s)`,
        observation: `${examples.join("; ")}${oneHotNoncanonicalZerosRows.length > examples.length ? `; plus ${formatNumber(oneHotNoncanonicalZerosRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The schema describes zeros as a boolean encoded by INT64. The pinned kernel treats every nonzero integer as enabled, but a value outside 0 or 1 depends on permissive boolean coercion and can reduce portability across validators or runtimes.",
        recommendation: "Canonicalize zeros to 0 or 1 and preserve the intended unknown-category behavior in preprocessing and runtime tests.",
        relevance: "attribute portability; boolean contract; runtime consistency",
      }));
    }
    if (oneHotInvalidCastRows.length) {
      const examples = oneHotInvalidCastRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.onehot_numeric_to_int64_invalid_count || 0)} unrepresentable value(s) [${(row.onehot_unknown_input_preview || []).join(", ") || "none"}]`);
      findings.push(finding({
        id: "EA-ONX-0037",
        category: "integrity",
        title: "ONNX-ML OneHotEncoder input cannot be represented by the pinned numeric-to-INT64 lookup cast",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(oneHotInvalidCastRows.length)} affected OneHotEncoder node(s); ${formatNumber(mlValueInference.exact_onehot_numeric_to_int64_invalid_count || 0)} unrepresentable value(s)`,
        observation: `${examples.join("; ")}${oneHotInvalidCastRows.length > examples.length ? `; plus ${formatNumber(oneHotInvalidCastRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned numeric kernel converts FLOAT32/FLOAT64 inputs to INT64 before lookup. NaN, infinity, or a finite value outside the signed INT64 range does not have a portable representable result, so DEEPBOM refuses to emulate implementation-undefined conversion behavior.",
        recommendation: "Constrain categorical inputs to finite signed-INT64-representable values, regenerate the artifact fixture, and require exact native ORT parity before downstream use.",
        relevance: "numeric conversion integrity; undefined cast avoidance; categorical identity",
      }));
    }
    if (linearRuntimeInvalidRows.length) {
      const examples = linearRuntimeInvalidRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${row.linear_pinned_ort_contract_reason || "pinned ORT runtime contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0038",
        category: "runtime_compatibility",
        title: "ONNX-ML linear model violates the pinned ORT CPU runtime contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(linearRuntimeInvalidRows.length)} affected linear-model node(s)`,
        observation: `${examples.join("; ")}${linearRuntimeInvalidRows.length > examples.length ? `; plus ${formatNumber(linearRuntimeInvalidRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The serialized node can satisfy part of the ONNX schema while still failing the pinned ORT CPU constructor or compute path because required runtime attributes, rank, feature count, coefficient cardinality, labels, targets, or dtype do not satisfy that implementation's executable contract.",
        recommendation: "Treat the affected node as release blocking, repair the exact reason recorded in structured evidence, and require both zero static contract failures and a successful pinned native ORT fixture before deployment qualification.",
        relevance: "deterministic runtime failure; ONNX/ORT contract gap; release blocking",
      }));
    }
    if (linearIgnoredParameterRows.length) {
      const examples = linearIgnoredParameterRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${formatNumber(row.linear_unused_coefficient_count || 0)} coefficient(s) and ${formatNumber(row.linear_ignored_intercept_count || 0)} intercept(s) ignored`);
      findings.push(finding({
        id: "EA-ONX-0039",
        category: "integrity",
        title: "ONNX-ML linear model carries parameters ignored by the pinned ORT kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(linearIgnoredParameterRows.length)} affected node(s); ${formatNumber(mlValueInference.exact_linear_unused_coefficient_count || 0)} unused coefficient(s); ${formatNumber(mlValueInference.exact_linear_ignored_intercept_count || 0)} ignored intercept(s)`,
        observation: `${examples.join("; ")}${linearIgnoredParameterRows.length > examples.length ? `; plus ${formatNumber(linearIgnoredParameterRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned kernel consumes only targets/classes multiplied by feature count coefficients, ignores trailing coefficients, and ignores a LinearRegressor intercept vector unless its length exactly equals targets. The artifact therefore contains serialized model state that does not affect the selected runtime result.",
        recommendation: "Regenerate the node with exact coefficient and intercept cardinalities, compare outputs with the training pipeline, and reject unexplained dead parameters from signed release artifacts.",
        relevance: "parameter conservation; exporter correctness; hidden dead state",
      }));
    }
    if (linearRegressorDtypeGapRows.length) {
      const examples = linearRegressorDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0040",
        category: "runtime_compatibility",
        title: "ONNX-ML LinearRegressor dtype is schema-valid but lacks a pinned ORT CPU compute path",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(linearRegressorDtypeGapRows.length)} affected LinearRegressor node(s)`,
        observation: `${examples.join("; ")}${linearRegressorDtypeGapRows.length > examples.length ? `; plus ${formatNumber(linearRegressorDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "LinearRegressor-1 permits FLOAT64, INT32, and INT64 in the pinned ONNX schema, but the pinned ORT CPU compute implementation executes only FLOAT32 and returns an unsupported-input error for the other schema-valid types.",
        recommendation: "Convert the input contract to FLOAT32 or bind measured assignment and successful execution evidence from another selected provider before release.",
        relevance: "schema/runtime gap; execution-provider compatibility; deterministic CPU failure",
      }));
    }
    if (linearPostTransformRows.length) {
      const examples = linearPostTransformRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name} ${row.linear_post_transform || "unresolved"}: ${(row.risk_codes || []).filter((code) => code.includes("post_transform") || code.includes("probit")).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0042",
        category: "numerical_integrity",
        title: "ONNX-ML linear post-transform has a pinned ORT semantic gap or unsafe branch",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(linearPostTransformRows.length)} affected linear-model node(s)`,
        observation: `${examples.join("; ")}${linearPostTransformRows.length > examples.length ? `; plus ${formatNumber(linearPostTransformRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned source inspection proves one of the following: a single-score transform is a no-op, binary complement expansion bypasses the requested non-PROBIT transform, binary PROBIT does not populate the added second-class score slot, or regressor PROBIT can produce non-finite values outside its mathematical input domain.",
        recommendation: "Do not infer probabilities from the operator name alone. Replace the hazardous transform or make it explicit outside the node, compare exact score tensors with a pinned native ORT fixture, and document whether consumers expect raw margins or transformed values.",
        relevance: "score semantics; probability contract; numerical safety",
      }));
    }
    if (linearMultiClassRows.length) {
      const examples = linearMultiClassRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} multi_class=${row.linear_multi_class_value || "unresolved"}`);
      findings.push(finding({
        id: "EA-ONX-0043",
        category: "integration_verification",
        title: "ONNX-ML LinearClassifier multi_class intent is ignored by the pinned ORT CPU implementation",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(linearMultiClassRows.length)} affected LinearClassifier node(s)`,
        observation: `${examples.join("; ")}${linearMultiClassRows.length > examples.length ? `; plus ${formatNumber(linearMultiClassRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned constructor parses and stores multi_class, but the selected CPU compute path does not consult it. A nonzero serialized value therefore communicates model intent without changing execution.",
        recommendation: "Remove reliance on multi_class for runtime behavior, validate the exact multiclass score and label contract against training-time reference outputs, and pin that parity fixture to the deployment runtime.",
        relevance: "ignored operator intent; multiclass semantics; runtime parity",
      }));
    }
    if (linearDuplicateLabelRows.length) {
      const examples = linearDuplicateLabelRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.linear_duplicate_label_count || 0)} duplicate label value(s)`);
      findings.push(finding({
        id: "EA-ONX-0044",
        category: "output_contract",
        title: "ONNX-ML LinearClassifier contains duplicate class labels",
        evidence: "OBSERVED/DERIVED",
        priority: "Medium",
        op: `${formatNumber(linearDuplicateLabelRows.length)} affected LinearClassifier node(s)`,
        observation: `${examples.join("; ")}${linearDuplicateLabelRows.length > examples.length ? `; plus ${formatNumber(linearDuplicateLabelRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Multiple score columns map to the same externally visible label. Argmax can still select an index, but downstream consumers cannot reconstruct a unique class identity from the emitted label value.",
        recommendation: "Make every serialized class label unique, regenerate the classifier, and verify label-map order together with score-column order in the application contract.",
        relevance: "label identity; output decoding; class-map integrity",
      }));
    }
    if (linearNumericalRiskRows.length) {
      const examples = linearNumericalRiskRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${formatNumber(row.linear_non_finite_parameter_count || 0)} non-finite parameter(s), ${formatNumber(row.linear_reference_non_finite_raw_score_count || 0)} non-finite reference score(s), ${formatNumber(row.linear_reference_decision_boundary_count || 0)} exact zero-margin decision(s)`);
      findings.push(finding({
        id: "EA-ONX-0045",
        category: "numerical_integrity",
        title: "ONNX-ML linear model has non-finite arithmetic or a reference decision-boundary instability",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(linearNumericalRiskRows.length)} affected linear-model node(s)`,
        observation: `${examples.join("; ")}${linearNumericalRiskRows.length > examples.length ? `; plus ${formatNumber(linearNumericalRiskRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Non-finite coefficients, intercepts, inputs, or derived scalar reference scores invalidate ordinary ranking semantics. For binary classifiers, an exact zero raw margin also sits on the pinned greater-than-zero decision boundary, where small accumulation-order differences can change the emitted label.",
        recommendation: "Reject non-finite model state, inspect zero-margin fixtures as numerical boundary cases, and compare pinned native ORT outputs across intended CPU feature profiles before release.",
        relevance: "numerical integrity; decision stability; microkernel accumulation sensitivity",
      }));
    }
    if (labelEncoderDtypeGapRows.length) {
      const examples = labelEncoderDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} v${row.resolved_schema_version} ${row.input_dtype}->${row.output_dtype}`);
      findings.push(finding({
        id: "EA-ONX-0046",
        category: "runtime_compatibility",
        title: "ONNX-ML LabelEncoder dtype pair lacks a pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(labelEncoderDtypeGapRows.length)} affected LabelEncoder node(s)`,
        observation: `${examples.join("; ")}${labelEncoderDtypeGapRows.length > examples.length ? `; plus ${formatNumber(labelEncoderDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The versioned ONNX schema accepts the serialized input/output element types, but the pinned ORT CPU registration does not provide that exact typed pair. Static shape remains valid while CPU kernel resolution is not.",
        recommendation: "Use a dtype pair present in the pinned CPU registration ledger or bind measured assignment and successful execution from the selected deployment provider before release.",
        relevance: "schema/runtime compatibility; execution-provider resolution; categorical preprocessing",
      }));
    }
    if (labelEncoderDuplicateRows.length) {
      const examples = labelEncoderDuplicateRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} v${row.resolved_schema_version}: ${formatNumber(row.label_encoder_duplicate_key_count || 0)} duplicate(s), ${formatNumber(row.label_encoder_schema_runtime_mismatch_count || 0)} artifact-known mismatch(es)`);
      findings.push(finding({
        id: "EA-ONX-0047",
        category: "integrity",
        title: "ONNX-ML LabelEncoder duplicate keys have versioned or runtime-dependent ownership",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(labelEncoderDuplicateRows.length)} affected LabelEncoder node(s); ${formatNumber(mlValueInference.exact_label_encoder_duplicate_key_hit_count || 0)} exact duplicate-key hit(s)`,
        observation: `${examples.join("; ")}${labelEncoderDuplicateRows.length > examples.length ? `; plus ${formatNumber(labelEncoderDuplicateRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "LabelEncoder-4 specifies that the last repeated key wins, while the pinned ORT CPU implementation inserts with emplace and retains the first value. Version 1 string classes also resolve repeated labels to the final index in the pinned runtime without an equally explicit schema rule. When artifact-known inputs reach the duplicated key, DEEPBOM reports both outputs and suppresses exact downstream value propagation.",
        recommendation: "Remove duplicate keys, preserve the intended unique mapping order, and compare the repaired node against both the pinned ONNX reference semantics and native ORT before release.",
        relevance: "deterministic output divergence; label identity; exporter integrity",
      }));
    }
    if (labelEncoderNanRows.length) {
      const examples = labelEncoderNanRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.label_encoder_nan_key_count || 0)} NaN key(s)`);
      findings.push(finding({
        id: "EA-ONX-0048",
        category: "runtime_compatibility",
        title: "LabelEncoder-2 NaN lookup contract differs from the pinned ORT CPU equality path",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(labelEncoderNanRows.length)} affected LabelEncoder node(s)`,
        observation: `${examples.join("; ")}${labelEncoderNanRows.length > examples.length ? `; plus ${formatNumber(labelEncoderNanRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The version 2 ONNX documentation promises bitwise float-key comparison so a NaN key can be mapped. The pinned ORT version 2 implementation uses its ordinary float hash/equality map, where NaN lookup does not match; version 4 introduces explicit NaN-aware equality. This is a source-backed cross-version semantic gap.",
        recommendation: "Do not use NaN as a LabelEncoder-2 key. Upgrade to a reviewed version 4 mapping or replace NaN before lookup, then require native ORT parity for the exact fixture.",
        relevance: "NaN semantics; version migration; preprocessing parity",
      }));
    }
    if (labelEncoderDefaultRows.length) {
      const examples = labelEncoderDefaultRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${formatNumber(row.label_encoder_exact_default_count || 0)} / ${formatNumber(row.label_encoder_exact_input_value_count || 0)} exact input(s) -> ${row.label_encoder_default_value}`);
      findings.push(finding({
        id: "EA-ONX-0049",
        category: "input_contract",
        title: "ONNX-ML LabelEncoder artifact-known inputs reach the default mapping path",
        evidence: "OBSERVED/DERIVED",
        priority: "Medium",
        op: `${formatNumber(labelEncoderDefaultRows.length)} affected LabelEncoder node(s); ${formatNumber(mlValueInference.exact_label_encoder_default_count || 0)} defaulted value(s)`,
        observation: `${examples.join("; ")}${labelEncoderDefaultRows.length > examples.length ? `; plus ${formatNumber(labelEncoderDefaultRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Exact artifact-known input values are absent from the serialized key set and therefore emit the active default. That behavior may be intentional unknown-category handling, but it is a concrete preprocessing event rather than an untested possibility.",
        recommendation: "Confirm that the default value is reserved and documented, verify downstream handling, and add representative unknown-category fixtures to the release contract.",
        relevance: "unknown-category policy; preprocessing contract; downstream interpretation",
      }));
    }
    if (labelEncoderNonfiniteRows.length) {
      const examples = labelEncoderNonfiniteRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} key/value non-finite ${formatNumber(row.label_encoder_non_finite_key_count || 0)} / ${formatNumber(row.label_encoder_non_finite_value_count || 0)}`);
      findings.push(finding({
        id: "EA-ONX-0050",
        category: "numerical_integrity",
        title: "ONNX-ML LabelEncoder mapping contains non-finite numeric state",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(labelEncoderNonfiniteRows.length)} affected LabelEncoder node(s)`,
        observation: `${examples.join("; ")}${labelEncoderNonfiniteRows.length > examples.length ? `; plus ${formatNumber(labelEncoderNonfiniteRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "NaN or infinity is serialized as a key, mapped value, or default. Equality and downstream numerical behavior are version-sensitive, and a non-finite emitted value can invalidate ordinary feature or label interpretation.",
        recommendation: "Replace non-finite mapping state with explicit finite sentinel categories, document the conversion, and require exact schema/reference/runtime parity before release.",
        relevance: "numerical integrity; category identity; downstream safety",
      }));
    }
    if (labelEncoderRuntimeInvalidRows.length) {
      const examples = labelEncoderRuntimeInvalidRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} v${row.resolved_schema_version}: ${formatNumber(row.label_encoder_key_count || 0)} key(s) / ${formatNumber(row.label_encoder_value_count || 0)} value(s)`);
      findings.push(finding({
        id: "EA-ONX-0051",
        category: "runtime_compatibility",
        title: "ONNX-ML LabelEncoder violates the pinned ORT key/value cardinality contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(labelEncoderRuntimeInvalidRows.length)} affected LabelEncoder node(s)`,
        observation: `${examples.join("; ")}${labelEncoderRuntimeInvalidRows.length > examples.length ? `; plus ${formatNumber(labelEncoderRuntimeInvalidRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The serialized LabelEncoder version does not require equal key/value counts at ONNX schema inference time, but the pinned ORT CPU constructor rejects the mismatch before execution. The ONNX-derived output shape and type remain valid; executable CPU behavior does not.",
        recommendation: "Emit exactly one value for every key, rerun the pinned ONNX/ORT parity fixture, and reject the artifact until the runtime contract passes.",
        relevance: "deterministic runtime failure; exporter integrity; categorical preprocessing",
      }));
    }
    if (svmRuntimeInvalidRows.length) {
      const examples = svmRuntimeInvalidRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${row.svm_pinned_ort_contract_reason || "pinned ORT runtime contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0052",
        category: "runtime_compatibility",
        title: "ONNX-ML SVM violates the pinned ORT CPU executable contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(svmRuntimeInvalidRows.length)} affected SVM node(s)`,
        observation: `${examples.join("; ")}${svmRuntimeInvalidRows.length > examples.length ? `; plus ${formatNumber(svmRuntimeInvalidRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The artifact cannot execute on the pinned ORT CPU path because its constructor/runtime parameter layout is invalid. This includes the source-proven case where ONNX permits kernel_params to be omitted but the pinned ORT constructor still requires the attribute to exist, as well as invalid vector, feature, coefficient, support-vector, rho, or probability cardinalities.",
        recommendation: "Repair the exact row-level contract reason, preserve an explicit empty or three-value kernel_params attribute as required by the pinned runtime, and require a successful serialized native ORT fixture before release.",
        relevance: "deterministic runtime failure; ONNX/ORT contract gap; release blocking",
      }));
    }
    if (svmScoreWidthMismatchRows.length) {
      const examples = svmScoreWidthMismatchRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} classes=${formatNumber(row.svm_class_label_count || 0)}, ONNX width=${formatNumber(row.svm_schema_score_width || 0)}, pinned ORT width=${formatNumber(row.svm_pinned_ort_score_width || 0)}`);
      findings.push(finding({
        id: "EA-ONX-0053",
        category: "integrity",
        title: "SVMClassifier ONNX score shape conflicts with pinned ORT output width",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(svmScoreWidthMismatchRows.length)} affected SVMClassifier node(s)`,
        observation: `${examples.join("; ")}${svmScoreWidthMismatchRows.length > examples.length ? `; plus ${formatNumber(svmScoreWidthMismatchRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "For a multiclass SVC without probability parameters, the ONNX schema describes one score per class while the pinned ORT kernel emits pairwise decision scores. At four or more classes, C and C(C-1)/2 differ, so downstream shape propagation would be false and is deliberately suppressed.",
        recommendation: "Do not release against an inferred [N,C] score contract. Add probability parameters or normalize the score interface outside the node, then bind the repaired serialized model to native ORT output-shape and value parity tests.",
        relevance: "schema/runtime divergence; output ABI; deterministic shape conflict",
      }));
    }
    if (svmDtypeGapRows.length) {
      const examples = svmDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} SVMRegressor ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0054",
        category: "runtime_compatibility",
        title: "SVMRegressor dtype is schema-valid but lacks the pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(svmDtypeGapRows.length)} affected SVMRegressor node(s)`,
        observation: `${examples.join("; ")}${svmDtypeGapRows.length > examples.length ? `; plus ${formatNumber(svmDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "SVMRegressor-1 permits FLOAT64, INT32, and INT64 inputs in the pinned ONNX schema, but the pinned ORT CPU registration exposes only FLOAT32. Static output type and shape do not establish an executable CPU path.",
        recommendation: "Convert the input contract to FLOAT32 or attach observed successful assignment and execution evidence from the intended provider before deployment qualification.",
        relevance: "schema/runtime gap; execution-provider compatibility; deterministic CPU failure",
      }));
    }
    if (svmIgnoredTransformRows.length) {
      const examples = svmIgnoredTransformRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} post_transform=${row.svm_post_transform || "UNRESOLVED"}`);
      findings.push(finding({
        id: "EA-ONX-0055",
        category: "integrity",
        title: "SVMRegressor post_transform is serialized but ignored by pinned ORT CPU",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(svmIgnoredTransformRows.length)} affected SVMRegressor node(s)`,
        observation: `${examples.join("; ")}${svmIgnoredTransformRows.length > examples.length ? `; plus ${formatNumber(svmIgnoredTransformRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned constructor stores post_transform, but its Compute path writes the raw regressor result without applying it. Consumers that interpret the output as LOGISTIC, SOFTMAX, SOFTMAX_ZERO, or PROBIT receive a different numerical contract from the serialized intent.",
        recommendation: "Use post_transform=NONE and apply an explicit reviewed transform outside SVMRegressor, or select a runtime with measured matching semantics and pin exact output fixtures.",
        relevance: "silent semantic divergence; output interpretation; runtime parity",
      }));
    }
    if (svmIgnoredOrForcedRows.length) {
      const examples = svmIgnoredOrForcedRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: unused support/coeff/rho/prob=${formatNumber(row.svm_unused_support_vector_value_count || 0)}/${formatNumber(row.svm_unused_coefficient_count || 0)}/${formatNumber(row.svm_unused_rho_count || 0)}/${formatNumber(row.svm_unused_probability_parameter_count || 0)}, kernel=${row.svm_kernel_type || "UNRESOLVED"}`);
      findings.push(finding({
        id: "EA-ONX-0056",
        category: "integrity",
        title: "ONNX-ML SVM carries ignored parameters or a kernel overridden by runtime mode",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(svmIgnoredOrForcedRows.length)} affected SVM node(s)`,
        observation: `${examples.join("; ")}${svmIgnoredOrForcedRows.length > examples.length ? `; plus ${formatNumber(svmIgnoredOrForcedRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned kernel consumes only its source-defined layout, ignores trailing serialized values, and forces LINEAR behavior when the support-vector count selects linear mode. The signed artifact therefore contains state or kernel intent that does not affect execution.",
        recommendation: "Regenerate exact-cardinality attributes, make the selected linear/SVC mode explicit, remove dead state, and compare repaired outputs with the training-time model and pinned native runtime.",
        relevance: "parameter conservation; exporter correctness; hidden dead state",
      }));
    }
    if (svmNumericalRiskRows.length) {
      const examples = svmNumericalRiskRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: non-finite parameter/reference=${formatNumber(row.svm_non_finite_parameter_count || 0)}/${formatNumber(row.svm_reference_non_finite_score_count || 0)}, boundary=${formatNumber(row.svm_reference_decision_boundary_count || 0)}`);
      findings.push(finding({
        id: "EA-ONX-0057",
        category: "integrity",
        title: "ONNX-ML SVM has non-finite arithmetic or a reference decision-boundary case",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(svmNumericalRiskRows.length)} affected SVM node(s)`,
        observation: `${examples.join("; ")}${svmNumericalRiskRows.length > examples.length ? `; plus ${formatNumber(svmNumericalRiskRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Non-finite parameters or scalar reference scores invalidate ordinary margin/probability semantics. An exact zero decision value is also sensitive to the executed MLAS accumulation and platform transcendental path, which static source analysis does not observe.",
        recommendation: "Reject non-finite state and validate every boundary fixture on pinned runtime builds across intended CPU feature profiles before release.",
        relevance: "numerical integrity; decision stability; microkernel and libm sensitivity",
      }));
    }
    if (svmSemanticHazardRows.length) {
      const examples = svmSemanticHazardRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${(row.risk_codes || []).filter((code) => code.startsWith("svm_") && !code.includes("runtime_contract")).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0058",
        category: "integrity",
        title: "ONNX-ML SVM exposes a source-backed label, transform, or flag semantic hazard",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(svmSemanticHazardRows.length)} affected SVM node(s)`,
        observation: `${examples.join("; ")}${svmSemanticHazardRows.length > examples.length ? `; plus ${formatNumber(svmSemanticHazardRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned source proves an externally visible ambiguity or special branch: duplicate class labels, a second transform after SVC probability coupling, binary complement expansion, an unwritten PROBIT score slot, or a noncanonical one_class value treated as true.",
        recommendation: "Canonicalize labels and flags, remove implicit transform branches, and lock the complete label/score tensor contract with serialized native ORT parity fixtures.",
        relevance: "output ABI; classification semantics; deterministic runtime behavior",
      }));
    }
    if (treeRuntimeInvalidRows.length) {
      const examples = treeRuntimeInvalidRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${row.tree_pinned_ort_contract_reason || row.tree_onnx_contract_reason || "pinned executable contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0059",
        category: "runtime_compatibility",
        title: "ONNX-ML TreeEnsemble violates a pinned executable topology or parameter contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(treeRuntimeInvalidRows.length)} affected TreeEnsemble node(s)`,
        observation: `${examples.join("; ")}${treeRuntimeInvalidRows.length > examples.length ? `; plus ${formatNumber(treeRuntimeInvalidRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The serialized tree cannot be executed under the pinned ONNX/ORT contract because a required tensor or tuple cardinality, root, child, feature, node identity, cycle, class/target, weight, mode, aggregate, transform, rank, or dtype invariant fails. Output propagation is suppressed instead of treating an initializer default as a result.",
        recommendation: "Repair the exact row-level reason, require a zero-failure topology ledger, and run the serialized fixture through the pinned native ORT CPU build before release.",
        relevance: "deterministic runtime failure; tree topology integrity; release blocking",
      }));
    }
    if (treeDtypeGapRows.length) {
      const examples = treeDtypeGapRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name} ${row.input_dtype || "UNKNOWN"}`);
      findings.push(finding({
        id: "EA-ONX-0060",
        category: "runtime_compatibility",
        title: "TreeEnsemble dtype is schema-valid but lacks the pinned ORT CPU kernel",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(treeDtypeGapRows.length)} affected TreeEnsemble node(s)`,
        observation: `${examples.join("; ")}${treeDtypeGapRows.length > examples.length ? `; plus ${formatNumber(treeDtypeGapRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The pinned schema permits this input type, but the pinned CPU registration does not: generic TreeEnsemble-5 has no FLOAT16 CPU kernel and legacy TreeEnsembleRegressor has no INT32 or INT64 CPU kernel. A valid inferred output type does not establish an executable CPU path.",
        recommendation: "Convert to a pinned CPU-supported dtype or attach observed successful assignment and value-parity evidence from the intended execution provider before qualification.",
        relevance: "schema/runtime gap; execution-provider compatibility; session initialization",
      }));
    }
    if (treeDeprecatedRows.length) {
      const examples = treeDeprecatedRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}-${formatNumber(row.resolved_schema_version || 5)}`);
      findings.push(finding({
        id: "EA-ONX-0061",
        category: "integration_verification",
        title: "Legacy TreeEnsemble operator remains serialized at the deprecation opset",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(treeDeprecatedRows.length)} deprecated legacy TreeEnsemble node(s)`,
        observation: `${examples.join("; ")}${treeDeprecatedRows.length > examples.length ? `; plus ${formatNumber(treeDeprecatedRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "TreeEnsembleClassifier and TreeEnsembleRegressor are deprecated at ai.onnx.ml opset 5 in favor of indexed TreeEnsemble-5. Existing runtime behavior can remain valid, but exporter and future-provider portability now depend on a legacy contract.",
        recommendation: "Plan a controlled conversion to TreeEnsemble-5 and bind pre/post conversion outputs, labels, aggregate mode, transform behavior, and missing-value paths to exact parity fixtures.",
        relevance: "operator lifecycle; exporter portability; runtime maintenance",
      }));
    }
    if (treeDeadOrNonTreeStateRows.length) {
      const examples = treeDeadOrNonTreeStateRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: orphan ${formatNumber(row.tree_orphan_node_or_leaf_count || 0)}, ignored nonleaf/single-target weights ${formatNumber(row.tree_ignored_nonleaf_weight_count || 0)}/${formatNumber(row.tree_single_target_ignored_weight_count || 0)}, multiple-parent nodes ${formatNumber(row.tree_multiple_parent_node_count || 0)}`);
      findings.push(finding({
        id: "EA-ONX-0062",
        category: "integrity",
        title: "TreeEnsemble contains unreachable, ignored, or non-tree serialized state",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(treeDeadOrNonTreeStateRows.length)} affected TreeEnsemble node(s); ${formatNumber(mlValueInference.exact_tree_ensemble_unused_weight_count || 0)} unused weight(s); ${formatNumber(mlValueInference.exact_tree_ensemble_orphan_node_or_leaf_count || 0)} orphan node/leaf record(s)`,
        observation: `${examples.join("; ")}${treeDeadOrNonTreeStateRows.length > examples.length ? `; plus ${formatNumber(treeDeadOrNonTreeStateRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned runtime semantics ignore weights attached to nonleaves and additional single-target leaf IDs; generic roots can also leave serialized nodes or leaves unreachable, while shared subtrees violate strict-tree ownership. The signed artifact therefore carries state that is dead or structurally surprising even when execution succeeds.",
        recommendation: "Regenerate a strict reachable forest with one parent per nonroot and only runtime-consumed leaf weights, then require exact weight conservation and output parity.",
        relevance: "parameter conservation; exporter integrity; hidden dead state",
      }));
    }
    if (treeMembershipRows.length) {
      const examples = treeMembershipRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} MEMBER values/duplicates/separators ${formatNumber(row.tree_membership_value_count || 0)}/${formatNumber(row.tree_membership_duplicate_value_count || 0)}/${formatNumber(row.tree_membership_separator_count || 0)}: ${(row.risk_codes || []).filter((code) => code.includes("membership") || code.includes("zero_member")).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0063",
        category: "integration_verification",
        title: "TreeEnsemble-5 MEMBER set has duplicate values or a pinned reference/runtime divergence",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(treeMembershipRows.length)} affected TreeEnsemble-5 node(s); ${formatNumber(mlValueInference.exact_tree_ensemble_membership_duplicate_value_count || 0)} duplicate member value(s)`,
        observation: `${examples.join("; ")}${treeMembershipRows.length > examples.length ? `; plus ${formatNumber(treeMembershipRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Duplicate MEMBER values add no branch information, and a zero member exposes a source-pinned divergence: ORT tests membership correctly while the pinned ONNX Python reference parser terminates its set loop on zero. Static path selection follows the selected ORT semantics and records the reference disagreement.",
        recommendation: "Deduplicate every membership set, avoid zero until the reference path is fixed or explicitly waive the divergence, and pin branch outcomes for zero, NaN, present, and absent members in native ORT fixtures.",
        relevance: "branch semantics; reference/runtime parity; test portability",
      }));
    }
    if (treeSemanticHazardRows.length) {
      const examples = treeSemanticHazardRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${(row.risk_codes || []).filter((code) => code.startsWith("tree_")).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0064",
        category: "integrity",
        title: "TreeEnsemble exposes a source-backed label, base-value, transform, or binary-score hazard",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(treeSemanticHazardRows.length)} affected TreeEnsemble node(s); ${formatNumber(mlValueInference.exact_tree_ensemble_reference_unwritten_score_count || 0)} unwritten reference score slot(s)`,
        observation: `${examples.join("; ")}${treeSemanticHazardRows.length > examples.length ? `; plus ${formatNumber(treeSemanticHazardRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Pinned source proves an externally visible special branch: duplicate labels, INT64 binary index semantics, underspecified one-base-value handling, a single-target transform that becomes a no-op, or a binary transform path that leaves a score slot unwritten. The operator name alone is insufficient to infer the application score contract.",
        recommendation: "Canonicalize labels and base values, remove implicit no-op/unwritten transform cases, and bind every label plus complete score tensor to serialized native ORT parity fixtures.",
        relevance: "output ABI; classification semantics; deterministic runtime behavior",
      }));
    }
    if (treeNumericalRiskRows.length) {
      const examples = treeNumericalRiskRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: non-finite parameter/reference ${formatNumber(row.tree_non_finite_parameter_count || 0)}/${formatNumber(row.tree_reference_non_finite_score_count || 0)}, boundary ${formatNumber(row.tree_reference_decision_boundary_count || 0)}`);
      findings.push(finding({
        id: "EA-ONX-0065",
        category: "integrity",
        title: "TreeEnsemble has non-finite arithmetic or an exact reference decision-boundary case",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(treeNumericalRiskRows.length)} affected TreeEnsemble node(s)`,
        observation: `${examples.join("; ")}${treeNumericalRiskRows.length > examples.length ? `; plus ${formatNumber(treeNumericalRiskRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Non-finite thresholds, weights, inputs, or scalar reference scores invalidate ordinary comparison and ranking semantics. An exact zero classifier decision also lies on a branch where runtime floating reduction order can change the selected label; the report therefore does not claim runtime-bit-exact scores.",
        recommendation: "Reject non-finite state and execute every boundary fixture on pinned runtime builds across intended CPU feature profiles and execution providers before release.",
        relevance: "numerical integrity; decision stability; runtime reduction sensitivity",
      }));
    }
    if (tfidfInvalidRows.length) {
      const examples = tfidfInvalidRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "TfIdfVectorizer contract failure"}`);
      findings.push(finding({
        id: "EA-ONX-0066",
        category: "runtime_compatibility",
        title: "TfIdfVectorizer violates a pinned schema or ORT CPU execution contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(tfidfInvalidRows.length)} invalid TfIdfVectorizer node(s)`,
        observation: `${examples.join("; ")}${tfidfInvalidRows.length > examples.length ? `; plus ${formatNumber(tfidfInvalidRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The serialized mode, gram bounds, skip count, pool type, level boundaries, whole-n-gram divisibility, definition/index/weight cardinality, output coordinate, duplicate active n-gram, input rank, or dtype cannot satisfy the pinned executable contract. Static output propagation is suppressed.",
        recommendation: "Regenerate the vectorizer with one type-matched nonempty pool, monotone in-bounds ngram_counts, one index and optional finite weight per n-gram, unique active n-grams, and a rank-one or positive-batch rank-two input; require native ORT parity before release.",
        relevance: "deterministic runtime failure; text feature ABI; release blocking",
      }));
    }
    if (tfidfWeightSemanticRows.length) {
      const examples = tfidfWeightSemanticRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.mode}: ${formatNumber(row.exact_weight_coordinate_value_disagreement_count || 0)} value disagreement(s) across ${formatNumber(row.exact_weight_coordinate_disagreement_count || 0)} remapped n-gram(s)`);
      findings.push(finding({
        id: "EA-ONX-0067",
        category: "integrity",
        title: "TfIdfVectorizer weight mapping differs between ONNX prose and pinned executable sources",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(tfidfWeightSemanticRows.length)} affected weighted vectorizer node(s)`,
        observation: `${examples.join("; ")}${tfidfWeightSemanticRows.length > examples.length ? `; plus ${formatNumber(tfidfWeightSemanticRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The ONNX schema text associates weights[i] with pool n-gram i, while both the pinned ONNX reference implementation and ORT CPU kernel read weights[ngram_indexes[i]]. A non-identity remap with unequal values therefore changes the feature vector relative to the prose contract.",
        recommendation: "Canonicalize ngram_indexes to identity order or reorder weights so pool-index and output-coordinate interpretations are numerically identical, then bind exact vectors from the intended runtime.",
        relevance: "feature semantics; reference contract divergence; model portability",
      }));
    }
    if (tfidfReferenceDivergenceRows.length) {
      const examples = tfidfReferenceDivergenceRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${formatNumber(row.exact_ort_reference_divergent_output_count || 0)} / ${formatNumber(row.exact_output_value_count || 0)} output value(s)`);
      findings.push(finding({
        id: "EA-ONX-0068",
        category: "numerical_behavior",
        title: "TfIdfVectorizer ORT repeated FLOAT32 weighting differs from ONNX reference arithmetic",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Medium",
        op: `${formatNumber(tfidfReferenceDivergenceRows.length)} exact static node(s)`,
        observation: `${examples.join("; ")}${tfidfReferenceDivergenceRows.length > examples.length ? `; plus ${formatNumber(tfidfReferenceDivergenceRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "For weighted TFIDF, ORT adds the FLOAT32 weight once per hit, while the ONNX Python reference multiplies the final integer frequency by the weight and casts once. Repeated rounding can produce different finite FLOAT32 outputs even with identical matches.",
        recommendation: "Use the intended runtime as the feature-vector authority, retain the exact divergent coordinates as a regression fixture, and avoid assuming ONNX reference bit parity for repeated weighted terms.",
        relevance: "numerical ABI; runtime/reference parity; downstream classifier inputs",
      }));
    }
    if (tfidfNoncanonicalRows.length) {
      const examples = tfidfNoncanonicalRows.slice(0, 8).map((row) => `#${padOp(row.node_index)}: ${(row.risk_codes || []).filter((code) => code.includes("pool_prefix") || code.includes("duplicate_ngram") || code.includes("share_output")).join("/")}`);
      findings.push(finding({
        id: "EA-ONX-0069",
        category: "integration_verification",
        title: "TfIdfVectorizer contains ignored, duplicate, or coordinate-aliased feature state",
        evidence: "OBSERVED/DERIVED",
        priority: "Medium",
        op: `${formatNumber(tfidfNoncanonicalRows.length)} noncanonical vectorizer node(s)`,
        observation: `${examples.join("; ")}${tfidfNoncanonicalRows.length > examples.length ? `; plus ${formatNumber(tfidfNoncanonicalRows.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A nonzero first ngram_counts offset leaves a pool prefix unread, duplicate inactive n-grams retain dead serialized state, and shared output coordinates merge distinct n-gram counts. These may execute but obscure feature ownership and converter intent.",
        recommendation: "Serialize a zero-based compact pool with unique n-grams and one output coordinate per feature unless aggregation is an explicitly tested application contract.",
        relevance: "feature ownership; dead state; exporter reproducibility",
      }));
    }
    if (tfidfBoundedResidualRows.length) {
      findings.push(finding({
        id: "EA-ONX-0070",
        category: "evidence_completeness",
        title: "TfIdfVectorizer exact static execution exceeds the emitted analysis bound",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "Informational",
        op: `${formatNumber(tfidfBoundedResidualRows.length)} bounded residual node(s)`,
        observation: tfidfBoundedResidualRows.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.static_execution_status}; work ${formatNumber(row.exact_static_work_units ?? 0)}; output width ${formatNumber(row.exact_output_width ?? 0)}`).join("; "),
        interpretation: "The artifact contract and output shape remain deterministic, but materializing every static output or enumerating every bounded skip path would exceed the declared browser work limit. No synthetic zero vector is emitted.",
        recommendation: "Run the pinned native fixture or raise the audited bound in a controlled build with memory and denial-of-service tests; keep the result partial until complete arithmetic is independently reconstructed.",
        relevance: "evidence completeness; bounded analysis; denial-of-service resistance",
      }));
    }
    if (shapeConflicts.length) {
      const examples = shapeConflicts.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name} ${row.tensor_name} ${row.field}: declared ${row.declared}, inferred ${row.inferred}`);
      findings.push(finding({
        id: "EA-ONX-0006",
        category: "integrity",
        title: "ONNX declared tensor contract contradicts deterministic shape inference",
        evidence: "OBSERVED/DERIVED",
        priority: "High",
        op: `${formatNumber(shapeConflicts.length)} declared/inferred conflict(s)`,
        observation: `${examples.join("; ")}${shapeConflicts.length > examples.length ? `; plus ${formatNumber(shapeConflicts.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "The artifact declares a dtype, rank, or concrete dimension that contradicts a deterministic local rule under the imported graph contract. The analyzer preserves the artifact declaration and does not replace it with the inferred value. Downstream MAC, payload, liveness, and runtime compatibility conclusions that depend on the disputed tensor are not trustworthy until the artifact is repaired.",
        recommendation: "Reject the artifact for release, run the pinned ONNX checker and reference shape inference, correct the exporter or stale value_info/output declaration, regenerate the model, and require a zero-conflict DeepBOM report before runtime qualification.",
        relevance: "artifact integrity; shape contract; downstream metric validity",
      }));
    }
    if (semanticShapeConflicts.length) {
      const examples = semanticShapeConflicts.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name}: ${row.reason}`);
      findings.push(finding({
        id: "EA-ONX-0071",
        category: "integrity",
        title: "ONNX operator contract is deterministically invalid",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(semanticShapeConflicts.length)} semantic contract conflict(s)`,
        observation: `${examples.join("; ")}${semanticShapeConflicts.length > examples.length ? `; plus ${formatNumber(semanticShapeConflicts.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "Artifact-known shapes, controls, or attributes violate the pinned operator contract. This is not missing analysis or an unknown runtime dimension. The invalid output and every dependent node are causally blocked, so downstream arithmetic, payload, liveness, and provider conclusions are not promoted.",
        recommendation: "Reject the artifact for release, repair the exporter or serialized control/shape contract, regenerate the model, and require zero semantic-contract conflicts before runtime qualification.",
        relevance: "artifact integrity; operator semantics; downstream metric validity",
      }));
    }
    if (conditionalInvalidTensors.length) {
      const uniqueFailures = new Map();
      for (const tensor of conditionalInvalidTensors) for (const failure of tensor.conditional_shape_contract?.variant_failures || []) {
        if (failure?.status !== "invalid") continue;
        const key = JSON.stringify({ reason: failure.reason || "conditional_shape_variant_invalid", conditions: failure.conditions || [] });
        if (!uniqueFailures.has(key)) uniqueFailures.set(key, { tensor, failure });
      }
      const examples = [...uniqueFailures.values()].slice(0, 8).map(({ tensor, failure }) => {
        const conditions = (failure.conditions || []).map((condition) => `${condition.key}=${condition.value}`).join(" & ") || "condition not named";
        return `T${formatNumber(tensor.index)} ${tensor.name || "unnamed"}: ${failure.reason || "conditional shape variant invalid"} under ${conditions}`;
      });
      findings.push(finding({
        id: "EA-ONX-0072",
        category: "integrity",
        title: "ONNX runtime branch has a deterministically invalid tensor contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(shapeInference.conditionally_invalid_node_output_count || conditionalInvalidTensors.length)} conditionally invalid output(s); ${formatNumber(shapeInference.conditional_invalid_variant_count || uniqueFailures.size)} invalid variant record(s)`,
        observation: `${examples.join("; ")}${uniqueFailures.size > examples.length ? `; plus ${formatNumber(uniqueFailures.size - examples.length)} more unique branch failure(s) in structured evidence` : ""}.`,
        interpretation: "At least one finite, artifact-derived control-flow branch produces a tensor rank or dimension contract that a downstream pinned ONNX operator cannot accept. Viable branches are retained, but the model-wide MAC, payload, and liveness totals are not promoted through the invalid branch.",
        recommendation: "Constrain and serialize the external input contract so the invalid branch is unreachable, or repair the branch/operator graph. Re-run the pinned ONNX checker and require zero conditionally invalid outputs before release qualification.",
        relevance: "artifact integrity; conditional control flow; downstream metric validity",
      }));
    }
    if (invalidSchemaForms.length || invalidOpsetImports.length) {
      const examples = invalidSchemaForms.slice(0, 8).map((row) => `#${padOp(row.node_index)} ${row.op_name} at opset ${row.imported_opset ?? "unresolved"}: ${(row.reason_codes || []).join("/") || "formal schema failure"}`);
      const importExamples = invalidOpsetImports.slice(0, Math.max(0, 8 - examples.length)).map((row) => `opset import #${row.index} ${row.domain}@${row.version}: ${(row.reason_codes || []).join("/") || "invalid import"}`);
      const totalFailures = invalidSchemaForms.length + invalidOpsetImports.length;
      findings.push(finding({
        id: "EA-ONX-0007",
        category: "integrity",
        title: "ONNX opset import or node violates the pinned schema contract",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(invalidOpsetImports.length)} invalid OperatorSetIdProto import(s); ${formatNumber(invalidSchemaForms.length)} invalid NodeProto schema form(s)`,
        observation: `${[...importExamples, ...examples].join("; ")}${totalFailures > importExamples.length + examples.length ? `; plus ${formatNumber(totalFailures - importExamples.length - examples.length)} more in structured evidence` : ""}.`,
        interpretation: "A domain import has no valid positive version, or the node's operator availability, formal input/output cardinality or omission, attribute name, required attribute, AttributeProto discriminator, or value-field type contradicts the greatest pinned ONNX schema version allowed by the model. Repeated valid imports are retained separately and resolved to the highest referenced version under the serialized ModelProto contract. Shape inference is suppressed only where the effective import or node form is invalid, so downstream size, MAC, liveness, and provider-precheck values cannot silently inherit unspecified semantics.",
        recommendation: "Reject the artifact for release, run the pinned ONNX checker, correct the exporter/opset declaration or node form, regenerate the model, and require zero OpSchema formal-contract failures before runtime qualification.",
        relevance: "artifact integrity; opset legality; downstream metric validity",
      }));
    }
    if (failedFunctionCalls.length || failedControlFlow.length || failedSequenceMaps.length) {
      const functionExamples = failedFunctionCalls.slice(0, 6).map((row) => `${row.function_id} at ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "call contract failed"}`);
      const controlExamples = failedControlFlow.slice(0, Math.max(0, 6 - functionExamples.length)).map((row) => `${row.op_name} at ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "body inference failed"}`);
      const mapExamples = failedSequenceMaps.slice(0, Math.max(0, 6 - functionExamples.length - controlExamples.length)).map((row) => `SequenceMap at ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "body inference failed"}`);
      const totalFailures = failedFunctionCalls.length + failedControlFlow.length + failedSequenceMaps.length;
      findings.push(finding({
        id: "EA-ONX-0009",
        category: "integrity",
        title: "Reachable ONNX function or graph-body contract fails recursive shape inference",
        evidence: "SOURCE_PINNED_AND_DERIVED",
        priority: "High",
        op: `${formatNumber(failedFunctionCalls.length)} FunctionProto call failure(s); ${formatNumber(failedControlFlow.length)} If/Loop/Scan failure(s); ${formatNumber(failedSequenceMaps.length)} SequenceMap failure(s)`,
        observation: `${[...functionExamples, ...controlExamples, ...mapExamples].join("; ")}${totalFailures > functionExamples.length + controlExamples.length + mapExamples.length ? `; plus ${formatNumber(totalFailures - functionExamples.length - controlExamples.length - mapExamples.length)} more in structured evidence` : ""}.`,
        interpretation: "A reachable local-function binding, function/model opset contract, referenced attribute, branch/body cardinality, tensor or sequence type, SequenceMap length, or control-flow axis contract is invalid under the pinned ONNX semantics. The affected output is not inferred and downstream shape-dependent metrics remain invalid or incomplete.",
        recommendation: "Reject the artifact for release, run the same pinned ONNX checker/reference inferencer, repair the exporter function or graph-body contract, and require zero recursive-scope failures before runtime qualification.",
        relevance: "artifact integrity; local functions; control flow; sequence mapping; downstream metric validity",
      }));
    }
    if (Number(shapeScope.reachable_exclusion_count || 0) > 0 || Number(shapeScope.reachable_scope_unresolved_output_count || 0) > 0
      || partialControlFlow.length || partialSequenceMaps.length || partialRecursiveScopes.length || partialMlValueRows.length) {
      const recursiveExamples = [
        ...partialControlFlow.map((row) => `${row.op_name} ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "runtime-dependent control"}`),
        ...partialSequenceMaps.map((row) => `SequenceMap ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "runtime-dependent sequence"}`),
        ...partialRecursiveScopes.map((row) => `${row.scope}: ${(row.reason_codes || []).join("/") || `${formatNumber(row.unresolved_output_count || 0)} unresolved output(s)`}`),
        ...partialMlValueRows.map((row) => `${row.op_name || "ONNX-ML"} ${row.scope}/#${padOp(row.node_index)}: ${(row.reason_codes || []).join("/") || "runtime-dependent map cardinality"}`),
      ].slice(0, 8);
      findings.push(finding({
        id: "EA-ONX-0008",
        category: "integration_verification",
        title: "Reachable ONNX shape scopes retain explicit static-analysis residuals",
        evidence: "OBSERVED/NOT_ASSESSABLE",
        priority: Number(shapeInference.unknown_node_output_count || 0) > 0 ? "High" : "Medium",
        op: `${formatNumber(shapeScope.unassessed_reachable_node_count || 0)} residual reachable node(s); ${formatNumber(partialControlFlow.length)} partial control-flow node(s); ${formatNumber(partialSequenceMaps.length)} partial SequenceMap node(s); ${formatNumber(partialRecursiveScopes.length)} partial recursive scope(s); ${formatNumber(partialMlValueRows.length)} partial ONNX-ML value-contract node(s)`,
        observation: `${formatNumber(shapeScope.executed_reachable_scope_count || 0)} reachable scope(s) were recursively executed and ${formatNumber(shapeScope.fully_assessed_reachable_scope_count || 0)} were fully assessed; ${formatNumber(shapeScope.unassessed_reachable_node_count || 0)} unsupported, schema-invalid, or failed-rule node(s), ${formatNumber(shapeScope.reachable_scope_unresolved_output_count || 0)} unresolved scope output(s), and ${formatNumber(partialMlValueRows.length)} runtime-cardinality ONNX-ML residual(s) remain.${recursiveExamples.length ? ` Residual examples: ${recursiveExamples.join("; ")}.` : ""}`,
        interpretation: "The residual count is reconciled against exact structural inventory, recursive execution evidence, and source-pinned partial control-flow/container/map rows. Dynamic Loop control, SequenceMap length, ZipMap batch/feature cardinality, or dense CastMap input cardinality remains explicitly unresolved even when the surrounding TypeProto is known. Downstream shape, container-state, MAC, payload, liveness, and provider conclusions that depend on these residuals remain partial.",
        recommendation: "Inspect the named residual rows, add or repair the missing pinned operator/type rule, and retain pinned ONNX checker/reference inference plus ORT session evidence before promoting dependent metrics to complete.",
        relevance: "shape completeness; control flow; local functions; downstream metric validity",
      }));
    }
    const incompleteExternalPayloads = Math.max(0, Number(externalData.tensor_count || 0) - Number(externalData.verified_payload_count || 0));
    if (incompleteExternalPayloads > 0 || Number(externalData.payload_verification_failed_count || 0) > 0) {
      findings.push(finding({
        id: "EA-ONX-0004",
        category: "integrity",
        title: "ONNX external TensorProto payload coverage is incomplete",
        evidence: "OBSERVED/NOT_ASSESSABLE",
        priority: "High",
        op: `${formatNumber(externalData.tensor_count)} external TensorProto reference(s)`,
        observation: `${formatNumber(externalData.entry_count)} external_data record(s); ${formatNumber(externalData.verified_payload_count)} verified, ${formatNumber(externalData.payload_verification_failed_count || 0)} failed verification, ${formatNumber(incompleteExternalPayloads)} incomplete, ${formatNumber(externalData.malformed_reference_count)} malformed reference(s). Verified payload ${formatBytes(externalData.verified_payload_bytes || 0)}; declared ${externalData.declared_payload_bytes == null ? "not fully declared" : formatBytes(externalData.declared_payload_bytes)}.`,
        interpretation: "Graph structure is visible, but any unavailable range prevents complete constant-byte, decoded-value, duplicate-payload, weight-integrity, and quantization-contract conclusions.",
        recommendation: "Select every model-relative external_data sidecar, correct range/checksum failures, and retain the emitted sidecar SHA-256 plus tensor range ledger in the release manifest.",
        relevance: "artifact integrity; weight verification; reproducible packaging",
      }));
    }
    if (Number(domains.external_custom_node_count || 0) > 0) {
      findings.push(finding({
        id: "EA-ONX-0001",
        category: "integration_verification",
        title: "External ONNX custom-op registry is required",
        evidence: "DERIVED",
        priority: "High",
        op: (domains.external_custom_domains || []).join(" / "),
        observation: `${formatNumber(domains.external_custom_node_count)} node(s) in ${(domains.external_custom_domains || []).join(" / ")} are neither standard ONNX/ONNX-ML, pinned ORT contrib schemas, nor artifact-defined local functions.`,
        interpretation: "The artifact cannot be executed by a stock ORT build unless a matching custom-op library or execution provider registers these schemas and kernels.",
        recommendation: "Bind the custom-op registry/library version, source commit, binary SHA-256, supported target matrix, and an observed ORT assignment profile to the release.",
        relevance: "runtime integration; custom operator provenance",
      }));
    }
    if ((domains.duplicate_function_ids || []).length || (domains.recursive_function_cycles || []).length) {
      findings.push(finding({
        id: "EA-ONX-0002",
        category: "integrity",
        title: "ONNX local function registry is ambiguous or recursive",
        evidence: "OBSERVED",
        priority: "High",
        op: "FunctionProto registry",
        observation: `${formatNumber((domains.duplicate_function_ids || []).length)} duplicate function ID(s); ${formatNumber((domains.recursive_function_cycles || []).length)} recursive dependency cycle(s).`,
        interpretation: "Static function resolution is not unique or terminates in a recursive cycle, so deterministic expansion and source-rule support assessment cannot be concluded.",
        recommendation: "Remove duplicate domain/name/overload definitions and recursive dependencies, then regenerate and validate the artifact with ONNX checker and the pinned deployment runtime.",
        relevance: "artifact integrity; deterministic runtime resolution",
      }));
    } else if (Number(domains.model_local_function_call_count || 0) > 0) {
      findings.push(finding({
        id: "EA-ONX-0003",
        category: "integration_verification",
        title: "Model-local ONNX functions require observed runtime resolution",
        evidence: "DERIVED",
        priority: "Medium",
        op: `${formatNumber(domains.model_local_function_call_count)} local function call(s)`,
        observation: "FunctionProto definitions and dependencies are present in the artifact and are not external custom-op registry requirements; direct EP kernel rows do not prove how ORT inlines or assigns their bodies.",
        interpretation: "The function composition is structurally known, while graph optimization and execution-provider assignment remain runtime decisions.",
        recommendation: "Capture a graph-optimization-disabled ORT profile for original-node identity and a production-optimization profile for final EP assignment; retain both with runtime/build hashes.",
        relevance: "execution-provider assignment; runtime reproducibility",
        evidenceJsonPointers: analysis?.ort_compatibility_evidence
          ? ["/evidence/static_analysis/onnx_domain_analysis/functions", "/evidence/static_analysis/ort_compatibility_evidence/execution_providers"]
          : ["/evidence/static_analysis/onnx_domain_analysis/functions"],
      }));
    }
    const epExclusions = (analysis?.ort_compatibility_evidence?.execution_providers || []).flatMap((ep) => (ep.ops || [])
      .filter((row) => row.definite_source_exclusion)
      .map((row) => ({ ep: ep.execution_provider, row })));
    if (epExclusions.length) {
      const top = epExclusions.slice(0, 8).map(({ ep, row }) => {
        const failures = (row.artifact_conditions || []).filter((condition) => condition.status === "DEFINITE_FAIL");
        return `${ep} #${padOp(row.op_index)} ${row.op_name}: ${failures.map((condition) => `${condition.condition_id} (${condition.observed})`).join(" / ") || row.status}`;
      });
      findings.push(finding({
        id: "EA-ONX-0005",
        category: "execution_provider_compatibility",
        title: "Pinned ORT EP source conditions exclude artifact nodes",
        evidence: "DERIVED",
        priority: "Medium",
        op: `${formatNumber(epExclusions.length)} node-provider exclusion(s)`,
        observation: `${top.join("; ")}${epExclusions.length > top.length ? `; plus ${formatNumber(epExclusions.length - top.length)} more in raw evidence` : ""}.`,
        interpretation: "The operator/schema version is registered, but an artifact-visible dtype, rank, constant-input, output-count, or explicit-attribute value contradicts a pinned source condition. This is a definite source-candidate exclusion, not an observed runtime assignment failure.",
        recommendation: "Review the exact condition ledger, select a compatible EP or artifact contract, and bind an optimization-disabled plus production ORT assignment profile before release.",
        relevance: "execution-provider compatibility; deployment planning; runtime verification",
      }));
    }
  }
  if (!isOnnx && Number(analysis?.subgraphs || 1) > 1) {
    const deep = analysis?.tflite_subgraph_deep_analysis || {};
    findings.push(finding({
      id: "EA-GRF-0001",
      category: "limitation",
      title: "Cross-subgraph execution aggregation requires runtime invocation counts",
      evidence: "OBSERVED/NOT_ASSESSABLE",
      priority: "Medium",
      op: `${formatNumber(deep.assessed_subgraph_count || 0)} of ${formatNumber(analysis.subgraphs)} subgraphs independently assessed`,
      observation: `${formatNumber(analysis.subgraphs)} serialized subgraphs are inventoried and ${formatNumber(deep.assessed_subgraph_count || 0)} receive independent per-op MAC, liveness, ArenaPlanner, quantization, weight-integrity, fixed-point, and source-pinned XNNPACK candidate ledgers. Those local rows are intentionally not summed across control flow.`,
      interpretation: "IF branches are conditional and WHILE, reduce, scatter, or initialization computations can execute zero, one, or multiple times. Serialized nested-op counts therefore cannot be promoted to complete whole-model execution totals without invocation evidence.",
      recommendation: "Use each emitted subgraph row for local review and import observed invocation counts and runtime placement before computing a whole-execution cost, arena, delegation, or latency total.",
      relevance: "analysis coverage; control-flow integrity; deployment verification",
    }));
  }
  if ((analysis?.format || "tflite") === "tflite" && !bundledRuntimeVersion) {
    const runtime = analysis?.runtime_compat || {};
    const mappedFloor = runtime.derived_min_runtime_version || "not determined";
    const completeMap = runtime.runtime_floor_status === "complete_for_observed_builtin_op_versions";
    findings.push(finding({
      id: "EA-RUN-0001",
      category: "integration_verification",
      title: "Bundled runtime version was not provided",
      evidence: "NOT_ASSESSABLE",
      priority: "Medium",
      op: "Runtime",
      observation: `Observed mapped-op necessary floor ${mappedFloor}; builtin op-version map coverage ${formatNumber(runtime.mapped_operator_code_count || 0)}/${formatNumber(runtime.builtin_operator_code_count || 0)} (${runtime.runtime_floor_status || "coverage status not emitted"}); declared min_runtime_version ${runtime.min_runtime_version || "not embedded"}; highest observed operator version ${runtime.max_op_version || 1}; deployed LiteRT/TFLite runtime build not provided.`,
      interpretation: completeMap
        ? "Every observed builtin op code/version is covered by the pinned necessary-floor map, but compatibility of the application bundle still cannot be concluded without the exact runtime version and build flags."
        : "The emitted number is only a partial necessary floor for mapped builtin op versions. Unmapped op versions and the absent application runtime prevent an effective compatibility conclusion.",
      recommendation: "Record the exact LiteRT/TFLite runtime build and build flags, resolve every unmapped builtin op version, and execute the artifact on that binary before claiming compatibility.",
      relevance: "integration verification; deployment reproducibility",
    }));
  }
  const weightIntegrity = analysis?.weight_integrity || {};
  if (Number(weightIntegrity.zero_kernel_slice_count || 0) > 0) {
    const details = Array.isArray(weightIntegrity.zero_kernel_slice_details) ? weightIntegrity.zero_kernel_slice_details : [];
    const quantizedDetails = details.filter((item) => ["INT8", "UINT8"].includes(String(item.dtype || "").toUpperCase()));
    const allDetailsQuantized = details.length > 0 && quantizedDetails.length === details.length;
    const vitalityByOp = new Map((analysis?.channel_vitality?.ops || []).map((row) => [Number(row.op_index), row]));
    const exactConstantProofs = [];
    for (const item of details) {
      const exactChannels = new Set((item.exact_zero_channels || []).map(Number));
      const opIndices = (item.consumer_ops || [])
        .map((label) => Number(String(label).match(/^#(\d+)/)?.[1]))
        .filter(Number.isInteger);
      for (const opIndex of opIndices) {
        const vitality = vitalityByOp.get(opIndex);
        if (!vitality || vitality.assessment_status !== "assessed") continue;
        const defaultConstant = new Set((vitality.default_constant_channel_indices || []).map(Number));
        const singleConstant = new Set((vitality.single_constant_channel_indices || []).map(Number));
        for (const channel of exactChannels) {
          if (!defaultConstant.has(channel) || !singleConstant.has(channel)) continue;
          const sampleIndex = (item.channels || []).findIndex((candidate) => Number(candidate) === channel);
          exactConstantProofs.push({
            tensorIndex: item.tensor_index,
            opIndex,
            channel,
            biasCode: sampleIndex >= 0 ? item.bias_code_sample?.[sampleIndex] : null,
            biasUtilization: sampleIndex >= 0 ? item.bias_int32_utilization_sample?.[sampleIndex] : null,
          });
        }
      }
    }
    const summary = details.length
      ? details.slice(0, 4).map((item) => {
        const channels = (item.channels || []).slice(0, 16).join(", ");
        const exactChannels = (item.exact_zero_channels || []).slice(0, 16).join(", ");
        const scale = (item.scale_sample || []).slice(0, 3).map((v) => Number(v).toExponential(3)).join(", ") || "-";
        const zp = (item.zero_point_sample || []).slice(0, 3).join(", ") || "-";
        const bias = item.bias_tensor_name
          ? `${item.bias_tensor_name} ${item.bias_dtype || ""}; real bias sample ${(item.bias_value_sample || []).slice(0, 3).map((v) => Number(v).toExponential(3)).join(", ") || "not decoded"}; raw code sample ${(item.bias_code_sample || []).slice(0, 3).map((v) => formatNumber(v)).join(", ") || "not decoded"}; INT32 use ${(item.bias_int32_utilization_sample || []).slice(0, 3).map((v) => formatPercent(v)).join(", ") || "not assessed"}`
          : "bias not decoded";
        const quantization = ["INT8", "UINT8"].includes(String(item.dtype || "").toUpperCase())
          ? `; scale/zp ${scale}/${zp}`
          : "";
        const arithmetic = item.zero_slice_arithmetic_share == null
          ? ""
          : `; zero-slice arithmetic proxy ${formatPercent(item.zero_slice_arithmetic_share)}`;
        return `${item.tensor_name || `T${item.tensor_index}`} ${item.dtype || ""} ${(item.shape || item.tensor_shape || []).join("x") || "shape unknown"}: near-zero decoded slice channel(s) ${channels}${Number(item.channel_count || 0) > 16 ? ", ..." : ""}; exact-zero stored subset ${exactChannels || "none"}${quantization}; ${bias}; consumers ${(item.consumer_ops || []).join(" / ") || "-"}${arithmetic}`;
      }).join(" | ")
      : `${formatNumber(weightIntegrity.zero_kernel_slice_count || 0)} near-zero decoded kernel slice(s), including ${formatNumber(weightIntegrity.exact_zero_kernel_slice_count || 0)} exact-zero stored slice(s), across ${formatNumber(weightIntegrity.zero_kernel_slice_tensors || 0)} tensor(s)`;
    const assessedSliceShares = details.filter((item) => item.zero_slice_arithmetic_share != null);
    const sliceMacShare = assessedSliceShares.reduce((sum, item) => sum + Number(item.zero_slice_arithmetic_share), 0);
    const consumerMacUpperBound = details.reduce((sum, item) => sum + Number(item.consumer_mac_percent || 0), 0);
    const numericSpace = allDetailsQuantized
      ? "after applying artifact quantization scale/zero-point metadata"
      : "in decoded stored-value space";
    const arithmeticImpact = assessedSliceShares.length
      ? ` Static zero-slice arithmetic-waste proxy ${formatPercent(sliceMacShare)} of assessed model MACs (direct-consumer MACs x flagged output slices / kernel output channels; not latency or accuracy impact).`
      : ` Direct-consumer MAC share upper bound ${formatPercent(consumerMacUpperBound)} (triage scope only; not zero-slice arithmetic, latency, or accuracy impact).`;
    findings.push(finding({
      id: "EA-WGT-0001",
      category: "numerical_structure_review",
      title: exactConstantProofs.length
        ? "Exact-zero kernel slices converge with constant-output channel proof"
        : "Near-zero decoded kernel slices require functional review",
      evidence: exactConstantProofs.length ? "DERIVED" : "OBSERVED",
      priority: exactConstantProofs.length
        ? "High"
        : Number(weightIntegrity.exact_zero_kernel_slice_count || 0) ? "Medium" : "Informational",
      op: details.flatMap((item) => item.consumer_ops || []).slice(0, 4).join(", ") || "constant tensor",
      tensor: details.slice(0, 4).map((item) => item.tensor_name || `T${item.tensor_index}`).join(", ") || "-",
      observation: `${formatNumber(weightIntegrity.zero_kernel_slice_count || 0)} decoded kernel output slice(s) are near-zero ${numericSpace}; ${formatNumber(weightIntegrity.exact_zero_kernel_slice_count || 0)} are exact-zero in stored centered-code space. ${summary}.${arithmeticImpact}${exactConstantProofs.length ? ` Cross-ledger join proves ${formatNumber(exactConstantProofs.length)} exact-zero coordinate(s) also emit one constant output code under both pinned fixed-point paths: ${exactConstantProofs.slice(0, 12).map((row) => `T${row.tensorIndex}/#${padOp(row.opIndex)}/ch${row.channel}${row.biasCode == null ? "" : ` bias ${formatNumber(row.biasCode)} (${formatPercent(row.biasUtilization)} INT32)`}`).join(", ")}.` : ""}`,
      interpretation: exactConstantProofs.length
        ? "The listed coordinates are exact local constant-output channels under both pinned TFLite fixed-point paths, not merely low-norm or scale-spread heuristics. This still does not prove the declared model outputs or task metric are invariant because downstream graph paths and representative inputs remain separate evidence."
        : "Functional channel inactivity is NOT_ASSESSABLE from kernel weights alone because bias, fused activation, residual paths, downstream consumers, and task behavior can still carry or transform signal.",
      recommendation: exactConstantProofs.length
        ? "Treat the convergent coordinates as a high-priority export defect review: compare against the source checkpoint/QAT graph, inspect raw INT32 bias generation, re-export, and run representative plus adversarial output regression before release."
        : "Classify as a defect only when kernel slice exact-zero, channel-output proof, downstream effect, and representative-output regression are joined. Until then, review bias values, fused activation, residual/downstream consumers, converter lineage, and representative regression.",
      relevance: allDetailsQuantized ? "quantized weight structure; numerical structure review" : "weight numerical structure review",
      findingClass: exactConstantProofs.length ? "exact cross-ledger numerical proof" : "",
      evidenceJsonPointers: [
        "/evidence/static_analysis/weight_integrity/zero_kernel_slice_details",
        ...(exactConstantProofs.length ? ["/evidence/static_analysis/channel_vitality/ops"] : []),
      ],
    }));
  }
  if (Number(weightIntegrity.low_grid_utilization_tensors || 0) > 0 || Number(weightIntegrity.saturated_quantized_tensors || 0) > 0) {
    findings.push(finding({
      id: "EA-QNT-0100",
      category: "quantization_calibration_review",
      title: "Quantized constant grid utilization requires calibration review",
      evidence: "HEURISTIC",
      priority: Number(weightIntegrity.saturated_quantized_tensors || 0) > 0 ? "High" : "Medium",
      op: "constant tensors",
      observation: `${weightIntegrity.quant_grid_detail || `${formatNumber(weightIntegrity.low_grid_utilization_tensors || 0)} low-utilization tensor(s), ${formatNumber(weightIntegrity.saturated_quantized_tensors || 0)} saturated tensor(s).`} Alert threshold is HEURISTIC: <25% 8-bit level utilization for tensors with at least 256 elements, or >1% endpoint saturation.`,
      interpretation: "The utilization and saturation values are deterministically derived from artifact constants; the alert threshold is a methodology heuristic for calibration review, not a task-accuracy claim.",
      recommendation: "Review calibration/QAT histograms for the flagged constants and compare representative-output drift against a re-calibrated export.",
      relevance: "quantization calibration integrity",
      evidenceJsonPointers: [
        ...(Object.prototype.hasOwnProperty.call(weightIntegrity, "quant_grid_detail")
          ? ["/evidence/static_analysis/weight_integrity/quant_grid_detail"] : []),
        ...(Array.isArray(weightIntegrity.quant_grid_details)
          ? ["/evidence/static_analysis/weight_integrity/quant_grid_details"] : []),
      ],
    }));
  }
  const representableChannels = quantizationContracts?.representable_kernel_channels;
  if (Number(representableChannels?.flagged_channels || 0) > 0) {
    const flagged = (representableChannels.details || [])
      .filter((row) => Number(row.flagged_channel_count || 0) > 0);
    const summary = flagged.slice(0, 6).map((row) => {
      const channelSample = (row.flagged_channels || []).slice(0, 8);
      const smallestRange = channelSample.length
        ? Math.min(...channelSample.map((channel) => Number(channel.maximum_representable_abs)))
        : 0;
      return `${row.tensor_name || `T${row.tensor_index}`}: ${formatNumber(row.flagged_channel_count)} channel(s), indices ${channelSample.map((channel) => channel.channel).join(", ") || "not emitted"}, minimum sampled max |real weight| ${formatScientific(smallestRange)}, scale spread ${formatScientific(row.maximum_to_minimum_scale_ratio)}x`;
    }).join(" | ");
    findings.push(finding({
      id: "EA-QNT-0118",
      category: "quantization_design_review",
      title: "Near-zero representable kernel channels require QAT review",
      evidence: "HEURISTIC",
      priority: "Medium",
      op: flagged.flatMap((row) => row.consumer_ops || []).slice(0, 6).join(", ") || "quantized compute kernels",
      tensor: flagged.slice(0, 6).map((row) => row.tensor_name || `T${row.tensor_index}`).join(", "),
      observation: `${formatNumber(representableChannels.flagged_channels)} of ${formatNumber(representableChannels.assessed_channels)} assessed per-axis kernel channel(s) meet both review thresholds: maximum representable |real weight| <=${formatScientific(representableChannels.maximum_representable_abs_threshold)} and scale outlier ratio >=${formatScientific(representableChannels.scale_outlier_ratio_threshold)}x. ${summary}.`,
      interpretation: "The code-domain range and scale ratios are deterministic artifact-derived values. The thresholds are methodology heuristics: they identify numerically collapsed relative channels, but do not by themselves prove a dead activation channel, QAT failure, or task-accuracy loss because bias, fan-in, fused activation, residual paths, and downstream behavior remain unassessed.",
      recommendation: "Inspect the named QAT/calibration layer histograms, verify the matching bias-scale contract, and compare representative outputs against a re-export with repaired per-axis ranges before classifying the channel as functionally dead.",
      relevance: "per-axis quantization range integrity; QAT calibration review",
      evidenceJsonPointers: FINDING_EVIDENCE_PATHS["EA-QNT-0118"],
    }));
  }
  const inputConvention = quantizationContracts?.input_quantization_convention;
  const unmatchedInputConventions = (inputConvention?.details || [])
    .filter((item) => item.status === "no_common_full_domain_match");
  if (unmatchedInputConventions.length) {
    const examples = unmatchedInputConventions.slice(0, 4).map((item) => {
      const closest = item.closest_convention;
      return `Input ${item.ordinal} ${item.dtype} scale ${Number(item.scale).toExponential(5)}, zp ${item.zero_point}, exact full-code range [${item.real_range.map((value) => Number(value).toExponential(5)).join(", ")}]; closest ${closest?.label || "none"} endpoint error ${closest ? Number(closest.maximum_endpoint_error).toExponential(5) : "N/A"}`;
    }).join(" | ");
    findings.push(finding({
      id: "EA-QNT-0119",
      category: "integration_verification",
      title: "Input quantization range does not match a common full-domain preprocessing convention",
      evidence: "DERIVED_WITH_REFERENCE_HEURISTIC",
      priority: "Medium",
      op: `${formatNumber(unmatchedInputConventions.length)} quantized model input(s)`,
      observation: `${examples}. The real ranges are exactly reconstructed from artifact scale/zero-point metadata; the convention comparison uses a one-quantization-step endpoint tolerance.`,
      interpretation: "This does not prove that calibration or application preprocessing is wrong: a custom asymmetric range can be intentional. It does prove that [0,1], [-1,1], [0,255], and applicable signed-code identity conventions cannot explain the full declared code domain within the disclosed tolerance.",
      recommendation: "Bind the production decoder-to-tensor preprocessing specification and checksum, state the intended real input domain explicitly, and replay representative tensors to confirm that application quantization uses the artifact scale and zero-point.",
      relevance: "input numerical contract; calibration lineage; deployment reproducibility",
    }));
  }
  const perTensorDepthwise = depthwisePerTensorQuantOps(analysis);
  if (perTensorDepthwise.length && Number(analysis?.per_channel_tensors || 0) === 0) {
    const affectedOps = perTensorDepthwise.map((item) => `#${padOp(item.op.index)} ${item.op.name}`).join(", ");
    const affectedTensors = [...new Set(perTensorDepthwise.map((item) => item.weight?.name || `T${item.op.inputs?.[1]}`))].join(", ");
    findings.push(finding({
      id: "EA-QNT-0101",
      category: "quantization_design_review",
      title: "Depthwise convolution weights use per-tensor quantization",
      evidence: "DERIVED",
      priority: "Medium",
      op: affectedOps,
      tensor: affectedTensors,
      observation: `${formatNumber(perTensorDepthwise.length)} quantized DEPTHWISE_CONV_2D op(s) use a single weight scale; per-channel tensors observed: ${formatNumber(analysis?.per_channel_tensors || 0)}. Complete affected-op inventory: ${affectedOps}.`,
      interpretation: "Per-tensor depthwise weight quantization is a design-review signal for mobile CNNs because channel distributions are forced onto one shared scale; it is not a confirmed defect without representative accuracy or calibration evidence.",
      recommendation: "Prefer per-channel/per-axis weight quantization for depthwise kernels where the target runtime supports it. Escalate to High only when accuracy regression, calibration error, or a product requirement is present.",
      relevance: "quantization configuration; converter lineage",
    }));
  }
  const biasMismatches = (quantizationContracts.bias_scale?.details || []).filter((item) => item.status === "fail");
  const onnxParameterIssues = (quantizationContracts.parameter_integrity?.details || []).filter((item) => item.status !== "pass");
  if (onnxParameterIssues.length) {
    const failures = onnxParameterIssues.filter((item) => item.status === "fail");
    const top = failures[0] || onnxParameterIssues[0];
    findings.push(finding({
      id: "EA-QNT-0105",
      category: failures.length ? "integrity" : "quantization",
      title: failures.length ? "ONNX quantization parameter contract is invalid" : "ONNX quantization parameters could not be fully assessed",
      evidence: "DERIVED",
      priority: failures.length ? "High" : "Medium",
      op: `#${padOp(top.op_index)} ${top.op_name}`,
      tensor: top.tensor_name || "-",
      observation: `${formatNumber(failures.length)} invalid and ${formatNumber(onnxParameterIssues.length - failures.length)} unresolved Q/DQ or QLinear parameter binding(s). Representative ${top.role}: ${(top.reasons || [top.status]).join(", ")}.`,
      interpretation: failures.length
        ? "Embedded scale, zero-point, axis, dtype, or cardinality values violate the selected ONNX standard operator contract."
        : "The artifact references quantization parameters that are missing, external, unsupported, or not shape-resolvable in this browser-only input set.",
      recommendation: failures.length
        ? "Block release of this artifact until the ONNX quantization parameters are regenerated or independently corrected and output parity is verified."
        : "Supply the external initializer data or regenerate a self-contained artifact, then rerun the deterministic contract check.",
      relevance: "ONNX quantization numerical contract",
    }));
  }
  if (biasMismatches.length) {
    const top = biasMismatches[0];
    findings.push(finding({
      id: "EA-QNT-0102",
      category: "quantization",
      title: "Bias quantization scale does not match input-scale times weight-scale",
      evidence: "DERIVED",
      priority: "High",
      op: `#${padOp(top.op_index)} ${top.op_name}`,
      tensor: `T${top.bias_tensor_index}`,
      observation: quantizationContracts.bias_scale.contract_kind === "onnx_implicit_qlinearconv_bias_scale"
        ? `${formatNumber(biasMismatches.length)} QLinearConv bias group(s) violate the INT32 dtype or output-channel cardinality contract. ${top.reasons?.join(", ") || "invalid signature"}.`
        : `${formatNumber(biasMismatches.length)} bias tensor group(s) violate the TFLite bias-scale contract at relative tolerance ${quantizationContracts.bias_scale.relative_tolerance}; largest relative error ${formatPercent(top.maximum_relative_error)} (channel ${top.worst_channel?.channel}: actual=${Number(top.worst_channel?.actual || 0).toExponential(4)}, expected=${Number(top.worst_channel?.expected || 0).toExponential(4)}).`,
      interpretation: quantizationContracts.bias_scale.contract_kind === "onnx_implicit_qlinearconv_bias_scale"
        ? "ONNX QLinearConv requires INT32 bias with one element per output channel; its real-value scale is implicitly x_scale times w_scale."
        : "For quantized TFLite conv/FC kernels, bias scale must match input_scale x weight_scale. A mismatch can silently change arithmetic meaning.",
      recommendation: quantizationContracts.bias_scale.contract_kind === "onnx_implicit_qlinearconv_bias_scale"
        ? "Regenerate the QLinearConv node with an INT32 bias of exactly one element per output channel, then verify reference-output parity."
        : "Regenerate the artifact with a converter that preserves the TFLite bias quantization contract, or block release until runtime behavior is verified against reference outputs.",
      relevance: "quantization numerical contract",
    }));
  }
  const addMismatches = (quantizationContracts.residual_add?.details || [])
    .filter((item) => item.status === "review")
    .sort((left, right) => Number(right.input_scale_ratio || 0) - Number(left.input_scale_ratio || 0) || Number(left.op_index || 0) - Number(right.op_index || 0));
  if (addMismatches.length) {
    const top = addMismatches[0];
    findings.push(finding({
      id: "EA-QNT-0103",
      category: "quantization",
      title: "Residual ADD input scales require requantization review",
      evidence: "DERIVED",
      priority: top.input_scale_ratio >= 8 ? "High" : "Medium",
      op: `#${padOp(top.op_index)} ADD`,
      observation: `${formatNumber(addMismatches.length)} ADD op(s) have input scale ratio >=${quantizationContracts.residual_add.review_threshold_ratio}. Largest ratio ${top.input_scale_ratio.toFixed(2)}x; input scales ${top.input_scales.map((value) => value.toExponential(4)).join(" / ")}; output scale ${top.output_scale == null ? "none" : top.output_scale.toExponential(4)}.${top.exhaustive_legal_code_pair_count == null ? "" : ` Exhaustive legal-code geometry: ${formatNumber(top.legal_domain_escape_pair_count)} / ${formatNumber(top.exhaustive_legal_code_pair_count)} endpoint escapes; globally finest containment ${Number(top.globally_finest_containment_scale_ratio).toFixed(3)}x scale with zero-point delta ${Number(top.globally_finest_containment_zero_point_delta) >= 0 ? "+" : ""}${top.globally_finest_containment_zero_point_delta} and ${formatNumber(top.globally_finest_containment_clamp_pair_count)} projected clamps.`}`,
      interpretation: "Residual merges with widely different scales can force requantization and reduce precision on one branch. The containment design is exhaustive uniform legal-code geometry, not an observed activation or accuracy result.",
      recommendation: "Inspect QAT/calibration for the residual branch pair and compare representative outputs. Treat the emitted containment contract only as a re-export counterfactual; changing output quantization requires regenerating and validating downstream contracts rather than editing this tensor in isolation.",
      relevance: "quantized residual arithmetic",
    }));
  }
  const lattice = analysis?.quantization_lattice || null;
  const latticeRows = new Map((lattice?.residual_adds || []).map((row) => [Number(row.op_index), row]));
  const containmentReview = (lattice?.domain_escape_ranking_op_indices || [])
    .map((index) => latticeRows.get(Number(index)))
    .find((row) => row?.assessment_status === "assessed" && Number(row.range_escape_pair_count || 0) > 0 && row.globally_finest_containment);
  if (containmentReview) {
    const global = containmentReview.globally_finest_containment;
    const fixed = containmentReview.fixed_zero_point_containment;
    const migrationRow = (analysis?.contract_migration?.migrations || [])
      .find((row) => Number(row.source_add_op_index) === Number(containmentReview.op_index));
    const migrationScenario = migrationRow?.scenarios?.find((scenario) => scenario.design === "globally_finest_minimum_containment");
    const migrationDetail = migrationScenario
      ? ` Re-export impact is deterministically scoped to ${formatNumber(migrationRow.direct_consumer_count)} direct parameter consumer(s): ${formatNumber(migrationScenario.assessed_kernel_channel_count)} kernel channel(s), ${formatNumber(migrationScenario.multiplier_encoding_changed_channel_count)} changed Q0.31 encoding(s), ${formatNumber(migrationScenario.multiplier_shift_changed_channel_count)} changed shift(s), ${formatNumber(migrationScenario.bias_code_changed_channel_count)} changed bias code(s), ${formatNumber(migrationScenario.bias_int32_overflow_channel_count)} candidate INT32 bias overflow(s), and ${formatNumber(migrationScenario.add_parameter_encoding_changed_count)} changed direct-ADD parameter encoding(s). Its structural behavior radius is ${formatNumber(migrationRow.reachable_downstream_op_count)} reachable op(s) at maximum edge depth ${formatNumber(migrationRow.maximum_downstream_edge_depth)}.`
      : " Direct-consumer parameter regeneration was not assessed.";
    findings.push(finding({
      id: "EA-QNT-0108",
      category: "quantization_design_review",
      title: "Residual ADD output contract does not contain the complete legal code domain",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(containmentReview.op_index)} ADD`,
      tensor: Number.isInteger(containmentReview.output_tensor_index) ? `T${containmentReview.output_tensor_index}` : "quantized ADD output",
      observation: `${formatNumber(lattice.range_escape_add_count)} / ${formatNumber(lattice.assessed_add_count)} assessed ADD output contract(s) exclude part of the uniform legal input-code sum domain. Top #${padOp(containmentReview.op_index)} has ${formatNumber(containmentReview.range_escape_pair_count)} / ${formatNumber(containmentReview.enumerated_code_pair_count)} endpoint escapes and ${formatNumber(containmentReview.rounded_projection_clamp_pair_count)} ideal projected clamps. Fixed-zero-point containment is ${fixed ? `${Number(fixed.scale_ratio_to_current).toFixed(3)}x scale with ${formatNumber(fixed.rounded_projection_clamp_pair_count)} clamps` : "unavailable"}; the globally finest full-containment contract is ${Number(global.scale_ratio_to_current).toFixed(3)}x scale with zero-point delta ${Number(global.signed_zero_point_delta) >= 0 ? "+" : ""}${global.signed_zero_point_delta} and ${formatNumber(global.rounded_projection_clamp_pair_count)} clamps.${migrationDetail}`,
      interpretation: "The counterfactual is exhaustive over uniform legal integer-code pairs. It proves output-contract geometry, not activation frequency, calibration quality, task accuracy, or executed fixed-point behavior.",
      recommendation: "Use the containment frontier and Contract Migration Impact ledger to scope a calibration or QAT re-export experiment. Regenerate the listed direct-consumer Q0.31 parameters and INT32 bias constants, verify every channel ledger, then validate representative and adversarial outputs. Do not patch the output tensor alone; reachable downstream ops are a behavior-impact radius, not a claim that every one needs metadata replacement.",
      relevance: "residual output quantization contract design",
      methodVersion: lattice.method_version,
    }));
  }
  const stepResponse = analysis?.residual_step_response;
  const stepTopIndex = stepResponse?.retention_cost_ranking_op_indices?.[0];
  const stepTop = (stepResponse?.residual_adds || []).find((row) => row.op_index === stepTopIndex);
  const stepCurrent = stepTop?.contracts?.find((contract) => contract.design === "current_artifact_contract");
  const stepGlobal = stepTop?.contracts?.find((contract) => contract.design === "globally_finest_minimum_containment");
  if (stepTop && stepCurrent && stepGlobal) {
    findings.push(finding({
      id: "EA-QNT-0109",
      category: "quantization_design_review",
      title: "Residual containment has an exact local step-visibility trade-off",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(stepTop.op_index)} ADD`,
      tensor: Number.isInteger(stepTop.output_tensor_index) ? `T${stepTop.output_tensor_index}` : "quantized ADD output",
      observation: `Across ${formatNumber(stepResponse.total_transition_count)} exhaustive adjacent branch transitions and ${formatNumber(stepResponse.total_joint_interior_cell_count)} joint interior cells, both containment candidates remove ${formatNumber(stepResponse.containment_removed_rounded_clamp_pair_count)} rounded clamp-pair instances in aggregate and add ${formatNumber(stepResponse.containment_additional_silent_transition_count)} silent transition instances versus their current contracts. Highest retention cost is #${padOp(stepTop.op_index)}: current has ${formatNumber(stepCurrent.rounded_projection_clamp_pair_count)} clamp pairs and ${formatNumber(stepCurrent.silent_transition_count)} silent transitions; globally finest containment has ${formatNumber(stepGlobal.rounded_projection_clamp_pair_count)} and ${formatNumber(stepGlobal.silent_transition_count)}, respectively (${stepGlobal.additional_silent_transitions_vs_current >= 0 ? "+" : ""}${formatNumber(stepGlobal.additional_silent_transitions_vs_current)} silent). Both-branch-visible interior cells change from ${formatNumber(stepCurrent.both_branches_visible_cell_count)} to ${formatNumber(stepGlobal.both_branches_visible_cell_count)}.`,
      interpretation: "This is exact local distinguishability over uniformly enumerated legal integer codes. It is not an activation distribution, entropy or mutual-information estimate, runtime branch-activity measurement, calibration verdict, or task-accuracy result.",
      recommendation: "Use the Residual Step Response ledger alongside domain containment and Contract Migration Impact when designing a calibration or QAT re-export experiment. Compare clamp removal, branch-specific silent transitions, and joint influence fields, then validate representative and adversarial task outputs before selecting a contract.",
      relevance: "residual output-contract local code-step geometry",
      methodVersion: stepResponse.method_version,
    }));
  }
  const distortion = analysis?.residual_contract_distortion;
  const distortionTopIndex = distortion?.distortion_ranking_op_indices?.[0];
  const distortionTop = (distortion?.residual_adds || []).find((row) => row.op_index === distortionTopIndex);
  const distortionGlobal = distortionTop?.scenarios?.find((scenario) => scenario.design === "globally_finest_minimum_containment");
  if (distortionTop && distortionGlobal) {
    findings.push(finding({
      id: "EA-QNT-0110",
      category: "quantization_design_review",
      title: "Clamp containment redistributes residual projection error across the legal code domain",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(distortionTop.op_index)} ADD`,
      tensor: Number.isInteger(distortionTop.output_tensor_index) ? `T${distortionTop.output_tensor_index}` : "quantized ADD output",
      observation: `Across ${formatNumber(distortion.total_enumerated_pair_count)} exact current-versus-containment pair comparisons, the candidates rescue ${formatNumber(distortion.rescued_current_clamp_pair_instance_count)} current rounded-clamp instances and retain ${formatNumber(distortion.candidate_clamped_pair_count)} candidate clamps. Ideal projection error decreases for ${formatNumber(distortion.ideal_error_improved_pair_count)} pairs, increases for ${formatNumber(distortion.ideal_error_worsened_pair_count)}, and is equal within the declared binary64 tolerance for ${formatNumber(distortion.ideal_error_equal_within_tolerance_pair_count)}; ${formatNumber(distortion.sign_class_changed_pair_count)} represented-value sign classes change. Highest RMS displacement is #${padOp(distortionTop.op_index)}. Its globally finest candidate rescues ${formatNumber(distortionGlobal.rescued_current_clamp_pair_count)} clamps, changes ${formatNumber(distortionGlobal.changed_represented_value_pair_count)} represented values, improves / worsens ${formatNumber(distortionGlobal.ideal_error_improved_pair_count)} / ${formatNumber(distortionGlobal.ideal_error_worsened_pair_count)} pairs, and has RMS / p99 / maximum displacement ${Number(distortionGlobal.root_mean_square_contract_delta_current_steps).toFixed(3)} / ${Number(distortionGlobal.p99_absolute_contract_delta_current_steps).toFixed(3)} / ${Number(distortionGlobal.maximum_absolute_contract_delta_current_steps).toFixed(3)} current output steps. Its uniform mean absolute ideal-projection error delta is ${Number(distortionGlobal.signed_mean_absolute_ideal_error_delta).toExponential(4)} real units.`,
      interpretation: "The candidates remove rare, large clamp errors while introducing smaller rounding changes over many interior code pairs. Pair counts are uniform-domain geometry, not activation probabilities; neither the lower mean error nor the larger worsened-pair count establishes task quality.",
      recommendation: "Use the Distortion Atlas with the Step Response and Contract Migration ledgers to choose re-export experiments. Validate representative and adversarial task outputs, especially around sign-class changes and the reported worst pair witnesses, before selecting a containment contract.",
      relevance: "residual output-contract counterfactual distortion geometry",
      methodVersion: distortion.method_version,
    }));
  }
  const kernelWitness = analysis?.kernel_extremum_witness;
  const witnessTopIndex = kernelWitness?.witness_ranking_op_indices?.[0];
  const witnessTop = (kernelWitness?.ops || []).find((row) => row.op_index === witnessTopIndex);
  const witnessChannel = witnessTop?.worst_channel || witnessTop?.top_channels?.[0];
  if (Number(kernelWitness?.build_mode_divergent_endpoint_count || 0) > 0 && witnessTop && witnessChannel) {
    const endpoint = [witnessChannel.minimum, witnessChannel.maximum]
      .find((item) => Number(item?.build_mode_output_delta_codes || 0) !== 0)
      || witnessChannel.maximum;
    findings.push(finding({
      id: "EA-QNT-0111",
      category: "quantization_design_review",
      title: "Pinned TFLite fixed-point build modes produce concrete endpoint code differences",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(witnessTop.op_index)} ${witnessTop.op_name}`,
      tensor: `${witnessTop.output_tensor_name || `T${witnessTop.output_tensor_index}`} channel ${witnessChannel.channel_index}`,
      observation: `${formatNumber(kernelWitness.witness_assignment_count)} canonical legal-code term assignments and ${formatNumber(kernelWitness.fixed_point_endpoint_evaluation_count)} pinned fixed-point endpoint executions expose ${formatNumber(kernelWitness.build_mode_divergent_endpoint_count)} default-versus-TFLITE_SINGLE_ROUNDING output-code difference(s). Default versus direct ideal differs at ${formatNumber(kernelWitness.default_ideal_mismatch_endpoint_count)} endpoint(s); single-rounding versus direct ideal differs at ${formatNumber(kernelWitness.single_ideal_mismatch_endpoint_count)}. The maximum absolute deltas are ${formatNumber(kernelWitness.maximum_default_ideal_delta_codes || 0)} and ${formatNumber(kernelWitness.maximum_single_ideal_delta_codes || 0)} output code(s), respectively. Reproducible witness: #${padOp(witnessTop.op_index)} channel ${witnessChannel.channel_index} ${endpoint.endpoint}; ideal/default/single output codes ${formatNumber(endpoint.ideal_output_code)} / ${formatNumber(endpoint.default_output_code)} / ${formatNumber(endpoint.single_output_code)}; pattern SHA-256 ${witnessChannel.witness_pattern_sha256}.`,
      interpretation: "The artifact does not embed the runtime compile-time rounding mode. These are exact synthetic full-valid receptive-field endpoints under pinned TensorFlow source semantics, not an observed runtime mismatch, an activation frequency, a full-model input, or an accuracy estimate.",
      recommendation: "Bind the deployed runtime commit, binary hash, compile flags, and TFLITE_SINGLE_ROUNDING state to the release. Replay the exported Witness JSON against the target runtime and treat the observed target result as authoritative for that build.",
      relevance: "quantized kernel reproducibility and runtime-build provenance",
      methodVersion: kernelWitness.method_version,
    }));
  }
  const vitality = analysis?.channel_vitality;
  const vitalityTop = (vitality?.vitality_ranking_op_indices || [])
    .map((opIndex) => (vitality?.ops || []).find((row) => row.op_index === opIndex))
    .find((row) => Number(row?.nonconstant_accumulator_dual_mode_constant_channel_count || 0) > 0);
  if (Number(vitality?.nonconstant_accumulator_dual_mode_constant_channel_count || 0) > 0 && vitalityTop) {
    const variableConstant = (vitalityTop.top_channels || []).filter((channel) => (
      BigInt(channel.accumulator_span_decimal || "0") !== 0n && channel.dual_mode_constant
    ));
    const modeDependent = (vitalityTop.top_channels || []).filter((channel) => channel.mode_dependent_constant);
    const coordinateText = (channels) => channels.length
      ? channels.map((channel) => `ch ${channel.channel_index}: post-bias ${channel.post_bias_minimum_decimal}..${channel.post_bias_maximum_decimal}, default preclamp ${channel.default_minimum_preclamp_code}..${channel.default_maximum_preclamp_code} -> output ${channel.default_minimum_output_code}..${channel.default_maximum_output_code} (${channel.default_constant_reason}), single preclamp ${channel.single_minimum_preclamp_code}..${channel.single_maximum_preclamp_code} -> output ${channel.single_minimum_output_code}..${channel.single_maximum_output_code} (${channel.single_constant_reason})`).join(" / ")
      : "coordinates retained in the structured vitality ledger";
    const graphOp = (artifactIrOperators(analysis) || []).find((op) => Number(op.index) === Number(vitalityTop.op_index));
    const constantRows = (vitality.ops || []).filter((row) => Number(row.dual_mode_constant_output_channel_count || 0) > 0);
    const constantCoordinates = constantRows.map((row) => {
      const single = new Set(row.single_constant_channel_indices || []);
      const indices = (row.default_constant_channel_indices || []).filter((index) => single.has(index));
      return `#${padOp(row.op_index)} ${row.op_name} [${indices.join(", ") || "indices unavailable"}]`;
    });
    const affectedOps = constantRows.map((row) => `#${padOp(row.op_index)} ${row.op_name}`).join(", ");
    const affectedTensors = constantRows.map((row) => {
      const op = (artifactIrOperators(analysis) || []).find((candidate) => Number(candidate.index) === Number(row.op_index));
      return Number.isInteger(Number(op?.outputs?.[0])) ? `T${Number(op.outputs[0])}` : null;
    }).filter(Boolean).join(", ");
    findings.push(finding({
      id: "EA-QNT-0112",
      category: "quantization_design_review",
      title: "Legal input-code domain cannot activate quantized output channels",
      evidence: "DERIVED",
      priority: "High",
      op: affectedOps || `#${padOp(vitalityTop.op_index)} ${vitalityTop.op_name}`,
      tensor: affectedTensors || (Number.isInteger(Number(graphOp?.outputs?.[0])) ? `T${Number(graphOp.outputs[0])}` : "quantized kernel output"),
      observation: `${formatNumber(vitality.assessed_channel_count)} channels were joined to exact stored-weight post-bias endpoints and both pinned TFLite fixed-point paths. ${formatNumber(vitality.dual_mode_constant_output_channel_count)} channel(s) are constant under both paths: ${formatNumber(vitality.constant_accumulator_channel_count)} have a constant post-bias accumulator and ${formatNumber(vitality.nonconstant_accumulator_dual_mode_constant_channel_count)} retain a variable accumulator yet collapse to one output code. Complete dual-mode constant-channel inventory: ${constantCoordinates.join(" / ")}. Highest-ranked detailed coordinates at #${padOp(vitalityTop.op_index)}: ${coordinateText(variableConstant)}. ${formatNumber(vitality.mode_dependent_constant_output_channel_count)} channel(s) change constant classification with TFLITE_SINGLE_ROUNDING; ${coordinateText(modeDependent)}. Op vitality ledger SHA-256 ${vitalityTop.vitality_ledger_sha256}.`,
      interpretation: "Equal endpoint outputs are an exact consequence of monotone positive-multiplier projection and clamp over the full-valid receptive-field legal-code domain. This is not observed runtime activation frequency, a full-model reachability claim, edge-padding behavior, delegate execution evidence, or a task-accuracy result.",
      recommendation: "Inspect the listed channels in the QAT or conversion graph, calibration ranges, bias, and activation contract. Re-export candidate artifacts, replay the selected channel proofs against the pinned runtime build, and compare representative plus adversarial task outputs before release.",
      relevance: "quantized channel vitality and fixed-point projection collapse",
      methodVersion: vitality.method_version,
    }));
  }
  const roundingEquivalence = analysis?.rounding_equivalence;
  const equivalenceTop = (roundingEquivalence?.equivalence_ranking_op_indices || [])
    .map((opIndex) => (roundingEquivalence?.ops || []).find((row) => row.op_index === opIndex))
    .find((row) => Number(row?.divergent_channel_count || 0) > 0);
  const equivalenceChannel = equivalenceTop?.top_channels?.find((channel) => BigInt(channel.divergent_state_count_decimal || "0") > 0n);
  if (Number(roundingEquivalence?.divergent_channel_count || 0) > 0 && equivalenceTop && equivalenceChannel) {
    const graphOp = (artifactIrOperators(analysis) || []).find((op) => Number(op.index) === Number(equivalenceTop.op_index));
    findings.push(finding({
      id: "EA-QNT-0113",
      category: "quantization_design_review",
      title: "Pinned TFLite rounding build modes are not bit-exact over the accumulator interval hull",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(equivalenceTop.op_index)} ${equivalenceTop.op_name}`,
      tensor: Number.isInteger(Number(graphOp?.outputs?.[0])) ? `T${Number(graphOp.outputs[0])} channel ${equivalenceChannel.channel_index}` : `quantized kernel output channel ${equivalenceChannel.channel_index}`,
      observation: `${formatNumber(roundingEquivalence.assessed_channel_count)} channels and ${formatNumber(roundingEquivalence.interval_state_count_decimal)} post-bias integer interval states were partitioned exactly into ${formatNumber(roundingEquivalence.pair_segment_count)} maximal ordered-output-pair segments. ${formatNumber(roundingEquivalence.divergent_channel_count)} channel(s) across ${formatNumber(roundingEquivalence.divergent_op_count)} op(s) contain ${formatNumber(roundingEquivalence.divergent_state_count_decimal)} default-versus-TFLITE_SINGLE_ROUNDING state difference(s) (${formatPercent(roundingEquivalence.divergent_state_ratio)} of the interval-hull portfolio); ${formatPercent(Number(roundingEquivalence.default_higher_state_count_decimal || 0) / Math.max(1, Number(roundingEquivalence.divergent_state_count_decimal || 0)))} of divergent states are default-higher and ${formatPercent(Number(roundingEquivalence.default_lower_state_count_decimal || 0) / Math.max(1, Number(roundingEquivalence.divergent_state_count_decimal || 0)))} are default-lower; maximum absolute delta ${formatNumber(roundingEquivalence.maximum_absolute_output_delta)} output code. First ranked counterexample: #${padOp(equivalenceTop.op_index)} channel ${equivalenceChannel.channel_index}, accumulator ${equivalenceChannel.first_divergent_accumulator_decimal}, default/single outputs ${equivalenceChannel.first_default_output_code}/${equivalenceChannel.first_single_output_code}; last counterexample ${equivalenceChannel.last_divergent_accumulator_decimal}; op certificate SHA-256 ${equivalenceTop.equivalence_ledger_sha256}.`,
      interpretation: "This is an exact build-mode equivalence certificate over each closed post-bias int32 interval hull. Interior accumulator integers can be unreachable because legal dot products are discrete and correlated, so the state ratio is not an activation probability, observed mismatch rate, or accuracy estimate. The artifact does not embed the runtime compile-time rounding mode.",
      recommendation: "Bind the TensorFlow/LiteRT source commit, runtime binary hash, compile flags, and TFLITE_SINGLE_ROUNDING state to the release. Replay the exported first counterexample and representative full-model inputs on every target build when bit-exact reproducibility is required.",
      relevance: "quantized bit-exact reproducibility and runtime-build provenance",
      methodVersion: roundingEquivalence.method_version,
    }));
  }
  const accumulatorReachability = analysis?.accumulator_reachability;
  const reachabilityTop = (accumulatorReachability?.reachability_ranking_op_indices || [])
    .map((opIndex) => (accumulatorReachability?.ops || []).find((row) => row.op_index === opIndex))
    .find((row) => BigInt(row?.exact_reachable_divergent_state_count_decimal || "0") > 0n);
  const reachabilityChannel = reachabilityTop?.top_channels?.find((channel) => BigInt(channel.exact_reachable_divergent_state_count_decimal || "0") > 0n);
  if (BigInt(accumulatorReachability?.exact_reachable_divergent_state_count_decimal || "0") > 0n && reachabilityTop && reachabilityChannel) {
    const graphOp = (artifactIrOperators(analysis) || []).find((op) => Number(op.index) === Number(reachabilityTop.op_index));
    findings.push(finding({
      id: "EA-QNT-0115",
      category: "quantization_design_review",
      title: "Pinned rounding build modes differ at constructively reachable kernel-local accumulator states",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(reachabilityTop.op_index)} ${reachabilityTop.op_name}`,
      tensor: Number.isInteger(Number(graphOp?.outputs?.[0])) ? `T${Number(graphOp.outputs[0])} channel ${reachabilityChannel.channel_index}` : `quantized kernel output channel ${reachabilityChannel.channel_index}`,
      observation: `${formatNumber(accumulatorReachability.assessed_channel_count)} channels were decomposed into bounded absolute-centered-weight denominations and intersected with the exact rounding pair partition. Of ${formatNumber(accumulatorReachability.interval_divergent_state_count_decimal)} interval-hull differences, ${formatNumber(accumulatorReachability.exact_reachable_divergent_state_count_decimal)} are constructively kernel-local reachable (${formatPercent(accumulatorReachability.exact_reachable_divergent_ratio)}), ${formatNumber(accumulatorReachability.provably_unreachable_divergent_state_count_decimal)} are excluded by lattice residue, and ${formatNumber(accumulatorReachability.unresolved_divergent_state_count_decimal)} remain congruence-compatible but unproven. ${formatNumber(accumulatorReachability.complete_integer_interval_channel_count)} channel(s) have complete integer coverage, ${formatNumber(accumulatorReachability.complete_modular_lattice_channel_count)} complete modular coverage, ${formatNumber(accumulatorReachability.partial_band_channel_count)} endpoint-band proofs, and ${formatNumber(accumulatorReachability.singleton_channel_count)} singleton accumulators. Ranked exact counterexample: #${padOp(reachabilityTop.op_index)} channel ${reachabilityChannel.channel_index}, accumulator ${reachabilityChannel.first_exact_reachable_divergent_accumulator_decimal}, default/single outputs ${reachabilityChannel.first_default_output_code}/${reachabilityChannel.first_single_output_code}; op reachability SHA-256 ${reachabilityTop.reachability_ledger_sha256}.`,
      interpretation: "The counterexample is exact for a full-valid kernel-local receptive field under independently selectable legal quantized input codes. It is stronger than interval-only exposure, but it is not proof that upstream model activations or a real model input realize the local assignment, that padding sites are free, that a declared output changes, or that the state occurs at runtime.",
      recommendation: "Export the aggregate coefficient witness and realize it as a concrete local input-code assignment, then replay the pinned default and TFLITE_SINGLE_ROUNDING kernels. Follow with paired full-model tests on captured representative and adversarial inputs before assigning model-output or task-level impact.",
      relevance: "constructive quantized numerical ABI counterexamples and cross-build reproducibility",
      methodVersion: accumulatorReachability.method_version,
    }));
  }
  const abiPropagation = analysis?.numerical_abi_propagation;
  const abiTop = (abiPropagation?.propagation_ranking_op_indices || [])
    .map((opIndex) => (abiPropagation?.sources || []).find((source) => source.op_index === opIndex))
    .find((source) => source?.local_reachability_status === "exact_local_counterexample" && source.reachable_model_output_tensor_count > 0);
  if (Number(abiPropagation?.exact_output_reachable_source_op_count || 0) > 0 && abiTop) {
    const firstOutput = abiTop.model_output_paths?.[0];
    findings.push(finding({
      id: "EA-QNT-0114",
      category: "quantization_design_review",
      title: "Exact kernel-local fixed-point counterexample has a structural path to declared model outputs",
      evidence: "DERIVED",
      priority: "Informational",
      op: `#${padOp(abiTop.op_index)} ${abiTop.op_name}`,
      tensor: firstOutput ? `T${firstOutput.output_tensor_index} ${firstOutput.output_tensor_name}` : "declared model output",
      observation: `${formatNumber(abiPropagation.exact_local_counterexample_source_op_count)} of ${formatNumber(abiPropagation.divergent_source_op_count)} divergent source op(s) contain at least one constructively reachable kernel-local counterexample and ${formatNumber(abiPropagation.exact_output_reachable_source_op_count)} of those have structural corridors to declared outputs. The divergent-state partition is ${formatNumber(abiPropagation.exact_local_divergent_state_count_decimal)} exact reachable, ${formatNumber(abiPropagation.residue_excluded_divergent_state_count_decimal)} residue-excluded, and ${formatNumber(abiPropagation.unresolved_divergent_state_count_decimal)} unresolved. Exact-qualified corridors span ${formatNumber(abiPropagation.exact_unique_reachable_op_count)} ops, ${formatNumber(abiPropagation.exact_unique_reachable_tensor_count)} tensors, and ${formatNumber(abiPropagation.exact_unique_predicted_boundary_edge_count)} predicted execution-domain edge(s) carrying ${abiPropagation.exact_unique_predicted_boundary_logical_payload_bytes == null ? "an incompletely assessed payload" : formatBytes(abiPropagation.exact_unique_predicted_boundary_logical_payload_bytes)}. Ranked source #${padOp(abiTop.op_index)} has ${formatNumber(abiTop.exact_reachable_divergent_state_count_decimal)} exact local difference(s) and reaches the output through ${formatNumber(abiTop.exact_model_output_graph_route_count_decimal)} exact graph route(s); reachability SHA-256 ${abiTop.source_reachability_ledger_sha256}; propagation SHA-256 ${abiTop.propagation_ledger_sha256}.`,
      interpretation: "The source counterexample is exact for one full-valid kernel-local receptive field under independent legal input codes. Its downstream corridor is structural only: it does not prove that a full-model input realizes the local assignment or that cancellation, requantization, pooling, clamp, runtime lowering, and task semantics preserve a declared-output difference.",
      recommendation: "Bind the deployed runtime build and TFLITE_SINGLE_ROUNDING state, then run paired full-model output comparisons across those builds. Prioritize exported source corridors with high route multiplicity, residual reconvergence, and predicted execution-domain crossings; treat observed target outputs as authoritative.",
      relevance: "numerical ABI compatibility, output reproducibility, and cross-build validation planning",
      methodVersion: abiPropagation.method_version,
    }));
  }
  const inputCounterexample = analysis?.input_counterexample;
  const inputWitness = inputCounterexample?.witnesses?.[0];
  const inputSource = inputWitness
    ? (inputCounterexample.sources || []).find((source) => source.op_index === inputWitness.source_op_index)
    : null;
  if (Number(inputCounterexample?.tensor_abi_constructive_source_op_count || 0) > 0 && inputWitness && inputSource) {
    findings.push(finding({
      id: "EA-QNT-0116",
      category: "quantization_design_review",
      title: "Pinned rounding ABI has a complete model-input counterexample",
      evidence: "DERIVED",
      priority: "Medium",
      op: `#${padOp(inputWitness.source_op_index)} ${inputWitness.source_op_name}`,
      tensor: `T${inputWitness.model_input_tensor_index} ${inputWitness.model_input_tensor_name} -> T${inputWitness.source_output_tensor_index} channel ${inputWitness.source_channel_index}`,
      observation: `${formatNumber(inputCounterexample.tensor_abi_constructive_source_op_count)} of ${formatNumber(inputCounterexample.exact_local_source_op_count)} exact-local divergent source op(s) is constructively realizable by a complete declared model-input tensor. The representative ${inputWitness.model_input_shape.join("x")} ${inputWitness.model_input_dtype} tensor contains ${formatNumber(inputWitness.model_input_element_count)} elements, fills all positions with zero-point code ${inputWitness.full_tensor_fill_code}, and applies ${formatNumber(inputWitness.sparse_override_count)} unique overrides in a full-valid ${inputWitness.effective_patch_shape.join("x")} patch. Its exact dot ${formatNumber(inputWitness.dot_product_decimal)} plus bias ${formatNumber(inputWitness.bias_decimal)} reaches accumulator ${formatNumber(inputWitness.post_bias_accumulator_decimal)}, producing default/single-rounding output codes ${inputWitness.default_output_code}/${inputWitness.single_rounding_output_code}. The source has ${formatNumber(inputSource.exact_reachable_divergent_channel_count)} constructive channel(s), ${formatNumber(inputSource.exact_reachable_divergent_state_count_decimal)} exact divergent state(s), and ${formatNumber(inputSource.exact_model_output_graph_route_count_decimal || 0)} structural declared-output route(s). Full tensor SHA-256 ${inputWitness.full_model_input_tensor_sha256}; witness ledger SHA-256 ${inputWitness.witness_ledger_sha256}.`,
      interpretation: "This is the action-bearing apex of the same rounding-build provenance chain documented by EA-QNT-0111, 0113, 0115, and 0114; those records remain Informational evidence stages rather than duplicate Medium actions. The certificate is exact at the quantized model tensor ABI, but it does not establish that an application's image or audio preprocessing can emit the tensor, that the deployed runtime selects either pinned rounding path, or that the one-code source difference survives to a declared model output or changes task behavior.",
      recommendation: "Bind one runtime-build action: replay the exported raw input tensor on clean, binary-hashed runtime builds with default and TFLITE_SINGLE_ROUNDING configurations and compare source-op and declared-output tensors bit-for-bit. Separately test whether the documented production preprocessing contract can realize or approximate the certified tensor; keep structural route multiplicity distinct from observed output impact.",
      relevance: "constructive model-input numerical ABI compatibility and cross-build release validation",
      methodVersion: inputCounterexample.method_version,
    }));
  }
  const preprocessing = analysis?.preprocessing_realizability;
  const exactPreprocessing = (preprocessing?.candidates || []).filter((candidate) => candidate.exact_tensor_realization);
  const nonExactPreprocessing = (preprocessing?.candidates || []).filter((candidate) => candidate.status === "assessed" && !candidate.exact_tensor_realization);
  if (exactPreprocessing.length && nonExactPreprocessing.length && inputWitness) {
    const bestNonExact = nonExactPreprocessing.find((candidate) => candidate.contract_id === preprocessing.best_non_exact_contract_id) || nonExactPreprocessing[0];
    const mostDivergent = [...nonExactPreprocessing].sort((left, right) => right.unrealizable_tensor_element_count - left.unrealizable_tensor_element_count)[0];
    const unitInterval = (preprocessing.candidates || []).find((candidate) => candidate.contract_id === "unit_interval_rgb");
    const unitIntervalCoverage = unitInterval
      ? ` The [0,1] RGB convention reaches ${unitInterval.channel_maps.map((row) => `${formatNumber(row.reachable_tensor_code_count)}/256`).join(", ")} tensor codes by channel and leaves ${unitInterval.channel_maps.map((row) => formatNumber(row.tensor_code_hole_count)).join(", ")} code holes.`
      : "";
    const centered128 = (preprocessing.candidates || []).find((candidate) => candidate.contract_id === "center_128_div_128_rgb");
    const textbookMinusOne = (preprocessing.candidates || []).find((candidate) => candidate.contract_id === "minus_one_to_one_rgb");
    const conventionReconciliation = centered128 && textbookMinusOne
      ? ` Artifact range matching and witness realizability answer different questions: the declared affine range matches the common [-1,1] endpoint convention within one artifact step, while exact code reproduction is achieved by \`${centered128.pixel_to_real_formula}\`; the textbook \`${textbookMinusOne.pixel_to_real_formula}\` formula leaves ${formatNumber(textbookMinusOne.unrealizable_tensor_element_count)} witness element(s) one or more tensor codes away.`
      : "";
    findings.push(finding({
      id: "EA-QNT-0117",
      category: "integration_verification",
      title: "The exact ABI witness depends materially on the source preprocessing contract",
      evidence: "DERIVED",
      priority: "Medium",
      op: `#${padOp(inputWitness.source_op_index)} ${inputWitness.source_op_name}`,
      tensor: `T${inputWitness.model_input_tensor_index} ${inputWitness.model_input_tensor_name}`,
      observation: `${formatNumber(preprocessing.candidate_evaluation_count)} explicit 8-bit image preprocessing counterfactual(s) were exhaustively evaluated over all 256 source codes per channel. ${formatNumber(preprocessing.exact_tensor_realization_candidate_count)} contract(s) reproduce all ${formatNumber(inputWitness.model_input_element_count)} witness elements exactly: ${preprocessing.exact_contract_ids.join(", ")}. ${formatNumber(preprocessing.non_exact_candidate_count)} do not. Best non-exact ${bestNonExact.contract_id} leaves ${formatNumber(bestNonExact.unrealizable_tensor_element_count)} element(s) unrealizable with minimum total absolute tensor-code error ${formatNumber(bestNonExact.minimum_total_absolute_tensor_code_error_decimal)}; ${mostDivergent.contract_id} leaves ${formatNumber(mostDivergent.unrealizable_tensor_element_count)} element(s) unrealizable.${unitIntervalCoverage}${conventionReconciliation} Exact fixture SHA-256 ${exactPreprocessing[0].nearest_rgb_fixture_sha256}; portfolio SHA-256 ${preprocessing.portfolio_ledger_sha256}.`,
      interpretation: "The finite-domain result is exact for each named formula, channel permutation, saturation rule, and half-away rounding equation. It does not reveal which preprocessing implementation the application deploys; the artifact does not bind decoder, resize, color order, normalization, or quantizer behavior. A candidate mismatch is not evidence that the production app is wrong unless its contract is independently matched to that row.",
      recommendation: "Bind the production preprocessing implementation and checksum to the release, match it to an explicit formula and channel order, then replay the corresponding exported PNG and raw tensor through the production decoder-to-tensor path. Compare the produced tensor SHA-256 before performing paired runtime-build output tests.",
      relevance: "source-data contract reproducibility, numerical ABI replay, and integration validation",
      methodVersion: preprocessing.method_version,
    }));
  }
  const zpRisk = (quantizationContracts.weight_zero_point?.details || []).filter((item) => item.status === "fail" || item.status === "review");
  if (zpRisk.length) {
    const top = zpRisk[0];
    const legacyCount = zpRisk.filter((item) => item.asymmetric_uint8_observed).length;
    const invalidCount = zpRisk.filter((item) => item.status === "fail").length;
    const int8WeightCount = zpRisk.filter((item) => item.dtype === "INT8").length;
    const invalid = top.status === "fail";
    const example = `${top.tensor_name || `T${top.tensor_index}`} ${top.dtype} zero-point(s) ${top.zero_points.join(", ") || "not emitted"}`;
    const tocoDescription = /toco/i.test(String(analysis?.metadata_presence?.description || ""));
    const conversionMetadataMissing = Number(analysis?.metadata_presence?.conversion_metadata_entry_count || 0) === 0;
    const perTensorDepthwiseCount = (artifactIrOperators(analysis) || []).filter((op) => (
      op.name === "DEPTHWISE_CONV_2D" && String(op.quant_scale_mode || "").toLowerCase().includes("per-tensor")
    )).length;
    const lineageSynthesis = tocoDescription
      ? ` Root-cause synthesis: artifact description is ${JSON.stringify(analysis.metadata_presence.description)}; CONVERSION_METADATA is ${conversionMetadataMissing ? "absent" : "present"}; ${formatNumber(legacyCount)} asymmetric UINT8 kernel tensor(s), ${formatNumber(perTensorDepthwiseCount)} per-tensor depthwise op(s), and ${formatNumber(analysis?.per_channel_tensors || 0)} per-channel tensor(s) are observed; runtime floor is ${analysis?.runtime_compat?.effective_min_runtime_version || analysis?.runtime_compat?.derived_min_runtime_version || "not derived"}. These facts are jointly consistent with a legacy TOCO UINT8 export, while the exact converter binary/version remains unbound.`
      : "";
    findings.push(finding({
      id: "EA-QNT-0104",
      category: invalid ? "quantization" : "quantization_design_review",
      title: invalid ? "Weight zero-point metadata is invalid" : "Asymmetric UINT8 weight quantization observed",
      evidence: invalid ? "OBSERVED" : "DERIVED",
      priority: invalid ? "High" : "Medium",
      op: "Conv/Depthwise/FC kernel",
      tensor: invalid ? (top.tensor_name || `T${top.tensor_index}`) : `${formatNumber(legacyCount)} UINT8 kernel tensor(s); representative ${top.tensor_name || `T${top.tensor_index}`}`,
      observation: invalid
        ? example
        : `Invalid zero-points: ${formatNumber(invalidCount)}. Symmetric INT8 violations: ${int8WeightCount ? formatNumber(zpRisk.filter((item) => item.symmetric_int8_violation).length) : "not applicable, no INT8 weight tensors"}. Asymmetric UINT8 weight tensors observed: ${formatNumber(legacyCount)}. Representative example: ${example}.${lineageSynthesis}`,
      interpretation: invalid
        ? (isOnnx
          ? "Weight zero-point metadata is outside the legal ONNX quantized dtype contract. ONNX QLinear signed weights are not assumed symmetric."
          : "Weight zero-point metadata is outside the legal dtype contract or violates the symmetric INT8 weight expectation.")
        : tocoDescription
          ? "The artifact directly joins a TOCO description with UINT8 asymmetric/per-tensor quantization evidence. This is a source-artifact lineage diagnosis, not proof of the exact TOCO release, command line, calibration data, or training provenance."
          : "The artifact directly shows UINT8 weight dtype, non-zero zero-point metadata, and per-tensor asymmetric quantization. This pattern is commonly associated with older UINT8 conversion paths, but exact converter lineage is not embedded in the artifact.",
      recommendation: invalid
        ? "Regenerate or repair the quantized artifact before release, then confirm outputs against reference data."
        : "Record converter lineage and compatibility reason; prefer modern per-channel symmetric INT8 exports when supported by the deployment runtime.",
      relevance: "quantization metadata integrity; converter lineage",
      findingClass: invalid ? "" : "conversion-lineage review",
    }));
  }
  const accumulatorRisk = quantizationContracts.accumulator_bound;
  if (accumulatorRisk?.bound_class === "exact_stored_weight_channel_integer_domain"
    && Number(accumulatorRisk.overflow_risk_channels || 0) > 0) {
    const failed = (accumulatorRisk.details || [])
      .filter((item) => item.status === "fail")
      .sort((left, right) => Number(right.maximum_int32_ratio || 0) - Number(left.maximum_int32_ratio || 0));
    const top = failed[0] || {};
    const overflowIndices = (top.overflow_channel_indices || []).slice(0, 16);
    findings.push(finding({
      id: "EA-QNT-0106",
      category: "integrity",
      title: "Exact stored-weight accumulator envelope exceeds INT32",
      evidence: "DERIVED",
      priority: "High",
      op: Number.isInteger(top.op_index) ? `#${padOp(top.op_index)} ${top.op_name || "quantized op"}` : "quantized Conv/Depthwise/FC",
      tensor: top.weight_tensor_name || (Number.isInteger(top.weight_tensor_index) ? `T${top.weight_tensor_index}` : "constant weight tensor"),
      observation: `${formatNumber(accumulatorRisk.overflow_risk_channels)} output channel(s) across ${formatNumber(accumulatorRisk.overflow_risk_ops)} op(s) exceed the signed INT32 interval under the exact stored-weight legal-code envelope. Maximum absolute envelope ${accumulatorRisk.maximum_absolute_accumulator_decimal || "not emitted"}; required width ${accumulatorRisk.maximum_required_signed_bits || "not emitted"} signed bits. Representative overflow channels: ${overflowIndices.length ? overflowIndices.join(", ") : "see structured evidence"}.`,
      interpretation: "For the pinned TFLite reference integer accumulation algebra, at least one legal full-field input-code assignment can exceed INT32 before or after stored bias. This is a deterministic arithmetic-domain defect signal, not an observed overflow frequency or a claim about a backend that may widen or transform accumulation.",
      recommendation: "Block release for an INT32 reference path. Re-export with safer input/weight scales or reduced fan-in, or bind observed runtime evidence proving a wider/saturating accumulation contract and verify representative plus adversarial output parity before accepting the artifact.",
      relevance: "quantized integer arithmetic integrity; runtime accumulator contract",
    }));
  }
  const requantizationRisk = quantizationContracts.requantization_fidelity;
  if (requantizationRisk?.bound_class === "pinned_q0_31_encoding_and_conservative_pre_clamp_rounding"
    && (Number(requantizationRisk.default_pre_shift_overflow_channels || 0) > 0
      || Number(requantizationRisk.single_rounding_encoding_divergence_channels || 0) > 0)) {
    const defaultOverflow = Number(requantizationRisk.default_pre_shift_overflow_channels || 0);
    const singleDivergence = Number(requantizationRisk.single_rounding_encoding_divergence_channels || 0);
    const top = (requantizationRisk.details || [])
      .filter((item) => Number(item.default_pre_shift_overflow_channels || 0) > 0
        || Number(item.single_rounding_encoding_divergence_channels || 0) > 0)
      .sort((left, right) => Number(right.default_pre_shift_overflow_channels || 0) - Number(left.default_pre_shift_overflow_channels || 0)
        || Number(right.single_rounding_encoding_divergence_channels || 0) - Number(left.single_rounding_encoding_divergence_channels || 0)
        || Number(left.op_index || 0) - Number(right.op_index || 0))[0] || {};
    findings.push(finding({
      id: "EA-QNT-0107",
      category: "quantization",
      title: defaultOverflow ? "Pinned default requantization pre-shift exceeds INT32" : "Requantization encoding depends on TFLITE_SINGLE_ROUNDING",
      evidence: "DERIVED",
      priority: defaultOverflow ? "High" : "Medium",
      op: Number.isInteger(top.op_index) ? `#${padOp(top.op_index)} ${top.op_name || "quantized op"}` : "quantized Conv/Depthwise/FC",
      tensor: Number.isInteger(top.output_tensor_index) ? `T${top.output_tensor_index}` : "quantized output tensor",
      observation: `${formatNumber(defaultOverflow)} channel(s) exceed the default double-rounding INT32 positive pre-shift contract; ${formatNumber(singleDivergence)} channel(s) receive a different saturated multiplier/shift under TFLITE_SINGLE_ROUNDING. Artifact shift range ${requantizationRisk.shift_range?.join(" .. ") || "not emitted"}; maximum encoding drift ${formatScientific(requantizationRisk.maximum_encoding_drift_bound_codes || 0)} output codes.`,
      interpretation: "The result is exact for the pinned source preparation rules, full intermediate accumulator envelope, and post-bias requantization domain, but the deployment artifact does not embed the TFLITE_SINGLE_ROUNDING compile flag or prove which backend executes the op.",
      recommendation: defaultOverflow
        ? "Do not accept the default reference integer path without binding the runtime build flags and an observed execution assignment. Re-export scales or use a verified path that avoids the unsafe positive pre-shift, then compare output parity on adversarial inputs."
        : "Bind the runtime build manifest and record TFLITE_SINGLE_ROUNDING explicitly. Preserve the matching channel ledger in release evidence and verify output parity for the affected operators.",
      relevance: "quantized requantization integrity; runtime build reproducibility",
    }));
  }
  const topQuant = maxBy(
    ops.filter((op) => op.quant_risk === "risk" || op.quant_risk === "warn"),
    (op) => (op.quant_scale_ratio_meaningful ? Math.log10(Math.max(1, op.quant_scale_ratio || 0)) : 0) + Number(op.quant_scale_cv || 0) * 1.5 + (op.quant_zero_point_status === "out-of-range" ? Math.abs(op.quant_zero_point_offset || 0) / 32 : 0),
  );
  if (topQuant) {
    const scaleRatioEvidence = topQuant.quant_scale_ratio_meaningful
      ? `meaningful scale ratio ${formatScientific(topQuant.quant_scale_ratio || 0)}`
      : "scale ratio not meaningful for this tensor contract";
    findings.push(finding({
      id: "EA-QNT-0001",
      category: "quantization",
      title: "Quantization numerical contract review trigger",
      evidence: "DERIVED",
      priority: topQuant.quant_risk === "risk" ? "High" : "Medium",
      op: `#${padOp(topQuant.index)} ${topQuant.name}`,
      tensor: "-",
      observation: `${scaleRatioEvidence}, CV ${Number(topQuant.quant_scale_cv || 0).toFixed(2)}, zero-point status ${topQuant.quant_zero_point_status || "none"}; emitted op predicate ${topQuant.quant_risk || "review"}.`,
      interpretation: "Artifact metadata indicates a quantization contract requiring review. CV is descriptive and is not, by itself, the risk predicate or evidence of accumulated rounding error.",
      recommendation: "Inspect exact scale channels, stored centered codes, bias-scale consistency, and the source checkpoint. Select QAT or calibration changes only after representative output comparison identifies the corresponding mechanism.",
      relevance: "quantization verification support; software technical characterization",
    }));
  }
  const fallback = (analysis?.fallback_traffic_by_op_family || [])[0];
  if (fallback && Number(fallback.byte_percent || 0) >= 0.01) {
    findings.push(finding({
      id: "EA-DEL-0001",
      category: "delegate",
      title: "Predicted fallback tensor traffic concentration",
      evidence: "PREDICTED",
      priority: Number(fallback.byte_percent || 0) >= 0.06 ? "High" : "Medium",
      observation: `${fallback.name}: ${formatBytes(fallback.estimated_bytes || 0)} (${formatPercent(fallback.byte_percent || 0)})`,
      interpretation: "A small number of op families may dominate predicted fallback/copy traffic.",
      recommendation: "Confirm with target runtime delegate logs and op-level profiling; consider graph rewrite if confirmed.",
      relevance: "runtime compatibility; performance verification input",
    }));
  }
  const { top: topBottleneck, totalUs: bottleneckTotalUs } = bottleneckDistributionData(analysis, analysis?.target_profile || {}, { limit: 3 });
  const dequantBottleneck = topBottleneck.find((item) => String(item.op?.name || "").toUpperCase() === "DEQUANTIZE");
  if (dequantBottleneck && bottleneckTotalUs > 0) {
    const share = dequantBottleneck.estimate.totalUs / bottleneckTotalUs;
    const rank = topBottleneck.indexOf(dequantBottleneck) + 1;
    findings.push(finding({
      id: "EA-QNT-0002",
      category: "quantization",
      title: "DEQUANTIZE op dominates static bottleneck estimate",
      evidence: "ESTIMATED",
      priority: rank === 1 ? "High" : "Medium",
      op: `#${padOp(dequantBottleneck.op.index)} DEQUANTIZE`,
      observation: `Static bottleneck rank #${rank}: ${formatUs(dequantBottleneck.estimate.totalUs)} / ${formatPercent(share)} of total static estimate`,
      interpretation: "An INT8 model with FLOAT32 output requires a full DEQUANTIZE pass over all output elements before the result is usable. For this target's memory bandwidth estimate, this copy dominates the graph's static bottleneck. This is not represented in the INT8 speedup estimate, which focuses on compute ops.",
      recommendation: "Set inference output type to INT8 in the TFLite converter to eliminate this pass, then dequantize externally if needed. Alternatively, keep FLOAT32 output and accept the overhead, but confirm with on-device profiling.",
      relevance: "quantization deployment architecture; runtime performance characterization",
    }));
  }
  const runtimeBuildRisk = [...(analysis?.delegation_repair?.runtime_build_risks || [])]
    .sort((left, right) => Number(right.affected_conditionally_delegatable_mac_ratio || 0) - Number(left.affected_conditionally_delegatable_mac_ratio || 0))[0];
  if (runtimeBuildRisk && Number(runtimeBuildRisk.affected_conditionally_delegatable_op_count || 0) > 0) {
    const completeCoverageCollapse = Number(runtimeBuildRisk.absent_condition_remaining_conditionally_delegatable_op_count || 0) === 0;
    findings.push(finding({
      id: "EA-DEL-0004",
      category: "delegate",
      title: completeCoverageCollapse
        ? "Required quantized XNNPACK build condition is not artifact-bound and conditional coverage collapses to zero"
        : "Required quantized XNNPACK build condition is not artifact-bound",
      evidence: "PREDICTED",
      priority: completeCoverageCollapse || Number(runtimeBuildRisk.affected_conditionally_delegatable_mac_ratio || 0) >= 0.5 ? "High" : "Medium",
      op: `${formatNumber(runtimeBuildRisk.affected_conditionally_delegatable_op_count)} conditionally delegatable op(s)`,
      observation: `\`${runtimeBuildRisk.required_build_configuration}\` is required by the pinned rulepack but is not embedded in the model or bound runtime manifest. If the build condition is absent, conditionally delegatable coverage changes from ${formatNumber(runtimeBuildRisk.baseline_conditionally_delegatable_op_count)} to ${formatNumber(runtimeBuildRisk.absent_condition_remaining_conditionally_delegatable_op_count)} op(s), affecting ${formatPercent(runtimeBuildRisk.affected_conditionally_delegatable_mac_ratio)} of modeled MACs and ${formatNumber(runtimeBuildRisk.affected_predicted_delegate_segment_count)} predicted delegate segment(s).`,
      interpretation: `${runtimeBuildRisk.interpretation_boundary} For legacy QU8, the pinned source describes support as experimental; static artifact eligibility must not be read as observed runtime assignment.`,
      recommendation: "Bind the deployed TFLite/XNNPACK commit, binary SHA-256, build configuration, and delegate assignment trace. Treat the no-flag scenario as the release default until the required flag is proven in the runtime manifest.",
      relevance: "runtime build reproducibility; conditional delegate coverage",
      findingClass: "runtime-build configuration risk",
    }));
  }
  const chainBreaks = Number(analysis?.xnnpack_effective_chain_breaks || 0);
  const boundaryInventory = predictedPartitionBoundaryInventory(analysis);
  const boundaryEdges = boundaryInventory?.edges || [];
  if (chainBreaks || boundaryEdges.length) {
    const breakOps = ops.filter((o) => o.xnnpack_chain_break);
    const shapeBreaks = breakOps.filter((op) => ["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"].includes(String(op.name || "").toUpperCase()));
    const exactPayload = Number.isFinite(boundaryInventory?.summed_edge_payload_bytes);
    const payloadBytes = exactPayload
      ? Number(boundaryInventory.summed_edge_payload_bytes)
      : Number(boundaryInventory?.assessed_edge_payload_bytes || 0);
    const payloadSummary = boundaryEdges.length
      ? ` Graph-derived ${exactPayload ? "summed logical edge payload" : "assessed partial edge payload"} ${formatBytes(payloadBytes)} across ${plural(boundaryEdges.length, "predicted internal partition edge")}: ${boundaryEdges.slice(0, 3).map((edge) => `T${edge.tensor_index} ${edge.tensor_name || "-"} ${edge.producer_domain}->${edge.consumer_domain} ${edge.payload_bytes == null ? "payload not assessed" : formatBytes(edge.payload_bytes)}`).join(" / ")}.`
      : "";
    const conditionalNote = breakOps.length > 0 && shapeBreaks.length === breakOps.length
      ? " All listed break ops are shape-only candidates; runtime may alias or elide copies."
      : "";
    const breakFamilies = [...new Set(breakOps.map((op) => String(op.name || "").toUpperCase()).filter(Boolean))];
    const exportIntervention = (analysis?.delegation_repair?.export_interventions || [])
      .find((item) => item.id === "se_global_pool_keepdims");
    const exportScenario = exportIntervention
      ? ` The exact SE motif portfolio matches ${formatNumber(exportIntervention.block_count)} block(s) and its fixed-graph assignment proxy changes delegate segments by ${formatNumber(exportIntervention.signed_delegate_segment_count)}, boundary edges by ${formatNumber(exportIntervention.signed_boundary_edge_count)}, and logical boundary payload by ${formatBytes(Math.abs(Number(exportIntervention.signed_boundary_payload_bytes || 0)))}; excluded rank4 MEAN ${(exportIntervention.unmatched_rank4_mean_op_indices || []).map((index) => `#${padOp(index)}`).join(", ") || "none"}.`
      : "";
    const recommendation = exportIntervention
      ? "Re-export the matched squeeze-excitation reductions with keepdims=True, remove the paired rank-restoring EXPAND_DIMS nodes, then re-audit the transformed artifact and confirm assignment in the bound runtime."
      : shapeBreaks.length === breakOps.length
      ? "Check delegate coverage logs on the target runtime before treating this as confirmed latency; profile before rewriting structural/view operators."
      : `Confirm the predicted ${breakFamilies.slice(0, 3).join(", ") || "non-delegated op"} fallback using target-runtime delegate logs and profiling before modifying the graph, delegate configuration, or affected operator implementation.`;
    findings.push(finding({
      id: "EA-DEL-0002",
      category: "delegate",
      title: "Predicted partition boundaries requiring runtime confirmation",
      evidence: "PREDICTED",
      priority: chainBreaks >= 4 || boundaryEdges.length >= 8 ? "High" : "Medium",
      op: breakOps.slice(0, 3).map((o) => `#${padOp(o.index)} ${o.name}`).join(", ") || "",
      observation: `${plural(chainBreaks, "non-structural predicted partition break")}; ${plural(shapeBreaks.length, "predicted non-delegated shape operation")}; ${plural(boundaryEdges.length, "graph-derived internal execution-domain edge")}.${payloadSummary}${conditionalNote}${exportScenario}`,
      interpretation: "Producer/consumer tensor edges and logical payloads are graph-derived from the static predicted assignment. The artifact alone cannot determine whether the runtime materializes copies, aliases buffers, converts layouts, or incurs measurable latency.",
      recommendation,
      relevance: "runtime compatibility; software architecture characterization",
    }));
  }
  const runtimeAssignment = runtimeEvidence?.runtimeAssignmentEvidence || runtimeEvidence?.runtime_assignment || null;
  const runtimeComparison = runtimeAssignment?.comparison || null;
  const placementMismatchCount = Number(runtimeComparison?.placement_assessment?.mismatch_count || 0);
  const boundaryMismatchCount = Number(runtimeComparison?.boundary_comparison?.mismatch_count || 0);
  if (placementMismatchCount || boundaryMismatchCount) {
    const mismatchMacRatio = runtimeComparison?.mac_comparison?.mismatch_mac_ratio;
    const topMismatches = (runtimeComparison.mismatches || []).slice(0, 5);
    findings.push(finding({
      id: "EA-DEL-0003",
      category: "delegate",
      title: "Observed runtime assignment differs from static delegation prediction",
      evidence: "DERIVED_FROM_OBSERVED_RUNTIME",
      priority: Number(mismatchMacRatio || 0) >= 0.1 || boundaryMismatchCount >= 4 ? "High" : "Medium",
      op: topMismatches.map((item) => `#${padOp(item.op_index)} ${item.op_name}`).join(", "),
      observation: `${formatNumber(placementMismatchCount)} classified op placement mismatch(es): ${formatNumber(runtimeComparison.placement_assessment?.overpredicted_delegation_count || 0)} overpredicted and ${formatNumber(runtimeComparison.placement_assessment?.underpredicted_delegation_count || 0)} underpredicted; ${formatNumber(boundaryMismatchCount)} assessed graph-edge boundary mismatch(es); MAC-weighted mismatch ${mismatchMacRatio == null ? "not assessed" : formatPercent(mismatchMacRatio)}.${topMismatches.length ? ` Top rows: ${topMismatches.map((item) => `#${padOp(item.op_index)} ${item.op_name} ${item.classification}`).join(" / ")}.` : ""}`,
      interpretation: "The pinned static rulepack does not reproduce the imported runtime/build assignment for this artifact and target-profile digest. Build flags, delegate version, operator lowering, dynamic preparation, or a stale rule may explain the difference; this is not by itself a model correctness defect.",
      recommendation: "Retain the runtime binary/build identity and verbose partition log, inspect each mismatch against the pinned delegate source, and update the target profile or rulepack only after the runtime-specific cause is established.",
      relevance: "runtime compatibility; delegation-rule validation; performance reproducibility",
    }));
  }
  const arenaReconciliation = runtimeAssignment?.arena_reconciliation || null;
  if (arenaReconciliation) {
    const peakDelta = arenaReconciliation.peak_delta_bytes;
    const hasAllocationDifference = Number(arenaReconciliation.runtime_only_allocation_count || 0) > 0
      || Number(arenaReconciliation.missing_observed_allocation_count || 0) > 0
      || Number(arenaReconciliation.size_mismatch_count || 0) > 0
      || Number(arenaReconciliation.alias_mismatch_count || 0) > 0;
    const hasPlacementDifference = Number(arenaReconciliation.offset_mismatch_count || 0) > 0;
    if (hasAllocationDifference || hasPlacementDifference || Number(peakDelta || 0) !== 0) {
      const signedPeakDelta = peakDelta == null
        ? "not assessed"
        : `${peakDelta > 0 ? "+" : peakDelta < 0 ? "-" : ""}${formatBytes(Math.abs(peakDelta))}`;
      findings.push(finding({
        id: "EA-MEM-0002",
        category: "memory_cache",
        title: "Observed TFLite arena allocation differs from the declared-shape projection",
        evidence: "DERIVED_FROM_OBSERVED_RUNTIME",
        priority: Number(peakDelta || 0) > 0 || hasAllocationDifference ? "Medium" : "Informational",
        op: "ArenaPlanner post-commit allocation",
        observation: `Pinned runtime peak ${formatBytes(arenaReconciliation.observed_peak_combined_arena_bytes)} versus declared-shape projection ${arenaReconciliation.projected_combined_arena_bytes == null ? "not assessed" : formatBytes(arenaReconciliation.projected_combined_arena_bytes)}; peak delta ${signedPeakDelta}. Root allocations projected/observed ${formatNumber(arenaReconciliation.projected_root_allocation_count)}/${formatNumber(arenaReconciliation.observed_root_allocation_count)}; runtime-only ${formatNumber(arenaReconciliation.runtime_only_allocation_count)}, missing observed ${formatNumber(arenaReconciliation.missing_observed_allocation_count)}, size differences ${formatNumber(arenaReconciliation.size_mismatch_count)}, offset differences ${formatNumber(arenaReconciliation.offset_mismatch_count)}, alias differences ${formatNumber(arenaReconciliation.alias_mismatch_count)}. Prepare-time runtime temporaries ${formatNumber(arenaReconciliation.runtime_temporary_allocation_count)} totaling ${formatBytes(arenaReconciliation.runtime_temporary_interval_bytes)} across allocation intervals.`,
        interpretation: "The difference is observed for this pinned runtime build and invocation. It is not automatically a defect: delegate replacement, Prepare-time tensor creation or resize, allocation-type changes, and execution-plan rewrites can legitimately change arena roots, sizes, aliases, and offsets.",
        recommendation: "Retain this allocation ledger with the runtime binary and invocation shapes; inspect runtime-only, missing, and size-different tensors against Prepare and delegation logs before changing model topology or capacity budgets. Account for delegate buffers, scratch, custom allocators, and process memory separately.",
        relevance: "runtime memory capacity; static-planner validation; deployment reproducibility",
      }));
    }
  }
  const l1Op = maxBy(ops, (op) => Number(op.row_working_set_ratio || 0));
  if (Number(l1Op?.row_working_set_ratio || 0) >= 0.9) {
    const rowWorkingSetBytes = Number(l1Op.row_working_set_bytes || 0);
    const l1Bytes = Number(analysis?.target_profile?.l1_data_bytes || 0);
    const l2Bytes = Number(analysis?.target_profile?.l2_bytes || 0);
    const l1Ratio = Number(l1Op.row_working_set_ratio || 0);
    const l2Ratio = l2Bytes > 0 ? rowWorkingSetBytes / l2Bytes : null;
    const exceeds = l1Ratio > 1;
    findings.push(finding({
      id: "EA-MEM-0001",
      category: "memory_cache",
      title: "L1 row working-set pressure",
      evidence: "ESTIMATED",
      priority: l1Ratio > 3 ? "High" : "Medium",
      op: `#${padOp(l1Op.index)} ${l1Op.name}`,
      observation: `${formatBytes(rowWorkingSetBytes)} row working set / ${formatBytes(l1Bytes)} L1D = ${l1Ratio.toFixed(2)}x; ${l2Ratio == null ? "L2 reference not emitted" : `L2: ${formatBytes(rowWorkingSetBytes)} / ${formatBytes(l2Bytes)} = ${l2Ratio.toFixed(2)}x`}.`,
      interpretation: `Static row working-set estimate ${exceeds ? "exceeds" : "is within 10% of"} the selected target L1D reference. Code, other live data, associativity, and runtime scratch leave less usable capacity than the nominal cache size.`,
      recommendation: "Validate executed tiling and L1D refill/miss counters on the target; consider smaller spatial tiles, channel alignment, or graph changes only if this op is latency-dominant.",
      relevance: "computational resource characterization",
    }));
  }
  const packingOp = maxBy(
    ops.filter((op) => op.weight_packing_risk === "warn"),
    (op) => Number(op.weight_packing_overhead_us || 0),
  );
  if (packingOp) {
    const warnCount = ops.filter((op) => op.weight_packing_risk === "warn").length;
    const packingUs = Number(packingOp.weight_packing_overhead_us || 0);
    findings.push(finding({
      id: "EA-PKG-0001",
      category: "packing",
      title: "Weight packing warmup cost watchlist",
      evidence: "ESTIMATED",
      priority: "Informational",
      op: `#${padOp(packingOp.index)} ${packingOp.name}`,
      observation: `${plural(warnCount, "op")} exceeded the static packing-watch threshold; largest estimate ${formatUs(packingUs)}. ${packingOp.weight_packing_detail || "No packing detail emitted."}`,
      interpretation: "Static weight-byte, setup, and target-bandwidth estimates indicate potential one-time interpreter preparation or first-invocation packing cost; this is not steady-state Invoke latency. Escalate only if the configured cold-start budget is tight.",
      recommendation: "Warm up before measuring steady-state latency, and report preparation/first-invocation packing separately from p50/p90 Invoke results.",
      relevance: "runtime performance characterization; benchmark protocol",
    }));
  }
  const alignOp = maxBy(ops.filter((op) => op.channel_alignment_status === "misaligned"), (op) => Number(op.channel_tail_overhead_percent || 0));
  if (alignOp) {
    const tailMin = Number(alignOp.channel_tail_overhead_percent_min ?? alignOp.channel_tail_overhead_percent ?? 0);
    const tailMax = Number(alignOp.channel_tail_overhead_percent_max ?? alignOp.channel_tail_overhead_percent ?? 0);
    findings.push(finding({
      id: "EA-CHN-0001",
      category: "channel_alignment",
      title: "SIMD/micro-kernel channel tail overhead",
      evidence: "ESTIMATED",
      // Static lane-packing signal never rises above Medium without profiler
      // evidence; Medium only when the op itself carries meaningful MAC share.
      priority: (Number(alignOp.macs || 0) / Math.max(1, Number(analysis?.total_macs || 0)) >= 0.05
        && Number(alignOp.channel_tail_overhead_percent || 0) >= 0.25) ? "Medium" : "Low",
      op: `#${padOp(alignOp.index)} ${alignOp.name}`,
      observation: `${ensureSentence(alignOp.channel_alignment_detail || alignmentLabel(alignOp))} Modeled candidate-set lane utilization: ${formatPercentRange(1 - tailMax, 1 - tailMin)}; op MAC share ${((Number(alignOp.macs || 0) / Math.max(1, Number(analysis?.total_macs || 0))) * 100).toFixed(1)}%; model-level impact not estimated.`,
      interpretation: "Actual overhead depends on runtime kernel selection, packing, and tail handling, and is not confirmed from the artifact alone.",
      recommendation: "Verify with a profiler or an aligned-head variant before acting; treat as an optimization opportunity, not a deployment defect.",
      relevance: "runtime performance verification input",
    }));
  }
  const runtimeWarn = runtimeBasinResult?.backend_interpretations?.find((item) => item.severity === "warn");
  if (runtimeWarn) {
    findings.push(finding({
      id: "EA-RUN-0002",
      category: "runtime",
      title: "Browser backend path mismatch",
      evidence: "MEASURED_SYNTHETIC",
      priority: "Medium",
      observation: `${runtimeWarn.backend}: ${runtimeWarn.note}`,
      interpretation: "Browser accelerator availability does not imply native delegate speed or numeric equivalence.",
      recommendation: "Use target runtime profiling for deployment claims and keep browser result as compatibility evidence.",
      relevance: "execution integrity; runtime compatibility",
    }));
  }
  const consistencyWarning = deployCurvatureResult?.drift_consistency?.warning;
  if (consistencyWarning) {
    findings.push(finding({
      id: "EA-SYN-0001",
      category: "synthetic_sensitivity",
      title: "Synthetic perturbation plateau or saturation",
      evidence: "MEASURED_SYNTHETIC",
      priority: "Medium",
      observation: consistencyWarning,
      interpretation: "Local synthetic perturbation does not show a monotonic amplitude response; this can happen with quantized plateaus.",
      recommendation: "Repeat with prepared/calibration inputs and compare output-specific drift before drawing robustness conclusions.",
      relevance: "numerical consistency; verification planning",
    }));
  }
  findings.push(finding({
    id: "EA-LIM-0001",
    category: "limitation",
    title: "Representative-data validation not assessable",
    evidence: "NOT_ASSESSABLE",
    priority: "Low",
    observation: "No training, validation, clinical, or representative calibration dataset is included in the deployment artifact.",
    interpretation: "Accuracy, clinical benefit, bias, subgroup performance, and generalizability cannot be concluded from this artifact alone.",
    recommendation: "Link this artifact report to external validation, calibration, and risk-management records.",
    relevance: "technical documentation boundary; regulatory limitation statement",
  }));
  return findings;
}

function tensorByIndex(analysis, index) {
  return Number.isInteger(index) && index >= 0 ? (artifactIrValues(analysis) || [])[index] : null;
}

function isQuantizedTensor(tensor) {
  return ["INT8", "UINT8"].includes(String(tensor?.dtype || "").toUpperCase()) || Number(tensor?.quant_scales || 0) > 0;
}

function depthwisePerTensorQuantOps(analysis) {
  if ((analysis?.format || "tflite") !== "tflite") return [];
  return (artifactIrOperators(analysis) || [])
    .filter((op) => String(op.name || "").toUpperCase() === "DEPTHWISE_CONV_2D")
    .map((op) => ({ op, weight: tensorByIndex(analysis, Number(op.inputs?.[1])) }))
    .filter((item) => isQuantizedTensor(item.weight) && Number(item.weight?.quant_scales || 0) <= 1);
}

function firstScale(tensor) {
  const scales = tensor?.scale_sample || [];
  return Number(scales[0] || 0);
}

function shapeText(shape = []) {
  return Array.isArray(shape) && shape.length ? shape.join("x") : "unknown-shape";
}

function plural(count, singular, pluralForm = `${singular}s`) {
  const n = Number(count || 0);
  return `${formatNumber(n)} ${n === 1 ? singular : pluralForm}`;
}

function ensureSentence(text) {
  const value = String(text || "").trim();
  if (!value) return "";
  return /[.!?]$/.test(value) ? value : `${value}.`;
}

function bundledRuntimeVersionText(runtimeEvidence = {}) {
  return runtimeEvidence?.bundledRuntimeVersion
    || runtimeEvidence?.litertRuntimeVersion
    || runtimeEvidence?.tfliteRuntimeVersion
    || runtimeEvidence?.runtimeVersion
    || "";
}

function outputSemanticObservation(analysis, output) {
  const dtype = output?.dtype || "unknown";
  const shape = shapeText(output?.shape);
  const scale = firstScale(output);
  const zp = (output?.zero_point_sample || [])[0];
  const dequant = ["INT8", "UINT8"].includes(String(dtype).toUpperCase()) && scale > 0
    ? ` Output dequantization contract: real_value = scale ${scale} * (q - zero_point ${zp ?? 0}); implied representable range ${quantizedRealRangeText(dtype, scale, zp)}; include this in the application-side interpretation spec.`
    : "";
  const outputIndex = Number.isInteger(output?.index) ? output.index : Number(output?.tensor_index);
  const producer = Number.isFinite(outputIndex)
    ? (artifactIrOperators(analysis) || []).find((op) => (op.outputs || []).includes(outputIndex))
    : null;
  if (producer) {
    const name = String(producer.name || "producer").toUpperCase();
    if (name === "SOFTMAX") {
      return `Output transform observed as SOFTMAX #${padOp(producer.index)} producing ${dtype} ${shape} output. Semantic class mapping, label order, and application-side output contract are not embedded.${dequant}`;
    }
    return `The graph output is produced by ${name} #${padOp(producer.index)} as ${dtype} ${shape}; no terminal SOFTMAX is observed, so probability semantics are NOT_ASSESSABLE from the artifact alone. No semantic label map, class ordering, or application-side output contract is embedded.${dequant}`;
  }
  return `The artifact exposes ${dtype} ${shape} output, but no semantic label map, class ordering, or application-side output contract is embedded.${dequant}`;
}

function quantizedRealRangeText(dtype, scale, zeroPoint = 0) {
  const upper = String(dtype || "").toUpperCase() === "INT8" ? 127 : 255;
  const lower = String(dtype || "").toUpperCase() === "INT8" ? -128 : 0;
  const zp = Number(zeroPoint || 0);
  const s = Number(scale || 0);
  if (!(s > 0)) return "not computable";
  return `[${(s * (lower - zp)).toExponential(4)}, ${(s * (upper - zp)).toExponential(4)}]`;
}
