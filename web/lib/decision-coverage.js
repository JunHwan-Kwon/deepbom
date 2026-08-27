import { markdownTable } from "./report-utils.js";

export const DECISION_COVERAGE_SCHEMA = "deepbom.decision_coverage.v1.10";

const METRIC_STATUSES = Object.freeze(["assessed", "partial", "not_assessed", "not_applicable", "suppressed"]);

const DECISION_DOMAINS = Object.freeze([
  Object.freeze({
    domain_id: "artifact_contract",
    label: "Artifact, graph, and interface contract",
    metric_ids: Object.freeze([
      "artifact.identity", "artifact.metadata", "artifact.size", "artifact.byte_integrity", "graph.inventory", "architecture.blocks", "contract.io", "tflite.sparse_storage", "tflite.subgraph_inventory", "tflite.subgraph_deep_analysis",
      "serialized.container_contract", "coreml.serialized_contract", "executorch.serialized_contract", "llm.on_device_contract",
      "runtime.artifact_requirements", "weights.integrity", "onnx.domains", "onnx.external_data",
      "onnx.shape_inference", "onnx.tensor_data_types", "onnx.type_proto_contract", "onnx.sparse_tensor_contract",
    ]),
    interpretation_boundary: "Closes artifact-visible identity, structure, ABI, payload, and bounded source-pinned shape/type contracts. Missing sidecars, dynamic values, unsupported schemas, and release lineage remain explicit residuals.",
  }),
  Object.freeze({
    domain_id: "numerical_contract",
    label: "Numerical and quantization contract",
    metric_prefixes: Object.freeze(["tflite.accumulator", "tflite.requantization", "tflite.kernel_witness", "tflite.channel_vitality", "tflite.rounding_equivalence", "tflite.accumulator_reachability", "tflite.numerical_abi", "tflite.input_witness", "tflite.preprocessing", "tflite.quantization_lattice", "tflite.contract_migration", "tflite.residual_step", "tflite.residual_distortion"]),
    metric_ids: Object.freeze(["compute.macs", "quantization.contracts", "weights.serialized_payload_integrity", "weights.serialized_storage_summary", "weights.safetensors_packed_quantization", "tflite.quant_research_coverage", "onnx.quantization_binding"]),
    interpretation_boundary: "Exact or bounded calculations apply only to eligible artifact contracts and declared legal-code domains. They do not establish runtime activation frequency, full-input reachability, model accuracy, or product acceptability.",
  }),
  Object.freeze({
    domain_id: "resource_performance_model",
    label: "Memory and static performance model",
    metric_ids: Object.freeze(["performance.static_posture", "memory.liveness", "memory.cache_payload", "cost.dynamic_shape", "tflite.arena", "tflite.movement_packing", "tflite.core_isolation_roofline"]),
    interpretation_boundary: "Logical payload, declared-shape arena, and disclosed planning estimates are static evidence. Actual allocator bytes, cache behavior, materialized copies, thermal state, scheduling, and latency require runtime measurement.",
  }),
  Object.freeze({
    domain_id: "deployment_compatibility",
    label: "Delegation and provider compatibility",
    metric_ids: Object.freeze([
      "tflite.delegation", "tflite.xnnpack_candidates", "tflite.alternate_delegate_candidates", "tflite.deployment_frontier", "tflite.deployment_delta",
      "tflite.delegation_repair", "onnx.ort_source_compatibility", "onnx.ep_portability", "onnx.tensorrt_static_preflight",
      "coreml.deployment_floor", "runtime.coreml_compute_plan", "runtime.environment", "runtime.backend_evidence_layers",
    ]),
    interpretation_boundary: "Source-backed candidates, necessary OS floors, build-environment bindings, compute-plan estimates, definite exclusions, and static partitions are not executed placement. CPU features, build inclusion, lowering, GetCapability, graph rewrites, selected microkernel, and partition identity require bound runtime evidence.",
  }),
  Object.freeze({
    domain_id: "runtime_observation",
    label: "Runtime execution observation",
    metric_ids: Object.freeze(["runtime.assignment", "runtime.onnx_internal_shape_binding", "runtime.data_movement", "runtime.arena_memory", "runtime.browser_execution", "validation.representative_dataset_capture"]),
    interpretation_boundary: "Assessed only when capture-bound assignment, allocator evidence, browser/native execution evidence, or a hash-bound representative-dataset capture is imported. Runtime memory snapshots remain scoped to the reported allocator and may exclude provider/delegate buffers, scratch, custom allocators, and process RSS. Synthetic timing and externally declared dataset representativeness do not establish production workload performance, task accuracy, or device-wide determinism.",
  }),
  Object.freeze({
    domain_id: "product_validation",
    label: "Task quality and release validation",
    metric_ids: Object.freeze(["validation.task_quality"]),
    interpretation_boundary: "Requires representative reference data, production preprocessing/postprocessing, task metrics, acceptance thresholds, and executed outputs. Static artifact evidence cannot certify accuracy, safety, clinical validity, or release readiness.",
  }),
  Object.freeze({
    domain_id: "evidence_delivery",
    label: "Findings, exports, and format isolation",
    metric_ids: Object.freeze(["findings.action_register", "exports.deterministic_derivatives", "performance.audit_timing", "tflite.ort_non_applicability", "onnx.tflite_non_applicability"]),
    interpretation_boundary: "Proves that calculated evidence is routed, scoped, and not cross-applied between formats, and preserves measured analyzer-workflow timing separately from inference timing. It is a delivery/conservation control, not independent proof that an upstream calculation is complete.",
  }),
]);

function matchingDecisionDomains(metricId) {
  return DECISION_DOMAINS.filter((domain) => (domain.metric_ids || []).includes(metricId)
    || (domain.metric_prefixes || []).some((prefix) => metricId.startsWith(prefix)));
}

function aggregateDecisionStatus(counts, metricCount) {
  const assessableCount = metricCount - counts.not_applicable - counts.suppressed;
  if (assessableCount <= 0) return "not_applicable";
  if (counts.not_assessed === assessableCount) return "not_assessed";
  if (counts.partial || counts.not_assessed) return "partial";
  return counts.assessed === assessableCount ? "assessed" : "not_assessed";
}

export function buildDecisionCoverageLedger(entries = [], format = "tflite") {
  const applicable = entries.filter((entry) => entry.applicability === format);
  const assignments = applicable.map((entry) => ({ entry, domains: matchingDecisionDomains(entry.metric_id) }));
  const unassignedMetricIds = assignments.filter((item) => item.domains.length === 0).map((item) => item.entry.metric_id);
  const duplicateMetricIds = assignments.filter((item) => item.domains.length > 1).map((item) => item.entry.metric_id);
  const rows = DECISION_DOMAINS.map((domain) => {
    const metrics = assignments.filter((item) => item.domains.some((candidate) => candidate.domain_id === domain.domain_id)).map((item) => item.entry);
    const statusCounts = Object.fromEntries(METRIC_STATUSES.map((status) => [status, metrics.filter((entry) => entry.status === status).length]));
    const residuals = metrics.filter((entry) => ["partial", "not_assessed"].includes(entry.status));
    return {
      domain_id: domain.domain_id,
      label: domain.label,
      status: aggregateDecisionStatus(statusCounts, metrics.length),
      metric_count: metrics.length,
      status_counts: statusCounts,
      evidence_classes: [...new Set(metrics.filter((entry) => ["assessed", "partial"].includes(entry.status)).map((entry) => entry.evidence_class))].sort(),
      metric_ids: metrics.map((entry) => entry.metric_id),
      residual_metric_ids: residuals.map((entry) => entry.metric_id),
      residual_labels: residuals.map((entry) => entry.label),
      interpretation_boundary: domain.interpretation_boundary,
    };
  });
  const domainStatusCounts = Object.fromEntries(["assessed", "partial", "not_assessed", "not_applicable"].map((status) => [status, rows.filter((row) => row.status === status).length]));
  return {
    schema: DECISION_COVERAGE_SCHEMA,
    format,
    status: unassignedMetricIds.length || duplicateMetricIds.length ? "fail"
      : domainStatusCounts.partial || domainStatusCounts.not_assessed ? "partial" : "complete",
    applicable_metric_count: applicable.length,
    assigned_metric_count: rows.reduce((sum, row) => sum + row.metric_count, 0),
    domain_count: rows.length,
    domain_status_counts: domainStatusCounts,
    unassigned_metric_ids: unassignedMetricIds,
    multiply_assigned_metric_ids: duplicateMetricIds,
    rows,
    interpretation_boundary: "Status is aggregated from the metric-family ledger without promoting partial or missing evidence. Assessed means the declared static or observed method completed for that domain's metrics; it never widens the claim beyond each row's interpretation boundary.",
  };
}

export function decisionCoverageMarkdown(manifest) {
  const ledger = manifest?.decision_coverage || {};
  const rows = (ledger.rows || []).map((row) => {
    const residualLabels = row.residual_labels || [];
    const residual = residualLabels.length
      ? `${residualLabels.slice(0, 4).join(" / ")}${residualLabels.length > 4 ? ` / +${residualLabels.length - 4} more in the metric ledger` : ""}`
      : "none within declared scope";
    return [
      row.label,
      row.status,
      `${row.status_counts?.assessed || 0} assessed / ${row.status_counts?.partial || 0} partial / ${row.status_counts?.not_assessed || 0} not assessed / ${row.status_counts?.not_applicable || 0} not applicable / ${row.status_counts?.suppressed || 0} suppressed`,
      row.evidence_classes?.join(" / ") || "NOT_ASSESSED",
      residual,
      row.interpretation_boundary,
    ];
  });
  const counts = ledger.domain_status_counts || {};
  return [
    "## Decision Coverage At A Glance (DERIVED)",
    `Schema \`${ledger.schema || DECISION_COVERAGE_SCHEMA}\`; status **${ledger.status || "not emitted"}**. ${ledger.assigned_metric_count || 0}/${ledger.applicable_metric_count || 0} applicable metric families are assigned exactly once across ${ledger.domain_count || 0} decision domains: ${counts.assessed || 0} assessed, ${counts.partial || 0} partial, ${counts.not_assessed || 0} not assessed, ${counts.not_applicable || 0} not applicable.`,
    markdownTable(["Decision domain", "Status", "Metric-family conservation", "Evidence", "Residual evidence", "Claim boundary"], rows),
    `> ${ledger.interpretation_boundary || "Decision coverage was not computed."}`,
  ].join("\n\n");
}
