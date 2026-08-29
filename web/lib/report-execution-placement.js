import { formatBytes, formatNumber } from "./format.js";
import { code, markdownTable } from "./report-utils.js";
import { unwrapRuntimeEvidence } from "./execution-placement-evidence.js";
import { buildRuntimeBackendEvidenceLedger } from "./runtime-backend-evidence-ledger.js";
import { buildRuntimeDataMovementEvidence } from "./runtime-data-movement-evidence.js";

export function executionPlacementRuntimeMetricResults(runtimeEvidence, { browserExecuted = false } = {}) {
  const runtime = unwrapRuntimeEvidence(runtimeEvidence);
  const runtimeBackendLedger = runtimeEvidence?.runtime_backend_evidence_ledger
    || runtimeEvidence?.runtimeBackendLedger
    || buildRuntimeBackendEvidenceLedger(runtime);
  return {
    runtime_assignment: Array.isArray(runtime?.assignments) ? runtime : null,
    runtime_environment: runtime?.schema === "deepbom.gguf_runtime_environment.v2" ? runtime : null,
    coreml_compute_plan: runtime?.schema === "deepbom.coreml_compute_plan.v1" ? runtime : null,
    runtime_backend_evidence_ledger: runtimeBackendLedger,
    runtime_data_movement_evidence: buildRuntimeDataMovementEvidence(runtime),
    representative_dataset_validation: runtimeEvidence?.calibrationValidationResult
      || runtimeEvidence?.representative_dataset_validation
      || null,
    browser_execution_status: browserExecuted ? "executed" : "not_run",
    runtime_execution_status: browserExecuted ? "executed" : "not_run",
  };
}

export function executionPlacementCoverageRows(evidence) {
  const observed = evidence.runtime_observation.status === "observed";
  return [
    ["Imported execution placement", observed
      ? `${evidence.runtime_observation.covered_item_count}/${evidence.runtime_observation.total_item_count} item(s) observed; ${evidence.flow.evidence_basis}`
      : "not observed"],
    ["Representative device performance validation", "not provided"],
  ];
}

export function executionPlacementMarkdown(evidence) {
  const flowRows = evidence.flow.segments.map((segment) => [
    segment.label,
    `${formatNumber(segment.item_count)} item(s)`,
    `${segment.start_item_id} to ${segment.end_item_id}`,
    segment.tone,
  ]);
  const portfolioRows = evidence.portfolios.map((profile) => [
    profile.label,
    `${formatNumber(profile.candidate_count)} / ${formatNumber(profile.total_count)}`,
    profile.evidence_class,
    profile.detail,
  ]);
  const staticProfileSections = (evidence.static_profiles || []).map((profile) => {
    const counts = profile.state_counts;
    const payload = profile.boundary_payload;
    const segmentRows = profile.segments.map((segment) => [
      formatNumber(segment.segment_index),
      segment.state,
      `${formatNumber(segment.op_count)} op(s)`,
      `#${segment.start_op_index}${segment.start_op_index === segment.end_op_index ? "" : ` to #${segment.end_op_index}`}`,
    ]);
    const boundaryRows = profile.boundary_edges.map((edge) => [
      `T${edge.tensor_index}${edge.tensor_name ? ` ${edge.tensor_name}` : ""}`,
      `#${edge.producer_op_index}`,
      `#${edge.consumer_op_index}`,
      `${edge.producer_state} -> ${edge.consumer_state}`,
      edge.logical_payload_bytes == null ? `NOT ASSESSED: ${edge.logical_payload_reason}` : formatBytes(edge.logical_payload_bytes),
    ]);
    const workloadRows = ["CONDITIONALLY_ELIGIBLE", "DEFINITE_EXCLUSION", "UNRESOLVED"].map((state) => {
      const row = profile.workload_envelope.by_state[state];
      return [
        state,
        formatNumber(row.op_count),
        row.complete_macs == null ? `${formatNumber(row.assessed_macs)} assessed (${row.assessed_mac_op_count}/${row.op_count} ops)` : formatNumber(row.complete_macs),
        row.complete_logical_bytes == null ? `${formatBytes(row.assessed_logical_bytes)} assessed (${row.assessed_logical_byte_op_count}/${row.op_count} ops)` : formatBytes(row.complete_logical_bytes),
        row.mac_equivalent_ops_per_logical_byte_decimal ?? `NOT ASSESSED: ${row.intensity_status}`,
        Object.entries(row.output_dtype_reference_counts).map(([dtype, count]) => `${dtype} ${count}`).join(" / ") || "none",
      ];
    });
    return [
      `### ${profile.label} Independent Static Projection`,
      `Conditionally eligible ${formatNumber(counts.CONDITIONALLY_ELIGIBLE)}; definite exclusions ${formatNumber(counts.DEFINITE_EXCLUSION)}; unresolved ${formatNumber(counts.UNRESOLVED)}; ${formatNumber(profile.boundary_edge_count)} cross-state tensor edge(s).`,
      `Logical edge payload: ${payload.summed_edge_payload_bytes == null ? `${formatBytes(payload.assessed_edge_payload_bytes)} assessed across ${formatNumber(payload.assessed_edge_count)}/${formatNumber(payload.edge_count)} edge(s)` : formatBytes(payload.summed_edge_payload_bytes)}. This is logical exposure, not an observed copy or transfer.`,
      segmentRows.length ? markdownTable(["Segment", "State", "Cardinality", "Op range"], segmentRows) : "No serialized operators are present.",
      `#### Artifact Workload Envelope\n\n${markdownTable(["Eligibility state", "Ops", "MACs", "Logical op bytes", "MAC-equivalent ops/B", "Output dtypes"], workloadRows)}`,
      `> ${profile.workload_envelope.interpretation_boundary} Backend cost model remains NOT ASSESSED: ${profile.workload_envelope.backend_cost_model.reason}`,
      boundaryRows.length ? `#### Cross-State Tensor Edges\n\n${markdownTable(["Tensor", "Producer", "Consumer", "Transition", "Logical payload"], boundaryRows)}` : "No cross-state tensor edge is present.",
      `> ${profile.interpretation_boundary}`,
    ].join("\n\n");
  });
  const configurationPreflights = (evidence.configuration_preflights || []).map((preflight) => [
    `### ${preflight.label}`,
    `Status ${code(preflight.status)}; path ${code(preflight.execution_path || "unbound")}; build profile SHA-256 ${code(preflight.build_profile_sha256 || "unbound")}; ${formatNumber(preflight.blocking_issue_count)} blocking and ${formatNumber(preflight.unresolved_issue_count)} unresolved issue(s).`,
    preflight.issues.length ? markdownTable(["Severity", "Issue", "Observation", "Required action"], preflight.issues.map((issue) => [
      issue.severity, issue.id, issue.observation, issue.action,
    ])) : "No static configuration issue is present.",
    preflight.optimization_profile_cost?.scenarios?.length ? `#### Conditional Optimization-Profile Costs\n\n${markdownTable(["Profile point", "Status", "MACs", "Peak live payload", "Unbound symbols"], preflight.optimization_profile_cost.scenarios.map((scenario) => [
      `${scenario.profile_id} / ${scenario.profile_point}`,
      scenario.status,
      scenario.total_macs_decimal ?? "not assessed",
      scenario.peak_live_payload_bytes == null ? scenario.peak_live_payload_bytes_decimal ?? "not assessed" : formatBytes(scenario.peak_live_payload_bytes),
      scenario.residual_symbol_ids.join(", ") || "none",
    ]))}\n\n> ${preflight.optimization_profile_cost.interpretation_boundary}` : "No optimization-profile cost scenario is emitted.",
    preflight.engine_inspector ? `#### Optimized Engine Inspector Evidence\n\n${markdownTable(["Field", "Value"], [
      ["Status / class", `${preflight.engine_inspector.status} / ${preflight.engine_inspector.evidence_class}`],
      ["Engine identity", `${preflight.engine_inspector.engine_sha256}; ${formatBytes(preflight.engine_inspector.engine_byte_length)}`],
      ["Inspector contract", `${preflight.engine_inspector.schema_generation}; ${preflight.engine_inspector.profiling_verbosity}; execution context ${preflight.engine_inspector.execution_context_bound ? "bound" : "not bound"}`],
      ["Optimized layers / I/O tensors", `${formatNumber(preflight.engine_inspector.engine_layer_count)} / ${formatNumber(preflight.engine_inspector.io_tensor_count)}`],
      ["Tactic-annotated / multi-source metadata layers", `${formatNumber(preflight.engine_inspector.tactic_annotated_layer_count)} / ${formatNumber(preflight.engine_inspector.multi_source_metadata_layer_count)}`],
      ["Dynamic-dimension tensor rows", formatNumber(preflight.engine_inspector.dynamic_dimension_tensor_count)],
      ["Layer types", preflight.engine_inspector.layer_type_inventory.map((row) => `${row.name}:${row.count}`).join(" / ") || "not exposed"],
      ["Separate data types", preflight.engine_inspector.data_type_inventory.map((row) => `${row.name}:${row.count}`).join(" / ") || "not exposed"],
      ["Original-op mapping", preflight.engine_inspector.source_mapping_status],
      ["Artifact-engine relation", preflight.engine_inspector.artifact_engine_relation],
    ])}\n\n> ${preflight.engine_inspector.interpretation_boundary}` : "No optimized-engine inspector evidence is imported.",
    `> Trust boundary: browser engine deserialization ${code(preflight.trust_boundary.browser_engine_deserialization)}; browser plan execution ${code(preflight.trust_boundary.browser_plan_execution)}. ${preflight.interpretation_boundary}`,
  ].join("\n\n"));
  return [
    "## Execution Placement Evidence",
    markdownTable(["Evidence level", "State", "Class", "Detail"], evidence.levels.map((level) => [
      level.label, level.state, level.evidence_class, level.detail,
    ])),
    evidence.banner ? `> ${evidence.banner}` : "",
    `### ${evidence.flow.label}`,
    flowRows.length
      ? markdownTable(["Run", "Cardinality", "Item range", "State"], flowRows)
      : "No joint placement flow is asserted for this evidence state.",
    portfolioRows.length ? `### Independent Source Portfolios\n\n${markdownTable(["Profile", "Candidates", "Evidence", "Meaning"], portfolioRows)}` : "",
    ...staticProfileSections,
    ...configurationPreflights,
    `> ${evidence.interpretation_boundary}`,
    `Structured evidence: ${code(evidence.schema)}; flow basis ${code(evidence.flow.evidence_basis)}; rendered/covered/scope ${formatNumber(evidence.flow.rendered_item_count)}/${formatNumber(evidence.flow.covered_item_count)}/${formatNumber(evidence.flow.scope_item_count)}.`,
  ].filter(Boolean).join("\n\n");
}
