export function coverageResiduals(analysis) {
  if (analysis?.format !== "onnx") return null;
  const shape = analysis.onnx_shape_inference || {};
  const mac = analysis.mac_assessment || {};
  const dynamic = analysis.dynamic_shape_cost_contract || {};
  const domain = analysis.onnx_domain_analysis || {};
  const unsupportedRows = shape.rule_unsupported_node_rows || [];
  return {
    schema: "deepbom.corpus_coverage_residuals.v1.5",
    analyzer_contracts: {
      onnx_shape_inference: String(shape.schema || "not_emitted"),
      dynamic_shape_cost: String(dynamic.schema || "not_emitted"),
    },
    graph_node_count: Number(analysis.operator_count || 0),
    external_custom_domain_node_count: Number(domain.external_custom_node_count || 0),
    ort_contrib_node_count: Number(domain.ort_contrib_node_count || 0),
    ort_contrib_shape_rule_unsupported_node_count: unsupportedRows.filter((row) => row.domain === "com.microsoft").length,
    shape_rule_unsupported_node_count: Number(shape.rule_unsupported_nodes || 0),
    shape_rule_unsupported_op_histogram: (shape.rule_unsupported_op_histogram || []).map((row) => ({
      name: String(row.name || "UNKNOWN"),
      count: Number(row.count || 0),
    })),
    shape_rule_unresolved_node_count: Number(shape.rule_unresolved_node_count || 0),
    shape_rule_unresolved_op_histogram: countRows(shape.rule_unresolved_nodes, "op_name"),
    shape_rule_unresolved_reason_histogram: countRows(shape.rule_unresolved_nodes, "reason"),
    shape_rule_unresolved_op_reason_histogram: countRows(
      (shape.rule_unresolved_nodes || []).map((row) => ({ key: `${row.op_name || "UNKNOWN"}:${row.reason || "unknown_reason"}` })),
      "key",
    ),
    unknown_node_output_count: Number(shape.unknown_node_output_count || 0),
    shape_contract_known_node_output_count: Number(shape.shape_contract_known_node_output_count || 0),
    shape_contract_unknown_node_output_count: Number(shape.shape_contract_unknown_node_output_count || 0),
    invalid_node_output_count: Number(shape.invalid_node_output_count || 0),
    conditionally_invalid_node_output_count: Number(shape.conditionally_invalid_node_output_count || 0),
    conditional_invalid_variant_count: Number(shape.conditional_invalid_variant_count || 0),
    conditional_unassessed_variant_count: Number(shape.conditional_unassessed_variant_count || 0),
    unresolved_nonconflict_shape_contract_node_output_count: Number(shape.unresolved_nonconflict_shape_contract_node_output_count || 0),
    declaration_conflict_count: Number(shape.declaration_conflict_count || 0),
    semantic_contract_conflict_count: Number(shape.semantic_contract_conflict_count || 0),
    blocked_by_upstream_contract_conflict_node_count: Number(shape.blocked_by_upstream_contract_conflict_node_count || 0),
    symbolic_shape_contract_node_output_count: Number(shape.symbolic_shape_contract_node_output_count || 0),
    conditional_shape_contract_node_output_count: Number(shape.conditional_shape_contract_node_output_count || 0),
    partial_conditional_shape_contract_node_output_count: Number(shape.partial_conditional_shape_contract_node_output_count || 0),
    unassessed_compute_op_count: Number(mac.not_assessed_compute_ops || 0),
    unassessed_compute_op_histogram: countRows(mac.not_assessed, "op_name"),
    unassessed_compute_reason_histogram: countRows(mac.not_assessed, "reason"),
    unassessed_compute_op_reason_histogram: countRows((mac.not_assessed || []).map((row) => ({ key: `${row.name || row.op_name || "UNKNOWN"}:${row.reason || "unknown_reason"}` })), "key"),
    algorithm_dependent_arithmetic_op_count: Number(mac.algorithm_dependent_arithmetic_ops || 0),
    algorithm_dependent_arithmetic_op_histogram: countRows(mac.algorithm_dependent_arithmetic, "op_name"),
    unresolved_dynamic_compute_op_count: Number(dynamic.unresolved_dynamic_compute_op_count || 0),
    unresolved_dynamic_compute_op_histogram: countRows(dynamic.unresolved_dynamic_compute_ops, "op_name"),
    unresolved_dynamic_compute_reason_histogram: countRows(dynamic.unresolved_dynamic_compute_ops, "reason"),
    dynamic_tensor_count: Number(dynamic.dynamic_tensor_count || 0),
    total_macs_unresolved_op_count: Number(dynamic.total_macs_unresolved_op_count || 0),
    total_macs_artifact_contract_conflict_op_count: Number(dynamic.total_macs_artifact_contract_conflict_op_count || 0),
    total_macs_external_binding_required_op_count: Number(dynamic.total_macs_external_binding_required_op_count || 0),
    total_macs_analyzer_or_contract_residual_op_count: Number(dynamic.total_macs_analyzer_or_contract_residual_op_count || 0),
    total_macs_unresolved_op_histogram: countRows(dynamic.total_macs_unresolved_ops, "op_name"),
    total_macs_unresolved_reason_histogram: countRows(dynamic.total_macs_unresolved_ops, "reason"),
    total_macs_unresolved_resolution_histogram: countRows(dynamic.total_macs_unresolved_ops, "resolution_class"),
    total_macs_unresolved_op_reason_histogram: countRows((dynamic.total_macs_unresolved_ops || []).map((row) => ({ key: `${row.op_name || "UNKNOWN"}:${row.reason || "unknown_reason"}` })), "key"),
    external_data_status: analysis.onnx_external_data?.status || "not_emitted",
  };
}

function countRows(rows, key) {
  const counts = new Map();
  for (const row of rows || []) {
    const name = String(row?.[key] || row?.name || "UNKNOWN");
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([name, count]) => ({ name, count }));
}
