export const PLACEMENT_COMPARISON_SCHEMA = "deepbom.placement_comparison.v1";

export function buildPlacementComparison(analysis, profiles, { selectedProfileIds = null } = {}) {
  const available = Array.isArray(profiles) ? profiles : [];
  const byId = new Map(available.map((profile) => [String(profile.profile_id), profile]));
  if (byId.size !== available.length) throw new Error("Placement comparison profile IDs must be unique.");
  const selected = selectedProfileIds == null
    ? [...byId.keys()]
    : [...new Set(selectedProfileIds.map(String))];
  for (const id of selected) if (!byId.has(id)) throw new Error(`Placement comparison profile is unavailable: ${id}.`);
  const rows = selected.map((id) => comparisonRow(byId.get(id)));
  const result = {
    schema: PLACEMENT_COMPARISON_SCHEMA,
    method_version: "1.0.0",
    artifact_sha256: String(analysis?.model_sha256 || ""),
    format: String(analysis?.format || "unknown").toLowerCase(),
    graph_op_count: Array.isArray(analysis?.ops) ? analysis.ops.length : 0,
    available_profile_ids: [...byId.keys()],
    selected_profile_ids: selected,
    rows,
    assessment_status: rows.length ? "assessed_independent_profiles" : "not_applicable_no_static_profiles",
    conservation: {
      selected_profile_count: selected.length,
      emitted_row_count: rows.length,
      all_rows_cover_graph: rows.every((row) => row.op_count === (Array.isArray(analysis?.ops) ? analysis.ops.length : 0)),
    },
    interpretation_boundary: "N-way comparison of independent source-backed or compiler-observed placement profiles over one canonical graph and tensor ledger. It does not infer provider priority, joint partitioning, physical transfer, generated kernels, latency, or task correctness.",
  };
  validatePlacementComparison(result, analysis, available);
  return result;
}

export function validatePlacementComparison(value, analysis = null, profiles = null) {
  const issues = [];
  if (value?.schema !== PLACEMENT_COMPARISON_SCHEMA) issues.push("schema mismatch");
  if (!Array.isArray(value?.available_profile_ids) || !Array.isArray(value?.selected_profile_ids) || !Array.isArray(value?.rows)) issues.push("profile inventories missing");
  if (new Set(value?.available_profile_ids || []).size !== (value?.available_profile_ids || []).length) issues.push("available profile IDs duplicate");
  if (new Set(value?.selected_profile_ids || []).size !== (value?.selected_profile_ids || []).length) issues.push("selected profile IDs duplicate");
  if ((value?.rows || []).length !== (value?.selected_profile_ids || []).length) issues.push("selected profile rows do not conserve selection");
  for (const row of value?.rows || []) {
    const sum = Number(row.conditionally_eligible_ops) + Number(row.definite_exclusion_ops) + Number(row.unresolved_ops);
    if (!value.selected_profile_ids.includes(row.profile_id) || sum !== row.op_count || row.op_count !== value.graph_op_count) {
      issues.push(`profile row does not conserve graph ops: ${row.profile_id}`);
    }
  }
  if (analysis) {
    if (String(value.artifact_sha256 || "") !== String(analysis.model_sha256 || "")) issues.push("artifact binding mismatch");
    if (Number(value.graph_op_count) !== (Array.isArray(analysis.ops) ? analysis.ops.length : 0)) issues.push("graph op count mismatch");
  }
  if (profiles) {
    const ids = profiles.map((profile) => String(profile.profile_id));
    if (JSON.stringify(value.available_profile_ids) !== JSON.stringify(ids)) issues.push("available profile inventory mismatch");
  }
  if (!value?.conservation?.all_rows_cover_graph || value?.conservation?.selected_profile_count !== value?.conservation?.emitted_row_count) issues.push("comparison conservation failed");
  if (issues.length) throw new Error(`Placement comparison is invalid: ${issues.join("; ")}`);
  return true;
}

export function placementProfileClass(profile) {
  const id = `${profile?.profile_id || ""} ${profile?.label || ""}`.toLowerCase();
  if (/(gpu|directml|webgpu|webnn|nnapi|qnn|coreml|edgetpu|tpu|npu|tensorrt|cuda|rocm|metal|vulkan|opencl)/.test(id)) return "accelerator";
  if (/(cpu|xnnpack|wasm)/.test(id)) return "cpu";
  return "other";
}

function comparisonRow(profile) {
  const counts = profile.state_counts || {};
  const workload = profile.workload_envelope || {};
  return {
    profile_id: String(profile.profile_id),
    label: String(profile.label),
    profile_class: placementProfileClass(profile),
    evidence_class: String(profile.evidence_class || "NOT_ASSESSED"),
    assessment_status: String(profile.assessment_status || "not_assessed"),
    op_count: Number(profile.op_count || 0),
    conditionally_eligible_ops: Number(counts.CONDITIONALLY_ELIGIBLE || 0),
    definite_exclusion_ops: Number(counts.DEFINITE_EXCLUSION || 0),
    unresolved_ops: Number(counts.UNRESOLVED || 0),
    segment_count: Number(profile.segment_count || 0),
    boundary_edge_count: Number(profile.boundary_edge_count || 0),
    boundary_payload: profile.boundary_payload || null,
    conditionally_eligible_mac_share_decimal: workload.conditionally_eligible_mac_share_decimal ?? null,
    conditionally_eligible_logical_byte_share_decimal: workload.conditionally_eligible_logical_byte_share_decimal ?? null,
    unresolved_condition_type_count: new Set((profile.rows || []).flatMap((row) => row.unresolved_predicates || [])).size,
    source: profile.source || null,
    interpretation_boundary: profile.interpretation_boundary,
  };
}
