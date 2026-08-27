import { formatBytes, formatNumber, formatPercent, formatUs, padOp } from "./format.js";
import { code, markdownTable } from "./report-utils.js";
import { staticL2RatioForTarget } from "./report-engineering-derivations.js";

function isOnnxAnalysis(analysis) {
  return String(analysis?.format || "").toLowerCase() === "onnx";
}
export function deploymentFrontierMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) {
    const frontier = analysis?.ort_ep_portability_frontier;
    if (!frontier) return [
      "## ONNX EP Portability Frontier (NOT_ASSESSED)",
      "The protected source-backed ORT rulepack was not loaded, so no cross-EP source match intersection is emitted.",
    ].join("\n");
    const topGaps = (frontier.providers || []).flatMap((provider) => (provider.top_gap_ops || []).slice(0, 6).map((gap) => ({ provider: provider.execution_provider, ...gap })))
      .sort((left, right) => Number(right.assessed_macs ?? -1) - Number(left.assessed_macs ?? -1) || left.op_index - right.op_index)
      .slice(0, 16);
    return [
      "## ONNX EP Portability Frontier (DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS)",
      markdownTable(["Metric", "Value"], [
        ["Schema / evidence class", `${frontier.schema || "not embedded"} / ${frontier.evidence_class || "not emitted"}`],
        ["Execution-provider rule sets", formatNumber(frontier.execution_provider_count)],
        ["All-EP source match intersection", `${formatNumber(frontier.all_ep_source_match_op_count)}/${formatNumber(frontier.op_count)} op(s) (${formatPercent(frontier.all_ep_source_match_op_ratio)})`],
        ["All-EP assessed-MAC intersection", frontier.all_ep_source_match_mac_ratio == null ? "not assessed; no positive assessed MAC denominator" : `${formatNumber(frontier.all_ep_source_match_macs)}/${formatNumber(frontier.assessed_mac_total)} (${formatPercent(frontier.all_ep_source_match_mac_ratio)})`],
        ["All-EP narrowed artifact-precheck candidates", `${formatNumber(frontier.all_ep_artifact_precheck_candidate_op_count)}/${formatNumber(frontier.op_count)} op(s) (${formatPercent(frontier.all_ep_artifact_precheck_candidate_op_ratio)})`],
        ["All-EP narrowed candidate assessed MACs", frontier.all_ep_artifact_precheck_candidate_mac_ratio == null ? `${formatNumber(frontier.all_ep_artifact_precheck_candidate_macs)}; ratio not assessed because there is no positive assessed-MAC denominator` : `${formatNumber(frontier.all_ep_artifact_precheck_candidate_macs)}/${formatNumber(frontier.assessed_mac_total)} (${formatPercent(frontier.all_ep_artifact_precheck_candidate_mac_ratio)})`],
        ["Method", frontier.method],
      ]),
      markdownTable(["EP", "Source matches", "Narrowed candidates", "Definite exclusions", "Narrowed assessed MACs", "Top gap/exclusion"], (frontier.providers || []).map((provider) => [
        provider.execution_provider,
        `${formatNumber(provider.source_match_op_count)}/${formatNumber(frontier.op_count)} (${formatPercent(provider.source_match_op_ratio)})`,
        `${formatNumber(provider.artifact_precheck_candidate_op_count)}/${formatNumber(frontier.op_count)} (${formatPercent(provider.artifact_precheck_candidate_op_ratio)})`,
        formatNumber(provider.artifact_precheck_definite_fail_op_count),
        provider.artifact_precheck_candidate_assessed_mac_ratio == null ? "N/A" : `${formatNumber(provider.artifact_precheck_candidate_assessed_macs)} (${formatPercent(provider.artifact_precheck_candidate_assessed_mac_ratio)})`,
        provider.top_gap_ops?.length ? `#${padOp(provider.top_gap_ops[0].op_index)} ${provider.top_gap_ops[0].op_name}: ${provider.top_gap_ops[0].gap_class} / ${provider.top_gap_ops[0].status}` : "none",
      ])),
      topGaps.length ? markdownTable(["EP", "Op", "Gap class", "Status", "Assessed MACs"], topGaps.map((gap) => [
        gap.provider,
        `#${padOp(gap.op_index)} ${gap.op_name}`,
        gap.gap_class,
        gap.status,
        gap.assessed_macs == null ? "N/A" : formatNumber(gap.assessed_macs),
      ])) : "No pinned source-version gap or definite artifact-condition exclusion was emitted.",
      `> ${frontier.evidence_boundary}`,
    ].join("\n\n");
  }
  const frontier = analysis?.deployment_frontier;
  if (!frontier) return [
    "## Deployment Frontier (NOT_ASSESSED)",
    analysis?.deployment_frontier_error || "Cross-target planning profiles were not evaluated.",
  ].join("\n");
  const selected = (frontier.ops || []).filter((op) => op.in_robust_coverage_union).slice(0, 20);
  const mostDivergentPair = [...(frontier.target_divergence?.pairs || [])]
    .sort((left, right) => right.normalized_jensen_shannon_divergence - left.normalized_jensen_shannon_divergence)[0] || null;
  const targetLabel = (targetId) => frontier.targets.find((target) => target.target_id === targetId)?.target_label || targetId;
  const allDivergenceDrivers = (mostDivergentPair?.drivers || []).filter((driver) => driver.normalized_js_contribution > 0);
  const divergenceDrivers = allDivergenceDrivers.slice(0, 12);
  const displayedDivergenceCoverage = divergenceDrivers.reduce((sum, driver) => sum + Number(driver.attribution_share || 0), 0);
  const targetById = new Map((frontier.targets || []).map((target) => [target.target_id, target]));
  const interventionTargetRows = (frontier.interventions || []).flatMap((item) => (item.per_target || []).map((row) => {
    const target = targetById.get(row.target_id) || {};
    const steadyDenominator = Number(target.steady_state_us ?? Math.max(0,
      Number(target.cold_start_us || 0)
      - Number(target.components?.packing_us || 0)
      - Number(target.components?.boundary_us || 0)));
    const denominatorKind = item.removed_component === "fallback" ? "steady" : "cold";
    const denominator = denominatorKind === "steady" ? steadyDenominator : Number(target.cold_start_us || 0);
    return [
      item.label,
      targetLabel(row.target_id),
      formatUs(row.recoverable_us),
      `${formatUs(denominator)} ${denominatorKind}`,
      formatPercent(row.recoverable_share),
      `${Number(row.upper_bound_speedup || 0).toFixed(3)}x`,
      item.removed_component === "packing"
        ? `${Number(target.weight_packing_bandwidth_gbps || 0).toFixed(1)} GB/s planning profile`
        : item.removed_component === "boundary"
          ? `${formatUs(target.boundary_setup_midpoint_us || 0)} setup midpoint`
          : `${formatUs(target.components?.fallback_us || 0)} fallback component`,
    ];
  }));
  const pointOrRange = (low, high) => Math.abs(Number(high || 0) - Number(low || 0)) <= 1e-9
    ? formatUs(low)
    : `${formatUs(low)}-${formatUs(high)}`;
  return [
    "## Deployment Frontier (DERIVED/ESTIMATED)",
    markdownTable(["Metric", "Value"], [
      ["Profiles", `${formatNumber(frontier.target_count)} pinned planning targets`],
      ["Robust steady-state 80% coverage union", `${formatNumber(frontier.robust_coverage.selected_op_count)}/${formatNumber(frontier.op_count)} op(s); minimum target coverage ${formatPercent(frontier.robust_coverage.minimum_union_coverage)}`],
      ["Mean normalized target divergence", frontier.target_divergence.mean_normalized_jensen_shannon_divergence.toFixed(4)],
      ["Maximum normalized target divergence", frontier.target_divergence.max_normalized_jensen_shannon_divergence.toFixed(4)],
      ["Minimum target-prefix overlap", formatPercent(frontier.target_divergence.min_coverage_prefix_jaccard)],
      ["Maximum hotspot rank span", `${formatNumber(Math.max(0, ...(frontier.ops || []).map((op) => Number(op.rank_span || 0))))} rank position(s)`],
      ["Method", frontier.method],
    ]),
    markdownTable(["Target", "L1D / L2", "Max row-WS L1 / L2", "Cache watch L1 / L2", "Steady / cold modeled total", "80% prefix ops", "Robust-union coverage", "Top op"], (frontier.targets || []).map((target) => {
      const coverage = frontier.robust_coverage.per_target.find((item) => item.target_id === target.target_id);
      const coldUs = Number(target.cold_start_us ?? target.total_us ?? 0);
      const steadyUs = Number(target.steady_state_us ?? Math.max(
        0,
        coldUs
          - Number(target.components?.packing_us || 0)
          - Number(target.components?.boundary_us || 0),
      ));
      return [
        target.target_label,
        `${formatBytes(target.l1_data_bytes)} / ${formatBytes(target.l2_bytes)}`,
        `${Number(target.max_l1_ratio || 0).toFixed(2)}x / ${target.max_l2_ratio == null ? `${staticL2RatioForTarget(analysis, target)} static (concurrency watch N/A)` : `${Number(target.max_l2_ratio).toFixed(2)}x`}`,
        `${formatNumber(target.l1_watch_count || 0)} / ${target.l2_watch_count == null ? `N/A (${target.l2_capacity_scope || "scope unbound"})` : formatNumber(target.l2_watch_count)} (>=${Number(frontier.cache_watch_ratio || 0.9).toFixed(2)}x)`,
        `${pointOrRange(target.steady_state_low_us ?? steadyUs, target.steady_state_high_us ?? steadyUs)} steady / ${pointOrRange(target.cold_start_low_us ?? coldUs, target.cold_start_high_us ?? coldUs)} cold${Math.abs(Number(target.cold_start_high_us ?? coldUs) - Number(target.cold_start_low_us ?? coldUs)) <= 1e-9 ? "" : ` (mid ${formatUs(coldUs)})`}`,
        formatNumber(coverage?.selected_prefix_op_count || 0),
        formatPercent(coverage?.union_coverage || 0),
        target.top_op_index == null ? "none" : `#${padOp(target.top_op_index)} ${target.top_op_name}: ${formatUs(target.top_op_steady_state_us ?? target.top_op_us)} steady`,
      ];
    })),
    markdownTable(["Counterfactual", "Removed modeled component", "Recoverable share range", "Upper-bound speedup range", "Evidence"], (frontier.interventions || []).map((item) => [
      item.label,
      item.removed_component,
      `${formatPercent(item.min_recoverable_share)} to ${formatPercent(item.max_recoverable_share)}`,
      `${item.min_upper_bound_speedup.toFixed(3)}x to ${item.max_upper_bound_speedup.toFixed(3)}x`,
      item.evidence_class,
    ])),
    interventionTargetRows.length
      ? `### Counterfactual Denominator Ledger\n\n${markdownTable(["Counterfactual", "Target", "Removed", "Denominator", "Share", "Speedup", "Profile constant"], interventionTargetRows)}\n\nPacking and predicted-boundary setup shares use each target's cold-start point total. Fallback shares use the steady-state point total because fallback remains in steady execution. Each row exposes its denominator explicitly.`
      : "",
    selected.length ? markdownTable(["Robust hotspot", "Contribution range", "Rank range", "Bound classes", "Dominant components"], selected.map((op) => [
      `#${padOp(op.op_index)} ${op.op_name}`,
      `${formatPercent(op.min_contribution_share)} to ${formatPercent(op.max_contribution_share)}`,
      op.rank_span ? `${op.best_rank}-${op.worst_rank}` : `#${op.best_rank}`,
      op.bound_classes.join(" / "),
      op.dominant_components.join(" / "),
    ])) : "No positive modeled hotspot entered the target coverage union.",
    mostDivergentPair ? [
      "### Maximum-Pair Divergence Attribution (DERIVED)",
      markdownTable(["Field", "Value"], [
        ["Target pair", `${targetLabel(mostDivergentPair.left_target_id)} / ${targetLabel(mostDivergentPair.right_target_id)}`],
        ["Normalized JSD", mostDivergentPair.normalized_jensen_shannon_divergence.toFixed(8)],
        ["Attribution conservation", `${mostDivergentPair.attribution_sum.toFixed(8)} = normalized JSD`],
        ["80% explanation prefix", `${formatNumber(mostDivergentPair.attribution_prefix_op_count)} op(s), ${formatPercent(mostDivergentPair.attribution_prefix_coverage)} of pair divergence`],
        ["Bound/component transitions", `${formatNumber(mostDivergentPair.bound_transition_op_count)} / ${formatNumber(mostDivergentPair.dominant_component_transition_op_count)}`],
      ]),
      divergenceDrivers.length ? markdownTable(["Op", "JSD attribution", "Modeled contribution left/right", "Rank left/right", "Largest signed component shift", "Class transition"], divergenceDrivers.map((driver) => {
        const component = driver.component_contribution_delta.largest_absolute_component;
        const delta = Number(driver.component_contribution_delta[component] || 0) * 100;
        const transitions = [
          driver.bound_transition ? `${driver.left_bound}->${driver.right_bound}` : "bound stable",
          driver.dominant_component_transition ? `${driver.left_dominant_component}->${driver.right_dominant_component}` : "component stable",
        ].join(" / ");
        return [
          `#${padOp(driver.op_index)} ${driver.op_name}`,
          formatPercent(driver.attribution_share),
          `${formatPercent(driver.left_contribution_share)} / ${formatPercent(driver.right_contribution_share)}`,
          `#${driver.left_rank} / #${driver.right_rank}`,
          `${component} ${delta > 0 ? "+" : ""}${delta.toFixed(2)} pp`,
          transitions,
        ];
      })) : "No positive per-op divergence term was present.",
      allDivergenceDrivers.length > divergenceDrivers.length
        ? `Displayed ${formatNumber(divergenceDrivers.length)} of ${formatNumber(allDivergenceDrivers.length)} positive attribution rows, accounting for ${formatPercent(displayedDivergenceCoverage)} of pair divergence. The 80% prefix count above is computed over the complete ordered driver ledger; ${formatNumber(allDivergenceDrivers.length - divergenceDrivers.length)} additional rows remain in structured evidence.`
        : `Displayed all ${formatNumber(allDivergenceDrivers.length)} positive attribution rows.`,
    ].join("\n\n") : "### Maximum-Pair Divergence Attribution (NOT_ASSESSED)\nNo target pair was emitted.",
    `> ${frontier.interpretation_boundary}`,
  ].join("\n\n");
}

export function deploymentDeltaMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const delta = analysis?.deployment_delta;
  if (!delta) return [
    "## Deployment Delta (NOT_ASSESSED)",
    analysis?.deployment_delta_error || "No distinct in-memory TFLite baseline was bound to this audit.",
  ].join("\n");
  const signedUs = (value) => `${Number(value) > 0 ? "+" : Number(value) < 0 ? "-" : ""}${formatUs(Math.abs(Number(value || 0)))}`;
  const signedPercent = (value) => value == null ? "N/A" : `${Number(value) > 0 ? "+" : ""}${formatPercent(value)}`;
  const signedNumber = (value) => `${Number(value) > 0 ? "+" : ""}${formatNumber(value)}`;
  const topDrivers = (delta.cross_target_drivers || [])
    .filter((driver) => driver.consistent_regression || driver.consistent_improvement)
    .slice(0, 16);
  const maxConservationError = Math.max(0, ...(delta.target_deltas || []).map((target) => Math.abs(Number(target.conservation_error_us || 0))));
  return [
    "## Deployment Delta (DERIVED/ESTIMATED)",
    markdownTable(["Field", "Baseline", "Candidate", "Change"], [
      ["Artifact", code(delta.baseline.filename), code(delta.candidate.filename), delta.alignment.artifact_relation],
      ["SHA-256", code(delta.baseline.sha256), code(delta.candidate.sha256), delta.baseline.sha256 === delta.candidate.sha256 ? "identical bytes" : "different bytes; lineage unproven"],
      ["File size", formatBytes(delta.baseline.file_size), formatBytes(delta.candidate.file_size), `${Number(delta.graph_delta.signed_file_size_bytes) >= 0 ? "+" : "-"}${formatBytes(Math.abs(delta.graph_delta.signed_file_size_bytes))}`],
      ["Operators", formatNumber(delta.baseline.operator_count), formatNumber(delta.candidate.operator_count), signedNumber(delta.graph_delta.signed_operator_count)],
      ["Tensors", formatNumber(delta.baseline.tensor_count), formatNumber(delta.candidate.tensor_count), signedNumber(delta.graph_delta.signed_tensor_count)],
      ["MACs", formatNumber(delta.baseline.total_macs), formatNumber(delta.candidate.total_macs), `${signedNumber(delta.graph_delta.signed_total_macs)} (${signedPercent(delta.graph_delta.relative_total_macs_delta)})`],
      ["Quantized compute MAC share", formatPercent(delta.baseline.quantized_compute_mac_ratio), formatPercent(delta.candidate.quantized_compute_mac_ratio), `${Number(delta.graph_delta.signed_quantized_compute_mac_ratio) * 100 >= 0 ? "+" : ""}${(Number(delta.graph_delta.signed_quantized_compute_mac_ratio) * 100).toFixed(1)} pp`],
      ["I/O contract", `${delta.baseline.input_contracts.join(", ")} -> ${delta.baseline.output_contracts.join(", ")}`, `${delta.candidate.input_contracts.join(", ")} -> ${delta.candidate.output_contracts.join(", ")}`, `input ${delta.graph_delta.input_contract_changed ? "changed" : "stable"}; output ${delta.graph_delta.output_contract_changed ? "changed" : "stable"}`],
    ]),
    markdownTable(["Alignment evidence", "Count / conclusion"], [
      ["Exact structural signatures", formatNumber(delta.alignment.exact_structural_match_count)],
      ["Op-type sequence coordinates", formatNumber(delta.alignment.op_sequence_match_count)],
      ["Matched / added / removed", `${formatNumber(delta.alignment.matched_op_count)} / ${formatNumber(delta.alignment.added_op_count)} / ${formatNumber(delta.alignment.removed_op_count)}`],
      ["Baseline / candidate match coverage", `${formatPercent(delta.alignment.baseline_match_ratio)} / ${formatPercent(delta.alignment.candidate_match_ratio)}`],
      ["Semantic identity", delta.alignment.semantic_identity_conclusion],
    ]),
    markdownTable(["Target", "Baseline", "Candidate", "Signed delta", "Compute", "Memory", "Packing", "Boundary", "Fallback", "Conservation error"], (delta.target_deltas || []).map((target) => [
      target.target_label,
      formatUs(target.baseline_total_us),
      formatUs(target.candidate_total_us),
      `${signedUs(target.signed_delta_us)} (${signedPercent(target.relative_delta)})`,
      signedUs(target.component_delta.compute_us),
      signedUs(target.component_delta.memory_us),
      signedUs(target.component_delta.packing_us),
      signedUs(target.component_delta.boundary_us),
      signedUs(target.component_delta.fallback_us),
      `${Number(target.conservation_error_us).toExponential(3)} us`,
    ])),
    topDrivers.length ? markdownTable(["Aligned entity", "Baseline op", "Candidate op", "Relation", "Target consistency", "Delta range"], topDrivers.map((driver) => [
      driver.entity_id,
      driver.baseline_op_index == null ? "-" : `#${padOp(driver.baseline_op_index)} ${driver.baseline_op_name}`,
      driver.candidate_op_index == null ? "-" : `#${padOp(driver.candidate_op_index)} ${driver.candidate_op_name}`,
      `${driver.relation}; ${driver.match_class}`,
      driver.consistent_regression ? `${driver.regression_target_count}/${delta.target_count} regressions` : `${driver.improvement_target_count}/${delta.target_count} improvements`,
      `${signedUs(driver.min_delta_us)} to ${signedUs(driver.max_delta_us)}`,
    ])) : "No cross-target-consistent nonzero driver was emitted.",
    `Maximum independent target-ledger conservation error: ${maxConservationError.toExponential(3)} us.`,
    `> ${delta.interpretation_boundary}`,
  ].join("\n\n");
}

export function delegationRepairMarkdown(analysis) {
  if (isOnnxAnalysis(analysis)) return "";
  const repair = analysis?.delegation_repair;
  if (!repair) return [
    "## Delegation Repair Lab (NOT_ASSESSED)",
    analysis?.delegation_repair_error || "Single-op delegation counterfactuals were not evaluated.",
  ].join("\n");
  const byIndex = new Map((repair.toggles || []).map((row) => [row.op_index, row]));
  const islandByIndex = new Map((repair.cpu_islands || []).map((row) => [row.island_index, row]));
  const repairRows = (repair.repair_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean).slice(0, 16);
  const fragilityRows = (repair.fragility_ranking_op_indices || []).map((index) => byIndex.get(index)).filter(Boolean).slice(0, 16);
  const unrankedRows = (repair.toggles || []).filter((row) => !row.repair_opportunity && !row.fragmentation_risk);
  const islandRows = (repair.cpu_island_ranking_indices || []).map((index) => islandByIndex.get(index)).filter(Boolean).slice(0, 16);
  const signed = (value) => `${Number(value) > 0 ? "+" : ""}${formatNumber(value || 0)}`;
  const signedBytes = (value) => value == null ? "N/A" : `${Number(value) > 0 ? "+" : Number(value) < 0 ? "-" : ""}${formatBytes(Math.abs(Number(value)))}`;
  const rankedRows = (rows, risk) => rows.map((row) => [
    `#${risk ? row.fragility_rank : row.repair_rank}`,
    `#${padOp(row.op_index)} ${row.op_name}`,
    row.baseline_xnnpack_reason,
    row.outcome_class,
    signed(row.signed_delegate_segment_count),
    signed(row.signed_boundary_edge_count),
    `${signedBytes(row.signed_boundary_payload_bytes)} net; ${row.baseline_incident_boundary_payload_bytes == null ? "N/A" : formatBytes(row.baseline_incident_boundary_payload_bytes)} gross incident`,
    `${formatNumber(row.removed_boundary_edge_count)} removed / ${formatNumber(row.added_boundary_edge_count)} added / ${formatNumber(row.reclassified_boundary_edge_count)} reclassified`,
  ]);
  const topRepair = repairRows[0] || null;
  const topIsland = islandRows.find((island) => island.group_only_repair) || islandRows[0] || null;
  const edgeLedgerMarkdown = (changes, empty) => changes?.length
    ? markdownTable(["Change", "Tensor", "Producer", "Consumer", "Payload", "Payload binding", "Direction"], changes.map((edge) => [
        edge.transition,
        `T${edge.tensor_index} ${code(edge.tensor_name || "-")} ${edge.tensor_dtype} [${(edge.tensor_shape || []).join("x")}]`,
        `#${padOp(edge.producer_op_index)} ${edge.producer_op_name}`,
        `#${padOp(edge.consumer_op_index)} ${edge.consumer_op_name}`,
        edge.payload_bytes == null ? "N/A" : formatBytes(edge.payload_bytes),
        `${edge.payload_status || "not assessed"}; ${edge.payload_binding || "unbound"}`,
        `${edge.baseline_direction || "none"} -> ${edge.counterfactual_direction || "none"}`,
      ]))
    : empty;
  const edgeLedger = edgeLedgerMarkdown(topRepair?.edge_changes, "No boundary edge changed for the highest-ranked repair candidate.");
  const islandEdgeLedger = edgeLedgerMarkdown(topIsland?.edge_changes, "No boundary edge changed for the selected complete CPU-island intervention.");
  return [
    "## Delegation Repair Lab (PREDICTED/DERIVED COUNTERFACTUAL)",
    markdownTable(["Metric", "Value"], [
      ["Target", `${repair.target_label} (${repair.target_id})`],
      ["Operators toggled exactly once", `${formatNumber(repair.toggles?.length || 0)}/${formatNumber(repair.operator_count)}`],
      ["Baseline predicted delegate / CPU segments", `${formatNumber(repair.baseline.delegate_segment_count)} / ${formatNumber(repair.baseline.cpu_segment_count)}`],
      ["Baseline boundary edges", `${formatNumber(repair.baseline.boundary_edge_count)} across ${formatNumber(repair.graph_edge_count)} graph edge(s)`],
      ["Baseline summed logical boundary payload", repair.baseline.summed_edge_payload_bytes == null ? `PARTIAL; ${formatBytes(repair.baseline.assessed_edge_payload_bytes)} assessed` : `${formatBytes(repair.baseline.summed_edge_payload_bytes)} (${repair.baseline.summed_edge_payload_bytes} B)`],
      ["Toggle outcome conservation", `${formatNumber(repair.toggles?.length || 0)} = ${formatNumber(repair.repair_opportunity_count)} repair + ${formatNumber(repair.fragmentation_risk_count)} fragility + ${formatNumber(unrankedRows.length)} unranked other effect(s); the unranked set contains ${formatNumber(repair.no_static_effect_count)} no-static-effect row(s)`],
      ["CPU islands / full-island repairs / group-only repairs", `${formatNumber(repair.cpu_island_count)} / ${formatNumber(repair.full_segment_repair_count)} / ${formatNumber(repair.group_only_repair_count)}`],
      ["Grouped export / runtime-build scenarios", `${formatNumber(repair.export_interventions?.length || 0)} / ${formatNumber(repair.runtime_build_risks?.length || 0)}`],
      ["Singleton delegate segments", formatNumber(repair.singleton_delegate_segments?.length || 0)],
      ["Method version", repair.method_version],
      ["Evidence class", repair.evidence_class],
    ]),
    repair.export_interventions?.length
      ? `### Actionable Export Interventions\n\n${markdownTable(["Intervention", "Matched / excluded evidence", "Combined assignment proxy", "Independent sum / interaction", "Action / boundary"], repair.export_interventions.map((item) => [
          item.title,
          `${formatNumber(item.block_count)} exact SE block(s); MEAN ${item.mean_op_indices.map((index) => `#${padOp(index)}`).join(", ")}; EXPAND_DIMS ${item.expand_dims_op_indices.map((index) => `#${padOp(index)}`).join(", ")}; excluded rank4 MEAN ${(item.unmatched_rank4_mean_op_indices || []).map((index) => `#${padOp(index)}`).join(", ") || "none"}; axes [${item.derived_reduction_axes.join(", ")}]`,
          `${signed(item.signed_delegate_segment_count)} delegate segment(s); ${signed(item.signed_cpu_segment_count)} CPU island(s); ${signed(item.signed_boundary_edge_count)} boundary edge(s); ${signedBytes(item.signed_boundary_payload_bytes)} payload`,
          `${signed(item.independent_single_toggle_signed_boundary_edge_count_sum)} edges / ${signedBytes(item.independent_single_toggle_signed_boundary_payload_bytes_sum)} independently; ${signed(item.interaction_signed_boundary_edge_count)} edges / ${signedBytes(item.interaction_signed_boundary_payload_bytes)} combined interaction`,
          `${item.action} ${item.interpretation_boundary}`,
        ]))}`
      : "### Actionable Export Interventions\n\nNo export-pattern intervention met the complete structural preconditions.",
    repair.runtime_build_risks?.length
      ? `### Runtime Build Configuration Risks\n\n${markdownTable(["Required configuration", "Conditionally delegatable ops affected", "If the build condition is absent", "Evidence boundary"], repair.runtime_build_risks.map((risk) => [
          code(risk.required_build_configuration),
          `${formatNumber(risk.affected_conditionally_delegatable_op_count)}/${formatNumber(risk.baseline_conditionally_delegatable_op_count)} conditionally delegatable op(s); ${formatNumber(risk.affected_predicted_delegate_segment_count)} predicted delegate segment(s); ${formatPercent(risk.affected_conditionally_delegatable_mac_ratio)} modeled MACs`,
          `${formatNumber(risk.absent_condition_remaining_conditionally_delegatable_op_count)} remaining conditionally delegatable op(s) / ${formatNumber(risk.absent_condition_remaining_predicted_delegate_segment_count)} remaining predicted delegate segment(s)`,
          risk.interpretation_boundary,
        ]))}`
      : "### Runtime Build Configuration Risks\n\nNo conditionally delegatable op carried an explicit build requirement.",
    repairRows.length
      ? `### Hypothetical Support-Extension Merge Candidates\n\n${markdownTable(["Rank", "Op", "Current rule", "Outcome", "Delegate segment delta", "Boundary edge delta", "Boundary payload delta", "Edge changes"], rankedRows(repairRows, false))}`
      : "### Hypothetical Support-Extension Merge Candidates\n\nNo predicted CPU op reduced a static boundary under a single-op toggle.",
    repair.repair_opportunity_count > repairRows.length
      ? `Displayed ${formatNumber(repairRows.length)} of ${formatNumber(repair.repair_opportunity_count)} repair rows; every toggle remains in structured evidence.`
      : "",
    islandRows.length
      ? `### Predicted CPU-Island Assignment Portfolio\n\n${markdownTable(["Rank", "Island / range", "Members", "Incident boundaries", "Outcome", "Boundary edge delta", "Payload delta", "Best single / additional reduction"], islandRows.map((island) => [
          `#${island.portfolio_rank}`,
          `island ${island.island_index}: #${padOp(island.first_op_index)}-#${padOp(island.last_op_index)}`,
          `${formatNumber(island.op_count)} op(s): ${island.op_names.join(" / ")}`,
          `${formatNumber(island.baseline_incident_boundary_edge_count)} / ${island.baseline_incident_boundary_payload_bytes == null ? "N/A" : formatBytes(island.baseline_incident_boundary_payload_bytes)}`,
          `${island.outcome_class}${island.group_only_repair ? "; GROUP_ONLY_REPAIR" : ""}`,
          `${signed(island.signed_boundary_edge_count)} edge(s)`,
          signedBytes(island.signed_boundary_payload_bytes),
          `${island.best_single_op_index == null ? "none" : `#${padOp(island.best_single_op_index)}`} / ${signed(island.additional_edge_reduction_over_best_single)} edge(s), ${signedBytes(island.additional_payload_reduction_over_best_single)}`,
        ]))}`
      : "### Predicted CPU-Island Assignment Portfolio\n\nNo predicted CPU segment exists for this target profile.",
    fragilityRows.length
      ? `### Predicted Support-Loss Fragmentation Risks\n\n${markdownTable(["Rank", "Op", "Current rule", "Outcome", "Delegate segment delta", "Boundary edge delta", "Boundary payload delta", "Edge changes"], rankedRows(fragilityRows, true))}`
      : "### Predicted Support-Loss Fragmentation Risks\n\nNo delegated op increased static fragmentation under a single-op toggle.",
    repair.fragmentation_risk_count > fragilityRows.length
      ? `Displayed ${formatNumber(fragilityRows.length)} of ${formatNumber(repair.fragmentation_risk_count)} fragility rows; ${formatNumber(repair.fragmentation_risk_count - fragilityRows.length)} additional rows remain in structured evidence.`
      : "",
    unrankedRows.length
      ? `### Unranked Toggle Outcomes\n\n${markdownTable(["Op", "Outcome", "Segment delta", "Edge delta", "Payload delta"], unrankedRows.map((row) => [
          `#${padOp(row.op_index)} ${row.op_name}`,
          row.outcome_class,
          signed(row.signed_delegate_segment_count),
          signed(row.signed_boundary_edge_count),
          signedBytes(row.signed_boundary_payload_bytes),
        ]))}\n\nThese rows are not silently discarded: they satisfy neither the repair nor fragility ranking predicate. A support-loss row can reduce net boundary payload when the counterfactual reclassifies the surrounding execution domain; that is not a repair recommendation.`
      : "",
    topRepair ? `### Highest-Ranked Repair Edge Ledger: #${padOp(topRepair.op_index)} ${topRepair.op_name}\n\n${edgeLedger}` : "",
    topIsland ? `### CPU-Island Edge Ledger: Island ${topIsland.island_index} (#${padOp(topIsland.first_op_index)}-#${padOp(topIsland.last_op_index)})\n\n${islandEdgeLedger}` : "",
    `Ranking basis: ${repair.ranking_basis}`,
    `CPU-island ranking basis: ${repair.island_ranking_basis}`,
    `Method: ${repair.method}`,
    `> ${repair.interpretation_boundary}`,
  ].filter(Boolean).join("\n\n");
}
