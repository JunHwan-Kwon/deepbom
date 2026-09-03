import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { formatNumber, formatPercent, padOp } from "./format.js";
import { markdownTable } from "./report-utils.js";

export function kernelSourceCandidatesMarkdown(analysis) {
  if (String(analysis?.format || "").toLowerCase() === "onnx") return "Not applicable; ONNX Runtime execution-provider kernels are not modeled.";
  const assessment = String(analysis?.xnnpack_selector_assessment_status || "not_reported");
  const selectorSchema = analysis?.xnnpack_selector_evidence_schema || "not embedded";
  const selectorAccess = analysis?.xnnpack_selector_evidence_access || "not reported";
  if (assessment === "not_loaded") {
    return `Assessment: ${assessment}; schema ${selectorSchema}; access ${selectorAccess}. The controlled source-backed selector module was not loaded, so open static HEURISTIC profile hints contain no pinned-source candidate or source-backed tile claim.`;
  }
  if (assessment === "not_available_for_profile") {
    return `Assessment: ${assessment}; schema ${selectorSchema}; access ${selectorAccess}. No pinned-source selector rulepack was executed for this planning profile.`;
  }
  const groups = new Map();
  const decisionRows = [];
  for (const op of artifactIrOperators(analysis) || []) {
    const facts = op.selector_artifact_facts || {};
    const factsText = [
      `dtype ${facts.activation_dtype || "unavailable"}`,
      facts.kernel_area_status === "not_applicable" ? "kernel N/A" : `kernel ${facts.kernel_area || 0} (${facts.kernel_area_status || "not assessed"})`,
      `per-channel weights ${facts.per_channel_weights ? "yes" : "no"}`,
      `C ${facts.output_channels || 0} (${facts.output_channels_status || "not assessed"})`,
    ].join("; ");
    const unresolved = (op.unresolved_selector_dimensions || []).join(", ") || "none";
    for (const candidate of op.xnnpack_kernel_candidates || []) {
      const key = [candidate.family, candidate.source_ref, candidate.architecture_condition, candidate.compile_condition, candidate.runtime_condition].join("|");
      const group = groups.get(key) || { candidate, count: 0, indices: [] };
      group.count += 1;
      if (group.indices.length < 8) group.indices.push(`#${padOp(op.index)}`);
      groups.set(key, group);
      decisionRows.push([
        `#${padOp(op.index)} ${op.name}`,
        factsText,
        candidate.family,
        `x${candidate.alignment_multiple}`,
        candidate.tail_projection_status === "assessed"
          ? `C=${facts.output_channels} -> padded ${candidate.padded_output_channels}; inactive ${candidate.inactive_output_channels} (${formatPercent(candidate.inactive_lane_ratio)})`
          : candidate.tail_projection_status,
        unresolved,
      ]);
    }
    if (op.xnnpack_kernel_evidence_class === "SOURCE_ENUMERATED_NO_MATCH") decisionRows.push([
      `#${padOp(op.index)} ${op.name}`,
      factsText,
      `NO MATCH: ${op.no_match_reason_code || "reason not emitted"}`,
      "N/A",
      "not assessed; no enumerated candidate",
      unresolved,
    ]);
  }
  const noMatch = (artifactIrOperators(analysis) || []).filter((op) => op.xnnpack_kernel_evidence_class === "SOURCE_ENUMERATED_NO_MATCH");
  if (!groups.size && !noMatch.length) return `Assessment: ${assessment}; schema ${selectorSchema}; access ${selectorAccess}. No per-op kernel selector was source-enumerated for this planning profile; profile-level kernel hints remain HEURISTIC.`;
  const provenance = analysis?.xnnpack_selector_evidence_provenance || {};
  const worstOps = (provenance.worst_case_tail_op_indices || []).map((index) => `#${padOp(index)}`).join(", ") || "none";
  return [
    `Assessment: ${assessment}; schema ${selectorSchema}; access ${selectorAccess}.`,
    markdownTable(["Decision-ledger metric", "Value"], [
      ["Evidence schema / method", `${provenance.schema || "not embedded"} / ${provenance.method_version || "not embedded"}`],
      ["Target binding", `${provenance.target_profile_id || "not embedded"}; SHA-256 ${provenance.target_profile_sha256 || "not embedded"}`],
      ["Pinned XNNPACK source commit", provenance.xnnpack_source_commit || "not embedded"],
      ["Pinned GEMM / DWCONV source SHA-256", `${provenance.gemm_config_sha256 || "not embedded"} / ${provenance.dwconv_config_sha256 || "not embedded"}`],
      ["Candidate / assessed eligible ops", `${formatNumber(provenance.candidate_op_count || 0)} / ${formatNumber(provenance.assessed_op_count || 0)}`],
      ["Assessed eligible ops", formatNumber(provenance.assessed_op_count || 0)],
      ["Unique candidate ops", formatNumber(provenance.unique_candidate_op_count || 0)],
      ["Ambiguous candidate-set ops", formatNumber(provenance.ambiguous_candidate_op_count || 0)],
      ["Source-enumerated no-match ops", formatNumber(provenance.no_match_op_count || 0)],
      ["Candidate configurations", formatNumber(provenance.candidate_configuration_count || 0)],
      ["Tail-assessed ops", formatNumber(provenance.tail_assessed_op_count || 0)],
      ["Worst candidate tail", `${formatPercent(provenance.worst_case_tail_ratio || 0)} at ${worstOps}`],
      ["Unresolved selector dimensions", `${formatNumber(provenance.unresolved_selector_dimension_count || 0)} across ${formatNumber(provenance.unresolved_selector_op_count || 0)} op(s)`],
      ["Evidence boundary", provenance.evidence_boundary || "not embedded"],
    ]),
    groups.size ? markdownTable(["Candidate configuration", "Tile", "Ops", "Architecture selector", "Compile selector", "Runtime selector", "Pinned source", "Source file SHA-256"], [...groups.values()].map(({ candidate, count, indices }) => [
      candidate.family,
      Number(candidate.tile_nr || 0) > 0
        ? `${candidate.tile_mr || "?"}x${candidate.tile_nr}`
        : `${candidate.primary_tile || "?"}p${candidate.channel_tile || "?"}c`,
      `${count}; ${indices.join(", ")}${count > indices.length ? ", ..." : ""}`,
      candidate.architecture_condition,
      candidate.compile_condition,
      candidate.runtime_condition,
      candidate.source_ref,
      candidate.source_file_sha256,
    ])) : "No matching configuration was found in the enumerated pinned GEMM/DWCONV source paths.",
    markdownTable(["Op", "Artifact selector facts", "Candidate / reason", "Alignment", "Deterministic output-channel tail", "Unresolved selector dimensions"], decisionRows),
    noMatch.length ? `> SOURCE_ENUMERATED_NO_MATCH applies to ${noMatch.length} op(s); it means the inspected pinned GEMM/DWCONV configuration paths did not produce a candidate for that signature.` : null,
    "> SOURCE_ENUMERATED_CANDIDATE_SET is the complete set represented by this rulepack after applying artifact-visible facts and the selected planning profile. It is not an observed runtime microkernel. Unbound compile flags, host classification, lowering path, hardware detection, and dispatch conditions remain explicit in the table.",
  ].filter(Boolean).join("\n");
}
