import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { formatBytes, formatNumber, padOp } from "./format.js";
import { code, markdownTable } from "./report-utils.js";

export function dynamicShapeCostMarkdown(analysis) {
  const contract = analysis?.dynamic_shape_cost_contract;
  const supportedSchema = contract?.schema === "deepbom.dynamic_shape_cost_contract.v1"
    || contract?.schema === "deepbom.dynamic_shape_cost_contract.v2"
    || contract?.schema === "deepbom.dynamic_shape_cost_contract.v2.1"
    || contract?.schema === "deepbom.dynamic_shape_cost_contract.v2.2";
  if (!contract || !supportedSchema) {
    return markdownTable(["Field", "Value"], [["Status", "not emitted"]]);
  }
  const totalFormula = contract.total_macs_formula;
  const liveness = contract.liveness || {};
  const batchProjection = deriveTfliteBatchOneProjection(analysis);
  const projectionText = batchProjection.status === "assumption_bound_batch_one"
    ? `ASSUMPTION_BOUND: serialized N=1 => ${formatNumber(batchProjection.projected_total_macs)} MACs${batchProjection.projected_peak_live_payload_bytes != null ? `, ${formatBytes(batchProjection.projected_peak_live_payload_bytes)} symbolic live-payload peak` : ""}; 0 non-batch dynamic input axes; ${formatNumber(batchProjection.internal_symbol_count)} propagated internal batch symbol(s)`
    : `${batchProjection.status || "not assessed"}; ${batchProjection.reason || "no default numeric projection"}`;
  const lines = [
    "## Dynamic Shape Cost Contract (DERIVED)",
    "",
    markdownTable(["Field", "Value"], [
      ["Schema / status / format", `${code(contract.schema)} / ${contract.status || "not_assessed"} / ${contract.format || "unknown"}`],
      ["Evidence class", contract.evidence_class || "not emitted"],
      ["Pinned ONNX source", contract.source
        ? `${code(`${contract.source.repository}@${contract.source.commit}`)} / ${code(contract.source.path)} / SHA-256 ${code(contract.source.sha256)}`
        : "not emitted"],
      ["Dynamic tensors / symbols", `${formatNumber(contract.dynamic_tensor_count || 0)} / ${formatNumber(contract.symbol_count || 0)}`],
      ["Serialized default numeric projection", projectionText],
      ["Dimension bounds", contract.dimension_bounds_status || "not assessed"],
      ["Tensor formulas", `${formatNumber(contract.tensor_formula_count || 0)} emitted`],
      ["Dynamic compute formulas", `${formatNumber(contract.op_formula_count || 0)}/${formatNumber(contract.dynamic_compute_op_count || 0)} emitted; ${formatNumber(contract.unresolved_dynamic_compute_op_count || 0)} unresolved`],
      ["Total-MAC blockers", `${formatNumber(contract.total_macs_unresolved_op_count || 0)} op(s)`],
      ["Total MAC formula", totalFormula?.expression || contract.total_macs_formula_status || "not assessed"],
      ["Total MAC preconditions", totalFormula?.preconditions?.length
        ? totalFormula.preconditions.map((guard) => `${guard.label || guard.kind}: ${guard.expression || guard.numerator || guard.left || "guarded"}`).join("; ")
        : "none beyond non-negative dimension binding"],
      ["Liveness formula coverage", `${liveness.status || "not assessed"}; ${formatNumber(liveness.exact_candidate_program_point_count || 0)}/${formatNumber(liveness.candidate_program_point_count || 0)} program point(s); ${formatNumber(liveness.unresolved_candidate_program_point_count || 0)} unresolved; ${formatNumber(liveness.distinct_exact_formula_count || 0)} distinct formula(s)`],
      ["Symbolic peak", `${liveness.peak_selection_status || "not assessed"}; ${liveness.peak_live_payload_formula?.expression || liveness.peak_live_payload_max_formula?.expression || "numeric ordering requires bound dimensions"}`],
      ["Arena projection", contract.arena_projection_status || "not assessed"],
      ["Method", contract.method || "not emitted"],
      ["Boundary", contract.interpretation_boundary || "not emitted"],
    ]),
  ];
  if ((contract.symbols || []).length) {
    lines.push(
      "",
      "### Dynamic Dimension Symbols",
      "",
      markdownTable(["Symbol", "Source", "Declared name", "Occurrences", "Bounds"], (contract.symbols || []).map((symbol) => [
        code(symbol.symbol_id || "-"),
        symbol.source || "-",
        symbol.declared_name ? code(symbol.declared_name) : "anonymous",
        (symbol.occurrences || []).map((occurrence) => `T${occurrence.tensor_index}:${occurrence.tensor_name || "-"}[${occurrence.axis}]`).join(" / ") || "none",
        `${symbol.bounds_status || "not embedded"}; ${symbol.lower_bound ?? "-"}..${symbol.upper_bound ?? (symbol.upper_bound_expression || "-")}`,
      ])),
    );
  }
  if ((contract.tensor_formulas || []).length) {
    lines.push(
      "",
      "### Dynamic Tensor Payload Formulas",
      "",
      markdownTable(["Tensor", "Shape / signature", "Elements", "Payload bytes", "Declared projection", "Status"], (contract.tensor_formulas || []).map((row) => [
        `T${row.tensor_index} ${code(row.tensor_name || "-")} ${row.dtype || "UNKNOWN"}`,
        `${(row.shape || []).join("x") || "scalar"}${Array.isArray(row.shape_signature) && row.shape_signature.length ? ` / ${(row.shape_signature || []).join("x")}` : ""}`,
        row.element_count_formula?.expression || "not assessed",
        row.payload_bytes_formula?.expression || row.payload_bytes_expression || "not assessed",
        row.declared_shape_projection_bytes == null ? row.declared_shape_projection_status || "not available" : `${formatBytes(row.declared_shape_projection_bytes)}; ${row.declared_shape_projection_status || "example only"}`,
        `${row.formula_status || "not assessed"}; ${row.reason || ""}`,
      ])),
    );
  }
  if ((contract.op_formulas || []).length || (contract.unresolved_dynamic_compute_ops || []).length) {
    lines.push(
      "",
      "### Dynamic Compute MAC Formulas",
      "",
      markdownTable(["Op", "MAC expression", "Declared projection", "Status / basis"], [
        ...(contract.op_formulas || []).map((row) => [
          `#${padOp(row.op_index)} ${row.op_name}`,
          row.macs_formula?.expression || "not assessed",
          row.declared_shape_projection_macs == null ? row.declared_shape_projection_status || "not available" : `${formatNumber(row.declared_shape_projection_macs)}; ${row.declared_shape_projection_status || "example only"}`,
          `${row.formula_status || "not assessed"}; ${row.macs_formula?.preconditions?.length || 0} explicit guard(s); ${row.reason || ""}`,
        ]),
        ...(contract.unresolved_dynamic_compute_ops || []).map((row) => [
          `#${padOp(row.op_index)} ${row.op_name}`,
          "not assessed",
          "not promoted",
          row.reason || "formula unavailable",
        ]),
      ]),
    );
  }
  if ((contract.total_macs_unresolved_ops || []).length) {
    lines.push(
      "",
      "### Total MAC Formula Blockers",
      "",
      markdownTable(["Op", "Reason"], (contract.total_macs_unresolved_ops || []).map((row) => [
        `#${padOp(row.op_index)} ${row.op_name}`,
        row.reason || "exact total contribution unavailable",
      ])),
    );
  }
  if ((liveness.candidates || []).length) {
    const shown = liveness.candidates.slice(0, 24);
    lines.push(
      "",
      "### Symbolic Live-Payload Candidates",
      "",
      markdownTable(["Program point(s)", "Occurrences", "Live payload expression"], shown.map((row) => [
        row.first_program_point === row.last_program_point ? `${row.first_program_point}` : `${row.first_program_point}..${row.last_program_point}`,
        formatNumber(row.occurrence_count || 0),
        row.live_payload_formula?.expression || "not assessed",
      ])),
    );
    if ((liveness.candidates || []).length > shown.length) lines.push("", `Table truncated after ${shown.length} of ${liveness.candidates.length} distinct formulas; the complete ledger remains in engineering_evidence.json.`);
    lines.push("", `> ${liveness.interpretation_boundary || "A numeric peak requires runtime dimension binding unless symbolic dominance is proven."}`);
  }
  return lines.join("\n");
}
