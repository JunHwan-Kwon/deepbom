import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { decodeXnnpReason, decodeRoofReason } from "./reason-codes.js";
export { markdownWithModelSha256 } from "./report-utils.js";

function fmtNum(n) {
  if (n == null || isNaN(n)) return "0";
  const r = Math.round(n);
  return r.toLocaleString("en-US");
}

function fmtPct(n) {
  return (n * 100).toFixed(1) + "%";
}

function fmtPctClamped(n) {
  const value = Number.isFinite(Number(n)) ? Number(n) : 0;
  return fmtPct(Math.min(1, Math.max(0, value)));
}

function tableCell(value) {
  return String(value ?? "-")
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|");
}

function intensityPosture(bound) {
  return bound === "compute-bound" ? "high-intensity"
    : bound === "memory-bound" ? "low-intensity"
    : bound === "mixed" ? "mixed-intensity" : (bound || "-");
}

export function buildStaticAuditMarkdown(analysis, modelSha256 = "") {
  if (!analysis) return "";
  if (String(analysis.format || "").toLowerCase() === "onnx" && analysis.markdown) return analysis.markdown;
  const tp = analysis.target_profile || {};
  const lines = [];

  lines.push(`# TFLite Static Audit: ${analysis.filename || "unknown"}`);
  lines.push("");
  lines.push("## Summary");
  if (modelSha256) lines.push(`- Model SHA-256: \`${modelSha256}\``);
  lines.push(`- Target: **${tp.label || tp.id || "unknown"}**`);
  const summaryOps = artifactIrOperators(analysis) || [];
  const totalOpCount = analysis.operator_count ?? summaryOps.length;
  const delegatedOps = summaryOps.filter((op) => Number(op.xnnpack_chain_id) >= 0).length;
  const nonDelegatedOps = totalOpCount - delegatedOps;
  const quantStatus = analysis.quantization_status || {};
  const computeOpCount = Number(quantStatus.compute_ops || 0);
  const quantComputeOps = Number(quantStatus.quantized_compute_ops || 0);
  const totalComputeMacs = Number(quantStatus.compute_macs ?? analysis.total_macs ?? 0);
  const quantComputeMacs = Number(quantStatus.quantized_compute_macs
    ?? (Number.isFinite(Number(quantStatus.quantized_compute_mac_percent))
      ? totalComputeMacs * Number(quantStatus.quantized_compute_mac_percent)
      : 0));
  const quantStateOps = (analysis.quantization_status?.op_state_counts || [])
    .filter((item) => item.name && item.name !== "none")
    .reduce((sum, item) => sum + Number(item.count || 0), 0);
  lines.push(`- Ops: ${totalOpCount} | Tensors: ${analysis.tensor_count ?? 0}`);
  lines.push(`- Total MACs: ${fmtNum(analysis.total_macs)}`);
  lines.push(`- XNNPACK conditional eligibility (PREDICTED): ${delegatedOps} conditionally delegatable / ${nonDelegatedOps} predicted fallback of ${totalOpCount} ops; predicted non-structural partition breaks: ${analysis.xnnpack_effective_chain_breaks ?? 0}`);
  lines.push(`- Quantized MAC-bearing compute ops: ${quantComputeOps}/${computeOpCount} (${computeOpCount > 0 ? fmtPctClamped(quantComputeOps / computeOpCount) : "N/A"})`);
  lines.push(`- Quantized compute MACs: ${fmtNum(quantComputeMacs)} / ${fmtNum(totalComputeMacs)} (${totalComputeMacs > 0 ? fmtPctClamped(quantComputeMacs / totalComputeMacs) : "N/A"})`);
  lines.push(`- Total graph ops carrying a quantized or boundary state: ${quantStateOps}/${totalOpCount} (${totalOpCount > 0 ? fmtPctClamped(quantStateOps / totalOpCount) : "N/A"})`);
  // Auto consistency check: header count = delegated + non-delegated = stage op counts = graph op list.
  const stageOpSum = (analysis.stages || []).reduce((s, st) => s + (st.op_count || 0), 0);
  const consistent = delegatedOps + nonDelegatedOps === totalOpCount
    && summaryOps.length === totalOpCount
    && ((analysis.stages || []).length === 0 || stageOpSum === totalOpCount);
  const quantConsistent = quantComputeOps <= computeOpCount
    && quantComputeMacs <= totalComputeMacs + 1e-6
    && quantComputeOps === Number(analysis.quantization_status?.quantized_compute_ops ?? quantComputeOps)
    && computeOpCount === Number(analysis.quantization_status?.compute_ops ?? computeOpCount);
  lines.push(consistent
    ? `- Consistency check: ok (${totalOpCount} = ${delegatedOps} conditionally delegatable + ${nonDelegatedOps} predicted fallback${(analysis.stages || []).length ? `; stage sum = ${stageOpSum}` : ""})`
    : `- **CONSISTENCY WARNING**: op totals disagree (header ${totalOpCount}, ops list ${summaryOps.length}, conditionally-delegatable+fallback ${delegatedOps + nonDelegatedOps}, stage sum ${stageOpSum}); report a bug.`);
  lines.push(quantConsistent
    ? `- Quantization consistency check: ok (${quantComputeOps} <= ${computeOpCount} compute ops; ${fmtNum(quantComputeMacs)} <= ${fmtNum(totalComputeMacs)} compute MACs)`
    : "- **QUANTIZATION CONSISTENCY WARNING**: quantized compute counts or MAC totals disagree with the compute-op denominator; report a bug.");
  if (analysis.quant_hole_count > 0) {
    lines.push(`- Activation precision boundary operators: ${analysis.quant_hole_count} (maximum graph-neighbor MAC share: ${fmtPct(analysis.quant_hole_mac_impact)})`);
  }
  lines.push("");

  lines.push("## XNNPACK Delegation (PREDICTED)");
  lines.push(`- Assumption: ${decodeXnnpReason(analysis.xnnpack_assumption ?? "")}`);
  lines.push(`- Predicted partition boundaries: ${analysis.xnnpack_chain_breaks ?? 0} (runtime delegate logs are required for confirmation)`);
  lines.push(`- Boundary anatomy: total = non-structural ${analysis.xnnpack_effective_chain_breaks ?? 0} + structural/view ${analysis.xnnpack_structural_chain_breaks ?? 0}; zero-MAC non-structural ${analysis.xnnpack_zero_mac_chain_breaks ?? 0} is a subset of non-structural. Zero modeled MAC does not imply a shape operation: pooling/reduction boundaries remain non-structural. Whether any boundary materializes a tensor copy is not assessable from the artifact alone.`);
  const boundaryInventory = analysis.predicted_partition_boundaries;
  if (boundaryInventory?.schema === "deepbom.predicted_partition_boundary_edges.v1.1") {
    const payload = boundaryInventory.summed_edge_payload_bytes == null
      ? `${fmtNum(boundaryInventory.assessed_edge_payload_bytes)} assessed B; ${boundaryInventory.unassessed_payload_edge_count} edge payload(s) not assessed`
      : `${fmtNum(boundaryInventory.summed_edge_payload_bytes)} B`;
    lines.push(`- Internal execution-domain edges: ${boundaryInventory.edge_count} edge(s), ${boundaryInventory.unique_tensor_count} unique tensor(s), summed logical edge payload ${payload} (${boundaryInventory.payload_coverage_status} coverage; PREDICTED assignment / DERIVED payload).`);
    if (boundaryInventory.edges?.length) {
      lines.push("");
      lines.push("| Tensor edge | Predicted transition | Logical payload | Materialization |");
      lines.push("|---|---|---:|---|");
      for (const edge of boundaryInventory.edges.slice(0, 24)) {
        lines.push(`| ${tableCell(`T${edge.tensor_index} ${edge.tensor_name || "-"}; #${edge.producer_op_index} ${edge.producer_op_name} -> #${edge.consumer_op_index} ${edge.consumer_op_name}`)} | ${tableCell(`${edge.producer_domain} -> ${edge.consumer_domain}`)} | ${edge.payload_bytes == null ? "not assessed" : `${fmtNum(edge.payload_bytes)} B`} | not assessable from static artifact |`);
      }
    }
  }
  const boundaryOps = summaryOps.filter((op) => op.xnnpack_chain_break);
  const structuralBoundaryNames = new Set(["RESHAPE", "SQUEEZE", "EXPAND_DIMS", "SHAPE"]);
  const poolingReductionBoundaryNames = new Set(["AVERAGE_POOL_2D", "MAX_POOL_2D", "L2_POOL_2D", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "SUM"]);
  const structuralBoundaryCount = boundaryOps.filter((op) => structuralBoundaryNames.has(op.name)).length;
  const poolingReductionBoundaryCount = boundaryOps.filter((op) => poolingReductionBoundaryNames.has(op.name)).length;
  const otherNonStructuralBoundaryCount = boundaryOps.length - structuralBoundaryCount - poolingReductionBoundaryCount;
  lines.push(`- Boundary operator categories: pooling/reduction ${poolingReductionBoundaryCount} · structural/view ${structuralBoundaryCount} · other non-structural ${otherNonStructuralBoundaryCount}.`);
  const fallback = (artifactIrOperators(analysis) || []).filter(op => !op.xnnpack_supported);
  if (fallback.length > 0) {
    lines.push("- Predicted fallback ops:");
    for (const op of fallback.slice(0, 20)) {
      lines.push(`  - ${op.name} (idx ${op.index}): ${decodeXnnpReason(op.xnnpack_reason || "")}`);
    }
    if (fallback.length > 20) lines.push(`  - ... and ${fallback.length - 20} more`);
  }
  lines.push("");

  lines.push("## Static Arithmetic-Intensity Posture");
  lines.push("Posture classes (low/mixed/high-intensity) come from heuristic intensity bands configured per target profile and dtype in the rulepack — the mem/compute thresholds shown in each Reason. They are NOT positions relative to the theoretical peak-compute/bandwidth ridge; in classical roofline terms every op below the theoretical ridge is memory-side. Treat the posture as static triage, not a measured roofline placement.");
  lines.push("");
  lines.push("Arithmetic-intensity convention: 1 MAC = 2 arithmetic operations. Estimated logical bytes = input activation reads + output activation writes + constant tensor reads; cache reuse, prefetch, packing, write allocation, runtime scratch, alignment padding, delegate copies, and in-place execution are not included. Row WS is the estimated logical working set for one output-row computation under the analyzer's static convolution model; L1 ratio is not a measured cache hit rate or fit probability.");
  lines.push("");
  lines.push("| Op | MACs | Bytes | Intensity | Posture (heuristic band) | Reason |");
  lines.push("|---|---|---|---|---|---|");
  const roofOps = (artifactIrOperators(analysis) || [])
    .filter(op => (op.estimated_ops ?? 0) > 0 || (op.estimated_bytes ?? 0) > 0)
    .slice(0, 30);
  for (const op of roofOps) {
    const intensity = (op.intensity_ops_per_byte ?? 0) > 0
      ? Number(op.intensity_ops_per_byte).toFixed(2)
      : "-";
    lines.push(`| ${[
      op.name,
      fmtNum(op.macs),
      fmtNum(op.estimated_bytes),
      intensity,
      intensityPosture(op.static_bound_guess),
      decodeRoofReason(op.roofline_reason || ""),
    ].map(tableCell).join(" | ")} |`);
  }
  lines.push("");

  lines.push("## Stage Breakdown");
  lines.push("Stage partitioning rule: stages are contiguous op-order runs with the same primary-output shape key. 4D tensors use H×W×channel-bucket keys (channel buckets 32, 64, 128, 256, 512, then 512-multiples); 3D tensors use sequence length; 2D/1D/scalar tensors use fixed structural keys. A new stage begins when this key changes.");
  for (const [stage, data] of Object.entries(analysis.stages || {})) {
    lines.push(`### ${stage}`);
    lines.push(`- MACs: ${fmtNum(data.macs)} | Ops: ${data.op_count ?? 0}`);
  }
  lines.push("");

  lines.push("## Top MAC Consumers");
  const topMac = (artifactIrOperators(analysis) || [])
    .filter(op => (op.macs ?? 0) > 0)
    .sort((a, b) => (b.macs ?? 0) - (a.macs ?? 0))
    .slice(0, 10);
  for (const op of topMac) {
    lines.push(`- ${op.name} [${op.index}]: ${fmtNum(op.macs)} MACs (${fmtPct(op.mac_percent ?? 0)})`);
  }
  lines.push("");

  if (analysis.quant_hole_count > 0) {
    lines.push("## Activation Precision Boundaries");
    const holes = (artifactIrOperators(analysis) || []).filter(op => op.quant_hole);
    for (const op of holes.slice(0, 20)) {
      lines.push(`- ${op.name} [${op.index}]: ${op.quantization_detail || ""}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}
