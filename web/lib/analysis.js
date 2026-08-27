import {
  countByArray,
  formatBytes,
  formatNumber,
  formatPercent,
  formatPercentRange,
  formatShapes,
  formatScientific,
  humanizeStageKey,
  sumNumbers,
  tensorShapeText,
} from "./format.js";
import { decodeXnnpReason, decodeRoofReason } from "./reason-codes.js";

export function contractHasDynamicDims(tensors = []) {
  return tensors.some((tensor) =>
    Array.isArray(tensor?.shape_signature) && tensor.shape_signature.some((dim) => Number(dim) < 0),
  );
}

export function contractDynamicDimSummary(tensors = []) {
  const dynamic = tensors
    .filter((tensor) => Array.isArray(tensor?.shape_signature) && tensor.shape_signature.some((dim) => Number(dim) < 0))
    .map((tensor) => `${tensor.name || "tensor"}:${tensorShapeText(tensor)}`);
  return dynamic.join(" / ");
}

export function predictedPartitionBoundaryInventory(analysis) {
  const inventory = analysis?.predicted_partition_boundaries;
  return inventory?.schema === "deepbom.predicted_partition_boundary_edges.v1.1" ? inventory : null;
}

export function predictedPartitionBoundaryEdgesForOp(analysis, opIndex) {
  const inventory = predictedPartitionBoundaryInventory(analysis);
  if (!inventory) return [];
  return (inventory.edges || []).filter((edge) => edge.producer_op_index === opIndex || edge.consumer_op_index === opIndex);
}

export function predictedPartitionBoundaryPayloadForOp(analysis, opIndex) {
  const edges = predictedPartitionBoundaryEdgesForOp(analysis, opIndex);
  const assessed = edges.filter((edge) => Number.isFinite(edge.payload_bytes));
  return {
    edge_count: edges.length,
    assessed_edge_count: assessed.length,
    unassessed_edge_count: edges.length - assessed.length,
    assessed_bytes: assessed.reduce((sum, edge) => sum + Number(edge.payload_bytes), 0),
    status: edges.every((edge) => Number.isFinite(edge.payload_bytes)) ? "complete" : "partial",
    edges,
  };
}

// modelQuantizationStatus: WASM computes this; read directly from analysis.quantization_status.
// Fallback kept for backward compatibility with older analysis objects.
export function modelQuantizationStatus(analysis) {
  if (analysis?.quantization_status?.label) return analysis.quantization_status;
  const format = String(analysis?.format || "tflite").toLowerCase();
  if (!["tflite", "onnx"].includes(format)) {
    return {
      classification: "not_assessed_format_contract_unbound",
      label: "Quantization not assessed",
      summary: `No normalized ${format || "serialized-artifact"} quantization contract was emitted.`,
      detail: "Absence of a normalized quantization contract is not evidence that stored tensors or executed kernels are floating-point.",
      quantized_tensor_percent: null,
      quantized_compute_mac_percent: null,
      compute_macs: null,
      quantized_compute_macs: null,
      op_state_counts: [],
      full_integer: null,
    };
  }
  // Fallback for stale analysis objects without WASM-computed status
  const quantizedTensors = Number(analysis?.quantized_tensors || 0);
  const tensorCount = Number(analysis?.tensor_count || 0);
  const ratio = tensorCount ? quantizedTensors / tensorCount : 0;
  return {
    classification: quantizedTensors ? "quantization_signals" : "not_quantized_float",
    label: quantizedTensors ? "Quantization signals detected" : "Not quantized",
    summary: quantizedTensors ? "Quantized tensor or op signals present." : "No quantized tensor or op signal detected.",
    detail: quantizedTensors ? `${formatNumber(quantizedTensors)}/${formatNumber(tensorCount)} tensors quantized (${formatPercent(ratio)}).` : "",
    quantized_tensor_percent: ratio,
    quantized_compute_mac_percent: 0,
    compute_macs: 0,
    quantized_compute_macs: 0,
    op_state_counts: [{ name: "none", count: (analysis?.ops || []).length }],
    full_integer: false,
  };
}

// Bottleneck estimates are computed in WASM (op.bottleneck_*).
// This function is now a read-only adapter: it reads WASM-computed fields from the op.
export function estimateOpBottleneck(op, target = {}) {
  const explicitlyUnassessed = op?.bottleneck_assessment_status === "not_assessed"
    || op?.macs_status === "not_assessed"
    || op?.estimated_bytes_status === "not_assessed"
    || op?.intensity_status === "not_assessed";
  const complete = op?.bottleneck_total_us != null && Number.isFinite(Number(op.bottleneck_total_us));
  if (explicitlyUnassessed || !complete) {
    return {
      assessed: false,
      reason: op?.bottleneck_reason || op?.macs_reason || op?.estimated_bytes_reason || "required shape, dtype, or target cost inputs are unavailable",
      computeUs: null,
      memoryUs: null,
      packingUs: null,
      breakUs: null,
      fallbackUs: null,
      totalUs: null,
      coldStartUs: null,
      steadyStateUs: null,
      dominantTone: "not-assessed",
      dominantLabel: "not assessed",
    };
  }
  const computeUs  = Number(op.bottleneck_compute_us  ?? 0);
  const memoryUs   = Number(op.bottleneck_memory_us   ?? 0);
  const packingUs  = Number(op.bottleneck_packing_us  ?? 0);
  const breakUs    = Number(op.bottleneck_break_us    ?? 0);
  const fallbackUs = Number(op.bottleneck_fallback_us ?? 0);
  const coldStartUs = Number(op.bottleneck_total_us ?? 0);
  const steadyStateUs = Math.max(0, coldStartUs - packingUs - breakUs);
  const dom        = op.bottleneck_dominant || "memory";
  const toneMap = { compute: "compute-bound", memory: "memory-bound", packing: "packing", break: "break", fallback: "fallback" };
  const labelMap = { compute: "compute", memory: "memory traffic", packing: "weight packing", break: "partition planning setup", fallback: "fallback traffic" };
  return {
    assessed: true,
    reason: null,
    computeUs,
    memoryUs,
    packingUs,
    breakUs,
    fallbackUs,
    totalUs: coldStartUs,
    coldStartUs,
    steadyStateUs,
    dominantTone: toneMap[dom] || dom,
    dominantLabel: labelMap[dom] || dom,
  };
}

export function normalizeUnassessedCostValues(analysis) {
  if (!analysis || typeof analysis !== "object") return analysis;
  const fields = [
    "bottleneck_compute_us",
    "bottleneck_memory_us",
    "bottleneck_packing_us",
    "bottleneck_break_us",
    "bottleneck_fallback_us",
    "bottleneck_total_us",
  ];
  for (const op of analysis.ops || []) {
    const explicitlyUnassessed = op?.macs_status === "not_assessed"
      || op?.estimated_bytes_status === "not_assessed"
      || op?.intensity_status === "not_assessed";
    const complete = op?.bottleneck_total_us != null && Number.isFinite(Number(op.bottleneck_total_us));
    if (explicitlyUnassessed || !complete) {
      op.bottleneck_assessment_status = "not_assessed";
      op.bottleneck_reason = op.macs_reason || op.estimated_bytes_reason || "required shape, dtype, or target cost inputs are unavailable";
      for (const key of fields) op[key] = null;
    } else {
      op.bottleneck_assessment_status = "assessed";
      op.bottleneck_reason = null;
    }
  }
  return analysis;
}

export function opSteadyStateUs(op) {
  return estimateOpBottleneck(op).steadyStateUs;
}

export function opColdStartUs(op) {
  return estimateOpBottleneck(op).coldStartUs;
}

export function opLogicalRowPayloadBytes(op) {
  const value = Number(
    op?.cache_payload?.logical_row_payload_bytes
      ?? op?.row_working_set_bytes,
  );
  return Number.isFinite(value) && value >= 0 ? value : null;
}

export function opLogicalL1Ratio(op, l1DataBytes) {
  const payload = opLogicalRowPayloadBytes(op);
  const capacity = Number(l1DataBytes);
  return payload != null && Number.isFinite(capacity) && capacity > 0
    ? payload / capacity
    : null;
}

export function bottleneckComponentTotals(estimates) {
  return estimates.reduce((totals, item) => {
    if (item.computeUs >= item.memoryUs) totals.computeUs += item.computeUs;
    else totals.memoryUs += item.memoryUs;
    totals.packingUs += item.packingUs;
    totals.breakUs += item.breakUs;
    totals.fallbackUs += item.fallbackUs;
    return totals;
  }, { computeUs: 0, memoryUs: 0, packingUs: 0, breakUs: 0, fallbackUs: 0 });
}

export function macDistributionData(analysis, { limit = 12 } = {}) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const totalMacs = Math.max(0, Number(analysis?.total_macs || sumNumbers(ops.map((op) => op.macs))));
  const top = [...ops]
    .filter((op) => Number(op.macs || 0) > 0)
    .sort((a, b) => Number(b.macs || 0) - Number(a.macs || 0))
    .slice(0, limit);
  const topMacs = sumNumbers(top.map((op) => op.macs));
  return {
    ops,
    totalMacs,
    top,
    topMacs,
    otherMacs: Math.max(0, totalMacs - topMacs),
  };
}

export function bottleneckDistributionData(analysis, target = {}, { limit = 12, operatingPoint = "steady" } = {}) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const rows = ops
    .map((op) => {
      const estimate = estimateOpBottleneck(op, target);
      const selectedUs = operatingPoint === "cold" ? estimate.coldStartUs : estimate.steadyStateUs;
      return { op, estimate: { ...estimate, totalUs: selectedUs }, operatingPoint };
    })
    .filter((item) => item.estimate.totalUs > 0)
    .sort((a, b) => b.estimate.totalUs - a.estimate.totalUs
      || Number(a.op.index) - Number(b.op.index));
  const top = rows.slice(0, limit);
  const totalUs = sumNumbers(rows.map((item) => item.estimate.totalUs));
  const topUs = sumNumbers(top.map((item) => item.estimate.totalUs));
  return {
    rows,
    top,
    totalUs,
    topUs,
    otherUs: Math.max(0, totalUs - topUs),
  };
}

// summarizeTargetComparison: reads WASM-computed bottleneck fields directly from ops.
export function summarizeTargetComparison(analysis) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const target = analysis?.target_profile || {};
  const estimates = ops.map(op => estimateOpBottleneck(op));
  const totals = bottleneckComponentTotals(estimates);
  const totalUs = sumNumbers(estimates.map((item) => item.steadyStateUs));
  const coldStartUs = sumNumbers(estimates.map((item) => item.coldStartUs));
  const memoryRatio = ops.length ? ops.filter(op => op.static_bound_guess === "memory-bound").length / ops.length : 0;
  const topOp = ops.reduce((best, op) => (
    !best || opSteadyStateUs(op) > opSteadyStateUs(best)
  ) ? op : best, null);
  return {
    id: target.id || "",
    label: target.label || analysis?.target_id || "Target",
    ridge: Number(target.ridge_point_ops_per_byte || 0),
    bandwidth: Number(target.effective_memory_bandwidth_gbps || 0),
    peak: Number(target.effective_peak_gops || 0),
    memoryRatio,
    chainBreaks: Number(analysis?.xnnpack_effective_chain_breaks ?? analysis?.xnnpack_chain_breaks ?? 0),
    speedup: Number(analysis?.estimated_int8_speedup || 1),
    isFp32: ["not_quantized_float", "float16_weight_storage"].includes(analysis?.quantization_status?.classification),
    topOp: topOp || null,
    topUs: topOp ? opSteadyStateUs(topOp) : 0,
    totals,
    totalUs,
    coldStartUs,
  };
}

export function summarizeFrontierTargetComparison(target) {
  const components = target?.components || {};
  const packingUs = Number(components.packing_us || 0);
  const boundarySetupUs = Number(components.boundary_us || 0);
  const coldStartUs = Number(target?.cold_start_us ?? target?.total_us ?? 0);
  const steadyStateUs = Number(
    target?.steady_state_us ?? Math.max(0, coldStartUs - packingUs - boundarySetupUs),
  );
  return {
    id: target?.target_id || "",
    label: target?.target_label || target?.target_id || "Target",
    ridge: Number(target?.ridge_point_ops_per_byte || 0),
    memoryRatio: Number(target?.low_intensity_op_ratio || 0),
    chainBreaks: Number(target?.predicted_effective_chain_breaks || 0),
    speedup: Number(target?.estimated_int8_speedup || 1),
    isFp32: Boolean(target?.float_artifact),
    topOp: target?.top_op_index == null
      ? null
      : { index: target.top_op_index, name: target.top_op_name || "" },
    topUs: Number(target?.top_op_steady_state_us ?? target?.top_op_us ?? 0),
    totals: {
      computeUs: Number(components.compute_us || 0),
      memoryUs: Number(components.memory_us || 0),
      packingUs,
      breakUs: boundarySetupUs,
      fallbackUs: Number(components.fallback_us || 0),
    },
    totalUs: steadyStateUs,
    coldStartUs,
    steadyStateLowUs: Number(target?.steady_state_low_us ?? steadyStateUs),
    steadyStateHighUs: Number(target?.steady_state_high_us ?? steadyStateUs),
    coldStartLowUs: Number(target?.cold_start_low_us ?? coldStartUs),
    coldStartHighUs: Number(target?.cold_start_high_us ?? coldStartUs),
  };
}

export function quantizationScopeExplanation(analysis) {
  const quant = modelQuantizationStatus(analysis);
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const format = String(analysis?.format || "tflite").toLowerCase();
  const opStateCounts = quant.op_state_counts || countByArray(ops.map((op) => op.quantization_state || "none"));
  const movementStates = opStateCounts
    .filter((item) => ["movement", "boundary", "requantization"].some((token) => String(item.name || "").includes(token)))
    .reduce((sum, item) => sum + Number(item.count || 0), 0);
  const quantizedStateOps = opStateCounts
    .filter((item) => item.name !== "none")
    .reduce((sum, item) => sum + Number(item.count || 0), 0);
  const tflite = format === "tflite";
  const fallbackOps = tflite ? ops.filter((op) => Number(op.xnnpack_chain_id) < 0).length : null;
  const zeroMacFallbackOps = tflite ? ops.filter((op) => Number(op.xnnpack_chain_id) < 0 && !Number(op.macs || 0)).length : null;
  return {
    compute_ops_denominator: Number(quant.compute_ops || 0),
    quantized_compute_ops: Number(quant.quantized_compute_ops || 0),
    quantized_compute_mac_percent: quant.quantized_compute_mac_percent == null
      ? null
      : Number(quant.quantized_compute_mac_percent),
    compute_macs: quant.compute_macs == null ? null : Number(quant.compute_macs),
    quantized_compute_macs: quant.quantized_compute_macs == null ? null : Number(quant.quantized_compute_macs),
    all_ops_denominator: ops.length,
    quantized_or_boundary_state_ops: quantizedStateOps,
    quantized_data_movement_or_boundary_ops: movementStates,
    op_state_counts: opStateCounts,
    xnnpack_fallback_or_break_ops: fallbackOps,
    zero_mac_fallback_or_break_ops: zeroMacFallbackOps,
    explanation: tflite
      ? "compute_ops counts MAC-bearing compute operators only; op_state_counts covers every graph operator, including zero-MAC data movement and quantization boundary ops. XNNPACK fallback/break counts are a delegate partition concept and can overlap with quantized_data_movement."
      : format === "onnx"
        ? "compute_ops counts MAC-bearing compute operators only; op_state_counts covers every graph operator, including Q/DQ boundaries. Execution-provider assignment is a separate source-backed or observed runtime contract."
        : "This serialized format has no TFLite XNNPACK scope. Quantization state is reported only when a normalized format-specific storage or numerical contract is emitted; absence is NOT_ASSESSED, not zero or floating-point proof.",
  };
}

export function quantizationScopeSummary(analysis) {
  const scope = quantizationScopeExplanation(analysis);
  if (!scope.compute_ops_denominator && !scope.all_ops_denominator) return "No op-level quantization scope available.";
  return `${formatNumber(scope.quantized_compute_ops)}/${formatNumber(scope.compute_ops_denominator)} MAC-bearing compute ops quantized; op-state inventory covers all ${formatNumber(scope.all_ops_denominator)} graph operators (QUANTIZE/DEQUANTIZE boundary counts are listed in Detail).`;
}

export function quantizationStatusTone(status) {
  const id = status?.classification || "";
  if (id === "full_integer" || id === "integer_internal_float_io") return "good";
  if (id === "mixed_quantization" || id === "dynamic_range_or_weight_only" || id === "qdq_signals_only" || id === "quantization_signals_partial_mac_assessment") return "warn";
  if (id === "not_quantized_float") return "neutral";
  return "neutral";
}

export function xnnpackLabel(op, analysis = null) {
  if (String(analysis?.format || "tflite").toLowerCase() === "onnx") return "not modeled";
  if (op.xnnpack_chain_break) return `break / ${op.xnnpack_break_class || "unclassified"}`;
  if (op.xnnpack_supported) return `chain ${op.xnnpack_chain_id}`;
  return "fallback";
}

export function quantLabel(op) {
  const risk = op.quant_risk || "none";
  const ratio = Number(op.quant_scale_ratio || 0);
  const cv = Number(op.quant_scale_cv || 0);
  const zp = Number(op.quant_zero_point_offset || 0);
  const mode = op.quant_scale_mode || "none";
  const ratioText = op.quant_scale_ratio_meaningful ? `r ${formatScientific(ratio)} / cv ${cv.toFixed(1)}` : `${mode} / scale N/A`;
  const zpStatus = op.quant_zero_point_status || "none";
  const zpText = zpStatus === "reinterpret" ? "zp=128 reinterpret" : zp ? `zp offset ${zp} ${zpStatus}` : `zp ${zpStatus}`;
  if (risk === "none" && mode === "none") return risk;
  return `${risk} / ${ratioText} / ${zpText}`;
}

export function quantStateLabel(op) {
  const state = op.quantization_state || (op.quantized_compute_path ? "quantized_compute" : op.quantized_path ? "quant_signal_only" : "none");
  if (state === "none") return "none";
  const compact = {
    quantized_compute: "INT8 compute",
    weight_only_or_dynamic_range: "weight-only",
    weight_metadata_only: "weight metadata",
    mixed_or_hybrid_compute: "mixed/hybrid",
    quant_boundary: "Q/DQ boundary",
    integer_requantization: "integer requantization",
    float16_constant_expansion: "FP16 storage -> FP32",
    quantized_constant_expansion: "8-bit storage -> float",
    constant_precision_conversion: "constant precision bridge",
    precision_boundary: "precision boundary",
    quantized_data_movement: "8-bit movement",
    quant_signal_only: "quant signal",
    serialized_quantization_transform: "serialized quant/compression transform",
  }[state] || state;
  const risk = op.quant_risk && op.quant_risk !== "none" ? ` / ${op.quant_risk}` : "";
  return `${compact}${risk}`;
}

export function opPrecisionLabel(op, analysis = null) {
  const candidate = String(op?.xnnpack_kernel_candidate || "").toUpperCase();
  if (candidate.includes("QU8")) return "UINT8";
  if (candidate.includes("QS8") || candidate.includes("QC8W")) return "INT8";
  if (candidate.includes("FP16")) return "FP16";
  if (candidate.includes("FP32")) return "FP32";

  const inputIndex = Number(op?.inputs?.[0]);
  const directTensor = Number.isInteger(inputIndex) ? analysis?.tensors?.[inputIndex] : null;
  const inputTensor = directTensor?.index === inputIndex
    ? directTensor
    : (analysis?.tensors || []).find((tensor) => Number(tensor?.index) === inputIndex);
  const dtype = String(inputTensor?.dtype || "").toUpperCase();
  if (["UINT8", "INT8", "FLOAT16", "FLOAT32", "BFLOAT16"].includes(dtype)) {
    return dtype === "FLOAT16" ? "FP16" : dtype === "FLOAT32" ? "FP32" : dtype;
  }
  if (op?.quantized_compute_path) return "8-bit";
  return "non-quantized";
}

export function quantToneClass(op) {
  if (op.quant_risk === "risk") return "risk-text";
  if (op.quant_risk === "warn") return "warn-text";
  const state = op.quantization_state || "";
  if (state === "quantized_compute") return "good-text";
  if (state && state !== "none") return "warn-text";
  return "muted-text";
}

export function alignmentLabel(op) {
  const status = op.channel_alignment_status || "none";
  const multiples = (op.xnnpack_kernel_alignment_multiples || []).length
    ? op.xnnpack_kernel_alignment_multiples.map((value) => `x${value}`).join("/")
    : `x${op.channel_alignment_multiple}`;
  if (status === "misaligned") {
    const min = Number(op.channel_tail_overhead_percent_min ?? op.channel_tail_overhead_percent ?? 0);
    const max = Number(op.channel_tail_overhead_percent_max ?? op.channel_tail_overhead_percent ?? 0);
    return `C${op.output_channels} vs ${multiples} / ${formatPercentRange(min, max)} tail`;
  }
  if (status === "aligned") {
    return `aligned ${multiples}`;
  }
  return "-";
}

export function packingLabel(op) {
  const risk = op.weight_packing_risk || "none";
  const us = Number(op.weight_packing_overhead_us || 0);
  if (!us || risk !== "warn") return "-";
  return `${risk} / ${us.toFixed(1)} us`;
}

export function fusionLabel(op) {
  const status = op.fusion_status || "-";
  return op.fused_activation && op.fused_activation !== "NONE" ? `${status} (${op.fused_activation})` : status;
}

export function boundRank(value) {
  if (value === "memory-bound") return 0;
  if (value === "mixed") return 1;
  return 2;
}

export function boundTone(value) {
  if (value === "compute-bound") return "compute-bound";
  if (value === "mixed") return "mixed";
  if (value === "memory-bound") return "memory-bound";
  return "neutral";
}

export function quantTileTone(op) {
  if (op.quant_risk === "risk") return "risk";
  if (op.quant_risk === "warn") return "warn";
  if (op.quantization_state === "quantized_compute") return "good";
  if (op.quantization_state && op.quantization_state !== "none") return "mixed";
  return "neutral";
}

export function stageSummaryText(stage) {
  if (String(stage?.summary_text || "").trim()) return String(stage.summary_text);
  const macCoverage = Number(stage.mac_not_assessed_ops || 0) > 0
    ? ` / MAC coverage ${stage.mac_assessed_ops || 0}/${(stage.mac_assessed_ops || 0) + stage.mac_not_assessed_ops}`
    : "";
  const channels = Array.isArray(stage.channels) ? stage.channels : [];
  return `ops ${stage.first_op}-${stage.last_op} / count ${stage.op_count} / C ${channels.join("/") || "-"} / MACs ${stage.macs == null ? "N/A" : formatNumber(stage.macs)} (${stage.mac_percent == null ? "N/A" : formatPercent(stage.mac_percent)})${macCoverage} / MAC-weighted conditionally delegatable ${formatPercent(stage.delegated_mac_percent || 0)} predicted fallback ${formatPercent(stage.fallback_mac_percent || 0)} / op-count conditionally delegatable ${formatPercent(stage.delegated_op_percent || 0)} predicted fallback ${formatPercent(stage.fallback_op_percent || 0)} / predicted partition breaks ${stage.xnnpack_chain_breaks || 0} / ${stage.patterns?.join(", ") || "no pattern"}`;
}

export function topMacOps(analysis, limit = 24) {
  return [...(analysis?.ops || [])]
    .filter((op) => op.macs_status !== "not_assessed")
    .sort((a, b) => Number(b.macs || 0) - Number(a.macs || 0))
    .slice(0, limit);
}

export function assessedOpLogicalBytes(op) {
  if (op?.estimated_bytes_status === "not_assessed" || op?.estimated_bytes == null) return null;
  const value = Number(op.estimated_bytes);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

export function rooflineRows(analysis, limit = 64) {
  return [...(analysis?.ops || [])]
    .filter((op) => op.intensity_ops_per_byte != null && Number(op.macs || 0) > 0)
    .sort((a, b) => {
      // MAC-bearing ops first within each bound class — zero-MAC shape ops (RESHAPE, PAD)
      // are automatically memory-bound but have no real compute bottleneck to address.
      const hasMacsA = Number(a.macs || 0) > 0 ? 1 : 0;
      const hasMacsB = Number(b.macs || 0) > 0 ? 1 : 0;
      if (hasMacsA !== hasMacsB) return hasMacsB - hasMacsA;
      const rankA = boundRank(a.static_bound_guess);
      const rankB = boundRank(b.static_bound_guess);
      if (rankA !== rankB) return rankA - rankB;
      const bytesA = assessedOpLogicalBytes(a);
      const bytesB = assessedOpLogicalBytes(b);
      if ((bytesA == null) !== (bytesB == null)) return bytesA == null ? 1 : -1;
      return Number(bytesB || 0) - Number(bytesA || 0);
    })
    .slice(0, limit);
}

export function buildGraphIndex(analysis) {
  const producers = new Map();
  const consumers = new Map();
  for (const op of analysis.ops) {
    for (const tensorId of op.outputs) {
      if (tensorId >= 0) producers.set(tensorId, op.index);
    }
    for (const tensorId of op.inputs) {
      if (tensorId < 0) continue;
      if (!consumers.has(tensorId)) consumers.set(tensorId, []);
      consumers.get(tensorId).push(op.index);
    }
  }
  return { producers, consumers };
}

export function opMatchesSearch(analysis, op, term) {
  if (!term) return true;
  if (`${op.index}` === term || op.name.toLowerCase().includes(term)) return true;
  if (op.stage_key && op.stage_key.toLowerCase().includes(term)) return true;
  if (op.stage_key && humanizeStageKey(op.stage_key).toLowerCase().includes(term)) return true;
  const tensorIds = [...op.inputs, ...op.outputs].filter((id) => id >= 0);
  if (tensorIds.some((id) => `${id}` === term || `t${id}` === term)) return true;
  return tensorIds.some((id) => {
    const tensor = analysis.tensors[id];
    return tensor?.name?.toLowerCase().includes(term);
  });
}

export function opDetailRows(op, analysis = null) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const isTflite = format === "tflite";
  const l2Bytes = Number(analysis?.target_profile?.l2_bytes || 0);
  const l2Ratio = op.row_working_set_bytes != null && l2Bytes > 0 ? Number(op.row_working_set_bytes) / l2Bytes : null;
  const l2Text = l2Ratio == null ? "N/A - target L2 unavailable" : `${formatBytes(op.row_working_set_bytes)} / ${l2Ratio.toFixed(2)}x L2 / ${l2Ratio >= 0.9 ? l2Ratio > 1 ? "exceeds" : "watch" : "ok"}`;
  const common = [
    ["Output", formatShapes(op.output_shapes)],
    ["MACs", op.macs == null ? `N/A - ${op.macs_reason || "required shape metadata is unavailable"}` : formatNumber(op.macs)],
    ["Bytes", assessedOpLogicalBytes(op) == null ? `N/A - ${op.estimated_bytes_reason || "required tensor shape or dtype is unavailable"}` : formatNumber(op.estimated_bytes)],
    ["Intensity", op.intensity_ops_per_byte != null
      ? `${Number(op.intensity_ops_per_byte).toFixed(2)} ops/byte`
      : `N/A - ${op.macs == null ? op.macs_reason || "MACs unavailable" : op.estimated_bytes_reason || "logical traffic unavailable"}`],
    ["Bound", op.static_bound_guess || "-"],
    ["Roofline reason", decodeRoofReason(op.roofline_reason || "") || "-"],
    ["Stage", op.stage_key != null ? `#${op.stage_index} ${humanizeStageKey(op.stage_key)}` : "-"],
    ["Raw stage key", op.stage_key || "-"],
  ];
  const runtimeSpecific = isTflite ? [
    ["XNNPACK", xnnpackLabel(op, analysis)],
    ["XNNPACK reason", decodeXnnpReason(op.xnnpack_reason || "") || "-"],
    ["Break class", op.xnnpack_chain_break ? `${op.xnnpack_break_class || "break"} / adjacent MACs ${formatPercent(op.chain_break_impact_mac_percent || 0)} / fallback bytes ${formatPercent(op.fallback_byte_percent || 0)}` : "-"],
    ["Partition-planning prior", op.xnnpack_chain_break || op.xnnpack_chain_role === "chain-break" ? `${Number(op.chain_break_overhead_us_low || 0).toFixed(0)}-${Number(op.chain_break_overhead_us_high || 0).toFixed(0)} us cold-start profile range; not measured per-inference latency` : "-"],
    ["Microkernel hint", op.target_microkernel_hint || "-"],
    ["L1 row working set", op.row_working_set_bytes ? `${formatBytes(op.row_working_set_bytes)} / ${Number(op.row_working_set_ratio || 0).toFixed(2)}x / ${op.row_working_set_severity || "none"}` : "-"],
    ["L2 row working set", op.row_working_set_bytes ? l2Text : "-"],
    ["Channel alignment", op.channel_alignment_detail || alignmentLabel(op)],
    ["Channel tail", op.channel_alignment_status === "misaligned" ? formatPercent(op.channel_tail_overhead_percent || 0) : "-"],
    ["Weight packing", op.weight_packing_detail || packingLabel(op)],
  ] : format === "coreml" ? [
    ["Core ML layer", op.coreml_layer_name || "-"],
    ["Core ML type field", op.coreml_layer_type_field ?? "-"],
    ["Pipeline stage", op.pipeline_model_index == null ? "-" : `#${op.pipeline_model_index} ${op.pipeline_model_name || "unnamed"} / ${op.pipeline_model_type || "unknown"}`],
    ["Classical contract", op.coreml_classical_model?.kind || "-"],
    ["Classical structure", op.coreml_classical_model?.kind?.startsWith("treeEnsemble")
      ? `${formatNumber(op.coreml_classical_model.tree_count)} tree(s); ${formatNumber(op.coreml_classical_model.branch_node_count)} branch / ${formatNumber(op.coreml_classical_model.leaf_node_count)} leaf; depth ${formatNumber(op.coreml_classical_model.maximum_depth)}`
      : op.coreml_classical_model?.kind?.startsWith("supportVector")
        ? `${op.coreml_classical_model.kernel.kind}; ${formatNumber(op.coreml_classical_model.support_vectors.count)} ${op.coreml_classical_model.support_vectors.kind} support vector(s)`
        : op.coreml_classical_model?.kind?.startsWith("glm")
          ? `${formatNumber(op.coreml_classical_model.coefficient_row_count)} x ${formatNumber(op.coreml_classical_model.coefficient_width)} coefficient matrix`
          : "-"],
    ["MAC definition", op.coreml_macs_definition || "not emitted"],
    ["Numerical parameter groups", formatNumber(op.coreml_classical_model?.parameters?.length || op.coreml_weights?.length || 0)],
    ["Weight scan", op.coreml_weight_scan_status || "not assessed"],
    ["Runtime compute unit", "Not observed - the artifact does not record CPU/GPU/ANE assignment"],
    ["Runtime partition", "Not observed - a Core ML execution trace was not supplied"],
  ] : [
    ["Static EP assignment", "Not assessed - no ONNX EP rulepack"],
    ["Runtime partition", "Not observed - profile has no partition ID"],
    ["Microkernel", "Not observed - not exposed by ORT node events"],
    ["L1 row working set", op.row_working_set_bytes == null
      ? `N/A - ${op.row_working_set_reason || "not assessed"}`
      : `${formatBytes(op.row_working_set_bytes)} / ${op.row_working_set_ratio == null ? "target L1 N/A" : `${Number(op.row_working_set_ratio).toFixed(2)}x L1`}`],
    ["L2 row working set", op.row_working_set_bytes == null ? `N/A - ${op.row_working_set_reason || "not assessed"}` : l2Text],
  ];
  return [
    ...common,
    ...runtimeSpecific,
    ["Quant state", quantStateLabel(op)],
    ["Quant state detail", op.quantization_detail || "-"],
    ["Quant risk", quantLabel(op)],
    ["Quant compute path", op.quantized_compute_path ? "8-bit activation input/output" : op.quantized_path ? "quant signal only" : "-"],
    ["Quant mode", `${op.quant_scale_mode || "none"} / ratio ${op.quant_scale_ratio_meaningful ? "meaningful" : "N/A"} / zp ${op.quant_zero_point_status || "none"}`],
    ["Quant detail", op.quant_risk_detail || "-"],
    ["Fusion", fusionLabel(op)],
    ["Pattern", op.patterns?.join(" / ") || "-"],
  ];
}

export function opInputTensorDetails(analysis, op, graphIndex = buildGraphIndex(analysis)) {
  return op.inputs.filter((id) => id >= 0).map((id) => {
    const producer = graphIndex.producers.get(id);
    return `${tensorSummary(analysis.tensors[id])} producer=${producer == null ? "model input" : `#${producer}`}`;
  });
}

export function opOutputTensorDetails(analysis, op, graphIndex = buildGraphIndex(analysis)) {
  return op.outputs.filter((id) => id >= 0).map((id) => {
    const consumers = graphIndex.consumers.get(id) || [];
    return `${tensorSummary(analysis.tensors[id])} consumers=${consumers.length ? consumers.map((idx) => `#${idx}`).join(", ") : "model output"}`;
  });
}

export function opLocalLinks(op, graphIndex) {
  return [
    ...op.inputs
      .filter((id) => graphIndex.producers.has(id))
      .map((id) => `#${graphIndex.producers.get(id)} -> #${op.index} via tensor ${id}`),
    ...op.outputs
      .flatMap((id) => graphIndex.consumers.get(id) || [])
      .map((idx) => `#${op.index} -> #${idx}`),
  ];
}

const SPATIAL_OPS = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"]);
export function opKernelLabel(op, analysis) {
  if (!SPATIAL_OPS.has(op.name)) return "-";
  const wIdx = op.inputs?.[1];
  if (wIdx == null || wIdx < 0) return "-";
  const shape = analysis?.tensors?.[wIdx]?.shape;
  if (!Array.isArray(shape) || shape.length !== 4) return "-";
  const kH = shape[1], kW = shape[2];
  return kH === kW ? `${kH}×${kH}` : `${kH}×${kW}`;
}

export function tensorSummary(tensor) {
  if (!tensor) return "unknown tensor";
  const q = tensor.quant_scales
    ? ` quant=${tensor.quant_scales} scale=${tensor.scale_sample?.join("/") || "-"} zp=${tensor.zero_point_sample?.join("/") || "-"} qdim=${tensor.quantized_dimension}`
    : "";
  return `T${tensor.index} ${tensor.name || "(unnamed)"} ${tensor.dtype}${tensorShapeText(tensor)}${q}`;
}

// Topology is now computed in WASM (op.topo_role, op.topo_depth, op.topo_fan_out_max, op.patterns).
// app.js uses buildTopologyAnnotationsFromWasm() which reads those fields directly.
// Empty stub kept to avoid crashing any stale import.
export function buildTopologyAnnotations(_analysis) {
  return { opAnnotations: new Map(), tensorFanOut: new Map() };
}
