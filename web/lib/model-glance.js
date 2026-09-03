import { artifactIrOperators } from "./artifact-ir-selectors.js";
import {
  modelQuantizationStatus,
  opLogicalRowPayloadBytes,
  predictedPartitionBoundaryInventory,
} from "./analysis.js";
import { deriveTfliteBatchOneProjection } from "./dynamic-shape-cost.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";

export const CACHE_WATCH_RATIO = 0.9;
export const MODEL_GLANCE_SCHEMA = "deepbom.model_at_a_glance.v1.3";

function finiteNonNegative(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function nullableFiniteNonNegative(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : 0;
}

export function cachePressureForAnalysis(analysis, {
  l1DataBytes = analysis?.target_profile?.l1_data_bytes,
  l2Bytes = analysis?.target_profile?.l2_bytes,
  l2CapacityScope = analysis?.target_profile?.l2_capacity_scope,
} = {}) {
  const rows = (artifactIrOperators(analysis) || [])
    .filter((op) => opLogicalRowPayloadBytes(op) > 0
      && (!op.row_working_set_ratio_status || op.row_working_set_ratio_status === "assessed"))
    .map((op) => ({
      opIndex: Number(op.index),
      opName: String(op.name || "UNKNOWN"),
      rowWorkingSetBytes: opLogicalRowPayloadBytes(op),
    }))
    .filter((row) => row.rowWorkingSetBytes > 0);
  const summarize = (bytes, assessmentStatus = "assessed") => {
    const denominator = finiteNonNegative(bytes);
    if (!denominator) return { bytes: 0, maxRatio: null, watchCount: null, assessmentStatus: "not_assessed_no_capacity" };
    const rawMaxRatio = rows.reduce((maximum, row) => Math.max(maximum, row.rowWorkingSetBytes / denominator), 0);
    if (assessmentStatus !== "assessed") {
      return { bytes: denominator, maxRatio: null, rawMaxRatio, watchCount: null, assessmentStatus };
    }
    return {
      bytes: denominator,
      maxRatio: rawMaxRatio,
      rawMaxRatio,
      watchCount: rows.filter((row) => row.rowWorkingSetBytes / denominator >= CACHE_WATCH_RATIO).length,
      assessmentStatus,
    };
  };
  const l1 = summarize(l1DataBytes);
  const l2Scope = String(l2CapacityScope || "unbound");
  const l2AssessmentStatus = l2Scope.startsWith("private_per_core")
    ? "assessed"
    : `not_assessed_${l2Scope}`;
  const l2 = summarize(l2Bytes, l2AssessmentStatus);
  return {
    assessedOpCount: rows.length,
    rows: rows
      .map((row) => ({
        ...row,
        l1Ratio: l1.bytes ? row.rowWorkingSetBytes / l1.bytes : null,
        l2Ratio: l2.assessmentStatus === "assessed" && l2.bytes ? row.rowWorkingSetBytes / l2.bytes : null,
        rawL2CapacityRatio: l2.bytes ? row.rowWorkingSetBytes / l2.bytes : null,
      }))
      .sort((left, right) => Number(right.l1Ratio || 0) - Number(left.l1Ratio || 0)
        || right.rowWorkingSetBytes - left.rowWorkingSetBytes
        || left.opIndex - right.opIndex),
    l1,
    l2,
  };
}

export function opLatencyComponentLedger(analysis) {
  const rows = [];
  for (const op of artifactIrOperators(analysis) || []) {
    const computeUs = finiteNonNegative(op.bottleneck_compute_us);
    const memoryUs = finiteNonNegative(op.bottleneck_memory_us);
    const packingUs = finiteNonNegative(op.bottleneck_packing_us);
    const boundaryUs = finiteNonNegative(op.bottleneck_break_us);
    const fallbackUs = finiteNonNegative(op.bottleneck_fallback_us);
    const totalUs = finiteNonNegative(op.bottleneck_total_us);
    if (!totalUs && !computeUs && !memoryUs && !packingUs && !boundaryUs && !fallbackUs) continue;
    const rooflineUs = Math.max(computeUs, memoryUs);
    const epsilon = Math.max(1e-12, rooflineUs * 1e-12);
    const activeComputeUs = computeUs > memoryUs + epsilon ? rooflineUs : 0;
    const activeMemoryUs = memoryUs > computeUs + epsilon ? rooflineUs : 0;
    const mixedRooflineUs = rooflineUs > 0 && !activeComputeUs && !activeMemoryUs ? rooflineUs : 0;
    const namedUs = activeComputeUs + activeMemoryUs + mixedRooflineUs + packingUs + boundaryUs + fallbackUs;
    const otherUs = Math.max(0, totalUs - namedUs);
    const reconstructedUs = namedUs + otherUs;
    const steadyStateUs = Math.max(0, totalUs - packingUs - boundaryUs);
    rows.push({
      opIndex: Number(op.index),
      opName: String(op.name || "UNKNOWN"),
      totalUs,
      coldStartUs: totalUs,
      steadyStateUs,
      activeComputeUs,
      activeMemoryUs,
      mixedRooflineUs,
      packingUs,
      boundaryUs,
      fallbackUs,
      otherUs,
      reconstructionDeltaUs: totalUs - reconstructedUs,
      dominant: String(op.bottleneck_dominant || ""),
      outputShapes: op.output_shapes || [],
      macs: finiteNonNegative(op.macs),
      intensity: op.arithmetic_intensity == null ? null : Number(op.arithmetic_intensity),
      rowWorkingSetBytes: op.row_working_set_bytes == null ? null : finiteNonNegative(op.row_working_set_bytes),
    });
  }
  rows.sort((left, right) => right.steadyStateUs - left.steadyStateUs
    || left.opIndex - right.opIndex);
  const totals = rows.reduce((sum, row) => {
    sum.totalUs += row.totalUs;
    sum.coldStartUs += row.coldStartUs;
    sum.steadyStateUs += row.steadyStateUs;
    sum.activeComputeUs += row.activeComputeUs;
    sum.activeMemoryUs += row.activeMemoryUs;
    sum.mixedRooflineUs += row.mixedRooflineUs;
    sum.packingUs += row.packingUs;
    sum.boundaryUs += row.boundaryUs;
    sum.fallbackUs += row.fallbackUs;
    sum.otherUs += row.otherUs;
    sum.absoluteReconstructionDeltaUs += Math.abs(row.reconstructionDeltaUs);
    return sum;
  }, {
    totalUs: 0,
    coldStartUs: 0,
    steadyStateUs: 0,
    activeComputeUs: 0,
    activeMemoryUs: 0,
    mixedRooflineUs: 0,
    packingUs: 0,
    boundaryUs: 0,
    fallbackUs: 0,
    otherUs: 0,
    absoluteReconstructionDeltaUs: 0,
  });
  return {
    rows,
    totals,
    conservationStatus: totals.absoluteReconstructionDeltaUs <= Math.max(1e-9, totals.totalUs * 1e-9)
      ? "conserved"
      : "not_conserved",
  };
}

function derivedCpuIslandCount(ops) {
  let count = 0;
  let inside = false;
  for (const op of ops) {
    const delegated = Number(op.xnnpack_chain_id) >= 0
      && op.xnnpack_supported !== false
      && !op.xnnpack_chain_break;
    if (!delegated && !inside) count += 1;
    inside = !delegated;
  }
  return count;
}

function scorePenalties(analysis) {
  const breakdown = analysis?.insights?.score_breakdown;
  if (!breakdown || typeof breakdown !== "object") return [];
  return Object.entries(breakdown)
    .filter(([key, value]) => key.endsWith("_penalty") && finiteNonNegative(value) > 0)
    .map(([key, value]) => ({ key, points: finiteNonNegative(value) }))
    .sort((left, right) => right.points - left.points || left.key.localeCompare(right.key));
}

function delegationMotifAttribution(analysis) {
  const ops = artifactIrOperators(analysis) || [];
  const breaks = ops.filter((op) => op.xnnpack_chain_break);
  const squeezeExcitationBlocks = (analysis?.block_inventory?.blocks || [])
    .filter((block) => block.block_type === "squeeze_excitation");
  const squeezeExcitationOps = new Set(
    squeezeExcitationBlocks.flatMap((block) => block.op_indices || []).map(Number),
  );
  const breaksInSqueezeExcitation = breaks.filter((op) => squeezeExcitationOps.has(Number(op.index)));
  const meanBreaks = breaks.filter((op) => op.name === "MEAN");
  const meanBreaksInSqueezeExcitation = meanBreaks
    .filter((op) => squeezeExcitationOps.has(Number(op.index)));
  const familyCounts = [...breaksInSqueezeExcitation.reduce((counts, op) => {
    counts.set(op.name, (counts.get(op.name) || 0) + 1);
    return counts;
  }, new Map())]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([opName, count]) => ({ opName, count }));
  let summary = "";
  if (meanBreaks.length > 0 && meanBreaksInSqueezeExcitation.length === meanBreaks.length) {
    summary = `All ${meanBreaks.length} MEAN predicted breaks occur inside ${squeezeExcitationBlocks.length} squeeze-excitation block(s), identifying global-pooling/export semantics as the first review scope.`;
  } else if (breaksInSqueezeExcitation.length > 0) {
    summary = `${breaksInSqueezeExcitation.length}/${breaks.length} predicted breaks occur inside ${squeezeExcitationBlocks.length} squeeze-excitation block(s).`;
  }
  return {
    status: summary ? "derived" : "not_identified",
    squeezeExcitationBlockCount: squeezeExcitationBlocks.length,
    predictedBreakCount: breaks.length,
    breaksInSqueezeExcitationCount: breaksInSqueezeExcitation.length,
    meanBreakCount: meanBreaks.length,
    meanBreaksInSqueezeExcitationCount: meanBreaksInSqueezeExcitation.length,
    familyCounts,
    summary,
  };
}

function padFusionDirectConsumerInventory(ops) {
  const convolutionFamilies = new Set(["CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"]);
  const rows = ops.filter((op) => op.name === "PAD" || op.name === "PADV2").map((op) => {
    const outputs = new Set((op.outputs || []).map(Number));
    const consumers = ops.filter((candidate) => (candidate.inputs || []).some((input) => outputs.has(Number(input))));
    const directConvolutionCandidate = outputs.size > 0
      && consumers.length === 1
      && convolutionFamilies.has(consumers[0].name);
    return {
      opIndex: Number(op.index),
      consumerOpIndices: consumers.map((consumer) => Number(consumer.index)),
      consumerFamilies: consumers.map((consumer) => String(consumer.name || "UNKNOWN")),
      directConvolutionCandidate,
      runtimeFusionObserved: false,
      evidenceClass: directConvolutionCandidate ? "DERIVED_DIRECT_CONV_CANDIDATE" : "DERIVED_NOT_DIRECT_CONV_CANDIDATE",
    };
  });
  return {
    candidateCount: rows.length,
    directConvolutionCandidateCount: rows.filter((row) => row.directConvolutionCandidate).length,
    allPadNodesAreDirectConvolutionCandidates: rows.length > 0 && rows.every((row) => row.directConvolutionCandidate),
    runtimeFusionObservedCount: 0,
    rows,
    interpretationBoundary: "A single convolution-family consumer identifies a PAD-folding candidate, not XNNPACK fusion eligibility. Static analysis does not prove that the selected runtime folds the PAD node or that its materialization cost is zero.",
  };
}

export function buildModelAtGlance(analysis, cacheOptions = {}) {
  const onnx = String(analysis?.format || "").toLowerCase() === "onnx";
  const ops = artifactIrOperators(analysis) || [];
  const batchOneProjection = onnx ? null : deriveTfliteBatchOneProjection(analysis);
  const batchOneBound = batchOneProjection?.status === "assumption_bound_batch_one";
  const shapeBindingRequired = batchOneProjection?.status === "requires_explicit_shape_binding"
    || batchOneProjection?.status === "batch_one_projection_formula_incomplete";
  const incompleteOnnxMacLedger = onnx && analysis?.total_macs == null;
  const quantization = modelQuantizationStatus(analysis);
  const quantResearchCoverage = onnx
    ? null
    : analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const assessedIntensityOps = shapeBindingRequired ? [] : ops.filter((op) => (
    op.intensity_status == null || op.intensity_status === "assessed"
  ) && ["compute-bound", "mixed", "memory-bound"].includes(op.static_bound_guess));
  const countBound = (bound) => assessedIntensityOps.filter((op) => op.static_bound_guess === bound).length;
  const boundary = onnx ? null : predictedPartitionBoundaryInventory(analysis);
  const cache = shapeBindingRequired ? {
    assessedOpCount: 0,
    rows: [],
    l1: { bytes: finiteNonNegative(cacheOptions.l1DataBytes ?? analysis?.target_profile?.l1_data_bytes), maxRatio: null, watchCount: 0 },
    l2: { bytes: finiteNonNegative(cacheOptions.l2Bytes ?? analysis?.target_profile?.l2_bytes), maxRatio: null, watchCount: null, assessmentStatus: "not_assessed_dynamic_shape" },
  } : cachePressureForAnalysis(analysis, cacheOptions);
  const latency = onnx || shapeBindingRequired ? {
    rows: [],
    totals: {
      totalUs: null,
      coldStartUs: null,
      steadyStateUs: null,
      activeComputeUs: null,
      activeMemoryUs: null,
      mixedRooflineUs: null,
      packingUs: null,
      boundaryUs: null,
      fallbackUs: null,
      otherUs: null,
      absoluteReconstructionDeltaUs: null,
    },
    conservationStatus: shapeBindingRequired ? "not_assessed_dynamic_shape" : "not_assessed",
  } : opLatencyComponentLedger(analysis);
  const delegatedOpCount = onnx
    ? 0
    : ops.filter((op) => Number(op.xnnpack_chain_id) >= 0).length;
  const repair = analysis?.delegation_repair;
  const cpuIslandCount = onnx
    ? 0
    : Number.isInteger(repair?.cpu_island_count)
      ? repair.cpu_island_count
      : derivedCpuIslandCount(ops);
  const quantizedTensorCount = finiteNonNegative(analysis?.quantized_tensors);
  const perChannelTensorCount = finiteNonNegative(analysis?.per_channel_tensors);
  const penalties = scorePenalties(analysis);
  const scoreValue = analysis?.insights?.score == null ? null : Number(analysis.insights.score);
  const scoreBase = analysis?.insights?.score_breakdown?.base == null
    ? null
    : Number(analysis.insights.score_breakdown.base);
  const scoreFinal = analysis?.insights?.score_breakdown?.final_score == null
    ? null
    : Number(analysis.insights.score_breakdown.final_score);
  const scorePenaltyTotal = penalties.reduce((sum, penalty) => sum + penalty.points, 0);
  const expectedScore = scoreBase == null ? null : Math.max(0, scoreBase - scorePenaltyTotal);
  const scoreConserved = expectedScore != null
    && scoreFinal != null
    && scoreValue != null
    && Math.abs(scoreFinal - expectedScore) <= 1e-12
    && Math.abs(scoreValue - scoreFinal) <= 1e-12;
  const frontierTargets = (shapeBindingRequired ? [] : analysis?.deployment_frontier?.targets || []).map((target) => ({
    id: String(target.target_id || ""),
    label: String(target.target_label || target.target_id || "Target"),
    totalUs: finiteNonNegative(target.cold_start_us ?? target.total_us),
    coldStartUs: finiteNonNegative(target.cold_start_us ?? target.total_us),
    steadyStateUs: finiteNonNegative(
      target.steady_state_us
        ?? Math.max(
          0,
          finiteNonNegative(target.total_us)
            - finiteNonNegative(target.components?.packing_us)
            - finiteNonNegative(target.components?.boundary_us),
        ),
    ),
    l1DataBytes: finiteNonNegative(target.l1_data_bytes),
    l2Bytes: finiteNonNegative(target.l2_bytes),
    maxL1Ratio: finiteNonNegative(target.max_l1_ratio),
    l1WatchCount: finiteNonNegative(target.l1_watch_count),
  }));
  const boundaryOverhead = onnx ? { lowUs: 0, midpointUs: 0, highUs: 0 } : ops
    .filter((op) => op.xnnpack_chain_break)
    .reduce((sum, op) => {
      const lowUs = finiteNonNegative(op.chain_break_overhead_us_low);
      const highUs = finiteNonNegative(op.chain_break_overhead_us_high);
      sum.lowUs += lowUs;
      sum.highUs += highUs;
      sum.midpointUs += finiteNonNegative(op.bottleneck_break_us) || (lowUs + highUs) / 2;
      return sum;
    }, { lowUs: 0, midpointUs: 0, highUs: 0 });
  const padFusion = onnx ? null : padFusionDirectConsumerInventory(ops);
  const candidatePadOpIndices = new Set((padFusion?.rows || [])
    .filter((row) => row.directConvolutionCandidate)
    .map((row) => row.opIndex));
  const padFusionRecoverableUpperBoundUs = latency.rows
    .filter((row) => {
      const op = ops.find((candidate) => Number(candidate.index) === row.opIndex);
      return candidatePadOpIndices.has(row.opIndex)
        && op?.fusion_status === "runtime folding unobserved";
    })
    .reduce((sum, row) => sum + row.steadyStateUs, 0);
  latency.range = onnx || shapeBindingRequired ? {
    status: shapeBindingRequired ? "not_assessed_dynamic_shape" : "not_assessed",
    reason: shapeBindingRequired ? "non-batch dynamic dimensions require an explicit shape binding" : "no cross-host static latency model is defined",
    steadyLowUs: null,
    steadyPointUs: null,
    steadyHighUs: null,
    coldLowUs: null,
    coldPointUs: null,
    coldHighUs: null,
  } : {
    status: boundaryOverhead.highUs > 0 ? "cold_setup_profile_range" : "point_only",
    steadyLowUs: Math.max(0, latency.totals.steadyStateUs - padFusionRecoverableUpperBoundUs),
    steadyPointUs: latency.totals.steadyStateUs,
    steadyHighUs: latency.totals.steadyStateUs,
    coldLowUs: Math.max(0, latency.totals.steadyStateUs - padFusionRecoverableUpperBoundUs)
      + latency.totals.packingUs
      + boundaryOverhead.lowUs,
    coldPointUs: latency.totals.coldStartUs,
    coldHighUs: latency.totals.steadyStateUs
      + latency.totals.packingUs
      + boundaryOverhead.highUs,
    padFusionRecoverableUpperBoundUs,
    method: "Steady state excludes one-time packing and the setup-only partition-planning profile. Cold low/high add packing plus the summed setup profile low/high. The low bound also removes complete direct PAD-to-convolution steady rows as a runtime-unobserved candidate upper bound; point/high retain explicit PAD materialization.",
  };
  const fileSizeBytes = finiteNonNegative(analysis?.file_size);
  const arenaBytes = onnx || shapeBindingRequired || analysis?.tensor_arena_plan?.combined_arena_bytes == null
    ? null
    : finiteNonNegative(analysis.tensor_arena_plan.combined_arena_bytes);
  const motifAttribution = onnx ? null : delegationMotifAttribution(analysis);
  return {
    schema: MODEL_GLANCE_SCHEMA,
    format: onnx ? "onnx" : "tflite",
    artifact: {
      filename: String(analysis?.filename || ""),
      fileSizeBytes,
      opCount: Number(analysis?.operator_count ?? ops.length),
      tensorCount: Number(analysis?.tensor_count || 0),
      totalMacs: shapeBindingRequired
        ? null
        : batchOneBound
        ? finiteNonNegative(batchOneProjection.projected_total_macs)
        : nullableFiniteNonNegative(analysis?.total_macs),
      totalMacsEvidenceClass: shapeBindingRequired
        ? "NOT_ASSESSED_DYNAMIC_SHAPE"
        : batchOneBound
          ? "ASSUMPTION_BOUND_N_EQ_1"
          : incompleteOnnxMacLedger
            ? "NOT_ASSESSED_INCOMPLETE_MAC_LEDGER"
            : "OBSERVED_OR_DERIVED",
      dynamicShapeProjection: batchOneProjection,
      macAssessedComputeOps: Number(analysis?.mac_assessment?.assessed_compute_ops || 0),
      macComputeOps: Number(analysis?.mac_assessment?.compute_ops || 0),
    },
    quantization: {
      classification: String(quantization.classification || "unknown"),
      label: String(quantization.label || quantization.classification || "Unknown"),
      artifactClass: quantResearchCoverage?.artifact_class || "not_applicable",
      artifactClassLabel: quantResearchCoverage?.artifact_class_label || "Not applicable",
      researchCoverage: quantResearchCoverage,
      quantizedTensorCount,
      perChannelTensorCount,
      perChannelTensorRatio: ratio(perChannelTensorCount, quantizedTensorCount),
      quantizedComputeMacRatio: quantization.quantized_compute_mac_percent == null
        ? null
        : Number(quantization.quantized_compute_mac_percent),
    },
    intensity: {
      highCount: countBound("compute-bound"),
      mixedCount: countBound("mixed"),
      lowCount: countBound("memory-bound"),
      assessedOpCount: assessedIntensityOps.length,
      opCount: ops.length,
      ridgeOpsPerByte: onnx ? null : finiteNonNegative(analysis?.target_profile?.ridge_point_ops_per_byte),
    },
    latency,
    delegation: onnx ? {
      status: "not_applicable",
      delegatedOpCount: 0,
      opCount: ops.length,
      delegatedMacRatio: null,
      boundaryEdgeCount: 0,
      boundaryPayloadBytes: null,
      boundaryPayloadStatus: "not_applicable",
      cpuIslandCount: 0,
    } : {
      status: "predicted",
      delegatedOpCount,
      opCount: ops.length,
      delegatedMacRatio: shapeBindingRequired || analysis?.delegated_mac_percent == null
        ? null
        : Number(analysis.delegated_mac_percent),
      boundaryEdgeCount: Number(boundary?.edge_count || 0),
      boundaryPayloadBytes: boundary?.summed_edge_payload_bytes == null
        ? finiteNonNegative(boundary?.assessed_edge_payload_bytes)
        : finiteNonNegative(boundary.summed_edge_payload_bytes),
      boundaryPayloadStatus: boundary?.summed_edge_payload_bytes == null ? "partial" : "complete",
      cpuIslandCount,
      boundaryOverhead,
      boundaryOverheadEvidenceClass: "HEURISTIC_PROFILE_SETUP_RANGE",
      motifAttribution,
      padFusion,
    },
    memory: {
      peakLiveBytes: shapeBindingRequired
        ? null
        : batchOneBound && batchOneProjection.projected_peak_live_payload_bytes != null
        ? finiteNonNegative(batchOneProjection.projected_peak_live_payload_bytes)
        : analysis?.tensor_liveness?.peak_bytes == null
          ? null
          : finiteNonNegative(analysis.tensor_liveness.peak_bytes),
      peakLiveEvidenceClass: shapeBindingRequired
        ? "NOT_ASSESSED_DYNAMIC_SHAPE"
        : batchOneBound
        ? "ASSUMPTION_BOUND_N_EQ_1"
        : analysis?.tensor_liveness?.peak_bytes == null
          ? "NOT_ASSESSED"
          : "DERIVED",
      arenaBytes,
      artifactPlusArenaBytes: arenaBytes == null ? null : fileSizeBytes + arenaBytes,
      artifactPlusArenaEvidenceClass: arenaBytes == null
        ? "NOT_ASSESSED"
        : "OBSERVED_ARTIFACT_PLUS_DERIVED_ARENA",
      cache,
      cacheAssumption: String(analysis?.target_profile?.cache_assumption || ""),
      cacheSourceUrl: String(analysis?.target_profile?.cache_source_url || ""),
      hardwareSpec: analysis?.target_profile?.hardware_spec || null,
      performanceModelEvidenceClass: String(analysis?.target_profile?.performance_model_evidence_class || "HEURISTIC"),
      cacheOverride: finiteNonNegative(cacheOptions.l1DataBytes) > 0
        && finiteNonNegative(cacheOptions.l1DataBytes) !== finiteNonNegative(analysis?.target_profile?.l1_data_bytes),
    },
    target: {
      id: String(analysis?.target_profile?.id || ""),
      label: String(analysis?.target_profile?.label || ""),
      l1DataBytes: finiteNonNegative(analysis?.target_profile?.l1_data_bytes),
      l2Bytes: finiteNonNegative(analysis?.target_profile?.l2_bytes),
      effectivePeakGops: finiteNonNegative(analysis?.target_profile?.effective_peak_gops),
      computeUtilizationFactor: Number(analysis?.target_profile?.compute_utilization_factor ?? 1),
      effectiveMemoryBandwidthGbps: finiteNonNegative(analysis?.target_profile?.effective_memory_bandwidth_gbps),
      memoryBoundIntensity: finiteNonNegative(analysis?.target_profile?.memory_bound_intensity),
      computeBoundIntensity: finiteNonNegative(analysis?.target_profile?.compute_bound_intensity),
      performanceModelEvidenceClass: String(analysis?.target_profile?.performance_model_evidence_class || "HEURISTIC"),
    },
    frontierTargets,
    score: {
      evidenceClass: String(analysis?.insights?.score_evidence_class || "HEURISTIC"),
      value: scoreValue,
      base: scoreBase,
      final: scoreFinal,
      penaltyTotal: scorePenaltyTotal,
      expected: expectedScore,
      conservationStatus: scoreValue == null ? "not_assessed" : scoreConserved ? "conserved" : "not_conserved",
      penalties,
      method: String(analysis?.insights?.score_method || ""),
    },
  };
}
