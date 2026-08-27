import { formatBytes, formatNumber } from "./format.js";
import { ANALYZER_METADATA } from "./report-metadata.js";
import { code } from "./report-utils.js";

export function intensityPosture(bound) {
  return bound === "compute-bound" ? "high-intensity"
    : bound === "memory-bound" ? "low-intensity"
      : bound === "mixed" ? "mixed-intensity" : (bound || "-");
}

export function simdAssumptionsForAnalysis(analysis = {}) {
  const targetProfile = analysis?.target_profile || {};
  const hardwareSpec = targetProfile.hardware_spec || null;
  const sourceCandidateOps = (analysis?.ops || []).filter((op) => String(op.xnnpack_kernel_evidence_class || "").startsWith("SOURCE_ENUMERATED_CANDIDATE"));
  const sourceCandidates = sourceCandidateOps.flatMap((op) => op.xnnpack_kernel_candidates || []);
  if (sourceCandidates.length) {
    const tiles = [...new Set(sourceCandidates.map((candidate) => {
      if (Number(candidate.tile_nr || 0) > 0) return `MRxNR ${candidate.tile_mr || "?"}x${candidate.tile_nr}`;
      if (Number(candidate.channel_tile || 0) > 0) return `depthwise ${candidate.primary_tile || "?"}p${candidate.channel_tile}c`;
      return null;
    }).filter(Boolean))];
    const classes = [...new Set(sourceCandidateOps.map((op) => op.xnnpack_kernel_evidence_class))];
    return `${classes.join("/")}: ${tiles.join(" / ")}; candidate configuration does not identify the executed runtime microkernel`;
  }
  const int8Lanes = Number(targetProfile.int8_lanes || 0);
  const fp16Lanes = Number(targetProfile.fp16_lanes || 0);
  const fp32Lanes = Number(targetProfile.fp32_lanes || 0);
  return [
    "HEURISTIC_PROFILE; output-channel microkernel tile not source-enumerated",
    int8Lanes ? `SIMD 8-bit integer lanes x${int8Lanes}` : null,
    fp16Lanes ? `128/256-bit register capacity FP16 x${fp16Lanes}; native vector arithmetic ${hardwareSpec ? hardwareSpec.fp16_vector_arithmetic ? "documented" : "not documented for this core" : "not source-bound"}` : null,
    fp32Lanes ? `SIMD FP32 lanes x${fp32Lanes}` : null,
    targetProfile.dot_product ? "dot-product" : null,
    targetProfile.sve2 ? "SVE2" : null,
    targetProfile.in_order ? "in-order core" : null,
  ].filter(Boolean).join(" / ") || "-";
}

export function formatRidge(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return number.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1");
}

export function xnnpackBuildRequirementsSummary(analysis) {
  const values = [...new Set((analysis?.ops || [])
    .filter((op) => op.xnnpack_supported)
    .map((op) => op.xnnpack_build_requirement)
    .filter(Boolean))];
  const applicable = values.map((value) => {
    const [literal, ...caveat] = String(value).split(";");
    return `${code(literal.trim())}${caveat.length ? ` (${caveat.join(";").trim()})` : ""}`;
  });
  const rulepackCount = Number(ANALYZER_METADATA.rulepackProvenance.xnnpackDelegateRuntimeOnlyRequirementCount || 0);
  return `${formatNumber(rulepackCount)} rulepack runtime-only requirement(s); ${formatNumber(applicable.length)} artifact-applicable${applicable.length ? `: ${applicable.join(" / ")}` : "; selected artifact does not exercise a conditional path"}`;
}

export function delegationRuleBasisText(analysis) {
  if (isOnnxAnalysis(analysis)) {
    const compatibility = analysis?.ort_compatibility_evidence;
    return compatibility
      ? `Pinned ${compatibility.source_ref || ANALYZER_METADATA.rulepackProvenance.ortSourceRef}; source/version and artifact-visible definite-exclusion precheck only; GetCapability and runtime assignment remain NOT_OBSERVED`
      : "ONNX Runtime execution-provider assignment is NOT_ASSESSABLE in this report; no protected pinned ORT EP rulepack or profiling/session assignment log was supplied";
  }
  const alternate = analysis?.tflite_delegate_compatibility_evidence;
  const selectorStatus = String(analysis?.xnnpack_selector_assessment_status || "not_reported");
  const selectorBoundary = ["not_loaded", "not_available_for_profile", "not_reported"].includes(selectorStatus)
    ? ` Per-op microkernel selection is ${selectorStatus}; no source-enumerated kernel candidate is claimed for this profile.`
    : ` Per-op selector evidence status is ${selectorStatus}; candidate sets are not observed dispatch.`;
  return alternate
    ? `${ANALYZER_METADATA.xnnpackSupportBasis} - XNNPACK predictions; TFLite GPU/NNAPI candidate prechecks from TensorFlow ${alternate.tensorflow_source_commit}; selected build, device acceptance, and runtime assignment remain unobserved.${selectorBoundary}`
    : `${ANALYZER_METADATA.xnnpackSupportBasis} - XNNPACK predictions only; runtime logs authoritative.${selectorBoundary}`;
}

export function peakArenaReconciliation(analysis, live) {
  const plan = analysis?.tensor_arena_plan;
  if (live?.peak_bytes == null || plan?.combined_arena_bytes == null) return "";
  const delta = Number(live.peak_bytes) - Number(plan.combined_arena_bytes);
  const allocationIds = new Set((plan.allocations || []).map((row) => Number(row.tensor_index)));
  const externalInput = (analysis?.inputs || []).find((tensor) => (
    staticTensorPayloadBytes(tensor) === Math.abs(delta)
    && !allocationIds.has(Number(tensor.index))
  ));
  if (delta > 0 && externalInput) {
    return `Peak-live exceeds the combined arena projection by ${formatBytes(delta)} (${formatNumber(delta)} B), exactly matching external input T${externalInput.index} ${code(externalInput.name || "unnamed")}. The liveness sweep includes its graph-input lifetime; the pinned ArenaPlanner projection does not allocate that external input as an arena root.`;
  }
  return `Peak-live minus combined arena projection is ${delta >= 0 ? "+" : ""}${formatNumber(delta)} B. No single external-input exclusion exactly explains the difference; inspect aliases, lifetime conventions, alignment, persistent allocation, and unmodeled scratch separately.`;
}

export function staticL2RatioForTarget(analysis, target) {
  const l2Bytes = Number(target?.l2_bytes || 0);
  if (!(l2Bytes > 0)) return "N/A";
  const maxBytes = Math.max(0, ...(analysis?.ops || []).map((op) => Number(op?.cache_payload?.logical_row_payload_bytes || 0)));
  return `${(maxBytes / l2Bytes).toFixed(2)}x`;
}

export function reachabilityDivergenceChannelPartition(reachability) {
  const result = { exactOnly: 0, unresolvedOnly: 0, both: 0, neither: 0, total: 0 };
  for (const row of reachability?.ops || []) {
    const exact = row.channel_exact_reachable_divergent_state_counts_decimal || [];
    const unresolved = row.channel_unresolved_divergent_state_counts_decimal || [];
    const count = Math.max(Number(row.assessed_channel_count || 0), exact.length, unresolved.length);
    for (let index = 0; index < count; index += 1) {
      const hasExact = BigInt(exact[index] || "0") > 0n;
      const hasUnresolved = BigInt(unresolved[index] || "0") > 0n;
      if (hasExact && hasUnresolved) result.both += 1;
      else if (hasExact) result.exactOnly += 1;
      else if (hasUnresolved) result.unresolvedOnly += 1;
      else result.neither += 1;
      result.total += 1;
    }
  }
  return result;
}

function staticTensorPayloadBytes(tensor) {
  const shape = Array.isArray(tensor?.shape) ? tensor.shape.map(Number) : [];
  if (!shape.length || shape.some((value) => !Number.isSafeInteger(value) || value < 0)) return null;
  const width = ({ FLOAT64: 8, INT64: 8, UINT64: 8, FLOAT32: 4, INT32: 4, UINT32: 4, FLOAT16: 2, BFLOAT16: 2, INT16: 2, UINT16: 2, INT8: 1, UINT8: 1, BOOL: 1 })[String(tensor?.dtype || "").toUpperCase()];
  if (!width) return null;
  return shape.reduce((product, value) => product * value, 1) * width;
}

function isOnnxAnalysis(analysis) {
  return String(analysis?.format || "").toLowerCase() === "onnx";
}
