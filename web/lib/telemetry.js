import { sha256Hex } from "./hash.js";
import { stableStringify, tensorShapeText } from "./format.js";

export function buildStructureTelemetryDraft(analysis, {
  targetId = "",
  targetProfileLabel = "",
  browserBucket = "",
} = {}) {
  const opHistogram = Object.fromEntries((analysis?.histogram || []).map((item) => [item.name, Number(item.count || 0)]));
  const stageStructure = (analysis?.stages || []).map((stage) => ({
    index: stage.index,
    key: stage.key,
    op_count: stage.op_count,
    mac_percent: stage.mac_percent || 0,
    delegated_mac_percent: stage.delegated_mac_percent || 0,
    xnnpack_chain_breaks: stage.xnnpack_chain_breaks || 0,
    patterns: stage.patterns || [],
  }));
  const boundMix = (analysis?.ops || []).reduce((acc, op) => {
    const key = op.static_bound_guess || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const patterns = (analysis?.patterns || []).map((pattern) => pattern.name || "").filter(Boolean);
  const tensorsById = Object.fromEntries((analysis?.tensors || []).map((t) => [t.index, t]));
  const opSequence = (analysis?.ops || []).slice(0, 1200).map((op) => {
    const inTensors = (op.inputs || []).map((idx) => tensorsById[idx]).filter(Boolean);
    const outTensors = (op.outputs || []).map((idx) => tensorsById[idx]).filter(Boolean);
    return {
      index: op.index,
      name: op.name,
      domain: op.domain || (analysis?.format === "onnx" ? "ai.onnx" : ""),
      input_shapes: inTensors.map((t) => t.shape || []),
      output_shapes: outTensors.map((t) => t.shape || []),
      input_dtypes: inTensors.map((t) => t.dtype || "UNKNOWN"),
      output_dtypes: outTensors.map((t) => t.dtype || "UNKNOWN"),
      macs: op.macs == null ? null : Number(op.macs),
      macs_status: op.macs_status || (op.macs == null ? "not_assessed" : "assessed"),
      estimated_bytes: op.estimated_bytes == null ? null : Number(op.estimated_bytes),
      estimated_bytes_status: op.estimated_bytes_status || (op.estimated_bytes == null ? "not_assessed" : "assessed"),
      quantized_path: Boolean(op.quantized_path),
      quant_scale_mode: op.quant_scale_mode || "",
      xnnpack_break_class: op.xnnpack_break_class || "",
    };
  });
  return {
    format: analysis?.format || "tflite",
    target: targetId,
    target_profile: analysis?.target_profile?.label || targetProfileLabel,
    op_histogram: opHistogram,
    stage_structure: stageStructure,
    op_sequence: opSequence,
    chain_breaks: Number(analysis?.xnnpack_chain_breaks || 0),
    effective_chain_breaks: Number(analysis?.xnnpack_effective_chain_breaks || 0),
    structural_chain_breaks: Number(analysis?.xnnpack_structural_chain_breaks || 0),
    total_macs: analysis?.total_macs == null ? null : Number(analysis.total_macs),
    op_count: (analysis?.ops || []).length,
    tensor_count: Number(analysis?.tensor_count || (analysis?.tensors || []).length || 0),
    input_contract: (analysis?.inputs || []).map((tensor) => `${tensor.dtype}${tensorShapeText(tensor)}`),
    output_contract: (analysis?.outputs || []).map((tensor) => `${tensor.dtype}${tensorShapeText(tensor)}`),
    bound_mix: boundMix,
    fallback_byte_percent: Number(analysis?.fallback_byte_percent || 0),
    delegated_mac_percent: Number(analysis?.delegated_mac_percent || 0),
    patterns,
    browser_bucket: browserBucket,
  };
}

export function structureFingerprintCanonical(structure) {
  return {
    format: structure.format,
    op_histogram: structure.op_histogram,
    stage_structure: structure.stage_structure,
    input_contract: structure.input_contract,
    output_contract: structure.output_contract,
  };
}

export function structureFingerprintText(structure) {
  return stableStringify(structureFingerprintCanonical(structure));
}

export async function buildStructureTelemetryPayload(analysis, context = {}) {
  const structure = buildStructureTelemetryDraft(analysis, context);
  structure.fingerprint = `sf_${(await sha256Hex(new TextEncoder().encode(structureFingerprintText(structure)))).slice(0, 64)}`;
  return structure;
}

export function buildBenchmarkRunTelemetry(result = {}, {
  targetId = "",
  browserBucket = "",
  preparedInput = false,
  runtimeStatus = "",
} = {}) {
  return {
    target: targetId,
    backend: result.backend,
    browser: browserBucket,
    compile_ms: result.compileMs,
    p50_ms: result.stats?.p50,
    p90_ms: result.stats?.p90,
    p95_ms: result.stats?.p95,
    p99_ms: result.stats?.p99,
    mean_ms: result.stats?.mean,
    warmup_count: result.warmup,
    run_count: result.runs,
    metadata: {
      output_count: result.outputCount,
      output_digest: result.outputDigest || "",
      first_run_ms: result.firstRunMs,
      timing_method: result.timingMethod || "",
      phase_counts: result.phaseCounts || null,
      steady_p50_ms: result.steadyStats?.p50,
      steady_mean_ms: result.steadyStats?.mean,
      steady_cv: result.steadyStats?.cv,
      min_ms: result.stats?.min,
      max_ms: result.stats?.max,
      stddev_ms: result.stats?.stddev,
      cv: result.stats?.cv,
      prepared_input: Boolean(preparedInput),
      runtime_status: runtimeStatus,
    },
  };
}

export function buildBenchmarkTelemetryPayload(structure, result = {}, context = {}) {
  return {
    consent: true,
    structure,
    run: buildBenchmarkRunTelemetry(result, context),
  };
}

export async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}
