// Numerical simulation only. Native model writers and execution are deliberately
// outside this module. Every position is evaluated; projections never sample.
import { compareValues, magnitude, tileProjection, kernelProjection } from "./numerical-ir/weight-math.js";
import { requireCondition, seal, SHA256 } from "./numerical-ir/common.js";
import { sha256BytesHex } from "./sha256-sync.js";

export const PRUNING_PREVIEW_SCHEMA = "deepbom.pruning_preview.v1";
const BOUNDARY = "Magnitude-mask simulation of one complete decoded tensor. Counts are exact; floating results are rounded binary64. Tensor dimensions are unchanged. Weight energy is not accuracy. No native model, sparse-backend eligibility, latency or task-quality result is produced.";
function accumulator() {
  let sum = 0, correction = 0;
  return { add(value) { const y = value - correction, t = sum + y; correction = (t - sum) - y; sum = t; }, get value() { return sum; } };
}
const removalCount = (p, percent) => Math.max(0, Math.floor(p.data.length * percent / 100) - p.zeroCount);
export function prepareMagnitudePruning({ matrix, row, source, weight_ir_sha256 }) {
  requireCondition(row?.status === "assessed", "pruning requires an assessed tensor");
  requireCondition(row.representation === "dequantized_real" || /^(?:B?FLOAT|BF|F)(?:8|16|32|64)(?:$|_)/.test(row.dtype), "pruning requires real values; integer codes without a proven dequantization contract cannot be ranked");
  requireCondition(SHA256.test(weight_ir_sha256 || "") && ["artifact_sha256", "model_ir_sha256"].every(k => SHA256.test(source?.[k] || "")) && (source.artifact_set_sha256 === null || SHA256.test(source.artifact_set_sha256 || "")), "pruning identity is incomplete");
  const { rows, columns } = matrix;
  requireCondition(Number.isSafeInteger(rows) && rows > 0 && rows <= 4096 && Number.isSafeInteger(columns) && columns > 0 && rows * columns === matrix.data.length && matrix.data.length <= 2_000_000, "pruning requires a complete matrix of at most 2,000,000 values and 4,096 channels");
  const data = Float64Array.from(matrix.data);
  requireCondition(data.every(Number.isFinite), "pruning requires finite values");
  const order = Uint32Array.from(data, (_, i) => i).filter(i => data[i] !== 0);
  order.sort((a, b) => Math.abs(data[a]) - Math.abs(data[b]) || a - b);
  const scale = magnitude(data), zeroCount = data.length - order.length;
  // Independent tails avoid cancellation when nearly all energy is removed.
  const retained = new Float64Array(order.length + 1), sum = accumulator();
  for (let i = order.length - 1; i >= 0; i--) { sum.add((data[order[i]] / scale) ** 2); retained[i] = sum.value; }
  const prepared = { data, rows, columns, order, scale, zeroCount, row, source, weight_ir_sha256 };
  prepared.curve = Array.from({ length: 101 }, (_, target) => {
    const k = removalCount(prepared, target);
    return { target_percent: target, zero_count: zeroCount + k, retained_energy_fraction: retained[0] ? retained[k] / retained[0] : null };
  });
  return prepared;
}

export function simulateMagnitudePruning(p, { target_percent = 50, channel = null } = {}) {
  requireCondition(Number.isInteger(target_percent) && target_percent >= 0 && target_percent <= 100, "pruning target must be an integer percentage from 0 to 100");
  requireCondition(channel === null || Number.isSafeInteger(channel) && channel >= 0 && channel < p.rows, "pruning channel is outside the tensor");
  const k = removalCount(p, target_percent), candidate = new Float64Array(p.data), mask = new Uint8Array(p.data.length);
  for (let i = 0; i < k; i++) { const position = p.order[i]; candidate[position] = 0; mask[position] = 1; }
  const channels = [];
  for (let row = 0; row < p.rows; row++) {
    const a = p.data.subarray(row * p.columns, (row + 1) * p.columns), b = candidate.subarray(row * p.columns, (row + 1) * p.columns);
    const scale = magnitude(a), original = accumulator(), kept = accumulator();
    let existing = 0, removed = 0;
    for (let j = 0; j < a.length; j++) {
      existing += Number(a[j] === 0); removed += mask[row * p.columns + j];
      if (scale) { original.add((a[j] / scale) ** 2); kept.add((b[j] / scale) ** 2); }
    }
    channels.push({ index: row, count: a.length, original_zero_count: existing, newly_zeroed_count: removed, candidate_zero_count: existing + removed, retained_energy_fraction: original.value ? kept.value / original.value : null });
  }
  const comparison = compareValues(p.data, candidate);
  const project = data => channel === null
    ? tileProjection({ data, rows: p.rows, columns: p.columns })
    : kernelProjection({ data, rows: p.rows, columns: p.columns }, p.row.shape, p.row.axes.channel_axis, p.row.axes.storage_order, channel).image;
  const body = {
    schema: PRUNING_PREVIEW_SCHEMA, source: p.source, weight_ir_sha256: p.weight_ir_sha256,
    weight_ref: p.row.weight_ref, dtype: p.row.dtype, representation: p.row.representation,
    shape: p.row.shape, axes: p.row.axes, method: "unstructured_smallest_absolute_value_mask",
    ordering: "channel_axis_unfolding_then_remaining_native_storage_order",
    tie_break: "ascending_unfolded_position", target_percent, target_zero_count: Math.floor(p.data.length * target_percent / 100),
    value_count: p.data.length, original_zero_count: p.zeroCount, newly_zeroed_count: k, candidate_zero_count: p.zeroCount + k,
    achieved_zero_fraction: (p.zeroCount + k) / p.data.length,
    retained_energy_fraction: p.curve[target_percent].retained_energy_fraction,
    last_removed_absolute_value: k ? Math.abs(p.data[p.order[k - 1]]) : null,
    mask_sha256: sha256BytesHex(mask), mask_encoding: "one_uint8_per_unfolded_position_1_newly_zeroed_0_unchanged",
    metrics: comparison.metrics, channels, curve: p.curve, selected_channel: channel, color_scale: p.scale,
    projections: { original: project(p.data), candidate: project(candidate), difference: project(comparison.differences), removed: project(Float64Array.from(mask)) },
    interpretation_boundary: BOUNDARY,
  };
  return seal(body, "pruning_preview_sha256");
}
