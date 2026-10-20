// Numerical simulation only. Native model writers and execution are deliberately
// outside this module. Every position is evaluated; projections never sample.
import { compareValues, tileProjection, kernelProjection } from "./numerical-ir/weight-math.js";
import { requireCondition, seal, SHA256 } from "./numerical-ir/common.js";
import { numericalWasm } from "./numerical-wasm.js";
import { sha256BytesHex } from "./sha256-sync.js";

export const PRUNING_PREVIEW_SCHEMA = "deepbom.pruning_preview.v1";
const BOUNDARY = "Magnitude-mask simulation of one complete decoded tensor. Counts are exact; floating results are rounded binary64. Tensor dimensions are unchanged. Weight energy is not accuracy. No native model, sparse-backend eligibility, latency or task-quality result is produced.";
export function prepareMagnitudePruning({ matrix, row, source, weight_ir_sha256 }) {
  requireCondition(row?.status === "assessed", "pruning requires an assessed tensor");
  requireCondition(row.representation === "dequantized_real" || /^(?:B?FLOAT|BF|F)(?:8|16|32|64)(?:$|_)/.test(row.dtype), "pruning requires real values; integer codes without a proven dequantization contract cannot be ranked");
  requireCondition(SHA256.test(weight_ir_sha256 || "") && ["artifact_sha256", "model_ir_sha256"].every(k => SHA256.test(source?.[k] || "")) && (source.artifact_set_sha256 === null || SHA256.test(source.artifact_set_sha256 || "")), "pruning identity is incomplete");
  const { rows, columns } = matrix;
  requireCondition(Number.isSafeInteger(rows) && rows > 0 && rows <= 4096 && Number.isSafeInteger(columns) && columns > 0 && rows * columns === matrix.data.length && matrix.data.length <= 2_000_000, "pruning requires a complete matrix of at most 2,000,000 values and 4,096 channels");
  const data = Float64Array.from(matrix.data);
  requireCondition(data.every(Number.isFinite), "pruning requires finite values");
  const prepared=numericalWasm().numerical_pruning_prepare(data,rows,columns);
  return {data,rows,columns,...prepared,row,source,weight_ir_sha256};
}

export function simulateMagnitudePruning(p, { target_percent = 50, channel = null } = {}) {
  requireCondition(Number.isInteger(target_percent) && target_percent >= 0 && target_percent <= 100, "pruning target must be an integer percentage from 0 to 100");
  requireCondition(channel === null || Number.isSafeInteger(channel) && channel >= 0 && channel < p.rows, "pruning channel is outside the tensor");
  const {k,candidate,mask,channels,...metrics}=numericalWasm().numerical_pruning_simulate(p.data,p.rows,p.columns,p.order,target_percent);
  const comparison = compareValues(p.data, candidate);
  const project = data => channel === null
    ? tileProjection({ data, rows: p.rows, columns: p.columns })
    : kernelProjection({ data, rows: p.rows, columns: p.columns }, p.row.shape, p.row.axes.channel_axis, p.row.axes.storage_order, channel).image;
  const body = {
    schema: PRUNING_PREVIEW_SCHEMA, source: p.source, weight_ir_sha256: p.weight_ir_sha256,
    weight_ref: p.row.weight_ref, dtype: p.row.dtype, representation: p.row.representation,
    shape: p.row.shape, axes: p.row.axes, method: "unstructured_smallest_absolute_value_mask",
    ordering: "channel_axis_unfolding_then_remaining_native_storage_order",
    tie_break: "ascending_unfolded_position", target_percent, target_zero_count: metrics.target_zero_count,
    value_count: p.data.length, original_zero_count: p.zeroCount, newly_zeroed_count: k, candidate_zero_count: p.zeroCount + k,
    achieved_zero_fraction: metrics.achieved_zero_fraction,
    retained_energy_fraction: p.curve[target_percent].retained_energy_fraction,
    last_removed_absolute_value: metrics.last_removed_absolute_value,
    mask_sha256: sha256BytesHex(mask), mask_encoding: "one_uint8_per_unfolded_position_1_newly_zeroed_0_unchanged",
    metrics: comparison.metrics, channels, curve: p.curve, selected_channel: channel, color_scale: p.scale,
    projections: { original: project(p.data), candidate: project(candidate), difference: project(comparison.differences), removed: project(Float64Array.from(mask)) },
    interpretation_boundary: BOUNDARY,
  };
  return seal(body, "pruning_preview_sha256");
}
