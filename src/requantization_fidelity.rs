use super::accumulator_atlas::{AccumulatorAtlasAnalysis, AccumulatorOpRow};
use super::quantization_math::quantize_multiplier;
use super::{OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const REQUANTIZATION_FIDELITY_SCHEMA: &str = "deepbom.requantization_fidelity.v1";
const REQUANTIZATION_FIDELITY_METHOD_VERSION: &str = "2026-07-17.2";
const TFLITE_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";

#[derive(Clone, Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Clone, Serialize)]
struct ShiftCount {
    shift: i32,
    channel_count: usize,
}

#[derive(Clone, Serialize)]
struct RequantizationChannelWitness {
    channel_index: usize,
    post_bias_accumulator_min_decimal: String,
    post_bias_accumulator_max_decimal: String,
    maximum_absolute_post_bias_accumulator_decimal: String,
    input_scale: f64,
    weight_scale: f64,
    output_scale: f64,
    real_multiplier: f64,
    quantized_multiplier: i32,
    shift: i32,
    represented_multiplier: f64,
    absolute_multiplier_error: f64,
    relative_multiplier_error: f64,
    multiplier_error_ppm: f64,
    encoding_drift_bound_codes: f64,
    encoding_drift_ceil_codes_decimal: String,
    default_double_rounding_bound_codes: Option<f64>,
    single_rounding_bound_codes: Option<f64>,
    default_pre_shift_int32_safe: bool,
    single_rounding_quantized_multiplier: i32,
    single_rounding_shift: i32,
    single_rounding_represented_multiplier: f64,
    single_rounding_encoding_diverges: bool,
}

#[derive(Serialize)]
pub(super) struct RequantizationOpRow {
    pub(super) op_index: usize,
    op_name: String,
    pub(super) assessment_status: &'static str,
    not_assessed_reason: String,
    input_tensor_index: Option<i32>,
    input_tensor_name: String,
    weight_tensor_index: Option<i32>,
    weight_tensor_name: String,
    output_tensor_index: Option<i32>,
    output_tensor_name: String,
    input_dtype: String,
    weight_dtype: String,
    pub(super) output_dtype: String,
    input_scale: Option<f64>,
    pub(super) output_scale: Option<f64>,
    pub(super) output_zero_point: Option<i64>,
    pub(super) output_code_range: Option<[i64; 2]>,
    weight_scale_mode: String,
    pub(super) assessed_channel_count: usize,
    fixed_point_bound_channel_count: usize,
    default_pre_shift_overflow_channel_count: usize,
    single_rounding_encoding_divergence_channel_count: usize,
    half_code_encoding_drift_channel_count: usize,
    one_code_encoding_drift_channel_count: usize,
    minimum_shift: Option<i32>,
    maximum_shift: Option<i32>,
    shift_histogram: Vec<ShiftCount>,
    maximum_relative_multiplier_error: Option<f64>,
    maximum_multiplier_error_ppm: Option<f64>,
    maximum_encoding_drift_bound_codes: Option<f64>,
    maximum_default_double_rounding_bound_codes: Option<f64>,
    maximum_single_rounding_bound_codes: Option<f64>,
    pub(super) channel_real_multipliers: Vec<f64>,
    pub(super) channel_quantized_multipliers: Vec<i32>,
    pub(super) channel_shifts: Vec<i32>,
    channel_represented_multipliers: Vec<f64>,
    channel_absolute_multiplier_errors: Vec<f64>,
    channel_relative_multiplier_errors: Vec<f64>,
    channel_encoding_drift_bound_codes: Vec<f64>,
    channel_default_double_rounding_bound_codes: Vec<Option<f64>>,
    channel_single_rounding_bound_codes: Vec<Option<f64>>,
    channel_default_pre_shift_int32_safe: Vec<bool>,
    pub(super) channel_single_rounding_quantized_multipliers: Vec<i32>,
    pub(super) channel_single_rounding_shifts: Vec<i32>,
    worst_channel: Option<RequantizationChannelWitness>,
    pub(super) channel_ledger_sha256: String,
    ledger_hash_method: &'static str,
}

impl RequantizationOpRow {
    fn not_assessed(op: &OpInfo, tensors: &[TensorInfo], reason: String) -> RequantizationOpRow {
        let input = op
            .inputs
            .first()
            .and_then(|index| tensor_at(tensors, *index));
        let weight = op
            .inputs
            .get(1)
            .and_then(|index| tensor_at(tensors, *index));
        let output = op
            .outputs
            .first()
            .and_then(|index| tensor_at(tensors, *index));
        RequantizationOpRow {
            op_index: op.index,
            op_name: op.name.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            input_tensor_index: op.inputs.first().copied(),
            input_tensor_name: input.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            weight_tensor_index: op.inputs.get(1).copied(),
            weight_tensor_name: weight.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            output_tensor_index: op.outputs.first().copied(),
            output_tensor_name: output.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            input_dtype: input.map(|tensor| tensor.dtype.clone()).unwrap_or_default(),
            weight_dtype: weight
                .map(|tensor| tensor.dtype.clone())
                .unwrap_or_default(),
            output_dtype: output
                .map(|tensor| tensor.dtype.clone())
                .unwrap_or_default(),
            input_scale: None,
            output_scale: None,
            output_zero_point: None,
            output_code_range: None,
            weight_scale_mode: String::new(),
            assessed_channel_count: 0,
            fixed_point_bound_channel_count: 0,
            default_pre_shift_overflow_channel_count: 0,
            single_rounding_encoding_divergence_channel_count: 0,
            half_code_encoding_drift_channel_count: 0,
            one_code_encoding_drift_channel_count: 0,
            minimum_shift: None,
            maximum_shift: None,
            shift_histogram: Vec::new(),
            maximum_relative_multiplier_error: None,
            maximum_multiplier_error_ppm: None,
            maximum_encoding_drift_bound_codes: None,
            maximum_default_double_rounding_bound_codes: None,
            maximum_single_rounding_bound_codes: None,
            channel_real_multipliers: Vec::new(),
            channel_quantized_multipliers: Vec::new(),
            channel_shifts: Vec::new(),
            channel_represented_multipliers: Vec::new(),
            channel_absolute_multiplier_errors: Vec::new(),
            channel_relative_multiplier_errors: Vec::new(),
            channel_encoding_drift_bound_codes: Vec::new(),
            channel_default_double_rounding_bound_codes: Vec::new(),
            channel_single_rounding_bound_codes: Vec::new(),
            channel_default_pre_shift_int32_safe: Vec::new(),
            channel_single_rounding_quantized_multipliers: Vec::new(),
            channel_single_rounding_shifts: Vec::new(),
            worst_channel: None,
            channel_ledger_sha256: String::new(),
            ledger_hash_method: ledger_hash_method(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct RequantizationFidelityAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    fixed_point_bound_channel_count: usize,
    per_tensor_weight_op_count: usize,
    per_axis_weight_op_count: usize,
    default_pre_shift_overflow_channel_count: usize,
    single_rounding_encoding_divergence_channel_count: usize,
    half_code_encoding_drift_channel_count: usize,
    one_code_encoding_drift_channel_count: usize,
    minimum_shift: Option<i32>,
    maximum_shift: Option<i32>,
    shift_histogram: Vec<ShiftCount>,
    maximum_relative_multiplier_error: Option<f64>,
    maximum_multiplier_error_ppm: Option<f64>,
    maximum_encoding_drift_bound_codes: Option<f64>,
    maximum_default_double_rounding_bound_codes: Option<f64>,
    maximum_single_rounding_bound_codes: Option<f64>,
    fidelity_ranking_op_indices: Vec<usize>,
    pub(super) ops: Vec<RequantizationOpRow>,
    source_commit: &'static str,
    source_references: Vec<SourceReference>,
    quantize_multiplier_formula: &'static str,
    encoding_drift_formula: &'static str,
    rounding_bound_formula: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn build_requantization_fidelity(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    atlas: &AccumulatorAtlasAnalysis,
) -> RequantizationFidelityAnalysis {
    let rows = atlas
        .ops
        .iter()
        .filter_map(|accumulator| {
            ops.iter()
                .find(|op| op.index == accumulator.op_index)
                .map(|op| assess_op(op, tensors, accumulator))
        })
        .collect::<Vec<_>>();
    let assessed_positions = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.assessment_status == "assessed")
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let mut ranked_positions = assessed_positions.clone();
    ranked_positions.sort_by(|left, right| {
        cmp_f64_desc(
            rows[*left]
                .maximum_encoding_drift_bound_codes
                .unwrap_or(0.0),
            rows[*right]
                .maximum_encoding_drift_bound_codes
                .unwrap_or(0.0),
        )
        .then_with(|| {
            cmp_f64_desc(
                rows[*left].maximum_relative_multiplier_error.unwrap_or(0.0),
                rows[*right]
                    .maximum_relative_multiplier_error
                    .unwrap_or(0.0),
            )
        })
        .then_with(|| rows[*left].op_index.cmp(&rows[*right].op_index))
    });
    let candidate_op_count = rows.len();
    let assessed_op_count = assessed_positions.len();
    let unassessed_op_count = candidate_op_count.saturating_sub(assessed_op_count);
    let mut global_shifts = BTreeMap::<i32, usize>::new();
    for position in &assessed_positions {
        for bin in &rows[*position].shift_histogram {
            *global_shifts.entry(bin.shift).or_default() += bin.channel_count;
        }
    }
    let status = if candidate_op_count == 0 {
        "not_applicable"
    } else if unassessed_op_count > 0 {
        "partial"
    } else {
        "assessed"
    };
    RequantizationFidelityAnalysis {
        schema: REQUANTIZATION_FIDELITY_SCHEMA,
        method_version: REQUANTIZATION_FIDELITY_METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        candidate_op_count,
        assessed_op_count,
        unassessed_op_count,
        assessed_channel_count: sum_rows(&rows, &assessed_positions, |row| row.assessed_channel_count),
        fixed_point_bound_channel_count: sum_rows(&rows, &assessed_positions, |row| row.fixed_point_bound_channel_count),
        per_tensor_weight_op_count: assessed_positions
            .iter()
            .filter(|position| rows[**position].weight_scale_mode == "per_tensor")
            .count(),
        per_axis_weight_op_count: assessed_positions
            .iter()
            .filter(|position| rows[**position].weight_scale_mode == "per_output_channel")
            .count(),
        default_pre_shift_overflow_channel_count: sum_rows(&rows, &assessed_positions, |row| row.default_pre_shift_overflow_channel_count),
        single_rounding_encoding_divergence_channel_count: sum_rows(&rows, &assessed_positions, |row| row.single_rounding_encoding_divergence_channel_count),
        half_code_encoding_drift_channel_count: sum_rows(&rows, &assessed_positions, |row| row.half_code_encoding_drift_channel_count),
        one_code_encoding_drift_channel_count: sum_rows(&rows, &assessed_positions, |row| row.one_code_encoding_drift_channel_count),
        minimum_shift: assessed_positions.iter().filter_map(|position| rows[*position].minimum_shift).min(),
        maximum_shift: assessed_positions.iter().filter_map(|position| rows[*position].maximum_shift).max(),
        shift_histogram: global_shifts
            .into_iter()
            .map(|(shift, channel_count)| ShiftCount { shift, channel_count })
            .collect(),
        maximum_relative_multiplier_error: max_row_metric(&rows, &assessed_positions, |row| row.maximum_relative_multiplier_error),
        maximum_multiplier_error_ppm: max_row_metric(&rows, &assessed_positions, |row| row.maximum_multiplier_error_ppm),
        maximum_encoding_drift_bound_codes: max_row_metric(&rows, &assessed_positions, |row| row.maximum_encoding_drift_bound_codes),
        maximum_default_double_rounding_bound_codes: max_row_metric(&rows, &assessed_positions, |row| row.maximum_default_double_rounding_bound_codes),
        maximum_single_rounding_bound_codes: max_row_metric(&rows, &assessed_positions, |row| row.maximum_single_rounding_bound_codes),
        fidelity_ranking_op_indices: ranked_positions.iter().map(|position| rows[*position].op_index).collect(),
        ops: rows,
        source_commit: TFLITE_SOURCE_COMMIT,
        source_references: source_references(),
        quantize_multiplier_formula: "real=(input_scale*weight_scale[channel])/output_scale; q=frexp(real,&shift); q_fixed=round(q*2^31); if q_fixed==2^31 then q_fixed/=2 and shift+=1; shift<-31 flushes to q_fixed=0,shift=0; TFLITE_SINGLE_ROUNDING additionally saturates shift>30 to q_fixed=2^31-1,shift=30",
        encoding_drift_formula: "represented=q_fixed*2^(shift-31); encoding_drift_codes=max_abs(post_bias_accumulator)*abs(real-represented)",
        rounding_bound_formula: "default double-rounding pre-clamp error <= encoding_drift + (shift<0 ? 0.5 + 2^(shift-1) : 0.5), provided the INT32 accumulator and positive pre-shift fit; single-rounding error <= encoding_drift(single encoding) + 0.5",
        method: "For each exactly bounded quantized TFLite Conv/Depthwise/FC output channel, derive the effective output scale from the artifact's full float32 quantization metadata, reproduce the pinned TensorFlow QuantizeMultiplier Q0.31 encoding for default and TFLITE_SINGLE_ROUNDING builds, and propagate encoding and rounding bounds over the exact stored-weight post-bias accumulator domain consumed by requantization. Preserve every channel multiplier, shift, error bound, build-mode divergence, and canonical ledger digest.",
        interpretation_boundary: "This is a source-derived quantization-parameter and conservative pre-clamp arithmetic analysis. The Accumulator Atlas full envelope remains authoritative for intermediate INT32 safety; this analysis uses its post-bias domain because that is the input to requantization. It does not identify the runtime's TFLITE_SINGLE_ROUNDING compile flag, prove that a delegate uses the reference integer path, count actual output mismatches, model activation frequency, or estimate model accuracy. The double-rounding bound assumes the full accumulator path fits INT32 and separately reports any positive-shift pre-multiply overflow over the post-bias domain. Runtime/build evidence remains authoritative for the executed rounding path.",
    }
}

fn assess_op(
    op: &OpInfo,
    tensors: &[TensorInfo],
    accumulator: &AccumulatorOpRow,
) -> RequantizationOpRow {
    if accumulator.assessment_status != "assessed" {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Accumulator atlas row is {}.",
                accumulator.assessment_status
            ),
        );
    }
    let Some(input) = op
        .inputs
        .first()
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Input tensor is unavailable.".to_string(),
        );
    };
    let Some(weight) = op
        .inputs
        .get(1)
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Weight tensor is unavailable.".to_string(),
        );
    };
    let Some(output) = op
        .outputs
        .first()
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Output tensor is unavailable.".to_string(),
        );
    };
    let channels = accumulator.output_channel_count.unwrap_or(0);
    if channels == 0 || accumulator.assessed_channel_count != channels {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Accumulator output-channel cardinality is unavailable.".to_string(),
        );
    }
    if input.scale_sample.len() != 1 || output.scale_sample.len() != 1 {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Input and output require one per-tensor scale.".to_string(),
        );
    }
    let input_scale = input.scale_sample[0] as f64;
    let output_scale = output.scale_sample[0] as f64;
    if !valid_scale(input_scale) || !valid_scale(output_scale) {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Input or output scale is non-finite or non-positive.".to_string(),
        );
    }
    if !(weight.scale_sample.len() == 1 || weight.scale_sample.len() == channels)
        || weight
            .scale_sample
            .iter()
            .any(|scale| !valid_scale(*scale as f64))
    {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            format!("Weight scale cardinality must be 1 or {channels}."),
        );
    }
    let Some(output_code_range) = quantized_code_range(&output.dtype) else {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            format!("Output dtype {} is not INT8 or UINT8.", output.dtype),
        );
    };
    if output.zero_point_sample.len() != 1
        || output.zero_point_sample[0] < output_code_range[0]
        || output.zero_point_sample[0] > output_code_range[1]
    {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Output requires one in-range zero-point.".to_string(),
        );
    }
    if accumulator.channel_accumulator_envelope_min_decimals.len() != channels
        || accumulator.channel_accumulator_envelope_max_decimals.len() != channels
        || accumulator.channel_post_bias_min_decimals.len() != channels
        || accumulator.channel_post_bias_max_decimals.len() != channels
    {
        return RequantizationOpRow::not_assessed(
            op,
            tensors,
            "Accumulator channel arrays do not match output-channel count.".to_string(),
        );
    }

    let mut witnesses = Vec::with_capacity(channels);
    let mut real_multipliers = Vec::with_capacity(channels);
    let mut quantized_multipliers = Vec::with_capacity(channels);
    let mut shifts = Vec::with_capacity(channels);
    let mut represented_multipliers = Vec::with_capacity(channels);
    let mut absolute_errors = Vec::with_capacity(channels);
    let mut relative_errors = Vec::with_capacity(channels);
    let mut encoding_drifts = Vec::with_capacity(channels);
    let mut default_bounds = Vec::with_capacity(channels);
    let mut single_bounds = Vec::with_capacity(channels);
    let mut pre_shift_safety = Vec::with_capacity(channels);
    let mut single_multipliers = Vec::with_capacity(channels);
    let mut single_shifts = Vec::with_capacity(channels);
    let mut shift_counts = BTreeMap::<i32, usize>::new();
    let mut ledger = Sha256::new();

    for channel in 0..channels {
        let weight_scale = weight.scale_sample[if weight.scale_sample.len() == 1 {
            0
        } else {
            channel
        }] as f64;
        let real_multiplier = input_scale * weight_scale / output_scale;
        if !valid_scale(real_multiplier) {
            return RequantizationOpRow::not_assessed(
                op,
                tensors,
                format!("Channel {channel} effective multiplier is invalid."),
            );
        }
        let default_encoding = quantize_multiplier(real_multiplier, false);
        let single_encoding = quantize_multiplier(real_multiplier, true);
        let envelope_minimum =
            match accumulator.channel_accumulator_envelope_min_decimals[channel].parse::<i128>() {
                Ok(value) => value,
                Err(_) => {
                    return RequantizationOpRow::not_assessed(
                        op,
                        tensors,
                        format!("Channel {channel} accumulator minimum is invalid."),
                    )
                }
            };
        let envelope_maximum =
            match accumulator.channel_accumulator_envelope_max_decimals[channel].parse::<i128>() {
                Ok(value) => value,
                Err(_) => {
                    return RequantizationOpRow::not_assessed(
                        op,
                        tensors,
                        format!("Channel {channel} accumulator maximum is invalid."),
                    )
                }
            };
        let post_bias_minimum =
            match accumulator.channel_post_bias_min_decimals[channel].parse::<i128>() {
                Ok(value) => value,
                Err(_) => {
                    return RequantizationOpRow::not_assessed(
                        op,
                        tensors,
                        format!("Channel {channel} post-bias accumulator minimum is invalid."),
                    )
                }
            };
        let post_bias_maximum =
            match accumulator.channel_post_bias_max_decimals[channel].parse::<i128>() {
                Ok(value) => value,
                Err(_) => {
                    return RequantizationOpRow::not_assessed(
                        op,
                        tensors,
                        format!("Channel {channel} post-bias accumulator maximum is invalid."),
                    )
                }
            };
        let maximum_absolute = post_bias_minimum.abs().max(post_bias_maximum.abs());
        let absolute_error = (real_multiplier - default_encoding.represented).abs();
        let relative_error = absolute_error / real_multiplier;
        let encoding_drift = maximum_absolute as f64 * absolute_error;
        let single_encoding_drift =
            maximum_absolute as f64 * (real_multiplier - single_encoding.represented).abs();
        let accumulator_fits_int32 =
            envelope_minimum >= i32::MIN as i128 && envelope_maximum <= i32::MAX as i128;
        let default_pre_shift_safe = accumulator_fits_int32
            && pre_shift_fits_int32(post_bias_minimum, post_bias_maximum, default_encoding.shift);
        let default_bound = default_pre_shift_safe
            .then_some(encoding_drift + default_rounding_only_bound(default_encoding.shift));
        let single_bound = accumulator_fits_int32.then_some(single_encoding_drift + 0.5);
        let single_diverges = default_encoding.multiplier != single_encoding.multiplier
            || default_encoding.shift != single_encoding.shift;
        let witness = RequantizationChannelWitness {
            channel_index: channel,
            post_bias_accumulator_min_decimal: post_bias_minimum.to_string(),
            post_bias_accumulator_max_decimal: post_bias_maximum.to_string(),
            maximum_absolute_post_bias_accumulator_decimal: maximum_absolute.to_string(),
            input_scale,
            weight_scale,
            output_scale,
            real_multiplier,
            quantized_multiplier: default_encoding.multiplier,
            shift: default_encoding.shift,
            represented_multiplier: default_encoding.represented,
            absolute_multiplier_error: absolute_error,
            relative_multiplier_error: relative_error,
            multiplier_error_ppm: relative_error * 1_000_000.0,
            encoding_drift_bound_codes: encoding_drift,
            encoding_drift_ceil_codes_decimal: ceil_decimal(encoding_drift),
            default_double_rounding_bound_codes: default_bound,
            single_rounding_bound_codes: single_bound,
            default_pre_shift_int32_safe: default_pre_shift_safe,
            single_rounding_quantized_multiplier: single_encoding.multiplier,
            single_rounding_shift: single_encoding.shift,
            single_rounding_represented_multiplier: single_encoding.represented,
            single_rounding_encoding_diverges: single_diverges,
        };
        ledger.update(ledger_row(op.index, &witness).as_bytes());
        *shift_counts.entry(default_encoding.shift).or_default() += 1;
        real_multipliers.push(real_multiplier);
        quantized_multipliers.push(default_encoding.multiplier);
        shifts.push(default_encoding.shift);
        represented_multipliers.push(default_encoding.represented);
        absolute_errors.push(absolute_error);
        relative_errors.push(relative_error);
        encoding_drifts.push(encoding_drift);
        default_bounds.push(default_bound);
        single_bounds.push(single_bound);
        pre_shift_safety.push(default_pre_shift_safe);
        single_multipliers.push(single_encoding.multiplier);
        single_shifts.push(single_encoding.shift);
        witnesses.push(witness);
    }
    witnesses.sort_by(|left, right| {
        cmp_f64_desc(
            left.encoding_drift_bound_codes,
            right.encoding_drift_bound_codes,
        )
        .then_with(|| {
            cmp_f64_desc(
                left.relative_multiplier_error,
                right.relative_multiplier_error,
            )
        })
        .then_with(|| left.channel_index.cmp(&right.channel_index))
    });
    let worst_channel = witnesses.first().cloned();
    RequantizationOpRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_tensor_index: Some(input.index as i32),
        input_tensor_name: input.name.clone(),
        weight_tensor_index: Some(weight.index as i32),
        weight_tensor_name: weight.name.clone(),
        output_tensor_index: Some(output.index as i32),
        output_tensor_name: output.name.clone(),
        input_dtype: input.dtype.clone(),
        weight_dtype: weight.dtype.clone(),
        output_dtype: output.dtype.clone(),
        input_scale: Some(input_scale),
        output_scale: Some(output_scale),
        output_zero_point: Some(output.zero_point_sample[0]),
        output_code_range: Some(output_code_range),
        weight_scale_mode: if weight.scale_sample.len() == 1 {
            "per_tensor"
        } else {
            "per_output_channel"
        }
        .to_string(),
        assessed_channel_count: channels,
        fixed_point_bound_channel_count: default_bounds
            .iter()
            .filter(|value| value.is_some())
            .count(),
        default_pre_shift_overflow_channel_count: pre_shift_safety
            .iter()
            .filter(|safe| !**safe)
            .count(),
        single_rounding_encoding_divergence_channel_count: witnesses
            .iter()
            .filter(|witness| witness.single_rounding_encoding_diverges)
            .count(),
        half_code_encoding_drift_channel_count: encoding_drifts
            .iter()
            .filter(|value| **value >= 0.5)
            .count(),
        one_code_encoding_drift_channel_count: encoding_drifts
            .iter()
            .filter(|value| **value >= 1.0)
            .count(),
        minimum_shift: shifts.iter().copied().min(),
        maximum_shift: shifts.iter().copied().max(),
        shift_histogram: shift_counts
            .into_iter()
            .map(|(shift, channel_count)| ShiftCount {
                shift,
                channel_count,
            })
            .collect(),
        maximum_relative_multiplier_error: max_f64(&relative_errors),
        maximum_multiplier_error_ppm: max_f64(&relative_errors).map(|value| value * 1_000_000.0),
        maximum_encoding_drift_bound_codes: max_f64(&encoding_drifts),
        maximum_default_double_rounding_bound_codes: max_optional_f64(&default_bounds),
        maximum_single_rounding_bound_codes: max_optional_f64(&single_bounds),
        channel_real_multipliers: real_multipliers,
        channel_quantized_multipliers: quantized_multipliers,
        channel_shifts: shifts,
        channel_represented_multipliers: represented_multipliers,
        channel_absolute_multiplier_errors: absolute_errors,
        channel_relative_multiplier_errors: relative_errors,
        channel_encoding_drift_bound_codes: encoding_drifts,
        channel_default_double_rounding_bound_codes: default_bounds,
        channel_single_rounding_bound_codes: single_bounds,
        channel_default_pre_shift_int32_safe: pre_shift_safety,
        channel_single_rounding_quantized_multipliers: single_multipliers,
        channel_single_rounding_shifts: single_shifts,
        worst_channel,
        channel_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: ledger_hash_method(),
    }
}

fn pre_shift_fits_int32(minimum: i128, maximum: i128, shift: i32) -> bool {
    if shift <= 0 {
        return true;
    }
    let Some(factor) = 1i128.checked_shl(shift as u32) else {
        return false;
    };
    minimum
        .checked_mul(factor)
        .zip(maximum.checked_mul(factor))
        .is_some_and(|(minimum, maximum)| {
            minimum >= i32::MIN as i128 && maximum <= i32::MAX as i128
        })
}

fn default_rounding_only_bound(shift: i32) -> f64 {
    if shift < 0 {
        0.5 + 2.0_f64.powi(shift - 1)
    } else {
        0.5
    }
}

fn ledger_row(op_index: usize, witness: &RequantizationChannelWitness) -> String {
    format!(
        "op={};channel={};real={};q={};shift={};represented={};abs_error={};relative_error={};encoding_drift={};default_bound={};pre_shift_safe={};single_q={};single_shift={};single_represented={};single_bound={};single_diverges={}\n",
        op_index,
        witness.channel_index,
        f64_bits(witness.real_multiplier),
        witness.quantized_multiplier,
        witness.shift,
        f64_bits(witness.represented_multiplier),
        f64_bits(witness.absolute_multiplier_error),
        f64_bits(witness.relative_multiplier_error),
        f64_bits(witness.encoding_drift_bound_codes),
        optional_f64_bits(witness.default_double_rounding_bound_codes),
        bool_digit(witness.default_pre_shift_int32_safe),
        witness.single_rounding_quantized_multiplier,
        witness.single_rounding_shift,
        f64_bits(witness.single_rounding_represented_multiplier),
        optional_f64_bits(witness.single_rounding_bound_codes),
        bool_digit(witness.single_rounding_encoding_diverges),
    )
}

fn ledger_hash_method() -> &'static str {
    "SHA-256 over UTF-8 rows with IEEE-754 binary64 fields encoded as 16 lowercase hexadecimal bits: op=<index>;channel=<index>;real=<bits>;q=<i32>;shift=<i32>;represented=<bits>;abs_error=<bits>;relative_error=<bits>;encoding_drift=<bits>;default_bound=<bits|na>;pre_shift_safe=<0|1>;single_q=<i32>;single_shift=<i32>;single_represented=<bits>;single_bound=<bits|na>;single_diverges=<0|1>\\n"
}

fn f64_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn optional_f64_bits(value: Option<f64>) -> String {
    value.map(f64_bits).unwrap_or_else(|| "na".to_string())
}

fn bool_digit(value: bool) -> usize {
    usize::from(value)
}

fn ceil_decimal(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        return "not_assessed".to_string();
    }
    format!("{:.0}", value.ceil())
}

fn valid_scale(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn quantized_code_range(dtype: &str) -> Option<[i64; 2]> {
    match dtype {
        "INT8" => Some([-128, 127]),
        "UINT8" => Some([0, 255]),
        _ => None,
    }
}

fn tensor_at(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    usize::try_from(index)
        .ok()
        .and_then(|position| tensors.get(position))
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn max_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}

fn max_optional_f64(values: &[Option<f64>]) -> Option<f64> {
    values.iter().filter_map(|value| *value).reduce(f64::max)
}

fn sum_rows<F>(rows: &[RequantizationOpRow], positions: &[usize], field: F) -> usize
where
    F: Fn(&RequantizationOpRow) -> usize,
{
    positions
        .iter()
        .map(|position| field(&rows[*position]))
        .sum()
}

fn max_row_metric<F>(rows: &[RequantizationOpRow], positions: &[usize], field: F) -> Option<f64>
where
    F: Fn(&RequantizationOpRow) -> Option<f64>,
{
    positions
        .iter()
        .filter_map(|position| field(&rows[*position]))
        .reduce(f64::max)
}

fn source_references() -> Vec<SourceReference> {
    vec![
        source_reference(
            "multiplier_encoding",
            "tensorflow/lite/kernels/internal/quantization_util.cc",
            "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
        ),
        source_reference(
            "effective_scale_preparation",
            "tensorflow/lite/kernels/kernel_util.cc",
            "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
        ),
        source_reference(
            "fixed_point_execution",
            "tensorflow/lite/kernels/internal/common.cc",
            "ba5308bf76383d600d033c948fe0659710939e6f1f15a800b5413e5fc822ddfa",
        ),
    ]
}

fn source_reference(
    role: &'static str,
    file: &'static str,
    sha256: &'static str,
) -> SourceReference {
    SourceReference {
        role,
        file,
        url: format!("https://github.com/tensorflow/tensorflow/blob/{TFLITE_SOURCE_COMMIT}/{file}"),
        sha256,
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_encoding_matches_q31_contract() {
        let quarter = quantize_multiplier(0.25, false);
        assert_eq!(quarter.multiplier, 1 << 30);
        assert_eq!(quarter.shift, -1);
        assert_eq!(quarter.represented, 0.25);

        let tenth = quantize_multiplier(0.1, false);
        assert_eq!(tenth.multiplier, 1_717_986_918);
        assert_eq!(tenth.shift, -3);
        assert_eq!(
            tenth.represented.to_bits(),
            (1_717_986_918_f64 * 2.0_f64.powi(-34)).to_bits()
        );

        let flushed = quantize_multiplier(2.0_f64.powi(-40), false);
        assert_eq!(flushed.multiplier, 0);
        assert_eq!(flushed.shift, 0);

        let default_large = quantize_multiplier(2.0_f64.powi(40), false);
        let single_large = quantize_multiplier(2.0_f64.powi(40), true);
        assert_eq!(default_large.shift, 41);
        assert_eq!(single_large.multiplier, i32::MAX);
        assert_eq!(single_large.shift, 30);
    }

    #[test]
    fn rounding_bounds_and_positive_pre_shift_are_fail_closed() {
        assert_eq!(default_rounding_only_bound(-1), 0.75);
        assert_eq!(default_rounding_only_bound(-5), 0.515625);
        assert_eq!(default_rounding_only_bound(0), 0.5);
        assert_eq!(default_rounding_only_bound(7), 0.5);
        assert!(pre_shift_fits_int32(-100, 100, 23));
        assert!(!pre_shift_fits_int32(-100, 100, 25));
        assert!(pre_shift_fits_int32(i32::MIN as i128, i32::MAX as i128, -1));
        assert!(!pre_shift_fits_int32(i32::MIN as i128, i32::MAX as i128, 1));
    }

    #[test]
    fn quantized_mobilenet_requantization_is_channel_complete() {
        let analysis = super::super::analyze_with_target(
            include_bytes!("../web/samples/mobilenet_v2_1.0_224_quant.tflite"),
            "mobilenet_v2_1.0_224_quant.tflite",
            "android_mid_a55",
        )
        .expect("sample should parse");
        let fidelity = analysis.requantization_fidelity;
        assert_eq!(fidelity.schema, REQUANTIZATION_FIDELITY_SCHEMA);
        assert_eq!(fidelity.candidate_op_count, 53);
        assert_eq!(fidelity.assessed_op_count, 53);
        assert_eq!(fidelity.unassessed_op_count, 0);
        assert_eq!(fidelity.assessed_channel_count, 18_057);
        assert_eq!(fidelity.fixed_point_bound_channel_count, 18_057);
        assert_eq!(fidelity.default_pre_shift_overflow_channel_count, 0);
        assert_eq!(
            fidelity.single_rounding_encoding_divergence_channel_count,
            0
        );
        assert_eq!(fidelity.half_code_encoding_drift_channel_count, 0);
        assert_eq!(fidelity.one_code_encoding_drift_channel_count, 0);
        assert_eq!(fidelity.minimum_shift, Some(-11));
        assert_eq!(fidelity.maximum_shift, Some(-1));
        assert_eq!(
            fidelity
                .shift_histogram
                .iter()
                .map(|bin| bin.channel_count)
                .sum::<usize>(),
            fidelity.assessed_channel_count
        );
        assert!(fidelity.maximum_encoding_drift_bound_codes.unwrap_or(1.0) < 0.000_005);
        for row in fidelity.ops {
            assert_eq!(row.assessment_status, "assessed");
            assert_eq!(
                row.channel_real_multipliers.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.channel_quantized_multipliers.len(),
                row.assessed_channel_count
            );
            assert_eq!(row.channel_shifts.len(), row.assessed_channel_count);
            assert_eq!(
                row.channel_encoding_drift_bound_codes.len(),
                row.assessed_channel_count
            );
            assert_eq!(row.channel_ledger_sha256.len(), 64);
        }
    }
}
