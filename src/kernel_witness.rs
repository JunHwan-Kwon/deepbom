use super::accumulator_atlas::{
    decode_bias, expanded_weight_zero_points, quantized_code_range, raw_code, weight_layout,
    AccumulatorAtlasAnalysis,
};
use super::quantization_math::{
    multiply_by_quantized_multiplier_default, multiply_by_quantized_multiplier_single_rounding,
    round_ties_away_from_zero,
};
use super::requantization_fidelity::{RequantizationFidelityAnalysis, RequantizationOpRow};
use super::{extract_tensor_buffer, OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const KERNEL_WITNESS_SCHEMA: &str = "deepbom.kernel_extremum_witness.v1";
const KERNEL_WITNESS_METHOD_VERSION: &str = "2026-07-17.1";
const TFLITE_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const TOP_WITNESS_LIMIT: usize = 8;
const LEDGER_PREFIX: &[u8] = b"deepbom.kernel_extremum_witness.v1\0";
const PATTERN_PREFIX: &[u8] = b"deepbom.kernel_extremum_pattern.v1\0";
const MISSING_I64: i64 = i64::MIN;

#[derive(Clone, Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Clone, Serialize)]
struct CodeCount {
    code: i64,
    count: usize,
}

#[derive(Clone, Serialize)]
struct EndpointProjection {
    endpoint: &'static str,
    witness_code_histogram: Vec<CodeCount>,
    dot_product_decimal: String,
    bias_decimal: String,
    post_bias_accumulator_decimal: String,
    ideal_preclamp_code: Option<i64>,
    ideal_output_code: Option<i64>,
    default_scaled_accumulator: Option<i64>,
    default_preclamp_code: Option<i64>,
    default_output_code: Option<i64>,
    default_activation_clamped: Option<bool>,
    default_ideal_delta_codes: Option<i64>,
    single_scaled_accumulator: Option<i64>,
    single_preclamp_code: Option<i64>,
    single_output_code: Option<i64>,
    single_activation_clamped: Option<bool>,
    single_ideal_delta_codes: Option<i64>,
    build_mode_output_delta_codes: Option<i64>,
}

struct EndpointWitnessInput {
    endpoint: &'static str,
    histogram: BTreeMap<i64, usize>,
    dot_product: i128,
    bias: i128,
    post_bias: i128,
}

struct ChannelProjectionContext<'a> {
    requant: &'a RequantizationOpRow,
    channel: usize,
    output_zero_point: i64,
    activation_range: [i64; 2],
}

pub(super) struct KernelChannelOutcome {
    pub(super) channel_index: usize,
    pub(super) post_bias_minimum: i128,
    pub(super) post_bias_maximum: i128,
    pub(super) default_minimum_preclamp_code: Option<i64>,
    pub(super) default_maximum_preclamp_code: Option<i64>,
    pub(super) default_minimum_output_code: Option<i64>,
    pub(super) default_maximum_output_code: Option<i64>,
    pub(super) single_minimum_preclamp_code: Option<i64>,
    pub(super) single_maximum_preclamp_code: Option<i64>,
    pub(super) single_minimum_output_code: Option<i64>,
    pub(super) single_maximum_output_code: Option<i64>,
}

#[derive(Clone, Serialize)]
struct KernelChannelWitness {
    channel_index: usize,
    term_count: usize,
    positive_centered_weight_count: usize,
    negative_centered_weight_count: usize,
    zero_centered_weight_count: usize,
    maximum_absolute_centered_weight: i64,
    maximum_absolute_term_contribution: i64,
    minimum: EndpointProjection,
    maximum: EndpointProjection,
    maximum_default_ideal_delta_codes: Option<i64>,
    maximum_single_ideal_delta_codes: Option<i64>,
    build_mode_divergent_endpoint_count: usize,
    witness_pattern_sha256: String,
}

#[derive(Serialize)]
pub(super) struct KernelWitnessOpRow {
    pub(super) op_index: usize,
    pub(super) op_name: String,
    pub(super) assessment_status: &'static str,
    pub(super) not_assessed_reason: String,
    input_tensor_index: Option<i32>,
    input_tensor_name: String,
    weight_tensor_index: Option<i32>,
    weight_tensor_name: String,
    output_tensor_index: Option<i32>,
    output_tensor_name: String,
    input_dtype: String,
    input_code_range: Option<[i64; 2]>,
    input_zero_point: Option<i64>,
    weight_dtype: String,
    weight_shape: Vec<i32>,
    output_channel_axis: Option<usize>,
    output_dtype: String,
    output_scale: Option<f64>,
    output_zero_point: Option<i64>,
    pub(super) output_code_range: Option<[i64; 2]>,
    fused_activation: String,
    pub(super) activation_code_range: Option<[i64; 2]>,
    pub(super) assessed_channel_count: usize,
    fixed_point_assessed_channel_count: usize,
    pub(super) accumulation_terms_per_channel: Option<usize>,
    witness_assignment_count: usize,
    fixed_point_endpoint_evaluation_count: usize,
    default_ideal_mismatch_endpoint_count: usize,
    single_ideal_mismatch_endpoint_count: usize,
    build_mode_divergent_endpoint_count: usize,
    default_activation_clamped_endpoint_count: usize,
    single_activation_clamped_endpoint_count: usize,
    default_collapsed_extrema_channel_count: usize,
    single_collapsed_extrema_channel_count: usize,
    maximum_default_ideal_delta_codes: Option<i64>,
    maximum_single_ideal_delta_codes: Option<i64>,
    top_channels: Vec<KernelChannelWitness>,
    worst_channel: Option<KernelChannelWitness>,
    pub(super) witness_ledger_sha256: String,
    ledger_hash_method: &'static str,
    #[serde(skip)]
    pub(super) channel_outcomes: Vec<KernelChannelOutcome>,
}

impl KernelWitnessOpRow {
    fn not_assessed(op: &OpInfo, tensors: &[TensorInfo], reason: String) -> Self {
        let input = tensor_for_input(tensors, op.inputs.first().copied());
        let weight = tensor_for_input(tensors, op.inputs.get(1).copied());
        let output = tensor_for_input(tensors, op.outputs.first().copied());
        Self {
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
            input_code_range: None,
            input_zero_point: None,
            weight_dtype: weight
                .map(|tensor| tensor.dtype.clone())
                .unwrap_or_default(),
            weight_shape: weight
                .map(|tensor| tensor.shape.clone())
                .unwrap_or_default(),
            output_channel_axis: None,
            output_dtype: output
                .map(|tensor| tensor.dtype.clone())
                .unwrap_or_default(),
            output_scale: None,
            output_zero_point: None,
            output_code_range: None,
            fused_activation: op.fused_activation.clone(),
            activation_code_range: None,
            assessed_channel_count: 0,
            fixed_point_assessed_channel_count: 0,
            accumulation_terms_per_channel: None,
            witness_assignment_count: 0,
            fixed_point_endpoint_evaluation_count: 0,
            default_ideal_mismatch_endpoint_count: 0,
            single_ideal_mismatch_endpoint_count: 0,
            build_mode_divergent_endpoint_count: 0,
            default_activation_clamped_endpoint_count: 0,
            single_activation_clamped_endpoint_count: 0,
            default_collapsed_extrema_channel_count: 0,
            single_collapsed_extrema_channel_count: 0,
            maximum_default_ideal_delta_codes: None,
            maximum_single_ideal_delta_codes: None,
            top_channels: Vec::new(),
            worst_channel: None,
            witness_ledger_sha256: String::new(),
            ledger_hash_method: ledger_hash_method(),
            channel_outcomes: Vec::new(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct KernelWitnessAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    fixed_point_assessed_channel_count: usize,
    witness_assignment_count: usize,
    fixed_point_endpoint_evaluation_count: usize,
    default_ideal_mismatch_endpoint_count: usize,
    single_ideal_mismatch_endpoint_count: usize,
    build_mode_divergent_endpoint_count: usize,
    default_activation_clamped_endpoint_count: usize,
    single_activation_clamped_endpoint_count: usize,
    default_collapsed_extrema_channel_count: usize,
    single_collapsed_extrema_channel_count: usize,
    maximum_default_ideal_delta_codes: Option<i64>,
    maximum_single_ideal_delta_codes: Option<i64>,
    witness_ranking_op_indices: Vec<usize>,
    pub(super) ops: Vec<KernelWitnessOpRow>,
    source_commit: &'static str,
    source_references: Vec<SourceReference>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn kernel_witness_not_computed() -> KernelWitnessAnalysis {
    KernelWitnessAnalysis {
        schema: KERNEL_WITNESS_SCHEMA,
        method_version: KERNEL_WITNESS_METHOD_VERSION,
        evidence_class: "NOT_ASSESSABLE",
        status: "not_computed_internal_scope",
        candidate_op_count: 0,
        assessed_op_count: 0,
        unassessed_op_count: 0,
        assessed_channel_count: 0,
        fixed_point_assessed_channel_count: 0,
        witness_assignment_count: 0,
        fixed_point_endpoint_evaluation_count: 0,
        default_ideal_mismatch_endpoint_count: 0,
        single_ideal_mismatch_endpoint_count: 0,
        build_mode_divergent_endpoint_count: 0,
        default_activation_clamped_endpoint_count: 0,
        single_activation_clamped_endpoint_count: 0,
        default_collapsed_extrema_channel_count: 0,
        single_collapsed_extrema_channel_count: 0,
        maximum_default_ideal_delta_codes: None,
        maximum_single_ideal_delta_codes: None,
        witness_ranking_op_indices: Vec::new(),
        ops: Vec::new(),
        source_commit: TFLITE_SOURCE_COMMIT,
        source_references: source_references(),
        method: method(),
        interpretation_boundary: interpretation_boundary(),
    }
}

pub(super) fn build_kernel_witnesses(
    model_bytes: &[u8],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    atlas: &AccumulatorAtlasAnalysis,
    fidelity: &RequantizationFidelityAnalysis,
) -> KernelWitnessAnalysis {
    let rows = ops
        .iter()
        .filter(|op| is_candidate(&op.name))
        .map(|op| assess_op(model_bytes, op, tensors, atlas, fidelity))
        .collect::<Vec<_>>();
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let mut ranking = assessed.iter().map(|row| row.op_index).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        let left_row = rows.iter().find(|row| row.op_index == *left).unwrap();
        let right_row = rows.iter().find(|row| row.op_index == *right).unwrap();
        right_row
            .maximum_default_ideal_delta_codes
            .unwrap_or(-1)
            .cmp(&left_row.maximum_default_ideal_delta_codes.unwrap_or(-1))
            .then_with(|| {
                right_row
                    .default_activation_clamped_endpoint_count
                    .cmp(&left_row.default_activation_clamped_endpoint_count)
            })
            .then_with(|| left.cmp(right))
    });
    let candidate_op_count = rows.len();
    let assessed_op_count = assessed.len();
    KernelWitnessAnalysis {
        schema: KERNEL_WITNESS_SCHEMA,
        method_version: KERNEL_WITNESS_METHOD_VERSION,
        evidence_class: if assessed_op_count > 0 {
            "DERIVED"
        } else {
            "NOT_ASSESSABLE"
        },
        status: if candidate_op_count == 0 {
            "not_applicable"
        } else if assessed_op_count == candidate_op_count {
            "assessed"
        } else if assessed_op_count > 0 {
            "partial"
        } else {
            "not_assessed"
        },
        candidate_op_count,
        assessed_op_count,
        unassessed_op_count: candidate_op_count - assessed_op_count,
        assessed_channel_count: sum_rows(&assessed, |row| row.assessed_channel_count),
        fixed_point_assessed_channel_count: sum_rows(&assessed, |row| {
            row.fixed_point_assessed_channel_count
        }),
        witness_assignment_count: sum_rows(&assessed, |row| row.witness_assignment_count),
        fixed_point_endpoint_evaluation_count: sum_rows(&assessed, |row| {
            row.fixed_point_endpoint_evaluation_count
        }),
        default_ideal_mismatch_endpoint_count: sum_rows(&assessed, |row| {
            row.default_ideal_mismatch_endpoint_count
        }),
        single_ideal_mismatch_endpoint_count: sum_rows(&assessed, |row| {
            row.single_ideal_mismatch_endpoint_count
        }),
        build_mode_divergent_endpoint_count: sum_rows(&assessed, |row| {
            row.build_mode_divergent_endpoint_count
        }),
        default_activation_clamped_endpoint_count: sum_rows(&assessed, |row| {
            row.default_activation_clamped_endpoint_count
        }),
        single_activation_clamped_endpoint_count: sum_rows(&assessed, |row| {
            row.single_activation_clamped_endpoint_count
        }),
        default_collapsed_extrema_channel_count: sum_rows(&assessed, |row| {
            row.default_collapsed_extrema_channel_count
        }),
        single_collapsed_extrema_channel_count: sum_rows(&assessed, |row| {
            row.single_collapsed_extrema_channel_count
        }),
        maximum_default_ideal_delta_codes: max_rows(&assessed, |row| {
            row.maximum_default_ideal_delta_codes
        }),
        maximum_single_ideal_delta_codes: max_rows(&assessed, |row| {
            row.maximum_single_ideal_delta_codes
        }),
        witness_ranking_op_indices: ranking,
        ops: rows,
        source_commit: TFLITE_SOURCE_COMMIT,
        source_references: source_references(),
        method: method(),
        interpretation_boundary: interpretation_boundary(),
    }
}

fn assess_op(
    model_bytes: &[u8],
    op: &OpInfo,
    tensors: &[TensorInfo],
    atlas: &AccumulatorAtlasAnalysis,
    fidelity: &RequantizationFidelityAnalysis,
) -> KernelWitnessOpRow {
    match try_assess_op(model_bytes, op, tensors, atlas, fidelity) {
        Ok(row) => row,
        Err(reason) => KernelWitnessOpRow::not_assessed(op, tensors, reason),
    }
}

fn try_assess_op(
    model_bytes: &[u8],
    op: &OpInfo,
    tensors: &[TensorInfo],
    atlas: &AccumulatorAtlasAnalysis,
    fidelity: &RequantizationFidelityAnalysis,
) -> Result<KernelWitnessOpRow, String> {
    let input = required_tensor(tensors, op.inputs.first().copied(), "input")?;
    let weight = required_tensor(tensors, op.inputs.get(1).copied(), "weight")?;
    let output = required_tensor(tensors, op.outputs.first().copied(), "output")?;
    let (qmin, qmax) = quantized_code_range(&input.dtype)
        .ok_or_else(|| format!("Input tensor {} is not INT8 or UINT8.", input.index))?;
    if input.zero_point_sample.len() != 1 {
        return Err(format!(
            "Input tensor {} does not expose one zero-point.",
            input.index
        ));
    }
    let input_zero_point = input.zero_point_sample[0];
    if input_zero_point < qmin || input_zero_point > qmax {
        return Err(format!(
            "Input zero-point {} lies outside [{}, {}].",
            input_zero_point, qmin, qmax
        ));
    }
    let (weight_qmin, weight_qmax) = quantized_code_range(&weight.dtype)
        .ok_or_else(|| format!("Weight tensor {} is not INT8 or UINT8.", weight.index))?;
    let layout = weight_layout(op, weight)?;
    let channels = layout.channels();
    let terms = layout.terms();
    let zero_points = expanded_weight_zero_points(weight, channels, weight_qmin, weight_qmax)?;
    let raw_weights = extract_tensor_buffer(model_bytes, weight).ok_or_else(|| {
        format!(
            "Weight tensor {} constant bytes are unavailable.",
            weight.index
        )
    })?;
    if raw_weights.len() != channels.saturating_mul(terms) {
        return Err(format!(
            "Weight tensor {} byte count does not match {} channels x {} terms.",
            weight.index, channels, terms
        ));
    }
    let (biases, _, _) = decode_bias(model_bytes, op, tensors, channels)?;
    let accumulator = atlas
        .ops
        .iter()
        .find(|row| row.op_index == op.index)
        .ok_or_else(|| format!("Accumulator evidence is missing for op #{}.", op.index))?;
    if accumulator.assessment_status != "assessed" || accumulator.assessed_channel_count != channels
    {
        return Err(format!(
            "Accumulator evidence is not assessed for all channels at op #{}.",
            op.index
        ));
    }
    let requant = fidelity
        .ops
        .iter()
        .find(|row| row.op_index == op.index)
        .ok_or_else(|| format!("Requantization evidence is missing for op #{}.", op.index))?;
    validate_requantization_row(requant, channels)?;
    let output_scale = requant
        .output_scale
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .ok_or_else(|| format!("Output scale is unavailable for op #{}.", op.index))?;
    let output_zero_point = requant
        .output_zero_point
        .ok_or_else(|| format!("Output zero-point is unavailable for op #{}.", op.index))?;
    let output_code_range = requant
        .output_code_range
        .ok_or_else(|| format!("Output code range is unavailable for op #{}.", op.index))?;
    let activation_code_range = activation_code_range(
        &op.fused_activation,
        output_scale,
        output_zero_point,
        output_code_range,
    )?;

    let mut witnesses = Vec::with_capacity(channels);
    let mut channel_outcomes = Vec::with_capacity(channels);
    let mut op_ledger = Sha256::new();
    op_ledger.update(LEDGER_PREFIX);
    let mut fixed_point_assessed_channel_count = 0usize;
    let mut default_ideal_mismatch_endpoint_count = 0usize;
    let mut single_ideal_mismatch_endpoint_count = 0usize;
    let mut build_mode_divergent_endpoint_count = 0usize;
    let mut default_activation_clamped_endpoint_count = 0usize;
    let mut single_activation_clamped_endpoint_count = 0usize;
    let mut default_collapsed_extrema_channel_count = 0usize;
    let mut single_collapsed_extrema_channel_count = 0usize;

    for channel in 0..channels {
        let weight_zero_point = zero_points[channel];
        let mut positive_count = 0usize;
        let mut negative_count = 0usize;
        let mut zero_count = 0usize;
        let mut maximum_absolute_centered_weight = 0i64;
        let mut maximum_absolute_term_contribution = 0i64;
        let mut dot_minimum = 0i128;
        let mut dot_maximum = 0i128;
        let mut minimum_histogram = BTreeMap::new();
        let mut maximum_histogram = BTreeMap::new();
        let mut term_rows = Vec::with_capacity(terms);
        for term in 0..terms {
            let centered_weight =
                raw_code(raw_weights[layout.raw_index(channel, term)], &weight.dtype)
                    - weight_zero_point;
            let (minimum_code, maximum_code) =
                canonical_witness_codes(centered_weight, qmin, qmax, input_zero_point);
            match centered_weight.cmp(&0) {
                Ordering::Greater => positive_count += 1,
                Ordering::Less => negative_count += 1,
                Ordering::Equal => zero_count += 1,
            }
            *minimum_histogram.entry(minimum_code).or_insert(0usize) += 1;
            *maximum_histogram.entry(maximum_code).or_insert(0usize) += 1;
            let minimum_contribution =
                centered_weight as i128 * (minimum_code - input_zero_point) as i128;
            let maximum_contribution =
                centered_weight as i128 * (maximum_code - input_zero_point) as i128;
            dot_minimum += minimum_contribution;
            dot_maximum += maximum_contribution;
            maximum_absolute_centered_weight =
                maximum_absolute_centered_weight.max(centered_weight.abs());
            maximum_absolute_term_contribution = maximum_absolute_term_contribution.max(
                i64::try_from(minimum_contribution.abs().max(maximum_contribution.abs()))
                    .map_err(|_| "A term contribution exceeds signed 64-bit range.".to_string())?,
            );
            term_rows.push((centered_weight, minimum_code, maximum_code));
        }
        let bias = biases[channel] as i128;
        let post_minimum = dot_minimum + bias;
        let post_maximum = dot_maximum + bias;
        let expected_minimum = accumulator.channel_post_bias_min_decimals[channel]
            .parse::<i128>()
            .map_err(|_| "Accumulator minimum is not an integer.".to_string())?;
        let expected_maximum = accumulator.channel_post_bias_max_decimals[channel]
            .parse::<i128>()
            .map_err(|_| "Accumulator maximum is not an integer.".to_string())?;
        if post_minimum != expected_minimum || post_maximum != expected_maximum {
            return Err(format!(
                "Canonical witness does not reproduce accumulator endpoints at op #{} channel {}.",
                op.index, channel
            ));
        }
        let projection_context = ChannelProjectionContext {
            requant,
            channel,
            output_zero_point,
            activation_range: activation_code_range,
        };
        let minimum = endpoint_projection(
            EndpointWitnessInput {
                endpoint: "minimum",
                histogram: minimum_histogram,
                dot_product: dot_minimum,
                bias,
                post_bias: post_minimum,
            },
            &projection_context,
        );
        let maximum = endpoint_projection(
            EndpointWitnessInput {
                endpoint: "maximum",
                histogram: maximum_histogram,
                dot_product: dot_maximum,
                bias,
                post_bias: post_maximum,
            },
            &projection_context,
        );
        channel_outcomes.push(KernelChannelOutcome {
            channel_index: channel,
            post_bias_minimum: post_minimum,
            post_bias_maximum: post_maximum,
            default_minimum_preclamp_code: minimum.default_preclamp_code,
            default_maximum_preclamp_code: maximum.default_preclamp_code,
            default_minimum_output_code: minimum.default_output_code,
            default_maximum_output_code: maximum.default_output_code,
            single_minimum_preclamp_code: minimum.single_preclamp_code,
            single_maximum_preclamp_code: maximum.single_preclamp_code,
            single_minimum_output_code: minimum.single_output_code,
            single_maximum_output_code: maximum.single_output_code,
        });
        let fixed_point_assessed = minimum.default_output_code.is_some()
            && maximum.default_output_code.is_some()
            && minimum.single_output_code.is_some()
            && maximum.single_output_code.is_some();
        if fixed_point_assessed {
            fixed_point_assessed_channel_count += 1;
        }
        for endpoint in [&minimum, &maximum] {
            default_ideal_mismatch_endpoint_count += usize::from(
                endpoint.default_output_code.is_some()
                    && endpoint.default_output_code != endpoint.ideal_output_code,
            );
            single_ideal_mismatch_endpoint_count += usize::from(
                endpoint.single_output_code.is_some()
                    && endpoint.single_output_code != endpoint.ideal_output_code,
            );
            build_mode_divergent_endpoint_count += usize::from(
                endpoint.default_output_code.is_some()
                    && endpoint.default_output_code != endpoint.single_output_code,
            );
            default_activation_clamped_endpoint_count +=
                usize::from(endpoint.default_activation_clamped == Some(true));
            single_activation_clamped_endpoint_count +=
                usize::from(endpoint.single_activation_clamped == Some(true));
        }
        default_collapsed_extrema_channel_count += usize::from(
            minimum.default_output_code.is_some()
                && minimum.default_output_code == maximum.default_output_code,
        );
        single_collapsed_extrema_channel_count += usize::from(
            minimum.single_output_code.is_some()
                && minimum.single_output_code == maximum.single_output_code,
        );
        let mut pattern = Sha256::new();
        pattern.update(PATTERN_PREFIX);
        update_i64(&mut pattern, op.index as i64);
        update_i64(&mut pattern, channel as i64);
        update_i64(&mut pattern, terms as i64);
        for (centered_weight, minimum_code, maximum_code) in &term_rows {
            update_i16(&mut pattern, *centered_weight)?;
            update_i16(&mut pattern, *minimum_code)?;
            update_i16(&mut pattern, *maximum_code)?;
        }
        let pattern_sha256 = hex_digest(pattern.finalize().as_slice());
        update_channel_ledger(
            &mut op_ledger,
            op.index,
            channel,
            terms,
            positive_count,
            negative_count,
            zero_count,
            bias,
            post_minimum,
            post_maximum,
            &minimum,
            &maximum,
            &term_rows,
        )?;
        let witness = KernelChannelWitness {
            channel_index: channel,
            term_count: terms,
            positive_centered_weight_count: positive_count,
            negative_centered_weight_count: negative_count,
            zero_centered_weight_count: zero_count,
            maximum_absolute_centered_weight,
            maximum_absolute_term_contribution,
            maximum_default_ideal_delta_codes: max_endpoint_delta(&minimum, &maximum, |endpoint| {
                endpoint.default_ideal_delta_codes
            }),
            maximum_single_ideal_delta_codes: max_endpoint_delta(&minimum, &maximum, |endpoint| {
                endpoint.single_ideal_delta_codes
            }),
            build_mode_divergent_endpoint_count: [&minimum, &maximum]
                .iter()
                .filter(|endpoint| {
                    endpoint.default_output_code.is_some()
                        && endpoint.default_output_code != endpoint.single_output_code
                })
                .count(),
            minimum,
            maximum,
            witness_pattern_sha256: pattern_sha256,
        };
        witnesses.push(witness);
    }
    witnesses.sort_by(compare_witnesses);
    let worst_channel = witnesses.first().cloned();
    Ok(KernelWitnessOpRow {
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
        input_code_range: Some([qmin, qmax]),
        input_zero_point: Some(input_zero_point),
        weight_dtype: weight.dtype.clone(),
        weight_shape: weight.shape.clone(),
        output_channel_axis: Some(layout.output_axis()),
        output_dtype: requant.output_dtype.clone(),
        output_scale: Some(output_scale),
        output_zero_point: Some(output_zero_point),
        output_code_range: Some(output_code_range),
        fused_activation: op.fused_activation.clone(),
        activation_code_range: Some(activation_code_range),
        assessed_channel_count: channels,
        fixed_point_assessed_channel_count,
        accumulation_terms_per_channel: Some(terms),
        witness_assignment_count: channels.saturating_mul(terms).saturating_mul(2),
        fixed_point_endpoint_evaluation_count: fixed_point_assessed_channel_count.saturating_mul(4),
        default_ideal_mismatch_endpoint_count,
        single_ideal_mismatch_endpoint_count,
        build_mode_divergent_endpoint_count,
        default_activation_clamped_endpoint_count,
        single_activation_clamped_endpoint_count,
        default_collapsed_extrema_channel_count,
        single_collapsed_extrema_channel_count,
        maximum_default_ideal_delta_codes: witnesses
            .iter()
            .filter_map(|witness| witness.maximum_default_ideal_delta_codes)
            .max(),
        maximum_single_ideal_delta_codes: witnesses
            .iter()
            .filter_map(|witness| witness.maximum_single_ideal_delta_codes)
            .max(),
        top_channels: witnesses.into_iter().take(TOP_WITNESS_LIMIT).collect(),
        worst_channel,
        witness_ledger_sha256: hex_digest(op_ledger.finalize().as_slice()),
        ledger_hash_method: ledger_hash_method(),
        channel_outcomes,
    })
}

fn endpoint_projection(
    input: EndpointWitnessInput,
    context: &ChannelProjectionContext<'_>,
) -> EndpointProjection {
    let EndpointWitnessInput {
        endpoint,
        histogram,
        dot_product,
        bias,
        post_bias,
    } = input;
    let ChannelProjectionContext {
        requant,
        channel,
        output_zero_point,
        activation_range,
    } = context;
    let channel = *channel;
    let output_zero_point = *output_zero_point;
    let activation_range = *activation_range;
    let real_multiplier = requant.channel_real_multipliers[channel];
    let ideal_preclamp = if post_bias.abs() <= (1i128 << 53) {
        Some(
            round_ties_away_from_zero(post_bias as f64 * real_multiplier)
                .saturating_add(output_zero_point),
        )
    } else {
        None
    };
    let ideal_output =
        ideal_preclamp.map(|code| code.clamp(activation_range[0], activation_range[1]));
    let post_i32 = i32::try_from(post_bias).ok();
    let default_scaled = post_i32.and_then(|value| {
        multiply_by_quantized_multiplier_default(
            value,
            requant.channel_quantized_multipliers[channel],
            requant.channel_shifts[channel],
        )
    });
    let single_scaled = post_i32.and_then(|value| {
        multiply_by_quantized_multiplier_single_rounding(
            value,
            requant.channel_single_rounding_quantized_multipliers[channel],
            requant.channel_single_rounding_shifts[channel],
        )
    });
    let default_preclamp = default_scaled.map(|value| value as i64 + output_zero_point);
    let single_preclamp = single_scaled.map(|value| value as i64 + output_zero_point);
    let default_output =
        default_preclamp.map(|code| code.clamp(activation_range[0], activation_range[1]));
    let single_output =
        single_preclamp.map(|code| code.clamp(activation_range[0], activation_range[1]));
    EndpointProjection {
        endpoint,
        witness_code_histogram: histogram
            .into_iter()
            .map(|(code, count)| CodeCount { code, count })
            .collect(),
        dot_product_decimal: dot_product.to_string(),
        bias_decimal: bias.to_string(),
        post_bias_accumulator_decimal: post_bias.to_string(),
        ideal_preclamp_code: ideal_preclamp,
        ideal_output_code: ideal_output,
        default_scaled_accumulator: default_scaled.map(i64::from),
        default_preclamp_code: default_preclamp,
        default_output_code: default_output,
        default_activation_clamped: default_preclamp
            .map(|code| code < activation_range[0] || code > activation_range[1]),
        default_ideal_delta_codes: output_delta(default_output, ideal_output),
        single_scaled_accumulator: single_scaled.map(i64::from),
        single_preclamp_code: single_preclamp,
        single_output_code: single_output,
        single_activation_clamped: single_preclamp
            .map(|code| code < activation_range[0] || code > activation_range[1]),
        single_ideal_delta_codes: output_delta(single_output, ideal_output),
        build_mode_output_delta_codes: output_delta(default_output, single_output),
    }
}

#[allow(clippy::too_many_arguments)]
fn update_channel_ledger(
    ledger: &mut Sha256,
    op_index: usize,
    channel: usize,
    terms: usize,
    positive_count: usize,
    negative_count: usize,
    zero_count: usize,
    bias: i128,
    post_minimum: i128,
    post_maximum: i128,
    minimum: &EndpointProjection,
    maximum: &EndpointProjection,
    term_rows: &[(i64, i64, i64)],
) -> Result<(), String> {
    for value in [
        op_index as i128,
        channel as i128,
        terms as i128,
        positive_count as i128,
        negative_count as i128,
        zero_count as i128,
        bias,
        post_minimum,
        post_maximum,
    ] {
        update_i64(
            ledger,
            i64::try_from(value).map_err(|_| {
                "Kernel witness ledger value exceeds signed 64-bit range.".to_string()
            })?,
        );
    }
    for value in endpoint_ledger_values(minimum)
        .into_iter()
        .chain(endpoint_ledger_values(maximum))
    {
        update_i64(ledger, value);
    }
    for (centered_weight, minimum_code, maximum_code) in term_rows {
        update_i16(ledger, *centered_weight)?;
        update_i16(ledger, *minimum_code)?;
        update_i16(ledger, *maximum_code)?;
    }
    Ok(())
}

fn endpoint_ledger_values(endpoint: &EndpointProjection) -> [i64; 9] {
    [
        optional_i64(endpoint.ideal_preclamp_code),
        optional_i64(endpoint.ideal_output_code),
        optional_i64(endpoint.default_scaled_accumulator),
        optional_i64(endpoint.default_preclamp_code),
        optional_i64(endpoint.default_output_code),
        optional_bool(endpoint.default_activation_clamped),
        optional_i64(endpoint.single_scaled_accumulator),
        optional_i64(endpoint.single_preclamp_code),
        optional_i64(endpoint.single_output_code),
    ]
}

fn canonical_witness_codes(
    centered_weight: i64,
    qmin: i64,
    qmax: i64,
    input_zero_point: i64,
) -> (i64, i64) {
    match centered_weight.cmp(&0) {
        Ordering::Greater => (qmin, qmax),
        Ordering::Less => (qmax, qmin),
        Ordering::Equal => (input_zero_point, input_zero_point),
    }
}

fn activation_code_range(
    fused_activation: &str,
    output_scale: f64,
    output_zero_point: i64,
    output_range: [i64; 2],
) -> Result<[i64; 2], String> {
    let quantize = |real: f64| {
        round_ties_away_from_zero(real / output_scale)
            .saturating_add(output_zero_point)
            .clamp(output_range[0], output_range[1])
    };
    match fused_activation {
        "NONE" => Ok(output_range),
        "RELU" => Ok([quantize(0.0), output_range[1]]),
        "RELU_N1_TO_1" => Ok([quantize(-1.0), quantize(1.0)]),
        "RELU6" => Ok([quantize(0.0), quantize(6.0)]),
        other => Err(format!(
            "Fused activation {other} is outside the pinned integer witness contract."
        )),
    }
}

fn validate_requantization_row(row: &RequantizationOpRow, channels: usize) -> Result<(), String> {
    if row.assessment_status != "assessed" || row.assessed_channel_count != channels {
        return Err(format!(
            "Requantization evidence is not assessed for all channels at op #{}.",
            row.op_index
        ));
    }
    for (name, length) in [
        ("real multiplier", row.channel_real_multipliers.len()),
        (
            "default multiplier",
            row.channel_quantized_multipliers.len(),
        ),
        ("default shift", row.channel_shifts.len()),
        (
            "single-rounding multiplier",
            row.channel_single_rounding_quantized_multipliers.len(),
        ),
        (
            "single-rounding shift",
            row.channel_single_rounding_shifts.len(),
        ),
    ] {
        if length != channels {
            return Err(format!(
                "Requantization {name} cardinality {length} does not match {channels} channels at op #{}.",
                row.op_index
            ));
        }
    }
    Ok(())
}

fn compare_witnesses(left: &KernelChannelWitness, right: &KernelChannelWitness) -> Ordering {
    right
        .maximum_default_ideal_delta_codes
        .unwrap_or(-1)
        .cmp(&left.maximum_default_ideal_delta_codes.unwrap_or(-1))
        .then_with(|| {
            right
                .build_mode_divergent_endpoint_count
                .cmp(&left.build_mode_divergent_endpoint_count)
        })
        .then_with(|| {
            let right_abs = endpoint_accumulator_abs(right);
            let left_abs = endpoint_accumulator_abs(left);
            right_abs.cmp(&left_abs)
        })
        .then_with(|| left.channel_index.cmp(&right.channel_index))
}

fn endpoint_accumulator_abs(witness: &KernelChannelWitness) -> i128 {
    [&witness.minimum, &witness.maximum]
        .iter()
        .filter_map(|endpoint| endpoint.post_bias_accumulator_decimal.parse::<i128>().ok())
        .map(i128::abs)
        .max()
        .unwrap_or(0)
}

fn max_endpoint_delta<F>(
    minimum: &EndpointProjection,
    maximum: &EndpointProjection,
    field: F,
) -> Option<i64>
where
    F: Fn(&EndpointProjection) -> Option<i64>,
{
    [field(minimum), field(maximum)]
        .into_iter()
        .flatten()
        .map(i64::abs)
        .max()
}

fn output_delta(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    Some(left?.saturating_sub(right?))
}

fn optional_i64(value: Option<i64>) -> i64 {
    value.unwrap_or(MISSING_I64)
}

fn optional_bool(value: Option<bool>) -> i64 {
    value.map(i64::from).unwrap_or(MISSING_I64)
}

fn update_i64(digest: &mut Sha256, value: i64) {
    digest.update(value.to_le_bytes());
}

fn update_i16(digest: &mut Sha256, value: i64) -> Result<(), String> {
    let value = i16::try_from(value)
        .map_err(|_| "Kernel witness term value exceeds signed 16-bit range.".to_string())?;
    digest.update(value.to_le_bytes());
    Ok(())
}

fn tensor_for_input(tensors: &[TensorInfo], index: Option<i32>) -> Option<&TensorInfo> {
    index
        .and_then(|value| usize::try_from(value).ok())
        .and_then(|position| tensors.get(position))
}

fn required_tensor<'a>(
    tensors: &'a [TensorInfo],
    index: Option<i32>,
    role: &str,
) -> Result<&'a TensorInfo, String> {
    tensor_for_input(tensors, index).ok_or_else(|| format!("The {role} tensor is unavailable."))
}

fn sum_rows<F>(rows: &[&KernelWitnessOpRow], field: F) -> usize
where
    F: Fn(&KernelWitnessOpRow) -> usize,
{
    rows.iter().map(|row| field(row)).sum()
}

fn max_rows<F>(rows: &[&KernelWitnessOpRow], field: F) -> Option<i64>
where
    F: Fn(&KernelWitnessOpRow) -> Option<i64>,
{
    rows.iter().filter_map(|row| field(row)).max()
}

fn is_candidate(name: &str) -> bool {
    matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED")
}

fn source_references() -> Vec<SourceReference> {
    vec![
        source_reference(
            "multiplier_encoding",
            "tensorflow/lite/kernels/internal/quantization_util.cc",
            "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
        ),
        source_reference(
            "effective_scale_and_activation_range",
            "tensorflow/lite/kernels/kernel_util.cc",
            "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
        ),
        source_reference(
            "default_and_single_rounding_execution",
            "tensorflow/lite/kernels/internal/common.cc",
            "ba5308bf76383d600d033c948fe0659710939e6f1f15a800b5413e5fc822ddfa",
        ),
        source_reference(
            "conv_integer_loop",
            "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h",
            "370f80020b9aa44e61bb3a0a0d081c035432f99aaf1cb66c6c3d072700a4bec8",
        ),
        source_reference(
            "depthwise_integer_loop",
            "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h",
            "98d0e11ff47bb32e3485f3f891be77cf733f6a47d52b6fc236d437fc581a069c",
        ),
        source_reference(
            "fully_connected_integer_loop",
            "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h",
            "88beede059cee0eba102c5a108c0750b33fe823af25393053e618dd29781abd6",
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

fn ledger_hash_method() -> &'static str {
    "SHA-256 over schema prefix, then channel-major rows: nine signed i64 LE header fields; nine signed i64 LE fields for minimum and maximum projections; then term-major triples of centered-weight, canonical-minimum-code, canonical-maximum-code as signed i16 LE. Missing projection values use INT64_MIN."
}

fn method() -> &'static str {
    "For every exactly bounded constant 8-bit TFLite Conv, Depthwise Conv, and rank-2 Fully Connected output channel, choose the canonical legal input code independently at every full-valid receptive-field term: qmin for positive weights and qmax for negative weights at the minimum, reversed at the maximum, and input zero-point for zero weights. Prove both dot-product endpoints against Accumulator Headroom, then execute the pinned default-double-rounding and TFLITE_SINGLE_ROUNDING fixed-point requantization equations, output zero-point addition, and fused-activation clamp."
}

fn interpretation_boundary() -> &'static str {
    "This is an exact per-output-channel synthetic witness for one full-valid receptive field under independent legal input codes and the pinned TFLite reference integer algebra. Different output channels generally require different input patterns, edge positions with padding use fewer nonzero terms, and the artifact does not identify the runtime build flag, delegate lowering, executed microkernel, or observed activation probability. The witness is not a full-model input, calibration recommendation, accuracy result, or proof that a deployed backend executed this path."
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_witness_codes_attain_each_signed_term_endpoint() {
        assert_eq!(canonical_witness_codes(7, -128, 127, -3), (-128, 127));
        assert_eq!(canonical_witness_codes(-9, -128, 127, -3), (127, -128));
        assert_eq!(canonical_witness_codes(0, -128, 127, -3), (-3, -3));
    }

    #[test]
    fn activation_ranges_are_derived_from_output_contract() {
        assert_eq!(
            activation_code_range("NONE", 0.25, 0, [-128, 127]),
            Ok([-128, 127])
        );
        assert_eq!(
            activation_code_range("RELU", 0.25, -4, [-128, 127]),
            Ok([-4, 127])
        );
        assert_eq!(
            activation_code_range("RELU6", 0.25, -4, [-128, 127]),
            Ok([-4, 20])
        );
        assert!(activation_code_range("TANH", 0.25, 0, [-128, 127]).is_err());
    }
}
