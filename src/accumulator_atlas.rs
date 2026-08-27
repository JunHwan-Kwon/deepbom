use super::{extract_tensor_buffer, OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

const ACCUMULATOR_ATLAS_SCHEMA: &str = "deepbom.accumulator_atlas.v1.3";
const ACCUMULATOR_ATLAS_METHOD_VERSION: &str = "2026-07-30.4";
const TFLITE_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const INT32_MIN: i128 = i32::MIN as i128;
const INT32_MAX: i128 = i32::MAX as i128;
const INT32_HALF_RANGE_MAX: i128 = INT32_MAX / 2;
const BIAS_HALF_RANGE_FLOAT32_TOLERANCE_CODES: i128 = 129;
const REQUIRED_BITS_HISTOGRAM_BINS: usize = 129;
const TOP_CHANNEL_LIMIT: usize = 8;

#[derive(Clone, Serialize)]
struct AccumulatorChannelWitness {
    channel_index: usize,
    positive_centered_weight_sum_decimal: String,
    negative_centered_weight_sum_decimal: String,
    bias_decimal: String,
    dot_product_min_decimal: String,
    dot_product_max_decimal: String,
    post_bias_min_decimal: String,
    post_bias_max_decimal: String,
    accumulator_envelope_min_decimal: String,
    accumulator_envelope_max_decimal: String,
    maximum_absolute_accumulator_decimal: String,
    required_signed_bits: usize,
    int32_ratio: f64,
    bias_int32_ratio: f64,
    bias_exceeds_half_range: bool,
    bias_half_range_excess_decimal: String,
    bias_half_range_classification: &'static str,
    exact_zero_centered_kernel: bool,
    fits_int32: bool,
}

#[derive(Serialize)]
pub(super) struct AccumulatorOpRow {
    pub(super) op_index: usize,
    pub(super) op_name: String,
    pub(super) assessment_status: &'static str,
    not_assessed_reason: String,
    input_tensor_index: Option<i32>,
    input_tensor_name: String,
    weight_tensor_index: Option<i32>,
    weight_tensor_name: String,
    bias_tensor_index: Option<i32>,
    bias_tensor_name: String,
    input_dtype: String,
    weight_dtype: String,
    bias_dtype: String,
    input_code_range: Option<[i64; 2]>,
    input_zero_point: Option<i64>,
    weight_shape: Vec<i32>,
    output_channel_axis: Option<usize>,
    pub(super) output_channel_count: Option<usize>,
    accumulation_terms_per_channel: Option<usize>,
    weight_zero_point_mode: String,
    bias_status: String,
    pub(super) assessed_channel_count: usize,
    stored_bias_channel_count: usize,
    int32_safe_channel_count: usize,
    int32_overflow_channel_count: usize,
    overflow_channel_indices: Vec<usize>,
    maximum_absolute_bias_decimal: Option<String>,
    maximum_bias_int32_ratio: Option<f64>,
    bias_half_range_exceedance_channel_count: usize,
    bias_half_range_exceedance_channel_indices: Vec<usize>,
    bias_half_range_guard_adjacent_channel_count: usize,
    bias_half_range_material_exceedance_channel_count: usize,
    bias_half_range_material_exceedance_channel_indices: Vec<usize>,
    exact_zero_kernel_channel_count: usize,
    exact_zero_bias_half_range_exceedance_channel_count: usize,
    exact_zero_bias_half_range_material_exceedance_channel_count: usize,
    maximum_absolute_accumulator_decimal: Option<String>,
    maximum_int32_ratio: Option<f64>,
    maximum_required_signed_bits: Option<usize>,
    minimum_int32_headroom_bits: Option<i32>,
    metadata_only_magnitude_bound_decimal: Option<String>,
    metadata_only_int32_ratio: Option<f64>,
    exact_tightening_factor: Option<f64>,
    pub(super) channel_accumulator_envelope_min_decimals: Vec<String>,
    pub(super) channel_accumulator_envelope_max_decimals: Vec<String>,
    pub(super) channel_post_bias_min_decimals: Vec<String>,
    pub(super) channel_post_bias_max_decimals: Vec<String>,
    channel_required_signed_bits: Vec<usize>,
    required_signed_bits_histogram: Vec<usize>,
    top_channels: Vec<AccumulatorChannelWitness>,
    worst_channel: Option<AccumulatorChannelWitness>,
    channel_ledger_sha256: String,
    ledger_hash_method: &'static str,
    source_file: String,
    source_url: String,
    formula: &'static str,
}

impl AccumulatorOpRow {
    fn not_assessed(op: &OpInfo, tensors: &[TensorInfo], reason: String) -> Self {
        let input = op
            .inputs
            .first()
            .and_then(|index| tensor_at(tensors, *index));
        let weight = op
            .inputs
            .get(1)
            .and_then(|index| tensor_at(tensors, *index));
        let bias = op
            .inputs
            .get(2)
            .and_then(|index| tensor_at(tensors, *index));
        Self {
            op_index: op.index,
            op_name: op.name.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            input_tensor_index: op.inputs.first().copied(),
            input_tensor_name: input.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            weight_tensor_index: op.inputs.get(1).copied(),
            weight_tensor_name: weight.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            bias_tensor_index: op.inputs.get(2).copied().filter(|index| *index >= 0),
            bias_tensor_name: bias.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            input_dtype: input.map(|tensor| tensor.dtype.clone()).unwrap_or_default(),
            weight_dtype: weight.map(|tensor| tensor.dtype.clone()).unwrap_or_default(),
            bias_dtype: bias.map(|tensor| tensor.dtype.clone()).unwrap_or_default(),
            input_code_range: None,
            input_zero_point: None,
            weight_shape: weight.map(|tensor| tensor.shape.clone()).unwrap_or_default(),
            output_channel_axis: None,
            output_channel_count: None,
            accumulation_terms_per_channel: None,
            weight_zero_point_mode: String::new(),
            bias_status: String::new(),
            assessed_channel_count: 0,
            stored_bias_channel_count: 0,
            int32_safe_channel_count: 0,
            int32_overflow_channel_count: 0,
            overflow_channel_indices: Vec::new(),
            maximum_absolute_bias_decimal: None,
            maximum_bias_int32_ratio: None,
            bias_half_range_exceedance_channel_count: 0,
            bias_half_range_exceedance_channel_indices: Vec::new(),
            bias_half_range_guard_adjacent_channel_count: 0,
            bias_half_range_material_exceedance_channel_count: 0,
            bias_half_range_material_exceedance_channel_indices: Vec::new(),
            exact_zero_kernel_channel_count: 0,
            exact_zero_bias_half_range_exceedance_channel_count: 0,
            exact_zero_bias_half_range_material_exceedance_channel_count: 0,
            maximum_absolute_accumulator_decimal: None,
            maximum_int32_ratio: None,
            maximum_required_signed_bits: None,
            minimum_int32_headroom_bits: None,
            metadata_only_magnitude_bound_decimal: None,
            metadata_only_int32_ratio: None,
            exact_tightening_factor: None,
            channel_accumulator_envelope_min_decimals: Vec::new(),
            channel_accumulator_envelope_max_decimals: Vec::new(),
            channel_post_bias_min_decimals: Vec::new(),
            channel_post_bias_max_decimals: Vec::new(),
            channel_required_signed_bits: Vec::new(),
            required_signed_bits_histogram: Vec::new(),
            top_channels: Vec::new(),
            worst_channel: None,
            channel_ledger_sha256: String::new(),
            ledger_hash_method: "SHA-256 over UTF-8 rows op=<index>;channel=<index>;post_min=<decimal>;post_max=<decimal>;min=<decimal>;max=<decimal>;bits=<unsigned>\\n",
            source_file: source_file(&op.name, input.map(|tensor| tensor.dtype.as_str())),
            source_url: source_url(&op.name, input.map(|tensor| tensor.dtype.as_str())),
            formula: "w=q_weight-zp_weight[channel]; x in [qmin-zp_input,qmax-zp_input]; dot_min=sum(w>=0?w*xmin:w*xmax); dot_max=sum(w>=0?w*xmax:w*xmin); envelope=hull(0,dot,[dot_min+bias,dot_max+bias])",
        }
    }
}

#[derive(Serialize)]
pub(super) struct AccumulatorAtlasAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    stored_bias_channel_count: usize,
    int32_safe_channel_count: usize,
    int32_overflow_channel_count: usize,
    overflow_op_count: usize,
    maximum_absolute_bias_decimal: Option<String>,
    maximum_bias_int32_ratio: Option<f64>,
    bias_half_range_exceedance_channel_count: usize,
    bias_half_range_exceedance_op_count: usize,
    bias_half_range_guard_adjacent_channel_count: usize,
    bias_half_range_material_exceedance_channel_count: usize,
    bias_half_range_material_exceedance_op_count: usize,
    exact_zero_kernel_channel_count: usize,
    exact_zero_bias_half_range_exceedance_channel_count: usize,
    exact_zero_bias_half_range_material_exceedance_channel_count: usize,
    bias_half_range_reference_decimal: String,
    bias_half_range_float32_tolerance_codes: String,
    bias_half_range_float32_tolerance_formula: &'static str,
    bias_half_range_reference_source_url: String,
    bias_half_range_reference_evidence_class: &'static str,
    maximum_absolute_accumulator_decimal: Option<String>,
    maximum_int32_ratio: Option<f64>,
    maximum_required_signed_bits: Option<usize>,
    minimum_int32_headroom_bits: Option<i32>,
    headroom_ranking_op_indices: Vec<usize>,
    required_signed_bits_histogram: Vec<usize>,
    pub(super) ops: Vec<AccumulatorOpRow>,
    source_commit: &'static str,
    source_urls: Vec<String>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Clone, Copy)]
pub(super) enum WeightLayout {
    OutputMajor { channels: usize, terms: usize },
    DepthwiseLastAxis { channels: usize, terms: usize },
}

impl WeightLayout {
    pub(super) fn channels(self) -> usize {
        match self {
            Self::OutputMajor { channels, .. } | Self::DepthwiseLastAxis { channels, .. } => {
                channels
            }
        }
    }

    pub(super) fn terms(self) -> usize {
        match self {
            Self::OutputMajor { terms, .. } | Self::DepthwiseLastAxis { terms, .. } => terms,
        }
    }

    pub(super) fn output_axis(self) -> usize {
        match self {
            Self::OutputMajor { .. } => 0,
            Self::DepthwiseLastAxis { .. } => 3,
        }
    }

    pub(super) fn raw_index(self, channel: usize, term: usize) -> usize {
        match self {
            Self::OutputMajor { terms, .. } => channel * terms + term,
            Self::DepthwiseLastAxis { channels, .. } => term * channels + channel,
        }
    }
}

pub(super) fn build_accumulator_atlas(
    model_bytes: &[u8],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> AccumulatorAtlasAnalysis {
    let rows = ops
        .iter()
        .filter(|op| is_candidate(&op.name))
        .map(|op| assess_op(model_bytes, op, tensors))
        .collect::<Vec<_>>();
    let mut ranked_positions = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.assessment_status == "assessed")
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    ranked_positions.sort_by(|left, right| {
        cmp_f64_desc(
            rows[*left].maximum_int32_ratio.unwrap_or(0.0),
            rows[*right].maximum_int32_ratio.unwrap_or(0.0),
        )
        .then_with(|| rows[*left].op_index.cmp(&rows[*right].op_index))
    });
    let assessed_positions = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.assessment_status == "assessed")
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let candidate_op_count = rows.len();
    let assessed_op_count = assessed_positions.len();
    let unassessed_op_count = candidate_op_count - assessed_op_count;
    let assessed_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].assessed_channel_count)
        .sum();
    let stored_bias_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].stored_bias_channel_count)
        .sum();
    let int32_safe_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].int32_safe_channel_count)
        .sum();
    let int32_overflow_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].int32_overflow_channel_count)
        .sum();
    let overflow_op_count = assessed_positions
        .iter()
        .filter(|position| rows[**position].int32_overflow_channel_count > 0)
        .count();
    let maximum_bias_row = assessed_positions
        .iter()
        .filter_map(|position| {
            rows[*position]
                .maximum_bias_int32_ratio
                .map(|ratio| (ratio, &rows[*position]))
        })
        .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let bias_half_range_exceedance_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].bias_half_range_exceedance_channel_count)
        .sum();
    let bias_half_range_exceedance_op_count = assessed_positions
        .iter()
        .filter(|position| rows[**position].bias_half_range_exceedance_channel_count > 0)
        .count();
    let bias_half_range_guard_adjacent_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].bias_half_range_guard_adjacent_channel_count)
        .sum();
    let bias_half_range_material_exceedance_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].bias_half_range_material_exceedance_channel_count)
        .sum();
    let bias_half_range_material_exceedance_op_count = assessed_positions
        .iter()
        .filter(|position| rows[**position].bias_half_range_material_exceedance_channel_count > 0)
        .count();
    let exact_zero_kernel_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].exact_zero_kernel_channel_count)
        .sum();
    let exact_zero_bias_half_range_exceedance_channel_count = assessed_positions
        .iter()
        .map(|position| rows[*position].exact_zero_bias_half_range_exceedance_channel_count)
        .sum();
    let exact_zero_bias_half_range_material_exceedance_channel_count = assessed_positions
        .iter()
        .map(|position| {
            rows[*position].exact_zero_bias_half_range_material_exceedance_channel_count
        })
        .sum();
    let maximum_row = ranked_positions.first().map(|position| &rows[*position]);
    let mut histogram = vec![0usize; REQUIRED_BITS_HISTOGRAM_BINS];
    for position in &assessed_positions {
        for (bin, count) in rows[*position]
            .required_signed_bits_histogram
            .iter()
            .enumerate()
        {
            histogram[bin] += count;
        }
    }
    let status = if candidate_op_count == 0 {
        "not_applicable"
    } else if unassessed_op_count > 0 {
        "partial"
    } else {
        "assessed"
    };
    AccumulatorAtlasAnalysis {
        schema: ACCUMULATOR_ATLAS_SCHEMA,
        method_version: ACCUMULATOR_ATLAS_METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        candidate_op_count,
        assessed_op_count,
        unassessed_op_count,
        assessed_channel_count,
        stored_bias_channel_count,
        int32_safe_channel_count,
        int32_overflow_channel_count,
        overflow_op_count,
        maximum_absolute_bias_decimal: maximum_bias_row
            .and_then(|(_, row)| row.maximum_absolute_bias_decimal.clone()),
        maximum_bias_int32_ratio: maximum_bias_row.map(|(ratio, _)| ratio),
        bias_half_range_exceedance_channel_count,
        bias_half_range_exceedance_op_count,
        bias_half_range_guard_adjacent_channel_count,
        bias_half_range_material_exceedance_channel_count,
        bias_half_range_material_exceedance_op_count,
        exact_zero_kernel_channel_count,
        exact_zero_bias_half_range_exceedance_channel_count,
        exact_zero_bias_half_range_material_exceedance_channel_count,
        bias_half_range_reference_decimal: INT32_HALF_RANGE_MAX.to_string(),
        bias_half_range_float32_tolerance_codes:
            BIAS_HALF_RANGE_FLOAT32_TOLERANCE_CODES.to_string(),
        bias_half_range_float32_tolerance_formula:
            "ceil(2 * 2^-24 * floor(INT32_MAX/2)) + 1 = 129 codes",
        bias_half_range_reference_source_url: "https://github.com/tensorflow/tensorflow/commit/01108a05710fc0c72e212901512b448f9abbb03e".to_string(),
        bias_half_range_reference_evidence_class: "SOURCE_BACKED_REFERENCE",
        maximum_absolute_accumulator_decimal: maximum_row.and_then(|row| row.maximum_absolute_accumulator_decimal.clone()),
        maximum_int32_ratio: maximum_row.and_then(|row| row.maximum_int32_ratio),
        maximum_required_signed_bits: assessed_positions.iter().filter_map(|position| rows[*position].maximum_required_signed_bits).max(),
        minimum_int32_headroom_bits: assessed_positions.iter().filter_map(|position| rows[*position].minimum_int32_headroom_bits).min(),
        headroom_ranking_op_indices: ranked_positions.iter().map(|position| rows[*position].op_index).collect(),
        required_signed_bits_histogram: histogram,
        ops: rows,
        source_commit: TFLITE_SOURCE_COMMIT,
        source_urls: vec![
            source_url("CONV_2D", Some("UINT8")),
            source_url("CONV_2D", Some("INT8")),
            source_url("DEPTHWISE_CONV_2D", Some("UINT8")),
            source_url("DEPTHWISE_CONV_2D", Some("INT8")),
            source_url("FULLY_CONNECTED", Some("UINT8")),
            source_url("FULLY_CONNECTED", Some("INT8")),
        ],
        method: "Decode every constant 8-bit CONV_2D, DEPTHWISE_CONV_2D, and 2-D FULLY_CONNECTED kernel. For each output channel, split centered stored weights by sign and solve the linear dot-product extrema over the complete legal centered input-code interval. Include zero, dot-product extrema, and post-bias extrema in the accumulator envelope; retain every channel envelope and signed-bit requirement as decimal-string evidence.",
        interpretation_boundary: "The channel envelope is exact for the pinned TFLite reference integer accumulation algebra, a full receptive field, independent legal input codes, stored weights, declared zero-points, and stored INT32 bias. Padding skips terms and therefore cannot enlarge this full-field envelope because the input zero-point represents centered zero. Strict bias half-range exceedance is derived from every stored INT32 bias code; comparison with INT32_MAX/2 is a source-backed quantizer-policy reference, not evidence that a particular converter pass executed or skipped. Guard-adjacent and material counts are separated because input and weight scales are serialized as float32: 129 codes conservatively covers two float32 relative roundoff terms at the half-range plus final integer rounding. This tolerance class is numerical methodology, not a TensorFlow policy constant. The result is not an observed activation distribution, saturation frequency, accuracy estimate, or executed delegate/microkernel trace. A backend may use algebraically transformed offset corrections, split or wider accumulators, a different accumulation order, or a different lowering; import runtime evidence before claiming executed-kernel safety or behavior.",
    }
}

fn assess_op(model_bytes: &[u8], op: &OpInfo, tensors: &[TensorInfo]) -> AccumulatorOpRow {
    let Some(input) = op
        .inputs
        .first()
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return AccumulatorOpRow::not_assessed(
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
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            "Weight tensor is unavailable.".to_string(),
        );
    };
    let (qmin, qmax) = match quantized_code_range(&input.dtype) {
        Some(value) => value,
        None => {
            return AccumulatorOpRow::not_assessed(
                op,
                tensors,
                format!(
                    "Input tensor {} uses {}; INT8 or UINT8 is required.",
                    input.index, input.dtype
                ),
            )
        }
    };
    if input.quant_scales != 1 || input.scale_sample.len() != 1 {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Input tensor {} does not expose one per-tensor scale.",
                input.index
            ),
        );
    }
    if !input.scale_sample[0].is_finite() || input.scale_sample[0] <= 0.0 {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Input tensor {} has an invalid quantization scale.",
                input.index
            ),
        );
    }
    if input.quant_zero_points != 1 || input.zero_point_sample.len() != 1 {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Input tensor {} does not expose one per-tensor zero-point.",
                input.index
            ),
        );
    }
    let input_zero_point = input.zero_point_sample[0];
    if input_zero_point < qmin || input_zero_point > qmax {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Input zero-point {} lies outside [{}, {}].",
                input_zero_point, qmin, qmax
            ),
        );
    }
    let Some((weight_qmin, weight_qmax)) = quantized_code_range(&weight.dtype) else {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Weight tensor {} uses {}; INT8 or UINT8 is required.",
                weight.index, weight.dtype
            ),
        );
    };
    let layout = match weight_layout(op, weight) {
        Ok(value) => value,
        Err(reason) => return AccumulatorOpRow::not_assessed(op, tensors, reason),
    };
    let channels = layout.channels();
    let terms = layout.terms();
    let scales = &weight.scale_sample;
    if !(scales.len() == 1 || scales.len() == channels)
        || scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Weight tensor {} scale cardinality is not 1 or output-channel count {}.",
                weight.index, channels
            ),
        );
    }
    if scales.len() > 1 && weight.quantized_dimension != layout.output_axis() as i32 {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Weight tensor {} per-axis dimension {} does not match output-channel axis {}.",
                weight.index,
                weight.quantized_dimension,
                layout.output_axis()
            ),
        );
    }
    let zero_points = match expanded_weight_zero_points(weight, channels, weight_qmin, weight_qmax)
    {
        Ok(value) => value,
        Err(reason) => return AccumulatorOpRow::not_assessed(op, tensors, reason),
    };
    let Some(raw_weights) = extract_tensor_buffer(model_bytes, weight) else {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Weight tensor {} constant bytes are unavailable.",
                weight.index
            ),
        );
    };
    let expected_weight_bytes = match channels.checked_mul(terms) {
        Some(value) => value,
        None => {
            return AccumulatorOpRow::not_assessed(
                op,
                tensors,
                "Weight element count exceeds the analyzer integer range.".to_string(),
            )
        }
    };
    if raw_weights.len() != expected_weight_bytes {
        return AccumulatorOpRow::not_assessed(
            op,
            tensors,
            format!(
                "Weight tensor {} exposes {} byte(s); shape requires {} 8-bit element(s).",
                weight.index,
                raw_weights.len(),
                expected_weight_bytes
            ),
        );
    }
    let (bias, bias_tensor, bias_status) = match decode_bias(model_bytes, op, tensors, channels) {
        Ok(value) => value,
        Err(reason) => return AccumulatorOpRow::not_assessed(op, tensors, reason),
    };
    let input_min = (qmin - input_zero_point) as i128;
    let input_max = (qmax - input_zero_point) as i128;
    let mut witnesses = Vec::with_capacity(channels);
    let mut envelope_mins = Vec::with_capacity(channels);
    let mut envelope_maxs = Vec::with_capacity(channels);
    let mut post_bias_mins = Vec::with_capacity(channels);
    let mut post_bias_maxs = Vec::with_capacity(channels);
    let mut required_bits = Vec::with_capacity(channels);
    let mut histogram = vec![0usize; REQUIRED_BITS_HISTOGRAM_BINS];
    let mut overflow_indices = Vec::new();
    let mut bias_half_range_exceedance_indices = Vec::new();
    let mut bias_half_range_material_exceedance_indices = Vec::new();
    let mut ledger = Sha256::new();
    let mut maximum_legal_weight_magnitude = 0i128;
    let mut maximum_bias_magnitude = 0i128;
    let mut exact_zero_kernel_channel_count = 0usize;
    let mut exact_zero_bias_half_range_exceedance_channel_count = 0usize;
    let mut exact_zero_bias_half_range_material_exceedance_channel_count = 0usize;
    let mut bias_half_range_guard_adjacent_channel_count = 0usize;
    let has_stored_bias = bias_tensor.is_some();
    for channel in 0..channels {
        let zero_point = zero_points[channel] as i128;
        maximum_legal_weight_magnitude = maximum_legal_weight_magnitude.max(
            ((weight_qmin as i128) - zero_point)
                .abs()
                .max(((weight_qmax as i128) - zero_point).abs()),
        );
        let mut positive_sum = 0i128;
        let mut negative_sum = 0i128;
        for term in 0..terms {
            let raw = raw_code(raw_weights[layout.raw_index(channel, term)], &weight.dtype);
            let centered = raw as i128 - zero_point;
            if centered >= 0 {
                positive_sum += centered;
            } else {
                negative_sum += centered;
            }
        }
        let dot_min = positive_sum * input_min + negative_sum * input_max;
        let dot_max = positive_sum * input_max + negative_sum * input_min;
        let bias_value = bias[channel] as i128;
        let exact_zero_centered_kernel = positive_sum == 0 && negative_sum == 0;
        if exact_zero_centered_kernel {
            exact_zero_kernel_channel_count += 1;
        }
        let bias_half_range_excess = if has_stored_bias {
            (bias_value.abs() - INT32_HALF_RANGE_MAX).max(0)
        } else {
            0
        };
        let bias_exceeds_half_range = bias_half_range_excess > 0;
        let bias_half_range_classification =
            classify_bias_half_range_excess(bias_half_range_excess);
        if bias_exceeds_half_range {
            bias_half_range_exceedance_indices.push(channel);
            if exact_zero_centered_kernel {
                exact_zero_bias_half_range_exceedance_channel_count += 1;
            }
        }
        if bias_half_range_classification == "float32_guard_adjacent" {
            bias_half_range_guard_adjacent_channel_count += 1;
        } else if bias_half_range_classification == "material_exceedance" {
            bias_half_range_material_exceedance_indices.push(channel);
            if exact_zero_centered_kernel {
                exact_zero_bias_half_range_material_exceedance_channel_count += 1;
            }
        }
        maximum_bias_magnitude = maximum_bias_magnitude.max(bias_value.abs());
        let post_bias_min = dot_min + bias_value;
        let post_bias_max = dot_max + bias_value;
        let envelope_min = 0i128.min(dot_min).min(post_bias_min);
        let envelope_max = 0i128.max(dot_max).max(post_bias_max);
        let maximum_absolute = abs_max(envelope_min, envelope_max);
        let bits = required_signed_bits(envelope_min, envelope_max);
        let fits_int32 = envelope_min >= INT32_MIN && envelope_max <= INT32_MAX;
        if !fits_int32 {
            overflow_indices.push(channel);
        }
        histogram[bits.min(REQUIRED_BITS_HISTOGRAM_BINS - 1)] += 1;
        let witness = AccumulatorChannelWitness {
            channel_index: channel,
            positive_centered_weight_sum_decimal: positive_sum.to_string(),
            negative_centered_weight_sum_decimal: negative_sum.to_string(),
            bias_decimal: bias_value.to_string(),
            dot_product_min_decimal: dot_min.to_string(),
            dot_product_max_decimal: dot_max.to_string(),
            post_bias_min_decimal: post_bias_min.to_string(),
            post_bias_max_decimal: post_bias_max.to_string(),
            accumulator_envelope_min_decimal: envelope_min.to_string(),
            accumulator_envelope_max_decimal: envelope_max.to_string(),
            maximum_absolute_accumulator_decimal: maximum_absolute.to_string(),
            required_signed_bits: bits,
            int32_ratio: maximum_absolute as f64 / INT32_MAX as f64,
            bias_int32_ratio: bias_value.abs() as f64 / INT32_MAX as f64,
            bias_exceeds_half_range,
            bias_half_range_excess_decimal: bias_half_range_excess.to_string(),
            bias_half_range_classification,
            exact_zero_centered_kernel,
            fits_int32,
        };
        ledger.update(
            format!(
                "op={};channel={};post_min={};post_max={};min={};max={};bits={}\n",
                op.index, channel, post_bias_min, post_bias_max, envelope_min, envelope_max, bits
            )
            .as_bytes(),
        );
        envelope_mins.push(envelope_min.to_string());
        envelope_maxs.push(envelope_max.to_string());
        post_bias_mins.push(post_bias_min.to_string());
        post_bias_maxs.push(post_bias_max.to_string());
        required_bits.push(bits);
        witnesses.push(witness);
    }
    witnesses.sort_by(|left, right| {
        decimal_abs_cmp_desc(
            &left.maximum_absolute_accumulator_decimal,
            &right.maximum_absolute_accumulator_decimal,
        )
        .then_with(|| left.channel_index.cmp(&right.channel_index))
    });
    let worst = witnesses.first().cloned();
    let maximum_absolute = worst
        .as_ref()
        .and_then(|item| {
            item.maximum_absolute_accumulator_decimal
                .parse::<i128>()
                .ok()
        })
        .unwrap_or(0);
    let maximum_required_bits = required_bits.iter().copied().max().unwrap_or(1);
    let input_magnitude = input_min.abs().max(input_max.abs());
    let metadata_bound =
        (terms as i128) * input_magnitude * maximum_legal_weight_magnitude + maximum_bias_magnitude;
    let metadata_ratio = metadata_bound as f64 / INT32_MAX as f64;
    AccumulatorOpRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_tensor_index: Some(input.index as i32),
        input_tensor_name: input.name.clone(),
        weight_tensor_index: Some(weight.index as i32),
        weight_tensor_name: weight.name.clone(),
        bias_tensor_index: bias_tensor.map(|tensor| tensor.index as i32),
        bias_tensor_name: bias_tensor.map(|tensor| tensor.name.clone()).unwrap_or_default(),
        input_dtype: input.dtype.clone(),
        weight_dtype: weight.dtype.clone(),
        bias_dtype: bias_tensor.map(|tensor| tensor.dtype.clone()).unwrap_or_default(),
        input_code_range: Some([qmin, qmax]),
        input_zero_point: Some(input_zero_point),
        weight_shape: weight.shape.clone(),
        output_channel_axis: Some(layout.output_axis()),
        output_channel_count: Some(channels),
        accumulation_terms_per_channel: Some(terms),
        weight_zero_point_mode: if weight.zero_point_sample.len() == 1 { "per_tensor".to_string() } else { "per_output_channel".to_string() },
        bias_status,
        assessed_channel_count: channels,
        stored_bias_channel_count: if has_stored_bias { channels } else { 0 },
        int32_safe_channel_count: channels - overflow_indices.len(),
        int32_overflow_channel_count: overflow_indices.len(),
        overflow_channel_indices: overflow_indices,
        maximum_absolute_bias_decimal: has_stored_bias
            .then_some(maximum_bias_magnitude.to_string()),
        maximum_bias_int32_ratio: has_stored_bias
            .then_some(maximum_bias_magnitude as f64 / INT32_MAX as f64),
        bias_half_range_exceedance_channel_count: bias_half_range_exceedance_indices.len(),
        bias_half_range_exceedance_channel_indices: bias_half_range_exceedance_indices,
        bias_half_range_guard_adjacent_channel_count,
        bias_half_range_material_exceedance_channel_count:
            bias_half_range_material_exceedance_indices.len(),
        bias_half_range_material_exceedance_channel_indices:
            bias_half_range_material_exceedance_indices,
        exact_zero_kernel_channel_count,
        exact_zero_bias_half_range_exceedance_channel_count,
        exact_zero_bias_half_range_material_exceedance_channel_count,
        maximum_absolute_accumulator_decimal: Some(maximum_absolute.to_string()),
        maximum_int32_ratio: Some(maximum_absolute as f64 / INT32_MAX as f64),
        maximum_required_signed_bits: Some(maximum_required_bits),
        minimum_int32_headroom_bits: Some(32 - maximum_required_bits as i32),
        metadata_only_magnitude_bound_decimal: Some(metadata_bound.to_string()),
        metadata_only_int32_ratio: Some(metadata_ratio),
        exact_tightening_factor: (maximum_absolute > 0).then_some(metadata_bound as f64 / maximum_absolute as f64),
        channel_accumulator_envelope_min_decimals: envelope_mins,
        channel_accumulator_envelope_max_decimals: envelope_maxs,
        channel_post_bias_min_decimals: post_bias_mins,
        channel_post_bias_max_decimals: post_bias_maxs,
        channel_required_signed_bits: required_bits,
        required_signed_bits_histogram: histogram,
        top_channels: witnesses.iter().take(TOP_CHANNEL_LIMIT).cloned().collect(),
        worst_channel: worst,
        channel_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: "SHA-256 over UTF-8 rows op=<index>;channel=<index>;post_min=<decimal>;post_max=<decimal>;min=<decimal>;max=<decimal>;bits=<unsigned>\\n",
        source_file: source_file(&op.name, Some(&input.dtype)),
        source_url: source_url(&op.name, Some(&input.dtype)),
        formula: "w=q_weight-zp_weight[channel]; x in [qmin-zp_input,qmax-zp_input]; dot_min=sum(w>=0?w*xmin:w*xmax); dot_max=sum(w>=0?w*xmax:w*xmin); envelope=hull(0,dot,[dot_min+bias,dot_max+bias])",
    }
}

pub(super) fn weight_layout(op: &OpInfo, weight: &TensorInfo) -> Result<WeightLayout, String> {
    let dimensions = weight
        .shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).ok().filter(|value| *value > 0))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            format!(
                "Weight tensor {} has a dynamic or non-positive shape.",
                weight.index
            )
        })?;
    match op.name.as_str() {
        "CONV_2D" if dimensions.len() == 4 => Ok(WeightLayout::OutputMajor {
            channels: dimensions[0],
            terms: checked_product(&dimensions[1..])?,
        }),
        "DEPTHWISE_CONV_2D" if dimensions.len() == 4 && dimensions[0] == 1 => {
            Ok(WeightLayout::DepthwiseLastAxis {
                channels: dimensions[3],
                terms: checked_product(&dimensions[1..3])?,
            })
        }
        "FULLY_CONNECTED" if dimensions.len() == 2 => Ok(WeightLayout::OutputMajor {
            channels: dimensions[0],
            terms: dimensions[1],
        }),
        "CONV_2D" => Err(format!(
            "CONV_2D weight tensor {} is not OHWI rank 4.",
            weight.index
        )),
        "DEPTHWISE_CONV_2D" => Err(format!(
            "DEPTHWISE_CONV_2D weight tensor {} is not [1,H,W,O].",
            weight.index
        )),
        "FULLY_CONNECTED" => Err(format!(
            "FULLY_CONNECTED weight tensor {} is not rank 2.",
            weight.index
        )),
        _ => Err(format!(
            "{} is not an accumulator-atlas candidate.",
            op.name
        )),
    }
}

pub(super) fn expanded_weight_zero_points(
    weight: &TensorInfo,
    channels: usize,
    qmin: i64,
    qmax: i64,
) -> Result<Vec<i64>, String> {
    let values = &weight.zero_point_sample;
    if !(values.len() == 1 || values.len() == channels) {
        return Err(format!(
            "Weight tensor {} zero-point cardinality is not 1 or output-channel count {}.",
            weight.index, channels
        ));
    }
    if values.iter().any(|value| *value < qmin || *value > qmax) {
        return Err(format!(
            "Weight tensor {} contains a zero-point outside [{}, {}].",
            weight.index, qmin, qmax
        ));
    }
    Ok(if values.len() == 1 {
        vec![values[0]; channels]
    } else {
        values.clone()
    })
}

pub(super) fn decode_bias<'a>(
    model_bytes: &[u8],
    op: &OpInfo,
    tensors: &'a [TensorInfo],
    channels: usize,
) -> Result<(Vec<i32>, Option<&'a TensorInfo>, String), String> {
    let Some(index) = op.inputs.get(2).copied().filter(|index| *index >= 0) else {
        return Ok((vec![0; channels], None, "absent_zero_bias".to_string()));
    };
    let tensor = tensor_at(tensors, index)
        .ok_or_else(|| format!("Bias tensor {} is unavailable.", index))?;
    if tensor.dtype != "INT32" {
        return Err(format!(
            "Bias tensor {} uses {}; INT32 is required for integer accumulation.",
            tensor.index, tensor.dtype
        ));
    }
    let raw = extract_tensor_buffer(model_bytes, tensor).ok_or_else(|| {
        format!(
            "Bias tensor {} constant bytes are unavailable.",
            tensor.index
        )
    })?;
    let expected = channels
        .checked_mul(4)
        .ok_or_else(|| "Bias byte count exceeds the analyzer integer range.".to_string())?;
    if raw.len() != expected {
        return Err(format!(
            "Bias tensor {} exposes {} byte(s); {} output channels require {} INT32 byte(s).",
            tensor.index,
            raw.len(),
            channels,
            expected
        ));
    }
    let values = raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok((values, Some(tensor), "stored_int32_bias".to_string()))
}

fn required_signed_bits(minimum: i128, maximum: i128) -> usize {
    for bits in 1..128 {
        let magnitude = 1i128 << (bits - 1);
        if minimum >= -magnitude && maximum < magnitude {
            return bits;
        }
    }
    128
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| "Weight element count exceeds the analyzer integer range.".to_string())
    })
}

pub(super) fn raw_code(byte: u8, dtype: &str) -> i64 {
    if dtype == "INT8" {
        (byte as i8) as i64
    } else {
        byte as i64
    }
}

pub(super) fn quantized_code_range(dtype: &str) -> Option<(i64, i64)> {
    match dtype {
        "INT8" => Some((-128, 127)),
        "UINT8" => Some((0, 255)),
        _ => None,
    }
}

fn abs_max(minimum: i128, maximum: i128) -> i128 {
    minimum.abs().max(maximum.abs())
}

fn classify_bias_half_range_excess(excess_codes: i128) -> &'static str {
    if excess_codes <= 0 {
        "within_source_threshold"
    } else if excess_codes <= BIAS_HALF_RANGE_FLOAT32_TOLERANCE_CODES {
        "float32_guard_adjacent"
    } else {
        "material_exceedance"
    }
}

fn is_candidate(name: &str) -> bool {
    matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED")
}

fn tensor_at(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    usize::try_from(index)
        .ok()
        .and_then(|position| tensors.get(position))
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn decimal_abs_cmp_desc(left: &str, right: &str) -> Ordering {
    let left = left.parse::<i128>().unwrap_or(0).abs();
    let right = right.parse::<i128>().unwrap_or(0).abs();
    right.cmp(&left)
}

fn source_file(op_name: &str, dtype: Option<&str>) -> String {
    let int8 = dtype == Some("INT8");
    match (op_name, int8) {
        ("CONV_2D", true) => "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h",
        ("DEPTHWISE_CONV_2D", true) => {
            "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
        }
        ("FULLY_CONNECTED", true) => {
            "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
        }
        ("CONV_2D", false) => "tensorflow/lite/kernels/internal/reference/conv.h",
        ("DEPTHWISE_CONV_2D", false) => {
            "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
        }
        ("FULLY_CONNECTED", false) => {
            "tensorflow/lite/kernels/internal/reference/fully_connected.h"
        }
        _ => "",
    }
    .to_string()
}

fn source_url(op_name: &str, dtype: Option<&str>) -> String {
    let file = source_file(op_name, dtype);
    if file.is_empty() {
        String::new()
    } else {
        format!(
            "https://github.com/tensorflow/tensorflow/blob/{}/{}",
            TFLITE_SOURCE_COMMIT, file
        )
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_bit_width_observes_asymmetric_positive_limit() {
        assert_eq!(required_signed_bits(0, 0), 1);
        assert_eq!(required_signed_bits(-1, 0), 1);
        assert_eq!(required_signed_bits(-128, 127), 8);
        assert_eq!(required_signed_bits(-128, 128), 9);
        assert_eq!(required_signed_bits(i32::MIN as i128, i32::MAX as i128), 32);
    }

    #[test]
    fn half_range_classification_separates_float32_guard_adjacency() {
        assert_eq!(
            classify_bias_half_range_excess(0),
            "within_source_threshold"
        );
        assert_eq!(classify_bias_half_range_excess(1), "float32_guard_adjacent");
        assert_eq!(
            classify_bias_half_range_excess(129),
            "float32_guard_adjacent"
        );
        assert_eq!(classify_bias_half_range_excess(130), "material_exceedance");
    }

    #[test]
    fn quantized_mobilenet_accumulator_atlas_is_channel_exact() {
        let analysis = super::super::analyze_with_target(
            include_bytes!("../web/samples/mobilenet_v2_1.0_224_quant.tflite"),
            "mobilenet_v2_1.0_224_quant.tflite",
            "android_mid_a55",
        )
        .expect("sample should parse");
        let atlas = analysis.accumulator_atlas;
        assert_eq!(atlas.schema, ACCUMULATOR_ATLAS_SCHEMA);
        assert_eq!(atlas.candidate_op_count, 53);
        assert_eq!(atlas.assessed_op_count, 53);
        assert_eq!(atlas.unassessed_op_count, 0);
        assert_eq!(atlas.int32_overflow_channel_count, 0);
        assert_eq!(atlas.bias_half_range_exceedance_channel_count, 0);
        assert_eq!(atlas.bias_half_range_exceedance_op_count, 0);
        assert_eq!(atlas.bias_half_range_guard_adjacent_channel_count, 0);
        assert_eq!(atlas.bias_half_range_material_exceedance_channel_count, 0);
        assert_eq!(atlas.bias_half_range_material_exceedance_op_count, 0);
        assert!(
            atlas.exact_zero_bias_half_range_exceedance_channel_count
                <= atlas.exact_zero_kernel_channel_count
        );
        assert!(atlas.stored_bias_channel_count <= atlas.assessed_channel_count);
        assert_eq!(
            atlas.required_signed_bits_histogram.iter().sum::<usize>(),
            atlas.assessed_channel_count
        );
        for row in atlas.ops {
            assert_eq!(row.assessment_status, "assessed");
            assert_eq!(
                row.channel_accumulator_envelope_min_decimals.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.channel_accumulator_envelope_max_decimals.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.channel_post_bias_min_decimals.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.channel_post_bias_max_decimals.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.channel_required_signed_bits.len(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.required_signed_bits_histogram.iter().sum::<usize>(),
                row.assessed_channel_count
            );
            assert_eq!(
                row.bias_half_range_exceedance_channel_indices.len(),
                row.bias_half_range_exceedance_channel_count
            );
            assert_eq!(
                row.bias_half_range_guard_adjacent_channel_count
                    + row.bias_half_range_material_exceedance_channel_count,
                row.bias_half_range_exceedance_channel_count
            );
            assert_eq!(
                row.bias_half_range_material_exceedance_channel_indices
                    .len(),
                row.bias_half_range_material_exceedance_channel_count
            );
            assert!(
                row.exact_zero_bias_half_range_exceedance_channel_count
                    <= row.exact_zero_kernel_channel_count
            );
            assert!(row.stored_bias_channel_count <= row.assessed_channel_count);
            assert_eq!(row.channel_ledger_sha256.len(), 64);
        }
    }
}
