use super::accumulator_atlas::{
    decode_bias, expanded_weight_zero_points, quantized_code_range, raw_code, weight_layout,
};
use super::accumulator_reachability::{
    construct_input_codes_for_accumulator, AccumulatorReachabilityAnalysis,
    AccumulatorReachabilityOpRow,
};
use super::numerical_abi_propagation::{NumericalAbiPropagationAnalysis, PropagationSource};
use super::requantization_fidelity::{RequantizationFidelityAnalysis, RequantizationOpRow};
use super::research::{conv_out_size, op_options_table, parse_conv2d_opts, parse_dw_conv_opts};
use super::rounding_equivalence::{
    rebuild_channel_segments, RoundingEquivalenceAnalysis, RoundingEquivalenceOpRow,
};
use super::{extract_tensor_buffer, Fb, OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const SCHEMA: &str = "deepbom.input_counterexample.v1";
const METHOD_VERSION: &str = "2026-07-18.3";
const WITNESS_LEDGER_PREFIX: &str = "deepbom.input_counterexample.witness.v1\0";
const PORTFOLIO_LEDGER_PREFIX: &str = "deepbom.input_counterexample.portfolio.v1\0";

#[derive(Clone, Serialize)]
struct InputCounterexampleSource {
    op_index: usize,
    op_name: String,
    input_tensor_index: Option<usize>,
    input_tensor_name: String,
    input_origin: &'static str,
    classification: &'static str,
    assessment_reason: String,
    exact_reachable_divergent_channel_count: usize,
    exact_reachable_divergent_state_count_decimal: String,
    reachable_model_output_tensor_count: usize,
    exact_model_output_graph_route_count_decimal: Option<String>,
    representative_witness_index: Option<usize>,
    source_reachability_ledger_sha256: String,
    source_propagation_ledger_sha256: String,
    representative_witness_ledger_sha256: String,
}

#[derive(Clone, Serialize)]
pub(super) struct SparseInputOverride {
    pub(super) input_linear_index: usize,
    pub(super) input_code: i64,
}

#[derive(Clone, Serialize)]
struct InputWitnessTerm {
    term_index: usize,
    kernel_coordinate: Vec<usize>,
    input_coordinate: Vec<usize>,
    input_linear_index: usize,
    input_code: i64,
    centered_input_code: i64,
    centered_weight: i64,
    term_product_decimal: String,
}

#[derive(Clone, Serialize)]
struct CodeCount {
    code: i64,
    count: usize,
}

#[derive(Clone, Serialize)]
pub(super) struct TensorAbiInputWitness {
    pub(super) source_op_index: usize,
    source_op_name: String,
    source_output_tensor_index: usize,
    source_channel_index: usize,
    source_output_coordinate: Vec<usize>,
    pub(super) model_input_tensor_index: usize,
    pub(super) model_input_tensor_name: String,
    pub(super) model_input_shape: Vec<i32>,
    pub(super) model_input_dtype: String,
    pub(super) model_input_scale: f64,
    pub(super) model_input_zero_point: i64,
    pub(super) model_input_code_range: [i64; 2],
    pub(super) model_input_element_count: usize,
    pub(super) full_tensor_fill_code: i64,
    full_tensor_fill_policy: &'static str,
    sparse_override_count: usize,
    pub(super) sparse_overrides: Vec<SparseInputOverride>,
    pub(super) full_model_input_tensor_sha256: String,
    kernel_shape: Vec<usize>,
    effective_patch_shape: Vec<usize>,
    patch_origin_yx: Vec<usize>,
    patch_codes_hwc: Vec<i64>,
    stride_hw: Vec<usize>,
    dilation_hw: Vec<usize>,
    padding: &'static str,
    full_valid_receptive_field: bool,
    terms: Vec<InputWitnessTerm>,
    input_code_histogram: Vec<CodeCount>,
    dot_product_decimal: String,
    bias_decimal: String,
    post_bias_accumulator_decimal: String,
    default_output_code: i64,
    single_rounding_output_code: i64,
    output_code_delta: i64,
    source_exact_reachable_divergent_state_count_decimal: String,
    source_reachability_ledger_sha256: String,
    source_propagation_ledger_sha256: String,
    pub(super) witness_ledger_sha256: String,
}

#[derive(Clone, Serialize)]
pub(super) struct InputCounterexampleAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    exact_local_source_op_count: usize,
    direct_model_input_source_op_count: usize,
    tensor_abi_constructive_source_op_count: usize,
    upstream_activation_unresolved_source_op_count: usize,
    not_assessed_source_op_count: usize,
    tensor_abi_constructive_channel_count: usize,
    tensor_abi_constructive_divergent_state_count_decimal: String,
    output_reachable_constructive_source_op_count: usize,
    representative_witness_count: usize,
    source_classification_conservation: String,
    pub(super) portfolio_ledger_sha256: String,
    sources: Vec<InputCounterexampleSource>,
    pub(super) witnesses: Vec<TensorAbiInputWitness>,
    source_evidence_schema: &'static str,
    method: &'static str,
    proof_scope: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn input_counterexample_not_computed() -> InputCounterexampleAnalysis {
    build_analysis(Vec::new(), Vec::new(), "not_computed_internal_scope")
}

pub(super) fn build_input_counterexamples(
    model_bytes: &[u8],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    model_input_tensor_indices: &[i32],
    evidence: (
        &AccumulatorReachabilityAnalysis,
        &RoundingEquivalenceAnalysis,
        &RequantizationFidelityAnalysis,
        &NumericalAbiPropagationAnalysis,
    ),
) -> InputCounterexampleAnalysis {
    let (reachability, equivalence, requantization, propagation) = evidence;
    let exact_rows = reachability
        .ops
        .iter()
        .filter(|row| decimal(&row.exact_reachable_divergent_state_count_decimal) > 0)
        .collect::<Vec<_>>();
    let mut sources = Vec::with_capacity(exact_rows.len());
    let mut witnesses = Vec::new();
    for reachability_row in exact_rows {
        let op = ops.iter().find(|op| op.index == reachability_row.op_index);
        let input = op
            .and_then(|op| op.inputs.first().copied())
            .and_then(|index| tensor_at(tensors, index));
        let propagation_row = propagation
            .sources
            .iter()
            .find(|row| row.op_index == reachability_row.op_index);
        let is_model_input = input
            .map(|tensor| model_input_tensor_indices.contains(&(tensor.index as i32)))
            .unwrap_or(false);
        let mut classification = if is_model_input {
            "direct_model_input_not_constructed"
        } else {
            "upstream_activation_constraint_unresolved"
        };
        let mut reason = if is_model_input {
            "The source consumes a declared model-input tensor, but no representative witness has been constructed yet."
                .to_string()
        } else {
            "The source consumes an intermediate activation; exact inversion through upstream nonlinear and quantized operators is not established."
                .to_string()
        };
        let mut witness_index = None;
        let mut witness_ledger = String::new();
        if is_model_input {
            let equivalence_row = equivalence
                .ops
                .iter()
                .find(|row| row.op_index == reachability_row.op_index);
            let requantization_row = requantization
                .ops
                .iter()
                .find(|row| row.op_index == reachability_row.op_index);
            match (
                op,
                input,
                equivalence_row,
                requantization_row,
                propagation_row,
            ) {
                (
                    Some(op),
                    Some(input),
                    Some(equivalence_row),
                    Some(requantization_row),
                    Some(propagation_row),
                ) => {
                    match construct_tensor_abi_witness(
                        model_bytes,
                        op,
                        input,
                        tensors,
                        reachability_row,
                        equivalence_row,
                        requantization_row,
                        propagation_row,
                    ) {
                        Ok(witness) => {
                            classification = "tensor_abi_constructive";
                            reason = "A complete model-input tensor is constructively specified at the tensor ABI and reproduces an exact local rounding divergence."
                                .to_string();
                            witness_index = Some(witnesses.len());
                            witness_ledger = witness.witness_ledger_sha256.clone();
                            witnesses.push(witness);
                        }
                        Err(error) => reason = error,
                    }
                }
                _ => {
                    reason = "Required graph, rounding, reachability, requantization, or propagation evidence is unavailable."
                        .to_string();
                }
            }
        }
        sources.push(InputCounterexampleSource {
            op_index: reachability_row.op_index,
            op_name: op
                .map(|op| op.name.clone())
                .unwrap_or_else(|| reachability_row.op_name.clone()),
            input_tensor_index: input.map(|tensor| tensor.index),
            input_tensor_name: input.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            input_origin: if is_model_input {
                "declared_model_input"
            } else if input.is_some() {
                "intermediate_activation"
            } else {
                "not_assessed"
            },
            classification,
            assessment_reason: reason,
            exact_reachable_divergent_channel_count: reachability_row
                .exact_reachable_divergent_channel_count,
            exact_reachable_divergent_state_count_decimal: reachability_row
                .exact_reachable_divergent_state_count_decimal
                .clone(),
            reachable_model_output_tensor_count: propagation_row
                .map(|row| row.reachable_model_output_tensor_count)
                .unwrap_or(0),
            exact_model_output_graph_route_count_decimal: propagation_row
                .and_then(|row| row.exact_model_output_graph_route_count_decimal.clone()),
            representative_witness_index: witness_index,
            source_reachability_ledger_sha256: reachability_row.reachability_ledger_sha256.clone(),
            source_propagation_ledger_sha256: propagation_row
                .map(|row| row.propagation_ledger_sha256.clone())
                .unwrap_or_default(),
            representative_witness_ledger_sha256: witness_ledger,
        });
    }
    let status = if sources.is_empty() {
        "not_applicable"
    } else if sources
        .iter()
        .any(|source| source.classification == "direct_model_input_not_constructed")
    {
        "partial"
    } else {
        "assessed"
    };
    build_analysis(sources, witnesses, status)
}

fn build_analysis(
    sources: Vec<InputCounterexampleSource>,
    witnesses: Vec<TensorAbiInputWitness>,
    status: &'static str,
) -> InputCounterexampleAnalysis {
    let direct = sources
        .iter()
        .filter(|source| source.input_origin == "declared_model_input")
        .count();
    let constructive = sources
        .iter()
        .filter(|source| source.classification == "tensor_abi_constructive")
        .collect::<Vec<_>>();
    let upstream = sources
        .iter()
        .filter(|source| source.classification == "upstream_activation_constraint_unresolved")
        .count();
    let not_assessed = sources.len().saturating_sub(constructive.len() + upstream);
    let constructive_states = constructive
        .iter()
        .map(|source| decimal(&source.exact_reachable_divergent_state_count_decimal))
        .sum::<u128>();
    let constructive_channels = constructive
        .iter()
        .map(|source| source.exact_reachable_divergent_channel_count)
        .sum::<usize>();
    let output_reachable = constructive
        .iter()
        .filter(|source| source.reachable_model_output_tensor_count > 0)
        .count();
    let portfolio_ledger_sha256 = portfolio_ledger(&sources);
    InputCounterexampleAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        exact_local_source_op_count: sources.len(),
        direct_model_input_source_op_count: direct,
        tensor_abi_constructive_source_op_count: constructive.len(),
        upstream_activation_unresolved_source_op_count: upstream,
        not_assessed_source_op_count: not_assessed,
        tensor_abi_constructive_channel_count: constructive_channels,
        tensor_abi_constructive_divergent_state_count_decimal: constructive_states.to_string(),
        output_reachable_constructive_source_op_count: output_reachable,
        representative_witness_count: witnesses.len(),
        source_classification_conservation: format!(
            "{} = {} constructive + {} upstream-unresolved + {} not-assessed",
            sources.len(),
            constructive.len(),
            upstream,
            not_assessed
        ),
        portfolio_ledger_sha256,
        sources,
        witnesses,
        source_evidence_schema:
            "deepbom.accumulator_reachability.v1 + deepbom.numerical_abi_propagation.v1.1",
        method: "For every exact-local divergent source, classify whether its activation input is a declared model input. For direct quantized Conv/Depthwise sources, construct a bounded-sum term assignment for the first ranked exact divergent accumulator, embed it in one full-valid receptive field, fill every other model-input element with the tensor zero point, and hash the complete raw tensor. Recompute dot product, bias, and both pinned output codes before issuing the certificate.",
        proof_scope: "A tensor_abi_constructive source proves existence of a complete quantized model-input tensor that produces the certified source-op output-code difference at one full-valid output coordinate under the pinned reference integer equations.",
        interpretation_boundary: "The witness is exact at the model tensor ABI, not necessarily realizable through an application's image/audio preprocessing contract. Its downstream graph corridor remains structural: this certificate does not prove a declared model output changes, identify the deployed runtime build flag or lowering, measure activation frequency, or imply task-accuracy impact.",
    }
}

#[allow(clippy::too_many_arguments)]
fn construct_tensor_abi_witness(
    model_bytes: &[u8],
    op: &OpInfo,
    input: &TensorInfo,
    tensors: &[TensorInfo],
    reachability: &AccumulatorReachabilityOpRow,
    equivalence: &RoundingEquivalenceOpRow,
    requantization: &RequantizationOpRow,
    propagation: &PropagationSource,
) -> Result<TensorAbiInputWitness, String> {
    if !matches!(op.name.as_str(), "CONV_2D" | "DEPTHWISE_CONV_2D") {
        return Err(format!(
            "Direct-input {} is not yet a spatial integer-kernel witness candidate.",
            op.name
        ));
    }
    let weight = tensor_at(tensors, op.inputs.get(1).copied().unwrap_or(-1))
        .ok_or_else(|| "Weight tensor is unavailable.".to_string())?;
    let output = tensor_at(tensors, op.outputs.first().copied().unwrap_or(-1))
        .ok_or_else(|| "Source output tensor is unavailable.".to_string())?;
    let [qmin, qmax] = quantized_code_range(&input.dtype)
        .map(|(minimum, maximum)| [minimum, maximum])
        .ok_or_else(|| "Model input is not INT8 or UINT8.".to_string())?;
    let input_zero_point = single_zero_point(input)?;
    let input_scale = single_scale(input)?;
    let layout = weight_layout(op, weight)?;
    let channels = layout.channels();
    let terms_per_channel = layout.terms();
    let channel = ranked_exact_channel(reachability)?;
    if channel >= channels {
        return Err("Ranked exact channel escapes the stored-weight layout.".to_string());
    }
    let target_accumulator = reachability
        .channel_first_exact_reachable_divergent_accumulators_decimal
        .get(channel)
        .and_then(|value| value.as_ref())
        .ok_or_else(|| "Ranked exact channel has no constructive accumulator.".to_string())?
        .parse::<i32>()
        .map_err(|_| "Constructive accumulator is not int32.".to_string())?;
    let weight_range = quantized_code_range(&weight.dtype)
        .ok_or_else(|| "Weight tensor is not INT8 or UINT8.".to_string())?;
    let zero_points =
        expanded_weight_zero_points(weight, channels, weight_range.0, weight_range.1)?;
    let raw_weights = extract_tensor_buffer(model_bytes, weight)
        .ok_or_else(|| "Stored weight bytes are unavailable.".to_string())?;
    if raw_weights.len() != channels.saturating_mul(terms_per_channel) {
        return Err("Stored weight bytes do not match the exact channel layout.".to_string());
    }
    let centered_weights = (0..terms_per_channel)
        .map(|term| {
            raw_code(raw_weights[layout.raw_index(channel, term)], &weight.dtype)
                - zero_points[channel]
        })
        .collect::<Vec<_>>();
    let (biases, _, _) = decode_bias(model_bytes, op, tensors, channels)?;
    let constructed = construct_input_codes_for_accumulator(
        &centered_weights,
        qmin,
        qmax,
        input_zero_point,
        biases[channel],
        target_accumulator,
    )?;
    let segments = rebuild_channel_segments(equivalence, requantization, channel)?;
    let segment = segments
        .iter()
        .find(|segment| {
            target_accumulator >= segment.accumulator_minimum
                && target_accumulator <= segment.accumulator_maximum
        })
        .ok_or_else(|| {
            "Constructive accumulator is absent from the rounding partition.".to_string()
        })?;
    if segment.default_output_code == segment.single_output_code {
        return Err("Constructive accumulator does not produce a rounding divergence.".to_string());
    }
    let geometry = spatial_geometry(model_bytes, op, input, output, weight, channel)?;
    if geometry.term_coordinates.len() != terms_per_channel {
        return Err("Spatial geometry does not conserve kernel term count.".to_string());
    }
    let shape = static_shape(&input.shape)?;
    let element_count = checked_product(&shape)?;
    let mut full_codes = vec![input_zero_point; element_count];
    let mut terms = Vec::with_capacity(terms_per_channel);
    for (term, centered_weight) in centered_weights
        .iter()
        .copied()
        .enumerate()
        .take(terms_per_channel)
    {
        let coordinate = geometry.term_coordinates[term].clone();
        let linear = nhwc_linear_index(&shape, &coordinate)?;
        let code = constructed.codes[term];
        full_codes[linear] = code;
        let product = constructed.centered_inputs[term]
            .checked_mul(centered_weight)
            .ok_or_else(|| "Witness term product exceeds i64.".to_string())?;
        terms.push(InputWitnessTerm {
            term_index: term,
            kernel_coordinate: geometry.kernel_coordinates[term].clone(),
            input_coordinate: coordinate,
            input_linear_index: linear,
            input_code: code,
            centered_input_code: constructed.centered_inputs[term],
            centered_weight,
            term_product_decimal: product.to_string(),
        });
    }
    let sparse_overrides = full_codes
        .iter()
        .enumerate()
        .filter(|(_, code)| **code != input_zero_point)
        .map(|(input_linear_index, input_code)| SparseInputOverride {
            input_linear_index,
            input_code: *input_code,
        })
        .collect::<Vec<_>>();
    let raw_input = encode_tensor_codes(&full_codes, &input.dtype)?;
    let full_tensor_sha256 = hex_digest(Sha256::digest(&raw_input).as_slice());
    let patch_codes_hwc = extract_patch_codes(&full_codes, &shape, &geometry)?;
    let histogram = code_histogram(&constructed.codes);
    let mut witness = TensorAbiInputWitness {
        source_op_index: op.index,
        source_op_name: op.name.clone(),
        source_output_tensor_index: output.index,
        source_channel_index: channel,
        source_output_coordinate: geometry.output_coordinate,
        model_input_tensor_index: input.index,
        model_input_tensor_name: input.name.clone(),
        model_input_shape: input.shape.clone(),
        model_input_dtype: input.dtype.clone(),
        model_input_scale: input_scale,
        model_input_zero_point: input_zero_point,
        model_input_code_range: [qmin, qmax],
        model_input_element_count: element_count,
        full_tensor_fill_code: input_zero_point,
        full_tensor_fill_policy: "fill_every_element_with_zero_point_then_apply_sparse_overrides",
        sparse_override_count: sparse_overrides.len(),
        sparse_overrides,
        full_model_input_tensor_sha256: full_tensor_sha256,
        kernel_shape: geometry.kernel_shape,
        effective_patch_shape: geometry.effective_patch_shape,
        patch_origin_yx: geometry.patch_origin_yx,
        patch_codes_hwc,
        stride_hw: geometry.stride_hw,
        dilation_hw: geometry.dilation_hw,
        padding: geometry.padding,
        full_valid_receptive_field: true,
        terms,
        input_code_histogram: histogram,
        dot_product_decimal: constructed.dot_product.to_string(),
        bias_decimal: constructed.bias.to_string(),
        post_bias_accumulator_decimal: constructed.post_bias_accumulator.to_string(),
        default_output_code: segment.default_output_code,
        single_rounding_output_code: segment.single_output_code,
        output_code_delta: segment.default_output_code - segment.single_output_code,
        source_exact_reachable_divergent_state_count_decimal: reachability
            .exact_reachable_divergent_state_count_decimal
            .clone(),
        source_reachability_ledger_sha256: reachability.reachability_ledger_sha256.clone(),
        source_propagation_ledger_sha256: propagation.propagation_ledger_sha256.clone(),
        witness_ledger_sha256: String::new(),
    };
    witness.witness_ledger_sha256 = witness_ledger(&witness);
    Ok(witness)
}

struct SpatialGeometry {
    kernel_shape: Vec<usize>,
    effective_patch_shape: Vec<usize>,
    patch_origin_yx: Vec<usize>,
    stride_hw: Vec<usize>,
    dilation_hw: Vec<usize>,
    padding: &'static str,
    output_coordinate: Vec<usize>,
    kernel_coordinates: Vec<Vec<usize>>,
    term_coordinates: Vec<Vec<usize>>,
}

fn spatial_geometry(
    model_bytes: &[u8],
    op: &OpInfo,
    input: &TensorInfo,
    output: &TensorInfo,
    weight: &TensorInfo,
    output_channel: usize,
) -> Result<SpatialGeometry, String> {
    let input_shape = static_shape(&input.shape)?;
    let output_shape = static_shape(&output.shape)?;
    let weight_shape = static_shape(&weight.shape)?;
    if input_shape.len() != 4 || output_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(
            "Spatial witness requires rank-4 NHWC input/output and rank-4 weights.".to_string(),
        );
    }
    if input_shape[0] != 1 || output_shape[0] != 1 {
        return Err("Spatial witness currently requires static batch size one.".to_string());
    }
    let fb = Fb::verified_tflite(model_bytes)?;
    let model = fb.root_table()?;
    let subgraph = fb
        .vector_tables(model, 2)
        .first()
        .copied()
        .ok_or_else(|| "Model subgraph is unavailable.".to_string())?;
    let op_table = fb
        .vector_tables(subgraph, 3)
        .get(op.index)
        .copied()
        .ok_or_else(|| "Operator table is unavailable.".to_string())?;
    let options = op_options_table(&fb, op_table);
    let (same, stride_h, stride_w, dilation_h, dilation_w, _) = match op.name.as_str() {
        "CONV_2D" => parse_conv2d_opts(&fb, options),
        "DEPTHWISE_CONV_2D" => parse_dw_conv_opts(&fb, options),
        _ => return Err("Operator is not a supported spatial kernel.".to_string()),
    };
    let kernel_h = weight_shape[1];
    let kernel_w = weight_shape[2];
    let effective_h = (kernel_h - 1)
        .checked_mul(dilation_h)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "Effective kernel height exceeds usize.".to_string())?;
    let effective_w = (kernel_w - 1)
        .checked_mul(dilation_w)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "Effective kernel width exceeds usize.".to_string())?;
    let (computed_h, pad_top) = conv_out_size(input_shape[1], kernel_h, stride_h, dilation_h, same);
    let (computed_w, pad_left) =
        conv_out_size(input_shape[2], kernel_w, stride_w, dilation_w, same);
    if computed_h != output_shape[1] || computed_w != output_shape[2] {
        return Err(format!(
            "Parsed convolution geometry does not reproduce the declared output shape: padding={}, stride={}x{}, dilation={}x{}, input={}x{}, kernel={}x{}, computed={}x{}, declared={}x{}.",
            if same { "SAME" } else { "VALID" },
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            input_shape[1],
            input_shape[2],
            kernel_h,
            kernel_w,
            computed_h,
            computed_w,
            output_shape[1],
            output_shape[2]
        ));
    }
    let (output_y, origin_y) = full_valid_axis(
        output_shape[1],
        input_shape[1],
        effective_h,
        stride_h,
        pad_top,
    )
    .ok_or_else(|| "No full-valid output row exists for this spatial kernel.".to_string())?;
    let (output_x, origin_x) = full_valid_axis(
        output_shape[2],
        input_shape[2],
        effective_w,
        stride_w,
        pad_left,
    )
    .ok_or_else(|| "No full-valid output column exists for this spatial kernel.".to_string())?;
    let mut kernel_coordinates = Vec::new();
    let mut term_coordinates = Vec::new();
    match op.name.as_str() {
        "CONV_2D" => {
            if weight_shape[3] != input_shape[3] {
                return Err(
                    "CONV_2D input-channel cardinality does not match OHWI weights.".to_string(),
                );
            }
            for kernel_y in 0..kernel_h {
                for kernel_x in 0..kernel_w {
                    for input_channel in 0..input_shape[3] {
                        kernel_coordinates.push(vec![kernel_y, kernel_x, input_channel]);
                        term_coordinates.push(vec![
                            0,
                            origin_y + kernel_y * dilation_h,
                            origin_x + kernel_x * dilation_w,
                            input_channel,
                        ]);
                    }
                }
            }
        }
        "DEPTHWISE_CONV_2D" => {
            if weight_shape[0] != 1 || output_shape[3] % input_shape[3] != 0 {
                return Err("DEPTHWISE_CONV_2D channel geometry is invalid.".to_string());
            }
            let depth_multiplier = output_shape[3] / input_shape[3];
            let input_channel = output_channel / depth_multiplier;
            for kernel_y in 0..kernel_h {
                for kernel_x in 0..kernel_w {
                    kernel_coordinates.push(vec![kernel_y, kernel_x, input_channel]);
                    term_coordinates.push(vec![
                        0,
                        origin_y + kernel_y * dilation_h,
                        origin_x + kernel_x * dilation_w,
                        input_channel,
                    ]);
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(SpatialGeometry {
        kernel_shape: vec![kernel_h, kernel_w, input_shape[3]],
        effective_patch_shape: vec![effective_h, effective_w, input_shape[3]],
        patch_origin_yx: vec![origin_y, origin_x],
        stride_hw: vec![stride_h, stride_w],
        dilation_hw: vec![dilation_h, dilation_w],
        padding: if same { "SAME" } else { "VALID" },
        output_coordinate: vec![0, output_y, output_x, output_channel],
        kernel_coordinates,
        term_coordinates,
    })
}

fn full_valid_axis(
    output_size: usize,
    input_size: usize,
    effective_kernel: usize,
    stride: usize,
    leading_padding: usize,
) -> Option<(usize, usize)> {
    (0..output_size).find_map(|output| {
        let unpadded = output.checked_mul(stride)?;
        let origin = unpadded.checked_sub(leading_padding)?;
        (origin.checked_add(effective_kernel)? <= input_size).then_some((output, origin))
    })
}

fn ranked_exact_channel(row: &AccumulatorReachabilityOpRow) -> Result<usize, String> {
    row.channel_exact_reachable_divergent_state_counts_decimal
        .iter()
        .enumerate()
        .filter_map(|(channel, count)| {
            let count = count.parse::<u128>().ok()?;
            let accumulator = row
                .channel_first_exact_reachable_divergent_accumulators_decimal
                .get(channel)
                .and_then(|value| value.as_ref())?;
            (count > 0).then_some((channel, count, accumulator))
        })
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|value| value.0)
        .ok_or_else(|| "No exact divergent channel has a constructive accumulator.".to_string())
}

fn extract_patch_codes(
    full_codes: &[i64],
    input_shape: &[usize],
    geometry: &SpatialGeometry,
) -> Result<Vec<i64>, String> {
    let mut patch = Vec::with_capacity(checked_product(&geometry.effective_patch_shape)?);
    for patch_y in 0..geometry.effective_patch_shape[0] {
        for patch_x in 0..geometry.effective_patch_shape[1] {
            for channel in 0..geometry.effective_patch_shape[2] {
                let coordinate = [
                    0,
                    geometry.patch_origin_yx[0] + patch_y,
                    geometry.patch_origin_yx[1] + patch_x,
                    channel,
                ];
                patch.push(full_codes[nhwc_linear_index(input_shape, &coordinate)?]);
            }
        }
    }
    Ok(patch)
}

fn witness_ledger(witness: &TensorAbiInputWitness) -> String {
    let mut canonical = String::from(WITNESS_LEDGER_PREFIX);
    canonical.push_str(&format!(
        "reachability={}\npropagation={}\n",
        witness.source_reachability_ledger_sha256, witness.source_propagation_ledger_sha256
    ));
    canonical.push_str(&format!(
        "op={};channel={};input={};output={};target={};default={};single={};dot={};bias={};tensor_sha={}\n",
        witness.source_op_index,
        witness.source_channel_index,
        witness.model_input_tensor_index,
        witness.source_output_tensor_index,
        witness.post_bias_accumulator_decimal,
        witness.default_output_code,
        witness.single_rounding_output_code,
        witness.dot_product_decimal,
        witness.bias_decimal,
        witness.full_model_input_tensor_sha256
    ));
    canonical.push_str(&format!(
        "fill={};elements={};shape={};output_coordinate={};patch_origin={};patch_shape={}\n",
        witness.full_tensor_fill_code,
        witness.model_input_element_count,
        join_i32(&witness.model_input_shape),
        join_usize(&witness.source_output_coordinate),
        join_usize(&witness.patch_origin_yx),
        join_usize(&witness.effective_patch_shape)
    ));
    for term in &witness.terms {
        canonical.push_str(&format!(
            "term={};linear={};code={};centered={};weight={};product={}\n",
            term.term_index,
            term.input_linear_index,
            term.input_code,
            term.centered_input_code,
            term.centered_weight,
            term.term_product_decimal
        ));
    }
    for value in &witness.sparse_overrides {
        canonical.push_str(&format!(
            "override={};code={}\n",
            value.input_linear_index, value.input_code
        ));
    }
    hex_digest(Sha256::digest(canonical.as_bytes()).as_slice())
}

fn portfolio_ledger(sources: &[InputCounterexampleSource]) -> String {
    let mut canonical = String::from(PORTFOLIO_LEDGER_PREFIX);
    for source in sources {
        canonical.push_str(&format!(
            "source={};class={};channels={};states={};outputs={};routes={};reachability={};propagation={};witness={}\n",
            source.op_index,
            source.classification,
            source.exact_reachable_divergent_channel_count,
            source.exact_reachable_divergent_state_count_decimal,
            source.reachable_model_output_tensor_count,
            source
                .exact_model_output_graph_route_count_decimal
                .as_deref()
                .unwrap_or("none"),
            source.source_reachability_ledger_sha256,
            source.source_propagation_ledger_sha256,
            source.representative_witness_ledger_sha256
        ));
    }
    hex_digest(Sha256::digest(canonical.as_bytes()).as_slice())
}

fn encode_tensor_codes(codes: &[i64], dtype: &str) -> Result<Vec<u8>, String> {
    match dtype {
        "UINT8" => codes
            .iter()
            .map(|code| {
                u8::try_from(*code).map_err(|_| "UINT8 witness code is invalid.".to_string())
            })
            .collect(),
        "INT8" => codes
            .iter()
            .map(|code| {
                i8::try_from(*code)
                    .map(|value| value as u8)
                    .map_err(|_| "INT8 witness code is invalid.".to_string())
            })
            .collect(),
        _ => Err(format!("{dtype} input witness encoding is unsupported.")),
    }
}

fn code_histogram(codes: &[i64]) -> Vec<CodeCount> {
    let mut counts = BTreeMap::<i64, usize>::new();
    for code in codes {
        *counts.entry(*code).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(code, count)| CodeCount { code, count })
        .collect()
}

fn nhwc_linear_index(shape: &[usize], coordinate: &[usize]) -> Result<usize, String> {
    if shape.len() != 4 || coordinate.len() != 4 {
        return Err("NHWC coordinate rank is not four.".to_string());
    }
    if coordinate
        .iter()
        .zip(shape)
        .any(|(index, size)| index >= size)
    {
        return Err("Witness coordinate escapes the model-input tensor.".to_string());
    }
    coordinate[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(coordinate[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(coordinate[2]))
        .and_then(|value| value.checked_mul(shape[3]))
        .and_then(|value| value.checked_add(coordinate[3]))
        .ok_or_else(|| "NHWC linear index exceeds usize.".to_string())
}

fn static_shape(shape: &[i32]) -> Result<Vec<usize>, String> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| "Tensor shape is dynamic or non-positive.".to_string())
        })
        .collect()
}

fn checked_product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| "Tensor element count exceeds usize.".to_string())
    })
}

fn single_zero_point(tensor: &TensorInfo) -> Result<i64, String> {
    (tensor.zero_point_sample.len() == 1)
        .then_some(tensor.zero_point_sample[0])
        .ok_or_else(|| "Model input does not have one zero point.".to_string())
}

fn single_scale(tensor: &TensorInfo) -> Result<f64, String> {
    (tensor.scale_sample.len() == 1
        && tensor.scale_sample[0].is_finite()
        && tensor.scale_sample[0] > 0.0)
        .then_some(f64::from(tensor.scale_sample[0]))
        .ok_or_else(|| "Model input does not have one finite positive scale.".to_string())
}

fn tensor_at(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    usize::try_from(index)
        .ok()
        .and_then(|index| tensors.iter().find(|tensor| tensor.index == index))
}

fn decimal(value: &str) -> u128 {
    value.parse().unwrap_or(0)
}

fn join_i32(values: &[i32]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_valid_axis_skips_leading_padding() {
        assert_eq!(full_valid_axis(4, 7, 3, 2, 1), Some((1, 1)));
        assert_eq!(full_valid_axis(1, 2, 3, 1, 0), None);
    }

    #[test]
    fn nhwc_index_is_row_major() {
        assert_eq!(nhwc_linear_index(&[1, 2, 3, 4], &[0, 1, 2, 3]), Ok(23));
    }
}
