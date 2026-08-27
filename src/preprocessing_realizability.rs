use super::input_counterexample::{InputCounterexampleAnalysis, TensorAbiInputWitness};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const SCHEMA: &str = "deepbom.preprocessing_realizability.v1";
const METHOD_VERSION: &str = "2026-07-18.1";
const CANDIDATE_LEDGER_PREFIX: &str = "deepbom.preprocessing_realizability.candidate.v1\0";
const PORTFOLIO_LEDGER_PREFIX: &str = "deepbom.preprocessing_realizability.portfolio.v1\0";

#[derive(Clone, Serialize)]
struct PixelCodeMap {
    tensor_channel: usize,
    source_pixel_channel: usize,
    source_pixel_channel_name: &'static str,
    reachable_tensor_code_count: usize,
    tensor_code_hole_count: usize,
    collision_tensor_code_count: usize,
    maximum_preimage_multiplicity: usize,
    reachable_tensor_code_min: i64,
    reachable_tensor_code_max: i64,
    pixel_to_tensor_codes: Vec<i64>,
}

#[derive(Clone, Serialize)]
struct WitnessCodeRealization {
    tensor_channel: usize,
    source_pixel_channel: usize,
    target_tensor_code: i64,
    tensor_element_count: usize,
    exact_source_pixel_codes: Vec<u8>,
    selected_source_pixel_code: u8,
    roundtrip_tensor_code: i64,
    absolute_tensor_code_error: u64,
}

#[derive(Clone, Serialize)]
struct FirstUnrealizableElement {
    tensor_linear_index: usize,
    tensor_coordinate_nhwc: Vec<usize>,
    tensor_channel: usize,
    source_pixel_channel: usize,
    target_tensor_code: i64,
    selected_source_pixel_code: u8,
    roundtrip_tensor_code: i64,
    absolute_tensor_code_error: u64,
}

#[derive(Clone, Serialize)]
struct PreprocessingCandidate {
    witness_index: usize,
    source_op_index: usize,
    contract_id: &'static str,
    contract_label: &'static str,
    contract_family: &'static str,
    source_pixel_order: &'static str,
    tensor_channel_order: &'static str,
    pixel_to_real_formula: &'static str,
    real_to_tensor_formula: &'static str,
    rounding_mode: &'static str,
    status: &'static str,
    assessment_reason: String,
    source_image_shape_hwc: Vec<usize>,
    channel_maps: Vec<PixelCodeMap>,
    witness_code_realizations: Vec<WitnessCodeRealization>,
    distinct_witness_channel_code_pair_count: usize,
    exact_witness_channel_code_pair_count: usize,
    witness_tensor_element_count: usize,
    exact_tensor_element_count: usize,
    unrealizable_tensor_element_count: usize,
    exact_tensor_realization: bool,
    minimum_total_absolute_tensor_code_error_decimal: String,
    maximum_absolute_tensor_code_error: u64,
    first_unrealizable_element: Option<FirstUnrealizableElement>,
    fixture_encoding: &'static str,
    fixture_kind: &'static str,
    nearest_rgb_fixture_sha256: String,
    exact_rgb_fixture_sha256: Option<String>,
    source_input_witness_ledger_sha256: String,
    candidate_ledger_sha256: String,
}

#[derive(Serialize)]
pub(super) struct PreprocessingRealizabilityAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    assessment_kind: &'static str,
    status: &'static str,
    source_input_counterexample_schema: &'static str,
    source_input_counterexample_portfolio_sha256: String,
    source_witness_count: usize,
    eligible_image_witness_count: usize,
    ineligible_witness_count: usize,
    candidate_contract_count: usize,
    candidate_evaluation_count: usize,
    assessed_candidate_count: usize,
    exact_tensor_realization_candidate_count: usize,
    non_exact_candidate_count: usize,
    exact_contract_ids: Vec<&'static str>,
    best_non_exact_contract_id: String,
    best_non_exact_unrealizable_element_count: Option<usize>,
    candidate_conservation: String,
    portfolio_ledger_sha256: String,
    candidates: Vec<PreprocessingCandidate>,
    method: &'static str,
    proof_scope: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Clone, Copy)]
enum Transform {
    RawStorage,
    ArtifactAffine,
    Center128,
    MinusOneToOne,
    UnitInterval,
    ImageNet,
}

#[derive(Clone, Copy)]
struct ContractSpec {
    id: &'static str,
    label: &'static str,
    family: &'static str,
    tensor_order: &'static str,
    pixel_to_real: &'static str,
    real_to_tensor: &'static str,
    rounding_mode: &'static str,
    transform: Transform,
    tensor_to_source: [usize; 3],
    imagenet_mean_milli: [i128; 3],
    imagenet_std_milli: [i128; 3],
}

const CONTRACTS: [ContractSpec; 8] = [
    ContractSpec {
        id: "raw_storage_rgb",
        label: "Raw tensor storage / RGB",
        family: "direct_storage_assignment",
        tensor_order: "RGB",
        pixel_to_real: "tensor storage byte := source RGB byte; no source-domain normalization",
        real_to_tensor: "q = decode_storage_byte(pixel, input_dtype); source quantizer bypassed",
        rounding_mode: "not_applicable_direct_storage",
        transform: Transform::RawStorage,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "raw_storage_bgr",
        label: "Raw tensor storage / BGR",
        family: "direct_storage_assignment",
        tensor_order: "BGR",
        pixel_to_real:
            "tensor storage bytes := source RGB bytes with R/B permutation; no normalization",
        real_to_tensor: "q = decode_storage_byte(pixel, input_dtype); source quantizer bypassed",
        rounding_mode: "not_applicable_direct_storage",
        transform: Transform::RawStorage,
        tensor_to_source: [2, 1, 0],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "artifact_affine_rgb",
        label: "Artifact quantization affine / RGB",
        family: "artifact_affine_inverse",
        tensor_order: "RGB",
        pixel_to_real: "real = input_scale * ((pixel + qmin) - input_zero_point)",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::ArtifactAffine,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "center_128_div_128_rgb",
        label: "(pixel - 128) / 128 / RGB",
        family: "fixed_affine_normalization",
        tensor_order: "RGB",
        pixel_to_real: "real = (pixel - 128) / 128",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::Center128,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "minus_one_to_one_rgb",
        label: "[0,255] to [-1,1] / RGB",
        family: "fixed_affine_normalization",
        tensor_order: "RGB",
        pixel_to_real: "real = (2 * pixel - 255) / 255",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::MinusOneToOne,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "unit_interval_rgb",
        label: "[0,255] to [0,1] / RGB",
        family: "fixed_affine_normalization",
        tensor_order: "RGB",
        pixel_to_real: "real = pixel / 255",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::UnitInterval,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [0; 3],
        imagenet_std_milli: [1; 3],
    },
    ContractSpec {
        id: "imagenet_mean_std_rgb",
        label: "ImageNet mean/std / RGB",
        family: "channelwise_mean_std_normalization",
        tensor_order: "RGB",
        pixel_to_real: "real[c] = (pixel[c] / 255 - mean_rgb[c]) / std_rgb[c]",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::ImageNet,
        tensor_to_source: [0, 1, 2],
        imagenet_mean_milli: [485, 456, 406],
        imagenet_std_milli: [229, 224, 225],
    },
    ContractSpec {
        id: "imagenet_mean_std_bgr",
        label: "ImageNet mean/std / BGR tensor",
        family: "channelwise_mean_std_normalization",
        tensor_order: "BGR",
        pixel_to_real: "tensor BGR[c] = (source RGB[2-c] / 255 - mean_bgr[c]) / std_bgr[c]",
        real_to_tensor: "q = clamp(round_half_away_from_zero(real / input_scale) + input_zero_point, qmin, qmax)",
        rounding_mode: "nearest_ties_away_from_zero",
        transform: Transform::ImageNet,
        tensor_to_source: [2, 1, 0],
        imagenet_mean_milli: [406, 456, 485],
        imagenet_std_milli: [225, 224, 229],
    },
];

pub(super) fn preprocessing_realizability_not_computed() -> PreprocessingRealizabilityAnalysis {
    build_analysis(
        "not_computed_internal_scope",
        String::new(),
        0,
        0,
        Vec::new(),
    )
}

pub(super) fn build_preprocessing_realizability(
    input: &InputCounterexampleAnalysis,
) -> PreprocessingRealizabilityAnalysis {
    let mut candidates = Vec::new();
    let mut eligible = 0usize;
    for (witness_index, witness) in input.witnesses.iter().enumerate() {
        if !eligible_image_witness(witness) {
            continue;
        }
        eligible += 1;
        for contract in CONTRACTS {
            candidates.push(evaluate_candidate(witness_index, witness, contract));
        }
    }
    let status = if input.witnesses.is_empty() {
        "not_applicable"
    } else if eligible == 0 {
        "not_applicable_no_image_witness"
    } else if candidates
        .iter()
        .any(|candidate| candidate.status != "assessed")
    {
        "partial"
    } else {
        "assessed"
    };
    build_analysis(
        status,
        input.portfolio_ledger_sha256.clone(),
        input.witnesses.len(),
        eligible,
        candidates,
    )
}

fn build_analysis(
    status: &'static str,
    source_portfolio: String,
    source_witness_count: usize,
    eligible: usize,
    candidates: Vec<PreprocessingCandidate>,
) -> PreprocessingRealizabilityAnalysis {
    let assessed = candidates
        .iter()
        .filter(|candidate| candidate.status == "assessed")
        .collect::<Vec<_>>();
    let exact = assessed
        .iter()
        .filter(|candidate| candidate.exact_tensor_realization)
        .collect::<Vec<_>>();
    let non_exact = assessed.len().saturating_sub(exact.len());
    let best_non_exact = assessed
        .iter()
        .filter(|candidate| !candidate.exact_tensor_realization)
        .min_by(|left, right| {
            left.unrealizable_tensor_element_count
                .cmp(&right.unrealizable_tensor_element_count)
                .then_with(|| {
                    decimal(&left.minimum_total_absolute_tensor_code_error_decimal).cmp(&decimal(
                        &right.minimum_total_absolute_tensor_code_error_decimal,
                    ))
                })
                .then_with(|| left.contract_id.cmp(right.contract_id))
        });
    let portfolio_ledger_sha256 = portfolio_ledger(&source_portfolio, &candidates);
    PreprocessingRealizabilityAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        assessment_kind: "EXPLICIT_PREPROCESSING_COUNTERFACTUALS",
        status,
        source_input_counterexample_schema: "deepbom.input_counterexample.v1",
        source_input_counterexample_portfolio_sha256: source_portfolio,
        source_witness_count,
        eligible_image_witness_count: eligible,
        ineligible_witness_count: source_witness_count.saturating_sub(eligible),
        candidate_contract_count: CONTRACTS.len(),
        candidate_evaluation_count: candidates.len(),
        assessed_candidate_count: assessed.len(),
        exact_tensor_realization_candidate_count: exact.len(),
        non_exact_candidate_count: non_exact,
        exact_contract_ids: exact.iter().map(|candidate| candidate.contract_id).collect(),
        best_non_exact_contract_id: best_non_exact
            .map(|candidate| candidate.contract_id.to_string())
            .unwrap_or_default(),
        best_non_exact_unrealizable_element_count: best_non_exact
            .map(|candidate| candidate.unrealizable_tensor_element_count),
        candidate_conservation: format!(
            "{} evaluations = {} assessed ({} exact + {} non-exact) + {} not-assessed",
            candidates.len(),
            assessed.len(),
            exact.len(),
            non_exact,
            candidates.len().saturating_sub(assessed.len())
        ),
        portfolio_ledger_sha256,
        candidates,
        method: "For each eligible UINT8/INT8 NHWC [1,H,W,3] constructive input witness, enumerate all 256 source-pixel codes independently for each tensor channel under eight explicit preprocessing counterfactuals. Direct-storage rows decode bytes without a source quantizer; normalized rows use exact half-away-from-zero quantization against the artifact's binary64 scale. Inventory reachable codes, holes, collisions, and inverse pixels, then choose the minimum-absolute-error pixel for every complete witness element and hash the resulting RGB fixture.",
        proof_scope: "An exact_tensor_realization candidate proves that one complete RGB8 source raster maps byte-for-byte to the certified model-input tensor under the declared candidate formula, channel permutation, and direct-storage or quantizer equation. A non-exact row proves the minimum per-element absolute tensor-code error within that same finite 256-code source domain.",
        interpretation_boundary: "The eight rows are explicit counterfactual contracts, not observations of the production application. Exact realizability does not identify which contract, decoder, resize kernel, color management, or runtime build is deployed; non-exactness applies only to the named formula and channel order. Declared-output impact and task-level effect remain unmeasured.",
    }
}

fn eligible_image_witness(witness: &TensorAbiInputWitness) -> bool {
    witness.model_input_shape.len() == 4
        && witness.model_input_shape[0] == 1
        && witness.model_input_shape[1] > 0
        && witness.model_input_shape[2] > 0
        && witness.model_input_shape[3] == 3
        && matches!(witness.model_input_dtype.as_str(), "UINT8" | "INT8")
        && witness.model_input_element_count
            == witness
                .model_input_shape
                .iter()
                .try_fold(1usize, |product, value| {
                    usize::try_from(*value)
                        .ok()
                        .and_then(|value| product.checked_mul(value))
                })
                .unwrap_or(0)
}

fn evaluate_candidate(
    witness_index: usize,
    witness: &TensorAbiInputWitness,
    contract: ContractSpec,
) -> PreprocessingCandidate {
    match evaluate_candidate_inner(witness_index, witness, contract) {
        Ok(candidate) => candidate,
        Err(error) => PreprocessingCandidate {
            witness_index,
            source_op_index: witness.source_op_index,
            contract_id: contract.id,
            contract_label: contract.label,
            contract_family: contract.family,
            source_pixel_order: "RGB",
            tensor_channel_order: contract.tensor_order,
            pixel_to_real_formula: contract.pixel_to_real,
            real_to_tensor_formula: contract.real_to_tensor,
            rounding_mode: contract.rounding_mode,
            status: "not_assessed",
            assessment_reason: error,
            source_image_shape_hwc: Vec::new(),
            channel_maps: Vec::new(),
            witness_code_realizations: Vec::new(),
            distinct_witness_channel_code_pair_count: 0,
            exact_witness_channel_code_pair_count: 0,
            witness_tensor_element_count: witness.model_input_element_count,
            exact_tensor_element_count: 0,
            unrealizable_tensor_element_count: witness.model_input_element_count,
            exact_tensor_realization: false,
            minimum_total_absolute_tensor_code_error_decimal: "0".to_string(),
            maximum_absolute_tensor_code_error: 0,
            first_unrealizable_element: None,
            fixture_encoding: "RGB8",
            fixture_kind: "not_assessed",
            nearest_rgb_fixture_sha256: String::new(),
            exact_rgb_fixture_sha256: None,
            source_input_witness_ledger_sha256: witness.witness_ledger_sha256.clone(),
            candidate_ledger_sha256: String::new(),
        },
    }
}

fn evaluate_candidate_inner(
    witness_index: usize,
    witness: &TensorAbiInputWitness,
    contract: ContractSpec,
) -> Result<PreprocessingCandidate, String> {
    let scale = exact_positive_f64_ratio(witness.model_input_scale).ok_or_else(|| {
        "The input scale cannot be represented inside the exact i128 rational budget.".to_string()
    })?;
    let mut channel_maps = Vec::with_capacity(3);
    for tensor_channel in 0..3 {
        let mut codes = Vec::with_capacity(256);
        for pixel in 0..=255u8 {
            codes.push(map_pixel_code(
                pixel,
                tensor_channel,
                witness,
                contract,
                scale,
            )?);
        }
        channel_maps.push(summarize_map(
            tensor_channel,
            contract.tensor_to_source[tensor_channel],
            codes,
        ));
    }

    let mut tensor_codes = vec![witness.full_tensor_fill_code; witness.model_input_element_count];
    for value in &witness.sparse_overrides {
        let slot = tensor_codes
            .get_mut(value.input_linear_index)
            .ok_or_else(|| {
                "A sparse witness override is outside the complete input tensor.".to_string()
            })?;
        *slot = value.input_code;
    }
    let height = usize::try_from(witness.model_input_shape[1])
        .map_err(|_| "The witness image height is invalid.".to_string())?;
    let width = usize::try_from(witness.model_input_shape[2])
        .map_err(|_| "The witness image width is invalid.".to_string())?;
    let mut fixture = vec![0u8; height * width * 3];
    let mut pair_counts = BTreeMap::<(usize, i64), usize>::new();
    let mut exact_elements = 0usize;
    let mut total_error = 0u128;
    let mut maximum_error = 0u64;
    let mut first_unrealizable = None;
    for (linear, target) in tensor_codes.iter().copied().enumerate() {
        let tensor_channel = linear % 3;
        *pair_counts.entry((tensor_channel, target)).or_default() += 1;
        let source_channel = contract.tensor_to_source[tensor_channel];
        let (pixel, mapped, error, exact_pixels) =
            choose_pixel(&channel_maps[tensor_channel].pixel_to_tensor_codes, target);
        let pixel_base = (linear / 3) * 3;
        fixture[pixel_base + source_channel] = pixel;
        if error == 0 {
            exact_elements += 1;
        } else if first_unrealizable.is_none() {
            first_unrealizable = Some(FirstUnrealizableElement {
                tensor_linear_index: linear,
                tensor_coordinate_nhwc: vec![
                    0,
                    linear / 3 / width,
                    linear / 3 % width,
                    tensor_channel,
                ],
                tensor_channel,
                source_pixel_channel: source_channel,
                target_tensor_code: target,
                selected_source_pixel_code: pixel,
                roundtrip_tensor_code: mapped,
                absolute_tensor_code_error: error,
            });
        }
        let _ = exact_pixels;
        total_error = total_error
            .checked_add(error as u128)
            .ok_or_else(|| "The witness approximation error total overflowed u128.".to_string())?;
        maximum_error = maximum_error.max(error);
    }

    let mut realizations = Vec::with_capacity(pair_counts.len());
    for ((tensor_channel, target), count) in pair_counts {
        let source_channel = contract.tensor_to_source[tensor_channel];
        let (pixel, mapped, error, exact_pixels) =
            choose_pixel(&channel_maps[tensor_channel].pixel_to_tensor_codes, target);
        realizations.push(WitnessCodeRealization {
            tensor_channel,
            source_pixel_channel: source_channel,
            target_tensor_code: target,
            tensor_element_count: count,
            exact_source_pixel_codes: exact_pixels,
            selected_source_pixel_code: pixel,
            roundtrip_tensor_code: mapped,
            absolute_tensor_code_error: error,
        });
    }
    let exact_pairs = realizations
        .iter()
        .filter(|row| row.absolute_tensor_code_error == 0)
        .count();
    let fixture_sha = hex_digest(Sha256::digest(&fixture).as_slice());
    let exact_tensor = exact_elements == tensor_codes.len();
    let mut candidate = PreprocessingCandidate {
        witness_index,
        source_op_index: witness.source_op_index,
        contract_id: contract.id,
        contract_label: contract.label,
        contract_family: contract.family,
        source_pixel_order: "RGB",
        tensor_channel_order: contract.tensor_order,
        pixel_to_real_formula: contract.pixel_to_real,
        real_to_tensor_formula: contract.real_to_tensor,
        rounding_mode: contract.rounding_mode,
        status: "assessed",
        assessment_reason: "All 256 source-pixel codes were exhaustively mapped for every tensor channel; every witness element uses the smallest exact inverse or the minimum-error pixel with a stable lower-code tie break.".to_string(),
        source_image_shape_hwc: vec![height, width, 3],
        channel_maps,
        distinct_witness_channel_code_pair_count: realizations.len(),
        exact_witness_channel_code_pair_count: exact_pairs,
        witness_code_realizations: realizations,
        witness_tensor_element_count: tensor_codes.len(),
        exact_tensor_element_count: exact_elements,
        unrealizable_tensor_element_count: tensor_codes.len().saturating_sub(exact_elements),
        exact_tensor_realization: exact_tensor,
        minimum_total_absolute_tensor_code_error_decimal: total_error.to_string(),
        maximum_absolute_tensor_code_error: maximum_error,
        first_unrealizable_element: first_unrealizable,
        fixture_encoding: "RGB8",
        fixture_kind: if exact_tensor {
            "exact_source_raster"
        } else {
            "minimum_code_error_counterfactual"
        },
        nearest_rgb_fixture_sha256: fixture_sha.clone(),
        exact_rgb_fixture_sha256: exact_tensor.then_some(fixture_sha),
        source_input_witness_ledger_sha256: witness.witness_ledger_sha256.clone(),
        candidate_ledger_sha256: String::new(),
    };
    candidate.candidate_ledger_sha256 = candidate_ledger(&candidate);
    Ok(candidate)
}

fn summarize_map(
    tensor_channel: usize,
    source_pixel_channel: usize,
    codes: Vec<i64>,
) -> PixelCodeMap {
    let mut counts = BTreeMap::<i64, usize>::new();
    for code in &codes {
        *counts.entry(*code).or_default() += 1;
    }
    PixelCodeMap {
        tensor_channel,
        source_pixel_channel,
        source_pixel_channel_name: ["R", "G", "B"][source_pixel_channel],
        reachable_tensor_code_count: counts.len(),
        tensor_code_hole_count: 256usize.saturating_sub(counts.len()),
        collision_tensor_code_count: counts.values().filter(|count| **count > 1).count(),
        maximum_preimage_multiplicity: counts.values().copied().max().unwrap_or(0),
        reachable_tensor_code_min: counts.keys().next().copied().unwrap_or(0),
        reachable_tensor_code_max: counts.keys().next_back().copied().unwrap_or(0),
        pixel_to_tensor_codes: codes,
    }
}

fn choose_pixel(codes: &[i64], target: i64) -> (u8, i64, u64, Vec<u8>) {
    let exact = codes
        .iter()
        .enumerate()
        .filter_map(|(pixel, code)| (*code == target).then_some(pixel as u8))
        .collect::<Vec<_>>();
    if let Some(pixel) = exact.first().copied() {
        return (pixel, target, 0, exact);
    }
    let (pixel, mapped, error) = codes
        .iter()
        .copied()
        .enumerate()
        .map(|(pixel, mapped)| (pixel as u8, mapped, mapped.abs_diff(target)))
        .min_by(|left, right| left.2.cmp(&right.2).then_with(|| left.0.cmp(&right.0)))
        .unwrap_or((0, target, 0));
    (pixel, mapped, error, exact)
}

fn map_pixel_code(
    pixel: u8,
    tensor_channel: usize,
    witness: &TensorAbiInputWitness,
    contract: ContractSpec,
    scale: (i128, i128),
) -> Result<i64, String> {
    let qmin = witness.model_input_code_range[0];
    let qmax = witness.model_input_code_range[1];
    match contract.transform {
        Transform::RawStorage => Ok(match witness.model_input_dtype.as_str() {
            "UINT8" => pixel as i64,
            "INT8" => (pixel as i8) as i64,
            _ => {
                return Err("The direct storage contract requires an 8-bit input dtype.".to_string())
            }
        }),
        Transform::ArtifactAffine => Ok(qmin + pixel as i64),
        Transform::Center128 => quantize_rational(
            pixel as i128 - 128,
            128,
            scale,
            witness.model_input_zero_point,
            qmin,
            qmax,
        ),
        Transform::MinusOneToOne => quantize_rational(
            pixel as i128 * 2 - 255,
            255,
            scale,
            witness.model_input_zero_point,
            qmin,
            qmax,
        ),
        Transform::UnitInterval => quantize_rational(
            pixel as i128,
            255,
            scale,
            witness.model_input_zero_point,
            qmin,
            qmax,
        ),
        Transform::ImageNet => quantize_rational(
            1000 * pixel as i128 - 255 * contract.imagenet_mean_milli[tensor_channel],
            255 * contract.imagenet_std_milli[tensor_channel],
            scale,
            witness.model_input_zero_point,
            qmin,
            qmax,
        ),
    }
}

fn quantize_rational(
    real_num: i128,
    real_den: i128,
    scale: (i128, i128),
    zero_point: i64,
    qmin: i64,
    qmax: i64,
) -> Result<i64, String> {
    let numerator = real_num
        .checked_mul(scale.1)
        .ok_or_else(|| "The exact preprocessing numerator exceeded i128.".to_string())?;
    let denominator = real_den
        .checked_mul(scale.0)
        .ok_or_else(|| "The exact preprocessing denominator exceeded i128.".to_string())?;
    if denominator <= 0 {
        return Err("The exact preprocessing denominator is not positive.".to_string());
    }
    let rounded = round_ratio_away(numerator, denominator)?;
    let shifted = rounded.checked_add(zero_point as i128).ok_or_else(|| {
        "The exact preprocessing zero-point addition overflowed i128.".to_string()
    })?;
    Ok(shifted.clamp(qmin as i128, qmax as i128) as i64)
}

fn round_ratio_away(numerator: i128, denominator: i128) -> Result<i128, String> {
    let negative = numerator < 0;
    let magnitude = numerator.checked_abs().ok_or_else(|| {
        "The exact preprocessing numerator magnitude overflowed i128.".to_string()
    })?;
    let quotient = magnitude / denominator;
    let remainder = magnitude % denominator;
    let twice = remainder
        .checked_mul(2)
        .ok_or_else(|| "The exact preprocessing remainder overflowed i128.".to_string())?;
    let rounded = quotient + i128::from(twice >= denominator);
    Ok(if negative { -rounded } else { rounded })
}

fn exact_positive_f64_ratio(value: f64) -> Option<(i128, i128)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mut numerator, mut exponent) = if exponent_bits == 0 {
        (fraction as i128, -1074)
    } else {
        (((1u64 << 52) | fraction) as i128, exponent_bits - 1023 - 52)
    };
    if numerator == 0 {
        return None;
    }
    if exponent < 0 {
        let reducible = numerator.trailing_zeros().min((-exponent) as u32);
        numerator >>= reducible;
        exponent += reducible as i32;
    }
    if exponent >= 0 {
        let shift = u32::try_from(exponent).ok()?;
        numerator.checked_shl(shift).map(|value| (value, 1))
    } else {
        let shift = u32::try_from(-exponent).ok()?;
        1i128
            .checked_shl(shift)
            .map(|denominator| (numerator, denominator))
    }
}

fn candidate_ledger(candidate: &PreprocessingCandidate) -> String {
    let mut canonical = String::from(CANDIDATE_LEDGER_PREFIX);
    canonical.push_str(&format!(
        "witness={};source={};contract={};order={};status={};input_witness={}\n",
        candidate.witness_index,
        candidate.source_op_index,
        candidate.contract_id,
        candidate.tensor_channel_order,
        candidate.status,
        candidate.source_input_witness_ledger_sha256
    ));
    for map in &candidate.channel_maps {
        canonical.push_str(&format!(
            "map={};source_channel={};reachable={};holes={};collisions={};multiplicity={};codes={}\n",
            map.tensor_channel,
            map.source_pixel_channel,
            map.reachable_tensor_code_count,
            map.tensor_code_hole_count,
            map.collision_tensor_code_count,
            map.maximum_preimage_multiplicity,
            join_i64(&map.pixel_to_tensor_codes)
        ));
    }
    for row in &candidate.witness_code_realizations {
        canonical.push_str(&format!(
            "code={};target={};source_channel={};count={};exact_pixels={};selected={};roundtrip={};error={}\n",
            row.tensor_channel,
            row.target_tensor_code,
            row.source_pixel_channel,
            row.tensor_element_count,
            join_u8(&row.exact_source_pixel_codes),
            row.selected_source_pixel_code,
            row.roundtrip_tensor_code,
            row.absolute_tensor_code_error
        ));
    }
    canonical.push_str(&format!(
        "summary={};exact_elements={};unrealizable={};total_error={};max_error={};nearest_sha={};exact_sha={}\n",
        candidate.witness_tensor_element_count,
        candidate.exact_tensor_element_count,
        candidate.unrealizable_tensor_element_count,
        candidate.minimum_total_absolute_tensor_code_error_decimal,
        candidate.maximum_absolute_tensor_code_error,
        candidate.nearest_rgb_fixture_sha256,
        candidate.exact_rgb_fixture_sha256.as_deref().unwrap_or("none")
    ));
    hex_digest(Sha256::digest(canonical.as_bytes()).as_slice())
}

fn portfolio_ledger(source: &str, candidates: &[PreprocessingCandidate]) -> String {
    let mut canonical = String::from(PORTFOLIO_LEDGER_PREFIX);
    canonical.push_str(&format!("input_counterexample={source}\n"));
    for candidate in candidates {
        canonical.push_str(&format!(
            "candidate={};witness={};status={};ledger={}\n",
            candidate.contract_id,
            candidate.witness_index,
            candidate.status,
            candidate.candidate_ledger_sha256
        ));
    }
    hex_digest(Sha256::digest(canonical.as_bytes()).as_slice())
}

fn join_i64(values: &[i64]) -> String {
    values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_u8(values: &[u8]) -> String {
    values
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn decimal(value: &str) -> u128 {
    value.parse().unwrap_or(u128::MAX)
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary64_ratio_is_exact_for_one_over_128() {
        assert_eq!(exact_positive_f64_ratio(0.0078125), Some((1, 128)));
    }

    #[test]
    fn ties_are_rounded_away_from_zero() {
        assert_eq!(round_ratio_away(1, 2).unwrap(), 1);
        assert_eq!(round_ratio_away(-1, 2).unwrap(), -1);
        assert_eq!(round_ratio_away(3, 2).unwrap(), 2);
        assert_eq!(round_ratio_away(-3, 2).unwrap(), -2);
    }

    #[test]
    fn pixel_inverse_uses_smallest_exact_and_nearest_code() {
        let mut codes = (0..=255).map(|value| value as i64).collect::<Vec<_>>();
        codes[12] = 11;
        assert_eq!(choose_pixel(&codes, 11), (11, 11, 0, vec![11, 12]));
        assert_eq!(choose_pixel(&codes, 12), (11, 11, 1, Vec::new()));
    }
}
