use super::quantization_lattice::{
    ContainmentCandidateEvaluation, QuantizationLatticeAnalysis, ResidualAddLatticeRow,
};
use super::quantization_math::round_ties_away_from_zero;
use super::{OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

const SCHEMA: &str = "deepbom.residual_contract_distortion.v1.1";
const METHOD_VERSION: &str = "2026-07-29.1";
const TILE_SIZE: usize = 16;
const HISTOGRAM_BINS: usize = 32;

#[derive(Clone, Copy)]
struct QuantContract {
    qmin: i64,
    qmax: i64,
    scale: f64,
    zero_point: i64,
}

#[derive(Clone, Copy)]
struct Projection {
    raw_code: i64,
    code: i64,
    represented_real: f64,
    absolute_ideal_error: f64,
    clipped: bool,
}

#[derive(Serialize)]
struct DistortionWitness {
    input_0_code: i64,
    input_1_code: i64,
    ideal_real_sum: f64,
    current_raw_code: i64,
    current_projected_code: i64,
    current_represented_real: f64,
    current_absolute_ideal_error: f64,
    current_clipped: bool,
    candidate_raw_code: i64,
    candidate_projected_code: i64,
    candidate_represented_real: f64,
    candidate_absolute_ideal_error: f64,
    candidate_clipped: bool,
    signed_contract_delta_real: f64,
    signed_contract_delta_current_steps: f64,
    absolute_contract_delta_current_steps: f64,
}

#[derive(Serialize)]
struct ContractDistortionScenario {
    design: &'static str,
    candidate_output_scale: f64,
    candidate_output_zero_point: i64,
    candidate_scale_ratio_to_current: f64,
    candidate_signed_zero_point_delta: i64,
    enumerated_pair_count: usize,
    current_clamped_pair_count: usize,
    candidate_clamped_pair_count: usize,
    rescued_current_clamp_pair_count: usize,
    persistent_clamp_pair_count: usize,
    introduced_clamp_pair_count: usize,
    same_represented_value_pair_count: usize,
    changed_represented_value_pair_count: usize,
    sign_class_changed_pair_count: usize,
    ideal_error_improved_pair_count: usize,
    ideal_error_worsened_pair_count: usize,
    ideal_error_equal_within_tolerance_pair_count: usize,
    error_comparison_tolerance_real: f64,
    mean_signed_contract_delta_real: f64,
    mean_absolute_contract_delta_real: f64,
    root_mean_square_contract_delta_real: f64,
    mean_signed_contract_delta_current_steps: f64,
    mean_absolute_contract_delta_current_steps: f64,
    root_mean_square_contract_delta_current_steps: f64,
    maximum_absolute_contract_delta_current_steps: f64,
    p50_absolute_contract_delta_current_steps: f64,
    p90_absolute_contract_delta_current_steps: f64,
    p99_absolute_contract_delta_current_steps: f64,
    within_half_current_step_pair_count: usize,
    within_one_current_step_pair_count: usize,
    within_two_current_steps_pair_count: usize,
    mean_absolute_ideal_error_current: f64,
    mean_absolute_ideal_error_candidate: f64,
    signed_mean_absolute_ideal_error_delta: f64,
    absolute_delta_histogram_bin_width_current_steps: f64,
    absolute_delta_histogram_counts: Vec<usize>,
    tile_size_codes: usize,
    tile_grid_dimension: usize,
    tile_pair_counts: Vec<usize>,
    tile_mean_signed_delta_current_steps: Vec<f64>,
    tile_mean_absolute_delta_current_steps: Vec<f64>,
    tile_maximum_absolute_delta_current_steps: Vec<f64>,
    tile_rescued_current_clamp_pair_counts: Vec<usize>,
    tile_ideal_error_improved_pair_counts: Vec<usize>,
    tile_ideal_error_worsened_pair_counts: Vec<usize>,
    tile_ideal_error_equal_pair_counts: Vec<usize>,
    tile_sign_class_changed_pair_counts: Vec<usize>,
    worst_absolute_contract_delta_pair: DistortionWitness,
    pair_ledger_sha256: String,
}

#[derive(Serialize)]
struct ResidualContractDistortionRow {
    op_index: usize,
    op_name: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    input_tensor_indices: Vec<i32>,
    output_tensor_index: Option<i32>,
    current_output_scale: Option<f64>,
    current_output_zero_point: Option<i64>,
    scenarios: Vec<ContractDistortionScenario>,
    maximum_rms_contract_delta_current_steps: Option<f64>,
    maximum_p99_contract_delta_current_steps: Option<f64>,
    maximum_rescued_current_clamp_pair_count: Option<usize>,
    distortion_rank: Option<usize>,
}

#[derive(Serialize)]
pub(super) struct ResidualContractDistortionAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_add_count: usize,
    assessed_add_count: usize,
    unassessed_add_count: usize,
    scenario_count: usize,
    total_enumerated_pair_count: usize,
    current_clamped_pair_instance_count: usize,
    scenario_current_clamped_pair_instance_count: usize,
    candidate_clamped_pair_count: usize,
    rescued_current_clamp_pair_instance_count: usize,
    changed_represented_value_pair_count: usize,
    ideal_error_improved_pair_count: usize,
    ideal_error_worsened_pair_count: usize,
    ideal_error_equal_within_tolerance_pair_count: usize,
    sign_class_changed_pair_count: usize,
    maximum_rms_contract_delta_current_steps: Option<f64>,
    maximum_p99_contract_delta_current_steps: Option<f64>,
    distortion_ranking_op_indices: Vec<usize>,
    residual_adds: Vec<ResidualContractDistortionRow>,
    projection_definition: &'static str,
    error_comparison_definition: &'static str,
    pair_ledger_hash_method: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

struct TileAccumulator {
    pair_counts: Vec<usize>,
    signed_delta_sum: Vec<f64>,
    absolute_delta_sum: Vec<f64>,
    maximum_absolute_delta: Vec<f64>,
    rescued_clamps: Vec<usize>,
    improved: Vec<usize>,
    worsened: Vec<usize>,
    equal: Vec<usize>,
    sign_changed: Vec<usize>,
}

impl TileAccumulator {
    fn new(tile_count: usize) -> Self {
        Self {
            pair_counts: vec![0; tile_count],
            signed_delta_sum: vec![0.0; tile_count],
            absolute_delta_sum: vec![0.0; tile_count],
            maximum_absolute_delta: vec![0.0; tile_count],
            rescued_clamps: vec![0; tile_count],
            improved: vec![0; tile_count],
            worsened: vec![0; tile_count],
            equal: vec![0; tile_count],
            sign_changed: vec![0; tile_count],
        }
    }
}

pub(super) fn build_residual_contract_distortion(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    lattice: &QuantizationLatticeAnalysis,
) -> ResidualContractDistortionAnalysis {
    let mut rows = lattice
        .residual_adds
        .iter()
        .map(|row| build_row(row, ops, tensors))
        .collect::<Vec<_>>();
    let mut ranked = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .map(|row| {
            (
                row.op_index,
                row.maximum_rms_contract_delta_current_steps.unwrap_or(0.0),
                row.maximum_p99_contract_delta_current_steps.unwrap_or(0.0),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        cmp_f64_desc(left.1, right.1)
            .then_with(|| cmp_f64_desc(left.2, right.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    for (rank, (op_index, _, _)) in ranked.iter().enumerate() {
        if let Some(row) = rows.iter_mut().find(|row| row.op_index == *op_index) {
            row.distortion_rank = Some(rank + 1);
        }
    }
    let scenarios = rows
        .iter()
        .flat_map(|row| row.scenarios.iter())
        .collect::<Vec<_>>();
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .count();
    ResidualContractDistortionAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status: if rows.is_empty() {
            "not_applicable"
        } else if assessed == rows.len() {
            "assessed"
        } else if assessed == 0 {
            "not_assessed"
        } else {
            "partial"
        },
        candidate_add_count: rows.len(),
        assessed_add_count: assessed,
        unassessed_add_count: rows.len() - assessed,
        scenario_count: scenarios.len(),
        total_enumerated_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.enumerated_pair_count)
            .sum(),
        current_clamped_pair_instance_count: rows
            .iter()
            .filter_map(|row| row.scenarios.first())
            .map(|scenario| scenario.current_clamped_pair_count)
            .sum(),
        scenario_current_clamped_pair_instance_count: scenarios
            .iter()
            .map(|scenario| scenario.current_clamped_pair_count)
            .sum(),
        candidate_clamped_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.candidate_clamped_pair_count)
            .sum(),
        rescued_current_clamp_pair_instance_count: scenarios
            .iter()
            .map(|scenario| scenario.rescued_current_clamp_pair_count)
            .sum(),
        changed_represented_value_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.changed_represented_value_pair_count)
            .sum(),
        ideal_error_improved_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.ideal_error_improved_pair_count)
            .sum(),
        ideal_error_worsened_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.ideal_error_worsened_pair_count)
            .sum(),
        ideal_error_equal_within_tolerance_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.ideal_error_equal_within_tolerance_pair_count)
            .sum(),
        sign_class_changed_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.sign_class_changed_pair_count)
            .sum(),
        maximum_rms_contract_delta_current_steps: max_optional(
            scenarios
                .iter()
                .map(|scenario| scenario.root_mean_square_contract_delta_current_steps),
        ),
        maximum_p99_contract_delta_current_steps: max_optional(
            scenarios
                .iter()
                .map(|scenario| scenario.p99_absolute_contract_delta_current_steps),
        ),
        distortion_ranking_op_indices: ranked.iter().map(|row| row.0).collect(),
        residual_adds: rows,
        projection_definition: "For each legal 8-bit input-code pair, compute the ideal real sum and independently project it through the current and candidate output contracts with round_ties_away_from_zero followed by legal output-code clamping. Dequantize both projected codes before comparing represented values.",
        error_comparison_definition: "Candidate ideal-projection error is improved or worsened only when the absolute-error difference exceeds max(current_output_scale, candidate_output_scale) * 2^-40; all remaining pairs are equal within the declared binary64 tolerance.",
        pair_ledger_hash_method: "SHA-256 over q0-major/q1-minor ordered pair rows. Each row encodes nine signed i64 little-endian fields (q0, q1, current_raw_code, current_projected_code, candidate_raw_code, candidate_projected_code, current_clipped_flag, candidate_clipped_flag, ideal_error_relation) followed by six IEEE-754 binary64 little-endian bit fields (ideal_real_sum, current_represented_real, candidate_represented_real, signed_contract_delta_current_steps, current_absolute_ideal_error, candidate_absolute_ideal_error).",
        method: "Exhaustively compare both minimum-containment residual output contracts with the artifact contract over every legal input-code pair. Preserve clamp-state transitions, represented-value displacement, ideal-projection error direction, sign-class changes, exact quantiles, fixed spatial tiles, worst witnesses, and canonical pair ledgers.",
        interpretation_boundary: "This is a uniform legal-code-domain counterfactual, not an observed activation distribution, probability-weighted error, calibration recommendation, task-accuracy result, runtime mismatch count, or executed fixed-point kernel trace. Lower uniform-domain error does not establish a better deployed model; representative and adversarial task outputs must be measured after a complete re-export.",
    }
}

pub(super) fn residual_contract_distortion_not_computed() -> ResidualContractDistortionAnalysis {
    ResidualContractDistortionAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status: "not_computed_internal_planning_scope",
        candidate_add_count: 0,
        assessed_add_count: 0,
        unassessed_add_count: 0,
        scenario_count: 0,
        total_enumerated_pair_count: 0,
        current_clamped_pair_instance_count: 0,
        scenario_current_clamped_pair_instance_count: 0,
        candidate_clamped_pair_count: 0,
        rescued_current_clamp_pair_instance_count: 0,
        changed_represented_value_pair_count: 0,
        ideal_error_improved_pair_count: 0,
        ideal_error_worsened_pair_count: 0,
        ideal_error_equal_within_tolerance_pair_count: 0,
        sign_class_changed_pair_count: 0,
        maximum_rms_contract_delta_current_steps: None,
        maximum_p99_contract_delta_current_steps: None,
        distortion_ranking_op_indices: Vec::new(),
        residual_adds: Vec::new(),
        projection_definition: "Not computed in internal target-planning scope.",
        error_comparison_definition: "Not computed in internal target-planning scope.",
        pair_ledger_hash_method: "Not computed in internal target-planning scope.",
        method: "Target-independent residual contract distortion is computed once in the user-facing full analysis, not in internal target-planning reanalysis.",
        interpretation_boundary: "Internal placeholder; never emitted as user-facing residual contract-distortion evidence.",
    }
}

fn build_row(
    lattice: &ResidualAddLatticeRow,
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> ResidualContractDistortionRow {
    let Some(op) = ops.iter().find(|op| op.index == lattice.op_index) else {
        return not_assessed(lattice.op_index, "ADD operator is unavailable.".to_string());
    };
    let failure = || ResidualContractDistortionRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "not_assessed",
        not_assessed_reason:
            "A complete residual lattice and both containment contracts are required.".to_string(),
        input_tensor_indices: op.inputs.iter().take(2).copied().collect(),
        output_tensor_index: op.outputs.first().copied(),
        current_output_scale: lattice.output_scale,
        current_output_zero_point: lattice.output_zero_point,
        scenarios: Vec::new(),
        maximum_rms_contract_delta_current_steps: None,
        maximum_p99_contract_delta_current_steps: None,
        maximum_rescued_current_clamp_pair_count: None,
        distortion_rank: None,
    };
    if op.inputs.len() < 2 || op.outputs.is_empty() {
        return failure();
    }
    let Some(input0) = tensor_contract(tensors, op.inputs[0]) else {
        return failure();
    };
    let Some(input1) = tensor_contract(tensors, op.inputs[1]) else {
        return failure();
    };
    let Some(current_output) = tensor_contract(tensors, op.outputs[0]) else {
        return failure();
    };
    let (Some(fixed), Some(global)) = (
        lattice.fixed_zero_point_containment.as_ref(),
        lattice.globally_finest_containment.as_ref(),
    ) else {
        return failure();
    };
    let scenarios = vec![
        evaluate_candidate(fixed, input0, input1, current_output),
        evaluate_candidate(global, input0, input1, current_output),
    ];
    ResidualContractDistortionRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_tensor_indices: op.inputs.iter().take(2).copied().collect(),
        output_tensor_index: op.outputs.first().copied(),
        current_output_scale: Some(current_output.scale),
        current_output_zero_point: Some(current_output.zero_point),
        maximum_rms_contract_delta_current_steps: max_optional(
            scenarios
                .iter()
                .map(|scenario| scenario.root_mean_square_contract_delta_current_steps),
        ),
        maximum_p99_contract_delta_current_steps: max_optional(
            scenarios
                .iter()
                .map(|scenario| scenario.p99_absolute_contract_delta_current_steps),
        ),
        maximum_rescued_current_clamp_pair_count: scenarios
            .iter()
            .map(|scenario| scenario.rescued_current_clamp_pair_count)
            .max(),
        scenarios,
        distortion_rank: None,
    }
}

fn not_assessed(op_index: usize, reason: String) -> ResidualContractDistortionRow {
    ResidualContractDistortionRow {
        op_index,
        op_name: "ADD".to_string(),
        assessment_status: "not_assessed",
        not_assessed_reason: reason,
        input_tensor_indices: Vec::new(),
        output_tensor_index: None,
        current_output_scale: None,
        current_output_zero_point: None,
        scenarios: Vec::new(),
        maximum_rms_contract_delta_current_steps: None,
        maximum_p99_contract_delta_current_steps: None,
        maximum_rescued_current_clamp_pair_count: None,
        distortion_rank: None,
    }
}

fn evaluate_candidate(
    candidate: &ContainmentCandidateEvaluation,
    input0: QuantContract,
    input1: QuantContract,
    current_output: QuantContract,
) -> ContractDistortionScenario {
    let candidate_output = QuantContract {
        scale: candidate.output_scale,
        zero_point: candidate.output_zero_point,
        ..current_output
    };
    evaluate_scenario(
        candidate.design,
        input0,
        input1,
        current_output,
        candidate_output,
    )
}

fn evaluate_scenario(
    design: &'static str,
    input0: QuantContract,
    input1: QuantContract,
    current_output: QuantContract,
    candidate_output: QuantContract,
) -> ContractDistortionScenario {
    let pair_count = ((input0.qmax - input0.qmin + 1) * (input1.qmax - input1.qmin + 1)) as usize;
    let grid = (input0.qmax - input0.qmin + 1) as usize / TILE_SIZE;
    let mut tiles = TileAccumulator::new(grid * grid);
    let tolerance = current_output.scale.max(candidate_output.scale) * 2f64.powi(-40);
    let mut current_clamped = 0usize;
    let mut candidate_clamped = 0usize;
    let mut rescued = 0usize;
    let mut persistent = 0usize;
    let mut introduced = 0usize;
    let mut same_represented = 0usize;
    let mut sign_changed = 0usize;
    let mut improved = 0usize;
    let mut worsened = 0usize;
    let mut equal = 0usize;
    let mut signed_delta_sum = 0.0;
    let mut absolute_delta_sum = 0.0;
    let mut squared_delta_sum = 0.0;
    let mut current_error_sum = 0.0;
    let mut candidate_error_sum = 0.0;
    let mut within_half = 0usize;
    let mut within_one = 0usize;
    let mut within_two = 0usize;
    let mut absolute_deltas = Vec::with_capacity(pair_count);
    let mut ledger = Sha256::new();
    let mut worst: Option<DistortionWitness> = None;
    for q0 in input0.qmin..=input0.qmax {
        for q1 in input1.qmin..=input1.qmax {
            let ideal_sum = (q0 - input0.zero_point) as f64 * input0.scale
                + (q1 - input1.zero_point) as f64 * input1.scale;
            let current = project(ideal_sum, current_output);
            let candidate = project(ideal_sum, candidate_output);
            let signed_delta_real = candidate.represented_real - current.represented_real;
            let signed_delta_steps = signed_delta_real / current_output.scale;
            let absolute_delta_steps = signed_delta_steps.abs();
            let relation = error_relation(
                current.absolute_ideal_error,
                candidate.absolute_ideal_error,
                tolerance,
            );
            current_clamped += usize::from(current.clipped);
            candidate_clamped += usize::from(candidate.clipped);
            rescued += usize::from(current.clipped && !candidate.clipped);
            persistent += usize::from(current.clipped && candidate.clipped);
            introduced += usize::from(!current.clipped && candidate.clipped);
            same_represented += usize::from(current.represented_real == candidate.represented_real);
            sign_changed += usize::from(
                sign_class(current.represented_real) != sign_class(candidate.represented_real),
            );
            match relation {
                -1 => improved += 1,
                1 => worsened += 1,
                _ => equal += 1,
            }
            signed_delta_sum += signed_delta_real;
            absolute_delta_sum += signed_delta_real.abs();
            squared_delta_sum += signed_delta_real * signed_delta_real;
            current_error_sum += current.absolute_ideal_error;
            candidate_error_sum += candidate.absolute_ideal_error;
            within_half += usize::from(absolute_delta_steps <= 0.5);
            within_one += usize::from(absolute_delta_steps <= 1.0);
            within_two += usize::from(absolute_delta_steps <= 2.0);
            absolute_deltas.push(absolute_delta_steps);
            let tile = ((q0 - input0.qmin) as usize / TILE_SIZE) * grid
                + (q1 - input1.qmin) as usize / TILE_SIZE;
            tiles.pair_counts[tile] += 1;
            tiles.signed_delta_sum[tile] += signed_delta_steps;
            tiles.absolute_delta_sum[tile] += absolute_delta_steps;
            tiles.maximum_absolute_delta[tile] =
                tiles.maximum_absolute_delta[tile].max(absolute_delta_steps);
            tiles.rescued_clamps[tile] += usize::from(current.clipped && !candidate.clipped);
            match relation {
                -1 => tiles.improved[tile] += 1,
                1 => tiles.worsened[tile] += 1,
                _ => tiles.equal[tile] += 1,
            }
            tiles.sign_changed[tile] += usize::from(
                sign_class(current.represented_real) != sign_class(candidate.represented_real),
            );
            let witness = DistortionWitness {
                input_0_code: q0,
                input_1_code: q1,
                ideal_real_sum: ideal_sum,
                current_raw_code: current.raw_code,
                current_projected_code: current.code,
                current_represented_real: current.represented_real,
                current_absolute_ideal_error: current.absolute_ideal_error,
                current_clipped: current.clipped,
                candidate_raw_code: candidate.raw_code,
                candidate_projected_code: candidate.code,
                candidate_represented_real: candidate.represented_real,
                candidate_absolute_ideal_error: candidate.absolute_ideal_error,
                candidate_clipped: candidate.clipped,
                signed_contract_delta_real: signed_delta_real,
                signed_contract_delta_current_steps: signed_delta_steps,
                absolute_contract_delta_current_steps: absolute_delta_steps,
            };
            if worst
                .as_ref()
                .map(|row| absolute_delta_steps > row.absolute_contract_delta_current_steps)
                .unwrap_or(true)
            {
                worst = Some(witness);
            }
            for value in [
                q0,
                q1,
                current.raw_code,
                current.code,
                candidate.raw_code,
                candidate.code,
                i64::from(current.clipped),
                i64::from(candidate.clipped),
                relation,
            ] {
                ledger.update(value.to_le_bytes());
            }
            for value in [
                ideal_sum,
                current.represented_real,
                candidate.represented_real,
                signed_delta_steps,
                current.absolute_ideal_error,
                candidate.absolute_ideal_error,
            ] {
                ledger.update(value.to_bits().to_le_bytes());
            }
        }
    }
    absolute_deltas.sort_by(|left, right| left.total_cmp(right));
    let maximum = absolute_deltas.last().copied().unwrap_or(0.0);
    let histogram_width = if maximum > 0.0 {
        maximum / HISTOGRAM_BINS as f64
    } else {
        1.0
    };
    let mut histogram = vec![0usize; HISTOGRAM_BINS];
    for value in &absolute_deltas {
        let bin = ((*value / histogram_width).floor() as usize).min(HISTOGRAM_BINS - 1);
        histogram[bin] += 1;
    }
    let tile_means = |sums: &[f64]| {
        sums.iter()
            .zip(tiles.pair_counts.iter())
            .map(|(sum, count)| sum / (*count).max(1) as f64)
            .collect::<Vec<_>>()
    };
    let denominator = pair_count.max(1) as f64;
    let mean_current_error = current_error_sum / denominator;
    let mean_candidate_error = candidate_error_sum / denominator;
    let tile_mean_signed_delta = tile_means(&tiles.signed_delta_sum);
    let tile_mean_absolute_delta = tile_means(&tiles.absolute_delta_sum);
    ContractDistortionScenario {
        design,
        candidate_output_scale: candidate_output.scale,
        candidate_output_zero_point: candidate_output.zero_point,
        candidate_scale_ratio_to_current: candidate_output.scale / current_output.scale,
        candidate_signed_zero_point_delta: candidate_output.zero_point - current_output.zero_point,
        enumerated_pair_count: pair_count,
        current_clamped_pair_count: current_clamped,
        candidate_clamped_pair_count: candidate_clamped,
        rescued_current_clamp_pair_count: rescued,
        persistent_clamp_pair_count: persistent,
        introduced_clamp_pair_count: introduced,
        same_represented_value_pair_count: same_represented,
        changed_represented_value_pair_count: pair_count - same_represented,
        sign_class_changed_pair_count: sign_changed,
        ideal_error_improved_pair_count: improved,
        ideal_error_worsened_pair_count: worsened,
        ideal_error_equal_within_tolerance_pair_count: equal,
        error_comparison_tolerance_real: tolerance,
        mean_signed_contract_delta_real: signed_delta_sum / denominator,
        mean_absolute_contract_delta_real: absolute_delta_sum / denominator,
        root_mean_square_contract_delta_real: (squared_delta_sum / denominator).sqrt(),
        mean_signed_contract_delta_current_steps: signed_delta_sum
            / denominator
            / current_output.scale,
        mean_absolute_contract_delta_current_steps: absolute_delta_sum
            / denominator
            / current_output.scale,
        root_mean_square_contract_delta_current_steps: (squared_delta_sum / denominator).sqrt()
            / current_output.scale,
        maximum_absolute_contract_delta_current_steps: maximum,
        p50_absolute_contract_delta_current_steps: quantile(&absolute_deltas, 0.50),
        p90_absolute_contract_delta_current_steps: quantile(&absolute_deltas, 0.90),
        p99_absolute_contract_delta_current_steps: quantile(&absolute_deltas, 0.99),
        within_half_current_step_pair_count: within_half,
        within_one_current_step_pair_count: within_one,
        within_two_current_steps_pair_count: within_two,
        mean_absolute_ideal_error_current: mean_current_error,
        mean_absolute_ideal_error_candidate: mean_candidate_error,
        signed_mean_absolute_ideal_error_delta: mean_candidate_error - mean_current_error,
        absolute_delta_histogram_bin_width_current_steps: histogram_width,
        absolute_delta_histogram_counts: histogram,
        tile_size_codes: TILE_SIZE,
        tile_grid_dimension: grid,
        tile_pair_counts: tiles.pair_counts,
        tile_mean_signed_delta_current_steps: tile_mean_signed_delta,
        tile_mean_absolute_delta_current_steps: tile_mean_absolute_delta,
        tile_maximum_absolute_delta_current_steps: tiles.maximum_absolute_delta,
        tile_rescued_current_clamp_pair_counts: tiles.rescued_clamps,
        tile_ideal_error_improved_pair_counts: tiles.improved,
        tile_ideal_error_worsened_pair_counts: tiles.worsened,
        tile_ideal_error_equal_pair_counts: tiles.equal,
        tile_sign_class_changed_pair_counts: tiles.sign_changed,
        worst_absolute_contract_delta_pair: worst.expect("8-bit domain is non-empty"),
        pair_ledger_sha256: format!("{:x}", ledger.finalize()),
    }
}

fn tensor_contract(tensors: &[TensorInfo], tensor_index: i32) -> Option<QuantContract> {
    let tensor = tensors
        .iter()
        .find(|tensor| tensor.index == tensor_index as usize)?;
    let (qmin, qmax) = match tensor.dtype.as_str() {
        "INT8" => (-128, 127),
        "UINT8" => (0, 255),
        _ => return None,
    };
    if tensor.scale_sample.len() != 1
        || tensor.zero_point_sample.len() != 1
        || !tensor.scale_sample[0].is_finite()
        || tensor.scale_sample[0] <= 0.0
        || !(qmin..=qmax).contains(&tensor.zero_point_sample[0])
    {
        return None;
    }
    Some(QuantContract {
        qmin,
        qmax,
        scale: tensor.scale_sample[0] as f64,
        zero_point: tensor.zero_point_sample[0],
    })
}

fn project(ideal_sum: f64, output: QuantContract) -> Projection {
    let raw_code = round_ties_away_from_zero(ideal_sum / output.scale) + output.zero_point;
    let code = raw_code.clamp(output.qmin, output.qmax);
    let represented_real = (code - output.zero_point) as f64 * output.scale;
    Projection {
        raw_code,
        code,
        represented_real,
        absolute_ideal_error: (represented_real - ideal_sum).abs(),
        clipped: raw_code < output.qmin || raw_code > output.qmax,
    }
}

fn error_relation(current: f64, candidate: f64, tolerance: f64) -> i64 {
    if candidate + tolerance < current {
        -1
    } else if current + tolerance < candidate {
        1
    } else {
        0
    }
}

fn sign_class(value: f64) -> i8 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (probability * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn max_optional(values: impl Iterator<Item = f64>) -> Option<f64> {
    values
        .filter(|value| value.is_finite())
        .max_by(f64::total_cmp)
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(scale: f64) -> QuantContract {
        QuantContract {
            qmin: 0,
            qmax: 255,
            scale,
            zero_point: 0,
        }
    }

    #[test]
    fn identical_contract_has_zero_counterfactual_distortion() {
        let scenario = evaluate_scenario(
            "same",
            contract(1.0),
            contract(1.0),
            contract(2.0),
            contract(2.0),
        );
        assert_eq!(scenario.enumerated_pair_count, 65_536);
        assert_eq!(scenario.same_represented_value_pair_count, 65_536);
        assert_eq!(scenario.changed_represented_value_pair_count, 0);
        assert_eq!(scenario.ideal_error_improved_pair_count, 0);
        assert_eq!(scenario.ideal_error_worsened_pair_count, 0);
        assert_eq!(
            scenario.ideal_error_equal_within_tolerance_pair_count,
            65_536
        );
        assert_eq!(scenario.root_mean_square_contract_delta_current_steps, 0.0);
    }

    #[test]
    fn containment_contract_rescues_every_current_upper_clamp() {
        let scenario = evaluate_scenario(
            "containment",
            contract(1.0),
            contract(1.0),
            contract(1.0),
            contract(2.0),
        );
        assert_eq!(scenario.current_clamped_pair_count, 32_640);
        assert_eq!(scenario.candidate_clamped_pair_count, 0);
        assert_eq!(scenario.rescued_current_clamp_pair_count, 32_640);
        assert_eq!(scenario.persistent_clamp_pair_count, 0);
        assert_eq!(scenario.introduced_clamp_pair_count, 0);
        assert_eq!(scenario.tile_pair_counts.iter().sum::<usize>(), 65_536);
    }
}
