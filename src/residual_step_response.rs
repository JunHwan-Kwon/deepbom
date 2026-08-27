use super::quantization_lattice::{
    ContainmentCandidateEvaluation, QuantizationLatticeAnalysis, ResidualAddLatticeRow,
};
use super::quantization_math::round_ties_away_from_zero;
use super::{OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

const SCHEMA: &str = "deepbom.residual_step_response.v1";
const METHOD_VERSION: &str = "2026-07-17.1";
const TILE_SIZE: usize = 16;

#[derive(Clone, Copy)]
struct QuantContract {
    tensor_index: usize,
    qmin: i64,
    qmax: i64,
    scale: f64,
    zero_point: i64,
}

#[derive(Clone, Copy)]
struct Projection {
    raw_code: i64,
    code: i64,
}

#[derive(Serialize)]
struct StepWitness {
    base_input_0_code: i64,
    base_input_1_code: i64,
    base_output_code: i64,
    adjacent_output_code: i64,
    output_code_delta: usize,
    clamp_associated: bool,
}

#[derive(Serialize)]
struct BranchStepResponse {
    branch_index: usize,
    input_tensor_index: usize,
    input_scale: f64,
    transition_count: usize,
    visible_transition_count: usize,
    silent_transition_count: usize,
    visible_transition_ratio: f64,
    silent_transition_ratio: f64,
    unclipped_transition_count: usize,
    unclipped_silent_transition_count: usize,
    clamp_associated_transition_count: usize,
    clamp_associated_silent_transition_count: usize,
    multi_code_jump_transition_count: usize,
    mean_output_code_delta: f64,
    maximum_output_code_delta: usize,
    mean_absolute_step_reproduction_error_output_steps: f64,
    maximum_absolute_step_reproduction_error_output_steps: f64,
    output_code_delta_histogram: Vec<usize>,
    worst_jump: Option<StepWitness>,
    first_unclipped_silent: Option<StepWitness>,
}

#[derive(Serialize)]
struct ContractStepResponse {
    design: &'static str,
    output_scale: f64,
    output_zero_point: i64,
    scale_ratio_to_current: f64,
    signed_zero_point_delta: i64,
    rounded_projection_clamp_pair_count: usize,
    rounded_projection_clamp_pair_ratio: f64,
    complete_rounded_domain_containment: bool,
    distinct_projected_output_code_count: usize,
    branch_responses: Vec<BranchStepResponse>,
    total_transition_count: usize,
    visible_transition_count: usize,
    silent_transition_count: usize,
    visible_transition_ratio: f64,
    silent_transition_ratio: f64,
    joint_interior_cell_count: usize,
    both_branches_visible_cell_count: usize,
    input_0_only_visible_cell_count: usize,
    input_1_only_visible_cell_count: usize,
    neither_branch_visible_cell_count: usize,
    both_branches_visible_ratio: f64,
    neither_branch_visible_ratio: f64,
    tile_size_codes: usize,
    tile_grid_dimension: usize,
    tile_joint_cell_counts: Vec<usize>,
    tile_both_branches_visible_counts: Vec<usize>,
    tile_input_0_only_visible_counts: Vec<usize>,
    tile_input_1_only_visible_counts: Vec<usize>,
    tile_neither_branch_visible_counts: Vec<usize>,
    removed_rounded_clamp_pairs_vs_current: i64,
    additional_silent_transitions_vs_current: i64,
    visible_transition_ratio_delta_vs_current: f64,
    transition_ledger_sha256: String,
}

#[derive(Serialize)]
struct ResidualStepResponseRow {
    op_index: usize,
    op_name: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    input_tensor_indices: Vec<i32>,
    output_tensor_index: Option<i32>,
    contracts: Vec<ContractStepResponse>,
    maximum_containment_silent_ratio_increase: Option<f64>,
    maximum_containment_additional_silent_transitions: Option<i64>,
    maximum_containment_removed_clamp_pairs: Option<i64>,
    retention_cost_rank: Option<usize>,
}

#[derive(Serialize)]
pub(super) struct ResidualStepResponseAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_add_count: usize,
    assessed_add_count: usize,
    unassessed_add_count: usize,
    contract_response_count: usize,
    total_transition_count: usize,
    total_joint_interior_cell_count: usize,
    current_silent_transition_count: usize,
    containment_silent_transition_count: usize,
    containment_additional_silent_transition_count: i64,
    current_rounded_projection_clamp_pair_count: usize,
    containment_removed_rounded_clamp_pair_count: i64,
    maximum_containment_silent_ratio_increase: Option<f64>,
    retention_cost_ranking_op_indices: Vec<usize>,
    residual_adds: Vec<ResidualStepResponseRow>,
    transition_definition: &'static str,
    joint_cell_definition: &'static str,
    transition_ledger_hash_method: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

struct BranchAccumulator {
    input: QuantContract,
    transitions: usize,
    visible: usize,
    silent: usize,
    unclipped: usize,
    unclipped_silent: usize,
    clamp_associated: usize,
    clamp_associated_silent: usize,
    multi_code: usize,
    delta_sum: usize,
    max_delta: usize,
    error_sum_steps: f64,
    error_max_steps: f64,
    histogram: [usize; 256],
    worst_jump: Option<StepWitness>,
    first_unclipped_silent: Option<StepWitness>,
}

#[derive(Clone, Copy)]
struct TransitionSample {
    branch_index: usize,
    q0: i64,
    q1: i64,
    base: Projection,
    adjacent: Projection,
}

impl BranchAccumulator {
    fn new(input: QuantContract) -> Self {
        Self {
            input,
            transitions: 0,
            visible: 0,
            silent: 0,
            unclipped: 0,
            unclipped_silent: 0,
            clamp_associated: 0,
            clamp_associated_silent: 0,
            multi_code: 0,
            delta_sum: 0,
            max_delta: 0,
            error_sum_steps: 0.0,
            error_max_steps: 0.0,
            histogram: [0; 256],
            worst_jump: None,
            first_unclipped_silent: None,
        }
    }

    fn record(
        &mut self,
        sample: TransitionSample,
        output: QuantContract,
        ledger: &mut Sha256,
    ) -> usize {
        let TransitionSample {
            branch_index,
            q0,
            q1,
            base,
            adjacent,
        } = sample;
        let delta = adjacent.code.saturating_sub(base.code) as usize;
        let unclipped = in_range(base.raw_code, output) && in_range(adjacent.raw_code, output);
        let witness = StepWitness {
            base_input_0_code: q0,
            base_input_1_code: q1,
            base_output_code: base.code,
            adjacent_output_code: adjacent.code,
            output_code_delta: delta,
            clamp_associated: !unclipped,
        };
        self.transitions += 1;
        self.delta_sum += delta;
        self.histogram[delta] += 1;
        if delta == 0 {
            self.silent += 1;
            if unclipped {
                self.unclipped_silent += 1;
                if self.first_unclipped_silent.is_none() {
                    self.first_unclipped_silent = Some(clone_witness(&witness));
                }
            } else {
                self.clamp_associated_silent += 1;
            }
        } else {
            self.visible += 1;
        }
        if delta > 1 {
            self.multi_code += 1;
        }
        if unclipped {
            self.unclipped += 1;
        } else {
            self.clamp_associated += 1;
        }
        if delta > self.max_delta {
            self.max_delta = delta;
            self.worst_jump = Some(clone_witness(&witness));
        }
        let error_steps = (delta as f64 * output.scale - self.input.scale).abs() / output.scale;
        self.error_sum_steps += error_steps;
        self.error_max_steps = self.error_max_steps.max(error_steps);
        for value in [
            branch_index as i64,
            q0,
            q1,
            base.raw_code,
            adjacent.raw_code,
            base.code,
            adjacent.code,
            delta as i64,
            i64::from(unclipped),
        ] {
            ledger.update(value.to_le_bytes());
        }
        delta
    }

    fn finish(self, branch_index: usize) -> BranchStepResponse {
        let last_nonzero = self
            .histogram
            .iter()
            .rposition(|count| *count > 0)
            .unwrap_or(0);
        BranchStepResponse {
            branch_index,
            input_tensor_index: self.input.tensor_index,
            input_scale: self.input.scale,
            transition_count: self.transitions,
            visible_transition_count: self.visible,
            silent_transition_count: self.silent,
            visible_transition_ratio: ratio(self.visible, self.transitions),
            silent_transition_ratio: ratio(self.silent, self.transitions),
            unclipped_transition_count: self.unclipped,
            unclipped_silent_transition_count: self.unclipped_silent,
            clamp_associated_transition_count: self.clamp_associated,
            clamp_associated_silent_transition_count: self.clamp_associated_silent,
            multi_code_jump_transition_count: self.multi_code,
            mean_output_code_delta: self.delta_sum as f64 / self.transitions.max(1) as f64,
            maximum_output_code_delta: self.max_delta,
            mean_absolute_step_reproduction_error_output_steps: self.error_sum_steps
                / self.transitions.max(1) as f64,
            maximum_absolute_step_reproduction_error_output_steps: self.error_max_steps,
            output_code_delta_histogram: self.histogram[..=last_nonzero].to_vec(),
            worst_jump: self.worst_jump,
            first_unclipped_silent: self.first_unclipped_silent,
        }
    }
}

pub(super) fn build_residual_step_response(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    lattice: &QuantizationLatticeAnalysis,
) -> ResidualStepResponseAnalysis {
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
                row.maximum_containment_silent_ratio_increase.unwrap_or(0.0),
                row.maximum_containment_additional_silent_transitions
                    .unwrap_or(0),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    for (rank, (op_index, _, _)) in ranked.iter().enumerate() {
        if let Some(row) = rows.iter_mut().find(|row| row.op_index == *op_index) {
            row.retention_cost_rank = Some(rank + 1);
        }
    }
    let responses = rows
        .iter()
        .flat_map(|row| row.contracts.iter())
        .collect::<Vec<_>>();
    let current = responses
        .iter()
        .filter(|response| response.design == "current_artifact_contract")
        .collect::<Vec<_>>();
    let containment = responses
        .iter()
        .filter(|response| response.design != "current_artifact_contract")
        .collect::<Vec<_>>();
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .count();
    ResidualStepResponseAnalysis {
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
        contract_response_count: responses.len(),
        total_transition_count: responses
            .iter()
            .map(|response| response.total_transition_count)
            .sum(),
        total_joint_interior_cell_count: responses
            .iter()
            .map(|response| response.joint_interior_cell_count)
            .sum(),
        current_silent_transition_count: current
            .iter()
            .map(|response| response.silent_transition_count)
            .sum(),
        containment_silent_transition_count: containment
            .iter()
            .map(|response| response.silent_transition_count)
            .sum(),
        containment_additional_silent_transition_count: containment
            .iter()
            .map(|response| response.additional_silent_transitions_vs_current)
            .sum(),
        current_rounded_projection_clamp_pair_count: current
            .iter()
            .map(|response| response.rounded_projection_clamp_pair_count)
            .sum(),
        containment_removed_rounded_clamp_pair_count: containment
            .iter()
            .map(|response| response.removed_rounded_clamp_pairs_vs_current)
            .sum(),
        maximum_containment_silent_ratio_increase: max_optional(
            containment
                .iter()
                .map(|response| -response.visible_transition_ratio_delta_vs_current),
        ),
        retention_cost_ranking_op_indices: ranked.iter().map(|row| row.0).collect(),
        residual_adds: rows,
        transition_definition: "For each branch independently, hold the other input code fixed and increment the selected legal 8-bit input code by one. Project both sums with round_ties_away_from_zero and output clamping; output-code delta zero is a silent transition.",
        joint_cell_definition: "For every common interior code pair (q0<qmax and q1<qmax), classify whether the independent +1 step on input 0 and input 1 changes the projected output code: both, input-0-only, input-1-only, or neither.",
        transition_ledger_hash_method: "SHA-256 over ordered transition rows; each row is nine signed i64 little-endian fields: branch_index, q0, q1, base_raw_code, adjacent_raw_code, base_projected_code, adjacent_projected_code, output_code_delta, unclipped_flag",
        method: "Exhaustively evaluate current, fixed-zero-point minimum-containment, and globally finest minimum-containment output contracts for every assessed 8-bit residual ADD. Preserve branch-complete transition histograms, joint influence tiles, clamp association, step-reproduction error, and a canonical transition SHA-256 ledger.",
        interpretation_boundary: "This is uniform legal-code-domain local distinguishability, not an observed activation distribution, mutual information estimate, calibration recommendation, task-accuracy result, or executed fixed-point kernel trace. A silent transition means one isolated input-code increment projects to the same ideal output code under the stated contract; it does not prove a runtime branch is inactive. Containment candidates remain counterfactual re-export contracts.",
    }
}

pub(super) fn residual_step_response_not_computed() -> ResidualStepResponseAnalysis {
    ResidualStepResponseAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status: "not_computed_internal_planning_scope",
        candidate_add_count: 0,
        assessed_add_count: 0,
        unassessed_add_count: 0,
        contract_response_count: 0,
        total_transition_count: 0,
        total_joint_interior_cell_count: 0,
        current_silent_transition_count: 0,
        containment_silent_transition_count: 0,
        containment_additional_silent_transition_count: 0,
        current_rounded_projection_clamp_pair_count: 0,
        containment_removed_rounded_clamp_pair_count: 0,
        maximum_containment_silent_ratio_increase: None,
        retention_cost_ranking_op_indices: Vec::new(),
        residual_adds: Vec::new(),
        transition_definition: "Not computed in internal target-planning scope.",
        joint_cell_definition: "Not computed in internal target-planning scope.",
        transition_ledger_hash_method: "Not computed in internal target-planning scope.",
        method: "Target-independent residual step response is computed once in the user-facing full analysis, not in internal target-planning reanalysis.",
        interpretation_boundary: "Internal placeholder; never emitted as user-facing residual step-response evidence.",
    }
}

fn build_row(
    lattice: &ResidualAddLatticeRow,
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> ResidualStepResponseRow {
    let Some(op) = ops.iter().find(|op| op.index == lattice.op_index) else {
        return not_assessed(lattice.op_index, "ADD op is unavailable.".to_string());
    };
    let failure = || ResidualStepResponseRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "not_assessed",
        not_assessed_reason:
            "A complete residual lattice and both containment contracts are required.".to_string(),
        input_tensor_indices: op.inputs.iter().take(2).copied().collect(),
        output_tensor_index: op.outputs.first().copied(),
        contracts: Vec::new(),
        maximum_containment_silent_ratio_increase: None,
        maximum_containment_additional_silent_transitions: None,
        maximum_containment_removed_clamp_pairs: None,
        retention_cost_rank: None,
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
    let mut contracts = vec![evaluate_contract(
        "current_artifact_contract",
        input0,
        input1,
        current_output,
        1.0,
        0,
    )];
    contracts.push(evaluate_candidate(fixed, input0, input1, current_output));
    contracts.push(evaluate_candidate(global, input0, input1, current_output));
    let current_clamps = contracts[0].rounded_projection_clamp_pair_count as i64;
    let current_silent = contracts[0].silent_transition_count as i64;
    let current_visible_ratio = contracts[0].visible_transition_ratio;
    for contract in &mut contracts {
        contract.removed_rounded_clamp_pairs_vs_current =
            current_clamps - contract.rounded_projection_clamp_pair_count as i64;
        contract.additional_silent_transitions_vs_current =
            contract.silent_transition_count as i64 - current_silent;
        contract.visible_transition_ratio_delta_vs_current =
            contract.visible_transition_ratio - current_visible_ratio;
    }
    let containment = &contracts[1..];
    ResidualStepResponseRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_tensor_indices: op.inputs.iter().take(2).copied().collect(),
        output_tensor_index: op.outputs.first().copied(),
        maximum_containment_silent_ratio_increase: max_optional(
            containment
                .iter()
                .map(|contract| -contract.visible_transition_ratio_delta_vs_current),
        ),
        maximum_containment_additional_silent_transitions: containment
            .iter()
            .map(|contract| contract.additional_silent_transitions_vs_current)
            .max(),
        maximum_containment_removed_clamp_pairs: containment
            .iter()
            .map(|contract| contract.removed_rounded_clamp_pairs_vs_current)
            .max(),
        contracts,
        retention_cost_rank: None,
    }
}

fn not_assessed(op_index: usize, reason: String) -> ResidualStepResponseRow {
    ResidualStepResponseRow {
        op_index,
        op_name: "ADD".to_string(),
        assessment_status: "not_assessed",
        not_assessed_reason: reason,
        input_tensor_indices: Vec::new(),
        output_tensor_index: None,
        contracts: Vec::new(),
        maximum_containment_silent_ratio_increase: None,
        maximum_containment_additional_silent_transitions: None,
        maximum_containment_removed_clamp_pairs: None,
        retention_cost_rank: None,
    }
}

fn evaluate_candidate(
    candidate: &ContainmentCandidateEvaluation,
    input0: QuantContract,
    input1: QuantContract,
    current_output: QuantContract,
) -> ContractStepResponse {
    evaluate_contract(
        candidate.design,
        input0,
        input1,
        QuantContract {
            scale: candidate.output_scale,
            zero_point: candidate.output_zero_point,
            ..current_output
        },
        candidate.output_scale / current_output.scale,
        candidate.output_zero_point - current_output.zero_point,
    )
}

fn evaluate_contract(
    design: &'static str,
    input0: QuantContract,
    input1: QuantContract,
    output: QuantContract,
    scale_ratio: f64,
    zero_point_delta: i64,
) -> ContractStepResponse {
    let width = (output.qmax - output.qmin + 1) as usize;
    let grid = width / TILE_SIZE;
    let mut projected_codes = [false; 256];
    let mut clamp_pairs = 0usize;
    let mut branch0 = BranchAccumulator::new(input0);
    let mut branch1 = BranchAccumulator::new(input1);
    let mut ledger = Sha256::new();
    let mut joint = 0usize;
    let mut both = 0usize;
    let mut input0_only = 0usize;
    let mut input1_only = 0usize;
    let mut neither = 0usize;
    let mut tile_cells = vec![0usize; grid * grid];
    let mut tile_both = vec![0usize; grid * grid];
    let mut tile_input0 = vec![0usize; grid * grid];
    let mut tile_input1 = vec![0usize; grid * grid];
    let mut tile_neither = vec![0usize; grid * grid];
    for q0 in input0.qmin..=input0.qmax {
        for q1 in input1.qmin..=input1.qmax {
            let base = project(q0, q1, input0, input1, output);
            projected_codes[(base.code - output.qmin) as usize] = true;
            if !in_range(base.raw_code, output) {
                clamp_pairs += 1;
            }
            let delta0 = (q0 < input0.qmax).then(|| {
                let adjacent = project(q0 + 1, q1, input0, input1, output);
                branch0.record(
                    TransitionSample {
                        branch_index: 0,
                        q0,
                        q1,
                        base,
                        adjacent,
                    },
                    output,
                    &mut ledger,
                )
            });
            let delta1 = (q1 < input1.qmax).then(|| {
                let adjacent = project(q0, q1 + 1, input0, input1, output);
                branch1.record(
                    TransitionSample {
                        branch_index: 1,
                        q0,
                        q1,
                        base,
                        adjacent,
                    },
                    output,
                    &mut ledger,
                )
            });
            if let (Some(delta0), Some(delta1)) = (delta0, delta1) {
                joint += 1;
                let tile = ((q0 - input0.qmin) as usize / TILE_SIZE) * grid
                    + (q1 - input1.qmin) as usize / TILE_SIZE;
                tile_cells[tile] += 1;
                match (delta0 > 0, delta1 > 0) {
                    (true, true) => {
                        both += 1;
                        tile_both[tile] += 1;
                    }
                    (true, false) => {
                        input0_only += 1;
                        tile_input0[tile] += 1;
                    }
                    (false, true) => {
                        input1_only += 1;
                        tile_input1[tile] += 1;
                    }
                    (false, false) => {
                        neither += 1;
                        tile_neither[tile] += 1;
                    }
                }
            }
        }
    }
    let branch_responses = vec![branch0.finish(0), branch1.finish(1)];
    let total_transitions = branch_responses
        .iter()
        .map(|branch| branch.transition_count)
        .sum::<usize>();
    let visible = branch_responses
        .iter()
        .map(|branch| branch.visible_transition_count)
        .sum::<usize>();
    let silent = total_transitions - visible;
    ContractStepResponse {
        design,
        output_scale: output.scale,
        output_zero_point: output.zero_point,
        scale_ratio_to_current: scale_ratio,
        signed_zero_point_delta: zero_point_delta,
        rounded_projection_clamp_pair_count: clamp_pairs,
        rounded_projection_clamp_pair_ratio: ratio(clamp_pairs, 256 * 256),
        complete_rounded_domain_containment: clamp_pairs == 0,
        distinct_projected_output_code_count: projected_codes.iter().filter(|used| **used).count(),
        branch_responses,
        total_transition_count: total_transitions,
        visible_transition_count: visible,
        silent_transition_count: silent,
        visible_transition_ratio: ratio(visible, total_transitions),
        silent_transition_ratio: ratio(silent, total_transitions),
        joint_interior_cell_count: joint,
        both_branches_visible_cell_count: both,
        input_0_only_visible_cell_count: input0_only,
        input_1_only_visible_cell_count: input1_only,
        neither_branch_visible_cell_count: neither,
        both_branches_visible_ratio: ratio(both, joint),
        neither_branch_visible_ratio: ratio(neither, joint),
        tile_size_codes: TILE_SIZE,
        tile_grid_dimension: grid,
        tile_joint_cell_counts: tile_cells,
        tile_both_branches_visible_counts: tile_both,
        tile_input_0_only_visible_counts: tile_input0,
        tile_input_1_only_visible_counts: tile_input1,
        tile_neither_branch_visible_counts: tile_neither,
        removed_rounded_clamp_pairs_vs_current: 0,
        additional_silent_transitions_vs_current: 0,
        visible_transition_ratio_delta_vs_current: 0.0,
        transition_ledger_sha256: format!("{:x}", ledger.finalize()),
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
        tensor_index: tensor.index,
        qmin,
        qmax,
        scale: tensor.scale_sample[0] as f64,
        zero_point: tensor.zero_point_sample[0],
    })
}

fn project(
    q0: i64,
    q1: i64,
    input0: QuantContract,
    input1: QuantContract,
    output: QuantContract,
) -> Projection {
    let real_sum = (q0 - input0.zero_point) as f64 * input0.scale
        + (q1 - input1.zero_point) as f64 * input1.scale;
    let raw_code = round_ties_away_from_zero(real_sum / output.scale) + output.zero_point;
    Projection {
        raw_code,
        code: raw_code.clamp(output.qmin, output.qmax),
    }
}

fn in_range(code: i64, contract: QuantContract) -> bool {
    (contract.qmin..=contract.qmax).contains(&code)
}

fn clone_witness(value: &StepWitness) -> StepWitness {
    StepWitness {
        base_input_0_code: value.base_input_0_code,
        base_input_1_code: value.base_input_1_code,
        base_output_code: value.base_output_code,
        adjacent_output_code: value.adjacent_output_code,
        output_code_delta: value.output_code_delta,
        clamp_associated: value.clamp_associated,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

fn max_optional(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.filter(|value| value.is_finite()).reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(index: usize, scale: f64) -> QuantContract {
        QuantContract {
            tensor_index: index,
            qmin: 0,
            qmax: 255,
            scale,
            zero_point: 128,
        }
    }

    #[test]
    fn coarse_output_contract_creates_exact_silent_steps() {
        let response = evaluate_contract(
            "current_artifact_contract",
            contract(0, 0.25),
            contract(1, 0.25),
            contract(2, 0.5),
            1.0,
            0,
        );
        assert_eq!(response.total_transition_count, 130_560);
        assert!(response.silent_transition_count > 0);
        assert_eq!(response.joint_interior_cell_count, 65_025);
        assert_eq!(
            response.both_branches_visible_cell_count
                + response.input_0_only_visible_cell_count
                + response.input_1_only_visible_cell_count
                + response.neither_branch_visible_cell_count,
            response.joint_interior_cell_count
        );
    }

    #[test]
    fn fine_output_contract_preserves_unclipped_unit_steps() {
        let response = evaluate_contract(
            "current_artifact_contract",
            contract(0, 0.25),
            contract(1, 0.25),
            contract(2, 0.25),
            1.0,
            0,
        );
        for branch in &response.branch_responses {
            assert_eq!(branch.unclipped_silent_transition_count, 0);
            assert_eq!(branch.maximum_output_code_delta, 1);
        }
    }
}
