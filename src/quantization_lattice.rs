use super::quantization_math::round_ties_away_from_zero;
use super::{OpInfo, TensorInfo};
use serde::Serialize;
use std::cmp::Ordering;

const LATTICE_SCHEMA: &str = "deepbom.quantization_lattice.v1.4";
const LATTICE_METHOD_VERSION: &str = "2026-07-17.3";
const TILE_SIZE: usize = 16;

/// Binary elementwise operators whose legal input-code domain is small enough to
/// enumerate exhaustively (two 8-bit inputs is 2^16 pairs). Accumulating
/// operators such as CONV_2D are out of reach of enumeration by construction and
/// are covered by the accumulator labs instead.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BinaryLatticeOp {
    Add,
    Sub,
    Mul,
    Maximum,
    Minimum,
}

impl BinaryLatticeOp {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "ADD" => Some(Self::Add),
            "SUB" => Some(Self::Sub),
            "MUL" => Some(Self::Mul),
            "MAXIMUM" => Some(Self::Maximum),
            "MINIMUM" => Some(Self::Minimum),
            _ => None,
        }
    }

    fn combine(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Add => left + right,
            Self::Sub => left - right,
            Self::Mul => left * right,
            Self::Maximum => left.max(right),
            Self::Minimum => left.min(right),
        }
    }

    /// Exact real-valued range the operator can produce from the two legal input
    /// ranges. The containment design is already generic over this range, so the
    /// range derivation is the only place the operator's algebra enters.
    fn legal_range(self, left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
        match self {
            Self::Add => [left[0] + right[0], left[1] + right[1]],
            Self::Sub => [left[0] - right[1], left[1] - right[0]],
            Self::Mul => {
                // A product's extremes sit at the corners of the input rectangle;
                // sign combinations mean neither is simply low*low or high*high.
                let corners = [
                    left[0] * right[0],
                    left[0] * right[1],
                    left[1] * right[0],
                    left[1] * right[1],
                ];
                let mut low = corners[0];
                let mut high = corners[0];
                for value in corners {
                    low = low.min(value);
                    high = high.max(value);
                }
                [low, high]
            }
            Self::Maximum => [left[0].max(right[0]), left[1].max(right[1])],
            Self::Minimum => [left[0].min(right[0]), left[1].min(right[1])],
        }
    }
}

/// Distance, in output code steps, from the unclamped ideal output code to the
/// nearer end of the representable range:
///
///   r = u / sout + zout          (ideal, before rounding and clamping)
///   m = min(r - qmin, qmax - r)
///
/// `m < 0` is exactly the escape condition, so one axis carries both "does this
/// fit" and "how much room is left". The axis is in output code steps, which
/// makes it comparable across operators that have different output scales.
const MARGIN_BIN_EDGES: [f64; 15] = [
    -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
];
const MARGIN_BIN_COUNT: usize = MARGIN_BIN_EDGES.len() + 1;
/// Unit-step resolution used internally so percentiles are exact within the
/// window; only the coarse bins above are published.
const MARGIN_FINE_HALF_WIDTH: i64 = 320;

fn margin_bin_index(margin: f64) -> usize {
    let mut index = 0;
    while index < MARGIN_BIN_EDGES.len() && margin >= MARGIN_BIN_EDGES[index] {
        index += 1;
    }
    index
}

#[derive(Clone, Serialize)]
pub(super) struct MarginProfile {
    /// Lower edge of every published bin; the first bin is unbounded below and
    /// the last unbounded above.
    bin_edges_output_code_steps: Vec<f64>,
    bin_pair_counts: Vec<usize>,
    escape_pair_count: usize,
    boundary_pressure_1_step_pair_count: usize,
    boundary_pressure_2_step_pair_count: usize,
    minimum_margin_output_code_steps: f64,
    percentile_1_margin_output_code_steps: f64,
    percentile_5_margin_output_code_steps: f64,
    median_margin_output_code_steps: f64,
}

/// Accumulates the margin distribution while the pair enumeration is already
/// running, so the profile costs no extra pass over the code domain.
struct MarginAccumulator {
    fine: Vec<usize>,
    below: usize,
    above: usize,
    total: usize,
    minimum: f64,
    escape: usize,
    within_1: usize,
    within_2: usize,
}

impl MarginAccumulator {
    fn new() -> Self {
        Self {
            fine: vec![0; (MARGIN_FINE_HALF_WIDTH * 2 + 1) as usize],
            below: 0,
            above: 0,
            total: 0,
            minimum: f64::INFINITY,
            escape: 0,
            within_1: 0,
            within_2: 0,
        }
    }

    fn observe(&mut self, margin: f64) {
        self.total += 1;
        if margin < self.minimum {
            self.minimum = margin;
        }
        if margin < 0.0 {
            self.escape += 1;
        }
        if margin < 1.0 {
            self.within_1 += 1;
        }
        if margin < 2.0 {
            self.within_2 += 1;
        }
        let step = margin.floor();
        if step < -(MARGIN_FINE_HALF_WIDTH as f64) {
            self.below += 1;
        } else if step > MARGIN_FINE_HALF_WIDTH as f64 {
            self.above += 1;
        } else {
            let index = (step as i64 + MARGIN_FINE_HALF_WIDTH) as usize;
            self.fine[index] += 1;
        }
    }

    fn percentile(&self, fraction: f64) -> f64 {
        if self.total == 0 {
            return f64::NAN;
        }
        let target = (fraction * self.total as f64).ceil().max(1.0) as usize;
        let mut seen = self.below;
        if seen >= target {
            return -(MARGIN_FINE_HALF_WIDTH as f64) - 1.0;
        }
        for (index, count) in self.fine.iter().enumerate() {
            seen += count;
            if seen >= target {
                return index as f64 - MARGIN_FINE_HALF_WIDTH as f64;
            }
        }
        MARGIN_FINE_HALF_WIDTH as f64 + 1.0
    }

    fn finish(&self) -> MarginProfile {
        let mut bins = vec![0usize; MARGIN_BIN_COUNT];
        bins[0] += self.below;
        bins[MARGIN_BIN_COUNT - 1] += self.above;
        for (index, count) in self.fine.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let step = index as f64 - MARGIN_FINE_HALF_WIDTH as f64;
            bins[margin_bin_index(step)] += count;
        }
        MarginProfile {
            bin_edges_output_code_steps: MARGIN_BIN_EDGES.to_vec(),
            bin_pair_counts: bins,
            escape_pair_count: self.escape,
            boundary_pressure_1_step_pair_count: self.within_1,
            boundary_pressure_2_step_pair_count: self.within_2,
            minimum_margin_output_code_steps: if self.minimum.is_finite() {
                self.minimum
            } else {
                0.0
            },
            percentile_1_margin_output_code_steps: self.percentile(0.01),
            percentile_5_margin_output_code_steps: self.percentile(0.05),
            median_margin_output_code_steps: self.percentile(0.5),
        }
    }
}

/// The direction in code space along which the operator's result increases.
/// Publishing it means the viewer never has to guess a 45-degree diagonal: for
/// unequal input scales the aligned axis is tilted, and the three lines below
/// are genuinely different lines.
#[derive(Clone, Serialize)]
pub(super) struct SumProjectionAxis {
    /// Gradient of the operator result with respect to (q0, q1).
    code_space_gradient: [f64; 2],
    /// Slope dq0/dq1 of the path where both inputs carry the same real value.
    co_activation_slope_q0_per_q1: f64,
    /// Slope dq0/dq1 of the path of steepest result increase.
    steepest_increase_slope_q0_per_q1: f64,
    /// Slope dq0/dq1 of the level sets, along which the result does not change.
    iso_result_slope_q0_per_q1: f64,
    /// Code pair at which both inputs represent zero.
    zero_crossing_code: [i64; 2],
    output_code_steps_per_unit_result: f64,
}

#[derive(Clone, Serialize)]
struct WorstProjectionPair {
    input_0_code: i64,
    input_1_code: i64,
    real_sum: f64,
    rounded_unclamped_output_code: i64,
    projected_output_code: i64,
    projected_real_value: f64,
    absolute_error: f64,
    absolute_error_output_steps: f64,
    range_class: &'static str,
}

#[derive(Clone, Serialize)]
struct ContainmentFrontierPoint {
    output_zero_point: i64,
    minimum_output_scale: f64,
    scale_ratio_to_current: f64,
    signed_zero_point_delta: i64,
    absolute_zero_point_shift: usize,
    negative_code_capacity: usize,
    positive_code_capacity: usize,
}

#[derive(Clone, Serialize)]
pub(super) struct ContainmentCandidateEvaluation {
    pub(super) design: &'static str,
    pub(super) output_zero_point: i64,
    pub(super) output_scale: f64,
    scale_ratio_to_current: f64,
    signed_zero_point_delta: i64,
    absolute_zero_point_shift: usize,
    output_real_range: [f64; 2],
    rounded_projection_clamp_pair_count: usize,
    distinct_projected_output_code_count: usize,
    projected_output_code_utilization_ratio: f64,
    mean_absolute_projection_error: f64,
    mean_absolute_projection_error_current_steps: f64,
    mean_absolute_projection_error_candidate_steps: f64,
    maximum_absolute_projection_error: f64,
    maximum_absolute_projection_error_current_steps: f64,
    maximum_absolute_projection_error_candidate_steps: f64,
}

#[derive(Clone, Serialize)]
pub(super) struct ResidualAddLatticeRow {
    pub(super) op_index: usize,
    op_name: String,
    fused_activation: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    input_tensor_indices: Vec<i32>,
    input_tensor_names: Vec<String>,
    pub(super) output_tensor_index: Option<i32>,
    pub(super) output_tensor_name: String,
    dtype_triplet: Vec<String>,
    input_scales: Vec<f64>,
    input_zero_points: Vec<i64>,
    pub(super) output_scale: Option<f64>,
    pub(super) output_zero_point: Option<i64>,
    input_code_ranges: Vec<[i64; 2]>,
    output_code_range: Option<[i64; 2]>,
    input_real_ranges: Vec<[f64; 2]>,
    legal_sum_real_range: Option<[f64; 2]>,
    output_real_range: Option<[f64; 2]>,
    continuous_sum_interval_coverage_ratio: Option<f64>,
    input_scale_ratio: Option<f64>,
    output_to_finest_input_step_ratio: Option<f64>,
    output_to_coarsest_input_step_ratio: Option<f64>,
    enumerated_code_pair_count: Option<usize>,
    range_escape_low_pair_count: Option<usize>,
    range_escape_high_pair_count: Option<usize>,
    range_escape_pair_count: Option<usize>,
    range_escape_pair_ratio: Option<f64>,
    rounded_projection_clamp_pair_count: Option<usize>,
    rounded_projection_clamp_pair_ratio: Option<f64>,
    complete_legal_domain_contained: Option<bool>,
    distinct_projected_output_code_count: Option<usize>,
    projected_output_code_utilization_ratio: Option<f64>,
    mean_in_range_rounding_error: Option<f64>,
    mean_in_range_rounding_error_steps: Option<f64>,
    maximum_in_range_rounding_error: Option<f64>,
    maximum_in_range_rounding_error_steps: Option<f64>,
    mean_clamped_projection_error: Option<f64>,
    mean_clamped_projection_error_steps: Option<f64>,
    maximum_clamped_projection_error: Option<f64>,
    maximum_clamped_projection_error_steps: Option<f64>,
    margin_profile: Option<MarginProfile>,
    sum_projection_axis: Option<SumProjectionAxis>,
    output_code_histogram: Vec<usize>,
    tile_size_codes: usize,
    tile_grid_dimension: usize,
    tile_range_escape_pair_counts: Vec<usize>,
    tile_mean_clamped_projection_error_steps: Vec<f64>,
    worst_projection_pair: Option<WorstProjectionPair>,
    containment_candidate_count: Option<usize>,
    containment_frontier: Vec<ContainmentFrontierPoint>,
    pub(super) fixed_zero_point_containment: Option<ContainmentCandidateEvaluation>,
    pub(super) globally_finest_containment: Option<ContainmentCandidateEvaluation>,
    domain_escape_rank: Option<usize>,
    formula: &'static str,
    containment_formula: &'static str,
}

impl ResidualAddLatticeRow {
    fn not_assessed(
        op: &OpInfo,
        reason: String,
        tensors: &[TensorInfo],
        binary_op: BinaryLatticeOp,
    ) -> Self {
        let input_indices = op.inputs.iter().take(2).copied().collect::<Vec<_>>();
        let input_tensors = input_indices
            .iter()
            .filter_map(|index| tensor_at(tensors, *index))
            .collect::<Vec<_>>();
        let output_index = op.outputs.first().copied();
        let output = output_index.and_then(|index| tensor_at(tensors, index));
        let mut dtype_triplet = input_tensors
            .iter()
            .map(|tensor| tensor.dtype.clone())
            .collect::<Vec<_>>();
        if let Some(tensor) = output {
            dtype_triplet.push(tensor.dtype.clone());
        }
        Self {
            op_index: op.index,
            op_name: op.name.clone(),
            fused_activation: op.fused_activation.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            input_tensor_indices: input_indices,
            input_tensor_names: input_tensors
                .iter()
                .map(|tensor| tensor.name.clone())
                .collect(),
            output_tensor_index: output_index,
            output_tensor_name: output.map(|tensor| tensor.name.clone()).unwrap_or_default(),
            dtype_triplet,
            input_scales: Vec::new(),
            input_zero_points: Vec::new(),
            output_scale: None,
            output_zero_point: None,
            input_code_ranges: Vec::new(),
            output_code_range: None,
            input_real_ranges: Vec::new(),
            legal_sum_real_range: None,
            output_real_range: None,
            continuous_sum_interval_coverage_ratio: None,
            input_scale_ratio: None,
            output_to_finest_input_step_ratio: None,
            output_to_coarsest_input_step_ratio: None,
            enumerated_code_pair_count: None,
            range_escape_low_pair_count: None,
            range_escape_high_pair_count: None,
            range_escape_pair_count: None,
            range_escape_pair_ratio: None,
            rounded_projection_clamp_pair_count: None,
            rounded_projection_clamp_pair_ratio: None,
            complete_legal_domain_contained: None,
            distinct_projected_output_code_count: None,
            projected_output_code_utilization_ratio: None,
            mean_in_range_rounding_error: None,
            mean_in_range_rounding_error_steps: None,
            maximum_in_range_rounding_error: None,
            maximum_in_range_rounding_error_steps: None,
            mean_clamped_projection_error: None,
            mean_clamped_projection_error_steps: None,
            maximum_clamped_projection_error: None,
            maximum_clamped_projection_error_steps: None,
            margin_profile: None,
            sum_projection_axis: None,
            output_code_histogram: Vec::new(),
            tile_size_codes: 0,
            tile_grid_dimension: 0,
            tile_range_escape_pair_counts: Vec::new(),
            tile_mean_clamped_projection_error_steps: Vec::new(),
            worst_projection_pair: None,
            containment_candidate_count: None,
            containment_frontier: Vec::new(),
            fixed_zero_point_containment: None,
            globally_finest_containment: None,
            domain_escape_rank: None,
            formula: combine_formula(binary_op),
            containment_formula: "for each legal z: s_min(z)=min_binary64{s>0 | (qmin-z)*s<=legal_sum_min and (qmax-z)*s>=legal_sum_max}; frontier minimizes (abs(z-z_current), s_min)",
        }
    }
}

#[derive(Serialize)]
pub(super) struct QuantizationLatticeAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_operator_count: usize,
    assessed_operator_count: usize,
    unassessed_operator_count: usize,
    residual_add_status: &'static str,
    candidate_add_count: usize,
    assessed_add_count: usize,
    unassessed_add_count: usize,
    /// Enumerable binary operators other than ADD. Kept separate so the ADD
    /// fields above keep meaning exactly what their consumers already assume.
    candidate_binary_count: usize,
    assessed_binary_count: usize,
    unassessed_binary_count: usize,
    binary_status: &'static str,
    range_escape_binary_count: usize,
    complete_domain_containment_binary_count: usize,
    binary_operator_coverage: Vec<BinaryOperatorCoverage>,
    binary_contracts: Vec<ResidualAddLatticeRow>,
    candidate_concatenation_count: usize,
    assessed_concatenation_count: usize,
    unassessed_concatenation_count: usize,
    concatenation_status: &'static str,
    concatenation_range_escape_count: usize,
    concatenation_contracts: Vec<ConcatenationLatticeRow>,
    margin_bin_edges_output_code_steps: Vec<f64>,
    margin_atlas: Vec<MarginAtlasRow>,
    residual_add_enumerated_code_pairs: usize,
    total_enumerated_code_pairs: usize,
    total_enumerated_concatenation_codes: usize,
    range_escape_add_count: usize,
    complete_domain_containment_add_count: usize,
    maximum_range_escape_pair_ratio: Option<f64>,
    maximum_mean_clamped_projection_error_steps: Option<f64>,
    containment_design_add_count: usize,
    fixed_zero_point_containment_add_count: usize,
    fixed_zero_point_scale_expansion_add_count: usize,
    global_zero_point_shift_add_count: usize,
    maximum_fixed_zero_point_scale_ratio: Option<f64>,
    maximum_global_finest_scale_ratio: Option<f64>,
    domain_escape_ranking_op_indices: Vec<usize>,
    pub(super) residual_adds: Vec<ResidualAddLatticeRow>,
    method: &'static str,
    rounding_rule: &'static str,
    containment_formula: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Clone, Copy)]
struct QuantContract {
    tensor_index: usize,
    qmin: i64,
    qmax: i64,
    scale: f64,
    zero_point: i64,
}

impl QuantContract {
    fn real_range(self) -> [f64; 2] {
        [
            (self.qmin - self.zero_point) as f64 * self.scale,
            (self.qmax - self.zero_point) as f64 * self.scale,
        ]
    }
}

struct ContainmentDesign {
    candidate_count: usize,
    frontier: Vec<ContainmentFrontierPoint>,
    fixed_zero_point: Option<ContainmentCandidateEvaluation>,
    globally_finest: Option<ContainmentCandidateEvaluation>,
}

fn lattice_family_status(candidate_count: usize, assessed_count: usize) -> &'static str {
    if candidate_count == 0 {
        "not_applicable"
    } else if assessed_count == candidate_count {
        "assessed"
    } else {
        "partial"
    }
}

pub(super) fn build_quantization_lattice(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> QuantizationLatticeAnalysis {
    // ADD keeps its own array and counts: the published ML-BOM properties, the
    // viewer's independent re-derivation, and the contract checks are all bound
    // to residual ADD semantics. The other enumerable binary operators are
    // reported alongside rather than folded into those fields.
    let mut rows = ops
        .iter()
        .filter(|op| op.name == "ADD")
        .map(|op| assess_binary_lattice(op, tensors, BinaryLatticeOp::Add))
        .collect::<Vec<_>>();
    let binary_contracts = ops
        .iter()
        .filter(|op| op.name != "ADD")
        .filter_map(|op| BinaryLatticeOp::from_name(&op.name).map(|kind| (op, kind)))
        .map(|(op, kind)| assess_binary_lattice(op, tensors, kind))
        .collect::<Vec<_>>();
    let concatenation_contracts = ops
        .iter()
        .filter(|op| op.name == "CONCATENATION")
        .map(|op| assess_concatenation_lattice(op, tensors))
        .collect::<Vec<_>>();
    let candidate_concatenation_count = concatenation_contracts.len();
    let assessed_concatenation_count = concatenation_contracts
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .count();
    let candidate_binary_count = binary_contracts.len();
    let assessed_binary_count = binary_contracts
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .count();
    let mut ranked_positions = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.assessment_status == "assessed")
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    ranked_positions.sort_by(|left_position, right_position| {
        let left = &rows[*left_position];
        let right = &rows[*right_position];
        cmp_f64_desc(
            left.range_escape_pair_ratio.unwrap_or(0.0),
            right.range_escape_pair_ratio.unwrap_or(0.0),
        )
        .then_with(|| {
            cmp_f64_desc(
                left.mean_clamped_projection_error_steps.unwrap_or(0.0),
                right.mean_clamped_projection_error_steps.unwrap_or(0.0),
            )
        })
        .then_with(|| left.op_index.cmp(&right.op_index))
    });
    for (rank, position) in ranked_positions.iter().enumerate() {
        rows[*position].domain_escape_rank = Some(rank + 1);
    }
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let candidate_add_count = rows.len();
    let assessed_add_count = assessed.len();
    let unassessed_add_count = candidate_add_count - assessed_add_count;
    let candidate_operator_count =
        candidate_add_count + candidate_binary_count + candidate_concatenation_count;
    let assessed_operator_count =
        assessed_add_count + assessed_binary_count + assessed_concatenation_count;
    let status = lattice_family_status(candidate_operator_count, assessed_operator_count);
    let residual_add_status = lattice_family_status(candidate_add_count, assessed_add_count);
    let binary_status = lattice_family_status(candidate_binary_count, assessed_binary_count);
    let concatenation_status =
        lattice_family_status(candidate_concatenation_count, assessed_concatenation_count);
    let residual_add_enumerated_code_pairs = assessed
        .iter()
        .filter_map(|row| row.enumerated_code_pair_count)
        .sum::<usize>();
    let assessed_other_binary = binary_contracts
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let total_enumerated_code_pairs = residual_add_enumerated_code_pairs
        + assessed_other_binary
            .iter()
            .filter_map(|row| row.enumerated_code_pair_count)
            .sum::<usize>();
    let total_enumerated_concatenation_codes = concatenation_contracts
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .filter_map(|row| row.enumerated_code_count)
        .sum::<usize>();
    let all_assessed_binary = assessed
        .iter()
        .copied()
        .chain(assessed_other_binary.iter().copied())
        .collect::<Vec<_>>();
    let maximum_range_escape_pair_ratio = max_optional(
        all_assessed_binary
            .iter()
            .filter_map(|row| row.range_escape_pair_ratio),
    );
    let maximum_mean_clamped_projection_error_steps = max_optional(
        all_assessed_binary
            .iter()
            .filter_map(|row| row.mean_clamped_projection_error_steps),
    );
    let margin_atlas = build_margin_atlas(&rows, &binary_contracts, &concatenation_contracts);
    QuantizationLatticeAnalysis {
        schema: LATTICE_SCHEMA,
        method_version: LATTICE_METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        candidate_operator_count,
        assessed_operator_count,
        unassessed_operator_count: candidate_operator_count - assessed_operator_count,
        residual_add_status,
        candidate_add_count,
        assessed_add_count,
        unassessed_add_count,
        candidate_binary_count,
        assessed_binary_count,
        unassessed_binary_count: candidate_binary_count - assessed_binary_count,
        binary_status,
        range_escape_binary_count: assessed_other_binary
            .iter()
            .filter(|row| row.range_escape_pair_count.unwrap_or(0) > 0)
            .count(),
        complete_domain_containment_binary_count: assessed_other_binary
            .iter()
            .filter(|row| row.complete_legal_domain_contained == Some(true))
            .count(),
        binary_operator_coverage: binary_operator_coverage(&binary_contracts),
        binary_contracts,
        candidate_concatenation_count,
        assessed_concatenation_count,
        unassessed_concatenation_count: candidate_concatenation_count - assessed_concatenation_count,
        concatenation_status,
        concatenation_range_escape_count: concatenation_contracts
            .iter()
            .filter(|row| row.range_escape_code_count.unwrap_or(0) > 0)
            .count(),
        concatenation_contracts,
        margin_bin_edges_output_code_steps: MARGIN_BIN_EDGES.to_vec(),
        margin_atlas,
        residual_add_enumerated_code_pairs,
        total_enumerated_code_pairs,
        total_enumerated_concatenation_codes,
        range_escape_add_count: assessed
            .iter()
            .filter(|row| row.range_escape_pair_count.unwrap_or(0) > 0)
            .count(),
        complete_domain_containment_add_count: assessed
            .iter()
            .filter(|row| row.complete_legal_domain_contained == Some(true))
            .count(),
        maximum_range_escape_pair_ratio,
        maximum_mean_clamped_projection_error_steps,
        containment_design_add_count: assessed
            .iter()
            .filter(|row| row.globally_finest_containment.is_some())
            .count(),
        fixed_zero_point_containment_add_count: assessed
            .iter()
            .filter(|row| row.fixed_zero_point_containment.is_some())
            .count(),
        fixed_zero_point_scale_expansion_add_count: assessed
            .iter()
            .filter(|row| {
                row.fixed_zero_point_containment
                    .as_ref()
                    .is_some_and(|candidate| candidate.scale_ratio_to_current > 1.0)
            })
            .count(),
        global_zero_point_shift_add_count: assessed
            .iter()
            .filter(|row| {
                row.globally_finest_containment
                    .as_ref()
                    .is_some_and(|candidate| candidate.absolute_zero_point_shift > 0)
            })
            .count(),
        maximum_fixed_zero_point_scale_ratio: max_optional(assessed.iter().filter_map(|row| {
            row.fixed_zero_point_containment
                .as_ref()
                .map(|candidate| candidate.scale_ratio_to_current)
        })),
        maximum_global_finest_scale_ratio: max_optional(assessed.iter().filter_map(|row| {
            row.globally_finest_containment
                .as_ref()
                .map(|candidate| candidate.scale_ratio_to_current)
        })),
        domain_escape_ranking_op_indices: ranked_positions
            .iter()
            .map(|position| rows[*position].op_index)
            .collect(),
        residual_adds: rows,
        method: "For every TFLite ADD, SUB, MUL, MAXIMUM, or MINIMUM with two per-tensor INT8/UINT8 inputs, one per-tensor INT8/UINT8 output, legal zero-points, and no fused activation, enumerate the complete 256x256 input-code Cartesian product. CONCATENATION inputs are independently enumerated over all 256 legal input codes. Compare each real result with the output endpoint interval and project it to the nearest output lattice using the declared rounding rule. For every legal binary output zero-point, derive the minimum positive binary64 scale whose emitted endpoints contain the complete legal-result interval, retain the non-dominated scale-versus-zero-point-shift frontier, and exhaustively reproject the fixed-zero-point and globally finest containment contracts. Aggregate exact code, histogram, margin-atlas, and 16x16 tile ledgers.",
        rounding_rule: "round_ties_away_from_zero, followed by output-code clamping",
        containment_formula: "for each legal z: s_min(z)=min_binary64{s>0 | (qmin-z)*s<=legal_sum_min and (qmax-z)*s>=legal_sum_max}; frontier minimizes (abs(z-z_current), s_min)",
        interpretation_boundary: "This is exhaustive over legal integer code pairs, not over observed activation values or their probability distribution. Pair ratios are uniform-domain geometry, not runtime saturation frequency, accuracy loss, or task risk. Containment candidates are counterfactual output contracts, not calibration recommendations or safe in-place FlatBuffer patches; changing an output scale or zero-point requires re-exporting and validating every downstream quantization contract. The projection is an ideal real-number nearest-lattice calculation; it does not claim the executed TFLite fixed-point multiplier, kernel rounding path, fused-activation behavior, hardware result, or task accuracy.",
    }
}

fn assess_binary_lattice(
    op: &OpInfo,
    tensors: &[TensorInfo],
    binary_op: BinaryLatticeOp,
) -> ResidualAddLatticeRow {
    if op.inputs.len() < 2 || op.outputs.is_empty() {
        return ResidualAddLatticeRow::not_assessed(
            op,
            format!(
                "{} does not expose two inputs and one output in the parsed graph.",
                op.name
            ),
            tensors,
            binary_op,
        );
    }
    if op.fused_activation != "NONE" {
        return ResidualAddLatticeRow::not_assessed(
            op,
            format!(
                "Fused activation {} is not modeled by quantization-lattice v1.",
                op.fused_activation
            ),
            tensors,
            binary_op,
        );
    }
    let Some(input_0) = tensor_at(tensors, op.inputs[0]) else {
        return ResidualAddLatticeRow::not_assessed(
            op,
            "Input tensor 0 is unavailable.".to_string(),
            tensors,
            binary_op,
        );
    };
    let Some(input_1) = tensor_at(tensors, op.inputs[1]) else {
        return ResidualAddLatticeRow::not_assessed(
            op,
            "Input tensor 1 is unavailable.".to_string(),
            tensors,
            binary_op,
        );
    };
    let Some(output) = tensor_at(tensors, op.outputs[0]) else {
        return ResidualAddLatticeRow::not_assessed(
            op,
            "Output tensor is unavailable.".to_string(),
            tensors,
            binary_op,
        );
    };
    let contracts = [input_0, input_1, output]
        .iter()
        .map(|tensor| quant_contract(tensor))
        .collect::<Result<Vec<_>, _>>();
    let contracts = match contracts {
        Ok(value) => value,
        Err(reason) => return ResidualAddLatticeRow::not_assessed(op, reason, tensors, binary_op),
    };
    enumerate_residual_add(
        op,
        [input_0, input_1],
        output,
        contracts[0],
        contracts[1],
        contracts[2],
        binary_op,
    )
}

fn build_containment_design(
    input_0: QuantContract,
    input_1: QuantContract,
    current_output: QuantContract,
    legal_sum_range: [f64; 2],
    binary_op: BinaryLatticeOp,
) -> ContainmentDesign {
    let mut candidates = (current_output.qmin..=current_output.qmax)
        .filter_map(|zero_point| {
            let scale = minimum_containment_scale(
                legal_sum_range,
                current_output.qmin,
                current_output.qmax,
                zero_point,
            )?;
            let signed_delta = zero_point - current_output.zero_point;
            Some(ContainmentFrontierPoint {
                output_zero_point: zero_point,
                minimum_output_scale: scale,
                scale_ratio_to_current: scale / current_output.scale,
                signed_zero_point_delta: signed_delta,
                absolute_zero_point_shift: signed_delta.unsigned_abs() as usize,
                negative_code_capacity: (zero_point - current_output.qmin) as usize,
                positive_code_capacity: (current_output.qmax - zero_point) as usize,
            })
        })
        .collect::<Vec<_>>();
    let globally_finest_point = candidates
        .iter()
        .min_by(|left, right| {
            left.minimum_output_scale
                .total_cmp(&right.minimum_output_scale)
                .then_with(|| {
                    left.absolute_zero_point_shift
                        .cmp(&right.absolute_zero_point_shift)
                })
                .then_with(|| left.output_zero_point.cmp(&right.output_zero_point))
        })
        .cloned();
    let fixed_zero_point = candidates
        .iter()
        .find(|candidate| candidate.output_zero_point == current_output.zero_point)
        .cloned()
        .map(|candidate| {
            evaluate_containment_candidate(
                "fixed_zero_point_minimum_containment",
                candidate,
                input_0,
                input_1,
                current_output,
                binary_op,
            )
        });
    let globally_finest = globally_finest_point.map(|candidate| {
        evaluate_containment_candidate(
            "globally_finest_minimum_containment",
            candidate,
            input_0,
            input_1,
            current_output,
            binary_op,
        )
    });
    candidates.sort_by(|left, right| {
        left.absolute_zero_point_shift
            .cmp(&right.absolute_zero_point_shift)
            .then_with(|| {
                left.minimum_output_scale
                    .total_cmp(&right.minimum_output_scale)
            })
            .then_with(|| left.output_zero_point.cmp(&right.output_zero_point))
    });
    let mut best_scale = f64::INFINITY;
    let frontier = candidates
        .iter()
        .filter(|candidate| {
            if candidate.minimum_output_scale < best_scale {
                best_scale = candidate.minimum_output_scale;
                true
            } else {
                false
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    ContainmentDesign {
        candidate_count: candidates.len(),
        frontier,
        fixed_zero_point,
        globally_finest,
    }
}

fn minimum_containment_scale(
    legal_sum_range: [f64; 2],
    qmin: i64,
    qmax: i64,
    zero_point: i64,
) -> Option<f64> {
    let negative_capacity = zero_point - qmin;
    let positive_capacity = qmax - zero_point;
    let lower_requirement = if legal_sum_range[0] < 0.0 {
        if negative_capacity == 0 {
            return None;
        }
        -legal_sum_range[0] / negative_capacity as f64
    } else {
        0.0
    };
    let upper_requirement = if legal_sum_range[1] > 0.0 {
        if positive_capacity == 0 {
            return None;
        }
        legal_sum_range[1] / positive_capacity as f64
    } else {
        0.0
    };
    let mut scale = lower_requirement.max(upper_requirement);
    if scale == 0.0 {
        scale = f64::from_bits(1);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    while !contract_contains_range(legal_sum_range, qmin, qmax, zero_point, scale) {
        let next = next_up_positive(scale);
        if !next.is_finite() || next == scale {
            return None;
        }
        scale = next;
    }
    loop {
        let previous = next_down_positive(scale);
        if previous <= 0.0
            || !contract_contains_range(legal_sum_range, qmin, qmax, zero_point, previous)
        {
            break;
        }
        scale = previous;
    }
    Some(scale)
}

fn contract_contains_range(
    legal_sum_range: [f64; 2],
    qmin: i64,
    qmax: i64,
    zero_point: i64,
    scale: f64,
) -> bool {
    (qmin - zero_point) as f64 * scale <= legal_sum_range[0]
        && (qmax - zero_point) as f64 * scale >= legal_sum_range[1]
}

fn next_up_positive(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

fn next_down_positive(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn evaluate_containment_candidate(
    design: &'static str,
    point: ContainmentFrontierPoint,
    input_0: QuantContract,
    input_1: QuantContract,
    current_output: QuantContract,
    binary_op: BinaryLatticeOp,
) -> ContainmentCandidateEvaluation {
    let candidate = QuantContract {
        tensor_index: current_output.tensor_index,
        qmin: current_output.qmin,
        qmax: current_output.qmax,
        scale: point.minimum_output_scale,
        zero_point: point.output_zero_point,
    };
    let output_width = (candidate.qmax - candidate.qmin + 1) as usize;
    let mut used_codes = vec![false; output_width];
    let mut clamp_count = 0usize;
    let mut error_sum = 0.0f64;
    let mut maximum_error = 0.0f64;
    let mut pair_count = 0usize;
    for q0 in input_0.qmin..=input_0.qmax {
        for q1 in input_1.qmin..=input_1.qmax {
            let real_sum = binary_op.combine(
                (q0 - input_0.zero_point) as f64 * input_0.scale,
                (q1 - input_1.zero_point) as f64 * input_1.scale,
            );
            let rounded =
                round_ties_away_from_zero(real_sum / candidate.scale) + candidate.zero_point;
            if rounded < candidate.qmin || rounded > candidate.qmax {
                clamp_count += 1;
            }
            let projected_code = rounded.clamp(candidate.qmin, candidate.qmax);
            used_codes[(projected_code - candidate.qmin) as usize] = true;
            let projected_real = (projected_code - candidate.zero_point) as f64 * candidate.scale;
            let error = (real_sum - projected_real).abs();
            error_sum += error;
            maximum_error = maximum_error.max(error);
            pair_count += 1;
        }
    }
    let mean_error = error_sum / pair_count as f64;
    let distinct_codes = used_codes.iter().filter(|used| **used).count();
    ContainmentCandidateEvaluation {
        design,
        output_zero_point: candidate.zero_point,
        output_scale: candidate.scale,
        scale_ratio_to_current: candidate.scale / current_output.scale,
        signed_zero_point_delta: point.signed_zero_point_delta,
        absolute_zero_point_shift: point.absolute_zero_point_shift,
        output_real_range: candidate.real_range(),
        rounded_projection_clamp_pair_count: clamp_count,
        distinct_projected_output_code_count: distinct_codes,
        projected_output_code_utilization_ratio: distinct_codes as f64 / output_width as f64,
        mean_absolute_projection_error: mean_error,
        mean_absolute_projection_error_current_steps: mean_error / current_output.scale,
        mean_absolute_projection_error_candidate_steps: mean_error / candidate.scale,
        maximum_absolute_projection_error: maximum_error,
        maximum_absolute_projection_error_current_steps: maximum_error / current_output.scale,
        maximum_absolute_projection_error_candidate_steps: maximum_error / candidate.scale,
    }
}

fn enumerate_residual_add(
    op: &OpInfo,
    inputs: [&TensorInfo; 2],
    output: &TensorInfo,
    input_0: QuantContract,
    input_1: QuantContract,
    output_contract: QuantContract,
    binary_op: BinaryLatticeOp,
) -> ResidualAddLatticeRow {
    let output_width = (output_contract.qmax - output_contract.qmin + 1) as usize;
    let grid_dimension = output_width / TILE_SIZE;
    let tile_count = grid_dimension * grid_dimension;
    let mut output_histogram = vec![0usize; output_width];
    let mut tile_escape_counts = vec![0usize; tile_count];
    let mut tile_error_sums = vec![0.0f64; tile_count];
    let mut range_escape_low = 0usize;
    let mut range_escape_high = 0usize;
    let mut clamp_count = 0usize;
    let mut in_range_count = 0usize;
    let mut in_range_error_sum = 0.0f64;
    let mut in_range_error_max = 0.0f64;
    let mut projection_error_sum = 0.0f64;
    let mut projection_error_max = 0.0f64;
    let mut worst = None::<WorstProjectionPair>;
    let mut margin = MarginAccumulator::new();
    let output_real_range = output_contract.real_range();

    for q0 in input_0.qmin..=input_0.qmax {
        for q1 in input_1.qmin..=input_1.qmax {
            let real_sum = binary_op.combine(
                (q0 - input_0.zero_point) as f64 * input_0.scale,
                (q1 - input_1.zero_point) as f64 * input_1.scale,
            );
            let range_class = if real_sum < output_real_range[0] {
                range_escape_low += 1;
                "below_output_range"
            } else if real_sum > output_real_range[1] {
                range_escape_high += 1;
                "above_output_range"
            } else {
                in_range_count += 1;
                "inside_output_range"
            };
            let rounded_unclamped = round_ties_away_from_zero(real_sum / output_contract.scale)
                + output_contract.zero_point;
            if rounded_unclamped < output_contract.qmin || rounded_unclamped > output_contract.qmax
            {
                clamp_count += 1;
            }
            let projected_code =
                rounded_unclamped.clamp(output_contract.qmin, output_contract.qmax);
            let projected_real =
                (projected_code - output_contract.zero_point) as f64 * output_contract.scale;
            let error = (real_sum - projected_real).abs();
            let error_steps = error / output_contract.scale;
            projection_error_sum += error;
            projection_error_max = projection_error_max.max(error);
            if range_class == "inside_output_range" {
                in_range_error_sum += error;
                in_range_error_max = in_range_error_max.max(error);
            }
            // Ideal (unclamped, unrounded) position on the output code axis and
            // its distance to the nearer representable end.
            let ideal_output_code =
                real_sum / output_contract.scale + output_contract.zero_point as f64;
            margin.observe(
                (ideal_output_code - output_contract.qmin as f64)
                    .min(output_contract.qmax as f64 - ideal_output_code),
            );
            let histogram_index = (projected_code - output_contract.qmin) as usize;
            output_histogram[histogram_index] += 1;
            let tile_row = ((q0 - input_0.qmin) as usize) / TILE_SIZE;
            let tile_col = ((q1 - input_1.qmin) as usize) / TILE_SIZE;
            let tile_index = tile_row * grid_dimension + tile_col;
            if range_class != "inside_output_range" {
                tile_escape_counts[tile_index] += 1;
            }
            tile_error_sums[tile_index] += error_steps;
            if worst
                .as_ref()
                .map(|candidate| error > candidate.absolute_error)
                .unwrap_or(true)
            {
                worst = Some(WorstProjectionPair {
                    input_0_code: q0,
                    input_1_code: q1,
                    real_sum,
                    rounded_unclamped_output_code: rounded_unclamped,
                    projected_output_code: projected_code,
                    projected_real_value: projected_real,
                    absolute_error: error,
                    absolute_error_output_steps: error_steps,
                    range_class,
                });
            }
        }
    }

    let pair_count =
        ((input_0.qmax - input_0.qmin + 1) * (input_1.qmax - input_1.qmin + 1)) as usize;
    let tile_pair_count = TILE_SIZE * TILE_SIZE;
    let input_0_range = input_0.real_range();
    let input_1_range = input_1.real_range();
    let sum_range = binary_op.legal_range(input_0_range, input_1_range);
    let containment_design =
        build_containment_design(input_0, input_1, output_contract, sum_range, binary_op);
    let intersection_width =
        (sum_range[1].min(output_real_range[1]) - sum_range[0].max(output_real_range[0])).max(0.0);
    let sum_width = sum_range[1] - sum_range[0];
    let range_escape_count = range_escape_low + range_escape_high;
    let finest_input_scale = input_0.scale.min(input_1.scale);
    let coarsest_input_scale = input_0.scale.max(input_1.scale);
    ResidualAddLatticeRow {
        op_index: op.index,
        op_name: op.name.clone(),
        fused_activation: op.fused_activation.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_tensor_indices: vec![input_0.tensor_index as i32, input_1.tensor_index as i32],
        input_tensor_names: inputs.iter().map(|tensor| tensor.name.clone()).collect(),
        output_tensor_index: Some(output_contract.tensor_index as i32),
        output_tensor_name: output.name.clone(),
        dtype_triplet: vec![
            inputs[0].dtype.clone(),
            inputs[1].dtype.clone(),
            output.dtype.clone(),
        ],
        input_scales: vec![input_0.scale, input_1.scale],
        input_zero_points: vec![input_0.zero_point, input_1.zero_point],
        output_scale: Some(output_contract.scale),
        output_zero_point: Some(output_contract.zero_point),
        input_code_ranges: vec![[input_0.qmin, input_0.qmax], [input_1.qmin, input_1.qmax]],
        output_code_range: Some([output_contract.qmin, output_contract.qmax]),
        input_real_ranges: vec![input_0_range, input_1_range],
        legal_sum_real_range: Some(sum_range),
        output_real_range: Some(output_real_range),
        continuous_sum_interval_coverage_ratio: Some(if sum_width > 0.0 {
            intersection_width / sum_width
        } else if sum_range[0] >= output_real_range[0] && sum_range[0] <= output_real_range[1] {
            1.0
        } else {
            0.0
        }),
        input_scale_ratio: Some(coarsest_input_scale / finest_input_scale),
        output_to_finest_input_step_ratio: Some(output_contract.scale / finest_input_scale),
        output_to_coarsest_input_step_ratio: Some(output_contract.scale / coarsest_input_scale),
        enumerated_code_pair_count: Some(pair_count),
        range_escape_low_pair_count: Some(range_escape_low),
        range_escape_high_pair_count: Some(range_escape_high),
        range_escape_pair_count: Some(range_escape_count),
        range_escape_pair_ratio: Some(range_escape_count as f64 / pair_count as f64),
        rounded_projection_clamp_pair_count: Some(clamp_count),
        rounded_projection_clamp_pair_ratio: Some(clamp_count as f64 / pair_count as f64),
        complete_legal_domain_contained: Some(range_escape_count == 0),
        distinct_projected_output_code_count: Some(
            output_histogram.iter().filter(|count| **count > 0).count(),
        ),
        projected_output_code_utilization_ratio: Some(
            output_histogram.iter().filter(|count| **count > 0).count() as f64
                / output_width as f64,
        ),
        mean_in_range_rounding_error: (in_range_count > 0)
            .then_some(in_range_error_sum / in_range_count as f64),
        mean_in_range_rounding_error_steps: (in_range_count > 0)
            .then_some(in_range_error_sum / in_range_count as f64 / output_contract.scale),
        maximum_in_range_rounding_error: (in_range_count > 0).then_some(in_range_error_max),
        maximum_in_range_rounding_error_steps: (in_range_count > 0)
            .then_some(in_range_error_max / output_contract.scale),
        mean_clamped_projection_error: Some(projection_error_sum / pair_count as f64),
        mean_clamped_projection_error_steps: Some(
            projection_error_sum / pair_count as f64 / output_contract.scale,
        ),
        maximum_clamped_projection_error: Some(projection_error_max),
        maximum_clamped_projection_error_steps: Some(projection_error_max / output_contract.scale),
        margin_profile: Some(margin.finish()),
        sum_projection_axis: Some(SumProjectionAxis {
            code_space_gradient: [input_0.scale, input_1.scale],
            co_activation_slope_q0_per_q1: input_1.scale / input_0.scale,
            steepest_increase_slope_q0_per_q1: input_0.scale / input_1.scale,
            iso_result_slope_q0_per_q1: -(input_1.scale / input_0.scale),
            zero_crossing_code: [input_0.zero_point, input_1.zero_point],
            output_code_steps_per_unit_result: 1.0 / output_contract.scale,
        }),
        output_code_histogram: output_histogram,
        tile_size_codes: TILE_SIZE,
        tile_grid_dimension: grid_dimension,
        tile_range_escape_pair_counts: tile_escape_counts,
        tile_mean_clamped_projection_error_steps: tile_error_sums
            .into_iter()
            .map(|sum| sum / tile_pair_count as f64)
            .collect(),
        worst_projection_pair: worst,
        containment_candidate_count: Some(containment_design.candidate_count),
        containment_frontier: containment_design.frontier,
        fixed_zero_point_containment: containment_design.fixed_zero_point,
        globally_finest_containment: containment_design.globally_finest,
        domain_escape_rank: None,
        formula:
            combine_formula(binary_op),
        containment_formula: "for each legal z: s_min(z)=min_binary64{s>0 | (qmin-z)*s<=legal_sum_min and (qmax-z)*s>=legal_sum_max}; frontier minimizes (abs(z-z_current), s_min)",
    }
}

/// CONCATENATION does not combine its inputs: every input code is requantized
/// into the output contract on its own. The domain is therefore n x 256 single
/// codes rather than a pair grid, which is why it needs its own enumeration
/// instead of the binary one.
#[derive(Clone, Serialize)]
struct ConcatInputProjection {
    input_position: usize,
    tensor_index: usize,
    scale: f64,
    zero_point: i64,
    scale_ratio_to_output: f64,
    input_real_range: [f64; 2],
    enumerated_code_count: usize,
    range_escape_code_count: usize,
    rounded_projection_clamp_code_count: usize,
    distinct_projected_output_code_count: usize,
    mean_absolute_projection_error_output_steps: f64,
    maximum_absolute_projection_error_output_steps: f64,
    legal_domain_contained: bool,
    margin_profile: MarginProfile,
}

#[derive(Clone, Serialize)]
struct ConcatenationLatticeRow {
    op_index: usize,
    op_name: String,
    fused_activation: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    input_count: usize,
    output_tensor_index: Option<usize>,
    output_real_range: Option<[f64; 2]>,
    enumerated_code_count: Option<usize>,
    range_escape_code_count: Option<usize>,
    rounded_projection_clamp_code_count: Option<usize>,
    complete_legal_domain_contained: Option<bool>,
    mean_absolute_projection_error_output_steps: Option<f64>,
    maximum_absolute_projection_error_output_steps: Option<f64>,
    inputs: Vec<ConcatInputProjection>,
    formula: &'static str,
}

const CONCAT_FORMULA: &str =
    "per input i: real=(q-zp_i)*s_i; qout=clamp(round_ties_away(real/sout)+zpout)";

fn assess_concatenation_lattice(op: &OpInfo, tensors: &[TensorInfo]) -> ConcatenationLatticeRow {
    let not_assessed = |reason: String, inputs: usize| ConcatenationLatticeRow {
        op_index: op.index,
        op_name: op.name.clone(),
        fused_activation: op.fused_activation.clone(),
        assessment_status: "not_assessed",
        not_assessed_reason: reason,
        input_count: inputs,
        output_tensor_index: None,
        output_real_range: None,
        enumerated_code_count: None,
        range_escape_code_count: None,
        rounded_projection_clamp_code_count: None,
        complete_legal_domain_contained: None,
        mean_absolute_projection_error_output_steps: None,
        maximum_absolute_projection_error_output_steps: None,
        inputs: Vec::new(),
        formula: CONCAT_FORMULA,
    };
    if op.inputs.is_empty() || op.outputs.is_empty() {
        return not_assessed(
            "CONCATENATION does not expose inputs and one output in the parsed graph.".to_string(),
            op.inputs.len(),
        );
    }
    if op.fused_activation != "NONE" {
        return not_assessed(
            format!(
                "Fused activation {} is not modeled by quantization-lattice v1.",
                op.fused_activation
            ),
            op.inputs.len(),
        );
    }
    let Some(output) = tensor_at(tensors, op.outputs[0]) else {
        return not_assessed("Output tensor is unavailable.".to_string(), op.inputs.len());
    };
    let output_contract = match quant_contract(output) {
        Ok(contract) => contract,
        Err(reason) => return not_assessed(reason, op.inputs.len()),
    };
    let mut input_contracts = Vec::with_capacity(op.inputs.len());
    for index in &op.inputs {
        let Some(tensor) = tensor_at(tensors, *index) else {
            return not_assessed(
                format!("Input tensor {index} is unavailable."),
                op.inputs.len(),
            );
        };
        match quant_contract(tensor) {
            Ok(contract) => input_contracts.push(contract),
            Err(reason) => return not_assessed(reason, op.inputs.len()),
        }
    }

    let output_real_range = output_contract.real_range();
    let output_width = (output_contract.qmax - output_contract.qmin + 1) as usize;
    let mut projections = Vec::with_capacity(input_contracts.len());
    let mut total_codes = 0usize;
    let mut total_escapes = 0usize;
    let mut total_clamps = 0usize;
    let mut total_error_steps = 0.0f64;
    let mut maximum_error_steps = 0.0f64;
    for (position, input) in input_contracts.iter().enumerate() {
        let mut used = vec![false; output_width];
        let mut escapes = 0usize;
        let mut clamps = 0usize;
        let mut error_sum = 0.0f64;
        let mut error_max = 0.0f64;
        let mut count = 0usize;
        let mut margin = MarginAccumulator::new();
        for code in input.qmin..=input.qmax {
            let real = (code - input.zero_point) as f64 * input.scale;
            if real < output_real_range[0] || real > output_real_range[1] {
                escapes += 1;
            }
            let ideal_output_code =
                real / output_contract.scale + output_contract.zero_point as f64;
            margin.observe(
                (ideal_output_code - output_contract.qmin as f64)
                    .min(output_contract.qmax as f64 - ideal_output_code),
            );
            let rounded = round_ties_away_from_zero(real / output_contract.scale)
                + output_contract.zero_point;
            if rounded < output_contract.qmin || rounded > output_contract.qmax {
                clamps += 1;
            }
            let projected = rounded.clamp(output_contract.qmin, output_contract.qmax);
            used[(projected - output_contract.qmin) as usize] = true;
            let projected_real =
                (projected - output_contract.zero_point) as f64 * output_contract.scale;
            let steps = (real - projected_real).abs() / output_contract.scale;
            error_sum += steps;
            error_max = error_max.max(steps);
            count += 1;
        }
        let input_real_range = input.real_range();
        total_codes += count;
        total_escapes += escapes;
        total_clamps += clamps;
        total_error_steps += error_sum;
        maximum_error_steps = maximum_error_steps.max(error_max);
        projections.push(ConcatInputProjection {
            input_position: position,
            tensor_index: input.tensor_index,
            scale: input.scale,
            zero_point: input.zero_point,
            scale_ratio_to_output: input.scale / output_contract.scale,
            input_real_range,
            enumerated_code_count: count,
            range_escape_code_count: escapes,
            rounded_projection_clamp_code_count: clamps,
            distinct_projected_output_code_count: used.iter().filter(|entry| **entry).count(),
            mean_absolute_projection_error_output_steps: if count == 0 {
                0.0
            } else {
                error_sum / count as f64
            },
            maximum_absolute_projection_error_output_steps: error_max,
            legal_domain_contained: escapes == 0,
            margin_profile: margin.finish(),
        });
    }

    ConcatenationLatticeRow {
        op_index: op.index,
        op_name: op.name.clone(),
        fused_activation: op.fused_activation.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_count: input_contracts.len(),
        output_tensor_index: Some(output_contract.tensor_index),
        output_real_range: Some(output_real_range),
        enumerated_code_count: Some(total_codes),
        range_escape_code_count: Some(total_escapes),
        rounded_projection_clamp_code_count: Some(total_clamps),
        complete_legal_domain_contained: Some(total_escapes == 0),
        mean_absolute_projection_error_output_steps: Some(if total_codes == 0 {
            0.0
        } else {
            total_error_steps / total_codes as f64
        }),
        maximum_absolute_projection_error_output_steps: Some(maximum_error_steps),
        inputs: projections,
        formula: CONCAT_FORMULA,
    }
}

/// One comparable row per assessed operator, in graph order. This is the
/// model-wide view: the per-operator lattice answers "how does this operator
/// behave", the atlas answers "which operator loses its margin first".
#[derive(Clone, Serialize)]
struct MarginAtlasRow {
    op_index: usize,
    op_name: String,
    projection_kind: &'static str,
    branch_position: Option<usize>,
    enumerated_count: usize,
    escape_count: usize,
    escape_ratio: f64,
    boundary_pressure_1_step_ratio: f64,
    boundary_pressure_2_step_ratio: f64,
    minimum_margin_output_code_steps: f64,
    percentile_1_margin_output_code_steps: f64,
    percentile_5_margin_output_code_steps: f64,
    median_margin_output_code_steps: f64,
    bin_pair_counts: Vec<usize>,
}

fn atlas_row(
    op_index: usize,
    op_name: &str,
    kind: &'static str,
    branch_position: Option<usize>,
    profile: &MarginProfile,
) -> MarginAtlasRow {
    let total = profile.bin_pair_counts.iter().sum::<usize>().max(1);
    MarginAtlasRow {
        op_index,
        op_name: op_name.to_string(),
        projection_kind: kind,
        branch_position,
        enumerated_count: profile.bin_pair_counts.iter().sum::<usize>(),
        escape_count: profile.escape_pair_count,
        escape_ratio: profile.escape_pair_count as f64 / total as f64,
        boundary_pressure_1_step_ratio: profile.boundary_pressure_1_step_pair_count as f64
            / total as f64,
        boundary_pressure_2_step_ratio: profile.boundary_pressure_2_step_pair_count as f64
            / total as f64,
        minimum_margin_output_code_steps: profile.minimum_margin_output_code_steps,
        percentile_1_margin_output_code_steps: profile.percentile_1_margin_output_code_steps,
        percentile_5_margin_output_code_steps: profile.percentile_5_margin_output_code_steps,
        median_margin_output_code_steps: profile.median_margin_output_code_steps,
        bin_pair_counts: profile.bin_pair_counts.clone(),
    }
}

#[derive(Clone, Serialize)]
struct BinaryOperatorCoverage {
    op_name: String,
    candidate_count: usize,
    assessed_count: usize,
    unassessed_count: usize,
}

fn build_margin_atlas(
    add_rows: &[ResidualAddLatticeRow],
    binary_rows: &[ResidualAddLatticeRow],
    concat_rows: &[ConcatenationLatticeRow],
) -> Vec<MarginAtlasRow> {
    let mut atlas = Vec::new();
    for row in add_rows.iter().chain(binary_rows.iter()) {
        if let Some(profile) = &row.margin_profile {
            atlas.push(atlas_row(
                row.op_index,
                &row.op_name,
                "binary_result",
                None,
                profile,
            ));
        }
    }
    for row in concat_rows {
        for branch in &row.inputs {
            atlas.push(atlas_row(
                row.op_index,
                &row.op_name,
                "concat_branch",
                Some(branch.input_position),
                &branch.margin_profile,
            ));
        }
    }
    atlas.sort_by(|left, right| {
        left.op_index
            .cmp(&right.op_index)
            .then_with(|| left.branch_position.cmp(&right.branch_position))
    });
    atlas
}

fn binary_operator_coverage(rows: &[ResidualAddLatticeRow]) -> Vec<BinaryOperatorCoverage> {
    let mut names: Vec<&str> = rows.iter().map(|row| row.op_name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let candidate_count = rows.iter().filter(|row| row.op_name == name).count();
            let assessed_count = rows
                .iter()
                .filter(|row| row.op_name == name && row.assessment_status == "assessed")
                .count();
            BinaryOperatorCoverage {
                op_name: name.to_string(),
                candidate_count,
                assessed_count,
                unassessed_count: candidate_count - assessed_count,
            }
        })
        .collect()
}

fn combine_formula(binary_op: BinaryLatticeOp) -> &'static str {
    match binary_op {
        BinaryLatticeOp::Add => {
            "real=(q0-zp0)*s0+(q1-zp1)*s1; qout=clamp(round_ties_away(real/sout)+zpout)"
        }
        BinaryLatticeOp::Sub => {
            "real=(q0-zp0)*s0-(q1-zp1)*s1; qout=clamp(round_ties_away(real/sout)+zpout)"
        }
        BinaryLatticeOp::Mul => {
            "real=((q0-zp0)*s0)*((q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)"
        }
        BinaryLatticeOp::Maximum => {
            "real=max((q0-zp0)*s0,(q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)"
        }
        BinaryLatticeOp::Minimum => {
            "real=min((q0-zp0)*s0,(q1-zp1)*s1); qout=clamp(round_ties_away(real/sout)+zpout)"
        }
    }
}

fn quant_contract(tensor: &TensorInfo) -> Result<QuantContract, String> {
    let (qmin, qmax) = match tensor.dtype.as_str() {
        "INT8" => (-128, 127),
        "UINT8" => (0, 255),
        _ => {
            return Err(format!(
                "Tensor {} uses {}; quantization-lattice v1 requires INT8 or UINT8.",
                tensor.index, tensor.dtype
            ))
        }
    };
    if tensor.quant_scales != 1 || tensor.scale_sample.len() != 1 {
        return Err(format!(
            "Tensor {} does not expose exactly one per-tensor quantization scale.",
            tensor.index
        ));
    }
    if tensor.quant_zero_points != 1 || tensor.zero_point_sample.len() != 1 {
        return Err(format!(
            "Tensor {} does not expose exactly one per-tensor zero-point.",
            tensor.index
        ));
    }
    let scale = tensor.scale_sample[0] as f64;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!(
            "Tensor {} has a non-positive or non-finite quantization scale.",
            tensor.index
        ));
    }
    let zero_point = tensor.zero_point_sample[0];
    if zero_point < qmin || zero_point > qmax {
        return Err(format!(
            "Tensor {} zero-point {} lies outside [{}, {}].",
            tensor.index, zero_point, qmin, qmax
        ));
    }
    Ok(QuantContract {
        tensor_index: tensor.index,
        qmin,
        qmax,
        scale,
        zero_point,
    })
}

fn tensor_at(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    usize::try_from(index)
        .ok()
        .and_then(|value| tensors.get(value))
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn max_optional(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_scale_is_binary64_minimal_and_endpoint_safe() {
        let legal_sum_range = [-255.0, 255.0];
        let scale = minimum_containment_scale(legal_sum_range, 0, 255, 128)
            .expect("interior zero-point should contain a signed interval");
        assert!(contract_contains_range(legal_sum_range, 0, 255, 128, scale));
        assert!(!contract_contains_range(
            legal_sum_range,
            0,
            255,
            128,
            next_down_positive(scale)
        ));
        assert!(minimum_containment_scale(legal_sum_range, 0, 255, 0).is_none());
        assert!(minimum_containment_scale(legal_sum_range, 0, 255, 255).is_none());
    }

    #[test]
    fn family_status_is_overall_and_not_add_only() {
        assert_eq!(lattice_family_status(0, 0), "not_applicable");
        assert_eq!(lattice_family_status(1, 0), "partial");
        assert_eq!(lattice_family_status(1, 1), "assessed");
        assert_eq!(BinaryLatticeOp::Mul.combine(-2.0, 3.0), -6.0);
        assert_eq!(
            BinaryLatticeOp::Mul.legal_range([-2.0, 4.0], [-3.0, 5.0]),
            [-12.0, 20.0]
        );
    }

    #[test]
    fn quantized_mobilenet_residual_domain_is_exhaustive_and_conservative() {
        let analysis = super::super::analyze_with_target(
            include_bytes!("../web/samples/mobilenet_v2_1.0_224_quant.tflite"),
            "mobilenet_v2_1.0_224_quant.tflite",
            "android_mid_a55",
        )
        .expect("sample should parse");
        let lattice = analysis.quantization_lattice;
        assert_eq!(lattice.schema, LATTICE_SCHEMA);
        assert_eq!(lattice.candidate_add_count, 10);
        assert_eq!(lattice.assessed_add_count, 10);
        assert_eq!(lattice.unassessed_add_count, 0);
        assert_eq!(lattice.status, "assessed");
        assert_eq!(lattice.residual_add_status, "assessed");
        assert_eq!(lattice.candidate_operator_count, 10);
        assert_eq!(lattice.assessed_operator_count, 10);
        assert_eq!(lattice.residual_add_enumerated_code_pairs, 655_360);
        assert_eq!(lattice.total_enumerated_code_pairs, 655_360);
        assert_eq!(lattice.containment_design_add_count, 10);
        assert_eq!(lattice.global_zero_point_shift_add_count, 10);
        assert_eq!(lattice.residual_adds.len(), 10);
        for row in lattice.residual_adds {
            assert_eq!(row.enumerated_code_pair_count, Some(65_536));
            assert_eq!(row.output_code_histogram.iter().sum::<usize>(), 65_536);
            assert_eq!(row.tile_range_escape_pair_counts.len(), 256);
            assert_eq!(row.tile_mean_clamped_projection_error_steps.len(), 256);
            assert!(row.maximum_in_range_rounding_error_steps.unwrap_or(1.0) <= 0.500_000_1);
            assert_eq!(row.containment_candidate_count, Some(254));
            assert!(!row.containment_frontier.is_empty());
            let candidate = row
                .globally_finest_containment
                .expect("every signed legal-sum interval should have a containment contract");
            assert_eq!(candidate.rounded_projection_clamp_pair_count, 0);
            assert!(candidate.absolute_zero_point_shift > 0);
        }
    }
}
