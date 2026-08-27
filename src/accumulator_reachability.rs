use super::accumulator_atlas::{
    expanded_weight_zero_points, quantized_code_range, raw_code, weight_layout,
};
use super::kernel_witness::{KernelWitnessAnalysis, KernelWitnessOpRow};
use super::requantization_fidelity::{RequantizationFidelityAnalysis, RequantizationOpRow};
use super::rounding_equivalence::{
    rebuild_channel_segments, RoundingEquivalenceAnalysis, RoundingEquivalenceOpRow,
    RoundingPairSegment,
};
use super::{extract_tensor_buffer, OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const SCHEMA: &str = "deepbom.accumulator_reachability.v1";
const METHOD_VERSION: &str = "2026-07-18.1";
const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LEDGER_PREFIX: &[u8] = b"deepbom.accumulator_reachability.v1\0";
const TOP_LIMIT: usize = 16;
const MISSING_U64: u64 = u64::MAX;
const MISSING_I64: i64 = i64::MIN;

#[derive(Clone, Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Clone, Serialize)]
struct DenominationCoverageStep {
    absolute_centered_weight: u64,
    normalized_denomination: u64,
    term_count: usize,
    aggregate_coefficient_capacity_decimal: String,
    reachable_prefix_before_decimal: String,
    reachable_prefix_after_decimal: String,
    coverage_status: &'static str,
}

struct CoverageProof {
    status: &'static str,
    failure_step_index: Option<usize>,
    total_lattice_steps: u64,
    certified_prefix_lattice_steps: u64,
    steps: Vec<DenominationCoverageStep>,
}

#[derive(Clone)]
struct ReachabilityChannelOutcome {
    channel_index: usize,
    term_count: usize,
    nonzero_term_count: usize,
    input_code_span: u64,
    lattice_gcd: u64,
    proof_status: &'static str,
    coverage_failure_step_index: Option<usize>,
    total_lattice_step_count: u64,
    certified_prefix_lattice_step_count: u64,
    interval_state_count: u64,
    lattice_compatible_state_count: u64,
    certified_reachable_state_count: u64,
    provably_unreachable_state_count: u64,
    unresolved_state_count: u64,
    interval_divergent_state_count: u64,
    exact_reachable_divergent_state_count: u64,
    provably_unreachable_divergent_state_count: u64,
    unresolved_divergent_state_count: u64,
    first_exact_reachable_divergent_accumulator: Option<i32>,
    first_default_output_code: Option<i64>,
    first_single_output_code: Option<i64>,
    last_exact_reachable_divergent_accumulator: Option<i32>,
    denominator_steps: Vec<DenominationCoverageStep>,
    first_witness_group_count: usize,
}

#[derive(Clone, Serialize)]
struct ReachabilityChannelWitness {
    channel_index: usize,
    term_count: usize,
    nonzero_term_count: usize,
    input_code_span: u64,
    lattice_gcd: u64,
    proof_status: &'static str,
    coverage_failure_step_index: Option<usize>,
    total_lattice_step_count_decimal: String,
    certified_prefix_lattice_step_count_decimal: String,
    interval_state_count_decimal: String,
    lattice_compatible_state_count_decimal: String,
    certified_reachable_state_count_decimal: String,
    provably_unreachable_state_count_decimal: String,
    unresolved_state_count_decimal: String,
    interval_divergent_state_count_decimal: String,
    exact_reachable_divergent_state_count_decimal: String,
    provably_unreachable_divergent_state_count_decimal: String,
    unresolved_divergent_state_count_decimal: String,
    first_exact_reachable_divergent_accumulator_decimal: Option<String>,
    first_default_output_code: Option<i64>,
    first_single_output_code: Option<i64>,
    last_exact_reachable_divergent_accumulator_decimal: Option<String>,
    denomination_group_count: usize,
    first_exact_reachable_witness_group_count: usize,
}

#[derive(Serialize)]
pub(super) struct AccumulatorReachabilityOpRow {
    pub(super) op_index: usize,
    pub(super) op_name: String,
    pub(super) assessment_status: &'static str,
    not_assessed_reason: String,
    assessed_channel_count: usize,
    complete_integer_interval_channel_count: usize,
    complete_modular_lattice_channel_count: usize,
    partial_band_channel_count: usize,
    singleton_channel_count: usize,
    pub(super) exact_reachable_divergent_channel_count: usize,
    pub(super) unresolved_divergent_channel_count: usize,
    pub(super) interval_only_divergent_channel_count: usize,
    interval_state_count_decimal: String,
    lattice_compatible_state_count_decimal: String,
    certified_reachable_state_count_decimal: String,
    provably_unreachable_state_count_decimal: String,
    unresolved_state_count_decimal: String,
    pub(super) interval_divergent_state_count_decimal: String,
    pub(super) exact_reachable_divergent_state_count_decimal: String,
    pub(super) provably_unreachable_divergent_state_count_decimal: String,
    pub(super) unresolved_divergent_state_count_decimal: String,
    exact_reachable_divergent_ratio: f64,
    maximum_lattice_gcd: Option<u64>,
    channel_lattice_gcds: Vec<u64>,
    channel_proof_statuses: Vec<&'static str>,
    channel_certified_reachable_state_counts_decimal: Vec<String>,
    channel_provably_unreachable_state_counts_decimal: Vec<String>,
    channel_unresolved_state_counts_decimal: Vec<String>,
    pub(super) channel_exact_reachable_divergent_state_counts_decimal: Vec<String>,
    channel_provably_unreachable_divergent_state_counts_decimal: Vec<String>,
    channel_unresolved_divergent_state_counts_decimal: Vec<String>,
    pub(super) channel_first_exact_reachable_divergent_accumulators_decimal: Vec<Option<String>>,
    top_channels: Vec<ReachabilityChannelWitness>,
    source_witness_ledger_sha256: String,
    source_rounding_equivalence_ledger_sha256: String,
    pub(super) reachability_ledger_sha256: String,
    ledger_hash_method: &'static str,
}

impl AccumulatorReachabilityOpRow {
    fn not_assessed(
        row: &RoundingEquivalenceOpRow,
        witness: Option<&KernelWitnessOpRow>,
        reason: String,
    ) -> Self {
        Self {
            op_index: row.op_index,
            op_name: row.op_name.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            assessed_channel_count: 0,
            complete_integer_interval_channel_count: 0,
            complete_modular_lattice_channel_count: 0,
            partial_band_channel_count: 0,
            singleton_channel_count: 0,
            exact_reachable_divergent_channel_count: 0,
            unresolved_divergent_channel_count: 0,
            interval_only_divergent_channel_count: 0,
            interval_state_count_decimal: "0".to_string(),
            lattice_compatible_state_count_decimal: "0".to_string(),
            certified_reachable_state_count_decimal: "0".to_string(),
            provably_unreachable_state_count_decimal: "0".to_string(),
            unresolved_state_count_decimal: "0".to_string(),
            interval_divergent_state_count_decimal: "0".to_string(),
            exact_reachable_divergent_state_count_decimal: "0".to_string(),
            provably_unreachable_divergent_state_count_decimal: "0".to_string(),
            unresolved_divergent_state_count_decimal: "0".to_string(),
            exact_reachable_divergent_ratio: 0.0,
            maximum_lattice_gcd: None,
            channel_lattice_gcds: Vec::new(),
            channel_proof_statuses: Vec::new(),
            channel_certified_reachable_state_counts_decimal: Vec::new(),
            channel_provably_unreachable_state_counts_decimal: Vec::new(),
            channel_unresolved_state_counts_decimal: Vec::new(),
            channel_exact_reachable_divergent_state_counts_decimal: Vec::new(),
            channel_provably_unreachable_divergent_state_counts_decimal: Vec::new(),
            channel_unresolved_divergent_state_counts_decimal: Vec::new(),
            channel_first_exact_reachable_divergent_accumulators_decimal: Vec::new(),
            top_channels: Vec::new(),
            source_witness_ledger_sha256: witness
                .map(|source| source.witness_ledger_sha256.clone())
                .unwrap_or_default(),
            source_rounding_equivalence_ledger_sha256: row.equivalence_ledger_sha256.clone(),
            reachability_ledger_sha256: String::new(),
            ledger_hash_method: ledger_hash_method(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct AccumulatorReachabilityAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    complete_integer_interval_channel_count: usize,
    complete_modular_lattice_channel_count: usize,
    partial_band_channel_count: usize,
    singleton_channel_count: usize,
    exact_reachable_divergent_channel_count: usize,
    unresolved_divergent_channel_count: usize,
    interval_only_divergent_channel_count: usize,
    interval_state_count_decimal: String,
    lattice_compatible_state_count_decimal: String,
    certified_reachable_state_count_decimal: String,
    provably_unreachable_state_count_decimal: String,
    unresolved_state_count_decimal: String,
    interval_divergent_state_count_decimal: String,
    exact_reachable_divergent_state_count_decimal: String,
    provably_unreachable_divergent_state_count_decimal: String,
    unresolved_divergent_state_count_decimal: String,
    exact_reachable_divergent_ratio: f64,
    maximum_lattice_gcd: Option<u64>,
    reachability_ranking_op_indices: Vec<usize>,
    pub(super) ops: Vec<AccumulatorReachabilityOpRow>,
    source_commit: &'static str,
    source_evidence_schema: &'static str,
    source_references: Vec<SourceReference>,
    bounded_sum_proof: &'static str,
    state_conservation: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn accumulator_reachability_not_computed() -> AccumulatorReachabilityAnalysis {
    build_analysis(Vec::new(), "not_computed_internal_scope")
}

pub(super) fn build_accumulator_reachability(
    model_bytes: &[u8],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    witness: &KernelWitnessAnalysis,
    fidelity: &RequantizationFidelityAnalysis,
    equivalence: &RoundingEquivalenceAnalysis,
) -> AccumulatorReachabilityAnalysis {
    let rows = equivalence
        .ops
        .iter()
        .map(|row| {
            let op = ops.iter().find(|op| op.index == row.op_index);
            let witness_row = witness
                .ops
                .iter()
                .find(|source| source.op_index == row.op_index);
            let requant = fidelity
                .ops
                .iter()
                .find(|source| source.op_index == row.op_index);
            match (op, witness_row, requant) {
                (Some(op), Some(witness_row), Some(requant)) => {
                    assess_op(model_bytes, op, tensors, witness_row, requant, row)
                }
                _ => AccumulatorReachabilityOpRow::not_assessed(
                    row,
                    witness_row,
                    "Required graph, weight, Witness, or Requantization source is unavailable."
                        .to_string(),
                ),
            }
        })
        .collect::<Vec<_>>();
    let status = if rows.is_empty() {
        "not_applicable"
    } else if rows.iter().all(|row| row.assessment_status == "assessed") {
        "assessed"
    } else if rows.iter().any(|row| row.assessment_status == "assessed") {
        "partial"
    } else {
        "not_assessed"
    };
    build_analysis(rows, status)
}

fn build_analysis(
    rows: Vec<AccumulatorReachabilityOpRow>,
    status: &'static str,
) -> AccumulatorReachabilityAnalysis {
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let interval_states = sum_decimal(&assessed, |row| &row.interval_state_count_decimal);
    let interval_divergent =
        sum_decimal(&assessed, |row| &row.interval_divergent_state_count_decimal);
    let exact_divergent = sum_decimal(&assessed, |row| {
        &row.exact_reachable_divergent_state_count_decimal
    });
    let mut ranking = assessed.iter().map(|row| row.op_index).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        let left = assessed.iter().find(|row| row.op_index == *left).unwrap();
        let right = assessed.iter().find(|row| row.op_index == *right).unwrap();
        compare_ops(left, right)
    });
    AccumulatorReachabilityAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        candidate_op_count: rows.len(),
        assessed_op_count: assessed.len(),
        unassessed_op_count: rows.len() - assessed.len(),
        assessed_channel_count: assessed.iter().map(|row| row.assessed_channel_count).sum(),
        complete_integer_interval_channel_count: assessed
            .iter()
            .map(|row| row.complete_integer_interval_channel_count)
            .sum(),
        complete_modular_lattice_channel_count: assessed
            .iter()
            .map(|row| row.complete_modular_lattice_channel_count)
            .sum(),
        partial_band_channel_count: assessed
            .iter()
            .map(|row| row.partial_band_channel_count)
            .sum(),
        singleton_channel_count: assessed
            .iter()
            .map(|row| row.singleton_channel_count)
            .sum(),
        exact_reachable_divergent_channel_count: assessed
            .iter()
            .map(|row| row.exact_reachable_divergent_channel_count)
            .sum(),
        unresolved_divergent_channel_count: assessed
            .iter()
            .map(|row| row.unresolved_divergent_channel_count)
            .sum(),
        interval_only_divergent_channel_count: assessed
            .iter()
            .map(|row| row.interval_only_divergent_channel_count)
            .sum(),
        interval_state_count_decimal: interval_states.to_string(),
        lattice_compatible_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.lattice_compatible_state_count_decimal
        })
        .to_string(),
        certified_reachable_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.certified_reachable_state_count_decimal
        })
        .to_string(),
        provably_unreachable_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.provably_unreachable_state_count_decimal
        })
        .to_string(),
        unresolved_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.unresolved_state_count_decimal
        })
        .to_string(),
        interval_divergent_state_count_decimal: interval_divergent.to_string(),
        exact_reachable_divergent_state_count_decimal: exact_divergent.to_string(),
        provably_unreachable_divergent_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.provably_unreachable_divergent_state_count_decimal
        })
        .to_string(),
        unresolved_divergent_state_count_decimal: sum_decimal(&assessed, |row| {
            &row.unresolved_divergent_state_count_decimal
        })
        .to_string(),
        exact_reachable_divergent_ratio: ratio(exact_divergent, interval_divergent),
        maximum_lattice_gcd: assessed.iter().filter_map(|row| row.maximum_lattice_gcd).max(),
        reachability_ranking_op_indices: ranking,
        ops: rows,
        source_commit: SOURCE_COMMIT,
        source_evidence_schema:
            "deepbom.kernel_extremum_witness.v1 + deepbom.rounding_equivalence.v1",
        source_references: source_references(),
        bounded_sum_proof: "From the exact minimum witness, each term contributes |w_i|*d_i with integer d_i in [0,input_code_span]. Equal |w| terms combine into every aggregate coefficient from zero through span*count. After dividing denominations by their gcd, a sorted denomination d with capacity c extends a proven prefix [0,R] to [0,R+d*c] iff d<=R+1. Complementing legal assignments proves a symmetric suffix. If all groups extend, every congruent lattice point in the full accumulator interval is constructively reachable.",
        state_conservation: "For every assessed channel, interval states = certified reachable + provably residue-incompatible + unresolved congruence-compatible. The same disjoint conservation is applied only to rounding-divergent pair segments. Complete lattices have zero unresolved states; partial proofs retain exact endpoint bands without promoting their unresolved middle.",
        method: "Decode stored centered weights, compress absolute denominations, prove bounded-sum coverage, and intersect the resulting arithmetic lattice and certified endpoint bands with the exact fixed-point ordered-pair partition one channel at a time. No full segment corpus is retained in WASM memory.",
        interpretation_boundary: "Certified reachability is exact only for a full-valid kernel-local receptive field whose quantized input codes may vary independently over the legal input code range. It does not prove full-model-input reachability, that upstream model activations can realize the local assignment, that padding positions are free, how frequently the state occurs, whether it changes a declared output, which runtime path executes, or any task-accuracy effect.",
    }
}

fn assess_op(
    model_bytes: &[u8],
    op: &OpInfo,
    tensors: &[TensorInfo],
    witness: &KernelWitnessOpRow,
    requant: &RequantizationOpRow,
    equivalence: &RoundingEquivalenceOpRow,
) -> AccumulatorReachabilityOpRow {
    match assess_op_inner(model_bytes, op, tensors, witness, requant, equivalence) {
        Ok(row) => row,
        Err(reason) => {
            AccumulatorReachabilityOpRow::not_assessed(equivalence, Some(witness), reason)
        }
    }
}

fn assess_op_inner(
    model_bytes: &[u8],
    op: &OpInfo,
    tensors: &[TensorInfo],
    witness: &KernelWitnessOpRow,
    requant: &RequantizationOpRow,
    equivalence: &RoundingEquivalenceOpRow,
) -> Result<AccumulatorReachabilityOpRow, String> {
    if witness.assessment_status != "assessed"
        || requant.assessment_status != "assessed"
        || equivalence.assessment_status != "assessed"
    {
        return Err(
            "Required Witness, Requantization, or Rounding evidence is not assessed.".to_string(),
        );
    }
    let input = required_tensor(tensors, op.inputs.first().copied(), "input")?;
    let weight = required_tensor(tensors, op.inputs.get(1).copied(), "weight")?;
    let (qmin, qmax) = quantized_code_range(&input.dtype)
        .ok_or_else(|| format!("Input tensor {} is not INT8 or UINT8.", input.index))?;
    let input_span =
        u64::try_from(qmax - qmin).map_err(|_| "Input code span is negative.".to_string())?;
    let (weight_qmin, weight_qmax) = quantized_code_range(&weight.dtype)
        .ok_or_else(|| format!("Weight tensor {} is not INT8 or UINT8.", weight.index))?;
    let layout = weight_layout(op, weight)?;
    let channels = layout.channels();
    let terms = layout.terms();
    if witness.assessed_channel_count != channels
        || requant.assessed_channel_count != channels
        || equivalence.assessed_channel_count != channels
    {
        return Err("Source channel cardinality is incomplete.".to_string());
    }
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
    let mut outcomes = Vec::with_capacity(channels);
    let mut ledger = Sha256::new();
    ledger.update(LEDGER_PREFIX);
    ledger.update(witness.witness_ledger_sha256.as_bytes());
    ledger.update(equivalence.equivalence_ledger_sha256.as_bytes());
    for channel in 0..channels {
        let mut denominations = BTreeMap::<u64, usize>::new();
        let zero_point = zero_points[channel];
        for term in 0..terms {
            let centered =
                raw_code(raw_weights[layout.raw_index(channel, term)], &weight.dtype) - zero_point;
            let absolute = centered.unsigned_abs();
            if absolute > 0 {
                *denominations.entry(absolute).or_default() += 1;
            }
        }
        let segments = rebuild_channel_segments(equivalence, requant, channel)?;
        let source = equivalence
            .channel_outcomes
            .get(channel)
            .ok_or_else(|| format!("Rounding channel {channel} is unavailable."))?;
        let outcome = analyze_channel(
            channel,
            terms,
            input_span,
            source.post_bias_minimum,
            source.post_bias_maximum,
            &denominations,
            &segments,
        )?;
        update_channel_ledger(&mut ledger, op.index, &outcome);
        outcomes.push(outcome);
    }
    let sums = ChannelSums::from_outcomes(&outcomes);
    let mut ranked = outcomes.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| compare_channels(left, right));
    let top_channels = ranked
        .into_iter()
        .take(TOP_LIMIT)
        .map(witness_row)
        .collect::<Vec<_>>();
    Ok(AccumulatorReachabilityOpRow {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        assessed_channel_count: channels,
        complete_integer_interval_channel_count: outcomes
            .iter()
            .filter(|row| row.proof_status == "complete_integer_interval")
            .count(),
        complete_modular_lattice_channel_count: outcomes
            .iter()
            .filter(|row| row.proof_status == "complete_modular_lattice")
            .count(),
        partial_band_channel_count: outcomes
            .iter()
            .filter(|row| row.proof_status == "partial_endpoint_bands")
            .count(),
        singleton_channel_count: outcomes
            .iter()
            .filter(|row| row.proof_status == "singleton")
            .count(),
        exact_reachable_divergent_channel_count: outcomes
            .iter()
            .filter(|row| row.exact_reachable_divergent_state_count > 0)
            .count(),
        unresolved_divergent_channel_count: outcomes
            .iter()
            .filter(|row| row.unresolved_divergent_state_count > 0)
            .count(),
        interval_only_divergent_channel_count: outcomes
            .iter()
            .filter(|row| {
                row.interval_divergent_state_count > 0
                    && row.exact_reachable_divergent_state_count == 0
            })
            .count(),
        interval_state_count_decimal: sums.interval.to_string(),
        lattice_compatible_state_count_decimal: sums.compatible.to_string(),
        certified_reachable_state_count_decimal: sums.certified.to_string(),
        provably_unreachable_state_count_decimal: sums.excluded.to_string(),
        unresolved_state_count_decimal: sums.unresolved.to_string(),
        interval_divergent_state_count_decimal: sums.interval_divergent.to_string(),
        exact_reachable_divergent_state_count_decimal: sums.exact_divergent.to_string(),
        provably_unreachable_divergent_state_count_decimal: sums.excluded_divergent.to_string(),
        unresolved_divergent_state_count_decimal: sums.unresolved_divergent.to_string(),
        exact_reachable_divergent_ratio: ratio(sums.exact_divergent, sums.interval_divergent),
        maximum_lattice_gcd: outcomes.iter().map(|row| row.lattice_gcd).max(),
        channel_lattice_gcds: outcomes.iter().map(|row| row.lattice_gcd).collect(),
        channel_proof_statuses: outcomes.iter().map(|row| row.proof_status).collect(),
        channel_certified_reachable_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.certified_reachable_state_count.to_string())
            .collect(),
        channel_provably_unreachable_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.provably_unreachable_state_count.to_string())
            .collect(),
        channel_unresolved_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.unresolved_state_count.to_string())
            .collect(),
        channel_exact_reachable_divergent_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.exact_reachable_divergent_state_count.to_string())
            .collect(),
        channel_provably_unreachable_divergent_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.provably_unreachable_divergent_state_count.to_string())
            .collect(),
        channel_unresolved_divergent_state_counts_decimal: outcomes
            .iter()
            .map(|row| row.unresolved_divergent_state_count.to_string())
            .collect(),
        channel_first_exact_reachable_divergent_accumulators_decimal: outcomes
            .iter()
            .map(|row| {
                row.first_exact_reachable_divergent_accumulator
                    .map(|value| value.to_string())
            })
            .collect(),
        top_channels,
        source_witness_ledger_sha256: witness.witness_ledger_sha256.clone(),
        source_rounding_equivalence_ledger_sha256: equivalence.equivalence_ledger_sha256.clone(),
        reachability_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: ledger_hash_method(),
    })
}

fn analyze_channel(
    channel_index: usize,
    term_count: usize,
    input_span: u64,
    post_bias_minimum: i32,
    post_bias_maximum: i32,
    denominations: &BTreeMap<u64, usize>,
    segments: &[RoundingPairSegment],
) -> Result<ReachabilityChannelOutcome, String> {
    if post_bias_minimum > post_bias_maximum {
        return Err("Post-bias interval is non-monotone.".to_string());
    }
    let interval_state_count = interval_len(post_bias_minimum, post_bias_maximum);
    let nonzero_term_count = denominations.values().sum::<usize>();
    let lattice_gcd = denominations.keys().copied().reduce(gcd).unwrap_or(0);
    let interval_span = u64::try_from(i64::from(post_bias_maximum) - i64::from(post_bias_minimum))
        .map_err(|_| "Accumulator interval span is negative.".to_string())?;
    let expected_span = denominations
        .iter()
        .try_fold(0u64, |total, (weight, count)| {
            let count = u64::try_from(*count).map_err(|_| "Term count exceeds u64.".to_string())?;
            total
                .checked_add(
                    weight
                        .checked_mul(input_span)
                        .and_then(|value| value.checked_mul(count))
                        .ok_or_else(|| "Bounded-sum span exceeds u64.".to_string())?,
                )
                .ok_or_else(|| "Bounded-sum span exceeds u64.".to_string())
        })?;
    if expected_span != interval_span {
        return Err(format!(
            "Stored-weight bounded-sum span {expected_span} does not reproduce accumulator span {interval_span}."
        ));
    }

    let coverage = coverage_proof(denominations, input_span, lattice_gcd)?;
    let proof_status = coverage.status;
    let total_steps = coverage.total_lattice_steps;
    let prefix_steps = coverage.certified_prefix_lattice_steps;
    let lattice_compatible_state_count = if lattice_gcd == 0 { 1 } else { total_steps + 1 };
    let certified_reachable_state_count = match proof_status {
        "singleton" => 1,
        "complete_integer_interval" | "complete_modular_lattice" => lattice_compatible_state_count,
        "partial_endpoint_bands" => prefix_steps
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| "Certified endpoint-band count exceeds u64.".to_string())?,
        _ => return Err("Unknown reachability proof status.".to_string()),
    };
    if certified_reachable_state_count > lattice_compatible_state_count {
        return Err("Certified reachable states exceed lattice-compatible states.".to_string());
    }
    let provably_unreachable_state_count = interval_state_count
        .checked_sub(lattice_compatible_state_count)
        .ok_or_else(|| "Lattice-compatible states exceed the interval.".to_string())?;
    let unresolved_state_count = lattice_compatible_state_count
        .checked_sub(certified_reachable_state_count)
        .ok_or_else(|| "Certified states exceed the compatible lattice.".to_string())?;
    let divergence = intersect_divergence(
        post_bias_minimum,
        post_bias_maximum,
        lattice_gcd,
        proof_status,
        prefix_steps,
        segments,
    )?;
    if divergence.interval != divergence.exact + divergence.excluded + divergence.unresolved {
        return Err("Divergent-state conservation failed.".to_string());
    }
    let first_witness_group_count = divergence
        .first_exact
        .map(|(accumulator, _, _)| {
            build_aggregate_witness(
                accumulator,
                post_bias_minimum,
                lattice_gcd,
                total_steps,
                prefix_steps,
                proof_status,
                &coverage.steps,
            )
        })
        .transpose()?
        .unwrap_or_default();
    Ok(ReachabilityChannelOutcome {
        channel_index,
        term_count,
        nonzero_term_count,
        input_code_span: input_span,
        lattice_gcd,
        proof_status,
        coverage_failure_step_index: coverage.failure_step_index,
        total_lattice_step_count: total_steps,
        certified_prefix_lattice_step_count: prefix_steps,
        interval_state_count,
        lattice_compatible_state_count,
        certified_reachable_state_count,
        provably_unreachable_state_count,
        unresolved_state_count,
        interval_divergent_state_count: divergence.interval,
        exact_reachable_divergent_state_count: divergence.exact,
        provably_unreachable_divergent_state_count: divergence.excluded,
        unresolved_divergent_state_count: divergence.unresolved,
        first_exact_reachable_divergent_accumulator: divergence.first_exact.map(|value| value.0),
        first_default_output_code: divergence.first_exact.map(|value| value.1),
        first_single_output_code: divergence.first_exact.map(|value| value.2),
        last_exact_reachable_divergent_accumulator: divergence.last_exact,
        denominator_steps: coverage.steps,
        first_witness_group_count,
    })
}

fn coverage_proof(
    denominations: &BTreeMap<u64, usize>,
    input_span: u64,
    lattice_gcd: u64,
) -> Result<CoverageProof, String> {
    if denominations.is_empty() {
        return Ok(CoverageProof {
            status: "singleton",
            failure_step_index: None,
            total_lattice_steps: 0,
            certified_prefix_lattice_steps: 0,
            steps: Vec::new(),
        });
    }
    if lattice_gcd == 0 {
        return Err("Nonzero denominations have zero gcd.".to_string());
    }
    let mut reachable = 0u64;
    let mut total = 0u64;
    let mut failure = None;
    let mut steps = Vec::with_capacity(denominations.len());
    for (index, (weight, count)) in denominations.iter().enumerate() {
        let denomination = weight / lattice_gcd;
        let capacity = input_span
            .checked_mul(u64::try_from(*count).map_err(|_| "Term count exceeds u64.")?)
            .ok_or_else(|| "Aggregate coefficient capacity exceeds u64.".to_string())?;
        let contribution = denomination
            .checked_mul(capacity)
            .ok_or_else(|| "Normalized bounded-sum span exceeds u64.".to_string())?;
        total = total
            .checked_add(contribution)
            .ok_or_else(|| "Normalized bounded-sum span exceeds u64.".to_string())?;
        let before = reachable;
        let status = if failure.is_some() {
            "after_gap_not_used"
        } else if denomination <= reachable.saturating_add(1) {
            reachable = reachable
                .checked_add(contribution)
                .ok_or_else(|| "Reachable prefix exceeds u64.".to_string())?;
            "extends_prefix"
        } else {
            failure = Some(index);
            "first_gap"
        };
        steps.push(DenominationCoverageStep {
            absolute_centered_weight: *weight,
            normalized_denomination: denomination,
            term_count: *count,
            aggregate_coefficient_capacity_decimal: capacity.to_string(),
            reachable_prefix_before_decimal: before.to_string(),
            reachable_prefix_after_decimal: reachable.to_string(),
            coverage_status: status,
        });
    }
    let status = if failure.is_none() {
        if lattice_gcd == 1 {
            "complete_integer_interval"
        } else {
            "complete_modular_lattice"
        }
    } else {
        "partial_endpoint_bands"
    };
    if failure.is_none() && reachable != total {
        return Err("Complete coverage does not span the normalized interval.".to_string());
    }
    Ok(CoverageProof {
        status,
        failure_step_index: failure,
        total_lattice_steps: total,
        certified_prefix_lattice_steps: reachable,
        steps,
    })
}

struct DivergenceIntersection {
    interval: u64,
    exact: u64,
    excluded: u64,
    unresolved: u64,
    first_exact: Option<(i32, i64, i64)>,
    last_exact: Option<i32>,
}

fn intersect_divergence(
    minimum: i32,
    maximum: i32,
    lattice_gcd: u64,
    proof_status: &str,
    prefix_steps: u64,
    segments: &[RoundingPairSegment],
) -> Result<DivergenceIntersection, String> {
    let modulus = lattice_gcd.max(1);
    let prefix_maximum = add_steps(minimum, modulus, prefix_steps)?;
    let suffix_minimum = subtract_steps(maximum, modulus, prefix_steps)?;
    let mut result = DivergenceIntersection {
        interval: 0,
        exact: 0,
        excluded: 0,
        unresolved: 0,
        first_exact: None,
        last_exact: None,
    };
    for segment in segments
        .iter()
        .filter(|segment| segment.default_output_code != segment.single_output_code)
    {
        let interval_count = interval_len(segment.accumulator_minimum, segment.accumulator_maximum);
        let compatible = count_progression(
            segment.accumulator_minimum,
            segment.accumulator_maximum,
            minimum,
            modulus,
        );
        let exact = match proof_status {
            "singleton" | "complete_integer_interval" | "complete_modular_lattice" => compatible,
            "partial_endpoint_bands" => {
                let prefix = count_progression(
                    segment.accumulator_minimum,
                    segment.accumulator_maximum.min(prefix_maximum),
                    minimum,
                    modulus,
                );
                let suffix = count_progression(
                    segment.accumulator_minimum.max(suffix_minimum),
                    segment.accumulator_maximum,
                    minimum,
                    modulus,
                );
                prefix
                    .checked_add(suffix)
                    .ok_or_else(|| "Exact divergent count exceeds u64.".to_string())?
            }
            _ => return Err("Unknown reachability proof status.".to_string()),
        };
        let excluded = interval_count
            .checked_sub(compatible)
            .ok_or_else(|| "Compatible divergent count exceeds segment width.".to_string())?;
        let unresolved = compatible
            .checked_sub(exact)
            .ok_or_else(|| "Exact divergent count exceeds compatible count.".to_string())?;
        result.interval += interval_count;
        result.exact += exact;
        result.excluded += excluded;
        result.unresolved += unresolved;
        if exact > 0 {
            let first = first_certified_in_segment(
                segment.accumulator_minimum,
                segment.accumulator_maximum,
                minimum,
                modulus,
                proof_status,
                prefix_maximum,
                suffix_minimum,
            )
            .ok_or_else(|| "Exact divergent witness is missing.".to_string())?;
            result.first_exact.get_or_insert((
                first,
                segment.default_output_code,
                segment.single_output_code,
            ));
            result.last_exact = last_certified_in_segment(
                segment.accumulator_minimum,
                segment.accumulator_maximum,
                minimum,
                modulus,
                proof_status,
                prefix_maximum,
                suffix_minimum,
            );
        }
    }
    Ok(result)
}

fn first_certified_in_segment(
    start: i32,
    end: i32,
    anchor: i32,
    modulus: u64,
    proof_status: &str,
    prefix_maximum: i32,
    suffix_minimum: i32,
) -> Option<i32> {
    if start > end {
        return None;
    }
    match proof_status {
        "singleton" | "complete_integer_interval" | "complete_modular_lattice" => {
            first_progression(start, end, anchor, modulus)
        }
        "partial_endpoint_bands" => {
            first_progression(start, end.min(prefix_maximum), anchor, modulus)
                .or_else(|| first_progression(start.max(suffix_minimum), end, anchor, modulus))
        }
        _ => None,
    }
}

fn last_certified_in_segment(
    start: i32,
    end: i32,
    anchor: i32,
    modulus: u64,
    proof_status: &str,
    prefix_maximum: i32,
    suffix_minimum: i32,
) -> Option<i32> {
    if start > end {
        return None;
    }
    match proof_status {
        "singleton" | "complete_integer_interval" | "complete_modular_lattice" => {
            last_progression(start, end, anchor, modulus)
        }
        "partial_endpoint_bands" => {
            last_progression(start.max(suffix_minimum), end, anchor, modulus)
                .or_else(|| last_progression(start, end.min(prefix_maximum), anchor, modulus))
        }
        _ => None,
    }
}

fn count_progression(start: i32, end: i32, anchor: i32, modulus: u64) -> u64 {
    let Some(first) = first_progression(start, end, anchor, modulus) else {
        return 0;
    };
    let last = last_progression(start, end, anchor, modulus).unwrap();
    (i64::from(last) - i64::from(first)) as u64 / modulus + 1
}

fn first_progression(start: i32, end: i32, anchor: i32, modulus: u64) -> Option<i32> {
    if start > end {
        return None;
    }
    let modulus_i64 = i64::try_from(modulus).ok()?;
    let residue = (i64::from(start) - i64::from(anchor)).rem_euclid(modulus_i64);
    let delta = if residue == 0 {
        0
    } else {
        modulus_i64 - residue
    };
    let value = i64::from(start).checked_add(delta)?;
    (value <= i64::from(end))
        .then(|| i32::try_from(value).ok())
        .flatten()
}

fn last_progression(start: i32, end: i32, anchor: i32, modulus: u64) -> Option<i32> {
    if start > end {
        return None;
    }
    let modulus_i64 = i64::try_from(modulus).ok()?;
    let residue = (i64::from(end) - i64::from(anchor)).rem_euclid(modulus_i64);
    let value = i64::from(end).checked_sub(residue)?;
    (value >= i64::from(start))
        .then(|| i32::try_from(value).ok())
        .flatten()
}

fn add_steps(value: i32, modulus: u64, steps: u64) -> Result<i32, String> {
    let delta = modulus
        .checked_mul(steps)
        .ok_or_else(|| "Lattice endpoint exceeds u64.".to_string())?;
    i32::try_from(
        i64::from(value) + i64::try_from(delta).map_err(|_| "Lattice endpoint exceeds i64.")?,
    )
    .map_err(|_| "Lattice endpoint exceeds int32.".to_string())
}

fn subtract_steps(value: i32, modulus: u64, steps: u64) -> Result<i32, String> {
    let delta = modulus
        .checked_mul(steps)
        .ok_or_else(|| "Lattice endpoint exceeds u64.".to_string())?;
    i32::try_from(
        i64::from(value) - i64::try_from(delta).map_err(|_| "Lattice endpoint exceeds i64.")?,
    )
    .map_err(|_| "Lattice endpoint exceeds int32.".to_string())
}

fn build_aggregate_witness(
    accumulator: i32,
    minimum: i32,
    lattice_gcd: u64,
    total_steps: u64,
    prefix_steps: u64,
    proof_status: &str,
    steps: &[DenominationCoverageStep],
) -> Result<usize, String> {
    Ok(aggregate_witness_coefficients(
        accumulator,
        minimum,
        lattice_gcd,
        total_steps,
        prefix_steps,
        proof_status,
        steps,
    )?
    .into_iter()
    .filter(|coefficient| *coefficient > 0)
    .count())
}

fn aggregate_witness_coefficients(
    accumulator: i32,
    minimum: i32,
    lattice_gcd: u64,
    total_steps: u64,
    prefix_steps: u64,
    proof_status: &str,
    steps: &[DenominationCoverageStep],
) -> Result<Vec<u64>, String> {
    if lattice_gcd == 0 {
        return Ok(Vec::new());
    }
    let offset = u64::try_from(i64::from(accumulator) - i64::from(minimum))
        .map_err(|_| "Reachable witness precedes the accumulator minimum.".to_string())?;
    if offset % lattice_gcd != 0 {
        return Err("Reachable witness is not congruent to the accumulator lattice.".to_string());
    }
    let target = offset / lattice_gcd;
    let use_complement = proof_status == "partial_endpoint_bands" && target > prefix_steps;
    let representation_target = if use_complement {
        total_steps
            .checked_sub(target)
            .ok_or_else(|| "Complement witness underflowed.".to_string())?
    } else {
        target
    };
    let usable = if proof_status == "partial_endpoint_bands" {
        steps
            .iter()
            .take_while(|step| step.coverage_status == "extends_prefix")
            .count()
    } else {
        steps.len()
    };
    let mut coefficients = vec![0u64; steps.len()];
    let mut remaining = representation_target;
    for index in (0..usable).rev() {
        let step = &steps[index];
        let capacity = step
            .aggregate_coefficient_capacity_decimal
            .parse::<u64>()
            .map_err(|_| "Coefficient capacity is not u64.".to_string())?;
        let selected = capacity.min(remaining / step.normalized_denomination);
        coefficients[index] = selected;
        remaining -= selected * step.normalized_denomination;
    }
    if remaining != 0 {
        return Err("Bounded coverage proof did not construct the requested witness.".to_string());
    }
    if use_complement {
        for (index, step) in steps.iter().enumerate() {
            let capacity = step
                .aggregate_coefficient_capacity_decimal
                .parse::<u64>()
                .map_err(|_| "Coefficient capacity is not u64.".to_string())?;
            coefficients[index] = capacity
                .checked_sub(coefficients[index])
                .ok_or_else(|| "Complement coefficient underflowed.".to_string())?;
        }
    }
    Ok(coefficients)
}

#[derive(Debug)]
pub(super) struct ConstructedInputCodes {
    pub(super) codes: Vec<i64>,
    pub(super) centered_inputs: Vec<i64>,
    pub(super) dot_product: i64,
    pub(super) bias: i64,
    pub(super) post_bias_accumulator: i32,
}

pub(super) fn construct_input_codes_for_accumulator(
    centered_weights: &[i64],
    qmin: i64,
    qmax: i64,
    input_zero_point: i64,
    bias: i32,
    target_accumulator: i32,
) -> Result<ConstructedInputCodes, String> {
    if qmin > qmax || input_zero_point < qmin || input_zero_point > qmax {
        return Err("Input code contract is invalid.".to_string());
    }
    let input_span =
        u64::try_from(qmax - qmin).map_err(|_| "Input code span is negative.".to_string())?;
    let mut denominations = BTreeMap::<u64, usize>::new();
    let mut minimum_codes = Vec::with_capacity(centered_weights.len());
    let mut dot_minimum = 0i64;
    for weight in centered_weights {
        let code = if *weight > 0 {
            qmin
        } else if *weight < 0 {
            qmax
        } else {
            input_zero_point
        };
        let centered = code - input_zero_point;
        dot_minimum = dot_minimum
            .checked_add(
                weight
                    .checked_mul(centered)
                    .ok_or_else(|| "Minimum witness term exceeds i64.".to_string())?,
            )
            .ok_or_else(|| "Minimum witness dot product exceeds i64.".to_string())?;
        minimum_codes.push(code);
        let absolute = weight.unsigned_abs();
        if absolute > 0 {
            *denominations.entry(absolute).or_default() += 1;
        }
    }
    let minimum = i32::try_from(
        dot_minimum
            .checked_add(i64::from(bias))
            .ok_or_else(|| "Minimum post-bias accumulator exceeds i64.".to_string())?,
    )
    .map_err(|_| "Minimum post-bias accumulator exceeds int32.".to_string())?;
    let lattice_gcd = denominations.keys().copied().reduce(gcd).unwrap_or(0);
    let coverage = coverage_proof(&denominations, input_span, lattice_gcd)?;
    let coefficients = aggregate_witness_coefficients(
        target_accumulator,
        minimum,
        lattice_gcd,
        coverage.total_lattice_steps,
        coverage.certified_prefix_lattice_steps,
        coverage.status,
        &coverage.steps,
    )?;
    let mut remaining_by_weight = coverage
        .steps
        .iter()
        .zip(coefficients)
        .map(|(step, coefficient)| (step.absolute_centered_weight, coefficient))
        .collect::<BTreeMap<_, _>>();
    let mut codes = Vec::with_capacity(centered_weights.len());
    for (term, weight) in centered_weights.iter().enumerate() {
        let absolute = weight.unsigned_abs();
        let delta = if absolute == 0 {
            0
        } else {
            let remaining = remaining_by_weight
                .get_mut(&absolute)
                .ok_or_else(|| "Constructive coefficient group is unavailable.".to_string())?;
            let selected = (*remaining).min(input_span);
            *remaining -= selected;
            i64::try_from(selected).map_err(|_| "Term delta exceeds i64.".to_string())?
        };
        let code = if *weight > 0 {
            minimum_codes[term]
                .checked_add(delta)
                .ok_or_else(|| "Positive input code exceeds i64.".to_string())?
        } else if *weight < 0 {
            minimum_codes[term]
                .checked_sub(delta)
                .ok_or_else(|| "Negative input code exceeds i64.".to_string())?
        } else {
            minimum_codes[term]
        };
        if code < qmin || code > qmax {
            return Err("Constructed input code escapes the legal range.".to_string());
        }
        codes.push(code);
    }
    if remaining_by_weight
        .values()
        .any(|remaining| *remaining != 0)
    {
        return Err("Constructive coefficient distribution is incomplete.".to_string());
    }
    let centered_inputs = codes
        .iter()
        .map(|code| code - input_zero_point)
        .collect::<Vec<_>>();
    let dot_product =
        centered_weights
            .iter()
            .zip(&centered_inputs)
            .try_fold(0i64, |sum, (weight, input)| {
                sum.checked_add(
                    weight
                        .checked_mul(*input)
                        .ok_or_else(|| "Constructed witness term exceeds i64.".to_string())?,
                )
                .ok_or_else(|| "Constructed witness dot product exceeds i64.".to_string())
            })?;
    let post_bias = dot_product
        .checked_add(i64::from(bias))
        .ok_or_else(|| "Constructed post-bias accumulator exceeds i64.".to_string())?;
    if post_bias != i64::from(target_accumulator) {
        return Err(format!(
            "Constructed witness reproduces accumulator {post_bias}, expected {target_accumulator}."
        ));
    }
    Ok(ConstructedInputCodes {
        codes,
        centered_inputs,
        dot_product,
        bias: i64::from(bias),
        post_bias_accumulator: target_accumulator,
    })
}

fn witness_row(outcome: &ReachabilityChannelOutcome) -> ReachabilityChannelWitness {
    ReachabilityChannelWitness {
        channel_index: outcome.channel_index,
        term_count: outcome.term_count,
        nonzero_term_count: outcome.nonzero_term_count,
        input_code_span: outcome.input_code_span,
        lattice_gcd: outcome.lattice_gcd,
        proof_status: outcome.proof_status,
        coverage_failure_step_index: outcome.coverage_failure_step_index,
        total_lattice_step_count_decimal: outcome.total_lattice_step_count.to_string(),
        certified_prefix_lattice_step_count_decimal: outcome
            .certified_prefix_lattice_step_count
            .to_string(),
        interval_state_count_decimal: outcome.interval_state_count.to_string(),
        lattice_compatible_state_count_decimal: outcome.lattice_compatible_state_count.to_string(),
        certified_reachable_state_count_decimal: outcome
            .certified_reachable_state_count
            .to_string(),
        provably_unreachable_state_count_decimal: outcome
            .provably_unreachable_state_count
            .to_string(),
        unresolved_state_count_decimal: outcome.unresolved_state_count.to_string(),
        interval_divergent_state_count_decimal: outcome.interval_divergent_state_count.to_string(),
        exact_reachable_divergent_state_count_decimal: outcome
            .exact_reachable_divergent_state_count
            .to_string(),
        provably_unreachable_divergent_state_count_decimal: outcome
            .provably_unreachable_divergent_state_count
            .to_string(),
        unresolved_divergent_state_count_decimal: outcome
            .unresolved_divergent_state_count
            .to_string(),
        first_exact_reachable_divergent_accumulator_decimal: outcome
            .first_exact_reachable_divergent_accumulator
            .map(|value| value.to_string()),
        first_default_output_code: outcome.first_default_output_code,
        first_single_output_code: outcome.first_single_output_code,
        last_exact_reachable_divergent_accumulator_decimal: outcome
            .last_exact_reachable_divergent_accumulator
            .map(|value| value.to_string()),
        denomination_group_count: outcome.denominator_steps.len(),
        first_exact_reachable_witness_group_count: outcome.first_witness_group_count,
    }
}

struct ChannelSums {
    interval: u128,
    compatible: u128,
    certified: u128,
    excluded: u128,
    unresolved: u128,
    interval_divergent: u128,
    exact_divergent: u128,
    excluded_divergent: u128,
    unresolved_divergent: u128,
}

impl ChannelSums {
    fn from_outcomes(rows: &[ReachabilityChannelOutcome]) -> Self {
        Self {
            interval: rows
                .iter()
                .map(|row| row.interval_state_count as u128)
                .sum(),
            compatible: rows
                .iter()
                .map(|row| row.lattice_compatible_state_count as u128)
                .sum(),
            certified: rows
                .iter()
                .map(|row| row.certified_reachable_state_count as u128)
                .sum(),
            excluded: rows
                .iter()
                .map(|row| row.provably_unreachable_state_count as u128)
                .sum(),
            unresolved: rows
                .iter()
                .map(|row| row.unresolved_state_count as u128)
                .sum(),
            interval_divergent: rows
                .iter()
                .map(|row| row.interval_divergent_state_count as u128)
                .sum(),
            exact_divergent: rows
                .iter()
                .map(|row| row.exact_reachable_divergent_state_count as u128)
                .sum(),
            excluded_divergent: rows
                .iter()
                .map(|row| row.provably_unreachable_divergent_state_count as u128)
                .sum(),
            unresolved_divergent: rows
                .iter()
                .map(|row| row.unresolved_divergent_state_count as u128)
                .sum(),
        }
    }
}

fn update_channel_ledger(ledger: &mut Sha256, op_index: usize, row: &ReachabilityChannelOutcome) {
    update_u64(ledger, op_index as u64);
    update_u64(ledger, row.channel_index as u64);
    update_u64(ledger, row.term_count as u64);
    update_u64(ledger, row.nonzero_term_count as u64);
    update_u64(ledger, row.input_code_span);
    update_u64(ledger, row.lattice_gcd);
    update_u64(ledger, proof_status_code(row.proof_status));
    update_u64(
        ledger,
        row.coverage_failure_step_index
            .map(|value| value as u64)
            .unwrap_or(MISSING_U64),
    );
    for value in [
        row.total_lattice_step_count,
        row.certified_prefix_lattice_step_count,
        row.interval_state_count,
        row.lattice_compatible_state_count,
        row.certified_reachable_state_count,
        row.provably_unreachable_state_count,
        row.unresolved_state_count,
        row.interval_divergent_state_count,
        row.exact_reachable_divergent_state_count,
        row.provably_unreachable_divergent_state_count,
        row.unresolved_divergent_state_count,
    ] {
        update_u64(ledger, value);
    }
    update_i64(
        ledger,
        row.first_exact_reachable_divergent_accumulator
            .map(i64::from)
            .unwrap_or(MISSING_I64),
    );
    update_i64(ledger, row.first_default_output_code.unwrap_or(MISSING_I64));
    update_i64(ledger, row.first_single_output_code.unwrap_or(MISSING_I64));
    update_i64(
        ledger,
        row.last_exact_reachable_divergent_accumulator
            .map(i64::from)
            .unwrap_or(MISSING_I64),
    );
    update_u64(ledger, row.denominator_steps.len() as u64);
    for step in &row.denominator_steps {
        update_u64(ledger, step.absolute_centered_weight);
        update_u64(ledger, step.normalized_denomination);
        update_u64(ledger, step.term_count as u64);
        update_u64(
            ledger,
            step.aggregate_coefficient_capacity_decimal.parse().unwrap(),
        );
        update_u64(
            ledger,
            step.reachable_prefix_before_decimal.parse().unwrap(),
        );
        update_u64(ledger, step.reachable_prefix_after_decimal.parse().unwrap());
        update_u64(ledger, coverage_status_code(step.coverage_status));
    }
}

fn compare_channels(
    left: &ReachabilityChannelOutcome,
    right: &ReachabilityChannelOutcome,
) -> Ordering {
    right
        .exact_reachable_divergent_state_count
        .cmp(&left.exact_reachable_divergent_state_count)
        .then_with(|| {
            right
                .unresolved_divergent_state_count
                .cmp(&left.unresolved_divergent_state_count)
        })
        .then_with(|| right.lattice_gcd.cmp(&left.lattice_gcd))
        .then_with(|| left.channel_index.cmp(&right.channel_index))
}

fn compare_ops(
    left: &AccumulatorReachabilityOpRow,
    right: &AccumulatorReachabilityOpRow,
) -> Ordering {
    decimal(&right.exact_reachable_divergent_state_count_decimal)
        .cmp(&decimal(
            &left.exact_reachable_divergent_state_count_decimal,
        ))
        .then_with(|| {
            decimal(&right.unresolved_divergent_state_count_decimal)
                .cmp(&decimal(&left.unresolved_divergent_state_count_decimal))
        })
        .then_with(|| left.op_index.cmp(&right.op_index))
}

fn required_tensor<'a>(
    tensors: &'a [TensorInfo],
    index: Option<i32>,
    role: &str,
) -> Result<&'a TensorInfo, String> {
    let index = index.ok_or_else(|| format!("{role} tensor index is absent."))?;
    usize::try_from(index)
        .ok()
        .and_then(|index| tensors.iter().find(|tensor| tensor.index == index))
        .ok_or_else(|| format!("{role} tensor {index} is unavailable."))
}

fn interval_len(minimum: i32, maximum: i32) -> u64 {
    if minimum > maximum {
        0
    } else {
        (i64::from(maximum) - i64::from(minimum) + 1) as u64
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn sum_decimal<F>(rows: &[&AccumulatorReachabilityOpRow], field: F) -> u128
where
    F: Fn(&AccumulatorReachabilityOpRow) -> &String,
{
    rows.iter().map(|row| decimal(field(row))).sum()
}

fn decimal(value: &str) -> u128 {
    value.parse().unwrap_or(0)
}

fn ratio(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn proof_status_code(status: &str) -> u64 {
    match status {
        "singleton" => 0,
        "complete_integer_interval" => 1,
        "complete_modular_lattice" => 2,
        "partial_endpoint_bands" => 3,
        _ => MISSING_U64,
    }
}

fn coverage_status_code(status: &str) -> u64 {
    match status {
        "extends_prefix" => 0,
        "first_gap" => 1,
        "after_gap_not_used" => 2,
        _ => MISSING_U64,
    }
}

fn update_u64(ledger: &mut Sha256, value: u64) {
    ledger.update(value.to_le_bytes());
}

fn update_i64(ledger: &mut Sha256, value: i64) {
    ledger.update(value.to_le_bytes());
}

fn ledger_hash_method() -> &'static str {
    "SHA-256 over the UTF-8 schema prefix, source Witness and Rounding-equivalence digest strings, then channel-major binary rows. Integer fields and denomination steps are unsigned 64-bit LE; accumulator/code witnesses are signed 64-bit LE; missing optionals use the corresponding integer maximum/minimum sentinel."
}

fn source_references() -> Vec<SourceReference> {
    vec![source_reference(
        "quantized_kernel_integer_domain",
        "tensorflow/lite/kernels/internal/common.cc",
        "ba5308bf76383d600d033c948fe0659710939e6f1f15a800b5413e5fc822ddfa",
    )]
}

fn source_reference(
    role: &'static str,
    file: &'static str,
    sha256: &'static str,
) -> SourceReference {
    SourceReference {
        role,
        file,
        url: format!("https://github.com/tensorflow/tensorflow/blob/{SOURCE_COMMIT}/{file}"),
        sha256,
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divergent_segment(minimum: i32, maximum: i32) -> RoundingPairSegment {
        RoundingPairSegment {
            accumulator_minimum: minimum,
            accumulator_maximum: maximum,
            default_output_code: 1,
            single_output_code: 2,
        }
    }

    #[test]
    fn complete_modular_lattice_conserves_divergence() {
        let denominations = BTreeMap::from([(2, 2usize), (4, 1usize)]);
        let outcome = analyze_channel(
            0,
            3,
            2,
            -4,
            12,
            &denominations,
            &[divergent_segment(-4, 12)],
        )
        .unwrap();
        assert_eq!(outcome.proof_status, "complete_modular_lattice");
        assert_eq!(outcome.lattice_gcd, 2);
        assert_eq!(outcome.certified_reachable_state_count, 9);
        assert_eq!(outcome.provably_unreachable_state_count, 8);
        assert_eq!(outcome.exact_reachable_divergent_state_count, 9);
        assert_eq!(outcome.provably_unreachable_divergent_state_count, 8);
        assert_eq!(outcome.unresolved_divergent_state_count, 0);
    }

    #[test]
    fn failed_coverage_preserves_endpoint_bands_and_unknown_middle() {
        let denominations = BTreeMap::from([(2, 1usize), (3, 1usize)]);
        let outcome =
            analyze_channel(0, 2, 1, 0, 5, &denominations, &[divergent_segment(0, 5)]).unwrap();
        assert_eq!(outcome.proof_status, "partial_endpoint_bands");
        assert_eq!(outcome.certified_reachable_state_count, 2);
        assert_eq!(outcome.unresolved_state_count, 4);
        assert_eq!(outcome.exact_reachable_divergent_state_count, 2);
        assert_eq!(outcome.unresolved_divergent_state_count, 4);
        assert_eq!(outcome.first_exact_reachable_divergent_accumulator, Some(0));
        assert_eq!(outcome.last_exact_reachable_divergent_accumulator, Some(5));
    }

    #[test]
    fn zero_weight_channel_is_exact_singleton() {
        let outcome = analyze_channel(
            0,
            4,
            255,
            7,
            7,
            &BTreeMap::new(),
            &[divergent_segment(7, 7)],
        )
        .unwrap();
        assert_eq!(outcome.proof_status, "singleton");
        assert_eq!(outcome.certified_reachable_state_count, 1);
        assert_eq!(outcome.exact_reachable_divergent_state_count, 1);
    }

    #[test]
    fn constructive_codes_reproduce_mixed_sign_accumulator() {
        let witness = construct_input_codes_for_accumulator(&[1, -2, 0], 0, 3, 1, 5, 7)
            .expect("mixed-sign witness");
        assert_eq!(witness.post_bias_accumulator, 7);
        assert_eq!(witness.dot_product + witness.bias, 7);
        assert!(witness.codes.iter().all(|code| (0..=3).contains(code)));
        assert_eq!(witness.codes[2], 1);
        assert_eq!(
            witness
                .centered_inputs
                .iter()
                .zip([1i64, -2, 0])
                .map(|(input, weight)| input * weight)
                .sum::<i64>(),
            witness.dot_product
        );
    }

    #[test]
    fn constructive_codes_reject_unreachable_residue() {
        let error = construct_input_codes_for_accumulator(&[2, 4], 0, 2, 0, 0, 3)
            .expect_err("odd accumulator should be residue-incompatible");
        assert!(error.contains("not congruent"));
    }
}
