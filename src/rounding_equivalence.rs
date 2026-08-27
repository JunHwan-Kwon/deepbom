use super::kernel_witness::{KernelChannelOutcome, KernelWitnessAnalysis, KernelWitnessOpRow};
use super::quantization_math::{
    multiply_by_quantized_multiplier_default, multiply_by_quantized_multiplier_single_rounding,
};
use super::requantization_fidelity::{RequantizationFidelityAnalysis, RequantizationOpRow};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

const SCHEMA: &str = "deepbom.rounding_equivalence.v1";
const METHOD_VERSION: &str = "2026-07-17.1";
const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LEDGER_PREFIX: &[u8] = b"deepbom.rounding_equivalence.v1\0";
const TOP_LIMIT: usize = 16;
const MISSING_I64: i64 = i64::MIN;

#[derive(Clone, Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProjectionPair {
    default_output_code: i64,
    single_output_code: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RoundingPairSegment {
    pub(super) accumulator_minimum: i32,
    pub(super) accumulator_maximum: i32,
    pub(super) default_output_code: i64,
    pub(super) single_output_code: i64,
}

#[derive(Clone)]
pub(super) struct RoundingChannelOutcome {
    pub(super) channel_index: usize,
    pub(super) post_bias_minimum: i32,
    pub(super) post_bias_maximum: i32,
    pub(super) interval_state_count: u64,
    pub(super) divergent_state_count: u64,
    pub(super) default_lower_state_count: u64,
    pub(super) default_higher_state_count: u64,
    pub(super) pair_segment_count: usize,
    pub(super) divergent_region_count: usize,
    pub(super) maximum_absolute_output_delta: i64,
    pub(super) first_divergent_accumulator: Option<i32>,
    pub(super) first_default_output_code: Option<i64>,
    pub(super) first_single_output_code: Option<i64>,
    pub(super) last_divergent_accumulator: Option<i32>,
    pub(super) last_default_output_code: Option<i64>,
    pub(super) last_single_output_code: Option<i64>,
    #[cfg(test)]
    pub(super) segments: Vec<RoundingPairSegment>,
}

#[derive(Clone, Serialize)]
struct RoundingEquivalenceWitness {
    channel_index: usize,
    post_bias_minimum_decimal: String,
    post_bias_maximum_decimal: String,
    interval_state_count_decimal: String,
    divergent_state_count_decimal: String,
    divergent_state_ratio: f64,
    default_lower_state_count_decimal: String,
    default_higher_state_count_decimal: String,
    pair_segment_count: usize,
    divergent_region_count: usize,
    maximum_absolute_output_delta: i64,
    first_divergent_accumulator_decimal: Option<String>,
    first_default_output_code: Option<i64>,
    first_single_output_code: Option<i64>,
    last_divergent_accumulator_decimal: Option<String>,
    last_default_output_code: Option<i64>,
    last_single_output_code: Option<i64>,
    default_quantized_multiplier: i32,
    default_shift: i32,
    single_quantized_multiplier: i32,
    single_shift: i32,
}

#[derive(Clone, Serialize)]
struct DivergenceHistogramBin {
    label: &'static str,
    channel_count: usize,
    interval_state_count_decimal: String,
    divergent_state_count_decimal: String,
}

#[derive(Serialize)]
pub(super) struct RoundingEquivalenceOpRow {
    pub(super) op_index: usize,
    pub(super) op_name: String,
    pub(super) assessment_status: &'static str,
    not_assessed_reason: String,
    pub(super) assessed_channel_count: usize,
    equivalent_channel_count: usize,
    pub(super) divergent_channel_count: usize,
    pub(super) interval_state_count_decimal: String,
    pub(super) divergent_state_count_decimal: String,
    pub(super) divergent_state_ratio: f64,
    default_lower_state_count_decimal: String,
    default_higher_state_count_decimal: String,
    pub(super) maximum_absolute_output_delta: Option<i64>,
    maximum_pair_segment_count: Option<usize>,
    maximum_divergent_region_count: Option<usize>,
    activation_code_range: Option<[i64; 2]>,
    output_zero_point: Option<i64>,
    channel_interval_state_counts_decimal: Vec<String>,
    channel_divergent_state_counts_decimal: Vec<String>,
    channel_default_lower_state_counts_decimal: Vec<String>,
    channel_default_higher_state_counts_decimal: Vec<String>,
    channel_pair_segment_counts: Vec<usize>,
    channel_divergent_region_counts: Vec<usize>,
    channel_maximum_absolute_output_deltas: Vec<i64>,
    channel_first_divergent_accumulators_decimal: Vec<Option<String>>,
    channel_first_default_output_codes: Vec<Option<i64>>,
    channel_first_single_output_codes: Vec<Option<i64>>,
    top_channels: Vec<RoundingEquivalenceWitness>,
    source_witness_ledger_sha256: String,
    source_requantization_ledger_sha256: String,
    pub(super) equivalence_ledger_sha256: String,
    ledger_hash_method: &'static str,
    #[serde(skip)]
    pub(super) channel_outcomes: Vec<RoundingChannelOutcome>,
}

impl RoundingEquivalenceOpRow {
    fn not_assessed(
        witness: &KernelWitnessOpRow,
        requant: Option<&RequantizationOpRow>,
        reason: String,
    ) -> Self {
        Self {
            op_index: witness.op_index,
            op_name: witness.op_name.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            assessed_channel_count: 0,
            equivalent_channel_count: 0,
            divergent_channel_count: 0,
            interval_state_count_decimal: "0".to_string(),
            divergent_state_count_decimal: "0".to_string(),
            divergent_state_ratio: 0.0,
            default_lower_state_count_decimal: "0".to_string(),
            default_higher_state_count_decimal: "0".to_string(),
            maximum_absolute_output_delta: None,
            maximum_pair_segment_count: None,
            maximum_divergent_region_count: None,
            activation_code_range: witness.activation_code_range,
            output_zero_point: requant.and_then(|row| row.output_zero_point),
            channel_interval_state_counts_decimal: Vec::new(),
            channel_divergent_state_counts_decimal: Vec::new(),
            channel_default_lower_state_counts_decimal: Vec::new(),
            channel_default_higher_state_counts_decimal: Vec::new(),
            channel_pair_segment_counts: Vec::new(),
            channel_divergent_region_counts: Vec::new(),
            channel_maximum_absolute_output_deltas: Vec::new(),
            channel_first_divergent_accumulators_decimal: Vec::new(),
            channel_first_default_output_codes: Vec::new(),
            channel_first_single_output_codes: Vec::new(),
            top_channels: Vec::new(),
            source_witness_ledger_sha256: witness.witness_ledger_sha256.clone(),
            source_requantization_ledger_sha256: requant
                .map(|row| row.channel_ledger_sha256.clone())
                .unwrap_or_default(),
            equivalence_ledger_sha256: String::new(),
            ledger_hash_method: ledger_hash_method(),
            channel_outcomes: Vec::new(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct RoundingEquivalenceAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    equivalent_channel_count: usize,
    divergent_channel_count: usize,
    divergent_op_count: usize,
    interval_state_count_decimal: String,
    divergent_state_count_decimal: String,
    divergent_state_ratio: f64,
    default_lower_state_count_decimal: String,
    default_higher_state_count_decimal: String,
    maximum_absolute_output_delta: Option<i64>,
    pair_segment_count: usize,
    divergent_region_count: usize,
    divergence_histogram: Vec<DivergenceHistogramBin>,
    equivalence_ranking_op_indices: Vec<usize>,
    pub(super) ops: Vec<RoundingEquivalenceOpRow>,
    source_commit: &'static str,
    source_evidence_schema: &'static str,
    source_references: Vec<SourceReference>,
    equivalence_proof: &'static str,
    segmentation_bound: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn rounding_equivalence_not_computed() -> RoundingEquivalenceAnalysis {
    build_analysis(Vec::new(), "not_computed_internal_scope")
}

pub(super) fn build_rounding_equivalence(
    witness: &KernelWitnessAnalysis,
    fidelity: &RequantizationFidelityAnalysis,
) -> RoundingEquivalenceAnalysis {
    let rows = witness
        .ops
        .iter()
        .map(|witness_row| {
            let requant = fidelity
                .ops
                .iter()
                .find(|row| row.op_index == witness_row.op_index);
            match requant {
                Some(requant) => assess_op(witness_row, requant),
                None => RoundingEquivalenceOpRow::not_assessed(
                    witness_row,
                    None,
                    "Requantization evidence is unavailable.".to_string(),
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
    rows: Vec<RoundingEquivalenceOpRow>,
    status: &'static str,
) -> RoundingEquivalenceAnalysis {
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let total_states = sum_decimal_rows(&assessed, |row| &row.interval_state_count_decimal);
    let divergent_states = sum_decimal_rows(&assessed, |row| &row.divergent_state_count_decimal);
    let default_lower = sum_decimal_rows(&assessed, |row| &row.default_lower_state_count_decimal);
    let default_higher = sum_decimal_rows(&assessed, |row| &row.default_higher_state_count_decimal);
    let mut ranking = assessed.iter().map(|row| row.op_index).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        let left = rows.iter().find(|row| row.op_index == *left).unwrap();
        let right = rows.iter().find(|row| row.op_index == *right).unwrap();
        compare_op_rows(left, right)
    });
    let outcomes = assessed
        .iter()
        .flat_map(|row| row.channel_outcomes.iter())
        .collect::<Vec<_>>();
    RoundingEquivalenceAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: if status == "not_computed_internal_scope" {
            "NOT_ASSESSABLE"
        } else {
            "DERIVED"
        },
        status,
        candidate_op_count: rows.len(),
        assessed_op_count: assessed.len(),
        unassessed_op_count: rows.len().saturating_sub(assessed.len()),
        assessed_channel_count: assessed.iter().map(|row| row.assessed_channel_count).sum(),
        equivalent_channel_count: assessed.iter().map(|row| row.equivalent_channel_count).sum(),
        divergent_channel_count: assessed.iter().map(|row| row.divergent_channel_count).sum(),
        divergent_op_count: assessed
            .iter()
            .filter(|row| row.divergent_channel_count > 0)
            .count(),
        interval_state_count_decimal: total_states.to_string(),
        divergent_state_count_decimal: divergent_states.to_string(),
        divergent_state_ratio: ratio_u128(divergent_states, total_states),
        default_lower_state_count_decimal: default_lower.to_string(),
        default_higher_state_count_decimal: default_higher.to_string(),
        maximum_absolute_output_delta: assessed
            .iter()
            .filter_map(|row| row.maximum_absolute_output_delta)
            .max(),
        pair_segment_count: assessed
            .iter()
            .map(|row| row.channel_pair_segment_counts.iter().sum::<usize>())
            .sum(),
        divergent_region_count: assessed
            .iter()
            .map(|row| row.channel_divergent_region_counts.iter().sum::<usize>())
            .sum(),
        divergence_histogram: divergence_histogram(&outcomes),
        equivalence_ranking_op_indices: ranking,
        ops: rows,
        source_commit: SOURCE_COMMIT,
        source_evidence_schema: "deepbom.kernel_extremum_witness.v1 + deepbom.requantization_fidelity.v1",
        source_references: source_references(),
        equivalence_proof: "Both pinned positive-multiplier fixed-point paths followed by output-zero-point addition and activation/code clamp are nondecreasing integer step functions. Starting at each segment minimum, binary search finds the greatest accumulator whose ordered output pair is unchanged. Advancing by one partitions every integer in the closed accumulator interval exactly once; a channel is build-mode equivalent iff every segment has equal output codes.",
        segmentation_bound: "Each path can emit at most the fused activation code-span cardinality. The merged ordered-pair partition therefore contains at most default_span + single_span - 1 segments (at most 511 for an 8-bit output), independent of accumulator interval width.",
        method: "For every fixed-point-assessed channel, traverse the exact closed post-bias int32 interval as monotone ordered-output-pair runs, locate every transition by integer binary search, and conserve interval states into equivalent, default-lower, and default-higher ledgers bound to the source Witness and Requantization digests.",
        interpretation_boundary: "This is an exact certificate over every integer in the channel's post-bias accumulator interval hull under the pinned TFLite reference equations. The stored-weight endpoint bounds are exact, but interior integers can be unreachable because legal dot products are discrete and correlated. Counts are therefore exact interval-hull exposure, not observed frequency or reachable-state probability. Runtime build flags, delegate lowering, microkernels, edge padding, model outputs, and task accuracy remain outside this static certificate.",
    }
}

fn assess_op(
    witness: &KernelWitnessOpRow,
    requant: &RequantizationOpRow,
) -> RoundingEquivalenceOpRow {
    if witness.assessment_status != "assessed" {
        return RoundingEquivalenceOpRow::not_assessed(
            witness,
            Some(requant),
            witness.not_assessed_reason.clone(),
        );
    }
    if requant.assessment_status != "assessed" {
        return RoundingEquivalenceOpRow::not_assessed(
            witness,
            Some(requant),
            "Requantization evidence is not assessed.".to_string(),
        );
    }
    match assess_op_inner(witness, requant) {
        Ok(row) => row,
        Err(reason) => RoundingEquivalenceOpRow::not_assessed(witness, Some(requant), reason),
    }
}

fn assess_op_inner(
    witness: &KernelWitnessOpRow,
    requant: &RequantizationOpRow,
) -> Result<RoundingEquivalenceOpRow, String> {
    let channels = witness.assessed_channel_count;
    if witness.channel_outcomes.len() != channels || requant.assessed_channel_count != channels {
        return Err("Source channel cardinality is incomplete.".to_string());
    }
    for array_len in [
        requant.channel_quantized_multipliers.len(),
        requant.channel_shifts.len(),
        requant.channel_single_rounding_quantized_multipliers.len(),
        requant.channel_single_rounding_shifts.len(),
    ] {
        if array_len != channels {
            return Err("Requantization channel arrays are incomplete.".to_string());
        }
    }
    let activation_range = witness
        .activation_code_range
        .ok_or_else(|| "Activation code range is unavailable.".to_string())?;
    let output_zero_point = requant
        .output_zero_point
        .ok_or_else(|| "Output zero-point is unavailable.".to_string())?;
    let mut outcomes = Vec::with_capacity(channels);
    let mut ledger = Sha256::new();
    ledger.update(LEDGER_PREFIX);
    ledger.update(witness.witness_ledger_sha256.as_bytes());
    ledger.update(requant.channel_ledger_sha256.as_bytes());
    for (expected_channel, source) in witness.channel_outcomes.iter().enumerate() {
        if source.channel_index != expected_channel {
            return Err(format!(
                "Kernel witness channel order is non-canonical at channel {expected_channel}."
            ));
        }
        let parameters = ProjectionParameters {
            default_multiplier: requant.channel_quantized_multipliers[expected_channel],
            default_shift: requant.channel_shifts[expected_channel],
            single_multiplier: requant.channel_single_rounding_quantized_multipliers
                [expected_channel],
            single_shift: requant.channel_single_rounding_shifts[expected_channel],
            output_zero_point,
            activation_range,
        };
        let outcome = analyze_channel(source, parameters)?;
        update_channel_ledger(&mut ledger, witness.op_index, &outcome);
        outcomes.push(outcome);
    }
    let total_states = outcomes
        .iter()
        .map(|outcome| outcome.interval_state_count as u128)
        .sum::<u128>();
    let divergent_states = outcomes
        .iter()
        .map(|outcome| outcome.divergent_state_count as u128)
        .sum::<u128>();
    let default_lower = outcomes
        .iter()
        .map(|outcome| outcome.default_lower_state_count as u128)
        .sum::<u128>();
    let default_higher = outcomes
        .iter()
        .map(|outcome| outcome.default_higher_state_count as u128)
        .sum::<u128>();
    let mut ranked = outcomes.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| compare_channel_outcomes(left, right));
    let top_channels = ranked
        .into_iter()
        .take(TOP_LIMIT)
        .map(|outcome| {
            witness_row(
                outcome,
                requant.channel_quantized_multipliers[outcome.channel_index],
                requant.channel_shifts[outcome.channel_index],
                requant.channel_single_rounding_quantized_multipliers[outcome.channel_index],
                requant.channel_single_rounding_shifts[outcome.channel_index],
            )
        })
        .collect::<Vec<_>>();
    Ok(RoundingEquivalenceOpRow {
        op_index: witness.op_index,
        op_name: witness.op_name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        assessed_channel_count: channels,
        equivalent_channel_count: outcomes
            .iter()
            .filter(|outcome| outcome.divergent_state_count == 0)
            .count(),
        divergent_channel_count: outcomes
            .iter()
            .filter(|outcome| outcome.divergent_state_count > 0)
            .count(),
        interval_state_count_decimal: total_states.to_string(),
        divergent_state_count_decimal: divergent_states.to_string(),
        divergent_state_ratio: ratio_u128(divergent_states, total_states),
        default_lower_state_count_decimal: default_lower.to_string(),
        default_higher_state_count_decimal: default_higher.to_string(),
        maximum_absolute_output_delta: outcomes
            .iter()
            .map(|outcome| outcome.maximum_absolute_output_delta)
            .max(),
        maximum_pair_segment_count: outcomes
            .iter()
            .map(|outcome| outcome.pair_segment_count)
            .max(),
        maximum_divergent_region_count: outcomes
            .iter()
            .map(|outcome| outcome.divergent_region_count)
            .max(),
        activation_code_range: Some(activation_range),
        output_zero_point: Some(output_zero_point),
        channel_interval_state_counts_decimal: outcomes
            .iter()
            .map(|outcome| outcome.interval_state_count.to_string())
            .collect(),
        channel_divergent_state_counts_decimal: outcomes
            .iter()
            .map(|outcome| outcome.divergent_state_count.to_string())
            .collect(),
        channel_default_lower_state_counts_decimal: outcomes
            .iter()
            .map(|outcome| outcome.default_lower_state_count.to_string())
            .collect(),
        channel_default_higher_state_counts_decimal: outcomes
            .iter()
            .map(|outcome| outcome.default_higher_state_count.to_string())
            .collect(),
        channel_pair_segment_counts: outcomes
            .iter()
            .map(|outcome| outcome.pair_segment_count)
            .collect(),
        channel_divergent_region_counts: outcomes
            .iter()
            .map(|outcome| outcome.divergent_region_count)
            .collect(),
        channel_maximum_absolute_output_deltas: outcomes
            .iter()
            .map(|outcome| outcome.maximum_absolute_output_delta)
            .collect(),
        channel_first_divergent_accumulators_decimal: outcomes
            .iter()
            .map(|outcome| {
                outcome
                    .first_divergent_accumulator
                    .map(|value| value.to_string())
            })
            .collect(),
        channel_first_default_output_codes: outcomes
            .iter()
            .map(|outcome| outcome.first_default_output_code)
            .collect(),
        channel_first_single_output_codes: outcomes
            .iter()
            .map(|outcome| outcome.first_single_output_code)
            .collect(),
        top_channels,
        source_witness_ledger_sha256: witness.witness_ledger_sha256.clone(),
        source_requantization_ledger_sha256: requant.channel_ledger_sha256.clone(),
        equivalence_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: ledger_hash_method(),
        channel_outcomes: outcomes,
    })
}

#[derive(Clone, Copy)]
struct ProjectionParameters {
    default_multiplier: i32,
    default_shift: i32,
    single_multiplier: i32,
    single_shift: i32,
    output_zero_point: i64,
    activation_range: [i64; 2],
}

pub(super) fn rebuild_channel_segments(
    row: &RoundingEquivalenceOpRow,
    requant: &RequantizationOpRow,
    channel_index: usize,
) -> Result<Vec<RoundingPairSegment>, String> {
    if row.assessment_status != "assessed" || requant.assessment_status != "assessed" {
        return Err("Rounding-equivalence source evidence is not assessed.".to_string());
    }
    let source = row
        .channel_outcomes
        .get(channel_index)
        .ok_or_else(|| format!("Rounding-equivalence channel {channel_index} is unavailable."))?;
    if source.channel_index != channel_index {
        return Err(format!(
            "Rounding-equivalence channel order is non-canonical at channel {channel_index}."
        ));
    }
    let activation_range = row
        .activation_code_range
        .ok_or_else(|| "Activation code range is unavailable.".to_string())?;
    let output_zero_point = requant
        .output_zero_point
        .ok_or_else(|| "Output zero-point is unavailable.".to_string())?;
    let default_multiplier = *requant
        .channel_quantized_multipliers
        .get(channel_index)
        .ok_or_else(|| "Default multiplier is unavailable.".to_string())?;
    let default_shift = *requant
        .channel_shifts
        .get(channel_index)
        .ok_or_else(|| "Default shift is unavailable.".to_string())?;
    let single_multiplier = *requant
        .channel_single_rounding_quantized_multipliers
        .get(channel_index)
        .ok_or_else(|| "Single-rounding multiplier is unavailable.".to_string())?;
    let single_shift = *requant
        .channel_single_rounding_shifts
        .get(channel_index)
        .ok_or_else(|| "Single-rounding shift is unavailable.".to_string())?;
    partition_interval(
        i128::from(source.post_bias_minimum),
        i128::from(source.post_bias_maximum),
        ProjectionParameters {
            default_multiplier,
            default_shift,
            single_multiplier,
            single_shift,
            output_zero_point,
            activation_range,
        },
    )
}

fn analyze_channel(
    source: &KernelChannelOutcome,
    parameters: ProjectionParameters,
) -> Result<RoundingChannelOutcome, String> {
    let minimum = i32::try_from(source.post_bias_minimum)
        .map_err(|_| "Post-bias minimum is outside int32.".to_string())?;
    let maximum = i32::try_from(source.post_bias_maximum)
        .map_err(|_| "Post-bias maximum is outside int32.".to_string())?;
    if minimum > maximum {
        return Err("Post-bias interval is non-monotone.".to_string());
    }
    let interval_state_count = interval_len(minimum, maximum);
    let segments = partition_interval(i128::from(minimum), i128::from(maximum), parameters)?;
    let mut divergent_state_count = 0u64;
    let mut default_lower_state_count = 0u64;
    let mut default_higher_state_count = 0u64;
    let mut divergent_region_count = 0usize;
    let mut maximum_absolute_output_delta = 0i64;
    let mut previous_divergent = false;
    let mut first = None;
    let mut last = None;
    for segment in &segments {
        let count = interval_len(segment.accumulator_minimum, segment.accumulator_maximum);
        let delta = segment.default_output_code - segment.single_output_code;
        let divergent = delta != 0;
        if divergent {
            divergent_state_count += count;
            if delta < 0 {
                default_lower_state_count += count;
            } else {
                default_higher_state_count += count;
            }
            maximum_absolute_output_delta = maximum_absolute_output_delta.max(delta.abs());
            if !previous_divergent {
                divergent_region_count += 1;
            }
            first.get_or_insert((
                segment.accumulator_minimum,
                segment.default_output_code,
                segment.single_output_code,
            ));
            last = Some((
                segment.accumulator_maximum,
                segment.default_output_code,
                segment.single_output_code,
            ));
        }
        previous_divergent = divergent;
    }
    let (first_accumulator, first_default, first_single) = split_witness(first);
    let (last_accumulator, last_default, last_single) = split_witness(last);
    Ok(RoundingChannelOutcome {
        channel_index: source.channel_index,
        post_bias_minimum: minimum,
        post_bias_maximum: maximum,
        interval_state_count,
        divergent_state_count,
        default_lower_state_count,
        default_higher_state_count,
        pair_segment_count: segments.len(),
        divergent_region_count,
        maximum_absolute_output_delta,
        first_divergent_accumulator: first_accumulator,
        first_default_output_code: first_default,
        first_single_output_code: first_single,
        last_divergent_accumulator: last_accumulator,
        last_default_output_code: last_default,
        last_single_output_code: last_single,
        #[cfg(test)]
        segments,
    })
}

fn partition_interval(
    minimum: i128,
    maximum: i128,
    parameters: ProjectionParameters,
) -> Result<Vec<RoundingPairSegment>, String> {
    let minimum =
        i32::try_from(minimum).map_err(|_| "Post-bias minimum is outside int32.".to_string())?;
    let maximum =
        i32::try_from(maximum).map_err(|_| "Post-bias maximum is outside int32.".to_string())?;
    if minimum > maximum {
        return Err("Post-bias interval is non-monotone.".to_string());
    }
    let mut segments = Vec::new();
    let mut cursor = minimum;
    loop {
        let pair = project_pair(cursor, parameters)?;
        let end = find_pair_run_end(cursor, maximum, pair, parameters)?;
        segments.push(RoundingPairSegment {
            accumulator_minimum: cursor,
            accumulator_maximum: end,
            default_output_code: pair.default_output_code,
            single_output_code: pair.single_output_code,
        });
        if end == maximum {
            break;
        }
        cursor = end
            .checked_add(1)
            .ok_or_else(|| "Accumulator segment advance overflowed.".to_string())?;
        if segments.len() > 511 {
            return Err("Ordered output-pair partition exceeds the 8-bit bound.".to_string());
        }
    }
    Ok(segments)
}

fn project_pair(
    accumulator: i32,
    parameters: ProjectionParameters,
) -> Result<ProjectionPair, String> {
    let default_scaled = multiply_by_quantized_multiplier_default(
        accumulator,
        parameters.default_multiplier,
        parameters.default_shift,
    )
    .ok_or_else(|| "Default fixed-point projection is outside its source contract.".to_string())?;
    let single_scaled = multiply_by_quantized_multiplier_single_rounding(
        accumulator,
        parameters.single_multiplier,
        parameters.single_shift,
    )
    .ok_or_else(|| "Single-rounding projection is outside its source contract.".to_string())?;
    let default_preclamp = i64::from(default_scaled) + parameters.output_zero_point;
    let single_preclamp = i64::from(single_scaled) + parameters.output_zero_point;
    Ok(ProjectionPair {
        default_output_code: default_preclamp.clamp(
            parameters.activation_range[0],
            parameters.activation_range[1],
        ),
        single_output_code: single_preclamp.clamp(
            parameters.activation_range[0],
            parameters.activation_range[1],
        ),
    })
}

fn find_pair_run_end(
    start: i32,
    maximum: i32,
    expected: ProjectionPair,
    parameters: ProjectionParameters,
) -> Result<i32, String> {
    if project_pair(maximum, parameters)? == expected {
        return Ok(maximum);
    }
    let mut same = i64::from(start);
    let mut different = i64::from(maximum);
    while same + 1 < different {
        let middle = same + (different - same) / 2;
        let middle = i32::try_from(middle).map_err(|_| "Binary-search midpoint overflowed.")?;
        if project_pair(middle, parameters)? == expected {
            same = i64::from(middle);
        } else {
            different = i64::from(middle);
        }
    }
    i32::try_from(same).map_err(|_| "Binary-search result overflowed.".to_string())
}

fn interval_len(minimum: i32, maximum: i32) -> u64 {
    (i64::from(maximum) - i64::from(minimum) + 1) as u64
}

fn split_witness(witness: Option<(i32, i64, i64)>) -> (Option<i32>, Option<i64>, Option<i64>) {
    match witness {
        Some((accumulator, default, single)) => (Some(accumulator), Some(default), Some(single)),
        None => (None, None, None),
    }
}

fn witness_row(
    outcome: &RoundingChannelOutcome,
    default_multiplier: i32,
    default_shift: i32,
    single_multiplier: i32,
    single_shift: i32,
) -> RoundingEquivalenceWitness {
    RoundingEquivalenceWitness {
        channel_index: outcome.channel_index,
        post_bias_minimum_decimal: outcome.post_bias_minimum.to_string(),
        post_bias_maximum_decimal: outcome.post_bias_maximum.to_string(),
        interval_state_count_decimal: outcome.interval_state_count.to_string(),
        divergent_state_count_decimal: outcome.divergent_state_count.to_string(),
        divergent_state_ratio: ratio_u64(
            outcome.divergent_state_count,
            outcome.interval_state_count,
        ),
        default_lower_state_count_decimal: outcome.default_lower_state_count.to_string(),
        default_higher_state_count_decimal: outcome.default_higher_state_count.to_string(),
        pair_segment_count: outcome.pair_segment_count,
        divergent_region_count: outcome.divergent_region_count,
        maximum_absolute_output_delta: outcome.maximum_absolute_output_delta,
        first_divergent_accumulator_decimal: outcome
            .first_divergent_accumulator
            .map(|value| value.to_string()),
        first_default_output_code: outcome.first_default_output_code,
        first_single_output_code: outcome.first_single_output_code,
        last_divergent_accumulator_decimal: outcome
            .last_divergent_accumulator
            .map(|value| value.to_string()),
        last_default_output_code: outcome.last_default_output_code,
        last_single_output_code: outcome.last_single_output_code,
        default_quantized_multiplier: default_multiplier,
        default_shift,
        single_quantized_multiplier: single_multiplier,
        single_shift,
    }
}

fn compare_channel_outcomes(
    left: &RoundingChannelOutcome,
    right: &RoundingChannelOutcome,
) -> Ordering {
    right
        .maximum_absolute_output_delta
        .cmp(&left.maximum_absolute_output_delta)
        .then_with(|| {
            compare_ratio_desc(
                left.divergent_state_count,
                left.interval_state_count,
                right.divergent_state_count,
                right.interval_state_count,
            )
        })
        .then_with(|| {
            right
                .divergent_region_count
                .cmp(&left.divergent_region_count)
        })
        .then_with(|| left.channel_index.cmp(&right.channel_index))
}

fn compare_op_rows(left: &RoundingEquivalenceOpRow, right: &RoundingEquivalenceOpRow) -> Ordering {
    right
        .maximum_absolute_output_delta
        .unwrap_or(0)
        .cmp(&left.maximum_absolute_output_delta.unwrap_or(0))
        .then_with(|| {
            compare_decimal_ratio_desc(
                &left.divergent_state_count_decimal,
                &left.interval_state_count_decimal,
                &right.divergent_state_count_decimal,
                &right.interval_state_count_decimal,
            )
        })
        .then_with(|| {
            right
                .divergent_channel_count
                .cmp(&left.divergent_channel_count)
        })
        .then_with(|| left.op_index.cmp(&right.op_index))
}

fn compare_ratio_desc(
    left_numerator: u64,
    left_denominator: u64,
    right_numerator: u64,
    right_denominator: u64,
) -> Ordering {
    (right_numerator as u128 * left_denominator as u128)
        .cmp(&(left_numerator as u128 * right_denominator as u128))
}

fn compare_decimal_ratio_desc(
    left_numerator: &str,
    left_denominator: &str,
    right_numerator: &str,
    right_denominator: &str,
) -> Ordering {
    let left_numerator = left_numerator.parse::<u128>().unwrap_or(0);
    let left_denominator = left_denominator.parse::<u128>().unwrap_or(1);
    let right_numerator = right_numerator.parse::<u128>().unwrap_or(0);
    let right_denominator = right_denominator.parse::<u128>().unwrap_or(1);
    (right_numerator * left_denominator).cmp(&(left_numerator * right_denominator))
}

fn divergence_histogram(outcomes: &[&RoundingChannelOutcome]) -> Vec<DivergenceHistogramBin> {
    const LABELS: [&str; 7] = [
        "0%",
        "(0,0.01%]",
        "(0.01%,0.1%]",
        "(0.1%,1%]",
        "(1%,10%]",
        "(10%,50%]",
        "(50%,100%]",
    ];
    let mut counts = [0usize; 7];
    let mut totals = [0u128; 7];
    let mut divergent = [0u128; 7];
    for outcome in outcomes {
        let bin = ratio_bin(outcome.divergent_state_count, outcome.interval_state_count);
        counts[bin] += 1;
        totals[bin] += outcome.interval_state_count as u128;
        divergent[bin] += outcome.divergent_state_count as u128;
    }
    LABELS
        .iter()
        .enumerate()
        .map(|(index, label)| DivergenceHistogramBin {
            label,
            channel_count: counts[index],
            interval_state_count_decimal: totals[index].to_string(),
            divergent_state_count_decimal: divergent[index].to_string(),
        })
        .collect()
}

fn ratio_bin(numerator: u64, denominator: u64) -> usize {
    if numerator == 0 {
        0
    } else if numerator as u128 * 10_000 <= denominator as u128 {
        1
    } else if numerator as u128 * 1_000 <= denominator as u128 {
        2
    } else if numerator as u128 * 100 <= denominator as u128 {
        3
    } else if numerator as u128 * 10 <= denominator as u128 {
        4
    } else if numerator as u128 * 2 <= denominator as u128 {
        5
    } else {
        6
    }
}

fn update_channel_ledger(ledger: &mut Sha256, op_index: usize, outcome: &RoundingChannelOutcome) {
    update_u64(ledger, op_index as u64);
    update_u64(ledger, outcome.channel_index as u64);
    update_i64(ledger, i64::from(outcome.post_bias_minimum));
    update_i64(ledger, i64::from(outcome.post_bias_maximum));
    update_u64(ledger, outcome.interval_state_count);
    update_u64(ledger, outcome.divergent_state_count);
    update_u64(ledger, outcome.default_lower_state_count);
    update_u64(ledger, outcome.default_higher_state_count);
    update_u64(ledger, outcome.pair_segment_count as u64);
    update_u64(ledger, outcome.divergent_region_count as u64);
    update_i64(ledger, outcome.maximum_absolute_output_delta);
    update_i64(
        ledger,
        outcome
            .first_divergent_accumulator
            .map(i64::from)
            .unwrap_or(MISSING_I64),
    );
    update_i64(
        ledger,
        outcome.first_default_output_code.unwrap_or(MISSING_I64),
    );
    update_i64(
        ledger,
        outcome.first_single_output_code.unwrap_or(MISSING_I64),
    );
    update_i64(
        ledger,
        outcome
            .last_divergent_accumulator
            .map(i64::from)
            .unwrap_or(MISSING_I64),
    );
    update_i64(
        ledger,
        outcome.last_default_output_code.unwrap_or(MISSING_I64),
    );
    update_i64(
        ledger,
        outcome.last_single_output_code.unwrap_or(MISSING_I64),
    );
}

fn update_i64(ledger: &mut Sha256, value: i64) {
    ledger.update(value.to_le_bytes());
}

fn update_u64(ledger: &mut Sha256, value: u64) {
    ledger.update(value.to_le_bytes());
}

fn sum_decimal_rows<F>(rows: &[&RoundingEquivalenceOpRow], field: F) -> u128
where
    F: Fn(&RoundingEquivalenceOpRow) -> &String,
{
    rows.iter()
        .filter_map(|row| field(row).parse::<u128>().ok())
        .sum()
}

fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio_u128(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ledger_hash_method() -> &'static str {
    "SHA-256 over UTF-8 schema prefix, source Witness digest, source Requantization digest, then channel-major 17-field binary rows: op/channel and counts as unsigned 64-bit LE; endpoints, deltas, and optional witnesses as signed 64-bit LE. Missing optional values use INT64_MIN."
}

fn source_references() -> Vec<SourceReference> {
    vec![
        source_reference(
            "multiplier_encoding",
            "tensorflow/lite/kernels/internal/quantization_util.cc",
            "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
        ),
        source_reference(
            "activation_range",
            "tensorflow/lite/kernels/kernel_util.cc",
            "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
        ),
        source_reference(
            "default_and_single_rounding_execution",
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

    fn source(minimum: i32, maximum: i32) -> KernelChannelOutcome {
        KernelChannelOutcome {
            channel_index: 0,
            post_bias_minimum: minimum as i128,
            post_bias_maximum: maximum as i128,
            default_minimum_preclamp_code: None,
            default_maximum_preclamp_code: None,
            default_minimum_output_code: None,
            default_maximum_output_code: None,
            single_minimum_preclamp_code: None,
            single_maximum_preclamp_code: None,
            single_minimum_output_code: None,
            single_maximum_output_code: None,
        }
    }

    #[test]
    fn exact_pair_partition_conserves_every_integer() {
        let outcome = analyze_channel(
            &source(-100, 100),
            ProjectionParameters {
                default_multiplier: 1 << 30,
                default_shift: -1,
                single_multiplier: 1 << 30,
                single_shift: -1,
                output_zero_point: 0,
                activation_range: [-128, 127],
            },
        )
        .unwrap();
        assert_eq!(outcome.interval_state_count, 201);
        assert_eq!(
            outcome
                .segments
                .iter()
                .map(|segment| interval_len(
                    segment.accumulator_minimum,
                    segment.accumulator_maximum
                ))
                .sum::<u64>(),
            201
        );
        for pair in outcome.segments.windows(2) {
            assert_eq!(
                pair[0].accumulator_maximum.checked_add(1),
                Some(pair[1].accumulator_minimum)
            );
            assert_ne!(
                (pair[0].default_output_code, pair[0].single_output_code),
                (pair[1].default_output_code, pair[1].single_output_code)
            );
        }
    }

    #[test]
    fn constant_clamped_interval_is_certified_equivalent() {
        let outcome = analyze_channel(
            &source(-50, -1),
            ProjectionParameters {
                default_multiplier: 1 << 30,
                default_shift: 0,
                single_multiplier: 1 << 30,
                single_shift: 0,
                output_zero_point: 0,
                activation_range: [0, 255],
            },
        )
        .unwrap();
        assert_eq!(outcome.segments.len(), 1);
        assert_eq!(outcome.divergent_state_count, 0);
        assert_eq!(outcome.maximum_absolute_output_delta, 0);
    }

    #[test]
    fn full_i32_interval_length_is_representable() {
        assert_eq!(interval_len(i32::MIN, i32::MAX), 1u64 << 32);
    }
}
