use super::kernel_witness::{KernelChannelOutcome, KernelWitnessAnalysis, KernelWitnessOpRow};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

const SCHEMA: &str = "deepbom.channel_vitality.v1";
const METHOD_VERSION: &str = "2026-07-17.1";
const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LEDGER_PREFIX: &[u8] = b"deepbom.channel_vitality.v1\0";
const MISSING_I64: i64 = i64::MIN;
const TOP_CHANNEL_LIMIT: usize = 16;

const REASON_NONCONSTANT: u8 = 0;
const REASON_CONSTANT_ACCUMULATOR: u8 = 1;
const REASON_LOWER_CODE_CLAMP: u8 = 2;
const REASON_UPPER_CODE_CLAMP: u8 = 3;
const REASON_PROJECTION_COLLAPSE: u8 = 4;

#[derive(Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Serialize)]
struct SpanHistogramBin {
    label: &'static str,
    minimum_inclusive_span: usize,
    maximum_inclusive_span: usize,
    default_channel_count: usize,
    single_rounding_channel_count: usize,
}

#[derive(Clone, Serialize)]
struct ChannelVitalityWitness {
    channel_index: usize,
    post_bias_minimum_decimal: String,
    post_bias_maximum_decimal: String,
    accumulator_span_decimal: String,
    post_bias_sign_class: &'static str,
    default_minimum_preclamp_code: Option<i64>,
    default_maximum_preclamp_code: Option<i64>,
    default_minimum_output_code: Option<i64>,
    default_maximum_output_code: Option<i64>,
    default_inclusive_code_span: Option<usize>,
    default_constant_reason: &'static str,
    single_minimum_preclamp_code: Option<i64>,
    single_maximum_preclamp_code: Option<i64>,
    single_minimum_output_code: Option<i64>,
    single_maximum_output_code: Option<i64>,
    single_inclusive_code_span: Option<usize>,
    single_constant_reason: &'static str,
    dual_mode_constant: bool,
    mode_dependent_constant: bool,
}

#[derive(Serialize)]
struct ChannelVitalityOpRow {
    op_index: usize,
    op_name: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    output_code_range: Option<[i64; 2]>,
    activation_code_range: Option<[i64; 2]>,
    assessed_channel_count: usize,
    fixed_point_assessed_channel_count: usize,
    constant_accumulator_channel_count: usize,
    post_bias_negative_locked_channel_count: usize,
    post_bias_positive_locked_channel_count: usize,
    post_bias_zero_containing_channel_count: usize,
    default_constant_output_channel_count: usize,
    single_constant_output_channel_count: usize,
    dual_mode_constant_output_channel_count: usize,
    nonconstant_accumulator_dual_mode_constant_channel_count: usize,
    mode_dependent_constant_output_channel_count: usize,
    default_severely_constrained_channel_count: usize,
    single_severely_constrained_channel_count: usize,
    default_full_activation_span_channel_count: usize,
    single_full_activation_span_channel_count: usize,
    minimum_default_inclusive_code_span: Option<usize>,
    minimum_single_inclusive_code_span: Option<usize>,
    default_minimum_output_codes: Vec<Option<i64>>,
    default_maximum_output_codes: Vec<Option<i64>>,
    single_minimum_output_codes: Vec<Option<i64>>,
    single_maximum_output_codes: Vec<Option<i64>>,
    post_bias_sign_codes: Vec<i8>,
    default_constant_reason_codes: Vec<u8>,
    single_constant_reason_codes: Vec<u8>,
    default_constant_channel_indices: Vec<usize>,
    single_constant_channel_indices: Vec<usize>,
    mode_dependent_constant_channel_indices: Vec<usize>,
    top_channels: Vec<ChannelVitalityWitness>,
    source_witness_ledger_sha256: String,
    vitality_ledger_sha256: String,
    ledger_hash_method: &'static str,
}

#[derive(Serialize)]
pub(super) struct ChannelVitalityAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    candidate_op_count: usize,
    assessed_op_count: usize,
    unassessed_op_count: usize,
    assessed_channel_count: usize,
    fixed_point_assessed_channel_count: usize,
    constant_accumulator_channel_count: usize,
    post_bias_negative_locked_channel_count: usize,
    post_bias_positive_locked_channel_count: usize,
    post_bias_zero_containing_channel_count: usize,
    default_constant_output_channel_count: usize,
    single_constant_output_channel_count: usize,
    dual_mode_constant_output_channel_count: usize,
    nonconstant_accumulator_dual_mode_constant_channel_count: usize,
    mode_dependent_constant_output_channel_count: usize,
    default_severely_constrained_channel_count: usize,
    single_severely_constrained_channel_count: usize,
    default_full_activation_span_channel_count: usize,
    single_full_activation_span_channel_count: usize,
    minimum_default_inclusive_code_span: Option<usize>,
    minimum_single_inclusive_code_span: Option<usize>,
    span_histogram: Vec<SpanHistogramBin>,
    vitality_ranking_op_indices: Vec<usize>,
    ops: Vec<ChannelVitalityOpRow>,
    source_commit: &'static str,
    source_evidence_schema: &'static str,
    source_references: Vec<SourceReference>,
    constant_proof: &'static str,
    span_definition: &'static str,
    reason_code_legend: Vec<&'static str>,
    sign_code_legend: Vec<&'static str>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn channel_vitality_not_computed() -> ChannelVitalityAnalysis {
    build_analysis(Vec::new(), "not_computed_internal_scope")
}

pub(super) fn build_channel_vitality(witness: &KernelWitnessAnalysis) -> ChannelVitalityAnalysis {
    let rows = witness.ops.iter().map(assess_op).collect::<Vec<_>>();
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
    rows: Vec<ChannelVitalityOpRow>,
    status: &'static str,
) -> ChannelVitalityAnalysis {
    let assessed = rows
        .iter()
        .filter(|row| row.assessment_status == "assessed")
        .collect::<Vec<_>>();
    let mut ranking = assessed.iter().map(|row| row.op_index).collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        let left = rows.iter().find(|row| row.op_index == *left).unwrap();
        let right = rows.iter().find(|row| row.op_index == *right).unwrap();
        compare_ops(left, right)
    });
    let default_spans = assessed
        .iter()
        .flat_map(|row| {
            spans(
                &row.default_minimum_output_codes,
                &row.default_maximum_output_codes,
            )
        })
        .collect::<Vec<_>>();
    let single_spans = assessed
        .iter()
        .flat_map(|row| {
            spans(
                &row.single_minimum_output_codes,
                &row.single_maximum_output_codes,
            )
        })
        .collect::<Vec<_>>();
    ChannelVitalityAnalysis {
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
        assessed_channel_count: sum(&assessed, |row| row.assessed_channel_count),
        fixed_point_assessed_channel_count: sum(&assessed, |row| {
            row.fixed_point_assessed_channel_count
        }),
        constant_accumulator_channel_count: sum(&assessed, |row| {
            row.constant_accumulator_channel_count
        }),
        post_bias_negative_locked_channel_count: sum(&assessed, |row| {
            row.post_bias_negative_locked_channel_count
        }),
        post_bias_positive_locked_channel_count: sum(&assessed, |row| {
            row.post_bias_positive_locked_channel_count
        }),
        post_bias_zero_containing_channel_count: sum(&assessed, |row| {
            row.post_bias_zero_containing_channel_count
        }),
        default_constant_output_channel_count: sum(&assessed, |row| {
            row.default_constant_output_channel_count
        }),
        single_constant_output_channel_count: sum(&assessed, |row| {
            row.single_constant_output_channel_count
        }),
        dual_mode_constant_output_channel_count: sum(&assessed, |row| {
            row.dual_mode_constant_output_channel_count
        }),
        nonconstant_accumulator_dual_mode_constant_channel_count: sum(&assessed, |row| {
            row.nonconstant_accumulator_dual_mode_constant_channel_count
        }),
        mode_dependent_constant_output_channel_count: sum(&assessed, |row| {
            row.mode_dependent_constant_output_channel_count
        }),
        default_severely_constrained_channel_count: sum(&assessed, |row| {
            row.default_severely_constrained_channel_count
        }),
        single_severely_constrained_channel_count: sum(&assessed, |row| {
            row.single_severely_constrained_channel_count
        }),
        default_full_activation_span_channel_count: sum(&assessed, |row| {
            row.default_full_activation_span_channel_count
        }),
        single_full_activation_span_channel_count: sum(&assessed, |row| {
            row.single_full_activation_span_channel_count
        }),
        minimum_default_inclusive_code_span: default_spans.iter().copied().min(),
        minimum_single_inclusive_code_span: single_spans.iter().copied().min(),
        span_histogram: span_histogram(&default_spans, &single_spans),
        vitality_ranking_op_indices: ranking,
        ops: rows,
        source_commit: SOURCE_COMMIT,
        source_evidence_schema: "deepbom.kernel_extremum_witness.v1",
        source_references: source_references(),
        constant_proof: "The stored-weight dot-product endpoints are exact over independent legal input codes. Both pinned positive-multiplier requantization functions, output-zero-point addition, and code/activation clamp are nondecreasing. Therefore equal minimum and maximum endpoint output codes prove one constant output code for every legal full-valid receptive-field assignment under that build path.",
        span_definition: "inclusive_endpoint_code_span = maximum_endpoint_output_code - minimum_endpoint_output_code + 1. It is the exact monotone output-code interval hull cardinality and an upper bound on distinct reachable output codes; it is not a proof that every interior code is reachable.",
        reason_code_legend: vec![
            "0=nonconstant",
            "1=constant_accumulator",
            "2=lower_code_clamp",
            "3=upper_code_clamp",
            "4=fixed_point_projection_collapse",
        ],
        sign_code_legend: vec![
            "-1=post_bias_negative_locked",
            "0=post_bias_zero_containing",
            "1=post_bias_positive_locked",
        ],
        method: "Join every exact stored-weight post-bias endpoint to the pinned default and TFLITE_SINGLE_ROUNDING endpoint projections, prove monotone interval hulls, classify constant-output causes, and bind compact per-channel arrays to the source witness ledger by SHA-256.",
        interpretation_boundary: "A span of one is an exact constant-output proof for one full-valid receptive field under the pinned TFLite reference integer path. A span above one is an interval-hull upper bound, not an exact reachable-code count. Edge padding, runtime build flags, delegate lowering, executed microkernels, observed activations, calibration, and task accuracy remain outside this static proof.",
    }
}

fn assess_op(row: &KernelWitnessOpRow) -> ChannelVitalityOpRow {
    if row.assessment_status != "assessed" {
        return not_assessed(row, row.not_assessed_reason.clone());
    }
    match assess_op_inner(row) {
        Ok(result) => result,
        Err(reason) => not_assessed(row, reason),
    }
}

fn assess_op_inner(row: &KernelWitnessOpRow) -> Result<ChannelVitalityOpRow, String> {
    let activation_range = row
        .activation_code_range
        .ok_or_else(|| "Activation code range is unavailable.".to_string())?;
    let output_range = row
        .output_code_range
        .ok_or_else(|| "Output code range is unavailable.".to_string())?;
    if row.channel_outcomes.len() != row.assessed_channel_count {
        return Err("Kernel witness outcome cardinality is incomplete.".to_string());
    }
    let activation_span = span_from_codes(activation_range[0], activation_range[1])?;
    let mut ledger = Sha256::new();
    ledger.update(LEDGER_PREFIX);
    ledger.update(row.witness_ledger_sha256.as_bytes());
    let mut witnesses = Vec::with_capacity(row.channel_outcomes.len());
    let mut default_minimum_output_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut default_maximum_output_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut single_minimum_output_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut single_maximum_output_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut sign_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut default_reason_codes = Vec::with_capacity(row.channel_outcomes.len());
    let mut single_reason_codes = Vec::with_capacity(row.channel_outcomes.len());

    for (expected_channel_index, outcome) in row.channel_outcomes.iter().enumerate() {
        if outcome.channel_index != expected_channel_index {
            return Err(format!(
                "Kernel witness channel order is non-canonical: expected channel {expected_channel_index}, found {}.",
                outcome.channel_index
            ));
        }
        let default_span = optional_span(
            outcome.default_minimum_output_code,
            outcome.default_maximum_output_code,
        )?;
        let single_span = optional_span(
            outcome.single_minimum_output_code,
            outcome.single_maximum_output_code,
        )?;
        let sign_code = sign_code(outcome);
        let default_reason = constant_reason(
            outcome,
            default_span,
            outcome.default_minimum_preclamp_code,
            outcome.default_maximum_preclamp_code,
            activation_range,
        );
        let single_reason = constant_reason(
            outcome,
            single_span,
            outcome.single_minimum_preclamp_code,
            outcome.single_maximum_preclamp_code,
            activation_range,
        );
        for value in [
            Some(row.op_index as i64),
            Some(outcome.channel_index as i64),
            outcome.default_minimum_preclamp_code,
            outcome.default_maximum_preclamp_code,
            outcome.default_minimum_output_code,
            outcome.default_maximum_output_code,
            outcome.single_minimum_preclamp_code,
            outcome.single_maximum_preclamp_code,
            outcome.single_minimum_output_code,
            outcome.single_maximum_output_code,
            Some(sign_code as i64),
            Some(default_reason as i64),
            Some(single_reason as i64),
        ] {
            ledger.update(value.unwrap_or(MISSING_I64).to_le_bytes());
        }
        default_minimum_output_codes.push(outcome.default_minimum_output_code);
        default_maximum_output_codes.push(outcome.default_maximum_output_code);
        single_minimum_output_codes.push(outcome.single_minimum_output_code);
        single_maximum_output_codes.push(outcome.single_maximum_output_code);
        sign_codes.push(sign_code);
        default_reason_codes.push(default_reason);
        single_reason_codes.push(single_reason);
        witnesses.push(channel_witness(
            outcome,
            default_span,
            single_span,
            sign_code,
            default_reason,
            single_reason,
        ));
    }

    witnesses.sort_by(compare_channels);
    let fixed_point_assessed_channel_count = row
        .channel_outcomes
        .iter()
        .filter(|outcome| {
            outcome.default_minimum_output_code.is_some()
                && outcome.default_maximum_output_code.is_some()
                && outcome.single_minimum_output_code.is_some()
                && outcome.single_maximum_output_code.is_some()
        })
        .count();
    let default_spans =
        spans(&default_minimum_output_codes, &default_maximum_output_codes).collect::<Vec<_>>();
    let single_spans =
        spans(&single_minimum_output_codes, &single_maximum_output_codes).collect::<Vec<_>>();
    let default_constant_indices = default_reason_codes
        .iter()
        .enumerate()
        .filter_map(|(index, reason)| (*reason != REASON_NONCONSTANT).then_some(index))
        .collect::<Vec<_>>();
    let single_constant_indices = single_reason_codes
        .iter()
        .enumerate()
        .filter_map(|(index, reason)| (*reason != REASON_NONCONSTANT).then_some(index))
        .collect::<Vec<_>>();
    let mode_dependent_indices = default_reason_codes
        .iter()
        .zip(&single_reason_codes)
        .enumerate()
        .filter_map(|(index, (default, single))| {
            ((*default == REASON_NONCONSTANT) != (*single == REASON_NONCONSTANT)).then_some(index)
        })
        .collect::<Vec<_>>();
    let dual_constant = default_reason_codes
        .iter()
        .zip(&single_reason_codes)
        .filter(|(default, single)| {
            **default != REASON_NONCONSTANT && **single != REASON_NONCONSTANT
        })
        .count();
    let nonconstant_accumulator_dual = row
        .channel_outcomes
        .iter()
        .zip(default_reason_codes.iter().zip(&single_reason_codes))
        .filter(|(outcome, (default, single))| {
            outcome.post_bias_minimum != outcome.post_bias_maximum
                && **default != REASON_NONCONSTANT
                && **single != REASON_NONCONSTANT
        })
        .count();

    Ok(ChannelVitalityOpRow {
        op_index: row.op_index,
        op_name: row.op_name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        output_code_range: Some(output_range),
        activation_code_range: Some(activation_range),
        assessed_channel_count: row.channel_outcomes.len(),
        fixed_point_assessed_channel_count,
        constant_accumulator_channel_count: row
            .channel_outcomes
            .iter()
            .filter(|outcome| outcome.post_bias_minimum == outcome.post_bias_maximum)
            .count(),
        post_bias_negative_locked_channel_count: sign_codes.iter().filter(|code| **code < 0).count(),
        post_bias_positive_locked_channel_count: sign_codes.iter().filter(|code| **code > 0).count(),
        post_bias_zero_containing_channel_count: sign_codes.iter().filter(|code| **code == 0).count(),
        default_constant_output_channel_count: default_constant_indices.len(),
        single_constant_output_channel_count: single_constant_indices.len(),
        dual_mode_constant_output_channel_count: dual_constant,
        nonconstant_accumulator_dual_mode_constant_channel_count: nonconstant_accumulator_dual,
        mode_dependent_constant_output_channel_count: mode_dependent_indices.len(),
        default_severely_constrained_channel_count: default_spans.iter().filter(|span| **span <= 15).count(),
        single_severely_constrained_channel_count: single_spans.iter().filter(|span| **span <= 15).count(),
        default_full_activation_span_channel_count: default_spans.iter().filter(|span| **span == activation_span).count(),
        single_full_activation_span_channel_count: single_spans.iter().filter(|span| **span == activation_span).count(),
        minimum_default_inclusive_code_span: default_spans.iter().copied().min(),
        minimum_single_inclusive_code_span: single_spans.iter().copied().min(),
        default_minimum_output_codes,
        default_maximum_output_codes,
        single_minimum_output_codes,
        single_maximum_output_codes,
        post_bias_sign_codes: sign_codes,
        default_constant_reason_codes: default_reason_codes,
        single_constant_reason_codes: single_reason_codes,
        default_constant_channel_indices: default_constant_indices,
        single_constant_channel_indices: single_constant_indices,
        mode_dependent_constant_channel_indices: mode_dependent_indices,
        top_channels: witnesses.into_iter().take(TOP_CHANNEL_LIMIT).collect(),
        source_witness_ledger_sha256: row.witness_ledger_sha256.clone(),
        vitality_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: "SHA-256 over deepbom.channel_vitality.v1 NUL, the 64-byte lowercase source witness-ledger digest, then 13 signed i64 little-endian fields per channel: op, channel, default min/max preclamp, default min/max output, single min/max preclamp, single min/max output, sign code, default reason code, single reason code; missing values use INT64_MIN.",
    })
}

fn not_assessed(row: &KernelWitnessOpRow, reason: String) -> ChannelVitalityOpRow {
    ChannelVitalityOpRow {
        op_index: row.op_index,
        op_name: row.op_name.clone(),
        assessment_status: "not_assessed",
        not_assessed_reason: reason,
        output_code_range: row.output_code_range,
        activation_code_range: row.activation_code_range,
        assessed_channel_count: 0,
        fixed_point_assessed_channel_count: 0,
        constant_accumulator_channel_count: 0,
        post_bias_negative_locked_channel_count: 0,
        post_bias_positive_locked_channel_count: 0,
        post_bias_zero_containing_channel_count: 0,
        default_constant_output_channel_count: 0,
        single_constant_output_channel_count: 0,
        dual_mode_constant_output_channel_count: 0,
        nonconstant_accumulator_dual_mode_constant_channel_count: 0,
        mode_dependent_constant_output_channel_count: 0,
        default_severely_constrained_channel_count: 0,
        single_severely_constrained_channel_count: 0,
        default_full_activation_span_channel_count: 0,
        single_full_activation_span_channel_count: 0,
        minimum_default_inclusive_code_span: None,
        minimum_single_inclusive_code_span: None,
        default_minimum_output_codes: Vec::new(),
        default_maximum_output_codes: Vec::new(),
        single_minimum_output_codes: Vec::new(),
        single_maximum_output_codes: Vec::new(),
        post_bias_sign_codes: Vec::new(),
        default_constant_reason_codes: Vec::new(),
        single_constant_reason_codes: Vec::new(),
        default_constant_channel_indices: Vec::new(),
        single_constant_channel_indices: Vec::new(),
        mode_dependent_constant_channel_indices: Vec::new(),
        top_channels: Vec::new(),
        source_witness_ledger_sha256: row.witness_ledger_sha256.clone(),
        vitality_ledger_sha256: String::new(),
        ledger_hash_method: "",
    }
}

fn channel_witness(
    outcome: &KernelChannelOutcome,
    default_span: Option<usize>,
    single_span: Option<usize>,
    sign_code: i8,
    default_reason: u8,
    single_reason: u8,
) -> ChannelVitalityWitness {
    ChannelVitalityWitness {
        channel_index: outcome.channel_index,
        post_bias_minimum_decimal: outcome.post_bias_minimum.to_string(),
        post_bias_maximum_decimal: outcome.post_bias_maximum.to_string(),
        accumulator_span_decimal: (outcome.post_bias_maximum - outcome.post_bias_minimum)
            .to_string(),
        post_bias_sign_class: sign_label(sign_code),
        default_minimum_preclamp_code: outcome.default_minimum_preclamp_code,
        default_maximum_preclamp_code: outcome.default_maximum_preclamp_code,
        default_minimum_output_code: outcome.default_minimum_output_code,
        default_maximum_output_code: outcome.default_maximum_output_code,
        default_inclusive_code_span: default_span,
        default_constant_reason: reason_label(default_reason),
        single_minimum_preclamp_code: outcome.single_minimum_preclamp_code,
        single_maximum_preclamp_code: outcome.single_maximum_preclamp_code,
        single_minimum_output_code: outcome.single_minimum_output_code,
        single_maximum_output_code: outcome.single_maximum_output_code,
        single_inclusive_code_span: single_span,
        single_constant_reason: reason_label(single_reason),
        dual_mode_constant: default_reason != REASON_NONCONSTANT
            && single_reason != REASON_NONCONSTANT,
        mode_dependent_constant: (default_reason == REASON_NONCONSTANT)
            != (single_reason == REASON_NONCONSTANT),
    }
}

fn constant_reason(
    outcome: &KernelChannelOutcome,
    span: Option<usize>,
    minimum_preclamp: Option<i64>,
    maximum_preclamp: Option<i64>,
    activation_range: [i64; 2],
) -> u8 {
    if span != Some(1) {
        return REASON_NONCONSTANT;
    }
    if outcome.post_bias_minimum == outcome.post_bias_maximum {
        REASON_CONSTANT_ACCUMULATOR
    } else if maximum_preclamp.is_some_and(|value| value <= activation_range[0]) {
        REASON_LOWER_CODE_CLAMP
    } else if minimum_preclamp.is_some_and(|value| value >= activation_range[1]) {
        REASON_UPPER_CODE_CLAMP
    } else {
        REASON_PROJECTION_COLLAPSE
    }
}

fn sign_code(outcome: &KernelChannelOutcome) -> i8 {
    if outcome.post_bias_maximum < 0 {
        -1
    } else if outcome.post_bias_minimum > 0 {
        1
    } else {
        0
    }
}

fn optional_span(minimum: Option<i64>, maximum: Option<i64>) -> Result<Option<usize>, String> {
    match (minimum, maximum) {
        (Some(minimum), Some(maximum)) => span_from_codes(minimum, maximum).map(Some),
        (None, None) => Ok(None),
        _ => Err("Endpoint output-code coverage is asymmetric.".to_string()),
    }
}

fn span_from_codes(minimum: i64, maximum: i64) -> Result<usize, String> {
    if maximum < minimum {
        return Err("Endpoint output codes violate monotonic ordering.".to_string());
    }
    usize::try_from(maximum as i128 - minimum as i128 + 1)
        .map_err(|_| "Endpoint code span exceeds usize.".to_string())
}

fn spans<'a>(
    minimum: &'a [Option<i64>],
    maximum: &'a [Option<i64>],
) -> impl Iterator<Item = usize> + 'a {
    minimum
        .iter()
        .zip(maximum)
        .filter_map(|(minimum, maximum)| match (minimum, maximum) {
            (Some(minimum), Some(maximum)) if maximum >= minimum => {
                usize::try_from(*maximum as i128 - *minimum as i128 + 1).ok()
            }
            _ => None,
        })
}

fn span_histogram(default_spans: &[usize], single_spans: &[usize]) -> Vec<SpanHistogramBin> {
    const BINS: [(&str, usize, usize); 7] = [
        ("1", 1, 1),
        ("2-3", 2, 3),
        ("4-15", 4, 15),
        ("16-63", 16, 63),
        ("64-127", 64, 127),
        ("128-255", 128, 255),
        ("256", 256, 256),
    ];
    BINS.into_iter()
        .map(|(label, minimum, maximum)| SpanHistogramBin {
            label,
            minimum_inclusive_span: minimum,
            maximum_inclusive_span: maximum,
            default_channel_count: default_spans
                .iter()
                .filter(|span| **span >= minimum && **span <= maximum)
                .count(),
            single_rounding_channel_count: single_spans
                .iter()
                .filter(|span| **span >= minimum && **span <= maximum)
                .count(),
        })
        .collect()
}

fn compare_ops(left: &ChannelVitalityOpRow, right: &ChannelVitalityOpRow) -> Ordering {
    right
        .nonconstant_accumulator_dual_mode_constant_channel_count
        .cmp(&left.nonconstant_accumulator_dual_mode_constant_channel_count)
        .then_with(|| {
            right
                .dual_mode_constant_output_channel_count
                .cmp(&left.dual_mode_constant_output_channel_count)
        })
        .then_with(|| {
            right
                .mode_dependent_constant_output_channel_count
                .cmp(&left.mode_dependent_constant_output_channel_count)
        })
        .then_with(|| {
            right
                .default_severely_constrained_channel_count
                .cmp(&left.default_severely_constrained_channel_count)
        })
        .then_with(|| {
            left.minimum_default_inclusive_code_span
                .cmp(&right.minimum_default_inclusive_code_span)
        })
        .then_with(|| left.op_index.cmp(&right.op_index))
}

fn compare_channels(left: &ChannelVitalityWitness, right: &ChannelVitalityWitness) -> Ordering {
    right
        .dual_mode_constant
        .cmp(&left.dual_mode_constant)
        .then_with(|| {
            right
                .mode_dependent_constant
                .cmp(&left.mode_dependent_constant)
        })
        .then_with(|| {
            let left_variable = left.accumulator_span_decimal != "0";
            let right_variable = right.accumulator_span_decimal != "0";
            right_variable.cmp(&left_variable)
        })
        .then_with(|| {
            left.default_inclusive_code_span
                .cmp(&right.default_inclusive_code_span)
        })
        .then_with(|| left.channel_index.cmp(&right.channel_index))
}

fn reason_label(code: u8) -> &'static str {
    match code {
        REASON_CONSTANT_ACCUMULATOR => "constant_accumulator",
        REASON_LOWER_CODE_CLAMP => "lower_code_clamp",
        REASON_UPPER_CODE_CLAMP => "upper_code_clamp",
        REASON_PROJECTION_COLLAPSE => "fixed_point_projection_collapse",
        _ => "nonconstant",
    }
}

fn sign_label(code: i8) -> &'static str {
    match code {
        -1 => "post_bias_negative_locked",
        1 => "post_bias_positive_locked",
        _ => "post_bias_zero_containing",
    }
}

fn sum<T>(rows: &[&T], value: impl Fn(&T) -> usize) -> usize {
    rows.iter().map(|row| value(row)).sum()
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn source_references() -> Vec<SourceReference> {
    [
        (
            "pinned fixed-point multiply and rounding execution",
            "tensorflow/lite/kernels/internal/common.cc",
            "ba5308bf76383d600d033c948fe0659710939e6f1f15a800b5413e5fc822ddfa",
        ),
        (
            "reference integer CONV endpoint path",
            "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h",
            "370f80020b9aa44e61bb3a0a0d081c035432f99aaf1cb66c6c3d072700a4bec8",
        ),
        (
            "reference integer DEPTHWISE_CONV endpoint path",
            "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h",
            "98d0e11ff47bb32e3485f3f891be77cf733f6a47d52b6fc236d437fc581a069c",
        ),
        (
            "reference integer FULLY_CONNECTED endpoint path",
            "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h",
            "88beede059cee0eba102c5a108c0750b33fe823af25393053e618dd29781abd6",
        ),
    ]
    .into_iter()
    .map(|(role, file, sha256)| SourceReference {
        role,
        file,
        url: format!("https://github.com/tensorflow/tensorflow/blob/{SOURCE_COMMIT}/{file}"),
        sha256,
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(minimum: i128, maximum: i128) -> KernelChannelOutcome {
        KernelChannelOutcome {
            channel_index: 0,
            post_bias_minimum: minimum,
            post_bias_maximum: maximum,
            default_minimum_preclamp_code: Some(-5),
            default_maximum_preclamp_code: Some(-1),
            default_minimum_output_code: Some(0),
            default_maximum_output_code: Some(0),
            single_minimum_preclamp_code: Some(-5),
            single_maximum_preclamp_code: Some(-1),
            single_minimum_output_code: Some(0),
            single_maximum_output_code: Some(0),
        }
    }

    #[test]
    fn variable_accumulator_below_lower_code_bound_is_provably_constant() {
        let outcome = outcome(-100, -1);
        assert_eq!(
            constant_reason(&outcome, Some(1), Some(-5), Some(-1), [0, 255]),
            REASON_LOWER_CODE_CLAMP
        );
        assert_eq!(sign_code(&outcome), -1);
    }

    #[test]
    fn constant_accumulator_reason_precedes_projection_or_clamp() {
        let outcome = outcome(-5, -5);
        assert_eq!(
            constant_reason(&outcome, Some(1), Some(-5), Some(-5), [0, 255]),
            REASON_CONSTANT_ACCUMULATOR
        );
    }

    #[test]
    fn inclusive_span_rejects_nonmonotone_endpoints() {
        assert_eq!(span_from_codes(4, 4).unwrap(), 1);
        assert!(span_from_codes(5, 4).is_err());
    }
}
