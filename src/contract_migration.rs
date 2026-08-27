use super::accumulator_atlas::{decode_bias, AccumulatorAtlasAnalysis};
use super::quantization_lattice::{
    ContainmentCandidateEvaluation, QuantizationLatticeAnalysis, ResidualAddLatticeRow,
};
use super::quantization_math::{quantize_multiplier, round_ties_away_from_zero};
use super::{OpInfo, TensorInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

const SCHEMA: &str = "deepbom.contract_migration.v1";
const METHOD_VERSION: &str = "2026-07-17.1";
const TFLITE_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const TOP_CHANNEL_LIMIT: usize = 8;

#[derive(Serialize)]
struct SourceReference {
    role: &'static str,
    file: &'static str,
    url: String,
    sha256: &'static str,
}

#[derive(Clone, Serialize)]
struct AffectedOp {
    op_index: usize,
    op_name: String,
    minimum_edge_depth: usize,
    direct_consumer: bool,
}

#[derive(Clone, Serialize)]
struct ConsumerIdentity {
    op_index: usize,
    op_name: String,
    input_slots: Vec<usize>,
    migration_class: &'static str,
}

#[derive(Clone, Serialize)]
struct KernelChannelWitness {
    channel_index: usize,
    weight_scale: f64,
    current_real_multiplier: f64,
    candidate_real_multiplier: f64,
    current_quantized_multiplier: i32,
    candidate_quantized_multiplier: i32,
    current_shift: i32,
    candidate_shift: i32,
    current_bias_code: i32,
    candidate_bias_code_decimal: String,
    bias_code_changed: bool,
    bias_int32_overflow: bool,
    preserved_bias_real_value: f64,
    candidate_bias_real_value: Option<f64>,
    absolute_bias_rebase_error: Option<f64>,
    absolute_bias_rebase_error_current_steps: Option<f64>,
    absolute_bias_rebase_error_candidate_steps: Option<f64>,
}

#[derive(Clone, Serialize)]
struct KernelConsumerMigration {
    op_index: usize,
    op_name: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    input_slots: Vec<usize>,
    input_tensor_index: i32,
    input_tensor_name: String,
    weight_tensor_index: Option<i32>,
    weight_tensor_name: String,
    bias_tensor_index: Option<i32>,
    bias_tensor_name: String,
    output_tensor_index: Option<i32>,
    output_tensor_name: String,
    current_input_scale: f64,
    candidate_input_scale: f64,
    input_scale_ratio: f64,
    current_input_zero_point: i64,
    candidate_input_zero_point: i64,
    input_zero_point_delta: i64,
    output_scale: Option<f64>,
    weight_scale_mode: String,
    bias_status: String,
    assessed_channel_count: usize,
    multiplier_encoding_changed_channel_count: usize,
    multiplier_shift_changed_channel_count: usize,
    bias_code_changed_channel_count: usize,
    bias_int32_overflow_channel_count: usize,
    maximum_absolute_bias_rebase_error: Option<f64>,
    maximum_absolute_bias_rebase_error_current_steps: Option<f64>,
    maximum_absolute_bias_rebase_error_candidate_steps: Option<f64>,
    channel_current_quantized_multipliers: Vec<i32>,
    channel_candidate_quantized_multipliers: Vec<i32>,
    channel_current_shifts: Vec<i32>,
    channel_candidate_shifts: Vec<i32>,
    channel_current_bias_codes: Vec<i32>,
    channel_candidate_bias_code_decimals: Vec<String>,
    channel_bias_rebase_error_current_steps: Vec<Option<f64>>,
    top_channels: Vec<KernelChannelWitness>,
    channel_ledger_sha256: String,
    ledger_hash_method: &'static str,
}

impl KernelConsumerMigration {
    fn not_assessed(
        op: &OpInfo,
        source: &TensorInfo,
        input_slots: Vec<usize>,
        current_scale: f64,
        current_zero_point: i64,
        candidate: CandidateContract,
        reason: String,
    ) -> Self {
        Self {
            op_index: op.index,
            op_name: op.name.clone(),
            assessment_status: "not_assessed",
            not_assessed_reason: reason,
            input_slots,
            input_tensor_index: source.index as i32,
            input_tensor_name: source.name.clone(),
            weight_tensor_index: op.inputs.get(1).copied().filter(|index| *index >= 0),
            weight_tensor_name: String::new(),
            bias_tensor_index: op.inputs.get(2).copied().filter(|index| *index >= 0),
            bias_tensor_name: String::new(),
            output_tensor_index: op.outputs.first().copied().filter(|index| *index >= 0),
            output_tensor_name: String::new(),
            current_input_scale: current_scale,
            candidate_input_scale: candidate.scale,
            input_scale_ratio: candidate.scale / current_scale,
            current_input_zero_point: current_zero_point,
            candidate_input_zero_point: candidate.zero_point,
            input_zero_point_delta: candidate.zero_point - current_zero_point,
            output_scale: None,
            weight_scale_mode: String::new(),
            bias_status: String::new(),
            assessed_channel_count: 0,
            multiplier_encoding_changed_channel_count: 0,
            multiplier_shift_changed_channel_count: 0,
            bias_code_changed_channel_count: 0,
            bias_int32_overflow_channel_count: 0,
            maximum_absolute_bias_rebase_error: None,
            maximum_absolute_bias_rebase_error_current_steps: None,
            maximum_absolute_bias_rebase_error_candidate_steps: None,
            channel_current_quantized_multipliers: Vec::new(),
            channel_candidate_quantized_multipliers: Vec::new(),
            channel_current_shifts: Vec::new(),
            channel_candidate_shifts: Vec::new(),
            channel_current_bias_codes: Vec::new(),
            channel_candidate_bias_code_decimals: Vec::new(),
            channel_bias_rebase_error_current_steps: Vec::new(),
            top_channels: Vec::new(),
            channel_ledger_sha256: String::new(),
            ledger_hash_method: ledger_hash_method(),
        }
    }
}

#[derive(Clone, Serialize)]
struct EncodedMultiplier {
    real_multiplier: f64,
    quantized_multiplier: i32,
    shift: i32,
    represented_multiplier: f64,
}

#[derive(Clone, Serialize)]
struct AddParameterSet {
    left_shift: i32,
    twice_max_input_scale: f64,
    input_offsets: [i64; 2],
    output_offset: i64,
    input_multipliers: [EncodedMultiplier; 2],
    output_multiplier: EncodedMultiplier,
}

#[derive(Clone, Serialize)]
struct AddConsumerMigration {
    op_index: usize,
    op_name: String,
    assessment_status: &'static str,
    not_assessed_reason: String,
    changed_input_slots: Vec<usize>,
    input_tensor_indices: Vec<i32>,
    input_tensor_names: Vec<String>,
    output_tensor_index: Option<i32>,
    output_tensor_name: String,
    current_parameters: Option<AddParameterSet>,
    candidate_parameters: Option<AddParameterSet>,
    changed_offset_count: usize,
    changed_multiplier_encoding_count: usize,
    changed_shift_count: usize,
}

#[derive(Clone, Serialize)]
struct UnassessedConsumerMigration {
    op_index: usize,
    op_name: String,
    input_slots: Vec<usize>,
    reason: String,
}

#[derive(Clone, Serialize)]
struct MigrationScenario {
    design: &'static str,
    candidate_output_scale: f64,
    candidate_output_zero_point: i64,
    scale_ratio_to_current: f64,
    signed_zero_point_delta: i64,
    assessed_consumer_count: usize,
    unassessed_consumer_count: usize,
    assessed_kernel_channel_count: usize,
    multiplier_encoding_changed_channel_count: usize,
    multiplier_shift_changed_channel_count: usize,
    bias_code_changed_channel_count: usize,
    bias_int32_overflow_channel_count: usize,
    add_parameter_encoding_changed_count: usize,
    kernel_consumers: Vec<KernelConsumerMigration>,
    add_consumers: Vec<AddConsumerMigration>,
    unassessed_consumers: Vec<UnassessedConsumerMigration>,
}

#[derive(Clone, Copy)]
struct CandidateContract {
    design: &'static str,
    scale: f64,
    zero_point: i64,
}

impl From<&ContainmentCandidateEvaluation> for CandidateContract {
    fn from(value: &ContainmentCandidateEvaluation) -> Self {
        Self {
            design: value.design,
            scale: value.output_scale,
            zero_point: value.output_zero_point,
        }
    }
}

#[derive(Clone, Serialize)]
struct ResidualContractMigration {
    source_add_op_index: usize,
    output_tensor_index: i32,
    output_tensor_name: String,
    current_output_scale: f64,
    current_output_zero_point: i64,
    direct_consumer_count: usize,
    direct_consumer_edge_count: usize,
    direct_consumers: Vec<ConsumerIdentity>,
    reachable_downstream_op_count: usize,
    maximum_downstream_edge_depth: usize,
    affected_ops: Vec<AffectedOp>,
    scenarios: Vec<MigrationScenario>,
}

#[derive(Serialize)]
pub(super) struct ContractMigrationAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    residual_contract_count: usize,
    candidate_scenario_count: usize,
    direct_consumer_count: usize,
    direct_consumer_edge_count: usize,
    kernel_consumer_count: usize,
    add_consumer_count: usize,
    other_consumer_count: usize,
    assessed_consumer_scenario_count: usize,
    unassessed_consumer_scenario_count: usize,
    assessed_kernel_channel_scenario_count: usize,
    multiplier_encoding_changed_channel_scenario_count: usize,
    multiplier_shift_changed_channel_scenario_count: usize,
    bias_code_changed_channel_scenario_count: usize,
    bias_int32_overflow_channel_scenario_count: usize,
    add_parameter_encoding_changed_scenario_count: usize,
    reachable_downstream_op_union_count: usize,
    maximum_downstream_edge_depth: usize,
    migrations: Vec<ResidualContractMigration>,
    source_commit: &'static str,
    source_references: Vec<SourceReference>,
    kernel_multiplier_formula: &'static str,
    bias_rebase_formula: &'static str,
    add_parameter_formula: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn build_contract_migration(
    model_bytes: &[u8],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    lattice: &QuantizationLatticeAnalysis,
    accumulator: &AccumulatorAtlasAnalysis,
) -> ContractMigrationAnalysis {
    let consumers = consumer_map(ops);
    let mut migrations = Vec::new();
    let mut downstream_union = HashSet::new();
    for row in &lattice.residual_adds {
        let Some(migration) =
            build_residual_migration(model_bytes, row, ops, tensors, accumulator, &consumers)
        else {
            continue;
        };
        downstream_union.extend(migration.affected_ops.iter().map(|item| item.op_index));
        migrations.push(migration);
    }
    let scenarios = migrations
        .iter()
        .flat_map(|migration| migration.scenarios.iter())
        .collect::<Vec<_>>();
    let identities = migrations
        .iter()
        .flat_map(|migration| migration.direct_consumers.iter())
        .collect::<Vec<_>>();
    let assessed_consumer_scenario_count = scenarios
        .iter()
        .map(|scenario| scenario.assessed_consumer_count)
        .sum();
    let unassessed_consumer_scenario_count = scenarios
        .iter()
        .map(|scenario| scenario.unassessed_consumer_count)
        .sum();
    let status = if migrations.is_empty() {
        "not_applicable"
    } else if unassessed_consumer_scenario_count > 0 {
        "partial"
    } else {
        "assessed"
    };
    ContractMigrationAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        residual_contract_count: migrations.len(),
        candidate_scenario_count: scenarios.len(),
        direct_consumer_count: identities.len(),
        direct_consumer_edge_count: identities.iter().map(|item| item.input_slots.len()).sum(),
        kernel_consumer_count: identities
            .iter()
            .filter(|item| item.migration_class == "integer_kernel")
            .count(),
        add_consumer_count: identities
            .iter()
            .filter(|item| item.migration_class == "quantized_add")
            .count(),
        other_consumer_count: identities
            .iter()
            .filter(|item| item.migration_class == "not_modeled")
            .count(),
        assessed_consumer_scenario_count,
        unassessed_consumer_scenario_count,
        assessed_kernel_channel_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.assessed_kernel_channel_count)
            .sum(),
        multiplier_encoding_changed_channel_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.multiplier_encoding_changed_channel_count)
            .sum(),
        multiplier_shift_changed_channel_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.multiplier_shift_changed_channel_count)
            .sum(),
        bias_code_changed_channel_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.bias_code_changed_channel_count)
            .sum(),
        bias_int32_overflow_channel_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.bias_int32_overflow_channel_count)
            .sum(),
        add_parameter_encoding_changed_scenario_count: scenarios
            .iter()
            .map(|scenario| scenario.add_parameter_encoding_changed_count)
            .sum(),
        reachable_downstream_op_union_count: downstream_union.len(),
        maximum_downstream_edge_depth: migrations
            .iter()
            .map(|migration| migration.maximum_downstream_edge_depth)
            .max()
            .unwrap_or(0),
        migrations,
        source_commit: TFLITE_SOURCE_COMMIT,
        source_references: source_references(),
        kernel_multiplier_formula: "m[channel]=input_scale*weight_scale[channel]/output_scale; (q,shift)=TensorFlow::QuantizeMultiplier(m)",
        bias_rebase_formula: "bias_scale_old=input_scale_old*weight_scale[channel]; bias_scale_new=input_scale_candidate*weight_scale[channel]; bias_int_new=round_ties_away(bias_int_old*input_scale_old/input_scale_candidate)",
        add_parameter_formula: "left_shift=20; twice_max=2*max(s0,s1); m0=s0/twice_max; m1=s1/twice_max; mout=twice_max/(2^left_shift*sout); each multiplier uses QuantizeMultiplierSmallerThanOneExp",
        method: "For both exhaustive residual containment candidates, trace every direct tensor consumer. Recompute pinned TensorFlow Q0.31 parameters for INT8/UINT8 CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, and ADD consumers. Decode every stored INT32 bias, derive the nearest ties-away integer that preserves its real value under the candidate input scale, test INT32 representability, and retain a channel-complete digest-bound ledger. Separately compute the complete reachable graph radius without claiming that every reachable op needs metadata regeneration.",
        interpretation_boundary: "This is a counterfactual re-export impact analysis, not a FlatBuffer patch, runtime trace, calibration recommendation, or accuracy claim. Direct consumers are the exact parameter-regeneration boundary modeled here; reachable downstream ops are a structural behavior-impact radius only. Bias rebasing preserves the artifact's current real bias as closely as the candidate lattice permits and does not prove that the candidate activation contract is statistically suitable. Q0.31 encodings match the pinned default TensorFlow source path; build flags, delegated lowering, and executed microkernels still require runtime evidence.",
    }
}

fn build_residual_migration(
    model_bytes: &[u8],
    row: &ResidualAddLatticeRow,
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    accumulator: &AccumulatorAtlasAnalysis,
    consumers: &HashMap<i32, Vec<usize>>,
) -> Option<ResidualContractMigration> {
    let output_index = row.output_tensor_index?;
    let source = tensor_at(tensors, output_index)?;
    let current_scale = row.output_scale?;
    let current_zero_point = row.output_zero_point?;
    let consumer_indices = consumers.get(&output_index).cloned().unwrap_or_default();
    let direct_consumers = consumer_indices
        .iter()
        .filter_map(|index| ops.get(*index))
        .map(|op| ConsumerIdentity {
            op_index: op.index,
            op_name: op.name.clone(),
            input_slots: matching_input_slots(op, output_index),
            migration_class: migration_class(&op.name),
        })
        .collect::<Vec<_>>();
    let affected_ops = downstream_ops(output_index, ops, consumers);
    let maximum_downstream_edge_depth = affected_ops
        .iter()
        .map(|item| item.minimum_edge_depth)
        .max()
        .unwrap_or(0);
    let mut candidates = Vec::new();
    if let Some(candidate) = &row.fixed_zero_point_containment {
        candidates.push(CandidateContract::from(candidate));
    }
    if let Some(candidate) = &row.globally_finest_containment {
        candidates.push(CandidateContract::from(candidate));
    }
    let scenarios = candidates
        .into_iter()
        .map(|candidate| {
            build_scenario(
                model_bytes,
                source,
                current_scale,
                current_zero_point,
                candidate,
                &consumer_indices,
                ops,
                tensors,
                accumulator,
            )
        })
        .collect::<Vec<_>>();
    Some(ResidualContractMigration {
        source_add_op_index: row.op_index,
        output_tensor_index: output_index,
        output_tensor_name: row.output_tensor_name.clone(),
        current_output_scale: current_scale,
        current_output_zero_point: current_zero_point,
        direct_consumer_count: direct_consumers.len(),
        direct_consumer_edge_count: direct_consumers
            .iter()
            .map(|consumer| consumer.input_slots.len())
            .sum(),
        direct_consumers,
        reachable_downstream_op_count: affected_ops.len(),
        maximum_downstream_edge_depth,
        affected_ops,
        scenarios,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_scenario(
    model_bytes: &[u8],
    source: &TensorInfo,
    current_scale: f64,
    current_zero_point: i64,
    candidate: CandidateContract,
    consumer_indices: &[usize],
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    accumulator: &AccumulatorAtlasAnalysis,
) -> MigrationScenario {
    let mut kernel_consumers = Vec::new();
    let mut add_consumers = Vec::new();
    let mut unassessed_consumers = Vec::new();
    for consumer_index in consumer_indices {
        let Some(op) = ops.get(*consumer_index) else {
            continue;
        };
        let input_slots = matching_input_slots(op, source.index as i32);
        if is_kernel(&op.name) {
            kernel_consumers.push(assess_kernel_consumer(
                model_bytes,
                op,
                source,
                input_slots,
                current_scale,
                current_zero_point,
                candidate,
                tensors,
                accumulator,
            ));
        } else if op.name == "ADD" {
            add_consumers.push(assess_add_consumer(
                op,
                source,
                input_slots,
                current_scale,
                current_zero_point,
                candidate,
                tensors,
            ));
        } else {
            unassessed_consumers.push(UnassessedConsumerMigration {
                op_index: op.index,
                op_name: op.name.clone(),
                input_slots,
                reason:
                    "Direct consumer has no source-backed migration rule in contract_migration.v1."
                        .to_string(),
            });
        }
    }
    let assessed_kernel_channel_count = kernel_consumers
        .iter()
        .map(|consumer| consumer.assessed_channel_count)
        .sum();
    let multiplier_encoding_changed_channel_count = kernel_consumers
        .iter()
        .map(|consumer| consumer.multiplier_encoding_changed_channel_count)
        .sum();
    let multiplier_shift_changed_channel_count = kernel_consumers
        .iter()
        .map(|consumer| consumer.multiplier_shift_changed_channel_count)
        .sum();
    let bias_code_changed_channel_count = kernel_consumers
        .iter()
        .map(|consumer| consumer.bias_code_changed_channel_count)
        .sum();
    let bias_int32_overflow_channel_count = kernel_consumers
        .iter()
        .map(|consumer| consumer.bias_int32_overflow_channel_count)
        .sum();
    let add_parameter_encoding_changed_count = add_consumers
        .iter()
        .map(|consumer| consumer.changed_multiplier_encoding_count)
        .sum();
    let assessed_consumer_count = kernel_consumers
        .iter()
        .filter(|consumer| consumer.assessment_status == "assessed")
        .count()
        + add_consumers
            .iter()
            .filter(|consumer| consumer.assessment_status == "assessed")
            .count();
    let unassessed_consumer_count = unassessed_consumers.len()
        + kernel_consumers
            .iter()
            .filter(|consumer| consumer.assessment_status != "assessed")
            .count()
        + add_consumers
            .iter()
            .filter(|consumer| consumer.assessment_status != "assessed")
            .count();
    MigrationScenario {
        design: candidate.design,
        candidate_output_scale: candidate.scale,
        candidate_output_zero_point: candidate.zero_point,
        scale_ratio_to_current: candidate.scale / current_scale,
        signed_zero_point_delta: candidate.zero_point - current_zero_point,
        assessed_consumer_count,
        unassessed_consumer_count,
        assessed_kernel_channel_count,
        multiplier_encoding_changed_channel_count,
        multiplier_shift_changed_channel_count,
        bias_code_changed_channel_count,
        bias_int32_overflow_channel_count,
        add_parameter_encoding_changed_count,
        kernel_consumers,
        add_consumers,
        unassessed_consumers,
    }
}

#[allow(clippy::too_many_arguments)]
fn assess_kernel_consumer(
    model_bytes: &[u8],
    op: &OpInfo,
    source: &TensorInfo,
    input_slots: Vec<usize>,
    current_scale: f64,
    current_zero_point: i64,
    candidate: CandidateContract,
    tensors: &[TensorInfo],
    accumulator: &AccumulatorAtlasAnalysis,
) -> KernelConsumerMigration {
    let fail = |reason: String| {
        KernelConsumerMigration::not_assessed(
            op,
            source,
            input_slots.clone(),
            current_scale,
            current_zero_point,
            candidate,
            reason,
        )
    };
    if input_slots != [0] {
        return fail(
            "Kernel migration requires the changed activation at input slot 0.".to_string(),
        );
    }
    let Some(weight) = op
        .inputs
        .get(1)
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return fail("Weight tensor is unavailable.".to_string());
    };
    let Some(output) = op
        .outputs
        .first()
        .and_then(|index| tensor_at(tensors, *index))
    else {
        return fail("Output tensor is unavailable.".to_string());
    };
    let Some(accumulator_row) = accumulator
        .ops
        .iter()
        .find(|row| row.op_index == op.index && row.assessment_status == "assessed")
    else {
        return fail("A channel-complete accumulator row is unavailable.".to_string());
    };
    let Some(channels) = accumulator_row.output_channel_count else {
        return fail("Output-channel cardinality is unavailable.".to_string());
    };
    if output.scale_sample.len() != 1 || !valid_scale(output.scale_sample[0] as f64) {
        return fail("Kernel output requires one finite positive scale.".to_string());
    }
    if !(weight.scale_sample.len() == 1 || weight.scale_sample.len() == channels)
        || weight
            .scale_sample
            .iter()
            .any(|scale| !valid_scale(*scale as f64))
    {
        return fail(format!("Weight scale cardinality must be 1 or {channels}."));
    }
    let (bias, bias_tensor, bias_status) = match decode_bias(model_bytes, op, tensors, channels) {
        Ok(value) => value,
        Err(reason) => return fail(reason),
    };
    let output_scale = output.scale_sample[0] as f64;
    let mut witnesses = Vec::with_capacity(channels);
    let mut current_multipliers = Vec::with_capacity(channels);
    let mut candidate_multipliers = Vec::with_capacity(channels);
    let mut current_shifts = Vec::with_capacity(channels);
    let mut candidate_shifts = Vec::with_capacity(channels);
    let mut candidate_bias_codes = Vec::with_capacity(channels);
    let mut bias_error_steps = Vec::with_capacity(channels);
    let mut ledger = Sha256::new();
    for (channel, &current_bias_code) in bias.iter().enumerate().take(channels) {
        let weight_scale = weight.scale_sample[if weight.scale_sample.len() == 1 {
            0
        } else {
            channel
        }] as f64;
        let current_real_multiplier = current_scale * weight_scale / output_scale;
        let candidate_real_multiplier = candidate.scale * weight_scale / output_scale;
        let current_encoding = quantize_multiplier(current_real_multiplier, false);
        let candidate_encoding = quantize_multiplier(candidate_real_multiplier, false);
        let candidate_bias_code =
            round_ties_away_from_zero(current_bias_code as f64 * current_scale / candidate.scale);
        let overflow =
            candidate_bias_code < i32::MIN as i64 || candidate_bias_code > i32::MAX as i64;
        let current_bias_scale = current_scale * weight_scale;
        let candidate_bias_scale = candidate.scale * weight_scale;
        let preserved_bias_real_value = current_bias_code as f64 * current_bias_scale;
        let candidate_bias_real_value =
            (!overflow).then_some(candidate_bias_code as f64 * candidate_bias_scale);
        let absolute_bias_rebase_error =
            candidate_bias_real_value.map(|value| (value - preserved_bias_real_value).abs());
        let current_steps = absolute_bias_rebase_error.map(|error| error / current_bias_scale);
        let candidate_steps = absolute_bias_rebase_error.map(|error| error / candidate_bias_scale);
        let witness = KernelChannelWitness {
            channel_index: channel,
            weight_scale,
            current_real_multiplier,
            candidate_real_multiplier,
            current_quantized_multiplier: current_encoding.multiplier,
            candidate_quantized_multiplier: candidate_encoding.multiplier,
            current_shift: current_encoding.shift,
            candidate_shift: candidate_encoding.shift,
            current_bias_code,
            candidate_bias_code_decimal: candidate_bias_code.to_string(),
            bias_code_changed: candidate_bias_code != current_bias_code as i64,
            bias_int32_overflow: overflow,
            preserved_bias_real_value,
            candidate_bias_real_value,
            absolute_bias_rebase_error,
            absolute_bias_rebase_error_current_steps: current_steps,
            absolute_bias_rebase_error_candidate_steps: candidate_steps,
        };
        ledger.update(kernel_ledger_row(op.index, &witness).as_bytes());
        current_multipliers.push(current_encoding.multiplier);
        candidate_multipliers.push(candidate_encoding.multiplier);
        current_shifts.push(current_encoding.shift);
        candidate_shifts.push(candidate_encoding.shift);
        candidate_bias_codes.push(candidate_bias_code.to_string());
        bias_error_steps.push(current_steps);
        witnesses.push(witness);
    }
    let encoding_changed = witnesses
        .iter()
        .filter(|channel| {
            channel.current_quantized_multiplier != channel.candidate_quantized_multiplier
                || channel.current_shift != channel.candidate_shift
        })
        .count();
    let shift_changed = witnesses
        .iter()
        .filter(|channel| channel.current_shift != channel.candidate_shift)
        .count();
    let bias_changed = witnesses
        .iter()
        .filter(|channel| channel.bias_code_changed)
        .count();
    let bias_overflow = witnesses
        .iter()
        .filter(|channel| channel.bias_int32_overflow)
        .count();
    let maximum_absolute_bias_rebase_error = max_optional(
        witnesses
            .iter()
            .filter_map(|channel| channel.absolute_bias_rebase_error),
    );
    let maximum_absolute_bias_rebase_error_current_steps = max_optional(
        witnesses
            .iter()
            .filter_map(|channel| channel.absolute_bias_rebase_error_current_steps),
    );
    let maximum_absolute_bias_rebase_error_candidate_steps = max_optional(
        witnesses
            .iter()
            .filter_map(|channel| channel.absolute_bias_rebase_error_candidate_steps),
    );
    witnesses.sort_by(|left, right| {
        right
            .absolute_bias_rebase_error_current_steps
            .unwrap_or(f64::INFINITY)
            .partial_cmp(
                &left
                    .absolute_bias_rebase_error_current_steps
                    .unwrap_or(f64::INFINITY),
            )
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.channel_index.cmp(&right.channel_index))
    });
    KernelConsumerMigration {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "assessed",
        not_assessed_reason: String::new(),
        input_slots,
        input_tensor_index: source.index as i32,
        input_tensor_name: source.name.clone(),
        weight_tensor_index: Some(weight.index as i32),
        weight_tensor_name: weight.name.clone(),
        bias_tensor_index: bias_tensor.map(|tensor| tensor.index as i32),
        bias_tensor_name: bias_tensor
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default(),
        output_tensor_index: Some(output.index as i32),
        output_tensor_name: output.name.clone(),
        current_input_scale: current_scale,
        candidate_input_scale: candidate.scale,
        input_scale_ratio: candidate.scale / current_scale,
        current_input_zero_point: current_zero_point,
        candidate_input_zero_point: candidate.zero_point,
        input_zero_point_delta: candidate.zero_point - current_zero_point,
        output_scale: Some(output_scale),
        weight_scale_mode: if weight.scale_sample.len() == 1 {
            "per_tensor"
        } else {
            "per_output_channel"
        }
        .to_string(),
        bias_status,
        assessed_channel_count: channels,
        multiplier_encoding_changed_channel_count: encoding_changed,
        multiplier_shift_changed_channel_count: shift_changed,
        bias_code_changed_channel_count: bias_changed,
        bias_int32_overflow_channel_count: bias_overflow,
        maximum_absolute_bias_rebase_error,
        maximum_absolute_bias_rebase_error_current_steps,
        maximum_absolute_bias_rebase_error_candidate_steps,
        channel_current_quantized_multipliers: current_multipliers,
        channel_candidate_quantized_multipliers: candidate_multipliers,
        channel_current_shifts: current_shifts,
        channel_candidate_shifts: candidate_shifts,
        channel_current_bias_codes: bias,
        channel_candidate_bias_code_decimals: candidate_bias_codes,
        channel_bias_rebase_error_current_steps: bias_error_steps,
        top_channels: witnesses.iter().take(TOP_CHANNEL_LIMIT).cloned().collect(),
        channel_ledger_sha256: hex_digest(ledger.finalize().as_slice()),
        ledger_hash_method: ledger_hash_method(),
    }
}

fn assess_add_consumer(
    op: &OpInfo,
    source: &TensorInfo,
    input_slots: Vec<usize>,
    current_scale: f64,
    current_zero_point: i64,
    candidate: CandidateContract,
    tensors: &[TensorInfo],
) -> AddConsumerMigration {
    let base = || AddConsumerMigration {
        op_index: op.index,
        op_name: op.name.clone(),
        assessment_status: "not_assessed",
        not_assessed_reason: String::new(),
        changed_input_slots: input_slots.clone(),
        input_tensor_indices: op.inputs.iter().take(2).copied().collect(),
        input_tensor_names: op
            .inputs
            .iter()
            .take(2)
            .filter_map(|index| tensor_at(tensors, *index))
            .map(|tensor| tensor.name.clone())
            .collect(),
        output_tensor_index: op.outputs.first().copied().filter(|index| *index >= 0),
        output_tensor_name: op
            .outputs
            .first()
            .and_then(|index| tensor_at(tensors, *index))
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default(),
        current_parameters: None,
        candidate_parameters: None,
        changed_offset_count: 0,
        changed_multiplier_encoding_count: 0,
        changed_shift_count: 0,
    };
    let mut result = base();
    if op.inputs.len() < 2 || op.outputs.is_empty() {
        result.not_assessed_reason = "ADD does not expose two inputs and one output.".to_string();
        return result;
    }
    let Some(input_0) = tensor_at(tensors, op.inputs[0]) else {
        result.not_assessed_reason = "ADD input 0 is unavailable.".to_string();
        return result;
    };
    let Some(input_1) = tensor_at(tensors, op.inputs[1]) else {
        result.not_assessed_reason = "ADD input 1 is unavailable.".to_string();
        return result;
    };
    let Some(output) = tensor_at(tensors, op.outputs[0]) else {
        result.not_assessed_reason = "ADD output is unavailable.".to_string();
        return result;
    };
    let Some(current_contracts) = add_contracts(input_0, input_1, output) else {
        result.not_assessed_reason =
            "ADD requires per-tensor INT8/UINT8 input and output contracts.".to_string();
        return result;
    };
    let mut candidate_contracts = current_contracts;
    for slot in &input_slots {
        if *slot < 2 {
            candidate_contracts.0[*slot] = candidate.scale;
            candidate_contracts.1[*slot] = candidate.zero_point;
        }
    }
    if source.scale_sample.first().copied().map(f64::from) != Some(current_scale)
        || source.zero_point_sample.first().copied() != Some(current_zero_point)
    {
        result.not_assessed_reason =
            "Lattice source contract does not match the parsed consumer input.".to_string();
        return result;
    }
    let Some(current_parameters) = derive_add_parameters(current_contracts) else {
        result.not_assessed_reason = "Current ADD multiplier domain is invalid.".to_string();
        return result;
    };
    let Some(candidate_parameters) = derive_add_parameters(candidate_contracts) else {
        result.not_assessed_reason = "Candidate ADD multiplier domain is invalid.".to_string();
        return result;
    };
    result.changed_offset_count = current_parameters
        .input_offsets
        .iter()
        .zip(candidate_parameters.input_offsets.iter())
        .filter(|(left, right)| left != right)
        .count();
    let current_encodings = [
        &current_parameters.input_multipliers[0],
        &current_parameters.input_multipliers[1],
        &current_parameters.output_multiplier,
    ];
    let candidate_encodings = [
        &candidate_parameters.input_multipliers[0],
        &candidate_parameters.input_multipliers[1],
        &candidate_parameters.output_multiplier,
    ];
    result.changed_multiplier_encoding_count = current_encodings
        .iter()
        .zip(candidate_encodings.iter())
        .filter(|(left, right)| {
            left.quantized_multiplier != right.quantized_multiplier || left.shift != right.shift
        })
        .count();
    result.changed_shift_count = current_encodings
        .iter()
        .zip(candidate_encodings.iter())
        .filter(|(left, right)| left.shift != right.shift)
        .count();
    result.assessment_status = "assessed";
    result.current_parameters = Some(current_parameters);
    result.candidate_parameters = Some(candidate_parameters);
    result
}

type AddContracts = ([f64; 2], [i64; 2], f64, i64);

fn add_contracts(
    input_0: &TensorInfo,
    input_1: &TensorInfo,
    output: &TensorInfo,
) -> Option<AddContracts> {
    if !matches!(input_0.dtype.as_str(), "INT8" | "UINT8")
        || input_0.dtype != input_1.dtype
        || input_0.dtype != output.dtype
        || input_0.scale_sample.len() != 1
        || input_1.scale_sample.len() != 1
        || output.scale_sample.len() != 1
        || input_0.zero_point_sample.len() != 1
        || input_1.zero_point_sample.len() != 1
        || output.zero_point_sample.len() != 1
    {
        return None;
    }
    let scales = [
        input_0.scale_sample[0] as f64,
        input_1.scale_sample[0] as f64,
    ];
    let output_scale = output.scale_sample[0] as f64;
    if !valid_scale(scales[0]) || !valid_scale(scales[1]) || !valid_scale(output_scale) {
        return None;
    }
    Some((
        scales,
        [input_0.zero_point_sample[0], input_1.zero_point_sample[0]],
        output_scale,
        output.zero_point_sample[0],
    ))
}

fn derive_add_parameters(contracts: AddContracts) -> Option<AddParameterSet> {
    let (input_scales, input_zero_points, output_scale, output_zero_point) = contracts;
    let left_shift = 20;
    let twice_max_input_scale = 2.0 * input_scales[0].max(input_scales[1]);
    let real_multipliers = [
        input_scales[0] / twice_max_input_scale,
        input_scales[1] / twice_max_input_scale,
        twice_max_input_scale / (2.0_f64.powi(left_shift) * output_scale),
    ];
    if real_multipliers
        .iter()
        .any(|value| !valid_scale(*value) || *value >= 1.0)
    {
        return None;
    }
    Some(AddParameterSet {
        left_shift,
        twice_max_input_scale,
        input_offsets: [-input_zero_points[0], -input_zero_points[1]],
        output_offset: output_zero_point,
        input_multipliers: [
            encoded_multiplier(real_multipliers[0]),
            encoded_multiplier(real_multipliers[1]),
        ],
        output_multiplier: encoded_multiplier(real_multipliers[2]),
    })
}

fn encoded_multiplier(real_multiplier: f64) -> EncodedMultiplier {
    let encoding = quantize_multiplier(real_multiplier, false);
    EncodedMultiplier {
        real_multiplier,
        quantized_multiplier: encoding.multiplier,
        shift: encoding.shift,
        represented_multiplier: encoding.represented,
    }
}

fn consumer_map(ops: &[OpInfo]) -> HashMap<i32, Vec<usize>> {
    let mut map = HashMap::<i32, Vec<usize>>::new();
    for op in ops {
        for input in op.inputs.iter().copied().filter(|input| *input >= 0) {
            let entry = map.entry(input).or_default();
            if !entry.contains(&op.index) {
                entry.push(op.index);
            }
        }
    }
    map
}

fn downstream_ops(
    source_tensor: i32,
    ops: &[OpInfo],
    consumers: &HashMap<i32, Vec<usize>>,
) -> Vec<AffectedOp> {
    let mut depths = BTreeMap::<usize, usize>::new();
    let mut queue = VecDeque::from([(source_tensor, 0usize)]);
    let mut visited_tensors = HashSet::from([source_tensor]);
    while let Some((tensor, tensor_depth)) = queue.pop_front() {
        for op_index in consumers.get(&tensor).into_iter().flatten() {
            let op_depth = tensor_depth + 1;
            depths
                .entry(*op_index)
                .and_modify(|current| *current = (*current).min(op_depth))
                .or_insert(op_depth);
            if let Some(op) = ops.get(*op_index) {
                for output in op.outputs.iter().copied().filter(|output| *output >= 0) {
                    if visited_tensors.insert(output) {
                        queue.push_back((output, op_depth));
                    }
                }
            }
        }
    }
    depths
        .into_iter()
        .filter_map(|(op_index, depth)| {
            ops.get(op_index).map(|op| AffectedOp {
                op_index,
                op_name: op.name.clone(),
                minimum_edge_depth: depth,
                direct_consumer: depth == 1,
            })
        })
        .collect()
}

fn matching_input_slots(op: &OpInfo, tensor_index: i32) -> Vec<usize> {
    op.inputs
        .iter()
        .enumerate()
        .filter_map(|(slot, input)| (*input == tensor_index).then_some(slot))
        .collect()
}

fn migration_class(op_name: &str) -> &'static str {
    if is_kernel(op_name) {
        "integer_kernel"
    } else if op_name == "ADD" {
        "quantized_add"
    } else {
        "not_modeled"
    }
}

fn is_kernel(op_name: &str) -> bool {
    matches!(op_name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED")
}

fn tensor_at(tensors: &[TensorInfo], index: i32) -> Option<&TensorInfo> {
    usize::try_from(index)
        .ok()
        .and_then(|value| tensors.get(value))
}

fn valid_scale(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn max_optional(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.reduce(f64::max)
}

fn f64_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn optional_f64_bits(value: Option<f64>) -> String {
    value.map(f64_bits).unwrap_or_else(|| "na".to_string())
}

fn kernel_ledger_row(op_index: usize, witness: &KernelChannelWitness) -> String {
    format!(
        "op={op_index};channel={};weight_scale={};current_real={};candidate_real={};current_q={};candidate_q={};current_shift={};candidate_shift={};current_bias={};candidate_bias={};overflow={};preserved_bias={};candidate_bias_real={};bias_error={};current_steps={};candidate_steps={}\n",
        witness.channel_index,
        f64_bits(witness.weight_scale),
        f64_bits(witness.current_real_multiplier),
        f64_bits(witness.candidate_real_multiplier),
        witness.current_quantized_multiplier,
        witness.candidate_quantized_multiplier,
        witness.current_shift,
        witness.candidate_shift,
        witness.current_bias_code,
        witness.candidate_bias_code_decimal,
        usize::from(witness.bias_int32_overflow),
        f64_bits(witness.preserved_bias_real_value),
        optional_f64_bits(witness.candidate_bias_real_value),
        optional_f64_bits(witness.absolute_bias_rebase_error),
        optional_f64_bits(witness.absolute_bias_rebase_error_current_steps),
        optional_f64_bits(witness.absolute_bias_rebase_error_candidate_steps),
    )
}

fn ledger_hash_method() -> &'static str {
    "SHA-256 over UTF-8 rows op=<index>;channel=<index>;weight_scale=<f64hex>;current_real=<f64hex>;candidate_real=<f64hex>;current_q=<i32>;candidate_q=<i32>;current_shift=<i32>;candidate_shift=<i32>;current_bias=<i32>;candidate_bias=<decimal>;overflow=<0|1>;preserved_bias=<f64hex>;candidate_bias_real=<f64hex|na>;bias_error=<f64hex|na>;current_steps=<f64hex|na>;candidate_steps=<f64hex|na>\\n"
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn source_references() -> Vec<SourceReference> {
    [
        (
            "ADD prepare-time parameter derivation",
            "tensorflow/lite/kernels/add.cc",
            "436dbd27aba268d8828b07ce1447e6c8a979324925667a0a3c8987d9185b6947",
        ),
        (
            "Q0.31 multiplier encoding",
            "tensorflow/lite/kernels/internal/quantization_util.cc",
            "22e46f15663437c407298f5230545600faa2f6b2f1b46488e20c97ff3a5c96f9",
        ),
        (
            "CONV/DW/FC effective-scale derivation",
            "tensorflow/lite/kernels/kernel_util.cc",
            "fb03b532b1f510ccf5d7d169eeebcc408791677c97cbce235893560b4379da49",
        ),
    ]
    .into_iter()
    .map(|(role, file, sha256)| SourceReference {
        role,
        file,
        url: format!("https://github.com/tensorflow/tensorflow/blob/{TFLITE_SOURCE_COMMIT}/{file}"),
        sha256,
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_parameters_follow_pinned_left_shift_twenty_path() {
        let params = derive_add_parameters(([0.25, 0.5], [128, 127], 0.75, 120))
            .expect("valid 8-bit ADD contracts");
        assert_eq!(params.left_shift, 20);
        assert_eq!(params.twice_max_input_scale, 1.0);
        assert_eq!(params.input_offsets, [-128, -127]);
        assert_eq!(params.input_multipliers[0].real_multiplier, 0.25);
        assert_eq!(params.input_multipliers[1].real_multiplier, 0.5);
        assert_eq!(params.output_multiplier.real_multiplier, 1.0 / 786_432.0);
    }

    #[test]
    fn bias_rebase_ratio_cancels_weight_scale() {
        let old_code = 101i32;
        let old_scale = 0.25;
        let candidate_scale = 0.5;
        let new_code = round_ties_away_from_zero(old_code as f64 * old_scale / candidate_scale);
        assert_eq!(new_code, 51);
        let weight_scale = 0.125;
        let old_real = old_code as f64 * old_scale * weight_scale;
        let new_real = new_code as f64 * candidate_scale * weight_scale;
        assert!((old_real - new_real).abs() <= candidate_scale * weight_scale * 0.5);
    }
}
