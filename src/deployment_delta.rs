use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen::prelude::*;

const DELTA_SCHEMA: &str = "deepbom.deployment_delta.v1.1";
const DELTA_METHOD_VERSION: &str = "2026-07-17.2";
const MAX_TARGETS: usize = 8;

#[derive(Clone, Copy, Default, Serialize)]
struct DeltaComponents {
    compute_us: f64,
    memory_us: f64,
    packing_us: f64,
    boundary_us: f64,
    fallback_us: f64,
    total_us: f64,
}

impl DeltaComponents {
    fn from_op(op: &OpInfo) -> Self {
        let (compute_us, memory_us) = active_roofline_components(
            finite_non_negative(op.bottleneck_compute_us),
            finite_non_negative(op.bottleneck_memory_us),
        );
        let packing_us = finite_non_negative(op.bottleneck_packing_us);
        let boundary_us = finite_non_negative(op.bottleneck_break_us);
        let fallback_us = finite_non_negative(op.bottleneck_fallback_us);
        Self {
            compute_us,
            memory_us,
            packing_us,
            boundary_us,
            fallback_us,
            total_us: compute_us + memory_us + packing_us + boundary_us + fallback_us,
        }
    }

    fn add(&mut self, other: Self) {
        self.compute_us += other.compute_us;
        self.memory_us += other.memory_us;
        self.packing_us += other.packing_us;
        self.boundary_us += other.boundary_us;
        self.fallback_us += other.fallback_us;
        self.total_us += other.total_us;
    }

    fn signed_delta(candidate: Self, baseline: Self) -> Self {
        Self {
            compute_us: candidate.compute_us - baseline.compute_us,
            memory_us: candidate.memory_us - baseline.memory_us,
            packing_us: candidate.packing_us - baseline.packing_us,
            boundary_us: candidate.boundary_us - baseline.boundary_us,
            fallback_us: candidate.fallback_us - baseline.fallback_us,
            total_us: candidate.total_us - baseline.total_us,
        }
    }
}

#[derive(Serialize)]
struct DeltaArtifact {
    role: &'static str,
    filename: String,
    sha256: String,
    file_size: usize,
    format: String,
    operator_count: usize,
    tensor_count: usize,
    total_macs: f64,
    quantization_classification: String,
    quantized_compute_mac_ratio: f64,
    predicted_effective_chain_breaks: usize,
    delegated_mac_ratio: f64,
    zero_kernel_slice_count: usize,
    low_grid_utilization_tensors: usize,
    saturated_quantized_tensors: usize,
    input_contracts: Vec<String>,
    output_contracts: Vec<String>,
}

#[derive(Clone)]
struct AlignmentEntry {
    entity_id: String,
    relation: &'static str,
    match_class: &'static str,
    baseline_position: Option<usize>,
    candidate_position: Option<usize>,
}

#[derive(Serialize)]
struct AlignmentRow {
    entity_id: String,
    relation: &'static str,
    match_class: &'static str,
    baseline_op_index: Option<usize>,
    candidate_op_index: Option<usize>,
    baseline_op_name: Option<String>,
    candidate_op_name: Option<String>,
    baseline_output_shapes: Option<Vec<Vec<i32>>>,
    candidate_output_shapes: Option<Vec<Vec<i32>>>,
    output_shape_changed: Option<bool>,
    baseline_quantization_state: Option<String>,
    candidate_quantization_state: Option<String>,
    quantization_transition: bool,
    baseline_static_assignment: Option<&'static str>,
    candidate_static_assignment: Option<&'static str>,
    static_assignment_transition: bool,
    baseline_macs: f64,
    candidate_macs: f64,
    signed_macs_delta: f64,
}

#[derive(Serialize)]
struct AlignmentSummary {
    method: &'static str,
    exact_structural_match_count: usize,
    op_sequence_match_count: usize,
    matched_op_count: usize,
    added_op_count: usize,
    removed_op_count: usize,
    baseline_match_ratio: f64,
    candidate_match_ratio: f64,
    artifact_relation: &'static str,
    semantic_identity_conclusion: &'static str,
}

#[derive(Clone, Serialize)]
struct TargetDriver {
    entity_id: String,
    relation: &'static str,
    match_class: &'static str,
    baseline_op_index: Option<usize>,
    candidate_op_index: Option<usize>,
    baseline_op_name: Option<String>,
    candidate_op_name: Option<String>,
    baseline_us: f64,
    candidate_us: f64,
    baseline_components: DeltaComponents,
    candidate_components: DeltaComponents,
    signed_delta_us: f64,
    relative_delta: Option<f64>,
    absolute_change_share: f64,
    positive_regression_share: f64,
    negative_improvement_share: f64,
    baseline_contribution_share: f64,
    candidate_contribution_share: f64,
    baseline_rank: Option<usize>,
    candidate_rank: Option<usize>,
    component_delta: DeltaComponents,
}

#[derive(Serialize)]
struct TargetDelta {
    target_id: String,
    target_label: String,
    target_profile_sha256: String,
    baseline_total_us: f64,
    candidate_total_us: f64,
    signed_delta_us: f64,
    relative_delta: Option<f64>,
    baseline_components: DeltaComponents,
    candidate_components: DeltaComponents,
    component_delta: DeltaComponents,
    positive_regression_us: f64,
    negative_improvement_us: f64,
    absolute_change_us: f64,
    driver_delta_sum_us: f64,
    conservation_error_us: f64,
    top_regression_entity_id: Option<String>,
    top_regression_delta_us: f64,
    top_improvement_entity_id: Option<String>,
    top_improvement_delta_us: f64,
    drivers: Vec<TargetDriver>,
}

#[derive(Default)]
struct CrossTargetAccumulator {
    regression_target_count: usize,
    improvement_target_count: usize,
    unchanged_target_count: usize,
    min_delta_us: f64,
    max_delta_us: f64,
    max_absolute_delta_us: f64,
    max_absolute_change_share: f64,
}

#[derive(Serialize)]
struct CrossTargetDriver {
    entity_id: String,
    relation: &'static str,
    match_class: &'static str,
    baseline_op_index: Option<usize>,
    candidate_op_index: Option<usize>,
    baseline_op_name: Option<String>,
    candidate_op_name: Option<String>,
    regression_target_count: usize,
    improvement_target_count: usize,
    unchanged_target_count: usize,
    consistent_regression: bool,
    consistent_improvement: bool,
    min_delta_us: f64,
    max_delta_us: f64,
    max_absolute_delta_us: f64,
    max_absolute_change_share: f64,
}

#[derive(Serialize)]
struct GraphDelta {
    signed_file_size_bytes: i64,
    signed_operator_count: i64,
    signed_tensor_count: i64,
    signed_total_macs: f64,
    relative_total_macs_delta: Option<f64>,
    signed_quantized_compute_mac_ratio: f64,
    signed_predicted_effective_chain_breaks: i64,
    signed_delegated_mac_ratio: f64,
    signed_zero_kernel_slice_count: i64,
    signed_low_grid_utilization_tensors: i64,
    signed_saturated_quantized_tensors: i64,
    input_contract_changed: bool,
    output_contract_changed: bool,
}

#[derive(Serialize)]
struct DeploymentDelta {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    baseline: DeltaArtifact,
    candidate: DeltaArtifact,
    graph_delta: GraphDelta,
    alignment: AlignmentSummary,
    alignment_rows: Vec<AlignmentRow>,
    target_count: usize,
    target_deltas: Vec<TargetDelta>,
    cross_target_drivers: Vec<CrossTargetDriver>,
    worst_relative_delta_target_id: Option<String>,
    worst_relative_delta: Option<f64>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[wasm_bindgen]
pub fn compute_deployment_delta(
    baseline_bytes: &[u8],
    baseline_filename: &str,
    candidate_bytes: &[u8],
    candidate_filename: &str,
    target_ids_json: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    if baseline_bytes.is_empty() || candidate_bytes.is_empty() {
        return Err(JsValue::from_str(
            "Both baseline and candidate model bytes are required.",
        ));
    }
    let target_ids =
        parse_delta_target_ids(target_ids_json).map_err(|error| JsValue::from_str(&error))?;
    let baseline_analyses = target_ids
        .iter()
        .map(|target_id| {
            analyze_with_target_without_step_response(baseline_bytes, baseline_filename, target_id)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| JsValue::from_str(&error))?;
    let candidate_analyses = target_ids
        .iter()
        .map(|target_id| {
            analyze_with_target_without_step_response(
                candidate_bytes,
                candidate_filename,
                target_id,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| JsValue::from_str(&error))?;
    let delta = build_deployment_delta(
        &baseline_analyses,
        &candidate_analyses,
        hex_lower(&Sha256::digest(baseline_bytes)),
        hex_lower(&Sha256::digest(candidate_bytes)),
    )
    .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&delta).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn parse_delta_target_ids(value: &str) -> Result<Vec<String>, String> {
    let parsed: Vec<String> =
        serde_json::from_str(value).map_err(|error| format!("Invalid target ID JSON: {error}"))?;
    let mut seen = BTreeSet::new();
    let target_ids = parsed
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .filter(|value| seen.insert(value.clone()))
        .collect::<Vec<_>>();
    if target_ids.len() < 2 {
        return Err("Deployment delta requires at least two distinct target profiles.".to_string());
    }
    if target_ids.len() > MAX_TARGETS {
        return Err(format!(
            "Deployment delta accepts at most {MAX_TARGETS} target profiles."
        ));
    }
    Ok(target_ids)
}

fn build_deployment_delta(
    baseline_analyses: &[Analysis],
    candidate_analyses: &[Analysis],
    baseline_sha256: String,
    candidate_sha256: String,
) -> Result<DeploymentDelta, String> {
    if baseline_analyses.len() != candidate_analyses.len() || baseline_analyses.len() < 2 {
        return Err("Baseline and candidate target ledgers are incomplete.".to_string());
    }
    let baseline = &baseline_analyses[0];
    let candidate = &candidate_analyses[0];
    if baseline.format != "tflite" || candidate.format != "tflite" {
        return Err("Deployment delta v1 supports TFLite artifacts only.".to_string());
    }
    for (left, right) in baseline_analyses.iter().zip(candidate_analyses) {
        if left.target_profile.id != right.target_profile.id {
            return Err("Baseline and candidate target profile order differs.".to_string());
        }
        if left.target_profile.profile_sha256 != right.target_profile.profile_sha256 {
            return Err(format!(
                "Baseline and candidate target profile digest differs for {}.",
                left.target_profile.id
            ));
        }
        for analysis in [left, right] {
            let component_total = aggregate_components(&analysis.ops).total_us;
            let bottleneck_total: f64 = analysis
                .ops
                .iter()
                .map(|op| finite_non_negative(op.bottleneck_total_us))
                .sum();
            if !near(component_total, bottleneck_total) {
                return Err(format!(
                    "Target {} roofline component ledger does not conserve bottleneck_total_us.",
                    analysis.target_profile.id
                ));
            }
        }
    }

    let entries = align_ops(baseline, candidate);
    let alignment_rows = entries
        .iter()
        .map(|entry| alignment_row(entry, baseline, candidate))
        .collect::<Vec<_>>();
    let exact_structural_match_count = entries
        .iter()
        .filter(|entry| entry.match_class == "exact_structural_signature")
        .count();
    let op_sequence_match_count = entries
        .iter()
        .filter(|entry| entry.match_class == "op_type_sequence_alignment")
        .count();
    let matched_op_count = exact_structural_match_count + op_sequence_match_count;
    let alignment = AlignmentSummary {
        method: "Deterministic two-pass sequence alignment. Pass 1 is Hirschberg LCS over op type plus input/output dtype, shape, constant-role, and quantization-presence signatures. Pass 2 applies Hirschberg LCS over op type only inside unmatched anchor gaps. Ties choose the earliest candidate split. Added and removed ops remain explicit.",
        exact_structural_match_count,
        op_sequence_match_count,
        matched_op_count,
        added_op_count: entries.iter().filter(|entry| entry.relation == "added").count(),
        removed_op_count: entries.iter().filter(|entry| entry.relation == "removed").count(),
        baseline_match_ratio: ratio(matched_op_count as f64, baseline.ops.len() as f64),
        candidate_match_ratio: ratio(matched_op_count as f64, candidate.ops.len() as f64),
        artifact_relation: if baseline_sha256 == candidate_sha256 {
            "identical_bytes"
        } else {
            "different_artifacts_lineage_unproven"
        },
        semantic_identity_conclusion: "NOT_CONCLUDED: sequence alignment establishes a deterministic comparison coordinate, not semantic layer identity or common training lineage.",
    };
    let baseline_artifact = delta_artifact("baseline", baseline, baseline_sha256);
    let candidate_artifact = delta_artifact("candidate", candidate, candidate_sha256);
    let graph_delta = graph_delta(&baseline_artifact, &candidate_artifact);
    let target_deltas = baseline_analyses
        .iter()
        .zip(candidate_analyses)
        .map(|(left, right)| target_delta(left, right, &entries))
        .collect::<Vec<_>>();
    let cross_target_drivers = cross_target_drivers(&entries, &target_deltas, baseline, candidate);
    let mut worst: Option<(String, f64)> = None;
    for target in &target_deltas {
        let Some(value) = target.relative_delta else {
            continue;
        };
        if worst.as_ref().is_none_or(|current| value > current.1) {
            worst = Some((target.target_id.clone(), value));
        }
    }
    Ok(DeploymentDelta {
        schema: DELTA_SCHEMA,
        method_version: DELTA_METHOD_VERSION,
        evidence_class: "DERIVED_FROM_TWO_ARTIFACTS_AND_PINNED_TARGET_PROFILE_STATIC_ESTIMATES",
        baseline: baseline_artifact,
        candidate: candidate_artifact,
        graph_delta,
        alignment,
        alignment_rows,
        target_count: target_deltas.len(),
        target_deltas,
        cross_target_drivers,
        worst_relative_delta_target_id: worst.as_ref().map(|item| item.0.clone()),
        worst_relative_delta: worst.map(|item| item.1),
        method: "Both artifacts are independently parsed for every requested pinned planning profile. Each op assigns max(compute, memory) exactly once to its active roofline-bound component, then adds packing, predicted-boundary, and predicted-fallback terms. For each target, candidate minus baseline modeled time is decomposed across matched, added, and removed alignment entities; the signed driver sum must equal the target total delta.",
        interpretation_boundary: "All time deltas are differences between static planning estimates, not device measurements. Op-type sequence matches are comparison coordinates only. Different artifact hashes do not establish lineage. Runtime graph transformations, placement, executed kernels, scheduler behavior, cache state, and numerical/task quality require separately bound evidence.",
    })
}

fn delta_artifact(role: &'static str, analysis: &Analysis, sha256: String) -> DeltaArtifact {
    DeltaArtifact {
        role,
        filename: analysis.filename.clone(),
        sha256,
        file_size: analysis.file_size,
        format: analysis.format.clone(),
        operator_count: analysis.ops.len(),
        tensor_count: analysis.tensor_count,
        total_macs: finite_non_negative(analysis.total_macs),
        quantization_classification: analysis.quantization_status.classification.clone(),
        quantized_compute_mac_ratio: finite_non_negative(
            analysis.quantization_status.quantized_compute_mac_percent,
        ),
        predicted_effective_chain_breaks: analysis.xnnpack_effective_chain_breaks,
        delegated_mac_ratio: finite_non_negative(analysis.delegated_mac_percent),
        zero_kernel_slice_count: analysis.weight_integrity.zero_kernel_slice_count,
        low_grid_utilization_tensors: analysis.weight_integrity.low_grid_utilization_tensors,
        saturated_quantized_tensors: analysis.weight_integrity.saturated_quantized_tensors,
        input_contracts: analysis.inputs.iter().map(tensor_contract).collect(),
        output_contracts: analysis.outputs.iter().map(tensor_contract).collect(),
    }
}

fn graph_delta(baseline: &DeltaArtifact, candidate: &DeltaArtifact) -> GraphDelta {
    GraphDelta {
        signed_file_size_bytes: candidate.file_size as i64 - baseline.file_size as i64,
        signed_operator_count: candidate.operator_count as i64 - baseline.operator_count as i64,
        signed_tensor_count: candidate.tensor_count as i64 - baseline.tensor_count as i64,
        signed_total_macs: candidate.total_macs - baseline.total_macs,
        relative_total_macs_delta: relative_delta(candidate.total_macs, baseline.total_macs),
        signed_quantized_compute_mac_ratio: candidate.quantized_compute_mac_ratio
            - baseline.quantized_compute_mac_ratio,
        signed_predicted_effective_chain_breaks: candidate.predicted_effective_chain_breaks as i64
            - baseline.predicted_effective_chain_breaks as i64,
        signed_delegated_mac_ratio: candidate.delegated_mac_ratio - baseline.delegated_mac_ratio,
        signed_zero_kernel_slice_count: candidate.zero_kernel_slice_count as i64
            - baseline.zero_kernel_slice_count as i64,
        signed_low_grid_utilization_tensors: candidate.low_grid_utilization_tensors as i64
            - baseline.low_grid_utilization_tensors as i64,
        signed_saturated_quantized_tensors: candidate.saturated_quantized_tensors as i64
            - baseline.saturated_quantized_tensors as i64,
        input_contract_changed: baseline.input_contracts != candidate.input_contracts,
        output_contract_changed: baseline.output_contracts != candidate.output_contracts,
    }
}

fn target_delta(
    baseline: &Analysis,
    candidate: &Analysis,
    entries: &[AlignmentEntry],
) -> TargetDelta {
    let baseline_components = aggregate_components(&baseline.ops);
    let candidate_components = aggregate_components(&candidate.ops);
    let component_delta = DeltaComponents::signed_delta(candidate_components, baseline_components);
    let baseline_ranks = descending_ranks(&baseline.ops);
    let candidate_ranks = descending_ranks(&candidate.ops);
    let raw = entries
        .iter()
        .map(|entry| {
            let baseline_op = entry
                .baseline_position
                .map(|position| &baseline.ops[position]);
            let candidate_op = entry
                .candidate_position
                .map(|position| &candidate.ops[position]);
            let baseline_component = baseline_op
                .map(DeltaComponents::from_op)
                .unwrap_or_default();
            let candidate_component = candidate_op
                .map(DeltaComponents::from_op)
                .unwrap_or_default();
            let signed_delta_us = candidate_component.total_us - baseline_component.total_us;
            (
                entry,
                baseline_op,
                candidate_op,
                baseline_component,
                candidate_component,
                signed_delta_us,
            )
        })
        .collect::<Vec<_>>();
    let positive_regression_us: f64 = raw.iter().map(|item| item.5.max(0.0)).sum();
    let negative_improvement_us: f64 = raw.iter().map(|item| item.5.min(0.0)).sum();
    let absolute_change_us: f64 = raw.iter().map(|item| item.5.abs()).sum();
    let mut drivers = raw
        .into_iter()
        .map(
            |(
                entry,
                baseline_op,
                candidate_op,
                baseline_component,
                candidate_component,
                signed_delta_us,
            )| TargetDriver {
                entity_id: entry.entity_id.clone(),
                relation: entry.relation,
                match_class: entry.match_class,
                baseline_op_index: baseline_op.map(|op| op.index),
                candidate_op_index: candidate_op.map(|op| op.index),
                baseline_op_name: baseline_op.map(|op| op.name.clone()),
                candidate_op_name: candidate_op.map(|op| op.name.clone()),
                baseline_us: baseline_component.total_us,
                candidate_us: candidate_component.total_us,
                baseline_components: baseline_component,
                candidate_components: candidate_component,
                signed_delta_us,
                relative_delta: relative_delta(
                    candidate_component.total_us,
                    baseline_component.total_us,
                ),
                absolute_change_share: ratio(signed_delta_us.abs(), absolute_change_us),
                positive_regression_share: ratio(signed_delta_us.max(0.0), positive_regression_us),
                negative_improvement_share: ratio(
                    (-signed_delta_us).max(0.0),
                    -negative_improvement_us,
                ),
                baseline_contribution_share: ratio(
                    baseline_component.total_us,
                    baseline_components.total_us,
                ),
                candidate_contribution_share: ratio(
                    candidate_component.total_us,
                    candidate_components.total_us,
                ),
                baseline_rank: entry
                    .baseline_position
                    .map(|position| baseline_ranks[position]),
                candidate_rank: entry
                    .candidate_position
                    .map(|position| candidate_ranks[position]),
                component_delta: DeltaComponents::signed_delta(
                    candidate_component,
                    baseline_component,
                ),
            },
        )
        .collect::<Vec<_>>();
    drivers.sort_by(|left, right| {
        compare_desc(left.signed_delta_us.abs(), right.signed_delta_us.abs())
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    let driver_delta_sum_us: f64 = drivers.iter().map(|driver| driver.signed_delta_us).sum();
    let signed_delta_us = candidate_components.total_us - baseline_components.total_us;
    let top_regression = drivers
        .iter()
        .filter(|driver| driver.signed_delta_us > 0.0)
        .max_by(|left, right| {
            left.signed_delta_us
                .partial_cmp(&right.signed_delta_us)
                .unwrap_or(Ordering::Equal)
        });
    let top_improvement = drivers
        .iter()
        .filter(|driver| driver.signed_delta_us < 0.0)
        .min_by(|left, right| {
            left.signed_delta_us
                .partial_cmp(&right.signed_delta_us)
                .unwrap_or(Ordering::Equal)
        });
    let top_regression_entity_id = top_regression.map(|driver| driver.entity_id.clone());
    let top_regression_delta_us = top_regression
        .map(|driver| driver.signed_delta_us)
        .unwrap_or(0.0);
    let top_improvement_entity_id = top_improvement.map(|driver| driver.entity_id.clone());
    let top_improvement_delta_us = top_improvement
        .map(|driver| driver.signed_delta_us)
        .unwrap_or(0.0);
    TargetDelta {
        target_id: baseline.target_profile.id.clone(),
        target_label: baseline.target_profile.label.clone(),
        target_profile_sha256: baseline.target_profile.profile_sha256.clone(),
        baseline_total_us: baseline_components.total_us,
        candidate_total_us: candidate_components.total_us,
        signed_delta_us,
        relative_delta: relative_delta(candidate_components.total_us, baseline_components.total_us),
        baseline_components,
        candidate_components,
        component_delta,
        positive_regression_us,
        negative_improvement_us,
        absolute_change_us,
        driver_delta_sum_us,
        conservation_error_us: driver_delta_sum_us - signed_delta_us,
        top_regression_entity_id,
        top_regression_delta_us,
        top_improvement_entity_id,
        top_improvement_delta_us,
        drivers,
    }
}

fn cross_target_drivers(
    entries: &[AlignmentEntry],
    target_deltas: &[TargetDelta],
    baseline: &Analysis,
    candidate: &Analysis,
) -> Vec<CrossTargetDriver> {
    let mut accumulators = BTreeMap::<String, CrossTargetAccumulator>::new();
    for target in target_deltas {
        for driver in &target.drivers {
            let item = accumulators
                .entry(driver.entity_id.clone())
                .or_insert_with(|| CrossTargetAccumulator {
                    min_delta_us: f64::INFINITY,
                    max_delta_us: f64::NEG_INFINITY,
                    ..CrossTargetAccumulator::default()
                });
            if driver.signed_delta_us > 1e-12 {
                item.regression_target_count += 1;
            } else if driver.signed_delta_us < -1e-12 {
                item.improvement_target_count += 1;
            } else {
                item.unchanged_target_count += 1;
            }
            item.min_delta_us = item.min_delta_us.min(driver.signed_delta_us);
            item.max_delta_us = item.max_delta_us.max(driver.signed_delta_us);
            item.max_absolute_delta_us =
                item.max_absolute_delta_us.max(driver.signed_delta_us.abs());
            item.max_absolute_change_share = item
                .max_absolute_change_share
                .max(driver.absolute_change_share);
        }
    }
    let mut rows = entries
        .iter()
        .map(|entry| {
            let acc = accumulators
                .get(&entry.entity_id)
                .expect("every alignment entity has target rows");
            let baseline_op = entry
                .baseline_position
                .map(|position| &baseline.ops[position]);
            let candidate_op = entry
                .candidate_position
                .map(|position| &candidate.ops[position]);
            CrossTargetDriver {
                entity_id: entry.entity_id.clone(),
                relation: entry.relation,
                match_class: entry.match_class,
                baseline_op_index: baseline_op.map(|op| op.index),
                candidate_op_index: candidate_op.map(|op| op.index),
                baseline_op_name: baseline_op.map(|op| op.name.clone()),
                candidate_op_name: candidate_op.map(|op| op.name.clone()),
                regression_target_count: acc.regression_target_count,
                improvement_target_count: acc.improvement_target_count,
                unchanged_target_count: acc.unchanged_target_count,
                consistent_regression: acc.regression_target_count == target_deltas.len(),
                consistent_improvement: acc.improvement_target_count == target_deltas.len(),
                min_delta_us: acc.min_delta_us,
                max_delta_us: acc.max_delta_us,
                max_absolute_delta_us: acc.max_absolute_delta_us,
                max_absolute_change_share: acc.max_absolute_change_share,
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .consistent_regression
            .cmp(&left.consistent_regression)
            .then_with(|| {
                compare_desc(
                    left.max_absolute_change_share,
                    right.max_absolute_change_share,
                )
            })
            .then_with(|| compare_desc(left.max_absolute_delta_us, right.max_absolute_delta_us))
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    rows
}

fn alignment_row(
    entry: &AlignmentEntry,
    baseline: &Analysis,
    candidate: &Analysis,
) -> AlignmentRow {
    let baseline_op = entry
        .baseline_position
        .map(|position| &baseline.ops[position]);
    let candidate_op = entry
        .candidate_position
        .map(|position| &candidate.ops[position]);
    let baseline_assignment = baseline_op.map(static_assignment);
    let candidate_assignment = candidate_op.map(static_assignment);
    AlignmentRow {
        entity_id: entry.entity_id.clone(),
        relation: entry.relation,
        match_class: entry.match_class,
        baseline_op_index: baseline_op.map(|op| op.index),
        candidate_op_index: candidate_op.map(|op| op.index),
        baseline_op_name: baseline_op.map(|op| op.name.clone()),
        candidate_op_name: candidate_op.map(|op| op.name.clone()),
        baseline_output_shapes: baseline_op.map(|op| op.output_shapes.clone()),
        candidate_output_shapes: candidate_op.map(|op| op.output_shapes.clone()),
        output_shape_changed: baseline_op
            .zip(candidate_op)
            .map(|(left, right)| left.output_shapes != right.output_shapes),
        baseline_quantization_state: baseline_op.map(|op| op.quantization_state.clone()),
        candidate_quantization_state: candidate_op.map(|op| op.quantization_state.clone()),
        quantization_transition: baseline_op
            .zip(candidate_op)
            .is_some_and(|(left, right)| left.quantization_state != right.quantization_state),
        baseline_static_assignment: baseline_assignment,
        candidate_static_assignment: candidate_assignment,
        static_assignment_transition: baseline_assignment
            .zip(candidate_assignment)
            .is_some_and(|(left, right)| left != right),
        baseline_macs: baseline_op
            .map(|op| finite_non_negative(op.macs))
            .unwrap_or(0.0),
        candidate_macs: candidate_op
            .map(|op| finite_non_negative(op.macs))
            .unwrap_or(0.0),
        signed_macs_delta: candidate_op
            .map(|op| finite_non_negative(op.macs))
            .unwrap_or(0.0)
            - baseline_op
                .map(|op| finite_non_negative(op.macs))
                .unwrap_or(0.0),
    }
}

fn align_ops(baseline: &Analysis, candidate: &Analysis) -> Vec<AlignmentEntry> {
    let baseline_signatures = baseline
        .ops
        .iter()
        .map(|op| structural_signature(baseline, op))
        .collect::<Vec<_>>();
    let candidate_signatures = candidate
        .ops
        .iter()
        .map(|op| structural_signature(candidate, op))
        .collect::<Vec<_>>();
    let exact_pairs = hirschberg_matches(&baseline_signatures, &candidate_signatures);
    let exact_set = exact_pairs.iter().copied().collect::<BTreeSet<_>>();
    let mut matched_pairs = exact_pairs.clone();
    let mut anchors = Vec::with_capacity(exact_pairs.len() + 2);
    anchors.push((usize::MAX, usize::MAX));
    anchors.extend(exact_pairs.iter().copied());
    anchors.push((baseline.ops.len(), candidate.ops.len()));
    for window in anchors.windows(2) {
        let (previous_baseline, previous_candidate) = window[0];
        let (next_baseline, next_candidate) = window[1];
        let baseline_start = if previous_baseline == usize::MAX {
            0
        } else {
            previous_baseline + 1
        };
        let candidate_start = if previous_candidate == usize::MAX {
            0
        } else {
            previous_candidate + 1
        };
        if baseline_start >= next_baseline || candidate_start >= next_candidate {
            continue;
        }
        let baseline_names = baseline.ops[baseline_start..next_baseline]
            .iter()
            .map(|op| op.name.clone())
            .collect::<Vec<_>>();
        let candidate_names = candidate.ops[candidate_start..next_candidate]
            .iter()
            .map(|op| op.name.clone())
            .collect::<Vec<_>>();
        matched_pairs.extend(
            hirschberg_matches(&baseline_names, &candidate_names)
                .into_iter()
                .map(|(left, right)| (left + baseline_start, right + candidate_start)),
        );
    }
    matched_pairs.sort_unstable();
    let mut entries = Vec::new();
    let mut baseline_cursor = 0usize;
    let mut candidate_cursor = 0usize;
    for (baseline_position, candidate_position) in matched_pairs {
        while baseline_cursor < baseline_position {
            push_alignment_entry(
                &mut entries,
                "removed",
                "unmatched_baseline",
                Some(baseline_cursor),
                None,
            );
            baseline_cursor += 1;
        }
        while candidate_cursor < candidate_position {
            push_alignment_entry(
                &mut entries,
                "added",
                "unmatched_candidate",
                None,
                Some(candidate_cursor),
            );
            candidate_cursor += 1;
        }
        push_alignment_entry(
            &mut entries,
            "matched",
            if exact_set.contains(&(baseline_position, candidate_position)) {
                "exact_structural_signature"
            } else {
                "op_type_sequence_alignment"
            },
            Some(baseline_position),
            Some(candidate_position),
        );
        baseline_cursor = baseline_position + 1;
        candidate_cursor = candidate_position + 1;
    }
    while baseline_cursor < baseline.ops.len() {
        push_alignment_entry(
            &mut entries,
            "removed",
            "unmatched_baseline",
            Some(baseline_cursor),
            None,
        );
        baseline_cursor += 1;
    }
    while candidate_cursor < candidate.ops.len() {
        push_alignment_entry(
            &mut entries,
            "added",
            "unmatched_candidate",
            None,
            Some(candidate_cursor),
        );
        candidate_cursor += 1;
    }
    entries
}

fn push_alignment_entry(
    entries: &mut Vec<AlignmentEntry>,
    relation: &'static str,
    match_class: &'static str,
    baseline_position: Option<usize>,
    candidate_position: Option<usize>,
) {
    entries.push(AlignmentEntry {
        entity_id: format!("E{:04}", entries.len()),
        relation,
        match_class,
        baseline_position,
        candidate_position,
    });
}

fn hirschberg_matches(left: &[String], right: &[String]) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    hirschberg_recursive(left, right, 0, 0, &mut matches);
    matches
}

fn hirschberg_recursive(
    left: &[String],
    right: &[String],
    left_offset: usize,
    right_offset: usize,
    matches: &mut Vec<(usize, usize)>,
) {
    if left.is_empty() || right.is_empty() {
        return;
    }
    if left.len() == 1 {
        if let Some(position) = right.iter().position(|value| value == &left[0]) {
            matches.push((left_offset, right_offset + position));
        }
        return;
    }
    let middle = left.len() / 2;
    let forward = lcs_lengths(&left[..middle], right);
    let reversed_left = left[middle..].iter().rev().cloned().collect::<Vec<_>>();
    let reversed_right = right.iter().rev().cloned().collect::<Vec<_>>();
    let backward = lcs_lengths(&reversed_left, &reversed_right);
    let mut split = 0usize;
    let mut best_score = forward[0] + backward[right.len()];
    for position in 1..=right.len() {
        let score = forward[position] + backward[right.len() - position];
        if score > best_score {
            best_score = score;
            split = position;
        }
    }
    hirschberg_recursive(
        &left[..middle],
        &right[..split],
        left_offset,
        right_offset,
        matches,
    );
    hirschberg_recursive(
        &left[middle..],
        &right[split..],
        left_offset + middle,
        right_offset + split,
        matches,
    );
}

fn lcs_lengths(left: &[String], right: &[String]) -> Vec<usize> {
    let mut previous = vec![0usize; right.len() + 1];
    let mut current = vec![0usize; right.len() + 1];
    for left_value in left {
        for (index, right_value) in right.iter().enumerate() {
            current[index + 1] = if left_value == right_value {
                previous[index] + 1
            } else {
                current[index].max(previous[index + 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    previous
}

fn structural_signature(analysis: &Analysis, op: &OpInfo) -> String {
    let inputs = op
        .inputs
        .iter()
        .map(|index| tensor_signature(analysis, *index))
        .collect::<Vec<_>>()
        .join(",");
    let outputs = op
        .outputs
        .iter()
        .map(|index| tensor_signature(analysis, *index))
        .collect::<Vec<_>>()
        .join(",");
    format!("{}|in={inputs}|out={outputs}", op.name)
}

fn tensor_signature(analysis: &Analysis, index: i32) -> String {
    if index < 0 {
        return "optional".to_string();
    }
    analysis
        .tensors
        .get(index as usize)
        .map(|tensor| {
            format!(
                "{}:{}:{:?}:q{}",
                if tensor.constant_buffer {
                    "const"
                } else {
                    "value"
                },
                tensor.dtype,
                tensor.shape,
                usize::from(tensor.quant_scales > 0)
            )
        })
        .unwrap_or_else(|| "missing".to_string())
}

fn tensor_contract(tensor: &TensorInfo) -> String {
    format!(
        "{}[{}]",
        tensor.dtype,
        tensor
            .shape
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn static_assignment(op: &OpInfo) -> &'static str {
    if op.xnnpack_supported && !op.xnnpack_chain_break {
        "predicted_delegate"
    } else {
        "predicted_cpu"
    }
}

fn aggregate_components(ops: &[OpInfo]) -> DeltaComponents {
    let mut total = DeltaComponents::default();
    for op in ops {
        total.add(DeltaComponents::from_op(op));
    }
    total
}

fn descending_ranks(ops: &[OpInfo]) -> Vec<usize> {
    let mut order = (0..ops.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        compare_desc(
            finite_non_negative(ops[*left].bottleneck_total_us),
            finite_non_negative(ops[*right].bottleneck_total_us),
        )
        .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0usize; ops.len()];
    for (rank, position) in order.into_iter().enumerate() {
        ranks[position] = rank + 1;
    }
    ranks
}

fn relative_delta(candidate: f64, baseline: f64) -> Option<f64> {
    (baseline > 0.0).then_some((candidate - baseline) / baseline)
}

fn finite_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

fn active_roofline_components(compute_us: f64, memory_us: f64) -> (f64, f64) {
    if compute_us >= memory_us {
        (compute_us, 0.0)
    } else {
        (0.0, memory_us)
    }
}

fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9_f64.max(right.abs() * 1e-10)
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn compare_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn roofline_bound_is_assigned_once() {
        assert_eq!(active_roofline_components(11.0, 5.0), (11.0, 0.0));
        assert_eq!(active_roofline_components(5.0, 11.0), (0.0, 11.0));
        assert_eq!(active_roofline_components(11.0, 11.0), (11.0, 0.0));
    }

    #[test]
    fn hirschberg_alignment_is_deterministic_and_monotonic() {
        let left = strings(&["A", "B", "C", "B", "D"]);
        let right = strings(&["A", "C", "B", "E", "D"]);
        let matches = hirschberg_matches(&left, &right);
        assert_eq!(matches, vec![(0, 0), (2, 1), (3, 2), (4, 4)]);
        assert!(matches
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0 && pair[0].1 < pair[1].1));
    }

    #[test]
    fn component_delta_conserves_named_terms() {
        let baseline = DeltaComponents {
            compute_us: 2.0,
            memory_us: 3.0,
            packing_us: 4.0,
            boundary_us: 5.0,
            fallback_us: 6.0,
            total_us: 20.0,
        };
        let candidate = DeltaComponents {
            compute_us: 3.0,
            memory_us: 5.0,
            packing_us: 4.0,
            boundary_us: 2.0,
            fallback_us: 1.0,
            total_us: 15.0,
        };
        let delta = DeltaComponents::signed_delta(candidate, baseline);
        assert_eq!(delta.total_us, -5.0);
        assert_eq!(
            delta.compute_us
                + delta.memory_us
                + delta.packing_us
                + delta.boundary_us
                + delta.fallback_us,
            delta.total_us
        );
    }

    #[test]
    fn relative_delta_is_withheld_for_zero_baseline() {
        assert_eq!(relative_delta(1.0, 0.0), None);
        assert_eq!(relative_delta(12.0, 10.0), Some(0.2));
    }
}
