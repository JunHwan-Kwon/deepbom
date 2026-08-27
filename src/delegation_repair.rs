use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use wasm_bindgen::prelude::*;

const REPAIR_SCHEMA: &str = "deepbom.delegation_repair.v1.3";
const REPAIR_METHOD_VERSION: &str = "2026-08-13.1";

#[derive(Clone, Copy, Default, Serialize)]
struct BoundarySummary {
    delegate_segment_count: usize,
    cpu_segment_count: usize,
    boundary_edge_count: usize,
    assessed_payload_edge_count: usize,
    unassessed_payload_edge_count: usize,
    assessed_edge_payload_bytes: usize,
    summed_edge_payload_bytes: Option<usize>,
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
struct EdgeKey {
    producer_op_index: usize,
    consumer_op_index: usize,
    tensor_index: usize,
}

#[derive(Clone)]
struct GraphEdge {
    key: EdgeKey,
    tensor_name: String,
    tensor_shape: Vec<i32>,
    tensor_dtype: String,
    payload_bytes: Option<usize>,
    payload_status: String,
    payload_binding: String,
    producer_position: usize,
    producer_op_name: String,
    consumer_position: usize,
    consumer_op_name: String,
}

#[derive(Clone)]
struct BoundaryEntry {
    edge: GraphEdge,
    direction: &'static str,
}

#[derive(Clone, Serialize)]
struct BoundaryEdgeChange {
    transition: &'static str,
    tensor_index: usize,
    tensor_name: String,
    tensor_shape: Vec<i32>,
    tensor_dtype: String,
    payload_bytes: Option<usize>,
    payload_status: String,
    payload_binding: String,
    producer_op_index: usize,
    producer_op_name: String,
    consumer_op_index: usize,
    consumer_op_name: String,
    baseline_direction: Option<&'static str>,
    counterfactual_direction: Option<&'static str>,
}

#[derive(Serialize)]
struct DelegationToggle {
    op_index: usize,
    op_name: String,
    baseline_assignment: &'static str,
    counterfactual_assignment: &'static str,
    counterfactual_class: &'static str,
    baseline_xnnpack_supported: bool,
    baseline_xnnpack_reason: String,
    baseline_chain_role: String,
    baseline_break_class: String,
    macs: f64,
    estimated_bytes: f64,
    dataflow_neighbor_count: usize,
    dataflow_incident_tensor_edge_count: usize,
    delegated_neighbor_count: usize,
    delegated_incident_tensor_edge_count: usize,
    distinct_neighbor_delegate_segment_count: usize,
    baseline_incident_boundary_edge_count: usize,
    baseline_incident_boundary_payload_bytes: Option<usize>,
    counterfactual: BoundarySummary,
    signed_delegate_segment_count: i64,
    signed_cpu_segment_count: i64,
    signed_boundary_edge_count: i64,
    signed_boundary_payload_bytes: Option<i64>,
    boundary_edge_reduction_count: i64,
    boundary_payload_reduction_bytes: Option<i64>,
    removed_boundary_edge_count: usize,
    added_boundary_edge_count: usize,
    reclassified_boundary_edge_count: usize,
    removed_boundary_payload_bytes: Option<usize>,
    added_boundary_payload_bytes: Option<usize>,
    reclassified_boundary_payload_bytes: Option<usize>,
    outcome_class: &'static str,
    repair_opportunity: bool,
    fragmentation_risk: bool,
    repair_rank: Option<usize>,
    fragility_rank: Option<usize>,
    edge_changes: Vec<BoundaryEdgeChange>,
}

#[derive(Serialize)]
struct CpuIslandIntervention {
    island_index: usize,
    execution_position_start: usize,
    execution_position_end: usize,
    first_op_index: usize,
    last_op_index: usize,
    op_indices: Vec<usize>,
    op_names: Vec<String>,
    op_count: usize,
    total_macs: f64,
    summed_estimated_bytes: f64,
    baseline_incident_boundary_edge_count: usize,
    baseline_incident_boundary_payload_bytes: Option<usize>,
    counterfactual: BoundarySummary,
    signed_delegate_segment_count: i64,
    signed_cpu_segment_count: i64,
    signed_boundary_edge_count: i64,
    signed_boundary_payload_bytes: Option<i64>,
    boundary_edge_reduction_count: i64,
    boundary_payload_reduction_bytes: Option<i64>,
    removed_boundary_edge_count: usize,
    added_boundary_edge_count: usize,
    reclassified_boundary_edge_count: usize,
    removed_boundary_payload_bytes: Option<usize>,
    added_boundary_payload_bytes: Option<usize>,
    member_single_repair_count: usize,
    best_single_op_index: Option<usize>,
    best_single_boundary_edge_reduction_count: i64,
    best_single_boundary_payload_reduction_bytes: Option<i64>,
    additional_edge_reduction_over_best_single: i64,
    additional_payload_reduction_over_best_single: Option<i64>,
    full_segment_repair: bool,
    group_only_repair: bool,
    outcome_class: &'static str,
    portfolio_rank: Option<usize>,
    edge_changes: Vec<BoundaryEdgeChange>,
}

#[derive(Serialize)]
struct ExportIntervention {
    id: &'static str,
    title: &'static str,
    evidence_class: &'static str,
    pattern: &'static str,
    block_ids: Vec<String>,
    block_count: usize,
    mean_op_indices: Vec<usize>,
    expand_dims_op_indices: Vec<usize>,
    downstream_fully_connected_op_indices: Vec<usize>,
    unmatched_rank4_mean_op_indices: Vec<usize>,
    unmatched_rank4_mean_reason: &'static str,
    derived_reduction_axes: Vec<usize>,
    assignment_toggle_op_indices: Vec<usize>,
    hypothesized_removed_op_count: usize,
    baseline: BoundarySummary,
    assignment_proxy: BoundarySummary,
    signed_delegate_segment_count: i64,
    signed_cpu_segment_count: i64,
    signed_boundary_edge_count: i64,
    signed_boundary_payload_bytes: Option<i64>,
    independent_single_toggle_signed_delegate_segment_count_sum: i64,
    independent_single_toggle_signed_cpu_segment_count_sum: i64,
    independent_single_toggle_signed_boundary_edge_count_sum: i64,
    independent_single_toggle_signed_boundary_payload_bytes_sum: Option<i64>,
    interaction_signed_delegate_segment_count: i64,
    interaction_signed_cpu_segment_count: i64,
    interaction_signed_boundary_edge_count: i64,
    interaction_signed_boundary_payload_bytes: Option<i64>,
    removed_boundary_edge_count: usize,
    added_boundary_edge_count: usize,
    removed_boundary_payload_bytes: Option<usize>,
    added_boundary_payload_bytes: Option<usize>,
    downstream_fully_connected_rule_status: &'static str,
    pinned_source_commit: &'static str,
    rule_source: &'static str,
    delegate_source: &'static str,
    action: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
    edge_changes: Vec<BoundaryEdgeChange>,
}

#[derive(Serialize)]
struct RuntimeBuildRisk {
    id: String,
    evidence_class: &'static str,
    required_build_configuration: String,
    configuration_binding_status: &'static str,
    baseline_conditionally_delegatable_op_count: usize,
    affected_conditionally_delegatable_op_count: usize,
    affected_predicted_delegate_segment_count: usize,
    affected_conditionally_delegatable_macs: f64,
    affected_conditionally_delegatable_mac_ratio: f64,
    absent_condition_remaining_conditionally_delegatable_op_count: usize,
    absent_condition_remaining_predicted_delegate_segment_count: usize,
    affected_op_indices: Vec<usize>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Serialize)]
struct SingletonDelegateSegment {
    segment_id: usize,
    op_index: usize,
    op_name: String,
    macs: f64,
    evidence_class: &'static str,
    interpretation: &'static str,
}

#[derive(Serialize)]
struct DelegationRepairResult {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    artifact_filename: String,
    artifact_sha256: String,
    format: String,
    target_id: String,
    target_label: String,
    target_profile_sha256: String,
    operator_count: usize,
    graph_edge_count: usize,
    baseline: BoundarySummary,
    repair_opportunity_count: usize,
    fragmentation_risk_count: usize,
    no_static_effect_count: usize,
    repair_ranking_op_indices: Vec<usize>,
    fragility_ranking_op_indices: Vec<usize>,
    toggles: Vec<DelegationToggle>,
    cpu_island_count: usize,
    full_segment_repair_count: usize,
    group_only_repair_count: usize,
    cpu_island_ranking_indices: Vec<usize>,
    cpu_islands: Vec<CpuIslandIntervention>,
    export_interventions: Vec<ExportIntervention>,
    runtime_build_risks: Vec<RuntimeBuildRisk>,
    singleton_delegate_segments: Vec<SingletonDelegateSegment>,
    ranking_basis: &'static str,
    island_ranking_basis: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[wasm_bindgen]
pub fn compute_delegation_repair(
    model_bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    if model_bytes.is_empty() {
        return Err(JsValue::from_str("TFLite model bytes are required."));
    }
    let analysis = analyze_with_target_without_step_response(model_bytes, filename, target_id)
        .map_err(|error| JsValue::from_str(&error))?;
    let result = build_delegation_repair(&analysis, hex_lower(&Sha256::digest(model_bytes)))
        .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&result).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn build_delegation_repair(
    analysis: &Analysis,
    artifact_sha256: String,
) -> Result<DelegationRepairResult, String> {
    if analysis.format != "tflite" {
        return Err("Delegation repair v1 supports TFLite artifacts only.".to_string());
    }
    if analysis.ops.is_empty() {
        return Err("Delegation repair requires at least one operator.".to_string());
    }
    let graph_edges = build_graph_edges(analysis);
    let baseline_assignments = analysis
        .ops
        .iter()
        .map(|op| op.xnnpack_chain_id >= 0)
        .collect::<Vec<_>>();
    let baseline_segments = assign_delegate_segments(&baseline_assignments);
    let baseline_boundaries =
        boundary_entries(&graph_edges, &baseline_assignments, &baseline_segments);
    let baseline = boundary_summary(
        &baseline_assignments,
        &baseline_segments,
        &baseline_boundaries,
    );
    verify_baseline_inventory(analysis, &baseline_boundaries, baseline)?;

    let neighbors = graph_neighbors(analysis.ops.len(), &graph_edges);
    let mut toggles = Vec::with_capacity(analysis.ops.len());
    for position in 0..analysis.ops.len() {
        let op = &analysis.ops[position];
        let baseline_delegated = baseline_assignments[position];
        let mut counterfactual_assignments = baseline_assignments.clone();
        counterfactual_assignments[position] = !baseline_delegated;
        let counterfactual_segments = assign_delegate_segments(&counterfactual_assignments);
        let counterfactual_boundaries = boundary_entries(
            &graph_edges,
            &counterfactual_assignments,
            &counterfactual_segments,
        );
        let counterfactual = boundary_summary(
            &counterfactual_assignments,
            &counterfactual_segments,
            &counterfactual_boundaries,
        );
        let edge_changes = edge_changes(&baseline_boundaries, &counterfactual_boundaries);
        let removed = edge_changes
            .iter()
            .filter(|edge| edge.transition == "removed")
            .cloned()
            .collect::<Vec<_>>();
        let added = edge_changes
            .iter()
            .filter(|edge| edge.transition == "added")
            .cloned()
            .collect::<Vec<_>>();
        let reclassified_boundary_edge_count = edge_changes
            .iter()
            .filter(|edge| edge.transition == "reclassified")
            .count();
        let signed_boundary_payload_bytes = signed_optional(
            counterfactual.summed_edge_payload_bytes,
            baseline.summed_edge_payload_bytes,
        );
        let boundary_payload_reduction_bytes = signed_boundary_payload_bytes.map(|value| -value);
        let signed_boundary_edge_count =
            counterfactual.boundary_edge_count as i64 - baseline.boundary_edge_count as i64;
        let signed_delegate_segment_count =
            counterfactual.delegate_segment_count as i64 - baseline.delegate_segment_count as i64;
        let repair_opportunity = !baseline_delegated
            && (signed_boundary_edge_count < 0
                || signed_delegate_segment_count < 0
                || boundary_payload_reduction_bytes.is_some_and(|value| value > 0));
        let fragmentation_risk = baseline_delegated
            && (signed_boundary_edge_count > 0
                || signed_delegate_segment_count > 0
                || signed_boundary_payload_bytes.is_some_and(|value| value > 0));
        let delegated_neighbors = neighbors[position]
            .iter()
            .filter(|neighbor| baseline_assignments[**neighbor])
            .copied()
            .collect::<Vec<_>>();
        let distinct_neighbor_delegate_segment_count = delegated_neighbors
            .iter()
            .filter_map(|neighbor| baseline_segments[*neighbor])
            .collect::<BTreeSet<_>>()
            .len();
        let incident_edges = graph_edges
            .iter()
            .filter(|edge| edge.producer_position == position || edge.consumer_position == position)
            .collect::<Vec<_>>();
        let delegated_incident_tensor_edge_count = incident_edges
            .iter()
            .filter(|edge| {
                let neighbor = if edge.producer_position == position {
                    edge.consumer_position
                } else {
                    edge.producer_position
                };
                baseline_assignments[neighbor]
            })
            .count();
        let baseline_incident_boundaries = baseline_boundaries
            .values()
            .filter(|boundary| {
                boundary.edge.producer_position == position
                    || boundary.edge.consumer_position == position
            })
            .collect::<Vec<_>>();
        let baseline_incident_boundary_payload_bytes = if baseline_incident_boundaries
            .iter()
            .all(|boundary| boundary.edge.payload_bytes.is_some())
        {
            Some(
                baseline_incident_boundaries
                    .iter()
                    .filter_map(|boundary| boundary.edge.payload_bytes)
                    .sum(),
            )
        } else {
            None
        };
        let reclassified = edge_changes
            .iter()
            .filter(|edge| edge.transition == "reclassified")
            .cloned()
            .collect::<Vec<_>>();
        toggles.push(DelegationToggle {
            op_index: op.index,
            op_name: op.name.clone(),
            baseline_assignment: assignment_label(baseline_delegated),
            counterfactual_assignment: assignment_label(!baseline_delegated),
            counterfactual_class: if baseline_delegated {
                "PREDICTED_SUPPORT_LOSS"
            } else {
                "HYPOTHETICAL_SUPPORT_EXTENSION"
            },
            baseline_xnnpack_supported: op.xnnpack_supported,
            baseline_xnnpack_reason: op.xnnpack_reason.clone(),
            baseline_chain_role: op.xnnpack_chain_role.clone(),
            baseline_break_class: op.xnnpack_break_class.clone(),
            macs: finite_non_negative(op.macs),
            estimated_bytes: finite_non_negative(op.estimated_bytes),
            dataflow_neighbor_count: neighbors[position].len(),
            dataflow_incident_tensor_edge_count: incident_edges.len(),
            delegated_neighbor_count: delegated_neighbors.len(),
            delegated_incident_tensor_edge_count,
            distinct_neighbor_delegate_segment_count,
            baseline_incident_boundary_edge_count: baseline_incident_boundaries.len(),
            baseline_incident_boundary_payload_bytes,
            counterfactual,
            signed_delegate_segment_count,
            signed_cpu_segment_count: counterfactual.cpu_segment_count as i64
                - baseline.cpu_segment_count as i64,
            signed_boundary_edge_count,
            signed_boundary_payload_bytes,
            boundary_edge_reduction_count: -signed_boundary_edge_count,
            boundary_payload_reduction_bytes,
            removed_boundary_edge_count: removed.len(),
            added_boundary_edge_count: added.len(),
            reclassified_boundary_edge_count,
            removed_boundary_payload_bytes: complete_payload_sum(&removed),
            added_boundary_payload_bytes: complete_payload_sum(&added),
            reclassified_boundary_payload_bytes: complete_payload_sum(&reclassified),
            outcome_class: outcome_class(
                baseline_delegated,
                signed_delegate_segment_count,
                signed_boundary_edge_count,
                signed_boundary_payload_bytes,
            ),
            repair_opportunity,
            fragmentation_risk,
            repair_rank: None,
            fragility_rank: None,
            edge_changes,
        });
    }

    let repair_ranking = ranked_positions(&toggles, false);
    let fragility_ranking = ranked_positions(&toggles, true);
    for (rank, position) in repair_ranking.iter().copied().enumerate() {
        toggles[position].repair_rank = Some(rank + 1);
    }
    for (rank, position) in fragility_ranking.iter().copied().enumerate() {
        toggles[position].fragility_rank = Some(rank + 1);
    }
    let repair_ranking_op_indices = repair_ranking
        .iter()
        .map(|position| toggles[*position].op_index)
        .collect::<Vec<_>>();
    let fragility_ranking_op_indices = fragility_ranking
        .iter()
        .map(|position| toggles[*position].op_index)
        .collect::<Vec<_>>();
    let repair_opportunity_count = toggles.iter().filter(|row| row.repair_opportunity).count();
    let fragmentation_risk_count = toggles.iter().filter(|row| row.fragmentation_risk).count();
    let no_static_effect_count = toggles
        .iter()
        .filter(|row| row.outcome_class == "no_static_boundary_effect")
        .count();
    let mut cpu_islands = build_cpu_island_interventions(
        analysis,
        &graph_edges,
        &baseline_assignments,
        &baseline_boundaries,
        baseline,
        &toggles,
    );
    let cpu_island_ranking_positions = ranked_cpu_island_positions(&cpu_islands);
    for (rank, position) in cpu_island_ranking_positions.iter().copied().enumerate() {
        cpu_islands[position].portfolio_rank = Some(rank + 1);
    }
    let cpu_island_ranking_indices = cpu_island_ranking_positions
        .iter()
        .map(|position| cpu_islands[*position].island_index)
        .collect::<Vec<_>>();
    let full_segment_repair_count = cpu_islands
        .iter()
        .filter(|island| island.full_segment_repair)
        .count();
    let group_only_repair_count = cpu_islands
        .iter()
        .filter(|island| island.group_only_repair)
        .count();
    let export_interventions = build_export_interventions(
        analysis,
        &graph_edges,
        &baseline_assignments,
        &baseline_segments,
        &baseline_boundaries,
        baseline,
    );
    let runtime_build_risks =
        build_runtime_build_risks(analysis, &baseline_assignments, &baseline_segments);
    let singleton_delegate_segments =
        build_singleton_delegate_segments(analysis, &baseline_segments);

    Ok(DelegationRepairResult {
        schema: REPAIR_SCHEMA,
        method_version: REPAIR_METHOD_VERSION,
        evidence_class: "PREDICTED_ASSIGNMENT_COUNTERFACTUAL_WITH_DERIVED_GRAPH_ARITHMETIC",
        artifact_filename: analysis.filename.clone(),
        artifact_sha256,
        format: analysis.format.clone(),
        target_id: analysis.target_profile.id.clone(),
        target_label: analysis.target_profile.label.clone(),
        target_profile_sha256: analysis.target_profile.profile_sha256.clone(),
        operator_count: analysis.ops.len(),
        graph_edge_count: graph_edges.len(),
        baseline,
        repair_opportunity_count,
        fragmentation_risk_count,
        no_static_effect_count,
        repair_ranking_op_indices,
        fragility_ranking_op_indices,
        toggles,
        cpu_island_count: cpu_islands.len(),
        full_segment_repair_count,
        group_only_repair_count,
        cpu_island_ranking_indices,
        cpu_islands,
        export_interventions,
        runtime_build_risks,
        singleton_delegate_segments,
        ranking_basis: "Repair rows: complete logical boundary-payload reduction descending, boundary-edge reduction descending, delegate-segment reduction descending, op index ascending. Fragility rows use the corresponding increases. Missing payload totals never outrank assessed payload totals.",
        island_ranking_basis: "CPU islands are maximal contiguous predicted-CPU runs in execution-plan order. Portfolio rows rank complete full-island logical boundary-payload reduction descending, boundary-edge reduction descending, delegate-segment reduction descending, member-op count ascending, then first op index ascending. Missing payload totals never outrank assessed totals.",
        method: "Build the non-constant producer-to-consumer tensor-edge graph and assign predicted delegate/CPU segments as contiguous runs in execution-plan order. First toggle exactly one op while all other assignments remain fixed. Then, for each maximal contiguous predicted-CPU run, toggle the complete run while all assignments outside that run remain fixed. Rebuild all segment IDs and boundary edges for every counterfactual and compare complete edge sets. Separately recognize source-export motifs and runtime-build requirements; these are emitted as explicitly conditional scenarios rather than mixed into single-op rankings.",
        interpretation_boundary: "These are static counterfactuals over PREDICTED XNNPACK eligibility, not proof that an unsupported op or CPU island can be implemented, delegated, fused, or made numerically correct. A full-island row is the exact consequence of that defined assignment toggle, not a claim about minimal implementation effort. Export interventions distinguish a graph-pattern diagnosis from a fixed-graph assignment proxy; the transformed artifact must be re-exported and re-audited. Runtime-build risks are conditional coverage-collapse scenarios, not observations of the deployed binary. Logical edge payload does not prove runtime copy materialization or latency. Actual delegate partitioning, graph rewrites, build flags, tensor layouts, executed kernels, and device behavior require bound runtime evidence.",
    })
}

fn build_export_interventions(
    analysis: &Analysis,
    graph_edges: &[GraphEdge],
    baseline_assignments: &[bool],
    baseline_segments: &[Option<usize>],
    baseline_boundaries: &BTreeMap<EdgeKey, BoundaryEntry>,
    baseline: BoundarySummary,
) -> Vec<ExportIntervention> {
    let position_by_index = analysis
        .ops
        .iter()
        .enumerate()
        .map(|(position, op)| (op.index, position))
        .collect::<HashMap<_, _>>();
    let mut block_ids = Vec::new();
    let mut mean_positions = Vec::new();
    let mut expand_positions = Vec::new();
    let mut fc_positions = Vec::new();

    for pattern in analysis
        .patterns
        .iter()
        .filter(|pattern| pattern.name == "SE block")
    {
        let Some(&start) = position_by_index.get(&pattern.first_op) else {
            continue;
        };
        let Some(sequence) = analysis.ops.get(start..start.saturating_add(6)) else {
            continue;
        };
        let names = sequence
            .iter()
            .map(|op| op.name.as_str())
            .collect::<Vec<_>>();
        if names
            != [
                "MEAN",
                "FULLY_CONNECTED",
                "FULLY_CONNECTED",
                "EXPAND_DIMS",
                "EXPAND_DIMS",
                "ADD",
            ]
            || sequence.last().map(|op| op.index) != Some(pattern.last_op)
            || ![1usize, 2, 5]
                .iter()
                .all(|offset| baseline_assignments.get(start + offset).copied() == Some(true))
            || ![0usize, 3, 4]
                .iter()
                .all(|offset| baseline_assignments.get(start + offset).copied() == Some(false))
        {
            continue;
        }
        let Some(mean_input) = sequence[0]
            .inputs
            .first()
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| analysis.tensors.get(index))
        else {
            continue;
        };
        let Some(mean_output) = sequence[0]
            .outputs
            .first()
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| analysis.tensors.get(index))
        else {
            continue;
        };
        let Some(first_expand_output) = sequence[3]
            .outputs
            .first()
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| analysis.tensors.get(index))
        else {
            continue;
        };
        let Some(second_expand_output) = sequence[4]
            .outputs
            .first()
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| analysis.tensors.get(index))
        else {
            continue;
        };
        let shape_contract = mean_input.shape.len() == 4
            && mean_output.shape.len() == 2
            && first_expand_output.shape.len() == 3
            && second_expand_output.shape.len() == 4
            && mean_input.shape.first() == mean_output.shape.first()
            && mean_input.shape.last() == mean_output.shape.last()
            && mean_output.shape.last() == second_expand_output.shape.last()
            && sequence[0]
                .xnnpack_reason
                .contains("primary_io_quant8_rank4");
        if !shape_contract {
            continue;
        }
        let block_id = analysis
            .block_inventory
            .blocks
            .iter()
            .find(|block| {
                block.block_type == "squeeze_excitation"
                    && block.op_indices.first() == Some(&pattern.first_op)
                    && block.op_indices.last() == Some(&pattern.last_op)
            })
            .map(|block| block.block_id.clone())
            .unwrap_or_else(|| format!("se_{}", pattern.first_op));
        block_ids.push(block_id);
        mean_positions.push(start);
        fc_positions.extend([start + 1, start + 2]);
        expand_positions.extend([start + 3, start + 4]);
    }
    if mean_positions.is_empty() || quantized_fully_connected_has_rank_predicate() {
        return Vec::new();
    }

    let mut toggle_positions = mean_positions
        .iter()
        .chain(expand_positions.iter())
        .copied()
        .collect::<Vec<_>>();
    toggle_positions.sort_unstable();
    toggle_positions.dedup();
    let matched_mean_positions = mean_positions.iter().copied().collect::<BTreeSet<_>>();
    let unmatched_rank4_mean_op_indices = analysis
        .ops
        .iter()
        .enumerate()
        .filter_map(|(position, op)| {
            if matched_mean_positions.contains(&position)
                || baseline_assignments.get(position).copied() != Some(false)
                || op.name != "MEAN"
                || !op.xnnpack_reason.contains("primary_io_quant8_rank4")
            {
                return None;
            }
            let input = op
                .inputs
                .first()
                .and_then(|index| usize::try_from(*index).ok())
                .and_then(|index| analysis.tensors.get(index))?;
            let output = op
                .outputs
                .first()
                .and_then(|index| usize::try_from(*index).ok())
                .and_then(|index| analysis.tensors.get(index))?;
            (input.shape.len() == 4 && output.shape.len() != 4).then_some(op.index)
        })
        .collect::<Vec<_>>();

    let mut independent_delegate_segment_sum = 0i64;
    let mut independent_cpu_segment_sum = 0i64;
    let mut independent_boundary_edge_sum = 0i64;
    let mut independent_boundary_payload_sum = Some(0i64);
    for position in &toggle_positions {
        let mut single_assignments = baseline_assignments.to_vec();
        single_assignments[*position] = true;
        let single_segments = assign_delegate_segments(&single_assignments);
        let single_boundaries =
            boundary_entries(graph_edges, &single_assignments, &single_segments);
        let single_summary =
            boundary_summary(&single_assignments, &single_segments, &single_boundaries);
        independent_delegate_segment_sum +=
            single_summary.delegate_segment_count as i64 - baseline.delegate_segment_count as i64;
        independent_cpu_segment_sum +=
            single_summary.cpu_segment_count as i64 - baseline.cpu_segment_count as i64;
        independent_boundary_edge_sum +=
            single_summary.boundary_edge_count as i64 - baseline.boundary_edge_count as i64;
        independent_boundary_payload_sum = match (
            independent_boundary_payload_sum,
            signed_optional(
                single_summary.summed_edge_payload_bytes,
                baseline.summed_edge_payload_bytes,
            ),
        ) {
            (Some(total), Some(delta)) => Some(total + delta),
            _ => None,
        };
    }
    let mut assignments = baseline_assignments.to_vec();
    for position in &toggle_positions {
        assignments[*position] = true;
    }
    let segments = assign_delegate_segments(&assignments);
    let boundaries = boundary_entries(graph_edges, &assignments, &segments);
    let assignment_proxy = boundary_summary(&assignments, &segments, &boundaries);
    let edge_changes = edge_changes(baseline_boundaries, &boundaries);
    let removed = edge_changes
        .iter()
        .filter(|edge| edge.transition == "removed")
        .cloned()
        .collect::<Vec<_>>();
    let added = edge_changes
        .iter()
        .filter(|edge| edge.transition == "added")
        .cloned()
        .collect::<Vec<_>>();
    let signed_boundary_payload_bytes = signed_optional(
        assignment_proxy.summed_edge_payload_bytes,
        baseline.summed_edge_payload_bytes,
    );
    let signed_delegate_segment_count =
        assignment_proxy.delegate_segment_count as i64 - baseline.delegate_segment_count as i64;
    let signed_cpu_segment_count =
        assignment_proxy.cpu_segment_count as i64 - baseline.cpu_segment_count as i64;
    let signed_boundary_edge_count =
        assignment_proxy.boundary_edge_count as i64 - baseline.boundary_edge_count as i64;
    let interaction_signed_boundary_payload_bytes = match (
        signed_boundary_payload_bytes,
        independent_boundary_payload_sum,
    ) {
        (Some(combined), Some(independent)) => Some(combined - independent),
        _ => None,
    };
    let baseline_segment_count = baseline_segments
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    debug_assert_eq!(baseline_segment_count, baseline.delegate_segment_count);

    vec![ExportIntervention {
        id: "se_global_pool_keepdims",
        title: "Preserve rank through squeeze-excitation global pooling",
        evidence_class: "DERIVED_PATTERN_WITH_PREDICTED_ASSIGNMENT_PROXY",
        pattern: "MEAN(rank4->rank2) -> FC -> FC -> EXPAND_DIMS -> EXPAND_DIMS -> ADD",
        block_count: block_ids.len(),
        block_ids,
        mean_op_indices: mean_positions
            .iter()
            .map(|position| analysis.ops[*position].index)
            .collect(),
        expand_dims_op_indices: expand_positions
            .iter()
            .map(|position| analysis.ops[*position].index)
            .collect(),
        downstream_fully_connected_op_indices: fc_positions
            .iter()
            .map(|position| analysis.ops[*position].index)
            .collect(),
        unmatched_rank4_mean_op_indices,
        unmatched_rank4_mean_reason:
            "rank4-input MEAN rejection outside the exact six-op squeeze-excitation motif; reported separately and not included in the keepdims intervention proxy",
        derived_reduction_axes: vec![1, 2],
        assignment_toggle_op_indices: toggle_positions
            .iter()
            .map(|position| analysis.ops[*position].index)
            .collect(),
        hypothesized_removed_op_count: expand_positions.len(),
        baseline,
        assignment_proxy,
        signed_delegate_segment_count,
        signed_cpu_segment_count,
        signed_boundary_edge_count,
        signed_boundary_payload_bytes,
        independent_single_toggle_signed_delegate_segment_count_sum:
            independent_delegate_segment_sum,
        independent_single_toggle_signed_cpu_segment_count_sum: independent_cpu_segment_sum,
        independent_single_toggle_signed_boundary_edge_count_sum: independent_boundary_edge_sum,
        independent_single_toggle_signed_boundary_payload_bytes_sum:
            independent_boundary_payload_sum,
        interaction_signed_delegate_segment_count: signed_delegate_segment_count
            - independent_delegate_segment_sum,
        interaction_signed_cpu_segment_count: signed_cpu_segment_count
            - independent_cpu_segment_sum,
        interaction_signed_boundary_edge_count: signed_boundary_edge_count
            - independent_boundary_edge_sum,
        interaction_signed_boundary_payload_bytes,
        removed_boundary_edge_count: removed.len(),
        added_boundary_edge_count: added.len(),
        removed_boundary_payload_bytes: complete_payload_sum(&removed),
        added_boundary_payload_bytes: complete_payload_sum(&added),
        downstream_fully_connected_rule_status:
            "no_input_rank_predicate_in_pinned_quantized_fully_connected_rule",
        pinned_source_commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c",
        rule_source: "reference/xnnpack-readme/rule-manifest-v2.21.0.json",
        delegate_source: "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc#VisitFullyConnectedNode",
        action: "Re-export GlobalAveragePooling2D/reduce_mean with keepdims=True, remove the two rank-restoring EXPAND_DIMS nodes per matched SE block, then re-run the audit on the transformed artifact.",
        method: "Recognize exact six-op squeeze-excitation motifs. Require rank4 MEAN input, rank2 MEAN output, rank3 then rank4 restoration, matching batch/channel dimensions, the pinned rank4 quantized-I/O rejection reason, delegated FC/ADD neighbors, and CPU-assigned MEAN/EXPAND_DIMS nodes. Force those existing nodes to delegate only to calculate an assignment proxy over the current graph. Also sum every matched node's isolated single-toggle delta and report the combined-minus-independent interaction term so the portfolio delta is arithmetically auditable.",
        interpretation_boundary: "The motif and rank transition are derived from the artifact. The fixed-graph assignment proxy quantifies partition arithmetic if the matched nodes become delegated; it does not prove that a keepdims export will preserve identical lowering, remove exactly those nodes, or pass runtime delegation. The pinned quantized FULLY_CONNECTED rule has no input-rank predicate, but the transformed model must be exported, re-audited, and runtime-validated.",
        edge_changes,
    }]
}

fn build_runtime_build_risks(
    analysis: &Analysis,
    baseline_assignments: &[bool],
    baseline_segments: &[Option<usize>],
) -> Vec<RuntimeBuildRisk> {
    let baseline_delegated_op_count = baseline_assignments.iter().filter(|value| **value).count();
    let total_macs = analysis
        .ops
        .iter()
        .map(|op| finite_non_negative(op.macs))
        .sum::<f64>();
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (position, op) in analysis.ops.iter().enumerate() {
        if baseline_assignments[position] && !op.xnnpack_build_requirement.trim().is_empty() {
            groups
                .entry(op.xnnpack_build_requirement.clone())
                .or_default()
                .push(position);
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(group_index, (requirement, positions))| {
            let affected_segments = positions
                .iter()
                .filter_map(|position| baseline_segments[*position])
                .collect::<BTreeSet<_>>();
            let affected_macs = positions
                .iter()
                .map(|position| finite_non_negative(analysis.ops[*position].macs))
                .sum::<f64>();
            let mut remaining = baseline_assignments.to_vec();
            for position in &positions {
                remaining[*position] = false;
            }
            let remaining_segments = assign_delegate_segments(&remaining);
            RuntimeBuildRisk {
                id: format!("xnnpack_required_build_configuration_unbound_{group_index}"),
                evidence_class: "CONDITIONAL_SOURCE_BACKED_CONFIGURATION_SCENARIO",
                required_build_configuration: requirement,
                configuration_binding_status: "not_embedded_in_model_artifact",
                baseline_conditionally_delegatable_op_count: baseline_delegated_op_count,
                affected_conditionally_delegatable_op_count: positions.len(),
                affected_predicted_delegate_segment_count: affected_segments.len(),
                affected_conditionally_delegatable_macs: affected_macs,
                affected_conditionally_delegatable_mac_ratio: if total_macs > 0.0 {
                    affected_macs / total_macs
                } else {
                    0.0
                },
                absent_condition_remaining_conditionally_delegatable_op_count: remaining
                    .iter()
                    .filter(|value| **value)
                    .count(),
                absent_condition_remaining_predicted_delegate_segment_count: remaining_segments
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len(),
                affected_op_indices: positions
                    .iter()
                    .map(|position| analysis.ops[*position].index)
                    .collect(),
                method: "Group conditionally delegatable operators by their emitted pinned-source build requirement, then remove the entire group from the predicted assignment to expose the conditional coverage collapse.",
                interpretation_boundary: "The model artifact does not identify the deployed runtime binary or its compile flags. This row states what the pinned rulepack requires and the assignment consequence if that requirement is absent; it does not observe that the deployed runtime lacks the flag.",
            }
        })
        .collect()
}

fn build_singleton_delegate_segments(
    analysis: &Analysis,
    segments: &[Option<usize>],
) -> Vec<SingletonDelegateSegment> {
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for (position, segment) in segments.iter().enumerate() {
        if let Some(segment) = segment {
            members.entry(*segment).or_default().push(position);
        }
    }
    members
        .into_iter()
        .filter_map(|(segment_id, positions)| {
            (positions.len() == 1).then(|| {
                let op = &analysis.ops[positions[0]];
                SingletonDelegateSegment {
                    segment_id,
                    op_index: op.index,
                    op_name: op.name.clone(),
                    macs: finite_non_negative(op.macs),
                    evidence_class: "DERIVED_ASSIGNMENT_STRUCTURE",
                    interpretation: "A one-op predicted delegate segment may not amortize delegate setup or transition overhead; profitability requires runtime measurement.",
                }
            })
        })
        .collect()
}

fn build_cpu_island_interventions(
    analysis: &Analysis,
    graph_edges: &[GraphEdge],
    baseline_assignments: &[bool],
    baseline_boundaries: &BTreeMap<EdgeKey, BoundaryEntry>,
    baseline: BoundarySummary,
    toggles: &[DelegationToggle],
) -> Vec<CpuIslandIntervention> {
    cpu_island_ranges(baseline_assignments)
        .into_iter()
        .enumerate()
        .map(|(island_position, (start, end))| {
            let mut assignments = baseline_assignments.to_vec();
            assignments[start..=end].fill(true);
            let segments = assign_delegate_segments(&assignments);
            let boundaries = boundary_entries(graph_edges, &assignments, &segments);
            let counterfactual = boundary_summary(&assignments, &segments, &boundaries);
            let edge_changes = edge_changes(baseline_boundaries, &boundaries);
            let removed = edge_changes
                .iter()
                .filter(|edge| edge.transition == "removed")
                .cloned()
                .collect::<Vec<_>>();
            let added = edge_changes
                .iter()
                .filter(|edge| edge.transition == "added")
                .cloned()
                .collect::<Vec<_>>();
            let reclassified_boundary_edge_count = edge_changes
                .iter()
                .filter(|edge| edge.transition == "reclassified")
                .count();
            let signed_delegate_segment_count = counterfactual.delegate_segment_count as i64
                - baseline.delegate_segment_count as i64;
            let signed_cpu_segment_count =
                counterfactual.cpu_segment_count as i64 - baseline.cpu_segment_count as i64;
            let signed_boundary_edge_count =
                counterfactual.boundary_edge_count as i64 - baseline.boundary_edge_count as i64;
            let signed_boundary_payload_bytes = signed_optional(
                counterfactual.summed_edge_payload_bytes,
                baseline.summed_edge_payload_bytes,
            );
            let boundary_edge_reduction_count = -signed_boundary_edge_count;
            let boundary_payload_reduction_bytes =
                signed_boundary_payload_bytes.map(|value| -value);
            let member_toggles = &toggles[start..=end];
            let mut repair_members = member_toggles
                .iter()
                .filter(|row| row.repair_opportunity)
                .collect::<Vec<_>>();
            repair_members.sort_by(|left, right| {
                compare_optional_metric(
                    left.boundary_payload_reduction_bytes,
                    right.boundary_payload_reduction_bytes,
                )
                .then_with(|| {
                    compare_desc_i64(
                        left.boundary_edge_reduction_count,
                        right.boundary_edge_reduction_count,
                    )
                })
                .then_with(|| {
                    compare_desc_i64(
                        -left.signed_delegate_segment_count,
                        -right.signed_delegate_segment_count,
                    )
                })
                .then_with(|| left.op_index.cmp(&right.op_index))
            });
            let best_single = repair_members.first().copied();
            let best_single_boundary_edge_reduction_count = best_single
                .map(|row| row.boundary_edge_reduction_count)
                .unwrap_or(0);
            let best_single_boundary_payload_reduction_bytes = best_single
                .map(|row| row.boundary_payload_reduction_bytes)
                .unwrap_or(Some(0));
            let full_segment_repair = boundary_edge_reduction_count > 0
                || signed_delegate_segment_count < 0
                || boundary_payload_reduction_bytes.is_some_and(|value| value > 0);
            let incident_payloads = baseline_boundaries
                .values()
                .filter(|entry| {
                    (start..=end).contains(&entry.edge.producer_position)
                        || (start..=end).contains(&entry.edge.consumer_position)
                })
                .map(|entry| entry.edge.payload_bytes)
                .collect::<Option<Vec<_>>>();
            let op_slice = &analysis.ops[start..=end];
            CpuIslandIntervention {
                island_index: island_position + 1,
                execution_position_start: start,
                execution_position_end: end,
                first_op_index: op_slice.first().map(|op| op.index).unwrap_or(0),
                last_op_index: op_slice.last().map(|op| op.index).unwrap_or(0),
                op_indices: op_slice.iter().map(|op| op.index).collect(),
                op_names: op_slice.iter().map(|op| op.name.clone()).collect(),
                op_count: op_slice.len(),
                total_macs: op_slice.iter().map(|op| finite_non_negative(op.macs)).sum(),
                summed_estimated_bytes: op_slice
                    .iter()
                    .map(|op| finite_non_negative(op.estimated_bytes))
                    .sum(),
                baseline_incident_boundary_edge_count: baseline_boundaries
                    .values()
                    .filter(|entry| {
                        (start..=end).contains(&entry.edge.producer_position)
                            || (start..=end).contains(&entry.edge.consumer_position)
                    })
                    .count(),
                baseline_incident_boundary_payload_bytes: incident_payloads
                    .map(|values| values.into_iter().sum()),
                counterfactual,
                signed_delegate_segment_count,
                signed_cpu_segment_count,
                signed_boundary_edge_count,
                signed_boundary_payload_bytes,
                boundary_edge_reduction_count,
                boundary_payload_reduction_bytes,
                removed_boundary_edge_count: removed.len(),
                added_boundary_edge_count: added.len(),
                reclassified_boundary_edge_count,
                removed_boundary_payload_bytes: complete_payload_sum(&removed),
                added_boundary_payload_bytes: complete_payload_sum(&added),
                member_single_repair_count: repair_members.len(),
                best_single_op_index: best_single.map(|row| row.op_index),
                best_single_boundary_edge_reduction_count,
                best_single_boundary_payload_reduction_bytes,
                additional_edge_reduction_over_best_single: boundary_edge_reduction_count
                    - best_single_boundary_edge_reduction_count,
                additional_payload_reduction_over_best_single: match (
                    boundary_payload_reduction_bytes,
                    best_single_boundary_payload_reduction_bytes,
                ) {
                    (Some(joint), Some(single)) => Some(joint - single),
                    _ => None,
                },
                full_segment_repair,
                group_only_repair: full_segment_repair && repair_members.is_empty(),
                outcome_class: cpu_island_outcome_class(
                    signed_delegate_segment_count,
                    signed_boundary_edge_count,
                    signed_boundary_payload_bytes,
                ),
                portfolio_rank: None,
                edge_changes,
            }
        })
        .collect()
}

fn build_graph_edges(analysis: &Analysis) -> Vec<GraphEdge> {
    let mut producer_by_tensor = HashMap::<i32, usize>::new();
    for (position, op) in analysis.ops.iter().enumerate() {
        for &tensor_index in &op.outputs {
            if tensor_index >= 0 {
                producer_by_tensor.insert(tensor_index, position);
            }
        }
    }
    let mut edges = Vec::new();
    for (consumer_position, consumer) in analysis.ops.iter().enumerate() {
        let mut seen = HashSet::new();
        for &tensor_index in &consumer.inputs {
            if tensor_index < 0 || !seen.insert(tensor_index) {
                continue;
            }
            let Some(&producer_position) = producer_by_tensor.get(&tensor_index) else {
                continue;
            };
            let Some(tensor) = analysis.tensors.get(tensor_index as usize) else {
                continue;
            };
            if tensor.constant_buffer {
                continue;
            }
            let payload = deterministic_tensor_payload_assessment(tensor);
            edges.push(GraphEdge {
                key: EdgeKey {
                    producer_op_index: analysis.ops[producer_position].index,
                    consumer_op_index: consumer.index,
                    tensor_index: tensor.index,
                },
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                payload_bytes: payload.bytes,
                payload_status: payload.status.to_string(),
                payload_binding: payload.binding.to_string(),
                producer_position,
                producer_op_name: analysis.ops[producer_position].name.clone(),
                consumer_position,
                consumer_op_name: consumer.name.clone(),
            });
        }
    }
    edges.sort_by(|left, right| left.key.cmp(&right.key));
    edges
}

fn assign_delegate_segments(assignments: &[bool]) -> Vec<Option<usize>> {
    let mut segment_ids = vec![None; assignments.len()];
    let mut next_segment = 0usize;
    let mut active_segment = None;
    for (position, delegated) in assignments.iter().copied().enumerate() {
        if delegated {
            let segment = *active_segment.get_or_insert_with(|| {
                let value = next_segment;
                next_segment += 1;
                value
            });
            segment_ids[position] = Some(segment);
        } else {
            active_segment = None;
        }
    }
    segment_ids
}

fn boundary_entries(
    edges: &[GraphEdge],
    assignments: &[bool],
    segments: &[Option<usize>],
) -> BTreeMap<EdgeKey, BoundaryEntry> {
    let mut result = BTreeMap::new();
    for edge in edges {
        let producer_delegated = assignments[edge.producer_position];
        let consumer_delegated = assignments[edge.consumer_position];
        let boundary = match (producer_delegated, consumer_delegated) {
            (true, true) => segments[edge.producer_position] != segments[edge.consumer_position],
            (false, false) => false,
            _ => true,
        };
        if boundary {
            result.insert(
                edge.key.clone(),
                BoundaryEntry {
                    edge: edge.clone(),
                    direction: boundary_direction(producer_delegated, consumer_delegated),
                },
            );
        }
    }
    result
}

fn boundary_summary(
    assignments: &[bool],
    segments: &[Option<usize>],
    boundaries: &BTreeMap<EdgeKey, BoundaryEntry>,
) -> BoundarySummary {
    let assessed_payload_edge_count = boundaries
        .values()
        .filter(|entry| entry.edge.payload_bytes.is_some())
        .count();
    let unassessed_payload_edge_count =
        boundaries.len().saturating_sub(assessed_payload_edge_count);
    let assessed_edge_payload_bytes = boundaries
        .values()
        .filter_map(|entry| entry.edge.payload_bytes)
        .sum();
    BoundarySummary {
        delegate_segment_count: segments
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        cpu_segment_count: contiguous_segment_count(assignments, false),
        boundary_edge_count: boundaries.len(),
        assessed_payload_edge_count,
        unassessed_payload_edge_count,
        assessed_edge_payload_bytes,
        summed_edge_payload_bytes: (unassessed_payload_edge_count == 0)
            .then_some(assessed_edge_payload_bytes),
    }
}

fn verify_baseline_inventory(
    analysis: &Analysis,
    boundaries: &BTreeMap<EdgeKey, BoundaryEntry>,
    summary: BoundarySummary,
) -> Result<(), String> {
    let inventory = &analysis.predicted_partition_boundaries;
    if summary.boundary_edge_count != inventory.edge_count
        || summary.assessed_payload_edge_count != inventory.assessed_payload_edge_count
        || summary.unassessed_payload_edge_count != inventory.unassessed_payload_edge_count
        || summary.assessed_edge_payload_bytes != inventory.assessed_edge_payload_bytes
        || summary.summed_edge_payload_bytes != inventory.summed_edge_payload_bytes
    {
        return Err(
            "Delegation repair baseline does not match the canonical predicted boundary inventory."
                .to_string(),
        );
    }
    let canonical = inventory
        .edges
        .iter()
        .map(|edge| EdgeKey {
            producer_op_index: edge.producer_op_index,
            consumer_op_index: edge.consumer_op_index,
            tensor_index: edge.tensor_index,
        })
        .collect::<BTreeSet<_>>();
    let rebuilt = boundaries.keys().cloned().collect::<BTreeSet<_>>();
    if canonical != rebuilt {
        return Err(
            "Delegation repair baseline boundary-edge identities differ from canonical evidence."
                .to_string(),
        );
    }
    Ok(())
}

fn graph_neighbors(op_count: usize, edges: &[GraphEdge]) -> Vec<BTreeSet<usize>> {
    let mut neighbors = vec![BTreeSet::new(); op_count];
    for edge in edges {
        neighbors[edge.producer_position].insert(edge.consumer_position);
        neighbors[edge.consumer_position].insert(edge.producer_position);
    }
    neighbors
}

fn edge_changes(
    baseline: &BTreeMap<EdgeKey, BoundaryEntry>,
    counterfactual: &BTreeMap<EdgeKey, BoundaryEntry>,
) -> Vec<BoundaryEdgeChange> {
    let keys = baseline
        .keys()
        .chain(counterfactual.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut changes = Vec::new();
    for key in keys {
        match (baseline.get(&key), counterfactual.get(&key)) {
            (Some(left), None) => changes.push(edge_change("removed", left, None)),
            (None, Some(right)) => changes.push(edge_change("added", right, None)),
            (Some(left), Some(right)) if left.direction != right.direction => {
                changes.push(edge_change("reclassified", left, Some(right.direction)));
            }
            _ => {}
        }
    }
    changes.sort_by(|left, right| {
        transition_rank(left.transition)
            .cmp(&transition_rank(right.transition))
            .then_with(|| left.producer_op_index.cmp(&right.producer_op_index))
            .then_with(|| left.consumer_op_index.cmp(&right.consumer_op_index))
            .then_with(|| left.tensor_index.cmp(&right.tensor_index))
    });
    changes
}

fn edge_change(
    transition: &'static str,
    entry: &BoundaryEntry,
    counterfactual_direction: Option<&'static str>,
) -> BoundaryEdgeChange {
    BoundaryEdgeChange {
        transition,
        tensor_index: entry.edge.key.tensor_index,
        tensor_name: entry.edge.tensor_name.clone(),
        tensor_shape: entry.edge.tensor_shape.clone(),
        tensor_dtype: entry.edge.tensor_dtype.clone(),
        payload_bytes: entry.edge.payload_bytes,
        payload_status: entry.edge.payload_status.clone(),
        payload_binding: entry.edge.payload_binding.clone(),
        producer_op_index: entry.edge.key.producer_op_index,
        producer_op_name: entry.edge.producer_op_name.clone(),
        consumer_op_index: entry.edge.key.consumer_op_index,
        consumer_op_name: entry.edge.consumer_op_name.clone(),
        baseline_direction: (transition != "added").then_some(entry.direction),
        counterfactual_direction: if transition == "added" {
            Some(entry.direction)
        } else {
            counterfactual_direction
        },
    }
}

fn ranked_positions(rows: &[DelegationToggle], fragility: bool) -> Vec<usize> {
    let mut positions = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            if fragility {
                row.fragmentation_risk
            } else {
                row.repair_opportunity
            }
        })
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    positions.sort_by(|left, right| {
        let left_row = &rows[*left];
        let right_row = &rows[*right];
        compare_optional_metric(
            if fragility {
                left_row.signed_boundary_payload_bytes
            } else {
                left_row.boundary_payload_reduction_bytes
            },
            if fragility {
                right_row.signed_boundary_payload_bytes
            } else {
                right_row.boundary_payload_reduction_bytes
            },
        )
        .then_with(|| {
            compare_desc_i64(
                if fragility {
                    left_row.signed_boundary_edge_count
                } else {
                    left_row.boundary_edge_reduction_count
                },
                if fragility {
                    right_row.signed_boundary_edge_count
                } else {
                    right_row.boundary_edge_reduction_count
                },
            )
        })
        .then_with(|| {
            compare_desc_i64(
                if fragility {
                    left_row.signed_delegate_segment_count
                } else {
                    -left_row.signed_delegate_segment_count
                },
                if fragility {
                    right_row.signed_delegate_segment_count
                } else {
                    -right_row.signed_delegate_segment_count
                },
            )
        })
        .then_with(|| left_row.op_index.cmp(&right_row.op_index))
    });
    positions
}

fn cpu_island_ranges(assignments: &[bool]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (position, delegated) in assignments.iter().copied().enumerate() {
        match (start, delegated) {
            (None, false) => start = Some(position),
            (Some(value), true) => {
                ranges.push((value, position - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(value) = start {
        ranges.push((value, assignments.len().saturating_sub(1)));
    }
    ranges
}

fn ranked_cpu_island_positions(rows: &[CpuIslandIntervention]) -> Vec<usize> {
    let mut positions = (0..rows.len()).collect::<Vec<_>>();
    positions.sort_by(|left, right| {
        let left_row = &rows[*left];
        let right_row = &rows[*right];
        compare_optional_metric(
            left_row.boundary_payload_reduction_bytes,
            right_row.boundary_payload_reduction_bytes,
        )
        .then_with(|| {
            compare_desc_i64(
                left_row.boundary_edge_reduction_count,
                right_row.boundary_edge_reduction_count,
            )
        })
        .then_with(|| {
            compare_desc_i64(
                -left_row.signed_delegate_segment_count,
                -right_row.signed_delegate_segment_count,
            )
        })
        .then_with(|| left_row.op_count.cmp(&right_row.op_count))
        .then_with(|| left_row.first_op_index.cmp(&right_row.first_op_index))
    });
    positions
}

fn cpu_island_outcome_class(
    signed_delegate_segments: i64,
    signed_boundary_edges: i64,
    signed_payload: Option<i64>,
) -> &'static str {
    if signed_delegate_segments < 0
        && (signed_boundary_edges < 0 || signed_payload.is_some_and(|value| value < 0))
    {
        "eliminates_cpu_island_and_merges_delegate_segments"
    } else if signed_boundary_edges < 0 || signed_payload.is_some_and(|value| value < 0) {
        "eliminates_cpu_island_and_reduces_boundaries"
    } else if signed_boundary_edges > 0 || signed_payload.is_some_and(|value| value > 0) {
        "eliminates_cpu_island_but_increases_boundaries"
    } else if signed_delegate_segments < 0 {
        "eliminates_cpu_island_and_merges_execution_segments"
    } else {
        "eliminates_cpu_island_without_boundary_reduction"
    }
}

fn outcome_class(
    baseline_delegated: bool,
    signed_delegate_segments: i64,
    signed_boundary_edges: i64,
    signed_payload: Option<i64>,
) -> &'static str {
    if !baseline_delegated {
        if signed_delegate_segments < 0 {
            "bridge_merges_delegate_segments"
        } else if signed_delegate_segments > 0 {
            "creates_delegate_island"
        } else if signed_boundary_edges < 0 || signed_payload.is_some_and(|value| value < 0) {
            "extends_delegate_coverage"
        } else if signed_boundary_edges > 0 || signed_payload.is_some_and(|value| value > 0) {
            "support_extension_increases_boundaries"
        } else {
            "no_static_boundary_effect"
        }
    } else if signed_delegate_segments > 0 {
        "splits_delegate_segment"
    } else if signed_delegate_segments < 0 {
        "removes_singleton_delegate_segment"
    } else if signed_boundary_edges > 0 || signed_payload.is_some_and(|value| value > 0) {
        "support_loss_increases_boundaries"
    } else if signed_boundary_edges < 0 || signed_payload.is_some_and(|value| value < 0) {
        "support_loss_reduces_boundaries"
    } else {
        "no_static_boundary_effect"
    }
}

fn complete_payload_sum(edges: &[BoundaryEdgeChange]) -> Option<usize> {
    edges
        .iter()
        .map(|edge| edge.payload_bytes)
        .collect::<Option<Vec<_>>>()
        .map(|values| values.into_iter().sum())
}

fn signed_optional(candidate: Option<usize>, baseline: Option<usize>) -> Option<i64> {
    Some(candidate? as i64 - baseline? as i64)
}

fn contiguous_segment_count(assignments: &[bool], value: bool) -> usize {
    assignments
        .iter()
        .copied()
        .enumerate()
        .filter(|(position, item)| {
            *item == value && (*position == 0 || assignments[*position - 1] != value)
        })
        .count()
}

fn assignment_label(delegated: bool) -> &'static str {
    if delegated {
        "predicted_delegate"
    } else {
        "predicted_cpu"
    }
}

fn boundary_direction(producer_delegated: bool, consumer_delegated: bool) -> &'static str {
    match (producer_delegated, consumer_delegated) {
        (true, false) => "delegate_to_cpu",
        (false, true) => "cpu_to_delegate",
        (true, true) => "delegate_partition_to_delegate_partition",
        (false, false) => "cpu_to_cpu",
    }
}

fn transition_rank(value: &str) -> usize {
    match value {
        "removed" => 0,
        "added" => 1,
        _ => 2,
    }
}

fn compare_optional_metric(left: Option<i64>, right: Option<i64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => compare_desc_i64(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_desc_i64(left: i64, right: i64) -> Ordering {
    right.cmp(&left)
}

fn finite_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_assignment_is_contiguous_and_stable() {
        let assignments = vec![true, true, false, true, false, false, true];
        assert_eq!(
            assign_delegate_segments(&assignments),
            vec![Some(0), Some(0), None, Some(1), None, None, Some(2)]
        );
        assert_eq!(contiguous_segment_count(&assignments, true), 3);
        assert_eq!(contiguous_segment_count(&assignments, false), 2);
    }

    #[test]
    fn bridge_and_split_outcomes_are_explicit() {
        assert_eq!(
            outcome_class(false, -1, -2, Some(-8008)),
            "bridge_merges_delegate_segments"
        );
        assert_eq!(
            outcome_class(true, 1, 2, Some(8008)),
            "splits_delegate_segment"
        );
        assert_eq!(
            outcome_class(false, 1, 0, Some(0)),
            "creates_delegate_island"
        );
    }

    #[test]
    fn missing_payload_never_outranks_assessed_payload() {
        assert_eq!(compare_optional_metric(Some(0), None), Ordering::Less);
        assert_eq!(compare_optional_metric(None, Some(0)), Ordering::Greater);
        assert_eq!(compare_optional_metric(Some(4), Some(2)), Ordering::Less);
    }

    #[test]
    fn complete_cpu_island_detects_group_only_boundary_repair() {
        let assignments = vec![true, false, false, true];
        assert_eq!(cpu_island_ranges(&assignments), vec![(1, 2)]);
        let edges = vec![
            test_edge(0, 1, 10),
            test_edge(1, 2, 11),
            test_edge(2, 3, 12),
        ];
        let baseline_segments = assign_delegate_segments(&assignments);
        let baseline_boundaries = boundary_entries(&edges, &assignments, &baseline_segments);
        let baseline = boundary_summary(&assignments, &baseline_segments, &baseline_boundaries);
        assert_eq!(baseline.delegate_segment_count, 2);
        assert_eq!(baseline.cpu_segment_count, 1);
        assert_eq!(baseline.boundary_edge_count, 2);
        assert_eq!(baseline.summed_edge_payload_bytes, Some(8));

        for position in [1usize, 2usize] {
            let mut single = assignments.clone();
            single[position] = true;
            let segments = assign_delegate_segments(&single);
            let boundaries = boundary_entries(&edges, &single, &segments);
            let summary = boundary_summary(&single, &segments, &boundaries);
            assert_eq!(summary.delegate_segment_count, 2);
            assert_eq!(summary.boundary_edge_count, 2);
            assert_eq!(summary.summed_edge_payload_bytes, Some(8));
        }

        let joint = vec![true, true, true, true];
        let joint_segments = assign_delegate_segments(&joint);
        let joint_boundaries = boundary_entries(&edges, &joint, &joint_segments);
        let joint_summary = boundary_summary(&joint, &joint_segments, &joint_boundaries);
        assert_eq!(joint_summary.delegate_segment_count, 1);
        assert_eq!(joint_summary.cpu_segment_count, 0);
        assert_eq!(joint_summary.boundary_edge_count, 0);
        assert_eq!(joint_summary.summed_edge_payload_bytes, Some(0));
        assert_eq!(
            cpu_island_outcome_class(-1, -2, Some(-8)),
            "eliminates_cpu_island_and_merges_delegate_segments"
        );
    }

    fn test_edge(producer: usize, consumer: usize, tensor: usize) -> GraphEdge {
        GraphEdge {
            key: EdgeKey {
                producer_op_index: producer,
                consumer_op_index: consumer,
                tensor_index: tensor,
            },
            tensor_name: format!("T{tensor}"),
            tensor_shape: vec![1],
            tensor_dtype: "FLOAT32".to_string(),
            payload_bytes: Some(4),
            payload_status: "assessed_static".to_string(),
            payload_binding: "static".to_string(),
            producer_position: producer,
            producer_op_name: format!("OP_{producer}"),
            consumer_position: consumer,
            consumer_op_name: format!("OP_{consumer}"),
        }
    }
}
