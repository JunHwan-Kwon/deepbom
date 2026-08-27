use super::accumulator_reachability::{
    AccumulatorReachabilityAnalysis, AccumulatorReachabilityOpRow,
};
use super::rounding_equivalence::{RoundingEquivalenceAnalysis, RoundingEquivalenceOpRow};
use super::{
    deterministic_tensor_payload_bytes, predicted_boundary_direction, predicted_execution_domain,
    OpInfo, TensorInfo,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

const SCHEMA: &str = "deepbom.numerical_abi_propagation.v1.1";
const METHOD_VERSION: &str = "2026-07-28.1";
const GRAPH_LEDGER_PREFIX: &[u8] = b"deepbom.numerical_abi_propagation.graph.v1\0";
const SOURCE_LEDGER_PREFIX: &[u8] = b"deepbom.numerical_abi_propagation.source.v1.1\0";

#[derive(Clone, Serialize)]
struct PropagationGraphEdge {
    edge_index: usize,
    producer_op_index: usize,
    producer_op_name: String,
    tensor_index: usize,
    tensor_name: String,
    tensor_shape: Vec<i32>,
    tensor_dtype: String,
    logical_payload_bytes: Option<usize>,
    payload_status: &'static str,
    consumer_op_index: usize,
    consumer_op_name: String,
    consumer_input_slots: Vec<usize>,
    producer_domain: String,
    consumer_domain: String,
    predicted_boundary: bool,
    predicted_boundary_direction: String,
}

#[derive(Clone, Serialize)]
struct MergePoint {
    op_index: usize,
    op_name: String,
    minimum_op_hops: usize,
    merge_class: &'static str,
    graph_input_tensor_count: usize,
    influenced_input_tensor_indices: Vec<usize>,
    uninfluenced_input_tensor_indices: Vec<usize>,
    predicted_execution_domain: String,
}

#[derive(Clone, Serialize)]
struct ModelOutputPath {
    output_tensor_index: usize,
    output_tensor_name: String,
    shortest_op_hops: usize,
    shortest_path_op_indices: Vec<usize>,
    shortest_path_edge_indices: Vec<usize>,
    shortest_path_predicted_boundary_count: usize,
    shortest_path_boundary_logical_payload_bytes: Option<usize>,
    exact_graph_route_count_decimal: Option<String>,
    route_count_status: &'static str,
}

#[derive(Clone, Serialize)]
pub(super) struct PropagationSource {
    pub(super) op_index: usize,
    pub(super) op_name: String,
    pub(super) assessment_status: &'static str,
    not_assessed_reason: String,
    source_output_tensor_indices: Vec<usize>,
    assessed_channel_count: usize,
    divergent_channel_count: usize,
    interval_state_count_decimal: String,
    divergent_state_count_decimal: String,
    divergent_state_ratio: f64,
    maximum_absolute_output_delta: Option<i64>,
    source_equivalence_ledger_sha256: String,
    local_reachability_status: &'static str,
    exact_reachable_divergent_channel_count: usize,
    unresolved_divergent_channel_count: usize,
    interval_only_divergent_channel_count: usize,
    exact_reachable_divergent_state_count_decimal: String,
    provably_unreachable_divergent_state_count_decimal: String,
    unresolved_divergent_state_count_decimal: String,
    source_reachability_ledger_sha256: String,
    direct_consumer_edge_count: usize,
    corridor_edge_count: usize,
    corridor_edge_indices: Vec<usize>,
    reachable_op_count: usize,
    reachable_op_indices: Vec<usize>,
    reachable_tensor_count: usize,
    reachable_tensor_indices: Vec<usize>,
    pub(super) reachable_model_output_tensor_count: usize,
    minimum_model_output_op_hops: Option<usize>,
    maximum_reachable_op_hops: Option<usize>,
    pub(super) exact_model_output_graph_route_count_decimal: Option<String>,
    route_count_status: &'static str,
    predicted_boundary_edge_count: usize,
    assessed_boundary_logical_payload_bytes: usize,
    unassessed_boundary_payload_edge_count: usize,
    reconvergence_op_count: usize,
    single_branch_merge_op_count: usize,
    merge_points: Vec<MergePoint>,
    model_output_paths: Vec<ModelOutputPath>,
    pub(super) propagation_ledger_sha256: String,
}

struct LocalReachabilityQualification {
    status: &'static str,
    exact_channel_count: usize,
    unresolved_channel_count: usize,
    interval_only_channel_count: usize,
    exact_state_count_decimal: String,
    excluded_state_count_decimal: String,
    unresolved_state_count_decimal: String,
    source_ledger_sha256: String,
}

#[derive(Serialize)]
pub(super) struct NumericalAbiPropagationAnalysis {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    status: &'static str,
    source_evidence_schema: &'static str,
    candidate_source_op_count: usize,
    divergent_source_op_count: usize,
    equivalent_source_op_count: usize,
    unassessed_source_op_count: usize,
    local_reachability_unassessed_source_op_count: usize,
    output_reachable_source_op_count: usize,
    output_isolated_source_op_count: usize,
    exact_local_counterexample_source_op_count: usize,
    residue_excluded_divergence_source_op_count: usize,
    unresolved_divergence_source_op_count: usize,
    exact_output_reachable_source_op_count: usize,
    interval_divergent_state_count_decimal: String,
    exact_local_divergent_state_count_decimal: String,
    residue_excluded_divergent_state_count_decimal: String,
    unresolved_divergent_state_count_decimal: String,
    graph_edge_count: usize,
    graph_ledger_sha256: String,
    source_corridor_edge_instance_count: usize,
    source_boundary_edge_instance_count: usize,
    assessed_source_boundary_edge_instance_payload_bytes_decimal: String,
    unassessed_source_boundary_edge_instance_payload_count: usize,
    unique_reachable_op_count: usize,
    unique_reachable_op_indices: Vec<usize>,
    unique_reachable_tensor_count: usize,
    unique_reachable_tensor_indices: Vec<usize>,
    unique_model_output_tensor_count: usize,
    unique_model_output_tensor_indices: Vec<usize>,
    unique_predicted_boundary_edge_count: usize,
    unique_predicted_boundary_edge_indices: Vec<usize>,
    unique_predicted_boundary_logical_payload_bytes: Option<usize>,
    exact_source_corridor_edge_instance_count: usize,
    exact_source_boundary_edge_instance_count: usize,
    exact_unique_reachable_op_count: usize,
    exact_unique_reachable_op_indices: Vec<usize>,
    exact_unique_reachable_tensor_count: usize,
    exact_unique_reachable_tensor_indices: Vec<usize>,
    exact_unique_predicted_boundary_edge_count: usize,
    exact_unique_predicted_boundary_edge_indices: Vec<usize>,
    exact_unique_predicted_boundary_logical_payload_bytes: Option<usize>,
    reconvergence_source_op_instance_count: usize,
    single_branch_merge_source_op_instance_count: usize,
    maximum_model_output_op_hops: Option<usize>,
    maximum_model_output_graph_route_count_decimal: Option<String>,
    graph_cycle_status: &'static str,
    propagation_ranking_op_indices: Vec<usize>,
    graph_edges: Vec<PropagationGraphEdge>,
    pub(super) sources: Vec<PropagationSource>,
    route_definition: &'static str,
    ranking_definition: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn numerical_abi_propagation_not_computed() -> NumericalAbiPropagationAnalysis {
    build_analysis(
        &[],
        &[],
        &[],
        &empty_rounding_equivalence(),
        None,
        "not_computed_internal_scope",
    )
}

pub(super) fn build_numerical_abi_propagation(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    output_tensor_indices: &[i32],
    equivalence: &RoundingEquivalenceAnalysis,
    reachability: &AccumulatorReachabilityAnalysis,
) -> NumericalAbiPropagationAnalysis {
    build_analysis(
        ops,
        tensors,
        output_tensor_indices,
        equivalence,
        Some(reachability),
        "assessed",
    )
}

fn build_analysis(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    output_tensor_indices: &[i32],
    equivalence: &RoundingEquivalenceAnalysis,
    reachability: Option<&AccumulatorReachabilityAnalysis>,
    requested_status: &'static str,
) -> NumericalAbiPropagationAnalysis {
    let graph_edges = build_graph_edges(ops, tensors);
    let graph_ledger_sha256 = graph_ledger_sha256(&graph_edges);
    let topological_order = topological_op_order(ops, &graph_edges);
    let graph_cycle = topological_order.is_none() && !ops.is_empty();
    let output_set = output_tensor_indices
        .iter()
        .copied()
        .filter(|index| *index >= 0)
        .map(|index| index as usize)
        .collect::<BTreeSet<_>>();
    let edges_by_tensor = edges_by_tensor(&graph_edges);
    let producer_by_tensor = producer_by_tensor(ops);
    let reachability_by_op = reachability
        .map(|evidence| {
            evidence
                .ops
                .iter()
                .map(|row| (row.op_index, row))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let sources = equivalence
        .ops
        .iter()
        .map(|row| {
            build_source(
                row,
                reachability_by_op.get(&row.op_index).copied(),
                ops,
                tensors,
                &output_set,
                &graph_edges,
                &edges_by_tensor,
                &producer_by_tensor,
                topological_order.as_deref(),
                &graph_ledger_sha256,
            )
        })
        .collect::<Vec<_>>();

    let divergent_sources = sources
        .iter()
        .filter(|source| source.assessment_status == "propagates_structurally")
        .collect::<Vec<_>>();
    let exact_sources = divergent_sources
        .iter()
        .copied()
        .filter(|source| source.local_reachability_status == "exact_local_counterexample")
        .collect::<Vec<_>>();
    let unique_reachable_op_indices = sorted_union(
        divergent_sources
            .iter()
            .flat_map(|source| source.reachable_op_indices.iter().copied()),
    );
    let unique_reachable_tensor_indices = sorted_union(
        divergent_sources
            .iter()
            .flat_map(|source| source.reachable_tensor_indices.iter().copied()),
    );
    let unique_model_output_tensor_indices =
        sorted_union(divergent_sources.iter().flat_map(|source| {
            source
                .model_output_paths
                .iter()
                .map(|path| path.output_tensor_index)
        }));
    let unique_predicted_boundary_edge_indices =
        sorted_union(divergent_sources.iter().flat_map(|source| {
            source
                .corridor_edge_indices
                .iter()
                .copied()
                .filter(|index| {
                    graph_edges
                        .get(*index)
                        .is_some_and(|edge| edge.predicted_boundary)
                })
        }));
    let unique_predicted_boundary_logical_payload_bytes = unique_predicted_boundary_edge_indices
        .iter()
        .try_fold(0usize, |total, index| {
            graph_edges
                .get(*index)
                .and_then(|edge| edge.logical_payload_bytes)
                .and_then(|value| total.checked_add(value))
        });
    let exact_unique_reachable_op_indices = sorted_union(
        exact_sources
            .iter()
            .flat_map(|source| source.reachable_op_indices.iter().copied()),
    );
    let exact_unique_reachable_tensor_indices = sorted_union(
        exact_sources
            .iter()
            .flat_map(|source| source.reachable_tensor_indices.iter().copied()),
    );
    let exact_unique_predicted_boundary_edge_indices =
        sorted_union(exact_sources.iter().flat_map(|source| {
            source
                .corridor_edge_indices
                .iter()
                .copied()
                .filter(|index| {
                    graph_edges
                        .get(*index)
                        .is_some_and(|edge| edge.predicted_boundary)
                })
        }));
    let exact_unique_predicted_boundary_logical_payload_bytes =
        exact_unique_predicted_boundary_edge_indices
            .iter()
            .try_fold(0usize, |total, index| {
                graph_edges
                    .get(*index)
                    .and_then(|edge| edge.logical_payload_bytes)
                    .and_then(|value| total.checked_add(value))
            });
    let mut ranking = divergent_sources
        .iter()
        .map(|source| source.op_index)
        .collect::<Vec<_>>();
    let by_index = sources
        .iter()
        .map(|source| (source.op_index, source))
        .collect::<HashMap<_, _>>();
    ranking.sort_by(|left, right| compare_sources(by_index[left], by_index[right]));
    let maximum_model_output_graph_route_count_decimal = divergent_sources
        .iter()
        .filter_map(|source| {
            source
                .exact_model_output_graph_route_count_decimal
                .as_deref()
                .and_then(|value| value.parse::<u128>().ok())
        })
        .max()
        .map(|value| value.to_string());
    let assessed_source_boundary_edge_instance_payload_bytes = divergent_sources
        .iter()
        .map(|source| source.assessed_boundary_logical_payload_bytes as u128)
        .sum::<u128>();
    let unassessed_source_op_count = sources
        .iter()
        .filter(|source| source.assessment_status == "not_assessed")
        .count();
    let local_reachability_unassessed_source_op_count = divergent_sources
        .iter()
        .filter(|source| source.local_reachability_status == "not_assessed")
        .count();
    let status = if requested_status != "assessed" {
        requested_status
    } else if unassessed_source_op_count > 0
        || local_reachability_unassessed_source_op_count > 0
        || graph_cycle
    {
        "partial"
    } else {
        "assessed"
    };

    NumericalAbiPropagationAnalysis {
        schema: SCHEMA,
        method_version: METHOD_VERSION,
        evidence_class: "DERIVED",
        status,
        source_evidence_schema:
            "deepbom.rounding_equivalence.v1 + deepbom.accumulator_reachability.v1",
        candidate_source_op_count: sources.len(),
        divergent_source_op_count: divergent_sources.len(),
        equivalent_source_op_count: sources
            .iter()
            .filter(|source| source.assessment_status == "complete_interval_equivalent")
            .count(),
        unassessed_source_op_count,
        local_reachability_unassessed_source_op_count,
        output_reachable_source_op_count: divergent_sources
            .iter()
            .filter(|source| source.reachable_model_output_tensor_count > 0)
            .count(),
        output_isolated_source_op_count: divergent_sources
            .iter()
            .filter(|source| source.reachable_model_output_tensor_count == 0)
            .count(),
        exact_local_counterexample_source_op_count: exact_sources.len(),
        residue_excluded_divergence_source_op_count: divergent_sources
            .iter()
            .filter(|source| {
                source.provably_unreachable_divergent_state_count_decimal != "0"
            })
            .count(),
        unresolved_divergence_source_op_count: divergent_sources
            .iter()
            .filter(|source| source.unresolved_divergent_state_count_decimal != "0")
            .count(),
        exact_output_reachable_source_op_count: exact_sources
            .iter()
            .filter(|source| source.reachable_model_output_tensor_count > 0)
            .count(),
        interval_divergent_state_count_decimal: sum_source_decimals(
            &divergent_sources,
            |source| &source.divergent_state_count_decimal,
        ),
        exact_local_divergent_state_count_decimal: sum_source_decimals(
            &divergent_sources,
            |source| &source.exact_reachable_divergent_state_count_decimal,
        ),
        residue_excluded_divergent_state_count_decimal: sum_source_decimals(
            &divergent_sources,
            |source| &source.provably_unreachable_divergent_state_count_decimal,
        ),
        unresolved_divergent_state_count_decimal: sum_source_decimals(
            &divergent_sources,
            |source| &source.unresolved_divergent_state_count_decimal,
        ),
        graph_edge_count: graph_edges.len(),
        graph_ledger_sha256,
        source_corridor_edge_instance_count: divergent_sources
            .iter()
            .map(|source| source.corridor_edge_count)
            .sum(),
        source_boundary_edge_instance_count: divergent_sources
            .iter()
            .map(|source| source.predicted_boundary_edge_count)
            .sum(),
        assessed_source_boundary_edge_instance_payload_bytes_decimal:
            assessed_source_boundary_edge_instance_payload_bytes.to_string(),
        unassessed_source_boundary_edge_instance_payload_count: divergent_sources
            .iter()
            .map(|source| source.unassessed_boundary_payload_edge_count)
            .sum(),
        unique_reachable_op_count: unique_reachable_op_indices.len(),
        unique_reachable_op_indices,
        unique_reachable_tensor_count: unique_reachable_tensor_indices.len(),
        unique_reachable_tensor_indices,
        unique_model_output_tensor_count: unique_model_output_tensor_indices.len(),
        unique_model_output_tensor_indices,
        unique_predicted_boundary_edge_count: unique_predicted_boundary_edge_indices.len(),
        unique_predicted_boundary_edge_indices,
        unique_predicted_boundary_logical_payload_bytes,
        exact_source_corridor_edge_instance_count: exact_sources
            .iter()
            .map(|source| source.corridor_edge_count)
            .sum(),
        exact_source_boundary_edge_instance_count: exact_sources
            .iter()
            .map(|source| source.predicted_boundary_edge_count)
            .sum(),
        exact_unique_reachable_op_count: exact_unique_reachable_op_indices.len(),
        exact_unique_reachable_op_indices,
        exact_unique_reachable_tensor_count: exact_unique_reachable_tensor_indices.len(),
        exact_unique_reachable_tensor_indices,
        exact_unique_predicted_boundary_edge_count:
            exact_unique_predicted_boundary_edge_indices.len(),
        exact_unique_predicted_boundary_edge_indices,
        exact_unique_predicted_boundary_logical_payload_bytes,
        reconvergence_source_op_instance_count: divergent_sources
            .iter()
            .map(|source| source.reconvergence_op_count)
            .sum(),
        single_branch_merge_source_op_instance_count: divergent_sources
            .iter()
            .map(|source| source.single_branch_merge_op_count)
            .sum(),
        maximum_model_output_op_hops: divergent_sources
            .iter()
            .filter_map(|source| source.minimum_model_output_op_hops)
            .max(),
        maximum_model_output_graph_route_count_decimal,
        graph_cycle_status: if graph_cycle {
            "cycle_detected_route_count_not_assessed"
        } else {
            "acyclic"
        },
        propagation_ranking_op_indices: ranking,
        graph_edges,
        sources,
        route_definition: "A graph route is a distinct producer-tensor-consumer edge sequence from one divergent source op output tensor to a declared model output tensor. Route multiplicity is counted exactly only for an acyclic operator graph.",
        ranking_definition: "Lexicographic, not a composite score: output reachability, predicted-boundary edge count, exact output-route multiplicity, reconvergence count, reachable-op count, divergent interval-state ratio, then ascending op index.",
        method: "Join every rounding-equivalence source op and accumulator-reachability partition to the artifact producer/tensor/consumer graph. Traverse all downstream nonconstant graph edges, retain every corridor edge index, classify branch reconvergence versus single-branch merge, count exact acyclic routes, and bind each source corridor to the graph, equivalence, and reachability ledgers. Logical payload uses the serialized concrete tensor shape; a dynamic shape signature does not suppress a payload when that concrete shape is fully bound.",
        interpretation_boundary: "Exact-local qualification proves at least one bounded-sum reachable kernel-local accumulator where the pinned rounding paths differ. Downstream corridors remain tensor-level structural potential: they do not prove a full-model input realizes the local assignment, that a declared output changes, or that cancellation, requantization, pooling, activation clamp, or task semantics preserve the difference. Predicted execution-domain crossings and logical payloads are static evidence, not observed copies, latency, runtime assignment, or executed microkernels.",
    }
}

fn qualify_local_reachability(
    equivalence: &RoundingEquivalenceOpRow,
    reachability: Option<&AccumulatorReachabilityOpRow>,
) -> LocalReachabilityQualification {
    let not_assessed = || LocalReachabilityQualification {
        status: "not_assessed",
        exact_channel_count: 0,
        unresolved_channel_count: 0,
        interval_only_channel_count: 0,
        exact_state_count_decimal: "0".to_string(),
        excluded_state_count_decimal: "0".to_string(),
        unresolved_state_count_decimal: "0".to_string(),
        source_ledger_sha256: String::new(),
    };
    if equivalence.assessment_status != "assessed" {
        return not_assessed();
    }
    let Some(reachability) = reachability else {
        return not_assessed();
    };
    let interval = reachability
        .interval_divergent_state_count_decimal
        .parse::<u128>()
        .ok();
    let exact = reachability
        .exact_reachable_divergent_state_count_decimal
        .parse::<u128>()
        .ok();
    let excluded = reachability
        .provably_unreachable_divergent_state_count_decimal
        .parse::<u128>()
        .ok();
    let unresolved = reachability
        .unresolved_divergent_state_count_decimal
        .parse::<u128>()
        .ok();
    let source_matches = reachability.assessment_status == "assessed"
        && reachability.interval_divergent_state_count_decimal
            == equivalence.divergent_state_count_decimal
        && matches!((interval, exact, excluded, unresolved), (Some(total), Some(a), Some(b), Some(c)) if a.checked_add(b).and_then(|value| value.checked_add(c)) == Some(total))
        && reachability.reachability_ledger_sha256.len() == 64
        && reachability
            .reachability_ledger_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !source_matches {
        return not_assessed();
    }
    let exact = exact.unwrap_or(0);
    let unresolved = unresolved.unwrap_or(0);
    let status = if interval == Some(0) {
        "complete_interval_equivalent"
    } else if exact > 0 {
        "exact_local_counterexample"
    } else if unresolved > 0 {
        "unresolved_interval_divergence"
    } else {
        "residue_excluded_interval_divergence"
    };
    LocalReachabilityQualification {
        status,
        exact_channel_count: reachability.exact_reachable_divergent_channel_count,
        unresolved_channel_count: reachability.unresolved_divergent_channel_count,
        interval_only_channel_count: reachability.interval_only_divergent_channel_count,
        exact_state_count_decimal: reachability
            .exact_reachable_divergent_state_count_decimal
            .clone(),
        excluded_state_count_decimal: reachability
            .provably_unreachable_divergent_state_count_decimal
            .clone(),
        unresolved_state_count_decimal: reachability
            .unresolved_divergent_state_count_decimal
            .clone(),
        source_ledger_sha256: reachability.reachability_ledger_sha256.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_source(
    row: &RoundingEquivalenceOpRow,
    reachability_row: Option<&AccumulatorReachabilityOpRow>,
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    output_set: &BTreeSet<usize>,
    graph_edges: &[PropagationGraphEdge],
    edges_by_tensor: &HashMap<usize, Vec<usize>>,
    producer_by_tensor: &HashMap<usize, usize>,
    topological_order: Option<&[usize]>,
    graph_ledger_sha256: &str,
) -> PropagationSource {
    let local = qualify_local_reachability(row, reachability_row);
    let source_output_tensor_indices = ops
        .get(row.op_index)
        .map(|op| {
            op.outputs
                .iter()
                .copied()
                .filter(|index| *index >= 0)
                .map(|index| index as usize)
                .filter(|index| {
                    tensors
                        .get(*index)
                        .is_some_and(|tensor| !tensor.constant_buffer)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let empty = |status: &'static str, reason: String| PropagationSource {
        op_index: row.op_index,
        op_name: row.op_name.clone(),
        assessment_status: status,
        not_assessed_reason: reason,
        source_output_tensor_indices: source_output_tensor_indices.clone(),
        assessed_channel_count: row.assessed_channel_count,
        divergent_channel_count: row.divergent_channel_count,
        interval_state_count_decimal: row.interval_state_count_decimal.clone(),
        divergent_state_count_decimal: row.divergent_state_count_decimal.clone(),
        divergent_state_ratio: row.divergent_state_ratio,
        maximum_absolute_output_delta: row.maximum_absolute_output_delta,
        source_equivalence_ledger_sha256: row.equivalence_ledger_sha256.clone(),
        local_reachability_status: local.status,
        exact_reachable_divergent_channel_count: local.exact_channel_count,
        unresolved_divergent_channel_count: local.unresolved_channel_count,
        interval_only_divergent_channel_count: local.interval_only_channel_count,
        exact_reachable_divergent_state_count_decimal: local.exact_state_count_decimal.clone(),
        provably_unreachable_divergent_state_count_decimal: local
            .excluded_state_count_decimal
            .clone(),
        unresolved_divergent_state_count_decimal: local.unresolved_state_count_decimal.clone(),
        source_reachability_ledger_sha256: local.source_ledger_sha256.clone(),
        direct_consumer_edge_count: 0,
        corridor_edge_count: 0,
        corridor_edge_indices: Vec::new(),
        reachable_op_count: 0,
        reachable_op_indices: Vec::new(),
        reachable_tensor_count: source_output_tensor_indices.len(),
        reachable_tensor_indices: source_output_tensor_indices.clone(),
        reachable_model_output_tensor_count: source_output_tensor_indices
            .iter()
            .filter(|index| output_set.contains(index))
            .count(),
        minimum_model_output_op_hops: source_output_tensor_indices
            .iter()
            .any(|index| output_set.contains(index))
            .then_some(0),
        maximum_reachable_op_hops: None,
        exact_model_output_graph_route_count_decimal: None,
        route_count_status: "not_applicable",
        predicted_boundary_edge_count: 0,
        assessed_boundary_logical_payload_bytes: 0,
        unassessed_boundary_payload_edge_count: 0,
        reconvergence_op_count: 0,
        single_branch_merge_op_count: 0,
        merge_points: Vec::new(),
        model_output_paths: Vec::new(),
        propagation_ledger_sha256: String::new(),
    };
    if row.assessment_status != "assessed" {
        return empty(
            "not_assessed",
            "Source rounding-equivalence row was not assessed.".to_string(),
        );
    }
    if row.divergent_channel_count == 0 {
        return empty(
            "complete_interval_equivalent",
            "Both pinned rounding paths are equivalent over every assessed channel interval hull."
                .to_string(),
        );
    }
    if source_output_tensor_indices.is_empty() {
        return empty(
            "not_assessed",
            "Divergent source op has no nonconstant output tensor.".to_string(),
        );
    }

    let mut tensor_depths = BTreeMap::<usize, usize>::new();
    let mut op_depths = BTreeMap::<usize, usize>::new();
    let mut queue = VecDeque::new();
    for tensor in &source_output_tensor_indices {
        tensor_depths.insert(*tensor, 0);
        queue.push_back(*tensor);
    }
    while let Some(tensor_index) = queue.pop_front() {
        let tensor_depth = tensor_depths[&tensor_index];
        for edge_index in edges_by_tensor.get(&tensor_index).into_iter().flatten() {
            let edge = &graph_edges[*edge_index];
            let op_depth = tensor_depth + 1;
            op_depths
                .entry(edge.consumer_op_index)
                .and_modify(|depth| *depth = (*depth).min(op_depth))
                .or_insert(op_depth);
            if let Some(op) = ops.get(edge.consumer_op_index) {
                for output in op
                    .outputs
                    .iter()
                    .copied()
                    .filter(|index| *index >= 0)
                    .map(|index| index as usize)
                {
                    let should_queue = match tensor_depths.get_mut(&output) {
                        Some(depth) if op_depth < *depth => {
                            *depth = op_depth;
                            true
                        }
                        Some(_) => false,
                        None => {
                            tensor_depths.insert(output, op_depth);
                            true
                        }
                    };
                    if should_queue {
                        queue.push_back(output);
                    }
                }
            }
        }
    }
    let reachable_op_indices = op_depths.keys().copied().collect::<Vec<_>>();
    let reachable_tensor_indices = tensor_depths.keys().copied().collect::<Vec<_>>();
    let corridor_edge_indices = graph_edges
        .iter()
        .filter(|edge| {
            tensor_depths.contains_key(&edge.tensor_index)
                && op_depths.contains_key(&edge.consumer_op_index)
        })
        .map(|edge| edge.edge_index)
        .collect::<Vec<_>>();
    let merge_points = build_merge_points(
        ops,
        graph_edges,
        &corridor_edge_indices,
        &tensor_depths,
        &op_depths,
    );
    let route_counts = topological_order.and_then(|order| {
        exact_route_counts(row.op_index, order, graph_edges, &corridor_edge_indices)
    });
    let model_output_paths = build_output_paths(OutputPathContext {
        source_op_index: row.op_index,
        tensors,
        output_set,
        graph_edges,
        corridor_edge_indices: &corridor_edge_indices,
        tensor_depths: &tensor_depths,
        producer_by_tensor,
        route_counts: route_counts.as_ref(),
    });
    let predicted_boundary_edges = corridor_edge_indices
        .iter()
        .filter_map(|index| graph_edges.get(*index))
        .filter(|edge| edge.predicted_boundary)
        .collect::<Vec<_>>();
    let assessed_boundary_logical_payload_bytes = predicted_boundary_edges
        .iter()
        .filter_map(|edge| edge.logical_payload_bytes)
        .sum();
    let unassessed_boundary_payload_edge_count = predicted_boundary_edges
        .iter()
        .filter(|edge| edge.logical_payload_bytes.is_none())
        .count();
    let exact_model_output_graph_route_count = model_output_paths
        .iter()
        .filter_map(|path| path.exact_graph_route_count_decimal.as_deref())
        .try_fold(0u128, |total, value| {
            value
                .parse::<u128>()
                .ok()
                .and_then(|parsed| total.checked_add(parsed))
        });
    let mut source = PropagationSource {
        op_index: row.op_index,
        op_name: row.op_name.clone(),
        assessment_status: "propagates_structurally",
        not_assessed_reason: String::new(),
        source_output_tensor_indices,
        assessed_channel_count: row.assessed_channel_count,
        divergent_channel_count: row.divergent_channel_count,
        interval_state_count_decimal: row.interval_state_count_decimal.clone(),
        divergent_state_count_decimal: row.divergent_state_count_decimal.clone(),
        divergent_state_ratio: row.divergent_state_ratio,
        maximum_absolute_output_delta: row.maximum_absolute_output_delta,
        source_equivalence_ledger_sha256: row.equivalence_ledger_sha256.clone(),
        local_reachability_status: local.status,
        exact_reachable_divergent_channel_count: local.exact_channel_count,
        unresolved_divergent_channel_count: local.unresolved_channel_count,
        interval_only_divergent_channel_count: local.interval_only_channel_count,
        exact_reachable_divergent_state_count_decimal: local.exact_state_count_decimal.clone(),
        provably_unreachable_divergent_state_count_decimal: local
            .excluded_state_count_decimal
            .clone(),
        unresolved_divergent_state_count_decimal: local.unresolved_state_count_decimal.clone(),
        source_reachability_ledger_sha256: local.source_ledger_sha256.clone(),
        direct_consumer_edge_count: corridor_edge_indices
            .iter()
            .filter(|index| {
                graph_edges
                    .get(**index)
                    .is_some_and(|edge| edge.producer_op_index == row.op_index)
            })
            .count(),
        corridor_edge_count: corridor_edge_indices.len(),
        corridor_edge_indices,
        reachable_op_count: reachable_op_indices.len(),
        reachable_op_indices,
        reachable_tensor_count: reachable_tensor_indices.len(),
        reachable_tensor_indices,
        reachable_model_output_tensor_count: model_output_paths.len(),
        minimum_model_output_op_hops: model_output_paths
            .iter()
            .map(|path| path.shortest_op_hops)
            .min(),
        maximum_reachable_op_hops: op_depths.values().copied().max(),
        exact_model_output_graph_route_count_decimal: exact_model_output_graph_route_count
            .map(|value| value.to_string()),
        route_count_status: if route_counts.is_some()
            && model_output_paths
                .iter()
                .all(|path| path.exact_graph_route_count_decimal.is_some())
        {
            "assessed_acyclic"
        } else {
            "not_assessed_cycle_or_overflow"
        },
        predicted_boundary_edge_count: predicted_boundary_edges.len(),
        assessed_boundary_logical_payload_bytes,
        unassessed_boundary_payload_edge_count,
        reconvergence_op_count: merge_points
            .iter()
            .filter(|point| point.merge_class == "reconvergence")
            .count(),
        single_branch_merge_op_count: merge_points
            .iter()
            .filter(|point| point.merge_class == "single_branch_merge")
            .count(),
        merge_points,
        model_output_paths,
        propagation_ledger_sha256: String::new(),
    };
    source.propagation_ledger_sha256 = source_ledger_sha256(&source, graph_ledger_sha256);
    source
}

fn build_graph_edges(ops: &[OpInfo], tensors: &[TensorInfo]) -> Vec<PropagationGraphEdge> {
    let mut producer_by_tensor = HashMap::<i32, &OpInfo>::new();
    for op in ops {
        for output in op.outputs.iter().copied().filter(|index| *index >= 0) {
            producer_by_tensor.insert(output, op);
        }
    }
    let mut pending = Vec::new();
    for consumer in ops {
        let mut input_slots = BTreeMap::<i32, Vec<usize>>::new();
        for (slot, tensor_index) in consumer.inputs.iter().copied().enumerate() {
            if tensor_index >= 0 {
                input_slots.entry(tensor_index).or_default().push(slot);
            }
        }
        for (tensor_index, consumer_input_slots) in input_slots {
            let Some(producer) = producer_by_tensor.get(&tensor_index).copied() else {
                continue;
            };
            let Some(tensor) = tensors.get(tensor_index as usize) else {
                continue;
            };
            if tensor.constant_buffer {
                continue;
            }
            let producer_domain = predicted_execution_domain(producer);
            let consumer_domain = predicted_execution_domain(consumer);
            let payload = deterministic_tensor_payload_bytes(tensor).ok();
            pending.push(PropagationGraphEdge {
                edge_index: 0,
                producer_op_index: producer.index,
                producer_op_name: producer.name.clone(),
                tensor_index: tensor.index,
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                logical_payload_bytes: payload,
                payload_status: if payload.is_some() {
                    "assessed"
                } else {
                    "not_assessed"
                },
                consumer_op_index: consumer.index,
                consumer_op_name: consumer.name.clone(),
                consumer_input_slots,
                predicted_boundary: producer_domain != consumer_domain,
                predicted_boundary_direction: if producer_domain != consumer_domain {
                    predicted_boundary_direction(producer, consumer).to_string()
                } else {
                    "same_predicted_domain".to_string()
                },
                producer_domain,
                consumer_domain,
            });
        }
    }
    pending.sort_by_key(|edge| {
        (
            edge.producer_op_index,
            edge.consumer_op_index,
            edge.tensor_index,
        )
    });
    for (index, edge) in pending.iter_mut().enumerate() {
        edge.edge_index = index;
    }
    pending
}

fn edges_by_tensor(edges: &[PropagationGraphEdge]) -> HashMap<usize, Vec<usize>> {
    let mut result = HashMap::<usize, Vec<usize>>::new();
    for edge in edges {
        result
            .entry(edge.tensor_index)
            .or_default()
            .push(edge.edge_index);
    }
    result
}

fn producer_by_tensor(ops: &[OpInfo]) -> HashMap<usize, usize> {
    let mut result = HashMap::new();
    for op in ops {
        for output in op
            .outputs
            .iter()
            .copied()
            .filter(|index| *index >= 0)
            .map(|index| index as usize)
        {
            result.insert(output, op.index);
        }
    }
    result
}

fn topological_op_order(ops: &[OpInfo], edges: &[PropagationGraphEdge]) -> Option<Vec<usize>> {
    let mut indegrees = vec![0usize; ops.len()];
    let mut adjacency = vec![BTreeSet::<usize>::new(); ops.len()];
    for edge in edges {
        if adjacency[edge.producer_op_index].insert(edge.consumer_op_index) {
            indegrees[edge.consumer_op_index] += 1;
        }
    }
    let mut ready = indegrees
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(ops.len());
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for consumer in &adjacency[index] {
            indegrees[*consumer] -= 1;
            if indegrees[*consumer] == 0 {
                ready.insert(*consumer);
            }
        }
    }
    (order.len() == ops.len()).then_some(order)
}

fn exact_route_counts(
    source_op_index: usize,
    topological_order: &[usize],
    edges: &[PropagationGraphEdge],
    corridor_edge_indices: &[usize],
) -> Option<HashMap<usize, u128>> {
    let corridor = corridor_edge_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut incoming = HashMap::<usize, Vec<&PropagationGraphEdge>>::new();
    for edge in edges
        .iter()
        .filter(|edge| corridor.contains(&edge.edge_index))
    {
        incoming
            .entry(edge.consumer_op_index)
            .or_default()
            .push(edge);
    }
    let mut routes = HashMap::<usize, u128>::from([(source_op_index, 1)]);
    for op_index in topological_order {
        if *op_index == source_op_index {
            continue;
        }
        let Some(op_edges) = incoming.get(op_index) else {
            continue;
        };
        let count = op_edges.iter().try_fold(0u128, |total, edge| {
            let parent = routes.get(&edge.producer_op_index).copied().unwrap_or(0);
            total.checked_add(parent)
        })?;
        if count > 0 {
            routes.insert(*op_index, count);
        }
    }
    Some(routes)
}

struct OutputPathContext<'a> {
    source_op_index: usize,
    tensors: &'a [TensorInfo],
    output_set: &'a BTreeSet<usize>,
    graph_edges: &'a [PropagationGraphEdge],
    corridor_edge_indices: &'a [usize],
    tensor_depths: &'a BTreeMap<usize, usize>,
    producer_by_tensor: &'a HashMap<usize, usize>,
    route_counts: Option<&'a HashMap<usize, u128>>,
}

fn build_output_paths(context: OutputPathContext<'_>) -> Vec<ModelOutputPath> {
    let OutputPathContext {
        source_op_index,
        tensors,
        output_set,
        graph_edges,
        corridor_edge_indices,
        tensor_depths,
        producer_by_tensor,
        route_counts,
    } = context;
    let corridor = corridor_edge_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut adjacency = HashMap::<usize, Vec<&PropagationGraphEdge>>::new();
    for edge in graph_edges
        .iter()
        .filter(|edge| corridor.contains(&edge.edge_index))
    {
        adjacency
            .entry(edge.producer_op_index)
            .or_default()
            .push(edge);
    }
    for edges in adjacency.values_mut() {
        edges.sort_by_key(|edge| edge.edge_index);
    }
    let mut distances = HashMap::<usize, usize>::from([(source_op_index, 0)]);
    let mut predecessor = HashMap::<usize, usize>::new();
    let mut queue = VecDeque::from([source_op_index]);
    while let Some(op_index) = queue.pop_front() {
        let next_distance = distances[&op_index] + 1;
        for edge in adjacency.get(&op_index).into_iter().flatten() {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                distances.entry(edge.consumer_op_index)
            {
                entry.insert(next_distance);
                predecessor.insert(edge.consumer_op_index, edge.edge_index);
                queue.push_back(edge.consumer_op_index);
            }
        }
    }
    output_set
        .iter()
        .filter(|index| tensor_depths.contains_key(index))
        .filter_map(|output_tensor_index| {
            let tensor = tensors.get(*output_tensor_index)?;
            let producer = producer_by_tensor.get(output_tensor_index).copied();
            let shortest_op_hops = producer
                .and_then(|index| distances.get(&index).copied())
                .unwrap_or(0);
            let mut path_edges = Vec::new();
            let mut current = producer.unwrap_or(source_op_index);
            while current != source_op_index {
                let edge_index = predecessor.get(&current).copied()?;
                path_edges.push(edge_index);
                current = graph_edges.get(edge_index)?.producer_op_index;
            }
            path_edges.reverse();
            let mut path_ops = vec![source_op_index];
            path_ops.extend(
                path_edges
                    .iter()
                    .filter_map(|index| graph_edges.get(*index))
                    .map(|edge| edge.consumer_op_index),
            );
            let boundary_edges = path_edges
                .iter()
                .filter_map(|index| graph_edges.get(*index))
                .filter(|edge| edge.predicted_boundary)
                .collect::<Vec<_>>();
            let boundary_payload = boundary_edges.iter().try_fold(0usize, |total, edge| {
                edge.logical_payload_bytes
                    .and_then(|value| total.checked_add(value))
            });
            let route_count = if producer == Some(source_op_index) || producer.is_none() {
                Some(1u128)
            } else {
                producer
                    .and_then(|index| route_counts.and_then(|counts| counts.get(&index).copied()))
            };
            Some(ModelOutputPath {
                output_tensor_index: *output_tensor_index,
                output_tensor_name: tensor.name.clone(),
                shortest_op_hops,
                shortest_path_op_indices: path_ops,
                shortest_path_edge_indices: path_edges,
                shortest_path_predicted_boundary_count: boundary_edges.len(),
                shortest_path_boundary_logical_payload_bytes: boundary_payload,
                exact_graph_route_count_decimal: route_count.map(|value| value.to_string()),
                route_count_status: if route_count.is_some() {
                    "assessed_acyclic"
                } else {
                    "not_assessed_cycle_or_overflow"
                },
            })
        })
        .collect()
}

fn build_merge_points(
    ops: &[OpInfo],
    graph_edges: &[PropagationGraphEdge],
    corridor_edge_indices: &[usize],
    tensor_depths: &BTreeMap<usize, usize>,
    op_depths: &BTreeMap<usize, usize>,
) -> Vec<MergePoint> {
    let mut incoming = HashMap::<usize, Vec<&PropagationGraphEdge>>::new();
    for edge in graph_edges {
        incoming
            .entry(edge.consumer_op_index)
            .or_default()
            .push(edge);
    }
    let corridor = corridor_edge_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut points = Vec::new();
    for op_index in op_depths.keys() {
        let graph_inputs = incoming.get(op_index).cloned().unwrap_or_default();
        if graph_inputs.len() < 2 {
            continue;
        }
        let influenced = graph_inputs
            .iter()
            .filter(|edge| corridor.contains(&edge.edge_index))
            .map(|edge| edge.tensor_index)
            .collect::<BTreeSet<_>>();
        if influenced.is_empty() {
            continue;
        }
        let uninfluenced = graph_inputs
            .iter()
            .filter(|edge| !tensor_depths.contains_key(&edge.tensor_index))
            .map(|edge| edge.tensor_index)
            .collect::<BTreeSet<_>>();
        let merge_class = if influenced.len() >= 2 {
            "reconvergence"
        } else {
            "single_branch_merge"
        };
        if let Some(op) = ops.get(*op_index) {
            points.push(MergePoint {
                op_index: *op_index,
                op_name: op.name.clone(),
                minimum_op_hops: op_depths[op_index],
                merge_class,
                graph_input_tensor_count: graph_inputs.len(),
                influenced_input_tensor_indices: influenced.into_iter().collect(),
                uninfluenced_input_tensor_indices: uninfluenced.into_iter().collect(),
                predicted_execution_domain: predicted_execution_domain(op),
            });
        }
    }
    points.sort_by_key(|point| (point.minimum_op_hops, point.op_index));
    points
}

fn graph_ledger_sha256(edges: &[PropagationGraphEdge]) -> String {
    let mut digest = Sha256::new();
    digest.update(GRAPH_LEDGER_PREFIX);
    for edge in edges {
        update_u64(&mut digest, edge.edge_index);
        update_u64(&mut digest, edge.producer_op_index);
        update_u64(&mut digest, edge.tensor_index);
        update_u64(&mut digest, edge.consumer_op_index);
        update_optional_u64(&mut digest, edge.logical_payload_bytes);
        update_bool(&mut digest, edge.predicted_boundary);
        update_i32_slice(&mut digest, &edge.tensor_shape);
        update_usize_slice(&mut digest, &edge.consumer_input_slots);
        update_string(&mut digest, &edge.tensor_dtype);
        update_string(&mut digest, &edge.producer_domain);
        update_string(&mut digest, &edge.consumer_domain);
        update_string(&mut digest, &edge.predicted_boundary_direction);
    }
    hex_digest(digest.finalize().as_slice())
}

fn source_ledger_sha256(source: &PropagationSource, graph_ledger_sha256: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(SOURCE_LEDGER_PREFIX);
    update_string(&mut digest, graph_ledger_sha256);
    update_string(&mut digest, &source.source_equivalence_ledger_sha256);
    update_string(&mut digest, &source.source_reachability_ledger_sha256);
    update_string(&mut digest, source.local_reachability_status);
    update_u64(&mut digest, source.op_index);
    update_u64(&mut digest, source.assessed_channel_count);
    update_u64(&mut digest, source.divergent_channel_count);
    update_string(&mut digest, &source.interval_state_count_decimal);
    update_string(&mut digest, &source.divergent_state_count_decimal);
    update_u64(&mut digest, source.exact_reachable_divergent_channel_count);
    update_u64(&mut digest, source.unresolved_divergent_channel_count);
    update_u64(&mut digest, source.interval_only_divergent_channel_count);
    update_string(
        &mut digest,
        &source.exact_reachable_divergent_state_count_decimal,
    );
    update_string(
        &mut digest,
        &source.provably_unreachable_divergent_state_count_decimal,
    );
    update_string(
        &mut digest,
        &source.unresolved_divergent_state_count_decimal,
    );
    update_usize_slice(&mut digest, &source.source_output_tensor_indices);
    update_usize_slice(&mut digest, &source.corridor_edge_indices);
    update_usize_slice(&mut digest, &source.reachable_op_indices);
    update_usize_slice(&mut digest, &source.reachable_tensor_indices);
    for path in &source.model_output_paths {
        update_u64(&mut digest, path.output_tensor_index);
        update_u64(&mut digest, path.shortest_op_hops);
        update_usize_slice(&mut digest, &path.shortest_path_op_indices);
        update_usize_slice(&mut digest, &path.shortest_path_edge_indices);
        update_string(
            &mut digest,
            path.exact_graph_route_count_decimal
                .as_deref()
                .unwrap_or(""),
        );
    }
    for point in &source.merge_points {
        update_u64(&mut digest, point.op_index);
        update_u64(&mut digest, point.minimum_op_hops);
        update_string(&mut digest, point.merge_class);
        update_usize_slice(&mut digest, &point.influenced_input_tensor_indices);
        update_usize_slice(&mut digest, &point.uninfluenced_input_tensor_indices);
    }
    hex_digest(digest.finalize().as_slice())
}

fn update_u64(digest: &mut Sha256, value: usize) {
    digest.update((value as u64).to_le_bytes());
}

fn update_optional_u64(digest: &mut Sha256, value: Option<usize>) {
    digest.update(
        value
            .map(|item| item as u64)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
}

fn update_bool(digest: &mut Sha256, value: bool) {
    digest.update([u8::from(value)]);
}

fn update_i32_slice(digest: &mut Sha256, values: &[i32]) {
    update_u64(digest, values.len());
    for value in values {
        digest.update(value.to_le_bytes());
    }
}

fn update_usize_slice(digest: &mut Sha256, values: &[usize]) {
    update_u64(digest, values.len());
    for value in values {
        update_u64(digest, *value);
    }
}

fn update_string(digest: &mut Sha256, value: &str) {
    update_u64(digest, value.len());
    digest.update(value.as_bytes());
}

fn compare_sources(left: &PropagationSource, right: &PropagationSource) -> Ordering {
    right
        .reachable_model_output_tensor_count
        .cmp(&left.reachable_model_output_tensor_count)
        .then_with(|| {
            right
                .predicted_boundary_edge_count
                .cmp(&left.predicted_boundary_edge_count)
        })
        .then_with(|| {
            decimal_u128(&right.exact_model_output_graph_route_count_decimal).cmp(&decimal_u128(
                &left.exact_model_output_graph_route_count_decimal,
            ))
        })
        .then_with(|| {
            right
                .reconvergence_op_count
                .cmp(&left.reconvergence_op_count)
        })
        .then_with(|| right.reachable_op_count.cmp(&left.reachable_op_count))
        .then_with(|| compare_ratio_desc(left, right))
        .then_with(|| left.op_index.cmp(&right.op_index))
}

fn compare_ratio_desc(left: &PropagationSource, right: &PropagationSource) -> Ordering {
    let left_numerator = left
        .divergent_state_count_decimal
        .parse::<u128>()
        .unwrap_or(0);
    let left_denominator = left
        .interval_state_count_decimal
        .parse::<u128>()
        .unwrap_or(1);
    let right_numerator = right
        .divergent_state_count_decimal
        .parse::<u128>()
        .unwrap_or(0);
    let right_denominator = right
        .interval_state_count_decimal
        .parse::<u128>()
        .unwrap_or(1);
    (right_numerator * left_denominator).cmp(&(left_numerator * right_denominator))
}

fn decimal_u128(value: &Option<String>) -> u128 {
    value
        .as_deref()
        .and_then(|item| item.parse::<u128>().ok())
        .unwrap_or(0)
}

fn sorted_union(values: impl Iterator<Item = usize>) -> Vec<usize> {
    values.collect::<BTreeSet<_>>().into_iter().collect()
}

fn sum_source_decimals(
    sources: &[&PropagationSource],
    field: impl for<'a> Fn(&'a PropagationSource) -> &'a str,
) -> String {
    sources
        .iter()
        .filter_map(|source| field(source).parse::<u128>().ok())
        .sum::<u128>()
        .to_string()
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn empty_rounding_equivalence() -> RoundingEquivalenceAnalysis {
    super::rounding_equivalence::rounding_equivalence_not_computed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topological_order_rejects_cycle() {
        let ops = vec![test_op(0, vec![1], vec![0]), test_op(1, vec![0], vec![1])];
        let tensors = vec![test_tensor(0), test_tensor(1)];
        let edges = build_graph_edges(&ops, &tensors);
        assert_eq!(edges.len(), 2);
        assert!(topological_op_order(&ops, &edges).is_none());
    }

    #[test]
    fn route_count_adds_reconverged_paths() {
        let ops = vec![
            test_op(0, vec![], vec![0]),
            test_op(1, vec![0], vec![1]),
            test_op(2, vec![0], vec![2]),
            test_op(3, vec![1, 2], vec![3]),
        ];
        let tensors = (0..4).map(test_tensor).collect::<Vec<_>>();
        let edges = build_graph_edges(&ops, &tensors);
        let order = topological_op_order(&ops, &edges).unwrap();
        let indices = edges.iter().map(|edge| edge.edge_index).collect::<Vec<_>>();
        let counts = exact_route_counts(0, &order, &edges, &indices).unwrap();
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&1));
        assert_eq!(counts.get(&3), Some(&2));
    }

    fn test_tensor(index: usize) -> TensorInfo {
        TensorInfo {
            index,
            name: format!("t{index}"),
            shape: vec![1],
            shape_signature: vec![],
            dtype: "UINT8".to_string(),
            buffer_index: 0,
            buffer_data_offset: 0,
            buffer_data_length: 0,
            constant_buffer: false,
            sparse_storage: false,
            sparse_encoding: None,
            quant_scales: 0,
            quant_zero_points: 0,
            quantized_dimension: 0,
            scale_sample: vec![],
            zero_point_sample: vec![],
            scale_min: 0.0,
            scale_max: 0.0,
            scale_ratio: 0.0,
            scale_mean: 0.0,
            scale_stddev: 0.0,
            scale_cv: 0.0,
            zero_point_min: 0,
            zero_point_max: 0,
            zero_point_offset_max: 0,
            zero_point_status: String::new(),
            zero_point_detail: String::new(),
            scale_mode: String::new(),
            scale_ratio_meaningful: false,
            quant_risk: String::new(),
            buffer_hash: String::new(),
            is_variable: false,
        }
    }

    fn test_op(index: usize, inputs: Vec<i32>, outputs: Vec<i32>) -> OpInfo {
        OpInfo {
            index,
            name: if inputs.len() > 1 { "ADD" } else { "CONV_2D" }.to_string(),
            version: 1,
            stage_index: None,
            stage_key: None,
            inputs,
            outputs,
            output_shapes: vec![vec![1]],
            macs: 0.0,
            mac_percent: 0.0,
            ops: 0.0,
            estimated_bytes: 0.0,
            fallback_byte_percent: 0.0,
            row_working_set_bytes: 0.0,
            row_working_set_ratio: 0.0,
            row_working_set_severity: String::new(),
            cache_payload: crate::CachePayloadBreakdown::not_applicable("test"),
            intensity_ops_per_byte: 0.0,
            static_bound_guess: String::new(),
            static_action: String::new(),
            roofline_reason: String::new(),
            fused_activation: String::new(),
            fusion_status: String::new(),
            fusion_detail: String::new(),
            xnnpack_supported: false,
            xnnpack_reason: String::new(),
            xnnpack_chain_id: -1,
            xnnpack_chain_role: String::new(),
            xnnpack_chain_break: false,
            xnnpack_break_class: String::new(),
            chain_break_impact_mac_percent: 0.0,
            chain_break_overhead_us_low: 0.0,
            chain_break_overhead_us_high: 0.0,
            target_microkernel_hint: String::new(),
            compute_kernel_class: String::new(),
            xnnpack_kernel_candidate: String::new(),
            xnnpack_kernel_tile_mr: 0,
            xnnpack_kernel_tile_nr: 0,
            xnnpack_kernel_channel_tile: 0,
            xnnpack_kernel_primary_tile: 0,
            xnnpack_kernel_source: String::new(),
            xnnpack_kernel_evidence_class: String::new(),
            xnnpack_kernel_selector_status: String::new(),
            xnnpack_build_requirement: String::new(),
            xnnpack_kernel_candidates: vec![],
            xnnpack_kernel_alignment_multiples: vec![],
            quantized_path: true,
            quantized_compute_path: true,
            quantization_state: String::new(),
            quantization_detail: String::new(),
            weight_bytes: 0.0,
            weight_packing_overhead_us: 0.0,
            weight_packing_risk: String::new(),
            weight_packing_detail: String::new(),
            output_channels: 1,
            channel_alignment_multiple: 1,
            channel_alignment_status: String::new(),
            channel_alignment_detail: String::new(),
            channel_tail_overhead_percent: 0.0,
            channel_tail_overhead_percent_min: 0.0,
            channel_tail_overhead_percent_max: 0.0,
            quant_scale_ratio: 0.0,
            quant_scale_cv: 0.0,
            quant_scale_mode: String::new(),
            quant_scale_ratio_meaningful: false,
            quant_zero_point_offset: 0,
            quant_zero_point_risk: String::new(),
            quant_zero_point_status: String::new(),
            quant_risk: String::new(),
            quant_risk_detail: String::new(),
            low_norm_filter_count: None,
            low_norm_filter_total: None,
            quant_hole: false,
            quant_hole_class: String::new(),
            quant_hole_detail: String::new(),
            patterns: vec![],
            topo_role: String::new(),
            topo_depth: 0,
            topo_fan_out_max: 0,
            bottleneck_compute_us: 0.0,
            bottleneck_memory_us: 0.0,
            bottleneck_packing_us: 0.0,
            bottleneck_break_us: 0.0,
            bottleneck_fallback_us: 0.0,
            bottleneck_total_us: 0.0,
            bottleneck_dominant: String::new(),
        }
    }
}
