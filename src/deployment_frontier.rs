use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use wasm_bindgen::prelude::*;

const FRONTIER_SCHEMA: &str = "deepbom.deployment_frontier.v1.6";
const FRONTIER_METHOD_VERSION: &str = "2026-07-29.1";
const COVERAGE_THRESHOLD: f64 = 0.8;
const MAX_TARGETS: usize = 8;

#[derive(Clone, Copy, Default, Serialize)]
struct FrontierComponents {
    compute_us: f64,
    memory_us: f64,
    packing_us: f64,
    boundary_us: f64,
    fallback_us: f64,
}

impl FrontierComponents {
    fn from_op(op: &OpInfo) -> Self {
        let (compute_us, memory_us) = active_roofline_components(
            finite_non_negative(op.bottleneck_compute_us),
            finite_non_negative(op.bottleneck_memory_us),
        );
        Self {
            compute_us,
            memory_us,
            packing_us: finite_non_negative(op.bottleneck_packing_us),
            boundary_us: finite_non_negative(op.bottleneck_break_us),
            fallback_us: finite_non_negative(op.bottleneck_fallback_us),
        }
    }

    fn total(self) -> f64 {
        self.compute_us + self.memory_us + self.packing_us + self.boundary_us + self.fallback_us
    }

    fn steady_total(self) -> f64 {
        self.compute_us + self.memory_us + self.fallback_us
    }

    fn add(&mut self, other: Self) {
        self.compute_us += other.compute_us;
        self.memory_us += other.memory_us;
        self.packing_us += other.packing_us;
        self.boundary_us += other.boundary_us;
        self.fallback_us += other.fallback_us;
    }

    fn named(self, component: &str) -> f64 {
        match component {
            "packing" => self.packing_us,
            "boundary" => self.boundary_us,
            "fallback" => self.fallback_us,
            _ => 0.0,
        }
    }
}

#[derive(Serialize)]
struct FrontierTarget {
    target_id: String,
    target_label: String,
    target_profile_sha256: String,
    l1_data_bytes: usize,
    l2_bytes: usize,
    l2_capacity_scope: String,
    cache_assumption: String,
    cache_source_url: String,
    hardware_spec_evidence_class: String,
    performance_model_evidence_class: String,
    effective_peak_gops: f64,
    effective_memory_bandwidth_gbps: f64,
    weight_packing_bandwidth_gbps: f64,
    compute_utilization_factor: f64,
    l1_watch_count: usize,
    l2_watch_count: Option<usize>,
    max_l1_ratio: f64,
    max_l2_ratio: Option<f64>,
    steady_state_us: f64,
    steady_state_low_us: f64,
    steady_state_high_us: f64,
    cold_start_us: f64,
    cold_start_low_us: f64,
    cold_start_high_us: f64,
    boundary_setup_low_us: f64,
    boundary_setup_midpoint_us: f64,
    boundary_setup_high_us: f64,
    pad_fusion_recoverable_upper_bound_us: f64,
    total_us: f64,
    components: FrontierComponents,
    assessed_op_count: usize,
    ridge_point_ops_per_byte: f64,
    low_intensity_op_ratio: f64,
    predicted_effective_chain_breaks: usize,
    estimated_int8_speedup: f64,
    float_artifact: bool,
    top_op_index: Option<usize>,
    top_op_name: Option<String>,
    top_op_us: f64,
    top_op_steady_state_us: f64,
    top_op_cold_start_us: f64,
}

#[derive(Clone, Serialize)]
struct FrontierOpTarget {
    target_id: String,
    steady_state_us: f64,
    cold_start_us: f64,
    total_us: f64,
    contribution_share: f64,
    rank: usize,
    bound: String,
    dominant_component: String,
    components: FrontierComponents,
}

#[derive(Serialize)]
struct FrontierOp {
    op_index: usize,
    op_name: String,
    mean_contribution_share: f64,
    min_contribution_share: f64,
    max_contribution_share: f64,
    best_rank: usize,
    worst_rank: usize,
    rank_span: usize,
    bound_classes: Vec<String>,
    dominant_components: Vec<String>,
    in_robust_coverage_union: bool,
    target_estimates: Vec<FrontierOpTarget>,
}

#[derive(Serialize)]
struct TargetCoverage {
    target_id: String,
    selected_prefix_op_count: usize,
    prefix_coverage: f64,
    union_coverage: f64,
}

#[derive(Serialize)]
struct RobustCoverage {
    threshold: f64,
    method: &'static str,
    selected_op_indices: Vec<usize>,
    selected_op_count: usize,
    per_target: Vec<TargetCoverage>,
    minimum_union_coverage: f64,
}

#[derive(Serialize)]
struct ComponentContributionDelta {
    compute: f64,
    memory: f64,
    packing: f64,
    boundary: f64,
    fallback: f64,
    largest_absolute_component: &'static str,
    largest_absolute_delta: f64,
}

#[derive(Serialize)]
struct DivergenceDriver {
    op_index: usize,
    op_name: String,
    normalized_js_contribution: f64,
    attribution_share: f64,
    left_contribution_share: f64,
    right_contribution_share: f64,
    signed_contribution_share_delta: f64,
    absolute_contribution_share_delta: f64,
    left_rank: usize,
    right_rank: usize,
    rank_delta: usize,
    left_bound: String,
    right_bound: String,
    bound_transition: bool,
    left_dominant_component: String,
    right_dominant_component: String,
    dominant_component_transition: bool,
    component_contribution_delta: ComponentContributionDelta,
}

#[derive(Serialize)]
struct TargetDivergence {
    left_target_id: String,
    right_target_id: String,
    normalized_jensen_shannon_divergence: f64,
    coverage_prefix_jaccard: f64,
    attribution_sum: f64,
    attribution_prefix_threshold: f64,
    attribution_prefix_op_count: usize,
    attribution_prefix_coverage: f64,
    top_driver_op_index: Option<usize>,
    top_driver_op_name: Option<String>,
    top_driver_attribution_share: f64,
    bound_transition_op_count: usize,
    dominant_component_transition_op_count: usize,
    drivers: Vec<DivergenceDriver>,
}

#[derive(Serialize)]
struct DivergenceSummary {
    method: &'static str,
    pair_count: usize,
    mean_normalized_jensen_shannon_divergence: f64,
    max_normalized_jensen_shannon_divergence: f64,
    min_coverage_prefix_jaccard: f64,
    pairs: Vec<TargetDivergence>,
}

#[derive(Serialize)]
struct InterventionTarget {
    target_id: String,
    recoverable_us: f64,
    recoverable_share: f64,
    upper_bound_speedup: f64,
}

#[derive(Serialize)]
struct FrontierIntervention {
    id: &'static str,
    label: &'static str,
    removed_component: &'static str,
    evidence_class: &'static str,
    min_recoverable_share: f64,
    median_recoverable_share: f64,
    max_recoverable_share: f64,
    min_upper_bound_speedup: f64,
    max_upper_bound_speedup: f64,
    per_target: Vec<InterventionTarget>,
    interpretation_boundary: &'static str,
}

#[derive(Serialize)]
struct EvidenceQueueItem {
    id: &'static str,
    evidence_needed: &'static str,
    decision_exposed_op_indices: Vec<usize>,
    decision_exposed_op_count: usize,
    max_single_target_contribution_share: f64,
    evidence_boundary: &'static str,
}

#[derive(Serialize)]
struct DeploymentFrontier {
    schema: &'static str,
    method_version: &'static str,
    evidence_class: &'static str,
    artifact_sha256: String,
    artifact_filename: String,
    target_count: usize,
    op_count: usize,
    cache_watch_ratio: f64,
    targets: Vec<FrontierTarget>,
    robust_coverage: RobustCoverage,
    target_divergence: DivergenceSummary,
    interventions: Vec<FrontierIntervention>,
    evidence_queue: Vec<EvidenceQueueItem>,
    ops: Vec<FrontierOp>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[wasm_bindgen]
pub fn compute_deployment_frontier(
    bytes: &[u8],
    filename: &str,
    target_ids_json: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    if bytes.is_empty() {
        return Err(JsValue::from_str("Model bytes are empty."));
    }
    let target_ids =
        parse_target_ids(target_ids_json).map_err(|error| JsValue::from_str(&error))?;
    let analyses = target_ids
        .iter()
        .map(|target_id| analyze_with_target_without_step_response(bytes, filename, target_id))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| JsValue::from_str(&error))?;
    let mut frontier =
        build_deployment_frontier(&analyses).map_err(|error| JsValue::from_str(&error))?;
    frontier.artifact_sha256 = hex_lower(&Sha256::digest(bytes));
    serde_wasm_bindgen::to_value(&frontier).map_err(|error| JsValue::from_str(&error.to_string()))
}

fn parse_target_ids(value: &str) -> Result<Vec<String>, String> {
    let parsed: Vec<String> =
        serde_json::from_str(value).map_err(|error| format!("Invalid target ID JSON: {error}"))?;
    let mut seen = BTreeSet::new();
    let target_ids: Vec<String> = parsed
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .filter(|value| seen.insert(value.clone()))
        .collect();
    if target_ids.len() < 2 {
        return Err(
            "Deployment frontier requires at least two distinct target profiles.".to_string(),
        );
    }
    if target_ids.len() > MAX_TARGETS {
        return Err(format!(
            "Deployment frontier accepts at most {MAX_TARGETS} target profiles."
        ));
    }
    Ok(target_ids)
}

fn build_deployment_frontier(analyses: &[Analysis]) -> Result<DeploymentFrontier, String> {
    let first = analyses
        .first()
        .ok_or_else(|| "No target analyses were supplied.".to_string())?;
    if analyses.len() < 2 {
        return Err("At least two target analyses are required.".to_string());
    }
    if first.ops.is_empty() {
        return Err("The model contains no operators to compare.".to_string());
    }
    for analysis in analyses.iter().skip(1) {
        if analysis.model_sha256 != first.model_sha256
            || analysis.ops.len() != first.ops.len()
            || analysis
                .ops
                .iter()
                .zip(&first.ops)
                .any(|(left, right)| left.index != right.index || left.name != right.name)
        {
            return Err("Target analyses do not describe the same artifact topology.".to_string());
        }
    }

    let mut target_components = Vec::with_capacity(analyses.len());
    let mut target_steady_totals = Vec::with_capacity(analyses.len());
    let mut target_cold_totals = Vec::with_capacity(analyses.len());
    let mut distributions = Vec::with_capacity(analyses.len());
    let mut ranks = Vec::with_capacity(analyses.len());
    let mut prefixes = Vec::with_capacity(analyses.len());
    let mut targets = Vec::with_capacity(analyses.len());

    for analysis in analyses {
        let components: Vec<FrontierComponents> = analysis
            .ops
            .iter()
            .map(FrontierComponents::from_op)
            .collect();
        let mut aggregate = FrontierComponents::default();
        for component in &components {
            aggregate.add(*component);
        }
        let cold_total = aggregate.total();
        let steady_total = aggregate.steady_total();
        let boundary_low_us: f64 = analysis
            .ops
            .iter()
            .filter(|op| op.xnnpack_chain_break)
            .map(|op| finite_non_negative(op.chain_break_overhead_us_low))
            .sum();
        let boundary_high_us: f64 = analysis
            .ops
            .iter()
            .filter(|op| op.xnnpack_chain_break)
            .map(|op| finite_non_negative(op.chain_break_overhead_us_high))
            .sum();
        let pad_fusion_recoverable_upper_bound_us = finite_non_negative(
            analysis
                .ops
                .iter()
                .zip(&components)
                .filter(|(op, _)| {
                    op.name == "PAD" && op.fusion_status == "runtime folding unobserved"
                })
                .map(|(_, component)| component.steady_total())
                .sum(),
        );
        let steady_low = (steady_total - pad_fusion_recoverable_upper_bound_us).max(0.0);
        let steady_high = steady_total;
        let bottleneck_total: f64 = analysis
            .ops
            .iter()
            .map(|op| finite_non_negative(op.bottleneck_total_us))
            .sum();
        if !near(cold_total, bottleneck_total) {
            return Err(format!(
                "Target {} roofline component ledger does not conserve bottleneck_total_us.",
                analysis.target_profile.id
            ));
        }
        let distribution: Vec<f64> = components
            .iter()
            .map(|component| ratio(component.steady_total(), steady_total))
            .collect();
        let rank = rank_descending(
            &components
                .iter()
                .map(|item| item.steady_total())
                .collect::<Vec<_>>(),
        );
        let prefix = coverage_prefix(&distribution, COVERAGE_THRESHOLD);
        let top_position = rank.iter().position(|value| *value == 1);
        targets.push(FrontierTarget {
            target_id: analysis.target_profile.id.clone(),
            target_label: analysis.target_profile.label.clone(),
            target_profile_sha256: analysis.target_profile.profile_sha256.clone(),
            l1_data_bytes: analysis.target_profile.l1_data_bytes,
            l2_bytes: analysis.target_profile.l2_bytes,
            l2_capacity_scope: analysis.target_profile.l2_capacity_scope.clone(),
            cache_assumption: analysis.target_profile.cache_assumption.clone(),
            cache_source_url: analysis.target_profile.cache_source_url.clone(),
            hardware_spec_evidence_class: analysis
                .target_profile
                .hardware_spec
                .as_ref()
                .map(|spec| spec.evidence_class.clone())
                .unwrap_or_else(|| "HEURISTIC_PROFILE".to_string()),
            performance_model_evidence_class: analysis
                .target_profile
                .performance_model_evidence_class
                .clone(),
            effective_peak_gops: analysis.target_profile.effective_peak_gops,
            effective_memory_bandwidth_gbps: analysis
                .target_profile
                .effective_memory_bandwidth_gbps,
            weight_packing_bandwidth_gbps: analysis.target_profile.weight_packing_bandwidth_gbps,
            compute_utilization_factor: analysis.target_profile.compute_utilization_factor,
            l1_watch_count: analysis.insights.l1_watch_count,
            l2_watch_count: analysis
                .target_profile
                .l2_capacity_scope
                .starts_with("private_per_core")
                .then(|| {
                    analysis
                        .ops
                        .iter()
                        .filter(|op| {
                            cache_working_set_is_watch(
                                op.row_working_set_bytes / analysis.target_profile.l2_bytes as f64,
                            )
                        })
                        .count()
                }),
            max_l1_ratio: analysis.insights.max_l1_ratio,
            max_l2_ratio: analysis
                .target_profile
                .l2_capacity_scope
                .starts_with("private_per_core")
                .then(|| {
                    analysis
                        .ops
                        .iter()
                        .map(|op| {
                            op.row_working_set_bytes / analysis.target_profile.l2_bytes as f64
                        })
                        .fold(0.0f64, f64::max)
                }),
            steady_state_us: steady_total,
            steady_state_low_us: steady_low,
            steady_state_high_us: steady_high,
            cold_start_us: cold_total,
            cold_start_low_us: steady_low + aggregate.packing_us + boundary_low_us,
            cold_start_high_us: steady_high + aggregate.packing_us + boundary_high_us,
            boundary_setup_low_us: boundary_low_us,
            boundary_setup_midpoint_us: aggregate.boundary_us,
            boundary_setup_high_us: boundary_high_us,
            pad_fusion_recoverable_upper_bound_us,
            total_us: cold_total,
            components: aggregate,
            assessed_op_count: components.len(),
            ridge_point_ops_per_byte: finite_non_negative(
                analysis.target_profile.ridge_point_ops_per_byte,
            ),
            low_intensity_op_ratio: ratio(
                analysis
                    .ops
                    .iter()
                    .filter(|op| op.static_bound_guess == "memory-bound")
                    .count() as f64,
                analysis.ops.len() as f64,
            ),
            predicted_effective_chain_breaks: analysis.xnnpack_effective_chain_breaks,
            estimated_int8_speedup: finite_non_negative(analysis.estimated_int8_speedup),
            float_artifact: matches!(
                analysis.quantization_status.classification.as_str(),
                "not_quantized_float" | "float16_weight_storage"
            ),
            top_op_index: top_position.map(|position| analysis.ops[position].index),
            top_op_name: top_position.map(|position| analysis.ops[position].name.clone()),
            top_op_us: top_position
                .map(|position| components[position].steady_total())
                .unwrap_or(0.0),
            top_op_steady_state_us: top_position
                .map(|position| components[position].steady_total())
                .unwrap_or(0.0),
            top_op_cold_start_us: top_position
                .map(|position| components[position].total())
                .unwrap_or(0.0),
        });
        target_components.push(components);
        target_steady_totals.push(steady_total);
        target_cold_totals.push(cold_total);
        distributions.push(distribution);
        ranks.push(rank);
        prefixes.push(prefix);
    }

    let union: BTreeSet<usize> = prefixes.iter().flatten().copied().collect();
    let robust_coverage = build_robust_coverage(analyses, &distributions, &prefixes, &union);
    let target_divergence = build_divergence(
        analyses,
        &target_components,
        &target_steady_totals,
        &distributions,
        &ranks,
        &prefixes,
    );
    let interventions = build_interventions(
        analyses,
        &target_components,
        &target_steady_totals,
        &target_cold_totals,
    );
    let mut ops = build_frontier_ops(analyses, &target_components, &distributions, &ranks, &union);
    ops.sort_by(|left, right| {
        compare_f64_desc(left.max_contribution_share, right.max_contribution_share)
            .then_with(|| {
                compare_f64_desc(left.mean_contribution_share, right.mean_contribution_share)
            })
            .then_with(|| left.op_index.cmp(&right.op_index))
    });
    let evidence_queue = build_evidence_queue(first, &ops, &union);

    Ok(DeploymentFrontier {
        schema: FRONTIER_SCHEMA,
        method_version: FRONTIER_METHOD_VERSION,
        evidence_class: "DERIVED_FROM_PINNED_TARGET_PROFILE_STATIC_ESTIMATES",
        artifact_sha256: first.model_sha256.clone(),
        artifact_filename: first.filename.clone(),
        target_count: analyses.len(),
        op_count: first.ops.len(),
        cache_watch_ratio: L1_WORKING_SET_WATCH_RATIO,
        targets,
        robust_coverage,
        target_divergence,
        interventions,
        evidence_queue,
        ops,
        method: "For each pinned planning profile, assign max(compute, memory) exactly once to the active roofline-bound component, then add predicted fallback traffic to the steady-state point ledger. One-time packing and the unmeasured partition-planning setup profile are retained only in the cold-start ledger. The cold low/high bounds replace the summed setup midpoint with the profile low/high. The low bound also subtracts the complete steady cost of directly connected PAD-to-convolution rows only as an unobserved fusion-recoverable upper bound; point/high retain explicit PAD materialization. Robust ranking, 80% prefixes, and pairwise Jensen-Shannon divergence use point-estimate steady-state op totals. The target row conserves cold_start_us = steady_state_us + packing_us + boundary_setup_us.",
        interpretation_boundary: "All latency and speedup values are target-profile HEURISTIC planning quantities, not device measurements. Frontier ranking and divergence exclude one-time packing and partition-planning setup. Packing and boundary-setup counterfactuals use the cold-start denominator; fallback uses the steady-state denominator. Cache ratios are deterministic against profile assumptions but do not add an uncalibrated cache-miss penalty to modeled latency. Runtime placement, executed fusion/tiling/microkernels, cache residency/conflicts, thermal state, scheduler behavior, and real memory traffic require separately bound observations.",
    })
}

fn build_frontier_ops(
    analyses: &[Analysis],
    components: &[Vec<FrontierComponents>],
    distributions: &[Vec<f64>],
    ranks: &[Vec<usize>],
    union: &BTreeSet<usize>,
) -> Vec<FrontierOp> {
    analyses[0]
        .ops
        .iter()
        .enumerate()
        .map(|(position, op)| {
            let target_estimates: Vec<FrontierOpTarget> = analyses
                .iter()
                .enumerate()
                .map(|(target_index, analysis)| FrontierOpTarget {
                    target_id: analysis.target_profile.id.clone(),
                    steady_state_us: components[target_index][position].steady_total(),
                    cold_start_us: components[target_index][position].total(),
                    total_us: components[target_index][position].total(),
                    contribution_share: distributions[target_index][position],
                    rank: ranks[target_index][position],
                    bound: analysis.ops[position].static_bound_guess.clone(),
                    dominant_component: steady_dominant_component(
                        components[target_index][position],
                    )
                    .to_string(),
                    components: components[target_index][position],
                })
                .collect();
            let shares: Vec<f64> = target_estimates
                .iter()
                .map(|item| item.contribution_share)
                .collect();
            let op_ranks: Vec<usize> = target_estimates.iter().map(|item| item.rank).collect();
            FrontierOp {
                op_index: op.index,
                op_name: op.name.clone(),
                mean_contribution_share: mean(&shares),
                min_contribution_share: min_f64(&shares),
                max_contribution_share: max_f64(&shares),
                best_rank: op_ranks.iter().copied().min().unwrap_or(0),
                worst_rank: op_ranks.iter().copied().max().unwrap_or(0),
                rank_span: op_ranks.iter().copied().max().unwrap_or(0)
                    - op_ranks.iter().copied().min().unwrap_or(0),
                bound_classes: unique_sorted(
                    target_estimates.iter().map(|item| item.bound.clone()),
                ),
                dominant_components: unique_sorted(
                    target_estimates
                        .iter()
                        .map(|item| item.dominant_component.clone()),
                ),
                in_robust_coverage_union: union.contains(&position),
                target_estimates,
            }
        })
        .collect()
}

fn build_robust_coverage(
    analyses: &[Analysis],
    distributions: &[Vec<f64>],
    prefixes: &[Vec<usize>],
    union: &BTreeSet<usize>,
) -> RobustCoverage {
    let per_target: Vec<TargetCoverage> = analyses
        .iter()
        .enumerate()
        .map(|(target_index, analysis)| TargetCoverage {
            target_id: analysis.target_profile.id.clone(),
            selected_prefix_op_count: prefixes[target_index].len(),
            prefix_coverage: prefixes[target_index]
                .iter()
                .map(|position| distributions[target_index][*position])
                .sum(),
            union_coverage: union
                .iter()
                .map(|position| distributions[target_index][*position])
                .sum(),
        })
        .collect();
    let minimum_union_coverage = per_target
        .iter()
        .map(|item| item.union_coverage)
        .fold(1.0, f64::min);
    RobustCoverage {
        threshold: COVERAGE_THRESHOLD,
        method: "Union of each target's deterministic descending steady-state op prefix whose cumulative modeled contribution first reaches 80%; one-time packing and partition-planning setup are excluded and ties are resolved by original op position.",
        selected_op_indices: union
            .iter()
            .map(|position| analyses[0].ops[*position].index)
            .collect(),
        selected_op_count: union.len(),
        per_target,
        minimum_union_coverage,
    }
}

fn build_divergence(
    analyses: &[Analysis],
    target_components: &[Vec<FrontierComponents>],
    target_totals: &[f64],
    distributions: &[Vec<f64>],
    ranks: &[Vec<usize>],
    prefixes: &[Vec<usize>],
) -> DivergenceSummary {
    let mut pairs = Vec::new();
    for left in 0..analyses.len() {
        for right in left + 1..analyses.len() {
            let divergence = normalized_js_divergence(&distributions[left], &distributions[right]);
            let mut drivers: Vec<DivergenceDriver> = analyses[0]
                .ops
                .iter()
                .enumerate()
                .map(|(position, op)| {
                    let left_share = distributions[left][position];
                    let right_share = distributions[right][position];
                    let contribution = normalized_js_contribution(left_share, right_share);
                    let left_bound = analyses[left].ops[position].static_bound_guess.clone();
                    let right_bound = analyses[right].ops[position].static_bound_guess.clone();
                    let left_dominant =
                        steady_dominant_component(target_components[left][position]).to_string();
                    let right_dominant =
                        steady_dominant_component(target_components[right][position]).to_string();
                    DivergenceDriver {
                        op_index: op.index,
                        op_name: op.name.clone(),
                        normalized_js_contribution: contribution,
                        attribution_share: ratio(contribution, divergence),
                        left_contribution_share: left_share,
                        right_contribution_share: right_share,
                        signed_contribution_share_delta: right_share - left_share,
                        absolute_contribution_share_delta: (right_share - left_share).abs(),
                        left_rank: ranks[left][position],
                        right_rank: ranks[right][position],
                        rank_delta: ranks[left][position].abs_diff(ranks[right][position]),
                        bound_transition: left_bound != right_bound,
                        left_bound,
                        right_bound,
                        dominant_component_transition: left_dominant != right_dominant,
                        left_dominant_component: left_dominant,
                        right_dominant_component: right_dominant,
                        component_contribution_delta: component_contribution_delta(
                            target_components[left][position],
                            target_components[right][position],
                            target_totals[left],
                            target_totals[right],
                        ),
                    }
                })
                .collect();
            drivers.sort_by(|left, right| {
                compare_f64_desc(
                    left.normalized_js_contribution,
                    right.normalized_js_contribution,
                )
                .then_with(|| {
                    compare_f64_desc(
                        left.absolute_contribution_share_delta,
                        right.absolute_contribution_share_delta,
                    )
                })
                .then_with(|| left.op_index.cmp(&right.op_index))
            });
            let attribution_sum: f64 = drivers
                .iter()
                .map(|driver| driver.normalized_js_contribution)
                .sum();
            let mut prefix_sum = 0.0;
            let mut prefix_count = 0usize;
            if divergence > 0.0 {
                for driver in &drivers {
                    if prefix_sum / divergence >= COVERAGE_THRESHOLD {
                        break;
                    }
                    if driver.normalized_js_contribution <= 0.0 {
                        break;
                    }
                    prefix_sum += driver.normalized_js_contribution;
                    prefix_count += 1;
                }
            }
            let top_driver = drivers
                .first()
                .filter(|driver| driver.normalized_js_contribution > 0.0);
            pairs.push(TargetDivergence {
                left_target_id: analyses[left].target_profile.id.clone(),
                right_target_id: analyses[right].target_profile.id.clone(),
                normalized_jensen_shannon_divergence: divergence,
                coverage_prefix_jaccard: set_jaccard(&prefixes[left], &prefixes[right]),
                attribution_sum,
                attribution_prefix_threshold: COVERAGE_THRESHOLD,
                attribution_prefix_op_count: prefix_count,
                attribution_prefix_coverage: if divergence > 0.0 {
                    prefix_sum / divergence
                } else {
                    1.0
                },
                top_driver_op_index: top_driver.map(|driver| driver.op_index),
                top_driver_op_name: top_driver.map(|driver| driver.op_name.clone()),
                top_driver_attribution_share: top_driver
                    .map(|driver| driver.attribution_share)
                    .unwrap_or(0.0),
                bound_transition_op_count: drivers
                    .iter()
                    .filter(|driver| driver.bound_transition)
                    .count(),
                dominant_component_transition_op_count: drivers
                    .iter()
                    .filter(|driver| driver.dominant_component_transition)
                    .count(),
                drivers,
            });
        }
    }
    DivergenceSummary {
        method: "Jensen-Shannon divergence of per-op steady-state modeled-time contribution distributions, normalized by ln(2) to [0,1]. Each pair's divergence is exactly decomposed into non-negative per-op terms 0.5*(p*ln(p/m)+q*ln(q/m))/ln(2), m=(p+q)/2; the attribution prefix is the deterministic descending set first explaining 80% of pair divergence. Component deltas exclude one-time packing and partition-planning setup, then divide steady components by the steady target total. Hotspot overlap is set Jaccard over target-specific 80% steady-state coverage prefixes.",
        pair_count: pairs.len(),
        mean_normalized_jensen_shannon_divergence: mean(
            &pairs
                .iter()
                .map(|item| item.normalized_jensen_shannon_divergence)
                .collect::<Vec<_>>(),
        ),
        max_normalized_jensen_shannon_divergence: pairs
            .iter()
            .map(|item| item.normalized_jensen_shannon_divergence)
            .fold(0.0, f64::max),
        min_coverage_prefix_jaccard: pairs
            .iter()
            .map(|item| item.coverage_prefix_jaccard)
            .fold(1.0, f64::min),
        pairs,
    }
}

fn component_contribution_delta(
    left: FrontierComponents,
    right: FrontierComponents,
    left_total: f64,
    right_total: f64,
) -> ComponentContributionDelta {
    let values = [
        (
            "compute",
            ratio(right.compute_us, right_total) - ratio(left.compute_us, left_total),
        ),
        (
            "memory",
            ratio(right.memory_us, right_total) - ratio(left.memory_us, left_total),
        ),
        ("packing", 0.0),
        ("boundary", 0.0),
        (
            "fallback",
            ratio(right.fallback_us, right_total) - ratio(left.fallback_us, left_total),
        ),
    ];
    let (largest_absolute_component, largest_delta) = values
        .iter()
        .copied()
        .max_by(|left, right| {
            left.1
                .abs()
                .partial_cmp(&right.1.abs())
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(("none", 0.0));
    ComponentContributionDelta {
        compute: values[0].1,
        memory: values[1].1,
        packing: values[2].1,
        boundary: values[3].1,
        fallback: values[4].1,
        largest_absolute_component,
        largest_absolute_delta: largest_delta.abs(),
    }
}

fn build_interventions(
    analyses: &[Analysis],
    target_components: &[Vec<FrontierComponents>],
    target_steady_totals: &[f64],
    target_cold_totals: &[f64],
) -> Vec<FrontierIntervention> {
    [
        (
            "packing_removed",
            "Prepacked weights retained",
            "packing",
            "Removes only bottleneck_packing_us; model structure and base compute/memory terms are unchanged.",
        ),
        (
            "predicted_boundaries_removed",
            "Predicted partition-planning setup eliminated",
            "boundary",
            "Removes only the setup-only bottleneck_break_us profile constant from cold start; runtime materialization, copies, and per-inference launch latency are not observed.",
        ),
        (
            "predicted_fallback_removed",
            "Predicted fallback traffic eliminated",
            "fallback",
            "Removes only bottleneck_fallback_us; replacement kernel cost and actual provider assignment are not modeled.",
        ),
    ]
    .into_iter()
    .map(|(id, label, component, boundary)| {
        let per_target: Vec<InterventionTarget> = analyses
            .iter()
            .enumerate()
            .map(|(target_index, analysis)| {
                let recoverable_us: f64 = target_components[target_index]
                    .iter()
                    .map(|item| item.named(component))
                    .sum();
                let total = if component == "packing" || component == "boundary" {
                    target_cold_totals[target_index]
                } else {
                    target_steady_totals[target_index]
                };
                let remaining = (total - recoverable_us).max(0.0);
                InterventionTarget {
                    target_id: analysis.target_profile.id.clone(),
                    recoverable_us,
                    recoverable_share: ratio(recoverable_us, total),
                    upper_bound_speedup: if total > 0.0 && remaining > 0.0 {
                        total / remaining
                    } else {
                        1.0
                    },
                }
            })
            .collect();
        let shares: Vec<f64> = per_target
            .iter()
            .map(|item| item.recoverable_share)
            .collect();
        let speedups: Vec<f64> = per_target
            .iter()
            .map(|item| item.upper_bound_speedup)
            .collect();
        FrontierIntervention {
            id,
            label,
            removed_component: component,
            evidence_class: "ESTIMATED_COUNTERFACTUAL_UPPER_BOUND",
            min_recoverable_share: min_f64(&shares),
            median_recoverable_share: median(&shares),
            max_recoverable_share: max_f64(&shares),
            min_upper_bound_speedup: min_f64(&speedups),
            max_upper_bound_speedup: max_f64(&speedups),
            per_target,
            interpretation_boundary: boundary,
        }
    })
    .collect()
}

fn build_evidence_queue(
    analysis: &Analysis,
    ops: &[FrontierOp],
    union: &BTreeSet<usize>,
) -> Vec<EvidenceQueueItem> {
    let robust_ops: Vec<&FrontierOp> = ops
        .iter()
        .filter(|item| item.in_robust_coverage_union)
        .collect();
    let microkernel_ops: Vec<&FrontierOp> = robust_ops
        .iter()
        .copied()
        .filter(|item| {
            analysis
                .ops
                .iter()
                .find(|op| op.index == item.op_index)
                .is_some_and(|op| op.macs > 0.0 && op.xnnpack_supported)
        })
        .collect();
    let boundary_ops: Vec<&FrontierOp> = robust_ops
        .iter()
        .copied()
        .filter(|item| {
            analysis
                .ops
                .iter()
                .find(|op| op.index == item.op_index)
                .is_some_and(|op| op.xnnpack_chain_break)
        })
        .collect();
    vec![
        evidence_item(
            "runtime_placement",
            "Observed runtime execution assignment for the selected target",
            &robust_ops,
            "Static target profiles predict placement; a bound native runtime assignment is required to resolve it.",
        ),
        evidence_item(
            "executed_microkernel",
            "Executed lowering and microkernel identity for robust compute hotspots",
            &microkernel_ops,
            "Source candidates and target profiles do not identify the executed dispatch function.",
        ),
        evidence_item(
            "boundary_materialization",
            "Observed materialization and timing at predicted execution-domain boundaries",
            &boundary_ops,
            "Graph-edge payload and modeled break time do not prove a runtime copy or conversion.",
        ),
        EvidenceQueueItem {
            id: "target_coverage",
            evidence_needed: "Observed runs on every planning target represented by this frontier",
            decision_exposed_op_indices: union
                .iter()
                .map(|position| analysis.ops[*position].index)
                .collect(),
            decision_exposed_op_count: union.len(),
            max_single_target_contribution_share: robust_ops
                .iter()
                .map(|item| item.max_contribution_share)
                .fold(0.0, f64::max),
            evidence_boundary: "Cross-target stability is derived from planning profiles until each target has bound measurements.",
        },
    ]
}

fn evidence_item(
    id: &'static str,
    evidence_needed: &'static str,
    ops: &[&FrontierOp],
    evidence_boundary: &'static str,
) -> EvidenceQueueItem {
    EvidenceQueueItem {
        id,
        evidence_needed,
        decision_exposed_op_indices: ops.iter().map(|item| item.op_index).collect(),
        decision_exposed_op_count: ops.len(),
        max_single_target_contribution_share: ops
            .iter()
            .map(|item| item.max_contribution_share)
            .fold(0.0, f64::max),
        evidence_boundary,
    }
}

fn coverage_prefix(distribution: &[f64], threshold: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..distribution.len()).collect();
    order.sort_by(|left, right| {
        compare_f64_desc(distribution[*left], distribution[*right]).then_with(|| left.cmp(right))
    });
    let mut cumulative = 0.0;
    let mut selected = Vec::new();
    for position in order {
        if cumulative >= threshold {
            break;
        }
        if distribution[position] <= 0.0 {
            break;
        }
        selected.push(position);
        cumulative += distribution[position];
    }
    selected
}

fn rank_descending(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| {
        compare_f64_desc(values[*left], values[*right]).then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0; values.len()];
    for (rank, position) in order.into_iter().enumerate() {
        ranks[position] = rank + 1;
    }
    ranks
}

fn normalized_js_divergence(left: &[f64], right: &[f64]) -> f64 {
    let value = left
        .iter()
        .zip(right)
        .map(|(p, q)| normalized_js_contribution(*p, *q))
        .sum::<f64>();
    value.clamp(0.0, 1.0)
}

fn normalized_js_contribution(left: f64, right: f64) -> f64 {
    let middle = (left + right) * 0.5;
    let left_term = if left > 0.0 && middle > 0.0 {
        left * (left / middle).ln()
    } else {
        0.0
    };
    let right_term = if right > 0.0 && middle > 0.0 {
        right * (right / middle).ln()
    } else {
        0.0
    };
    (0.5 * (left_term + right_term) / std::f64::consts::LN_2).max(0.0)
}

fn set_jaccard(left: &[usize], right: &[usize]) -> f64 {
    let left: BTreeSet<usize> = left.iter().copied().collect();
    let right: BTreeSet<usize> = right.iter().copied().collect();
    let union = left.union(&right).count();
    if union == 0 {
        return 1.0;
    }
    left.intersection(&right).count() as f64 / union as f64
}

fn unique_sorted(values: impl Iterator<Item = String>) -> Vec<String> {
    values.collect::<BTreeSet<_>>().into_iter().collect()
}

fn finite_non_negative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
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

fn steady_dominant_component(components: FrontierComponents) -> &'static str {
    [
        ("compute", components.compute_us),
        ("memory", components.memory_us),
        ("fallback", components.fallback_us),
    ]
    .into_iter()
    .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
    .map(|item| item.0)
    .unwrap_or("none")
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

fn compare_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roofline_bound_is_assigned_once() {
        assert_eq!(active_roofline_components(7.0, 3.0), (7.0, 0.0));
        assert_eq!(active_roofline_components(3.0, 7.0), (0.0, 7.0));
        assert_eq!(active_roofline_components(7.0, 7.0), (7.0, 0.0));
    }

    #[test]
    fn target_ids_are_distinct_and_bounded() {
        assert_eq!(
            parse_target_ids(r#"["a","b","a"]"#).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(parse_target_ids(r#"["a"]"#).is_err());
        assert!(parse_target_ids(r#"["a","b","c","d","e","f","g","h","i"]"#).is_err());
    }

    #[test]
    fn coverage_prefix_is_deterministic_and_reaches_threshold() {
        let distribution = vec![0.4, 0.1, 0.3, 0.2];
        let selected = coverage_prefix(&distribution, 0.8);
        assert_eq!(selected, vec![0, 2, 3]);
        let coverage: f64 = selected.iter().map(|index| distribution[*index]).sum();
        assert!((coverage - 0.9).abs() < 1e-12);
    }

    #[test]
    fn normalized_js_divergence_has_exact_endpoints() {
        assert!(normalized_js_divergence(&[0.5, 0.5], &[0.5, 0.5]).abs() < 1e-12);
        assert!((normalized_js_divergence(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_js_attribution_is_non_negative_and_exact() {
        let left = [0.5, 0.3, 0.2];
        let right = [0.2, 0.3, 0.5];
        let contributions: Vec<f64> = left
            .iter()
            .zip(right)
            .map(|(p, q)| normalized_js_contribution(*p, q))
            .collect();
        assert!(contributions.iter().all(|value| *value >= 0.0));
        assert!(
            (contributions.iter().sum::<f64>() - normalized_js_divergence(&left, &right)).abs()
                < 1e-12
        );
        assert!(contributions[1].abs() < 1e-12);
    }

    #[test]
    fn steady_component_contribution_delta_excludes_cold_start_setup() {
        let left = FrontierComponents {
            compute_us: 20.0,
            memory_us: 30.0,
            packing_us: 0.0,
            boundary_us: 5.0,
            fallback_us: 0.0,
        };
        let right = FrontierComponents {
            compute_us: 60.0,
            memory_us: 20.0,
            packing_us: 20.0,
            boundary_us: 10.0,
            fallback_us: 0.0,
        };
        let delta = component_contribution_delta(left, right, 100.0, 200.0);
        assert!((delta.compute - 0.1).abs() < 1e-12);
        assert!((delta.memory + 0.2).abs() < 1e-12);
        assert_eq!(delta.packing, 0.0);
        assert_eq!(delta.boundary, 0.0);
        assert_eq!(delta.largest_absolute_component, "memory");
        assert!((delta.largest_absolute_delta - 0.2).abs() < 1e-12);
    }

    #[test]
    fn component_counterfactual_uses_only_named_component() {
        let components = FrontierComponents {
            compute_us: 30.0,
            memory_us: 50.0,
            packing_us: 10.0,
            boundary_us: 5.0,
            fallback_us: 5.0,
        };
        assert_eq!(components.total(), 100.0);
        assert_eq!(components.steady_total(), 85.0);
        assert_eq!(components.named("packing"), 10.0);
        assert_eq!(components.named("boundary"), 5.0);
        assert_eq!(components.named("fallback"), 5.0);
    }

    #[test]
    fn median_and_rank_ties_are_stable() {
        assert_eq!(median(&[0.4, 0.1, 0.3, 0.2]), 0.25);
        assert_eq!(rank_descending(&[2.0, 2.0, 1.0]), vec![1, 2, 3]);
    }
}
