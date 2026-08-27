use super::{OpInfo, TargetProfile};
use serde::Serialize;

const SCHEMA: &str = "deepbom.core_isolation_roofline.v1";

#[derive(Serialize)]
pub(super) struct CoreIsolationAnalysis {
    schema: String,
    status: String,
    evidence_class: String,
    target_profile_id: String,
    target_profile_sha256: String,
    performance_reference_core_count: Option<usize>,
    system_core_count_options: Vec<usize>,
    scenarios: Vec<CoreIsolationScenario>,
    method: String,
    isolation_evidence_boundary: String,
    unavailable_reason: Option<String>,
}

#[derive(Serialize)]
struct CoreIsolationScenario {
    scenario_id: String,
    system_core_count: usize,
    ai_assigned_core_count: usize,
    housekeeping_core_count: usize,
    allocation_class: String,
    int8_issue_ceiling_gops: f64,
    fp32_issue_ceiling_gops: f64,
    shared_memory_bandwidth_ceiling_gbps: f64,
    memory_bandwidth_scaling: String,
    int8_ridge_ops_per_byte: f64,
    fp32_ridge_ops_per_byte: f64,
    theoretical_compute_floor_us: f64,
    theoretical_memory_floor_us: f64,
    theoretical_roofline_floor_us: f64,
    utilization_adjusted_compute_floor_us: f64,
    utilization_adjusted_roofline_estimate_us: f64,
    compute_dominant_floor_us: f64,
    memory_dominant_floor_us: f64,
    predicted_runtime_overhead_us: f64,
    steady_state_estimate_us: f64,
    one_time_packing_estimate_us: f64,
    cold_start_estimate_us: f64,
    assessed_op_count: usize,
    compute_dominant_op_count: usize,
    memory_dominant_op_count: usize,
    estimate_ratio_vs_one_ai_core: f64,
}

#[derive(Default)]
struct ScenarioTotals {
    theoretical_compute_floor_us: f64,
    theoretical_memory_floor_us: f64,
    theoretical_roofline_floor_us: f64,
    utilization_adjusted_compute_floor_us: f64,
    utilization_adjusted_roofline_estimate_us: f64,
    compute_dominant_floor_us: f64,
    memory_dominant_floor_us: f64,
    predicted_runtime_overhead_us: f64,
    one_time_packing_estimate_us: f64,
    compute_dominant_op_count: usize,
    memory_dominant_op_count: usize,
}

pub(super) fn analyze(ops: &[OpInfo], target: &TargetProfile) -> CoreIsolationAnalysis {
    let reference_cores = target.performance_reference_core_count;
    let system_options = system_core_options(target);
    let unavailable_reason = if reference_cores.is_none() {
        Some("The target profile does not bind the core count represented by effective_peak_gops. Create a custom target with an exact system core count and peak-reference core count before comparing isolated-core scenarios.".to_string())
    } else if system_options.is_empty() {
        Some("The target profile does not bind an exact supported system core-count option. Core partition performance is intentionally not inferred from the ISA label alone.".to_string())
    } else {
        None
    };

    let mut scenarios = Vec::new();
    if let Some(reference) = reference_cores {
        if reference > 0 {
            for system_cores in &system_options {
                let mut group = Vec::new();
                for assigned_cores in 1..=*system_cores {
                    group.push(build_scenario(
                        ops,
                        target,
                        reference,
                        *system_cores,
                        assigned_cores,
                    ));
                }
                let one_core = group
                    .first()
                    .map(|scenario| scenario.steady_state_estimate_us)
                    .unwrap_or(0.0);
                for scenario in &mut group {
                    scenario.estimate_ratio_vs_one_ai_core =
                        if scenario.steady_state_estimate_us > 0.0 {
                            one_core / scenario.steady_state_estimate_us
                        } else {
                            1.0
                        };
                }
                scenarios.extend(group);
            }
        }
    }

    CoreIsolationAnalysis {
        schema: SCHEMA.to_string(),
        status: if scenarios.is_empty() { "not_assessed" } else { "assessed" }.to_string(),
        evidence_class: "ESTIMATED_STATIC_RESOURCE_PARTITION".to_string(),
        target_profile_id: target.id.clone(),
        target_profile_sha256: target.profile_sha256.clone(),
        performance_reference_core_count: reference_cores,
        system_core_count_options: system_options,
        scenarios,
        method: "For k assigned homogeneous cores, INT8 issue ceiling = target effective_peak_gops x k / performance_reference_core_count; FP32 ceiling = INT8 ceiling / fp32_compute_factor. This is a linear issue-ceiling counterfactual with the profile's kernel utilization held constant, not an observed scaling law. For each op, arithmetic floor = max(op.ops, 2 x op.macs) / precision-specific issue ceiling; utilization-adjusted compute estimate additionally divides by the bound scalar or kernel-family utilization. Memory floor = logical estimated bytes / the unchanged full-interface bandwidth ceiling. Per-op roofline values are max(compute, memory), then summed. Predicted delegate-break/fallback and one-time packing terms remain separate. Memory bandwidth is never multiplied by core count.".to_string(),
        isolation_evidence_boundary: "These scenarios change a static compute-resource denominator only. The full-interface bandwidth is an optimistic ceiling at every allocation; per-core attainable bandwidth, parallel efficiency, worker occupancy, and scaling loss are unbound. The analysis does not observe scheduler affinity, cpuset exclusivity, IRQ routing, nohz_full, frequency governor, thermal state, memory contention, XNNPACK worker placement, or target-device latency. A full-system-core scenario has no housekeeping core and is explicitly not an isolated-core claim.".to_string(),
        unavailable_reason,
    }
}

fn system_core_options(target: &TargetProfile) -> Vec<usize> {
    match (target.core_count_min, target.core_count_max) {
        (Some(minimum), Some(maximum)) if minimum > 0 && maximum >= minimum => {
            if minimum == maximum {
                vec![minimum]
            } else {
                // A range on a product-family profile represents documented
                // variants, not proof that every intermediate count exists.
                vec![minimum, maximum]
            }
        }
        _ => Vec::new(),
    }
}

fn build_scenario(
    ops: &[OpInfo],
    target: &TargetProfile,
    reference_cores: usize,
    system_cores: usize,
    assigned_cores: usize,
) -> CoreIsolationScenario {
    let core_fraction = assigned_cores as f64 / reference_cores as f64;
    let int8_peak = (target.effective_peak_gops * core_fraction).max(f64::MIN_POSITIVE);
    let fp32_peak = (int8_peak / target.fp32_compute_factor.max(0.1)).max(f64::MIN_POSITIVE);
    let bandwidth = target.effective_memory_bandwidth_gbps.max(0.25);
    let totals = scenario_totals(ops, target, int8_peak, fp32_peak, bandwidth);
    let steady =
        totals.utilization_adjusted_roofline_estimate_us + totals.predicted_runtime_overhead_us;

    CoreIsolationScenario {
        scenario_id: format!("system-{system_cores}-ai-{assigned_cores}"),
        system_core_count: system_cores,
        ai_assigned_core_count: assigned_cores,
        housekeeping_core_count: system_cores.saturating_sub(assigned_cores),
        allocation_class: if assigned_cores < system_cores {
            "exclusive_isolation_candidate"
        } else {
            "full_set_non_isolated_baseline"
        }
        .to_string(),
        int8_issue_ceiling_gops: int8_peak,
        fp32_issue_ceiling_gops: fp32_peak,
        shared_memory_bandwidth_ceiling_gbps: bandwidth,
        memory_bandwidth_scaling: "shared_interface_ceiling_not_core_scaled".to_string(),
        int8_ridge_ops_per_byte: int8_peak / bandwidth,
        fp32_ridge_ops_per_byte: fp32_peak / bandwidth,
        theoretical_compute_floor_us: totals.theoretical_compute_floor_us,
        theoretical_memory_floor_us: totals.theoretical_memory_floor_us,
        theoretical_roofline_floor_us: totals.theoretical_roofline_floor_us,
        utilization_adjusted_compute_floor_us: totals.utilization_adjusted_compute_floor_us,
        utilization_adjusted_roofline_estimate_us: totals.utilization_adjusted_roofline_estimate_us,
        compute_dominant_floor_us: totals.compute_dominant_floor_us,
        memory_dominant_floor_us: totals.memory_dominant_floor_us,
        predicted_runtime_overhead_us: totals.predicted_runtime_overhead_us,
        steady_state_estimate_us: steady,
        one_time_packing_estimate_us: totals.one_time_packing_estimate_us,
        cold_start_estimate_us: steady + totals.one_time_packing_estimate_us,
        assessed_op_count: ops.len(),
        compute_dominant_op_count: totals.compute_dominant_op_count,
        memory_dominant_op_count: totals.memory_dominant_op_count,
        estimate_ratio_vs_one_ai_core: 1.0,
    }
}

fn scenario_totals(
    ops: &[OpInfo],
    target: &TargetProfile,
    int8_peak: f64,
    fp32_peak: f64,
    bandwidth: f64,
) -> ScenarioTotals {
    let mut totals = ScenarioTotals::default();
    for op in ops {
        let peak = if op.quantized_compute_path {
            int8_peak
        } else {
            fp32_peak
        };
        let utilization = target
            .compute_utilization_by_kernel_class
            .get(&op.compute_kernel_class)
            .copied()
            .unwrap_or(target.compute_utilization_factor)
            .clamp(0.01, 1.0);
        let operation_count = op.ops.max(op.macs * 2.0).max(0.0);
        let bytes = op.estimated_bytes.max(0.0);
        let theoretical_compute = micros(operation_count, peak);
        let adjusted_compute = micros(operation_count, peak * utilization);
        let memory = micros(bytes, bandwidth);
        let theoretical_roofline = theoretical_compute.max(memory);
        let adjusted_roofline = adjusted_compute.max(memory);

        totals.theoretical_compute_floor_us += theoretical_compute;
        totals.theoretical_memory_floor_us += memory;
        totals.theoretical_roofline_floor_us += theoretical_roofline;
        totals.utilization_adjusted_compute_floor_us += adjusted_compute;
        totals.utilization_adjusted_roofline_estimate_us += adjusted_roofline;
        if adjusted_compute >= memory {
            totals.compute_dominant_floor_us += adjusted_roofline;
            totals.compute_dominant_op_count += 1;
        } else {
            totals.memory_dominant_floor_us += adjusted_roofline;
            totals.memory_dominant_op_count += 1;
        }
        if op.xnnpack_chain_break {
            totals.predicted_runtime_overhead_us +=
                (op.chain_break_overhead_us_low + op.chain_break_overhead_us_high) / 2.0;
        }
        if op.xnnpack_chain_id < 0 || op.xnnpack_chain_break {
            totals.predicted_runtime_overhead_us += micros(bytes * 0.5, bandwidth);
        }
        totals.one_time_packing_estimate_us += op.weight_packing_overhead_us.max(0.0);
    }
    totals
}

fn micros(work: f64, rate_giga_per_second: f64) -> f64 {
    if work <= 0.0 {
        0.0
    } else {
        work / (rate_giga_per_second.max(f64::MIN_POSITIVE) * 1e9) * 1e6
    }
}

pub(super) fn render_csv(analysis: &CoreIsolationAnalysis) -> String {
    let mut output = String::from("scenario_id,system_cores,ai_cores,housekeeping_cores,allocation_class,int8_issue_ceiling_gops,fp32_issue_ceiling_gops,shared_bandwidth_ceiling_gbps,theoretical_roofline_floor_us,utilization_adjusted_roofline_estimate_us,predicted_runtime_overhead_us,steady_state_estimate_us,one_time_packing_estimate_us,cold_start_estimate_us,estimate_ratio_vs_one_ai_core,evidence_class\n");
    for row in &analysis.scenarios {
        output.push_str(&format!(
            "{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}\n",
            row.scenario_id,
            row.system_core_count,
            row.ai_assigned_core_count,
            row.housekeeping_core_count,
            row.allocation_class,
            row.int8_issue_ceiling_gops,
            row.fp32_issue_ceiling_gops,
            row.shared_memory_bandwidth_ceiling_gbps,
            row.theoretical_roofline_floor_us,
            row.utilization_adjusted_roofline_estimate_us,
            row.predicted_runtime_overhead_us,
            row.steady_state_estimate_us,
            row.one_time_packing_estimate_us,
            row.cold_start_estimate_us,
            row.estimate_ratio_vs_one_ai_core,
            analysis.evidence_class,
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_options_do_not_invent_intermediate_product_variants() {
        let mut target =
            crate::target_profiles::target_profile("zynq_ultrascale_plus_a53").unwrap();
        target.performance_reference_core_count = Some(4);
        assert_eq!(system_core_options(&target), vec![2, 4]);
    }
}
