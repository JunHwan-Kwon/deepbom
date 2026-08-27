use crate::block_inventory::BlockRecord;
use crate::{
    bytes_per_type, compute_bottleneck_estimates, compute_tensor_arena_plan,
    compute_tensor_liveness, estimate_op, logical_cache_payload_for_op, Analysis, OpInfo,
    TargetProfile, TensorInfo, L1_WORKING_SET_WATCH_RATIO,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const SCHEMA: &str = "deepbom.redesign_projection.v1.1";

#[derive(Clone, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct RedesignRequest {
    pub(crate) schema: String,
    pub(crate) source_sha256: String,
    pub(crate) input_height: Option<usize>,
    pub(crate) input_width: Option<usize>,
    pub(crate) width_multiplier: f64,
    pub(crate) activation_dtype: String,
    pub(crate) block_edits: Vec<RedesignBlockEdit>,
}

impl Default for RedesignRequest {
    fn default() -> Self {
        Self {
            schema: "deepbom.redesign_request.v1".to_string(),
            source_sha256: String::new(),
            input_height: None,
            input_width: None,
            width_multiplier: 1.0,
            activation_dtype: "source".to_string(),
            block_edits: Vec::new(),
        }
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct RedesignBlockEdit {
    pub(crate) block_id: String,
    pub(crate) output_channels: Option<usize>,
    pub(crate) expand_ratio: Option<f64>,
    pub(crate) repeat: Option<usize>,
    pub(crate) kernel_size: Option<usize>,
}

#[derive(Clone, Serialize)]
pub(crate) struct RedesignProjection {
    schema: String,
    status: String,
    projection_status: String,
    source: RedesignSourceBinding,
    request: RedesignRequestEcho,
    metrics: RedesignMetricComparison,
    cache_points: Vec<RedesignCachePoint>,
    block_diffs: Vec<RedesignBlockDiff>,
    op_projections: Vec<RedesignOpProjection>,
    propagation_edges: Vec<RedesignPropagationEdge>,
    impact_summary: RedesignImpactSummary,
    implementation_plan: RedesignImplementationPlan,
    constraints: Vec<RedesignConstraint>,
    projection_coverage: RedesignCoverage,
    evaluated: Vec<String>,
    not_evaluated: Vec<String>,
    required_next_steps: Vec<String>,
    method: String,
    interpretation_boundary: String,
}

#[derive(Clone, Serialize)]
struct RedesignSourceBinding {
    filename: String,
    sha256_before: String,
    sha256_after: String,
    loaded_source_bytes_unchanged: bool,
    target_id: String,
    target_profile_sha256: String,
    block_inventory_schema: String,
}

#[derive(Clone, Serialize)]
struct RedesignRequestEcho {
    input_height: usize,
    input_width: usize,
    width_multiplier: f64,
    activation_dtype: String,
    block_edit_count: usize,
}

#[derive(Clone, Serialize)]
struct RedesignMetricComparison {
    source: RedesignMetrics,
    projected: RedesignMetrics,
    delta: RedesignMetricDelta,
}

#[derive(Clone, Default, Serialize)]
struct RedesignMetrics {
    operator_count: usize,
    macs: f64,
    operations: f64,
    parameter_elements: usize,
    serialized_parameter_bytes: usize,
    modeled_latency_ms: f64,
    latency_evidence_class: String,
    l1_max_ratio: Option<f64>,
    l1_watch_count: usize,
    peak_live_activation_bytes: Option<usize>,
    arena_bytes: Option<usize>,
    predicted_break_count: Option<usize>,
    delegation_evidence_class: String,
}

#[derive(Clone, Serialize)]
struct RedesignMetricDelta {
    mac_percent: Option<f64>,
    operations_percent: Option<f64>,
    parameter_percent: Option<f64>,
    modeled_latency_percent: Option<f64>,
    peak_live_activation_percent: Option<f64>,
    arena_percent: Option<f64>,
}

#[derive(Clone, Serialize)]
struct RedesignCachePoint {
    op_index: usize,
    op_name: String,
    block_id: String,
    source_width: Option<usize>,
    source_channels: Option<usize>,
    source_logical_row_payload_bytes: Option<usize>,
    source_l1_ratio: Option<f64>,
    projected_width: Option<usize>,
    projected_channels: Option<usize>,
    projected_logical_row_payload_bytes: Option<usize>,
    projected_l1_ratio: Option<f64>,
    evidence_class: String,
}

#[derive(Clone, Serialize)]
struct RedesignBlockDiff {
    block_id: String,
    display_name: String,
    changes: Vec<String>,
    source_macs: f64,
    projected_macs: f64,
    source_l1_max_ratio: Option<f64>,
    projected_l1_max_ratio: Option<f64>,
}

#[derive(Clone, Serialize)]
struct RedesignTensorContract {
    tensor_index: usize,
    shape: Vec<i32>,
    dtype: String,
    constant: bool,
}

#[derive(Clone, Serialize)]
struct RedesignOpProjection {
    op_index: usize,
    op_name: String,
    block_id: String,
    change_class: String,
    shape_rule_status: String,
    direct_edit_fields: Vec<String>,
    propagation_source_op_indices: Vec<usize>,
    related_op_indices: Vec<usize>,
    source_inputs: Vec<RedesignTensorContract>,
    projected_inputs: Vec<RedesignTensorContract>,
    source_outputs: Vec<RedesignTensorContract>,
    projected_outputs: Vec<RedesignTensorContract>,
    source_macs: f64,
    projected_macs: f64,
    source_steady_us: f64,
    projected_steady_us: f64,
    source_cold_us: f64,
    projected_cold_us: f64,
    source_l1_ratio: Option<f64>,
    projected_l1_ratio: Option<f64>,
}

#[derive(Clone, Serialize)]
struct RedesignPropagationEdge {
    tensor_index: usize,
    producer_op_index: usize,
    consumer_op_index: usize,
    source_shape: Vec<i32>,
    projected_shape: Vec<i32>,
    source_dtype: String,
    projected_dtype: String,
    changed: bool,
    change_class: String,
}

#[derive(Clone, Serialize)]
struct RedesignImpactSummary {
    direct_edit_op_count: usize,
    propagated_op_count: usize,
    global_projection_op_count: usize,
    unchanged_op_count: usize,
    changed_tensor_count: usize,
    changed_edge_count: usize,
    unresolved_contract_count: usize,
}

#[derive(Clone, Serialize)]
struct RedesignImplementationPlan {
    schema: String,
    status: String,
    evidence_class: String,
    model_inputs: Vec<RedesignImplementationTensor>,
    model_outputs: Vec<RedesignImplementationTensor>,
    nodes: Vec<RedesignImplementationNode>,
    exact_codegen_op_count: usize,
    scaffold_codegen_op_count: usize,
    unsupported_codegen_op_count: usize,
    non_materialized_repeat_edit_count: usize,
    exportable: bool,
    mapped_source_layer_count: usize,
    framework_targets: Vec<String>,
    method: String,
    interpretation_boundary: String,
}

#[derive(Clone, Serialize)]
struct RedesignImplementationTensor {
    tensor_index: usize,
    artifact_name: String,
    projected_shape: Vec<i32>,
    dtype: String,
}

#[derive(Clone, Serialize)]
struct RedesignImplementationNode {
    op_index: usize,
    op_name: String,
    generated_symbol: String,
    block_id: String,
    source_layer_ref: String,
    source_layer_evidence_class: String,
    source_tensor_refs: Vec<String>,
    module_kind: String,
    codegen_status: String,
    codegen_reason: String,
    activation_inputs: Vec<usize>,
    outputs: Vec<usize>,
    input_shapes: Vec<Vec<i32>>,
    output_shapes: Vec<Vec<i32>>,
    input_channels: Option<usize>,
    output_channels: Option<usize>,
    kernel_height: Option<usize>,
    kernel_width: Option<usize>,
    stride_height: Option<usize>,
    stride_width: Option<usize>,
    concatenation_axis_nhwc: Option<usize>,
    fused_activation: String,
}

#[derive(Serialize)]
pub(crate) struct RedesignParetoSearch {
    schema: String,
    status: String,
    source_sha256: String,
    target_id: String,
    evaluated_candidate_count: usize,
    accepted_candidate_count: usize,
    rejected_candidate_count: usize,
    frontier_candidate_count: usize,
    candidates: Vec<RedesignParetoCandidate>,
    objectives: Vec<String>,
    method: String,
    interpretation_boundary: String,
}

#[derive(Clone, Serialize)]
struct RedesignParetoCandidate {
    candidate_id: String,
    pareto_optimal: bool,
    status: String,
    request: RedesignRequest,
    retained_structure_proxy: f64,
    modeled_latency_ms: f64,
    macs: f64,
    parameter_elements: usize,
    peak_live_activation_bytes: Option<usize>,
    l1_max_ratio: Option<f64>,
    error_constraint_count: usize,
    warning_constraint_count: usize,
    projection_coverage_status: String,
}

#[derive(Clone, Serialize)]
struct RedesignConstraint {
    severity: String,
    code: String,
    scope: String,
    detail: String,
    evidence_class: String,
}

#[derive(Clone, Serialize)]
struct RedesignCoverage {
    status: String,
    op_count: usize,
    exact_shape_rule_op_count: usize,
    scaled_shape_fallback_op_count: usize,
    unassessed_op_count: usize,
    repeat_edit_count: usize,
}

#[derive(Clone, Copy)]
enum ShapeRuleStatus {
    Exact,
    ScaledFallback,
    Unassessed,
}

impl ShapeRuleStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ScaledFallback => "serialized_shape_scaling",
            Self::Unassessed => "not_assessed",
        }
    }
}

#[derive(Clone, Copy)]
struct ShapeProjection {
    source_h: usize,
    source_w: usize,
    projected_h: usize,
    projected_w: usize,
    width_multiplier: f64,
    alignment: usize,
}

pub(crate) fn build_redesign_projection(
    bytes: &[u8],
    analysis: &Analysis,
    request: RedesignRequest,
) -> Result<RedesignProjection, String> {
    validate_request(analysis, &request)?;
    let source_hash = sha256(bytes);
    if !request.source_sha256.is_empty()
        && !request.source_sha256.eq_ignore_ascii_case(&source_hash)
    {
        return Err(
            "Redesign source SHA-256 does not match the loaded artifact bytes.".to_string(),
        );
    }

    let source_input = analysis
        .inputs
        .iter()
        .find(|tensor| tensor.shape.len() == 4)
        .ok_or_else(|| "Redesign requires a statically shaped rank-4 TFLite input.".to_string())?;
    let source_h = positive(source_input.shape[1])
        .ok_or_else(|| "Source input height is dynamic or invalid.".to_string())?;
    let source_w = positive(source_input.shape[2])
        .ok_or_else(|| "Source input width is dynamic or invalid.".to_string())?;
    let projected_h = request.input_height.unwrap_or(source_h);
    let projected_w = request.input_width.unwrap_or(source_w);
    if !(1..=8192).contains(&projected_h) || !(1..=8192).contains(&projected_w) {
        return Err("Projected input dimensions must be in 1..=8192.".to_string());
    }

    let edits = request
        .block_edits
        .iter()
        .map(|edit| (edit.block_id.clone(), edit.clone()))
        .collect::<HashMap<_, _>>();
    let block_by_op = block_ownership(&analysis.block_inventory.blocks);
    let output_indices = analysis
        .output_tensor_indices
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .collect::<HashSet<_>>();
    let input_indices = analysis
        .input_tensor_indices
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .collect::<HashSet<_>>();
    let mut tensors = analysis.tensors.clone();
    let mut ops = analysis.ops.clone();
    let activation_dtype = request.activation_dtype.to_ascii_lowercase();
    let target = &analysis.target_profile;
    let structural_edit =
        request_has_effective_edit(&request, source_h, source_w, projected_h, projected_w);

    if structural_edit {
        project_input_and_storage(
            &mut tensors,
            &input_indices,
            source_h,
            source_w,
            projected_h,
            projected_w,
            &activation_dtype,
        );
    }

    let mut exact_shape_rule_op_count = 0usize;
    let mut scaled_shape_fallback_op_count = 0usize;
    let mut unassessed_op_count = 0usize;
    let mut constraints = Vec::<RedesignConstraint>::new();
    let mut block_projected_macs = BTreeMap::<String, f64>::new();
    let mut block_projected_l1 = BTreeMap::<String, f64>::new();
    let mut shape_rule_status_by_op = BTreeMap::<usize, ShapeRuleStatus>::new();

    for op in &mut ops {
        let owner = block_by_op.get(&op.index).copied();
        let edit = owner.and_then(|block| edits.get(&block.block_id));
        let status = if structural_edit {
            project_op_shape(
                op,
                &analysis.ops,
                &analysis.tensors,
                &mut tensors,
                source_h,
                source_w,
                projected_h,
                projected_w,
                request.width_multiplier,
                target.channel_alignment_multiple.max(1),
                &input_indices,
                &output_indices,
                owner,
                edit,
            )
        } else {
            ShapeRuleStatus::Exact
        };
        match status {
            ShapeRuleStatus::Exact => exact_shape_rule_op_count += 1,
            ShapeRuleStatus::ScaledFallback => scaled_shape_fallback_op_count += 1,
            ShapeRuleStatus::Unassessed => unassessed_op_count += 1,
        }
        shape_rule_status_by_op.insert(op.index, status);
        if structural_edit {
            project_constant_storage(op, &mut tensors, &activation_dtype);
            let output_shapes = op
                .outputs
                .iter()
                .filter_map(|index| usize::try_from(*index).ok())
                .filter_map(|index| tensors.get(index))
                .map(|tensor| tensor.shape.clone())
                .collect::<Vec<_>>();
            let (macs, operations, estimated_bytes, input_strip) =
                estimate_op(&op.name, &op.inputs, &op.outputs, &tensors);
            let source_cache = analysis
                .ops
                .iter()
                .find(|source| source.index == op.index)
                .map(|source| &source.cache_payload);
            let dilation_h = source_cache
                .and_then(|cache| {
                    cache
                        .effective_kernel_height
                        .zip(cache.kernel_height)
                        .and_then(|(effective, kernel)| {
                            (kernel > 1).then(|| (effective.saturating_sub(1)) / (kernel - 1))
                        })
                })
                .unwrap_or(1)
                .max(1);
            let cache_payload = logical_cache_payload_for_op(
                &op.name,
                &op.inputs,
                &op.outputs,
                &tensors,
                dilation_h,
            );
            op.output_shapes = output_shapes;
            op.macs = macs;
            op.ops = operations;
            op.estimated_bytes = estimated_bytes;
            op.row_working_set_bytes = cache_payload
                .input_strip_bytes
                .map(|value| value as f64)
                .unwrap_or(input_strip);
            op.row_working_set_ratio = if target.l1_data_bytes > 0 {
                op.row_working_set_bytes / target.l1_data_bytes as f64
            } else {
                0.0
            };
            op.cache_payload = cache_payload;
            op.quantized_path =
                activation_dtype == "int8" || (activation_dtype == "source" && op.quantized_path);
            op.quantized_compute_path = activation_dtype == "int8"
                || (activation_dtype == "source" && op.quantized_compute_path);
        }
        if let Some(block) = owner {
            *block_projected_macs
                .entry(block.block_id.clone())
                .or_default() += op.macs;
            if let Some(bytes) = op.cache_payload.logical_row_payload_bytes {
                if target.l1_data_bytes > 0 {
                    let ratio = bytes as f64 / target.l1_data_bytes as f64;
                    block_projected_l1
                        .entry(block.block_id.clone())
                        .and_modify(|current| *current = current.max(ratio))
                        .or_insert(ratio);
                }
            }
        }
    }
    if structural_edit {
        compute_bottleneck_estimates(&mut ops, target);
    }

    let repeat_edits =
        validate_and_apply_repeat_edits(&analysis.block_inventory.blocks, &edits, &mut constraints);
    let repeat_mac_delta = repeat_edits
        .iter()
        .map(|(block, extra)| {
            let projected_once = block_projected_macs
                .get(&block.block_id)
                .copied()
                .unwrap_or(block.aggregates.macs);
            projected_once * *extra as f64
        })
        .sum::<f64>();
    let repeat_op_delta = repeat_edits
        .iter()
        .map(|(block, extra)| block.aggregates.op_count.saturating_mul(*extra))
        .sum::<usize>();
    let repeat_parameter_delta = repeat_edits
        .iter()
        .map(|(block, extra)| block.aggregates.parameter_elements.saturating_mul(*extra))
        .sum::<usize>();
    let repeat_serialized_parameter_delta = repeat_edits
        .iter()
        .map(|(block, extra)| {
            block
                .aggregates
                .serialized_parameter_bytes
                .saturating_mul(*extra)
        })
        .sum::<usize>();
    let repeat_latency_delta = repeat_edits
        .iter()
        .map(|(block, extra)| block.aggregates.modeled_time_ms * *extra as f64)
        .sum::<f64>();

    let source_metrics = source_metrics(analysis);
    let mut projected_metrics = if structural_edit {
        projected_metrics(&ops, &tensors, analysis, &activation_dtype, true)
    } else {
        source_metrics.clone()
    };
    projected_metrics.operator_count = projected_metrics
        .operator_count
        .saturating_add(repeat_op_delta);
    projected_metrics.macs += repeat_mac_delta;
    projected_metrics.parameter_elements = projected_metrics
        .parameter_elements
        .saturating_add(repeat_parameter_delta);
    projected_metrics.serialized_parameter_bytes = projected_metrics
        .serialized_parameter_bytes
        .saturating_add(repeat_serialized_parameter_delta);
    projected_metrics.modeled_latency_ms += repeat_latency_delta;
    if !repeat_edits.is_empty() {
        projected_metrics.peak_live_activation_bytes = None;
        projected_metrics.arena_bytes = None;
        constraints.push(RedesignConstraint {
            severity: "info".to_string(),
            code: "RD-MEM-REPEAT-001".to_string(),
            scope: "model".to_string(),
            detail: "Peak liveness and arena totals are suppressed after repeat edits because the hypothetical duplicated tensor lifetimes are not materialized in a serialized graph.".to_string(),
            evidence_class: "NOT_ASSESSABLE".to_string(),
        });
    }

    append_global_constraints(
        analysis,
        &request,
        projected_h,
        projected_w,
        source_h,
        source_w,
        &tensors,
        &ops,
        &mut constraints,
    );
    append_merge_contract_constraints(&tensors, &ops, &mut constraints);
    let block_diffs = build_block_diffs(
        &analysis.block_inventory.blocks,
        &edits,
        &block_projected_macs,
        &block_projected_l1,
        request.width_multiplier,
        projected_h,
        projected_w,
        source_h,
        source_w,
        &activation_dtype,
    );
    let cache_points = build_cache_points(
        &analysis.ops,
        &ops,
        &analysis.block_inventory.blocks,
        target,
    );
    let (op_projections, propagation_edges, impact_summary) = build_projection_ledger(
        &analysis.ops,
        &ops,
        &analysis.tensors,
        &tensors,
        &analysis.block_inventory.blocks,
        &edits,
        &shape_rule_status_by_op,
        request.width_multiplier,
        projected_h,
        projected_w,
        source_h,
        source_w,
        &activation_dtype,
        &constraints,
        target,
    );
    let implementation_plan = build_implementation_plan(
        &ops,
        &tensors,
        &block_by_op,
        &input_indices,
        &output_indices,
        repeat_edits.len(),
    );
    let coverage_status = if unassessed_op_count > 0 {
        "partial"
    } else if scaled_shape_fallback_op_count > 0 {
        "assessed_with_serialized_shape_scaling"
    } else {
        "assessed"
    };
    let status = if constraints.iter().any(|item| item.severity == "error") {
        "blocked"
    } else if coverage_status == "partial" {
        "partial"
    } else {
        "assessed"
    };
    let source_hash_after = sha256(bytes);
    Ok(RedesignProjection {
        schema: SCHEMA.to_string(),
        status: status.to_string(),
        projection_status: "PROJECTED_UNTRAINED".to_string(),
        source: RedesignSourceBinding {
            filename: analysis.filename.clone(),
            sha256_before: source_hash.clone(),
            sha256_after: source_hash_after.clone(),
            loaded_source_bytes_unchanged: source_hash == source_hash_after,
            target_id: target.id.clone(),
            target_profile_sha256: target.profile_sha256.clone(),
            block_inventory_schema: analysis.block_inventory.schema.clone(),
        },
        request: RedesignRequestEcho {
            input_height: projected_h,
            input_width: projected_w,
            width_multiplier: request.width_multiplier,
            activation_dtype: activation_dtype.clone(),
            block_edit_count: request.block_edits.len(),
        },
        metrics: RedesignMetricComparison {
            delta: metric_delta(&source_metrics, &projected_metrics),
            source: source_metrics,
            projected: projected_metrics,
        },
        cache_points,
        block_diffs,
        op_projections,
        propagation_edges,
        impact_summary,
        implementation_plan,
        constraints,
        projection_coverage: RedesignCoverage {
            status: coverage_status.to_string(),
            op_count: ops.len(),
            exact_shape_rule_op_count,
            scaled_shape_fallback_op_count,
            unassessed_op_count,
            repeat_edit_count: repeat_edits.len(),
        },
        evaluated: vec![
            "serialized-shape propagation and supported op shape contracts".to_string(),
            "MAC and arithmetic-operation projection".to_string(),
            "projected constant element count and storage payload".to_string(),
            "logical input-strip/output-row cache payload against the selected L1/L2 references"
                .to_string(),
            "target-profile modeled latency".to_string(),
            "declared tensor liveness and pinned arena-planner projection when topology is unchanged"
                .to_string(),
            "node-level source/projected contracts and changed producer-tensor-consumer propagation edges"
                .to_string(),
        ],
        not_evaluated: vec![
            "accuracy, calibration, robustness, fairness, clinical performance, and safety"
                .to_string(),
            "training convergence or equivalence to the source model".to_string(),
            "executed delegate partition, lowering, microkernel, cache residency, or device latency"
                .to_string(),
            "regulatory equivalence or clearance impact".to_string(),
        ],
        required_next_steps: vec![
            "materialize the proposed architecture in a training framework".to_string(),
            "retrain or fine-tune with the governed dataset and preprocessing contract".to_string(),
            "validate task metrics, calibration, robustness, and subgroup behavior against the source"
                .to_string(),
            "export a new deployment artifact and rerun the complete static and runtime audit"
                .to_string(),
        ],
        method: "The source artifact is re-analyzed in WASM; edits are applied to cloned tensor/operator metadata. Supported shape rules are propagated deterministically, op arithmetic and logical payload are recomputed, and unchanged-topology liveness/arena planning is rerun. Unsupported transformations remain explicit residuals.".to_string(),
        interpretation_boundary: "This object is a non-trained structural projection, not a transformed model, accuracy forecast, deployable artifact, or runtime measurement. It never mutates the loaded source bytes, source analysis, target profile, findings, or report binding.".to_string(),
    })
}

fn build_implementation_plan(
    projected_ops: &[OpInfo],
    projected_tensors: &[TensorInfo],
    block_by_op: &HashMap<usize, &BlockRecord>,
    input_indices: &HashSet<usize>,
    output_indices: &HashSet<usize>,
    repeat_edit_count: usize,
) -> RedesignImplementationPlan {
    let model_inputs = implementation_tensors(projected_tensors, input_indices);
    let model_outputs = implementation_tensors(projected_tensors, output_indices);
    let mut exact_codegen_op_count = 0usize;
    let mut scaffold_codegen_op_count = 0usize;
    let mut unsupported_codegen_op_count = 0usize;
    let mut mapped_source_layer_count = 0usize;
    let mut nodes = Vec::<RedesignImplementationNode>::with_capacity(projected_ops.len());

    for op in projected_ops {
        let activation_inputs = op
            .inputs
            .iter()
            .filter_map(|index| usize::try_from(*index).ok())
            .filter(|index| {
                projected_tensors
                    .get(*index)
                    .map(|tensor| !tensor.constant_buffer)
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        let outputs = op
            .outputs
            .iter()
            .filter_map(|index| usize::try_from(*index).ok())
            .collect::<Vec<_>>();
        let input_shapes = activation_inputs
            .iter()
            .filter_map(|index| {
                projected_tensors
                    .get(*index)
                    .map(|tensor| tensor.shape.clone())
            })
            .collect::<Vec<_>>();
        let output_shapes = outputs
            .iter()
            .filter_map(|index| {
                projected_tensors
                    .get(*index)
                    .map(|tensor| tensor.shape.clone())
            })
            .collect::<Vec<_>>();
        let input_channels = input_shapes
            .first()
            .and_then(|shape| shape.last())
            .and_then(|value| positive(*value));
        let output_channels = output_shapes
            .first()
            .and_then(|shape| shape.last())
            .and_then(|value| positive(*value));
        let input_h = input_shapes
            .first()
            .and_then(|shape| shape.get(1))
            .and_then(|value| positive(*value));
        let input_w = input_shapes
            .first()
            .and_then(|shape| shape.get(2))
            .and_then(|value| positive(*value));
        let output_h = output_shapes
            .first()
            .and_then(|shape| shape.get(1))
            .and_then(|value| positive(*value));
        let output_w = output_shapes
            .first()
            .and_then(|shape| shape.get(2))
            .and_then(|value| positive(*value));
        let stride_height = infer_projected_stride(input_h, output_h);
        let stride_width = infer_projected_stride(input_w, output_w);
        let kernel_height = op.cache_payload.kernel_height;
        let kernel_width = op.cache_payload.kernel_width;
        let (module_kind, codegen_status, codegen_reason, concatenation_axis_nhwc) =
            reviewed_implementation_codegen_contract(&ReviewedCodegenInput {
                op_name: &op.name,
                activation_input_count: activation_inputs.len(),
                output_count: outputs.len(),
                input_shapes: &input_shapes,
                output_shapes: &output_shapes,
                input_channels,
                output_channels,
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
            });
        match codegen_status.as_str() {
            "exact_structure" => exact_codegen_op_count += 1,
            "reconstruction_scaffold" => scaffold_codegen_op_count += 1,
            _ => unsupported_codegen_op_count += 1,
        }
        let (source_layer_ref, source_layer_evidence_class, source_tensor_refs) =
            source_layer_mapping(op, projected_tensors);
        if !source_layer_ref.is_empty() {
            mapped_source_layer_count += 1;
        }
        nodes.push(RedesignImplementationNode {
            op_index: op.index,
            op_name: op.name.clone(),
            generated_symbol: format!("op_{:03}", op.index),
            block_id: block_by_op
                .get(&op.index)
                .map(|block| block.block_id.clone())
                .unwrap_or_default(),
            source_layer_ref,
            source_layer_evidence_class,
            source_tensor_refs,
            module_kind,
            codegen_status,
            codegen_reason,
            activation_inputs,
            outputs,
            input_shapes,
            output_shapes,
            input_channels,
            output_channels,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            concatenation_axis_nhwc,
            fused_activation: op.fused_activation.clone(),
        });
    }

    let status = if repeat_edit_count > 0 {
        "blocked_non_materialized_repeat"
    } else if unsupported_codegen_op_count > 0 {
        "partial"
    } else {
        "assessed_scaffold"
    };
    RedesignImplementationPlan {
        schema: "deepbom.redesign_implementation_plan.v1".to_string(),
        status: status.to_string(),
        evidence_class: "DERIVED_FROM_PROJECTED_STRUCTURE".to_string(),
        model_inputs,
        model_outputs,
        nodes,
        exact_codegen_op_count,
        scaffold_codegen_op_count,
        unsupported_codegen_op_count,
        non_materialized_repeat_edit_count: repeat_edit_count,
        exportable: repeat_edit_count == 0,
        mapped_source_layer_count,
        framework_targets: vec!["pytorch".to_string(), "keras_litert".to_string()],
        method: "The WASM projector converts the projected operator/tensor ledger into a framework-neutral implementation plan. Artifact-embedded tensor paths are retained as source-like references; convolution and pooling options that are not serialized in the analysis contract remain explicitly scaffold-class evidence.".to_string(),
        interpretation_boundary: "Generated source is a weight-free architecture reconstruction scaffold. It is not the original training source, does not recover optimizer/data/loss contracts, and must pass framework execution, conversion, and task-validation gates before use.".to_string(),
    }
}

fn implementation_tensors(
    tensors: &[TensorInfo],
    indices: &HashSet<usize>,
) -> Vec<RedesignImplementationTensor> {
    let mut ordered = indices.iter().copied().collect::<Vec<_>>();
    ordered.sort_unstable();
    ordered
        .into_iter()
        .filter_map(|index| {
            tensors
                .get(index)
                .map(|tensor| RedesignImplementationTensor {
                    tensor_index: index,
                    artifact_name: tensor.name.clone(),
                    projected_shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                })
        })
        .collect()
}

fn implementation_codegen_contract(op_name: &str) -> (&'static str, &'static str, &'static str) {
    match op_name {
        "ADD" => ("add", "exact_structure", "Elementwise addition is emitted from explicit activation edges."),
        "SUB" => ("subtract", "exact_structure", "Elementwise subtraction is emitted from explicit activation edges."),
        "MUL" => ("multiply", "exact_structure", "Elementwise multiplication is emitted from explicit activation edges."),
        "MAXIMUM" => ("maximum", "exact_structure", "Elementwise maximum is emitted from explicit activation edges."),
        "MINIMUM" => ("minimum", "exact_structure", "Elementwise minimum is emitted from explicit activation edges."),
        "RESHAPE" => ("reshape", "exact_structure", "The projected output shape is emitted directly."),
        "SQUEEZE" => ("squeeze", "exact_structure", "Singleton dimensions are removed from the projected tensor contract."),
        "RELU" => ("relu", "exact_structure", "ReLU semantics are explicit in the serialized operator."),
        "RELU6" => ("relu6", "exact_structure", "ReLU6 semantics are explicit in the serialized operator."),
        "LOGISTIC" => ("sigmoid", "exact_structure", "Logistic semantics are explicit in the serialized operator."),
        "SOFTMAX" => ("softmax", "exact_structure", "Softmax is emitted on the feature/channel dimension."),
        "QUANTIZE" | "DEQUANTIZE" => ("precision_boundary", "exact_structure", "The weight-free float scaffold preserves this boundary as an annotated identity."),
        "CONV_2D" => ("conv2d", "reconstruction_scaffold", "Channels, kernel, stride, and projected shapes are derived; exact source-framework padding and layer naming are not claimed."),
        "DEPTHWISE_CONV_2D" => ("depthwise_conv2d", "reconstruction_scaffold", "Depthwise grouping and projected shapes are derived; exact source-framework layer construction is not claimed."),
        "FULLY_CONNECTED" => ("linear", "reconstruction_scaffold", "Input/output features are derived from projected tensor contracts."),
        "AVERAGE_POOL_2D" => ("average_pool2d", "reconstruction_scaffold", "The projected output shape is preserved; source-framework pool option reconstruction remains a scaffold."),
        "MAX_POOL_2D" => ("max_pool2d", "reconstruction_scaffold", "The projected output shape is preserved; source-framework pool option reconstruction remains a scaffold."),
        "CONCATENATION" => ("concatenate", "reconstruction_scaffold", "The channel axis is selected only when projected shapes prove channel concatenation."),
        "PAD" | "PADV2" => ("unsupported", "unsupported", "The serialized padding vector is not present in the projected implementation contract; emitting an identity or guessed split would violate shape semantics."),
        _ => ("unsupported", "unsupported", "No reviewed weight-free framework emitter is registered for this operator."),
    }
}

struct ReviewedCodegenInput<'a> {
    op_name: &'a str,
    activation_input_count: usize,
    output_count: usize,
    input_shapes: &'a [Vec<i32>],
    output_shapes: &'a [Vec<i32>],
    input_channels: Option<usize>,
    output_channels: Option<usize>,
    kernel_height: Option<usize>,
    kernel_width: Option<usize>,
    stride_height: Option<usize>,
    stride_width: Option<usize>,
}

fn reviewed_implementation_codegen_contract(
    input: &ReviewedCodegenInput<'_>,
) -> (String, String, String, Option<usize>) {
    let op_name = input.op_name;
    let activation_input_count = input.activation_input_count;
    let output_count = input.output_count;
    let input_shapes = input.input_shapes;
    let output_shapes = input.output_shapes;
    let input_channels = input.input_channels;
    let output_channels = input.output_channels;
    let kernel_height = input.kernel_height;
    let kernel_width = input.kernel_width;
    let stride_height = input.stride_height;
    let stride_width = input.stride_width;
    let (module_kind, status, reason) = implementation_codegen_contract(op_name);
    if output_count != 1 {
        return (
            "unsupported".to_string(),
            "unsupported".to_string(),
            format!(
                "{} has {} projected outputs; the reviewed emitters require exactly one.",
                op_name, output_count
            ),
            None,
        );
    }
    let required_inputs = match op_name {
        "ADD" | "SUB" | "MUL" | "MAXIMUM" | "MINIMUM" => Some(2),
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "AVERAGE_POOL_2D" | "MAX_POOL_2D"
        | "RESHAPE" | "SQUEEZE" | "RELU" | "RELU6" | "LOGISTIC" | "SOFTMAX" | "QUANTIZE"
        | "DEQUANTIZE" => Some(1),
        _ => None,
    };
    if required_inputs
        .map(|required| activation_input_count != required)
        .unwrap_or(false)
    {
        return (
            "unsupported".to_string(),
            "unsupported".to_string(),
            format!(
                "{} has {} activation input(s); its remaining serialized operands are constants or unavailable, so weight-free code emission is not exact.",
                op_name, activation_input_count
            ),
            None,
        );
    }
    if matches!(op_name, "CONV_2D" | "DEPTHWISE_CONV_2D")
        && (input_shapes.first().map(Vec::len) != Some(4)
            || output_shapes.first().map(Vec::len) != Some(4)
            || [
                input_channels,
                output_channels,
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
            ]
            .iter()
            .any(Option::is_none))
    {
        return (
            "unsupported".to_string(),
            "unsupported".to_string(),
            format!(
                "{} lacks a complete static rank-4 channel/kernel/stride contract.",
                op_name
            ),
            None,
        );
    }
    if op_name == "DEPTHWISE_CONV_2D"
        && !matches!((input_channels, output_channels), (Some(input), Some(output)) if output % input == 0)
    {
        return (
            "unsupported".to_string(),
            "unsupported".to_string(),
            "DEPTHWISE_CONV_2D output channels are not an integer multiple of input channels."
                .to_string(),
            None,
        );
    }
    if op_name == "CONCATENATION" {
        if activation_input_count < 2 {
            return (
                "unsupported".to_string(),
                "unsupported".to_string(),
                "CONCATENATION requires at least two addressable activation inputs.".to_string(),
                None,
            );
        }
        let Some(axis) = unique_concatenation_axis(input_shapes, output_shapes.first()) else {
            return (
                "unsupported".to_string(),
                "unsupported".to_string(),
                "No unique concatenation axis satisfies the projected input/output shape conservation rule.".to_string(),
                None,
            );
        };
        return (
            module_kind.to_string(),
            status.to_string(),
            format!(
                "The NHWC axis {} is uniquely derived by summing input extents and conserving every other projected dimension.",
                axis
            ),
            Some(axis),
        );
    }
    (
        module_kind.to_string(),
        status.to_string(),
        reason.to_string(),
        None,
    )
}

fn unique_concatenation_axis(
    input_shapes: &[Vec<i32>],
    output_shape: Option<&Vec<i32>>,
) -> Option<usize> {
    let output = output_shape?;
    if input_shapes.len() < 2
        || output.is_empty()
        || input_shapes.iter().any(|shape| shape.len() != output.len())
    {
        return None;
    }
    let candidates = (0..output.len())
        .filter(|axis| {
            let non_axis_dimensions_match = input_shapes.iter().all(|shape| {
                shape
                    .iter()
                    .enumerate()
                    .all(|(index, value)| index == *axis || (*value > 0 && *value == output[index]))
            });
            let summed = input_shapes.iter().try_fold(0_i64, |sum, shape| {
                let value = i64::from(shape[*axis]);
                (value > 0).then(|| sum.saturating_add(value))
            });
            non_axis_dimensions_match && summed == Some(i64::from(output[*axis]))
        })
        .collect::<Vec<_>>();
    if candidates.len() == 1 {
        candidates.first().copied()
    } else {
        None
    }
}

fn infer_projected_stride(input: Option<usize>, output: Option<usize>) -> Option<usize> {
    match (input, output) {
        (Some(input), Some(output)) if output > 0 => Some(input.div_ceil(output).max(1)),
        _ => None,
    }
}

fn source_layer_mapping(op: &OpInfo, tensors: &[TensorInfo]) -> (String, String, Vec<String>) {
    let mut constant_refs = op
        .inputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .filter(|tensor| tensor.constant_buffer && !tensor.name.is_empty())
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    constant_refs.sort();
    constant_refs.dedup();
    if let Some(reference) = constant_refs
        .iter()
        .find(|name| name.contains("/weights") || name.contains("/kernel"))
        .or_else(|| constant_refs.first())
    {
        return (
            source_like_parent(reference),
            "ARTIFACT_TENSOR_PATH".to_string(),
            constant_refs,
        );
    }
    let mut output_refs = op
        .outputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .filter(|tensor| !tensor.name.is_empty())
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    output_refs.sort();
    output_refs.dedup();
    if let Some(reference) = output_refs.first() {
        return (
            source_like_parent(reference),
            "ARTIFACT_OUTPUT_PATH".to_string(),
            output_refs,
        );
    }
    (String::new(), "NOT_AVAILABLE".to_string(), Vec::new())
}

fn source_like_parent(name: &str) -> String {
    let markers = [
        "/weights",
        "/kernel",
        "/bias",
        "/Conv2D_Fold_bias",
        "/depthwise_Fold_bias",
        "_Fold_bias",
        "/ReadVariableOp",
        "/FakeQuant",
        "/Relu",
        "/relu",
    ];
    let cut = markers
        .iter()
        .filter_map(|marker| name.find(marker))
        .min()
        .unwrap_or(name.len());
    let trimmed = name[..cut].trim_end_matches('/');
    if trimmed.is_empty() {
        name.to_string()
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn build_redesign_pareto(
    bytes: &[u8],
    analysis: &Analysis,
    base_request: RedesignRequest,
) -> Result<RedesignParetoSearch, String> {
    validate_request(analysis, &base_request)?;
    let source_input = analysis
        .inputs
        .iter()
        .find(|tensor| tensor.shape.len() == 4)
        .ok_or_else(|| {
            "Pareto exploration requires a statically shaped rank-4 TFLite input.".to_string()
        })?;
    let source_h = positive(source_input.shape[1])
        .ok_or_else(|| "Source input height is dynamic or invalid.".to_string())?;
    let source_w = positive(source_input.shape[2])
        .ok_or_else(|| "Source input width is dynamic or invalid.".to_string())?;
    let source_hash = sha256(bytes);
    let mut dimensions = BTreeSet::<(usize, usize)>::new();
    for scale in [1.0_f64, 0.875, 0.75, 0.625, 0.5] {
        dimensions.insert((
            ((source_h as f64 * scale).round() as usize).max(1),
            ((source_w as f64 * scale).round() as usize).max(1),
        ));
    }
    dimensions.insert((
        base_request.input_height.unwrap_or(source_h),
        base_request.input_width.unwrap_or(source_w),
    ));
    let mut widths = vec![
        1.0_f64,
        0.875,
        0.75,
        0.625,
        0.5,
        base_request.width_multiplier,
    ];
    widths.sort_by(|left, right| left.total_cmp(right));
    widths.dedup_by(|left, right| (*left - *right).abs() < 1e-9);

    let evaluated_candidate_count = dimensions.len().saturating_mul(widths.len());
    let mut rejected_candidate_count = 0usize;
    let mut candidates = Vec::<RedesignParetoCandidate>::new();
    for (height, width) in dimensions {
        for width_multiplier in &widths {
            let mut request = base_request.clone();
            request.source_sha256 = source_hash.clone();
            request.input_height = Some(height);
            request.input_width = Some(width);
            request.width_multiplier = *width_multiplier;
            let projection = match build_redesign_projection(bytes, analysis, request.clone()) {
                Ok(projection) => projection,
                Err(_) => {
                    rejected_candidate_count += 1;
                    continue;
                }
            };
            let projected = &projection.metrics.projected;
            let area_ratio = (height.saturating_mul(width)) as f64
                / (source_h.saturating_mul(source_w)).max(1) as f64;
            let retained_structure_proxy = area_ratio * width_multiplier.powi(2);
            candidates.push(RedesignParetoCandidate {
                candidate_id: format!(
                    "h{}-w{}-c{}",
                    height,
                    width,
                    format!("{width_multiplier:.3}").replace('.', "p")
                ),
                pareto_optimal: false,
                status: projection.status.clone(),
                request,
                retained_structure_proxy,
                modeled_latency_ms: projected.modeled_latency_ms,
                macs: projected.macs,
                parameter_elements: projected.parameter_elements,
                peak_live_activation_bytes: projected.peak_live_activation_bytes,
                l1_max_ratio: projected.l1_max_ratio,
                error_constraint_count: projection
                    .constraints
                    .iter()
                    .filter(|item| item.severity == "error")
                    .count(),
                warning_constraint_count: projection
                    .constraints
                    .iter()
                    .filter(|item| matches!(item.severity.as_str(), "warn" | "warning"))
                    .count(),
                projection_coverage_status: projection.projection_coverage.status.clone(),
            });
        }
    }

    let accepted_indices = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.status != "blocked" && candidate.error_constraint_count == 0
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for index in &accepted_indices {
        let dominated = accepted_indices.iter().any(|other_index| {
            index != other_index && pareto_dominates(&candidates[*other_index], &candidates[*index])
        });
        candidates[*index].pareto_optimal = !dominated;
    }
    candidates.sort_by(|left, right| {
        right
            .pareto_optimal
            .cmp(&left.pareto_optimal)
            .then_with(|| {
                right
                    .retained_structure_proxy
                    .total_cmp(&left.retained_structure_proxy)
            })
            .then_with(|| left.modeled_latency_ms.total_cmp(&right.modeled_latency_ms))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let frontier_candidate_count = candidates
        .iter()
        .filter(|candidate| candidate.pareto_optimal)
        .count();
    let accepted_candidate_count = accepted_indices.len();
    Ok(RedesignParetoSearch {
        schema: "deepbom.redesign_pareto.v1".to_string(),
        status: if accepted_candidate_count > 0 { "assessed" } else { "blocked" }.to_string(),
        source_sha256: source_hash,
        target_id: analysis.target_profile.id.clone(),
        evaluated_candidate_count,
        accepted_candidate_count,
        rejected_candidate_count,
        frontier_candidate_count,
        candidates,
        objectives: vec![
            "maximize retained_structure_proxy = input_area_ratio * width_multiplier^2".to_string(),
            "minimize target-profile modeled latency".to_string(),
            "minimize projected parameter elements".to_string(),
            "minimize peak live activation when assessed".to_string(),
        ],
        method: "A deterministic grid of input-resolution and width-multiplier candidates is projected against one bound WASM analysis. A candidate is Pareto-optimal when no accepted candidate preserves at least as much structural capacity while being no worse in latency, parameters, and any jointly assessed peak-live-activation value, with at least one strict improvement.".to_string(),
        interpretation_boundary: "Retained structure is an architecture-capacity proxy, not task accuracy, calibration, robustness, or clinical utility. Frontier membership does not make a candidate deployable; materialization, training, conversion, and complete audit remain required.".to_string(),
    })
}

fn pareto_dominates(left: &RedesignParetoCandidate, right: &RedesignParetoCandidate) -> bool {
    let retention = left.retained_structure_proxy + 1e-12 >= right.retained_structure_proxy;
    let latency = left.modeled_latency_ms <= right.modeled_latency_ms + 1e-12;
    let parameters = left.parameter_elements <= right.parameter_elements;
    let activation = match (
        left.peak_live_activation_bytes,
        right.peak_live_activation_bytes,
    ) {
        (Some(left), Some(right)) => left <= right,
        _ => true,
    };
    let strict = left.retained_structure_proxy > right.retained_structure_proxy + 1e-12
        || left.modeled_latency_ms + 1e-12 < right.modeled_latency_ms
        || left.parameter_elements < right.parameter_elements
        || matches!(
            (left.peak_live_activation_bytes, right.peak_live_activation_bytes),
            (Some(left), Some(right)) if left < right
        );
    retention && latency && parameters && activation && strict
}

fn validate_request(analysis: &Analysis, request: &RedesignRequest) -> Result<(), String> {
    if request.schema != "deepbom.redesign_request.v1" {
        return Err("Unsupported redesign request schema.".to_string());
    }
    if !request.width_multiplier.is_finite() || !(0.25..=2.0).contains(&request.width_multiplier) {
        return Err("width_multiplier must be finite and in 0.25..=2.0.".to_string());
    }
    if !matches!(
        request.activation_dtype.to_ascii_lowercase().as_str(),
        "source" | "int8" | "float32"
    ) {
        return Err("activation_dtype must be source, int8, or float32.".to_string());
    }
    let known = analysis
        .block_inventory
        .blocks
        .iter()
        .map(|block| block.block_id.as_str())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::<&str>::new();
    for edit in &request.block_edits {
        if !known.contains(edit.block_id.as_str()) {
            return Err(format!("Unknown redesign block_id '{}'.", edit.block_id));
        }
        if !seen.insert(edit.block_id.as_str()) {
            return Err(format!("Duplicate redesign edit for '{}'.", edit.block_id));
        }
        if edit.output_channels == Some(0) {
            return Err("output_channels must be positive.".to_string());
        }
        if edit
            .expand_ratio
            .map(|value| !value.is_finite() || !(1.0..=16.0).contains(&value))
            .unwrap_or(false)
        {
            return Err("expand_ratio must be finite and in 1..=16.".to_string());
        }
        if edit
            .repeat
            .map(|value| !(1..=8).contains(&value))
            .unwrap_or(false)
        {
            return Err("repeat must be in 1..=8.".to_string());
        }
        if edit
            .kernel_size
            .map(|value| !(1..=7).contains(&value) || value % 2 == 0)
            .unwrap_or(false)
        {
            return Err("kernel_size must be an odd value in 1..=7.".to_string());
        }
    }
    Ok(())
}

fn project_input_and_storage(
    tensors: &mut [TensorInfo],
    input_indices: &HashSet<usize>,
    source_h: usize,
    source_w: usize,
    projected_h: usize,
    projected_w: usize,
    activation_dtype: &str,
) {
    for tensor in tensors {
        if tensor.shape.len() == 4 && !tensor.constant_buffer {
            if input_indices.contains(&tensor.index) {
                tensor.shape[1] = projected_h as i32;
                tensor.shape[2] = projected_w as i32;
            } else {
                tensor.shape[1] = scale_dimension(tensor.shape[1], projected_h, source_h) as i32;
                tensor.shape[2] = scale_dimension(tensor.shape[2], projected_w, source_w) as i32;
            }
        }
        if activation_dtype != "source" && !tensor.constant_buffer && !is_shape_dtype(&tensor.dtype)
        {
            apply_hypothetical_dtype(tensor, activation_dtype);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn project_op_shape(
    op: &OpInfo,
    source_ops: &[OpInfo],
    source_tensors: &[TensorInfo],
    tensors: &mut [TensorInfo],
    source_h: usize,
    source_w: usize,
    projected_h: usize,
    projected_w: usize,
    width_multiplier: f64,
    alignment: usize,
    input_indices: &HashSet<usize>,
    output_indices: &HashSet<usize>,
    owner: Option<&BlockRecord>,
    edit: Option<&RedesignBlockEdit>,
) -> ShapeRuleStatus {
    let source_op = source_ops
        .iter()
        .find(|source| source.index == op.index)
        .unwrap_or(op);
    let shape_projection = ShapeProjection {
        source_h,
        source_w,
        projected_h,
        projected_w,
        width_multiplier,
        alignment,
    };
    if let Some(kernel_size) = edit.and_then(|item| item.kernel_size) {
        if matches!(op.name.as_str(), "DEPTHWISE_CONV_2D" | "CONV_2D") {
            if let Some(weight) = op
                .inputs
                .get(1)
                .and_then(|index| usize::try_from(*index).ok())
                .and_then(|index| tensors.get_mut(index))
            {
                let source_spatial_kernel =
                    weight.shape.len() == 4 && (weight.shape[1] > 1 || weight.shape[2] > 1);
                if source_spatial_kernel {
                    weight.shape[1] = kernel_size as i32;
                    weight.shape[2] = kernel_size as i32;
                }
            }
        }
    }
    let input_shape = op
        .inputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .find(|tensor| !tensor.constant_buffer)
        .map(|tensor| tensor.shape.clone());
    let source_input_shape = source_op
        .inputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| source_tensors.get(index))
        .find(|tensor| !tensor.constant_buffer)
        .map(|tensor| tensor.shape.clone());
    let source_output_shape = source_op.output_shapes.first().cloned();
    let output_indices_usize = op
        .outputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .collect::<Vec<_>>();
    if output_indices_usize.is_empty() {
        return ShapeRuleStatus::Unassessed;
    }

    let mut projected = source_output_shape.clone().unwrap_or_default();
    let mut status = ShapeRuleStatus::ScaledFallback;
    if let Some(input) = input_shape.as_ref() {
        match op.name.as_str() {
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV" => {
                if input.len() == 4 && projected.len() == 4 {
                    let kernel = op
                        .inputs
                        .get(1)
                        .and_then(|index| usize::try_from(*index).ok())
                        .and_then(|index| tensors.get(index))
                        .map(|tensor| tensor.shape.clone());
                    let source_input = source_input_shape.clone().unwrap_or_else(|| input.clone());
                    let source_output = source_output_shape
                        .clone()
                        .unwrap_or_else(|| projected.clone());
                    if source_input.len() == 4 && source_output.len() == 4 {
                        projected[0] = input[0];
                        projected[1] = project_spatial_axis(
                            positive_i32(source_input[1]),
                            positive_i32(source_output[1]),
                            positive_i32(input[1]),
                            kernel
                                .as_ref()
                                .and_then(|shape| shape.get(1).copied())
                                .and_then(positive_i32),
                        ) as i32;
                        projected[2] = project_spatial_axis(
                            positive_i32(source_input[2]),
                            positive_i32(source_output[2]),
                            positive_i32(input[2]),
                            kernel
                                .as_ref()
                                .and_then(|shape| shape.get(2).copied())
                                .and_then(positive_i32),
                        ) as i32;
                        let source_out_channels = positive_i32(source_output[3]).unwrap_or(1);
                        let is_model_output = output_indices_usize
                            .iter()
                            .any(|index| output_indices.contains(index));
                        let mut output_channels = if op.name == "DEPTHWISE_CONV_2D" {
                            let source_in_channels = positive_i32(source_input[3]).unwrap_or(1);
                            let depth_multiplier =
                                source_out_channels.div_ceil(source_in_channels).max(1);
                            positive_i32(input[3]).unwrap_or(1) * depth_multiplier
                        } else if is_model_output {
                            source_out_channels
                        } else {
                            aligned_channels(source_out_channels, width_multiplier, alignment)
                        };
                        if let Some(block) = owner {
                            if let Some(edit) = edit {
                                let conv_positions = block
                                    .op_indices
                                    .iter()
                                    .filter(|index| {
                                        source_ops
                                            .iter()
                                            .find(|candidate| candidate.index == **index)
                                            .map(|candidate| {
                                                matches!(
                                                    candidate.name.as_str(),
                                                    "CONV_2D"
                                                        | "DEPTHWISE_CONV_2D"
                                                        | "FULLY_CONNECTED"
                                                )
                                            })
                                            .unwrap_or(false)
                                    })
                                    .copied()
                                    .collect::<Vec<_>>();
                                if conv_positions.first() == Some(&op.index)
                                    && block.block_type == "inverted_bottleneck"
                                {
                                    if let Some(ratio) = edit.expand_ratio {
                                        output_channels =
                                            aligned_value(input[3] as f64 * ratio, alignment);
                                    }
                                }
                                if conv_positions.last() == Some(&op.index) {
                                    output_channels =
                                        edit.output_channels.unwrap_or(output_channels);
                                }
                            }
                        }
                        projected[3] = output_channels.max(1) as i32;
                        status = ShapeRuleStatus::Exact;
                    }
                }
            }
            "AVERAGE_POOL_2D" | "MAX_POOL_2D" => {
                if input.len() == 4 && projected.len() == 4 {
                    projected[0] = input[0];
                    projected[3] = input[3];
                    projected[1] = scale_dimension(projected[1], projected_h, source_h) as i32;
                    projected[2] = scale_dimension(projected[2], projected_w, source_w) as i32;
                    status = ShapeRuleStatus::ScaledFallback;
                }
            }
            "ADD" | "MUL" | "SUB" | "DIV" | "MAXIMUM" | "MINIMUM" | "RELU" | "RELU6"
            | "LOGISTIC" | "TANH" | "QUANTIZE" | "DEQUANTIZE" | "HARD_SWISH" | "LEAKY_RELU"
            | "PRELU" => {
                projected = input.clone();
                status = ShapeRuleStatus::Exact;
            }
            "MEAN" | "SUM" | "REDUCE_MAX" | "REDUCE_MIN" | "REDUCE_PROD" => {
                if !projected.is_empty() {
                    if let Some(last) = projected.last_mut() {
                        if let Some(channel) = input.last().copied() {
                            *last = channel;
                        }
                    }
                    status = ShapeRuleStatus::ScaledFallback;
                }
            }
            "RESHAPE" | "SQUEEZE" | "EXPAND_DIMS" => {
                if !projected.is_empty() {
                    projected = scaled_serialized_shape(&projected, &shape_projection, false);
                    status = ShapeRuleStatus::ScaledFallback;
                }
            }
            "FULLY_CONNECTED" | "BATCH_MATMUL" => {
                if !projected.is_empty() {
                    let is_model_output = output_indices_usize
                        .iter()
                        .any(|index| output_indices.contains(index));
                    if !is_model_output {
                        if let Some(last) = projected.last_mut() {
                            if let Some(channel) = positive_i32(*last) {
                                *last =
                                    aligned_channels(channel, width_multiplier, alignment) as i32;
                            }
                        }
                    }
                    status = ShapeRuleStatus::Exact;
                }
            }
            _ => {
                projected = scaled_serialized_shape(
                    &projected,
                    &shape_projection,
                    output_indices_usize
                        .iter()
                        .any(|index| output_indices.contains(index)),
                );
            }
        }
    }
    for output_index in output_indices_usize {
        if let Some(tensor) = tensors.get_mut(output_index) {
            if !projected.is_empty() {
                tensor.shape = projected.clone();
                tensor.shape_signature = projected.clone();
            }
            if input_indices.contains(&output_index) {
                status = ShapeRuleStatus::Unassessed;
            }
        }
    }
    status
}

fn project_constant_storage(op: &OpInfo, tensors: &mut [TensorInfo], activation_dtype: &str) {
    let input_shape = op
        .inputs
        .first()
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index))
        .map(|tensor| tensor.shape.clone());
    let output_shape = op
        .outputs
        .first()
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index))
        .map(|tensor| tensor.shape.clone());
    match op.name.as_str() {
        "CONV_2D" => {
            if let (Some(input), Some(output), Some(weight_index)) = (
                input_shape,
                output_shape,
                op.inputs
                    .get(1)
                    .and_then(|index| usize::try_from(*index).ok()),
            ) {
                if let Some(weight) = tensors.get_mut(weight_index) {
                    if weight.shape.len() == 4 && input.len() == 4 && output.len() == 4 {
                        weight.shape[0] = output[3];
                        weight.shape[3] = input[3];
                        apply_weight_storage(weight, activation_dtype, false);
                    }
                }
                project_bias(op, tensors, output[3], activation_dtype);
            }
        }
        "DEPTHWISE_CONV_2D" => {
            if let (Some(output), Some(weight_index)) = (
                output_shape,
                op.inputs
                    .get(1)
                    .and_then(|index| usize::try_from(*index).ok()),
            ) {
                if let Some(weight) = tensors.get_mut(weight_index) {
                    if weight.shape.len() == 4 && output.len() == 4 {
                        weight.shape[3] = output[3];
                        apply_weight_storage(weight, activation_dtype, false);
                    }
                }
                project_bias(op, tensors, output[3], activation_dtype);
            }
        }
        "FULLY_CONNECTED" => {
            if let (Some(input), Some(output), Some(weight_index)) = (
                input_shape,
                output_shape,
                op.inputs
                    .get(1)
                    .and_then(|index| usize::try_from(*index).ok()),
            ) {
                if let Some(weight) = tensors.get_mut(weight_index) {
                    if weight.shape.len() >= 2 {
                        if let Some(last) = input.last() {
                            let rank = weight.shape.len();
                            weight.shape[rank - 1] = *last;
                        }
                        if let Some(last) = output.last() {
                            weight.shape[0] = *last;
                        }
                        apply_weight_storage(weight, activation_dtype, false);
                    }
                }
                if let Some(last) = output.last() {
                    project_bias(op, tensors, *last, activation_dtype);
                }
            }
        }
        _ => {}
    }
}

fn project_bias(op: &OpInfo, tensors: &mut [TensorInfo], channels: i32, dtype: &str) {
    let Some(index) = op
        .inputs
        .get(2)
        .and_then(|value| usize::try_from(*value).ok())
    else {
        return;
    };
    let Some(bias) = tensors.get_mut(index) else {
        return;
    };
    if !bias.constant_buffer {
        return;
    }
    bias.shape = vec![channels.max(1)];
    if dtype == "int8" {
        bias.dtype = "INT32".to_string();
    } else if dtype == "float32" {
        bias.dtype = "FLOAT32".to_string();
    }
    update_projected_buffer_length(bias);
}

fn apply_weight_storage(tensor: &mut TensorInfo, dtype: &str, bias: bool) {
    if dtype == "int8" {
        tensor.dtype = if bias { "INT32" } else { "INT8" }.to_string();
    } else if dtype == "float32" {
        tensor.dtype = "FLOAT32".to_string();
    }
    update_projected_buffer_length(tensor);
}

fn update_projected_buffer_length(tensor: &mut TensorInfo) {
    let count = tensor.shape.iter().try_fold(1usize, |product, dimension| {
        product.checked_mul(positive(*dimension)?)
    });
    let width = storage_bytes(&tensor.dtype);
    tensor.buffer_data_length = count
        .zip(width)
        .and_then(|(count, width)| count.checked_mul(width))
        .unwrap_or(0);
}

fn apply_hypothetical_dtype(tensor: &mut TensorInfo, dtype: &str) {
    tensor.dtype = if dtype == "int8" { "INT8" } else { "FLOAT32" }.to_string();
    tensor.quant_scales = 0;
    tensor.quant_zero_points = 0;
    tensor.scale_sample.clear();
    tensor.zero_point_sample.clear();
    tensor.scale_mode = "hypothetical_unparameterized".to_string();
}

fn is_shape_dtype(dtype: &str) -> bool {
    matches!(dtype, "INT32" | "INT64" | "BOOL")
}

fn block_ownership(blocks: &[BlockRecord]) -> HashMap<usize, &BlockRecord> {
    let mut result = HashMap::new();
    for block in blocks {
        for op_index in &block.op_indices {
            result.entry(*op_index).or_insert(block);
        }
    }
    result
}

fn build_cache_points(
    source_ops: &[OpInfo],
    projected_ops: &[OpInfo],
    blocks: &[BlockRecord],
    target: &TargetProfile,
) -> Vec<RedesignCachePoint> {
    let block_by_op = block_ownership(blocks);
    let projected_by_index = projected_ops
        .iter()
        .map(|op| (op.index, op))
        .collect::<HashMap<_, _>>();
    source_ops
        .iter()
        .filter_map(|source| {
            let projected = projected_by_index.get(&source.index).copied()?;
            if source.cache_payload.status != "assessed"
                && projected.cache_payload.status != "assessed"
            {
                return None;
            }
            let source_payload = &source.cache_payload;
            let projected_payload = &projected.cache_payload;
            Some(RedesignCachePoint {
                op_index: source.index,
                op_name: source.name.clone(),
                block_id: block_by_op
                    .get(&source.index)
                    .map(|block| block.block_id.clone())
                    .unwrap_or_default(),
                source_width: source_payload.input_width,
                source_channels: source_payload.input_channels,
                source_logical_row_payload_bytes: source_payload.logical_row_payload_bytes,
                source_l1_ratio: cache_ratio(
                    source_payload.logical_row_payload_bytes,
                    target.l1_data_bytes,
                ),
                projected_width: projected_payload.input_width,
                projected_channels: projected_payload.input_channels,
                projected_logical_row_payload_bytes: projected_payload.logical_row_payload_bytes,
                projected_l1_ratio: cache_ratio(
                    projected_payload.logical_row_payload_bytes,
                    target.l1_data_bytes,
                ),
                evidence_class: if source_payload.status == "assessed"
                    && projected_payload.status == "assessed"
                {
                    "DERIVED_FROM_SCENARIO"
                } else {
                    "NOT_ASSESSABLE"
                }
                .to_string(),
            })
        })
        .collect()
}

fn cache_ratio(bytes: Option<usize>, denominator: usize) -> Option<f64> {
    (denominator > 0)
        .then_some(denominator)
        .and_then(|denominator| bytes.map(|bytes| bytes as f64 / denominator as f64))
}

fn validate_and_apply_repeat_edits<'a>(
    blocks: &'a [BlockRecord],
    edits: &HashMap<String, RedesignBlockEdit>,
    constraints: &mut Vec<RedesignConstraint>,
) -> Vec<(&'a BlockRecord, usize)> {
    let mut result = Vec::new();
    for block in blocks {
        let Some(repeat) = edits.get(&block.block_id).and_then(|edit| edit.repeat) else {
            continue;
        };
        if repeat == 1 {
            continue;
        }
        let repeatable = block.params.stride_h == Some(1)
            && block.params.stride_w == Some(1)
            && block.channels.input == block.channels.output
            && block.residual;
        if repeatable {
            result.push((block, repeat - 1));
        } else {
            constraints.push(RedesignConstraint {
                severity: "error".to_string(),
                code: "RD-STRUCT-REPEAT-001".to_string(),
                scope: block.block_id.clone(),
                detail: "Repeat edits are supported only for stride-1 residual blocks with equal input/output channels; this block fails that structural invariant.".to_string(),
                evidence_class: "DERIVED".to_string(),
            });
        }
    }
    result
}

fn source_metrics(analysis: &Analysis) -> RedesignMetrics {
    let parameter_indices = parameter_tensor_indices(&analysis.ops, &analysis.tensors);
    RedesignMetrics {
        operator_count: analysis.ops.len(),
        macs: analysis.total_macs,
        operations: analysis.total_ops,
        parameter_elements: parameter_indices
            .iter()
            .filter_map(|index| analysis.tensors.get(*index))
            .filter_map(tensor_element_count)
            .sum(),
        serialized_parameter_bytes: parameter_indices
            .iter()
            .filter_map(|index| analysis.tensors.get(*index))
            .map(|tensor| tensor.buffer_data_length)
            .sum(),
        modeled_latency_ms: analysis
            .ops
            .iter()
            .map(|op| op.bottleneck_total_us)
            .sum::<f64>()
            / 1000.0,
        latency_evidence_class: "ESTIMATED_TARGET_PROFILE".to_string(),
        l1_max_ratio: max_logical_l1_ratio(&analysis.ops, &analysis.target_profile),
        l1_watch_count: logical_l1_watch_count(&analysis.ops, &analysis.target_profile),
        peak_live_activation_bytes: analysis.tensor_liveness.peak_bytes_value,
        arena_bytes: analysis.tensor_arena_plan.combined_arena_bytes,
        predicted_break_count: Some(analysis.xnnpack_chain_breaks),
        delegation_evidence_class: "PREDICTED_SOURCE_ARTIFACT".to_string(),
    }
}

fn projected_metrics(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    analysis: &Analysis,
    activation_dtype: &str,
    structural_edit: bool,
) -> RedesignMetrics {
    let liveness = compute_tensor_liveness(
        ops,
        tensors,
        &analysis.input_tensor_indices,
        &analysis.output_tensor_indices,
    );
    let arena = compute_tensor_arena_plan(
        ops,
        tensors,
        &analysis.input_tensor_indices,
        &analysis.output_tensor_indices,
    );
    let parameter_indices = parameter_tensor_indices(ops, tensors);
    RedesignMetrics {
        operator_count: ops.len(),
        macs: ops.iter().map(|op| op.macs).sum(),
        operations: ops.iter().map(|op| op.ops).sum(),
        parameter_elements: parameter_indices
            .iter()
            .filter_map(|index| tensors.get(*index))
            .filter_map(tensor_element_count)
            .sum(),
        serialized_parameter_bytes: parameter_indices
            .iter()
            .filter_map(|index| tensors.get(*index))
            .map(|tensor| tensor.buffer_data_length)
            .sum(),
        modeled_latency_ms: ops.iter().map(|op| op.bottleneck_total_us).sum::<f64>() / 1000.0,
        latency_evidence_class: "ESTIMATED_TARGET_PROFILE".to_string(),
        l1_max_ratio: max_logical_l1_ratio(ops, &analysis.target_profile),
        l1_watch_count: logical_l1_watch_count(ops, &analysis.target_profile),
        peak_live_activation_bytes: liveness.peak_bytes_value,
        arena_bytes: arena.combined_arena_bytes,
        predicted_break_count: if activation_dtype == "source" && !structural_edit {
            Some(analysis.xnnpack_chain_breaks)
        } else {
            None
        },
        delegation_evidence_class: if activation_dtype == "source" && !structural_edit {
            "PREDICTED_SOURCE_ARTIFACT"
        } else {
            "NOT_ASSESSABLE_HYPOTHETICAL_ARTIFACT_NOT_MATERIALIZED"
        }
        .to_string(),
    }
}

fn request_has_effective_edit(
    request: &RedesignRequest,
    source_h: usize,
    source_w: usize,
    projected_h: usize,
    projected_w: usize,
) -> bool {
    (request.width_multiplier - 1.0).abs() > f64::EPSILON
        || !request.activation_dtype.eq_ignore_ascii_case("source")
        || projected_h != source_h
        || projected_w != source_w
        || request.block_edits.iter().any(|edit| {
            edit.output_channels.is_some()
                || edit.expand_ratio.is_some()
                || edit.repeat.is_some()
                || edit.kernel_size.is_some()
        })
}

fn parameter_tensor_indices(ops: &[OpInfo], tensors: &[TensorInfo]) -> BTreeSet<usize> {
    let mut indices = BTreeSet::new();
    for op in ops {
        let slots: &[usize] = match op.name.as_str() {
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" => &[1, 2],
            "TRANSPOSE_CONV" => &[1, 3],
            "BATCH_MATMUL" => &[1],
            _ => &[],
        };
        for slot in slots {
            let Some(input) = op.inputs.get(*slot) else {
                continue;
            };
            let Some(index) = usize::try_from(*input).ok() else {
                continue;
            };
            if tensors
                .get(index)
                .map(|tensor| tensor.constant_buffer)
                .unwrap_or(false)
            {
                indices.insert(index);
            }
        }
    }
    indices
}

fn tensor_element_count(tensor: &TensorInfo) -> Option<usize> {
    tensor.shape.iter().try_fold(1usize, |product, dimension| {
        product.checked_mul(positive(*dimension)?)
    })
}

fn max_logical_l1_ratio(ops: &[OpInfo], target: &TargetProfile) -> Option<f64> {
    if target.l1_data_bytes == 0 {
        return None;
    }
    ops.iter()
        .filter_map(|op| op.cache_payload.logical_row_payload_bytes)
        .map(|bytes| bytes as f64 / target.l1_data_bytes as f64)
        .max_by(f64::total_cmp)
}

fn logical_l1_watch_count(ops: &[OpInfo], target: &TargetProfile) -> usize {
    if target.l1_data_bytes == 0 {
        return 0;
    }
    ops.iter()
        .filter_map(|op| op.cache_payload.logical_row_payload_bytes)
        .filter(|bytes| *bytes as f64 / target.l1_data_bytes as f64 >= L1_WORKING_SET_WATCH_RATIO)
        .count()
}

#[allow(clippy::too_many_arguments)]
fn append_global_constraints(
    analysis: &Analysis,
    request: &RedesignRequest,
    projected_h: usize,
    projected_w: usize,
    source_h: usize,
    source_w: usize,
    tensors: &[TensorInfo],
    ops: &[OpInfo],
    constraints: &mut Vec<RedesignConstraint>,
) {
    let alignment = analysis.target_profile.channel_alignment_multiple.max(1);
    for op in ops {
        let channels = op
            .outputs
            .first()
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| tensors.get(index))
            .and_then(|tensor| tensor.shape.last().copied())
            .and_then(positive);
        let Some(channels) = channels else {
            continue;
        };
        if matches!(
            op.name.as_str(),
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED"
        ) && channels % alignment != 0
        {
            let padded = channels.div_ceil(alignment) * alignment;
            let overhead = (padded - channels) as f64 / channels as f64 * 100.0;
            constraints.push(RedesignConstraint {
                severity: "warning".to_string(),
                code: "RD-ALIGN-001".to_string(),
                scope: format!("#{:03} {}", op.index, op.name),
                detail: format!(
                    "{channels} output channels are not divisible by source-backed target alignment {alignment}; padded lane overhead is {overhead:.1}% for that candidate multiple."
                ),
                evidence_class: "DERIVED_FROM_TARGET_PROFILE".to_string(),
            });
        }
    }
    if projected_h * 2 < source_h || projected_w * 2 < source_w {
        constraints.push(RedesignConstraint {
            severity: "warning".to_string(),
            code: "RD-INPUT-001".to_string(),
            scope: "model input".to_string(),
            detail: format!(
                "Projected input {projected_h}x{projected_w} is below half of source {source_h}x{source_w} on at least one axis; task accuracy and small-object/segmentation behavior are not modeled."
            ),
            evidence_class: "DERIVED_GEOMETRY_WITH_UNASSESSED_TASK_EFFECT".to_string(),
        });
    }
    if !request.activation_dtype.eq_ignore_ascii_case("source") {
        constraints.push(RedesignConstraint {
            severity: "warning".to_string(),
            code: "RD-QUANT-001".to_string(),
            scope: "model".to_string(),
            detail: "Storage-width projection does not synthesize calibration data, valid scales/zero-points, QAT behavior, or a transformed artifact. Delegate assignment is therefore suppressed.".to_string(),
            evidence_class: "NOT_ASSESSABLE_HYPOTHETICAL_QUANTIZATION".to_string(),
        });
    }
    for edit in &request.block_edits {
        if edit
            .expand_ratio
            .map(|ratio| !(2.0..=10.0).contains(&ratio))
            .unwrap_or(false)
        {
            constraints.push(RedesignConstraint {
                severity: "warning".to_string(),
                code: "RD-EXPAND-001".to_string(),
                scope: edit.block_id.clone(),
                detail: format!(
                    "Expand ratio {:.2} is outside the conventional 2..=10 review band; this is a heuristic architecture warning, not a correctness failure.",
                    edit.expand_ratio.unwrap_or_default()
                ),
                evidence_class: "HEURISTIC".to_string(),
            });
        }
    }
}

fn append_merge_contract_constraints(
    tensors: &[TensorInfo],
    ops: &[OpInfo],
    constraints: &mut Vec<RedesignConstraint>,
) {
    for op in ops {
        if !matches!(
            op.name.as_str(),
            "ADD" | "MUL" | "SUB" | "DIV" | "MAXIMUM" | "MINIMUM"
        ) {
            continue;
        }
        let inputs = op
            .inputs
            .iter()
            .filter_map(|index| usize::try_from(*index).ok())
            .filter_map(|index| tensors.get(index))
            .filter(|tensor| !tensor.constant_buffer)
            .collect::<Vec<_>>();
        for left_index in 0..inputs.len() {
            for right_index in (left_index + 1)..inputs.len() {
                let left = inputs[left_index];
                let right = inputs[right_index];
                if broadcast_shapes_compatible(&left.shape, &right.shape) {
                    continue;
                }
                constraints.push(RedesignConstraint {
                    severity: "error".to_string(),
                    code: "RD-SHAPE-MERGE-001".to_string(),
                    scope: format!("#{:03} {}", op.index, op.name),
                    detail: format!(
                        "Projected nonconstant inputs T{} {:?} and T{} {:?} are not broadcast-compatible. The redesign is blocked; inserting or changing a projection branch is an architecture edit that cannot be inferred from the deployment artifact.",
                        left.index, left.shape, right.index, right.shape
                    ),
                    evidence_class: "DERIVED_PROJECTED_TENSOR_CONTRACT".to_string(),
                });
            }
        }
    }
}

fn broadcast_shapes_compatible(left: &[i32], right: &[i32]) -> bool {
    let rank = left.len().max(right.len());
    (0..rank).all(|offset| {
        let left_dimension = left
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| left.get(index))
            .copied()
            .unwrap_or(1);
        let right_dimension = right
            .len()
            .checked_sub(offset + 1)
            .and_then(|index| right.get(index))
            .copied()
            .unwrap_or(1);
        left_dimension > 0
            && right_dimension > 0
            && (left_dimension == right_dimension || left_dimension == 1 || right_dimension == 1)
    })
}

#[allow(clippy::too_many_arguments)]
fn build_projection_ledger(
    source_ops: &[OpInfo],
    projected_ops: &[OpInfo],
    source_tensors: &[TensorInfo],
    projected_tensors: &[TensorInfo],
    blocks: &[BlockRecord],
    edits: &HashMap<String, RedesignBlockEdit>,
    shape_status_by_op: &BTreeMap<usize, ShapeRuleStatus>,
    width_multiplier: f64,
    projected_h: usize,
    projected_w: usize,
    source_h: usize,
    source_w: usize,
    activation_dtype: &str,
    constraints: &[RedesignConstraint],
    target: &TargetProfile,
) -> (
    Vec<RedesignOpProjection>,
    Vec<RedesignPropagationEdge>,
    RedesignImpactSummary,
) {
    let source_tensor_by_index = source_tensors
        .iter()
        .map(|tensor| (tensor.index, tensor))
        .collect::<BTreeMap<_, _>>();
    let projected_tensor_by_index = projected_tensors
        .iter()
        .map(|tensor| (tensor.index, tensor))
        .collect::<BTreeMap<_, _>>();
    let source_op_by_index = source_ops
        .iter()
        .map(|op| (op.index, op))
        .collect::<BTreeMap<_, _>>();
    let block_by_op = block_ownership(blocks);
    let producer_by_tensor = projected_ops
        .iter()
        .flat_map(|op| {
            op.outputs.iter().filter_map(move |tensor| {
                usize::try_from(*tensor)
                    .ok()
                    .map(|tensor_index| (tensor_index, op.index))
            })
        })
        .collect::<BTreeMap<_, _>>();
    let global_projection = (width_multiplier - 1.0).abs() > f64::EPSILON
        || projected_h != source_h
        || projected_w != source_w
        || activation_dtype != "source";

    let changed_tensor_indices = projected_tensor_by_index
        .iter()
        .filter_map(|(index, projected)| {
            source_tensor_by_index
                .get(index)
                .filter(|source| source.shape != projected.shape || source.dtype != projected.dtype)
                .map(|_| *index)
        })
        .collect::<BTreeSet<_>>();

    let direct_fields_by_op = projected_ops
        .iter()
        .filter_map(|op| {
            let block = block_by_op.get(&op.index).copied()?;
            let edit = edits.get(&block.block_id)?;
            let fields = direct_edit_fields(op, block, edit, source_ops);
            (!fields.is_empty()).then_some((op.index, fields))
        })
        .collect::<BTreeMap<_, _>>();

    let mut propagation_edges = Vec::new();
    let mut incoming_changed_producers = BTreeMap::<usize, BTreeSet<usize>>::new();
    for consumer in projected_ops {
        for tensor in &consumer.inputs {
            let Some(tensor_index) = usize::try_from(*tensor).ok() else {
                continue;
            };
            let Some(producer_op_index) = producer_by_tensor.get(&tensor_index).copied() else {
                continue;
            };
            let Some(source_tensor) = source_tensor_by_index.get(&tensor_index).copied() else {
                continue;
            };
            let Some(projected_tensor) = projected_tensor_by_index.get(&tensor_index).copied()
            else {
                continue;
            };
            let changed = source_tensor.shape != projected_tensor.shape
                || source_tensor.dtype != projected_tensor.dtype;
            if changed {
                incoming_changed_producers
                    .entry(consumer.index)
                    .or_default()
                    .insert(producer_op_index);
            }
            propagation_edges.push(RedesignPropagationEdge {
                tensor_index,
                producer_op_index,
                consumer_op_index: consumer.index,
                source_shape: source_tensor.shape.clone(),
                projected_shape: projected_tensor.shape.clone(),
                source_dtype: source_tensor.dtype.clone(),
                projected_dtype: projected_tensor.dtype.clone(),
                changed,
                change_class: if changed {
                    if direct_fields_by_op.contains_key(&producer_op_index) {
                        "direct_edit_output"
                    } else {
                        "propagated_contract"
                    }
                } else {
                    "unchanged"
                }
                .to_string(),
            });
        }
    }
    propagation_edges.sort_by_key(|edge| {
        (
            edge.producer_op_index,
            edge.consumer_op_index,
            edge.tensor_index,
        )
    });

    let mut roots_by_op = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut change_class_by_op = BTreeMap::<usize, String>::new();
    for op in projected_ops {
        let mut roots = BTreeSet::new();
        if direct_fields_by_op.contains_key(&op.index) {
            roots.insert(op.index);
        }
        for producer in incoming_changed_producers
            .get(&op.index)
            .into_iter()
            .flatten()
        {
            if direct_fields_by_op.contains_key(producer) {
                roots.insert(*producer);
            }
            if let Some(producer_roots) = roots_by_op.get(producer) {
                roots.extend(producer_roots);
            }
        }
        let has_changed_contract = op
            .inputs
            .iter()
            .chain(op.outputs.iter())
            .filter_map(|index| usize::try_from(*index).ok())
            .any(|index| changed_tensor_indices.contains(&index));
        let change_class = if direct_fields_by_op.contains_key(&op.index) {
            "direct_edit"
        } else if !roots.is_empty() && has_changed_contract {
            "propagated_contract"
        } else if global_projection && has_changed_contract {
            "global_projection"
        } else {
            "unchanged"
        };
        roots_by_op.insert(op.index, roots);
        change_class_by_op.insert(op.index, change_class.to_string());
    }

    let mut rows = Vec::with_capacity(projected_ops.len());
    for projected_op in projected_ops {
        let Some(source_op) = source_op_by_index.get(&projected_op.index).copied() else {
            continue;
        };
        let roots = roots_by_op
            .get(&projected_op.index)
            .cloned()
            .unwrap_or_default();
        let related = projected_ops
            .iter()
            .filter(|candidate| candidate.index != projected_op.index)
            .filter(|candidate| {
                let candidate_roots = roots_by_op.get(&candidate.index);
                !roots.is_empty() && candidate_roots.is_some_and(|items| !items.is_disjoint(&roots))
            })
            .map(|candidate| candidate.index)
            .collect::<Vec<_>>();
        let source_steady_us = (source_op.bottleneck_total_us
            - source_op.bottleneck_packing_us
            - source_op.bottleneck_break_us)
            .max(0.0);
        let projected_steady_us = (projected_op.bottleneck_total_us
            - projected_op.bottleneck_packing_us
            - projected_op.bottleneck_break_us)
            .max(0.0);
        rows.push(RedesignOpProjection {
            op_index: projected_op.index,
            op_name: projected_op.name.clone(),
            block_id: block_by_op
                .get(&projected_op.index)
                .map(|block| block.block_id.clone())
                .unwrap_or_default(),
            change_class: change_class_by_op
                .get(&projected_op.index)
                .cloned()
                .unwrap_or_else(|| "unchanged".to_string()),
            shape_rule_status: shape_status_by_op
                .get(&projected_op.index)
                .copied()
                .unwrap_or(ShapeRuleStatus::Unassessed)
                .label()
                .to_string(),
            direct_edit_fields: direct_fields_by_op
                .get(&projected_op.index)
                .cloned()
                .unwrap_or_default(),
            propagation_source_op_indices: roots.into_iter().collect(),
            related_op_indices: related,
            source_inputs: tensor_contracts(&source_op.inputs, &source_tensor_by_index),
            projected_inputs: tensor_contracts(&projected_op.inputs, &projected_tensor_by_index),
            source_outputs: tensor_contracts(&source_op.outputs, &source_tensor_by_index),
            projected_outputs: tensor_contracts(&projected_op.outputs, &projected_tensor_by_index),
            source_macs: source_op.macs,
            projected_macs: projected_op.macs,
            source_steady_us,
            projected_steady_us,
            source_cold_us: source_op.bottleneck_total_us,
            projected_cold_us: projected_op.bottleneck_total_us,
            source_l1_ratio: logical_l1_ratio(source_op, target),
            projected_l1_ratio: logical_l1_ratio(projected_op, target),
        });
    }

    let count_class = |class_name: &str| {
        rows.iter()
            .filter(|row| row.change_class == class_name)
            .count()
    };
    let impact_summary = RedesignImpactSummary {
        direct_edit_op_count: count_class("direct_edit"),
        propagated_op_count: count_class("propagated_contract"),
        global_projection_op_count: count_class("global_projection"),
        unchanged_op_count: count_class("unchanged"),
        changed_tensor_count: changed_tensor_indices.len(),
        changed_edge_count: propagation_edges.iter().filter(|edge| edge.changed).count(),
        unresolved_contract_count: constraints
            .iter()
            .filter(|constraint| {
                constraint.severity == "error" && constraint.code.starts_with("RD-SHAPE-")
            })
            .count(),
    };
    (rows, propagation_edges, impact_summary)
}

fn direct_edit_fields(
    op: &OpInfo,
    block: &BlockRecord,
    edit: &RedesignBlockEdit,
    source_ops: &[OpInfo],
) -> Vec<String> {
    let compute_positions = block
        .op_indices
        .iter()
        .filter(|index| {
            source_ops
                .iter()
                .find(|candidate| candidate.index == **index)
                .is_some_and(|candidate| {
                    matches!(
                        candidate.name.as_str(),
                        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED"
                    )
                })
        })
        .copied()
        .collect::<Vec<_>>();
    let mut fields = Vec::new();
    if edit.output_channels.is_some() && compute_positions.last() == Some(&op.index) {
        fields.push("output_channels".to_string());
    }
    if edit.expand_ratio.is_some()
        && block.block_type == "inverted_bottleneck"
        && compute_positions.first() == Some(&op.index)
    {
        fields.push("expand_ratio".to_string());
    }
    if edit.kernel_size.is_some() && matches!(op.name.as_str(), "CONV_2D" | "DEPTHWISE_CONV_2D") {
        fields.push("kernel_size".to_string());
    }
    if edit.repeat.is_some() && block.op_indices.first() == Some(&op.index) {
        fields.push("repeat".to_string());
    }
    fields
}

fn tensor_contracts(
    indices: &[i32],
    tensors: &BTreeMap<usize, &TensorInfo>,
) -> Vec<RedesignTensorContract> {
    indices
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| {
            tensors.get(&index).map(|tensor| RedesignTensorContract {
                tensor_index: index,
                shape: tensor.shape.clone(),
                dtype: tensor.dtype.clone(),
                constant: tensor.constant_buffer,
            })
        })
        .collect()
}

fn logical_l1_ratio(op: &OpInfo, target: &TargetProfile) -> Option<f64> {
    (target.l1_data_bytes > 0)
        .then(|| {
            op.cache_payload
                .logical_row_payload_bytes
                .map(|bytes| bytes as f64 / target.l1_data_bytes as f64)
        })
        .flatten()
}

#[allow(clippy::too_many_arguments)]
fn build_block_diffs(
    blocks: &[BlockRecord],
    edits: &HashMap<String, RedesignBlockEdit>,
    projected_macs: &BTreeMap<String, f64>,
    projected_l1: &BTreeMap<String, f64>,
    width_multiplier: f64,
    projected_h: usize,
    projected_w: usize,
    source_h: usize,
    source_w: usize,
    activation_dtype: &str,
) -> Vec<RedesignBlockDiff> {
    blocks
        .iter()
        .filter_map(|block| {
            let edit = edits.get(&block.block_id);
            let mut changes = Vec::new();
            if (width_multiplier - 1.0).abs() > f64::EPSILON {
                changes.push(format!("global width multiplier 1.0 -> {width_multiplier:.3}"));
            }
            if projected_h != source_h || projected_w != source_w {
                changes.push(format!(
                    "input-bound spatial projection {source_h}x{source_w} -> {projected_h}x{projected_w}"
                ));
            }
            if activation_dtype != "source" {
                changes.push(format!("storage dtype source -> {activation_dtype}"));
            }
            if let Some(edit) = edit {
                if let Some(value) = edit.output_channels {
                    changes.push(format!(
                        "output channels {} -> {value}",
                        block
                            .channels
                            .output
                            .map(|item| item.to_string())
                            .unwrap_or_else(|| "unresolved".to_string())
                    ));
                }
                if let Some(value) = edit.expand_ratio {
                    changes.push(format!(
                        "expand ratio {} -> {value:.2}",
                        block
                            .params
                            .expand_ratio
                            .map(|item| format!("{item:.2}"))
                            .unwrap_or_else(|| "unresolved".to_string())
                    ));
                }
                if let Some(value) = edit.repeat {
                    changes.push(format!("repeat 1 -> {value}"));
                }
                if let Some(value) = edit.kernel_size {
                    changes.push(format!(
                        "kernel {} -> {value}",
                        block
                            .params
                            .kernel_h
                            .map(|item| item.to_string())
                            .unwrap_or_else(|| "unresolved".to_string())
                    ));
                }
            }
            if changes.is_empty() {
                return None;
            }
            Some(RedesignBlockDiff {
                block_id: block.block_id.clone(),
                display_name: block.display_name.clone(),
                changes,
                source_macs: block.aggregates.macs,
                projected_macs: projected_macs
                    .get(&block.block_id)
                    .copied()
                    .unwrap_or(block.aggregates.macs),
                source_l1_max_ratio: block.aggregates.l1_max_ratio,
                projected_l1_max_ratio: projected_l1.get(&block.block_id).copied(),
            })
        })
        .collect()
}

fn metric_delta(source: &RedesignMetrics, projected: &RedesignMetrics) -> RedesignMetricDelta {
    RedesignMetricDelta {
        mac_percent: percent_delta(source.macs, projected.macs),
        operations_percent: percent_delta(source.operations, projected.operations),
        parameter_percent: percent_delta(
            source.parameter_elements as f64,
            projected.parameter_elements as f64,
        ),
        modeled_latency_percent: percent_delta(
            source.modeled_latency_ms,
            projected.modeled_latency_ms,
        ),
        peak_live_activation_percent: option_percent_delta(
            source.peak_live_activation_bytes,
            projected.peak_live_activation_bytes,
        ),
        arena_percent: option_percent_delta(source.arena_bytes, projected.arena_bytes),
    }
}

fn percent_delta(source: f64, projected: f64) -> Option<f64> {
    (source > 0.0 && source.is_finite() && projected.is_finite())
        .then_some((projected - source) / source * 100.0)
}

fn option_percent_delta(source: Option<usize>, projected: Option<usize>) -> Option<f64> {
    source
        .zip(projected)
        .and_then(|(source, projected)| percent_delta(source as f64, projected as f64))
}

fn scaled_serialized_shape(
    source: &[i32],
    projection: &ShapeProjection,
    preserve_last: bool,
) -> Vec<i32> {
    let mut shape = source.to_vec();
    if shape.len() == 4 {
        shape[1] = scale_dimension(shape[1], projection.projected_h, projection.source_h) as i32;
        shape[2] = scale_dimension(shape[2], projection.projected_w, projection.source_w) as i32;
        if !preserve_last {
            if let Some(channel) = positive_i32(shape[3]) {
                shape[3] =
                    aligned_channels(channel, projection.width_multiplier, projection.alignment)
                        as i32;
            }
        }
    }
    shape
}

fn project_spatial_axis(
    source_input: Option<usize>,
    source_output: Option<usize>,
    projected_input: Option<usize>,
    kernel: Option<usize>,
) -> usize {
    let (Some(source_input), Some(source_output), Some(projected_input)) =
        (source_input, source_output, projected_input)
    else {
        return source_output.unwrap_or(1);
    };
    if source_input == source_output {
        return projected_input;
    }
    for stride in 1usize..=8 {
        if source_input.div_ceil(stride) == source_output {
            return projected_input.div_ceil(stride);
        }
        if let Some(kernel) = kernel {
            if source_input >= kernel && (source_input - kernel) / stride + 1 == source_output {
                return if projected_input >= kernel {
                    (projected_input - kernel) / stride + 1
                } else {
                    1
                };
            }
        }
    }
    scale_dimension(source_output as i32, projected_input, source_input)
}

fn scale_dimension(source: i32, projected_base: usize, source_base: usize) -> usize {
    let source = positive_i32(source).unwrap_or(1);
    source
        .saturating_mul(projected_base)
        .div_ceil(source_base.max(1))
        .max(1)
}

fn aligned_channels(source: usize, multiplier: f64, alignment: usize) -> usize {
    if (multiplier - 1.0).abs() <= f64::EPSILON {
        return source.max(1);
    }
    aligned_value(source as f64 * multiplier, alignment)
}

fn aligned_value(value: f64, alignment: usize) -> usize {
    let rounded = value.round().max(1.0) as usize;
    rounded.div_ceil(alignment.max(1)) * alignment.max(1)
}

fn storage_bytes(dtype: &str) -> Option<usize> {
    let value = bytes_per_type(dtype);
    (value.is_finite() && value >= 1.0).then_some(value as usize)
}

fn positive(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

fn positive_i32(value: i32) -> Option<usize> {
    positive(value)
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        reviewed_implementation_codegen_contract, unique_concatenation_axis, ReviewedCodegenInput,
    };

    #[test]
    fn concatenation_axis_requires_one_shape_conserving_dimension() {
        let inputs = vec![vec![1, 8, 8, 4], vec![1, 8, 8, 6]];
        let output = vec![1, 8, 8, 10];
        assert_eq!(unique_concatenation_axis(&inputs, Some(&output)), Some(3));

        let ambiguous_inputs = vec![vec![1, 1], vec![1, 1]];
        let ambiguous_output = vec![2, 2];
        assert_eq!(
            unique_concatenation_axis(&ambiguous_inputs, Some(&ambiguous_output)),
            None
        );
    }

    #[test]
    fn weight_free_binary_codegen_rejects_hidden_constant_operands() {
        let input_shapes = [vec![1, 8]];
        let output_shapes = [vec![1, 8]];
        let (_, status, reason, _) =
            reviewed_implementation_codegen_contract(&ReviewedCodegenInput {
                op_name: "MUL",
                activation_input_count: 1,
                output_count: 1,
                input_shapes: &input_shapes,
                output_shapes: &output_shapes,
                input_channels: Some(8),
                output_channels: Some(8),
                kernel_height: None,
                kernel_width: None,
                stride_height: None,
                stride_width: None,
            });
        assert_eq!(status, "unsupported");
        assert!(reason.contains("serialized operands are constants"));
    }

    #[test]
    fn depthwise_codegen_requires_integral_channel_multiplier() {
        let input_shapes = [vec![1, 8, 8, 3]];
        let output_shapes = [vec![1, 8, 8, 5]];
        let (kind, status, _, _) =
            reviewed_implementation_codegen_contract(&ReviewedCodegenInput {
                op_name: "DEPTHWISE_CONV_2D",
                activation_input_count: 1,
                output_count: 1,
                input_shapes: &input_shapes,
                output_shapes: &output_shapes,
                input_channels: Some(3),
                output_channels: Some(5),
                kernel_height: Some(3),
                kernel_width: Some(3),
                stride_height: Some(1),
                stride_width: Some(1),
            });
        assert_eq!(kind, "unsupported");
        assert_eq!(status, "unsupported");
    }
}
