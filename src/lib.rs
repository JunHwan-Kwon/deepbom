mod accumulator_atlas;
mod accumulator_reachability;
mod arena_planner;
mod artifact_byte_ledger;
mod block_inventory;
mod channel_vitality;
mod contract_migration;
mod core_isolation;
mod delegation_repair;
mod deployment_delta;
mod deployment_frontier;
mod dynamic_shape_cost;
mod findings;
mod influence;
mod input_counterexample;
mod kernel_witness;
mod numerical_abi_propagation;
mod preprocessing;
mod preprocessing_realizability;
mod quantization_lattice;
mod quantization_math;
mod redesign;
mod report_exports;
mod requantization_fidelity;
mod research;
mod residual_contract_distortion;
mod residual_step_response;
mod rounding_equivalence;
mod runtime_version;
mod target_profiles;
mod tflite_deep_scopes;
mod tflite_metadata;
mod tflite_operator_names;
mod tflite_scalar;
mod tflite_sparse;
mod tflite_subgraphs;
mod xnnpack_delegate;
mod xnnpack_rulepack_generated;

pub use delegation_repair::*;
pub use deployment_delta::*;
pub use deployment_frontier::*;
use report_exports::{render_roofline_csv, render_stage_mermaid};
pub use research::*;
use research::{bias_to_f64, weight_to_f64};
use runtime_version::pinned_runtime_version_for_op;

use accumulator_atlas::{build_accumulator_atlas, AccumulatorAtlasAnalysis};
use accumulator_reachability::{
    accumulator_reachability_not_computed, build_accumulator_reachability,
    AccumulatorReachabilityAnalysis,
};
use arena_planner::{
    compute_movement_analysis, compute_tensor_arena_plan, declared_tensor_payload_bytes,
};
use artifact_byte_ledger::{build_artifact_byte_integrity_ledger, ArtifactByteIntegrityLedger};
use block_inventory::{build_block_inventory, BlockInventory};
use channel_vitality::{
    build_channel_vitality, channel_vitality_not_computed, ChannelVitalityAnalysis,
};
use contract_migration::{build_contract_migration, ContractMigrationAnalysis};
use dynamic_shape_cost::{build_tflite_dynamic_shape_cost_contract, DynamicShapeCostContract};
use findings::{build_findings_from_analysis, Finding, FindingAnalysisContext};
use influence::{compute_influence_bwd, compute_influence_fwd, uf_find};
use input_counterexample::{
    build_input_counterexamples, input_counterexample_not_computed, InputCounterexampleAnalysis,
};
use kernel_witness::{build_kernel_witnesses, kernel_witness_not_computed, KernelWitnessAnalysis};
use numerical_abi_propagation::{
    build_numerical_abi_propagation, numerical_abi_propagation_not_computed,
    NumericalAbiPropagationAnalysis,
};
#[cfg(test)]
use preprocessing::pack_rgba_float32;
use preprocessing_realizability::{
    build_preprocessing_realizability, preprocessing_realizability_not_computed,
    PreprocessingRealizabilityAnalysis,
};
use quantization_lattice::{build_quantization_lattice, QuantizationLatticeAnalysis};
use redesign::{build_redesign_pareto, build_redesign_projection, RedesignRequest};
use requantization_fidelity::{build_requantization_fidelity, RequantizationFidelityAnalysis};
use residual_contract_distortion::{
    build_residual_contract_distortion, residual_contract_distortion_not_computed,
    ResidualContractDistortionAnalysis,
};
use residual_step_response::{
    build_residual_step_response, residual_step_response_not_computed, ResidualStepResponseAnalysis,
};
use rounding_equivalence::{
    build_rounding_equivalence, rounding_equivalence_not_computed, RoundingEquivalenceAnalysis,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
#[cfg(test)]
use target_profiles::custom_target_profile;
use target_profiles::{
    all_target_profiles, is_cortex_a55_profile, is_delegate_break_suspect, target_profile,
};
use tflite_deep_scopes::{
    build_deep_analysis, NestedScopeInput, PrimaryScopeInput, TfliteSubgraphDeepAnalysis,
};
use tflite_metadata::{
    bind_packed_associated_files, parse_packed_metadata_archive, parse_tflite_model_metadata,
    MetadataAssociatedFile, MetadataProcessUnit, PackedMetadataFile, ParsedTfliteModelMetadata,
};
use tflite_operator_names::operator_code_name;
use tflite_scalar::{f16_to_f32, tensor_type_name};
use tflite_sparse::{
    build_sparse_storage_contract, parse_sparsity, SparseStorageContract, SparseTensorEncoding,
};
use tflite_subgraphs::{build_tflite_subgraph_inventory, TfliteSubgraphInventory};
use wasm_bindgen::prelude::*;
use xnnpack_delegate::{
    quantized_fully_connected_has_rank_predicate, support_for_op as xnnpack_support_for_op,
    XnnpackOpContext,
};

const PACKING_WARN_OVERHEAD_US: f64 = 10.0;
const MEANINGFUL_CHAIN_MAC_PERCENT: f64 = 0.01;
const MEANINGFUL_FALLBACK_BYTE_PERCENT: f64 = 0.01;
const RUNTIME_OK: u32 = 0;
const RUNTIME_EXPIRED: u32 = 1;
const RUNTIME_CLOCK_INVALID: u32 = 2;
const WEIGHT_NEAR_ZERO_EPS: f64 = 1e-8;

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

#[wasm_bindgen]
pub fn runtime_guard() -> u32 {
    runtime_guard_code()
}

#[wasm_bindgen]
pub fn analyze_tflite(bytes: &[u8], filename: &str) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let analysis = analyze(bytes, filename).map_err(|err| JsValue::from_str(&err))?;
    serde_wasm_bindgen::to_value(&analysis).map_err(|err| JsValue::from_str(&err.to_string()))
}

#[wasm_bindgen]
pub fn analyze_tflite_for_target(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let analysis =
        analyze_with_target(bytes, filename, target_id).map_err(|err| JsValue::from_str(&err))?;
    serde_wasm_bindgen::to_value(&analysis).map_err(|err| JsValue::from_str(&err.to_string()))
}

#[wasm_bindgen]
pub fn project_tflite_redesign(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
    request: JsValue,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let request: RedesignRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("Invalid redesign request: {error}")))?;
    let analysis = analyze_with_target_without_step_response(bytes, filename, target_id)
        .map_err(|error| JsValue::from_str(&error))?;
    let projection = build_redesign_projection(bytes, &analysis, request)
        .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&projection).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
pub fn explore_tflite_redesign_pareto(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
    request: JsValue,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let request: RedesignRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("Invalid Pareto base request: {error}")))?;
    let analysis = analyze_with_target_without_step_response(bytes, filename, target_id)
        .map_err(|error| JsValue::from_str(&error))?;
    let search = build_redesign_pareto(bytes, &analysis, request)
        .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&search).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Compute weight histogram + filter stats for a specific tensor by index.
/// Returns null if the tensor is not a constant buffer or is unsupported dtype.
#[wasm_bindgen]
pub fn compute_weight_histogram(
    bytes: &[u8],
    filename: &str,
    tensor_index: usize,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let _ = (filename, target_id);
    let tensors =
        read_primary_subgraph_tensors(bytes).map_err(|error| JsValue::from_str(&error))?;
    let tensor = tensors
        .get(tensor_index)
        .ok_or_else(|| JsValue::from_str("tensor_index out of range"))?;
    let hist = compute_weight_histogram_for_tensor(bytes, tensor)
        .ok_or_else(|| JsValue::from_str("tensor has no constant buffer or unsupported dtype"))?;
    serde_wasm_bindgen::to_value(&hist).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Backward (input) influence: spatial BFS from clicked op's output → model inputs.
#[wasm_bindgen]
pub fn compute_input_influence(
    bytes: &[u8],
    filename: &str,
    op_index: usize,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let tid = if target_id.is_empty() {
        "android_mid_a55"
    } else {
        target_id
    };
    let analysis = analyze_with_target_without_step_response(bytes, filename, tid)
        .map_err(|e| JsValue::from_str(&e))?;
    match compute_influence_bwd(&analysis, bytes, op_index) {
        Some(r) => serde_wasm_bindgen::to_value(&r).map_err(|e| JsValue::from_str(&e.to_string())),
        None => Ok(JsValue::NULL),
    }
}

/// Forward (output) influence: spatial BFS from clicked op's output → model outputs.
#[wasm_bindgen]
pub fn compute_output_influence(
    bytes: &[u8],
    filename: &str,
    op_index: usize,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let tid = if target_id.is_empty() {
        "android_mid_a55"
    } else {
        target_id
    };
    let analysis = analyze_with_target_without_step_response(bytes, filename, tid)
        .map_err(|e| JsValue::from_str(&e))?;
    match compute_influence_fwd(&analysis, bytes, op_index) {
        Some(r) => serde_wasm_bindgen::to_value(&r).map_err(|e| JsValue::from_str(&e.to_string())),
        None => Ok(JsValue::NULL),
    }
}

/// Fast low-norm-filter count for an op weight tensor (L2 < 2% of max).
#[wasm_bindgen]
pub fn compute_quick_low_norm_stat(
    bytes: &[u8],
    filename: &str,
    tensor_index: usize,
    target_id: &str,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let _ = (filename, target_id);
    let tensors =
        read_primary_subgraph_tensors(bytes).map_err(|error| JsValue::from_str(&error))?;
    let tensor = tensors
        .get(tensor_index)
        .ok_or_else(|| JsValue::from_str("tensor_index out of range"))?;
    let inferred_depthwise = tensor.shape.len() == 4 && tensor.shape.first() == Some(&1);
    let stat = compute_quick_low_norm_stat_for_tensor(bytes, tensor, inferred_depthwise)
        .ok_or_else(|| JsValue::from_str("tensor has no constant buffer or unsupported dtype"))?;
    serde_wasm_bindgen::to_value(&stat).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute correction factor between static latency estimate and WASM-measured runtime.
/// Pass measured_ms = 0.0 if no runtime data is available (returns static estimate only).
#[wasm_bindgen]
pub fn compute_static_runtime_calibration(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
    measured_ms: f64,
) -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    let analysis = analyze_with_target_without_step_response(bytes, filename, target_id)
        .map_err(|err| JsValue::from_str(&err))?;
    // Runtime benchmark percentiles are post-warmup, so the calibration denominator
    // excludes one-time packing and the setup-only partition-planning profile.
    let assessed_op_count = analysis
        .ops
        .iter()
        .filter(|op| op.bottleneck_total_us.is_finite() && op.bottleneck_total_us >= 0.0)
        .count();
    let cold_start_static_estimate_ms = analysis
        .ops
        .iter()
        .filter_map(|op| {
            (op.bottleneck_total_us.is_finite() && op.bottleneck_total_us >= 0.0)
                .then_some(op.bottleneck_total_us)
        })
        .sum::<f64>()
        / 1000.0;
    let one_time_packing_ms = analysis
        .ops
        .iter()
        .map(|op| op.bottleneck_packing_us.max(0.0))
        .sum::<f64>()
        / 1000.0;
    let boundary_setup_ms = analysis
        .ops
        .iter()
        .map(|op| op.bottleneck_break_us.max(0.0))
        .sum::<f64>()
        / 1000.0;
    let static_estimate_ms =
        (cold_start_static_estimate_ms - one_time_packing_ms - boundary_setup_ms).max(0.0);

    let (correction_factor, interpretation, confidence) = if measured_ms > 0.0
        && static_estimate_ms > 0.0
    {
        let factor = measured_ms / static_estimate_ms;
        let interp = if factor < 0.5 {
            format!("WASM runtime ({:.1}ms) is {:.1}x faster than the static target estimate; runtime kernels and the selected static target profile differ", measured_ms, 1.0 / factor)
        } else if factor <= 2.0 {
            format!(
                "WASM runtime ({:.1}ms) is {:.2}x of the static target estimate ({:.1}ms)",
                measured_ms, factor, static_estimate_ms
            )
        } else {
            format!(
                "WASM runtime ({:.1}ms) is {:.1}x slower than the static target estimate; {}",
                measured_ms,
                factor,
                if analysis.xnnpack_chain_breaks > 0 {
                    format!(
                        "the static graph predicts {} partition boundary operator(s), while runtime attribution remains unmeasured",
                        analysis.xnnpack_chain_breaks
                    )
                } else {
                    "runtime overhead attribution is not measured".to_string()
                }
            )
        };
        (Some(factor), interp, "static+wasm".to_string())
    } else if measured_ms > 0.0 {
        (
            None,
            format!(
                "WASM runtime measured at {:.1}ms, but the static estimate denominator is zero; no correction factor was computed",
                measured_ms
            ),
            "not-assessed".to_string(),
        )
    } else {
        (
            None,
            format!(
                "Static estimate only: {:.1}ms (no WASM runtime data provided)",
                static_estimate_ms
            ),
            "static".to_string(),
        )
    };

    let calibration = StaticRuntimeCalibration {
        static_estimate_ms,
        cold_start_static_estimate_ms,
        one_time_packing_ms,
        boundary_setup_ms,
        measured_ms,
        correction_factor,
        interpretation,
        confidence,
        method: "steady_static_ms = sum(op.bottleneck_total_us - op.bottleneck_packing_us - op.bottleneck_break_us) / 1000; cold_start_static_ms = steady_static_ms + one_time_packing_ms + boundary_setup_ms. Post-warmup measured_ms is compared only with steady_static_ms; boundary_setup_ms is an unmeasured planning-profile constant, not observed per-inference latency.".to_string(),
        assessed_op_count,
        total_op_count: analysis.ops.len(),
    };
    serde_wasm_bindgen::to_value(&calibration).map_err(|err| JsValue::from_str(&err.to_string()))
}

#[wasm_bindgen]
pub fn target_profiles() -> Result<JsValue, JsValue> {
    ensure_runtime_allowed()?;
    serde_wasm_bindgen::to_value(&all_target_profiles())
        .map_err(|err| JsValue::from_str(&err.to_string()))
}

fn ensure_runtime_allowed() -> Result<(), JsValue> {
    match runtime_guard_code() {
        RUNTIME_OK => Ok(()),
        RUNTIME_EXPIRED => Err(JsValue::from_str("APP_RUNTIME_EXPIRED")),
        RUNTIME_CLOCK_INVALID => Err(JsValue::from_str("APP_RUNTIME_CLOCK_INVALID")),
        _ => Err(JsValue::from_str("APP_RUNTIME_LOCKED")),
    }
}

fn runtime_guard_code() -> u32 {
    let now_ms = js_sys::Date::now();
    if !now_ms.is_finite() || now_ms < APP_NOT_BEFORE_EPOCH_MS {
        return RUNTIME_CLOCK_INVALID;
    }
    if now_ms > APP_EXPIRES_AT_EPOCH_MS {
        return RUNTIME_EXPIRED;
    }
    RUNTIME_OK
}

#[derive(Clone, Serialize)]
struct TensorInfo {
    index: usize,
    name: String,
    shape: Vec<i32>,
    shape_signature: Vec<i32>,
    dtype: String,
    buffer_index: i32,
    buffer_data_offset: usize,
    buffer_data_length: usize,
    constant_buffer: bool,
    sparse_storage: bool,
    #[serde(skip)]
    sparse_encoding: Option<SparseTensorEncoding>,
    quant_scales: usize,
    quant_zero_points: usize,
    quantized_dimension: i32,
    scale_sample: Vec<f32>,
    zero_point_sample: Vec<i64>,
    scale_min: f32,
    scale_max: f32,
    scale_ratio: f64,
    scale_mean: f64,
    scale_stddev: f64,
    scale_cv: f64,
    zero_point_min: i64,
    zero_point_max: i64,
    zero_point_offset_max: i64,
    zero_point_status: String,
    zero_point_detail: String,
    scale_mode: String,
    scale_ratio_meaningful: bool,
    quant_risk: String,
    // FNV-1a64 fingerprint of the constant buffer bytes (hex; empty when not constant).
    // Same (length, hash) pair across tensors ⇒ duplicated weight data in the file.
    buffer_hash: String,
    // Tensor.is_variable (schema field 5): stateful runtime tensor (RNN/LSTM state).
    is_variable: bool,
}

#[derive(Clone, Serialize)]
struct CachePayloadBreakdown {
    schema: String,
    status: String,
    evidence_class: String,
    input_strip_bytes: Option<usize>,
    output_row_bytes: Option<usize>,
    logical_row_payload_bytes: Option<usize>,
    serialized_kernel_bytes: Option<usize>,
    serialized_bias_bytes: Option<usize>,
    input_width: Option<usize>,
    input_channels: Option<usize>,
    output_width: Option<usize>,
    output_channels: Option<usize>,
    kernel_height: Option<usize>,
    kernel_width: Option<usize>,
    effective_kernel_height: Option<usize>,
    input_dtype: Option<String>,
    output_dtype: Option<String>,
    method: String,
    interpretation_boundary: String,
}

impl CachePayloadBreakdown {
    fn not_applicable(op_name: &str) -> Self {
        Self {
            schema: "deepbom.cache_payload.v1".to_string(),
            status: "not_applicable".to_string(),
            evidence_class: "NOT_APPLICABLE".to_string(),
            input_strip_bytes: None,
            output_row_bytes: None,
            logical_row_payload_bytes: None,
            serialized_kernel_bytes: None,
            serialized_bias_bytes: None,
            input_width: None,
            input_channels: None,
            output_width: None,
            output_channels: None,
            kernel_height: None,
            kernel_width: None,
            effective_kernel_height: None,
            input_dtype: None,
            output_dtype: None,
            method: format!(
                "No logical row-payload decomposition is defined for TFLite {}.",
                op_name
            ),
            interpretation_boundary: "Not a cache-residency observation.".to_string(),
        }
    }
}

#[derive(Clone, Serialize)]
struct OpInfo {
    index: usize,
    name: String,
    version: i32,
    stage_index: Option<usize>,
    stage_key: Option<String>,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    output_shapes: Vec<Vec<i32>>,
    macs: f64,
    mac_percent: f64,
    ops: f64,
    estimated_bytes: f64,
    fallback_byte_percent: f64,
    row_working_set_bytes: f64,
    row_working_set_ratio: f64,
    row_working_set_severity: String,
    cache_payload: CachePayloadBreakdown,
    intensity_ops_per_byte: f64,
    static_bound_guess: String,
    static_action: String,
    roofline_reason: String,
    fused_activation: String,
    fusion_status: String,
    fusion_detail: String,
    xnnpack_supported: bool,
    xnnpack_reason: String,
    xnnpack_chain_id: i32,
    xnnpack_chain_role: String,
    xnnpack_chain_break: bool,
    xnnpack_break_class: String,
    chain_break_impact_mac_percent: f64,
    chain_break_overhead_us_low: f64,
    chain_break_overhead_us_high: f64,
    target_microkernel_hint: String,
    /// Stable {dtype}_{kernel} key naming the compute kernel family this op
    /// lowers to, so a target profile can carry a measured utilization per
    /// family instead of one scalar for the whole device.
    compute_kernel_class: String,
    xnnpack_kernel_candidate: String,
    xnnpack_kernel_tile_mr: usize,
    xnnpack_kernel_tile_nr: usize,
    xnnpack_kernel_channel_tile: usize,
    xnnpack_kernel_primary_tile: usize,
    xnnpack_kernel_source: String,
    xnnpack_kernel_evidence_class: String,
    xnnpack_kernel_selector_status: String,
    xnnpack_build_requirement: String,
    xnnpack_kernel_candidates: Vec<KernelSourceCandidate>,
    xnnpack_kernel_alignment_multiples: Vec<usize>,
    quantized_path: bool,
    quantized_compute_path: bool,
    quantization_state: String,
    quantization_detail: String,
    weight_bytes: f64,
    weight_packing_overhead_us: f64,
    weight_packing_risk: String,
    weight_packing_detail: String,
    output_channels: i32,
    channel_alignment_multiple: usize,
    channel_alignment_status: String,
    channel_alignment_detail: String,
    channel_tail_overhead_percent: f64,
    channel_tail_overhead_percent_min: f64,
    channel_tail_overhead_percent_max: f64,
    quant_scale_ratio: f64,
    quant_scale_cv: f64,
    quant_scale_mode: String,
    quant_scale_ratio_meaningful: bool,
    quant_zero_point_offset: i64,
    quant_zero_point_risk: String,
    quant_zero_point_status: String,
    quant_risk: String,
    quant_risk_detail: String,
    low_norm_filter_count: Option<usize>,
    low_norm_filter_total: Option<usize>,
    quant_hole: bool,
    quant_hole_class: String,
    quant_hole_detail: String,
    patterns: Vec<String>,
    // Topology annotations (computed after full graph is parsed)
    topo_role: String, // "through" | "branch-merge" | "branch-split" | "quant-boundary"
    topo_depth: usize,
    topo_fan_out_max: usize,
    // Per-op roofline bottleneck estimate (μs)
    bottleneck_compute_us: f64,
    bottleneck_memory_us: f64,
    bottleneck_packing_us: f64,
    bottleneck_break_us: f64,
    bottleneck_fallback_us: f64,
    bottleneck_total_us: f64,
    bottleneck_dominant: String, // steady-state: "compute" | "memory" | "fallback"
}

#[derive(Serialize)]
struct QuantHoleInfo {
    op_index: usize,
    op_name: String,
    hole_class: String,
    prev_op_name: String,
    next_op_name: String,
    from_dtype: String,
    to_dtype: String,
    adjacent_mac_percent: f64,
    detail: String,
}

#[derive(Clone, Serialize)]
struct CountItem {
    name: String,
    count: usize,
}

#[derive(Clone, Serialize)]
struct StageInfo {
    index: usize,
    key: String,
    first_op: usize,
    last_op: usize,
    op_count: usize,
    macs: f64,
    mac_percent: f64,
    delegated_macs: f64,
    fallback_macs: f64,
    delegated_mac_percent: f64,
    fallback_mac_percent: f64,
    delegated_ops: usize,
    fallback_ops: usize,
    delegated_op_percent: f64,
    fallback_op_percent: f64,
    estimated_bytes: f64,
    channels: Vec<i32>,
    xnnpack_chain_breaks: usize,
    patterns: Vec<String>,
}

#[derive(Serialize)]
struct TargetHardwareSource {
    document: String,
    revision: String,
    pages: String,
    sha256: String,
    url: String,
    scope: String,
}

#[derive(Serialize)]
struct TargetHardwareSpec {
    evidence_class: String,
    scope: String,
    configuration_context: String,
    core_configuration: String,
    max_clock_mhz: Option<usize>,
    l1_instruction_bytes: usize,
    l1_data_bytes: usize,
    l1_instruction_ways: usize,
    l1_data_ways: usize,
    l1_line_bytes: usize,
    l2_bytes: usize,
    l2_ways: usize,
    l2_line_bytes: usize,
    advanced_simd: bool,
    fp16_vector_arithmetic: bool,
    sources: Vec<TargetHardwareSource>,
}

#[derive(Clone, Serialize)]
struct KernelUtilizationEntry {
    kernel_class: String,
    utilization: f64,
}

/// Where a custom profile came from, so a retuned number can always be traced
/// back to the built-in it was derived from and to the evidence behind it.
#[derive(Clone, Serialize)]
struct TargetProfileDerivation {
    base_profile_id: String,
    base_profile_sha256: String,
    overridden_fields: Vec<String>,
    evidence_note: String,
}

#[derive(Serialize)]
struct TargetProfile {
    id: String,
    label: String,
    profile_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    derived_from: Option<TargetProfileDerivation>,
    architecture: String,
    core_count_min: Option<usize>,
    core_count_max: Option<usize>,
    /// Number of homogeneous cores represented by effective_peak_gops. Core
    /// partition scenarios are suppressed when this denominator is unbound.
    performance_reference_core_count: Option<usize>,
    l1_data_bytes: usize,
    l2_bytes: usize,
    l2_capacity_scope: String,
    cache_assumption: String,
    cache_source_url: String,
    hardware_spec: Option<TargetHardwareSpec>,
    performance_model_evidence_class: String,
    performance_model_assumption: String,
    simd_width_bits: usize,
    fp32_lanes: usize,
    fp16_lanes: usize,
    int8_lanes: usize,
    in_order: bool,
    dot_product: bool,
    sve2: bool,
    xnnpack_kernel_family: String,
    effective_memory_bandwidth_gbps: f64,
    effective_peak_gops: f64,
    compute_utilization_factor: f64,
    /// Measured utilization per compute kernel family, keyed by
    /// OpInfo::compute_kernel_class. A family present here overrides the scalar
    /// for its ops; one scalar per device does not survive contact with real
    /// kernels, whose achieved fraction of peak differs by several times.
    ///
    /// Kept out of the serialized form because serde_wasm_bindgen emits a Rust
    /// map as a JS `Map`, which callers cannot read as a plain object. The
    /// entry list below is the serialized view, matching how this module
    /// already exposes counted maps.
    #[serde(skip)]
    compute_utilization_by_kernel_class: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compute_utilization_entries: Vec<KernelUtilizationEntry>,
    ridge_point_ops_per_byte: f64,
    memory_bound_intensity: f64,
    compute_bound_intensity: f64,
    // Ratio of INT8 peak SIMD throughput to FP32 peak throughput on this target.
    // FP32 roofline thresholds = INT8 thresholds / fp32_compute_factor.
    // Used to give accurate bound classification for unquantized ops.
    fp32_compute_factor: f64,
    int8_speedup_estimate: f64,
    channel_alignment_multiple: usize,
    weight_packing_bandwidth_gbps: f64,
    chain_break_overhead_us_low: f64,
    chain_break_overhead_us_high: f64,
}

#[derive(Clone, Serialize)]
struct XnnpackChainInfo {
    id: i32,
    first_op: usize,
    last_op: usize,
    op_count: usize,
    macs: f64,
    mac_percent: f64,
    chain_class: String,
    target_hint: String,
}

#[derive(Clone, Serialize)]
struct PredictedPartitionBoundaryEdge {
    tensor_index: usize,
    tensor_name: String,
    tensor_shape: Vec<i32>,
    tensor_dtype: String,
    payload_bytes: Option<usize>,
    payload_status: String,
    payload_binding: String,
    payload_reason: String,
    producer_op_index: usize,
    producer_op_name: String,
    producer_domain: String,
    consumer_op_index: usize,
    consumer_op_name: String,
    consumer_domain: String,
    direction: String,
    materialization_status: String,
}

#[derive(Clone, Serialize)]
struct PredictedPartitionBoundaryInventory {
    schema: String,
    status: String,
    assignment_evidence_class: String,
    payload_evidence_class: String,
    edge_count: usize,
    unique_tensor_count: usize,
    assessed_payload_edge_count: usize,
    unassessed_payload_edge_count: usize,
    payload_coverage_status: String,
    payload_binding: String,
    assessed_edge_payload_bytes: usize,
    summed_edge_payload_bytes: Option<usize>,
    assessed_unique_tensor_payload_bytes: usize,
    unique_tensor_payload_bytes: Option<usize>,
    edges: Vec<PredictedPartitionBoundaryEdge>,
    interpretation_boundary: String,
}

struct TensorPayloadAssessment {
    bytes: Option<usize>,
    status: &'static str,
    binding: &'static str,
    reason: String,
}

#[derive(Clone, Serialize)]
struct TrafficItem {
    name: String,
    count: usize,
    estimated_bytes: f64,
    byte_percent: f64,
    macs: f64,
    mac_percent: f64,
}

#[derive(Clone, Serialize)]
struct PatternInfo {
    name: String,
    first_op: usize,
    last_op: usize,
    op_count: usize,
    summary: String,
}

#[derive(Clone, Serialize)]
struct Recommendation {
    priority: usize,
    tone: String,
    title: String,
    detail: String,
    op_index: i32,
}

#[derive(Clone, Serialize)]
struct QuantizationStatus {
    classification: String,
    label: String,
    summary: String,
    detail: String,
    quantized_tensor_percent: f64,
    quantized_compute_mac_percent: f64,
    compute_macs: f64,
    quantized_compute_macs: f64,
    quantized_compute_ops: usize,
    compute_ops: usize,
    quantize_ops: usize,
    dequantize_ops: usize,
    activation_quantize_ops: usize,
    activation_dequantize_ops: usize,
    activation_8bit_float_boundary_ops: usize,
    integer_requantization_ops: usize,
    constant_precision_conversion_ops: usize,
    float16_constant_expansion_ops: usize,
    int8_tensors: usize,
    uint8_tensors: usize,
    float16_tensors: usize,
    float_tensors: usize,
    input_dtypes: Vec<String>,
    output_dtypes: Vec<String>,
    op_state_counts: Vec<CountItem>,
    full_integer: bool,
}

struct QuantRiskSummary {
    scale_ratio: f64,
    scale_cv: f64,
    scale_mode: String,
    scale_ratio_meaningful: bool,
    zero_point_offset: i64,
    zero_point_risk: String,
    zero_point_status: String,
    label: String,
    detail: String,
}

struct WeightPackingSummary {
    weight_bytes: f64,
    overhead_us: f64,
    risk: String,
    detail: String,
}

struct ChannelAlignmentSummary {
    output_channels: i32,
    multiple: usize,
    status: String,
    detail: String,
    tail_overhead_percent: f64,
    tail_overhead_percent_min: f64,
    tail_overhead_percent_max: f64,
}

struct ChannelAlignmentContext<'a> {
    is_float_compute: bool,
    compute_precision_label: &'a str,
    source_tile_multiple: usize,
    source_tile_multiples: &'a [usize],
    kernel_evidence_class: &'a str,
}

#[derive(Clone, Serialize)]
struct KernelSourceCandidate {
    family: String,
    tile_mr: usize,
    tile_nr: usize,
    channel_tile: usize,
    primary_tile: usize,
    architecture_condition: String,
    compile_condition: String,
    runtime_condition: String,
    source_ref: String,
    source_file_sha256: String,
}

#[derive(Default)]
struct KernelCandidateSummary {
    candidate: String,
    tile_mr: usize,
    tile_nr: usize,
    channel_tile: usize,
    primary_tile: usize,
    source: String,
    evidence_class: String,
    selector_status: String,
    build_requirement: String,
    candidates: Vec<KernelSourceCandidate>,
    alignment_multiples: Vec<usize>,
}

// ── Insights ─────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Default)]
struct InsightSignal {
    label: String,
    value: String,
    tone: String,
}

#[derive(Clone, Serialize, Default)]
struct InsightScoreBreakdown {
    base: i32,
    quantization_coverage_penalty: i32,
    graph_runtime_pressure_penalty: i32,
    memory_posture_signal_points: i32,
    suspect_op_signal_points: i32,
    predicted_boundary_signal_points: i32,
    fallback_byte_signal_points: i32,
    copy_like_op_signal_points: i32,
    l1_watch_penalty: i32,
    exact_zero_kernel_signal_points: i32,
    quantization_risk_penalty: i32,
    dynamic_non_batch_input_penalty: i32,
    final_score: i32,
}

#[derive(Clone, Serialize, Default)]
struct Insights {
    score: i32,
    score_evidence_class: String,
    score_method: String,
    score_breakdown: InsightScoreBreakdown,
    tone: String,
    label: String,
    rationale: String,
    bound_compute: usize,
    bound_memory: usize,
    bound_mixed: usize,
    memory_ratio: f64,
    quant_ratio: f64,
    per_channel_ratio: f64,
    l1_watch_count: usize,
    max_l1_ratio: f64,
    chain_breaks: usize,
    effective_chain_breaks: usize,
    chain_summary: String,
    delegated_mac_ratio: f64,
    fallback_byte_ratio: f64,
    quant_risk_ops: usize,
    exact_zero_kernel_slices: usize,
    misaligned_ops: usize,
    packing_warn_ops: usize,
    suspect_total: usize,
    suspect_summary: String,
    dynamic_input_count: usize,
    dynamic_non_batch_input_count: usize,
    copy_like_op_count: usize,
    signals: Vec<InsightSignal>,
}

// ── Weight Analysis Structs ───────────────────────────────────────────────────

#[derive(Serialize)]
struct WeightHistogram {
    name: String,
    dtype: String,
    element_count: usize,
    shape: Vec<i32>,
    val_min: f64,
    val_max: f64,
    mean: f64,
    std_dev: f64,
    p05: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    p95: f64,
    sparsity: f64,
    entropy_bits: f64,
    range_utilization: f64,
    bin_min: f64,
    bin_max: f64,
    bins: Vec<u32>,
    low_norm_filters: Option<usize>,
    total_filters: Option<usize>,
    eff_rank: Option<usize>,
    diversity: Option<f64>,
    per_channel_scales: Vec<f32>,
    // Filter-level arrays for kernel grid visualization (empty for non-4D or 1×1 kernels)
    filter_norms: Vec<f32>,
    filter_sort_order: Vec<usize>,
    filter_sign_ratios: Vec<f32>,
    cluster_map: Vec<i32>,
    cluster_count: usize,
    // Dequantized float values for kernel pixel drawing (empty if >500k elements or 1×1)
    raw_values: Vec<f32>,
}

#[derive(Serialize)]
struct QuickLowNormStat {
    low_norm: usize,
    total: usize,
}

// ── Kernel Haar Decomposition Structs ────────────────────────────────────────

#[derive(Serialize, Clone)]
struct KernelHaarEnergy {
    ll: f32,              // DC energy fraction
    lh: f32,              // horizontal-edge energy fraction (top−bottom)
    hl: f32,              // vertical-edge energy fraction (left−right)
    hh: f32,              // diagonal/checker energy fraction
    center_surround: f32, // center pixel − ring mean (signed)
}

#[derive(Serialize)]
struct KernelHaarOpResult {
    op_index: usize,
    op_name: String,
    num_filters: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_channels: usize,
    mean_energy: KernelHaarEnergy,
    dominant: String,
    dominant_ll: usize,
    dominant_lh: usize,
    dominant_hl: usize,
    dominant_hh: usize,
    orientation_ratio: f32, // lh_energy / hl_energy
    edge_dc_ratio: f32,     // (lh+hl+hh) / ll
    energy_proxy: f32,      // mean |w| / sqrt(filter_volume)
    sparsity_proxy: f32,    // fraction of near-zero weights
}

#[derive(Serialize)]
struct KernelHaarSummary {
    conv_op_count: usize,
    edge_heavy_ops: usize,
    dc_heavy_ops: usize,
    lh_dominant_ops: usize,
    hl_dominant_ops: usize,
    hh_dominant_ops: usize,
    ll_dominant_ops: usize,
    orientation_bias: f32, // mean(lh_energy − hl_energy): >0 = H-edge bias
    global_dominant: String,
}

#[derive(Serialize)]
struct KernelHaarResult {
    ops: Vec<KernelHaarOpResult>,
    summary: KernelHaarSummary,
}

// Internal spatial influence map used during BFS — not serialized
#[derive(Clone)]
struct InflMap {
    map: Vec<f32>,
    h: usize,
    w: usize,
}

#[derive(Serialize)]
struct InfluenceMapResult {
    map: Vec<f32>,
    h: usize,
    w: usize,
    max_val: f32,
    chain_len: usize,
}

#[derive(Clone, Serialize)]
struct TensorPayloadIssue {
    tensor_index: Option<usize>,
    tensor_name: String,
    reason: String,
}

#[derive(Clone, Serialize)]
struct TensorLiveness {
    assessed: bool,
    status: String,
    #[serde(rename = "peak_bytes")]
    peak_bytes_value: Option<usize>,
    #[serde(skip)]
    peak_bytes: usize,
    #[serde(rename = "peak_at_op")]
    peak_at_op_value: Option<usize>,
    #[serde(skip)]
    peak_at_op: usize,
    #[serde(rename = "peak_at_op_name")]
    peak_at_op_name_value: Option<String>,
    #[serde(skip)]
    peak_at_op_name: String,
    assessed_tensor_count: usize,
    unassessed_tensor_count: usize,
    unassessed_tensors: Vec<TensorPayloadIssue>,
    method: String,
}

#[derive(Clone, Serialize)]
struct ArenaPlanTensorIssue {
    tensor_index: Option<usize>,
    tensor_name: String,
    reason: String,
}

#[derive(Clone, Serialize)]
struct ArenaPlanAllocation {
    tensor_index: usize,
    tensor_name: String,
    tensor_shape: Vec<i32>,
    tensor_dtype: String,
    arena: String,
    size_bytes: Option<usize>,
    offset_bytes: Option<usize>,
    first_node: usize,
    last_node: Option<usize>,
    shared_with_tensor_index: Option<usize>,
    allocation_status: String,
}

#[derive(Clone, Serialize)]
struct ArenaPlanAlias {
    tensor_index: usize,
    tensor_name: String,
    shared_with_tensor_index: usize,
    shared_with_tensor_name: String,
    op_index: usize,
    op_name: String,
    input_slot: usize,
    data_unmodified: bool,
    source: String,
}

#[derive(Clone, Serialize)]
struct TensorArenaPlanProjection {
    schema: String,
    status: String,
    evidence_class: String,
    source_commit: String,
    planner_source_url: String,
    arena_source_url: String,
    registration_source_basis: Vec<String>,
    tensor_alignment_bytes: usize,
    preserve_all_tensors: bool,
    non_persistent_arena_bytes: Option<usize>,
    persistent_arena_bytes: Option<usize>,
    combined_arena_bytes: Option<usize>,
    planned_tensor_count: usize,
    root_allocation_count: usize,
    non_persistent_allocation_count: usize,
    persistent_allocation_count: usize,
    shared_tensor_count: usize,
    dynamic_shape_signature_tensor_count: usize,
    source_comparator_tie_group_count: usize,
    source_comparator_tied_tensor_count: usize,
    source_comparator_fully_orders_projection: bool,
    deterministic_tie_break: String,
    unassessed_tensor_count: usize,
    calculation_issue_count: usize,
    allocations: Vec<ArenaPlanAllocation>,
    aliases: Vec<ArenaPlanAlias>,
    unassessed_tensors: Vec<ArenaPlanTensorIssue>,
    method: String,
    interpretation_boundary: String,
}

#[derive(Clone, Serialize, Default)]
struct ZeroKernelSliceDetail {
    tensor_index: usize,
    tensor_name: String,
    dtype: String,
    shape: Vec<i32>,
    channels: Vec<usize>,
    channel_count: usize,
    exact_zero_channels: Vec<usize>,
    exact_zero_channel_count: usize,
    scale_sample: Vec<f32>,
    zero_point_sample: Vec<i64>,
    bias_tensor_index: i32,
    bias_tensor_name: String,
    bias_dtype: String,
    bias_value_sample: Vec<f64>,
    bias_code_sample: Vec<i32>,
    bias_int32_utilization_sample: Vec<f64>,
    bias_nonzero_for_flagged_channels: bool,
    fused_activation: String,
    residual_path: String,
    next_consumers: Vec<String>,
    functional_status: String,
    consumer_ops: Vec<String>,
    consumer_mac_percent: f64,
}

#[derive(Clone, Serialize, Default)]
struct WeightIntegrityReport {
    weight_tensors_scanned: usize,
    sparse_constant_tensors_decoded: usize,
    sparse_constant_tensors_not_decoded: usize,
    sparse_logical_elements: usize,
    sparse_stored_elements: usize,
    sparse_implicit_zero_elements: usize,
    constant_value_coverage_status: String,
    quantized_constant_tensors_scanned: usize,
    elements_scanned: usize,
    eligible_kernel_tensors_scanned: usize,
    output_channels_evaluated: usize,
    nan_tensors: usize,
    inf_tensors: usize,
    all_zero_tensors: usize,
    zero_kernel_slice_tensors: usize, // tensors with >=1 all-zero decoded kernel output slice
    zero_kernel_slice_count: usize,
    exact_zero_kernel_slice_tensors: usize,
    exact_zero_kernel_slice_count: usize,
    max_abs_weight: f64,
    large_magnitude_tensors: usize, // |w| > 1e4 present
    mean_sparsity: f64,             // fraction of near-zero elements across scanned tensors
    high_sparsity_tensors: usize,   // >50% near-zero
    zero_kernel_slice_details: Vec<ZeroKernelSliceDetail>,
    low_grid_utilization_tensors: usize,
    saturated_quantized_tensors: usize,
    min_grid_utilization: f64,
    threshold_eligible_quantized_constant_tensors: usize,
    min_threshold_eligible_grid_utilization: Option<f64>,
    max_saturation_percent: f64,
    quant_grid_detail: String,
    status: String, // "ok" | "warn" | "risk"
    detail: String,
}

#[derive(Clone, Serialize)]
struct ArtifactSizeBreakdown {
    file_size: usize,
    constant_tensor_count: usize,
    sparse_constant_tensor_count: usize,
    physical_constant_buffer_count: usize,
    logical_constant_reference_bytes: usize,
    constant_bytes: usize,
    stored_scalar_elements: usize,
    unique_constant_bytes: usize,
    duplicate_constant_bytes: usize,
    metadata_bytes: usize,
    structure_overhead_bytes: usize, // file minus constants minus metadata
    float_constant_bytes: usize,
    theoretical_fp16_constant_bytes: usize,
    theoretical_int8_constant_bytes: usize,
    zero_constant_byte_ratio: f64,
    detail: String,
}

#[derive(Clone, Serialize)]
struct RuntimeCompat {
    min_runtime_version: String, // from model metadata if present, else ""
    derived_min_runtime_version: String, // from pinned TensorFlow op/version table where mapped
    effective_min_runtime_version: String, // max(declared, derived) only with complete builtin map coverage
    runtime_floor_status: String,
    runtime_floor_evidence_class: String,
    operator_code_count: usize,
    builtin_operator_code_count: usize,
    mapped_operator_code_count: usize,
    custom_operator_code_count: usize,
    runtime_version_basis: String,
    unmapped_versioned_ops: Vec<String>,
    max_op_version: i32, // highest operator_code.version in the graph
    version_driving_ops: Vec<String>, // op names at that max version
    detail: String,
}

#[derive(Clone, Serialize)]
struct ArtifactMetadataPresence {
    format: String,
    schema: String,
    status: String,
    has_signature_defs: bool,
    signature_count: usize,
    signature_keys: Vec<String>,
    has_model_metadata: bool,
    metadata_entries: Vec<String>,
    conversion_metadata_entry_count: usize,
    conversion_metadata_status: String,
    converter_tensorflow_version: String,
    converter_api_version: Option<u32>,
    converter_model_type: String,
    converter_optimization_mode_codes: Vec<i32>,
    converter_optimization_modes: Vec<String>,
    conversion_metadata_schema_source_commit: String,
    conversion_metadata_schema_source_file: String,
    conversion_metadata_schema_sha256: String,
    has_description: bool,
    description: String,
    documented_preprocessing: bool,
    preprocessing_contract_status: String,
    output_semantics_documented: bool,
    metadata_schema_identifier: String,
    metadata_min_parser_version: String,
    metadata_model_name: String,
    metadata_model_description: String,
    metadata_model_version: String,
    metadata_author: String,
    metadata_license: String,
    model_metadata_entry_count: usize,
    subgraph_metadata_count: usize,
    input_tensor_metadata_count: usize,
    output_tensor_metadata_count: usize,
    described_input_tensor_count: usize,
    described_output_tensor_count: usize,
    input_process_unit_count: usize,
    recognized_input_process_unit_count: usize,
    invalid_input_process_unit_count: usize,
    unrecognized_input_process_unit_count: usize,
    normalization_unit_count: usize,
    input_process_units: Vec<MetadataProcessUnit>,
    output_associated_file_count: usize,
    output_label_file_count: usize,
    verified_output_associated_file_count: usize,
    missing_output_associated_file_count: usize,
    verified_output_label_file_count: usize,
    missing_output_label_file_count: usize,
    invalid_output_label_file_count: usize,
    verified_output0_label_file_count: usize,
    payload_verified_file_count: usize,
    payload_invalid_file_count: usize,
    payload_unsupported_file_count: usize,
    label_cardinality_match_count: usize,
    label_cardinality_mismatch_count: usize,
    label_cardinality_ambiguous_count: usize,
    label_cardinality_unresolved_count: usize,
    output_associated_files: Vec<MetadataAssociatedFile>,
    associated_file_archive_status: String,
    associated_file_archive_detail: String,
    packed_associated_file_count: usize,
    packed_associated_files: Vec<PackedMetadataFile>,
    detail: String,
}

#[derive(Clone, Serialize)]
struct MovementAnalysis {
    status: String,
    #[serde(rename = "total_movement_bytes")]
    total_movement_bytes_value: Option<usize>,
    #[serde(skip)]
    total_movement_bytes: usize,
    assessed_movement_bytes: usize,
    movement_op_count: usize,
    #[serde(rename = "xnn_break_movement_bytes")]
    xnn_break_movement_bytes_value: Option<usize>,
    #[serde(skip)]
    xnn_break_movement_bytes: usize,
    assessed_xnn_break_movement_bytes: usize,
    assessed_output_tensor_count: usize,
    unassessed_output_tensor_count: usize,
    calculation_issue_count: usize,
    unassessed_tensors: Vec<TensorPayloadIssue>,
    movement_op_ratio: f64,
}

#[derive(Clone, Serialize)]
struct InputContract {
    schema: String,
    tensor_index: usize,
    name: String,
    shape: Vec<i32>,
    dtype: String,
    is_quantized: bool,
    expected_range_low: Option<f64>,
    expected_range_high: Option<f64>,
    range_note: String,
    tensor_numerical_contract_status: String,
    source_data_to_tensor_preprocessing_status: String,
    layout: Option<String>,
    layout_status: String,
    layout_evidence_class: String,
    layout_source_op_index: Option<usize>,
    layout_source_op_name: Option<String>,
    layout_reason: String,
    channel_axis: Option<usize>,
    channels: Option<i32>,
    risks: Vec<String>,
}

#[derive(Clone, Serialize)]
struct StaticRuntimeCalibration {
    static_estimate_ms: f64,
    cold_start_static_estimate_ms: f64,
    one_time_packing_ms: f64,
    boundary_setup_ms: f64,
    measured_ms: f64,
    correction_factor: Option<f64>,
    interpretation: String,
    confidence: String,
    method: String,
    assessed_op_count: usize,
    total_op_count: usize,
}

#[derive(Serialize)]
struct Analysis {
    format: String,
    filename: String,
    file_size: usize,
    model_sha256: String,
    target_profile: TargetProfile,
    version: u32,
    subgraphs: usize,
    tflite_subgraph_inventory: TfliteSubgraphInventory,
    tflite_subgraph_deep_analysis: TfliteSubgraphDeepAnalysis,
    operator_codes: usize,
    operator_count: usize,
    tensor_count: usize,
    tensors: Vec<TensorInfo>,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    input_tensor_indices: Vec<i32>,
    output_tensor_indices: Vec<i32>,
    histogram: Vec<CountItem>,
    tensor_types: Vec<CountItem>,
    quantized_tensors: usize,
    per_channel_tensors: usize,
    quantization_status: QuantizationStatus,
    accumulator_atlas: AccumulatorAtlasAnalysis,
    requantization_fidelity: RequantizationFidelityAnalysis,
    kernel_extremum_witness: KernelWitnessAnalysis,
    channel_vitality: ChannelVitalityAnalysis,
    rounding_equivalence: RoundingEquivalenceAnalysis,
    accumulator_reachability: AccumulatorReachabilityAnalysis,
    numerical_abi_propagation: NumericalAbiPropagationAnalysis,
    input_counterexample: InputCounterexampleAnalysis,
    preprocessing_realizability: PreprocessingRealizabilityAnalysis,
    quantization_lattice: QuantizationLatticeAnalysis,
    contract_migration: ContractMigrationAnalysis,
    residual_step_response: ResidualStepResponseAnalysis,
    residual_contract_distortion: ResidualContractDistortionAnalysis,
    total_macs: f64,
    total_ops: f64,
    delegated_macs: f64,
    fallback_macs: f64,
    delegated_mac_percent: f64,
    delegated_estimated_bytes: f64,
    fallback_estimated_bytes: f64,
    fallback_byte_percent: f64,
    fallback_traffic_by_op_family: Vec<TrafficItem>,
    estimated_int8_speedup: f64,
    estimated_int8_speedup_detail: String,
    suspects: Vec<CountItem>,
    stages: Vec<StageInfo>,
    ops: Vec<OpInfo>,
    xnnpack_assumption: String,
    xnnpack_selector_assessment_status: String,
    xnnpack_selector_evidence_schema: String,
    xnnpack_selector_evidence_access: String,
    xnnpack_chain_breaks: usize,
    xnnpack_effective_chain_breaks: usize,
    xnnpack_structural_chain_breaks: usize,
    xnnpack_zero_mac_chain_breaks: usize,
    xnnpack_longest_chain: usize,
    xnnpack_chains: Vec<XnnpackChainInfo>,
    predicted_partition_boundaries: PredictedPartitionBoundaryInventory,
    fc_ops: usize,
    fc_packing_warn_ops: usize,
    conv_weight_ops: usize,
    conv_packing_warn_ops: usize,
    patterns: Vec<PatternInfo>,
    block_inventory: BlockInventory,
    recommendations: Vec<Recommendation>,
    quant_holes: Vec<QuantHoleInfo>,
    quant_hole_count: usize,
    quant_hole_mac_impact: f64,
    stage_mermaid: String,
    roofline_csv: String,
    core_isolation_analysis: core_isolation::CoreIsolationAnalysis,
    core_isolation_csv: String,
    tensor_liveness: TensorLiveness,
    tensor_arena_plan: TensorArenaPlanProjection,
    movement_analysis: MovementAnalysis,
    artifact_byte_integrity: ArtifactByteIntegrityLedger,
    input_contracts: Vec<InputContract>,
    dynamic_shape_cost_contract: DynamicShapeCostContract,
    weight_integrity: WeightIntegrityReport,
    tflite_sparse_storage_contract: SparseStorageContract,
    runtime_compat: RuntimeCompat,
    size_breakdown: ArtifactSizeBreakdown,
    metadata_presence: ArtifactMetadataPresence,
    findings: Vec<Finding>,
    insights: Insights,
}

mod verified_flatbuffer;
#[cfg(test)]
use verified_flatbuffer::conversion_optimization_mode_name;
use verified_flatbuffer::{
    parse_conversion_metadata, BufferDataLocation, Fb, ParsedConversionMetadata,
};
// Topology annotations implemented in the Rust/WASM analysis core.

fn compute_topology_annotations(ops: &mut [OpInfo]) {
    // Build producer (tensor→producing op) and consumer (tensor→consuming ops) maps
    let mut producers: HashMap<usize, usize> = HashMap::new();
    let mut consumers_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter() {
        for &id in &op.outputs {
            if id >= 0 {
                producers.insert(id as usize, op.index);
            }
        }
        for &id in &op.inputs {
            if id >= 0 {
                consumers_map.entry(id as usize).or_default().push(op.index);
            }
        }
    }

    // Topological depth (TFLite ops are in execution order, so iterating by index is valid)
    let mut topo_depth: HashMap<usize, usize> = HashMap::new();
    for op in ops.iter() {
        let depth = op
            .inputs
            .iter()
            .filter(|&&id| id >= 0 && producers.contains_key(&(id as usize)))
            .map(|&id| {
                topo_depth
                    .get(producers.get(&(id as usize)).unwrap())
                    .copied()
                    .unwrap_or(0)
                    + 1
            })
            .max()
            .unwrap_or(0);
        topo_depth.insert(op.index, depth);
    }

    // Snapshot op names + inputs for pattern lookups without borrowing ops mutably
    let op_name_inputs: HashMap<usize, (String, Vec<i32>)> = ops
        .iter()
        .map(|op| (op.index, (op.name.clone(), op.inputs.clone())))
        .collect();
    let op_quant_hole: HashMap<usize, bool> =
        ops.iter().map(|op| (op.index, op.quant_hole)).collect();

    // Only activation-domain DEQUANTIZE outputs can seed a Q/DQ island.
    // FP16 or quantized constant expansion feeds weights into compute and is
    // not an activation precision island.
    let dequant_outputs: std::collections::HashSet<usize> = ops
        .iter()
        .filter(|op| op.name == "DEQUANTIZE" && op.quantization_state == "quant_boundary")
        .flat_map(|op| {
            op.outputs
                .iter()
                .filter(|&&id| id >= 0)
                .map(|&id| id as usize)
        })
        .collect();

    const MERGE_OPS: &[&str] = &["ADD", "CONCATENATION", "MUL", "MAXIMUM", "MINIMUM"];
    const QUANT_OPS: &[&str] = &["QUANTIZE", "DEQUANTIZE"];
    const RESIZE_OPS: &[&str] = &["RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"];
    const MATMUL_OPS: &[&str] = &["BATCH_MATMUL", "FULLY_CONNECTED"];

    // BFS ancestor search for RESIZE (fpn-merge detection)
    let has_resize_ancestor = |start_idx: usize, max_hops: usize| -> bool {
        let mut queue = vec![(start_idx, 0usize)];
        let mut seen = std::collections::HashSet::new();
        while let Some((idx, d)) = queue.pop() {
            if d > max_hops || !seen.insert(idx) {
                continue;
            }
            if let Some((name, inputs)) = op_name_inputs.get(&idx) {
                if RESIZE_OPS.contains(&name.as_str()) {
                    return true;
                }
                if d < max_hops {
                    for &id in inputs {
                        if id >= 0 {
                            if let Some(&pred) = producers.get(&(id as usize)) {
                                queue.push((pred, d + 1));
                            }
                        }
                    }
                }
            }
        }
        false
    };

    // Compute per-op role, depth, fan_out_max, and patterns
    let mut ann: HashMap<usize, (String, usize, usize, Vec<String>)> = HashMap::new();
    for op in ops.iter() {
        let fan_out_max = op
            .outputs
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| {
                consumers_map
                    .get(&(id as usize))
                    .map(|v| v.len())
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let role = if QUANT_OPS.contains(&op.name.as_str()) {
            "quant-boundary"
        } else if MERGE_OPS.contains(&op.name.as_str()) {
            let active_preds = op
                .inputs
                .iter()
                .filter(|&&id| id >= 0 && producers.contains_key(&(id as usize)))
                .count();
            if active_preds >= 2 {
                "branch-merge"
            } else {
                "through"
            }
        } else if fan_out_max > 1 {
            "branch-split"
        } else {
            "through"
        };
        let depth = topo_depth.get(&op.index).copied().unwrap_or(0);
        let mut pats: Vec<String> = op.patterns.clone(); // preserve existing stage patterns

        // Residual: ADD/MUL branch-merge with predecessor depth gap ≥ 3
        if (op.name == "ADD" || op.name == "MUL") && role == "branch-merge" {
            let depths: Vec<usize> = op
                .inputs
                .iter()
                .filter(|&&id| id >= 0 && producers.contains_key(&(id as usize)))
                .filter_map(|&id| {
                    topo_depth
                        .get(producers.get(&(id as usize)).unwrap())
                        .copied()
                })
                .collect();
            if depths.len() >= 2 {
                let max_d = depths.iter().max().copied().unwrap_or(0);
                let min_d = depths.iter().min().copied().unwrap_or(0);
                if max_d.saturating_sub(min_d) >= 3 && !pats.contains(&"residual".to_string()) {
                    pats.push("residual".to_string());
                }
            }
        }
        // FPN-merge: ADD/CONCAT merge where any branch passed through RESIZE
        if (op.name == "ADD" || op.name == "CONCATENATION") && role == "branch-merge" {
            let found_fpn = op
                .inputs
                .iter()
                .filter(|&&id| id >= 0 && producers.contains_key(&(id as usize)))
                .any(|&id| has_resize_ancestor(*producers.get(&(id as usize)).unwrap(), 5));
            if found_fpn && !pats.contains(&"fpn-merge".to_string()) {
                pats.push("fpn-merge".to_string());
            }
        }
        // QDQ island: compute op directly downstream of DEQUANTIZE (or already flagged quant_hole)
        let is_compute_op = is_mac_bearing_compute_op(&op.name);
        let feeds_from_dequant = op
            .inputs
            .iter()
            .any(|&id| id >= 0 && dequant_outputs.contains(&(id as usize)));
        if (op_quant_hole.get(&op.index).copied().unwrap_or(false)
            || (is_compute_op && feeds_from_dequant))
            && !pats.contains(&"qdq-island".to_string())
        {
            pats.push("qdq-island".to_string());
        }
        ann.insert(op.index, (role.to_string(), depth, fan_out_max, pats));
    }

    // Attention: SOFTMAX flanked by MATMUL/FC on both sides → propagate to neighbors
    let mut attention_ops: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for op in ops.iter() {
        if op.name != "SOFTMAX" {
            continue;
        }
        let pred_idxs: Vec<usize> = op
            .inputs
            .iter()
            .filter(|&&id| id >= 0 && producers.contains_key(&(id as usize)))
            .filter_map(|&id| producers.get(&(id as usize)).copied())
            .collect();
        let succ_idxs: Vec<usize> = op
            .outputs
            .iter()
            .filter(|&&id| id >= 0)
            .flat_map(|&id| {
                consumers_map
                    .get(&(id as usize))
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        let has_pred_matmul = pred_idxs.iter().any(|&i| {
            op_name_inputs
                .get(&i)
                .map(|(n, _)| MATMUL_OPS.contains(&n.as_str()))
                .unwrap_or(false)
        });
        let has_succ_matmul = succ_idxs.iter().any(|&i| {
            op_name_inputs
                .get(&i)
                .map(|(n, _)| MATMUL_OPS.contains(&n.as_str()))
                .unwrap_or(false)
        });
        if has_pred_matmul && has_succ_matmul {
            attention_ops.insert(op.index);
            for &i in &pred_idxs {
                if op_name_inputs
                    .get(&i)
                    .map(|(n, _)| MATMUL_OPS.contains(&n.as_str()))
                    .unwrap_or(false)
                {
                    attention_ops.insert(i);
                }
            }
            for &i in &succ_idxs {
                if op_name_inputs
                    .get(&i)
                    .map(|(n, _)| MATMUL_OPS.contains(&n.as_str()))
                    .unwrap_or(false)
                {
                    attention_ops.insert(i);
                }
            }
        }
    }
    for idx in attention_ops {
        if let Some(a) = ann.get_mut(&idx) {
            if !a.3.contains(&"attention".to_string()) {
                a.3.push("attention".to_string());
            }
        }
    }

    // Write back annotations to ops
    for op in ops.iter_mut() {
        if let Some((role, depth, fan_out_max, pats)) = ann.remove(&op.index) {
            op.topo_role = role;
            op.topo_depth = depth;
            op.topo_fan_out_max = fan_out_max;
            op.patterns = pats;
        }
    }
}

// ── Per-Op Bottleneck Estimates ───────────────────────────────────────────────

fn compute_bottleneck_estimates(ops: &mut [OpInfo], target: &TargetProfile) {
    let bandwidth_gbps = target.effective_memory_bandwidth_gbps.max(0.25);
    for op in ops.iter_mut() {
        // A family-specific measured utilization wins over the device scalar.
        let utilization = target
            .compute_utilization_by_kernel_class
            .get(&op.compute_kernel_class)
            .copied()
            .unwrap_or(target.compute_utilization_factor)
            .clamp(0.01, 1.0);
        let precision_peak_gops = if op.quantized_compute_path {
            target.effective_peak_gops
        } else {
            target.effective_peak_gops / target.fp32_compute_factor.max(0.1)
        };
        let peak_gops = (precision_peak_gops * utilization).max(f64::MIN_POSITIVE);
        let ops_count = op.ops.max(op.macs * 2.0);
        let bytes = op.estimated_bytes.max(0.0);
        let compute_us = if ops_count > 0.0 {
            ops_count / (peak_gops * 1e9) * 1e6
        } else {
            0.0
        };
        let memory_us = if bytes > 0.0 {
            bytes / (bandwidth_gbps * 1e9) * 1e6
        } else {
            0.0
        };
        let packing_us = op.weight_packing_overhead_us.max(0.0);
        let break_us = if op.xnnpack_chain_break {
            (op.chain_break_overhead_us_low + op.chain_break_overhead_us_high) / 2.0
        } else {
            0.0
        };
        let fallback_us = if op.xnnpack_chain_id < 0 || op.xnnpack_chain_break {
            (bytes * 0.5) / (bandwidth_gbps * 1e9) * 1e6
        } else {
            0.0
        };
        let bound_us = compute_us.max(memory_us);
        let total_us = bound_us + packing_us + break_us + fallback_us;
        let dominant = [
            ("compute", compute_us),
            ("memory", memory_us),
            ("fallback", fallback_us),
        ]
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(k, _)| *k)
        .unwrap_or("memory");
        op.bottleneck_compute_us = compute_us;
        op.bottleneck_memory_us = memory_us;
        op.bottleneck_packing_us = packing_us;
        op.bottleneck_break_us = break_us;
        op.bottleneck_fallback_us = fallback_us;
        op.bottleneck_total_us = total_us;
        op.bottleneck_dominant = dominant.to_string();
    }
}

// ── Insights Engine ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn compute_insights(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    inputs: &[TensorInfo],
    quantized_tensors: usize,
    per_channel_tensors: usize,
    xnnpack_chain_breaks: usize,
    xnnpack_effective_chain_breaks: usize,
    xnnpack_chains: &[XnnpackChainInfo],
    delegated_mac_percent: f64,
    fallback_byte_percent: f64,
    fallback_traffic: &[TrafficItem],
    suspects: &[CountItem],
    exact_zero_kernel_slices: usize,
) -> Insights {
    let total_ops_count = ops.len().max(1);
    let tensor_count = tensors.len();

    let bound_compute = ops
        .iter()
        .filter(|op| op.static_bound_guess == "compute-bound")
        .count();
    let bound_memory = ops
        .iter()
        .filter(|op| op.static_bound_guess == "memory-bound")
        .count();
    let bound_mixed = ops
        .iter()
        .filter(|op| op.static_bound_guess == "mixed")
        .count();
    let memory_ratio = bound_memory as f64 / total_ops_count as f64;

    let quant_ratio = if tensor_count > 0 {
        quantized_tensors as f64 / tensor_count as f64
    } else {
        0.0
    };
    let per_channel_ratio = if quantized_tensors > 0 {
        per_channel_tensors as f64 / quantized_tensors as f64
    } else {
        0.0
    };
    let quantized_model = quantized_tensors > 0;

    let l1_watch_count = ops
        .iter()
        .filter(|op| cache_working_set_is_watch(op.row_working_set_ratio))
        .count();
    let max_l1_ratio = ops
        .iter()
        .map(|op| op.row_working_set_ratio)
        .fold(0.0f64, f64::max);

    let chain_summary = {
        let shown: Vec<String> = xnnpack_chains
            .iter()
            .take(8)
            .map(|c| format!("C{}:{}ops/{:.1}%", c.id, c.op_count, c.mac_percent * 100.0))
            .collect();
        let rem = xnnpack_chains.len().saturating_sub(8);
        let mut s = shown.join(" / ");
        if rem > 0 {
            s.push_str(&format!(" / +{} more", rem));
        }
        s
    };

    const COPY_LIKE: &[&str] = &[
        "CONCATENATION",
        "GATHER",
        "PAD",
        "PACK",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SLICE",
        "SPLIT",
        "STRIDED_SLICE",
        "TRANSPOSE",
        "UNPACK",
    ];
    let copy_like_op_count = ops
        .iter()
        .filter(|op| COPY_LIKE.contains(&op.name.as_str()))
        .count();
    let dynamic_input_count = inputs
        .iter()
        .filter(|t| t.shape_signature.iter().any(|&d| d < 0))
        .count();
    let dynamic_non_batch_input_count = inputs
        .iter()
        .filter(|tensor| {
            tensor
                .shape_signature
                .iter()
                .enumerate()
                .any(|(axis, dimension)| axis != 0 && *dimension < 0)
        })
        .count();

    let quant_risk_ops = ops
        .iter()
        .filter(|op| op.quant_risk == "risk" || op.quant_risk == "warn")
        .count();
    let misaligned_ops = ops
        .iter()
        .filter(|op| op.channel_alignment_status == "misaligned")
        .count();
    let packing_warn_ops = ops
        .iter()
        .filter(|op| op.weight_packing_risk == "warn")
        .count();
    let suspect_total: usize = suspects.iter().map(|s| s.count).sum();
    let suspect_summary = suspects
        .iter()
        .take(4)
        .map(|s| format!("{}:{}", s.name, s.count))
        .collect::<Vec<_>>()
        .join(" / ");

    let quantization_coverage_penalty = if quantized_model {
        if quant_ratio < 0.25 {
            18
        } else if quant_ratio < 0.6 {
            10
        } else {
            0
        }
    } else {
        0
    };
    let memory_posture_signal_points = if memory_ratio > 0.45 {
        18
    } else if memory_ratio > 0.25 {
        10
    } else {
        0
    };
    let l1_watch_penalty = (l1_watch_count as i32 * 3).min(16);
    let suspect_op_signal_points = (suspect_total as i32 * 2).min(14);
    let predicted_boundary_signal_points = (xnnpack_effective_chain_breaks as i32 * 5).min(16);
    let fallback_byte_signal_points = if fallback_byte_percent >= 0.06 {
        12
    } else if fallback_byte_percent >= 0.02 {
        6
    } else {
        0
    };
    let risk_ops = ops.iter().filter(|op| op.quant_risk == "risk").count();
    let warn_ops = ops.iter().filter(|op| op.quant_risk == "warn").count();
    let op_quantization_risk_penalty = (risk_ops as i32 * 6 + warn_ops as i32 * 2).min(16);
    let exact_zero_kernel_signal_points = if exact_zero_kernel_slices >= 8 {
        16
    } else if exact_zero_kernel_slices > 0 {
        10
    } else {
        0
    };
    let quantization_risk_penalty =
        op_quantization_risk_penalty.max(exact_zero_kernel_signal_points);
    let copy_like_op_signal_points = if copy_like_op_count > 8 {
        8
    } else if copy_like_op_count > 3 {
        4
    } else {
        0
    };
    let graph_runtime_pressure_penalty = [
        memory_posture_signal_points,
        suspect_op_signal_points,
        predicted_boundary_signal_points,
        fallback_byte_signal_points,
        copy_like_op_signal_points,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let dynamic_non_batch_input_penalty = if dynamic_non_batch_input_count > 0 {
        6
    } else {
        0
    };
    let total_penalty = quantization_coverage_penalty
        + graph_runtime_pressure_penalty
        + l1_watch_penalty
        + quantization_risk_penalty
        + dynamic_non_batch_input_penalty;
    let score = (100 - total_penalty).clamp(0, 100);
    let score_breakdown = InsightScoreBreakdown {
        base: 100,
        quantization_coverage_penalty,
        graph_runtime_pressure_penalty,
        memory_posture_signal_points,
        suspect_op_signal_points,
        predicted_boundary_signal_points,
        fallback_byte_signal_points,
        copy_like_op_signal_points,
        l1_watch_penalty,
        exact_zero_kernel_signal_points,
        quantization_risk_penalty,
        dynamic_non_batch_input_penalty,
        final_score: score,
    };

    let tone = if score >= 80 {
        "good"
    } else if score >= 60 {
        "warn"
    } else {
        "risk"
    }
    .to_string();
    let label = (if quantization_risk_penalty >= 12 {
        "Quantization contract review is the dominant static signal"
    } else if score >= 80 {
        "Strong static preflight posture"
    } else if score >= 60 {
        "No high-severity structural signal in the static heuristic index"
    } else {
        "Static preflight suggests graph/runtime cleanup"
    })
    .to_string();
    let rationale = (if quantization_risk_penalty >= 12 {
        "Artifact-derived quantization predicates materially reduce the heuristic index. Review the exact channel and arithmetic ledgers before graph/runtime optimization."
    } else if score >= 80 {
        "The graph shape looks friendly to local execution; validate with real delegate coverage and Invoke latency."
    } else if score >= 60 {
        "No high-severity structural defect was detected. Integration contracts remain incomplete, and runtime behavior is unmeasured."
    } else {
        "The static heuristic index is below 60. Use the quantified findings and evidence classes below; this score is not deployment readiness."
    }).to_string();

    // Signals
    let mut signals: Vec<InsightSignal> = Vec::new();
    if let Some(top) = ops
        .iter()
        .filter(|op| op.macs > 0.0)
        .max_by(|a, b| a.macs.total_cmp(&b.macs))
    {
        signals.push(InsightSignal {
            label: "Top compute op".to_string(),
            value: format!("#{:03} {} / {:.0} MACs", top.index, top.name, top.macs),
            tone: "compute-bound".to_string(),
        });
    }
    if let Some(top) = ops
        .iter()
        .filter(|op| op.estimated_bytes > 0.0)
        .max_by(|a, b| a.estimated_bytes.total_cmp(&b.estimated_bytes))
    {
        signals.push(InsightSignal {
            label: "Largest tensor traffic estimate".to_string(),
            value: format!(
                "#{:03} {} / {}",
                top.index,
                top.name,
                fmt_mb(top.estimated_bytes as usize)
            ),
            tone: top.static_bound_guess.clone(),
        });
    }
    if let Some(top) = ops
        .iter()
        .filter(|op| op.row_working_set_bytes > 0.0)
        .max_by(|a, b| a.row_working_set_bytes.total_cmp(&b.row_working_set_bytes))
    {
        let l1_tone = if cache_working_set_is_watch(top.row_working_set_ratio) {
            "warn"
        } else {
            "good"
        };
        signals.push(InsightSignal {
            label: "Largest naive L1 row working set".to_string(),
            value: format!(
                "#{:03} {} / {}",
                top.index,
                top.name,
                fmt_mb(top.row_working_set_bytes as usize)
            ),
            tone: l1_tone.to_string(),
        });
    }
    if xnnpack_chain_breaks > 0 {
        let first_break = ops
            .iter()
            .filter(|op| op.xnnpack_chain_break)
            .max_by(|a, b| {
                (a.chain_break_impact_mac_percent + a.fallback_byte_percent)
                    .total_cmp(&(b.chain_break_impact_mac_percent + b.fallback_byte_percent))
            });
        let val = if let Some(brk) = first_break {
            format!(
                "#{:03} {} / {} / {:.0}-{:.0} us",
                brk.index,
                brk.name,
                if brk.xnnpack_break_class.is_empty() {
                    "break"
                } else {
                    &brk.xnnpack_break_class
                },
                brk.chain_break_overhead_us_low,
                brk.chain_break_overhead_us_high
            )
        } else {
            format!("{} break candidates", xnnpack_chain_breaks)
        };
        signals.push(InsightSignal {
            label: "XNNPACK predicted partition-break impact".to_string(),
            value: val,
            tone: if xnnpack_effective_chain_breaks > 0 {
                "warn"
            } else {
                "neutral"
            }
            .to_string(),
        });
    }
    if let Some(top) = fallback_traffic.first() {
        signals.push(InsightSignal {
            label: "Fallback traffic family".to_string(),
            value: format!(
                "{} / {} / {:.0}%",
                top.name,
                fmt_mb(top.estimated_bytes as usize),
                top.byte_percent * 100.0
            ),
            tone: if top.byte_percent >= 0.03 {
                "warn"
            } else {
                "neutral"
            }
            .to_string(),
        });
    }
    let top_qrisk = ops
        .iter()
        .filter(|op| op.quant_risk != "none")
        .max_by(|a, b| {
            let sa = (if a.quant_scale_ratio_meaningful {
                a.quant_scale_ratio.max(1.0).log10() + a.quant_scale_cv * 1.5
            } else {
                0.0
            }) + (if a.quant_zero_point_status == "out-of-range" {
                a.quant_zero_point_offset.unsigned_abs() as f64 / 32.0
            } else {
                0.0
            });
            let sb = (if b.quant_scale_ratio_meaningful {
                b.quant_scale_ratio.max(1.0).log10() + b.quant_scale_cv * 1.5
            } else {
                0.0
            }) + (if b.quant_zero_point_status == "out-of-range" {
                b.quant_zero_point_offset.unsigned_abs() as f64 / 32.0
            } else {
                0.0
            });
            sa.total_cmp(&sb)
        });
    if let Some(top) = top_qrisk {
        if top.quant_scale_ratio >= 1000.0 || top.quant_zero_point_status == "out-of-range" {
            let mode_text = if top.quant_scale_ratio_meaningful {
                format!(
                    "r {:.2e} / cv {:.2}",
                    top.quant_scale_ratio, top.quant_scale_cv
                )
            } else {
                format!("{} / scale N/A", top.quant_scale_mode)
            };
            signals.push(InsightSignal {
                label: "Quantization numerical contract".to_string(),
                value: format!(
                    "#{:03} {} / {} / {}",
                    top.index, top.name, top.quant_risk, mode_text
                ),
                tone: top.quant_risk.clone(),
            });
        }
    }
    if let Some(top) = ops
        .iter()
        .filter(|op| op.channel_alignment_status == "misaligned")
        .max_by(|a, b| {
            a.channel_tail_overhead_percent
                .total_cmp(&b.channel_tail_overhead_percent)
        })
    {
        signals.push(InsightSignal {
            label: "Channel alignment tail".to_string(),
            value: format!(
                "#{:03} {} / {}",
                top.index, top.name, top.channel_alignment_detail
            ),
            tone: "warn".to_string(),
        });
    }
    if let Some(top) = ops
        .iter()
        .filter(|op| op.weight_packing_risk == "warn")
        .max_by(|a, b| {
            a.weight_packing_overhead_us
                .total_cmp(&b.weight_packing_overhead_us)
        })
    {
        signals.push(InsightSignal {
            label: "Weight packing estimate".to_string(),
            value: format!(
                "#{:03} {} / {}",
                top.index, top.name, top.weight_packing_detail
            ),
            tone: "warn".to_string(),
        });
    }
    if !suspect_summary.is_empty() {
        signals.push(InsightSignal {
            label: "Delegate-risk families".to_string(),
            value: suspect_summary.clone(),
            tone: "warn".to_string(),
        });
    }
    if dynamic_input_count > 0 {
        let names: Vec<String> = inputs
            .iter()
            .filter(|t| t.shape_signature.iter().any(|&d| d < 0))
            .map(|t| {
                format!(
                    "{}[{}]",
                    t.name,
                    t.shape_signature
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("×")
                )
            })
            .collect();
        signals.push(InsightSignal {
            label: "Dynamic input dimensions".to_string(),
            value: names.join(" / "),
            tone: if dynamic_non_batch_input_count > 0 {
                "warn"
            } else {
                "neutral"
            }
            .to_string(),
        });
    }

    Insights {
        score,
        score_evidence_class: "HEURISTIC".to_string(),
        score_method: "max(0, 100 - quantization_coverage - max(memory_posture_signal, suspect_op_signal, predicted_boundary_signal, fallback_byte_signal, copy_like_op_signal) - l1_watch - max(op_quantization_risk, exact_zero_kernel_signal) - dynamic_non_batch_input); exact-zero stored kernel slices are scheme-independent artifact integrity signals, correlated graph/runtime signals contribute only their maximum once, and every term is emitted in score_breakdown".to_string(),
        score_breakdown,
        tone,
        label,
        rationale,
        bound_compute,
        bound_memory,
        bound_mixed,
        memory_ratio,
        quant_ratio,
        per_channel_ratio,
        l1_watch_count,
        max_l1_ratio,
        chain_breaks: xnnpack_chain_breaks,
        effective_chain_breaks: xnnpack_effective_chain_breaks,
        chain_summary,
        delegated_mac_ratio: delegated_mac_percent,
        fallback_byte_ratio: fallback_byte_percent,
        quant_risk_ops,
        exact_zero_kernel_slices,
        misaligned_ops,
        packing_warn_ops,
        suspect_total,
        suspect_summary,
        dynamic_input_count,
        dynamic_non_batch_input_count,
        copy_like_op_count,
        signals,
    }
}

// Weight histograms implemented in the Rust/WASM analysis core.

fn extract_tensor_buffer<'a>(model_bytes: &'a [u8], tensor: &TensorInfo) -> Option<Cow<'a, [u8]>> {
    if !tensor.constant_buffer || (!tensor.sparse_storage && tensor.buffer_data_length == 0) {
        return None;
    }
    let end = tensor
        .buffer_data_offset
        .checked_add(tensor.buffer_data_length)?;
    let raw = model_bytes.get(tensor.buffer_data_offset..end)?;
    tensor
        .sparse_encoding
        .as_ref()
        .map(|encoding| encoding.densify(raw).map(Cow::Owned))
        .unwrap_or_else(|| Some(Cow::Borrowed(raw)))
}

fn compute_weight_histogram_for_tensor(
    model_bytes: &[u8],
    tensor: &TensorInfo,
) -> Option<WeightHistogram> {
    let raw = extract_tensor_buffer(model_bytes, tensor)?;
    let n = raw.len();
    if n == 0 {
        return None;
    }

    // Parse values into f64 working array
    let values: Vec<f64> = match tensor.dtype.as_str() {
        "INT8" => raw.iter().map(|&b| (b as i8) as f64).collect(),
        "UINT8" => raw.iter().map(|&b| b as f64).collect(),
        "INT32" => {
            if n % 4 != 0 {
                return None;
            }
            (0..n / 4)
                .map(|i| {
                    i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
                        as f64
                })
                .collect()
        }
        "FLOAT32" => {
            if n % 4 != 0 {
                return None;
            }
            (0..n / 4)
                .map(|i| {
                    let bits = u32::from_le_bytes([
                        raw[i * 4],
                        raw[i * 4 + 1],
                        raw[i * 4 + 2],
                        raw[i * 4 + 3],
                    ]);
                    f32::from_bits(bits) as f64
                })
                .collect()
        }
        "FLOAT16" => {
            if n % 2 != 0 {
                return None;
            }
            (0..n / 2)
                .map(|i| {
                    let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                    f16_to_f32(bits) as f64
                })
                .collect()
        }
        _ => return None,
    };
    let elem_count = values.len();
    if elem_count == 0 {
        return None;
    }

    // Single-pass statistics (Welford's online algorithm)
    let mut min_v = values[0];
    let mut max_v = values[0];
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut zeros = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        if v == 0.0 {
            zeros += 1;
        }
        let delta = v - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (v - mean);
    }
    let std_dev = (m2 / elem_count as f64).max(0.0).sqrt();

    let exact8 = tensor.dtype == "INT8" || tensor.dtype == "UINT8";
    let bins_count = if exact8 { 256usize } else { 64usize };
    let (bin_min, bin_max) = if exact8 {
        if tensor.dtype == "INT8" {
            (-128.0f64, 127.0f64)
        } else {
            (0.0f64, 255.0f64)
        }
    } else {
        (min_v, max_v)
    };
    let range = (bin_max - bin_min).max(1e-30);
    let mut bins = vec![0u32; bins_count];
    if exact8 {
        let off: i64 = if tensor.dtype == "INT8" { 128 } else { 0 };
        for &v in &values {
            let idx = ((v as i64) + off).clamp(0, bins_count as i64 - 1) as usize;
            bins[idx] += 1;
        }
    } else {
        for &v in &values {
            let idx = ((v - bin_min) / range * bins_count as f64)
                .floor()
                .clamp(0.0, (bins_count - 1) as f64) as usize;
            bins[idx] += 1;
        }
    }

    let percentile = |frac: f64| -> f64 {
        let target = (frac * elem_count as f64) as usize;
        let mut cumul = 0usize;
        for (i, &c) in bins.iter().enumerate() {
            cumul += c as usize;
            if cumul >= target {
                return bin_min + (i as f64 + 0.5) * range / bins_count as f64;
            }
        }
        bin_max
    };

    let mut entropy = 0.0f64;
    for &c in &bins {
        if c > 0 {
            let p = c as f64 / elem_count as f64;
            entropy -= p * p.log2();
        }
    }
    let range_utilization = if exact8 { (max_v - min_v) / 255.0 } else { 0.0 };

    // Quick filter stats (low-norm filters + effective rank) for conv weight tensors.
    // DEPTHWISE shape=[1,kH,kW,k_in]: each in-channel is a separate kH×kW filter.
    // CONV_2D  shape=[k_out,kH,kW,k_in]: each out-channel is a contiguous filter.
    let (low_norm_filters, total_filters, eff_rank, diversity) = if tensor.shape.len() == 4 {
        let (k_out, k_h, k_w, k_in) = (
            tensor.shape[0] as usize,
            tensor.shape[1] as usize,
            tensor.shape[2] as usize,
            tensor.shape[3] as usize,
        );
        let is_dw = k_out == 1 && k_in > 1;
        let n_filters = if is_dw { k_in } else { k_out };
        let elem_total = k_out * k_h * k_w * k_in;
        if elem_total == 0 || n_filters == 0 || values.len() != elem_total {
            (None, None, None, None)
        } else {
            let scales_ch = if is_dw { k_in } else { k_out };
            let scales_ref: Option<&[f32]> = if tensor.scale_sample.len() == scales_ch {
                Some(&tensor.scale_sample)
            } else {
                None
            };
            let mut norms = vec![0.0f64; n_filters];
            if is_dw {
                let hw = k_h * k_w;
                for ic in 0..k_in {
                    let l2 = (0..hw)
                        .map(|p| {
                            let v = values[p * k_in + ic];
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt();
                    norms[ic] = if let Some(s) = scales_ref {
                        s[ic] as f64 * l2
                    } else {
                        l2
                    };
                }
            } else {
                let filter_size = k_h * k_w * k_in;
                for oc in 0..k_out {
                    let base = oc * filter_size;
                    let l2 = values[base..base + filter_size]
                        .iter()
                        .map(|&v| v * v)
                        .sum::<f64>()
                        .sqrt();
                    norms[oc] = if let Some(s) = scales_ref {
                        s[oc] as f64 * l2
                    } else {
                        l2
                    };
                }
            }
            let max_norm = norms.iter().cloned().fold(0.0f64, f64::max);
            let dead = norms.iter().filter(|&&n| n < 0.02 * max_norm).count();
            // Effective rank: min filters covering 90% energy
            let mut sorted_e: Vec<f64> = norms.iter().map(|&n| n * n).collect();
            sorted_e.sort_by(|a, b| b.total_cmp(a));
            let total_e: f64 = sorted_e.iter().sum();
            let mut cumul_e = 0.0f64;
            let mut eff = n_filters;
            for (k, &e) in sorted_e.iter().enumerate() {
                cumul_e += e;
                if cumul_e >= 0.9 * total_e {
                    eff = k + 1;
                    break;
                }
            }
            // Cosine diversity (sampled, max 1500 pairs)
            let div = if n_filters >= 2 {
                let n_pairs = n_filters * (n_filters - 1) / 2;
                let step = if n_pairs <= 1500 {
                    1
                } else {
                    (n_pairs / 1500).max(1)
                };
                let mut sim_sum = 0.0f64;
                let mut pair_count = 0usize;
                let mut pair_idx = 0usize;
                if is_dw {
                    let hw = k_h * k_w;
                    'outer_dw: for i in 0..k_in {
                        for j in (i + 1)..k_in {
                            if pair_idx.is_multiple_of(step) {
                                let (mut dot, mut ni2, mut nj2) = (0.0f64, 0.0f64, 0.0f64);
                                for p in 0..hw {
                                    let vi = values[p * k_in + i];
                                    let vj = values[p * k_in + j];
                                    dot += vi * vj;
                                    ni2 += vi * vi;
                                    nj2 += vj * vj;
                                }
                                sim_sum += (dot / (ni2 * nj2).sqrt().max(1e-30)).abs();
                                pair_count += 1;
                                if pair_count >= 1500 {
                                    break 'outer_dw;
                                }
                            }
                            pair_idx += 1;
                        }
                    }
                } else {
                    let filter_size = k_h * k_w * k_in;
                    'outer: for i in 0..k_out {
                        for j in (i + 1)..k_out {
                            if pair_idx.is_multiple_of(step) {
                                let bi = i * filter_size;
                                let bj = j * filter_size;
                                let (mut dot, mut ni2, mut nj2) = (0.0f64, 0.0f64, 0.0f64);
                                for k in 0..filter_size {
                                    dot += values[bi + k] * values[bj + k];
                                    ni2 += values[bi + k] * values[bi + k];
                                    nj2 += values[bj + k] * values[bj + k];
                                }
                                sim_sum += (dot / (ni2 * nj2).sqrt().max(1e-30)).abs();
                                pair_count += 1;
                                if pair_count >= 1500 {
                                    break 'outer;
                                }
                            }
                            pair_idx += 1;
                        }
                    }
                }
                if pair_count > 0 {
                    Some(1.0 - sim_sum / pair_count as f64)
                } else {
                    None
                }
            } else {
                None
            };
            (Some(dead), Some(n_filters), Some(eff), div)
        }
    } else {
        (None, None, None, None)
    };

    let per_channel_scales: Vec<f32> = tensor.scale_sample.clone();

    // ── Filter-level arrays for kernel grid visualization ─────────────────────
    let (
        filter_norms,
        filter_sort_order,
        filter_sign_ratios,
        cluster_map,
        cluster_count,
        raw_values,
    ) = if tensor.shape.len() == 4 {
        let (k_out, k_h, k_w, k_in) = (
            tensor.shape[0] as usize,
            tensor.shape[1] as usize,
            tensor.shape[2] as usize,
            tensor.shape[3] as usize,
        );
        // Depthwise: shape [1, kH, kW, inCh] — one spatial filter per input channel.
        // JS renders numFilters = inCh, so we must generate inCh entries, not 1.
        let is_dw = k_out == 1 && k_in > 1 && k_h * k_w > 1;
        let n_filt = if is_dw { k_in } else { k_out };
        let patch_size = if is_dw { k_h * k_w } else { k_h * k_w * k_in };

        // val_for: returns the weight value at spatial position p within filter f
        let val_for = |f: usize, p: usize| -> f64 {
            if is_dw {
                let ky = p / k_w;
                let kx = p % k_w;
                values[ky * k_w * k_in + kx * k_in + f]
            } else {
                values[f * patch_size + p]
            }
        };

        if patch_size > 0 && n_filt > 0 && values.len() == k_out * k_h * k_w * k_in {
            // Per-filter L2 norms
            let fn_: Vec<f32> = (0..n_filt)
                .map(|f| {
                    (0..patch_size)
                        .map(|p| {
                            let v = val_for(f, p);
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt() as f32
                })
                .collect();

            // Descending sort order by norm
            let mut order: Vec<usize> = (0..n_filt).collect();
            order.sort_by(|&a, &b| {
                fn_[b]
                    .partial_cmp(&fn_[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Positive weight ratio per filter
            let sr: Vec<f32> = (0..n_filt)
                .map(|f| {
                    (0..patch_size).filter(|&p| val_for(f, p) > 0.0).count() as f32
                        / patch_size as f32
                })
                .collect();

            // Union-find clustering (cosine similarity ≥ 0.9, sampled)
            let n_pairs = n_filt * (n_filt - 1) / 2;
            let step = if n_pairs <= 1000 {
                1usize
            } else {
                (n_pairs / 1000).max(1)
            };
            let mut parent: Vec<i32> = (0..n_filt as i32).collect();
            let mut pair_idx = 0usize;
            for i in 0..n_filt {
                for j in (i + 1)..n_filt {
                    if pair_idx.is_multiple_of(step) {
                        let (mut dot, mut ni2, mut nj2) = (0.0f64, 0.0f64, 0.0f64);
                        for p in 0..patch_size {
                            let vi = val_for(i, p);
                            let vj = val_for(j, p);
                            dot += vi * vj;
                            ni2 += vi * vi;
                            nj2 += vj * vj;
                        }
                        let denom = (ni2 * nj2).sqrt();
                        if denom > 1e-9 && (dot / denom).abs() >= 0.9 {
                            let ri = uf_find(&mut parent, i as i32);
                            let rj = uf_find(&mut parent, j as i32);
                            if ri != rj {
                                parent[ri as usize] = rj;
                            }
                        }
                    }
                    pair_idx += 1;
                }
            }
            // Assign cluster IDs (singleton → -1)
            let roots: Vec<i32> = (0..n_filt)
                .map(|i| uf_find(&mut parent, i as i32))
                .collect();
            let mut root_sizes: std::collections::HashMap<i32, usize> =
                std::collections::HashMap::new();
            for &r in &roots {
                *root_sizes.entry(r).or_insert(0) += 1;
            }
            let mut cluster_ids: std::collections::HashMap<i32, i32> =
                std::collections::HashMap::new();
            let mut next_cid = 0i32;
            let cm: Vec<i32> = roots
                .iter()
                .map(|&r| {
                    if *root_sizes.get(&r).unwrap_or(&0) >= 2 {
                        *cluster_ids.entry(r).or_insert_with(|| {
                            let id = next_cid;
                            next_cid += 1;
                            id
                        })
                    } else {
                        -1
                    }
                })
                .collect();
            let cc = next_cid as usize;

            // Raw values for pixel drawing — keep original flat layout (JS drawKernelCanvas
            // handles both CONV and DEPTHWISE addressing based on outCh == 1)
            let rv: Vec<f32> = if k_h * k_w > 1 && values.len() <= 500_000 {
                values.iter().map(|&v| v as f32).collect()
            } else {
                Vec::new()
            };

            (fn_, order, sr, cm, cc, rv)
        } else {
            (vec![], vec![], vec![], vec![], 0, vec![])
        }
    } else {
        (vec![], vec![], vec![], vec![], 0, vec![])
    };

    Some(WeightHistogram {
        name: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        element_count: elem_count,
        shape: tensor.shape.clone(),
        val_min: min_v,
        val_max: max_v,
        mean,
        std_dev,
        p05: percentile(0.05),
        p25: percentile(0.25),
        p50: percentile(0.50),
        p75: percentile(0.75),
        p95: percentile(0.95),
        sparsity: zeros as f64 / elem_count as f64,
        entropy_bits: entropy,
        range_utilization,
        bin_min,
        bin_max,
        bins,
        low_norm_filters,
        total_filters,
        eff_rank,
        diversity,
        per_channel_scales,
        filter_norms,
        filter_sort_order,
        filter_sign_ratios,
        cluster_map,
        cluster_count,
        raw_values,
    })
}

fn compute_quick_low_norm_stat_for_tensor(
    model_bytes: &[u8],
    tensor: &TensorInfo,
    is_depthwise: bool,
) -> Option<QuickLowNormStat> {
    let raw = extract_tensor_buffer(model_bytes, tensor)?;
    if tensor.shape.len() < 2 {
        return None;
    }

    let quantized_values: Option<Vec<i32>> = match tensor.dtype.as_str() {
        "INT8" => Some(raw.iter().map(|&value| i32::from(value as i8)).collect()),
        "UINT8" => Some(raw.iter().map(|&value| i32::from(value)).collect()),
        _ => None,
    };
    let float_values: Option<Vec<f64>> = match tensor.dtype.as_str() {
        "FLOAT32" => {
            if !raw.len().is_multiple_of(4) {
                return None;
            }
            let n = raw.len() / 4;
            Some(
                (0..n)
                    .map(|i| {
                        f32::from_bits(u32::from_le_bytes([
                            raw[i * 4],
                            raw[i * 4 + 1],
                            raw[i * 4 + 2],
                            raw[i * 4 + 3],
                        ])) as f64
                    })
                    .collect(),
            )
        }
        "INT8" | "UINT8" => None,
        _ => return None,
    };

    if tensor.shape.iter().any(|dimension| *dimension <= 0) {
        return None;
    }
    let shape = tensor
        .shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).ok())
        .collect::<Option<Vec<_>>>()?;
    let (out_ch, filter_size, filter_axis) = if is_depthwise {
        if shape.len() != 4 || shape[0] != 1 {
            return None;
        }
        let in_ch = *shape.last()?;
        (in_ch, shape[1].checked_mul(shape[2])?, 3usize)
    } else {
        let out_channels = *shape.first()?;
        let element_count = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))?;
        let size = element_count.checked_div(out_channels)?;
        if size == 0 {
            return None;
        }
        (out_channels, size, 0usize)
    };
    if out_ch == 0 || filter_size == 0 {
        return None;
    }

    let element_count = out_ch.checked_mul(filter_size)?;
    if float_values
        .as_ref()
        .is_some_and(|values| values.len() != element_count)
        || quantized_values
            .as_ref()
            .is_some_and(|values| values.len() != element_count)
    {
        return None;
    }
    let quant_params = if quantized_values.is_some() {
        if tensor.scale_sample.is_empty()
            || tensor
                .scale_sample
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return None;
        }
        if tensor.scale_sample.len() > 1
            && (tensor.scale_sample.len() != out_ch
                || usize::try_from(tensor.quantized_dimension).ok() != Some(filter_axis))
        {
            return None;
        }
        if !matches!(tensor.zero_point_sample.len(), 0 | 1)
            && tensor.zero_point_sample.len() != out_ch
        {
            return None;
        }
        Some((&tensor.scale_sample[..], &tensor.zero_point_sample[..]))
    } else {
        None
    };
    let value_at = |filter: usize, element: usize| -> Option<f64> {
        let index = if is_depthwise {
            element.checked_mul(out_ch)?.checked_add(filter)?
        } else {
            filter.checked_mul(filter_size)?.checked_add(element)?
        };
        if let Some(values) = &float_values {
            return values.get(index).copied();
        }
        let values = quantized_values.as_ref()?;
        let (scales, zero_points) = quant_params?;
        let scale = if scales.len() == 1 {
            f64::from(scales[0])
        } else {
            f64::from(*scales.get(filter)?)
        };
        let zero_point = match zero_points.len() {
            0 => 0,
            1 => zero_points[0],
            _ => *zero_points.get(filter)?,
        };
        Some((i64::from(*values.get(index)?) - zero_point) as f64 * scale)
    };
    let mut max_norm = 0.0f64;
    let mut norms = Vec::with_capacity(out_ch);
    for filter in 0..out_ch {
        let mut sum_squares = 0.0f64;
        for element in 0..filter_size {
            let value = value_at(filter, element)?;
            sum_squares += value * value;
        }
        let norm = sum_squares.sqrt();
        max_norm = max_norm.max(norm);
        norms.push(norm);
    }
    let thr = 0.02 * max_norm;
    let low_norm = norms.iter().filter(|&&n| n < thr).count();
    Some(QuickLowNormStat {
        low_norm,
        total: out_ch,
    })
}

// ── Union-find helper (used by filter clustering + influence BFS) ─────────────

fn unassessed_tensor_liveness(
    assessed_tensor_count: usize,
    issues: Vec<TensorPayloadIssue>,
) -> TensorLiveness {
    TensorLiveness {
        assessed: false,
        status: "not_assessed".to_string(),
        peak_bytes_value: None,
        peak_bytes: 0,
        peak_at_op_value: None,
        peak_at_op: 0,
        peak_at_op_name_value: None,
        peak_at_op_name: String::new(),
        assessed_tensor_count,
        unassessed_tensor_count: issues.len(),
        unassessed_tensors: issues,
        method: "Exact declared-shape payload sweep from producer through last consumer; no unknown dimension, dtype, or overflow substitution.".to_string(),
    }
}

fn compute_tensor_liveness(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    input_tensor_indices: &[i32],
    output_tensor_indices: &[i32],
) -> TensorLiveness {
    let output_set: std::collections::HashSet<usize> = output_tensor_indices
        .iter()
        .filter(|&&i| i >= 0)
        .map(|&i| i as usize)
        .collect();

    // last-consumer map: tensor_idx → op.index of last use
    let mut last_consumer: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        for &idx in &op.inputs {
            if idx >= 0 {
                let e = last_consumer.entry(idx as usize).or_insert(0);
                if op.index > *e {
                    *e = op.index;
                }
            }
        }
    }
    for &idx in output_tensor_indices {
        if idx >= 0 {
            last_consumer.insert(idx as usize, usize::MAX);
        }
    }

    let mut relevant = HashSet::<usize>::new();
    relevant.extend(
        input_tensor_indices
            .iter()
            .filter(|index| **index >= 0)
            .map(|index| *index as usize),
    );
    for op in ops {
        relevant.extend(
            op.outputs
                .iter()
                .filter(|index| **index >= 0)
                .map(|index| *index as usize),
        );
    }
    let mut sizes = HashMap::<usize, usize>::new();
    let mut issues = Vec::<TensorPayloadIssue>::new();
    let mut relevant_indices = relevant.into_iter().collect::<Vec<_>>();
    relevant_indices.sort_unstable();
    for tensor_index in relevant_indices {
        let Some(tensor) = tensors.get(tensor_index) else {
            issues.push(TensorPayloadIssue {
                tensor_index: Some(tensor_index),
                tensor_name: String::new(),
                reason: "Tensor index is outside the parsed tensor inventory.".to_string(),
            });
            continue;
        };
        if tensor.constant_buffer {
            continue;
        }
        match declared_tensor_payload_bytes(tensor) {
            Ok(size) => {
                sizes.insert(tensor_index, size);
            }
            Err(reason) => issues.push(TensorPayloadIssue {
                tensor_index: Some(tensor_index),
                tensor_name: tensor.name.clone(),
                reason,
            }),
        }
    }
    if !issues.is_empty() {
        return unassessed_tensor_liveness(sizes.len(), issues);
    }

    let mut current = 0usize;
    let mut peak_at_op = None::<usize>;
    let mut peak_at_op_name = None::<String>;
    let mut live = HashSet::<usize>::new();

    // model inputs live from the start
    for &idx in input_tensor_indices {
        if idx >= 0 {
            let tensor_index = idx as usize;
            if live.insert(tensor_index) {
                let size = sizes.get(&tensor_index).copied().unwrap_or(0);
                let Some(next) = current.checked_add(size) else {
                    return unassessed_tensor_liveness(
                        sizes.len(),
                        vec![TensorPayloadIssue {
                            tensor_index: Some(tensor_index),
                            tensor_name: tensors
                                .get(tensor_index)
                                .map(|tensor| tensor.name.clone())
                                .unwrap_or_default(),
                            reason: "Live payload sum exceeds the analyzer integer range."
                                .to_string(),
                        }],
                    );
                };
                current = next;
            }
        }
    }
    let mut peak = current;

    for op in ops {
        for &idx in &op.outputs {
            if idx >= 0 {
                let tensor_index = idx as usize;
                if live.insert(tensor_index) {
                    let size = sizes.get(&tensor_index).copied().unwrap_or(0);
                    let Some(next) = current.checked_add(size) else {
                        return unassessed_tensor_liveness(
                            sizes.len(),
                            vec![TensorPayloadIssue {
                                tensor_index: Some(tensor_index),
                                tensor_name: tensors
                                    .get(tensor_index)
                                    .map(|tensor| tensor.name.clone())
                                    .unwrap_or_default(),
                                reason: "Live payload sum exceeds the analyzer integer range."
                                    .to_string(),
                            }],
                        );
                    };
                    current = next;
                }
            }
        }
        if current > peak {
            peak = current;
            peak_at_op = Some(op.index);
            peak_at_op_name = Some(op.name.clone());
        }
        // free tensors whose last use is this op
        for &idx in op.inputs.iter().chain(op.outputs.iter()) {
            if idx < 0 {
                continue;
            }
            let uidx = idx as usize;
            if output_set.contains(&uidx) {
                continue;
            }
            if last_consumer.get(&uidx).copied() == Some(op.index) && live.remove(&uidx) {
                current = current.saturating_sub(sizes.get(&uidx).copied().unwrap_or(0));
            }
        }
    }

    TensorLiveness {
        assessed: true,
        status: "assessed".to_string(),
        peak_bytes_value: Some(peak),
        peak_bytes: peak,
        peak_at_op_value: peak_at_op,
        peak_at_op: peak_at_op.unwrap_or(0),
        peak_at_op_name_value: peak_at_op_name.clone(),
        peak_at_op_name: peak_at_op_name.unwrap_or_else(|| "graph input".to_string()),
        assessed_tensor_count: sizes.len(),
        unassessed_tensor_count: 0,
        unassessed_tensors: Vec::new(),
        method: "Exact declared-shape payload sweep from producer through last consumer; constants excluded and no arena reuse applied.".to_string(),
    }
}

const TFLITE_ARENA_SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const TFLITE_TENSOR_ALIGNMENT_BYTES: usize = 64;
fn validate_input_contracts(
    tensors: &[TensorInfo],
    input_tensor_indices: &[i32],
    ops: &[OpInfo],
) -> Vec<InputContract> {
    input_tensor_indices.iter().filter_map(|&idx| {
        if idx < 0 { return None; }
        let t = tensors.get(idx as usize)?;
        let is_quantized = t.quant_scales > 0;
        let quantized_code_range = match t.dtype.as_str() {
            "UINT8" => Some((0_i64, 255_i64)),
            "INT8" => Some((-128, 127)),
            "UINT16" => Some((0, 65_535)),
            "INT16" => Some((-32_768, 32_767)),
            "INT32" => Some((i32::MIN as i64, i32::MAX as i64)),
            "INT4" => Some((-8, 7)),
            "UINT4" => Some((0, 15)),
            "UINT2" => Some((0, 3)),
            "INT2" => Some((-2, 1)),
            _ => None,
        };
        let scalar_quantization = t.quant_scales == 1
            && t.quant_zero_points == 1
            && t.scale_sample.len() == 1
            && t.zero_point_sample.len() == 1;
        let valid_scale = t
            .scale_sample
            .first()
            .copied()
            .filter(|scale| scale.is_finite() && *scale > 0.0);
        let valid_zero_point = t.zero_point_sample.first().copied().filter(|zero_point| {
            quantized_code_range
                .map(|(minimum, maximum)| *zero_point >= minimum && *zero_point <= maximum)
                .unwrap_or(false)
        });
        let (range_low, range_high, range_note, tensor_numerical_contract_status) = if let (
            true,
            true,
            Some(scale),
            Some(zero_point),
            Some((type_min, type_max)),
        ) = (
            is_quantized,
            scalar_quantization,
            valid_scale,
            valid_zero_point,
            quantized_code_range,
        ) {
            let scale = scale as f64;
            let low = scale * (type_min - zero_point) as f64;
            let high = scale * (type_max - zero_point) as f64;
            let note = format!(
                "{} dequantized code domain [{:.9}, {:.9}] from scalar scale {:.9} and zero point {}",
                t.dtype, low, high, scale, zero_point
            );
            (
                Some(low),
                Some(high),
                note,
                "known_from_artifact_quantization_metadata",
            )
        } else if is_quantized {
            let status = if t.quant_scales != 1 || t.quant_zero_points != 1 {
                "not_assessed_non_scalar_input_quantization"
            } else if quantized_code_range.is_none() {
                "not_assessed_unsupported_quantized_input_dtype"
            } else {
                "invalid_or_incomplete_quantization_metadata"
            };
            (
                None,
                None,
                format!(
                    "{} input declares {} scale(s) and {} zero point(s); no scalar real range is emitted because a complete, positive, in-range scalar quantization contract was not established",
                    t.dtype, t.quant_scales, t.quant_zero_points
                ),
                status,
            )
        } else if t.dtype == "FLOAT32" {
            (
                None,
                None,
                "FLOAT32 normalization range is not encoded in the tensor contract; verify it against the training pipeline.".to_string(),
                "not_embedded_in_artifact",
            )
        } else {
            (
                None,
                None,
                format!("{} input has no artifact-bound scalar real range", t.dtype),
                "not_embedded_in_artifact",
            )
        };

        let semantic_consumer = if t.shape.len() == 4 {
            ops.iter().find(|op| {
                op.inputs.first().copied() == Some(idx)
                    && matches!(
                        op.name.as_str(),
                        "CONV_2D"
                            | "DEPTHWISE_CONV_2D"
                            | "AVERAGE_POOL_2D"
                            | "MAX_POOL_2D"
                            | "RESIZE_BILINEAR"
                            | "RESIZE_NEAREST_NEIGHBOR"
                    )
            })
        } else {
            None
        };
        let (layout, layout_status, layout_evidence_class, layout_source_op_index, layout_source_op_name, layout_reason, channel_axis, channels) =
            if t.shape.len() != 4 {
                (
                    None,
                    "not_applicable_non_4d_input".to_string(),
                    "NOT_APPLICABLE".to_string(),
                    None,
                    None,
                    format!("Rank {} input does not carry a four-dimensional image-layout contract.", t.shape.len()),
                    None,
                    None,
                )
            } else if let Some(consumer) = semantic_consumer {
                (
                    Some("NHWC".to_string()),
                    "derived_nhwc_from_direct_consumer_semantics".to_string(),
                    "DERIVED".to_string(),
                    Some(consumer.index),
                    Some(consumer.name.clone()),
                    format!(
                        "Graph input T{} is activation input 0 of TFLite {} #{:03}; the pinned operator tensor semantics require NHWC.",
                        idx, consumer.name, consumer.index
                    ),
                    Some(3),
                    t.shape.get(3).copied(),
                )
            } else {
                (
                    None,
                    "not_assessed_no_direct_layout_semantic_consumer".to_string(),
                    "NOT_ASSESSABLE".to_string(),
                    None,
                    None,
                    "Rank alone does not determine layout, and no supported direct activation-input consumer fixes the channel axis.".to_string(),
                    None,
                    None,
                )
            };
        let mut risks = Vec::new();
        if t.dtype == "FLOAT32" && !is_quantized {
            risks.push("FLOAT32 normalization range is unknown and must match the training pipeline exactly.".to_string());
        }
        if layout.is_none() && t.shape.len() == 4 {
            risks.push("Input layout and channel axis are not determined by the supported direct-consumer semantics.".to_string());
        }
        if channels == Some(3) {
            risks.push("Channel order (RGB vs BGR) cannot be determined from graph structure; a mismatch may silently degrade task performance.".to_string());
        }
        Some(InputContract {
            schema: "deepbom.input_tensor_contract.v1".to_string(),
            tensor_index: idx as usize,
            name: t.name.clone(),
            shape: t.shape.clone(),
            dtype: t.dtype.clone(),
            is_quantized,
            expected_range_low: range_low,
            expected_range_high: range_high,
            range_note,
            tensor_numerical_contract_status: tensor_numerical_contract_status.to_string(),
            source_data_to_tensor_preprocessing_status: "not_embedded_in_artifact".to_string(),
            layout,
            layout_status,
            layout_evidence_class,
            layout_source_op_index,
            layout_source_op_name,
            layout_reason,
            channel_axis,
            channels,
            risks,
        })
    }).collect()
}

fn fmt_mb(bytes: usize) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    if mb >= 1.0 {
        format!("{:.1} MB", mb)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}

fn read_primary_subgraph_tensors(bytes: &[u8]) -> Result<Vec<TensorInfo>, String> {
    let fb = Fb::verified_tflite(bytes)?;
    let model = fb.root_table()?;
    let subgraph = fb
        .vector_tables(model, 2)
        .first()
        .copied()
        .ok_or_else(|| "Model has no subgraph".to_string())?;
    let buffer_locations = read_buffer_locations(&fb, model);
    fb.vector_tables(subgraph, 0)
        .iter()
        .enumerate()
        .map(|(index, table)| read_tensor(&fb, index, *table, &buffer_locations))
        .collect::<Result<Vec<_>, _>>()
}

fn analyze(bytes: &[u8], filename: &str) -> Result<Analysis, String> {
    analyze_with_target(bytes, filename, "android_mid_a55")
}

fn analyze_with_target(bytes: &[u8], filename: &str, target_id: &str) -> Result<Analysis, String> {
    analyze_with_target_scope(bytes, filename, target_id, true)
}

fn analyze_with_target_without_step_response(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
) -> Result<Analysis, String> {
    analyze_with_target_scope(bytes, filename, target_id, false)
}

fn analyze_with_target_scope(
    bytes: &[u8],
    filename: &str,
    target_id: &str,
    include_step_response: bool,
) -> Result<Analysis, String> {
    let target = target_profile(target_id)?;
    let fb = Fb::verified_tflite(bytes)?;
    let model = fb.root_table()?;
    let version = fb.checked_u32_field(model, 0, 0, "Model.version")?;

    let operator_code_tables = fb.vector_tables(model, 1);
    let operator_names = operator_code_tables
        .iter()
        .map(|table| operator_code_name(&fb, *table))
        .collect::<Result<Vec<_>, _>>()?;
    let operator_versions = operator_code_tables
        .iter()
        .map(|table| {
            fb.checked_i32_field(*table, 2, 1, "OperatorCode.version")
                .map(|value| value.max(1))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let subgraph_tables = fb.vector_tables(model, 2);
    let Some(subgraph) = subgraph_tables.first().copied() else {
        return Err("Model has no subgraph".to_string());
    };

    let buffer_locations = read_buffer_locations(&fb, model);
    let scoped_tensors = subgraph_tables
        .iter()
        .map(|subgraph| {
            fb.vector_tables(*subgraph, 0)
                .iter()
                .enumerate()
                .map(|(index, table)| read_tensor(&fb, index, *table, &buffer_locations))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = scoped_tensors[0].clone();
    let tflite_subgraph_inventory = build_tflite_subgraph_inventory(
        &fb,
        model,
        &operator_names,
        &operator_versions,
        &scoped_tensors,
    )?;

    let input_tensor_indices = fb.vector_i32(subgraph, 1);
    let output_tensor_indices = fb.vector_i32(subgraph, 2);
    let inputs = input_tensor_indices
        .iter()
        .filter_map(|idx| tensors.get(*idx as usize).cloned())
        .collect::<Vec<_>>();
    let outputs = output_tensor_indices
        .iter()
        .filter_map(|idx| tensors.get(*idx as usize).cloned())
        .collect::<Vec<_>>();

    let mut tensor_type_counts = BTreeMap::<String, usize>::new();
    let mut quantized_tensors = 0usize;
    let mut per_channel_tensors = 0usize;
    for tensor in &tensors {
        *tensor_type_counts.entry(tensor.dtype.clone()).or_default() += 1;
        if tensor.quant_scales > 0 {
            quantized_tensors += 1;
        }
        if tensor.quant_scales > 1 {
            per_channel_tensors += 1;
        }
    }

    let build_scope_ops =
        |scoped_subgraph: usize,
         scoped_tensors: &[TensorInfo],
         operator_intrinsics: &[tflite_subgraphs::SubgraphOperatorIntrinsic]|
         -> Result<(Vec<OpInfo>, BTreeMap<String, usize>), String> {
            let operator_tables = fb.vector_tables(scoped_subgraph, 3);
            if operator_intrinsics.len() != operator_tables.len() {
                return Err("TFLite operator/intrinsic cardinality mismatch".to_string());
            }
            let tensors = scoped_tensors;
            let mut histogram = BTreeMap::<String, usize>::new();
            let mut ops = Vec::<OpInfo>::new();
            for (index, table) in operator_tables.iter().enumerate() {
                let intrinsic = &operator_intrinsics[index];
                let name = intrinsic.name.clone();
                let op_version = intrinsic.version;
                *histogram.entry(name.clone()).or_default() += 1;

                let inputs_idx = intrinsic.inputs.clone();
                let outputs_idx = intrinsic.outputs.clone();
                let options_table = fb.table_field(*table, 4);
                let fused_activation = fused_activation_for_op(&fb, &name, options_table);
                let output_shapes = outputs_idx
                    .iter()
                    .filter_map(|idx| {
                        tensors
                            .get(*idx as usize)
                            .map(|tensor| tensor.shape.clone())
                    })
                    .collect::<Vec<_>>();
                let macs = intrinsic.raw_macs;
                let ops_count = intrinsic.raw_ops;
                let estimated_bytes = intrinsic.raw_estimated_bytes;
                let estimated_input_strip = intrinsic.raw_estimated_input_strip;
                let positive_option = |field: usize| {
                    options_table
                        .and_then(|table| fb.field_pos(table, field))
                        .and_then(|position| fb.i32(position))
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .unwrap_or(1)
                };
                let conv_kernel_geometry = (name == "CONV_2D").then(|| Conv2dKernelGeometry {
                    stride_width: positive_option(1),
                    stride_height: positive_option(2),
                    dilation_width: positive_option(4),
                    dilation_height: positive_option(5),
                });
                let dilation_h = match name.as_str() {
                    "CONV_2D" => {
                        conv_kernel_geometry.map(|geometry| geometry.dilation_height as i32)
                    }
                    "DEPTHWISE_CONV_2D" => options_table
                        .and_then(|table| fb.field_pos(table, 6))
                        .and_then(|position| fb.i32(position)),
                    _ => Some(1),
                }
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .unwrap_or(1);
                let cache_payload = logical_cache_payload_for_op(
                    &name,
                    &inputs_idx,
                    &outputs_idx,
                    tensors,
                    dilation_h,
                );
                let row_ws = cache_payload
                    .logical_row_payload_bytes
                    .map(|value| value as f64)
                    .unwrap_or(estimated_input_strip);
                let row_ws_ratio = if target.l1_data_bytes > 0 {
                    row_ws / target.l1_data_bytes as f64
                } else {
                    0.0
                };
                let intensity = if estimated_bytes > 0.0 {
                    ops_count / estimated_bytes
                } else {
                    0.0
                };
                // Detect FP32 from first input tensor dtype to select correct roofline thresholds.
                let is_float_op = inputs_idx
                    .first()
                    .and_then(|&i| tensors.get(i as usize))
                    .map(|t| is_float_dtype(&t.dtype))
                    .unwrap_or(false);
                let bound = static_bound_guess_for_target(
                    &name,
                    ops_count,
                    estimated_bytes,
                    intensity,
                    &target,
                    is_float_op,
                )
                .to_string();
                let roofline_reason = roofline_reason_for_op(
                    &name,
                    ops_count,
                    estimated_bytes,
                    intensity,
                    &target,
                    is_float_op,
                );
                let (xnnpack_supported, xnnpack_reason) =
                    xnnpack_support_for_op(&XnnpackOpContext {
                        name: &name,
                        inputs: &inputs_idx,
                        outputs: &outputs_idx,
                        tensors,
                        model_bytes: bytes,
                        fb: &fb,
                        options_table,
                        fused_activation: &fused_activation,
                    });
                let quantized_path = op_has_8bit_quant(&inputs_idx, &outputs_idx, tensors);
                let quantized_compute_path =
                    op_has_quantized_compute_path(&name, &inputs_idx, &outputs_idx, tensors);
                let (quantization_state, quantization_detail) =
                    classify_op_quantization(&name, &inputs_idx, &outputs_idx, tensors);
                let quant_summary = quant_risk_for_op(&name, &inputs_idx, &outputs_idx, tensors);
                let kernel_candidate =
                    source_kernel_candidate_for_op(&name, &inputs_idx, tensors, &target);
                let compute_precision_label = inputs_idx
                    .first()
                    .and_then(|idx| tensors.get(*idx as usize))
                    .map(|tensor| precision_label_for_dtype(&tensor.dtype))
                    .unwrap_or(if quantized_compute_path {
                        "8-bit"
                    } else {
                        "non-quantized"
                    });
                let kernel_class =
                    compute_kernel_class(&name, &inputs_idx, tensors, conv_kernel_geometry);
                let target_microkernel_hint = if kernel_candidate
                    .evidence_class
                    .starts_with("SOURCE_ENUMERATED_")
                {
                    kernel_candidate.candidate.clone()
                } else {
                    microkernel_hint(&name, &inputs_idx, &outputs_idx, tensors, &target)
                };
                let packing =
                    weight_packing_for_op(&name, &inputs_idx, &outputs_idx, tensors, &target);
                let alignment = channel_alignment_for_op(
                    &name,
                    &outputs_idx,
                    tensors,
                    &target,
                    ChannelAlignmentContext {
                        is_float_compute: !quantized_compute_path,
                        compute_precision_label,
                        source_tile_multiple: kernel_candidate
                            .channel_tile
                            .max(kernel_candidate.tile_nr),
                        source_tile_multiples: &kernel_candidate.alignment_multiples,
                        kernel_evidence_class: &kernel_candidate.evidence_class,
                    },
                );
                let (low_norm_filter_count, low_norm_filter_total) = if include_step_response
                    && matches!(
                        name.as_str(),
                        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "TRANSPOSE_CONV"
                    ) {
                    inputs_idx
                        .get(1)
                        .and_then(|index| usize::try_from(*index).ok())
                        .and_then(|index| tensors.get(index))
                        .and_then(|tensor| {
                            compute_quick_low_norm_stat_for_tensor(
                                bytes,
                                tensor,
                                name == "DEPTHWISE_CONV_2D",
                            )
                        })
                        .map(|stat| (Some(stat.low_norm), Some(stat.total)))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                let op = OpInfo {
                    index,
                    name: name.clone(),
                    version: op_version,
                    stage_index: None,
                    stage_key: None,
                    inputs: inputs_idx,
                    outputs: outputs_idx,
                    output_shapes,
                    macs,
                    mac_percent: 0.0,
                    ops: ops_count,
                    estimated_bytes,
                    fallback_byte_percent: 0.0,
                    row_working_set_bytes: row_ws,
                    row_working_set_ratio: row_ws_ratio,
                    row_working_set_severity: l1_working_set_severity(row_ws_ratio).to_string(),
                    cache_payload,
                    intensity_ops_per_byte: intensity,
                    static_bound_guess: bound,
                    static_action: roofline_action(
                        &name,
                        ops_count,
                        estimated_bytes,
                        intensity,
                        RooflineActionContext {
                            target: &target,
                            is_float: is_float_op,
                            quantized_compute_path,
                            xnnpack_supported,
                            channel_alignment_status: &alignment.status,
                        },
                    )
                    .to_string(),
                    roofline_reason,
                    fused_activation,
                    fusion_status: "unclassified".to_string(),
                    fusion_detail: String::new(),
                    xnnpack_supported,
                    xnnpack_reason,
                    xnnpack_chain_id: -1,
                    xnnpack_chain_role: "fallback".to_string(),
                    xnnpack_chain_break: false,
                    xnnpack_break_class: "none".to_string(),
                    chain_break_impact_mac_percent: 0.0,
                    chain_break_overhead_us_low: 0.0,
                    chain_break_overhead_us_high: 0.0,
                    target_microkernel_hint,
                    compute_kernel_class: kernel_class,
                    xnnpack_kernel_candidate: kernel_candidate.candidate,
                    xnnpack_kernel_tile_mr: kernel_candidate.tile_mr,
                    xnnpack_kernel_tile_nr: kernel_candidate.tile_nr,
                    xnnpack_kernel_channel_tile: kernel_candidate.channel_tile,
                    xnnpack_kernel_primary_tile: kernel_candidate.primary_tile,
                    xnnpack_kernel_source: kernel_candidate.source,
                    xnnpack_kernel_evidence_class: kernel_candidate.evidence_class,
                    xnnpack_kernel_selector_status: kernel_candidate.selector_status,
                    xnnpack_build_requirement: kernel_candidate.build_requirement,
                    xnnpack_kernel_candidates: kernel_candidate.candidates,
                    xnnpack_kernel_alignment_multiples: kernel_candidate.alignment_multiples,
                    quantized_path,
                    quantized_compute_path,
                    quantization_state,
                    quantization_detail,
                    weight_bytes: packing.weight_bytes,
                    weight_packing_overhead_us: packing.overhead_us,
                    weight_packing_risk: packing.risk,
                    weight_packing_detail: packing.detail,
                    output_channels: alignment.output_channels,
                    channel_alignment_multiple: alignment.multiple,
                    channel_alignment_status: alignment.status,
                    channel_alignment_detail: alignment.detail,
                    channel_tail_overhead_percent: alignment.tail_overhead_percent,
                    channel_tail_overhead_percent_min: alignment.tail_overhead_percent_min,
                    channel_tail_overhead_percent_max: alignment.tail_overhead_percent_max,
                    quant_scale_ratio: quant_summary.scale_ratio,
                    quant_scale_cv: quant_summary.scale_cv,
                    quant_scale_mode: quant_summary.scale_mode,
                    quant_scale_ratio_meaningful: quant_summary.scale_ratio_meaningful,
                    quant_zero_point_offset: quant_summary.zero_point_offset,
                    quant_zero_point_risk: quant_summary.zero_point_risk,
                    quant_zero_point_status: quant_summary.zero_point_status,
                    quant_risk: quant_summary.label,
                    quant_risk_detail: quant_summary.detail,
                    low_norm_filter_count,
                    low_norm_filter_total,
                    quant_hole: false,
                    quant_hole_class: String::new(),
                    quant_hole_detail: String::new(),
                    patterns: Vec::new(),
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
                };
                ops.push(op);
            }

            Ok((ops, histogram))
        };

    let (mut ops, histogram) = build_scope_ops(
        subgraph,
        &tensors,
        &tflite_subgraph_inventory.rows[0].operator_intrinsics,
    )?;

    suppress_graph_output_channel_alignment(&mut ops, &output_tensor_indices);
    annotate_fusion(&mut ops);
    let patterns = detect_patterns(&ops, &tensors);
    annotate_patterns(&mut ops, &patterns);
    let total_macs = ops.iter().map(|op| op.macs).sum::<f64>().abs();
    let total_ops = ops.iter().map(|op| op.ops).sum::<f64>().abs();
    let mut xnnpack_chains = annotate_xnnpack_chains(&mut ops, &target);
    for chain in &mut xnnpack_chains {
        chain.mac_percent = if total_macs > 0.0 {
            chain.macs / total_macs
        } else {
            0.0
        };
        chain.chain_class = xnnpack_chain_class(chain);
    }
    let xnnpack_longest_chain = xnnpack_chains
        .iter()
        .map(|chain| chain.op_count)
        .max()
        .unwrap_or(0);
    let delegated_macs = ops
        .iter()
        .filter(|op| op.xnnpack_chain_id >= 0)
        .map(|op| op.macs)
        .sum::<f64>()
        .abs();
    let fallback_macs = (total_macs - delegated_macs).max(0.0);
    let delegated_mac_percent = if total_macs > 0.0 {
        delegated_macs / total_macs
    } else {
        0.0
    };
    let total_estimated_bytes = ops.iter().map(|op| op.estimated_bytes).sum::<f64>().abs();
    let delegated_estimated_bytes = ops
        .iter()
        .filter(|op| op.xnnpack_chain_id >= 0)
        .map(|op| op.estimated_bytes)
        .sum::<f64>()
        .abs();
    let fallback_estimated_bytes = (total_estimated_bytes - delegated_estimated_bytes).max(0.0);
    let fallback_byte_percent = if total_estimated_bytes > 0.0 {
        fallback_estimated_bytes / total_estimated_bytes
    } else {
        0.0
    };
    for op in &mut ops {
        op.mac_percent = if total_macs > 0.0 {
            op.macs / total_macs
        } else {
            0.0
        };
        op.fallback_byte_percent = if op.xnnpack_chain_id < 0 && total_estimated_bytes > 0.0 {
            op.estimated_bytes / total_estimated_bytes
        } else {
            0.0
        };
    }
    annotate_chain_break_impact(&mut ops, &xnnpack_chains, total_macs, total_estimated_bytes);
    let predicted_partition_boundaries =
        compute_predicted_partition_boundary_inventory(&ops, &tensors);
    let xnnpack_chain_breaks = ops.iter().filter(|op| op.xnnpack_chain_break).count();
    let xnnpack_effective_chain_breaks = ops
        .iter()
        .filter(|op| op.xnnpack_chain_break && !is_structural_view_op(&op.name))
        .count();
    let xnnpack_structural_chain_breaks = ops
        .iter()
        .filter(|op| op.xnnpack_chain_break && is_structural_view_op(&op.name))
        .count();
    let xnnpack_zero_mac_chain_breaks = ops
        .iter()
        .filter(|op| op.xnnpack_chain_break && op.macs == 0.0 && !is_structural_view_op(&op.name))
        .count();
    let fallback_traffic_by_op_family =
        fallback_traffic_by_family(&ops, total_estimated_bytes, total_macs);
    let (estimated_int8_speedup, estimated_int8_speedup_detail) = estimate_model_int8_speedup(
        &ops,
        total_macs,
        quantized_tensors,
        tensors.len(),
        fallback_byte_percent,
        &target,
    );
    let quantization_status = classify_model_quantization(
        &ops,
        &tensors,
        &inputs,
        &outputs,
        quantized_tensors,
        total_macs,
    );
    let mut suspects_map = BTreeMap::<String, usize>::new();
    for op in ops.iter().filter(|op| {
        !op.xnnpack_supported
            && is_delegate_break_suspect(&op.name)
            && !is_structural_view_op(&op.name)
    }) {
        *suspects_map.entry(op.name.clone()).or_default() += 1;
    }
    let suspects = count_items(suspects_map.clone());

    let stages = build_stages(&ops, total_macs, &patterns);
    annotate_op_stages(&mut ops, &stages)?;
    let fc_ops = ops.iter().filter(|op| op.name == "FULLY_CONNECTED").count();
    let fc_packing_warn_ops = ops
        .iter()
        .filter(|op| op.name == "FULLY_CONNECTED" && op.weight_packing_risk == "warn")
        .count();
    let conv_weight_ops = ops
        .iter()
        .filter(|op| matches!(op.name.as_str(), "CONV_2D" | "DEPTHWISE_CONV_2D"))
        .count();
    let conv_packing_warn_ops = ops
        .iter()
        .filter(|op| {
            matches!(op.name.as_str(), "CONV_2D" | "DEPTHWISE_CONV_2D")
                && op.weight_packing_risk == "warn"
        })
        .count();
    let quant_holes = detect_quant_holes(
        &mut ops,
        &input_tensor_indices,
        &output_tensor_indices,
        &tensors,
        total_macs,
    );
    let quant_hole_count = quant_holes.len();
    let quant_hole_mac_impact = quant_holes
        .iter()
        .map(|h| h.adjacent_mac_percent)
        .fold(0.0_f64, f64::max);

    // Topology annotations and bottleneck estimates are computed here.
    compute_topology_annotations(&mut ops);
    compute_bottleneck_estimates(&mut ops, &target);
    let core_isolation_analysis = core_isolation::analyze(&ops, &target);
    let core_isolation_csv = core_isolation::render_csv(&core_isolation_analysis);
    let recommendations = build_recommendations(
        &ops,
        &target,
        xnnpack_chain_breaks,
        xnnpack_effective_chain_breaks,
        fallback_byte_percent,
        &fallback_traffic_by_op_family,
        &patterns,
    );
    let block_inventory = build_block_inventory(&ops, &tensors, &patterns, &target, total_macs);
    let weight_integrity = compute_weight_integrity(bytes, &tensors, &ops, total_macs);

    // Insights are computed here and emitted as structured evidence.
    let insights = compute_insights(
        &ops,
        &tensors,
        &inputs,
        quantized_tensors,
        per_channel_tensors,
        xnnpack_chain_breaks,
        xnnpack_effective_chain_breaks,
        &xnnpack_chains,
        delegated_mac_percent,
        fallback_byte_percent,
        &fallback_traffic_by_op_family,
        &suspects,
        weight_integrity.exact_zero_kernel_slice_count,
    );

    // Findings are computed here and emitted as structured evidence.
    let tensor_liveness = compute_tensor_liveness(
        &ops,
        &tensors,
        &input_tensor_indices,
        &output_tensor_indices,
    );
    let tensor_arena_plan = compute_tensor_arena_plan(
        &ops,
        &tensors,
        &input_tensor_indices,
        &output_tensor_indices,
    );
    let dynamic_shape_cost_contract = build_tflite_dynamic_shape_cost_contract(
        &ops,
        &tensors,
        &input_tensor_indices,
        &output_tensor_indices,
    );
    let movement_analysis = compute_movement_analysis(&ops, &tensors);
    let input_contracts = validate_input_contracts(&tensors, &input_tensor_indices, &ops);
    let accumulator_atlas = build_accumulator_atlas(bytes, &ops, &tensors);
    let requantization_fidelity = build_requantization_fidelity(&ops, &tensors, &accumulator_atlas);
    let kernel_extremum_witness = if include_step_response {
        build_kernel_witnesses(
            bytes,
            &ops,
            &tensors,
            &accumulator_atlas,
            &requantization_fidelity,
        )
    } else {
        kernel_witness_not_computed()
    };
    let channel_vitality = if include_step_response {
        build_channel_vitality(&kernel_extremum_witness)
    } else {
        channel_vitality_not_computed()
    };
    let rounding_equivalence = if include_step_response {
        build_rounding_equivalence(&kernel_extremum_witness, &requantization_fidelity)
    } else {
        rounding_equivalence_not_computed()
    };
    let accumulator_reachability = if include_step_response {
        build_accumulator_reachability(
            bytes,
            &ops,
            &tensors,
            &kernel_extremum_witness,
            &requantization_fidelity,
            &rounding_equivalence,
        )
    } else {
        accumulator_reachability_not_computed()
    };
    let numerical_abi_propagation = if include_step_response {
        build_numerical_abi_propagation(
            &ops,
            &tensors,
            &output_tensor_indices,
            &rounding_equivalence,
            &accumulator_reachability,
        )
    } else {
        numerical_abi_propagation_not_computed()
    };
    let input_counterexample = if include_step_response {
        build_input_counterexamples(
            bytes,
            &ops,
            &tensors,
            &input_tensor_indices,
            (
                &accumulator_reachability,
                &rounding_equivalence,
                &requantization_fidelity,
                &numerical_abi_propagation,
            ),
        )
    } else {
        input_counterexample_not_computed()
    };
    let preprocessing_realizability = if include_step_response {
        build_preprocessing_realizability(&input_counterexample)
    } else {
        preprocessing_realizability_not_computed()
    };
    let quantization_lattice = build_quantization_lattice(&ops, &tensors);
    let contract_migration = build_contract_migration(
        bytes,
        &ops,
        &tensors,
        &quantization_lattice,
        &accumulator_atlas,
    );
    let residual_step_response = if include_step_response {
        build_residual_step_response(&ops, &tensors, &quantization_lattice)
    } else {
        residual_step_response_not_computed()
    };
    let residual_contract_distortion = if include_step_response {
        build_residual_contract_distortion(&ops, &tensors, &quantization_lattice)
    } else {
        residual_contract_distortion_not_computed()
    };
    let protected_selector_available =
        is_cortex_a55_profile(&target.id) || target.id == "wasm_simd";
    let packed_metadata_archive = parse_packed_metadata_archive(fb.data);
    let metadata_presence = compute_metadata_presence(
        &fb,
        model,
        subgraph,
        &buffer_locations,
        &outputs,
        packed_metadata_archive.clone(),
    );
    let tflite_sparse_storage_contract = build_sparse_storage_contract(&scoped_tensors)?;
    let mut nested_scope_inputs = Vec::with_capacity(subgraph_tables.len().saturating_sub(1));
    for subgraph_index in 1..subgraph_tables.len() {
        let row = &tflite_subgraph_inventory.rows[subgraph_index];
        let (nested_ops, _) = build_scope_ops(
            subgraph_tables[subgraph_index],
            &scoped_tensors[subgraph_index],
            &row.operator_intrinsics,
        )?;
        nested_scope_inputs.push(NestedScopeInput {
            subgraph_index,
            name: &row.name,
            reachable_from_entrypoint: row.reachable_from_entrypoint,
            invocation_semantics: &row.invocation_semantics,
            ops: nested_ops,
            tensors: &scoped_tensors[subgraph_index],
            input_tensor_indices: &row.input_tensor_indices,
            output_tensor_indices: &row.output_tensor_indices,
            model_bytes: bytes,
            target: &target,
            include_advanced_proofs: include_step_response,
        });
    }
    let primary_scope = &tflite_subgraph_inventory.rows[0];
    let tflite_subgraph_deep_analysis = build_deep_analysis(
        PrimaryScopeInput {
            subgraph_index: 0,
            name: &primary_scope.name,
            reachable_from_entrypoint: primary_scope.reachable_from_entrypoint,
            invocation_semantics: &primary_scope.invocation_semantics,
            ops: &ops,
            tensors: &tensors,
            total_macs,
            total_ops,
            quantized_tensor_count: quantized_tensors,
            per_axis_tensor_count: per_channel_tensors,
            quantization_status: &quantization_status,
            chains: &xnnpack_chains,
            predicted_partition_boundaries: &predicted_partition_boundaries,
            tensor_liveness: &tensor_liveness,
            tensor_arena_plan: &tensor_arena_plan,
            movement_analysis: &movement_analysis,
            weight_integrity: &weight_integrity,
            target: &target,
        },
        nested_scope_inputs,
    )?;

    let artifact_byte_integrity = build_artifact_byte_integrity_ledger(
        bytes,
        &fb.referenced_byte_ranges(),
        &buffer_locations,
        &packed_metadata_archive,
    );
    let findings = build_findings_from_analysis(FindingAnalysisContext {
        ops: &ops,
        tensors: &tensors,
        liveness: &tensor_liveness,
        movement: &movement_analysis,
        contracts: &input_contracts,
        boundaries: &predicted_partition_boundaries,
        weight_integrity: &weight_integrity,
        byte_integrity: &artifact_byte_integrity,
    });
    let mut analysis = Analysis {
        format: "tflite".to_string(),
        filename: filename.to_string(),
        file_size: bytes.len(),
        model_sha256: String::new(),
        target_profile: target,
        version,
        subgraphs: subgraph_tables.len(),
        tflite_subgraph_inventory,
        tflite_subgraph_deep_analysis,
        operator_codes: operator_code_tables.len(),
        operator_count: ops.len(),
        tensor_count: tensors.len(),
        tensors: tensors.clone(),
        inputs,
        outputs,
        input_tensor_indices,
        output_tensor_indices,
        histogram: count_items(histogram),
        tensor_types: count_items(tensor_type_counts),
        quantized_tensors,
        per_channel_tensors,
        quantization_status,
        accumulator_atlas,
        requantization_fidelity,
        kernel_extremum_witness,
        channel_vitality,
        rounding_equivalence,
        accumulator_reachability,
        numerical_abi_propagation,
        input_counterexample,
        preprocessing_realizability,
        quantization_lattice,
        contract_migration,
        residual_step_response,
        residual_contract_distortion,
        total_macs,
        total_ops,
        delegated_macs,
        fallback_macs,
        delegated_mac_percent,
        delegated_estimated_bytes,
        fallback_estimated_bytes,
        fallback_byte_percent,
        fallback_traffic_by_op_family,
        estimated_int8_speedup,
        estimated_int8_speedup_detail,
        suspects,
        stages,
        ops,
        xnnpack_assumption: "XNNP:PRED:STATIC".to_string(),
        xnnpack_selector_assessment_status: if protected_selector_available {
            "not_loaded".to_string()
        } else {
            "not_available_for_profile".to_string()
        },
        xnnpack_selector_evidence_schema: String::new(),
        xnnpack_selector_evidence_access: if protected_selector_available {
            "research_authorization_required".to_string()
        } else {
            "not_applicable".to_string()
        },
        xnnpack_chain_breaks,
        xnnpack_effective_chain_breaks,
        xnnpack_structural_chain_breaks,
        xnnpack_zero_mac_chain_breaks,
        xnnpack_longest_chain,
        xnnpack_chains,
        predicted_partition_boundaries,
        fc_ops,
        fc_packing_warn_ops,
        conv_weight_ops,
        conv_packing_warn_ops,
        patterns,
        block_inventory,
        recommendations,
        quant_holes,
        quant_hole_count,
        quant_hole_mac_impact,
        stage_mermaid: String::new(),
        roofline_csv: String::new(),
        core_isolation_analysis,
        core_isolation_csv,
        tensor_liveness,
        tensor_arena_plan,
        movement_analysis,
        artifact_byte_integrity,
        input_contracts,
        dynamic_shape_cost_contract,
        weight_integrity,
        tflite_sparse_storage_contract,
        runtime_compat: compute_runtime_compat(
            &fb,
            model,
            &operator_code_tables,
            &buffer_locations,
        )?,
        size_breakdown: compute_size_breakdown(
            &fb,
            model,
            &tensors,
            &buffer_locations,
            bytes.len(),
        ),
        metadata_presence,
        findings,
        insights,
    };
    analysis.stage_mermaid = render_stage_mermaid(&analysis);
    analysis.roofline_csv = render_roofline_csv(&analysis);
    Ok(analysis)
}

// ── Weight integrity: NaN/Inf, all-zero kernel slices, magnitude, sparsity ────
fn compute_weight_integrity(
    bytes: &[u8],
    tensors: &[TensorInfo],
    ops: &[OpInfo],
    total_macs: f64,
) -> WeightIntegrityReport {
    let mut r = WeightIntegrityReport {
        weight_tensors_scanned: 0,
        sparse_constant_tensors_decoded: 0,
        sparse_constant_tensors_not_decoded: 0,
        sparse_logical_elements: 0,
        sparse_stored_elements: 0,
        sparse_implicit_zero_elements: 0,
        constant_value_coverage_status: String::new(),
        quantized_constant_tensors_scanned: 0,
        elements_scanned: 0,
        eligible_kernel_tensors_scanned: 0,
        output_channels_evaluated: 0,
        nan_tensors: 0,
        inf_tensors: 0,
        all_zero_tensors: 0,
        zero_kernel_slice_tensors: 0,
        zero_kernel_slice_count: 0,
        exact_zero_kernel_slice_tensors: 0,
        exact_zero_kernel_slice_count: 0,
        max_abs_weight: 0.0,
        large_magnitude_tensors: 0,
        mean_sparsity: 0.0,
        high_sparsity_tensors: 0,
        zero_kernel_slice_details: Vec::new(),
        low_grid_utilization_tensors: 0,
        saturated_quantized_tensors: 0,
        min_grid_utilization: 1.0,
        threshold_eligible_quantized_constant_tensors: 0,
        min_threshold_eligible_grid_utilization: None,
        max_saturation_percent: 0.0,
        quant_grid_detail: String::new(),
        status: "ok".to_string(),
        detail: String::new(),
    };
    let mut sparsity_acc = 0.0f64;
    for t in tensors {
        if !t.constant_buffer {
            continue;
        }
        if t.sparse_storage {
            let Some(sparse) = t
                .sparse_encoding
                .as_ref()
                .filter(|encoding| encoding.status == "assessed")
            else {
                r.sparse_constant_tensors_not_decoded += 1;
                continue;
            };
            r.sparse_constant_tensors_decoded += 1;
            r.sparse_logical_elements += sparse.logical_element_count;
            r.sparse_stored_elements += sparse.stored_element_count;
            r.sparse_implicit_zero_elements += sparse.implicit_zero_element_count;
        }
        let Some(w) = weight_to_f64(bytes, t) else {
            continue;
        };
        if w.is_empty() {
            continue;
        }
        r.weight_tensors_scanned += 1;
        let quantized_codes = quantized_raw_values(bytes, t);
        if matches!(t.dtype.as_str(), "INT8" | "UINT8") {
            r.quantized_constant_tensors_scanned += 1;
            if let Some(q) = quantized_codes.as_ref() {
                let mut seen = [false; 256];
                let (qmin, qmax) = if t.dtype == "INT8" {
                    (-128, 127)
                } else {
                    (0, 255)
                };
                let mut saturated = 0usize;
                for value in q {
                    let bucket = (*value - qmin).clamp(0, 255) as usize;
                    seen[bucket] = true;
                    if *value == qmin || *value == qmax {
                        saturated += 1;
                    }
                }
                let used = seen.iter().filter(|&&v| v).count();
                let utilization = used as f64 / 256.0;
                let saturation = if q.is_empty() {
                    0.0
                } else {
                    saturated as f64 / q.len() as f64
                };
                if utilization < r.min_grid_utilization {
                    r.min_grid_utilization = utilization;
                }
                if saturation > r.max_saturation_percent {
                    r.max_saturation_percent = saturation;
                }
                if q.len() >= 256 && utilization < 0.25 {
                    r.low_grid_utilization_tensors += 1;
                }
                if q.len() >= 256 {
                    r.threshold_eligible_quantized_constant_tensors += 1;
                    r.min_threshold_eligible_grid_utilization = Some(
                        r.min_threshold_eligible_grid_utilization
                            .map(|current| current.min(utilization))
                            .unwrap_or(utilization),
                    );
                }
                if q.len() >= 256 && saturation > 0.01 {
                    r.saturated_quantized_tensors += 1;
                }
            }
        }
        r.elements_scanned += w.len();
        let mut has_nan = false;
        let mut has_inf = false;
        let mut near_zero = 0usize;
        let mut nonzero = 0usize;
        let mut tmax = 0.0f64;
        for &v in &w {
            if v.is_nan() {
                has_nan = true;
            } else if v.is_infinite() {
                has_inf = true;
            } else {
                let a = v.abs();
                if a > tmax {
                    tmax = a;
                }
                if a < WEIGHT_NEAR_ZERO_EPS {
                    near_zero += 1;
                } else {
                    nonzero += 1;
                }
            }
        }
        if has_nan {
            r.nan_tensors += 1;
        }
        if has_inf {
            r.inf_tensors += 1;
        }
        if nonzero == 0 && !has_nan && !has_inf {
            r.all_zero_tensors += 1;
        }
        if tmax > r.max_abs_weight {
            r.max_abs_weight = tmax;
        }
        if tmax > 1e4 {
            r.large_magnitude_tensors += 1;
        }
        let sparsity = if !w.is_empty() {
            near_zero as f64 / w.len() as f64
        } else {
            0.0
        };
        sparsity_acc += sparsity;
        if sparsity > 0.5 {
            r.high_sparsity_tensors += 1;
        }
        // All-zero kernel output slices: decode only constants bound to the
        // actual weight slot of a supported compute operator. Shape alone is
        // not sufficient because lookup tables and metadata can also be 2D/4D.
        // Conv/FC [OC, ...] are contiguous by output channel; depthwise
        // [1, kH, kW, IC] uses IC as the output-channel grouping.
        let shape: Vec<usize> = t.shape.iter().map(|&d| (d as usize).max(1)).collect();
        let kernel_consumers = ops
            .iter()
            .filter(|op| {
                let weight_slot = match op.name.as_str() {
                    "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" => 1,
                    "TRANSPOSE_CONV" => 1,
                    _ => return false,
                };
                op.inputs.get(weight_slot).copied() == Some(t.index as i32)
            })
            .collect::<Vec<_>>();
        if !kernel_consumers.is_empty() && (shape.len() == 2 || shape.len() == 4) {
            r.eligible_kernel_tensors_scanned += 1;
            let is_depthwise = shape.len() == 4 && shape[0] == 1 && shape[3] > 1;
            let oc = if is_depthwise { shape[3] } else { shape[0] }.max(1);
            r.output_channels_evaluated += oc;
            let per = w.len() / oc.max(1);
            if per > 0 {
                let mut zero_slice_channels = Vec::<usize>::new();
                let mut exact_zero_slice_channels = Vec::<usize>::new();
                if is_depthwise {
                    let kh = shape[1];
                    let kw = shape[2];
                    for c in 0..oc {
                        let mut all_zero = true;
                        let mut exact_zero = true;
                        let zero_point = t
                            .zero_point_sample
                            .get(c)
                            .copied()
                            .or_else(|| t.zero_point_sample.first().copied())
                            .unwrap_or(0);
                        for h in 0..kh {
                            for k in 0..kw {
                                let idx = ((h * kw + k) * oc) + c;
                                if w.get(idx)
                                    .map(|v| v.abs() >= WEIGHT_NEAR_ZERO_EPS)
                                    .unwrap_or(false)
                                {
                                    all_zero = false;
                                }
                                if quantized_codes
                                    .as_ref()
                                    .and_then(|values| values.get(idx))
                                    .map(|value| i64::from(*value) != zero_point)
                                    .unwrap_or_else(|| {
                                        w.get(idx).map(|value| *value != 0.0).unwrap_or(true)
                                    })
                                {
                                    exact_zero = false;
                                }
                            }
                        }
                        if all_zero {
                            zero_slice_channels.push(c);
                        }
                        if exact_zero {
                            exact_zero_slice_channels.push(c);
                        }
                    }
                } else {
                    for c in 0..oc {
                        let slice = &w[c * per..((c + 1) * per).min(w.len())];
                        if slice.iter().all(|v| v.abs() < WEIGHT_NEAR_ZERO_EPS) {
                            zero_slice_channels.push(c);
                        }
                        let zero_point = t
                            .zero_point_sample
                            .get(c)
                            .copied()
                            .or_else(|| t.zero_point_sample.first().copied())
                            .unwrap_or(0);
                        let start = c * per;
                        let end = ((c + 1) * per).min(w.len());
                        let exact_zero = quantized_codes
                            .as_ref()
                            .map(|values| {
                                values.get(start..end).is_some_and(|codes| {
                                    codes.iter().all(|value| i64::from(*value) == zero_point)
                                })
                            })
                            .unwrap_or_else(|| slice.iter().all(|value| *value == 0.0));
                        if exact_zero {
                            exact_zero_slice_channels.push(c);
                        }
                    }
                }
                let zero_slice_count = zero_slice_channels.len();
                if zero_slice_count > 0 {
                    r.zero_kernel_slice_tensors += 1;
                    r.zero_kernel_slice_count += zero_slice_count;
                    if !exact_zero_slice_channels.is_empty() {
                        r.exact_zero_kernel_slice_tensors += 1;
                        r.exact_zero_kernel_slice_count += exact_zero_slice_channels.len();
                    }
                    let consumers = kernel_consumers;
                    let consumer_macs: f64 = consumers.iter().map(|op| op.macs).sum();
                    let primary_consumer = consumers.first().copied();
                    let flagged_scale_sample: Vec<f32> = zero_slice_channels
                        .iter()
                        .take(8)
                        .filter_map(|&channel| {
                            t.scale_sample
                                .get(channel)
                                .copied()
                                .or_else(|| t.scale_sample.first().copied())
                        })
                        .collect();
                    let flagged_zero_point_sample: Vec<i64> = zero_slice_channels
                        .iter()
                        .take(8)
                        .filter_map(|&channel| {
                            t.zero_point_sample
                                .get(channel)
                                .copied()
                                .or_else(|| t.zero_point_sample.first().copied())
                        })
                        .collect();
                    let mut bias_tensor_index = -1i32;
                    let mut bias_tensor_name = String::new();
                    let mut bias_dtype = String::new();
                    let mut bias_value_sample = Vec::<f64>::new();
                    let mut bias_code_sample = Vec::<i32>::new();
                    let mut bias_int32_utilization_sample = Vec::<f64>::new();
                    let mut bias_nonzero_for_flagged_channels = false;
                    let mut fused_activation = "NOT_APPLICABLE".to_string();
                    let mut residual_path = "not observed in direct consumer edge".to_string();
                    let mut next_consumers = Vec::<String>::new();
                    if let Some(consumer) = primary_consumer {
                        fused_activation = consumer.fused_activation.clone();
                        let bias_slot = if consumer.name == "TRANSPOSE_CONV" {
                            3
                        } else {
                            2
                        };
                        if let Some(&bias_id) = consumer.inputs.get(bias_slot) {
                            bias_tensor_index = bias_id;
                            if bias_id >= 0 {
                                if let Some(bias_tensor) = tensors.get(bias_id as usize) {
                                    bias_tensor_name = bias_tensor.name.clone();
                                    bias_dtype = bias_tensor.dtype.clone();
                                    let input_scale = consumer
                                        .inputs
                                        .first()
                                        .and_then(|&id| {
                                            if id >= 0 {
                                                tensors.get(id as usize)
                                            } else {
                                                None
                                            }
                                        })
                                        .and_then(|input| input.scale_sample.first().copied())
                                        .unwrap_or(1.0)
                                        as f64;
                                    let bias_values =
                                        bias_to_f64(bytes, bias_tensor, input_scale, t);
                                    let bias_codes = int32_raw_values(bytes, bias_tensor);
                                    for &channel in zero_slice_channels.iter().take(8) {
                                        if let Some(&value) = bias_values.get(channel) {
                                            if value.abs() >= WEIGHT_NEAR_ZERO_EPS {
                                                bias_nonzero_for_flagged_channels = true;
                                            }
                                            bias_value_sample.push(value);
                                        }
                                        if let Some(code) = bias_codes
                                            .as_ref()
                                            .and_then(|values| values.get(channel))
                                            .copied()
                                        {
                                            bias_code_sample.push(code);
                                            bias_int32_utilization_sample.push(
                                                (f64::from(code).abs() / f64::from(i32::MAX))
                                                    .min(1.0),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        for &output_id in &consumer.outputs {
                            if output_id < 0 {
                                continue;
                            }
                            for next in ops
                                .iter()
                                .filter(|op| op.inputs.contains(&output_id))
                                .take(6)
                            {
                                if next.name == "ADD" {
                                    residual_path =
                                        format!("direct residual ADD consumer #{}", next.index);
                                }
                                next_consumers.push(format!("#{} {}", next.index, next.name));
                            }
                        }
                        next_consumers.sort();
                        next_consumers.dedup();
                        next_consumers.truncate(6);
                    }
                    r.zero_kernel_slice_details.push(ZeroKernelSliceDetail {
                        tensor_index: t.index,
                        tensor_name: t.name.clone(),
                        dtype: t.dtype.clone(),
                        shape: t.shape.clone(),
                        channels: zero_slice_channels.iter().copied().take(64).collect(),
                        channel_count: zero_slice_count,
                        exact_zero_channels: exact_zero_slice_channels
                            .iter()
                            .copied()
                            .take(64)
                            .collect(),
                        exact_zero_channel_count: exact_zero_slice_channels.len(),
                        scale_sample: flagged_scale_sample,
                        zero_point_sample: flagged_zero_point_sample,
                        bias_tensor_index,
                        bias_tensor_name,
                        bias_dtype,
                        bias_value_sample,
                        bias_code_sample,
                        bias_int32_utilization_sample,
                        bias_nonzero_for_flagged_channels,
                        fused_activation,
                        residual_path,
                        next_consumers,
                        functional_status: "NOT_ASSESSABLE: all-zero kernel slice only; bias, fused activation, residual/downstream behavior, and task outputs are not evaluated here".to_string(),
                        consumer_ops: consumers
                            .iter()
                            .take(4)
                            .map(|op| format!("#{} {}", op.index, op.name))
                            .collect(),
                        consumer_mac_percent: if total_macs > 0.0 {
                            consumer_macs / total_macs
                        } else {
                            0.0
                        },
                    });
                }
            }
        }
    }
    if r.weight_tensors_scanned > 0 {
        r.mean_sparsity = sparsity_acc / r.weight_tensors_scanned as f64;
    }
    if r.quantized_constant_tensors_scanned == 0 {
        r.min_grid_utilization = 0.0;
    }
    r.constant_value_coverage_status = if r.sparse_constant_tensors_not_decoded == 0 {
        "complete_for_supported_dense_and_sparse_storage".to_string()
    } else if r.weight_tensors_scanned == 0 {
        "not_assessed_sparse_storage".to_string()
    } else {
        "partial_unassessed_sparse_storage".to_string()
    };
    let risk = r.nan_tensors > 0 || r.inf_tensors > 0 || r.all_zero_tensors > 0;
    let warn = r.zero_kernel_slice_tensors > 0
        || r.large_magnitude_tensors > 0
        || r.low_grid_utilization_tensors > 0
        || r.saturated_quantized_tensors > 0;
    r.status = if risk {
        "risk".to_string()
    } else if warn {
        "warn".to_string()
    } else {
        "ok".to_string()
    };
    r.quant_grid_detail = format!(
        "Quantized constant grid scan: {} tensor(s) decoded; all-size minimum 8-bit level utilization {:.1}%; threshold-eligible minimum {}; {} tensor(s) with at least 256 elements assessed against the <25% threshold, of which {} are below it; max qmin/qmax saturation {:.2}%; {} tensor(s) above 1% endpoint saturation.",
        r.quantized_constant_tensors_scanned,
        r.min_grid_utilization * 100.0,
        r.min_threshold_eligible_grid_utilization
            .map(|value| format!("{:.1}%", value * 100.0))
            .unwrap_or_else(|| "N/A (no tensor has at least 256 elements)".to_string()),
        r.threshold_eligible_quantized_constant_tensors,
        r.low_grid_utilization_tensors,
        r.max_saturation_percent * 100.0,
        r.saturated_quantized_tensors,
    );
    r.detail = format!(
        "Scanned {} decodable constant tensor(s), including {} quantized tensor(s), {} logical elements. Sparse-storage constants decoded/not decoded: {}/{}; sparse logical/stored/implicit-zero elements: {}/{}/{} (coverage {}). Eligible kernel tensors {}, output channels evaluated {}. INT8/UINT8 constants are dequantized with artifact scale/zero-point metadata before magnitude checks. Near-zero criterion |x| < {:.0e}; high-sparsity criterion >50% near-zero elements; near-zero decoded slice means every decoded element in that output-channel slice is below the criterion; exact-zero stored slice separately requires every centered quantized code (or stored float value) to equal zero. NaN {} / Inf {} / all-near-zero {} tensors; kernel tensors containing near-zero decoded slices {} ({} slices total), including {} tensor(s) / {} slice(s) that are exact-zero in stored centered-code space; max |decoded constant| {:.3e}; {} tensor(s) with |decoded constant|>1e4; mean near-zero sparsity {:.1}% ({} tensor(s) >50%). {}",
        r.weight_tensors_scanned,
        r.quantized_constant_tensors_scanned,
        r.elements_scanned,
        r.sparse_constant_tensors_decoded,
        r.sparse_constant_tensors_not_decoded,
        r.sparse_logical_elements,
        r.sparse_stored_elements,
        r.sparse_implicit_zero_elements,
        r.constant_value_coverage_status,
        r.eligible_kernel_tensors_scanned,
        r.output_channels_evaluated,
        WEIGHT_NEAR_ZERO_EPS,
        r.nan_tensors,
        r.inf_tensors,
        r.all_zero_tensors,
        r.zero_kernel_slice_tensors,
        r.zero_kernel_slice_count,
        r.exact_zero_kernel_slice_tensors,
        r.exact_zero_kernel_slice_count,
        r.max_abs_weight,
        r.large_magnitude_tensors,
        r.mean_sparsity * 100.0,
        r.high_sparsity_tensors,
        r.quant_grid_detail,
    );
    r
}

fn quantized_raw_values(bytes: &[u8], tensor: &TensorInfo) -> Option<Vec<i32>> {
    if !matches!(tensor.dtype.as_str(), "INT8" | "UINT8") {
        return None;
    }
    let slice = extract_tensor_buffer(bytes, tensor)?;
    if tensor.dtype == "INT8" {
        Some(slice.iter().map(|&b| (b as i8) as i32).collect())
    } else {
        Some(slice.iter().map(|&b| b as i32).collect())
    }
}

fn int32_raw_values(bytes: &[u8], tensor: &TensorInfo) -> Option<Vec<i32>> {
    if tensor.dtype != "INT32" {
        return None;
    }
    let slice = extract_tensor_buffer(bytes, tensor)?;
    if !slice.len().is_multiple_of(4) {
        return None;
    }
    Some(
        slice
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

fn buffer_string(fb: &Fb, loc: BufferDataLocation) -> String {
    fb.data
        .get(loc.offset..loc.offset + loc.length)
        .map(|slice| {
            String::from_utf8_lossy(slice)
                .trim_matches('\0')
                .to_string()
        })
        .unwrap_or_default()
}

// ── Artifact size breakdown: where the bytes live ─────────────────────────────
fn compute_size_breakdown(
    fb: &Fb,
    model: usize,
    tensors: &[TensorInfo],
    buffer_locations: &[BufferDataLocation],
    file_size: usize,
) -> ArtifactSizeBreakdown {
    let mut constant_tensor_count = 0usize;
    let mut sparse_constant_tensor_count = 0usize;
    let mut logical_constant_reference_bytes = 0usize;
    let mut physical_buffers = BTreeMap::<(usize, usize), &TensorInfo>::new();
    for tensor in tensors {
        if !tensor.constant_buffer {
            continue;
        }
        constant_tensor_count += 1;
        sparse_constant_tensor_count += usize::from(tensor.sparse_storage);
        if tensor.buffer_data_length == 0 {
            continue;
        }
        logical_constant_reference_bytes =
            logical_constant_reference_bytes.saturating_add(tensor.buffer_data_length);
        physical_buffers
            .entry((tensor.buffer_data_offset, tensor.buffer_data_length))
            .or_insert(tensor);
    }
    let physical_constant_buffer_count = physical_buffers.len();
    let constant_bytes = physical_buffers
        .keys()
        .map(|(_, length)| *length)
        .sum::<usize>();
    let mut stored_scalar_elements = 0usize;
    let mut float_constant_bytes = 0usize;
    let mut zero_bytes = 0usize;
    let mut seen_content = HashMap::<(u64, usize), Vec<(usize, usize)>>::new();
    let mut unique_constant_bytes = 0usize;
    for (&(offset, length), tensor) in &physical_buffers {
        stored_scalar_elements += dtype_storage_bytes(&tensor.dtype)
            .map(|width| length / width)
            .unwrap_or(0);
        if tensor.dtype == "FLOAT32" || tensor.dtype == "FLOAT16" {
            float_constant_bytes += length;
        }
        let Some(slice) = fb.data.get(offset..offset + length) else {
            continue;
        };
        zero_bytes += slice.iter().filter(|&&byte| byte == 0).count();
        let key = (fnv1a64(slice), length);
        let representatives = seen_content.entry(key).or_default();
        let exact_duplicate = representatives
            .iter()
            .any(|(candidate_offset, candidate_length)| {
                fb.data
                    .get(*candidate_offset..*candidate_offset + *candidate_length)
                    .is_some_and(|candidate| candidate == slice)
            });
        if !exact_duplicate {
            unique_constant_bytes += length;
            representatives.push((offset, length));
        }
    }
    let duplicate_constant_bytes = constant_bytes.saturating_sub(unique_constant_bytes);
    let constant_locations = physical_buffers.keys().copied().collect::<HashSet<_>>();
    let mut metadata_locations = HashSet::<(usize, usize)>::new();
    for meta in fb.vector_tables(model, 6) {
        let buf_idx = fb
            .field_pos(meta, 1)
            .and_then(|pos| fb.u32(pos))
            .unwrap_or(0) as usize;
        if let Some(loc) = buffer_locations.get(buf_idx) {
            let location = (loc.offset, loc.length);
            if loc.length > 0 && !constant_locations.contains(&location) {
                metadata_locations.insert(location);
            }
        }
    }
    let metadata_bytes = metadata_locations
        .iter()
        .map(|(_, length)| *length)
        .sum::<usize>();
    let structure_overhead_bytes =
        file_size.saturating_sub(constant_bytes.saturating_add(metadata_bytes));
    // FP32 constants shrink 2x to FP16 and 4x to INT8 in the ideal case; FP16
    // constants can halve again to INT8. Non-float constants keep their size.
    let mut fp16 = 0usize;
    let mut int8 = 0usize;
    for (&(_, length), tensor) in &physical_buffers {
        match tensor.dtype.as_str() {
            "FLOAT32" => {
                fp16 += length / 2;
                int8 += length / 4;
            }
            "FLOAT16" => {
                fp16 += length;
                int8 += length / 2;
            }
            _ => {
                fp16 += length;
                int8 += length;
            }
        }
    }
    let zero_constant_byte_ratio = if constant_bytes > 0 {
        zero_bytes as f64 / constant_bytes as f64
    } else {
        0.0
    };
    let detail = format!(
        "File {} B = physical raw constant buffers {} B ({} unique locations referenced by {} tensors, {} logical reference bytes; {} sparse-storage tensor reference(s); {} stored scalar elements) + disjoint metadata-buffer payload {} B + residual bytes outside those payload classes {} B. The residual is not an ownership claim: FlatBuffer structure/alignment, terminal associated-file ZIP bytes, and unowned suffix bytes are separated by artifact_byte_integrity. Content-unique constant bytes {} B (separately stored exact duplicate payload {} B; shared tensor references are not duplicates). Raw 0x00 byte ratio in physical constant buffers {:.1}% (compressibility hint, not scalar sparsity).",
        file_size,
        constant_bytes,
        physical_constant_buffer_count,
        constant_tensor_count,
        logical_constant_reference_bytes,
        sparse_constant_tensor_count,
        stored_scalar_elements,
        metadata_bytes,
        structure_overhead_bytes,
        unique_constant_bytes,
        duplicate_constant_bytes,
        zero_constant_byte_ratio * 100.0,
    );
    ArtifactSizeBreakdown {
        file_size,
        constant_tensor_count,
        sparse_constant_tensor_count,
        physical_constant_buffer_count,
        logical_constant_reference_bytes,
        constant_bytes,
        stored_scalar_elements,
        unique_constant_bytes,
        duplicate_constant_bytes,
        metadata_bytes,
        structure_overhead_bytes,
        float_constant_bytes,
        theoretical_fp16_constant_bytes: fp16,
        theoretical_int8_constant_bytes: int8,
        zero_constant_byte_ratio,
        detail,
    }
}

// ── Runtime version compatibility: min_runtime_version + max op version ───────
fn compute_runtime_compat(
    fb: &Fb,
    model: usize,
    operator_code_tables: &[usize],
    buffer_locations: &[BufferDataLocation],
) -> Result<RuntimeCompat, String> {
    // min_runtime_version lives in the Model.metadata vector (field 6): an entry
    // whose name is "min_runtime_version" pointing at a buffer holding the string.
    let mut min_runtime_version = String::new();
    for meta in fb.vector_tables(model, 6) {
        let name = fb
            .checked_string_field(meta, 0, "Metadata.name")?
            .unwrap_or_default();
        if name == "min_runtime_version" {
            let buf_idx = fb.checked_u32_field(meta, 1, 0, "Metadata.buffer")? as usize;
            if let Some(loc) = buffer_locations.get(buf_idx) {
                min_runtime_version = buffer_string(fb, *loc);
            }
        }
    }
    let mut max_op_version = 0i32;
    let mut driving: Vec<String> = Vec::new();
    let mut derived_min_runtime_version = String::new();
    let mut derived_drivers: Vec<String> = Vec::new();
    let mut unmapped_versioned_ops: Vec<String> = Vec::new();
    let operator_code_count = operator_code_tables.len();
    let mut builtin_operator_code_count = 0usize;
    let mut mapped_operator_code_count = 0usize;
    let mut custom_operator_code_count = 0usize;
    for &oc in operator_code_tables {
        let version = fb
            .checked_i32_field(oc, 2, 1, "OperatorCode.version")?
            .max(1);
        let name = operator_code_name(fb, oc)?;
        if name.starts_with("CUSTOM") {
            custom_operator_code_count += 1;
        } else {
            builtin_operator_code_count += 1;
        }
        if let Some(required) = pinned_runtime_version_for_op(&name, version) {
            mapped_operator_code_count += 1;
            if derived_min_runtime_version.is_empty()
                || runtime_version_less(&derived_min_runtime_version, required)
            {
                derived_min_runtime_version = required.to_string();
                derived_drivers = vec![format!("{name} v{version}")];
            } else if derived_min_runtime_version == required {
                derived_drivers.push(format!("{name} v{version}"));
            }
        } else if !name.starts_with("CUSTOM") {
            unmapped_versioned_ops.push(format!("{name} v{version}"));
        }
        if version > max_op_version {
            max_op_version = version;
            driving = vec![name];
        } else if version == max_op_version {
            driving.push(name);
        }
    }
    driving.sort();
    driving.dedup();
    derived_drivers.sort();
    derived_drivers.dedup();
    unmapped_versioned_ops.sort();
    unmapped_versioned_ops.dedup();
    let candidate_effective_min_runtime_version = if min_runtime_version.is_empty() {
        derived_min_runtime_version.clone()
    } else if derived_min_runtime_version.is_empty()
        || runtime_version_less(&derived_min_runtime_version, &min_runtime_version)
    {
        min_runtime_version.clone()
    } else {
        derived_min_runtime_version.clone()
    };
    let runtime_floor_status = if unmapped_versioned_ops.is_empty() {
        "complete_for_observed_builtin_op_versions".to_string()
    } else {
        "partial_unmapped_builtin_op_versions".to_string()
    };
    let runtime_floor_evidence_class = if derived_min_runtime_version.is_empty() {
        "NOT_ASSESSABLE".to_string()
    } else {
        "DERIVED_NECESSARY_MINIMUM".to_string()
    };
    let effective_min_runtime_version = if unmapped_versioned_ops.is_empty() {
        candidate_effective_min_runtime_version
    } else {
        String::new()
    };
    let runtime_version_basis = if derived_min_runtime_version.is_empty() {
        if unmapped_versioned_ops.is_empty() {
            "No mapped builtin op/version runtime floor was found.".to_string()
        } else {
            format!(
                "Pinned TensorFlow runtime-version map did not cover: {}.",
                unmapped_versioned_ops.join(", ")
            )
        }
    } else if unmapped_versioned_ops.is_empty() {
        format!(
            "Necessary floor derived from the pinned TensorFlow runtime-version map with complete observed builtin op-code coverage ({}/{}); driver(s): {}. This is not an execution-compatibility guarantee.",
            mapped_operator_code_count,
            builtin_operator_code_count,
            derived_drivers.join(", "),
        )
    } else {
        format!(
            "Partial necessary floor derived from the pinned TensorFlow runtime-version map for {}/{} observed builtin op codes; driver(s): {}; unmapped: {}. No effective floor is emitted.",
            mapped_operator_code_count,
            builtin_operator_code_count,
            derived_drivers.join(", "),
            unmapped_versioned_ops.join(", "),
        )
    };
    let detail = format!(
        "{}{}Highest operator version in graph: {} (driven by {}). Builtin op-version map coverage: {}/{}; custom op codes: {}. Effective minimum runtime floor: {}.",
        if min_runtime_version.is_empty() {
            "No min_runtime_version metadata embedded. ".to_string()
        } else {
            format!("Declared min_runtime_version: {min_runtime_version}. ")
        },
        if derived_min_runtime_version.is_empty() {
            "No derived runtime floor available from mapped builtin op versions. ".to_string()
        } else {
            format!("Derived min_runtime_version: {derived_min_runtime_version}. ")
        },
        max_op_version,
        if driving.is_empty() { "n/a".to_string() } else { driving.join(", ") },
        mapped_operator_code_count,
        builtin_operator_code_count,
        custom_operator_code_count,
        if effective_min_runtime_version.is_empty() {
            if derived_min_runtime_version.is_empty() {
                "not determined".to_string()
            } else {
                format!(
                    "not emitted; mapped-op necessary floor is {} but map coverage is partial",
                    derived_min_runtime_version
                )
            }
        } else {
            effective_min_runtime_version.clone()
        },
    );
    Ok(RuntimeCompat {
        min_runtime_version,
        derived_min_runtime_version,
        effective_min_runtime_version,
        runtime_floor_status,
        runtime_floor_evidence_class,
        operator_code_count,
        builtin_operator_code_count,
        mapped_operator_code_count,
        custom_operator_code_count,
        runtime_version_basis,
        unmapped_versioned_ops,
        max_op_version,
        version_driving_ops: driving,
        detail,
    })
}

fn runtime_version_less(left: &str, right: &str) -> bool {
    let mut left_parts = left.split('.').map(|part| part.parse::<i32>().unwrap_or(0));
    let mut right_parts = right
        .split('.')
        .map(|part| part.parse::<i32>().unwrap_or(0));
    for _ in 0..4 {
        let a = left_parts.next().unwrap_or(0);
        let b = right_parts.next().unwrap_or(0);
        if a != b {
            return a < b;
        }
    }
    false
}

// ── Artifact metadata / signature presence checklist ──────────────────────────
const CONVERSION_METADATA_SCHEMA_SOURCE_FILE: &str =
    "tensorflow/compiler/mlir/lite/schema/conversion_metadata.fbs";
const CONVERSION_METADATA_SCHEMA_SHA256: &str =
    "2464449e30bfa6032c0218b53a1a83b224c6eda9b5cfd9f12211c4c0017dc20e";

fn compute_metadata_presence(
    fb: &Fb,
    model: usize,
    _subgraph: usize,
    buffer_locations: &[BufferDataLocation],
    outputs: &[TensorInfo],
    packed_metadata_archive: tflite_metadata::PackedMetadataArchive,
) -> ArtifactMetadataPresence {
    let signatures = fb.vector_tables(model, 7); // SignatureDef vector
    let signature_count = signatures.len();
    let signature_keys: Vec<String> = signatures
        .iter()
        .filter_map(|&sig| fb.string_field(sig, 2))
        .filter(|key| !key.is_empty())
        .collect();
    let description = fb.string_field(model, 3).unwrap_or_default();
    let mut metadata_entries: Vec<String> = Vec::new();
    let mut has_tflite_metadata = false;
    let mut model_metadata_entry_count = 0usize;
    let mut conversion_metadata_entry_count = 0usize;
    let mut conversion_metadata = ParsedConversionMetadata {
        status: "not_present".to_string(),
        ..ParsedConversionMetadata::default()
    };
    let mut parsed_metadata = ParsedTfliteModelMetadata::unavailable(
        "not_present",
        "No TFLITE_METADATA entry is present in Model.metadata.",
    );
    for meta in fb.vector_tables(model, 6) {
        let name = fb.string_field(meta, 0).unwrap_or_default();
        if !name.is_empty() {
            if name == "TFLITE_METADATA" {
                has_tflite_metadata = true;
                model_metadata_entry_count += 1;
                let buffer_index = fb
                    .field_pos(meta, 1)
                    .and_then(|position| fb.u32(position))
                    .map(|value| value as usize);
                parsed_metadata = match buffer_index.and_then(|index| buffer_locations.get(index)) {
                    Some(location) => fb
                        .data
                        .get(location.offset..location.offset.saturating_add(location.length))
                        .map(parse_tflite_model_metadata)
                        .unwrap_or_else(|| {
                            ParsedTfliteModelMetadata::unavailable(
                                "metadata_buffer_out_of_bounds",
                                "TFLITE_METADATA references bytes outside the artifact.",
                            )
                        }),
                    None => ParsedTfliteModelMetadata::unavailable(
                        "metadata_buffer_reference_missing",
                        "TFLITE_METADATA does not resolve to a Model.buffers entry.",
                    ),
                };
            }
            if name == "CONVERSION_METADATA" {
                conversion_metadata_entry_count += 1;
                let buffer_index = fb
                    .field_pos(meta, 1)
                    .and_then(|position| fb.u32(position))
                    .map(|value| value as usize);
                conversion_metadata =
                    match buffer_index.and_then(|index| buffer_locations.get(index)) {
                        Some(location) => fb
                            .data
                            .get(location.offset..location.offset.saturating_add(location.length))
                            .map(parse_conversion_metadata)
                            .unwrap_or_else(|| ParsedConversionMetadata {
                                status: "metadata_buffer_out_of_bounds".to_string(),
                                ..ParsedConversionMetadata::default()
                            }),
                        None => ParsedConversionMetadata {
                            status: "metadata_buffer_reference_missing".to_string(),
                            ..ParsedConversionMetadata::default()
                        },
                    };
            }
            metadata_entries.push(name);
        }
    }
    if conversion_metadata_entry_count > 1 {
        conversion_metadata = ParsedConversionMetadata {
            status: "invalid_duplicate_entries".to_string(),
            ..ParsedConversionMetadata::default()
        };
    }
    let output_shapes = outputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let associated_binding = bind_packed_associated_files(
        &mut parsed_metadata,
        packed_metadata_archive,
        &output_shapes,
    );
    let input_process_unit_count = parsed_metadata.input_process_units.len();
    let preprocessing_contract_status = if !has_tflite_metadata {
        "absent_no_model_metadata"
    } else if parsed_metadata.status != "parsed" {
        "not_assessed_metadata_parse_failure"
    } else if input_process_unit_count == 0 {
        "absent_no_explicit_input_process_units"
    } else if parsed_metadata.recognized_input_process_unit_count == input_process_unit_count
        && parsed_metadata.invalid_input_process_unit_count == 0
        && parsed_metadata.unrecognized_input_process_unit_count == 0
    {
        "assessed_explicit_input_process_units"
    } else {
        "partial_invalid_or_unsupported_input_process_units"
    }
    .to_string();
    let documented_preprocessing =
        preprocessing_contract_status == "assessed_explicit_input_process_units";
    let output_semantics_documented = parsed_metadata.status == "parsed"
        && associated_binding.verified_output0_label_file_count > 0;
    let detail = format!(
        "Signature defs: {}. Model metadata entries: {}. TFLITE_METADATA parse status: {}. CONVERSION_METADATA: {} (TensorFlow {}, API {}, source model {}, optimization modes {}). Description: {}. Preprocessing contract status: {}. Output label mapping declarations/semantically verified: {}/{}. Packed payloads verified/invalid/unsupported: {}/{}/{}. Associated-file archive: {}. {} {}",
        if signature_count > 0 { format!("{signature_count} present") } else { "none".to_string() },
        if metadata_entries.is_empty() { "none".to_string() } else { metadata_entries.join(", ") },
        parsed_metadata.status,
        conversion_metadata.status,
        if conversion_metadata.tensorflow_version.is_empty() { "not declared".to_string() } else { conversion_metadata.tensorflow_version.clone() },
        conversion_metadata.api_version.map(|value| value.to_string()).unwrap_or_else(|| "not declared".to_string()),
        if conversion_metadata.model_type.is_empty() { "not declared".to_string() } else { conversion_metadata.model_type.clone() },
        if conversion_metadata.optimization_modes.is_empty() {
            "not declared".to_string()
        } else {
            conversion_metadata.optimization_modes.join(", ")
        },
        if description.is_empty() { "none".to_string() } else { description.clone() },
        preprocessing_contract_status,
        parsed_metadata.output_label_file_count,
        associated_binding.verified_output_label_file_count,
        associated_binding.payload_verified_file_count,
        associated_binding.payload_invalid_file_count,
        associated_binding.payload_unsupported_file_count,
        associated_binding.archive_status,
        parsed_metadata.detail,
        associated_binding.archive_detail,
    );
    ArtifactMetadataPresence {
        format: "tflite".to_string(),
        schema: "deepbom.artifact_metadata.v1.4".to_string(),
        status: if has_tflite_metadata {
            parsed_metadata.status.clone()
        } else {
            "assessed_no_model_metadata".to_string()
        },
        has_signature_defs: signature_count > 0,
        signature_count,
        signature_keys,
        has_model_metadata: has_tflite_metadata,
        metadata_entries,
        conversion_metadata_entry_count,
        conversion_metadata_status: conversion_metadata.status,
        converter_tensorflow_version: conversion_metadata.tensorflow_version,
        converter_api_version: conversion_metadata.api_version,
        converter_model_type: conversion_metadata.model_type,
        converter_optimization_mode_codes: conversion_metadata.optimization_mode_codes,
        converter_optimization_modes: conversion_metadata.optimization_modes,
        conversion_metadata_schema_source_commit: TFLITE_ARENA_SOURCE_COMMIT.to_string(),
        conversion_metadata_schema_source_file: CONVERSION_METADATA_SCHEMA_SOURCE_FILE.to_string(),
        conversion_metadata_schema_sha256: CONVERSION_METADATA_SCHEMA_SHA256.to_string(),
        has_description: !description.is_empty(),
        description,
        documented_preprocessing,
        preprocessing_contract_status,
        output_semantics_documented,
        metadata_schema_identifier: parsed_metadata.schema_identifier,
        metadata_min_parser_version: parsed_metadata.min_parser_version,
        metadata_model_name: parsed_metadata.model_name,
        metadata_model_description: parsed_metadata.model_description,
        metadata_model_version: parsed_metadata.model_version,
        metadata_author: parsed_metadata.author,
        metadata_license: parsed_metadata.license,
        model_metadata_entry_count,
        subgraph_metadata_count: parsed_metadata.subgraph_metadata_count,
        input_tensor_metadata_count: parsed_metadata.input_tensor_metadata_count,
        output_tensor_metadata_count: parsed_metadata.output_tensor_metadata_count,
        described_input_tensor_count: parsed_metadata.described_input_tensor_count,
        described_output_tensor_count: parsed_metadata.described_output_tensor_count,
        input_process_unit_count,
        recognized_input_process_unit_count: parsed_metadata.recognized_input_process_unit_count,
        invalid_input_process_unit_count: parsed_metadata.invalid_input_process_unit_count,
        unrecognized_input_process_unit_count: parsed_metadata
            .unrecognized_input_process_unit_count,
        normalization_unit_count: parsed_metadata.normalization_unit_count,
        input_process_units: parsed_metadata.input_process_units,
        output_associated_file_count: parsed_metadata.output_associated_files.len(),
        output_label_file_count: parsed_metadata.output_label_file_count,
        verified_output_associated_file_count: associated_binding
            .verified_output_associated_file_count,
        missing_output_associated_file_count: associated_binding
            .missing_output_associated_file_count,
        verified_output_label_file_count: associated_binding.verified_output_label_file_count,
        missing_output_label_file_count: associated_binding.missing_output_label_file_count,
        invalid_output_label_file_count: associated_binding.invalid_output_label_file_count,
        verified_output0_label_file_count: associated_binding.verified_output0_label_file_count,
        payload_verified_file_count: associated_binding.payload_verified_file_count,
        payload_invalid_file_count: associated_binding.payload_invalid_file_count,
        payload_unsupported_file_count: associated_binding.payload_unsupported_file_count,
        label_cardinality_match_count: associated_binding.label_cardinality_match_count,
        label_cardinality_mismatch_count: associated_binding.label_cardinality_mismatch_count,
        label_cardinality_ambiguous_count: associated_binding.label_cardinality_ambiguous_count,
        label_cardinality_unresolved_count: associated_binding.label_cardinality_unresolved_count,
        output_associated_files: parsed_metadata.output_associated_files,
        associated_file_archive_status: associated_binding.archive_status,
        associated_file_archive_detail: associated_binding.archive_detail,
        packed_associated_file_count: associated_binding.packed_files.len(),
        packed_associated_files: associated_binding.packed_files,
        detail,
    }
}

fn read_buffer_locations(fb: &Fb, model: usize) -> Vec<BufferDataLocation> {
    fb.vector_tables(model, 4)
        .iter()
        .map(|table| read_buffer_location(fb, *table))
        .collect()
}

fn read_buffer_location(fb: &Fb, table: usize) -> BufferDataLocation {
    let inline = fb
        .vector_location(table, 0, 1)
        .map(|(offset, length)| BufferDataLocation { offset, length })
        .filter(|location| location.length > 0);
    inline
        .or_else(|| {
            let offset = fb.field_pos(table, 1).and_then(|pos| fb.u64(pos))?;
            let size = fb.field_pos(table, 2).and_then(|pos| fb.u64(pos))?;
            bounded_buffer_location(offset, size, fb.data.len())
        })
        .unwrap_or_default()
}

fn bounded_buffer_location(
    offset: u64,
    size: u64,
    file_length: usize,
) -> Option<BufferDataLocation> {
    if offset <= 1 || size == 0 {
        return None;
    }
    let offset = usize::try_from(offset).ok()?;
    let length = usize::try_from(size).ok()?;
    let end = offset.checked_add(length)?;
    if end > file_length {
        return None;
    }
    Some(BufferDataLocation { offset, length })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn read_tensor(
    fb: &Fb,
    index: usize,
    table: usize,
    buffer_locations: &[BufferDataLocation],
) -> Result<TensorInfo, String> {
    let shape = fb.vector_i32(table, 0);
    let shape_signature = fb.vector_i32(table, 7);
    let dtype_code = fb.checked_i8_field(table, 1, 0, "Tensor.type")?;
    let dtype_name = tensor_type_name(dtype_code).to_string();
    let name = fb
        .checked_string_field(table, 3, "Tensor.name")?
        .unwrap_or_default();
    let buffer_index = fb.checked_u32_field(table, 2, 0, "Tensor.buffer")? as i32;
    let buffer_location = buffer_index
        .try_into()
        .ok()
        .and_then(|idx: usize| buffer_locations.get(idx).copied())
        .unwrap_or_default();
    let sparsity_table = fb.checked_table_field(table, 6, "Tensor.sparsity")?;
    let sparse_storage = sparsity_table.is_some();
    let sparse_encoding = sparsity_table
        .map(|sparsity| {
            parse_sparsity(
                fb,
                sparsity,
                &shape,
                dtype_storage_bytes(&dtype_name),
                (buffer_index > 0).then_some(buffer_location.length),
            )
        })
        .transpose()?;
    let quant_table = fb.checked_table_field(table, 4, "Tensor.quantization")?;
    let scales = quant_table.map(|q| fb.vector_f32(q, 2)).unwrap_or_default();
    let zero_points = quant_table.map(|q| fb.vector_i64(q, 3)).unwrap_or_default();
    let quantized_dimension = match quant_table {
        Some(quant) => {
            fb.checked_i32_field(quant, 6, 0, "QuantizationParameters.quantized_dimension")?
        }
        None => 0,
    };
    let positive_scales = scales
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    let scale_min = positive_scales
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let scale_max = positive_scales.iter().copied().fold(0.0_f32, f32::max);
    let scale_min = if scale_min.is_finite() {
        scale_min
    } else {
        0.0
    };
    let scale_ratio_meaningful = positive_scales.len() > 1;
    let scale_ratio = if scale_ratio_meaningful && scale_min > 0.0 && scale_max > 0.0 {
        scale_max as f64 / scale_min as f64
    } else {
        0.0
    };
    let scale_mode = if positive_scales.len() > 1 {
        "per-axis"
    } else if positive_scales.len() == 1 {
        "per-tensor"
    } else {
        "none"
    }
    .to_string();
    let scale_mean = if positive_scales.is_empty() {
        0.0
    } else {
        positive_scales
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / positive_scales.len() as f64
    };
    let scale_stddev = if positive_scales.len() <= 1 || scale_mean <= 0.0 {
        0.0
    } else {
        let variance = positive_scales
            .iter()
            .map(|value| {
                let delta = *value as f64 - scale_mean;
                delta * delta
            })
            .sum::<f64>()
            / positive_scales.len() as f64;
        variance.sqrt()
    };
    let scale_cv = if scale_mean > 0.0 {
        scale_stddev / scale_mean
    } else {
        0.0
    };
    let zero_point_min = zero_points.iter().copied().min().unwrap_or(0);
    let zero_point_max = zero_points.iter().copied().max().unwrap_or(0);
    let zero_point = zero_point_summary(&zero_points, &dtype_name);
    let quant_risk = quant_risk_label(scale_ratio).to_string();
    let is_variable = fb.checked_i8_field(table, 5, 0, "Tensor.is_variable")? != 0;
    let buffer_hash = if buffer_location.length > 0 {
        fb.data
            .get(buffer_location.offset..buffer_location.offset + buffer_location.length)
            .map(|slice| format!("{:016x}", fnv1a64(slice)))
            .unwrap_or_default()
    } else {
        String::new()
    };
    Ok(TensorInfo {
        index,
        name,
        shape,
        shape_signature,
        dtype: dtype_name,
        buffer_index,
        buffer_data_offset: buffer_location.offset,
        buffer_data_length: buffer_location.length,
        constant_buffer: buffer_index > 0 && (buffer_location.length > 0 || sparse_storage),
        sparse_storage,
        sparse_encoding,
        quant_scales: scales.len(),
        quant_zero_points: zero_points.len(),
        quantized_dimension,
        scale_sample: scales.into_iter().collect(),
        zero_point_sample: zero_points.into_iter().collect(),
        scale_min,
        scale_max,
        scale_ratio,
        scale_mean,
        scale_stddev,
        scale_cv,
        zero_point_min,
        zero_point_max,
        zero_point_offset_max: zero_point.offset_max,
        zero_point_status: zero_point.status,
        zero_point_detail: zero_point.detail,
        scale_mode,
        scale_ratio_meaningful,
        quant_risk,
        buffer_hash,
        is_variable,
    })
}

struct ZeroPointSummary {
    offset_max: i64,
    status: String,
    detail: String,
}

fn zero_point_summary(zero_points: &[i64], dtype: &str) -> ZeroPointSummary {
    if zero_points.is_empty() {
        return ZeroPointSummary {
            offset_max: 0,
            status: "none".to_string(),
            detail: "No zero-point metadata".to_string(),
        };
    }
    let mut offset_max = 0_i64;
    let mut has_out_of_range = false;
    let mut has_reinterpret = false;
    for zp in zero_points {
        let (offset, status) = zero_point_offset_and_status(*zp, dtype);
        offset_max = offset_max.max(offset);
        if status == "out-of-range" {
            has_out_of_range = true;
        } else if status == "reinterpret" {
            has_reinterpret = true;
        }
    }
    let status = if has_out_of_range {
        "out-of-range"
    } else if has_reinterpret {
        "reinterpret"
    } else if offset_max >= 80 {
        "asymmetric"
    } else {
        "ok"
    };
    let detail = match status {
        "reinterpret" => "INT8 zero_point=128 is outside the signed range; treated as a possible legacy UINT8->INT8 reinterpretation sentinel rather than an automatic overflow risk. Verify converter/runtime quantization contract.".to_string(),
        "out-of-range" => format!("{} zero_point metadata is outside the legal dtype range; verify FlatBuffer quantization contract.", dtype),
        "asymmetric" => format!("zero_point offset {} is large enough to suggest strongly asymmetric activation distribution or clipping pressure.", offset_max),
        _ => format!("zero_point metadata is within expected {} range.", dtype),
    };
    ZeroPointSummary {
        offset_max,
        status: status.to_string(),
        detail,
    }
}

fn zero_point_offset_and_status(zero_point: i64, dtype: &str) -> (i64, &'static str) {
    match dtype {
        "UINT8" => {
            if !(0..=255).contains(&zero_point) {
                ((zero_point - 128).abs(), "out-of-range")
            } else {
                ((zero_point - 128).abs(), "ok")
            }
        }
        "INT8" => {
            if (-128..=127).contains(&zero_point) {
                (zero_point.abs(), "ok")
            } else if zero_point == 128 {
                (0, "reinterpret")
            } else {
                (zero_point.abs(), "out-of-range")
            }
        }
        _ => (zero_point.abs(), "ok"),
    }
}

fn positive_dim(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}

fn dtype_storage_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "FLOAT64" | "INT64" => Some(8),
        "FLOAT32" | "INT32" | "UINT32" => Some(4),
        "FLOAT16" | "BFLOAT16" | "INT16" | "UINT16" => Some(2),
        "INT8" | "UINT8" | "BOOL" => Some(1),
        _ => None,
    }
}

fn checked_product_options(values: &[Option<usize>]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul((*value)?))
}

fn logical_cache_payload_for_op(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
    dilation_h: usize,
) -> CachePayloadBreakdown {
    let data_input_slot = if name == "TRANSPOSE_CONV" { 2 } else { 0 };
    let weight_input_slot = 1;
    let Some(input) = inputs
        .get(data_input_slot)
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index))
    else {
        return CachePayloadBreakdown::not_applicable(name);
    };
    let Some(output) = outputs
        .first()
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index))
    else {
        return CachePayloadBreakdown::not_applicable(name);
    };
    let input_bytes = dtype_storage_bytes(&input.dtype);
    let output_bytes = dtype_storage_bytes(&output.dtype);
    let weight = inputs
        .get(weight_input_slot)
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index));
    let bias_slot = if name == "TRANSPOSE_CONV" { 3 } else { 2 };
    let bias = inputs
        .get(bias_slot)
        .and_then(|index| usize::try_from(*index).ok())
        .and_then(|index| tensors.get(index))
        .filter(|tensor| weight.map(|item| item.index) != Some(tensor.index));
    let serialized_kernel_bytes = weight
        .filter(|tensor| tensor.constant_buffer)
        .map(|tensor| tensor.buffer_data_length);
    let serialized_bias_bytes = bias
        .filter(|tensor| tensor.constant_buffer)
        .map(|tensor| tensor.buffer_data_length);

    let (input_width, input_channels, output_width, output_channels, kernel_height, kernel_width) =
        match name {
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV" => {
                let Some(weight) = weight else {
                    return CachePayloadBreakdown::not_applicable(name);
                };
                if input.shape.len() != 4 || output.shape.len() != 4 || weight.shape.len() != 4 {
                    return CachePayloadBreakdown::not_applicable(name);
                }
                (
                    positive_dim(input.shape[2]),
                    positive_dim(input.shape[3]),
                    positive_dim(output.shape[2]),
                    positive_dim(output.shape[3]),
                    positive_dim(weight.shape[1]),
                    positive_dim(weight.shape[2]),
                )
            }
            "FULLY_CONNECTED" | "BATCH_MATMUL" => (
                input.shape.last().copied().and_then(positive_dim),
                Some(1),
                output.shape.last().copied().and_then(positive_dim),
                Some(1),
                Some(1),
                Some(1),
            ),
            _ => return CachePayloadBreakdown::not_applicable(name),
        };
    let effective_kernel_height = kernel_height.and_then(|kernel| {
        kernel
            .saturating_sub(1)
            .checked_mul(dilation_h.max(1))
            .and_then(|value| value.checked_add(1))
    });
    let input_strip_bytes = match name {
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV" => checked_product_options(&[
            effective_kernel_height,
            input_width,
            input_channels,
            input_bytes,
        ]),
        "FULLY_CONNECTED" | "BATCH_MATMUL" => checked_product_options(&[input_width, input_bytes]),
        _ => None,
    };
    let output_row_bytes = match name {
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV" => {
            checked_product_options(&[output_width, output_channels, output_bytes])
        }
        "FULLY_CONNECTED" | "BATCH_MATMUL" => {
            checked_product_options(&[output_width, output_bytes])
        }
        _ => None,
    };
    let logical_row_payload_bytes = input_strip_bytes
        .zip(output_row_bytes)
        .and_then(|(input, output)| input.checked_add(output));
    let status = if logical_row_payload_bytes.is_some() {
        "assessed"
    } else {
        "not_assessed"
    };
    CachePayloadBreakdown {
        schema: "deepbom.cache_payload.v1".to_string(),
        status: status.to_string(),
        evidence_class: if status == "assessed" {
            "DERIVED"
        } else {
            "NOT_ASSESSABLE"
        }
        .to_string(),
        input_strip_bytes,
        output_row_bytes,
        logical_row_payload_bytes,
        serialized_kernel_bytes,
        serialized_bias_bytes,
        input_width,
        input_channels,
        output_width,
        output_channels,
        kernel_height,
        kernel_width,
        effective_kernel_height,
        input_dtype: Some(input.dtype.clone()),
        output_dtype: Some(output.dtype.clone()),
        method: "Logical row payload = effective-kernel-height input strip + one output row. Serialized kernel and bias payloads are reported separately and are not assumed simultaneously resident.".to_string(),
        interpretation_boundary: "A deterministic artifact-derived logical payload, not an executed microkernel tile, cache-line footprint, cache hit rate, residency proof, or measured DRAM traffic.".to_string(),
    }
}

fn estimate_op(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> (f64, f64, f64, f64) {
    let tensor_at = |indices: &[i32], slot: usize| {
        indices
            .get(slot)
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| tensors.get(index))
    };
    let input_bytes = inputs
        .iter()
        .filter_map(|idx| tensors.get(*idx as usize))
        .map(tensor_bytes)
        .sum::<f64>();
    let output_bytes = outputs
        .iter()
        .filter_map(|idx| tensors.get(*idx as usize))
        .map(tensor_bytes)
        .sum::<f64>();
    let estimated_bytes = input_bytes + output_bytes;
    let mut macs = 0.0;
    let mut row_ws = 0.0;

    if name == "CONV_2D" && inputs.len() >= 2 && !outputs.is_empty() {
        if let (Some(inp), Some(weight), Some(out)) = (
            tensors.get(inputs[0] as usize),
            tensors.get(inputs[1] as usize),
            tensors.get(outputs[0] as usize),
        ) {
            if inp.shape.len() == 4 && weight.shape.len() == 4 && out.shape.len() == 4 {
                let batch = out.shape[0] as f64;
                let out_h = out.shape[1] as f64;
                let out_w = out.shape[2] as f64;
                let out_c = weight.shape[0] as f64;
                let kernel_h = weight.shape[1] as f64;
                let kernel_w = weight.shape[2] as f64;
                let in_c = weight.shape[3] as f64;
                macs = batch * out_h * out_w * out_c * kernel_h * kernel_w * in_c;
                row_ws = kernel_h
                    * inp.shape[2] as f64
                    * inp.shape[3] as f64
                    * bytes_per_type(&inp.dtype);
            }
        }
    } else if name == "DEPTHWISE_CONV_2D" && inputs.len() >= 2 && !outputs.is_empty() {
        if let (Some(inp), Some(weight), Some(out)) = (
            tensors.get(inputs[0] as usize),
            tensors.get(inputs[1] as usize),
            tensors.get(outputs[0] as usize),
        ) {
            if inp.shape.len() == 4 && weight.shape.len() == 4 && out.shape.len() == 4 {
                let batch = out.shape[0] as f64;
                let out_h = out.shape[1] as f64;
                let out_w = out.shape[2] as f64;
                let out_c = out.shape[3] as f64;
                let kernel_h = weight.shape[1] as f64;
                let kernel_w = weight.shape[2] as f64;
                macs = batch * out_h * out_w * out_c * kernel_h * kernel_w;
                row_ws = kernel_h
                    * inp.shape[2] as f64
                    * inp.shape[3] as f64
                    * bytes_per_type(&inp.dtype);
            }
        }
    } else if name == "FULLY_CONNECTED" && inputs.len() >= 2 && !outputs.is_empty() {
        if let (Some(weight), Some(out)) = (
            tensors.get(inputs[1] as usize),
            tensors.get(outputs[0] as usize),
        ) {
            let in_units = weight.shape.last().copied().unwrap_or(0) as f64;
            macs = element_count(&out.shape) * in_units;
            // Weight row working set: each output neuron loads one full input vector.
            // Weight shape is [out_units, in_units]; row_ws = in_units × bytes.
            row_ws = in_units * bytes_per_type(&weight.dtype);
        }
    } else if name == "BATCH_MATMUL" && inputs.len() >= 2 && !outputs.is_empty() {
        if let (Some(lhs), Some(rhs), Some(out)) = (
            tensors.get(inputs[0] as usize),
            tensors.get(inputs[1] as usize),
            tensors.get(outputs[0] as usize),
        ) {
            // [batch, M, K] × [batch, K, N] → [batch, M, N]
            if lhs.shape.len() >= 2 && rhs.shape.len() >= 2 && out.shape.len() >= 2 {
                let n = out.shape.last().copied().unwrap_or(0) as f64;
                let k = rhs.shape[rhs.shape.len().saturating_sub(2)] as f64;
                let batch_m: f64 = out.shape[..out.shape.len() - 1]
                    .iter()
                    .fold(1.0, |a, d| a * *d as f64);
                macs = batch_m * n * k;
                row_ws = k * bytes_per_type(&lhs.dtype);
            }
        }
    } else if name == "TRANSPOSE_CONV" && inputs.len() >= 3 && !outputs.is_empty() {
        // inputs: [output_shape_tensor, filter, input_activation]
        // filter shape: [out_C, kernel_H, kernel_W, in_C] (OHWI)
        if let (Some(filter), Some(inp), Some(out)) = (
            tensors.get(inputs[1] as usize),
            tensors.get(inputs[2] as usize),
            tensors.get(outputs[0] as usize),
        ) {
            if filter.shape.len() == 4 && out.shape.len() == 4 {
                let batch = out.shape[0] as f64;
                let out_h = out.shape[1] as f64;
                let out_w = out.shape[2] as f64;
                let out_c = filter.shape[0] as f64;
                let kernel_h = filter.shape[1] as f64;
                let kernel_w = filter.shape[2] as f64;
                let in_c = filter.shape[3] as f64;
                macs = batch * out_h * out_w * out_c * kernel_h * kernel_w * in_c;
                row_ws = kernel_h
                    * inp.shape.get(2).copied().unwrap_or(0) as f64
                    * inp.shape.get(3).copied().unwrap_or(0) as f64
                    * bytes_per_type(&inp.dtype);
            }
        }
    }

    // ── Non-GEMM ops: ops > 0, macs = 0 ─────────────────────────────────────
    let mut extra_ops = 0.0_f64;

    if macs == 0.0 {
        let out_elems = || {
            outputs
                .first()
                .and_then(|idx| tensors.get(*idx as usize))
                .map(|t| element_count(&t.shape))
                .unwrap_or(0.0)
        };
        let in_elems = || {
            inputs
                .first()
                .and_then(|idx| tensors.get(*idx as usize))
                .map(|t| element_count(&t.shape))
                .unwrap_or(0.0)
        };

        extra_ops = match name {
            // Convolutions (GEMM-like, non-2D)
            "CONV_3D" | "CONV_3D_TRANSPOSE" => {
                // Filters are DHWIO for Conv3D and DHWOI for Conv3DTranspose.
                let transposed = name == "CONV_3D_TRANSPOSE";
                if let (Some(inp), Some(filt), Some(out)) = (
                    tensor_at(inputs, if transposed { 2 } else { 0 }),
                    tensor_at(inputs, 1),
                    tensor_at(outputs, 0),
                ) {
                    if filt.shape.len() == 5 && out.shape.len() == 5 {
                        let (n, od, oh, ow) = (
                            out.shape[0] as f64,
                            out.shape[1] as f64,
                            out.shape[2] as f64,
                            out.shape[3] as f64,
                        );
                        let (kd, kh, kw, channel_3, channel_4) = (
                            filt.shape[0] as f64,
                            filt.shape[1] as f64,
                            filt.shape[2] as f64,
                            filt.shape[3] as f64,
                            filt.shape[4] as f64,
                        );
                        let (ic, oc) = if transposed {
                            (channel_4, channel_3)
                        } else {
                            (channel_3, channel_4)
                        };
                        // Store in macs for 2× ops convention
                        macs = n * od * oh * ow * oc * kd * kh * kw * ic;
                        if inp.shape.len() == 5 {
                            row_ws = kd
                                * kh
                                * inp.shape[3] as f64
                                * inp.shape[4] as f64
                                * bytes_per_type(&inp.dtype);
                        }
                    }
                }
                0.0
            }
            // StableHLO dot/conv: approximate as matmul over last two dims
            "STABLEHLO_CONVOLUTION" | "STABLEHLO_DOT_GENERAL" => {
                if let (Some(lhs), Some(rhs), Some(out)) = (
                    tensor_at(inputs, 0),
                    tensor_at(inputs, 1),
                    tensor_at(outputs, 0),
                ) {
                    if lhs.shape.len() >= 2 && rhs.shape.len() >= 2 && out.shape.len() >= 2 {
                        let k = rhs.shape[rhs.shape.len() - 2] as f64;
                        let n = out.shape.last().copied().unwrap_or(0) as f64;
                        let batch_m: f64 = out.shape[..out.shape.len() - 1]
                            .iter()
                            .fold(1.0, |a, d| a * *d as f64);
                        macs = batch_m * n * k;
                    }
                }
                0.0
            }
            // RNN family — estimate from weight tensor sizes
            "LSTM" | "UNIDIRECTIONAL_SEQUENCE_LSTM" => {
                // inputs[0]: [batch, seq_len, input_size]
                // inputs[2]: input_to_forget_weights [num_units, input_size]  (always present)
                // inputs[6]: recurrent_to_forget_weights [num_units, num_units]
                if let (Some(inp), Some(ifw)) = (tensor_at(inputs, 0), tensor_at(inputs, 2)) {
                    if inp.shape.len() >= 3 && ifw.shape.len() == 2 {
                        let batch = inp.shape[0] as f64;
                        let seq_len = inp.shape[1] as f64;
                        let num_units = ifw.shape[0] as f64;
                        let input_size = ifw.shape[1] as f64;
                        // 4 gates × (input_proj + recurrent_proj) per timestep
                        macs = 4.0 * num_units * (input_size + num_units) * batch * seq_len;
                    }
                }
                0.0
            }
            "BIDIRECTIONAL_SEQUENCE_LSTM" => {
                if let (Some(inp), Some(ifw)) = (tensor_at(inputs, 0), tensor_at(inputs, 2)) {
                    if inp.shape.len() >= 3 && ifw.shape.len() == 2 {
                        let batch = inp.shape[0] as f64;
                        let seq_len = inp.shape[1] as f64;
                        let num_units = ifw.shape[0] as f64;
                        let input_size = ifw.shape[1] as f64;
                        macs = 2.0 * 4.0 * num_units * (input_size + num_units) * batch * seq_len;
                    }
                }
                0.0
            }
            "RNN" | "UNIDIRECTIONAL_SEQUENCE_RNN" => {
                // inputs[1]: weights [num_units, input_size]
                // inputs[2]: recurrent_weights [num_units, num_units]
                if let (Some(inp), Some(w)) = (tensor_at(inputs, 0), tensor_at(inputs, 1)) {
                    if inp.shape.len() >= 3 && w.shape.len() == 2 {
                        let batch = inp.shape[0] as f64;
                        let seq_len = inp.shape[1] as f64;
                        let num_units = w.shape[0] as f64;
                        let input_size = w.shape[1] as f64;
                        macs = num_units * (input_size + num_units) * batch * seq_len;
                    }
                }
                0.0
            }
            "SVDF" => {
                // inputs[1]: weights_feature [rank, input_size]
                // inputs[2]: weights_time [num_filters, memory_size]
                // out: [batch, num_units] where num_units = num_filters / rank
                if let (Some(inp), Some(wf)) = (tensor_at(inputs, 0), tensor_at(inputs, 1)) {
                    if wf.shape.len() == 2 {
                        let batch = inp.shape.first().copied().unwrap_or(1) as f64;
                        let rank = wf.shape[0] as f64;
                        let input_size = wf.shape[1] as f64;
                        macs = batch * rank * input_size;
                    }
                }
                0.0
            }
            // Pooling — estimate kernel area from spatial downsampling ratio
            "AVERAGE_POOL_2D" | "MAX_POOL_2D" | "L2_POOL_2D" => {
                if let (Some(inp), Some(out)) = (tensor_at(inputs, 0), tensor_at(outputs, 0)) {
                    if inp.shape.len() == 4 && out.shape.len() == 4 {
                        let (n, oh, ow, c) = (
                            out.shape[0] as f64,
                            out.shape[1] as f64,
                            out.shape[2] as f64,
                            out.shape[3] as f64,
                        );
                        let kh = (inp.shape[1] as f64 / oh).ceil().max(1.0);
                        let kw = (inp.shape[2] as f64 / ow).ceil().max(1.0);
                        n * oh * ow * c * kh * kw
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            // Global reductions: scan entire input
            "MEAN" | "SUM" | "REDUCE_MAX" | "REDUCE_MIN" | "REDUCE_PROD" | "REDUCE_ANY"
            | "REDUCE_ALL" | "REDUCE_WINDOW" => in_elems(),
            // Transcendental / normalization: multiple ops per element
            "SOFTMAX" | "LOG_SOFTMAX" => 5.0 * out_elems(),
            "L2_NORMALIZATION" | "LOCAL_RESPONSE_NORMALIZATION" => 4.0 * out_elems(),
            // Element-wise activation (1 op / element)
            "RELU" | "RELU6" | "RELU_N1_TO_1" | "RELU_0_TO_1" | "ELU" | "TANH" | "LOGISTIC"
            | "LEAKY_RELU" | "HARD_SWISH" | "GELU" | "ABS" | "NEG" | "SQRT" | "RSQRT"
            | "SQUARE" | "CEIL" | "FLOOR" | "ROUND" | "LOG" | "EXP" | "SIN" | "SIGN" | "ATAN2" => {
                out_elems()
            }
            // Element-wise binary (1 op / element)
            "ADD" | "SUB" | "MUL" | "DIV" | "MAXIMUM" | "MINIMUM" | "SQUARED_DIFFERENCE"
            | "PRELU" | "FLOOR_DIV" | "FLOOR_MOD" | "POW" => out_elems(),
            // StableHLO element-wise (same pattern)
            "STABLEHLO_ADD"
            | "STABLEHLO_SUBTRACT"
            | "STABLEHLO_MULTIPLY"
            | "STABLEHLO_DIVIDE"
            | "STABLEHLO_MAXIMUM"
            | "STABLEHLO_MINIMUM"
            | "STABLEHLO_ABS"
            | "STABLEHLO_NEGATE"
            | "STABLEHLO_LOGISTIC"
            | "STABLEHLO_TANH"
            | "STABLEHLO_EXPONENTIAL"
            | "STABLEHLO_LOG"
            | "STABLEHLO_FLOOR"
            | "STABLEHLO_RSQRT"
            | "STABLEHLO_POWER"
            | "STABLEHLO_COSINE"
            | "STABLEHLO_CBRT" => out_elems(),
            "STABLEHLO_REDUCE" | "STABLEHLO_REDUCE_WINDOW" => in_elems(),
            _ => 0.0,
        };
    }

    let total_ops = macs * 2.0 + extra_ops;
    (macs, total_ops, estimated_bytes, row_ws)
}

fn element_count(shape: &[i32]) -> f64 {
    if shape.iter().any(|dim| *dim <= 0) {
        return 0.0;
    }
    shape.iter().fold(1.0, |acc, dim| acc * *dim as f64)
}

fn tensor_bytes(tensor: &TensorInfo) -> f64 {
    element_count(&tensor.shape) * bytes_per_type(&tensor.dtype)
}

fn bytes_per_type(dtype: &str) -> f64 {
    match dtype {
        "FLOAT32" => 4.0,
        "FLOAT16" => 2.0,
        "INT8" => 1.0,
        "UINT8" => 1.0,
        "INT16" => 2.0,
        "INT32" => 4.0,
        "INT64" => 8.0,
        "BOOL" => 1.0,
        "FLOAT64" => 8.0,
        _ => 4.0,
    }
}

fn static_bound_guess_for_target(
    name: &str,
    ops: f64,
    estimated_bytes: f64,
    intensity: f64,
    target: &TargetProfile,
    is_float: bool,
) -> &'static str {
    if ops == 0.0 && estimated_bytes > 0.0 {
        return "memory-bound";
    }
    let (memory_threshold, compute_threshold) = bound_thresholds_for_op(name, target, is_float);
    if intensity < memory_threshold {
        return "memory-bound";
    }
    if intensity >= compute_threshold {
        return "compute-bound";
    }
    "mixed"
}

fn bound_thresholds_for_op(name: &str, target: &TargetProfile, is_float: bool) -> (f64, f64) {
    // FP32 ops use scaled-down thresholds because FP32 hardware throughput is lower
    // than INT8 SIMD throughput (fp32_compute_factor captures the ratio).
    let scale = if is_float {
        1.0 / target.fp32_compute_factor
    } else {
        1.0
    };
    let mut memory_threshold = target.memory_bound_intensity * scale;
    let mut compute_threshold = target.compute_bound_intensity * scale;
    if name == "DEPTHWISE_CONV_2D" && target.id == "x86_avx2" {
        memory_threshold *= 1.25;
        compute_threshold *= 1.15;
    } else if name == "DEPTHWISE_CONV_2D" && target.id == "x86_sse4" {
        memory_threshold *= 1.4;
        compute_threshold *= 1.25;
    } else if name == "DEPTHWISE_CONV_2D" && target.id == "wasm_simd" {
        memory_threshold *= 1.2;
        compute_threshold *= 1.1;
    }
    (memory_threshold, compute_threshold)
}

fn roofline_reason_for_op(
    name: &str,
    ops: f64,
    estimated_bytes: f64,
    intensity: f64,
    target: &TargetProfile,
    is_float: bool,
) -> String {
    if ops == 0.0 && estimated_bytes > 0.0 {
        return format!("ROOF:MEM:0:{}", target.id);
    }
    let (mem_th, cmp_th) = bound_thresholds_for_op(name, target, is_float);
    let dtype = if is_float { "F" } else { "Q" };
    let mut code = format!(
        "ROOF:{}:{:.2}:{:.1}:{:.1}:{}",
        dtype, intensity, mem_th, cmp_th, target.id
    );
    if name == "DEPTHWISE_CONV_2D" && matches!(target.id.as_str(), "x86_avx2" | "x86_sse4") {
        code.push_str(":GATHER");
    } else if target.sve2 && !is_float {
        code.push_str(":SVE2");
    }
    code
}

struct RooflineActionContext<'a> {
    target: &'a TargetProfile,
    is_float: bool,
    quantized_compute_path: bool,
    xnnpack_supported: bool,
    channel_alignment_status: &'a str,
}

fn roofline_action(
    name: &str,
    ops: f64,
    estimated_bytes: f64,
    intensity: f64,
    context: RooflineActionContext<'_>,
) -> &'static str {
    let RooflineActionContext {
        target,
        is_float,
        quantized_compute_path,
        xnnpack_supported,
        channel_alignment_status,
    } = context;
    let guess =
        static_bound_guess_for_target(name, ops, estimated_bytes, intensity, target, is_float);
    if name == "QUANTIZE" {
        return "graph input/output quantize; may be absorbed by delegate input path or executed as separate TFLite kernel";
    }
    if name == "AVERAGE_POOL_2D" && ops == 0.0 && estimated_bytes > 0.0 {
        return "zero-MAC pooling traffic in static estimate; verify whether XNNPACK handles this dtype/quant contract";
    }
    if matches!(name, "SHAPE" | "REDUCE_PROD") {
        return "structural shape/reduction plumbing; separate from breaks with high adjacent delegated-MAC exposure";
    }
    if matches!(
        name,
        "BATCH_MATMUL" | "GATHER" | "SQUARED_DIFFERENCE" | "REDUCE_MAX"
    ) {
        return "fallback-sensitive op; rank by fallback tensor traffic rather than raw break count";
    }
    if matches!(
        name,
        "PAD" | "RESIZE_BILINEAR" | "TRANSPOSE" | "RESHAPE" | "CONCATENATION"
    ) {
        if name == "RESHAPE" {
            return "static RESHAPE may be folded/no-op; verify delegate log before treating it as a real copy";
        }
        return "remove/collapse layout or copy; check delegate support";
    }
    if name == "DEPTHWISE_CONV_2D" && matches!(target.id.as_str(), "x86_avx2" | "x86_sse4") {
        return "check NHWC depthwise layout/gather cost and channel-tail padding";
    }
    if name == "DEPTHWISE_CONV_2D" && target.id == "wasm_simd" {
        return "validate browser WASM SIMD depthwise path, threading, and layout copies";
    }
    if matches!(name, "ADD" | "MUL" | "MEAN" | "SOFTMAX") {
        return "check fusion/delegate; reduce full-tensor passes";
    }
    if matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED") {
        if is_float {
            if guess == "compute-bound" {
                return "FP32 compute-bound; quantize to INT8 for further throughput gain on this target";
            }
            if guess == "memory-bound" {
                return "FP32 low-intensity memory-traffic candidate; INT8 conversion can reduce eligible tensor payload width by up to 4x, subject to quantization coverage, packing, and conversion boundaries";
            }
            return "FP32 mixed; INT8 conversion may cut eligible payload width by up to 4x and raise SIMD throughput, subject to coverage and runtime support";
        }
        if quantized_compute_path && !xnnpack_supported {
            return "predicted quantized fallback; confirm runtime assignment before changing arithmetic or layout";
        }
        if quantized_compute_path && channel_alignment_status == "misaligned" {
            return "source-backed channel-tail candidate; confirm selected microkernel and measured tail cost before changing width";
        }
        if guess == "compute-bound" {
            return "no artifact-local defect identified; bind runtime kernel selection, utilization, and thread evidence";
        }
        if guess == "memory-bound" {
            return "low-intensity posture only; optimize traffic or fusion only when target profiling confirms a latency contribution";
        }
        return "mixed static posture; use target profiling before selecting a compute or traffic intervention";
    }
    if is_delegate_break_suspect(name) {
        return "check delegate coverage/fallback";
    }
    "profile on target"
}

fn detect_quant_holes(
    ops: &mut [OpInfo],
    input_tensor_indices: &[i32],
    output_tensor_indices: &[i32],
    tensors: &[TensorInfo],
    total_macs: f64,
) -> Vec<QuantHoleInfo> {
    let graph_inputs: std::collections::HashSet<i32> =
        input_tensor_indices.iter().copied().collect();
    let graph_outputs: std::collections::HashSet<i32> =
        output_tensor_indices.iter().copied().collect();

    // Freeze graph connectivity before mutating the Q/DQ annotations. Operator
    // serialization order is not a dataflow edge and must not drive impact.
    struct QuantOpSnapshot {
        index: usize,
        name: String,
        inputs: Vec<i32>,
        outputs: Vec<i32>,
        macs: f64,
    }

    let op_snapshot: Vec<QuantOpSnapshot> = ops
        .iter()
        .map(|op| QuantOpSnapshot {
            index: op.index,
            name: op.name.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            macs: op.macs,
        })
        .collect();
    let mut producers_by_tensor = HashMap::<i32, Vec<usize>>::new();
    let mut consumers_by_tensor = HashMap::<i32, Vec<usize>>::new();
    for (position, snapshot) in op_snapshot.iter().enumerate() {
        for &tensor_index in snapshot.inputs.iter().filter(|&&index| index >= 0) {
            consumers_by_tensor
                .entry(tensor_index)
                .or_default()
                .push(position);
        }
        for &tensor_index in snapshot.outputs.iter().filter(|&&index| index >= 0) {
            producers_by_tensor
                .entry(tensor_index)
                .or_default()
                .push(position);
        }
    }

    let mut holes = Vec::<QuantHoleInfo>::new();

    for (position, op) in ops.iter_mut().enumerate() {
        let name = op.name.as_str();
        if name != "DEQUANTIZE" && name != "QUANTIZE" {
            continue;
        }

        // A serialized conversion at either external interface is a boundary
        // contract, not a mid-graph hole. This covers the common float-I/O /
        // integer-interior pattern as well as integer-I/O / float-interior.
        let valid_inputs: Vec<i32> = op
            .inputs
            .iter()
            .copied()
            .filter(|&index| index >= 0)
            .collect();
        let valid_outputs: Vec<i32> = op
            .outputs
            .iter()
            .copied()
            .filter(|&index| index >= 0)
            .collect();
        let touches_graph_input =
            !valid_inputs.is_empty() && valid_inputs.iter().all(|idx| graph_inputs.contains(idx));
        let touches_graph_output = !valid_outputs.is_empty()
            && valid_outputs.iter().all(|idx| graph_outputs.contains(idx));
        let is_boundary = touches_graph_input || touches_graph_output;

        if is_boundary {
            continue;
        }

        let source_tensor = op
            .inputs
            .first()
            .filter(|&&index| index >= 0)
            .and_then(|&index| tensors.get(index as usize));
        let output_tensor = op
            .outputs
            .first()
            .filter(|&&index| index >= 0)
            .and_then(|&index| tensors.get(index as usize));
        let from_dtype = source_tensor
            .map(|tensor| tensor.dtype.as_str())
            .unwrap_or("?");
        let to_dtype = output_tensor
            .map(|tensor| tensor.dtype.as_str())
            .unwrap_or("?");

        // A conversion of serialized constants is a storage/compute precision
        // bridge, not an activation island. In particular, TFLite FP16 models
        // commonly expand every stored FP16 kernel through DEQUANTIZE before
        // otherwise-FP32 compute.
        if source_tensor.is_some_and(|tensor| tensor.constant_buffer) {
            continue;
        }

        // This audit's quantization-hole contract is intentionally limited to
        // 8-bit activation transitions. Other precision casts are retained in
        // the op inventory but must not be described as FP32 islands in INT8.
        let hole_class = match (name, from_dtype, to_dtype) {
            ("DEQUANTIZE", "INT8" | "UINT8", "FLOAT32") => "int8-to-fp32",
            ("QUANTIZE", "FLOAT32", "INT8" | "UINT8") => "fp32-to-int8",
            _ => continue,
        };

        let mut upstream_positions = valid_inputs
            .iter()
            .flat_map(|tensor_index| producers_by_tensor.get(tensor_index).into_iter().flatten())
            .copied()
            .filter(|&neighbor_position| neighbor_position != position)
            .collect::<Vec<_>>();
        let mut downstream_positions = valid_outputs
            .iter()
            .flat_map(|tensor_index| consumers_by_tensor.get(tensor_index).into_iter().flatten())
            .copied()
            .filter(|&neighbor_position| neighbor_position != position)
            .collect::<Vec<_>>();
        upstream_positions.sort_unstable();
        upstream_positions.dedup();
        downstream_positions.sort_unstable();
        downstream_positions.dedup();

        let neighbor_label = |positions: &[usize], external: &str| {
            if positions.is_empty() {
                external.to_string()
            } else {
                positions
                    .iter()
                    .map(|&position| {
                        let snapshot = &op_snapshot[position];
                        format!("#{} {}", snapshot.index, snapshot.name)
                    })
                    .collect::<Vec<_>>()
                    .join(" / ")
            }
        };
        let prev_op_name = neighbor_label(&upstream_positions, "graph-input");
        let next_op_name = neighbor_label(&downstream_positions, "graph-output");

        // Adjacent MAC impact is the maximum graph-neighbor contribution,
        // divided by the exact total model MAC denominator.
        let prev_macs = upstream_positions
            .iter()
            .map(|&position| op_snapshot[position].macs)
            .fold(0.0_f64, f64::max);
        let next_macs = downstream_positions
            .iter()
            .map(|&position| op_snapshot[position].macs)
            .fold(0.0_f64, f64::max);
        let adjacent_mac_percent = if total_macs > 0.0 {
            prev_macs.max(next_macs) / total_macs
        } else {
            0.0
        };

        let detail = format!(
            "#{} {} ({} -> {}): graph-adjacent upstream {} and downstream {}; maximum neighbor {:.1}% of total model MACs",
            op.index,
            name,
            from_dtype,
            to_dtype,
            prev_op_name,
            next_op_name,
            adjacent_mac_percent * 100.0
        );

        op.quant_hole = true;
        op.quant_hole_class = hole_class.to_string();
        op.quant_hole_detail = detail.clone();

        holes.push(QuantHoleInfo {
            op_index: op.index,
            op_name: name.to_string(),
            hole_class: hole_class.to_string(),
            prev_op_name,
            next_op_name,
            from_dtype: from_dtype.to_string(),
            to_dtype: to_dtype.to_string(),
            adjacent_mac_percent,
            detail,
        });
    }

    // Sort by adjacent_mac_percent descending (worst first)
    holes.sort_by(|a, b| {
        b.adjacent_mac_percent
            .partial_cmp(&a.adjacent_mac_percent)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    holes
}

fn fused_activation_for_op(fb: &Fb, name: &str, options_table: Option<usize>) -> String {
    let Some(table) = options_table else {
        return "NONE".to_string();
    };
    let field_index = match name {
        "ADD" | "MUL" | "SUB" | "DIV" | "FULLY_CONNECTED" | "L2_NORMALIZATION" => Some(0),
        "CONCATENATION" => Some(1),
        "CONV_2D" => Some(3),
        "AVERAGE_POOL_2D" | "MAX_POOL_2D" => Some(5),
        "DEPTHWISE_CONV_2D" => Some(4),
        _ => None,
    };
    field_index
        .and_then(|field| fb.field_pos(table, field))
        .and_then(|pos| fb.i8(pos))
        .map(activation_name)
        .unwrap_or("NONE")
        .to_string()
}

fn activation_name(code: i8) -> &'static str {
    match code {
        1 => "RELU",
        2 => "RELU_N1_TO_1",
        3 => "RELU6",
        4 => "TANH",
        5 => "SIGN_BIT",
        _ => "NONE",
    }
}

fn is_8bit_quantized(dtype: &str) -> bool {
    matches!(dtype, "INT8" | "UINT8")
}

fn is_float_dtype(dtype: &str) -> bool {
    matches!(dtype, "FLOAT32" | "FLOAT16" | "BFLOAT16" | "FLOAT64")
}

fn op_has_8bit_quant(inputs: &[i32], outputs: &[i32], tensors: &[TensorInfo]) -> bool {
    inputs
        .iter()
        .chain(outputs.iter())
        .filter_map(|idx| tensors.get(*idx as usize))
        .any(|tensor| is_8bit_quantized(&tensor.dtype))
}

#[derive(Clone, Copy)]
struct Conv2dKernelGeometry {
    stride_width: usize,
    stride_height: usize,
    dilation_width: usize,
    dilation_height: usize,
}

/// Static kernel-family eligibility on the same naming axis used by XNNPACK
/// profiling. This is not an observed dispatch: a CONV_2D is called GEMM only
/// when the artifact proves the unit-stride, unit-dilation, non-grouped 1x1
/// geometry required by that lowering. Every other convolution remains IGEMM.
fn compute_kernel_class(
    name: &str,
    inputs: &[i32],
    tensors: &[TensorInfo],
    conv_geometry: Option<Conv2dKernelGeometry>,
) -> String {
    let dtype = inputs
        .first()
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| match tensor.dtype.as_str() {
            "UINT8" => "qu8",
            "INT8" => "qs8",
            "INT16" => "qs16",
            "FLOAT16" => "f16",
            "FLOAT32" => "f32",
            "" => "unknown",
            _ => "other",
        })
        .unwrap_or("unknown");
    let conv_is_static_gemm_eligible = || {
        let Some(geometry) = conv_geometry else {
            return false;
        };
        let Some(input) = inputs.first().and_then(|idx| tensors.get(*idx as usize)) else {
            return false;
        };
        let Some(filter) = inputs.get(1).and_then(|idx| tensors.get(*idx as usize)) else {
            return false;
        };
        input.shape.len() == 4
            && filter.shape.len() == 4
            && filter.shape[1] == 1
            && filter.shape[2] == 1
            && input.shape[3] > 0
            && filter.shape[3] == input.shape[3]
            && geometry.stride_width == 1
            && geometry.stride_height == 1
            && geometry.dilation_width == 1
            && geometry.dilation_height == 1
    };
    let kernel = match name {
        "FULLY_CONNECTED" => "gemm",
        "CONV_2D" => {
            if conv_is_static_gemm_eligible() {
                "gemm"
            } else {
                "igemm"
            }
        }
        "DEPTHWISE_CONV_2D" => "dwconv",
        "TRANSPOSE_CONV" => "deconv",
        "BATCH_MATMUL" => "batch_matmul",
        "AVERAGE_POOL_2D" | "MAX_POOL_2D" | "L2_POOL_2D" | "MEAN" => "pool",
        "ADD" | "SUB" | "MUL" | "DIV" | "MAXIMUM" | "MINIMUM" => "binary",
        _ => "other",
    };
    format!("{dtype}_{kernel}")
}

fn op_has_quantized_compute_path(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> bool {
    if !is_quantized_compute_candidate(name) {
        return false;
    }
    let activation_input_slot = if matches!(name, "TRANSPOSE_CONV" | "CONV_3D_TRANSPOSE") {
        2
    } else {
        0
    };
    let input0_q = inputs
        .get(activation_input_slot)
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| is_8bit_quantized(&tensor.dtype))
        .unwrap_or(false);
    let output0_q = outputs
        .first()
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| is_8bit_quantized(&tensor.dtype))
        .unwrap_or(false);
    match name {
        "CONV_2D"
        | "DEPTHWISE_CONV_2D"
        | "TRANSPOSE_CONV"
        | "CONV_3D"
        | "CONV_3D_TRANSPOSE"
        | "FULLY_CONNECTED"
        | "LSTM"
        | "UNIDIRECTIONAL_SEQUENCE_LSTM"
        | "BIDIRECTIONAL_SEQUENCE_LSTM"
        | "RNN"
        | "UNIDIRECTIONAL_SEQUENCE_RNN"
        | "SVDF" => input0_q && output0_q,
        "BATCH_MATMUL" => {
            let input1_q = inputs
                .get(1)
                .and_then(|idx| tensors.get(*idx as usize))
                .map(|tensor| is_8bit_quantized(&tensor.dtype))
                .unwrap_or(false);
            input0_q && input1_q && output0_q
        }
        _ => input0_q && output0_q && op_has_8bit_quant(inputs, outputs, tensors),
    }
}

fn is_quantized_compute_candidate(name: &str) -> bool {
    is_mac_bearing_compute_op(name)
        || matches!(
            name,
            "ADD"
                | "SUB"
                | "MUL"
                | "DIV"
                | "MEAN"
                | "AVERAGE_POOL_2D"
                | "MAX_POOL_2D"
                | "SOFTMAX"
                | "LOGISTIC"
                | "TANH"
                | "RELU"
                | "RELU6"
        )
}

fn is_mac_bearing_compute_op(name: &str) -> bool {
    matches!(
        name,
        "CONV_2D"
            | "DEPTHWISE_CONV_2D"
            | "TRANSPOSE_CONV"
            | "CONV_3D"
            | "CONV_3D_TRANSPOSE"
            | "FULLY_CONNECTED"
            | "BATCH_MATMUL"
            | "STABLEHLO_CONVOLUTION"
            | "STABLEHLO_DOT_GENERAL"
            | "LSTM"
            | "UNIDIRECTIONAL_SEQUENCE_LSTM"
            | "BIDIRECTIONAL_SEQUENCE_LSTM"
            | "RNN"
            | "UNIDIRECTIONAL_SEQUENCE_RNN"
            | "SVDF"
    )
}

fn tensor_dtype_at(indices: &[i32], tensors: &[TensorInfo], position: usize) -> String {
    indices
        .get(position)
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| tensor.dtype.clone())
        .unwrap_or_else(|| "missing".to_string())
}

fn tensor_is_8bit_at(indices: &[i32], tensors: &[TensorInfo], position: usize) -> bool {
    indices
        .get(position)
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| is_8bit_quantized(&tensor.dtype))
        .unwrap_or(false)
}

fn tensor_has_quant_metadata_at(indices: &[i32], tensors: &[TensorInfo], position: usize) -> bool {
    indices
        .get(position)
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| tensor.quant_scales > 0)
        .unwrap_or(false)
}

fn precision_conversion_state(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> Option<&'static str> {
    if !matches!(name, "QUANTIZE" | "DEQUANTIZE") {
        return None;
    }
    let source = inputs
        .first()
        .filter(|&&index| index >= 0)
        .and_then(|&index| tensors.get(index as usize));
    let output = outputs
        .first()
        .filter(|&&index| index >= 0)
        .and_then(|&index| tensors.get(index as usize));
    let from_dtype = source.map(|tensor| tensor.dtype.as_str()).unwrap_or("?");
    let to_dtype = output.map(|tensor| tensor.dtype.as_str()).unwrap_or("?");

    if source.is_some_and(|tensor| tensor.constant_buffer) {
        return Some(
            if name == "DEQUANTIZE" && from_dtype == "FLOAT16" && to_dtype == "FLOAT32" {
                "float16_constant_expansion"
            } else if name == "DEQUANTIZE"
                && is_8bit_quantized(from_dtype)
                && is_float_dtype(to_dtype)
            {
                "quantized_constant_expansion"
            } else {
                "constant_precision_conversion"
            },
        );
    }

    if matches!(
        (name, from_dtype, to_dtype),
        ("DEQUANTIZE", "INT8" | "UINT8", "FLOAT32") | ("QUANTIZE", "FLOAT32", "INT8" | "UINT8")
    ) {
        Some("quant_boundary")
    } else if name == "QUANTIZE"
        && is_8bit_quantized(from_dtype)
        && is_8bit_quantized(to_dtype)
        && from_dtype != to_dtype
    {
        Some("integer_requantization")
    } else {
        Some("precision_boundary")
    }
}

fn classify_op_quantization(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> (String, String) {
    let input0_q = tensor_is_8bit_at(inputs, tensors, 0);
    let input1_q = tensor_is_8bit_at(inputs, tensors, 1);
    let output0_q = tensor_is_8bit_at(outputs, tensors, 0);
    let input0_dtype = tensor_dtype_at(inputs, tensors, 0);
    let input1_dtype = tensor_dtype_at(inputs, tensors, 1);
    let output0_dtype = tensor_dtype_at(outputs, tensors, 0);
    let any_8bit = op_has_8bit_quant(inputs, outputs, tensors);
    let any_quant_metadata = inputs
        .iter()
        .chain(outputs.iter())
        .filter_map(|idx| tensors.get(*idx as usize))
        .any(|tensor| tensor.quant_scales > 0);
    let compute_like = matches!(
        name,
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "BATCH_MATMUL"
    );
    let compute_q = op_has_quantized_compute_path(name, inputs, outputs, tensors);
    let weight_like_q = matches!(
        name,
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "BATCH_MATMUL"
    ) && input1_q;
    let weight_like_metadata = matches!(
        name,
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "BATCH_MATMUL"
    ) && tensor_has_quant_metadata_at(inputs, tensors, 1);

    let conversion_state = precision_conversion_state(name, inputs, outputs, tensors);
    let state = if let Some(state) = conversion_state {
        state
    } else if compute_q {
        "quantized_compute"
    } else if compute_like && weight_like_q && !input0_q && !output0_q {
        "weight_only_or_dynamic_range"
    } else if compute_like && weight_like_metadata && !input0_q && !output0_q {
        "weight_metadata_only"
    } else if compute_like && any_8bit {
        "mixed_or_hybrid_compute"
    } else if !compute_like && input0_q && output0_q {
        "quantized_data_movement"
    } else if any_8bit || any_quant_metadata {
        "quant_signal_only"
    } else {
        "none"
    };

    let detail = match state {
        "quant_boundary" => format!(
            "{} converts or bridges quantized and floating domains; verify whether delegate absorbs it at graph boundaries.",
            name
        ),
        "float16_constant_expansion" => format!(
            "{} expands a serialized FP16 constant to FLOAT32 compute precision; this is a weight-storage conversion, not an activation quantization hole.",
            name
        ),
        "quantized_constant_expansion" => format!(
            "{} expands a serialized 8-bit constant into floating compute precision; this is a constant-storage conversion, not an activation quantization hole.",
            name
        ),
        "constant_precision_conversion" => format!(
            "{} converts a serialized constant from {} to {}; this is a storage precision bridge, not an activation quantization hole.",
            name, input0_dtype, output0_dtype
        ),
        "precision_boundary" => format!(
            "{} converts activation precision from {} to {}; this transition is outside the strict 8-bit activation-hole contract.",
            name, input0_dtype, output0_dtype
        ),
        "integer_requantization" => format!(
            "{} converts the activation contract from {} to {} without entering floating point; this is an integer-domain boundary, not an FP32 island.",
            name, input0_dtype, output0_dtype
        ),
        "quantized_compute" => format!(
            "activation input/output are 8-bit for this compute op: input0={} weight/input1={} output0={}",
            input0_dtype, input1_dtype, output0_dtype
        ),
        "weight_only_or_dynamic_range" => format!(
            "weight/input1 is 8-bit but activation input/output are not both 8-bit: input0={} weight/input1={} output0={}",
            input0_dtype, input1_dtype, output0_dtype
        ),
        "weight_metadata_only" => format!(
            "weight/input1 carries quant metadata but dtype path is not clearly 8-bit activation compute: input0={} weight/input1={} output0={}",
            input0_dtype, input1_dtype, output0_dtype
        ),
        "mixed_or_hybrid_compute" => format!(
            "partial 8-bit signal around compute op; input0={} weight/input1={} output0={}",
            input0_dtype, input1_dtype, output0_dtype
        ),
        "quantized_data_movement" => format!(
            "non-compute op moves 8-bit tensors: input0={} output0={}",
            input0_dtype, output0_dtype
        ),
        "quant_signal_only" => "quantization metadata or 8-bit tensor signal exists, but no strict 8-bit compute path is inferred.".to_string(),
        _ => "no quantized tensor or quantization metadata signal for this op.".to_string(),
    };

    (state.to_string(), detail)
}

fn unique_dtypes(tensors: &[TensorInfo]) -> Vec<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for tensor in tensors {
        *counts.entry(tensor.dtype.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(dtype, count)| format!("{}:{}", dtype, count))
        .collect()
}

fn classify_model_quantization(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    inputs: &[TensorInfo],
    outputs: &[TensorInfo],
    quantized_tensors: usize,
    _total_macs: f64,
) -> QuantizationStatus {
    let int8_tensors = tensors
        .iter()
        .filter(|tensor| tensor.dtype == "INT8")
        .count();
    let uint8_tensors = tensors
        .iter()
        .filter(|tensor| tensor.dtype == "UINT8")
        .count();
    let float16_tensors = tensors
        .iter()
        .filter(|tensor| tensor.dtype == "FLOAT16")
        .count();
    let float_tensors = tensors
        .iter()
        .filter(|tensor| is_float_dtype(&tensor.dtype))
        .count();
    let input_dtypes = unique_dtypes(inputs);
    let output_dtypes = unique_dtypes(outputs);
    let all_inputs_8bit =
        !inputs.is_empty() && inputs.iter().all(|tensor| is_8bit_quantized(&tensor.dtype));
    let all_outputs_8bit = !outputs.is_empty()
        && outputs
            .iter()
            .all(|tensor| is_8bit_quantized(&tensor.dtype));
    let any_float_io = inputs
        .iter()
        .chain(outputs.iter())
        .any(|tensor| is_float_dtype(&tensor.dtype));
    let quantize_ops = ops.iter().filter(|op| op.name == "QUANTIZE").count();
    let dequantize_ops = ops.iter().filter(|op| op.name == "DEQUANTIZE").count();
    let activation_quantize_ops = ops
        .iter()
        .filter(|op| {
            op.name == "QUANTIZE"
                && !matches!(
                    op.quantization_state.as_str(),
                    "float16_constant_expansion"
                        | "quantized_constant_expansion"
                        | "constant_precision_conversion"
                )
        })
        .count();
    let activation_dequantize_ops = ops
        .iter()
        .filter(|op| {
            op.name == "DEQUANTIZE"
                && !matches!(
                    op.quantization_state.as_str(),
                    "float16_constant_expansion"
                        | "quantized_constant_expansion"
                        | "constant_precision_conversion"
                )
        })
        .count();
    let activation_8bit_float_boundary_ops = ops
        .iter()
        .filter(|op| op.quantization_state == "quant_boundary")
        .count();
    let integer_requantization_ops = ops
        .iter()
        .filter(|op| op.quantization_state == "integer_requantization")
        .count();
    let constant_precision_conversion_ops = ops
        .iter()
        .filter(|op| {
            matches!(
                op.quantization_state.as_str(),
                "float16_constant_expansion"
                    | "quantized_constant_expansion"
                    | "constant_precision_conversion"
            )
        })
        .count();
    let float16_constant_expansion_ops = ops
        .iter()
        .filter(|op| op.quantization_state == "float16_constant_expansion")
        .count();
    let compute_ops = ops
        .iter()
        .filter(|op| is_mac_bearing_compute_op(&op.name))
        .collect::<Vec<_>>();
    let quantized_compute_ops = compute_ops
        .iter()
        .filter(|op| op.quantized_compute_path)
        .count();
    let mut op_state_map = BTreeMap::<String, usize>::new();
    for op in ops {
        *op_state_map
            .entry(op.quantization_state.clone())
            .or_default() += 1;
    }
    let quantized_compute_macs_sum = compute_ops
        .iter()
        .filter(|op| op.quantized_compute_path)
        .map(|op| {
            if op.macs.is_finite() && op.macs > 0.0 {
                op.macs
            } else {
                0.0
            }
        })
        .sum::<f64>();
    let quantized_compute_macs = if quantized_compute_macs_sum == 0.0 {
        0.0
    } else {
        quantized_compute_macs_sum
    };
    let quantized_tensor_percent = if tensors.is_empty() {
        0.0
    } else {
        quantized_tensors as f64 / tensors.len() as f64
    };
    let compute_macs_sum = compute_ops
        .iter()
        .map(|op| {
            if op.macs.is_finite() && op.macs > 0.0 {
                op.macs
            } else {
                0.0
            }
        })
        .sum::<f64>();
    let compute_macs = if compute_macs_sum == 0.0 {
        0.0
    } else {
        compute_macs_sum
    };
    let quantized_compute_mac_percent = if compute_macs > 0.0 {
        // Exact zero must serialize as +0.0. IEEE clamp can preserve a -0.0
        // produced by summing zero-MAC floating-path rows.
        (quantized_compute_macs / compute_macs)
            .clamp(0.0, 1.0)
            .abs()
    } else {
        0.0
    };
    let has_integer_signal = quantized_tensors > 0
        || int8_tensors > 0
        || uint8_tensors > 0
        || activation_8bit_float_boundary_ops > 0
        || integer_requantization_ops > 0;

    let (classification, label, summary, full_integer) = if !has_integer_signal
        && float16_constant_expansion_ops > 0
    {
        (
            "float16_weight_storage",
            "FP16 weight storage",
            "Serialized FP16 constants are expanded to FP32 compute precision; no 8-bit activation quantization path is inferred.",
            false,
        )
    } else if !has_integer_signal {
        (
            "not_quantized_float",
            "Not quantized",
            "No INT8/UINT8 tensors, quantization metadata, or 8-bit activation Q/DQ boundaries were detected.",
            false,
        )
    } else if all_inputs_8bit
        && all_outputs_8bit
        && !compute_ops.is_empty()
        && quantized_compute_ops == compute_ops.len()
        && float_tensors == 0
    {
        (
            "full_integer",
            "Full integer activation path",
            "Model I/O is 8-bit, every MAC-bearing compute op has 8-bit activation input/output, and no FLOAT tensor is serialized.",
            true,
        )
    } else if any_float_io && quantized_compute_mac_percent >= 0.80 {
        (
            "integer_internal_float_io",
            "Internal INT8 with float I/O",
            "Most compute MACs appear quantized, but model inputs or outputs remain floating point.",
            false,
        )
    } else if quantized_compute_mac_percent >= 0.20 {
        (
            "mixed_quantization",
            "Mixed quantization",
            "Some compute path is quantized, but FP and integer regions are both present.",
            false,
        )
    } else if quantized_tensors > 0 || int8_tensors > 0 || uint8_tensors > 0 {
        (
            "dynamic_range_or_weight_only",
            "Weight-only or dynamic-range quantization",
            "Quantized tensors exist, but activation input/output around compute ops is mostly not 8-bit.",
            false,
        )
    } else {
        (
            "qdq_signals_only",
            "Q/DQ signals only",
            "Quantize/Dequantize operators were detected, but no clear 8-bit compute path was inferred.",
            false,
        )
    };

    let detail = format!(
        "Quantized tensors: {}/{} ({:.1}%). Quantized compute MACs: {:.1}% across {}/{} compute ops. I/O dtype contract: inputs [{}], outputs [{}]. Serialized Q/DQ ops: QUANTIZE={} / DEQUANTIZE={}; activation conversions: Q={} / DQ={}; 8-bit/float boundaries: {}; integer-domain requantizations: {}; constant precision conversions: {} (FP16-to-FP32 {}).",
        quantized_tensors,
        tensors.len(),
        // .abs() after .max(): IEEE max(-0.0, 0.0) may return -0.0, printing "-0.0%"
        (quantized_tensor_percent * 100.0).max(0.0).abs(),
        (quantized_compute_mac_percent * 100.0).max(0.0).abs(),
        quantized_compute_ops,
        compute_ops.len(),
        if input_dtypes.is_empty() { "-".to_string() } else { input_dtypes.join(" / ") },
        if output_dtypes.is_empty() { "-".to_string() } else { output_dtypes.join(" / ") },
        quantize_ops,
        dequantize_ops,
        activation_quantize_ops,
        activation_dequantize_ops,
        activation_8bit_float_boundary_ops,
        integer_requantization_ops,
        constant_precision_conversion_ops,
        float16_constant_expansion_ops
    );

    QuantizationStatus {
        classification: classification.to_string(),
        label: label.to_string(),
        summary: summary.to_string(),
        detail,
        quantized_tensor_percent,
        quantized_compute_mac_percent,
        compute_macs,
        quantized_compute_macs,
        quantized_compute_ops,
        compute_ops: compute_ops.len(),
        quantize_ops,
        dequantize_ops,
        activation_quantize_ops,
        activation_dequantize_ops,
        activation_8bit_float_boundary_ops,
        integer_requantization_ops,
        constant_precision_conversion_ops,
        float16_constant_expansion_ops,
        int8_tensors,
        uint8_tensors,
        float16_tensors,
        float_tensors,
        input_dtypes,
        output_dtypes,
        op_state_counts: count_items(op_state_map),
        full_integer,
    }
}

fn precision_label_for_dtype(dtype: &str) -> &str {
    match dtype {
        "FLOAT16" => "FP16",
        "FLOAT32" => "FP32",
        "BFLOAT16" => "BF16",
        "INT8" => "INT8",
        "UINT8" => "UINT8",
        value if !value.is_empty() => value,
        _ => "unknown",
    }
}

fn xnnpack_build_requirement_for_dtype(dtype: &str) -> String {
    match dtype {
        "INT8" => "--define tflite_with_xnnpack_qs8=true; runtime build configuration not embedded".to_string(),
        "UINT8" => "--define tflite_with_xnnpack_qu8=true; legacy QU8 support is experimental and runtime build configuration is not embedded".to_string(),
        _ => "XNNPACK delegate enabled in the deployment runtime; runtime build configuration not embedded".to_string(),
    }
}

fn source_kernel_candidate_for_op(
    name: &str,
    inputs: &[i32],
    tensors: &[TensorInfo],
    target: &TargetProfile,
) -> KernelCandidateSummary {
    let dtype = inputs
        .first()
        .and_then(|idx| tensors.get(*idx as usize))
        .map(|tensor| tensor.dtype.as_str())
        .unwrap_or("");
    let protected_selector_available = (is_cortex_a55_profile(&target.id)
        || target.id == "wasm_simd")
        && matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED")
        && matches!(dtype, "FLOAT32" | "FLOAT16" | "INT8" | "UINT8");
    KernelCandidateSummary {
        candidate: target.xnnpack_kernel_family.clone(),
        evidence_class: "HEURISTIC_PROFILE".to_string(),
        selector_status: if protected_selector_available {
            "ADVANCED_SELECTOR_NOT_LOADED: exact pinned-source configuration enumeration is available only from the capability-authorized DeepBOM WASM module; the public analyzer emits no source candidate or source-backed tile claim"
        } else {
            "No protected source selector is available for this planning-profile/op/dtype signature; runtime kernel selection remains unconfirmed"
        }
        .to_string(),
        build_requirement: xnnpack_build_requirement_for_dtype(dtype),
        ..KernelCandidateSummary::default()
    }
}

fn microkernel_hint(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
    target: &TargetProfile,
) -> String {
    let quantized_dtype = inputs
        .iter()
        .chain(outputs.iter())
        .filter_map(|idx| tensors.get(*idx as usize))
        .find(|tensor| is_8bit_quantized(&tensor.dtype))
        .map(|tensor| tensor.dtype.as_str());
    if !matches!(
        name,
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "ADD" | "MUL" | "MEAN"
    ) {
        return target.xnnpack_kernel_family.clone();
    }
    if target.sve2 {
        return format!(
            "{}; prefer long delegated chains to amortize SVE2 setup/packing",
            target.xnnpack_kernel_family
        );
    }
    if quantized_dtype == Some("UINT8") {
        return format!(
            "{}; legacy QU8 path must not be described as a signed DOT-product kernel without runtime evidence",
            target.xnnpack_kernel_family
        );
    }
    if quantized_dtype == Some("INT8") && target.dot_product {
        return format!(
            "{}; INT8 dot-product path likely benefits from channel multiples of {}",
            target.xnnpack_kernel_family, target.int8_lanes
        );
    }
    if quantized_dtype.is_some() && !target.dot_product {
        return format!(
            "{}; no dot-product/VNNI, so INT8 benefit is more memory/packing dependent",
            target.xnnpack_kernel_family
        );
    }
    if target.in_order {
        return format!(
            "{}; in-order core makes prefetch-friendly contiguous chains important",
            target.xnnpack_kernel_family
        );
    }
    target.xnnpack_kernel_family.clone()
}

fn quant_risk_for_op(
    op_name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> QuantRiskSummary {
    let mut best_score = 0.0;
    let mut best_tensor = None::<&TensorInfo>;
    let mut relevant_indices = if matches!(
        op_name,
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "TRANSPOSE_CONV"
    ) {
        inputs.iter().take(2).copied().collect::<Vec<_>>()
    } else {
        inputs.to_vec()
    };
    relevant_indices.extend(outputs.iter().copied());
    relevant_indices.sort_unstable();
    relevant_indices.dedup();
    for tensor in relevant_indices
        .iter()
        .filter_map(|idx| tensors.get(*idx as usize))
    {
        let scale_signal = if tensor.scale_ratio_meaningful {
            tensor.scale_ratio.max(1.0).log10() + tensor.scale_cv.max(0.0) * 1.5
        } else {
            0.0
        };
        let zero_point_signal = zero_point_score_signal(tensor);
        let score = scale_signal + (zero_point_signal as f64 / 32.0);
        if tensor.quant_scales > 0 && (best_tensor.is_none() || score > best_score) {
            best_score = score;
            best_tensor = Some(tensor);
        }
    }
    let Some(tensor) = best_tensor else {
        return QuantRiskSummary {
            scale_ratio: 0.0,
            scale_cv: 0.0,
            scale_mode: "none".to_string(),
            scale_ratio_meaningful: false,
            zero_point_offset: 0,
            zero_point_risk: "none".to_string(),
            zero_point_status: "none".to_string(),
            label: "none".to_string(),
            detail: "No quantized scale/zero-point signal".to_string(),
        };
    };
    let zero_point_signal = zero_point_score_signal(tensor);
    let zero_point_risk =
        zero_point_risk_label_for_status(&tensor.zero_point_status, tensor.zero_point_offset_max)
            .to_string();
    let mut label = combined_quant_risk_label(
        if tensor.scale_ratio_meaningful {
            tensor.scale_ratio
        } else {
            0.0
        },
        if tensor.scale_ratio_meaningful {
            tensor.scale_cv
        } else {
            0.0
        },
        zero_point_signal,
        &tensor.zero_point_status,
    )
    .to_string();
    if label == "none" && tensor.quant_scales > 0 {
        label = "ok".to_string();
    }
    let scale_text = if tensor.scale_ratio_meaningful {
        format!(
            "scale ratio {:.2e} min {:.2e} max {:.2e} cv {:.2}",
            tensor.scale_ratio, tensor.scale_min, tensor.scale_max, tensor.scale_cv
        )
    } else if tensor.quant_scales == 1 {
        "per-tensor scale; ratio/cv N/A".to_string()
    } else {
        "no scale metadata".to_string()
    };
    let zero_point_text = if tensor.dtype == "INT8"
        && tensor.quant_scales > 1
        && tensor.zero_point_min == 0
        && tensor.zero_point_max == 0
        && tensor.zero_point_status == "ok"
    {
        "zero_point=0 is required by the symmetric per-axis INT8 weight contract and is not an independent pass signal".to_string()
    } else {
        format!(
            "zero_point range {}..{} status={} max_offset {} risk_signal {} ({}) - {}",
            tensor.zero_point_min,
            tensor.zero_point_max,
            tensor.zero_point_status,
            tensor.zero_point_offset_max,
            zero_point_signal,
            zero_point_risk,
            tensor.zero_point_detail
        )
    };
    let detail = format!(
        "T{} {} {}; {}",
        tensor.index, tensor.name, scale_text, zero_point_text
    );
    QuantRiskSummary {
        scale_ratio: if tensor.scale_ratio_meaningful {
            tensor.scale_ratio
        } else {
            0.0
        },
        scale_cv: if tensor.scale_ratio_meaningful {
            tensor.scale_cv
        } else {
            0.0
        },
        scale_mode: tensor.scale_mode.clone(),
        scale_ratio_meaningful: tensor.scale_ratio_meaningful,
        zero_point_offset: tensor.zero_point_offset_max,
        zero_point_risk,
        zero_point_status: tensor.zero_point_status.clone(),
        label,
        detail,
    }
}

fn quant_risk_label(ratio: f64) -> &'static str {
    if ratio >= 1_000_000.0 {
        "risk"
    } else if ratio >= 1_000.0 {
        "warn"
    } else if ratio > 0.0 {
        "ok"
    } else {
        "none"
    }
}

fn zero_point_score_signal(tensor: &TensorInfo) -> i64 {
    if tensor.zero_point_status == "out-of-range" {
        tensor.zero_point_offset_max
    } else {
        0
    }
}

fn combined_quant_risk_label(
    ratio: f64,
    scale_cv: f64,
    zero_point_offset: i64,
    zero_point_status: &str,
) -> &'static str {
    if zero_point_status == "out-of-range"
        || ratio >= 1_000_000.0
        || scale_cv >= 10.0
        || zero_point_offset >= 112
    {
        "risk"
    } else if ratio >= 1_000.0 || scale_cv >= 2.0 {
        "warn"
    } else if ratio > 0.0 || scale_cv > 0.0 || zero_point_offset > 0 {
        "ok"
    } else {
        "none"
    }
}

fn zero_point_risk_label(offset: i64) -> &'static str {
    if offset >= 112 {
        "risk"
    } else if offset >= 80 {
        "warn"
    } else if offset > 0 {
        "ok"
    } else {
        "none"
    }
}

fn zero_point_risk_label_for_status(status: &str, offset: i64) -> &'static str {
    if status == "out-of-range" {
        zero_point_risk_label(offset)
    } else if status == "asymmetric" {
        "watch"
    } else if status == "reinterpret" {
        "reinterpret"
    } else {
        "none"
    }
}

const L1_WORKING_SET_WATCH_RATIO: f64 = 0.9;

fn cache_working_set_is_watch(ratio: f64) -> bool {
    ratio >= L1_WORKING_SET_WATCH_RATIO
}

fn l1_working_set_severity(ratio: f64) -> &'static str {
    if ratio > 3.0 {
        "critical"
    } else if ratio > 2.0 {
        "high"
    } else if ratio > 1.0 {
        "warn"
    } else if cache_working_set_is_watch(ratio) {
        "watch"
    } else if ratio > 0.0 {
        "ok"
    } else {
        "none"
    }
}

fn weight_packing_for_op(
    name: &str,
    inputs: &[i32],
    _outputs: &[i32],
    tensors: &[TensorInfo],
    target: &TargetProfile,
) -> WeightPackingSummary {
    if !matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED") || inputs.len() < 2 {
        return WeightPackingSummary {
            weight_bytes: 0.0,
            overhead_us: 0.0,
            risk: "none".to_string(),
            detail: "No static weight packing signal".to_string(),
        };
    }
    let Some(weight) = tensors.get(inputs[1] as usize) else {
        return WeightPackingSummary {
            weight_bytes: 0.0,
            overhead_us: 0.0,
            risk: "none".to_string(),
            detail: "Missing static weight tensor".to_string(),
        };
    };
    let weight_bytes = tensor_bytes(weight);
    let bandwidth = target.weight_packing_bandwidth_gbps.max(1.0);
    let setup_us = if target.in_order { 8.0 } else { 4.0 };
    let overhead_us = if weight_bytes > 0.0 {
        weight_bytes / (bandwidth * 1_000_000_000.0) * 1_000_000.0 + setup_us
    } else {
        0.0
    };
    let (risk, reason) = if weight_bytes == 0.0 {
        ("none", "no static weight tensor".to_string())
    } else if overhead_us >= PACKING_WARN_OVERHEAD_US {
        (
            "warn",
            format!(
                "estimated packing >= {:.0} us threshold",
                PACKING_WARN_OVERHEAD_US
            ),
        )
    } else {
        (
            "ok",
            format!(
                "below {:.0} us general packing warning threshold",
                PACKING_WARN_OVERHEAD_US
            ),
        )
    };
    let detail = format!(
        "weight tensor T{} {} = {:.0} B ({}); potential one-time packing cost {:.1} us on {} during interpreter preparation or first invocation, depending on runtime implementation. Formula: {:.0} B / {:.1}e9 B/s + setup {:.1} us = {:.1} us. Setup and bandwidth are rulepack heuristics, not device-calibrated measurements; criteria: {}",
        weight.index,
        weight.name,
        weight_bytes,
        fmt_bytes(weight_bytes),
        overhead_us,
        target.label,
        weight_bytes,
        bandwidth,
        setup_us,
        overhead_us,
        reason
    );
    WeightPackingSummary {
        weight_bytes,
        overhead_us,
        risk: risk.to_string(),
        detail,
    }
}

fn channel_alignment_for_op(
    name: &str,
    outputs: &[i32],
    tensors: &[TensorInfo],
    target: &TargetProfile,
    context: ChannelAlignmentContext<'_>,
) -> ChannelAlignmentSummary {
    let int8_multiple = target.channel_alignment_multiple.max(1);
    let mut multiples = if !context.source_tile_multiples.is_empty() {
        context.source_tile_multiples.to_vec()
    } else if context.source_tile_multiple > 0 {
        vec![context.source_tile_multiple]
    } else if context.is_float_compute {
        vec![(int8_multiple / 2).max(4)]
    } else {
        vec![int8_multiple]
    };
    multiples.retain(|value| *value > 0);
    multiples.sort_unstable();
    multiples.dedup();
    if multiples.is_empty() {
        multiples.push(1);
    }
    let source_backed =
        !context.source_tile_multiples.is_empty() || context.source_tile_multiple > 0;
    let default_multiple = multiples[0];
    let Some(output) = outputs.first().and_then(|idx| tensors.get(*idx as usize)) else {
        return ChannelAlignmentSummary {
            output_channels: 0,
            multiple: default_multiple,
            status: "none".to_string(),
            detail: "No output tensor".to_string(),
            tail_overhead_percent: 0.0,
            tail_overhead_percent_min: 0.0,
            tail_overhead_percent_max: 0.0,
        };
    };
    let channels = output_channels_for_shape(name, &output.shape);
    if channels <= 0 || !matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED") {
        return ChannelAlignmentSummary {
            output_channels: channels,
            multiple: default_multiple,
            status: "not-applicable".to_string(),
            detail: "No channel-tail alignment check for this op".to_string(),
            tail_overhead_percent: 0.0,
            tail_overhead_percent_min: 0.0,
            tail_overhead_percent_max: 0.0,
        };
    }
    let modeled = multiples
        .iter()
        .map(|multiple| {
            let padded = (channels as usize).div_ceil(*multiple) * *multiple;
            let inactive = if padded > 0 {
                (padded as f64 - channels as f64) / padded as f64
            } else {
                0.0
            };
            (*multiple, padded, inactive)
        })
        .collect::<Vec<_>>();
    let min_waste = modeled
        .iter()
        .map(|(_, _, waste)| *waste)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let (multiple, padded, max_waste) = modeled
        .iter()
        .copied()
        .max_by(|left, right| left.2.total_cmp(&right.2))
        .unwrap_or((default_multiple, channels as usize, 0.0));
    let multiples_text = multiples
        .iter()
        .map(|value| format!("x{value}"))
        .collect::<Vec<_>>()
        .join("/");
    if max_waste == 0.0 {
        return ChannelAlignmentSummary {
            output_channels: channels,
            multiple,
            status: "aligned".to_string(),
            detail: format!(
                "output channels {} align to every {} multiple {} ({})",
                channels,
                if source_backed {
                    "pinned-source candidate-set tile"
                } else {
                    "planning-profile assumed tile"
                },
                multiples_text,
                context.kernel_evidence_class
            ),
            tail_overhead_percent: 0.0,
            tail_overhead_percent_min: 0.0,
            tail_overhead_percent_max: 0.0,
        };
    }
    let inactive_lane_range = if (max_waste - min_waste).abs() <= 1e-12 {
        format!("{:.0}%", min_waste * 100.0)
    } else {
        format!("{:.0}% to {:.0}%", min_waste * 100.0, max_waste * 100.0)
    };
    ChannelAlignmentSummary {
        output_channels: channels,
        multiple,
        status: "misaligned".to_string(),
        detail: format!(
            "output channels {} vs {} {} output-channel candidate multiples {}: modeled inactive-lane range {}; worst case is x{} padding to {} channels ({:.0}% occupancy); runtime kernel selection and tail handling are not confirmed from the artifact",
            channels,
            if source_backed { "pinned-source candidate-set" } else { "planning-profile assumed" },
            context.compute_precision_label,
            multiples_text,
            inactive_lane_range,
            multiple,
            padded,
            (1.0 - max_waste) * 100.0,
        ),
        tail_overhead_percent: max_waste,
        tail_overhead_percent_min: min_waste,
        tail_overhead_percent_max: max_waste,
    }
}

fn suppress_graph_output_channel_alignment(ops: &mut [OpInfo], output_tensor_indices: &[i32]) {
    let mut semantic_output_tensors = output_tensor_indices
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    for op in ops.iter().rev() {
        let reaches_output = op
            .outputs
            .iter()
            .any(|output| semantic_output_tensors.contains(output));
        if reaches_output
            && matches!(
                op.name.as_str(),
                "RESHAPE"
                    | "SQUEEZE"
                    | "EXPAND_DIMS"
                    | "SOFTMAX"
                    | "LOGISTIC"
                    | "QUANTIZE"
                    | "DEQUANTIZE"
                    | "CAST"
            )
        {
            if let Some(input) = op.inputs.first().copied().filter(|index| *index >= 0) {
                semantic_output_tensors.insert(input);
            }
        }
    }
    for op in ops.iter_mut().filter(|op| {
        op.channel_alignment_status == "misaligned"
            && op
                .outputs
                .iter()
                .any(|output| semantic_output_tensors.contains(output))
    }) {
        op.channel_alignment_status = "graph-output-contract".to_string();
        op.channel_alignment_detail = format!(
            "Output channel/feature count {} reaches the serialized graph output through contract-preserving structural or output transforms. A generic SIMD padding recommendation is suppressed because changing this semantic axis would change the model interface; inspect the selected runtime kernel tail only with task-contract evidence.",
            op.output_channels
        );
        op.static_action = "graph-output semantic axis; generic width/alignment rewrite suppressed; inspect runtime tail only with task-contract evidence".to_string();
        op.channel_tail_overhead_percent = 0.0;
        op.channel_tail_overhead_percent_min = 0.0;
        op.channel_tail_overhead_percent_max = 0.0;
    }
}

fn output_channels_for_shape(name: &str, shape: &[i32]) -> i32 {
    if matches!(name, "CONV_2D" | "DEPTHWISE_CONV_2D") && shape.len() == 4 {
        return shape[3];
    }
    if name == "FULLY_CONNECTED" && shape.len() >= 2 {
        return *shape.last().unwrap_or(&0);
    }
    0
}

fn estimate_model_int8_speedup(
    ops: &[OpInfo],
    total_macs: f64,
    quantized_tensors: usize,
    tensor_count: usize,
    fallback_byte_percent: f64,
    target: &TargetProfile,
) -> (f64, String) {
    if total_macs <= 0.0 {
        return (
            1.0,
            "No Conv/FC MAC estimate, so INT8 speedup is not modeled.".to_string(),
        );
    }
    let compute_ops = ops
        .iter()
        .filter(|op| is_mac_bearing_compute_op(&op.name))
        .collect::<Vec<_>>();
    let quantized_compute_macs = compute_ops
        .iter()
        .filter(|op| op.quantized_compute_path)
        .map(|op| op.macs)
        .sum::<f64>();
    // IEEE 754 -0.0 guard: add 0.0 to normalize negative zero produced by any intermediate arithmetic.
    let quantized_ratio = ((quantized_compute_macs / total_macs).clamp(0.0, 1.0) + 0.0).max(0.0);
    let target_speedup = target
        .int8_speedup_estimate
        .min(target.fp32_compute_factor.max(1.0))
        .max(1.0);

    // Compute-op MAC totals used for potential speedup (FP32 path). Zero-MAC
    // runtime overhead is intentionally not modeled here; the reported value is
    // a compute-kernel ceiling, not an end-to-end latency estimate.
    let compute_mac_total: f64 = compute_ops.iter().map(|op| op.macs).sum();
    if compute_mac_total <= 0.0 {
        return (
            1.0,
            "No Conv/FC MAC estimate, so INT8 compute-kernel ceiling is not modeled.".to_string(),
        );
    }

    // Helper: per-op effective speedup given bound guess and op family.
    let op_effective_speedup = |op: &&OpInfo| -> f64 {
        let bound_factor = match op.static_bound_guess.as_str() {
            "compute-bound" => 1.0,
            "mixed" => 0.72,
            "memory-bound" => 0.42,
            _ => 0.55,
        };
        let op_factor = match op.name.as_str() {
            "FULLY_CONNECTED" => 0.65,
            "DEPTHWISE_CONV_2D" => 0.72,
            _ => 1.0,
        };
        (1.0 + (target_speedup - 1.0) * bound_factor * op_factor).max(1.0)
    };

    let total_ops = ops.len().max(1) as f64;
    let memory_bound_ratio = ops
        .iter()
        .filter(|op| op.static_bound_guess == "memory-bound")
        .count() as f64
        / total_ops;
    let tensor_quant_ratio = if tensor_count > 0 {
        (quantized_tensors as f64 / tensor_count as f64).max(0.0)
    } else {
        0.0
    };
    // Fully FP32 model: report potential speedup from converting to INT8 rather
    // than the current Amdahl ratio (which would be ~1.00x for 0% quantized, misleadingly
    // implying INT8 gives no benefit).
    if quantized_ratio < 0.01 && compute_mac_total > 0.0 {
        let mut pot_norm_time = 0.0;
        for op in &compute_ops {
            let mac_share = (op.macs / compute_mac_total).max(0.0);
            pot_norm_time += mac_share / op_effective_speedup(op);
        }
        let potential_speedup = if pot_norm_time > 0.0 {
            (1.0 / pot_norm_time).max(1.0)
        } else {
            1.0
        };
        let compute_pct = (compute_mac_total / total_macs * 100.0).max(0.0);
        return (
            potential_speedup,
            format!(
                "FP32 model: current INT8 speedup is 1.00x because 0.0% of Conv/FC MACs run on an 8-bit path. \
                 If fully quantized to INT8, the MAC-weighted compute-kernel ceiling is ~{:.1}x on {}. \
                 Method: compute-kernel ceiling = 1 / sum(compute_mac_share / op_effective_speedup), where compute_mac_share is normalized over Conv/FC MACs only. \
                 op_effective_speedup = 1 + (target_INT8_vs_FP32 {:.1}x - 1) * intensity_discount * op_family_discount. \
                 The intensity discount carries the conservative payload-width effect for low-intensity ops, so peak ratio and 4x byte-width reduction are not multiplied together. \
                 Non-compute runtime share and fallback runtime penalty are not modeled without runtime evidence. \
                 Inputs: {:.1}% of MACs are in quantizable compute ops and {:.1}% of ops are memory/copy traffic candidates. \
                 Quantize with full-integer PTQ or QAT to test whether this opportunity is realizable.",
                potential_speedup,
                target.label,
                target_speedup,
                compute_pct,
                memory_bound_ratio * 100.0,
            ),
        );
    }

    // Partially or fully quantized model: standard Amdahl estimate.
    let mut normalized_time = 0.0;
    for op in compute_ops {
        let mac_share = (op.macs / compute_mac_total).max(0.0);
        if !op.quantized_compute_path {
            normalized_time += mac_share;
            continue;
        }
        normalized_time += mac_share / op_effective_speedup(&op);
    }
    let adjusted_speedup = if normalized_time > 0.0 {
        (1.0 / normalized_time).max(1.0)
    } else {
        1.0
    };
    (
        adjusted_speedup,
        format!(
            "MAC-weighted compute-kernel ceiling: speedup = 1 / sum(compute_mac_share / op_effective_speedup), with compute_mac_share normalized over Conv/FC MACs only. \
             op_effective_speedup = 1 + (target_INT8_vs_FP32 {:.2}x - 1) * intensity_discount * op_family_discount. \
             {:.1}% of Conv/FC MACs have 8-bit activation I/O, {:.1}% of tensors carry quant metadata (these ratios are intentionally different). \
             Non-compute runtime share and fallback runtime penalty are not modeled without runtime evidence. \
             FC/depthwise and low-intensity ops are discounted; {:.1}% of ops are memory/copy traffic candidates; fallback traffic {:.1}% of static bytes on {} is reported separately from this ceiling.",
            target_speedup,
            quantized_ratio * 100.0,
            tensor_quant_ratio * 100.0,
            memory_bound_ratio * 100.0,
            (fallback_byte_percent * 100.0).max(0.0),
            target.label,
        ),
    )
}

fn annotate_fusion(ops: &mut [OpInfo]) {
    for i in 0..ops.len() {
        let fused = ops[i].fused_activation != "NONE";
        if fused {
            ops[i].fusion_status = "fused confirmed".to_string();
            ops[i].fusion_detail = format!(
                "builtin_options.fused_activation_function={}",
                ops[i].fused_activation
            );
            continue;
        }
        if matches!(ops[i].name.as_str(), "ADD") {
            ops[i].fusion_status = "fusion not applicable".to_string();
            ops[i].fusion_detail = "ADD is usually a residual/SE merge unless immediately followed by a standalone activation.".to_string();
            continue;
        }
        if ops[i].name == "PAD"
            && ops.get(i + 1).is_some_and(|consumer| {
                matches!(consumer.name.as_str(), "CONV_2D" | "DEPTHWISE_CONV_2D")
                    && ops_are_directly_connected(&ops[i], consumer)
            })
        {
            ops[i].fusion_status = "runtime folding unobserved".to_string();
            ops[i].fusion_detail = "The static ledger prices PAD as a standalone logical pass. A runtime may fold equivalent padding into the directly connected convolution kernel; without an execution plan this cost is conditional and must not be claimed as measured standalone latency.".to_string();
            continue;
        }
        if is_fusable_producer(&ops[i].name) && next_is_direct_activation(ops, i) {
            ops[i].fusion_status = "fusion review needed".to_string();
            ops[i].fusion_detail = "A standalone activation follows this op; converter/export settings may have missed a fusion pattern.".to_string();
            continue;
        }
        if is_fusable_producer(&ops[i].name) {
            ops[i].fusion_status = "no fused activation".to_string();
            ops[i].fusion_detail = "No activation fusion is encoded on this op.".to_string();
        } else {
            ops[i].fusion_status = "fusion not applicable".to_string();
            ops[i].fusion_detail =
                "This op is not an activation-fusion producer in the TFLite schema.".to_string();
        }
    }
}

fn is_fusable_producer(name: &str) -> bool {
    matches!(
        name,
        "CONV_2D"
            | "DEPTHWISE_CONV_2D"
            | "FULLY_CONNECTED"
            | "AVERAGE_POOL_2D"
            | "MAX_POOL_2D"
            | "ADD"
            | "MUL"
            | "SUB"
            | "DIV"
            | "CONCATENATION"
    )
}

fn next_is_direct_activation(ops: &[OpInfo], index: usize) -> bool {
    let Some(current) = ops.get(index) else {
        return false;
    };
    let Some(next) = ops.get(index + 1) else {
        return false;
    };
    if !matches!(
        next.name.as_str(),
        "RELU" | "RELU6" | "RELU_N1_TO_1" | "TANH"
    ) {
        return false;
    }
    current
        .outputs
        .iter()
        .any(|out| next.inputs.iter().any(|input| input == out))
}

fn detect_patterns(ops: &[OpInfo], tensors: &[TensorInfo]) -> Vec<PatternInfo> {
    let mut patterns = Vec::<PatternInfo>::new();
    let mut mbconv_covered_positions = HashSet::<usize>::new();

    for i in 0..ops.len() {
        if matches_sequence(
            ops,
            i,
            &[
                "MEAN",
                "FULLY_CONNECTED",
                "FULLY_CONNECTED",
                "EXPAND_DIMS",
                "EXPAND_DIMS",
            ],
        ) && ops
            .get(i + 5)
            .map(|op| matches!(op.name.as_str(), "ADD" | "MUL"))
            .unwrap_or(false)
        {
            patterns.push(PatternInfo {
                name: "SE block".to_string(),
                first_op: ops[i].index,
                last_op: ops[i + 5].index,
                op_count: 6,
                summary: "MEAN -> FC -> FC -> EXPAND_DIMS -> EXPAND_DIMS -> scale/merge sequence"
                    .to_string(),
            });
        }
        if is_mbconv_block(ops, tensors, i) {
            patterns.push(PatternInfo {
                name: "MBConv-like block".to_string(),
                first_op: ops[i].index,
                last_op: ops[i + 3].index,
                op_count: 4,
                summary: "1x1 expansion, depthwise spatial, 1x1 projection with residual ADD"
                    .to_string(),
            });
            mbconv_covered_positions.extend(i..=i + 3);
        }
        if is_depthwise_separable_block(ops, tensors, i)
            && !mbconv_covered_positions.contains(&i)
            && !mbconv_covered_positions.contains(&(i + 1))
        {
            patterns.push(PatternInfo {
                name: "Depthwise-separable convolution pair".to_string(),
                first_op: ops[i].index,
                last_op: ops[i + 1].index,
                op_count: 2,
                summary: "depthwise spatial convolution followed by 1x1 pointwise projection"
                    .to_string(),
            });
        }
        if matches_sequence(ops, i, &["RESIZE_BILINEAR", "CONCATENATION", "CONV_2D"])
            || matches_sequence(ops, i, &["RESIZE_BILINEAR", "ADD", "CONV_2D"])
        {
            patterns.push(PatternInfo {
                name: "FPN/upsample merge".to_string(),
                first_op: ops[i].index,
                last_op: ops[i + 2].index,
                op_count: 3,
                summary: "upsample -> lateral merge -> convolution pattern".to_string(),
            });
        }
    }
    patterns
}

fn is_mbconv_block(ops: &[OpInfo], tensors: &[TensorInfo], start: usize) -> bool {
    if !matches_sequence(ops, start, &["CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D"]) {
        return false;
    }
    let expand = &ops[start];
    let depthwise = &ops[start + 1];
    let project = &ops[start + 2];
    if !ops_are_directly_connected(expand, depthwise)
        || !ops_are_directly_connected(depthwise, project)
        || !conv_kernel_is_1x1(expand, tensors)
        || !conv_kernel_is_1x1(project, tensors)
    {
        return false;
    }
    let input_channels = op_input_channels(expand, tensors).unwrap_or(0);
    let expanded_channels = op_output_channels(expand).unwrap_or(0);
    let projected_channels = op_output_channels(project).unwrap_or(0);
    let has_expansion = input_channels > 0 && expanded_channels > input_channels;
    let has_projection = projected_channels > 0 && projected_channels < expanded_channels;
    has_expansion
        && has_projection
        && has_residual_add_after_projection(ops, start + 2, expand.inputs.first().copied())
}

fn is_depthwise_separable_block(ops: &[OpInfo], tensors: &[TensorInfo], start: usize) -> bool {
    if !matches_sequence(ops, start, &["DEPTHWISE_CONV_2D", "CONV_2D"]) {
        return false;
    }
    let depthwise = &ops[start];
    let pointwise = &ops[start + 1];
    ops_are_directly_connected(depthwise, pointwise) && conv_kernel_is_1x1(pointwise, tensors)
}

fn ops_are_directly_connected(producer: &OpInfo, consumer: &OpInfo) -> bool {
    producer
        .outputs
        .iter()
        .any(|output| consumer.inputs.iter().any(|input| input == output))
}

fn conv_kernel_is_1x1(op: &OpInfo, tensors: &[TensorInfo]) -> bool {
    let Some(weight_idx) = op.inputs.get(1).copied().filter(|idx| *idx >= 0) else {
        return false;
    };
    let Some(weight) = tensors.get(weight_idx as usize) else {
        return false;
    };
    weight.shape.len() == 4 && weight.shape[1] == 1 && weight.shape[2] == 1
}

fn op_input_channels(op: &OpInfo, tensors: &[TensorInfo]) -> Option<i32> {
    op.inputs
        .first()
        .copied()
        .filter(|idx| *idx >= 0)
        .and_then(|idx| tensors.get(idx as usize))
        .and_then(|tensor| tensor.shape.last().copied())
}

fn op_output_channels(op: &OpInfo) -> Option<i32> {
    op.output_shapes
        .iter()
        .find(|shape| shape.len() == 4)
        .and_then(|shape| shape.last().copied())
}

fn has_residual_add_after_projection(
    ops: &[OpInfo],
    project_position: usize,
    block_input_tensor: Option<i32>,
) -> bool {
    let Some(block_input_tensor) = block_input_tensor else {
        return false;
    };
    let Some(project) = ops.get(project_position) else {
        return false;
    };
    let Some(project_output) = project.outputs.first().copied() else {
        return false;
    };
    let Some(next) = ops.get(project_position + 1) else {
        return false;
    };
    next.name == "ADD"
        && next.inputs.contains(&project_output)
        && next.inputs.contains(&block_input_tensor)
}

fn matches_sequence(ops: &[OpInfo], start: usize, names: &[&str]) -> bool {
    if start + names.len() > ops.len() {
        return false;
    }
    names
        .iter()
        .enumerate()
        .all(|(offset, name)| ops[start + offset].name == *name)
}

fn annotate_patterns(ops: &mut [OpInfo], patterns: &[PatternInfo]) {
    for pattern in patterns {
        for op in ops
            .iter_mut()
            .filter(|op| op.index >= pattern.first_op && op.index <= pattern.last_op)
        {
            if !op.patterns.contains(&pattern.name) {
                op.patterns.push(pattern.name.clone());
            }
        }
    }
}

fn annotate_xnnpack_chains(ops: &mut [OpInfo], target: &TargetProfile) -> Vec<XnnpackChainInfo> {
    let mut chains = Vec::<XnnpackChainInfo>::new();
    let mut current: Option<XnnpackChainInfo> = None;
    let mut last_supported_seen = false;
    let mut pending_break_index: Option<usize> = None;
    let mut chain_id = 0_i32;

    for index in 0..ops.len() {
        if ops[index].xnnpack_supported {
            if current.is_none() {
                if let Some(break_index) = pending_break_index.take() {
                    if let Some(op) = ops.get_mut(break_index) {
                        op.xnnpack_chain_break = true;
                        op.xnnpack_chain_role = "chain-break".to_string();
                        op.chain_break_overhead_us_low = target.chain_break_overhead_us_low;
                        op.chain_break_overhead_us_high = target.chain_break_overhead_us_high;
                    }
                }
                current = Some(XnnpackChainInfo {
                    id: chain_id,
                    first_op: ops[index].index,
                    last_op: ops[index].index,
                    op_count: 0,
                    macs: 0.0,
                    mac_percent: 0.0,
                    chain_class: "unclassified".to_string(),
                    target_hint: target.xnnpack_kernel_family.clone(),
                });
                chain_id += 1;
            }
            let chain = current.as_mut().expect("chain exists");
            chain.last_op = ops[index].index;
            chain.op_count += 1;
            chain.macs += ops[index].macs;
            ops[index].xnnpack_chain_id = chain.id;
            ops[index].xnnpack_chain_role = "delegated".to_string();
            last_supported_seen = true;
        } else {
            if let Some(chain) = current.take() {
                chains.push(chain);
            }
            if last_supported_seen && pending_break_index.is_none() {
                pending_break_index = Some(index);
                ops[index].xnnpack_chain_break = true;
                ops[index].xnnpack_chain_role = "chain-break".to_string();
                ops[index].chain_break_overhead_us_low = target.chain_break_overhead_us_low;
                ops[index].chain_break_overhead_us_high = target.chain_break_overhead_us_high;
            } else {
                ops[index].xnnpack_chain_role = "fallback".to_string();
            }
        }
    }
    if let Some(chain) = current {
        chains.push(chain);
    }
    chains
}

fn compute_predicted_partition_boundary_inventory(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> PredictedPartitionBoundaryInventory {
    let mut producer_by_tensor = HashMap::<i32, &OpInfo>::new();
    for op in ops {
        for &tensor_index in &op.outputs {
            if tensor_index >= 0 {
                producer_by_tensor.insert(tensor_index, op);
            }
        }
    }

    let mut edges = Vec::<PredictedPartitionBoundaryEdge>::new();
    for consumer in ops {
        let mut seen_inputs = HashSet::<i32>::new();
        for &tensor_index in &consumer.inputs {
            if tensor_index < 0 || !seen_inputs.insert(tensor_index) {
                continue;
            }
            let Some(producer) = producer_by_tensor.get(&tensor_index).copied() else {
                continue;
            };
            let producer_domain = predicted_execution_domain(producer);
            let consumer_domain = predicted_execution_domain(consumer);
            if producer_domain == consumer_domain {
                continue;
            }
            let Some(tensor) = tensors.get(tensor_index as usize) else {
                continue;
            };
            if tensor.constant_buffer {
                continue;
            }
            let payload = deterministic_tensor_payload_assessment(tensor);
            edges.push(PredictedPartitionBoundaryEdge {
                tensor_index: tensor.index,
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                payload_bytes: payload.bytes,
                payload_status: payload.status.to_string(),
                payload_binding: payload.binding.to_string(),
                payload_reason: payload.reason,
                producer_op_index: producer.index,
                producer_op_name: producer.name.clone(),
                producer_domain,
                consumer_op_index: consumer.index,
                consumer_op_name: consumer.name.clone(),
                consumer_domain,
                direction: predicted_boundary_direction(producer, consumer).to_string(),
                materialization_status: "NOT_ASSESSABLE_FROM_STATIC_ARTIFACT".to_string(),
            });
        }
    }
    edges.sort_by_key(|edge| {
        (
            edge.producer_op_index,
            edge.consumer_op_index,
            edge.tensor_index,
        )
    });

    let assessed_payload_edge_count = edges
        .iter()
        .filter(|edge| edge.payload_bytes.is_some())
        .count();
    let unassessed_payload_edge_count = edges.len().saturating_sub(assessed_payload_edge_count);
    let assessed_edge_payload_bytes = edges.iter().filter_map(|edge| edge.payload_bytes).sum();
    let mut unique_tensor_payloads = BTreeMap::<usize, Option<usize>>::new();
    for edge in &edges {
        unique_tensor_payloads
            .entry(edge.tensor_index)
            .or_insert(edge.payload_bytes);
    }
    let unique_tensor_count = unique_tensor_payloads.len();
    let assessed_unique_tensor_payload_bytes = unique_tensor_payloads
        .values()
        .filter_map(|value| *value)
        .sum();
    let all_payloads_assessed = unassessed_payload_edge_count == 0;
    let all_unique_payloads_assessed = unique_tensor_payloads.values().all(Option::is_some);

    let payload_bindings = edges
        .iter()
        .filter(|edge| edge.payload_bytes.is_some())
        .map(|edge| edge.payload_binding.as_str())
        .collect::<BTreeSet<_>>();
    let payload_binding = if !all_payloads_assessed {
        "partial"
    } else if payload_bindings.is_empty() {
        "none"
    } else if payload_bindings.len() == 1 {
        payload_bindings.iter().next().copied().unwrap_or("none")
    } else {
        "mixed"
    };

    PredictedPartitionBoundaryInventory {
        schema: "deepbom.predicted_partition_boundary_edges.v1.1".to_string(),
        status: "assessed".to_string(),
        assignment_evidence_class: "PREDICTED".to_string(),
        payload_evidence_class: "DERIVED".to_string(),
        edge_count: edges.len(),
        unique_tensor_count,
        assessed_payload_edge_count,
        unassessed_payload_edge_count,
        payload_coverage_status: if all_payloads_assessed { "complete" } else { "partial" }
            .to_string(),
        payload_binding: payload_binding.to_string(),
        assessed_edge_payload_bytes,
        summed_edge_payload_bytes: all_payloads_assessed.then_some(assessed_edge_payload_bytes),
        assessed_unique_tensor_payload_bytes,
        unique_tensor_payload_bytes: all_unique_payloads_assessed
            .then_some(assessed_unique_tensor_payload_bytes),
        edges,
        interpretation_boundary: "Internal producer-to-consumer edges whose static predicted execution domains differ. Logical tensor payload is derived from a fixed serialized shape and fixed-width dtype. A serialized_batch1_projection binding means the shape signature leaves only batch dynamic while the artifact's declared shape binds batch to one; it is not a claim for other batch sizes. Runtime copy materialization, aliasing, layout conversion, allocator behavior, and latency remain unconfirmed.".to_string(),
    }
}

fn predicted_execution_domain(op: &OpInfo) -> String {
    if op.xnnpack_chain_id >= 0 {
        format!("XNNPACK:C{}", op.xnnpack_chain_id)
    } else {
        "TFLITE_CPU".to_string()
    }
}

fn predicted_boundary_direction(producer: &OpInfo, consumer: &OpInfo) -> &'static str {
    match (
        producer.xnnpack_chain_id >= 0,
        consumer.xnnpack_chain_id >= 0,
    ) {
        (true, false) => "delegate_to_cpu",
        (false, true) => "cpu_to_delegate",
        (true, true) => "delegate_partition_to_delegate_partition",
        (false, false) => "cpu_to_cpu",
    }
}

fn deterministic_tensor_payload_bytes(tensor: &TensorInfo) -> Result<usize, String> {
    let assessment = deterministic_tensor_payload_assessment(tensor);
    assessment.bytes.ok_or(assessment.reason)
}

fn deterministic_tensor_payload_assessment(tensor: &TensorInfo) -> TensorPayloadAssessment {
    if tensor.shape.iter().any(|dim| *dim < 0) {
        return TensorPayloadAssessment {
            bytes: None,
            status: "not_assessed",
            binding: "unbound",
            reason: "The serialized tensor shape contains a dynamic or unknown dimension."
                .to_string(),
        };
    }
    let signature = &tensor.shape_signature;
    let binding = if signature.is_empty()
        || (signature.len() == tensor.shape.len()
            && signature
                .iter()
                .zip(tensor.shape.iter())
                .all(|(declared, serialized)| *declared >= 0 && declared == serialized))
    {
        "static"
    } else {
        let serialized_batch_one = signature.len() == tensor.shape.len()
            && signature.first() == Some(&-1)
            && tensor.shape.first() == Some(&1)
            && signature
                .iter()
                .zip(tensor.shape.iter())
                .skip(1)
                .all(|(declared, serialized)| *declared >= 0 && declared == serialized);
        if !serialized_batch_one {
            return TensorPayloadAssessment {
                bytes: None,
                status: "not_assessed",
                binding: "unbound",
                reason: "A non-batch dynamic dimension, multiple dynamic dimensions, or a signature/serialized-shape mismatch prevents a deterministic payload projection.".to_string(),
            };
        }
        "serialized_batch1_projection"
    };
    match declared_tensor_payload_bytes(tensor) {
        Ok(bytes) => TensorPayloadAssessment {
            bytes: Some(bytes),
            status: if binding == "static" {
                "assessed_static"
            } else {
                "assessed_serialized_batch1"
            },
            binding,
            reason: if binding == "static" {
                "Static tensor shape and fixed-width dtype determine the logical payload."
                    .to_string()
            } else {
                "Only batch is dynamic in the shape signature; the artifact's serialized batch=1 shape and fixed-width dtype determine this projection.".to_string()
            },
        },
        Err(reason) => TensorPayloadAssessment {
            bytes: None,
            status: "not_assessed",
            binding: "unbound",
            reason,
        },
    }
}

fn xnnpack_chain_class(chain: &XnnpackChainInfo) -> String {
    if chain.macs == 0.0 {
        "zero-MAC candidate segment".to_string()
    } else if chain.mac_percent >= MEANINGFUL_CHAIN_MAC_PERCENT {
        "high-MAC-share candidate segment".to_string()
    } else {
        "low-MAC-share candidate segment".to_string()
    }
}

fn annotate_chain_break_impact(
    ops: &mut [OpInfo],
    chains: &[XnnpackChainInfo],
    total_macs: f64,
    total_estimated_bytes: f64,
) {
    for op in ops.iter_mut() {
        if !op.xnnpack_chain_break {
            continue;
        }
        let prev_chain = chains
            .iter()
            .filter(|chain| chain.last_op < op.index)
            .max_by_key(|chain| chain.last_op);
        let next_chain = chains
            .iter()
            .filter(|chain| chain.first_op > op.index)
            .min_by_key(|chain| chain.first_op);
        let adjacent_macs = prev_chain
            .map(|chain| chain.macs)
            .unwrap_or(0.0)
            .max(next_chain.map(|chain| chain.macs).unwrap_or(0.0));
        let adjacent_mac_percent = if total_macs > 0.0 {
            adjacent_macs / total_macs
        } else {
            0.0
        };
        let fallback_byte_percent = if total_estimated_bytes > 0.0 {
            op.estimated_bytes / total_estimated_bytes
        } else {
            0.0
        };
        let break_class = if adjacent_mac_percent >= MEANINGFUL_CHAIN_MAC_PERCENT {
            "high-adjacent-mac-exposure"
        } else if fallback_byte_percent >= MEANINGFUL_FALLBACK_BYTE_PERCENT {
            "memory-traffic"
        } else if op.macs == 0.0 && is_structural_view_op(&op.name) {
            "structural-zero-mac"
        } else if op.macs == 0.0 {
            "zero-modeled-mac-nonstructural"
        } else {
            "low-impact-nonstructural"
        };
        op.xnnpack_break_class = break_class.to_string();
        op.chain_break_impact_mac_percent = adjacent_mac_percent;
        if break_class == "structural-zero-mac" {
            op.chain_break_overhead_us_low *= 0.35;
            op.chain_break_overhead_us_high *= 0.45;
        }
    }
}

fn is_structural_view_op(name: &str) -> bool {
    matches!(name, "RESHAPE" | "SQUEEZE" | "EXPAND_DIMS" | "SHAPE")
}

fn fallback_traffic_by_family(
    ops: &[OpInfo],
    total_estimated_bytes: f64,
    total_macs: f64,
) -> Vec<TrafficItem> {
    let mut map = BTreeMap::<String, (usize, f64, f64)>::new();
    for op in ops
        .iter()
        .filter(|op| op.xnnpack_chain_id < 0 && op.estimated_bytes > 0.0)
    {
        let entry = map.entry(op.name.clone()).or_insert((0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += op.estimated_bytes;
        entry.2 += op.macs;
    }
    let mut items = map
        .into_iter()
        .map(|(name, (count, estimated_bytes, macs))| TrafficItem {
            name,
            count,
            estimated_bytes,
            byte_percent: if total_estimated_bytes > 0.0 {
                estimated_bytes / total_estimated_bytes
            } else {
                0.0
            },
            macs,
            mac_percent: if total_macs > 0.0 {
                macs / total_macs
            } else {
                0.0
            },
        })
        .collect::<Vec<_>>();
    items.sort_by(|a, b| {
        b.estimated_bytes
            .partial_cmp(&a.estimated_bytes)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    items
}

fn build_recommendations(
    ops: &[OpInfo],
    target: &TargetProfile,
    xnnpack_chain_breaks: usize,
    xnnpack_effective_chain_breaks: usize,
    fallback_byte_percent: f64,
    fallback_traffic_by_op_family: &[TrafficItem],
    patterns: &[PatternInfo],
) -> Vec<Recommendation> {
    let mut items = Vec::<Recommendation>::new();

    if let Some(op) = ops
        .iter()
        .filter(|op| op.quant_risk == "risk" || op.quant_risk == "warn")
        .max_by(|a, b| {
            let score_a = a.quant_scale_ratio.max(1.0).log10()
                + a.quant_scale_cv * 1.5
                + a.quant_zero_point_offset as f64 / 32.0;
            let score_b = b.quant_scale_ratio.max(1.0).log10()
                + b.quant_scale_cv * 1.5
                + b.quant_zero_point_offset as f64 / 32.0;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        items.push(Recommendation {
            priority: 1,
            tone: op.quant_risk.clone(),
            title: format!("#{:03} {} quant numerical contract", op.index, op.name),
            detail: format!(
                "{}; this is a numerical/accuracy-contract signal, not a recoverable-latency estimate. Inspect the named kernel or activation tensor, compare against the source checkpoint, and re-export or retrain the affected layer family. Weight-scale spread alone does not justify an activation PTQ-calibration remedy.",
                op.quant_risk_detail
            ),
            op_index: op.index as i32,
        });
    }

    if xnnpack_chain_breaks > 0 {
        let break_low_us = ops
            .iter()
            .filter(|op| op.xnnpack_chain_break)
            .map(|op| op.chain_break_overhead_us_low)
            .sum::<f64>();
        let break_midpoint_us = ops.iter().map(|op| op.bottleneck_break_us).sum::<f64>();
        let break_high_us = ops
            .iter()
            .filter(|op| op.xnnpack_chain_break)
            .map(|op| op.chain_break_overhead_us_high)
            .sum::<f64>();
        let first_break = ops
            .iter()
            .filter(|op| {
                matches!(
                    op.xnnpack_break_class.as_str(),
                    "high-adjacent-mac-exposure"
                        | "memory-traffic"
                        | "zero-modeled-mac-nonstructural"
                        | "low-impact-nonstructural"
                )
            })
            .max_by(|a, b| {
                (a.chain_break_impact_mac_percent + a.fallback_byte_percent)
                    .partial_cmp(&(b.chain_break_impact_mac_percent + b.fallback_byte_percent))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .or_else(|| ops.iter().find(|op| op.xnnpack_chain_break));
        let break_summary = first_break
            .map(|op| {
                format!(
                    "top break at #{:03} {} class={} adjacent_chain_macs={:.2}% fallback_bytes={:.2}% ({})",
                    op.index,
                    op.name,
                    op.xnnpack_break_class,
                    op.chain_break_impact_mac_percent * 100.0,
                    op.fallback_byte_percent * 100.0,
                    op.xnnpack_reason
                )
            })
            .unwrap_or_else(|| "break locations detected between delegated chains".to_string());
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: if xnnpack_effective_chain_breaks > 0 { "warn" } else { "neutral" }.to_string(),
            title: format!(
                "XNNPACK predicted partition breaks: {} total / {} non-structural",
                xnnpack_chain_breaks, xnnpack_effective_chain_breaks
            ),
            detail: format!(
                "{}; setup-only planning-profile range {:.1}-{:.1} us, midpoint {:.1} us. This constant is independent of logical boundary payload and is not measured copy or launch latency. Structural/view boundaries and zero-MAC non-structural operators are classified separately.",
                break_summary, break_low_us, break_high_us, break_midpoint_us
            ),
            op_index: first_break.map(|op| op.index as i32).unwrap_or(-1),
        });
    }

    if fallback_byte_percent >= 0.02 {
        let modeled_fallback_us = ops.iter().map(|op| op.bottleneck_fallback_us).sum::<f64>();
        let top = fallback_traffic_by_op_family.first();
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: if fallback_byte_percent >= 0.06 { "risk" } else { "warn" }.to_string(),
            title: format!("Fallback tensor traffic {:.1}% of static bytes", fallback_byte_percent * 100.0),
            detail: top
                .map(|item| {
                    format!(
                        "Top family {} contributes {} across {} ops ({:.1}% of total static bytes); modeled fallback-traffic component {:.1} us. Prioritize fallback traffic by family before optimizing zero-MAC break count.",
                        item.name,
                        fmt_bytes(item.estimated_bytes),
                        item.count,
                        item.byte_percent * 100.0,
                        modeled_fallback_us,
                    )
                })
                .unwrap_or_else(|| "Fallback traffic is elevated; inspect unsupported op families and runtime partition logs.".to_string()),
            op_index: -1,
        });
    }

    if let Some(op) = ops
        .iter()
        .filter(|op| cache_working_set_is_watch(op.row_working_set_ratio))
        .max_by(|a, b| {
            a.row_working_set_bytes
                .partial_cmp(&b.row_working_set_bytes)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: "warn".to_string(),
            title: format!(
                "#{:03} {} {} {} L1 row budget",
                op.index,
                op.name,
                if op.row_working_set_ratio > 1.0 {
                    "exceeds"
                } else {
                    "approaches"
                },
                target.label
            ),
            detail: format!(
                "Naive row working set {} vs target L1 {} ({:.2}x); watch begins at {:.2}x because code, other live data, associativity, and runtime scratch also consume cache. Validate executed tiling before changing the graph.",
                fmt_bytes(op.row_working_set_bytes),
                fmt_bytes(target.l1_data_bytes as f64),
                op.row_working_set_ratio,
                L1_WORKING_SET_WATCH_RATIO
            ),
            op_index: op.index as i32,
        });
    }

    let fusion_reviews = ops
        .iter()
        .filter(|op| op.fusion_status == "fusion review needed")
        .count();
    if fusion_reviews > 0 {
        let first = ops
            .iter()
            .find(|op| op.fusion_status == "fusion review needed");
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: "warn".to_string(),
            title: format!("Fusion review candidates: {}", fusion_reviews),
            detail: first
                .map(|op| format!("Start at #{:03} {}; standalone activation may be foldable during conversion.", op.index, op.name))
                .unwrap_or_else(|| "Review converter fusion patterns.".to_string()),
            op_index: first.map(|op| op.index as i32).unwrap_or(-1),
        });
    }

    if let Some(op) = ops
        .iter()
        .filter(|op| op.channel_alignment_status == "misaligned")
        .max_by(|a, b| {
            a.channel_tail_overhead_percent
                .partial_cmp(&b.channel_tail_overhead_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.index.cmp(&a.index))
        })
    {
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: if op.channel_tail_overhead_percent >= 1.0 { "risk" } else { "warn" }.to_string(),
            title: format!("#{:03} {} channel tail not aligned", op.index, op.name),
            detail: format!("{}; prefer channel counts that are multiples of {} for this target when model quality allows.", op.channel_alignment_detail, op.channel_alignment_multiple),
            op_index: op.index as i32,
        });
    }

    let mut packing_warns = ops
        .iter()
        .filter(|op| op.weight_packing_risk == "warn")
        .collect::<Vec<_>>();
    packing_warns.sort_by(|a, b| {
        b.weight_packing_overhead_us
            .partial_cmp(&a.weight_packing_overhead_us)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.index.cmp(&b.index))
    });
    if let Some(op) = packing_warns.first() {
        let warning_packing_us = packing_warns
            .iter()
            .map(|candidate| candidate.weight_packing_overhead_us)
            .sum::<f64>();
        let all_op_packing_us = ops
            .iter()
            .map(|candidate| candidate.weight_packing_overhead_us)
            .sum::<f64>();
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: "warn".to_string(),
            title: format!("Weight packing watchlist: {} ops", packing_warns.len()),
            detail: format!(
                "{} op(s) at or above the single {:.1} us watch threshold contribute {:.1} us; the all-op one-time packing ledger is {:.1} us, including sub-threshold rows. Largest warning #{:03} {} {:.1} us. {}; warm up before measuring Invoke latency and report first-run compile/packing separately from steady-state p50.",
                packing_warns.len(),
                PACKING_WARN_OVERHEAD_US,
                warning_packing_us,
                all_op_packing_us,
                op.index,
                op.name,
                op.weight_packing_overhead_us,
                op.weight_packing_detail
            ),
            op_index: op.index as i32,
        });
    }

    let reduce_reexpand_pairs = ops
        .windows(2)
        .filter(|pair| {
            pair[0].name == "MEAN"
                && pair[1].name == "EXPAND_DIMS"
                && ops_are_directly_connected(&pair[0], &pair[1])
        })
        .collect::<Vec<_>>();
    if let Some(first_pair) = reduce_reexpand_pairs.first() {
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: "neutral".to_string(),
            title: format!(
                "Reduce-then-reexpand structural pairs: {}",
                reduce_reexpand_pairs.len()
            ),
            detail: format!(
                "First direct MEAN -> EXPAND_DIMS pair is #{:03} -> #{:03}. Review whether MEAN keep_dims=true can preserve the same serialized output shape and quantization contract, then re-run delegation analysis. No latency recovery is claimed without runtime assignment.",
                first_pair[0].index, first_pair[1].index
            ),
            op_index: first_pair[0].index as i32,
        });
    }

    let se_count = patterns
        .iter()
        .filter(|pattern| pattern.name == "SE block")
        .count();
    if se_count > 0 {
        items.push(Recommendation {
            priority: items.len() + 1,
            tone: "neutral".to_string(),
            title: format!("Detected SE blocks: {}", se_count),
            detail: "SE blocks are valid mobile patterns, but EXPAND_DIMS/reshape edges often create predicted delegate-partition pressure; inspect partition-break badges around them.".to_string(),
            op_index: patterns
                .iter()
                .find(|pattern| pattern.name == "SE block")
                .map(|pattern| pattern.first_op as i32)
                .unwrap_or(-1),
        });
    }

    if items.is_empty() {
        items.push(Recommendation {
            priority: 1,
            tone: "good".to_string(),
            title: "No high-priority static action detected".to_string(),
            detail: "Proceed to target-device profiling with benchmark_model/simpleperf and compare delegate coverage against this static preflight.".to_string(),
            op_index: -1,
        });
    }

    items.sort_by(|a, b| {
        recommendation_modeled_impact_us(b, ops)
            .total_cmp(&recommendation_modeled_impact_us(a, ops))
            .then_with(|| recommendation_rank(b).cmp(&recommendation_rank(a)))
            .then_with(|| a.op_index.cmp(&b.op_index))
    });
    for (index, item) in items.iter_mut().enumerate() {
        item.priority = index + 1;
    }
    items.truncate(7);
    items
}

fn recommendation_modeled_impact_us(item: &Recommendation, ops: &[OpInfo]) -> f64 {
    let title = item.title.to_ascii_lowercase();
    if title.contains("chain") || title.contains("boundar") {
        return ops.iter().map(|op| op.bottleneck_break_us).sum();
    }
    if title.contains("packing") {
        return ops.iter().map(|op| op.bottleneck_packing_us).sum();
    }
    if title.contains("fallback") {
        return ops.iter().map(|op| op.bottleneck_fallback_us).sum();
    }
    0.0
}

fn recommendation_rank(item: &Recommendation) -> i32 {
    let mut score = match item.tone.as_str() {
        "risk" => 300,
        "warn" => 200,
        "neutral" => 100,
        "good" => 0,
        _ => 50,
    };
    let title = item.title.to_ascii_lowercase();
    if title.contains("channel") {
        score += 55;
    }
    if title.contains("fallback") {
        score += 50;
    }
    if title.contains("quant") {
        score += 45;
    }
    if title.contains("chain") {
        score += 40;
    }
    if title.contains("packing") {
        score += 35;
    }
    if title.contains("l1") {
        score += 30;
    }
    score
}

fn ranges_overlap(a0: usize, a1: usize, b0: usize, b1: usize) -> bool {
    a0 <= b1 && b0 <= a1
}

fn build_stages(ops: &[OpInfo], total_macs: f64, patterns: &[PatternInfo]) -> Vec<StageInfo> {
    let mut stages = Vec::<StageInfo>::new();
    for op in ops {
        let shape = primary_shape(op);
        let key = stage_key(shape);
        let channel = if shape.len() == 4 {
            Some(shape[3])
        } else {
            None
        };
        if stages.last().map(|stage| stage.key.as_str()) != Some(key.as_str()) {
            stages.push(StageInfo {
                index: stages.len(),
                key,
                first_op: op.index,
                last_op: op.index,
                op_count: 0,
                macs: 0.0,
                mac_percent: 0.0,
                delegated_macs: 0.0,
                fallback_macs: 0.0,
                delegated_mac_percent: 0.0,
                fallback_mac_percent: 0.0,
                delegated_ops: 0,
                fallback_ops: 0,
                delegated_op_percent: 0.0,
                fallback_op_percent: 0.0,
                estimated_bytes: 0.0,
                channels: Vec::new(),
                xnnpack_chain_breaks: 0,
                patterns: Vec::new(),
            });
        }
        let stage = stages.last_mut().expect("stage exists");
        stage.last_op = op.index;
        stage.op_count += 1;
        stage.macs += op.macs;
        if op.xnnpack_chain_id >= 0 {
            stage.delegated_macs += op.macs;
            stage.delegated_ops += 1;
        } else {
            stage.fallback_macs += op.macs;
            stage.fallback_ops += 1;
        }
        stage.estimated_bytes += op.estimated_bytes;
        if op.xnnpack_chain_break {
            stage.xnnpack_chain_breaks += 1;
        }
        if let Some(c) = channel {
            if !stage.channels.contains(&c) {
                stage.channels.push(c);
            }
        }
    }
    for stage in &mut stages {
        stage.mac_percent = if total_macs > 0.0 {
            stage.macs / total_macs
        } else {
            0.0
        };
        stage.delegated_mac_percent = if stage.macs > 0.0 {
            stage.delegated_macs / stage.macs
        } else {
            0.0
        };
        stage.fallback_mac_percent = if stage.macs > 0.0 {
            stage.fallback_macs / stage.macs
        } else {
            0.0
        };
        stage.delegated_op_percent = if stage.op_count > 0 {
            stage.delegated_ops as f64 / stage.op_count as f64
        } else {
            0.0
        };
        stage.fallback_op_percent = if stage.op_count > 0 {
            stage.fallback_ops as f64 / stage.op_count as f64
        } else {
            0.0
        };
        for pattern in patterns {
            if ranges_overlap(
                stage.first_op,
                stage.last_op,
                pattern.first_op,
                pattern.last_op,
            ) {
                let label = stage_pattern_participation_label(stage, pattern, ops);
                if !stage.patterns.contains(&label) {
                    stage.patterns.push(label);
                }
            }
        }
    }
    stages
}

fn annotate_op_stages(ops: &mut [OpInfo], stages: &[StageInfo]) -> Result<(), String> {
    let mut stage_position = 0usize;
    for op in ops {
        while stage_position < stages.len() && op.index > stages[stage_position].last_op {
            stage_position += 1;
        }
        let stage = stages
            .get(stage_position)
            .filter(|stage| op.index >= stage.first_op && op.index <= stage.last_op)
            .ok_or_else(|| {
                format!(
                    "structural stage coverage is incomplete at operator #{}",
                    op.index
                )
            })?;
        op.stage_index = Some(stage.index);
        op.stage_key = Some(stage.key.clone());
    }
    Ok(())
}

fn stage_pattern_participation_label(
    stage: &StageInfo,
    pattern: &PatternInfo,
    ops: &[OpInfo],
) -> String {
    if stage.first_op <= pattern.first_op && stage.last_op >= pattern.last_op {
        return format!(
            "local {} #{:03}-{:03}",
            pattern.name, pattern.first_op, pattern.last_op
        );
    }
    let roles = ops
        .iter()
        .filter(|op| op.index >= stage.first_op && op.index <= stage.last_op)
        .filter(|op| op.index >= pattern.first_op && op.index <= pattern.last_op)
        .map(|op| pattern_operator_role(&pattern.name, op))
        .collect::<Vec<_>>();
    let role = if roles.is_empty() {
        "overlap".to_string()
    } else {
        roles.join("+")
    };
    format!(
        "cross-stage {} of {} #{:03}-{:03}",
        role, pattern.name, pattern.first_op, pattern.last_op
    )
}

fn pattern_operator_role(pattern_name: &str, op: &OpInfo) -> String {
    if pattern_name == "Depthwise-separable convolution pair" {
        return match op.name.as_str() {
            "DEPTHWISE_CONV_2D" => "depthwise half".to_string(),
            "CONV_2D" => "pointwise half".to_string(),
            _ => format!("#{:03}", op.index),
        };
    }
    if pattern_name == "MBConv-like block" {
        return match op.name.as_str() {
            "CONV_2D" => {
                if op.index == 0 {
                    "1x1 conv".to_string()
                } else {
                    "expand/project conv".to_string()
                }
            }
            "DEPTHWISE_CONV_2D" => "depthwise middle".to_string(),
            "ADD" => "residual add".to_string(),
            _ => format!("#{:03}", op.index),
        };
    }
    op.name.to_lowercase()
}

fn primary_shape(op: &OpInfo) -> &[i32] {
    for shape in &op.output_shapes {
        if shape.len() == 4 {
            return shape;
        }
    }
    op.output_shapes.first().map(Vec::as_slice).unwrap_or(&[])
}

fn stage_key(shape: &[i32]) -> String {
    if shape.len() == 4 {
        let h = shape[1];
        let w = shape[2];
        return format!("spatial/{}x{}", h, w);
    }
    if shape.len() == 3 {
        // [batch, seq_len, features] — attention/transformer layout
        return format!("seq/{}", shape[1]);
    }
    if shape.len() == 2 {
        return "vector/head".to_string();
    }
    if shape.len() == 1 {
        return "shape/vector".to_string();
    }
    if shape.is_empty() {
        return "scalar/shape".to_string();
    }
    "other".to_string()
}

fn count_items(mut counts: BTreeMap<String, usize>) -> Vec<CountItem> {
    let mut items = counts
        .iter_mut()
        .map(|(name, count)| CountItem {
            name: name.clone(),
            count: *count,
        })
        .collect::<Vec<_>>();
    items.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
    items
}

fn fmt_bytes(value: f64) -> String {
    if value >= 1024.0 * 1024.0 {
        format!("{:.1} MiB", value / (1024.0 * 1024.0))
    } else if value >= 1024.0 {
        format!("{:.1} KiB", value / 1024.0)
    } else {
        format!("{:.0} B", value)
    }
}

// ── Kernel Haar Decomposition ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn appended_buffer_location_requires_an_in_bounds_file_slice() {
        let location = bounded_buffer_location(128, 64, 256).expect("valid location");
        assert_eq!(location.offset, 128);
        assert_eq!(location.length, 64);
        assert!(bounded_buffer_location(0, 64, 256).is_none());
        assert!(bounded_buffer_location(128, 0, 256).is_none());
        assert!(bounded_buffer_location(240, 32, 256).is_none());
        assert!(bounded_buffer_location(u64::MAX, 2, 256).is_none());
    }

    #[test]
    fn buffer_table_reads_v3c_appended_offset_and_size_fields() {
        let mut bytes = vec![0u8; 128];
        bytes[0..2].copy_from_slice(&10u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&24u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&8u16.to_le_bytes());
        bytes[8..10].copy_from_slice(&16u16.to_le_bytes());
        bytes[16..20].copy_from_slice(&16i32.to_le_bytes());
        bytes[24..32].copy_from_slice(&64u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&16u64.to_le_bytes());
        let fb = Fb::new_for_test(&bytes);
        let location = read_buffer_location(&fb, 16);
        assert_eq!(location.offset, 64);
        assert_eq!(location.length, 16);
    }

    #[test]
    fn size_breakdown_counts_shared_storage_once_and_content_duplicates_separately() {
        let bytes = vec![0u8; 128];
        let fb = Fb::new_for_test(&bytes);
        let mut first = tensor_with_index_shape(0, vec![16]);
        first.constant_buffer = true;
        first.buffer_data_offset = 64;
        first.buffer_data_length = 16;
        let mut shared = first.clone();
        shared.index = 1;
        shared.name = "T1".to_string();
        let mut separately_stored_duplicate = first.clone();
        separately_stored_duplicate.index = 2;
        separately_stored_duplicate.name = "T2".to_string();
        separately_stored_duplicate.buffer_data_offset = 80;

        let size = compute_size_breakdown(
            &fb,
            0,
            &[first, shared, separately_stored_duplicate],
            &[],
            bytes.len(),
        );
        assert_eq!(size.constant_tensor_count, 3);
        assert_eq!(size.physical_constant_buffer_count, 2);
        assert_eq!(size.logical_constant_reference_bytes, 48);
        assert_eq!(size.constant_bytes, 32);
        assert_eq!(size.unique_constant_bytes, 16);
        assert_eq!(size.duplicate_constant_bytes, 16);
        assert_eq!(size.structure_overhead_bytes, 96);
        assert_eq!(
            size.constant_bytes + size.metadata_bytes + size.structure_overhead_bytes,
            size.file_size
        );
    }

    fn tensor_with_shape(shape: Vec<i32>) -> TensorInfo {
        tensor_with_index_shape(0, shape)
    }

    fn tensor_with_index_shape(index: usize, shape: Vec<i32>) -> TensorInfo {
        TensorInfo {
            index,
            name: format!("T{index}"),
            shape_signature: shape.clone(),
            shape,
            dtype: "INT8".to_string(),
            buffer_index: 0,
            buffer_data_offset: 0,
            buffer_data_length: 0,
            constant_buffer: false,
            sparse_storage: false,
            sparse_encoding: None,
            quant_scales: 0,
            quant_zero_points: 0,
            quantized_dimension: 0,
            scale_sample: Vec::new(),
            zero_point_sample: Vec::new(),
            scale_min: 0.0,
            scale_max: 0.0,
            scale_ratio: 0.0,
            scale_mean: 0.0,
            scale_stddev: 0.0,
            scale_cv: 0.0,
            zero_point_min: 0,
            zero_point_max: 0,
            zero_point_offset_max: 0,
            zero_point_status: "none".to_string(),
            zero_point_detail: "none".to_string(),
            scale_mode: "none".to_string(),
            scale_ratio_meaningful: false,
            quant_risk: "none".to_string(),
            buffer_hash: String::new(),
            is_variable: false,
        }
    }

    #[test]
    fn conv_kernel_class_requires_complete_static_gemm_geometry() {
        let mut input = tensor_with_index_shape(0, vec![1, 16, 16, 8]);
        input.dtype = "FLOAT32".to_string();
        let mut pointwise = tensor_with_index_shape(1, vec![16, 1, 1, 8]);
        pointwise.dtype = "FLOAT32".to_string();
        let tensors = vec![input, pointwise];
        let unit = Conv2dKernelGeometry {
            stride_width: 1,
            stride_height: 1,
            dilation_width: 1,
            dilation_height: 1,
        };
        assert_eq!(
            compute_kernel_class("CONV_2D", &[0, 1], &tensors, Some(unit)),
            "f32_gemm"
        );
        assert_eq!(
            compute_kernel_class(
                "CONV_2D",
                &[0, 1],
                &tensors,
                Some(Conv2dKernelGeometry {
                    stride_width: 2,
                    ..unit
                })
            ),
            "f32_igemm"
        );
        assert_eq!(
            compute_kernel_class("CONV_2D", &[0, 1], &tensors, None),
            "f32_igemm"
        );

        let mut grouped_filter = tensor_with_index_shape(1, vec![16, 1, 1, 4]);
        grouped_filter.dtype = "FLOAT32".to_string();
        assert_eq!(
            compute_kernel_class(
                "CONV_2D",
                &[0, 1],
                &[tensors[0].clone(), grouped_filter],
                Some(unit)
            ),
            "f32_igemm"
        );
    }

    fn constant_weight_tensor(
        dtype: &str,
        shape: Vec<i32>,
        byte_length: usize,
        scales: Vec<f32>,
        zero_points: Vec<i64>,
        quantized_dimension: i32,
    ) -> TensorInfo {
        let mut tensor = tensor_with_shape(shape);
        tensor.dtype = dtype.to_string();
        tensor.buffer_data_length = byte_length;
        tensor.constant_buffer = true;
        tensor.quant_scales = scales.len();
        tensor.quant_zero_points = zero_points.len();
        tensor.scale_sample = scales;
        tensor.zero_point_sample = zero_points;
        tensor.quantized_dimension = quantized_dimension;
        tensor
    }

    fn op_with(
        index: usize,
        name: &str,
        inputs: Vec<i32>,
        outputs: Vec<i32>,
        output_shape: Vec<i32>,
    ) -> OpInfo {
        OpInfo {
            index,
            name: name.to_string(),
            version: 1,
            stage_index: None,
            stage_key: None,
            inputs,
            outputs,
            output_shapes: vec![output_shape],
            macs: 0.0,
            mac_percent: 0.0,
            ops: 0.0,
            estimated_bytes: 0.0,
            fallback_byte_percent: 0.0,
            row_working_set_bytes: 0.0,
            row_working_set_ratio: 0.0,
            row_working_set_severity: "none".to_string(),
            cache_payload: CachePayloadBreakdown::not_applicable(name),
            intensity_ops_per_byte: 0.0,
            static_bound_guess: "mixed".to_string(),
            static_action: String::new(),
            roofline_reason: String::new(),
            fused_activation: "NONE".to_string(),
            fusion_status: "none".to_string(),
            fusion_detail: String::new(),
            xnnpack_supported: true,
            xnnpack_reason: String::new(),
            xnnpack_chain_id: 0,
            xnnpack_chain_role: "delegated".to_string(),
            xnnpack_chain_break: false,
            xnnpack_break_class: "none".to_string(),
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
            xnnpack_kernel_candidates: Vec::new(),
            xnnpack_kernel_alignment_multiples: Vec::new(),
            quantized_path: false,
            quantized_compute_path: false,
            quantization_state: "float".to_string(),
            quantization_detail: String::new(),
            weight_bytes: 0.0,
            weight_packing_overhead_us: 0.0,
            weight_packing_risk: "none".to_string(),
            weight_packing_detail: String::new(),
            output_channels: 0,
            channel_alignment_multiple: 0,
            channel_alignment_status: "none".to_string(),
            channel_alignment_detail: String::new(),
            channel_tail_overhead_percent: 0.0,
            channel_tail_overhead_percent_min: 0.0,
            channel_tail_overhead_percent_max: 0.0,
            quant_scale_ratio: 0.0,
            quant_scale_cv: 0.0,
            quant_scale_mode: "none".to_string(),
            quant_scale_ratio_meaningful: false,
            quant_zero_point_offset: 0,
            quant_zero_point_risk: "none".to_string(),
            quant_zero_point_status: "none".to_string(),
            quant_risk: "none".to_string(),
            quant_risk_detail: String::new(),
            low_norm_filter_count: None,
            low_norm_filter_total: None,
            quant_hole: false,
            quant_hole_class: "none".to_string(),
            quant_hole_detail: String::new(),
            patterns: Vec::new(),
            topo_role: "through".to_string(),
            topo_depth: 0,
            topo_fan_out_max: 0,
            bottleneck_compute_us: 0.0,
            bottleneck_memory_us: 0.0,
            bottleneck_packing_us: 0.0,
            bottleneck_break_us: 0.0,
            bottleneck_fallback_us: 0.0,
            bottleneck_total_us: 0.0,
            bottleneck_dominant: "memory".to_string(),
        }
    }

    #[test]
    fn structural_stages_are_bound_to_every_operator() {
        let mut ops = vec![
            op_with(0, "CONV_2D", vec![0], vec![1], vec![1, 112, 112, 16]),
            op_with(
                1,
                "DEPTHWISE_CONV_2D",
                vec![1],
                vec![2],
                vec![1, 112, 112, 16],
            ),
            op_with(2, "CONV_2D", vec![2], vec![3], vec![1, 56, 56, 24]),
            op_with(3, "RESHAPE", vec![3], vec![4], vec![1, 1001]),
        ];
        let stages = build_stages(&ops, 0.0, &[]);

        annotate_op_stages(&mut ops, &stages).expect("stage coverage");

        assert_eq!(stages.len(), 3);
        assert_eq!(ops[0].stage_index, Some(0));
        assert_eq!(ops[1].stage_index, Some(0));
        assert_eq!(ops[2].stage_index, Some(1));
        assert_eq!(ops[3].stage_index, Some(2));
        assert_eq!(ops[0].stage_key.as_deref(), Some("spatial/112x112"));
        assert_eq!(ops[2].stage_key.as_deref(), Some("spatial/56x56"));
        assert_eq!(ops[3].stage_key.as_deref(), Some("vector/head"));
        assert!(ops
            .iter()
            .all(|op| op.stage_index.is_some() && op.stage_key.is_some()));
    }

    #[test]
    fn external_qdq_conversions_are_not_mid_graph_holes() {
        let mut tensors = (0..4)
            .map(|index| tensor_with_index_shape(index, vec![1, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "FLOAT32".to_string();
        tensors[1].dtype = "INT8".to_string();
        tensors[2].dtype = "INT8".to_string();
        tensors[3].dtype = "FLOAT32".to_string();

        let mut ops = vec![
            op_with(0, "QUANTIZE", vec![0], vec![1], vec![1, 4]),
            op_with(1, "CONV_2D", vec![1], vec![2], vec![1, 4]),
            op_with(2, "DEQUANTIZE", vec![2], vec![3], vec![1, 4]),
        ];
        ops[1].macs = 64.0;

        let holes = detect_quant_holes(&mut ops, &[0], &[3], &tensors, 64.0);

        assert!(holes.is_empty());
        assert!(ops.iter().all(|item| !item.quant_hole));
    }

    #[test]
    fn fp16_constant_expansion_is_not_an_activation_quant_hole() {
        let mut tensors = (0..3)
            .map(|index| tensor_with_index_shape(index, vec![4, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "FLOAT16".to_string();
        tensors[0].constant_buffer = true;
        tensors[0].buffer_data_length = 32;
        tensors[1].dtype = "FLOAT32".to_string();
        tensors[2].dtype = "FLOAT32".to_string();

        let mut ops = vec![
            op_with(0, "DEQUANTIZE", vec![0], vec![1], vec![4, 4]),
            op_with(1, "CONV_2D", vec![2, 1], vec![2], vec![1, 4]),
        ];
        let (state, detail) =
            classify_op_quantization(&ops[0].name, &ops[0].inputs, &ops[0].outputs, &tensors);
        ops[0].quantization_state = state;
        ops[0].quantization_detail = detail;

        let holes = detect_quant_holes(&mut ops, &[2], &[2], &tensors, 64.0);

        assert!(holes.is_empty());
        assert!(!ops[0].quant_hole);
        assert_eq!(ops[0].quantization_state, "float16_constant_expansion");
        assert!(ops[0]
            .quantization_detail
            .contains("weight-storage conversion"));
        let status = classify_model_quantization(
            &ops,
            &tensors,
            &[tensors[2].clone()],
            &[tensors[2].clone()],
            0,
            64.0,
        );
        assert_eq!(status.classification, "float16_weight_storage");
        assert_eq!(status.activation_dequantize_ops, 0);
        assert_eq!(status.constant_precision_conversion_ops, 1);
        assert_eq!(status.float16_constant_expansion_ops, 1);
    }

    #[test]
    fn int8_constant_dequantization_is_not_an_activation_quant_hole() {
        let mut tensors = (0..3)
            .map(|index| tensor_with_index_shape(index, vec![4, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "INT8".to_string();
        tensors[0].constant_buffer = true;
        tensors[0].buffer_data_length = 16;
        tensors[1].dtype = "FLOAT32".to_string();
        tensors[2].dtype = "FLOAT32".to_string();

        let mut ops = vec![
            op_with(0, "DEQUANTIZE", vec![0], vec![1], vec![4, 4]),
            op_with(1, "CONV_2D", vec![2, 1], vec![2], vec![1, 4]),
        ];
        let (state, detail) =
            classify_op_quantization(&ops[0].name, &ops[0].inputs, &ops[0].outputs, &tensors);
        ops[0].quantization_state = state;
        ops[0].quantization_detail = detail;

        let holes = detect_quant_holes(&mut ops, &[2], &[2], &tensors, 64.0);

        assert!(holes.is_empty());
        assert!(!ops[0].quant_hole);
        assert_eq!(ops[0].quantization_state, "quantized_constant_expansion");
    }

    #[test]
    fn uint8_to_int8_quantize_is_integer_requantization_not_float_hole() {
        let mut tensors = (0..2)
            .map(|index| tensor_with_index_shape(index, vec![1, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "UINT8".to_string();
        tensors[1].dtype = "INT8".to_string();
        let mut ops = vec![op_with(0, "QUANTIZE", vec![0], vec![1], vec![1, 4])];
        let (state, detail) =
            classify_op_quantization(&ops[0].name, &ops[0].inputs, &ops[0].outputs, &tensors);
        ops[0].quantization_state = state;
        ops[0].quantization_detail = detail;

        assert_eq!(ops[0].quantization_state, "integer_requantization");
        assert!(ops[0]
            .quantization_detail
            .contains("without entering floating point"));
        assert!(detect_quant_holes(&mut ops, &[0], &[1], &tensors, 0.0).is_empty());
        let status = classify_model_quantization(
            &ops,
            &tensors,
            &[tensors[0].clone()],
            &[tensors[1].clone()],
            2,
            0.0,
        );
        assert_eq!(status.activation_quantize_ops, 1);
        assert_eq!(status.activation_8bit_float_boundary_ops, 0);
        assert_eq!(status.integer_requantization_ops, 1);
    }

    #[test]
    fn full_integer_classification_requires_complete_internal_integer_compute_evidence() {
        let mut tensors = (0..4)
            .map(|index| tensor_with_index_shape(index, vec![1, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "UINT8".to_string();
        tensors[1].dtype = "INT8".to_string();
        tensors[2].dtype = "INT32".to_string();
        tensors[3].dtype = "UINT8".to_string();
        let mut conv = op_with(0, "CONV_2D", vec![0, 1, 2], vec![3], vec![1, 4]);
        conv.macs = 64.0;
        conv.quantized_compute_path = true;
        conv.quantization_state = "quantized_compute".to_string();

        let complete = classify_model_quantization(
            &[conv.clone()],
            &tensors,
            &[tensors[0].clone()],
            &[tensors[3].clone()],
            4,
            64.0,
        );
        assert!(complete.full_integer);
        assert_eq!(complete.classification, "full_integer");
        assert_eq!(complete.compute_ops, 1);
        assert_eq!(complete.quantized_compute_ops, 1);
        assert_eq!(complete.compute_macs, 64.0);
        assert_eq!(complete.quantized_compute_macs, 64.0);
        assert_eq!(complete.quantized_compute_mac_percent, 1.0);

        let mut float_internal = tensors.clone();
        float_internal[1].dtype = "FLOAT32".to_string();
        let float_status = classify_model_quantization(
            &[conv.clone()],
            &float_internal,
            &[float_internal[0].clone()],
            &[float_internal[3].clone()],
            3,
            64.0,
        );
        assert!(!float_status.full_integer);

        let mut unquantized_conv = conv;
        unquantized_conv.quantized_compute_path = false;
        unquantized_conv.quantization_state = "float_compute".to_string();
        let incomplete_compute = classify_model_quantization(
            &[unquantized_conv],
            &tensors,
            &[tensors[0].clone()],
            &[tensors[3].clone()],
            4,
            64.0,
        );
        assert!(!incomplete_compute.full_integer);
    }

    #[test]
    fn compute_mac_denominator_uses_the_complete_shared_compute_op_set() {
        let mut tensors = (0..4)
            .map(|index| tensor_with_index_shape(index, vec![1, 4]))
            .collect::<Vec<_>>();
        tensors[0].dtype = "UINT8".to_string();
        tensors[1].dtype = "INT8".to_string();
        tensors[2].dtype = "INT32".to_string();
        tensors[3].dtype = "UINT8".to_string();

        let mut conv = op_with(0, "CONV_2D", vec![0, 1, 2], vec![3], vec![1, 4]);
        conv.macs = 64.0;
        conv.quantized_compute_path = true;
        conv.quantization_state = "quantized_compute".to_string();
        let mut transpose_conv = op_with(1, "TRANSPOSE_CONV", vec![0, 1, 2], vec![3], vec![1, 4]);
        transpose_conv.macs = 36.0;
        transpose_conv.quantized_compute_path = false;
        transpose_conv.quantization_state = "float_compute".to_string();
        let mut add = op_with(2, "ADD", vec![0, 3], vec![3], vec![1, 4]);
        add.macs = 900.0;

        let status = classify_model_quantization(
            &[conv, transpose_conv, add],
            &tensors,
            &[tensors[0].clone()],
            &[tensors[3].clone()],
            4,
            1_000.0,
        );
        assert_eq!(status.compute_ops, 2);
        assert_eq!(status.quantized_compute_ops, 1);
        assert_eq!(status.compute_macs, 100.0);
        assert_eq!(status.quantized_compute_macs, 64.0);
        assert!((status.quantized_compute_mac_percent - 0.64).abs() < f64::EPSILON);

        let mut unresolved_conv = op_with(3, "CONV_2D", vec![0, 1, 2], vec![3], vec![1, 4]);
        unresolved_conv.macs = f64::NAN;
        unresolved_conv.quantized_compute_path = true;
        let unresolved = classify_model_quantization(
            &[unresolved_conv],
            &tensors,
            &[tensors[0].clone()],
            &[tensors[3].clone()],
            4,
            0.0,
        );
        assert_eq!(unresolved.compute_ops, 1);
        assert_eq!(unresolved.quantized_compute_ops, 1);
        assert_eq!(unresolved.compute_macs, 0.0);
        assert_eq!(unresolved.quantized_compute_macs, 0.0);
        assert_eq!(unresolved.quantized_compute_mac_percent, 0.0);
        assert!(unresolved.compute_macs.is_finite());
        assert!(unresolved.quantized_compute_macs.is_finite());
    }

    #[test]
    fn constant_dequantization_does_not_seed_qdq_island_motif() {
        let mut storage_ops = vec![
            op_with(0, "DEQUANTIZE", vec![0], vec![1], vec![4, 4]),
            op_with(1, "CONV_2D", vec![2, 1], vec![3], vec![1, 4]),
        ];
        storage_ops[0].quantization_state = "float16_constant_expansion".to_string();
        compute_topology_annotations(&mut storage_ops);
        assert!(!storage_ops[1].patterns.contains(&"qdq-island".to_string()));

        let mut activation_ops = vec![
            op_with(0, "DEQUANTIZE", vec![0], vec![1], vec![4, 4]),
            op_with(1, "CONV_2D", vec![1, 2], vec![3], vec![1, 4]),
        ];
        activation_ops[0].quantization_state = "quant_boundary".to_string();
        compute_topology_annotations(&mut activation_ops);
        assert!(activation_ops[1]
            .patterns
            .contains(&"qdq-island".to_string()));
    }

    #[test]
    fn quant_hole_impact_uses_graph_neighbors_not_serialization_neighbors() {
        let mut tensors = (0..12)
            .map(|index| tensor_with_index_shape(index, vec![1, 4]))
            .collect::<Vec<_>>();
        tensors[1].dtype = "INT8".to_string();
        tensors[2].dtype = "FLOAT32".to_string();

        let mut ops = vec![
            op_with(0, "CONV_2D", vec![0], vec![1], vec![1, 4]),
            op_with(1, "FULLY_CONNECTED", vec![10], vec![11], vec![1, 4]),
            op_with(2, "DEQUANTIZE", vec![1], vec![2], vec![1, 4]),
            op_with(3, "ADD", vec![2], vec![3], vec![1, 4]),
        ];
        ops[0].macs = 30.0;
        ops[1].macs = 90.0;
        ops[3].macs = 70.0;

        let holes = detect_quant_holes(&mut ops, &[0, 10], &[3, 11], &tensors, 100.0);

        assert_eq!(holes.len(), 1);
        assert!((holes[0].adjacent_mac_percent - 0.7).abs() < f64::EPSILON);
        assert_eq!(holes[0].prev_op_name, "#0 CONV_2D");
        assert_eq!(holes[0].next_op_name, "#3 ADD");
        assert!(holes[0].detail.contains("70.0% of total model MACs"));
    }

    #[test]
    fn logical_cache_payload_separates_dilated_input_output_and_serialized_constants() {
        let mut input = tensor_with_index_shape(0, vec![1, 10, 20, 4]);
        input.dtype = "FLOAT32".to_string();
        let mut weight = constant_weight_tensor(
            "FLOAT32",
            vec![8, 3, 3, 4],
            8 * 3 * 3 * 4 * 4,
            vec![],
            vec![],
            0,
        );
        weight.index = 1;
        let mut bias = constant_weight_tensor("FLOAT32", vec![8], 8 * 4, vec![], vec![], 0);
        bias.index = 2;
        let mut output = tensor_with_index_shape(3, vec![1, 10, 20, 8]);
        output.dtype = "FLOAT32".to_string();
        let tensors = vec![input, weight, bias, output];

        let payload = logical_cache_payload_for_op("CONV_2D", &[0, 1, 2], &[3], &tensors, 2);

        assert_eq!(payload.status, "assessed");
        assert_eq!(payload.effective_kernel_height, Some(5));
        assert_eq!(payload.input_strip_bytes, Some(5 * 20 * 4 * 4));
        assert_eq!(payload.output_row_bytes, Some(20 * 8 * 4));
        assert_eq!(payload.logical_row_payload_bytes, Some(2240));
        assert_eq!(payload.serialized_kernel_bytes, Some(8 * 3 * 3 * 4 * 4));
        assert_eq!(payload.serialized_bias_bytes, Some(8 * 4));
        assert_eq!(
            payload.logical_row_payload_bytes,
            payload
                .input_strip_bytes
                .zip(payload.output_row_bytes)
                .map(|(input, output)| input + output)
        );
    }

    #[test]
    fn quick_low_norm_centers_uint8_weights_before_measuring_filters() {
        let bytes = [128u8, 128, 130, 128];
        let tensor =
            constant_weight_tensor("UINT8", vec![2, 2], bytes.len(), vec![0.5], vec![128], 0);

        let stat = compute_quick_low_norm_stat_for_tensor(&bytes, &tensor, false).unwrap();
        assert_eq!(stat.low_norm, 1);
        assert_eq!(stat.total, 2);
    }

    #[test]
    fn quick_low_norm_uses_depthwise_interleaved_filter_axis() {
        let bytes = [128u8, 130, 128, 130];
        let mut tensor = constant_weight_tensor(
            "UINT8",
            vec![1, 1, 2, 2],
            bytes.len(),
            vec![0.25, 0.25],
            vec![128, 128],
            3,
        );

        let stat = compute_quick_low_norm_stat_for_tensor(&bytes, &tensor, true).unwrap();
        assert_eq!(stat.low_norm, 1);
        assert_eq!(stat.total, 2);
        tensor.quantized_dimension = 0;
        assert!(compute_quick_low_norm_stat_for_tensor(&bytes, &tensor, true).is_none());
    }

    #[test]
    fn weight_slice_integrity_requires_an_actual_compute_weight_edge() {
        let bytes = [0u8, 0, 1, 2];
        let mut input = tensor_with_index_shape(0, vec![1, 2]);
        input.dtype = "INT8".to_string();
        let mut kernel =
            constant_weight_tensor("INT8", vec![2, 2], 4, vec![1.0, 1.0], vec![0, 0], 0);
        kernel.index = 1;
        kernel.name = "fc_kernel".to_string();
        let output = tensor_with_index_shape(2, vec![1, 2]);
        let mut unrelated =
            constant_weight_tensor("INT8", vec![2, 2], 4, vec![1.0, 1.0], vec![0, 0], 0);
        unrelated.index = 3;
        unrelated.name = "unrelated_matrix".to_string();
        let op = op_with(0, "FULLY_CONNECTED", vec![0, 1, -1], vec![2], vec![1, 2]);

        let report =
            compute_weight_integrity(&bytes, &[input, kernel, output, unrelated], &[op], 0.0);

        assert_eq!(report.eligible_kernel_tensors_scanned, 1);
        assert_eq!(report.output_channels_evaluated, 2);
        assert_eq!(report.zero_kernel_slice_count, 1);
        assert_eq!(report.exact_zero_kernel_slice_count, 1);
        assert_eq!(report.zero_kernel_slice_details[0].tensor_name, "fc_kernel");
        assert_eq!(report.zero_kernel_slice_details[0].channels, vec![0]);
        assert_eq!(
            report.zero_kernel_slice_details[0].exact_zero_channels,
            vec![0]
        );
    }

    #[test]
    fn grid_utilization_separates_all_size_minimum_from_threshold_population() {
        let mut bytes = vec![0u8, 1, 2, 3, 4];
        bytes.extend((0..256).map(|index| (index % 82) as u8));
        let mut small = constant_weight_tensor("INT8", vec![1, 5], 5, vec![1.0], vec![0], 0);
        small.index = 0;
        small.name = "small_constant".to_string();
        let mut large = constant_weight_tensor("INT8", vec![1, 256], 256, vec![1.0], vec![0], 0);
        large.index = 1;
        large.name = "threshold_eligible_constant".to_string();
        large.buffer_data_offset = 5;

        let report = compute_weight_integrity(&bytes, &[small, large], &[], 0.0);

        assert_close(report.min_grid_utilization, 5.0 / 256.0, 1e-12);
        assert_eq!(report.threshold_eligible_quantized_constant_tensors, 1);
        assert_close(
            report.min_threshold_eligible_grid_utilization.unwrap(),
            82.0 / 256.0,
            1e-12,
        );
        assert_eq!(report.low_grid_utilization_tensors, 0);
        assert!(report.quant_grid_detail.contains(
            "all-size minimum 8-bit level utilization 2.0%; threshold-eligible minimum 32.0%"
        ));
    }

    fn quantized_weight_integrity_fixture() -> WeightIntegrityReport {
        WeightIntegrityReport {
            weight_tensors_scanned: 1,
            quantized_constant_tensors_scanned: 1,
            elements_scanned: 9,
            eligible_kernel_tensors_scanned: 1,
            output_channels_evaluated: 1,
            zero_kernel_slice_tensors: 1,
            zero_kernel_slice_count: 1,
            max_abs_weight: 1.0,
            mean_sparsity: 1.0,
            high_sparsity_tensors: 1,
            zero_kernel_slice_details: vec![ZeroKernelSliceDetail {
                tensor_index: 1,
                tensor_name: "dw_weight".to_string(),
                dtype: "UINT8".to_string(),
                shape: vec![1, 3, 3, 1],
                channels: vec![0],
                channel_count: 1,
                scale_sample: vec![0.25],
                zero_point_sample: vec![128],
                bias_tensor_index: 2,
                bias_tensor_name: "dw_bias".to_string(),
                bias_dtype: "INT32".to_string(),
                bias_value_sample: vec![1.0],
                bias_nonzero_for_flagged_channels: true,
                functional_status: "NOT_ASSESSABLE".to_string(),
                consumer_ops: vec!["#0 DEPTHWISE_CONV_2D".to_string()],
                consumer_mac_percent: 1.0,
                ..Default::default()
            }],
            low_grid_utilization_tensors: 1,
            min_grid_utilization: 2.0 / 256.0,
            quant_grid_detail: "1 tensor below the 25% utilization review threshold".to_string(),
            status: "warn".to_string(),
            detail: "fixture".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn pinned_runtime_floor_covers_quantized_resize_mean_and_depthwise_versions() {
        assert_eq!(
            pinned_runtime_version_for_op("RESIZE_BILINEAR", 3),
            Some("2.2.0")
        );
        assert_eq!(
            pinned_runtime_version_for_op("RESIZE_BILINEAR", 4),
            Some("2.5.0")
        );
        assert_eq!(pinned_runtime_version_for_op("MEAN", 2), Some("1.14.0"));
        assert_eq!(pinned_runtime_version_for_op("MEAN", 3), Some("2.4.0"));
        assert_eq!(
            pinned_runtime_version_for_op("DEPTHWISE_CONV_2D", 4),
            Some("2.2.0")
        );
        assert_eq!(
            pinned_runtime_version_for_op("DEPTHWISE_CONV_2D", 7),
            Some("2.11.0")
        );
        assert_eq!(pinned_runtime_version_for_op("RESIZE_BILINEAR", 99), None);
    }

    #[test]
    fn input_contract_uses_direct_operator_semantics_and_exact_scalar_quantization() {
        let mut input = tensor_with_index_shape(0, vec![1, 224, 224, 3]);
        input.name = "image".to_string();
        input.dtype = "UINT8".to_string();
        input.quant_scales = 1;
        input.quant_zero_points = 1;
        input.scale_sample = vec![1.0 / 128.0];
        input.zero_point_sample = vec![128];
        let conv = op_with(7, "CONV_2D", vec![0, 1, 2], vec![3], vec![1, 112, 112, 16]);

        let contracts = validate_input_contracts(&[input], &[0], &[conv]);
        let contract = &contracts[0];
        assert_eq!(contract.schema, "deepbom.input_tensor_contract.v1");
        assert_eq!(contract.layout.as_deref(), Some("NHWC"));
        assert_eq!(
            contract.layout_status,
            "derived_nhwc_from_direct_consumer_semantics"
        );
        assert_eq!(contract.layout_evidence_class, "DERIVED");
        assert_eq!(contract.layout_source_op_index, Some(7));
        assert_eq!(contract.layout_source_op_name.as_deref(), Some("CONV_2D"));
        assert_eq!(contract.channel_axis, Some(3));
        assert_eq!(contract.channels, Some(3));
        assert_eq!(
            contract.tensor_numerical_contract_status,
            "known_from_artifact_quantization_metadata"
        );
        assert_close(contract.expected_range_low.unwrap(), -1.0, 1e-12);
        assert_close(contract.expected_range_high.unwrap(), 127.0 / 128.0, 1e-12);
        assert!(contract
            .risks
            .iter()
            .any(|risk| risk.contains("RGB vs BGR")));
    }

    #[test]
    fn input_contract_never_promotes_rank_or_incomplete_quantization_to_fact() {
        let mut input = tensor_with_index_shape(0, vec![1, 8, 8, 3]);
        input.dtype = "INT8".to_string();
        input.quant_scales = 3;
        input.quant_zero_points = 3;
        input.scale_sample = vec![0.1, 0.2, 0.3];
        input.zero_point_sample = vec![0, 0, 0];
        let resize = op_with(
            2,
            "RESIZE_BILINEAR",
            vec![1, 0],
            vec![3],
            vec![1, 16, 16, 3],
        );

        let contracts = validate_input_contracts(&[input], &[0], &[resize]);
        let contract = &contracts[0];
        assert_eq!(contract.layout, None);
        assert_eq!(contract.channel_axis, None);
        assert_eq!(contract.channels, None);
        assert_eq!(contract.layout_evidence_class, "NOT_ASSESSABLE");
        assert_eq!(
            contract.layout_status,
            "not_assessed_no_direct_layout_semantic_consumer"
        );
        assert_eq!(contract.expected_range_low, None);
        assert_eq!(contract.expected_range_high, None);
        assert_eq!(
            contract.tensor_numerical_contract_status,
            "not_assessed_non_scalar_input_quantization"
        );
    }

    #[test]
    fn quantized_artifact_facts_enter_public_wasm_findings_without_overclaiming() {
        let activation = tensor_with_index_shape(0, vec![1, 4, 4, 1]);
        let mut weight = tensor_with_index_shape(1, vec![1, 3, 3, 1]);
        weight.name = "dw_weight".to_string();
        weight.dtype = "UINT8".to_string();
        weight.constant_buffer = true;
        weight.quant_scales = 1;
        weight.quant_zero_points = 1;
        weight.scale_sample = vec![0.25];
        weight.zero_point_sample = vec![128];
        let op = op_with(
            0,
            "DEPTHWISE_CONV_2D",
            vec![0, 1],
            vec![2],
            vec![1, 2, 2, 1],
        );
        let mut findings = Vec::new();

        findings::append_quantized_artifact_findings(
            &mut findings,
            &[op],
            &[activation, weight],
            &quantized_weight_integrity_fixture(),
        );

        assert_eq!(findings.len(), 4);
        let ids = findings
            .iter()
            .map(|finding| finding.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"weight-integrity-zero-kernel-slices"));
        assert!(ids.contains(&"quant-grid-utilization"));
        assert!(ids.contains(&"depthwise-per-tensor-weights"));
        assert!(ids.contains(&"asymmetric-uint8-weights"));
        let zero_slice = findings
            .iter()
            .find(|finding| finding.id == "weight-integrity-zero-kernel-slices")
            .unwrap();
        assert_eq!(zero_slice.severity, "informational");
        assert!(zero_slice.evidence.iter().any(|evidence| evidence
            .text
            .contains("Neither alone proves model-output inactivity")));
        assert!(zero_slice
            .impact
            .contains("model-output inactivity is not established"));
        let asymmetric = findings
            .iter()
            .find(|finding| finding.id == "asymmetric-uint8-weights")
            .unwrap();
        assert!(asymmetric
            .evidence
            .iter()
            .any(|evidence| evidence.text.contains("legal dtype values")));
    }

    #[test]
    fn pack_rgba_float32_handles_luma_rgb_bgr_and_safe_std() {
        let rgba = [255, 0, 128, 64];

        let luma = pack_rgba_float32(&rgba, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], false);
        let expected_luma = 0.299 + 0.114 * (128.0 / 255.0);
        assert_close(luma[0] as f64, expected_luma, 1e-6);

        let rgb = pack_rgba_float32(&rgba, 3, [0.0, 0.0, 0.0], [1.0, 2.0, 0.0], false);
        assert_close(rgb[0] as f64, 1.0, 1e-6);
        assert_close(rgb[1] as f64, 0.0, 1e-6);
        assert_close(rgb[2] as f64, 128.0 / 255.0, 1e-6);

        let bgr = pack_rgba_float32(&rgba, 3, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], true);
        assert_close(bgr[0] as f64, 128.0 / 255.0, 1e-6);
        assert_close(bgr[2] as f64, 1.0, 1e-6);
    }

    #[test]
    fn target_profiles_encode_device_specific_constraints() {
        let a72 = target_profile("rpi4_a72").unwrap();
        let a55 = target_profile("android_mid_a55").unwrap();
        let zynq = target_profile("zynq_ultrascale_plus_a53").unwrap();
        let x3 = target_profile("android_flagship_x3_a715").unwrap();
        let avx2 = target_profile("x86_avx2").unwrap();
        let sse4 = target_profile("x86_sse4").unwrap();
        let wasm = target_profile("wasm_simd").unwrap();

        assert_eq!(a72.l1_data_bytes, 32 * 1024);
        assert_eq!(a55.l1_data_bytes, 32 * 1024);
        assert_eq!(zynq.l1_data_bytes, 32 * 1024);
        assert_eq!(zynq.l2_bytes, 1024 * 1024);
        assert_eq!(a72.core_count_min, Some(4));
        assert_eq!(a72.core_count_max, Some(4));
        assert_eq!(a55.core_count_min, None);
        assert_eq!(a55.core_count_max, None);
        assert_eq!(zynq.core_count_min, Some(2));
        assert_eq!(zynq.core_count_max, Some(4));
        assert!(!zynq.dot_product);
        assert_eq!(zynq.performance_model_evidence_class, "HEURISTIC");
        let zynq_spec = zynq.hardware_spec.as_ref().unwrap();
        assert_eq!(zynq_spec.evidence_class, "SOURCE_BACKED_PRODUCT");
        assert_eq!(zynq_spec.max_clock_mhz, Some(1500));
        assert_eq!(zynq_spec.l1_instruction_bytes, 32 * 1024);
        assert_eq!(zynq_spec.l1_data_bytes, zynq.l1_data_bytes);
        assert_eq!(zynq_spec.l1_instruction_ways, 2);
        assert_eq!(zynq_spec.l1_data_ways, 4);
        assert_eq!(zynq_spec.l1_line_bytes, 64);
        assert_eq!(zynq_spec.l2_bytes, zynq.l2_bytes);
        assert_eq!(zynq_spec.l2_ways, 16);
        assert_eq!(zynq_spec.l2_line_bytes, 64);
        assert!(zynq_spec.advanced_simd);
        assert!(!zynq_spec.fp16_vector_arithmetic);
        assert_eq!(zynq_spec.sources.len(), 2);
        assert!(zynq_spec.sources.iter().any(|source| {
            source.sha256 == "1badf7142690c573987f3eacd788620ff8a8392425f13124f928aaed152265e9"
        }));
        assert!(zynq_spec.sources.iter().any(|source| {
            source.sha256 == "52b19d733bdacfbd1cffd108b277bfbc115839aab0a9f5d51f43b6dfa7c33369"
        }));
        let a72_spec = a72.hardware_spec.as_ref().unwrap();
        assert_eq!(a72_spec.l1_instruction_bytes, 48 * 1024);
        assert_eq!(a72_spec.l1_data_bytes, a72.l1_data_bytes);
        assert_eq!(a72_spec.l1_instruction_ways, 3);
        assert_eq!(a72_spec.l1_data_ways, 2);
        assert_eq!(a72_spec.l1_line_bytes, 64);
        assert_eq!(a72_spec.l2_ways, 16);
        assert_eq!(a72_spec.l2_line_bytes, 64);
        assert!(a72_spec
            .configuration_context
            .contains("implementation-configurable"));
        assert!(a72_spec.sources.iter().any(|source| {
            source.sha256 == "47f52c93806507962c2cd0b77991d7aebacfaae5734ac617fc45a5babdff738f"
        }));
        assert_eq!(
            target_profile("android_mid_a55_l1_16k")
                .unwrap()
                .l1_data_bytes,
            16 * 1024
        );
        assert_eq!(
            target_profile("android_mid_a55_l1_64k")
                .unwrap()
                .l1_data_bytes,
            64 * 1024
        );
        assert!(a55.cache_assumption.contains("L1D 8-64 KiB"));
        assert_eq!(a55.profile_sha256.len(), 64);
        assert!(a55
            .profile_sha256
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
        assert!(a55.in_order);
        assert!(a55.dot_product);
        assert_eq!(a55.channel_alignment_multiple, 16);
        assert_ne!(a72.profile_sha256, zynq.profile_sha256);
        assert!(x3.sve2);
        assert!(x3.ridge_point_ops_per_byte > avx2.ridge_point_ops_per_byte);
        assert!(wasm.ridge_point_ops_per_byte < sse4.ridge_point_ops_per_byte);
        assert!(wasm.effective_peak_gops < sse4.effective_peak_gops);
        assert!(wasm.effective_memory_bandwidth_gbps < sse4.effective_memory_bandwidth_gbps);
        assert!(wasm.chain_break_overhead_us_low > sse4.chain_break_overhead_us_low);
        assert_ne!(
            bound_thresholds_for_op("DEPTHWISE_CONV_2D", &wasm, false),
            bound_thresholds_for_op("DEPTHWISE_CONV_2D", &sse4, false)
        );
        assert!(target_profile("unknown-target").is_err());
    }

    #[test]
    fn target_profile_hash_binds_every_retuned_performance_semantic() {
        let scalar_a = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"compute_utilization_factor":0.12345671}}"#,
        )
        .unwrap();
        let scalar_b = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"compute_utilization_factor":0.12345672}}"#,
        )
        .unwrap();
        assert_ne!(scalar_a.profile_sha256, scalar_b.profile_sha256);

        let kernel_a = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"compute_utilization_by_kernel_class":{"qu8_igemm":0.11}}}"#,
        )
        .unwrap();
        let kernel_b = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"compute_utilization_by_kernel_class":{"qu8_igemm":0.12}}}"#,
        )
        .unwrap();
        assert_ne!(kernel_a.profile_sha256, kernel_b.profile_sha256);

        let scope_a = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"l2_capacity_scope":"private_per_core"}}"#,
        )
        .unwrap();
        let scope_b = custom_target_profile(
            r#"{"base":"android_mid_a55","id":"custom:hash-test","label":"Hash test","evidence_class":"USER_DECLARED","evidence_note":"same note","overrides":{"l2_capacity_scope":"shared_cluster"}}"#,
        )
        .unwrap();
        assert_ne!(scope_a.profile_sha256, scope_b.profile_sha256);
    }

    #[test]
    fn conversion_metadata_parser_reads_source_declared_environment() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&12u32.to_le_bytes());
        bytes[4..6].copy_from_slice(&8u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&8u16.to_le_bytes());
        bytes[8..10].copy_from_slice(&4u16.to_le_bytes());
        bytes[12..16].copy_from_slice(&8i32.to_le_bytes());
        bytes[16..20].copy_from_slice(&20u32.to_le_bytes());
        bytes[24..26].copy_from_slice(&10u16.to_le_bytes());
        bytes[26..28].copy_from_slice(&16u16.to_le_bytes());
        bytes[28..30].copy_from_slice(&4u16.to_le_bytes());
        bytes[30..32].copy_from_slice(&8u16.to_le_bytes());
        bytes[32..34].copy_from_slice(&12u16.to_le_bytes());
        bytes[36..40].copy_from_slice(&12i32.to_le_bytes());
        bytes[40..44].copy_from_slice(&12u32.to_le_bytes());
        bytes[44..48].copy_from_slice(&2u32.to_le_bytes());
        bytes[48..52].copy_from_slice(&2i32.to_le_bytes());
        bytes[52..56].copy_from_slice(&6u32.to_le_bytes());
        bytes[56..62].copy_from_slice(b"2.15.0");

        let parsed = parse_conversion_metadata(&bytes);
        assert_eq!(parsed.status, "parsed");
        assert_eq!(parsed.tensorflow_version, "2.15.0");
        assert_eq!(parsed.api_version, Some(2));
        assert_eq!(parsed.model_type, "KERAS_MODEL");
        assert!(parsed.optimization_mode_codes.is_empty());
        assert!(parsed.optimization_modes.is_empty());
        assert_eq!(conversion_optimization_mode_name(1003), "PTQ_FULL_INTEGER");
        assert_eq!(
            conversion_optimization_mode_name(2000),
            "QUANTIZATION_AWARE_TRAINING"
        );
        assert_eq!(conversion_optimization_mode_name(42), "UNKNOWN(42)");

        let malformed = parse_conversion_metadata(&[255, 255, 255, 255]);
        assert!(malformed.status.starts_with("invalid_flatbuffer:"));

        let missing_string_payload = parse_conversion_metadata(&bytes[..52]);
        assert!(missing_string_payload
            .status
            .starts_with("invalid_flatbuffer:"));

        for cut in 0..bytes.len() {
            let truncated = parse_conversion_metadata(&bytes[..cut]);
            if truncated.status == "parsed" {
                assert_eq!(truncated.tensorflow_version, parsed.tensorflow_version);
                assert_eq!(truncated.api_version, parsed.api_version);
                assert_eq!(truncated.model_type, parsed.model_type);
                assert_eq!(
                    truncated.optimization_mode_codes,
                    parsed.optimization_mode_codes
                );
                assert_eq!(truncated.optimization_modes, parsed.optimization_modes);
            }
        }
    }

    #[test]
    fn tflite_truncation_is_rejected_or_preserves_structural_totals() {
        use std::collections::BTreeSet;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/web/samples/mobilenet_v2_1.0_224_quant.tflite"
        );
        let bytes = std::fs::read(path).expect("public quantized TFLite fixture");
        let baseline = analyze_with_target_without_step_response(
            &bytes,
            "mobilenet_v2_1.0_224_quant.tflite",
            "android_mid_a55",
        )
        .expect("baseline analysis");
        let mut cuts = BTreeSet::new();
        cuts.extend(0..bytes.len().min(256));
        cuts.extend((0..bytes.len()).step_by(64 * 1024));
        cuts.extend(bytes.len().saturating_sub(256)..bytes.len());
        cuts.insert(bytes.len() / 2);
        cuts.insert(bytes.len() * 99 / 100);

        for cut in cuts {
            let candidate_bytes = &bytes[..cut];
            if Fb::verified_tflite(candidate_bytes).is_err() {
                continue;
            }
            let candidate = analyze_with_target_without_step_response(
                candidate_bytes,
                "mobilenet_v2_1.0_224_quant.tflite",
                "android_mid_a55",
            )
            .expect("a gate-accepted prefix must remain analyzable");
            assert_eq!(
                candidate.operator_count, baseline.operator_count,
                "cut={cut}"
            );
            assert_eq!(candidate.tensor_count, baseline.tensor_count, "cut={cut}");
            assert_eq!(candidate.total_macs, baseline.total_macs, "cut={cut}");
            assert_eq!(candidate.total_ops, baseline.total_ops, "cut={cut}");
        }
    }

    #[test]
    fn public_tflite_byte_ledger_conserves_every_artifact_byte() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/web/samples/mobilenet_v2_1.0_224_quant.tflite"
        );
        let bytes = std::fs::read(path).expect("public quantized TFLite fixture");
        let analysis = analyze_with_target_without_step_response(
            &bytes,
            "mobilenet_v2_1.0_224_quant.tflite",
            "android_mid_a55",
        )
        .expect("byte-ledger analysis");
        assert_eq!(
            analysis.artifact_byte_integrity.conservation_status,
            "exact"
        );
        assert_eq!(
            analysis.artifact_byte_integrity.classified_bytes,
            Some(bytes.len())
        );
        assert_eq!(analysis.artifact_byte_integrity.status, "risk");
        assert_eq!(analysis.artifact_byte_integrity.unowned_trailing_bytes, 8);
        assert_eq!(
            analysis.artifact_byte_integrity.unowned_trailing_ranges[0].offset,
            bytes.len() - 8
        );
        assert_eq!(
            &bytes[bytes.len() - 8..],
            &[6, 0, 0, 0, 0, 0, 0, 3],
            "the hash-pinned public fixture retains a non-zero unowned suffix"
        );
    }

    #[test]
    fn public_kernel_analysis_never_emits_protected_source_candidates() {
        let target = target_profile("android_mid_a55").unwrap();
        let mut activation = tensor_with_index_shape(0, vec![1, 8, 8, 16]);
        activation.dtype = "FLOAT32".to_string();
        let summary = source_kernel_candidate_for_op("CONV_2D", &[0], &[activation], &target);

        assert_eq!(summary.candidate, target.xnnpack_kernel_family);
        assert_eq!(summary.evidence_class, "HEURISTIC_PROFILE");
        assert!(summary.candidates.is_empty());
        assert!(summary.alignment_multiples.is_empty());
        assert_eq!(summary.tile_mr, 0);
        assert_eq!(summary.tile_nr, 0);
        assert_eq!(summary.channel_tile, 0);
        assert_eq!(summary.primary_tile, 0);
        assert!(summary.source.is_empty());
        assert!(summary
            .selector_status
            .contains("ADVANCED_SELECTOR_NOT_LOADED"));
    }

    #[test]
    fn risk_threshold_helpers_are_stable() {
        assert_eq!(quant_risk_label(0.0), "none");
        assert_eq!(quant_risk_label(1.0), "ok");
        assert_eq!(quant_risk_label(1_000.0), "warn");
        assert_eq!(quant_risk_label(1_000_000.0), "risk");

        assert_eq!(zero_point_risk_label(0), "none");
        assert_eq!(zero_point_risk_label(1), "ok");
        assert_eq!(zero_point_risk_label(80), "warn");
        assert_eq!(zero_point_risk_label(112), "risk");
        assert_eq!(
            zero_point_risk_label_for_status("reinterpret", 128),
            "reinterpret"
        );
        assert_eq!(zero_point_risk_label_for_status("asymmetric", 64), "watch");

        assert_eq!(l1_working_set_severity(0.0), "none");
        assert_eq!(l1_working_set_severity(0.5), "ok");
        assert_eq!(l1_working_set_severity(0.89), "ok");
        assert_eq!(l1_working_set_severity(0.9), "watch");
        assert_eq!(l1_working_set_severity(1.5), "warn");
        assert_eq!(l1_working_set_severity(2.5), "high");
        assert_eq!(l1_working_set_severity(3.5), "critical");
    }

    #[test]
    fn output_channels_and_alignment_follow_target_microkernel_multiple() {
        assert_eq!(output_channels_for_shape("CONV_2D", &[1, 8, 8, 32]), 32);
        assert_eq!(
            output_channels_for_shape("DEPTHWISE_CONV_2D", &[1, 8, 8, 16]),
            16
        );
        assert_eq!(output_channels_for_shape("FULLY_CONNECTED", &[1, 10]), 10);
        assert_eq!(output_channels_for_shape("RESHAPE", &[1, 8, 8, 3]), 0);

        let tensors = vec![tensor_with_shape(vec![1, 8, 8, 3])];
        let a55 = target_profile("android_mid_a55").unwrap();
        let summary = channel_alignment_for_op(
            "CONV_2D",
            &[0],
            &tensors,
            &a55,
            ChannelAlignmentContext {
                is_float_compute: false,
                compute_precision_label: "INT8",
                source_tile_multiple: 0,
                source_tile_multiples: &[],
                kernel_evidence_class: "HEURISTIC_PROFILE",
            },
        );

        assert_eq!(summary.output_channels, 3);
        assert_eq!(summary.multiple, 16);
        assert_eq!(summary.status, "misaligned");
        // inactive-lane fraction: (16 - 3) / 16
        assert_close(summary.tail_overhead_percent, 13.0 / 16.0, 1e-9);
        assert!(summary.detail.contains("modeled inactive-lane range 81%"));
        assert!(!summary.detail.contains("%.."));

        // FP32 path uses the narrower assumed multiple (16/2 = 8)
        let fp32 = channel_alignment_for_op(
            "CONV_2D",
            &[0],
            &tensors,
            &a55,
            ChannelAlignmentContext {
                is_float_compute: true,
                compute_precision_label: "FP32",
                source_tile_multiple: 0,
                source_tile_multiples: &[],
                kernel_evidence_class: "HEURISTIC_PROFILE",
            },
        );
        assert_eq!(fp32.multiple, 8);
        assert_close(fp32.tail_overhead_percent, 5.0 / 8.0, 1e-9);

        let fp16_source_candidate = channel_alignment_for_op(
            "CONV_2D",
            &[0],
            &tensors,
            &a55,
            ChannelAlignmentContext {
                is_float_compute: true,
                compute_precision_label: "FP16",
                source_tile_multiple: 16,
                source_tile_multiples: &[],
                kernel_evidence_class: "SOURCE_ENUMERATED_CANDIDATE",
            },
        );
        assert_eq!(fp16_source_candidate.multiple, 16);
        assert!(fp16_source_candidate.detail.contains("FP16"));
        assert!(fp16_source_candidate
            .detail
            .contains("pinned-source candidate"));

        let aligned_tensors = vec![tensor_with_shape(vec![1, 8, 8, 32])];
        let aligned = channel_alignment_for_op(
            "CONV_2D",
            &[0],
            &aligned_tensors,
            &a55,
            ChannelAlignmentContext {
                is_float_compute: false,
                compute_precision_label: "INT8",
                source_tile_multiple: 0,
                source_tile_multiples: &[],
                kernel_evidence_class: "HEURISTIC_PROFILE",
            },
        );
        assert_eq!(aligned.status, "aligned");
        assert_close(aligned.tail_overhead_percent, 0.0, 1e-12);
    }

    #[test]
    fn graph_output_semantic_axis_suppresses_generic_channel_padding() {
        let mut classifier = op_with(0, "CONV_2D", vec![0, 1, 2], vec![10], vec![1, 1, 1, 2]);
        classifier.channel_alignment_status = "misaligned".to_string();
        classifier.output_channels = 2;
        classifier.channel_tail_overhead_percent = 0.75;
        classifier.channel_tail_overhead_percent_min = 0.75;
        classifier.channel_tail_overhead_percent_max = 0.75;
        let reshape = op_with(1, "RESHAPE", vec![10, 11], vec![12], vec![1, 2]);
        let mut internal = op_with(2, "CONV_2D", vec![20, 21, 22], vec![23], vec![1, 8, 8, 3]);
        internal.channel_alignment_status = "misaligned".to_string();
        internal.output_channels = 3;
        internal.channel_tail_overhead_percent = 13.0 / 16.0;
        let mut segmentation_head =
            op_with(3, "CONV_2D", vec![24, 25, 26], vec![30], vec![1, 8, 8, 2]);
        segmentation_head.channel_alignment_status = "misaligned".to_string();
        segmentation_head.output_channels = 2;
        segmentation_head.channel_tail_overhead_percent = 0.75;
        segmentation_head.channel_tail_overhead_percent_min = 0.75;
        segmentation_head.channel_tail_overhead_percent_max = 0.75;
        let mut ops = vec![classifier, reshape, internal, segmentation_head];

        suppress_graph_output_channel_alignment(&mut ops, &[12, 30]);

        assert_eq!(ops[0].channel_alignment_status, "graph-output-contract");
        assert_eq!(ops[0].channel_tail_overhead_percent, 0.0);
        assert!(ops[0].channel_alignment_detail.contains("graph output"));
        assert!(ops[0]
            .static_action
            .contains("generic width/alignment rewrite suppressed"));
        assert_eq!(ops[2].channel_alignment_status, "misaligned");
        assert_eq!(ops[3].channel_alignment_status, "graph-output-contract");
        assert_eq!(ops[3].channel_tail_overhead_percent, 0.0);
        assert!(ops[3].static_action.contains("graph-output semantic axis"));
    }

    #[test]
    fn fully_connected_macs_include_every_output_batch_element() {
        let tensors = vec![
            tensor_with_index_shape(0, vec![2, 4]),
            tensor_with_index_shape(1, vec![3, 4]),
            tensor_with_index_shape(2, vec![2, 3]),
        ];
        let (macs, _, _, _) = estimate_op("FULLY_CONNECTED", &[0, 1], &[2], &tensors);
        assert_eq!(macs, 24.0);
    }

    #[test]
    fn tflite_dynamic_fully_connected_uses_signature_symbol_not_example_batch() {
        let mut input = tensor_with_index_shape(0, vec![2, 4]);
        input.shape_signature = vec![-1, 4];
        let mut weight = tensor_with_index_shape(1, vec![3, 4]);
        weight.constant_buffer = true;
        let mut output = tensor_with_index_shape(2, vec![2, 3]);
        output.shape_signature = vec![-1, 3];
        let mut op = op_with(0, "FULLY_CONNECTED", vec![0, 1], vec![2], vec![2, 3]);
        op.macs = 24.0;

        let contract =
            build_tflite_dynamic_shape_cost_contract(&[op], &[input, weight, output], &[0], &[2]);
        let value = serde_json::to_value(contract).unwrap();
        assert_eq!(value["status"], "assessed");
        assert_eq!(value["symbol_count"], 2);
        assert_eq!(
            value["op_formulas"][0]["macs_formula"]["expression"],
            "12*D1"
        );
        assert_eq!(
            value["op_formulas"][0]["declared_shape_projection_macs"],
            24
        );
        assert_eq!(
            value["op_formulas"][0]["declared_shape_projection_status"],
            "available_example_not_bound"
        );
    }

    #[test]
    fn chain_break_taxonomy_separates_zero_mac_pooling_from_structural_views() {
        let mut pooling = op_with(1, "AVERAGE_POOL_2D", vec![0], vec![1], vec![1, 1, 1, 8]);
        pooling.xnnpack_chain_break = true;
        pooling.estimated_bytes = 4096.0;
        let mut reshape = op_with(2, "RESHAPE", vec![1], vec![2], vec![1, 8]);
        reshape.xnnpack_chain_break = true;
        reshape.estimated_bytes = 4096.0;
        let mut ops = vec![pooling, reshape];

        annotate_chain_break_impact(&mut ops, &[], 1_000_000.0, 1_000_000_000.0);

        assert_eq!(ops[0].xnnpack_break_class, "zero-modeled-mac-nonstructural");
        assert_eq!(ops[1].xnnpack_break_class, "structural-zero-mac");
        assert!(!is_structural_view_op("AVERAGE_POOL_2D"));
        assert!(is_structural_view_op("RESHAPE"));
    }

    #[test]
    fn chain_break_taxonomy_keeps_exposure_and_operator_anatomy_orthogonal() {
        let mut fallback = op_with(1, "AVERAGE_POOL_2D", vec![0], vec![1], vec![1, 1, 1, 8]);
        fallback.xnnpack_chain_break = true;
        fallback.estimated_bytes = 4096.0;
        let mut ops = vec![fallback];
        let chains = vec![XnnpackChainInfo {
            id: 0,
            first_op: 0,
            last_op: 0,
            op_count: 1,
            macs: 600_000.0,
            mac_percent: 0.6,
            chain_class: "high-MAC-share candidate segment".to_string(),
            target_hint: "test".to_string(),
        }];

        annotate_chain_break_impact(&mut ops, &chains, 1_000_000.0, 1_000_000_000.0);

        assert_eq!(ops[0].xnnpack_break_class, "high-adjacent-mac-exposure");
        assert_close(ops[0].chain_break_impact_mac_percent, 0.6, 1e-12);
        assert_eq!(ops[0].macs, 0.0);
        assert!(!is_structural_view_op(&ops[0].name));
    }

    #[test]
    fn predicted_partition_boundaries_inventory_exact_graph_edges_and_payloads() {
        let mut tensors = (0..5)
            .map(|index| tensor_with_index_shape(index, vec![1, 1001]))
            .collect::<Vec<_>>();
        for tensor in &mut tensors {
            tensor.dtype = "FLOAT32".to_string();
        }
        let mut producer = op_with(0, "CONV_2D", vec![0], vec![1], vec![1, 1001]);
        producer.xnnpack_chain_id = 0;
        let mut fallback = op_with(1, "SQUEEZE", vec![1], vec![2], vec![1, 1001]);
        fallback.xnnpack_supported = false;
        fallback.xnnpack_chain_id = -1;
        let mut consumer = op_with(2, "SOFTMAX", vec![2], vec![3], vec![1, 1001]);
        consumer.xnnpack_chain_id = 1;
        let mut second_cpu_consumer = op_with(3, "RESHAPE", vec![1], vec![4], vec![1, 1001]);
        second_cpu_consumer.xnnpack_supported = false;
        second_cpu_consumer.xnnpack_chain_id = -1;

        let inventory = compute_predicted_partition_boundary_inventory(
            &[producer, fallback, consumer, second_cpu_consumer],
            &tensors,
        );
        assert_eq!(inventory.edge_count, 3);
        assert_eq!(inventory.unique_tensor_count, 2);
        assert_eq!(
            inventory.schema,
            "deepbom.predicted_partition_boundary_edges.v1.1"
        );
        assert_eq!(inventory.payload_binding, "static");
        assert_eq!(inventory.summed_edge_payload_bytes, Some(12_012));
        assert_eq!(inventory.unique_tensor_payload_bytes, Some(8_008));
        assert_eq!(inventory.payload_coverage_status, "complete");
        assert_eq!(inventory.edges[0].direction, "delegate_to_cpu");
        assert_eq!(inventory.edges[2].direction, "cpu_to_delegate");

        let mut dynamic = tensor_with_index_shape(0, vec![1, 1001]);
        dynamic.shape_signature = vec![-1, 1001];
        let batch_one = deterministic_tensor_payload_assessment(&dynamic);
        assert_eq!(batch_one.bytes, Some(1001));
        assert_eq!(batch_one.status, "assessed_serialized_batch1");
        assert_eq!(batch_one.binding, "serialized_batch1_projection");

        dynamic.shape = vec![1, 32, 1001];
        dynamic.shape_signature = vec![-1, -1, 1001];
        assert!(deterministic_tensor_payload_bytes(&dynamic).is_err());
    }

    #[test]
    fn pattern_detector_distinguishes_mobilenet_v1_from_mbconv() {
        let tensors = vec![
            tensor_with_index_shape(0, vec![1, 112, 112, 8]),
            tensor_with_index_shape(1, vec![1, 3, 3, 8]),
            tensor_with_index_shape(2, vec![1, 112, 112, 8]),
            tensor_with_index_shape(3, vec![16, 1, 1, 8]),
            tensor_with_index_shape(4, vec![1, 112, 112, 16]),
        ];
        let ops = vec![
            op_with(
                0,
                "DEPTHWISE_CONV_2D",
                vec![0, 1],
                vec![2],
                vec![1, 112, 112, 8],
            ),
            op_with(1, "CONV_2D", vec![2, 3], vec![4], vec![1, 112, 112, 16]),
        ];
        let patterns = detect_patterns(&ops, &tensors);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].name, "Depthwise-separable convolution pair");
        assert!(!patterns.iter().any(|p| p.name == "MBConv-like block"));
    }

    #[test]
    fn pattern_detector_requires_expansion_projection_and_residual_for_mbconv() {
        let tensors = vec![
            tensor_with_index_shape(0, vec![1, 56, 56, 16]),
            tensor_with_index_shape(1, vec![96, 1, 1, 16]),
            tensor_with_index_shape(2, vec![1, 56, 56, 96]),
            tensor_with_index_shape(3, vec![1, 3, 3, 96]),
            tensor_with_index_shape(4, vec![1, 56, 56, 96]),
            tensor_with_index_shape(5, vec![16, 1, 1, 96]),
            tensor_with_index_shape(6, vec![1, 56, 56, 16]),
            tensor_with_index_shape(7, vec![1, 56, 56, 16]),
        ];
        let without_residual = vec![
            op_with(0, "CONV_2D", vec![0, 1], vec![2], vec![1, 56, 56, 96]),
            op_with(
                1,
                "DEPTHWISE_CONV_2D",
                vec![2, 3],
                vec![4],
                vec![1, 56, 56, 96],
            ),
            op_with(2, "CONV_2D", vec![4, 5], vec![6], vec![1, 56, 56, 16]),
        ];
        assert!(!detect_patterns(&without_residual, &tensors)
            .iter()
            .any(|p| p.name == "MBConv-like block"));

        let with_residual = vec![
            op_with(0, "CONV_2D", vec![0, 1], vec![2], vec![1, 56, 56, 96]),
            op_with(
                1,
                "DEPTHWISE_CONV_2D",
                vec![2, 3],
                vec![4],
                vec![1, 56, 56, 96],
            ),
            op_with(2, "CONV_2D", vec![4, 5], vec![6], vec![1, 56, 56, 16]),
            op_with(3, "ADD", vec![6, 0], vec![7], vec![1, 56, 56, 16]),
        ];
        let patterns = detect_patterns(&with_residual, &tensors);
        assert!(patterns.iter().any(|p| p.name == "MBConv-like block"));
        let mbconv = patterns
            .iter()
            .find(|p| p.name == "MBConv-like block")
            .expect("MBConv-like block should be detected");
        assert_eq!(mbconv.first_op, 0);
        assert_eq!(mbconv.last_op, 3);
        assert_eq!(mbconv.op_count, 4);
        assert!(!patterns
            .iter()
            .any(|p| p.name == "Depthwise-separable convolution pair"));
    }

    #[test]
    fn arena_projection_models_pinned_reshape_alias_and_declared_shape_bytes() {
        let mut tensors = vec![
            tensor_with_index_shape(0, vec![1, 16]),
            tensor_with_index_shape(1, vec![1, 128]),
            tensor_with_index_shape(2, vec![1, 128]),
            tensor_with_index_shape(3, vec![1, 64]),
        ];
        tensors[1].shape_signature = vec![-1, 128];
        let ops = vec![
            op_with(0, "CONV_2D", vec![0], vec![1], vec![1, 128]),
            op_with(1, "RESHAPE", vec![1], vec![2], vec![1, 128]),
            op_with(2, "CONV_2D", vec![2], vec![3], vec![1, 64]),
        ];

        let projection = compute_tensor_arena_plan(&ops, &tensors, &[0], &[3]);
        assert_eq!(projection.status, "assessed");
        assert_eq!(projection.source_commit, TFLITE_ARENA_SOURCE_COMMIT);
        assert_eq!(projection.tensor_alignment_bytes, 64);
        assert_eq!(projection.non_persistent_arena_bytes, Some(256));
        assert_eq!(projection.persistent_arena_bytes, Some(0));
        assert_eq!(projection.combined_arena_bytes, Some(256));
        assert_eq!(projection.root_allocation_count, 3);
        assert_eq!(projection.shared_tensor_count, 1);
        assert_eq!(projection.dynamic_shape_signature_tensor_count, 1);
        assert_eq!(projection.source_comparator_tie_group_count, 0);
        assert!(projection.source_comparator_fully_orders_projection);
        assert_eq!(projection.aliases[0].tensor_index, 2);
        assert_eq!(projection.aliases[0].shared_with_tensor_index, 1);
        let root = projection
            .allocations
            .iter()
            .find(|allocation| allocation.tensor_index == 1)
            .unwrap();
        let alias = projection
            .allocations
            .iter()
            .find(|allocation| allocation.tensor_index == 2)
            .unwrap();
        assert_eq!(root.offset_bytes, Some(64));
        assert_eq!(alias.offset_bytes, root.offset_bytes);
        assert_eq!(alias.allocation_status, "shared_in_place");
    }

    #[test]
    fn unknown_width_suppresses_arena_and_liveness_totals() {
        let mut tensors = vec![
            tensor_with_index_shape(0, vec![1, 16]),
            tensor_with_index_shape(1, vec![1]),
        ];
        tensors[1].dtype = "STRING".to_string();
        let ops = vec![op_with(0, "RESHAPE", vec![0], vec![1], vec![1])];

        let projection = compute_tensor_arena_plan(&ops, &tensors, &[0], &[1]);
        assert_eq!(projection.status, "partial");
        assert_eq!(projection.non_persistent_arena_bytes, None);
        assert_eq!(projection.combined_arena_bytes, None);
        assert_eq!(projection.unassessed_tensor_count, 1);

        let liveness = compute_tensor_liveness(&ops, &tensors, &[0], &[1]);
        assert!(!liveness.assessed);
        assert_eq!(liveness.peak_bytes_value, None);
        assert_eq!(liveness.unassessed_tensor_count, 1);
    }
}
