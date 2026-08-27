use super::*;

#[derive(Serialize)]
pub(super) struct TfliteSubgraphDeepAnalysis {
    schema: &'static str,
    status: &'static str,
    evidence_class: &'static str,
    assessed_subgraph_count: usize,
    subgraph_count: usize,
    primary_subgraph_index: usize,
    rows: Vec<TfliteSubgraphDeepRow>,
    method: &'static str,
    execution_count_boundary: &'static str,
}

#[derive(Serialize)]
pub(super) struct TfliteSubgraphDeepRow {
    subgraph_index: usize,
    name: String,
    status: &'static str,
    evidence_class: &'static str,
    reachable_from_entrypoint: bool,
    invocation_semantics: String,
    operator_count: usize,
    tensor_count: usize,
    total_macs: f64,
    total_ops: f64,
    quantized_tensor_count: usize,
    per_axis_tensor_count: usize,
    quantization_status: QuantizationStatus,
    delegate: ScopedDelegateSummary,
    predicted_partition_boundaries: PredictedPartitionBoundaryInventory,
    tensor_liveness: TensorLiveness,
    tensor_arena_plan: TensorArenaPlanProjection,
    movement_analysis: MovementAnalysis,
    weight_integrity: WeightIntegrityReport,
    operator_evidence: Vec<ScopedOpEvidence>,
    advanced_numerical_storage: &'static str,
    advanced_numerical_evidence_pointers: Vec<String>,
    advanced_numerical_evidence: Option<ScopedAdvancedNumericalEvidence>,
    interpretation_boundary: &'static str,
}

#[derive(Serialize)]
struct ScopedDelegateSummary {
    assignment_evidence_class: &'static str,
    target_profile_id: String,
    assessed_operator_count: usize,
    predicted_delegated_operator_count: usize,
    predicted_fallback_operator_count: usize,
    chain_count: usize,
    predicted_chain_break_count: usize,
    predicted_delegated_macs: f64,
    predicted_fallback_macs: f64,
    predicted_delegated_mac_ratio: f64,
    interpretation_boundary: &'static str,
}

#[derive(Serialize)]
struct ScopedOpEvidence {
    operator_index: usize,
    name: String,
    version: i32,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    output_shapes: Vec<Vec<i32>>,
    nominal_macs: f64,
    nominal_ops: f64,
    logical_io_bytes: f64,
    row_payload_bytes: Option<usize>,
    quantization_state: String,
    quantization_risk: String,
    xnnpack_source_candidate: bool,
    xnnpack_reason: String,
    predicted_chain_id: i32,
    predicted_chain_role: String,
    predicted_break_class: String,
    weight_bytes: f64,
    channel_alignment_status: String,
}

#[derive(Serialize)]
struct ScopedAdvancedNumericalEvidence {
    accumulator_atlas: AccumulatorAtlasAnalysis,
    requantization_fidelity: RequantizationFidelityAnalysis,
    quantization_lattice: QuantizationLatticeAnalysis,
    advanced_proof_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    kernel_extremum_witness: Option<KernelWitnessAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel_vitality: Option<ChannelVitalityAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rounding_equivalence: Option<RoundingEquivalenceAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accumulator_reachability: Option<AccumulatorReachabilityAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    numerical_abi_propagation: Option<NumericalAbiPropagationAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    contract_migration: Option<ContractMigrationAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    residual_step_response: Option<ResidualStepResponseAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    residual_contract_distortion: Option<ResidualContractDistortionAnalysis>,
}

pub(super) struct PrimaryScopeInput<'a> {
    pub subgraph_index: usize,
    pub name: &'a str,
    pub reachable_from_entrypoint: bool,
    pub invocation_semantics: &'a str,
    pub ops: &'a [OpInfo],
    pub tensors: &'a [TensorInfo],
    pub total_macs: f64,
    pub total_ops: f64,
    pub quantized_tensor_count: usize,
    pub per_axis_tensor_count: usize,
    pub quantization_status: &'a QuantizationStatus,
    pub chains: &'a [XnnpackChainInfo],
    pub predicted_partition_boundaries: &'a PredictedPartitionBoundaryInventory,
    pub tensor_liveness: &'a TensorLiveness,
    pub tensor_arena_plan: &'a TensorArenaPlanProjection,
    pub movement_analysis: &'a MovementAnalysis,
    pub weight_integrity: &'a WeightIntegrityReport,
    pub target: &'a TargetProfile,
}

pub(super) struct NestedScopeInput<'a> {
    pub subgraph_index: usize,
    pub name: &'a str,
    pub reachable_from_entrypoint: bool,
    pub invocation_semantics: &'a str,
    pub ops: Vec<OpInfo>,
    pub tensors: &'a [TensorInfo],
    pub input_tensor_indices: &'a [i32],
    pub output_tensor_indices: &'a [i32],
    pub model_bytes: &'a [u8],
    pub target: &'a TargetProfile,
    pub include_advanced_proofs: bool,
}

pub(super) fn build_deep_analysis(
    primary: PrimaryScopeInput<'_>,
    nested: Vec<NestedScopeInput<'_>>,
) -> Result<TfliteSubgraphDeepAnalysis, String> {
    let expected = 1usize
        .checked_add(nested.len())
        .ok_or_else(|| "TFLite deep-scope count overflow".to_string())?;
    let mut rows = Vec::with_capacity(expected);
    rows.push(build_primary_row(primary));
    for scope in nested {
        rows.push(build_nested_row(scope)?);
    }
    rows.sort_by_key(|row| row.subgraph_index);
    if rows.len() != expected
        || rows
            .iter()
            .enumerate()
            .any(|(index, row)| row.subgraph_index != index)
    {
        return Err(
            "TFLite deep-scope analysis does not conserve serialized subgraph identity".to_string(),
        );
    }
    Ok(TfliteSubgraphDeepAnalysis {
        schema: "deepbom.tflite_subgraph_deep_analysis.v1",
        status: "assessed_all_serialized_subgraphs",
        evidence_class: "OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED",
        assessed_subgraph_count: rows.len(),
        subgraph_count: rows.len(),
        primary_subgraph_index: 0,
        rows,
        method: "Build every subgraph's operator evidence through the same target-aware TFLite op builder, then independently derive quantization, XNNPACK source-candidate chains and tensor boundaries, logical liveness, pinned ArenaPlanner placement, movement, stored-weight integrity, accumulator, requantization, and applicable fixed-point proof ledgers.",
        execution_count_boundary: "Every row is one serialized subgraph scope. Rows are never summed into a model execution total because IF branches are conditional, WHILE/computation bodies can execute repeatedly, CALL_ONCE is lifecycle-scoped, and SignatureDef entrypoints are alternatives. Predicted XNNPACK placement is not observed delegation.",
    })
}

fn build_primary_row(input: PrimaryScopeInput<'_>) -> TfliteSubgraphDeepRow {
    TfliteSubgraphDeepRow {
        subgraph_index: input.subgraph_index,
        name: input.name.to_string(),
        status: "assessed_referenced_top_level",
        evidence_class: "OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED",
        reachable_from_entrypoint: input.reachable_from_entrypoint,
        invocation_semantics: input.invocation_semantics.to_string(),
        operator_count: input.ops.len(),
        tensor_count: input.tensors.len(),
        total_macs: input.total_macs,
        total_ops: input.total_ops,
        quantized_tensor_count: input.quantized_tensor_count,
        per_axis_tensor_count: input.per_axis_tensor_count,
        quantization_status: input.quantization_status.clone(),
        delegate: delegate_summary(input.ops, input.chains, input.total_macs, input.target),
        predicted_partition_boundaries: input.predicted_partition_boundaries.clone(),
        tensor_liveness: input.tensor_liveness.clone(),
        tensor_arena_plan: input.tensor_arena_plan.clone(),
        movement_analysis: input.movement_analysis.clone(),
        weight_integrity: input.weight_integrity.clone(),
        operator_evidence: op_evidence(input.ops),
        advanced_numerical_storage: "referenced_top_level_without_duplication",
        advanced_numerical_evidence_pointers: primary_evidence_pointers(),
        advanced_numerical_evidence: None,
        interpretation_boundary: scope_boundary(),
    }
}

fn build_nested_row(input: NestedScopeInput<'_>) -> Result<TfliteSubgraphDeepRow, String> {
    let mut ops = input.ops;
    suppress_graph_output_channel_alignment(&mut ops, input.output_tensor_indices);
    annotate_fusion(&mut ops);
    let patterns = detect_patterns(&ops, input.tensors);
    annotate_patterns(&mut ops, &patterns);
    let total_macs = ops.iter().map(|op| op.macs).sum::<f64>().abs();
    let total_ops = ops.iter().map(|op| op.ops).sum::<f64>().abs();
    let total_estimated_bytes = ops.iter().map(|op| op.estimated_bytes).sum::<f64>().abs();
    let mut chains = annotate_xnnpack_chains(&mut ops, input.target);
    for chain in &mut chains {
        chain.mac_percent = if total_macs > 0.0 {
            chain.macs / total_macs
        } else {
            0.0
        };
        chain.chain_class = xnnpack_chain_class(chain);
    }
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
    annotate_chain_break_impact(&mut ops, &chains, total_macs, total_estimated_bytes);
    compute_topology_annotations(&mut ops);
    compute_bottleneck_estimates(&mut ops, input.target);

    let inputs = input
        .input_tensor_indices
        .iter()
        .filter_map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|index| input.tensors.get(index))
                .cloned()
        })
        .collect::<Vec<_>>();
    let outputs = input
        .output_tensor_indices
        .iter()
        .filter_map(|index| {
            usize::try_from(*index)
                .ok()
                .and_then(|index| input.tensors.get(index))
                .cloned()
        })
        .collect::<Vec<_>>();
    let quantized_tensor_count = input
        .tensors
        .iter()
        .filter(|tensor| tensor.quant_scales > 0)
        .count();
    let per_axis_tensor_count = input
        .tensors
        .iter()
        .filter(|tensor| tensor.quant_scales > 1)
        .count();
    let quantization_status = classify_model_quantization(
        &ops,
        input.tensors,
        &inputs,
        &outputs,
        quantized_tensor_count,
        total_macs,
    );
    let predicted_partition_boundaries =
        compute_predicted_partition_boundary_inventory(&ops, input.tensors);
    let tensor_liveness = compute_tensor_liveness(
        &ops,
        input.tensors,
        input.input_tensor_indices,
        input.output_tensor_indices,
    );
    let tensor_arena_plan = compute_tensor_arena_plan(
        &ops,
        input.tensors,
        input.input_tensor_indices,
        input.output_tensor_indices,
    );
    let movement_analysis = compute_movement_analysis(&ops, input.tensors);
    let weight_integrity =
        compute_weight_integrity(input.model_bytes, input.tensors, &ops, total_macs);
    let accumulator_atlas = build_accumulator_atlas(input.model_bytes, &ops, input.tensors);
    let requantization_fidelity =
        build_requantization_fidelity(&ops, input.tensors, &accumulator_atlas);
    let quantization_lattice = build_quantization_lattice(&ops, input.tensors);

    let advanced_numerical_evidence = if input.include_advanced_proofs {
        let kernel_extremum_witness = build_kernel_witnesses(
            input.model_bytes,
            &ops,
            input.tensors,
            &accumulator_atlas,
            &requantization_fidelity,
        );
        let channel_vitality = build_channel_vitality(&kernel_extremum_witness);
        let rounding_equivalence =
            build_rounding_equivalence(&kernel_extremum_witness, &requantization_fidelity);
        let accumulator_reachability = build_accumulator_reachability(
            input.model_bytes,
            &ops,
            input.tensors,
            &kernel_extremum_witness,
            &requantization_fidelity,
            &rounding_equivalence,
        );
        let numerical_abi_propagation = build_numerical_abi_propagation(
            &ops,
            input.tensors,
            input.output_tensor_indices,
            &rounding_equivalence,
            &accumulator_reachability,
        );
        let contract_migration = build_contract_migration(
            input.model_bytes,
            &ops,
            input.tensors,
            &quantization_lattice,
            &accumulator_atlas,
        );
        let residual_step_response =
            build_residual_step_response(&ops, input.tensors, &quantization_lattice);
        let residual_contract_distortion =
            build_residual_contract_distortion(&ops, input.tensors, &quantization_lattice);
        Some(ScopedAdvancedNumericalEvidence {
            accumulator_atlas,
            requantization_fidelity,
            quantization_lattice,
            advanced_proof_status: "assessed_full_scope",
            kernel_extremum_witness: Some(kernel_extremum_witness),
            channel_vitality: Some(channel_vitality),
            rounding_equivalence: Some(rounding_equivalence),
            accumulator_reachability: Some(accumulator_reachability),
            numerical_abi_propagation: Some(numerical_abi_propagation),
            contract_migration: Some(contract_migration),
            residual_step_response: Some(residual_step_response),
            residual_contract_distortion: Some(residual_contract_distortion),
        })
    } else {
        Some(ScopedAdvancedNumericalEvidence {
            accumulator_atlas,
            requantization_fidelity,
            quantization_lattice,
            advanced_proof_status: "not_computed_for_fast_scope",
            kernel_extremum_witness: None,
            channel_vitality: None,
            rounding_equivalence: None,
            accumulator_reachability: None,
            numerical_abi_propagation: None,
            contract_migration: None,
            residual_step_response: None,
            residual_contract_distortion: None,
        })
    };

    Ok(TfliteSubgraphDeepRow {
        subgraph_index: input.subgraph_index,
        name: input.name.to_string(),
        status: "assessed_embedded_scope_evidence",
        evidence_class: "OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED",
        reachable_from_entrypoint: input.reachable_from_entrypoint,
        invocation_semantics: input.invocation_semantics.to_string(),
        operator_count: ops.len(),
        tensor_count: input.tensors.len(),
        total_macs,
        total_ops,
        quantized_tensor_count,
        per_axis_tensor_count,
        quantization_status,
        delegate: delegate_summary(&ops, &chains, total_macs, input.target),
        predicted_partition_boundaries,
        tensor_liveness,
        tensor_arena_plan,
        movement_analysis,
        weight_integrity,
        operator_evidence: op_evidence(&ops),
        advanced_numerical_storage: "embedded_in_scope_row",
        advanced_numerical_evidence_pointers: Vec::new(),
        advanced_numerical_evidence,
        interpretation_boundary: scope_boundary(),
    })
}

fn delegate_summary(
    ops: &[OpInfo],
    chains: &[XnnpackChainInfo],
    total_macs: f64,
    target: &TargetProfile,
) -> ScopedDelegateSummary {
    let delegated_macs = ops
        .iter()
        .filter(|op| op.xnnpack_chain_id >= 0)
        .map(|op| op.macs)
        .sum::<f64>()
        .abs();
    ScopedDelegateSummary {
        assignment_evidence_class: "PREDICTED_SOURCE_PINNED",
        target_profile_id: target.id.clone(),
        assessed_operator_count: ops.len(),
        predicted_delegated_operator_count: ops.iter().filter(|op| op.xnnpack_chain_id >= 0).count(),
        predicted_fallback_operator_count: ops.iter().filter(|op| op.xnnpack_chain_id < 0).count(),
        chain_count: chains.len(),
        predicted_chain_break_count: ops.iter().filter(|op| op.xnnpack_chain_break).count(),
        predicted_delegated_macs: delegated_macs,
        predicted_fallback_macs: (total_macs - delegated_macs).max(0.0),
        predicted_delegated_mac_ratio: if total_macs > 0.0 { delegated_macs / total_macs } else { 0.0 },
        interpretation_boundary: "Pinned static XNNPACK source-candidate prediction for this subgraph and selected target profile; not selected-build inclusion, runtime partitioning, lowering, or execution.",
    }
}

fn op_evidence(ops: &[OpInfo]) -> Vec<ScopedOpEvidence> {
    ops.iter()
        .map(|op| ScopedOpEvidence {
            operator_index: op.index,
            name: op.name.clone(),
            version: op.version,
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            output_shapes: op.output_shapes.clone(),
            nominal_macs: op.macs,
            nominal_ops: op.ops,
            logical_io_bytes: op.estimated_bytes,
            row_payload_bytes: op.cache_payload.logical_row_payload_bytes,
            quantization_state: op.quantization_state.clone(),
            quantization_risk: op.quant_risk.clone(),
            xnnpack_source_candidate: op.xnnpack_supported,
            xnnpack_reason: op.xnnpack_reason.clone(),
            predicted_chain_id: op.xnnpack_chain_id,
            predicted_chain_role: op.xnnpack_chain_role.clone(),
            predicted_break_class: op.xnnpack_break_class.clone(),
            weight_bytes: op.weight_bytes,
            channel_alignment_status: op.channel_alignment_status.clone(),
        })
        .collect()
}

fn primary_evidence_pointers() -> Vec<String> {
    [
        "/ops",
        "/quantization_status",
        "/predicted_partition_boundaries",
        "/tensor_liveness",
        "/tensor_arena_plan",
        "/movement_analysis",
        "/weight_integrity",
        "/accumulator_atlas",
        "/requantization_fidelity",
        "/kernel_extremum_witness",
        "/channel_vitality",
        "/rounding_equivalence",
        "/accumulator_reachability",
        "/numerical_abi_propagation",
        "/quantization_lattice",
        "/contract_migration",
        "/residual_step_response",
        "/residual_contract_distortion",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn scope_boundary() -> &'static str {
    "This row is an intrinsic one-subgraph analysis. Cross-subgraph execution counts, liveness, arena sharing, delegate partition fusion, and latency require runtime invocation evidence and are not inferred or summed."
}
