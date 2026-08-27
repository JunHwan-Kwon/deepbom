use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use crate::verified_flatbuffer::Fb;
use crate::{deterministic_tensor_payload_assessment, estimate_op, TensorInfo};

const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const SCHEMA_SOURCE: &str = "tensorflow/compiler/mlir/lite/schema/schema.fbs";
const SCHEMA_SHA256: &str = "3bfa613428459de18db5d70d8581e7b6afd127c4522bb18ff59c8e589c3b75a1";
const IF_SOURCE: &str = "tensorflow/lite/kernels/if.cc";
const IF_SOURCE_SHA256: &str = "e23c06a2a3984ae704153c667f891be2bbd31d03be523a85cc976ea6ca75a428";
const WHILE_SOURCE: &str = "tensorflow/lite/kernels/while.cc";
const WHILE_SOURCE_SHA256: &str =
    "6ac2e230d01317af3c39bb2a730c8d1aa632cc707f9043b3d8bb6e2123389005";
const CALL_ONCE_SOURCE: &str = "tensorflow/lite/kernels/call_once.cc";
const CALL_ONCE_SOURCE_SHA256: &str =
    "21bf79b3ba4c47bf63090b5cf3b7734591f79ddcecc8b4364b18997ca159a44d";
const CONTROL_FLOW_COMMON_SOURCE: &str = "tensorflow/lite/kernels/control_flow_common.h";
const CONTROL_FLOW_COMMON_SOURCE_SHA256: &str =
    "acce5ae7657930db1eac7a7548c8e74ff91314fa2ad03681d3968125f6665a46";
const CONV_2D_SOURCE: &str = "tensorflow/lite/kernels/conv.cc";
const CONV_2D_SOURCE_SHA256: &str =
    "66a2fef9a8e7fe81b7bdd9d18bd099cc589546ac29cca7665711de890fba9281";
const DEPTHWISE_CONV_2D_SOURCE: &str = "tensorflow/lite/kernels/depthwise_conv.cc";
const DEPTHWISE_CONV_2D_SOURCE_SHA256: &str =
    "343f85c01e6adf2b21dbcd7e610ae04acf78f4ba1fea912e2fb02e33c92f6629";
const FULLY_CONNECTED_SOURCE: &str = "tensorflow/lite/kernels/fully_connected.cc";
const FULLY_CONNECTED_SOURCE_SHA256: &str =
    "a2667242af7d0d933d31408a0393974718e82da221248db9cb25aac2a8d3c585";
const CONV_3D_SOURCE: &str = "tensorflow/lite/kernels/conv3d.cc";
const CONV_3D_SOURCE_SHA256: &str =
    "7dfd75d047b7d22f76c365d48ecb1facad4656897ed3d58a661afcb0ad503b36";

#[derive(Serialize)]
struct OperatorCount {
    name: String,
    count: usize,
}

#[derive(Clone, Serialize)]
struct SubgraphTensorIntrinsic {
    tensor_index: usize,
    name: String,
    shape: Vec<i32>,
    shape_signature: Vec<i32>,
    dtype: String,
    logical_payload_bytes: Option<usize>,
    payload_status: &'static str,
    payload_binding: &'static str,
    payload_reason: String,
    constant_buffer: bool,
    sparse_storage: bool,
    buffer_data_offset: usize,
    buffer_data_length: usize,
}

#[derive(Clone, Serialize)]
pub(super) struct SubgraphOperatorIntrinsic {
    pub(super) operator_index: usize,
    pub(super) name: String,
    pub(super) version: i32,
    pub(super) inputs: Vec<i32>,
    pub(super) outputs: Vec<i32>,
    nominal_macs: Option<f64>,
    nominal_macs_decimal: Option<String>,
    mac_assessment_status: &'static str,
    mac_formula_class: &'static str,
    mac_assessment_reason: String,
    logical_io_payload_bytes: Option<usize>,
    assessed_logical_io_payload_bytes: usize,
    logical_io_payload_status: &'static str,
    present_io_tensor_slot_count: usize,
    assessed_io_tensor_slot_count: usize,
    unassessed_io_tensor_slot_count: usize,
    #[serde(skip)]
    pub(super) raw_macs: f64,
    #[serde(skip)]
    pub(super) raw_ops: f64,
    #[serde(skip)]
    pub(super) raw_estimated_bytes: f64,
    #[serde(skip)]
    pub(super) raw_estimated_input_strip: f64,
}

#[derive(Serialize)]
struct SubgraphIntrinsicCost {
    schema: &'static str,
    status: &'static str,
    evidence_class: &'static str,
    invocation_basis: &'static str,
    mac_compute_operator_count: usize,
    assessed_nominal_mac_operator_count: usize,
    modeled_scenario_mac_operator_count: usize,
    unassessed_mac_operator_count: usize,
    complete_nominal_macs: Option<f64>,
    complete_nominal_macs_decimal: Option<String>,
    assessed_nominal_macs: Option<f64>,
    assessed_nominal_macs_decimal: String,
    modeled_scenario_macs: Option<f64>,
    modeled_scenario_macs_decimal: String,
    logical_tensor_payload_bytes: Option<usize>,
    assessed_logical_tensor_payload_bytes: usize,
    assessed_tensor_payload_count: usize,
    unassessed_tensor_payload_count: usize,
    logical_operator_io_payload_bytes: Option<usize>,
    assessed_logical_operator_io_payload_bytes: usize,
    assessed_operator_io_tensor_slot_count: usize,
    unassessed_operator_io_tensor_slot_count: usize,
    graph_input_payload_bytes: Option<usize>,
    graph_output_payload_bytes: Option<usize>,
    logical_constant_reference_bytes: usize,
    physical_unique_constant_bytes: usize,
    physical_unique_constant_buffer_count: usize,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Serialize)]
pub(super) struct SubgraphRow {
    pub(super) subgraph_index: usize,
    pub(super) name: String,
    tensor_count: usize,
    pub(super) input_tensor_indices: Vec<i32>,
    pub(super) output_tensor_indices: Vec<i32>,
    operator_count: usize,
    constant_tensor_count: usize,
    quantized_tensor_count: usize,
    per_axis_tensor_count: usize,
    sparse_tensor_count: usize,
    control_flow_reference_count: usize,
    incoming_reference_count: usize,
    pub(super) reachable_from_entrypoint: bool,
    pub(super) invocation_semantics: String,
    intrinsic_cost: SubgraphIntrinsicCost,
    tensor_intrinsics: Vec<SubgraphTensorIntrinsic>,
    pub(super) operator_intrinsics: Vec<SubgraphOperatorIntrinsic>,
    operator_histogram: Vec<OperatorCount>,
}

#[derive(Serialize)]
struct SubgraphReference {
    source_subgraph_index: usize,
    source_op_index: usize,
    source_op_name: String,
    role: String,
    target_subgraph_index: usize,
    target_subgraph_name: String,
}

#[derive(Serialize)]
struct SignatureEntrypoint {
    signature_key: String,
    subgraph_index: usize,
    subgraph_name: String,
}

#[derive(Serialize)]
struct ControlFlowContract {
    source_subgraph_index: usize,
    source_op_index: usize,
    source_op_name: String,
    status: &'static str,
    source_input_count: usize,
    source_output_count: usize,
    target_subgraph_indices: Vec<usize>,
    condition_contract_status: Option<&'static str>,
    method: String,
}

#[derive(Serialize)]
struct SourceFile {
    role: &'static str,
    path: &'static str,
    sha256: &'static str,
}

#[derive(Serialize)]
pub(super) struct TfliteSubgraphInventory {
    schema: &'static str,
    status: &'static str,
    evidence_class: &'static str,
    subgraph_count: usize,
    parsed_subgraph_count: usize,
    primary_subgraph_index: usize,
    primary_operator_count: usize,
    primary_tensor_count: usize,
    serialized_operator_count: usize,
    serialized_tensor_count: usize,
    nested_operator_count: usize,
    nested_tensor_count: usize,
    control_flow_reference_count: usize,
    signature_entrypoint_count: usize,
    reachable_subgraph_count: usize,
    unreachable_subgraph_indices: Vec<usize>,
    control_flow_contract_count: usize,
    assessed_control_flow_contract_count: usize,
    partial_control_flow_contract_count: usize,
    pub(super) rows: Vec<SubgraphRow>,
    references: Vec<SubgraphReference>,
    signature_entrypoints: Vec<SignatureEntrypoint>,
    control_flow_contracts: Vec<ControlFlowContract>,
    source_commit: &'static str,
    schema_source: &'static str,
    schema_source_sha256: &'static str,
    control_flow_sources: Vec<SourceFile>,
    nominal_mac_sources: Vec<SourceFile>,
    method: &'static str,
    execution_count_boundary: &'static str,
}

pub(super) fn build_tflite_subgraph_inventory(
    fb: &Fb<'_>,
    model: usize,
    operator_names: &[String],
    operator_versions: &[i32],
    scoped_tensors: &[Vec<TensorInfo>],
) -> Result<TfliteSubgraphInventory, String> {
    let subgraphs = fb.vector_tables(model, 2);
    if subgraphs.is_empty() || subgraphs.len() != scoped_tensors.len() {
        return Err("TFLite subgraph table/tensor-scope cardinality mismatch".to_string());
    }
    let names = subgraphs
        .iter()
        .enumerate()
        .map(|(index, &table)| {
            fb.checked_string_field(table, 4, "SubGraph.name")
                .map(|name| {
                    name.filter(|value| !value.is_empty())
                        .unwrap_or_else(|| format!("subgraph_{index}"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::with_capacity(subgraphs.len());
    let mut references = Vec::new();
    let mut control_flow_contracts = Vec::new();
    for (subgraph_index, (&table, tensors)) in subgraphs.iter().zip(scoped_tensors).enumerate() {
        let inputs = checked_tensor_indices(fb, table, 1, "SubGraph.inputs", tensors.len(), true)?;
        let outputs =
            checked_tensor_indices(fb, table, 2, "SubGraph.outputs", tensors.len(), true)?;
        let operators = fb.vector_tables(table, 3);
        let tensor_intrinsics = tensors
            .iter()
            .map(build_tensor_intrinsic)
            .collect::<Vec<_>>();
        let mut histogram = BTreeMap::<String, usize>::new();
        let mut operator_intrinsics = Vec::with_capacity(operators.len());
        let reference_start = references.len();
        for (op_index, &operator) in operators.iter().enumerate() {
            let opcode_index =
                fb.checked_u32_field(operator, 0, 0, "Operator.opcode_index")? as usize;
            let op_name = operator_names.get(opcode_index).ok_or_else(|| {
                format!("Subgraph {subgraph_index} operator {op_index} references missing opcode {opcode_index}")
            })?;
            let op_version = operator_versions.get(opcode_index).copied().ok_or_else(|| {
                format!("Subgraph {subgraph_index} operator {op_index} references missing opcode version {opcode_index}")
            })?;
            let op_inputs =
                checked_tensor_indices(fb, operator, 1, "Operator.inputs", tensors.len(), true)?;
            let op_outputs =
                checked_tensor_indices(fb, operator, 2, "Operator.outputs", tensors.len(), true)?;
            operator_intrinsics.push(build_operator_intrinsic(
                op_index,
                op_name,
                op_version,
                op_inputs.clone(),
                op_outputs.clone(),
                tensors,
                &tensor_intrinsics,
            )?);
            *histogram.entry(op_name.clone()).or_default() += 1;
            let op_reference_start = references.len();
            collect_subgraph_references(
                fb,
                operator,
                subgraph_index,
                op_index,
                op_name,
                &names,
                &mut references,
            )?;
            if let Some(contract) = validate_control_flow_contract(
                fb,
                &subgraphs,
                scoped_tensors,
                subgraph_index,
                op_index,
                op_name,
                (&op_inputs, &op_outputs, &references[op_reference_start..]),
            )? {
                control_flow_contracts.push(contract);
            }
        }
        let intrinsic_cost = build_subgraph_intrinsic_cost(
            &tensor_intrinsics,
            &inputs,
            &outputs,
            &operator_intrinsics,
        )?;
        rows.push(SubgraphRow {
            subgraph_index,
            name: names[subgraph_index].clone(),
            tensor_count: tensors.len(),
            input_tensor_indices: inputs,
            output_tensor_indices: outputs,
            operator_count: operators.len(),
            constant_tensor_count: tensors
                .iter()
                .filter(|tensor| tensor.constant_buffer)
                .count(),
            quantized_tensor_count: tensors
                .iter()
                .filter(|tensor| tensor.quant_scales > 0)
                .count(),
            per_axis_tensor_count: tensors
                .iter()
                .filter(|tensor| tensor.quant_scales > 1)
                .count(),
            sparse_tensor_count: tensors
                .iter()
                .filter(|tensor| tensor.sparse_storage)
                .count(),
            control_flow_reference_count: references.len() - reference_start,
            incoming_reference_count: 0,
            reachable_from_entrypoint: false,
            invocation_semantics: String::new(),
            intrinsic_cost,
            tensor_intrinsics,
            operator_intrinsics,
            operator_histogram: histogram
                .into_iter()
                .map(|(name, count)| OperatorCount { name, count })
                .collect(),
        });
    }
    for reference in &references {
        rows[reference.target_subgraph_index].incoming_reference_count += 1;
    }
    let signature_entrypoints = fb
        .vector_tables(model, 7)
        .into_iter()
        .map(|signature| {
            let subgraph_index =
                fb.checked_u32_field(signature, 3, 0, "SignatureDef.subgraph_index")? as usize;
            let subgraph_name = names.get(subgraph_index).ok_or_else(|| {
                format!("SignatureDef references missing subgraph {subgraph_index}")
            })?;
            Ok(SignatureEntrypoint {
                signature_key: fb
                    .checked_string_field(signature, 2, "SignatureDef.signature_key")?
                    .unwrap_or_default(),
                subgraph_index,
                subgraph_name: subgraph_name.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut reachable = BTreeSet::from([0usize]);
    reachable.extend(
        signature_entrypoints
            .iter()
            .map(|entry| entry.subgraph_index),
    );
    let mut queue = VecDeque::from_iter(reachable.iter().copied());
    while let Some(source) = queue.pop_front() {
        for target in references
            .iter()
            .filter(|reference| reference.source_subgraph_index == source)
            .map(|reference| reference.target_subgraph_index)
        {
            if reachable.insert(target) {
                queue.push_back(target);
            }
        }
    }
    for row in &mut rows {
        row.reachable_from_entrypoint = reachable.contains(&row.subgraph_index);
        row.invocation_semantics =
            invocation_semantics(row.subgraph_index, &references, &signature_entrypoints);
    }
    let serialized_operator_count = rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row.operator_count)
            .ok_or_else(|| "TFLite serialized operator count overflows".to_string())
    })?;
    let serialized_tensor_count = rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row.tensor_count)
            .ok_or_else(|| "TFLite serialized tensor count overflows".to_string())
    })?;
    let primary_operator_count = rows[0].operator_count;
    let primary_tensor_count = rows[0].tensor_count;
    let assessed_control_flow_contract_count = control_flow_contracts
        .iter()
        .filter(|row| row.status == "assessed")
        .count();
    let partial_control_flow_contract_count =
        control_flow_contracts.len() - assessed_control_flow_contract_count;
    Ok(TfliteSubgraphInventory {
        schema: "deepbom.tflite_subgraph_inventory.v1.3",
        status: "assessed",
        evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED",
        subgraph_count: rows.len(),
        parsed_subgraph_count: rows.len(),
        primary_subgraph_index: 0,
        primary_operator_count,
        primary_tensor_count,
        serialized_operator_count,
        serialized_tensor_count,
        nested_operator_count: serialized_operator_count - primary_operator_count,
        nested_tensor_count: serialized_tensor_count - primary_tensor_count,
        control_flow_reference_count: references.len(),
        signature_entrypoint_count: signature_entrypoints.len(),
        reachable_subgraph_count: reachable.len(),
        unreachable_subgraph_indices: (0..rows.len()).filter(|index| !reachable.contains(index)).collect(),
        control_flow_contract_count: control_flow_contracts.len(),
        assessed_control_flow_contract_count,
        partial_control_flow_contract_count,
        rows,
        references,
        signature_entrypoints,
        control_flow_contracts,
        source_commit: SOURCE_COMMIT,
        schema_source: SCHEMA_SOURCE,
        schema_source_sha256: SCHEMA_SHA256,
        control_flow_sources: vec![
            SourceFile { role: "if_prepare_contract", path: IF_SOURCE, sha256: IF_SOURCE_SHA256 },
            SourceFile { role: "while_prepare_contract", path: WHILE_SOURCE, sha256: WHILE_SOURCE_SHA256 },
            SourceFile { role: "call_once_prepare_contract", path: CALL_ONCE_SOURCE, sha256: CALL_ONCE_SOURCE_SHA256 },
            SourceFile { role: "control_flow_tensor_propagation", path: CONTROL_FLOW_COMMON_SOURCE, sha256: CONTROL_FLOW_COMMON_SOURCE_SHA256 },
        ],
        nominal_mac_sources: vec![
            SourceFile { role: "conv_2d_ohwi_nhwc_contract", path: CONV_2D_SOURCE, sha256: CONV_2D_SOURCE_SHA256 },
            SourceFile { role: "depthwise_conv_2d_1hwo_nhwc_contract", path: DEPTHWISE_CONV_2D_SOURCE, sha256: DEPTHWISE_CONV_2D_SOURCE_SHA256 },
            SourceFile { role: "fully_connected_output_input_contract", path: FULLY_CONNECTED_SOURCE, sha256: FULLY_CONNECTED_SOURCE_SHA256 },
            SourceFile { role: "conv_3d_dhwio_ndhwc_contract", path: CONV_3D_SOURCE, sha256: CONV_3D_SOURCE_SHA256 },
        ],
        method: "Parse every serialized SubGraph tensor/operator/I/O vector, validate local tensor and opcode references, derive a one-subgraph-invocation intrinsic nominal-MAC and logical-payload ledger through the shared TFLite arithmetic/payload functions, decode every schema-defined control-flow/computation subgraph index, reproduce pinned IF/WHILE/CALL_ONCE Prepare-time interface checks, and derive reachability from primary and SignatureDef entrypoints.",
        execution_count_boundary: "Every row is an intrinsic one-subgraph-invocation ledger. Rows are never summed into a model execution total: IF branches are conditional; WHILE/reduce/scatter computations can execute zero, one, or multiple times; CALL_ONCE has interpreter-lifecycle semantics; and SignatureDef entrypoints are alternatives. Executed invocation counts, liveness across calls, arena reuse across subgraphs, delegate placement, and latency require runtime evidence.",
    })
}

const JS_SAFE_INTEGER: u128 = 9_007_199_254_740_991;

struct PayloadLedger {
    total_bytes: Option<usize>,
    assessed_bytes: usize,
    present_slot_count: usize,
    assessed_slot_count: usize,
    unassessed_slot_count: usize,
}

fn build_tensor_intrinsic(tensor: &TensorInfo) -> SubgraphTensorIntrinsic {
    let payload = deterministic_tensor_payload_assessment(tensor);
    SubgraphTensorIntrinsic {
        tensor_index: tensor.index,
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        shape_signature: tensor.shape_signature.clone(),
        dtype: tensor.dtype.clone(),
        logical_payload_bytes: payload.bytes,
        payload_status: payload.status,
        payload_binding: payload.binding,
        payload_reason: payload.reason,
        constant_buffer: tensor.constant_buffer,
        sparse_storage: tensor.sparse_storage,
        buffer_data_offset: tensor.buffer_data_offset,
        buffer_data_length: tensor.buffer_data_length,
    }
}

fn build_operator_intrinsic(
    operator_index: usize,
    name: &str,
    version: i32,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    tensors: &[TensorInfo],
    tensor_intrinsics: &[SubgraphTensorIntrinsic],
) -> Result<SubgraphOperatorIntrinsic, String> {
    let (estimated_macs, estimated_ops, estimated_bytes, estimated_input_strip) =
        estimate_op(name, &inputs, &outputs, tensors);
    let (mac_assessment_status, mac_formula_class, mac_value, mac_reason) =
        assess_intrinsic_macs(name, &inputs, &outputs, tensors, estimated_macs);
    let io_payload = payload_ledger(
        inputs.iter().chain(outputs.iter()).copied(),
        tensor_intrinsics,
    )?;
    Ok(SubgraphOperatorIntrinsic {
        operator_index,
        name: name.to_string(),
        version,
        inputs,
        outputs,
        nominal_macs: mac_value.map(|value| value as f64),
        nominal_macs_decimal: mac_value.map(|value| value.to_string()),
        mac_assessment_status,
        mac_formula_class,
        mac_assessment_reason: mac_reason,
        logical_io_payload_bytes: io_payload.total_bytes,
        assessed_logical_io_payload_bytes: io_payload.assessed_bytes,
        logical_io_payload_status: if io_payload.unassessed_slot_count == 0 {
            "assessed"
        } else if io_payload.assessed_slot_count > 0 {
            "partial"
        } else {
            "not_assessed"
        },
        present_io_tensor_slot_count: io_payload.present_slot_count,
        assessed_io_tensor_slot_count: io_payload.assessed_slot_count,
        unassessed_io_tensor_slot_count: io_payload.unassessed_slot_count,
        raw_macs: estimated_macs,
        raw_ops: estimated_ops,
        raw_estimated_bytes: estimated_bytes,
        raw_estimated_input_strip: estimated_input_strip,
    })
}

fn assess_intrinsic_macs(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
    estimated_macs: f64,
) -> (&'static str, &'static str, Option<u64>, String) {
    let formula_class = mac_formula_class(name);
    if formula_class == "not_applicable" {
        return (
            "not_applicable",
            formula_class,
            None,
            "The operator is not classified as a multiply-accumulate family; its elementwise or structural work is not relabeled as MACs.".to_string(),
        );
    }
    if formula_class == "derived_nominal_dense" {
        if let Err(reason) = validate_nominal_mac_shape_contract(name, inputs, outputs, tensors) {
            return ("not_assessed", formula_class, None, reason);
        }
    }
    let Some(value) = lossless_nonnegative_integer(estimated_macs) else {
        return (
            "not_assessed",
            formula_class,
            None,
            format!(
                "The shared TFLite arithmetic estimator did not produce a finite non-negative integer within the JSON safe-integer range for {name}."
            ),
        );
    };
    if formula_class == "modeled_scenario" && value == 0 {
        return (
            "not_assessed",
            formula_class,
            None,
            format!(
                "The serialized {name} signature does not close the existing scenario formula; zero is not substituted for an unassessed compute cost."
            ),
        );
    }
    if formula_class == "modeled_scenario" {
        (
            "modeled_scenario",
            formula_class,
            Some(value),
            "The shared analyzer scenario formula produced a deterministic value for the serialized shape, but the operator contract is not promoted to the complete nominal-MAC subtotal.".to_string(),
        )
    } else {
        (
            "assessed_nominal",
            formula_class,
            Some(value),
            "The shared nominal dense-MAC formula and every required serialized tensor shape are closed for one invocation of this subgraph.".to_string(),
        )
    }
}

fn mac_formula_class(name: &str) -> &'static str {
    match name {
        "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" | "CONV_3D" => "derived_nominal_dense",
        "BATCH_MATMUL"
        | "TRANSPOSE_CONV"
        | "CONV_3D_TRANSPOSE"
        | "STABLEHLO_CONVOLUTION"
        | "STABLEHLO_DOT_GENERAL"
        | "LSTM"
        | "UNIDIRECTIONAL_SEQUENCE_LSTM"
        | "BIDIRECTIONAL_SEQUENCE_LSTM"
        | "RNN"
        | "UNIDIRECTIONAL_SEQUENCE_RNN"
        | "SVDF" => "modeled_scenario",
        _ => "not_applicable",
    }
}

fn validate_nominal_mac_shape_contract(
    name: &str,
    inputs: &[i32],
    outputs: &[i32],
    tensors: &[TensorInfo],
) -> Result<(), String> {
    let tensor = |indices: &[i32], slot: usize, role: &str| -> Result<&TensorInfo, String> {
        let index = indices
            .get(slot)
            .copied()
            .ok_or_else(|| format!("{name} is missing required {role} tensor slot {slot}"))?;
        let index = usize::try_from(index)
            .map_err(|_| format!("{name} required {role} tensor slot {slot} is omitted"))?;
        tensors
            .get(index)
            .ok_or_else(|| format!("{name} required {role} tensor slot {slot} is out of range"))
    };
    let static_rank = |value: &TensorInfo, rank: usize, role: &str| -> Result<(), String> {
        if value.shape.len() != rank {
            return Err(format!(
                "{name} {role} rank is {}; the nominal formula requires rank {rank}",
                value.shape.len()
            ));
        }
        if value.shape.iter().any(|dimension| *dimension < 0) {
            return Err(format!(
                "{name} {role} contains a dynamic or unknown serialized dimension"
            ));
        }
        Ok(())
    };
    match name {
        "CONV_2D" => {
            let input = tensor(inputs, 0, "input")?;
            let filter = tensor(inputs, 1, "filter")?;
            let output = tensor(outputs, 0, "output")?;
            static_rank(input, 4, "input")?;
            static_rank(filter, 4, "filter")?;
            static_rank(output, 4, "output")?;
            if input.shape[0] != output.shape[0]
                || filter.shape[0] != output.shape[3]
                || filter.shape[3] != input.shape[3]
            {
                return Err("CONV_2D input/filter/output channel or batch dimensions do not satisfy the nominal OHWI/NHWC contract".to_string());
            }
        }
        "DEPTHWISE_CONV_2D" => {
            let input = tensor(inputs, 0, "input")?;
            let filter = tensor(inputs, 1, "filter")?;
            let output = tensor(outputs, 0, "output")?;
            static_rank(input, 4, "input")?;
            static_rank(filter, 4, "filter")?;
            static_rank(output, 4, "output")?;
            if input.shape[0] != output.shape[0]
                || filter.shape[0] != 1
                || filter.shape[3] != output.shape[3]
                || input.shape[3] > 0 && output.shape[3] % input.shape[3] != 0
            {
                return Err("DEPTHWISE_CONV_2D input/filter/output dimensions do not satisfy the nominal 1HWO/NHWC contract".to_string());
            }
        }
        "FULLY_CONNECTED" => {
            let filter = tensor(inputs, 1, "weights")?;
            let output = tensor(outputs, 0, "output")?;
            static_rank(filter, 2, "weights")?;
            if output.shape.is_empty() || output.shape.iter().any(|dimension| *dimension < 0) {
                return Err(
                    "FULLY_CONNECTED output shape is dynamic, unknown, or rank zero".to_string(),
                );
            }
            if output.shape.last().copied() != Some(filter.shape[0]) {
                return Err("FULLY_CONNECTED output width does not equal the serialized weight output-unit dimension".to_string());
            }
        }
        "CONV_3D" => {
            let input = tensor(inputs, 0, "input")?;
            let filter = tensor(inputs, 1, "filter")?;
            let output = tensor(outputs, 0, "output")?;
            static_rank(input, 5, "input")?;
            static_rank(filter, 5, "filter")?;
            static_rank(output, 5, "output")?;
            if input.shape[0] != output.shape[0]
                || filter.shape[3] != input.shape[4]
                || filter.shape[4] != output.shape[4]
            {
                return Err("CONV_3D input/filter/output channel or batch dimensions do not satisfy the pinned nominal DHWIO/NDHWC contract".to_string());
            }
        }
        _ => {
            return Err(format!(
                "No complete nominal-MAC shape contract is registered for {name}"
            ))
        }
    }
    Ok(())
}

fn lossless_nonnegative_integer(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > JS_SAFE_INTEGER as f64 {
        return None;
    }
    Some(value as u64)
}

fn payload_ledger(
    indices: impl Iterator<Item = i32>,
    tensors: &[SubgraphTensorIntrinsic],
) -> Result<PayloadLedger, String> {
    let mut assessed_bytes = 0usize;
    let mut present_slot_count = 0usize;
    let mut assessed_slot_count = 0usize;
    let mut unassessed_slot_count = 0usize;
    for index in indices {
        if index < 0 {
            continue;
        }
        present_slot_count += 1;
        let tensor = tensors
            .get(index as usize)
            .ok_or_else(|| format!("Intrinsic payload ledger references missing tensor {index}"))?;
        if let Some(bytes) = tensor.logical_payload_bytes {
            assessed_bytes = assessed_bytes
                .checked_add(bytes)
                .ok_or_else(|| "Intrinsic logical payload subtotal overflows".to_string())?;
            assessed_slot_count += 1;
        } else {
            unassessed_slot_count += 1;
        }
    }
    Ok(PayloadLedger {
        total_bytes: (unassessed_slot_count == 0).then_some(assessed_bytes),
        assessed_bytes,
        present_slot_count,
        assessed_slot_count,
        unassessed_slot_count,
    })
}

fn build_subgraph_intrinsic_cost(
    tensor_intrinsics: &[SubgraphTensorIntrinsic],
    inputs: &[i32],
    outputs: &[i32],
    operators: &[SubgraphOperatorIntrinsic],
) -> Result<SubgraphIntrinsicCost, String> {
    let tensor_payload = payload_ledger(
        (0..tensor_intrinsics.len()).map(|index| index as i32),
        tensor_intrinsics,
    )?;
    let input_payload = payload_ledger(inputs.iter().copied(), tensor_intrinsics)?;
    let output_payload = payload_ledger(outputs.iter().copied(), tensor_intrinsics)?;
    let mac_compute_operator_count = operators
        .iter()
        .filter(|row| row.mac_formula_class != "not_applicable")
        .count();
    let assessed_nominal_mac_operator_count = operators
        .iter()
        .filter(|row| row.mac_assessment_status == "assessed_nominal")
        .count();
    let modeled_scenario_mac_operator_count = operators
        .iter()
        .filter(|row| row.mac_assessment_status == "modeled_scenario")
        .count();
    let unassessed_mac_operator_count = operators
        .iter()
        .filter(|row| row.mac_assessment_status == "not_assessed")
        .count();
    let sum_macs = |status: &str| -> Result<u128, String> {
        operators
            .iter()
            .filter(|row| row.mac_assessment_status == status)
            .try_fold(0u128, |sum, row| {
                let value = row
                    .nominal_macs
                    .and_then(lossless_nonnegative_integer)
                    .ok_or_else(|| {
                        format!(
                            "S{} {} lost its assessed MAC value",
                            row.operator_index, row.name
                        )
                    })?;
                sum.checked_add(value as u128)
                    .ok_or_else(|| "Intrinsic MAC subtotal overflows u128".to_string())
            })
    };
    let assessed_nominal_macs = sum_macs("assessed_nominal")?;
    let modeled_scenario_macs = sum_macs("modeled_scenario")?;
    let complete_nominal_macs = (mac_compute_operator_count > 0
        && modeled_scenario_mac_operator_count == 0
        && unassessed_mac_operator_count == 0)
        .then_some(assessed_nominal_macs);
    let to_safe_number = |value: u128| (value <= JS_SAFE_INTEGER).then_some(value as f64);
    let mut assessed_operator_io_payload_bytes = 0usize;
    let mut assessed_operator_io_tensor_slot_count = 0usize;
    let mut unassessed_operator_io_tensor_slot_count = 0usize;
    for operator in operators {
        assessed_operator_io_payload_bytes = assessed_operator_io_payload_bytes
            .checked_add(operator.assessed_logical_io_payload_bytes)
            .ok_or_else(|| "Intrinsic operator I/O payload subtotal overflows".to_string())?;
        assessed_operator_io_tensor_slot_count += operator.assessed_io_tensor_slot_count;
        unassessed_operator_io_tensor_slot_count += operator.unassessed_io_tensor_slot_count;
    }
    let logical_operator_io_payload_bytes = (unassessed_operator_io_tensor_slot_count == 0)
        .then_some(assessed_operator_io_payload_bytes);
    let mut logical_constant_reference_bytes = 0usize;
    let mut physical_constant_buffers = BTreeSet::<(usize, usize)>::new();
    for tensor in tensor_intrinsics
        .iter()
        .filter(|tensor| tensor.constant_buffer)
    {
        logical_constant_reference_bytes = logical_constant_reference_bytes
            .checked_add(tensor.buffer_data_length)
            .ok_or_else(|| "Intrinsic logical constant-reference subtotal overflows".to_string())?;
        if tensor.buffer_data_length > 0 {
            physical_constant_buffers
                .insert((tensor.buffer_data_offset, tensor.buffer_data_length));
        }
    }
    let physical_unique_constant_bytes =
        physical_constant_buffers
            .iter()
            .try_fold(0usize, |sum, (_, length)| {
                sum.checked_add(*length).ok_or_else(|| {
                    "Intrinsic physical constant-byte subtotal overflows".to_string()
                })
            })?;
    let status = if mac_compute_operator_count == 0
        && tensor_payload.unassessed_slot_count == 0
        && unassessed_operator_io_tensor_slot_count == 0
    {
        "assessed_no_mac_compute"
    } else if modeled_scenario_mac_operator_count == 0
        && unassessed_mac_operator_count == 0
        && tensor_payload.unassessed_slot_count == 0
        && unassessed_operator_io_tensor_slot_count == 0
    {
        "assessed"
    } else {
        "partial"
    };
    Ok(SubgraphIntrinsicCost {
        schema: "deepbom.tflite_subgraph_intrinsic_cost.v1",
        status,
        evidence_class: "OBSERVED/DERIVED",
        invocation_basis: "one_invocation_of_this_serialized_subgraph",
        mac_compute_operator_count,
        assessed_nominal_mac_operator_count,
        modeled_scenario_mac_operator_count,
        unassessed_mac_operator_count,
        complete_nominal_macs: complete_nominal_macs.and_then(to_safe_number),
        complete_nominal_macs_decimal: complete_nominal_macs.map(|value| value.to_string()),
        assessed_nominal_macs: to_safe_number(assessed_nominal_macs),
        assessed_nominal_macs_decimal: assessed_nominal_macs.to_string(),
        modeled_scenario_macs: to_safe_number(modeled_scenario_macs),
        modeled_scenario_macs_decimal: modeled_scenario_macs.to_string(),
        logical_tensor_payload_bytes: tensor_payload.total_bytes,
        assessed_logical_tensor_payload_bytes: tensor_payload.assessed_bytes,
        assessed_tensor_payload_count: tensor_payload.assessed_slot_count,
        unassessed_tensor_payload_count: tensor_payload.unassessed_slot_count,
        logical_operator_io_payload_bytes,
        assessed_logical_operator_io_payload_bytes: assessed_operator_io_payload_bytes,
        assessed_operator_io_tensor_slot_count,
        unassessed_operator_io_tensor_slot_count,
        graph_input_payload_bytes: input_payload.total_bytes,
        graph_output_payload_bytes: output_payload.total_bytes,
        logical_constant_reference_bytes,
        physical_unique_constant_bytes,
        physical_unique_constant_buffer_count: physical_constant_buffers.len(),
        method: "Use the same TFLite estimate_op implementation as primary-graph analysis exactly once per serialized operator. Promote only shape-validated Conv2D, DepthwiseConv2D, FullyConnected, and Conv3D dense nominal formulas; retain other existing formulas in a separate modeled-scenario subtotal. Compute payloads through deterministic_tensor_payload_assessment with checked integer sums and no unknown-to-zero substitution.",
        interpretation_boundary: "Intrinsic values describe one invocation of this subgraph at its serialized shape. Logical tensor and operator-I/O payloads are cardinality ledgers, not arena size, cache traffic, physical copies, or runtime memory. Nominal MACs are graph-complexity definitions, not retired hardware instructions. Cross-subgraph execution totals require observed invocation counts.",
    })
}

fn invocation_semantics(
    subgraph_index: usize,
    references: &[SubgraphReference],
    signature_entrypoints: &[SignatureEntrypoint],
) -> String {
    let signature = signature_entrypoints
        .iter()
        .any(|entry| entry.subgraph_index == subgraph_index);
    if subgraph_index == 0 {
        return if signature {
            "primary_and_signature_entrypoint; one row is not an invocation count".to_string()
        } else {
            "primary_entrypoint; one row is not an invocation count".to_string()
        };
    }
    let roles = references
        .iter()
        .filter(|reference| reference.target_subgraph_index == subgraph_index)
        .map(|reference| reference.role.as_str())
        .collect::<BTreeSet<_>>();
    let reference_semantics = if roles.contains("initialization") {
        "CALL_ONCE initialization; interpreter-lifecycle dependent"
    } else if roles.contains("condition") || roles.contains("body") {
        "condition/body computation; loop or reducer invocation count is runtime dependent"
    } else if roles.contains("then") || roles.contains("else") || roles.contains("branch") {
        "conditional branch; selection and invocation count are runtime dependent"
    } else if roles.is_empty() {
        "no incoming serialized computation reference"
    } else {
        "referenced computation; invocation count is runtime dependent"
    };
    if signature {
        format!("signature entrypoint and {reference_semantics}")
    } else {
        reference_semantics.to_string()
    }
}

type ControlFlowEdges<'a> = (&'a [i32], &'a [i32], &'a [SubgraphReference]);

fn validate_control_flow_contract(
    fb: &Fb<'_>,
    subgraphs: &[usize],
    scoped_tensors: &[Vec<TensorInfo>],
    source_subgraph_index: usize,
    source_op_index: usize,
    source_op_name: &str,
    edges: ControlFlowEdges<'_>,
) -> Result<Option<ControlFlowContract>, String> {
    let (op_inputs, op_outputs, references) = edges;
    let contract = match source_op_name {
        "IF" => {
            require_present_indices(op_inputs, "IF inputs")?;
            if op_inputs.is_empty() {
                return Err(format!(
                    "IF in subgraph {source_subgraph_index} has no condition input"
                ));
            }
            let condition = scoped_tensors[source_subgraph_index]
                .get(op_inputs[0] as usize)
                .ok_or_else(|| {
                    "IF condition tensor is missing after local-index validation".to_string()
                })?;
            let condition_status = if_condition_contract(condition)?;
            let then_target = reference_target(references, "then", "IF")?;
            let else_target = reference_target(references, "else", "IF")?;
            for (role, target) in [("then", then_target), ("else", else_target)] {
                let (inputs, outputs) = subgraph_interface(fb, subgraphs, scoped_tensors, target)?;
                if inputs.len() != op_inputs.len() - 1 || outputs.len() != op_outputs.len() {
                    return Err(format!(
                        "IF {role} subgraph {target} interface is {}/{} input/output tensor(s); expected {}/{} from source op S{source_subgraph_index}/O{source_op_index}",
                        inputs.len(), outputs.len(), op_inputs.len() - 1, op_outputs.len()
                    ));
                }
            }
            ControlFlowContract {
                source_subgraph_index,
                source_op_index,
                source_op_name: source_op_name.to_string(),
                status: condition_status_to_row_status(condition_status),
                source_input_count: op_inputs.len(),
                source_output_count: op_outputs.len(),
                target_subgraph_indices: vec![then_target, else_target],
                condition_contract_status: Some(condition_status),
                method: "Pinned IF Prepare contract: BOOL singleton condition; each branch receives source inputs after the condition and exposes every source output.".to_string(),
            }
        }
        "WHILE" => {
            require_present_indices(op_inputs, "WHILE inputs")?;
            require_present_indices(op_outputs, "WHILE outputs")?;
            if op_inputs.len() != op_outputs.len() {
                return Err(format!(
                    "WHILE in subgraph {source_subgraph_index} has {}/{} input/output tensor(s); pinned Prepare requires equal counts",
                    op_inputs.len(), op_outputs.len()
                ));
            }
            let condition_target = reference_target(references, "condition", "WHILE")?;
            let body_target = reference_target(references, "body", "WHILE")?;
            if condition_target == body_target {
                return Err(format!("WHILE in subgraph {source_subgraph_index} uses the same condition and body subgraph {condition_target}"));
            }
            let (condition_inputs, condition_outputs) =
                subgraph_interface(fb, subgraphs, scoped_tensors, condition_target)?;
            let (body_inputs, body_outputs) =
                subgraph_interface(fb, subgraphs, scoped_tensors, body_target)?;
            require_present_indices(&condition_outputs, "WHILE condition outputs")?;
            require_present_indices(&body_inputs, "WHILE body inputs")?;
            require_present_indices(&body_outputs, "WHILE body outputs")?;
            if condition_inputs.len() != op_inputs.len() || condition_outputs.len() != 1 {
                return Err(format!(
                    "WHILE condition subgraph {condition_target} interface is {}/{} input/output tensor(s); expected {}/1",
                    condition_inputs.len(), condition_outputs.len(), op_inputs.len()
                ));
            }
            if body_inputs.len() != op_inputs.len() || body_outputs.len() != op_inputs.len() {
                return Err(format!(
                    "WHILE body subgraph {body_target} interface is {}/{} input/output tensor(s); expected {}/{}",
                    body_inputs.len(), body_outputs.len(), op_inputs.len(), op_inputs.len()
                ));
            }
            let condition = &scoped_tensors[condition_target][condition_outputs[0] as usize];
            let condition_status = while_condition_contract(condition)?;
            for index in 0..op_inputs.len() {
                let source_input =
                    &scoped_tensors[source_subgraph_index][op_inputs[index] as usize];
                let body_output = &scoped_tensors[body_target][body_outputs[index] as usize];
                if source_input.dtype != body_output.dtype {
                    return Err(format!(
                        "WHILE loop-carried value {index} dtype contract differs: source input {}, body output {}",
                        source_input.dtype, body_output.dtype
                    ));
                }
            }
            ControlFlowContract {
                source_subgraph_index,
                source_op_index,
                source_op_name: source_op_name.to_string(),
                status: condition_status_to_row_status(condition_status),
                source_input_count: op_inputs.len(),
                source_output_count: op_outputs.len(),
                target_subgraph_indices: vec![condition_target, body_target],
                condition_contract_status: Some(condition_status),
                method: "Pinned WHILE Prepare contract: equal source input/output counts; condition N-to-1 BOOL scalar-or-[1] when statically shaped; body N-to-N with source-input type propagation and body input/output dtype agreement.".to_string(),
            }
        }
        "CALL_ONCE" => {
            let target = reference_target(references, "initialization", "CALL_ONCE")?;
            let (inputs, outputs) = subgraph_interface(fb, subgraphs, scoped_tensors, target)?;
            if target == source_subgraph_index
                || !op_inputs.is_empty()
                || !op_outputs.is_empty()
                || !inputs.is_empty()
                || !outputs.is_empty()
            {
                return Err(format!(
                    "CALL_ONCE S{source_subgraph_index}/O{source_op_index} must target a distinct zero-input/zero-output initialization subgraph"
                ));
            }
            ControlFlowContract {
                source_subgraph_index,
                source_op_index,
                source_op_name: source_op_name.to_string(),
                status: "assessed",
                source_input_count: 0,
                source_output_count: 0,
                target_subgraph_indices: vec![target],
                condition_contract_status: None,
                method: "Pinned CALL_ONCE Prepare contract: source op and distinct initialization subgraph both expose zero inputs and zero outputs.".to_string(),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(contract))
}

fn subgraph_interface(
    fb: &Fb<'_>,
    subgraphs: &[usize],
    scoped_tensors: &[Vec<TensorInfo>],
    index: usize,
) -> Result<(Vec<i32>, Vec<i32>), String> {
    let table = *subgraphs
        .get(index)
        .ok_or_else(|| format!("control-flow contract references missing subgraph {index}"))?;
    let tensor_count = scoped_tensors[index].len();
    Ok((
        checked_tensor_indices(fb, table, 1, "SubGraph.inputs", tensor_count, true)?,
        checked_tensor_indices(fb, table, 2, "SubGraph.outputs", tensor_count, true)?,
    ))
}

fn reference_target(
    references: &[SubgraphReference],
    role: &str,
    op_name: &str,
) -> Result<usize, String> {
    let matches = references
        .iter()
        .filter(|row| row.role == role)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "{op_name} must serialize exactly one {role} subgraph reference"
        ));
    }
    Ok(matches[0].target_subgraph_index)
}

fn require_present_indices(indices: &[i32], label: &str) -> Result<(), String> {
    if indices.iter().any(|&index| index < 0) {
        return Err(format!("{label} contain an optional tensor index unsupported by the pinned control-flow Prepare contract"));
    }
    Ok(())
}

fn require_bool(tensor: &TensorInfo, label: &str) -> Result<(), String> {
    if tensor.dtype != "BOOL" {
        return Err(format!(
            "{label} dtype is {}; pinned runtime requires BOOL",
            tensor.dtype
        ));
    }
    Ok(())
}

fn if_condition_contract(tensor: &TensorInfo) -> Result<&'static str, String> {
    let label = "IF condition";
    require_bool(tensor, label)?;
    if tensor.shape.iter().any(|&dimension| dimension < 0) {
        return Ok("partial_dynamic_cardinality");
    }
    let cardinality = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|value| product.checked_mul(value))
                .ok_or_else(|| format!("{label} static cardinality overflows"))
        })?;
    if cardinality != 1 {
        return Err(format!(
            "{label} has static cardinality {cardinality}; pinned runtime requires one BOOL value"
        ));
    }
    Ok("assessed_static_single_bool")
}

fn while_condition_contract(tensor: &TensorInfo) -> Result<&'static str, String> {
    let label = "WHILE condition output";
    require_bool(tensor, label)?;
    if tensor.shape.iter().any(|&dimension| dimension < 0)
        || tensor
            .shape_signature
            .iter()
            .any(|&dimension| dimension < 0)
    {
        return Ok("partial_dynamic_cardinality");
    }
    if tensor.shape.is_empty() || tensor.shape.as_slice() == [1] {
        return Ok("assessed_static_single_bool");
    }
    Err(format!(
        "{label} shape {:?}; pinned runtime requires a scalar or one-dimensional [1] BOOL when statically shaped",
        tensor.shape
    ))
}

fn condition_status_to_row_status(status: &str) -> &'static str {
    if status == "assessed_static_single_bool" {
        "assessed"
    } else {
        "partial"
    }
}

fn checked_tensor_indices(
    fb: &Fb<'_>,
    table: usize,
    field: usize,
    label: &str,
    tensor_count: usize,
    allow_optional: bool,
) -> Result<Vec<i32>, String> {
    let indices = fb.checked_vector_i32(table, field, label)?;
    for &index in &indices {
        if allow_optional && index == -1 {
            continue;
        }
        if index < 0 || index as usize >= tensor_count {
            return Err(format!(
                "{label} contains out-of-range tensor index {index} for {tensor_count} tensor(s)"
            ));
        }
    }
    Ok(indices)
}

fn collect_subgraph_references(
    fb: &Fb<'_>,
    operator: usize,
    source_subgraph_index: usize,
    source_op_index: usize,
    source_op_name: &str,
    subgraph_names: &[String],
    output: &mut Vec<SubgraphReference>,
) -> Result<(), String> {
    let options = fb.checked_table_field(operator, 4, "Operator.builtin_options")?;
    let mut scalar_roles = Vec::<(&str, usize)>::new();
    let mut vector_roles = Vec::<(&str, usize)>::new();
    match source_op_name {
        "IF" => scalar_roles.extend([("then", 0), ("else", 1)]),
        "WHILE" | "STABLEHLO_WHILE" => scalar_roles.extend([("condition", 0), ("body", 1)]),
        "CALL_ONCE" => scalar_roles.push(("initialization", 0)),
        "STABLEHLO_REDUCE_WINDOW" => scalar_roles.push(("body", 5)),
        "STABLEHLO_SORT" => scalar_roles.push(("comparator", 2)),
        "STABLEHLO_REDUCE" => scalar_roles.push(("body", 1)),
        "STABLEHLO_SCATTER" => scalar_roles.push(("update_computation", 6)),
        "STABLEHLO_COMPOSITE" => scalar_roles.push(("decomposition", 1)),
        "STABLEHLO_CASE" => vector_roles.push(("branch", 0)),
        "STABLEHLO_CUSTOM_CALL" => vector_roles.push(("called_computation", 4)),
        _ => return Ok(()),
    }
    let options = options.ok_or_else(|| {
        format!("{source_op_name} in subgraph {source_subgraph_index} has no builtin options table")
    })?;
    for (role, field) in scalar_roles {
        let target = fb.checked_i32_field(options, field, 0, source_op_name)?;
        push_reference(
            source_subgraph_index,
            source_op_index,
            source_op_name,
            role,
            target,
            subgraph_names,
            output,
        )?;
    }
    for (role, field) in vector_roles {
        for target in fb.checked_vector_i32(options, field, source_op_name)? {
            push_reference(
                source_subgraph_index,
                source_op_index,
                source_op_name,
                role,
                target,
                subgraph_names,
                output,
            )?;
        }
    }
    Ok(())
}

fn push_reference(
    source_subgraph_index: usize,
    source_op_index: usize,
    source_op_name: &str,
    role: &str,
    target: i32,
    subgraph_names: &[String],
    output: &mut Vec<SubgraphReference>,
) -> Result<(), String> {
    let target_subgraph_index = usize::try_from(target).map_err(|_| {
        format!("{source_op_name} subgraph reference {role} has negative index {target}")
    })?;
    let target_subgraph_name = subgraph_names.get(target_subgraph_index).ok_or_else(|| {
        format!("{source_op_name} subgraph reference {role} targets missing subgraph {target_subgraph_index}")
    })?;
    output.push(SubgraphReference {
        source_subgraph_index,
        source_op_index,
        source_op_name: source_op_name.to_string(),
        role: role.to_string(),
        target_subgraph_index,
        target_subgraph_name: target_subgraph_name.clone(),
    });
    Ok(())
}
