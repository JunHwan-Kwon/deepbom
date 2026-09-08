use std::collections::BTreeSet;

use serde::Serialize;

use crate::verified_flatbuffer::Fb;
use crate::TensorInfo;

const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const SPACE_TO_BATCH_SOURCE: &str = "tensorflow/lite/kernels/space_to_batch_nd.cc";
const SPACE_TO_BATCH_SOURCE_SHA256: &str =
    "3478660fb9e787917d0aa3a2dec1efa9411786b02d6e4d28987e4cc30a8261f0";
const ELEMENTWISE_SOURCE: &str = "tensorflow/lite/kernels/elementwise.cc";
const ELEMENTWISE_SOURCE_SHA256: &str =
    "d342b6889ba4d95f2b98c8dcdaf41191a641de216b35e81e7f9f8099d776921f";
const CONV_SOURCE: &str = "tensorflow/lite/kernels/conv.cc";
const CONV_SOURCE_SHA256: &str = "66a2fef9a8e7fe81b7bdd9d18bd099cc589546ac29cca7665711de890fba9281";

#[derive(Clone, Serialize)]
struct ReconciledTensorShape {
    subgraph_index: usize,
    tensor_index: usize,
    tensor_name: String,
    serialized_shape: Vec<i32>,
    shape_signature: Vec<i32>,
    effective_shape: Vec<i32>,
    status: String,
    rule_ids: Vec<String>,
}

#[derive(Clone, Serialize)]
struct ShapeSource {
    role: &'static str,
    path: &'static str,
    commit: &'static str,
    sha256: &'static str,
}

#[derive(Clone, Serialize)]
pub(super) struct TfliteShapeReconciliation {
    schema: &'static str,
    status: String,
    evidence_class: &'static str,
    tensor_count: usize,
    reconciled_tensor_count: usize,
    serialized_conflict_count: usize,
    unresolved_tensor_count: usize,
    rows: Vec<ReconciledTensorShape>,
    sources: Vec<ShapeSource>,
    method: &'static str,
    interpretation_boundary: &'static str,
}

#[derive(Clone)]
struct ShapeState {
    serialized: Vec<i32>,
    exact: Vec<bool>,
    rules: BTreeSet<String>,
}

fn initial_state(tensor: &mut TensorInfo) -> ShapeState {
    let serialized = tensor.shape.clone();
    let exact = if tensor.shape_signature.len() == tensor.shape.len() {
        tensor
            .shape_signature
            .iter()
            .zip(&tensor.shape)
            .map(|(signature, shape)| *signature >= 0 && signature == shape)
            .collect::<Vec<_>>()
    } else {
        tensor
            .shape
            .iter()
            .map(|dimension| *dimension >= 0)
            .collect()
    };
    for (dimension, is_exact) in tensor.shape.iter_mut().zip(&exact) {
        if !is_exact {
            *dimension = -1;
        }
    }
    ShapeState {
        serialized,
        exact,
        rules: BTreeSet::new(),
    }
}

fn tensor_index(value: i32, tensors: &[TensorInfo]) -> Option<usize> {
    usize::try_from(value)
        .ok()
        .filter(|index| *index < tensors.len())
}

fn constant_i32_values(fb: &Fb<'_>, tensor: &TensorInfo) -> Option<Vec<i32>> {
    if tensor.dtype != "INT32"
        || !tensor.constant_buffer
        || tensor.buffer_data_length == 0
        || tensor.buffer_data_length % 4 != 0
    {
        return None;
    }
    let bytes = fb.data.get(
        tensor.buffer_data_offset
            ..tensor
                .buffer_data_offset
                .checked_add(tensor.buffer_data_length)?,
    )?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

fn set_shape(tensor: &mut TensorInfo, state: &mut ShapeState, shape: Vec<i32>, rule: &str) -> bool {
    if shape.is_empty() || shape.iter().any(|dimension| *dimension < 0) {
        return false;
    }
    let changed = tensor.shape != shape || state.exact.iter().any(|exact| !exact);
    tensor.shape = shape;
    state.exact = vec![true; tensor.shape.len()];
    state.rules.insert(rule.to_string());
    changed
}

fn set_dimension(
    tensor: &mut TensorInfo,
    state: &mut ShapeState,
    axis: usize,
    value: i32,
    rule: &str,
) -> bool {
    if value < 0 || axis >= tensor.shape.len() || axis >= state.exact.len() {
        return false;
    }
    let changed = tensor.shape[axis] != value || !state.exact[axis];
    tensor.shape[axis] = value;
    state.exact[axis] = true;
    state.rules.insert(rule.to_string());
    changed
}

fn space_to_batch_shape(
    fb: &Fb<'_>,
    inputs: &[i32],
    tensors: &[TensorInfo],
    states: &[ShapeState],
) -> Option<Vec<i32>> {
    let input_index = tensor_index(*inputs.first()?, tensors)?;
    let block_index = tensor_index(*inputs.get(1)?, tensors)?;
    let paddings_index = tensor_index(*inputs.get(2)?, tensors)?;
    let input = tensors.get(input_index)?;
    if input.shape.len() != 4 || states.get(input_index)?.exact.iter().any(|exact| !exact) {
        return None;
    }
    let block = constant_i32_values(fb, tensors.get(block_index)?)?;
    let paddings = constant_i32_values(fb, tensors.get(paddings_index)?)?;
    if block.len() != 2
        || paddings.len() != 4
        || block.iter().any(|value| *value <= 0)
        || paddings.iter().any(|value| *value < 0)
    {
        return None;
    }
    let padded_height = input.shape[1]
        .checked_add(paddings[0])?
        .checked_add(paddings[1])?;
    let padded_width = input.shape[2]
        .checked_add(paddings[2])?
        .checked_add(paddings[3])?;
    if padded_height < 0
        || padded_width < 0
        || padded_height % block[0] != 0
        || padded_width % block[1] != 0
    {
        return None;
    }
    Some(vec![
        input.shape[0]
            .checked_mul(block[0])?
            .checked_mul(block[1])?,
        padded_height / block[0],
        padded_width / block[1],
        input.shape[3],
    ])
}

fn reconcile_scope(
    fb: &Fb<'_>,
    subgraph: usize,
    operator_names: &[String],
    tensors: &mut [TensorInfo],
    states: &mut [ShapeState],
) -> Result<(), String> {
    let operators = fb.vector_tables(subgraph, 3);
    for _ in 0..=operators.len() {
        let mut changed = false;
        for operator in &operators {
            let opcode_index =
                fb.checked_u32_field(*operator, 0, 0, "Operator.opcode_index")? as usize;
            let name = operator_names.get(opcode_index).ok_or_else(|| {
                format!("Shape reconciliation references missing opcode {opcode_index}")
            })?;
            let inputs = fb.vector_i32(*operator, 1);
            let outputs = fb.vector_i32(*operator, 2);
            let Some(output_index) = outputs
                .first()
                .and_then(|value| tensor_index(*value, tensors))
            else {
                continue;
            };
            match name.as_str() {
                "SPACE_TO_BATCH_ND" => {
                    if let Some(shape) = space_to_batch_shape(fb, &inputs, tensors, states) {
                        changed |= set_shape(
                            &mut tensors[output_index],
                            &mut states[output_index],
                            shape,
                            "tflite.prepare.space_to_batch_nd",
                        );
                    }
                }
                "SIN" => {
                    let Some(input_index) = inputs
                        .first()
                        .and_then(|value| tensor_index(*value, tensors))
                    else {
                        continue;
                    };
                    if states[input_index].exact.iter().all(|exact| *exact) {
                        let input_shape = tensors[input_index].shape.clone();
                        changed |= set_shape(
                            &mut tensors[output_index],
                            &mut states[output_index],
                            input_shape,
                            "tflite.prepare.elementwise.same_shape",
                        );
                    }
                }
                "CONV_2D" => {
                    let Some(input_index) = inputs
                        .first()
                        .and_then(|value| tensor_index(*value, tensors))
                    else {
                        continue;
                    };
                    let Some(filter_index) = inputs
                        .get(1)
                        .and_then(|value| tensor_index(*value, tensors))
                    else {
                        continue;
                    };
                    if tensors[input_index].shape.len() == 4
                        && tensors[filter_index].shape.len() == 4
                        && tensors[output_index].shape.len() == 4
                    {
                        if states[input_index].exact.first() == Some(&true) {
                            let input_batch = tensors[input_index].shape[0];
                            changed |= set_dimension(
                                &mut tensors[output_index],
                                &mut states[output_index],
                                0,
                                input_batch,
                                "tflite.prepare.conv2d.batch",
                            );
                        }
                        if states[filter_index].exact.first() == Some(&true) {
                            let output_channels = tensors[filter_index].shape[0];
                            changed |= set_dimension(
                                &mut tensors[output_index],
                                &mut states[output_index],
                                3,
                                output_channels,
                                "tflite.prepare.conv2d.output_channels",
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

pub(super) fn reconcile_tflite_shapes(
    fb: &Fb<'_>,
    model: usize,
    operator_names: &[String],
    serialized_scopes: &[Vec<TensorInfo>],
) -> Result<(Vec<Vec<TensorInfo>>, TfliteShapeReconciliation), String> {
    let subgraphs = fb.vector_tables(model, 2);
    if subgraphs.len() != serialized_scopes.len() {
        return Err("TFLite shape-reconciliation scope count mismatch".to_string());
    }
    let mut scopes = serialized_scopes.to_vec();
    let mut scope_states = scopes
        .iter_mut()
        .map(|tensors| tensors.iter_mut().map(initial_state).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for (scope_index, subgraph) in subgraphs.iter().enumerate() {
        reconcile_scope(
            fb,
            *subgraph,
            operator_names,
            &mut scopes[scope_index],
            &mut scope_states[scope_index],
        )?;
    }

    let mut rows = Vec::new();
    let mut serialized_conflict_count = 0usize;
    let mut unresolved_tensor_count = 0usize;
    for (subgraph_index, (tensors, states)) in scopes.iter().zip(&scope_states).enumerate() {
        for (tensor, state) in tensors.iter().zip(states) {
            let unresolved = state.exact.iter().any(|exact| !exact);
            let conflict = !state.rules.is_empty() && state.serialized != tensor.shape;
            if unresolved {
                unresolved_tensor_count += 1;
            }
            if conflict {
                serialized_conflict_count += 1;
            }
            if unresolved || conflict {
                rows.push(ReconciledTensorShape {
                    subgraph_index,
                    tensor_index: tensor.index,
                    tensor_name: tensor.name.clone(),
                    serialized_shape: state.serialized.clone(),
                    shape_signature: tensor.shape_signature.clone(),
                    effective_shape: tensor.shape.clone(),
                    status: if unresolved {
                        "unresolved_dynamic_shape".to_string()
                    } else if conflict {
                        "source_derived_shape_overrides_serialized_placeholder".to_string()
                    } else {
                        "source_derived_shape".to_string()
                    },
                    rule_ids: state.rules.iter().cloned().collect(),
                });
            }
        }
    }
    let status = if unresolved_tensor_count > 0 {
        "partial_unresolved_dynamic_shapes"
    } else if serialized_conflict_count > 0 {
        "assessed_with_source_derived_reconciliation"
    } else {
        "not_applicable_no_shape_reconciliation"
    };
    Ok((
        scopes,
        TfliteShapeReconciliation {
            schema: "deepbom.tflite_shape_reconciliation.v1",
            status: status.to_string(),
            evidence_class: "SOURCE_BACKED_DERIVED",
            tensor_count: serialized_scopes.iter().map(Vec::len).sum(),
            reconciled_tensor_count: rows.len(),
            serialized_conflict_count,
            unresolved_tensor_count,
            rows,
            sources: vec![
                ShapeSource {
                    role: "space_to_batch_nd_prepare",
                    path: SPACE_TO_BATCH_SOURCE,
                    commit: SOURCE_COMMIT,
                    sha256: SPACE_TO_BATCH_SOURCE_SHA256,
                },
                ShapeSource {
                    role: "elementwise_prepare",
                    path: ELEMENTWISE_SOURCE,
                    commit: SOURCE_COMMIT,
                    sha256: ELEMENTWISE_SOURCE_SHA256,
                },
                ShapeSource {
                    role: "conv2d_prepare",
                    path: CONV_SOURCE,
                    commit: SOURCE_COMMIT,
                    sha256: CONV_SOURCE_SHA256,
                },
            ],
            method: "Preserve every serialized tensor shape, mark negative shape-signature dimensions unbound, and replace only dimensions proven by pinned TFLite Prepare contracts and constant operands.",
            interpretation_boundary: "Effective shapes are source-backed static-analysis facts, not executed tensor dimensions. Unknown runtime dimensions remain -1 and prevent a complete numeric MAC total.",
        },
    ))
}
