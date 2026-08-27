use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArenaPlanClass {
    NonPersistent,
    Persistent,
}

impl ArenaPlanClass {
    fn label(self) -> &'static str {
        match self {
            Self::NonPersistent => "kTfLiteArenaRw",
            Self::Persistent => "kTfLiteArenaRwPersistent",
        }
    }
}

#[derive(Clone)]
struct ArenaPlanCandidate {
    tensor_index: usize,
    size_bytes: usize,
    first_node: usize,
    last_node: Option<usize>,
    arena: ArenaPlanClass,
}

#[derive(Clone)]
struct PlacedArenaAllocation {
    candidate: ArenaPlanCandidate,
    offset_bytes: usize,
}

#[derive(Default)]
struct ArenaPlacementState {
    high_water_mark: usize,
    allocations: Vec<PlacedArenaAllocation>,
}

#[derive(Clone, Copy)]
struct InPlaceRegistration {
    input_slots: &'static [usize],
    data_unmodified: bool,
    source_file: &'static str,
}

fn pinned_in_place_registration(op_name: &str) -> Option<InPlaceRegistration> {
    const INPUT_0: &[usize] = &[0];
    const INPUT_0_1: &[usize] = &[0, 1];
    match op_name {
        "RESHAPE" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: true,
            source_file: "tensorflow/lite/kernels/reshape.cc",
        }),
        "SQUEEZE" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: true,
            source_file: "tensorflow/lite/kernels/squeeze.cc",
        }),
        "BITCAST" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: true,
            source_file: "tensorflow/lite/kernels/bitcast.cc",
        }),
        "EXPAND_DIMS" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: true,
            source_file: "tensorflow/lite/kernels/expand_dims.cc",
        }),
        "SOFTMAX" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/activations.cc",
        }),
        "DYNAMIC_UPDATE_SLICE" => Some(InPlaceRegistration {
            input_slots: INPUT_0,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/dynamic_update_slice.cc",
        }),
        "ADD" => Some(InPlaceRegistration {
            input_slots: INPUT_0_1,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/add.cc",
        }),
        "SUB" => Some(InPlaceRegistration {
            input_slots: INPUT_0_1,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/sub.cc",
        }),
        "MUL" => Some(InPlaceRegistration {
            input_slots: INPUT_0_1,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/mul.cc",
        }),
        "DIV" => Some(InPlaceRegistration {
            input_slots: INPUT_0_1,
            data_unmodified: false,
            source_file: "tensorflow/lite/kernels/div.cc",
        }),
        _ => None,
    }
}

pub(super) fn declared_tensor_payload_bytes(tensor: &TensorInfo) -> Result<usize, String> {
    if tensor.shape.iter().any(|dim| *dim < 0) {
        return Err("Declared tensor shape contains a dynamic or unknown dimension.".to_string());
    }
    let elements = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, dim| product.checked_mul(*dim as usize))
        .ok_or_else(|| "Tensor element count exceeds the analyzer integer range.".to_string())?;
    if tensor.dtype == "INT4" {
        return elements
            .checked_add(1)
            .map(|value| value / 2)
            .ok_or_else(|| "Packed INT4 payload exceeds the analyzer integer range.".to_string());
    }
    let scalar_bytes = match tensor.dtype.as_str() {
        "COMPLEX128" => 16usize,
        "FLOAT64" | "INT64" | "UINT64" | "COMPLEX64" => 8,
        "FLOAT32" | "INT32" | "UINT32" => 4,
        "FLOAT16" | "BFLOAT16" | "INT16" | "UINT16" => 2,
        "INT8" | "UINT8" | "BOOL" => 1,
        _ => {
            return Err(format!(
                "{} does not have a fixed inline scalar width for arena projection.",
                tensor.dtype
            ))
        }
    };
    elements
        .checked_mul(scalar_bytes)
        .ok_or_else(|| "Tensor payload exceeds the analyzer integer range.".to_string())
}

fn arena_class_for_tensor(tensor: &TensorInfo) -> Option<ArenaPlanClass> {
    if tensor.constant_buffer {
        None
    } else if tensor.is_variable {
        Some(ArenaPlanClass::Persistent)
    } else {
        Some(ArenaPlanClass::NonPersistent)
    }
}

fn resolve_shared_tensor(tensor_index: usize, shared_roots: &HashMap<usize, usize>) -> usize {
    shared_roots
        .get(&tensor_index)
        .copied()
        .unwrap_or(tensor_index)
}

fn usage_intervals_overlap(
    left_first: usize,
    left_last: Option<usize>,
    right_first: usize,
    right_last: Option<usize>,
) -> bool {
    let left_last = left_last.unwrap_or(usize::MAX);
    let right_last = right_last.unwrap_or(usize::MAX);
    !(left_last < right_first || left_first > right_last)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, String> {
    if alignment == 0 {
        return Err("Arena alignment must be non-zero.".to_string());
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| "Aligned arena offset exceeds the analyzer integer range.".to_string())
    }
}

fn place_arena_candidate(
    state: &mut ArenaPlacementState,
    candidate: ArenaPlanCandidate,
    alignment: usize,
) -> Result<usize, String> {
    if candidate.size_bytes == 0 {
        return Ok(0);
    }
    let mut current_offset = 0usize;
    let mut best_offset = None;
    let mut best_offset_fit = usize::MAX;
    for allocation in &state.allocations {
        if !usage_intervals_overlap(
            allocation.candidate.first_node,
            allocation.candidate.last_node,
            candidate.first_node,
            candidate.last_node,
        ) {
            continue;
        }
        let aligned_current_offset = align_up(current_offset, alignment)?;
        let aligned_current_end = aligned_current_offset
            .checked_add(candidate.size_bytes)
            .ok_or_else(|| {
                "Arena allocation end exceeds the analyzer integer range.".to_string()
            })?;
        if aligned_current_end <= allocation.offset_bytes {
            let fit = allocation
                .offset_bytes
                .saturating_sub(aligned_current_offset);
            if fit < best_offset_fit {
                best_offset = Some(aligned_current_offset);
                best_offset_fit = allocation.offset_bytes.saturating_sub(current_offset);
            }
        }
        current_offset = current_offset.max(
            allocation
                .offset_bytes
                .checked_add(allocation.candidate.size_bytes)
                .ok_or_else(|| {
                    "Existing arena allocation exceeds the analyzer integer range.".to_string()
                })?,
        );
        if best_offset_fit == 0 {
            break;
        }
    }
    let offset = match best_offset {
        Some(value) => value,
        None => align_up(current_offset, alignment)?,
    };
    let required = offset
        .checked_add(candidate.size_bytes)
        .ok_or_else(|| "Arena high-water mark exceeds the analyzer integer range.".to_string())?;
    state.high_water_mark = state.high_water_mark.max(required);
    let insertion = state
        .allocations
        .partition_point(|allocation| allocation.offset_bytes <= offset);
    state.allocations.insert(
        insertion,
        PlacedArenaAllocation {
            candidate,
            offset_bytes: offset,
        },
    );
    Ok(offset)
}

pub(super) fn compute_tensor_arena_plan(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    input_tensor_indices: &[i32],
    output_tensor_indices: &[i32],
) -> TensorArenaPlanProjection {
    let input_set = input_tensor_indices
        .iter()
        .filter(|index| **index >= 0)
        .map(|index| *index as usize)
        .collect::<HashSet<_>>();
    let output_set = output_tensor_indices
        .iter()
        .filter(|index| **index >= 0)
        .map(|index| *index as usize)
        .collect::<HashSet<_>>();
    let mut producer_by_tensor = HashMap::<usize, usize>::new();
    let mut logical_last_consumer = HashMap::<usize, usize>::new();
    for op in ops {
        for &tensor_index in &op.outputs {
            if tensor_index >= 0 {
                producer_by_tensor
                    .entry(tensor_index as usize)
                    .or_insert(op.index);
            }
        }
        for &tensor_index in &op.inputs {
            if tensor_index >= 0 {
                logical_last_consumer
                    .entry(tensor_index as usize)
                    .and_modify(|last| *last = (*last).max(op.index))
                    .or_insert(op.index);
            }
        }
    }
    let considered = tensors
        .iter()
        .filter(|tensor| {
            !tensor.constant_buffer
                && (input_set.contains(&tensor.index)
                    || output_set.contains(&tensor.index)
                    || tensor.is_variable
                    || producer_by_tensor.contains_key(&tensor.index))
        })
        .map(|tensor| tensor.index)
        .collect::<HashSet<_>>();

    let mut allocation_nodes = vec![None::<usize>; tensors.len()];
    for &tensor_index in input_set.iter() {
        if tensor_index < allocation_nodes.len() {
            allocation_nodes[tensor_index] = Some(0);
        }
    }
    for tensor in tensors.iter().filter(|tensor| tensor.is_variable) {
        allocation_nodes[tensor.index] = Some(0);
    }
    for op in ops {
        for &tensor_index in &op.outputs {
            if tensor_index >= 0 {
                let tensor_index = tensor_index as usize;
                if tensor_index < allocation_nodes.len() && allocation_nodes[tensor_index].is_none()
                {
                    allocation_nodes[tensor_index] = Some(op.index);
                }
            }
        }
    }

    let mut original_refcounts = vec![0usize; tensors.len()];
    for tensor_index in output_set.iter().chain(input_set.iter()).copied().chain(
        tensors
            .iter()
            .filter(|tensor| tensor.is_variable)
            .map(|tensor| tensor.index),
    ) {
        if tensor_index < original_refcounts.len() {
            original_refcounts[tensor_index] += 1;
        }
    }
    for op in ops {
        for &tensor_index in &op.inputs {
            if tensor_index >= 0 && (tensor_index as usize) < original_refcounts.len() {
                original_refcounts[tensor_index as usize] += 1;
            }
        }
    }

    let declared_sizes = tensors
        .iter()
        .map(declared_tensor_payload_bytes)
        .collect::<Vec<_>>();
    let mut shared_roots = HashMap::<usize, usize>::new();
    let mut aliases = Vec::<ArenaPlanAlias>::new();
    for op in ops {
        let Some(registration) = pinned_in_place_registration(&op.name) else {
            continue;
        };
        let Some(&output_index) = op.outputs.first() else {
            continue;
        };
        if output_index < 0 {
            continue;
        }
        let output_index = output_index as usize;
        let Some(output_tensor) = tensors.get(output_index) else {
            continue;
        };
        if output_set.contains(&output_index) || output_tensor.constant_buffer {
            continue;
        }
        for &input_slot in registration.input_slots {
            let Some(&input_index) = op.inputs.get(input_slot) else {
                continue;
            };
            if input_index < 0 {
                continue;
            }
            let input_index = input_index as usize;
            let Some(input_tensor) = tensors.get(input_index) else {
                continue;
            };
            if input_set.contains(&input_index) || input_tensor.constant_buffer {
                continue;
            }
            let Some(input_arena) = arena_class_for_tensor(input_tensor) else {
                continue;
            };
            let Some(output_arena) = arena_class_for_tensor(output_tensor) else {
                continue;
            };
            if input_arena != ArenaPlanClass::NonPersistent
                || output_arena != ArenaPlanClass::NonPersistent
            {
                continue;
            }
            let (Ok(input_bytes), Ok(output_bytes)) =
                (&declared_sizes[input_index], &declared_sizes[output_index])
            else {
                continue;
            };
            if input_bytes != output_bytes {
                continue;
            }
            if !registration.data_unmodified
                && (*input_bytes <= 4 || original_refcounts[input_index] > 1)
            {
                continue;
            }
            let root = resolve_shared_tensor(input_index, &shared_roots);
            if !registration.data_unmodified
                && original_refcounts.get(root).copied().unwrap_or(usize::MAX) > 1
            {
                continue;
            }
            shared_roots.insert(output_index, root);
            aliases.push(ArenaPlanAlias {
                tensor_index: output_index,
                tensor_name: output_tensor.name.clone(),
                shared_with_tensor_index: root,
                shared_with_tensor_name: tensors
                    .get(root)
                    .map(|tensor| tensor.name.clone())
                    .unwrap_or_default(),
                op_index: op.index,
                op_name: op.name.clone(),
                input_slot,
                data_unmodified: registration.data_unmodified,
                source: format!(
                    "tensorflow/tensorflow@{}/{}",
                    TFLITE_ARENA_SOURCE_COMMIT, registration.source_file
                ),
            });
            break;
        }
    }

    let mut runtime_refcounts = vec![0usize; tensors.len()];
    for tensor_index in output_set.iter().chain(input_set.iter()).copied().chain(
        tensors
            .iter()
            .filter(|tensor| tensor.is_variable)
            .map(|tensor| tensor.index),
    ) {
        let root = resolve_shared_tensor(tensor_index, &shared_roots);
        if root < runtime_refcounts.len() {
            runtime_refcounts[root] += 1;
        }
    }
    for op in ops {
        for &tensor_index in &op.inputs {
            if tensor_index < 0 {
                continue;
            }
            let root = resolve_shared_tensor(tensor_index as usize, &shared_roots);
            if root < runtime_refcounts.len() {
                runtime_refcounts[root] += 1;
            }
        }
    }
    let mut deallocation_nodes = vec![None::<usize>; tensors.len()];
    for op in ops {
        for &tensor_index in &op.inputs {
            if tensor_index < 0 {
                continue;
            }
            let root = resolve_shared_tensor(tensor_index as usize, &shared_roots);
            let Some(refcount) = runtime_refcounts.get_mut(root) else {
                continue;
            };
            if *refcount > 0 {
                *refcount -= 1;
                if *refcount == 0 {
                    deallocation_nodes[root] = Some(op.index);
                }
            }
        }
    }

    let mut candidates = Vec::<ArenaPlanCandidate>::new();
    let mut issues = Vec::<ArenaPlanTensorIssue>::new();
    let mut dynamic_shape_signature_tensor_count = 0usize;
    let mut considered_indices = considered.iter().copied().collect::<Vec<_>>();
    considered_indices.sort_unstable();
    for &tensor_index in &considered_indices {
        let Some(tensor) = tensors.get(tensor_index) else {
            continue;
        };
        if tensor.shape_signature.iter().any(|dim| *dim < 0) {
            dynamic_shape_signature_tensor_count += 1;
        }
        if shared_roots.contains_key(&tensor_index) {
            continue;
        }
        let Some(first_node) = allocation_nodes[tensor_index] else {
            continue;
        };
        let Some(arena) = arena_class_for_tensor(tensor) else {
            continue;
        };
        match &declared_sizes[tensor_index] {
            Ok(size_bytes) => candidates.push(ArenaPlanCandidate {
                tensor_index,
                size_bytes: *size_bytes,
                first_node,
                last_node: if input_set.contains(&tensor_index)
                    || output_set.contains(&tensor_index)
                    || tensor.is_variable
                {
                    None
                } else {
                    deallocation_nodes[tensor_index]
                },
                arena,
            }),
            Err(reason) => issues.push(ArenaPlanTensorIssue {
                tensor_index: Some(tensor_index),
                tensor_name: tensor.name.clone(),
                reason: reason.clone(),
            }),
        }
    }

    let mut comparator_ties = BTreeMap::<(usize, usize), usize>::new();
    for candidate in &candidates {
        if !(candidate.first_node == 0 && candidate.last_node.is_none()) {
            *comparator_ties
                .entry((candidate.size_bytes, candidate.first_node))
                .or_insert(0) += 1;
        }
    }
    let source_comparator_tie_group_count =
        comparator_ties.values().filter(|count| **count > 1).count();
    let source_comparator_tied_tensor_count =
        comparator_ties.values().filter(|count| **count > 1).sum();
    candidates.sort_by(|left, right| {
        let left_full = left.first_node == 0 && left.last_node.is_none();
        let right_full = right.first_node == 0 && right.last_node.is_none();
        match (left_full, right_full) {
            (true, true) => left.tensor_index.cmp(&right.tensor_index),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => right
                .size_bytes
                .cmp(&left.size_bytes)
                .then_with(|| left.first_node.cmp(&right.first_node))
                .then_with(|| left.tensor_index.cmp(&right.tensor_index)),
        }
    });

    let mut non_persistent = ArenaPlacementState::default();
    let mut persistent = ArenaPlacementState::default();
    let mut placed_offsets = HashMap::<usize, usize>::new();
    let mut placement_failure = None::<String>;
    for candidate in candidates.iter().cloned() {
        let state = if candidate.arena == ArenaPlanClass::Persistent {
            &mut persistent
        } else {
            &mut non_persistent
        };
        match place_arena_candidate(state, candidate.clone(), TFLITE_TENSOR_ALIGNMENT_BYTES) {
            Ok(offset) => {
                placed_offsets.insert(candidate.tensor_index, offset);
            }
            Err(error) => {
                placement_failure = Some(error);
                break;
            }
        }
    }
    if let Some(reason) = placement_failure {
        issues.push(ArenaPlanTensorIssue {
            tensor_index: None,
            tensor_name: "arena placement".to_string(),
            reason,
        });
    }

    let candidate_by_tensor = candidates
        .iter()
        .map(|candidate| (candidate.tensor_index, candidate))
        .collect::<HashMap<_, _>>();
    let mut allocations = Vec::<ArenaPlanAllocation>::new();
    for &tensor_index in &considered_indices {
        let Some(tensor) = tensors.get(tensor_index) else {
            continue;
        };
        if let Some(&root) = shared_roots.get(&tensor_index) {
            allocations.push(ArenaPlanAllocation {
                tensor_index,
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                arena: arena_class_for_tensor(tensor)
                    .map(ArenaPlanClass::label)
                    .unwrap_or("not_arena")
                    .to_string(),
                size_bytes: declared_sizes[tensor_index].as_ref().ok().copied(),
                offset_bytes: placed_offsets.get(&root).copied(),
                first_node: allocation_nodes[tensor_index].unwrap_or(0),
                last_node: if output_set.contains(&tensor_index) {
                    None
                } else {
                    logical_last_consumer.get(&tensor_index).copied()
                },
                shared_with_tensor_index: Some(root),
                allocation_status: "shared_in_place".to_string(),
            });
        } else if let Some(candidate) = candidate_by_tensor.get(&tensor_index) {
            allocations.push(ArenaPlanAllocation {
                tensor_index,
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                arena: candidate.arena.label().to_string(),
                size_bytes: Some(candidate.size_bytes),
                offset_bytes: placed_offsets.get(&tensor_index).copied(),
                first_node: candidate.first_node,
                last_node: candidate.last_node,
                shared_with_tensor_index: None,
                allocation_status: if placed_offsets.contains_key(&tensor_index) {
                    "allocated"
                } else {
                    "not_assessed"
                }
                .to_string(),
            });
        } else if allocation_nodes[tensor_index].is_some() {
            allocations.push(ArenaPlanAllocation {
                tensor_index,
                tensor_name: tensor.name.clone(),
                tensor_shape: tensor.shape.clone(),
                tensor_dtype: tensor.dtype.clone(),
                arena: arena_class_for_tensor(tensor)
                    .map(ArenaPlanClass::label)
                    .unwrap_or("not_arena")
                    .to_string(),
                size_bytes: None,
                offset_bytes: None,
                first_node: allocation_nodes[tensor_index].unwrap_or(0),
                last_node: logical_last_consumer.get(&tensor_index).copied(),
                shared_with_tensor_index: None,
                allocation_status: "not_assessed".to_string(),
            });
        }
    }

    let non_persistent_allocation_count = candidates
        .iter()
        .filter(|candidate| candidate.arena == ArenaPlanClass::NonPersistent)
        .count();
    let persistent_allocation_count = candidates
        .iter()
        .filter(|candidate| candidate.arena == ArenaPlanClass::Persistent)
        .count();
    let status = if issues.is_empty() {
        "assessed"
    } else if candidates.is_empty() {
        "not_assessed"
    } else {
        "partial"
    };
    let arena_totals_assessed = issues.is_empty();
    let combined = arena_totals_assessed
        .then(|| {
            non_persistent
                .high_water_mark
                .checked_add(persistent.high_water_mark)
        })
        .flatten();
    let unassessed_tensor_count = issues
        .iter()
        .filter(|issue| issue.tensor_index.is_some())
        .count();
    TensorArenaPlanProjection {
        schema: "deepbom.tflite_arena_plan_projection.v1".to_string(),
        status: status.to_string(),
        evidence_class: "DERIVED".to_string(),
        source_commit: TFLITE_ARENA_SOURCE_COMMIT.to_string(),
        planner_source_url: format!(
            "https://github.com/tensorflow/tensorflow/blob/{}/tensorflow/lite/arena_planner.cc",
            TFLITE_ARENA_SOURCE_COMMIT
        ),
        arena_source_url: format!(
            "https://github.com/tensorflow/tensorflow/blob/{}/tensorflow/lite/simple_memory_arena.cc",
            TFLITE_ARENA_SOURCE_COMMIT
        ),
        registration_source_basis: vec![
            "reshape.cc, squeeze.cc, bitcast.cc, expand_dims.cc: input0 shared / data unmodified".to_string(),
            "add.cc, sub.cc, mul.cc, div.cc: input0 or input1 shared when changed-buffer constraints pass".to_string(),
            "activations.cc SOFTMAX and dynamic_update_slice.cc: input0 shared when changed-buffer constraints pass".to_string(),
        ],
        tensor_alignment_bytes: TFLITE_TENSOR_ALIGNMENT_BYTES,
        preserve_all_tensors: false,
        non_persistent_arena_bytes: arena_totals_assessed
            .then_some(non_persistent.high_water_mark),
        persistent_arena_bytes: arena_totals_assessed.then_some(persistent.high_water_mark),
        combined_arena_bytes: combined,
        planned_tensor_count: allocations.len(),
        root_allocation_count: candidates.len(),
        non_persistent_allocation_count,
        persistent_allocation_count,
        shared_tensor_count: aliases.len(),
        dynamic_shape_signature_tensor_count,
        source_comparator_tie_group_count,
        source_comparator_tied_tensor_count,
        source_comparator_fully_orders_projection: source_comparator_tie_group_count == 0,
        deterministic_tie_break: "For non-full-lifetime tensors with equal byte size and allocation node, DeepBOM orders by tensor index. TensorFlow's pinned comparator leaves that case equivalent under std::sort.".to_string(),
        unassessed_tensor_count,
        calculation_issue_count: issues.len(),
        allocations,
        aliases,
        unassessed_tensors: issues,
        method: "Pinned ArenaPlanner lifetimes and in-place aliases followed by SimpleMemoryArena size-descending, 64-byte-aligned best-fit offset placement over declared tensor shapes; tensor index is the declared deterministic extension for source-comparator ties.".to_string(),
        interpretation_boundary: "A pinned-source declared-shape projection, not observed runtime memory. Source-comparator ties are reported explicitly because C++ std::sort does not promise an order among comparator-equivalent tensors. FlatBuffer-unrepresented Prepare-time resize, node temporaries, kernel scratch, custom allocations/resolvers, delegate-owned buffers, allocation-type changes, and runtime build differences are excluded; dynamic shape signatures are counted explicitly.".to_string(),
    }
}

const MOVEMENT_OP_NAMES: &[&str] = &[
    "TRANSPOSE",
    "PAD",
    "PAD_V2",
    "MIRROR_PAD",
    "CONCATENATION",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "QUANTIZE",
    "DEQUANTIZE",
    "SLICE",
    "STRIDED_SLICE",
    "GATHER",
    "GATHER_ND",
    "SPLIT",
    "SPLIT_V",
    "TILE",
];

pub(super) fn compute_movement_analysis(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
) -> MovementAnalysis {
    let mut total_bytes: usize = 0;
    let mut xnn_break_bytes: usize = 0;
    let movement_op_count = ops
        .iter()
        .filter(|op| MOVEMENT_OP_NAMES.contains(&op.name.as_str()))
        .count();
    let mut assessed_output_tensor_count = 0usize;
    let mut issues = Vec::<TensorPayloadIssue>::new();
    for op in ops {
        if !MOVEMENT_OP_NAMES.contains(&op.name.as_str()) {
            continue;
        }
        let mut out_bytes = 0usize;
        for tensor_index in op.outputs.iter().copied().filter(|index| *index >= 0) {
            let tensor_index = tensor_index as usize;
            let Some(tensor) = tensors.get(tensor_index) else {
                issues.push(TensorPayloadIssue {
                    tensor_index: Some(tensor_index),
                    tensor_name: String::new(),
                    reason: "Tensor index is outside the parsed tensor inventory.".to_string(),
                });
                continue;
            };
            match declared_tensor_payload_bytes(tensor) {
                Ok(size) => {
                    if let Some(next) = out_bytes.checked_add(size) {
                        assessed_output_tensor_count += 1;
                        out_bytes = next;
                    } else {
                        issues.push(TensorPayloadIssue {
                            tensor_index: Some(tensor_index),
                            tensor_name: tensor.name.clone(),
                            reason: "Movement payload sum exceeds the analyzer integer range."
                                .to_string(),
                        });
                    }
                }
                Err(reason) => issues.push(TensorPayloadIssue {
                    tensor_index: Some(tensor_index),
                    tensor_name: tensor.name.clone(),
                    reason,
                }),
            }
        }
        if let Some(next) = total_bytes.checked_add(out_bytes) {
            total_bytes = next;
        } else {
            issues.push(TensorPayloadIssue {
                tensor_index: None,
                tensor_name: format!("movement op #{} {}", op.index, op.name),
                reason: "Total movement payload exceeds the analyzer integer range.".to_string(),
            });
        }
        if op.xnnpack_chain_break {
            if let Some(next) = xnn_break_bytes.checked_add(out_bytes) {
                xnn_break_bytes = next;
            } else {
                issues.push(TensorPayloadIssue {
                    tensor_index: None,
                    tensor_name: format!("predicted break op #{} {}", op.index, op.name),
                    reason: "Predicted-break movement payload exceeds the analyzer integer range."
                        .to_string(),
                });
            }
        }
    }
    let ratio = if ops.is_empty() {
        0.0
    } else {
        movement_op_count as f64 / ops.len() as f64
    };
    let unassessed_output_tensor_count = issues
        .iter()
        .filter(|issue| issue.tensor_index.is_some())
        .count();
    MovementAnalysis {
        status: if issues.is_empty() {
            "assessed"
        } else {
            "partial"
        }
        .to_string(),
        total_movement_bytes_value: issues.is_empty().then_some(total_bytes),
        total_movement_bytes: total_bytes,
        assessed_movement_bytes: total_bytes,
        movement_op_count,
        xnn_break_movement_bytes_value: issues.is_empty().then_some(xnn_break_bytes),
        xnn_break_movement_bytes: xnn_break_bytes,
        assessed_xnn_break_movement_bytes: xnn_break_bytes,
        assessed_output_tensor_count,
        unassessed_output_tensor_count,
        calculation_issue_count: issues.len(),
        unassessed_tensors: issues,
        movement_op_ratio: ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_memory_arena_projection_uses_aligned_non_overlapping_offsets() {
        let mut state = ArenaPlacementState::default();
        let first = ArenaPlanCandidate {
            tensor_index: 0,
            size_bytes: 100,
            first_node: 0,
            last_node: Some(2),
            arena: ArenaPlanClass::NonPersistent,
        };
        let disjoint = ArenaPlanCandidate {
            tensor_index: 1,
            size_bytes: 80,
            first_node: 3,
            last_node: Some(4),
            arena: ArenaPlanClass::NonPersistent,
        };
        let bridge = ArenaPlanCandidate {
            tensor_index: 2,
            size_bytes: 32,
            first_node: 1,
            last_node: Some(3),
            arena: ArenaPlanClass::NonPersistent,
        };

        assert_eq!(place_arena_candidate(&mut state, first, 64).unwrap(), 0);
        assert_eq!(place_arena_candidate(&mut state, disjoint, 64).unwrap(), 0);
        assert_eq!(place_arena_candidate(&mut state, bridge, 64).unwrap(), 128);
        assert_eq!(state.high_water_mark, 160);
        assert!(state
            .allocations
            .iter()
            .all(|allocation| allocation.offset_bytes % 64 == 0));
    }
}
