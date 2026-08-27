use crate::{OpInfo, PatternInfo, TargetProfile, TensorInfo, L1_WORKING_SET_WATCH_RATIO};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const SCHEMA: &str = "deepbom.block_inventory.v1.1";

#[derive(Clone, Serialize)]
pub(crate) struct BlockInventory {
    pub(crate) schema: String,
    pub(crate) status: String,
    pub(crate) evidence_class: String,
    pub(crate) stage_count: usize,
    pub(crate) block_count: usize,
    pub(crate) named_block_count: usize,
    pub(crate) semantic_block_count: usize,
    pub(crate) unnamed_block_count: usize,
    pub(crate) stages: Vec<BlockStage>,
    pub(crate) blocks: Vec<BlockRecord>,
    pub(crate) method: String,
    pub(crate) interpretation_boundary: String,
}

#[derive(Clone, Serialize)]
pub(crate) struct BlockStage {
    pub(crate) stage_id: String,
    pub(crate) index: usize,
    pub(crate) display_name: String,
    pub(crate) spatial: SpatialShape,
    pub(crate) block_ids: Vec<String>,
    pub(crate) op_indices: Vec<usize>,
    pub(crate) aggregates: BlockAggregates,
}

#[derive(Clone, Serialize)]
pub(crate) struct BlockRecord {
    pub(crate) block_id: String,
    pub(crate) stage_index: usize,
    pub(crate) block_type: String,
    pub(crate) display_name: String,
    pub(crate) extraction: BlockExtraction,
    pub(crate) op_indices: Vec<usize>,
    pub(crate) spatial: SpatialShape,
    pub(crate) channels: BlockChannels,
    pub(crate) params: BlockParams,
    pub(crate) repeat: BlockRepeat,
    pub(crate) residual: bool,
    pub(crate) aggregates: BlockAggregates,
}

#[derive(Clone, Serialize)]
pub(crate) struct BlockExtraction {
    pub(crate) method: String,
    pub(crate) confidence: String,
    pub(crate) source_pattern: String,
}

#[derive(Clone, Copy, Default, Serialize)]
pub(crate) struct SpatialShape {
    pub(crate) input_h: Option<usize>,
    pub(crate) input_w: Option<usize>,
    pub(crate) output_h: Option<usize>,
    pub(crate) output_w: Option<usize>,
}

#[derive(Clone, Copy, Default, Serialize)]
pub(crate) struct BlockChannels {
    pub(crate) input: Option<usize>,
    pub(crate) expand: Option<usize>,
    pub(crate) output: Option<usize>,
}

#[derive(Clone, Copy, Default, Serialize)]
pub(crate) struct BlockParams {
    pub(crate) expand_ratio: Option<f64>,
    pub(crate) stride_h: Option<usize>,
    pub(crate) stride_w: Option<usize>,
    pub(crate) kernel_h: Option<usize>,
    pub(crate) kernel_w: Option<usize>,
}

#[derive(Clone, Copy, Default, Serialize)]
pub(crate) struct BlockRepeat {
    pub(crate) index: usize,
    pub(crate) total: usize,
}

#[derive(Clone, Default, Serialize)]
pub(crate) struct IntensityCounts {
    pub(crate) compute: usize,
    pub(crate) mixed: usize,
    pub(crate) memory: usize,
    pub(crate) not_assessed: usize,
}

#[derive(Clone, Default, Serialize)]
pub(crate) struct BlockAggregates {
    pub(crate) op_count: usize,
    pub(crate) macs: f64,
    pub(crate) mac_percent: f64,
    pub(crate) modeled_time_ms: f64,
    pub(crate) modeled_cold_start_time_ms: f64,
    pub(crate) time_evidence_class: String,
    pub(crate) l1_max_ratio: Option<f64>,
    pub(crate) l1_max_op_index: Option<usize>,
    pub(crate) l1_watch_count: usize,
    pub(crate) cache_assessed_op_count: usize,
    pub(crate) logical_traffic_bytes: usize,
    pub(crate) predicted_break_count: usize,
    pub(crate) parameter_elements: usize,
    pub(crate) serialized_parameter_bytes: usize,
    pub(crate) intensity: IntensityCounts,
}

#[derive(Clone)]
struct BlockDraft {
    block_type: String,
    display_name: String,
    extraction: BlockExtraction,
    op_indices: Vec<usize>,
}

pub(crate) fn build_block_inventory(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    patterns: &[PatternInfo],
    target: &TargetProfile,
    total_macs: f64,
) -> BlockInventory {
    if ops.is_empty() {
        return BlockInventory {
            schema: SCHEMA.to_string(),
            status: "not_assessed_empty_graph".to_string(),
            evidence_class: "NOT_ASSESSABLE".to_string(),
            stage_count: 0,
            block_count: 0,
            named_block_count: 0,
            semantic_block_count: 0,
            unnamed_block_count: 0,
            stages: Vec::new(),
            blocks: Vec::new(),
            method: "No operators were available for block extraction.".to_string(),
            interpretation_boundary: "No architecture block claim was emitted.".to_string(),
        };
    }

    let op_by_index = ops
        .iter()
        .map(|op| (op.index, op))
        .collect::<HashMap<_, _>>();
    let mut drafts = semantic_drafts(patterns, &op_by_index);
    let mut covered = drafts
        .iter()
        .flat_map(|draft| draft.op_indices.iter().copied())
        .collect::<HashSet<_>>();
    drafts.extend(named_and_shape_drafts(ops, tensors, &mut covered));
    drafts.sort_by_key(|draft| draft.op_indices.first().copied().unwrap_or(usize::MAX));

    let mut blocks = drafts
        .into_iter()
        .enumerate()
        .map(|(index, draft)| {
            let block_ops = draft
                .op_indices
                .iter()
                .filter_map(|op_index| op_by_index.get(op_index).copied())
                .collect::<Vec<_>>();
            let spatial = block_spatial(&block_ops, tensors);
            let channels = block_channels(&block_ops, tensors);
            let params = block_params(&block_ops, tensors, spatial, channels);
            let residual = has_block_residual(&block_ops);
            let structural_prelude = block_ops
                .iter()
                .all(|op| matches!(op.name.as_str(), "PAD" | "PADV2"));
            BlockRecord {
                block_id: format!("block_{index:03}"),
                stage_index: 0,
                block_type: if structural_prelude {
                    "spatial_prelude".to_string()
                } else {
                    draft.block_type
                },
                display_name: if structural_prelude {
                    "Padding prelude".to_string()
                } else {
                    draft.display_name
                },
                extraction: draft.extraction,
                op_indices: draft.op_indices,
                spatial,
                channels,
                params,
                repeat: BlockRepeat { index: 0, total: 1 },
                residual,
                aggregates: aggregate_ops(&block_ops, tensors, target, total_macs),
            }
        })
        .collect::<Vec<_>>();

    let stage_groups = group_blocks_into_stages(&blocks);
    let mut stages = Vec::<BlockStage>::new();
    for (stage_index, block_indices) in stage_groups.into_iter().enumerate() {
        let mut op_indices = block_indices
            .iter()
            .flat_map(|index| blocks[*index].op_indices.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        op_indices.sort_unstable();
        let stage_ops = op_indices
            .iter()
            .filter_map(|index| op_by_index.get(index).copied())
            .collect::<Vec<_>>();
        let block_ids = block_indices
            .iter()
            .map(|index| blocks[*index].block_id.clone())
            .collect::<Vec<_>>();
        let spatial = merge_stage_spatial(
            block_indices
                .iter()
                .filter_map(|index| blocks.get(*index).map(|block| block.spatial)),
        );
        let display_name = stage_display_name(
            block_indices
                .iter()
                .filter_map(|index| blocks.get(*index).map(|block| block.block_type.as_str())),
        );
        for (repeat_index, block_index) in block_indices.iter().enumerate() {
            blocks[*block_index].stage_index = stage_index;
            blocks[*block_index].repeat = repeat_for_block(*block_index, &blocks);
            if blocks[*block_index].repeat.total == 1 {
                blocks[*block_index].repeat.index = repeat_index;
            }
        }
        stages.push(BlockStage {
            stage_id: format!("stage_{stage_index:02}"),
            index: stage_index,
            display_name,
            spatial,
            block_ids,
            op_indices,
            aggregates: aggregate_ops(&stage_ops, tensors, target, total_macs),
        });
    }

    let semantic_block_count = blocks
        .iter()
        .filter(|block| block.extraction.method == "graph_pattern_matched")
        .count();
    let named_block_count = blocks
        .iter()
        .filter(|block| block.extraction.method == "name_parsed")
        .count();
    let unnamed_block_count = blocks
        .iter()
        .filter(|block| block.extraction.method == "shape_grouped")
        .count();
    BlockInventory {
        schema: SCHEMA.to_string(),
        status: "assessed".to_string(),
        evidence_class: "DERIVED".to_string(),
        stage_count: stages.len(),
        block_count: blocks.len(),
        named_block_count,
        semantic_block_count,
        unnamed_block_count,
        stages,
        blocks,
        method: "Graph-semantic patterns are assigned first; artifact names label otherwise-unclaimed contiguous groups; remaining operators are grouped by contiguous spatial transition. PAD/PADV2-only preludes inherit the following compute block's stage key. Stage aggregates use unique op indices, steady/cold time ledgers, MAC/traffic sums, and maximum logical-row-payload/L1 ratio.".to_string(),
        interpretation_boundary: "Names never prove an architecture type. Pattern matches describe serialized deployment-graph motifs, not the original training modules. Modeled time is target-profile estimated; logical traffic is not measured DRAM traffic; cache ratios are not residency observations.".to_string(),
    }
}

fn semantic_drafts(
    patterns: &[PatternInfo],
    op_by_index: &HashMap<usize, &OpInfo>,
) -> Vec<BlockDraft> {
    let mut drafts = Vec::new();
    for pattern in patterns {
        let op_indices = (pattern.first_op..=pattern.last_op)
            .filter(|index| op_by_index.contains_key(index))
            .collect::<Vec<_>>();
        if op_indices.is_empty() {
            continue;
        }
        let (block_type, display_name) = match pattern.name.as_str() {
            "MBConv-like block" => ("inverted_bottleneck", "Inverted bottleneck"),
            "Depthwise-separable convolution pair" => {
                ("depthwise_separable", "Depthwise-separable block")
            }
            "SE block" => ("squeeze_excitation", "Squeeze-excitation"),
            "FPN/upsample merge" => ("feature_pyramid_merge", "FPN upsample merge"),
            "LayerNorm-like block" => ("layer_normalization", "Layer normalization"),
            "ASPP-like block" => ("aspp", "Atrous spatial pyramid"),
            _ => ("recognized_pattern", pattern.name.as_str()),
        };
        drafts.push(BlockDraft {
            block_type: block_type.to_string(),
            display_name: display_name.to_string(),
            extraction: BlockExtraction {
                method: "graph_pattern_matched".to_string(),
                confidence: "high".to_string(),
                source_pattern: pattern.name.clone(),
            },
            op_indices,
        });
    }
    drafts
}

fn named_and_shape_drafts(
    ops: &[OpInfo],
    tensors: &[TensorInfo],
    covered: &mut HashSet<usize>,
) -> Vec<BlockDraft> {
    let mut drafts = Vec::new();
    let mut cursor = 0usize;
    while cursor < ops.len() {
        if covered.contains(&ops[cursor].index) {
            cursor += 1;
            continue;
        }
        let name_key = op_name_key(&ops[cursor], tensors);
        let spatial_key = op_spatial_key(&ops[cursor]);
        let mut end = cursor + 1;
        while end < ops.len() && !covered.contains(&ops[end].index) {
            let next_name = op_name_key(&ops[end], tensors);
            let next_spatial = op_spatial_key(&ops[end]);
            let same = if name_key.is_some() {
                next_name == name_key
            } else {
                next_name.is_none() && next_spatial == spatial_key
            };
            if !same {
                break;
            }
            end += 1;
        }
        let op_indices = ops[cursor..end]
            .iter()
            .map(|op| op.index)
            .collect::<Vec<_>>();
        covered.extend(op_indices.iter().copied());
        let (block_type, display_name, extraction) = if let Some(key) = name_key {
            let block_type = named_block_type(&key, &ops[cursor..end]);
            (
                block_type.to_string(),
                humanize_name_key(&key),
                BlockExtraction {
                    method: "name_parsed".to_string(),
                    confidence: if block_type == "named_group" {
                        "medium"
                    } else {
                        "medium_high"
                    }
                    .to_string(),
                    source_pattern: key,
                },
            )
        } else {
            (
                "unnamed_group".to_string(),
                spatial_key
                    .as_deref()
                    .map(humanize_spatial_key)
                    .unwrap_or_else(|| "Unresolved operator group".to_string()),
                BlockExtraction {
                    method: "shape_grouped".to_string(),
                    confidence: "low".to_string(),
                    source_pattern: spatial_key.unwrap_or_else(|| "no spatial key".to_string()),
                },
            )
        };
        drafts.push(BlockDraft {
            block_type,
            display_name,
            extraction,
            op_indices,
        });
        cursor = end;
    }
    drafts
}

fn op_name_key(op: &OpInfo, tensors: &[TensorInfo]) -> Option<String> {
    let names = op
        .inputs
        .iter()
        .chain(op.outputs.iter())
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .map(|tensor| tensor.name.to_ascii_lowercase());
    for name in names {
        for segment in name.split('/') {
            for prefix in [
                "expanded_conv",
                "patch_extraction",
                "aspp",
                "block_",
                "se_",
                "dense_",
            ] {
                if let Some(position) = segment.find(prefix) {
                    let suffix = &segment[position..];
                    return Some(trim_numeric_key(suffix, prefix));
                }
            }
            if segment.ends_with("_pad") {
                return Some(segment.to_string());
            }
        }
    }
    None
}

fn trim_numeric_key(segment: &str, prefix: &str) -> String {
    if matches!(prefix, "aspp" | "patch_extraction") {
        return prefix.to_string();
    }
    let mut end = prefix.len();
    for character in segment[prefix.len()..].chars() {
        if character.is_ascii_digit() || character == '_' {
            end += character.len_utf8();
        } else {
            break;
        }
    }
    segment[..end.max(prefix.len())]
        .trim_end_matches('_')
        .to_string()
}

fn named_block_type<'a>(key: &str, ops: &'a [OpInfo]) -> &'a str {
    let names = ops.iter().map(|op| op.name.as_str()).collect::<Vec<_>>();
    if key.starts_with("expanded_conv")
        && names
            .windows(3)
            .any(|window| window == ["CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D"])
    {
        "inverted_bottleneck"
    } else if key.starts_with("se_")
        && names.contains(&"MEAN")
        && names.iter().any(|name| matches!(*name, "MUL" | "ADD"))
    {
        "squeeze_excitation"
    } else if key.starts_with("aspp") {
        "aspp_named_component"
    } else if key.starts_with("dense_") {
        "fully_connected_group"
    } else if key.starts_with("patch_extraction") {
        "preprocessing"
    } else if key.ends_with("_pad") {
        "padding"
    } else {
        "named_group"
    }
}

fn humanize_name_key(key: &str) -> String {
    key.replace('_', " ")
        .split_whitespace()
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn op_spatial_key(op: &OpInfo) -> Option<String> {
    op.output_shapes.iter().find_map(|shape| {
        if shape.len() == 4 {
            Some(format!("{}x{}", shape[1], shape[2]))
        } else {
            None
        }
    })
}

fn humanize_spatial_key(key: &str) -> String {
    format!("Spatial group {key}")
}

fn first_activation_input<'a>(op: &OpInfo, tensors: &'a [TensorInfo]) -> Option<&'a TensorInfo> {
    op.inputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .find(|tensor| !tensor.constant_buffer)
}

fn first_output<'a>(op: &OpInfo, tensors: &'a [TensorInfo]) -> Option<&'a TensorInfo> {
    op.outputs
        .iter()
        .filter_map(|index| usize::try_from(*index).ok())
        .filter_map(|index| tensors.get(index))
        .next()
}

fn block_spatial(ops: &[&OpInfo], tensors: &[TensorInfo]) -> SpatialShape {
    let input = ops
        .first()
        .and_then(|op| first_activation_input(op, tensors));
    let output = ops.last().and_then(|op| first_output(op, tensors));
    SpatialShape {
        input_h: input
            .and_then(|tensor| tensor.shape.get(1).copied())
            .and_then(positive),
        input_w: input
            .and_then(|tensor| tensor.shape.get(2).copied())
            .and_then(positive),
        output_h: output
            .and_then(|tensor| tensor.shape.get(1).copied())
            .and_then(positive),
        output_w: output
            .and_then(|tensor| tensor.shape.get(2).copied())
            .and_then(positive),
    }
}

fn block_channels(ops: &[&OpInfo], tensors: &[TensorInfo]) -> BlockChannels {
    let input = ops
        .first()
        .and_then(|op| first_activation_input(op, tensors))
        .and_then(|tensor| tensor.shape.last().copied())
        .and_then(positive);
    let output = ops
        .last()
        .and_then(|op| first_output(op, tensors))
        .and_then(|tensor| tensor.shape.last().copied())
        .and_then(positive);
    let expand = ops
        .iter()
        .flat_map(|op| op.output_shapes.iter())
        .filter_map(|shape| shape.last().copied())
        .filter_map(positive)
        .max()
        .filter(|value| Some(*value) != input && Some(*value) != output);
    BlockChannels {
        input,
        expand,
        output,
    }
}

fn block_params(
    ops: &[&OpInfo],
    tensors: &[TensorInfo],
    spatial: SpatialShape,
    channels: BlockChannels,
) -> BlockParams {
    let mut kernel_h = None::<usize>;
    let mut kernel_w = None::<usize>;
    for op in ops {
        if !matches!(
            op.name.as_str(),
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "TRANSPOSE_CONV"
        ) {
            continue;
        }
        if let Some(weight) = op
            .inputs
            .get(1)
            .and_then(|index| usize::try_from(*index).ok())
            .and_then(|index| tensors.get(index))
        {
            kernel_h = weight.shape.get(1).copied().and_then(positive).or(kernel_h);
            kernel_w = weight.shape.get(2).copied().and_then(positive).or(kernel_w);
            if kernel_h != Some(1) || kernel_w != Some(1) {
                break;
            }
        }
    }
    let stride_h = spatial
        .input_h
        .zip(spatial.output_h)
        .map(|(input, output)| inferred_stride(input, output));
    let stride_w = spatial
        .input_w
        .zip(spatial.output_w)
        .map(|(input, output)| inferred_stride(input, output));
    BlockParams {
        expand_ratio: channels
            .input
            .zip(channels.expand)
            .filter(|(input, _)| *input > 0)
            .map(|(input, expand)| expand as f64 / input as f64),
        stride_h,
        stride_w,
        kernel_h,
        kernel_w,
    }
}

fn inferred_stride(input: usize, output: usize) -> usize {
    if output == 0 || input <= output {
        return 1;
    }
    (1usize..=8)
        .find(|stride| input.div_ceil(*stride) == output)
        .unwrap_or_else(|| input.div_ceil(output).max(1))
}

fn has_block_residual(ops: &[&OpInfo]) -> bool {
    let Some(first_input) = ops.first().and_then(|op| op.inputs.first()).copied() else {
        return false;
    };
    ops.iter().any(|op| {
        op.name == "ADD"
            && op.inputs.contains(&first_input)
            && op.inputs.iter().any(|input| *input != first_input)
    })
}

fn aggregate_ops(
    ops: &[&OpInfo],
    tensors: &[TensorInfo],
    target: &TargetProfile,
    total_macs: f64,
) -> BlockAggregates {
    let mut aggregates = BlockAggregates {
        time_evidence_class: "ESTIMATED_TARGET_PROFILE".to_string(),
        ..BlockAggregates::default()
    };
    let mut parameter_tensors = BTreeSet::<usize>::new();
    for op in ops {
        aggregates.op_count += 1;
        aggregates.macs += op.macs;
        aggregates.modeled_time_ms +=
            (op.bottleneck_total_us - op.bottleneck_packing_us - op.bottleneck_break_us).max(0.0)
                / 1000.0;
        aggregates.modeled_cold_start_time_ms += op.bottleneck_total_us / 1000.0;
        aggregates.logical_traffic_bytes = aggregates
            .logical_traffic_bytes
            .saturating_add(op.estimated_bytes.max(0.0).round() as usize);
        if op.xnnpack_chain_break {
            aggregates.predicted_break_count += 1;
        }
        match op.static_bound_guess.as_str() {
            "compute-bound" => aggregates.intensity.compute += 1,
            "mixed" => aggregates.intensity.mixed += 1,
            "memory-bound" => aggregates.intensity.memory += 1,
            _ => aggregates.intensity.not_assessed += 1,
        }
        if let Some(bytes) = op.cache_payload.logical_row_payload_bytes {
            aggregates.cache_assessed_op_count += 1;
            if target.l1_data_bytes > 0 {
                let ratio = bytes as f64 / target.l1_data_bytes as f64;
                if ratio >= L1_WORKING_SET_WATCH_RATIO {
                    aggregates.l1_watch_count += 1;
                }
                if aggregates
                    .l1_max_ratio
                    .map(|current| ratio > current)
                    .unwrap_or(true)
                {
                    aggregates.l1_max_ratio = Some(ratio);
                    aggregates.l1_max_op_index = Some(op.index);
                }
            }
        }
        let parameter_slots: &[usize] = match op.name.as_str() {
            "CONV_2D" | "DEPTHWISE_CONV_2D" | "FULLY_CONNECTED" => &[1, 2],
            "TRANSPOSE_CONV" => &[1, 3],
            "BATCH_MATMUL" => &[1],
            _ => &[],
        };
        for slot in parameter_slots {
            let Some(tensor_index) = op.inputs.get(*slot) else {
                continue;
            };
            let Some(index) = usize::try_from(*tensor_index).ok() else {
                continue;
            };
            let Some(tensor) = tensors.get(index) else {
                continue;
            };
            if tensor.constant_buffer {
                parameter_tensors.insert(index);
            }
        }
    }
    aggregates.mac_percent = if total_macs > 0.0 {
        aggregates.macs / total_macs
    } else {
        0.0
    };
    for tensor_index in parameter_tensors {
        let Some(tensor) = tensors.get(tensor_index) else {
            continue;
        };
        aggregates.parameter_elements = aggregates
            .parameter_elements
            .saturating_add(tensor_element_count(tensor).unwrap_or(0));
        aggregates.serialized_parameter_bytes = aggregates
            .serialized_parameter_bytes
            .saturating_add(tensor.buffer_data_length);
    }
    aggregates
}

fn tensor_element_count(tensor: &TensorInfo) -> Option<usize> {
    tensor.shape.iter().try_fold(1usize, |product, dimension| {
        product.checked_mul(positive(*dimension)?)
    })
}

fn group_blocks_into_stages(blocks: &[BlockRecord]) -> Vec<Vec<usize>> {
    let mut stages = Vec::<Vec<usize>>::new();
    let mut current_key = None::<(Option<usize>, Option<usize>)>;
    for (index, block) in blocks.iter().enumerate() {
        let key = if block.block_type == "spatial_prelude" {
            blocks[index + 1..]
                .iter()
                .find(|candidate| candidate.block_type != "spatial_prelude")
                .map(|candidate| (candidate.spatial.output_h, candidate.spatial.output_w))
                .unwrap_or((block.spatial.output_h, block.spatial.output_w))
        } else {
            (block.spatial.output_h, block.spatial.output_w)
        };
        if current_key != Some(key) {
            stages.push(Vec::new());
            current_key = Some(key);
        }
        stages.last_mut().expect("stage exists").push(index);
    }
    stages
}

fn merge_stage_spatial(spatials: impl Iterator<Item = SpatialShape>) -> SpatialShape {
    let values = spatials.collect::<Vec<_>>();
    SpatialShape {
        input_h: values.first().and_then(|value| value.input_h),
        input_w: values.first().and_then(|value| value.input_w),
        output_h: values.last().and_then(|value| value.output_h),
        output_w: values.last().and_then(|value| value.output_w),
    }
}

fn stage_display_name<'a>(types: impl Iterator<Item = &'a str>) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for block_type in types {
        *counts.entry(block_type.to_string()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(name, count)| {
            let label = name.replace('_', " ");
            if count > 1 {
                format!("{label} x{count}")
            } else {
                label
            }
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn repeat_for_block(index: usize, blocks: &[BlockRecord]) -> BlockRepeat {
    let block = &blocks[index];
    let peers = blocks
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.stage_index == block.stage_index && candidate.block_type == block.block_type
        })
        .map(|(peer_index, _)| peer_index)
        .collect::<Vec<_>>();
    let position = peers
        .iter()
        .position(|peer_index| *peer_index == index)
        .unwrap_or(0);
    BlockRepeat {
        index: position,
        total: peers.len().max(1),
    }
}

fn positive(value: i32) -> Option<usize> {
    usize::try_from(value).ok().filter(|value| *value > 0)
}
