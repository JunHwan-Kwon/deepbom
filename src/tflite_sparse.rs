use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::verified_flatbuffer::Fb;
use crate::TensorInfo;

pub(super) const SOURCE_COMMIT: &str = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
pub(super) const SCHEMA_SOURCE: &str = "tensorflow/compiler/mlir/lite/schema/schema.fbs";
pub(super) const SCHEMA_SHA256: &str =
    "3bfa613428459de18db5d70d8581e7b6afd127c4522bb18ff59c8e589c3b75a1";
pub(super) const CONVERTER_SOURCE: &str =
    "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc";
pub(super) const CONVERTER_SHA256: &str =
    "2c032a4202d549a39c09978b4951d2200014b4429b061235b70f66da148a418c";

#[derive(Clone, Serialize)]
pub(super) struct SparseDimensionEvidence {
    pub level: usize,
    pub expanded_dimension: usize,
    pub format: &'static str,
    pub dense_size: usize,
    pub segment_index_type: &'static str,
    pub segment_count: usize,
    pub segment_terminal: usize,
    pub coordinate_index_type: &'static str,
    pub coordinate_count: usize,
    pub coordinate_min: Option<usize>,
    pub coordinate_max: Option<usize>,
    #[serde(skip)]
    segments: Vec<usize>,
    #[serde(skip)]
    indices: Vec<usize>,
}

#[derive(Clone, Serialize)]
pub(super) struct SparseTensorEncoding {
    pub schema: &'static str,
    pub status: &'static str,
    pub evidence_class: &'static str,
    pub traversal_order: Vec<i32>,
    pub block_map: Vec<i32>,
    pub dimensions: Vec<SparseDimensionEvidence>,
    pub logical_element_count: usize,
    pub stored_element_count: usize,
    pub implicit_zero_element_count: usize,
    pub stored_value_bytes: Option<usize>,
    pub expected_stored_value_bytes: Option<usize>,
    pub storage_width_bytes: Option<usize>,
    pub canonical_metadata_sha256: String,
    pub source_commit: &'static str,
    pub schema_source: &'static str,
    pub schema_source_sha256: &'static str,
    pub converter_source: &'static str,
    pub converter_source_sha256: &'static str,
    pub method: &'static str,
    pub interpretation_boundary: &'static str,
    #[serde(skip)]
    dense_linear_indices: Vec<usize>,
}

#[derive(Clone, Serialize)]
pub(super) struct SparseStorageRow {
    subgraph_index: usize,
    tensor_index: usize,
    tensor_name: String,
    dtype: String,
    shape: Vec<i32>,
    buffer_index: i32,
    constant_buffer: bool,
    encoding: SparseTensorEncoding,
}

#[derive(Clone, Serialize)]
pub(super) struct SparseStorageContract {
    schema: &'static str,
    status: &'static str,
    evidence_class: &'static str,
    sparse_tensor_count: usize,
    serialized_value_tensor_count: usize,
    fully_decoded_tensor_count: usize,
    partial_tensor_count: usize,
    logical_element_count: usize,
    stored_element_count: usize,
    implicit_zero_element_count: usize,
    serialized_value_bytes: usize,
    rows: Vec<SparseStorageRow>,
    source_commit: &'static str,
    schema_source: &'static str,
    schema_source_sha256: &'static str,
    converter_source: &'static str,
    converter_source_sha256: &'static str,
    method: &'static str,
    interpretation_boundary: &'static str,
}

pub(super) fn build_sparse_storage_contract(
    scoped_tensors: &[Vec<TensorInfo>],
) -> Result<SparseStorageContract, String> {
    let rows = scoped_tensors
        .iter()
        .enumerate()
        .flat_map(|(subgraph_index, tensors)| {
            tensors.iter().filter_map(move |tensor| {
                tensor
                    .sparse_encoding
                    .as_ref()
                    .cloned()
                    .map(|encoding| SparseStorageRow {
                        subgraph_index,
                        tensor_index: tensor.index,
                        tensor_name: tensor.name.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                        buffer_index: tensor.buffer_index,
                        constant_buffer: tensor.constant_buffer,
                        encoding,
                    })
            })
        })
        .collect::<Vec<_>>();
    let fully_decoded_tensor_count = rows
        .iter()
        .filter(|row| row.encoding.status == "assessed")
        .count();
    let serialized_value_tensor_count = rows
        .iter()
        .filter(|row| row.encoding.stored_value_bytes.is_some())
        .count();
    let partial_tensor_count = rows
        .iter()
        .filter(|row| row.encoding.status.starts_with("partial_"))
        .count();
    let status = if rows.is_empty() {
        "not_applicable"
    } else if partial_tensor_count > 0 {
        "partial"
    } else {
        "assessed"
    };
    let checked_sum = |selector: fn(&SparseStorageRow) -> usize, label: &str| {
        rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(selector(row))
                .ok_or_else(|| format!("TFLite sparse {label} total overflows"))
        })
    };
    Ok(SparseStorageContract {
        schema: "deepbom.tflite_sparse_storage_contract.v1",
        status,
        evidence_class: "DERIVED",
        sparse_tensor_count: rows.len(),
        serialized_value_tensor_count,
        fully_decoded_tensor_count,
        partial_tensor_count,
        logical_element_count: checked_sum(
            |row| row.encoding.logical_element_count,
            "logical-element",
        )?,
        stored_element_count: checked_sum(
            |row| row.encoding.stored_element_count,
            "stored-element",
        )?,
        implicit_zero_element_count: checked_sum(
            |row| row.encoding.implicit_zero_element_count,
            "implicit-zero",
        )?,
        serialized_value_bytes: checked_sum(
            |row| row.encoding.stored_value_bytes.unwrap_or(0),
            "serialized-byte",
        )?,
        rows,
        source_commit: SOURCE_COMMIT,
        schema_source: SCHEMA_SOURCE,
        schema_source_sha256: SCHEMA_SHA256,
        converter_source: CONVERTER_SOURCE,
        converter_source_sha256: CONVERTER_SHA256,
        method: "Every serialized SparsityParameters record is validated and expanded through one shared source-bound decoder before any numerical consumer sees a logical dense value stream.",
        interpretation_boundary: "Storage reconstruction does not establish that the selected runtime retained sparse execution, selected a sparse kernel, or avoided densification.",
    })
}

impl SparseTensorEncoding {
    pub(super) fn densify(&self, raw: &[u8]) -> Option<Vec<u8>> {
        let width = self.storage_width_bytes?;
        if raw.len() != self.expected_stored_value_bytes? {
            return None;
        }
        let dense_bytes = self.logical_element_count.checked_mul(width)?;
        let mut dense = vec![0u8; dense_bytes];
        for (source_index, &dense_index) in self.dense_linear_indices.iter().enumerate() {
            let source = source_index.checked_mul(width)?;
            let target = dense_index.checked_mul(width)?;
            dense
                .get_mut(target..target + width)?
                .copy_from_slice(raw.get(source..source + width)?);
        }
        Some(dense)
    }
}

pub(super) fn parse_sparsity(
    fb: &Fb<'_>,
    table: usize,
    shape: &[i32],
    storage_width_bytes: Option<usize>,
    stored_value_bytes: Option<usize>,
) -> Result<SparseTensorEncoding, String> {
    let traversal_order = fb.checked_vector_i32(table, 0, "SparsityParameters.traversal_order")?;
    let block_map = fb.checked_vector_i32(table, 1, "SparsityParameters.block_map")?;
    let dimension_tables = fb.vector_tables(table, 2);
    let mut dimensions = Vec::with_capacity(dimension_tables.len());
    for (level, dimension) in dimension_tables.into_iter().enumerate() {
        let format = fb.checked_i8_field(dimension, 0, 0, "DimensionMetadata.format")?;
        let dense_size = nonnegative(
            fb.checked_i32_field(dimension, 1, 0, "DimensionMetadata.dense_size")?,
            "DimensionMetadata.dense_size",
        )?;
        let segment_type =
            fb.checked_i8_field(dimension, 2, 0, "DimensionMetadata.array_segments_type")?;
        let segment_table =
            fb.checked_table_field(dimension, 3, "DimensionMetadata.array_segments")?;
        let coordinate_type =
            fb.checked_i8_field(dimension, 4, 0, "DimensionMetadata.array_indices_type")?;
        let coordinate_table =
            fb.checked_table_field(dimension, 5, "DimensionMetadata.array_indices")?;
        let (segment_index_type, segments) = read_index_vector(
            fb,
            segment_type,
            segment_table,
            "DimensionMetadata.array_segments",
        )?;
        let (coordinate_index_type, indices) = read_index_vector(
            fb,
            coordinate_type,
            coordinate_table,
            "DimensionMetadata.array_indices",
        )?;
        let format = match format {
            0 => "DENSE",
            1 => "SPARSE_CSR",
            other => {
                return Err(format!(
                    "Unsupported TFLite sparse dimension format {other}"
                ))
            }
        };
        if format == "DENSE" {
            if dense_size == 0 {
                return Err(format!("Sparse level {level} has a zero DENSE size"));
            }
            if !segments.is_empty() || !indices.is_empty() {
                return Err(format!(
                    "Sparse level {level} is DENSE but carries CSR arrays"
                ));
            }
        } else {
            if dense_size != 0 {
                return Err(format!(
                    "Sparse level {level} is SPARSE_CSR but dense_size is nonzero"
                ));
            }
            validate_csr_vectors(level, &segments, &indices)?;
        }
        dimensions.push(SparseDimensionEvidence {
            level,
            expanded_dimension: 0,
            format,
            dense_size,
            segment_index_type,
            segment_count: segments.len(),
            segment_terminal: segments.last().copied().unwrap_or(0),
            coordinate_index_type,
            coordinate_count: indices.len(),
            coordinate_min: indices.iter().copied().min(),
            coordinate_max: indices.iter().copied().max(),
            segments,
            indices,
        });
    }
    let logical_element_count = shape.iter().try_fold(1usize, |total, &dimension| {
        let dimension = usize::try_from(dimension)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                "Sparse tensor shape must contain only positive dimensions".to_string()
            })?;
        total
            .checked_mul(dimension)
            .ok_or_else(|| "Sparse tensor logical element count overflows".to_string())
    })?;
    let dense_linear_indices =
        validate_and_expand(shape, &traversal_order, &block_map, &mut dimensions)?;
    let stored_element_count = dense_linear_indices.len();
    let expected_stored_value_bytes = storage_width_bytes
        .map(|width| {
            stored_element_count
                .checked_mul(width)
                .ok_or_else(|| "Sparse stored-value byte count overflows".to_string())
        })
        .transpose()?;
    if expected_stored_value_bytes
        .zip(stored_value_bytes)
        .is_some_and(|(expected, stored)| expected != stored)
    {
        return Err(format!(
            "TFLite sparse value buffer has {} byte(s), but the validated leaf count requires {}",
            stored_value_bytes.unwrap(),
            expected_stored_value_bytes.unwrap()
        ));
    }
    let canonical_metadata_sha256 = metadata_digest(
        shape,
        &traversal_order,
        &block_map,
        &dimensions,
        &dense_linear_indices,
    );
    Ok(SparseTensorEncoding {
        schema: "deepbom.tflite_sparse_storage.v1",
        status: match (storage_width_bytes, stored_value_bytes) {
            (Some(_), Some(_)) => "assessed",
            (Some(_), None) => "assessed_metadata_runtime_values_external",
            (None, _) => "partial_unsupported_value_encoding",
        },
        evidence_class: "DERIVED",
        traversal_order,
        block_map,
        dimensions,
        logical_element_count,
        stored_element_count,
        implicit_zero_element_count: logical_element_count - stored_element_count,
        stored_value_bytes,
        expected_stored_value_bytes,
        storage_width_bytes,
        canonical_metadata_sha256,
        source_commit: SOURCE_COMMIT,
        schema_source: SCHEMA_SOURCE,
        schema_source_sha256: SCHEMA_SHA256,
        converter_source: CONVERTER_SOURCE,
        converter_source_sha256: CONVERTER_SHA256,
        method: "Validate traversal/block permutations and every CSR segment/index vector, reproduce the pinned TFLite sparse-to-dense traversal, reject duplicate or out-of-range dense coordinates, and bind the resulting leaf count to the exact stored value bytes.",
        interpretation_boundary: "Implicit sparse entries reproduce the pinned converter's stored scalar code/value zero. This is an artifact storage reconstruction, not evidence that a selected runtime retained sparse execution or selected a sparse microkernel.",
        dense_linear_indices,
    })
}

fn read_index_vector(
    fb: &Fb<'_>,
    vector_type: i8,
    table: Option<usize>,
    label: &str,
) -> Result<(&'static str, Vec<usize>), String> {
    let Some(table) = table else {
        return if vector_type == 0 {
            Ok(("NONE", Vec::new()))
        } else {
            Err(format!("{label} type {vector_type} has no table"))
        };
    };
    let (name, stride) = match vector_type {
        1 => ("INT32", 4),
        2 => ("UINT16", 2),
        3 => ("UINT8", 1),
        other => return Err(format!("{label} type {other} is unsupported")),
    };
    let (start, count) = fb
        .vector_location(table, 0, stride)
        .ok_or_else(|| format!("{label} values are missing or out of bounds"))?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let position = start + index * stride;
        let value = match vector_type {
            1 => nonnegative(
                fb.i32(position)
                    .ok_or_else(|| format!("{label}[{index}] is truncated"))?,
                label,
            )?,
            2 => usize::from(
                fb.u16(position)
                    .ok_or_else(|| format!("{label}[{index}] is truncated"))?,
            ),
            3 => usize::from(
                fb.u8(position)
                    .ok_or_else(|| format!("{label}[{index}] is truncated"))?,
            ),
            _ => unreachable!(),
        };
        values.push(value);
    }
    Ok((name, values))
}

fn nonnegative(value: i32, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} contains negative value {value}"))
}

fn validate_csr_vectors(level: usize, segments: &[usize], indices: &[usize]) -> Result<(), String> {
    if segments.is_empty() || segments[0] != 0 {
        return Err(format!(
            "Sparse level {level} CSR segments must begin with zero"
        ));
    }
    if segments.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(format!(
            "Sparse level {level} CSR segments are not monotonic"
        ));
    }
    if segments.last().copied() != Some(indices.len()) {
        return Err(format!(
            "Sparse level {level} CSR terminal segment does not equal its index count"
        ));
    }
    for range in segments.windows(2) {
        if indices[range[0]..range[1]]
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(format!(
                "Sparse level {level} CSR coordinates are duplicated or unsorted within a segment"
            ));
        }
    }
    Ok(())
}

fn validate_and_expand(
    shape: &[i32],
    traversal_order: &[i32],
    block_map: &[i32],
    dimensions: &mut [SparseDimensionEvidence],
) -> Result<Vec<usize>, String> {
    let rank = shape.len();
    let block_rank = block_map.len();
    let expanded_rank = rank
        .checked_add(block_rank)
        .ok_or_else(|| "Sparse expanded rank overflows".to_string())?;
    if traversal_order.len() != expanded_rank || dimensions.len() != expanded_rank {
        return Err(format!(
            "Sparse traversal/dimension count must be rank + block rank ({expanded_rank})"
        ));
    }
    validate_permutation(&traversal_order[..rank], 0, rank, "original traversal")?;
    validate_permutation(
        &traversal_order[rank..],
        rank,
        expanded_rank,
        "block traversal",
    )?;
    let block_map = block_map
        .iter()
        .map(|&value| nonnegative(value, "SparsityParameters.block_map"))
        .collect::<Result<Vec<_>, _>>()?;
    if block_map.iter().any(|&value| value >= rank)
        || block_map.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(
            "Sparse block_map must be strictly increasing original-dimension indices".to_string(),
        );
    }
    let traversal = traversal_order
        .iter()
        .map(|&value| nonnegative(value, "SparsityParameters.traversal_order"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut block_sizes = vec![1usize; block_rank];
    for block in 0..block_rank {
        let dimension_id = rank + block;
        let level = traversal
            .iter()
            .position(|&value| value == dimension_id)
            .ok_or_else(|| format!("Sparse block dimension {dimension_id} is not traversed"))?;
        let metadata = &dimensions[level];
        if metadata.format != "DENSE" {
            return Err(format!(
                "Sparse block dimension {dimension_id} is not DENSE"
            ));
        }
        block_sizes[block] = metadata.dense_size;
        let original = block_map[block];
        let original_size = nonnegative(shape[original], "Sparse tensor shape")?;
        if original_size == 0 || original_size % block_sizes[block] != 0 {
            return Err(format!(
                "Sparse block size {} does not divide shape dimension {} of size {}",
                block_sizes[block], original, original_size
            ));
        }
    }
    let mut expanded_sizes = shape
        .iter()
        .map(|&value| nonnegative(value, "Sparse tensor shape"))
        .collect::<Result<Vec<_>, _>>()?;
    for (block, &original) in block_map.iter().enumerate() {
        expanded_sizes[original] /= block_sizes[block];
        expanded_sizes.push(block_sizes[block]);
    }
    for (level, metadata) in dimensions.iter_mut().enumerate() {
        let dimension = traversal[level];
        metadata.expanded_dimension = dimension;
        let expected_size = expanded_sizes[dimension];
        if metadata.format == "DENSE" && metadata.dense_size != expected_size {
            return Err(format!(
                "Sparse level {level} DENSE size {} does not match expanded dimension {dimension} size {expected_size}",
                metadata.dense_size
            ));
        }
        if metadata.format == "SPARSE_CSR"
            && metadata.indices.iter().any(|&value| value >= expected_size)
        {
            return Err(format!(
                "Sparse level {level} contains a coordinate outside expanded dimension {dimension} size {expected_size}"
            ));
        }
    }
    let dense_shape = shape
        .iter()
        .map(|&value| nonnegative(value, "Sparse tensor shape"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut state = ExpansionState {
        rank,
        traversal: &traversal,
        block_map: &block_map,
        block_sizes: &block_sizes,
        dense_shape: &dense_shape,
        dimensions,
        coordinates: vec![0; expanded_rank],
        parents_seen: vec![0; expanded_rank],
        dense_indices: Vec::new(),
    };
    state.walk(0, 0)?;
    for (level, dimension) in state.dimensions.iter().enumerate() {
        if dimension.format == "SPARSE_CSR"
            && dimension.segment_count != state.parents_seen[level].saturating_add(1)
        {
            return Err(format!(
                "Sparse level {level} has {} segment entries for {} reached parent coordinate(s)",
                dimension.segment_count, state.parents_seen[level]
            ));
        }
    }
    let mut unique = state.dense_indices.clone();
    unique.sort_unstable();
    if unique.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(
            "Sparse metadata maps multiple stored values to one dense coordinate".to_string(),
        );
    }
    Ok(state.dense_indices)
}

struct ExpansionState<'a> {
    rank: usize,
    traversal: &'a [usize],
    block_map: &'a [usize],
    block_sizes: &'a [usize],
    dense_shape: &'a [usize],
    dimensions: &'a [SparseDimensionEvidence],
    coordinates: Vec<usize>,
    parents_seen: Vec<usize>,
    dense_indices: Vec<usize>,
}

impl ExpansionState<'_> {
    fn walk(&mut self, level: usize, parent: usize) -> Result<(), String> {
        if level == self.dimensions.len() {
            let mut original = vec![0usize; self.rank];
            for level in 0..self.rank {
                original[self.traversal[level]] = self.coordinates[level];
            }
            for level in self.rank..self.coordinates.len() {
                let block = self.traversal[level] - self.rank;
                let dimension = self.block_map[block];
                original[dimension] = original[dimension]
                    .checked_mul(self.block_sizes[block])
                    .and_then(|value| value.checked_add(self.coordinates[level]))
                    .ok_or_else(|| "Sparse block coordinate overflows".to_string())?;
            }
            if original
                .iter()
                .zip(self.dense_shape)
                .any(|(coordinate, size)| coordinate >= size)
            {
                return Err("Sparse metadata produces an out-of-range dense coordinate".to_string());
            }
            let linear = original.iter().zip(self.dense_shape).try_fold(
                0usize,
                |linear, (&coordinate, &size)| {
                    linear
                        .checked_mul(size)
                        .and_then(|value| value.checked_add(coordinate))
                        .ok_or_else(|| "Sparse dense linear index overflows".to_string())
                },
            )?;
            self.dense_indices.push(linear);
            return Ok(());
        }
        self.parents_seen[level] = self.parents_seen[level]
            .checked_add(1)
            .ok_or_else(|| "Sparse parent count overflows".to_string())?;
        let dimension = &self.dimensions[level];
        if dimension.format == "DENSE" {
            for coordinate in 0..dimension.dense_size {
                self.coordinates[level] = coordinate;
                let next_parent = parent
                    .checked_mul(dimension.dense_size)
                    .and_then(|value| value.checked_add(coordinate))
                    .ok_or_else(|| "Sparse dense parent index overflows".to_string())?;
                self.walk(level + 1, next_parent)?;
            }
        } else {
            let end_slot = parent
                .checked_add(1)
                .ok_or_else(|| "Sparse CSR parent index overflows".to_string())?;
            let (&start, &end) = dimension
                .segments
                .get(parent)
                .zip(dimension.segments.get(end_slot))
                .ok_or_else(|| {
                    format!("Sparse level {level} has no CSR segment for parent {parent}")
                })?;
            for index in start..end {
                self.coordinates[level] = *dimension.indices.get(index).ok_or_else(|| {
                    format!("Sparse level {level} CSR segment references missing index {index}")
                })?;
                self.walk(level + 1, index)?;
            }
        }
        Ok(())
    }
}

fn validate_permutation(
    values: &[i32],
    start: usize,
    end: usize,
    label: &str,
) -> Result<(), String> {
    let mut normalized = values
        .iter()
        .map(|&value| nonnegative(value, label))
        .collect::<Result<Vec<_>, _>>()?;
    normalized.sort_unstable();
    if normalized != (start..end).collect::<Vec<_>>() {
        return Err(format!(
            "Sparse {label} is not a permutation of [{start}, {end})"
        ));
    }
    Ok(())
}

fn metadata_digest(
    shape: &[i32],
    traversal: &[i32],
    block_map: &[i32],
    dimensions: &[SparseDimensionEvidence],
    dense_indices: &[usize],
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"deepbom.tflite_sparse_storage.v1\0");
    update_i32s(&mut hash, shape);
    update_i32s(&mut hash, traversal);
    update_i32s(&mut hash, block_map);
    for dimension in dimensions {
        hash.update([u8::from(dimension.format == "SPARSE_CSR")]);
        update_usizes(&mut hash, &[dimension.dense_size]);
        update_usizes(&mut hash, &dimension.segments);
        update_usizes(&mut hash, &dimension.indices);
    }
    update_usizes(&mut hash, dense_indices);
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn update_i32s(hash: &mut Sha256, values: &[i32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
}

fn update_usizes(hash: &mut Sha256, values: &[usize]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update((*value as u64).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(level: usize, size: usize) -> SparseDimensionEvidence {
        SparseDimensionEvidence {
            level,
            expanded_dimension: 0,
            format: "DENSE",
            dense_size: size,
            segment_index_type: "NONE",
            segment_count: 0,
            segment_terminal: 0,
            coordinate_index_type: "NONE",
            coordinate_count: 0,
            coordinate_min: None,
            coordinate_max: None,
            segments: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn csr(level: usize, segments: &[usize], indices: &[usize]) -> SparseDimensionEvidence {
        SparseDimensionEvidence {
            level,
            expanded_dimension: 0,
            format: "SPARSE_CSR",
            dense_size: 0,
            segment_index_type: "INT32",
            segment_count: segments.len(),
            segment_terminal: segments.last().copied().unwrap_or(0),
            coordinate_index_type: "INT32",
            coordinate_count: indices.len(),
            coordinate_min: indices.iter().copied().min(),
            coordinate_max: indices.iter().copied().max(),
            segments: segments.to_vec(),
            indices: indices.to_vec(),
        }
    }

    #[test]
    fn csr_coordinates_reconstruct_row_major_dense_indices() {
        let mut dimensions = vec![dense(0, 2), csr(1, &[0, 2, 3], &[0, 2, 1])];
        let indices = validate_and_expand(&[2, 3], &[0, 1], &[], &mut dimensions).unwrap();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn block_sparse_coordinates_reconstruct_original_shape() {
        let mut dimensions = vec![csr(0, &[0, 1], &[1]), dense(1, 2), dense(2, 2)];
        let indices = validate_and_expand(&[4, 2], &[0, 1, 2], &[0], &mut dimensions).unwrap();
        assert_eq!(indices, vec![4, 6, 5, 7]);
    }

    #[test]
    fn malformed_sparse_metadata_fails_closed() {
        let mut duplicate = vec![dense(0, 1), csr(1, &[0, 2], &[1, 1])];
        assert!(validate_and_expand(&[1, 3], &[0, 1], &[], &mut duplicate).is_err());
        assert!(validate_csr_vectors(1, &[1, 1], &[]).is_err());
        let mut out_of_range = vec![dense(0, 1), csr(1, &[0, 1], &[3])];
        assert!(validate_and_expand(&[1, 3], &[0, 1], &[], &mut out_of_range).is_err());
    }

    #[test]
    fn densification_copies_stored_values_and_preserves_implicit_zeroes() {
        let encoding = SparseTensorEncoding {
            schema: "deepbom.tflite_sparse_storage.v1",
            status: "assessed",
            evidence_class: "DERIVED",
            traversal_order: vec![0, 1],
            block_map: vec![],
            dimensions: vec![],
            logical_element_count: 6,
            stored_element_count: 3,
            implicit_zero_element_count: 3,
            stored_value_bytes: Some(3),
            expected_stored_value_bytes: Some(3),
            storage_width_bytes: Some(1),
            canonical_metadata_sha256: String::new(),
            source_commit: SOURCE_COMMIT,
            schema_source: SCHEMA_SOURCE,
            schema_source_sha256: SCHEMA_SHA256,
            converter_source: CONVERTER_SOURCE,
            converter_source_sha256: CONVERTER_SHA256,
            method: "test",
            interpretation_boundary: "test",
            dense_linear_indices: vec![0, 2, 4],
        };
        assert_eq!(
            encoding.densify(&[7, 8, 9]).unwrap(),
            vec![7, 0, 8, 0, 9, 0]
        );
    }
}
