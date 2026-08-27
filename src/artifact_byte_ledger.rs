use super::BufferDataLocation;
use crate::tflite_metadata::PackedMetadataArchive;
use serde::Serialize;
use std::collections::BTreeSet;

const MAX_TERMINAL_ZERO_ALIGNMENT_BYTES: usize = 16;
const MAX_METADATA_ARCHIVE_ENCODED_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Serialize)]
pub(super) struct ArtifactByteRange {
    pub(super) offset: usize,
    pub(super) end: usize,
    pub(super) length: usize,
    pub(super) class: String,
}

#[derive(Clone, Serialize)]
pub(super) struct ArtifactByteIntegrityLedger {
    pub(super) schema: String,
    pub(super) status: String,
    pub(super) evidence_class: String,
    pub(super) file_size: usize,
    pub(super) flatbuffer_referenced_range_count: usize,
    pub(super) flatbuffer_referenced_bytes: usize,
    pub(super) flatbuffer_referenced_end: usize,
    pub(super) flatbuffer_envelope_bytes: usize,
    pub(super) flatbuffer_internal_alignment_or_unreferenced_bytes: usize,
    pub(super) flatbuffer_referenced_ranges: Vec<ArtifactByteRange>,
    pub(super) terminal_zero_alignment_bytes: usize,
    pub(super) metadata_archive_status: String,
    pub(super) metadata_archive_start: Option<usize>,
    pub(super) metadata_archive_end: Option<usize>,
    pub(super) metadata_archive_central_directory_start: Option<usize>,
    pub(super) metadata_archive_central_directory_end: Option<usize>,
    pub(super) metadata_archive_eocd_offset: Option<usize>,
    pub(super) metadata_archive_bytes: usize,
    pub(super) metadata_archive_file_count: usize,
    pub(super) metadata_archive_case_insensitive_name_collision_count: usize,
    pub(super) metadata_archive_size_policy_bytes: usize,
    pub(super) metadata_archive_size_policy_exceeded: bool,
    pub(super) unowned_trailing_bytes: usize,
    pub(super) unowned_trailing_ranges: Vec<ArtifactByteRange>,
    pub(super) exact_shared_buffer_range_count: usize,
    pub(super) partial_buffer_overlap_count: usize,
    pub(super) flatbuffer_archive_overlap_bytes: usize,
    pub(super) classified_bytes: Option<usize>,
    pub(super) conservation_status: String,
    pub(super) issue_count: usize,
    pub(super) issues: Vec<String>,
    pub(super) method: String,
    pub(super) detail: String,
}

pub(super) fn build_artifact_byte_integrity_ledger(
    bytes: &[u8],
    referenced_ranges: &[BufferDataLocation],
    buffer_locations: &[BufferDataLocation],
    archive: &PackedMetadataArchive,
) -> ArtifactByteIntegrityLedger {
    let file_size = bytes.len();
    let referenced = normalize_ranges(referenced_ranges, file_size);
    let flatbuffer_referenced_bytes = referenced.iter().map(|range| range.length).sum::<usize>();
    let flatbuffer_referenced_end = referenced.last().map(|range| range.end).unwrap_or(0);
    let flatbuffer_internal_alignment_or_unreferenced_bytes =
        flatbuffer_referenced_end.saturating_sub(flatbuffer_referenced_bytes);

    let archive_range = archive
        .archive_start
        .zip(archive.archive_end)
        .filter(|(start, end)| *start <= *end && *end <= file_size);
    let archive_start = archive_range.map(|range| range.0);
    let metadata_archive_bytes = archive_range
        .map(|(start, end)| end.saturating_sub(start))
        .unwrap_or(0);
    let flatbuffer_archive_overlap_bytes = archive_start
        .map(|start| flatbuffer_referenced_end.saturating_sub(start))
        .unwrap_or(0);

    let trailing_limit = archive_start.unwrap_or(file_size);
    let trailing_start = flatbuffer_referenced_end.min(trailing_limit);
    let terminal_gap = bytes
        .get(trailing_start..trailing_limit)
        .unwrap_or_default();
    let terminal_zero_alignment_bytes = if terminal_gap.len() <= MAX_TERMINAL_ZERO_ALIGNMENT_BYTES
        && terminal_gap.iter().all(|byte| *byte == 0)
    {
        terminal_gap.len()
    } else {
        0
    };
    let unowned_trailing_bytes = terminal_gap
        .len()
        .saturating_sub(terminal_zero_alignment_bytes);
    let unowned_trailing_ranges = if unowned_trailing_bytes > 0 {
        vec![ArtifactByteRange {
            offset: trailing_start,
            end: trailing_limit,
            length: trailing_limit.saturating_sub(trailing_start),
            class: "unowned_trailing_bytes".to_string(),
        }]
    } else {
        Vec::new()
    };

    let (exact_shared_buffer_range_count, partial_buffer_overlap_count) =
        buffer_overlap_counts(buffer_locations, file_size);
    let metadata_archive_size_policy_exceeded =
        metadata_archive_bytes > MAX_METADATA_ARCHIVE_ENCODED_BYTES;
    let mut issues = Vec::new();
    if archive.status == "malformed" {
        issues.push(format!(
            "terminal_metadata_archive_malformed: {}",
            archive.detail
        ));
    }
    if flatbuffer_archive_overlap_bytes > 0 {
        issues.push(format!(
            "flatbuffer_metadata_archive_overlap: {} byte(s)",
            flatbuffer_archive_overlap_bytes
        ));
    }
    if unowned_trailing_bytes > 0 {
        issues.push(format!(
            "unowned_trailing_bytes: [{}..{}) = {} byte(s)",
            trailing_start, trailing_limit, unowned_trailing_bytes
        ));
    }
    if partial_buffer_overlap_count > 0 {
        issues.push(format!(
            "partial_buffer_range_overlap: {} pair(s)",
            partial_buffer_overlap_count
        ));
    }
    if archive.case_insensitive_name_collision_count > 0 {
        issues.push(format!(
            "case_insensitive_metadata_filename_collision: {} collision(s)",
            archive.case_insensitive_name_collision_count
        ));
    }
    if metadata_archive_size_policy_exceeded {
        issues.push(format!(
            "metadata_archive_encoded_size_limit_exceeded: {} > {} byte(s)",
            metadata_archive_bytes, MAX_METADATA_ARCHIVE_ENCODED_BYTES
        ));
    }

    let classified_bytes = if flatbuffer_archive_overlap_bytes == 0 {
        flatbuffer_referenced_end
            .checked_add(terminal_gap.len())
            .and_then(|value| value.checked_add(metadata_archive_bytes))
    } else {
        None
    };
    let conservation_status = match classified_bytes {
        Some(total) if total == file_size => "exact".to_string(),
        Some(_) => "failed_sum_mismatch".to_string(),
        None => "failed_overlapping_ownership".to_string(),
    };
    if conservation_status != "exact" {
        issues.push(format!("byte_conservation_{conservation_status}"));
    }
    let status = if issues.is_empty() {
        "assessed_clean"
    } else if flatbuffer_archive_overlap_bytes > 0
        || unowned_trailing_bytes > 0
        || partial_buffer_overlap_count > 0
        || archive.status == "malformed"
        || conservation_status != "exact"
    {
        "risk"
    } else {
        "warn"
    }
    .to_string();
    let detail = format!(
        "File {} B = FlatBuffer envelope {} B ({} B in {} merged referenced range(s), {} B internal alignment/unreferenced) + terminal zero-alignment candidate {} B + unowned trailing {} B + terminal metadata ZIP {} B. Conservation: {}. Buffer locations: {} exact shared range(s), {} partial overlap pair(s).",
        file_size,
        flatbuffer_referenced_end,
        flatbuffer_referenced_bytes,
        referenced.len(),
        flatbuffer_internal_alignment_or_unreferenced_bytes,
        terminal_zero_alignment_bytes,
        unowned_trailing_bytes,
        metadata_archive_bytes,
        conservation_status,
        exact_shared_buffer_range_count,
        partial_buffer_overlap_count,
    );

    ArtifactByteIntegrityLedger {
        schema: "deepbom.tflite_artifact_byte_integrity.v1".to_string(),
        status,
        evidence_class: "DERIVED".to_string(),
        file_size,
        flatbuffer_referenced_range_count: referenced.len(),
        flatbuffer_referenced_bytes,
        flatbuffer_referenced_end,
        flatbuffer_envelope_bytes: flatbuffer_referenced_end,
        flatbuffer_internal_alignment_or_unreferenced_bytes,
        flatbuffer_referenced_ranges: referenced,
        terminal_zero_alignment_bytes,
        metadata_archive_status: archive.status.clone(),
        metadata_archive_start: archive_range.map(|range| range.0),
        metadata_archive_end: archive_range.map(|range| range.1),
        metadata_archive_central_directory_start: archive.central_directory_start,
        metadata_archive_central_directory_end: archive.central_directory_end,
        metadata_archive_eocd_offset: archive.eocd_offset,
        metadata_archive_bytes,
        metadata_archive_file_count: archive.files.len(),
        metadata_archive_case_insensitive_name_collision_count: archive
            .case_insensitive_name_collision_count,
        metadata_archive_size_policy_bytes: MAX_METADATA_ARCHIVE_ENCODED_BYTES,
        metadata_archive_size_policy_exceeded,
        unowned_trailing_bytes,
        unowned_trailing_ranges,
        exact_shared_buffer_range_count,
        partial_buffer_overlap_count,
        flatbuffer_archive_overlap_bytes,
        classified_bytes,
        conservation_status,
        issue_count: issues.len(),
        issues,
        method: "Union every verified FlatBuffer table/vtable/vector/string/external-buffer range; take the greatest referenced end as the FlatBuffer envelope; parse a terminal bounded ZIP central directory and local records; classify only a <=16-byte all-zero terminal gap as an alignment candidate; classify every remaining gap before the ZIP or EOF as unowned trailing bytes; reject overlapping ownership; require exact file-size conservation.".to_string(),
        detail,
    }
}

fn normalize_ranges(ranges: &[BufferDataLocation], file_size: usize) -> Vec<ArtifactByteRange> {
    let mut normalized = ranges
        .iter()
        .filter_map(|range| {
            let end = range.offset.checked_add(range.length)?;
            (range.length > 0 && end <= file_size).then_some((range.offset, end))
        })
        .collect::<Vec<_>>();
    normalized.sort_unstable();
    let mut merged = Vec::<(usize, usize)>::new();
    for (start, end) in normalized {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
        .into_iter()
        .map(|(offset, end)| ArtifactByteRange {
            offset,
            end,
            length: end - offset,
            class: "verified_flatbuffer_reference".to_string(),
        })
        .collect()
}

fn buffer_overlap_counts(ranges: &[BufferDataLocation], file_size: usize) -> (usize, usize) {
    let mut unique = BTreeSet::<(usize, usize)>::new();
    let mut exact_shared = 0usize;
    for range in ranges {
        let Some(end) = range.offset.checked_add(range.length) else {
            continue;
        };
        if range.length == 0 || end > file_size {
            continue;
        }
        if !unique.insert((range.offset, end)) {
            exact_shared += 1;
        }
    }
    let unique = unique.into_iter().collect::<Vec<_>>();
    let mut partial = 0usize;
    for left in 0..unique.len() {
        for right in left + 1..unique.len() {
            if unique[right].0 >= unique[left].1 {
                break;
            }
            if unique[left].0 < unique[right].1 && unique[right].0 < unique[left].1 {
                partial += 1;
            }
        }
    }
    (exact_shared, partial)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive(status: &str, start: Option<usize>, end: Option<usize>) -> PackedMetadataArchive {
        PackedMetadataArchive {
            status: status.to_string(),
            files: Vec::new(),
            archive_start: start,
            archive_end: end,
            central_directory_start: start,
            central_directory_end: end,
            eocd_offset: end,
            case_insensitive_name_collision_count: 0,
            detail: status.to_string(),
        }
    }

    #[test]
    fn conserves_clean_flatbuffer_and_terminal_archive() {
        let bytes = vec![1u8; 24];
        let result = build_artifact_byte_integrity_ledger(
            &bytes,
            &[BufferDataLocation {
                offset: 0,
                length: 16,
            }],
            &[],
            &archive("assessed", Some(16), Some(24)),
        );
        assert_eq!(result.conservation_status, "exact");
        assert_eq!(result.unowned_trailing_bytes, 0);
        assert_eq!(result.metadata_archive_bytes, 8);
    }

    #[test]
    fn reports_exact_nonzero_trailing_range() {
        let mut bytes = vec![0u8; 20];
        bytes[16..].copy_from_slice(&[1, 2, 3, 4]);
        let result = build_artifact_byte_integrity_ledger(
            &bytes,
            &[BufferDataLocation {
                offset: 0,
                length: 16,
            }],
            &[],
            &archive("not_present", None, None),
        );
        assert_eq!(result.unowned_trailing_bytes, 4);
        assert_eq!(result.unowned_trailing_ranges[0].offset, 16);
        assert_eq!(result.unowned_trailing_ranges[0].end, 20);
        assert_eq!(result.status, "risk");
    }

    #[test]
    fn distinguishes_shared_ranges_from_partial_overlap() {
        let result = build_artifact_byte_integrity_ledger(
            &[0u8; 32],
            &[BufferDataLocation {
                offset: 0,
                length: 32,
            }],
            &[
                BufferDataLocation {
                    offset: 8,
                    length: 8,
                },
                BufferDataLocation {
                    offset: 8,
                    length: 8,
                },
                BufferDataLocation {
                    offset: 12,
                    length: 8,
                },
            ],
            &archive("not_present", None, None),
        );
        assert_eq!(result.exact_shared_buffer_range_count, 1);
        assert_eq!(result.partial_buffer_overlap_count, 1);
    }
}
