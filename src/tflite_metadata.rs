use miniz_oxide::inflate::decompress_to_vec_with_limit;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const METADATA_IDENTIFIER: &[u8; 4] = b"M001";
const MAX_ASSOCIATED_FILE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ASSOCIATED_ARCHIVE_DECODED_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Serialize)]
pub(crate) struct MetadataProcessUnit {
    pub(crate) scope: String,
    pub(crate) input_ordinal: Option<usize>,
    pub(crate) tensor_name: String,
    pub(crate) options_type: String,
    pub(crate) options_type_code: u8,
    pub(crate) status: String,
    pub(crate) mean: Vec<f32>,
    pub(crate) std: Vec<f32>,
    pub(crate) associated_files: Vec<String>,
    pub(crate) detail: String,
}

#[derive(Clone, Serialize)]
pub(crate) struct MetadataAssociatedFile {
    pub(crate) output_ordinal: usize,
    pub(crate) tensor_name: String,
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) file_type: String,
    pub(crate) file_type_code: u8,
    pub(crate) locale: String,
    pub(crate) version: String,
    pub(crate) packed_status: String,
    pub(crate) payload_status: String,
    pub(crate) payload_sha256: String,
    pub(crate) payload_bytes: Option<usize>,
    pub(crate) crc32_verified: Option<bool>,
    pub(crate) text_encoding_status: String,
    pub(crate) label_entry_count: Option<usize>,
    pub(crate) blank_label_entry_count: Option<usize>,
    pub(crate) output_shape: Vec<i32>,
    pub(crate) cardinality_status: String,
    pub(crate) matching_output_axes: Vec<usize>,
    pub(crate) validation_detail: String,
}

#[derive(Clone, Serialize)]
pub(crate) struct PackedMetadataFile {
    pub(crate) name: String,
    pub(crate) compressed_bytes: u32,
    pub(crate) uncompressed_bytes: u32,
    pub(crate) compression_method: u16,
    pub(crate) crc32: u32,
    pub(crate) local_header_offset: u32,
    pub(crate) payload_status: String,
    pub(crate) payload_sha256: String,
    pub(crate) decoded_bytes: Option<usize>,
    pub(crate) crc32_verified: Option<bool>,
    pub(crate) detail: String,
    #[serde(skip_serializing)]
    payload: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct PackedMetadataArchive {
    pub(crate) status: String,
    pub(crate) files: Vec<PackedMetadataFile>,
    pub(crate) archive_start: Option<usize>,
    pub(crate) archive_end: Option<usize>,
    pub(crate) central_directory_start: Option<usize>,
    pub(crate) central_directory_end: Option<usize>,
    pub(crate) eocd_offset: Option<usize>,
    pub(crate) case_insensitive_name_collision_count: usize,
    pub(crate) detail: String,
}

pub(crate) struct BoundMetadataAssociatedFiles {
    pub(crate) archive_status: String,
    pub(crate) archive_detail: String,
    pub(crate) packed_files: Vec<PackedMetadataFile>,
    pub(crate) verified_output_associated_file_count: usize,
    pub(crate) missing_output_associated_file_count: usize,
    pub(crate) verified_output_label_file_count: usize,
    pub(crate) missing_output_label_file_count: usize,
    pub(crate) invalid_output_label_file_count: usize,
    pub(crate) verified_output0_label_file_count: usize,
    pub(crate) payload_verified_file_count: usize,
    pub(crate) payload_invalid_file_count: usize,
    pub(crate) payload_unsupported_file_count: usize,
    pub(crate) label_cardinality_match_count: usize,
    pub(crate) label_cardinality_mismatch_count: usize,
    pub(crate) label_cardinality_ambiguous_count: usize,
    pub(crate) label_cardinality_unresolved_count: usize,
}

#[derive(Clone)]
pub(crate) struct ParsedTfliteModelMetadata {
    pub(crate) status: String,
    pub(crate) schema_identifier: String,
    pub(crate) min_parser_version: String,
    pub(crate) model_name: String,
    pub(crate) model_description: String,
    pub(crate) model_version: String,
    pub(crate) author: String,
    pub(crate) license: String,
    pub(crate) subgraph_metadata_count: usize,
    pub(crate) input_tensor_metadata_count: usize,
    pub(crate) output_tensor_metadata_count: usize,
    pub(crate) described_input_tensor_count: usize,
    pub(crate) described_output_tensor_count: usize,
    pub(crate) input_process_units: Vec<MetadataProcessUnit>,
    pub(crate) recognized_input_process_unit_count: usize,
    pub(crate) invalid_input_process_unit_count: usize,
    pub(crate) unrecognized_input_process_unit_count: usize,
    pub(crate) normalization_unit_count: usize,
    pub(crate) output_associated_files: Vec<MetadataAssociatedFile>,
    pub(crate) output_label_file_count: usize,
    pub(crate) detail: String,
}

impl ParsedTfliteModelMetadata {
    pub(crate) fn unavailable(status: &str, detail: &str) -> Self {
        Self {
            status: status.to_string(),
            schema_identifier: String::new(),
            min_parser_version: String::new(),
            model_name: String::new(),
            model_description: String::new(),
            model_version: String::new(),
            author: String::new(),
            license: String::new(),
            subgraph_metadata_count: 0,
            input_tensor_metadata_count: 0,
            output_tensor_metadata_count: 0,
            described_input_tensor_count: 0,
            described_output_tensor_count: 0,
            input_process_units: Vec::new(),
            recognized_input_process_unit_count: 0,
            invalid_input_process_unit_count: 0,
            unrecognized_input_process_unit_count: 0,
            normalization_unit_count: 0,
            output_associated_files: Vec::new(),
            output_label_file_count: 0,
            detail: detail.to_string(),
        }
    }
}

pub(crate) fn parse_tflite_model_metadata(bytes: &[u8]) -> ParsedTfliteModelMetadata {
    if bytes.len() < 8 {
        return ParsedTfliteModelMetadata::unavailable(
            "malformed_metadata_buffer",
            "TFLite Model Metadata buffer is shorter than the FlatBuffer header.",
        );
    }
    let identifier = String::from_utf8_lossy(&bytes[4..8]).to_string();
    if &bytes[4..8] != METADATA_IDENTIFIER {
        let mut result = ParsedTfliteModelMetadata::unavailable(
            "unsupported_metadata_identifier",
            "TFLite Model Metadata buffer does not carry the supported M001 identifier.",
        );
        result.schema_identifier = identifier;
        return result;
    }
    let fb = MetadataFb::new(bytes);
    let Some(root) = fb.root_table() else {
        let mut result = ParsedTfliteModelMetadata::unavailable(
            "malformed_metadata_buffer",
            "TFLite Model Metadata root table is outside the metadata buffer.",
        );
        result.schema_identifier = identifier;
        return result;
    };
    if let Err(error) = fb.require_bounded_metadata(root) {
        let mut result = ParsedTfliteModelMetadata::unavailable(
            "malformed_metadata_buffer",
            &format!("TFLite Model Metadata is truncated or corrupt: {error}."),
        );
        result.schema_identifier = identifier;
        return result;
    }

    let subgraphs = fb.vector_tables(root, 3);
    let mut input_process_units = Vec::new();
    let mut output_associated_files = Vec::new();
    let mut input_tensor_metadata_count = 0usize;
    let mut output_tensor_metadata_count = 0usize;
    let mut described_input_tensor_count = 0usize;
    let mut described_output_tensor_count = 0usize;

    if let Some(&main) = subgraphs.first() {
        let input_metadata = fb.vector_tables(main, 2);
        let output_metadata = fb.vector_tables(main, 3);
        input_tensor_metadata_count = input_metadata.len();
        output_tensor_metadata_count = output_metadata.len();

        for (ordinal, tensor) in input_metadata.iter().copied().enumerate() {
            let name = fb.string_field(tensor, 0).unwrap_or_default();
            let description = fb.string_field(tensor, 1).unwrap_or_default();
            if !name.is_empty() || !description.is_empty() {
                described_input_tensor_count += 1;
            }
            for process_unit in fb.vector_tables(tensor, 4) {
                input_process_units.push(parse_process_unit(
                    &fb,
                    process_unit,
                    "input_tensor",
                    Some(ordinal),
                    &name,
                ));
            }
        }
        for process_unit in fb.vector_tables(main, 5) {
            input_process_units.push(parse_process_unit(
                &fb,
                process_unit,
                "subgraph_input",
                None,
                "",
            ));
        }

        for (ordinal, tensor) in output_metadata.iter().copied().enumerate() {
            let name = fb.string_field(tensor, 0).unwrap_or_default();
            let description = fb.string_field(tensor, 1).unwrap_or_default();
            if !name.is_empty() || !description.is_empty() {
                described_output_tensor_count += 1;
            }
            for associated_file in fb.vector_tables(tensor, 6) {
                output_associated_files.push(parse_associated_file(
                    &fb,
                    associated_file,
                    ordinal,
                    &name,
                ));
            }
        }
    }

    let recognized_input_process_unit_count = input_process_units
        .iter()
        .filter(|unit| unit.status == "assessed")
        .count();
    let invalid_input_process_unit_count = input_process_units
        .iter()
        .filter(|unit| unit.status == "invalid_options")
        .count();
    let unrecognized_input_process_unit_count = input_process_units
        .iter()
        .filter(|unit| unit.status == "unsupported_options_type")
        .count();
    let normalization_unit_count = input_process_units
        .iter()
        .filter(|unit| unit.options_type == "NormalizationOptions")
        .count();
    let output_label_file_count = output_associated_files
        .iter()
        .filter(|file| matches!(file.file_type_code, 2 | 3))
        .count();
    let detail = format!(
        "M001 metadata parsed: {} subgraph metadata record(s); main subgraph input/output tensor metadata {}/{}; explicit input process units {} ({} assessed, {} invalid, {} unsupported), including {} normalization unit(s); output associated files {} ({} label mapping declaration(s)).",
        subgraphs.len(),
        input_tensor_metadata_count,
        output_tensor_metadata_count,
        input_process_units.len(),
        recognized_input_process_unit_count,
        invalid_input_process_unit_count,
        unrecognized_input_process_unit_count,
        normalization_unit_count,
        output_associated_files.len(),
        output_label_file_count,
    );

    ParsedTfliteModelMetadata {
        status: "parsed".to_string(),
        schema_identifier: identifier,
        min_parser_version: fb.string_field(root, 7).unwrap_or_default(),
        model_name: fb.string_field(root, 0).unwrap_or_default(),
        model_description: fb.string_field(root, 1).unwrap_or_default(),
        model_version: fb.string_field(root, 2).unwrap_or_default(),
        author: fb.string_field(root, 4).unwrap_or_default(),
        license: fb.string_field(root, 5).unwrap_or_default(),
        subgraph_metadata_count: subgraphs.len(),
        input_tensor_metadata_count,
        output_tensor_metadata_count,
        described_input_tensor_count,
        described_output_tensor_count,
        input_process_units,
        recognized_input_process_unit_count,
        invalid_input_process_unit_count,
        unrecognized_input_process_unit_count,
        normalization_unit_count,
        output_associated_files,
        output_label_file_count,
        detail,
    }
}

fn parse_process_unit(
    fb: &MetadataFb<'_>,
    table: usize,
    scope: &str,
    input_ordinal: Option<usize>,
    tensor_name: &str,
) -> MetadataProcessUnit {
    let options_type_code = fb
        .field_pos(table, 0)
        .and_then(|position| fb.u8(position))
        .unwrap_or(0);
    let options_type = process_unit_type_name(options_type_code).to_string();
    let options = fb.table_field(table, 1);
    let mut mean = Vec::new();
    let mut std = Vec::new();
    let mut associated_files = Vec::new();
    let (valid, detail) = match (options_type_code, options) {
        (1, Some(options)) => {
            mean = fb.vector_f32(options, 0);
            std = fb.vector_f32(options, 1);
            let cardinality_compatible = !mean.is_empty()
                && !std.is_empty()
                && (mean.len() == std.len() || mean.len() == 1 || std.len() == 1);
            let values_valid = mean.iter().all(|value| value.is_finite())
                && std
                    .iter()
                    .all(|value| value.is_finite() && value.abs() > f32::EPSILON);
            (
                cardinality_compatible && values_valid,
                format!(
                    "mean/std cardinality {}/{}; finite means and finite non-zero std values required",
                    mean.len(),
                    std.len()
                ),
            )
        }
        (2 | 3, Some(_)) => (
            true,
            "score processing options table is present; scalar defaults remain part of the schema contract"
                .to_string(),
        ),
        (4, Some(options)) => {
            associated_files = associated_file_names(fb, options, 0);
            (
                !associated_files.is_empty(),
                format!("Bert tokenizer vocabulary file declarations {}", associated_files.len()),
            )
        }
        (5, Some(options)) => {
            associated_files = associated_file_names(fb, options, 0);
            let vocabulary = associated_file_names(fb, options, 1);
            associated_files.extend(vocabulary);
            (
                !associated_files.is_empty(),
                format!("SentencePiece model/vocabulary file declarations {}", associated_files.len()),
            )
        }
        (6, Some(options)) => {
            let pattern = fb.string_field(options, 0).unwrap_or_default();
            associated_files = associated_file_names(fb, options, 1);
            (
                !pattern.is_empty() && !associated_files.is_empty(),
                format!(
                    "regex pattern {} and {} vocabulary file declaration(s)",
                    if pattern.is_empty() { "missing" } else { "present" },
                    associated_files.len()
                ),
            )
        }
        (0, _) => (false, "ProcessUnit options union is NONE or absent".to_string()),
        (1..=6, None) => (false, "ProcessUnit options table is absent".to_string()),
        _ => (
            false,
            format!("ProcessUnit options type code {options_type_code} is not in metadata schema 1.5.0"),
        ),
    };
    let status = if options_type_code > 6 {
        "unsupported_options_type"
    } else if valid {
        "assessed"
    } else {
        "invalid_options"
    };
    MetadataProcessUnit {
        scope: scope.to_string(),
        input_ordinal,
        tensor_name: tensor_name.to_string(),
        options_type,
        options_type_code,
        status: status.to_string(),
        mean,
        std,
        associated_files,
        detail,
    }
}

fn associated_file_names(fb: &MetadataFb<'_>, table: usize, field: usize) -> Vec<String> {
    fb.vector_tables(table, field)
        .iter()
        .filter_map(|associated_file| fb.string_field(*associated_file, 0))
        .filter(|name| !name.is_empty())
        .collect()
}

fn parse_associated_file(
    fb: &MetadataFb<'_>,
    table: usize,
    output_ordinal: usize,
    tensor_name: &str,
) -> MetadataAssociatedFile {
    let file_type_code = fb
        .field_pos(table, 2)
        .and_then(|position| fb.u8(position))
        .unwrap_or(0);
    MetadataAssociatedFile {
        output_ordinal,
        tensor_name: tensor_name.to_string(),
        name: fb.string_field(table, 0).unwrap_or_default(),
        description: fb.string_field(table, 1).unwrap_or_default(),
        file_type: associated_file_type_name(file_type_code).to_string(),
        file_type_code,
        locale: fb.string_field(table, 3).unwrap_or_default(),
        version: fb.string_field(table, 4).unwrap_or_default(),
        packed_status: "not_assessed".to_string(),
        payload_status: "not_assessed".to_string(),
        payload_sha256: String::new(),
        payload_bytes: None,
        crc32_verified: None,
        text_encoding_status: "not_assessed".to_string(),
        label_entry_count: None,
        blank_label_entry_count: None,
        output_shape: Vec::new(),
        cardinality_status: "not_assessed".to_string(),
        matching_output_axes: Vec::new(),
        validation_detail: String::new(),
    }
}

pub(crate) fn parse_packed_metadata_archive(bytes: &[u8]) -> PackedMetadataArchive {
    let Some(eocd) = find_zip_eocd(bytes) else {
        return PackedMetadataArchive {
            status: "not_present".to_string(),
            files: Vec::new(),
            archive_start: None,
            archive_end: None,
            central_directory_start: None,
            central_directory_end: None,
            eocd_offset: None,
            case_insensitive_name_collision_count: 0,
            detail: "No terminal ZIP end-of-central-directory record was found.".to_string(),
        };
    };
    let disk = read_u16(bytes, eocd + 4);
    let central_disk = read_u16(bytes, eocd + 6);
    let disk_entries = read_u16(bytes, eocd + 8);
    let total_entries = read_u16(bytes, eocd + 10);
    let central_size = read_u32(bytes, eocd + 12);
    let central_offset = read_u32(bytes, eocd + 16);
    let Some((disk, central_disk, disk_entries, total_entries, central_size, central_offset)) =
        disk.zip(central_disk)
            .zip(disk_entries)
            .zip(total_entries)
            .zip(central_size)
            .zip(central_offset)
            .map(
                |(
                    ((((disk, central_disk), disk_entries), total_entries), central_size),
                    central_offset,
                )| {
                    (
                        disk,
                        central_disk,
                        disk_entries,
                        total_entries,
                        central_size,
                        central_offset,
                    )
                },
            )
    else {
        return malformed_archive("ZIP end-of-central-directory fields exceed artifact bounds.");
    };
    if disk != 0 || central_disk != 0 || disk_entries != total_entries {
        return malformed_archive("Multi-disk ZIP associated-file archives are not supported.");
    }
    if total_entries == u16::MAX || central_size == u32::MAX || central_offset == u32::MAX {
        return malformed_archive(
            "ZIP64 associated-file archives are not supported by this parser.",
        );
    }
    let central_start = central_offset as usize;
    let Some(central_end) = central_start.checked_add(central_size as usize) else {
        return malformed_archive("ZIP central-directory byte range overflowed.");
    };
    if central_end > eocd || central_end > bytes.len() {
        return malformed_archive("ZIP central directory exceeds the terminal archive bounds.");
    }

    let mut position = central_start;
    let mut files = Vec::with_capacity(total_entries as usize);
    let mut names = HashSet::new();
    let mut folded_names = HashMap::<String, String>::new();
    let mut case_insensitive_name_collision_count = 0usize;
    let mut archive_start = central_start;
    let mut local_ranges = Vec::<(usize, usize)>::new();
    let mut decoded_budget = 0usize;
    for _ in 0..total_entries {
        if read_u32(bytes, position) != Some(0x0201_4b50) {
            return malformed_archive("ZIP central-directory entry signature is missing.");
        }
        let (
            Some(flags),
            Some(compression_method),
            Some(crc32),
            Some(compressed_bytes),
            Some(uncompressed_bytes),
            Some(name_length),
            Some(extra_length),
            Some(comment_length),
            Some(local_header_offset),
        ) = (
            read_u16(bytes, position + 8),
            read_u16(bytes, position + 10),
            read_u32(bytes, position + 16),
            read_u32(bytes, position + 20),
            read_u32(bytes, position + 24),
            read_u16(bytes, position + 28),
            read_u16(bytes, position + 30),
            read_u16(bytes, position + 32),
            read_u32(bytes, position + 42),
        )
        else {
            return malformed_archive(
                "ZIP central-directory entry header exceeds artifact bounds.",
            );
        };
        let name_length = name_length as usize;
        let extra_length = extra_length as usize;
        let comment_length = comment_length as usize;
        if compressed_bytes == u32::MAX
            || uncompressed_bytes == u32::MAX
            || local_header_offset == u32::MAX
        {
            return malformed_archive("ZIP64 central-directory entry fields are not supported.");
        }
        let Some(name_start) = position.checked_add(46) else {
            return malformed_archive("ZIP central-directory name offset overflowed.");
        };
        let Some(name_end) = name_start.checked_add(name_length) else {
            return malformed_archive("ZIP central-directory name range overflowed.");
        };
        let Some(next) = name_end
            .checked_add(extra_length)
            .and_then(|value| value.checked_add(comment_length))
        else {
            return malformed_archive("ZIP central-directory variable fields overflowed.");
        };
        let Some(name_bytes) = bytes.get(name_start..name_end) else {
            return malformed_archive("ZIP central-directory filename exceeds artifact bounds.");
        };
        if flags & 0x0800 == 0 && !name_bytes.is_ascii() {
            return malformed_archive(
                "A ZIP filename is neither UTF-8 flagged nor ASCII, so exact metadata-name binding is unavailable.",
            );
        }
        let Ok(name) = std::str::from_utf8(name_bytes) else {
            return malformed_archive("A UTF-8 flagged ZIP filename is not valid UTF-8.");
        };
        if name.is_empty() {
            return malformed_archive("A ZIP associated-file entry has an empty filename.");
        }
        if !names.insert(name.to_string()) {
            return malformed_archive(
                "The ZIP central directory contains a duplicate associated-file name.",
            );
        }
        let folded = name.to_lowercase();
        if folded_names
            .insert(folded, name.to_string())
            .is_some_and(|previous| previous != name)
        {
            case_insensitive_name_collision_count += 1;
        }
        let local = local_header_offset as usize;
        if local >= central_start || read_u32(bytes, local) != Some(0x0403_4b50) {
            return malformed_archive(
                "A ZIP central-directory entry does not bind a valid local file header.",
            );
        }
        let Some(local_flags) = read_u16(bytes, local + 6) else {
            return malformed_archive("A ZIP local header exceeds artifact bounds.");
        };
        let Some(local_method) = read_u16(bytes, local + 8) else {
            return malformed_archive("A ZIP local header compression field is missing.");
        };
        let Some(local_name_length) = read_u16(bytes, local + 26).map(usize::from) else {
            return malformed_archive("A ZIP local header filename length is missing.");
        };
        let Some(local_extra_length) = read_u16(bytes, local + 28).map(usize::from) else {
            return malformed_archive("A ZIP local header extra-field length is missing.");
        };
        if local_flags != flags || local_method != compression_method {
            return malformed_archive(
                "A ZIP local header disagrees with its central-directory flags or compression method.",
            );
        }
        let Some(local_name_start) = local.checked_add(30) else {
            return malformed_archive("A ZIP local filename offset overflowed.");
        };
        let Some(local_name_end) = local_name_start.checked_add(local_name_length) else {
            return malformed_archive("A ZIP local filename range overflowed.");
        };
        if bytes.get(local_name_start..local_name_end) != Some(name_bytes) {
            return malformed_archive(
                "A ZIP local filename does not match its central-directory filename.",
            );
        }
        let Some(data_start) = local_name_end.checked_add(local_extra_length) else {
            return malformed_archive("A ZIP local payload offset overflowed.");
        };
        let Some(data_end) = data_start.checked_add(compressed_bytes as usize) else {
            return malformed_archive("A ZIP compressed payload range overflowed.");
        };
        let Some(compressed_payload) = bytes.get(data_start..data_end) else {
            return malformed_archive("A ZIP compressed payload exceeds artifact bounds.");
        };
        if data_end > central_start {
            return malformed_archive("A ZIP compressed payload overlaps the central directory.");
        }
        if local_ranges
            .iter()
            .any(|(start, end)| local < *end && *start < data_end)
        {
            return malformed_archive("ZIP local file records overlap each other.");
        }
        local_ranges.push((local, data_end));
        archive_start = archive_start.min(local);
        let declared_decoded = uncompressed_bytes as usize;
        let within_budget = declared_decoded <= MAX_ASSOCIATED_FILE_BYTES
            && decoded_budget
                .checked_add(declared_decoded)
                .is_some_and(|total| total <= MAX_ASSOCIATED_ARCHIVE_DECODED_BYTES);
        let (payload, payload_status, payload_detail) = if flags & 0x0001 != 0 {
            (
                Vec::new(),
                "encrypted_not_supported",
                "Encrypted ZIP entries are not decoded.",
            )
        } else if !within_budget {
            (
                Vec::new(),
                "decoded_size_limit_exceeded",
                "Declared decoded size exceeds the per-file or aggregate safety limit.",
            )
        } else {
            decoded_budget += declared_decoded;
            match compression_method {
                0 if compressed_bytes == uncompressed_bytes => (
                    compressed_payload.to_vec(),
                    "decoded",
                    "Stored payload decoded from the exact local-header byte range.",
                ),
                0 => (
                    Vec::new(),
                    "stored_size_mismatch",
                    "Stored ZIP entry declares different compressed and uncompressed sizes.",
                ),
                8 => match decompress_to_vec_with_limit(compressed_payload, declared_decoded) {
                    Ok(payload) => (
                        payload,
                        "decoded",
                        "Raw DEFLATE payload decoded within the declared-size limit.",
                    ),
                    Err(_) => (
                        Vec::new(),
                        "deflate_decode_failed",
                        "Raw DEFLATE payload did not decode within the declared-size limit.",
                    ),
                },
                _ => (
                    Vec::new(),
                    "unsupported_compression_method",
                    "Only stored and DEFLATE ZIP entries are decoded.",
                ),
            }
        };
        let decoded_bytes = (payload_status == "decoded").then_some(payload.len());
        let decoded_size_matches = decoded_bytes == Some(declared_decoded);
        let computed_crc32 = decoded_bytes.map(|_| crc32_ieee(&payload));
        let crc32_verified = computed_crc32.map(|value| value == crc32);
        let final_status = if payload_status != "decoded" {
            payload_status
        } else if !decoded_size_matches {
            "decoded_size_mismatch"
        } else if crc32_verified != Some(true) {
            "crc32_mismatch"
        } else {
            "verified"
        };
        let payload_sha256 = decoded_bytes
            .map(|_| format!("{:x}", Sha256::digest(&payload)))
            .unwrap_or_default();
        files.push(PackedMetadataFile {
            name: name.to_string(),
            compressed_bytes,
            uncompressed_bytes,
            compression_method,
            crc32,
            local_header_offset,
            payload_status: final_status.to_string(),
            payload_sha256,
            decoded_bytes,
            crc32_verified,
            detail: format!(
                "{} Declared decoded/compressed bytes {}/{}; decoded bytes {}; CRC32 declaration 0x{crc32:08x}{}.",
                payload_detail,
                uncompressed_bytes,
                compressed_bytes,
                decoded_bytes
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "not available".to_string()),
                computed_crc32
                    .map(|value| format!("; recomputed 0x{value:08x}"))
                    .unwrap_or_default(),
            ),
            payload,
        });
        position = next;
    }
    if position != central_end {
        return malformed_archive(
            "ZIP central-directory size does not equal the parsed entry ledger size.",
        );
    }
    PackedMetadataArchive {
        status: "assessed".to_string(),
        archive_start: Some(archive_start),
        archive_end: Some(bytes.len()),
        central_directory_start: Some(central_start),
        central_directory_end: Some(central_end),
        eocd_offset: Some(eocd),
        case_insensitive_name_collision_count,
        detail: format!(
            "Terminal ZIP central directory parsed exactly: {} file entry or entries, {} central-directory bytes at artifact offset {}; archive range [{}..{}); case-insensitive filename collisions {}.",
            files.len(), central_size, central_offset, archive_start, bytes.len(), case_insensitive_name_collision_count
        ),
        files,
    }
}

pub(crate) fn bind_packed_associated_files(
    metadata: &mut ParsedTfliteModelMetadata,
    archive: PackedMetadataArchive,
    output_shapes: &[Vec<i32>],
) -> BoundMetadataAssociatedFiles {
    let packed_by_name = archive
        .files
        .iter()
        .map(|file| (file.name.as_str(), file))
        .collect::<HashMap<_, _>>();
    for unit in &mut metadata.input_process_units {
        if unit.associated_files.is_empty() {
            continue;
        }
        let missing_files = unit
            .associated_files
            .iter()
            .filter(|name| !packed_by_name.contains_key(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let invalid_files = unit
            .associated_files
            .iter()
            .filter_map(|name| {
                packed_by_name
                    .get(name.as_str())
                    .filter(|file| file.payload_status != "verified")
                    .map(|file| format!("{} ({})", name, file.payload_status))
            })
            .collect::<Vec<_>>();
        if archive.status != "assessed" || !missing_files.is_empty() || !invalid_files.is_empty() {
            unit.status = "invalid_options".to_string();
            unit.detail = format!(
                "{}; associated-file archive status {}; missing packed file(s) {}; payload-invalid file(s) {}",
                unit.detail,
                archive.status,
                if missing_files.is_empty() {
                    "none".to_string()
                } else {
                    missing_files.join(", ")
                },
                if invalid_files.is_empty() {
                    "none".to_string()
                } else {
                    invalid_files.join(", ")
                },
            );
        } else {
            unit.detail = format!(
                "{}; every declared associated file has a verified decoded payload",
                unit.detail
            );
        }
    }
    metadata.recognized_input_process_unit_count = metadata
        .input_process_units
        .iter()
        .filter(|unit| unit.status == "assessed")
        .count();
    metadata.invalid_input_process_unit_count = metadata
        .input_process_units
        .iter()
        .filter(|unit| unit.status == "invalid_options")
        .count();
    metadata.unrecognized_input_process_unit_count = metadata
        .input_process_units
        .iter()
        .filter(|unit| unit.status == "unsupported_options_type")
        .count();
    for file in &mut metadata.output_associated_files {
        file.output_shape = output_shapes
            .get(file.output_ordinal)
            .cloned()
            .unwrap_or_default();
        if archive.status != "assessed" {
            file.packed_status = "not_assessed_archive_unavailable".to_string();
            file.payload_status = "not_assessed_archive_unavailable".to_string();
            file.cardinality_status = "not_assessed_archive_unavailable".to_string();
            file.validation_detail = archive.detail.clone();
            continue;
        }
        let Some(packed) = packed_by_name.get(file.name.as_str()).copied() else {
            file.packed_status = "missing_from_archive".to_string();
            file.payload_status = "missing_from_archive".to_string();
            file.cardinality_status = "not_assessed_missing_payload".to_string();
            file.validation_detail =
                "The metadata filename has no exact terminal ZIP central-directory entry."
                    .to_string();
            continue;
        };
        file.payload_status = packed.payload_status.clone();
        file.payload_sha256 = packed.payload_sha256.clone();
        file.payload_bytes = packed.decoded_bytes;
        file.crc32_verified = packed.crc32_verified;
        if packed.payload_status != "verified" {
            file.packed_status = "present_payload_not_verified".to_string();
            file.cardinality_status = "not_assessed_payload_not_verified".to_string();
            file.validation_detail = packed.detail.clone();
            continue;
        }
        file.packed_status = "verified_payload".to_string();
        if !matches!(file.file_type_code, 2 | 3) {
            file.text_encoding_status = "not_applicable_non_label_file".to_string();
            file.cardinality_status = "not_applicable_non_label_file".to_string();
            file.validation_detail = "Associated-file payload integrity verified.".to_string();
            continue;
        }
        let Ok(text) = std::str::from_utf8(&packed.payload) else {
            file.text_encoding_status = "invalid_utf8".to_string();
            file.cardinality_status = "not_assessed_invalid_utf8".to_string();
            file.validation_detail =
                "Label payload integrity passed, but the decoded bytes are not valid UTF-8."
                    .to_string();
            continue;
        };
        file.text_encoding_status = "valid_utf8".to_string();
        let (label_entry_count, blank_label_entry_count) = count_text_entries(text);
        file.label_entry_count = Some(label_entry_count);
        file.blank_label_entry_count = Some(blank_label_entry_count);
        if label_entry_count == 0 {
            file.cardinality_status = "invalid_empty_label_file".to_string();
            file.validation_detail =
                "The verified label payload contains zero entries.".to_string();
            continue;
        }
        if file.file_type_code == 3 {
            file.cardinality_status = "not_applicable_value_labels".to_string();
            file.validation_detail = format!(
                "Verified UTF-8 TENSOR_VALUE_LABELS payload contains {} entries ({} blank); tensor-axis cardinality is not applicable to value-to-label mapping.",
                label_entry_count, blank_label_entry_count
            );
            continue;
        }
        let dynamic_shape = file.output_shape.iter().any(|dimension| *dimension < 0);
        file.matching_output_axes = file
            .output_shape
            .iter()
            .enumerate()
            .filter(|(axis, dimension)| {
                !(*axis == 0 && file.output_shape.len() > 1 && **dimension == 1)
                    && **dimension > 0
                    && **dimension as usize == label_entry_count
            })
            .map(|(axis, _)| axis)
            .collect();
        file.cardinality_status = match (file.matching_output_axes.len(), dynamic_shape) {
            (1, false) => "verified_unique_axis_match",
            (1, true) => "unresolved_dynamic_shape_with_known_axis_match",
            (0, true) => "unresolved_dynamic_shape_no_known_axis_match",
            (0, false) => "mismatch_no_output_axis",
            (_, _) => "ambiguous_multiple_output_axes",
        }
        .to_string();
        file.validation_detail = format!(
            "Verified UTF-8 TENSOR_AXIS_LABELS payload contains {} entries ({} blank); output shape [{}]; matching non-batch axis indices [{}].",
            label_entry_count,
            blank_label_entry_count,
            file.output_shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            file.matching_output_axes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    let verified_output_associated_file_count = metadata
        .output_associated_files
        .iter()
        .filter(|file| file.packed_status == "verified_payload")
        .count();
    let missing_output_associated_file_count = metadata
        .output_associated_files
        .iter()
        .filter(|file| file.packed_status == "missing_from_archive")
        .count();
    let label_is_verified = |file: &&MetadataAssociatedFile| {
        file.packed_status == "verified_payload"
            && file.text_encoding_status == "valid_utf8"
            && file.label_entry_count.unwrap_or(0) > 0
            && (file.file_type_code == 3 || file.cardinality_status == "verified_unique_axis_match")
    };
    let verified_output_label_file_count = metadata
        .output_associated_files
        .iter()
        .filter(|file| matches!(file.file_type_code, 2 | 3))
        .filter(label_is_verified)
        .count();
    let missing_output_label_file_count = metadata
        .output_associated_files
        .iter()
        .filter(|file| {
            matches!(file.file_type_code, 2 | 3) && file.packed_status == "missing_from_archive"
        })
        .count();
    let verified_output0_label_file_count = metadata
        .output_associated_files
        .iter()
        .filter(|file| file.output_ordinal == 0 && matches!(file.file_type_code, 2 | 3))
        .filter(label_is_verified)
        .count();
    let payload_verified_file_count = archive
        .files
        .iter()
        .filter(|file| file.payload_status == "verified")
        .count();
    let payload_unsupported_file_count = archive
        .files
        .iter()
        .filter(|file| {
            matches!(
                file.payload_status.as_str(),
                "unsupported_compression_method"
                    | "encrypted_not_supported"
                    | "decoded_size_limit_exceeded"
            )
        })
        .count();
    let payload_invalid_file_count = archive
        .files
        .len()
        .saturating_sub(payload_verified_file_count + payload_unsupported_file_count);
    let axis_label_files = metadata
        .output_associated_files
        .iter()
        .filter(|file| file.file_type_code == 2)
        .collect::<Vec<_>>();
    let label_cardinality_match_count = axis_label_files
        .iter()
        .filter(|file| file.cardinality_status == "verified_unique_axis_match")
        .count();
    let label_cardinality_mismatch_count = axis_label_files
        .iter()
        .filter(|file| file.cardinality_status == "mismatch_no_output_axis")
        .count();
    let label_cardinality_ambiguous_count = axis_label_files
        .iter()
        .filter(|file| file.cardinality_status == "ambiguous_multiple_output_axes")
        .count();
    let label_cardinality_unresolved_count = axis_label_files.len().saturating_sub(
        label_cardinality_match_count
            + label_cardinality_mismatch_count
            + label_cardinality_ambiguous_count,
    );
    let invalid_output_label_file_count = metadata
        .output_label_file_count
        .saturating_sub(verified_output_label_file_count + missing_output_label_file_count);
    drop(packed_by_name);
    BoundMetadataAssociatedFiles {
        archive_status: archive.status,
        archive_detail: archive.detail,
        packed_files: archive.files,
        verified_output_associated_file_count,
        missing_output_associated_file_count,
        verified_output_label_file_count,
        missing_output_label_file_count,
        invalid_output_label_file_count,
        verified_output0_label_file_count,
        payload_verified_file_count,
        payload_invalid_file_count,
        payload_unsupported_file_count,
        label_cardinality_match_count,
        label_cardinality_mismatch_count,
        label_cardinality_ambiguous_count,
        label_cardinality_unresolved_count,
    }
}

fn count_text_entries(text: &str) -> (usize, usize) {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if text.is_empty() {
        return (0, 0);
    }
    let mut lines = text.split('\n').collect::<Vec<_>>();
    if text.ends_with('\n') {
        lines.pop();
    }
    let blank = lines
        .iter()
        .filter(|line| line.strip_suffix('\r').unwrap_or(line).is_empty())
        .count();
    (lines.len(), blank)
}

fn malformed_archive(detail: &str) -> PackedMetadataArchive {
    PackedMetadataArchive {
        status: "malformed".to_string(),
        files: Vec::new(),
        archive_start: None,
        archive_end: None,
        central_directory_start: None,
        central_directory_end: None,
        eocd_offset: None,
        case_insensitive_name_collision_count: 0,
        detail: detail.to_string(),
    }
}

fn find_zip_eocd(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 22 {
        return None;
    }
    let start = bytes.len().saturating_sub(65_557);
    (start..=bytes.len() - 22).rev().find(|position| {
        if read_u32(bytes, *position) != Some(0x0605_4b50) {
            return false;
        }
        let Some(comment_length) = read_u16(bytes, *position + 20) else {
            return false;
        };
        position
            .checked_add(22 + comment_length as usize)
            .is_some_and(|end| end == bytes.len())
    })
}

fn read_u16(bytes: &[u8], position: usize) -> Option<u16> {
    let bytes = bytes.get(position..position.checked_add(2)?)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(bytes: &[u8], position: usize) -> Option<u32> {
    let bytes = bytes.get(position..position.checked_add(4)?)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

fn process_unit_type_name(code: u8) -> &'static str {
    match code {
        0 => "NONE",
        1 => "NormalizationOptions",
        2 => "ScoreCalibrationOptions",
        3 => "ScoreThresholdingOptions",
        4 => "BertTokenizerOptions",
        5 => "SentencePieceTokenizerOptions",
        6 => "RegexTokenizerOptions",
        _ => "UNKNOWN",
    }
}

fn associated_file_type_name(code: u8) -> &'static str {
    match code {
        0 => "UNKNOWN",
        1 => "DESCRIPTIONS",
        2 => "TENSOR_AXIS_LABELS",
        3 => "TENSOR_VALUE_LABELS",
        4 => "TENSOR_AXIS_SCORE_CALIBRATION",
        5 => "VOCABULARY",
        6 => "SCANN_INDEX_FILE",
        _ => "UNRECOGNIZED",
    }
}

struct MetadataFb<'a> {
    data: &'a [u8],
}

impl<'a> MetadataFb<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    fn root_table(&self) -> Option<usize> {
        let root = self.u32(0)? as usize;
        (root < self.data.len()).then_some(root)
    }

    fn u8(&self, position: usize) -> Option<u8> {
        self.data.get(position).copied()
    }

    fn u16(&self, position: usize) -> Option<u16> {
        let bytes = self.data.get(position..position.checked_add(2)?)?;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i32(&self, position: usize) -> Option<i32> {
        let bytes = self.data.get(position..position.checked_add(4)?)?;
        Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u32(&self, position: usize) -> Option<u32> {
        let bytes = self.data.get(position..position.checked_add(4)?)?;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&self, position: usize) -> Option<f32> {
        let bytes = self.data.get(position..position.checked_add(4)?)?;
        Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn vtable(&self, table: usize) -> Option<usize> {
        let offset = self.i32(table)? as isize;
        let position = table as isize - offset;
        (position >= 0 && (position as usize) < self.data.len()).then_some(position as usize)
    }

    fn field_pos(&self, table: usize, field_index: usize) -> Option<usize> {
        let vtable = self.vtable(table)?;
        let vtable_offset = 4usize.checked_add(field_index.checked_mul(2)?)?;
        let vtable_size = self.u16(vtable)? as usize;
        if vtable_offset >= vtable_size {
            return None;
        }
        let offset = self.u16(vtable.checked_add(vtable_offset)?)? as usize;
        if offset == 0 {
            return None;
        }
        let position = table.checked_add(offset)?;
        (position < self.data.len()).then_some(position)
    }

    fn table_field(&self, table: usize, field_index: usize) -> Option<usize> {
        let position = self.field_pos(table, field_index)?;
        let target = position.checked_add(self.u32(position)? as usize)?;
        (target < self.data.len()).then_some(target)
    }

    fn string_field(&self, table: usize, field_index: usize) -> Option<String> {
        let position = self.field_pos(table, field_index)?;
        let string = position.checked_add(self.u32(position)? as usize)?;
        let length = self.u32(string)? as usize;
        let start = string.checked_add(4)?;
        let bytes = self.data.get(start..start.checked_add(length)?)?;
        Some(String::from_utf8_lossy(bytes).to_string())
    }

    fn table_within_file(&self, table: usize) -> bool {
        let Some(vtable) = self.vtable(table) else {
            return false;
        };
        let Some(vtable_size) = self.u16(vtable).map(usize::from) else {
            return false;
        };
        let Some(table_size) = self.u16(vtable + 2).map(usize::from) else {
            return false;
        };
        if vtable_size < 4 || vtable_size % 2 != 0 || table_size < 4 {
            return false;
        }
        let Some(vtable_end) = vtable.checked_add(vtable_size) else {
            return false;
        };
        let Some(table_end) = table.checked_add(table_size) else {
            return false;
        };
        if vtable_end > self.data.len() || table_end > self.data.len() {
            return false;
        }
        (4..vtable_size).step_by(2).all(|entry| {
            self.u16(vtable + entry).is_some_and(|offset| {
                offset == 0 || (usize::from(offset) >= 4 && usize::from(offset) < table_size)
            })
        })
    }

    fn checked_field_pos(
        &self,
        table: usize,
        field_index: usize,
        width: usize,
        label: &str,
    ) -> Result<Option<usize>, String> {
        if !self.table_within_file(table) {
            return Err(format!("{label} table is outside the metadata buffer"));
        }
        let Some(position) = self.field_pos(table, field_index) else {
            return Ok(None);
        };
        let vtable = self
            .vtable(table)
            .ok_or_else(|| format!("{label} vtable is invalid"))?;
        let table_size = usize::from(
            self.u16(vtable + 2)
                .ok_or_else(|| format!("{label} table size is truncated"))?,
        );
        let field_end = position
            .checked_add(width)
            .ok_or_else(|| format!("{label} field extent overflows"))?;
        let table_end = table
            .checked_add(table_size)
            .ok_or_else(|| format!("{label} table extent overflows"))?;
        if position < table || field_end > table_end || field_end > self.data.len() {
            return Err(format!("{label} field exceeds its table"));
        }
        Ok(Some(position))
    }

    fn checked_u8_field(
        &self,
        table: usize,
        field_index: usize,
        default: u8,
        label: &str,
    ) -> Result<u8, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 1, label)? else {
            return Ok(default);
        };
        self.u8(position)
            .ok_or_else(|| format!("{label} scalar is truncated"))
    }

    fn checked_table_field(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<Option<usize>, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(None);
        };
        let target = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} table offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} table offset overflows"))?;
        if !self.table_within_file(target) {
            return Err(format!(
                "{label} referenced table is outside the metadata buffer"
            ));
        }
        Ok(Some(target))
    }

    fn require_string_field(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<(), String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(());
        };
        let string = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} string offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} string offset overflows"))?;
        let length =
            self.u32(string)
                .ok_or_else(|| format!("{label} string length is truncated"))? as usize;
        let start = string
            .checked_add(4)
            .ok_or_else(|| format!("{label} string start overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| format!("{label} string extent overflows"))?;
        let bytes = self
            .data
            .get(start..end)
            .ok_or_else(|| format!("{label} string exceeds the metadata buffer"))?;
        if self.data.get(end).copied() != Some(0) {
            return Err(format!("{label} string terminator is missing"));
        }
        std::str::from_utf8(bytes).map_err(|_| format!("{label} string is not valid UTF-8"))?;
        Ok(())
    }

    fn checked_vector_location(
        &self,
        table: usize,
        field_index: usize,
        stride: usize,
        label: &str,
    ) -> Result<Option<(usize, usize)>, String> {
        let Some(position) = self.checked_field_pos(table, field_index, 4, label)? else {
            return Ok(None);
        };
        let vector = position
            .checked_add(
                self.u32(position)
                    .ok_or_else(|| format!("{label} vector offset is truncated"))?
                    as usize,
            )
            .ok_or_else(|| format!("{label} vector offset overflows"))?;
        let length =
            self.u32(vector)
                .ok_or_else(|| format!("{label} vector length is truncated"))? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| format!("{label} vector start overflows"))?;
        let end = start
            .checked_add(
                length
                    .checked_mul(stride)
                    .ok_or_else(|| format!("{label} vector length overflows"))?,
            )
            .ok_or_else(|| format!("{label} vector extent overflows"))?;
        if end > self.data.len() {
            return Err(format!("{label} vector exceeds the metadata buffer"));
        }
        Ok(Some((start, length)))
    }

    fn checked_vector_tables(
        &self,
        table: usize,
        field_index: usize,
        label: &str,
    ) -> Result<Vec<usize>, String> {
        let Some((start, length)) = self.checked_vector_location(table, field_index, 4, label)?
        else {
            return Ok(Vec::new());
        };
        (0..length)
            .map(|index| {
                let entry = start + index * 4;
                let target = entry
                    .checked_add(
                        self.u32(entry)
                            .ok_or_else(|| format!("{label} table offset is truncated"))?
                            as usize,
                    )
                    .ok_or_else(|| format!("{label} table offset overflows"))?;
                if !self.table_within_file(target) {
                    return Err(format!("{label} contains an invalid table"));
                }
                Ok(target)
            })
            .collect()
    }

    fn require_associated_file(&self, table: usize) -> Result<(), String> {
        self.require_string_field(table, 0, "AssociatedFile.name")?;
        self.require_string_field(table, 1, "AssociatedFile.description")?;
        self.checked_u8_field(table, 2, 0, "AssociatedFile.type")?;
        self.require_string_field(table, 3, "AssociatedFile.locale")?;
        self.require_string_field(table, 4, "AssociatedFile.version")?;
        Ok(())
    }

    fn require_process_unit(&self, table: usize) -> Result<(), String> {
        let code = self.checked_u8_field(table, 0, 0, "ProcessUnit.options_type")?;
        let options = self.checked_table_field(table, 1, "ProcessUnit.options")?;
        let Some(options) = options else {
            return Ok(());
        };
        match code {
            1 => {
                self.checked_vector_location(options, 0, 4, "NormalizationOptions.mean")?;
                self.checked_vector_location(options, 1, 4, "NormalizationOptions.std")?;
            }
            4 => {
                for file in
                    self.checked_vector_tables(options, 0, "BertTokenizerOptions.vocab_file")?
                {
                    self.require_associated_file(file)?;
                }
            }
            5 => {
                for field in 0..=1 {
                    for file in self.checked_vector_tables(
                        options,
                        field,
                        "SentencePieceTokenizerOptions.files",
                    )? {
                        self.require_associated_file(file)?;
                    }
                }
            }
            6 => {
                self.require_string_field(options, 0, "RegexTokenizerOptions.delim_regex_pattern")?;
                for file in
                    self.checked_vector_tables(options, 1, "RegexTokenizerOptions.vocab_file")?
                {
                    self.require_associated_file(file)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn require_tensor_metadata(&self, table: usize) -> Result<(), String> {
        self.require_string_field(table, 0, "TensorMetadata.name")?;
        self.require_string_field(table, 1, "TensorMetadata.description")?;
        for unit in self.checked_vector_tables(table, 4, "TensorMetadata.process_units")? {
            self.require_process_unit(unit)?;
        }
        for file in self.checked_vector_tables(table, 6, "TensorMetadata.associated_files")? {
            self.require_associated_file(file)?;
        }
        Ok(())
    }

    fn require_bounded_metadata(&self, root: usize) -> Result<(), String> {
        if !self.table_within_file(root) {
            return Err("ModelMetadata root table is outside the metadata buffer".to_string());
        }
        for (field, label) in [
            (0usize, "ModelMetadata.name"),
            (1usize, "ModelMetadata.description"),
            (2usize, "ModelMetadata.version"),
            (4usize, "ModelMetadata.author"),
            (5usize, "ModelMetadata.license"),
            (7usize, "ModelMetadata.min_parser_version"),
        ] {
            self.require_string_field(root, field, label)?;
        }
        for subgraph in self.checked_vector_tables(root, 3, "ModelMetadata.subgraph_metadata")? {
            for field in [2usize, 3usize] {
                for tensor in
                    self.checked_vector_tables(subgraph, field, "SubGraphMetadata.tensor_metadata")?
                {
                    self.require_tensor_metadata(tensor)?;
                }
            }
            for unit in
                self.checked_vector_tables(subgraph, 5, "SubGraphMetadata.input_process_units")?
            {
                self.require_process_unit(unit)?;
            }
        }
        Ok(())
    }

    fn vector_location(&self, table: usize, field_index: usize) -> Option<(usize, usize)> {
        let position = self.field_pos(table, field_index)?;
        let vector = position.checked_add(self.u32(position)? as usize)?;
        let length = self.u32(vector)? as usize;
        Some((vector.checked_add(4)?, length))
    }

    fn vector_f32(&self, table: usize, field_index: usize) -> Vec<f32> {
        let Some((start, length)) = self.vector_location(table, field_index) else {
            return Vec::new();
        };
        (0..length)
            .filter_map(|index| self.f32(start.checked_add(index.checked_mul(4)?)?))
            .collect()
    }

    fn vector_tables(&self, table: usize, field_index: usize) -> Vec<usize> {
        let Some((start, length)) = self.vector_location(table, field_index) else {
            return Vec::new();
        };
        (0..length)
            .filter_map(|index| {
                let entry = start.checked_add(index.checked_mul(4)?)?;
                let target = entry.checked_add(self.u32(entry)? as usize)?;
                (target < self.data.len()).then_some(target)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::deflate::compress_to_vec;

    const FIXTURE_HEX: &str = "1c0000004d303031140020001c001800140010000c00080000000400140000001c00000024000000300000007800000034000000380000005400000005000000312e352e300000000a0000004170616368652d322e3000000700000044656570424f4d0003000000312e30001b00000053796e746865746963206d6574616461746120636f6e747261637400100000004d6574616461746120666978747572650000000001000000100000000c0010000c000000080004000c000000180000001c00000004000000040000006d61696e00000000010000002000000001000000b00000000000120010000c000800000000000000000004001200000030000000080000001c00000013000000636c6173732070726f626162696c6974696573000600000073636f726573000001000000100000000c00140010000c000b0004000c0000001000000000000002100000002000000002000000656e00000c000000636c617373206c6162656c73000000000a0000006c6162656c732e747874000000000e0010000c0008000000000004000e0000002800000008000000140000000900000052474220696d61676500000006000000706978656c730000010000000c00000008000c000b00040008000000100000000000000108000c000800040008000000080000000c000000010000000000ff42010000000000ff42";

    fn decode_hex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let text = std::str::from_utf8(pair).expect("fixture hex is ASCII");
                u8::from_str_radix(text, 16).expect("fixture hex is valid")
            })
            .collect()
    }

    #[test]
    fn parses_explicit_normalization_and_output_label_contracts() {
        let parsed = parse_tflite_model_metadata(&decode_hex(FIXTURE_HEX));
        assert_eq!(parsed.status, "parsed");
        assert_eq!(parsed.schema_identifier, "M001");
        assert_eq!(parsed.min_parser_version, "1.5.0");
        assert_eq!(parsed.model_name, "Metadata fixture");
        assert_eq!(parsed.input_tensor_metadata_count, 1);
        assert_eq!(parsed.output_tensor_metadata_count, 1);
        assert_eq!(parsed.input_process_units.len(), 1);
        assert_eq!(parsed.recognized_input_process_unit_count, 1);
        assert_eq!(parsed.normalization_unit_count, 1);
        assert_eq!(parsed.input_process_units[0].mean, vec![127.5]);
        assert_eq!(parsed.input_process_units[0].std, vec![127.5]);
        assert_eq!(parsed.output_label_file_count, 1);
        assert_eq!(parsed.output_associated_files[0].name, "labels.txt");
        assert_eq!(
            parsed.output_associated_files[0].file_type,
            "TENSOR_AXIS_LABELS"
        );
    }

    #[test]
    fn metadata_truncation_is_rejected_or_preserves_the_full_contract() {
        let bytes = decode_hex(FIXTURE_HEX);
        let baseline = parse_tflite_model_metadata(&bytes);
        assert_eq!(baseline.status, "parsed");

        for cut in 0..bytes.len() {
            let candidate = parse_tflite_model_metadata(&bytes[..cut]);
            if candidate.status != "parsed" {
                continue;
            }
            assert_eq!(
                candidate.min_parser_version, baseline.min_parser_version,
                "cut={cut}"
            );
            assert_eq!(candidate.model_name, baseline.model_name, "cut={cut}");
            assert_eq!(
                candidate.input_tensor_metadata_count, baseline.input_tensor_metadata_count,
                "cut={cut}"
            );
            assert_eq!(
                candidate.output_tensor_metadata_count, baseline.output_tensor_metadata_count,
                "cut={cut}"
            );
            assert_eq!(
                candidate.recognized_input_process_unit_count,
                baseline.recognized_input_process_unit_count,
                "cut={cut}"
            );
            assert_eq!(
                candidate.output_label_file_count, baseline.output_label_file_count,
                "cut={cut}"
            );
        }
    }

    #[test]
    fn rejects_missing_or_wrong_metadata_identifiers() {
        let short = parse_tflite_model_metadata(&[0; 7]);
        assert_eq!(short.status, "malformed_metadata_buffer");
        let mut wrong = decode_hex(FIXTURE_HEX);
        wrong[4..8].copy_from_slice(b"BAD!");
        let parsed = parse_tflite_model_metadata(&wrong);
        assert_eq!(parsed.status, "unsupported_metadata_identifier");
        assert_eq!(parsed.schema_identifier, "BAD!");
    }

    #[test]
    fn parses_prefixed_terminal_zip_central_directory() {
        let bytes = prefixed_stored_zip("labels.txt", b"zero\none\n");
        let parsed = parse_packed_metadata_archive(&bytes);
        assert_eq!(parsed.status, "assessed");
        assert_eq!(parsed.files.len(), 1);
        assert_eq!(parsed.files[0].name, "labels.txt");
        assert_eq!(parsed.files[0].compression_method, 0);
        assert_eq!(parsed.files[0].compressed_bytes, 9);
        assert_eq!(parsed.files[0].uncompressed_bytes, 9);
        assert_eq!(parsed.files[0].payload_status, "verified");
        assert_eq!(parsed.files[0].decoded_bytes, Some(9));
        assert_eq!(parsed.files[0].crc32_verified, Some(true));
        assert_eq!(parsed.files[0].payload_sha256.len(), 64);

        let mut malformed = bytes;
        let central = malformed
            .windows(4)
            .position(|window| window == b"PK\x01\x02")
            .expect("fixture has a central directory");
        malformed[central] = 0;
        assert_eq!(
            parse_packed_metadata_archive(&malformed).status,
            "malformed"
        );
    }

    #[test]
    fn binds_only_packed_output_label_declarations() {
        let mut verified_metadata = parse_tflite_model_metadata(&decode_hex(FIXTURE_HEX));
        let verified = bind_packed_associated_files(
            &mut verified_metadata,
            parse_packed_metadata_archive(&prefixed_stored_zip("labels.txt", b"zero\none\n")),
            &[vec![1, 2]],
        );
        assert_eq!(verified.archive_status, "assessed");
        assert_eq!(verified.verified_output_associated_file_count, 1);
        assert_eq!(verified.missing_output_associated_file_count, 0);
        assert_eq!(verified.verified_output_label_file_count, 1);
        assert_eq!(verified.verified_output0_label_file_count, 1);
        assert_eq!(
            verified_metadata.output_associated_files[0].packed_status,
            "verified_payload"
        );
        assert_eq!(
            verified_metadata.output_associated_files[0].cardinality_status,
            "verified_unique_axis_match"
        );
        assert_eq!(
            verified_metadata.output_associated_files[0].label_entry_count,
            Some(2)
        );

        let mut missing_metadata = parse_tflite_model_metadata(&decode_hex(FIXTURE_HEX));
        let missing = bind_packed_associated_files(
            &mut missing_metadata,
            parse_packed_metadata_archive(&decode_hex(FIXTURE_HEX)),
            &[vec![1, 2]],
        );
        assert_eq!(missing.archive_status, "not_present");
        assert_eq!(missing.verified_output_associated_file_count, 0);
        assert_eq!(missing.missing_output_associated_file_count, 0);
        assert_eq!(missing.verified_output_label_file_count, 0);
        assert_eq!(missing.missing_output_label_file_count, 0);
        assert_eq!(missing.invalid_output_label_file_count, 1);
        assert_eq!(missing.verified_output0_label_file_count, 0);
        assert_eq!(
            missing_metadata.output_associated_files[0].packed_status,
            "not_assessed_archive_unavailable"
        );
    }

    #[test]
    fn verifies_deflate_crc_and_rejects_label_cardinality_mismatch() {
        let payload = b"zero\none\n";
        let deflated = prefixed_deflated_zip("labels.txt", payload);
        let parsed = parse_packed_metadata_archive(&deflated);
        assert_eq!(parsed.status, "assessed");
        assert_eq!(parsed.files[0].compression_method, 8);
        assert_eq!(parsed.files[0].payload_status, "verified");
        assert_eq!(parsed.files[0].decoded_bytes, Some(payload.len()));

        let mut metadata = parse_tflite_model_metadata(&decode_hex(FIXTURE_HEX));
        let binding = bind_packed_associated_files(&mut metadata, parsed, &[vec![1, 3]]);
        assert_eq!(binding.verified_output_label_file_count, 0);
        assert_eq!(binding.invalid_output_label_file_count, 1);
        assert_eq!(binding.label_cardinality_mismatch_count, 1);
        assert_eq!(
            metadata.output_associated_files[0].cardinality_status,
            "mismatch_no_output_axis"
        );

        let mut corrupt = prefixed_stored_zip("labels.txt", payload);
        let payload_start = b"TFLite prefix".len() + 30 + "labels.txt".len();
        corrupt[payload_start] ^= 1;
        let corrupt_archive = parse_packed_metadata_archive(&corrupt);
        assert_eq!(corrupt_archive.files[0].payload_status, "crc32_mismatch");
        assert_eq!(corrupt_archive.files[0].crc32_verified, Some(false));
    }

    fn prefixed_stored_zip(name: &str, payload: &[u8]) -> Vec<u8> {
        prefixed_zip(name, payload, 0, payload.to_vec())
    }

    fn prefixed_deflated_zip(name: &str, payload: &[u8]) -> Vec<u8> {
        prefixed_zip(name, payload, 8, compress_to_vec(payload, 6))
    }

    fn prefixed_zip(name: &str, payload: &[u8], method: u16, encoded: Vec<u8>) -> Vec<u8> {
        let mut bytes = b"TFLite prefix".to_vec();
        let local_offset = bytes.len() as u32;
        let crc32 = crc32_ieee(payload);
        push_u32(&mut bytes, 0x0403_4b50);
        push_u16(&mut bytes, 20);
        push_u16(&mut bytes, 0x0800);
        push_u16(&mut bytes, method);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, crc32);
        push_u32(&mut bytes, encoded.len() as u32);
        push_u32(&mut bytes, payload.len() as u32);
        push_u16(&mut bytes, name.len() as u16);
        push_u16(&mut bytes, 0);
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&encoded);

        let central_offset = bytes.len() as u32;
        push_u32(&mut bytes, 0x0201_4b50);
        push_u16(&mut bytes, 20);
        push_u16(&mut bytes, 20);
        push_u16(&mut bytes, 0x0800);
        push_u16(&mut bytes, method);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, crc32);
        push_u32(&mut bytes, encoded.len() as u32);
        push_u32(&mut bytes, payload.len() as u32);
        push_u16(&mut bytes, name.len() as u16);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, local_offset);
        bytes.extend_from_slice(name.as_bytes());
        let central_size = bytes.len() as u32 - central_offset;

        push_u32(&mut bytes, 0x0605_4b50);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 1);
        push_u16(&mut bytes, 1);
        push_u32(&mut bytes, central_size);
        push_u32(&mut bytes, central_offset);
        push_u16(&mut bytes, 0);
        bytes
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
