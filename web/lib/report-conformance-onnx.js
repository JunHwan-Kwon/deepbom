import { ANALYZER_METADATA } from "./report-metadata.js";
import { validateNormalizerRowAgainstEvidence } from "./onnx-ml-normalizer-conformance.js";
import { validateScalerRowAgainstEvidence } from "./onnx-ml-scaler-conformance.js";
import { validateImputerRowAgainstEvidence } from "./onnx-ml-imputer-conformance.js";
import { validateOneHotEncoderRowAgainstEvidence } from "./onnx-ml-one-hot-encoder-conformance.js";
import { validateLinearModelRowAgainstEvidence } from "./onnx-ml-linear-model-conformance.js";
import { validateLabelEncoderRowAgainstEvidence } from "./onnx-ml-label-encoder-conformance.js";
import { validateSvmRowAgainstEvidence } from "./onnx-ml-svm-conformance.js";
import { validateTreeEnsembleRowAgainstEvidence } from "./onnx-ml-tree-ensemble-conformance.js";
import { buildOnnxTreeConformanceFacts, onnxTreeLedgerConserves, onnxTreeMlBomConserves, registerOnnxTreeConformanceChecks } from "./report-conformance-onnx-tree.js";
import { onnxTfIdfMlBomConserves, registerOnnxTfIdfConformanceChecks } from "./report-conformance-onnx-tfidf.js";
import { formatIntegerForConformance, markdownCellForConformance, validateStaticCanonicalTextLedger, validateStaticSignedZeroLedger } from "./report-conformance-common.js";
import { externalLocationStatus, normalizeExternalLocation, nullableClose, onnxTensorPayloadBytes, parseExternalDataDecimal, sameArray, sourceLedgerProblems, tensorElementCount } from "./report-conformance-runtime-helpers.js";
import { exactNonnegativeRatio } from "./exact-rational.js";

const PINNED_ONNX_TENSOR_TYPES = Object.freeze([
  [0, "UNDEFINED", 0], [1, "FLOAT32", 32], [2, "UINT8", 8], [3, "INT8", 8], [4, "UINT16", 16], [5, "INT16", 16],
  [6, "INT32", 32], [7, "INT64", 64], [8, "STRING", 0], [9, "BOOL", 8], [10, "FLOAT16", 16], [11, "FLOAT64", 64],
  [12, "UINT32", 32], [13, "UINT64", 64], [14, "COMPLEX64", 64], [15, "COMPLEX128", 128], [16, "BFLOAT16", 16],
  [17, "FLOAT8E4M3FN", 8], [18, "FLOAT8E4M3FNUZ", 8], [19, "FLOAT8E5M2", 8], [20, "FLOAT8E5M2FNUZ", 8],
  [21, "UINT4", 4], [22, "INT4", 4], [23, "FLOAT4E2M1", 4], [24, "FLOAT8E8M0", 8], [25, "UINT2", 2], [26, "INT2", 2],
]);

const PINNED_ONNX_SHAPE_SOURCES = Object.freeze([
  ["model_opset_import_contract", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/onnx.proto", "a05cfbcd1370608b809c5b84c44e3198d3369036458e0b5f297e76ceaf9c4e1b"],
  ["shape_inference_contract", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/shape_inference.h", "9602d530ea9bf4c1f3d8418c29114c264563ccbf10639ddd82a484fcd8bfc530"],
  ["shape_inference_implementation", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/shape_inference.cc", "3051554a17b7f632d90362aec2987b1e7374b3d69e421160db4f21aeda98363e"],
  ["current_tensor_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/tensor/defs.cc", "22681df3a131c55524dceb8e366dcc24dcce4acbbf198ac7ae5216313e619652"],
  ["historical_tensor_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/tensor/old.cc", "405e2ece240dae4e6a1929eb6786f3e49bf5bfd8c1095280b6835e194c6703a0"],
  ["current_neural_network_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/nn/defs.cc", "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02"],
  ["current_matrix_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/math/defs.cc", "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594"],
  ["current_random_generator_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/generator/defs.cc", "838c87511348b700000f133bf98522bc79f84cea6ff18e09e5f255b28ac183dd"],
  ["current_recurrent_operator_rules", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/rnn/defs.cc", "6262fe369d07727433a7ff49128dfedcc59209958e0edf3050e55bea5a932791"],
  ["operator_schema_history", "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/docs/Changelog.md", "315293e31dd0f415efc7dd821380b53418845f3a719a4930c8df87a30023b6e3"],
  ["ort_standard_domain_extension_schemas", "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/onnxruntime/core/graph/contrib_ops/contrib_defs.cc", "e313a9ec5b8c11620445961c1f36da5ed894f70765ad94771e141de13b3e45ca"],
  ["ort_transformer_contrib_schemas", "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/onnxruntime/contrib_ops/transformers/bert/bert_defs.cc", "5783df1b2d56a6af0ba8a8b228ce165526fc3fcc5d162b00aa047709da621258"],
  ["ort_transformer_shared_shape_functions", "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/onnxruntime/core/graph/contrib_ops/shape_inference_functions.cc", "de6b66a75150dd390f52f9eaf5affe8a846ff5ff8b6641cf353d25e89fdd59db"],
  ["ort_quantized_recurrent_contrib_schema", "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/onnxruntime/core/graph/contrib_ops/quantization_defs.cc", "de215366d115e5b49fbc7ac0bbd09af19738cb05b9278469dd8a0a2537d28d6c"],
]);

export function registerOnnxConformance({
  staticAnalysis, findingsRegister, mlBomDocument, engineeringReport,
  engineeringReportText, compactMlBomEvidence, check,
}) {
    const staticTensors = staticAnalysis?.tensors || [];
    const exactStaticSignedZeroValues = staticTensors.reduce((sum, tensor) => sum + Number(tensor.static_values_negative_zero_count || 0), 0);
    const staticSignedZeroTensors = staticTensors.filter((tensor) => Number(tensor.static_values_negative_zero_count || 0) > 0);
    const staticCanonicalTextTensors = staticTensors.filter((tensor) => tensor.static_values_canonical_text_complete === true);
    const exactStaticCanonicalTextValues = staticCanonicalTextTensors.reduce((sum, tensor) => sum + (tensor.static_values_canonical_texts || []).length, 0);
    check("CF-STATIC-001", validateStaticSignedZeroLedger(staticTensors),
      "Every public ONNX static-value array must be JSON-safe and preserve signed-zero positions through a complete, unique, in-bounds index ledger.", ["/evidence/static_analysis/tensors"]);
    check("CF-STATIC-002", validateStaticCanonicalTextLedger(staticTensors),
      "Every canonical-text static ledger must conserve initializer cardinality and remain exclusive with the JSON-safe numeric array.", ["/evidence/static_analysis/tensors"]);
    const quantizationBinding = staticAnalysis?.onnx_quantization_binding || {};
    const annotationBindings = (quantizationBinding.bindings || []).filter((item) => item.binding_source === "graph_quantization_annotation");
    const validAnnotations = annotationBindings.filter((item) => item.status === "pass").length;
    const invalidAnnotations = annotationBindings.filter((item) => item.status === "fail").length;
    const unresolvedAnnotations = annotationBindings.filter((item) => String(item.status || "").startsWith("not_assessed")).length;
    check("CF-ONNX-QUANT-ANNOTATION-001", quantizationBinding.schema === "deepbom.onnx_quantization_binding.v1.1"
      && Number(quantizationBinding.main_graph_annotation_count || 0) === annotationBindings.length
      && Number(quantizationBinding.valid_annotation_count || 0) === validAnnotations
      && Number(quantizationBinding.invalid_annotation_count || 0) === invalidAnnotations
      && Number(quantizationBinding.unresolved_annotation_count || 0) === unresolvedAnnotations
      && Number(quantizationBinding.nested_graph_annotation_count || 0) === (quantizationBinding.nested_graph_annotations || []).length
      && ["all_serialized_graph_annotations_bound", "main_graph_bound_nested_graph_annotations_inventoried_not_bound"].includes(quantizationBinding.annotation_scope_status)
      && /^https:\/\/raw\.githubusercontent\.com\/onnx\/onnx\/[a-f0-9]{40}\//.test(quantizationBinding.annotation_source_ref || "")
      && /^[a-f0-9]{64}$/.test(quantizationBinding.annotation_source_sha256 || "")
      && annotationBindings.every((item) => item.source_ref === quantizationBinding.annotation_source_ref
        && item.source_sha256 === quantizationBinding.annotation_source_sha256
        && item.op_index === null && item.op_name === "GraphProto.TensorAnnotation"
        && ["DERIVED", "NOT_ASSESSABLE"].includes(item.evidence_class)),
    "ONNX GraphProto TensorAnnotation bindings must conserve main/nested scope counts, status classes, and pinned source identity.", ["/evidence/static_analysis/onnx_quantization_binding"]);
    check("CF-ONNX-QUANT-ANNOTATION-002", engineeringReportText.includes(quantizationBinding.schema)
      && engineeringReportText.includes(quantizationBinding.annotation_scope_status)
      && engineeringReportText.includes(quantizationBinding.annotation_source_ref)
      && engineeringReportText.includes(quantizationBinding.annotation_source_sha256)
      && annotationBindings.every((item) => engineeringReportText.includes(item.tensor_name || "unnamed")
        && engineeringReportText.includes(item.status)
        && engineeringReportText.includes(item.operator_cross_check_status)),
    "Engineering Report must render every main-graph ONNX TensorAnnotation contract and its pinned source/scope boundary.", ["/evidence/static_analysis/onnx_quantization_binding", "/engineering_report.md"]);
    const metadata = staticAnalysis?.metadata_presence || {};
    const metadataProperties = metadata.metadata_properties || [];
    const metadataTextBytes = [
      metadata.producer_name,
      metadata.producer_version,
      metadata.model_domain,
      metadata.model_doc_string,
      staticAnalysis?.graph_name,
      metadata.graph_doc_string,
      ...metadataProperties.flatMap((entry) => [entry.key, entry.value]),
    ].reduce((total, value) => total + new TextEncoder().encode(String(value || "")).byteLength, 0);
    check("CF-METADATA-001", metadata.schema === ANALYZER_METADATA.schemas.artifactMetadata
      && metadata.format === "onnx" && metadata.status === "assessed"
      && Number(metadata.metadata_property_count || 0) === metadataProperties.length
      && metadataProperties.every((entry) => ["model", "graph"].includes(entry.scope))
      && Number(metadata.metadata_text_bytes || 0) === metadataTextBytes
      && metadata.documented_preprocessing === false
      && metadata.output_semantics_documented === false
      && Number(metadata.output_label_file_count || 0) === 0,
    "ONNX metadata ledger must conserve decoded ModelProto/GraphProto text without promoting untyped properties to preprocessing or output-label contracts.", ["/evidence/static_analysis/metadata_presence"]);
    check("CF-METADATA-002", engineeringReportText.includes(metadata.schema)
      && engineeringReportText.includes(metadata.preprocessing_contract_status)
      && metadataProperties.slice(0, 24).every((entry) => engineeringReportText.includes(entry.key) && engineeringReportText.includes(entry.value)),
    "Engineering report must preserve the ONNX metadata schema, preprocessing evidence boundary, and rendered property rows.", ["/evidence/static_analysis/metadata_presence", "/engineering_report.md"]);
    const dataTypeContract = staticAnalysis?.onnx_tensor_data_type_contract || {};
    const dataTypeRows = dataTypeContract.types || [];
    const expectedDataTypeRows = PINNED_ONNX_TENSOR_TYPES.map(([id, name, storageBits]) => ({ id, name, storage_bits: storageBits }));
    check("CF-ONNX-DTYPE-001", dataTypeContract.schema === ANALYZER_METADATA.schemas.onnxTensorDataTypeContract
      && dataTypeContract.status === "complete_for_pinned_onnx_release"
      && dataTypeContract.evidence_class === "SOURCE_PINNED_AND_IMPLEMENTATION_TESTED"
      && dataTypeContract.source_release === "v1.21.0"
      && dataTypeContract.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && dataTypeContract.source_sha256 === "f4cbc198df3a0f3f4519d4d38cd2262e8f84057583b7313e2d0f981b3f68c213"
      && dataTypeContract.source_ref?.endsWith("/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/onnx.in.proto")
      && Number(dataTypeContract.concrete_data_type_count) === 26
      && Number(dataTypeContract.fixed_width_numeric_data_type_count) === 25
      && Number(dataTypeContract.raw_numeric_decoder_count) === 25
      && Number(dataTypeContract.typed_numeric_decoder_count) === 25
      && Number(dataTypeContract.packed_data_type_count) === 5
      && JSON.stringify(dataTypeRows) === JSON.stringify(expectedDataTypeRows)
      && JSON.stringify(dataTypeContract.packed_data_types) === JSON.stringify(["UINT4", "INT4", "FLOAT4E2M1", "UINT2", "INT2"]),
    "ONNX TensorProto dtype IDs, bit widths, packed types, or pinned source identity differ from the independent ONNX 1.21 contract.", ["/evidence/static_analysis/onnx_tensor_data_type_contract"]);
    check("CF-ONNX-DTYPE-002", engineeringReportText.includes("## ONNX TensorProto Data-Type Contract")
      && engineeringReportText.includes(dataTypeContract.schema)
      && engineeringReportText.includes(dataTypeContract.source_sha256)
      && engineeringReportText.includes(dataTypeContract.packing_rule)
      && expectedDataTypeRows.filter((row) => row.id > 0).every((row) => engineeringReportText.includes(`${row.id}:${row.name}/${row.storage_bits}b`)),
    "Engineering report must preserve the pinned TensorProto dtype source, complete type inventory, and packed-storage rule.", ["/evidence/static_analysis/onnx_tensor_data_type_contract", "/engineering_report.md"]);
    const mlBomProperties = mlBomDocument?.properties || [];
    const mlBomValue = (name) => mlBomProperties.find((item) => item.name === name)?.value;
    check("CF-ONNX-DTYPE-003", compactMlBomEvidence || mlBomValue("deepbom:model:onnxTensorDataTypeContractSchema") === dataTypeContract.schema
      && mlBomValue("deepbom:model:onnxTensorDataTypeSourceCommit") === dataTypeContract.source_commit
      && mlBomValue("deepbom:model:onnxTensorDataTypeSourceSha256") === dataTypeContract.source_sha256
      && mlBomValue("deepbom:model:onnxConcreteDataTypeCount") === String(dataTypeContract.concrete_data_type_count)
      && mlBomValue("deepbom:model:onnxPackedDataTypeCount") === String(dataTypeContract.packed_data_type_count)
      && mlBomValue("deepbom:model:onnxTensorPackingRule") === dataTypeContract.packing_rule,
    "ML-BOM must preserve the pinned ONNX TensorProto dtype and packing contract.", ["/evidence/static_analysis/onnx_tensor_data_type_contract", "/evidence/mlbom_cyclonedx"]);
    const externalData = staticAnalysis?.onnx_external_data || {};
    const externalRows = externalData.tensors || [];
    const sourceExternalTensors = externalData.source_tensor_declarations || [];
    const externalEntryCount = externalRows.reduce((total, row) => total + Number(row.entry_count || 0), 0);
    const malformedExternalCount = externalRows.filter((row) => row.reference_status === "malformed_reference").length;
    const unsafeExternalCount = externalRows.filter((row) => externalLocationStatus(row.location) !== "safe_relative_path").length;
    const missingExternalLocationCount = externalRows.filter((row) => !row.location).length;
    const duplicateExternalKeyCount = externalRows.reduce((total, row) => total + (row.duplicate_keys || []).length, 0);
    const invalidExternalRangeCount = externalRows.filter((row) => row.offset_status === "invalid" || row.length_status === "invalid" || (row.length != null && row.range_end == null)).length;
    const invalidExternalChecksumCount = externalRows.filter((row) => row.checksum_status === "invalid_declaration").length;
    const embeddedExternalConflictCount = externalRows.filter((row) => row.embedded_payload_conflict === true).length;
    const externalDataLocationMismatchCount = externalRows.filter((row) => row.data_location_mismatch === true).length;
    const fullyDeclaredExternalBytes = externalRows.length > 0 && externalRows.every((row) => row.length_status === "declared")
      ? externalRows.reduce((total, row) => total + Number(row.length || 0), 0) : null;
    const suppliedExternalRows = externalRows.filter((row) => Boolean(row.sidecar_path));
    const verifiedExternalRows = externalRows.filter((row) => row.payload_status === "verified");
    const failedExternalRows = externalRows.filter((row) => row.reference_status === "payload_verification_failed");
    const externalRangeFailureCount = externalRows.filter((row) => row.payload_status === "range_out_of_bounds").length;
    const externalSizeFailureCount = externalRows.filter((row) => row.payload_status === "payload_size_mismatch").length;
    const externalChecksumFailureCount = externalRows.filter((row) => row.payload_status === "checksum_mismatch").length;
    const verifiedExternalBytes = verifiedExternalRows.reduce((total, row) => total + Number(row.payload_bytes || 0), 0);
    const suppliedExternalFiles = Array.isArray(externalData.supplied_files) ? externalData.supplied_files : [];
    const uniqueExternalFilePaths = new Set(suppliedExternalFiles.map((file) => String(file.path || "")));
    const suppliedFileLedgerValid = uniqueExternalFilePaths.size === suppliedExternalFiles.length
      && suppliedExternalFiles.every((file) => externalLocationStatus(file.path) === "safe_relative_path"
        && Number.isSafeInteger(file.byte_length) && file.byte_length >= 0
        && /^[a-f0-9]{64}$/.test(file.sha256 || "")
        && (!file.sha1 || /^[a-f0-9]{40}$/.test(file.sha1))
        && typeof file.used === "boolean");
    const externalRowsMatchSource = externalRows.every((row) => {
      const source = sourceExternalTensors.find((tensor) => tensor.scope === row.scope
        && tensor.tensor_role === row.tensor_role && tensor.name === row.tensor_name);
      const entries = source?.external_data || [];
      const byKey = new Map();
      const seen = new Set();
      const duplicateKeys = [];
      for (const entry of entries) {
        if (seen.has(entry.key)) duplicateKeys.push(entry.key);
        else {
          seen.add(entry.key);
          byKey.set(entry.key, entry.value);
        }
      }
      const parsedOffset = parseExternalDataDecimal(byKey.get("offset"), 0);
      const parsedLength = parseExternalDataDecimal(byKey.get("length"), null);
      const rangeEnd = parsedOffset.value == null || parsedLength.value == null ? null : parsedOffset.value + parsedLength.value;
      const checksum = String(byKey.get("checksum") || "");
      const normalizedLocation = normalizeExternalLocation(byKey.get("location"));
      const expectedBytes = onnxTensorPayloadBytes(source?.dtype, tensorElementCount(source?.shape));
      const malformed = !byKey.get("location") || duplicateKeys.length > 0
        || externalLocationStatus(byKey.get("location")) !== "safe_relative_path"
        || !parsedOffset.valid || !parsedLength.valid || (checksum && !/^[a-f0-9]{40}$/i.test(checksum))
        || source?.external_embedded_payload_conflict === true || Number(source?.data_location || 0) !== 1
        || (rangeEnd != null && !Number.isSafeInteger(rangeEnd));
      const file = suppliedExternalFiles.find((candidate) => candidate.path === row.sidecar_path);
      const verified = row.payload_status === "verified";
      const supplied = Boolean(row.sidecar_path);
      const payloadStateValid = malformed
        ? row.reference_status === "malformed_reference" && row.payload_status === "malformed_reference" && !supplied
        : verified
          ? row.reference_status === "verified_reference_and_payload"
            && source?.external_payload_verified === true
            && Number(row.payload_bytes) === expectedBytes
            && Number(source?.initializer_bytes || 0) === expectedBytes
            && Number(source?.initializer_raw_data_bytes || 0) === expectedBytes
            && Boolean(file) && Number(row.sidecar_bytes) === Number(file.byte_length)
            && row.sidecar_sha256 === file.sha256 && row.sidecar_sha1 === file.sha1
            && Number(row.offset || 0) + Number(row.payload_bytes || 0) <= Number(row.sidecar_bytes)
            && (!checksum || row.checksum_status === "verified" && row.sidecar_sha1 === checksum.toLowerCase())
          : supplied
            ? row.reference_status === "payload_verification_failed"
              && source?.external_payload_verified === false
              && ["range_out_of_bounds", "payload_size_mismatch", "checksum_mismatch"].includes(row.payload_status)
              && Boolean(file) && row.sidecar_sha256 === file.sha256 && row.sidecar_sha1 === file.sha1
            : row.reference_status === "declared_payload_not_supplied"
              && row.payload_status === "not_supplied" && source?.external_payload_verified === false;
      return Boolean(source)
        && Number(row.entry_count || 0) === entries.length
        && JSON.stringify(row.entries || []) === JSON.stringify(entries)
        && JSON.stringify(row.duplicate_keys || []) === JSON.stringify(duplicateKeys)
        && row.location === String(byKey.get("location") || "")
        && row.normalized_location === normalizedLocation
        && row.location_status === externalLocationStatus(byKey.get("location"))
        && row.offset === parsedOffset.value && row.offset_status === parsedOffset.status
        && row.length === parsedLength.value && row.length_status === parsedLength.status
        && row.range_end === (rangeEnd != null && Number.isSafeInteger(rangeEnd) ? rangeEnd : null)
        && row.expected_payload_bytes === expectedBytes
        && row.checksum === checksum
        && row.embedded_payload_conflict === (source.external_embedded_payload_conflict === true)
        && row.data_location_mismatch === (Number(source.data_location || 0) !== 1)
        && payloadStateValid;
    });
    const expectedExternalStatus = externalRows.length === 0 ? "assessed_absent"
      : malformedExternalCount > 0 ? "malformed_reference"
        : failedExternalRows.length > 0 ? "payload_verification_failed"
          : verifiedExternalRows.length === externalRows.length ? "verified_payloads"
            : verifiedExternalRows.length > 0 ? "partial_payload_coverage" : "not_assessed_payload_not_supplied";
    const expectedExternalEvidenceClass = externalRows.length === 0 ? "OBSERVED"
      : verifiedExternalRows.length === externalRows.length ? "OBSERVED/DERIVED" : "OBSERVED/NOT_ASSESSABLE";
    check("CF-ONNX-EXTERNAL-001", externalData.schema === ANALYZER_METADATA.schemas.onnxExternalData
      && Number(externalData.tensor_count || 0) === externalRows.length
      && Number(externalData.source_tensor_declaration_count || 0) === sourceExternalTensors.length
      && externalRows.length === sourceExternalTensors.length
      && Number(externalData.entry_count || 0) === externalEntryCount
      && Number(externalData.malformed_reference_count || 0) === malformedExternalCount
      && Number(externalData.unsafe_location_count || 0) === unsafeExternalCount
      && Number(externalData.missing_location_count || 0) === missingExternalLocationCount
      && Number(externalData.duplicate_key_count || 0) === duplicateExternalKeyCount
      && Number(externalData.invalid_range_count || 0) === invalidExternalRangeCount
      && Number(externalData.invalid_checksum_count || 0) === invalidExternalChecksumCount
      && Number(externalData.embedded_payload_conflict_count || 0) === embeddedExternalConflictCount
      && Number(externalData.data_location_mismatch_count || 0) === externalDataLocationMismatchCount
      && externalData.declared_payload_bytes === fullyDeclaredExternalBytes
      && Number(externalData.supplied_payload_count || 0) === suppliedExternalRows.length
      && Number(externalData.verified_payload_count || 0) === verifiedExternalRows.length
      && Number(externalData.payload_verification_failed_count || 0) === failedExternalRows.length
      && Number(externalData.range_out_of_bounds_count || 0) === externalRangeFailureCount
      && Number(externalData.payload_size_mismatch_count || 0) === externalSizeFailureCount
      && Number(externalData.checksum_mismatch_count || 0) === externalChecksumFailureCount
      && Number(externalData.verified_payload_bytes || 0) === verifiedExternalBytes
      && Number(externalData.supplied_file_count || 0) === suppliedExternalFiles.length
      && Number(externalData.supplied_file_bytes || 0) === suppliedExternalFiles.reduce((total, file) => total + Number(file.byte_length || 0), 0)
      && Number(externalData.used_file_count || 0) === suppliedExternalFiles.filter((file) => file.used).length
      && Number(externalData.unused_file_count || 0) === suppliedExternalFiles.filter((file) => !file.used).length
      && suppliedFileLedgerValid
      && externalRowsMatchSource
      && externalData.status === expectedExternalStatus
      && externalData.evidence_class === expectedExternalEvidenceClass,
    "ONNX external_data references, sidecar identities, range/checksum verification, and payload coverage do not conserve from parsed all-scope TensorProto evidence.", ["/evidence/static_analysis/onnx_external_data"]);
    check("CF-ONNX-EXTERNAL-002", engineeringReportText.includes(externalData.schema)
      && engineeringReportText.includes(externalData.status)
      && engineeringReportText.includes(externalData.detail)
      && externalRows.slice(0, 24).every((row) => engineeringReportText.includes(row.scope)
        && engineeringReportText.includes(row.tensor_role)
        && engineeringReportText.includes(row.tensor_name)
        && engineeringReportText.includes(row.location || "(missing)")
        && (!row.normalized_location || engineeringReportText.includes(row.normalized_location))
        && (!row.sidecar_sha256 || engineeringReportText.includes(row.sidecar_sha256))),
    "Engineering report must preserve ONNX external_data coverage and every rendered reference row.", ["/evidence/static_analysis/onnx_external_data", "/engineering_report.md"]);
    const externalFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0004");
    const externalPayloadIncomplete = verifiedExternalRows.length < externalRows.length || failedExternalRows.length > 0;
    check("CF-ONNX-EXTERNAL-003", externalPayloadIncomplete
      ? Boolean(externalFinding)
        && externalFinding.technical_priority === "High"
        && String(externalFinding.observation || "").includes(`${verifiedExternalRows.length.toLocaleString("en-US")} verified`)
        && (externalFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_external_data")
      : !externalFinding,
    "EA-ONX-0004 must exist exactly when external initializer payload coverage is incomplete or verification failed.", ["/evidence/static_analysis/onnx_external_data", "/evidence/findings_register/findings"]);
    const typeProtoContract = staticAnalysis?.onnx_type_proto_contract || {};
    const typeProtoRows = typeProtoContract.rows || [];
    const declaredTypeRows = typeProtoRows.filter((row) => row.status !== "not_declared");
    const invalidTypeProtoRows = typeProtoRows.filter((row) => row.status === "fail");
    const typeKindCount = (typeProtoContract.kind_counts || []).reduce((sum, row) => sum + Number(row.count || 0), 0);
    check("CF-ONNX-TYPE-001", typeProtoContract.schema === ANALYZER_METADATA.schemas.onnxTypeProtoContract
      && typeProtoContract.source_release === "v1.21.0"
      && typeProtoContract.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && /^[0-9a-f]{64}$/.test(typeProtoContract.source_sha256 || "")
      && typeProtoRows.length === Number(typeProtoContract.declaration_count || 0)
      && declaredTypeRows.length === Number(typeProtoContract.declared_type_count || 0)
      && typeProtoRows.filter((row) => row.status === "not_declared").length === Number(typeProtoContract.undeclared_optional_type_count || 0)
      && typeProtoRows.filter((row) => row.status === "pass").length === Number(typeProtoContract.valid_type_count || 0)
      && invalidTypeProtoRows.length === Number(typeProtoContract.invalid_type_count || 0)
      && typeKindCount === declaredTypeRows.length
      && typeProtoRows.reduce((sum, row) => sum + Number(row.type_node_count || 0), 0) === Number(typeProtoContract.type_node_count || 0)
      && typeProtoRows.reduce((sum, row) => sum + Number(row.type_text_bytes || 0), 0) === Number(typeProtoContract.type_text_bytes || 0)
      && typeProtoRows.reduce((sum, row) => sum + Number(row.symbolic_dimension_count || 0), 0) === Number(typeProtoContract.symbolic_dimension_count || 0)
      && typeProtoRows.reduce((sum, row) => sum + Number(row.unknown_dimension_count || 0), 0) === Number(typeProtoContract.unknown_dimension_count || 0)
      && typeProtoContract.status === (invalidTypeProtoRows.length ? "fail" : "assessed"),
    "ONNX recursive TypeProto declaration, kind, source, or count ledger does not conserve.", ["/evidence/static_analysis/onnx_type_proto_contract"]);
    check("CF-ONNX-TYPE-002", engineeringReportText.includes("## ONNX TypeProto Contract")
      && engineeringReportText.includes(typeProtoContract.schema)
      && engineeringReportText.includes(typeProtoContract.source_commit)
      && engineeringReportText.includes(typeProtoContract.source_sha256)
      && (!invalidTypeProtoRows.length || engineeringReportText.includes("Invalid TypeProto Declarations")),
    "Engineering report does not preserve the ONNX TypeProto source, summary, or invalid declaration rows.", ["/evidence/static_analysis/onnx_type_proto_contract", "/engineering_report.md"]);
    const typeProtoFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0010");
    check("CF-ONNX-TYPE-003", invalidTypeProtoRows.length
      ? Boolean(typeProtoFinding) && typeProtoFinding.technical_priority === "High"
        && (typeProtoFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_type_proto_contract/invalid_rows")
      : !typeProtoFinding,
    "EA-ONX-0010 must exist exactly when a pinned ONNX TypeProto declaration is invalid.", ["/evidence/static_analysis/onnx_type_proto_contract/invalid_rows", "/evidence/findings_register/findings"]);

    const sparseContract = staticAnalysis?.onnx_sparse_tensor_contract || {};
    const sparseRows = sparseContract.rows || [];
    const invalidSparseRows = sparseRows.filter((row) => row.status === "fail");
    const partialSparseRows = sparseRows.filter((row) => row.status === "partial");
    const sparseNnzTotal = sparseRows.every((row) => row.nnz != null) ? sparseRows.reduce((sum, row) => sum + Number(row.nnz), 0) : null;
    const sparseDenseElementTotal = sparseRows.every((row) => row.dense_logical_elements != null) ? sparseRows.reduce((sum, row) => sum + Number(row.dense_logical_elements), 0) : null;
    const sparseExternalComponents = sparseRows.reduce((sum, row) => sum + Number(row.external_payload_component_count || 0), 0);
    const sparseVerifiedExternalComponents = sparseRows.reduce((sum, row) => sum + Number(row.verified_external_payload_component_count || 0), 0);
    check("CF-ONNX-SPARSE-001", sparseContract.schema === ANALYZER_METADATA.schemas.onnxSparseTensorContract
      && sparseContract.source_release === "v1.21.0"
      && sparseContract.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && /^[0-9a-f]{64}$/.test(sparseContract.source_sha256 || "")
      && sparseRows.length === Number(sparseContract.sparse_tensor_count || 0)
      && sparseRows.filter((row) => row.status === "pass").length === Number(sparseContract.valid_sparse_tensor_count || 0)
      && partialSparseRows.length === Number(sparseContract.partially_assessed_sparse_tensor_count || 0)
      && invalidSparseRows.length === Number(sparseContract.invalid_sparse_tensor_count || 0)
      && sparseNnzTotal === sparseContract.declared_nnz_total
      && sparseDenseElementTotal === sparseContract.dense_logical_element_total
      && sparseExternalComponents === Number(sparseContract.external_payload_component_count || 0)
      && sparseVerifiedExternalComponents === Number(sparseContract.verified_external_payload_component_count || 0)
      && sparseRows.filter((row) => ["assessed", "fail"].includes(row.index_content_status)).length === Number(sparseContract.index_content_assessed_sparse_tensor_count || 0)
      && sparseRows.filter((row) => row.index_content_status === "fail").length === Number(sparseContract.index_content_failed_sparse_tensor_count || 0)
      && sparseRows.filter((row) => String(row.index_content_status || "").startsWith("not_assessed")).length === Number(sparseContract.index_content_unassessed_sparse_tensor_count || 0)
      && sparseRows.reduce((sum, row) => sum + Number(row.assessed_index_count || 0), 0) === Number(sparseContract.assessed_index_count || 0)
      && sparseRows.reduce((sum, row) => sum + Number(row.out_of_bounds_index_count || 0), 0) === Number(sparseContract.out_of_bounds_index_count || 0)
      && sparseRows.reduce((sum, row) => sum + Number(row.duplicate_index_count || 0), 0) === Number(sparseContract.duplicate_index_count || 0)
      && sparseRows.reduce((sum, row) => sum + Number(row.unsorted_index_count || 0), 0) === Number(sparseContract.unsorted_index_count || 0)
      && sparseContract.status === (invalidSparseRows.length ? "fail" : partialSparseRows.length ? "partial" : "assessed"),
    "ONNX SparseTensorProto structure, payload, index-content, source, or count ledger does not conserve.", ["/evidence/static_analysis/onnx_sparse_tensor_contract"]);
    check("CF-ONNX-SPARSE-002", engineeringReportText.includes("## ONNX SparseTensorProto Contract")
      && engineeringReportText.includes(sparseContract.schema)
      && engineeringReportText.includes(sparseContract.source_commit)
      && engineeringReportText.includes(sparseContract.source_sha256)
      && (!invalidSparseRows.length || engineeringReportText.includes("Invalid SparseTensorProto Records")),
    "Engineering report does not preserve the ONNX SparseTensorProto source, index coverage, or invalid rows.", ["/evidence/static_analysis/onnx_sparse_tensor_contract", "/engineering_report.md"]);
    const sparseFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0011");
    check("CF-ONNX-SPARSE-003", invalidSparseRows.length
      ? Boolean(sparseFinding) && sparseFinding.technical_priority === "High"
        && (sparseFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_sparse_tensor_contract/invalid_rows")
      : !sparseFinding,
    "EA-ONX-0011 must exist exactly when a pinned ONNX SparseTensorProto record is invalid.", ["/evidence/static_analysis/onnx_sparse_tensor_contract/invalid_rows", "/evidence/findings_register/findings"]);
    const shapeInference = staticAnalysis?.onnx_shape_inference || null;
    const shapeHistogramCount = (shapeInference?.rule_unsupported_op_histogram || [])
      .reduce((total, item) => total + Number(item.count || 0), 0);
    const shapeUnresolvedRows = shapeInference?.rule_unresolved_nodes || [];
    const shapeConflicts = shapeInference?.declaration_conflicts || [];
    const semanticShapeConflicts = shapeInference?.semantic_contract_conflicts || [];
    const conditionallyInvalidShapeOutputs = Number(shapeInference?.conditionally_invalid_node_output_count || 0);
    const shapeSources = shapeInference?.source_documents || [];
    const shapeSchemaRows = shapeInference?.schema_form_rows || [];
    const invalidShapeSchemaRows = shapeSchemaRows.filter((row) => row.status === "fail");
    const unresolvedShapeSchemaRows = shapeSchemaRows.filter((row) => row.status === "unresolved");
    const validShapeSchemaRows = shapeSchemaRows.filter((row) => row.status === "pass");
    const shapeScope = shapeInference?.shape_scope || {};
    const shapeScopeExclusions = shapeScope.exclusions || [];
    const shapeScopeRows = shapeScope.scope_execution_rows || [];
    const reachableNestedGraphScopeRows = shapeScopeRows.filter((row) => row.scope_class === "nested_graph");
    const reachableLocalFunctionScopeRows = shapeScopeRows.filter((row) => row.scope_class === "local_function_body");
    const extendedShape = shapeInference?.extended_scope_inference || {};
    const extendedFunctionRows = extendedShape.function_call_rows || [];
    const extendedControlRows = extendedShape.control_flow_rows || [];
    const extendedLoopRows = extendedControlRows.filter((row) => row.op_name === "Loop");
    const extendedSequenceMapRows = extendedShape.sequence_map_rows || [];
    const extendedScopeRows = extendedShape.scope_rows || [];
    const extendedSources = extendedShape.source_documents || [];
    const intrinsicCostVariants = extendedScopeRows.flatMap((row) => row.intrinsic_cost_variants || []);
    const validExactMirror = (decimal, value) => {
      if (!/^(0|[1-9][0-9]*)$/.test(String(decimal ?? ""))) return false;
      const exact = BigInt(decimal);
      return exact <= BigInt(Number.MAX_SAFE_INTEGER) ? value === Number(exact) : value == null;
    };
    const validIntrinsicCost = (cost) => Boolean(cost
      && cost.schema === "deepbom.onnx_scope_intrinsic_cost.v1"
      && ["assessed", "partial"].includes(cost.status)
      && cost.evidence_class === "SOURCE_PINNED_AND_DERIVED"
      && cost.source_release === "v1.21.0"
      && cost.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && Array.isArray(cost.source_documents) && cost.source_documents.length === 2
      && new Map(cost.source_documents.map((source) => [source.role, source.sha256])).get("neural_network_operator_schemas") === "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02"
      && new Map(cost.source_documents.map((source) => [source.role, source.sha256])).get("matrix_operator_schemas") === "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594"
      && cost.source_documents.every((source) => /^https:\/\/raw\.githubusercontent\.com\/onnx\/onnx\/[0-9a-f]{40}\/onnx\/defs\/(nn|math)\/defs\.cc$/.test(source.source_ref || ""))
      && Number.isSafeInteger(cost.operator_count) && cost.operator_count >= 0
      && Number.isSafeInteger(cost.mac_compute_operator_count) && cost.mac_compute_operator_count >= 0
      && Number.isSafeInteger(cost.assessed_nominal_mac_operator_count) && cost.assessed_nominal_mac_operator_count >= 0
      && Number.isSafeInteger(cost.unassessed_nominal_mac_operator_count) && cost.unassessed_nominal_mac_operator_count >= 0
      && cost.assessed_nominal_mac_operator_count + cost.unassessed_nominal_mac_operator_count === cost.mac_compute_operator_count
      && Number.isSafeInteger(cost.assessed_operator_io_count) && cost.assessed_operator_io_count >= 0
      && Number.isSafeInteger(cost.unassessed_operator_io_count) && cost.unassessed_operator_io_count >= 0
      && cost.assessed_operator_io_count + cost.unassessed_operator_io_count === cost.operator_count
      && Array.isArray(cost.mac_residuals) && cost.mac_residuals.length === cost.unassessed_nominal_mac_operator_count
      && Array.isArray(cost.payload_residuals) && cost.payload_residuals.length === cost.unassessed_operator_io_count
      && cost.status === (cost.mac_residuals.length || cost.payload_residuals.length ? "partial" : "assessed")
      && validExactMirror(cost.assessed_nominal_macs_decimal, cost.assessed_nominal_macs)
      && validExactMirror(cost.assessed_operator_io_payload_bytes_decimal, cost.assessed_operator_io_payload_bytes)
      && (cost.mac_residuals.length
        ? cost.complete_nominal_macs == null && cost.complete_nominal_macs_decimal == null
        : validExactMirror(cost.complete_nominal_macs_decimal, cost.complete_nominal_macs)
          && cost.complete_nominal_macs_decimal === cost.assessed_nominal_macs_decimal)
      && (cost.payload_residuals.length
        ? cost.complete_operator_io_payload_bytes == null && cost.complete_operator_io_payload_bytes_decimal == null
        : validExactMirror(cost.complete_operator_io_payload_bytes_decimal, cost.complete_operator_io_payload_bytes)
          && cost.complete_operator_io_payload_bytes_decimal === cost.assessed_operator_io_payload_bytes_decimal));
    const expectedExtendedStatus = extendedFunctionRows.some((row) => row.status === "fail")
      || extendedControlRows.some((row) => row.status === "fail")
      || extendedSequenceMapRows.some((row) => row.status === "fail")
      || extendedScopeRows.some((row) => row.status === "fail")
      ? "fail"
      : extendedControlRows.some((row) => row.status === "partial")
        || extendedSequenceMapRows.some((row) => row.status === "partial")
        || extendedScopeRows.some((row) => row.status === "partial") ? "partial" : "assessed";
    const containerInference = shapeInference?.container_value_inference || {};
    const containerRows = containerInference.rows || [];
    const containerFailedRows = containerInference.failed_rows || [];
    const containerPartialRows = containerInference.partial_rows || [];
    const containerSources = containerInference.source_documents || [];
    const containerPassRows = containerRows.filter((row) => row.status === "pass");
    const independentlyExactSequenceLengths = containerRows.reduce((sum, row) => sum
      + (row.sequence_lengths || []).filter((value) => Number.isSafeInteger(value) && value >= 0).length, 0);
    const independentlyExactOptionalPresence = containerRows.reduce((sum, row) => sum
      + (row.optional_presence || []).filter((value) => typeof value === "boolean").length, 0);
    const validContainerRows = containerRows.every((row) => String(row.scope || "").length > 0
      && Number.isSafeInteger(row.node_index) && row.node_index >= 0
      && String(row.op_name || "").length > 0
      && Number.isSafeInteger(row.imported_opset) && row.imported_opset > 0
      && ["pass", "partial", "fail"].includes(row.status)
      && Array.isArray(row.input_names) && Array.isArray(row.output_names)
      && Array.isArray(row.output_kinds) && Array.isArray(row.canonical_output_types)
      && Array.isArray(row.sequence_lengths) && Array.isArray(row.optional_presence)
      && Array.isArray(row.reason_codes)
      && row.output_kinds.length === row.canonical_output_types.length
      && row.output_kinds.length === row.sequence_lengths.length
      && row.output_kinds.length === row.optional_presence.length
      && (row.status === "pass" ? row.reason_codes.length === 0 : row.reason_codes.length > 0)
      && (row.status === "fail" ? row.output_kinds.length === 0 : row.output_kinds.length <= row.output_names.length));
    const tfidfInference = shapeInference?.tfidf_vectorizer_inference || {};
    const mlValueInference = shapeInference?.ml_value_inference || {};
    const mlValueRows = mlValueInference.rows || [];
    const mlValueFailedRows = mlValueInference.failed_rows || [];
    const mlValuePartialRows = mlValueInference.partial_rows || [];
    const mlValuePassRows = mlValueRows.filter((row) => row.status === "pass");
    const mlValueSources = mlValueInference.source_documents || [];
    const expectedMlValueSources = new Map([
      ["traditional_ml_operator_schema", "4588b9efe493ea820d54c5b65b6af6ad8a7860f625f97f5d6edcfd5bf06125e6"],
      ["traditional_ml_schema_history", "fa3d663df091a0cadc85d902a12d84d7465bfec8cf7433861f82b99f921278a4"],
      ["traditional_ml_historical_operator_schema", "6cb50c9e803a7295e5581ee3416e8098115806627a7248791eb3730750a8a94f"],
      ["tree_ensemble_reference", "eaae9d894a32acab1a1e3a0a847ec87dc1f17b57837af0b98d925747b276d47d"],
      ["tree_ensemble_classifier_reference", "987217215e7ef6db4855e1d8bdbf15e81fff86aa4de5d99d4aee64398cfa2294"],
      ["tree_ensemble_regressor_reference", "01ee6cae9715d9aef6a1e25a63004f1cf5120f83f7b314af8035ee7e5090ba29"],
      ["tree_ensemble_legacy_reference_helper", "feee50e93820f12f949175d3b878a63ce54bdf9e9a9744fe94cac5ad70f5f619"],
    ]);
    const mlValueRuntimeSources = mlValueInference.runtime_reference_documents || [];
    const expectedMlValueRuntimeSources = new Map([
      ["ort_cpu_binarizer_kernel", "5253671998fd3c5493d2c8acfcca34ba5747b575f05c635ad7533696a53a43bf"],
      ["ort_cpu_binarizer_contract", "cbd925dc84bc46d13bf4193bbe3e00a28d485bf1666fa2ebfe51476b1ae653c6"],
      ["ort_cpu_binarizer_tests", "b39121bd0431449e6e2d99b9c2cc41085eb0ea70f21930807bdc80df9d2cf1f0"],
      ["ort_cpu_normalizer_kernel", "50b0a8eb826fd730b3c895f5493d36a8c12e477e9b91337b10170413b73af20c"],
      ["ort_cpu_normalizer_contract", "c8742d10e18154d83e20482c7e57b1263d427e0a5b167f3c4a710bfaf5d4310c"],
      ["ort_cpu_normalizer_tests", "74b0c6f57b8c04a549b9969b1feef378d0bb0dd9acc3e2a3109f35b9613ba80b"],
      ["ort_cpu_scaler_kernel", "08ee63f5e1b4a2341f190537198761d648528262752e4cf24b083cbee1fdaee3"],
      ["ort_cpu_scaler_contract", "1c3cf3fd8063b892dc46d49c6391dd340442e26410a1fc2efff48d27c611b13c"],
      ["ort_cpu_scaler_tests", "0aa327aeec7543785e0b8c903500c84c884c9f8de5753906658cb07dd6e1c1d2"],
      ["ort_cpu_imputer_kernel", "10de709f7625306815ef374afdf3fd6dd930b4c8a6bd65e8f0fc348cda5f4dea"],
      ["ort_cpu_imputer_contract", "c2497b44ad5346190c3ea2f627e82aa6d61b8a4fae2e61508b63efd6c238b019"],
      ["ort_cpu_imputer_tests", "1b65b1904c8cea3ba3a1ed3c20fe668a6b54bb1a9a22cb7c2a492a5ab11e7a16"],
      ["ort_cpu_one_hot_encoder_kernel", "8b5bd9bcdf8455326ec857b540743465cf57bdb50f7f26820f734a179c3431ef"],
      ["ort_cpu_one_hot_encoder_contract", "f9a8522b075b1c8e33b2f4031c719db69bbab9bc5280f8b532c468286b8c4c95"],
      ["ort_cpu_one_hot_encoder_tests", "644a6caffd60f1a1721bb3dd282a0c9d372bbae2bc949b85c52bfc68b9815c10"],
      ["ort_cpu_linear_classifier_kernel", "8fcef175e9db50c017a94ac4db1b3c118294dea7eab93315d58dabfdae95d052"],
      ["ort_cpu_linear_classifier_contract", "ec8288d8b9f01115c26f9d993d586d53cb6c7936ca4b312f96b2b395aa417344"],
      ["ort_cpu_linear_classifier_tests", "2e55ce60ffd7f2a9a5b1c5059b9ec0fab5a237ded10ea27908957d8005c51f6e"],
      ["ort_cpu_linear_regressor_kernel", "615259243fec59bd088299bb4778f6f484b84af6f1ea8985497e73bc58ee11e2"],
      ["ort_cpu_linear_regressor_contract", "af47c85ed17ea6c633b16936b6bcc296a390814391ac12acce9b9b92910154b5"],
      ["ort_cpu_linear_regressor_tests", "5e30730fe8925ae3d8044872b0542f477164108a1c89dec5764d14d272b3b360"],
      ["ort_cpu_ml_post_transform_contract", "fabd40f04a61f02a882c87d4056a4fb94a7f923cba6290a46ca84bcec8c0493f"],
      ["ort_cpu_svm_classifier_kernel", "36e2d1d5d69cfbadc7eefea0363d5d5bbaf6ee52b069bb6e9cf52863e6035488"],
      ["ort_cpu_svm_classifier_contract", "affd924534a267c8fb723beff1bc2072431d27442e98348fb2f7e19abcfb03b7"],
      ["ort_cpu_svm_classifier_tests", "5ca65fcd3ebe1874c560d1f72471d328d94768871e21bf453e039979328c9428"],
      ["ort_cpu_svm_regressor_kernel", "79badf47a76df5d78005604247e9150f7d63b0604b5668fca1a00655bd8dc5fb"],
      ["ort_cpu_svm_regressor_contract", "9ef36f2efd61ba1f2a13315cce44c2bfe845b45809a8671258cf53213d98a462"],
      ["ort_cpu_svm_regressor_tests", "293d3412fa6b4fa411ea3e9f9f79a880a557d307402ab919cb8f20582566100d"],
      ["ort_cpu_label_encoder_kernel", "477280d9b83624831f4959f4e279c5c8dd8213b787570e6a4cc084eec0214edc"],
      ["ort_cpu_label_encoder_contract", "52accbc66502dedc8babb3c469d2cb969b0869e8ced3a8618de56080dd62729b"],
      ["ort_cpu_label_encoder_tests", "e8f8118f08e40f6b272ee57ba81353837807c3d69834dba8b2720b420406a9e2"],
      ["ort_cpu_zipmap_kernel", "f9205124962c59fcaf2f56aee5e0f47af05a7f21b2bb897e0e08dc39ef7f481a"],
      ["ort_cpu_zipmap_tests", "e81aa0042a93681ebec50037e5effdca5161076dfa5929991d5ebb77c05351c1"],
      ["ort_cpu_dict_vectorizer_kernel", "cbde6289fc3b17a518c6bd9d07404e798a546374be33557ce0dbabbe5779ac38"],
      ["ort_cpu_dict_vectorizer_contract", "b372ee05451ff430e7a6f112addabde1da826792411061b456dbca0ce78b69b8"],
      ["ort_cpu_dict_vectorizer_tests", "421dde93972699d765938fa82ca28848f3cfef399367f2f053c4167669004097"],
      ["ort_cpu_category_mapper_kernel", "b957901eb947300fff754cb7c9538c5dbdf3f2fa38a54cfae0e0dfda3c94ddfb"],
      ["ort_cpu_category_mapper_contract", "0ed0df8a1616d08c291d8fd41d6c6bada42d489bf801ff86071187314f12d248"],
      ["ort_cpu_category_mapper_tests", "7c619f23f2cc7945ab57f759789014ca71a9cffe998664e85b28671a1b97fead"],
      ["ort_cpu_feature_vectorizer_kernel", "fa429c30a643bbdc5694bc868c5a8355222df8e809581cbf0e3e685990de00c0"],
      ["ort_cpu_feature_vectorizer_contract", "dc335549227277c0a9a38c4448322ca6538114e6453695ad36f6eeb6782f4f8d"],
      ["ort_cpu_feature_vectorizer_tests", "cbe443bd3aea11d02ecd005b6ecb7e46c2ed6a08c68b942b302cc44e01b0cab1"],
      ["ort_cpu_array_feature_extractor_kernel", "d06602c071cb1a8ab7d1d84b50c6df3edac6c551a289366d01bf5f84276a5975"],
      ["ort_cpu_array_feature_extractor_contract", "2f0eebb9eca913d3211fcd70ca3a8c2b49c5bce5bd8b265ce16fae89891d9279"],
      ["ort_cpu_array_feature_extractor_tests", "f7f7e24bcedde18377e0b261b907d2fc473516bf5cc11bac2c804fb84922a33c"],
      ["ort_cpu_tree_ensemble_kernel", "bbe851002cdf367cb73dbb2a8fe135759d23b047a32fe120fc53ed643e306f0c"],
      ["ort_cpu_tree_ensemble_contract", "e8190efe63c7ba697120f268b7dd413b3afe6ffec899fe810bad464b28a49bdf"],
      ["ort_cpu_tree_classifier_kernel", "72247de06649e6d63d9f45b7c4e4a1177422f4578971917160e5da7157e70c8f"],
      ["ort_cpu_tree_classifier_contract", "c03b4f72bd360bf13efb84b66e4f05646124b142a3c11f4e8fb1ce5093624bb1"],
      ["ort_cpu_tree_regressor_kernel", "2c206015893789228a24981c8a2d8ada0512750d996633be0dd54e6d8c0d7791"],
      ["ort_cpu_tree_regressor_contract", "c829863e6ed10c417c0c740cf8a4fc899399babbc54041294c34c07690d99342"],
      ["ort_cpu_tree_common_contract", "57ca7bd8296c3b3d754e653d5e4b7f5876a7e6f3b1354a74d50f3535b47956b3"],
      ["ort_cpu_tree_attribute_contract", "16501a65083fefdeb3938c62052213590396b4b9db14b081b466234f00955f81"],
      ["ort_cpu_tree_aggregator_contract", "f83f53ad0cbcc33636b5cd22c07b15bcc635fd2691119451c5c9ed62fcf1e478"],
      ["ort_cpu_tree_helper_kernel", "93dd51c289e622e6ba8f910234e234cfb42f294253ce7f3037dbfe0203bc4aec"],
      ["ort_cpu_tree_helper_contract", "f8daa5d4bf7cc19f0273540d2dcd19ed4412e2657655bcf64e616d6cd77a51da"],
      ["ort_cpu_tree_classifier_tests", "402fe3a4279a78c05312ca0456f3e3c68bd654ac9350cb620523a157748d0282"],
      ["ort_cpu_tree_regressor_and_v5_tests", "5b11601f71a89323cfd77601106de63adaa01c869005249be51d4b8fb7171a2c"],
    ]);
    const exactMlSequenceLengths = mlValueRows.filter((row) => row.status !== "fail"
      && Number.isSafeInteger(row.exact_output_sequence_length) && row.exact_output_sequence_length >= 0).length;
    const exactMlClassKeys = mlValueRows.reduce((sum, row) => sum + Number(row.class_key_count || 0), 0);
    const duplicateMlClassKeys = mlValueRows.reduce((sum, row) => sum + Number(row.duplicate_key_count || 0), 0);
    const duplicateMlClassKeyNodes = mlValueRows.filter((row) => Number(row.duplicate_key_count || 0) > 0).length;
    const mlMapProducerNodes = mlValueRows.filter((row) => row.contract_kind === "map_producer").length;
    const mlMapConsumerNodes = mlValueRows.filter((row) => row.contract_kind === "map_consumer").length;
    const mlTensorMapperNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_mapper").length;
    const mlTensorAggregatorNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_aggregator").length;
    const mlTensorSelectorNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_selector").length;
    const mlTensorNormalizationNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_normalization").length;
    const mlTensorAffineScalerNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_affine_scaler").length;
    const mlTensorImputationNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_imputation").length;
    const mlTensorEncoderNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_encoder").length;
    const mlTensorLabelMappingNodes = mlValueRows.filter((row) => row.contract_kind === "tensor_label_mapping").length;
    const exactMlDenseOutputShapes = mlValueRows.filter((row) => row.status !== "fail" && row.output_kind === "tensor"
      && Array.isArray(row.exact_output_shape) && row.exact_output_shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)).length;
    const exactMlVocabularyEntries = mlValueRows.reduce((sum, row) => sum + Number(row.vocabulary_count || 0), 0);
    const duplicateMlVocabularyEntries = mlValueRows.reduce((sum, row) => sum + Number(row.duplicate_vocabulary_count || 0), 0);
    const duplicateMlVocabularyNodes = mlValueRows.filter((row) => Number(row.duplicate_vocabulary_count || 0) > 0).length;
    const duplicateDictVocabularyRows = mlValueRows.filter((row) => row.op_name === "DictVectorizer"
      && Number(row.duplicate_vocabulary_count || 0) > 0);
    const exactMlCategoryPairs = mlValueRows.reduce((sum, row) => sum + Number(row.category_pair_count || 0), 0);
    const duplicateMlCategoryActiveKeys = mlValueRows.reduce((sum, row) => sum + Number(row.active_duplicate_key_count || 0), 0);
    const duplicateMlCategoryActiveKeyNodes = mlValueRows.filter((row) => Number(row.active_duplicate_key_count || 0) > 0).length;
    const featureVectorizerRows = mlValueRows.filter((row) => row.op_name === "FeatureVectorizer");
    const featureVectorizerExactWidthRows = featureVectorizerRows.filter((row) => Number.isSafeInteger(row.total_configured_feature_count) && row.total_configured_feature_count >= 0);
    const exactFeatureVectorizerConfiguredFeatures = featureVectorizerExactWidthRows.reduce((sum, row) => sum + row.total_configured_feature_count, 0);
    const featureVectorizerTruncatingRows = featureVectorizerRows.filter((row) => Number(row.exact_truncated_feature_count_per_batch || 0) > 0);
    const exactFeatureVectorizerTruncatedPerBatch = featureVectorizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.exact_truncated_feature_count_per_batch) ? row.exact_truncated_feature_count_per_batch : 0), 0);
    const exactFeatureVectorizerPaddedPerBatch = featureVectorizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.exact_padded_feature_count_per_batch) ? row.exact_padded_feature_count_per_batch : 0), 0);
    const arrayFeatureExtractorRows = mlValueRows.filter((row) => row.op_name === "ArrayFeatureExtractor");
    const arrayFeatureExtractorExactIndexRows = arrayFeatureExtractorRows.filter((row) => Number.isSafeInteger(row.exact_index_count) && row.exact_index_count >= 0);
    const exactArrayFeatureExtractorIndices = arrayFeatureExtractorExactIndexRows.reduce((sum, row) => sum + row.exact_index_count, 0);
    const arrayFeatureExtractorDuplicateIndices = arrayFeatureExtractorRows.reduce((sum, row) => sum + Number(row.duplicate_index_count || 0), 0);
    const arrayFeatureExtractorBoundsAssessedRows = arrayFeatureExtractorRows.filter((row) => ["assessed_pass", "fail"].includes(row.index_bounds_status));
    const arrayFeatureExtractorBoundsFailureRows = arrayFeatureExtractorRows.filter((row) => row.index_bounds_status === "fail");
    const binarizerRows = mlValueRows.filter((row) => row.op_name === "Binarizer");
    const binarizerExactStaticRows = binarizerRows.filter((row) => row.static_value_assessment_status === "assessed_exact");
    const exactBinarizerInputValues = binarizerExactStaticRows.reduce((sum, row) => sum + Number(row.exact_static_input_value_count || 0), 0);
    const exactBinarizerAboveThreshold = binarizerExactStaticRows.reduce((sum, row) => sum + Number(row.exact_above_threshold_count || 0), 0);
    const exactBinarizerAtOrBelowThreshold = binarizerExactStaticRows.reduce((sum, row) => sum + Number(row.exact_at_or_below_threshold_count || 0), 0);
    const exactBinarizerEqualThreshold = binarizerExactStaticRows.reduce((sum, row) => sum + Number(row.exact_equal_threshold_count || 0), 0);
    const binarizerSchemaDefaultRows = binarizerRows.filter((row) => row.threshold_source === "onnx_schema_default_0");
    const binarizerNonfiniteThresholdRows = binarizerRows.filter((row) => row.threshold_finite === false);
    const riskyBinarizerRows = binarizerRows.filter((row) => (row.risk_codes || []).some((code) => code === "binarizer_non_finite_threshold" || code === "binarizer_static_input_contains_non_finite_or_unsafe_value"));
    const ortUnsupportedBinarizerRows = binarizerRows.filter((row) => (row.risk_codes || []).includes("binarizer_dtype_unsupported_by_pinned_ort_cpu"));
    const normalizerRows = mlValueRows.filter((row) => row.op_name === "Normalizer");
    const normalizerStaticAssessedRows = normalizerRows.filter((row) => String(row.normalizer_static_assessment_status || "").startsWith("assessed_"));
    const normalizerMaterializedRows = normalizerRows.filter((row) => row.normalizer_output_materialized === true);
    const exactNormalizerInputValues = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_exact_input_value_count) ? row.normalizer_exact_input_value_count : 0), 0);
    const exactNormalizerZeroDivisorRows = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_zero_divisor_row_count) ? row.normalizer_zero_divisor_row_count : 0), 0);
    const exactNormalizerNegativeMaxRows = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_negative_max_divisor_row_count) ? row.normalizer_negative_max_divisor_row_count : 0), 0);
    const exactNormalizerIntegerRoundingValues = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_integer_float32_rounding_count) ? row.normalizer_integer_float32_rounding_count : 0), 0);
    const exactNormalizerSignedOverflowValues = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_signed_overflow_value_count) ? row.normalizer_signed_overflow_value_count : 0), 0);
    const exactNormalizerNonfiniteOutputs = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_non_finite_output_count) ? row.normalizer_non_finite_output_count : 0), 0);
    const exactNormalizerSignedZeroOutputs = normalizerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.normalizer_signed_zero_output_count) ? row.normalizer_signed_zero_output_count : 0), 0);
    const normalizerDefaultModeRows = normalizerRows.filter((row) => row.normalizer_mode_source === "onnx_schema_default_MAX");
    const normalizerOverflowRows = normalizerRows.filter((row) => (row.risk_codes || []).includes("normalizer_signed_integer_abs_or_square_overflow"));
    const normalizerNegativeMaxRows = normalizerRows.filter((row) => (row.risk_codes || []).includes("normalizer_negative_signed_max_divisor"));
    const normalizerIntegerRoundingRows = normalizerRows.filter((row) => (row.risk_codes || []).includes("normalizer_integer_to_float32_precision_loss"));
    const normalizerNonfiniteRows = normalizerRows.filter((row) => (row.risk_codes || []).some((code) => code === "normalizer_non_finite_float32_projection" || code === "normalizer_static_input_contains_non_finite_or_unsafe_value"));
    const scalerRows = mlValueRows.filter((row) => row.op_name === "Scaler");
    const scalerStaticAssessedRows = scalerRows.filter((row) => String(row.scaler_static_assessment_status || "").startsWith("assessed_"));
    const scalerMaterializedRows = scalerRows.filter((row) => row.scaler_output_materialized === true);
    const scalerInvalidContractRows = scalerRows.filter((row) => row.scaler_parameter_contract_status === "fail");
    const exactScalerInputValues = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_exact_input_value_count) ? row.scaler_exact_input_value_count : 0), 0);
    const exactScalerIntegerRoundingValues = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_integer_float32_rounding_count) ? row.scaler_integer_float32_rounding_count : 0), 0);
    const exactScalerNonfiniteParameters = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_non_finite_parameter_count) ? row.scaler_non_finite_parameter_count : 0), 0);
    const exactScalerNonfiniteOutputs = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_non_finite_output_count) ? row.scaler_non_finite_output_count : 0), 0);
    const exactScalerSignedZeroOutputs = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_signed_zero_output_count) ? row.scaler_signed_zero_output_count : 0), 0);
    const exactScalerZeroScales = scalerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.scaler_zero_scale_count) ? row.scaler_zero_scale_count : 0), 0);
    const scalerPrecisionRows = scalerRows.filter((row) => (row.risk_codes || []).includes("scaler_integer_to_float32_precision_loss"));
    const scalerNonfiniteRows = scalerRows.filter((row) => (row.risk_codes || []).includes("scaler_non_finite_parameter_input_or_output"));
    const imputerRows = mlValueRows.filter((row) => row.op_name === "Imputer");
    const imputerStaticAssessedRows = imputerRows.filter((row) => String(row.imputer_static_assessment_status || "").startsWith("assessed_"));
    const imputerMaterializedRows = imputerRows.filter((row) => row.imputer_output_materialized === true);
    const imputerInvalidContractRows = imputerRows.filter((row) => row.imputer_parameter_contract_status === "fail");
    const imputerScalarFirstRows = imputerRows.filter((row) => row.imputer_parameter_mode === "scalar_first_fallback");
    const imputerPinnedCpuDtypeGapRows = imputerRows.filter((row) => (row.risk_codes || []).includes("imputer_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const imputerNonfiniteRows = imputerRows.filter((row) => (row.risk_codes || []).includes("imputer_non_finite_imputed_or_output"));
    const exactImputerInputValues = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_exact_input_value_count) ? row.imputer_exact_input_value_count : 0), 0);
    const exactImputerReplacements = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_exact_replacement_count) ? row.imputer_exact_replacement_count : 0), 0);
    const exactImputerNanReplacements = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_exact_nan_replacement_count) ? row.imputer_exact_nan_replacement_count : 0), 0);
    const exactImputerUnchanged = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_exact_unchanged_count) ? row.imputer_exact_unchanged_count : 0), 0);
    const exactImputerIgnoredValues = imputerRows.reduce((sum, row) => sum + Number(row.imputer_ignored_imputed_value_count || 0), 0);
    const exactImputerNonfiniteValues = imputerRows.reduce((sum, row) => sum + Number(row.imputer_non_finite_imputed_value_count || 0), 0);
    const exactImputerNonfiniteOutputs = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_non_finite_output_count) ? row.imputer_non_finite_output_count : 0), 0);
    const exactImputerSignedZeroOutputs = imputerRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.imputer_signed_zero_output_count) ? row.imputer_signed_zero_output_count : 0), 0);
    const oneHotRows = mlValueRows.filter((row) => row.op_name === "OneHotEncoder");
    const oneHotStaticAssessedRows = oneHotRows.filter((row) => String(row.onehot_static_assessment_status || "").startsWith("assessed_"));
    const oneHotMaterializedRows = oneHotRows.filter((row) => row.onehot_output_materialized === true);
    const oneHotInvalidContractRows = oneHotRows.filter((row) => row.onehot_parameter_contract_status === "fail");
    const oneHotDuplicateRows = oneHotRows.filter((row) => Number(row.onehot_duplicate_category_count || 0) > 0);
    const oneHotUnknownAllZeroRows = oneHotRows.filter((row) => (row.risk_codes || []).includes("onehot_unknown_categories_all_zero_encoding"));
    const oneHotGuaranteedFailureRows = oneHotRows.filter((row) => row.onehot_guaranteed_runtime_failure === true);
    const oneHotPinnedCpuDtypeGapRows = oneHotRows.filter((row) => (row.risk_codes || []).includes("onehot_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const oneHotNoncanonicalZerosRows = oneHotRows.filter((row) => row.onehot_zeros_canonical_boolean === false);
    const oneHotInvalidCastRows = oneHotRows.filter((row) => Number(row.onehot_numeric_to_int64_invalid_count || 0) > 0);
    const exactOneHotInputValues = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_input_value_count) ? row.onehot_exact_input_value_count : 0), 0);
    const exactOneHotMatchedInputs = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_matched_input_count) ? row.onehot_exact_matched_input_count : 0), 0);
    const exactOneHotUnknownInputs = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_unknown_input_count) ? row.onehot_exact_unknown_input_count : 0), 0);
    const exactOneHotChangedCasts = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_numeric_to_int64_changed_count) ? row.onehot_numeric_to_int64_changed_count : 0), 0);
    const exactOneHotInvalidCasts = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_numeric_to_int64_invalid_count) ? row.onehot_numeric_to_int64_invalid_count : 0), 0);
    const exactOneHotOutputOnes = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_output_one_count) ? row.onehot_exact_output_one_count : 0), 0);
    const exactOneHotOutputZeros = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_output_zero_count) ? row.onehot_exact_output_zero_count : 0), 0);
    const exactOneHotAssessedOutputElements = oneHotRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.onehot_exact_output_one_count) && Number.isSafeInteger(row.onehot_exact_output_zero_count)
        && Number.isSafeInteger(row.exact_dense_output_element_count) ? row.exact_dense_output_element_count : 0), 0);
    const exactOneHotDuplicateCategories = oneHotRows.reduce((sum, row) => sum + Number(row.onehot_duplicate_category_count || 0), 0);
    const exactOneHotUnreachableColumns = oneHotRows.reduce((sum, row) => sum + Number(row.onehot_unreachable_duplicate_column_count || 0), 0);
    const labelEncoderRows = mlValueRows.filter((row) => row.op_name === "LabelEncoder");
    const labelEncoderStaticRows = labelEncoderRows.filter((row) => String(row.label_encoder_static_assessment_status || "").startsWith("assessed_"));
    const labelEncoderMaterializedRows = labelEncoderRows.filter((row) => row.label_encoder_output_materialized === true);
    const labelEncoderOnnxFailureRows = labelEncoderRows.filter((row) => row.label_encoder_onnx_contract_status === "fail");
    const labelEncoderRuntimeFailureRows = labelEncoderRows.filter((row) => row.label_encoder_pinned_ort_contract_status === "fail");
    const labelEncoderDtypeGapRows = labelEncoderRows.filter((row) => (row.risk_codes || []).includes("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel"));
    const labelEncoderDuplicateConflictRows = labelEncoderRows.filter((row) => (row.risk_codes || []).includes("label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict"));
    const labelEncoderNanConflictRows = labelEncoderRows.filter((row) => (row.risk_codes || []).includes("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched"));
    const labelEncoderRuntimeInvalidRows = labelEncoderRows.filter((row) => (row.risk_codes || []).includes("label_encoder_pinned_ort_runtime_contract_invalid"));
    const labelEncoderDefaultRows = labelEncoderRows.filter((row) => Number(row.label_encoder_exact_default_count || 0) > 0);
    const labelEncoderOutputMismatchRows = labelEncoderRows.filter((row) => Number(row.label_encoder_schema_runtime_mismatch_count || 0) > 0);
    const exactLabelEncoderKeys = labelEncoderRows.reduce((sum, row) => sum + Number(row.label_encoder_key_count || 0), 0);
    const exactLabelEncoderInputs = labelEncoderRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.label_encoder_exact_input_value_count) ? row.label_encoder_exact_input_value_count : 0), 0);
    const exactLabelEncoderMatches = labelEncoderRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.label_encoder_exact_match_count) ? row.label_encoder_exact_match_count : 0), 0);
    const exactLabelEncoderDefaults = labelEncoderRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.label_encoder_exact_default_count) ? row.label_encoder_exact_default_count : 0), 0);
    const exactLabelEncoderDuplicateHits = labelEncoderRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.label_encoder_exact_duplicate_key_hit_count) ? row.label_encoder_exact_duplicate_key_hit_count : 0), 0);
    const exactLabelEncoderMismatches = labelEncoderRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.label_encoder_schema_runtime_mismatch_count) ? row.label_encoder_schema_runtime_mismatch_count : 0), 0);
    const linearRows = mlValueRows.filter((row) => ["LinearClassifier", "LinearRegressor"].includes(row.op_name));
    const linearClassifierRows = linearRows.filter((row) => row.op_name === "LinearClassifier");
    const linearRegressorRows = linearRows.filter((row) => row.op_name === "LinearRegressor");
    const linearOnnxFailureRows = linearRows.filter((row) => row.linear_onnx_contract_status === "fail");
    const linearRuntimeFailureRows = linearRows.filter((row) => row.linear_pinned_ort_contract_status === "fail");
    const linearReferenceRows = linearRows.filter((row) => String(row.linear_reference_assessment_status || "").startsWith("assessed_"));
    const linearDtypeGapRows = linearRows.filter((row) => (row.risk_codes || []).includes("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const linearPostTransformHazardRows = linearRows.filter((row) => (row.risk_codes || []).some((code) => [
      "linear_classifier_single_score_post_transform_noop",
      "linear_classifier_binary_probit_second_score_unwritten",
      "linear_classifier_binary_post_transform_ignored_for_complement_expansion",
      "linear_regressor_single_target_post_transform_noop",
      "linear_regressor_probit_may_emit_non_finite",
    ].includes(code)));
    const linearUnusedCoefficientRows = linearRows.filter((row) => Number(row.linear_unused_coefficient_count || 0) > 0);
    const linearIgnoredInterceptRows = linearRows.filter((row) => Number(row.linear_ignored_intercept_count || 0) > 0);
    const linearIgnoredParameterRows = linearRows.filter((row) => Number(row.linear_unused_coefficient_count || 0) > 0
      || Number(row.linear_ignored_intercept_count || 0) > 0);
    const linearSchemaDefaultTargetRows = linearRows.filter((row) => row.op_name === "LinearRegressor"
      && row.linear_targets_source === "onnx_schema_default_1_materialized_by_ort_schema_resolution");
    const linearMultiClassRows = linearRows.filter((row) => (row.risk_codes || []).includes("linear_classifier_multi_class_nonzero_ignored_by_pinned_ort"));
    const linearDuplicateLabelRows = linearRows.filter((row) => (row.risk_codes || []).includes("linear_classifier_duplicate_labels_ambiguous_output_semantics"));
    const linearNumericalRiskRows = linearRows.filter((row) => (row.risk_codes || []).some((code) => [
      "linear_classifier_non_finite_parameter_or_reference_score",
      "linear_classifier_reference_decision_boundary",
      "linear_regressor_non_finite_parameter_or_reference_score",
    ].includes(code)));
    const exactLinearCoefficients = linearRows.reduce((sum, row) => sum + Number(row.linear_coefficient_count || 0), 0);
    const exactLinearUsedCoefficients = linearRows.reduce((sum, row) => sum + Number(row.linear_used_coefficient_count || 0), 0);
    const exactLinearUnusedCoefficients = linearRows.reduce((sum, row) => sum + Number(row.linear_unused_coefficient_count || 0), 0);
    const exactLinearUnresolvedCoefficientUses = linearRows.reduce((sum, row) => sum
      + Math.max(0, Number(row.linear_coefficient_count || 0) - Number(row.linear_used_coefficient_count || 0)
        - Number(row.linear_unused_coefficient_count || 0)), 0);
    const exactLinearIgnoredIntercepts = linearRows.reduce((sum, row) => sum + Number(row.linear_ignored_intercept_count || 0), 0);
    const exactLinearReferenceInputs = linearRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.linear_reference_input_value_count) ? row.linear_reference_input_value_count : 0), 0);
    const exactLinearReferenceScores = linearRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.linear_reference_raw_score_count) ? row.linear_reference_raw_score_count : 0), 0);
    const svmRows = mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name));
    const svmClassifierRows = svmRows.filter((row) => row.op_name === "SVMClassifier");
    const svmRegressorRows = svmRows.filter((row) => row.op_name === "SVMRegressor");
    const svmLinearRows = svmRows.filter((row) => row.svm_mode === "linear");
    const svmSvcRows = svmRows.filter((row) => row.svm_mode === "svc");
    const svmOnnxFailureRows = svmRows.filter((row) => row.svm_onnx_contract_status === "fail");
    const svmPinnedRuntimeFailureRows = svmRows.filter((row) => row.svm_pinned_ort_contract_status === "fail");
    const svmRuntimeInvalidRows = svmRows.filter((row) => (row.risk_codes || [])
      .some((code) => ["svm_classifier_pinned_ort_runtime_contract_invalid", "svm_regressor_pinned_ort_runtime_contract_invalid"].includes(code)));
    const svmDtypeGapRows = svmRegressorRows.filter((row) => (row.risk_codes || [])
      .includes("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"));
    const svmScoreWidthMismatchRows = svmClassifierRows.filter((row) => row.svm_schema_runtime_score_width_mismatch === true);
    const svmIgnoredTransformRows = svmRegressorRows.filter((row) => row.svm_post_transform !== "NONE"
      && row.svm_post_transform_applied_by_pinned_ort === false);
    const svmIgnoredParameterRows = svmRows.filter((row) => (row.risk_codes || [])
      .some((code) => ["svm_classifier_serialized_parameters_ignored_by_pinned_ort", "svm_regressor_serialized_parameters_ignored_by_pinned_ort"].includes(code)));
    const svmForcedKernelRows = svmRows.filter((row) => (row.risk_codes || [])
      .some((code) => ["svm_classifier_linear_mode_forces_linear_kernel", "svm_regressor_linear_mode_forces_linear_kernel"].includes(code)));
    const svmNumericalRiskRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_non_finite_parameter_or_reference_score",
      "svm_classifier_reference_decision_boundary",
      "svm_regressor_non_finite_parameter_or_reference_score",
      "svm_regressor_reference_decision_boundary",
    ].includes(code)));
    const svmSemanticHazardRows = svmRows.filter((row) => (row.risk_codes || []).some((code) => [
      "svm_classifier_duplicate_labels_ambiguous_output_semantics",
      "svm_classifier_probability_scores_receive_additional_post_transform",
      "svm_classifier_binary_probit_second_score_unwritten",
      "svm_classifier_binary_post_transform_uses_complement_expansion",
      "svm_regressor_noncanonical_one_class_flag",
    ].includes(code)));
    const svmReferenceRows = svmRows.filter((row) => String(row.svm_reference_assessment_status || "").startsWith("assessed_"));
    const exactSvmVectors = svmRows.reduce((sum, row) => sum + Number(row.svm_vector_count || 0), 0);
    const exactSvmPairs = svmClassifierRows.reduce((sum, row) => sum + Number(row.svm_pairwise_classifier_count || 0), 0);
    const exactSvmSupportValues = svmRows.reduce((sum, row) => sum + Number(row.svm_support_vector_value_count || 0), 0);
    const exactSvmUsedSupportValues = svmRows.reduce((sum, row) => sum + Number(row.svm_used_support_vector_value_count || 0), 0);
    const exactSvmUnusedSupportValues = svmRows.reduce((sum, row) => sum + Number(row.svm_unused_support_vector_value_count || 0), 0);
    const exactSvmUnresolvedSupportUses = svmRows.reduce((sum, row) => sum
      + Math.max(0, Number(row.svm_support_vector_value_count || 0) - Number(row.svm_used_support_vector_value_count || 0)
        - Number(row.svm_unused_support_vector_value_count || 0)), 0);
    const exactSvmCoefficients = svmRows.reduce((sum, row) => sum + Number(row.svm_coefficient_count || 0), 0);
    const exactSvmUsedCoefficients = svmRows.reduce((sum, row) => sum + Number(row.svm_used_coefficient_count || 0), 0);
    const exactSvmUnusedCoefficients = svmRows.reduce((sum, row) => sum + Number(row.svm_unused_coefficient_count || 0), 0);
    const exactSvmUnresolvedCoefficientUses = svmRows.reduce((sum, row) => sum
      + Math.max(0, Number(row.svm_coefficient_count || 0) - Number(row.svm_used_coefficient_count || 0)
        - Number(row.svm_unused_coefficient_count || 0)), 0);
    const exactSvmRhos = svmRows.reduce((sum, row) => sum + Number(row.svm_rho_count || 0), 0);
    const exactSvmUsedRhos = svmRows.reduce((sum, row) => sum + Number(row.svm_used_rho_count || 0), 0);
    const exactSvmUnusedRhos = svmRows.reduce((sum, row) => sum + Number(row.svm_unused_rho_count || 0), 0);
    const exactSvmUnresolvedRhoUses = svmRows.reduce((sum, row) => sum
      + Math.max(0, Number(row.svm_rho_count || 0) - Number(row.svm_used_rho_count || 0)
        - Number(row.svm_unused_rho_count || 0)), 0);
    const exactSvmReferenceInputs = svmRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.svm_reference_input_value_count) ? row.svm_reference_input_value_count : 0), 0);
    const exactSvmReferenceScores = svmRows.reduce((sum, row) => sum
      + (Number.isSafeInteger(row.svm_reference_raw_score_count) ? row.svm_reference_raw_score_count : 0), 0);
    const treeFacts = buildOnnxTreeConformanceFacts(mlValueRows);
    const { treeRows } = treeFacts;
    const validMlCommon = (row) => String(row.scope || "").length > 0
      && Number.isSafeInteger(row.node_index) && row.node_index >= 0
      && ["Binarizer", "Normalizer", "Scaler", "Imputer", "OneHotEncoder", "LabelEncoder", "LinearClassifier", "LinearRegressor", "SVMClassifier", "SVMRegressor", "TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor", "ZipMap", "CastMap", "DictVectorizer", "CategoryMapper", "FeatureVectorizer", "ArrayFeatureExtractor"].includes(row.op_name)
      && row.contract_kind === (row.op_name === "Binarizer" ? "tensor_threshold"
        : row.op_name === "Normalizer" ? "tensor_normalization"
        : row.op_name === "Scaler" ? "tensor_affine_scaler"
        : row.op_name === "Imputer" ? "tensor_imputation"
        : row.op_name === "OneHotEncoder" ? "tensor_encoder"
        : row.op_name === "LabelEncoder" ? "tensor_label_mapping"
        : row.op_name === "LinearClassifier" ? "linear_classifier"
        : row.op_name === "LinearRegressor" ? "linear_regressor"
        : row.op_name === "SVMClassifier" ? "svm_classifier"
        : row.op_name === "SVMRegressor" ? "svm_regressor"
        : row.op_name === "TreeEnsemble" ? "tree_ensemble_v5"
        : row.op_name === "TreeEnsembleClassifier" ? "tree_ensemble_classifier"
        : row.op_name === "TreeEnsembleRegressor" ? "tree_ensemble_regressor"
        : row.op_name === "ZipMap" ? "map_producer"
        : row.op_name === "CategoryMapper" ? "tensor_mapper"
          : row.op_name === "FeatureVectorizer" ? "tensor_aggregator"
            : row.op_name === "ArrayFeatureExtractor" ? "tensor_selector" : "map_consumer")
      && Number.isSafeInteger(row.imported_opset) && row.imported_opset >= 1
      && ["pass", "partial", "fail"].includes(row.status)
      && Array.isArray(row.input_shape)
      && Array.isArray(row.exact_output_shape)
      && Array.isArray(row.class_key_preview) && Array.isArray(row.vocabulary_preview)
      && Array.isArray(row.reason_codes) && Array.isArray(row.risk_codes)
      && Number.isSafeInteger(row.class_key_count) && row.class_key_count >= 0
      && Number.isSafeInteger(row.duplicate_key_count) && row.duplicate_key_count >= 0 && row.duplicate_key_count <= row.class_key_count
      && Number.isSafeInteger(row.vocabulary_count) && row.vocabulary_count >= 0
      && Number.isSafeInteger(row.duplicate_vocabulary_count) && row.duplicate_vocabulary_count >= 0 && row.duplicate_vocabulary_count <= row.vocabulary_count
      && (row.exact_input_map_key_count == null || Number.isSafeInteger(row.exact_input_map_key_count) && row.exact_input_map_key_count >= 0)
      && ["not_applicable", "assessed_pass", "fail", "not_assessed_runtime_keys", "not_assessed_invalid_or_unresolved_max_map"].includes(row.sparse_key_bounds_status)
      && (row.exact_dense_output_element_count == null || Number.isSafeInteger(row.exact_dense_output_element_count) && row.exact_dense_output_element_count >= 0)
      && String(row.output_shape_basis || "").length > 0
      && ["pinned_ort_cpu_implementation", "pinned_ort_cpu_float32_kernel_only", "pinned_ort_cpu_float32_output_kernel", "pinned_ort_cpu_scaler_kernel", "pinned_ort_cpu_imputer_kernel", "pinned_ort_cpu_int64_float_double_string_kernels", "pinned_ort_cpu_label_encoder_versioned_kernels", "pinned_ort_cpu_linear_model_kernel_and_ml_common", "pinned_ort_cpu_svm_kernel_and_ml_common", "pinned_ort_cpu_tree_ensemble_kernel_common_attribute_aggregator", "onnx_schema_only_no_pinned_ort_cpu_kernel"].includes(row.runtime_reference_status)
      && (row.status === "pass" ? row.reason_codes.length === 0 : row.reason_codes.length > 0);
    const validBinarizerRow = (row) => {
      const threshold = row.threshold_value_text === "NaN" ? Number.NaN
        : row.threshold_value_text === "Infinity" ? Number.POSITIVE_INFINITY
          : row.threshold_value_text === "-Infinity" ? Number.NEGATIVE_INFINITY
            : Number(row.threshold_value_text);
      const inputEvidence = (staticAnalysis?.tensors || []).find((tensor) => tensor.name === row.input_name);
      const outputEvidence = (staticAnalysis?.tensors || []).find((tensor) => tensor.name === row.output_name);
      let exactInput = null;
      let integerInput = false;
      if (row.input_dtype === "INT64" && inputEvidence?.initializer_integer_values_exact_complete === true
        && Array.isArray(inputEvidence.initializer_integer_values_exact_decimals)) {
        try {
          exactInput = inputEvidence.initializer_integer_values_exact_decimals.map((value) => BigInt(value));
          integerInput = true;
        } catch {
          return false;
        }
      } else if (["FLOAT32", "FLOAT64", "INT32"].includes(row.input_dtype)
        && inputEvidence?.static_values_complete === true && Array.isArray(inputEvidence.static_values)) {
        exactInput = inputEvidence.static_values;
      }
      const expected = exactInput?.map((value) => {
        if (!integerInput) return value > threshold ? 1 : 0;
        if (Number.isNaN(threshold) || threshold === Number.POSITIVE_INFINITY) return 0;
        if (threshold === Number.NEGATIVE_INFINITY) return 1;
        return value > BigInt(Math.floor(threshold)) ? 1 : 0;
      }) || null;
      const expectedAbove = expected?.reduce((sum, value) => sum + value, 0) ?? null;
      const expectedAtOrBelow = expected ? expected.length - expectedAbove : null;
      const expectedEqual = exactInput?.filter((value) => integerInput
        ? Number.isFinite(threshold) && Number.isInteger(threshold) && value === BigInt(threshold)
        : value === threshold).length ?? null;
      const shapeElements = safeShapeProduct(row.exact_output_shape);
      const expectedStaticStatus = exactInput ? "assessed_exact"
        : inputEvidence?.role === "initializer"
          ? inputEvidence.static_values_status || "not_assessed_initializer_values" : "not_assessed_runtime_values";
      return ["tensor", "unresolved"].includes(row.input_kind)
        && ["FLOAT32", "FLOAT64", "INT32", "INT64", "UNKNOWN"].includes(row.input_dtype)
        && row.output_kind === "tensor" && row.output_dtype === row.input_dtype
        && row.input_rank === row.exact_output_rank
        && JSON.stringify(row.input_shape) === JSON.stringify(row.exact_output_shape)
        && row.exact_dense_output_element_count === shapeElements
        && row.output_shape_basis === "pinned_onnx_same_type_same_shape_propagation"
        && row.runtime_reference_status === "pinned_ort_cpu_float32_kernel_only"
        && ["explicit_attribute", "onnx_schema_default_0"].includes(row.threshold_source)
        && typeof row.threshold_value_text === "string" && row.threshold_value_text.length > 0
        && row.threshold_finite === Number.isFinite(threshold)
        && (row.threshold_finite ? Object.is(row.threshold_value, threshold) || row.threshold_value === threshold : row.threshold_value == null)
        && (row.threshold_source !== "onnx_schema_default_0" || Object.is(threshold, 0))
        && row.static_value_assessment_status === expectedStaticStatus
        && row.exact_static_input_value_count === (expected?.length ?? null)
        && row.exact_above_threshold_count === expectedAbove
        && row.exact_at_or_below_threshold_count === expectedAtOrBelow
        && row.exact_equal_threshold_count === expectedEqual
        && row.exact_output_zero_count === expectedAtOrBelow
        && row.exact_output_one_count === expectedAbove
        && (expected == null || row.status === "fail" || outputEvidence?.static_values_complete === true
          && JSON.stringify(outputEvidence.static_values) === JSON.stringify(expected))
        && row.sparse_key_bounds_status === "not_applicable"
        && row.risk_codes.includes("binarizer_non_finite_threshold") === !Number.isFinite(threshold)
        && row.risk_codes.includes("binarizer_dtype_unsupported_by_pinned_ort_cpu")
          === (["FLOAT64", "INT32", "INT64"].includes(row.input_dtype))
        && row.risk_codes.includes("binarizer_static_input_contains_non_finite_or_unsafe_value")
          === (inputEvidence?.static_values_status === "not_assessed_non_finite_or_unsafe_value"
            && ["FLOAT32", "FLOAT64"].includes(row.input_dtype));
    };
    const validNormalizerRow = (row) => validateNormalizerRowAgainstEvidence(row, staticAnalysis?.tensors || []);
    const validScalerRow = (row) => validateScalerRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || []);
    const validImputerRow = (row) => validateImputerRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || []);
    const validOneHotEncoderRow = (row) => validateOneHotEncoderRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || []);
    const validLinearModelRow = (row) => validateLinearModelRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || []);
    const validZipMapRow = (row) => row.input_kind === "tensor"
      && ([null, 1, 2].includes(row.input_rank) || row.status === "fail" && Number.isSafeInteger(row.input_rank) && row.input_rank >= 0)
      && ["STRING", "INT64", "UNDEFINED"].includes(row.class_key_type)
      && row.class_key_preview.length <= Math.min(8, row.class_key_count)
      && (row.exact_batch_count == null || Number.isSafeInteger(row.exact_batch_count) && row.exact_batch_count >= 0)
      && (row.exact_feature_count == null || Number.isSafeInteger(row.exact_feature_count) && row.exact_feature_count >= 0)
      && (row.exact_output_sequence_length == null || Number.isSafeInteger(row.exact_output_sequence_length) && row.exact_output_sequence_length >= 0)
      && row.exact_output_sequence_length === row.exact_batch_count
      && row.output_kind === "sequence" && row.exact_output_rank == null && row.exact_dense_output_element_count == null
      && row.output_shape_basis === "pinned_onnx_schema_and_ort_cpu_batch_semantics"
      && row.runtime_reference_status === "pinned_ort_cpu_implementation"
      && row.sparse_key_bounds_status === "not_applicable"
      && (row.status === "fail" || row.class_key_type !== "UNDEFINED" && String(row.canonical_output_type || "").startsWith("sequence<map<"))
      && (row.duplicate_key_count > 0) === row.risk_codes.includes("zip_map_duplicate_class_keys_information_loss_risk");
    const validCastMapRow = (row) => {
      const expectedOutputDtype = { TO_FLOAT: "FLOAT32", TO_STRING: "STRING", TO_INT64: "INT64" }[row.cast_to] || "UNKNOWN";
      const exactLengthContract = row.map_form === "SPARSE"
        ? row.max_map == null || row.max_map < 0 || row.exact_dense_output_element_count === row.max_map
        : row.exact_dense_output_element_count == null || row.exact_dense_output_element_count === row.exact_input_map_key_count;
      return ["map", "unresolved", "tensor"].includes(row.input_kind)
      && ["INT64", "STRING", "UNDEFINED"].includes(row.input_map_key_type)
      && ["FLOAT32", "STRING", "UNKNOWN"].includes(row.input_map_value_dtype)
      && row.output_dtype === expectedOutputDtype
      && ["DENSE", "SPARSE"].includes(row.map_form) === !row.reason_codes.some((reason) => reason.startsWith("cast_map_form_invalid:"))
      && (row.max_map == null || Number.isSafeInteger(row.max_map))
      && row.output_kind === "tensor" && row.exact_output_rank === 1 && row.exact_output_shape.length === 1
      && (row.exact_dense_output_element_count == null
        ? row.exact_output_shape[0] === -1
        : row.exact_output_shape[0] === row.exact_dense_output_element_count)
      && exactLengthContract
      && row.output_shape_basis === (row.map_form === "SPARSE"
        ? "pinned_onnx_schema_sparse_max_map"
        : row.exact_input_map_key_count == null ? "rank_only_length_runtime_unknown" : "artifact_exact_input_map_cardinality")
      && row.sparse_key_bounds_status === (row.map_form !== "SPARSE" ? "not_applicable"
        : row.reason_codes.some((reason) => reason.startsWith("cast_map_sparse_key_out_of_bounds:")) ? "fail"
          : row.reason_codes.includes("cast_map_sparse_key_bounds_runtime_unknown") ? "not_assessed_runtime_keys"
            : row.max_map == null || row.max_map < 0 ? "not_assessed_invalid_or_unresolved_max_map" : "assessed_pass")
      && row.runtime_reference_status === "onnx_schema_only_no_pinned_ort_cpu_kernel"
      && row.risk_codes.length === 0;
    };
    const validDictVectorizerRow = (row) => ["map", "unresolved", "tensor"].includes(row.input_kind)
      && ["STRING", "INT64", "UNDEFINED"].includes(row.input_map_key_type)
      && ["STRING", "INT64", "FLOAT32", "FLOAT64", "UNKNOWN"].includes(row.input_map_value_dtype)
      && ["STRING", "INT64", "UNDEFINED"].includes(row.vocabulary_type)
      && row.vocabulary_preview.length <= Math.min(8, row.vocabulary_count)
      && row.output_kind === "tensor" && row.exact_output_rank === 2
      && row.exact_output_shape.length === 2 && row.exact_output_shape[0] === 1
      && row.exact_output_shape[1] === row.vocabulary_count
      && row.exact_dense_output_element_count === row.vocabulary_count
      && row.output_dtype === row.input_map_value_dtype
      && row.output_shape_basis === "pinned_onnx_type_constraint_and_ort_cpu_vocabulary_size_allocation"
      && row.runtime_reference_status === "pinned_ort_cpu_implementation"
      && row.sparse_key_bounds_status === "not_applicable"
      && (row.duplicate_vocabulary_count > 0) === row.risk_codes.includes("dict_vectorizer_duplicate_vocabulary_columns");
    const validCategoryMapperRow = (row) => {
      const direction = row.input_dtype === "STRING" ? "STRING_TO_INT64" : row.input_dtype === "INT64" ? "INT64_TO_STRING" : "UNRESOLVED";
      const outputDtype = direction === "STRING_TO_INT64" ? "INT64" : direction === "INT64_TO_STRING" ? "STRING" : "UNKNOWN";
      const activeDuplicates = row.category_string_count !== row.category_int64_count ? 0
        : direction === "STRING_TO_INT64" ? row.duplicate_string_key_count
          : direction === "INT64_TO_STRING" ? row.duplicate_int64_key_count : 0;
      const shapeKnown = Array.isArray(row.exact_output_shape) && row.exact_output_shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0);
      const shapeProduct = shapeKnown ? row.exact_output_shape.reduce((product, dimension) => product * dimension, 1) : null;
      const shapeElements = Number.isSafeInteger(shapeProduct) && shapeProduct >= 0 ? shapeProduct : null;
      return ["tensor", "unresolved"].includes(row.input_kind)
        && ["STRING", "INT64", "UNKNOWN"].includes(row.input_dtype)
        && row.mapping_direction === direction && row.output_dtype === outputDtype
        && Number.isSafeInteger(row.category_pair_count) && row.category_pair_count >= 0
        && Number.isSafeInteger(row.category_string_count) && row.category_string_count >= 0
        && Number.isSafeInteger(row.category_int64_count) && row.category_int64_count >= 0
        && row.category_pair_count === (row.category_string_count === row.category_int64_count ? row.category_string_count : 0)
        && Number.isSafeInteger(row.duplicate_string_key_count) && row.duplicate_string_key_count >= 0 && row.duplicate_string_key_count <= row.category_string_count
        && Number.isSafeInteger(row.duplicate_int64_key_count) && row.duplicate_int64_key_count >= 0 && row.duplicate_int64_key_count <= row.category_int64_count
        && row.active_duplicate_key_count === activeDuplicates
        && ["STRING", "INT64", "UNDEFINED"].includes(row.active_default_type)
        && typeof row.active_default_value === "string"
        && Array.isArray(row.category_string_preview) && row.category_string_preview.length <= Math.min(8, row.category_string_count)
        && Array.isArray(row.category_int64_preview) && row.category_int64_preview.length <= Math.min(8, row.category_int64_count)
        && row.output_kind === "tensor" && row.input_rank === row.exact_output_rank
        && JSON.stringify(row.input_shape) === JSON.stringify(row.exact_output_shape)
        && (shapeElements == null ? row.exact_dense_output_element_count == null : row.exact_dense_output_element_count === shapeElements)
        && row.output_shape_basis === "pinned_onnx_shape_propagation_and_ort_cpu_same_shape_allocation"
        && row.runtime_reference_status === "pinned_ort_cpu_implementation"
        && row.sparse_key_bounds_status === "not_applicable"
        && (row.active_duplicate_key_count > 0) === row.risk_codes.includes("category_mapper_duplicate_active_keys_last_write_wins");
    };
    const safeShapeProduct = (shape) => {
      if (!Array.isArray(shape) || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)) return null;
      let total = 1;
      for (const dimension of shape) {
        if (total > Math.floor(Number.MAX_SAFE_INTEGER / Math.max(1, dimension))) return null;
        total *= dimension;
      }
      return total;
    };
    const safeNonnegativeDecimal = (value) => {
      try {
        const parsed = BigInt(value);
        return parsed >= 0n && parsed <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(parsed) : null;
      } catch {
        return null;
      }
    };
    const exactArraySum = (values) => {
      if (!Array.isArray(values) || values.some((value) => !Number.isSafeInteger(value) || value < 0)) return null;
      let total = 0;
      for (const value of values) {
        if (total > Number.MAX_SAFE_INTEGER - value) return null;
        total += value;
      }
      return total;
    };
    const validFeatureVectorizerRow = (row) => {
      const inputNames = row.input_names || [];
      const inputDtypes = row.input_dtypes || [];
      const inputRanks = row.input_ranks || [];
      const inputShapes = row.input_shapes || [];
      const configuredDecimals = row.configured_feature_dimensions || [];
      const configured = configuredDecimals.map(safeNonnegativeDecimal);
      const configuredTotal = configured.length === configuredDecimals.length ? exactArraySum(configured) : null;
      const batchCounts = inputRanks.map((rank, index) => rank == null || rank === 0 ? null
        : rank === 1 ? 1
          : Number.isSafeInteger(inputShapes[index]?.[0]) && inputShapes[index][0] >= 0 ? inputShapes[index][0] : null);
      const rowWidths = inputRanks.map((rank, index) => rank == null || rank === 0 ? null
        : safeShapeProduct(rank === 1 ? inputShapes[index] : inputShapes[index]?.slice(1)));
      const copied = rowWidths.map((actual, index) => actual == null || configured[index] == null ? null : Math.min(actual, configured[index]));
      const padded = rowWidths.map((actual, index) => actual == null || configured[index] == null ? null : Math.max(0, configured[index] - actual));
      const truncated = rowWidths.map((actual, index) => actual == null || configured[index] == null ? null : Math.max(0, actual - configured[index]));
      const copiedTotal = exactArraySum(copied);
      const paddedTotal = exactArraySum(padded);
      const truncatedTotal = exactArraySum(truncated);
      const outputShape = [batchCounts[0] ?? -1, configuredTotal ?? -1];
      const outputElements = safeShapeProduct(outputShape);
      const knownDtypes = [...new Set(inputDtypes.filter((dtype) => dtype !== "UNKNOWN"))];
      const expectedInputDtype = knownDtypes.length === 1 ? knownDtypes[0] : knownDtypes.length > 1 ? "MIXED" : "UNKNOWN";
      return row.input_kind === "tensor_list"
        && Array.isArray(inputNames) && inputNames.length > 0
        && inputDtypes.length === inputNames.length && inputRanks.length === inputNames.length
        && inputShapes.length === inputNames.length && inputShapes.every(Array.isArray)
        && configuredDecimals.length === row.configured_feature_dimension_count
        && configuredDecimals.every((value) => /^-?\d+$/.test(String(value)))
        && row.input_dtype === expectedInputDtype
        && JSON.stringify(row.exact_batch_counts) === JSON.stringify(batchCounts)
        && row.exact_batch_count === (batchCounts[0] ?? null)
        && JSON.stringify(row.exact_input_row_feature_counts) === JSON.stringify(rowWidths)
        && row.total_configured_feature_count === configuredTotal
        && row.exact_feature_count === configuredTotal
        && JSON.stringify(row.copied_feature_counts_per_input) === JSON.stringify(copied)
        && JSON.stringify(row.padded_feature_counts_per_input) === JSON.stringify(padded)
        && JSON.stringify(row.truncated_feature_counts_per_input) === JSON.stringify(truncated)
        && row.exact_copied_feature_count_per_batch === copiedTotal
        && row.exact_padded_feature_count_per_batch === paddedTotal
        && row.exact_truncated_feature_count_per_batch === truncatedTotal
        && row.padded_input_count === padded.filter((value) => value > 0).length
        && row.truncated_input_count === truncated.filter((value) => value > 0).length
        && row.output_kind === "tensor" && row.output_dtype === "FLOAT32" && row.exact_output_rank === 2
        && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputShape)
        && row.exact_dense_output_element_count === outputElements
        && row.output_shape_basis === "pinned_ort_cpu_feature_dimension_allocation"
        && row.runtime_reference_status === "pinned_ort_cpu_implementation"
        && row.sparse_key_bounds_status === "not_applicable"
        && (truncatedTotal > 0) === row.risk_codes.includes("feature_vectorizer_truncates_input_features");
    };
    const validArrayFeatureExtractorRow = (row) => {
      const inputNames = row.input_names || [];
      const inputDtypes = row.input_dtypes || [];
      const inputRanks = row.input_ranks || [];
      const inputShapes = row.input_shapes || [];
      const indexShape = row.index_input_shape || [];
      const exactIndexCount = row.index_input_rank == null ? null : safeShapeProduct(indexShape);
      let exactValues = null;
      if (row.exact_index_values_status === "assessed_exact") {
        try {
          exactValues = (row.exact_index_values || []).map((value) => BigInt(value));
        } catch {
          return false;
        }
      }
      const indexTensorEvidence = (staticAnalysis?.tensors || []).find((tensor) => tensor.name === row.index_input_name);
      const evidenceExactIndexValues = indexTensorEvidence?.initializer_integer_values_exact_complete === true
        && Array.isArray(indexTensorEvidence.initializer_integer_values_exact_decimals)
        ? indexTensorEvidence.initializer_integer_values_exact_decimals.map(String)
        : indexTensorEvidence?.initializer_integer_values_complete === true
          && Array.isArray(indexTensorEvidence.initializer_integer_values)
          ? indexTensorEvidence.initializer_integer_values.map(String)
          : indexTensorEvidence?.static_values_complete === true
            && Array.isArray(indexTensorEvidence.static_values)
            && indexTensorEvidence.static_values.every(Number.isSafeInteger)
            ? indexTensorEvidence.static_values.map(String) : null;
      const duplicateIndices = exactValues ? exactValues.length - new Set(exactValues.map(String)).size : 0;
      const lastAxis = row.input_rank != null && row.input_rank > 0 && Number.isSafeInteger(row.input_shape.at(-1)) && row.input_shape.at(-1) >= 0
        ? row.input_shape.at(-1) : null;
      const invalidIndices = exactValues ? exactValues.filter((value) => value < 0n || lastAxis != null && value >= BigInt(lastAxis)).length : 0;
      const expectedBoundsStatus = exactValues
        ? invalidIndices ? "fail" : lastAxis == null ? "not_assessed_dynamic_axis" : "assessed_pass"
        : exactIndexCount != null && exactIndexCount > 0 && lastAxis === 0 ? "fail"
          : exactIndexCount == null ? "not_assessed_dynamic_cardinality" : "not_assessed_runtime_values";
      const outputShape = row.input_rank == null || row.input_rank === 0 ? []
        : row.input_rank === 1 ? [1, exactIndexCount ?? -1]
          : [...row.input_shape.slice(0, -1), exactIndexCount ?? -1];
      const outputElements = outputShape.length ? safeShapeProduct(outputShape) : null;
      return row.input_kind === "tensor"
        && inputNames.length === 2 && inputDtypes.length === 2 && inputRanks.length === 2 && inputShapes.length === 2
        && row.input_name === inputNames[0] && row.index_input_name === inputNames[1]
        && row.input_dtype === inputDtypes[0] && row.index_input_dtype === inputDtypes[1]
        && row.input_rank === inputRanks[0] && row.index_input_rank === inputRanks[1]
        && JSON.stringify(row.input_shape) === JSON.stringify(inputShapes[0])
        && JSON.stringify(indexShape) === JSON.stringify(inputShapes[1])
        && row.exact_index_count === exactIndexCount
        && Array.isArray(row.exact_index_values) && Array.isArray(row.exact_index_preview)
        && (exactValues == null || row.exact_index_values.length === exactValues.length
          && JSON.stringify(row.exact_index_preview) === JSON.stringify(row.exact_index_values.slice(0, 16)))
        && (exactValues == null || evidenceExactIndexValues != null
          && JSON.stringify(row.exact_index_values) === JSON.stringify(evidenceExactIndexValues))
        && row.duplicate_index_count === duplicateIndices
        && row.index_bounds_status === expectedBoundsStatus
        && row.out_of_bounds_index_count === (exactValues ? invalidIndices
          : expectedBoundsStatus === "fail" && exactIndexCount != null ? exactIndexCount : 0)
        && row.output_kind === "tensor" && row.output_dtype === row.input_dtype
        && row.exact_output_rank === (outputShape.length || null)
        && JSON.stringify(row.exact_output_shape) === JSON.stringify(outputShape)
        && row.exact_dense_output_element_count === outputElements
        && row.output_shape_basis === "pinned_onnx_last_axis_shape_rule_and_ort_cpu_rank1_compatibility"
        && row.runtime_reference_status === "pinned_ort_cpu_implementation"
        && row.sparse_key_bounds_status === "not_applicable"
        && row.risk_codes.length === 0;
    };
    const validMlValueRows = mlValueRows.every((row) => validMlCommon(row)
      && (row.op_name === "Binarizer" ? validBinarizerRow(row)
        : row.op_name === "Normalizer" ? validNormalizerRow(row)
        : row.op_name === "Scaler" ? validScalerRow(row)
        : row.op_name === "Imputer" ? validImputerRow(row)
        : row.op_name === "OneHotEncoder" ? validOneHotEncoderRow(row)
        : row.op_name === "LabelEncoder" ? validateLabelEncoderRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || [])
        : ["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? validateSvmRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || [])
        : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name) ? validateTreeEnsembleRowAgainstEvidence(row, staticAnalysis?.tensors || [], staticAnalysis?.ops || [])
        : ["LinearClassifier", "LinearRegressor"].includes(row.op_name) ? validLinearModelRow(row)
        : row.op_name === "ZipMap" ? validZipMapRow(row)
        : row.op_name === "CastMap" ? validCastMapRow(row)
          : row.op_name === "DictVectorizer" ? validDictVectorizerRow(row)
            : row.op_name === "CategoryMapper" ? validCategoryMapperRow(row)
              : row.op_name === "FeatureVectorizer" ? validFeatureVectorizerRow(row) : validArrayFeatureExtractorRow(row)));
    const mainLocalFunctionCallCount = extendedFunctionRows.filter((row) => row.scope === "main_graph").length;
    const opsetImportContract = shapeInference?.opset_import_contract || {};
    const opsetImportRows = opsetImportContract.rows || [];
    const invalidOpsetImportRows = opsetImportRows.filter((row) => row.status === "fail");
    const opsetDomainCounts = new Map();
    for (const row of opsetImportRows) opsetDomainCounts.set(row.domain, (opsetDomainCounts.get(row.domain) || 0) + 1);
    const validOpsetRowsByDomain = new Map();
    for (const row of opsetImportRows.filter((item) => item.status === "pass")) {
      const matches = validOpsetRowsByDomain.get(row.domain) || [];
      matches.push(row);
      validOpsetRowsByDomain.set(row.domain, matches);
    }
    const expectedEffectiveOpsets = [...validOpsetRowsByDomain].map(([domain, matches]) => ({
      domain,
      version: Math.max(...matches.map((row) => Number(row.version))),
      source_indices: matches.map((row) => Number(row.index)),
      distinct_source_versions: [...new Set(matches.map((row) => Number(row.version)))].sort((a, b) => a - b),
      resolution: new Set(matches.map((row) => Number(row.version))).size > 1 ? "highest_referenced_version" : "single_effective_version",
    })).sort((left, right) => left.domain.localeCompare(right.domain));
    const expectedUnresolvableOpsetDomains = [...opsetDomainCounts.keys()].filter((domain) => !validOpsetRowsByDomain.has(domain)).sort();
    const repeatedOpsetGroups = [...opsetDomainCounts].filter(([, count]) => count > 1).map(([domain]) => opsetImportRows.filter((row) => row.domain === domain));
    const shapeMustFail = shapeConflicts.length > 0 || semanticShapeConflicts.length > 0 || invalidShapeSchemaRows.length > 0
      || opsetImportContract.status === "fail" || shapeScope.registry_status === "fail"
      || Number(shapeInference?.extended_rule_failed_node_count || 0) > 0
      || containerFailedRows.length > 0
      || mlValueFailedRows.length > 0
      || Number(shapeScope.failed_reachable_scope_count || 0) > 0;
    const mlSourceProblems = sourceLedgerProblems(mlValueSources, expectedMlValueSources);
    const mlRuntimeSourceProblems = sourceLedgerProblems(mlValueRuntimeSources, expectedMlValueRuntimeSources,
      `https://raw.githubusercontent.com/microsoft/onnxruntime/${mlValueInference.runtime_reference_commit}/`);
    check("CF-SHAPE-ML-SOURCE-001", mlSourceProblems.length === 0,
    mlSourceProblems[0] || "ONNX-ML schema source role/hash ledger does not match the independent pinned inventory.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents"]);
    check("CF-SHAPE-ML-RUNTIME-SOURCE-001", mlRuntimeSourceProblems.length === 0,
      mlRuntimeSourceProblems[0] || "ONNX-ML ORT runtime source role/URL/hash ledger does not match the independent pinned inventory.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents"]);
    check("CF-SHAPE-ML-ROW-002", validMlValueRows,
      "At least one ONNX-ML row does not independently reconstruct from public tensor, node, attribute, schema, and pinned-runtime evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
    check("CF-SHAPE-SVM-LEDGER-001", svmRows.length === Number(mlValueInference.svm_model_node_count || 0)
      && svmClassifierRows.length === Number(mlValueInference.svm_classifier_node_count || 0)
      && svmRegressorRows.length === Number(mlValueInference.svm_regressor_node_count || 0)
      && svmLinearRows.length === Number(mlValueInference.svm_linear_mode_node_count || 0)
      && svmSvcRows.length === Number(mlValueInference.svm_svc_mode_node_count || 0)
      && svmOnnxFailureRows.length === Number(mlValueInference.svm_onnx_contract_failure_node_count || 0)
      && svmPinnedRuntimeFailureRows.length === Number(mlValueInference.svm_pinned_ort_contract_failure_node_count || 0)
      && svmDtypeGapRows.length === Number(mlValueInference.svm_regressor_pinned_cpu_dtype_gap_node_count || 0)
      && svmScoreWidthMismatchRows.length === Number(mlValueInference.svm_schema_runtime_score_width_mismatch_node_count || 0)
      && svmIgnoredTransformRows.length === Number(mlValueInference.svm_ignored_post_transform_node_count || 0)
      && svmIgnoredParameterRows.length === Number(mlValueInference.svm_ignored_parameter_node_count || 0)
      && svmReferenceRows.length === Number(mlValueInference.svm_reference_assessed_node_count || 0)
      && exactSvmUsedSupportValues + exactSvmUnusedSupportValues + exactSvmUnresolvedSupportUses === exactSvmSupportValues
      && exactSvmUsedCoefficients + exactSvmUnusedCoefficients + exactSvmUnresolvedCoefficientUses === exactSvmCoefficients
      && exactSvmUsedRhos + exactSvmUnusedRhos + exactSvmUnresolvedRhoUses === exactSvmRhos,
    "ONNX-ML SVM mode, contract, reference, or serialized-parameter use ledgers do not conserve.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference"]);
    check("CF-SHAPE-TREE-LEDGER-001", onnxTreeLedgerConserves(treeFacts, mlValueInference),
    "ONNX-ML TreeEnsemble topology, runtime-contract, reference, or serialized-weight ledgers do not conserve.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference"]);
    check("CF-SHAPE-001", Boolean(shapeInference)
      && shapeInference.schema === ANALYZER_METADATA.schemas.onnxShapeInference
      && shapeInference.source_release === "v1.21.0"
      && shapeInference.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && shapeSources.length === PINNED_ONNX_SHAPE_SOURCES.length
      && new Set(shapeSources.map((source) => source.role)).size === shapeSources.length
      && PINNED_ONNX_SHAPE_SOURCES.every(([role, sourceRef, sha256]) => shapeSources.some((source) => source.role === role
        && source.source_ref === sourceRef && source.sha256 === sha256))
      && opsetImportContract.schema === "deepbom.onnx_opset_import_contract.v1.1"
      && opsetImportRows.length === Number(opsetImportContract.import_count || 0)
      && opsetImportRows.filter((row) => row.status === "pass").length === Number(opsetImportContract.valid_import_count || 0)
      && invalidOpsetImportRows.length === Number(opsetImportContract.invalid_import_count || 0)
      && opsetImportContract.status === (invalidOpsetImportRows.length ? "fail" : "pass")
      && new Set(invalidOpsetImportRows.map((row) => row.domain)).size === (opsetImportContract.invalid_domains || []).length
      && [...opsetDomainCounts.values()].filter((count) => count > 1).length === Number(opsetImportContract.duplicate_domain_count || 0)
      && repeatedOpsetGroups.filter((rows) => rows.every((row) => row.status === "pass") && new Set(rows.map((row) => Number(row.version))).size === 1).length === Number(opsetImportContract.duplicate_identical_domain_count || 0)
      && repeatedOpsetGroups.filter((rows) => new Set(rows.filter((row) => row.status === "pass").map((row) => Number(row.version))).size > 1).length === Number(opsetImportContract.duplicate_version_variant_domain_count || 0)
      && opsetImportRows.filter((row) => (row.reason_codes || []).includes("opset_version_not_positive_safe_integer")).length === Number(opsetImportContract.invalid_version_count || 0)
      && expectedEffectiveOpsets.length === Number(opsetImportContract.effective_domain_count || 0)
      && JSON.stringify(opsetImportContract.effective_imports || []) === JSON.stringify(expectedEffectiveOpsets)
      && JSON.stringify(opsetImportContract.unresolvable_domains || []) === JSON.stringify(expectedUnresolvableOpsetDomains)
      && (opsetImportContract.effective_imports || []).every((effective) => opsetImportRows.filter((row) => row.domain === effective.domain && row.status === "pass")
        .every((row) => row.selected_effective_import === (Number(row.version) === Number(effective.version))))
      && Number(shapeInference.attempted_nodes || 0) === (staticAnalysis?.ops || []).length
      && Number(shapeInference.rule_supported_nodes || 0) + Number(shapeInference.rule_unsupported_nodes || 0) === Number(shapeInference.attempted_nodes || 0)
      && shapeHistogramCount === Number(shapeInference.rule_unsupported_nodes || 0)
      && shapeUnresolvedRows.length === Number(shapeInference.rule_unresolved_node_count || 0)
      && JSON.stringify(shapeUnresolvedRows.map((row) => row.node_index)) === JSON.stringify(shapeInference.rule_unresolved_node_indices || [])
      && shapeSchemaRows.length === Number(shapeInference.schema_form_assessed_node_count || 0)
      && shapeSchemaRows.length + mainLocalFunctionCallCount === Number(shapeInference.rule_supported_nodes || 0)
      && validShapeSchemaRows.length === Number(shapeInference.schema_form_valid_node_count || 0)
      && invalidShapeSchemaRows.length === Number(shapeInference.schema_form_invalid_node_count || 0)
      && unresolvedShapeSchemaRows.length === Number(shapeInference.schema_form_unresolved_node_count || 0)
      && shapeInference.schema_form_assessment_status === (invalidShapeSchemaRows.length ? "fail" : unresolvedShapeSchemaRows.length ? "partial" : "pass")
      && shapeConflicts.length === Number(shapeInference.declaration_conflict_count || 0)
      && semanticShapeConflicts.length === Number(shapeInference.semantic_contract_conflict_count || 0)
      && (shapeMustFail ? shapeInference.status === "fail" : shapeInference.status !== "fail")
      && shapeScope.schema === "deepbom.onnx_shape_scope.v2.1"
      && Number(shapeScope.main_graph_node_count || 0) === Number(shapeInference.attempted_nodes || 0)
      && shapeScopeExclusions.length === Number(shapeScope.reachable_exclusion_count || 0)
      && shapeScopeExclusions.reduce((sum, row) => sum + Number(row.node_count || 0), 0) === Number(shapeScope.unassessed_reachable_node_count || 0)
      && Number(shapeScope.function_default_graph_count || 0) <= Number(shapeScope.nested_graph_count || 0)
      && Number(shapeScope.function_default_graph_node_count || 0) <= Number(shapeScope.nested_graph_node_count || 0)
      && Number(shapeScope.reachable_local_function_definition_count || 0) <= Number(shapeScope.local_function_definition_count || 0)
      && Number(shapeScope.reachable_local_function_body_node_count || 0) <= Number(shapeScope.local_function_body_node_count || 0)
      && shapeScopeRows.length === Number(shapeScope.reachable_scope_count || 0)
      && reachableNestedGraphScopeRows.length === Number(shapeScope.reachable_nested_graph_count || 0)
      && reachableNestedGraphScopeRows.reduce((sum, row) => sum + Number(row.node_count || 0), 0) === Number(shapeScope.reachable_nested_graph_node_count || 0)
      && reachableLocalFunctionScopeRows.length === Number(shapeScope.reachable_local_function_definition_count || 0)
      && reachableLocalFunctionScopeRows.reduce((sum, row) => sum + Number(row.node_count || 0), 0) === Number(shapeScope.reachable_local_function_body_node_count || 0)
      && shapeScopeRows.every((row) => Number(row.assessed_node_count || 0) + Number(row.unassessed_node_count || 0) === Number(row.node_count || 0))
      && shapeScopeRows.filter((row) => Number(row.execution_count || 0) > 0).length === Number(shapeScope.executed_reachable_scope_count || 0)
      && shapeScopeRows.filter((row) => row.status === "assessed").length === Number(shapeScope.fully_assessed_reachable_scope_count || 0)
      && shapeScopeRows.reduce((sum, row) => sum + Number(row.unresolved_output_count || 0), 0) === Number(shapeScope.reachable_scope_unresolved_output_count || 0)
      && shapeScope.status === (shapeScope.registry_status === "fail" || Number(shapeScope.failed_reachable_scope_count || 0) > 0
        ? "fail" : shapeScopeExclusions.length ? "partial" : shapeScopeRows.length ? "assessed_reachable_scope" : "assessed_main_graph_scope")
      && extendedShape.schema === ANALYZER_METADATA.schemas.onnxExtendedShapeInference
      && extendedShape.evidence_class === "SOURCE_PINNED_AND_DERIVED"
      && extendedShape.status === expectedExtendedStatus
      && extendedShape.source_release === "v1.21.0"
      && extendedShape.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && extendedSources.length === 4
      && new Set(extendedSources.map((source) => source.role)).size === extendedSources.length
      && extendedSources.every((source) => /^https:\/\/raw\.githubusercontent\.com\/onnx\/onnx\/[0-9a-f]{40}\//.test(source.source_ref || "") && /^[0-9a-f]{64}$/.test(source.sha256 || ""))
      && extendedFunctionRows.length === Number(extendedShape.local_function_call_count || 0)
      && extendedFunctionRows.filter((row) => row.status === "pass").length === Number(extendedShape.local_function_call_pass_count || 0)
      && extendedFunctionRows.filter((row) => row.status === "fail").length === Number(extendedShape.local_function_call_fail_count || 0)
      && extendedControlRows.length === Number(extendedShape.control_flow_node_count || 0)
      && extendedControlRows.filter((row) => row.status === "pass").length === Number(extendedShape.control_flow_pass_count || 0)
      && extendedControlRows.filter((row) => row.status === "partial").length === Number(extendedShape.control_flow_partial_count || 0)
      && extendedControlRows.filter((row) => row.status === "fail").length === Number(extendedShape.control_flow_fail_count || 0)
      && extendedLoopRows.length === Number(extendedShape.loop_node_count || 0)
      && extendedLoopRows.filter((row) => row.exact_expansion_status === "assessed").length === Number(extendedShape.loop_exact_expansion_count || 0)
      && extendedLoopRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.exact_iteration_count) ? row.exact_iteration_count : 0), 0) === Number(extendedShape.loop_exact_iteration_count || 0)
      && extendedLoopRows.reduce((sum, row) => sum + Number(row.exact_body_node_evaluation_count || 0), 0) === Number(extendedShape.loop_exact_body_node_evaluation_count || 0)
      && extendedLoopRows.reduce((sum, row) => sum + Number(row.non_dense_state_variable_count || 0), 0) === Number(extendedShape.loop_non_dense_state_variable_count || 0)
      && extendedLoopRows.every((row) => Number.isSafeInteger(row.body_node_count) && row.body_node_count >= 0
        && Number.isSafeInteger(row.state_variable_count) && row.state_variable_count >= 0
        && Number.isSafeInteger(row.scan_output_count) && row.scan_output_count >= 0
        && Array.isArray(row.state_value_kinds) && row.state_value_kinds.length === row.state_variable_count
        && Number.isSafeInteger(row.non_dense_state_variable_count) && row.non_dense_state_variable_count >= 0
        && row.non_dense_state_variable_count === row.state_value_kinds.filter((kind) => kind !== "tensor" && kind !== "unresolved").length
        && ["not_assessed", "partial", "assessed", "fail"].includes(row.exact_expansion_status)
        && (row.exact_iteration_count == null || Number.isSafeInteger(row.exact_iteration_count) && row.exact_iteration_count >= 0)
        && Number.isSafeInteger(row.exact_body_node_evaluation_count) && row.exact_body_node_evaluation_count >= 0
        && Array.isArray(row.exact_iteration_state_contracts)
        && row.exact_iteration_state_contracts.every((iteration, iterationIndex) => iteration.iteration === iterationIndex
          && Array.isArray(iteration.states) && iteration.states.length === row.state_variable_count
          && iteration.states.every((state, stateIndex) => state.state_index === stateIndex
            && Array.isArray(state.shape) && Array.isArray(state.static_values) && Array.isArray(state.sequence_element_types)
            && state.sequence_element_type_count === state.sequence_element_types.length))
        && Array.isArray(row.exact_final_state_contracts)
        && row.exact_final_state_contracts.length <= row.state_variable_count
        && row.exact_final_state_contracts.every((state, stateIndex) => state.state_index === stateIndex
          && Array.isArray(state.shape) && Array.isArray(state.static_values) && Array.isArray(state.sequence_element_types)
          && state.sequence_element_type_count === state.sequence_element_types.length)
        && Array.isArray(row.exact_nested_failure_rows)
        && (row.exact_expansion_status !== "assessed" || Number.isSafeInteger(row.exact_iteration_count)
          && row.exact_body_node_evaluation_count === row.exact_iteration_count * row.body_node_count
          && row.exact_iteration_state_contracts.length === row.exact_iteration_count))
      && extendedSequenceMapRows.length === Number(extendedShape.sequence_map_node_count || 0)
      && extendedSequenceMapRows.filter((row) => row.status === "pass").length === Number(extendedShape.sequence_map_pass_count || 0)
      && extendedSequenceMapRows.filter((row) => row.status === "partial").length === Number(extendedShape.sequence_map_partial_count || 0)
      && extendedSequenceMapRows.filter((row) => row.status === "fail").length === Number(extendedShape.sequence_map_fail_count || 0)
      && extendedSequenceMapRows.every((row) => Number.isSafeInteger(row.node_index) && row.node_index >= 0
        && Number.isSafeInteger(row.imported_opset) && row.imported_opset >= 17
        && ["pass", "partial", "fail"].includes(row.status)
        && Number.isSafeInteger(row.element_expansion_count) && row.element_expansion_count >= 0
        && Number.isSafeInteger(row.element_node_evaluation_count) && row.element_node_evaluation_count >= 0
        && row.element_node_evaluation_count >= row.element_expansion_count)
      && extendedScopeRows.length === Number(extendedShape.scope_definition_count || 0)
      && extendedScopeRows.reduce((sum, row) => sum + Number(row.execution_count || 0), 0) === Number(extendedShape.scope_execution_count || 0)
      && extendedScopeRows.filter((row) => row.status === "assessed").length === Number(extendedShape.fully_assessed_scope_count || 0)
      && extendedScopeRows.reduce((sum, row) => sum + Number(row.unassessed_node_count || 0), 0) === Number(extendedShape.residual_unassessed_node_count || 0)
      && extendedScopeRows.reduce((sum, row) => sum + Number(row.unresolved_output_count || 0), 0) === Number(extendedShape.residual_unresolved_output_count || 0)
      && extendedScopeRows.every((row) => String(row.scope || "").length > 0
        && ["nested_graph", "local_function_body"].includes(row.scope_class)
        && ["assessed", "partial", "fail"].includes(row.status)
        && Number.isSafeInteger(row.node_count) && row.node_count >= 0
        && Number.isSafeInteger(row.execution_count) && row.execution_count > 0
        && Number.isSafeInteger(row.assessed_node_count) && row.assessed_node_count >= 0
        && Number.isSafeInteger(row.unassessed_node_count) && row.unassessed_node_count >= 0
        && row.assessed_node_count + row.unassessed_node_count === row.node_count
        && Number.isSafeInteger(row.unresolved_output_count) && row.unresolved_output_count >= 0
        && Number.isSafeInteger(row.intrinsic_cost_variant_count) && row.intrinsic_cost_variant_count === (row.intrinsic_cost_variants || []).length
        && Number.isSafeInteger(row.intrinsic_cost_variant_overflow_count) && row.intrinsic_cost_variant_overflow_count >= 0
        && Number.isSafeInteger(row.intrinsic_cost_unassessed_execution_count) && row.intrinsic_cost_unassessed_execution_count >= 0
        && (row.intrinsic_cost_variants || []).every((cost) => Number.isSafeInteger(cost.observation_count) && cost.observation_count > 0 && validIntrinsicCost(cost))
        && (row.intrinsic_cost_variants || []).reduce((sum, cost) => sum + cost.observation_count, 0)
          + row.intrinsic_cost_variant_overflow_count + row.intrinsic_cost_unassessed_execution_count === row.execution_count
        && Array.isArray(row.reason_codes))
      && validIntrinsicCost(extendedShape.main_graph_intrinsic_cost)
      && intrinsicCostVariants.length === Number(extendedShape.intrinsic_cost_variant_count || 0)
      && extendedScopeRows.reduce((sum, row) => sum + row.intrinsic_cost_variant_overflow_count, 0) === Number(extendedShape.intrinsic_cost_variant_overflow_count || 0)
      && extendedScopeRows.reduce((sum, row) => sum + row.intrinsic_cost_unassessed_execution_count, 0) === Number(extendedShape.intrinsic_cost_unassessed_execution_count || 0)
      && containerInference.schema === ANALYZER_METADATA.schemas.onnxContainerValueInference
      && containerInference.source_release === "v1.21.0"
      && containerInference.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && containerSources.length === 4
      && new Set(containerSources.map((source) => source.role)).size === containerSources.length
      && containerSources.every((source) => /^https:\/\/raw\.githubusercontent\.com\/onnx\/onnx\/[0-9a-f]{40}\//.test(source.source_ref || "") && /^[0-9a-f]{64}$/.test(source.sha256 || ""))
      && validContainerRows
      && containerRows.length === Number(containerInference.assessed_node_count || 0)
      && containerPassRows.length === Number(containerInference.passed_node_count || 0)
      && containerPartialRows.length === Number(containerInference.partially_assessed_node_count || 0)
      && containerFailedRows.length === Number(containerInference.failed_node_count || 0)
      && JSON.stringify(containerRows.filter((row) => row.status === "partial")) === JSON.stringify(containerPartialRows)
      && JSON.stringify(containerRows.filter((row) => row.status === "fail")) === JSON.stringify(containerFailedRows)
      && independentlyExactSequenceLengths === Number(containerInference.exact_sequence_length_output_count || 0)
      && independentlyExactOptionalPresence === Number(containerInference.exact_optional_presence_output_count || 0)
      && containerInference.status === (containerFailedRows.length ? "fail" : containerPartialRows.length ? "partial" : containerRows.length ? "assessed" : "not_applicable")
      && mlValueInference.schema === ANALYZER_METADATA.schemas.onnxMlValueInference
      && mlValueInference.source_release === "v1.21.0"
      && mlValueInference.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
      && mlValueSources.length === expectedMlValueSources.size
      && new Set(mlValueSources.map((source) => source.role)).size === mlValueSources.length
      && mlValueSources.every((source) => /^https:\/\/raw\.githubusercontent\.com\/onnx\/onnx\/[0-9a-f]{40}\//.test(source.source_ref || "") && /^[0-9a-f]{64}$/.test(source.sha256 || ""))
      && mlValueSources.every((source) => expectedMlValueSources.get(source.role) === source.sha256)
      && mlValueInference.runtime_reference_commit === "8c546c37b43caaca1fa25db430dab94b901cf277"
      && mlValueRuntimeSources.length === expectedMlValueRuntimeSources.size
      && new Set(mlValueRuntimeSources.map((source) => source.role)).size === mlValueRuntimeSources.length
      && mlValueRuntimeSources.every((source) => String(source.source_ref || "").startsWith(`https://raw.githubusercontent.com/microsoft/onnxruntime/${mlValueInference.runtime_reference_commit}/`)
        && expectedMlValueRuntimeSources.get(source.role) === source.sha256)
      && validMlValueRows
      && mlValueRows.length === Number(mlValueInference.assessed_node_count || 0)
      && mlValuePassRows.length === Number(mlValueInference.passed_node_count || 0)
      && mlValuePartialRows.length === Number(mlValueInference.partially_assessed_node_count || 0)
      && mlValueFailedRows.length === Number(mlValueInference.failed_node_count || 0)
      && JSON.stringify(mlValueRows.filter((row) => row.status === "partial")) === JSON.stringify(mlValuePartialRows)
      && JSON.stringify(mlValueRows.filter((row) => row.status === "fail")) === JSON.stringify(mlValueFailedRows)
      && exactMlSequenceLengths === Number(mlValueInference.exact_sequence_length_output_count || 0)
      && exactMlClassKeys === Number(mlValueInference.exact_class_key_count || 0)
      && duplicateMlClassKeys === Number(mlValueInference.duplicate_class_key_count || 0)
      && duplicateMlClassKeyNodes === Number(mlValueInference.duplicate_class_key_node_count || 0)
      && mlMapProducerNodes === Number(mlValueInference.map_producer_node_count || 0)
      && mlMapConsumerNodes === Number(mlValueInference.map_consumer_node_count || 0)
      && mlTensorMapperNodes === Number(mlValueInference.tensor_mapper_node_count || 0)
      && mlTensorAggregatorNodes === Number(mlValueInference.tensor_aggregator_node_count || 0)
      && mlTensorSelectorNodes === Number(mlValueInference.tensor_selector_node_count || 0)
      && mlTensorNormalizationNodes === Number(mlValueInference.tensor_normalization_node_count || 0)
      && mlTensorAffineScalerNodes === Number(mlValueInference.tensor_affine_scaler_node_count || 0)
      && mlTensorImputationNodes === Number(mlValueInference.tensor_imputation_node_count || 0)
      && mlTensorEncoderNodes === Number(mlValueInference.tensor_encoder_node_count || 0)
      && mlTensorLabelMappingNodes === Number(mlValueInference.tensor_label_mapping_node_count || 0)
      && linearRows.length === Number(mlValueInference.linear_model_node_count || 0)
      && svmRows.length === Number(mlValueInference.svm_model_node_count || 0)
      && exactMlDenseOutputShapes === Number(mlValueInference.exact_dense_output_shape_count || 0)
      && exactMlVocabularyEntries === Number(mlValueInference.exact_vocabulary_entry_count || 0)
      && duplicateMlVocabularyEntries === Number(mlValueInference.duplicate_vocabulary_entry_count || 0)
      && duplicateMlVocabularyNodes === Number(mlValueInference.duplicate_vocabulary_node_count || 0)
      && exactMlCategoryPairs === Number(mlValueInference.exact_category_pair_count || 0)
      && duplicateMlCategoryActiveKeys === Number(mlValueInference.duplicate_category_active_key_count || 0)
      && duplicateMlCategoryActiveKeyNodes === Number(mlValueInference.duplicate_category_active_key_node_count || 0)
      && featureVectorizerRows.length === Number(mlValueInference.feature_vectorizer_node_count || 0)
      && featureVectorizerExactWidthRows.length === Number(mlValueInference.feature_vectorizer_exact_width_node_count || 0)
      && exactFeatureVectorizerConfiguredFeatures === Number(mlValueInference.exact_feature_vectorizer_configured_feature_count || 0)
      && featureVectorizerTruncatingRows.length === Number(mlValueInference.feature_vectorizer_truncating_node_count || 0)
      && exactFeatureVectorizerTruncatedPerBatch === Number(mlValueInference.exact_feature_vectorizer_truncated_feature_count_per_batch || 0)
      && exactFeatureVectorizerPaddedPerBatch === Number(mlValueInference.exact_feature_vectorizer_padded_feature_count_per_batch || 0)
      && arrayFeatureExtractorRows.length === Number(mlValueInference.array_feature_extractor_node_count || 0)
      && arrayFeatureExtractorExactIndexRows.length === Number(mlValueInference.array_feature_extractor_exact_index_node_count || 0)
      && exactArrayFeatureExtractorIndices === Number(mlValueInference.exact_array_feature_extractor_index_count || 0)
      && arrayFeatureExtractorDuplicateIndices === Number(mlValueInference.array_feature_extractor_duplicate_index_count || 0)
      && arrayFeatureExtractorBoundsAssessedRows.length === Number(mlValueInference.array_feature_extractor_bounds_assessed_node_count || 0)
      && arrayFeatureExtractorBoundsFailureRows.length === Number(mlValueInference.array_feature_extractor_bounds_failure_node_count || 0)
      && binarizerRows.length === Number(mlValueInference.binarizer_node_count || 0)
      && binarizerExactStaticRows.length === Number(mlValueInference.binarizer_exact_static_node_count || 0)
      && exactBinarizerInputValues === Number(mlValueInference.exact_binarizer_input_value_count || 0)
      && exactBinarizerAboveThreshold === Number(mlValueInference.exact_binarizer_above_threshold_count || 0)
      && exactBinarizerAtOrBelowThreshold === Number(mlValueInference.exact_binarizer_at_or_below_threshold_count || 0)
      && exactBinarizerEqualThreshold === Number(mlValueInference.exact_binarizer_equal_threshold_count || 0)
      && exactBinarizerAboveThreshold + exactBinarizerAtOrBelowThreshold === exactBinarizerInputValues
      && binarizerSchemaDefaultRows.length === Number(mlValueInference.binarizer_schema_default_threshold_node_count || 0)
      && binarizerNonfiniteThresholdRows.length === Number(mlValueInference.binarizer_nonfinite_threshold_node_count || 0)
      && normalizerRows.length === Number(mlValueInference.normalizer_node_count || 0)
      && normalizerStaticAssessedRows.length === Number(mlValueInference.normalizer_static_assessed_node_count || 0)
      && normalizerMaterializedRows.length === Number(mlValueInference.normalizer_output_materialized_node_count || 0)
      && exactNormalizerInputValues === Number(mlValueInference.exact_normalizer_input_value_count || 0)
      && exactNormalizerZeroDivisorRows === Number(mlValueInference.exact_normalizer_zero_divisor_row_count || 0)
      && exactNormalizerNegativeMaxRows === Number(mlValueInference.exact_normalizer_negative_max_divisor_row_count || 0)
      && exactNormalizerIntegerRoundingValues === Number(mlValueInference.exact_normalizer_integer_float32_rounding_count || 0)
      && exactNormalizerSignedOverflowValues === Number(mlValueInference.exact_normalizer_signed_overflow_value_count || 0)
      && exactNormalizerNonfiniteOutputs === Number(mlValueInference.exact_normalizer_non_finite_output_count || 0)
      && exactNormalizerSignedZeroOutputs === Number(mlValueInference.exact_normalizer_signed_zero_output_count || 0)
      && normalizerDefaultModeRows.length === Number(mlValueInference.normalizer_schema_default_mode_node_count || 0)
      && scalerRows.length === Number(mlValueInference.scaler_node_count || 0)
      && scalerStaticAssessedRows.length === Number(mlValueInference.scaler_static_assessed_node_count || 0)
      && scalerMaterializedRows.length === Number(mlValueInference.scaler_output_materialized_node_count || 0)
      && scalerInvalidContractRows.length === Number(mlValueInference.scaler_invalid_runtime_contract_node_count || 0)
      && exactScalerInputValues === Number(mlValueInference.exact_scaler_input_value_count || 0)
      && exactScalerIntegerRoundingValues === Number(mlValueInference.exact_scaler_integer_float32_rounding_count || 0)
      && exactScalerNonfiniteParameters === Number(mlValueInference.exact_scaler_non_finite_parameter_count || 0)
      && exactScalerNonfiniteOutputs === Number(mlValueInference.exact_scaler_non_finite_output_count || 0)
      && exactScalerSignedZeroOutputs === Number(mlValueInference.exact_scaler_signed_zero_output_count || 0)
      && exactScalerZeroScales === Number(mlValueInference.exact_scaler_zero_scale_count || 0)
      && imputerRows.length === Number(mlValueInference.imputer_node_count || 0)
      && imputerStaticAssessedRows.length === Number(mlValueInference.imputer_static_assessed_node_count || 0)
      && imputerMaterializedRows.length === Number(mlValueInference.imputer_output_materialized_node_count || 0)
      && imputerInvalidContractRows.length === Number(mlValueInference.imputer_invalid_runtime_contract_node_count || 0)
      && imputerScalarFirstRows.length === Number(mlValueInference.imputer_scalar_first_fallback_node_count || 0)
      && imputerPinnedCpuDtypeGapRows.length === Number(mlValueInference.imputer_pinned_cpu_dtype_gap_node_count || 0)
      && exactImputerInputValues === Number(mlValueInference.exact_imputer_input_value_count || 0)
      && exactImputerReplacements === Number(mlValueInference.exact_imputer_replacement_count || 0)
      && exactImputerNanReplacements === Number(mlValueInference.exact_imputer_nan_replacement_count || 0)
      && exactImputerUnchanged === Number(mlValueInference.exact_imputer_unchanged_count || 0)
      && exactImputerReplacements + exactImputerUnchanged === exactImputerInputValues
      && exactImputerIgnoredValues === Number(mlValueInference.exact_imputer_ignored_imputed_value_count || 0)
      && exactImputerNonfiniteValues === Number(mlValueInference.exact_imputer_non_finite_imputed_value_count || 0)
      && exactImputerNonfiniteOutputs === Number(mlValueInference.exact_imputer_non_finite_output_count || 0)
      && exactImputerSignedZeroOutputs === Number(mlValueInference.exact_imputer_signed_zero_output_count || 0)
      && oneHotRows.length === Number(mlValueInference.onehot_encoder_node_count || 0)
      && oneHotStaticAssessedRows.length === Number(mlValueInference.onehot_static_assessed_node_count || 0)
      && oneHotMaterializedRows.length === Number(mlValueInference.onehot_output_materialized_node_count || 0)
      && oneHotInvalidContractRows.length === Number(mlValueInference.onehot_invalid_contract_node_count || 0)
      && oneHotDuplicateRows.length === Number(mlValueInference.onehot_duplicate_vocabulary_node_count || 0)
      && oneHotUnknownAllZeroRows.length === Number(mlValueInference.onehot_unknown_all_zero_node_count || 0)
      && oneHotGuaranteedFailureRows.length === Number(mlValueInference.onehot_guaranteed_runtime_failure_node_count || 0)
      && oneHotPinnedCpuDtypeGapRows.length === Number(mlValueInference.onehot_pinned_cpu_dtype_gap_node_count || 0)
      && oneHotNoncanonicalZerosRows.length === Number(mlValueInference.onehot_noncanonical_zeros_node_count || 0)
      && oneHotInvalidCastRows.length === Number(mlValueInference.onehot_unrepresentable_numeric_cast_node_count || 0)
      && exactOneHotInputValues === Number(mlValueInference.exact_onehot_input_value_count || 0)
      && exactOneHotMatchedInputs === Number(mlValueInference.exact_onehot_matched_input_count || 0)
      && exactOneHotUnknownInputs === Number(mlValueInference.exact_onehot_unknown_input_count || 0)
      && exactOneHotChangedCasts === Number(mlValueInference.exact_onehot_numeric_to_int64_changed_count || 0)
      && exactOneHotInvalidCasts === Number(mlValueInference.exact_onehot_numeric_to_int64_invalid_count || 0)
      && exactOneHotMatchedInputs + exactOneHotUnknownInputs + exactOneHotInvalidCasts === exactOneHotInputValues
      && exactOneHotOutputOnes === Number(mlValueInference.exact_onehot_output_one_count || 0)
      && exactOneHotOutputZeros === Number(mlValueInference.exact_onehot_output_zero_count || 0)
      && exactOneHotOutputOnes + exactOneHotOutputZeros === exactOneHotAssessedOutputElements
      && exactOneHotDuplicateCategories === Number(mlValueInference.exact_onehot_duplicate_category_count || 0)
      && exactOneHotUnreachableColumns === Number(mlValueInference.exact_onehot_unreachable_duplicate_column_count || 0)
      && exactOneHotDuplicateCategories === exactOneHotUnreachableColumns
      && labelEncoderRows.length === Number(mlValueInference.label_encoder_node_count || 0)
      && labelEncoderStaticRows.length === Number(mlValueInference.label_encoder_static_assessed_node_count || 0)
      && labelEncoderMaterializedRows.length === Number(mlValueInference.label_encoder_output_materialized_node_count || 0)
      && labelEncoderOnnxFailureRows.length === Number(mlValueInference.label_encoder_onnx_contract_failure_node_count || 0)
      && labelEncoderRuntimeFailureRows.length === Number(mlValueInference.label_encoder_pinned_ort_contract_failure_node_count || 0)
      && labelEncoderDtypeGapRows.length === Number(mlValueInference.label_encoder_pinned_cpu_dtype_pair_gap_node_count || 0)
      && labelEncoderDuplicateConflictRows.length === Number(mlValueInference.label_encoder_duplicate_semantic_conflict_node_count || 0)
      && labelEncoderNanConflictRows.length === Number(mlValueInference.label_encoder_nan_semantic_conflict_node_count || 0)
      && labelEncoderDefaultRows.length === Number(mlValueInference.label_encoder_default_path_node_count || 0)
      && labelEncoderOutputMismatchRows.length === Number(mlValueInference.label_encoder_schema_runtime_output_mismatch_node_count || 0)
      && exactLabelEncoderKeys === Number(mlValueInference.exact_label_encoder_key_count || 0)
      && exactLabelEncoderInputs === Number(mlValueInference.exact_label_encoder_input_value_count || 0)
      && exactLabelEncoderMatches === Number(mlValueInference.exact_label_encoder_match_count || 0)
      && exactLabelEncoderDefaults === Number(mlValueInference.exact_label_encoder_default_count || 0)
      && exactLabelEncoderMatches + exactLabelEncoderDefaults === exactLabelEncoderInputs
      && exactLabelEncoderDuplicateHits === Number(mlValueInference.exact_label_encoder_duplicate_key_hit_count || 0)
      && exactLabelEncoderMismatches === Number(mlValueInference.exact_label_encoder_schema_runtime_mismatch_count || 0)
      && linearClassifierRows.length === Number(mlValueInference.linear_classifier_node_count || 0)
      && linearRegressorRows.length === Number(mlValueInference.linear_regressor_node_count || 0)
      && linearOnnxFailureRows.length === Number(mlValueInference.linear_onnx_contract_failure_node_count || 0)
      && linearRuntimeFailureRows.length === Number(mlValueInference.linear_pinned_ort_contract_failure_node_count || 0)
      && linearReferenceRows.length === Number(mlValueInference.linear_reference_assessed_node_count || 0)
      && linearDtypeGapRows.length === Number(mlValueInference.linear_pinned_cpu_dtype_gap_node_count || 0)
      && linearPostTransformHazardRows.length === Number(mlValueInference.linear_post_transform_hazard_node_count || 0)
      && linearUnusedCoefficientRows.length === Number(mlValueInference.linear_unused_coefficient_node_count || 0)
      && linearIgnoredInterceptRows.length === Number(mlValueInference.linear_ignored_intercept_node_count || 0)
      && exactLinearCoefficients === Number(mlValueInference.exact_linear_coefficient_count || 0)
      && exactLinearUsedCoefficients === Number(mlValueInference.exact_linear_used_coefficient_count || 0)
      && exactLinearUnusedCoefficients === Number(mlValueInference.exact_linear_unused_coefficient_count || 0)
      && exactLinearUnresolvedCoefficientUses === Number(mlValueInference.exact_linear_unresolved_coefficient_use_count || 0)
      && exactLinearUsedCoefficients + exactLinearUnusedCoefficients + exactLinearUnresolvedCoefficientUses === exactLinearCoefficients
      && exactLinearIgnoredIntercepts === Number(mlValueInference.exact_linear_ignored_intercept_count || 0)
      && exactLinearReferenceInputs === Number(mlValueInference.exact_linear_reference_input_value_count || 0)
      && exactLinearReferenceScores === Number(mlValueInference.exact_linear_reference_raw_score_count || 0)
      && svmClassifierRows.length === Number(mlValueInference.svm_classifier_node_count || 0)
      && svmRegressorRows.length === Number(mlValueInference.svm_regressor_node_count || 0)
      && svmLinearRows.length === Number(mlValueInference.svm_linear_mode_node_count || 0)
      && svmSvcRows.length === Number(mlValueInference.svm_svc_mode_node_count || 0)
      && svmOnnxFailureRows.length === Number(mlValueInference.svm_onnx_contract_failure_node_count || 0)
      && svmPinnedRuntimeFailureRows.length === Number(mlValueInference.svm_pinned_ort_contract_failure_node_count || 0)
      && svmDtypeGapRows.length === Number(mlValueInference.svm_regressor_pinned_cpu_dtype_gap_node_count || 0)
      && svmScoreWidthMismatchRows.length === Number(mlValueInference.svm_schema_runtime_score_width_mismatch_node_count || 0)
      && svmIgnoredTransformRows.length === Number(mlValueInference.svm_ignored_post_transform_node_count || 0)
      && svmIgnoredParameterRows.length === Number(mlValueInference.svm_ignored_parameter_node_count || 0)
      && svmRows.filter((row) => Number(row.svm_non_finite_parameter_count || 0)
        + Number(row.svm_reference_non_finite_score_count || 0) > 0).length === Number(mlValueInference.svm_non_finite_node_count || 0)
      && svmReferenceRows.length === Number(mlValueInference.svm_reference_assessed_node_count || 0)
      && exactSvmVectors === Number(mlValueInference.exact_svm_vector_count || 0)
      && exactSvmPairs === Number(mlValueInference.exact_svm_pairwise_classifier_count || 0)
      && exactSvmSupportValues === Number(mlValueInference.exact_svm_support_vector_value_count || 0)
      && exactSvmUsedSupportValues === Number(mlValueInference.exact_svm_used_support_vector_value_count || 0)
      && exactSvmUnusedSupportValues === Number(mlValueInference.exact_svm_unused_support_vector_value_count || 0)
      && exactSvmUnresolvedSupportUses === Number(mlValueInference.exact_svm_unresolved_support_vector_use_count || 0)
      && exactSvmUsedSupportValues + exactSvmUnusedSupportValues + exactSvmUnresolvedSupportUses === exactSvmSupportValues
      && exactSvmCoefficients === Number(mlValueInference.exact_svm_coefficient_count || 0)
      && exactSvmUsedCoefficients === Number(mlValueInference.exact_svm_used_coefficient_count || 0)
      && exactSvmUnusedCoefficients === Number(mlValueInference.exact_svm_unused_coefficient_count || 0)
      && exactSvmUnresolvedCoefficientUses === Number(mlValueInference.exact_svm_unresolved_coefficient_use_count || 0)
      && exactSvmUsedCoefficients + exactSvmUnusedCoefficients + exactSvmUnresolvedCoefficientUses === exactSvmCoefficients
      && exactSvmRhos === Number(mlValueInference.exact_svm_rho_count || 0)
      && exactSvmUsedRhos === Number(mlValueInference.exact_svm_used_rho_count || 0)
      && exactSvmUnusedRhos === Number(mlValueInference.exact_svm_unused_rho_count || 0)
      && exactSvmUnresolvedRhoUses === Number(mlValueInference.exact_svm_unresolved_rho_use_count || 0)
      && exactSvmUsedRhos + exactSvmUnusedRhos + exactSvmUnresolvedRhoUses === exactSvmRhos
      && exactSvmReferenceInputs === Number(mlValueInference.exact_svm_reference_input_value_count || 0)
      && exactSvmReferenceScores === Number(mlValueInference.exact_svm_reference_raw_score_count || 0)
      && mlValueInference.status === (mlValueFailedRows.length ? "fail" : mlValuePartialRows.length ? "partial" : mlValueRows.length ? "assessed" : "not_applicable")
      && Number(shapeInference.propagated_static_value_tensor_count || 0) <= Number(shapeInference.node_output_count || 0)
      && Number(shapeInference.propagated_symbolic_shape_value_tensor_count || 0) <= Number(shapeInference.shape_contract_known_node_output_count || 0)
      && Number(shapeInference.known_node_output_count || 0) + Number(shapeInference.unknown_node_output_count || 0) + Number(shapeInference.non_dense_node_output_count || 0) === Number(shapeInference.node_output_count || 0)
      && Number(shapeInference.tensor_node_output_count || 0) + Number(shapeInference.non_dense_node_output_count || 0) === Number(shapeInference.node_output_count || 0)
      && Number(shapeInference.shape_contract_known_node_output_count || 0) + Number(shapeInference.shape_contract_unknown_node_output_count || 0) === Number(shapeInference.tensor_node_output_count || 0)
      && Number(shapeInference.known_node_output_count || 0) <= Number(shapeInference.shape_contract_known_node_output_count || 0)
      && Number(shapeInference.symbolic_shape_contract_node_output_count || 0) === Math.max(0,
        Number(shapeInference.shape_contract_known_node_output_count || 0) - Number(shapeInference.known_node_output_count || 0)
          - Number(shapeInference.conditional_shape_contract_node_output_count || 0))
      && Number(shapeInference.known_non_dense_node_output_count || 0) + Number(shapeInference.unresolved_non_dense_node_output_count || 0) === Number(shapeInference.non_dense_node_output_count || 0)
      && Number(shapeInference.known_value_node_output_count || 0) === Number(shapeInference.known_node_output_count || 0) + Number(shapeInference.known_non_dense_node_output_count || 0)
      && Number(shapeInference.inferred_outputs || 0) <= Number(shapeInference.known_node_output_count || 0)
      && Number(shapeInference.inferred_non_dense_outputs || 0) <= Number(shapeInference.known_non_dense_node_output_count || 0)
      && (shapeInference.unknown_tensor_indices || []).length === Number(shapeInference.unknown_tensor_count || 0)
      && (shapeInference.non_dense_value_indices || []).length === Number(shapeInference.non_dense_value_count || 0)
      && (shapeInference.non_dense_node_output_names || []).length === Number(shapeInference.non_dense_node_output_count || 0)
      && (shapeInference.known_non_dense_node_output_names || []).length === Number(shapeInference.known_non_dense_node_output_count || 0)
      && nullableClose(shapeInference.node_value_assessment_ratio, Number(shapeInference.node_output_count || 0) > 0
        ? Number(shapeInference.known_value_node_output_count || 0) / Number(shapeInference.node_output_count || 0) : 1)
      && nullableClose(shapeInference.node_output_assessment_ratio, Number(shapeInference.tensor_node_output_count || 0) > 0
        ? Number(shapeInference.known_node_output_count || 0) / Number(shapeInference.tensor_node_output_count || 0) : 1), "ONNX shape-inference rule, dense/non-dense output, unknown-tensor, or ratio ledgers do not conserve.", ["/evidence/static_analysis/onnx_shape_inference"]);
    check("CF-SHAPE-002", String(engineeringReport || "").includes("## ONNX Shape Inference Coverage")
      && String(engineeringReport || "").includes(ANALYZER_METADATA.schemas.onnxShapeInference)
      && String(engineeringReport || "").includes(shapeInference.source_commit)
      && shapeSources.every((source) => String(engineeringReport || "").includes(source.sha256))
      && (extendedShape.source_documents || []).every((source) => String(engineeringReport || "").includes(source.sha256))
      && String(engineeringReport || "").includes("OpSchema formal contract")
      && String(engineeringReport || "").includes("OperatorSetIdProto imports")
      && String(engineeringReport || "").includes("Extended shape scope")
      && String(engineeringReport || "").includes("Recursive scope engine")
      && String(engineeringReport || "").includes("Sequence / Optional value engine")
      && containerSources.every((source) => String(engineeringReport || "").includes(source.sha256))
      && (!containerRows.length || String(engineeringReport || "").includes("Sequence / Optional Value Contracts"))
      && String(engineeringReport || "").includes("ONNX-ML value-contract engine")
      && String(engineeringReport || "").includes(ANALYZER_METADATA.schemas.onnxMlValueInference)
      && mlValueSources.every((source) => String(engineeringReport || "").includes(source.sha256))
      && String(engineeringReport || "").includes(mlValueInference.runtime_reference_commit)
      && mlValueRuntimeSources.every((source) => String(engineeringReport || "").includes(source.sha256))
      && (!mlValueRows.length || String(engineeringReport || "").includes("ONNX-ML Value Contracts"))
      && (!treeRows.length || String(engineeringReport || "").includes("TreeEnsemble topology/runtime contracts"))
      && (!shapeConflicts.length || String(engineeringReport || "").includes("Declared Vs Inferred Shape Contract Conflicts"))
      && (!invalidShapeSchemaRows.length || String(engineeringReport || "").includes("OpSchema Formal Contract Failures"))
      && (!invalidOpsetImportRows.length || String(engineeringReport || "").includes("OperatorSet Import Contract Failures"))
      && (!shapeScopeRows.length || String(engineeringReport || "").includes("Recursive Shape Scope Assessment"))
      && (!extendedFunctionRows.length || String(engineeringReport || "").includes("FunctionProto Call Contracts"))
      && (!extendedControlRows.length || String(engineeringReport || "").includes("If / Loop / Scan Shape Contracts"))
      && (!extendedLoopRows.length || String(engineeringReport || "").includes("Bounded exact Loop expansion")
        && String(engineeringReport || "").includes("Exact Loop expansion"))
      && (!extendedLoopRows.some((row) => (row.exact_iteration_state_contracts || []).length || (row.exact_final_state_contracts || []).length)
        || String(engineeringReport || "").includes("Exact Loop State Ledger"))
      && (!extendedLoopRows.some((row) => (row.exact_nested_failure_rows || []).length)
        || String(engineeringReport || "").includes("Reached Loop Contract Failures"))
      && (!extendedSequenceMapRows.length || String(engineeringReport || "").includes("SequenceMap Value Contracts"))
      && (!extendedScopeRows.length || String(engineeringReport || "").includes("Recursive Engine Execution Ledger"))
      && String(engineeringReport || "").includes("All-kind output contract coverage")
      && (!shapeScopeExclusions.length || String(engineeringReport || "").includes("Extended Shape Scope Exclusions")), "Engineering report does not preserve ONNX shape-inference provenance, schema-form/scope coverage, conflict evidence, or its interpretation boundary.", ["/evidence/static_analysis/onnx_shape_inference", "/engineering_report.md"]);
    const shapeUnknown = new Set(shapeInference?.unknown_tensor_indices || []);
    const independentlyUnknown = (staticAnalysis?.tensors || [])
      .filter((tensor) => !tensor.value_kind || ["tensor", "unresolved", "undefined"].includes(tensor.value_kind))
      .filter((tensor) => tensor.contract_status === "invalid" || tensor.shape_declared !== true || !Array.isArray(tensor.shape) || tensor.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 0))
      .map((tensor) => tensor.index);
    check("CF-SHAPE-003", shapeUnknown.size === independentlyUnknown.length
      && independentlyUnknown.every((index) => shapeUnknown.has(index)), "ONNX shape coverage must treat an explicitly declared empty shape as a rank-0 scalar and distinguish it from an absent or unresolved shape.", ["/evidence/static_analysis/tensors", "/evidence/static_analysis/onnx_shape_inference/unknown_tensor_indices"]);
    const denseNodeOutputIndices = new Set((staticAnalysis?.ops || []).flatMap((op) => op.outputs || []).map(Number).filter((index) => index >= 0));
    const denseNodeOutputTensors = (staticAnalysis?.tensors || []).filter((tensor) => denseNodeOutputIndices.has(Number(tensor.index))
      && (!tensor.value_kind || ["tensor", "unresolved", "undefined"].includes(tensor.value_kind)));
    const independentlyCompleteShapeContracts = denseNodeOutputTensors.filter((tensor) => {
      if (tensor.contract_status === "invalid" || !tensor.dtype || tensor.dtype === "UNKNOWN") return false;
      const dimensions = tensor.type_proto?.shapeDimensions || [];
      const unconditional = tensor.shape_declared === true && Array.isArray(tensor.shape)
        && tensor.shape.every((dimension, index) => Number.isSafeInteger(dimension) && dimension >= 0
          || dimensions[index]?.kind === "symbolic" && Boolean(String(dimensions[index]?.parameter || "")));
      const conditional = tensor.conditional_shape_contract?.status === "assessed_complete"
        && Array.isArray(tensor.conditional_shape_variants) && tensor.conditional_shape_variants.length > 0
        && tensor.conditional_shape_variants.every((variant) => variant.shape_declared === true && variant.dtype && variant.dtype !== "UNKNOWN"
          && Array.isArray(variant.shape) && variant.shape.every((dimension, index) => Number.isSafeInteger(dimension) && dimension >= 0
            || variant.type_proto?.shapeDimensions?.[index]?.kind === "symbolic"
              && Boolean(String(variant.type_proto.shapeDimensions[index].parameter || ""))));
      return unconditional || conditional;
    }).length;
    const independentlyPartialConditionalShapeContracts = denseNodeOutputTensors.filter((tensor) => (
      tensor.conditional_shape_contract?.status === "assessed_partial"
    )).length;
    const independentlyConditionallyInvalidShapeOutputs = denseNodeOutputTensors.filter((tensor) => (
      (tensor.conditional_shape_contract?.variant_failures || []).some((row) => row?.status === "invalid")
    )).length;
    const independentlyConditionalInvalidVariants = denseNodeOutputTensors.reduce((total, tensor) => total
      + (tensor.conditional_shape_contract?.variant_failures || []).filter((row) => row?.status === "invalid").length, 0);
    const independentlyConditionalUnassessedVariants = denseNodeOutputTensors.reduce((total, tensor) => total
      + (tensor.conditional_shape_contract?.variant_failures || []).filter((row) => row?.status !== "invalid").length, 0);
    check("CF-SHAPE-003B", Number(shapeInference?.shape_contract_known_node_output_count ?? independentlyCompleteShapeContracts) === independentlyCompleteShapeContracts
      && Number(shapeInference?.shape_contract_unknown_node_output_count ?? denseNodeOutputTensors.length - independentlyCompleteShapeContracts) === denseNodeOutputTensors.length - independentlyCompleteShapeContracts
      && Number(shapeInference?.partial_conditional_shape_contract_node_output_count || 0) === independentlyPartialConditionalShapeContracts
      && Number(shapeInference?.conditionally_invalid_node_output_count || 0) === independentlyConditionallyInvalidShapeOutputs
      && Number(shapeInference?.conditional_invalid_variant_count || 0) === independentlyConditionalInvalidVariants
      && Number(shapeInference?.conditional_unassessed_variant_count || 0) === independentlyConditionalUnassessedVariants
      && String(engineeringReport || "").includes("Complete dense shape contracts"), "ONNX symbolic shape-contract coverage must be independently conserved and remain distinct from numeric concreteness.", ["/evidence/static_analysis/tensors", "/evidence/static_analysis/onnx_shape_inference/shape_contract_known_node_output_count", "/engineering_report.md"]);
    const shapeConflictFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0006");
    check("CF-SHAPE-004", shapeConflicts.length
      ? Boolean(shapeConflictFinding)
        && shapeConflictFinding.technical_priority === "High"
        && (shapeConflictFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/declaration_conflicts")
      : !shapeConflictFinding, "EA-ONX-0006 must exist exactly when deterministic inferred tensor contracts contradict artifact declarations.", ["/evidence/static_analysis/onnx_shape_inference/declaration_conflicts", "/evidence/findings_register/findings"]);
    const semanticShapeConflictFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0071");
    check("CF-SHAPE-004B", semanticShapeConflicts.length
      ? Boolean(semanticShapeConflictFinding)
        && semanticShapeConflictFinding.technical_priority === "High"
        && (semanticShapeConflictFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/semantic_contract_conflicts")
      : !semanticShapeConflictFinding, "EA-ONX-0071 must exist exactly when artifact-known values violate a pinned ONNX operator contract.", ["/evidence/static_analysis/onnx_shape_inference/semantic_contract_conflicts", "/evidence/findings_register/findings"]);
    const conditionalShapeConflictFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0072");
    check("CF-SHAPE-004C", conditionallyInvalidShapeOutputs > 0
      ? Boolean(conditionalShapeConflictFinding)
        && conditionalShapeConflictFinding.technical_priority === "High"
        && (conditionalShapeConflictFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/conditionally_invalid_node_output_count")
        && (conditionalShapeConflictFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !conditionalShapeConflictFinding, "EA-ONX-0072 must exist exactly when a finite runtime branch has an artifact-derived invalid downstream tensor contract.", ["/evidence/static_analysis/onnx_shape_inference/conditionally_invalid_node_output_count", "/evidence/static_analysis/tensors", "/evidence/findings_register/findings"]);
    const shapeSchemaFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0007");
    check("CF-SHAPE-006", invalidShapeSchemaRows.length || invalidOpsetImportRows.length
      ? Boolean(shapeSchemaFinding)
        && shapeSchemaFinding.technical_priority === "High"
        && (shapeSchemaFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/opset_import_contract")
        && (shapeSchemaFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/schema_form_rows")
      : !shapeSchemaFinding, "EA-ONX-0007 must exist exactly when a supported node violates the pinned imported-opset formal schema.", ["/evidence/static_analysis/onnx_shape_inference/schema_form_rows", "/evidence/findings_register/findings"]);
    const shapeScopeFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0008");
    check("CF-SHAPE-007", shapeScopeExclusions.length || Number(shapeScope.reachable_scope_unresolved_output_count || 0) > 0
      || extendedControlRows.some((row) => row.status === "partial")
      || extendedSequenceMapRows.some((row) => row.status === "partial")
      || extendedScopeRows.some((row) => row.status === "partial")
      || mlValuePartialRows.length
      ? Boolean(shapeScopeFinding)
        && ["High", "Medium"].includes(shapeScopeFinding.technical_priority)
        && (shapeScopeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/shape_scope")
        && (shapeScopeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/extended_scope_inference")
        && (shapeScopeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/partial_rows")
      : !shapeScopeFinding, "EA-ONX-0008 must exist exactly when reachable recursively evaluated scopes retain unassessed nodes or unresolved outputs.", ["/evidence/static_analysis/onnx_shape_inference/shape_scope", "/evidence/findings_register/findings"]);
    const recursiveShapeFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0009");
    check("CF-SHAPE-008", Number(extendedShape.local_function_call_fail_count || 0) + Number(extendedShape.control_flow_fail_count || 0) + Number(extendedShape.sequence_map_fail_count || 0) > 0
      ? Boolean(recursiveShapeFinding)
        && recursiveShapeFinding.technical_priority === "High"
        && (recursiveShapeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/extended_scope_inference")
        && (recursiveShapeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/shape_scope")
      : !recursiveShapeFinding, "EA-ONX-0009 must exist exactly when a reachable FunctionProto, If/Loop/Scan, or SequenceMap recursive shape contract fails.", ["/evidence/static_analysis/onnx_shape_inference/extended_scope_inference", "/evidence/findings_register/findings"]);
    const containerValueFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0012");
    check("CF-SHAPE-009", containerFailedRows.length
      ? Boolean(containerValueFinding)
        && containerValueFinding.technical_priority === "High"
        && (containerValueFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/container_value_inference/failed_rows")
        && (containerValueFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/container_value_inference/source_documents")
      : !containerValueFinding, "EA-ONX-0012 must exist exactly when a pinned direct Sequence/Optional value contract fails.", ["/evidence/static_analysis/onnx_shape_inference/container_value_inference/failed_rows", "/evidence/findings_register/findings"]);
    const mlValueFailureFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0013");
    check("CF-SHAPE-010", mlValueFailedRows.length
      ? Boolean(mlValueFailureFinding)
        && mlValueFailureFinding.technical_priority === "High"
        && (mlValueFailureFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/failed_rows")
        && (mlValueFailureFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents")
      : !mlValueFailureFinding, "EA-ONX-0013 must exist exactly when a pinned ONNX-ML value contract fails.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/failed_rows", "/evidence/findings_register/findings"]);
    const mlValueDuplicateFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0014");
    check("CF-SHAPE-011", duplicateMlClassKeyNodes > 0
      ? Boolean(mlValueDuplicateFinding)
        && mlValueDuplicateFinding.technical_priority === "High"
        && (mlValueDuplicateFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueDuplicateFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/source_documents")
      : !mlValueDuplicateFinding, "EA-ONX-0014 must exist exactly when ZipMap labels contain duplicate map keys.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueDuplicateVocabularyFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0015");
    check("CF-SHAPE-012", duplicateDictVocabularyRows.length > 0
      ? Boolean(mlValueDuplicateVocabularyFinding)
        && mlValueDuplicateVocabularyFinding.technical_priority === "Medium"
        && (mlValueDuplicateVocabularyFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueDuplicateVocabularyFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueDuplicateVocabularyFinding, "EA-ONX-0015 must exist exactly when DictVectorizer vocabulary repeats output columns.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueDuplicateCategoryFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0016");
    check("CF-SHAPE-013", duplicateMlCategoryActiveKeyNodes > 0
      ? Boolean(mlValueDuplicateCategoryFinding)
        && mlValueDuplicateCategoryFinding.technical_priority === "High"
        && (mlValueDuplicateCategoryFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueDuplicateCategoryFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueDuplicateCategoryFinding, "EA-ONX-0016 must exist exactly when CategoryMapper active-direction keys overwrite earlier mappings.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueFeatureTruncationFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0017");
    check("CF-SHAPE-014", featureVectorizerTruncatingRows.length > 0
      ? Boolean(mlValueFeatureTruncationFinding)
        && mlValueFeatureTruncationFinding.technical_priority === "Medium"
        && (mlValueFeatureTruncationFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueFeatureTruncationFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueFeatureTruncationFinding, "EA-ONX-0017 must exist exactly when FeatureVectorizer discards artifact-known input features.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueBinarizerRiskFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0018");
    check("CF-SHAPE-015", riskyBinarizerRows.length > 0
      ? Boolean(mlValueBinarizerRiskFinding)
        && mlValueBinarizerRiskFinding.technical_priority === "High"
        && (mlValueBinarizerRiskFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueBinarizerRiskFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueBinarizerRiskFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueBinarizerRiskFinding, "EA-ONX-0018 must exist exactly when Binarizer has a non-finite threshold or statically non-finite FLOAT input.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueBinarizerDtypeFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0019");
    check("CF-SHAPE-016", ortUnsupportedBinarizerRows.length > 0
      ? Boolean(mlValueBinarizerDtypeFinding)
        && mlValueBinarizerDtypeFinding.technical_priority === "High"
        && (mlValueBinarizerDtypeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueBinarizerDtypeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueBinarizerDtypeFinding, "EA-ONX-0019 must exist exactly when a schema-valid Binarizer dtype lacks a pinned ORT CPU kernel.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueNormalizerOverflowFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0020");
    check("CF-SHAPE-017", normalizerOverflowRows.length > 0
      ? Boolean(mlValueNormalizerOverflowFinding)
        && mlValueNormalizerOverflowFinding.technical_priority === "High"
        && (mlValueNormalizerOverflowFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueNormalizerOverflowFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueNormalizerOverflowFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueNormalizerOverflowFinding, "EA-ONX-0020 must exist exactly when pinned Normalizer integer abs/square arithmetic provably overflows.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueNormalizerNegativeMaxFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0021");
    check("CF-SHAPE-018", normalizerNegativeMaxRows.length > 0
      ? Boolean(mlValueNormalizerNegativeMaxFinding)
        && mlValueNormalizerNegativeMaxFinding.technical_priority === "Medium"
        && (mlValueNormalizerNegativeMaxFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueNormalizerNegativeMaxFinding, "EA-ONX-0021 must exist exactly when a static MAX Normalizer row has a negative signed divisor.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueNormalizerRoundingFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0022");
    check("CF-SHAPE-019", normalizerIntegerRoundingRows.length > 0
      ? Boolean(mlValueNormalizerRoundingFinding)
        && mlValueNormalizerRoundingFinding.technical_priority === "Medium"
        && (mlValueNormalizerRoundingFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueNormalizerRoundingFinding, "EA-ONX-0022 must exist exactly when a static integer Normalizer input changes under the required FLOAT32 projection.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueNormalizerNonfiniteFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0023");
    check("CF-SHAPE-020", normalizerNonfiniteRows.length > 0
      ? Boolean(mlValueNormalizerNonfiniteFinding)
        && mlValueNormalizerNonfiniteFinding.technical_priority === "High"
        && (mlValueNormalizerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueNormalizerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueNormalizerNonfiniteFinding, "EA-ONX-0023 must exist exactly when a Normalizer input or static FLOAT32 projection is non-finite.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-021", !normalizerRows.length || String(engineeringReport || "").includes("Normalizer exact row effects")
      && String(engineeringReport || "").includes("signed overflow")
      && String(engineeringReport || "").includes("integer->FLOAT32 changed")
      && String(engineeringReport || "").includes("signed-zero outputs"), "Engineering Report must expose Normalizer aggregate and row-level arithmetic evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    check("CF-SHAPE-022", String(engineeringReport || "").includes("JSON-safe static signed-zero ledger")
      && String(engineeringReport || "").includes(`${formatIntegerForConformance(exactStaticSignedZeroValues)} signed-zero value(s)`)
      && String(engineeringReport || "").includes(`${formatIntegerForConformance(staticSignedZeroTensors.length)} tensor(s)`),
    "Engineering Report must expose the JSON-safe signed-zero ledger totals.", ["/evidence/static_analysis/tensors", "/engineering_report.md"]);
    check("CF-SHAPE-056", String(engineeringReport || "").includes("Canonical static non-finite / unsafe-value ledger")
      && String(engineeringReport || "").includes(`${formatIntegerForConformance(exactStaticCanonicalTextValues)} canonical value(s)`)
      && String(engineeringReport || "").includes(`${formatIntegerForConformance(staticCanonicalTextTensors.length)} tensor(s)`),
    "Engineering Report must expose canonical non-finite and unsafe static-value ledger totals.", ["/evidence/static_analysis/tensors", "/engineering_report.md"]);
    const mlValueScalerContractFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0024");
    check("CF-SHAPE-023", scalerInvalidContractRows.length > 0
      ? Boolean(mlValueScalerContractFinding)
        && mlValueScalerContractFinding.technical_priority === "High"
        && (mlValueScalerContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows")
        && (mlValueScalerContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueScalerContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/ops")
      : !mlValueScalerContractFinding, "EA-ONX-0024 must exist exactly when a Scaler contract is invalid for the pinned ORT CPU implementation.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueScalerPrecisionFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0025");
    check("CF-SHAPE-024", scalerPrecisionRows.length > 0
      ? Boolean(mlValueScalerPrecisionFinding)
        && mlValueScalerPrecisionFinding.technical_priority === "Medium"
        && (mlValueScalerPrecisionFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueScalerPrecisionFinding, "EA-ONX-0025 must exist exactly when Scaler integer input values change under FLOAT32 projection.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueScalerNonfiniteFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0026");
    check("CF-SHAPE-025", scalerNonfiniteRows.length > 0
      ? Boolean(mlValueScalerNonfiniteFinding)
        && mlValueScalerNonfiniteFinding.technical_priority === "High"
        && (mlValueScalerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueScalerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
        && (mlValueScalerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/ops")
      : !mlValueScalerNonfiniteFinding, "EA-ONX-0026 must exist exactly when Scaler has a non-finite parameter, input, or output path.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-026", !scalerRows.length || String(engineeringReport || "").includes("Scaler exact affine effects")
      && String(engineeringReport || "").includes("integer->FLOAT32 changed")
      && String(engineeringReport || "").includes("non-finite parameters")
      && String(engineeringReport || "").includes("signed-zero outputs"), "Engineering Report must expose Scaler aggregate and row-level arithmetic evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    const mlValueImputerContractFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0027");
    check("CF-SHAPE-027", imputerInvalidContractRows.length > 0
      ? Boolean(mlValueImputerContractFinding)
        && mlValueImputerContractFinding.technical_priority === "High"
        && (mlValueImputerContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
        && (mlValueImputerContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/ops")
      : !mlValueImputerContractFinding, "EA-ONX-0027 must exist exactly when an Imputer contract is invalid for the pinned ORT CPU implementation.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueImputerFallbackFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0028");
    check("CF-SHAPE-028", imputerScalarFirstRows.length > 0
      ? Boolean(mlValueImputerFallbackFinding)
        && mlValueImputerFallbackFinding.technical_priority === "High"
        && (mlValueImputerFallbackFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueImputerFallbackFinding, "EA-ONX-0028 must exist exactly when pinned ORT Imputer scalar-first fallback ignores trailing values.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueImputerDtypeFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0029");
    check("CF-SHAPE-029", imputerPinnedCpuDtypeGapRows.length > 0
      ? Boolean(mlValueImputerDtypeFinding)
        && mlValueImputerDtypeFinding.technical_priority === "High"
        && (mlValueImputerDtypeFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueImputerDtypeFinding, "EA-ONX-0029 must exist exactly when an Imputer dtype is schema-valid but lacks a pinned ORT CPU kernel.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueImputerNonfiniteFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0030");
    check("CF-SHAPE-030", imputerNonfiniteRows.length > 0
      ? Boolean(mlValueImputerNonfiniteFinding)
        && mlValueImputerNonfiniteFinding.technical_priority === "High"
        && (mlValueImputerNonfiniteFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/tensors")
      : !mlValueImputerNonfiniteFinding, "EA-ONX-0030 must exist exactly when Imputer has a non-finite imputed or output path.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-031", !imputerRows.length || String(engineeringReport || "").includes("Imputer exact replacement effects")
      && String(engineeringReport || "").includes("scalar-first fallback")
      && String(engineeringReport || "").includes("NaN-marker replacement")
      && String(engineeringReport || "").includes("ignored trailing imputed value"), "Engineering Report must expose Imputer aggregate and row-level replacement evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    const mlValueOneHotContractFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0031");
    check("CF-SHAPE-032", oneHotInvalidContractRows.length > 0
      ? Boolean(mlValueOneHotContractFinding) && mlValueOneHotContractFinding.technical_priority === "High"
        && (mlValueOneHotContractFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/onnx_shape_inference/ml_value_inference/runtime_reference_documents")
      : !mlValueOneHotContractFinding, "EA-ONX-0031 must exist exactly when a OneHotEncoder category contract is invalid.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotDuplicateFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0032");
    check("CF-SHAPE-033", oneHotDuplicateRows.length > 0
      ? Boolean(mlValueOneHotDuplicateFinding) && mlValueOneHotDuplicateFinding.technical_priority === "Medium"
      : !mlValueOneHotDuplicateFinding, "EA-ONX-0032 must exist exactly when duplicate OneHotEncoder categories create unreachable columns.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotUnknownFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0033");
    check("CF-SHAPE-034", oneHotUnknownAllZeroRows.length > 0
      ? Boolean(mlValueOneHotUnknownFinding) && mlValueOneHotUnknownFinding.technical_priority === "Medium"
      : !mlValueOneHotUnknownFinding, "EA-ONX-0033 must exist exactly when exact unknown categories map to all-zero slices.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotFailureFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0034");
    check("CF-SHAPE-035", oneHotGuaranteedFailureRows.length > 0
      ? Boolean(mlValueOneHotFailureFinding) && mlValueOneHotFailureFinding.technical_priority === "High"
      : !mlValueOneHotFailureFinding, "EA-ONX-0034 must exist exactly when exact OneHotEncoder inputs prove an unknown-category runtime failure.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotDtypeFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0035");
    check("CF-SHAPE-036", oneHotPinnedCpuDtypeGapRows.length > 0
      ? Boolean(mlValueOneHotDtypeFinding) && mlValueOneHotDtypeFinding.technical_priority === "High"
      : !mlValueOneHotDtypeFinding, "EA-ONX-0035 must exist exactly when a schema-valid OneHotEncoder dtype lacks a pinned ORT CPU kernel.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotZerosFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0036");
    check("CF-SHAPE-037", oneHotNoncanonicalZerosRows.length > 0
      ? Boolean(mlValueOneHotZerosFinding) && mlValueOneHotZerosFinding.technical_priority === "Medium"
      : !mlValueOneHotZerosFinding, "EA-ONX-0036 must exist exactly when OneHotEncoder zeros is outside canonical boolean values.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const mlValueOneHotCastFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-ONX-0037");
    check("CF-SHAPE-038", oneHotInvalidCastRows.length > 0
      ? Boolean(mlValueOneHotCastFinding) && mlValueOneHotCastFinding.technical_priority === "High"
      : !mlValueOneHotCastFinding, "EA-ONX-0037 must exist exactly when an exact numeric OneHotEncoder input is not representable as INT64.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-039", !oneHotRows.length || String(engineeringReport || "").includes("OneHotEncoder exact encoding effects")
      && String(engineeringReport || "").includes("numeric truncation change")
      && String(engineeringReport || "").includes("unreachable duplicate column")
      && String(engineeringReport || "").includes("guaranteed runtime failure"), "Engineering Report must expose OneHotEncoder aggregate and row-level encoding evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    const linearFinding = (id) => (findingsRegister?.findings || []).find((item) => item.finding_id === id);
    check("CF-SHAPE-040", linearRuntimeFailureRows.length > 0
      ? linearFinding("EA-ONX-0038")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0038"), "EA-ONX-0038 must exist exactly when a pinned ORT linear-model contract fails.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-041", linearIgnoredParameterRows.length > 0
      ? linearFinding("EA-ONX-0039")?.technical_priority === "Medium"
      : !linearFinding("EA-ONX-0039"), "EA-ONX-0039 must exist exactly when the pinned linear kernel ignores serialized coefficients or intercepts.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-042", linearDtypeGapRows.length > 0
      ? linearFinding("EA-ONX-0040")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0040"), "EA-ONX-0040 must exist exactly when a schema-valid LinearRegressor dtype lacks the pinned ORT CPU compute path.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-043", linearSchemaDefaultTargetRows.every((row) => row.linear_targets_value === "1"
      && row.linear_pinned_ort_contract_reason !== "linear_regressor_pinned_ort_requires_explicit_targets")
      && !linearFinding("EA-ONX-0041"), "LinearRegressor targets=1 schema defaults must be treated as ORT-resolved runtime input and must not emit the retired false-positive EA-ONX-0041.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-044", linearPostTransformHazardRows.length > 0
      ? linearFinding("EA-ONX-0042")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0042"), "EA-ONX-0042 must exist exactly when pinned source proves a linear post-transform semantic gap or unsafe branch.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-045", linearMultiClassRows.length > 0
      ? linearFinding("EA-ONX-0043")?.technical_priority === "Medium"
      : !linearFinding("EA-ONX-0043"), "EA-ONX-0043 must exist exactly when LinearClassifier multi_class is nonzero but not consulted by pinned ORT compute.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-046", linearDuplicateLabelRows.length > 0
      ? linearFinding("EA-ONX-0044")?.technical_priority === "Medium"
      : !linearFinding("EA-ONX-0044"), "EA-ONX-0044 must exist exactly when LinearClassifier has duplicate externally visible labels.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-047", linearNumericalRiskRows.length > 0
      ? linearFinding("EA-ONX-0045")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0045"), "EA-ONX-0045 must exist exactly when a linear model has non-finite state, scores, or a reference decision-boundary case.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-048", !linearRows.length || String(engineeringReport || "").includes("Linear model coefficient/runtime contracts")
      && String(engineeringReport || "").includes("reference values are not claimed runtime-bit-exact")
      && String(engineeringReport || "").includes("coefficients expected/used/serialized"), "Engineering Report must expose linear-model aggregate, parameter conservation, runtime contract, output-shape, and reference-boundary evidence.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    check("CF-SHAPE-049", labelEncoderDtypeGapRows.length > 0
      ? linearFinding("EA-ONX-0046")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0046"), "EA-ONX-0046 must exist exactly when a schema-valid LabelEncoder dtype pair lacks a pinned ORT CPU registration.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-050", labelEncoderDuplicateConflictRows.length > 0 || labelEncoderRows.some((row) => (row.risk_codes || []).includes("label_encoder_v1_duplicate_class_runtime_last_index"))
      ? linearFinding("EA-ONX-0047")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0047"), "EA-ONX-0047 must exist exactly when LabelEncoder duplicate-key ownership is schema/runtime-sensitive.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-051", labelEncoderNanConflictRows.length > 0
      ? linearFinding("EA-ONX-0048")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0048"), "EA-ONX-0048 must exist exactly when LabelEncoder-2 contains a NaN key with the pinned source equality gap.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-052", labelEncoderDefaultRows.length > 0
      ? linearFinding("EA-ONX-0049")?.technical_priority === "Medium"
      : !linearFinding("EA-ONX-0049"), "EA-ONX-0049 must exist exactly when artifact-known LabelEncoder inputs reach the default path.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    const labelEncoderNonfiniteRows = labelEncoderRows.filter((row) => (row.risk_codes || []).includes("label_encoder_non_finite_mapping_state"));
    check("CF-SHAPE-053", labelEncoderNonfiniteRows.length > 0
      ? linearFinding("EA-ONX-0050")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0050"), "EA-ONX-0050 must exist exactly when LabelEncoder mapping state contains NaN or infinity.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-054", !labelEncoderRows.length || String(engineeringReport || "").includes("LabelEncoder exact mapping effects")
      && String(engineeringReport || "").includes("schema/runtime mismatches")
      && String(engineeringReport || "").includes("first-key / last-key"), "Engineering Report must expose LabelEncoder version, mapping conservation, default use, and schema/runtime ownership boundaries.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    check("CF-SHAPE-055", labelEncoderRuntimeInvalidRows.length > 0
      ? linearFinding("EA-ONX-0051")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0051"), "EA-ONX-0051 must exist exactly when a LabelEncoder key/value cardinality mismatch deterministically violates the pinned ORT CPU contract.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-056", svmRuntimeInvalidRows.length > 0
      ? linearFinding("EA-ONX-0052")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0052"), "EA-ONX-0052 must exist exactly when a pinned ORT SVM executable contract fails.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-057", svmScoreWidthMismatchRows.length > 0
      ? linearFinding("EA-ONX-0053")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0053"), "EA-ONX-0053 must exist exactly when SVMClassifier schema and pinned ORT score widths differ.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-058", svmDtypeGapRows.length > 0
      ? linearFinding("EA-ONX-0054")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0054"), "EA-ONX-0054 must exist exactly when a schema-valid SVMRegressor dtype lacks the pinned ORT CPU kernel.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-059", svmIgnoredTransformRows.length > 0
      ? linearFinding("EA-ONX-0055")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0055"), "EA-ONX-0055 must exist exactly when pinned ORT ignores a non-NONE SVMRegressor post_transform.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-060", svmIgnoredParameterRows.length > 0 || svmForcedKernelRows.length > 0
      ? linearFinding("EA-ONX-0056")?.technical_priority === "Medium"
      : !linearFinding("EA-ONX-0056"), "EA-ONX-0056 must exist exactly when pinned ORT ignores serialized SVM parameters or forces LINEAR kernel mode.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-061", svmNumericalRiskRows.length > 0
      ? linearFinding("EA-ONX-0057")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0057"), "EA-ONX-0057 must exist exactly for non-finite SVM state/reference results or static decision-boundary cases.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-062", svmSemanticHazardRows.length > 0
      ? linearFinding("EA-ONX-0058")?.technical_priority === "High"
      : !linearFinding("EA-ONX-0058"), "EA-ONX-0058 must exist exactly for source-backed SVM label, transform, binary-score, or one_class semantic hazards.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference/rows", "/evidence/findings_register/findings"]);
    check("CF-SHAPE-063", !svmRows.length || String(engineeringReport || "").includes("SVM support-vector/runtime contracts")
      && String(engineeringReport || "").includes("support/coefficients/rho expected/used/serialized")
      && String(engineeringReport || "").includes("reference values are not claimed runtime-bit-exact")
      && String(engineeringReport || "").includes("schema score width / pinned ORT score width"), "Engineering Report must expose SVM mode, parameter conservation, schema/runtime score width, runtime contract, and scalar-reference boundaries.", ["/evidence/static_analysis/onnx_shape_inference/ml_value_inference", "/engineering_report.md"]);
    registerOnnxTreeConformanceChecks({
      check,
      facts: treeFacts,
      finding: linearFinding,
      engineeringReport,
    });
    registerOnnxTfIdfConformanceChecks({
      check,
      inference: tfidfInference,
      tensors: staticAnalysis?.tensors || [],
      ops: staticAnalysis?.ops || [],
      findings: findingsRegister?.findings || [],
      engineeringReport,
    });
    check("CF-FORMAT-001", !("has_signature_defs" in metadata) && !("signature_count" in metadata) && !("signature_keys" in metadata), "ONNX metadata leaked TFLite signature fields.", ["/evidence/static_analysis/metadata_presence"]);
    check("CF-FORMAT-002", !Array.isArray(staticAnalysis?.structural_watchlist), "ONNX export used the retired structural_watchlist field.", ["/evidence/static_analysis"]);
    const softmax = (staticAnalysis?.runtime_review_watchlist || []).find((item) => item.name === "Softmax");
    check("CF-FORMAT-003", !softmax || softmax.reason_code === "OUTPUT_REDUCTION_KERNEL", "ONNX Softmax must be classified as an output reduction kernel.", ["/evidence/static_analysis/runtime_review_watchlist"]);
    check("CF-FORMAT-004", !String(engineeringReport || "").includes("INT8 conversion may reduce eligible tensor payload width"), "ONNX report contains a TFLite/INT8 conversion recommendation that was not assessed.", ["/engineering_report.md"]);
    const topLevelProperties = Array.isArray(mlBomDocument?.properties) ? mlBomDocument.properties : [];
    const componentProperties = Array.isArray(mlBomDocument?.metadata?.component?.properties) ? mlBomDocument.metadata.component.properties : [];
    const properties = [...topLevelProperties, ...componentProperties];
    const propertyValue = (name) => properties.find((item) => item.name === name)?.value;
    check("CF-ONNX-TYPE-004", compactMlBomEvidence || (propertyValue("deepbom:model:onnxTypeProtoSchema") === typeProtoContract.schema
      && propertyValue("deepbom:model:onnxTypeProtoStatus") === typeProtoContract.status
      && propertyValue("deepbom:model:onnxTypeProtoSourceCommit") === typeProtoContract.source_commit
      && propertyValue("deepbom:model:onnxTypeProtoSourceSha256") === typeProtoContract.source_sha256
      && Number(propertyValue("deepbom:model:onnxTypeProtoDeclarations")) === Number(typeProtoContract.declaration_count || 0)
      && Number(propertyValue("deepbom:model:onnxTypeProtoInvalidDeclarations")) === Number(typeProtoContract.invalid_type_count || 0)
      && Number(propertyValue("deepbom:model:onnxTypeProtoNonDenseValues")) === Number(typeProtoContract.non_dense_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxTypeProtoSymbolicDimensions")) === Number(typeProtoContract.symbolic_dimension_count || 0)),
    "ML-BOM does not preserve the pinned ONNX TypeProto contract and decision counts.", ["/evidence/static_analysis/onnx_type_proto_contract", "/mlbom.cdx.json"]);
    check("CF-ONNX-SPARSE-004", compactMlBomEvidence || (propertyValue("deepbom:model:onnxSparseTensorSchema") === sparseContract.schema
      && propertyValue("deepbom:model:onnxSparseTensorStatus") === sparseContract.status
      && propertyValue("deepbom:model:onnxSparseTensorSourceCommit") === sparseContract.source_commit
      && propertyValue("deepbom:model:onnxSparseTensorSourceSha256") === sparseContract.source_sha256
      && Number(propertyValue("deepbom:model:onnxSparseTensorCount")) === Number(sparseContract.sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseTensorInvalidCount")) === Number(sparseContract.invalid_sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseTensorPartialCount")) === Number(sparseContract.partially_assessed_sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseIndexAssessedTensorCount")) === Number(sparseContract.index_content_assessed_sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseIndexFailedTensorCount")) === Number(sparseContract.index_content_failed_sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseIndexUnassessedTensorCount")) === Number(sparseContract.index_content_unassessed_sparse_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxSparseIndexViolationCount")) === Number(sparseContract.out_of_bounds_index_count || 0) + Number(sparseContract.duplicate_index_count || 0) + Number(sparseContract.unsorted_index_count || 0)
      && propertyValue("deepbom:model:onnxSparseExternalPayloadCoverage") === sparseContract.external_payload_coverage_status),
    "ML-BOM does not preserve the pinned ONNX SparseTensorProto contract, index coverage, and external-component status.", ["/evidence/static_analysis/onnx_sparse_tensor_contract", "/mlbom.cdx.json"]);
    check("CF-SHAPE-005", compactMlBomEvidence || (propertyValue("deepbom:model:onnxShapeInferenceSchema") === shapeInference.schema
      && propertyValue("deepbom:model:onnxShapeInferenceStatus") === shapeInference.status
      && propertyValue("deepbom:model:onnxShapeInferenceEvidenceClass") === shapeInference.evidence_class
      && propertyValue("deepbom:model:onnxShapeInferenceSourceCommit") === shapeInference.source_commit
      && Number(propertyValue("deepbom:model:onnxShapeRuleSupportedNodes")) === Number(shapeInference.rule_supported_nodes || 0)
      && Number(propertyValue("deepbom:model:onnxShapeRuleUnsupportedNodes")) === Number(shapeInference.rule_unsupported_nodes || 0)
      && Number(propertyValue("deepbom:model:onnxShapeRuleUnresolvedNodes")) === Number(shapeInference.rule_unresolved_node_count || 0)
      && propertyValue("deepbom:model:onnxOpsetImportContractStatus") === opsetImportContract.status
      && Number(propertyValue("deepbom:model:onnxOpsetImportCount")) === Number(opsetImportContract.import_count || 0)
      && Number(propertyValue("deepbom:model:onnxInvalidOpsetImports")) === Number(opsetImportContract.invalid_import_count || 0)
      && Number(propertyValue("deepbom:model:onnxDuplicateOpsetImportDomains")) === Number(opsetImportContract.duplicate_domain_count || 0)
      && propertyValue("deepbom:model:onnxShapeSchemaFormStatus") === shapeInference.schema_form_assessment_status
      && Number(propertyValue("deepbom:model:onnxShapeSchemaFormValidNodes")) === Number(shapeInference.schema_form_valid_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeSchemaFormInvalidNodes")) === Number(shapeInference.schema_form_invalid_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeSchemaFormUnresolvedNodes")) === Number(shapeInference.schema_form_unresolved_node_count || 0)
      && propertyValue("deepbom:model:onnxShapeScopeStatus") === shapeScope.status
      && Number(propertyValue("deepbom:model:onnxShapeReachableNestedGraphs")) === Number(shapeScope.reachable_nested_graph_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeReachableNestedGraphNodes")) === Number(shapeScope.reachable_nested_graph_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeFunctionDefaultGraphs")) === Number(shapeScope.function_default_graph_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeFunctionDefaultGraphNodes")) === Number(shapeScope.function_default_graph_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeReachableLocalFunctions")) === Number(shapeScope.reachable_local_function_definition_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeReachableLocalFunctionBodyNodes")) === Number(shapeScope.reachable_local_function_body_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeUnassessedReachableNodes")) === Number(shapeScope.unassessed_reachable_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeExecutedReachableScopes")) === Number(shapeScope.executed_reachable_scope_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeFullyAssessedReachableScopes")) === Number(shapeScope.fully_assessed_reachable_scope_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeReachableScopeUnresolvedOutputs")) === Number(shapeScope.reachable_scope_unresolved_output_count || 0)
      && propertyValue("deepbom:model:onnxExtendedShapeStatus") === extendedShape.status
      && Number(propertyValue("deepbom:model:onnxLocalFunctionCalls")) === Number(extendedShape.local_function_call_count || 0)
      && Number(propertyValue("deepbom:model:onnxLocalFunctionCallFailures")) === Number(extendedShape.local_function_call_fail_count || 0)
      && Number(propertyValue("deepbom:model:onnxControlFlowShapeNodes")) === Number(extendedShape.control_flow_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxControlFlowShapeFailures")) === Number(extendedShape.control_flow_fail_count || 0)
      && Number(propertyValue("deepbom:model:onnxLoopNodes")) === Number(extendedShape.loop_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxLoopExactExpansions")) === Number(extendedShape.loop_exact_expansion_count || 0)
      && Number(propertyValue("deepbom:model:onnxLoopExactIterations")) === Number(extendedShape.loop_exact_iteration_count || 0)
      && Number(propertyValue("deepbom:model:onnxLoopExactBodyNodeEvaluations")) === Number(extendedShape.loop_exact_body_node_evaluation_count || 0)
      && Number(propertyValue("deepbom:model:onnxLoopNonDenseStateVariables")) === Number(extendedShape.loop_non_dense_state_variable_count || 0)
      && Number(propertyValue("deepbom:model:onnxSequenceMapNodes")) === Number(extendedShape.sequence_map_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxSequenceMapPasses")) === Number(extendedShape.sequence_map_pass_count || 0)
      && Number(propertyValue("deepbom:model:onnxSequenceMapPartials")) === Number(extendedShape.sequence_map_partial_count || 0)
      && Number(propertyValue("deepbom:model:onnxSequenceMapFailures")) === Number(extendedShape.sequence_map_fail_count || 0)
      && Number(propertyValue("deepbom:model:onnxRecursiveScopeExecutions")) === Number(extendedShape.scope_execution_count || 0)
      && Number(propertyValue("deepbom:model:onnxRecursiveScopeDefinitions")) === Number(extendedShape.scope_definition_count || 0)
      && Number(propertyValue("deepbom:model:onnxRecursiveScopesFullyAssessed")) === Number(extendedShape.fully_assessed_scope_count || 0)
      && Number(propertyValue("deepbom:model:onnxRecursiveResidualUnassessedNodes")) === Number(extendedShape.residual_unassessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxRecursiveResidualUnresolvedOutputs")) === Number(extendedShape.residual_unresolved_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapeDeclarationConflicts")) === Number(shapeInference.declaration_conflict_count || 0)
      && Number(propertyValue("deepbom:model:onnxShapePropagatedStaticValueTensors")) === Number(shapeInference.propagated_static_value_tensor_count || 0)
      && Number(propertyValue("deepbom:model:onnxKnownNodeOutputs")) === Number(shapeInference.known_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxTensorNodeOutputs")) === Number(shapeInference.tensor_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxUnknownNodeOutputs")) === Number(shapeInference.unknown_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxNonDenseNodeOutputs")) === Number(shapeInference.non_dense_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxKnownNonDenseNodeOutputs")) === Number(shapeInference.known_non_dense_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxUnresolvedNonDenseNodeOutputs")) === Number(shapeInference.unresolved_non_dense_node_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxKnownValueNodeOutputs")) === Number(shapeInference.known_value_node_output_count || 0)
      && nullableClose(Number(propertyValue("deepbom:model:onnxNodeValueAssessmentRatio")), Number(shapeInference.node_value_assessment_ratio || 0))
      && Number(propertyValue("deepbom:model:onnxInferredNonDenseOutputs")) === Number(shapeInference.inferred_non_dense_outputs || 0)
      && Number(propertyValue("deepbom:model:onnxNonDenseValues")) === Number(shapeInference.non_dense_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxNodeOutputs")) === Number(shapeInference.node_output_count || 0)
      && propertyValue("deepbom:model:onnxContainerValueInferenceSchema") === containerInference.schema
      && propertyValue("deepbom:model:onnxContainerValueInferenceStatus") === containerInference.status
      && propertyValue("deepbom:model:onnxContainerValueSourceCommit") === containerInference.source_commit
      && Number(propertyValue("deepbom:model:onnxContainerValueAssessedNodes")) === Number(containerInference.assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxContainerValuePartialNodes")) === Number(containerInference.partially_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxContainerValueFailedNodes")) === Number(containerInference.failed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxExactSequenceLengthOutputs")) === Number(containerInference.exact_sequence_length_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxExactOptionalPresenceOutputs")) === Number(containerInference.exact_optional_presence_output_count || 0)
      && propertyValue("deepbom:model:onnxMlValueInferenceSchema") === mlValueInference.schema
      && propertyValue("deepbom:model:onnxMlValueInferenceStatus") === mlValueInference.status
      && propertyValue("deepbom:model:onnxMlValueSourceCommit") === mlValueInference.source_commit
      && propertyValue("deepbom:model:onnxMlValueRuntimeReferenceCommit") === mlValueInference.runtime_reference_commit
      && Number(propertyValue("deepbom:model:onnxMlValueAssessedNodes")) === Number(mlValueInference.assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlValuePartialNodes")) === Number(mlValueInference.partially_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlValueFailedNodes")) === Number(mlValueInference.failed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSequenceLengthOutputs")) === Number(mlValueInference.exact_sequence_length_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactClassKeys")) === Number(mlValueInference.exact_class_key_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlDuplicateClassKeys")) === Number(mlValueInference.duplicate_class_key_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlDuplicateClassKeyNodes")) === Number(mlValueInference.duplicate_class_key_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlBinarizerNodes")) === Number(mlValueInference.binarizer_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlBinarizerExactStaticNodes")) === Number(mlValueInference.binarizer_exact_static_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactBinarizerInputValues")) === Number(mlValueInference.exact_binarizer_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactBinarizerAboveThreshold")) === Number(mlValueInference.exact_binarizer_above_threshold_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactBinarizerAtOrBelowThreshold")) === Number(mlValueInference.exact_binarizer_at_or_below_threshold_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactBinarizerEqualThreshold")) === Number(mlValueInference.exact_binarizer_equal_threshold_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlBinarizerSchemaDefaultThresholdNodes")) === Number(mlValueInference.binarizer_schema_default_threshold_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlBinarizerNonfiniteThresholdNodes")) === Number(mlValueInference.binarizer_nonfinite_threshold_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlTensorNormalizationNodes")) === Number(mlValueInference.tensor_normalization_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlNormalizerNodes")) === Number(mlValueInference.normalizer_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlNormalizerStaticAssessedNodes")) === Number(mlValueInference.normalizer_static_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlNormalizerMaterializedOutputs")) === Number(mlValueInference.normalizer_output_materialized_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerInputValues")) === Number(mlValueInference.exact_normalizer_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerZeroDivisorRows")) === Number(mlValueInference.exact_normalizer_zero_divisor_row_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerNegativeMaxRows")) === Number(mlValueInference.exact_normalizer_negative_max_divisor_row_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerIntegerFloat32Changes")) === Number(mlValueInference.exact_normalizer_integer_float32_rounding_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerSignedOverflowValues")) === Number(mlValueInference.exact_normalizer_signed_overflow_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerNonfiniteOutputs")) === Number(mlValueInference.exact_normalizer_non_finite_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactNormalizerSignedZeroOutputs")) === Number(mlValueInference.exact_normalizer_signed_zero_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxStaticSignedZeroValues")) === exactStaticSignedZeroValues
      && Number(propertyValue("deepbom:model:onnxStaticSignedZeroTensors")) === staticSignedZeroTensors.length
      && Number(propertyValue("deepbom:model:onnxStaticCanonicalTextValues")) === exactStaticCanonicalTextValues
      && Number(propertyValue("deepbom:model:onnxStaticCanonicalTextTensors")) === staticCanonicalTextTensors.length
      && Number(propertyValue("deepbom:model:onnxMlNormalizerSchemaDefaultModeNodes")) === Number(mlValueInference.normalizer_schema_default_mode_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlTensorAffineScalerNodes")) === Number(mlValueInference.tensor_affine_scaler_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlScalerNodes")) === Number(mlValueInference.scaler_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlScalerStaticAssessedNodes")) === Number(mlValueInference.scaler_static_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlScalerMaterializedOutputs")) === Number(mlValueInference.scaler_output_materialized_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlScalerInvalidRuntimeContracts")) === Number(mlValueInference.scaler_invalid_runtime_contract_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerInputValues")) === Number(mlValueInference.exact_scaler_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerIntegerFloat32Changes")) === Number(mlValueInference.exact_scaler_integer_float32_rounding_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerNonfiniteParameters")) === Number(mlValueInference.exact_scaler_non_finite_parameter_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerNonfiniteOutputs")) === Number(mlValueInference.exact_scaler_non_finite_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerSignedZeroOutputs")) === Number(mlValueInference.exact_scaler_signed_zero_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactScalerZeroScales")) === Number(mlValueInference.exact_scaler_zero_scale_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlTensorImputationNodes")) === Number(mlValueInference.tensor_imputation_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlTensorEncoderNodes")) === Number(mlValueInference.tensor_encoder_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerNodes")) === Number(mlValueInference.imputer_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerStaticAssessedNodes")) === Number(mlValueInference.imputer_static_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerMaterializedOutputs")) === Number(mlValueInference.imputer_output_materialized_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerInvalidRuntimeContracts")) === Number(mlValueInference.imputer_invalid_runtime_contract_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerScalarFirstFallbackNodes")) === Number(mlValueInference.imputer_scalar_first_fallback_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlImputerPinnedCpuDtypeGapNodes")) === Number(mlValueInference.imputer_pinned_cpu_dtype_gap_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerInputValues")) === Number(mlValueInference.exact_imputer_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerReplacements")) === Number(mlValueInference.exact_imputer_replacement_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerNanReplacements")) === Number(mlValueInference.exact_imputer_nan_replacement_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerUnchanged")) === Number(mlValueInference.exact_imputer_unchanged_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerIgnoredValues")) === Number(mlValueInference.exact_imputer_ignored_imputed_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerNonfiniteValues")) === Number(mlValueInference.exact_imputer_non_finite_imputed_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerNonfiniteOutputs")) === Number(mlValueInference.exact_imputer_non_finite_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactImputerSignedZeroOutputs")) === Number(mlValueInference.exact_imputer_signed_zero_output_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotEncoderNodes")) === Number(mlValueInference.onehot_encoder_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotStaticAssessedNodes")) === Number(mlValueInference.onehot_static_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotMaterializedOutputs")) === Number(mlValueInference.onehot_output_materialized_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotInvalidContracts")) === Number(mlValueInference.onehot_invalid_contract_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotDuplicateVocabularyNodes")) === Number(mlValueInference.onehot_duplicate_vocabulary_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotUnknownAllZeroNodes")) === Number(mlValueInference.onehot_unknown_all_zero_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotGuaranteedRuntimeFailures")) === Number(mlValueInference.onehot_guaranteed_runtime_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotPinnedCpuDtypeGapNodes")) === Number(mlValueInference.onehot_pinned_cpu_dtype_gap_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotNoncanonicalZerosNodes")) === Number(mlValueInference.onehot_noncanonical_zeros_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlOneHotUnrepresentableCastNodes")) === Number(mlValueInference.onehot_unrepresentable_numeric_cast_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotInputValues")) === Number(mlValueInference.exact_onehot_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotMatchedInputs")) === Number(mlValueInference.exact_onehot_matched_input_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotUnknownInputs")) === Number(mlValueInference.exact_onehot_unknown_input_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotChangedCasts")) === Number(mlValueInference.exact_onehot_numeric_to_int64_changed_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotInvalidCasts")) === Number(mlValueInference.exact_onehot_numeric_to_int64_invalid_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotOutputOnes")) === Number(mlValueInference.exact_onehot_output_one_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotOutputZeros")) === Number(mlValueInference.exact_onehot_output_zero_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotDuplicateCategories")) === Number(mlValueInference.exact_onehot_duplicate_category_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactOneHotUnreachableColumns")) === Number(mlValueInference.exact_onehot_unreachable_duplicate_column_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlTensorLabelMappingNodes")) === Number(mlValueInference.tensor_label_mapping_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderNodes")) === Number(mlValueInference.label_encoder_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderStaticAssessedNodes")) === Number(mlValueInference.label_encoder_static_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderMaterializedOutputs")) === Number(mlValueInference.label_encoder_output_materialized_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderOnnxContractFailures")) === Number(mlValueInference.label_encoder_onnx_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderPinnedOrtContractFailures")) === Number(mlValueInference.label_encoder_pinned_ort_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderPinnedCpuDtypePairGaps")) === Number(mlValueInference.label_encoder_pinned_cpu_dtype_pair_gap_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderDuplicateSemanticConflicts")) === Number(mlValueInference.label_encoder_duplicate_semantic_conflict_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderNanSemanticConflicts")) === Number(mlValueInference.label_encoder_nan_semantic_conflict_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderDefaultPathNodes")) === Number(mlValueInference.label_encoder_default_path_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLabelEncoderOutputMismatchNodes")) === Number(mlValueInference.label_encoder_schema_runtime_output_mismatch_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderKeys")) === Number(mlValueInference.exact_label_encoder_key_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderInputValues")) === Number(mlValueInference.exact_label_encoder_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderMatches")) === Number(mlValueInference.exact_label_encoder_match_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderDefaults")) === Number(mlValueInference.exact_label_encoder_default_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderDuplicateKeyHits")) === Number(mlValueInference.exact_label_encoder_duplicate_key_hit_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLabelEncoderSchemaRuntimeMismatches")) === Number(mlValueInference.exact_label_encoder_schema_runtime_mismatch_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearModelNodes")) === Number(mlValueInference.linear_model_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearClassifierNodes")) === Number(mlValueInference.linear_classifier_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearRegressorNodes")) === Number(mlValueInference.linear_regressor_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearOnnxContractFailures")) === Number(mlValueInference.linear_onnx_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearPinnedOrtContractFailures")) === Number(mlValueInference.linear_pinned_ort_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearReferenceAssessedNodes")) === Number(mlValueInference.linear_reference_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearPinnedCpuDtypeGapNodes")) === Number(mlValueInference.linear_pinned_cpu_dtype_gap_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearPostTransformHazardNodes")) === Number(mlValueInference.linear_post_transform_hazard_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearUnusedCoefficientNodes")) === Number(mlValueInference.linear_unused_coefficient_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlLinearIgnoredInterceptNodes")) === Number(mlValueInference.linear_ignored_intercept_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearCoefficients")) === Number(mlValueInference.exact_linear_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearUsedCoefficients")) === Number(mlValueInference.exact_linear_used_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearUnusedCoefficients")) === Number(mlValueInference.exact_linear_unused_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearUnresolvedCoefficientUse")) === Number(mlValueInference.exact_linear_unresolved_coefficient_use_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearIgnoredIntercepts")) === Number(mlValueInference.exact_linear_ignored_intercept_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearReferenceInputValues")) === Number(mlValueInference.exact_linear_reference_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactLinearReferenceRawScores")) === Number(mlValueInference.exact_linear_reference_raw_score_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmModelNodes")) === Number(mlValueInference.svm_model_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmClassifierNodes")) === Number(mlValueInference.svm_classifier_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmRegressorNodes")) === Number(mlValueInference.svm_regressor_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmLinearModeNodes")) === Number(mlValueInference.svm_linear_mode_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmSvcModeNodes")) === Number(mlValueInference.svm_svc_mode_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmOnnxContractFailures")) === Number(mlValueInference.svm_onnx_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmPinnedOrtContractFailures")) === Number(mlValueInference.svm_pinned_ort_contract_failure_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmRegressorPinnedCpuDtypeGaps")) === Number(mlValueInference.svm_regressor_pinned_cpu_dtype_gap_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmScoreWidthMismatches")) === Number(mlValueInference.svm_schema_runtime_score_width_mismatch_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmIgnoredPostTransforms")) === Number(mlValueInference.svm_ignored_post_transform_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmIgnoredParameterNodes")) === Number(mlValueInference.svm_ignored_parameter_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmNonfiniteNodes")) === Number(mlValueInference.svm_non_finite_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlSvmReferenceAssessedNodes")) === Number(mlValueInference.svm_reference_assessed_node_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmVectors")) === Number(mlValueInference.exact_svm_vector_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmPairwiseClassifiers")) === Number(mlValueInference.exact_svm_pairwise_classifier_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmSupportValues")) === Number(mlValueInference.exact_svm_support_vector_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUsedSupportValues")) === Number(mlValueInference.exact_svm_used_support_vector_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnusedSupportValues")) === Number(mlValueInference.exact_svm_unused_support_vector_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnresolvedSupportUses")) === Number(mlValueInference.exact_svm_unresolved_support_vector_use_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmCoefficients")) === Number(mlValueInference.exact_svm_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUsedCoefficients")) === Number(mlValueInference.exact_svm_used_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnusedCoefficients")) === Number(mlValueInference.exact_svm_unused_coefficient_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnresolvedCoefficientUses")) === Number(mlValueInference.exact_svm_unresolved_coefficient_use_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmRhos")) === Number(mlValueInference.exact_svm_rho_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUsedRhos")) === Number(mlValueInference.exact_svm_used_rho_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnusedRhos")) === Number(mlValueInference.exact_svm_unused_rho_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmUnresolvedRhoUses")) === Number(mlValueInference.exact_svm_unresolved_rho_use_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmReferenceInputValues")) === Number(mlValueInference.exact_svm_reference_input_value_count || 0)
      && Number(propertyValue("deepbom:model:onnxMlExactSvmReferenceRawScores")) === Number(mlValueInference.exact_svm_reference_raw_score_count || 0)
      && onnxTreeMlBomConserves(propertyValue, mlValueInference)
      && onnxTfIdfMlBomConserves(propertyValue, tfidfInference)
      && shapeSources.every((source) => String(propertyValue("deepbom:model:onnxShapeInferenceSourceDocuments") || "").includes(`${source.role}:${source.sha256}`))
      && containerSources.every((source) => String(propertyValue("deepbom:model:onnxContainerValueSourceDocuments") || "").includes(`${source.role}:${source.sha256}`))
      && mlValueSources.every((source) => String(propertyValue("deepbom:model:onnxMlValueSourceDocuments") || "").includes(`${source.role}:${source.sha256}`))
      && mlValueRuntimeSources.every((source) => String(propertyValue("deepbom:model:onnxMlValueRuntimeReferenceDocuments") || "").includes(`${source.role}:${source.sha256}`))
      && (extendedShape.source_documents || []).every((source) => String(propertyValue("deepbom:model:onnxExtendedShapeSourceDocuments") || "").includes(`${source.role}:${source.sha256}`))), "ML-BOM does not preserve the ONNX shape-inference source, recursive-scope coverage, static-value, and declaration-conflict contract.", ["/evidence/static_analysis/onnx_shape_inference", "/mlbom.cdx.json"]);
    const usedExternalFiles = suppliedExternalFiles.filter((file) => file.used);
    const externalFileComponents = (mlBomDocument?.components || []).filter((component) => component?.type === "file"
      && (component.properties || []).some((item) => item.name === "deepbom:file:role" && item.value === "external_weights"));
    const subjectRef = mlBomDocument?.metadata?.component?.["bom-ref"];
    const subjectDependency = (mlBomDocument?.dependencies || []).find((dependency) => dependency.ref === subjectRef);
    const compactExternalDataSummary = compactMlBomEvidence || (Number(propertyValue("mlbom:model:onnxExternalDataTensorCount")) === externalRows.length
      && Number(propertyValue("mlbom:model:onnxExternalDataVerifiedTensorCount")) === verifiedExternalRows.length
      && Number(propertyValue("mlbom:model:onnxExternalDataIncompleteTensorCount")) === externalRows.length - verifiedExternalRows.length
      && propertyValue("mlbom:model:onnxExternalDataIntegrityStatus") === (externalRows.length === verifiedExternalRows.length ? "complete" : failedExternalRows.length ? "verification_failed" : "incomplete")
      && Number(propertyValue("mlbom:model:onnxExternalDataVerifiedPayloadBytes")) === verifiedExternalBytes
      && Number(propertyValue("mlbom:model:onnxExternalDataUsedFileCount")) === usedExternalFiles.length);
    check("CF-ONNX-EXTERNAL-004", compactExternalDataSummary
      && externalFileComponents.length === usedExternalFiles.length
      && usedExternalFiles.every((file) => externalFileComponents.some((component) => component.name === file.path
        && component.type === "file" && component.scope === "required"
        && (subjectDependency?.dependsOn || []).includes(component["bom-ref"])
        && (component.hashes || []).some((hash) => hash.alg === "SHA-256" && hash.content === file.sha256)
        && (!file.sha1 || (component.hashes || []).some((hash) => hash.alg === "SHA-1" && hash.content === file.sha1)))),
    "ONNX ML-BOM must preserve external-data coverage and one hash-bound required file component for every used sidecar.", ["/evidence/static_analysis/onnx_external_data", "/evidence/mlbom_cyclonedx"]);
    check("CF-FORMAT-005", !properties.some((item) => String(item.name || "").includes("staticStructuralWatchlist")), "ONNX ML-BOM contains the retired structural-watchlist property.", ["/evidence/mlbom_cyclonedx"]);
    const assessedMacRows = (staticAnalysis?.ops || []).filter((op) => op.macs_status === "assessed");
    const assessedMacTotal = assessedMacRows.reduce((sum, op) => sum + BigInt(op.macs_decimal), 0n);
    check("CF-FORMAT-006", (staticAnalysis?.ops || []).filter((op) => op.macs_status === "not_assessed").every((op) => op.macs === null), "ONNX MAC-unassessed ops must use null, not a numeric zero default.", ["/evidence/static_analysis/ops"]);
    check("CF-FORMAT-006E", assessedMacRows.every((op) => /^(?:0|[1-9][0-9]*)$/.test(String(op.macs_decimal || ""))
      && (BigInt(op.macs_decimal) <= BigInt(Number.MAX_SAFE_INTEGER) ? op.macs === Number(BigInt(op.macs_decimal)) : op.macs === null)), "ONNX assessed per-op MACs must preserve an exact decimal and expose a Number mirror only inside JavaScript's exact-integer range.", ["/evidence/static_analysis/ops"]);
    check("CF-FORMAT-006A", (staticAnalysis?.ops || []).every((op) => op.estimated_bytes_status === "assessed"
      ? Number.isFinite(op.estimated_bytes) && op.estimated_bytes >= 0 && Number(op.unassessed_payload_tensor_count || 0) === 0
      : op.estimated_bytes_status === "not_assessed" && op.estimated_bytes === null && Number(op.unassessed_payload_tensor_count || 0) > 0
        && Boolean(String(op.estimated_bytes_reason || "").trim())), "ONNX logical-byte assessment must preserve unknown payloads as null with an explicit reason instead of numeric zero.", ["/evidence/static_analysis/ops"]);
    const exactIntensity = (op) => {
      const denominator = BigInt(op.estimated_bytes);
      return exactNonnegativeRatio(BigInt(op.macs_decimal) * 2n, denominator);
    };
    check("CF-FORMAT-006C", (staticAnalysis?.ops || []).every((op) => op.intensity_status === "assessed"
      ? op.macs_status === "assessed" && op.estimated_bytes_status === "assessed" && op.estimated_bytes > 0
        && Number.isFinite(op.intensity_ops_per_byte) && op.intensity_ops_per_byte >= 0
        && nullableClose(op.intensity_ops_per_byte, exactIntensity(op))
      : op.intensity_ops_per_byte === null && (op.intensity_status === "not_applicable_zero_logical_bytes"
        ? op.macs_status === "assessed" && op.estimated_bytes_status === "assessed" && op.estimated_bytes === 0
        : op.intensity_status === "not_assessed_exact_ratio_outside_numeric_range"
          ? op.macs_status === "assessed" && op.estimated_bytes_status === "assessed" && op.estimated_bytes > 0
          : op.intensity_status === "not_assessed" && (op.macs_status !== "assessed" || op.estimated_bytes_status !== "assessed"))), "ONNX arithmetic intensity must be derived only from assessed exact MACs and complete positive logical-byte traffic, otherwise remain null with an explicit status.", ["/evidence/static_analysis/ops"]);
    check("CF-FORMAT-006B", (staticAnalysis?.ops || []).every((op) => op.row_working_set_status === "assessed"
      ? op.standard_domain === true && ["Conv", "QLinearConv", "ConvInteger"].includes(op.name)
        && Number.isFinite(op.row_working_set_bytes) && op.row_working_set_bytes > 0
      : op.row_working_set_bytes === null && ["not_applicable", "not_assessed"].includes(op.row_working_set_status)), "ONNX row working-set values must be positive assessed standard-domain Conv/QLinearConv/ConvInteger results or null with an explicit non-assessed status.", ["/evidence/static_analysis/ops"]);
    const l1Bytes = Number(staticAnalysis?.target_profile?.l1_data_bytes || 0);
    check("CF-FORMAT-006D", (staticAnalysis?.ops || []).every((op) => op.row_working_set_ratio_status === "assessed"
      ? op.row_working_set_status === "assessed" && l1Bytes > 0 && Number.isFinite(op.row_working_set_ratio)
        && nullableClose(op.row_working_set_ratio, Number(op.row_working_set_bytes) / l1Bytes)
      : op.row_working_set_ratio === null && (op.row_working_set_status === "assessed"
        ? l1Bytes <= 0 && op.row_working_set_ratio_status === "not_assessed_target_l1_unavailable"
        : op.row_working_set_ratio_status === op.row_working_set_status)), "ONNX L1 row-working-set ratio must conserve the assessed byte numerator and target L1 denominator or remain null with a matching status.", ["/evidence/static_analysis/ops", "/evidence/static_analysis/target_profile/l1_data_bytes"]);
    const customDomainOps = (staticAnalysis?.ops || []).filter((op) => op.domain !== "ai.onnx");
    check("CF-DOMAIN-001", customDomainOps.every((op) => op.standard_domain === false
      && op.macs_status === "not_assessed" && op.macs === null
      && op.quantized_compute_path === false && op.row_working_set_bytes === null
      && String(op.macs_reason || "").includes("is not an ai.onnx operator")), "Custom-domain operators must not inherit standard ONNX shape, MAC, quantized-compute, or row-working-set semantics from an op-name collision.", ["/evidence/static_analysis/ops"]);
    const live = staticAnalysis?.tensor_liveness || {};
    const liveNonDenseValues = live.non_dense_values || [];
    const liveResidualCount = Number(live.unassessed_tensor_count || 0) + Number(live.non_dense_value_count || 0);
    check("CF-LIVENESS-001", ["assessed", "partial", "not_assessed"].includes(live.status)
      && Number(live.assessed_tensor_count || 0) >= 0
      && Number(live.unassessed_tensor_count || 0) === (live.unassessed_tensors || []).length
      && Number(live.non_dense_value_count || 0) === liveNonDenseValues.length
      && liveNonDenseValues.every((row) => row.value_kind && !["tensor", "unresolved", "undefined"].includes(row.value_kind))
      && (live.status === "assessed" ? liveResidualCount === 0 && live.peak_bytes_status === "assessed" && Number.isFinite(live.peak_bytes)
        : live.status === "partial" ? Number(live.assessed_tensor_count || 0) > 0 && liveResidualCount > 0 && live.peak_bytes_status === "assessed_tensor_lower_bound" && Number.isFinite(live.peak_bytes)
          : Number(live.assessed_tensor_count || 0) === 0 && live.peak_bytes === null)
      && (!liveNonDenseValues.length || String(engineeringReport || "").includes("Non-dense runtime values excluded")), "ONNX liveness must distinguish complete dense-tensor peak, dense-tensor lower bound, non-dense values, and unassessed results without zero substitution.", ["/evidence/static_analysis/tensor_liveness", "/engineering_report.md"]);
    const assessedMacDecimal = String(staticAnalysis?.mac_assessment?.total_assessed_macs_decimal ?? "");
    const assessedOpsDecimal = String(staticAnalysis?.mac_assessment?.total_assessed_ops_decimal ?? "");
    const safeMacMirror = assessedMacTotal <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(assessedMacTotal) : null;
    const assessedOpsTotal = assessedMacTotal * 2n;
    const safeOpsMirror = assessedOpsTotal <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(assessedOpsTotal) : null;
    const completeMacLedger = Number(staticAnalysis?.mac_assessment?.not_assessed_compute_ops || 0) === 0;
    const topLevelMacTotalsValid = completeMacLedger
      ? staticAnalysis?.total_macs === safeMacMirror
        && String(staticAnalysis?.total_macs_decimal || "") === assessedMacTotal.toString()
        && staticAnalysis?.total_ops === safeOpsMirror
        && String(staticAnalysis?.total_ops_decimal || "") === assessedOpsTotal.toString()
      : staticAnalysis?.total_macs === null
        && staticAnalysis?.total_macs_decimal === null
        && staticAnalysis?.total_ops === null
        && staticAnalysis?.total_ops_decimal === null;
    check("CF-FORMAT-007", assessedMacDecimal === assessedMacTotal.toString()
      && assessedOpsDecimal === assessedOpsTotal.toString()
      && staticAnalysis?.mac_assessment?.total_assessed_macs === safeMacMirror
      && staticAnalysis?.mac_assessment?.total_assessed_ops === safeOpsMirror
      && topLevelMacTotalsValid
      && staticAnalysis?.mac_assessment?.safe_number_mirror_status === (safeMacMirror == null || safeOpsMirror == null ? "exact_decimal_only" : "safe_integer_mirrors_available"), "ONNX assessed MAC/op subtotals must equal the exact BigInt per-op sum, while incomplete top-level totals and unsafe Number mirrors remain null.", ["/evidence/static_analysis/total_macs", "/evidence/static_analysis/mac_assessment"]);
    check("CF-FORMAT-008", Boolean(staticAnalysis?.mac_assessment) && String(engineeringReport || "").includes("Assessed MAC total"), "ONNX MAC assessment coverage must be present in structured evidence and the engineering report.", ["/evidence/static_analysis/mac_assessment", "/engineering_report.md"]);
    check("CF-FORMAT-009", compactMlBomEvidence || properties.some((item) => item.name === "mlbom:model:macAssessmentStatus"), "ONNX ML-BOM must expose MAC assessment coverage semantics.", ["/evidence/mlbom_cyclonedx"]);
    const quantMacRatio = staticAnalysis?.quantization_status?.quantized_compute_mac_percent;
    const mlBomQuantRatio = properties.find((item) => item.name === "mlbom:model:quantizedComputeMacRatio")?.value;
    const mlBomQuantAssessment = properties.find((item) => item.name === "mlbom:model:quantizedComputeMacAssessment")?.value;
    check("CF-FORMAT-010", quantMacRatio != null || compactMlBomEvidence || (mlBomQuantRatio === "" && mlBomQuantAssessment === "not_assessed_mac_coverage_incomplete"), "ONNX ML-BOM must preserve an unassessed quantized-MAC ratio instead of serializing it as zero.", ["/evidence/static_analysis/quantization_status/quantized_compute_mac_percent", "/evidence/mlbom_cyclonedx"]);
    const ortAssessment = String(staticAnalysis?.ort_compatibility_assessment_status || "not_loaded");
    const ortCompatibility = staticAnalysis?.ort_compatibility_evidence || null;
    check("CF-ORT-001", ortAssessment === "complete"
      ? ortCompatibility?.schema === ANALYZER_METADATA.schemas.ortSourceCompatibility
        && staticAnalysis?.ort_compatibility_evidence_schema === ANALYZER_METADATA.schemas.ortSourceCompatibility
        && staticAnalysis?.ort_compatibility_evidence_access === "research"
      : ortAssessment === "not_loaded" && !ortCompatibility, "ORT source compatibility status, schema, access tier, and payload must agree.", ["/evidence/static_analysis/ort_compatibility_assessment_status", "/evidence/static_analysis/ort_compatibility_evidence"]);
    if (ortAssessment === "complete") {
      const floor = ortCompatibility.runtime_floor || {};
      const eps = ortCompatibility.execution_providers || [];
      const completeOrtFloor = ["assessed_onnx_and_model_local_domains", "assessed_onnx_model_local_and_source_backed_contrib_domains"].includes(floor.status);
      check("CF-ORT-002", floor.evidence_class === "DERIVED_NECESSARY_MINIMUM"
        ? staticAnalysis.runtime_compat?.derived_min_runtime_version === floor.minimum_ort_version
          && staticAnalysis.runtime_compat?.effective_min_runtime_version === (completeOrtFloor ? floor.minimum_ort_version : "")
        : !staticAnalysis.runtime_compat?.derived_min_runtime_version, "ONNX runtime compatibility fields do not preserve the necessary-floor versus complete-floor distinction.", ["/evidence/static_analysis/runtime_compat", "/evidence/static_analysis/ort_compatibility_evidence/runtime_floor"]);
      const nativeRegistrationEps = new Set(["qnn", "directml", "coreml", "nnapi", "xnnpack"]);
      check("CF-ORT-003", eps.length === 9 && eps.every((ep) => ep.assignment_evidence_class === "NOT_OBSERVED"
        && ep.support_evidence_class === (nativeRegistrationEps.has(ep.execution_provider)
          ? "SOURCE_REGISTRATION_CANDIDATE_WITH_UNRESOLVED_GET_CAPABILITY_PREDICATES"
          : "SOURCE_SCHEMA_KERNEL_VERSION_WITH_DEFINITE_ARTIFACT_EXCLUSIONS_ONLY")
        && String(ep.source_scope || "")
        && String(ep.evaluator_coverage || "")
        && ep.assessed_op_count === (staticAnalysis.ops || []).length
        && (ep.ops || []).length === (staticAnalysis.ops || []).length
        && ep.artifact_condition_count === (ep.ops || []).reduce((sum, row) => sum + Number(row.artifact_condition_count || 0), 0)
        && ep.artifact_condition_pass_count === (ep.ops || []).reduce((sum, row) => sum + Number(row.artifact_condition_pass_count || 0), 0)
        && ep.artifact_condition_fail_count === (ep.ops || []).reduce((sum, row) => sum + Number(row.artifact_condition_fail_count || 0), 0)
        && ep.artifact_condition_unresolved_count === (ep.ops || []).reduce((sum, row) => sum + Number(row.artifact_condition_unresolved_count || 0), 0)
        && ep.source_candidate_after_artifact_precheck_count === (ep.ops || []).filter((row) => row.source_candidate_after_artifact_precheck).length
        && (ep.ops || []).every((row) => row.resolved_schema_version == null
          || (Number.isSafeInteger(row.resolved_schema_version) && row.resolved_schema_version > 0 && row.resolved_schema_version <= row.imported_opset
            && /^[a-f0-9]{64}$/.test(row.schema_source_sha256 || "")))
        && /^[a-f0-9]{64}$/.test(ep.source_sha256 || "")), "ORT EP source assessments must cover every graph op without claiming runtime assignment.", ["/evidence/static_analysis/ort_compatibility_evidence/execution_providers"]);
      const nonPassingConditions = eps.flatMap((ep) => (ep.ops || []).flatMap((row) => (row.artifact_conditions || [])
        .filter((condition) => condition.status !== "PASS").map((condition) => ({ ep, row, condition }))))
        .sort((left, right) => (left.condition.status === "DEFINITE_FAIL" ? 0 : 1) - (right.condition.status === "DEFINITE_FAIL" ? 0 : 1)
          || left.row.op_index - right.row.op_index || left.ep.execution_provider.localeCompare(right.ep.execution_provider));
      check("CF-ORT-004", String(engineeringReport || "").includes("## Execution Provider Source Compatibility (SOURCE+ARTIFACT_PRECHECK/NOT_OBSERVED)")
        && String(engineeringReport || "").includes("DERIVED_NECESSARY_MINIMUM")
        && eps.every((ep) => String(engineeringReport || "").includes(ep.source_sha256) && String(engineeringReport || "").includes(ep.source_ref) && String(engineeringReport || "").includes(ep.source_scope) && String(engineeringReport || "").includes(ep.evaluator_coverage)
          && (ep.ops || []).filter((row) => row.schema_source_ref).every((row) => String(engineeringReport || "").includes(row.schema_source_ref) && String(engineeringReport || "").includes(row.schema_source_sha256)))
        && nonPassingConditions.slice(0, 32).every(({ ep, row, condition }) => String(engineeringReport || "").includes(ep.execution_provider)
          && String(engineeringReport || "").includes(`#${String(row.op_index).padStart(3, "0")} ${row.op_name}`)
          && String(engineeringReport || "").includes(condition.condition_id)
          && String(engineeringReport || "").includes(markdownCellForConformance(condition.expected))
          && String(engineeringReport || "").includes(markdownCellForConformance(condition.reason))), "Engineering report does not disclose the ORT floor boundary, artifact precheck issues, and complete pinned EP/schema source provenance.", ["/evidence/static_analysis/ort_compatibility_evidence", "/engineering_report.md"]);
      const floorSources = floor.source_documents || [];
      check("CF-ORT-005", floorSources.length >= 1
        && floorSources.every((source) => /^[a-f0-9]{64}$/.test(source.sha256 || "")
          && String(engineeringReport || "").includes(source.sha256)
          && String(engineeringReport || "").includes(source.source_ref))
        && JSON.stringify(floor.source_refs || []) === JSON.stringify(floorSources.map((source) => source.source_ref)), "ORT runtime floor must bind and render every source document by SHA-256.", ["/evidence/static_analysis/ort_compatibility_evidence/runtime_floor/source_documents", "/engineering_report.md"]);
      const portability = ortCompatibility.portability_frontier || null;
      check("CF-ORT-006", portability?.schema === ANALYZER_METADATA.schemas.ortEpPortabilityFrontier
        && staticAnalysis.ort_ep_portability_frontier?.schema === ANALYZER_METADATA.schemas.ortEpPortabilityFrontier
        && portability.op_count === (staticAnalysis.ops || []).length
        && portability.execution_provider_count === eps.length
        && portability.evidence_class === "DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS"
        && String(engineeringReport || "").includes("## ONNX EP Portability Frontier (DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS)"), "ONNX EP portability frontier is missing, unbound, or absent from the engineering report.", ["/evidence/static_analysis/ort_ep_portability_frontier", "/engineering_report.md"]);
      const exclusionCount = eps.reduce((sum, ep) => sum + Number(ep.artifact_precheck_definite_fail_op_count || 0), 0);
      const exclusionFindings = (findingsRegister?.findings || []).filter((finding) => finding.finding_id === "EA-ONX-0005");
      check("CF-ORT-007", exclusionCount > 0
        ? exclusionFindings.length === 1 && String(engineeringReport || "").includes("Pinned ORT EP source conditions exclude artifact nodes")
        : exclusionFindings.length === 0, "EA-ONX-0005 must exist exactly when pinned EP artifact conditions definitely exclude one or more node-provider candidates.", ["/evidence/static_analysis/ort_compatibility_evidence/execution_providers", "/evidence/findings_register/findings", "/engineering_report.md"]);
      const mlBomProperty = (name) => properties.find((item) => item.name === name)?.value;
      check("CF-ORT-008", compactMlBomEvidence || (mlBomProperty("deepbom:model:ortEpPortabilityFrontierSchema") === portability.schema
        && mlBomProperty("deepbom:model:ortEpPortabilityEvidenceClass") === portability.evidence_class
        && Number(mlBomProperty("deepbom:model:ortAllEpSourceMatchOps")) === portability.all_ep_source_match_op_count
        && Number(mlBomProperty("deepbom:model:ortAllEpArtifactPrecheckCandidateOps")) === portability.all_ep_artifact_precheck_candidate_op_count
        && String(mlBomProperty("deepbom:model:ortEpArtifactPrecheckBoundary") || "").includes("not support")), "ONNX ML-BOM must preserve the source-only and artifact-precheck candidate distinction and its non-assignment boundary.", ["/evidence/static_analysis/ort_ep_portability_frontier", "/evidence/mlbom_cyclonedx"]);
      const conditionInventory = ortCompatibility.source_condition_inventory || {};
      check("CF-ORT-009", conditionInventory.schema === "deepbom.ort_source_artifact_contract.v1"
        && conditionInventory.source_rule_count === 1167
        && conditionInventory.cpu_registration_variant_count === 611
        && conditionInventory.machine_condition_count === 540
        && conditionInventory.versioned_scalar_schema_default_binding_count === 80
        && conditionInventory.unresolved_source_fragment_count === 432
        && conditionInventory.informational_source_note_count === 425
        && (conditionInventory.execution_providers || []).length === 9
        && String(engineeringReport || "").includes("Source-condition extractor")
        && String(engineeringReport || "").includes("1,167 / 611")
        && String(engineeringReport || "").includes(conditionInventory.evidence_boundary), "Engineering Report must disclose the complete pinned source-condition extractor inventory and unresolved-fragment boundary.", ["/evidence/static_analysis/ort_compatibility_evidence/source_condition_inventory", "/engineering_report.md"]);
    }

    const weightIntegrity = staticAnalysis?.weight_integrity || {};
    const details = Array.isArray(weightIntegrity.zero_kernel_slice_details) ? weightIntegrity.zero_kernel_slice_details : [];
    const quantGridDetails = Array.isArray(weightIntegrity.quant_grid_details) ? weightIntegrity.quant_grid_details : [];
    const weightFinding = (findingsRegister?.findings || []).find((item) => item.finding_id === "EA-WGT-0001");
    const reportText = String(engineeringReport || "");
    const assessedWeights = weightIntegrity.status === "assessed";
    const initializerResults = weightIntegrity.tensor_results || [];
    const assessedInitializerResults = initializerResults.filter((row) => row.status === "assessed");
    const assessedLearnedParameterResults = assessedInitializerResults.filter((row) => row.constant_role === "learned_parameter");
    const resultLogicalElements = assessedInitializerResults.reduce((sum, row) => sum + Number(row.elements_scanned || 0), 0);
    const resultStoredValues = assessedInitializerResults.reduce((sum, row) => sum + Number(row.stored_weight_values_decoded || 0), 0);
    const resultImplicitZeros = assessedInitializerResults.reduce((sum, row) => sum + Number(row.implicit_zero_elements || 0), 0);
    const resultFiniteElements = assessedLearnedParameterResults.reduce((sum, row) => sum + Number(row.finite_elements || 0), 0);
    const resultNearZeroElements = assessedLearnedParameterResults.reduce((sum, row) => sum + Number(row.near_zero_elements || 0), 0);
    const learnedElements = assessedLearnedParameterResults.reduce((sum, row) => sum + Number(row.elements_scanned || 0), 0);
    const logicalInitializerInventory = (staticAnalysis?.tensors || []).filter((tensor) => tensor.role === "initializer" || tensor.constant_buffer);
    check("CF-WGT-000", initializerResults.length === Number(weightIntegrity.initializer_tensors_present || 0)
      && initializerResults.length === logicalInitializerInventory.length
      && assessedInitializerResults.length === Number(weightIntegrity.constant_tensors_scanned || 0)
      && assessedLearnedParameterResults.length === Number(weightIntegrity.weight_tensors_scanned || 0)
      && initializerResults.filter((row) => row.status !== "assessed").length === Number(weightIntegrity.initializer_tensors_unassessed || 0)
      && resultLogicalElements === Number(weightIntegrity.logical_elements_assessed || 0)
      && resultLogicalElements === Number(weightIntegrity.elements_scanned || 0)
      && learnedElements === Number(weightIntegrity.learned_parameter_elements_scanned || 0)
      && resultStoredValues === Number(weightIntegrity.stored_weight_values_decoded || 0)
      && resultImplicitZeros === Number(weightIntegrity.implicit_zero_elements || 0)
      && resultStoredValues + resultImplicitZeros === resultLogicalElements
      && Number(weightIntegrity.dense_initializer_tensors || 0) + Number(weightIntegrity.sparse_initializer_tensors || 0) === initializerResults.length
      && Number(weightIntegrity.initializer_elements_present || 0) === logicalInitializerInventory.reduce((sum, tensor) => sum + Number(tensor.initializer_elements || 0), 0)
      && Object.values(weightIntegrity.constant_role_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0) === assessedInitializerResults.length
      && nullableClose(weightIntegrity.mean_sparsity, resultFiniteElements > 0 ? resultNearZeroElements / resultFiniteElements : assessedLearnedParameterResults.length ? 0 : null),
    "ONNX dense+sparse initializer integrity inventory, decoded-storage count, implicit-zero count, or logical sparsity does not conserve.", ["/evidence/static_analysis/weight_integrity", "/evidence/static_analysis/tensors"]);
    check("CF-WGT-001", assessedWeights
      ? reportText.includes("Initializer-value integrity assessed") && !reportText.includes("Initializer-value integrity was not assessed")
      : reportText.includes("Initializer-value integrity was not assessable"), "Static audit conclusion does not match structured ONNX initializer-integrity status.", ["/evidence/static_analysis/weight_integrity/status", "/engineering_report.md"]);
    check("CF-WGT-002", assessedWeights
      ? reportText.includes("Initializer value decoding | Implemented for") && reportText.includes("Weight statistics | Assessed only for confirmed learned parameters")
      : reportText.includes("Initializer value decoding | Not assessable"), "Analysis Completeness does not match structured ONNX initializer-decoder coverage.", ["/evidence/static_analysis/weight_integrity", "/engineering_report.md"]);
    check("CF-WGT-002A", !assessedWeights || (reportText.includes(`${weightIntegrity.stored_weight_values_decoded} stored value(s) decoded`)
      && reportText.includes(`${weightIntegrity.implicit_zero_elements} sparse implicit zero(s)`)), "Engineering report does not expose ONNX stored-value versus sparse implicit-zero integrity coverage.", ["/evidence/static_analysis/weight_integrity", "/engineering_report.md"]);
    check("CF-WGT-003", details.every((item) => {
      const expected = Array.isArray(item.bias_value_sample)
        && item.bias_value_sample.some((value) => Number.isFinite(Number(value)) && Math.abs(Number(value)) >= 1e-8);
      return Boolean(item.bias_nonzero_for_flagged_channels) === expected;
    }), "ONNX zero-slice bias-nonzero flag is inconsistent with decoded bias samples.", ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details"]);
    check("CF-WGT-004", details.every((item) => {
      const consumer = (staticAnalysis?.ops || []).find((op) => op.index === item.consumer_op_index);
      return Boolean(consumer)
        && consumer.name === item.consumer_op_name
        && (item.consumer_ops || []).includes(`#${consumer.index} ${consumer.name}`);
    }), "ONNX zero-slice consumer evidence does not match the parsed operator table.", ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details", "/evidence/static_analysis/ops"]);
    check("CF-WGT-005", details.every((item) => {
      const tensor = (staticAnalysis?.tensors || []).find((candidate) => candidate.name === item.tensor_name);
      return Boolean(tensor) && sameArray(item.shape || item.tensor_shape, tensor.shape);
    }), "ONNX zero-slice tensor shape does not match the parsed tensor inventory.", ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details", "/evidence/static_analysis/tensors"]);
    const floatDetails = details.filter((item) => String(item.dtype || "").toUpperCase().startsWith("FLOAT"));
    check("CF-FORMAT-011", !floatDetails.length || !/quantized|quantization scale|zero-point metadata/i.test(`${weightFinding?.title || ""} ${weightFinding?.observation || ""}`), "FLOAT ONNX weight finding contains quantized-only wording.", ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details", "/evidence/findings_register/findings"]);
    check("CF-FINDING-003", Number(weightIntegrity.zero_kernel_slice_count || 0) === 0 || (Boolean(weightFinding)
      && (weightFinding.evidence_json_pointers || []).includes("/evidence/static_analysis/weight_integrity/zero_kernel_slice_details")
      && details.slice(0, 4).every((item) => String(weightFinding.observation || "").includes(item.tensor_name))), "EA-WGT-0001 must be synthesized from and point to the zero-kernel-slice evidence records.", ["/evidence/static_analysis/weight_integrity/zero_kernel_slice_details", "/evidence/findings_register/findings"]);
    check("CF-REPORT-001", assessedWeights
      ? reportText.includes("## Weight Integrity (OBSERVED)") && !reportText.includes("## Weight Integrity (NOT_ASSESSABLE)")
      : reportText.includes("## Weight Integrity (NOT_ASSESSABLE)"), "Engineering report weight-integrity heading does not match structured evidence status.", ["/evidence/static_analysis/weight_integrity/status", "/engineering_report.md"]);
    check("CF-WGT-006", compactMlBomEvidence || (propertyValue("deepbom:model:onnxWeightIntegrityStatus") === weightIntegrity.status
      && propertyValue("deepbom:model:onnxWeightIntegrityCoverage") === weightIntegrity.coverage_status
      && propertyValue("deepbom:model:onnxWeightIntegrityAssessedInitializers") === String(weightIntegrity.weight_tensors_scanned ?? "not_assessed")
      && propertyValue("deepbom:model:onnxWeightIntegrityLogicalElements") === String(weightIntegrity.logical_elements_assessed ?? "not_assessed")
      && propertyValue("deepbom:model:onnxWeightIntegrityStoredValuesDecoded") === String(weightIntegrity.stored_weight_values_decoded ?? "not_assessed")
      && propertyValue("deepbom:model:onnxWeightIntegritySparseImplicitZeros") === String(weightIntegrity.implicit_zero_elements ?? "not_assessed")),
    "ML-BOM does not preserve ONNX dense+sparse weight-integrity coverage arithmetic.", ["/evidence/static_analysis/weight_integrity", "/mlbom.cdx.json"]);
    check("CF-WGT-007", quantGridDetails.length === Number(weightIntegrity.quantized_constant_tensors_scanned || 0)
      && quantGridDetails.every((row) => Number(row.elements_scanned || 0) === Number(row.stored_values_decoded || 0) + Number(row.implicit_zero_elements || 0)
        && Number(row.unique_integer_levels || 0) <= Number(row.legal_integer_levels || 0)
        && nullableClose(row.grid_utilization, Number(row.unique_integer_levels || 0) / Number(row.legal_integer_levels || 1))
        && nullableClose(row.saturation_ratio, Number(row.endpoint_elements || 0) / Number(row.elements_scanned || 1)))
      && nullableClose(weightIntegrity.min_grid_utilization, quantGridDetails.length ? Math.min(...quantGridDetails.map((row) => Number(row.grid_utilization))) : null)
      && nullableClose(weightIntegrity.max_saturation_percent, quantGridDetails.length ? Math.max(...quantGridDetails.map((row) => Number(row.saturation_ratio))) : null)
      && (!quantGridDetails.length || (reportText.includes("### Quantized Kernel Grid Ledger")
        && quantGridDetails.every((row) => reportText.includes(row.tensor_name) && reportText.includes(row.formula)))),
    "ONNX quantized-kernel logical grid arithmetic or tensor-level Engineering Report ledger does not conserve.", ["/evidence/static_analysis/weight_integrity/quant_grid_details", "/engineering_report.md"]);

    const size = staticAnalysis?.size_breakdown || {};
    const initializerTensors = (staticAnalysis?.tensors || []).filter((tensor) => tensor.role === "initializer" || tensor.constant_buffer);
    const embeddedConstants = initializerTensors.filter((tensor) => Number(tensor.initializer_external_component_count || 0) === 0);
    const externalComponentCount = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_external_component_count || 0), 0);
    const verifiedExternalComponentCount = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_verified_external_component_count || 0), 0);
    const expectedFp16Bytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_projected_embedded_fp16_bytes || 0), 0);
    const expectedInt8Bytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_projected_embedded_int8_bytes || 0), 0);
    const expectedConstantBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_embedded_bytes || 0), 0);
    const expectedStoredElements = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_embedded_stored_elements || 0), 0);
    const expectedFloatBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_embedded_float_bytes || 0), 0);
    const expectedVerifiedExternalBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_verified_external_bytes || 0), 0);
    const expectedVerifiedExternalElements = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_verified_external_stored_elements || 0), 0);
    const expectedAvailableBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_available_bytes || 0), 0);
    const expectedAvailableElements = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_available_stored_elements || 0), 0);
    const expectedLogicalElements = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_elements || 0), 0);
    const expectedRawBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_raw_data_bytes || 0), 0);
    const expectedRawZeroBytes = initializerTensors.reduce((total, tensor) => total + Number(tensor.initializer_raw_zero_bytes || 0), 0);
    check("CF-SIZE-001", Number(size.theoretical_fp16_constant_bytes) === expectedFp16Bytes && Number(size.theoretical_int8_constant_bytes) === expectedInt8Bytes, "ONNX scalar-width size projections do not match initializer dtype and element evidence.", ["/evidence/static_analysis/size_breakdown", "/evidence/static_analysis/tensors"]);
    check("CF-SIZE-002", size.metrics?.theoretical_fp16_constant_bytes?.status === "assessed"
      && size.metrics?.theoretical_fp16_constant_bytes?.value === size.theoretical_fp16_constant_bytes
      && size.metrics?.theoretical_int8_constant_bytes?.status === "assessed"
      && size.metrics?.theoretical_int8_constant_bytes?.value === size.theoretical_int8_constant_bytes
      && size.metadata_bytes === null, "ONNX size metrics must distinguish assessed projections from unseparated metadata bytes.", ["/evidence/static_analysis/size_breakdown/metrics"]);
    const numberFormat = new Intl.NumberFormat("en-US");
    check("CF-SIZE-003", reportText.includes("Projected embedded initializer payload at FP16")
      && reportText.includes(`${numberFormat.format(expectedFp16Bytes)} B`)
      && reportText.includes("Projected embedded initializer payload at INT8")
      && reportText.includes(`${numberFormat.format(expectedInt8Bytes)} B`), "Engineering report does not render the deterministic ONNX initializer-width projections.", ["/evidence/static_analysis/size_breakdown", "/engineering_report.md"]);
    check("CF-SIZE-004", Number(size.constant_tensor_count) === initializerTensors.length
      && Number(size.embedded_constant_tensor_count) === embeddedConstants.length
      && Number(size.external_data_tensor_count) === externalComponentCount
      && Number(size.constant_bytes) === expectedConstantBytes
      && Number(size.stored_scalar_elements) === expectedStoredElements
      && Number(size.verified_external_payload_bytes) === expectedVerifiedExternalBytes
      && Number(size.verified_external_scalar_elements) === expectedVerifiedExternalElements
      && Number(size.available_initializer_bytes) === expectedAvailableBytes
      && Number(size.available_initializer_scalar_elements) === expectedAvailableElements
      && Number(size.logical_initializer_elements) === expectedLogicalElements
      && Number(size.float_constant_bytes) === expectedFloatBytes,
    "ONNX size breakdown does not conserve initializer declarations, embedded and verified-external payloads, available scalar elements, or FLOAT bytes.", ["/evidence/static_analysis/size_breakdown", "/evidence/static_analysis/tensors"]);
    const completeExternalCoverage = verifiedExternalComponentCount === externalComponentCount;
    const expectedZeroStatus = expectedAvailableBytes > 0 && completeExternalCoverage && expectedRawBytes === expectedAvailableBytes
      ? "assessed"
      : !completeExternalCoverage
        ? "not_assessed_external_data"
        : expectedAvailableBytes === 0
          ? "not_applicable_no_available_initializer_payload"
          : "not_assessed_typed_tensor_fields";
    const expectedZeroRatio = expectedZeroStatus === "assessed" ? expectedRawZeroBytes / expectedAvailableBytes : null;
    check("CF-SIZE-005", size.metrics?.zero_constant_byte_ratio?.status === expectedZeroStatus
      && size.metrics?.zero_constant_byte_ratio?.value === expectedZeroRatio
      && size.zero_constant_byte_ratio === expectedZeroRatio,
    "ONNX raw zero-byte ratio or assessment status does not match independently reconstructed available initializer coverage.", ["/evidence/static_analysis/size_breakdown/metrics/zero_constant_byte_ratio", "/evidence/static_analysis/tensors"]);
    check("CF-SIZE-006", reportText.includes(`${numberFormat.format(initializerTensors.length)} total / ${numberFormat.format(embeddedConstants.length)} fully embedded / ${numberFormat.format(externalComponentCount)} external TensorProto component`)
      && reportText.includes(`${numberFormat.format(expectedStoredElements)} stored values/indices element(s)`)
      && reportText.includes(`${numberFormat.format(expectedAvailableBytes)} B`)
      && reportText.includes(`${numberFormat.format(expectedAvailableElements)} stored values/indices element(s)`),
    "Engineering report does not distinguish ONNX initializer declarations, embedded payloads, and verified available payload coverage.", ["/evidence/static_analysis/size_breakdown", "/engineering_report.md"]);

}
