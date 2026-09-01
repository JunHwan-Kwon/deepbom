import { markdownTable } from "./report-utils.js";
import { buildDecisionCoverageLedger, decisionCoverageMarkdown } from "./decision-coverage.js";
import { buildQuantResearchCoverage } from "./quant-research-applicability.js";

export { buildDecisionCoverageLedger, decisionCoverageMarkdown } from "./decision-coverage.js";

const SCHEMA = "deepbom.metric_coverage_manifest.v1.53";
const MAX_FIELD_DISCOVERY_VISITS = 2_000_000;
const SPECIAL_FIELD_EXPORT_ROUTES = new Map([
  ["roofline_csv", "supplemental_sources.roofline_csv"],
  ["core_isolation_csv", "supplemental_sources.core_isolation_roofline_csv"],
  ["stage_mermaid", "supplemental_sources.stage_graph_mermaid"],
  ["findings", "evidence.findings_register"],
  ["recommendations", "evidence.findings_register"],
  ["suspects", "evidence.findings_register"],
]);
const FIELD_METRIC_PREFIX_OVERRIDES = [
  ["/ops/[]/cache_payload/", "memory.cache_payload"],
];

function intrinsicCostReportFields(prefix, { identity = false, residuals = false, observation = false } = {}) {
  const fields = [
    "status", "operator_count", "mac_compute_operator_count", "assessed_nominal_mac_operator_count",
    "unassessed_nominal_mac_operator_count", "complete_nominal_macs", "complete_nominal_macs_decimal",
    "assessed_nominal_macs", "assessed_nominal_macs_decimal", "assessed_operator_io_count",
    "unassessed_operator_io_count", "complete_operator_io_payload_bytes", "complete_operator_io_payload_bytes_decimal",
    "assessed_operator_io_payload_bytes", "assessed_operator_io_payload_bytes_decimal",
  ];
  if (identity) fields.push("schema", "evidence_class", "source_release", "source_commit",
    "source_documents/[]/role", "source_documents/[]/source_ref", "source_documents/[]/sha256");
  if (residuals) fields.push(...["mac_residuals", "payload_residuals"].flatMap((ledger) => ["node_index", "op_name", "reason"].map((field) => `${ledger}/[]/${field}`)), "method", "interpretation_boundary");
  if (observation) fields.push("observation_count");
  return fields.map((field) => `${prefix}/${field}`);
}

// Decision-critical summaries must be read by the Engineering Report whenever
// the analyzer emits them. Large proof arrays remain raw-evidence-only.
const REQUIRED_REPORT_FIELD_PATTERNS = new Map([
  ["executorch.serialized_contract", [
    "/executorch_container", "/version", "/subgraphs", "/operator_count", "/tensor_count",
    "/executorch_program/identifier", "/executorch_program/source/repository",
    "/executorch_program/source/commit", "/executorch_program/kernel_instruction_count",
    "/executorch_program/delegate_instruction_count",
    "/executorch_program/external_tensor_data/status", "/executorch_program/graph_boundary",
    "/executorch_flat_tensor/identifier", "/executorch_flat_tensor/source/repository",
    "/executorch_flat_tensor/source/commit",
    "/mac_assessment/status", "/mac_assessment/complete", "/mac_assessment/detail",
    "/tensor_liveness/status", "/tensor_liveness/planned_non_const_memory_decimal",
    "/tensor_liveness/per_device/[]/device", "/tensor_liveness/per_device/[]/bytes_decimal",
    "/size_breakdown/status", "/size_breakdown/appended_segment_bytes_decimal",
    "/weight_integrity/status", "/weight_integrity/assessed_tensors", "/weight_integrity/detail",
    "/runtime_compat/runtime_version_basis", "/runtime_compat/detail",
  ]],
  ["performance.audit_timing", [
    "/static_audit_timing/evidence_class",
    "/static_audit_timing/wall_ms", "/static_audit_timing/core_static_analysis_ms",
    "/static_audit_timing/comparison_target_count",
  ]],
  ["artifact.identity", [
    "/filename", "/file_size", "/format", "/model_sha256",
    "/target_profile/id", "/target_profile/label", "/target_profile/profile_sha256",
    "/target_profile/architecture", "/target_profile/l1_data_bytes", "/target_profile/l2_bytes",
    "/target_profile/cache_assumption", "/target_profile/cache_source_url",
    "/target_profile/hardware_spec", "/target_profile/performance_model_evidence_class",
    "/target_profile/performance_model_assumption",
    "/target_profile/simd_width_bits", "/target_profile/channel_alignment_multiple",
    "/target_profile/int8_lanes", "/target_profile/fp16_lanes", "/target_profile/fp32_lanes",
    "/target_profile/effective_peak_gops", "/target_profile/effective_memory_bandwidth_gbps",
    "/target_profile/weight_packing_bandwidth_gbps", "/target_profile/ridge_point_ops_per_byte",
    "/target_profile/chain_break_overhead_us_low", "/target_profile/chain_break_overhead_us_high",
    "/target_profile/xnnpack_kernel_family", "/target_profile/compute_bound_intensity",
    "/target_profile/memory_bound_intensity", "/target_profile/dot_product", "/target_profile/sve2",
    "/cpu_cost_target_binding/schema", "/cpu_cost_target_binding/profile_id",
    "/cpu_cost_target_binding/profile_sha256", "/cpu_cost_target_binding/binding_source",
    "/cpu_cost_target_binding/host_observed", "/cpu_cost_target_binding/source_input",
    "/accelerator_bindings/[]/schema", "/accelerator_bindings/[]/profile_id",
    "/accelerator_bindings/[]/provider", "/accelerator_bindings/[]/backend",
    "/accelerator_bindings/[]/device_class", "/accelerator_bindings/[]/binding_source",
    "/accelerator_bindings/[]/evidence_stage", "/accelerator_bindings/[]/artifact_sha256",
    "/accelerator_bindings/[]/source_rulepack_sha256", "/accelerator_bindings/[]/selected_build_sha256",
    "/accelerator_bindings/[]/compiled_plan_sha256", "/accelerator_bindings/[]/runtime_trace_sha256",
    "/accelerator_bindings/[]/binding_sha256", "/accelerator_bindings/[]/interpretation_boundary",
  ]],
  ["graph.inventory", [
    "/operator_count", "/tensor_count", "/subgraphs",
    "/tensors/[]/static_values_negative_zero_count",
    "/tensors/[]/static_values_negative_zero_indices/[]",
    "/tensors/[]/static_values_canonical_text_complete",
    "/tensors/[]/static_values_canonical_texts/[]",
  ]],
  ["architecture.blocks", [
    "/block_inventory/schema", "/block_inventory/status", "/block_inventory/evidence_class",
    "/block_inventory/stage_count", "/block_inventory/block_count",
    "/block_inventory/semantic_block_count", "/block_inventory/unnamed_block_count",
    "/block_inventory/method",
    "/block_inventory/interpretation_boundary",
    "/block_inventory/blocks/[]/block_id", "/block_inventory/blocks/[]/stage_index",
    "/block_inventory/blocks/[]/block_type", "/block_inventory/blocks/[]/display_name",
    "/block_inventory/blocks/[]/extraction/method", "/block_inventory/blocks/[]/extraction/confidence",
    "/block_inventory/blocks/[]/op_indices/[]",
    "/block_inventory/blocks/[]/spatial/input_h", "/block_inventory/blocks/[]/spatial/input_w",
    "/block_inventory/blocks/[]/spatial/output_h", "/block_inventory/blocks/[]/spatial/output_w",
    "/block_inventory/blocks/[]/channels/input", "/block_inventory/blocks/[]/channels/expand",
    "/block_inventory/blocks/[]/channels/output",
    "/block_inventory/blocks/[]/aggregates/mac_percent",
    "/block_inventory/blocks/[]/aggregates/modeled_time_ms",
    "/block_inventory/blocks/[]/aggregates/time_evidence_class",
    "/block_inventory/blocks/[]/aggregates/l1_max_ratio",
    "/block_inventory/blocks/[]/aggregates/logical_traffic_bytes",
    "/block_inventory/blocks/[]/aggregates/predicted_break_count",
  ]],
  ["memory.cache_payload", [
    "/ops/[]/cache_payload/schema", "/ops/[]/cache_payload/status",
    "/ops/[]/cache_payload/evidence_class", "/ops/[]/cache_payload/input_strip_bytes",
    "/ops/[]/cache_payload/output_row_bytes", "/ops/[]/cache_payload/logical_row_payload_bytes",
    "/ops/[]/cache_payload/serialized_kernel_bytes", "/ops/[]/cache_payload/serialized_bias_bytes",
    "/ops/[]/cache_payload/input_width", "/ops/[]/cache_payload/input_channels",
    "/ops/[]/cache_payload/output_width", "/ops/[]/cache_payload/output_channels",
    "/ops/[]/cache_payload/kernel_height", "/ops/[]/cache_payload/kernel_width",
    "/ops/[]/cache_payload/effective_kernel_height", "/ops/[]/cache_payload/input_dtype",
    "/ops/[]/cache_payload/output_dtype", "/ops/[]/cache_payload/method",
    "/ops/[]/cache_payload/interpretation_boundary",
  ]],
  ["contract.io", [
    "/input_tensor_indices/[]", "/output_tensor_indices/[]",
    "/inputs/[]/name", "/inputs/[]/shape/[]", "/inputs/[]/dtype",
    "/inputs/[]/shape_signature/[]", "/outputs/[]/name", "/outputs/[]/shape/[]",
    "/outputs/[]/dtype", "/outputs/[]/shape_signature/[]",
    "/input_contracts/[]/schema", "/input_contracts/[]/tensor_index",
    "/input_contracts/[]/name", "/input_contracts/[]/shape/[]",
    "/input_contracts/[]/dtype", "/input_contracts/[]/is_quantized",
    "/input_contracts/[]/expected_range_low", "/input_contracts/[]/expected_range_high",
    "/input_contracts/[]/range_note", "/input_contracts/[]/tensor_numerical_contract_status",
    "/input_contracts/[]/source_data_to_tensor_preprocessing_status",
    "/input_contracts/[]/layout", "/input_contracts/[]/layout_status",
    "/input_contracts/[]/layout_evidence_class", "/input_contracts/[]/layout_source_op_index",
    "/input_contracts/[]/layout_source_op_name", "/input_contracts/[]/layout_reason",
    "/input_contracts/[]/channel_axis", "/input_contracts/[]/channels",
    "/input_contracts/[]/risks/[]",
  ]],
  ["artifact.metadata", [
    "/metadata_presence/format", "/metadata_presence/schema", "/metadata_presence/status",
    "/metadata_presence/graph_input_count", "/metadata_presence/graph_output_count",
    "/metadata_presence/has_model_metadata", "/metadata_presence/metadata_entries",
    "/metadata_presence/has_description", "/metadata_presence/description",
    "/metadata_presence/documented_preprocessing", "/metadata_presence/preprocessing_contract_status",
    "/metadata_presence/output_semantics_documented", "/metadata_presence/output_label_file_count",
    "/metadata_presence/metadata_property_count", "/metadata_presence/metadata_text_bytes",
    "/metadata_presence/model_doc_string", "/metadata_presence/graph_doc_string",
    "/metadata_presence/producer_name", "/metadata_presence/producer_version",
    "/metadata_presence/model_domain", "/metadata_presence/model_version",
    "/metadata_presence/has_signature_defs", "/metadata_presence/signature_count",
    "/metadata_presence/signature_keys", "/metadata_presence/model_metadata_entry_count",
    "/metadata_presence/metadata_schema_identifier", "/metadata_presence/metadata_min_parser_version",
    "/metadata_presence/metadata_model_name", "/metadata_presence/metadata_model_description",
    "/metadata_presence/metadata_model_version", "/metadata_presence/metadata_author",
    "/metadata_presence/metadata_license", "/metadata_presence/subgraph_metadata_count",
    "/metadata_presence/input_tensor_metadata_count", "/metadata_presence/output_tensor_metadata_count",
    "/metadata_presence/described_input_tensor_count", "/metadata_presence/described_output_tensor_count",
    "/metadata_presence/input_process_unit_count", "/metadata_presence/recognized_input_process_unit_count",
    "/metadata_presence/invalid_input_process_unit_count", "/metadata_presence/unrecognized_input_process_unit_count",
    "/metadata_presence/normalization_unit_count", "/metadata_presence/output_associated_file_count",
    "/metadata_presence/verified_output_associated_file_count", "/metadata_presence/missing_output_associated_file_count",
    "/metadata_presence/verified_output_label_file_count", "/metadata_presence/missing_output_label_file_count",
    "/metadata_presence/invalid_output_label_file_count", "/metadata_presence/verified_output0_label_file_count",
    "/metadata_presence/payload_verified_file_count", "/metadata_presence/payload_invalid_file_count",
    "/metadata_presence/payload_unsupported_file_count", "/metadata_presence/label_cardinality_match_count",
    "/metadata_presence/label_cardinality_mismatch_count", "/metadata_presence/label_cardinality_ambiguous_count",
    "/metadata_presence/label_cardinality_unresolved_count", "/metadata_presence/associated_file_archive_status",
    "/metadata_presence/associated_file_archive_detail", "/metadata_presence/packed_associated_file_count",
    "/metadata_presence/detail",
  ]],
  ["onnx.external_data", [
    "/onnx_external_data/schema", "/onnx_external_data/status", "/onnx_external_data/evidence_class",
    "/onnx_external_data/tensor_count", "/onnx_external_data/entry_count",
    "/onnx_external_data/malformed_reference_count", "/onnx_external_data/unsafe_location_count",
    "/onnx_external_data/missing_location_count", "/onnx_external_data/duplicate_key_count",
    "/onnx_external_data/invalid_range_count", "/onnx_external_data/declared_payload_bytes",
    "/onnx_external_data/supplied_payload_count", "/onnx_external_data/verified_payload_count",
    "/onnx_external_data/invalid_checksum_count", "/onnx_external_data/embedded_payload_conflict_count",
    "/onnx_external_data/data_location_mismatch_count", "/onnx_external_data/payload_verification_failed_count",
    "/onnx_external_data/range_out_of_bounds_count", "/onnx_external_data/payload_size_mismatch_count",
    "/onnx_external_data/checksum_mismatch_count", "/onnx_external_data/verified_payload_bytes",
    "/onnx_external_data/supplied_file_count", "/onnx_external_data/supplied_file_bytes",
    "/onnx_external_data/used_file_count", "/onnx_external_data/unused_file_count",
    "/onnx_external_data/detail",
  ]],
  ["runtime.artifact_requirements", [
    "/runtime_compat/derived_min_runtime_version", "/runtime_compat/effective_min_runtime_version",
    "/runtime_compat/max_op_version", "/runtime_compat/runtime_version_basis",
    "/runtime_compat/min_runtime_version", "/runtime_compat/unmapped_versioned_ops",
    "/runtime_compat/operator_code_count", "/runtime_compat/builtin_operator_code_count",
    "/runtime_compat/mapped_operator_code_count", "/runtime_compat/custom_operator_code_count",
    "/runtime_compat/model_local_function_domains", "/runtime_compat/runtime_floor_status",
    "/runtime_compat/runtime_floor_evidence_class", "/runtime_compat/unresolved_runtime_floor_domains",
  ]],
  ["artifact.size", [
    "/size_breakdown/file_size", "/size_breakdown/constant_bytes",
    "/size_breakdown/constant_tensor_count", "/size_breakdown/dense_initializer_count",
    "/size_breakdown/sparse_initializer_count", "/size_breakdown/embedded_constant_tensor_count",
    "/size_breakdown/external_data_tensor_count", "/size_breakdown/stored_scalar_elements",
    "/size_breakdown/verified_external_payload_bytes", "/size_breakdown/verified_external_scalar_elements",
    "/size_breakdown/available_initializer_bytes", "/size_breakdown/available_initializer_scalar_elements",
    "/size_breakdown/logical_initializer_elements",
    "/size_breakdown/available_unique_constant_bytes", "/size_breakdown/available_duplicate_constant_bytes",
    "/size_breakdown/float_constant_bytes", "/size_breakdown/metadata_bytes",
    "/size_breakdown/structure_overhead_bytes", "/size_breakdown/zero_constant_byte_ratio",
    "/size_breakdown/metrics/zero_constant_byte_ratio/status",
    "/size_breakdown/detail",
  ]],
  ["artifact.byte_integrity", [
    "/artifact_byte_integrity/schema", "/artifact_byte_integrity/status",
    "/artifact_byte_integrity/file_size",
    "/artifact_byte_integrity/flatbuffer_referenced_bytes",
    "/artifact_byte_integrity/flatbuffer_referenced_end",
    "/artifact_byte_integrity/flatbuffer_internal_alignment_or_unreferenced_bytes",
    "/artifact_byte_integrity/metadata_archive_status",
    "/artifact_byte_integrity/metadata_archive_start",
    "/artifact_byte_integrity/metadata_archive_bytes",
    "/artifact_byte_integrity/unowned_trailing_bytes",
    "/artifact_byte_integrity/partial_buffer_overlap_count",
    "/artifact_byte_integrity/classified_bytes",
    "/artifact_byte_integrity/conservation_status",
    "/artifact_byte_integrity/issues", "/artifact_byte_integrity/method",
  ]],
  ["tflite.sparse_storage", [
    "/tflite_sparse_storage_contract/schema", "/tflite_sparse_storage_contract/status",
    "/tflite_sparse_storage_contract/evidence_class", "/tflite_sparse_storage_contract/sparse_tensor_count",
    "/tflite_sparse_storage_contract/serialized_value_tensor_count", "/tflite_sparse_storage_contract/fully_decoded_tensor_count",
    "/tflite_sparse_storage_contract/partial_tensor_count", "/tflite_sparse_storage_contract/logical_element_count",
    "/tflite_sparse_storage_contract/stored_element_count", "/tflite_sparse_storage_contract/implicit_zero_element_count",
    "/tflite_sparse_storage_contract/serialized_value_bytes", "/tflite_sparse_storage_contract/source_commit",
    "/tflite_sparse_storage_contract/schema_source_sha256", "/tflite_sparse_storage_contract/converter_source_sha256",
    "/tflite_sparse_storage_contract/method", "/tflite_sparse_storage_contract/interpretation_boundary",
  ]],
  ["tflite.subgraph_inventory", [
    "/tflite_subgraph_inventory/schema", "/tflite_subgraph_inventory/status",
    "/tflite_subgraph_inventory/evidence_class", "/tflite_subgraph_inventory/subgraph_count",
    "/tflite_subgraph_inventory/parsed_subgraph_count", "/tflite_subgraph_inventory/primary_subgraph_index",
    "/tflite_subgraph_inventory/primary_operator_count", "/tflite_subgraph_inventory/primary_tensor_count",
    "/tflite_subgraph_inventory/serialized_operator_count", "/tflite_subgraph_inventory/serialized_tensor_count",
    "/tflite_subgraph_inventory/nested_operator_count", "/tflite_subgraph_inventory/nested_tensor_count",
    "/tflite_subgraph_inventory/control_flow_reference_count", "/tflite_subgraph_inventory/signature_entrypoint_count",
    "/tflite_subgraph_inventory/reachable_subgraph_count", "/tflite_subgraph_inventory/unreachable_subgraph_indices/[]",
    "/tflite_subgraph_inventory/control_flow_contract_count", "/tflite_subgraph_inventory/assessed_control_flow_contract_count",
    "/tflite_subgraph_inventory/partial_control_flow_contract_count",
    "/tflite_subgraph_inventory/rows/[]/subgraph_index",
    "/tflite_subgraph_inventory/rows/[]/name",
    "/tflite_subgraph_inventory/rows/[]/tensor_count",
    "/tflite_subgraph_inventory/rows/[]/input_tensor_indices/[]",
    "/tflite_subgraph_inventory/rows/[]/output_tensor_indices/[]",
    "/tflite_subgraph_inventory/rows/[]/operator_count",
    "/tflite_subgraph_inventory/rows/[]/constant_tensor_count",
    "/tflite_subgraph_inventory/rows/[]/quantized_tensor_count",
    "/tflite_subgraph_inventory/rows/[]/per_axis_tensor_count",
    "/tflite_subgraph_inventory/rows/[]/sparse_tensor_count",
    "/tflite_subgraph_inventory/rows/[]/control_flow_reference_count",
    "/tflite_subgraph_inventory/rows/[]/incoming_reference_count",
    "/tflite_subgraph_inventory/rows/[]/reachable_from_entrypoint",
    "/tflite_subgraph_inventory/rows/[]/invocation_semantics",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/schema",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/status",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/evidence_class",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/invocation_basis",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/mac_compute_operator_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_nominal_mac_operator_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/modeled_scenario_mac_operator_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/unassessed_mac_operator_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/complete_nominal_macs",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/complete_nominal_macs_decimal",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_nominal_macs",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_nominal_macs_decimal",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/modeled_scenario_macs",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/modeled_scenario_macs_decimal",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/logical_tensor_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_logical_tensor_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_tensor_payload_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/unassessed_tensor_payload_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/logical_operator_io_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_logical_operator_io_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/assessed_operator_io_tensor_slot_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/unassessed_operator_io_tensor_slot_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/graph_input_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/graph_output_payload_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/logical_constant_reference_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/physical_unique_constant_bytes",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/physical_unique_constant_buffer_count",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/method",
    "/tflite_subgraph_inventory/rows/[]/intrinsic_cost/interpretation_boundary",
    "/tflite_subgraph_inventory/rows/[]/operator_histogram/[]/name",
    "/tflite_subgraph_inventory/rows/[]/operator_histogram/[]/count",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/source_subgraph_index",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/source_op_index",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/source_op_name",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/status",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/source_input_count",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/source_output_count",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/target_subgraph_indices/[]",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/condition_contract_status",
    "/tflite_subgraph_inventory/control_flow_contracts/[]/method",
    "/tflite_subgraph_inventory/control_flow_sources/[]/role",
    "/tflite_subgraph_inventory/control_flow_sources/[]/path",
    "/tflite_subgraph_inventory/control_flow_sources/[]/sha256",
    "/tflite_subgraph_inventory/nominal_mac_sources/[]/role",
    "/tflite_subgraph_inventory/nominal_mac_sources/[]/path",
    "/tflite_subgraph_inventory/nominal_mac_sources/[]/sha256",
    "/tflite_subgraph_inventory/schema_source_sha256", "/tflite_subgraph_inventory/method",
    "/tflite_subgraph_inventory/execution_count_boundary",
  ]],
  ["tflite.subgraph_deep_analysis", [
    "/tflite_subgraph_deep_analysis/schema", "/tflite_subgraph_deep_analysis/status",
    "/tflite_subgraph_deep_analysis/evidence_class", "/tflite_subgraph_deep_analysis/assessed_subgraph_count",
    "/tflite_subgraph_deep_analysis/subgraph_count", "/tflite_subgraph_deep_analysis/primary_subgraph_index",
    "/tflite_subgraph_deep_analysis/rows/[]/subgraph_index", "/tflite_subgraph_deep_analysis/rows/[]/name",
    "/tflite_subgraph_deep_analysis/rows/[]/status", "/tflite_subgraph_deep_analysis/rows/[]/evidence_class",
    "/tflite_subgraph_deep_analysis/rows/[]/reachable_from_entrypoint", "/tflite_subgraph_deep_analysis/rows/[]/invocation_semantics",
    "/tflite_subgraph_deep_analysis/rows/[]/operator_count", "/tflite_subgraph_deep_analysis/rows/[]/tensor_count",
    "/tflite_subgraph_deep_analysis/rows/[]/total_macs",
    "/tflite_subgraph_deep_analysis/rows/[]/quantized_tensor_count", "/tflite_subgraph_deep_analysis/rows/[]/per_axis_tensor_count",
    "/tflite_subgraph_deep_analysis/rows/[]/quantization_status", "/tflite_subgraph_deep_analysis/rows/[]/delegate",
    "/tflite_subgraph_deep_analysis/rows/[]/predicted_partition_boundaries", "/tflite_subgraph_deep_analysis/rows/[]/tensor_liveness",
    "/tflite_subgraph_deep_analysis/rows/[]/tensor_arena_plan", "/tflite_subgraph_deep_analysis/rows/[]/movement_analysis",
    "/tflite_subgraph_deep_analysis/rows/[]/weight_integrity",
    "/tflite_subgraph_deep_analysis/rows/[]/advanced_numerical_storage",
    "/tflite_subgraph_deep_analysis/method", "/tflite_subgraph_deep_analysis/execution_count_boundary",
  ]],
  ["compute.macs", [
    "/total_macs", "/mac_assessment/status", "/mac_assessment/compute_ops",
    "/mac_assessment/assessed_compute_ops", "/mac_assessment/not_assessed_compute_ops",
    "/mac_assessment/total_assessed_macs", "/mac_assessment/total_assessed_macs_decimal",
    "/mac_assessment/total_assessed_ops", "/mac_assessment/total_assessed_ops_decimal",
    "/mac_assessment/safe_number_mirror_status",
  ]],
  ["cost.dynamic_shape", [
    "/dynamic_shape_cost_contract/schema", "/dynamic_shape_cost_contract/status",
    "/dynamic_shape_cost_contract/evidence_class", "/dynamic_shape_cost_contract/format",
    "/dynamic_shape_cost_contract/dynamic_tensor_count", "/dynamic_shape_cost_contract/symbol_count",
    "/dynamic_shape_cost_contract/symbols/[]/symbol_id", "/dynamic_shape_cost_contract/symbols/[]/source",
    "/dynamic_shape_cost_contract/symbols/[]/declared_name", "/dynamic_shape_cost_contract/symbols/[]/lower_bound",
    "/dynamic_shape_cost_contract/symbols/[]/upper_bound", "/dynamic_shape_cost_contract/symbols/[]/upper_bound_expression",
    "/dynamic_shape_cost_contract/symbols/[]/bounds_status",
    "/dynamic_shape_cost_contract/tensor_formula_count", "/dynamic_shape_cost_contract/dynamic_compute_op_count",
    "/dynamic_shape_cost_contract/op_formula_count", "/dynamic_shape_cost_contract/unresolved_dynamic_compute_op_count",
    "/dynamic_shape_cost_contract/total_macs_unresolved_op_count",
    "/dynamic_shape_cost_contract/total_macs_formula_status",
    "/dynamic_shape_cost_contract/total_macs_formula/expression",
    "/dynamic_shape_cost_contract/liveness/status",
    "/dynamic_shape_cost_contract/liveness/candidate_program_point_count",
    "/dynamic_shape_cost_contract/liveness/exact_candidate_program_point_count",
    "/dynamic_shape_cost_contract/liveness/unresolved_candidate_program_point_count",
    "/dynamic_shape_cost_contract/liveness/distinct_exact_formula_count",
    "/dynamic_shape_cost_contract/liveness/peak_selection_status",
    "/dynamic_shape_cost_contract/liveness/peak_live_payload_formula/expression",
    "/dynamic_shape_cost_contract/arena_projection_status",
    "/dynamic_shape_cost_contract/dimension_bounds_status",
    "/dynamic_shape_cost_contract/method", "/dynamic_shape_cost_contract/interpretation_boundary",
  ]],
  ["performance.static_posture", [
    "/insights/score", "/insights/score_evidence_class", "/insights/score_method",
    "/insights/label", "/insights/tone", "/insights/bound_compute", "/insights/bound_mixed",
    "/insights/chain_breaks", "/insights/effective_chain_breaks", "/insights/copy_like_op_count",
    "/insights/max_l1_ratio", "/insights/misaligned_ops", "/insights/per_channel_ratio",
    "/insights/quant_ratio", "/insights/quant_risk_ops", "/insights/suspect_total",
    "/insights/score_breakdown/base", "/insights/score_breakdown/graph_runtime_pressure_penalty",
    "/insights/score_breakdown/dynamic_non_batch_input_penalty", "/insights/score_breakdown/copy_like_op_signal_points",
    "/insights/score_breakdown/fallback_byte_signal_points", "/insights/score_breakdown/memory_posture_signal_points",
    "/insights/score_breakdown/predicted_boundary_signal_points", "/insights/score_breakdown/suspect_op_signal_points",
    "/insights/score_breakdown/final_score", "/insights/score_breakdown/l1_watch_penalty",
    "/insights/score_breakdown/quantization_coverage_penalty",
    "/insights/score_breakdown/exact_zero_kernel_signal_points",
    "/insights/score_breakdown/quantization_risk_penalty",
    "/insights/dynamic_non_batch_input_count",
  ]],
  ["memory.liveness", [
    "/tensor_liveness/status", "/tensor_liveness/peak_bytes",
    "/tensor_liveness/assessed_tensor_count", "/tensor_liveness/unassessed_tensor_count",
    "/tensor_liveness/unknown_activation_tensors", "/tensor_liveness/evidence_class",
    "/tensor_liveness/peak_bytes_status",
    "/tensor_liveness/non_dense_value_count", "/tensor_liveness/non_dense_values/[]/tensor_index",
    "/tensor_liveness/non_dense_values/[]/tensor_name", "/tensor_liveness/non_dense_values/[]/value_kind",
    "/tensor_liveness/non_dense_values/[]/reason",
  ]],
  ["weights.integrity", [
    "/weight_integrity/status", "/weight_integrity/coverage_status",
    "/weight_integrity/weight_tensors_scanned", "/weight_integrity/elements_scanned",
    "/weight_integrity/logical_elements_assessed", "/weight_integrity/stored_weight_values_decoded",
    "/weight_integrity/implicit_zero_elements", "/weight_integrity/dense_initializer_tensors",
    "/weight_integrity/sparse_initializer_tensors",
    "/weight_integrity/initializer_tensors_unassessed", "/weight_integrity/nan_tensors",
    "/weight_integrity/inf_tensors", "/weight_integrity/output_channels_evaluated",
    "/weight_integrity/zero_kernel_slice_count", "/weight_integrity/min_grid_utilization",
    "/weight_integrity/max_saturation_percent", "/weight_integrity/evidence_class",
    "/weight_integrity/initializer_tensors_present",
    "/weight_integrity/initializer_elements_present", "/weight_integrity/initializer_value_decoding",
    "/weight_integrity/large_magnitude_tensors", "/weight_integrity/high_sparsity_tensors",
    "/weight_integrity/quant_grid_details/[]/tensor_name", "/weight_integrity/quant_grid_details/[]/storage_kind",
    "/weight_integrity/quant_grid_details/[]/shape/[]", "/weight_integrity/quant_grid_details/[]/elements_scanned",
    "/weight_integrity/quant_grid_details/[]/stored_values_decoded", "/weight_integrity/quant_grid_details/[]/implicit_zero_elements",
    "/weight_integrity/quant_grid_details/[]/unique_integer_levels", "/weight_integrity/quant_grid_details/[]/legal_integer_levels",
    "/weight_integrity/quant_grid_details/[]/grid_utilization", "/weight_integrity/quant_grid_details/[]/endpoint_elements",
    "/weight_integrity/quant_grid_details/[]/saturation_ratio", "/weight_integrity/quant_grid_details/[]/low_utilization_review",
    "/weight_integrity/quant_grid_details/[]/saturation_review", "/weight_integrity/quant_grid_details/[]/formula",
    "/weight_integrity/detail",
  ]],
  ["quantization.contracts", [
    "/quantization_status/classification", "/quantization_status/quantized_tensor_percent",
    "/quantization_status/quantized_compute_mac_percent", "/quantization_status/quantized_compute_ops",
    "/quantization_status/compute_ops", "/quantization_status/mac_assessed_compute_ops",
    "/quantization_status/mac_coverage_complete", "/quantization_status/quantize_ops",
    "/quantization_status/dequantize_ops", "/quantization_status/activation_quantize_ops",
    "/quantization_status/activation_dequantize_ops", "/quantization_status/activation_8bit_float_boundary_ops",
    "/quantization_status/integer_requantization_ops", "/quantization_status/constant_precision_conversion_ops",
    "/quantization_status/float16_constant_expansion_ops", "/quantization_status/int8_tensors",
    "/quantization_status/uint8_tensors", "/quantization_status/float16_tensors", "/quantization_status/float_tensors",
    "/quantization_status/dense_tensor_count", "/quantization_status/non_dense_value_count",
    "/quantization_status/full_integer", "/quantization_status/summary",
  ]],
  ["tflite.delegation", [
    "/delegated_mac_percent", "/fallback_byte_percent",
    "/predicted_partition_boundaries/status", "/predicted_partition_boundaries/edge_count",
    "/predicted_partition_boundaries/assessed_payload_edge_count",
    "/predicted_partition_boundaries/unassessed_payload_edge_count",
    "/predicted_partition_boundaries/payload_coverage_status",
    "/predicted_partition_boundaries/assessed_edge_payload_bytes",
    "/predicted_partition_boundaries/summed_edge_payload_bytes",
    "/predicted_partition_boundaries/assessed_unique_tensor_payload_bytes",
    "/predicted_partition_boundaries/unique_tensor_payload_bytes",
    "/predicted_partition_boundaries/interpretation_boundary",
  ]],
  ["tflite.xnnpack_candidates", [
    "/xnnpack_selector_assessment_status", "/xnnpack_selector_evidence_schema",
    "/xnnpack_selector_evidence_access", "/xnnpack_selector_evidence_provenance/schema",
    "/xnnpack_selector_evidence_provenance/method_version",
    "/xnnpack_selector_evidence_provenance/target_profile_id",
    "/xnnpack_selector_evidence_provenance/target_profile_sha256",
    "/xnnpack_selector_evidence_provenance/xnnpack_source_commit",
    "/xnnpack_selector_evidence_provenance/gemm_config_sha256",
    "/xnnpack_selector_evidence_provenance/dwconv_config_sha256",
    "/xnnpack_selector_evidence_provenance/candidate_op_count",
  ]],
  ["tflite.alternate_delegate_candidates", [
    "/tflite_delegate_compatibility_evidence/schema",
    "/tflite_delegate_compatibility_evidence/assessment_status",
    "/tflite_delegate_compatibility_evidence/evidence_class",
    "/tflite_delegate_compatibility_evidence/rulepack_sha256",
    "/tflite_delegate_compatibility_evidence/tensorflow_source_commit",
    "/tflite_delegate_compatibility_evidence/graph_op_count",
    "/tflite_delegate_compatibility_evidence/profiles/[]/source_candidate_after_artifact_precheck_count",
    "/tflite_delegate_compatibility_evidence/profiles/[]/definite_exclusion_count",
    "/tflite_delegate_compatibility_evidence/build_requirements/[]/binding_status",
  ]],
  ["tflite.arena", [
    "/tensor_arena_plan/schema", "/tensor_arena_plan/status",
    "/tensor_arena_plan/non_persistent_arena_bytes", "/tensor_arena_plan/persistent_arena_bytes",
    "/tensor_arena_plan/combined_arena_bytes", "/tensor_arena_plan/planned_tensor_count",
    "/tensor_arena_plan/root_allocation_count", "/tensor_arena_plan/non_persistent_allocation_count",
    "/tensor_arena_plan/persistent_allocation_count", "/tensor_arena_plan/shared_tensor_count",
    "/tensor_arena_plan/preserve_all_tensors", "/tensor_arena_plan/deterministic_tie_break",
    "/tensor_arena_plan/unassessed_tensor_count", "/tensor_arena_plan/calculation_issue_count",
  ]],
  ["tflite.movement_packing", [
    "/movement_analysis/status", "/movement_analysis/total_movement_bytes",
    "/movement_analysis/assessed_movement_bytes", "/movement_analysis/movement_op_count",
    "/movement_analysis/xnn_break_movement_bytes", "/movement_analysis/assessed_xnn_break_movement_bytes",
    "/movement_analysis/assessed_output_tensor_count", "/movement_analysis/unassessed_output_tensor_count",
    "/movement_analysis/calculation_issue_count",
  ]],
  ["onnx.domains", [
    "/onnx_domain_analysis/schema", "/onnx_domain_analysis/status",
    "/onnx_domain_analysis/standard_node_count", "/onnx_domain_analysis/external_custom_node_count",
    "/onnx_domain_analysis/ort_contrib_node_count", "/onnx_domain_analysis/scope",
  ]],
  ["onnx.shape_inference", [
    "/onnx_shape_inference/schema", "/onnx_shape_inference/status", "/onnx_shape_inference/evidence_class",
    "/onnx_shape_inference/engine", "/onnx_shape_inference/attempted_nodes", "/onnx_shape_inference/node_output_count",
    "/onnx_shape_inference/source_release", "/onnx_shape_inference/source_commit",
    "/onnx_shape_inference/source_documents/[]/role", "/onnx_shape_inference/source_documents/[]/source_ref",
    "/onnx_shape_inference/source_documents/[]/sha256",
    "/onnx_shape_inference/opset_import_contract/status", "/onnx_shape_inference/opset_import_contract/valid_import_count",
    "/onnx_shape_inference/opset_import_contract/invalid_import_count", "/onnx_shape_inference/opset_import_contract/effective_domain_count",
    "/onnx_shape_inference/opset_import_contract/duplicate_domain_count", "/onnx_shape_inference/opset_import_contract/duplicate_identical_domain_count",
    "/onnx_shape_inference/opset_import_contract/duplicate_version_variant_domain_count", "/onnx_shape_inference/opset_import_contract/resolution_rule",
    "/onnx_shape_inference/tensor_node_output_count", "/onnx_shape_inference/known_node_output_count",
    "/onnx_shape_inference/unknown_node_output_count", "/onnx_shape_inference/non_dense_node_output_count",
    "/onnx_shape_inference/shape_contract_known_node_output_count", "/onnx_shape_inference/shape_contract_unknown_node_output_count",
    "/onnx_shape_inference/invalid_node_output_count", "/onnx_shape_inference/conditionally_invalid_node_output_count",
    "/onnx_shape_inference/conditional_invalid_variant_count", "/onnx_shape_inference/conditional_unassessed_variant_count",
    "/onnx_shape_inference/unresolved_nonconflict_shape_contract_node_output_count",
    "/onnx_shape_inference/symbolic_shape_contract_node_output_count", "/onnx_shape_inference/conditional_shape_contract_node_output_count",
    "/onnx_shape_inference/partial_conditional_shape_contract_node_output_count",
    "/onnx_shape_inference/propagated_symbolic_shape_value_tensor_count",
    "/onnx_shape_inference/non_dense_node_output_names/[]", "/onnx_shape_inference/known_non_dense_node_output_count",
    "/onnx_shape_inference/known_non_dense_node_output_names/[]", "/onnx_shape_inference/unresolved_non_dense_node_output_count",
    "/onnx_shape_inference/known_value_node_output_count", "/onnx_shape_inference/node_value_assessment_ratio",
    "/onnx_shape_inference/inferred_non_dense_outputs",
    "/onnx_shape_inference/rule_supported_nodes", "/onnx_shape_inference/rule_unsupported_nodes",
    "/onnx_shape_inference/rule_unsupported_op_histogram/[]/name", "/onnx_shape_inference/rule_unsupported_op_histogram/[]/count",
    "/onnx_shape_inference/rule_unresolved_node_count", "/onnx_shape_inference/rule_unresolved_node_indices/[]",
    "/onnx_shape_inference/rule_unresolved_nodes/[]/node_index", "/onnx_shape_inference/rule_unresolved_nodes/[]/op_name",
    "/onnx_shape_inference/rule_unresolved_nodes/[]/reason", "/onnx_shape_inference/declaration_conflict_count",
    "/onnx_shape_inference/semantic_contract_conflict_count",
    "/onnx_shape_inference/schema_form_assessment_status",
    "/onnx_shape_inference/schema_form_valid_node_count", "/onnx_shape_inference/schema_form_invalid_node_count",
    "/onnx_shape_inference/schema_form_unresolved_node_count",
    "/onnx_shape_inference/shape_scope/status",
    "/onnx_shape_inference/shape_scope/reachable_scope_count",
    "/onnx_shape_inference/shape_scope/reachable_nested_graph_count", "/onnx_shape_inference/shape_scope/reachable_nested_graph_node_count",
    "/onnx_shape_inference/shape_scope/function_default_graph_count", "/onnx_shape_inference/shape_scope/function_default_graph_node_count",
    "/onnx_shape_inference/shape_scope/local_function_definition_count",
    "/onnx_shape_inference/shape_scope/reachable_local_function_definition_count", "/onnx_shape_inference/shape_scope/reachable_local_function_body_node_count",
    "/onnx_shape_inference/shape_scope/unassessed_reachable_node_count", "/onnx_shape_inference/shape_scope/executed_reachable_scope_count",
    "/onnx_shape_inference/shape_scope/fully_assessed_reachable_scope_count", "/onnx_shape_inference/shape_scope/reachable_scope_unresolved_output_count",
    "/onnx_shape_inference/extended_scope_inference/schema", "/onnx_shape_inference/extended_scope_inference/evidence_class",
    "/onnx_shape_inference/extended_scope_inference/status",
    "/onnx_shape_inference/extended_scope_inference/source_release", "/onnx_shape_inference/extended_scope_inference/source_commit",
    "/onnx_shape_inference/extended_scope_inference/source_documents/[]/role", "/onnx_shape_inference/extended_scope_inference/source_documents/[]/source_ref",
    "/onnx_shape_inference/extended_scope_inference/source_documents/[]/sha256",
    "/onnx_shape_inference/extended_scope_inference/local_function_call_count",
    "/onnx_shape_inference/extended_scope_inference/local_function_call_pass_count",
    "/onnx_shape_inference/extended_scope_inference/local_function_call_fail_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_node_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_pass_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_partial_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_fail_count",
    "/onnx_shape_inference/extended_scope_inference/loop_node_count",
    "/onnx_shape_inference/extended_scope_inference/loop_exact_expansion_count",
    "/onnx_shape_inference/extended_scope_inference/loop_exact_iteration_count",
    "/onnx_shape_inference/extended_scope_inference/loop_exact_body_node_evaluation_count",
    "/onnx_shape_inference/extended_scope_inference/loop_non_dense_state_variable_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/scope",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/node_index",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/op_name",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/imported_opset",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/status",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/body_node_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/state_variable_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/scan_output_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/state_value_kinds/[]",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/non_dense_state_variable_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_expansion_status",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_body_node_evaluation_count",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/iteration",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/state_index",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/value_kind",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/dtype",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/shape/[]",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/sequence_length",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_iteration_state_contracts/[]/states/[]/sequence_element_types/[]",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_final_state_contracts/[]/state_index",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_final_state_contracts/[]/output_name",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_nested_failure_rows/[]/scope",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/exact_nested_failure_rows/[]/reason_codes/[]",
    "/onnx_shape_inference/extended_scope_inference/control_flow_rows/[]/reason_codes/[]",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_node_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_pass_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_partial_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_fail_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/scope",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/node_index",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/imported_opset",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/status",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/exact_input_sequence_length",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/element_expansion_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/element_node_evaluation_count",
    "/onnx_shape_inference/extended_scope_inference/sequence_map_rows/[]/reason_codes/[]",
    "/onnx_shape_inference/extended_scope_inference/scope_execution_count",
    "/onnx_shape_inference/extended_scope_inference/scope_definition_count",
    "/onnx_shape_inference/extended_scope_inference/fully_assessed_scope_count",
    "/onnx_shape_inference/extended_scope_inference/residual_unassessed_node_count",
    "/onnx_shape_inference/extended_scope_inference/residual_unresolved_output_count",
    "/onnx_shape_inference/extended_scope_inference/intrinsic_cost_variant_count",
    "/onnx_shape_inference/extended_scope_inference/intrinsic_cost_variant_overflow_count",
    "/onnx_shape_inference/extended_scope_inference/intrinsic_cost_unassessed_execution_count",
    ...intrinsicCostReportFields("/onnx_shape_inference/extended_scope_inference/main_graph_intrinsic_cost", { identity: true, residuals: true }),
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/scope",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/scope_class",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/status",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/node_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/execution_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/assessed_node_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/unassessed_node_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/unresolved_output_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/intrinsic_cost_variant_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/intrinsic_cost_variant_overflow_count",
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/intrinsic_cost_unassessed_execution_count",
    ...intrinsicCostReportFields("/onnx_shape_inference/extended_scope_inference/scope_rows/[]/intrinsic_cost_variants/[]", { observation: true }),
    "/onnx_shape_inference/extended_scope_inference/scope_rows/[]/reason_codes/[]",
    "/onnx_shape_inference/extended_scope_inference/method",
    "/onnx_shape_inference/extended_scope_inference/interpretation_boundary",
    "/onnx_shape_inference/tfidf_vectorizer_inference/schema",
    "/onnx_shape_inference/tfidf_vectorizer_inference/status",
    "/onnx_shape_inference/tfidf_vectorizer_inference/evidence_class",
    "/onnx_shape_inference/tfidf_vectorizer_inference/source_release",
    "/onnx_shape_inference/tfidf_vectorizer_inference/source_commit",
    "/onnx_shape_inference/tfidf_vectorizer_inference/runtime_reference_release",
    "/onnx_shape_inference/tfidf_vectorizer_inference/runtime_reference_commit",
    "/onnx_shape_inference/tfidf_vectorizer_inference/source_documents/[]/role",
    "/onnx_shape_inference/tfidf_vectorizer_inference/source_documents/[]/source_ref",
    "/onnx_shape_inference/tfidf_vectorizer_inference/source_documents/[]/sha256",
    "/onnx_shape_inference/tfidf_vectorizer_inference/assessed_node_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/passed_node_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/partially_assessed_node_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/failed_node_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_static_node_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_ngram_definition_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_active_ngram_definition_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_match_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_output_value_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_duplicate_output_coordinate_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_weight_coordinate_value_disagreement_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/exact_ort_reference_divergent_output_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/scope",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/node_index",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/op_name",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/imported_opset",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/status",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/input_name",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/input_dtype",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/input_shape/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/output_name",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/output_dtype",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_output_shape/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_output_width",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/mode",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/minimum_gram_length",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/maximum_gram_length",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/maximum_skip_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/pool_kind",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_pool_item_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_ngram_definition_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_active_ngram_definition_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_unused_pool_prefix_item_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_duplicate_active_ngram_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_duplicate_inactive_ngram_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_ngram_index_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_duplicate_output_coordinate_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_unaddressed_output_coordinate_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/weights_present",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_weight_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_weight_coordinate_disagreement_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_weight_coordinate_value_disagreement_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/static_input_status",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_static_input_value_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/static_execution_status",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_static_work_units",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_match_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_nonzero_frequency_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_output_value_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_nonzero_output_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_negative_zero_output_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_output_values/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_output_negative_zero_indices/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_ort_reference_divergent_output_count",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/exact_ort_reference_divergent_output_indices/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/reason_codes/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/rows/[]/risk_codes/[]",
    "/onnx_shape_inference/tfidf_vectorizer_inference/method",
    "/onnx_shape_inference/tfidf_vectorizer_inference/interpretation_boundary",
    "/onnx_shape_inference/container_value_inference/schema", "/onnx_shape_inference/container_value_inference/status",
    "/onnx_shape_inference/container_value_inference/evidence_class", "/onnx_shape_inference/container_value_inference/source_release",
    "/onnx_shape_inference/container_value_inference/source_commit",
    "/onnx_shape_inference/container_value_inference/source_documents/[]/role",
    "/onnx_shape_inference/container_value_inference/source_documents/[]/source_ref",
    "/onnx_shape_inference/container_value_inference/source_documents/[]/sha256",
    "/onnx_shape_inference/container_value_inference/assessed_node_count",
    "/onnx_shape_inference/container_value_inference/passed_node_count",
    "/onnx_shape_inference/container_value_inference/partially_assessed_node_count",
    "/onnx_shape_inference/container_value_inference/failed_node_count",
    "/onnx_shape_inference/container_value_inference/exact_sequence_length_output_count",
    "/onnx_shape_inference/container_value_inference/exact_optional_presence_output_count",
    "/onnx_shape_inference/container_value_inference/rows/[]/scope",
    "/onnx_shape_inference/container_value_inference/rows/[]/node_index",
    "/onnx_shape_inference/container_value_inference/rows/[]/op_name",
    "/onnx_shape_inference/container_value_inference/rows/[]/status",
    "/onnx_shape_inference/container_value_inference/rows/[]/canonical_output_types/[]",
    "/onnx_shape_inference/container_value_inference/rows/[]/sequence_lengths/[]",
    "/onnx_shape_inference/container_value_inference/rows/[]/optional_presence/[]",
    "/onnx_shape_inference/container_value_inference/rows/[]/reason_codes/[]",
    "/onnx_shape_inference/container_value_inference/method",
    "/onnx_shape_inference/container_value_inference/interpretation_boundary",
    "/onnx_shape_inference/ml_value_inference/schema", "/onnx_shape_inference/ml_value_inference/status",
    "/onnx_shape_inference/ml_value_inference/evidence_class", "/onnx_shape_inference/ml_value_inference/source_release",
    "/onnx_shape_inference/ml_value_inference/source_commit",
    "/onnx_shape_inference/ml_value_inference/source_documents/[]/role",
    "/onnx_shape_inference/ml_value_inference/source_documents/[]/source_ref",
    "/onnx_shape_inference/ml_value_inference/source_documents/[]/sha256",
    "/onnx_shape_inference/ml_value_inference/runtime_reference_commit",
    "/onnx_shape_inference/ml_value_inference/runtime_reference_documents/[]/role",
    "/onnx_shape_inference/ml_value_inference/runtime_reference_documents/[]/source_ref",
    "/onnx_shape_inference/ml_value_inference/runtime_reference_documents/[]/sha256",
    "/onnx_shape_inference/ml_value_inference/assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/passed_node_count",
    "/onnx_shape_inference/ml_value_inference/partially_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/failed_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_sequence_length_output_count",
    "/onnx_shape_inference/ml_value_inference/exact_class_key_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_class_key_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_class_key_node_count",
    "/onnx_shape_inference/ml_value_inference/map_producer_node_count",
    "/onnx_shape_inference/ml_value_inference/map_consumer_node_count",
    "/onnx_shape_inference/ml_value_inference/tensor_mapper_node_count",
    "/onnx_shape_inference/ml_value_inference/tensor_aggregator_node_count",
    "/onnx_shape_inference/ml_value_inference/tensor_selector_node_count",
    "/onnx_shape_inference/ml_value_inference/tensor_normalization_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_dense_output_shape_count",
    "/onnx_shape_inference/ml_value_inference/exact_vocabulary_entry_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_vocabulary_entry_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_vocabulary_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_category_pair_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_category_active_key_count",
    "/onnx_shape_inference/ml_value_inference/duplicate_category_active_key_node_count",
    "/onnx_shape_inference/ml_value_inference/feature_vectorizer_node_count",
    "/onnx_shape_inference/ml_value_inference/feature_vectorizer_exact_width_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_feature_vectorizer_configured_feature_count",
    "/onnx_shape_inference/ml_value_inference/feature_vectorizer_truncating_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_feature_vectorizer_truncated_feature_count_per_batch",
    "/onnx_shape_inference/ml_value_inference/exact_feature_vectorizer_padded_feature_count_per_batch",
    "/onnx_shape_inference/ml_value_inference/array_feature_extractor_node_count",
    "/onnx_shape_inference/ml_value_inference/array_feature_extractor_exact_index_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_array_feature_extractor_index_count",
    "/onnx_shape_inference/ml_value_inference/array_feature_extractor_duplicate_index_count",
    "/onnx_shape_inference/ml_value_inference/array_feature_extractor_bounds_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/array_feature_extractor_bounds_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/binarizer_node_count",
    "/onnx_shape_inference/ml_value_inference/binarizer_exact_static_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_binarizer_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_binarizer_above_threshold_count",
    "/onnx_shape_inference/ml_value_inference/exact_binarizer_at_or_below_threshold_count",
    "/onnx_shape_inference/ml_value_inference/exact_binarizer_equal_threshold_count",
    "/onnx_shape_inference/ml_value_inference/binarizer_schema_default_threshold_node_count",
    "/onnx_shape_inference/ml_value_inference/binarizer_nonfinite_threshold_node_count",
    "/onnx_shape_inference/ml_value_inference/normalizer_node_count",
    "/onnx_shape_inference/ml_value_inference/normalizer_static_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/normalizer_output_materialized_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_zero_divisor_row_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_negative_max_divisor_row_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_integer_float32_rounding_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_signed_overflow_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/exact_normalizer_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/normalizer_schema_default_mode_node_count",
    "/onnx_shape_inference/ml_value_inference/tensor_affine_scaler_node_count",
    "/onnx_shape_inference/ml_value_inference/scaler_node_count",
    "/onnx_shape_inference/ml_value_inference/scaler_static_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/scaler_output_materialized_node_count",
    "/onnx_shape_inference/ml_value_inference/scaler_invalid_runtime_contract_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_integer_float32_rounding_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/exact_scaler_zero_scale_count",
    "/onnx_shape_inference/ml_value_inference/tensor_imputation_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_static_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_output_materialized_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_invalid_runtime_contract_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_scalar_first_fallback_node_count",
    "/onnx_shape_inference/ml_value_inference/imputer_pinned_cpu_dtype_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_replacement_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_nan_replacement_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_unchanged_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_ignored_imputed_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_non_finite_imputed_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/exact_imputer_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/tensor_encoder_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_encoder_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_static_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_output_materialized_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_invalid_contract_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_duplicate_vocabulary_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_unknown_all_zero_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_guaranteed_runtime_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_pinned_cpu_dtype_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_noncanonical_zeros_node_count",
    "/onnx_shape_inference/ml_value_inference/onehot_unrepresentable_numeric_cast_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_matched_input_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_unknown_input_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_numeric_to_int64_changed_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_numeric_to_int64_invalid_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_output_one_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_output_zero_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_duplicate_category_count",
    "/onnx_shape_inference/ml_value_inference/exact_onehot_unreachable_duplicate_column_count",
    "/onnx_shape_inference/ml_value_inference/tensor_label_mapping_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_static_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_output_materialized_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_onnx_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_pinned_ort_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_pinned_cpu_dtype_pair_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_duplicate_semantic_conflict_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_nan_semantic_conflict_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_default_path_node_count",
    "/onnx_shape_inference/ml_value_inference/label_encoder_schema_runtime_output_mismatch_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_key_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_match_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_default_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_duplicate_key_hit_count",
    "/onnx_shape_inference/ml_value_inference/exact_label_encoder_schema_runtime_mismatch_count",
    "/onnx_shape_inference/ml_value_inference/linear_model_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_classifier_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_regressor_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_onnx_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_pinned_ort_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_reference_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_pinned_cpu_dtype_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_post_transform_hazard_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_unused_coefficient_node_count",
    "/onnx_shape_inference/ml_value_inference/linear_ignored_intercept_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_used_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_unused_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_unresolved_coefficient_use_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_ignored_intercept_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_linear_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/svm_model_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_classifier_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_regressor_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_linear_mode_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_svc_mode_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_onnx_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_pinned_ort_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_regressor_pinned_cpu_dtype_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_schema_runtime_score_width_mismatch_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_ignored_post_transform_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_ignored_parameter_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_non_finite_node_count",
    "/onnx_shape_inference/ml_value_inference/svm_reference_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_vector_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_pairwise_classifier_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_used_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unused_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unresolved_support_vector_use_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_used_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unused_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unresolved_coefficient_use_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_rho_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_used_rho_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unused_rho_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_unresolved_rho_use_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_svm_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_model_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_classifier_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_regressor_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_deprecated_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_onnx_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_pinned_ort_contract_failure_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_pinned_cpu_dtype_gap_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_reference_assessed_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_non_finite_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_reference_boundary_node_count",
    "/onnx_shape_inference/ml_value_inference/tree_ensemble_semantic_hazard_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_tree_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_root_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_branch_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_leaf_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reachable_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reachable_leaf_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_orphan_node_or_leaf_count",
    "/onnx_shape_inference/ml_value_inference/maximum_tree_ensemble_depth",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_cycle_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_duplicate_node_identity_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_invalid_child_reference_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_invalid_feature_id_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_root_mismatch_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_multiple_parent_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_weight_tuple_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_used_weight_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_unused_weight_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_unresolved_weight_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_ignored_nonleaf_weight_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_invalid_weight_reference_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_invalid_weight_id_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_single_target_ignored_weight_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_membership_node_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_membership_set_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_membership_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_membership_duplicate_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_membership_separator_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_row_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_path_step_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_output_score_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_non_finite_score_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_decision_boundary_count",
    "/onnx_shape_inference/ml_value_inference/exact_tree_ensemble_reference_unwritten_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scope",
    "/onnx_shape_inference/ml_value_inference/rows/[]/node_index",
    "/onnx_shape_inference/ml_value_inference/rows/[]/op_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/contract_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imported_opset",
    "/onnx_shape_inference/ml_value_inference/rows/[]/status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/output_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_map_key_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_map_value_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_input_map_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/sparse_key_bounds_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_rank",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_shape/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_batch_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_feature_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/class_key_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/class_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/duplicate_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/class_key_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_output_sequence_length",
    "/onnx_shape_inference/ml_value_inference/rows/[]/canonical_output_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/output_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/output_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_output_rank",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_output_shape/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_dense_output_element_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/output_shape_basis",
    "/onnx_shape_inference/ml_value_inference/rows/[]/runtime_reference_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/attribute_mode",
    "/onnx_shape_inference/ml_value_inference/rows/[]/cast_to",
    "/onnx_shape_inference/ml_value_inference/rows/[]/map_form",
    "/onnx_shape_inference/ml_value_inference/rows/[]/max_map",
    "/onnx_shape_inference/ml_value_inference/rows/[]/vocabulary_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/vocabulary_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/duplicate_vocabulary_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/vocabulary_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/mapping_direction",
    "/onnx_shape_inference/ml_value_inference/rows/[]/category_pair_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/category_string_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/category_int64_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/duplicate_string_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/duplicate_int64_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/active_duplicate_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/active_default_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/active_default_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/category_string_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/category_int64_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_names/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_dtypes/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_ranks/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/input_shapes/[]/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_batch_counts/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_input_row_feature_counts/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/configured_feature_dimensions/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/configured_feature_dimension_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/total_configured_feature_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/copied_feature_counts_per_input/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/padded_feature_counts_per_input/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/truncated_feature_counts_per_input/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_copied_feature_count_per_batch",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_padded_feature_count_per_batch",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_truncated_feature_count_per_batch",
    "/onnx_shape_inference/ml_value_inference/rows/[]/padded_input_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/truncated_input_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/index_input_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/index_input_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/index_input_rank",
    "/onnx_shape_inference/ml_value_inference/rows/[]/index_input_shape/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_index_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_index_values_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_index_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/duplicate_index_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/index_bounds_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/out_of_bounds_index_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/threshold_value_text",
    "/onnx_shape_inference/ml_value_inference/rows/[]/threshold_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/static_value_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_static_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_above_threshold_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_at_or_below_threshold_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/exact_equal_threshold_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_mode",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_mode_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_static_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_exact_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_batch_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_row_width",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_divisor_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_divisor_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_zero_divisor_row_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_negative_max_divisor_row_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_integer_float32_rounding_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_signed_overflow_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_output_materialized",
    "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_parameter_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_parameter_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_parameter_mode",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_feature_stride",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_scale_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_offset_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_scale_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_offset_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_zero_scale_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_static_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_exact_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_integer_float32_rounding_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_output_materialized",
    "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_parameter_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_parameter_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_parameter_mode",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_attribute_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_feature_stride",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_imputed_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_imputed_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_replaced_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_replaced_value_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_ignored_imputed_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_non_finite_imputed_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_static_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_exact_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_exact_replacement_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_exact_nan_replacement_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_exact_unchanged_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_non_finite_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_signed_zero_output_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_output_materialized",
    "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_parameter_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_parameter_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_category_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_category_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_category_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_duplicate_category_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_unreachable_duplicate_column_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_unreachable_duplicate_column_indices/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_zeros_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_zeros_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_zeros_enabled",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_zeros_canonical_boolean",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_static_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_exact_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_exact_matched_input_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_exact_unknown_input_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_numeric_to_int64_changed_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_numeric_to_int64_invalid_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_guaranteed_runtime_failure",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_exact_output_one_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_exact_output_zero_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_output_materialized",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_unknown_input_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/resolved_schema_version",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_onnx_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_pinned_ort_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_pinned_ort_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_key_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_value_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_key_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_value_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_default_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_default_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_key_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_value_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_duplicate_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_nan_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_non_finite_key_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_non_finite_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_runtime_duplicate_policy",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_schema_duplicate_policy",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_static_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_exact_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_exact_match_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_exact_default_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_exact_duplicate_key_hit_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_schema_runtime_mismatch_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_output_materialized",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_runtime_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_schema_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/label_encoder_mismatch_input_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/output_names/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/canonical_output_types/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/canonical_output_shapes/[]/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_label_output_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_score_output_name",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_label_output_dtype",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_label_output_shape/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_score_output_shape/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_score_class_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/classifier_binary_score_expansion",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_onnx_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_onnx_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_pinned_ort_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_pinned_ort_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_class_or_target_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_expected_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_used_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_unused_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_intercept_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_intercepts_used",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_ignored_intercept_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_label_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_label_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_duplicate_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_multi_class_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_multi_class_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_multi_class_used_by_pinned_ort",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_targets_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_targets_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_post_transform",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_post_transform_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_non_finite_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_decision_boundary_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_raw_score_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_output_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_label_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_boundary",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_onnx_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_onnx_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_pinned_ort_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_pinned_ort_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_mode",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_kernel_type",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_kernel_type_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_kernel_params/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_kernel_params_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_post_transform",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_post_transform_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_post_transform_applied_by_pinned_ort",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_class_label_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_class_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_class_label_values/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_duplicate_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_vectors_per_class/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_n_supports",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_one_class_value",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_one_class_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_vector_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_pairwise_classifier_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_schema_score_width",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_pinned_ort_score_width",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_schema_runtime_score_width_mismatch",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_expected_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_used_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_unused_support_vector_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_expected_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_used_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_unused_coefficient_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_rho_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_expected_rho_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_used_rho_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_unused_rho_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_prob_a_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_prob_b_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_probability_enabled",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_expected_probability_parameter_count_per_array",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_used_probability_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_unused_probability_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_linear_mode_forced_kernel",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_non_finite_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_decision_boundary_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_raw_score_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_output_score_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_label_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/svm_reference_boundary",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_encoding",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_deprecated_operator",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_onnx_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_onnx_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_pinned_ort_contract_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_pinned_ort_contract_reason",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_aggregate_function",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_aggregate_function_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_post_transform",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_post_transform_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_base_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_base_value_source",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_class_or_target_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_class_label_kind",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_class_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_duplicate_class_label_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_class_label_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_exact_tree_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_exact_root_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_exact_node_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_exact_branch_node_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_exact_leaf_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reachable_node_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reachable_leaf_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_orphan_node_or_leaf_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_max_depth",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_cycle_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_duplicate_node_identity_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_invalid_child_reference_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_invalid_feature_id_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_root_mismatch_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_multiple_parent_node_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_weight_tuple_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_used_weight_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_unused_weight_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_unresolved_weight_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_ignored_nonleaf_weight_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_invalid_weight_reference_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_invalid_weight_id_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_single_target_ignored_weight_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_membership_node_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_membership_set_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_membership_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_membership_duplicate_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_membership_separator_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_non_finite_parameter_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_pinned_cpu_dtype_gap",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_assessment_status",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_input_value_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_row_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_path_step_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_raw_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_output_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_non_finite_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_decision_boundary_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_unwritten_score_count",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_raw_score_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_output_score_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_label_preview/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/tree_reference_boundary",
    "/onnx_shape_inference/ml_value_inference/rows/[]/reason_codes/[]",
    "/onnx_shape_inference/ml_value_inference/rows/[]/risk_codes/[]",
    "/onnx_shape_inference/ml_value_inference/method",
    "/onnx_shape_inference/ml_value_inference/interpretation_boundary",
    "/onnx_shape_inference/declaration_conflicts/[]/node_index", "/onnx_shape_inference/declaration_conflicts/[]/op_name",
    "/onnx_shape_inference/declaration_conflicts/[]/tensor_name", "/onnx_shape_inference/declaration_conflicts/[]/field",
    "/onnx_shape_inference/declaration_conflicts/[]/declared", "/onnx_shape_inference/declaration_conflicts/[]/inferred",
    "/onnx_shape_inference/semantic_contract_conflicts/[]/node_index", "/onnx_shape_inference/semantic_contract_conflicts/[]/op_name",
    "/onnx_shape_inference/semantic_contract_conflicts/[]/output_names/[]", "/onnx_shape_inference/semantic_contract_conflicts/[]/reason",
    "/onnx_shape_inference/propagated_static_value_tensor_count", "/onnx_shape_inference/inferred_outputs",
    "/onnx_shape_inference/unknown_tensor_count", "/onnx_shape_inference/non_dense_value_count",
    "/onnx_shape_inference/node_output_assessment_ratio",
    "/onnx_shape_inference/method", "/onnx_shape_inference/interpretation_boundary",
  ]],
  ["onnx.tensor_data_types", [
    "/onnx_tensor_data_type_contract/schema", "/onnx_tensor_data_type_contract/status",
    "/onnx_tensor_data_type_contract/evidence_class", "/onnx_tensor_data_type_contract/source_release",
    "/onnx_tensor_data_type_contract/source_commit", "/onnx_tensor_data_type_contract/source_ref",
    "/onnx_tensor_data_type_contract/source_sha256", "/onnx_tensor_data_type_contract/concrete_data_type_count",
    "/onnx_tensor_data_type_contract/fixed_width_numeric_data_type_count",
    "/onnx_tensor_data_type_contract/packed_data_type_count",
    "/onnx_tensor_data_type_contract/raw_numeric_decoder_count",
    "/onnx_tensor_data_type_contract/typed_numeric_decoder_count",
    "/onnx_tensor_data_type_contract/packed_data_types",
    "/onnx_tensor_data_type_contract/packing_rule",
    "/onnx_tensor_data_type_contract/numerical_integrity_projection",
    "/onnx_tensor_data_type_contract/types/[]/id",
    "/onnx_tensor_data_type_contract/types/[]/name",
    "/onnx_tensor_data_type_contract/types/[]/storage_bits",
  ]],
  ["onnx.type_proto_contract", [
    "/onnx_type_proto_contract/schema", "/onnx_type_proto_contract/status",
    "/onnx_type_proto_contract/evidence_class", "/onnx_type_proto_contract/source_release",
    "/onnx_type_proto_contract/source_commit", "/onnx_type_proto_contract/source_ref",
    "/onnx_type_proto_contract/source_sha256", "/onnx_type_proto_contract/declaration_count",
    "/onnx_type_proto_contract/declared_type_count", "/onnx_type_proto_contract/undeclared_optional_type_count",
    "/onnx_type_proto_contract/valid_type_count", "/onnx_type_proto_contract/invalid_type_count",
    "/onnx_type_proto_contract/type_node_count", "/onnx_type_proto_contract/maximum_type_depth",
    "/onnx_type_proto_contract/type_text_bytes", "/onnx_type_proto_contract/tensor_value_count",
    "/onnx_type_proto_contract/sparse_tensor_value_count", "/onnx_type_proto_contract/non_dense_value_count",
    "/onnx_type_proto_contract/sequence_value_count", "/onnx_type_proto_contract/map_value_count",
    "/onnx_type_proto_contract/optional_value_count", "/onnx_type_proto_contract/opaque_value_count",
    "/onnx_type_proto_contract/symbolic_dimension_count", "/onnx_type_proto_contract/unknown_dimension_count",
    "/onnx_type_proto_contract/kind_counts/[]/name", "/onnx_type_proto_contract/kind_counts/[]/count",
    "/onnx_type_proto_contract/invalid_rows/[]/scope", "/onnx_type_proto_contract/invalid_rows/[]/role",
    "/onnx_type_proto_contract/invalid_rows/[]/value_name", "/onnx_type_proto_contract/invalid_rows/[]/kind",
    "/onnx_type_proto_contract/invalid_rows/[]/canonical_type", "/onnx_type_proto_contract/invalid_rows/[]/reason_codes/[]",
    "/onnx_type_proto_contract/method", "/onnx_type_proto_contract/interpretation_boundary",
  ]],
  ["onnx.sparse_tensor_contract", [
    "/onnx_sparse_tensor_contract/schema", "/onnx_sparse_tensor_contract/status",
    "/onnx_sparse_tensor_contract/evidence_class", "/onnx_sparse_tensor_contract/source_release",
    "/onnx_sparse_tensor_contract/source_commit", "/onnx_sparse_tensor_contract/source_ref",
    "/onnx_sparse_tensor_contract/source_sha256", "/onnx_sparse_tensor_contract/sparse_tensor_count",
    "/onnx_sparse_tensor_contract/graph_sparse_initializer_count",
    "/onnx_sparse_tensor_contract/attribute_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/valid_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/invalid_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/partially_assessed_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/declared_nnz_total",
    "/onnx_sparse_tensor_contract/dense_logical_element_total",
    "/onnx_sparse_tensor_contract/embedded_payload_bytes",
    "/onnx_sparse_tensor_contract/external_payload_component_count",
    "/onnx_sparse_tensor_contract/verified_external_payload_component_count",
    "/onnx_sparse_tensor_contract/external_payload_coverage_status",
    "/onnx_sparse_tensor_contract/index_content_assessed_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/index_content_failed_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/index_content_unassessed_sparse_tensor_count",
    "/onnx_sparse_tensor_contract/assessed_index_count",
    "/onnx_sparse_tensor_contract/out_of_bounds_index_count",
    "/onnx_sparse_tensor_contract/duplicate_index_count",
    "/onnx_sparse_tensor_contract/unsorted_index_count",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/scope",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/tensor_role",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/sparse_tensor_name",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/reason_codes/[]",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/values_dtype",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/values_shape/[]",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/indices_dtype",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/indices_shape/[]",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/dense_shape/[]",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/dense_rank",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/nnz",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/dense_logical_elements",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/index_encoding",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/index_content_status",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/assessed_index_count",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/out_of_bounds_index_count",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/duplicate_index_count",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/unsorted_index_count",
    "/onnx_sparse_tensor_contract/invalid_rows/[]/payload_status",
    "/onnx_sparse_tensor_contract/method", "/onnx_sparse_tensor_contract/interpretation_boundary",
  ]],
  ["onnx.quantization_binding", [
    "/onnx_quantization_binding/schema", "/onnx_quantization_binding/status",
    "/onnx_quantization_binding/evidence_class",
    "/onnx_quantization_binding/binding_count", "/onnx_quantization_binding/valid_binding_count",
    "/onnx_quantization_binding/invalid_binding_count", "/onnx_quantization_binding/unresolved_binding_count",
    "/onnx_quantization_binding/explicit_qdq_boundary_count",
    "/onnx_quantization_binding/integer_compute_without_real_scale_count",
    "/onnx_quantization_binding/main_graph_annotation_count",
    "/onnx_quantization_binding/valid_annotation_count",
    "/onnx_quantization_binding/invalid_annotation_count",
    "/onnx_quantization_binding/unresolved_annotation_count",
    "/onnx_quantization_binding/nested_graph_annotation_count",
    "/onnx_quantization_binding/annotation_scope_status",
    "/onnx_quantization_binding/annotation_source_ref",
    "/onnx_quantization_binding/annotation_source_sha256",
  ]],
  ["onnx.ort_source_compatibility", [
    "/ort_compatibility_assessment_status", "/ort_compatibility_evidence_schema",
    "/ort_compatibility_evidence_access", "/ort_compatibility_evidence/schema",
    "/ort_compatibility_evidence/method_version", "/ort_compatibility_evidence/source_commit",
    "/ort_compatibility_evidence/access_scope", "/ort_compatibility_evidence/assessment_status",
    "/ort_compatibility_evidence/runtime_floor/status",
    "/ort_compatibility_evidence/runtime_floor/model_ir_version",
    "/ort_compatibility_evidence/runtime_floor/minimum_ort_version",
    "/ort_compatibility_evidence/runtime_floor/standard_minimum_ort_version",
    "/ort_compatibility_evidence/runtime_floor/contrib_minimum_ort_version",
    "/ort_compatibility_evidence/runtime_floor/standard_domain_opset",
    "/ort_compatibility_evidence/runtime_floor/standard_ml_domain_opset",
    "/ort_compatibility_evidence/runtime_floor/unresolved_domains",
    "/ort_compatibility_evidence/runtime_floor/model_local_function_domains",
    "/ort_compatibility_evidence/runtime_floor/source_backed_external_domains",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/domain",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/op_name",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/imported_opset",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/minimum_ort_version",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/source_ref",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/source_sha256",
    "/ort_compatibility_evidence/runtime_floor/contrib_operator_floors/[]/evidence_class",
    "/ort_compatibility_evidence/runtime_floor/basis",
    "/ort_compatibility_evidence/runtime_floor/source_refs",
    "/ort_compatibility_evidence/runtime_floor/source_documents/[]/role",
    "/ort_compatibility_evidence/runtime_floor/source_documents/[]/source_ref",
    "/ort_compatibility_evidence/runtime_floor/source_documents/[]/sha256",
    "/ort_compatibility_evidence/runtime_floor/source_documents/[]/detail",
    "/ort_compatibility_evidence/source_condition_inventory/schema",
    "/ort_compatibility_evidence/source_condition_inventory/status",
    "/ort_compatibility_evidence/source_condition_inventory/source_rule_count",
    "/ort_compatibility_evidence/source_condition_inventory/cpu_registration_variant_count",
    "/ort_compatibility_evidence/source_condition_inventory/cpu_registration_variant_with_signature_count",
    "/ort_compatibility_evidence/source_condition_inventory/cpu_registration_variant_with_type_constraint_count",
    "/ort_compatibility_evidence/source_condition_inventory/machine_condition_count",
    "/ort_compatibility_evidence/source_condition_inventory/versioned_scalar_schema_default_binding_count",
    "/ort_compatibility_evidence/source_condition_inventory/unresolved_source_fragment_count",
    "/ort_compatibility_evidence/source_condition_inventory/informational_source_note_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_condition_count",
    "/ort_compatibility_evidence/execution_providers/[]/source_scope",
    "/ort_compatibility_evidence/execution_providers/[]/evaluator_coverage",
    "/ort_compatibility_evidence/execution_providers/[]/support_evidence_class",
    "/ort_compatibility_evidence/execution_providers/[]/ops/[]/artifact_conditions/[]/source_ref",
    "/ort_compatibility_evidence/execution_providers/[]/ops/[]/artifact_conditions/[]/source_sha256",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_condition_pass_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_condition_fail_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_condition_unresolved_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_precheck_pass_op_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_precheck_definite_fail_op_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_precheck_unresolved_op_count",
    "/ort_compatibility_evidence/execution_providers/[]/artifact_precheck_no_condition_op_count",
    "/ort_compatibility_evidence/execution_providers/[]/source_candidate_after_artifact_precheck_count",
    "/ort_compatibility_evidence/execution_providers/[]/source_candidate_after_artifact_precheck_ratio",
    "/ort_assignment_capture_capability/status", "/ort_assignment_capture_capability/evidence_class",
    "/ort_assignment_capture_capability/pinned_ort_web_version",
    "/ort_assignment_capture_capability/automatic_browser_assignment_capture",
    "/ort_assignment_capture_capability/start_profiling_status",
    "/ort_assignment_capture_capability/end_profiling_status",
    "/ort_assignment_capture_capability/external_runtime_profile_import",
    "/ort_assignment_capture_capability/pinned_native_capture_available",
    "/ort_assignment_capture_capability/pinned_native_runtime",
    "/ort_assignment_capture_capability/native_capture_command",
    "/ort_assignment_capture_capability/native_capture_schema",
    "/ort_assignment_capture_capability/native_profile_schema",
    "/ort_assignment_capture_capability/native_profile_roles",
    "/ort_assignment_capture_capability/required_observation_path",
  ]],
  ["onnx.ep_portability", [
    "/ort_ep_portability_frontier/schema", "/ort_ep_portability_frontier/evidence_class",
    "/ort_ep_portability_frontier/all_ep_source_match_op_count",
    "/ort_ep_portability_frontier/all_ep_artifact_precheck_candidate_op_count",
    "/ort_ep_portability_frontier/all_ep_artifact_precheck_candidate_op_ratio",
    "/ort_ep_portability_frontier/all_ep_artifact_precheck_candidate_macs",
    "/ort_ep_portability_frontier/all_ep_artifact_precheck_candidate_mac_ratio",
  ]],
  ...[
    ["tflite.accumulator", "accumulator_atlas"],
    ["tflite.requantization", "requantization_fidelity"],
    ["tflite.kernel_witness", "kernel_extremum_witness"],
    ["tflite.channel_vitality", "channel_vitality", ["source_evidence_schema", "minimum_default_inclusive_code_span", "minimum_single_inclusive_code_span"]],
    ["tflite.rounding_equivalence", "rounding_equivalence", ["source_evidence_schema"]],
    ["tflite.accumulator_reachability", "accumulator_reachability", ["source_evidence_schema", "lattice_compatible_state_count_decimal"]],
    ["tflite.numerical_abi", "numerical_abi_propagation", ["source_evidence_schema", "exact_source_boundary_edge_instance_count", "unassessed_source_boundary_edge_instance_payload_count"]],
    ["tflite.input_witness", "input_counterexample", ["source_evidence_schema"]],
    ["tflite.preprocessing", "preprocessing_realizability", ["source_input_counterexample_schema", "source_input_counterexample_portfolio_sha256", "ineligible_witness_count"]],
    ["tflite.quantization_lattice", "quantization_lattice"],
    ["tflite.contract_migration", "contract_migration"],
    ["tflite.residual_step", "residual_step_response", ["candidate_add_count", "containment_silent_transition_count"]],
    ["tflite.residual_distortion", "residual_contract_distortion", ["candidate_add_count"]],
  ].map(([metricId, key, extra = []]) => [metricId, [
    `/${key}/status`, `/${key}/evidence_class`, `/${key}/schema`,
    ...extra.map((field) => `/${key}/${field}`),
  ]]),
]);

const COMPUTATION_OBJECT_KEYS = new Set([
  "size_breakdown", "artifact_byte_integrity", "tensor_liveness", "tensor_arena_plan", "movement_analysis", "block_inventory",
  "tflite_sparse_storage_contract", "tflite_subgraph_inventory", "tflite_subgraph_deep_analysis",
  "dynamic_shape_cost_contract",
  "weight_integrity", "quantization_status", "mac_assessment", "runtime_compat",
  "predicted_partition_boundaries", "xnnpack_selector_evidence_provenance", "tflite_delegate_compatibility_evidence",
  "deployment_frontier", "deployment_delta", "delegation_repair", "quantization_lattice",
  "accumulator_atlas", "requantization_fidelity", "kernel_extremum_witness",
  "channel_vitality", "rounding_equivalence", "accumulator_reachability",
  "numerical_abi_propagation", "input_counterexample", "preprocessing_realizability",
  "contract_migration", "residual_step_response", "residual_contract_distortion",
  "quant_research_coverage",
  "onnx_domain_analysis", "onnx_quantization_binding", "ort_compatibility_evidence",
  "ort_ep_portability_frontier", "onnx_shape_inference", "onnx_tensor_data_type_contract",
  "onnx_type_proto_contract", "onnx_sparse_tensor_contract", "tensorrt_static_preflight", "coreml_blob_integrity",
  "flexible_input_scenarios",
  "executorch_program", "executorch_flat_tensor",
]);

function hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value || {}, key);
}

function emitted(value) {
  return value != null && (typeof value !== "object" || Array.isArray(value) || Object.keys(value).length > 0);
}

function objectStatus(value, fallback = "not_assessed") {
  if (!emitted(value)) return fallback;
  const status = String(value?.assessment_status || value?.status || "assessed").toLowerCase();
  if (status.startsWith("not_applicable")) return "not_applicable";
  if (status.startsWith("not_assessed") || status.startsWith("not_assessable") || status.startsWith("not_loaded")) return "not_assessed";
  if (status.startsWith("suppressed")) return "suppressed";
  if (status.startsWith("partial") || status.startsWith("partially_assessed") || status.startsWith("incomplete")) return "partial";
  return "assessed";
}

function quantResearchMetricStatus(analysis, evidenceKey) {
  const coverage = analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const row = (coverage?.labs || []).find((candidate) => candidate.evidence_key === evidenceKey);
  return row?.status || objectStatus(analysis?.[evidenceKey]);
}

function spec(id, label, {
  formats = ["tflite", "onnx"],
  keys = [],
  status,
  evidenceClass = "DERIVED",
  pointers = [],
  report,
  viewer = [],
  exports = ["engineering_evidence.json", "static/static_analysis.json"],
  method,
}) {
  return { id, label, formats, keys, status, evidenceClass, pointers, report, viewer, exports, method };
}

const SPECS = [
  spec("executorch.serialized_contract", "ExecuTorch ET12/FT01 serialized contract", {
    formats: ["executorch"],
    keys: [
      "schema", "format", "filename", "file_size", "file_size_bytes", "model_sha256", "version",
      "executorch_container", "executorch_program", "executorch_flat_tensor", "executorch_sections_suppressed",
      "subgraphs", "operator_count", "tensor_count", "input_tensor_indices", "output_tensor_indices",
      "inputs", "outputs", "tensors", "ops", "histogram", "stages",
      "total_macs", "total_macs_decimal", "total_ops", "mac_assessment", "tensor_liveness", "size_breakdown",
      "weight_integrity", "metadata_presence", "runtime_compat", "static_audit_timing",
      "xnnpack_assumption", "xnnpack_chains", "xnnpack_chain_breaks", "markdown",
      "findings", "recommendations", "suspects", "artifact_bundle",
    ],
    status: (a) => {
      const container = String(a?.executorch_container || "");
      const common = a?.schema === "deepbom.static_analysis.executorch.v1.1"
        && Number.isSafeInteger(a?.file_size) && a.file_size >= 0
        && Array.isArray(a?.ops) && a.ops.length === Number(a?.operator_count)
        && Array.isArray(a?.tensors) && a.tensors.length === Number(a?.tensor_count)
        && a?.size_breakdown?.status === "assessed"
        && ["pass", "warn"].includes(String(a?.weight_integrity?.status || ""));
      if (!common) return "partial";
      if (container === "pte") {
        return a?.executorch_program?.identifier === "ET12"
          && Array.isArray(a.executorch_program?.plans)
          && a.executorch_program.plans.length === Number(a?.subgraphs)
          && a?.tensor_liveness?.status === "observed_aot_plan"
          && Number(a?.mac_assessment?.kernel_instruction_count || 0) >= Number(a?.mac_assessment?.source_bound_kernel_instruction_count || 0)
          && (a?.mac_assessment?.complete ? /^(?:0|[1-9]\d*)$/.test(String(a?.total_macs_decimal || "")) : a?.total_macs === null)
          ? "assessed" : "partial";
      }
      if (container === "ptd") {
        return a?.executorch_flat_tensor?.identifier === "FT01"
          && Number(a?.operator_count) === 0 && Number(a?.subgraphs) === 0
          && a?.mac_assessment?.status === "not_applicable_data_container"
          && a?.tensor_liveness?.status === "not_applicable_data_container"
          ? "assessed" : "partial";
      }
      return "partial";
    },
    evidenceClass: "OBSERVED/SOURCE_PINNED/DERIVED",
    pointers: [
      "/evidence/static_analysis/executorch_container",
      "/evidence/static_analysis/executorch_program",
      "/evidence/static_analysis/executorch_flat_tensor",
      "/evidence/static_analysis/mac_assessment",
      "/evidence/static_analysis/tensor_liveness",
      "/evidence/static_analysis/size_breakdown",
      "/evidence/static_analysis/weight_integrity",
    ],
    report: "## ExecuTorch Serialized Contract",
    viewer: ["Overview", "Explorer", "Reports"],
    method: "Decode bounded ET12 Program or FT01 FlatTensor FlatBuffers against pinned pytorch/executorch schemas; bind matching KernelCall rows to a generated 209-operator portable signature registry; preserve processed delegate payload byte ranges and SHA-256; validate public-schema FlatBuffer root envelopes; optionally bind a duplicate-key-checked selected-build attestation containing exact source/build inputs, backend/operator inventories, and runtime binary digests; conserve argument/result aliases, plans, instructions, EValues, segments, constant ranges, exact nominal tensor-contraction MACs where semantics and shapes close, and AOT planned non-constant memory. Preserve custom or mismatched kernels, delegate internals, runtime allocation, executed placement, physical transfer, and latency as explicitly unassessed.",
  }),
  spec("serialized.container_contract", "Serialized tensor-container contract", {
    formats: ["gguf", "safetensors"],
    keys: [
      "schema", "format", "filename", "file_size", "file_size_bytes", "model_sha256",
      "operator_count", "tensor_count", "quantized_tensors", "per_channel_tensors",
      "per_tensor_tensors", "total_macs", "mac_assessment", "ops", "inputs", "outputs",
      "tensors", "tensor_inventory", "metadata_presence", "quantization_status", "weight_integrity",
      "format_extensions", "gguf", "safetensors", "static_audit_timing", "markdown",
      "findings", "recommendations", "suspects",
      "artifact_bundle",
    ],
    status: (a, format) => {
      const detail = format === "gguf" ? a?.gguf : a?.safetensors;
      return Array.isArray(a?.tensors) && detail?.payload_coverage_status === "complete_without_gaps_or_overlaps"
        ? "assessed"
        : "partial";
    },
    evidenceClass: "OBSERVED/SOURCE_PINNED",
    pointers: ["/evidence/static_analysis/tensors", "/evidence/static_analysis/quantization_status"],
    report: (format) => format === "gguf" ? "## GGUF Tensor Encoding Evidence" : "## SafeTensors Storage Evidence",
    viewer: ["Overview", "Tensor Encoding / Storage Contract", "Reports"],
    method: "Parse the bounded serialized header, validate every tensor shape and byte range, enforce non-overlap and complete payload coverage, and interpret encodings only against the pinned format implementation source.",
  }),
  spec("coreml.serialized_contract", "Core ML serialized graph and numerical contract", {
    formats: ["coreml"],
    keys: [
      "schema", "format", "filename", "file_size", "file_size_bytes", "model_sha256",
      "operator_count", "tensor_count", "quantized_tensors", "per_channel_tensors",
      "per_tensor_tensors", "total_macs", "mac_assessment", "ops", "inputs", "outputs",
      "tensors", "tensor_inventory", "metadata_presence", "quantization_status",
      "format_extensions", "coreml", "histogram", "static_audit_timing", "markdown",
      "findings", "recommendations", "suspects",
      "input_tensor_indices", "output_tensor_indices", "runtime_requirements", "states", "training_inputs",
      "weight_integrity", "coreml_blob_references", "coreml_blob_integrity", "size_breakdown", "tensor_liveness", "flexible_input_scenarios",
      "artifact_bundle",
    ],
    status: (a) => a?.quantization_status?.assessment_status === "assessed"
      && ((Array.isArray(a?.coreml?.neural_network?.layers) && a.coreml.neural_network.layers.length === Number(a?.operator_count || 0))
        || (a?.coreml?.model_type === "mlProgram" && Array.isArray(a?.ops) && a.ops.length === Number(a?.operator_count || 0))
        || (a?.coreml?.classical_model && Array.isArray(a?.ops) && a.ops.length === 1 && Number(a?.operator_count) === 1)
        || (a?.coreml?.pipeline?.model_summaries?.every?.((row) => row.graph_status === "decoded")
          && Array.isArray(a?.ops) && a.ops.length === Number(a?.operator_count || 0)))
      && a?.weight_integrity?.status === "assessed"
      && String(a?.mac_assessment?.status || "").startsWith("assessed_")
      && objectStatus(a?.tensor_liveness) === "assessed"
      && a?.size_breakdown?.status === "assessed"
      ? "assessed"
      : "partial",
    evidenceClass: "OBSERVED/SOURCE_PINNED",
    pointers: ["/evidence/static_analysis/coreml", "/evidence/static_analysis/ops", "/evidence/static_analysis/quantization_status"],
    report: "## Core ML Serialized Graph And Numerical Evidence",
    viewer: ["Overview", "Graph", "Numerical Contract", "Reports"],
    method: "Decode source-pinned Core ML NeuralNetwork layers, MIL SSA, GLM/SVM/TreeEnsemble payloads, and named pipeline stages; bind WeightParams/blob or classical FLOAT64 cardinality and exact payload digests; validate tree/support-vector/pipeline invariants; derive implemented operation shapes, arithmetic counts, byte conservation, static tensor liveness, source-defined cond/while peak envelopes, and bounded flexible-input scenarios while preserving runtime placement and task accuracy as unassessed.",
  }),
  spec("coreml.deployment_floor", "Core ML specification and observed-feature OS floor", {
    formats: ["coreml"], keys: [], status: (a) => objectStatus(a?.coreml?.deployment_floor), evidenceClass: "OBSERVED/SOURCE_PINNED/DERIVED",
    pointers: ["/evidence/static_analysis/coreml/deployment_floor", "/evidence/static_analysis/runtime_requirements"],
    report: "## Core ML Serialized Graph And Numerical Evidence", viewer: ["Overview", "Reports"],
    method: "Map the serialized specificationVersion through the exact pinned Model.proto OS-availability table and independently derive a necessary observed-feature version from representation, external dtype/flexibility, multi-function/state declarations, updatability, and MIL opset identity; reject declared versions below the observed-feature floor." }),
  spec("artifact.identity", "Artifact identity and byte size", {
    keys: ["schema", "format", "filename", "file_size", "model_sha256", "version", "target_profile", "cpu_cost_target_binding"],
    status: (a) => hasOwn(a, "file_size") ? "assessed" : "not_assessed",
    evidenceClass: "OBSERVED", pointers: ["/evidence/static_analysis/file_size", "/evidence/static_analysis/model_sha256"],
    report: "## Artifact And Target", viewer: ["Overview", "Reports"], method: "Read format, byte length, and digest from the locally selected artifact." }),
  spec("artifact.metadata", "Artifact metadata and signature inventory", {
    keys: ["metadata_presence"], status: (a) => objectStatus(a?.metadata_presence), evidenceClass: "OBSERVED",
    pointers: ["/evidence/static_analysis/metadata_presence"], report: "## Artifact Metadata & Signatures", viewer: ["Overview", "Reports"],
    method: "Inventory only metadata and signature structures actually embedded in the selected artifact format." }),
  spec("graph.inventory", "Graph, tensor, stage, and pattern inventory", {
    keys: ["subgraphs", "graph_name", "producer", "onnx_ir_version", "opsets", "operator_codes", "operator_count", "tensor_count", "ops", "tensors", "histogram", "stages", "patterns", "tensor_types", "graph_topology"],
    status: (a) => Array.isArray(a?.ops) && Array.isArray(a?.tensors) ? "assessed" : "not_assessed",
    evidenceClass: "OBSERVED/DERIVED", pointers: ["/evidence/static_analysis/ops", "/evidence/static_analysis/tensors", "/evidence/static_analysis/stages", "/evidence/static_analysis/patterns", "/evidence/static_analysis/graph_topology"],
    report: "## Stage And Pattern Summary", viewer: ["Overview", "Explorer"], method: "Parse graph records, then derive deterministic stage and topology-pattern groupings." }),
  spec("architecture.blocks", "Graph-semantic architecture block inventory", {
    formats: ["tflite"], keys: ["block_inventory"],
    status: (a) => objectStatus(a?.block_inventory), evidenceClass: "DERIVED",
    pointers: ["/evidence/static_analysis/block_inventory"],
    report: "## Architecture Block Inventory", viewer: ["Explorer / Blocks"],
    method: "Match graph-semantic motifs before name/shape fallback grouping, require complete unique op ownership, and conserve block/stage aggregates from owned op rows." }),
  spec("memory.cache_payload", "Logical cache payload decomposition", {
    keys: [],
    status: (a) => (a?.ops || []).some((op) => op?.cache_payload?.status === "assessed") ? "assessed" : "not_assessed",
    evidenceClass: "DERIVED",
    pointers: ["/evidence/static_analysis/ops"],
    report: "## Memory And Cache Hotspots", viewer: ["Explorer / Cache"],
    method: "Derive dilation-aware input-strip and output-row bytes from operator semantics; retain serialized kernel/bias bytes separately and divide only by the selected target cache reference." }),
  spec("contract.io", "Input and output tensor contract", {
    keys: ["inputs", "outputs", "input_contracts", "input_tensor_indices", "output_tensor_indices"],
    status: (a) => Array.isArray(a?.inputs) && Array.isArray(a?.outputs) ? "assessed" : "not_assessed",
    evidenceClass: "OBSERVED/DERIVED", pointers: ["/evidence/static_analysis/inputs", "/evidence/static_analysis/outputs"],
    report: "## Input/Output Contract", viewer: ["Overview", "Explorer"], method: "Read artifact tensor ABI; derive layout only from consumer operator semantics when possible." }),
  spec("runtime.artifact_requirements", "Artifact-side runtime requirements", {
    keys: ["runtime_compat"], status: (a, format) => {
      if (!emitted(a?.runtime_compat)) return "not_assessed";
      if (format === "onnx") return String(a?.ort_compatibility_assessment_status || "not_loaded") === "complete"
        && String(a?.runtime_compat?.effective_min_runtime_version || "") ? "assessed" : "partial";
      return a?.runtime_compat?.runtime_floor_status === "complete_for_observed_builtin_op_versions"
        && String(a?.runtime_compat?.effective_min_runtime_version || "") ? "assessed" : "partial";
    }, evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/runtime_compat"], report: (format) => format === "onnx" ? "## ONNX Runtime Requirements" : "## Artifact-side Runtime Requirements", viewer: ["Overview", "Reports"],
    method: "Read artifact operator/opset requirements and derive only the necessary runtime floor supported by the pinned compatibility source." }),
  spec("compute.macs", "Arithmetic and MAC accounting", {
    keys: ["mac_assessment", "total_macs", "total_macs_decimal", "total_ops", "total_ops_decimal"], status: (a, f) => f === "onnx" ? objectStatus(a?.mac_assessment) : hasOwn(a, "total_macs") ? "assessed" : "not_assessed",
    pointers: ["/evidence/static_analysis/total_macs", "/evidence/static_analysis/mac_assessment", "/evidence/static_analysis/ops"],
    report: "## Compute Hotspots", viewer: ["Overview", "Explorer"], method: "Sum only operator rows whose complete shape-dependent arithmetic formula is assessable; preserve unassessed rows as null." }),
  spec("cost.dynamic_shape", "Symbolic dynamic-shape cost contract", {
    keys: ["dynamic_shape_cost_contract"], status: (a) => {
      const status = String(a?.dynamic_shape_cost_contract?.status || "not_assessed");
      if (status === "not_applicable_static_shapes") return "not_applicable";
      return objectStatus(a?.dynamic_shape_cost_contract);
    }, evidenceClass: "DERIVED",
    pointers: ["/evidence/static_analysis/dynamic_shape_cost_contract"],
    report: "## Dynamic Shape Cost Contract", viewer: ["Overview", "Explorer"],
    method: "Replace every artifact-unknown dimension with an explicit symbol; derive exact integer tensor-payload, supported compute-MAC, and live-set polynomials without inventing numeric bounds." }),
  spec("performance.static_posture", "Static target posture, timing estimate, and triage", {
    formats: ["tflite"], keys: ["insights", "estimated_int8_speedup", "estimated_int8_speedup_detail"],
    status: (a) => emitted(a?.insights) || hasOwn(a, "estimated_int8_speedup") ? "assessed" : "not_assessed", evidenceClass: "HEURISTIC/ESTIMATED",
    pointers: ["/evidence/static_analysis/insights", "/evidence/static_analysis/estimated_int8_speedup"],
    report: "## Static Structural Triage", viewer: ["Overview", "Explorer"], method: "Apply disclosed target-profile assumptions and a separately labeled heuristic triage formula." }),
  spec("performance.audit_timing", "Browser static-audit workflow timing", {
    keys: ["static_audit_timing"], status: (a) => Number.isFinite(Number(a?.static_audit_timing?.wall_ms)) ? "assessed" : "not_assessed",
    evidenceClass: "MEASURED_BROWSER_WALL_CLOCK", pointers: ["/evidence/static_analysis/static_audit_timing"],
    report: "## Computed Analysis Coverage", viewer: ["Model", "Reports"],
    method: "Measure browser wall-clock time around the complete local audit workflow and preserve the separately measured core static-analysis interval and comparison-target count; this is analyzer workflow timing, not model inference latency." }),
  spec("artifact.size", "Artifact and constant payload breakdown", {
    keys: ["size_breakdown"], status: (a) => objectStatus(a?.size_breakdown), evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/size_breakdown"], report: "## Artifact Size Breakdown", viewer: ["Overview", "Reports"],
    method: "Count stored payload bytes and derive scalar-width projections without treating unseparated metadata as zero." }),
  spec("artifact.byte_integrity", "TFLite artifact byte ownership and conservation", {
    formats: ["tflite"], keys: ["artifact_byte_integrity"],
    status: (a) => objectStatus(a?.artifact_byte_integrity), evidenceClass: "DERIVED",
    pointers: ["/evidence/static_analysis/artifact_byte_integrity"],
    report: "## Artifact Byte Integrity Ledger", viewer: ["Overview", "Reports"],
    method: "Union verified FlatBuffer references, bind a terminal validated metadata ZIP, classify exact intervening/trailing ranges, reject overlapping ownership, and require file-size conservation." }),
  spec("tflite.sparse_storage", "TFLite sparse storage reconstruction", {
    formats: ["tflite"], keys: ["tflite_sparse_storage_contract"],
    status: (a) => objectStatus(a?.tflite_sparse_storage_contract), evidenceClass: "SOURCE_PINNED/DERIVED",
    pointers: ["/evidence/static_analysis/tflite_sparse_storage_contract"], report: "## TFLite Sparse Storage Contract", viewer: ["Overview", "Explorer", "Reports"],
    method: "Validate every serialized traversal, block map, dense level, CSR segment/index vector, and stored leaf byte count; reconstruct logical dense values through the pinned TFLite converter ordering while conserving logical = stored + implicit-zero elements." }),
  spec("tflite.subgraph_inventory", "TFLite all-subgraph and control-flow inventory", {
    formats: ["tflite"], keys: ["tflite_subgraph_inventory"], status: (a) => objectStatus(a?.tflite_subgraph_inventory), evidenceClass: "OBSERVED/SOURCE_PINNED/DERIVED",
    pointers: ["/evidence/static_analysis/tflite_subgraph_inventory"], report: "## TFLite Subgraph And Control-flow Inventory", viewer: ["Overview", "Explorer", "Reports"],
    method: "Parse every SubGraph and local tensor/operator/I/O index, decode schema-defined control-flow and StableHLO computation references, reproduce pinned IF/WHILE/CALL_ONCE Prepare-time interface checks, and derive entrypoint reachability while keeping nested serialization counts separate from runtime execution counts." }),
  spec("tflite.subgraph_deep_analysis", "TFLite independent per-subgraph deep analysis", {
    formats: ["tflite"], keys: ["tflite_subgraph_deep_analysis"], status: (a) => objectStatus(a?.tflite_subgraph_deep_analysis), evidenceClass: "OBSERVED/DERIVED/PREDICTED_SOURCE_PINNED",
    pointers: ["/evidence/static_analysis/tflite_subgraph_deep_analysis"], report: "## TFLite Per-subgraph Deep Analysis", viewer: ["Overview", "Explorer", "Reports"],
    method: "Run the same target-aware operator, quantization, source-pinned XNNPACK candidate, boundary, liveness, ArenaPlanner, movement, weight-integrity, and fixed-point proof builders independently for every serialized subgraph; retain each scope separately because serialized control flow does not encode invocation multiplicity." }),
  spec("memory.liveness", "Declared-shape live activation payload", {
    keys: ["tensor_liveness"], status: (a) => objectStatus(a?.tensor_liveness),
    pointers: ["/evidence/static_analysis/tensor_liveness"], report: "## Peak Live Activation Payload", viewer: ["Overview", "Explorer"],
    method: "Sweep producer-to-last-consumer lifetimes over statically known non-constant tensor payloads; unknown payloads produce an explicit partial lower bound, never a zero substitution." }),
  spec("weights.integrity", "Decoded constant and channel integrity", {
    keys: ["weight_integrity"], status: (a) => objectStatus(a?.weight_integrity), evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/weight_integrity"], report: "## Weight Integrity", viewer: ["Findings", "Explorer"],
    method: "Decode supported embedded constant encodings and evaluate finite values, zero slices, channel vitality indicators, and coverage." }),
  spec("weights.serialized_payload_integrity", "Serialized tensor payload numerical integrity", {
    formats: ["gguf", "safetensors"], keys: ["tensor_numerical_integrity"],
    status: (a) => objectStatus(a?.tensor_numerical_integrity), evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/tensor_numerical_integrity"], report: "## Serialized Tensor Numerical Integrity", viewer: ["Overview", "Storage", "Findings", "Reports"],
    method: "Hash every exact declared tensor range and stream-decode source-bound scalar, GGML block, SafeTensors FP8, and PyTorch-packed F4 payloads; conserve assessed plus explicitly unassessed bytes without substituting defaults, and retain F6 as unassessed unless a producer packing contract is bound." }),
  spec("weights.serialized_storage_summary", "Serialized tensor parameter and encoding ledger", {
    formats: ["gguf", "safetensors"], keys: ["tensor_storage_summary"],
    status: (a) => objectStatus(a?.tensor_storage_summary), evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/tensor_storage_summary"], report: "## Serialized Tensor Storage Ledger", viewer: ["Overview", "Storage", "Reports"],
    method: "Sum exact tensor shape cardinalities and declared bytes once, group by stored encoding, and derive effective bits per element; reuse existing full-payload SHA-256 records for content-addressed duplicate candidates without rescanning payloads." }),
  spec("weights.safetensors_packed_quantization", "SafeTensors source-bound packed-weight contract", {
    formats: ["safetensors"], keys: [],
    status: (a) => ["assessed", "fail"].includes(a?.safetensors?.quantization_contract?.status) ? "assessed" : "not_assessed",
    evidenceClass: "OBSERVED/DERIVED_FROM_PINNED_FORMAT_SOURCE",
    pointers: ["/evidence/static_analysis/safetensors/quantization_contract"],
    report: "## SafeTensors Packed-weight Quantization Contract",
    viewer: ["Overview", "Storage", "Reports"],
    method: "Bind repository quantization declarations into model-source identity; group method-specific packed tensors by module; validate source-pinned AWQ GEMM, GPTQ, HQQ encoded-state, or compressed-tensors pack-quantized dtypes, shapes, group cardinality, zero-code transforms, and logical-bit versus serialized-storage conservation. Unsupported dynamic or nonuniform schemes remain explicitly unassessed; reconstructed weights, calibration quality, runtime kernels, and task accuracy remain separate evidence." }),
  spec("llm.on_device_contract", "On-device LLM architecture, tokenizer, state, memory, and claim boundary", {
    formats: ["tflite", "onnx", "gguf", "safetensors"], keys: ["on_device_llm"],
    status: (a) => objectStatus(a?.on_device_llm), evidenceClass: "OBSERVED/SOURCE_BACKED/DERIVED/DECLARED/OBSERVED_RUNTIME",
    pointers: [
      "/evidence/static_analysis/on_device_llm",
      "/evidence/static_analysis/on_device_llm/storage/layer_storage",
      "/evidence/static_analysis/on_device_llm/storage/layer_storage/conservation",
      "/evidence/static_analysis/on_device_llm/static_memory_placement",
    ],
    report: "## On-device LLM Evidence Contract", viewer: ["Overview", "LLM", "Reports"],
    method: "For ONNX/TFLite, inventory explicit serialized transformer operators, bounded primitive motifs, external state-name candidates, and constant storage without promoting graph motifs to an LLM architecture or deriving KV dimensions. For GGUF or a hash-bound SafeTensors/config repository, normalize source-backed architecture contracts; reuse the parameter ledger; derive an exact source-registered layer versus non-layer serialized-byte ledger with byte conservation; derive dense, sparse-MoE active/total, transformer-KV, and SSM recurrent-state scenarios without estimating excluded work; add exact serialized tensor bytes to logical state for lower-bound-only capacity checks; when an artifact-bound static profile declares CPU/accelerator capacities, reserves, layer order, and non-layer/state residency, enumerate exact conditional per-pool lower-bound candidates and emit only insufficiency proofs, never fit claims; bind tokenizer, generation, deployment, and validated runtime residency/offload/paging sidecars by SHA-256 while preserving evidence classes and never equating serialized layer bytes with runtime residency or backend packing." }),
  spec("quantization.contracts", "Quantization inventory and numerical contracts", {
    keys: ["quantization_status", "quantized_tensors", "per_channel_tensors", "quantized_compute_ops", "quant_hole_count", "quant_hole_mac_impact", "quant_holes", "conv_weight_ops", "fc_ops"], status: (a) => hasOwn(a, "quantized_tensors") || emitted(a?.quantization_status) ? "assessed" : "not_assessed",
    pointers: ["/evidence/static_analysis/quantization_status", "/evidence/quantization/quantization_contract_checks"],
    report: "## Quantization Contract Checks", viewer: ["Overview", "Findings", "Explorer"],
    method: "Join tensor quantization parameters to operator semantics and verify applicable bias, zero-point, Q/DQ, accumulator, and I/O contracts." }),
  spec("findings.action_register", "Authoritative findings and actions", {
    keys: ["findings", "recommendations", "suspects"],
    status: (_a, _f, c) => Array.isArray(c?.findings) ? "assessed" : "not_assessed", evidenceClass: "DERIVED",
    pointers: ["/evidence/findings_register/findings"], report: "## Engineer Action Queue", viewer: ["Findings", "Reports"],
    method: "Normalize evidence-backed conditions into stable finding IDs, priorities, observations, actions, and JSON pointers." }),
  spec("exports.deterministic_derivatives", "Roofline CSV, core-allocation CSV, and stage graph derivatives", {
    keys: ["roofline_csv", "core_isolation_csv", "stage_mermaid", "markdown"],
    status: (a) => String(a?.roofline_csv || "").trim() || String(a?.core_isolation_csv || "").trim() || String(a?.stage_mermaid || "").trim() ? "assessed" : "not_assessed",
    pointers: ["/supplemental_sources/roofline_csv", "/supplemental_sources/core_isolation_roofline_csv", "/supplemental_sources/stage_graph_mermaid"], report: "## Computed Analysis Coverage", viewer: ["Reports", "Explorer"],
    exports: ["engineering_evidence.json", "static/roofline.csv", "static/core_isolation_roofline.csv", "static/stage_graph.mmd"], method: "Serialize analyzer-emitted per-op roofline rows, static core-allocation resource partitions, and deterministic stage graph topology." }),
  spec("tflite.core_isolation_roofline", "TFLite core-allocation roofline scenarios", {
    formats: ["tflite"], keys: ["core_isolation_analysis"],
    status: (a) => a?.core_isolation_analysis?.status || "not_assessed", evidenceClass: "ESTIMATED_STATIC_RESOURCE_PARTITION",
    pointers: ["/evidence/static_analysis/core_isolation_analysis", "/supplemental_sources/core_isolation_roofline_csv"],
    report: "## Core Allocation Roofline", viewer: ["Overview", "Roofline", "Reports"],
    exports: ["engineering_evidence.json", "static/static_analysis.json", "static/core_isolation_roofline.csv"],
    method: "Scale only the bound homogeneous-core compute ceiling by assigned/reference cores, retain the shared interface-bandwidth ceiling unchanged, and keep theoretical floors, utilization-adjusted estimates, predicted runtime overhead, and cold packing as separate conserved terms." }),

  spec("tflite.delegation", "TFLite XNNPACK delegation and boundary prediction", {
    formats: ["tflite"], keys: ["predicted_partition_boundaries", "xnnpack_assumption", "xnnpack_chains", "xnnpack_chain_breaks", "xnnpack_effective_chain_breaks", "xnnpack_structural_chain_breaks", "xnnpack_zero_mac_chain_breaks", "xnnpack_longest_chain", "delegated_macs", "fallback_macs", "delegated_mac_percent", "delegated_estimated_bytes", "fallback_estimated_bytes", "fallback_byte_percent", "fallback_traffic_by_op_family"], status: (a) => emitted(a?.predicted_partition_boundaries) ? objectStatus(a.predicted_partition_boundaries, "assessed") : Array.isArray(a?.ops) ? "assessed" : "not_assessed",
    evidenceClass: "PREDICTED/DERIVED", pointers: ["/evidence/static_analysis/predicted_partition_boundaries", "/evidence/static_analysis/xnnpack_chains"],
    report: "## Delegation Prediction Coverage", viewer: ["Overview", "Explorer", "Deployment Sensitivity Proxy"], method: "Apply the pinned delegate rule basis, build contiguous predicted domains, and derive exact graph-edge logical payloads." }),
  spec("tflite.xnnpack_candidates", "Pinned XNNPACK lowering and microkernel candidates", {
    formats: ["tflite"], keys: ["xnnpack_selector_assessment_status", "xnnpack_selector_evidence_schema", "xnnpack_selector_evidence_access", "xnnpack_selector_evidence_provenance"], status: (a) => String(a?.xnnpack_selector_assessment_status || "not_loaded") === "complete" ? "assessed" : "not_assessed",
    evidenceClass: "SOURCE_ENUMERATED_CANDIDATE", pointers: ["/evidence/static_analysis/xnnpack_selector_evidence_provenance", "/evidence/static_analysis/ops"],
    report: "## Pinned XNNPACK Kernel Candidates", viewer: ["Explorer", "Deployment Sensitivity Proxy"], method: "Filter pinned source configurations by artifact-visible facts and retain every unresolved runtime selector dimension." }),
  spec("tflite.alternate_delegate_candidates", "Pinned TFLite GPU and NNAPI source candidates", {
    formats: ["tflite"], keys: ["tflite_delegate_compatibility_evidence"], status: (a) => String(a?.tflite_delegate_compatibility_evidence?.assessment_status || "not_loaded").startsWith("assessed_") ? "assessed" : "not_assessed",
    evidenceClass: "SOURCE_PINNED/DERIVED_PARTIAL", pointers: ["/evidence/static_analysis/tflite_delegate_compatibility_evidence"],
    report: "## TFLite GPU and NNAPI Source Compatibility", viewer: ["Overview", "Explorer", "Deployment Sensitivity Proxy"], method: "Enumerate pinned TensorFlow GPU parser and NNAPI Validate registrations, apply only artifact-visible definite exclusions, and preserve selected-build, runtime/device, and assignment predicates as unresolved requirements." }),
  spec("tflite.arena", "TFLite ArenaPlanner declared-shape projection", {
    formats: ["tflite"], keys: ["tensor_arena_plan"], status: (a) => objectStatus(a?.tensor_arena_plan),
    pointers: ["/evidence/static_analysis/tensor_arena_plan"], report: "## TFLite ArenaPlanner Declared-Shape Projection", viewer: ["Overview", "Explorer"],
    method: "Reproduce the pinned ArenaPlanner/SimpleMemoryArena lifetime, alias, alignment, and greedy-offset rules over declared shapes." }),
  spec("tflite.movement_packing", "Movement, boundary payload, and packing estimates", {
    formats: ["tflite"], keys: ["movement_analysis", "conv_packing_warn_ops", "fc_packing_warn_ops"], status: (a) => objectStatus(a?.movement_analysis), evidenceClass: "DERIVED/ESTIMATED",
    pointers: ["/evidence/static_analysis/movement_analysis", "/evidence/static_analysis/predicted_partition_boundaries", "/evidence/static_analysis/ops"],
    report: "## Movement And Packing Estimates", viewer: ["Overview", "Explorer"], method: "Count explicit movement payload and graph-boundary payload exactly; apply disclosed target-profile bandwidth and setup assumptions only to packing estimates." }),
  spec("tflite.deployment_frontier", "Cross-target deployment frontier", {
    formats: ["tflite"], keys: ["deployment_frontier", "deployment_frontier_error"], status: (a) => objectStatus(a?.deployment_frontier), evidenceClass: "DERIVED/ESTIMATED",
    pointers: ["/evidence/static_analysis/deployment_frontier"], report: "## Deployment Frontier", viewer: ["Deployment Sensitivity Proxy"], method: "Re-evaluate the same op ledger across pinned target profiles and conserve normalized target-pair divergence attribution." }),
  spec("tflite.deployment_delta", "Bound prior-artifact deployment delta", {
    formats: ["tflite"], keys: ["deployment_delta", "deployment_delta_error"], status: (a) => objectStatus(a?.deployment_delta), evidenceClass: "DERIVED/ESTIMATED",
    pointers: ["/evidence/static_analysis/deployment_delta"], report: "## Deployment Delta", viewer: ["Drift Analysis", "Deployment Sensitivity Proxy"], method: "Align independently parsed artifacts and conserve signed target-profile component deltas without claiming semantic lineage." }),
  spec("tflite.delegation_repair", "Delegation repair counterfactuals", {
    formats: ["tflite"], keys: ["delegation_repair", "delegation_repair_error"], status: (a) => objectStatus(a?.delegation_repair), evidenceClass: "PREDICTED/DERIVED",
    pointers: ["/evidence/static_analysis/delegation_repair"], report: "## Delegation Repair Lab", viewer: ["Deployment Sensitivity Proxy"], method: "Toggle each op and complete predicted CPU island while rebuilding segments and boundary edges exactly." }),
  spec("tflite.quant_research_coverage", "Quantization research artifact class and lab denominator", {
    formats: ["tflite"], keys: ["quant_research_coverage"], status: (a) => objectStatus(a?.quant_research_coverage || buildQuantResearchCoverage(a)),
    evidenceClass: "DERIVED", pointers: ["/evidence/static_analysis/quant_research_coverage"],
    report: "## Quantization Research Coverage", viewer: ["Overview", "Explorer", "Reports"],
    method: "Classify the artifact quantization contract once, then apply one canonical applicability, empty-state, and scan-denominator policy to all 15 research labs.",
  }),
  ...[
    ["accumulator", "Accumulator integer headroom", "accumulator_atlas", "## Accumulator Headroom Lab", "Derive exact stored-weight, legal-input-code, and bias accumulator interval bounds per output channel."],
    ["requantization", "Fixed-point requantization fidelity", "requantization_fidelity", "## Requantization Fidelity Lab", "Encode the pinned Q0.31 multiplier/shift path and bound output-code drift over the exact post-bias interval."],
    ["kernel_witness", "Quantized kernel extremum witnesses", "kernel_extremum_witness", "## Quantized Kernel Witness Lab", "Construct sign-selected legal-code receptive-field witnesses and verify exact endpoint accumulators and fixed-point outputs."],
    ["channel_vitality", "Quantized channel vitality proof", "channel_vitality", "## Quantized Channel Vitality Atlas", "Project exact monotone interval endpoints through both pinned fixed-point build modes."],
    ["rounding_equivalence", "Fixed-point build-mode equivalence", "rounding_equivalence", "## Fixed-Point Rounding Equivalence Lab", "Partition every integer in each exact accumulator interval into maximal equal output-pair runs."],
    ["accumulator_reachability", "Accumulator reachability lattice", "accumulator_reachability", "## Accumulator Reachability Lattice", "Combine bounded weight denominations, GCD residues, and interval coverage into exact, excluded, and unresolved states."],
    ["numerical_abi", "Numerical ABI propagation", "numerical_abi_propagation", "## Numerical ABI Propagation Atlas", "Join exact local divergent states to the complete producer/tensor/consumer graph without claiming full-input reachability."],
    ["input_witness", "Model-input tensor ABI witness", "input_counterexample", "## Model Input Tensor ABI Witness", "Construct and hash a complete model-input tensor reproducing each eligible exact local counterexample."],
    ["preprocessing", "Pixel-to-tensor preprocessing realizability", "preprocessing_realizability", "## Pixel-to-Tensor Contract Lab", "Exhaustively map all source pixel codes under explicit preprocessing contracts and derive exact/minimum-error fixtures."],
    ["quantization_lattice", "Quantization lattice", "quantization_lattice", "## Quantization Lattice Lab", "Enumerate every legal 8-bit binary input-code pair and CONCATENATION input code, with residual containment candidates."],
    ["contract_migration", "Residual contract migration", "contract_migration", "## Contract Migration Impact Lab", "Regenerate direct-consumer fixed-point parameters and independently rebase stored bias for each candidate contract."],
    ["residual_step", "Residual adjacent-code step response", "residual_step_response", "## Residual Step Response Lab", "Enumerate every legal adjacent input-code transition for current and candidate residual contracts."],
    ["residual_distortion", "Residual contract distortion", "residual_contract_distortion", "## Residual Contract Distortion Atlas", "Compare current and candidate projections over every legal residual input-code pair."],
  ].map(([id, label, key, report, method]) => spec(`tflite.${id}`, label, {
    formats: ["tflite"], keys: [key], status: (a) => quantResearchMetricStatus(a, key), pointers: [`/evidence/static_analysis/${key}`],
    report, viewer: ["Explorer", "Findings"], method,
  })),
  spec("tflite.ort_non_applicability", "ORT compatibility non-applicability envelope", {
    formats: ["tflite"], keys: ["ort_compatibility_assessment_status", "ort_compatibility_evidence_schema", "ort_compatibility_evidence_access", "ort_compatibility_evidence", "ort_assignment_capture_capability", "ort_ep_portability_frontier"],
    status: () => "suppressed", evidenceClass: "NOT_APPLICABLE", pointers: [], report: "## Analysis Completeness", viewer: ["Reports"],
    method: "Preserve an explicit not-applicable protected ORT envelope without applying ONNX Runtime semantics to TFLite." }),

  spec("onnx.domains", "ONNX domain, function, and custom-op registry", {
    formats: ["onnx"], keys: ["onnx_domain_analysis", "onnx_external_data_tensor_count", "runtime_review_watchlist", "onnx_sections_suppressed"], status: (a) => objectStatus(a?.onnx_domain_analysis), evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/static_analysis/onnx_domain_analysis"], report: "## ONNX Domain and Function Registry", viewer: ["Explorer", "Findings"],
    method: "Resolve imported opsets, local functions, standard domains, and unresolved custom-domain nodes without substituting TFLite rules." }),
  spec("onnx.external_data", "All-scope ONNX external TensorProto reference and payload coverage", {
    formats: ["onnx"], keys: ["onnx_external_data"], status: (a) => objectStatus(a?.onnx_external_data), evidenceClass: "OBSERVED/NOT_ASSESSABLE",
    pointers: ["/evidence/static_analysis/onnx_external_data"], report: "## Artifact Size Breakdown", viewer: ["Findings", "Reports"],
    method: "Parse every TensorProto external_data key/value record; validate duplicate keys, relative path safety, decimal range syntax, dtype/shape byte cardinality, selected sidecar bounds, browser-computed whole-file SHA-256, and optional standard whole-file SHA-1 checksum before decoding a verified range." }),
  spec("onnx.shape_inference", "ONNX static shape-inference coverage", {
    formats: ["onnx"], keys: ["onnx_shape_inference"], status: (a) => objectStatus(a?.onnx_shape_inference),
    pointers: ["/evidence/static_analysis/onnx_shape_inference"], report: "## ONNX Shape Inference Coverage", viewer: ["Overview", "Explorer"],
    method: "Apply the pinned local rule set in graph order, recursively bind tensor-valued FunctionProto and If/Loop/Scan scopes, derive direct Sequence/Optional type, length, presence, scalar, index, split, and concat facts, and conserve dense plus non-dense output counts." }),
  spec("onnx.tensor_data_types", "Pinned ONNX TensorProto data-type contract", {
    formats: ["onnx"], keys: ["onnx_tensor_data_type_contract"], status: (a) => objectStatus(a?.onnx_tensor_data_type_contract), evidenceClass: "SOURCE_PINNED_AND_IMPLEMENTATION_TESTED",
    pointers: ["/evidence/static_analysis/onnx_tensor_data_type_contract"], report: "## ONNX TensorProto Data-Type Contract", viewer: ["Overview", "Explorer"],
    method: "Bind TensorProto DataType IDs 1-26 to the pinned ONNX 1.21 schema, compute packed 4/2-bit payload cardinality with ceiling division, and decode raw/typed numerical payloads under the declared bit layouts." }),
  spec("onnx.type_proto_contract", "Recursive ONNX TypeProto contract", {
    formats: ["onnx"], keys: ["onnx_type_proto_contract"], status: (a) => objectStatus(a?.onnx_type_proto_contract), evidenceClass: "SOURCE_PINNED_AND_OBSERVED",
    pointers: ["/evidence/static_analysis/onnx_type_proto_contract"], report: "## ONNX TypeProto Contract", viewer: ["Overview", "Explorer"],
    method: "Parse the pinned recursive TypeProto oneof and validate tensor, sparse tensor, sequence, map, optional, and opaque declarations without projecting non-dense values into tensor arithmetic." }),
  spec("onnx.sparse_tensor_contract", "ONNX SparseTensorProto structure and payload contract", {
    formats: ["onnx"], keys: ["onnx_sparse_tensor_contract"], status: (a) => objectStatus(a?.onnx_sparse_tensor_contract), evidenceClass: "SOURCE_PINNED_AND_OBSERVED",
    pointers: ["/evidence/static_analysis/onnx_sparse_tensor_contract"], report: "## ONNX SparseTensorProto Contract", viewer: ["Overview", "Explorer", "Findings"],
    method: "Validate every parsed SparseTensorProto values, indices, dense dimensions, NNZ encoding, and independently verified embedded or external TensorProto component." }),
  spec("onnx.quantization_binding", "ONNX Q/DQ and quantized-kernel binding", {
    formats: ["onnx"], keys: ["onnx_quantization_binding"], status: (a) => objectStatus(a?.onnx_quantization_binding),
    pointers: ["/evidence/static_analysis/onnx_quantization_binding", "/evidence/quantization/quantization_contract_checks"],
    report: "## Quantization Contract Checks", viewer: ["Explorer", "Findings"], method: "Bind Q/DQ parameters and quantized-kernel inputs by tensor identity, axis, dtype, and operator schema." }),
  spec("onnx.ort_source_compatibility", "Pinned ORT runtime floor and EP source compatibility", {
    formats: ["onnx"], keys: ["ort_compatibility_assessment_status", "ort_compatibility_evidence_schema", "ort_compatibility_evidence_access", "ort_compatibility_evidence", "ort_assignment_capture_capability"], status: (a) => objectStatus(a?.ort_compatibility_evidence), evidenceClass: "DERIVED_NECESSARY_MINIMUM/SOURCE_MATCH",
    pointers: ["/evidence/static_analysis/ort_compatibility_evidence", "/evidence/static_analysis/ort_assignment_capture_capability", "/evidence/static_analysis/runtime_compat"],
    report: "## Execution Provider Source Compatibility", viewer: ["Explorer", "Deployment Sensitivity Proxy"], method: "Resolve imported opsets to pinned schemas, evaluate machine-registered source type/rank/constant/output/explicit-attribute conditions against artifact facts, and remove only definite failures without inferring GetCapability assignment." }),
  spec("onnx.ep_portability", "ONNX EP portability frontier", {
    formats: ["onnx"], keys: ["ort_ep_portability_frontier"], status: (a) => objectStatus(a?.ort_ep_portability_frontier), evidenceClass: "DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS",
    pointers: ["/evidence/static_analysis/ort_ep_portability_frontier"], report: "## ONNX EP Portability Frontier", viewer: ["Deployment Sensitivity Proxy"],
    method: "Preserve source-version intersections and separately intersect the candidate sets remaining after deterministic artifact-visible definite exclusions." }),
  spec("onnx.tensorrt_static_preflight", "TensorRT configuration, parser, and optimized-engine inspector evidence", {
    formats: ["onnx"], keys: ["tensorrt_static_preflight"], status: (a) => objectStatus(a?.tensorrt_static_preflight), evidenceClass: "DERIVED_CONFIGURATION_PREFLIGHT/PARSER_OBSERVED_CONFIGURATION_BOUND",
    pointers: ["/evidence/static_analysis/tensorrt_static_preflight"], report: "## Execution Placement Evidence", viewer: ["Explorer", "Reports"],
    method: "Validate an explicit native TensorRT or ORT TensorRT EP build profile; bind min/opt/max points to ONNX dimension expressions for exact conditional MAC, logical payload, and graph-live-payload scenarios; preserve unresolved parser acceptance until an identity-bound capture exists; project observed parser subgraphs; and optionally import identity-bound optimized-engine inspector rows and selected tactic identifiers without claiming tactic timing, kernel execution, physical transfer, memory allocation, latency, or original-op assignment." }),
  spec("onnx.tflite_non_applicability", "TFLite delegate placeholder isolation", {
    formats: ["onnx"], keys: ["xnnpack_assumption", "xnnpack_chains", "xnnpack_chain_breaks", "xnnpack_effective_chain_breaks", "xnnpack_structural_chain_breaks", "xnnpack_zero_mac_chain_breaks", "delegated_mac_percent", "fallback_byte_percent", "conv_packing_warn_ops", "fc_packing_warn_ops"],
    status: () => "suppressed", evidenceClass: "NOT_APPLICABLE", pointers: [], report: "## Analysis Completeness", viewer: ["Reports"],
    method: "Keep shared-view compatibility placeholders explicitly suppressed; no TFLite delegate or packing semantics are applied to ONNX." }),
  spec("runtime.assignment", "Imported runtime assignment and boundary comparison", {
    formats: ["tflite", "onnx"],
    status: (_a, _f, c) => c?.runtimeAssignment ? objectStatus(c.runtimeAssignment) : "not_assessed", evidenceClass: "OBSERVED/DERIVED",
    pointers: ["/evidence/runtime_results/runtime_assignment", "/evidence/runtime_results/runtime_assignment_comparison"],
    report: "## Runtime Environment And Reproducibility", viewer: ["Benchmark", "Deployment Sensitivity Proxy"],
    exports: ["engineering_evidence.json", "runtime/assignment_comparison.csv", "runtime/boundary_comparison.csv"], method: "Import capture-bound runtime placement, map only identities supported by the source, and compare assessed graph relations." }),
  spec("runtime.onnx_internal_shape_binding", "Observed ONNX internal shape and runtime-bound cost closure", {
    formats: ["onnx"],
    status: (_a, _f, c) => {
      const binding = c?.runtimeShapeBinding;
      if (!binding) return "not_assessed";
      if (binding.status === "fail" || String(binding.status || "").startsWith("partial")) return "partial";
      if (String(binding.status || "").startsWith("not_applicable")) return "not_applicable";
      return "assessed";
    },
    evidenceClass: "OBSERVED_RUNTIME_INTERNAL_SHAPES/DERIVED",
    pointers: ["/evidence/runtime_results/onnx_runtime_shape_binding"],
    report: "### ONNX Runtime Internal Shape And Cost Binding",
    viewer: ["Explorer", "Reports"],
    exports: ["engineering_evidence.json", "raw-evidence/runtime_results.json"],
    method: "Bind optimization-disabled ORT input/output type-shape events to uniquely mapped original ops, require dtype/static-shape/repeat/output-size agreement, then rerun the same source-pinned ONNX MAC formulas and preserve every residual instead of substituting zero.",
  }),
  spec("runtime.backend_evidence_layers", "Selected runtime backend build, capability, assignment, and execution ledger", {
    formats: ["onnx"],
    status: (_a, _f, c) => c?.runtimeBackendLedger ? objectStatus(c.runtimeBackendLedger) : "not_assessed", evidenceClass: "OBSERVED/DERIVED_WITHOUT_CROSS_LAYER_PROMOTION",
    pointers: ["/evidence/runtime_results/runtime_backend_evidence_ledger"],
    report: "### Selected Runtime Backend Evidence Ledger", viewer: ["Explorer", "Reports"],
    exports: ["engineering_evidence.json"], method: "Normalize QNN, NNAPI, Core ML, WebGPU, and WebNN source, selected-build, capability, original-op assignment, and execution evidence as independent layers; never interpret an absent assignment as rejection." }),
  spec("runtime.data_movement", "Observed runtime copy-node payload and physical-transfer boundary", {
    formats: ["onnx"],
    status: (_a, _f, c) => c?.runtimeDataMovement
      ? c.runtimeDataMovement.status.startsWith("observed_") ? "assessed" : "partial"
      : "not_assessed",
    evidenceClass: "OBSERVED_RUNTIME_PROFILE/NOT_ASSESSED_PHYSICAL_TRANSFER",
    pointers: ["/evidence/runtime_results/runtime_data_movement_evidence"],
    report: "### Runtime Copy-Node Data Movement", viewer: ["Explorer", "Reports"],
    exports: ["engineering_evidence.json", "raw-evidence/runtime_results.json"],
    method: "Count only executed ORT MemcpyFromHost, MemcpyToHost, or Memcpy profile nodes, preserve per-event output_size and invocation cardinality, and withhold physical transfer bytes, zero-copy, synchronization, residency, and latency claims.",
  }),
  spec("runtime.environment", "Imported instrumented runtime build, scheduler graph, split, and backend-assignment binding", {
    formats: ["gguf"], status: (_a, _f, c) => c?.runtimeEnvironment ? objectStatus(c.runtimeEnvironment) : "not_assessed", evidenceClass: "BOUND_RUNTIME_ENVIRONMENT",
    pointers: ["/evidence/runtime_results/runtime_environment"],
    report: "### GGUF Runtime Build And Configuration Binding", viewer: ["Overview", "Reports"],
    exports: ["engineering_evidence.json"], method: "Fail-closed validation of artifact SHA-256, llama.cpp source revision, runtime binary SHA-256, CMake cache SHA-256, complete backend-option inventory, requested compiled backend, context/batch configuration, device identity, and collector-declared process observations without constructing per-op assignment." }),
  spec("runtime.coreml_compute_plan", "Imported Core ML compiled-model compute plan", {
    formats: ["coreml"], status: (_a, _f, c) => c?.coreMlComputePlan ? objectStatus(c.coreMlComputePlan) : "not_assessed", evidenceClass: "COREML_COMPUTE_PLAN_ESTIMATE",
    pointers: ["/evidence/runtime_results/coreml_compute_plan"], report: "### Core ML MLComputePlan Estimate", viewer: ["Overview", "Explorer", "Reports"],
    exports: ["engineering_evidence.json"], method: "Bind a macOS-produced MLComputePlan to the exact artifact, compiled-model content digest, selected function, compute-unit configuration, macOS build, hardware and available-device inventory, collector source hash, compute-plan source hash, and decoded operation order; preserve preferred/supported devices and relative-cost weights as anticipated plan evidence rather than executed placement." }),
  spec("runtime.arena_memory", "Observed TFLite arena allocation and static reconciliation", {
    formats: ["tflite"], status: (_a, _f, c) => c?.runtimeAssignment?.runtime_memory && c?.runtimeAssignment?.arena_reconciliation
      ? objectStatus(c.runtimeAssignment.arena_reconciliation)
      : "not_assessed", evidenceClass: "OBSERVED_RUNTIME/DERIVED_FROM_OBSERVED_RUNTIME",
    pointers: ["/evidence/runtime_results/runtime_assignment/runtime_memory", "/evidence/runtime_results/runtime_assignment/arena_reconciliation", "/evidence/static_analysis/tensor_arena_plan"],
    report: "### Static Projection Vs Observed TFLite Arena", viewer: ["Explorer"],
    exports: ["engineering_evidence.json", "runtime/arena_reconciliation.csv"], method: "Validate post-commit ArenaPlanner allocation snapshots and independently join the final owning allocations and aliases to the pinned declared-shape projection by tensor index." }),
  spec("runtime.browser_execution", "Browser runtime benchmark and consequence evidence", {
    status: (_a, _f, c) => c?.browserExecuted ? "assessed" : "not_assessed", evidenceClass: "MEASURED_SYNTHETIC",
    pointers: ["/evidence/runtime_results/benchmark_results", "/evidence/runtime_results/preprocessing_consequence_atlas"],
    report: "## Runtime Benchmark Evidence", viewer: ["Benchmark", "Artifact Geometry", "Drift Analysis"],
    exports: ["engineering_evidence.json", "raw-evidence/runtime_results.json"], method: "Record browser-local prepared-input executions and preserve compile, first-run, steady-state, digest, and replay scope separately." }),
  spec("validation.representative_dataset_capture", "Hash-bound representative dataset saturation, reference drift, and repeat evidence", {
    formats: ["tflite", "onnx", "gguf", "safetensors", "coreml"],
    status: (_a, _f, c) => c?.calibrationValidation ? objectStatus(c.calibrationValidation) : "not_assessed",
    evidenceClass: "DERIVED_FROM_HASH_BOUND_CAPTURED_DATASET",
    pointers: ["/evidence/runtime_results/representative_dataset_validation"],
    report: "## Representative Dataset Validation", viewer: ["Drift Analysis", "Reports"],
    exports: ["engineering_evidence.json", "raw-evidence/runtime_results.json"],
    method: "Require exact artifact and dataset-manifest binding, then independently reconstruct bounded-integer input endpoint counts, same-contract reference-output differences, repeated-run differences, and the RFC8785-JCS/SHA-256 ledger without promoting them to task-accuracy or representativeness claims." }),
  spec("validation.task_quality", "Task-level accuracy and product-validation evidence", {
    status: () => "not_assessed", evidenceClass: "NOT_ASSESSED", pointers: [],
    report: "## Static Audit Conclusion", viewer: ["Reports"], exports: ["engineering_evidence.json"],
    method: "Require a bound reference dataset, production preprocessing/postprocessing contract, acceptance metric, and executed outputs; never infer task accuracy or release readiness from artifact structure alone." }),
];

function applies(specification, format) {
  return specification.formats.includes(format);
}


function pointerExists(root, pointer) {
  let current = root;
  for (const token of String(pointer || "").split("/").slice(1)) {
    const key = token.replace(/~1/g, "/").replace(/~0/g, "~");
    if (current == null || !Object.prototype.hasOwnProperty.call(current, key)) return false;
    current = current[key];
  }
  return current !== undefined;
}

function escapePointerToken(value) {
  return String(value).replaceAll("~", "~0").replaceAll("/", "~1");
}

function fieldPath(path, key, parentIsArray = false) {
  const token = parentIsArray && /^\d+$/.test(String(key)) ? "[]" : escapePointerToken(key);
  return `${path}/${token}`;
}

function fieldValueKind(value) {
  if (value === null) return "null";
  if (ArrayBuffer.isView(value)) return value instanceof DataView ? "data_view" : `typed_array:${value.constructor.name}`;
  if (Array.isArray(value)) return value.length ? "array" : "empty_array";
  if (typeof value === "object") return Object.keys(value).length ? "object" : "empty_object";
  if (typeof value === "number" && !Number.isFinite(value)) return "non_finite_number";
  if (typeof value === "number" && Object.is(value, -0)) return "negative_zero_number";
  return typeof value;
}

function traversableFieldValue(value) {
  return value !== null && typeof value === "object" && !ArrayBuffer.isView(value);
}

export function collectAnalysisFieldPatterns(root) {
  const kindsByPath = new Map();
  const ancestors = new Set();
  let visits = 0;
  let limitExceeded = false;

  const record = (path, value) => {
    if (!path || path.startsWith("/_")) return;
    const kinds = kindsByPath.get(path) || new Set();
    kinds.add(fieldValueKind(value));
    kindsByPath.set(path, kinds);
  };
  const walk = (value, path) => {
    if (["undefined", "function", "symbol"].includes(typeof value)) return;
    visits += 1;
    if (visits > MAX_FIELD_DISCOVERY_VISITS) {
      limitExceeded = true;
      return;
    }
    if (!traversableFieldValue(value)) {
      record(path, value);
      return;
    }
    if (ancestors.has(value)) {
      record(path, "[cyclic reference]");
      return;
    }
    const keys = Object.keys(value);
    if (!keys.length) {
      record(path, value);
      return;
    }
    ancestors.add(value);
    for (const key of keys) {
      if (!path && key.startsWith("_")) continue;
      walk(value[key], fieldPath(path, key, Array.isArray(value)));
      if (limitExceeded) break;
    }
    ancestors.delete(value);
  };
  walk(root || {}, "");
  return {
    status: limitExceeded ? "incomplete_visit_limit" : "complete",
    visit_count: visits,
    visit_limit: MAX_FIELD_DISCOVERY_VISITS,
    fields: [...kindsByPath.entries()]
      .map(([path, kinds]) => ({ field_path: path, value_kinds: [...kinds].sort() }))
      .sort((a, b) => a.field_path.localeCompare(b.field_path)),
  };
}

export function createAnalysisFieldAccessTracker(root) {
  const accessed = new Set();
  const proxyByObjectAndPath = new WeakMap();

  const wrap = (value, path) => {
    if (!traversableFieldValue(value)) return value;
    let byPath = proxyByObjectAndPath.get(value);
    if (!byPath) {
      byPath = new Map();
      proxyByObjectAndPath.set(value, byPath);
    }
    if (byPath.has(path)) return byPath.get(path);
    const proxy = new Proxy(value, {
      get(target, property, receiver) {
        if (typeof property !== "string") return Reflect.get(target, property, receiver);
        const own = Object.prototype.hasOwnProperty.call(target, property);
        const nextPath = own ? fieldPath(path, property, Array.isArray(target)) : path;
        if (own && nextPath && !nextPath.startsWith("/_")) accessed.add(nextPath);
        const result = Reflect.get(target, property, receiver);
        return own ? wrap(result, nextPath) : result;
      },
    });
    byPath.set(path, proxy);
    return proxy;
  };

  return {
    analysis: wrap(root || {}, ""),
    accessedFieldPaths: () => [...accessed].sort(),
  };
}

function firstPointerToken(path) {
  return String(path || "").split("/")[1]?.replace(/~1/g, "/").replace(/~0/g, "~") || "";
}

function requiredReportFieldApplies(metricId, path, analysis) {
  if (metricId === "architecture.blocks" && String(analysis?.block_inventory?.status || "") !== "assessed") {
    return path === "/block_inventory/status";
  }
  if (metricId.startsWith("tflite.")) {
    const evidenceKey = new Map([
      ["tflite.accumulator", "accumulator_atlas"],
      ["tflite.requantization", "requantization_fidelity"],
      ["tflite.kernel_witness", "kernel_extremum_witness"],
      ["tflite.channel_vitality", "channel_vitality"],
      ["tflite.rounding_equivalence", "rounding_equivalence"],
      ["tflite.accumulator_reachability", "accumulator_reachability"],
      ["tflite.numerical_abi", "numerical_abi_propagation"],
      ["tflite.input_witness", "input_counterexample"],
      ["tflite.preprocessing", "preprocessing_realizability"],
      ["tflite.quantization_lattice", "quantization_lattice"],
      ["tflite.contract_migration", "contract_migration"],
      ["tflite.residual_step", "residual_step_response"],
      ["tflite.residual_distortion", "residual_contract_distortion"],
    ]).get(metricId);
    if (evidenceKey) {
      const coverage = analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
      const lab = (coverage?.labs || []).find((row) => row.evidence_key === evidenceKey);
      if (lab?.status === "not_applicable") return false;
    }
  }
  if (metricId === "memory.cache_payload") {
    return (analysis?.ops || []).some((op) => op?.cache_payload?.status === "assessed");
  }
  if (metricId === "artifact.size" && path === "/size_breakdown/zero_constant_byte_ratio") {
    return analysis?.size_breakdown?.metrics?.zero_constant_byte_ratio?.status === "assessed";
  }
  if (metricId === "weights.integrity") {
    if ([
      "/weight_integrity/min_grid_utilization",
      "/weight_integrity/max_saturation_percent",
    ].includes(path)) {
      return Number(analysis?.weight_integrity?.quantized_constant_tensors_scanned || 0) > 0;
    }
    if ([
      "/weight_integrity/nan_tensors",
      "/weight_integrity/inf_tensors",
      "/weight_integrity/output_channels_evaluated",
      "/weight_integrity/zero_kernel_slice_count",
      "/weight_integrity/large_magnitude_tensors",
      "/weight_integrity/high_sparsity_tensors",
    ].includes(path)) {
      return analysis?.weight_integrity?.status === "assessed";
    }
  }
  return true;
}

function buildFieldCoverage(analysis, applicableSpecs, reportAccessedFieldPaths, evidenceRoot) {
  const discovered = collectAnalysisFieldPatterns(analysis);
  const reportAccessed = new Set(reportAccessedFieldPaths || []);
  const staticAnalysis = evidenceRoot?.evidence?.static_analysis || null;
  const staticPatterns = staticAnalysis ? collectAnalysisFieldPatterns(staticAnalysis) : null;
  const staticPatternSet = new Set((staticPatterns?.fields || []).map((item) => item.field_path));
  const quantResearchCoverage = String(analysis?.format || "tflite").toLowerCase() === "onnx"
    ? null
    : analysis?.quant_research_coverage || buildQuantResearchCoverage(analysis);
  const classExcludedEvidenceKeys = new Set((quantResearchCoverage?.labs || [])
    .filter((row) => !row.class_supported && row.evidence_key)
    .map((row) => row.evidence_key));
  const metricByKey = new Map();
  for (const item of applicableSpecs) {
    for (const key of item.keys) {
      const rows = metricByKey.get(key) || [];
      rows.push(item.id);
      metricByKey.set(key, rows);
    }
  }
  const ledger = discovered.fields.map((field) => {
    const key = firstPointerToken(field.field_path);
    const specialRoute = SPECIAL_FIELD_EXPORT_ROUTES.get(key) || "";
    const rawBinding = staticAnalysis
      ? staticPatternSet.has(field.field_path) ? "static_analysis_json_path"
        : classExcludedEvidenceKeys.has(key) ? "suppressed_not_applicable"
        : specialRoute ? specialRoute : "unbound"
      : specialRoute || "not_checked";
    return {
      ...field,
      metric_ids: (() => {
        const override = FIELD_METRIC_PREFIX_OVERRIDES.find(([prefix]) => field.field_path.startsWith(prefix));
        return override ? [override[1]] : metricByKey.get(key) || [];
      })(),
      engineering_report_access: reportAccessed.has(field.field_path) ? "consumed" : "not_consumed",
      raw_evidence_binding: rawBinding,
    };
  });
  const reportConsumed = ledger.filter((item) => item.engineering_report_access === "consumed");
  const applicabilitySuppressed = ledger.filter((item) => item.raw_evidence_binding === "suppressed_not_applicable");
  const rawOnly = ledger.filter((item) => item.engineering_report_access !== "consumed"
    && !["unbound", "suppressed_not_applicable"].includes(item.raw_evidence_binding));
  const unbound = ledger.filter((item) => item.raw_evidence_binding === "unbound");
  const nonJsonSafe = ledger.filter((item) => item.raw_evidence_binding !== "suppressed_not_applicable"
    && item.value_kinds.some((kind) => kind === "non_finite_number" || kind === "negative_zero_number" || kind === "bigint" || kind === "data_view" || kind.startsWith("typed_array:")));
  const discoveredFieldPaths = new Set(ledger.map((item) => item.field_path));
  const metricRows = applicableSpecs.map((item) => {
    const rows = ledger.filter((field) => field.metric_ids.includes(item.id));
    const rawOnlyRows = rows.filter((field) => field.engineering_report_access !== "consumed"
      && !["unbound", "suppressed_not_applicable"].includes(field.raw_evidence_binding));
    const applicabilitySuppressedRows = rows.filter((field) => field.raw_evidence_binding === "suppressed_not_applicable");
    const unboundRows = rows.filter((field) => field.raw_evidence_binding === "unbound");
    const requiredRows = (REQUIRED_REPORT_FIELD_PATTERNS.get(item.id) || [])
      .filter((path) => discoveredFieldPaths.has(path) && requiredReportFieldApplies(item.id, path, analysis));
    const missingRequiredRows = requiredRows.filter((path) => !reportAccessed.has(path));
    return {
      metric_id: item.id,
      leaf_field_pattern_count: rows.length,
      report_consumed_field_pattern_count: rows.filter((field) => field.engineering_report_access === "consumed").length,
      required_report_field_pattern_count: requiredRows.length,
      required_report_field_consumed_count: requiredRows.length - missingRequiredRows.length,
      missing_required_report_field_count: missingRequiredRows.length,
      missing_required_report_field_paths: missingRequiredRows,
      raw_evidence_only_field_pattern_count: rawOnlyRows.length,
      raw_evidence_only_field_path_examples: rawOnlyRows.slice(0, 3).map((field) => field.field_path),
      applicability_suppressed_field_pattern_count: applicabilitySuppressedRows.length,
      applicability_suppressed_field_path_examples: applicabilitySuppressedRows.slice(0, 3).map((field) => field.field_path),
      unbound_field_pattern_count: unboundRows.length,
      unbound_field_path_examples: unboundRows.slice(0, 3).map((field) => field.field_path),
    };
  });
  const requiredReportFields = [...new Set(metricRows.flatMap((row) => (REQUIRED_REPORT_FIELD_PATTERNS.get(row.metric_id) || [])
    .filter((path) => discoveredFieldPaths.has(path) && requiredReportFieldApplies(row.metric_id, path, analysis))))].sort();
  const missingRequiredReportFields = metricRows.flatMap((row) => row.missing_required_report_field_paths);
  return {
    discovery_status: discovered.status,
    discovery_visit_count: discovered.visit_count,
    discovery_visit_limit: discovered.visit_limit,
    static_export_discovery_status: staticPatterns?.status || "not_checked",
    leaf_field_pattern_count: ledger.length,
    report_consumed_field_pattern_count: reportConsumed.length,
    required_report_field_pattern_count: requiredReportFields.length,
    required_report_field_consumed_count: requiredReportFields.length - missingRequiredReportFields.length,
    required_report_field_paths: requiredReportFields,
    missing_required_report_field_count: missingRequiredReportFields.length,
    missing_required_report_field_paths: missingRequiredReportFields,
    raw_evidence_only_field_pattern_count: rawOnly.length,
    applicability_suppressed_field_pattern_count: applicabilitySuppressed.length,
    applicability_suppressed_field_paths: applicabilitySuppressed.map((item) => item.field_path),
    unbound_field_pattern_count: unbound.length,
    unbound_field_paths: unbound.map((item) => item.field_path),
    non_json_safe_field_pattern_count: nonJsonSafe.length,
    non_json_safe_field_paths: nonJsonSafe.map((item) => item.field_path),
    metric_family_ledger: metricRows,
    field_ledger: ledger,
    interpretation_boundary: "A report-consumed field was read by the Engineering Report generator while constructing this report. Decision-critical emitted summaries are additionally registered as required report fields and fail conformance when not consumed. Class-excluded quant-research detail is intentionally suppressed and bound to the shared applicability ledger; it is neither raw evidence nor an unbound field. Large applicable proof ledgers may remain raw-evidence-only. Any unbound, missing-required, or applicable non-JSON-safe field fails conformance.",
  };
}

export function buildMetricCoverageEntries(analysis, {
  findings = null,
  runtimeResults = null,
  evidenceRoot = null,
} = {}) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const context = {
    findings,
    runtimeAssignment: runtimeResults?.runtime_assignment || null,
    runtimeEnvironment: runtimeResults?.runtime_environment || null,
    coreMlComputePlan: runtimeResults?.coreml_compute_plan || null,
    runtimeBackendLedger: runtimeResults?.runtime_backend_evidence_ledger || null,
    runtimeDataMovement: runtimeResults?.runtime_data_movement_evidence || null,
    runtimeShapeBinding: runtimeResults?.onnx_runtime_shape_binding || null,
    calibrationValidation: runtimeResults?.representative_dataset_validation || null,
    browserExecuted: runtimeResults?.browser_execution_status === "executed"
      || Boolean(runtimeResults?.benchmark_results?.length || runtimeResults?.runtime_basin || runtimeResults?.preprocessing_consequence_atlas),
    runtimeExecuted: runtimeResults?.runtime_execution_status === "executed",
  };
  return SPECS.map((item) => {
    const applicable = applies(item, format);
    const status = applicable ? item.status(analysis || {}, format, context) : "not_applicable";
    const pointers = applicable ? item.pointers : [];
    return {
      metric_id: item.id,
      label: item.label,
      applicability: applicable ? format : "not_applicable_for_format",
      status,
      evidence_class: status === "not_assessed" || status === "not_applicable" ? "NOT_ASSESSED" : item.evidenceClass,
      calculation_method: item.method,
      analysis_object_keys: item.keys,
      evidence_json_pointers: pointers,
      report_section: typeof item.report === "function" ? item.report(format) : item.report,
      viewer_tabs: item.viewer,
      export_artifacts: item.exports,
      pointer_binding_status: evidenceRoot && ["assessed", "partial"].includes(status)
        ? pointers.some((pointer) => pointerExists(evidenceRoot, pointer)) ? "bound" : "unbound"
        : evidenceRoot ? "not_required" : "not_checked",
    };
  });
}

export function buildMetricCoverageManifest(analysis, {
  findings = null,
  runtimeResults = null,
  evidenceRoot = null,
  reportAccessedFieldPaths = [],
} = {}) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const entries = buildMetricCoverageEntries(analysis, { findings, runtimeResults, evidenceRoot });
  const applicableSpecs = SPECS.filter((item) => applies(item, format));
  const registeredKeys = new Set(applicableSpecs.flatMap((item) => item.keys));
  const analysisKeys = Object.keys(analysis || {}).filter((key) => !key.startsWith("_")).sort();
  const analysisKeyLedger = analysisKeys.map((key) => ({
    analysis_object_key: key,
    value_kind: Array.isArray(analysis[key]) ? "array" : analysis[key] === null ? "null" : typeof analysis[key],
    metric_ids: applicableSpecs.filter((item) => item.keys.includes(key)).map((item) => item.id),
  }));
  const unregisteredAnalysisKeys = analysisKeyLedger.filter((item) => item.metric_ids.length === 0).map((item) => item.analysis_object_key);
  const multiplyRegisteredAnalysisKeys = analysisKeyLedger.filter((item) => item.metric_ids.length > 1).map((item) => item.analysis_object_key);
  const discoveredKeys = Object.entries(analysis || {})
    .filter(([key, value]) => value && typeof value === "object"
      && !Array.isArray(value)
      && (COMPUTATION_OBJECT_KEYS.has(key) || hasOwn(value, "schema") || hasOwn(value, "evidence_class") || hasOwn(value, "method_version")))
    .map(([key]) => key)
    .sort();
  const unregistered = discoveredKeys.filter((key) => !registeredKeys.has(key));
  const counts = Object.fromEntries(["assessed", "partial", "not_assessed", "not_applicable", "suppressed"].map((status) => [status, entries.filter((entry) => entry.status === status).length]));
  const unbound = entries.filter((entry) => entry.pointer_binding_status === "unbound").map((entry) => entry.metric_id);
  const fieldCoverage = buildFieldCoverage(analysis || {}, applicableSpecs, reportAccessedFieldPaths, evidenceRoot);
  const decisionCoverage = buildDecisionCoverageLedger(entries, format);
  return {
    schema: SCHEMA,
    format,
    registry_metric_count: entries.length,
    applicable_metric_count: entries.filter((entry) => entry.applicability === format).length,
    status_counts: counts,
    discovered_computation_object_keys: discoveredKeys,
    unregistered_computation_object_keys: unregistered,
    analysis_object_key_count: analysisKeys.length,
    registered_analysis_object_key_count: analysisKeys.length - unregisteredAnalysisKeys.length,
    analysis_object_key_ledger: analysisKeyLedger,
    unregistered_analysis_object_keys: unregisteredAnalysisKeys,
    multiply_registered_analysis_object_keys: multiplyRegisteredAnalysisKeys,
    unbound_assessed_metric_ids: unbound,
    decision_coverage: decisionCoverage,
    field_coverage: fieldCoverage,
    coverage_status: unregistered.length || unregisteredAnalysisKeys.length || multiplyRegisteredAnalysisKeys.length || unbound.length
      || decisionCoverage.status === "fail"
      || fieldCoverage.discovery_status !== "complete" || fieldCoverage.static_export_discovery_status === "incomplete_visit_limit"
      || fieldCoverage.unbound_field_pattern_count || fieldCoverage.missing_required_report_field_count || fieldCoverage.non_json_safe_field_pattern_count
      ? "fail" : counts.partial || counts.not_assessed ? "partial" : "complete",
    interpretation_boundary: "Every non-private top-level analysis key is owned by exactly one declared metric family. Nested leaf patterns are separately classified as consumed by the Engineering Report generator, preserved only in raw evidence, or unbound. NOT_ASSESSED is never equivalent to numeric zero. Runtime placement, executed kernels, device latency, and task accuracy require corresponding runtime or dataset evidence.",
    entries,
  };
}

export function metricCoverageMarkdown(manifest) {
  const fieldCoverage = manifest?.field_coverage || {};
  const formatApplicableEntries = (manifest?.entries || []).filter((entry) => entry.applicability === manifest?.format);
  const formatApplicableCounts = Object.fromEntries(
    ["assessed", "partial", "not_assessed", "not_applicable", "suppressed"]
      .map((status) => [status, formatApplicableEntries.filter((entry) => entry.status === status).length]),
  );
  const formatInapplicableCount = (manifest?.entries || []).filter((entry) => entry.applicability !== manifest?.format).length;
  const rows = (manifest?.entries || []).map((entry) => [
    entry.metric_id,
    entry.status,
    entry.evidence_class,
    entry.report_section.replace(/^#{1,6}\s+/, ""),
    entry.viewer_tabs.join(" / ") || "raw export only",
    entry.calculation_method,
  ]);
  return [
    "## Metric Coverage Manifest (DERIVED)",
    `Schema \`${manifest?.schema || SCHEMA}\`; ${manifest?.applicable_metric_count || 0} format-applicable metric families = ${formatApplicableCounts.assessed || 0} assessed + ${formatApplicableCounts.partial || 0} partial + ${formatApplicableCounts.not_assessed || 0} not assessed + ${formatApplicableCounts.not_applicable || 0} not applicable to this artifact + ${formatApplicableCounts.suppressed || 0} suppressed. Registered format-inapplicable families: ${formatInapplicableCount}. Registry conservation: ${manifest?.registry_metric_count || 0} = ${manifest?.applicable_metric_count || 0} format-applicable + ${formatInapplicableCount} format-inapplicable.`,
    markdownTable(["Metric family", "Status", "Evidence", "Report binding", "Viewer", "Calculation basis"], rows),
    `Structured computation discovery: ${(manifest?.discovered_computation_object_keys || []).length} object(s); unregistered ${manifest?.unregistered_computation_object_keys?.length || 0}; assessed pointer bindings are verified in \`engineering_evidence.json\`.`,
    `Top-level analysis-key ownership: ${manifest?.registered_analysis_object_key_count || 0}/${manifest?.analysis_object_key_count || 0} mapped to exactly one metric family; unregistered ${(manifest?.unregistered_analysis_object_keys || []).join(" / ") || "none"}; multiply registered ${(manifest?.multiply_registered_analysis_object_keys || []).join(" / ") || "none"}. Full key ledger is embedded in \`engineering_evidence.json\` and \`raw-evidence/metric_coverage.json\`.`,
    `Nested field coverage: ${fieldCoverage.leaf_field_pattern_count || 0} normalized leaf pattern(s); ${fieldCoverage.report_consumed_field_pattern_count || 0} consumed while constructing this Engineering Report; required decision fields ${fieldCoverage.required_report_field_consumed_count || 0}/${fieldCoverage.required_report_field_pattern_count || 0} consumed, missing ${fieldCoverage.missing_required_report_field_count || 0}; ${fieldCoverage.raw_evidence_only_field_pattern_count || 0} retained only in machine-readable evidence; ${fieldCoverage.applicability_suppressed_field_pattern_count || 0} class-excluded and intentionally suppressed; ${fieldCoverage.unbound_field_pattern_count || 0} unbound; ${fieldCoverage.non_json_safe_field_pattern_count || 0} applicable non-JSON-safe. Full field ledger and export route are embedded in \`engineering_evidence.json\` and \`raw-evidence/metric_coverage.json\`.`,
    markdownTable(["Metric family", "Leaf patterns", "Report-consumed", "Required consumed", "Required missing", "Raw-only", "Class-suppressed", "Raw-only examples", "Unbound"], (fieldCoverage.metric_family_ledger || [])
      .filter((row) => row.leaf_field_pattern_count > 0)
      .map((row) => [row.metric_id, row.leaf_field_pattern_count, row.report_consumed_field_pattern_count, `${row.required_report_field_consumed_count}/${row.required_report_field_pattern_count}`, row.missing_required_report_field_count, row.raw_evidence_only_field_pattern_count, row.applicability_suppressed_field_pattern_count || 0, (row.raw_evidence_only_field_path_examples || []).join(" / ") || "none", row.unbound_field_pattern_count])),
    `> ${fieldCoverage.interpretation_boundary || "Field-level coverage was not computed."}`,
    `> ${manifest?.interpretation_boundary || ""}`,
  ].join("\n\n");
}


export function validateMetricCoverageManifest(manifest, engineeringReport = "") {
  const entries = manifest?.entries || [];
  const ids = entries.map((entry) => entry.metric_id);
  const failures = [];
  if (manifest?.schema !== SCHEMA) failures.push("metric coverage schema mismatch");
  if (new Set(ids).size !== ids.length) failures.push("duplicate metric IDs");
  if (Number(manifest?.registry_metric_count || 0) !== entries.length) failures.push("registry metric count mismatch");
  const applicableCount = entries.filter((entry) => entry.applicability === manifest?.format).length;
  if (Number(manifest?.applicable_metric_count || 0) !== applicableCount) failures.push("applicable metric count mismatch");
  const expectedStatusCounts = Object.fromEntries(["assessed", "partial", "not_assessed", "not_applicable", "suppressed"].map((status) => [status, entries.filter((entry) => entry.status === status).length]));
  if (JSON.stringify(manifest?.status_counts || {}) !== JSON.stringify(expectedStatusCounts)) failures.push("metric status-count conservation mismatch");
  if (entries.some((entry) => !["assessed", "partial", "not_assessed", "not_applicable", "suppressed"].includes(entry.status))) failures.push("invalid metric status");
  const expectedDecisionCoverage = buildDecisionCoverageLedger(entries, String(manifest?.format || "tflite"));
  if (JSON.stringify(manifest?.decision_coverage || null) !== JSON.stringify(expectedDecisionCoverage)) failures.push("decision coverage does not reconstruct from metric entries");
  if (expectedDecisionCoverage.status === "fail") failures.push(`decision coverage assignment failed: ${[...expectedDecisionCoverage.unassigned_metric_ids, ...expectedDecisionCoverage.multiply_assigned_metric_ids].join(", ")}`);
  if (!String(engineeringReport || "").includes(decisionCoverageMarkdown(manifest))) failures.push("Engineering Report decision-coverage summary is missing or stale");
  if ((manifest?.unregistered_computation_object_keys || []).length) failures.push(`unregistered computation objects: ${manifest.unregistered_computation_object_keys.join(", ")}`);
  if ((manifest?.unregistered_analysis_object_keys || []).length) failures.push(`unregistered analysis keys: ${manifest.unregistered_analysis_object_keys.join(", ")}`);
  if ((manifest?.multiply_registered_analysis_object_keys || []).length) failures.push(`multiply registered analysis keys: ${manifest.multiply_registered_analysis_object_keys.join(", ")}`);
  if ((manifest?.unbound_assessed_metric_ids || []).length) failures.push(`unbound assessed metrics: ${manifest.unbound_assessed_metric_ids.join(", ")}`);
  const fieldCoverage = manifest?.field_coverage || {};
  if (fieldCoverage.discovery_status !== "complete") failures.push(`analysis field discovery is ${fieldCoverage.discovery_status || "missing"}`);
  if (fieldCoverage.static_export_discovery_status !== "complete") failures.push(`static export field discovery is ${fieldCoverage.static_export_discovery_status || "missing"}`);
  const fieldLedger = fieldCoverage.field_ledger || [];
  if (fieldLedger.length !== Number(fieldCoverage.leaf_field_pattern_count || 0)) failures.push("field coverage ledger count mismatch");
  if (new Set(fieldLedger.map((item) => item.field_path)).size !== fieldLedger.length) failures.push("duplicate field coverage paths");
  const reportConsumedCount = fieldLedger.filter((item) => item.engineering_report_access === "consumed").length;
  const rawOnlyCount = fieldLedger.filter((item) => item.engineering_report_access !== "consumed"
    && !["unbound", "suppressed_not_applicable"].includes(item.raw_evidence_binding)).length;
  const applicabilitySuppressedCount = fieldLedger.filter((item) => item.raw_evidence_binding === "suppressed_not_applicable").length;
  if (reportConsumedCount !== Number(fieldCoverage.report_consumed_field_pattern_count || 0)) failures.push("report-consumed field count mismatch");
  if (rawOnlyCount !== Number(fieldCoverage.raw_evidence_only_field_pattern_count || 0)) failures.push("raw-only field count mismatch");
  if (applicabilitySuppressedCount !== Number(fieldCoverage.applicability_suppressed_field_pattern_count || 0)) failures.push("applicability-suppressed field count mismatch");
  if (fieldLedger.length && reportConsumedCount === 0) failures.push("Engineering Report consumed no analysis leaf fields");
  if (fieldLedger.some((item) => !Array.isArray(item.metric_ids) || item.metric_ids.length !== 1)) failures.push("field paths must belong to exactly one metric family");
  if (Number(fieldCoverage.unbound_field_pattern_count || 0) > 0) failures.push(`unbound analysis fields: ${(fieldCoverage.unbound_field_paths || []).slice(0, 8).join(", ")}`);
  const missingRequired = fieldCoverage.missing_required_report_field_paths || [];
  const requiredPaths = fieldCoverage.required_report_field_paths || [];
  if (requiredPaths.length !== Number(fieldCoverage.required_report_field_pattern_count || 0)) failures.push("required report field path count mismatch");
  if (new Set(requiredPaths).size !== requiredPaths.length) failures.push("duplicate required report field paths");
  if (missingRequired.length !== Number(fieldCoverage.missing_required_report_field_count || 0)) failures.push("missing required report field count mismatch");
  if (Number(fieldCoverage.required_report_field_consumed_count || 0) + Number(fieldCoverage.missing_required_report_field_count || 0) !== Number(fieldCoverage.required_report_field_pattern_count || 0)) failures.push("required report field arithmetic mismatch");
  if (missingRequired.length) failures.push(`required report fields not consumed: ${missingRequired.slice(0, 8).join(", ")}`);
  if (Number(fieldCoverage.non_json_safe_field_pattern_count || 0) > 0) failures.push(`non-JSON-safe analysis fields: ${(fieldCoverage.non_json_safe_field_paths || []).slice(0, 8).join(", ")}`);
  for (const entry of entries.filter((item) => ["assessed", "partial"].includes(item.status))) {
    if (!entry.evidence_json_pointers.length) failures.push(`${entry.metric_id} has no evidence pointer`);
    if (!entry.report_section || !String(engineeringReport).includes(entry.report_section)) failures.push(`${entry.metric_id} has no rendered report binding`);
    if (!entry.viewer_tabs.length && !entry.export_artifacts.length) failures.push(`${entry.metric_id} has no viewer or export binding`);
  }
  return failures;
}
