import { formatBytes, formatNumber, formatPercent, padOp } from "./format.js";
import { code, markdownTable } from "./report-utils.js";

function isOnnxAnalysis(analysis) {
  return String(analysis?.format || "").toLowerCase() === "onnx";
}
export function onnxShapeInferenceMarkdown(analysis) {
  if (!isOnnxAnalysis(analysis)) return "";
  const shape = analysis?.onnx_shape_inference;
  if (!shape) return "## ONNX Shape Inference Coverage (NOT_ASSESSED)\nNo shape-inference ledger was emitted.";
  const unsupported = (shape.rule_unsupported_op_histogram || [])
    .map((item) => `${item.name}:${formatNumber(item.count)}`).join(" / ") || "none";
  const unresolvedRows = (shape.rule_unresolved_nodes || []).slice(0, 48).map((row) => [
    `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`,
    row.reason || "not emitted",
  ]);
  const conflictRows = (shape.declaration_conflicts || []).slice(0, 48).map((row) => [
    `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`,
    code(row.tensor_name || "unnamed"),
    row.field || "contract",
    String(row.declared),
    String(row.inferred),
  ]);
  const semanticConflictRows = (shape.semantic_contract_conflicts || []).slice(0, 48).map((row) => [
    `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`,
    (row.output_names || []).map(code).join(" / ") || "unnamed",
    row.reason || "operator_contract_invalid",
    row.details == null ? "none" : code(JSON.stringify(row.details)),
  ]);
  const sourceDocuments = (shape.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const opsetContract = shape.opset_import_contract || {};
  const invalidOpsetRows = (opsetContract.rows || []).filter((row) => row.status !== "pass");
  const displayedInvalidOpsetRows = invalidOpsetRows.slice(0, 48);
  const schemaRows = (shape.schema_form_rows || []).filter((row) => row.status !== "pass");
  const displayedSchemaRows = schemaRows.slice(0, 48).map((row) => [
    `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`,
    row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
    row.schema_since_version == null ? "none" : `${row.op_name}-${row.schema_since_version}`,
    row.status || "unresolved",
    (row.reason_codes || []).join(" / ") || row.detail || "not emitted",
  ]);
  const scope = shape.shape_scope || {};
  const extended = shape.extended_scope_inference || {};
  const container = shape.container_value_inference || {};
  const tfidf = shape.tfidf_vectorizer_inference || {};
  const mlValue = shape.ml_value_inference || {};
  const staticSignedZeroTensors = (analysis?.tensors || []).filter((tensor) => Number(tensor.static_values_negative_zero_count || 0) > 0);
  const staticSignedZeroValues = staticSignedZeroTensors.reduce((sum, tensor) => sum + Number(tensor.static_values_negative_zero_count || 0), 0);
  const staticSignedZeroDetails = staticSignedZeroTensors.slice(0, 16).map((tensor) => {
    const indices = tensor.static_values_negative_zero_indices || [];
    return `${code(tensor.name || `#${tensor.index}`)} [${indices.slice(0, 16).map((index) => formatNumber(index)).join(", ")}${indices.length > 16 ? ", ..." : ""}]`;
  }).join(" / ") || "none";
  const staticCanonicalTextTensors = (analysis?.tensors || []).filter((tensor) => tensor.static_values_canonical_text_complete === true);
  const staticCanonicalTextValues = staticCanonicalTextTensors.reduce((sum, tensor) => sum + (tensor.static_values_canonical_texts || []).length, 0);
  const staticCanonicalTextDetails = staticCanonicalTextTensors.slice(0, 16).map((tensor) => {
    const values = tensor.static_values_canonical_texts || [];
    return `${code(tensor.name || `#${tensor.index}`)} [${values.slice(0, 16).map(code).join(" / ")}${values.length > 16 ? " / ..." : ""}]`;
  }).join(" / ") || "none";
  const extendedSources = (extended.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const exactCostValue = (complete, assessed, completeMirror, assessedMirror) => {
    if (complete != null) return `${complete} complete`;
    if (assessed != null) return `${assessed} assessed subtotal${assessedMirror == null && completeMirror == null ? "; safe-number mirror withheld" : ""}`;
    return "not assessed";
  };
  const intrinsicCostSummary = (cost) => {
    if (!cost) return "not assessed";
    return `${cost.status || "not assessed"}; MACs ${exactCostValue(cost.complete_nominal_macs_decimal, cost.assessed_nominal_macs_decimal, cost.complete_nominal_macs, cost.assessed_nominal_macs)}; logical op I/O ${exactCostValue(cost.complete_operator_io_payload_bytes_decimal, cost.assessed_operator_io_payload_bytes_decimal, cost.complete_operator_io_payload_bytes, cost.assessed_operator_io_payload_bytes)} B; ${formatNumber(cost.assessed_nominal_mac_operator_count || 0)}/${formatNumber(cost.mac_compute_operator_count || 0)} MAC-bearing op(s), ${formatNumber(cost.unassessed_nominal_mac_operator_count || 0)} residual; ${formatNumber(cost.assessed_operator_io_count || 0)}/${formatNumber(cost.operator_count || 0)} payload-assessed op(s), ${formatNumber(cost.unassessed_operator_io_count || 0)} residual`;
  };
  const mainIntrinsicCost = extended.main_graph_intrinsic_cost || {};
  const mainIntrinsicCostSummary = `${code(mainIntrinsicCost.schema || "not emitted")}; ${mainIntrinsicCost.evidence_class || "not emitted"}; source ${mainIntrinsicCost.source_release || "not emitted"} / ${code(mainIntrinsicCost.source_commit || "not emitted")}; ${intrinsicCostSummary(mainIntrinsicCost)}`;
  const intrinsicCostSources = (mainIntrinsicCost.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const intrinsicCostResidualRows = [
    ...(mainIntrinsicCost.mac_residuals || []).map((row) => ["MAC", `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`, row.reason || "not emitted"]),
    ...(mainIntrinsicCost.payload_residuals || []).map((row) => ["Logical op I/O", `#${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`, row.reason || "not emitted"]),
  ].slice(0, 48);
  const scopeRows = (scope.scope_execution_rows || []).slice(0, 48).map((row) => [
    row.scope_class || "unresolved",
    code(row.scope || "not emitted"),
    row.status || "not assessed",
    formatNumber(row.execution_count || 0),
    `${formatNumber(row.assessed_node_count || 0)} / ${formatNumber(row.node_count || 0)}`,
    formatNumber(row.unresolved_output_count || 0),
  ]);
  const recursiveScopeRows = (extended.scope_rows || []).slice(0, 48).map((row) => [
    row.scope_class || "unresolved",
    code(row.scope || "not emitted"),
    row.status || "not assessed",
    formatNumber(row.execution_count || 0),
    `${formatNumber(row.assessed_node_count || 0)} / ${formatNumber(row.node_count || 0)}`,
    formatNumber(row.unassessed_node_count || 0),
    formatNumber(row.unresolved_output_count || 0),
    `${formatNumber(row.intrinsic_cost_variant_count || 0)} retained / ${formatNumber(row.intrinsic_cost_variant_overflow_count || 0)} overflow / ${formatNumber(row.intrinsic_cost_unassessed_execution_count || 0)} unassessed; ${(row.intrinsic_cost_variants || []).map((cost, index) => `v${index + 1}: ${formatNumber(cost.observation_count || 0)} observation(s); ${intrinsicCostSummary(cost)}`).join(" / ") || "no retained variant"}`,
    (row.reason_codes || []).join(" / ") || "none",
  ]);
  const exclusionRows = (scope.exclusions || []).slice(0, 48).map((row) => [
    row.scope_class || "unresolved",
    code(row.scope || "not emitted"),
    formatNumber(row.node_count || 0),
    [row.reason_code, ...(row.reason_codes || [])].filter(Boolean).join(" / ") || "not emitted",
  ]);
  const functionRows = (extended.function_call_rows || []).slice(0, 48).map((row) => [
    code(row.function_id || "not emitted"),
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
    row.status || "not assessed",
    `${formatNumber(row.input_count || 0)} / ${formatNumber(row.output_count || 0)}`,
    (row.reason_codes || []).join(" / ") || row.body_status || "pass",
  ]);
  const controlRows = (extended.control_flow_rows || []).slice(0, 48).map((row) => [
    row.op_name || "UNKNOWN",
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
    row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
    row.status || "not assessed",
    row.op_name === "Loop"
      ? `${formatNumber(row.state_variable_count || 0)} state(s), ${formatNumber(row.non_dense_state_variable_count || 0)} non-dense [${(row.state_value_kinds || []).join(" / ") || "none"}], ${formatNumber(row.scan_output_count || 0)} scan output(s)`
      : row.state_variable_count == null ? "N/A" : `${formatNumber(row.state_variable_count || 0)} state(s), ${formatNumber(row.scan_output_count || 0)} scan output(s)`,
    row.op_name === "Loop"
      ? `${row.exact_expansion_status || "not assessed"}; ${row.exact_iteration_count == null ? "runtime-dependent" : `${formatNumber(row.exact_iteration_count)} iteration(s)`}; ${formatNumber(row.body_node_count || 0)} body node(s), ${formatNumber(row.exact_body_node_evaluation_count || 0)} body-node evaluation(s)`
      : "N/A",
    (row.reason_codes || []).join(" / ") || row.body_status || (row.branch_statuses || []).join(" / ") || "pass",
  ]);
  const allLoopStateRows = (extended.control_flow_rows || []).filter((row) => row.op_name === "Loop").flatMap((row) => [
    ...(row.exact_iteration_state_contracts || []).flatMap((iteration) => (iteration.states || []).map((state) => [
      `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
      `iteration ${formatNumber(iteration.iteration)}`,
      `state ${formatNumber(state.state_index)}`,
      state.value_kind || "unresolved",
      state.dtype || "UNKNOWN",
      state.shape_declared ? `[${(state.shape || []).join(", ")}]` : "rank not declared",
      state.sequence_length_status === "assessed_exact" ? `length ${formatNumber(state.sequence_length)}; ${(state.sequence_element_types || []).join(" / ") || "empty inventory"}` : "N/A",
    ])),
    ...(row.exact_final_state_contracts || []).map((state) => [
      `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
      "final",
      `state ${formatNumber(state.state_index)} -> ${code(state.output_name || "unnamed")}`,
      state.value_kind || "unresolved",
      state.dtype || "UNKNOWN",
      state.shape_declared ? `[${(state.shape || []).join(", ")}]` : "rank not declared",
      state.sequence_length_status === "assessed_exact" ? `length ${formatNumber(state.sequence_length)}; ${(state.sequence_element_types || []).join(" / ") || "empty inventory"}` : "N/A",
    ]),
  ]);
  const loopStateRows = allLoopStateRows.slice(0, 96);
  const allLoopFailureRows = (extended.control_flow_rows || []).filter((row) => row.op_name === "Loop").flatMap((row) => (
    row.exact_nested_failure_rows || []
  ).map((failure) => [
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
    `${code(failure.scope || "nested")} / #${padOp(failure.node_index)}`,
    failure.op_name || "UNKNOWN",
    (failure.reason_codes || []).join(" / ") || "failed without a reason code",
    (failure.scope_failure_details || []).map((detail) => `${detail.op_name || "UNKNOWN"}: ${(detail.reason_codes || []).join(" / ")}`).join("; ") || "none",
  ]));
  const loopFailureRows = allLoopFailureRows.slice(0, 48);
  const sequenceMapRows = (extended.sequence_map_rows || []).slice(0, 48).map((row) => [
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)}`,
    row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
    row.status || "not assessed",
    row.exact_input_sequence_length == null ? "runtime unknown" : formatNumber(row.exact_input_sequence_length),
    `${formatNumber(row.element_expansion_count || 0)} element(s) / ${formatNumber(row.element_node_evaluation_count || 0)} body-node evaluation(s)`,
    (row.reason_codes || []).join(" / ") || row.body_status || "pass",
  ]);
  const containerSources = (container.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const containerRows = (container.rows || []).slice(0, 64).map((row) => [
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}`,
    row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
    row.status || "not assessed",
    (row.canonical_output_types || []).map(code).join(" / ") || "unresolved",
    (row.sequence_lengths || []).map((value) => value == null ? "runtime unknown" : formatNumber(value)).join(" / ") || "N/A",
    (row.optional_presence || []).map((value) => value == null ? "runtime unknown" : value ? "present" : "empty").join(" / ") || "N/A",
    (row.reason_codes || []).join(" / ") || "pass",
  ]);
  const tfidfSources = (tfidf.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const tfidfRows = (tfidf.rows || []).slice(0, 64).map((row) => [
    `${code(row.scope || "main_graph")} / #${padOp(row.node_index)} ${row.op_name || "TfIdfVectorizer"}`,
    row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
    row.status || "not assessed",
    `${code(row.input_name || "unnamed")} ${row.input_dtype || "UNKNOWN"} ${code(JSON.stringify(row.input_shape || []))} -> ${code(row.output_name || "unnamed")} ${row.output_dtype || "UNKNOWN"} ${code(JSON.stringify(row.exact_output_shape || []))}`,
    `${row.mode || "?"}; gram ${row.minimum_gram_length ?? "?"}-${row.maximum_gram_length ?? "?"}; skip <= ${row.maximum_skip_count ?? "?"}; pool ${row.pool_kind || "?"} ${row.exact_pool_item_count ?? "?"} item(s); definitions ${row.exact_active_ngram_definition_count ?? "?"} active / ${row.exact_ngram_definition_count ?? "?"} total; unused prefix ${row.exact_unused_pool_prefix_item_count ?? "?"}; duplicate active/inactive ${row.exact_duplicate_active_ngram_count ?? "?"} / ${row.exact_duplicate_inactive_ngram_count ?? "?"}`,
    `width ${row.exact_output_width ?? "?"}; indexes ${row.exact_ngram_index_count ?? "?"}; duplicate coordinates ${row.exact_duplicate_output_coordinate_count ?? "?"}; unaddressed coordinates ${row.exact_unaddressed_output_coordinate_count ?? "?"}; weights ${row.weights_present ? "present" : "absent/implicit 1"} count ${row.exact_weight_count ?? "?"}; coordinate/value mapping disagreements ${row.exact_weight_coordinate_disagreement_count ?? "?"} / ${row.exact_weight_coordinate_value_disagreement_count ?? "?"}`,
    row.static_execution_status === "assessed_exact"
      ? `${row.static_input_status || "not assessed input"}; ${row.exact_static_input_value_count ?? "?"} input value(s); ${row.static_execution_status}; work ${row.exact_static_work_units ?? "?"}; ${formatNumber(row.exact_match_count || 0)} match(es), ${formatNumber(row.exact_nonzero_frequency_count || 0)} nonzero frequency coordinate(s), ${formatNumber(row.exact_output_value_count || 0)} output value(s), ${formatNumber(row.exact_nonzero_output_count || 0)} nonzero, ${formatNumber(row.exact_negative_zero_output_count || 0)} signed zero at [${(row.exact_output_negative_zero_indices || []).map(formatNumber).join(", ") || "none"}], ${formatNumber(row.exact_ort_reference_divergent_output_count || 0)} ORT/reference divergence(s) at [${(row.exact_ort_reference_divergent_output_indices || []).map(formatNumber).join(", ") || "none"}]; output ${code(JSON.stringify(row.exact_output_values || []))}`
      : `${row.static_input_status || "not assessed input"}; ${row.exact_static_input_value_count ?? "?"} input value(s); ${row.static_execution_status || "not assessed"}; work ${row.exact_static_work_units ?? "?"}; matches ${row.exact_match_count ?? "?"}; ${row.exact_nonzero_frequency_count ?? "?"} nonzero frequency coordinate(s); output values/nonzero ${row.exact_output_value_count ?? "?"} / ${row.exact_nonzero_output_count ?? "?"}; signed zero ${row.exact_negative_zero_output_count ?? "?"} at [${(row.exact_output_negative_zero_indices || []).map(formatNumber).join(", ") || "none"}]; ORT/reference divergence ${row.exact_ort_reference_divergent_output_count ?? "?"} at [${(row.exact_ort_reference_divergent_output_indices || []).map(formatNumber).join(", ") || "none"}]; exact static output unavailable`,
    [...(row.reason_codes || []), ...(row.risk_codes || [])].filter((value, index, values) => values.indexOf(value) === index).join(" / ") || "pass",
  ]);
  const mlValueSources = (mlValue.source_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const mlValueRuntimeSources = (mlValue.runtime_reference_documents || [])
    .map((source) => `${source.role}: ${source.sha256}; ${source.source_ref}`).join(" / ") || "not emitted";
  const mlValueRows = (mlValue.rows || []).slice(0, 64).map((row) => {
    const contractKind = row.contract_kind || "unresolved";
    const inputKind = row.input_kind || "unresolved";
    const inputMapKeyType = row.input_map_key_type || "UNDEFINED";
    const inputMapValueDtype = row.input_map_value_dtype || "UNKNOWN";
    const exactInputMapKeyCount = row.exact_input_map_key_count;
    const sparseKeyBoundsStatus = row.sparse_key_bounds_status || "not emitted";
    const inputDtype = row.input_dtype || "UNKNOWN";
    const inputRank = row.input_rank;
    const inputName = row.input_name || "";
    const inputShape = row.input_shape || [];
    const outputName = row.output_name || "";
    const exactBatchCount = row.exact_batch_count;
    const exactFeatureCount = row.exact_feature_count;
    const classKeyType = row.class_key_type || "UNDEFINED";
    const classKeyCount = row.class_key_count || 0;
    const exactOutputSequenceLength = row.exact_output_sequence_length;
    const outputKind = row.output_kind || "unresolved";
    const outputDtype = row.output_dtype || "UNKNOWN";
    const outputRank = row.exact_output_rank;
    const outputShape = row.exact_output_shape || [];
    const outputElements = row.exact_dense_output_element_count;
    const outputShapeBasis = row.output_shape_basis || "not emitted";
    const runtimeReferenceStatus = row.runtime_reference_status || "not emitted";
    const attributeMode = row.attribute_mode || "unresolved";
    const vocabularyType = row.vocabulary_type || "UNDEFINED";
    const vocabularyCount = row.vocabulary_count || 0;
    const duplicateVocabularyCount = row.duplicate_vocabulary_count || 0;
    const vocabularyPreview = row.vocabulary_preview || [];
    const mappingDirection = row.mapping_direction || "UNRESOLVED";
    const categoryPairCount = row.category_pair_count || 0;
    const categoryStringCount = row.category_string_count || 0;
    const categoryInt64Count = row.category_int64_count || 0;
    const duplicateStringKeyCount = row.duplicate_string_key_count || 0;
    const duplicateInt64KeyCount = row.duplicate_int64_key_count || 0;
    const activeDuplicateKeyCount = row.active_duplicate_key_count || 0;
    const activeDefaultType = row.active_default_type || "UNDEFINED";
    const activeDefaultValue = row.active_default_value ?? "";
    const categoryStringPreview = row.category_string_preview || [];
    const categoryInt64Preview = row.category_int64_preview || [];
    const inputNames = row.input_names || [];
    const inputDtypes = row.input_dtypes || [];
    const inputRanks = row.input_ranks || [];
    const inputShapes = row.input_shapes || [];
    const inputListContract = inputNames.map((name, index) => `${code(name || "unnamed")} ${inputDtypes[index] || "UNKNOWN"} rank ${inputRanks[index] == null ? "?" : formatNumber(inputRanks[index])} ${code(JSON.stringify(inputShapes[index] || []))}`).join(" / ");
    const exactBatchCounts = row.exact_batch_counts || [];
    const inputRowWidths = row.exact_input_row_feature_counts || [];
    const configuredFeatureDimensions = row.configured_feature_dimensions || [];
    const configuredFeatureDimensionCount = row.configured_feature_dimension_count || 0;
    const totalConfiguredFeatureCount = row.total_configured_feature_count;
    const copiedFeatureCounts = row.copied_feature_counts_per_input || [];
    const paddedFeatureCounts = row.padded_feature_counts_per_input || [];
    const truncatedFeatureCounts = row.truncated_feature_counts_per_input || [];
    const copiedFeatureTotal = row.exact_copied_feature_count_per_batch;
    const paddedFeatureTotal = row.exact_padded_feature_count_per_batch;
    const truncatedFeatureTotal = row.exact_truncated_feature_count_per_batch;
    const paddedInputCount = row.padded_input_count || 0;
    const truncatedInputCount = row.truncated_input_count || 0;
    const indexInputName = row.index_input_name || "";
    const indexInputDtype = row.index_input_dtype || "UNKNOWN";
    const indexInputRank = row.index_input_rank;
    const indexInputShape = row.index_input_shape || [];
    const exactIndexCount = row.exact_index_count;
    const exactIndexValuesStatus = row.exact_index_values_status || "not emitted";
    const exactIndexPreview = row.exact_index_preview || [];
    const duplicateIndexCount = row.duplicate_index_count || 0;
    const indexBoundsStatus = row.index_bounds_status || "not emitted";
    const outOfBoundsIndexCount = row.out_of_bounds_index_count || 0;
    const thresholdText = row.threshold_value_text || "not emitted";
    const thresholdSource = row.threshold_source || "not emitted";
    const staticValueStatus = row.static_value_assessment_status || "not emitted";
    const exactStaticInputCount = row.exact_static_input_value_count;
    const exactAboveThresholdCount = row.exact_above_threshold_count;
    const exactAtOrBelowThresholdCount = row.exact_at_or_below_threshold_count;
    const exactEqualThresholdCount = row.exact_equal_threshold_count;
    const normalizerMode = row.normalizer_mode || "unresolved";
    const normalizerModeSource = row.normalizer_mode_source || "not emitted";
    const normalizerStaticStatus = row.normalizer_static_assessment_status || "not emitted";
    const normalizerInputValues = row.normalizer_exact_input_value_count;
    const normalizerBatchCount = row.normalizer_batch_count;
    const normalizerRowWidth = row.normalizer_row_width;
    const normalizerDivisorKind = row.normalizer_divisor_kind || "not emitted";
    const normalizerDivisorPreview = row.normalizer_divisor_preview || [];
    const normalizerZeroRows = row.normalizer_zero_divisor_row_count;
    const normalizerNegativeMaxRows = row.normalizer_negative_max_divisor_row_count;
    const normalizerIntegerRounding = row.normalizer_integer_float32_rounding_count;
    const normalizerSignedOverflow = row.normalizer_signed_overflow_value_count;
    const normalizerNonfinite = row.normalizer_non_finite_output_count;
    const normalizerSignedZero = row.normalizer_signed_zero_output_count;
    const normalizerOutputPreview = row.normalizer_output_preview || [];
    const scalerContractStatus = row.scaler_parameter_contract_status || "not assessed";
    const scalerContractReason = row.scaler_parameter_contract_reason || "not emitted";
    const scalerParameterMode = row.scaler_parameter_mode || "unresolved";
    const scalerFeatureStride = row.scaler_feature_stride;
    const scalerScaleValues = row.scaler_scale_values || [];
    const scalerOffsetValues = row.scaler_offset_values || [];
    const scalerStaticStatus = row.scaler_static_assessment_status || "not emitted";
    const scalerInputValues = row.scaler_exact_input_value_count;
    const scalerIntegerRounding = row.scaler_integer_float32_rounding_count;
    const scalerNonfiniteParameters = row.scaler_non_finite_parameter_count;
    const scalerNonfiniteOutputs = row.scaler_non_finite_output_count;
    const scalerSignedZeroOutputs = row.scaler_signed_zero_output_count;
    const scalerZeroScales = row.scaler_zero_scale_count;
    const scalerOutputPreview = row.scaler_output_preview || [];
    const imputerContractStatus = row.imputer_parameter_contract_status || "not assessed";
    const imputerContractReason = row.imputer_parameter_contract_reason || "not emitted";
    const imputerParameterMode = row.imputer_parameter_mode || "unresolved";
    const imputerFeatureStride = row.imputer_feature_stride;
    const imputerValues = row.imputer_imputed_values || [];
    const imputerStaticStatus = row.imputer_static_assessment_status || "not emitted";
    const imputerOutputPreview = row.imputer_output_preview || [];
    const oneHotCategories = row.onehot_category_values || [];
    const oneHotVocabularyPreview = row.vocabulary_preview || [];
    const labelEncoderKeys = row.label_encoder_key_values || [];
    const labelEncoderValues = row.label_encoder_value_values || [];
    const oneHotUnknownPreview = row.onehot_unknown_input_preview || [];
    const oneHotOutputPreview = row.onehot_output_preview || [];
    const linearOutputNames = row.output_names || [];
    const linearOutputTypes = row.canonical_output_types || [];
    const linearOutputShapes = row.canonical_output_shapes || [];
    const linearRawScorePreview = row.linear_reference_raw_score_preview || [];
    const linearOutputPreview = row.linear_reference_output_preview || [];
    const linearLabelPreview = row.linear_reference_label_preview || [];
    const svmRawScorePreview = row.svm_reference_raw_score_preview || [];
    const svmOutputScorePreview = row.svm_reference_output_score_preview || [];
    const svmLabelPreview = row.svm_reference_label_preview || [];
    const svmClassifierOnlyLedger = `classifier-only pairwise ${formatNumber(row.svm_pairwise_classifier_count || 0)}, score-width conflict ${row.svm_schema_runtime_score_width_mismatch ? "yes" : "no"}, probability ${row.svm_probability_enabled ? "enabled" : "disabled"}, A/B ${formatNumber(row.svm_prob_a_count || 0)}/${formatNumber(row.svm_prob_b_count || 0)}, expected per array / used total / unused total ${row.svm_expected_probability_parameter_count_per_array == null ? "unresolved" : formatNumber(row.svm_expected_probability_parameter_count_per_array)}/${formatNumber(row.svm_used_probability_parameter_count || 0)}/${formatNumber(row.svm_unused_probability_parameter_count || 0)}`;
    return [
      `${code(row.scope || "main_graph")} / #${padOp(row.node_index)} ${row.op_name || "UNKNOWN"}; ${contractKind.replaceAll("_", " ")}`,
      row.imported_opset == null ? "unresolved" : formatNumber(row.imported_opset),
      row.status || "not assessed",
      row.op_name === "FeatureVectorizer"
        ? `${inputNames.length} inputs ${inputListContract || "none"} -> ${code(outputName || "unnamed")}; first ${code(inputName || "unnamed")} ${code(JSON.stringify(inputShape))}; batches ${exactBatchCounts.map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"}`
        : row.op_name === "ArrayFeatureExtractor"
          ? `${inputListContract || "inputs unresolved"} -> ${code(outputName || "unnamed")}; data ${code(inputName || "unnamed")} ${inputDtype} rank ${inputRank == null ? "?" : formatNumber(inputRank)} ${code(JSON.stringify(inputShape))}; indices ${code(indexInputName || "unnamed")} ${indexInputDtype} rank ${indexInputRank == null ? "?" : formatNumber(indexInputRank)} ${code(JSON.stringify(indexInputShape))}`
          : inputKind === "map"
        ? `${code(inputName || "unnamed")} -> ${code(outputName || "unnamed")}; map<${inputMapKeyType}, ${inputMapValueDtype}>; exact entries ${exactInputMapKeyCount == null ? "not encoded" : formatNumber(exactInputMapKeyCount)}`
        : `${code(inputName || "unnamed")} -> ${code(outputName || "unnamed")}; ${inputDtype}; rank ${inputRank == null ? "?" : formatNumber(inputRank)} ${code(JSON.stringify(inputShape))}; batch ${exactBatchCount == null ? "runtime unknown" : formatNumber(exactBatchCount)}`,
      row.op_name === "Binarizer"
        ? `${attributeMode} ${code(thresholdText)} (${thresholdSource}); static values ${staticValueStatus}; exact ${exactStaticInputCount == null ? "runtime unknown" : formatNumber(exactStaticInputCount)}, above ${exactAboveThresholdCount == null ? "runtime unknown" : formatNumber(exactAboveThresholdCount)}, at/below ${exactAtOrBelowThresholdCount == null ? "runtime unknown" : formatNumber(exactAtOrBelowThresholdCount)}, equal ${exactEqualThresholdCount == null ? "runtime unknown" : formatNumber(exactEqualThresholdCount)}`
        : row.op_name === "Normalizer"
          ? `${attributeMode} ${normalizerMode} (${normalizerModeSource}); ${normalizerDivisorKind}; static ${normalizerStaticStatus}; ${normalizerInputValues == null ? "runtime values" : `${formatNumber(normalizerInputValues)} exact input value(s)`}; ${normalizerBatchCount == null ? "runtime batch" : `${formatNumber(normalizerBatchCount)} batch row(s)`} x ${normalizerRowWidth == null ? "runtime width" : formatNumber(normalizerRowWidth)}; divisors ${normalizerDivisorPreview.map(code).join(" / ") || "not materialized"}`
        : row.op_name === "Scaler"
          ? `${attributeMode}; contract ${scalerContractStatus} (${scalerContractReason}); ${scalerParameterMode}; feature stride ${scalerFeatureStride == null ? "unresolved" : formatNumber(scalerFeatureStride)}; scale x${formatNumber(row.scaler_scale_count || 0)} [${scalerScaleValues.slice(0, 8).map(code).join(" / ") || "none"}${scalerScaleValues.length > 8 ? ` / plus ${formatNumber(scalerScaleValues.length - 8)} more` : ""}]; offset x${formatNumber(row.scaler_offset_count || 0)} [${scalerOffsetValues.slice(0, 8).map(code).join(" / ") || "none"}${scalerOffsetValues.length > 8 ? ` / plus ${formatNumber(scalerOffsetValues.length - 8)} more` : ""}]; static ${scalerStaticStatus}; ${scalerInputValues == null ? "runtime values" : `${formatNumber(scalerInputValues)} exact input value(s)`}`
        : row.op_name === "Imputer"
          ? `${attributeMode}; contract ${imputerContractStatus} (${imputerContractReason}); ${imputerParameterMode}; ${row.imputer_attribute_kind || "unresolved"}; feature stride ${imputerFeatureStride == null ? "unresolved" : formatNumber(imputerFeatureStride)}; imputed x${formatNumber(row.imputer_imputed_value_count || 0)} [${imputerValues.slice(0, 8).map(code).join(" / ") || "none"}${imputerValues.length > 8 ? ` / plus ${formatNumber(imputerValues.length - 8)} more` : ""}]; replace ${code(row.imputer_replaced_value || "not emitted")} (${row.imputer_replaced_value_source || "not emitted"}); static ${imputerStaticStatus}; ${row.imputer_exact_input_value_count == null ? "runtime values" : `${formatNumber(row.imputer_exact_input_value_count)} exact input value(s)`}`
        : row.op_name === "OneHotEncoder"
          ? `${attributeMode}; contract ${row.onehot_parameter_contract_status || "not assessed"} (${row.onehot_parameter_contract_reason || "not emitted"}); ${row.onehot_category_kind || "unresolved"} vocabulary x${formatNumber(row.onehot_category_count || 0)} [${oneHotCategories.slice(0, 8).map(code).join(" / ") || "none"}${oneHotCategories.length > 8 ? ` / plus ${formatNumber(oneHotCategories.length - 8)} more` : ""}]; bounded generic preview [${oneHotVocabularyPreview.map(code).join(" / ") || "none"}]; duplicate/unreachable ${formatNumber(row.onehot_duplicate_category_count || 0)} / ${formatNumber(row.onehot_unreachable_duplicate_column_count || 0)} columns [${(row.onehot_unreachable_duplicate_column_indices || []).join(" / ") || "none"}]; zeros=${code(row.onehot_zeros_value || "unresolved")} (${row.onehot_zeros_source || "not emitted"}; ${row.onehot_zeros_enabled ? "unknowns emit zero slices" : "unknowns fail"}; ${row.onehot_zeros_canonical_boolean ? "canonical" : "noncanonical"}); static ${row.onehot_static_assessment_status || "not assessed"}`
        : row.op_name === "LabelEncoder"
          ? `${attributeMode}; v${row.resolved_schema_version || "?"} ${row.label_encoder_key_dtype || "UNKNOWN"}->${row.label_encoder_value_dtype || "UNKNOWN"}; ONNX/ORT ${row.label_encoder_onnx_contract_status || "not assessed"} / ${row.label_encoder_pinned_ort_contract_status || "not assessed"} (${row.label_encoder_pinned_ort_contract_reason || "not emitted"}); keys/values ${formatNumber(row.label_encoder_key_count || 0)} / ${formatNumber(row.label_encoder_value_count || 0)} from ${row.label_encoder_key_source || "unresolved"} / ${row.label_encoder_value_source || "unresolved"}; key preview [${labelEncoderKeys.slice(0, 8).map(code).join(" / ") || "none"}${labelEncoderKeys.length > 8 ? ` / plus ${formatNumber(labelEncoderKeys.length - 8)} more` : ""}]; value preview [${labelEncoderValues.slice(0, 8).map(code).join(" / ") || "none"}${labelEncoderValues.length > 8 ? ` / plus ${formatNumber(labelEncoderValues.length - 8)} more` : ""}]; bounded generic key preview [${vocabularyPreview.map(code).join(" / ") || "none"}]; duplicate/NaN/non-finite keys/values ${formatNumber(row.label_encoder_duplicate_key_count || 0)} / ${formatNumber(row.label_encoder_nan_key_count || 0)} / ${formatNumber(row.label_encoder_non_finite_key_count || 0)} / ${formatNumber(row.label_encoder_non_finite_value_count || 0)}; first-key / last-key policy ${row.label_encoder_runtime_duplicate_policy || "unresolved"} runtime / ${row.label_encoder_schema_duplicate_policy || "unresolved"} schema; default ${code(row.label_encoder_default_value || "unresolved")} (${row.label_encoder_default_source || "not emitted"}); static ${row.label_encoder_static_assessment_status || "not assessed"}`
        : row.op_name === "LinearClassifier"
          ? `${attributeMode}; ONNX contract ${row.linear_onnx_contract_status || "not assessed"} (${row.linear_onnx_contract_reason || "not emitted"}); pinned ORT contract ${row.linear_pinned_ort_contract_status || "not assessed"} (${row.linear_pinned_ort_contract_reason || "not emitted"}); classes ${row.linear_class_or_target_count == null ? "unresolved" : formatNumber(row.linear_class_or_target_count)}; labels ${row.linear_label_kind || "unresolved"} x${formatNumber(row.linear_label_count || 0)} [${(row.linear_label_values || []).slice(0, 8).map(code).join(" / ") || "none"}]; duplicate labels ${formatNumber(row.linear_duplicate_label_count || 0)}; label output ${code(row.classifier_label_output_name || "unnamed")} ${row.classifier_label_output_dtype || "UNKNOWN"} ${code(JSON.stringify(row.classifier_label_output_shape || []))}; score output ${code(row.classifier_score_output_name || "unnamed")} FLOAT32 ${code(JSON.stringify(row.classifier_score_output_shape || []))}, ${formatNumber(row.classifier_score_class_count || 0)} column(s), binary expansion ${row.classifier_binary_score_expansion ? "yes" : "no"}; coefficients expected/used/serialized ${row.linear_expected_coefficient_count == null ? "unresolved" : formatNumber(row.linear_expected_coefficient_count)} / ${formatNumber(row.linear_used_coefficient_count || 0)} / ${formatNumber(row.linear_coefficient_count || 0)} (${formatNumber(row.linear_unused_coefficient_count || 0)} ignored); intercepts ${formatNumber(row.linear_intercept_count || 0)} (${row.linear_intercepts_used ? "used" : "not used"}); multi_class=${code(row.linear_multi_class_value || "unresolved")} (${row.linear_multi_class_source || "not emitted"}; ${row.linear_multi_class_used_by_pinned_ort ? "used" : "not consulted by pinned ORT compute"}); post_transform=${row.linear_post_transform || "unresolved"} (${row.linear_post_transform_source || "not emitted"})`
        : row.op_name === "LinearRegressor"
          ? `${attributeMode}; ONNX contract ${row.linear_onnx_contract_status || "not assessed"} (${row.linear_onnx_contract_reason || "not emitted"}); pinned ORT contract ${row.linear_pinned_ort_contract_status || "not assessed"} (${row.linear_pinned_ort_contract_reason || "not emitted"}); targets ${row.linear_targets_value || "unresolved"}, resolved count ${row.linear_class_or_target_count == null ? "unresolved" : formatNumber(row.linear_class_or_target_count)} (${row.linear_targets_source || "not emitted"}); coefficients expected/used/serialized ${row.linear_expected_coefficient_count == null ? "unresolved" : formatNumber(row.linear_expected_coefficient_count)} / ${formatNumber(row.linear_used_coefficient_count || 0)} / ${formatNumber(row.linear_coefficient_count || 0)} (${formatNumber(row.linear_unused_coefficient_count || 0)} ignored); intercepts ${formatNumber(row.linear_intercept_count || 0)} (${row.linear_intercepts_used ? "used" : `${formatNumber(row.linear_ignored_intercept_count || 0)} ignored`}); post_transform=${row.linear_post_transform || "unresolved"} (${row.linear_post_transform_source || "not emitted"})`
        : row.op_name === "SVMClassifier"
          ? `${attributeMode}; schema v${formatNumber(row.resolved_schema_version || 1)}; ONNX contract ${row.svm_onnx_contract_status || "not assessed"} (${row.svm_onnx_contract_reason || "not emitted"}); pinned ORT contract ${row.svm_pinned_ort_contract_status || "not assessed"} (${row.svm_pinned_ort_contract_reason || "not emitted"}); mode ${row.svm_mode || "unresolved"}; kernel ${row.svm_kernel_type || "UNRESOLVED"} (${row.svm_kernel_type_source || "not emitted"}; ${row.svm_linear_mode_forced_kernel ? "forced LINEAR by mode" : "serialized kernel active"}); kernel_params [${(row.svm_kernel_params || []).map(code).join(" / ") || "explicit empty"}] (${row.svm_kernel_params_source || "not emitted"}); classes ${formatNumber(row.svm_class_label_count || 0)} ${row.svm_class_label_kind || "unresolved"} [${(row.svm_class_label_values || []).slice(0, 8).map(code).join(" / ") || "none"}], duplicates ${formatNumber(row.svm_duplicate_label_count || 0)}; vectors ${formatNumber(row.svm_vector_count || 0)} [${(row.svm_vectors_per_class || []).map(code).join(" / ") || "linear mode"}], pairwise ${formatNumber(row.svm_pairwise_classifier_count || 0)}; schema score width / pinned ORT score width ${row.svm_schema_score_width == null ? "unresolved" : formatNumber(row.svm_schema_score_width)} / ${row.svm_pinned_ort_score_width == null ? "unresolved" : formatNumber(row.svm_pinned_ort_score_width)}${row.svm_schema_runtime_score_width_mismatch ? " CONFLICT" : ""}; support/coefficients/rho expected/used/serialized ${row.svm_expected_support_vector_value_count == null ? "unresolved" : formatNumber(row.svm_expected_support_vector_value_count)}/${formatNumber(row.svm_used_support_vector_value_count || 0)}/${formatNumber(row.svm_support_vector_value_count || 0)} | ${row.svm_expected_coefficient_count == null ? "unresolved" : formatNumber(row.svm_expected_coefficient_count)}/${formatNumber(row.svm_used_coefficient_count || 0)}/${formatNumber(row.svm_coefficient_count || 0)} | ${row.svm_expected_rho_count == null ? "unresolved" : formatNumber(row.svm_expected_rho_count)}/${formatNumber(row.svm_used_rho_count || 0)}/${formatNumber(row.svm_rho_count || 0)}; unused support/coefficients/rho ${formatNumber(row.svm_unused_support_vector_value_count || 0)}/${formatNumber(row.svm_unused_coefficient_count || 0)}/${formatNumber(row.svm_unused_rho_count || 0)}; probability ${row.svm_probability_enabled ? "enabled" : "disabled"}, A/B ${formatNumber(row.svm_prob_a_count || 0)}/${formatNumber(row.svm_prob_b_count || 0)}, expected per array / used total / unused total ${row.svm_expected_probability_parameter_count_per_array == null ? "unresolved" : formatNumber(row.svm_expected_probability_parameter_count_per_array)}/${formatNumber(row.svm_used_probability_parameter_count || 0)}/${formatNumber(row.svm_unused_probability_parameter_count || 0)}; post_transform=${row.svm_post_transform || "UNRESOLVED"} (${row.svm_post_transform_source || "not emitted"})`
        : row.op_name === "SVMRegressor"
          ? `${attributeMode}; schema v${formatNumber(row.resolved_schema_version || 1)}; ONNX contract ${row.svm_onnx_contract_status || "not assessed"} (${row.svm_onnx_contract_reason || "not emitted"}); pinned ORT contract ${row.svm_pinned_ort_contract_status || "not assessed"} (${row.svm_pinned_ort_contract_reason || "not emitted"}); mode ${row.svm_mode || "unresolved"}; kernel ${row.svm_kernel_type || "UNRESOLVED"} (${row.svm_kernel_type_source || "not emitted"}; ${row.svm_linear_mode_forced_kernel ? "forced LINEAR by mode" : "serialized kernel active"}); kernel_params [${(row.svm_kernel_params || []).map(code).join(" / ") || "explicit empty"}] (${row.svm_kernel_params_source || "not emitted"}); n_supports/vector count ${row.svm_n_supports == null ? "unresolved" : formatNumber(row.svm_n_supports)} / ${formatNumber(row.svm_vector_count || 0)}; one_class=${code(row.svm_one_class_value || "0")} (${row.svm_one_class_source || "not emitted"}); schema score width / pinned ORT score width ${formatNumber(row.svm_schema_score_width || 0)} / ${formatNumber(row.svm_pinned_ort_score_width || 0)}; support/coefficients/rho expected/used/serialized ${row.svm_expected_support_vector_value_count == null ? "unresolved" : formatNumber(row.svm_expected_support_vector_value_count)}/${formatNumber(row.svm_used_support_vector_value_count || 0)}/${formatNumber(row.svm_support_vector_value_count || 0)} | ${row.svm_expected_coefficient_count == null ? "unresolved" : formatNumber(row.svm_expected_coefficient_count)}/${formatNumber(row.svm_used_coefficient_count || 0)}/${formatNumber(row.svm_coefficient_count || 0)} | ${row.svm_expected_rho_count == null ? "unresolved" : formatNumber(row.svm_expected_rho_count)}/${formatNumber(row.svm_used_rho_count || 0)}/${formatNumber(row.svm_rho_count || 0)}; unused support/coefficients/rho ${formatNumber(row.svm_unused_support_vector_value_count || 0)}/${formatNumber(row.svm_unused_coefficient_count || 0)}/${formatNumber(row.svm_unused_rho_count || 0)}; post_transform=${row.svm_post_transform || "UNRESOLVED"} (${row.svm_post_transform_source || "not emitted"}; ${row.svm_post_transform_applied_by_pinned_ort ? "applied" : "not applied by pinned ORT"}); ${svmClassifierOnlyLedger} (not applicable to regressor)`
        : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
          ? `${attributeMode}; schema v${formatNumber(row.resolved_schema_version || 1)}; encoding ${row.tree_encoding || "unresolved"}${row.tree_deprecated_operator ? "; DEPRECATED" : ""}; ONNX / pinned ORT contract ${row.tree_onnx_contract_status || "not assessed"} (${row.tree_onnx_contract_reason || "pass"}) / ${row.tree_pinned_ort_contract_status || "not assessed"} (${row.tree_pinned_ort_contract_reason || "pass"}); aggregate ${row.tree_aggregate_function || "unresolved"} (${row.tree_aggregate_function_source || "not emitted"}); post_transform ${row.tree_post_transform || "unresolved"} (${row.tree_post_transform_source || "not emitted"}); targets/classes ${formatNumber(row.tree_class_or_target_count || 0)}, labels ${row.tree_class_label_kind || "not applicable"} x${formatNumber(row.tree_class_label_count || 0)} [${(row.tree_class_label_preview || []).map(code).join(" / ") || "none"}], duplicate labels ${formatNumber(row.tree_duplicate_class_label_count || 0)}; base values ${formatNumber(row.tree_base_value_count || 0)} (${row.tree_base_value_source || "not emitted"}); topology trees/roots/nodes/branches/leaves ${formatNumber(row.tree_exact_tree_count || 0)}/${formatNumber(row.tree_exact_root_count || 0)}/${formatNumber(row.tree_exact_node_count || 0)}/${formatNumber(row.tree_exact_branch_node_count || 0)}/${formatNumber(row.tree_exact_leaf_count || 0)}, reachable nodes/leaves ${formatNumber(row.tree_reachable_node_count || 0)}/${formatNumber(row.tree_reachable_leaf_count || 0)}, orphan ${formatNumber(row.tree_orphan_node_or_leaf_count || 0)}, max depth ${formatNumber(row.tree_max_depth || 0)}, cycles ${formatNumber(row.tree_cycle_count || 0)}; duplicate node / invalid child / invalid feature / root mismatch / multiple parent ${formatNumber(row.tree_duplicate_node_identity_count || 0)}/${formatNumber(row.tree_invalid_child_reference_count || 0)}/${formatNumber(row.tree_invalid_feature_id_count || 0)}/${formatNumber(row.tree_root_mismatch_count || 0)}/${formatNumber(row.tree_multiple_parent_node_count || 0)}; weights used/unused/unresolved/serialized ${formatNumber(row.tree_used_weight_count || 0)}/${formatNumber(row.tree_unused_weight_count || 0)}/${formatNumber(row.tree_unresolved_weight_count || 0)}/${formatNumber(row.tree_weight_tuple_count || 0)}, ignored nonleaf/single-target ${formatNumber(row.tree_ignored_nonleaf_weight_count || 0)}/${formatNumber(row.tree_single_target_ignored_weight_count || 0)}, invalid ref/id ${formatNumber(row.tree_invalid_weight_reference_count || 0)}/${formatNumber(row.tree_invalid_weight_id_count || 0)}; MEMBER nodes/sets/values/duplicates/separators ${formatNumber(row.tree_membership_node_count || 0)}/${formatNumber(row.tree_membership_set_count || 0)}/${formatNumber(row.tree_membership_value_count || 0)}/${formatNumber(row.tree_membership_duplicate_value_count || 0)}/${formatNumber(row.tree_membership_separator_count || 0)}`
        : row.op_name === "ZipMap"
          ? `${attributeMode}; class labels ${classKeyType} x${formatNumber(classKeyCount)}; duplicates ${formatNumber(row.duplicate_key_count || 0)}; preview ${(row.class_key_preview || []).map((value) => code(value)).join(" / ") || "none"}; features ${exactFeatureCount == null ? "runtime unknown" : formatNumber(exactFeatureCount)}`
        : row.op_name === "CastMap"
          ? `${attributeMode}; cast_to ${row.cast_to || "TO_FLOAT"}; map_form ${row.map_form || "DENSE"}; max_map ${row.max_map == null ? "not exactly decoded" : formatNumber(row.max_map)}; sparse key bounds ${sparseKeyBoundsStatus}`
          : row.op_name === "DictVectorizer"
            ? `${attributeMode}; vocabulary ${vocabularyType} x${formatNumber(vocabularyCount)}; duplicates ${formatNumber(duplicateVocabularyCount)}; preview ${vocabularyPreview.map((value) => code(value)).join(" / ") || "none"}`
            : row.op_name === "CategoryMapper"
              ? `${attributeMode}; ${mappingDirection}; ${formatNumber(categoryPairCount)} pair(s), arrays ${formatNumber(categoryStringCount)} string / ${formatNumber(categoryInt64Count)} int64; duplicate string/int64/active keys ${formatNumber(duplicateStringKeyCount)} / ${formatNumber(duplicateInt64KeyCount)} / ${formatNumber(activeDuplicateKeyCount)}; default ${activeDefaultType} ${code(activeDefaultValue)}; strings ${categoryStringPreview.map((value) => code(value)).join(" / ") || "none"}; int64 ${categoryInt64Preview.map((value) => code(value)).join(" / ") || "none"}`
              : row.op_name === "FeatureVectorizer"
                ? `${attributeMode}; ${formatNumber(configuredFeatureDimensionCount)} configured width(s) [${configuredFeatureDimensions.map(code).join(" / ") || "none"}] -> ${totalConfiguredFeatureCount == null ? "runtime arithmetic unresolved" : formatNumber(totalConfiguredFeatureCount)} output features; actual row widths ${inputRowWidths.map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"}; copied ${copiedFeatureCounts.map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"} (${copiedFeatureTotal == null ? "?" : formatNumber(copiedFeatureTotal)} total); padded ${paddedFeatureCounts.map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"} (${paddedFeatureTotal == null ? "?" : formatNumber(paddedFeatureTotal)} total / ${formatNumber(paddedInputCount)} input(s)); truncated ${truncatedFeatureCounts.map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"} (${truncatedFeatureTotal == null ? "?" : formatNumber(truncatedFeatureTotal)} total / ${formatNumber(truncatedInputCount)} input(s))`
                : `${attributeMode}; exact indices ${exactIndexCount == null ? "runtime unknown" : formatNumber(exactIndexCount)}; values ${exactIndexValuesStatus}; preview ${exactIndexPreview.map(code).join(" / ") || "none"}; duplicates ${formatNumber(duplicateIndexCount)}; bounds ${indexBoundsStatus}; out of bounds ${formatNumber(outOfBoundsIndexCount)}`,
      outputKind === "sequence"
        ? `${code(row.canonical_output_type || "unresolved")}; exact sequence length ${exactOutputSequenceLength == null ? "runtime unknown" : formatNumber(exactOutputSequenceLength)}; basis ${outputShapeBasis}; runtime reference ${runtimeReferenceStatus}`
        : `${outputDtype}; rank ${outputRank == null ? "?" : formatNumber(outputRank)} ${code(JSON.stringify(outputShape))}; exact elements ${outputElements == null ? "runtime unknown" : formatNumber(outputElements)}; ${code(row.canonical_output_type || "unresolved")}; basis ${outputShapeBasis}; runtime reference ${runtimeReferenceStatus}`,
      row.op_name === "Normalizer"
        ? `zero divisor rows ${normalizerZeroRows == null ? "not assessed" : formatNumber(normalizerZeroRows)}; negative signed-MAX rows ${normalizerNegativeMaxRows == null ? "not assessed" : formatNumber(normalizerNegativeMaxRows)}; integer->FLOAT32 changed ${normalizerIntegerRounding == null ? "not assessed" : formatNumber(normalizerIntegerRounding)}; signed overflow ${normalizerSignedOverflow == null ? "not assessed" : formatNumber(normalizerSignedOverflow)}; non-finite outputs ${normalizerNonfinite == null ? "not assessed" : formatNumber(normalizerNonfinite)}; signed-zero outputs ${normalizerSignedZero == null ? "not assessed" : formatNumber(normalizerSignedZero)}; output ${row.normalizer_output_materialized ? "materialized" : "aggregate/preview only"} ${normalizerOutputPreview.map(code).join(" / ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "Scaler"
          ? `zero scales ${scalerZeroScales == null ? "not assessed" : formatNumber(scalerZeroScales)}; integer->FLOAT32 changed ${scalerIntegerRounding == null ? "not assessed" : formatNumber(scalerIntegerRounding)}; non-finite parameters ${scalerNonfiniteParameters == null ? "not assessed" : formatNumber(scalerNonfiniteParameters)}; non-finite outputs ${scalerNonfiniteOutputs == null ? "not assessed" : formatNumber(scalerNonfiniteOutputs)}; signed-zero outputs ${scalerSignedZeroOutputs == null ? "not assessed" : formatNumber(scalerSignedZeroOutputs)}; output ${row.scaler_output_materialized ? "materialized" : "aggregate/preview only"} ${scalerOutputPreview.map(code).join(" / ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "Imputer"
          ? `replaced ${row.imputer_exact_replacement_count == null ? "not assessed" : formatNumber(row.imputer_exact_replacement_count)} (${row.imputer_exact_nan_replacement_count == null ? "not assessed" : formatNumber(row.imputer_exact_nan_replacement_count)} NaN marker); unchanged ${row.imputer_exact_unchanged_count == null ? "not assessed" : formatNumber(row.imputer_exact_unchanged_count)}; ignored imputed values ${formatNumber(row.imputer_ignored_imputed_value_count || 0)}; non-finite imputed/output ${formatNumber(row.imputer_non_finite_imputed_value_count || 0)} / ${row.imputer_non_finite_output_count == null ? "not assessed" : formatNumber(row.imputer_non_finite_output_count)}; signed-zero outputs ${row.imputer_signed_zero_output_count == null ? "not assessed" : formatNumber(row.imputer_signed_zero_output_count)}; output ${row.imputer_output_materialized ? "materialized" : "aggregate/preview only"} ${imputerOutputPreview.map(code).join(" / ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "OneHotEncoder"
          ? `input ${row.onehot_exact_input_value_count == null ? "not assessed" : formatNumber(row.onehot_exact_input_value_count)} = matched ${row.onehot_exact_matched_input_count == null ? "not assessed" : formatNumber(row.onehot_exact_matched_input_count)} + unknown ${row.onehot_exact_unknown_input_count == null ? "not assessed" : formatNumber(row.onehot_exact_unknown_input_count)} + invalid cast ${row.onehot_numeric_to_int64_invalid_count == null ? "not assessed" : formatNumber(row.onehot_numeric_to_int64_invalid_count)}; numeric truncation changed ${row.onehot_numeric_to_int64_changed_count == null ? "not assessed" : formatNumber(row.onehot_numeric_to_int64_changed_count)}; output one/zero ${row.onehot_exact_output_one_count == null ? "not assessed" : formatNumber(row.onehot_exact_output_one_count)} / ${row.onehot_exact_output_zero_count == null ? "not assessed" : formatNumber(row.onehot_exact_output_zero_count)}; guaranteed runtime failure ${row.onehot_guaranteed_runtime_failure ? "yes" : "no"}; unknowns [${oneHotUnknownPreview.map(code).join(" / ") || "none"}]; output ${row.onehot_output_materialized ? "materialized" : "aggregate/preview only"} [${oneHotOutputPreview.map(code).join(" / ") || "none"}]; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "LabelEncoder"
          ? `input ${row.label_encoder_exact_input_value_count == null ? "not assessed" : formatNumber(row.label_encoder_exact_input_value_count)} = matched ${row.label_encoder_exact_match_count == null ? "not assessed" : formatNumber(row.label_encoder_exact_match_count)} + defaulted ${row.label_encoder_exact_default_count == null ? "not assessed" : formatNumber(row.label_encoder_exact_default_count)}; duplicate-key hits ${row.label_encoder_exact_duplicate_key_hit_count == null ? "not assessed" : formatNumber(row.label_encoder_exact_duplicate_key_hit_count)}; schema/runtime mismatches ${row.label_encoder_schema_runtime_mismatch_count == null ? "not assessed" : formatNumber(row.label_encoder_schema_runtime_mismatch_count)}; runtime [${(row.label_encoder_runtime_output_preview || []).map(code).join(" / ") || "runtime values"}]; schema [${(row.label_encoder_schema_output_preview || []).map(code).join(" / ") || "same or unassessed"}]; mismatch inputs [${(row.label_encoder_mismatch_input_preview || []).map(code).join(" / ") || "none"}]; output ${row.label_encoder_output_materialized ? "materialized" : "not propagated/preview only"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "LinearClassifier" || row.op_name === "LinearRegressor"
          ? `outputs ${linearOutputNames.map((name, index) => `${code(name || "unnamed")} ${code(linearOutputTypes[index] || "unresolved")} ${code(JSON.stringify(linearOutputShapes[index] || []))}`).join(" / ") || "unresolved"}; scalar FLOAT32 reference ${row.linear_reference_assessment_status || "not assessed"}; input/raw-score count ${row.linear_reference_input_value_count == null ? "runtime unknown" : formatNumber(row.linear_reference_input_value_count)} / ${row.linear_reference_raw_score_count == null ? "runtime unknown" : formatNumber(row.linear_reference_raw_score_count)}; raw [${linearRawScorePreview.map(code).join(" / ") || "none"}]; transformed [${linearOutputPreview.map(code).join(" / ") || "none"}]; labels [${linearLabelPreview.map(code).join(" / ") || "none"}]; non-finite parameters/scores ${formatNumber(row.linear_non_finite_parameter_count || 0)} / ${formatNumber(row.linear_reference_non_finite_raw_score_count || 0)}; zero-margin decisions ${formatNumber(row.linear_reference_decision_boundary_count || 0)}; boundary: ${row.linear_reference_boundary || "not emitted"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : row.op_name === "SVMClassifier" || row.op_name === "SVMRegressor"
          ? `outputs ${(row.output_names || []).map((name, index) => `${code(name || "unnamed")} ${code((row.canonical_output_types || [])[index] || "unresolved")} ${code(JSON.stringify((row.canonical_output_shapes || [])[index] || []))}`).join(" / ") || `${code(row.output_name || "unnamed")} ${code(row.canonical_output_type || "unresolved")}`}; scalar FLOAT32 reference ${row.svm_reference_assessment_status || "not assessed"}; input/raw-score count ${row.svm_reference_input_value_count == null ? "runtime unknown" : formatNumber(row.svm_reference_input_value_count)} / ${row.svm_reference_raw_score_count == null ? "runtime unknown" : formatNumber(row.svm_reference_raw_score_count)}; raw [${svmRawScorePreview.map(code).join(" / ") || "none"}]; output scores [${svmOutputScorePreview.map(code).join(" / ") || "none"}]; labels [${svmLabelPreview.map(code).join(" / ") || "none"}]; non-finite parameters/scores ${formatNumber(row.svm_non_finite_parameter_count || 0)} / ${formatNumber(row.svm_reference_non_finite_score_count || 0)}; decision-boundary cases ${formatNumber(row.svm_reference_decision_boundary_count || 0)}; boundary: ${row.svm_reference_boundary || "not emitted"}; reference values are not claimed runtime-bit-exact; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
          ? `outputs ${(row.output_names || []).map((name, index) => `${code(name || "unnamed")} ${code((row.canonical_output_types || [])[index] || "unresolved")} ${code(JSON.stringify((row.canonical_output_shapes || [])[index] || []))}`).join(" / ") || `${code(row.output_name || "unnamed")} ${code(row.canonical_output_type || "unresolved")}`}; scalar source-order reference ${row.tree_reference_assessment_status || "not assessed"}; rows/input/path steps/raw/output scores ${row.tree_reference_row_count == null ? "runtime unknown" : formatNumber(row.tree_reference_row_count)}/${row.tree_reference_input_value_count == null ? "runtime unknown" : formatNumber(row.tree_reference_input_value_count)}/${row.tree_reference_path_step_count == null ? "runtime unknown" : formatNumber(row.tree_reference_path_step_count)}/${row.tree_reference_raw_score_count == null ? "runtime unknown" : formatNumber(row.tree_reference_raw_score_count)}/${row.tree_reference_output_score_count == null ? "runtime unknown" : formatNumber(row.tree_reference_output_score_count)}; raw [${(row.tree_reference_raw_score_preview || []).map(code).join(" / ") || "none"}]; output scores [${(row.tree_reference_output_score_preview || []).map(code).join(" / ") || "none"}]; labels [${(row.tree_reference_label_preview || []).map(code).join(" / ") || "none"}]; non-finite parameters/scores ${formatNumber(row.tree_non_finite_parameter_count || 0)}/${formatNumber(row.tree_reference_non_finite_score_count || 0)}; decision-boundary / unwritten-score cases ${formatNumber(row.tree_reference_decision_boundary_count || 0)}/${formatNumber(row.tree_reference_unwritten_score_count || 0)}; boundary: ${row.tree_reference_boundary || "not emitted"}; reference values are not claimed runtime-bit-exact; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
        : [...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass",
    ];
  });
  return [
    `## ONNX Shape Inference Coverage (${shape.evidence_class || "DERIVED"}${shape.status === "assessed" ? "" : ` / ${String(shape.status || "partial").toUpperCase()}`})`,
    `Schema ${code(shape.schema || "not embedded")}.`,
    markdownTable(["Coverage field", "Value"], [
      ["Inference engine", shape.engine || "not emitted"],
      ["Pinned ONNX release / commit", `${shape.source_release || "not emitted"} / ${code(shape.source_commit || "not emitted")}`],
      ["Pinned source documents", sourceDocuments],
      ["Main-graph nodes", formatNumber(shape.attempted_nodes || 0)],
      ["OperatorSetIdProto imports", `${opsetContract.status || "not emitted"}; ${formatNumber(opsetContract.valid_import_count || 0)} valid / ${formatNumber(opsetContract.invalid_import_count || 0)} invalid; ${formatNumber(opsetContract.effective_domain_count || 0)} effective domain(s); ${formatNumber(opsetContract.duplicate_domain_count || 0)} repeated domain(s) (${formatNumber(opsetContract.duplicate_identical_domain_count || 0)} identical-version / ${formatNumber(opsetContract.duplicate_version_variant_domain_count || 0)} multi-version), resolved by ${opsetContract.resolution_rule || "not emitted"}`],
      ["Rule-supported / unsupported nodes", `${formatNumber(shape.rule_supported_nodes || 0)} / ${formatNumber(shape.rule_unsupported_nodes || 0)}`],
      ["OpSchema formal contract", `${shape.schema_form_assessment_status || "not emitted"}; valid ${formatNumber(shape.schema_form_valid_node_count || 0)}, invalid ${formatNumber(shape.schema_form_invalid_node_count || 0)}, unresolved ${formatNumber(shape.schema_form_unresolved_node_count || 0)}`],
      ["Supported but unresolved nodes", `${formatNumber(shape.rule_unresolved_node_count || 0)}${shape.rule_unresolved_node_indices?.length ? `; indices ${shape.rule_unresolved_node_indices.slice(0, 32).map((index) => `#${padOp(index)}`).join(" / ")}${shape.rule_unresolved_node_indices.length > 32 ? " / ..." : ""}` : ""}`],
      ["Extended shape scope", `${scope.status || "not emitted"}; ${formatNumber(scope.reachable_scope_count || 0)} reachable scope row(s); nested-graph execution scopes ${formatNumber(scope.reachable_nested_graph_count || 0)} (${formatNumber(scope.reachable_nested_graph_node_count || 0)} node(s)); reachable local-function bodies ${formatNumber(scope.reachable_local_function_definition_count || 0)} (${formatNumber(scope.reachable_local_function_body_node_count || 0)} body node(s))`],
      ["FunctionProto default GraphProto inventory", `${formatNumber(scope.function_default_graph_count || 0)} definition(s) / ${formatNumber(scope.function_default_graph_node_count || 0)} node(s); definition inventory is separate from invocation-scoped execution rows`],
      ["Recursive scope schema", code(extended.schema || "not emitted")],
      ["Recursive scope engine", `${extended.evidence_class || "not emitted"}; ${extended.status || "not emitted"}; schema ${code(extended.schema || "not emitted")}; source ${extended.source_release || "not emitted"} / ${code(extended.source_commit || "not emitted")}`],
      ["Recursive scope source documents", extendedSources],
      ["Recursive execution ledger", `${formatNumber(extended.scope_execution_count || 0)} execution(s) across ${formatNumber(extended.scope_definition_count || 0)} scope definition(s); ${formatNumber(extended.fully_assessed_scope_count || 0)} fully assessed; residual ${formatNumber(extended.residual_unassessed_node_count || 0)} unassessed node(s) / ${formatNumber(extended.residual_unresolved_output_count || 0)} unresolved output(s)`],
      ["Main-graph one-invocation intrinsic cost", mainIntrinsicCostSummary],
      ["Nested-scope intrinsic cost variants", `${formatNumber(extended.intrinsic_cost_variant_count || 0)} retained variant(s); ${formatNumber(extended.intrinsic_cost_variant_overflow_count || 0)} overflowed execution(s); ${formatNumber(extended.intrinsic_cost_unassessed_execution_count || 0)} unassessed execution(s)`],
      ["Intrinsic-cost source documents", intrinsicCostSources],
      ["Sequence / Optional value engine", `${container.evidence_class || "not emitted"}; ${container.status || "not emitted"}; schema ${code(container.schema || "not emitted")}; source ${container.source_release || "not emitted"} / ${code(container.source_commit || "not emitted")}; ${formatNumber(container.passed_node_count || 0)} pass / ${formatNumber(container.partially_assessed_node_count || 0)} partial / ${formatNumber(container.failed_node_count || 0)} fail across ${formatNumber(container.assessed_node_count || 0)} node(s)`],
      ["Sequence / Optional source documents", containerSources],
      ["Exact container facts", `${formatNumber(container.exact_sequence_length_output_count || 0)} sequence length output(s); ${formatNumber(container.exact_optional_presence_output_count || 0)} optional-presence output(s)`],
      ["TfIdfVectorizer-9 engine", `${tfidf.evidence_class || "not emitted"}; ${tfidf.status || "not emitted"}; schema ${code(tfidf.schema || "not emitted")}; ONNX ${tfidf.source_release || "not emitted"} / ${code(tfidf.source_commit || "not emitted")}; ORT ${tfidf.runtime_reference_release || "not emitted"} / ${code(tfidf.runtime_reference_commit || "not emitted")}; ${formatNumber(tfidf.passed_node_count || 0)} pass / ${formatNumber(tfidf.partially_assessed_node_count || 0)} partial / ${formatNumber(tfidf.failed_node_count || 0)} fail across ${formatNumber(tfidf.assessed_node_count || 0)} node(s)`],
      ["TfIdfVectorizer-9 source documents", tfidfSources],
      ["Exact TfIdfVectorizer facts", `${formatNumber(tfidf.exact_static_node_count || 0)} exact static node(s); definitions ${formatNumber(tfidf.exact_active_ngram_definition_count || 0)} active / ${formatNumber(tfidf.exact_ngram_definition_count || 0)} total; ${tfidf.exact_match_count == null ? "runtime-dependent matches" : `${formatNumber(tfidf.exact_match_count)} exact match(es)`}; ${tfidf.exact_output_value_count == null ? "runtime-dependent outputs" : `${formatNumber(tfidf.exact_output_value_count)} exact output value(s)`}; duplicate output coordinates ${tfidf.exact_duplicate_output_coordinate_count == null ? "not assessed" : formatNumber(tfidf.exact_duplicate_output_coordinate_count)}; weight-value mapping disagreements ${tfidf.exact_weight_coordinate_value_disagreement_count == null ? "not assessed" : formatNumber(tfidf.exact_weight_coordinate_value_disagreement_count)}; ORT/reference divergent outputs ${tfidf.exact_ort_reference_divergent_output_count == null ? "not assessed" : formatNumber(tfidf.exact_ort_reference_divergent_output_count)}`],
      ["ONNX-ML value-contract engine", `${mlValue.evidence_class || "not emitted"}; ${mlValue.status || "not emitted"}; schema ${code(mlValue.schema || "not emitted")}; source ${mlValue.source_release || "not emitted"} / ${code(mlValue.source_commit || "not emitted")}; ${formatNumber(mlValue.passed_node_count || 0)} pass / ${formatNumber(mlValue.partially_assessed_node_count || 0)} partial / ${formatNumber(mlValue.failed_node_count || 0)} fail across ${formatNumber(mlValue.assessed_node_count || 0)} node(s)`],
      ["ONNX-ML value-contract source documents", mlValueSources],
      ["ONNX-ML runtime reference", `${code(mlValue.runtime_reference_commit || "not emitted")}; ${mlValueRuntimeSources}`],
      ["Exact ONNX-ML value facts", `${formatNumber(mlValue.map_producer_node_count || 0)} map producer(s) / ${formatNumber(mlValue.map_consumer_node_count || 0)} map consumer(s) / ${formatNumber(mlValue.tensor_mapper_node_count || 0)} tensor mapper(s) / ${formatNumber(mlValue.tensor_label_mapping_node_count || 0)} tensor label mapper(s) / ${formatNumber(mlValue.tensor_aggregator_node_count || 0)} tensor aggregator(s) / ${formatNumber(mlValue.tensor_selector_node_count || 0)} tensor selector(s) / ${formatNumber(mlValue.tensor_normalization_node_count || 0)} tensor normalization node(s) / ${formatNumber(mlValue.tensor_affine_scaler_node_count || 0)} tensor affine scaler node(s) / ${formatNumber(mlValue.tensor_imputation_node_count || 0)} tensor imputation node(s) / ${formatNumber(mlValue.tensor_encoder_node_count || 0)} tensor encoder node(s); ${formatNumber(mlValue.exact_sequence_length_output_count || 0)} exact sequence length output(s); ${formatNumber(mlValue.exact_dense_output_shape_count || 0)} exact dense output shape(s); ${formatNumber(mlValue.exact_class_key_count || 0)} class key(s); ${formatNumber(mlValue.exact_vocabulary_entry_count || 0)} vocabulary entry(ies); ${formatNumber(mlValue.exact_category_pair_count || 0)} category pair(s)`],
      ["Binarizer exact threshold effects", `${formatNumber(mlValue.binarizer_exact_static_node_count || 0)} / ${formatNumber(mlValue.binarizer_node_count || 0)} node(s) with complete artifact-known inputs; ${formatNumber(mlValue.exact_binarizer_input_value_count || 0)} exact value(s) = ${formatNumber(mlValue.exact_binarizer_above_threshold_count || 0)} above + ${formatNumber(mlValue.exact_binarizer_at_or_below_threshold_count || 0)} at/below; ${formatNumber(mlValue.exact_binarizer_equal_threshold_count || 0)} exactly equal; ${formatNumber(mlValue.binarizer_schema_default_threshold_node_count || 0)} schema-default threshold node(s); ${formatNumber(mlValue.binarizer_nonfinite_threshold_node_count || 0)} non-finite threshold(s)`],
      ["Normalizer exact row effects", `${formatNumber(mlValue.normalizer_static_assessed_node_count || 0)} / ${formatNumber(mlValue.normalizer_node_count || 0)} node(s) statically assessed; ${formatNumber(mlValue.normalizer_output_materialized_node_count || 0)} output(s) materialized; ${formatNumber(mlValue.exact_normalizer_input_value_count || 0)} exact input value(s); ${formatNumber(mlValue.exact_normalizer_zero_divisor_row_count || 0)} zero-divisor row(s); ${formatNumber(mlValue.exact_normalizer_negative_max_divisor_row_count || 0)} negative signed-MAX row(s); ${formatNumber(mlValue.exact_normalizer_integer_float32_rounding_count || 0)} integer value(s) changed by FLOAT32 cast; ${formatNumber(mlValue.exact_normalizer_signed_overflow_value_count || 0)} signed-overflow value(s); ${formatNumber(mlValue.exact_normalizer_non_finite_output_count || 0)} non-finite output(s); ${formatNumber(mlValue.exact_normalizer_signed_zero_output_count || 0)} signed-zero output(s); ${formatNumber(mlValue.normalizer_schema_default_mode_node_count || 0)} schema-default MAX node(s)`],
      ["Scaler exact affine effects", `${formatNumber(mlValue.scaler_static_assessed_node_count || 0)} / ${formatNumber(mlValue.scaler_node_count || 0)} node(s) statically assessed; ${formatNumber(mlValue.scaler_output_materialized_node_count || 0)} output(s) materialized; ${formatNumber(mlValue.scaler_invalid_runtime_contract_node_count || 0)} pinned-runtime-invalid contract(s); ${formatNumber(mlValue.exact_scaler_input_value_count || 0)} exact input value(s); ${formatNumber(mlValue.exact_scaler_integer_float32_rounding_count || 0)} integer->FLOAT32 changed; ${formatNumber(mlValue.exact_scaler_non_finite_parameter_count || 0)} non-finite parameters; ${formatNumber(mlValue.exact_scaler_non_finite_output_count || 0)} non-finite outputs; ${formatNumber(mlValue.exact_scaler_signed_zero_output_count || 0)} signed-zero outputs; ${formatNumber(mlValue.exact_scaler_zero_scale_count || 0)} zero scale(s)`],
      ["Imputer exact replacement effects", `${formatNumber(mlValue.imputer_static_assessed_node_count || 0)} / ${formatNumber(mlValue.imputer_node_count || 0)} node(s) statically assessed; ${formatNumber(mlValue.imputer_output_materialized_node_count || 0)} output(s) materialized; ${formatNumber(mlValue.imputer_invalid_runtime_contract_node_count || 0)} invalid pinned-runtime contract(s); ${formatNumber(mlValue.imputer_scalar_first_fallback_node_count || 0)} scalar-first fallback node(s); ${formatNumber(mlValue.imputer_pinned_cpu_dtype_gap_node_count || 0)} pinned CPU dtype gap(s); ${formatNumber(mlValue.exact_imputer_input_value_count || 0)} exact input value(s) = ${formatNumber(mlValue.exact_imputer_replacement_count || 0)} replaced + ${formatNumber(mlValue.exact_imputer_unchanged_count || 0)} unchanged; ${formatNumber(mlValue.exact_imputer_nan_replacement_count || 0)} NaN-marker replacement(s); ${formatNumber(mlValue.exact_imputer_ignored_imputed_value_count || 0)} ignored trailing imputed value(s); ${formatNumber(mlValue.exact_imputer_non_finite_imputed_value_count || 0)} non-finite imputed value(s); ${formatNumber(mlValue.exact_imputer_non_finite_output_count || 0)} non-finite output(s); ${formatNumber(mlValue.exact_imputer_signed_zero_output_count || 0)} signed-zero output(s)`],
      ["OneHotEncoder exact encoding effects", `${formatNumber(mlValue.onehot_static_assessed_node_count || 0)} / ${formatNumber(mlValue.onehot_encoder_node_count || 0)} node(s) statically assessed; ${formatNumber(mlValue.onehot_output_materialized_node_count || 0)} output(s) materialized; ${formatNumber(mlValue.onehot_invalid_contract_node_count || 0)} invalid contract(s); ${formatNumber(mlValue.onehot_duplicate_vocabulary_node_count || 0)} duplicate-vocabulary node(s); ${formatNumber(mlValue.onehot_unknown_all_zero_node_count || 0)} all-zero unknown node(s); ${formatNumber(mlValue.onehot_guaranteed_runtime_failure_node_count || 0)} guaranteed runtime failure(s); ${formatNumber(mlValue.onehot_pinned_cpu_dtype_gap_node_count || 0)} pinned CPU dtype gap(s); ${formatNumber(mlValue.onehot_noncanonical_zeros_node_count || 0)} noncanonical zeros node(s); ${formatNumber(mlValue.onehot_unrepresentable_numeric_cast_node_count || 0)} unrepresentable-cast node(s); ${formatNumber(mlValue.exact_onehot_input_value_count || 0)} exact input value(s) = ${formatNumber(mlValue.exact_onehot_matched_input_count || 0)} matched + ${formatNumber(mlValue.exact_onehot_unknown_input_count || 0)} unknown + ${formatNumber(mlValue.exact_onehot_numeric_to_int64_invalid_count || 0)} invalid cast; ${formatNumber(mlValue.exact_onehot_numeric_to_int64_changed_count || 0)} numeric truncation change(s); output ${formatNumber(mlValue.exact_onehot_output_one_count || 0)} one(s) + ${formatNumber(mlValue.exact_onehot_output_zero_count || 0)} zero(s); ${formatNumber(mlValue.exact_onehot_duplicate_category_count || 0)} duplicate category occurrence(s) / ${formatNumber(mlValue.exact_onehot_unreachable_duplicate_column_count || 0)} unreachable duplicate column(s)`],
      ["LabelEncoder exact mapping effects", `${formatNumber(mlValue.label_encoder_static_assessed_node_count || 0)} / ${formatNumber(mlValue.label_encoder_node_count || 0)} node(s) statically assessed; ${formatNumber(mlValue.label_encoder_output_materialized_node_count || 0)} output(s) materialized; ${formatNumber(mlValue.label_encoder_onnx_contract_failure_node_count || 0)} ONNX contract failure(s); ${formatNumber(mlValue.label_encoder_pinned_ort_contract_failure_node_count || 0)} pinned ORT contract failure(s); ${formatNumber(mlValue.label_encoder_pinned_cpu_dtype_pair_gap_node_count || 0)} CPU dtype-pair gap(s); ${formatNumber(mlValue.label_encoder_duplicate_semantic_conflict_node_count || 0)} first-key / last-key conflict node(s); ${formatNumber(mlValue.label_encoder_nan_semantic_conflict_node_count || 0)} v2 NaN equality conflict node(s); ${formatNumber(mlValue.label_encoder_default_path_node_count || 0)} default-path node(s); ${formatNumber(mlValue.label_encoder_schema_runtime_output_mismatch_node_count || 0)} artifact-known output mismatch node(s); ${formatNumber(mlValue.exact_label_encoder_key_count || 0)} exact key(s); ${formatNumber(mlValue.exact_label_encoder_input_value_count || 0)} exact input(s) = ${formatNumber(mlValue.exact_label_encoder_match_count || 0)} matched + ${formatNumber(mlValue.exact_label_encoder_default_count || 0)} defaulted; ${formatNumber(mlValue.exact_label_encoder_duplicate_key_hit_count || 0)} duplicate-key hit(s); ${formatNumber(mlValue.exact_label_encoder_schema_runtime_mismatch_count || 0)} schema/runtime mismatches; conflicting exact values are not propagated downstream`],
      ["Linear model coefficient/runtime contracts", `${formatNumber(mlValue.linear_model_node_count || 0)} total = ${formatNumber(mlValue.linear_classifier_node_count || 0)} classifier(s) + ${formatNumber(mlValue.linear_regressor_node_count || 0)} regressor(s); ${formatNumber(mlValue.linear_onnx_contract_failure_node_count || 0)} ONNX contract failure(s); ${formatNumber(mlValue.linear_pinned_ort_contract_failure_node_count || 0)} pinned ORT contract failure(s); ${formatNumber(mlValue.linear_pinned_cpu_dtype_gap_node_count || 0)} CPU dtype gap(s); ${formatNumber(mlValue.linear_post_transform_hazard_node_count || 0)} post-transform hazard(s); ${formatNumber(mlValue.linear_unused_coefficient_node_count || 0)} node(s) with ignored coefficients / ${formatNumber(mlValue.linear_ignored_intercept_node_count || 0)} node(s) with ignored intercepts; coefficient conservation ${formatNumber(mlValue.exact_linear_used_coefficient_count || 0)} used + ${formatNumber(mlValue.exact_linear_unused_coefficient_count || 0)} ignored + ${formatNumber(mlValue.exact_linear_unresolved_coefficient_use_count || 0)} unresolved = ${formatNumber(mlValue.exact_linear_coefficient_count || 0)} serialized; ${formatNumber(mlValue.exact_linear_ignored_intercept_count || 0)} ignored intercept(s); ${formatNumber(mlValue.linear_reference_assessed_node_count || 0)} scalar reference-assessed node(s), ${formatNumber(mlValue.exact_linear_reference_input_value_count || 0)} input value(s) / ${formatNumber(mlValue.exact_linear_reference_raw_score_count || 0)} raw score(s); reference values are not claimed runtime-bit-exact without observed MLAS microkernel accumulation order`],
      ["SVM support-vector/runtime contracts", `${formatNumber(mlValue.svm_model_node_count || 0)} total = ${formatNumber(mlValue.svm_classifier_node_count || 0)} classifier(s) + ${formatNumber(mlValue.svm_regressor_node_count || 0)} regressor(s); modes ${formatNumber(mlValue.svm_linear_mode_node_count || 0)} linear / ${formatNumber(mlValue.svm_svc_mode_node_count || 0)} SVC; ${formatNumber(mlValue.svm_onnx_contract_failure_node_count || 0)} ONNX contract failure(s); ${formatNumber(mlValue.svm_pinned_ort_contract_failure_node_count || 0)} pinned ORT contract failure(s); ${formatNumber(mlValue.svm_regressor_pinned_cpu_dtype_gap_node_count || 0)} CPU dtype gap(s); ${formatNumber(mlValue.svm_schema_runtime_score_width_mismatch_node_count || 0)} schema score width / pinned ORT score width conflict(s); ${formatNumber(mlValue.svm_ignored_post_transform_node_count || 0)} ignored regressor post-transform(s); ${formatNumber(mlValue.svm_ignored_parameter_node_count || 0)} node(s) with ignored parameters; ${formatNumber(mlValue.svm_non_finite_node_count || 0)} non-finite parameter/reference node(s); vectors/pairwise classifiers ${formatNumber(mlValue.exact_svm_vector_count || 0)} / ${formatNumber(mlValue.exact_svm_pairwise_classifier_count || 0)}; support/coefficients/rho expected/used/serialized conservation: support ${formatNumber(mlValue.exact_svm_used_support_vector_value_count || 0)} used + ${formatNumber(mlValue.exact_svm_unused_support_vector_value_count || 0)} ignored + ${formatNumber(mlValue.exact_svm_unresolved_support_vector_use_count || 0)} unresolved = ${formatNumber(mlValue.exact_svm_support_vector_value_count || 0)} serialized; coefficients ${formatNumber(mlValue.exact_svm_used_coefficient_count || 0)} + ${formatNumber(mlValue.exact_svm_unused_coefficient_count || 0)} + ${formatNumber(mlValue.exact_svm_unresolved_coefficient_use_count || 0)} = ${formatNumber(mlValue.exact_svm_coefficient_count || 0)}; rho ${formatNumber(mlValue.exact_svm_used_rho_count || 0)} + ${formatNumber(mlValue.exact_svm_unused_rho_count || 0)} + ${formatNumber(mlValue.exact_svm_unresolved_rho_use_count || 0)} = ${formatNumber(mlValue.exact_svm_rho_count || 0)}; ${formatNumber(mlValue.svm_reference_assessed_node_count || 0)} scalar reference-assessed node(s), ${formatNumber(mlValue.exact_svm_reference_input_value_count || 0)} input value(s) / ${formatNumber(mlValue.exact_svm_reference_raw_score_count || 0)} raw score(s); reference values are not claimed runtime-bit-exact without observed MLAS microkernel accumulation and platform libm paths`],
      ["TreeEnsemble topology/runtime contracts", `${formatNumber(mlValue.tree_ensemble_model_node_count || 0)} total = ${formatNumber(mlValue.tree_ensemble_node_count || 0)} generic v5 + ${formatNumber(mlValue.tree_ensemble_classifier_node_count || 0)} classifier + ${formatNumber(mlValue.tree_ensemble_regressor_node_count || 0)} regressor; ${formatNumber(mlValue.tree_ensemble_deprecated_node_count || 0)} deprecated legacy node(s); ${formatNumber(mlValue.tree_ensemble_onnx_contract_failure_node_count || 0)} ONNX contract failure(s); ${formatNumber(mlValue.tree_ensemble_pinned_ort_contract_failure_node_count || 0)} pinned ORT contract failure(s); ${formatNumber(mlValue.tree_ensemble_pinned_cpu_dtype_gap_node_count || 0)} CPU dtype gap(s); non-finite / boundary / semantic-hazard nodes ${formatNumber(mlValue.tree_ensemble_non_finite_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_reference_boundary_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_semantic_hazard_node_count || 0)}; topology trees/roots/nodes/branches/leaves ${formatNumber(mlValue.exact_tree_ensemble_tree_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_root_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_node_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_branch_node_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_leaf_count || 0)}, reachable nodes/leaves ${formatNumber(mlValue.exact_tree_ensemble_reachable_node_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reachable_leaf_count || 0)}, orphan ${formatNumber(mlValue.exact_tree_ensemble_orphan_node_or_leaf_count || 0)}, maximum depth ${formatNumber(mlValue.maximum_tree_ensemble_depth || 0)}, cycles/duplicate nodes/invalid child/invalid feature/root mismatch/multiple parent ${formatNumber(mlValue.exact_tree_ensemble_cycle_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_duplicate_node_identity_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_invalid_child_reference_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_invalid_feature_id_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_root_mismatch_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_multiple_parent_node_count || 0)}; weight conservation ${formatNumber(mlValue.exact_tree_ensemble_used_weight_count || 0)} used + ${formatNumber(mlValue.exact_tree_ensemble_unused_weight_count || 0)} unused + ${formatNumber(mlValue.exact_tree_ensemble_unresolved_weight_count || 0)} unresolved = ${formatNumber(mlValue.exact_tree_ensemble_weight_tuple_count || 0)} serialized; ignored nonleaf/single-target ${formatNumber(mlValue.exact_tree_ensemble_ignored_nonleaf_weight_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_single_target_ignored_weight_count || 0)}, invalid weight ref/id ${formatNumber(mlValue.exact_tree_ensemble_invalid_weight_reference_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_invalid_weight_id_count || 0)}; MEMBER nodes/sets/values/duplicates/separators ${formatNumber(mlValue.exact_tree_ensemble_membership_node_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_membership_set_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_membership_value_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_membership_duplicate_value_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_membership_separator_count || 0)}; ${formatNumber(mlValue.tree_ensemble_reference_assessed_node_count || 0)} scalar reference-assessed node(s), rows/input/path steps/raw/output scores ${formatNumber(mlValue.exact_tree_ensemble_reference_row_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_input_value_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_path_step_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_raw_score_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_output_score_count || 0)}; non-finite parameter/score ${formatNumber(mlValue.exact_tree_ensemble_non_finite_parameter_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_non_finite_score_count || 0)}, decision boundary/unwritten score ${formatNumber(mlValue.exact_tree_ensemble_reference_decision_boundary_count || 0)}/${formatNumber(mlValue.exact_tree_ensemble_reference_unwritten_score_count || 0)}; reference values are not claimed runtime-bit-exact without observed ORT thread partition, floating reduction order, optimized graph, selected EP, and platform libm`],
      ["JSON-safe static signed-zero ledger", `${formatNumber(staticSignedZeroValues)} signed-zero value(s) across ${formatNumber(staticSignedZeroTensors.length)} tensor(s); exact zero-based indices ${staticSignedZeroDetails}`],
      ["Canonical static non-finite / unsafe-value ledger", `${formatNumber(staticCanonicalTextValues)} canonical value(s) across ${formatNumber(staticCanonicalTextTensors.length)} tensor(s); exact ordered values ${staticCanonicalTextDetails}`],
      ["ONNX-ML duplicate key/column risks", `${formatNumber(mlValue.duplicate_class_key_count || 0)} duplicate ZipMap key occurrence(s) across ${formatNumber(mlValue.duplicate_class_key_node_count || 0)} node(s); ${formatNumber(mlValue.duplicate_vocabulary_entry_count || 0)} duplicate DictVectorizer/OneHotEncoder vocabulary occurrence(s) across ${formatNumber(mlValue.duplicate_vocabulary_node_count || 0)} node(s)`],
      ["ONNX-ML duplicate active category keys", `${formatNumber(mlValue.duplicate_category_active_key_count || 0)} last-write-wins occurrence(s) across ${formatNumber(mlValue.duplicate_category_active_key_node_count || 0)} CategoryMapper node(s)`],
      ["FeatureVectorizer exact width effects", `${formatNumber(mlValue.feature_vectorizer_exact_width_node_count || 0)} / ${formatNumber(mlValue.feature_vectorizer_node_count || 0)} node(s) exact; ${formatNumber(mlValue.exact_feature_vectorizer_configured_feature_count || 0)} configured output feature(s); ${formatNumber(mlValue.exact_feature_vectorizer_padded_feature_count_per_batch || 0)} padded / ${formatNumber(mlValue.exact_feature_vectorizer_truncated_feature_count_per_batch || 0)} truncated value(s) per batch row across exact nodes; ${formatNumber(mlValue.feature_vectorizer_truncating_node_count || 0)} truncating node(s)`],
      ["ArrayFeatureExtractor exact index effects", `${formatNumber(mlValue.array_feature_extractor_exact_index_node_count || 0)} / ${formatNumber(mlValue.array_feature_extractor_node_count || 0)} node(s) with exact index count; ${formatNumber(mlValue.exact_array_feature_extractor_index_count || 0)} index value(s); ${formatNumber(mlValue.array_feature_extractor_duplicate_index_count || 0)} duplicate occurrence(s); bounds assessed for ${formatNumber(mlValue.array_feature_extractor_bounds_assessed_node_count || 0)} node(s), ${formatNumber(mlValue.array_feature_extractor_bounds_failure_node_count || 0)} failure(s)`],
      ["FunctionProto calls", `${formatNumber(extended.local_function_call_pass_count || 0)} pass / ${formatNumber(extended.local_function_call_fail_count || 0)} fail across ${formatNumber(extended.local_function_call_count || 0)} call(s)`],
      ["If / Loop / Scan nodes", `${formatNumber(extended.control_flow_pass_count || 0)} pass / ${formatNumber(extended.control_flow_partial_count || 0)} partial / ${formatNumber(extended.control_flow_fail_count || 0)} fail across ${formatNumber(extended.control_flow_node_count || 0)} node(s)`],
      ["Bounded exact Loop expansion", `${formatNumber(extended.loop_exact_expansion_count || 0)} / ${formatNumber(extended.loop_node_count || 0)} Loop node(s); ${formatNumber(extended.loop_exact_iteration_count || 0)} proven iteration(s), ${formatNumber(extended.loop_exact_body_node_evaluation_count || 0)} body-node evaluation(s), ${formatNumber(extended.loop_non_dense_state_variable_count || 0)} non-dense state variable(s)`],
      ["SequenceMap nodes", `${formatNumber(extended.sequence_map_pass_count || 0)} pass / ${formatNumber(extended.sequence_map_partial_count || 0)} partial / ${formatNumber(extended.sequence_map_fail_count || 0)} fail across ${formatNumber(extended.sequence_map_node_count || 0)} node(s)`],
      ["Executed / fully assessed reachable scopes", `${formatNumber(scope.executed_reachable_scope_count || 0)} / ${formatNumber(scope.fully_assessed_reachable_scope_count || 0)}`],
      ["Unassessed reachable extended-scope nodes", formatNumber(scope.unassessed_reachable_node_count || 0)],
      ["Reachable-scope unresolved outputs", formatNumber(scope.reachable_scope_unresolved_output_count || 0)],
      ["Unused local-function inventory", `${formatNumber(Math.max(0, Number(scope.local_function_definition_count || 0) - Number(scope.reachable_local_function_definition_count || 0)))} definition(s); does not reduce main-graph completeness`],
      ["Declared/inferred contract conflicts", formatNumber(shape.declaration_conflict_count || 0)],
      ["Deterministic operator semantic conflicts", formatNumber(shape.semantic_contract_conflict_count || 0)],
      ["Exact static shape-value tensors propagated", formatNumber(shape.propagated_static_value_tensor_count || 0)],
      ["Exact symbolic shape-value tensors propagated", formatNumber(shape.propagated_symbolic_shape_value_tensor_count || 0)],
      ["Node-output conservation", `${formatNumber(shape.known_node_output_count || 0)} numerically concrete dense + ${formatNumber(shape.unknown_node_output_count || 0)} numerically unbound dense + ${formatNumber(shape.known_non_dense_node_output_count || 0)} known non-dense + ${formatNumber(shape.unresolved_non_dense_node_output_count || 0)} unresolved non-dense = ${formatNumber(shape.node_output_count || 0)} total`],
      ["Complete dense shape contracts", `${formatNumber(shape.shape_contract_known_node_output_count ?? shape.known_node_output_count ?? 0)} / ${formatNumber(shape.tensor_node_output_count ?? 0)}; includes ${formatNumber(shape.symbolic_shape_contract_node_output_count || 0)} unconditional symbolic and ${formatNumber(shape.conditional_shape_contract_node_output_count || 0)} finite conditional contract(s)`],
      ["Partial conditional shape contracts", `${formatNumber(shape.partial_conditional_shape_contract_node_output_count || 0)} output(s); ${formatNumber(shape.conditionally_invalid_node_output_count || 0)} output(s) include an artifact-invalid branch; failures ${formatNumber(shape.conditional_invalid_variant_count || 0)} invalid / ${formatNumber(shape.conditional_unassessed_variant_count || 0)} unresolved; viable variants remain condition-bound`],
      ["Missing dense shape contracts", `${formatNumber(shape.shape_contract_unknown_node_output_count ?? shape.unknown_node_output_count ?? 0)} total = ${formatNumber(shape.invalid_node_output_count || 0)} unconditionally invalid + ${formatNumber(shape.conditionally_invalid_node_output_count || 0)} conditionally invalid + ${formatNumber(shape.unresolved_nonconflict_shape_contract_node_output_count || 0)} non-conflict residual`],
      ["All-kind output contract coverage", `${formatNumber(shape.known_value_node_output_count || 0)} / ${formatNumber(shape.node_output_count || 0)} output(s) (${formatPercent(shape.node_value_assessment_ratio ?? 0)})`],
      ["Numerically concrete dense tensor outputs", `${formatNumber(shape.known_node_output_count || 0)} / ${formatNumber(shape.tensor_node_output_count ?? Math.max(0, Number(shape.node_output_count || 0) - Number(shape.non_dense_node_output_count || 0)))} tensor output(s) (${formatPercent(shape.node_output_assessment_ratio ?? 0)})`],
      ["Numerically unbound dense outputs", formatNumber(shape.unknown_node_output_count ?? Math.max(0, Number(shape.node_output_count || 0) - Number(shape.known_node_output_count || 0) - Number(shape.non_dense_node_output_count || 0)))],
      ["Known / unresolved non-dense outputs", `${formatNumber(shape.known_non_dense_node_output_count || 0)} / ${formatNumber(shape.unresolved_non_dense_node_output_count || 0)} = ${formatNumber(shape.non_dense_node_output_count || 0)} total; known ${(shape.known_non_dense_node_output_names || []).map((name) => code(name)).join(" / ") || "none"}; all ${(shape.non_dense_node_output_names || []).map((name) => code(name)).join(" / ") || "none"}`],
      ["Newly inferred dense / non-dense outputs", `${formatNumber(shape.inferred_outputs || 0)} / ${formatNumber(shape.inferred_non_dense_outputs || 0)}`],
      ["Unknown dense or unresolved-kind values after pass", formatNumber(shape.unknown_tensor_count || 0)],
      ["Declared non-dense values", formatNumber(shape.non_dense_value_count || 0)],
      ["Unsupported op histogram", unsupported],
    ]),
    conflictRows.length ? `### Declared Vs Inferred Shape Contract Conflicts\n\n${markdownTable(["Node", "Tensor", "Field", "Declared", "Inferred"], conflictRows)}` : "",
    (shape.declaration_conflicts || []).length > conflictRows.length ? `> Conflict table truncated: ${formatNumber(conflictRows.length)} / ${formatNumber(shape.declaration_conflicts.length)} rows shown; complete rows remain in engineering evidence.` : "",
    semanticConflictRows.length ? `### Deterministic Operator Semantic Conflicts\n\n${markdownTable(["Node", "Blocked outputs", "Reason", "Artifact-known details"], semanticConflictRows)}` : "",
    (shape.semantic_contract_conflicts || []).length > semanticConflictRows.length ? `> Semantic-conflict table truncated: ${formatNumber(semanticConflictRows.length)} / ${formatNumber(shape.semantic_contract_conflicts.length)} rows shown; complete rows remain in engineering evidence.` : "",
    displayedInvalidOpsetRows.length ? `### OperatorSet Import Contract Failures\n\n${markdownTable(["Import", "Domain", "Version", "Reason"], displayedInvalidOpsetRows.map((row) => [formatNumber(row.index), row.domain || "ai.onnx", String(row.version), (row.reason_codes || []).join(" / ") || "invalid import"]))}` : "",
    invalidOpsetRows.length > displayedInvalidOpsetRows.length ? `> OperatorSet import table truncated: ${formatNumber(displayedInvalidOpsetRows.length)} / ${formatNumber(invalidOpsetRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    displayedSchemaRows.length ? `### OpSchema Formal Contract Failures\n\n${markdownTable(["Node", "Imported opset", "Resolved schema", "Status", "Fail-closed reason"], displayedSchemaRows)}` : "",
    schemaRows.length > displayedSchemaRows.length ? `> OpSchema table truncated: ${formatNumber(displayedSchemaRows.length)} / ${formatNumber(schemaRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    scopeRows.length ? `### Recursive Shape Scope Assessment\n\n${markdownTable(["Scope class", "Scope", "Status", "Executions", "Assessed nodes", "Unresolved outputs"], scopeRows)}` : "",
    (scope.scope_execution_rows || []).length > scopeRows.length ? `> Shape-scope assessment table truncated: ${formatNumber(scopeRows.length)} / ${formatNumber(scope.scope_execution_rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    recursiveScopeRows.length ? `### Recursive Engine Execution Ledger\n\n${markdownTable(["Scope class", "Scope", "Status", "Executions", "Assessed nodes", "Unassessed nodes", "Unresolved outputs", "One-invocation intrinsic cost variants", "Reason"], recursiveScopeRows)}\n\n> Intrinsic variants are reported per serialized scope invocation and are not summed across FunctionProto calls, branch alternatives, Loop/Scan iterations, or SequenceMap elements without an artifact-known execution count and selection contract.` : "",
    (extended.scope_rows || []).length > recursiveScopeRows.length ? `> Recursive-engine execution table truncated: ${formatNumber(recursiveScopeRows.length)} / ${formatNumber(extended.scope_rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    intrinsicCostResidualRows.length ? `### Main-Graph Intrinsic Cost Residuals\n\n${markdownTable(["Ledger", "Operator", "Reason"], intrinsicCostResidualRows)}` : "",
    mainIntrinsicCost.method || "",
    mainIntrinsicCost.interpretation_boundary ? `> ${mainIntrinsicCost.interpretation_boundary}` : "",
    functionRows.length ? `### FunctionProto Call Contracts\n\n${markdownTable(["Function", "Call site", "Status", "Inputs / outputs", "Result"], functionRows)}` : "",
    (extended.function_call_rows || []).length > functionRows.length ? `> Function-call table truncated: ${formatNumber(functionRows.length)} / ${formatNumber(extended.function_call_rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    controlRows.length ? `### If / Loop / Scan Shape Contracts\n\n${markdownTable(["Operator", "Call site", "Imported opset", "Status", "State / scan contract", "Exact Loop expansion", "Result"], controlRows)}` : "",
    (extended.control_flow_rows || []).length > controlRows.length ? `> Control-flow table truncated: ${formatNumber(controlRows.length)} / ${formatNumber(extended.control_flow_rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    loopStateRows.length ? `### Exact Loop State Ledger\n\n${markdownTable(["Loop", "Point", "State", "Value kind", "Dtype", "Shape", "Sequence contract"], loopStateRows)}` : "",
    allLoopStateRows.length > loopStateRows.length ? `> Exact Loop state table truncated: ${formatNumber(loopStateRows.length)} / ${formatNumber(allLoopStateRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    loopFailureRows.length ? `### Reached Loop Contract Failures\n\n${markdownTable(["Loop", "Reached scope", "Operator", "Reason", "Root contract detail"], loopFailureRows)}` : "",
    allLoopFailureRows.length > loopFailureRows.length ? `> Reached Loop failure table truncated: ${formatNumber(loopFailureRows.length)} / ${formatNumber(allLoopFailureRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    sequenceMapRows.length ? `### SequenceMap Value Contracts\n\n${markdownTable(["Call site", "Imported opset", "Status", "Exact input length", "Element expansion work", "Result"], sequenceMapRows)}` : "",
    (extended.sequence_map_rows || []).length > sequenceMapRows.length ? `> SequenceMap table truncated: ${formatNumber(sequenceMapRows.length)} / ${formatNumber(extended.sequence_map_rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    extended.method || "",
    extended.interpretation_boundary ? `> ${extended.interpretation_boundary}` : "",
    containerRows.length ? `### Sequence / Optional Value Contracts\n\n${markdownTable(["Node", "Opset", "Status", "Output TypeProto", "Sequence length", "Optional presence", "Result"], containerRows)}` : "",
    (container.rows || []).length > containerRows.length ? `> Container-value table truncated: ${formatNumber(containerRows.length)} / ${formatNumber(container.rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    container.method || "",
    container.interpretation_boundary ? `> ${container.interpretation_boundary}` : "",
    tfidfRows.length ? `### TfIdfVectorizer-9 Contracts\n\n${markdownTable(["Node", "Opset", "Status", "Input -> output", "Gram / pool contract", "Coordinate contract", "Exact static result", "Result / risk"], tfidfRows)}` : "",
    (tfidf.rows || []).length > tfidfRows.length ? `> TfIdfVectorizer table truncated: ${formatNumber(tfidfRows.length)} / ${formatNumber(tfidf.rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    tfidf.method || "",
    tfidf.interpretation_boundary ? `> ${tfidf.interpretation_boundary}` : "",
    mlValueRows.length ? `### ONNX-ML Value Contracts\n\n${markdownTable(["Node", "Opset", "Status", "Input contract", "Attributes / keys", "Output contract", "Result / risk"], mlValueRows)}` : "",
    (mlValue.rows || []).length > mlValueRows.length ? `> ONNX-ML value-contract table truncated: ${formatNumber(mlValueRows.length)} / ${formatNumber(mlValue.rows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    mlValue.method || "",
    mlValue.interpretation_boundary ? `> ${mlValue.interpretation_boundary}` : "",
    exclusionRows.length ? `### Extended Shape Scope Exclusions\n\n${markdownTable(["Scope class", "Scope", "Nodes", "Reason"], exclusionRows)}` : "",
    (scope.exclusions || []).length > exclusionRows.length ? `> Shape-scope exclusion table truncated: ${formatNumber(exclusionRows.length)} / ${formatNumber(scope.exclusions.length)} rows shown; complete rows remain in engineering evidence.` : "",
    unresolvedRows.length ? `### Supported Rules Remaining Unresolved\n\n${markdownTable(["Node", "Fail-Closed Reason"], unresolvedRows)}` : "",
    (shape.rule_unresolved_nodes || []).length > unresolvedRows.length ? `> Unresolved-rule table truncated: ${formatNumber(unresolvedRows.length)} / ${formatNumber(shape.rule_unresolved_nodes.length)} rows shown; complete rows remain in engineering evidence.` : "",
    shape.method || "",
    `> ${shape.interpretation_boundary || "Dynamic/symbolic dimensions and unsupported operators remain unassessed."}`,
  ].filter(Boolean).join("\n\n");
}

export function onnxTensorDataTypeContractMarkdown(analysis) {
  if (!isOnnxAnalysis(analysis)) return "";
  const contract = analysis?.onnx_tensor_data_type_contract;
  if (!contract) return "## ONNX TensorProto Data-Type Contract (NOT_ASSESSED)\nNo pinned data-type contract was emitted.";
  const inventory = (contract.types || [])
    .filter((type) => Number(type.id) > 0)
    .map((type) => `${type.id}:${type.name}/${type.storage_bits}b`)
    .join(" / ");
  return [
    `## ONNX TensorProto Data-Type Contract (${contract.evidence_class || "SOURCE_PINNED"})`,
    markdownTable(["Field", "Value"], [
      ["Schema / status", `${contract.schema || "not embedded"} / ${contract.status || "not assessed"}`],
      ["Pinned ONNX release / commit", `${contract.source_release || "-"} / ${code(contract.source_commit || "not embedded")}`],
      ["Source / SHA-256", `${contract.source_ref || "not embedded"} / ${code(contract.source_sha256 || "not embedded")}`],
      ["Concrete / fixed-width numeric types", `${formatNumber(contract.concrete_data_type_count || 0)} / ${formatNumber(contract.fixed_width_numeric_data_type_count || 0)}`],
      ["Raw / typed numeric decoders", `${formatNumber(contract.raw_numeric_decoder_count || 0)} / ${formatNumber(contract.typed_numeric_decoder_count || 0)}`],
      ["Packed data types", `${formatNumber(contract.packed_data_type_count || 0)}; ${(contract.packed_data_types || []).join(" / ") || "none"}`],
      ["Packing rule", contract.packing_rule || "not emitted"],
      ["Numerical-integrity projection", contract.numerical_integrity_projection || "not emitted"],
      ["Pinned type inventory", inventory || "none"],
    ]),
    "> SOURCE_PINNED_AND_IMPLEMENTATION_TESTED describes parser coverage for the pinned TensorProto contract. It is not evidence that a selected ONNX Runtime execution provider implements every listed dtype.",
  ].join("\n");
}

export function onnxTypeProtoContractMarkdown(analysis) {
  if (!isOnnxAnalysis(analysis)) return "";
  const contract = analysis?.onnx_type_proto_contract;
  if (!contract) return "## ONNX TypeProto Contract (NOT_ASSESSED)\nNo recursive TypeProto contract was emitted.";
  const invalidRows = contract.invalid_rows || [];
  const displayedInvalidRows = invalidRows.slice(0, 48);
  const kindInventory = (contract.kind_counts || [])
    .map((row) => `${row.name}:${formatNumber(row.count)}`).join(" / ") || "none";
  return [
    `## ONNX TypeProto Contract (${contract.evidence_class || "SOURCE_PINNED_AND_OBSERVED"}${contract.status === "assessed" ? "" : ` / ${String(contract.status || "not assessed").toUpperCase()}`})`,
    markdownTable(["Field", "Value"], [
      ["Schema / status", `${contract.schema || "not embedded"} / ${contract.status || "not assessed"}`],
      ["Pinned ONNX release / commit", `${contract.source_release || "not emitted"} / ${code(contract.source_commit || "not emitted")}`],
      ["Pinned source / SHA-256", `${contract.source_ref || "not emitted"} / ${code(contract.source_sha256 || "not emitted")}`],
      ["Declarations", `${formatNumber(contract.declaration_count || 0)} total / ${formatNumber(contract.declared_type_count || 0)} typed / ${formatNumber(contract.undeclared_optional_type_count || 0)} optional undeclared`],
      ["Valid / invalid declarations", `${formatNumber(contract.valid_type_count || 0)} / ${formatNumber(contract.invalid_type_count || 0)}`],
      ["Recursive type nodes / maximum depth", `${formatNumber(contract.type_node_count || 0)} / ${formatNumber(contract.maximum_type_depth || 0)}`],
      ["Type text bytes", formatBytes(contract.type_text_bytes || 0)],
      ["Dense / sparse tensor values", `${formatNumber(contract.tensor_value_count || 0)} / ${formatNumber(contract.sparse_tensor_value_count || 0)}`],
      ["Non-dense graph values", formatNumber(contract.non_dense_value_count || 0)],
      ["Sequence / map / optional / opaque values", `${formatNumber(contract.sequence_value_count || 0)} / ${formatNumber(contract.map_value_count || 0)} / ${formatNumber(contract.optional_value_count || 0)} / ${formatNumber(contract.opaque_value_count || 0)}`],
      ["Symbolic / unknown dimensions", `${formatNumber(contract.symbolic_dimension_count || 0)} / ${formatNumber(contract.unknown_dimension_count || 0)}`],
      ["Declared kind inventory", kindInventory],
    ]),
    displayedInvalidRows.length ? `### Invalid TypeProto Declarations\n\n${markdownTable(["Scope", "Role", "Value", "Kind", "Canonical type", "Reason"], displayedInvalidRows.map((row) => [
      code(row.scope || "not emitted"),
      row.role || "not emitted",
      code(row.value_name || "unnamed"),
      row.kind || "undefined",
      code(row.canonical_type || "undefined"),
      (row.reason_codes || []).join(" / ") || "not emitted",
    ]))}` : "No malformed TypeProto declaration was observed.",
    invalidRows.length > displayedInvalidRows.length ? `> TypeProto failure table truncated: ${formatNumber(displayedInvalidRows.length)} / ${formatNumber(invalidRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    `Method: ${contract.method || "not emitted"}`,
    `> ${contract.interpretation_boundary || "Only dense tensor declarations participate in dense shape, MAC, payload, liveness, and browser-input calculations."}`,
  ].filter(Boolean).join("\n\n");
}

export function onnxSparseTensorContractMarkdown(analysis) {
  if (!isOnnxAnalysis(analysis)) return "";
  const contract = analysis?.onnx_sparse_tensor_contract;
  if (!contract) return "## ONNX SparseTensorProto Contract (NOT_ASSESSED)\nNo SparseTensorProto contract was emitted.";
  const invalidRows = contract.invalid_rows || [];
  const displayedInvalidRows = invalidRows.slice(0, 48);
  const exactCount = (value) => value == null ? "not assessable" : formatNumber(value);
  return [
    `## ONNX SparseTensorProto Contract (${contract.evidence_class || "SOURCE_PINNED_AND_OBSERVED"}${contract.status === "assessed" ? "" : ` / ${String(contract.status || "not assessed").toUpperCase()}`})`,
    markdownTable(["Field", "Value"], [
      ["Schema / status", `${contract.schema || "not embedded"} / ${contract.status || "not assessed"}`],
      ["Pinned ONNX release / commit", `${contract.source_release || "not emitted"} / ${code(contract.source_commit || "not emitted")}`],
      ["Pinned source / SHA-256", `${contract.source_ref || "not emitted"} / ${code(contract.source_sha256 || "not emitted")}`],
      ["SparseTensorProto records", formatNumber(contract.sparse_tensor_count || 0)],
      ["Graph sparse initializers / attribute values", `${formatNumber(contract.graph_sparse_initializer_count || 0)} / ${formatNumber(contract.attribute_sparse_tensor_count || 0)}`],
      ["Valid / partial / invalid records", `${formatNumber(contract.valid_sparse_tensor_count || 0)} / ${formatNumber(contract.partially_assessed_sparse_tensor_count || 0)} / ${formatNumber(contract.invalid_sparse_tensor_count || 0)}`],
      ["Declared NNZ total", exactCount(contract.declared_nnz_total)],
      ["Dense logical element total", exactCount(contract.dense_logical_element_total)],
      ["Embedded component payload", formatBytes(contract.embedded_payload_bytes || 0)],
      ["External component verification", `${contract.external_payload_coverage_status || "not assessed"}; ${formatNumber(contract.verified_external_payload_component_count || 0)} / ${formatNumber(contract.external_payload_component_count || 0)} values/indices component(s)`],
      ["Index-content coverage", `${formatNumber(contract.index_content_assessed_sparse_tensor_count || 0)} decoded (${formatNumber(contract.index_content_failed_sparse_tensor_count || 0)} invalid) / ${formatNumber(contract.index_content_unassessed_sparse_tensor_count || 0)} not assessed; ${formatNumber(contract.assessed_index_count || 0)} index row(s) decoded`],
      ["Index-content violations", `${formatNumber(contract.out_of_bounds_index_count || 0)} out-of-bounds / ${formatNumber(contract.duplicate_index_count || 0)} duplicate / ${formatNumber(contract.unsorted_index_count || 0)} unsorted`],
    ]),
    displayedInvalidRows.length ? `### Invalid SparseTensorProto Records\n\n${markdownTable(["Scope", "Role / name", "Values", "Indices", "Dense shape / NNZ", "Index encoding", "Payload", "Reason"], displayedInvalidRows.map((row) => [
      code(row.scope || "not emitted"),
      `${row.tensor_role || "not emitted"} / ${code(row.sparse_tensor_name || "unnamed")}`,
      `${row.values_dtype || "UNKNOWN"} [${(row.values_shape || []).join("x")}]`,
      `${row.indices_dtype || "UNKNOWN"} [${(row.indices_shape || []).join("x")}]`,
      `[${(row.dense_shape || []).join("x")}] rank ${formatNumber(row.dense_rank || 0)} / ${row.nnz == null ? "unresolved" : formatNumber(row.nnz)}${row.dense_logical_elements == null ? "" : ` of ${formatNumber(row.dense_logical_elements)}`}`,
      row.index_encoding || "unresolved",
      `${row.payload_status || "not assessed"}; index ${row.index_content_status || "not assessed"}; decoded ${formatNumber(row.assessed_index_count || 0)}, OOB ${formatNumber(row.out_of_bounds_index_count || 0)}, duplicate ${formatNumber(row.duplicate_index_count || 0)}, unsorted ${formatNumber(row.unsorted_index_count || 0)}`,
      (row.reason_codes || []).join(" / ") || "not emitted",
    ]))}` : "No malformed SparseTensorProto record was observed.",
    invalidRows.length > displayedInvalidRows.length ? `> SparseTensorProto failure table truncated: ${formatNumber(displayedInvalidRows.length)} / ${formatNumber(invalidRows.length)} rows shown; complete rows remain in engineering evidence.` : "",
    `Method: ${contract.method || "not emitted"}`,
    `> ${contract.interpretation_boundary || "Sparse storage is not projected into dense compute or memory arithmetic without a sparse-aware contract."}`,
  ].filter(Boolean).join("\n\n");
}
