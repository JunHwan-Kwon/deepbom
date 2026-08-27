import { formatNumber } from "./format.js";

export function renderOnnxDomainViewer(panel, analysis) {
  if (!panel) return;
  const onnx = String(analysis?.format || "").toLowerCase() === "onnx";
  panel.hidden = !onnx;
  if (!onnx) return;
  const body = panel.querySelector("[data-onnx-domain-body]");
  const count = panel.querySelector("[data-onnx-domain-count]");
  if (!body || !count) return;
  const evidence = analysis.onnx_domain_analysis;
  const shape = analysis.onnx_shape_inference || {};
  const typeContract = analysis.onnx_type_proto_contract || {};
  const sparseContract = analysis.onnx_sparse_tensor_contract || {};
  if (!evidence) {
    count.textContent = "Not assessed";
    body.replaceChildren(message("Domain inventory was not emitted for this artifact."));
    return;
  }

  const domains = evidence.domains || [];
  const functions = evidence.functions || [];
  const nodes = evidence.nodes || [];
  const externalNodes = nodes.filter((node) => node.resolution_class === "external_custom_registry");
  const nestedNodeCount = nodes.filter((node) => node.scope_class === "nested_graph").length;
  const duplicateCount = (evidence.duplicate_function_ids || []).length;
  const cycleCount = (evidence.recursive_function_cycles || []).length;
  const schemaFailures = (shape.schema_form_rows || []).filter((row) => row.status !== "pass");
  const scopeExclusions = shape.shape_scope?.exclusions || [];
  const scopeAssessments = shape.shape_scope?.scope_execution_rows || [];
  const extended = shape.extended_scope_inference || {};
  const container = shape.container_value_inference || {};
  const tfidf = shape.tfidf_vectorizer_inference || {};
  const mlValue = shape.ml_value_inference || {};
  const functionCalls = extended.function_call_rows || [];
  const controlFlow = extended.control_flow_rows || [];
  const sequenceMaps = extended.sequence_map_rows || [];
  const recursiveScopeAssessments = extended.scope_rows || [];
  const typeRows = typeContract.rows || [];
  const sparseRows = sparseContract.rows || [];
  const sparseIndexViolations = Number(sparseContract.out_of_bounds_index_count || 0)
    + Number(sparseContract.duplicate_index_count || 0)
    + Number(sparseContract.unsorted_index_count || 0);
  const extensionFamilies = [
    ["custom or contrib domains", domains.some((domain) => domain.domain !== "ai.onnx") || evidence.external_custom_node_count || evidence.ort_contrib_node_count],
    ["local functions", functions.length || evidence.model_local_function_call_count],
    ["nested control flow", nestedNodeCount || functionCalls.length || controlFlow.length || sequenceMaps.length],
    ["sparse values", sparseContract.sparse_tensor_count],
    ["container values", container.assessed_node_count],
    ["TfIdfVectorizer", tfidf.assessed_node_count],
    ["ONNX-ML values", mlValue.assessed_node_count],
  ].filter(([, present]) => Boolean(present)).map(([label]) => label);
  const semanticIssueItems = [
    ["TfIdf semantic divergence", Number(tfidf.exact_weight_coordinate_value_disagreement_count || 0)
      + Number(tfidf.exact_ort_reference_divergent_output_count || 0)],
    ["Duplicate identity or vocabulary entries", Number(mlValue.duplicate_class_key_count || 0)
      + Number(mlValue.duplicate_vocabulary_entry_count || 0)
      + Number(mlValue.duplicate_category_active_key_count || 0)
      + Number(mlValue.onehot_duplicate_vocabulary_node_count || 0)
      + Number(mlValue.label_encoder_duplicate_semantic_conflict_node_count || 0)],
    ["Data-loss or fallback paths", Number(mlValue.feature_vectorizer_truncating_node_count || 0)
      + Number(mlValue.imputer_scalar_first_fallback_node_count || 0)
      + Number(mlValue.exact_imputer_ignored_imputed_value_count || 0)
      + Number(mlValue.exact_onehot_unknown_input_count || 0)
      + Number(mlValue.exact_label_encoder_default_count || 0)
      + Number(mlValue.exact_linear_unused_coefficient_count || 0)
      + Number(mlValue.exact_svm_unused_support_vector_value_count || 0)
      + Number(mlValue.exact_svm_unused_coefficient_count || 0)
      + Number(mlValue.exact_tree_ensemble_orphan_node_or_leaf_count || 0)],
    ["Schema or pinned-runtime contract mismatches", Number(mlValue.scaler_invalid_runtime_contract_node_count || 0)
      + Number(mlValue.imputer_invalid_runtime_contract_node_count || 0)
      + Number(mlValue.onehot_invalid_contract_node_count || 0)
      + Number(mlValue.onehot_guaranteed_runtime_failure_node_count || 0)
      + Number(mlValue.label_encoder_onnx_contract_failure_node_count || 0)
      + Number(mlValue.label_encoder_pinned_ort_contract_failure_node_count || 0)
      + Number(mlValue.exact_label_encoder_schema_runtime_mismatch_count || 0)
      + Number(mlValue.linear_onnx_contract_failure_node_count || 0)
      + Number(mlValue.linear_pinned_ort_contract_failure_node_count || 0)
      + Number(mlValue.svm_onnx_contract_failure_node_count || 0)
      + Number(mlValue.svm_pinned_ort_contract_failure_node_count || 0)
      + Number(mlValue.svm_schema_runtime_score_width_mismatch_node_count || 0)
      + Number(mlValue.tree_ensemble_onnx_contract_failure_node_count || 0)
      + Number(mlValue.tree_ensemble_pinned_ort_contract_failure_node_count || 0)],
    ["Numerical hazard counters", Number(mlValue.exact_normalizer_negative_max_divisor_row_count || 0)
      + Number(mlValue.exact_normalizer_signed_overflow_value_count || 0)
      + Number(mlValue.exact_normalizer_non_finite_output_count || 0)
      + Number(mlValue.exact_scaler_non_finite_parameter_count || 0)
      + Number(mlValue.exact_scaler_non_finite_output_count || 0)
      + Number(mlValue.exact_imputer_non_finite_output_count || 0)
      + Number(mlValue.svm_non_finite_node_count || 0)],
  ].filter(([, value]) => value > 0);
  const semanticIssueCounterCount = semanticIssueItems.reduce((sum, [, value]) => sum + value, 0);
  const issueCounterCount = duplicateCount + cycleCount + schemaFailures.length
    + Number(shape.shape_scope?.unassessed_reachable_node_count || 0)
    + Number(shape.shape_scope?.reachable_scope_unresolved_output_count || 0)
    + Number(extended.local_function_call_fail_count || 0)
    + Number(extended.control_flow_fail_count || 0)
    + Number(extended.residual_unassessed_node_count || 0)
    + Number(extended.residual_unresolved_output_count || 0)
    + Number(typeContract.invalid_type_count || 0)
    + Number(sparseContract.invalid_sparse_tensor_count || 0)
    + Number(sparseContract.partially_assessed_sparse_tensor_count || 0)
    + sparseIndexViolations
    + Number(container.partially_assessed_node_count || 0)
    + Number(container.failed_node_count || 0)
    + Number(tfidf.partially_assessed_node_count || 0)
    + Number(tfidf.failed_node_count || 0)
    + Number(mlValue.partially_assessed_node_count || 0)
    + Number(mlValue.failed_node_count || 0)
    + semanticIssueCounterCount;
  count.textContent = `${formatNumber(domains.length)} ${domains.length === 1 ? "domain" : "domains"} / ${formatNumber(functions.length)} local ${functions.length === 1 ? "function" : "functions"}`;
  const metrics = document.createElement("div");
  metrics.className = "onnx-domain-metrics";
  metrics.append(
    metric("External registry", evidence.external_custom_node_count || 0, evidence.external_custom_node_count ? "warn" : "ok"),
    metric("ORT contrib", evidence.ort_contrib_node_count || 0, "neutral"),
    metric("Local definitions", functions.length, "neutral"),
    metric("Local calls", evidence.model_local_function_call_count || 0, "neutral"),
    metric("Nested nodes", nestedNodeCount, "neutral"),
    metric("Registry issues", duplicateCount + cycleCount, duplicateCount || cycleCount ? "warn" : "ok"),
    metric("Schema-form issues", schemaFailures.length, schemaFailures.length ? "warn" : "ok"),
    metric("Shape-scope exclusions", shape.shape_scope?.unassessed_reachable_node_count || 0, scopeExclusions.length ? "warn" : "ok"),
    metric("Assessed scopes", shape.shape_scope?.fully_assessed_reachable_scope_count || 0, "neutral"),
    metric("Scope unresolved outputs", shape.shape_scope?.reachable_scope_unresolved_output_count || 0, shape.shape_scope?.reachable_scope_unresolved_output_count ? "warn" : "ok"),
    metric("Function-call failures", extended.local_function_call_fail_count || 0, extended.local_function_call_fail_count ? "warn" : "ok"),
    metric("Control-flow failures", extended.control_flow_fail_count || 0, extended.control_flow_fail_count ? "warn" : "ok"),
    metric("Loop exact expansion", `${formatNumber(extended.loop_exact_expansion_count || 0)} / ${formatNumber(extended.loop_node_count || 0)}`, extended.loop_node_count && extended.loop_exact_expansion_count !== extended.loop_node_count ? "warn" : "ok"),
    metric("Loop iterations / work / non-dense", `${formatNumber(extended.loop_exact_iteration_count || 0)} / ${formatNumber(extended.loop_exact_body_node_evaluation_count || 0)} / ${formatNumber(extended.loop_non_dense_state_variable_count || 0)}`, "neutral"),
    metric("SequenceMap P / ? / F", `${formatNumber(extended.sequence_map_pass_count || 0)} / ${formatNumber(extended.sequence_map_partial_count || 0)} / ${formatNumber(extended.sequence_map_fail_count || 0)}`, extended.sequence_map_fail_count || extended.sequence_map_partial_count ? "warn" : "ok"),
    metric("Recursive engine", extended.status || "not assessed", extended.status === "assessed" ? "ok" : "warn"),
    metric("Recursive executions", `${formatNumber(extended.scope_execution_count || 0)} / ${formatNumber(extended.scope_definition_count || 0)} scopes`, "neutral"),
    metric("Recursive residual N / O", `${formatNumber(extended.residual_unassessed_node_count || 0)} / ${formatNumber(extended.residual_unresolved_output_count || 0)}`, extended.residual_unassessed_node_count || extended.residual_unresolved_output_count ? "warn" : "ok"),
    metric("Type declarations", typeContract.declaration_count || 0, "neutral"),
    metric("Non-dense values", typeContract.non_dense_value_count || 0, "neutral"),
    metric("Type defects", typeContract.invalid_type_count || 0, typeContract.invalid_type_count ? "warn" : "ok"),
    metric("Sparse records", sparseContract.sparse_tensor_count || 0, "neutral"),
    metric("Sparse incomplete", Number(sparseContract.invalid_sparse_tensor_count || 0) + Number(sparseContract.partially_assessed_sparse_tensor_count || 0), sparseContract.status && sparseContract.status !== "assessed" ? "warn" : "ok"),
    metric("Index violations", sparseIndexViolations, sparseIndexViolations ? "warn" : "ok"),
    metric("Container ops", container.assessed_node_count || 0, "neutral"),
    metric("Container partial", container.partially_assessed_node_count || 0, container.partially_assessed_node_count ? "warn" : "ok"),
    metric("Container failures", container.failed_node_count || 0, container.failed_node_count ? "warn" : "ok"),
    metric("Exact length / presence", `${formatNumber(container.exact_sequence_length_output_count || 0)} / ${formatNumber(container.exact_optional_presence_output_count || 0)}`, "neutral"),
    metric("TfIdf P / ? / F", `${formatNumber(tfidf.passed_node_count || 0)} / ${formatNumber(tfidf.partially_assessed_node_count || 0)} / ${formatNumber(tfidf.failed_node_count || 0)}`, tfidf.failed_node_count || tfidf.partially_assessed_node_count ? "warn" : "ok"),
    metric("TfIdf exact static", `${formatNumber(tfidf.exact_static_node_count || 0)} / ${formatNumber(tfidf.assessed_node_count || 0)}`, tfidf.assessed_node_count && tfidf.exact_static_node_count !== tfidf.assessed_node_count ? "warn" : "ok"),
    metric("TfIdf definitions active / total", `${formatNumber(tfidf.exact_active_ngram_definition_count || 0)} / ${formatNumber(tfidf.exact_ngram_definition_count || 0)}`, "neutral"),
    metric("TfIdf matches / output values", tfidf.assessed_node_count ? `${tfidf.exact_match_count == null ? "?" : formatNumber(tfidf.exact_match_count)} / ${tfidf.exact_output_value_count == null ? "?" : formatNumber(tfidf.exact_output_value_count)}` : "N/A", "neutral"),
    metric("TfIdf coordinate aliases", tfidf.assessed_node_count ? tfidf.exact_duplicate_output_coordinate_count ?? "not assessed" : "N/A", tfidf.exact_duplicate_output_coordinate_count ? "warn" : "ok"),
    metric("TfIdf weight / reference divergence", tfidf.assessed_node_count ? `${tfidf.exact_weight_coordinate_value_disagreement_count == null ? "?" : formatNumber(tfidf.exact_weight_coordinate_value_disagreement_count)} / ${tfidf.exact_ort_reference_divergent_output_count == null ? "?" : formatNumber(tfidf.exact_ort_reference_divergent_output_count)}` : "N/A", tfidf.exact_weight_coordinate_value_disagreement_count || tfidf.exact_ort_reference_divergent_output_count ? "warn" : "ok"),
    metric("ML value P / ? / F", `${formatNumber(mlValue.passed_node_count || 0)} / ${formatNumber(mlValue.partially_assessed_node_count || 0)} / ${formatNumber(mlValue.failed_node_count || 0)}`, mlValue.failed_node_count || mlValue.partially_assessed_node_count ? "warn" : "ok"),
    metric("ML exact length / keys", `${formatNumber(mlValue.exact_sequence_length_output_count || 0)} / ${formatNumber(mlValue.exact_class_key_count || 0)}`, "neutral"),
    metric("ML duplicate keys", mlValue.duplicate_class_key_count || 0, mlValue.duplicate_class_key_count ? "warn" : "ok"),
    metric("ML producer / consumer / mapper", `${formatNumber(mlValue.map_producer_node_count || 0)} / ${formatNumber(mlValue.map_consumer_node_count || 0)} / ${formatNumber(mlValue.tensor_mapper_node_count || 0)}`, "neutral"),
    metric("ML exact dense / vocabulary", `${formatNumber(mlValue.exact_dense_output_shape_count || 0)} / ${formatNumber(mlValue.exact_vocabulary_entry_count || 0)}`, "neutral"),
    metric("ML duplicate vocabulary", mlValue.duplicate_vocabulary_entry_count || 0, mlValue.duplicate_vocabulary_entry_count ? "warn" : "ok"),
    metric("ML category pairs", mlValue.exact_category_pair_count || 0, "neutral"),
    metric("ML duplicate active categories", mlValue.duplicate_category_active_key_count || 0, mlValue.duplicate_category_active_key_count ? "warn" : "ok"),
    metric("ML aggregate / select", `${formatNumber(mlValue.tensor_aggregator_node_count || 0)} / ${formatNumber(mlValue.tensor_selector_node_count || 0)}`, "neutral"),
    metric("Feature width exact", `${formatNumber(mlValue.feature_vectorizer_exact_width_node_count || 0)} / ${formatNumber(mlValue.feature_vectorizer_node_count || 0)}`, "neutral"),
    metric("Feature pad / truncate", `${formatNumber(mlValue.exact_feature_vectorizer_padded_feature_count_per_batch || 0)} / ${formatNumber(mlValue.exact_feature_vectorizer_truncated_feature_count_per_batch || 0)}`, mlValue.feature_vectorizer_truncating_node_count ? "warn" : "ok"),
    metric("Array index exact", `${formatNumber(mlValue.array_feature_extractor_exact_index_node_count || 0)} / ${formatNumber(mlValue.array_feature_extractor_node_count || 0)}`, "neutral"),
    metric("Array bounds fail", mlValue.array_feature_extractor_bounds_failure_node_count || 0, mlValue.array_feature_extractor_bounds_failure_node_count ? "warn" : "ok"),
    metric("Binarizer static exact", `${formatNumber(mlValue.binarizer_exact_static_node_count || 0)} / ${formatNumber(mlValue.binarizer_node_count || 0)}`, "neutral"),
    metric("Binarizer one / zero", `${formatNumber(mlValue.exact_binarizer_above_threshold_count || 0)} / ${formatNumber(mlValue.exact_binarizer_at_or_below_threshold_count || 0)}`, mlValue.binarizer_nonfinite_threshold_node_count ? "warn" : "neutral"),
    metric("Normalizer static exact", `${formatNumber(mlValue.normalizer_static_assessed_node_count || 0)} / ${formatNumber(mlValue.normalizer_node_count || 0)}`, "neutral"),
    metric("Normalizer zero / negative MAX", `${formatNumber(mlValue.exact_normalizer_zero_divisor_row_count || 0)} / ${formatNumber(mlValue.exact_normalizer_negative_max_divisor_row_count || 0)}`, mlValue.exact_normalizer_negative_max_divisor_row_count ? "warn" : "neutral"),
    metric("Normalizer cast / s0 / overflow / non-finite", `${formatNumber(mlValue.exact_normalizer_integer_float32_rounding_count || 0)} / ${formatNumber(mlValue.exact_normalizer_signed_zero_output_count || 0)} / ${formatNumber(mlValue.exact_normalizer_signed_overflow_value_count || 0)} / ${formatNumber(mlValue.exact_normalizer_non_finite_output_count || 0)}`, mlValue.exact_normalizer_signed_overflow_value_count || mlValue.exact_normalizer_non_finite_output_count ? "warn" : "neutral"),
    metric("Scaler static exact", `${formatNumber(mlValue.scaler_static_assessed_node_count || 0)} / ${formatNumber(mlValue.scaler_node_count || 0)}`, "neutral"),
    metric("Scaler invalid runtime contract", mlValue.scaler_invalid_runtime_contract_node_count || 0, mlValue.scaler_invalid_runtime_contract_node_count ? "warn" : "ok"),
    metric("Scaler cast / non-finite param / output / s0 / zero-scale", `${formatNumber(mlValue.exact_scaler_integer_float32_rounding_count || 0)} / ${formatNumber(mlValue.exact_scaler_non_finite_parameter_count || 0)} / ${formatNumber(mlValue.exact_scaler_non_finite_output_count || 0)} / ${formatNumber(mlValue.exact_scaler_signed_zero_output_count || 0)} / ${formatNumber(mlValue.exact_scaler_zero_scale_count || 0)}`, mlValue.exact_scaler_non_finite_parameter_count || mlValue.exact_scaler_non_finite_output_count ? "warn" : "neutral"),
    metric("Imputer static exact", `${formatNumber(mlValue.imputer_static_assessed_node_count || 0)} / ${formatNumber(mlValue.imputer_node_count || 0)}`, "neutral"),
    metric("Imputer invalid / fallback / dtype gap", `${formatNumber(mlValue.imputer_invalid_runtime_contract_node_count || 0)} / ${formatNumber(mlValue.imputer_scalar_first_fallback_node_count || 0)} / ${formatNumber(mlValue.imputer_pinned_cpu_dtype_gap_node_count || 0)}`, mlValue.imputer_invalid_runtime_contract_node_count || mlValue.imputer_scalar_first_fallback_node_count || mlValue.imputer_pinned_cpu_dtype_gap_node_count ? "warn" : "ok"),
    metric("Imputer replaced / NaN / ignored / non-finite", `${formatNumber(mlValue.exact_imputer_replacement_count || 0)} / ${formatNumber(mlValue.exact_imputer_nan_replacement_count || 0)} / ${formatNumber(mlValue.exact_imputer_ignored_imputed_value_count || 0)} / ${formatNumber(mlValue.exact_imputer_non_finite_output_count || 0)}`, mlValue.exact_imputer_ignored_imputed_value_count || mlValue.exact_imputer_non_finite_output_count ? "warn" : "neutral"),
    metric("OneHot static exact", `${formatNumber(mlValue.onehot_static_assessed_node_count || 0)} / ${formatNumber(mlValue.onehot_encoder_node_count || 0)}`, "neutral"),
    metric("OneHot invalid / duplicate / fail / dtype", `${formatNumber(mlValue.onehot_invalid_contract_node_count || 0)} / ${formatNumber(mlValue.onehot_duplicate_vocabulary_node_count || 0)} / ${formatNumber(mlValue.onehot_guaranteed_runtime_failure_node_count || 0)} / ${formatNumber(mlValue.onehot_pinned_cpu_dtype_gap_node_count || 0)}`, mlValue.onehot_invalid_contract_node_count || mlValue.onehot_duplicate_vocabulary_node_count || mlValue.onehot_guaranteed_runtime_failure_node_count || mlValue.onehot_pinned_cpu_dtype_gap_node_count ? "warn" : "ok"),
    metric("OneHot matched / unknown / cast / one / zero", `${formatNumber(mlValue.exact_onehot_matched_input_count || 0)} / ${formatNumber(mlValue.exact_onehot_unknown_input_count || 0)} / ${formatNumber(mlValue.exact_onehot_numeric_to_int64_changed_count || 0)} / ${formatNumber(mlValue.exact_onehot_output_one_count || 0)} / ${formatNumber(mlValue.exact_onehot_output_zero_count || 0)}`, mlValue.exact_onehot_unknown_input_count || mlValue.exact_onehot_numeric_to_int64_invalid_count ? "warn" : "neutral"),
    metric("LabelEncoder static / materialized", `${formatNumber(mlValue.label_encoder_static_assessed_node_count || 0)} / ${formatNumber(mlValue.label_encoder_output_materialized_node_count || 0)} / ${formatNumber(mlValue.label_encoder_node_count || 0)}`, "neutral"),
    metric("LabelEncoder ONNX / ORT / dtype / duplicate / NaN", `${formatNumber(mlValue.label_encoder_onnx_contract_failure_node_count || 0)} / ${formatNumber(mlValue.label_encoder_pinned_ort_contract_failure_node_count || 0)} / ${formatNumber(mlValue.label_encoder_pinned_cpu_dtype_pair_gap_node_count || 0)} / ${formatNumber(mlValue.label_encoder_duplicate_semantic_conflict_node_count || 0)} / ${formatNumber(mlValue.label_encoder_nan_semantic_conflict_node_count || 0)}`, mlValue.label_encoder_onnx_contract_failure_node_count || mlValue.label_encoder_pinned_ort_contract_failure_node_count || mlValue.label_encoder_duplicate_semantic_conflict_node_count || mlValue.label_encoder_nan_semantic_conflict_node_count ? "warn" : "ok"),
    metric("LabelEncoder matched / default / mismatch", `${formatNumber(mlValue.exact_label_encoder_match_count || 0)} / ${formatNumber(mlValue.exact_label_encoder_default_count || 0)} / ${formatNumber(mlValue.exact_label_encoder_schema_runtime_mismatch_count || 0)}`, mlValue.exact_label_encoder_default_count || mlValue.exact_label_encoder_schema_runtime_mismatch_count ? "warn" : "neutral"),
    metric("Linear classifier / regressor", `${formatNumber(mlValue.linear_classifier_node_count || 0)} / ${formatNumber(mlValue.linear_regressor_node_count || 0)}`, "neutral"),
    metric("Linear ONNX / ORT / dtype / transform", `${formatNumber(mlValue.linear_onnx_contract_failure_node_count || 0)} / ${formatNumber(mlValue.linear_pinned_ort_contract_failure_node_count || 0)} / ${formatNumber(mlValue.linear_pinned_cpu_dtype_gap_node_count || 0)} / ${formatNumber(mlValue.linear_post_transform_hazard_node_count || 0)}`, mlValue.linear_onnx_contract_failure_node_count || mlValue.linear_pinned_ort_contract_failure_node_count || mlValue.linear_pinned_cpu_dtype_gap_node_count || mlValue.linear_post_transform_hazard_node_count ? "warn" : "ok"),
    metric("Linear coefficients used / ignored / unresolved", `${formatNumber(mlValue.exact_linear_used_coefficient_count || 0)} / ${formatNumber(mlValue.exact_linear_unused_coefficient_count || 0)} / ${formatNumber(mlValue.exact_linear_unresolved_coefficient_use_count || 0)}`, mlValue.exact_linear_unused_coefficient_count || mlValue.exact_linear_unresolved_coefficient_use_count ? "warn" : "neutral"),
    metric("SVM classifier / regressor", `${formatNumber(mlValue.svm_classifier_node_count || 0)} / ${formatNumber(mlValue.svm_regressor_node_count || 0)}`, "neutral"),
    metric("SVM linear / SVC", `${formatNumber(mlValue.svm_linear_mode_node_count || 0)} / ${formatNumber(mlValue.svm_svc_mode_node_count || 0)}`, "neutral"),
    metric("SVM ONNX / ORT / dtype / width", `${formatNumber(mlValue.svm_onnx_contract_failure_node_count || 0)} / ${formatNumber(mlValue.svm_pinned_ort_contract_failure_node_count || 0)} / ${formatNumber(mlValue.svm_regressor_pinned_cpu_dtype_gap_node_count || 0)} / ${formatNumber(mlValue.svm_schema_runtime_score_width_mismatch_node_count || 0)}`, mlValue.svm_onnx_contract_failure_node_count || mlValue.svm_pinned_ort_contract_failure_node_count || mlValue.svm_regressor_pinned_cpu_dtype_gap_node_count || mlValue.svm_schema_runtime_score_width_mismatch_node_count ? "warn" : "ok"),
    metric("SVM support used / ignored / unresolved", `${formatNumber(mlValue.exact_svm_used_support_vector_value_count || 0)} / ${formatNumber(mlValue.exact_svm_unused_support_vector_value_count || 0)} / ${formatNumber(mlValue.exact_svm_unresolved_support_vector_use_count || 0)}`, mlValue.exact_svm_unused_support_vector_value_count || mlValue.exact_svm_unresolved_support_vector_use_count ? "warn" : "neutral"),
    metric("SVM coeff used / ignored / unresolved", `${formatNumber(mlValue.exact_svm_used_coefficient_count || 0)} / ${formatNumber(mlValue.exact_svm_unused_coefficient_count || 0)} / ${formatNumber(mlValue.exact_svm_unresolved_coefficient_use_count || 0)}`, mlValue.exact_svm_unused_coefficient_count || mlValue.exact_svm_unresolved_coefficient_use_count ? "warn" : "neutral"),
    metric("SVM ignored transform / non-finite", `${formatNumber(mlValue.svm_ignored_post_transform_node_count || 0)} / ${formatNumber(mlValue.svm_non_finite_node_count || 0)}`, mlValue.svm_ignored_post_transform_node_count || mlValue.svm_non_finite_node_count ? "warn" : "ok"),
    metric("Tree v5 / classifier / regressor", `${formatNumber(mlValue.tree_ensemble_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_classifier_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_regressor_node_count || 0)}`, "neutral"),
    metric("Tree ONNX / ORT / dtype / deprecated", `${formatNumber(mlValue.tree_ensemble_onnx_contract_failure_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_pinned_ort_contract_failure_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_pinned_cpu_dtype_gap_node_count || 0)} / ${formatNumber(mlValue.tree_ensemble_deprecated_node_count || 0)}`, mlValue.tree_ensemble_onnx_contract_failure_node_count || mlValue.tree_ensemble_pinned_ort_contract_failure_node_count || mlValue.tree_ensemble_deprecated_node_count ? "warn" : "ok"),
    metric("Tree / node / leaf / max depth", `${formatNumber(mlValue.exact_tree_ensemble_tree_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_node_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_leaf_count || 0)} / ${formatNumber(mlValue.maximum_tree_ensemble_depth || 0)}`, "neutral"),
    metric("Tree reachable / orphan / cycle", `${formatNumber(mlValue.exact_tree_ensemble_reachable_node_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_orphan_node_or_leaf_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_cycle_count || 0)}`, mlValue.exact_tree_ensemble_orphan_node_or_leaf_count || mlValue.exact_tree_ensemble_cycle_count ? "warn" : "ok"),
    metric("Tree weights used / unused / unresolved", `${formatNumber(mlValue.exact_tree_ensemble_used_weight_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_unused_weight_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_unresolved_weight_count || 0)}`, mlValue.exact_tree_ensemble_unused_weight_count || mlValue.exact_tree_ensemble_unresolved_weight_count ? "warn" : "neutral"),
    metric("Tree MEMBER sets / values / duplicates", `${formatNumber(mlValue.exact_tree_ensemble_membership_set_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_membership_value_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_membership_duplicate_value_count || 0)}`, mlValue.exact_tree_ensemble_membership_duplicate_value_count ? "warn" : "neutral"),
    metric("Tree reference nodes / paths / boundary / unwritten", `${formatNumber(mlValue.tree_ensemble_reference_assessed_node_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_reference_path_step_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_reference_decision_boundary_count || 0)} / ${formatNumber(mlValue.exact_tree_ensemble_reference_unwritten_score_count || 0)}`, mlValue.exact_tree_ensemble_reference_decision_boundary_count || mlValue.exact_tree_ensemble_reference_unwritten_score_count ? "warn" : "neutral"),
  );

  const domainTable = dataTable(
    ["Domain", "Opset", "Nodes (main/nested/function)", "Definitions", "Resolution", "Operators"],
    domains.map((domain) => {
      const resolution = (domain.resolution_classes || []).join(" / ") || (domain.unused_import ? "unused import" : "unresolved");
      return [
      domain.domain,
      domain.imported_opset ?? "missing",
      `${formatNumber(domain.node_count)} (${domain.main_graph_node_count}/${domain.nested_graph_node_count}/${domain.function_body_node_count})`,
      formatNumber(domain.local_function_definition_count || 0),
      resolution.replaceAll("_", " "),
      (domain.op_types || []).join(", ") || "-",
      ];
    }),
  );
  const sections = [section("Imported Domain Inventory", domainTable)];

  const containerRows = container.rows || [];
  if (containerRows.length) {
    const shown = containerRows.slice(0, 200);
    sections.push(section(`Sequence / Optional Value Contracts (${formatNumber(shown.length)} of ${formatNumber(containerRows.length)})`, dataTable(
      ["Scope / node", "Opset", "Status", "Output TypeProto", "Sequence length", "Optional presence", "Result"],
      shown.map((row) => [
        `${row.scope} / #${row.node_index} ${row.op_name}`,
        row.imported_opset ?? "missing",
        row.status,
        (row.canonical_output_types || []).join(" / ") || "unresolved",
        (row.sequence_lengths || []).map((value) => value == null ? "runtime unknown" : formatNumber(value)).join(" / ") || "N/A",
        (row.optional_presence || []).map((value) => value == null ? "runtime unknown" : value ? "present" : "empty").join(" / ") || "N/A",
        (row.reason_codes || []).join(" / ") || "pass",
      ]),
    )));
  }

  const tfidfRows = tfidf.rows || [];
  if (tfidfRows.length) {
    const shown = tfidfRows.slice(0, 200);
    sections.push(section(`TfIdfVectorizer-9 Contracts (${formatNumber(shown.length)} of ${formatNumber(tfidfRows.length)})`, dataTable(
      ["Scope / node", "Status", "Input -> output", "Gram / pool contract", "Coordinate contract", "Exact static result", "Result / risk"],
      shown.map((row) => [
        `${row.scope || "main_graph"} / #${row.node_index} ${row.op_name || "TfIdfVectorizer"}; opset ${row.imported_opset ?? "missing"}`,
        row.status || "not assessed",
        `${row.input_name || "unnamed"} ${row.input_dtype || "UNKNOWN"} ${JSON.stringify(row.input_shape || [])} -> ${row.output_name || "unnamed"} FLOAT32 ${JSON.stringify(row.exact_output_shape || [])}`,
        `${row.mode || "?"}; gram ${row.minimum_gram_length ?? "?"}-${row.maximum_gram_length ?? "?"}; skip <= ${row.maximum_skip_count ?? "?"}; ${row.pool_kind || "?"} pool ${row.exact_pool_item_count ?? "?"}; definitions ${row.exact_active_ngram_definition_count ?? "?"} active / ${row.exact_ngram_definition_count ?? "?"} total; unused prefix ${row.exact_unused_pool_prefix_item_count ?? "?"}`,
        `width/indexes ${row.exact_output_width ?? "?"} / ${row.exact_ngram_index_count ?? "?"}; aliases ${row.exact_duplicate_output_coordinate_count ?? "?"}; unaddressed ${row.exact_unaddressed_output_coordinate_count ?? "?"}; weights ${row.weights_present ? row.exact_weight_count ?? "?" : "implicit 1"}; mapping disagreement ${row.exact_weight_coordinate_value_disagreement_count ?? "?"}`,
        row.static_execution_status === "assessed_exact"
          ? `${row.static_execution_status}; work ${row.exact_static_work_units ?? "?"}; matches ${row.exact_match_count ?? "?"}; nonzero frequency/output ${row.exact_nonzero_frequency_count ?? "?"} / ${row.exact_nonzero_output_count ?? "?"}; signed zero ${row.exact_negative_zero_output_count ?? "?"}; ORT/reference divergence ${row.exact_ort_reference_divergent_output_count ?? "?"}; output ${JSON.stringify(row.exact_output_values || [])}`
          : `${row.static_execution_status || "not assessed"}; exact static output unavailable`,
        [...(row.reason_codes || []), ...(row.risk_codes || [])].filter((value, index, values) => values.indexOf(value) === index).join(" / ") || "pass",
      ]),
    )));
  }

  const mlValueRows = mlValue.rows || [];
  if (mlValueRows.length) {
    const shown = mlValueRows.slice(0, 200);
    sections.push(section(`ONNX-ML Value Contracts (${formatNumber(shown.length)} of ${formatNumber(mlValueRows.length)})`, dataTable(
      ["Scope / node", "Status", "Input contract", "Attributes / keys", "Output contract", "Result / risk"],
      shown.map((row) => [
        `${row.scope} / #${row.node_index} ${row.op_name}; opset ${row.imported_opset ?? "missing"}`,
        row.status,
        row.op_name === "FeatureVectorizer"
          ? `${(row.input_names || []).length} inputs; dtypes ${(row.input_dtypes || []).join(" / ") || "unresolved"}; shapes ${(row.input_shapes || []).map((shape) => JSON.stringify(shape)).join(" / ") || "none"}`
          : row.op_name === "ArrayFeatureExtractor"
            ? `${row.input_name || "unnamed"}; ${row.input_dtype || "UNKNOWN"} ${JSON.stringify(row.input_shape || [])}; indices ${row.index_input_name || "unnamed"} ${row.index_input_dtype || "UNKNOWN"} ${JSON.stringify(row.index_input_shape || [])}`
            : row.input_kind === "map"
          ? `${row.input_name || "unnamed"}; map<${row.input_map_key_type || "UNDEFINED"}, ${row.input_map_value_dtype || "UNKNOWN"}>; exact entries ${row.exact_input_map_key_count == null ? "not encoded" : formatNumber(row.exact_input_map_key_count)}`
          : `${row.input_name || "unnamed"}; ${row.input_dtype || "UNKNOWN"}; rank ${row.input_rank ?? "?"} ${JSON.stringify(row.input_shape || [])}`,
        row.op_name === "Binarizer"
          ? `threshold ${row.threshold_value_text || "not emitted"} (${row.threshold_source || "not emitted"}); static ${row.static_value_assessment_status || "not assessed"}; exact ${row.exact_static_input_value_count == null ? "runtime unknown" : formatNumber(row.exact_static_input_value_count)}, above ${row.exact_above_threshold_count == null ? "runtime unknown" : formatNumber(row.exact_above_threshold_count)}, at/below ${row.exact_at_or_below_threshold_count == null ? "runtime unknown" : formatNumber(row.exact_at_or_below_threshold_count)}, equal ${row.exact_equal_threshold_count == null ? "runtime unknown" : formatNumber(row.exact_equal_threshold_count)}`
          : row.op_name === "Normalizer"
            ? `${row.normalizer_mode || "unresolved"} (${row.normalizer_mode_source || "not emitted"}); ${row.normalizer_divisor_kind || "divisor unresolved"}; static ${row.normalizer_static_assessment_status || "not assessed"}; rows ${row.normalizer_batch_count == null ? "?" : formatNumber(row.normalizer_batch_count)} x ${row.normalizer_row_width == null ? "?" : formatNumber(row.normalizer_row_width)}; divisors ${(row.normalizer_divisor_preview || []).join(", ") || "runtime values"}`
          : row.op_name === "Scaler"
            ? `${row.scaler_parameter_mode || "unresolved"}; contract ${row.scaler_parameter_contract_status || "not assessed"} (${row.scaler_parameter_contract_reason || "not emitted"}); stride ${row.scaler_feature_stride ?? "?"}; scale x${formatNumber(row.scaler_scale_count || 0)} [${(row.scaler_scale_values || []).slice(0, 8).join(", ") || "none"}${(row.scaler_scale_values || []).length > 8 ? `, plus ${formatNumber(row.scaler_scale_values.length - 8)} more` : ""}]; offset x${formatNumber(row.scaler_offset_count || 0)} [${(row.scaler_offset_values || []).slice(0, 8).join(", ") || "none"}${(row.scaler_offset_values || []).length > 8 ? `, plus ${formatNumber(row.scaler_offset_values.length - 8)} more` : ""}]; static ${row.scaler_static_assessment_status || "not assessed"}`
          : row.op_name === "Imputer"
            ? `${row.imputer_parameter_mode || "unresolved"}; contract ${row.imputer_parameter_contract_status || "not assessed"} (${row.imputer_parameter_contract_reason || "not emitted"}); ${row.imputer_attribute_kind || "unresolved"}; stride ${row.imputer_feature_stride ?? "?"}; imputed x${formatNumber(row.imputer_imputed_value_count || 0)} [${(row.imputer_imputed_values || []).slice(0, 8).join(", ") || "none"}${(row.imputer_imputed_values || []).length > 8 ? `, plus ${formatNumber(row.imputer_imputed_values.length - 8)} more` : ""}]; replace ${row.imputer_replaced_value || "?"} (${row.imputer_replaced_value_source || "not emitted"}); static ${row.imputer_static_assessment_status || "not assessed"}`
          : row.op_name === "OneHotEncoder"
            ? `${row.onehot_category_kind || "unresolved"} vocabulary x${formatNumber(row.onehot_category_count || 0)} [${(row.onehot_category_values || []).slice(0, 8).join(", ") || "none"}${(row.onehot_category_values || []).length > 8 ? `, plus ${formatNumber(row.onehot_category_values.length - 8)} more` : ""}]; contract ${row.onehot_parameter_contract_status || "not assessed"} (${row.onehot_parameter_contract_reason || "not emitted"}); duplicate/unreachable ${formatNumber(row.onehot_duplicate_category_count || 0)} / ${formatNumber(row.onehot_unreachable_duplicate_column_count || 0)} [${(row.onehot_unreachable_duplicate_column_indices || []).join(", ") || "none"}]; zeros=${row.onehot_zeros_value || "?"} (${row.onehot_zeros_source || "not emitted"}); static ${row.onehot_static_assessment_status || "not assessed"}`
          : row.op_name === "LabelEncoder"
            ? `v${row.resolved_schema_version || "?"} ${row.label_encoder_key_dtype || "UNKNOWN"}->${row.label_encoder_value_dtype || "UNKNOWN"}; keys/values ${formatNumber(row.label_encoder_key_count || 0)} / ${formatNumber(row.label_encoder_value_count || 0)}; keys [${(row.label_encoder_key_values || []).slice(0, 8).join(", ") || "none"}]; values [${(row.label_encoder_value_values || []).slice(0, 8).join(", ") || "none"}]; duplicate/NaN/non-finite keys/values ${formatNumber(row.label_encoder_duplicate_key_count || 0)} / ${formatNumber(row.label_encoder_nan_key_count || 0)} / ${formatNumber(row.label_encoder_non_finite_key_count || 0)} / ${formatNumber(row.label_encoder_non_finite_value_count || 0)}; ownership ${row.label_encoder_runtime_duplicate_policy || "?"} runtime / ${row.label_encoder_schema_duplicate_policy || "?"} schema; default ${row.label_encoder_default_value ?? "?"} (${row.label_encoder_default_source || "?"}); ONNX/ORT ${row.label_encoder_onnx_contract_status || "?"} / ${row.label_encoder_pinned_ort_contract_status || "?"}`
          : row.op_name === "LinearClassifier"
            ? `classes ${row.linear_class_or_target_count ?? "?"}; labels ${row.linear_label_kind || "unresolved"} x${formatNumber(row.linear_label_count || 0)} [${(row.linear_label_values || []).slice(0, 8).join(", ") || "none"}]; coefficients ${formatNumber(row.linear_used_coefficient_count || 0)} used / ${formatNumber(row.linear_coefficient_count || 0)} serialized / ${formatNumber(row.linear_unused_coefficient_count || 0)} ignored; intercepts ${formatNumber(row.linear_intercept_count || 0)}; multi_class=${row.linear_multi_class_value || "?"} (${row.linear_multi_class_used_by_pinned_ort ? "used" : "not consulted"}); transform ${row.linear_post_transform || "unresolved"}; ONNX/ORT ${row.linear_onnx_contract_status || "?"} / ${row.linear_pinned_ort_contract_status || "?"}`
          : row.op_name === "LinearRegressor"
            ? `targets ${row.linear_targets_value || "?"} (${row.linear_targets_source || "not emitted"}); coefficients ${formatNumber(row.linear_used_coefficient_count || 0)} used / ${formatNumber(row.linear_coefficient_count || 0)} serialized / ${formatNumber(row.linear_unused_coefficient_count || 0)} ignored; intercepts ${formatNumber(row.linear_intercept_count || 0)} (${row.linear_intercepts_used ? "used" : `${formatNumber(row.linear_ignored_intercept_count || 0)} ignored`}); transform ${row.linear_post_transform || "unresolved"}; ONNX/ORT ${row.linear_onnx_contract_status || "?"} / ${row.linear_pinned_ort_contract_status || "?"}`
          : row.op_name === "SVMClassifier"
            ? `${row.svm_mode || "unresolved"} ${row.svm_kernel_type || "UNRESOLVED"}; kernel_params ${(row.svm_kernel_params || []).join(", ") || "explicit empty"} (${row.svm_kernel_params_source || "not emitted"}); classes ${formatNumber(row.svm_class_label_count || 0)} [${(row.svm_class_label_values || []).slice(0, 8).join(", ") || "none"}], duplicate ${formatNumber(row.svm_duplicate_label_count || 0)}; vectors ${formatNumber(row.svm_vector_count || 0)} [${(row.svm_vectors_per_class || []).join(", ") || "linear"}], pairs ${formatNumber(row.svm_pairwise_classifier_count || 0)}; score width schema/ORT ${row.svm_schema_score_width ?? "?"} / ${row.svm_pinned_ort_score_width ?? "?"}${row.svm_schema_runtime_score_width_mismatch ? " CONFLICT" : ""}; support/coeff/rho expected-used-serialized ${row.svm_expected_support_vector_value_count ?? "?"}-${row.svm_used_support_vector_value_count ?? "?"}-${row.svm_support_vector_value_count ?? "?"} / ${row.svm_expected_coefficient_count ?? "?"}-${row.svm_used_coefficient_count ?? "?"}-${row.svm_coefficient_count ?? "?"} / ${row.svm_expected_rho_count ?? "?"}-${row.svm_used_rho_count ?? "?"}-${row.svm_rho_count ?? "?"}; transform ${row.svm_post_transform || "?"}; ONNX/ORT ${row.svm_onnx_contract_status || "?"} / ${row.svm_pinned_ort_contract_status || "?"}`
          : row.op_name === "SVMRegressor"
            ? `${row.svm_mode || "unresolved"} ${row.svm_kernel_type || "UNRESOLVED"}; kernel_params ${(row.svm_kernel_params || []).join(", ") || "explicit empty"} (${row.svm_kernel_params_source || "not emitted"}); n_supports/vectors ${row.svm_n_supports ?? "?"} / ${row.svm_vector_count ?? "?"}; one_class ${row.svm_one_class_value ?? "?"}; support/coeff/rho expected-used-serialized ${row.svm_expected_support_vector_value_count ?? "?"}-${row.svm_used_support_vector_value_count ?? "?"}-${row.svm_support_vector_value_count ?? "?"} / ${row.svm_expected_coefficient_count ?? "?"}-${row.svm_used_coefficient_count ?? "?"}-${row.svm_coefficient_count ?? "?"} / ${row.svm_expected_rho_count ?? "?"}-${row.svm_used_rho_count ?? "?"}-${row.svm_rho_count ?? "?"}; transform ${row.svm_post_transform || "?"} (${row.svm_post_transform_applied_by_pinned_ort ? "applied" : "not applied"}); ONNX/ORT ${row.svm_onnx_contract_status || "?"} / ${row.svm_pinned_ort_contract_status || "?"}`
          : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
            ? `${row.tree_encoding || "unresolved"}${row.tree_deprecated_operator ? " DEPRECATED" : ""}; aggregate/transform ${row.tree_aggregate_function || "?"}/${row.tree_post_transform || "?"}; ONNX/ORT ${row.tree_onnx_contract_status || "?"} (${row.tree_onnx_contract_reason || "pass"}) / ${row.tree_pinned_ort_contract_status || "?"} (${row.tree_pinned_ort_contract_reason || "pass"}); targets/classes ${formatNumber(row.tree_class_or_target_count || 0)}, labels ${row.tree_class_label_kind || "N/A"} x${formatNumber(row.tree_class_label_count || 0)}, duplicate ${formatNumber(row.tree_duplicate_class_label_count || 0)}; trees/roots/nodes/branches/leaves ${formatNumber(row.tree_exact_tree_count || 0)}/${formatNumber(row.tree_exact_root_count || 0)}/${formatNumber(row.tree_exact_node_count || 0)}/${formatNumber(row.tree_exact_branch_node_count || 0)}/${formatNumber(row.tree_exact_leaf_count || 0)}; reachable node/leaf ${formatNumber(row.tree_reachable_node_count || 0)}/${formatNumber(row.tree_reachable_leaf_count || 0)}, orphan/depth/cycle ${formatNumber(row.tree_orphan_node_or_leaf_count || 0)}/${formatNumber(row.tree_max_depth || 0)}/${formatNumber(row.tree_cycle_count || 0)}; duplicate/invalid-child/invalid-feature/root-mismatch/multiple-parent ${formatNumber(row.tree_duplicate_node_identity_count || 0)}/${formatNumber(row.tree_invalid_child_reference_count || 0)}/${formatNumber(row.tree_invalid_feature_id_count || 0)}/${formatNumber(row.tree_root_mismatch_count || 0)}/${formatNumber(row.tree_multiple_parent_node_count || 0)}; weights used/unused/unresolved/serialized ${formatNumber(row.tree_used_weight_count || 0)}/${formatNumber(row.tree_unused_weight_count || 0)}/${formatNumber(row.tree_unresolved_weight_count || 0)}/${formatNumber(row.tree_weight_tuple_count || 0)}; MEMBER sets/values/duplicates/separators ${formatNumber(row.tree_membership_set_count || 0)}/${formatNumber(row.tree_membership_value_count || 0)}/${formatNumber(row.tree_membership_duplicate_value_count || 0)}/${formatNumber(row.tree_membership_separator_count || 0)}`
          : row.op_name === "ZipMap"
            ? `class labels ${row.class_key_type || "UNDEFINED"} x${formatNumber(row.class_key_count || 0)}; duplicates ${formatNumber(row.duplicate_key_count || 0)}; ${(row.class_key_preview || []).join(", ") || "none"}; features ${row.exact_feature_count == null ? "runtime unknown" : formatNumber(row.exact_feature_count)}`
          : row.op_name === "CastMap"
            ? `cast_to ${row.cast_to || "TO_FLOAT"}; map_form ${row.map_form || "DENSE"}; max_map ${row.max_map == null ? "not exactly decoded" : formatNumber(row.max_map)}; sparse key bounds ${row.sparse_key_bounds_status || "not emitted"}`
            : row.op_name === "DictVectorizer"
              ? `vocabulary ${row.vocabulary_type || "UNDEFINED"} x${formatNumber(row.vocabulary_count || 0)}; duplicates ${formatNumber(row.duplicate_vocabulary_count || 0)}; ${(row.vocabulary_preview || []).join(", ") || "none"}`
              : row.op_name === "CategoryMapper"
                ? `${row.mapping_direction || "UNRESOLVED"}; ${formatNumber(row.category_pair_count || 0)} pair(s), arrays ${formatNumber(row.category_string_count || 0)} string / ${formatNumber(row.category_int64_count || 0)} int64; duplicate string/int64/active ${formatNumber(row.duplicate_string_key_count || 0)} / ${formatNumber(row.duplicate_int64_key_count || 0)} / ${formatNumber(row.active_duplicate_key_count || 0)}; default ${row.active_default_type || "UNDEFINED"} ${row.active_default_value ?? ""}; strings ${(row.category_string_preview || []).join(", ") || "none"}; int64 ${(row.category_int64_preview || []).join(", ") || "none"}`
                : row.op_name === "FeatureVectorizer"
                  ? `widths ${(row.configured_feature_dimensions || []).join(" / ") || "none"} -> ${row.total_configured_feature_count == null ? "?" : formatNumber(row.total_configured_feature_count)}; row ${(row.exact_input_row_feature_counts || []).map((value) => value == null ? "?" : formatNumber(value)).join(" / ") || "none"}; copy/pad/truncate ${row.exact_copied_feature_count_per_batch ?? "?"} / ${row.exact_padded_feature_count_per_batch ?? "?"} / ${row.exact_truncated_feature_count_per_batch ?? "?"}`
                  : `indices ${row.exact_index_count == null ? "?" : formatNumber(row.exact_index_count)}; ${(row.exact_index_preview || []).join(", ") || "runtime values"}; duplicates ${formatNumber(row.duplicate_index_count || 0)}; bounds ${row.index_bounds_status || "not assessed"}; invalid ${formatNumber(row.out_of_bounds_index_count || 0)}`,
        ["LinearClassifier", "LinearRegressor", "SVMClassifier", "TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
          ? (row.output_names || []).map((name, index) => `${name || "unnamed"} ${(row.canonical_output_types || [])[index] || "unresolved"} ${JSON.stringify((row.canonical_output_shapes || [])[index] || [])}`).join(" / ") || "unresolved"
          : row.output_kind === "sequence"
          ? `${row.output_name || "unnamed"}; sequence ${row.exact_output_sequence_length == null ? "runtime unknown" : formatNumber(row.exact_output_sequence_length)}; ${row.canonical_output_type || "unresolved"}; ${row.output_shape_basis || "basis not emitted"}; ${row.runtime_reference_status || "runtime reference not emitted"}`
          : `${row.output_name || "unnamed"}; ${row.output_dtype || "UNKNOWN"}; rank ${row.exact_output_rank ?? "?"} ${JSON.stringify(row.exact_output_shape || [])}; elements ${row.exact_dense_output_element_count == null ? "runtime unknown" : formatNumber(row.exact_dense_output_element_count)}; ${row.canonical_output_type || "unresolved"}; ${row.output_shape_basis || "basis not emitted"}; ${row.runtime_reference_status || "runtime reference not emitted"}`,
        row.op_name === "Normalizer"
          ? `zero ${row.normalizer_zero_divisor_row_count ?? "?"}; negative MAX ${row.normalizer_negative_max_divisor_row_count ?? "?"}; cast ${row.normalizer_integer_float32_rounding_count ?? "?"}; signed zero ${row.normalizer_signed_zero_output_count ?? "?"}; overflow ${row.normalizer_signed_overflow_value_count ?? "?"}; non-finite ${row.normalizer_non_finite_output_count ?? "?"}; output ${row.normalizer_output_materialized ? "materialized" : "aggregate/preview"} ${(row.normalizer_output_preview || []).join(", ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "Scaler"
            ? `zero scale ${row.scaler_zero_scale_count ?? "?"}; cast ${row.scaler_integer_float32_rounding_count ?? "?"}; non-finite param/output ${row.scaler_non_finite_parameter_count ?? "?"} / ${row.scaler_non_finite_output_count ?? "?"}; signed zero ${row.scaler_signed_zero_output_count ?? "?"}; output ${row.scaler_output_materialized ? "materialized" : "aggregate/preview"} ${(row.scaler_output_preview || []).join(", ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "Imputer"
            ? `replaced ${row.imputer_exact_replacement_count ?? "?"}; NaN ${row.imputer_exact_nan_replacement_count ?? "?"}; unchanged ${row.imputer_exact_unchanged_count ?? "?"}; ignored ${row.imputer_ignored_imputed_value_count ?? "?"}; non-finite value/output ${row.imputer_non_finite_imputed_value_count ?? "?"} / ${row.imputer_non_finite_output_count ?? "?"}; signed zero ${row.imputer_signed_zero_output_count ?? "?"}; output ${row.imputer_output_materialized ? "materialized" : "aggregate/preview"} ${(row.imputer_output_preview || []).join(", ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "OneHotEncoder"
            ? `input/matched/unknown ${row.onehot_exact_input_value_count ?? "?"} / ${row.onehot_exact_matched_input_count ?? "?"} / ${row.onehot_exact_unknown_input_count ?? "?"}; cast changed/invalid ${row.onehot_numeric_to_int64_changed_count ?? "?"} / ${row.onehot_numeric_to_int64_invalid_count ?? "?"}; one/zero ${row.onehot_exact_output_one_count ?? "?"} / ${row.onehot_exact_output_zero_count ?? "?"}; runtime failure ${row.onehot_guaranteed_runtime_failure ? "yes" : "no"}; unknowns ${(row.onehot_unknown_input_preview || []).join(", ") || "none"}; output ${row.onehot_output_materialized ? "materialized" : "aggregate/preview"} ${(row.onehot_output_preview || []).join(", ") || "none"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "LabelEncoder"
            ? `input/matched/default ${row.label_encoder_exact_input_value_count ?? "?"} / ${row.label_encoder_exact_match_count ?? "?"} / ${row.label_encoder_exact_default_count ?? "?"}; duplicate hits ${row.label_encoder_exact_duplicate_key_hit_count ?? "?"}; schema/runtime mismatches ${row.label_encoder_schema_runtime_mismatch_count ?? "?"}; runtime ${(row.label_encoder_runtime_output_preview || []).join(", ") || "runtime values"}; schema ${(row.label_encoder_schema_output_preview || []).join(", ") || "same or unassessed"}; output ${row.label_encoder_output_materialized ? "materialized" : "not propagated/preview"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "LinearClassifier" || row.op_name === "LinearRegressor"
            ? `scalar FLOAT32 reference ${row.linear_reference_assessment_status || "not assessed"}; input/raw ${row.linear_reference_input_value_count ?? "?"} / ${row.linear_reference_raw_score_count ?? "?"}; raw ${(row.linear_reference_raw_score_preview || []).join(", ") || "none"}; transformed ${(row.linear_reference_output_preview || []).join(", ") || "none"}; labels ${(row.linear_reference_label_preview || []).join(", ") || "none"}; non-finite param/score ${row.linear_non_finite_parameter_count ?? "?"} / ${row.linear_reference_non_finite_raw_score_count ?? "?"}; boundary decisions ${row.linear_reference_decision_boundary_count ?? "N/A"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : row.op_name === "SVMClassifier" || row.op_name === "SVMRegressor"
            ? `scalar FLOAT32 reference ${row.svm_reference_assessment_status || "not assessed"}; input/raw ${row.svm_reference_input_value_count ?? "?"} / ${row.svm_reference_raw_score_count ?? "?"}; raw ${(row.svm_reference_raw_score_preview || []).join(", ") || "none"}; scores ${(row.svm_reference_output_score_preview || []).join(", ") || "none"}; labels ${(row.svm_reference_label_preview || []).join(", ") || "none"}; non-finite param/score ${row.svm_non_finite_parameter_count ?? "?"} / ${row.svm_reference_non_finite_score_count ?? "?"}; boundary ${row.svm_reference_decision_boundary_count ?? "N/A"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
            ? `source-order reference ${row.tree_reference_assessment_status || "not assessed"}; rows/input/path/raw/output ${row.tree_reference_row_count ?? "?"} / ${row.tree_reference_input_value_count ?? "?"} / ${row.tree_reference_path_step_count ?? "?"} / ${row.tree_reference_raw_score_count ?? "?"} / ${row.tree_reference_output_score_count ?? "?"}; raw ${(row.tree_reference_raw_score_preview || []).join(", ") || "none"}; scores ${(row.tree_reference_output_score_preview || []).join(", ") || "none"}; labels ${(row.tree_reference_label_preview || []).join(", ") || "none"}; non-finite param/score ${row.tree_non_finite_parameter_count ?? "?"} / ${row.tree_reference_non_finite_score_count ?? "?"}; boundary/unwritten ${row.tree_reference_decision_boundary_count ?? "?"} / ${row.tree_reference_unwritten_score_count ?? "?"}; ${[...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass"}`
          : [...(row.reason_codes || []), ...(row.risk_codes || [])].join(" / ") || "pass",
      ]),
    )));
  }

  if (sequenceMaps.length) {
    const shown = sequenceMaps.slice(0, 200);
    sections.push(section(`SequenceMap Value Contracts (${formatNumber(shown.length)} of ${formatNumber(sequenceMaps.length)})`, dataTable(
      ["Scope / node", "Opset", "Status", "Exact input length", "Element expansion work", "Result"],
      shown.map((row) => [
        `${row.scope} / #${row.node_index}`,
        row.imported_opset ?? "missing",
        row.status,
        row.exact_input_sequence_length == null ? "runtime unknown" : formatNumber(row.exact_input_sequence_length),
        `${formatNumber(row.element_expansion_count || 0)} element(s) / ${formatNumber(row.element_node_evaluation_count || 0)} body-node evaluation(s)`,
        (row.reason_codes || []).join(" / ") || row.body_status || "pass",
      ]),
    )));
  }

  if (typeRows.length) {
    const shown = typeRows.slice(0, 200);
    sections.push(section(`TypeProto Declarations (${formatNumber(shown.length)} of ${formatNumber(typeRows.length)})`, dataTable(
      ["Scope", "Role", "Value", "Kind", "Canonical type", "Status", "Reason"],
      shown.map((row) => [
        row.scope,
        row.role,
        row.value_name,
        row.kind,
        row.canonical_type,
        row.status,
        (row.reason_codes || []).join(" / ") || "valid",
      ]),
    )));
  }

  if (sparseRows.length) {
    const shown = sparseRows.slice(0, 200);
    sections.push(section(`SparseTensorProto Records (${formatNumber(shown.length)} of ${formatNumber(sparseRows.length)})`, dataTable(
      ["Scope", "Role / name", "Dense shape", "NNZ", "Index encoding", "Index content", "Payload", "Status / reason"],
      shown.map((row) => [
        row.scope,
        `${row.tensor_role} / ${row.sparse_tensor_name || "unnamed"}`,
        (row.dense_shape || []).join("x") || "scalar",
        row.nnz == null ? "unresolved" : formatNumber(row.nnz),
        row.index_encoding,
        `${row.index_content_status}; ${formatNumber(row.assessed_index_count || 0)} decoded, ${formatNumber(Number(row.out_of_bounds_index_count || 0) + Number(row.duplicate_index_count || 0) + Number(row.unsorted_index_count || 0))} violation(s)`,
        `${row.payload_status}; ${formatNumber(row.embedded_payload_bytes || 0)} embedded B; ${formatNumber(row.verified_external_payload_component_count || 0)}/${formatNumber(row.external_payload_component_count || 0)} external verified`,
        `${row.status}${(row.reason_codes || []).length ? ` / ${row.reason_codes.join(" / ")}` : ""}`,
      ]),
    )));
  }

  if (functions.length) {
    sections.push(section("Model-local FunctionProto Registry", dataTable(
      ["Function", "Body nodes", "Local dependencies", "Recursive"],
      functions.map((fn) => [
        fn.id,
        formatNumber(fn.body_node_count || 0),
        (fn.local_function_dependencies || []).join(" / ") || "none",
        fn.recursive_cycle ? "yes" : "no",
      ]),
    )));
  }

  if (externalNodes.length) {
    sections.push(section("External Custom-op Registry Requirements", dataTable(
      ["Scope", "Node", "Domain / opset", "Operator", "Top-level index"],
      externalNodes.map((node) => [
        node.scope,
        node.node_name || "-",
        `${node.domain} / ${node.imported_opset ?? "missing"}`,
        node.op_name,
        node.top_level_op_index ?? "nested/function",
      ]),
    )));
  }

  if (duplicateCount || cycleCount) {
    const registryAlert = document.createElement("p");
    registryAlert.className = "onnx-domain-registry-alert";
    registryAlert.textContent = `Function registry is ${evidence.status}: ${formatNumber(duplicateCount)} duplicate ID(s), ${formatNumber(cycleCount)} recursive cycle(s).`;
    sections.push(registryAlert);
  }

  if (schemaFailures.length) {
    const shown = schemaFailures.slice(0, 200);
    sections.push(section(`OpSchema Formal Contract Failures (${formatNumber(shown.length)} of ${formatNumber(schemaFailures.length)})`, dataTable(
      ["Node", "Imported opset", "Resolved schema", "Status", "Reason"],
      shown.map((row) => [
        `#${row.node_index} ${row.op_name || "UNKNOWN"}`,
        row.imported_opset ?? "missing",
        row.schema_since_version == null ? "none" : `${row.op_name}-${row.schema_since_version}`,
        row.status,
        (row.reason_codes || []).join(" / ") || row.detail,
      ]),
    )));
  }

  if (scopeExclusions.length) {
    const shown = scopeExclusions.slice(0, 200);
    sections.push(section(`Extended Shape Scope Exclusions (${formatNumber(shown.length)} of ${formatNumber(scopeExclusions.length)})`, dataTable(
      ["Class", "Scope", "Nodes", "Reason"],
      shown.map((row) => [
        row.scope_class,
        row.scope,
        formatNumber(row.node_count || 0),
        row.reason_code,
      ]),
    )));
  }

  if (scopeAssessments.length) {
    const shown = scopeAssessments.slice(0, 200);
    sections.push(section(`Recursive Shape Scope Assessment (${formatNumber(shown.length)} of ${formatNumber(scopeAssessments.length)})`, dataTable(
      ["Class", "Scope", "Status", "Executions", "Assessed nodes", "Unresolved outputs"],
      shown.map((row) => [
        row.scope_class,
        row.scope,
        row.status,
        formatNumber(row.execution_count || 0),
        `${formatNumber(row.assessed_node_count || 0)} / ${formatNumber(row.node_count || 0)}`,
        formatNumber(row.unresolved_output_count || 0),
      ]),
    )));
  }

  if (recursiveScopeAssessments.length) {
    const shown = recursiveScopeAssessments.slice(0, 200);
    sections.push(section(`Recursive Engine Execution Ledger (${formatNumber(shown.length)} of ${formatNumber(recursiveScopeAssessments.length)})`, dataTable(
      ["Class", "Scope", "Status", "Executions", "Assessed nodes", "Unassessed nodes", "Unresolved outputs", "Reason"],
      shown.map((row) => [
        row.scope_class,
        row.scope,
        row.status,
        formatNumber(row.execution_count || 0),
        `${formatNumber(row.assessed_node_count || 0)} / ${formatNumber(row.node_count || 0)}`,
        formatNumber(row.unassessed_node_count || 0),
        formatNumber(row.unresolved_output_count || 0),
        (row.reason_codes || []).join(" / ") || "none",
      ]),
    )));
  }

  if (functionCalls.length) {
    const shown = functionCalls.slice(0, 200);
    sections.push(section(`FunctionProto Call Contracts (${formatNumber(shown.length)} of ${formatNumber(functionCalls.length)})`, dataTable(
      ["Function", "Scope / node", "Status", "Inputs / outputs", "Result"],
      shown.map((row) => [
        row.function_id,
        `${row.scope} / #${row.node_index}`,
        row.status,
        `${formatNumber(row.input_count || 0)} / ${formatNumber(row.output_count || 0)}`,
        (row.reason_codes || []).join(" / ") || row.body_status || "pass",
      ]),
    )));
  }

  if (controlFlow.length) {
    const shown = controlFlow.slice(0, 200);
    sections.push(section(`If / Loop / Scan Shape Contracts (${formatNumber(shown.length)} of ${formatNumber(controlFlow.length)})`, dataTable(
      ["Operator", "Scope / node", "Opset", "Status", "State / scan contract", "Exact Loop expansion", "Result"],
      shown.map((row) => [
        row.op_name,
        `${row.scope} / #${row.node_index}`,
        row.imported_opset ?? "missing",
        row.status,
        row.op_name === "Loop"
          ? `${formatNumber(row.state_variable_count || 0)} state(s), ${formatNumber(row.non_dense_state_variable_count || 0)} non-dense [${(row.state_value_kinds || []).join(" / ") || "none"}], ${formatNumber(row.scan_output_count || 0)} scan`
          : row.state_variable_count == null ? "N/A" : `${formatNumber(row.state_variable_count || 0)} state(s), ${formatNumber(row.scan_output_count || 0)} scan`,
        row.op_name === "Loop"
          ? `${row.exact_expansion_status || "not assessed"}; ${row.exact_iteration_count == null ? "runtime-dependent" : `${formatNumber(row.exact_iteration_count)} iteration(s)`}; ${formatNumber(row.exact_body_node_evaluation_count || 0)} body-node evaluation(s)`
          : "N/A",
        (row.reason_codes || []).join(" / ") || row.body_status || (row.branch_statuses || []).join(" / ") || "pass",
      ]),
    )));
  }

  const boundary = document.createElement("p");
  boundary.className = "onnx-domain-boundary";
  boundary.textContent = [evidence.interpretation_boundary, typeContract.interpretation_boundary, sparseContract.interpretation_boundary, shape.interpretation_boundary, extended.interpretation_boundary, container.interpretation_boundary, tfidf.interpretation_boundary, mlValue.interpretation_boundary].filter(Boolean).join(" ");
  const summaryMetrics = document.createElement("div");
  summaryMetrics.className = "onnx-domain-metrics onnx-domain-summary-metrics";
  summaryMetrics.append(
    metric("Imported domains", domains.length, "neutral"),
    metric("Extension families", extensionFamilies.length, extensionFamilies.length ? "neutral" : "ok"),
    metric("Type declarations", typeContract.declaration_count || 0, "neutral"),
    metric("Issue counters", issueCounterCount, issueCounterCount ? "warn" : "ok"),
  );
  const priorityItems = [
    ["External registry nodes", evidence.external_custom_node_count || 0, "observed"],
    ["ORT contrib nodes", evidence.ort_contrib_node_count || 0, "observed"],
    ["Local function definitions", functions.length, "observed"],
    ["Local function calls", evidence.model_local_function_call_count || 0, "observed"],
    ["Nested graph nodes", nestedNodeCount, "observed"],
    ["Sparse records", sparseContract.sparse_tensor_count || 0, "observed"],
    ["Container ops", container.assessed_node_count || 0, "observed"],
    ["TfIdfVectorizer ops", tfidf.assessed_node_count || 0, "observed"],
    ["ONNX-ML value ops", mlValue.assessed_node_count || 0, "observed"],
    ["Registry issues", duplicateCount + cycleCount, "issue"],
    ["Schema-form issues", schemaFailures.length, "issue"],
    ["Unassessed reachable nodes", shape.shape_scope?.unassessed_reachable_node_count || 0, "unassessed"],
    ["Unresolved scope outputs", shape.shape_scope?.reachable_scope_unresolved_output_count || 0, "unassessed"],
    ["Function-call failures", extended.local_function_call_fail_count || 0, "issue"],
    ["Control-flow failures", extended.control_flow_fail_count || 0, "issue"],
    ["Recursive residual nodes", extended.residual_unassessed_node_count || 0, "unassessed"],
    ["Recursive unresolved outputs", extended.residual_unresolved_output_count || 0, "unassessed"],
    ["Type defects", typeContract.invalid_type_count || 0, "issue"],
    ["Sparse incomplete", Number(sparseContract.invalid_sparse_tensor_count || 0) + Number(sparseContract.partially_assessed_sparse_tensor_count || 0), "unassessed"],
    ["Sparse index violations", sparseIndexViolations, "issue"],
    ["Container partial or failed", Number(container.partially_assessed_node_count || 0) + Number(container.failed_node_count || 0), "unassessed"],
    ["TfIdf partial or failed", Number(tfidf.partially_assessed_node_count || 0) + Number(tfidf.failed_node_count || 0), "unassessed"],
    ["ONNX-ML partial or failed", Number(mlValue.partially_assessed_node_count || 0) + Number(mlValue.failed_node_count || 0), "unassessed"],
    ...semanticIssueItems.map(([label, value]) => [label, value, "issue"]),
  ].filter(([, value]) => Number(value) > 0);
  const priority = document.createElement("section");
  priority.className = "onnx-domain-priority";
  const priorityHeading = document.createElement("h4");
  priorityHeading.textContent = "Observed, unresolved, and issue-bearing evidence";
  const groupedPriority = [
    ["issue", "Issues requiring review"],
    ["unassessed", "Unresolved or partially assessed"],
    ["observed", "Observed extension evidence"],
  ].map(([state, label]) => priorityGroup(state, label, priorityItems.filter((item) => item[2] === state)))
    .filter(Boolean);
  priority.append(priorityHeading, ...(groupedPriority.length
    ? groupedPriority
    : [message("No extension-family observation, issue, or unresolved counter applies to this artifact.")]));
  const applicability = message(extensionFamilies.length
    ? `Serialized extension families: ${extensionFamilies.join(", ")}.`
    : "Not applicable to this artifact: no custom/contrib, local-function, nested-control-flow, sparse, container, TfIdf, or ONNX-ML contract is serialized.");
  applicability.className = "onnx-domain-applicability";
  const ledger = document.createElement("details");
  ledger.className = "onnx-domain-ledger";
  ledger.open = false;
  const ledgerHeading = document.createElement("summary");
  ledgerHeading.textContent = `Full domain and value-contract ledger - ${formatNumber(metrics.childElementCount)} counters / ${formatNumber(sections.length)} tables`;
  ledger.append(ledgerHeading, metrics, ...sections, boundary);
  body.replaceChildren(summaryMetrics, applicability, priority, ledger);
}

function dataTable(headers, rows) {
  const tableWrap = document.createElement("div");
  tableWrap.className = "onnx-domain-table-wrap";
  const table = document.createElement("table");
  table.className = "onnx-domain-table";
  const head = table.createTHead().insertRow();
  headers.forEach((label) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  });
  const tbody = table.createTBody();
  rows.forEach((values) => {
    const row = tbody.insertRow();
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.dataset.label = headers[index] || "Value";
      const cellValue = document.createElement("span");
      cellValue.className = "onnx-domain-cell-value";
      cellValue.textContent = String(value);
      cell.append(cellValue);
    });
  });
  tableWrap.append(table);
  return tableWrap;
}

function section(title, content) {
  const wrapper = document.createElement("section");
  wrapper.className = "onnx-domain-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  wrapper.append(heading, content);
  return wrapper;
}

function metric(label, value, tone, state = "") {
  const item = document.createElement("div");
  item.className = `onnx-domain-metric ${tone}`;
  if (state) item.dataset.state = state;
  const span = document.createElement("span");
  span.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = typeof value === "string" ? value : formatNumber(value);
  item.append(span, strong);
  return item;
}

function priorityGroup(state, label, items) {
  if (!items.length) return null;
  const group = document.createElement("section");
  group.className = `onnx-domain-priority-group state-${state}`;
  group.dataset.state = state;
  const heading = document.createElement("h5");
  heading.textContent = `${label} - ${formatNumber(items.length)}`;
  const metrics = document.createElement("div");
  metrics.className = "onnx-domain-metrics onnx-domain-priority-metrics";
  for (const [itemLabel, value] of items) {
    metrics.append(metric(itemLabel, value, state === "observed" ? "neutral" : "warn", state));
  }
  group.append(heading, metrics);
  return group;
}

function message(text) {
  const node = document.createElement("p");
  node.className = "visual-empty";
  node.textContent = text;
  return node;
}
