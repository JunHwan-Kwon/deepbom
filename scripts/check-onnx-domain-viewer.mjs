import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "sample_cnn_float.onnx");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-onnx-domain-viewer-"));
const server = createStaticServer(ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  await page.locator("#onnxDomainPanel").waitFor({ state: "visible" });
  const sampleState = await domainState(page);
  if (sampleState.count !== "1 domain / 0 local functions" || !sampleState.text.includes("ai.onnx") || !sampleState.text.includes("9 (9/0/0)")
    || !sampleState.text.includes("TypeProto Declarations")) {
    throw new Error(`Actual ONNX upload did not render the parsed standard-domain inventory: ${JSON.stringify(sampleState)}`);
  }
  const samplePresentation = await page.locator("#onnxDomainPanel").evaluate((panel) => ({
    summaryMetricCount: panel.querySelectorAll(".onnx-domain-summary-metrics > .onnx-domain-metric").length,
    ledgerOpen: Boolean(panel.querySelector(".onnx-domain-ledger")?.open),
    applicability: panel.querySelector(".onnx-domain-applicability")?.textContent || "",
    priorityGroupCount: panel.querySelectorAll(".onnx-domain-priority-group").length,
  }));
  if (samplePresentation.summaryMetricCount !== 4 || samplePresentation.ledgerOpen || samplePresentation.priorityGroupCount !== 0
    || !samplePresentation.applicability.includes("Not applicable to this artifact")) {
    throw new Error(`Standard ONNX graph did not collapse non-applicable extension counters: ${JSON.stringify(samplePresentation)}`);
  }

  await page.evaluate(async () => {
    const [{ buildOnnxDomainAnalysis }, { renderOnnxDomainViewer }] = await Promise.all([
      import("./lib/onnx-domain-analysis.js"),
      import("./lib/onnx-domain-viewer.js"),
    ]);
    const imports = [
      { domain: "", version: 13 },
      { domain: "ai.onnx.ml", version: 3 },
      { domain: "com.microsoft", version: 1 },
      { domain: "com.deepbom.local", version: 1 },
      { domain: "com.acme", version: 2 },
    ];
    const localFunction = {
      name: "FusedBlock",
      domain: "com.deepbom.local",
      overload: "fp32",
      inputs: ["X"],
      outputs: ["Y"],
      attributes: [],
      opsets: [{ domain: "", version: 13 }],
      nodes: [{ opType: "Relu", domain: "", overload: "", attributes: new Map() }],
    };
    const nestedGraph = { nodes: [{ opType: "NestedCustom", domain: "com.acme", overload: "", attributes: new Map() }] };
    const graph = { nodes: [
      { opType: "Conv", domain: "", overload: "", attributes: new Map() },
      { opType: "TreeEnsembleClassifier", domain: "ai.onnx.ml", overload: "", attributes: new Map() },
      { opType: "Attention", domain: "com.microsoft", overload: "", attributes: new Map() },
      { opType: "FusedBlock", domain: "com.deepbom.local", overload: "fp32", attributes: new Map() },
      { opType: "ExternalKernel", domain: "com.acme", overload: "", attributes: new Map([["body", { graph: nestedGraph, graphs: [] }]]) },
    ] };
    renderOnnxDomainViewer(document.querySelector("#onnxDomainPanel"), {
      format: "onnx",
      onnx_domain_analysis: buildOnnxDomainAnalysis({ graph, opsets: imports, functions: [localFunction] }),
      onnx_type_proto_contract: {
        declaration_count: 2, non_dense_value_count: 1, invalid_type_count: 0,
        rows: [
          { scope: "main_graph", role: "graph_input", value_name: "X", kind: "tensor", canonical_type: "tensor<FLOAT32[1,4]>", status: "pass", reason_codes: [] },
          { scope: "main_graph", role: "graph_value_info", value_name: "S", kind: "sequence", canonical_type: "sequence<tensor<FLOAT32[?,4]>>", status: "pass", reason_codes: [] },
        ],
        interpretation_boundary: "Only dense tensor values enter tensor-byte calculations.",
      },
      onnx_sparse_tensor_contract: {
        status: "assessed", sparse_tensor_count: 1, invalid_sparse_tensor_count: 0, partially_assessed_sparse_tensor_count: 0,
        out_of_bounds_index_count: 0, duplicate_index_count: 0, unsorted_index_count: 0,
        rows: [{ scope: "main_graph/sparse_initializer:w", tensor_role: "graph_sparse_initializer", sparse_tensor_name: "w", dense_shape: [2, 4], nnz: 2, index_encoding: "linear_indices", index_content_status: "assessed", assessed_index_count: 2, out_of_bounds_index_count: 0, duplicate_index_count: 0, unsorted_index_count: 0, payload_status: "embedded_or_empty", embedded_payload_bytes: 24, verified_external_payload_component_count: 0, external_payload_component_count: 0, status: "pass", reason_codes: [] }],
        interpretation_boundary: "Sparse storage is distinct from logical dense initializer semantics.",
      },
      onnx_shape_inference: {
        extended_scope_inference: {
          status: "assessed",
          control_flow_fail_count: 0,
          loop_node_count: 1,
          loop_exact_expansion_count: 1,
          loop_exact_iteration_count: 2,
          loop_exact_body_node_evaluation_count: 4,
          loop_non_dense_state_variable_count: 1,
          control_flow_rows: [{ scope: "main_graph", node_index: 4, op_name: "Loop", imported_opset: 16, status: "pass", body_node_count: 2, state_variable_count: 1, scan_output_count: 0, state_value_kinds: ["sequence"], non_dense_state_variable_count: 1, exact_expansion_status: "assessed", exact_iteration_count: 2, exact_body_node_evaluation_count: 4, reason_codes: [], body_status: "assessed" }],
          sequence_map_node_count: 1,
          sequence_map_pass_count: 1,
          sequence_map_partial_count: 0,
          sequence_map_fail_count: 0,
          sequence_map_rows: [{ scope: "main_graph", node_index: 3, imported_opset: 18, status: "pass", exact_input_sequence_length: 2, element_expansion_count: 2, element_node_evaluation_count: 4, reason_codes: [], body_status: "assessed" }],
          scope_execution_count: 1,
          scope_definition_count: 1,
          fully_assessed_scope_count: 1,
          residual_unassessed_node_count: 0,
          residual_unresolved_output_count: 0,
          scope_rows: [{ scope: "main_graph/node:3/attribute:body", scope_class: "nested_graph", status: "assessed", node_count: 2, execution_count: 1, assessed_node_count: 2, unassessed_node_count: 0, unresolved_output_count: 0, reason_codes: [] }],
        },
        container_value_inference: {
          assessed_node_count: 3,
          partially_assessed_node_count: 1,
          failed_node_count: 0,
          exact_sequence_length_output_count: 2,
          exact_optional_presence_output_count: 1,
          rows: [
            { scope: "main_graph", node_index: 0, op_name: "SequenceConstruct", imported_opset: 13, status: "pass", canonical_output_types: ["sequence<tensor<FLOAT32[1,4]>>"], sequence_lengths: [2], optional_presence: [null], reason_codes: [] },
            { scope: "main_graph", node_index: 1, op_name: "Optional", imported_opset: 15, status: "pass", canonical_output_types: ["optional<tensor<FLOAT32[1,4]>>"], sequence_lengths: [null], optional_presence: [true], reason_codes: [] },
            { scope: "main_graph", node_index: 2, op_name: "SequenceLength", imported_opset: 13, status: "partial", canonical_output_types: ["tensor<INT64[]>"], sequence_lengths: [null], optional_presence: [null], reason_codes: ["sequence_length_runtime_unknown"] },
          ],
        },
        tfidf_vectorizer_inference: {
          assessed_node_count: 1, passed_node_count: 1, partially_assessed_node_count: 0, failed_node_count: 0,
          exact_static_node_count: 1, exact_ngram_definition_count: 2, exact_active_ngram_definition_count: 2,
          exact_match_count: 2, exact_output_value_count: 2, exact_duplicate_output_coordinate_count: 0,
          exact_weight_coordinate_value_disagreement_count: 2, exact_ort_reference_divergent_output_count: 1,
          rows: [{
            scope: "main_graph", node_index: 18, op_name: "TfIdfVectorizer", imported_opset: 9, status: "pass",
            input_name: "tokens", input_dtype: "INT32", input_shape: [2], output_name: "features", exact_output_shape: [2],
            mode: "TFIDF", minimum_gram_length: 1, maximum_gram_length: 1, maximum_skip_count: 0,
            pool_kind: "int64", exact_pool_item_count: 2, exact_ngram_definition_count: 2,
            exact_active_ngram_definition_count: 2, exact_unused_pool_prefix_item_count: 0,
            exact_output_width: 2, exact_ngram_index_count: 2, exact_duplicate_output_coordinate_count: 0,
            exact_unaddressed_output_coordinate_count: 0, weights_present: true, exact_weight_count: 2,
            exact_weight_coordinate_value_disagreement_count: 2, static_execution_status: "assessed_exact",
            exact_static_work_units: 2, exact_match_count: 2, exact_nonzero_frequency_count: 2,
            exact_nonzero_output_count: 2, exact_negative_zero_output_count: 0,
            exact_ort_reference_divergent_output_count: 1, exact_output_values: [2, 3],
            reason_codes: [], risk_codes: ["tfidf_weight_coordinate_semantics_divergence", "tfidf_ort_repeated_addition_differs_from_onnx_reference"],
          }],
          interpretation_boundary: "Exact static text-vectorizer values do not prove selected-EP inclusion or optimized runtime assignment.",
        },
        ml_value_inference: {
          passed_node_count: 13,
          partially_assessed_node_count: 1,
          failed_node_count: 0,
          exact_sequence_length_output_count: 1,
          exact_class_key_count: 3,
          duplicate_class_key_count: 1,
          map_producer_node_count: 1,
          map_consumer_node_count: 2,
          tensor_mapper_node_count: 1,
          tensor_aggregator_node_count: 1,
          tensor_selector_node_count: 1,
          tensor_normalization_node_count: 1,
          tensor_affine_scaler_node_count: 1,
          tensor_imputation_node_count: 1,
          tensor_encoder_node_count: 1,
          tensor_label_mapping_node_count: 1,
          exact_dense_output_shape_count: 12,
          exact_vocabulary_entry_count: 9,
          duplicate_vocabulary_entry_count: 3,
          exact_category_pair_count: 3,
          duplicate_category_active_key_count: 1,
          feature_vectorizer_node_count: 1,
          feature_vectorizer_exact_width_node_count: 1,
          exact_feature_vectorizer_padded_feature_count_per_batch: 1,
          exact_feature_vectorizer_truncated_feature_count_per_batch: 1,
          feature_vectorizer_truncating_node_count: 1,
          array_feature_extractor_node_count: 1,
          array_feature_extractor_exact_index_node_count: 1,
          array_feature_extractor_bounds_failure_node_count: 0,
          binarizer_node_count: 1,
          binarizer_exact_static_node_count: 1,
          exact_binarizer_input_value_count: 4,
          exact_binarizer_above_threshold_count: 2,
          exact_binarizer_at_or_below_threshold_count: 2,
          exact_binarizer_equal_threshold_count: 1,
          binarizer_schema_default_threshold_node_count: 0,
          binarizer_nonfinite_threshold_node_count: 0,
          normalizer_node_count: 1,
          normalizer_static_assessed_node_count: 1,
          normalizer_output_materialized_node_count: 1,
          exact_normalizer_input_value_count: 2,
          exact_normalizer_zero_divisor_row_count: 0,
          exact_normalizer_negative_max_divisor_row_count: 1,
          exact_normalizer_integer_float32_rounding_count: 0,
           exact_normalizer_signed_overflow_value_count: 0,
           exact_normalizer_non_finite_output_count: 0,
           exact_normalizer_signed_zero_output_count: 0,
          normalizer_schema_default_mode_node_count: 1,
          scaler_node_count: 1,
          scaler_static_assessed_node_count: 1,
          scaler_output_materialized_node_count: 1,
          scaler_invalid_runtime_contract_node_count: 0,
          exact_scaler_input_value_count: 2,
          exact_scaler_integer_float32_rounding_count: 1,
          exact_scaler_non_finite_parameter_count: 0,
          exact_scaler_non_finite_output_count: 0,
          exact_scaler_signed_zero_output_count: 1,
          exact_scaler_zero_scale_count: 1,
          imputer_node_count: 1,
          imputer_static_assessed_node_count: 1,
          imputer_output_materialized_node_count: 1,
          imputer_invalid_runtime_contract_node_count: 0,
          imputer_scalar_first_fallback_node_count: 1,
          imputer_pinned_cpu_dtype_gap_node_count: 0,
          exact_imputer_input_value_count: 3,
          exact_imputer_replacement_count: 2,
          exact_imputer_nan_replacement_count: 0,
          exact_imputer_unchanged_count: 1,
          exact_imputer_ignored_imputed_value_count: 1,
          exact_imputer_non_finite_imputed_value_count: 0,
          exact_imputer_non_finite_output_count: 0,
          exact_imputer_signed_zero_output_count: 0,
          onehot_encoder_node_count: 1,
          onehot_static_assessed_node_count: 1,
          onehot_output_materialized_node_count: 1,
          onehot_invalid_contract_node_count: 0,
          onehot_duplicate_vocabulary_node_count: 1,
          onehot_unknown_all_zero_node_count: 1,
          onehot_guaranteed_runtime_failure_node_count: 0,
          onehot_pinned_cpu_dtype_gap_node_count: 0,
          onehot_noncanonical_zeros_node_count: 0,
          onehot_unrepresentable_numeric_cast_node_count: 0,
          exact_onehot_input_value_count: 3,
          exact_onehot_matched_input_count: 2,
          exact_onehot_unknown_input_count: 1,
          exact_onehot_numeric_to_int64_changed_count: 0,
          exact_onehot_numeric_to_int64_invalid_count: 0,
          exact_onehot_output_one_count: 2,
          exact_onehot_output_zero_count: 7,
          exact_onehot_duplicate_category_count: 1,
          exact_onehot_unreachable_duplicate_column_count: 1,
          label_encoder_node_count: 1,
          label_encoder_static_assessed_node_count: 1,
          label_encoder_output_materialized_node_count: 0,
          label_encoder_onnx_contract_failure_node_count: 0,
          label_encoder_pinned_ort_contract_failure_node_count: 0,
          label_encoder_pinned_cpu_dtype_pair_gap_node_count: 0,
          label_encoder_duplicate_semantic_conflict_node_count: 1,
          label_encoder_nan_semantic_conflict_node_count: 0,
          label_encoder_default_path_node_count: 1,
          label_encoder_schema_runtime_output_mismatch_node_count: 1,
          exact_label_encoder_key_count: 3,
          exact_label_encoder_input_value_count: 3,
          exact_label_encoder_match_count: 2,
          exact_label_encoder_default_count: 1,
          exact_label_encoder_duplicate_key_hit_count: 1,
          exact_label_encoder_schema_runtime_mismatch_count: 1,
          linear_model_node_count: 2,
          linear_classifier_node_count: 1,
          linear_regressor_node_count: 1,
          linear_onnx_contract_failure_node_count: 0,
          linear_pinned_ort_contract_failure_node_count: 0,
          linear_reference_assessed_node_count: 2,
          linear_pinned_cpu_dtype_gap_node_count: 0,
          linear_post_transform_hazard_node_count: 0,
          linear_unused_coefficient_node_count: 1,
          linear_ignored_intercept_node_count: 0,
          exact_linear_coefficient_count: 9,
          exact_linear_used_coefficient_count: 8,
          exact_linear_unused_coefficient_count: 1,
          exact_linear_unresolved_coefficient_use_count: 0,
          exact_linear_ignored_intercept_count: 0,
          exact_linear_reference_input_value_count: 4,
          exact_linear_reference_raw_score_count: 4,
          rows: [{
            scope: "main_graph", node_index: 5, op_name: "ZipMap", imported_opset: 1, status: "pass",
            contract_kind: "map_producer", input_kind: "tensor", input_name: "scores", output_name: "probabilities", input_dtype: "FLOAT32", input_rank: 2, input_shape: [1, 3],
            exact_feature_count: 3, class_key_type: "STRING", class_key_count: 3, duplicate_key_count: 1,
            class_key_preview: ["cat", "cat", "bird"], exact_output_sequence_length: 1,
            attribute_mode: "class_labels", sparse_key_bounds_status: "not_applicable", output_kind: "sequence", canonical_output_type: "sequence<map<STRING,tensor<FLOAT32[]>>>",
            output_shape_basis: "pinned_onnx_schema_and_ort_cpu_batch_semantics", runtime_reference_status: "pinned_ort_cpu_implementation", reason_codes: [],
            risk_codes: ["zip_map_duplicate_class_keys_information_loss_risk"],
          }, {
            scope: "main_graph", node_index: 6, op_name: "CastMap", imported_opset: 1, status: "partial",
            contract_kind: "map_consumer", input_kind: "map", input_name: "score_map", output_name: "dense_scores",
            input_map_key_type: "INT64", input_map_value_dtype: "FLOAT32", exact_input_map_key_count: null, cast_to: "TO_STRING", map_form: "SPARSE", max_map: 5,
            attribute_mode: "SPARSE", sparse_key_bounds_status: "not_assessed_runtime_keys", output_kind: "tensor", output_dtype: "STRING", exact_output_rank: 1,
            exact_output_shape: [5], exact_dense_output_element_count: 5, canonical_output_type: "tensor<STRING[5]>",
            output_shape_basis: "pinned_onnx_schema_sparse_max_map", runtime_reference_status: "onnx_schema_only_no_pinned_ort_cpu_kernel", reason_codes: ["cast_map_sparse_key_bounds_runtime_unknown"], risk_codes: [],
          }, {
            scope: "main_graph", node_index: 7, op_name: "DictVectorizer", imported_opset: 1, status: "pass",
            contract_kind: "map_consumer", input_kind: "map", input_name: "features", output_name: "feature_vector",
            input_map_key_type: "STRING", input_map_value_dtype: "FLOAT64", exact_input_map_key_count: null, vocabulary_type: "STRING", vocabulary_count: 3,
            duplicate_vocabulary_count: 1, vocabulary_preview: ["age", "age", "weight"], attribute_mode: "vocabulary", sparse_key_bounds_status: "not_applicable",
            output_kind: "tensor", output_dtype: "FLOAT64", exact_output_rank: 2, exact_output_shape: [1, 3],
            exact_dense_output_element_count: 3, canonical_output_type: "tensor<FLOAT64[1,3]>",
            output_shape_basis: "pinned_onnx_type_constraint_and_ort_cpu_vocabulary_size_allocation", runtime_reference_status: "pinned_ort_cpu_implementation", reason_codes: [],
            risk_codes: ["dict_vectorizer_duplicate_vocabulary_columns"],
          }, {
            scope: "main_graph", node_index: 8, op_name: "CategoryMapper", imported_opset: 1, status: "pass",
            contract_kind: "tensor_mapper", input_kind: "tensor", input_name: "categories", output_name: "category_ids",
            input_dtype: "STRING", input_rank: 2, input_shape: [2, 2], mapping_direction: "STRING_TO_INT64",
            category_pair_count: 3, category_string_count: 3, category_int64_count: 3, duplicate_string_key_count: 1, duplicate_int64_key_count: 0, active_duplicate_key_count: 1,
            active_default_type: "INT64", active_default_value: "99", category_string_preview: ["red", "red", "blue"], category_int64_preview: ["10", "20", "30"],
            attribute_mode: "bidirectional_categories", sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "INT64",
            exact_output_rank: 2, exact_output_shape: [2, 2], exact_dense_output_element_count: 4, canonical_output_type: "tensor<INT64[2,2]>",
            output_shape_basis: "pinned_onnx_shape_propagation_and_ort_cpu_same_shape_allocation", runtime_reference_status: "pinned_ort_cpu_implementation", reason_codes: [],
            risk_codes: ["category_mapper_duplicate_active_keys_last_write_wins"],
          }, {
            scope: "main_graph", node_index: 9, op_name: "FeatureVectorizer", imported_opset: 1, status: "pass",
            contract_kind: "tensor_aggregator", input_kind: "tensor_list", input_name: "x0", input_names: ["x0", "x1"], output_name: "joined",
            input_dtype: "INT32", input_dtypes: ["INT32", "INT32"], input_rank: 2, input_ranks: [2, 3], input_shape: [2, 3], input_shapes: [[2, 3], [2, 2, 2]],
            configured_feature_dimensions: ["2", "5"], total_configured_feature_count: 7, exact_input_row_feature_counts: [3, 4],
            exact_copied_feature_count_per_batch: 6, exact_padded_feature_count_per_batch: 1, exact_truncated_feature_count_per_batch: 1,
            attribute_mode: "input_dimensions", sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 2, exact_output_shape: [2, 7], exact_dense_output_element_count: 14, canonical_output_type: "tensor<FLOAT32[2,7]>",
            output_shape_basis: "pinned_ort_cpu_feature_dimension_allocation", runtime_reference_status: "pinned_ort_cpu_implementation", reason_codes: [],
            risk_codes: ["feature_vectorizer_truncates_input_features"],
          }, {
            scope: "main_graph", node_index: 10, op_name: "ArrayFeatureExtractor", imported_opset: 1, status: "pass",
            contract_kind: "tensor_selector", input_kind: "tensor", input_name: "matrix", input_names: ["matrix", "indices"], output_name: "selected",
            input_dtype: "INT32", input_dtypes: ["INT32", "INT64"], input_rank: 2, input_ranks: [2, 1], input_shape: [2, 4], input_shapes: [[2, 4], [3]],
            index_input_name: "indices", index_input_dtype: "INT64", index_input_shape: [3], exact_index_count: 3,
            exact_index_preview: ["0", "3", "3"], duplicate_index_count: 1, index_bounds_status: "assessed_pass", out_of_bounds_index_count: 0,
            attribute_mode: "last_axis_indices", sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "INT32",
            exact_output_rank: 2, exact_output_shape: [2, 3], exact_dense_output_element_count: 6, canonical_output_type: "tensor<INT32[2,3]>",
            output_shape_basis: "pinned_onnx_last_axis_shape_rule_and_ort_cpu_rank1_compatibility", runtime_reference_status: "pinned_ort_cpu_implementation", reason_codes: [], risk_codes: [],
          }, {
            scope: "main_graph", node_index: 11, op_name: "Binarizer", imported_opset: 1, status: "pass",
            contract_kind: "tensor_threshold", input_kind: "tensor", input_name: "raw_scores", output_name: "binary_scores",
            input_dtype: "FLOAT32", input_rank: 1, input_shape: [4], threshold_value: 0.25, threshold_value_text: "0.25", threshold_source: "explicit_attribute",
            static_value_assessment_status: "assessed_exact", exact_static_input_value_count: 4, exact_above_threshold_count: 2,
            exact_at_or_below_threshold_count: 2, exact_equal_threshold_count: 1, exact_output_one_count: 2, exact_output_zero_count: 2,
            attribute_mode: "threshold", sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 1, exact_output_shape: [4], exact_dense_output_element_count: 4, canonical_output_type: "tensor<FLOAT32[4]>",
            output_shape_basis: "pinned_onnx_same_type_same_shape_propagation", runtime_reference_status: "pinned_ort_cpu_float32_kernel_only", reason_codes: [], risk_codes: [],
          }, {
            scope: "main_graph", node_index: 12, op_name: "Normalizer", imported_opset: 1, status: "pass",
            contract_kind: "tensor_normalization", input_kind: "tensor", input_name: "signed_scores", output_name: "normalized_scores",
            input_dtype: "FLOAT32", input_rank: 1, input_shape: [2], exact_batch_count: 1, exact_feature_count: 2,
            attribute_mode: "normalization_mode", normalizer_mode: "MAX", normalizer_mode_source: "onnx_schema_default_MAX",
            normalizer_static_assessment_status: "assessed_pinned_ort_float32", normalizer_exact_input_value_count: 2,
            normalizer_batch_count: 1, normalizer_row_width: 2, normalizer_divisor_kind: "signed_max", normalizer_divisor_preview: ["-1"],
            normalizer_zero_divisor_row_count: 0, normalizer_negative_max_divisor_row_count: 1,
            normalizer_integer_float32_rounding_count: 0, normalizer_signed_overflow_value_count: 0, normalizer_non_finite_output_count: 0,
            normalizer_signed_zero_output_count: 0,
            normalizer_output_materialized: true, normalizer_output_preview: ["2", "1"],
            sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 1, exact_output_shape: [2], exact_dense_output_element_count: 2, canonical_output_type: "tensor<FLOAT32[2]>",
            output_shape_basis: "pinned_onnx_float_output_same_shape_and_ort_row_semantics", runtime_reference_status: "pinned_ort_cpu_float32_output_kernel",
            reason_codes: [], risk_codes: ["normalizer_negative_signed_max_divisor"],
          }, {
            scope: "main_graph", node_index: 13, op_name: "Scaler", imported_opset: 1, status: "pass",
            contract_kind: "tensor_affine_scaler", input_kind: "tensor", input_name: "integer_scores", output_name: "scaled_scores",
            input_dtype: "INT64", input_rank: 1, input_shape: [2], exact_batch_count: 1, exact_feature_count: 2,
            attribute_mode: "offset_then_scale", scaler_parameter_contract_status: "pass", scaler_parameter_contract_reason: "scaler_scalar_parameters",
            scaler_parameter_mode: "scalar", scaler_feature_stride: 2, scaler_scale_count: 1, scaler_offset_count: 1,
            scaler_scale_values: ["-0"], scaler_offset_values: ["0"], scaler_zero_scale_count: 1, scaler_non_finite_parameter_count: 0,
            scaler_static_assessment_status: "assessed_pinned_ort_float32", scaler_exact_input_value_count: 2,
            scaler_integer_float32_rounding_count: 1, scaler_non_finite_output_count: 0, scaler_signed_zero_output_count: 1,
            scaler_output_materialized: true, scaler_output_preview: ["-0", "0"],
            sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 1, exact_output_shape: [2], exact_dense_output_element_count: 2, canonical_output_type: "tensor<FLOAT32[2]>",
            output_shape_basis: "pinned_ort_cpu_float_output_same_shape_and_second_dimension_feature_stride", runtime_reference_status: "pinned_ort_cpu_scaler_kernel",
            reason_codes: [], risk_codes: ["scaler_integer_to_float32_precision_loss"],
          }, {
            scope: "main_graph", node_index: 14, op_name: "Imputer", imported_opset: 1, status: "pass",
            contract_kind: "tensor_imputation", input_kind: "tensor", input_name: "missing_scores", output_name: "imputed_scores",
            input_dtype: "FLOAT32", input_rank: 2, input_shape: [1, 3], exact_batch_count: 1, exact_feature_count: 3,
            attribute_mode: "replace_marker", imputer_parameter_contract_status: "pass", imputer_parameter_contract_reason: "imputer_pinned_ort_scalar_first_fallback",
            imputer_parameter_mode: "scalar_first_fallback", imputer_attribute_kind: "float", imputer_feature_stride: 3,
            imputer_imputed_value_count: 2, imputer_imputed_values: ["9", "8"], imputer_replaced_value: "0",
            imputer_replaced_value_source: "onnx_schema_default_0", imputer_ignored_imputed_value_count: 1,
            imputer_static_assessment_status: "assessed_exact_pinned_ort_semantics", imputer_exact_input_value_count: 3,
            imputer_exact_replacement_count: 2, imputer_exact_nan_replacement_count: 0, imputer_exact_unchanged_count: 1,
            imputer_non_finite_imputed_value_count: 0, imputer_non_finite_output_count: 0, imputer_signed_zero_output_count: 0,
            imputer_output_materialized: true, imputer_output_preview: ["9", "9", "1"],
            sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 2, exact_output_shape: [1, 3], exact_dense_output_element_count: 3, canonical_output_type: "tensor<FLOAT32[1,3]>",
            output_shape_basis: "pinned_onnx_same_dtype_same_shape_and_ort_second_dimension_feature_stride", runtime_reference_status: "pinned_ort_cpu_imputer_kernel",
            reason_codes: [], risk_codes: ["imputer_attribute_length_outside_onnx_one_or_feature_count"],
          }, {
            scope: "main_graph", node_index: 15, op_name: "OneHotEncoder", imported_opset: 1, status: "pass",
            contract_kind: "tensor_encoder", input_kind: "tensor", input_name: "category_scores", output_name: "one_hot_scores",
            input_dtype: "INT64", input_rank: 1, input_shape: [3], attribute_mode: "single_category_vocabulary_and_zeros_policy",
            onehot_parameter_contract_status: "pass", onehot_parameter_contract_reason: "onehot_single_nonempty_category_list",
            onehot_category_kind: "int64", onehot_category_count: 3, onehot_category_values: ["1", "1", "2"],
            onehot_duplicate_category_count: 1, onehot_unreachable_duplicate_column_count: 1, onehot_unreachable_duplicate_column_indices: [0],
            onehot_zeros_value: "1", onehot_zeros_source: "onnx_schema_default_1", onehot_zeros_enabled: true, onehot_zeros_canonical_boolean: true,
            onehot_static_assessment_status: "assessed_exact_pinned_ort_semantics", onehot_exact_input_value_count: 3,
            onehot_exact_matched_input_count: 2, onehot_exact_unknown_input_count: 1, onehot_numeric_to_int64_changed_count: 0,
            onehot_numeric_to_int64_invalid_count: 0, onehot_guaranteed_runtime_failure: false,
            onehot_exact_output_one_count: 2, onehot_exact_output_zero_count: 7, onehot_output_materialized: true,
            onehot_unknown_input_preview: ["9"], onehot_output_preview: [0, 1, 0, 0, 0, 1, 0, 0, 0],
            vocabulary_type: "INT64", vocabulary_count: 3, duplicate_vocabulary_count: 1, vocabulary_preview: ["1", "1", "2"],
            sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "FLOAT32",
            exact_output_rank: 2, exact_output_shape: [3, 3], exact_dense_output_element_count: 9, canonical_output_type: "tensor<FLOAT32[3,3]>",
            output_shape_basis: "pinned_onnx_append_category_axis_and_ort_cpu_lookup", runtime_reference_status: "pinned_ort_cpu_int64_float_double_string_kernels",
            reason_codes: [], risk_codes: ["onehot_duplicate_categories_last_write_wins", "onehot_unknown_categories_all_zero_encoding"],
          }, {
            scope: "main_graph", node_index: 16, op_name: "LinearClassifier", imported_opset: 1, status: "pass",
            contract_kind: "linear_classifier", input_kind: "tensor", input_name: "linear_features", output_name: "linear_scores",
            input_dtype: "FLOAT32", input_rank: 2, input_shape: [1, 2], attribute_mode: "linear_coefficients_intercepts_labels_targets_and_post_transform",
            output_names: ["linear_labels", "linear_scores"], canonical_output_types: ["tensor<INT64[1]>", "tensor<FLOAT32[1,2]>"], canonical_output_shapes: [[1], [1, 2]],
            output_kind: "tensor", output_dtype: "FLOAT32", exact_output_rank: 2, exact_output_shape: [1, 2], exact_dense_output_element_count: 2,
            classifier_label_output_name: "linear_labels", classifier_score_output_name: "linear_scores", classifier_label_output_dtype: "INT64",
            classifier_label_output_shape: [1], classifier_score_output_shape: [1, 2], classifier_score_class_count: 2, classifier_binary_score_expansion: false,
            linear_onnx_contract_status: "pass", linear_pinned_ort_contract_status: "pass", linear_class_or_target_count: 2,
            linear_label_kind: "int64", linear_label_count: 2, linear_label_values: ["10", "20"], linear_duplicate_label_count: 0,
            linear_coefficient_count: 5, linear_used_coefficient_count: 4, linear_unused_coefficient_count: 1, linear_intercept_count: 2, linear_intercepts_used: true,
            linear_multi_class_value: "0", linear_multi_class_used_by_pinned_ort: false, linear_post_transform: "NONE",
            linear_reference_assessment_status: "assessed_scalar_float32_reference_not_runtime_bit_exact", linear_reference_input_value_count: 2,
            linear_reference_raw_score_count: 2, linear_reference_raw_score_preview: ["1", "2"], linear_reference_output_preview: ["1", "2"],
            linear_reference_label_preview: ["20"], linear_non_finite_parameter_count: 0, linear_reference_non_finite_raw_score_count: 0,
            linear_reference_decision_boundary_count: 0, output_shape_basis: "pinned_onnx_linear_model_rank_and_pinned_ort_attribute_contract",
            runtime_reference_status: "pinned_ort_cpu_linear_model_kernel_and_ml_common", reason_codes: [], risk_codes: ["linear_classifier_unused_coefficients_ignored"],
          }, {
            scope: "main_graph", node_index: 17, op_name: "LinearRegressor", imported_opset: 1, status: "pass",
            contract_kind: "linear_regressor", input_kind: "tensor", input_name: "regression_features", output_name: "regression_scores",
            input_dtype: "FLOAT32", input_rank: 2, input_shape: [1, 2], attribute_mode: "linear_coefficients_intercepts_labels_targets_and_post_transform",
            output_names: ["regression_scores"], canonical_output_types: ["tensor<FLOAT32[1,2]>"], canonical_output_shapes: [[1, 2]],
            output_kind: "tensor", output_dtype: "FLOAT32", exact_output_rank: 2, exact_output_shape: [1, 2], exact_dense_output_element_count: 2,
            linear_onnx_contract_status: "pass", linear_pinned_ort_contract_status: "pass", linear_class_or_target_count: 2,
            linear_targets_value: "2", linear_targets_source: "explicit_attribute", linear_coefficient_count: 4, linear_used_coefficient_count: 4,
            linear_unused_coefficient_count: 0, linear_intercept_count: 2, linear_intercepts_used: true, linear_ignored_intercept_count: 0,
            linear_post_transform: "NONE", linear_reference_assessment_status: "assessed_scalar_float32_reference_not_runtime_bit_exact",
            linear_reference_input_value_count: 2, linear_reference_raw_score_count: 2, linear_reference_raw_score_preview: ["1.5", "1.5"],
            linear_reference_output_preview: ["1.5", "1.5"], linear_reference_label_preview: [], linear_non_finite_parameter_count: 0,
            linear_reference_non_finite_raw_score_count: 0, linear_reference_decision_boundary_count: null,
            output_shape_basis: "pinned_onnx_linear_model_rank_and_pinned_ort_attribute_contract",
            runtime_reference_status: "pinned_ort_cpu_linear_model_kernel_and_ml_common", reason_codes: [], risk_codes: [],
          }, {
            scope: "main_graph", node_index: 18, op_name: "LabelEncoder", imported_opset: 4, resolved_schema_version: 4, status: "pass",
            contract_kind: "tensor_label_mapping", input_kind: "tensor", input_name: "raw_labels", output_name: "encoded_labels",
            input_dtype: "STRING", input_rank: 1, input_shape: [3], attribute_mode: "v4_parallel_key_value_attributes",
            label_encoder_onnx_contract_status: "pass", label_encoder_pinned_ort_contract_status: "pass",
            label_encoder_pinned_ort_contract_reason: "pinned_ort_v4_typed_dtype_pair_kernel",
            label_encoder_key_dtype: "STRING", label_encoder_value_dtype: "INT16", label_encoder_key_count: 3, label_encoder_value_count: 3,
            label_encoder_key_values: ["a", "a", "b"], label_encoder_value_values: ["1", "2", "3"],
            label_encoder_duplicate_key_count: 1, label_encoder_nan_key_count: 0, label_encoder_non_finite_key_count: 0, label_encoder_non_finite_value_count: 0,
            label_encoder_runtime_duplicate_policy: "first_key_wins", label_encoder_schema_duplicate_policy: "last_key_wins",
            label_encoder_default_value: "-1", label_encoder_default_source: "explicit_default_tensor",
            label_encoder_exact_input_value_count: 3, label_encoder_exact_match_count: 2, label_encoder_exact_default_count: 1,
            label_encoder_exact_duplicate_key_hit_count: 1, label_encoder_schema_runtime_mismatch_count: 1,
            label_encoder_runtime_output_preview: ["1", "3", "-1"], label_encoder_schema_output_preview: ["2", "3", "-1"],
            label_encoder_output_materialized: false, vocabulary_type: "STRING", vocabulary_count: 3, duplicate_vocabulary_count: 1,
            vocabulary_preview: ["a", "a", "b"], sparse_key_bounds_status: "not_applicable", output_kind: "tensor", output_dtype: "INT16",
            exact_output_rank: 1, exact_output_shape: [3], exact_dense_output_element_count: 3, canonical_output_type: "tensor<INT16[3]>",
            output_shape_basis: "pinned_onnx_label_encoder_v4_same_shape_mapping", runtime_reference_status: "pinned_ort_cpu_label_encoder_versioned_kernels",
            reason_codes: [], risk_codes: ["label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict", "label_encoder_artifact_known_default_path_reached", "label_encoder_artifact_known_schema_runtime_output_mismatch"],
          }],
          interpretation_boundary: "Nineteen ONNX-ML value contracts are source-pinned; linear, SVM, and TreeEnsemble scalar references are not runtime-bit-exact without observed runtime execution details.",
        },
      },
    });
  });

  const complexState = await domainState(page);
  const priorityState = await page.locator("#onnxDomainPanel").evaluate((panel) => ({
    groups: [...panel.querySelectorAll(".onnx-domain-priority-group")].map((group) => group.dataset.state),
    ledgerOpen: Boolean(panel.querySelector(".onnx-domain-ledger")?.open),
    issueCount: panel.querySelectorAll('.onnx-domain-priority-group[data-state="issue"] .onnx-domain-metric').length,
    unassessedCount: panel.querySelectorAll('.onnx-domain-priority-group[data-state="unassessed"] .onnx-domain-metric').length,
    observedCount: panel.querySelectorAll('.onnx-domain-priority-group[data-state="observed"] .onnx-domain-metric').length,
  }));
  if (priorityState.groups.join("|") !== "issue|unassessed|observed"
    || priorityState.ledgerOpen || !priorityState.issueCount || !priorityState.unassessedCount || !priorityState.observedCount) {
    throw new Error(`ONNX priority evidence is not ordered issue -> unassessed -> observed: ${JSON.stringify(priorityState)}`);
  }
  for (const expected of [
    "5 domains / 1 local function",
    "External registry2",
    "ORT contrib1",
    "Local definitions1",
    "Local calls1",
    "Nested nodes1",
    "Registry issues0",
    "Type declarations2",
    "Non-dense values1",
    "Sparse records1",
    "Container ops3",
    "Container partial1",
    "Container failures0",
    "Exact length / presence2 / 1",
    "TfIdf P / ? / F1 / 0 / 0",
    "TfIdf exact static1 / 1",
    "TfIdf definitions active / total2 / 2",
    "TfIdf matches / output values2 / 2",
    "TfIdf coordinate aliases0",
    "TfIdf weight / reference divergence2 / 1",
    "ML value P / ? / F13 / 1 / 0",
    "ML exact length / keys1 / 3",
    "ML duplicate keys1",
    "ML producer / consumer / mapper1 / 2 / 1",
    "ML exact dense / vocabulary12 / 9",
    "ML duplicate vocabulary3",
    "ML category pairs3",
    "ML aggregate / select1 / 1",
    "Feature width exact1 / 1",
    "Feature pad / truncate1 / 1",
    "Array index exact1 / 1",
    "Array bounds fail0",
    "Binarizer static exact1 / 1",
    "Binarizer one / zero2 / 2",
    "Normalizer static exact1 / 1",
    "Normalizer zero / negative MAX0 / 1",
    "Normalizer cast / s0 / overflow / non-finite0 / 0 / 0 / 0",
    "Scaler static exact1 / 1",
    "Scaler invalid runtime contract0",
    "Scaler cast / non-finite param / output / s0 / zero-scale1 / 0 / 0 / 1 / 1",
    "Imputer static exact1 / 1",
    "Imputer invalid / fallback / dtype gap0 / 1 / 0",
    "Imputer replaced / NaN / ignored / non-finite2 / 0 / 1 / 0",
    "OneHot static exact1 / 1",
    "OneHot invalid / duplicate / fail / dtype0 / 1 / 0 / 0",
    "OneHot matched / unknown / cast / one / zero2 / 1 / 0 / 2 / 7",
    "Linear classifier / regressor1 / 1",
    "Linear ONNX / ORT / dtype / transform0 / 0 / 0 / 0",
    "Linear coefficients used / ignored / unresolved8 / 1 / 0",
    "#16 LinearClassifier",
    "coefficients 4 used / 5 serialized / 1 ignored",
    "not consulted",
    "scalar FLOAT32 reference assessed_scalar_float32_reference_not_runtime_bit_exact",
    "#17 LinearRegressor",
    "targets 2 (explicit_attribute)",
    "ML duplicate active categories1",
    "SequenceMap P / ? / F1 / 0 / 0",
    "Recursive engineassessed",
    "Recursive executions1 / 1 scopes",
    "Recursive residual N / O0 / 0",
    "Loop exact expansion1 / 1",
    "Loop iterations / work / non-dense2 / 4 / 1",
    "If / Loop / Scan Shape Contracts",
    "1 state(s), 1 non-dense [sequence], 0 scan",
    "assessed; 2 iteration(s); 4 body-node evaluation(s)",
    "Sequence / Optional Value Contracts",
    "TfIdfVectorizer-9 Contracts",
    "gram 1-1; skip <= 0",
    "mapping disagreement 2",
    "ORT/reference divergence 1",
    "ONNX-ML Value Contracts",
    "scores; FLOAT32; rank 2 [1,3]",
    "class labels STRING x3; duplicates 1; cat, cat, bird; features 3",
    "probabilities; sequence 1",
    "zip_map_duplicate_class_keys_information_loss_risk",
    "score_map; map<INT64, FLOAT32>",
    "cast_to TO_STRING; map_form SPARSE; max_map 5",
    "sparse key bounds not_assessed_runtime_keys",
    "cast_map_sparse_key_bounds_runtime_unknown",
    "dense_scores; STRING; rank 1 [5]; elements 5",
    "pinned_onnx_schema_sparse_max_map",
    "onnx_schema_only_no_pinned_ort_cpu_kernel",
    "features; map<STRING, FLOAT64>",
    "vocabulary STRING x3; duplicates 1; age, age, weight",
    "feature_vector; FLOAT64; rank 2 [1,3]; elements 3",
    "pinned_onnx_type_constraint_and_ort_cpu_vocabulary_size_allocation",
    "dict_vectorizer_duplicate_vocabulary_columns",
    "categories; STRING; rank 2 [2,2]",
    "STRING_TO_INT64; 3 pair(s), arrays 3 string / 3 int64; duplicate string/int64/active 1 / 0 / 1",
    "category_ids; INT64; rank 2 [2,2]; elements 4",
    "category_mapper_duplicate_active_keys_last_write_wins",
    "2 inputs; dtypes INT32 / INT32; shapes [2,3] / [2,2,2]",
    "widths 2 / 5 -> 7; row 3 / 4; copy/pad/truncate 6 / 1 / 1",
    "joined; FLOAT32; rank 2 [2,7]; elements 14",
    "feature_vectorizer_truncates_input_features",
    "matrix; INT32 [2,4]; indices indices INT64 [3]",
    "indices 3; 0, 3, 3; duplicates 1; bounds assessed_pass; invalid 0",
    "selected; INT32; rank 2 [2,3]; elements 6",
    "raw_scores; FLOAT32; rank 1 [4]",
    "threshold 0.25 (explicit_attribute); static assessed_exact; exact 4, above 2, at/below 2, equal 1",
    "binary_scores; FLOAT32; rank 1 [4]; elements 4",
    "pinned_ort_cpu_float32_kernel_only",
    "signed_scores; FLOAT32; rank 1 [2]",
    "MAX (onnx_schema_default_MAX); signed_max; static assessed_pinned_ort_float32; rows 1 x 2; divisors -1",
    "normalized_scores; FLOAT32; rank 1 [2]; elements 2",
    "zero 0; negative MAX 1; cast 0; signed zero 0; overflow 0; non-finite 0; output materialized 2, 1",
    "normalizer_negative_signed_max_divisor",
    "integer_scores; INT64; rank 1 [2]",
    "scalar; contract pass (scaler_scalar_parameters); stride 2; scale x1 [-0]; offset x1 [0]; static assessed_pinned_ort_float32",
    "scaled_scores; FLOAT32; rank 1 [2]; elements 2",
    "zero scale 1; cast 1; non-finite param/output 0 / 0; signed zero 1; output materialized -0, 0",
    "scaler_integer_to_float32_precision_loss",
    "missing_scores; FLOAT32; rank 2 [1,3]",
    "scalar_first_fallback; contract pass (imputer_pinned_ort_scalar_first_fallback); float; stride 3; imputed x2 [9, 8]; replace 0 (onnx_schema_default_0); static assessed_exact_pinned_ort_semantics",
    "imputed_scores; FLOAT32; rank 2 [1,3]; elements 3",
    "replaced 2; NaN 0; unchanged 1; ignored 1; non-finite value/output 0 / 0; signed zero 0; output materialized 9, 9, 1",
    "imputer_attribute_length_outside_onnx_one_or_feature_count",
    "category_scores; INT64; rank 1 [3]",
    "int64 vocabulary x3 [1, 1, 2]",
    "duplicate/unreachable 1 / 1 [0]",
    "one_hot_scores; FLOAT32; rank 2 [3,3]; elements 9",
    "input/matched/unknown 3 / 2 / 1",
    "onehot_duplicate_categories_last_write_wins",
    "onehot_unknown_categories_all_zero_encoding",
    "LabelEncoder matched / default / mismatch",
    "schema/runtime mismatches 1",
    "label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict",
    "Tree v5 / classifier / regressor",
    "Tree ONNX / ORT / dtype / deprecated",
    "Tree / node / leaf / max depth",
    "Tree reference nodes / paths / boundary / unwritten",
    "SequenceMap Value Contracts",
    "Recursive Engine Execution Ledger",
    "main_graph/node:3/attribute:body",
    "2 element(s) / 4 body-node evaluation(s)",
    "sequence_length_runtime_unknown",
    "TypeProto Declarations",
    "SparseTensorProto Records",
    "linear_indices",
    "com.deepbom.local::FusedBlock::fp32",
    "External Custom-op Registry Requirements",
    "ExternalKernel",
    "NestedCustom",
    "main_graph/node:4/attribute:body",
  ]) {
    if (!complexState.compactText.includes(expected.replaceAll(" ", "")) && !complexState.text.includes(expected)) {
      throw new Error(`ONNX domain viewer omitted ${expected}: ${JSON.stringify(complexState)}`);
    }
  }
  const desktopLayout = await page.evaluate(() => {
    const metrics = document.querySelector(".onnx-domain-ledger .onnx-domain-metrics");
    const children = [...(metrics?.children || [])];
    return {
      metricCount: children.length,
      metricRows: new Set(children.map((item) => Math.round(item.getBoundingClientRect().top))).size,
      metricOverflow: Math.max(0, (metrics?.scrollWidth || 0) - (metrics?.clientWidth || 0)),
    };
  });
  if (desktopLayout.metricCount !== 80 || desktopLayout.metricRows !== 12 || desktopLayout.metricOverflow > 1) {
    throw new Error(`ONNX domain viewer desktop metric layout is invalid: ${JSON.stringify(desktopLayout)}`);
  }
  const desktopPath = path.join(output, "onnx-domain-desktop.png");
  await page.locator("#onnxDomainPanel").screenshot({ path: desktopPath });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#onnxDomainPanel").scrollIntoViewIfNeeded();
  const mobile = await page.evaluate(() => {
    const panel = document.querySelector("#onnxDomainPanel");
    const metrics = document.querySelector(".onnx-domain-ledger .onnx-domain-metrics");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, (panel?.scrollWidth || 0) - (panel?.clientWidth || 0)),
      metricColumns: getComputedStyle(metrics).gridTemplateColumns.split(" ").length,
      metricRows: new Set([...(metrics?.children || [])].map((item) => Math.round(item.getBoundingClientRect().top))).size,
    };
  });
  const mobilePath = path.join(output, "onnx-domain-mobile.png");
  await page.locator("#onnxDomainPanel").screenshot({ path: mobilePath });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.metricColumns !== 2 || mobile.metricRows !== 40) {
    throw new Error(`ONNX domain viewer mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log("ONNX domain viewer passed (actual upload, five resolution domains, local FunctionProto, nested custom scope, desktop/mobile overflow 0).");
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    domain: document.querySelector("#onnxDomainPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function domainState(browserPage) {
  return browserPage.locator("#onnxDomainPanel").evaluate((panel) => ({
    count: panel.querySelector("[data-onnx-domain-count]")?.textContent || "",
    text: panel.textContent || "",
    compactText: (panel.textContent || "").replaceAll(/\s/g, ""),
  }));
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      if (url.pathname.startsWith("/api/")) return send(response, 404, "application/json", '{"error":"not_found"}');
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(root, relative);
      if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      send(response, 200, mimeType(file), await readFile(file));
    } catch {
      send(response, 404, "text/plain", "not found");
    }
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
