const TREE_OPS = new Set(["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"]);
const TREE_EVIDENCE_POINTER = "/evidence/static_analysis/onnx_shape_inference/ml_value_inference";
const TREE_ROW_POINTER = `${TREE_EVIDENCE_POINTER}/rows`;
const FINDINGS_POINTER = "/evidence/findings_register/findings";

const SEMANTIC_HAZARDS = new Set([
  "tree_ensemble_nonleaf_weights_ignored_by_pinned_ort",
  "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort",
  "tree_classifier_binary_post_transform_leaves_score_unwritten",
  "tree_regressor_single_target_post_transform_noop",
  "tree_ensemble_v5_single_target_post_transform_noop",
  "tree_classifier_binary_single_base_value_semantics_underspecified",
  "tree_classifier_pinned_ort_binary_label_index_semantics",
  "tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser",
]);

const DEAD_OR_NON_TREE_RISKS = new Set([
  "tree_ensemble_nonleaf_weights_ignored_by_pinned_ort",
  "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort",
  "tree_ensemble_shared_subtree_not_strict_tree",
  "tree_ensemble_v5_shared_subtree_not_strict_tree",
  "tree_ensemble_v5_unreachable_serialized_nodes_or_leaves",
]);

const MEMBERSHIP_RISKS = new Set([
  "tree_ensemble_v5_duplicate_membership_values",
  "tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser",
]);

const OUTPUT_SEMANTIC_RISKS = new Set([
  "tree_classifier_duplicate_labels_ambiguous_output_semantics",
  "tree_classifier_binary_post_transform_leaves_score_unwritten",
  "tree_regressor_single_target_post_transform_noop",
  "tree_ensemble_v5_single_target_post_transform_noop",
  "tree_classifier_binary_single_base_value_semantics_underspecified",
  "tree_classifier_pinned_ort_binary_label_index_semantics",
]);

const TREE_MLBOM_BINDINGS = [
  ["deepbom:model:onnxMlTreeEnsembleModelNodes", "tree_ensemble_model_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleV5Nodes", "tree_ensemble_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleClassifierNodes", "tree_ensemble_classifier_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleRegressorNodes", "tree_ensemble_regressor_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleDeprecatedNodes", "tree_ensemble_deprecated_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleOnnxContractFailures", "tree_ensemble_onnx_contract_failure_node_count"],
  ["deepbom:model:onnxMlTreeEnsemblePinnedOrtContractFailures", "tree_ensemble_pinned_ort_contract_failure_node_count"],
  ["deepbom:model:onnxMlTreeEnsemblePinnedCpuDtypeGaps", "tree_ensemble_pinned_cpu_dtype_gap_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleReferenceAssessedNodes", "tree_ensemble_reference_assessed_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleNonfiniteNodes", "tree_ensemble_non_finite_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleBoundaryNodes", "tree_ensemble_reference_boundary_node_count"],
  ["deepbom:model:onnxMlTreeEnsembleSemanticHazardNodes", "tree_ensemble_semantic_hazard_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleTrees", "exact_tree_ensemble_tree_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleRoots", "exact_tree_ensemble_root_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleNodes", "exact_tree_ensemble_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleBranches", "exact_tree_ensemble_branch_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleLeaves", "exact_tree_ensemble_leaf_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReachableNodes", "exact_tree_ensemble_reachable_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReachableLeaves", "exact_tree_ensemble_reachable_leaf_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleOrphans", "exact_tree_ensemble_orphan_node_or_leaf_count"],
  ["deepbom:model:onnxMlMaximumTreeEnsembleDepth", "maximum_tree_ensemble_depth"],
  ["deepbom:model:onnxMlExactTreeEnsembleCycles", "exact_tree_ensemble_cycle_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleDuplicateNodes", "exact_tree_ensemble_duplicate_node_identity_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleInvalidChildren", "exact_tree_ensemble_invalid_child_reference_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleInvalidFeatures", "exact_tree_ensemble_invalid_feature_id_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleRootMismatches", "exact_tree_ensemble_root_mismatch_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleMultipleParents", "exact_tree_ensemble_multiple_parent_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleWeights", "exact_tree_ensemble_weight_tuple_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleUsedWeights", "exact_tree_ensemble_used_weight_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleUnusedWeights", "exact_tree_ensemble_unused_weight_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleUnresolvedWeights", "exact_tree_ensemble_unresolved_weight_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleIgnoredNonleafWeights", "exact_tree_ensemble_ignored_nonleaf_weight_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleInvalidWeightReferences", "exact_tree_ensemble_invalid_weight_reference_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleInvalidWeightIds", "exact_tree_ensemble_invalid_weight_id_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleSingleTargetIgnoredWeights", "exact_tree_ensemble_single_target_ignored_weight_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleMemberNodes", "exact_tree_ensemble_membership_node_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleMemberSets", "exact_tree_ensemble_membership_set_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleMemberValues", "exact_tree_ensemble_membership_value_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleDuplicateMemberValues", "exact_tree_ensemble_membership_duplicate_value_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleMemberSeparators", "exact_tree_ensemble_membership_separator_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleNonfiniteParameters", "exact_tree_ensemble_non_finite_parameter_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceInputs", "exact_tree_ensemble_reference_input_value_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceRows", "exact_tree_ensemble_reference_row_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferencePathSteps", "exact_tree_ensemble_reference_path_step_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceRawScores", "exact_tree_ensemble_reference_raw_score_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceOutputScores", "exact_tree_ensemble_reference_output_score_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceNonfiniteScores", "exact_tree_ensemble_reference_non_finite_score_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceBoundaries", "exact_tree_ensemble_reference_decision_boundary_count"],
  ["deepbom:model:onnxMlExactTreeEnsembleReferenceUnwrittenScores", "exact_tree_ensemble_reference_unwritten_score_count"],
];

export function buildOnnxTreeConformanceFacts(rows = []) {
  const treeRows = rows.filter((row) => TREE_OPS.has(row.op_name));
  const sum = (field) => treeRows.reduce((total, row) => total + Number(row[field] || 0), 0);
  const safeSum = (field) => treeRows.reduce((total, row) => total
    + (Number.isSafeInteger(row[field]) ? row[field] : 0), 0);
  const hasRisk = (row, risks) => (row.risk_codes || []).some((risk) => risks.has(risk));
  const treePinnedRuntimeFailureRows = treeRows.filter((row) => row.tree_pinned_ort_contract_status === "fail");

  return {
    treeRows,
    treeGenericRows: treeRows.filter((row) => row.op_name === "TreeEnsemble"),
    treeClassifierRows: treeRows.filter((row) => row.op_name === "TreeEnsembleClassifier"),
    treeRegressorRows: treeRows.filter((row) => row.op_name === "TreeEnsembleRegressor"),
    treeDeprecatedRows: treeRows.filter((row) => row.tree_deprecated_operator === true),
    treeOnnxFailureRows: treeRows.filter((row) => row.tree_onnx_contract_status === "fail"),
    treePinnedRuntimeFailureRows,
    treeCpuGapRows: treeRows.filter((row) => row.tree_pinned_cpu_dtype_gap === true),
    treeRuntimeInvalidRows: treePinnedRuntimeFailureRows.filter((row) => row.tree_pinned_cpu_dtype_gap !== true),
    treeReferenceRows: treeRows.filter((row) => String(row.tree_reference_assessment_status || "").startsWith("assessed_")),
    treeNonFiniteRows: treeRows.filter((row) => Number(row.tree_non_finite_parameter_count || 0)
      + Number(row.tree_reference_non_finite_score_count || 0) > 0),
    treeBoundaryRows: treeRows.filter((row) => Number(row.tree_reference_decision_boundary_count || 0) > 0),
    treeSemanticHazardRows: treeRows.filter((row) => hasRisk(row, SEMANTIC_HAZARDS)),
    treeDeadOrNonTreeRows: treeRows.filter((row) => hasRisk(row, DEAD_OR_NON_TREE_RISKS)),
    treeMembershipRiskRows: treeRows.filter((row) => hasRisk(row, MEMBERSHIP_RISKS)),
    treeOutputSemanticRows: treeRows.filter((row) => hasRisk(row, OUTPUT_SEMANTIC_RISKS)),
    exactTreeCount: sum("tree_exact_tree_count"),
    exactTreeRoots: sum("tree_exact_root_count"),
    exactTreeNodes: sum("tree_exact_node_count"),
    exactTreeBranches: sum("tree_exact_branch_node_count"),
    exactTreeLeaves: sum("tree_exact_leaf_count"),
    exactTreeReachableNodes: sum("tree_reachable_node_count"),
    exactTreeReachableLeaves: sum("tree_reachable_leaf_count"),
    exactTreeOrphans: sum("tree_orphan_node_or_leaf_count"),
    maximumTreeDepth: treeRows.reduce((maximum, row) => Math.max(maximum, Number(row.tree_max_depth || 0)), 0),
    exactTreeCycles: sum("tree_cycle_count"),
    exactTreeDuplicateNodes: sum("tree_duplicate_node_identity_count"),
    exactTreeInvalidChildren: sum("tree_invalid_child_reference_count"),
    exactTreeInvalidFeatures: sum("tree_invalid_feature_id_count"),
    exactTreeRootMismatches: sum("tree_root_mismatch_count"),
    exactTreeMultipleParents: sum("tree_multiple_parent_node_count"),
    exactTreeWeights: sum("tree_weight_tuple_count"),
    exactTreeUsedWeights: sum("tree_used_weight_count"),
    exactTreeUnusedWeights: sum("tree_unused_weight_count"),
    exactTreeUnresolvedWeights: sum("tree_unresolved_weight_count"),
    exactTreeIgnoredNonleafWeights: sum("tree_ignored_nonleaf_weight_count"),
    exactTreeInvalidWeightReferences: sum("tree_invalid_weight_reference_count"),
    exactTreeInvalidWeightIds: sum("tree_invalid_weight_id_count"),
    exactTreeSingleTargetIgnoredWeights: sum("tree_single_target_ignored_weight_count"),
    exactTreeMemberNodes: sum("tree_membership_node_count"),
    exactTreeMemberSets: sum("tree_membership_set_count"),
    exactTreeMemberValues: sum("tree_membership_value_count"),
    exactTreeDuplicateMemberValues: sum("tree_membership_duplicate_value_count"),
    exactTreeMemberSeparators: sum("tree_membership_separator_count"),
    exactTreeNonFiniteParameters: sum("tree_non_finite_parameter_count"),
    exactTreeReferenceInputs: safeSum("tree_reference_input_value_count"),
    exactTreeReferenceRows: safeSum("tree_reference_row_count"),
    exactTreeReferencePathSteps: safeSum("tree_reference_path_step_count"),
    exactTreeReferenceScores: safeSum("tree_reference_raw_score_count"),
    exactTreeReferenceOutputScores: safeSum("tree_reference_output_score_count"),
    exactTreeReferenceNonFiniteScores: safeSum("tree_reference_non_finite_score_count"),
    exactTreeReferenceBoundaries: safeSum("tree_reference_decision_boundary_count"),
    exactTreeReferenceUnwrittenScores: safeSum("tree_reference_unwritten_score_count"),
  };
}

export function onnxTreeLedgerConserves(facts, evidence = {}) {
  const expected = [
    [facts.treeRows.length, "tree_ensemble_model_node_count"],
    [facts.treeGenericRows.length, "tree_ensemble_node_count"],
    [facts.treeClassifierRows.length, "tree_ensemble_classifier_node_count"],
    [facts.treeRegressorRows.length, "tree_ensemble_regressor_node_count"],
    [facts.treeDeprecatedRows.length, "tree_ensemble_deprecated_node_count"],
    [facts.treeOnnxFailureRows.length, "tree_ensemble_onnx_contract_failure_node_count"],
    [facts.treePinnedRuntimeFailureRows.length, "tree_ensemble_pinned_ort_contract_failure_node_count"],
    [facts.treeCpuGapRows.length, "tree_ensemble_pinned_cpu_dtype_gap_node_count"],
    [facts.treeReferenceRows.length, "tree_ensemble_reference_assessed_node_count"],
    [facts.treeNonFiniteRows.length, "tree_ensemble_non_finite_node_count"],
    [facts.treeBoundaryRows.length, "tree_ensemble_reference_boundary_node_count"],
    [facts.treeSemanticHazardRows.length, "tree_ensemble_semantic_hazard_node_count"],
    [facts.exactTreeCount, "exact_tree_ensemble_tree_count"],
    [facts.exactTreeRoots, "exact_tree_ensemble_root_count"],
    [facts.exactTreeNodes, "exact_tree_ensemble_node_count"],
    [facts.exactTreeBranches, "exact_tree_ensemble_branch_node_count"],
    [facts.exactTreeLeaves, "exact_tree_ensemble_leaf_count"],
    [facts.exactTreeReachableNodes, "exact_tree_ensemble_reachable_node_count"],
    [facts.exactTreeReachableLeaves, "exact_tree_ensemble_reachable_leaf_count"],
    [facts.exactTreeOrphans, "exact_tree_ensemble_orphan_node_or_leaf_count"],
    [facts.maximumTreeDepth, "maximum_tree_ensemble_depth"],
    [facts.exactTreeCycles, "exact_tree_ensemble_cycle_count"],
    [facts.exactTreeDuplicateNodes, "exact_tree_ensemble_duplicate_node_identity_count"],
    [facts.exactTreeInvalidChildren, "exact_tree_ensemble_invalid_child_reference_count"],
    [facts.exactTreeInvalidFeatures, "exact_tree_ensemble_invalid_feature_id_count"],
    [facts.exactTreeRootMismatches, "exact_tree_ensemble_root_mismatch_count"],
    [facts.exactTreeMultipleParents, "exact_tree_ensemble_multiple_parent_node_count"],
    [facts.exactTreeWeights, "exact_tree_ensemble_weight_tuple_count"],
    [facts.exactTreeUsedWeights, "exact_tree_ensemble_used_weight_count"],
    [facts.exactTreeUnusedWeights, "exact_tree_ensemble_unused_weight_count"],
    [facts.exactTreeUnresolvedWeights, "exact_tree_ensemble_unresolved_weight_count"],
    [facts.exactTreeIgnoredNonleafWeights, "exact_tree_ensemble_ignored_nonleaf_weight_count"],
    [facts.exactTreeInvalidWeightReferences, "exact_tree_ensemble_invalid_weight_reference_count"],
    [facts.exactTreeInvalidWeightIds, "exact_tree_ensemble_invalid_weight_id_count"],
    [facts.exactTreeSingleTargetIgnoredWeights, "exact_tree_ensemble_single_target_ignored_weight_count"],
    [facts.exactTreeMemberNodes, "exact_tree_ensemble_membership_node_count"],
    [facts.exactTreeMemberSets, "exact_tree_ensemble_membership_set_count"],
    [facts.exactTreeMemberValues, "exact_tree_ensemble_membership_value_count"],
    [facts.exactTreeDuplicateMemberValues, "exact_tree_ensemble_membership_duplicate_value_count"],
    [facts.exactTreeMemberSeparators, "exact_tree_ensemble_membership_separator_count"],
    [facts.exactTreeNonFiniteParameters, "exact_tree_ensemble_non_finite_parameter_count"],
    [facts.exactTreeReferenceInputs, "exact_tree_ensemble_reference_input_value_count"],
    [facts.exactTreeReferenceRows, "exact_tree_ensemble_reference_row_count"],
    [facts.exactTreeReferencePathSteps, "exact_tree_ensemble_reference_path_step_count"],
    [facts.exactTreeReferenceScores, "exact_tree_ensemble_reference_raw_score_count"],
    [facts.exactTreeReferenceOutputScores, "exact_tree_ensemble_reference_output_score_count"],
    [facts.exactTreeReferenceNonFiniteScores, "exact_tree_ensemble_reference_non_finite_score_count"],
    [facts.exactTreeReferenceBoundaries, "exact_tree_ensemble_reference_decision_boundary_count"],
    [facts.exactTreeReferenceUnwrittenScores, "exact_tree_ensemble_reference_unwritten_score_count"],
  ];
  return expected.every(([actual, field]) => actual === Number(evidence[field] || 0))
    && facts.exactTreeRoots === facts.exactTreeCount
    && facts.exactTreeUsedWeights + facts.exactTreeUnusedWeights + facts.exactTreeUnresolvedWeights === facts.exactTreeWeights;
}

export function registerOnnxTreeConformanceChecks({ check, facts, finding, engineeringReport }) {
  const findingCheck = (id, rows, priority, message, code) => check(code,
    rows.length > 0 ? finding(id)?.technical_priority === priority : !finding(id),
    message, [TREE_ROW_POINTER, FINDINGS_POINTER]);

  findingCheck("EA-ONX-0059", facts.treeRuntimeInvalidRows, "High", "EA-ONX-0059 must exist exactly when a TreeEnsemble pinned executable contract fails independently of a CPU dtype registration gap.", "CF-SHAPE-064");
  findingCheck("EA-ONX-0060", facts.treeCpuGapRows, "High", "EA-ONX-0060 must exist exactly when a schema-valid TreeEnsemble dtype lacks the pinned ORT CPU kernel.", "CF-SHAPE-065");
  findingCheck("EA-ONX-0061", facts.treeDeprecatedRows, "Medium", "EA-ONX-0061 must exist exactly for legacy TreeEnsemble operators resolved at their deprecation opset.", "CF-SHAPE-066");
  findingCheck("EA-ONX-0062", facts.treeDeadOrNonTreeRows, "Medium", "EA-ONX-0062 must exist exactly for unreachable, ignored, shared-subtree, or otherwise non-tree serialized state.", "CF-SHAPE-067");
  findingCheck("EA-ONX-0063", facts.treeMembershipRiskRows, "Medium", "EA-ONX-0063 must exist exactly for duplicate MEMBER values or the source-pinned zero-member reference/runtime divergence.", "CF-SHAPE-068");
  findingCheck("EA-ONX-0064", facts.treeOutputSemanticRows, "High", "EA-ONX-0064 must exist exactly for source-backed TreeEnsemble label, base-value, transform, or binary-score hazards.", "CF-SHAPE-069");
  findingCheck("EA-ONX-0065", [...facts.treeNonFiniteRows, ...facts.treeBoundaryRows], "High", "EA-ONX-0065 must exist exactly for TreeEnsemble non-finite state/reference results or exact decision-boundary cases.", "CF-SHAPE-070");

  const report = String(engineeringReport || "");
  check("CF-SHAPE-071", !facts.treeRows.length || report.includes("TreeEnsemble topology/runtime contracts")
    && report.includes("weight conservation")
    && report.includes("MEMBER nodes/sets/values/duplicates/separators")
    && report.includes("reference values are not claimed runtime-bit-exact"),
  "Engineering Report must expose TreeEnsemble topology, serialized-weight conservation, MEMBER sets, runtime contract, and scalar-reference boundary.", [TREE_EVIDENCE_POINTER, "/engineering_report.md"]);
}

export function onnxTreeMlBomConserves(propertyValue, evidence = {}) {
  return TREE_MLBOM_BINDINGS.every(([propertyName, fieldName]) => Number(propertyValue(propertyName)) === Number(evidence[fieldName] || 0));
}
