import {
  ONNX_BOOL_SAME_SHAPE_OPS as BOOL_SAME_SHAPE,
  ONNX_BROADCAST_BOOL_OPS as BROADCAST_BOOL,
  ONNX_BROADCAST_SAME_TYPE_OPS as BROADCAST_SAME_TYPE,
  ONNX_REDUCE_OPS as REDUCE_OPS,
  ONNX_SAME_SHAPE_OPS as SAME_SHAPE,
  ONNX_SHAPE_INFERENCE_OPS,
} from "./onnx-shape-ops.js";
import { assessOnnxNodeSchemaForm, ONNX_SHAPE_SCHEMA_SOURCE } from "./onnx-schema-legality.js";
import { inferOnnxContainerNode, ONNX_CONTAINER_VALUE_SOURCE } from "./onnx-container-inference.js";
import {
  assessOnnxMlNodeSchemaForm,
  canInferOnnxMlNode,
  inferOnnxMlNode,
  ONNX_ML_VALUE_OPS,
  ONNX_ML_VALUE_SOURCE,
} from "./onnx-ml-value-inference.js";
import {
  assessOnnxOrtExtensionSchemaForm,
  canInferOnnxOrtExtensionNode,
  inferOnnxOrtExtensionNode,
  ONNX_ORT_EXTENSION_SHAPE_SOURCE,
} from "./onnx-ort-extension-shape-inference.js";
import { assessOnnxOpsetImports } from "./onnx-opset-imports.js";
import {
  makeOnnxTensorType,
  makeOnnxTensorTypeFromDimensions,
  onnxShapeDimensionsFromValue,
  onnxTypeProtoFromValue,
  onnxTypeProtoKnown,
  onnxValueDescriptorFromType,
  unionOnnxTypeProtos,
  unifyOnnxTypeProtos,
} from "./onnx-type-proto.js";
import { parseOnnxEinsumEquation } from "./onnx-einsum-contract.js";
import {
  canInferOnnxTfIdfVectorizer,
  inferOnnxTfIdfVectorizerNode,
  ONNX_TFIDF_VECTORIZER_SOURCE,
} from "./onnx-tfidf-vectorizer.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const MAX_CONDITIONAL_SHAPE_VARIANTS = 64;
const RUNTIME_DIMENSION_PREFIX = "deepbom_runtime:";
export const ONNX_SHAPE_INFERENCE_SCHEMA = "deepbom.onnx_shape_inference.v1.30";

export const ONNX_SHAPE_INFERENCE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  documents: Object.freeze([
    Object.freeze({
      role: "model_opset_import_contract",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/onnx.proto`,
      sha256: "a05cfbcd1370608b809c5b84c44e3198d3369036458e0b5f297e76ceaf9c4e1b",
    }),
    Object.freeze({
      role: "shape_inference_contract",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/shape_inference.h`,
      sha256: "9602d530ea9bf4c1f3d8418c29114c264563ccbf10639ddd82a484fcd8bfc530",
    }),
    Object.freeze({
      role: "shape_inference_implementation",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/shape_inference.cc`,
      sha256: "3051554a17b7f632d90362aec2987b1e7374b3d69e421160db4f21aeda98363e",
    }),
    Object.freeze({
      role: "current_tensor_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/tensor/defs.cc`,
      sha256: "22681df3a131c55524dceb8e366dcc24dcce4acbbf198ac7ae5216313e619652",
    }),
    Object.freeze({
      role: "historical_tensor_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/tensor/old.cc`,
      sha256: "405e2ece240dae4e6a1929eb6786f3e49bf5bfd8c1095280b6835e194c6703a0",
    }),
    Object.freeze({
      role: "current_neural_network_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/nn/defs.cc`,
      sha256: "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02",
    }),
    Object.freeze({
      role: "current_matrix_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/math/defs.cc`,
      sha256: "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594",
    }),
    Object.freeze({
      role: "current_random_generator_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/generator/defs.cc`,
      sha256: "838c87511348b700000f133bf98522bc79f84cea6ff18e09e5f255b28ac183dd",
    }),
    Object.freeze({
      role: "current_recurrent_operator_rules",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/rnn/defs.cc`,
      sha256: "6262fe369d07727433a7ff49128dfedcc59209958e0edf3050e55bea5a932791",
    }),
    Object.freeze({
      role: "operator_schema_history",
      source_ref: ONNX_SHAPE_SCHEMA_SOURCE.source_ref,
      sha256: ONNX_SHAPE_SCHEMA_SOURCE.sha256,
    }),
  ]),
});

export { ONNX_SHAPE_INFERENCE_OPS } from "./onnx-shape-ops.js";

export function inferOnnxShapes(graph, tensorMap, opsets, tensorTypeName, functions = [], domainAnalysis = null, options = {}) {
  const opsetImportContract = assessOnnxOpsetImports(opsets);
  const imported = new Map(opsetImportContract.effective_imports.map((row) => [row.domain, row.version]));
  const unresolvableImportDomains = new Set(opsetImportContract.unresolvable_domains);
  let inferredOutputs = 0;
  let attemptedNodes = 0;
  let ruleSupportedNodes = 0;
  let propagatedValueTensorCount = 0;
  let propagatedSymbolicShapeValueTensorCount = 0;
  let inferredNonDenseOutputs = 0;
  const unsupportedOps = [];
  const unsupportedNodes = [];
  const unresolvedNodes = [];
  const failedExtendedNodes = [];
  const declarationConflicts = [];
  const semanticContractConflicts = [];
  const schemaFormRows = [];
  const containerRows = [];
  const mlValueRows = [];
  const tfidfRows = [];

  for (const node of graph.nodes || []) {
    const nodeIndex = attemptedNodes++;
    const domain = normalizeDomain(node.domain);
    const standard = domain === "ai.onnx";
    const importedOpset = imported.get(domain);
    const ortExtensionResolvable = canInferOnnxOrtExtensionNode(node, importedOpset);
    const mlResolvable = canInferOnnxMlNode(node);
    const tfidfResolvable = canInferOnnxTfIdfVectorizer(node);
    const storedOverride = options.nodeResultOverrides?.get(node) || null;
    const extendedResolvable = Boolean(storedOverride) || options.canResolveNode?.(node) === true;
    const supported = extendedResolvable || mlResolvable || tfidfResolvable || ortExtensionResolvable || (standard && ONNX_SHAPE_INFERENCE_OPS.has(node.opType));
    if (!supported) {
      unsupportedOps.push(standard ? node.opType || "UNKNOWN" : `${domain}:${node.opType || "UNKNOWN"}`);
      unsupportedNodes.push({ node_index: nodeIndex, op_name: node.opType || "UNKNOWN", domain, reason: "shape_rule_not_implemented" });
      continue;
    }
    ruleSupportedNodes += 1;
    if ((standard || mlResolvable || ortExtensionResolvable) && unresolvableImportDomains.has(domain)) {
      schemaFormRows.push({
        node_index: nodeIndex,
        op_name: node.opType,
        imported_opset: null,
        schema_since_version: null,
        status: "fail",
        reason_codes: ["opset_import_contract_invalid"],
        detail: `The ${domain} OperatorSetIdProto records do not contain a positive safe-integer version.`,
      });
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: "opset_import_contract_invalid" });
      continue;
    }
    if ((standard || mlResolvable || ortExtensionResolvable) && (!Number.isSafeInteger(importedOpset) || importedOpset <= 0)) {
      schemaFormRows.push({
        node_index: nodeIndex,
        op_name: node.opType,
        imported_opset: importedOpset ?? null,
        schema_since_version: null,
        status: "fail",
        reason_codes: ["standard_domain_opset_missing"],
        detail: `The ${domain} OperatorSetIdProto import is absent.`,
      });
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: "standard_domain_opset_missing" });
      continue;
    }
    if (standard) {
      const schemaForm = { node_index: nodeIndex, ...(ortExtensionResolvable
        ? assessOnnxOrtExtensionSchemaForm(node, importedOpset)
        : assessOnnxNodeSchemaForm(node, importedOpset)) };
      schemaFormRows.push(schemaForm);
      if (schemaForm.status !== "pass") {
        unresolvedNodes.push({
          node_index: nodeIndex,
          op_name: node.opType,
          reason: schemaForm.status === "fail" ? "opset_schema_form_invalid" : "opset_schema_form_unresolved",
        });
        continue;
      }
    } else if (ortExtensionResolvable) {
      const schemaForm = { node_index: nodeIndex, ...assessOnnxOrtExtensionSchemaForm(node, importedOpset) };
      schemaFormRows.push(schemaForm);
      if (schemaForm.status !== "pass") {
        unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: schemaForm.status === "fail" ? "opset_schema_form_invalid" : "opset_schema_form_unresolved" });
        continue;
      }
    } else if (mlResolvable) {
      const schemaForm = { node_index: nodeIndex, ...assessOnnxMlNodeSchemaForm(node, importedOpset) };
      schemaFormRows.push(schemaForm);
      if (schemaForm.status !== "pass") {
        unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: "opset_schema_form_invalid" });
        continue;
      }
    }
    const recursiveOverride = storedOverride || (extendedResolvable
      ? options.resolveNodeResult?.({ node, nodeIndex, tensorMap, importedOpset, scope: options.scope || "main_graph" }) || null
      : null);
    const mlOverride = !recursiveOverride && mlResolvable ? inferOnnxMlNode({
      node, tensorMap, nodeIndex, importedOpset, scope: options.scope || "main_graph",
    }) : null;
    if (mlOverride?.row) mlValueRows.push(mlOverride.row);
    const tfidfOverride = !recursiveOverride && !mlOverride && tfidfResolvable ? inferOnnxTfIdfVectorizerNode({
      node, tensorMap, nodeIndex, importedOpset, scope: options.scope || "main_graph",
    }) : null;
    if (tfidfOverride?.row) tfidfRows.push(tfidfOverride.row);
    const override = recursiveOverride || mlOverride || tfidfOverride;
    if (override?.status === "fail" || override?.status === "not_assessed") {
      const reason = override.reason || "extended_shape_rule_not_assessed";
      failedExtendedNodes.push({
        node_index: nodeIndex,
        op_name: node.opType,
        status: override.status,
        reason,
        failure_class: override.failure_class || "analysis_residual",
      });
      if (override.status === "fail" && override.failure_class === "artifact_contract_conflict") {
        const conflict = {
          node_index: nodeIndex,
          op_name: node.opType,
          output_names: (node.outputs || []).filter(Boolean),
          reason,
          details: override.conflict_details || null,
        };
        semanticContractConflicts.push(conflict);
        unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason, failure_class: "artifact_semantic_conflict" });
        markNodeOutputsInvalid(node, tensorMap, { ...conflict, root_conflict: override.conflict_details || conflict });
      } else {
        unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason });
      }
      continue;
    }
    const containerOverride = !override && standard ? inferOnnxContainerNode({
      node, tensorMap, tensorTypeName, nodeIndex, importedOpset, scope: options.scope || "main_graph",
    }) : null;
    if (containerOverride?.row) containerRows.push(containerOverride.row);
    if (containerOverride?.status === "fail") {
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: containerOverride.reason || "container_value_inference_failed" });
      continue;
    }
    const ortExtensionResult = !override && !containerOverride && ortExtensionResolvable
      ? inferOnnxOrtExtensionNode(node, tensorMap, tensorTypeName)
      : null;
    const invalidInput = firstInvalidInputContract(node, tensorMap);
    if (invalidInput) {
      const reason = `blocked_by_upstream_contract_conflict:${invalidInput.tensor_name}`;
      unresolvedNodes.push({
        node_index: nodeIndex,
        op_name: node.opType,
        reason,
        blocked_by: invalidInput,
      });
      markNodeOutputsInvalid(node, tensorMap, {
        node_index: nodeIndex,
        op_name: node.opType,
        reason,
        root_conflict: invalidInput.root_conflict || invalidInput,
      });
      continue;
    }
    const result = override?.result || containerOverride?.result || ortExtensionResult
      || inferNodeWithConditionalShapes(node, tensorMap, tensorTypeName);
    if (result.status === "invalid") {
      const conflict = {
        node_index: nodeIndex,
        op_name: node.opType,
        output_names: (node.outputs || []).filter(Boolean),
        reason: result.reason || "operator_contract_invalid",
        details: result.details || null,
      };
      semanticContractConflicts.push(conflict);
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: conflict.reason, failure_class: "artifact_semantic_conflict" });
      markNodeOutputsInvalid(node, tensorMap, { ...conflict, root_conflict: conflict });
      continue;
    }
    if (!result.outputs.length && node.outputs.some(Boolean)) {
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: result.reason || "rule_inputs_not_fully_known" });
    }
    let nodeConflict = false;
    for (const [name, patch] of result.outputs) {
      const before = tensorMap.get(name);
      const beforeKnown = valueContractKnown(before);
      const conflict = mergeOnnxInferredTensor(tensorMap, name, patch, nodeIndex, node.opType);
      if (conflict) {
        declarationConflicts.push(conflict);
        markTensorContractInvalid(tensorMap, name, {
          ...conflict,
          reason: "inferred_contract_conflicts_with_declared_output",
          root_conflict: conflict,
        });
        nodeConflict = true;
        continue;
      }
      const after = tensorMap.get(name);
      if (!beforeKnown && valueContractKnown(after)) {
        if (declaredNonDenseValue(after)) inferredNonDenseOutputs += 1;
        else inferredOutputs += 1;
      }
      if (patch.staticValuesComplete === true) propagatedValueTensorCount += 1;
      if (patch.staticDimensionValuesComplete === true && patch.staticValuesComplete !== true) propagatedSymbolicShapeValueTensorCount += 1;
    }
    if (nodeConflict && !unresolvedNodes.some((row) => row.node_index === nodeIndex)) {
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: "inferred_contract_conflicts_with_declared_output" });
    }
    if (!nodeConflict && node.outputs.some((name) => name && !valueContractKnown(tensorMap.get(name)))
      && !unresolvedNodes.some((row) => row.node_index === nodeIndex)) {
      unresolvedNodes.push({ node_index: nodeIndex, op_name: node.opType, reason: result.reason || "one_or_more_outputs_not_inferred" });
    }
  }

  const scopeLedger = buildShapeScopeLedger(graph, functions, domainAnalysis, options.scopeExecution || null);
  const values = [...tensorMap.values()];
  const nodeOutputNames = (graph.nodes || []).flatMap((node) => node.outputs || []).filter(Boolean);
  const knownNodeOutputCount = nodeOutputNames.filter((name) => tensorKnown(tensorMap.get(name))).length;
  const shapeContractKnownNodeOutputCount = nodeOutputNames.filter((name) => tensorShapeContractKnown(tensorMap.get(name))).length;
  const nonDenseNodeOutputNames = nodeOutputNames.filter((name) => declaredNonDenseValue(tensorMap.get(name)));
  const tensorNodeOutputNames = nodeOutputNames.filter((name) => !declaredNonDenseValue(tensorMap.get(name)));
  const knownNonDenseNodeOutputNames = nonDenseNodeOutputNames.filter((name) => valueContractKnown(tensorMap.get(name)));
  const unknownTensorIndices = values.map((tensor, index) => ({ tensor, index }))
    .filter(({ tensor }) => !tensorKnown(tensor) && !declaredNonDenseValue(tensor))
    .map(({ index }) => index);
  const nonDenseValueIndices = values.map((tensor, index) => ({ tensor, index }))
    .filter(({ tensor }) => declaredNonDenseValue(tensor))
    .map(({ index }) => index);
  const nodeOutputCount = nodeOutputNames.length;
  const nonDenseNodeOutputCount = nonDenseNodeOutputNames.length;
  const knownNonDenseNodeOutputCount = knownNonDenseNodeOutputNames.length;
  const unresolvedNonDenseNodeOutputCount = nonDenseNodeOutputCount - knownNonDenseNodeOutputCount;
  const tensorNodeOutputCount = nodeOutputCount - nonDenseNodeOutputCount;
  const unknownNodeOutputCount = tensorNodeOutputCount - knownNodeOutputCount;
  const shapeContractUnknownNodeOutputCount = tensorNodeOutputCount - shapeContractKnownNodeOutputCount;
  const invalidNodeOutputCount = tensorNodeOutputNames.filter((name) => tensorMap.get(name)?.contractStatus === "invalid").length;
  const conditionallyInvalidNodeOutputCount = tensorNodeOutputNames.filter((name) => {
    const tensor = tensorMap.get(name);
    return tensor?.contractStatus !== "invalid"
      && (tensor?.conditionalShapeContract?.variant_failures || []).some((row) => row?.status === "invalid");
  }).length;
  const conditionalInvalidVariantCount = tensorNodeOutputNames.reduce((total, name) => total
    + (tensorMap.get(name)?.conditionalShapeContract?.variant_failures || []).filter((row) => row?.status === "invalid").length, 0);
  const conditionalUnassessedVariantCount = tensorNodeOutputNames.reduce((total, name) => total
    + (tensorMap.get(name)?.conditionalShapeContract?.variant_failures || []).filter((row) => row?.status !== "invalid").length, 0);
  const blockedByUpstreamContractConflictNodeCount = unresolvedNodes.filter((row) => String(row.reason || "").startsWith("blocked_by_upstream_contract_conflict:")).length;
  const containerFailures = containerRows.filter((row) => row.status === "fail");
  const containerPartials = containerRows.filter((row) => row.status === "partial");
  const mlValueFailures = mlValueRows.filter((row) => row.status === "fail");
  const mlValuePartials = mlValueRows.filter((row) => row.status === "partial");
  const tfidfFailures = tfidfRows.filter((row) => row.status === "fail");
  const tfidfPartials = tfidfRows.filter((row) => row.status === "partial");
  const incomplete = unsupportedOps.length > 0 || unresolvedNodes.length > 0 || unknownTensorIndices.length > 0
    || unresolvedNonDenseNodeOutputCount > 0 || containerPartials.length > 0 || mlValuePartials.length > 0
    || tfidfPartials.length > 0
    || scopeLedger.reachable_exclusion_count > 0
    || scopeLedger.reachable_scope_unresolved_output_count > 0;
  const invalidSchemaForms = schemaFormRows.filter((row) => row.status === "fail");
  const unresolvedSchemaForms = schemaFormRows.filter((row) => row.status === "unresolved");
  const status = declarationConflicts.length || semanticContractConflicts.length || invalidSchemaForms.length || failedExtendedNodes.length || containerFailures.length
    || tfidfFailures.length
    || opsetImportContract.status === "fail" || scopeLedger.registry_status === "fail"
    || scopeLedger.failed_reachable_scope_count > 0 ? "fail"
    : !incomplete ? "assessed"
      : knownNodeOutputCount + knownNonDenseNodeOutputCount > 0 ? "partial" : "not_assessed";
  return {
    schema: ONNX_SHAPE_INFERENCE_SCHEMA,
    evidence_class: "SOURCE_PINNED_AND_DERIVED",
    status,
    engine: "source-pinned ONNX tensor, finite conditional-shape, runtime-symbolic dimension, text-vectorizer, container, ONNX-ML value-contract, and recursive-scope inference",
    source_release: ONNX_SHAPE_INFERENCE_SOURCE.release,
    source_commit: ONNX_SHAPE_INFERENCE_SOURCE.commit,
    source_documents: [
      ...ONNX_SHAPE_INFERENCE_SOURCE.documents.map((source) => ({ ...source })),
      ...ONNX_ORT_EXTENSION_SHAPE_SOURCE.documents.map((source) => ({ ...source })),
    ],
    ort_extension_source_commit: ONNX_ORT_EXTENSION_SHAPE_SOURCE.commit,
    ort_extension_interpretation_boundary: ONNX_ORT_EXTENSION_SHAPE_SOURCE.interpretation_boundary,
    opset_import_contract: opsetImportContract,
    attempted_nodes: attemptedNodes,
    rule_supported_nodes: ruleSupportedNodes,
    rule_unsupported_nodes: attemptedNodes - ruleSupportedNodes,
    rule_unsupported_node_indices: unsupportedNodes.map((row) => row.node_index),
    rule_unsupported_node_rows: unsupportedNodes,
    rule_unsupported_op_histogram: countBy(unsupportedOps),
    rule_unresolved_node_count: unresolvedNodes.length,
    rule_unresolved_node_indices: unresolvedNodes.map((row) => row.node_index),
    rule_unresolved_nodes: unresolvedNodes,
    schema_form_assessment_status: invalidSchemaForms.length ? "fail" : unresolvedSchemaForms.length ? "partial" : "pass",
    schema_form_assessed_node_count: schemaFormRows.length,
    schema_form_valid_node_count: schemaFormRows.filter((row) => row.status === "pass").length,
    schema_form_invalid_node_count: invalidSchemaForms.length,
    schema_form_unresolved_node_count: unresolvedSchemaForms.length,
    schema_form_rows: schemaFormRows,
    extended_rule_failed_node_count: failedExtendedNodes.length,
    extended_rule_failed_nodes: failedExtendedNodes,
    tfidf_vectorizer_inference: {
      schema: "deepbom.onnx_tfidf_vectorizer_inference.v1",
      evidence_class: "SOURCE_PINNED_AND_DERIVED",
      status: tfidfFailures.length ? "fail" : tfidfPartials.length ? "partial" : tfidfRows.length ? "assessed" : "not_applicable",
      source_release: ONNX_TFIDF_VECTORIZER_SOURCE.onnx_release,
      source_commit: ONNX_TFIDF_VECTORIZER_SOURCE.onnx_commit,
      runtime_reference_release: ONNX_TFIDF_VECTORIZER_SOURCE.ort_release,
      runtime_reference_commit: ONNX_TFIDF_VECTORIZER_SOURCE.ort_commit,
      source_documents: ONNX_TFIDF_VECTORIZER_SOURCE.documents.map((source) => ({ ...source })),
      assessed_node_count: tfidfRows.length,
      passed_node_count: tfidfRows.filter((row) => row.status === "pass").length,
      partially_assessed_node_count: tfidfPartials.length,
      failed_node_count: tfidfFailures.length,
      exact_static_node_count: tfidfRows.filter((row) => row.static_execution_status === "assessed_exact").length,
      exact_ngram_definition_count: tfidfRows.reduce((sum, row) => sum + Number(row.exact_ngram_definition_count || 0), 0),
      exact_active_ngram_definition_count: tfidfRows.reduce((sum, row) => sum + Number(row.exact_active_ngram_definition_count || 0), 0),
      exact_match_count: nullableSum(tfidfRows, "exact_match_count"),
      exact_output_value_count: nullableSum(tfidfRows, "exact_output_value_count"),
      exact_duplicate_output_coordinate_count: nullableSum(tfidfRows, "exact_duplicate_output_coordinate_count"),
      exact_weight_coordinate_value_disagreement_count: nullableSum(tfidfRows, "exact_weight_coordinate_value_disagreement_count"),
      exact_ort_reference_divergent_output_count: nullableSum(tfidfRows, "exact_ort_reference_divergent_output_count"),
      failed_rows: tfidfFailures,
      partial_rows: tfidfPartials,
      rows: tfidfRows,
      method: "Validate TfIdfVectorizer-9 schema/runtime attributes and, for complete bounded static inputs, reproduce every skip-distance n-gram hit plus pinned ORT CPU FLOAT32 weighting order.",
      interpretation_boundary: "Shape and attribute results are artifact-derived. Exact output values require complete static tokens and bounded work; provider inclusion, optimized assignment, runtime token distributions, and reduced ORT Web/WASM builds remain separate evidence.",
    },
    container_value_inference: {
      schema: "deepbom.onnx_container_value_inference.v1",
      evidence_class: "SOURCE_PINNED_AND_DERIVED",
      status: containerFailures.length ? "fail" : containerPartials.length ? "partial" : containerRows.length ? "assessed" : "not_applicable",
      source_release: ONNX_CONTAINER_VALUE_SOURCE.release,
      source_commit: ONNX_CONTAINER_VALUE_SOURCE.commit,
      source_documents: ONNX_CONTAINER_VALUE_SOURCE.documents.map((source) => ({ ...source })),
      assessed_node_count: containerRows.length,
      passed_node_count: containerRows.filter((row) => row.status === "pass").length,
      partially_assessed_node_count: containerPartials.length,
      failed_node_count: containerFailures.length,
      exact_sequence_length_output_count: containerRows.reduce((sum, row) => sum + (row.sequence_lengths || []).filter((value) => value != null).length, 0),
      exact_optional_presence_output_count: containerRows.reduce((sum, row) => sum + (row.optional_presence || []).filter((value) => value != null).length, 0),
      failed_rows: containerFailures,
      partial_rows: containerPartials,
      rows: containerRows,
      method: "Execute the pinned ONNX Sequence and Optional TypeAndShapeInferenceFunction contracts over parsed TypeProto inputs; preserve exact sequence length, element inventory, optional presence, static scalar results, and constant index bounds when artifact values determine them.",
      interpretation_boundary: "SequenceMap, non-dense If unions, and schema-versioned non-dense Loop state are delegated to the recursive scope engine. Binarizer, Normalizer, Scaler, Imputer, ZipMap, CastMap, DictVectorizer, CategoryMapper, FeatureVectorizer, and ArrayFeatureExtractor are delegated to the ONNX-ML value-contract engine. Scan is tensor-only under the pinned schemas. Dynamic Loop control, runtime sequence/map contents, dynamic positions, and runtime feature indices remain explicit residuals. No sequence, optional, map, or sparse value is projected into dense payload, MAC, liveness, or provider arithmetic.",
    },
    ml_value_inference: {
      schema: "deepbom.onnx_ml_value_inference.v1.12",
      evidence_class: "SOURCE_PINNED_AND_DERIVED",
      status: mlValueFailures.length ? "fail" : mlValuePartials.length ? "partial" : mlValueRows.length ? "assessed" : "not_applicable",
      source_release: ONNX_ML_VALUE_SOURCE.release,
      source_commit: ONNX_ML_VALUE_SOURCE.commit,
      source_documents: ONNX_ML_VALUE_SOURCE.documents.map((source) => ({ ...source })),
      runtime_reference_commit: ONNX_ML_VALUE_SOURCE.runtime_reference_commit,
      runtime_reference_documents: ONNX_ML_VALUE_SOURCE.runtime_reference_documents.map((source) => ({ ...source })),
      assessed_node_count: mlValueRows.length,
      passed_node_count: mlValueRows.filter((row) => row.status === "pass").length,
      partially_assessed_node_count: mlValuePartials.length,
      failed_node_count: mlValueFailures.length,
      exact_sequence_length_output_count: mlValueRows.filter((row) => row.status !== "fail"
        && Number.isSafeInteger(row.exact_output_sequence_length) && row.exact_output_sequence_length >= 0).length,
      exact_class_key_count: mlValueRows.reduce((sum, row) => sum + Number(row.class_key_count || 0), 0),
      duplicate_class_key_count: mlValueRows.reduce((sum, row) => sum + Number(row.duplicate_key_count || 0), 0),
      duplicate_class_key_node_count: mlValueRows.filter((row) => Number(row.duplicate_key_count || 0) > 0).length,
      map_producer_node_count: mlValueRows.filter((row) => row.contract_kind === "map_producer").length,
      map_consumer_node_count: mlValueRows.filter((row) => row.contract_kind === "map_consumer").length,
      tensor_mapper_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_mapper").length,
      tensor_aggregator_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_aggregator").length,
      tensor_selector_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_selector").length,
      tensor_normalization_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_normalization").length,
      tensor_affine_scaler_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_affine_scaler").length,
      tensor_imputation_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_imputation").length,
      tensor_encoder_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_encoder").length,
      tensor_label_mapping_node_count: mlValueRows.filter((row) => row.contract_kind === "tensor_label_mapping").length,
      linear_model_node_count: mlValueRows.filter((row) => ["linear_classifier", "linear_regressor"].includes(row.contract_kind)).length,
      svm_model_node_count: mlValueRows.filter((row) => ["svm_classifier", "svm_regressor"].includes(row.contract_kind)).length,
      tree_ensemble_model_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)).length,
      exact_dense_output_shape_count: mlValueRows.filter((row) => row.status !== "fail" && row.output_kind === "tensor"
        && Array.isArray(row.exact_output_shape) && row.exact_output_shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)).length,
      exact_vocabulary_entry_count: mlValueRows.reduce((sum, row) => sum + Number(row.vocabulary_count || 0), 0),
      duplicate_vocabulary_entry_count: mlValueRows.reduce((sum, row) => sum + Number(row.duplicate_vocabulary_count || 0), 0),
      duplicate_vocabulary_node_count: mlValueRows.filter((row) => Number(row.duplicate_vocabulary_count || 0) > 0).length,
      exact_category_pair_count: mlValueRows.reduce((sum, row) => sum + Number(row.category_pair_count || 0), 0),
      duplicate_category_active_key_count: mlValueRows.reduce((sum, row) => sum + Number(row.active_duplicate_key_count || 0), 0),
      duplicate_category_active_key_node_count: mlValueRows.filter((row) => Number(row.active_duplicate_key_count || 0) > 0).length,
      feature_vectorizer_node_count: mlValueRows.filter((row) => row.op_name === "FeatureVectorizer").length,
      feature_vectorizer_exact_width_node_count: mlValueRows.filter((row) => row.op_name === "FeatureVectorizer"
        && Number.isSafeInteger(row.total_configured_feature_count) && row.total_configured_feature_count >= 0).length,
      exact_feature_vectorizer_configured_feature_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "FeatureVectorizer" && Number.isSafeInteger(row.total_configured_feature_count) ? row.total_configured_feature_count : 0), 0),
      feature_vectorizer_truncating_node_count: mlValueRows.filter((row) => row.op_name === "FeatureVectorizer"
        && Number(row.exact_truncated_feature_count_per_batch || 0) > 0).length,
      exact_feature_vectorizer_truncated_feature_count_per_batch: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "FeatureVectorizer" && Number.isSafeInteger(row.exact_truncated_feature_count_per_batch) ? row.exact_truncated_feature_count_per_batch : 0), 0),
      exact_feature_vectorizer_padded_feature_count_per_batch: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "FeatureVectorizer" && Number.isSafeInteger(row.exact_padded_feature_count_per_batch) ? row.exact_padded_feature_count_per_batch : 0), 0),
      array_feature_extractor_node_count: mlValueRows.filter((row) => row.op_name === "ArrayFeatureExtractor").length,
      array_feature_extractor_exact_index_node_count: mlValueRows.filter((row) => row.op_name === "ArrayFeatureExtractor"
        && Number.isSafeInteger(row.exact_index_count) && row.exact_index_count >= 0).length,
      exact_array_feature_extractor_index_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "ArrayFeatureExtractor" && Number.isSafeInteger(row.exact_index_count) ? row.exact_index_count : 0), 0),
      array_feature_extractor_duplicate_index_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "ArrayFeatureExtractor" ? Number(row.duplicate_index_count || 0) : 0), 0),
      array_feature_extractor_bounds_assessed_node_count: mlValueRows.filter((row) => row.op_name === "ArrayFeatureExtractor"
        && ["assessed_pass", "fail"].includes(row.index_bounds_status)).length,
      array_feature_extractor_bounds_failure_node_count: mlValueRows.filter((row) => row.op_name === "ArrayFeatureExtractor"
        && row.index_bounds_status === "fail").length,
      binarizer_node_count: mlValueRows.filter((row) => row.op_name === "Binarizer").length,
      binarizer_exact_static_node_count: mlValueRows.filter((row) => row.op_name === "Binarizer"
        && row.static_value_assessment_status === "assessed_exact").length,
      exact_binarizer_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Binarizer" && Number.isSafeInteger(row.exact_static_input_value_count) ? row.exact_static_input_value_count : 0), 0),
      exact_binarizer_above_threshold_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Binarizer" && Number.isSafeInteger(row.exact_above_threshold_count) ? row.exact_above_threshold_count : 0), 0),
      exact_binarizer_at_or_below_threshold_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Binarizer" && Number.isSafeInteger(row.exact_at_or_below_threshold_count) ? row.exact_at_or_below_threshold_count : 0), 0),
      exact_binarizer_equal_threshold_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Binarizer" && Number.isSafeInteger(row.exact_equal_threshold_count) ? row.exact_equal_threshold_count : 0), 0),
      binarizer_schema_default_threshold_node_count: mlValueRows.filter((row) => row.op_name === "Binarizer"
        && row.threshold_source === "onnx_schema_default_0").length,
      binarizer_nonfinite_threshold_node_count: mlValueRows.filter((row) => row.op_name === "Binarizer"
        && row.threshold_finite === false).length,
      normalizer_node_count: mlValueRows.filter((row) => row.op_name === "Normalizer").length,
      normalizer_static_assessed_node_count: mlValueRows.filter((row) => row.op_name === "Normalizer"
        && String(row.normalizer_static_assessment_status || "").startsWith("assessed_")).length,
      normalizer_output_materialized_node_count: mlValueRows.filter((row) => row.op_name === "Normalizer"
        && row.normalizer_output_materialized === true).length,
      exact_normalizer_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_exact_input_value_count) ? row.normalizer_exact_input_value_count : 0), 0),
      exact_normalizer_zero_divisor_row_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_zero_divisor_row_count) ? row.normalizer_zero_divisor_row_count : 0), 0),
      exact_normalizer_negative_max_divisor_row_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_negative_max_divisor_row_count) ? row.normalizer_negative_max_divisor_row_count : 0), 0),
      exact_normalizer_integer_float32_rounding_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_integer_float32_rounding_count) ? row.normalizer_integer_float32_rounding_count : 0), 0),
      exact_normalizer_signed_overflow_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_signed_overflow_value_count) ? row.normalizer_signed_overflow_value_count : 0), 0),
      exact_normalizer_non_finite_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_non_finite_output_count) ? row.normalizer_non_finite_output_count : 0), 0),
      exact_normalizer_signed_zero_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Normalizer" && Number.isSafeInteger(row.normalizer_signed_zero_output_count) ? row.normalizer_signed_zero_output_count : 0), 0),
      normalizer_schema_default_mode_node_count: mlValueRows.filter((row) => row.op_name === "Normalizer"
        && row.normalizer_mode_source === "onnx_schema_default_MAX").length,
      scaler_node_count: mlValueRows.filter((row) => row.op_name === "Scaler").length,
      scaler_static_assessed_node_count: mlValueRows.filter((row) => row.op_name === "Scaler"
        && String(row.scaler_static_assessment_status || "").startsWith("assessed_")).length,
      scaler_output_materialized_node_count: mlValueRows.filter((row) => row.op_name === "Scaler"
        && row.scaler_output_materialized === true).length,
      scaler_invalid_runtime_contract_node_count: mlValueRows.filter((row) => row.op_name === "Scaler"
        && row.scaler_parameter_contract_status === "fail").length,
      exact_scaler_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" && Number.isSafeInteger(row.scaler_exact_input_value_count) ? row.scaler_exact_input_value_count : 0), 0),
      exact_scaler_integer_float32_rounding_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" && Number.isSafeInteger(row.scaler_integer_float32_rounding_count) ? row.scaler_integer_float32_rounding_count : 0), 0),
      exact_scaler_non_finite_parameter_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" && Number.isSafeInteger(row.scaler_non_finite_parameter_count) ? row.scaler_non_finite_parameter_count : 0), 0),
      exact_scaler_non_finite_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" && Number.isSafeInteger(row.scaler_non_finite_output_count) ? row.scaler_non_finite_output_count : 0), 0),
      exact_scaler_signed_zero_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" && Number.isSafeInteger(row.scaler_signed_zero_output_count) ? row.scaler_signed_zero_output_count : 0), 0),
      exact_scaler_zero_scale_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Scaler" ? Number(row.scaler_zero_scale_count || 0) : 0), 0),
      imputer_node_count: mlValueRows.filter((row) => row.op_name === "Imputer").length,
      imputer_static_assessed_node_count: mlValueRows.filter((row) => row.op_name === "Imputer"
        && String(row.imputer_static_assessment_status || "").startsWith("assessed_")).length,
      imputer_output_materialized_node_count: mlValueRows.filter((row) => row.op_name === "Imputer"
        && row.imputer_output_materialized === true).length,
      imputer_invalid_runtime_contract_node_count: mlValueRows.filter((row) => row.op_name === "Imputer"
        && row.imputer_parameter_contract_status === "fail").length,
      imputer_scalar_first_fallback_node_count: mlValueRows.filter((row) => row.op_name === "Imputer"
        && row.imputer_parameter_mode === "scalar_first_fallback").length,
      imputer_pinned_cpu_dtype_gap_node_count: mlValueRows.filter((row) => row.op_name === "Imputer"
        && (row.risk_codes || []).includes("imputer_schema_dtype_missing_pinned_ort_cpu_kernel")).length,
      exact_imputer_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_exact_input_value_count) ? row.imputer_exact_input_value_count : 0), 0),
      exact_imputer_replacement_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_exact_replacement_count) ? row.imputer_exact_replacement_count : 0), 0),
      exact_imputer_nan_replacement_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_exact_nan_replacement_count) ? row.imputer_exact_nan_replacement_count : 0), 0),
      exact_imputer_unchanged_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_exact_unchanged_count) ? row.imputer_exact_unchanged_count : 0), 0),
      exact_imputer_ignored_imputed_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" ? Number(row.imputer_ignored_imputed_value_count || 0) : 0), 0),
      exact_imputer_non_finite_imputed_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" ? Number(row.imputer_non_finite_imputed_value_count || 0) : 0), 0),
      exact_imputer_non_finite_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_non_finite_output_count) ? row.imputer_non_finite_output_count : 0), 0),
      exact_imputer_signed_zero_output_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "Imputer" && Number.isSafeInteger(row.imputer_signed_zero_output_count) ? row.imputer_signed_zero_output_count : 0), 0),
      onehot_encoder_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder").length,
      onehot_static_assessed_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && String(row.onehot_static_assessment_status || "").startsWith("assessed_")).length,
      onehot_output_materialized_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && row.onehot_output_materialized === true).length,
      onehot_invalid_contract_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && row.onehot_parameter_contract_status === "fail").length,
      onehot_duplicate_vocabulary_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && Number(row.onehot_duplicate_category_count || 0) > 0).length,
      onehot_unknown_all_zero_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && (row.risk_codes || []).includes("onehot_unknown_categories_all_zero_encoding")).length,
      onehot_guaranteed_runtime_failure_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && row.onehot_guaranteed_runtime_failure === true).length,
      onehot_pinned_cpu_dtype_gap_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && (row.risk_codes || []).includes("onehot_schema_dtype_missing_pinned_ort_cpu_kernel")).length,
      onehot_noncanonical_zeros_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && row.onehot_zeros_canonical_boolean === false).length,
      onehot_unrepresentable_numeric_cast_node_count: mlValueRows.filter((row) => row.op_name === "OneHotEncoder"
        && Number(row.onehot_numeric_to_int64_invalid_count || 0) > 0).length,
      exact_onehot_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_exact_input_value_count) ? row.onehot_exact_input_value_count : 0), 0),
      exact_onehot_matched_input_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_exact_matched_input_count) ? row.onehot_exact_matched_input_count : 0), 0),
      exact_onehot_unknown_input_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_exact_unknown_input_count) ? row.onehot_exact_unknown_input_count : 0), 0),
      exact_onehot_numeric_to_int64_changed_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_numeric_to_int64_changed_count) ? row.onehot_numeric_to_int64_changed_count : 0), 0),
      exact_onehot_numeric_to_int64_invalid_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_numeric_to_int64_invalid_count) ? row.onehot_numeric_to_int64_invalid_count : 0), 0),
      exact_onehot_output_one_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_exact_output_one_count) ? row.onehot_exact_output_one_count : 0), 0),
      exact_onehot_output_zero_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" && Number.isSafeInteger(row.onehot_exact_output_zero_count) ? row.onehot_exact_output_zero_count : 0), 0),
      exact_onehot_duplicate_category_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" ? Number(row.onehot_duplicate_category_count || 0) : 0), 0),
      exact_onehot_unreachable_duplicate_column_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "OneHotEncoder" ? Number(row.onehot_unreachable_duplicate_column_count || 0) : 0), 0),
      label_encoder_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder").length,
      label_encoder_static_assessed_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && String(row.label_encoder_static_assessment_status || "").startsWith("assessed_")).length,
      label_encoder_output_materialized_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && row.label_encoder_output_materialized === true).length,
      label_encoder_onnx_contract_failure_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && row.label_encoder_onnx_contract_status === "fail").length,
      label_encoder_pinned_ort_contract_failure_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && row.label_encoder_pinned_ort_contract_status === "fail").length,
      label_encoder_pinned_cpu_dtype_pair_gap_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && (row.risk_codes || []).includes("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel")).length,
      label_encoder_duplicate_semantic_conflict_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && (row.risk_codes || []).includes("label_encoder_v4_schema_last_vs_ort_first_duplicate_conflict")).length,
      label_encoder_nan_semantic_conflict_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && (row.risk_codes || []).includes("label_encoder_v2_schema_bitwise_nan_vs_ort_unmatched")).length,
      label_encoder_default_path_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && Number(row.label_encoder_exact_default_count || 0) > 0).length,
      label_encoder_schema_runtime_output_mismatch_node_count: mlValueRows.filter((row) => row.op_name === "LabelEncoder"
        && Number(row.label_encoder_schema_runtime_mismatch_count || 0) > 0).length,
      exact_label_encoder_key_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" ? Number(row.label_encoder_key_count || 0) : 0), 0),
      exact_label_encoder_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" && Number.isSafeInteger(row.label_encoder_exact_input_value_count) ? row.label_encoder_exact_input_value_count : 0), 0),
      exact_label_encoder_match_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" && Number.isSafeInteger(row.label_encoder_exact_match_count) ? row.label_encoder_exact_match_count : 0), 0),
      exact_label_encoder_default_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" && Number.isSafeInteger(row.label_encoder_exact_default_count) ? row.label_encoder_exact_default_count : 0), 0),
      exact_label_encoder_duplicate_key_hit_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" && Number.isSafeInteger(row.label_encoder_exact_duplicate_key_hit_count) ? row.label_encoder_exact_duplicate_key_hit_count : 0), 0),
      exact_label_encoder_schema_runtime_mismatch_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "LabelEncoder" && Number.isSafeInteger(row.label_encoder_schema_runtime_mismatch_count) ? row.label_encoder_schema_runtime_mismatch_count : 0), 0),
      linear_classifier_node_count: mlValueRows.filter((row) => row.op_name === "LinearClassifier").length,
      linear_regressor_node_count: mlValueRows.filter((row) => row.op_name === "LinearRegressor").length,
      linear_onnx_contract_failure_node_count: mlValueRows.filter((row) => ["LinearClassifier", "LinearRegressor"].includes(row.op_name)
        && row.linear_onnx_contract_status === "fail").length,
      linear_pinned_ort_contract_failure_node_count: mlValueRows.filter((row) => ["LinearClassifier", "LinearRegressor"].includes(row.op_name)
        && row.linear_pinned_ort_contract_status === "fail").length,
      linear_reference_assessed_node_count: mlValueRows.filter((row) => ["LinearClassifier", "LinearRegressor"].includes(row.op_name)
        && String(row.linear_reference_assessment_status || "").startsWith("assessed_")).length,
      linear_pinned_cpu_dtype_gap_node_count: mlValueRows.filter((row) => (row.risk_codes || [])
        .includes("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel")).length,
      linear_post_transform_hazard_node_count: mlValueRows.filter((row) => (row.risk_codes || []).some((code) => [
        "linear_classifier_single_score_post_transform_noop",
        "linear_classifier_binary_probit_second_score_unwritten",
        "linear_classifier_binary_post_transform_ignored_for_complement_expansion",
        "linear_regressor_single_target_post_transform_noop",
        "linear_regressor_probit_may_emit_non_finite",
      ].includes(code))).length,
      linear_unused_coefficient_node_count: mlValueRows.filter((row) => Number(row.linear_unused_coefficient_count || 0) > 0).length,
      linear_ignored_intercept_node_count: mlValueRows.filter((row) => Number(row.linear_ignored_intercept_count || 0) > 0).length,
      exact_linear_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) ? Number(row.linear_coefficient_count || 0) : 0), 0),
      exact_linear_used_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) ? Number(row.linear_used_coefficient_count || 0) : 0), 0),
      exact_linear_unused_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) ? Number(row.linear_unused_coefficient_count || 0) : 0), 0),
      exact_linear_unresolved_coefficient_use_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name)
          ? Math.max(0, Number(row.linear_coefficient_count || 0) - Number(row.linear_used_coefficient_count || 0)
            - Number(row.linear_unused_coefficient_count || 0)) : 0), 0),
      exact_linear_ignored_intercept_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) ? Number(row.linear_ignored_intercept_count || 0) : 0), 0),
      exact_linear_reference_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) && Number.isSafeInteger(row.linear_reference_input_value_count)
          ? row.linear_reference_input_value_count : 0), 0),
      exact_linear_reference_raw_score_count: mlValueRows.reduce((sum, row) => sum
        + (["LinearClassifier", "LinearRegressor"].includes(row.op_name) && Number.isSafeInteger(row.linear_reference_raw_score_count)
          ? row.linear_reference_raw_score_count : 0), 0),
      svm_classifier_node_count: mlValueRows.filter((row) => row.op_name === "SVMClassifier").length,
      svm_regressor_node_count: mlValueRows.filter((row) => row.op_name === "SVMRegressor").length,
      svm_linear_mode_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && row.svm_mode === "linear").length,
      svm_svc_mode_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && row.svm_mode === "svc").length,
      svm_onnx_contract_failure_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && row.svm_onnx_contract_status === "fail").length,
      svm_pinned_ort_contract_failure_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && row.svm_pinned_ort_contract_status === "fail").length,
      svm_regressor_pinned_cpu_dtype_gap_node_count: mlValueRows.filter((row) => row.op_name === "SVMRegressor"
        && (row.risk_codes || []).includes("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel")).length,
      svm_schema_runtime_score_width_mismatch_node_count: mlValueRows.filter((row) => row.op_name === "SVMClassifier"
        && row.svm_schema_runtime_score_width_mismatch === true).length,
      svm_ignored_post_transform_node_count: mlValueRows.filter((row) => row.op_name === "SVMRegressor"
        && row.svm_post_transform_applied_by_pinned_ort === false && row.svm_post_transform !== "NONE").length,
      svm_ignored_parameter_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && Number(row.svm_unused_support_vector_value_count || 0) + Number(row.svm_unused_coefficient_count || 0)
          + Number(row.svm_unused_rho_count || 0) + Number(row.svm_unused_probability_parameter_count || 0) > 0).length,
      svm_non_finite_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && Number(row.svm_non_finite_parameter_count || 0) + Number(row.svm_reference_non_finite_score_count || 0) > 0).length,
      svm_reference_assessed_node_count: mlValueRows.filter((row) => ["SVMClassifier", "SVMRegressor"].includes(row.op_name)
        && String(row.svm_reference_assessment_status || "").startsWith("assessed_")).length,
      exact_svm_vector_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_vector_count || 0) : 0), 0),
      exact_svm_pairwise_classifier_count: mlValueRows.reduce((sum, row) => sum
        + (row.op_name === "SVMClassifier" ? Number(row.svm_pairwise_classifier_count || 0) : 0), 0),
      exact_svm_support_vector_value_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_support_vector_value_count || 0) : 0), 0),
      exact_svm_used_support_vector_value_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_used_support_vector_value_count || 0) : 0), 0),
      exact_svm_unused_support_vector_value_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_unused_support_vector_value_count || 0) : 0), 0),
      exact_svm_unresolved_support_vector_use_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name)
          ? Math.max(0, Number(row.svm_support_vector_value_count || 0) - Number(row.svm_used_support_vector_value_count || 0)
            - Number(row.svm_unused_support_vector_value_count || 0)) : 0), 0),
      exact_svm_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_coefficient_count || 0) : 0), 0),
      exact_svm_used_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_used_coefficient_count || 0) : 0), 0),
      exact_svm_unused_coefficient_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_unused_coefficient_count || 0) : 0), 0),
      exact_svm_unresolved_coefficient_use_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name)
          ? Math.max(0, Number(row.svm_coefficient_count || 0) - Number(row.svm_used_coefficient_count || 0)
            - Number(row.svm_unused_coefficient_count || 0)) : 0), 0),
      exact_svm_rho_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_rho_count || 0) : 0), 0),
      exact_svm_used_rho_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_used_rho_count || 0) : 0), 0),
      exact_svm_unused_rho_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) ? Number(row.svm_unused_rho_count || 0) : 0), 0),
      exact_svm_unresolved_rho_use_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name)
          ? Math.max(0, Number(row.svm_rho_count || 0) - Number(row.svm_used_rho_count || 0)
            - Number(row.svm_unused_rho_count || 0)) : 0), 0),
      exact_svm_reference_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) && Number.isSafeInteger(row.svm_reference_input_value_count)
          ? row.svm_reference_input_value_count : 0), 0),
      exact_svm_reference_raw_score_count: mlValueRows.reduce((sum, row) => sum
        + (["SVMClassifier", "SVMRegressor"].includes(row.op_name) && Number.isSafeInteger(row.svm_reference_raw_score_count)
          ? row.svm_reference_raw_score_count : 0), 0),
      tree_ensemble_node_count: mlValueRows.filter((row) => row.op_name === "TreeEnsemble").length,
      tree_ensemble_classifier_node_count: mlValueRows.filter((row) => row.op_name === "TreeEnsembleClassifier").length,
      tree_ensemble_regressor_node_count: mlValueRows.filter((row) => row.op_name === "TreeEnsembleRegressor").length,
      tree_ensemble_deprecated_node_count: mlValueRows.filter((row) => row.tree_deprecated_operator === true).length,
      tree_ensemble_onnx_contract_failure_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
        && row.tree_onnx_contract_status === "fail").length,
      tree_ensemble_pinned_ort_contract_failure_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
        && row.tree_pinned_ort_contract_status === "fail").length,
      tree_ensemble_pinned_cpu_dtype_gap_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleRegressor"].includes(row.op_name)
        && row.tree_pinned_cpu_dtype_gap === true).length,
      tree_ensemble_reference_assessed_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
        && String(row.tree_reference_assessment_status || "").startsWith("assessed_")).length,
      tree_ensemble_non_finite_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
        && Number(row.tree_non_finite_parameter_count || 0) + Number(row.tree_reference_non_finite_score_count || 0) > 0).length,
      tree_ensemble_reference_boundary_node_count: mlValueRows.filter((row) => ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(row.op_name)
        && Number(row.tree_reference_decision_boundary_count || 0) > 0).length,
      tree_ensemble_semantic_hazard_node_count: mlValueRows.filter((row) => (row.risk_codes || []).some((risk) => [
        "tree_ensemble_nonleaf_weights_ignored_by_pinned_ort",
        "tree_ensemble_single_target_additional_leaf_weights_ignored_by_pinned_ort",
        "tree_classifier_binary_post_transform_leaves_score_unwritten",
        "tree_regressor_single_target_post_transform_noop",
        "tree_ensemble_v5_single_target_post_transform_noop",
        "tree_classifier_binary_single_base_value_semantics_underspecified",
        "tree_classifier_pinned_ort_binary_label_index_semantics",
        "tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser",
      ].includes(risk))).length,
      exact_tree_ensemble_tree_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_exact_tree_count || 0), 0),
      exact_tree_ensemble_root_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_exact_root_count || 0), 0),
      exact_tree_ensemble_node_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_exact_node_count || 0), 0),
      exact_tree_ensemble_branch_node_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_exact_branch_node_count || 0), 0),
      exact_tree_ensemble_leaf_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_exact_leaf_count || 0), 0),
      exact_tree_ensemble_reachable_node_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_reachable_node_count || 0), 0),
      exact_tree_ensemble_reachable_leaf_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_reachable_leaf_count || 0), 0),
      exact_tree_ensemble_orphan_node_or_leaf_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_orphan_node_or_leaf_count || 0), 0),
      maximum_tree_ensemble_depth: mlValueRows.reduce((maximum, row) => Math.max(maximum, Number(row.tree_max_depth || 0)), 0),
      exact_tree_ensemble_cycle_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_cycle_count || 0), 0),
      exact_tree_ensemble_duplicate_node_identity_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_duplicate_node_identity_count || 0), 0),
      exact_tree_ensemble_invalid_child_reference_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_invalid_child_reference_count || 0), 0),
      exact_tree_ensemble_invalid_feature_id_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_invalid_feature_id_count || 0), 0),
      exact_tree_ensemble_root_mismatch_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_root_mismatch_count || 0), 0),
      exact_tree_ensemble_multiple_parent_node_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_multiple_parent_node_count || 0), 0),
      exact_tree_ensemble_weight_tuple_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_weight_tuple_count || 0), 0),
      exact_tree_ensemble_used_weight_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_used_weight_count || 0), 0),
      exact_tree_ensemble_unused_weight_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_unused_weight_count || 0), 0),
      exact_tree_ensemble_unresolved_weight_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_unresolved_weight_count || 0), 0),
      exact_tree_ensemble_ignored_nonleaf_weight_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_ignored_nonleaf_weight_count || 0), 0),
      exact_tree_ensemble_invalid_weight_reference_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_invalid_weight_reference_count || 0), 0),
      exact_tree_ensemble_invalid_weight_id_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_invalid_weight_id_count || 0), 0),
      exact_tree_ensemble_single_target_ignored_weight_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_single_target_ignored_weight_count || 0), 0),
      exact_tree_ensemble_membership_node_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_membership_node_count || 0), 0),
      exact_tree_ensemble_membership_set_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_membership_set_count || 0), 0),
      exact_tree_ensemble_membership_value_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_membership_value_count || 0), 0),
      exact_tree_ensemble_membership_duplicate_value_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_membership_duplicate_value_count || 0), 0),
      exact_tree_ensemble_membership_separator_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_membership_separator_count || 0), 0),
      exact_tree_ensemble_non_finite_parameter_count: mlValueRows.reduce((sum, row) => sum + Number(row.tree_non_finite_parameter_count || 0), 0),
      exact_tree_ensemble_reference_input_value_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_input_value_count) ? row.tree_reference_input_value_count : 0), 0),
      exact_tree_ensemble_reference_row_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_row_count) ? row.tree_reference_row_count : 0), 0),
      exact_tree_ensemble_reference_path_step_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_path_step_count) ? row.tree_reference_path_step_count : 0), 0),
      exact_tree_ensemble_reference_raw_score_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_raw_score_count) ? row.tree_reference_raw_score_count : 0), 0),
      exact_tree_ensemble_reference_output_score_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_output_score_count) ? row.tree_reference_output_score_count : 0), 0),
      exact_tree_ensemble_reference_non_finite_score_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_non_finite_score_count) ? row.tree_reference_non_finite_score_count : 0), 0),
      exact_tree_ensemble_reference_decision_boundary_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_decision_boundary_count) ? row.tree_reference_decision_boundary_count : 0), 0),
      exact_tree_ensemble_reference_unwritten_score_count: mlValueRows.reduce((sum, row) => sum
        + (Number.isSafeInteger(row.tree_reference_unwritten_score_count) ? row.tree_reference_unwritten_score_count : 0), 0),
      failed_rows: mlValueFailures,
      partial_rows: mlValuePartials,
      rows: mlValueRows,
      method: "Resolve the pinned ai.onnx.ml value-contract inventory, including legacy TreeEnsembleClassifier/Regressor-1/3/5 and indexed TreeEnsemble-5. Validate tree tuple/tensor cardinality, roots, branch and leaf references, cycles, reachability, feature bounds, class/target weights, aggregate/post-transform state, and MEMBER-set delimiters; derive output TypeProto/dtype/rank/cardinality; evaluate bounded initializer paths and scalar source-order scores; and cross-check operational behavior against separately pinned ONNX reference and ORT CPU sources/tests.",
      interpretation_boundary: "Tree structure, serialized-parameter conservation, path selection for artifact-known inputs, and source-order scalar references are deterministic. Score previews are not runtime-bit-exact because the executed ORT thread partition, floating reduction order, optimized graph, selected EP, and platform libm are not observed; they are never propagated as static tensors. TreeEnsemble-5 FLOAT16 and legacy TreeEnsembleRegressor INT32/INT64 remain schema-valid CPU registration gaps. Dynamic inputs retain exact structure without invented values. Proven invalid runtime contracts suppress output propagation. Remaining unsupported ai.onnx.ml algebra is never coerced into ai.onnx tensor rules.",
    },
    shape_scope: scopeLedger,
    declaration_conflict_count: declarationConflicts.length,
    declaration_conflicts: declarationConflicts,
    semantic_contract_conflict_count: semanticContractConflicts.length,
    semantic_contract_conflicts: semanticContractConflicts,
    propagated_static_value_tensor_count: propagatedValueTensorCount,
    propagated_symbolic_shape_value_tensor_count: propagatedSymbolicShapeValueTensorCount,
    node_output_count: nodeOutputCount,
    tensor_node_output_count: tensorNodeOutputCount,
    known_node_output_count: knownNodeOutputCount,
    unknown_node_output_count: unknownNodeOutputCount,
    shape_contract_known_node_output_count: shapeContractKnownNodeOutputCount,
    shape_contract_unknown_node_output_count: shapeContractUnknownNodeOutputCount,
    invalid_node_output_count: invalidNodeOutputCount,
    conditionally_invalid_node_output_count: conditionallyInvalidNodeOutputCount,
    conditional_invalid_variant_count: conditionalInvalidVariantCount,
    conditional_unassessed_variant_count: conditionalUnassessedVariantCount,
    unresolved_nonconflict_shape_contract_node_output_count: shapeContractUnknownNodeOutputCount - invalidNodeOutputCount - conditionallyInvalidNodeOutputCount,
    blocked_by_upstream_contract_conflict_node_count: blockedByUpstreamContractConflictNodeCount,
    conditional_shape_contract_node_output_count: nodeOutputNames.filter((name) => {
      const tensor = tensorMap.get(name);
      return !unconditionalShapeContractKnown(tensor) && conditionalShapeContractKnown(tensor);
    }).length,
    partial_conditional_shape_contract_node_output_count: nodeOutputNames.filter((name) => (
      tensorMap.get(name)?.conditionalShapeContract?.status === "assessed_partial"
    )).length,
    symbolic_shape_contract_node_output_count: nodeOutputNames.filter((name) => {
      const tensor = tensorMap.get(name);
      return !tensorKnown(tensor) && unconditionalShapeContractKnown(tensor);
    }).length,
    non_dense_node_output_count: nonDenseNodeOutputCount,
    non_dense_node_output_names: nonDenseNodeOutputNames,
    known_non_dense_node_output_count: knownNonDenseNodeOutputCount,
    known_non_dense_node_output_names: knownNonDenseNodeOutputNames,
    unresolved_non_dense_node_output_count: unresolvedNonDenseNodeOutputCount,
    known_value_node_output_count: knownNodeOutputCount + knownNonDenseNodeOutputCount,
    node_value_assessment_ratio: nodeOutputCount ? (knownNodeOutputCount + knownNonDenseNodeOutputCount) / nodeOutputCount : 1,
    node_output_assessment_ratio: tensorNodeOutputCount ? knownNodeOutputCount / tensorNodeOutputCount : 1,
    inferred_outputs: inferredOutputs,
    inferred_non_dense_outputs: inferredNonDenseOutputs,
    unknown_tensor_count: unknownTensorIndices.length,
    unknown_tensor_indices: unknownTensorIndices,
    non_dense_value_count: nonDenseValueIndices.length,
    non_dense_value_indices: nonDenseValueIndices,
    supported_ops: [...ONNX_SHAPE_INFERENCE_OPS].sort(),
    supported_ml_ops: [...ONNX_ML_VALUE_OPS].sort(),
    symbolic_dimension_method: "Preserve ONNX dim_param identity, runtime-value symbols, finite If branch variants, and exact derived expressions through supported shape-data, spatial, Slice, NonZero, Range, Reshape, Flatten, ConvTranspose, Gemm, and MatMul rules. Fixed-cardinality runtime Slice starts/ends become explicit slice_len expressions; only dtype-specific INT32/INT64 extrema are open-bound sentinels. NonZero emits a runtime NNZ symbol with exact zero lower bound and a logical-input-cardinality upper expression. Rank-varying If outputs keep the ordinary ONNX union type plus a separate finite conditional contract and propagate each compatible condition through downstream rules. A symbolic dimension is never replaced by 1. Deterministically incompatible artifact-known operator contracts fail; dependent nodes retain the root conflict instead of being relabeled as generic unknowns.",
    method: "Resolve each supported ai.onnx or ai.onnx.ml node to a pinned schema not exceeding the imported domain opset; validate formal input/output cardinality, omission, attribute presence, and AttributeProto type; then walk the topologically ordered graph with pinned tensor, unconditional symbolic, finite conditional-shape, runtime-value dimension, Sequence/Optional, ONNX-ML value-contract, and bounded exact static-value rules while rejecting contradictory declarations and artifact-known semantic violations.",
    interpretation_boundary: "The formal OpSchema contract is checked for the emitted local rule set. Conditional contracts enumerate only finite artifact-derived branch alternatives; they do not select a runtime branch or invent a numeric extent. Runtime-value symbols require a deployment/profile binding before numeric MAC, payload, or memory evaluation. The recursive extension evaluates FunctionProto, If, Loop, Scan-8/9+, SequenceMap, non-dense If unions, Loop-13 sequence state, and Loop-16+ optional state; bounded exact Loop expansion runs only when artifact-known control values close every reached iteration. Remaining ai.onnx.ml algebra, sparse-value operator algebra, unsupported operators, conditional variants that exceed the explicit bound, bounded-work overflows, and runtime rewrites remain residuals.",
  };
}

function declaredNonDenseValue(tensor) {
  const kind = String(tensor?.valueKind || "");
  return Boolean(kind) && kind !== "tensor" && kind !== "unresolved" && kind !== "undefined";
}

function buildShapeScopeLedger(graph, functions, domainAnalysis, scopeExecution = null) {
  const normalizedFunctions = Array.isArray(domainAnalysis?.functions) ? domainAnalysis.functions : (functions || []).map((fn, index) => ({
    id: functionId(fn.domain, fn.name, fn.overload),
    index,
    body_node_count: (fn.nodes || []).length,
    local_function_dependencies: [],
  }));
  const functionById = new Map(normalizedFunctions.map((fn) => [fn.id, fn]));
  const functionIds = new Set(functionById.keys());
  const duplicateIds = new Set(domainAnalysis?.duplicate_function_ids || []);
  const graphScopes = [];
  const localCalls = [];
  collectShapeScopes(graph?.nodes || [], "main_graph", "main_graph", functionIds, graphScopes, localCalls);
  for (const fn of functions || []) {
    const id = functionId(fn.domain, fn.name, fn.overload);
    collectShapeScopes(fn.nodes || [], `function:${id}`, id, functionIds, graphScopes, localCalls);
    for (const [attributeIndex, attribute] of (fn.attributeProtos || []).entries()) {
      const attributeName = attribute.name || `attribute_${attributeIndex}`;
      const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
      for (const [graphIndex, nested] of graphs.entries()) {
        const suffix = graphs.length === 1 ? attributeName : `${attributeName}[${graphIndex}]`;
        const scope = `function:${id}/default_attribute:${suffix}`;
        graphScopes.push({ scope, owner: id, node_count: (nested.nodes || []).length, definition_only: true });
        collectShapeScopes(nested.nodes || [], scope, id, functionIds, graphScopes, localCalls, true);
      }
    }
  }

  const callsByOwner = new Map();
  for (const call of localCalls) {
    const rows = callsByOwner.get(call.owner) || [];
    rows.push(call);
    callsByOwner.set(call.owner, rows);
  }
  const mainCalls = callsByOwner.get("main_graph") || [];
  const reachableFunctions = new Set();
  const queue = mainCalls.map((row) => row.function_id).filter(Boolean);
  while (queue.length) {
    const id = queue.shift();
    if (!id || reachableFunctions.has(id)) continue;
    reachableFunctions.add(id);
    for (const dependency of (callsByOwner.get(id) || []).map((row) => row.function_id)) queue.push(dependency);
  }
  const reachableCalls = localCalls.filter((row) => row.owner === "main_graph" || reachableFunctions.has(row.owner));
  const reachableGraphScopes = graphScopes.filter((scope) => !scope.definition_only
    && (scope.owner === "main_graph" || reachableFunctions.has(scope.owner)));
  const reachableFunctionRows = normalizedFunctions.filter((fn) => reachableFunctions.has(fn.id));
  const reachableCycles = (domainAnalysis?.recursive_function_cycles || []).filter((cycle) => cycle.some((id) => reachableFunctions.has(id)));
  const reachableDuplicateIds = [...duplicateIds].filter((id) => reachableFunctions.has(id));
  const executionByScope = scopeExecution?.rows instanceof Map ? scopeExecution.rows : new Map();
  const structurallyNamedScopes = new Set([
    ...reachableGraphScopes.map((item) => item.scope),
    ...reachableFunctionRows.map((fn) => `function:${fn.id}`),
  ]);
  const executionOnlyRows = [...executionByScope.entries()]
    .filter(([scope]) => !structurallyNamedScopes.has(scope))
    .map(([scope, execution]) => scopeAssessmentRow(
      scope,
      execution.scope_class || "nested_graph",
      "bound_recursive_execution",
      Number(execution.node_count || 0),
      execution,
    ));
  const scopeRows = [
    ...reachableGraphScopes.map((item) => scopeAssessmentRow(item.scope, "nested_graph", item.owner, item.node_count, executionByScope.get(item.scope))),
    ...reachableFunctionRows.map((fn) => scopeAssessmentRow(`function:${fn.id}`, "local_function_body", fn.id, fn.body_node_count, executionByScope.get(`function:${fn.id}`))),
    ...executionOnlyRows,
  ].sort((left, right) => left.scope.localeCompare(right.scope) || left.scope_class.localeCompare(right.scope_class));
  const reachableNestedGraphRows = scopeRows.filter((row) => row.scope_class === "nested_graph");
  const reachableFunctionScopeRows = scopeRows.filter((row) => row.scope_class === "local_function_body");
  const exclusions = scopeRows.filter((row) => row.unassessed_node_count > 0 || row.execution_count === 0 || row.status === "fail").map((row) => ({
    scope_class: row.scope_class,
    scope: row.scope,
    owner: row.owner,
    node_count: row.unassessed_node_count,
    reason_code: row.status === "fail"
      ? "reachable_scope_shape_inference_failed"
      : row.execution_count > 0 ? "reachable_scope_shape_inference_residual" : "reachable_scope_shape_inference_not_executed",
    reason_codes: row.reason_codes,
  }));
  const unassessedNodeCount = exclusions.reduce((sum, row) => sum + row.node_count, 0);
  const registryFail = reachableCycles.length > 0 || reachableDuplicateIds.length > 0;
  const scopeFail = scopeRows.some((row) => row.status === "fail");
  return {
    schema: "deepbom.onnx_shape_scope.v2.1",
    status: registryFail || scopeFail ? "fail" : exclusions.length ? "partial" : scopeRows.length ? "assessed_reachable_scope" : "assessed_main_graph_scope",
    registry_status: registryFail ? "fail" : "pass",
    main_graph_node_count: (graph?.nodes || []).length,
    nested_graph_count: graphScopes.length,
    nested_graph_node_count: graphScopes.reduce((sum, scope) => sum + scope.node_count, 0),
    function_default_graph_count: graphScopes.filter((scope) => scope.definition_only).length,
    function_default_graph_node_count: graphScopes.filter((scope) => scope.definition_only).reduce((sum, scope) => sum + scope.node_count, 0),
    reachable_scope_count: scopeRows.length,
    reachable_nested_graph_count: reachableNestedGraphRows.length,
    reachable_nested_graph_node_count: reachableNestedGraphRows.reduce((sum, row) => sum + row.node_count, 0),
    local_function_definition_count: normalizedFunctions.length,
    local_function_body_node_count: normalizedFunctions.reduce((sum, fn) => sum + Number(fn.body_node_count || 0), 0),
    reachable_local_function_definition_count: reachableFunctionScopeRows.length,
    reachable_local_function_body_node_count: reachableFunctionScopeRows.reduce((sum, row) => sum + row.node_count, 0),
    local_function_call_count: localCalls.length,
    reachable_local_function_call_count: reachableCalls.length,
    reachable_recursive_function_cycle_count: reachableCycles.length,
    reachable_recursive_function_cycles: reachableCycles,
    reachable_duplicate_function_id_count: reachableDuplicateIds.length,
    reachable_duplicate_function_ids: reachableDuplicateIds,
    executed_reachable_scope_count: scopeRows.filter((row) => row.execution_count > 0).length,
    fully_assessed_reachable_scope_count: scopeRows.filter((row) => row.status === "assessed").length,
    partially_assessed_reachable_scope_count: scopeRows.filter((row) => row.status === "partial").length,
    failed_reachable_scope_count: scopeRows.filter((row) => row.status === "fail").length,
    reachable_scope_unresolved_output_count: scopeRows.reduce((sum, row) => sum + row.unresolved_output_count, 0),
    scope_execution_rows: scopeRows,
    reachable_exclusion_count: exclusions.length,
    unassessed_reachable_node_count: unassessedNodeCount,
    exclusions,
    method: "Inventory every nested GraphProto attribute, FunctionProto default graph, and FunctionProto body; close structural reachability from the main graph; then reconcile that inventory with recursive scope execution. Function default graphs remain definition-only until a bound call executes them. Reachable nested-graph and local-function counts are reconstructed from the emitted scope rows, including execution-only dependencies reached through a bound default graph. Count only unsupported, schema-invalid, or failed-rule nodes as unassessed residuals; separately preserve unresolved output counts.",
    interpretation_boundary: "Definition inventory and reachable execution scopes are distinct cardinalities: one default GraphProto definition can be bound at multiple invocation sites, so reachable nested-graph scope counts are not constrained to be less than definition counts. Unused local-function and default-graph definitions do not reduce main-graph completeness. Non-tensor TypeProto variants and unresolved symbolic or runtime dimensions remain explicit unresolved outputs rather than unassessed nodes.",
  };
}

function scopeAssessmentRow(scope, scopeClass, owner, nodeCount, execution) {
  if (!execution) {
    return {
      scope,
      scope_class: scopeClass,
      owner,
      status: nodeCount ? "not_assessed" : "assessed",
      node_count: Number(nodeCount || 0),
      execution_count: 0,
      assessed_node_count: 0,
      unassessed_node_count: Number(nodeCount || 0),
      unresolved_output_count: 0,
      reason_codes: nodeCount ? ["reachable_scope_shape_inference_not_executed"] : [],
    };
  }
  const statuses = execution.statuses || [];
  const status = statuses.includes("fail") ? "fail" : statuses.includes("partial") ? "partial" : "assessed";
  const unassessed = Math.min(Number(nodeCount || 0), Number(execution.unassessed_node_count || 0));
  return {
    scope,
    scope_class: scopeClass,
    owner,
    status,
    node_count: Number(nodeCount || 0),
    execution_count: Number(execution.execution_count || 0),
    assessed_node_count: Math.max(0, Number(nodeCount || 0) - unassessed),
    unassessed_node_count: unassessed,
    unresolved_output_count: Number(execution.unresolved_output_count || 0),
    reason_codes: [...(execution.reason_codes || [])].sort(),
  };
}

function collectShapeScopes(nodes, prefix, owner, functionIds, graphOutput, callOutput, definitionOnly = false) {
  for (const [nodeIndex, node] of (nodes || []).entries()) {
    const calledFunction = functionId(node.domain, node.opType, node.overload);
    if (!definitionOnly && functionIds.has(calledFunction)) {
      callOutput.push({
        scope: prefix,
        owner,
        scope_node_index: nodeIndex,
        function_id: calledFunction,
      });
    }
    for (const [attributeName, attribute] of node.attributes || []) {
      const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
      for (const [graphIndex, nested] of graphs.entries()) {
        const suffix = graphs.length === 1 ? attributeName : `${attributeName}[${graphIndex}]`;
        const scope = `${prefix}/node:${nodeIndex}/attribute:${suffix}`;
        graphOutput.push({ scope, owner, node_count: (nested.nodes || []).length, definition_only: definitionOnly });
        collectShapeScopes(nested.nodes || [], scope, owner, functionIds, graphOutput, callOutput, definitionOnly);
      }
    }
  }
}

function functionId(domain, name, overload) {
  return `${normalizeDomain(domain)}::${String(name || "")}::${String(overload || "")}`;
}

function inferNodeWithConditionalShapes(node, tensors, tensorTypeName) {
  const base = inferNode(node, tensors, tensorTypeName);
  const variantInputs = (node.inputs || []).map((name, inputIndex) => ({
    name,
    inputIndex,
    variants: name ? propagatableConditionalShapeVariants(tensors.get(name)) : [],
  })).filter((row) => row.variants.length);
  if (!variantInputs.length) return base;

  let combinations = [{ replacements: [], conditions: [] }];
  for (const input of variantInputs) {
    const expanded = [];
    for (const combination of combinations) {
      for (const variant of input.variants) {
        const conditions = mergeConditionalConditions(combination.conditions, variant.conditions || []);
        if (!conditions) continue;
        expanded.push({
          replacements: [...combination.replacements, { name: input.name, patch: variant }],
          conditions,
        });
        if (expanded.length > MAX_CONDITIONAL_SHAPE_VARIANTS) {
          return { ...base, reason: base.reason || "conditional_shape_variant_limit_exceeded" };
        }
      }
    }
    combinations = expanded;
  }
  if (!combinations.length) return invalid("conditional_shape_conditions_have_no_compatible_assignment");

  const outputVariants = new Map();
  const variantFailures = deduplicateConditionalFailures(variantInputs.flatMap((input) => (
    tensors.get(input.name)?.conditionalShapeContract?.variant_failures || []
  )));
  for (const combination of combinations) {
    const scoped = new Map(tensors);
    for (const replacement of combination.replacements) {
      const existing = tensors.get(replacement.name) || {};
      const patch = { ...replacement.patch };
      delete patch.conditions;
      delete patch.conditionalShapeVariants;
      scoped.set(replacement.name, { ...existing, ...patch, name: replacement.name, contractStatus: "assessed_conditional_variant" });
    }
    const result = inferNode(node, scoped, tensorTypeName);
    if (result.status === "invalid") {
      variantFailures.push({
        status: "invalid",
        reason: result.reason || "conditional_shape_variant_invalid",
        conditions: combination.conditions,
        details: result.details || null,
      });
      continue;
    }
    if (!result.outputs.length && (node.outputs || []).some(Boolean)) {
      variantFailures.push({
        status: "not_assessed",
        reason: result.reason || base.reason || "conditional_shape_variant_not_inferred",
        conditions: combination.conditions,
        details: null,
      });
      continue;
    }
    const outputByName = new Map(result.outputs);
    const incompleteOutput = (node.outputs || []).filter(Boolean)
      .find((name) => {
        const output = outputByName.get(name);
        return !unconditionalShapeContractKnown(output) && !propagatableConditionalShapeVariants(output).length;
      });
    if (incompleteOutput) {
      variantFailures.push({
        status: "not_assessed",
        reason: result.reason || "conditional_shape_variant_output_incomplete",
        output_name: incompleteOutput,
        conditions: combination.conditions,
        details: null,
      });
      continue;
    }
    for (const [name, patch] of result.outputs) {
      const rows = outputVariants.get(name) || [];
      const nestedVariants = propagatableConditionalShapeVariants(patch);
      if (nestedVariants.length) {
        for (const nested of nestedVariants) {
          const conditions = mergeConditionalConditions(combination.conditions, nested.conditions || []);
          if (!conditions) continue;
          const flattened = { ...nested, conditions };
          delete flattened.conditionalShapeVariants;
          delete flattened.conditionalShapeContract;
          rows.push(flattened);
        }
        for (const failure of patch.conditionalShapeContract?.variant_failures || []) {
          const conditions = mergeConditionalConditions(combination.conditions, failure.conditions || []);
          if (!conditions) continue;
          variantFailures.push({ ...failure, conditions });
        }
      } else {
        rows.push({ ...patch, conditions: combination.conditions });
      }
      outputVariants.set(name, rows);
    }
  }

  const outputs = [];
  if (![...outputVariants.values()].some((rows) => rows.length)) {
    if (variantFailures.length && variantFailures.every((row) => row.status === "invalid")) {
      return invalid("conditional_shape_all_variants_invalid", { variant_failures: variantFailures });
    }
    return {
      ...base,
      reason: "conditional_shape_no_complete_variant",
      conditionalVariantFailures: variantFailures,
    };
  }
  for (const name of (node.outputs || []).filter(Boolean)) {
    const variants = deduplicateConditionalVariants(outputVariants.get(name) || []);
    if (!variants.length) continue;
    const merged = unionOnnxTypeProtos(variants.map((variant) => onnxTypeProtoFromValue(variant)));
    if (merged.status !== "pass" || !merged.type) return invalid("conditional_shape_variant_type_conflict", { output_name: name });
    const patch = onnxValueDescriptorFromType(merged.type);
    patch.conditionalShapeVariants = variants;
    patch.conditionalShapeContract = {
      schema: "deepbom.onnx_conditional_shape_contract.v1",
      status: variantFailures.length ? "assessed_partial" : "assessed_complete",
      variant_count: variants.length,
      unassessed_variant_count: variantFailures.filter((row) => row.status !== "invalid").length,
      invalid_variant_count: variantFailures.filter((row) => row.status === "invalid").length,
      condition_keys: [...new Set([
        ...variants.flatMap((variant) => (variant.conditions || []).map((condition) => condition.key)),
        ...variantFailures.flatMap((failure) => (failure.conditions || []).map((condition) => condition.key)),
      ])].sort(),
      variant_failures: structuredClone(deduplicateConditionalFailures(variantFailures)),
    };
    outputs.push([name, patch]);
  }
  return outputs.length
    ? resolved(outputs, variantFailures.length ? "conditional_shape_contract_partially_propagated" : "conditional_shape_contract_propagated")
    : base;
}

function inferNode(node, tensors, tensorTypeName) {
  const op = node.opType;
  const input = (index) => {
    const tensor = tensors.get(node.inputs[index]);
    if (tensor?.valueKind && tensor.valueKind !== "tensor") return { ...tensor, dtype: "UNKNOWN", shape: null, shapeDeclared: false };
    return tensor && tensor.shapeDeclared !== true ? { ...tensor, shape: null } : tensor;
  };
  const input0 = input(0);
  const input1 = input(1);
  const dtype = knownDtype(input0) || knownDtype(input1) || "UNKNOWN";
  const outputs = [];
  const setDescriptor = (index, patch) => {
    const name = node.outputs[index];
    if (!name || !patch) return false;
    outputs.push([name, patch]);
    return true;
  };
  const set = (index, shape, outDtype = dtype, values = null) => {
    const dimensions = normalizeShapeDimensions(shape);
    if (!dimensions) return false;
    const typeProto = makeOnnxTensorTypeFromDimensions(outDtype, dimensions, true);
    const patch = { dtype: outDtype, shape: [...typeProto.shape], shapeDeclared: true, typeProto };
    if (values && Array.isArray(values.values)) Object.assign(patch, staticValuePatch(values.values, values.source || op));
    if (values && Array.isArray(values.dimensions)) Object.assign(patch, staticDimensionValuePatch(values.dimensions, values.source || op));
    if (values && Array.isArray(values.canonicalTexts)) Object.assign(patch, staticCanonicalTextPatch(values.canonicalTexts, values.source || op));
    return setDescriptor(index, patch);
  };

  if (SAME_SHAPE.has(op)) {
    if (!set(0, tensorShapeDimensions(input0), dtype, inheritedValues(input0, op))) return unresolved("input_shape_unknown");
    if (op === "Dropout" && node.outputs[1]) set(1, tensorShapeDimensions(input0), "BOOL");
    return resolved(outputs);
  }
  if (op === "CumSum" || op === "ScatterElements") {
    return set(0, tensorShapeDimensions(input0), dtype) ? resolved(outputs) : unresolved(`${op === "CumSum" ? "cumsum" : "scatter_elements"}_input_shape_unknown`);
  }
  if (BOOL_SAME_SHAPE.has(op)) return set(0, tensorShapeDimensions(input0), "BOOL") ? resolved(outputs) : unresolved("input_shape_unknown");
  if (BROADCAST_SAME_TYPE.has(op) || BROADCAST_BOOL.has(op)) {
    const inputs = node.inputs.filter(Boolean).map((_, index) => input(index));
    if (!inputs.length || inputs.some((tensor) => !tensorRankKnown(tensor))) return unresolved("broadcast_input_shape_unknown");
    const shape = broadcastManyDimensions(inputs.map(tensorShapeDimensions));
    if (!shape) return unresolved("broadcast_shapes_incompatible");
    const outDtype = BROADCAST_BOOL.has(op) ? "BOOL" : dtype;
    const propagated = BROADCAST_BOOL.has(op)
      ? comparisonElementwiseValues(op, inputs, shape)
      : integerElementwiseValues(op, inputs, shape);
    return set(0, shape, outDtype, propagated) ? resolved(outputs) : unresolved("broadcast_output_unresolved");
  }
  if (op === "Where") {
    const inputs = [input(0), input(1), input(2)];
    if (inputs.some((tensor) => !tensorRankKnown(tensor))) return unresolved("where_input_shape_unknown");
    const shape = broadcastManyDimensions(inputs.map(tensorShapeDimensions));
    const propagated = shape ? whereElementwiseValues(inputs, shape) : null;
    return shape && set(0, shape, knownDtype(inputs[1]) || knownDtype(inputs[2]), propagated) ? resolved(outputs) : unresolved("where_broadcast_incompatible");
  }
  if (op === "Cast" || op === "CastLike") {
    const outDtype = op === "Cast" ? tensorTypeName(attrInt(node, "to", 0)) : knownDtype(input1);
    const values = integerDtype(outDtype) ? inheritedValues(input0, op) : null;
    return set(0, tensorShapeDimensions(input0), outDtype, values) ? resolved(outputs) : unresolved("cast_shape_or_dtype_unknown");
  }
  if (op === "QuantizeLinear") {
    const zeroPoint = input(2);
    const declared = attrInt(node, "output_dtype", 0);
    const outDtype = knownDtype(zeroPoint) || (declared > 0 ? tensorTypeName(declared) : "UINT8");
    return set(0, tensorShapeDimensions(input0), outDtype) ? resolved(outputs) : unresolved("quantize_input_shape_unknown");
  }
  if (op === "DequantizeLinear") {
    const declared = attrInt(node, "output_dtype", 0);
    return set(0, tensorShapeDimensions(input0), declared > 0 ? tensorTypeName(declared) : "FLOAT32") ? resolved(outputs) : unresolved("dequantize_input_shape_unknown");
  }
  if (op === "DynamicQuantizeLinear") {
    if (!set(0, tensorShapeDimensions(input0), "UINT8")) return unresolved("dynamic_quantize_input_shape_unknown");
    set(1, [], "FLOAT32");
    set(2, [], "UINT8");
    return resolved(outputs);
  }
  if (["Conv", "ConvInteger", "QLinearConv"].includes(op)) {
    const weight = op === "QLinearConv" ? input(3) : input1;
    if (tensorRankKnown(input0) && tensorRankKnown(weight) && input0.shape.length !== weight.shape.length) {
      return invalid("conv_input_weight_rank_mismatch", {
        input_rank: input0.shape.length,
        weight_rank: weight.shape.length,
      });
    }
    const shape = inferConvShape(node, input0, weight);
    const outDtype = op === "ConvInteger" ? "INT32" : op === "QLinearConv" ? knownDtype(input(7)) || "UINT8" : dtype;
    return set(0, shape, outDtype) ? resolved(outputs, shape.some((dimension) => !dimensionKnown(dimension))
      ? "conv_runtime_dimensions_unbound_rank_and_output_channel_inferred" : "") : unresolved("conv_shape_contract_unresolved");
  }
  if (op === "DeformConv") {
    const rankConflict = tensorRankKnown(input0) && tensorRankKnown(input1) && input0.shape.length !== input1.shape.length;
    if (rankConflict) return invalid("deform_conv_input_weight_rank_mismatch", { input_rank: input0.shape.length, weight_rank: input1.shape.length });
    const shape = inferConvShape(node, input0, input1);
    const contract = validateDeformConvContract(node, [input0, input1, input(2), input(3), input(4)], shape);
    if (contract.status === "invalid") return contract;
    return set(0, shape, dtype) ? resolved(outputs, contract.reason) : unresolved("deform_conv_shape_contract_unresolved");
  }
  if (op === "ConvTranspose") {
    const shape = inferConvTransposeShape(node, input0, input1);
    return set(0, shape, dtype) ? resolved(outputs, shape.some((dimension) => !dimensionKnown(dimension))
      ? "conv_transpose_runtime_dimensions_unbound_rank_and_output_channel_inferred" : "") : unresolved("conv_transpose_shape_contract_unresolved");
  }
  if (["MaxPool", "AveragePool"].includes(op)) {
    const shape = inferPoolShape(node, input0);
    if (!set(0, shape)) return unresolved("pool_shape_contract_unresolved");
    if (op === "MaxPool" && node.outputs[1]) set(1, shape, "INT64");
    return resolved(outputs, shape.some((dimension) => !dimensionKnown(dimension))
      ? "pool_runtime_dimensions_unbound_rank_and_nonspatial_axes_inferred" : "");
  }
  if (["GlobalAveragePool", "GlobalMaxPool"].includes(op)) {
    if (!tensorRankKnown(input0) || input0.shape.length < 3) return unresolved("global_pool_input_rank_unknown");
    const dimensions = tensorShapeDimensions(input0);
    return set(0, [dimensions[0], dimensions[1], ...dimensions.slice(2).map(() => valueDimension(1))]) ? resolved(outputs) : unresolved("global_pool_output_unresolved");
  }
  if (op === "Flatten") {
    const shape = inferFlattenShape(node, input0);
    return set(0, shape, dtype, inheritedValues(input0, op)) ? resolved(outputs) : unresolved("flatten_input_shape_unknown");
  }
  if (op === "Gemm") {
    const shape = inferGemmShape(node, input0, input1);
    return set(0, shape) ? resolved(outputs) : unresolved("gemm_shape_contract_unresolved");
  }
  if (["MatMul", "MatMulInteger", "QLinearMatMul"].includes(op)) {
    const right = op === "QLinearMatMul" ? input(3) : input1;
    const shape = inferMatMulShape(input0, right);
    const outDtype = op === "MatMulInteger" ? "INT32" : op === "QLinearMatMul" ? knownDtype(input(7)) || "UINT8" : dtype;
    return set(0, shape, outDtype) ? resolved(outputs) : unresolved("matmul_shape_contract_unresolved");
  }
  if (op === "Einsum") return attachOutputs(inferEinsumShape(node, node.inputs.filter(Boolean).map((_, index) => input(index)), set), outputs);
  if (op === "Attention") return attachOutputs(inferAttentionShape(node, [input0, input1, input(2), input(3), input(4), input(5), input(6)], set), outputs);
  if (op === "LayerNormalization") {
    if (!tensorRankKnown(input0)) return unresolved("layer_normalization_input_rank_unknown");
    const dimensions = tensorShapeDimensions(input0);
    const axis = normalizeAxis(attrInt(node, "axis", -1), dimensions.length);
    if (axis == null) return unresolved("layer_normalization_axis_out_of_range");
    if (!set(0, dimensions, dtype)) return unresolved("layer_normalization_output_unresolved");
    const stashDtype = tensorTypeName(attrInt(node, "stash_type", 1));
    const reducedShape = dimensions.map((dimension, index) => index < axis ? dimension : valueDimension(1));
    if (node.outputs[1] && !set(1, reducedShape, stashDtype)) return unresolved("layer_normalization_mean_unresolved");
    if (node.outputs[2] && !set(2, reducedShape, stashDtype)) return unresolved("layer_normalization_inv_std_dev_unresolved");
    return resolved(outputs);
  }
  if (op === "RandomNormalLike" || op === "RandomUniformLike") {
    const declaredDtype = node.attributes?.has("dtype") ? tensorTypeName(attrInt(node, "dtype", 0)) : "";
    const outputDtype = declaredDtype || knownDtype(input0);
    return outputDtype && set(0, tensorShapeDimensions(input0), outputDtype)
      ? resolved(outputs) : unresolved("random_like_shape_or_dtype_unknown");
  }
  if (op === "NonZero") {
    if (!tensorRankKnown(input0)) return unresolved("nonzero_input_rank_unknown");
    const inputName = String(node.inputs?.[0] || "input");
    const nnz = symbolicDimension(`${RUNTIME_DIMENSION_PREFIX}nnz:${encodeURIComponent(inputName)}`);
    if (!set(0, [valueDimension(input0.shape.length), nnz], "INT64")) return unresolved("nonzero_output_unresolved");
    const inputDimensions = tensorShapeDimensions(input0);
    outputs[outputs.length - 1][1].runtimeDimensionBounds = [{
      axis: 1,
      symbol: dimensionParameter(nnz),
      lower_bound_decimal: "0",
      upper_bound_expression: inputDimensions?.every(dimensionKnown)
        ? `deepbom_expr:mul(${inputDimensions.map(dimensionKey).join(",")})`
        : null,
      basis: "ONNX NonZero output cardinality is bounded by the logical input element count",
    }];
    return resolved(outputs, "nonzero_runtime_nnz_symbolically_bounded");
  }
  if (op === "LSTM") return attachOutputs(inferRnnShape(node, input0, set), outputs);
  if (op === "Transpose") {
    if (!tensorRankKnown(input0)) return unresolved("transpose_input_shape_unknown");
    const dimensions = tensorShapeDimensions(input0);
    const rank = dimensions.length;
    const perm = attrInts(node, "perm").length ? attrInts(node, "perm") : [...Array(rank).keys()].reverse();
    if (!validPermutation(perm, rank)) return unresolved("transpose_perm_invalid");
    return set(0, perm.map((axis) => dimensions[axis])) ? resolved(outputs) : unresolved("transpose_output_unresolved");
  }
  if (op === "Concat") return attachOutputs(inferConcat(node, tensors, set), outputs);
  if (op === "Reshape") return attachOutputs(inferReshape(node, input0, input1, set), outputs);
  if (op === "Squeeze") return attachOutputs(inferSqueeze(node, input0, input1, set, setDescriptor), outputs);
  if (op === "Unsqueeze") return attachOutputs(inferUnsqueeze(node, input0, input1, set), outputs);
  if (op === "Shape") {
    if (!tensorRankKnown(input0)) return unresolved("shape_input_rank_or_dimensions_unknown");
    const dimensions = tensorShapeDimensions(input0);
    const rank = dimensions.length;
    const start = normalizeSliceBound(attrInt(node, "start", 0), rank, false);
    const end = normalizeSliceBound(attrInt(node, "end", rank), rank, true);
    if (start > end) return unresolved("shape_start_exceeds_end");
    const values = dimensions.slice(start, end);
    return set(0, [values.length], "INT64", { dimensions: values, source: "Shape" }) ? resolved(outputs) : unresolved("shape_output_unresolved");
  }
  if (op === "Size") {
    if (!tensorKnownShape(input0)) return unresolved("size_input_shape_unknown");
    const elements = elementCount(input0.shape);
    return elements != null && set(0, [], "INT64", { values: [elements], source: "Size" }) ? resolved(outputs) : unresolved("size_output_outside_exact_static_integer_range");
  }
  if (op === "Expand") {
    const target = exactDimensionValues(input1);
    if (!tensorRankKnown(input0) || !target || target.some((dimension) => !dimensionKnown(dimension))) return unresolved("expand_shape_input_not_static");
    const shape = broadcastManyDimensions([tensorShapeDimensions(input0), target]);
    return shape && set(0, shape)
      ? resolved(outputs) : invalid("expand_target_not_broadcast_compatible", {
        input_shape: canonicalShapeDimensions(tensorShapeDimensions(input0)),
        target_shape: canonicalShapeDimensions(target),
      });
  }
  if (op === "Gather") return attachOutputs(inferGather(node, input0, input1, set), outputs);
  if (op === "GatherElements") {
    if (!tensorKnownShape(input0) || !tensorKnownShape(input1) || input0.shape.length !== input1.shape.length) return unresolved("gather_elements_rank_unknown_or_mismatch");
    return set(0, input1.shape) ? resolved(outputs) : unresolved("gather_elements_output_unresolved");
  }
  if (op === "GatherND") return attachOutputs(inferGatherNd(node, input0, input1, set), outputs);
  if (op === "ScatterND") return set(0, tensorShapeDimensions(input0), dtype) ? resolved(outputs) : unresolved("scatter_nd_data_shape_unknown");
  if (REDUCE_OPS.has(op)) return attachOutputs(inferReduce(node, input0, input1, set, dtype), outputs);
  if (op === "ArgMax" || op === "ArgMin") return attachOutputs(inferArgReduce(node, input0, set), outputs);
  if (op === "Pad") return attachOutputs(inferPad(node, input0, input1, input(3), set), outputs);
  if (op === "Tile") return attachOutputs(inferTile(input0, input1, set), outputs);
  if (op === "Split") return attachOutputs(inferSplit(node, input0, input1, set), outputs);
  if (op === "Slice") return attachOutputs(inferSlice(node, input0, [input1, input(2), input(3), input(4)], set), outputs);
  if (op === "Resize") return attachOutputs(inferResize(input0, input(2), input(3), set), outputs);
  if (op === "TopK") return attachOutputs(inferTopK(node, input0, input1, set), outputs);
  if (op === "Constant") return attachOutputs(inferConstant(node, set, tensorTypeName), outputs);
  if (op === "ConstantOfShape") return attachOutputs(inferConstantOfShape(node, input0, set, tensorTypeName), outputs);
  if (op === "DepthToSpace") return attachOutputs(inferDepthToSpace(node, input0, set), outputs);
  if (op === "Range") return attachOutputs(inferRange(input0, input1, input(2), set), outputs);
  if (op === "STFT") return attachOutputs(inferStft(node, input0, input1, input(2), input(3), set), outputs);
  return unresolved("implemented_rule_dispatch_missing");
}

function inferConcat(node, tensors, set) {
  const inputs = node.inputs.filter(Boolean).map((name) => tensors.get(name));
  const inputDtypes = inputs.map(knownDtype).filter(Boolean);
  const distinctDtypes = [...new Set(inputDtypes)];
  if (inputDtypes.length === inputs.length && distinctDtypes.length > 1) {
    return invalid("concat_input_dtype_mismatch", { input_dtypes: inputDtypes });
  }
  if (!inputs.length || inputs.some((tensor) => !tensorRankKnown(tensor))) return unresolved("concat_input_shape_unknown");
  const dimensions = inputs.map(tensorShapeDimensions);
  const rank = dimensions[0].length;
  if (dimensions.some((shape) => shape.length !== rank)) return unresolved("concat_rank_mismatch");
  const axis = normalizeAxis(attrInt(node, "axis", 0), rank);
  if (axis == null) return unresolved("concat_axis_invalid");
  const shape = dimensions[0].map(cloneDimension);
  shape[axis] = valueDimension(0);
  for (const [inputIndex, tensor] of inputs.entries()) {
    for (let index = 0; index < rank; index += 1) {
      if (index !== axis && !dimensionsCompatible(dimensions[inputIndex][index], dimensions[0][index])) return unresolved("concat_non_axis_dimension_mismatch");
    }
    shape[axis] = addDimensions(shape[axis], dimensions[inputIndex][axis]);
  }
  const dimensionValues = inputs.map(exactDimensionValues);
  const values = rank === 1 && axis === 0 && dimensionValues.every(Boolean)
    ? { dimensions: dimensionValues.flat(), source: "Concat" } : null;
  return set(0, shape, knownDtype(inputs[0]), values) ? resolved([]) : unresolved("concat_output_unresolved");
}

function inferReshape(node, data, requestedTensor, set) {
  if (!tensorRankKnown(data)) return unresolved("reshape_data_shape_unknown");
  const dataDimensions = tensorShapeDimensions(data);
  const requested = exactDimensionValues(requestedTensor);
  if (!requested) return unresolved("reshape_shape_input_not_static_integer_data");
  const allowZero = attrInt(node, "allowzero", 0) === 1;
  const numericRequested = requested.map(dimensionValue);
  if (allowZero && numericRequested.includes(0) && numericRequested.includes(-1)) return unresolved("reshape_allowzero_one_cannot_mix_zero_and_minus_one");
  const target = [];
  let inferIndex = -1;
  for (let index = 0; index < requested.length; index += 1) {
    const dimension = requested[index];
    const value = dimensionValue(dimension);
    if (value === 0 && !allowZero) {
      if (index >= dataDimensions.length) return unresolved("reshape_zero_copy_axis_out_of_range");
      target.push(cloneDimension(dataDimensions[index]));
    } else if (value === -1) {
      if (inferIndex >= 0) return unresolved("reshape_multiple_minus_one_dimensions");
      inferIndex = index;
      target.push(unknownDimension());
    } else if (value != null && value >= 0) target.push(valueDimension(value));
    else if (dimension?.kind === "symbolic") target.push(cloneDimension(dimension));
    else return unresolved("reshape_negative_dimension_invalid");
  }
  if (knownShape(data.shape) && target.every((dimension, index) => index === inferIndex || dimensionValue(dimension) != null)) {
    const inputElements = exactElementCount(data.shape);
    const knownProduct = target.reduce((product, dimension, index) => index === inferIndex ? product : product * BigInt(dimensionValue(dimension)), 1n);
    if (inputElements == null) return unresolved("reshape_input_cardinality_invalid");
    if (inferIndex >= 0) {
      if (knownProduct === 0n || inputElements % knownProduct !== 0n) return unresolved("reshape_inferred_dimension_not_integral");
      const inferred = inputElements / knownProduct;
      if (inferred > BigInt(Number.MAX_SAFE_INTEGER)) return unresolved("reshape_inferred_dimension_outside_exact_static_integer_range");
      target[inferIndex] = valueDimension(Number(inferred));
    } else if (knownProduct !== inputElements) return unresolved("reshape_element_count_mismatch");
  }
  if (inferIndex >= 0 && !dimensionKnown(target[inferIndex])) {
    const inferred = inferSymbolicReshapeDimension(dataDimensions, target.filter((_, index) => index !== inferIndex));
    if (dimensionKnown(inferred)) target[inferIndex] = inferred;
  }
  const values = inheritedValues(data, "Reshape");
  return set(0, target, knownDtype(data), values) ? resolved([]) : unresolved("reshape_output_unresolved");
}

function inferSqueeze(node, data, axesTensor, set, setDescriptor) {
  if (!tensorRankKnown(data)) return unresolved("squeeze_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  let axes = exactIntegerValues(axesTensor);
  if (!axes) axes = attrInts(node, "axes");
  if (!axes?.length) {
    const conditionalAxes = dimensions.map((dimension, index) => dimensionValue(dimension) == null ? index : null).filter((axis) => axis != null);
    if (conditionalAxes.length) {
      if (2 ** conditionalAxes.length > MAX_CONDITIONAL_SHAPE_VARIANTS) return unresolved("squeeze_implicit_axes_conditional_variant_limit_exceeded");
      let variants = [{ remove: new Set(), conditions: [] }];
      for (const axis of conditionalAxes) {
        const key = implicitSqueezeConditionKey(node, dimensions[axis], axis);
        variants = variants.flatMap((variant) => [
          { remove: new Set([...variant.remove, axis]), conditions: [...variant.conditions, { key, value: "true" }] },
          { remove: new Set(variant.remove), conditions: [...variant.conditions, { key, value: "false" }] },
        ]);
      }
      const alwaysRemove = new Set(dimensions.map((dimension, index) => dimensionValue(dimension) === 1 ? index : null).filter((axis) => axis != null));
      const patches = variants.map((variant) => {
        const remove = new Set([...alwaysRemove, ...variant.remove]);
        const shape = dimensions.filter((_, index) => !remove.has(index));
        const typeProto = makeOnnxTensorTypeFromDimensions(knownDtype(data), shape, true);
        return {
          dtype: knownDtype(data),
          shape: [...typeProto.shape],
          shapeDeclared: true,
          typeProto,
          conditions: variant.conditions,
        };
      });
      const merged = unionOnnxTypeProtos(patches.map((patch) => patch.typeProto));
      if (merged.status !== "pass" || !merged.type) return unresolved("squeeze_implicit_axes_conditional_type_union_failed");
      const patch = onnxValueDescriptorFromType(merged.type);
      patch.conditionalShapeVariants = deduplicateConditionalVariants(patches);
      patch.conditionalShapeContract = {
        schema: "deepbom.onnx_conditional_shape_contract.v1",
        status: "assessed_complete",
        variant_count: patch.conditionalShapeVariants.length,
        unassessed_variant_count: 0,
        invalid_variant_count: 0,
        condition_keys: [...new Set(patches.flatMap((variant) => variant.conditions.map((condition) => condition.key)))].sort(),
        variant_failures: [],
      };
      return setDescriptor(0, patch)
        ? resolved([], "squeeze_implicit_axes_finite_conditional_contract")
        : unresolved("squeeze_output_unresolved");
    }
    axes = dimensions.map((dimension, index) => dimensionValue(dimension) === 1 ? index : null).filter((axis) => axis != null);
  }
  const normalized = normalizeAxes(axes, dimensions.length);
  if (!normalized) return invalid("squeeze_axes_invalid", { axes, input_rank: dimensions.length });
  const nonUnitAxis = normalized.find((axis) => {
    const value = dimensionValue(dimensions[axis]);
    return value != null && value !== 1;
  });
  if (nonUnitAxis != null) return invalid("squeeze_axis_dimension_not_one", {
    axis: nonUnitAxis,
    dimension: dimensionValue(dimensions[nonUnitAxis]),
  });
  const conditionalAxes = normalized.filter((axis) => dimensionValue(dimensions[axis]) == null);
  if (conditionalAxes.length) {
    if (2 ** conditionalAxes.length > MAX_CONDITIONAL_SHAPE_VARIANTS) return unresolved("squeeze_explicit_axes_conditional_variant_limit_exceeded");
    const conditionRows = conditionalAxes.map((axis) => ({ axis, key: implicitSqueezeConditionKey(node, dimensions[axis], axis) }));
    let assignments = [{ values: [], conditions: [] }];
    for (const row of conditionRows) {
      assignments = assignments.flatMap((assignment) => [true, false].map((value) => ({
        values: [...assignment.values, { axis: row.axis, value }],
        conditions: [...assignment.conditions, { key: row.key, value: String(value) }],
      })));
    }
    const remove = new Set(normalized);
    const shape = dimensions.filter((_, index) => !remove.has(index));
    const typeProto = makeOnnxTensorTypeFromDimensions(knownDtype(data), shape, true);
    const valid = assignments.filter((assignment) => assignment.values.every((row) => row.value));
    const failures = assignments.filter((assignment) => assignment.values.some((row) => !row.value)).map((assignment) => ({
      status: "invalid",
      reason: "squeeze_axis_dimension_not_one",
      conditions: assignment.conditions,
      details: { axes: assignment.values.filter((row) => !row.value).map((row) => row.axis) },
    }));
    const patch = onnxValueDescriptorFromType(typeProto);
    patch.conditionalShapeVariants = valid.map((assignment) => ({
      dtype: knownDtype(data),
      shape: [...typeProto.shape],
      shapeDeclared: true,
      typeProto,
      conditions: assignment.conditions,
    }));
    patch.conditionalShapeContract = {
      schema: "deepbom.onnx_conditional_shape_contract.v1",
      status: "assessed_partial",
      variant_count: patch.conditionalShapeVariants.length,
      unassessed_variant_count: 0,
      invalid_variant_count: failures.length,
      condition_keys: conditionRows.map((row) => row.key).sort(),
      variant_failures: failures,
    };
    return setDescriptor(0, patch)
      ? resolved([], "squeeze_explicit_axes_guarded_contract")
      : unresolved("squeeze_output_unresolved");
  }
  const remove = new Set(normalized);
  const shape = dimensions.filter((_, index) => !remove.has(index));
  return set(0, shape, knownDtype(data), inheritedValues(data, "Squeeze")) ? resolved([]) : unresolved("squeeze_output_unresolved");
}

function implicitSqueezeConditionKey(node, dimension, axis) {
  const symbol = dimensionParameter(dimension);
  return symbol
    ? `onnx:dimension:${encodeURIComponent(symbol)}:equals:1`
    : `onnx:tensor:${encodeURIComponent(String(node.inputs?.[0] || "input"))}:axis:${axis}:equals:1`;
}

function inferUnsqueeze(node, data, axesTensor, set) {
  if (!tensorRankKnown(data)) return unresolved("unsqueeze_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const axes = exactIntegerValues(axesTensor) || attrInts(node, "axes");
  if (!axes?.length) return unresolved("unsqueeze_axes_not_static");
  const outputRank = dimensions.length + axes.length;
  const normalized = normalizeAxes(axes, outputRank);
  if (!normalized) return unresolved("unsqueeze_axes_invalid");
  const inserted = new Set(normalized);
  const shape = [];
  let inputIndex = 0;
  for (let index = 0; index < outputRank; index += 1) shape.push(inserted.has(index) ? valueDimension(1) : dimensions[inputIndex++]);
  return set(0, shape, knownDtype(data), inheritedValues(data, "Unsqueeze")) ? resolved([]) : unresolved("unsqueeze_output_unresolved");
}

function inferGather(node, data, indices, set) {
  if (!tensorRankKnown(data) || !tensorRankKnown(indices)) return unresolved("gather_input_shape_unknown");
  const dataDimensions = tensorShapeDimensions(data);
  const indexDimensions = tensorShapeDimensions(indices);
  const axis = normalizeAxis(attrInt(node, "axis", 0), dataDimensions.length);
  if (axis == null) return invalid("gather_axis_out_of_range", {
    axis: attrInt(node, "axis", 0),
    data_rank: dataDimensions.length,
  });
  const shape = [...dataDimensions.slice(0, axis), ...indexDimensions, ...dataDimensions.slice(axis + 1)];
  let values = null;
  const source = exactDimensionValues(data);
  const indexValues = exactIntegerValues(indices);
  if (source && indexValues && dataDimensions.length === 1 && axis === 0) {
    const selected = [];
    for (let index of indexValues) {
      if (index < 0) index += source.length;
      if (index < 0 || index >= source.length) return invalid("gather_index_out_of_range", {
        index,
        axis_extent: source.length,
      });
      selected.push(source[index]);
    }
    values = { dimensions: selected, source: "Gather" };
  }
  return set(0, shape, knownDtype(data), values) ? resolved([]) : unresolved("gather_output_unresolved");
}

function inferGatherNd(node, data, indices, set) {
  if (!tensorRankKnown(data) || !tensorRankKnown(indices) || !indices.shape.length) return unresolved("gather_nd_input_rank_unknown");
  const dataDimensions = tensorShapeDimensions(data);
  const indexDimensions = tensorShapeDimensions(indices);
  const batchDims = attrInt(node, "batch_dims", 0);
  const k = dimensionValue(indexDimensions.at(-1));
  if (!Number.isSafeInteger(batchDims) || batchDims < 0 || batchDims >= Math.min(dataDimensions.length, indexDimensions.length)) {
    return invalid("gather_nd_batch_dims_invalid", { batch_dims: batchDims, data_rank: dataDimensions.length, indices_rank: indexDimensions.length });
  }
  if (k == null) return unresolved("gather_nd_index_tuple_length_runtime_unknown");
  if (k < 0 || k > dataDimensions.length - batchDims) {
    return invalid("gather_nd_index_tuple_length_invalid", { index_tuple_length: k, maximum: dataDimensions.length - batchDims });
  }
  for (let index = 0; index < batchDims; index += 1) {
    const left = dimensionValue(dataDimensions[index]);
    const right = dimensionValue(indexDimensions[index]);
    if (left != null && right != null && left !== right) {
      return invalid("gather_nd_batch_dimension_mismatch", { axis: index, data_dimension: left, indices_dimension: right });
    }
  }
  const shape = [...indexDimensions.slice(0, -1).map(cloneDimension), ...dataDimensions.slice(batchDims + k).map(cloneDimension)];
  return set(0, shape, knownDtype(data)) ? resolved([]) : unresolved("gather_nd_output_unresolved");
}

function inferReduce(node, data, axesTensor, set, dtype) {
  if (!tensorRankKnown(data)) return unresolved("reduce_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const axesInputProvided = Boolean(node.inputs?.[1]);
  const axesAttributeProvided = node.attributes?.has("axes") === true;
  let axes = exactIntegerValues(axesTensor);
  if (!axes) axes = attrInts(node, "axes");
  const noop = attrInt(node, "noop_with_empty_axes", 0) === 1;
  if (!axes?.length) {
    if (noop && (axesInputProvided || axesAttributeProvided)) return set(0, dimensions, dtype, inheritedValues(data, node.opType)) ? resolved([]) : unresolved("reduce_noop_output_unresolved");
    axes = [...Array(dimensions.length).keys()];
  }
  const normalized = normalizeAxes(axes, dimensions.length);
  if (!normalized) return unresolved("reduce_axes_invalid");
  const reduced = new Set(normalized);
  const keepDims = attrInt(node, "keepdims", 1) !== 0;
  const shape = keepDims ? dimensions.map((dim, index) => reduced.has(index) ? valueDimension(1) : dim) : dimensions.filter((_, index) => !reduced.has(index));
  return set(0, shape, dtype) ? resolved([]) : unresolved("reduce_output_unresolved");
}

function inferArgReduce(node, data, set) {
  if (!tensorRankKnown(data)) return unresolved("arg_reduce_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const axis = normalizeAxis(attrInt(node, "axis", 0), dimensions.length);
  if (axis == null) return unresolved("arg_reduce_axis_invalid");
  const keepDims = attrInt(node, "keepdims", 1) !== 0;
  const shape = keepDims ? dimensions.map((dim, index) => index === axis ? valueDimension(1) : dim) : dimensions.filter((_, index) => index !== axis);
  return set(0, shape, "INT64") ? resolved([]) : unresolved("arg_reduce_output_unresolved");
}

function inferPad(node, data, padsTensor, axesTensor, set) {
  if (!tensorRankKnown(data)) return unresolved("pad_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const padsInputPresent = Boolean(node.inputs?.[1]);
  const padsAttributePresent = node.attributes?.has("pads") === true;
  const pads = padsInputPresent ? exactIntegerValues(padsTensor)
    : padsAttributePresent ? attrInts(node, "pads") : null;
  const axesInputPresent = Boolean(node.inputs?.[3]);
  const axes = axesInputPresent ? exactIntegerValues(axesTensor) : [...Array(dimensions.length).keys()];
  const normalized = axes ? normalizeAxes(axes, dimensions.length) : null;

  const vectorLength = (tensor) => {
    const shape = tensorShapeDimensions(tensor);
    return shape?.length === 1 ? dimensionValue(shape[0]) : null;
  };
  const padCount = pads?.length ?? (padsInputPresent ? vectorLength(padsTensor) : null);
  const axisCount = normalized?.length ?? (axesInputPresent ? vectorLength(axesTensor) : dimensions.length);
  if (!padsInputPresent && !padsAttributePresent) return unresolved("pad_values_missing");
  if (padsInputPresent && tensorRankKnown(padsTensor) && tensorShapeDimensions(padsTensor).length !== 1) {
    return unresolved("pad_values_tensor_rank_invalid");
  }
  if (axesInputPresent && tensorRankKnown(axesTensor) && tensorShapeDimensions(axesTensor).length !== 1) {
    return unresolved("pad_axes_tensor_rank_invalid");
  }
  if (axes && !normalized) return unresolved("pad_axes_invalid");
  if (padCount != null && axisCount != null && padCount !== axisCount * 2) {
    return unresolved("pad_axes_or_cardinality_invalid");
  }

  if (!pads || !normalized) {
    const affected = normalized ? new Set(normalized) : null;
    const shape = dimensions.map((dimension, axis) => affected && !affected.has(axis)
      ? cloneDimension(dimension) : unknownDimension());
    if (!set(0, shape, knownDtype(data))) return unresolved("pad_partial_output_unresolved");
    return {
      outputs: [],
      reason: !pads ? "pad_values_runtime_bound_preserve_rank_only" : "pad_axes_runtime_bound_preserve_rank_only",
    };
  }
  const shape = dimensions.map(cloneDimension);
  for (let index = 0; index < normalized.length; index += 1) {
    const axis = normalized[index];
    shape[axis] = addDimensions(shape[axis], valueDimension(pads[index] + pads[index + normalized.length]));
    if (dimensionValue(shape[axis]) != null && dimensionValue(shape[axis]) < 0) return unresolved("pad_crop_exceeds_input_dimension");
  }
  return set(0, shape, knownDtype(data)) ? resolved([]) : unresolved("pad_output_unresolved");
}

function inferTile(data, repeatsTensor, set) {
  if (!tensorRankKnown(data)) return unresolved("tile_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const repeats = exactIntegerValues(repeatsTensor);
  if (!repeats || repeats.length !== dimensions.length || repeats.some((value) => value < 0)) return unresolved("tile_repeats_not_static_or_invalid");
  return set(0, dimensions.map((dim, index) => multiplyDimension(dim, repeats[index])), knownDtype(data)) ? resolved([]) : unresolved("tile_output_unresolved");
}

function inferSplit(node, data, splitTensor, set) {
  if (!tensorRankKnown(data) || !node.outputs.length) return unresolved("split_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const axis = normalizeAxis(attrInt(node, "axis", 0), dimensions.length);
  if (axis == null) return unresolved("split_axis_invalid");
  let splits = exactIntegerValues(splitTensor) || attrInts(node, "split");
  if (!splits?.length) {
    const count = attrInt(node, "num_outputs", node.outputs.filter(Boolean).length);
    const axisSize = dimensionValue(dimensions[axis]);
    if (count <= 0 || axisSize == null || axisSize % count !== 0) return unresolved("split_equal_partition_not_integral");
    splits = Array(count).fill(axisSize / count);
  }
  const axisSize = dimensionValue(dimensions[axis]);
  if (splits.length !== node.outputs.filter(Boolean).length || splits.some((value) => value < 0)
    || axisSize != null && splits.reduce((sum, value) => sum + value, 0) !== axisSize) return unresolved("split_sizes_do_not_conserve_axis");
  const sourceValues = dimensions.length === 1 && axis === 0 ? exactDimensionValues(data) : null;
  let offset = 0;
  for (let index = 0; index < node.outputs.length; index += 1) {
    if (!node.outputs[index]) continue;
    const shape = dimensions.map(cloneDimension);
    shape[axis] = valueDimension(splits[index]);
    const values = sourceValues ? { dimensions: sourceValues.slice(offset, offset + splits[index]), source: "Split" } : null;
    set(index, shape, knownDtype(data), values);
    offset += splits[index];
  }
  return resolved([]);
}

function sliceControl(node, tensor, inputIndex, attributeName, allowUnbounded = false) {
  const inputPresent = Boolean(node.inputs?.[inputIndex]);
  if (inputPresent) {
    const numericValues = exactNumericValues(tensor);
    const canonicalTextPresent = tensor?.staticValuesCanonicalTextComplete === true;
    const values = exactSliceIntegerValues(tensor, allowUnbounded);
    let dimensionValues = allowUnbounded ? exactDimensionValues(tensor) : null;
    const dimensions = tensorShapeDimensions(tensor);
    const dtype = knownDtype(tensor);
    const declaredLength = dimensions?.length === 1 ? dimensionValue(dimensions[0]) : null;
    const invalid = Boolean((numericValues || canonicalTextPresent) && !values)
      || Boolean(dtype && !/^(?:INT32|INT64)$/.test(dtype))
      || Boolean(dimensions && dimensions.length !== 1)
      || Boolean(values && declaredLength != null && values.length !== declaredLength);
    if (!invalid && !dimensionValues && !values && allowUnbounded && /^(?:INT32|INT64)$/.test(dtype)
      && Number.isSafeInteger(declaredLength) && declaredLength >= 0 && declaredLength <= 4096) {
      const tensorName = encodeURIComponent(String(node.inputs[inputIndex] || `${attributeName}_input`));
      dimensionValues = Array.from({ length: declaredLength }, (_, index) => (
        symbolicDimension(`${RUNTIME_DIMENSION_PREFIX}value:${tensorName}:${index}`)
      ));
    }
    const length = values?.length ?? dimensionValues?.length ?? declaredLength;
    return {
      present: true,
      values,
      dimensionValues,
      length: Number.isSafeInteger(length) && length >= 0 ? length : null,
      invalid,
    };
  }

  const attribute = node.attributes?.get(attributeName);
  if (!attribute) return { present: false, values: null, dimensionValues: null, length: null, invalid: false };
  const rawValues = Array.isArray(attribute.ints) ? attribute.ints : [];
  const values = rawValues.every(Number.isSafeInteger) ? [...rawValues] : null;
  return {
    present: true,
    values,
    dimensionValues: values?.map(valueDimension) || null,
    length: values?.length ?? null,
    invalid: !values,
  };
}

function inferSlice(node, data, controls, set) {
  if (!tensorRankKnown(data)) return unresolved("slice_input_shape_unknown");
  const dimensions = tensorShapeDimensions(data);
  const startsControl = sliceControl(node, controls[0], 1, "starts", true);
  const endsControl = sliceControl(node, controls[1], 2, "ends", true);
  const axesControl = sliceControl(node, controls[2], 3, "axes");
  const stepsControl = sliceControl(node, controls[3], 4, "steps");
  const controlsInvalid = [startsControl, endsControl, axesControl, stepsControl].some((control) => control.invalid);
  if (controlsInvalid || !startsControl.present || !endsControl.present) return invalid("slice_control_tensor_contract_invalid", {
    starts_present: startsControl.present,
    ends_present: endsControl.present,
    invalid_control_present: controlsInvalid,
  });

  const cardinalities = [startsControl.length, endsControl.length];
  if (axesControl.present) cardinalities.push(axesControl.length);
  if (stepsControl.present) cardinalities.push(stepsControl.length);
  const knownCardinalities = cardinalities.filter(Number.isSafeInteger);
  if (new Set(knownCardinalities).size > 1) return invalid("slice_control_cardinality_mismatch", { cardinalities: knownCardinalities });
  const controlCount = knownCardinalities[0] ?? null;
  if (controlCount != null && controlCount > dimensions.length) return invalid("slice_control_cardinality_exceeds_rank", {
    control_count: controlCount,
    input_rank: dimensions.length,
  });

  const starts = startsControl.values;
  const ends = endsControl.values;
  const startTerms = starts || startsControl.dimensionValues;
  const endTerms = ends || endsControl.dimensionValues;
  const axes = axesControl.present
    ? axesControl.values
    : controlCount == null ? null : [...Array(controlCount).keys()];
  const steps = stepsControl.present
    ? stepsControl.values
    : controlCount == null ? null : Array(controlCount).fill(1);
  const normalized = axes ? normalizeAxesInOrder(axes, dimensions.length) : null;
  if (axes && !normalized) return invalid("slice_axes_invalid", { axes, input_rank: dimensions.length });
  if (steps?.some((step) => step === 0)) return invalid("slice_step_zero_invalid", { steps });

  const symbolicBoundsKnown = startTerms?.every(sliceTermKnown) && endTerms?.every(sliceTermKnown);
  if (!symbolicBoundsKnown || !steps || !normalized) {
    const shape = normalized
      ? dimensions.map((dimension, axis) => normalized.includes(axis) ? unknownDimension() : cloneDimension(dimension))
      : dimensions.map(() => unknownDimension());
    if (!set(0, shape, knownDtype(data))) return unresolved("slice_partial_output_unresolved");
    return { outputs: [], reason: "slice_dynamic_controls_preserve_rank_only" };
  }

  if (startTerms.length !== endTerms.length || startTerms.length !== steps.length || normalized.length !== startTerms.length) {
    return invalid("slice_control_cardinality_mismatch", {
      starts: startTerms.length,
      ends: endTerms.length,
      steps: steps.length,
      axes: normalized.length,
    });
  }
  const shape = dimensions.map(cloneDimension);
  const normalizedRanges = [];
  for (let index = 0; index < startTerms.length; index += 1) {
    const axis = normalized[index];
    const inferred = inferSliceDimension(dimensions[axis], startTerms[index], endTerms[index], steps[index]);
    if (!inferred) return unresolved("slice_range_invalid_or_unsafe");
    shape[axis] = inferred.dimension;
    normalizedRanges.push(inferred.range ? { axis, ...inferred.range } : null);
  }
  let values = null;
  const source = exactDimensionValues(data);
  if (source && dimensions.length === 1 && normalizedRanges.length === 1 && normalizedRanges[0]?.axis === 0 && normalizedRanges[0].length != null) {
    const range = normalizedRanges[0];
    const selected = [];
    for (let index = range.start; range.step > 0 ? index < range.end : index > range.end; index += range.step) selected.push(source[index]);
    values = { dimensions: selected, source: "Slice" };
  }
  return set(0, shape, knownDtype(data), values) ? resolved([]) : unresolved("slice_output_unresolved");
}

function inferSliceDimension(inputDimension, start, end, step) {
  if (!dimensionKnown(inputDimension) || !sliceTermKnown(start) || !sliceTermKnown(end)
    || !Number.isSafeInteger(step) || step === 0) return null;
  const startValue = sliceTermValue(start);
  const endValue = sliceTermValue(end);
  const inputValue = dimensionValue(inputDimension);
  if (startValue != null && endValue != null && inputValue != null) {
    const range = normalizeSliceRange(inputValue, startValue, endValue, step);
    return range ? { dimension: valueDimension(range.length), range } : null;
  }
  if (startValue === 0 && endValue === Number.POSITIVE_INFINITY && step === 1) {
    return { dimension: cloneDimension(inputDimension), range: null };
  }
  return {
    dimension: symbolicDimension(`deepbom_expr:slice_len(${dimensionKey(inputDimension)},${sliceTermKey(start)},${sliceTermKey(end)},${step})`),
    range: null,
  };
}

function sliceTermKnown(term) {
  if (typeof term === "bigint") return true;
  return typeof term === "number"
    ? Number.isSafeInteger(term) || term === Number.NEGATIVE_INFINITY || term === Number.POSITIVE_INFINITY
    : dimensionKnown(term);
}

function sliceTermValue(term) {
  return typeof term === "number" || typeof term === "bigint" ? term : dimensionValue(term);
}

function sliceTermKey(term) {
  const value = sliceTermValue(term);
  if (value === Number.NEGATIVE_INFINITY) return "neg_inf";
  if (value === Number.POSITIVE_INFINITY) return "pos_inf";
  if (typeof value === "bigint") return `i64:${value}`;
  return value != null ? `v:${value}` : dimensionKey(term);
}

function inferResize(data, scalesTensor, sizesTensor, set) {
  if (!tensorRankKnown(data)) return unresolved("resize_input_rank_unknown");
  const inputDimensions = tensorShapeDimensions(data);
  const sizes = exactDimensionValues(sizesTensor);
  let shape = null;
  if (sizes?.length === inputDimensions.length && sizes.every((dimension) => dimensionKnown(dimension)
    && (dimensionValue(dimension) == null || dimensionValue(dimension) >= 0))) shape = sizes;
  if (!shape) {
    const scales = exactNumericValues(scalesTensor);
    const inputShape = inputDimensions.map(dimensionValue);
    if (scales?.length === inputDimensions.length && inputShape.every((dimension) => dimension != null)
      && scales.every((value) => Number.isFinite(value) && value > 0)) {
      shape = inputShape.map((dim, index) => Math.floor(dim * scales[index]));
    }
  }
  return shape && set(0, shape, knownDtype(data)) ? resolved([]) : unresolved("resize_sizes_or_scales_not_static");
}

function inferTopK(node, data, kTensor, set) {
  if (!tensorKnownShape(data)) return unresolved("topk_input_shape_unknown");
  const values = exactIntegerValues(kTensor);
  const k = values?.length === 1 ? values[0] : null;
  const axis = normalizeAxis(attrInt(node, "axis", -1), data.shape.length);
  if (!Number.isSafeInteger(k) || k < 0 || axis == null || k > data.shape[axis]) return unresolved("topk_k_or_axis_invalid");
  const shape = [...data.shape];
  shape[axis] = k;
  set(0, shape, knownDtype(data));
  set(1, shape, "INT64");
  return resolved([]);
}

function inferConstant(node, set, tensorTypeName) {
  const value = node.attributes.get("value")?.tensor || null;
  if (value) {
    const values = exactNumericValues(value);
    const canonicalTexts = value.staticValuesCanonicalTextComplete === true
      ? value.staticValuesCanonicalTexts : null;
    const propagated = values ? { values, source: "Constant" }
      : canonicalTexts ? { canonicalTexts, source: "Constant" } : null;
    return set(0, value.shape || [], value.dtype || "UNKNOWN", propagated) ? resolved([]) : unresolved("constant_tensor_contract_invalid");
  }
  for (const [name, dtype, field, list] of [
    ["value_int", "INT64", "i", false], ["value_ints", "INT64", "ints", true],
    ["value_float", "FLOAT32", "f", false], ["value_floats", "FLOAT32", "floats", true],
  ]) {
    const attribute = node.attributes.get(name);
    if (!attribute) continue;
    const values = list ? attribute[field] : [attribute[field]];
    const shape = list ? [values.length] : [];
    const propagated = values.every((item) => Number.isFinite(Number(item))) ? { values: values.map(Number), source: "Constant" } : null;
    return set(0, shape, dtype || tensorTypeName(0), propagated) ? resolved([]) : unresolved("constant_attribute_invalid");
  }
  return unresolved("constant_value_attribute_unsupported_or_missing");
}

function inferConstantOfShape(node, shapeTensor, set, tensorTypeName) {
  const shape = exactDimensionValues(shapeTensor);
  if (!shape || shape.some((dimension) => !dimensionKnown(dimension) || (dimensionValue(dimension) != null && dimensionValue(dimension) < 0))) return unresolved("constant_of_shape_input_not_static");
  const value = node.attributes.get("value")?.tensor || null;
  const dtype = value?.dtype || tensorTypeName(1);
  const numericShape = shape.map(dimensionValue);
  const elementTotal = numericShape.every((dimension) => dimension != null && dimension >= 0)
    ? elementCount(numericShape) : null;
  const fillValues = value ? exactNumericValues(value) : [0];
  const values = Number.isSafeInteger(elementTotal) && elementTotal <= 65536 && fillValues?.length === 1
    ? { values: Array(elementTotal).fill(fillValues[0]), source: "ConstantOfShape" } : null;
  return set(0, shape, dtype, values) ? resolved([]) : unresolved("constant_of_shape_output_unresolved");
}

function validateDeformConvContract(node, tensors, outputShape) {
  const [input, weight, offset, bias, mask] = tensors;
  const required = [input, weight, offset];
  const dtypes = [...required, bias, mask].filter(Boolean).map(knownDtype).filter(Boolean);
  if (dtypes.length && new Set(dtypes).size > 1) return invalid("deform_conv_input_dtype_mismatch", { input_dtypes: dtypes });
  if (!required.every(tensorRankKnown) || !Array.isArray(outputShape)) return { status: "partial", reason: "deform_conv_auxiliary_contract_runtime_unbound" };
  const rank = input.shape.length;
  if (rank < 3 || weight.shape.length !== rank || offset.shape.length !== rank) return invalid("deform_conv_required_rank_mismatch");
  const spatialRank = rank - 2;
  const group = attrInt(node, "group", 1);
  const offsetGroup = attrInt(node, "offset_group", 1);
  const kernel = weight.shape.slice(2);
  const kernelVolume = kernel.every(positive) ? kernel.reduce((product, value) => product * value, 1) : null;
  if (!positive(group) || !positive(offsetGroup) || !kernelVolume || !positive(weight.shape[0]) || !positive(weight.shape[1])) {
    return invalid("deform_conv_group_or_kernel_contract_invalid");
  }
  if (positive(input.shape[1]) && (input.shape[1] !== weight.shape[1] * group || input.shape[1] % offsetGroup !== 0)
    || weight.shape[0] % group !== 0) return invalid("deform_conv_channel_group_contract_invalid");
  const expectedOffset = [input.shape[0], offsetGroup * kernelVolume * spatialRank, ...outputShape.slice(2).map(dimensionValue)];
  if (knownShapeMismatch(offset.shape, expectedOffset)) return invalid("deform_conv_offset_shape_mismatch", { expected_shape: expectedOffset, observed_shape: offset.shape });
  if (bias && tensorRankKnown(bias) && (bias.shape.length !== 1 || positive(bias.shape[0]) && bias.shape[0] !== weight.shape[0])) {
    return invalid("deform_conv_bias_shape_mismatch");
  }
  if (mask && tensorRankKnown(mask)) {
    const expectedMask = [input.shape[0], offsetGroup * kernelVolume, ...outputShape.slice(2).map(dimensionValue)];
    if (mask.shape.length !== rank || knownShapeMismatch(mask.shape, expectedMask)) {
      return invalid("deform_conv_mask_shape_mismatch", { expected_shape: expectedMask, observed_shape: mask.shape });
    }
  }
  const auxiliaryComplete = [offset, ...(bias ? [bias] : []), ...(mask ? [mask] : [])].every((tensor) => tensorKnownShape(tensor));
  return { status: auxiliaryComplete ? "assessed" : "partial", reason: auxiliaryComplete ? "" : "deform_conv_auxiliary_dimensions_runtime_unbound" };
}

function inferEinsumShape(node, inputs, set) {
  if (!inputs.length || inputs.some((tensor) => !tensorRankKnown(tensor))) return unresolved("einsum_input_rank_unknown");
  const parsed = parseOnnxEinsumEquation(attrString(node, "equation", ""), inputs.map((tensor) => tensor.shape.length));
  if (parsed.status !== "assessed") return invalid(parsed.reason);
  const dtypes = inputs.map(knownDtype);
  if (dtypes.some((dtype) => !dtype)) return unresolved("einsum_input_dtype_unknown");
  if (new Set(dtypes).size !== 1) return invalid("einsum_input_dtype_mismatch", { input_dtypes: dtypes });
  const dimensionsByLabel = new Map();
  for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
    const dimensions = tensorShapeDimensions(inputs[inputIndex]);
    for (let axis = 0; axis < parsed.operands[inputIndex].length; axis += 1) {
      const label = parsed.operands[inputIndex][axis];
      const rows = dimensionsByLabel.get(label) || [];
      rows.push(dimensions[axis]);
      dimensionsByLabel.set(label, rows);
    }
  }
  const merged = new Map();
  let runtimeEquality = false;
  for (const [label, dimensions] of dimensionsByLabel) {
    if (label.startsWith("@ellipsis:")) {
      const broadcast = broadcastManyDimensions(dimensions.map((dimension) => [dimension]));
      if (!broadcast) return invalid("einsum_ellipsis_dimensions_not_broadcast_compatible", { label });
      merged.set(label, broadcast[0]);
      continue;
    }
    let current = cloneDimension(dimensions[0]);
    for (const dimension of dimensions.slice(1)) {
      const left = dimensionValue(current), right = dimensionValue(dimension);
      if (left != null && right != null && left !== right) return invalid("einsum_label_dimension_mismatch", { label, left, right });
      if (dimensionKey(current) === dimensionKey(dimension)) continue;
      if (!dimensionKnown(current)) current = cloneDimension(dimension);
      else if (dimensionKnown(dimension)) {
        current = unknownDimension();
        runtimeEquality = true;
      }
    }
    merged.set(label, current);
  }
  const output = parsed.output.map((label) => cloneDimension(merged.get(label) || unknownDimension()));
  return set(0, output, dtypes[0]) ? resolved([], runtimeEquality ? "einsum_symbolic_label_equality_runtime_guard" : "") : unresolved("einsum_output_unresolved");
}

function inferAttentionShape(node, tensors, set) {
  const [query, key, value, attentionMask, pastKey, pastValue, nonpadLength] = tensors;
  if (![query, key, value].every(tensorRankKnown)) return unresolved("attention_qkv_rank_unknown");
  const ranks = [query.shape.length, key.shape.length, value.shape.length];
  if (new Set(ranks).size !== 1 || ![3, 4].includes(ranks[0])) return invalid("attention_qkv_rank_mismatch", { ranks });
  const dtypes = [knownDtype(query), knownDtype(key), knownDtype(value)];
  if (dtypes.some((dtype) => !dtype)) return unresolved("attention_qkv_dtype_unknown");
  if (dtypes[0] !== dtypes[1]) return invalid("attention_query_key_dtype_mismatch");
  const q = tensorShapeDimensions(query), k = tensorShapeDimensions(key), v = tensorShapeDimensions(value);
  if (knownDimensionMismatch(q[0], k[0]) || knownDimensionMismatch(q[0], v[0])) return invalid("attention_batch_dimension_mismatch");
  const pastPair = Boolean(pastKey) || Boolean(pastValue);
  if (pastPair && !(pastKey && pastValue)) return invalid("attention_past_key_value_pair_incomplete");
  if (nonpadLength && pastPair) return invalid("attention_nonpad_length_with_past_cache");
  let qHeads, kvHeads, qSequence, kvSequence, qkHeadSize, valueHeadSize, outputShape;
  if (ranks[0] === 4) {
    [qHeads, qSequence, qkHeadSize] = [q[1], q[2], q[3]];
    [kvHeads, kvSequence] = [k[1], k[2]];
    valueHeadSize = v[3];
    if (knownDimensionMismatch(k[2], v[2]) || knownDimensionMismatch(k[1], v[1]) || knownDimensionMismatch(qkHeadSize, k[3])) return invalid("attention_4d_head_or_sequence_contract_mismatch");
    const qHeadCount = dimensionValue(qHeads), kvHeadCount = dimensionValue(kvHeads);
    if (qHeadCount != null && kvHeadCount != null
      && (qHeadCount < kvHeadCount || qHeadCount % kvHeadCount !== 0)) return invalid("attention_4d_head_count_contract_invalid");
    outputShape = [q[0], qHeads, qSequence, valueHeadSize];
  } else {
    const qHeadCount = attrInt(node, "q_num_heads", 0), kvHeadCount = attrInt(node, "kv_num_heads", 0);
    if (!positive(qHeadCount) || !positive(kvHeadCount) || qHeadCount < kvHeadCount || qHeadCount % kvHeadCount !== 0) return invalid("attention_3d_head_attributes_invalid");
    const qHidden = dimensionValue(q[2]), kHidden = dimensionValue(k[2]), vHidden = dimensionValue(v[2]);
    if (qHidden != null && qHidden % qHeadCount !== 0 || kHidden != null && kHidden % kvHeadCount !== 0 || vHidden != null && vHidden % kvHeadCount !== 0) {
      return invalid("attention_hidden_size_not_divisible_by_head_count");
    }
    qHeads = valueDimension(qHeadCount); kvHeads = valueDimension(kvHeadCount); qSequence = q[1]; kvSequence = k[1];
    qkHeadSize = qHidden == null ? unknownDimension() : valueDimension(qHidden / qHeadCount);
    const keyHeadSize = kHidden == null ? unknownDimension() : valueDimension(kHidden / kvHeadCount);
    if (knownDimensionMismatch(qkHeadSize, keyHeadSize) || knownDimensionMismatch(k[1], v[1])) return invalid("attention_3d_head_or_sequence_contract_mismatch");
    valueHeadSize = vHidden == null ? unknownDimension() : valueDimension(vHidden / kvHeadCount);
    outputShape = [q[0], qSequence, multiplyDimension(valueHeadSize, qHeadCount)];
  }
  let pastSequence = valueDimension(0);
  const presentPair = Boolean(node.outputs[1]) || Boolean(node.outputs[2]);
  if (presentPair && !(node.outputs[1] && node.outputs[2])) return invalid("attention_present_key_value_pair_incomplete");
  if (pastPair !== presentPair) return invalid("attention_past_present_cache_contract_incomplete");
  if (nonpadLength && presentPair) return invalid("attention_nonpad_length_with_present_cache");
  if (pastPair) {
    if (![pastKey, pastValue].every(tensorRankKnown) || pastKey.shape.length !== 4 || pastValue.shape.length !== 4) return invalid("attention_past_cache_rank_mismatch");
    const pk = tensorShapeDimensions(pastKey), pv = tensorShapeDimensions(pastValue);
    if (knownDtype(pastKey) && knownDtype(pastKey) !== dtypes[0]) return invalid("attention_past_key_dtype_mismatch");
    if (knownDtype(pastValue) && knownDtype(pastValue) !== dtypes[2]) return invalid("attention_past_value_dtype_mismatch");
    if (knownDimensionMismatch(pk[0], q[0]) || knownDimensionMismatch(pk[1], kvHeads) || knownDimensionMismatch(pk[2], pv[2])
      || knownDimensionMismatch(pk[3], qkHeadSize) || knownDimensionMismatch(pv[0], q[0]) || knownDimensionMismatch(pv[1], kvHeads)
      || knownDimensionMismatch(pv[3], valueHeadSize)) return invalid("attention_past_cache_contract_mismatch");
    pastSequence = pk[2];
  }
  const totalSequence = addDimensions(pastSequence, kvSequence);
  let runtimeGuard = [qkHeadSize, valueHeadSize].some((dimension) => !dimensionKnown(dimension));
  if (attentionMask) {
    if (!tensorRankKnown(attentionMask)) return unresolved("attention_mask_rank_unknown");
    const mask = tensorShapeDimensions(attentionMask);
    if (mask.length > 4) return invalid("attention_mask_rank_invalid");
    const target = [q[0], qHeads, qSequence, totalSequence];
    const padded = [...Array(4 - mask.length).fill(valueDimension(1)), ...mask];
    for (let axis = 0; axis < 4; axis += 1) {
      const observed = dimensionValue(padded[axis]), expected = dimensionValue(target[axis]);
      if (axis === 3) {
        if (observed != null && expected != null && observed !== 1 && observed > expected) return invalid("attention_mask_last_dimension_exceeds_total_sequence");
      } else if (observed != null && expected != null && observed !== 1 && observed !== expected) {
        return invalid("attention_mask_not_broadcastable", { axis, observed, expected });
      }
      if (observed == null || expected == null) runtimeGuard = true;
    }
  }
  if (!set(0, outputShape, dtypes[0])) return unresolved("attention_output_unresolved");
  if (node.outputs[1] && !set(1, [q[0], kvHeads, totalSequence, qkHeadSize], dtypes[0])) return unresolved("attention_present_key_unresolved");
  if (node.outputs[2] && !set(2, [q[0], kvHeads, totalSequence, valueHeadSize], dtypes[2])) return unresolved("attention_present_value_unresolved");
  if (node.outputs[3] && !set(3, [q[0], qHeads, qSequence, totalSequence], dtypes[0])) return unresolved("attention_qk_output_unresolved");
  return resolved([], runtimeGuard ? "attention_symbolic_dimension_runtime_guard" : "");
}

function knownDimensionMismatch(left, right) {
  const a = dimensionValue(left), b = dimensionValue(right);
  return a != null && b != null && a !== b;
}

function knownShapeMismatch(observed, expected) {
  return observed.length !== expected.length || observed.some((value, index) => (
    knownDimension(value) && knownDimension(expected[index]) && Number(value) !== Number(expected[index])
  ));
}

function inferConvShape(node, input, weight) {
  const w = weight?.shape || [];
  if (!tensorRankKnown(weight) || w.length < 3) return null;
  const x = tensorRankKnown(input)
    ? tensorShapeDimensions(input)
    : w.map(() => unknownDimension());
  if (x.length !== w.length) return null;
  const spatialRank = x.length - 2;
  const kernel = attrInts(node, "kernel_shape");
  const kernels = w.slice(2).map((dim, index) => positive(dim) || positive(kernel[index]));
  const outChannels = positive(w[0]);
  const group = Math.max(1, attrInt(node, "group", 1));
  const inputChannels = dimensionValue(x[1]);
  if (!positive(w[1]) || !outChannels || kernels.some((dim) => !dim)
    || inputChannels != null && inputChannels !== w[1] * group) return null;
  const spatial = inferSpatialDimensions(node, x.slice(2), kernels);
  return spatial?.length === spatialRank ? [x[0], valueDimension(outChannels), ...spatial] : null;
}

function inferConvTransposeShape(node, input, weight) {
  const w = weight?.shape || [];
  if (!tensorRankKnown(weight) || w.length < 3) return null;
  const x = tensorRankKnown(input)
    ? tensorShapeDimensions(input)
    : w.map(() => unknownDimension());
  if (x.length !== w.length) return null;
  const spatialRank = x.length - 2;
  const group = positive(attrInt(node, "group", 1));
  const inputChannels = dimensionValue(x[1]);
  if (!group || !positive(w[0]) || inputChannels != null && inputChannels !== w[0] || !positive(w[1])) return null;
  const outputShape = attrInts(node, "output_shape");
  if (node.attributes?.has("output_shape")
    && (outputShape.length !== spatialRank || outputShape.some((dim) => !knownDimension(dim)))) return null;
  const kernels = w.slice(2);
  if (kernels.some((dim) => !positive(dim))) return null;
  const kernelShape = strictSpatialAttribute(node, "kernel_shape", spatialRank, 0, false, false);
  if (node.attributes?.has("kernel_shape") && !kernelShape) return null;
  if (kernelShape && kernelShape.some((value, axis) => value !== kernels[axis])) return null;
  const strides = strictSpatialAttribute(node, "strides", spatialRank, 1, false, true);
  const dilations = strictSpatialAttribute(node, "dilations", spatialRank, 1, false, true);
  const outputPadding = strictSpatialAttribute(node, "output_padding", spatialRank, 0, true, true);
  if (!strides || !dilations || !outputPadding) return null;
  if (outputPadding.some((value, axis) => value >= strides[axis] && value >= dilations[axis])) return null;
  let spatial = outputShape.map(valueDimension);
  if (!spatial.length) {
    const autoPad = attrString(node, "auto_pad", "NOTSET");
    if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
      spatial = x.slice(2).map((dimension, axis) => multiplyDimension(dimension, strides[axis]));
    }
    else {
      if (autoPad !== "NOTSET" && autoPad !== "VALID") return null;
      const normalizedPads = strictSpatialAttribute(node, "pads", spatialRank * 2, 0, true, true);
      if (!normalizedPads) return null;
      spatial = x.slice(2).map((dimension, axis) => {
        const offset = outputPadding[axis] + (kernels[axis] - 1) * dilations[axis] + 1
          - normalizedPads[axis] - normalizedPads[axis + spatialRank] - strides[axis];
        return addDimensionOffset(multiplyDimension(dimension, strides[axis]), offset);
      });
    }
  }
  return spatial.every((dimension) => dimensionValue(dimension) == null || dimensionValue(dimension) >= 0)
    ? [cloneDimension(x[0]), valueDimension(w[1] * group), ...spatial] : null;
}

function strictSpatialAttribute(node, name, length, fallback, allowZero, optional) {
  if (!node.attributes?.has(name)) return optional ? Array(length).fill(fallback) : null;
  const values = attrInts(node, name);
  if (values.length !== length || values.some((value) => allowZero ? value < 0 : value <= 0)) return null;
  return values;
}

function inferDepthToSpace(node, data, set) {
  if (!tensorRankKnown(data) || data.shape.length !== 4) return unresolved("depth_to_space_input_rank_not_four");
  const dimensions = tensorShapeDimensions(data);
  const block = attrInt(node, "blocksize", 0);
  const channels = dimensionValue(dimensions[1]);
  if (!positive(block) || channels == null || channels % (block * block) !== 0) return unresolved("depth_to_space_block_or_channel_contract_invalid");
  return set(0, [dimensions[0], valueDimension(channels / (block * block)), multiplyDimension(dimensions[2], block), multiplyDimension(dimensions[3], block)], knownDtype(data))
    ? resolved([]) : unresolved("depth_to_space_output_unresolved");
}

function inferRange(startTensor, limitTensor, deltaTensor, set) {
  const start = exactNumericValues(startTensor)?.[0];
  const limit = exactNumericValues(limitTensor)?.[0];
  const delta = exactNumericValues(deltaTensor)?.[0];
  if ([start, limit, delta].every(Number.isFinite)) {
    if (delta === 0) return unresolved("range_delta_zero_invalid");
    const length = Math.max(0, Math.ceil((limit - start) / delta));
    if (!Number.isSafeInteger(length)) return unresolved("range_output_cardinality_unsafe");
    let values = null;
    if (length <= 65536) {
      const generated = Array.from({ length }, (_, index) => start + index * delta);
      if (generated.every(Number.isFinite)) values = { values: generated, source: "Range" };
    }
    return set(0, [length], knownDtype(startTensor), values) ? resolved([]) : unresolved("range_output_unresolved");
  }

  const terms = [startTensor, limitTensor, deltaTensor].map(rangeControlTerm);
  if (terms.some((term) => !rangeTermKnown(term))) return unresolved("range_controls_not_static_finite_scalars");
  const deltaValue = rangeTermValue(terms[2]);
  if (!Number.isFinite(deltaValue)) return unresolved("range_delta_not_artifact_bound");
  if (deltaValue === 0) return unresolved("range_delta_zero_invalid");
  const expression = symbolicDimension(`deepbom_expr:range_len(${terms.map(rangeTermKey).join(",")})`);
  return set(0, [expression], knownDtype(startTensor)) ? resolved([]) : unresolved("range_output_unresolved");
}

function rangeControlTerm(tensor) {
  const numeric = exactNumericValues(tensor);
  if (numeric?.length === 1) return numeric[0];
  const dimensions = exactDimensionValues(tensor);
  return dimensions?.length === 1 ? dimensions[0] : null;
}

function rangeTermKnown(term) {
  return typeof term === "number" ? Number.isFinite(term) : Boolean(term)
    && (dimensionValue(term) != null || term.kind === "symbolic" && Boolean(term.parameter));
}

function rangeTermValue(term) {
  return typeof term === "number" ? term : dimensionValue(term);
}

function rangeTermKey(term) {
  const value = rangeTermValue(term);
  return value != null ? `v:${value}` : dimensionKey(term);
}

function inferStft(node, signal, frameStepTensor, window, frameLengthTensor, set) {
  if (!tensorRankKnown(signal) || signal.shape.length < 2 || signal.shape.length > 3) return unresolved("stft_signal_shape_unknown_or_invalid");
  const signalDimensions = tensorShapeDimensions(signal);
  const frameStep = exactIntegerValues(frameStepTensor)?.[0];
  const declaredFrameLength = exactIntegerValues(frameLengthTensor)?.[0];
  const windowDimensions = tensorShapeDimensions(window);
  const windowLength = windowDimensions?.length === 1 ? dimensionValue(windowDimensions[0]) : null;
  const frameLength = declaredFrameLength || windowLength;
  const signalLength = signalDimensions[1];
  const concreteSignalLength = dimensionValue(signalLength);
  if (!positive(frameStep) || !positive(frameLength) || !dimensionKnown(signalLength)
    || concreteSignalLength != null && concreteSignalLength < frameLength) return unresolved("stft_frame_contract_not_static_or_invalid");
  const complexDimension = signal.shape.length === 3 ? dimensionValue(signalDimensions[2]) : null;
  if (signal.shape.length === 3 && ![1, 2].includes(complexDimension)) return unresolved("stft_signal_complex_dimension_invalid");
  const onesided = attrInt(node, "onesided", 1);
  if (signal.shape.length === 3 && complexDimension === 2 && onesided !== 0) return unresolved("stft_onesided_complex_input_invalid");
  const frameCount = concreteSignalLength == null
    ? symbolicDimension(`deepbom_expr:add(floor_div(deepbom_expr:add(${dimensionKey(signalLength)},-${frameLength}),${frameStep}),1)`)
    : valueDimension(1 + Math.floor((concreteSignalLength - frameLength) / frameStep));
  const bins = onesided === 0 ? frameLength : Math.floor(frameLength / 2) + 1;
  return set(0, [signalDimensions[0], frameCount, valueDimension(bins), valueDimension(2)], knownDtype(signal)) ? resolved([]) : unresolved("stft_output_unresolved");
}

function inferPoolShape(node, input) {
  const kernel = attrInts(node, "kernel_shape");
  if (!kernel.length || kernel.some((dim) => !positive(dim))) return null;
  const dimensions = tensorRankKnown(input)
    ? tensorShapeDimensions(input)
    : Array.from({ length: kernel.length + 2 }, () => unknownDimension());
  if (dimensions.length !== kernel.length + 2) return null;
  const spatial = inferSpatialDimensions(node, dimensions.slice(2), kernel);
  return spatial ? [dimensions[0], dimensions[1], ...spatial] : null;
}

function inferSpatialDimensions(node, inputs, kernels) {
  if (inputs.every((dimension) => dimensionValue(dimension) != null)) {
    const exact = inferSpatial(node, inputs.map(dimensionValue), kernels);
    return exact?.map(valueDimension) || null;
  }
  const rank = inputs.length;
  const strides = padTo(attrInts(node, "strides"), rank, 1);
  const dilations = padTo(attrInts(node, "dilations"), rank, 1);
  let pads = padTo(attrInts(node, "pads"), rank * 2, 0, true);
  const autoPad = attrString(node, "auto_pad", "NOTSET");
  if (!["NOTSET", "VALID", "", "SAME_UPPER", "SAME_LOWER"].includes(autoPad)) return null;
  if (autoPad === "VALID") pads = Array(rank * 2).fill(0);
  const ceilMode = attrInt(node, "ceil_mode", 0) === 1;
  return inputs.map((dimension, index) => {
    if (!dimensionKnown(dimension)) return unknownDimension();
    const input = dimensionKey(dimension);
    if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
      return strides[index] === 1 ? cloneDimension(dimension)
        : symbolicDimension(`deepbom_expr:ceil_div(${input},${strides[index]})`);
    }
    const adjustment = pads[index] + pads[index + rank] - dilations[index] * (kernels[index] - 1) - 1;
    if (strides[index] === 1) return addDimensionOffset(dimension, adjustment + 1);
    const rounding = ceilMode ? "ceil_div" : "floor_div";
    return symbolicDimension(`deepbom_expr:add(${rounding}(deepbom_expr:add(${input},${adjustment}),${strides[index]}),1)`);
  });
}

function inferSpatial(node, inputs, kernels) {
  const rank = inputs.length;
  const strides = padTo(attrInts(node, "strides"), rank, 1);
  const dilations = padTo(attrInts(node, "dilations"), rank, 1);
  let pads = padTo(attrInts(node, "pads"), rank * 2, 0, true);
  const autoPad = attrString(node, "auto_pad", "NOTSET");
  if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") return inputs.map((dim, index) => Math.ceil(dim / strides[index]));
  if (!["NOTSET", "VALID", ""].includes(autoPad)) return null;
  if (autoPad === "VALID") pads = Array(rank * 2).fill(0);
  const ceilMode = attrInt(node, "ceil_mode", 0) === 1;
  const outputs = inputs.map((input, index) => {
    const raw = (input + pads[index] + pads[index + rank] - dilations[index] * (kernels[index] - 1) - 1) / strides[index] + 1;
    let output = ceilMode ? Math.ceil(raw) : Math.floor(raw);
    if (ceilMode && (output - 1) * strides[index] >= input + pads[index]) output -= 1;
    return output;
  });
  return outputs.every((dim) => dim >= 0) ? outputs : null;
}

function inferFlattenShape(node, input) {
  if (!tensorRankKnown(input)) return null;
  const dimensions = tensorShapeDimensions(input);
  let axis = attrInt(node, "axis", 1);
  if (axis < 0) axis += dimensions.length;
  if (axis < 0 || axis > dimensions.length) return null;
  return [productDimensions(dimensions.slice(0, axis)), productDimensions(dimensions.slice(axis))];
}

function inferGemmShape(node, input, weight) {
  const a = tensorShapeDimensions(input) || [];
  const b = tensorShapeDimensions(weight) || [];
  if (!tensorRankKnown(input) || !tensorRankKnown(weight) || a.length !== 2 || b.length !== 2) return null;
  const transA = attrInt(node, "transA", 0) === 1;
  const transB = attrInt(node, "transB", 0) === 1;
  const m = transA ? a[1] : a[0];
  const kA = transA ? a[0] : a[1];
  const kB = transB ? b[1] : b[0];
  const n = transB ? b[0] : b[1];
  return contractionDimensionsCanMatch(kA, kB) ? [m, n] : null;
}

function inferMatMulShape(input, weight) {
  const a = tensorShapeDimensions(input) || [];
  const b = tensorShapeDimensions(weight) || [];
  if (!tensorRankKnown(input) || !tensorRankKnown(weight) || !a.length || !b.length) return null;
  const aVector = a.length === 1;
  const bVector = b.length === 1;
  const promotedA = aVector ? [valueDimension(1), a[0]] : a;
  const promotedB = bVector ? [b[0], valueDimension(1)] : b;
  if (!contractionDimensionsCanMatch(promotedA.at(-1), promotedB.at(-2))) return null;
  const batch = broadcastManyDimensions([promotedA.slice(0, -2), promotedB.slice(0, -2)]);
  if (!batch) return null;
  const shape = [...batch.map(cloneDimension), cloneDimension(promotedA.at(-2)), cloneDimension(promotedB.at(-1))];
  if (aVector) shape.splice(shape.length - 2, 1);
  if (bVector) shape.pop();
  return shape;
}

function inferRnnShape(node, input, set) {
  if (!tensorRankKnown(input) || input.shape.length !== 3) return unresolved("lstm_input_rank_unknown_or_invalid");
  const dimensions = tensorShapeDimensions(input);
  const layout = attrInt(node, "layout", 0);
  if (![0, 1].includes(layout)) return unresolved("lstm_layout_invalid");
  const direction = attrString(node, "direction", "forward");
  const directionCount = direction === "bidirectional" ? valueDimension(2)
    : ["forward", "reverse"].includes(direction) ? valueDimension(1) : unknownDimension();
  const hiddenSizeValue = attrInt(node, "hidden_size", -1);
  const hiddenSize = hiddenSizeValue > 0 ? valueDimension(hiddenSizeValue) : unknownDimension();
  const sequenceLength = cloneDimension(dimensions[layout === 0 ? 0 : 1]);
  const batchSize = cloneDimension(dimensions[layout === 0 ? 1 : 0]);
  const dtype = knownDtype(input);
  if (!dtype) return unresolved("lstm_input_dtype_unknown");
  const outputShape = layout === 0
    ? [sequenceLength, directionCount, batchSize, hiddenSize]
    : [batchSize, sequenceLength, directionCount, hiddenSize];
  const stateShape = layout === 0
    ? [directionCount, batchSize, hiddenSize]
    : [batchSize, directionCount, hiddenSize];
  if (node.outputs?.[0]) set(0, outputShape, dtype);
  if (node.outputs?.[1]) set(1, stateShape, dtype);
  if (node.outputs?.[2]) set(2, stateShape, dtype);
  return node.outputs?.some(Boolean) ? resolved([]) : unresolved("lstm_output_omitted");
}

export function mergeOnnxInferredTensor(tensorMap, name, patch, nodeIndex, opName) {
  const existing = tensorMap.get(name) || { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  const existingKind = normalizedValueKind(existing, false);
  const inferredKind = normalizedValueKind(patch, true);
  const denseInference = inferredKind === "tensor" && (existingKind === "tensor" || existingKind === "unresolved");
  if (!denseInference) {
    if (existingKind !== "unresolved" && inferredKind !== "unresolved" && existingKind !== inferredKind) {
      return conflict(nodeIndex, opName, name, "value_kind", existingKind, inferredKind);
    }
    const declaredType = existingKind === "unresolved" ? null : onnxTypeProtoFromValue(existing);
    const inferredType = inferredKind === "unresolved" ? null : onnxTypeProtoFromValue(patch);
    const merged = unifyOnnxTypeProtos(declaredType, inferredType);
    if (merged.status === "fail") return conflict(nodeIndex, opName, name, "type_proto", merged.reason, inferredType?.kind || "unresolved");
    if (!merged.type) return conflict(nodeIndex, opName, name, "type_proto", existingKind, inferredKind);
    tensorMap.set(name, {
      ...existing,
      ...patch,
      ...onnxValueDescriptorFromType(merged.type),
      name,
    });
    return null;
  }
  const existingDtype = knownDtype(existing);
  const inferredDtype = knownDtype(patch);
  if (existingDtype && inferredDtype && existingDtype !== inferredDtype) {
    return conflict(nodeIndex, opName, name, "dtype", existingDtype, inferredDtype);
  }
  let shape = patch.shapeDeclared === true ? patch.shape : existing.shape;
  if (existing.shapeDeclared === true && patch.shapeDeclared === true) {
    if (existing.shape.length !== patch.shape.length) return conflict(nodeIndex, opName, name, "rank", existing.shape.length, patch.shape.length);
    shape = existing.shape.map((declared, index) => {
      const inferred = patch.shape[index];
      if (knownDimension(declared) && knownDimension(inferred) && declared !== inferred) return null;
      return knownDimension(declared) ? declared : inferred;
    });
    const mismatch = shape.findIndex((dim) => dim === null);
    if (mismatch >= 0) return conflict(nodeIndex, opName, name, `dimension_${mismatch}`, existing.shape[mismatch], patch.shape[mismatch]);
  }
  const dtype = inferredDtype || existing.dtype || "UNKNOWN";
  const shapeDeclared = patch.shapeDeclared === true || existing.shapeDeclared === true;
  const inferredType = patch.typeProto || makeOnnxTensorType(dtype, Array.isArray(shape) ? shape : existing.shape, shapeDeclared);
  const mergedType = unifyOnnxTypeProtos(onnxTypeProtoFromValue(existing), inferredType);
  if (mergedType.status === "fail") return conflict(nodeIndex, opName, name, "type_proto", mergedType.reason, "tensor");
  tensorMap.set(name, {
    ...existing,
    ...patch,
    ...onnxValueDescriptorFromType(mergedType.type || inferredType),
    name,
    valueKind: "tensor",
  });
  return null;
}

export function onnxTensorHasConcreteTypeAndShape(tensor) {
  return tensorKnownShape(tensor) && Boolean(knownDtype(tensor));
}

function conflict(nodeIndex, opName, tensorName, field, declared, inferred) {
  return { node_index: nodeIndex, op_name: opName, tensor_name: tensorName, field, declared, inferred };
}

function staticValuePatch(values, source) {
  return {
    staticValuesStatus: "assessed_exact_static_data",
    staticValuesComplete: true,
    staticValues: [...values],
    staticValuesSource: source,
  };
}

function staticCanonicalTextPatch(values, source) {
  return {
    staticValuesStatus: "complete_canonical_text_only_non_finite_or_unsafe_value",
    staticValuesComplete: false,
    staticValues: [],
    staticValuesCanonicalTextComplete: true,
    staticValuesCanonicalTexts: [...values],
    staticValuesSource: source,
  };
}

function staticDimensionValuePatch(dimensions, source) {
  const normalized = dimensions.map(normalizeStaticDimensionValue);
  const numeric = normalized.every((dimension) => dimension.kind === "value")
    ? normalized.map((dimension) => dimension.value) : null;
  return {
    ...(numeric ? staticValuePatch(numeric, source) : {}),
    staticDimensionValuesStatus: "assessed_exact_symbolic_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: normalized,
    staticDimensionValuesSource: source,
  };
}

function inheritedValues(tensor, source) {
  const dimensions = exactDimensionValues(tensor);
  if (dimensions) return { dimensions, source };
  const values = exactNumericValues(tensor);
  return values ? { values, source } : null;
}

function integerElementwiseValues(op, inputs, outputShape) {
  if (!["Add", "Sub", "Mul", "Div", "Max", "Min", "Sum"].includes(op)) return null;
  const shapeNumbers = (outputShape || []).map(dimensionValue);
  if (shapeNumbers.some((value) => value == null || value < 0)) return null;
  const outputElements = boundedStaticElementCount(shapeNumbers);
  if (outputElements == null) return null;
  const dimensionInputs = inputs.map(exactDimensionValues);
  if (dimensionInputs.every(Boolean)) {
    if (!dimensionInputs.every((item) => item.length === 1 || item.length === outputElements)) return null;
    const dimensions = [];
    for (let index = 0; index < outputElements; index += 1) {
      const terms = dimensionInputs.map((item) => item.length === 1 ? item[0] : item[index]);
      const result = evaluateDimensionElementwise(op, terms);
      if (!result) return null;
      dimensions.push(result);
    }
    return { dimensions, source: op };
  }
  const values = inputs.map(exactIntegerValues);
  if (values.some((item) => !item)) return null;
  if (!values.every((item) => item.length === 1 || item.length === outputElements)) return null;
  const result = [];
  for (let index = 0; index < outputElements; index += 1) {
    const terms = values.map((item) => item.length === 1 ? item[0] : item[index]);
    let value = terms[0];
    for (const term of terms.slice(1)) {
      if (op === "Add" || op === "Sum") value += term;
      else if (op === "Sub") value -= term;
      else if (op === "Mul") value *= term;
      else if (op === "Div") {
        if (term === 0) return null;
        value = Math.trunc(value / term);
      } else if (op === "Max") value = Math.max(value, term);
      else if (op === "Min") value = Math.min(value, term);
    }
    if (!Number.isSafeInteger(value)) return null;
    result.push(value);
  }
  return { values: result, source: op };
}

function comparisonElementwiseValues(op, inputs, outputShape) {
  if (!BROADCAST_BOOL.has(op) || inputs.length !== 2) return null;
  const shapeNumbers = (outputShape || []).map(dimensionValue);
  if (shapeNumbers.some((value) => value == null || value < 0)) return null;
  const outputElements = boundedStaticElementCount(shapeNumbers);
  if (outputElements == null) return null;
  const dimensionInputs = inputs.map(exactDimensionValues);
  if (dimensionInputs.every(Boolean)
    && dimensionInputs.every((item) => item.length === 1 || item.length === outputElements)) {
    const values = [];
    for (let index = 0; index < outputElements; index += 1) {
      const left = dimensionInputs[0][dimensionInputs[0].length === 1 ? 0 : index];
      const right = dimensionInputs[1][dimensionInputs[1].length === 1 ? 0 : index];
      const value = compareDimensionTerms(op, left, right);
      if (value == null) return null;
      values.push(value);
    }
    return { values, source: op };
  }
  const numericInputs = inputs.map(exactNumericValues);
  if (numericInputs.some((item) => !item)
    || !numericInputs.every((item) => item.length === 1 || item.length === outputElements)) return null;
  const values = [];
  for (let index = 0; index < outputElements; index += 1) {
    const left = numericInputs[0][numericInputs[0].length === 1 ? 0 : index];
    const right = numericInputs[1][numericInputs[1].length === 1 ? 0 : index];
    values.push(compareNumbers(op, left, right));
  }
  return { values, source: op };
}

function compareDimensionTerms(op, left, right) {
  const leftValue = dimensionValue(left);
  const rightValue = dimensionValue(right);
  if (leftValue != null && rightValue != null) return compareNumbers(op, leftValue, rightValue);
  if (left?.kind === "symbolic" && rightValue != null && rightValue < 0) {
    return op === "Equal" ? false : op === "Greater" || op === "GreaterOrEqual";
  }
  if (right?.kind === "symbolic" && leftValue != null && leftValue < 0) {
    return op === "Equal" ? false : op === "Less" || op === "LessOrEqual";
  }
  if (!dimensionKnown(left) || !dimensionKnown(right)) return null;
  if (dimensionKey(left) === dimensionKey(right)) return compareNumbers(op, 0, 0);
  return null;
}

function compareNumbers(op, left, right) {
  if (op === "Equal") return left === right;
  if (op === "Greater") return left > right;
  if (op === "GreaterOrEqual") return left >= right;
  if (op === "Less") return left < right;
  return left <= right;
}

function whereElementwiseValues(inputs, outputShape) {
  if (knownDtype(inputs[0]) !== "BOOL") return null;
  const shapeNumbers = (outputShape || []).map(dimensionValue);
  if (shapeNumbers.some((value) => value == null || value < 0)) return null;
  const outputElements = boundedStaticElementCount(shapeNumbers);
  if (outputElements == null) return null;
  const conditions = exactNumericValues(inputs[0]);
  if (!conditions || !conditions.every((value) => value === 0 || value === 1)
    || !(conditions.length === 1 || conditions.length === outputElements)) return null;
  const dimensionInputs = [exactDimensionValues(inputs[1]), exactDimensionValues(inputs[2])];
  if (dimensionInputs.every(Boolean)
    && dimensionInputs.every((item) => item.length === 1 || item.length === outputElements)) {
    const dimensions = Array.from({ length: outputElements }, (_, index) => {
      const branch = conditions[conditions.length === 1 ? 0 : index] ? 0 : 1;
      const values = dimensionInputs[branch];
      return cloneDimension(values[values.length === 1 ? 0 : index]);
    });
    return dimensions.every(dimensionKnown) ? { dimensions, source: "Where" } : null;
  }
  const numericInputs = [exactNumericValues(inputs[1]), exactNumericValues(inputs[2])];
  if (numericInputs.some((item) => !item)
    || !numericInputs.every((item) => item.length === 1 || item.length === outputElements)) return null;
  return {
    values: Array.from({ length: outputElements }, (_, index) => {
      const branch = conditions[conditions.length === 1 ? 0 : index] ? 0 : 1;
      const values = numericInputs[branch];
      return values[values.length === 1 ? 0 : index];
    }),
    source: "Where",
  };
}

function boundedStaticElementCount(shape, limit = 65536) {
  const count = elementCount(shape);
  return Number.isSafeInteger(count) && count >= 0 && count <= limit ? count : null;
}

function evaluateDimensionElementwise(op, terms) {
  let current = cloneDimension(terms[0]);
  for (const term of terms.slice(1)) {
    const left = dimensionValue(current);
    const right = dimensionValue(term);
    if (left != null && right != null) {
      if ((op === "Div" && right === 0) || !["Add", "Sum", "Sub", "Mul", "Div", "Max", "Min"].includes(op)) return null;
      const value = op === "Add" || op === "Sum" ? left + right
        : op === "Sub" ? left - right
          : op === "Mul" ? left * right
            : op === "Div" ? Math.trunc(left / right)
              : op === "Max" ? Math.max(left, right) : Math.min(left, right);
      if (!Number.isSafeInteger(value)) return null;
      current = valueDimension(value);
      continue;
    }
    if (!dimensionKnown(current) || !dimensionKnown(term)) return null;
    if ((op === "Add" || op === "Sum") && right === 0 || op === "Mul" && right === 1) continue;
    if ((op === "Add" || op === "Sum") && left === 0 || op === "Mul" && left === 1) {
      current = cloneDimension(term);
      continue;
    }
    const operator = op === "Sum" ? "add" : op.toLowerCase();
    current = symbolicDimension(`deepbom_expr:${operator}(${dimensionKey(current)},${dimensionKey(term)})`);
  }
  return current;
}

function exactNumericValues(tensor) {
  if (tensor?.staticValuesComplete !== true || !Array.isArray(tensor.staticValues)) return null;
  return tensor.staticValues.every((value) => Number.isFinite(Number(value))) ? tensor.staticValues.map(Number) : null;
}

function exactIntegerValues(tensor) {
  const values = exactNumericValues(tensor);
  return values?.every(Number.isSafeInteger) ? values : null;
}

function exactSliceIntegerValues(tensor, allowUnbounded) {
  const values = exactIntegerValues(tensor);
  if (values) return allowUnbounded ? values.map((value) => normalizeSliceSentinel(tensor, value)) : values;
  if (tensor?.staticValuesCanonicalTextComplete !== true || !Array.isArray(tensor.staticValuesCanonicalTexts)) return null;
  const minimum = BigInt(Number.MIN_SAFE_INTEGER);
  const maximum = BigInt(Number.MAX_SAFE_INTEGER);
  const output = [];
  for (const text of tensor.staticValuesCanonicalTexts) {
    if (!/^-?(?:0|[1-9]\d*)$/.test(String(text))) return null;
    const value = BigInt(text);
    if (value >= minimum && value <= maximum) output.push(allowUnbounded
      ? normalizeSliceSentinel(tensor, Number(value)) : Number(value));
    else if (allowUnbounded) output.push(normalizeSliceSentinel(tensor, value));
    else return null;
  }
  return output;
}

function normalizeSliceSentinel(tensor, value) {
  const dtype = knownDtype(tensor);
  if (dtype === "INT32") {
    if (value === 2147483647) return Number.POSITIVE_INFINITY;
    if (value === -2147483648) return Number.NEGATIVE_INFINITY;
  }
  if (dtype === "INT64") {
    if (value === 9223372036854775807n) return Number.POSITIVE_INFINITY;
    if (value === -9223372036854775808n) return Number.NEGATIVE_INFINITY;
  }
  return value;
}

function exactDimensionValues(tensor) {
  if (tensor?.staticDimensionValuesComplete === true && Array.isArray(tensor.staticDimensionValues)) {
    return tensor.staticDimensionValues.map(normalizeStaticDimensionValue);
  }
  const values = exactIntegerValues(tensor);
  return values ? values.map(valueDimension) : null;
}

function normalizeDomain(domain) { return !domain || domain === "ai.onnx" ? "ai.onnx" : domain; }
function knownDtype(tensor) {
  return tensor?.contractStatus !== "invalid" && (!tensor?.valueKind || tensor.valueKind === "tensor")
    && tensor?.dtype && tensor.dtype !== "UNKNOWN" ? tensor.dtype : "";
}
function integerDtype(dtype) { return /^(?:U?INT(?:2|4|8|16|32|64)|BOOL)$/.test(String(dtype || "")); }
function knownDimension(dim) { return Number.isSafeInteger(Number(dim)) && Number(dim) >= 0; }
function knownShape(shape) { return Array.isArray(shape) && shape.every(knownDimension); }
function tensorKnownShape(tensor) {
  return tensor?.contractStatus !== "invalid" && tensor?.conditionalShapeContract?.status !== "assessed_partial"
    && (!tensor?.valueKind || tensor.valueKind === "tensor")
    && tensor?.shapeDeclared === true && knownShape(tensor.shape);
}
function tensorRankKnown(tensor) {
  return tensor?.contractStatus !== "invalid" && (!tensor?.valueKind || tensor.valueKind === "tensor")
    && tensor?.shapeDeclared === true && Array.isArray(tensor.shape);
}
function tensorShapeDimensions(tensor) { return tensorRankKnown(tensor) ? onnxShapeDimensionsFromValue(tensor) : null; }
function tensorKnown(tensor) {
  return tensorKnownShape(tensor) && Boolean(knownDtype(tensor));
}
function unconditionalShapeContractKnown(tensor) {
  if (tensor?.contractStatus === "invalid" || tensor?.conditionalShapeContract?.status === "assessed_partial") return false;
  const dimensions = tensorShapeDimensions(tensor);
  return Boolean(knownDtype(tensor)) && Array.isArray(dimensions) && dimensions.every(dimensionKnown);
}
function conditionalShapeVariants(tensor) {
  if (tensor?.contractStatus === "invalid" || tensor?.conditionalShapeContract?.status !== "assessed_complete"
    || !Array.isArray(tensor.conditionalShapeVariants) || !tensor.conditionalShapeVariants.length) return [];
  return tensor.conditionalShapeVariants.every((variant) => unconditionalShapeContractKnown(variant))
    ? tensor.conditionalShapeVariants : [];
}
function propagatableConditionalShapeVariants(tensor) {
  const status = tensor?.conditionalShapeContract?.status;
  if (tensor?.contractStatus === "invalid" || !["assessed_complete", "assessed_partial"].includes(status)
    || !Array.isArray(tensor.conditionalShapeVariants) || !tensor.conditionalShapeVariants.length) return [];
  return tensor.conditionalShapeVariants.every((variant) => unconditionalShapeContractKnown(variant))
    ? tensor.conditionalShapeVariants : [];
}
function conditionalShapeContractKnown(tensor) { return conditionalShapeVariants(tensor).length > 0; }
function tensorShapeContractKnown(tensor) {
  return unconditionalShapeContractKnown(tensor) || conditionalShapeContractKnown(tensor);
}
function valueContractKnown(value) {
  if (!value) return false;
  if (!declaredNonDenseValue(value)) return tensorShapeContractKnown(value);
  return onnxTypeProtoKnown(onnxTypeProtoFromValue(value));
}
function firstInvalidInputContract(node, tensorMap) {
  for (const name of node.inputs || []) {
    if (!name) continue;
    const tensor = tensorMap.get(name);
    if (tensor?.contractStatus === "invalid") {
      return {
        tensor_name: name,
        reason: tensor.contractConflict?.reason || "upstream_contract_invalid",
        root_conflict: tensor.contractConflict?.root_conflict || tensor.contractConflict || null,
      };
    }
  }
  return null;
}
function markNodeOutputsInvalid(node, tensorMap, conflictEvidence) {
  for (const name of node.outputs || []) {
    if (!name) continue;
    markTensorContractInvalid(tensorMap, name, conflictEvidence);
  }
}
function markTensorContractInvalid(tensorMap, name, conflictEvidence) {
  const existing = tensorMap.get(name) || { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  tensorMap.set(name, {
    ...existing,
    name,
    contractStatus: "invalid",
    contractConflict: conflictEvidence,
  });
}
function mergeConditionalConditions(left, right) {
  const values = new Map();
  for (const condition of [...(left || []), ...(right || [])]) {
    const key = String(condition?.key || "");
    const value = String(condition?.value || "");
    if (!key || !value) return null;
    if (values.has(key) && values.get(key) !== value) return null;
    values.set(key, value);
  }
  return [...values].sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => ({ key, value }));
}
function deduplicateConditionalVariants(variants) {
  const unique = new Map();
  for (const variant of variants) {
    const key = JSON.stringify({
      conditions: variant.conditions || [],
      dtype: knownDtype(variant),
      dimensions: canonicalShapeDimensions(tensorShapeDimensions(variant)),
    });
    if (!unique.has(key)) unique.set(key, variant);
  }
  return [...unique.values()];
}
function deduplicateConditionalFailures(failures) {
  const unique = new Map();
  for (const failure of failures || []) {
    const normalized = {
      status: failure?.status || "not_assessed",
      reason: failure?.reason || "conditional_shape_variant_not_assessed",
      output_name: failure?.output_name || undefined,
      conditions: structuredClone(failure?.conditions || []),
      details: failure?.details || null,
    };
    const key = JSON.stringify(normalized);
    if (!unique.has(key)) unique.set(key, normalized);
  }
  return [...unique.values()];
}
function normalizedValueKind(value, implicitTensor) {
  const kind = String(value?.valueKind || value?.value_kind || "");
  if (kind && kind !== "undefined" && kind !== "unresolved") return kind;
  if (!kind && (implicitTensor || value?.shapeDeclared === true || value?.shape_declared === true
    || value?.dtype && value.dtype !== "UNKNOWN" || value?.typeProto?.kind === "tensor" || value?.type_proto?.kind === "tensor")) return "tensor";
  return implicitTensor ? "tensor" : "unresolved";
}
function exactElementCount(shape) {
  if (!Array.isArray(shape)) return null;
  let product = 1n;
  for (const rawDimension of shape) {
    const dimension = Number(rawDimension);
    if (!Number.isSafeInteger(dimension) || dimension < 0) return null;
    product *= BigInt(dimension);
  }
  return product;
}
function elementCount(shape) {
  const exact = exactElementCount(shape);
  return exact != null && exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null;
}
function positive(value) { return Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : 0; }

function attrInts(node, name) {
  const values = node.attributes?.get(name)?.ints || [];
  return values.every(Number.isSafeInteger) ? values : [];
}
function attrInt(node, name, fallback) {
  const attribute = node.attributes?.get(name);
  if (Number.isSafeInteger(attribute?.i)) return attribute.i;
  return Number.isSafeInteger(attribute?.ints?.[0]) ? attribute.ints[0] : fallback;
}
function attrString(node, name, fallback = "") { return node.attributes?.get(name)?.s || fallback; }

function normalizeAxis(axis, rank) {
  let value = Number(axis);
  if (!Number.isSafeInteger(value) || rank < 0) return null;
  if (value < 0) value += rank;
  return value >= 0 && value < rank ? value : null;
}
function normalizeAxes(axes, rank) {
  if (!Array.isArray(axes)) return null;
  const normalized = axes.map((axis) => normalizeAxis(axis, rank));
  return normalized.some((axis) => axis == null) || new Set(normalized).size !== normalized.length ? null : normalized.sort((a, b) => a - b);
}
function normalizeAxesInOrder(axes, rank) {
  if (!Array.isArray(axes)) return null;
  const normalized = axes.map((axis) => normalizeAxis(axis, rank));
  return normalized.some((axis) => axis == null) || new Set(normalized).size !== normalized.length ? null : normalized;
}
function normalizeSliceBound(value, rank, isEnd) {
  let bound = Number(value);
  if (!Number.isSafeInteger(bound)) return isEnd ? rank : 0;
  if (bound < 0) bound += rank;
  return Math.min(rank, Math.max(0, bound));
}
function normalizeSliceRange(dim, start, end, step) {
  const validBound = (value) => Number.isSafeInteger(value) || typeof value === "bigint"
    || value === Number.NEGATIVE_INFINITY || value === Number.POSITIVE_INFINITY;
  if (!Number.isSafeInteger(dim) || dim < 0 || !validBound(start) || !validBound(end) || !Number.isSafeInteger(step) || step === 0) return null;
  if (step > 0) {
    const first = normalizeConcreteSliceBound(start, dim, 0, dim);
    const last = normalizeConcreteSliceBound(end, dim, 0, dim);
    return { start: first, end: last, step, length: Math.max(0, Math.ceil((last - first) / step)) };
  }
  const first = normalizeConcreteSliceBound(start, dim, -1, dim - 1);
  const last = normalizeConcreteSliceBound(end, dim, -1, dim - 1);
  return { start: first, end: last, step, length: Math.max(0, Math.ceil((first - last) / -step)) };
}

function normalizeConcreteSliceBound(bound, dim, minimum, maximum) {
  if (bound === Number.NEGATIVE_INFINITY) return minimum;
  if (bound === Number.POSITIVE_INFINITY) return maximum;
  if (typeof bound === "bigint") {
    const dimension = BigInt(dim);
    const adjusted = bound < 0n ? bound + dimension : bound;
    if (adjusted <= BigInt(minimum)) return minimum;
    if (adjusted >= BigInt(maximum)) return maximum;
    return Number(adjusted);
  }
  const adjusted = bound < 0 ? bound + dim : bound;
  return Math.min(maximum, Math.max(minimum, adjusted));
}

function normalizeShapeDimensions(shape) {
  if (!Array.isArray(shape)) return null;
  return shape.map((dimension) => {
    const normalized = normalizeStaticDimensionValue(dimension);
    return normalized.kind === "value" && normalized.value < 0 ? unknownDimension() : normalized;
  });
}

function normalizeStaticDimensionValue(dimension) {
  if (dimension?.kind === "value" && Number.isSafeInteger(Number(dimension.value))) return valueDimension(Number(dimension.value));
  if (dimension?.kind === "symbolic" && String(dimension.parameter || "")) return symbolicDimension(String(dimension.parameter));
  if (Number.isSafeInteger(Number(dimension))) return valueDimension(Number(dimension));
  return unknownDimension();
}

function valueDimension(value) {
  return { kind: "value", value: Number(value), parameter: "", denotation: "", valueFieldCount: 1 };
}

function symbolicDimension(parameter) {
  return { kind: "symbolic", value: null, parameter: String(parameter), denotation: "", valueFieldCount: 1 };
}

function unknownDimension() {
  return { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 };
}

function cloneDimension(dimension) { return { ...normalizeStaticDimensionValue(dimension) }; }
function dimensionValue(dimension) { return dimension?.kind === "value" && Number.isSafeInteger(Number(dimension.value)) ? Number(dimension.value) : null; }
function dimensionKnown(dimension) { return dimensionValue(dimension) != null && dimensionValue(dimension) >= 0 || dimension?.kind === "symbolic" && Boolean(String(dimension.parameter || "")); }
function dimensionKey(dimension) {
  const value = dimensionValue(dimension);
  if (value != null) return `v:${value}`;
  if (dimension?.kind === "symbolic" && dimension.parameter) return `s:${dimension.parameter}`;
  return "?";
}
function dimensionParameter(dimension) {
  return dimension?.kind === "symbolic" && dimension.parameter ? String(dimension.parameter) : null;
}
function canonicalShapeDimensions(dimensions) {
  return Array.isArray(dimensions) ? dimensions.map((dimension) => ({
    kind: dimension?.kind || "unknown",
    value: dimensionValue(dimension),
    parameter: dimensionParameter(dimension),
  })) : [];
}
function dimensionsCompatible(left, right) {
  return dimensionKey(left) === dimensionKey(right) || dimensionValue(left) == null || dimensionValue(right) == null;
}
function addDimensions(left, right) {
  const leftValue = dimensionValue(left);
  const rightValue = dimensionValue(right);
  if (leftValue != null && rightValue != null) return valueDimension(leftValue + rightValue);
  if (leftValue === 0) return cloneDimension(right);
  if (rightValue === 0) return cloneDimension(left);
  if (!dimensionKnown(left) || !dimensionKnown(right)) return unknownDimension();
  return symbolicDimension(`deepbom_expr:add(${dimensionKey(left)},${dimensionKey(right)})`);
}
function addDimensionOffset(dimension, offset) {
  if (!Number.isSafeInteger(offset) || !dimensionKnown(dimension)) return unknownDimension();
  const value = dimensionValue(dimension);
  if (value != null) return value + offset >= 0 ? valueDimension(value + offset) : unknownDimension();
  if (offset === 0) return cloneDimension(dimension);
  return symbolicDimension(`deepbom_expr:add(${dimensionKey(dimension)},${offset})`);
}
function multiplyDimension(dimension, factor) {
  const value = dimensionValue(dimension);
  if (!Number.isSafeInteger(Number(factor)) || Number(factor) < 0 || !dimensionKnown(dimension)) return unknownDimension();
  if (value != null) return valueDimension(value * Number(factor));
  if (Number(factor) === 0) return valueDimension(0);
  if (Number(factor) === 1) return cloneDimension(dimension);
  return symbolicDimension(`deepbom_expr:mul(${dimensionKey(dimension)},${Number(factor)})`);
}

function multiplyDimensions(left, right) {
  const leftValue = dimensionValue(left);
  const rightValue = dimensionValue(right);
  if (leftValue != null && rightValue != null) {
    const value = leftValue * rightValue;
    return Number.isSafeInteger(value) && value >= 0 ? valueDimension(value) : unknownDimension();
  }
  if (leftValue === 0 || rightValue === 0) return valueDimension(0);
  if (leftValue === 1) return cloneDimension(right);
  if (rightValue === 1) return cloneDimension(left);
  if (!dimensionKnown(left) || !dimensionKnown(right)) return unknownDimension();
  return symbolicDimension(`deepbom_expr:mul(${dimensionKey(left)},${dimensionKey(right)})`);
}

function productDimensions(dimensions) {
  return dimensions.reduce((product, dimension) => multiplyDimensions(product, dimension), valueDimension(1));
}

function inferSymbolicReshapeDimension(inputDimensions, knownTargetDimensions) {
  const numerator = dimensionProductFactors(inputDimensions);
  const denominator = dimensionProductFactors(knownTargetDimensions);
  if (!numerator || !denominator || denominator.numeric === 0n) return unknownDimension();

  const denominatorSymbols = new Map();
  for (const symbol of denominator.symbols) denominatorSymbols.set(symbol, (denominatorSymbols.get(symbol) || 0) + 1);
  const numeratorSymbols = [];
  for (const symbol of numerator.symbols) {
    const available = denominatorSymbols.get(symbol) || 0;
    if (available > 0) denominatorSymbols.set(symbol, available - 1);
    else numeratorSymbols.push(symbol);
  }
  const remainingDenominatorSymbols = [...denominatorSymbols.entries()]
    .flatMap(([symbol, count]) => Array(count).fill(symbol));

  if (!remainingDenominatorSymbols.length && numerator.numeric % denominator.numeric === 0n) {
    const scalar = numerator.numeric / denominator.numeric;
    if (scalar <= BigInt(Number.MAX_SAFE_INTEGER)) {
      let result = valueDimension(Number(scalar));
      for (const symbol of numeratorSymbols) result = multiplyDimensions(result, symbolicDimension(symbol));
      return result;
    }
  }

  return symbolicDimension(`deepbom_expr:reshape_quotient(${dimensionProductExpression(numerator.numeric, numeratorSymbols)},${dimensionProductExpression(denominator.numeric, remainingDenominatorSymbols)})`);
}

function dimensionProductFactors(dimensions) {
  let numeric = 1n;
  const symbols = [];
  for (const dimension of dimensions || []) {
    const value = dimensionValue(dimension);
    if (value != null && value >= 0) numeric *= BigInt(value);
    else if (dimension?.kind === "symbolic" && dimension.parameter) symbols.push(String(dimension.parameter));
    else return null;
  }
  return { numeric, symbols: symbols.sort() };
}

function dimensionProductExpression(numeric, symbols) {
  const factors = [];
  if (numeric !== 1n || !symbols.length) factors.push(`v:${numeric}`);
  factors.push(...symbols.map((symbol) => `s:${symbol}`));
  return factors.length === 1 ? factors[0] : `deepbom_expr:mul(${factors.join(",")})`;
}

function contractionDimensionsCanMatch(left, right) {
  const leftValue = dimensionValue(left);
  const rightValue = dimensionValue(right);
  return leftValue == null || rightValue == null || leftValue === rightValue;
}

function broadcastManyDimensions(shapes) {
  if (!Array.isArray(shapes) || shapes.some((shape) => !Array.isArray(shape))) return null;
  const rank = Math.max(0, ...shapes.map((shape) => shape.length));
  const output = [];
  for (let offset = 1; offset <= rank; offset += 1) {
    const dimensions = shapes.map((shape) => shape.at(-offset) || valueDimension(1));
    const nonOne = dimensions.filter((dimension) => dimensionValue(dimension) !== 1);
    if (!nonOne.length) {
      output.unshift(valueDimension(1));
      continue;
    }
    const concrete = [...new Set(nonOne.map(dimensionValue).filter((value) => value != null))];
    if (concrete.length > 1) return null;
    if (concrete.length === 1) {
      output.unshift(valueDimension(concrete[0]));
      continue;
    }
    const symbols = [...new Set(nonOne.filter((dimension) => dimension?.kind === "symbolic").map((dimension) => dimension.parameter))].sort();
    const hasUnknown = nonOne.some((dimension) => !dimensionKnown(dimension));
    output.unshift(hasUnknown ? unknownDimension()
      : symbols.length === 1 ? symbolicDimension(symbols[0])
        : symbols.length > 1 ? symbolicDimension(`deepbom_expr:broadcast_dim(${symbols.map((symbol) => `s:${symbol}`).join(",")})`)
          : unknownDimension());
  }
  return output;
}

function broadcastMany(shapes) {
  if (!Array.isArray(shapes) || shapes.some((shape) => !knownShape(shape))) return null;
  const rank = Math.max(0, ...shapes.map((shape) => shape.length));
  const output = [];
  for (let offset = 1; offset <= rank; offset += 1) {
    const dims = shapes.map((shape) => shape.at(-offset) ?? 1);
    const nonOne = [...new Set(dims.filter((dim) => dim !== 1))];
    if (nonOne.length > 1) return null;
    output.unshift(nonOne[0] ?? 1);
  }
  return output;
}
function sameShape(left, right) { return left.length === right.length && left.every((dim, index) => dim === right[index]); }
function validPermutation(perm, rank) { return perm.length === rank && perm.every((axis) => Number.isSafeInteger(axis) && axis >= 0 && axis < rank) && new Set(perm).size === rank; }
function padTo(values, length, fallback, allowZero = false) {
  const result = Array.isArray(values) ? values.slice(0, length) : [];
  while (result.length < length) result.push(fallback);
  return result.map((value) => Number.isSafeInteger(value) && (allowZero ? value >= 0 : value > 0) ? value : fallback);
}
function countBy(values) {
  const counts = new Map();
  for (const value of values) counts.set(value || "UNKNOWN", (counts.get(value || "UNKNOWN") || 0) + 1);
  return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}
function nullableSum(rows, field) {
  if (!rows.length || rows.some((row) => !Number.isSafeInteger(row[field]) || row[field] < 0)) return null;
  const total = rows.reduce((sum, row) => sum + row[field], 0);
  return Number.isSafeInteger(total) ? total : null;
}
function resolved(outputs, reason = "") { return { outputs, reason }; }
function unresolved(reason) { return { outputs: [], reason }; }
function invalid(reason, details = null) { return { status: "invalid", outputs: [], reason, details }; }
function attachOutputs(result, outputs) { return { ...result, outputs }; }
