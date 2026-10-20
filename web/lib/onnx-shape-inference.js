import { onnxSemantic, onnxTensorRecord, onnxNodeSemantic } from "./onnx-portable.js";
import "./portable-contracts.js";
import "./scalar-numeric.js";
import { ONNX_SHAPE_INFERENCE_OPS } from "./onnx-shape-ops.js";
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
import { onnxDimensionInteger, onnxShapeDimensionsFromValue, onnxTypeProtoFromValue, onnxTypeProtoKnown } from "./onnx-type-proto.js";
import "./onnx-einsum-contract.js";
import {
  canInferOnnxTfIdfVectorizer,
  inferOnnxTfIdfVectorizerNode,
  ONNX_TFIDF_VECTORIZER_SOURCE,
} from "./onnx-tfidf-vectorizer.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const MAX_CONDITIONAL_SHAPE_VARIANTS = 64;
const RUNTIME_DIMENSION_PREFIX = "deepbom_runtime:";
export const ONNX_SHAPE_INFERENCE_SCHEMA = "deepbom.onnx_shape_inference.v1.31";

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

function inferNodeWithConditionalShapes(node, tensors) {
  return onnxNodeSemantic("conditional_node", node, tensors);
}

export function mergeOnnxInferredTensor(tensorMap,name,patch,nodeIndex,opName) {
  const existing=tensorMap.get(name)||{name,dtype:"UNKNOWN",shape:[],shapeDeclared:false};
  const result=onnxSemantic("merge_tensor",{existing:onnxTensorRecord(existing),patch:onnxTensorRecord(patch),name,node_index:nodeIndex,op_name:opName});
  if(result.conflict)return result.conflict;
  tensorMap.set(name,{...existing,...patch,...result.descriptor,name});
  return null;
}
export function onnxTensorHasConcreteTypeAndShape(tensor) {
  return onnxSemantic("concrete_tensor",{existing:onnxTensorRecord(tensor)});
}

function conflict(nodeIndex, opName, tensorName, field, declared, inferred) {
  return { node_index: nodeIndex, op_name: opName, tensor_name: tensorName, field, declared, inferred };
}

function normalizeDomain(domain) { return !domain || domain === "ai.onnx" ? "ai.onnx" : domain; }
function knownDtype(tensor) {
  return tensor?.contractStatus !== "invalid" && (!tensor?.valueKind || tensor.valueKind === "tensor")
    && tensor?.dtype && tensor.dtype !== "UNKNOWN" ? tensor.dtype : "";
}

function knownDimension(dim) { const value = onnxDimensionInteger(dim); return value != null && value >= 0; }
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

function dimensionValue(dimension) { return dimension?.kind === "value" && onnxDimensionInteger(dimension.value) != null ? Number(dimension.value) : null; }
function dimensionKnown(dimension) { return dimensionValue(dimension) != null && dimensionValue(dimension) >= 0 || dimension?.kind === "symbolic" && Boolean(String(dimension.parameter || "")); }

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

// Adapter boundary: extract attributes once; semantic arithmetic is owned by WASM.
