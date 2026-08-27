import {
  inferOnnxShapes,
  mergeOnnxInferredTensor,
} from "./onnx-shape-inference.js";
import { assessOnnxOpsetImports } from "./onnx-opset-imports.js";
import {
  assessOnnxAttributeProto,
  resolveOnnxSchemaSinceVersion,
} from "./onnx-schema-legality.js";
import {
  canonicalOnnxTypeProto,
  cloneOnnxTypeProto,
  makeOnnxOptionalType,
  makeOnnxSequenceType,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxTypeProtoKnown,
  onnxValueDescriptorFromType,
  unionOnnxTypeProtos,
} from "./onnx-type-proto.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const CONTROL_FLOW_OPS = new Set(["If", "Loop", "Scan"]);
const MAX_SCOPE_DEPTH = 128;
const MAX_SEQUENCE_MAP_ELEMENT_NODE_EVALUATIONS = 16_384;
const MAX_LOOP_BODY_NODE_EVALUATIONS = 16_384;
const MAX_LOOP_EXACT_ITERATIONS = 4_096;
const MAX_INTRINSIC_COST_VARIANTS = 1_024;

export const ONNX_EXTENDED_SHAPE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  documents: Object.freeze([
    Object.freeze({
      role: "function_proto_contract",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/onnx.in.proto`,
      sha256: "f4cbc198df3a0f3f4519d4d38cd2262e8f84057583b7313e2d0f981b3f68c213",
    }),
    Object.freeze({
      role: "control_flow_shape_inference",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/controlflow/utils.cc`,
      sha256: "48ced14e52a8c2d9a8e230f1be3c6428c6bd074e923d035f962a0626215d3d33",
    }),
    Object.freeze({
      role: "current_control_flow_schema",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/controlflow/defs.cc`,
      sha256: "67d03c30742c96fae1f79831b28d2409d5dcb4921ea77f694a4172f66ceebeff",
    }),
    Object.freeze({
      role: "historical_control_flow_schema",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/controlflow/old.cc`,
      sha256: "de85b9527008a725e718c593313c38c7aeea30c8781027ddea546a5dd80f5283",
    }),
  ]),
});

export function inferOnnxShapesWithReachableScopes(graph, tensorMap, model, tensorTypeName, domainAnalysis = null, intrinsicCostAssessor = null) {
  const functions = model?.functions || [];
  const functionById = new Map(functions.map((fn) => [functionId(fn.domain, fn.name, fn.overload), fn]));
  const context = {
    tensorTypeName,
    functions,
    functionById,
    domainAnalysis,
    modelOpsets: model?.opsets || [],
    scopeExecution: { rows: new Map(), intrinsicCostVariantCount: 0, intrinsicCostVariantOverflowCount: 0 },
    intrinsicCostAssessor,
    functionCallRows: [],
    controlFlowRows: [],
    sequenceMapRows: [],
  };
  const evidence = executeGraph(graph, tensorMap, context.modelOpsets, "main_graph", context, [], false);
  const mainGraphIntrinsicCost = intrinsicCostAssessor ? intrinsicCostAssessor(graph, tensorMap) : null;
  const scopeRows = [...context.scopeExecution.rows.values()]
    .map(finalizeScopeRow)
    .sort((left, right) => left.scope.localeCompare(right.scope) || left.scope_class.localeCompare(right.scope_class));
  const failedFunctionCalls = context.functionCallRows.filter((row) => row.status === "fail");
  const failedControlFlow = context.controlFlowRows.filter((row) => row.status === "fail");
  const partialControlFlow = context.controlFlowRows.filter((row) => row.status === "partial");
  const failedSequenceMaps = context.sequenceMapRows.filter((row) => row.status === "fail");
  const partialSequenceMaps = context.sequenceMapRows.filter((row) => row.status === "partial");
  const failedScopes = scopeRows.filter((row) => row.status === "fail");
  const partialScopes = scopeRows.filter((row) => row.status === "partial");
  const loopRows = context.controlFlowRows.filter((row) => row.op_name === "Loop");
  evidence.extended_scope_inference = {
    schema: "deepbom.onnx_extended_shape_inference.v1.6",
    evidence_class: "SOURCE_PINNED_AND_DERIVED",
    status: failedFunctionCalls.length || failedControlFlow.length || failedSequenceMaps.length || failedScopes.length
      ? "fail"
      : partialControlFlow.length || partialSequenceMaps.length || partialScopes.length ? "partial" : "assessed",
    source_release: ONNX_EXTENDED_SHAPE_SOURCE.release,
    source_commit: ONNX_EXTENDED_SHAPE_SOURCE.commit,
    source_documents: ONNX_EXTENDED_SHAPE_SOURCE.documents.map((row) => ({ ...row })),
    local_function_call_count: context.functionCallRows.length,
    local_function_call_pass_count: context.functionCallRows.filter((row) => row.status === "pass").length,
    local_function_call_fail_count: failedFunctionCalls.length,
    function_call_rows: context.functionCallRows,
    control_flow_node_count: context.controlFlowRows.length,
    control_flow_pass_count: context.controlFlowRows.filter((row) => row.status === "pass").length,
    control_flow_partial_count: context.controlFlowRows.filter((row) => row.status === "partial").length,
    control_flow_fail_count: failedControlFlow.length,
    control_flow_rows: context.controlFlowRows,
    loop_node_count: loopRows.length,
    loop_exact_expansion_count: loopRows.filter((row) => row.exact_expansion_status === "assessed").length,
    loop_exact_iteration_count: loopRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.exact_iteration_count) ? row.exact_iteration_count : 0), 0),
    loop_exact_body_node_evaluation_count: loopRows.reduce((sum, row) => sum + (Number.isSafeInteger(row.exact_body_node_evaluation_count) ? row.exact_body_node_evaluation_count : 0), 0),
    loop_non_dense_state_variable_count: loopRows.reduce((sum, row) => sum + Number(row.non_dense_state_variable_count || 0), 0),
    sequence_map_node_count: context.sequenceMapRows.length,
    sequence_map_pass_count: context.sequenceMapRows.filter((row) => row.status === "pass").length,
    sequence_map_partial_count: context.sequenceMapRows.filter((row) => row.status === "partial").length,
    sequence_map_fail_count: failedSequenceMaps.length,
    sequence_map_rows: context.sequenceMapRows,
    scope_execution_count: scopeRows.reduce((sum, row) => sum + row.execution_count, 0),
    scope_definition_count: scopeRows.length,
    fully_assessed_scope_count: scopeRows.filter((row) => row.status === "assessed").length,
    residual_unassessed_node_count: scopeRows.reduce((sum, row) => sum + row.unassessed_node_count, 0),
    residual_unresolved_output_count: scopeRows.reduce((sum, row) => sum + row.unresolved_output_count, 0),
    intrinsic_cost_variant_count: context.scopeExecution.intrinsicCostVariantCount,
    intrinsic_cost_variant_overflow_count: context.scopeExecution.intrinsicCostVariantOverflowCount,
    intrinsic_cost_unassessed_execution_count: scopeRows.reduce((sum, row) => sum + row.intrinsic_cost_unassessed_execution_count, 0),
    main_graph_intrinsic_cost: mainGraphIntrinsicCost,
    scope_rows: scopeRows,
    method: "Recursively bind model-local FunctionProto calls and execute pinned If, Loop, Scan, and SequenceMap graph-inference contracts. Each scope uses its own opset imports, graph inputs, lexical captures, initializer metadata, value_info, and output declarations. Dynamic If preserves the ordinary ONNX union type and a bounded finite set of condition-keyed tensor shape variants; compatible downstream rules execute once per condition assignment. When supplied, one shared operation-cost assessor records unique one-invocation intrinsic MAC/payload variants without multiplying them by scope execution counts. Bounded exact expansion preserves SequenceMap inventories and evaluates Loop iterations when the trip count and every reached condition are artifact-known.",
    interpretation_boundary: "A finite conditional shape contract proves the complete set of statically reachable branch contracts but does not observe which branch executes. Direct Sequence and Optional operators inside an executed scope use the pinned container-value pass. Model-local function outputs, If branch unions, Loop-13 sequence state, and Loop-16+ optional state preserve non-dense TypeProto contracts; bounded exact Loop expansion additionally preserves final container state and exact scan leading dimensions. Scan remains tensor-only under the pinned schemas. Dynamic Loop control, incompatible or over-limit conditional products, work-budget overflows, ai.onnx.ml map-producing operators, sparse-value operator algebra, and unresolved runtime values remain explicit residuals.",
  };
  return evidence;
}

function executeGraph(graph, tensorMap, opsets, scope, context, callStack, recordScope, scopeClass = "nested_graph") {
  if (callStack.length > MAX_SCOPE_DEPTH) {
    const evidence = failedGraphEvidence(graph, "extended_shape_scope_depth_limit_exceeded");
    if (recordScope) recordScopeExecution(context.scopeExecution, scope, scopeClass, graph?.nodes?.length || 0, evidence, null, Boolean(context.intrinsicCostAssessor));
    return evidence;
  }
  const options = {
    scope,
    scopeExecution: context.scopeExecution,
    canResolveNode: (node) => canResolveExtendedNode(node, context),
    resolveNodeResult: ({ node, nodeIndex, tensorMap: liveTensors, importedOpset }) => resolveExtendedNode({
      node, nodeIndex, tensorMap: liveTensors, importedOpset, opsets, scope, context, callStack,
    }),
  };
  const evidence = inferOnnxShapes(
    graph,
    tensorMap,
    opsets,
    context.tensorTypeName,
    context.functions,
    context.domainAnalysis,
    options,
  );
  const intrinsicCost = context.intrinsicCostAssessor ? context.intrinsicCostAssessor(graph, tensorMap) : null;
  if (recordScope) recordScopeExecution(context.scopeExecution, scope, scopeClass, graph?.nodes?.length || 0, evidence, intrinsicCost, Boolean(context.intrinsicCostAssessor));
  return evidence;
}

function canResolveExtendedNode(node, context) {
  const domain = normalizeDomain(node?.domain);
  if (domain === "ai.onnx" && (CONTROL_FLOW_OPS.has(node?.opType) || node?.opType === "SequenceMap")) return true;
  return context.functionById.has(functionId(node?.domain, node?.opType, node?.overload));
}

function resolveExtendedNode(args) {
  const domain = normalizeDomain(args.node.domain);
  if (domain === "ai.onnx" && args.node.opType === "If") return inferIf(args);
  if (domain === "ai.onnx" && args.node.opType === "Loop") return inferLoop(args);
  if (domain === "ai.onnx" && args.node.opType === "Scan") return inferScan(args);
  if (domain === "ai.onnx" && args.node.opType === "SequenceMap") return inferSequenceMap(args);
  return inferFunctionCall(args);
}

function inferSequenceMap({ node, nodeIndex, tensorMap, opsets, scope, context, callStack, importedOpset }) {
  const row = {
    scope,
    node_index: nodeIndex,
    op_name: "SequenceMap",
    imported_opset: importedOpset,
    status: "fail",
    input_count: (node.inputs || []).filter(Boolean).length,
    output_count: (node.outputs || []).filter(Boolean).length,
    exact_input_sequence_length: null,
    element_expansion_count: 0,
    element_node_evaluation_count: 0,
    reason_codes: [],
  };
  const fail = (reason) => {
    row.reason_codes.push(reason);
    row.reason_codes = [...new Set(row.reason_codes.filter(Boolean))];
    row.status = "fail";
    context.sequenceMapRows.push(row);
    return extendedFailure(row, row.reason_codes[0]);
  };
  const body = node.attributes?.get("body")?.graph;
  if (!body) return fail("sequence_map_body_graph_missing");
  if (!row.input_count || !row.output_count) return fail("sequence_map_input_or_output_missing");
  if ((body.inputs || []).length !== row.input_count || (body.outputs || []).length !== row.output_count) {
    return fail("sequence_map_body_cardinality_mismatch");
  }

  const inputValues = (node.inputs || []).map((name) => descriptor(tensorMap.get(name)));
  const inputTypes = inputValues.map(onnxTypeProtoFromValue);
  if (!validSequenceMapSequenceType(inputTypes[0])) return fail("sequence_map_first_input_not_typed_tensor_sequence");
  for (let index = 1; index < inputTypes.length; index += 1) {
    if (inputTypes[index]?.kind === "sequence" ? !validSequenceMapSequenceType(inputTypes[index]) : inputTypes[index]?.kind !== "tensor" || !onnxTypeProtoKnown(inputTypes[index])) {
      return fail(`sequence_map_additional_input_type_invalid:${index}`);
    }
  }

  const firstLength = exactSequenceLength(inputValues[0]);
  row.exact_input_sequence_length = firstLength;
  let partial = firstLength == null;
  if (partial) row.reason_codes.push("sequence_map_input_length_runtime_unknown");
  for (let index = 1; index < inputTypes.length; index += 1) {
    if (inputTypes[index]?.kind !== "sequence") continue;
    const length = exactSequenceLength(inputValues[index]);
    if (firstLength != null && length != null && firstLength !== length) return fail(`sequence_map_sequence_length_mismatch:${index}`);
    if (length == null) {
      partial = true;
      row.reason_codes.push(`sequence_map_additional_sequence_length_runtime_unknown:${index}`);
    }
  }

  const unionBindings = inputTypes.map((type, index) => type.kind === "sequence"
    ? onnxValueDescriptorFromType(type.elementType)
    : descriptor(inputValues[index]));
  const bodyScope = nestedScope(scope, nodeIndex, "body");
  const prepared = prepareGraphTensorMap(body, tensorMap, unionBindings);
  if (prepared.reason_codes.length) {
    row.reason_codes.push(...prepared.reason_codes);
    return fail("sequence_map_body_binding_failed");
  }
  const bodyEvidence = executeGraph(body, prepared.tensorMap, opsets, bodyScope, context, callStack, true);
  row.body_status = bodyEvidence.status;
  appendScopeFailureReasons(row, bodyEvidence, "sequence_map_body");
  if (bodyEvidence.status === "fail") return fail("sequence_map_body_inference_failed");

  const outputElementTypes = [];
  for (const output of body.outputs || []) {
    const type = onnxTypeProtoFromValue(prepared.tensorMap.get(output?.name));
    if (type?.kind !== "tensor" || !onnxTypeProtoKnown(type)) return fail("sequence_map_body_output_not_typed_tensor");
    outputElementTypes.push(type);
  }

  let outputInventories = null;
  const sequenceInputs = inputTypes.map((type, index) => type.kind === "sequence" ? exactSequenceInventory(inputValues[index]) : null);
  const expansionWork = firstLength == null ? null : firstLength * Math.max(1, (body.nodes || []).length);
  const inventoriesAvailable = firstLength != null && inputTypes.every((type, index) => type.kind !== "sequence"
    || Array.isArray(sequenceInputs[index]) && sequenceInputs[index].length === firstLength);
  if (inventoriesAvailable && expansionWork <= MAX_SEQUENCE_MAP_ELEMENT_NODE_EVALUATIONS) {
    outputInventories = outputElementTypes.map(() => []);
    const expansionContext = forkInferenceContext(context);
    for (let elementIndex = 0; elementIndex < firstLength; elementIndex += 1) {
      const bindings = inputTypes.map((type, inputIndex) => type.kind === "sequence"
        ? onnxValueDescriptorFromType(sequenceInputs[inputIndex][elementIndex])
        : descriptor(inputValues[inputIndex]));
      const elementPrepared = prepareGraphTensorMap(body, tensorMap, bindings);
      if (elementPrepared.reason_codes.length) {
        outputInventories = null;
        row.reason_codes.push(`sequence_map_element_binding_failed:${elementIndex}`);
        partial = true;
        break;
      }
      const elementEvidence = executeGraph(body, elementPrepared.tensorMap, opsets, `${bodyScope}/element:${elementIndex}`, expansionContext, callStack, false);
      if (elementEvidence.status === "fail") {
        outputInventories = null;
        row.reason_codes.push(`sequence_map_element_inference_failed:${elementIndex}`);
        partial = true;
        break;
      }
      for (let outputIndex = 0; outputIndex < body.outputs.length; outputIndex += 1) {
        const type = onnxTypeProtoFromValue(elementPrepared.tensorMap.get(body.outputs[outputIndex]?.name));
        if (type?.kind !== "tensor" || !onnxTypeProtoKnown(type)) {
          outputInventories = null;
          row.reason_codes.push(`sequence_map_element_output_unresolved:${elementIndex}:${outputIndex}`);
          partial = true;
          break;
        }
        outputInventories[outputIndex].push(type);
      }
      if (!outputInventories) break;
      row.element_expansion_count += 1;
      row.element_node_evaluation_count += Math.max(1, (body.nodes || []).length);
    }
  } else if (firstLength != null) {
    partial = true;
    row.reason_codes.push(inventoriesAvailable ? "sequence_map_element_expansion_work_limit" : "sequence_map_element_inventory_unavailable");
  }

  const outputs = outputElementTypes.map((elementType, index) => {
    const inventory = outputInventories?.[index] || null;
    const patch = onnxValueDescriptorFromType(makeOnnxSequenceType(elementType), {
      sequenceLengthStatus: firstLength == null ? "not_assessed_runtime_length" : "assessed_exact",
      sequenceLength: firstLength,
      sequenceElementInventoryStatus: inventory ? "assessed_exact" : "not_assessed",
      sequenceElementTypes: inventory ? inventory.map(cloneOnnxTypeProto) : [],
    });
    return [node.outputs[index], patch];
  }).filter(([name]) => Boolean(name));
  row.reason_codes = [...new Set(row.reason_codes.filter(Boolean))];
  row.status = partial ? "partial" : "pass";
  context.sequenceMapRows.push(row);
  return { status: row.status, reason: row.reason_codes[0] || "", result: { outputs } };
}

function inferFunctionCall({ node, nodeIndex, tensorMap, opsets, scope, context, callStack }) {
  const id = functionId(node.domain, node.opType, node.overload);
  const fn = context.functionById.get(id);
  const row = {
    scope,
    node_index: nodeIndex,
    function_id: id,
    input_count: node.inputs?.length || 0,
    output_count: node.outputs?.length || 0,
    status: "fail",
    reason_codes: [],
  };
  if (!fn) row.reason_codes.push("local_function_definition_missing");
  if (callStack.includes(id)) row.reason_codes.push("recursive_local_function_call");
  const binding = fn ? bindFunctionCall(fn, node, context.modelOpsets) : null;
  if (binding?.reason_codes?.length) row.reason_codes.push(...binding.reason_codes);
  if (row.reason_codes.length) return finishFunctionFailure(context, row);

  const bodyGraph = functionBodyGraph(fn, binding.nodes);
  const inputBindings = fn.inputs.map((_, index) => descriptor(tensorMap.get(node.inputs[index])));
  const prepared = prepareGraphTensorMap(bodyGraph, null, inputBindings);
  if (prepared.reason_codes.length) {
    row.reason_codes.push(...prepared.reason_codes);
    return finishFunctionFailure(context, row);
  }
  const functionScope = `function:${id}`;
  const bodyEvidence = executeGraph(bodyGraph, prepared.tensorMap, fn.opsets || [], functionScope, context, [...callStack, id], true, "local_function_body");
  const outputs = mapNamedOutputs(node.outputs, fn.outputs, prepared.tensorMap);
  if (outputs.length !== node.outputs.filter(Boolean).length) row.reason_codes.push("local_function_output_type_unresolved");
  row.body_status = bodyEvidence.status;
  appendScopeFailureReasons(row, bodyEvidence, "function_body");
  row.body_known_output_count = bodyEvidence.known_node_output_count;
  row.body_node_output_count = bodyEvidence.node_output_count;
  row.status = bodyEvidence.status === "fail" || row.reason_codes.length ? "fail" : "pass";
  context.functionCallRows.push(row);
  return row.status === "fail"
    ? extendedFailure(row, row.reason_codes[0] || "local_function_body_shape_inference_failed")
    : { status: outputs.every(([, patch]) => concreteValue(patch)) ? "pass" : "partial", reason: "local_function_output_not_fully_concrete", result: { outputs } };
}

function bindFunctionCall(fn, node, modelOpsets) {
  const reasons = [];
  if (!distinctNonEmpty(fn.inputs)) reasons.push("function_formal_inputs_not_unique_nonempty");
  if (!distinctNonEmpty(fn.outputs)) reasons.push("function_formal_outputs_not_unique_nonempty");
  if ((node.inputs || []).length !== fn.inputs.length || node.inputs.some((name) => !name)) reasons.push("function_call_input_cardinality_mismatch");
  if ((node.outputs || []).length !== fn.outputs.length || node.outputs.some((name) => !name)) reasons.push("function_call_output_cardinality_mismatch");
  if ((node.duplicateAttributeNames || []).length) reasons.push("function_call_duplicate_attribute_name");

  const requiredNames = new Set(fn.attributes || []);
  const defaults = new Map();
  for (const attribute of fn.attributeProtos || []) {
    if (!attribute.name || defaults.has(attribute.name)) reasons.push("function_default_attribute_name_invalid_or_duplicate");
    else defaults.set(attribute.name, attribute);
    const assessment = assessOnnxAttributeProto(attribute);
    if (assessment.status !== "pass") reasons.push(`function_default_attribute_invalid:${attribute.name || "(missing)"}:${assessment.reason}`);
  }
  for (const name of requiredNames) {
    if (!name || defaults.has(name)) reasons.push("function_attribute_required_default_overlap_or_empty");
  }
  const allowed = new Set([...requiredNames, ...defaults.keys()]);
  for (const name of requiredNames) if (!node.attributes?.has(name)) reasons.push(`function_required_attribute_missing:${name}`);
  for (const name of node.attributes?.keys?.() || []) if (!allowed.has(name)) reasons.push(`function_call_attribute_not_declared:${name}`);

  const selected = new Map(defaults);
  for (const [name, attribute] of node.attributes || []) selected.set(name, attribute);
  const resolvedNodes = [];
  for (const bodyNode of fn.nodes || []) resolvedNodes.push(resolveFunctionNode(bodyNode, selected, reasons));
  assessFunctionOpsetCompatibility(fn, modelOpsets, resolvedNodes, reasons);
  return { nodes: resolvedNodes, reason_codes: [...new Set(reasons)] };
}

function resolveFunctionNode(node, selected, reasons) {
  const attributes = new Map();
  for (const [name, attribute] of node.attributes || []) {
    let resolved = attribute;
    if (attribute.refAttrName) {
      const source = selected.get(attribute.refAttrName);
      if (!source) {
        reasons.push(`function_attribute_reference_unbound:${attribute.refAttrName}`);
        continue;
      }
      const sourceAssessment = assessOnnxAttributeProto(source);
      const referenceAssessment = assessOnnxAttributeProto(attribute, { allowReference: true });
      if (sourceAssessment.status !== "pass" || referenceAssessment.status !== "pass" || sourceAssessment.type !== referenceAssessment.type) {
        reasons.push(`function_attribute_reference_type_mismatch:${attribute.refAttrName}`);
        continue;
      }
      resolved = { ...source, name, refAttrName: "" };
    }
    attributes.set(name, resolveNestedGraphAttributes(resolved, selected, reasons));
  }
  return { ...node, attributes };
}

function resolveNestedGraphAttributes(attribute, selected, reasons) {
  const resolveGraph = (graph) => graph ? { ...graph, nodes: (graph.nodes || []).map((node) => resolveFunctionNode(node, selected, reasons)) } : graph;
  return {
    ...attribute,
    graph: resolveGraph(attribute.graph),
    graphs: (attribute.graphs || []).map(resolveGraph),
  };
}

function assessFunctionOpsetCompatibility(fn, modelOpsets, nodes, reasons) {
  const functionImports = validOpsetMap(fn.opsets || []);
  const modelImports = validOpsetMap(modelOpsets || []);
  if (!functionImports || !modelImports) {
    reasons.push("function_or_model_opset_import_contract_invalid");
    return;
  }
  walkNodes(nodes, (node) => {
    const domain = normalizeDomain(node.domain);
    if (domain !== "ai.onnx") return;
    const functionVersion = functionImports.get(domain);
    const modelVersion = modelImports.get(domain);
    if (!functionVersion) {
      reasons.push("function_standard_domain_opset_missing");
      return;
    }
    if (!modelVersion) {
      reasons.push("model_standard_domain_opset_missing");
      return;
    }
    const functionSchema = resolveOnnxSchemaSinceVersion(node.opType, functionVersion);
    const modelSchema = resolveOnnxSchemaSinceVersion(node.opType, modelVersion);
    if (functionSchema != null && modelSchema != null && functionSchema !== modelSchema) {
      reasons.push(`function_model_opset_schema_incompatible:${node.opType}:${functionSchema ?? "none"}:${modelSchema ?? "none"}`);
    }
  });
}

function inferIf(args) {
  const { node, nodeIndex, tensorMap, opsets, scope, context, callStack } = args;
  const thenGraph = node.attributes?.get("then_branch")?.graph;
  const elseGraph = node.attributes?.get("else_branch")?.graph;
  const row = controlRow(args, "If");
  if (!thenGraph || !elseGraph) return finishControlFailure(context, row, "if_branch_graph_missing");
  const thenScope = nestedScope(scope, nodeIndex, "then_branch");
  const elseScope = nestedScope(scope, nodeIndex, "else_branch");
  const thenPrepared = prepareGraphTensorMap(thenGraph, tensorMap, []);
  const elsePrepared = prepareGraphTensorMap(elseGraph, tensorMap, []);
  if (thenPrepared.reason_codes.length || elsePrepared.reason_codes.length) {
    row.reason_codes.push(...thenPrepared.reason_codes, ...elsePrepared.reason_codes);
    return finishControlFailure(context, row);
  }
  const selectedCondition = exactSingletonBoolean(descriptor(tensorMap.get(node.inputs?.[0])));
  if (selectedCondition != null) {
    const selectedGraph = selectedCondition ? thenGraph : elseGraph;
    const selectedPrepared = selectedCondition ? thenPrepared : elsePrepared;
    const selectedScope = selectedCondition ? thenScope : elseScope;
    const unselectedGraph = selectedCondition ? elseGraph : thenGraph;
    const unselectedPrepared = selectedCondition ? elsePrepared : thenPrepared;
    const unselectedScope = selectedCondition ? elseScope : thenScope;
    const selectedEvidence = executeGraph(selectedGraph, selectedPrepared.tensorMap, opsets, selectedScope, context, callStack, true);
    const validationContext = forkInferenceContext(context);
    const unselectedEvidence = executeGraph(unselectedGraph, unselectedPrepared.tensorMap, opsets, unselectedScope, validationContext, callStack, true);
    if (thenGraph.outputs.length !== elseGraph.outputs.length || selectedGraph.outputs.length !== node.outputs.length) {
      return finishControlFailure(context, row, "if_branch_output_cardinality_mismatch");
    }
    const outputs = mapNamedOutputs(node.outputs, selectedGraph.outputs.map((output) => output.name), selectedPrepared.tensorMap);
    if (outputs.length !== node.outputs.filter(Boolean).length) row.reason_codes.push("if_selected_branch_output_type_unresolved");
    row.condition_status = "assessed_static_single_bool";
    row.selected_branch = selectedCondition ? "then_branch" : "else_branch";
    row.branch_statuses = selectedCondition
      ? [selectedEvidence.status, unselectedEvidence.status]
      : [unselectedEvidence.status, selectedEvidence.status];
    if (selectedEvidence.status === "fail") {
      row.reason_codes.push(`if_selected_branch_inference_failed:${firstGraphInferenceFailureReason(selectedEvidence) || "unknown"}`);
    }
    if (unselectedEvidence.status === "fail") {
      row.reason_codes.push(`if_unselected_branch_inference_failed:${firstGraphInferenceFailureReason(unselectedEvidence) || "unknown"}`);
    }
    appendScopeFailureReasons(row, selectedEvidence, `if_${row.selected_branch}`);
    appendScopeFailureReasons(row, unselectedEvidence, selectedCondition ? "if_else_branch" : "if_then_branch");
    const unselectedStructuralFailure = graphHasUnconditionalStructuralFailure(unselectedEvidence);
    row.status = selectedEvidence.status === "fail" || unselectedStructuralFailure
      ? "fail"
      : selectedEvidence.status === "assessed" && unselectedEvidence.status === "assessed"
        && outputs.every(([, patch]) => concreteValue(patch)) ? "pass" : "partial";
    context.controlFlowRows.push(row);
    return row.status === "fail"
      ? extendedFailure(row, row.reason_codes[0] || "if_branch_shape_inference_failed")
      : { status: row.status, reason: "if_selected_branch_output_not_fully_concrete", result: { outputs } };
  }
  const thenEvidence = executeGraph(thenGraph, thenPrepared.tensorMap, opsets, thenScope, context, callStack, true);
  const elseEvidence = executeGraph(elseGraph, elsePrepared.tensorMap, opsets, elseScope, context, callStack, true);
  if (thenGraph.outputs.length !== elseGraph.outputs.length || thenGraph.outputs.length !== node.outputs.length) {
    return finishControlFailure(context, row, "if_branch_output_cardinality_mismatch");
  }
  const outputs = [];
  for (let index = 0; index < node.outputs.length; index += 1) {
    const merged = unionBranchValues(
      thenPrepared.tensorMap.get(thenGraph.outputs[index]?.name),
      elsePrepared.tensorMap.get(elseGraph.outputs[index]?.name),
      {
        key: `if:${scope}:node:${nodeIndex}:condition:${encodeURIComponent(String(node.inputs?.[0] || "anonymous"))}`,
        thenValue: "then_branch",
        elseValue: "else_branch",
      },
    );
    if (merged.status === "fail") return finishControlFailure(context, row, `if_branch_output_type_conflict:${index}`);
    if (node.outputs[index] && merged.patch) outputs.push([node.outputs[index], merged.patch]);
  }
  row.branch_statuses = [thenEvidence.status, elseEvidence.status];
  appendScopeFailureReasons(row, thenEvidence, "if_then_branch");
  appendScopeFailureReasons(row, elseEvidence, "if_else_branch");
  row.status = thenEvidence.status === "fail" || elseEvidence.status === "fail" ? "fail"
    : outputs.every(([, patch]) => concreteValue(patch)) ? "pass" : "partial";
  context.controlFlowRows.push(row);
  return row.status === "fail"
    ? extendedFailure(row, "if_branch_shape_inference_failed")
    : { status: row.status, reason: "if_branch_union_not_fully_concrete", result: { outputs } };
}

function inferLoop(args) {
  const { node, nodeIndex, tensorMap, importedOpset, opsets, scope, context, callStack } = args;
  const row = controlRow(args, "Loop");
  const body = node.attributes?.get("body")?.graph;
  const stateCount = Math.max(0, node.inputs.length - 2);
  const stateInputs = node.inputs.slice(2).map((name) => descriptor(tensorMap.get(name)));
  row.body_node_count = (body?.nodes || []).length;
  row.state_variable_count = stateCount;
  row.scan_output_count = Math.max(0, node.outputs.length - stateCount);
  row.state_value_kinds = stateInputs.map((value) => onnxTypeProtoFromValue(value)?.kind || "unresolved");
  row.non_dense_state_variable_count = row.state_value_kinds.filter((kind) => kind !== "tensor" && kind !== "unresolved").length;
  row.exact_expansion_status = "not_assessed";
  row.exact_iteration_count = null;
  row.exact_body_node_evaluation_count = 0;
  row.exact_final_state_contracts = [];
  if (!body) return finishControlFailure(context, row, "loop_body_graph_missing");
  if (node.outputs.length < stateCount) return finishControlFailure(context, row, "loop_output_count_below_state_count");
  for (let index = 0; index < stateInputs.length; index += 1) {
    const support = assessLoopStateType(stateInputs[index], importedOpset);
    if (support.status === "fail") return finishControlFailure(context, row, `${support.reason}:${index}`);
    if (support.status === "partial") row.reason_codes.push(`${support.reason}:${index}`);
  }
  const bindings = [
    tensorPatch("INT64", [], true),
    descriptor(tensorMap.get(node.inputs[1])) || tensorPatch("BOOL", [], true),
    ...stateInputs.map(clearLoopStateDescriptor),
  ];
  const prepared = prepareGraphTensorMap(body, tensorMap, bindings);
  if (prepared.reason_codes.length) {
    row.reason_codes.push(...prepared.reason_codes);
    return finishControlFailure(context, row);
  }
  const bodyScope = nestedScope(scope, nodeIndex, "body");
  const bodyEvidence = executeGraph(body, prepared.tensorMap, opsets, bodyScope, context, callStack, true);
  if (body.outputs.length !== node.outputs.length + 1) return finishControlFailure(context, row, "loop_body_output_cardinality_mismatch");
  let outputs = [];
  for (let index = 0; index < node.outputs.length; index += 1) {
    const bodyValue = descriptor(prepared.tensorMap.get(body.outputs[index + 1]?.name));
    let patch = null;
    if (index < stateCount) {
      const merged = mergeLoopStateContracts(stateInputs[index], bodyValue, importedOpset);
      if (merged.status === "fail") return finishControlFailure(context, row, `loop_state_type_conflict:${index}`);
      patch = merged.patch;
    } else {
      if (!denseTensorDescriptor(bodyValue)) return finishControlFailure(context, row, `loop_scan_output_non_tensor_value:${index - stateCount}`);
      patch = bodyValue.shapeDeclared ? denseShapePatch(bodyValue, [-1, ...bodyValue.shape]) : clearStatic(bodyValue);
    }
    if (node.outputs[index] && patch) outputs.push([node.outputs[index], patch]);
  }
  const exact = evaluateExactLoop({ node, body, stateCount, stateInputs, tensorMap, opsets, bodyScope, context, callStack, importedOpset, genericBodyMap: prepared.tensorMap });
  row.exact_expansion_status = exact.status;
  row.exact_iteration_count = exact.iteration_count;
  row.exact_body_node_evaluation_count = exact.body_node_evaluation_count;
  row.exact_iteration_state_contracts = exact.iteration_state_contracts || [];
  row.exact_nested_failure_rows = exact.nested_failure_rows || [];
  row.exact_final_state_contracts = exact.outputs.slice(0, stateCount).map(([name, patch], index) => ({
    state_index: index,
    output_name: name,
    ...valueContractSummary(patch),
  }));
  row.reason_codes.push(...exact.reason_codes);
  if (exact.status === "fail") {
    row.deterministic_contract_failure = true;
    return finishControlFailure(context, row, exact.reason_codes[0] || "loop_exact_expansion_failed");
  }
  if (exact.status === "assessed") outputs = exact.outputs;
  row.body_status = bodyEvidence.status;
  appendScopeFailureReasons(row, bodyEvidence, "loop_body");
  row.status = bodyEvidence.status === "fail" ? "fail"
    : outputs.length !== node.outputs.filter(Boolean).length ? "fail"
      : exact.status === "assessed" && outputs.every(([, patch]) => concreteValue(patch)) ? "pass" : "partial";
  if (row.status === "fail") row.reason_codes.push("loop_body_shape_inference_failed_or_output_unresolved");
  row.reason_codes = [...new Set(row.reason_codes)];
  context.controlFlowRows.push(row);
  return row.status === "fail"
    ? extendedFailure(row, row.reason_codes[0])
    : { status: row.status, reason: row.status === "pass" ? "" : "loop_state_or_iteration_dimension_not_statically_fixed", result: { outputs } };
}

function assessLoopStateType(value, importedOpset) {
  const type = onnxTypeProtoFromValue(value);
  if (!type) return { status: "partial", reason: "loop_state_type_unresolved" };
  if (type.kind === "tensor") {
    return onnxTypeProtoKnown(type)
      ? { status: "pass", reason: "" }
      : { status: "partial", reason: "loop_state_tensor_element_type_unresolved" };
  }
  if (type.kind === "sequence") {
    if (Number(importedOpset || 0) < 13) return { status: "fail", reason: "loop_sequence_state_requires_opset_13" };
    if (type.elementType?.kind !== "tensor") return { status: "fail", reason: "loop_sequence_state_element_not_tensor" };
    return onnxTypeProtoKnown(type)
      ? { status: "pass", reason: "" }
      : { status: "partial", reason: "loop_sequence_state_element_type_unresolved" };
  }
  if (type.kind === "optional") {
    if (Number(importedOpset || 0) < 16) return { status: "fail", reason: "loop_optional_state_requires_opset_16" };
    const child = type.elementType;
    if (child?.kind !== "tensor" && !(child?.kind === "sequence" && child.elementType?.kind === "tensor")) {
      return { status: "fail", reason: "loop_optional_state_element_not_tensor_or_tensor_sequence" };
    }
    return onnxTypeProtoKnown(type)
      ? { status: "pass", reason: "" }
      : { status: "partial", reason: "loop_optional_state_element_type_unresolved" };
  }
  return { status: "fail", reason: `loop_state_value_kind_not_supported:${type.kind || "undefined"}` };
}

function clearLoopStateDescriptor(value) {
  const type = clearTypeProtoShapes(onnxTypeProtoFromValue(value));
  if (!type) return null;
  const state = {};
  if (type.kind === "sequence") {
    state.sequenceLengthStatus = "not_assessed_loop_variant";
    state.sequenceLength = null;
    state.sequenceElementInventoryStatus = "not_assessed_loop_variant";
    state.sequenceElementTypes = [];
  } else if (type.kind === "optional") {
    state.optionalPresenceStatus = "not_assessed_loop_variant";
    state.optionalPresence = null;
  }
  return onnxValueDescriptorFromType(type, state);
}

function clearTypeProtoShapes(type) {
  if (!type) return null;
  if (type.kind === "tensor") return makeOnnxTensorType(type.dtype || type.elementTypeName || "UNKNOWN", [], false);
  if (type.kind === "sequence") return makeOnnxSequenceType(clearTypeProtoShapes(type.elementType));
  if (type.kind === "optional") return makeOnnxOptionalType(clearTypeProtoShapes(type.elementType));
  return cloneOnnxTypeProto(type);
}

function mergeLoopStateContracts(initialValue, bodyValue, importedOpset) {
  const initialSupport = assessLoopStateType(initialValue, importedOpset);
  const bodySupport = assessLoopStateType(bodyValue, importedOpset);
  if (initialSupport.status === "fail" || bodySupport.status === "fail") return { status: "fail", patch: null };
  const merged = mergeLoopTypesWithoutShape(onnxTypeProtoFromValue(initialValue), onnxTypeProtoFromValue(bodyValue));
  if (merged.status !== "pass") return { status: "fail", patch: null };
  return { status: initialSupport.status === "partial" || bodySupport.status === "partial" ? "partial" : "pass", patch: clearLoopStateDescriptor(onnxValueDescriptorFromType(merged.type)) };
}

function mergeLoopTypesWithoutShape(left, right) {
  if (!left || !right) return { status: "unresolved", type: null };
  if (left.kind !== right.kind) return { status: "fail", type: null };
  if (left.kind === "tensor") {
    const leftDtype = left.dtype || left.elementTypeName || "UNKNOWN";
    const rightDtype = right.dtype || right.elementTypeName || "UNKNOWN";
    if (leftDtype !== "UNKNOWN" && rightDtype !== "UNKNOWN" && leftDtype !== rightDtype) return { status: "fail", type: null };
    return { status: "pass", type: makeOnnxTensorType(leftDtype !== "UNKNOWN" ? leftDtype : rightDtype, [], false) };
  }
  if (left.kind === "sequence" || left.kind === "optional") {
    const child = mergeLoopTypesWithoutShape(left.elementType, right.elementType);
    if (child.status !== "pass") return child;
    return { status: "pass", type: left.kind === "sequence" ? makeOnnxSequenceType(child.type) : makeOnnxOptionalType(child.type) };
  }
  return { status: "fail", type: null };
}

function evaluateExactLoop({ node, body, stateCount, stateInputs, tensorMap, opsets, bodyScope, context, callStack, importedOpset, genericBodyMap }) {
  const reasons = [];
  const tripInputName = node.inputs?.[0] || "";
  const conditionInputName = node.inputs?.[1] || "";
  const tripCount = tripInputName ? exactScalarInteger(descriptor(tensorMap.get(tripInputName))) : null;
  let condition = conditionInputName ? exactScalarBoolean(descriptor(tensorMap.get(conditionInputName))) : true;
  if (condition === false || tripCount != null && tripCount <= 0) {
    return exactLoopOutputs(node, body, stateCount, stateInputs, genericBodyMap, 0, 0, reasons);
  }
  if (tripInputName && tripCount == null) reasons.push("loop_trip_count_runtime_unknown");
  if (conditionInputName && condition == null) reasons.push("loop_initial_condition_runtime_unknown");
  if (condition == null || tripCount == null) {
    return { status: "partial", outputs: [], iteration_count: null, body_node_evaluation_count: 0, reason_codes: reasons };
  }
  const maximumIterations = Math.max(0, tripCount);
  const bodyNodeCount = (body.nodes || []).length;
  const workIterationLimit = bodyNodeCount ? Math.floor(MAX_LOOP_BODY_NODE_EVALUATIONS / bodyNodeCount) : MAX_LOOP_EXACT_ITERATIONS;
  const expansionLimit = Math.min(maximumIterations, MAX_LOOP_EXACT_ITERATIONS, workIterationLimit);

  const expansionContext = forkInferenceContext(context);
  let states = stateInputs.map(descriptor);
  const scanTypes = Array.from({ length: Math.max(0, node.outputs.length - stateCount) }, () => []);
  const iterationStateContracts = [];
  let iterations = 0;
  for (let iteration = 0; iteration < expansionLimit && condition; iteration += 1) {
    const bindings = [staticScalarDescriptor("INT64", iteration, "LoopIteration"), staticScalarDescriptor("BOOL", condition ? 1 : 0, "LoopCondition"), ...states.map(descriptor)];
    const prepared = prepareGraphTensorMap(body, tensorMap, bindings);
    if (prepared.reason_codes.length) {
      return exactLoopPartial(iterations, bodyNodeCount, [...reasons, `loop_exact_body_binding_failed:${iteration}`], iterations, iterationStateContracts);
    }
    const evidence = executeGraph(body, prepared.tensorMap, opsets, `${bodyScope}/iteration:${iteration}`, expansionContext, callStack, false);
    if (evidence.status === "fail") {
      const nestedFailureRows = contextFailureRows(expansionContext);
      const failureReason = firstContextInferenceFailureReason(expansionContext)
        || firstGraphInferenceFailureReason(evidence);
      const failureReasons = [
        ...reasons,
        `loop_exact_body_inference_failed:${iteration}`,
        ...(failureReason ? [`loop_exact_body_failure_reason:${iteration}:${failureReason}`] : []),
      ];
      return graphHasDeterministicContractFailure(evidence, nestedFailureRows)
        ? exactLoopFailure(iterations, bodyNodeCount, failureReasons, iterations + 1, iterationStateContracts, nestedFailureRows)
        : exactLoopPartial(iterations, bodyNodeCount, failureReasons, iterations + 1, iterationStateContracts, nestedFailureRows);
    }
    const nextCondition = exactScalarBoolean(descriptor(prepared.tensorMap.get(body.outputs[0]?.name)));
    const nextStates = [];
    for (let index = 0; index < stateCount; index += 1) {
      const next = descriptor(prepared.tensorMap.get(body.outputs[index + 1]?.name));
      if (mergeLoopStateContracts(stateInputs[index], next, importedOpset).status === "fail") {
        return { status: "fail", outputs: [], iteration_count: iterations, body_node_evaluation_count: (iterations + 1) * bodyNodeCount, reason_codes: [`loop_exact_state_type_conflict:${iteration}:${index}`], iteration_state_contracts: iterationStateContracts };
      }
      nextStates.push(next);
    }
    iterationStateContracts.push({
      iteration,
      states: nextStates.map((state, stateIndex) => ({ state_index: stateIndex, ...valueContractSummary(state) })),
    });
    for (let scanIndex = 0; scanIndex < scanTypes.length; scanIndex += 1) {
      const value = descriptor(prepared.tensorMap.get(body.outputs[stateCount + 1 + scanIndex]?.name));
      const type = onnxTypeProtoFromValue(value);
      if (type?.kind !== "tensor" || !onnxTypeProtoKnown(type)) {
        return { status: "fail", outputs: [], iteration_count: iterations, body_node_evaluation_count: (iterations + 1) * bodyNodeCount, reason_codes: [`loop_exact_scan_output_non_tensor_or_untyped:${iteration}:${scanIndex}`], iteration_state_contracts: iterationStateContracts };
      }
      if (type.shapeDeclared !== true) {
        return exactLoopPartial(iterations, bodyNodeCount, [...reasons, `loop_exact_scan_output_shape_unresolved:${iteration}:${scanIndex}`], iterations + 1, iterationStateContracts);
      }
      const previous = scanTypes[scanIndex][0];
      if (previous && !sameExactTensorType(previous, type)) {
        return { status: "fail", outputs: [], iteration_count: iterations, body_node_evaluation_count: (iterations + 1) * bodyNodeCount, reason_codes: [`loop_exact_scan_output_type_or_shape_changed:${iteration}:${scanIndex}`], iteration_state_contracts: iterationStateContracts };
      }
      scanTypes[scanIndex].push(type);
    }
    states = nextStates;
    iterations += 1;
    condition = nextCondition;
    if (condition == null && iterations < maximumIterations) {
      return exactLoopPartial(iterations, bodyNodeCount, [...reasons, `loop_body_condition_runtime_unknown_after_iteration:${iterations - 1}`], iterations, iterationStateContracts);
    }
  }
  if (condition && iterations < maximumIterations) {
    if (iterations >= MAX_LOOP_EXACT_ITERATIONS) reasons.push("loop_exact_iteration_limit_exceeded");
    if (bodyNodeCount && (iterations + 1) * bodyNodeCount > MAX_LOOP_BODY_NODE_EVALUATIONS) reasons.push("loop_exact_body_node_evaluation_limit_exceeded");
    return exactLoopPartial(iterations, bodyNodeCount, reasons, iterations, iterationStateContracts);
  }
  return exactLoopOutputs(node, body, stateCount, states, genericBodyMap, iterations, iterations * bodyNodeCount, reasons, scanTypes, iterationStateContracts);
}

function firstGraphInferenceFailureReason(evidence) {
  const declaration = (evidence?.declaration_conflicts || [])[0];
  if (declaration) return `${declaration.op_name || "UNKNOWN"}:declared_${declaration.field || "contract"}_conflict`;
  const semantic = (evidence?.semantic_contract_conflicts || [])[0];
  if (semantic?.reason) return `${semantic.op_name || "UNKNOWN"}:${semantic.reason}`;
  const row = (evidence?.rule_unresolved_nodes || [])[0];
  if (row?.reason) return `${row.op_name || "UNKNOWN"}:${row.reason}`;
  const extended = evidence?.extended_scope_inference;
  const failedControl = (extended?.control_flow_rows || []).find((item) => item.status === "fail");
  if (failedControl) return `${failedControl.op_name || "CONTROL_FLOW"}:${failedControl.reason_codes?.[0] || "failed"}`;
  return "";
}

function firstContextInferenceFailureReason(context) {
  const controlRows = [...(context?.controlFlowRows || [])].reverse();
  for (const row of controlRows) {
    const semantic = (row.scope_failure_details || []).find((detail) => detail.field === "semantic_contract" && detail.reason_codes?.[0]);
    if (semantic) return `${semantic.op_name || row.op_name || "UNKNOWN"}:${semantic.reason_codes[0]}`;
  }
  const failedControl = controlRows.find((item) => item.status === "fail" && item.reason_codes?.length)
    || controlRows.find((item) => item.status === "fail");
  if (failedControl) return `${failedControl.op_name || "CONTROL_FLOW"}:${failedControl.reason_codes?.[0] || "failed"}`;
  const failedFunction = [...(context?.functionCallRows || [])].reverse().find((item) => item.status === "fail");
  if (failedFunction) return `${failedFunction.function_id || "FUNCTION"}:${failedFunction.reason_codes?.[0] || "failed"}`;
  const failedSequenceMap = [...(context?.sequenceMapRows || [])].reverse().find((item) => item.status === "fail");
  if (failedSequenceMap) return `SequenceMap:${failedSequenceMap.reason_codes?.[0] || "failed"}`;
  return "";
}

function valueContractSummary(value) {
  const type = onnxTypeProtoFromValue(value);
  const summary = {
    value_kind: type?.kind || "unresolved",
    dtype: type?.kind === "tensor" ? type.dtype || type.elementTypeName || "UNKNOWN" : "UNKNOWN",
    shape_declared: type?.kind === "tensor" ? type.shapeDeclared === true : false,
    shape: type?.kind === "tensor" ? [...(type.shape || [])] : [],
    static_values_complete: value?.staticValuesComplete === true,
    static_values: value?.staticValuesComplete === true ? structuredClone(value.staticValues || []) : [],
    sequence_length_status: value?.sequenceLengthStatus || "not_applicable",
    sequence_length: value?.sequenceLength ?? null,
    sequence_element_inventory_status: value?.sequenceElementInventoryStatus || "not_applicable",
    sequence_element_type_count: Array.isArray(value?.sequenceElementTypes) ? value.sequenceElementTypes.length : 0,
    sequence_element_types: Array.isArray(value?.sequenceElementTypes)
      ? value.sequenceElementTypes.map(canonicalOnnxTypeProto) : [],
  };
  return summary;
}

function graphHasUnconditionalStructuralFailure(evidence) {
  return evidence?.opset_import_contract?.status === "fail"
    || Number(evidence?.schema_form_invalid_node_count || 0) > 0
    || Number(evidence?.declaration_conflict_count || 0) > 0
    || evidence?.shape_scope?.registry_status === "fail";
}

function exactLoopPartial(iterations, bodyNodeCount, reasons, evaluatedIterations = iterations, iterationStateContracts = [], nestedFailureRows = []) {
  return {
    status: "partial",
    outputs: [],
    iteration_count: iterations,
    body_node_evaluation_count: evaluatedIterations * bodyNodeCount,
    reason_codes: [...new Set(reasons)],
    iteration_state_contracts: iterationStateContracts,
    nested_failure_rows: nestedFailureRows,
  };
}

function exactLoopFailure(iterations, bodyNodeCount, reasons, evaluatedIterations, iterationStateContracts, nestedFailureRows) {
  return {
    ...exactLoopPartial(iterations, bodyNodeCount, reasons, evaluatedIterations, iterationStateContracts, nestedFailureRows),
    status: "fail",
  };
}

function graphHasDeterministicContractFailure(evidence, nestedFailureRows) {
  if (evidence?.opset_import_contract?.status === "fail"
    || Number(evidence?.schema_form_invalid_node_count || 0) > 0
    || Number(evidence?.declaration_conflict_count || 0) > 0
    || Number(evidence?.semantic_contract_conflict_count || 0) > 0
    || evidence?.shape_scope?.registry_status === "fail") return true;
  return (nestedFailureRows || []).some((row) => (row.scope_failure_details || []).some((detail) => [
    "semantic_contract", "schema_form",
  ].includes(detail.field)) || (row.reason_codes || []).some((reason) => /selected_branch_inference_failed:.*(?:mismatch|invalid|conflict)/.test(reason)));
}

function contextFailureRows(context) {
  return [
    ...(context?.controlFlowRows || []).filter((row) => row.status === "fail").map((row) => ({
      scope: row.scope,
      node_index: row.node_index,
      op_name: row.op_name,
      reason_codes: [...(row.reason_codes || [])],
      scope_failure_details: structuredClone(row.scope_failure_details || []),
    })),
    ...(context?.functionCallRows || []).filter((row) => row.status === "fail").map((row) => ({
      scope: row.scope,
      node_index: row.node_index,
      op_name: row.function_id || "FunctionProto",
      reason_codes: [...(row.reason_codes || [])],
    })),
    ...(context?.sequenceMapRows || []).filter((row) => row.status === "fail").map((row) => ({
      scope: row.scope,
      node_index: row.node_index,
      op_name: "SequenceMap",
      reason_codes: [...(row.reason_codes || [])],
    })),
  ];
}

function exactLoopOutputs(node, body, stateCount, states, genericBodyMap, iterations, work, reasons, scanTypes = null, iterationStateContracts = []) {
  const outputs = [];
  for (let index = 0; index < stateCount; index += 1) {
    const patch = descriptor(states[index]);
    if (!patch) return { status: "partial", outputs: [], iteration_count: iterations, body_node_evaluation_count: work, reason_codes: [...reasons, `loop_exact_final_state_unresolved:${index}`], iteration_state_contracts: iterationStateContracts };
    if (node.outputs[index]) outputs.push([node.outputs[index], patch]);
  }
  for (let scanIndex = 0; scanIndex < node.outputs.length - stateCount; scanIndex += 1) {
    const samples = scanTypes?.[scanIndex] || [];
    const bodyValue = descriptor(genericBodyMap.get(body.outputs[stateCount + 1 + scanIndex]?.name));
    const type = samples[0] || onnxTypeProtoFromValue(bodyValue);
    if (type?.kind !== "tensor" || !onnxTypeProtoKnown(type)) {
      return { status: "partial", outputs: [], iteration_count: iterations, body_node_evaluation_count: work, reason_codes: [...reasons, `loop_exact_scan_output_contract_unresolved:${scanIndex}`], iteration_state_contracts: iterationStateContracts };
    }
    const patch = type.shapeDeclared === true
      ? onnxValueDescriptorFromType(makeOnnxTensorType(type.dtype || type.elementTypeName, [iterations, ...type.shape], true))
      : onnxValueDescriptorFromType(makeOnnxTensorType(type.dtype || type.elementTypeName));
    if (node.outputs[stateCount + scanIndex]) outputs.push([node.outputs[stateCount + scanIndex], patch]);
  }
  return { status: "assessed", outputs, iteration_count: iterations, body_node_evaluation_count: work, reason_codes: [...new Set(reasons)], iteration_state_contracts: iterationStateContracts };
}

function exactScalarInteger(value) {
  if (knownDtype(value) !== "INT64") return null;
  const values = exactScalarValues(value);
  return values && Number.isSafeInteger(Number(values[0])) ? Number(values[0]) : null;
}

function exactScalarBoolean(value) {
  if (knownDtype(value) !== "BOOL") return null;
  const values = exactScalarValues(value);
  if (!values) return null;
  const item = values[0];
  if (typeof item === "boolean") return item;
  if (Number(item) === 0) return false;
  if (Number(item) === 1) return true;
  return null;
}

function exactSingletonBoolean(value) {
  if (knownDtype(value) !== "BOOL" || value?.shapeDeclared !== true || value.staticValuesComplete !== true
    || !Array.isArray(value.staticValues) || value.staticValues.length !== 1
    || value.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)
    || value.shape.reduce((product, dimension) => product * dimension, 1) !== 1) return null;
  const item = value.staticValues[0];
  if (typeof item === "boolean") return item;
  if (Number(item) === 0) return false;
  if (Number(item) === 1) return true;
  return null;
}

function exactScalarValues(value) {
  return value?.shapeDeclared === true && value.shape.length === 0 && value.staticValuesComplete === true
    && Array.isArray(value.staticValues) && value.staticValues.length === 1 ? value.staticValues : null;
}

function staticScalarDescriptor(dtype, value, source) {
  return {
    ...tensorPatch(dtype, [], true),
    staticValuesStatus: "assessed_exact_static_data",
    staticValuesComplete: true,
    staticValues: [value],
    staticValuesSource: source,
  };
}

function sameExactTensorType(left, right) {
  const leftDtype = left?.dtype || left?.elementTypeName || "UNKNOWN";
  const rightDtype = right?.dtype || right?.elementTypeName || "UNKNOWN";
  return leftDtype === rightDtype && left?.shapeDeclared === true && right?.shapeDeclared === true
    && JSON.stringify(left.shape) === JSON.stringify(right.shape);
}

function forkInferenceContext(context) {
  return {
    ...context,
    scopeExecution: { rows: new Map(), intrinsicCostVariantCount: 0, intrinsicCostVariantOverflowCount: 0 },
    functionCallRows: [],
    controlFlowRows: [],
    sequenceMapRows: [],
  };
}

function inferScan(args) {
  const schemaVersion = Number(args.importedOpset || 0);
  return schemaVersion === 8 ? inferScan8(args) : inferScan9Plus(args);
}

function inferScan9Plus(args) {
  const { node, nodeIndex, tensorMap, opsets, scope, context, callStack } = args;
  const row = controlRow(args, "Scan");
  const body = node.attributes?.get("body")?.graph;
  const scanCount = attributeInteger(node, "num_scan_inputs");
  if (!body || !Number.isSafeInteger(scanCount) || scanCount < 1 || scanCount > node.inputs.length) {
    return finishControlFailure(context, row, "scan_body_or_num_scan_inputs_invalid");
  }
  const stateCount = node.inputs.length - scanCount;
  const scanOutputCount = node.outputs.length - stateCount;
  if (scanOutputCount < 0) return finishControlFailure(context, row, "scan_output_count_below_state_count");
  const inputAxes = attributeIntegers(node, "scan_input_axes", scanCount, 0);
  const outputAxes = attributeIntegers(node, "scan_output_axes", scanOutputCount, 0);
  if (!inputAxes || !outputAxes) return finishControlFailure(context, row, "scan_axis_attribute_cardinality_invalid");
  const bindings = [];
  let sequenceDimension = -1;
  for (let index = 0; index < node.inputs.length; index += 1) {
    const inputTensor = descriptor(tensorMap.get(node.inputs[index]));
    if (index < stateCount) {
      if (!denseTensorDescriptor(inputTensor)) return finishControlFailure(context, row, `scan_state_input_non_tensor_value:${index}`);
      bindings.push(inputTensor);
      continue;
    }
    const removed = removeAxis(inputTensor, inputAxes[index - stateCount]);
    if (removed.status === "fail") return finishControlFailure(context, row, `scan_input_axis_out_of_range:${index - stateCount}`);
    sequenceDimension = mergeDimension(sequenceDimension, removed.dimension);
    if (sequenceDimension == null) return finishControlFailure(context, row, "scan_input_sequence_dimension_conflict");
    bindings.push(removed.patch);
  }
  const prepared = prepareGraphTensorMap(body, tensorMap, bindings);
  if (prepared.reason_codes.length) {
    row.reason_codes.push(...prepared.reason_codes);
    return finishControlFailure(context, row);
  }
  const bodyScope = nestedScope(scope, nodeIndex, "body");
  const bodyEvidence = executeGraph(body, prepared.tensorMap, opsets, bodyScope, context, callStack, true);
  if (body.outputs.length !== node.outputs.length) return finishControlFailure(context, row, "scan_body_output_cardinality_mismatch");
  const outputs = [];
  for (let index = 0; index < node.outputs.length; index += 1) {
    const bodyTensor = descriptor(prepared.tensorMap.get(body.outputs[index]?.name));
    let patch = bodyTensor;
    if (index < stateCount) patch = mergeTensorContracts(descriptor(tensorMap.get(node.inputs[index])), bodyTensor).patch;
    else patch = insertAxis(bodyTensor, outputAxes[index - stateCount], sequenceDimension);
    if (!patch) return finishControlFailure(context, row, `scan_output_type_unresolved:${index}`);
    if (node.outputs[index]) outputs.push([node.outputs[index], clearStatic(patch)]);
  }
  row.body_status = bodyEvidence.status;
  appendScopeFailureReasons(row, bodyEvidence, "scan_body");
  row.state_variable_count = stateCount;
  row.scan_input_count = scanCount;
  row.scan_output_count = scanOutputCount;
  row.sequence_dimension = sequenceDimension;
  row.status = bodyEvidence.status === "fail" ? "fail" : outputs.every(([, patch]) => concreteTensor(patch)) ? "pass" : "partial";
  context.controlFlowRows.push(row);
  return row.status === "fail"
    ? extendedFailure(row, "scan_body_shape_inference_failed")
    : { status: row.status, reason: "scan_output_not_fully_concrete", result: { outputs } };
}

function inferScan8(args) {
  const { node, nodeIndex, tensorMap, opsets, scope, context, callStack } = args;
  const row = controlRow(args, "Scan-8");
  const body = node.attributes?.get("body")?.graph;
  const scanCount = attributeInteger(node, "num_scan_inputs");
  if (!body || !Number.isSafeInteger(scanCount) || scanCount < 1 || scanCount >= node.inputs.length) {
    return finishControlFailure(context, row, "scan8_body_or_num_scan_inputs_invalid");
  }
  const stateCount = node.inputs.length - 1 - scanCount;
  const scanOutputCount = node.outputs.length - stateCount;
  if (stateCount < 0 || scanOutputCount < 0) return finishControlFailure(context, row, "scan8_state_or_output_count_invalid");
  const bindings = [];
  let batchDimension = -1;
  let sequenceDimension = -1;
  for (let index = 1; index < node.inputs.length; index += 1) {
    const inputTensor = descriptor(tensorMap.get(node.inputs[index]));
    if (index - 1 < stateCount) {
      if (!denseTensorDescriptor(inputTensor)) return finishControlFailure(context, row, `scan8_state_input_non_tensor_value:${index - 1}`);
      const removed = removeLeadingDimensions(inputTensor, 1);
      if (!removed) return finishControlFailure(context, row, "scan8_state_rank_unknown_or_zero");
      batchDimension = mergeDimension(batchDimension, inputTensor.shape[0]);
      if (batchDimension == null) return finishControlFailure(context, row, "scan8_batch_dimension_conflict");
      bindings.push(removed);
    } else {
      const removed = removeLeadingDimensions(inputTensor, 2);
      if (!removed) return finishControlFailure(context, row, "scan8_input_rank_below_two");
      batchDimension = mergeDimension(batchDimension, inputTensor.shape[0]);
      sequenceDimension = mergeDimension(sequenceDimension, inputTensor.shape[1]);
      if (batchDimension == null || sequenceDimension == null) return finishControlFailure(context, row, "scan8_batch_or_sequence_dimension_conflict");
      bindings.push(removed);
    }
  }
  const prepared = prepareGraphTensorMap(body, tensorMap, bindings);
  if (prepared.reason_codes.length) {
    row.reason_codes.push(...prepared.reason_codes);
    return finishControlFailure(context, row);
  }
  const bodyScope = nestedScope(scope, nodeIndex, "body");
  const bodyEvidence = executeGraph(body, prepared.tensorMap, opsets, bodyScope, context, callStack, true);
  if (body.outputs.length !== node.outputs.length) return finishControlFailure(context, row, "scan8_body_output_cardinality_mismatch");
  const outputs = [];
  for (let index = 0; index < node.outputs.length; index += 1) {
    const bodyTensor = descriptor(prepared.tensorMap.get(body.outputs[index]?.name));
    if (!denseTensorDescriptor(bodyTensor)) return finishControlFailure(context, row, `scan8_body_output_non_tensor_value:${index}`);
    if (!bodyTensor?.shapeDeclared) return finishControlFailure(context, row, `scan8_body_output_shape_unresolved:${index}`);
    const prefix = index < stateCount ? [batchDimension] : [batchDimension, sequenceDimension];
    const patch = denseShapePatch(bodyTensor, [...prefix, ...bodyTensor.shape]);
    if (node.outputs[index]) outputs.push([node.outputs[index], patch]);
  }
  row.body_status = bodyEvidence.status;
  appendScopeFailureReasons(row, bodyEvidence, "scan8_body");
  row.state_variable_count = stateCount;
  row.scan_input_count = scanCount;
  row.scan_output_count = scanOutputCount;
  row.batch_dimension = batchDimension;
  row.sequence_dimension = sequenceDimension;
  row.status = bodyEvidence.status === "fail" ? "fail" : outputs.every(([, patch]) => concreteTensor(patch)) ? "pass" : "partial";
  context.controlFlowRows.push(row);
  return row.status === "fail"
    ? extendedFailure(row, "scan8_body_shape_inference_failed")
    : { status: row.status, reason: "scan8_output_not_fully_concrete", result: { outputs } };
}

function prepareGraphTensorMap(graph, lexicalTensors, inputBindings) {
  const tensorMap = new Map();
  const freeNames = graphFreeNames(graph);
  for (const [name, tensor] of lexicalTensors || []) {
    if (freeNames.has(name)) tensorMap.set(name, descriptor(tensor));
  }
  const localDefinitions = new Set([
    ...(graph.inputs || []).map((value) => value.name),
    ...(graph.initializers || []).map((tensor) => tensor.name),
    ...(graph.sparseInitializers || []).map((sparse) => sparse.values?.name),
    ...(graph.nodes || []).flatMap((node) => node.outputs || []),
  ].filter(Boolean));
  for (const name of localDefinitions) tensorMap.delete(name);
  for (const value of [...(graph.inputs || []), ...(graph.outputs || []), ...(graph.valueInfo || [])]) {
    if (value.name) mergeDeclaration(tensorMap, value.name, value);
  }
  for (const initializer of graph.initializers || []) {
    if (!initializer.name) continue;
    tensorMap.set(initializer.name, initializerPatch(initializer));
  }
  for (const sparse of graph.sparseInitializers || []) {
    const name = sparse.values?.name || "";
    if (name) tensorMap.set(name, sparseInitializerPatch(sparse));
  }
  for (const node of graph.nodes || []) {
    for (const name of [...(node.inputs || []), ...(node.outputs || [])]) {
      if (name && !tensorMap.has(name)) tensorMap.set(name, unknownTensor(name));
    }
  }
  const reasons = [];
  if ((graph.inputs || []).length !== inputBindings.length) reasons.push("graph_input_binding_cardinality_mismatch");
  for (let index = 0; index < Math.min(graph.inputs.length, inputBindings.length); index += 1) {
    const name = graph.inputs[index].name;
    const patch = inputBindings[index];
    if (!name || !patch) {
      reasons.push(`graph_input_binding_missing:${index}`);
      continue;
    }
    const conflict = mergeOnnxInferredTensor(tensorMap, name, patch, -1, "scope_input_binding");
    if (conflict) reasons.push(`graph_input_binding_conflict:${index}:${conflict.field}`);
  }
  return { tensorMap, reason_codes: reasons };
}

function graphFreeNames(graph) {
  const definitions = new Set([
    ...(graph.inputs || []).map((value) => value.name),
    ...(graph.initializers || []).map((tensor) => tensor.name),
    ...(graph.sparseInitializers || []).map((sparse) => sparse.values?.name),
    ...(graph.nodes || []).flatMap((node) => node.outputs || []),
  ].filter(Boolean));
  const references = new Set();
  for (const node of graph.nodes || []) {
    for (const name of node.inputs || []) if (name) references.add(name);
    for (const attribute of node.attributes?.values?.() || []) {
      for (const nested of [attribute.graph, ...(attribute.graphs || [])].filter(Boolean)) {
        for (const name of graphFreeNames(nested)) references.add(name);
      }
    }
  }
  for (const name of definitions) references.delete(name);
  return references;
}

function functionBodyGraph(fn, nodes) {
  const declarations = new Map((fn.valueInfo || []).map((value) => [value.name, value]));
  const value = (name) => declarations.get(name) || { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  return {
    name: fn.name,
    nodes,
    inputs: fn.inputs.map(value),
    outputs: fn.outputs.map(value),
    valueInfo: fn.valueInfo || [],
    initializers: [],
  };
}

function recordScopeExecution(target, scope, scopeClass, nodeCount, evidence, intrinsicCost = null, intrinsicCostExpected = false) {
  const invalidSchemaIndices = (evidence.schema_form_rows || []).filter((row) => row.status !== "pass").map((row) => row.node_index);
  const unassessed = new Set([
    ...(evidence.rule_unsupported_node_indices || []),
    ...invalidSchemaIndices,
    ...(evidence.extended_rule_failed_nodes || []).map((row) => row.node_index),
  ]);
  const unresolvedOutputCount = Number(evidence.unknown_node_output_count || 0)
    + Number(evidence.unresolved_non_dense_node_output_count || 0);
  const status = evidence.status === "fail" ? "fail" : unassessed.size ? "partial"
    : unresolvedOutputCount ? "partial" : "assessed";
  const existing = target.rows.get(scope) || {
    scope,
    scope_class: scopeClass,
    node_count: nodeCount,
    execution_count: 0,
    statuses: [],
    unassessed_node_count: 0,
    unresolved_output_count: 0,
    reason_codes: new Set(),
    intrinsic_cost_statuses: [],
    intrinsic_cost_variants: new Map(),
    intrinsic_cost_variant_overflow_count: 0,
    intrinsic_cost_unassessed_execution_count: 0,
  };
  existing.execution_count += 1;
  existing.statuses.push(status);
  existing.unassessed_node_count = Math.max(existing.unassessed_node_count, unassessed.size);
  existing.unresolved_output_count = Math.max(existing.unresolved_output_count, unresolvedOutputCount);
  if (intrinsicCost) {
    existing.intrinsic_cost_statuses.push(intrinsicCost.status);
    const key = JSON.stringify(intrinsicCost);
    const variant = existing.intrinsic_cost_variants.get(key);
    if (variant) variant.observation_count += 1;
    else if (target.intrinsicCostVariantCount < MAX_INTRINSIC_COST_VARIANTS) {
      existing.intrinsic_cost_variants.set(key, { ...intrinsicCost, observation_count: 1 });
      target.intrinsicCostVariantCount += 1;
    } else {
      existing.intrinsic_cost_variant_overflow_count += 1;
      target.intrinsicCostVariantOverflowCount += 1;
      existing.reason_codes.add("intrinsic_cost_variant_limit_exceeded");
    }
  } else if (intrinsicCostExpected) existing.intrinsic_cost_unassessed_execution_count += 1;
  for (const row of evidence.rule_unsupported_node_rows || []) existing.reason_codes.add(row.reason);
  for (const row of evidence.schema_form_rows || []) if (row.status !== "pass") existing.reason_codes.add("opset_schema_form_not_valid");
  for (const row of evidence.extended_rule_failed_nodes || []) existing.reason_codes.add(row.reason);
  if (Number(evidence.declaration_conflict_count || 0) > 0) existing.reason_codes.add("declared_shape_or_dtype_conflict");
  if (evidence.opset_import_contract?.status === "fail") existing.reason_codes.add("opset_import_contract_invalid");
  if (evidence.shape_scope?.registry_status === "fail") existing.reason_codes.add("local_function_registry_invalid");
  target.rows.set(scope, existing);
}

function finalizeScopeRow(row) {
  const status = row.statuses.includes("fail") ? "fail"
    : row.statuses.includes("partial") || row.intrinsic_cost_statuses.includes("partial")
      || row.intrinsic_cost_variant_overflow_count || row.intrinsic_cost_unassessed_execution_count ? "partial" : "assessed";
  return {
    scope: row.scope,
    scope_class: row.scope_class,
    status,
    node_count: row.node_count,
    execution_count: row.execution_count,
    assessed_node_count: Math.max(0, row.node_count - row.unassessed_node_count),
    unassessed_node_count: row.unassessed_node_count,
    unresolved_output_count: row.unresolved_output_count,
    intrinsic_cost_variant_count: row.intrinsic_cost_variants.size,
    intrinsic_cost_variant_overflow_count: row.intrinsic_cost_variant_overflow_count,
    intrinsic_cost_unassessed_execution_count: row.intrinsic_cost_unassessed_execution_count,
    intrinsic_cost_variants: [...row.intrinsic_cost_variants.values()],
    reason_codes: [...row.reason_codes].sort(),
  };
}

function controlRow({ nodeIndex, scope, importedOpset }, opName) {
  return { scope, node_index: nodeIndex, op_name: opName, imported_opset: importedOpset, status: "fail", reason_codes: [] };
}

function finishFunctionFailure(context, row) {
  row.reason_codes = [...new Set(row.reason_codes)];
  context.functionCallRows.push(row);
  return extendedFailure(row, row.reason_codes[0] || "local_function_call_contract_invalid");
}

function finishControlFailure(context, row, reason = "") {
  if (reason) row.reason_codes.push(reason);
  row.reason_codes = [...new Set(row.reason_codes)];
  row.status = "fail";
  context.controlFlowRows.push(row);
  return extendedFailure(row, row.reason_codes[0] || "control_flow_shape_inference_failed");
}

function extendedFailure(row, reason) {
  const conflict = deterministicArtifactConflict(row, reason);
  return {
    status: "fail",
    reason,
    failure_class: conflict ? "artifact_contract_conflict" : "analysis_residual",
    conflict_details: conflict,
    result: { outputs: [] },
  };
}

function deterministicArtifactConflict(row, reason) {
  const directDetails = (row.scope_failure_details || []).find(scopeFailureIsArtifactConflict);
  const nestedDetails = (row.exact_nested_failure_rows || [])
    .flatMap((nested) => nested.scope_failure_details || [])
    .find(scopeFailureIsArtifactConflict);
  const detail = directDetails || nestedDetails || null;
  if (detail) {
    return {
      scope: detail.scope || row.scope,
      node_index: detail.node_index ?? row.node_index,
      op_name: detail.op_name || row.op_name,
      field: detail.field,
      reason: detail.reason_codes?.[0] || reason,
      details: detail.details || null,
    };
  }
  const exactFailureReason = (row.reason_codes || []).find((value) => String(value || "").startsWith("loop_exact_body_failure_reason:"));
  if (row.deterministic_contract_failure) {
    return {
      scope: row.scope,
      node_index: row.node_index,
      op_name: row.op_name,
      field: "serialized_contract",
      reason: exactFailureReason || reason,
      details: null,
    };
  }
  const explicitReason = [...(row.reason_codes || []), reason].find((value) => (
    /^(graph_input_binding_(?:conflict|cardinality_mismatch|missing)|.*(?:_semantic_contract_conflict|_declaration_conflict|_schema_contract_invalid|_type_conflict|_cardinality_mismatch|_graph_missing|_axis_out_of_range|_non_tensor_value|_output_count_below_))/.test(String(value || ""))
  ));
  return explicitReason ? {
    scope: row.scope,
    node_index: row.node_index,
    op_name: row.op_name,
    field: "serialized_contract",
    reason: explicitReason,
    details: null,
  } : null;
}

function scopeFailureIsArtifactConflict(detail) {
  return ["semantic_contract", "declaration", "schema_form"].includes(detail?.field)
    || /(?:conflict|invalid|mismatch|out_of_range|missing)/.test(String(detail?.reason_codes?.[0] || ""));
}

function appendScopeFailureReasons(row, evidence, prefix) {
  if (evidence.opset_import_contract?.status === "fail") row.reason_codes.push(`${prefix}_opset_import_contract_invalid`);
  if (Number(evidence.schema_form_invalid_node_count || 0) > 0) row.reason_codes.push(`${prefix}_schema_contract_invalid`);
  if (Number(evidence.declaration_conflict_count || 0) > 0) row.reason_codes.push(`${prefix}_declaration_conflict`);
  const semanticConflict = (evidence.semantic_contract_conflicts || [])[0];
  if (semanticConflict) row.reason_codes.push(`${prefix}_semantic_contract_conflict:${semanticConflict.op_name || "UNKNOWN"}:${semanticConflict.reason || "unknown"}`);
  if (evidence.shape_scope?.registry_status === "fail") row.reason_codes.push(`${prefix}_function_registry_invalid`);
  if (Number(evidence.extended_rule_failed_node_count || 0) > 0) row.reason_codes.push(`${prefix}_extended_rule_failed`);
  const details = [
    ...(evidence.declaration_conflicts || []).map((item) => ({
      scope: prefix,
      node_index: item.node_index,
      op_name: item.op_name,
      tensor_name: item.tensor_name,
      field: item.field,
      declared: item.declared,
      inferred: item.inferred,
    })),
    ...(evidence.semantic_contract_conflicts || []).map((item) => ({
      scope: prefix,
      node_index: item.node_index,
      op_name: item.op_name,
      field: "semantic_contract",
      reason_codes: [item.reason],
      details: item.details || null,
    })),
    ...(evidence.schema_form_rows || []).filter((item) => item.status !== "pass").map((item) => ({
      scope: prefix,
      node_index: item.node_index,
      op_name: item.op_name,
      field: "schema_form",
      reason_codes: item.reason_codes,
    })),
  ];
  if (details.length) row.scope_failure_details = [...(row.scope_failure_details || []), ...details].slice(0, 32);
}

function failedGraphEvidence(graph, reason) {
  return {
    status: "fail",
    attempted_nodes: graph?.nodes?.length || 0,
    rule_unsupported_node_indices: [],
    schema_form_rows: [],
    extended_rule_failed_nodes: [{ node_index: -1, reason }],
    unknown_node_output_count: (graph?.nodes || []).flatMap((node) => node.outputs || []).filter(Boolean).length,
  };
}

function unionBranchValues(left, right, condition = null) {
  const leftType = onnxTypeProtoFromValue(left);
  const rightType = onnxTypeProtoFromValue(right);
  const leftKind = leftType?.kind || "unresolved";
  const rightKind = rightType?.kind || "unresolved";
  if (leftKind !== "tensor" || rightKind !== "tensor") {
    if (!leftType || !rightType || leftKind !== rightKind) return { status: "fail", patch: null };
    const merged = unionOnnxTypeProtos([leftType, rightType]);
    if (merged.status !== "pass") return { status: "fail", patch: null };
    const state = {};
    if (leftKind === "sequence") {
      const leftLength = exactSequenceLength(left);
      const rightLength = exactSequenceLength(right);
      if (leftLength != null && leftLength === rightLength) {
        state.sequenceLengthStatus = "assessed_exact";
        state.sequenceLength = leftLength;
      } else {
        state.sequenceLengthStatus = "not_assessed_branch_dependent";
        state.sequenceLength = null;
      }
      const leftInventory = exactSequenceInventory(left);
      const rightInventory = exactSequenceInventory(right);
      const inventory = unionSequenceInventories(leftInventory, rightInventory);
      state.sequenceElementInventoryStatus = inventory ? "assessed_exact" : "not_assessed_branch_dependent";
      state.sequenceElementTypes = inventory || [];
    } else if (leftKind === "optional") {
      const leftPresence = exactOptionalPresence(left);
      const rightPresence = exactOptionalPresence(right);
      if (leftPresence != null && leftPresence === rightPresence) {
        state.optionalPresenceStatus = "assessed_exact";
        state.optionalPresence = leftPresence;
      } else {
        state.optionalPresenceStatus = "not_assessed_branch_dependent";
        state.optionalPresence = null;
      }
    }
    return { status: "pass", patch: onnxValueDescriptorFromType(merged.type, state) };
  }
  return unionBranchTensors(left, right, condition);
}

function unionBranchTensors(left, right, condition = null) {
  if (!denseTensorDescriptor(left) || !denseTensorDescriptor(right)) return { status: "fail", patch: null };
  const leftDtype = knownDtype(left);
  const rightDtype = knownDtype(right);
  if (leftDtype && rightDtype && leftDtype !== rightDtype) return { status: "fail", patch: null };
  const mergedType = unionOnnxTypeProtos([onnxTypeProtoFromValue(left), onnxTypeProtoFromValue(right)]);
  if (mergedType.status !== "pass") return { status: "fail", patch: null };
  const patch = onnxValueDescriptorFromType(mergedType.type);
  if (sameStaticValues(left, right)) Object.assign(patch, staticPatch(left));
  if (sameStaticDimensionValues(left, right)) Object.assign(patch, staticDimensionPatch(left));
  if (condition && completeTensorShapeContract(left) && completeTensorShapeContract(right)) {
    const variants = [
      ...branchShapeVariants(left, { key: condition.key, value: condition.thenValue }),
      ...branchShapeVariants(right, { key: condition.key, value: condition.elseValue }),
    ];
    patch.conditionalShapeVariants = deduplicateBranchVariants(variants);
    patch.conditionalShapeContract = {
      schema: "deepbom.onnx_conditional_shape_contract.v1",
      status: "assessed_complete",
      variant_count: patch.conditionalShapeVariants.length,
      condition_keys: [...new Set(patch.conditionalShapeVariants.flatMap((variant) => (
        (variant.conditions || []).map((item) => item.key)
      )))].sort(),
    };
  }
  return { status: "pass", patch };
}

function unionSequenceInventories(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return null;
  const output = [];
  for (let index = 0; index < left.length; index += 1) {
    const merged = unionOnnxTypeProtos([left[index], right[index]]);
    if (merged.status !== "pass") return null;
    output.push(merged.type);
  }
  return output;
}

function validSequenceMapSequenceType(type) {
  return type?.kind === "sequence" && type.elementType?.kind === "tensor" && onnxTypeProtoKnown(type.elementType);
}

function exactSequenceLength(value) {
  return value?.sequenceLengthStatus === "assessed_exact" && Number.isSafeInteger(value.sequenceLength) && value.sequenceLength >= 0
    ? value.sequenceLength : null;
}

function exactSequenceInventory(value) {
  return value?.sequenceElementInventoryStatus === "assessed_exact" && Array.isArray(value.sequenceElementTypes)
    ? value.sequenceElementTypes.map(cloneOnnxTypeProto) : null;
}

function exactOptionalPresence(value) {
  return value?.optionalPresenceStatus === "assessed_exact" && typeof value.optionalPresence === "boolean"
    ? value.optionalPresence : null;
}

function mergeTensorContracts(left, right) {
  if (!left) return { status: right ? "pass" : "unresolved", patch: right || null };
  if (!right) return { status: "pass", patch: left };
  if (!denseTensorDescriptor(left) || !denseTensorDescriptor(right)) return { status: "fail", patch: null };
  const dtypeLeft = knownDtype(left);
  const dtypeRight = knownDtype(right);
  if (dtypeLeft && dtypeRight && dtypeLeft !== dtypeRight) return { status: "fail", patch: null };
  if (left.shapeDeclared && right.shapeDeclared && left.shape.length !== right.shape.length) return { status: "fail", patch: null };
  const patch = { dtype: dtypeLeft || dtypeRight || "UNKNOWN", shape: [], shapeDeclared: false };
  if (left.shapeDeclared || right.shapeDeclared) {
    const a = left.shapeDeclared ? left.shape : right.shape;
    const b = right.shapeDeclared ? right.shape : left.shape;
    patch.shapeDeclared = true;
    patch.shape = a.map((dim, index) => {
      const other = b[index];
      if (knownDimension(dim) && knownDimension(other) && dim !== other) return null;
      return knownDimension(dim) ? dim : other;
    });
    if (patch.shape.some((dim) => dim === null)) return { status: "fail", patch: null };
  }
  return { status: "pass", patch };
}

function removeAxis(tensor, axis) {
  if (!denseTensorDescriptor(tensor)) return { status: "fail", patch: null, dimension: null };
  if (!tensor?.shapeDeclared) return { status: "partial", patch: clearShape(tensor), dimension: -1 };
  const rank = tensor.shape.length;
  const normalized = normalizeAxis(axis, rank);
  if (normalized == null) return { status: "fail", patch: null, dimension: null };
  const shape = tensor.shape.slice();
  const [dimension] = shape.splice(normalized, 1);
  return { status: "pass", patch: denseShapePatch(tensor, shape), dimension };
}

function insertAxis(tensor, axis, dimension) {
  if (!denseTensorDescriptor(tensor)) return null;
  if (!tensor?.shapeDeclared) return clearShape(tensor);
  const rank = tensor.shape.length + 1;
  const normalized = normalizeAxis(axis, rank);
  if (normalized == null) return null;
  const shape = tensor.shape.slice();
  shape.splice(normalized, 0, dimension);
  return denseShapePatch(tensor, shape);
}

function removeLeadingDimensions(tensor, count) {
  if (!denseTensorDescriptor(tensor)) return null;
  if (!tensor?.shapeDeclared || tensor.shape.length < count) return null;
  return denseShapePatch(tensor, tensor.shape.slice(count));
}

function mergeDimension(current, next) {
  if (!knownDimension(current)) return knownDimension(next) ? next : -1;
  if (!knownDimension(next)) return current;
  return current === next ? current : null;
}

function normalizeAxis(axis, rank) {
  const value = Number(axis);
  if (!Number.isSafeInteger(value) || value < -rank || value >= rank) return null;
  return value < 0 ? value + rank : value;
}

function mapNamedOutputs(callOutputs, formalOutputs, tensorMap) {
  const outputs = [];
  for (let index = 0; index < Math.min(callOutputs.length, formalOutputs.length); index += 1) {
    const name = callOutputs[index];
    const patch = descriptor(tensorMap.get(formalOutputs[index]));
    if (name && concreteValue(patch)) outputs.push([name, patch]);
  }
  return outputs;
}

function initializerPatch(tensor) {
  const patch = tensorPatch(tensor.dtype || "UNKNOWN", tensor.shape || [], true);
  if (tensor.staticValuesComplete === true) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = [...(tensor.staticValues || [])];
    patch.staticValuesSource = tensor.staticValuesSource || "initializer";
  }
  return patch;
}

function sparseInitializerPatch(sparse) {
  return { ...tensorPatch(sparse?.values?.dtype || "UNKNOWN", sparse?.dims || [], true), initializerStorageKind: "sparse_tensor_proto" };
}

function mergeDeclaration(tensorMap, name, value) {
  if (!tensorMap.has(name)) tensorMap.set(name, unknownTensor(name));
  const patch = descriptor(value);
  mergeOnnxInferredTensor(tensorMap, name, patch, -1, "scope_declaration");
}

function descriptor(tensor) {
  if (!tensor) return null;
  const patch = {
    dtype: tensor.dtype || "UNKNOWN",
    shape: Array.isArray(tensor.shape) ? [...tensor.shape] : [],
    shapeDeclared: tensor.shapeDeclared === true || tensor.shape_declared === true,
  };
  if (tensor.valueKind || tensor.value_kind) patch.valueKind = tensor.valueKind || tensor.value_kind;
  if (tensor.typeProto || tensor.type_proto) patch.typeProto = onnxTypeProtoFromValue(tensor);
  if (["assessed_complete", "assessed_partial"].includes(tensor.conditionalShapeContract?.status)
    && Array.isArray(tensor.conditionalShapeVariants)) {
    patch.conditionalShapeContract = structuredClone(tensor.conditionalShapeContract);
    patch.conditionalShapeVariants = tensor.conditionalShapeVariants.map((variant) => ({
      ...descriptor(variant),
      conditions: structuredClone(variant.conditions || []),
    }));
  }
  if (Array.isArray(tensor.runtimeDimensionBounds)) patch.runtimeDimensionBounds = structuredClone(tensor.runtimeDimensionBounds);
  for (const key of ["sequenceLengthStatus", "sequenceLength", "sequenceElementInventoryStatus", "optionalPresenceStatus", "optionalPresence"]) {
    if (key in tensor) patch[key] = tensor[key];
  }
  if (Array.isArray(tensor.sequenceElementTypes)) patch.sequenceElementTypes = tensor.sequenceElementTypes.map(cloneOnnxTypeProto);
  if (tensor.staticValuesComplete === true || tensor.static_values_complete === true) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = [...(tensor.staticValues || tensor.static_values || [])];
    patch.staticValuesSource = tensor.staticValuesSource || tensor.static_values_source || "propagated";
  }
  if (tensor.staticDimensionValuesComplete === true || tensor.static_dimension_values_complete === true) {
    patch.staticDimensionValuesStatus = "assessed_exact_symbolic_shape_data";
    patch.staticDimensionValuesComplete = true;
    patch.staticDimensionValues = structuredClone(tensor.staticDimensionValues || tensor.static_dimension_values || []);
    patch.staticDimensionValuesSource = tensor.staticDimensionValuesSource || tensor.static_dimension_values_source || "propagated";
  }
  return patch;
}

function tensorPatch(dtype, shape, shapeDeclared) {
  return { dtype, shape: [...shape], shapeDeclared, valueKind: "tensor" };
}

function unknownTensor(name = "") {
  return { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
}

function clearShape(tensor) {
  return denseTensorDescriptor(tensor) ? denseShapePatch(tensor, [], false) : null;
}

function denseShapePatch(tensor, shape, shapeDeclared = true) {
  if (!denseTensorDescriptor(tensor)) return null;
  const copy = clearStatic(tensor);
  delete copy.typeProto;
  delete copy.type_proto;
  return {
    ...copy,
    dtype: knownDtype(tensor) || tensor.dtype || "UNKNOWN",
    shape: [...shape],
    shapeDeclared,
    valueKind: "tensor",
  };
}

function clearStatic(tensor) {
  if (!tensor) return null;
  const copy = { ...tensor };
  delete copy.staticValuesStatus;
  delete copy.staticValuesComplete;
  delete copy.staticValues;
  delete copy.staticValuesSource;
  delete copy.staticDimensionValuesStatus;
  delete copy.staticDimensionValuesComplete;
  delete copy.staticDimensionValues;
  delete copy.staticDimensionValuesSource;
  return copy;
}

function staticPatch(tensor) {
  return {
    staticValuesStatus: "complete",
    staticValuesComplete: true,
    staticValues: [...tensor.staticValues],
    staticValuesSource: tensor.staticValuesSource || "branch_union",
  };
}

function staticDimensionPatch(tensor) {
  return {
    staticDimensionValuesStatus: "assessed_exact_symbolic_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: structuredClone(tensor.staticDimensionValues),
    staticDimensionValuesSource: tensor.staticDimensionValuesSource || "branch_union",
  };
}

function sameStaticValues(left, right) {
  return left?.staticValuesComplete === true && right?.staticValuesComplete === true
    && JSON.stringify(left.staticValues) === JSON.stringify(right.staticValues);
}

function sameStaticDimensionValues(left, right) {
  return left?.staticDimensionValuesComplete === true && right?.staticDimensionValuesComplete === true
    && JSON.stringify(left.staticDimensionValues) === JSON.stringify(right.staticDimensionValues);
}

function concreteTensor(tensor) {
  const dimensions = tensor?.typeProto?.shapeDimensions || tensor?.type_proto?.shapeDimensions || [];
  return denseTensorDescriptor(tensor) && Boolean(knownDtype(tensor)) && tensor?.shapeDeclared === true
    && tensor.shape.every((dimension, index) => knownDimension(dimension)
      || dimensions[index]?.kind === "symbolic" && Boolean(dimensions[index]?.parameter));
}

function concreteValue(value) {
  if (denseTensorDescriptor(value)) return completeTensorShapeContract(value);
  return onnxTypeProtoKnown(onnxTypeProtoFromValue(value));
}

function completeTensorShapeContract(value) {
  return concreteTensor(value) || value?.conditionalShapeContract?.status === "assessed_complete"
    && Array.isArray(value.conditionalShapeVariants) && value.conditionalShapeVariants.length > 0
    && value.conditionalShapeVariants.every(concreteTensor);
}

function branchShapeVariants(value, condition) {
  const source = value?.conditionalShapeContract?.status === "assessed_complete"
    && Array.isArray(value.conditionalShapeVariants) && value.conditionalShapeVariants.length
    ? value.conditionalShapeVariants : [value];
  const variants = [];
  for (const item of source) {
    if (!concreteTensor(item)) continue;
    const conditions = mergeBranchConditions(item.conditions || [], [condition]);
    if (!conditions) continue;
    const patch = descriptor(item);
    delete patch.conditionalShapeContract;
    delete patch.conditionalShapeVariants;
    variants.push({ ...patch, conditions });
  }
  return variants;
}

function mergeBranchConditions(left, right) {
  const values = new Map();
  for (const condition of [...left, ...right]) {
    const key = String(condition?.key || "");
    const value = String(condition?.value || "");
    if (!key || !value || values.has(key) && values.get(key) !== value) return null;
    values.set(key, value);
  }
  return [...values].sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => ({ key, value }));
}

function deduplicateBranchVariants(variants) {
  const unique = new Map();
  for (const variant of variants) {
    const key = JSON.stringify({
      conditions: variant.conditions || [],
      dtype: variant.dtype,
      type: onnxTypeProtoFromValue(variant),
    });
    if (!unique.has(key)) unique.set(key, variant);
  }
  return [...unique.values()];
}

function knownDtype(tensor) {
  return denseTensorDescriptor(tensor) && tensor?.dtype && tensor.dtype !== "UNKNOWN" ? tensor.dtype : "";
}

function denseTensorDescriptor(tensor) {
  const kind = String(tensor?.valueKind || tensor?.value_kind || "");
  return Boolean(tensor) && (!kind || kind === "tensor" || kind === "unresolved" || kind === "undefined");
}

function knownDimension(value) {
  return Number.isSafeInteger(Number(value)) && Number(value) >= 0;
}

function attributeInteger(node, name) {
  const attribute = node.attributes?.get(name);
  if (Number.isSafeInteger(attribute?.i)) return attribute.i;
  if (/^-?\d+$/.test(attribute?.iExactDecimal || "")) {
    const value = Number(attribute.iExactDecimal);
    if (Number.isSafeInteger(value)) return value;
  }
  return null;
}

function attributeIntegers(node, name, count, fallback) {
  const attribute = node.attributes?.get(name);
  if (!attribute) return Array(count).fill(fallback);
  const values = (attribute.ints || []).map(Number);
  return values.length === count && values.every(Number.isSafeInteger) ? values : null;
}

function validOpsetMap(opsets) {
  const contract = assessOnnxOpsetImports(opsets);
  if (contract.invalid_import_count || contract.unresolvable_domains.length) return null;
  return new Map(contract.effective_imports.map((row) => [row.domain, row.version]));
}

function walkNodes(nodes, visit) {
  for (const node of nodes || []) {
    visit(node);
    for (const attribute of node.attributes?.values?.() || []) {
      for (const graph of [attribute.graph, ...(attribute.graphs || [])].filter(Boolean)) walkNodes(graph.nodes, visit);
    }
  }
}

function distinctNonEmpty(values) {
  return Array.isArray(values) && values.every(Boolean) && new Set(values).size === values.length;
}

function nestedScope(scope, nodeIndex, attributeName) {
  return `${scope}/node:${nodeIndex}/attribute:${attributeName}`;
}

function functionId(domain, name, overload) {
  return `${normalizeDomain(domain)}::${String(name || "")}::${String(overload || "")}`;
}

function normalizeDomain(domain) {
  const value = String(domain || "").trim();
  return !value || value === "ai.onnx" ? "ai.onnx" : value;
}
