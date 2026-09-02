import {
  compactStrings, dimensions, exact, exactInteger, integerArray, list, logicalTensorBytes,
  nonNegativeInteger, optionalInteger, optionalText, positiveStorageBytes, scopedStorageId, storageId, tensorIndex,
} from "./shared.js";

export function buildSerializedGraph(analysis, format, tensors) {
  const ops = canonicalOps(analysis.ops);
  const scope = primaryScope(analysis, format);
  const valueByIndex = new Map(tensors.map((tensor, position) => [tensorIndex(tensor, position), tensor]));
  const producer = new Map();
  const consumers = new Map();
  const operators = ops.map((op) => {
    const inputPorts = tensorPorts(op.inputs, scope.id);
    const outputPorts = tensorPorts(op.outputs, scope.id);
    const operatorRef = operatorId(scope.id, op.index);
    outputPorts.forEach(({ native_index: index, port }) => {
      if (!producer.has(index)) producer.set(index, { operator_ref: operatorRef, port });
    });
    inputPorts.forEach(({ native_index: index, port }) => {
      if (!consumers.has(index)) consumers.set(index, []);
      consumers.get(index).push({ operator_ref: operatorRef, port });
    });
    return {
      id: operatorRef,
      legacy_graph_node_id: `op:${op.index}`,
      scope_ref: scope.id,
      native_index: op.index,
      op_type: String(op.name || `OP_${op.index}`),
      name: optionalText(op.graph_node_name || op.coreml_layer_name || op.name),
      domain: String(op.domain || format),
      version: optionalInteger(op.version ?? op.op_version),
      inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
      outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
      metrics: {
        macs: exactInteger(op.macs_decimal ?? op.macs),
        mac_assessment_status: String(op.macs_status || (op.macs_decimal != null || op.macs != null ? "assessed" : "not_assessed")),
        logical_io_bytes: exactInteger(op.estimated_bytes),
      },
      quantization_summary: {
        state: String(op.quantization_state || (op.quantized_compute_path ? "8bit_compute" : "none")),
        risk: String(op.quant_risk || "none"),
      },
      topology: {
        depth: nonNegativeInteger(op.topo_depth),
        role: optionalText(op.topo_role),
        stage: optionalText(op.stage_key ?? op.stage_index),
      },
      native_source: nativeOperatorLocator(format, scope, op),
      completeness: "serialized_operator_with_available_canonical_contract",
    };
  });
  const graphInputs = new Set(integerArray(analysis.input_tensor_indices));
  const graphOutputs = new Set(integerArray(analysis.output_tensor_indices));
  const values = [...valueByIndex.entries()].sort(([left], [right]) => left - right).map(([index, tensor]) => ({
    id: valueId(scope.id, index),
    scope_ref: scope.id,
    native_index: index,
    name: String(tensor.name || `tensor_${index}`),
    value_kind: String(tensor.value_kind || "tensor"),
    dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(tensor.shape),
    shape_signature: Array.isArray(tensor.shape_signature) ? dimensions(tensor.shape_signature) : null,
    shape_contract_status: String(tensor.contract_status || tensor.buffer_data_status || "not_declared"),
    producer: producer.get(index) || null,
    consumers: uniqueConsumers(consumers.get(index) || []),
    roles: [
      ...(graphInputs.has(index) || tensor.role === "input" ? ["graph_input"] : []),
      ...(graphOutputs.has(index) || tensor.role === "output" ? ["graph_output"] : []),
      ...(tensor.constant_buffer || positiveStorageBytes(tensor, format) > 0 ? ["serialized_constant_or_storage"] : []),
    ],
    storage_refs: positiveStorageBytes(tensor, format) > 0 ? [storageId(index)] : [],
    logical_byte_length: logicalTensorBytes(tensor),
    native_source: nativeValueLocator(format, scope, tensor, index),
  }));
  const additional = materializeAdditionalScopes(analysis, format, scope.id);
  const scopes = [scope, ...additional.scopes];
  const allOperators = [...operators, ...additional.operators];
  const allValues = [...values, ...additional.values];
  const macAssessment = graphMacAssessment(analysis, operators);
  const serializedScopeAssessedMacs = allOperators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  return {
    status: "serialized",
    executable_graph_status: "serialized_artifact_graph",
    primary_scope_ref: scope.id,
    scopes,
    scope_relationships: additional.scope_relationships,
    operators: allOperators,
    values: allValues,
    inputs: values.filter((value) => value.roles.includes("graph_input")).map((value) => value.id),
    outputs: values.filter((value) => value.roles.includes("graph_output")).map((value) => value.id),
    totals: {
      scope_count: scopes.length,
      materialized_scope_count: scopes.filter((row) => row.materialization_status === "materialized").length,
      scope_relationship_count: additional.scope_relationships.length,
      operator_count: allOperators.length,
      value_count: allValues.length,
      relationship_count: allValues.reduce((sum, value) => sum + value.consumers.length, 0),
      macs: macAssessment.total,
      assessed_macs: macAssessment.assessed,
      serialized_scope_assessed_macs: exact(serializedScopeAssessedMacs),
      mac_assessment: macAssessment.assessment,
    },
    completeness: additional.scopes.length
      ? additional.scopes.every((row) => row.materialization_status === "materialized")
        ? "all_serialized_scopes_materialized"
        : "primary_scope_materialized_nested_scope_inventory_preserved"
      : "serialized_scope_materialized",
    interpretation_boundary: "Operators, values, ports, and producer-consumer relationships come from the serialized artifact. Primary-scope assessed_macs is the model-entry nominal total. serialized_scope_assessed_macs is only the sum of independently serialized scope rows and is not an execution total for conditional, repeated, or function scopes.",
  };
}

export function notSerializedGraph(format) {
  return {
    status: "not_serialized",
    executable_graph_status: "not_serialized_by_format",
    primary_scope_ref: null,
    scopes: [], scope_relationships: [], operators: [], values: [], inputs: [], outputs: [],
    totals: { scope_count: 0, materialized_scope_count: 0, scope_relationship_count: 0, operator_count: 0, value_count: 0, relationship_count: 0, macs: null, assessed_macs: null, serialized_scope_assessed_macs: null, mac_assessment: null },
    completeness: "complete_not_applicable",
    interpretation_boundary: `${format || "This container"} does not serialize an executable operator graph. Runtime frameworks may construct one externally; DEEPBOM does not infer it from tensor names or architecture order.`,
  };
}

function primaryScope(analysis, format) {
  const base = { native_index: 0, materialization_status: "materialized", parent_scope_ref: null, invocation_semantics: "model_entrypoint" };
  if (format === "tflite") return { ...base, id: "scope:tflite:subgraph:0", kind: "tflite_subgraph", name: String(analysis?.tflite_subgraph_inventory?.rows?.[0]?.name || "subgraph_0") };
  if (format === "onnx") return { ...base, id: "scope:onnx:main_graph", kind: "onnx_graph", name: String(analysis.graph_name || "main_graph") };
  if (format === "coreml" || format === "mlmodel") return { ...base, id: "scope:coreml:primary", kind: "coreml_serialized_program", name: String(analysis.coreml?.selected_function || "primary") };
  return { ...base, id: `scope:${format}:primary`, kind: `${format}_serialized_program`, name: "primary" };
}

function materializeAdditionalScopes(analysis, format, primaryId) {
  if (format === "tflite") return materializeTfliteScopes(analysis, primaryId);
  if (format === "onnx") return materializeOnnxScopes(analysis, primaryId);
  return { scopes: [], operators: [], values: [], scope_relationships: [] };
}

function materializeTfliteScopes(analysis, primaryId) {
  const rows = list(analysis?.tflite_subgraph_inventory?.rows).filter((row) => Number(row.subgraph_index) !== 0);
  const scopes = [];
  const operators = [];
  const values = [];
  for (const row of rows) {
    const nativeIndex = nonNegativeInteger(row.subgraph_index);
    if (nativeIndex == null) continue;
    const scopeId = `scope:tflite:subgraph:${nativeIndex}`;
    const tensorRows = list(row.tensor_intrinsics);
    const operatorRows = list(row.operator_intrinsics);
    const materialized = tensorRows.length === nonNegativeInteger(row.tensor_count)
      && operatorRows.length === nonNegativeInteger(row.operator_count);
    scopes.push({
      id: scopeId,
      kind: "tflite_subgraph",
      native_index: nativeIndex,
      name: String(row.name || `subgraph_${nativeIndex}`),
      materialization_status: materialized ? "materialized" : "inventory_only",
      parent_scope_ref: null,
      invocation_semantics: String(row.invocation_semantics || "serialized_subgraph_execution_count_not_bound"),
      declared_operator_count: nonNegativeInteger(row.operator_count),
      declared_value_count: nonNegativeInteger(row.tensor_count),
    });
    if (!materialized) continue;
    const producer = new Map();
    const consumers = new Map();
    for (const [position, operator] of operatorRows.entries()) {
      const index = nonNegativeInteger(operator.operator_index) ?? position;
      const operatorRef = operatorId(scopeId, index);
      const inputPorts = tensorPorts(operator.inputs, scopeId);
      const outputPorts = tensorPorts(operator.outputs, scopeId);
      for (const { native_index: tensorIndexValue, port } of outputPorts) {
        if (!producer.has(tensorIndexValue)) producer.set(tensorIndexValue, { operator_ref: operatorRef, port });
      }
      for (const { native_index: tensorIndexValue, port } of inputPorts) {
        if (!consumers.has(tensorIndexValue)) consumers.set(tensorIndexValue, []);
        consumers.get(tensorIndexValue).push({ operator_ref: operatorRef, port });
      }
      operators.push({
        id: operatorRef,
        legacy_graph_node_id: `scope:${nativeIndex}:op:${index}`,
        scope_ref: scopeId,
        native_index: index,
        op_type: String(operator.name || `OP_${index}`),
        name: optionalText(operator.name),
        domain: "tflite",
        version: optionalInteger(operator.version),
        inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        metrics: {
          macs: exactInteger(operator.nominal_macs_decimal ?? operator.nominal_macs),
          mac_assessment_status: String(operator.mac_assessment_status || "not_assessed"),
          logical_io_bytes: exactInteger(operator.logical_io_payload_bytes),
        },
        quantization_summary: { state: "not_projected_from_scope_intrinsics", risk: "not_assessed" },
        topology: { depth: null, role: null, stage: null },
        native_source: { format: "tflite", path: `SubGraph[${nativeIndex}].operators[${index}]`, byte_range: null },
        completeness: "serialized_nested_operator_intrinsic_contract",
      });
    }
    const graphInputs = new Set(integerArray(row.input_tensor_indices).filter((index) => index >= 0));
    const graphOutputs = new Set(integerArray(row.output_tensor_indices).filter((index) => index >= 0));
    for (const [position, tensor] of tensorRows.entries()) {
      const index = nonNegativeInteger(tensor.tensor_index) ?? position;
      const storageBytes = nonNegativeInteger(tensor.buffer_data_length) || 0;
      values.push({
        id: valueId(scopeId, index),
        scope_ref: scopeId,
        native_index: index,
        name: String(tensor.name || `tensor_${index}`),
        value_kind: "tensor",
        dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(tensor.shape),
        shape_signature: Array.isArray(tensor.shape_signature) && tensor.shape_signature.length ? dimensions(tensor.shape_signature) : null,
        shape_contract_status: String(tensor.payload_status || "serialized_shape"),
        producer: producer.get(index) || null,
        consumers: uniqueConsumers(consumers.get(index) || []),
        roles: [
          ...(graphInputs.has(index) ? ["graph_input"] : []),
          ...(graphOutputs.has(index) ? ["graph_output"] : []),
          ...(tensor.constant_buffer || storageBytes > 0 ? ["serialized_constant_or_storage"] : []),
        ],
        storage_refs: storageBytes > 0 ? [scopedStorageId(scopeId, index)] : [],
        logical_byte_length: exactInteger(tensor.logical_payload_bytes),
        native_source: { format: "tflite", path: `SubGraph[${nativeIndex}].tensors[${index}]` },
      });
    }
  }
  const scopeIds = new Set(scopes.map((row) => row.id));
  const scopeRelationships = list(analysis?.tflite_subgraph_inventory?.references).map((row, index) => {
    const sourceScopeRef = `scope:tflite:subgraph:${row.source_subgraph_index}`;
    const targetScopeRef = `scope:tflite:subgraph:${row.target_subgraph_index}`;
    return {
      id: `scope-relationship:tflite:${index}`,
      relation: "serialized_subgraph_reference",
      source_scope_ref: sourceScopeRef,
      source_operator_ref: operatorId(sourceScopeRef, row.source_op_index),
      target_scope_ref: targetScopeRef,
      role: String(row.role || "subgraph_reference"),
    };
  }).filter((row) => (row.source_scope_ref === primaryId || scopeIds.has(row.source_scope_ref)) && scopeIds.has(row.target_scope_ref));
  return { scopes, operators, values, scope_relationships: scopeRelationships };
}

function materializeOnnxScopes(analysis, primaryId) {
  const ledger = list(analysis?.onnx_domain_analysis?.scope_ledger);
  const nested = ledger.filter((row) => String(row.scope || "main_graph") !== "main_graph")
    .sort((left, right) => String(left.scope).localeCompare(String(right.scope)));
  const scopeIdByName = new Map([["main_graph", primaryId]]);
  nested.forEach((row, index) => scopeIdByName.set(String(row.scope), `scope:onnx:nested:${index}`));
  const scopes = nested.map((row, index) => ({
    id: scopeIdByName.get(String(row.scope)),
    kind: String(row.scope_class || "nested_graph"),
    native_index: index,
    name: String(row.name || row.scope),
    materialization_status: Array.isArray(row.nodes) && Array.isArray(row.values) ? "materialized" : "inventory_only",
    parent_scope_ref: scopeIdByName.get(String(row.parent_scope || "main_graph")) || primaryId,
    invocation_semantics: String(row.invocation_semantics || (row.scope_class === "local_function_body" ? "model_local_function_definition" : "serialized_nested_graph")),
    declared_operator_count: nonNegativeInteger(row.node_count),
    declared_value_count: nonNegativeInteger(row.value_count),
  }));
  const operators = [];
  const values = [];
  const scopeRelationships = [];
  for (const [scopePosition, row] of nested.entries()) {
    const scopeId = scopeIdByName.get(String(row.scope));
    if (!Array.isArray(row.nodes) || !Array.isArray(row.values)) continue;
    const producer = new Map();
    const consumers = new Map();
    for (const [position, node] of row.nodes.entries()) {
      const index = nonNegativeInteger(node.node_index) ?? position;
      const operatorRef = operatorId(scopeId, index);
      const inputPorts = namedValuePorts(node.inputs, scopeId, row.values);
      const outputPorts = namedValuePorts(node.outputs, scopeId, row.values);
      for (const { native_index: valueIndex, port } of outputPorts) if (!producer.has(valueIndex)) producer.set(valueIndex, { operator_ref: operatorRef, port });
      for (const { native_index: valueIndex, port } of inputPorts) {
        if (!consumers.has(valueIndex)) consumers.set(valueIndex, []);
        consumers.get(valueIndex).push({ operator_ref: operatorRef, port });
      }
      operators.push({
        id: operatorRef,
        legacy_graph_node_id: `scope:${scopePosition + 1}:op:${index}`,
        scope_ref: scopeId,
        native_index: index,
        op_type: String(node.op_name || `OP_${index}`),
        name: optionalText(node.node_name || node.op_name),
        domain: String(node.domain || "ai.onnx"),
        version: optionalInteger(node.imported_opset),
        inputs: inputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        outputs: outputPorts.map(({ port, value_ref }) => ({ port, value_ref })),
        metrics: { macs: null, mac_assessment_status: "not_assessed_scope_local_cost_not_projected", logical_io_bytes: null },
        quantization_summary: { state: "not_projected_from_nested_scope", risk: "not_assessed" },
        topology: { depth: null, role: null, stage: null },
        native_source: { format: "onnx", path: String(node.native_path || `${row.scope}.node[${index}]`), byte_range: null },
        completeness: "serialized_nested_operator_and_port_identity",
      });
    }
    for (const [position, value] of row.values.entries()) {
      const index = nonNegativeInteger(value.value_index) ?? position;
      values.push({
        id: valueId(scopeId, index),
        scope_ref: scopeId,
        native_index: index,
        name: String(value.name || `value_${index}`),
        value_kind: String(value.value_kind || "tensor"),
        dtype: String(value.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(value.shape),
        shape_signature: null,
        shape_contract_status: String(value.contract_status || "serialized_name_only"),
        producer: producer.get(index) || null,
        consumers: uniqueConsumers(consumers.get(index) || []),
        roles: compactStrings(value.roles),
        storage_refs: [],
        logical_byte_length: null,
        native_source: { format: "onnx", path: String(value.native_path || `${row.scope}.value[${JSON.stringify(String(value.name || ""))}]`) },
      });
    }
    const parent = scopeIdByName.get(String(row.parent_scope || "main_graph")) || primaryId;
    const ownerNodeIndex = nonNegativeInteger(row.owner_node_index);
    scopeRelationships.push({
      id: `scope-relationship:onnx:${scopePosition}`,
      relation: row.scope_class === "local_function_body" ? "serialized_local_function_definition" : "serialized_nested_graph_ownership",
      source_scope_ref: parent,
      source_operator_ref: ownerNodeIndex == null ? null : operatorId(parent, ownerNodeIndex),
      target_scope_ref: scopeId,
      role: String(row.owner_role || row.scope_class || "nested_scope"),
    });
  }
  return { scopes, operators, values, scope_relationships: scopeRelationships };
}

function graphMacAssessment(analysis, operators) {
  const ledger = analysis?.mac_assessment;
  const compute = nonNegativeInteger(ledger?.compute_ops);
  const assessedCount = nonNegativeInteger(ledger?.assessed_compute_ops);
  const unassessedCount = nonNegativeInteger(ledger?.not_assessed_compute_ops);
  const assessed = exactInteger(ledger?.total_assessed_macs_decimal ?? ledger?.total_assessed_macs);
  if (compute != null && assessedCount != null && unassessedCount != null && assessedCount + unassessedCount === compute && assessed) {
    return {
      total: unassessedCount === 0 ? assessed : null,
      assessed,
      assessment: { status: unassessedCount === 0 ? "complete" : assessedCount ? "partial" : "not_assessed", compute_op_count: compute, assessed_compute_op_count: assessedCount, unassessed_compute_op_count: unassessedCount, scope: String(ledger.metric_scope || "nominal tensor-contraction MACs") },
    };
  }
  const sum = operators.reduce((total, row) => total + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  return { total: exact(sum), assessed: exact(sum), assessment: { status: "complete_without_separate_compute_denominator", compute_op_count: null, assessed_compute_op_count: null, unassessed_compute_op_count: null, scope: "sum_of_operator_macs_present_in_artifact_ir" } };
}

function nativeOperatorLocator(format, scope, op) {
  if (format === "tflite") return { format, path: `SubGraph[${scope.native_index}].operators[${op.index}]`, byte_range: null };
  if (format === "onnx") return { format, path: `ModelProto.graph.node[${op.index}]`, byte_range: null };
  if (format === "coreml" || format === "mlmodel") return { format: "coreml", path: `Model.primary.operations[${op.index}]`, byte_range: null };
  if (["executorch", "pte", "ptd"].includes(format)) return { format: "executorch", path: `Program.execution_plan[0].chains.instructions[${op.index}]`, byte_range: null };
  return { format, path: `operators[${op.index}]`, byte_range: null };
}

function nativeValueLocator(format, scope, tensor, index) {
  if (format === "tflite") return { format, path: `SubGraph[${scope.native_index}].tensors[${index}]` };
  if (format === "onnx") return { format, path: `ModelProto.graph.value[name=${JSON.stringify(String(tensor.name || ""))}]` };
  return { format, path: `values[${index}]` };
}

function uniqueConsumers(rows) {
  const values = new Map();
  for (const row of rows) values.set(`${row.operator_ref}:${row.port}`, row);
  return [...values.values()].sort((left, right) => left.operator_ref.localeCompare(right.operator_ref) || left.port - right.port);
}

function tensorPorts(values, scopeId) {
  return list(values).map((value, port) => ({ native_index: Number(value), port }))
    .filter((row) => Number.isSafeInteger(row.native_index) && row.native_index >= 0)
    .map((row) => ({ ...row, value_ref: valueId(scopeId, row.native_index) }));
}

function namedValuePorts(names, scopeId, values) {
  const indexByName = new Map(list(values).map((row, index) => [String(row.name || ""), nonNegativeInteger(row.value_index) ?? index]));
  return list(names).map((name, port) => ({ native_index: indexByName.get(String(name || "")), port }))
    .filter((row) => row.native_index != null)
    .map((row) => ({ ...row, value_ref: valueId(scopeId, row.native_index) }));
}

function canonicalOps(value) {
  return list(value).map((op, position) => ({ ...op, index: Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position }))
    .sort((left, right) => left.index - right.index);
}

function operatorId(scopeId, index) { return `operator:${scopeId}:${index}`; }

function valueId(scopeId, index) { return `value:${scopeId}:${index}`; }
