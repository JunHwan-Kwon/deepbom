import { compareCanonicalText } from "../../report-utils.js";
import { clone, list, sortById } from "./shared.js";
import { buildStructuralBlocks } from "./blocks.js";

export function buildProgramModel(artifactIr, nativeFactLedger = null) {
  const graph = artifactIr.graph;
  if (graph.status !== "serialized") return notSerializedProgram(artifactIr.artifact.format);

  const regions = graph.scopes.map((scope) => ({
    id: regionId(scope.id),
    native_subject_ref: scope.id,
    kind: String(scope.kind || "serialized_region"),
    name: String(scope.name || scope.id),
    parent_region_ref: scope.parent_scope_ref ? regionId(scope.parent_scope_ref) : null,
    source_order: scope.native_index ?? null,
    materialization_status: String(scope.materialization_status || "unknown"),
    invocation_semantics: String(scope.invocation_semantics || "not_declared"),
    native_source: clone(scope.native_source || null),
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  }));
  const regionIds = new Set(regions.map((row) => row.id));
  const values = graph.values.map((value) => ({
    id: value.id,
    native_subject_ref: value.id,
    region_ref: regionId(value.scope_ref),
    native_index: value.native_index,
    name: value.name,
    kind: value.value_kind,
    dtype: value.dtype,
    shape: clone(value.shape),
    ...(value.type_contract ? { type_contract: clone(value.type_contract) } : {}),
    shape_signature: clone(value.shape_signature),
    roles: clone(value.roles),
    logical_byte_length: clone(value.logical_byte_length),
    storage_refs: clone(value.storage_refs),
    native_source: clone(value.native_source),
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  }));
  const valueIds = new Set(values.map((row) => row.id));
  const operations = graph.operators.map((operator) => ({
    id: operator.id,
    native_subject_ref: operator.id,
    region_ref: regionId(operator.scope_ref),
    native_index: operator.native_index,
    kind: "operation",
    native_op: { domain: operator.domain, name: operator.op_type, version: operator.version },
    semantic_family: null,
    name: operator.name,
    source_order: operator.native_index,
    dependency_order: null,
    display_order: null,
    runtime_order: null,
    input_port_refs: operator.inputs.map((_, index) => portId(operator.id, "input", index)),
    output_port_refs: operator.outputs.map((_, index) => portId(operator.id, "output", index)),
    metrics: clone(operator.metrics),
    ...(operator.metric_contracts ? { metric_contracts: clone(operator.metric_contracts), attributes: clone(operator.attributes), semantic_contract: clone(operator.semantic_contract) } : {}),
    quantization_summary: clone(operator.quantization_summary),
    native_source: clone(operator.native_source),
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  }));
  const ports = graph.operators.flatMap((operator) => [
    ...operator.inputs.map((port, index) => ({
      id: portId(operator.id, "input", index), operation_ref: operator.id, direction: "input",
      position: Number(port.port ?? index), value_ref: port.value_ref, evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    })),
    ...operator.outputs.map((port, index) => ({
      id: portId(operator.id, "output", index), operation_ref: operator.id, direction: "output",
      position: Number(port.port ?? index), value_ref: port.value_ref, evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    })),
  ]);
  const relationships = [];
  for (const value of graph.values) {
    for (const consumer of list(value.consumers)) {
      if (!value.producer) continue;
      relationships.push({
        id: `relationship:data:${encodeId(value.id)}:${encodeId(consumer.operator_ref)}:${consumer.port}`,
        kind: "data_dependency",
        from_ref: value.producer.operator_ref,
        to_ref: consumer.operator_ref,
        value_ref: value.id,
        native_relationship: "serialized_producer_consumer",
        evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      });
    }
    for (const storageRef of list(value.storage_refs)) relationships.push({
      id: `relationship:storage:${encodeId(value.id)}:${encodeId(storageRef)}`,
      kind: "storage_binding", from_ref: value.id, to_ref: storageRef, value_ref: value.id,
      native_relationship: "serialized_value_storage_binding", evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    });
  }
  for (const operation of operations) relationships.push({
    id: `relationship:ownership:${encodeId(operation.region_ref)}:${encodeId(operation.id)}`,
    kind: "region_ownership", from_ref: operation.region_ref, to_ref: operation.id, value_ref: null,
    native_relationship: "serialized_scope_membership", evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  });
  for (const relationship of graph.scope_relationships) relationships.push({
    id: relationship.id,
    kind: "region_ownership",
    from_ref: relationship.source_operator_ref || regionId(relationship.source_scope_ref),
    to_ref: regionId(relationship.target_scope_ref),
    value_ref: null,
    native_relationship: relationship.relation,
    role: relationship.role,
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  });
  for (const dependency of list(nativeFactLedger?.facts?.control_dependencies)) {
    if (dependency.status !== "resolved" || !dependency.source_operation_ref || !dependency.target_operation_ref) continue;
    relationships.push({
      id: `relationship:control:${encodeId(dependency.id)}`,
      kind: "control_dependency",
      from_ref: dependency.source_operation_ref,
      to_ref: dependency.target_operation_ref,
      value_ref: null,
      native_relationship: "serialized_control_input",
      native_source: clone(dependency.native_source),
      evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    });
  }

  applyDependencyAndDisplayOrder(operations, relationships);
  const blocks = buildStructuralBlocks(operations, regions, { ports, values, relationships });
  const program = {
    id: "program:0",
    entry_region_refs: graph.primary_scope_ref ? [regionId(graph.primary_scope_ref)] : [],
    input_value_refs: clone(graph.inputs),
    output_value_refs: clone(graph.outputs),
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  };
  const functions = regions.filter((row) => row.invocation_semantics === "model_local_function_definition").map((row) => ({
    id: `function:${encodeId(row.id)}`,
    region_ref: row.id,
    name: row.name,
    evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
  }));
  return {
    status: "serialized",
    executable_graph_status: graph.executable_graph_status,
    programs: [program], functions, regions, blocks, operations, values, ports,
    relationships: sortById(relationships),
    orders: buildOrderSummary(operations, relationships, regionIds, valueIds),
    completeness: {
      status: graph.completeness,
      blocks: blocks.length ? "complete_reversible_structural_partition" : "not_applicable_no_operations",
      control_dependencies: relationships.some((row) => row.kind === "control_dependency") ? "materialized" : "not_exposed_by_source_projection",
      runtime_order: "not_observed_in_static_artifact",
    },
    interpretation_boundary: "Program regions, operations, values, ports, and dependencies are projected from serialized facts. Dependency order is a partial order; deterministic display order is not an execution schedule. Control dependencies are included only when an allow-listed native fact resolves to both serialized operations. Runtime order remains absent unless independently observed.",
  };
}

function notSerializedProgram(format) {
  return {
    status: "not_serialized", executable_graph_status: "not_serialized_by_format",
    programs: [], functions: [], regions: [], blocks: [], operations: [], values: [], ports: [], relationships: [],
    orders: { source_order_status: "not_applicable", dependency_order_status: "not_applicable", display_order_status: "not_applicable", runtime_order_status: "not_observed" },
    completeness: { status: "complete_not_applicable", blocks: "not_applicable", control_dependencies: "not_applicable", runtime_order: "not_observed" },
    interpretation_boundary: `${format} does not serialize an executable program graph. No operation, dependency, call, control-flow, or runtime order is synthesized from tensor names, metadata, or storage order.`,
  };
}

function applyDependencyAndDisplayOrder(operations, relationships) {
  const byRegion = new Map();
  for (const operation of operations) {
    if (!byRegion.has(operation.region_ref)) byRegion.set(operation.region_ref, []);
    byRegion.get(operation.region_ref).push(operation);
  }
  const dependencies = relationships.filter((row) => ["data_dependency", "control_dependency"].includes(row.kind));
  for (const rows of byRegion.values()) {
    const ids = new Set(rows.map((row) => row.id));
    const predecessors = new Map(rows.map((row) => [row.id, new Set()]));
    const outgoing = new Map(rows.map((row) => [row.id, new Set()]));
    for (const edge of dependencies) {
      if (!ids.has(edge.from_ref) || !ids.has(edge.to_ref) || edge.from_ref === edge.to_ref) continue;
      predecessors.get(edge.to_ref).add(edge.from_ref);
      outgoing.get(edge.from_ref).add(edge.to_ref);
    }
    const remainingIncoming = new Map([...predecessors].map(([id, refs]) => [id, new Set(refs)]));
    const dependencyDepth = new Map();
    const ready = rows.filter((row) => remainingIncoming.get(row.id).size === 0).sort(nativeOrder);
    const ordered = [];
    while (ready.length) {
      const current = ready.shift();
      ordered.push(current);
      const currentPredecessors = predecessors.get(current.id);
      dependencyDepth.set(current.id, currentPredecessors.size
        ? Math.max(...[...currentPredecessors].map((id) => dependencyDepth.get(id) ?? 0)) + 1 : 0);
      for (const successor of outgoing.get(current.id)) {
        remainingIncoming.get(successor).delete(current.id);
        if (remainingIncoming.get(successor).size === 0) {
          ready.push(rows.find((row) => row.id === successor));
          ready.sort(nativeOrder);
        }
      }
    }
    const remaining = rows.filter((row) => !ordered.includes(row)).sort(nativeOrder);
    for (const [index, operation] of [...ordered, ...remaining].entries()) {
      operation.display_order = index;
      operation.dependency_order = dependencyDepth.get(operation.id) ?? null;
    }
  }
}

function buildOrderSummary(operations, relationships) {
  const data = relationships.filter((row) => ["data_dependency", "control_dependency"].includes(row.kind));
  return {
    source_order_status: operations.every((row) => Number.isSafeInteger(row.source_order)) ? "complete" : "partial",
    dependency_order_status: operations.every((row) => Number.isSafeInteger(row.dependency_order)) ? "complete_acyclic_projection" : data.length ? "partial_or_cyclic" : "complete_no_inter_operation_dependencies",
    display_order_status: operations.every((row) => Number.isSafeInteger(row.display_order)) ? "complete_deterministic" : "partial",
    runtime_order_status: "not_observed",
  };
}

function nativeOrder(left, right) { return Number(left.native_index ?? Number.MAX_SAFE_INTEGER) - Number(right.native_index ?? Number.MAX_SAFE_INTEGER) || compareCanonicalText(left.id, right.id); }
function regionId(scopeId) { return `region:${scopeId}`; }
function portId(operationId, direction, position) { return `port:${operationId}:${direction}:${position}`; }
function encodeId(value) { return encodeURIComponent(String(value)).replaceAll("%", "_"); }
