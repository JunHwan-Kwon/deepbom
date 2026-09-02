import { ARTIFACT_IR_METHOD_VERSION, ARTIFACT_IR_SCHEMA, SHA256 } from "./constants.js";
import { validateArtifactIrRuntimeReconciliation } from "../../artifact-ir-runtime.js";
import { exact, list, text, uniqueIds } from "./shared.js";

export function validateArtifactIrBody(value) {
  if (value?.schema !== ARTIFACT_IR_SCHEMA || value?.method_version !== ARTIFACT_IR_METHOD_VERSION) throw new Error("Artifact IR schema identity is invalid.");
  if (value?.hash_contract?.algorithm !== "SHA-256"
    || value?.hash_contract?.canonicalization !== "RFC8785-JCS"
    || value?.hash_contract?.source_encoding !== "UTF-8"
    || JSON.stringify(value?.hash_contract?.excluded_pointers) !== JSON.stringify(["/artifact_ir_sha256"])) {
    throw new Error("Artifact IR hash contract is invalid.");
  }
  if (!SHA256.test(String(value.artifact?.sha256 || "")) || !text(value.artifact?.filename, 1000) || !text(value.artifact?.format, 40)) throw new Error("Artifact IR artifact identity is invalid.");
  const graph = value.graph;
  if (!graph || !Array.isArray(graph.scopes) || !Array.isArray(graph.scope_relationships) || !Array.isArray(graph.operators) || !Array.isArray(graph.values)) throw new Error("Artifact IR graph ledger is invalid.");
  if (graph.status === "not_serialized" && (graph.scopes.length || graph.operators.length || graph.values.length || graph.totals.relationship_count)) {
    throw new Error("Artifact IR fabricated an executable graph for a graphless format.");
  }
  if (graph.totals.scope_count !== graph.scopes.length || graph.totals.materialized_scope_count !== graph.scopes.filter((row) => row.materialization_status === "materialized").length
    || graph.totals.scope_relationship_count !== graph.scope_relationships.length || graph.totals.operator_count !== graph.operators.length || graph.totals.value_count !== graph.values.length) throw new Error("Artifact IR graph count conservation failed.");
  const scopeIds = uniqueIds(graph.scopes, "graph scope");
  const operatorIds = uniqueIds(graph.operators, "operator");
  const valueIds = uniqueIds(graph.values, "value");
  const operatorById = new Map(graph.operators.map((row) => [row.id, row]));
  const valueById = new Map(graph.values.map((row) => [row.id, row]));
  if (graph.status === "serialized" && !scopeIds.has(graph.primary_scope_ref)) throw new Error("Artifact IR primary graph scope reference is invalid.");
  for (const scope of graph.scopes) {
    if (scope.parent_scope_ref != null && (!scopeIds.has(scope.parent_scope_ref) || scope.parent_scope_ref === scope.id)) {
      throw new Error("Artifact IR graph scope parent reference is invalid.");
    }
    if (scope.materialization_status === "materialized") {
      const scopeOperatorCount = graph.operators.filter((row) => row.scope_ref === scope.id).length;
      const scopeValueCount = graph.values.filter((row) => row.scope_ref === scope.id).length;
      if (scope.declared_operator_count != null && scope.declared_operator_count !== scopeOperatorCount) throw new Error("Artifact IR materialized scope operator count is inconsistent.");
      if (scope.declared_value_count != null && scope.declared_value_count !== scopeValueCount) throw new Error("Artifact IR materialized scope value count is inconsistent.");
    }
  }
  for (const operator of graph.operators) {
    if (!scopeIds.has(operator.scope_ref)) throw new Error("Artifact IR operator scope reference is invalid.");
    for (const port of [...list(operator.inputs), ...list(operator.outputs)]) {
      const graphValue = valueById.get(port.value_ref);
      if (!graphValue) throw new Error("Artifact IR operator port references an unknown value.");
      if (graphValue.scope_ref !== operator.scope_ref) throw new Error("Artifact IR operator port crosses graph scopes.");
    }
  }
  for (const relationship of graph.scope_relationships) {
    if (!scopeIds.has(relationship.source_scope_ref) || !scopeIds.has(relationship.target_scope_ref)
      || (relationship.source_operator_ref && !operatorIds.has(relationship.source_operator_ref))) {
      throw new Error("Artifact IR scope relationship reference is invalid.");
    }
    if (relationship.source_operator_ref && operatorById.get(relationship.source_operator_ref)?.scope_ref !== relationship.source_scope_ref) {
      throw new Error("Artifact IR scope relationship source operator belongs to a different scope.");
    }
  }
  let relationshipCount = 0;
  for (const value of graph.values) {
    if (!scopeIds.has(value.scope_ref)) throw new Error("Artifact IR value scope reference is invalid.");
    if (value.producer && !operatorIds.has(value.producer.operator_ref)) throw new Error("Artifact IR value producer reference is invalid.");
    for (const consumer of list(value.consumers)) if (!operatorIds.has(consumer.operator_ref)) throw new Error("Artifact IR value consumer reference is invalid.");
    if (value.producer && operatorById.get(value.producer.operator_ref)?.scope_ref !== value.scope_ref) throw new Error("Artifact IR value producer crosses graph scopes.");
    for (const consumer of list(value.consumers)) {
      if (operatorById.get(consumer.operator_ref)?.scope_ref !== value.scope_ref) throw new Error("Artifact IR value consumer crosses graph scopes.");
    }
    relationshipCount += list(value.consumers).length;
  }
  if (relationshipCount !== graph.totals.relationship_count) throw new Error("Artifact IR relationship conservation failed.");
  for (const inputRef of list(graph.inputs)) {
    const graphValue = valueById.get(inputRef);
    if (!graphValue || graphValue.scope_ref !== graph.primary_scope_ref || !list(graphValue.roles).includes("graph_input")) throw new Error("Artifact IR graph input reference is invalid.");
  }
  for (const outputRef of list(graph.outputs)) {
    const graphValue = valueById.get(outputRef);
    if (!graphValue || graphValue.scope_ref !== graph.primary_scope_ref || !list(graphValue.roles).includes("graph_output")) throw new Error("Artifact IR graph output reference is invalid.");
  }
  const primaryOperators = graph.operators.filter((row) => row.scope_ref === graph.primary_scope_ref);
  const assessedMacs = primaryOperators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  const serializedScopeMacs = graph.operators.reduce((sum, row) => sum + BigInt(row.metrics?.macs?.decimal || "0"), 0n);
  if (graph.status === "serialized" && (String(graph.totals.assessed_macs?.decimal || "0") !== assessedMacs.toString()
    || String(graph.totals.serialized_scope_assessed_macs?.decimal || "0") !== serializedScopeMacs.toString())) throw new Error("Artifact IR MAC conservation failed.");
  const storage = value.storage_topology;
  if (!storage || !Array.isArray(storage.objects) || storage.totals.object_count !== storage.objects.length) throw new Error("Artifact IR storage count conservation failed.");
  const storageIds = uniqueIds(storage.objects, "storage object");
  const storageBytes = storage.objects.reduce((sum, row) => sum + BigInt(row.serialized_byte_length?.decimal || "0"), 0n);
  if (String(storage.totals.serialized_object_bytes_sum?.decimal || "0") !== storageBytes.toString()) throw new Error("Artifact IR storage byte conservation failed.");
  if (storage.totals.exact_range_count !== storage.objects.filter((row) => row.byte_range?.status === "exact").length
    || storage.totals.payload_digest_count !== storage.objects.filter((row) => SHA256.test(String(row.payload_sha256 || ""))).length) {
    throw new Error("Artifact IR storage evidence count conservation failed.");
  }
  for (const graphValue of graph.values) {
    for (const storageRef of list(graphValue.storage_refs)) {
      if (!storageIds.has(storageRef)) throw new Error("Artifact IR value references an unknown storage object.");
    }
  }
  const architecture = value.architecture_projection;
  if (!architecture || !Array.isArray(architecture.nodes) || !Array.isArray(architecture.relationships)) throw new Error("Artifact IR architecture projection is invalid.");
  const architectureIds = uniqueIds(architecture.nodes, "architecture node");
  if (architecture.totals.node_count !== architecture.nodes.length || architecture.totals.relationship_count !== architecture.relationships.length) throw new Error("Artifact IR architecture count conservation failed.");
  if (graph.status === "not_serialized" && architecture.relationships.length) throw new Error("Artifact IR graphless architecture projection contains fabricated relationships.");
  for (const node of architecture.nodes) {
    for (const storageRef of list(node.storage_object_refs)) if (!storageIds.has(storageRef)) throw new Error("Artifact IR architecture node references an unknown storage object.");
  }
  const quantization = value.quantization_contracts;
  if (!quantization || !Array.isArray(quantization.records) || !quantization.totals) throw new Error("Artifact IR quantization ledger is invalid.");
  const quantIds = uniqueIds(quantization.records, "quantization record");
  if (quantIds.size !== quantization.totals.record_count) throw new Error("Artifact IR quantization count conservation failed.");
  if (quantization.totals.affine_record_count !== quantization.records.filter((row) => row.mapping?.family === "affine").length
    || quantization.totals.block_encoding_record_count !== quantization.records.filter((row) => row.mapping?.family === "format_defined_block_encoding").length
    || quantization.totals.complete_record_count !== quantization.records.filter((row) => row.completeness === "complete_for_serialized_contract").length
    || quantization.totals.partial_record_count !== quantization.records.filter((row) => row.completeness !== "complete_for_serialized_contract").length) {
    throw new Error("Artifact IR quantization classification count conservation failed.");
  }
  const subjects = new Set([...valueIds, ...storageIds]);
  for (const row of quantization.records) {
    if (!subjects.has(row.subject_ref)) throw new Error("Artifact IR quantization subject reference is invalid.");
    for (const storageRef of list(row.related_storage_refs)) if (!storageIds.has(storageRef)) throw new Error("Artifact IR quantization record references an unknown related storage object.");
  }
  if (!value.overlays || !Array.isArray(value.overlays.static) || !Array.isArray(value.overlays.runtime)) throw new Error("Artifact IR overlay ledger is invalid.");
  const overlaySubjects = new Set([...operatorIds, ...architectureIds]);
  const allOverlays = [...value.overlays.static, ...value.overlays.runtime];
  uniqueIds(allOverlays, "overlay");
  for (const overlay of allOverlays) {
    for (const row of list(overlay.rows)) if (!overlaySubjects.has(row.subject_ref)) throw new Error("Artifact IR overlay subject reference is invalid.");
    if (overlay.kind === "runtime_assignment") validateArtifactIrRuntimeReconciliation(overlay, overlaySubjects);
  }
  if (value.completeness?.static_overlay_count !== value.overlays.static.length
    || value.completeness?.runtime_overlay_count !== value.overlays.runtime.length
    || value.completeness?.unknown_is_zero !== false) {
    throw new Error("Artifact IR completeness summary is inconsistent.");
  }
  if (!text(value.interpretation_boundary, 2400)) throw new Error("Artifact IR interpretation boundary is missing.");
}
