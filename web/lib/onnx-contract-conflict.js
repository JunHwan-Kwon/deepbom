import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const ONNX_CONTRACT_CONFLICT_CAPSULE_SCHEMA = "deepbom.onnx_contract_conflict_capsule.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const PRIMARY_SCOPE = "scope:onnx:main_graph";

export function attachOnnxContractConflictCapsule(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "onnx") return analysis;
  analysis.onnx_contract_conflict = buildOnnxContractConflictCapsule(analysis);
  return analysis;
}

export function buildOnnxContractConflictCapsule(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "onnx") return null;
  const artifactSha256 = normalizeSha256(analysis.model_sha256 || analysis.artifact_sha256);
  if (!artifactSha256) throw new Error("ONNX contract-conflict evidence requires the analyzed artifact SHA-256.");
  const shape = analysis.onnx_shape_inference;
  const dynamic = analysis.dynamic_shape_cost_contract;
  if (!shape || !dynamic) return signedCapsule({
    schema: ONNX_CONTRACT_CONFLICT_CAPSULE_SCHEMA,
    status: "NOT_ASSESSED",
    evidence_class: "NOT_ASSESSABLE",
    artifact: artifactIdentity(analysis, artifactSha256),
    analyzer_contracts: {
      onnx_shape_inference: shape?.schema || null,
      dynamic_shape_cost: dynamic?.schema || null,
    },
    summary: zeroSummary(),
    root_conflicts: [],
    condition_bound_conflicts: [],
    affected_values: [],
    downstream_blocked_nodes: [],
    blocked_mac_rows: [],
    method: "No capsule is derived unless both the ONNX shape ledger and dynamic-cost ledger are present.",
    interpretation_boundary: "NOT_ASSESSED is not evidence that the serialized contract is valid.",
  });

  const declarationRoots = list(shape.declaration_conflicts).map((row, index) => declarationRoot(row, index));
  const semanticRoots = list(shape.semantic_contract_conflicts).map((row, index) => semanticRoot(row, index));
  const rootConflicts = [...declarationRoots, ...semanticRoots];
  const rootIndex = buildRootIndex(rootConflicts);
  const conditionBoundConflicts = conditionalConflicts(analysis.tensors);
  const conditionIndex = buildConditionalIndex(conditionBoundConflicts);
  const outputTensorIndices = new Set(list(analysis.ops).flatMap((op) => integerList(op.outputs)));
  const affectedValues = list(analysis.tensors)
    .filter((tensor, index) => outputTensorIndices.has(nativeIndex(tensor, index)) && tensorContractInvalid(tensor))
    .map((tensor, index) => affectedValue(tensor, nativeIndex(tensor, index), rootIndex));
  const downstreamBlockedNodes = list(shape.rule_unresolved_nodes)
    .filter((row) => String(row?.reason || "").startsWith("blocked_by_upstream_contract_conflict:"))
    .map((row) => ({
      subject_ref: operatorSubject(row.node_index),
      node_index: safeIndex(row.node_index),
      op_name: text(row.op_name) || "UNKNOWN",
      reason: text(row.reason) || "blocked_by_upstream_contract_conflict",
      blocking_tensor_name: text(row.blocked_by?.tensor_name) || text(row.reason).split(":").slice(1).join(":"),
      root_conflict_refs: rootRefs(row.blocked_by?.root_conflict || row.blocked_by, rootIndex, conditionIndex),
    }));
  const blockedMacRows = list(dynamic.total_macs_unresolved_ops)
    .filter((row) => row?.resolution_class === "artifact_contract_conflict")
    .map((row) => ({
      subject_ref: operatorSubject(row.op_index),
      op_index: safeIndex(row.op_index),
      op_name: text(row.op_name) || "UNKNOWN",
      reason: text(row.reason) || "artifact_contract_conflict",
      blocking_values: list(row.blocking_tensors).map((tensor) => ({
        subject_ref: valueSubject(tensor.index),
        tensor_index: safeIndex(tensor.index),
        tensor_name: text(tensor.name),
        role: text(tensor.role) || "unknown",
      })),
      root_conflict_refs: unique(list(row.root_conflicts).flatMap((root) => rootRefs(root, rootIndex, conditionIndex))),
    }));
  const opHistogram = histogram(blockedMacRows, "op_name");
  const summary = {
    unconditional_root_conflict_count: rootConflicts.length,
    declaration_root_conflict_count: declarationRoots.length,
    semantic_root_conflict_count: semanticRoots.length,
    condition_bound_invalid_variant_count: conditionBoundConflicts.length,
    invalid_node_output_count: nonnegative(shape.invalid_node_output_count),
    conditionally_invalid_node_output_count: nonnegative(shape.conditionally_invalid_node_output_count),
    downstream_blocked_node_count: downstreamBlockedNodes.length,
    blocked_mac_row_count: blockedMacRows.length,
    blocked_mac_op_histogram: opHistogram,
    unresolved_root_reference_count: [
      ...affectedValues,
      ...downstreamBlockedNodes,
      ...blockedMacRows,
    ].filter((row) => row.root_conflict_refs.length === 0).length,
  };
  const hasConflict = rootConflicts.length > 0 || conditionBoundConflicts.length > 0;
  return signedCapsule({
    schema: ONNX_CONTRACT_CONFLICT_CAPSULE_SCHEMA,
    status: hasConflict ? "INVALID_CONTRACT" : "ASSESSED_NO_CONFLICT",
    evidence_class: "OBSERVED_DERIVED",
    artifact: artifactIdentity(analysis, artifactSha256),
    analyzer_contracts: {
      onnx_shape_inference: shape.schema || null,
      dynamic_shape_cost: dynamic.schema || null,
    },
    summary,
    root_conflicts: rootConflicts,
    condition_bound_conflicts: conditionBoundConflicts,
    affected_values: affectedValues,
    downstream_blocked_nodes: downstreamBlockedNodes,
    blocked_mac_rows: blockedMacRows,
    method: "Normalize the source-pinned ONNX shape-inference conflict ledger and dynamic-cost blocker ledger without re-running inference. Preserve serialized declarations, deterministic inferences, condition predicates, native indices, canonical Artifact IR subject references, and exact blocker denominators.",
    interpretation_boundary: "INVALID_CONTRACT prevents affected shape and MAC rows from being promoted to exact values. The capsule does not repair the model, infer runtime dimensions, prove ONNX Runtime execution, or establish task quality.",
  });
}

export function validateOnnxContractConflictCapsule(capsule, artifactIr = null) {
  if (capsule?.schema !== ONNX_CONTRACT_CONFLICT_CAPSULE_SCHEMA) throw new Error("ONNX contract-conflict capsule schema is invalid.");
  if (!SHA256.test(String(capsule?.artifact?.sha256 || ""))) throw new Error("ONNX contract-conflict capsule artifact SHA-256 is invalid.");
  const expectedHash = sha256TextHex(canonicalJson(withoutHash(capsule)));
  if (capsule.capsule_sha256 !== expectedHash) throw new Error("ONNX contract-conflict capsule digest does not match its canonical content.");
  const summary = capsule.summary || {};
  requireCount(summary.unconditional_root_conflict_count, list(capsule.root_conflicts).length, "unconditional root conflicts");
  requireCount(summary.declaration_root_conflict_count, list(capsule.root_conflicts).filter((row) => row.kind === "declared_inferred_conflict").length, "declaration roots");
  requireCount(summary.semantic_root_conflict_count, list(capsule.root_conflicts).filter((row) => row.kind === "operator_semantic_conflict").length, "semantic roots");
  requireCount(summary.condition_bound_invalid_variant_count, list(capsule.condition_bound_conflicts).length, "condition-bound variants");
  requireCount(summary.downstream_blocked_node_count, list(capsule.downstream_blocked_nodes).length, "downstream blocked nodes");
  requireCount(summary.blocked_mac_row_count, list(capsule.blocked_mac_rows).length, "blocked MAC rows");
  requireCount(histogramTotal(summary.blocked_mac_op_histogram), list(capsule.blocked_mac_rows).length, "blocked MAC histogram");
  const expectedStatus = Number(summary.unconditional_root_conflict_count || 0) + Number(summary.condition_bound_invalid_variant_count || 0) > 0
    ? "INVALID_CONTRACT" : capsule.status === "NOT_ASSESSED" ? "NOT_ASSESSED" : "ASSESSED_NO_CONFLICT";
  if (capsule.status !== expectedStatus) throw new Error("ONNX contract-conflict capsule status does not match its conflict counts.");
  const resolvableRefs = new Set([...list(capsule.root_conflicts), ...list(capsule.condition_bound_conflicts)].map((row) => row.conflict_ref));
  const rows = [...list(capsule.affected_values), ...list(capsule.downstream_blocked_nodes), ...list(capsule.blocked_mac_rows)];
  for (const row of rows) for (const ref of list(row.root_conflict_refs)) {
    if (!resolvableRefs.has(ref)) throw new Error(`ONNX contract-conflict capsule contains an unresolved root reference: ${ref}`);
  }
  requireCount(summary.unresolved_root_reference_count, rows.filter((row) => list(row.root_conflict_refs).length === 0).length, "unresolved root references");
  if (artifactIr) validateAgainstArtifactIr(capsule, artifactIr);
  return capsule;
}

function validateAgainstArtifactIr(capsule, artifactIr) {
  if (artifactIr?.artifact?.sha256 !== capsule.artifact.sha256) throw new Error("ONNX conflict capsule and Artifact IR identify different artifact bytes.");
  const operatorRefs = new Set(list(artifactIr?.graph?.operators).map((row) => row.id));
  const valueRefs = new Set(list(artifactIr?.graph?.values).map((row) => row.id));
  for (const row of [...list(capsule.root_conflicts), ...list(capsule.downstream_blocked_nodes), ...list(capsule.blocked_mac_rows)]) {
    if (row.subject_ref && !operatorRefs.has(row.subject_ref)) throw new Error(`ONNX conflict operator subject is absent from Artifact IR: ${row.subject_ref}`);
  }
  for (const row of [...list(capsule.condition_bound_conflicts), ...list(capsule.affected_values)]) {
    if (row.subject_ref && !valueRefs.has(row.subject_ref)) throw new Error(`ONNX conflict value subject is absent from Artifact IR: ${row.subject_ref}`);
  }
  for (const row of list(capsule.blocked_mac_rows)) for (const value of list(row.blocking_values)) {
    if (value.subject_ref && !valueRefs.has(value.subject_ref)) throw new Error(`ONNX conflict blocking value is absent from Artifact IR: ${value.subject_ref}`);
  }
}

function declarationRoot(row, index) {
  return {
    conflict_ref: `onnx-conflict:declaration:${index}`,
    kind: "declared_inferred_conflict",
    subject_ref: operatorSubject(row.node_index),
    node_index: safeIndex(row.node_index),
    op_name: text(row.op_name) || "UNKNOWN",
    tensor_name: text(row.tensor_name),
    field: text(row.field) || "contract",
    declared: row.declared ?? null,
    inferred: row.inferred ?? null,
  };
}

function semanticRoot(row, index) {
  return {
    conflict_ref: `onnx-conflict:semantic:${index}`,
    kind: "operator_semantic_conflict",
    subject_ref: operatorSubject(row.node_index),
    node_index: safeIndex(row.node_index),
    op_name: text(row.op_name) || "UNKNOWN",
    output_names: list(row.output_names).map(text).filter(Boolean),
    reason: text(row.reason) || "operator_contract_invalid",
    details: row.details ?? null,
  };
}

function conditionalConflicts(tensors) {
  const rows = [];
  for (const [position, tensor] of list(tensors).entries()) {
    const contract = tensor?.conditional_shape_contract || tensor?.conditionalShapeContract || {};
    for (const [variantIndex, failure] of list(contract.variant_failures).entries()) {
      if (failure?.status !== "invalid") continue;
      const tensorIndex = nativeIndex(tensor, position);
      rows.push({
        conflict_ref: `onnx-conflict:conditional:${tensorIndex}:${variantIndex}`,
        kind: "condition_bound_invalid_variant",
        subject_ref: valueSubject(tensorIndex),
        tensor_index: tensorIndex,
        tensor_name: text(tensor?.name) || `tensor_${tensorIndex}`,
        variant_index: variantIndex,
        reason: text(failure.reason) || "conditionally_invalid_tensor_contract",
        conditions: structuredClone(list(failure.conditions)),
        details: failure.details ?? null,
      });
    }
  }
  return rows;
}

function affectedValue(tensor, index, rootIndex) {
  const conflict = tensor.contract_conflict || tensor.contractConflict || {};
  return {
    subject_ref: valueSubject(index),
    tensor_index: index,
    tensor_name: text(tensor.name) || `tensor_${index}`,
    reason: text(conflict.reason) || "invalid_serialized_tensor_contract",
    root_conflict_refs: rootRefs(conflict.root_conflict || conflict.rootConflict || conflict, rootIndex, new Map()),
  };
}

function buildRootIndex(rows) {
  const index = new Map();
  for (const row of rows) {
    for (const source of rootLineage(row)) {
      for (const key of rootKeys(source)) if (!index.has(key)) index.set(key, row.conflict_ref);
    }
  }
  return index;
}

function rootLineage(row) {
  const lineage = [];
  let current = row;
  const seen = new Set();
  while (current && typeof current === "object" && !seen.has(current)) {
    seen.add(current);
    if (current.node_index != null || current.reason || current.tensor_name) lineage.push(current);
    current = current.details;
  }
  return lineage;
}

function buildConditionalIndex(rows) {
  const index = new Map();
  for (const row of rows) {
    const key = `${row.tensor_name}|${row.reason}|${canonicalJson(row.conditions || [])}`;
    if (!index.has(key)) index.set(key, row.conflict_ref);
  }
  return index;
}

function rootRefs(root, rootIndex, conditionIndex) {
  if (!root || typeof root !== "object") return [];
  const direct = rootKeys(root).map((key) => rootIndex.get(key)).filter(Boolean);
  if (direct.length) return unique(direct);
  const conditionalKey = `${text(root.tensor_name)}|${text(root.reason)}|${canonicalJson(root.conditions || [])}`;
  const conditional = conditionIndex.get(conditionalKey);
  return conditional ? [conditional] : [];
}

function rootKeys(row) {
  const node = safeIndex(row.node_index);
  return unique([
    `${node}|${text(row.tensor_name)}|${text(row.field)}|${text(row.reason)}`,
    `${node}|${text(row.tensor_name)}|${text(row.field)}|`,
    `${node}|||${text(row.reason)}`,
    `${node}|${text(row.output_names?.[0])}||${text(row.reason)}`,
  ]);
}

function artifactIdentity(analysis, sha256) {
  const source = analysis?.artifact_set?.source || {};
  return {
    sha256,
    filename: text(analysis.filename) || "model.onnx",
    byte_length: nonnegativeOrNull(analysis.file_size_bytes ?? analysis.file_size),
    artifact_set_sha256: normalizeSha256(analysis?.artifact_set?.artifact_set_sha256),
    source_locator: text(source.canonical_locator) || null,
    source_revision: text(source.revision || source.immutability?.revision) || null,
  };
}

function zeroSummary() {
  return {
    unconditional_root_conflict_count: 0,
    declaration_root_conflict_count: 0,
    semantic_root_conflict_count: 0,
    condition_bound_invalid_variant_count: 0,
    invalid_node_output_count: 0,
    conditionally_invalid_node_output_count: 0,
    downstream_blocked_node_count: 0,
    blocked_mac_row_count: 0,
    blocked_mac_op_histogram: [],
    unresolved_root_reference_count: 0,
  };
}

function signedCapsule(body) {
  const capsule = { ...body, capsule_sha256: sha256TextHex(canonicalJson(body)) };
  return validateOnnxContractConflictCapsule(capsule);
}

function withoutHash(capsule) {
  const { capsule_sha256: _hash, ...body } = capsule || {};
  return body;
}

function histogram(rows, key) {
  const counts = new Map();
  for (const row of rows) {
    const name = text(row?.[key]) || "UNKNOWN";
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([name, count]) => ({ name, count }));
}

function histogramTotal(rows) { return list(rows).reduce((sum, row) => sum + Number(row?.count || 0), 0); }
function operatorSubject(index) { const value = safeIndex(index); return value == null ? null : `operator:${PRIMARY_SCOPE}:${value}`; }
function valueSubject(index) { const value = safeIndex(index); return value == null ? null : `value:${PRIMARY_SCOPE}:${value}`; }
function tensorContractInvalid(tensor) { return tensor?.contract_status === "invalid" || tensor?.contractStatus === "invalid"; }
function nativeIndex(row, fallback) { return safeIndex(row?.index) ?? fallback; }
function safeIndex(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : null; }
function nonnegative(value) { return safeIndex(value) ?? 0; }
function nonnegativeOrNull(value) { const number = Number(value); return Number.isFinite(number) && number >= 0 ? number : null; }
function integerList(values) { return list(values).map(safeIndex).filter((value) => value != null); }
function normalizeSha256(value) { const digest = text(value).toLowerCase(); return SHA256.test(digest) ? digest : null; }
function requireCount(actual, expected, label) { if (Number(actual) !== expected) throw new Error(`ONNX contract-conflict capsule does not conserve ${label}.`); }
function unique(values) { return [...new Set(values.filter(Boolean))]; }
function list(value) { return Array.isArray(value) ? value : []; }
function text(value) { return String(value ?? "").trim(); }
