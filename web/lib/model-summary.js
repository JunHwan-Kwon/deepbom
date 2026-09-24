import { compareCanonicalText } from "./report-utils.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { validateModelIr } from "./model-ir.js";

export const MODEL_SUMMARY_SCHEMA = "deepbom.model_summary.v1";
export const MODEL_SUMMARY_METHOD_VERSION = "1.1.0";
export const MODEL_SUMMARY_LEVELS = Object.freeze(["auto", "operation", "block", "storage"]);

export function buildModelSummary(input, { level = "auto" } = {}) {
  const modelIr = validateModelIr(input);
  if (!MODEL_SUMMARY_LEVELS.includes(level)) throw new Error(`Model summary level must be one of: ${MODEL_SUMMARY_LEVELS.join(", ")}.`);
  const selectedLevel = selectLevel(modelIr, level);
  const rows = selectedLevel === "operation"
    ? operationRows(modelIr)
    : selectedLevel === "block"
      ? blockRows(modelIr)
      : selectedLevel === "storage"
        ? storageRows(modelIr)
        : [];
  const totals = buildTotals(modelIr);
  const status = rows.length
    ? "materialized"
    : selectedLevel === "identity"
      ? "not_assessable_safe_envelope_only"
      : "not_applicable_for_selected_level";
  const body = {
    schema: MODEL_SUMMARY_SCHEMA,
    method_version: MODEL_SUMMARY_METHOD_VERSION,
    source_contract: {
      schema: modelIr.schema,
      model_ir_sha256: modelIr.model_ir_sha256,
      artifact_sha256: modelIr.artifact.sha256,
    },
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      source_encoding: "UTF-8",
      excluded_pointers: ["/model_summary_sha256"],
    },
    artifact: structuredClone(modelIr.artifact),
    artifact_role: modelIr.artifact_role,
    projection: {
      requested_level: level,
      selected_level: selectedLevel,
      status,
      selection_basis: selectionBasis(modelIr, selectedLevel),
      ordering: orderingContract(modelIr, selectedLevel),
    },
    columns: columnContract(selectedLevel),
    rows,
    totals,
    coverage: {
      program: modelIr.program.completeness,
      storage: modelIr.tensors_and_storage.completeness,
      bindings: modelIr.weight_bindings.status,
      quantization: modelIr.quantization.status,
      loss_count: modelIr.loss_ledger.count,
      unknown_is_zero: false,
    },
    trainability: {
      status: "not_assessable_from_serialized_deployment_artifact",
      trainable_parameter_count: null,
      non_trainable_parameter_count: null,
      explanation: "Serialized values and storage are reported separately from framework trainability. DEEPBOM does not relabel bound constants as trainable parameters without an identity-bound declaration or source-framework contract.",
    },
    loss_ledger: structuredClone(modelIr.loss_ledger),
    interpretation_boundary: "This is a deterministic tabular projection of deepbom.model_ir.v1. Operation rows are serialized program subjects; block rows are reversible structural groups, not framework layers or runtime fusion; storage rows are serialized objects, not trainable parameters or runtime allocations. Display order is not runtime order. Unknown and not-assessable values are never converted to zero. This engineering evidence does not establish model quality, safety, regulatory compliance, standard conformance, or actual runtime behavior.",
  };
  validateModelSummaryBody(body);
  return deepFreeze({ ...body, model_summary_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateModelSummary(document) {
  canonicalJson(document);
  const body = structuredClone(document);
  const digest = String(body.model_summary_sha256 || "").toLowerCase();
  delete body.model_summary_sha256;
  validateModelSummaryBody(body);
  if (!/^[a-f0-9]{64}$/.test(digest) || digest !== sha256TextHex(canonicalJson(body))) throw new Error("Model summary SHA-256 is invalid.");
  return deepFreeze({ ...body, model_summary_sha256: digest });
}

export function validateModelSummaryAgainstSource(document, modelIr) {
  const summary = validateModelSummary(document);
  if (summary.method_version !== MODEL_SUMMARY_METHOD_VERSION) throw new Error("Legacy summary requires its original projection method for source validation.");
  const expected = buildModelSummary(modelIr, { level: summary.projection.requested_level });
  if (canonicalJson(summary) !== canonicalJson(expected)) throw new Error("Model summary differs from its independently supplied Model IR source.");
  return summary;
}

export function renderModelSummaryTable(input) {
  const summary = validateModelSummary(input);
  const lines = [
    `DEEPBOM format-neutral model summary`,
    `Artifact: ${summary.artifact.filename} | ${summary.artifact.format} | sha256:${summary.artifact.sha256}`,
    `Projection: ${summary.projection.selected_level} | ${summary.projection.status} | Model IR sha256:${summary.source_contract.model_ir_sha256}`,
  ];
  if (summary.rows.length) {
    lines.push("ORDER\tNAME\tNATIVE TYPE\tOUTPUT CONTRACT\tBOUND STORAGE\tSERIALIZED ELEMENTS\tSERIALIZED BYTES\tMACS\tEVIDENCE");
    for (const row of summary.rows) lines.push(summaryRowCells(row).join("\t"));
  } else {
    lines.push("No operation, block, or storage row can be materialized for this artifact/profile.");
  }
  lines.push(
    `Totals: ${summary.totals.operation_count} operations | ${summary.totals.logical_value_count} logical values | ${summary.totals.storage_object_count} storage objects | ${exactText(summary.totals.unique_serialized_element_count)} serialized elements | ${exactText(summary.totals.unique_serialized_byte_length)} serialized bytes`,
    `Trainability: ${summary.trainability.status}`,
    `Summary: sha256:${summary.model_summary_sha256}`,
    `Evidence boundary: ${summary.interpretation_boundary}`,
  );
  return `${lines.join("\n")}\n`;
}

export function renderModelSummaryMarkdown(input) {
  const summary = validateModelSummary(input);
  const lines = [
    "# DEEPBOM model summary",
    "",
    `- Artifact: \`${escapeInline(summary.artifact.filename)}\` (\`${summary.artifact.format}\`)`,
    `- Artifact SHA-256: \`${summary.artifact.sha256}\``,
    `- Model IR SHA-256: \`${summary.source_contract.model_ir_sha256}\``,
    `- Projection: \`${summary.projection.selected_level}\` — ${summary.projection.status}`,
    "",
  ];
  if (summary.rows.length) {
    lines.push("| Order | Name | Native type | Output contract | Bound storage | Elements | Bytes | MACs | Evidence |", "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |");
    for (const row of summary.rows) lines.push(`| ${summaryRowCells(row).map(markdownCell).join(" | ")} |`);
  } else {
    lines.push("No operation, block, or storage row can be materialized for this artifact/profile.");
  }
  lines.push(
    "",
    `Totals: **${summary.totals.operation_count}** operations, **${summary.totals.logical_value_count}** logical values, **${summary.totals.storage_object_count}** storage objects, **${exactText(summary.totals.unique_serialized_element_count)}** serialized elements, **${exactText(summary.totals.unique_serialized_byte_length)}** serialized bytes.`,
    "",
    `Trainability: \`${summary.trainability.status}\`.`,
    "",
    `Model summary SHA-256: \`${summary.model_summary_sha256}\``,
    "",
    `> Evidence boundary: ${summary.interpretation_boundary}`,
    "",
  );
  return lines.join("\n");
}

export function compactModelSummaryForConversation(input, { maximumRows = 40 } = {}) {
  const summary = validateModelSummary(input);
  const rows = summary.rows.slice(0, maximumRows).map((row) => ({
    order: row.order.display ?? row.order.source ?? null,
    subject_ref: row.subject_ref,
    name: row.name,
    native_type: `${row.native_type.domain || ""}:${row.native_type.name || "unknown"}`,
    outputs: row.outputs.slice(0, 8).map((value) => ({ name: value.name, dtype: value.dtype, shape: value.shape, type_contract: value.type_contract || null, roles: value.roles })),
    predecessor_refs: row.predecessor_relationships.slice(0, 8).map((edge) => edge.from_ref),
    bound_storage_count: row.aggregates.storage_object_count ?? 0,
    serialized_element_count: row.aggregates.serialized_element_count || null,
    serialized_byte_length: row.aggregates.serialized_byte_length || null,
    macs: row.metrics?.macs || null,
    evidence_class: row.evidence_class,
    completeness: row.completeness,
  }));
  return {
    schema: "deepbom.model_summary_conversation.v1",
    model_summary_sha256: summary.model_summary_sha256,
    model_ir_sha256: summary.source_contract.model_ir_sha256,
    selected_level: summary.projection.selected_level,
    status: summary.projection.status,
    ordering: summary.projection.ordering,
    row_count: summary.rows.length,
    rows,
    truncated: rows.length < summary.rows.length,
    totals: summary.totals,
    trainability: summary.trainability,
    interpretation_boundary: summary.interpretation_boundary,
  };
}

function selectLevel(modelIr, requested) {
  if (requested !== "auto") return requested;
  if (modelIr.program.status === "serialized" && modelIr.program.operations.length) return "operation";
  if (modelIr.tensors_and_storage.storage_objects.length) return "storage";
  return "identity";
}

function operationRows(modelIr) {
  if (modelIr.program.status !== "serialized") return [];
  const values = new Map(modelIr.program.values.map((row) => [row.id, row]));
  const ports = new Map(modelIr.program.ports.map((row) => [row.id, row]));
  const storage = new Map(modelIr.tensors_and_storage.storage_objects.map((row) => [row.id, row]));
  const bindingsByOperation = groupBy(modelIr.weight_bindings.bindings, "operation_ref");
  const blocksByMember = new Map();
  for (const block of modelIr.program.blocks) for (const ref of block.member_refs || []) blocksByMember.set(ref, block.id);
  const incoming = new Map();
  for (const edge of modelIr.program.relationships.filter((row) => ["data_dependency", "control_dependency", "state_dependency", "call"].includes(row.kind))) {
    if (!incoming.has(edge.to_ref)) incoming.set(edge.to_ref, []);
    incoming.get(edge.to_ref).push({ relationship_ref: edge.id, from_ref: edge.from_ref, kind: edge.kind, evidence_class: edge.evidence_class });
  }
  return [...modelIr.program.operations].sort(displayOrder).map((operation) => {
    const outputValues = operation.output_port_refs.map((ref) => values.get(ports.get(ref)?.value_ref)).filter(Boolean).map(valueContract);
    const inputValues = operation.input_port_refs.map((ref) => values.get(ports.get(ref)?.value_ref)).filter(Boolean).map(valueContract);
    const bound = uniqueBy((bindingsByOperation.get(operation.id) || []).map((binding) => boundStorageContract(binding, storage.get(binding.storage_ref), modelIr)), "storage_ref");
    return {
      id: `summary-row:${operation.id}`,
      subject_ref: operation.id,
      kind: "operation",
      order: { source: operation.source_order, dependency: operation.dependency_order, display: operation.display_order, runtime: operation.runtime_order },
      name: operation.name || operation.id,
      native_type: { domain: operation.native_op.domain, name: operation.native_op.name, version: operation.native_op.version },
      semantic_family: operation.semantic_family,
      block_ref: blocksByMember.get(operation.id) || null,
      inputs: inputValues,
      outputs: outputValues,
      predecessor_relationships: (incoming.get(operation.id) || []).sort((a, b) => compareCanonicalText(a.relationship_ref, b.relationship_ref)),
      bound_storage: bound,
      aggregates: aggregateStorage(bound),
      metrics: structuredClone(operation.metrics || null),
      evidence_class: operation.evidence_class,
      completeness: operationCompleteness(outputValues, bound),
    };
  });
}

function blockRows(modelIr) {
  if (modelIr.program.status !== "serialized") return [];
  const operations = new Map(operationRows(modelIr).map((row) => [row.subject_ref, row]));
  return modelIr.program.blocks.map((block) => {
    const members = (block.member_refs || []).map((ref) => operations.get(ref)).filter(Boolean);
    const bound = uniqueBy(members.flatMap((row) => row.bound_storage), "storage_ref");
    return {
      id: `summary-row:${block.id}`,
      subject_ref: block.id,
      kind: "structural_block",
      order: { source: minimum(members.map((row) => row.order.source)), dependency: minimum(members.map((row) => row.order.dependency)), display: minimum(members.map((row) => row.order.display)), runtime: null },
      name: block.kind || block.id,
      native_type: { domain: "deepbom", name: "reversible_structural_group", version: block.grouping_rule_version },
      semantic_family: null,
      member_refs: [...block.member_refs],
      outputs: members.at(-1)?.outputs || [],
      predecessor_relationships: [],
      bound_storage: bound,
      aggregates: { ...aggregateStorage(bound), operation_count: members.length, macs: sumMetric(members, "macs") },
      metrics: { macs: sumMetric(members, "macs") },
      evidence_class: block.evidence_class,
      completeness: members.length === block.member_refs.length ? "complete_reversible_membership" : "partial_membership",
    };
  }).sort(displayOrder);
}

function storageRows(modelIr) {
  const quantBySubject = new Map(modelIr.quantization.records.map((row) => [row.subject_ref, row]));
  return modelIr.tensors_and_storage.storage_objects.map((storage, index) => ({
    id: `summary-row:${storage.id}`,
    subject_ref: storage.id,
    kind: "serialized_storage",
    order: { source: storage.native_source?.index ?? index, dependency: null, display: index, runtime: null },
    name: storage.name || storage.id,
    native_type: { domain: modelIr.format_profile.id, name: "storage_object", version: null },
    semantic_family: null,
    outputs: [{ value_ref: null, name: storage.name || storage.id, dtype: storage.dtype, shape: structuredClone(storage.shape), shape_signature: null, roles: ["serialized_storage"] }],
    predecessor_relationships: [],
    bound_storage: [storageContract(storage, quantBySubject.get(storage.id) || null)],
    aggregates: aggregateStorage([storageContract(storage, quantBySubject.get(storage.id) || null)]),
    metrics: null,
    evidence_class: storage.evidence_class,
    completeness: storage.serialized_byte_length?.decimal == null ? "partial_serialized_size_unknown" : "reported_as_serialized",
  }));
}

function buildTotals(modelIr) {
  const storage = modelIr.tensors_and_storage.storage_objects;
  const boundRefs = new Set(modelIr.weight_bindings.bindings.map((row) => row.storage_ref));
  const boundStorage = storage.filter((row) => boundRefs.has(row.id));
  const entries = new Set(modelIr.program.programs.flatMap(row => row.entry_region_refs));
  const entryOperations = modelIr.program.operations.filter(row => entries.has(row.region_ref));
  return {
    operation_count: modelIr.program.operations.length,
    structural_block_count: modelIr.program.blocks.length,
    logical_value_count: modelIr.tensors_and_storage.logical_values.length,
    storage_object_count: storage.length,
    bound_storage_object_count: boundStorage.length,
    unbound_storage_object_count: storage.length - boundStorage.length,
    unique_serialized_element_count: exactSum(storage.map((row) => row.element_count)),
    unique_serialized_byte_length: exactSum(storage.map((row) => row.serialized_byte_length)),
    bound_serialized_element_count: exactSum(boundStorage.map((row) => row.element_count)),
    bound_serialized_byte_length: exactSum(boundStorage.map((row) => row.serialized_byte_length)),
    quantization_record_count: modelIr.quantization.records.length,
    macs: sumMetric(entryOperations, "macs", modelIr.program.status === "serialized"),
    mac_scope: "entry_regions_nominal_serialized_operations_not_runtime_execution_count",
    trainable_parameter_count: null,
  };
}

function boundStorageContract(binding, storage, modelIr) {
  if (!storage) return { storage_ref: binding.storage_ref, status: "unresolved", evidence_class: "NOT_ASSESSABLE" };
  const quant = modelIr.quantization.records.find((row) => row.subject_ref === binding.value_ref || row.subject_ref === storage.id || row.related_storage_refs?.includes(storage.id));
  return { ...storageContract(storage, quant), value_ref: binding.value_ref, port_ref: binding.port_ref, structural_role: binding.parameter_role, semantic_role: binding.semantic_role, binding_status: binding.binding_status };
}

function storageContract(storage, quantization) {
  return {
    storage_ref: storage.id,
    status: "resolved",
    name: storage.name,
    dtype: storage.dtype,
    encoding: structuredClone(storage.encoding),
    shape: structuredClone(storage.shape),
    dimension_order: storage.dimension_order,
    element_count: structuredClone(storage.element_count),
    serialized_byte_length: structuredClone(storage.serialized_byte_length),
    quantization: quantization ? { record_ref: quantization.id, family: quantization.family, encoding: quantization.encoding, granularity: quantization.granularity, axis: structuredClone(quantization.axis), completeness: quantization.completeness } : null,
    parameter_classification: "serialized_bound_or_stored_value_not_trainability_claim",
    evidence_class: storage.evidence_class,
  };
}

function valueContract(value) {
  return { value_ref: value.id, name: value.name, dtype: value.dtype, shape: structuredClone(value.shape), shape_signature: structuredClone(value.shape_signature), ...(value.type_contract ? { type_contract: structuredClone(value.type_contract) } : {}), roles: structuredClone(value.roles) };
}

function aggregateStorage(rows) {
  const resolved = uniqueBy(rows.filter((row) => row.status === "resolved"), "storage_ref");
  return { storage_object_count: resolved.length, serialized_element_count: exactSum(resolved.map((row) => row.element_count)), serialized_byte_length: exactSum(resolved.map((row) => row.serialized_byte_length)) };
}

function exactSum(values) {
  const known = values.map(exactDecimal);
  const assessed = known.filter((value) => value !== null);
  if (values.length && !assessed.length) return { decimal: null, number: null, status: "not_assessable", assessed_count: 0, total_count: values.length };
  const sum = assessed.reduce((total, value) => total + BigInt(value), 0n);
  return { decimal: sum.toString(), number: sum <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(sum) : null, status: assessed.length === values.length ? "complete" : assessed.length ? "partial" : "not_assessable", assessed_count: assessed.length, total_count: values.length };
}

function sumMetric(rows, key, assessedEmpty = true) {
  if (!rows.length && !assessedEmpty) return { decimal: null, number: null, status: "not_assessable", assessed_count: 0, total_count: 0 };
  return exactSum(rows.map(row => row.metrics?.[key]));
}

function orderingContract(modelIr, level) {
  if (level === "operation" || level === "block") return { primary: "display_order", status: modelIr.program.orders.display_order_status, runtime_order_claim: false, explanation: "Deterministic topological display order with native source identity as tie-break; not an execution schedule." };
  if (level === "storage") return { primary: "serialized_storage_ledger_order", status: "complete_as_projected", runtime_order_claim: false, explanation: "Stable storage projection order; no execution order is implied." };
  return { primary: "none", status: "not_assessable", runtime_order_claim: false, explanation: "No program or storage ledger is materialized." };
}

function selectionBasis(modelIr, level) {
  if (level === "operation") return modelIr.program.status === "serialized" ? "serialized_program_operations" : "requested_program_level_not_serialized";
  if (level === "block") return modelIr.program.status === "serialized" ? "reversible_structural_blocks_over_serialized_operations" : "requested_block_level_not_serialized";
  if (level === "storage") return "serialized_storage_objects_without_synthesized_execution_graph";
  return "safe_envelope_identity_and_loss_ledger_only";
}

function columnContract(level) {
  return {
    level,
    fields: ["order", "name", "native_type", "outputs", "bound_storage", "aggregates", "metrics", "evidence_class", "completeness"],
    shape_order: "format_native_declared_order_as_recorded_by_model_ir",
    exact_integer_representation: "decimal string with optional safe JSON number",
    missing_value_policy: "null_or_explicit_not_assessable_never_zero_filled",
  };
}

function operationCompleteness(outputs, storage) {
  if (!outputs.length) return "partial_no_output_contract";
  if (storage.some((row) => row.status !== "resolved")) return "partial_unresolved_storage_binding";
  return "reported_as_serialized";
}

function validateModelSummaryBody(value) {
  if (value?.schema !== MODEL_SUMMARY_SCHEMA || !["1.0.0", MODEL_SUMMARY_METHOD_VERSION].includes(value?.method_version)) throw new Error("Model summary schema identity is invalid.");
  if (value?.source_contract?.schema !== "deepbom.model_ir.v1" || !/^[a-f0-9]{64}$/.test(String(value.source_contract.model_ir_sha256 || "")) || !/^[a-f0-9]{64}$/.test(String(value.source_contract.artifact_sha256 || ""))) throw new Error("Model summary source binding is invalid.");
  if (value?.hash_contract?.canonicalization !== "RFC8785-JCS" || JSON.stringify(value.hash_contract.excluded_pointers) !== JSON.stringify(["/model_summary_sha256"])) throw new Error("Model summary hash contract is invalid.");
  if (!Array.isArray(value.rows) || !value.projection || !value.totals || !value.coverage || !value.trainability) throw new Error("Model summary required sections are missing.");
  if (!["operation", "block", "storage", "identity"].includes(value.projection.selected_level)) throw new Error("Model summary projection level is invalid.");
  if (value.projection.ordering.runtime_order_claim !== false || value.coverage.unknown_is_zero !== false) throw new Error("Model summary evidence boundary was promoted beyond its source.");
  if (value.trainability.trainable_parameter_count !== null || value.trainability.non_trainable_parameter_count !== null) throw new Error("Model summary must not infer trainability from deployment storage.");
  if (value.method_version === MODEL_SUMMARY_METHOD_VERSION) {
    for (const [key, count] of Object.entries(value.totals).filter(([key]) => key.endsWith("_count") && !key.includes("element") && key !== "trainable_parameter_count")) {
      if (!Number.isSafeInteger(count) || count < 0) throw new Error(`Model summary count is invalid: ${key}.`);
    }
    if (value.totals.bound_storage_object_count + value.totals.unbound_storage_object_count !== value.totals.storage_object_count) throw new Error("Model summary storage counts do not reconcile.");
    for (const key of ["unique_serialized_element_count", "unique_serialized_byte_length", "bound_serialized_element_count", "bound_serialized_byte_length", "macs"]) validateAssessedSum(value.totals[key]);
    if (value.totals.mac_scope !== "entry_regions_nominal_serialized_operations_not_runtime_execution_count") throw new Error("Model summary MAC scope is invalid.");
  }
  const ids = new Set();
  for (const row of value.rows) {
    if (!row?.id || !row.subject_ref || ids.has(row.id) || !row.order || !Array.isArray(row.outputs) || !Array.isArray(row.bound_storage) || !row.evidence_class || !row.completeness) throw new Error("Model summary row contract is invalid.");
    ids.add(row.id);
  }
  if (value.projection.selected_level === "operation" && value.rows.some((row) => row.kind !== "operation")) throw new Error("Model summary operation projection contains a non-operation row.");
  if (value.projection.selected_level === "storage" && value.rows.some((row) => row.kind !== "serialized_storage")) throw new Error("Model summary storage projection contains a non-storage row.");
  if (!String(value.interpretation_boundary || "").trim()) throw new Error("Model summary interpretation boundary is missing.");
}

function validateAssessedSum(value) {
  if (!value || !Number.isSafeInteger(value.total_count) || !Number.isSafeInteger(value.assessed_count) || value.assessed_count < 0 || value.total_count < value.assessed_count) throw new Error("Model summary assessment coverage is invalid.");
  const expected = value.assessed_count === value.total_count ? "complete" : value.assessed_count ? "partial" : "not_assessable";
  if (value.status === "not_assessable" && value.assessed_count === 0 && value.decimal === null && value.number === null) return;
  if (value.status !== expected || !/^(0|[1-9][0-9]*)$/.test(value.decimal)) throw new Error("Model summary exact subtotal is invalid.");
  const exact = BigInt(value.decimal), number = exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null;
  if (value.number !== number) throw new Error("Model summary exact subtotal numeric mirror is invalid.");
}

function summaryRowCells(row) {
  const output = row.outputs.map((value) => `${value.dtype || "?"}${shapeText(value.shape, value.type_contract)}`).join(", ") || "not assessable";
  return [row.order.display ?? row.order.source ?? "-", clean(row.name), `${clean(row.native_type.domain)}:${clean(row.native_type.name)}`, output, row.aggregates.storage_object_count ?? 0, exactText(row.aggregates.serialized_element_count), exactText(row.aggregates.serialized_byte_length), exactText(row.metrics?.macs), row.evidence_class];
}

function exactText(value) { return value?.status === "not_assessable" || value?.decimal == null ? "N/A" : value.status === "partial" ? `${value.decimal} (partial)` : value.decimal; }
function exactDecimal(value) { const raw = value && typeof value === "object" ? value.decimal : value; return /^\d+$/.test(String(raw ?? "")) ? String(raw) : null; }
function displayOrder(a, b) { return Number(a.order?.display ?? a.display_order ?? Number.MAX_SAFE_INTEGER) - Number(b.order?.display ?? b.display_order ?? Number.MAX_SAFE_INTEGER) || compareCanonicalText(a.subject_ref || a.id, b.subject_ref || b.id); }
function minimum(values) { const rows = values.filter(Number.isSafeInteger); return rows.length ? Math.min(...rows) : null; }
function groupBy(rows, key) { const result = new Map(); for (const row of rows) { const id = row[key]; if (!result.has(id)) result.set(id, []); result.get(id).push(row); } return result; }
function uniqueBy(rows, key) { const seen = new Set(); return rows.filter((row) => { const id = row[key]; if (seen.has(id)) return false; seen.add(id); return true; }); }
function shapeText(shape, type) { if (type?.root?.rank_status === "ranked" && type.root.dimensions.length === 0) return "[]"; if (type && !["tensor", "sparse_tensor"].includes(type.root.kind)) return `<${type.root.kind}>`; return Array.isArray(shape) && shape.length ? `[${shape.map((value) => value == null ? "?" : value).join(",")}]` : "[?]"; }
function clean(value) { return String(value ?? "").replace(/[\t\r\n|]+/g, " ").trim() || "-"; }
function markdownCell(value) { return String(value ?? "").replaceAll("|", "\\|").replace(/[\r\n]+/g, " "); }
function escapeInline(value) { return String(value ?? "").replaceAll("`", "\\`"); }
function deepFreeze(value) { if (!value || typeof value !== "object" || Object.isFrozen(value)) return value; Object.freeze(value); for (const child of Object.values(value)) deepFreeze(child); return value; }
