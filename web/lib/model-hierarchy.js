import { validateModelIr } from "./model-ir.js";
import { buildModelSummary, summarizeModelMembers } from "./model-summary.js";
import { indexModelProgram, partitionConnectedOperations, MODEL_STRUCTURAL_GROUP_LIMIT } from "./model-program-boundary.js";
import { canonicalJson, compareCanonicalText } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

// A versioned Model IR projection with source annotation evidence. This is not
// another Evidence IR family member and does not alter the frozen source IR.
export const MODEL_HIERARCHY_SCHEMA = "deepbom.model_hierarchy.v1";
export const MODEL_HIERARCHY_VIEW_SCHEMA = "deepbom.model_hierarchy_view.v1";
const HASH = /^[a-f0-9]{64}$/;
const clone = value => structuredClone(value);
const verified = new WeakMap();
const defaultPresentations = new WeakMap(), fallbackHierarchies = new WeakMap();
const seal = (body, field) => freeze({ ...body, [field]: sha256TextHex(canonicalJson(body)) });
const BOUNDARY = "Source paths/classes are exporter declarations observed in the artifact, not validated native module objects or runtime calls. Groups preserve operation membership and port-level edges. Shared serialized storage is not proof of shared trainable parameters. Reachability is not measured influence; MACs are not latency. Missing evidence is not zero.";

export function buildModelHierarchy(input, { annotations = [] } = {}) {
  const model = validateModelIr(input);
  validateAnnotations(annotations, model);
  const byAnnotation = new Map(annotations.map(row => [row.subject_ref, row]));
  const index = indexModelProgram(model);
  const rows = buildModelSummary(model, { level: "operation" }).rows;
  // Root functional operations can occur before and after the network. Group
  // only connected root operations; merging them all creates false-looking
  // cycles in the collapsed presentation (without changing the real DAG).
  const rootOps = new Set(annotations.filter(row => row.status === "observed_exporter_annotation"
    && sourceScopeCount(row) === 0).map(row => row.subject_ref));
  const adjacent = new Map([...rootOps].map(ref => [ref, []]));
  for (const edge of index.edges) if (rootOps.has(edge.from_ref) && rootOps.has(edge.to_ref)) {
    adjacent.get(edge.from_ref).push(edge.to_ref); adjacent.get(edge.to_ref).push(edge.from_ref);
  }
  const rootComponent = new Map();
  for (const ref of rootOps) {
    if (rootComponent.has(ref)) continue;
    const queue = [ref]; rootComponent.set(ref, ref);
    for (let i = 0; i < queue.length; i++) for (const next of adjacent.get(queue[i])) if (!rootComponent.has(next)) { rootComponent.set(next, ref); queue.push(next); }
  }
  const groups = new Map(), assignments = [];
  const make = (id, parent, path, kind, region) => {
    if (!groups.has(id)) groups.set(id, { id, parent_ref: parent, path, kind, region_ref: region,
      class_names: [], child_refs: [], direct_member_refs: [], member_refs: [] });
    return groups.get(id);
  };
  const annotated = annotations.some(row => row.status === "observed_exporter_annotation");
  const blockByOp = new Map(model.program.blocks.flatMap(row => row.member_refs.map(ref => [ref, row])));
  const usableBlock = op => {
    const block = !annotated ? blockByOp.get(op.id) : null;
    return block && block.member_refs.length <= MODEL_STRUCTURAL_GROUP_LIMIT
      && !["head", "tail"].includes(block.kind) ? block : null;
  };
  const fallbackByOp = new Map();
  for (const region of model.program.regions) {
    const missing = model.program.operations.filter(op => op.region_ref === region.id
      && byAnnotation.get(op.id)?.status !== "observed_exporter_annotation" && !usableBlock(op))
      .sort((a, b) => a.display_order - b.display_order || compareCanonicalText(a.id, b.id));
    const parts = partitionConnectedOperations(missing, index.edges);
    for (const [partIndex, members] of parts.entries()) for (const op of members) fallbackByOp.set(op.id, {
      id: `hierarchy:${annotated ? "unclassified" : "structural"}:${region.id}:${partIndex}`,
      label: `${annotated ? "Unclassified" : "Connected structural group"} ${partIndex + 1} (${members.length} operations)`,
    });
  }
  for (const op of [...model.program.operations].sort((a, b) => a.display_order - b.display_order || compareCanonicalText(a.id, b.id))) {
    const root = make(`hierarchy:region:${op.region_ref}`, null,
      model.program.regions.find(row => row.id === op.region_ref)?.name || op.region_ref, "graph_region", op.region_ref);
    const annotation = byAnnotation.get(op.id);
    const chain = [root];
    let status = annotation?.status || "unavailable", callSite = null;
    if (annotation?.status === "observed_exporter_annotation") {
      const names = annotation.scope_names, classes = annotation.class_names;
      let last = names.length;
      if (classes.length === names.length && /^(aten|prims|_operator|operator|builtins)\./.test(classes.at(-1))) {
        callSite = names.at(-1); last--;
      }
      for (let i = 0; i < last; i++) {
        if (!names[i]) continue;
        const id = `hierarchy:source:${sha256TextHex(canonicalJson([op.region_ref, names.slice(0, i + 1)]))}`;
        const group = make(id, chain.at(-1).id, names[i], "source_scope", op.region_ref);
        if (classes[i] && !group.class_names.includes(classes[i])) group.class_names.push(classes[i]);
        chain.push(group);
      }
      if (chain.length === 1) chain.push(make(`hierarchy:functional:${rootComponent.get(op.id) || op.id}`, root.id,
        "Root functional operations", "root_functional", op.region_ref));
    } else {
      const block = usableBlock(op);
      const fallback = fallbackByOp.get(op.id);
      chain.push(block
        ? make(`hierarchy:structural:${block.id}`, root.id, `${block.kind === "repeated_structure" ? "Repeated structure" : "Connected operations"} (${block.member_refs.length} operations)`, "derived_structural_group", op.region_ref)
        : make(fallback.id, root.id, fallback.label, annotated ? "unclassified" : "derived_structural_group", op.region_ref));
    }
    for (let i = 0; i < chain.length; i++) {
      chain[i].member_refs.push(op.id);
      if (i && !chain[i - 1].child_refs.includes(chain[i].id)) chain[i - 1].child_refs.push(chain[i].id);
    }
    chain.at(-1).direct_member_refs.push(op.id);
    assignments.push({ subject_ref: op.id, group_ref: chain.at(-1).id, status,
      call_site_name: callSite, call_identity_status: annotation?.call_identity_status || "not_exposed",
      source_pointer: annotation?.source_pointer || null, metadata_sha256: annotation?.metadata_sha256 || null,
      reason: annotation?.reason || null });
  }
  // A shallow exporter scope may contain hundreds of direct operations. Keep
  // that declared scope, but make expansion navigable through derived children.
  const assignmentByOp = new Map(assignments.map(row => [row.subject_ref, row]));
  for (const group of groups.values()) {
    if (group.direct_member_refs.length <= MODEL_STRUCTURAL_GROUP_LIMIT) continue;
    const members = group.direct_member_refs.map(ref => index.operations.get(ref));
    for (const [partIndex, part] of partitionConnectedOperations(members, index.edges).entries()) {
      const child = make(`hierarchy:detail:${group.id}:${partIndex}`, group.id,
        `Connected operations ${partIndex + 1} (${part.length} operations)`, "derived_structural_group", group.region_ref);
      child.member_refs = part.map(op => op.id); child.direct_member_refs = [...child.member_refs];
      group.child_refs.push(child.id);
      for (const ref of child.member_refs) assignmentByOp.get(ref).group_ref = child.id;
    }
    group.direct_member_refs = [];
  }
  const materialized = [...groups.values()].map(group => ({ ...group,
    class_names: group.class_names.sort(compareCanonicalText),
    aggregation: summarizeModelMembers(model, group.member_refs, { rows, index }),
  }));
  const otherRelationships = model.program.relationships.filter(row => ["control_dependency", "state_dependency", "call"].includes(row.kind));
  const body = {
    schema: MODEL_HIERARCHY_SCHEMA, method_version: annotations.some(row=>row.source_pointer.endsWith('/source_annotation')) ? "1.1.0" : "1.0.0",
    source: { model_ir_schema: model.schema, model_ir_sha256: model.model_ir_sha256,
      artifact_ir_sha256: model.source_contract.sha256, artifact_sha256: model.artifact.sha256 },
    status: model.program.operations.length ? annotated ? "source_annotated" : "structural_fallback" : "not_applicable_no_program",
    annotations: clone(annotations), groups: materialized, assignments,
    non_data_relationships: clone(otherRelationships),
    edges: index.edges.map(edge => ({ ...edge, shape: clone(index.values.get(edge.value_ref).shape),
      dtype: index.values.get(edge.value_ref).dtype, layout: "not_inferred" })),
    coverage: { operation_count: model.program.operations.length,
      annotated_operation_count: assignments.filter(row => row.status === "observed_exporter_annotation").length,
      invalid_annotation_count: assignments.filter(row => row.status === "invalid_annotation").length,
      unannotated_operation_count: assignments.filter(row => row.status === "unavailable").length,
      operator_edge_count: index.edges.filter(edge => edge.from_ref && edge.to_ref).length,
      boundary_edge_count: index.edges.filter(edge => !edge.from_ref || !edge.to_ref).length,
      non_data_relationship_count: otherRelationships.length,
      runtime_call_identity_assessed: false, unknown_is_zero: false },
    interpretation_boundary: BOUNDARY,
  };
  const result = seal(body, "hierarchy_sha256");
  verified.set(result, model.model_ir_sha256);
  return result;
}

// Default presentation shared by Web, CLI human summaries and document exports.
// Machine-readable operation/storage ledgers keep their existing contracts.
export function resolveModelHierarchyPresentation(input, suppliedHierarchy = null) {
  const model = validateModelIr(input);
  if (suppliedHierarchy) suppliedHierarchy = validateModelHierarchy(suppliedHierarchy, model);
  if (!model.program.operations.length) return null;
  let hierarchy = suppliedHierarchy || fallbackHierarchies.get(model);
  if (!hierarchy) { hierarchy = buildModelHierarchy(model); fallbackHierarchies.set(model, hierarchy); }
  if (!defaultPresentations.has(hierarchy)) defaultPresentations.set(hierarchy, Object.freeze({
    hierarchy, view: projectModelHierarchy(model, hierarchy),
  }));
  return defaultPresentations.get(hierarchy);
}

function sourceScopeCount(annotation) {
  const { scope_names: names, class_names: classes } = annotation;
  const call = classes.length === names.length && /^(aten|prims|_operator|operator|builtins)\./.test(classes.at(-1));
  return names.slice(0, call ? -1 : undefined).filter(Boolean).length;
}

export function validateModelHierarchy(document, model) {
  model = validateModelIr(model);
  if (verified.get(document) === model?.model_ir_sha256) return document;
  canonicalJson(document);
  const { hierarchy_sha256, ...body } = clone(document || {});
  if (body.schema !== MODEL_HIERARCHY_SCHEMA || !HASH.test(hierarchy_sha256 || "")
    || hierarchy_sha256 !== sha256TextHex(canonicalJson(body))) throw new Error("Model hierarchy digest or schema is invalid.");
  if (!model) throw new Error("Model hierarchy validation requires its Model IR source.");
  const expected = buildModelHierarchy(model, { annotations: body.annotations });
  if (canonicalJson(expected) !== canonicalJson(document)) throw new Error("Model hierarchy differs from its Model IR and source annotations.");
  return expected;
}

export function projectModelHierarchy(model, hierarchy, { expanded = [], selected = null, search = "", focus = null } = {}) {
  // Callers retain the immutable, validated context. Imports must first use
  // validateModelHierarchy; the binding check also prevents cross-model views.
  hierarchy = validateModelHierarchy(hierarchy, model);
  if (hierarchy.source.model_ir_sha256 !== model.model_ir_sha256 || hierarchy.source.artifact_sha256 !== model.artifact.sha256) throw new Error("Hierarchy view source binding mismatch.");
  const groups = new Map(hierarchy.groups.map(row => [row.id, row]));
  const operations = new Map(model.program.operations.map(row => [row.id, row]));
  const expandedSet = new Set(expanded);
  if (expandedSet.size !== expanded.length || expanded.some(ref => !groups.has(ref))) throw new Error("Unknown or duplicated expanded group.");
  if (selected !== null && !groups.has(selected) && !operations.has(selected)) throw new Error("Unknown selected hierarchy subject.");
  if (typeof search !== "string" || search.length > 256) throw new Error("Invalid hierarchy search.");
  const selectedMembers = selected === null ? [...operations.keys()] : groups.get(selected)?.member_refs || [selected];
  const included = new Set(selectedMembers), visible = [], owner = new Map();
  const q = search.toLowerCase();
  const ancestors = new Set();
  let ancestor = groups.get(selected)?.parent_ref || hierarchy.assignments.find(row => row.subject_ref === selected)?.group_ref;
  while (ancestor) { ancestors.add(ancestor); ancestor = groups.get(ancestor)?.parent_ref; }
  const searchMembers = new Set();
  if (q) {
    for (const group of groups.values()) if ([group.path, ...group.class_names].some(text => text.toLowerCase().includes(q))) group.member_refs.forEach(ref => searchMembers.add(ref));
    for (const op of operations.values()) if (`${op.name} ${op.native_op.name} ${op.id}`.toLowerCase().includes(q)) searchMembers.add(op.id);
  }
  const visit = group => {
    const members = group.member_refs.filter(ref => included.has(ref) && (!q || searchMembers.has(ref)));
    if (!members.length) return;
    const open = group.kind === "graph_region" || expandedSet.has(group.id)
      || ancestors.has(group.id) || (q && members.length < group.member_refs.length);
    if (open) {
      for (const ref of group.direct_member_refs.filter(ref => included.has(ref) && (!q || searchMembers.has(ref)))) {
        const op = operations.get(ref);
        visible.push({ id: ref, kind: "operation", label: op.name, native_type: op.native_op.name, member_refs: [ref] }); owner.set(ref, ref);
      }
      group.child_refs.forEach(ref => visit(groups.get(ref)));
    } else {
      visible.push({ id: group.id, kind: "group", label: group.path, native_type: group.class_names.join(" / "), member_refs: members });
      members.forEach(ref => owner.set(ref, group.id));
    }
  };
  hierarchy.groups.filter(row => row.parent_ref === null).forEach(visit);
  const bundles = new Map(), internal = [], omitted = [];
  for (const edge of hierarchy.edges) {
    const from = owner.get(edge.from_ref), to = owner.get(edge.to_ref);
    if (!from && !to) { omitted.push(edge.id); continue; }
    if (from && from === to) { internal.push(edge.id); continue; }
    const key = canonicalJson([from || null, to || null]);
    if (!bundles.has(key)) bundles.set(key, { id: `bundle:${sha256TextHex(key)}`, from_ref: from || null, to_ref: to || null, edge_refs: [] });
    bundles.get(key).edge_refs.push(edge.id);
  }
  const nodeRefs = visible.flatMap(row => row.member_refs);
  const edgeRefs = [...internal, ...omitted, ...[...bundles.values()].flatMap(row => row.edge_refs)];
  const expectedEdges = new Set(hierarchy.edges.map(edge => edge.id));
  if (new Set(nodeRefs).size !== nodeRefs.length || new Set(edgeRefs).size !== edgeRefs.length
    || edgeRefs.length !== hierarchy.edges.length || edgeRefs.some(ref => !expectedEdges.has(ref))) throw new Error("Hierarchy projection conservation failed.");
  let highlighted = [];
  if (focus !== null) {
    if (!operations.has(focus)) throw new Error("Path focus must be an operation reference.");
    const reached = new Set([focus]), incoming = new Map();
    for (const edge of hierarchy.edges) if (edge.from_ref && edge.to_ref) {
      if (!incoming.has(edge.to_ref)) incoming.set(edge.to_ref, []);
      incoming.get(edge.to_ref).push(edge.from_ref);
    }
    const queue = [focus];
    while (queue.length) for (const ref of incoming.get(queue.pop()) || []) if (!reached.has(ref)) { reached.add(ref); queue.push(ref); }
    highlighted = [...reached].sort(compareCanonicalText);
  }
  const index = indexModelProgram(model), summary = buildModelSummary(model, { level: "operation" });
  const rows = visible.map(row => ({ ...row, aggregation: groups.get(row.id)?.member_refs.length === row.member_refs.length
    ? groups.get(row.id).aggregation : summarizeModelMembers(model, row.member_refs, { rows: summary.rows, index }) }));
  return seal({ schema: MODEL_HIERARCHY_VIEW_SCHEMA, method_version: "1.0.0",
    source: clone(hierarchy.source), hierarchy_sha256: hierarchy.hierarchy_sha256,
    selection: { selected, expanded: [...expanded], search, focus }, rows, edges: [...bundles.values()],
    non_data_relationship_refs: hierarchy.non_data_relationships.map(row => row.id),
    highlighted_operation_refs: highlighted,
    conservation: { status: "conserved", total_operations: operations.size, represented_operations: nodeRefs.length,
      omitted_operation_refs: [...operations.keys()].filter(ref => !owner.has(ref)),
      total_edges: hierarchy.edges.length, internal_edge_refs: internal, omitted_edge_refs: omitted,
      boundary_policy: "all_incident_edges_retained_even_when_opposite_endpoint_is_outside_view" },
    interpretation_boundary: BOUNDARY,
  }, "view_sha256");
}

export function queryModelHierarchy(model, hierarchy, { parent_ref = null, offset = 0, limit = 40, search = "" } = {}) {
  hierarchy = validateModelHierarchy(hierarchy, model);
  if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100 || typeof search !== "string" || search.length > 256) throw new Error("Invalid hierarchy query bounds.");
  if (hierarchy.source.model_ir_sha256 !== model.model_ir_sha256) throw new Error("Hierarchy query source mismatch.");
  if (parent_ref !== null && !hierarchy.groups.some(row => row.id === parent_ref)) throw new Error("Unknown hierarchy parent.");
  const all = hierarchy.groups.filter(row => parent_ref === null || row.parent_ref === parent_ref);
  const matched = all.filter(row => `${row.path} ${row.class_names.join(" ")}`.toLowerCase().includes(search.toLowerCase()));
  const rows = matched.slice(offset, offset + limit).map(group => ({
    id: group.id, parent_ref: group.parent_ref, path: group.path, kind: group.kind, class_names: group.class_names,
    operation_count: group.member_refs.length, child_count: group.child_refs.length,
    macs: group.aggregation.macs, mac_assessment: group.aggregation.mac_assessment,
    serialized_element_count: group.aggregation.serialized_element_count,
    serialized_byte_length: group.aggregation.serialized_byte_length,
    trainable_parameter_count: null,
    input_value_count: group.aggregation.inputs.length, output_value_count: group.aggregation.outputs.length,
    details_pointer: `/model_hierarchy/groups/${hierarchy.groups.indexOf(group)}`,
  }));
  return { schema: "deepbom.model_hierarchy_query.v1", source: clone(hierarchy.source), hierarchy_sha256: hierarchy.hierarchy_sha256,
    query: { parent_ref, offset, limit, search }, total: all.length, matched: matched.length, returned: rows.length,
    next_offset: offset + rows.length < matched.length ? offset + rows.length : null, rows: clone(rows), interpretation_boundary: BOUNDARY };
}

function validateAnnotations(rows, model) {
  if (!Array.isArray(rows) || rows.length > model.program.operations.length) throw new Error("Invalid source annotation count.");
  const ids = new Set(model.program.operations.map(row => row.id)), seen = new Set();
  const keys = ["subject_ref", "scope_names", "class_names", "status", "reason", "source_pointer", "metadata_sha256", "call_identity_status"];
  for (const row of rows) {
    if (Object.keys(row).length !== keys.length || keys.some(key => !Object.hasOwn(row, key)) || !ids.has(row.subject_ref) || seen.has(row.subject_ref)) throw new Error("Invalid source annotation identity or fields.");
    seen.add(row.subject_ref);
    if (!["observed_exporter_annotation", "invalid_annotation"].includes(row.status)
      || !HASH.test(row.metadata_sha256 || "") || typeof row.source_pointer !== "string" || !/^\/ops\/\d+\/(onnx_metadata_properties|source_annotation)$/.test(row.source_pointer) || row.source_pointer.length > 128
      || row.call_identity_status !== "exporter_annotation_not_runtime_invocation"
      || !Array.isArray(row.scope_names) || !Array.isArray(row.class_names)
      || row.scope_names.length > 128 || row.class_names.length > 128
      || [...row.scope_names, ...row.class_names].some(s => typeof s !== "string" || s.length > 4096 || !s.isWellFormed() || /[\u0000-\u001f\u007f]/.test(s))) throw new Error("Invalid source annotation contract.");
    if (row.status === "observed_exporter_annotation" && (!row.scope_names.length || row.reason !== null
      || (row.class_names.length && row.class_names.length !== row.scope_names.length))) throw new Error("Inconsistent source annotation scope.");
    if (row.status === "invalid_annotation" && (row.scope_names.length || row.class_names.length || typeof row.reason !== "string" || !row.reason || row.reason.length > 256)) throw new Error("Invalid annotation must retain its failure reason.");
  }
}

function freeze(value) { if (value && typeof value === "object" && !Object.isFrozen(value)) { Object.values(value).forEach(freeze); Object.freeze(value); } return value; }
