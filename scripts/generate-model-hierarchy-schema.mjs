import { readFile, writeFile } from "node:fs/promises";

// JSON shape validation complements source rederivation in model-hierarchy.js.
const obj = properties => ({ type: "object", additionalProperties: false, required: Object.keys(properties), properties });
const ref = key => ({ $ref: `#/$defs/${key}` });
const list = items => ({ type: "array", items });
const str = { type: "string" }, count = { type: "integer", minimum: 0 };
const nullable = schema => ({ anyOf: [schema, { type: "null" }] });
const refs = { ...list(str), uniqueItems: true };
const summary = JSON.parse(await readFile("docs/schemas/deepbom-model-summary-v1.schema.json", "utf8"));
const defs = Object.fromEntries(["sha256", "nonEmpty", "nullableInteger", "nullableString", "exactInteger", "assessedExactInteger", "value"].map(key => [key, summary.$defs[key]]));
defs.source = obj({ model_ir_schema: { const: "deepbom.model_ir.v1" }, model_ir_sha256: ref("sha256"), artifact_ir_sha256: ref("sha256"), artifact_sha256: ref("sha256") });
defs.annotation = obj({ subject_ref: str, scope_names: { ...list({ ...str, maxLength: 4096 }), maxItems: 128 }, class_names: { ...list({ ...str, maxLength: 4096 }), maxItems: 128 },
  status: { enum: ["observed_exporter_annotation", "invalid_annotation"] }, reason: nullable({ ...str, maxLength: 256 }), source_pointer: { ...str, maxLength: 128 }, metadata_sha256: ref("sha256"), call_identity_status: { const: "exporter_annotation_not_runtime_invocation" } });
defs.boundary = obj({ input_value_refs: refs, output_value_refs: refs, internal_edge_refs: refs, incoming_edge_refs: refs, outgoing_edge_refs: refs });
defs.assessment = { type: "object", additionalProperties: count };
defs.aggregation = obj({ operation_count: count, storage_object_count: count, serialized_element_count: ref("assessedExactInteger"), serialized_byte_length: ref("assessedExactInteger"), storage_refs: refs,
  macs: ref("assessedExactInteger"), mac_assessment: ref("assessment"), inputs: list(ref("value")), outputs: list(ref("value")), boundary: ref("boundary"), trainable_parameter_count: { type: "null" } });
defs.group = obj({ id: str, parent_ref: nullable(str), path: str, kind: { enum: ["graph_region", "source_scope", "root_functional", "derived_structural_group", "unclassified"] }, region_ref: str,
  class_names: refs, child_refs: refs, direct_member_refs: refs, member_refs: refs, aggregation: ref("aggregation") });
defs.assignment = obj({ subject_ref: str, group_ref: str, status: { enum: ["observed_exporter_annotation", "invalid_annotation", "unavailable"] }, call_site_name: nullable(str),
  call_identity_status: { enum: ["exporter_annotation_not_runtime_invocation", "not_exposed"] }, source_pointer: nullable(str), metadata_sha256: nullable(ref("sha256")), reason: nullable(str) });
defs.edge = obj({ id: str, value_ref: str, from_ref: nullable(str), to_ref: nullable(str), from_port_ref: nullable(str), to_port_ref: nullable(str),
  from_port_position: nullable(count), to_port_position: nullable(count), boundary_program_ref: nullable(str), boundary_position: nullable(count),
  kind: { enum: ["data_dependency", "storage_input", "external_input", "model_output"] }, shape: summary.$defs.value.properties.shape, dtype: nullable(str), layout: { const: "not_inferred" } });
const base = { method_version: { const: "1.0.0" }, source: ref("source"), interpretation_boundary: str };
defs.hierarchy = obj({ schema: { const: "deepbom.model_hierarchy.v1" }, ...base, status: { enum: ["source_annotated", "structural_fallback", "not_applicable_no_program"] },
  annotations: list(ref("annotation")), groups: list(ref("group")), assignments: list(ref("assignment")), edges: list(ref("edge")), hierarchy_sha256: ref("sha256"),
  non_data_relationships: list({ type: "object", required: ["id", "kind"], properties: { id: str, kind: { enum: ["control_dependency", "state_dependency", "call"] } } }),
  coverage: obj({ operation_count: count, annotated_operation_count: count, invalid_annotation_count: count, unannotated_operation_count: count, operator_edge_count: count, boundary_edge_count: count, non_data_relationship_count: count, runtime_call_identity_assessed: { const: false }, unknown_is_zero: { const: false } }) });
defs.view = obj({ schema: { const: "deepbom.model_hierarchy_view.v1" }, ...base, hierarchy_sha256: ref("sha256"), view_sha256: ref("sha256"),
  selection: obj({ selected: nullable(str), expanded: refs, search: { ...str, maxLength: 256 }, focus: nullable(str) }),
  rows: list(obj({ id: str, kind: { enum: ["group", "operation"] }, label: str, native_type: str, member_refs: refs, aggregation: ref("aggregation") })),
  edges: list(obj({ id: str, from_ref: nullable(str), to_ref: nullable(str), edge_refs: refs })), highlighted_operation_refs: refs, non_data_relationship_refs: refs,
  conservation: obj({ status: { const: "conserved" }, total_operations: count, represented_operations: count, omitted_operation_refs: refs, total_edges: count,
    internal_edge_refs: refs, omitted_edge_refs: refs, boundary_policy: { const: "all_incident_edges_retained_even_when_opposite_endpoint_is_outside_view" } }) });
defs.query = obj({ schema: { const: "deepbom.model_hierarchy_query.v1" }, source: ref("source"), hierarchy_sha256: ref("sha256"), interpretation_boundary: str,
  query: obj({ parent_ref: nullable(str), offset: count, limit: { ...count, minimum: 1, maximum: 100 }, search: { ...str, maxLength: 256 } }), total: count, matched: count, returned: count, next_offset: nullable(count),
  rows: list(obj({ id: str, parent_ref: nullable(str), path: str, kind: str, class_names: refs, operation_count: count, child_count: count,
    macs: ref("assessedExactInteger"), mac_assessment: ref("assessment"), serialized_element_count: ref("assessedExactInteger"), serialized_byte_length: ref("assessedExactInteger"),
    trainable_parameter_count: { type: "null" }, input_value_count: count, output_value_count: count, details_pointer: str })) });
const schema = { $schema: "https://json-schema.org/draft/2020-12/schema", $id: "https://deepbom.org/schemas/deepbom-model-hierarchy-v1.schema.json",
  title: "Model IR source hierarchy projections", description: "Validate shape here, then rederive against Model IR with validateModelHierarchy. A valid digest is not exporter authenticity. No independent IR family or runtime claim.",
  oneOf: [ref("hierarchy"), ref("view"), ref("query")], $defs: defs };
const filename = "docs/schemas/deepbom-model-hierarchy-v1.schema.json", text = JSON.stringify(schema, null, 2) + "\n";
if (process.argv.includes("--check")) { if (await readFile(filename, "utf8") !== text) throw Error("Model hierarchy schema is stale"); }
else await writeFile(filename, text);
console.log("Model hierarchy JSON Schema is current.");
