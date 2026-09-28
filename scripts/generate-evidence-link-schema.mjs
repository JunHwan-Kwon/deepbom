import { readFile, writeFile } from "node:fs/promises";
import { LINK_INPUT_SCHEMA, OMOP_INPUT_SCHEMA, LINK_IR_SCHEMA, LINK_LIMITS, NODE_KINDS, RELATION_ROLES } from "../web/lib/evidence-links/contracts.js";
import { EVIDENCE_LINK_SUMMARY_SCHEMA } from "../web/lib/evidence-links/summary.js";
const text = { type: "string", maxLength: LINK_LIMITS.text };
const nonempty = { ...text, minLength: 1, pattern: "\\S" };
const id = { type: "string", maxLength: 256, pattern: "^[a-zA-Z0-9][a-zA-Z0-9._:/#-]*$" };
const hash = { type: "string", pattern: "^[a-f0-9]{64}$" };
const nullable = value => ({ anyOf: [value, { type: "null" }] });
const integer = { type: "integer", minimum: 0, maximum: Number.MAX_SAFE_INTEGER };
const scalar = { anyOf: [text, { type: "boolean" }, { type: "null" }, { type: "number", minimum: -Number.MAX_SAFE_INTEGER, maximum: Number.MAX_SAFE_INTEGER }] };
const array = (items, maxItems) => ({ type: "array", items, maxItems });
const object = (properties, required = Object.keys(properties)) => ({ type: "object", properties, required, additionalProperties: false });
const attribute = object({ name: { ...nonempty, maxLength: 256 }, value: scalar });
const bom = object({ format: nonempty, spec_version: nonempty, document_id: nonempty, document_version: nullable({ ...integer, minimum: 1 }), element_ref: nonempty });
const nodeFields = { id, kind: { enum: NODE_KINDS }, name: nonempty, version: nonempty, uri: nonempty, sha256: hash, file: { type: "string", maxLength: 1024, pattern: "^(?!/)(?!.*\\\\)(?!.*:)(?!.*(?:^|/)\\.{1,2}(?:/|$))[^/]+(?:/[^/]+)*$" }, attributes: array(attribute, LINK_LIMITS.attributes), bom };
const node = object(nodeFields, ["id", "kind", "name"]);
const edgeFields = { id, from: id, to: id, role: { enum: RELATION_ROLES }, evidence_refs: { ...array(id, LINK_LIMITS.nodes), uniqueItems: true } };
const edge = object(edgeFields, ["id", "from", "to", "role"]);
const source = object({ artifact_sha256: hash, artifact_set_sha256: nullable(hash), model_ir_sha256: hash });
const subject = object(source.properties, ["artifact_sha256", "artifact_set_sha256"]);
const cdmRow = { type: "object", maxProperties: LINK_LIMITS.attributes, propertyNames: { type: "string", pattern: "^[a-zA-Z][a-zA-Z0-9_]*$", maxLength: 128 }, additionalProperties: scalar };
const omop = object({ cdm_version: { ...nonempty, maxLength: 256 }, instance_id: { ...nonempty, maxLength: 256 }, release_id: { ...nonempty, maxLength: 256 }, cdm_source: { anyOf: [cdmRow, { ...array(cdmRow, 1), minItems: 1 }] } });
const input = schema => object({ schema: { const: schema }, subject, nodes: array(node, LINK_LIMITS.nodes), relationships: array(edge, LINK_LIMITS.relationships), ...(schema === OMOP_INPUT_SCHEMA ? { omop } : {}) });
const inputUnion = { oneOf: [input(LINK_INPUT_SCHEMA), input(OMOP_INPUT_SCHEMA)] };
const observation = object({ node_ref: id, sha256: hash, byte_length: { ...integer, maximum: LINK_LIMITS.file_bytes }, document: { type: "object" } }, ["node_ref", "sha256", "byte_length"]);
const check = object({ subject_ref: id, kind: nonempty, status: { enum: ["match", "mismatch", "not_assessed", "unresolved", "unsupported"] }, message: nonempty });
const field = object({ source_document: { enum: ["input_document", "evidence_link_ir"] }, source_pointer: nonempty, subject_ref: id, name: text, value: {}, status: { enum: ["mapped_declared", "preserved_unsupported_mapping"] } });
const coverageKeys = ["node_count", "relationship_count", "observed_file_count", "check_count", "field_count", "mapped_declared_field_count", "unsupported_field_count", "attested_relationship_count", "check_counts"];
const coverage = object(Object.fromEntries(coverageKeys.map(key => [key, EVIDENCE_LINK_SUMMARY_SCHEMA.properties[key]])));
const irNode = object({ ...nodeFields, attributes: array(attribute, LINK_LIMITS.attributes + 3), evidence_class: { enum: ["OBSERVED_SERIALIZED_ARTIFACT", "DECLARED_UNVERIFIED"] } }, ["id", "kind", "name", "attributes", "evidence_class"]);
const irEdge = object({ ...edgeFields, evidence_class: { const: "DECLARED_UNVERIFIED" }, resolution: { enum: ["resolved", "unresolved", "kind_mismatch"] }, claim_verification: { const: "not_assessed" } });
const ir = object({
  schema: { const: LINK_IR_SCHEMA }, method_version: { const: "1.0.0" },
  hash_contract: object({ algorithm: { const: "SHA-256" }, canonicalization: { const: "RFC8785-JCS" }, source_encoding: { const: "UTF-8" }, excluded_pointers: { const: ["/evidence_link_ir_sha256"] } }),
  source, input_document: inputUnion, input_sha256: hash,
  profile: object({ id: { enum: ["deepbom.omop_metadata_profile.v1", "deepbom.generic_metadata_profile.v1"] }, status: { enum: ["supported_import_profile", "unsupported_cdm_version"] }, cdm_version: nonempty, missing_source_fields: array(nonempty, LINK_LIMITS.attributes) }, ["id", "status"]),
  nodes: array(irNode, LINK_LIMITS.nodes + 2), relationships: array(irEdge, LINK_LIMITS.relationships + 1), observations: array(observation, LINK_LIMITS.nodes), checks: array(check, 1000000), field_ledger: array(field, 1000000), coverage,
  verdict: object({ status: EVIDENCE_LINK_SUMMARY_SCHEMA.properties.status, scope: { const: "binding_reference_and_supplied_file_consistency" }, metadata_truth_verified: { const: false }, publisher_authenticity: { const: "not_verified" } }),
  interpretation_boundary: nonempty, evidence_link_ir_sha256: hash,
});
const document = { $schema: "https://json-schema.org/draft/2020-12/schema", $id: "https://deepbom.org/schemas/deepbom-evidence-links-v1.schema.json", title: "DEEPBOM optional metadata input, connection IR and bounded host summary", description: "Structural contract only. Semantic validators additionally enforce exact source binding, role compatibility, hash recomputation, coverage conservation, case collisions, file boundaries and byte limits. This is a DEEPBOM contract, not an OMOP or CycloneDX standard extension.", oneOf: [input(LINK_INPUT_SCHEMA), input(OMOP_INPUT_SCHEMA), ir, EVIDENCE_LINK_SUMMARY_SCHEMA] };
const output = JSON.stringify(document, null, 2) + "\n", target = "docs/schemas/deepbom-evidence-links-v1.schema.json";
if (process.argv.includes("--check")) { if (await readFile(target, "utf8") !== output) throw Error("Evidence-link schema is stale. Run node scripts/generate-evidence-link-schema.mjs."); }
else await writeFile(target, output);
