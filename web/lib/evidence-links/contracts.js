import { canonicalJson } from "../report-utils.js";

export const LINK_INPUT_SCHEMA = "deepbom.evidence_link_input.v1";
export const OMOP_INPUT_SCHEMA = "deepbom.omop_metadata_input.v1";
export const LINK_IR_SCHEMA = "deepbom.evidence_link_ir.v1";
export const LINK_METHOD_VERSION = "1.0.0";
export const PRIMARY_MODEL_REF = "artifact:primary";
export const LINK_LIMITS = Object.freeze({ input_bytes: 2 * 1024 * 1024, file_bytes: 32 * 1024 * 1024, total_file_bytes: 64 * 1024 * 1024, nodes: 256, relationships: 1024, evidence_refs: 4096, attributes: 128, text: 4096 });
export const NODE_KINDS = Object.freeze(["model_artifact", "dataset", "data_release", "manifest", "cohort_definition", "feature_contract", "software", "environment", "run", "evaluation_report", "quality_report", "protocol", "bom", "document"]);
export const RELATION_ROLES = Object.freeze(["associated_with", "uses_training_data", "uses_evaluation_data", "uses_calibration_data", "derived_from", "generated_by", "uses_code", "uses_environment", "defined_by", "uses_features", "evaluated_by", "has_quality_report", "references_bom", "documented_by"]);
export const SHA256 = /^[a-f0-9]{64}$/;

export function requireLink(condition, message) { if (!condition) throw new Error(`Evidence links: ${message}`); }
export function objectKeys(value, required, optional, label) {
  requireLink(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object.`);
  requireLink(Object.keys(value).every(key => [...required, ...optional].includes(key)), `${label} contains an unknown field.`);
  requireLink(required.every(key => Object.hasOwn(value, key)), `${label} is missing a required field.`);
}
export function text(value, label, { empty = false, limit = LINK_LIMITS.text } = {}) {
  requireLink(typeof value === "string" && value.length <= limit && (empty || value.trim().length > 0) && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value), `${label} must be bounded text.`);
  return value;
}
export function identifier(value, label) {
  text(value, label, { limit: 256 });
  requireLink(/^[a-zA-Z0-9][a-zA-Z0-9._:/#-]*$/.test(value), `${label} contains unsupported identifier characters.`);
  return value;
}
export function localEvidencePath(value) {
  text(value, "file", { limit: 1024 });
  requireLink(!value.startsWith("/") && !/[\\:\u0000]/.test(value) && value.split("/").every(part => part && part !== "." && part !== ".."), "file must be a relative path without traversal or a URL.");
  return value;
}
export function digest(value, label) { requireLink(typeof value === "string" && SHA256.test(value), `${label} must be a lowercase full SHA-256.`); return value; }
export function list(value, limit, label) { requireLink(Array.isArray(value) && value.length <= limit, `${label} exceeds its array bound or is not an array.`); return value; }
export function primitive(value, label) {
  requireLink(value === null || typeof value === "boolean" || typeof value === "string" || (typeof value === "number" && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))), `${label} must be a JSON scalar; encode large integers as decimal strings.`);
  if (typeof value === "string") text(value, label, { empty: true });
}
export function validateLinkInput(input) {
  const json = canonicalJson(input);
  requireLink(new TextEncoder().encode(json).length <= LINK_LIMITS.input_bytes, "metadata input exceeds 2 MiB.");
  objectKeys(input, ["schema", "subject", "nodes", "relationships"], input.schema === OMOP_INPUT_SCHEMA ? ["omop"] : [], "input");
  requireLink([LINK_INPUT_SCHEMA, OMOP_INPUT_SCHEMA].includes(input.schema), "unsupported metadata input schema.");
  if (input.schema === OMOP_INPUT_SCHEMA) requireLink(Object.hasOwn(input, "omop"), "OMOP input requires omop metadata.");
  objectKeys(input.subject, ["artifact_sha256", "artifact_set_sha256"], ["model_ir_sha256"], "subject");
  digest(input.subject.artifact_sha256, "subject.artifact_sha256");
  if (input.subject.artifact_set_sha256 !== null) digest(input.subject.artifact_set_sha256, "subject.artifact_set_sha256");
  if (input.subject.model_ir_sha256 !== undefined) digest(input.subject.model_ir_sha256, "subject.model_ir_sha256");
  const seen = new Set([PRIMARY_MODEL_REF, "omop:release", "omop:context"]);
  list(input.nodes, LINK_LIMITS.nodes, "nodes").forEach((node, index) => {
    const label = `nodes[${index}]`;
    objectKeys(node, ["id", "kind", "name"], ["version", "uri", "sha256", "file", "attributes", "bom"], label);
    identifier(node.id, `${label}.id`); requireLink(!seen.has(node.id), `${label}.id is duplicated or reserved.`); seen.add(node.id);
    requireLink(NODE_KINDS.includes(node.kind), `${label}.kind is unsupported.`); text(node.name, `${label}.name`);
    for (const key of ["version", "uri"]) if (node[key] !== undefined) text(node[key], `${label}.${key}`);
    if (node.sha256 !== undefined) digest(node.sha256, `${label}.sha256`);
    if (node.file !== undefined) localEvidencePath(node.file);
    const attributes = new Set();
    list(node.attributes || [], LINK_LIMITS.attributes, `${label}.attributes`).forEach((attribute, at) => {
      objectKeys(attribute, ["name", "value"], [], `${label}.attributes[${at}]`);
      text(attribute.name, "attribute name", { limit: 256 }); primitive(attribute.value, "attribute value");
      requireLink(!attributes.has(attribute.name), `${label} repeats an attribute name.`); attributes.add(attribute.name);
    });
    if (node.bom !== undefined) {
      requireLink(node.kind === "bom", `${label}.bom requires kind=bom.`);
      objectKeys(node.bom, ["format", "spec_version", "document_id", "document_version", "element_ref"], [], `${label}.bom`);
      for (const key of ["format", "spec_version", "document_id", "element_ref"]) text(node.bom[key], `${label}.bom.${key}`);
      requireLink(node.bom.document_version === null || (Number.isSafeInteger(node.bom.document_version) && node.bom.document_version >= 1), "BOM document_version must be a positive safe integer or null.");
    }
  });
  const relationshipIds = new Set(); let referenceCount = 0;
  list(input.relationships, LINK_LIMITS.relationships, "relationships").forEach((edge, index) => {
    objectKeys(edge, ["id", "from", "to", "role"], ["evidence_refs"], `relationships[${index}]`);
    for (const key of ["id", "from", "to"]) identifier(edge[key], `relationship.${key}`);
    requireLink(!seen.has(edge.id) && !relationshipIds.has(edge.id), "relationship id is duplicated, reserved or collides with a node."); relationshipIds.add(edge.id);
    requireLink(RELATION_ROLES.includes(edge.role), "relationship role is unsupported.");
    const refs = list(edge.evidence_refs || [], LINK_LIMITS.nodes, "evidence_refs");
    referenceCount += refs.length; requireLink(referenceCount <= LINK_LIMITS.evidence_refs, "relationship evidence references exceed the total bound.");
    refs.forEach(ref => identifier(ref, "evidence_ref")); requireLink(new Set(refs).size === refs.length, "evidence_refs contains duplicates.");
  });
  return JSON.parse(json);
}
