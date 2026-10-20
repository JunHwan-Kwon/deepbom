import { portableContract } from "../portable-contracts.js";
import { canonicalJson } from "../report-utils.js";

export const OMOP_INPUT_SCHEMA = "deepbom.omop_metadata_input.v1";
import { PROVENANCE_IR_SCHEMA, PROVENANCE_INPUT_SCHEMA } from "../evidence-ir.js";
export { PROVENANCE_IR_SCHEMA, PROVENANCE_INPUT_SCHEMA };
export const PROVENANCE_METHOD_VERSION = "1.0.0";
export const PRIMARY_MODEL_REF = "artifact:primary";
export const PROVENANCE_LIMITS = Object.freeze({ input_bytes: 2 * 1024 * 1024, file_bytes: 32 * 1024 * 1024, total_file_bytes: 64 * 1024 * 1024, nodes: 256, relationships: 1024, evidence_refs: 4096, attributes: 128, text: 4096 });
export const NODE_KINDS = Object.freeze(["model_artifact", "dataset", "data_release", "manifest", "cohort_definition", "feature_contract", "software", "environment", "run", "evaluation_report", "quality_report", "protocol", "bom", "document"]);
export const RELATION_ROLES = Object.freeze(["associated_with", "uses_training_data", "uses_evaluation_data", "uses_calibration_data", "derived_from", "generated_by", "uses_code", "uses_environment", "defined_by", "uses_features", "evaluated_by", "has_quality_report", "references_bom", "documented_by"]);
export const SHA256 = /^[a-f0-9]{64}$/;

export function requireProvenance(condition, message) { if (!condition) throw new Error(`Provenance: ${message}`); }
export function objectKeys(value, required, optional, label) {
  requireProvenance(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object.`);
  requireProvenance(Object.keys(value).every(key => [...required, ...optional].includes(key)), `${label} contains an unknown field.`);
  requireProvenance(required.every(key => Object.hasOwn(value, key)), `${label} is missing a required field.`);
}
export function text(value, label, { empty = false, limit = PROVENANCE_LIMITS.text } = {}) {
  requireProvenance(typeof value === "string" && value.length <= limit && (empty || value.trim().length > 0) && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value), `${label} must be bounded text.`);
  return value;
}
export function identifier(value, label) {
  text(value, label, { limit: 256 });
  requireProvenance(/^[a-zA-Z0-9][a-zA-Z0-9._:/#-]*$/.test(value), `${label} contains unsupported identifier characters.`);
  return value;
}
export function localEvidencePath(value) {
  text(value, "file", { limit: 1024 });
  requireProvenance(!value.startsWith("/") && !/[\\:\u0000]/.test(value) && value.split("/").every(part => part && part !== "." && part !== ".."), "file must be a relative path without traversal or a URL.");
  return value;
}
export function digest(value, label) { requireProvenance(typeof value === "string" && SHA256.test(value), `${label} must be a lowercase full SHA-256.`); return value; }
export function list(value, limit, label) { requireProvenance(Array.isArray(value) && value.length <= limit, `${label} exceeds its array bound or is not an array.`); return value; }
export function primitive(value, label) {
  requireProvenance(value === null || typeof value === "boolean" || typeof value === "string" || (typeof value === "number" && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))), `${label} must be a JSON scalar; encode large integers as decimal strings.`);
  if (typeof value === "string") text(value, label, { empty: true });
}
export function validateProvenanceInput(input) {
  canonicalJson(input);
  return portableContract("provenance",{operation:"input",value:input});
}
