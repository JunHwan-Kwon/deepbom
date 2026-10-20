import { portableContract } from "./portable-contracts.js";
// Leaf contract: no IR builders, filesystem, DOM or network dependencies.
import { canonicalJsonInput } from "./report-utils.js";

export const SHA256 = /^[a-f0-9]{64}$/;
export const REFERENCE_KINDS = Object.freeze(["artifact_file", "artifact_set", "native_state", "snapshot", "evidence_document"]);
export const canonicalEvidenceJson = value => portableContract("evidence_identity", {operation:"canonical", value:canonicalJsonInput(value)});
export const canonicalEvidenceDigest = value => portableContract("evidence_identity", {operation:"digest", value:canonicalJsonInput(value)});
export function evidenceContract(operation, value, options = {}) {
  const error = portableContract("evidence_workflow", canonicalJsonInput({operation, value, ...options}));
  requireEvidence(!error, error);
}
export function sealDocument(body, field) {
  // Capture once: caller mutations and getters cannot change sealed content.
  // Keep source key order for existing JSON file/export byte compatibility.
  const captured = structuredClone(canonicalJsonInput(body, {preserveOrder:true}));
  return { ...captured, [field]: canonicalEvidenceDigest(captured) };
}
export function requireEvidence(condition, message) { if (!condition) throw new Error(`Evidence: ${message}`); }
export function plainRecord(value, label) {
  requireEvidence(value !== null && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  canonicalJsonInput(value);
  return value;
}
export function exactFields(value, required, optional = [], label = "document") {
  plainRecord(value, label);
  requireEvidence(required.every(k => Object.hasOwn(value, k)) && Object.keys(value).every(k => required.includes(k) || optional.includes(k)), `${label} fields do not match its contract`);
}
export function evidenceText(value, label, max = 4096) {
  requireEvidence(typeof value === "string" && value.trim().length > 0 && value.length <= max, `${label} must be nonempty bounded text`);
  return value;
}
export function assertDocumentDigest(document, field) {
  plainRecord(document, "evidence document");
  const { [field]: digest, ...body } = document;
  requireEvidence(SHA256.test(digest || "") && canonicalEvidenceDigest(body) === digest, `${field} mismatch`);
  return document;
}
export function subjectReference(kind, sha256, { schema = null, subject_ref = null } = {}) {
  return validateSubjectReference({ kind, schema, sha256, digest_scope: kind === "artifact_file" ? "file_bytes" : "canonical_document", subject_ref });
}
export function validateSubjectReference(ref) {
  plainRecord(ref, "subject reference");
  const error = portableContract("evidence_contract", { operation: "subject", value: ref });
  requireEvidence(!error, error);
  return Object.freeze({ ...ref });
}
export function referenceKey(ref) { return canonicalEvidenceJson(validateSubjectReference(ref)); }
export function sameReference(a, b) { return referenceKey(a) === referenceKey(b); }
export function uniqueEvidenceIds(rows, label = "records") {
  const error = portableContract("evidence_contract", { operation: "unique_ids", value: rows, label });
  requireEvidence(!error, error);
}

// Millisecond-resolution RFC3339 subset; reject calendar rollovers and local time.
export function evidenceTimestamp(value) {
  const result = portableContract("evidence_contract", { operation: "timestamp", value });
  requireEvidence(!result.error, result.error);
  return result.value;
}
