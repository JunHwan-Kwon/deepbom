// Leaf contract: no IR builders, filesystem, DOM or network dependencies.
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const SHA256 = /^[a-f0-9]{64}$/;
export const REFERENCE_KINDS = Object.freeze(["artifact_file", "artifact_set", "native_state", "snapshot", "evidence_document"]);
export const canonicalEvidenceDigest = value => sha256TextHex(canonicalJson(value));
export function sealDocument(body, field) { return { ...body, [field]: canonicalEvidenceDigest(body) }; }
export function requireEvidence(condition, message) { if (!condition) throw new Error(`Evidence: ${message}`); }
export function plainRecord(value, label) {
  requireEvidence(value !== null && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  canonicalJson(value);
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
  exactFields(ref, ["kind", "schema", "sha256", "digest_scope", "subject_ref"], [], "subject reference");
  requireEvidence(REFERENCE_KINDS.includes(ref.kind), "unknown subject kind");
  requireEvidence(typeof ref.sha256 === "string" && SHA256.test(ref.sha256), "subject SHA-256 is invalid");
  requireEvidence(ref.digest_scope === (ref.kind === "artifact_file" ? "file_bytes" : "canonical_document"), "subject digest scope disagrees with kind");
  if (ref.kind === "artifact_file") requireEvidence(ref.schema === null && ref.subject_ref === null, "file bytes have no document schema or internal IR subject");
  else evidenceText(ref.schema, "subject schema");
  if (ref.kind === "artifact_set") requireEvidence(ref.schema === "deepbom.artifact_set.v1", "artifact set schema mismatch");
  if (ref.kind === "native_state") requireEvidence(ref.schema === "deepbom.model_state_snapshot.v1", "native state schema mismatch");
  if (ref.kind === "snapshot") requireEvidence(ref.schema === "deepbom.snapshot_ir.v1", "snapshot schema mismatch");
  if (ref.subject_ref !== null) evidenceText(ref.subject_ref, "internal subject reference");
  return Object.freeze({ ...ref });
}
export function referenceKey(ref) { return canonicalJson(validateSubjectReference(ref)); }
export function sameReference(a, b) { return referenceKey(a) === referenceKey(b); }
export function uniqueEvidenceIds(rows, label = "records") {
  requireEvidence(Array.isArray(rows) && rows.length <= 10000, `${label} must be a bounded array`);
  rows.forEach(r => evidenceText(r.id, `${label} id`));
  requireEvidence(new Set(rows.map(r => r.id)).size === rows.length, `${label} IDs must be unique`);
}

// Millisecond-resolution RFC3339 subset; reject calendar rollovers and local time.
export function evidenceTimestamp(value) {
  requireEvidence(typeof value === 'string', 'timestamp must be text');
  const m=/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  requireEvidence(m, 'timestamp requires seconds, explicit timezone, and at most millisecond precision');
  const [year,month,day,hour,minute,second]=m.slice(1,7).map(Number);
  const date=new Date(0);date.setUTCFullYear(year,month-1,day);date.setUTCHours(hour,minute,second,0);
  requireEvidence(date.getUTCFullYear()===year&&date.getUTCMonth()===month-1&&date.getUTCDate()===day&&hour<24&&minute<60&&second<60, 'invalid calendar timestamp');
  if(m[8]!=='Z')requireEvidence(Number(m[8].slice(1,3))<24&&Number(m[8].slice(4))<60,'invalid timezone offset');
  const time=Date.parse(value);requireEvidence(Number.isFinite(time),'invalid timestamp');return time;
}
