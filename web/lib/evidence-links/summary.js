// Host conversations receive counts and digests only, never metadata rows.
const hash = { type: "string", pattern: "^[a-f0-9]{64}$" };
const count = { type: "integer", minimum: 0, maximum: 1000000 };
const statuses = ["match", "mismatch", "not_assessed", "unresolved", "unsupported"];
const counts = ["node_count", "relationship_count", "observed_file_count", "check_count", "field_count", "mapped_declared_field_count", "unsupported_field_count", "attested_relationship_count"];
const properties = {
  schema: { type: "string", const: "deepbom.evidence_link_summary.v1" },
  artifact_sha256: hash, model_ir_sha256: hash, evidence_link_ir_sha256: hash,
  status: { type: "string", enum: ["contradiction_observed", "incomplete", "no_contradiction_observed"] },
  ...Object.fromEntries(counts.map(key => [key, count])),
  check_counts: { type: "object", properties: Object.fromEntries(statuses.map(key => [key, count])), required: statuses, additionalProperties: false },
  metadata_truth_verified: { type: "boolean", const: false },
  publisher_authenticity: { type: "string", const: "not_verified" },
};
export const EVIDENCE_LINK_SUMMARY_SCHEMA = Object.freeze({ type: "object", properties, required: Object.keys(properties), additionalProperties: false });

export function validateEvidenceLinkSummary(value, artifactSha256, modelIrSha256) {
  const fail = () => { throw new Error("Evidence-link summary has invalid fields, source binding or coverage counts."); };
  if (!value || Array.isArray(value) || typeof value !== "object" || Object.keys(value).length !== Object.keys(properties).length || Object.keys(value).some(key => !Object.hasOwn(properties, key))) fail();
  if (value.schema !== properties.schema.const || value.metadata_truth_verified !== false || value.publisher_authenticity !== "not_verified" || value.attested_relationship_count !== 0) fail();
  for (const key of ["artifact_sha256", "model_ir_sha256", "evidence_link_ir_sha256"]) if (typeof value[key] !== "string" || !/^[a-f0-9]{64}$/.test(value[key])) fail();
  if (value.artifact_sha256 !== artifactSha256 || value.model_ir_sha256 !== modelIrSha256) fail();
  for (const key of counts) if (!Number.isSafeInteger(value[key]) || value[key] < 0 || value[key] > count.maximum) fail();
  const checks = value.check_counts;
  if (!checks || Array.isArray(checks) || Object.keys(checks).length !== statuses.length || statuses.some(key => !Number.isSafeInteger(checks[key]) || checks[key] < 0 || checks[key] > count.maximum)) fail();
  if (statuses.reduce((sum, key) => sum + checks[key], 0) !== value.check_count || value.mapped_declared_field_count + value.unsupported_field_count !== value.field_count || value.node_count < 1 || value.node_count > 258 || value.relationship_count > 1025 || value.observed_file_count >= value.node_count) fail();
  const expected = checks.mismatch ? "contradiction_observed" : checks.unresolved || checks.not_assessed || checks.unsupported ? "incomplete" : "no_contradiction_observed";
  if (value.status !== expected) fail();
  return value;
}

export function evidenceLinkSummaryText(value) {
  return `Optional metadata links: ${value.status}; ${value.node_count} records, ${value.relationship_count} declared relationships, ${value.check_counts.mismatch} consistency mismatches, ${value.unsupported_field_count} unmapped fields. Relationship truth and publisher authenticity remain unverified. Evidence Link IR SHA-256: ${value.evidence_link_ir_sha256}.`;
}
