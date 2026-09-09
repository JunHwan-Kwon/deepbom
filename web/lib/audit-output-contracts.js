export const AUDIT_DEFAULT_OUTPUT_FORMAT = "summary";

export const AUDIT_OUTPUT_FORMATS = Object.freeze([
  "summary",
  "envelope",
  "json",
  "json-compact",
  "cyclonedx",
  "sarif",
]);

export const AUDIT_OUTPUT_CONTRACTS = Object.freeze({
  summary: Object.freeze({
    media_type: "text/plain; charset=utf-8",
    derived_from: "deepbom.review_summary.v1",
    stability: "bounded_human_projection",
  }),
  json: Object.freeze({
    media_type: "application/json",
    stability: "format_specific_complete_evidence",
  }),
  "json-compact": Object.freeze({
    media_type: "application/json",
    stability: "format_specific_complete_evidence",
    serialization: "compact",
  }),
  envelope: Object.freeze({
    media_type: "application/json",
    schema: "deepbom.artifact_evidence_envelope.v1",
    stability: "canonical_cross_format_contract",
  }),
  cyclonedx: Object.freeze({
    media_type: "application/vnd.cyclonedx+json",
    spec_version: "1.7",
  }),
  sarif: Object.freeze({
    media_type: "application/sarif+json",
    version: "2.1.0",
  }),
});

export function cloneAuditOutputContracts() {
  return Object.fromEntries(Object.entries(AUDIT_OUTPUT_CONTRACTS).map(([name, contract]) => [name, { ...contract }]));
}
