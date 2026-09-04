import { createCheck } from "./check-assert.mjs";
import {
  auditCycloneDxPerspectives,
  renderCycloneDxPerspectiveAuditHtml,
  validateCycloneDxPerspectiveAudit,
} from "../web/lib/cyclonedx-perspective-audit.js";

const { done, expect, expectDeepEqual, expectEqual } = createCheck("CycloneDX perspective audit check");
const mappings = [
  ["$.components[*].supplier", "required"],
  ["$.components[*].name", "required"],
  ["$.components[*].version", "required"],
  ["$.components[*]['purl','cpe']", "required"],
  ["$.dependencies[*]", "required"],
  ["$.metadata.authors", "required"],
  ["$.metadata.timestamp", "required"],
].map(([expression, relevance]) => ({ expression, relevance }));
const document = {
  specFormat: "CycloneDX",
  specVersion: "2.0",
  metadata: { authors: [{ name: "Author" }], timestamp: "2026-09-04T00:00:00Z" },
  definitions: {
    components: [
      { name: "A", version: "1", supplier: { name: "S" }, purl: "pkg:generic/a@1", cpe: "cpe:2.3:a:a:a:1:*:*:*:*:*:*:*" },
      { name: "B", version: "2", supplier: { name: "S" }, purl: "pkg:generic/b@2", cpe: "cpe:2.3:a:b:b:2:*:*:*:*:*:*:*" },
    ],
  },
  dependencies: [{ ref: "A" }],
  perspectives: [{ "bom-ref": "perspective", name: "Fixture", mappings }],
};

const raw = auditCycloneDxPerspectives(document);
expectEqual(raw.schema, "deepbom.cyclonedx_perspective_audit.v1", "audit schema");
expectDeepEqual(raw.perspectives[0].mappings.map((row) => row.match_count), [0, 0, 0, 0, 1, 1, 1], "raw fixture match counts");
expectEqual(raw.summary.required_zero_match_count, 4, "required zero-match observations");
expectEqual(raw.policy_interpretation.decision_status, "NOT_ASSESSABLE", "no invented required-cardinality policy");
expect(validateCycloneDxPerspectiveAudit(raw).valid, "raw audit validates");

const projection = {
  schema: "deepbom.cyclonedx_perspective_projection.v1",
  profile_id: "fixture-root-components-candidate",
  rules: [{ operation: "copy_if_absent", source_pointer: "/definitions/components", target_pointer: "/components" }],
  expected_types: { "$.metadata.timestamp": ["number"] },
};
const candidate = auditCycloneDxPerspectives(document, {
  mode: "explicit_candidate_projection",
  projection,
  expectedTypes: projection.expected_types,
});
expectDeepEqual(candidate.perspectives[0].mappings.map((row) => row.match_count), [2, 2, 2, 4, 1, 1, 1], "explicit projection match counts");
expectEqual(candidate.projection.normative_status, "CANDIDATE_ONLY", "projection cannot become normative implicitly");
expectEqual(candidate.perspectives[0].mappings[6].match_status, "type_mismatch", "explicit expected-type mismatch");
expectEqual(candidate.perspectives[0].mappings[1].matches[0].json_pointer, "/components/0/name", "normalized JSON Pointer");
expectEqual(candidate.perspectives[0].mappings[1].matches[0].value_preview, "A", "bounded scalar preview");
expect(/^[a-f0-9]{64}$/.test(candidate.perspectives[0].mappings[0].matches[0].value_sha256), "match value digest");
expect(candidate.subject.document_sha256 !== candidate.subject.evaluation_document_sha256, "projection changes the evaluated document identity");

const rootAudit = auditCycloneDxPerspectives({
  perspectives: [{ name: "Root", mappings: [{ expression: "$", relevance: "optional" }] }],
});
expectEqual(rootAudit.perspectives[0].mappings[0].matches[0].json_pointer, "", "RFC 6901 root pointer");
expect(validateCycloneDxPerspectiveAudit(rootAudit).valid, "root-pointer audit validates");

const html = renderCycloneDxPerspectiveAuditHtml(candidate);
expect(html.startsWith("<!doctype html>"), "read-only HTML report");
expect(html.includes('id="deepbom-perspective-audit"'), "HTML embeds the exact audit ledger");
expect(!html.includes("<script>alert"), "HTML escapes mapped content");

for (const invalidProjection of [
  { ...projection, rules: [{ operation: "copy_if_absent", source_pointer: "/missing", target_pointer: "/components" }] },
  { ...projection, rules: [{ operation: "replace", source_pointer: "/definitions/components", target_pointer: "/components" }] },
  { ...projection, rules: [{ operation: "copy_if_absent", source_pointer: "/definitions/components", target_pointer: "/__proto__/components" }] },
]) {
  let failed = false;
  try { auditCycloneDxPerspectives(document, { mode: "explicit_candidate_projection", projection: invalidProjection }); } catch { failed = true; }
  expect(failed, "invalid projection fails closed");
}

done("CycloneDX perspective audit passed (RFC 9535 matches, JSON Pointers, explicit candidate projection, type/empty/cardinality states, and policy boundary).");
