import { NODE_KINDS, RELATION_ROLES } from "./provenance/contracts.js";
import { assertDocumentDigest, exactFields, requireEvidence as need, uniqueEvidenceIds, validateSubjectReference, sealDocument, canonicalEvidenceDigest } from "./evidence-identity.js";

import {TYPED_PROVENANCE_IR_SCHEMA} from "./evidence-ir.js";
export const PROVENANCE_V2_SCHEMA = TYPED_PROVENANCE_IR_SCHEMA;
export const PROVENANCE_INPUT_V2_SCHEMA = "deepbom.provenance_input.v2";
const KINDS = [...NODE_KINDS, "model_state", "configuration"];
const TARGETS = {
  uses_training_data: ["dataset", "data_release"], uses_evaluation_data: ["dataset", "data_release"], uses_calibration_data: ["dataset", "data_release"],
  generated_by: ["run"], uses_code: ["software"], uses_environment: ["environment"], defined_by: ["cohort_definition", "protocol"], uses_features: ["feature_contract"],
  evaluated_by: ["evaluation_report"], has_quality_report: ["quality_report"], references_bom: ["bom"], documented_by: ["document"],
};
function validateInput(input) {
  exactFields(input, ["schema", "subject", "nodes", "relationships"], [], "provenance input");
  need(input.schema === PROVENANCE_INPUT_V2_SCHEMA, "unsupported provenance input");
  validateSubjectReference(input.subject);
  uniqueEvidenceIds(input.nodes, "provenance nodes"); uniqueEvidenceIds(input.relationships, "relationships");
  need(input.nodes.length <= 256 && input.relationships.length <= 1024, "provenance size limit exceeded");
  const nodes = new Map(input.nodes.map(n => [n.id, n]));
  for (const n of nodes.values()) {
    exactFields(n, ["id", "kind", "ref"], ["attributes"], "provenance node");
    need(KINDS.includes(n.kind), "unknown provenance node kind"); validateSubjectReference(n.ref);
    if (n.attributes !== undefined) need(n.attributes && typeof n.attributes === "object" && !Array.isArray(n.attributes), "attributes must be an object");
  }
  const parents = new Map();
  for (const r of input.relationships) {
    exactFields(r, ["id", "from", "to", "role"], [], "relationship");
    need(nodes.has(r.from) && nodes.has(r.to) && r.from !== r.to, "relationship has missing or identical endpoints");
    need(RELATION_ROLES.includes(r.role), "unsupported relationship role");
    need(!TARGETS[r.role] || TARGETS[r.role].includes(nodes.get(r.to).kind), "relationship endpoint kind mismatch");
    if (r.role === "derived_from") {
      const list = parents.get(r.from) || []; list.push(r.to); parents.set(r.from, list);
    }
  }
  const done = new Set();
  function visit(id, active) { need(!active.has(id), "cyclic derivation"); if(done.has(id))return; active.add(id); for(const p of parents.get(id)||[])visit(p,active); active.delete(id); done.add(id); }
  for (const n of nodes.keys()) visit(n, new Set());
  return input;
}
export function buildProvenanceV2(input, context) {
  validateInput(input);
  const checks = [{ id: "subject", ...context.resolve(input.subject) }, ...input.nodes.map(n => ({ id: n.id, ...context.resolve(n.ref) }))];
  const counts = { total: checks.length, matched: checks.filter(r => r.status === "matched").length, unresolved: checks.filter(r => r.status === "unresolved").length };
  return sealDocument({ schema: PROVENANCE_V2_SCHEMA, method_version: "1.0.0", input: structuredClone(input), checks, counts, context_sha256: context.context_sha256,
    relationships_status: "declared_unverified", attributes_status: "preserved_not_semantically_assessed",
    status: counts.unresolved ? "incomplete" : "no_contradiction_observed" }, "provenance_ir_sha256");
}
export function validateProvenanceV2(doc, context = null) {
  exactFields(doc, ["schema", "method_version", "input", "checks", "counts", "context_sha256", "relationships_status", "attributes_status", "status", "provenance_ir_sha256"], [], "Provenance IR v2");
  need(doc.schema === PROVENANCE_V2_SCHEMA && doc.method_version === "1.0.0", "unsupported Provenance IR");
  validateInput(doc.input); assertDocumentDigest(doc, "provenance_ir_sha256");
  need(doc.relationships_status === "declared_unverified" && doc.attributes_status === "preserved_not_semantically_assessed", "unsupported provenance truth assertion");
  need(context, "provenance verification requires the original evidence context");
  if (context) need(canonicalEvidenceDigest(buildProvenanceV2(doc.input,context)) === canonicalEvidenceDigest(doc), "provenance does not reconstruct from context");
  else {
    need(doc.checks.length === doc.input.nodes.length + 1 && doc.counts.total === doc.checks.length, "provenance check count mismatch");
    need(doc.counts.matched === doc.checks.filter(r=>r.status === "matched").length && doc.counts.unresolved === doc.checks.filter(r=>r.status === "unresolved").length && doc.counts.total === doc.counts.matched + doc.counts.unresolved, "provenance counts mismatch");
    need(doc.status === (doc.counts.unresolved ? "incomplete" : "no_contradiction_observed"), "provenance status mismatch");
  }
  return doc;
}
