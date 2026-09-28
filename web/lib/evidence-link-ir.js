import { canonicalJson, compareCanonicalText } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { modelEvidenceSource, evidenceHashContract, sealEvidence, assertEvidenceDigest, freezeEvidence } from "./ir-evidence-contract.js";
import { LINK_IR_SCHEMA, LINK_METHOD_VERSION, PRIMARY_MODEL_REF, LINK_LIMITS, requireLink, objectKeys, digest, list } from "./evidence-links/contracts.js";
import { normalizeMetadataInput } from "./evidence-links/omop.js";
import { inspectBomReference } from "./evidence-links/bom-reference.js";

export { LINK_IR_SCHEMA };
const DIGEST_FIELD = "evidence_link_ir_sha256";
const BOUNDARY = "Metadata relationships are supplied declarations, not proof of training, deployment, execution, clinical suitability or FAIR conformance. Matching hashes bind only the bytes actually provided. Manifest identity is not database identity. Schema, reference resolution, content integrity, attribute comparison and publisher authenticity are separate checks; signatures are not verified.";
const STATUSES = ["match", "mismatch", "not_assessed", "unresolved", "unsupported"];
const REPORT_KINDS = new Set(["manifest", "cohort_definition", "feature_contract", "run", "evaluation_report", "quality_report", "protocol", "bom", "document"]);

export function buildEvidenceLinkIr(modelInput, metadata, { observations = [] } = {}) {
  const { model, source } = modelEvidenceSource(modelInput);
  const normalized = normalizeMetadataInput(metadata), { input, profile } = normalized;
  requireLink(input.subject.artifact_sha256 === source.artifact_sha256 && input.subject.artifact_set_sha256 === source.artifact_set_sha256, "metadata subject does not match the active artifact and artifact-set identity.");
  if (input.subject.model_ir_sha256 !== undefined) requireLink(input.subject.model_ir_sha256 === source.model_ir_sha256, "metadata references a different Model IR.");
  const nodes = [{ id: PRIMARY_MODEL_REF, kind: "model_artifact", name: model.artifact.filename, sha256: source.artifact_sha256, evidence_class: "OBSERVED_SERIALIZED_ARTIFACT", attributes: [] }, ...normalized.nodes.map(node => ({ ...node, attributes: node.attributes || [], evidence_class: "DECLARED_UNVERIFIED" }))].sort((a, b) => compareCanonicalText(a.id, b.id));
  const index = new Map(nodes.map(node => [node.id, node]));
  const observed = validateObservations(observations, index);
  const byObservation = new Map(observed.map(row => [row.node_ref, row]));
  const checks = [], fieldLedger = (profile?.field_ledger || []).map(row => ({ ...row, source_document: "input_document", subject_ref: "omop:release" }));
  const check = (subject_ref, kind, status, message) => checks.push({ subject_ref, kind, status, message });
  check(PRIMARY_MODEL_REF, "model_binding", "match", "Metadata is bound to the current artifact identity.");
  check(PRIMARY_MODEL_REF, "artifact_set_binding", source.artifact_set_sha256 === null ? "not_assessed" : "match", source.artifact_set_sha256 === null ? "Artifact-set identity is not declared in this Model IR." : "The declared artifact-set digest matches the current Model IR source.");
  if (profile) {
    check("omop:release", "omop_profile", profile.status === "supported_import_profile" ? "match" : "unsupported", `OMOP ${profile.cdm_version} metadata import profile; this is not CDM conformance validation.`);
    for (const name of profile.missing_fields) check("omop:release", `missing:${name}`, "not_assessed", `CDM_SOURCE.${name} was not supplied.`);
    if (profile.field_ledger.some(row => row.status === "preserved_unsupported_mapping")) check("omop:release", "omop_field_mapping", "unsupported", "Unmapped CDM_SOURCE fields are preserved in the field ledger.");
  }
  input.nodes.forEach((node, at) => (node.attributes || []).forEach((attribute, index) => fieldLedger.push({ source_document: "input_document", source_pointer: `/nodes/${at}/attributes/${index}`, subject_ref: node.id, ...attribute, status: "preserved_unsupported_mapping" })));
  for (const node of nodes) {
    if (node.id === PRIMARY_MODEL_REF) continue;
    if (node.id !== "omop:release" && node.attributes.length) check(node.id, "attribute_mapping", "unsupported", "Supplied attributes are preserved as declarations; their meaning and truth have not been verified.");
    const observation = byObservation.get(node.id);
    if (node.sha256 || node.file || observation) {
      const status = !observation || !node.sha256 ? "not_assessed" : observation.sha256 === node.sha256 ? "match" : "mismatch";
      check(node.id, "file_sha256", status, !observation ? "Referenced bytes were not provided." : !node.sha256 ? "File SHA-256 was measured, but no independent expected digest was declared." : "Compare the provided file bytes with the declared digest; relationship truth remains unverified.");
    }
    if (node.kind === "bom") {
      const currentModel = normalized.relationships.some(edge => edge.from === PRIMARY_MODEL_REF && edge.to === node.id && edge.role === "references_bom");
      const result = inspectBomReference(node, observation, source.artifact_sha256, currentModel, observed.indexOf(observation));
      checks.push(...result.checks); fieldLedger.push(...result.fields);
    }
  }
  const relationships = normalized.relationships.map(edge => {
    const from = index.get(edge.from), to = index.get(edge.to), refs = edge.evidence_refs || [];
    const resolved = Boolean(from && to);
    const kindMatch = resolved && edge.from !== edge.to && compatibleRole(edge.role, from.kind, to.kind);
    check(edge.id, "relationship_reference", !resolved ? "unresolved" : kindMatch ? "match" : "mismatch", !resolved ? "Relationship endpoint is absent from the node inventory." : kindMatch ? "Endpoints and relationship kinds are consistent; the relationship remains declared." : "Relationship endpoints contradict the role or form a self-reference.");
    for (const ref of refs) check(edge.id, `evidence_reference:${ref}`, !index.has(ref) ? "unresolved" : REPORT_KINDS.has(index.get(ref).kind) ? "match" : "mismatch", "Evidence references must resolve to a supporting document, manifest or run record.");
    return { ...edge, evidence_refs: refs, evidence_class: "DECLARED_UNVERIFIED", resolution: !resolved ? "unresolved" : kindMatch ? "resolved" : "kind_mismatch", claim_verification: "not_assessed" };
  }).sort((a, b) => compareCanonicalText(a.id, b.id));
  if (hasDerivationCycle(relationships)) check("relationships", "derivation_cycle", "mismatch", "Derived-from relationships contain a cycle.");
  checks.sort((a, b) => compareCanonicalText(`${a.subject_ref}\0${a.kind}`, `${b.subject_ref}\0${b.kind}`));
  const counts = Object.fromEntries(STATUSES.map(status => [status, checks.filter(row => row.status === status).length]));
  const status = counts.mismatch ? "contradiction_observed" : counts.unresolved || counts.not_assessed || counts.unsupported ? "incomplete" : "no_contradiction_observed";
  const body = {
    schema: LINK_IR_SCHEMA, method_version: LINK_METHOD_VERSION, hash_contract: evidenceHashContract(DIGEST_FIELD), source,
    input_document: input, input_sha256: sha256TextHex(canonicalJson(input)),
    profile: profile ? { id: "deepbom.omop_metadata_profile.v1", cdm_version: profile.cdm_version, status: profile.status, missing_source_fields: profile.missing_fields } : { id: "deepbom.generic_metadata_profile.v1", status: "supported_import_profile" },
    nodes, relationships, observations: observed, checks, field_ledger: fieldLedger,
    coverage: { node_count: nodes.length, relationship_count: relationships.length, observed_file_count: observed.length, check_count: checks.length, check_counts: counts, field_count: fieldLedger.length, mapped_declared_field_count: fieldLedger.filter(row => row.status === "mapped_declared").length, unsupported_field_count: fieldLedger.filter(row => row.status === "preserved_unsupported_mapping").length, attested_relationship_count: 0 },
    verdict: { status, scope: "binding_reference_and_supplied_file_consistency", metadata_truth_verified: false, publisher_authenticity: "not_verified" },
    interpretation_boundary: BOUNDARY,
  };
  return freezeEvidence(sealEvidence(body, DIGEST_FIELD));
}

export function validateEvidenceLinkIr(document, model) {
  assertEvidenceDigest(document, DIGEST_FIELD);
  const expected = buildEvidenceLinkIr(model, document.input_document, { observations: document.observations });
  requireLink(canonicalJson(document) === canonicalJson(expected), "IR does not reproduce from its bound source, input and observed-file records.");
  return expected;
}

export function evidenceLinkSummary(document) {
  return { schema: "deepbom.evidence_link_summary.v1", artifact_sha256: document.source.artifact_sha256, model_ir_sha256: document.source.model_ir_sha256, evidence_link_ir_sha256: document.evidence_link_ir_sha256, status: document.verdict.status, ...document.coverage, metadata_truth_verified: false, publisher_authenticity: "not_verified" };
}

function validateObservations(rows, nodes) {
  list(rows, LINK_LIMITS.nodes, "observations");
  const seen = new Set(); let bytes = 0;
  return rows.map(row => {
    objectKeys(row, ["node_ref", "sha256", "byte_length"], ["document"], "observation");
    requireLink(nodes.has(row.node_ref) && row.node_ref !== PRIMARY_MODEL_REF && !seen.has(row.node_ref), "observation must refer to one unique metadata node."); seen.add(row.node_ref);
    digest(row.sha256, "observed SHA-256");
    requireLink(Number.isSafeInteger(row.byte_length) && row.byte_length >= 0 && row.byte_length <= LINK_LIMITS.file_bytes, "observed file length exceeds its bound."); bytes += row.byte_length;
    requireLink(bytes <= LINK_LIMITS.total_file_bytes, "evidence files exceed the 64 MiB total bound.");
    if (row.document !== undefined) requireLink(nodes.get(row.node_ref).kind === "bom" && row.document && typeof row.document === "object" && !Array.isArray(row.document), "only BOM observations may include a parsed document.");
    return JSON.parse(canonicalJson(row));
  }).sort((a, b) => compareCanonicalText(a.node_ref, b.node_ref));
}

function compatibleRole(role, from, to) {
  if (["uses_training_data", "uses_evaluation_data", "uses_calibration_data"].includes(role)) return ["model_artifact", "run"].includes(from) && ["dataset", "data_release"].includes(to);
  if (role === "generated_by") return ["model_artifact", "dataset", "data_release", "evaluation_report", "quality_report"].includes(from) && to === "run";
  if (role === "uses_code") return to === "software";
  if (role === "uses_environment") return to === "environment";
  if (role === "defined_by") return ["cohort_definition", "protocol"].includes(to);
  if (role === "uses_features") return to === "feature_contract";
  if (role === "evaluated_by") return from === "model_artifact" && to === "evaluation_report";
  if (role === "has_quality_report") return to === "quality_report";
  if (role === "references_bom") return to === "bom";
  if (role === "documented_by") return REPORT_KINDS.has(to);
  if (role === "derived_from") return ["model_artifact", "dataset", "data_release"].includes(from) && ["model_artifact", "dataset", "data_release"].includes(to);
  return true;
}

function hasDerivationCycle(edges) {
  const adjacency = new Map(), visiting = new Set(), complete = new Set();
  for (const edge of edges.filter(edge => edge.role === "derived_from")) { if (!adjacency.has(edge.from)) adjacency.set(edge.from, []); adjacency.get(edge.from).push(edge.to); }
  function visit(id) { if (visiting.has(id)) return true; if (complete.has(id)) return false; visiting.add(id); for (const next of adjacency.get(id) || []) if (visit(next)) return true; visiting.delete(id); complete.add(id); return false; }
  return [...adjacency.keys()].some(visit);
}
