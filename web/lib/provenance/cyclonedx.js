import { canonicalJson } from "../report-utils.js";
import { sha256TextHex } from "../sha256-sync.js";
import { validateProvenanceIr } from "../provenance-ir.js";
import { PRIMARY_MODEL_REF, requireProvenance } from "./contracts.js";

const property = (name, value) => ({ name: `deepbom:provenance:${name}`, value: typeof value === "string" ? value : canonicalJson(value) });
export const PROVENANCE_FILENAME = "deepbom_provenance_ir.json";
export function provenanceJson(document) { return `${canonicalJson(document)}\n`; }

export function projectProvenanceToCycloneDx(baseDocument, document, model) {
  const ir = validateProvenanceIr(document, model);
  const bom = JSON.parse(canonicalJson(baseDocument));
  requireProvenance(bom.bomFormat === "CycloneDX" && bom.specVersion === "1.7", "metadata projection requires a CycloneDX 1.7 base.");
  const subject = bom.metadata?.component;
  requireProvenance(subject && subject.hashes?.some(hash => hash.alg === "SHA-256" && hash.content === ir.source.artifact_sha256), "BOM subject differs from Provenance IR.");
  requireProvenance(!(bom.properties || []).some(row => row.name === "deepbom:provenance:schema"), "metadata has already been projected into this BOM.");
  const prefix = `deepbom-provenance:${ir.provenance_ir_sha256}:`;
  const refs = new Map(ir.nodes.map(node => [node.id, node.id === PRIMARY_MODEL_REF ? subject["bom-ref"] : `${prefix}${node.id}`]));
  requireProvenance(typeof refs.get(PRIMARY_MODEL_REF) === "string", "base model needs a bom-ref.");
  const observations = new Map(ir.observations.map(row => [row.node_ref, row]));
  const components = ir.nodes.filter(node => node.id !== PRIMARY_MODEL_REF).map(node => {
    const observed = observations.get(node.id), hash = observed?.sha256 || node.sha256;
    const type = ["dataset", "data_release"].includes(node.kind) ? "data" : node.kind === "model_artifact" ? "machine-learning-model" : node.kind === "software" ? "application" : "file";
    return {
      type, "bom-ref": refs.get(node.id), name: node.name,
      ...(node.version ? { version: node.version } : {}), ...(hash ? { hashes: [{ alg: "SHA-256", content: hash }] } : {}),
      ...(type === "data" ? { data: [{ type: "dataset", name: node.name, ...(node.uri ? { contents: { url: node.uri } } : {}) }] } : {}),
      ...(node.uri ? { externalReferences: [{ type: node.kind === "bom" ? "bom" : "documentation", url: node.uri }] } : {}),
      properties: [property("nodeId", node.id), property("kind", node.kind), property("evidenceClass", "DECLARED_UNVERIFIED"), property("digestBasis", observed ? "observed_provided_file" : node.sha256 ? "declared_expected_digest" : "not_provided"), property("declaredAttributes", node.attributes), ...(node.sha256 ? [property("expectedSha256", node.sha256)] : []), ...(node.file ? [property("relativeFile", node.file)] : []), ...(node.bom ? [property("bomReference", node.bom)] : [])],
    };
  });
  // Only resolved, kind-consistent declarations enter standard reference fields.
  // Every other relationship remains in the explicit mapping ledger.
  const usable = ir.relationships.filter(edge => edge.resolution === "resolved");
  const relevantRuns = new Set(usable.filter(edge => edge.from === PRIMARY_MODEL_REF && edge.role === "generated_by").map(edge => edge.to));
  const datasets = [...new Set(usable.filter(edge => (edge.from === PRIMARY_MODEL_REF || relevantRuns.has(edge.from)) && ["uses_training_data", "uses_evaluation_data"].includes(edge.role)).map(edge => refs.get(edge.to)))];
  if (datasets.length) {
    subject.modelCard ||= {}; subject.modelCard.modelParameters ||= {};
    const prior = subject.modelCard.modelParameters.datasets || [];
    subject.modelCard.modelParameters.datasets = [...prior, ...datasets.filter(ref => !prior.some(row => row.ref === ref)).map(ref => ({ ref }))];
  }
  const workflows = ir.nodes.filter(node => node.kind === "run").map(node => ({
    "bom-ref": `${prefix}workflow:${node.id}`, uid: `${prefix}workflow:${node.id}`, name: node.name, taskTypes: ["other"],
    inputs: usable.filter(edge => edge.from === node.id && ["uses_training_data", "uses_evaluation_data", "uses_calibration_data", "uses_code", "uses_environment", "uses_features"].includes(edge.role)).map(edge => ({ resource: { ref: refs.get(edge.to) }, properties: [property("relationshipRole", edge.role)] })),
    outputs: usable.filter(edge => edge.to === node.id && edge.role === "generated_by").map(edge => ({ resource: { ref: refs.get(edge.from) } })),
    properties: [property("evidenceClass", "DECLARED_UNVERIFIED"), property("runNode", refs.get(node.id))],
  }));
  bom.components = [...(bom.components || []), ...components];
  if (workflows.length) bom.formulation = [...(bom.formulation || []), { "bom-ref": `${prefix}formulation`, workflows }];
  assertUniqueBomReferences(bom);
  bom.properties = [...(bom.properties || []), property("schema", ir.schema), property("irSha256", ir.provenance_ir_sha256), property("source", ir.source), property("inputSha256", ir.input_sha256), property("status", ir.verdict.status), property("coverage", ir.coverage), property("relationships", ir.relationships), property("checks", ir.checks), property("observedFiles", ir.observations.map(({ node_ref, sha256, byte_length }) => ({ node_ref, sha256, byte_length }))), property("fieldLedger", ir.field_ledger), property("interpretationBoundary", ir.interpretation_boundary)];
  // Revisions of the evidence document must not reuse the original document's
  // identity with changed content. The model component identity remains stable.
  const hash = sha256TextHex(canonicalJson(bom));
  // UUIDv8 carries a custom deterministic digest; UUIDv5 would imply SHA-1
  // namespace/name derivation, which this document hashing scheme does not use.
  bom.serialNumber = `urn:uuid:${hash.slice(0, 8)}-${hash.slice(8, 12)}-8${hash.slice(13, 16)}-a${hash.slice(17, 20)}-${hash.slice(20, 32)}`;
  bom.version = 1;
  return bom;
}

function assertUniqueBomReferences(document) {
  const pending = [document], seen = new Set(); let visited = 0;
  while (pending.length) {
    const value = pending.pop(); if (!value || typeof value !== "object") continue;
    requireProvenance(++visited <= 100000, "BOM reference inventory exceeds its traversal bound.");
    if (!Array.isArray(value) && Object.hasOwn(value, "bom-ref")) {
      requireProvenance(typeof value["bom-ref"] === "string" && !seen.has(value["bom-ref"]), "BOM projection contains a duplicate or invalid bom-ref.");
      seen.add(value["bom-ref"]);
    }
    pending.push(...Object.values(value).filter(item => item && typeof item === "object"));
  }
}
