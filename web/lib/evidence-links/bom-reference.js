import { requireLink, LINK_LIMITS } from "./contracts.js";
import { canonicalJson } from "../report-utils.js";

// Reference resolution only. Attribute reconciliation is a distinct operation.
export function inspectBomReference(node, observation, modelSha256, refersToCurrentModel, observationIndex) {
  const checks = [], fields = [];
  const add = (kind, status, message) => checks.push({ subject_ref: node.id, kind, status, message });
  if (!node.bom) { add("bom_reference", "not_assessed", "BOM document/element identity was not supplied."); return { checks, fields }; }
  if (!observation?.document) { add("bom_reference", "unresolved", "BOM bytes were not supplied; a URI is not fetched automatically."); return { checks, fields }; }
  const bom = observation.document, reference = node.bom;
  requireLink(new TextEncoder().encode(canonicalJson(bom)).length <= LINK_LIMITS.input_bytes, "parsed BOM exceeds 2 MiB.");
  if (reference.format !== "CycloneDX" || reference.spec_version !== "1.7") {
    add("bom_reference", "unsupported", "This resolver supports CycloneDX 1.7 reference identity; other formats remain preserved and unresolved.");
    return { checks, fields };
  }
  if (bom.bomFormat !== reference.format || bom.specVersion !== reference.spec_version || bom.serialNumber !== reference.document_id || bom.version !== reference.document_version) {
    add("bom_document_identity", "mismatch", "BOM format, specification version, document identity or revision differs from the reference.");
    return { checks, fields };
  }
  add("bom_document_identity", "match", "Supplied BOM document identity matches; this is not full schema or signature validation.");
  const inventory = [], pending = [...(bom.metadata?.component ? [{ component: bom.metadata.component, pointer: "/metadata/component" }] : []), ...(Array.isArray(bom.components) ? bom.components.map((component, index) => ({ component, pointer: `/components/${index}` })) : [])];
  while (pending.length) {
    requireLink(inventory.length < 4096, "BOM component reference inventory exceeds 4096 entries.");
    const entry = pending.pop(), { component, pointer } = entry;
    requireLink(component && typeof component === "object" && !Array.isArray(component), "BOM component must be an object.");
    inventory.push(entry);
    if (component.components !== undefined) { requireLink(Array.isArray(component.components), "BOM components must be an array."); pending.push(...component.components.map((component, index) => ({ component, pointer: `${pointer}/components/${index}` }))); }
  }
  const refs = inventory.map(({ component }) => component["bom-ref"]).filter(value => typeof value === "string");
  if (new Set(refs).size !== refs.length) add("bom_element_identity", "mismatch", "BOM contains duplicate component references.");
  const selected = inventory.filter(({ component }) => component["bom-ref"] === reference.element_ref);
  if (selected.length !== 1) { add("bom_element_identity", selected.length ? "mismatch" : "unresolved", "The BOM element does not resolve to exactly one component."); return { checks, fields }; }
  add("bom_element_identity", "match", "Referenced component was found in the supplied document.");
  const { component, pointer } = selected[0];
  if (refersToCurrentModel) {
    add("bom_subject_kind", component.type === "machine-learning-model" ? "match" : "mismatch", "A model BOM reference must resolve to a machine-learning-model component.");
    const hashes = (Array.isArray(component.hashes) ? component.hashes : []).filter(hash => hash?.alg === "SHA-256");
    add("bom_model_sha256", hashes.length === 0 ? "not_assessed" : hashes.length === 1 && String(hashes[0].content).toLowerCase() === modelSha256 ? "match" : "mismatch", hashes.length ? "Compare the referenced component SHA-256 with the active model bytes." : "The referenced component does not declare SHA-256.");
  }
  const properties = component.properties || [];
  requireLink(Array.isArray(properties) && properties.length <= 4096, "BOM property inventory exceeds its bound.");
  properties.forEach((property, index) => fields.push({ source_document: "evidence_link_ir", source_pointer: `/observations/${observationIndex}/document${pointer}/properties/${index}`, subject_ref: node.id, name: typeof property?.name === "string" ? property.name : "<invalid property>", value: property?.value ?? null, status: "preserved_unsupported_mapping" }));
  if (properties.length) add("bom_attribute_reconciliation", "unsupported", "Component properties are inventoried, not verified by reference resolution. Use verify --bom for supported artifact-attribute reconciliation.");
  return { checks, fields };
}
