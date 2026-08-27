import { buildCycloneDxEvidenceDocument } from "./report-export-contracts.js";
import { buildMlBomCompatibilityProjection } from "./report-mlbom-compat.js";

function mergeProperties(primary = [], compatibility = []) {
  const values = new Map(primary.map((item) => [item.name, item]));
  for (const item of compatibility) {
    if (!values.has(item.name)) values.set(item.name, item);
  }
  return [...values.values()];
}

export function buildMlBomDocument(analysis, options = {}) {
  const compatibility = buildMlBomCompatibilityProjection(analysis, options);
  const document = buildCycloneDxEvidenceDocument(analysis, {
    ...options,
    engineeringEvidenceUrl: options.engineeringEvidenceUrl || "engineering_evidence.json",
    generatedAt: options.timestamp || new Date().toISOString(),
    runtimeEvidence: options.runtimeAssignmentEvidence || null,
    author: { name: "DEEPBOM", email: "", orcid: "" },
  });

  document.metadata.component.properties = mergeProperties(
    document.metadata.component.properties,
    compatibility.componentProperties,
  );
  document.properties = mergeProperties(document.properties, compatibility.documentProperties);
  if (compatibility.serialNumber) document.serialNumber = compatibility.serialNumber;
  return document;
}
