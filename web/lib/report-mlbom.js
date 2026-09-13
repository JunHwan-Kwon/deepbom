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
  const evidenceMode = options.evidenceMode || "standalone";
  const compatibility = buildMlBomCompatibilityProjection(analysis, {
    ...options,
    detailLocation: evidenceMode === "bundle"
      ? "engineering_evidence.json#/evidence/static_analysis"
      : "#/declarations/evidence/0/data/0/contents/attachment",
  });
  const document = buildCycloneDxEvidenceDocument(analysis, {
    ...options,
    evidenceMode,
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
