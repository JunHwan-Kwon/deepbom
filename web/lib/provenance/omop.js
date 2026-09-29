import { PROVENANCE_INPUT_SCHEMA, OMOP_INPUT_SCHEMA, PRIMARY_MODEL_REF, PROVENANCE_LIMITS, objectKeys, primitive, text, requireProvenance, validateProvenanceInput } from "./contracts.js";
import { canonicalJson } from "../report-utils.js";

// Source semantics: OHDSI CommonDataModel cdm54.html/cdm55.html, CDM_SOURCE.
// This is a DEEPBOM import profile, not an OMOP extension table or vocabulary.
export const OMOP_SOURCE_FIELDS = Object.freeze(["cdm_source_name", "cdm_source_abbreviation", "cdm_holder", "source_description", "source_documentation_reference", "cdm_etl_reference", "source_release_date", "cdm_release_date", "cdm_version", "cdm_version_concept_id", "vocabulary_version"]);
export const OMOP_REQUIRED_SOURCE_FIELDS = Object.freeze(["cdm_source_name", "cdm_source_abbreviation", "cdm_holder", "source_release_date", "cdm_release_date", "cdm_version_concept_id", "vocabulary_version"]);

export function normalizeCdmSource(value) {
  const row = Array.isArray(value) ? (requireProvenance(value.length === 1, "CDM_SOURCE import requires exactly one explicitly selected row."), value[0]) : value;
  requireProvenance(row && typeof row === "object" && !Array.isArray(row), "CDM_SOURCE must be a JSON object or a one-row JSON array.");
  requireProvenance(Object.keys(row).length <= PROVENANCE_LIMITS.attributes, "CDM_SOURCE has too many fields.");
  const normalized = {};
  for (const [key, value] of Object.entries(row)) {
    text(key, "CDM_SOURCE column", { limit: 128 }); primitive(value, `CDM_SOURCE.${key}`);
    const name = key.toLowerCase();
    requireProvenance(/^[a-z][a-z0-9_]*$/.test(name) && !Object.hasOwn(normalized, name), "CDM_SOURCE contains invalid or case-colliding column names.");
    Object.defineProperty(normalized, name, { value, enumerable: true });
  }
  return normalized;
}

export function omopProfile(input) {
  const profile = input.omop;
  objectKeys(profile, ["cdm_version", "instance_id", "release_id", "cdm_source"], [], "omop");
  for (const key of ["cdm_version", "instance_id", "release_id"]) text(profile[key], `omop.${key}`, { limit: 256 });
  const fields = normalizeCdmSource(profile.cdm_source);
  const version = profile.cdm_version.replace(/^v/, "");
  requireProvenance(/^\d+\.\d+(\.\d+)?$/.test(version), "omop.cdm_version must be an explicit version.");
  if (fields.cdm_version != null && fields.cdm_version !== "") requireProvenance(String(fields.cdm_version).replace(/^v/, "") === version, "CDM_SOURCE.cdm_version contradicts omop.cdm_version.");
  for (const key of ["source_release_date", "cdm_release_date"]) {
    const date = fields[key]; if (date == null || date === "") continue;
    requireProvenance(typeof date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(date) && Number.isFinite(Date.parse(date)) && new Date(date).toISOString().slice(0, 10) === date, `CDM_SOURCE.${key} must be a real ISO calendar date.`);
  }
  const concept = fields.cdm_version_concept_id;
  if (concept != null && concept !== "") requireProvenance((Number.isSafeInteger(concept) && concept >= 0) || (typeof concept === "string" && /^(0|[1-9][0-9]*)$/.test(concept)), "CDM_SOURCE.cdm_version_concept_id must be an exact nonnegative integer.");
  const supported = ["5.4", "5.5"].includes(version);
  const known = new Set([...OMOP_SOURCE_FIELDS, ...(version === "5.5" ? ["cdm_release_identifier"] : [])]);
  if (supported) for (const key of known) {
    if (key === "cdm_version_concept_id" || fields[key] == null || fields[key] === "") continue;
    requireProvenance(typeof fields[key] === "string", `CDM_SOURCE.${key} must retain its text or date representation.`);
    const maximum = key === "source_description" ? PROVENANCE_LIMITS.text : key === "cdm_source_abbreviation" ? 25 : key === "vocabulary_version" ? 20 : key === "cdm_version" ? 10 : 255;
    requireProvenance([...fields[key]].length <= maximum, `CDM_SOURCE.${key} exceeds the import profile length bound.`);
  }
  if (version === "5.5" && fields.cdm_release_identifier != null && fields.cdm_release_identifier !== "") requireProvenance(String(fields.cdm_release_identifier) === profile.release_id, "CDM_SOURCE.cdm_release_identifier contradicts omop.release_id.");
  const originalRow = Array.isArray(profile.cdm_source) ? profile.cdm_source[0] : profile.cdm_source;
  const fieldLedger = Object.keys(originalRow).map(column => ({
    source_pointer: `/omop/cdm_source/${Array.isArray(profile.cdm_source) ? "0/" : ""}${column}`, name: `omop:cdm_source:${column.toLowerCase()}`, value: fields[column.toLowerCase()],
    status: supported && known.has(column.toLowerCase()) ? "mapped_declared" : "preserved_unsupported_mapping",
  }));
  const missing = OMOP_REQUIRED_SOURCE_FIELDS.filter(key => fields[key] == null || (typeof fields[key] === "string" && !fields[key].trim()));
  const node = {
    id: "omop:release", kind: "data_release", name: String(typeof fields.cdm_source_name === "string" && fields.cdm_source_name.trim() ? fields.cdm_source_name : profile.instance_id), version: profile.release_id,
    attributes: [
      { name: "omop:instance_id", value: profile.instance_id }, { name: "omop:release_id", value: profile.release_id }, { name: "omop:cdm_version", value: version },
      ...fieldLedger.map(row => ({ name: row.name, value: row.value })),
    ],
  };
  return { node, field_ledger: fieldLedger, missing_fields: missing, status: supported ? "supported_import_profile" : "unsupported_cdm_version", cdm_version: version };
}

export function normalizeMetadataInput(value) {
  const input = validateProvenanceInput(value);
  const profile = input.schema === OMOP_INPUT_SCHEMA ? omopProfile(input) : null;
  return { input, profile, nodes: [...(profile ? [profile.node] : []), ...input.nodes], relationships: [
    ...(profile ? [{ id: "omop:context", from: PRIMARY_MODEL_REF, to: "omop:release", role: "associated_with", evidence_refs: [] }] : []),
    ...input.relationships,
  ] };
}

export function metadataTemplate(model, profile = "omop") {
  requireProvenance(["omop", "generic"].includes(profile), "template profile must be omop or generic.");
  return {
    schema: profile === "omop" ? OMOP_INPUT_SCHEMA : PROVENANCE_INPUT_SCHEMA,
    subject: { artifact_sha256: model.artifact.sha256, artifact_set_sha256: model.artifact_set.artifact_set_sha256 },
    ...(profile === "omop" ? { omop: { cdm_version: "5.5", instance_id: "", release_id: "", cdm_source: {} } } : {}),
    nodes: [], relationships: [],
  };
}

export function fillOmopTemplate(template, sourceRow, { instanceId, releaseId, cdmVersion } = {}) {
  requireProvenance(template.schema === OMOP_INPUT_SCHEMA, "select an OMOP template before importing CDM_SOURCE.");
  const result = JSON.parse(canonicalJson(template)), fields = normalizeCdmSource(sourceRow);
  result.omop.cdm_source = fields;
  result.omop.cdm_version = cdmVersion || (fields.cdm_version ? String(fields.cdm_version).replace(/^v/, "") : result.omop.cdm_version);
  result.omop.instance_id = instanceId || result.omop.instance_id;
  result.omop.release_id = releaseId || fields.cdm_release_identifier || result.omop.release_id;
  return result;
}
