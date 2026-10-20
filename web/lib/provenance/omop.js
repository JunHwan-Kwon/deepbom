import { portableContract } from "../portable-contracts.js";
import { PROVENANCE_INPUT_SCHEMA, OMOP_INPUT_SCHEMA, requireProvenance } from "./contracts.js";
import { canonicalJson } from "../report-utils.js";

// Source semantics: OHDSI CommonDataModel cdm54.html/cdm55.html, CDM_SOURCE.
// This is a DEEPBOM import profile, not an OMOP extension table or vocabulary.
export const OMOP_SOURCE_FIELDS = Object.freeze(["cdm_source_name", "cdm_source_abbreviation", "cdm_holder", "source_description", "source_documentation_reference", "cdm_etl_reference", "source_release_date", "cdm_release_date", "cdm_version", "cdm_version_concept_id", "vocabulary_version"]);
export const OMOP_REQUIRED_SOURCE_FIELDS = Object.freeze(["cdm_source_name", "cdm_source_abbreviation", "cdm_holder", "source_release_date", "cdm_release_date", "cdm_version_concept_id", "vocabulary_version"]);

export function normalizeCdmSource(value) { canonicalJson(value); return portableContract("provenance",{operation:"cdm_source",value}); }

export function omopProfile(value) { canonicalJson(value); return portableContract("provenance",{operation:"omop_profile",value}); }

export function normalizeMetadataInput(value) { canonicalJson(value); return portableContract("provenance",{operation:"normalize",value}); }

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
