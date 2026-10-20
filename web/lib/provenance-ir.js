import { portableContract } from "./portable-contracts.js";
import { canonicalJson } from "./report-utils.js";
import { modelEvidenceSource, freezeEvidence } from "./ir-evidence-contract.js";
import { PROVENANCE_IR_SCHEMA, PROVENANCE_SUMMARY_SCHEMA, PROVENANCE_DIGEST_FIELD as DIGEST_FIELD } from "./evidence-ir.js";
export { PROVENANCE_IR_SCHEMA };
export function buildProvenanceIr(modelInput,metadata,{observations=[]}={}) {
  const {model}=modelEvidenceSource(modelInput);
  canonicalJson(metadata); canonicalJson(observations);
  return freezeEvidence(portableContract("provenance",{operation:"build",model,value:metadata,observations}));
}
export function validateProvenanceIr(document,modelInput) {
  const {model}=modelEvidenceSource(modelInput);
  canonicalJson(document);
  return freezeEvidence(portableContract("provenance",{operation:"validate",model,value:document}));
}
export function provenanceSummary(document) {
  return { schema: PROVENANCE_SUMMARY_SCHEMA, artifact_sha256: document.source.artifact_sha256, model_ir_sha256: document.source.model_ir_sha256, [DIGEST_FIELD]: document[DIGEST_FIELD], status: document.verdict.status, ...document.coverage, metadata_truth_verified: false, publisher_authenticity: "not_verified" };
}
