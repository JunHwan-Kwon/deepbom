import { validateModelIr } from "./model-ir.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { sealDocument } from "./evidence-identity.js";

// Shared by optional evidence layers. Adding a sidecar must not change Model IR.
export function modelEvidenceSource(input) {
  const model = validateModelIr(input);
  return { model, source: { model_ir_sha256: model.model_ir_sha256, artifact_sha256: model.artifact.sha256, artifact_set_sha256: model.artifact_set.artifact_set_sha256 } };
}

export function evidenceHashContract(field) {
  return { algorithm: "SHA-256", canonicalization: "RFC8785-JCS", source_encoding: "UTF-8", excluded_pointers: [`/${field}`] };
}

export function sealEvidence(body, field) {
  return sealDocument(body, field);
}

export function assertEvidenceDigest(document, field) {
  canonicalJson(document);
  const { [field]: digest, ...body } = document;
  if (typeof digest !== "string" || !/^[a-f0-9]{64}$/.test(digest) || digest !== sha256TextHex(canonicalJson(body))) throw new Error(`Evidence IR: ${field} mismatch.`);
}

export function freezeEvidence(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeEvidence(child);
    Object.freeze(value);
  }
  return value;
}
