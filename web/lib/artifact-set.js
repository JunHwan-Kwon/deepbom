import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const ARTIFACT_SET_SCHEMA = "deepbom.artifact_set.v1";
const SHA256 = /^[a-f0-9]{64}$/;

export function finalizeArtifactSet(document) {
  const body = clone(document);
  delete body.artifact_set_sha256;
  validateBody(body);
  return Object.freeze({ ...body, artifact_set_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateArtifactSet(document) {
  const body = clone(document);
  const declared = String(body.artifact_set_sha256 || "").toLowerCase();
  delete body.artifact_set_sha256;
  validateBody(body);
  if (!SHA256.test(declared) || sha256TextHex(canonicalJson(body)) !== declared) throw new Error("Artifact-set SHA-256 is invalid.");
  return Object.freeze({ ...body, artifact_set_sha256: declared });
}

function validateBody(value) {
  if (value?.schema !== ARTIFACT_SET_SCHEMA || value.evidence_class !== "OBSERVED_ACQUISITION") throw new Error("Artifact-set identity is invalid.");
  const source = value.source;
  if (!source || !["local", "huggingface", "https", "gcs"].includes(source.kind)
    || !text(source.canonical_locator, 2048) || !text(source.immutability?.kind, 80)
    || !text(source.immutability?.value, 256)) throw new Error("Artifact-set source binding is invalid.");
  if (!Array.isArray(value.files) || !value.files.length || value.files.length > 20_000) throw new Error("Artifact-set file inventory is invalid.");
  if (value.subject != null) {
    if (!text(value.subject.filename, 1024) || !text(value.subject.format, 80)
      || !SHA256.test(String(value.subject.sha256 || "")) || !text(value.subject.identity_basis, 120)) {
      throw new Error("Artifact-set subject identity is invalid.");
    }
    validateExactBytes(value.subject.byte_length);
  }
  const paths = new Set();
  for (const file of value.files) {
    if (!["primary", "sidecar", "shard", "manifest", "configuration"].includes(file.role)
      || !safePath(file.path) || paths.has(file.path) || !SHA256.test(String(file.sha256 || ""))) {
      throw new Error("Artifact-set file identity is invalid.");
    }
    validateExactBytes(file.byte_length);
    paths.add(file.path);
  }
  if (value.files.filter((file) => file.role === "primary").length !== 1) throw new Error("Artifact-set requires exactly one primary file.");
  if (value.trust?.model_code_execution !== "forbidden" || value.trust?.pickle_execution !== "forbidden"
    || value.trust?.remote_code_execution !== "forbidden") throw new Error("Artifact-set code-execution boundary is invalid.");
}

function validateExactBytes(value) {
  const decimal = String(value?.decimal || "");
  if (!/^\d+$/.test(decimal) || BigInt(decimal) < 0n) throw new Error("Artifact-set byte length is invalid.");
  const exact = BigInt(decimal);
  const expected = exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null;
  if ((value.number == null ? null : Number(value.number)) !== expected) throw new Error("Artifact-set byte length representations differ.");
}

function safePath(value) {
  const path = String(value || "").replaceAll("\\", "/");
  return path.length > 0 && path.length <= 1024 && !path.startsWith("/") && !/^[A-Za-z]:/.test(path)
    && path.split("/").every((part) => part && part !== "." && part !== "..");
}

function text(value, maximum) {
  const normalized = String(value || "").trim();
  return normalized.length > 0 && normalized.length <= maximum;
}

function clone(value) { return JSON.parse(JSON.stringify(value)); }
