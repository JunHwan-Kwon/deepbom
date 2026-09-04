import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const CONVERSION_RECEIPT_SCHEMA = "deepbom.conversion_receipt.v1";
export const BOUND_CONVERSION_RECEIPT_SCHEMA = "deepbom.bound_conversion_receipt.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const FORMAT = /^[a-z0-9][a-z0-9_.+-]{0,63}$/;
const DECIMAL = /^(0|[1-9][0-9]*)$/;
const SECRET_ARGUMENT = /(?:^|[-_])(?:token|password|passwd|secret|api[-_]?key|authorization)(?:=|$)|\bbearer\s+/i;
const RECEIPT_KEYS = new Set([
  "schema", "hash_contract", "created_at", "source_artifacts", "converter",
  "supporting_inputs", "output_artifact", "interpretation_boundary", "receipt_sha256",
]);

function plainObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function exactKeys(value, allowed, label) {
  for (const key of Object.keys(value)) if (!allowed.has(key)) throw new Error(`${label} contains unsupported field ${key}.`);
}

function text(value, label, maximum = 1024) {
  const normalized = String(value ?? "").trim();
  if (!normalized || normalized.length > maximum) throw new Error(`${label} must be a non-empty string no longer than ${maximum} characters.`);
  return normalized;
}

function digest(value, label) {
  const normalized = String(value || "").toLowerCase();
  if (!SHA256.test(normalized)) throw new Error(`${label} must be a lowercase SHA-256 digest.`);
  return normalized;
}

function optionalDigest(value, label) {
  return value == null || value === "" ? null : digest(value, label);
}

function decimal(value, label) {
  if (value == null || value === "") return null;
  const normalized = String(value);
  if (!DECIMAL.test(normalized)) throw new Error(`${label} must be an unsigned decimal integer string.`);
  return normalized;
}

function format(value, label) {
  const normalized = String(value || "").trim().toLowerCase();
  if (!FORMAT.test(normalized)) throw new Error(`${label} is invalid.`);
  return normalizeFormat(normalized);
}

function normalizeFormat(value) {
  if (["mlmodel", "mlpackage"].includes(value)) return "coreml";
  if (["pte", "ptd"].includes(value)) return "executorch";
  return value;
}

function artifactRecord(value, label) {
  const row = plainObject(value, label);
  exactKeys(row, new Set(["filename", "format", "sha256", "byte_length_decimal"]), label);
  return {
    filename: text(row.filename, `${label}.filename`),
    format: format(row.format, `${label}.format`),
    sha256: digest(row.sha256, `${label}.sha256`),
    byte_length_decimal: decimal(row.byte_length_decimal, `${label}.byte_length_decimal`),
  };
}

function supportingInput(value, index) {
  const label = `supporting_inputs[${index}]`;
  const row = plainObject(value, label);
  exactKeys(row, new Set(["role", "filename", "sha256", "byte_length_decimal"]), label);
  return {
    role: text(row.role, `${label}.role`, 120),
    filename: text(row.filename, `${label}.filename`),
    sha256: digest(row.sha256, `${label}.sha256`),
    byte_length_decimal: decimal(row.byte_length_decimal, `${label}.byte_length_decimal`),
  };
}

function converterRecord(value) {
  const row = plainObject(value, "converter");
  exactKeys(row, new Set(["name", "version", "executable_sha256", "invocation", "environment"]), "converter");
  const invocation = plainObject(row.invocation, "converter.invocation");
  exactKeys(invocation, new Set(["argv"]), "converter.invocation");
  if (!Array.isArray(invocation.argv) || invocation.argv.length < 1 || invocation.argv.length > 128) {
    throw new Error("converter.invocation.argv must contain between 1 and 128 arguments.");
  }
  const argv = invocation.argv.map((argument, index) => {
    const normalized = text(argument, `converter.invocation.argv[${index}]`, 4096);
    if (SECRET_ARGUMENT.test(normalized)) throw new Error(`converter.invocation.argv[${index}] appears to contain a credential-bearing argument.`);
    return normalized;
  });
  const environment = plainObject(row.environment, "converter.environment");
  exactKeys(environment, new Set([
    "manifest_sha256", "os", "architecture", "runtime", "container_image_digest",
  ]), "converter.environment");
  const manifestSha256 = digest(environment.manifest_sha256, "converter.environment.manifest_sha256");
  const containerImageDigest = environment.container_image_digest == null || environment.container_image_digest === ""
    ? null
    : text(environment.container_image_digest, "converter.environment.container_image_digest", 256);
  if (containerImageDigest && !/^sha256:[a-f0-9]{64}$/.test(containerImageDigest)) {
    throw new Error("converter.environment.container_image_digest must use the sha256:<digest> form.");
  }
  return {
    name: text(row.name, "converter.name", 256),
    version: text(row.version, "converter.version", 256),
    executable_sha256: optionalDigest(row.executable_sha256, "converter.executable_sha256"),
    invocation: { argv },
    environment: {
      manifest_sha256: manifestSha256,
      os: row.environment.os == null ? null : text(row.environment.os, "converter.environment.os", 120),
      architecture: row.environment.architecture == null ? null : text(row.environment.architecture, "converter.environment.architecture", 120),
      runtime: row.environment.runtime == null ? null : text(row.environment.runtime, "converter.environment.runtime", 256),
      container_image_digest: containerImageDigest,
    },
  };
}

function hashContract(value) {
  const row = plainObject(value, "hash_contract");
  exactKeys(row, new Set(["algorithm", "canonicalization", "source_encoding", "excluded_pointers"]), "hash_contract");
  if (row.algorithm !== "SHA-256" || row.canonicalization !== "RFC8785-JCS" || row.source_encoding !== "UTF-8"
    || JSON.stringify(row.excluded_pointers) !== JSON.stringify(["/receipt_sha256"])) {
    throw new Error("Conversion receipt hash contract is invalid.");
  }
  return {
    algorithm: "SHA-256",
    canonicalization: "RFC8785-JCS",
    source_encoding: "UTF-8",
    excluded_pointers: ["/receipt_sha256"],
  };
}

function normalizedReceiptBody(document) {
  const row = plainObject(document, "conversion receipt");
  exactKeys(row, RECEIPT_KEYS, "conversion receipt");
  if (row.schema !== CONVERSION_RECEIPT_SCHEMA) throw new Error("Conversion receipt schema identity is invalid.");
  const createdAt = text(row.created_at, "created_at", 64);
  if (!Number.isFinite(Date.parse(createdAt)) || new Date(createdAt).toISOString() !== createdAt) {
    throw new Error("created_at must be a canonical ISO-8601 timestamp.");
  }
  if (!Array.isArray(row.source_artifacts) || row.source_artifacts.length < 1 || row.source_artifacts.length > 128) {
    throw new Error("source_artifacts must contain between 1 and 128 hash-identified inputs.");
  }
  if (row.supporting_inputs != null && (!Array.isArray(row.supporting_inputs) || row.supporting_inputs.length > 256)) {
    throw new Error("supporting_inputs must be an array with no more than 256 entries.");
  }
  const sourceArtifacts = row.source_artifacts.map((item, index) => artifactRecord(item, `source_artifacts[${index}]`));
  const supportingInputs = (row.supporting_inputs || []).map(supportingInput);
  const identities = [...sourceArtifacts, ...supportingInputs].map((item) => `${item.sha256}:${item.filename}`);
  if (new Set(identities).size !== identities.length) throw new Error("Conversion receipt input identities must be unique.");
  return {
    schema: CONVERSION_RECEIPT_SCHEMA,
    hash_contract: hashContract(row.hash_contract),
    created_at: createdAt,
    source_artifacts: sourceArtifacts,
    converter: converterRecord(row.converter),
    supporting_inputs: supportingInputs,
    output_artifact: artifactRecord(row.output_artifact, "output_artifact"),
    interpretation_boundary: text(row.interpretation_boundary, "interpretation_boundary", 2400),
  };
}

export function buildConversionReceipt(input) {
  const body = normalizedReceiptBody({
    ...input,
    schema: CONVERSION_RECEIPT_SCHEMA,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      source_encoding: "UTF-8",
      excluded_pointers: ["/receipt_sha256"],
    },
  });
  return Object.freeze({ ...body, receipt_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateConversionReceipt(document) {
  const body = normalizedReceiptBody(document);
  const observed = digest(document.receipt_sha256, "receipt_sha256");
  const expected = sha256TextHex(canonicalJson(body));
  if (observed !== expected) throw new Error("Conversion receipt SHA-256 is invalid.");
  return Object.freeze({ ...body, receipt_sha256: observed });
}

export function bindConversionReceipt(document, activeArtifact, { receiptFileSha256 = null } = {}) {
  const receipt = validateConversionReceipt(document);
  const active = artifactRecord(activeArtifact, "active_artifact");
  if (receipt.output_artifact.sha256 !== active.sha256) {
    throw new Error("Conversion receipt output SHA-256 is not bound to the active artifact.");
  }
  if (receipt.output_artifact.format !== active.format) {
    throw new Error("Conversion receipt output format is not bound to the active artifact format.");
  }
  const body = {
    schema: BOUND_CONVERSION_RECEIPT_SCHEMA,
    receipt_sha256: receipt.receipt_sha256,
    receipt_file_sha256: optionalDigest(receiptFileSha256, "receipt_file_sha256"),
    active_artifact: active,
    status: "exact_output_hash_and_format_match",
    evidence_class: "DECLARED_UNVERIFIED",
    output_binding_evidence_class: "OBSERVED",
    receipt,
    interpretation_boundary: "The active deployment artifact hash and format are observed and exactly bound. Source artifacts, converter identity, invocation, and environment are declarations carried by the receipt and are not re-executed or independently observed by this audit.",
  };
  return Object.freeze({ ...body, binding_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateBoundConversionReceipt(document, activeArtifact = null) {
  const row = plainObject(document, "bound conversion receipt");
  exactKeys(row, new Set([
    "schema", "receipt_sha256", "receipt_file_sha256", "active_artifact", "status",
    "evidence_class", "output_binding_evidence_class", "receipt", "interpretation_boundary", "binding_sha256",
  ]), "bound conversion receipt");
  if (row.schema !== BOUND_CONVERSION_RECEIPT_SCHEMA
    || row.status !== "exact_output_hash_and_format_match"
    || row.evidence_class !== "DECLARED_UNVERIFIED"
    || row.output_binding_evidence_class !== "OBSERVED") {
    throw new Error("Bound conversion receipt classification is invalid.");
  }
  const receipt = validateConversionReceipt(row.receipt);
  if (digest(row.receipt_sha256, "receipt_sha256") !== receipt.receipt_sha256) throw new Error("Bound receipt identity is inconsistent.");
  const active = artifactRecord(row.active_artifact, "active_artifact");
  if (receipt.output_artifact.sha256 !== active.sha256 || receipt.output_artifact.format !== active.format) {
    throw new Error("Bound conversion receipt output identity is inconsistent.");
  }
  if (activeArtifact) {
    const expectedActive = artifactRecord(activeArtifact, "expected_active_artifact");
    if (active.sha256 !== expectedActive.sha256 || active.format !== expectedActive.format) {
      throw new Error("Bound conversion receipt does not match the expected active artifact.");
    }
  }
  optionalDigest(row.receipt_file_sha256, "receipt_file_sha256");
  text(row.interpretation_boundary, "interpretation_boundary", 2400);
  const body = { ...row };
  const observed = digest(body.binding_sha256, "binding_sha256");
  delete body.binding_sha256;
  if (observed !== sha256TextHex(canonicalJson(body))) throw new Error("Bound conversion receipt SHA-256 is invalid.");
  return Object.freeze({ ...body, binding_sha256: observed });
}

export function buildConversionLineageEvidence(analysis, activeArtifact) {
  if (!analysis?.conversion_receipt) return Object.freeze({
    status: "not_provided",
    evidence_class: "NOT_ASSESSABLE",
    conversion: null,
    interpretation_boundary: "No hash-bound conversion receipt was provided. Converter, source checkpoint, command, and environment lineage are not inferred from the deployment artifact.",
  });
  const conversion = validateBoundConversionReceipt(analysis.conversion_receipt, activeArtifact);
  return Object.freeze({
    status: "output_bound_source_declared",
    evidence_class: "DECLARED_UNVERIFIED",
    conversion,
    interpretation_boundary: conversion.interpretation_boundary,
  });
}
