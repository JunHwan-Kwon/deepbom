import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { deepBomCorpusCacheDir, ensureHashBoundSourceFile, hashBoundSourcePath, huggingFaceRevisionUrl } from "./hash-bound-corpus-cache.mjs";

export const SAFETENSORS_CORPUS_PATH = "corpus/safetensors-architecture-corpus.v1.json";
export const SAFETENSORS_CORPUS_SCHEMA = "deepbom.safetensors_architecture_corpus.v1";
export const SAFETENSORS_RECEIPT_SCHEMA = "deepbom.safetensors_architecture_corpus_receipt.v1";

export async function readSafeTensorsArchitectureCorpus(filename = SAFETENSORS_CORPUS_PATH) {
  return validateSafeTensorsArchitectureCorpus(JSON.parse(await readFile(filename, "utf8")));
}

export function validateSafeTensorsArchitectureCorpus(manifest, { requireBaselines = false } = {}) {
  if (manifest?.schema !== SAFETENSORS_CORPUS_SCHEMA || manifest.format !== "safetensors") {
    throw new Error(`Unsupported SafeTensors architecture corpus schema: ${manifest?.schema || "missing"}.`);
  }
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length !== 3 || manifest.artifact_count !== 3) {
    throw new Error("The SafeTensors architecture corpus must contain exactly three family anchors.");
  }
  const expectedClasses = new Set(["dense_decoder", "sparse_moe_decoder", "ssm_recurrent"]);
  const ids = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) throw new Error(`Invalid or duplicate corpus id ${artifact.id || "missing"}.`);
    if (!expectedClasses.delete(artifact.architecture_class)) throw new Error(`${artifact.id}: architecture class is invalid or duplicated.`);
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(artifact.repository || "")) || !/^[0-9a-f]{40}$/.test(String(artifact.revision || ""))) {
      throw new Error(`${artifact.id}: repository or immutable revision is invalid.`);
    }
    if (artifact.repository_visibility !== "public_ungated" || artifact.license_metadata_status !== "not_declared") {
      throw new Error(`${artifact.id}: source visibility or license boundary changed without a corpus schema update.`);
    }
    if (!Array.isArray(artifact.files) || artifact.files.length !== 2 || artifact.files.map((row) => row.path).sort().join("|") !== "config.json|model.safetensors") {
      throw new Error(`${artifact.id}: config and SafeTensors file inventory is incomplete.`);
    }
    for (const file of artifact.files) {
      if (!Number.isSafeInteger(file.size_bytes) || file.size_bytes < 1 || !/^[0-9a-f]{64}$/.test(String(file.sha256 || ""))) {
        throw new Error(`${artifact.id}/${file.path}: byte identity is invalid.`);
      }
    }
    if (requireBaselines) validateBaseline(artifact);
    ids.add(artifact.id);
  }
  if (expectedClasses.size) throw new Error(`Missing architecture classes: ${[...expectedClasses].join(", ")}.`);
  return manifest;
}

function validateBaseline(artifact) {
  const baseline = artifact.baseline;
  if (!baseline || baseline.status !== "assessed" || baseline.architecture_kind !== artifact.architecture_class) {
    throw new Error(`${artifact.id}: assessed architecture baseline is missing or mismatched.`);
  }
  for (const key of ["tensor_count", "payload_byte_length", "canonical_tensor_shape_mismatch_count"]) {
    if (!Number.isSafeInteger(baseline[key]) || baseline[key] < 0) throw new Error(`${artifact.id}: baseline ${key} is invalid.`);
  }
  for (const key of ["bundle_sha256", "analysis_receipt_sha256"]) {
    if (!/^[0-9a-f]{64}$/.test(String(baseline[key] || ""))) throw new Error(`${artifact.id}: baseline ${key} is invalid.`);
  }
}

export function safeTensorsCorpusCacheDir() {
  return deepBomCorpusCacheDir("safetensors-architecture-corpus-v1");
}

export function corpusFilePath(cacheDir, artifact, file) {
  return hashBoundSourcePath(cacheDir, artifact, file);
}

export function corpusFileUrl(artifact, file) {
  return huggingFaceRevisionUrl(artifact, file);
}

export async function ensureCorpusFile(cacheDir, artifact, file, { offline = false } = {}) {
  return ensureHashBoundSourceFile(cacheDir, artifact, file, { offline });
}

export async function analyzeSafeTensorsCorpusArtifact(cacheDir, artifact, { offline = false } = {}) {
  const selected = [];
  for (const file of artifact.files) {
    const cached = await ensureCorpusFile(cacheDir, artifact, file, { offline });
    const bytes = await readFile(cached.filename);
    const browserFile = new File([bytes], path.basename(file.path), { type: file.path.endsWith(".json") ? "application/json" : "application/octet-stream" });
    Object.defineProperty(browserFile, "webkitRelativePath", { value: `${artifact.id}/${file.path}` });
    selected.push(browserFile);
  }
  const { analysis } = await readArtifactBundle(selected);
  const contract = analysis.safetensors?.hf_architecture_contract || {};
  const receiptBody = {
    artifact_id: artifact.id,
    repository: artifact.repository,
    revision: artifact.revision,
    source_files: artifact.files.map((file) => ({ path: file.path, size_bytes: file.size_bytes, sha256: file.sha256 })),
    bundle_sha256: analysis.artifact_bundle?.bundle_sha256 || null,
    status: contract.status || "missing",
    model_type: contract.model_type || null,
    architecture_kind: contract.architecture_kind || (contract.status === "assessed" ? "dense_decoder" : null),
    tensor_count: Number(analysis.safetensors?.tensor_count || 0),
    payload_byte_length: Number(analysis.safetensors?.payload_byte_length || 0),
    payload_coverage_status: analysis.safetensors?.payload_coverage_status || null,
    numerical_integrity_status: analysis.tensor_numerical_integrity?.status || null,
    canonical_tensor_shape_mismatch_count: Number(contract.tensor_contract?.canonical_tensor_shape_mismatch_count || 0),
    canonical_tensor_missing_count: Number(contract.tensor_contract?.canonical_tensor_missing_count || 0),
    canonical_tensor_unexpected_count: Number(contract.tensor_contract?.canonical_tensor_unexpected_count || 0),
    llm_contract_schema: analysis.on_device_llm?.schema || null,
    llm_architecture_kind: analysis.on_device_llm?.architecture?.kind || null,
    projection: projectionReceipt(contract),
  };
  return {
    schema: SAFETENSORS_RECEIPT_SCHEMA,
    ...receiptBody,
    analysis_receipt_sha256: sha256Text(canonicalJson(receiptBody)),
  };
}

function projectionReceipt(contract) {
  if (contract.architecture_kind === "sparse_moe_decoder") return {
    expert_count: contract.fields?.num_local_experts ?? null,
    active_experts_per_token: contract.fields?.num_experts_per_tok ?? null,
    active_projection_macs_per_layer_per_token: contract.compute_projection?.active_projection_macs_per_layer_per_token?.decimal ?? null,
  };
  if (contract.architecture_kind === "ssm_recurrent") return {
    state_size: contract.fields?.state_size ?? null,
    recurrent_state_elements_all_layers_per_batch: contract.recurrent_state_projection?.recurrent_state_elements_all_layers_per_batch?.decimal ?? null,
    accounted_macs_per_layer_per_token: contract.compute_projection?.accounted_macs_per_layer_per_token?.decimal ?? null,
  };
  return {
    kv_elements_per_token_all_layers: contract.kv_state_projection?.elements_per_token_per_batch?.decimal ?? null,
    dense_projection_macs_all_layers_per_token: contract.compute_projection?.dense_projection_macs_all_layers_per_token?.decimal ?? null,
  };
}

export function baselineFromReceipt(receipt) {
  return {
    status: receipt.status,
    architecture_kind: receipt.architecture_kind,
    tensor_count: receipt.tensor_count,
    payload_byte_length: receipt.payload_byte_length,
    canonical_tensor_shape_mismatch_count: receipt.canonical_tensor_shape_mismatch_count,
    canonical_tensor_missing_count: receipt.canonical_tensor_missing_count,
    canonical_tensor_unexpected_count: receipt.canonical_tensor_unexpected_count,
    bundle_sha256: receipt.bundle_sha256,
    analysis_receipt_sha256: receipt.analysis_receipt_sha256,
    projection: receipt.projection,
  };
}

export function receiptMatchesBaseline(receipt, artifact) {
  return canonicalJson(baselineFromReceipt(receipt)) === canonicalJson(artifact.baseline);
}

function sha256Text(value) {
  return createHash("sha256").update(value).digest("hex");
}
