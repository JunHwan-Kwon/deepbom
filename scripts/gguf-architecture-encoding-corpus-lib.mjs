import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { deepBomCorpusCacheDir, ensureHashBoundSourceFile } from "./hash-bound-corpus-cache.mjs";

export const GGUF_CORPUS_PATH = "corpus/gguf-architecture-encoding-corpus.v1.json";
export const GGUF_CORPUS_SCHEMA = "deepbom.gguf_architecture_encoding_corpus.v1";
export const GGUF_RECEIPT_SCHEMA = "deepbom.gguf_architecture_encoding_corpus_receipt.v1";

export async function readGgufArchitectureEncodingCorpus(filename = GGUF_CORPUS_PATH) {
  return validateGgufArchitectureEncodingCorpus(JSON.parse(await readFile(filename, "utf8")));
}

export function validateGgufArchitectureEncodingCorpus(manifest, { requireBaselines = false } = {}) {
  if (manifest?.schema !== GGUF_CORPUS_SCHEMA || manifest.format !== "gguf") throw new Error(`Unsupported GGUF corpus schema ${manifest?.schema || "missing"}.`);
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length !== 8 || manifest.artifact_count !== 8) throw new Error("GGUF corpus must contain eight selected strata.");
  const ids = new Set();
  let encodingStrata = 0;
  const families = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) throw new Error(`Invalid or duplicate GGUF corpus id ${artifact.id || "missing"}.`);
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(artifact.repository || "")) || !/^[0-9a-f]{40}$/.test(String(artifact.revision || ""))) throw new Error(`${artifact.id}: repository or revision is invalid.`);
    if (!['encoding_stratum', 'architecture_stratum'].includes(artifact.purpose)) throw new Error(`${artifact.id}: purpose is invalid.`);
    if (artifact.purpose === "encoding_stratum") encodingStrata += 1;
    families.add(artifact.model_family);
    const file = artifact.file;
    if (!file?.path?.endsWith(".gguf") || !Number.isSafeInteger(file.size_bytes) || file.size_bytes < 1 || !/^[0-9a-f]{64}$/.test(String(file.sha256 || ""))) throw new Error(`${artifact.id}: file identity is invalid.`);
    if (artifact.repository_visibility !== "public_ungated" || !["not_declared", "declared_mit"].includes(artifact.license_metadata_status)) throw new Error(`${artifact.id}: source boundary changed without a schema update.`);
    if (requireBaselines) validateBaseline(artifact);
    ids.add(artifact.id);
  }
  if (encodingStrata !== 5 || families.size < 4) throw new Error("GGUF corpus strata do not cover five encoding candidates and four model families.");
  return manifest;
}

function validateBaseline(artifact) {
  const baseline = artifact.baseline;
  if (!baseline || !baseline.architecture || !Number.isSafeInteger(baseline.tensor_count) || baseline.tensor_count < 1) throw new Error(`${artifact.id}: analyzed baseline is missing.`);
  if (!Array.isArray(baseline.encoding_histogram) || !baseline.encoding_histogram.length || baseline.encoding_histogram.some((row) => !row.encoding || !Number.isSafeInteger(row.tensor_count) || row.tensor_count < 1)) throw new Error(`${artifact.id}: encoding histogram is invalid.`);
  if (baseline.encoding_histogram.reduce((sum, row) => sum + row.tensor_count, 0) !== baseline.tensor_count) throw new Error(`${artifact.id}: encoding histogram does not conserve tensor count.`);
  for (const key of ["artifact_sha256", "analysis_receipt_sha256"]) if (!/^[0-9a-f]{64}$/.test(String(baseline[key] || ""))) throw new Error(`${artifact.id}: ${key} is invalid.`);
}

export function ggufCorpusCacheDir() {
  return deepBomCorpusCacheDir("gguf-architecture-encoding-corpus-v1");
}

export async function analyzeGgufCorpusArtifact(cacheDir, artifact, { offline = false } = {}) {
  const cached = await ensureHashBoundSourceFile(cacheDir, artifact, artifact.file, { offline });
  const bytes = await readFile(cached.filename);
  const { analysis } = await readMetadataModelFile(new File([bytes], artifact.file.path), "gguf");
  const counts = new Map();
  for (const tensor of analysis.tensors || []) {
    const encoding = String(tensor.encoding || tensor.dtype || "UNKNOWN");
    counts.set(encoding, (counts.get(encoding) || 0) + 1);
  }
  const receiptBody = {
    artifact_id: artifact.id,
    repository: artifact.repository,
    revision: artifact.revision,
    source_file: artifact.file,
    declared_quantization_label: artifact.declared_quantization_label,
    artifact_sha256: artifact.file.sha256,
    architecture: analysis.gguf?.architecture || null,
    tensor_count: Number(analysis.tensor_count || 0),
    payload_byte_length: Number(analysis.gguf?.payload_byte_length || 0),
    payload_coverage_status: analysis.gguf?.payload_coverage_status || null,
    numerical_integrity_status: analysis.tensor_numerical_integrity?.status || null,
    unsupported_encoding_tensor_count: Number(analysis.quantization_status?.unsupported_encoding_tensor_count || 0),
    encoding_histogram: [...counts.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([encoding, tensor_count]) => ({ encoding, tensor_count })),
    semantic_contract_status: analysis.gguf?.semantic_contract?.status || analysis.on_device_llm?.status || null,
    llm_contract_schema: analysis.on_device_llm?.schema || null,
  };
  return { schema: GGUF_RECEIPT_SCHEMA, ...receiptBody, analysis_receipt_sha256: sha256Text(canonicalJson(receiptBody)) };
}

export function baselineFromReceipt(receipt) {
  return {
    artifact_sha256: receipt.artifact_sha256,
    architecture: receipt.architecture,
    tensor_count: receipt.tensor_count,
    payload_byte_length: receipt.payload_byte_length,
    payload_coverage_status: receipt.payload_coverage_status,
    numerical_integrity_status: receipt.numerical_integrity_status,
    unsupported_encoding_tensor_count: receipt.unsupported_encoding_tensor_count,
    encoding_histogram: receipt.encoding_histogram,
    semantic_contract_status: receipt.semantic_contract_status,
    analysis_receipt_sha256: receipt.analysis_receipt_sha256,
  };
}

export function receiptMatchesBaseline(receipt, artifact) {
  return canonicalJson(baselineFromReceipt(receipt)) === canonicalJson(artifact.baseline);
}

function sha256Text(value) {
  return createHash("sha256").update(value).digest("hex");
}
