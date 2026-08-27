import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { gunzip, gzip } from "node:zlib";
import { promisify } from "node:util";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const gzipAsync = promisify(gzip);
const gunzipAsync = promisify(gunzip);

export const HF_CORPUS_SCHEMA = "deepbom.huggingface_community_corpus.v1";
export const HF_CLASSIFIER_SCHEMA = "deepbom.model_scale_tier_classifier.v1";
export const HF_RECEIPT_SCHEMA = "deepbom.huggingface_download_receipt.v1";
export const HF_SWEEP_SCHEMA = "deepbom.huggingface_community_sweep.v1";
export const HF_RESULT_SCHEMA = "deepbom.huggingface_community_result.v1";
export const HF_CORPUS_PATH = "corpus/huggingface-community-corpus.v1.json.gz";
export const HF_SUMMARY_PATH = "corpus/huggingface-community-corpus.v1.summary.json";
export const HF_ORGANIZATIONS = Object.freeze(["litert-community", "onnx-community"]);

const CURRENT_ANALYZER_FORMATS = new Set(["tflite", "onnx"]);
const LARGE_PIPELINES = new Set([
  "audio-text-to-text",
  "image-text-to-text",
  "image-to-text",
  "text-generation",
  "text-to-image",
  "text-to-video",
  "video-text-to-text",
  "visual-question-answering",
]);
const MICRO_PIPELINES = new Set([
  "audio-classification",
  "image-classification",
  "image-feature-extraction",
  "object-detection",
  "sensor-classification",
]);
const MICRO_EXPLICIT = /\b(?:tflite[\s_-]*micro|tensorflow[\s_-]*lite[\s_-]*micro|tinyml|microcontroller|cortex[\s_-]*m\d*|ethos[\s_-]*u|arduino|embedded[\s_-]*mcu|mcu)\b/i;
const LARGE_FAMILY = /(?:^|[^a-z0-9])(?:llm|vlm|gemma\d*|llama\d*|qwen\d*|mistral\d*|phi[-_ ]?\d+|deepseek|smollm|minicpm|paligemma|causal[-_ ]?lm|vision[-_ ]?language|text[-_ ]?generation|forconditionalgeneration|[a-z0-9]+[_-]vl(?:[^a-z0-9]|$))/i;
const MODEL_EXTENSIONS = new Map([
  [".tflite", { kind: "model", format: "tflite", support: "current" }],
  [".onnx", { kind: "model", format: "onnx", support: "current" }],
  [".litertlm", { kind: "model", format: "litertlm", support: "future_large" }],
  [".task", { kind: "model_bundle", format: "mediapipe_task", support: "future_bundle" }],
  [".pte", { kind: "model", format: "executorch", support: "future_runtime" }],
  [".gguf", { kind: "model", format: "gguf", support: "future_large" }],
  [".safetensors", { kind: "weights", format: "safetensors", support: "metadata_only" }],
  [".bin", { kind: "binary", format: "binary", support: "companion_or_weights" }],
  [".data", { kind: "sidecar", format: "external_data", support: "onnx_companion" }],
  [".onnx_data", { kind: "sidecar", format: "external_data", support: "onnx_companion" }],
]);
const SUPPORT_FILE_NAMES = new Set([
  "added_tokens.json",
  "chat_template.jinja",
  "config.json",
  "generation_config.json",
  "preprocessor_config.json",
  "processor_config.json",
  "special_tokens_map.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
]);

export async function readHfCorpus(filename = HF_CORPUS_PATH) {
  const bytes = await readFile(filename);
  const json = filename.endsWith(".gz") ? await gunzipAsync(bytes) : bytes;
  const manifest = JSON.parse(json.toString("utf8"));
  return validateHfCorpus(manifest);
}

export async function writeHfCorpus(manifest, filename = HF_CORPUS_PATH) {
  validateHfCorpus(manifest);
  const canonical = `${JSON.stringify(sortObjectKeys(manifest), null, 2)}\n`;
  await mkdir(path.dirname(filename), { recursive: true });
  if (filename.endsWith(".gz")) {
    await writeFile(filename, await gzipAsync(Buffer.from(canonical), { level: 9, mtime: 0 }));
  } else {
    await writeFile(filename, canonical, "utf8");
  }
}

export function validateHfCorpus(manifest) {
  if (manifest?.schema !== HF_CORPUS_SCHEMA) throw new Error(`Unsupported Hugging Face corpus schema: ${manifest?.schema || "(missing)"}.`);
  if (manifest.classifier_schema !== HF_CLASSIFIER_SCHEMA) throw new Error("Hugging Face corpus classifier schema is missing or unsupported.");
  if (!Array.isArray(manifest.organizations) || !Array.isArray(manifest.repositories)) throw new Error("Hugging Face corpus collections are missing.");
  const ids = new Set();
  for (const repository of manifest.repositories) {
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(repository.id || "")) || ids.has(repository.id)) {
      throw new Error(`Invalid or duplicate Hugging Face repository id: ${repository.id || "(missing)"}.`);
    }
    if (!/^[0-9a-f]{40}$/.test(String(repository.revision || ""))) throw new Error(`${repository.id}: revision is not a commit SHA.`);
    if (!["micro", "mid", "large"].includes(repository.tier?.name)) throw new Error(`${repository.id}: scale tier is invalid.`);
    if (!["explicit", "derived", "candidate"].includes(repository.tier?.confidence)) throw new Error(`${repository.id}: scale-tier confidence is invalid.`);
    if (!Array.isArray(repository.files)) throw new Error(`${repository.id}: file inventory is missing.`);
    const paths = new Set();
    for (const file of repository.files) {
      if (!isSafeRelativePath(file.path) || paths.has(file.path)) throw new Error(`${repository.id}: invalid or duplicate file path ${file.path}.`);
      if (file.size_bytes !== null && (!Number.isSafeInteger(file.size_bytes) || file.size_bytes < 0)) throw new Error(`${repository.id}/${file.path}: size is invalid.`);
      if (file.lfs_sha256 && !/^[0-9a-f]{64}$/.test(file.lfs_sha256)) throw new Error(`${repository.id}/${file.path}: LFS SHA-256 is invalid.`);
      if (file.blob_id && !/^[0-9a-f]{40,64}$/.test(file.blob_id)) throw new Error(`${repository.id}/${file.path}: blob id is invalid.`);
      paths.add(file.path);
    }
    ids.add(repository.id);
  }
  if (manifest.summary?.repository_count !== manifest.repositories.length) throw new Error("Hugging Face corpus repository count does not conserve the repository inventory.");
  return manifest;
}

export function normalizeHfRepository(detail) {
  const files = (detail.siblings || [])
    .map(normalizeHfFile)
    .sort((left, right) => left.path.localeCompare(right.path));
  const id = String(detail.id || detail.modelId || "");
  const organization = id.split("/")[0] || "";
  const metadata = normalizeUpstreamMetadata(detail);
  const tier = classifyRepository({ id, metadata, files });
  const sizedFiles = files.filter((file) => file.size_bytes !== null);
  const modelFiles = files.filter((file) => file.kind === "model" || file.kind === "model_bundle");
  const modelPayloadFiles = files.filter((file) => ["model", "model_bundle", "weights", "sidecar", "binary"].includes(file.kind));
  const analyzerFiles = files.filter((file) => file.analyzer_support === "current");
  return {
    id,
    organization,
    name: id.slice(organization.length + 1),
    revision: String(detail.sha || ""),
    created_at: detail.createdAt || null,
    last_modified: detail.lastModified || null,
    gated: detail.gated ?? false,
    private: detail.private === true,
    disabled: detail.disabled === true,
    pipeline_tag: detail.pipeline_tag || null,
    library_name: detail.library_name || null,
    tier,
    runtime_family: inferRuntimeFamily(detail, files),
    metadata,
    upstream_metadata: Object.fromEntries(Object.entries(detail)
      .filter(([key]) => !["_id", "id", "modelId", "sha", "siblings"].includes(key))),
    byte_summary: {
      known_file_bytes: sum(sizedFiles, (file) => file.size_bytes),
      unknown_size_file_count: files.length - sizedFiles.length,
      model_container_bytes: sum(modelFiles.filter((file) => file.size_bytes !== null), (file) => file.size_bytes),
      model_payload_bytes: sum(modelPayloadFiles.filter((file) => file.size_bytes !== null), (file) => file.size_bytes),
      analyzer_supported_bytes: sum(analyzerFiles.filter((file) => file.size_bytes !== null), (file) => file.size_bytes),
    },
    file_count: files.length,
    model_file_count: modelFiles.length,
    analyzer_supported_file_count: analyzerFiles.length,
    files,
  };
}

export function normalizeHfFile(file) {
  const filePath = String(file.rfilename || file.path || "");
  const descriptor = classifyFile(filePath);
  const size = Number(file.size ?? file.lfs?.size);
  const blobId = String(file.blobId || file.oid || "");
  const lfsSha256 = String(file.lfs?.sha256 || (file.lfs?.oid && !String(file.lfs.oid).includes("*") ? file.lfs.oid : ""));
  return {
    path: filePath.replaceAll("\\", "/"),
    size_bytes: Number.isSafeInteger(size) && size >= 0 ? size : null,
    blob_id: /^[0-9a-f]{40,64}$/.test(blobId) ? blobId : null,
    lfs_sha256: /^[0-9a-f]{64}$/.test(lfsSha256) ? lfsSha256 : null,
    storage: file.lfs ? "lfs_or_xet" : "git",
    kind: descriptor.kind,
    format: descriptor.format,
    analyzer_support: descriptor.support,
  };
}

export function classifyFile(filePath) {
  const normalized = String(filePath || "").toLowerCase();
  const basename = path.posix.basename(normalized);
  if (SUPPORT_FILE_NAMES.has(basename) || /(?:tokenizer|vocab|merges)\.(?:json|txt|model)$/.test(basename)) {
    return { kind: "runtime_support", format: "metadata", support: "companion" };
  }
  if (/(?:^|\/)(?:model|weights?)\.onnx(?:_data|\.data)$/.test(normalized) || /\.onnx(?:_data|\.data)(?:-\d+-of-\d+)?$/.test(normalized)) {
    return { kind: "sidecar", format: "external_data", support: "onnx_companion" };
  }
  const compound = [...MODEL_EXTENSIONS.entries()].find(([extension]) => normalized.endsWith(extension));
  if (compound) return compound[1];
  const extension = path.posix.extname(normalized);
  if ([".md", ".txt", ".json", ".yaml", ".yml", ".jinja", ".py"].includes(extension)) return { kind: "metadata", format: "text", support: "metadata_only" };
  if ([".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"].includes(extension)) return { kind: "documentation_asset", format: "image", support: "metadata_only" };
  return { kind: "repository_file", format: extension.slice(1) || "unknown", support: "metadata_only" };
}

export function classifyRepository({ id, metadata = {}, files = [] }) {
  const searchable = [
    id,
    metadata.pipeline_tag,
    metadata.library_name,
    ...(metadata.tags || []),
    ...(metadata.architectures || []),
    metadata.model_type,
    ...files.map((file) => file.path),
  ].filter(Boolean).join(" ");
  const modelBytes = sum(files.filter((file) => ["model", "model_bundle", "weights", "sidecar", "binary"].includes(file.kind) && file.size_bytes !== null), (file) => file.size_bytes);
  const formats = new Set(files.map((file) => file.format));
  const largeReasons = [];
  if (LARGE_PIPELINES.has(metadata.pipeline_tag)) largeReasons.push(`generative_or_multimodal_pipeline:${metadata.pipeline_tag}`);
  if (/litert-lm|onnxruntime-genai/i.test(`${metadata.library_name || ""} ${(metadata.tags || []).join(" ")}`)) largeReasons.push("generative_runtime_family");
  if (LARGE_FAMILY.test(searchable)) largeReasons.push("llm_or_vlm_family_evidence");
  if (formats.has("litertlm") || formats.has("gguf")) largeReasons.push("large_model_container");
  if (files.some((file) => /-\d{5}-of-\d{5}|(?:^|[._-])shard/i.test(file.path))) largeReasons.push("sharded_model_payload");
  if (modelBytes >= 1024 ** 3) largeReasons.push("repository_model_payload_at_least_1_gib");
  if (largeReasons.length) {
    return {
      name: "large",
      confidence: largeReasons.some((reason) => reason !== "repository_model_payload_at_least_1_gib") ? "explicit" : "derived",
      reasons: largeReasons,
    };
  }

  const explicitMicro = MICRO_EXPLICIT.test(searchable);
  const tfliteFiles = files.filter((file) => file.format === "tflite");
  const boundedTflite = tfliteFiles.length > 0
    && tfliteFiles.every((file) => file.size_bytes !== null && file.size_bytes <= 16 * 1024 ** 2)
    && sum(tfliteFiles, (file) => file.size_bytes) <= 64 * 1024 ** 2;
  const taskAllowsMicroCandidate = !metadata.pipeline_tag || MICRO_PIPELINES.has(metadata.pipeline_tag);
  if (explicitMicro || (boundedTflite && taskAllowsMicroCandidate)) {
    return {
      name: "micro",
      confidence: explicitMicro ? "explicit" : "candidate",
      reasons: explicitMicro
        ? ["explicit_tflite_micro_or_mcu_metadata"]
        : ["tflite_payloads_at_most_16_mib_each", "microcontroller_fit_requires_arena_and_op_resolver_validation"],
    };
  }
  return {
    name: "mid",
    confidence: "derived",
    reasons: ["standalone_runtime_artifact_without_large_or_explicit_micro_evidence"],
  };
}

export function buildCorpusSummary(repositories) {
  const files = repositories.flatMap((repository) => repository.files);
  const knownFiles = files.filter((file) => file.size_bytes !== null);
  return {
    repository_count: repositories.length,
    organization_counts: countBy(repositories, (repository) => repository.organization),
    tier_counts: countBy(repositories, (repository) => repository.tier.name),
    tier_confidence_counts: countBy(repositories, (repository) => `${repository.tier.name}:${repository.tier.confidence}`),
    runtime_family_counts: countBy(repositories, (repository) => repository.runtime_family),
    file_count: files.length,
    known_size_file_count: knownFiles.length,
    unknown_size_file_count: files.length - knownFiles.length,
    known_file_bytes: sum(knownFiles, (file) => file.size_bytes),
    model_file_count: files.filter((file) => file.kind === "model" || file.kind === "model_bundle").length,
    analyzer_supported_file_count: files.filter((file) => file.analyzer_support === "current").length,
    analyzer_supported_bytes: sum(files.filter((file) => file.analyzer_support === "current" && file.size_bytes !== null), (file) => file.size_bytes),
    default_sweep_eligible_file_count: files.filter((file) => file.analyzer_support === "current" && file.size_bytes !== null && file.size_bytes <= 256 * 1024 ** 2).length,
    default_sweep_eligible_bytes: sum(files.filter((file) => file.analyzer_support === "current" && file.size_bytes !== null && file.size_bytes <= 256 * 1024 ** 2), (file) => file.size_bytes),
    format_counts: countBy(files.filter((file) => file.kind === "model" || file.kind === "model_bundle"), (file) => file.format),
  };
}

export function compactCorpusSummary(manifest) {
  return {
    schema: "deepbom.huggingface_community_corpus_summary.v1",
    corpus_schema: manifest.schema,
    classifier_schema: manifest.classifier_schema,
    generated_at: manifest.generated_at,
    source: manifest.source,
    summary: manifest.summary,
    tiers: {
      micro: "Explicit TFLite Micro/TinyML/MCU evidence, or a <=16 MiB-per-TFLite MCU candidate. Candidate status is not a memory-fit proof.",
      mid: "Standalone TFLite/ONNX or other per-model runtime artifact without large-model evidence.",
      large: "LLM/VLM/generative runtime, large-model container, shard evidence, or >=1 GiB repository model payload.",
    },
    current_analyzer_formats: [...CURRENT_ANALYZER_FORMATS],
    download_policy: "Metadata snapshot is tracked; model bytes remain in the user-local cache and are never deployed.",
  };
}

export function hfCorpusCacheDir() {
  const base = process.env.LOCALAPPDATA || process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "DeepBOM", "huggingface-community-corpus-v1");
}

export function hfCorpusMetadataCacheDir() {
  return path.join(hfCorpusCacheDir(), "metadata-api-v1");
}

export async function resolveHfToken(explicit = "") {
  if (explicit) return String(explicit).trim();
  if (process.env.HF_TOKEN) return String(process.env.HF_TOKEN).trim();
  const candidates = [
    path.join(os.homedir(), ".cache", "huggingface", "token"),
    path.join(os.homedir(), ".huggingface", "token"),
  ];
  for (const candidate of candidates) {
    try {
      const token = (await readFile(candidate, "utf8")).trim();
      if (token) return token;
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  return "";
}

export function hfRepositoryCacheDir(cacheDir, repository) {
  return path.join(cacheDir, ...repository.id.split("/"), repository.revision);
}

export function hfFileCachePath(cacheDir, repository, file) {
  if (!isSafeRelativePath(file.path)) throw new Error(`Unsafe Hugging Face file path: ${file.path}.`);
  return path.join(hfRepositoryCacheDir(cacheDir, repository), ...file.path.split("/"));
}

export function hfFileUrl(repository, file) {
  const encodedPath = file.path.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${repository.id}/resolve/${repository.revision}/${encodedPath}?download=true`;
}

export function selectRepositories(manifest, options = {}) {
  return manifest.repositories.filter((repository) => {
    if (options.organizations?.length && !options.organizations.includes(repository.organization)) return false;
    if (options.tiers?.length && !options.tiers.includes(repository.tier.name)) return false;
    if (options.repositoryIds?.length && !options.repositoryIds.includes(repository.id)) return false;
    return true;
  });
}

export function selectRepositoryFiles(repository, scope = "testable", formats = []) {
  const selectedFormats = new Set(formats);
  const primary = repository.files.filter((file) => {
    if (selectedFormats.size && !selectedFormats.has(file.format)) return false;
    if (scope === "repository") return true;
    if (scope === "model") return ["model", "model_bundle", "weights", "sidecar", "runtime_support"].includes(file.kind);
    return file.analyzer_support === "current";
  });
  if (scope !== "testable" || !primary.some((file) => file.format === "onnx")) return primary;
  const companion = repository.files.filter((file) => file.analyzer_support === "onnx_companion");
  return uniqueFiles([...primary, ...companion]);
}

export async function ensureHfFile(repository, file, cacheDir, {
  offline = false,
  token = "",
  onProgress = null,
} = {}) {
  if (file.size_bytes === null) throw new Error(`${repository.id}/${file.path}: upstream size is unknown.`);
  const filename = hfFileCachePath(cacheDir, repository, file);
  const existing = await verifyHfFile(filename, file);
  if (existing.valid) return { filename, downloaded: false, identity: existing.identity };
  if (offline) throw new Error(`${repository.id}/${file.path}: verified cache entry is unavailable in offline mode.`);
  await mkdir(path.dirname(filename), { recursive: true });
  const partial = `${filename}.partial`;
  let offset = 0;
  try {
    const partialInfo = await stat(partial);
    if (partialInfo.isFile() && partialInfo.size < file.size_bytes) offset = partialInfo.size;
    else await rm(partial, { force: true });
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const headers = { Accept: "application/octet-stream" };
  const resolvedToken = await resolveHfToken(token);
  if (resolvedToken) headers.Authorization = `Bearer ${resolvedToken}`;
  if (offset) headers.Range = `bytes=${offset}-`;
  const response = await fetch(hfFileUrl(repository, file), { headers, redirect: "follow" });
  if (!(response.status === 200 || (offset && response.status === 206)) || !response.body) {
    throw new Error(`${repository.id}/${file.path}: download failed with HTTP ${response.status}.`);
  }
  if (offset && response.status === 200) {
    offset = 0;
    await rm(partial, { force: true });
  }
  const stream = createWriteStream(partial, { flags: offset ? "a" : "wx" });
  let received = offset;
  const tracking = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      onProgress?.({ repository: repository.id, path: file.path, received, total: file.size_bytes });
      controller.enqueue(chunk);
    },
  });
  await pipeline(Readable.fromWeb(response.body.pipeThrough(tracking)), stream);
  const verified = await verifyHfFile(partial, file);
  if (!verified.valid) {
    await rm(partial, { force: true });
    throw new Error(`${repository.id}/${file.path}: ${verified.reason}.`);
  }
  await rm(filename, { force: true });
  await rename(partial, filename);
  return { filename, downloaded: true, identity: verified.identity };
}

export async function verifyHfFile(filename, file) {
  try {
    const info = await stat(filename);
    if (!info.isFile()) return { valid: false, reason: "cache path is not a file", identity: null };
    if (info.size !== file.size_bytes) return { valid: false, reason: `size ${info.size} does not match ${file.size_bytes}`, identity: null };
    const sha256 = createHash("sha256");
    const gitBlobSha1 = createHash("sha1");
    gitBlobSha1.update(`blob ${info.size}\0`);
    for await (const chunk of createReadStream(filename)) {
      sha256.update(chunk);
      gitBlobSha1.update(chunk);
    }
    const identity = { size_bytes: info.size, sha256: sha256.digest("hex"), git_blob_sha1: gitBlobSha1.digest("hex") };
    if (file.lfs_sha256 && identity.sha256 !== file.lfs_sha256) return { valid: false, reason: "LFS SHA-256 mismatch", identity };
    if (!file.lfs_sha256 && file.blob_id?.length === 40 && identity.git_blob_sha1 !== file.blob_id) return { valid: false, reason: "Git blob SHA-1 mismatch", identity };
    return { valid: true, reason: "", identity };
  } catch (error) {
    if (error?.code === "ENOENT") return { valid: false, reason: "cache file is absent", identity: null };
    throw error;
  }
}

export async function writeDownloadReceipt(cacheDir, receipt) {
  if (receipt.schema !== HF_RECEIPT_SCHEMA) throw new Error("Download receipt schema is invalid.");
  await mkdir(cacheDir, { recursive: true });
  const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
  const receiptDir = path.join(cacheDir, "receipts");
  await mkdir(receiptDir, { recursive: true });
  const timestamp = String(receipt.completed_at || new Date().toISOString()).replace(/[:.]/g, "-");
  const historyPath = path.join(receiptDir, `download-receipt-${timestamp}.json`);
  await writeFile(historyPath, serialized, "utf8");
  await writeFile(path.join(cacheDir, "download-receipt.json"), serialized, "utf8");
  return { latest: path.join(cacheDir, "download-receipt.json"), history: historyPath };
}

export async function fetchJsonWithRetry(url, {
  token = "",
  attempts = 8,
  baseDelayMs = 750,
} = {}) {
  let lastError = null;
  const resolvedToken = await resolveHfToken(token);
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const headers = { Accept: "application/json", "User-Agent": "DeepBOM-HF-Corpus/1" };
      if (resolvedToken) headers.Authorization = `Bearer ${resolvedToken}`;
      const response = await fetch(url, { headers });
      if (response.ok) return { data: await response.json(), response };
      if (![429, 500, 502, 503, 504].includes(response.status)) {
        throw new Error(`HTTP ${response.status} for ${url}`);
      }
      lastError = new Error(`HTTP ${response.status} for ${url}`);
      const retryAfter = Number(response.headers.get("retry-after") || 0) * 1000;
      await delay(Math.max(retryAfter, baseDelayMs * 2 ** (attempt - 1)));
    } catch (error) {
      lastError = error;
      if (attempt === attempts) break;
      await delay(baseDelayMs * 2 ** (attempt - 1));
    }
  }
  throw lastError || new Error(`Failed to fetch ${url}.`);
}

export async function readHfMetadataCheckpoint(cacheDir, id, revision) {
  const filename = hfMetadataCheckpointPath(cacheDir, id, revision);
  try {
    const value = JSON.parse(await readFile(filename, "utf8"));
    return value?.id === id && value?.sha === revision ? value : null;
  } catch (error) {
    if (error?.code === "ENOENT" || error instanceof SyntaxError) return null;
    throw error;
  }
}

export async function writeHfMetadataCheckpoint(cacheDir, detail) {
  const filename = hfMetadataCheckpointPath(cacheDir, detail.id, detail.sha);
  await mkdir(path.dirname(filename), { recursive: true });
  await writeFile(filename, `${JSON.stringify(detail)}\n`, "utf8");
  return filename;
}

export function nextLink(response) {
  const header = response.headers.get("link") || "";
  const match = header.match(/<([^>]+)>;\s*rel="next"/);
  return match?.[1] || "";
}

export function sha256Text(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function isSafeRelativePath(value) {
  const normalized = String(value || "").replaceAll("\\", "/");
  return Boolean(normalized)
    && !normalized.includes("\0")
    && !normalized.startsWith("/")
    && !/^[a-z][a-z0-9+.-]*:/i.test(normalized)
    && normalized.split("/").every((part) => part && part !== "." && part !== "..");
}

function normalizeUpstreamMetadata(detail) {
  const cardData = detail.cardData && typeof detail.cardData === "object" ? detail.cardData : {};
  const config = detail.config && typeof detail.config === "object" ? detail.config : {};
  return {
    downloads: Number.isSafeInteger(detail.downloads) ? detail.downloads : null,
    likes: Number.isSafeInteger(detail.likes) ? detail.likes : null,
    trending_score: Number.isFinite(detail.trendingScore) ? detail.trendingScore : null,
    pipeline_tag: detail.pipeline_tag || cardData.pipeline_tag || null,
    library_name: detail.library_name || cardData.library_name || null,
    tags: uniqueStrings(detail.tags || []),
    license: cardData.license || tagValue(detail.tags, "license") || null,
    languages: uniqueStrings(asArray(cardData.language)),
    datasets: uniqueStrings(asArray(cardData.datasets)),
    metrics: uniqueStrings(asArray(cardData.metrics)),
    base_models: uniqueStrings([
      ...asArray(cardData.base_model),
      ...(detail.tags || []).filter((tag) => String(tag).startsWith("base_model:")).map((tag) => String(tag).replace(/^base_model:(?:finetune:|adapter:|quantized:)?/, "")),
    ]),
    architectures: uniqueStrings(asArray(config.architectures)),
    model_type: config.model_type || null,
    quantization_config: config.quantization_config || null,
    safetensors: detail.safetensors || null,
    used_storage_bytes: Number.isSafeInteger(detail.usedStorage) ? detail.usedStorage : null,
  };
}

function inferRuntimeFamily(detail, files) {
  const formats = new Set(files.map((file) => file.format));
  const evidence = `${detail.library_name || ""} ${(detail.tags || []).join(" ")}`;
  if (formats.has("litertlm") || /litert-lm/i.test(evidence)) return "litert_lm";
  if (/onnxruntime-genai/i.test(evidence)) return "onnxruntime_genai";
  if (formats.has("tflite") && formats.has("onnx")) return "mixed_tflite_onnx";
  if (formats.has("tflite")) return "litert";
  if (formats.has("onnx")) return "onnxruntime";
  if (formats.has("mediapipe_task")) return "mediapipe_tasks";
  return "future_or_metadata_only";
}

function tagValue(tags, prefix) {
  const match = (tags || []).find((tag) => String(tag).startsWith(`${prefix}:`));
  return match ? String(match).slice(prefix.length + 1) : "";
}

function asArray(value) {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}

function uniqueStrings(values) {
  return [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))].sort();
}

function uniqueFiles(files) {
  return [...new Map(files.map((file) => [file.path, file])).values()].sort((left, right) => left.path.localeCompare(right.path));
}

function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = selector(row) || "unknown";
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

function sum(rows, selector) {
  return rows.reduce((total, row) => total + Number(selector(row) || 0), 0);
}

function sortObjectKeys(value) {
  if (Array.isArray(value)) return value.map(sortObjectKeys);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortObjectKeys(value[key])]));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hfMetadataCheckpointPath(cacheDir, id, revision) {
  const digest = createHash("sha256").update(`${id}@${revision}`).digest("hex");
  return path.join(cacheDir, `${digest}.json`);
}
