import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const CORPUS_MANIFEST_PATH = "corpus/public-tflite-corpus.v1.json";
export const CORPUS_SCHEMA = "deepbom.public_tflite_corpus.v1";
export const SWEEP_SCHEMA = "deepbom.public_model_corpus_sweep.v1";
export const RESULT_SCHEMA = "deepbom.public_model_corpus_result.v1";

export async function readCorpusManifest(filename = CORPUS_MANIFEST_PATH) {
  const manifest = JSON.parse(await readFile(filename, "utf8"));
  validateCorpusManifest(manifest);
  return manifest;
}

export function validateCorpusManifest(manifest) {
  if (manifest?.schema !== CORPUS_SCHEMA) throw new Error(`Unsupported corpus schema: ${manifest?.schema || "(missing)"}`);
  if (manifest.format !== "tflite") throw new Error("The v1 public corpus must contain TFLite artifacts.");
  if (!Array.isArray(manifest.models) || manifest.models.length !== 20 || manifest.artifact_count !== 20) {
    throw new Error("The public corpus must contain exactly 20 artifacts.");
  }
  const ids = new Set();
  const objects = new Set();
  for (const model of manifest.models) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(model.id || "")) || ids.has(model.id)) throw new Error(`Invalid or duplicate model id: ${model.id}`);
    if (!/^[a-z_]+$/.test(String(model.task || ""))) throw new Error(`Invalid task for ${model.id}.`);
    if (!/^[a-z0-9_]+$/.test(String(model.published_precision || ""))) throw new Error(`Invalid precision for ${model.id}.`);
    if (!/^[^?]+\/1\/[^/]+\.tflite$/.test(String(model.object || "")) || objects.has(model.object)) throw new Error(`Invalid or duplicate versioned object for ${model.id}.`);
    if (!/^\d{16}$/.test(String(model.generation || ""))) throw new Error(`Invalid GCS generation for ${model.id}.`);
    if (!Number.isSafeInteger(model.size_bytes) || model.size_bytes < 1) throw new Error(`Invalid size for ${model.id}.`);
    if (!/^[0-9a-f]{64}$/.test(String(model.sha256 || ""))) throw new Error(`Invalid SHA-256 for ${model.id}.`);
    ids.add(model.id);
    objects.add(model.object);
  }
  return manifest;
}

export function corpusCacheDir() {
  const base = process.env.LOCALAPPDATA || process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "DeepBOM", "public-model-corpus-v1");
}

export function corpusModelPath(cacheDir, model) {
  return path.join(cacheDir, model.object.replaceAll("/", "__"));
}

export function corpusModelUrl(model) {
  const objectPath = model.object.split("/").map(encodeURIComponent).join("/");
  return `https://storage.googleapis.com/mediapipe-models/${objectPath}?generation=${model.generation}`;
}

export async function ensureCorpusModel(model, cacheDir, { offline = false } = {}) {
  await mkdir(cacheDir, { recursive: true });
  const filename = corpusModelPath(cacheDir, model);
  const current = await fileIdentity(filename);
  if (current?.size === model.size_bytes && current.sha256 === model.sha256) return { filename, downloaded: false };
  if (offline) throw new Error(`${model.id}: verified cache entry is unavailable in offline mode.`);
  const partial = `${filename}.partial`;
  await rm(partial, { force: true });
  const response = await fetch(corpusModelUrl(model), { redirect: "error" });
  if (!response.ok || !response.body) throw new Error(`${model.id}: download failed with HTTP ${response.status}.`);
  const observedGeneration = response.headers.get("x-goog-generation");
  if (observedGeneration !== model.generation) throw new Error(`${model.id}: object generation ${observedGeneration || "(missing)"} does not match manifest.`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(partial, { flags: "wx" }));
  const downloaded = await fileIdentity(partial);
  if (downloaded?.size !== model.size_bytes || downloaded.sha256 !== model.sha256) {
    await rm(partial, { force: true });
    throw new Error(`${model.id}: downloaded artifact identity does not match the manifest.`);
  }
  await rm(filename, { force: true });
  await rename(partial, filename);
  return { filename, downloaded: true };
}

export async function fileIdentity(filename) {
  try {
    const info = await stat(filename);
    if (!info.isFile()) return null;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(filename)) hash.update(chunk);
    return { size: info.size, sha256: hash.digest("hex") };
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

export function sha256Text(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function findNonFinite(value, pathPrefix = "$", output = [], limit = 20) {
  if (output.length >= limit) return output;
  if (typeof value === "number" && !Number.isFinite(value)) {
    output.push(pathPrefix);
  } else if (Array.isArray(value)) {
    for (let index = 0; index < value.length && output.length < limit; index += 1) {
      findNonFinite(value[index], `${pathPrefix}[${index}]`, output, limit);
    }
  } else if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      findNonFinite(child, `${pathPrefix}.${key}`, output, limit);
      if (output.length >= limit) break;
    }
  }
  return output;
}

export function validateCorpusResult(result) {
  if (result?.schema !== RESULT_SCHEMA) throw new Error("Corpus worker result schema is invalid.");
  if (!/^[0-9a-f]{64}$/.test(String(result.artifact_sha256 || ""))) throw new Error(`${result?.id}: artifact digest is invalid.`);
  if (!/^[0-9a-f]{64}$/.test(String(result.analysis_sha256 || ""))) throw new Error(`${result?.id}: analysis digest is invalid.`);
  const coverage = result.quant_research_coverage;
  if (coverage.lab_count !== 15
    || coverage.class_supported + coverage.class_excluded !== 15
    || coverage.assessed + coverage.partial + coverage.not_assessed + coverage.not_applicable !== 15) {
    throw new Error(`${result.id}: quant research denominator is not conserved.`);
  }
  if (result.quality.non_finite_paths.length) throw new Error(`${result.id}: non-finite values at ${result.quality.non_finite_paths.join(", ")}.`);
  if (result.quality.phantom_reference_count) throw new Error(`${result.id}: phantom op references were emitted.`);
  if (!result.quality.op_index_unique) throw new Error(`${result.id}: operator indices are not unique.`);
  return result;
}
