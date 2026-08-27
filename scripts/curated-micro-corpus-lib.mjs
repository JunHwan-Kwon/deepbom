import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const CURATED_MICRO_CORPUS_PATH = "corpus/curated-micro-corpus.v1.json";
export const CURATED_MICRO_CORPUS_SCHEMA = "deepbom.curated_micro_corpus.v1";
export const CURATED_MICRO_SWEEP_SCHEMA = "deepbom.curated_micro_corpus_sweep.v1";

export async function readCuratedMicroCorpus(filename = CURATED_MICRO_CORPUS_PATH) {
  const manifest = JSON.parse(await readFile(filename, "utf8"));
  return validateCuratedMicroCorpus(manifest);
}

export function validateCuratedMicroCorpus(manifest) {
  if (manifest?.schema !== CURATED_MICRO_CORPUS_SCHEMA) {
    throw new Error(`Unsupported curated micro corpus schema: ${manifest?.schema || "(missing)"}.`);
  }
  if (manifest.tier !== "micro" || manifest.format !== "tflite") {
    throw new Error("The v1 curated micro corpus must contain explicit-micro TFLite artifacts.");
  }
  if (!Array.isArray(manifest.sources) || !Array.isArray(manifest.artifacts)
    || manifest.artifact_count !== manifest.artifacts.length) {
    throw new Error("Curated micro source or artifact inventory is incomplete.");
  }
  const sourceIds = new Set();
  for (const source of manifest.sources) {
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(source.id || ""))
      || sourceIds.has(source.id)
      || !/^[0-9a-f]{40}$/.test(String(source.revision || ""))) {
      throw new Error(`Invalid or duplicate curated source: ${source.id || "(missing)"}.`);
    }
    sourceIds.add(source.id);
  }
  const ids = new Set();
  const urls = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) {
      throw new Error(`Invalid or duplicate curated artifact id: ${artifact.id || "(missing)"}.`);
    }
    if (!sourceIds.has(artifact.source_id)) throw new Error(`${artifact.id}: source_id is unresolved.`);
    if (!/^[A-Za-z0-9_.-]+\.tflite$/.test(String(artifact.filename || ""))) {
      throw new Error(`${artifact.id}: filename is invalid.`);
    }
    if (!/^https:\/\//.test(String(artifact.url || "")) || urls.has(artifact.url)) {
      throw new Error(`${artifact.id}: URL is invalid or duplicated.`);
    }
    if (!Number.isSafeInteger(artifact.size_bytes) || artifact.size_bytes < 1) {
      throw new Error(`${artifact.id}: byte size is invalid.`);
    }
    if (!/^[0-9a-f]{64}$/.test(String(artifact.sha256 || ""))) {
      throw new Error(`${artifact.id}: SHA-256 is invalid.`);
    }
    ids.add(artifact.id);
    urls.add(artifact.url);
  }
  return manifest;
}

export function curatedMicroCacheDir() {
  const base = process.env.LOCALAPPDATA || process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "DeepBOM", "curated-micro-corpus-v1");
}

export function curatedMicroArtifactPath(cacheDir, manifest, artifact) {
  const source = manifest.sources.find((candidate) => candidate.id === artifact.source_id);
  if (!source) throw new Error(`${artifact.id}: source_id is unresolved.`);
  return path.join(cacheDir, ...source.id.split("/"), source.revision, artifact.filename);
}

export async function ensureCuratedMicroArtifact(manifest, artifact, cacheDir, {
  offline = false,
  onProgress = null,
} = {}) {
  const filename = curatedMicroArtifactPath(cacheDir, manifest, artifact);
  const current = await fileIdentity(filename);
  if (current?.size === artifact.size_bytes && current.sha256 === artifact.sha256) {
    return { filename, downloaded: false };
  }
  if (offline) throw new Error(`${artifact.id}: verified cache entry is unavailable in offline mode.`);
  await mkdir(path.dirname(filename), { recursive: true });
  const partial = `${filename}.partial`;
  let offset = 0;
  try {
    const info = await stat(partial);
    if (info.isFile() && info.size < artifact.size_bytes) offset = info.size;
    else await rm(partial, { force: true });
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const headers = offset ? { Range: `bytes=${offset}-` } : {};
  const response = await fetch(artifact.url, { headers, redirect: "follow" });
  if (!(response.status === 200 || (offset && response.status === 206)) || !response.body) {
    throw new Error(`${artifact.id}: download failed with HTTP ${response.status}.`);
  }
  if (offset && response.status === 200) {
    offset = 0;
    await rm(partial, { force: true });
  }
  let received = offset;
  const tracking = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      onProgress?.({ id: artifact.id, received, total: artifact.size_bytes });
      controller.enqueue(chunk);
    },
  });
  await pipeline(
    Readable.fromWeb(response.body.pipeThrough(tracking)),
    createWriteStream(partial, { flags: offset ? "a" : "wx" }),
  );
  const downloaded = await fileIdentity(partial);
  if (downloaded?.size !== artifact.size_bytes || downloaded.sha256 !== artifact.sha256) {
    await rm(partial, { force: true });
    throw new Error(`${artifact.id}: downloaded bytes do not match the pinned artifact identity.`);
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
