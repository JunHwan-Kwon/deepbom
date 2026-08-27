import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export function deepBomCorpusCacheDir(namespace) {
  if (!/^[a-z0-9][a-z0-9-]+$/.test(String(namespace || ""))) throw new Error("Corpus cache namespace is invalid.");
  const base = process.env.LOCALAPPDATA || process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "DeepBOM", namespace);
}

export function hashBoundSourcePath(cacheDir, artifact, file) {
  return path.join(cacheDir, ...artifact.repository.split("/"), artifact.revision, file.path);
}

export function huggingFaceRevisionUrl(artifact, file) {
  const encodedPath = file.path.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${artifact.repository}/resolve/${artifact.revision}/${encodedPath}?download=true`;
}

export async function ensureHashBoundSourceFile(cacheDir, artifact, file, { offline = false } = {}) {
  const filename = hashBoundSourcePath(cacheDir, artifact, file);
  const observed = await fileIdentity(filename);
  if (observed?.size === file.size_bytes && observed.sha256 === file.sha256) return { filename, downloaded: false };
  if (offline) throw new Error(`${artifact.id}/${file.path}: verified cache entry is unavailable in offline mode.`);
  await mkdir(path.dirname(filename), { recursive: true });
  const partial = `${filename}.partial`;
  await rm(partial, { force: true });
  const response = await fetch(huggingFaceRevisionUrl(artifact, file), { redirect: "follow" });
  if (!response.ok || !response.body) throw new Error(`${artifact.id}/${file.path}: download failed with HTTP ${response.status}.`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(partial, { flags: "wx" }));
  const downloaded = await fileIdentity(partial);
  if (downloaded?.size !== file.size_bytes || downloaded.sha256 !== file.sha256) {
    await rm(partial, { force: true });
    throw new Error(`${artifact.id}/${file.path}: downloaded bytes do not match the pinned identity.`);
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
