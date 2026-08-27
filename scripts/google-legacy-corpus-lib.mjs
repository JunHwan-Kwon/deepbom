import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const GOOGLE_LEGACY_MANIFEST_PATH = "corpus/google_legacy/manifest.v1.json";
export const GOOGLE_LEGACY_COHORTS_PATH = "corpus/google_legacy/converter-cohorts.v1.json";
export const GOOGLE_LEGACY_MANIFEST_SCHEMA = "deepbom.google_legacy_corpus.v1";
export const GOOGLE_LEGACY_SWEEP_SCHEMA = "deepbom.google_legacy_corpus_sweep.v1";

export async function readGoogleLegacyManifest(filename = GOOGLE_LEGACY_MANIFEST_PATH) {
  return validateGoogleLegacyManifest(JSON.parse(await readFile(filename, "utf8")));
}

export function validateGoogleLegacyManifest(manifest) {
  if (manifest?.schema !== GOOGLE_LEGACY_MANIFEST_SCHEMA) {
    throw new Error(`Unsupported Google legacy corpus schema: ${manifest?.schema || "(missing)"}.`);
  }
  if (!Array.isArray(manifest.artifacts) || manifest.artifact_count !== manifest.artifacts.length) {
    throw new Error("Google legacy artifact inventory is incomplete.");
  }
  if (!Array.isArray(manifest.source_pages) || manifest.source_pages.length < 1) {
    throw new Error("Google legacy source-page provenance is missing.");
  }
  const ids = new Set();
  const urls = new Set();
  const archiveNames = new Set();
  let quantized = 0;
  let controls = 0;
  for (const artifact of manifest.artifacts) {
    const id = String(artifact?.id || "");
    if (!/^[a-z0-9][a-z0-9.-]+$/.test(id) || ids.has(id)) {
      throw new Error(`Invalid or duplicate Google legacy artifact id: ${id || "(missing)"}.`);
    }
    if (!/^https:\/\/storage\.googleapis\.com\/download\.tensorflow\.org\/models\//.test(String(artifact.source_url || ""))
      || urls.has(artifact.source_url)) {
      throw new Error(`${id}: source URL is invalid or duplicated.`);
    }
    validateIdentity(artifact.archive, `${id}: archive`);
    validateIdentity(artifact.member, `${id}: member`);
    if (!/^[A-Za-z0-9_.-]+\.(?:tgz|zip)$/.test(String(artifact.archive.filename || ""))
      || archiveNames.has(artifact.archive.filename)) {
      throw new Error(`${id}: archive filename is invalid or duplicated.`);
    }
    const safeMember = normalizeMemberPath(artifact.member.path);
    if (path.posix.basename(safeMember) !== artifact.member.filename
      || !/^[A-Za-z0-9_.-]+\.tflite$/.test(String(artifact.member.filename || ""))) {
      throw new Error(`${id}: archive member filename is invalid.`);
    }
    const baseline = artifact.baseline;
    if (!baseline || !Number.isSafeInteger(baseline.operator_count)
      || !Number.isSafeInteger(baseline.tensor_count)
      || !Number.isSafeInteger(baseline.total_macs)
      || baseline.lab_count !== 15
      || baseline.class_supported_lab_count + (15 - baseline.class_supported_lab_count) !== 15
      || baseline.assessed_lab_count + baseline.partial_lab_count
        + baseline.not_assessed_lab_count + baseline.not_applicable_lab_count !== 15) {
      throw new Error(`${id}: measured baseline is incomplete.`);
    }
    if (artifact.cohort === "legacy_quantized") quantized += 1;
    else if (artifact.cohort === "historical_float_control") controls += 1;
    else throw new Error(`${id}: unsupported cohort ${artifact.cohort || "(missing)"}.`);
    ids.add(id);
    urls.add(artifact.source_url);
    archiveNames.add(artifact.archive.filename);
  }
  if (quantized !== manifest.quantized_artifact_count || controls !== manifest.float_control_count
    || quantized + controls !== manifest.artifact_count) {
    throw new Error("Google legacy cohort totals do not reconcile.");
  }
  return manifest;
}

function validateIdentity(identity, label) {
  if (!identity || !Number.isSafeInteger(identity.size_bytes) || identity.size_bytes < 1) {
    throw new Error(`${label} size is invalid.`);
  }
  if (!/^[0-9a-f]{64}$/.test(String(identity.sha256 || ""))) {
    throw new Error(`${label} SHA-256 is invalid.`);
  }
}

export function normalizeMemberPath(value) {
  const raw = String(value || "").replace(/^\.\//, "");
  if (!raw || raw.includes("\\") || path.posix.isAbsolute(raw)) {
    throw new Error(`Unsafe archive member path: ${value || "(missing)"}.`);
  }
  const normalized = path.posix.normalize(raw);
  if (normalized === ".." || normalized.startsWith("../") || normalized !== raw) {
    throw new Error(`Unsafe archive member path: ${value}.`);
  }
  return normalized;
}

export function googleLegacyCacheDir() {
  const base = process.env.LOCALAPPDATA || process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "DeepBOM", "google-legacy-corpus-v1");
}

export function googleLegacyArchivePath(cacheDir, artifact) {
  return path.join(cacheDir, "archives", artifact.archive.filename);
}

export function googleLegacyModelPath(cacheDir, artifact) {
  return path.join(cacheDir, "models", artifact.id, artifact.member.filename);
}

export async function ensureGoogleLegacyArtifact(artifact, cacheDir, {
  offline = false,
  onProgress = null,
} = {}) {
  const archivePath = googleLegacyArchivePath(cacheDir, artifact);
  const archive = await ensureDownload(artifact, archivePath, { offline, onProgress });
  const modelPath = googleLegacyModelPath(cacheDir, artifact);
  const current = await fileIdentity(modelPath);
  if (identityMatches(current, artifact.member)) {
    return { archivePath, modelPath, downloaded: archive.downloaded, extracted: false };
  }
  await mkdir(path.dirname(modelPath), { recursive: true });
  const partial = `${modelPath}.partial`;
  await rm(partial, { force: true });
  try {
    await extractExactMember(archivePath, artifact.member.path, partial);
    const extracted = await fileIdentity(partial);
    if (!identityMatches(extracted, artifact.member)) {
      throw new Error(`${artifact.id}: extracted model does not match its pinned identity.`);
    }
    await rm(modelPath, { force: true });
    await rename(partial, modelPath);
  } catch (error) {
    await rm(partial, { force: true });
    throw error;
  }
  return { archivePath, modelPath, downloaded: archive.downloaded, extracted: true };
}

async function ensureDownload(artifact, filename, { offline, onProgress }) {
  const current = await fileIdentity(filename);
  if (identityMatches(current, artifact.archive)) return { downloaded: false };
  if (offline) throw new Error(`${artifact.id}: verified archive is unavailable in offline mode.`);
  await mkdir(path.dirname(filename), { recursive: true });
  const partial = `${filename}.partial`;
  let offset = 0;
  try {
    const info = await stat(partial);
    if (info.isFile() && info.size < artifact.archive.size_bytes) offset = info.size;
    else await rm(partial, { force: true });
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const response = await fetch(artifact.source_url, {
    headers: offset ? { Range: `bytes=${offset}-` } : {},
    redirect: "follow",
  });
  if (!(response.status === 200 || (offset > 0 && response.status === 206)) || !response.body) {
    throw new Error(`${artifact.id}: archive download failed with HTTP ${response.status}.`);
  }
  if (offset > 0 && response.status === 200) {
    offset = 0;
    await rm(partial, { force: true });
  }
  let received = offset;
  const tracking = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      onProgress?.({ id: artifact.id, received, total: artifact.archive.size_bytes });
      controller.enqueue(chunk);
    },
  });
  await pipeline(
    Readable.fromWeb(response.body.pipeThrough(tracking)),
    createWriteStream(partial, { flags: offset ? "a" : "wx" }),
  );
  const downloaded = await fileIdentity(partial);
  if (!identityMatches(downloaded, artifact.archive)) {
    await rm(partial, { force: true });
    throw new Error(`${artifact.id}: downloaded archive does not match its pinned identity.`);
  }
  await rm(filename, { force: true });
  await rename(partial, filename);
  return { downloaded: true };
}

async function extractExactMember(archivePath, memberPath, outputPath) {
  normalizeMemberPath(memberPath);
  const child = spawn("tar", ["-xOf", archivePath, memberPath], {
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
    if (stderr.length > 8192) stderr = stderr.slice(-8192);
  });
  const exit = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`tar extraction failed with exit ${code}: ${stderr.trim()}`));
    });
  });
  await Promise.all([
    pipeline(child.stdout, createWriteStream(outputPath, { flags: "wx" })),
    exit,
  ]);
}

function identityMatches(actual, expected) {
  return actual?.size === expected.size_bytes && actual.sha256 === expected.sha256;
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

export function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function sha256Text(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = String(selector(row) || "unknown");
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

export function csvText(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => quoteCsv(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function quoteCsv(value) {
  const text = value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}
