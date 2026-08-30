import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

const SHA256 = /^[a-f0-9]{64}$/;
const HF_COMMIT = /^[a-f0-9]{40}$/;
const DEFAULT_MAX_BYTES = 50n * 1024n * 1024n * 1024n;
const DOWNLOAD_TIMEOUT_MS = 30 * 60 * 1000;
const HEADER_TIMEOUT_MS = 60 * 1000;
const IDLE_TIMEOUT_MS = 60 * 1000;

export async function resolveArtifactSource(spec, {
  cacheDir = defaultArtifactCacheDir(),
  expectedSha256 = "",
  offline = false,
  maximumBytes = DEFAULT_MAX_BYTES,
  fetchImpl = fetch,
  progress = defaultProgress,
  environment = process.env,
  headerTimeoutMs = HEADER_TIMEOUT_MS,
  idleTimeoutMs = IDLE_TIMEOUT_MS,
  downloadTimeoutMs = DOWNLOAD_TIMEOUT_MS,
} = {}) {
  const source = parseArtifactSource(spec, expectedSha256);
  if (source.kind === "local") return { path: source.path, acquisition: null };
  const sourceKey = sha256Text(source.cache_identity);
  const receiptPath = path.join(cacheDir, "sources", `${sourceKey}.json`);
  const cached = await readCachedReceipt(cacheDir, receiptPath, source);
  if (cached) return cached;
  if (offline) throw new Error(`Verified remote artifact is unavailable in offline cache: ${source.canonical_locator}`);
  await mkdir(path.join(cacheDir, "incoming"), { recursive: true });
  const temporary = path.join(cacheDir, "incoming", `${sourceKey}.${process.pid}.${randomUUID()}.partial`);
  try {
    const request = await buildRequest(source, fetchImpl, environment);
    const controller = new AbortController();
    const response = await fetchHeaders(fetchImpl, request.url, {
      redirect: "follow",
      headers: request.headers,
      signal: controller.signal,
    }, controller, headerTimeoutMs);
    if (!response.ok || !response.body) throw new Error(`Remote artifact download failed with HTTP ${response.status}.`);
    const declared = parseContentLength(response.headers.get("content-length"));
    if (declared != null && declared > maximumBytes) throw new Error(`Remote artifact exceeds the ${maximumBytes}-byte download limit.`);
    const observed = await streamResponse(response.body, temporary, {
      maximumBytes, declared, progress, label: source.display_name, controller, idleTimeoutMs, downloadTimeoutMs,
    });
    if (source.expected_sha256 && observed.sha256 !== source.expected_sha256) {
      throw new Error(`Remote artifact SHA-256 mismatch: expected ${source.expected_sha256}, observed ${observed.sha256}.`);
    }
    if (request.generation && response.headers.get("x-goog-generation")
      && response.headers.get("x-goog-generation") !== request.generation) throw new Error("GCS response generation differs from the pinned generation.");
    const relative = path.join("sha256", observed.sha256.slice(0, 2), observed.sha256, source.display_name);
    const destination = path.join(cacheDir, relative);
    await mkdir(path.dirname(destination), { recursive: true });
    await installContentAddressed(temporary, destination, observed);
    const acquisition = acquisitionRecord(source, observed, relative.replaceAll("\\", "/"), response.url || request.url, request.generation);
    await writeReceipt(receiptPath, acquisition);
    return { path: destination, acquisition };
  } finally {
    await rm(temporary, { force: true });
  }
}

export function parseArtifactSource(spec, expectedSha256 = "") {
  const value = String(spec || "").trim();
  const expected = String(expectedSha256 || "").toLowerCase();
  if (expected && !SHA256.test(expected)) throw new Error("--expected-sha256 must be 64 lowercase hexadecimal characters.");
  if (value.startsWith("hf://")) {
    const match = /^hf:\/\/([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)@([a-f0-9]{40})\/(.+)$/.exec(value);
    if (!match || !safeRelativePath(match[3])) throw new Error("Hugging Face input must use hf://owner/repo@<40-hex-commit>/<safe-path>.");
    const [repository, revision, file] = [match[1], match[2], match[3]];
    return {
      kind: "huggingface", path: null, repository, revision, file, display_name: path.posix.basename(file), expected_sha256: expected || null,
      canonical_locator: `hf://${repository}@${revision}/${file}`,
      cache_identity: `huggingface\0${repository}\0${revision}\0${file}\0${expected}`,
      request_url: `https://huggingface.co/${repository}/resolve/${revision}/${file.split("/").map(encodeURIComponent).join("/")}`,
      immutability: { kind: "repository_commit", value: revision },
    };
  }
  if (value.startsWith("gs://")) {
    const match = /^gs:\/\/([a-z0-9][a-z0-9._-]{1,221})\/(.+)#(?:generation=)?([1-9][0-9]*)$/.exec(value);
    if (!match || !safeRelativePath(match[2])) throw new Error("GCS input must use gs://bucket/object#generation=<positive-generation>.");
    const [bucket, object, generation] = [match[1], match[2], match[3]];
    return {
      kind: "gcs", path: null, bucket, object, generation, display_name: path.posix.basename(object), expected_sha256: expected || null,
      canonical_locator: `gs://${bucket}/${object}#generation=${generation}`,
      cache_identity: `gcs\0${bucket}\0${object}\0${generation}\0${expected}`,
      immutability: { kind: "gcs_generation", value: generation },
    };
  }
  if (/^https:\/\//i.test(value)) {
    const url = new URL(value);
    if (url.username || url.password) throw new Error("HTTPS artifact URLs must not embed credentials.");
    const fragment = url.hash.replace(/^#(?:sha256=)?/i, "").toLowerCase();
    const pinned = expected || fragment;
    if (!SHA256.test(pinned)) throw new Error("HTTPS artifact input requires #sha256=<64-hex> or --expected-sha256.");
    url.hash = "";
    const displayName = decodeURIComponent(path.posix.basename(url.pathname));
    if (!displayName || displayName === "." || displayName === "..") throw new Error("HTTPS artifact URL has no safe filename.");
    const canonical = new URL(url.origin + url.pathname);
    return {
      kind: "https", path: null, display_name: displayName, expected_sha256: pinned,
      canonical_locator: `${canonical.href}#sha256=${pinned}`,
      cache_identity: `https\0${canonical.href}\0${pinned}`,
      request_url: url.href,
      immutability: { kind: "sha256", value: pinned },
    };
  }
  if (/^kaggle:\/\//i.test(value)) throw new Error("Kaggle URI resolution is not yet available; use a local verified download or HTTPS URL with #sha256.");
  if (/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(value)) throw new Error(`Unsupported artifact source scheme: ${value.split(":", 1)[0]}.`);
  return { kind: "local", path: path.resolve(value) };
}

async function buildRequest(source, fetchImpl, environment) {
  if (source.kind === "huggingface") {
    const token = String(environment.HF_TOKEN || environment.HUGGING_FACE_HUB_TOKEN || "").trim();
    return { url: source.request_url, headers: { "User-Agent": "DeepBOM-CLI/1", ...(token ? { Authorization: `Bearer ${token}` } : {}) }, generation: null };
  }
  if (source.kind === "https") return { url: source.request_url, headers: { "User-Agent": "DeepBOM-CLI/1" }, generation: null };
  const encodedObject = encodeURIComponent(source.object);
  const metadataUrl = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(source.bucket)}/o/${encodedObject}?generation=${source.generation}`;
  const token = String(environment.GOOGLE_OAUTH_ACCESS_TOKEN || "").trim();
  const headers = { "User-Agent": "DeepBOM-CLI/1", ...(token ? { Authorization: `Bearer ${token}` } : {}) };
  const metadataController = new AbortController();
  const metadata = await fetchHeaders(fetchImpl, metadataUrl, { headers, signal: metadataController.signal }, metadataController, HEADER_TIMEOUT_MS);
  if (!metadata.ok) throw new Error(`GCS object metadata lookup failed with HTTP ${metadata.status}.`);
  const body = await metadata.json();
  if (String(body.generation || "") !== source.generation || String(body.bucket || "") !== source.bucket || String(body.name || "") !== source.object) {
    throw new Error("GCS metadata does not match the pinned object generation.");
  }
  return {
    url: `https://storage.googleapis.com/download/storage/v1/b/${encodeURIComponent(source.bucket)}/o/${encodedObject}?alt=media&generation=${source.generation}`,
    headers,
    generation: source.generation,
  };
}

async function streamResponse(body, temporary, { maximumBytes, declared, progress, label, controller, idleTimeoutMs, downloadTimeoutMs }) {
  const hash = createHash("sha256");
  let bytes = 0n;
  let last = 0;
  let idleTimer = null;
  const totalTimer = setTimeout(() => controller.abort(new Error("Remote artifact download exceeded its total time limit.")), downloadTimeoutMs);
  const resetIdleTimer = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => controller.abort(new Error("Remote artifact download made no progress before the idle timeout.")), idleTimeoutMs);
  };
  resetIdleTimer();
  const meter = new Transform({
    transform(chunk, _encoding, callback) {
      resetIdleTimer();
      bytes += BigInt(chunk.length);
      if (bytes > maximumBytes) return callback(new Error(`Remote artifact exceeds the ${maximumBytes}-byte download limit.`));
      hash.update(chunk);
      const now = Date.now();
      if (progress && now - last >= 1000) { progress({ stage: "DOWNLOAD", label, bytes, total: declared }); last = now; }
      callback(null, chunk);
    },
  });
  try {
    await pipeline(
      Readable.fromWeb(body),
      meter,
      createWriteStream(temporary, { flags: "wx", mode: 0o600 }),
      { signal: controller.signal },
    );
  } catch (error) {
    if (controller.signal.aborted) throw new Error(String(controller.signal.reason?.message || controller.signal.reason || "Remote artifact download timed out."));
    throw error;
  } finally {
    clearTimeout(idleTimer);
    clearTimeout(totalTimer);
  }
  if (declared != null && bytes !== declared) throw new Error(`Remote artifact length changed during download: expected ${declared}, observed ${bytes}.`);
  if (progress) progress({ stage: "DOWNLOAD", label, bytes, total: declared, complete: true });
  return { sha256: hash.digest("hex"), byte_length: bytes };
}

async function fetchHeaders(fetchImpl, url, options, controller, timeoutMs) {
  const timer = setTimeout(() => controller.abort(new Error("Remote server did not return response headers before the timeout.")), timeoutMs);
  try {
    return await fetchImpl(url, options);
  } catch (error) {
    if (controller.signal.aborted) throw new Error(String(controller.signal.reason?.message || controller.signal.reason || "Remote response header timeout."));
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function installContentAddressed(temporary, destination, observed) {
  try {
    const present = await stat(destination);
    if (BigInt(present.size) !== observed.byte_length || await sha256File(destination) !== observed.sha256) {
      throw new Error("Content-addressed cache entry conflicts with its SHA-256 path.");
    }
    return;
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  try { await rename(temporary, destination); }
  catch (error) {
    if (error?.code !== "EEXIST") throw error;
    if (await sha256File(destination) !== observed.sha256) throw new Error("Concurrent content-addressed cache write failed verification.");
  }
}

async function readCachedReceipt(cacheDir, receiptPath, source) {
  try {
    const receipt = JSON.parse(await readFile(receiptPath, "utf8"));
    if (receipt.schema !== "deepbom.remote_artifact_acquisition.v1" || receipt.source.canonical_locator !== source.canonical_locator
      || (source.expected_sha256 && receipt.file.sha256 !== source.expected_sha256) || !SHA256.test(receipt.file.sha256)) return null;
    const candidate = path.resolve(cacheDir, ...String(receipt.cache_relative_path).split("/"));
    const root = `${path.resolve(cacheDir)}${path.sep}`;
    if (!candidate.startsWith(root)) return null;
    const metadata = await stat(candidate);
    if (!metadata.isFile() || BigInt(metadata.size).toString() !== receipt.file.byte_length.decimal) return null;
    if (await sha256File(candidate) !== receipt.file.sha256) return null;
    return { path: candidate, acquisition: receipt };
  } catch (error) {
    if (error?.code === "ENOENT" || error instanceof SyntaxError) return null;
    throw error;
  }
}

function acquisitionRecord(source, observed, cacheRelativePath, responseUrl, generation) {
  return {
    schema: "deepbom.remote_artifact_acquisition.v1",
    evidence_class: "OBSERVED_ACQUISITION",
    source: { kind: source.kind, canonical_locator: source.canonical_locator, immutability: source.immutability },
    file: { path: source.display_name, sha256: observed.sha256, byte_length: exact(observed.byte_length) },
    cache_relative_path: cacheRelativePath,
    transport: {
      final_origin: new URL(responseUrl).origin,
      gcs_generation_verified: generation ? true : null,
      authorization_persisted: false,
    },
    trust: { remote_code_execution: "forbidden", pickle_execution: "forbidden", model_code_execution: "forbidden" },
  };
}

async function writeReceipt(filename, receipt) {
  await mkdir(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(receipt, null, 2)}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
  await rm(filename, { force: true });
  await rename(temporary, filename);
}

function parseContentLength(value) {
  if (value == null || value === "") return null;
  if (!/^\d+$/.test(value)) throw new Error("Remote Content-Length is invalid.");
  return BigInt(value);
}

function safeRelativePath(value) {
  return value.length <= 2048 && !value.startsWith("/") && !/^[A-Za-z]:/.test(value)
    && value.split("/").every((part) => part && part !== "." && part !== "..");
}

function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }
function sha256Text(value) { return createHash("sha256").update(String(value), "utf8").digest("hex"); }
export function defaultArtifactCacheDir() { return path.join(process.env.LOCALAPPDATA || path.join(homedir(), ".cache"), "deepbom", "artifacts-v1"); }
function defaultProgress({ stage, label, bytes, total, complete }) {
  const suffix = total == null ? "" : ` / ${formatBytes(total)}`;
  process.stderr.write(`[${stage}] ${label}: ${formatBytes(bytes)}${suffix}${complete ? " complete" : ""}\n`);
}
function formatBytes(value) {
  let scaled = Number(value), index = 0;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  while (scaled >= 1024 && index < units.length - 1) { scaled /= 1024; index += 1; }
  return `${scaled.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
async function sha256File(filename) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest("hex");
}
