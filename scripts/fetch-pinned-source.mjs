import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const gitSources = new Map();
const forceGitRepositories = new Set();
const temporaryRepositories = [];

export function parseGitHubRawSource(sourceRef) {
  const match = String(sourceRef).match(/^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([A-Za-z0-9._-]+)\/(.+)$/);
  if (!match) throw new Error(`Source URL is not a bounded GitHub raw URL: ${sourceRef}`);
  return { owner: match[1], repository: match[2], ref: match[3], path: match[4] };
}

export async function fetchPinnedBytes(sourceRef, { label = "Pinned" } = {}) {
  const source = parseGitHubRawSource(sourceRef);
  const repositoryKey = `${source.owner}/${source.repository}`;
  if (forceGitRepositories.has(repositoryKey)) {
    try {
      return readGitSource(source);
    } catch (gitError) {
      return fetchRawWithRetry(sourceRef, label, gitError);
    }
  }

  const response = await fetch(sourceRef, {
    redirect: "error",
    headers: { "User-Agent": "DEEPBOM-source-pin-verifier", Accept: "text/plain" },
  });
  if (response.ok) return new Uint8Array(await response.arrayBuffer());
  if (response.status !== 429 && response.status < 500) {
    throw new Error(`${label} source fetch failed (${response.status}): ${sourceRef}`);
  }
  await response.body?.cancel().catch(() => {});
  forceGitRepositories.add(repositoryKey);
  console.warn(`${label} fetch returned ${response.status}; verifying ${repositoryKey}@${source.ref} through Git protocol.`);
  try {
    return readGitSource(source);
  } catch (gitError) {
    return fetchRawWithRetry(sourceRef, label, gitError);
  }
}

export async function fetchPinnedText(sourceRef, options) {
  return new TextDecoder().decode(await fetchPinnedBytes(sourceRef, options));
}

function readGitSource(source) {
  const key = `${source.owner}/${source.repository}@${source.ref}`;
  let repository = gitSources.get(key);
  if (!repository) {
    repository = mkdtempSync(join(tmpdir(), "deepbom-source-pin-"));
    temporaryRepositories.push(repository);
    execFileSync("git", ["init", "-q"], { cwd: repository, stdio: "ignore" });
    execFileSync("git", ["remote", "add", "origin", `https://github.com/${source.owner}/${source.repository}.git`], { cwd: repository, stdio: "ignore" });
    execFileSync("git", ["-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.ref], {
      cwd: repository,
      stdio: "ignore",
    });
    gitSources.set(key, repository);
  }
  let lastError;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return new Uint8Array(execFileSync("git", ["show", `FETCH_HEAD:${source.path}`], {
        cwd: repository,
        encoding: "buffer",
        maxBuffer: 64 * 1024 * 1024,
        stdio: ["ignore", "pipe", "ignore"],
      }));
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Git could not read ${source.owner}/${source.repository}@${source.ref}:${source.path}`, { cause: lastError });
}

async function fetchRawWithRetry(sourceRef, label, gitError) {
  let lastStatus = 0;
  for (let attempt = 0; attempt < 3; attempt++) {
    const response = await fetch(sourceRef, {
      redirect: "error",
      headers: { "User-Agent": "DEEPBOM-source-pin-verifier", Accept: "text/plain" },
    });
    if (response.ok) return new Uint8Array(await response.arrayBuffer());
    lastStatus = response.status;
    const retryable = response.status === 429 || response.status >= 500;
    if (!retryable || attempt === 2) break;
    await response.body?.cancel().catch(() => {});
    await new Promise((resolve) => setTimeout(resolve, 1_000 * (2 ** attempt)));
  }
  throw new Error(`${label} source verification failed through Git and raw HTTP (${lastStatus}): ${sourceRef}`, { cause: gitError });
}

process.on("exit", () => {
  for (const repository of temporaryRepositories) rmSync(repository, { recursive: true, force: true });
});
