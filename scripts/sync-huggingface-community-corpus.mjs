import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  buildCorpusSummary,
  compactCorpusSummary,
  fetchJsonWithRetry,
  HF_CLASSIFIER_SCHEMA,
  HF_CORPUS_PATH,
  HF_CORPUS_SCHEMA,
  HF_ORGANIZATIONS,
  HF_SUMMARY_PATH,
  hfCorpusMetadataCacheDir,
  nextLink,
  normalizeHfRepository,
  readHfMetadataCheckpoint,
  sha256Text,
  writeHfMetadataCheckpoint,
  writeHfCorpus,
} from "./huggingface-community-corpus-lib.mjs";

const options = parseArgs(process.argv.slice(2));
const repositories = [];
const organizationRows = [];

for (const organization of options.organizations) {
  const listed = await listOrganizationRepositories(organization);
  console.log(`${organization}: ${listed.length} repositories listed`);
  const detailed = await mapConcurrent(listed, options.concurrency, async (row, index) => {
    let data = await readHfMetadataCheckpoint(options.cacheDir, row.id, row.sha);
    if (!data) {
      const url = `https://huggingface.co/api/models/${row.id}?blobs=true`;
      ({ data } = await fetchJsonWithRetry(url));
    }
    data = { ...data, ...row, id: row.id, sha: row.sha, siblings: data.siblings };
    await writeHfMetadataCheckpoint(options.cacheDir, data);
    if ((index + 1) % 25 === 0 || index + 1 === listed.length) {
      console.log(`${organization}: metadata ${index + 1}/${listed.length}`);
    }
    return normalizeHfRepository(data);
  });
  repositories.push(...detailed);
  organizationRows.push({
    id: organization,
    url: `https://huggingface.co/${organization}/models`,
    repository_count: detailed.length,
  });
}

repositories.sort((left, right) => left.id.localeCompare(right.id));
const generatedAt = new Date().toISOString();
const manifest = {
  schema: HF_CORPUS_SCHEMA,
  classifier_schema: HF_CLASSIFIER_SCHEMA,
  generated_at: generatedAt,
  source: {
    api: "https://huggingface.co/api/models",
    revision_policy: "Every repository and download URL is bound to the observed 40-hex commit SHA.",
    file_identity_policy: "LFS/Xet files retain upstream SHA-256; Git files retain blob SHA-1 and are verified with the canonical Git blob header after download.",
    license_policy: "Metadata is provenance only. Download and use remain subject to each repository license and access policy.",
  },
  organizations: organizationRows,
  summary: buildCorpusSummary(repositories),
  repositories,
};

await writeHfCorpus(manifest, options.output);
const snapshotBytes = await import("node:fs/promises").then(({ readFile }) => readFile(options.output));
const summary = {
  ...compactCorpusSummary(manifest),
  snapshot_file: path.basename(options.output),
  snapshot_sha256: sha256Text(snapshotBytes),
  snapshot_bytes: snapshotBytes.byteLength,
};
await mkdir(path.dirname(options.summary), { recursive: true });
await writeFile(options.summary, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(`Wrote ${options.output} (${snapshotBytes.byteLength} compressed bytes)`);
console.log(`Wrote ${options.summary}`);
console.log(JSON.stringify(manifest.summary, null, 2));

async function listOrganizationRepositories(organization) {
  const rows = [];
  let url = `https://huggingface.co/api/models?author=${encodeURIComponent(organization)}&limit=1000&full=true&config=true`;
  while (url) {
    const { data, response } = await fetchJsonWithRetry(url);
    if (!Array.isArray(data)) throw new Error(`${organization}: model-list response is not an array.`);
    rows.push(...data);
    url = nextLink(response);
  }
  return [...new Map(rows.map((row) => [row.id, row])).values()];
}

async function mapConcurrent(rows, concurrency, worker) {
  const results = new Array(rows.length);
  let cursor = 0;
  async function consume() {
    while (cursor < rows.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await worker(rows[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, rows.length) }, consume));
  return results;
}

function parseArgs(argv) {
  const output = {
    organizations: [...HF_ORGANIZATIONS],
    output: HF_CORPUS_PATH,
    summary: HF_SUMMARY_PATH,
    cacheDir: hfCorpusMetadataCacheDir(),
    concurrency: 6,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--organization") output.organizations = [required(argv, ++index, key)];
    else if (key === "--output") output.output = required(argv, ++index, key);
    else if (key === "--summary") output.summary = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--concurrency") output.concurrency = boundedInteger(required(argv, ++index, key), 1, 16, key);
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}

function required(argv, index, key) {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`);
  return value;
}

function boundedInteger(value, minimum, maximum, key) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) throw new Error(`${key} must be ${minimum}..${maximum}.`);
  return parsed;
}
