import process from "node:process";

import {
  ensureHfFile,
  HF_RECEIPT_SCHEMA,
  hfCorpusCacheDir,
  readHfCorpus,
  selectRepositories,
  selectRepositoryFiles,
  writeDownloadReceipt,
} from "./huggingface-community-corpus-lib.mjs";

const options = parseArgs(process.argv.slice(2));
const manifest = await readHfCorpus(options.manifest);
const repositories = selectRepositories(manifest, options);
const selections = repositories.flatMap((repository) => selectRepositoryFiles(repository, options.scope, options.formats)
  .map((file) => ({ repository, file })));
const knownBytes = selections.reduce((total, row) => total + Number(row.file.size_bytes || 0), 0);
const unknownSizes = selections.filter((row) => row.file.size_bytes === null);

console.log(`Selected ${repositories.length} repositories / ${selections.length} files / ${formatBytes(knownBytes)} known bytes`);
console.log(`Tier counts: ${JSON.stringify(countBy(repositories, (repository) => repository.tier.name))}`);
console.log(`Format counts: ${JSON.stringify(countBy(selections, (row) => row.file.format))}`);
if (unknownSizes.length) console.log(`Unknown-size files: ${unknownSizes.length} (download refuses them until metadata is complete)`);
if (!options.download) {
  console.log("Plan only. Add --download and a sufficient --max-total-gib budget; use --yes-unbounded only for an intentional unbounded mirror.");
  process.exit(0);
}
if (unknownSizes.length) throw new Error("Selection contains unknown-size files. Re-sync metadata or narrow the selection.");
const limitBytes = options.yesUnbounded ? Number.POSITIVE_INFINITY : options.maxTotalGib * 1024 ** 3;
if (!(knownBytes <= limitBytes)) {
  throw new Error(`Selection is ${formatBytes(knownBytes)}, above --max-total-gib ${options.maxTotalGib}. No bytes were downloaded.`);
}

const cacheDir = options.cacheDir || hfCorpusCacheDir();
const rows = [];
for (let index = 0; index < selections.length; index += 1) {
  const { repository, file } = selections[index];
  const label = `[${index + 1}/${selections.length}] ${repository.id}/${file.path}`;
  process.stdout.write(`${label}: `);
  try {
    const result = await ensureHfFile(repository, file, cacheDir, {
      offline: options.offline,
      onProgress: ({ received, total }) => {
        if (total > 64 * 1024 ** 2 && received % (64 * 1024 ** 2) < 1024 * 1024) {
          process.stdout.write(`\r${label}: ${formatBytes(received)}/${formatBytes(total)}`);
        }
      },
    });
    rows.push({
      repository_id: repository.id,
      revision: repository.revision,
      tier: repository.tier.name,
      path: file.path,
      format: file.format,
      size_bytes: file.size_bytes,
      sha256: result.identity.sha256,
      source_identity: file.lfs_sha256 ? "lfs_sha256" : "git_blob_sha1",
      downloaded: result.downloaded,
      status: "verified",
    });
    console.log(result.downloaded ? "downloaded and verified" : "cached and verified");
  } catch (error) {
    rows.push({
      repository_id: repository.id,
      revision: repository.revision,
      tier: repository.tier.name,
      path: file.path,
      format: file.format,
      size_bytes: file.size_bytes,
      status: "failed",
      error: String(error?.message || error),
    });
    console.log(`failed: ${error?.message || error}`);
    if (!options.keepGoing) break;
  }
}

const receipt = {
  schema: HF_RECEIPT_SCHEMA,
  corpus_generated_at: manifest.generated_at,
  completed_at: new Date().toISOString(),
  cache_dir: cacheDir,
  selection: {
    organizations: options.organizations,
    tiers: options.tiers,
    repository_ids: options.repositoryIds,
    formats: options.formats,
    scope: options.scope,
  },
  selected_file_count: selections.length,
  verified_file_count: rows.filter((row) => row.status === "verified").length,
  failed_file_count: rows.filter((row) => row.status === "failed").length,
  selected_bytes: knownBytes,
  rows,
};
const receiptPaths = await writeDownloadReceipt(cacheDir, receipt);
console.log(`Receipt: ${receiptPaths.latest}`);
console.log(`Receipt history: ${receiptPaths.history}`);
if (receipt.failed_file_count) process.exitCode = 1;

function parseArgs(argv) {
  const output = {
    manifest: "corpus/huggingface-community-corpus.v1.json.gz",
    cacheDir: "",
    organizations: [],
    tiers: [],
    repositoryIds: [],
    formats: [],
    scope: "testable",
    download: false,
    offline: false,
    keepGoing: true,
    yesUnbounded: false,
    maxTotalGib: 4,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--organization") output.organizations.push(required(argv, ++index, key));
    else if (key === "--tier") output.tiers.push(enumValue(required(argv, ++index, key), ["micro", "mid", "large"], key));
    else if (key === "--repo") output.repositoryIds.push(required(argv, ++index, key));
    else if (key === "--format") output.formats.push(required(argv, ++index, key).toLowerCase());
    else if (key === "--scope") output.scope = enumValue(required(argv, ++index, key), ["testable", "model", "repository"], key);
    else if (key === "--download") output.download = true;
    else if (key === "--offline") output.offline = true;
    else if (key === "--fail-fast") output.keepGoing = false;
    else if (key === "--yes-unbounded") output.yesUnbounded = true;
    else if (key === "--max-total-gib") output.maxTotalGib = positiveNumber(required(argv, ++index, key), key);
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}

function required(argv, index, key) {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`);
  return value;
}

function enumValue(value, allowed, key) {
  if (!allowed.includes(value)) throw new Error(`${key} must be one of ${allowed.join(", ")}.`);
  return value;
}

function positiveNumber(value, key) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${key} must be positive.`);
  return parsed;
}

function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = selector(row) || "unknown";
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

function formatBytes(value) {
  if (value >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(2)} TiB`;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(2)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(2)} KiB`;
  return `${value} B`;
}
