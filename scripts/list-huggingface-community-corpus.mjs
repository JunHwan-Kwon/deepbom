import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  readHfCorpus,
  selectRepositories,
} from "./huggingface-community-corpus-lib.mjs";

const options = parseArgs(process.argv.slice(2));
const manifest = await readHfCorpus(options.manifest);
const repositories = selectRepositories(manifest, options);
const artifacts = repositories.flatMap((repository) => repository.files
  .filter((file) => ["model", "model_bundle"].includes(file.kind))
  .filter((file) => !options.formats.length || options.formats.includes(file.format))
  .map((file) => ({ repository, file })));

await mkdir(options.outputDir, { recursive: true });
const repositoryPath = path.join(options.outputDir, "repositories.csv");
const artifactPath = path.join(options.outputDir, "artifacts.csv");
const summaryPath = path.join(options.outputDir, "catalog-summary.json");

await writeFile(repositoryPath, repositoryCsv(repositories), "utf8");
await writeFile(artifactPath, artifactCsv(artifacts), "utf8");
await writeFile(summaryPath, `${JSON.stringify({
  schema: "deepbom.huggingface_community_catalog_export.v1",
  corpus_generated_at: manifest.generated_at,
  repository_count: repositories.length,
  artifact_count: artifacts.length,
  repository_csv: path.basename(repositoryPath),
  artifact_csv: path.basename(artifactPath),
  filters: {
    organizations: options.organizations,
    tiers: options.tiers,
    repository_ids: options.repositoryIds,
    formats: options.formats,
  },
}, null, 2)}\n`, "utf8");

console.log(`Repositories: ${repositories.length}`);
console.log(`Model artifacts: ${artifacts.length}`);
console.log(`Tier counts: ${JSON.stringify(countBy(repositories, (repository) => repository.tier.name))}`);
console.log(`Format counts: ${JSON.stringify(countBy(artifacts, ({ file }) => file.format))}`);
console.log(`Wrote ${repositoryPath}`);
console.log(`Wrote ${artifactPath}`);
console.log(`Wrote ${summaryPath}`);

function repositoryCsv(rows) {
  const columns = [
    "repository_id", "organization", "revision", "last_modified", "tier",
    "tier_confidence", "tier_reasons", "runtime_family", "pipeline_tag",
    "library_name", "license", "architectures", "model_type", "gated",
    "private", "file_count", "model_file_count", "analyzer_supported_file_count",
    "known_file_bytes", "model_container_bytes", "model_payload_bytes",
  ];
  return toCsv(columns, rows.map((repository) => ({
    repository_id: repository.id,
    organization: repository.organization,
    revision: repository.revision,
    last_modified: repository.last_modified,
    tier: repository.tier.name,
    tier_confidence: repository.tier.confidence,
    tier_reasons: repository.tier.reasons.join(";"),
    runtime_family: repository.runtime_family,
    pipeline_tag: repository.pipeline_tag,
    library_name: repository.library_name,
    license: repository.metadata.license,
    architectures: repository.metadata.architectures.join(";"),
    model_type: repository.metadata.model_type,
    gated: repository.gated,
    private: repository.private,
    file_count: repository.file_count,
    model_file_count: repository.model_file_count,
    analyzer_supported_file_count: repository.analyzer_supported_file_count,
    known_file_bytes: repository.byte_summary.known_file_bytes,
    model_container_bytes: repository.byte_summary.model_container_bytes,
    model_payload_bytes: repository.byte_summary.model_payload_bytes,
  })));
}

function artifactCsv(rows) {
  const columns = [
    "repository_id", "revision", "tier", "tier_confidence", "runtime_family",
    "path", "format", "analyzer_support", "size_bytes", "storage",
    "lfs_sha256", "blob_id", "resolve_url",
  ];
  return toCsv(columns, rows.map(({ repository, file }) => ({
    repository_id: repository.id,
    revision: repository.revision,
    tier: repository.tier.name,
    tier_confidence: repository.tier.confidence,
    runtime_family: repository.runtime_family,
    path: file.path,
    format: file.format,
    analyzer_support: file.analyzer_support,
    size_bytes: file.size_bytes,
    storage: file.storage,
    lfs_sha256: file.lfs_sha256,
    blob_id: file.blob_id,
    resolve_url: `https://huggingface.co/${repository.id}/resolve/${repository.revision}/${file.path.split("/").map(encodeURIComponent).join("/")}?download=true`,
  })));
}

function toCsv(columns, rows) {
  return `${[
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n")}\n`;
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}

function parseArgs(argv) {
  const output = {
    manifest: "corpus/huggingface-community-corpus.v1.json.gz",
    outputDir: "reports/huggingface-community-corpus-v1/catalog",
    organizations: [],
    tiers: [],
    repositoryIds: [],
    formats: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--organization") output.organizations.push(required(argv, ++index, key));
    else if (key === "--tier") output.tiers.push(required(argv, ++index, key));
    else if (key === "--repo") output.repositoryIds.push(required(argv, ++index, key));
    else if (key === "--format") output.formats.push(required(argv, ++index, key));
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}

function required(argv, index, key) {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`);
  return value;
}

function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = selector(row) || "unknown";
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}
