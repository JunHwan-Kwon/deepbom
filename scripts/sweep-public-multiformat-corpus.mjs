import { gzipSync } from "node:zlib";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  primaryArtifactSha256,
  publicMultiformatCacheDir,
  readPublicMultiformatCorpus,
} from "./public-multiformat-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readPublicMultiformatCorpus(args.manifest);
const selected = manifest.artifacts.filter((artifact) => (!args.formats.length || args.formats.includes(artifact.format))
  && (!args.artifactIds.length || args.artifactIds.includes(artifact.id)));
const cacheDir = args.cacheDir || publicMultiformatCacheDir();
const workerPath = fileURLToPath(new URL("./public-multiformat-corpus-worker.mjs", import.meta.url));
const outputDir = path.resolve(args.outputDir);
if (!args.formats.length && !args.artifactIds.length) {
  await rm(path.join(outputDir, "receipts"), { recursive: true, force: true });
  await rm(path.join(outputDir, "analysis"), { recursive: true, force: true });
}
await mkdir(path.join(outputDir, "receipts"), { recursive: true });
await mkdir(path.join(outputDir, "analysis"), { recursive: true });

const rows = [];
for (let index = 0; index < selected.length; index += 1) {
  const artifact = selected[index];
  try {
    const repeats = [];
    for (let repeat = 0; repeat < args.repeat; repeat += 1) {
      const resultPath = path.join(outputDir, `.worker-${artifact.id}-${repeat + 1}.json`);
      const child = spawnSync(process.execPath, [
        workerPath,
        "--manifest", args.manifest,
        "--artifact", artifact.id,
        "--cache-dir", cacheDir,
        "--output", resultPath,
        ...(args.offline ? ["--offline"] : []),
      ], { cwd: process.cwd(), encoding: "utf8", timeout: args.workerTimeoutMs, maxBuffer: 4 * 1024 * 1024 });
      if (child.error || child.status !== 0) throw new Error(child.error?.message || child.stderr.trim() || `isolated worker exited ${child.status}`);
      repeats.push(JSON.parse(await readFile(resultPath, "utf8")));
      await rm(resultPath, { force: true });
    }
    const digestSet = new Set(repeats.map((row) => `${row.receipt.analysis_sha256}:${row.receipt.receipt_sha256}`));
    if (digestSet.size !== 1) throw new Error("Analysis or receipt digest changed across deterministic repeats.");
    const first = repeats[0];
    await writeFile(path.join(outputDir, "receipts", `${artifact.id}.json`), `${JSON.stringify(first.receipt, null, 2)}\n`, "utf8");
    await writeFile(path.join(outputDir, "analysis", `${artifact.id}.analysis.json.gz`), gzipSync(Buffer.from(`${JSON.stringify(first.analysis)}\n`), { level: 9, mtime: 0 }));
    rows.push({
      artifact_id: artifact.id,
      format: artifact.format,
      primary_artifact_sha256: primaryArtifactSha256(artifact),
      status: "passed",
      repeat_count: args.repeat,
      deterministic: true,
      receipt_sha256: first.receipt.receipt_sha256,
      analysis_sha256: first.receipt.analysis_sha256,
      analysis_summary: first.receipt.analysis_summary,
      cyclonedx_observation: first.receipt.cyclonedx_observation,
    });
    console.log(`[${index + 1}/${selected.length}] ${artifact.id}: passed`);
  } catch (error) {
    rows.push({ artifact_id: artifact.id, format: artifact.format, primary_artifact_sha256: primaryArtifactSha256(artifact), status: "failed", error: error?.stack || String(error) });
    console.error(`[${index + 1}/${selected.length}] ${artifact.id}: FAILED ${error?.message || error}`);
  }
}

const uniqueRows = deduplicateRows(rows);
const sweep = {
  schema: "deepbom.public_multiformat_corpus_sweep.v1",
  corpus_schema: manifest.schema,
  corpus_generated_at: manifest.generated_at,
  completed_at: new Date().toISOString(),
  method: {
    repeats_per_path_record: args.repeat,
    process_isolation: "fresh_node_process_per_repeat",
    worker_timeout_ms: args.workerTimeoutMs,
    path_records_preserved: true,
    unique_population_key: "primary_artifact_sha256",
    original_bytes_retained: false,
    full_analysis_location: "analysis/<artifact-id>.analysis.json.gz",
    compact_receipt_location: "receipts/<artifact-id>.json",
  },
  path_record_count: rows.length,
  unique_primary_artifact_count: uniqueRows.length,
  status_counts: countBy(rows, (row) => row.status),
  format_path_counts: countBy(rows, (row) => row.format),
  format_unique_counts: countBy(uniqueRows, (row) => row.format),
  rows,
};
await writeFile(path.join(outputDir, "public-multiformat-corpus-sweep.json"), `${JSON.stringify(sweep, null, 2)}\n`, "utf8");
console.log(`Wrote ${path.join(outputDir, "public-multiformat-corpus-sweep.json")}: ${rows.length} path / ${uniqueRows.length} unique.`);
if (rows.some((row) => row.status !== "passed")) process.exitCode = 1;

function deduplicateRows(rows) {
  const bySha = new Map();
  for (const row of rows) {
    const existing = bySha.get(row.primary_artifact_sha256);
    if (!existing || (existing.status !== "passed" && row.status === "passed")) bySha.set(row.primary_artifact_sha256, row);
  }
  return [...bySha.values()];
}

function parseArgs(argv) {
  const output = {
    manifest: "corpus/public-multiformat-corpus.v1.json",
    cacheDir: "",
    outputDir: ".local-validation/public-multiformat-corpus-v1",
    formats: [], artifactIds: [], repeat: 2, offline: false, workerTimeoutMs: 600_000,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--format") output.formats.push(required(argv, ++index, key));
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
    else if (key === "--repeat") output.repeat = boundedInteger(required(argv, ++index, key), 1, 4, key);
    else if (key === "--worker-timeout-ms") output.workerTimeoutMs = boundedInteger(required(argv, ++index, key), 10_000, 3_600_000, key);
    else if (key === "--offline") output.offline = true;
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}
function required(argv, index, key) { const value = argv[index]; if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`); return value; }
function boundedInteger(value, minimum, maximum, key) { const number = Number(value); if (!Number.isSafeInteger(number) || number < minimum || number > maximum) throw new Error(`${key} must be ${minimum}..${maximum}.`); return number; }
function countBy(rows, selector) { const counts = new Map(); for (const row of rows) { const key = selector(row) || "unknown"; counts.set(key, (counts.get(key) || 0) + 1); } return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right))); }
