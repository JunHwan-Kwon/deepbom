import { spawnSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  CURATED_MICRO_CORPUS_PATH,
  CURATED_MICRO_SWEEP_SCHEMA,
  curatedMicroCacheDir,
  ensureCuratedMicroArtifact,
  readCuratedMicroCorpus,
  sha256Text,
} from "./curated-micro-corpus-lib.mjs";
import { validateCorpusResult } from "./public-model-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readCuratedMicroCorpus(args.manifest);
const cacheDir = path.resolve(args.cacheDir || curatedMicroCacheDir());
const outputDir = path.resolve(args.outputDir || path.join("reports", manifest.corpus_id));
const selected = args.artifactIds.length
  ? manifest.artifacts.filter((artifact) => args.artifactIds.includes(artifact.id))
  : manifest.artifacts;
if (selected.length !== (args.artifactIds.length || manifest.artifacts.length)) {
  throw new Error("One or more requested curated artifact ids are absent from the corpus.");
}
await mkdir(outputDir, { recursive: true });

const rows = [];
const startedAt = new Date().toISOString();
for (let artifactIndex = 0; artifactIndex < selected.length; artifactIndex += 1) {
  const artifact = selected[artifactIndex];
  process.stdout.write(`[${artifactIndex + 1}/${selected.length}] ${artifact.id}: cache`);
  try {
    const cached = await ensureCuratedMicroArtifact(manifest, artifact, cacheDir, {
      offline: args.offline,
      onProgress: ({ received, total }) => {
        if (process.stdout.isTTY) process.stdout.write(`\r[${artifactIndex + 1}/${selected.length}] ${artifact.id}: ${formatBytes(received)} / ${formatBytes(total)}`);
      },
    });
    process.stdout.write(cached.downloaded ? " downloaded; analyze" : " verified; analyze");
    const repeats = [];
    for (let repeat = 0; repeat < args.repeat; repeat += 1) {
      const resultPath = path.join(outputDir, `.${artifact.id}.repeat-${repeat + 1}.json`);
      const child = spawnSync(process.execPath, [
        "scripts/run-public-model-corpus-sweep.mjs",
        "--worker",
        "--artifact", cached.filename,
        "--model-id", artifact.id,
        "--task", artifact.task,
        "--published-precision", artifact.published_precision,
        "--target", manifest.target_profile_id,
        "--result", resultPath,
      ], {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: args.timeoutMs,
        maxBuffer: 4 * 1024 * 1024,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) throw new Error(child.stderr.trim() || `worker exited ${child.status}`);
      repeats.push(validateCorpusResult(JSON.parse(await readFile(resultPath, "utf8"))));
    }
    if (repeats.some((row) => row.artifact_sha256 !== artifact.sha256
      || row.artifact_size_bytes !== artifact.size_bytes)) {
      throw new Error("worker artifact identity does not match the curated manifest.");
    }
    const digests = new Set(repeats.map((row) => row.analysis_sha256));
    if (digests.size !== 1) throw new Error(`analysis digest changed across ${args.repeat} isolated process runs.`);
    if (repeats[0].artifact_class !== "full_integer") {
      throw new Error(`expected full_integer, observed ${repeats[0].artifact_class}.`);
    }
    const upstreamMacs = Number(artifact.upstream_metrics?.macs);
    const macRelativeDelta = Number.isFinite(upstreamMacs) && upstreamMacs > 0
      ? Math.abs(Number(repeats[0].total_macs) - upstreamMacs) / upstreamMacs
      : null;
    if (macRelativeDelta !== null && macRelativeDelta > 0.01) {
      throw new Error(`observed MACs differ from the rounded upstream value by ${(macRelativeDelta * 100).toFixed(2)}%.`);
    }
    rows.push({
      ...repeats[0],
      source_id: artifact.source_id,
      source_revision: manifest.sources.find((source) => source.id === artifact.source_id)?.revision || null,
      upstream_metrics: artifact.upstream_metrics,
      upstream_macs_relative_delta: macRelativeDelta,
      repeat_count: args.repeat,
      deterministic: true,
      status: "passed",
    });
    console.log(` ${repeats[0].operator_count} ops; ${repeats[0].quant_research_coverage.assessed}/15 labs passed`);
  } catch (error) {
    rows.push({
      id: artifact.id,
      source_id: artifact.source_id,
      status: "failed",
      error: String(error?.message || error),
    });
    console.log(` failed: ${error?.message || error}`);
  }
}

const result = {
  schema: CURATED_MICRO_SWEEP_SCHEMA,
  corpus_id: manifest.corpus_id,
  manifest_sha256: sha256Text(await readFile(args.manifest, "utf8")),
  analysis_target_profile_id: manifest.target_profile_id,
  analysis_target_semantics: manifest.analysis_target_semantics,
  started_at: startedAt,
  completed_at: new Date().toISOString(),
  requested_artifact_count: selected.length,
  passed_artifact_count: rows.filter((row) => row.status === "passed").length,
  failed_artifact_count: rows.filter((row) => row.status === "failed").length,
  deterministic_artifact_count: rows.filter((row) => row.deterministic).length,
  artifact_class_counts: countBy(rows.filter((row) => row.status === "passed"), (row) => row.artifact_class),
  rows,
};
await writeFile(path.join(outputDir, "curated-micro-sweep.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
await writeFile(path.join(outputDir, "curated-micro-sweep.csv"), csv(result), "utf8");
console.log(`Wrote ${path.join(outputDir, "curated-micro-sweep.json")}`);
if (result.failed_artifact_count) process.exitCode = 1;

function parseArgs(argv) {
  const output = {
    manifest: CURATED_MICRO_CORPUS_PATH,
    cacheDir: "",
    outputDir: "",
    offline: false,
    repeat: 2,
    timeoutMs: 10 * 60_000,
    artifactIds: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--offline") output.offline = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--repeat") output.repeat = boundedInteger(required(argv, ++index, key), 1, 5, key);
    else if (key === "--timeout-ms") output.timeoutMs = boundedInteger(required(argv, ++index, key), 1_000, 3_600_000, key);
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
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
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${key} must be ${minimum}..${maximum}.`);
  }
  return parsed;
}

function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = selector(row);
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

function csv(result) {
  const columns = [
    "id", "task", "status", "artifact_class", "operator_count", "tensor_count",
    "total_macs", "analyzer_elapsed_ms", "upstream_macs_relative_delta",
    "deterministic", "analysis_sha256", "error",
  ];
  const lines = [columns.join(",")];
  for (const row of result.rows) lines.push(columns.map((column) => quoteCsv(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function quoteCsv(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}

function formatBytes(value) {
  return `${(Number(value || 0) / 1024 ** 2).toFixed(2)} MiB`;
}
