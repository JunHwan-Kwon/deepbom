import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  HF_RESULT_SCHEMA,
  HF_SWEEP_SCHEMA,
  hfCorpusCacheDir,
  hfFileCachePath,
  hfRepositoryCacheDir,
  isSafeRelativePath,
  readHfCorpus,
  selectRepositories,
  selectRepositoryFiles,
  verifyHfFile,
} from "./huggingface-community-corpus-lib.mjs";
import { findNonFinite } from "./public-model-corpus-lib.mjs";
import { coverageResiduals } from "./corpus-coverage-residuals.mjs";

const scriptPath = fileURLToPath(import.meta.url);
const options = parseArgs(process.argv.slice(2));
if (options.worker) await runWorker(options);
else await runSweep(options);

async function runSweep(args) {
  const manifest = await readHfCorpus(args.manifest);
  const cacheDir = args.cacheDir || hfCorpusCacheDir();
  const outputDir = path.resolve(args.outputDir);
  const repositories = selectRepositories(manifest, args);
  const selected = repositories.flatMap((repository) => selectRepositoryFiles(repository, "testable", args.formats)
    .filter((file) => file.analyzer_support === "current")
    .map((file) => ({ repository, file })));
  await mkdir(outputDir, { recursive: true });
  const rows = [];
  for (let index = 0; index < selected.length; index += 1) {
    const { repository, file } = selected[index];
    const filename = hfFileCachePath(cacheDir, repository, file);
    const verified = await verifyHfFile(filename, file);
    if (!verified.valid) {
      rows.push(baseRow(repository, file, "not_downloaded", verified.reason));
      console.log(`[${index + 1}/${selected.length}] ${repository.id}/${file.path}: not downloaded`);
      continue;
    }
    if (file.size_bytes > args.maxArtifactMib * 1024 ** 2) {
      rows.push(baseRow(repository, file, "skipped_safety_limit", `Artifact exceeds ${args.maxArtifactMib} MiB sweep limit.`));
      console.log(`[${index + 1}/${selected.length}] ${repository.id}/${file.path}: safety skip`);
      continue;
    }
    const repeats = [];
    for (let repeat = 0; repeat < args.repeat; repeat += 1) {
      const resultPath = path.join(outputDir, `.worker-${index}-${repeat}.json`);
      const child = spawnSync(process.execPath, [
        scriptPath,
        "--worker",
        "--artifact", filename,
        "--repository-id", repository.id,
        "--revision", repository.revision,
        "--worker-tier", repository.tier.name,
        "--worker-format", file.format,
        "--worker-path", file.path,
        "--repository-cache-dir", hfRepositoryCacheDir(cacheDir, repository),
        "--result", resultPath,
      ], {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: args.timeoutMs,
        maxBuffer: 4 * 1024 * 1024,
      });
      if (child.error || child.status !== 0) {
        repeats.push(baseRow(repository, file, "failed", child.error?.message || child.stderr.trim() || `worker exited ${child.status}`));
        break;
      }
      repeats.push(JSON.parse(await readFile(resultPath, "utf8")));
    }
    const successful = repeats.filter((row) => isPassingStatus(row.status));
    if (successful.length === args.repeat && new Set(successful.map((row) => row.analysis_sha256)).size === 1) {
      rows.push({ ...successful[0], repeat_count: args.repeat, deterministic: true });
    } else if (successful.length === args.repeat) {
      rows.push(baseRow(repository, file, "failed", "Analysis digest changed across isolated repeats."));
    } else {
      rows.push(repeats.find((row) => row.status !== "passed") || baseRow(repository, file, "failed", "Worker produced no result."));
    }
    console.log(`[${index + 1}/${selected.length}] ${repository.id}/${file.path}: ${rows.at(-1).status}`);
  }
  const sweep = {
    schema: HF_SWEEP_SCHEMA,
    corpus_generated_at: manifest.generated_at,
    completed_at: new Date().toISOString(),
    selected_artifact_count: selected.length,
    status_counts: countBy(rows, (row) => row.status),
    tier_counts: countBy(rows, (row) => row.tier),
    format_counts: countBy(rows, (row) => row.format),
    rows,
  };
  await writeFile(path.join(outputDir, "huggingface-corpus-sweep.json"), `${JSON.stringify(sweep, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.join(outputDir, "huggingface-corpus-sweep.json")}`);
  if (rows.some((row) => row.status === "failed")) process.exitCode = 1;
}

async function runWorker(args) {
  const bytes = new Uint8Array(await readFile(args.artifact));
  const started = performance.now();
  let analysis;
  if (args.workerFormat === "tflite") {
    const { initSync, analyze_tflite_for_target } = await import("../pkg/tflite_wasm_audit.js");
    initSync({ module: await readFile("pkg/tflite_wasm_audit_bg.wasm") });
    analysis = analyze_tflite_for_target(bytes, path.basename(args.artifact), "android_mid_a55");
  } else if (args.workerFormat === "onnx") {
    const { analyzeOnnxModel } = await import("../web/onnx.js");
    analysis = analyzeOnnxModel(bytes, path.basename(args.artifact));
    const externalDataFiles = await loadOnnxExternalDataFiles(analysis, args.workerPath, args.repositoryCacheDir);
    if (externalDataFiles.length) {
      analysis = analyzeOnnxModel(bytes, path.basename(args.artifact), null, { externalDataFiles });
    }
  } else {
    throw new Error(`Unsupported worker format: ${args.workerFormat}.`);
  }
  const serialized = JSON.stringify(analysis);
  const nonFinite = findNonFinite(analysis);
  const externalStatus = analysis.onnx_external_data?.status || null;
  const analysisStatus = nonFinite.length
    ? "failed"
    : externalStatus && !["assessed_absent", "verified_payloads"].includes(externalStatus)
      ? "passed_partial_external_data"
      : "passed";
  const result = {
    schema: HF_RESULT_SCHEMA,
    repository_id: args.repositoryId,
    revision: args.revision,
    tier: args.workerTier,
    path: args.workerPath,
    format: args.workerFormat,
    status: analysisStatus,
    error: nonFinite.length ? `Non-finite values at ${nonFinite.join(", ")}.` : "",
    artifact_size_bytes: bytes.byteLength,
    artifact_sha256: createHash("sha256").update(bytes).digest("hex"),
    analysis_sha256: createHash("sha256").update(serialized).digest("hex"),
    analyzer_elapsed_ms: performance.now() - started,
    operator_count: Number(analysis.operator_count || analysis.ops?.length || 0),
    tensor_count: Number(analysis.tensor_count || analysis.tensors?.length || 0),
    finding_count: Number(analysis.findings?.length || 0),
    onnx_external_data_status: externalStatus,
    onnx_external_data_tensor_count: Number(analysis.onnx_external_data?.tensor_count || 0),
    onnx_external_data_verified_payload_count: Number(analysis.onnx_external_data?.verified_payload_count || 0),
    coverage_residuals: coverageResiduals(analysis),
    non_finite_paths: nonFinite,
  };
  await writeFile(args.result, `${JSON.stringify(result, null, 2)}\n`, "utf8");
}

function baseRow(repository, file, status, error) {
  return {
    schema: HF_RESULT_SCHEMA,
    repository_id: repository.id,
    revision: repository.revision,
    tier: repository.tier.name,
    path: file.path,
    format: file.format,
    status,
    error,
  };
}

function parseArgs(argv) {
  const output = {
    worker: false,
    manifest: "corpus/huggingface-community-corpus.v1.json.gz",
    cacheDir: "",
    outputDir: "reports/huggingface-community-corpus-v1",
    organizations: [],
    tiers: [],
    repositoryIds: [],
    formats: [],
    repeat: 2,
    timeoutMs: 10 * 60_000,
    maxArtifactMib: 256,
    workerTier: "",
    workerFormat: "",
    workerPath: "",
    repositoryCacheDir: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--worker") output.worker = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--organization") output.organizations.push(required(argv, ++index, key));
    else if (key === "--tier") output.tiers.push(required(argv, ++index, key));
    else if (key === "--repo") output.repositoryIds.push(required(argv, ++index, key));
    else if (key === "--format") output.formats.push(required(argv, ++index, key));
    else if (key === "--repeat") output.repeat = boundedInteger(required(argv, ++index, key), 1, 5, key);
    else if (key === "--timeout-ms") output.timeoutMs = boundedInteger(required(argv, ++index, key), 1_000, 3_600_000, key);
    else if (key === "--max-artifact-mib") output.maxArtifactMib = boundedInteger(required(argv, ++index, key), 1, 1024, key);
    else if (key === "--artifact") output.artifact = required(argv, ++index, key);
    else if (key === "--repository-id") output.repositoryId = required(argv, ++index, key);
    else if (key === "--revision") output.revision = required(argv, ++index, key);
    else if (key === "--worker-tier") output.workerTier = required(argv, ++index, key);
    else if (key === "--worker-format") output.workerFormat = required(argv, ++index, key);
    else if (key === "--worker-path") output.workerPath = required(argv, ++index, key);
    else if (key === "--repository-cache-dir") output.repositoryCacheDir = required(argv, ++index, key);
    else if (key === "--result") output.result = required(argv, ++index, key);
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

function countBy(rows, selector) {
  return Object.fromEntries([...rows.reduce((counts, row) => {
    const key = selector(row) || "unknown";
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

async function loadOnnxExternalDataFiles(analysis, modelPath, repositoryCacheDir) {
  const references = [...new Set((analysis.onnx_external_data?.tensors || [])
    .map((row) => String(row.location || ""))
    .filter(Boolean))];
  if (!references.length) return [];
  const modelDirectory = path.posix.dirname(modelPath);
  const rows = [];
  let aggregateBytes = 0;
  for (const reference of references) {
    const repositoryPath = path.posix.normalize(path.posix.join(modelDirectory, reference));
    if (!isSafeRelativePath(repositoryPath)) throw new Error(`Unsafe ONNX external-data path: ${reference}.`);
    const filename = path.resolve(repositoryCacheDir, ...repositoryPath.split("/"));
    const root = path.resolve(repositoryCacheDir);
    if (filename !== root && !filename.startsWith(`${root}${path.sep}`)) throw new Error(`ONNX external-data path escaped repository cache: ${reference}.`);
    let bytes;
    try {
      bytes = new Uint8Array(await readFile(filename));
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (bytes.byteLength > 536_870_912) continue;
    aggregateBytes += bytes.byteLength;
    if (aggregateBytes > 1_073_741_824) return [];
    rows.push({
      path: reference,
      bytes,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      sha1: createHash("sha1").update(bytes).digest("hex"),
    });
  }
  return rows;
}

function isPassingStatus(status) {
  return status === "passed" || status === "passed_partial_external_data";
}
