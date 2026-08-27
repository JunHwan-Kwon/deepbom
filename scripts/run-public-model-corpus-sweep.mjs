import { spawnSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  corpusCacheDir,
  ensureCorpusModel,
  findNonFinite,
  readCorpusManifest,
  RESULT_SCHEMA,
  sha256Bytes,
  sha256Text,
  SWEEP_SCHEMA,
  validateCorpusResult,
} from "./public-model-corpus-lib.mjs";

const scriptPath = fileURLToPath(import.meta.url);
const args = parseArgs(process.argv.slice(2));

if (args.worker) {
  await runWorker(args);
} else {
  await runSweep(args);
}

async function runSweep(options) {
  const manifest = await readCorpusManifest(options.manifest);
  const cacheDir = path.resolve(options.cacheDir || corpusCacheDir());
  const outputDir = path.resolve(options.outputDir || path.join("reports", manifest.corpus_id));
  const selected = options.modelIds.length
    ? manifest.models.filter((model) => options.modelIds.includes(model.id))
    : manifest.models;
  if (selected.length !== (options.modelIds.length || manifest.models.length)) throw new Error("One or more requested model ids are absent from the corpus.");
  await mkdir(outputDir, { recursive: true });
  const startedAt = new Date().toISOString();
  const rows = [];

  for (let modelIndex = 0; modelIndex < selected.length; modelIndex += 1) {
    const model = selected[modelIndex];
    process.stdout.write(`[${modelIndex + 1}/${selected.length}] ${model.id}: cache`);
    try {
      const artifact = await ensureCorpusModel(model, cacheDir, { offline: options.offline });
      process.stdout.write(artifact.downloaded ? " downloaded; analyze" : " verified; analyze");
      const repeats = [];
      for (let repeat = 0; repeat < options.repeat; repeat += 1) {
        const resultPath = path.join(outputDir, `.${model.id}.repeat-${repeat + 1}.json`);
        const child = spawnSync(process.execPath, [
          scriptPath,
          "--worker",
          "--artifact", artifact.filename,
          "--model-id", model.id,
          "--task", model.task,
          "--published-precision", model.published_precision,
          "--target", manifest.target_profile_id,
          "--result", resultPath,
        ], {
          cwd: process.cwd(),
          encoding: "utf8",
          timeout: options.timeoutMs,
          maxBuffer: 4 * 1024 * 1024,
        });
        if (child.error) throw child.error;
        if (child.status !== 0) throw new Error(child.stderr.trim() || `worker exited ${child.status}`);
        repeats.push(validateCorpusResult(JSON.parse(await readFile(resultPath, "utf8"))));
      }
      if (repeats.some((row) => row.artifact_sha256 !== model.sha256 || row.artifact_size_bytes !== model.size_bytes)) {
        throw new Error("worker artifact identity does not match the generation-pinned manifest.");
      }
      const digests = new Set(repeats.map((row) => row.analysis_sha256));
      if (digests.size !== 1) throw new Error(`analysis digest changed across ${options.repeat} isolated process runs.`);
      rows.push({ ...repeats[0], repeat_count: options.repeat, deterministic: true, status: "passed" });
      console.log(` ${repeats[0].artifact_class} ${repeats[0].quant_research_coverage.class_supported}/15 passed`);
    } catch (error) {
      rows.push({
        schema: RESULT_SCHEMA,
        id: model.id,
        task: model.task,
        published_precision: model.published_precision,
        status: "failed",
        error: String(error?.message || error),
      });
      console.log(` failed: ${error?.message || error}`);
    }
  }

  const result = {
    schema: SWEEP_SCHEMA,
    corpus_id: manifest.corpus_id,
    manifest_sha256: sha256Text(await readFile(options.manifest, "utf8")),
    target_profile_id: manifest.target_profile_id,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    requested_artifact_count: selected.length,
    passed_artifact_count: rows.filter((row) => row.status === "passed").length,
    failed_artifact_count: rows.filter((row) => row.status === "failed").length,
    deterministic_artifact_count: rows.filter((row) => row.deterministic).length,
    artifact_class_counts: countBy(rows.filter((row) => row.status === "passed"), (row) => row.artifact_class),
    rows,
  };
  await writeFile(path.join(outputDir, "corpus-sweep.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await writeFile(path.join(outputDir, "corpus-sweep.csv"), csv(result), "utf8");
  console.log(`Wrote ${path.join(outputDir, "corpus-sweep.json")}`);
  if (result.failed_artifact_count) process.exitCode = 1;
}

async function runWorker(options) {
  const { initSync, analyze_tflite_for_target } = await import("../pkg/tflite_wasm_audit.js");
  const { ensureQuantResearchCoverage } = await import("../web/lib/quant-research-applicability.js");
  initSync({ module: await readFile("pkg/tflite_wasm_audit_bg.wasm") });
  const bytes = new Uint8Array(await readFile(options.artifact));
  const started = performance.now();
  const analysis = analyze_tflite_for_target(bytes, path.basename(options.artifact), options.target);
  const elapsedMs = performance.now() - started;
  const coverage = ensureQuantResearchCoverage(analysis);
  const serialized = JSON.stringify(analysis);
  const phantomReferences = phantomOpReferences(serialized, analysis.ops);
  const result = {
    schema: RESULT_SCHEMA,
    id: options.modelId,
    task: options.task,
    published_precision: options.publishedPrecision,
    format: analysis.format,
    artifact_sha256: sha256Bytes(bytes),
    artifact_size_bytes: bytes.byteLength,
    analysis_sha256: sha256Text(serialized),
    analyzer_elapsed_ms: elapsedMs,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs ?? null,
    quantization_classification: analysis.quantization_status?.classification || "unknown",
    artifact_class: coverage.artifact_class,
    quant_research_coverage: {
      lab_count: coverage.lab_count,
      class_supported: coverage.class_supported_lab_count,
      class_excluded: coverage.class_excluded_lab_count,
      assessed: coverage.assessed_lab_count,
      partial: coverage.partial_lab_count,
      not_assessed: coverage.not_assessed_lab_count,
      not_applicable: coverage.not_applicable_lab_count,
    },
    finding_counts: countBy(analysis.findings || [], (finding) => String(finding.priority || finding.severity || "unknown")),
    quality: {
      non_finite_paths: findNonFinite(analysis),
      phantom_reference_count: phantomReferences.length,
      phantom_references: phantomReferences,
      op_index_unique: new Set((analysis.ops || []).map((op) => op.index)).size === (analysis.ops || []).length,
    },
  };
  await writeFile(options.result, `${JSON.stringify(result, null, 2)}\n`, "utf8");
}

function phantomOpReferences(serialized, ops) {
  const observed = new Map((ops || []).map((op) => [Number(op.index), String(op.name)]));
  const failures = [];
  for (const match of serialized.matchAll(/#(\d+)\s+([A-Z][A-Z0-9_]+)(?![a-z])/g)) {
    const index = Number(match[1]);
    const expected = observed.get(index);
    if (!expected || expected !== match[2]) failures.push(match[0]);
    if (failures.length >= 20) break;
  }
  if (serialized.includes("#undefined")) failures.push("#undefined");
  return [...new Set(failures)];
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
    "id", "task", "published_precision", "status", "artifact_class",
    "operator_count", "tensor_count", "total_macs", "analyzer_elapsed_ms",
    "class_supported", "assessed", "not_assessed", "not_applicable",
    "deterministic", "analysis_sha256", "error",
  ];
  const lines = [columns.join(",")];
  for (const row of result.rows) {
    const values = { ...row, ...row.quant_research_coverage };
    lines.push(columns.map((column) => quoteCsv(values[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function quoteCsv(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}

function parseArgs(argv) {
  const output = {
    worker: false,
    manifest: "corpus/public-tflite-corpus.v1.json",
    cacheDir: "",
    outputDir: "",
    offline: false,
    repeat: 2,
    timeoutMs: 10 * 60_000,
    modelIds: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--worker") output.worker = true;
    else if (key === "--offline") output.offline = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--repeat") output.repeat = boundedInteger(required(argv, ++index, key), 1, 5, key);
    else if (key === "--timeout-ms") output.timeoutMs = boundedInteger(required(argv, ++index, key), 1_000, 3_600_000, key);
    else if (key === "--model") output.modelIds.push(required(argv, ++index, key));
    else if (key === "--artifact") output.artifact = required(argv, ++index, key);
    else if (key === "--model-id") output.modelId = required(argv, ++index, key);
    else if (key === "--task") output.task = required(argv, ++index, key);
    else if (key === "--published-precision") output.publishedPrecision = required(argv, ++index, key);
    else if (key === "--target") output.target = required(argv, ++index, key);
    else if (key === "--result") output.result = required(argv, ++index, key);
    else throw new Error(`Unknown argument: ${key}`);
  }
  if (output.worker && ![output.artifact, output.modelId, output.task, output.publishedPrecision, output.target, output.result].every(Boolean)) {
    throw new Error("Corpus worker arguments are incomplete.");
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
