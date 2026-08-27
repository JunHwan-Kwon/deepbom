import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  countBy,
  csvText,
  sha256Text,
} from "./google-legacy-corpus-lib.mjs";
import { hfCorpusCacheDir } from "./huggingface-community-corpus-lib.mjs";
import {
  GOOGLE_MODERN_COMPARATOR_PATH,
  ensureGoogleModernComparator,
  loadResolvedGoogleModernComparators,
  readGoogleModernComparators,
} from "./google-modern-comparator-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readGoogleModernComparators(args.manifest);
const allResolved = await loadResolvedGoogleModernComparators(manifest);
const selected = args.artifactIds.length
  ? allResolved.filter((row) => args.artifactIds.includes(row.artifact.id))
  : allResolved;
if (selected.length !== (args.artifactIds.length || allResolved.length)) {
  throw new Error("One or more requested modern comparator ids are absent.");
}
const cacheDir = path.resolve(args.cacheDir || hfCorpusCacheDir());
const outputDir = path.resolve(args.outputDir || path.join("reports", manifest.corpus_id));
await mkdir(outputDir, { recursive: true });

const rows = [];
for (let index = 0; index < selected.length; index += 1) {
  const resolved = selected[index];
  process.stdout.write(`[${index + 1}/${selected.length}] ${resolved.artifact.id}: cache`);
  try {
    const cached = await ensureGoogleModernComparator(resolved, cacheDir, { offline: args.offline });
    process.stdout.write(cached.downloaded ? " downloaded; analyze" : " verified; analyze");
    const repeats = [];
    for (let repeat = 0; repeat < args.repeat; repeat += 1) {
      const resultPath = path.join(outputDir, `.${resolved.artifact.id}.repeat-${repeat + 1}.json`);
      const child = spawnSync(process.execPath, [
        "scripts/run-google-legacy-corpus-sweep.mjs",
        "--worker",
        "--artifact-path", cached.filename,
        "--artifact-id", resolved.artifact.id,
        "--target", manifest.target_profile_id,
        "--result", resultPath,
      ], {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: args.timeoutMs,
        maxBuffer: 16 * 1024 * 1024,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) throw new Error(child.stderr.trim() || `worker exited ${child.status}`);
      repeats.push(JSON.parse(await readFile(resultPath, "utf8")));
      await rm(resultPath, { force: true });
    }
    if (repeats.some((row) => row.artifact_sha256 !== resolved.artifact.sha256
      || row.artifact_size_bytes !== resolved.artifact.size_bytes)) {
      throw new Error("worker artifact identity differs from the paired manifest.");
    }
    if (new Set(repeats.map((row) => row.analysis_sha256)).size !== 1) {
      throw new Error(`analysis digest changed across ${args.repeat} isolated runs.`);
    }
    const observed = repeats[0];
    if (observed.quality.non_finite_paths.length
      || observed.quality.phantom_reference_count
      || !observed.quality.op_index_unique) {
      throw new Error("analyzer quality gate failed.");
    }
    rows.push({
      ...observed,
      architecture: resolved.artifact.architecture,
      variant: resolved.artifact.variant,
      converter_generation: resolved.artifact.converter_generation,
      quantization_recipe: resolved.artifact.quantization_recipe || null,
      repository_id: resolved.artifact.repository_id,
      revision: resolved.artifact.revision,
      model_card_blob_sha1: resolved.artifact.model_card_blob_sha1,
      license_status: resolved.artifact.license_status,
      repeat_count: args.repeat,
      deterministic: true,
      status: "passed",
    });
    console.log(` ${observed.artifact_class}; ${observed.quant_research_coverage.assessed}/15 labs`);
  } catch (error) {
    rows.push({
      id: resolved.artifact.id,
      architecture: resolved.artifact.architecture,
      variant: resolved.artifact.variant,
      status: "failed",
      error: String(error?.message || error),
    });
    console.log(` failed: ${error?.message || error}`);
  }
}

const result = {
  schema: "deepbom.paired_modern_comparator_sweep.v1",
  corpus_id: manifest.corpus_id,
  manifest_sha256: sha256Text(await readFile(args.manifest, "utf8")),
  completed_at: new Date().toISOString(),
  repeat_count: args.repeat,
  passed_artifact_count: rows.filter((row) => row.status === "passed").length,
  failed_artifact_count: rows.filter((row) => row.status === "failed").length,
  artifact_class_counts: countBy(rows.filter((row) => row.status === "passed"), (row) => row.artifact_class),
  rows,
};
await writeFile(path.join(outputDir, "paired-modern-sweep.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
await writeFile(path.join(outputDir, "paired-modern-models.csv"), modelCsv(rows), "utf8");
await writeFile(path.join(outputDir, "paired-modern-labs.csv"), labCsv(rows), "utf8");
console.log(`Wrote ${path.join(outputDir, "paired-modern-sweep.json")}`);
if (result.failed_artifact_count) process.exitCode = 1;

function modelCsv(rowsToWrite) {
  return csvText(rowsToWrite.map((row) => ({
    ...row,
    assessed_labs: row.quant_research_coverage?.assessed,
    applicable_labs: row.quant_research_coverage?.artifact_applicable,
    quantized_weight_output_channels: row.denominators?.quantized_weight_output_channel_count,
    exact_zero_kernel_slices: row.observed_signals?.exact_zero_kernel_slice_count,
    maximum_bias_int32_ratio: row.observed_signals?.maximum_bias_int32_ratio,
    bias_half_range_exceedance_channels:
      row.observed_signals?.bias_half_range_exceedance_channel_count,
    bias_half_range_guard_adjacent_channels:
      row.observed_signals?.bias_half_range_guard_adjacent_channel_count,
    bias_half_range_material_exceedance_channels:
      row.observed_signals?.bias_half_range_material_exceedance_channel_count,
    exact_zero_bias_half_range_exceedance_channels:
      row.observed_signals?.exact_zero_bias_half_range_exceedance_channel_count,
    dual_mode_constant_output_channels: row.observed_signals?.dual_mode_constant_output_channel_count,
    near_zero_representable_channels: row.observed_signals?.near_zero_representable_channel_count,
  })), [
    "id", "architecture", "variant", "converter_generation", "quantization_recipe",
    "status", "artifact_class", "quantization_classification", "operator_count",
    "tensor_count", "total_macs", "applicable_labs", "assessed_labs",
    "quantized_weight_output_channels", "exact_zero_kernel_slices",
    "maximum_bias_int32_ratio", "bias_half_range_exceedance_channels",
    "bias_half_range_guard_adjacent_channels",
    "bias_half_range_material_exceedance_channels",
    "exact_zero_bias_half_range_exceedance_channels",
    "dual_mode_constant_output_channels", "near_zero_representable_channels",
    "license_status", "deterministic", "analysis_sha256", "error",
  ]);
}

function labCsv(rowsToWrite) {
  const flattened = rowsToWrite.flatMap((row) => (row.quant_research_coverage?.labs || []).map((lab) => ({
    artifact_id: row.id,
    architecture: row.architecture,
    variant: row.variant,
    artifact_class: row.artifact_class,
    ...lab,
  })));
  return csvText(flattened, [
    "artifact_id", "architecture", "variant", "artifact_class", "id", "status",
    "class_supported", "artifact_applicable", "reason_code", "candidate_count",
    "assessed_count", "unassessed_count", "assessed_channel_count",
    "unassessed_reason_counts", "reason",
  ]);
}

function parseArgs(argv) {
  const output = {
    manifest: GOOGLE_MODERN_COMPARATOR_PATH,
    cacheDir: "",
    outputDir: "",
    offline: false,
    repeat: 2,
    timeoutMs: 20 * 60_000,
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
