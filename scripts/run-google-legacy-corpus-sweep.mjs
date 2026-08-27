import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  GOOGLE_LEGACY_MANIFEST_PATH,
  GOOGLE_LEGACY_SWEEP_SCHEMA,
  countBy,
  csvText,
  ensureGoogleLegacyArtifact,
  googleLegacyCacheDir,
  readGoogleLegacyManifest,
  sha256Bytes,
  sha256Text,
} from "./google-legacy-corpus-lib.mjs";
import { summarizeInterfaceContracts } from "./interface-quantization-contracts.mjs";

const args = parseArgs(process.argv.slice(2));
if (args.worker) {
  await runWorker(args);
} else {
  await runSweep(args);
}

async function runSweep(options) {
  const manifest = await readGoogleLegacyManifest(options.manifest);
  const cacheDir = path.resolve(options.cacheDir || googleLegacyCacheDir());
  const outputDir = path.resolve(options.outputDir || path.join("reports", manifest.corpus_id));
  const selected = options.artifactIds.length
    ? manifest.artifacts.filter((artifact) => options.artifactIds.includes(artifact.id))
    : manifest.artifacts;
  if (selected.length !== (options.artifactIds.length || manifest.artifacts.length)) {
    throw new Error("One or more requested Google legacy artifact ids are absent from the manifest.");
  }
  await mkdir(outputDir, { recursive: true });
  const rows = [];
  const startedAt = new Date().toISOString();
  for (let artifactIndex = 0; artifactIndex < selected.length; artifactIndex += 1) {
    const artifact = selected[artifactIndex];
    process.stdout.write(`[${artifactIndex + 1}/${selected.length}] ${artifact.id}: cache`);
    try {
      const cached = await ensureGoogleLegacyArtifact(artifact, cacheDir, {
        offline: options.offline,
        onProgress: ({ received, total }) => {
          if (process.stdout.isTTY) {
            process.stdout.write(`\r[${artifactIndex + 1}/${selected.length}] ${artifact.id}: ${formatBytes(received)} / ${formatBytes(total)}`);
          }
        },
      });
      process.stdout.write(cached.downloaded ? " downloaded; analyze" : " verified; analyze");
      const repeats = [];
      for (let repeat = 0; repeat < options.repeat; repeat += 1) {
        const resultPath = path.join(outputDir, `.${artifact.id}.repeat-${repeat + 1}.json`);
        const child = spawnSync(process.execPath, [
          path.resolve(process.argv[1]),
          "--worker",
          "--artifact-path", cached.modelPath,
          "--artifact-id", artifact.id,
          "--target", manifest.target_profile_id,
          "--result", resultPath,
        ], {
          cwd: process.cwd(),
          encoding: "utf8",
          timeout: options.timeoutMs,
          maxBuffer: 16 * 1024 * 1024,
        });
        if (child.error) throw child.error;
        if (child.status !== 0) throw new Error(child.stderr.trim() || `worker exited ${child.status}`);
        repeats.push(JSON.parse(await readFile(resultPath, "utf8")));
        await rm(resultPath, { force: true });
      }
      validateRepeatedResults(artifact, repeats, options.repeat);
      const observed = repeats[0];
      validateBaseline(artifact, observed);
      rows.push({
        ...observed,
        model: artifact.model,
        cohort: artifact.cohort,
        converter_generation: artifact.converter_generation,
        converter_evidence: artifact.converter_evidence,
        architecture_year: artifact.architecture_year,
        artifact_release_year: artifact.artifact_release_year ?? null,
        task: artifact.task,
        published_precision: artifact.published_precision,
        repeat_count: options.repeat,
        deterministic: true,
        status: "passed",
      });
      console.log(` ${observed.operator_count} ops; ${observed.quant_research_coverage.assessed}/15 labs`);
    } catch (error) {
      rows.push({
        id: artifact.id,
        model: artifact.model,
        cohort: artifact.cohort,
        status: "failed",
        error: String(error?.message || error),
      });
      console.log(` failed: ${error?.message || error}`);
    }
  }
  const result = {
    schema: GOOGLE_LEGACY_SWEEP_SCHEMA,
    corpus_id: manifest.corpus_id,
    manifest_sha256: sha256Text(await readFile(options.manifest, "utf8")),
    analysis_target_profile_id: manifest.target_profile_id,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    repeat_count: options.repeat,
    requested_artifact_count: selected.length,
    passed_artifact_count: rows.filter((row) => row.status === "passed").length,
    failed_artifact_count: rows.filter((row) => row.status === "failed").length,
    deterministic_artifact_count: rows.filter((row) => row.deterministic).length,
    artifact_class_counts: countBy(rows.filter((row) => row.status === "passed"), (row) => row.artifact_class),
    quantization_classification_counts: countBy(
      rows.filter((row) => row.status === "passed"),
      (row) => row.quantization_classification,
    ),
    denominator_policy: "Use per-artifact class and per-lab applicability. Do not count not_applicable or not_assessed as defect absence. Channel denominators remain lab-specific.",
    rows,
  };
  await writeFile(path.join(outputDir, "google-legacy-sweep.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await writeFile(path.join(outputDir, "google-legacy-models.csv"), modelCsv(rows), "utf8");
  await writeFile(path.join(outputDir, "google-legacy-labs.csv"), labCsv(rows), "utf8");
  await writeFile(path.join(outputDir, "google-legacy-reasons.csv"), reasonCsv(rows), "utf8");
  console.log(`Wrote ${path.join(outputDir, "google-legacy-sweep.json")}`);
  if (result.failed_artifact_count) process.exitCode = 1;
}

async function runWorker(options) {
  if (!options.artifactPath || !options.artifactId || !options.result) {
    throw new Error("Worker requires --artifact-path, --artifact-id, and --result.");
  }
  const analyzerJavaScriptBefore = await readFile("pkg/tflite_wasm_audit.js");
  const { initSync, analyze_tflite_for_target } = await import("../pkg/tflite_wasm_audit.js");
  const { ensureQuantResearchCoverage } = await import("../web/lib/quant-research-applicability.js");
  const {
    buildBiasScaleCheck,
    buildRepresentableKernelChannelCheck,
  } = await import("../web/lib/quantization-contract-summary.js");
  const analyzerJavaScriptAfter = await readFile("pkg/tflite_wasm_audit.js");
  const javascriptSha256 = sha256Bytes(analyzerJavaScriptBefore);
  if (javascriptSha256 !== sha256Bytes(analyzerJavaScriptAfter)) {
    throw new Error("Analyzer JavaScript changed while the isolated worker was loading it.");
  }
  const wasmBytes = await readFile("pkg/tflite_wasm_audit_bg.wasm");
  const analyzerArtifactIdentity = {
    wasm: { size: wasmBytes.byteLength, sha256: sha256Bytes(wasmBytes) },
    javascript: {
      size: analyzerJavaScriptBefore.byteLength,
      sha256: javascriptSha256,
    },
  };
  initSync({ module: wasmBytes });
  const bytes = new Uint8Array(await readFile(options.artifactPath));
  const started = performance.now();
  const analysis = analyze_tflite_for_target(bytes, path.basename(options.artifactPath), options.target);
  const elapsedMs = performance.now() - started;
  const coverage = ensureQuantResearchCoverage(analysis);
  const biasScale = buildBiasScaleCheck(analysis);
  const representableChannels = buildRepresentableKernelChannelCheck(analysis);
  const serialized = JSON.stringify(analysis);
  const labs = coverage.labs.map((row) => detailedLabRow(analysis, row));
  const phantomReferences = phantomOpReferences(serialized, analysis.ops);
  const interfaceContracts = summarizeInterfaceContracts(analysis);
  const result = {
    schema: "deepbom.google_legacy_corpus_worker_result.v1",
    id: options.artifactId,
    format: analysis.format,
    artifact_sha256: sha256Bytes(bytes),
    artifact_size_bytes: bytes.byteLength,
    analyzer_artifact_identity: analyzerArtifactIdentity,
    analysis_sha256: sha256Text(serialized),
    analyzer_elapsed_ms: elapsedMs,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs ?? null,
    quantization_classification: analysis.quantization_status?.classification || "unknown",
    artifact_class: coverage.artifact_class,
    conversion_metadata: {
      status: analysis.metadata_presence?.conversion_metadata_status || "not_present",
      tensorflow_version: analysis.metadata_presence?.converter_tensorflow_version || null,
      api_version: analysis.metadata_presence?.converter_api_version ?? null,
      model_type: analysis.metadata_presence?.converter_model_type || null,
      optimization_mode_codes:
        analysis.metadata_presence?.converter_optimization_mode_codes || [],
      optimization_modes: analysis.metadata_presence?.converter_optimization_modes || [],
    },
    interface_contracts: interfaceContracts,
    quant_research_coverage: {
      lab_count: coverage.lab_count,
      class_supported: coverage.class_supported_lab_count,
      class_excluded: coverage.class_excluded_lab_count,
      artifact_applicable: coverage.artifact_applicable_lab_count,
      assessed: coverage.assessed_lab_count,
      partial: coverage.partial_lab_count,
      not_assessed: coverage.not_assessed_lab_count,
      not_applicable: coverage.not_applicable_lab_count,
      labs,
    },
    denominators: {
      decoded_weight_tensor_count: finiteOrNull(analysis.weight_integrity?.quantized_constant_tensors_scanned),
      sparse_constant_tensors_not_decoded: finiteOrNull(
        analysis.weight_integrity?.sparse_constant_tensors_not_decoded,
      ),
      constant_value_coverage_status:
        analysis.weight_integrity?.constant_value_coverage_status || "unknown",
      quantized_weight_output_channel_count: coverage.artifact_class === "float"
        ? null
        : finiteOrNull(analysis.weight_integrity?.output_channels_evaluated),
      assessed_channel_count_by_lab: Object.fromEntries(labs
        .filter((row) => row.assessed_channel_count !== null)
        .map((row) => [row.id, row.assessed_channel_count])),
    },
    observed_signals: {
      near_zero_decoded_kernel_slice_count: finiteOrNull(analysis.weight_integrity?.zero_kernel_slice_count),
      exact_zero_kernel_slice_count: finiteOrNull(analysis.weight_integrity?.exact_zero_kernel_slice_count),
      per_axis_kernel_tensor_count: finiteOrNull(representableChannels.assessed_kernel_tensors),
      per_axis_kernel_channel_count: finiteOrNull(representableChannels.assessed_channels),
      near_zero_representable_channel_count: finiteOrNull(representableChannels.flagged_channels),
      bias_scale_checked_channel_count: finiteOrNull(biasScale.checked_channels),
      bias_scale_mismatch_group_count: finiteOrNull(biasScale.mismatch_groups),
      stored_bias_channel_count: finiteOrNull(analysis.accumulator_atlas?.stored_bias_channel_count),
      maximum_bias_int32_ratio: finiteOrNull(analysis.accumulator_atlas?.maximum_bias_int32_ratio),
      bias_half_range_exceedance_channel_count: finiteOrNull(
        analysis.accumulator_atlas?.bias_half_range_exceedance_channel_count,
      ),
      bias_half_range_exceedance_op_count: finiteOrNull(
        analysis.accumulator_atlas?.bias_half_range_exceedance_op_count,
      ),
      bias_half_range_guard_adjacent_channel_count: finiteOrNull(
        analysis.accumulator_atlas?.bias_half_range_guard_adjacent_channel_count,
      ),
      bias_half_range_material_exceedance_channel_count: finiteOrNull(
        analysis.accumulator_atlas?.bias_half_range_material_exceedance_channel_count,
      ),
      bias_half_range_material_exceedance_op_count: finiteOrNull(
        analysis.accumulator_atlas?.bias_half_range_material_exceedance_op_count,
      ),
      accumulator_exact_zero_kernel_channel_count: finiteOrNull(
        analysis.accumulator_atlas?.exact_zero_kernel_channel_count,
      ),
      exact_zero_bias_half_range_exceedance_channel_count: finiteOrNull(
        analysis.accumulator_atlas?.exact_zero_bias_half_range_exceedance_channel_count,
      ),
      exact_zero_bias_half_range_material_exceedance_channel_count: finiteOrNull(
        analysis.accumulator_atlas
          ?.exact_zero_bias_half_range_material_exceedance_channel_count,
      ),
      accumulator_overflow_channel_count: finiteOrNull(analysis.accumulator_atlas?.int32_overflow_channel_count),
      dual_mode_constant_output_channel_count: finiteOrNull(analysis.channel_vitality?.dual_mode_constant_output_channel_count),
      variable_accumulator_constant_output_channel_count: finiteOrNull(
        analysis.channel_vitality?.nonconstant_accumulator_dual_mode_constant_channel_count,
      ),
      rounding_divergent_channel_count: finiteOrNull(analysis.rounding_equivalence?.divergent_channel_count),
      operator_histogram: countBy(analysis.ops || [], (op) => op.name),
      fused_activation_histogram: countBy(analysis.ops || [], (op) => op.fused_activation || "NONE"),
      quantization_state_histogram: countBy(analysis.ops || [], (op) => op.quantization_state || "none"),
    },
    finding_counts: countBy(analysis.findings || [], (finding) => finding.priority || finding.severity || "unknown"),
    quality: {
      non_finite_paths: findNonFinite(analysis),
      phantom_reference_count: phantomReferences.length,
      phantom_references: phantomReferences,
      op_index_unique: new Set((analysis.ops || []).map((op) => op.index)).size === (analysis.ops || []).length,
      size_breakdown_conservation_valid: Number(analysis.size_breakdown?.constant_bytes || 0)
        + Number(analysis.size_breakdown?.metadata_bytes || 0)
        + Number(analysis.size_breakdown?.structure_overhead_bytes || 0)
        === Number(analysis.file_size || analysis.file_size_bytes || bytes.byteLength),
    },
  };
  await writeFile(options.result, `${JSON.stringify(result, null, 2)}\n`, "utf8");
}

function detailedLabRow(analysis, coverageRow) {
  const evidence = coverageRow.evidence_key ? analysis?.[coverageRow.evidence_key] : null;
  const rows = evidenceRows(evidence);
  const unassessedRows = rows.filter((row) => {
    const status = String(row?.assessment_status || "").toLowerCase();
    return status && status !== "assessed" && row?.reason_code;
  });
  const assessedChannelCount = coverageRow.id === "weight_scale_integrity"
    ? coverageRow.status === "assessed"
      ? finiteOrNull(analysis.weight_integrity?.output_channels_evaluated)
      : null
    : finiteOrNull(evidence?.assessed_channel_count);
  return {
    id: coverageRow.id,
    status: coverageRow.status,
    class_supported: coverageRow.class_supported,
    artifact_applicable: coverageRow.artifact_applicable,
    reason_code: coverageRow.reason_code,
    reason: coverageRow.reason,
    candidate_count: firstFinite(evidence, [
      "candidate_op_count", "candidate_add_count", "candidate_source_op_count", "source_witness_count",
    ]),
    assessed_count: firstFinite(evidence, [
      "assessed_op_count", "assessed_add_count", "assessed_source_count", "assessed_witness_count",
    ]),
    unassessed_count: firstFinite(evidence, [
      "unassessed_op_count", "unassessed_add_count", "unassessed_source_op_count", "not_assessed_source_op_count",
    ]),
    assessed_channel_count: assessedChannelCount,
    unassessed_reason_counts: countBy(unassessedRows, (row) => row.reason_code || "UNSPECIFIED"),
  };
}

function evidenceRows(evidence) {
  for (const key of ["ops", "residual_adds", "sources", "candidates"]) {
    if (Array.isArray(evidence?.[key])) return evidence[key];
  }
  return [];
}

function firstFinite(object, keys) {
  for (const key of keys) {
    const value = Number(object?.[key]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function validateRepeatedResults(artifact, repeats, repeatCount) {
  if (repeats.length !== repeatCount) throw new Error(`${artifact.id}: isolated repeat count is incomplete.`);
  for (const row of repeats) {
    if (row.schema !== "deepbom.google_legacy_corpus_worker_result.v1") {
      throw new Error(`${artifact.id}: worker result schema is invalid.`);
    }
    if (row.artifact_sha256 !== artifact.member.sha256
      || row.artifact_size_bytes !== artifact.member.size_bytes) {
      throw new Error(`${artifact.id}: worker artifact identity does not match the manifest.`);
    }
    if (row.quality.non_finite_paths.length || row.quality.phantom_reference_count || !row.quality.op_index_unique) {
      throw new Error(`${artifact.id}: analyzer quality gate failed.`);
    }
  }
  if (new Set(repeats.map((row) => row.analysis_sha256)).size !== 1) {
    throw new Error(`${artifact.id}: analysis digest changed across ${repeatCount} isolated process runs.`);
  }
}

function validateBaseline(artifact, observed) {
  const baseline = artifact.baseline;
  const coverage = observed.quant_research_coverage;
  const pairs = [
    ["artifact_class", observed.artifact_class, baseline.artifact_class],
    ["quantization_classification", observed.quantization_classification, baseline.quantization_classification],
    ["operator_count", observed.operator_count, baseline.operator_count],
    ["tensor_count", observed.tensor_count, baseline.tensor_count],
    ["total_macs", observed.total_macs, baseline.total_macs],
    ["lab_count", coverage.lab_count, baseline.lab_count],
    ["class_supported_lab_count", coverage.class_supported, baseline.class_supported_lab_count],
    ["assessed_lab_count", coverage.assessed, baseline.assessed_lab_count],
    ["partial_lab_count", coverage.partial, baseline.partial_lab_count],
    ["not_assessed_lab_count", coverage.not_assessed, baseline.not_assessed_lab_count],
    ["not_applicable_lab_count", coverage.not_applicable, baseline.not_applicable_lab_count],
  ];
  const mismatch = pairs.find(([, actual, expected]) => actual !== expected);
  if (mismatch) {
    throw new Error(`${artifact.id}: baseline ${mismatch[0]} expected ${mismatch[2]}, observed ${mismatch[1]}.`);
  }
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

function findNonFinite(value, currentPath = "$", failures = []) {
  if (typeof value === "number" && !Number.isFinite(value)) failures.push(currentPath);
  else if (Array.isArray(value)) value.forEach((item, index) => findNonFinite(item, `${currentPath}[${index}]`, failures));
  else if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) findNonFinite(child, `${currentPath}.${key}`, failures);
  }
  return failures.slice(0, 100);
}

function modelCsv(rows) {
  const normalized = rows.map((row) => ({
    ...row,
    class_supported_labs: row.quant_research_coverage?.class_supported,
    applicable_labs: row.quant_research_coverage?.artifact_applicable,
    assessed_labs: row.quant_research_coverage?.assessed,
    partial_labs: row.quant_research_coverage?.partial,
    not_assessed_labs: row.quant_research_coverage?.not_assessed,
    not_applicable_labs: row.quant_research_coverage?.not_applicable,
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
    operator_histogram: row.observed_signals?.operator_histogram,
    fused_activation_histogram: row.observed_signals?.fused_activation_histogram,
  }));
  return csvText(normalized, [
    "id", "model", "cohort", "converter_generation", "task", "status",
    "artifact_class", "quantization_classification", "operator_count", "tensor_count", "total_macs",
    "class_supported_labs", "applicable_labs", "assessed_labs", "partial_labs",
    "not_assessed_labs", "not_applicable_labs", "quantized_weight_output_channels",
    "exact_zero_kernel_slices", "maximum_bias_int32_ratio",
    "bias_half_range_exceedance_channels",
    "bias_half_range_guard_adjacent_channels",
    "bias_half_range_material_exceedance_channels",
    "exact_zero_bias_half_range_exceedance_channels", "dual_mode_constant_output_channels",
    "near_zero_representable_channels", "operator_histogram", "fused_activation_histogram",
    "analyzer_elapsed_ms", "deterministic", "analysis_sha256", "error",
  ]);
}

function labCsv(rows) {
  const flattened = rows.flatMap((row) => (row.quant_research_coverage?.labs || []).map((lab) => ({
    artifact_id: row.id,
    cohort: row.cohort,
    artifact_class: row.artifact_class,
    ...lab,
  })));
  return csvText(flattened, [
    "artifact_id", "cohort", "artifact_class", "id", "status", "class_supported",
    "artifact_applicable", "reason_code", "candidate_count", "assessed_count",
    "unassessed_count", "assessed_channel_count", "unassessed_reason_counts", "reason",
  ]);
}

function reasonCsv(rows) {
  const flattened = [];
  for (const row of rows) {
    for (const lab of row.quant_research_coverage?.labs || []) {
      if (lab.status !== "assessed") {
        flattened.push({
          artifact_id: row.id,
          cohort: row.cohort,
          lab_id: lab.id,
          scope: "lab",
          reason_code: lab.reason_code,
          count: 1,
        });
      }
      for (const [reasonCode, count] of Object.entries(lab.unassessed_reason_counts || {})) {
        flattened.push({
          artifact_id: row.id,
          cohort: row.cohort,
          lab_id: lab.id,
          scope: "evidence_row",
          reason_code: reasonCode,
          count,
        });
      }
    }
  }
  return csvText(flattened, ["artifact_id", "cohort", "lab_id", "scope", "reason_code", "count"]);
}

function parseArgs(argv) {
  const output = {
    manifest: GOOGLE_LEGACY_MANIFEST_PATH,
    cacheDir: "",
    outputDir: "",
    offline: false,
    repeat: 2,
    timeoutMs: 20 * 60_000,
    artifactIds: [],
    worker: false,
    artifactPath: "",
    artifactId: "",
    target: "android_mid_a55",
    result: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--offline") output.offline = true;
    else if (key === "--worker") output.worker = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--repeat") output.repeat = boundedInteger(required(argv, ++index, key), 1, 5, key);
    else if (key === "--timeout-ms") output.timeoutMs = boundedInteger(required(argv, ++index, key), 1_000, 3_600_000, key);
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
    else if (key === "--artifact-path") output.artifactPath = required(argv, ++index, key);
    else if (key === "--artifact-id") output.artifactId = required(argv, ++index, key);
    else if (key === "--target") output.target = required(argv, ++index, key);
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
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${key} must be ${minimum}..${maximum}.`);
  }
  return parsed;
}

function formatBytes(value) {
  return `${(Number(value || 0) / 1024 ** 2).toFixed(2)} MiB`;
}
