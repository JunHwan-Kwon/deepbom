import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";

import {
  GOOGLE_LEGACY_MANIFEST_PATH,
  fileIdentity,
  sha256Text,
} from "./google-legacy-corpus-lib.mjs";
import { GOOGLE_MODERN_COMPARATOR_PATH } from "./google-modern-comparator-lib.mjs";

const legacyPath = "reports/google-legacy-hosted-models-2026-07-29/google-legacy-sweep.json";
const modernPath = "reports/litert-modern-paired-comparators-2026-07-29/paired-modern-sweep.json";
const outputPath = "corpus/google_legacy/measurement-baseline.v1.json.gz";
const legacy = JSON.parse(await readFile(legacyPath, "utf8"));
const modern = JSON.parse(await readFile(modernPath, "utf8"));
if (legacy.failed_artifact_count || modern.failed_artifact_count
  || legacy.repeat_count < 2 || modern.repeat_count < 2) {
  throw new Error("A two-repeat, zero-failure legacy and modern sweep is required.");
}
const wasm = await fileIdentity("pkg/tflite_wasm_audit_bg.wasm");
if (!wasm) throw new Error("Analyzer WASM is unavailable.");
const git = spawnSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" });
if (git.status !== 0) throw new Error("Cannot resolve git HEAD.");
const dirty = spawnSync("git", ["status", "--porcelain"], { encoding: "utf8" });
if (dirty.status !== 0) throw new Error("Cannot inspect working-tree status.");
if (dirty.stdout.trim()) {
  throw new Error("Refusing to publish a measurement baseline from a dirty working tree.");
}

const result = {
  schema: "deepbom.google_converter_measurement_baseline.v1",
  generated_at: new Date().toISOString(),
  analyzer: {
    git_commit: git.stdout.trim(),
    working_tree_dirty: false,
    wasm_size_bytes: wasm.size,
    wasm_sha256: wasm.sha256,
    identification_policy: "The clean Git commit and WASM content hash jointly identify the executed analyzer.",
  },
  inputs: {
    legacy_manifest_sha256: sha256Text(await readFile(GOOGLE_LEGACY_MANIFEST_PATH, "utf8")),
    modern_manifest_sha256: sha256Text(await readFile(GOOGLE_MODERN_COMPARATOR_PATH, "utf8")),
    legacy_repeat_count: legacy.repeat_count,
    modern_repeat_count: modern.repeat_count,
  },
  denominator_policy: "Artifact class and lab applicability define denominators. not_applicable and not_assessed are not defect absence. Channel counts remain lab-specific.",
  interpretation_boundary: "Exact-zero stored kernel slices, near-zero representable channels, and dual-mode constant output channels are distinct signals. Converter causality requires matched source checkpoint and recipe evidence.",
  rows: [
    ...legacy.rows.map((row) => compactRow("legacy_google_hosted", row)),
    ...modern.rows.map((row) => compactRow("modern_paired", row)),
  ],
};
await writeFile(outputPath, gzipSync(Buffer.from(`${JSON.stringify(result, null, 2)}\n`), { level: 9, mtime: 0 }));
console.log(`Wrote ${outputPath}`);

function compactRow(corpus, row) {
  return {
    corpus,
    id: row.id,
    architecture: row.architecture || row.model || null,
    variant: row.variant || row.published_precision || null,
    cohort: row.cohort || row.converter_generation || null,
    artifact_sha256: row.artifact_sha256,
    artifact_size_bytes: row.artifact_size_bytes,
    analysis_sha256: row.analysis_sha256,
    deterministic: row.deterministic,
    artifact_class: row.artifact_class,
    quantization_classification: row.quantization_classification,
    operator_count: row.operator_count,
    tensor_count: row.tensor_count,
    total_macs: row.total_macs,
    lab_summary: {
      class_supported: row.quant_research_coverage?.class_supported,
      artifact_applicable: row.quant_research_coverage?.artifact_applicable,
      assessed: row.quant_research_coverage?.assessed,
      partial: row.quant_research_coverage?.partial,
      not_assessed: row.quant_research_coverage?.not_assessed,
      not_applicable: row.quant_research_coverage?.not_applicable,
    },
    assessed_channel_count_by_lab: row.denominators?.assessed_channel_count_by_lab || {},
    signals: {
      near_zero_decoded_kernel_slice_count: row.observed_signals?.near_zero_decoded_kernel_slice_count,
      exact_zero_kernel_slice_count: row.observed_signals?.exact_zero_kernel_slice_count,
      per_axis_kernel_channel_count: row.observed_signals?.per_axis_kernel_channel_count,
      near_zero_representable_channel_count: row.observed_signals?.near_zero_representable_channel_count,
      bias_scale_checked_channel_count: row.observed_signals?.bias_scale_checked_channel_count,
      bias_scale_mismatch_group_count: row.observed_signals?.bias_scale_mismatch_group_count,
      stored_bias_channel_count: row.observed_signals?.stored_bias_channel_count,
      maximum_bias_int32_ratio: row.observed_signals?.maximum_bias_int32_ratio,
      bias_half_range_exceedance_channel_count:
        row.observed_signals?.bias_half_range_exceedance_channel_count,
      bias_half_range_guard_adjacent_channel_count:
        row.observed_signals?.bias_half_range_guard_adjacent_channel_count,
      bias_half_range_material_exceedance_channel_count:
        row.observed_signals?.bias_half_range_material_exceedance_channel_count,
      exact_zero_bias_half_range_exceedance_channel_count:
        row.observed_signals?.exact_zero_bias_half_range_exceedance_channel_count,
      exact_zero_bias_half_range_material_exceedance_channel_count:
        row.observed_signals?.exact_zero_bias_half_range_material_exceedance_channel_count,
      accumulator_overflow_channel_count: row.observed_signals?.accumulator_overflow_channel_count,
      dual_mode_constant_output_channel_count: row.observed_signals?.dual_mode_constant_output_channel_count,
      variable_accumulator_constant_output_channel_count: row.observed_signals?.variable_accumulator_constant_output_channel_count,
      rounding_divergent_channel_count: row.observed_signals?.rounding_divergent_channel_count,
    },
    fused_activation_histogram: row.observed_signals?.fused_activation_histogram || {},
  };
}
