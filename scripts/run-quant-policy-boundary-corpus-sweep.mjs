import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  curatedMicroCacheDir,
  ensureCuratedMicroArtifact,
  readCuratedMicroCorpus,
} from "./curated-micro-corpus-lib.mjs";
import {
  ensureGoogleModernComparator,
  loadResolvedGoogleModernComparators,
  readGoogleModernComparators,
} from "./google-modern-comparator-lib.mjs";
import {
  countBy,
  csvText,
  ensureGoogleLegacyArtifact,
  googleLegacyCacheDir,
  readGoogleLegacyManifest,
  sha256Text,
} from "./google-legacy-corpus-lib.mjs";
import { hfCorpusCacheDir } from "./huggingface-community-corpus-lib.mjs";
import {
  corpusCacheDir,
  ensureCorpusModel,
  readCorpusManifest,
} from "./public-model-corpus-lib.mjs";
import {
  buildInterfaceContractCorpusSummary,
} from "./interface-quantization-contracts.mjs";

const MANIFEST_PATH = "corpus/quant_policy/manifest.v1.json";
const args = parseArgs(process.argv.slice(2));
const manifest = JSON.parse(await readFile(args.manifest, "utf8"));
if (manifest.schema !== "deepbom.quant_policy_boundary_corpus.v1") {
  throw new Error(`Unsupported quant-policy corpus schema: ${manifest.schema || "(missing)"}.`);
}
const analyzerArtifactIdentity = await readAnalyzerArtifactIdentity();

const targets = await resolvePublicTargets(manifest, args.offline);
if (targets.length !== manifest.public_artifact_count) {
  throw new Error(`Expected ${manifest.public_artifact_count} public targets, resolved ${targets.length}.`);
}
const anchor = await resolveOptionalAnchor(manifest, args.caseStudyPath);
if (anchor) targets.push(anchor);

const outputDir = path.resolve(args.outputDir || path.join("reports", manifest.corpus_id));
await mkdir(outputDir, { recursive: true });
const rows = [];
const startedAt = new Date().toISOString();
for (let index = 0; index < targets.length; index += 1) {
  const target = targets[index];
  process.stdout.write(`[${index + 1}/${targets.length}] ${target.qualified_id}: cache`);
  try {
    const cached = await target.ensure();
    process.stdout.write(cached.downloaded ? " downloaded; analyze" : " verified; analyze");
    const repeats = [];
    for (let repeat = 0; repeat < args.repeat; repeat += 1) {
      const resultPath = path.join(
        outputDir,
        `.${safeFilename(target.qualified_id)}.repeat-${repeat + 1}.json`,
      );
      const child = spawnSync(process.execPath, [
        "scripts/run-google-legacy-corpus-sweep.mjs",
        "--worker",
        "--artifact-path", cached.filename,
        "--artifact-id", target.qualified_id,
        "--target", manifest.target_profile_id,
        "--result", resultPath,
      ], {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: args.timeoutMs,
        maxBuffer: 16 * 1024 * 1024,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) {
        throw new Error(child.stderr.trim() || `worker exited ${child.status}.`);
      }
      repeats.push(JSON.parse(await readFile(resultPath, "utf8")));
      await rm(resultPath, { force: true });
    }
    validateRepeats(target, repeats, args.repeat, analyzerArtifactIdentity);
    const observed = repeats[0];
    const workflow = classifyWorkflow(target, observed);
    const kernelGranularity = classifyKernelGranularity(observed);
    rows.push({
      ...observed,
      id: target.id,
      qualified_id: target.qualified_id,
      subcohort_id: target.subcohort_id,
      source_manifest: target.source_manifest,
      public_corpus_member: target.public_corpus_member,
      workflow_class: workflow.classification,
      workflow_evidence_class: workflow.evidence_class,
      workflow_evidence: workflow.evidence,
      kernel_quantization_granularity: kernelGranularity,
      policy_stratum: `${workflow.classification}/${kernelGranularity}`,
      repeat_count: args.repeat,
      deterministic: true,
      status: "passed",
    });
    console.log(
      ` ${observed.artifact_class}; ${workflow.classification}; ${kernelGranularity}; `
      + `${observed.observed_signals.bias_half_range_exceedance_channel_count || 0} strict / `
      + `${observed.observed_signals.bias_half_range_material_exceedance_channel_count || 0} material / `
      + `${observed.observed_signals.accumulator_overflow_channel_count || 0} full-domain INT32 exceedance`,
    );
  } catch (error) {
    rows.push({
      id: target.id,
      qualified_id: target.qualified_id,
      subcohort_id: target.subcohort_id,
      public_corpus_member: target.public_corpus_member,
      status: "failed",
      error: String(error?.message || error),
    });
    console.log(` failed: ${error?.message || error}`);
  }
}

const passed = rows.filter((row) => row.status === "passed");
const publicRows = passed.filter((row) => row.public_corpus_member);
const interfaceContractSummary = buildInterfaceContractCorpusSummary(publicRows);
const completedAnalyzerArtifactIdentity = await readAnalyzerArtifactIdentity();
if (JSON.stringify(completedAnalyzerArtifactIdentity) !== JSON.stringify(analyzerArtifactIdentity)) {
  throw new Error("Analyzer artifacts changed during the corpus sweep; no mixed-build result was written.");
}
const result = {
  schema: "deepbom.quant_policy_boundary_corpus_sweep.v1.1",
  corpus_id: manifest.corpus_id,
  manifest_sha256: sha256Text(await readFile(args.manifest, "utf8")),
  analysis_target_profile_id: manifest.target_profile_id,
  analyzer_artifact_identity: analyzerArtifactIdentity,
  started_at: startedAt,
  completed_at: new Date().toISOString(),
  repeat_count: args.repeat,
  requested_public_artifact_count: manifest.public_artifact_count,
  passed_public_artifact_count: publicRows.length,
  failed_public_artifact_count: rows.filter(
    (row) => row.public_corpus_member && row.status === "failed",
  ).length,
  case_study_anchor_status: anchor
    ? rows.find((row) => !row.public_corpus_member)?.status || "failed"
    : "not_provided",
  artifact_class_counts: countBy(publicRows, (row) => row.artifact_class),
  workflow_class_counts: countBy(publicRows, (row) => row.workflow_class),
  workflow_evidence_class_counts: countBy(publicRows, (row) => row.workflow_evidence_class),
  kernel_quantization_granularity_counts: countBy(
    publicRows,
    (row) => row.kernel_quantization_granularity,
  ),
  policy_stratum_counts: countBy(publicRows, (row) => row.policy_stratum),
  interface_quantization_contract_summary: interfaceContractSummary,
  primary_outcomes: {
    public_artifacts_with_bias_half_range_exceedance: publicRows.filter(
      (row) => (row.observed_signals?.bias_half_range_exceedance_channel_count || 0) > 0,
    ).length,
    public_bias_half_range_exceedance_channels: sum(
      publicRows,
      (row) => row.observed_signals?.bias_half_range_exceedance_channel_count,
    ),
    public_bias_half_range_guard_adjacent_channels: sum(
      publicRows,
      (row) => row.observed_signals?.bias_half_range_guard_adjacent_channel_count,
    ),
    public_artifacts_with_material_bias_half_range_exceedance: publicRows.filter(
      (row) => (row.observed_signals?.bias_half_range_material_exceedance_channel_count || 0) > 0,
    ).length,
    public_material_bias_half_range_exceedance_channels: sum(
      publicRows,
      (row) => row.observed_signals?.bias_half_range_material_exceedance_channel_count,
    ),
    public_exact_zero_kernel_channels: sum(
      publicRows,
      (row) => row.observed_signals?.accumulator_exact_zero_kernel_channel_count,
    ),
    public_exact_zero_bias_half_range_exceedance_channels: sum(
      publicRows,
      (row) => row.observed_signals?.exact_zero_bias_half_range_exceedance_channel_count,
    ),
    public_exact_zero_bias_half_range_material_exceedance_channels: sum(
      publicRows,
      (row) => row.observed_signals
        ?.exact_zero_bias_half_range_material_exceedance_channel_count,
    ),
    public_artifacts_with_full_code_domain_int32_envelope_exceedance: publicRows.filter(
      (row) => (row.observed_signals?.accumulator_overflow_channel_count || 0) > 0,
    ).length,
    public_full_code_domain_int32_envelope_exceedance_channels: sum(
      publicRows,
      (row) => row.observed_signals?.accumulator_overflow_channel_count,
    ),
  },
  denominator_policy: "Report artifact and channel denominators by workflow-evidence and kernel-granularity stratum. Unknown workflow provenance is not relabeled as PTQ. Not-assessed and not-applicable labs are not defect-negative observations.",
  interpretation_boundary: manifest.interpretation_boundary,
  rows,
};

await writeFile(
  path.join(outputDir, "quant-policy-boundary-sweep.json"),
  `${JSON.stringify(result, null, 2)}\n`,
  "utf8",
);
await writeFile(
  path.join(outputDir, "quant-policy-boundary-models.csv"),
  modelCsv(rows),
  "utf8",
);
console.log(`Wrote ${path.join(outputDir, "quant-policy-boundary-sweep.json")}`);
if (result.failed_public_artifact_count || result.case_study_anchor_status === "failed") {
  process.exitCode = 1;
}

async function resolvePublicTargets(rootManifest, offline) {
  const resolved = [];
  for (const cohort of rootManifest.subcohorts) {
    if (cohort.kind === "public_manifest") {
      const source = await readCorpusManifest(cohort.manifest);
      const cache = corpusCacheDir();
      for (const artifact of source.models) {
        resolved.push(targetRecord(cohort, artifact, {
          sha256: artifact.sha256,
          size_bytes: artifact.size_bytes,
          declared_workflow: "unknown",
          workflow_evidence_class: "NOT_DECLARED",
          workflow_evidence: "No workflow label is inferred from filename or precision.",
          ensure: async () => {
            const cached = await ensureCorpusModel(artifact, cache, { offline });
            return { filename: cached.filename, downloaded: cached.downloaded };
          },
        }));
      }
    } else if (cohort.kind === "curated_micro_manifest") {
      const source = await readCuratedMicroCorpus(cohort.manifest);
      const cache = curatedMicroCacheDir();
      for (const artifact of source.artifacts) {
        resolved.push(targetRecord(cohort, artifact, {
          sha256: artifact.sha256,
          size_bytes: artifact.size_bytes,
          declared_workflow: "ptq",
          workflow_evidence_class: "SOURCE_BACKED_EXTERNAL_DECLARATION",
          workflow_evidence: source.sources.find(
            (candidate) => candidate.id === artifact.source_id,
          )?.quantization_evidence || "Official repository declares post-training quantization.",
          ensure: async () => {
            const cached = await ensureCuratedMicroArtifact(source, artifact, cache, { offline });
            return { filename: cached.filename, downloaded: cached.downloaded };
          },
        }));
      }
    } else if (cohort.kind === "google_legacy_manifest") {
      const source = await readGoogleLegacyManifest(cohort.manifest);
      const cache = googleLegacyCacheDir();
      for (const artifact of source.artifacts.filter(
        (candidate) => candidate.cohort === "legacy_quantized",
      )) {
        const officialQat = new Set(["inception-v3-quant", "ssd-mobilenet-v1-quant"])
          .has(artifact.id);
        resolved.push(targetRecord(cohort, artifact, {
          sha256: artifact.member.sha256,
          size_bytes: artifact.member.size_bytes,
          declared_workflow: officialQat ? "qat" : "unknown",
          workflow_evidence_class: officialQat
            ? "SOURCE_BACKED_EXTERNAL_DECLARATION"
            : "NOT_DECLARED",
          workflow_evidence: officialQat
            ? "Official TensorFlow publication identifies this hosted artifact as QAT."
            : artifact.converter_evidence,
          ensure: async () => {
            const cached = await ensureGoogleLegacyArtifact(artifact, cache, { offline });
            return { filename: cached.modelPath, downloaded: cached.downloaded };
          },
        }));
      }
    } else if (cohort.kind === "huggingface_pinned_manifest") {
      const source = await readGoogleModernComparators(cohort.manifest);
      const cache = hfCorpusCacheDir();
      for (const row of await loadResolvedGoogleModernComparators(source)) {
        resolved.push(targetRecord(cohort, row.artifact, {
          sha256: row.artifact.sha256,
          size_bytes: row.artifact.size_bytes,
          declared_workflow: "unknown",
          workflow_evidence_class: "NOT_DECLARED",
          workflow_evidence: "Pinned model card declares static W8A8 channelwise quantization but does not declare QAT or PTQ.",
          ensure: async () => {
            const cached = await ensureGoogleModernComparator(row, cache, { offline });
            return { filename: cached.filename, downloaded: cached.downloaded };
          },
        }));
      }
    } else {
      throw new Error(`Unsupported subcohort kind: ${cohort.kind}.`);
    }
  }
  return resolved;
}

function targetRecord(cohort, artifact, details) {
  return {
    id: artifact.id,
    qualified_id: `${cohort.id}/${artifact.id}`,
    subcohort_id: cohort.id,
    source_manifest: cohort.manifest,
    public_corpus_member: true,
    ...details,
  };
}

async function resolveOptionalAnchor(rootManifest, explicitPath) {
  const descriptor = rootManifest.case_study_anchors[0];
  const filename = explicitPath || process.env[descriptor.local_path_env];
  if (!filename) return null;
  const identity = await fileIdentity(filename);
  if (!identity || identity.sha256 !== descriptor.sha256) {
    throw new Error(`${descriptor.id}: local case-study identity does not match the pinned SHA-256.`);
  }
  return {
    id: descriptor.id,
    qualified_id: `case-study/${descriptor.id}`,
    subcohort_id: "case-study-anchor",
    source_manifest: null,
    public_corpus_member: false,
    sha256: descriptor.sha256,
    size_bytes: identity.size,
    declared_workflow: "qat",
    workflow_evidence_class: descriptor.expected_provenance.evidence_class,
    workflow_evidence: "Pinned local case study expects embedded TensorFlow 2.14 conversion metadata with QAT mode 2000.",
    ensure: async () => ({ filename: path.resolve(filename), downloaded: false }),
  };
}

function validateRepeats(target, repeats, expectedCount, analyzerIdentity) {
  if (repeats.length !== expectedCount) throw new Error("isolated repeat count is incomplete.");
  for (const row of repeats) {
    if (row.artifact_sha256 !== target.sha256 || row.artifact_size_bytes !== target.size_bytes) {
      throw new Error("worker artifact identity differs from the pinned manifest.");
    }
    if (JSON.stringify(row.analyzer_artifact_identity) !== JSON.stringify(analyzerIdentity)) {
      throw new Error("worker analyzer identity differs from the sweep-bound analyzer artifacts.");
    }
    if (row.quality.non_finite_paths.length
      || row.quality.phantom_reference_count
      || !row.quality.op_index_unique
      || !row.quality.size_breakdown_conservation_valid) {
      throw new Error("analyzer quality gate failed.");
    }
  }
  if (new Set(repeats.map((row) => row.analysis_sha256)).size !== 1) {
    throw new Error(`analysis digest changed across ${expectedCount} isolated runs.`);
  }
  if (new Set(repeats.map((row) => row.interface_contracts?.ledger_sha256)).size !== 1
    || !repeats[0].interface_contracts?.ledger_sha256) {
    throw new Error(`interface-contract ledger changed across ${expectedCount} isolated runs.`);
  }
}

async function readAnalyzerArtifactIdentity() {
  const identity = {
    wasm: await fileIdentity("pkg/tflite_wasm_audit_bg.wasm"),
    javascript: await fileIdentity("pkg/tflite_wasm_audit.js"),
  };
  if (!identity.wasm || !identity.javascript) {
    throw new Error("Analyzer WASM and JavaScript artifacts must exist before a corpus sweep.");
  }
  return identity;
}

function classifyWorkflow(target, observed) {
  const codes = observed.conversion_metadata?.optimization_mode_codes || [];
  if (codes.includes(2000)) {
    return {
      classification: "qat",
      evidence_class: "OBSERVED_EMBEDDED_CONVERSION_METADATA",
      evidence: `CONVERSION_METADATA optimization modes: ${codes.join(", ")}.`,
    };
  }
  if (codes.some((code) => code >= 1001 && code <= 1004)) {
    return {
      classification: "ptq",
      evidence_class: "OBSERVED_EMBEDDED_CONVERSION_METADATA",
      evidence: `CONVERSION_METADATA optimization modes: ${codes.join(", ")}.`,
    };
  }
  return {
    classification: target.declared_workflow,
    evidence_class: target.workflow_evidence_class,
    evidence: target.workflow_evidence,
  };
}

function classifyKernelGranularity(row) {
  if (row.artifact_class === "float") return "not_quantized";
  if ((row.observed_signals?.per_axis_kernel_tensor_count || 0) > 0) {
    return "per_axis_kernel_observed";
  }
  if ((row.observed_signals?.stored_bias_channel_count || 0) > 0) {
    return "per_tensor_kernel_observed";
  }
  return "not_assessed";
}

function sum(rows, selector) {
  return rows.reduce((total, row) => total + (Number(selector(row)) || 0), 0);
}

async function fileIdentity(filename) {
  try {
    const info = await stat(filename);
    if (!info.isFile()) return null;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(filename)) hash.update(chunk);
    return { size: info.size, sha256: hash.digest("hex") };
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function modelCsv(rows) {
  return csvText(rows.map((row) => ({
    ...row,
    optimization_mode_codes: row.conversion_metadata?.optimization_mode_codes,
    per_axis_kernel_tensors: row.observed_signals?.per_axis_kernel_tensor_count,
    per_axis_kernel_channels: row.observed_signals?.per_axis_kernel_channel_count,
    stored_bias_channels: row.observed_signals?.stored_bias_channel_count,
    maximum_bias_int32_ratio: row.observed_signals?.maximum_bias_int32_ratio,
    half_range_exceedance_channels:
      row.observed_signals?.bias_half_range_exceedance_channel_count,
    half_range_guard_adjacent_channels:
      row.observed_signals?.bias_half_range_guard_adjacent_channel_count,
    half_range_material_exceedance_channels:
      row.observed_signals?.bias_half_range_material_exceedance_channel_count,
    exact_zero_kernel_channels:
      row.observed_signals?.accumulator_exact_zero_kernel_channel_count,
    exact_zero_half_range_exceedance_channels:
      row.observed_signals?.exact_zero_bias_half_range_exceedance_channel_count,
    exact_zero_half_range_material_exceedance_channels:
      row.observed_signals?.exact_zero_bias_half_range_material_exceedance_channel_count,
    full_code_domain_int32_envelope_exceedance_channels:
      row.observed_signals?.accumulator_overflow_channel_count,
    interface_parameter_count: row.interface_contracts?.parameter_count,
    interface_boundary_status:
      row.interface_contracts?.boundary_contract?.status,
    input_boundary_status:
      row.interface_contracts?.boundary_contract?.inputs?.status,
    output_boundary_status:
      row.interface_contracts?.boundary_contract?.outputs?.status,
    float32_interface_parameter_count:
      row.interface_contracts?.boundary_contract?.float32_parameter_count,
    quantized_interface_parameter_count:
      row.interface_contracts?.quantized_parameter_count,
    per_tensor_interface_parameter_count:
      row.interface_contracts?.per_tensor_parameter_count,
    per_axis_interface_parameter_count:
      row.interface_contracts?.per_axis_parameter_count,
    distinct_interface_quantization_contract_count:
      row.interface_contracts?.distinct_complete_quantization_contract_count,
    multiple_interface_quantization_contracts:
      row.interface_contracts?.multiple_complete_quantization_contracts_within_artifact,
    interface_contract_ledger_sha256:
      row.interface_contracts?.ledger_sha256,
  })), [
    "qualified_id", "subcohort_id", "public_corpus_member", "status", "artifact_sha256",
    "artifact_size_bytes", "artifact_class", "quantization_classification",
    "workflow_class", "workflow_evidence_class", "kernel_quantization_granularity",
    "policy_stratum", "optimization_mode_codes", "operator_count", "tensor_count",
    "per_axis_kernel_tensors", "per_axis_kernel_channels", "stored_bias_channels",
    "maximum_bias_int32_ratio", "half_range_exceedance_channels",
    "half_range_guard_adjacent_channels", "half_range_material_exceedance_channels",
    "exact_zero_kernel_channels", "exact_zero_half_range_exceedance_channels",
    "exact_zero_half_range_material_exceedance_channels",
    "full_code_domain_int32_envelope_exceedance_channels",
    "interface_parameter_count", "interface_boundary_status", "input_boundary_status",
    "output_boundary_status", "float32_interface_parameter_count",
    "quantized_interface_parameter_count",
    "per_tensor_interface_parameter_count", "per_axis_interface_parameter_count",
    "distinct_interface_quantization_contract_count",
    "multiple_interface_quantization_contracts",
    "interface_contract_ledger_sha256",
    "repeat_count", "deterministic", "analysis_sha256", "error",
  ]);
}

function safeFilename(value) {
  return value.replaceAll(/[^A-Za-z0-9_.-]/g, "__");
}

function parseArgs(argv) {
  const output = {
    manifest: MANIFEST_PATH,
    outputDir: "",
    caseStudyPath: "",
    offline: false,
    repeat: 2,
    timeoutMs: 20 * 60_000,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--offline") output.offline = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--case-study") output.caseStudyPath = required(argv, ++index, key);
    else if (key === "--repeat") {
      output.repeat = boundedInteger(required(argv, ++index, key), 1, 5, key);
    } else if (key === "--timeout-ms") {
      output.timeoutMs = boundedInteger(required(argv, ++index, key), 1_000, 3_600_000, key);
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
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
