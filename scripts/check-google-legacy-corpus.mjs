import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";

import {
  GOOGLE_LEGACY_COHORTS_PATH,
  GOOGLE_LEGACY_MANIFEST_PATH,
  normalizeMemberPath,
  readGoogleLegacyManifest,
} from "./google-legacy-corpus-lib.mjs";
import {
  loadResolvedGoogleModernComparators,
  readGoogleModernComparators,
} from "./google-modern-comparator-lib.mjs";

const manifest = await readGoogleLegacyManifest(GOOGLE_LEGACY_MANIFEST_PATH);
const cohorts = JSON.parse(await readFile(GOOGLE_LEGACY_COHORTS_PATH, "utf8"));
const packageJson = JSON.parse(await readFile("package.json", "utf8"));
const applicability = await readFile("web/lib/quant-research-applicability.js", "utf8");
const modern = await readGoogleModernComparators();
const modernResolved = await loadResolvedGoogleModernComparators(modern);
const measurementBaseline = JSON.parse(gunzipSync(await readFile(
  "corpus/google_legacy/measurement-baseline.v1.json.gz",
)).toString("utf8"));

assert.equal(manifest.artifact_count, 8);
assert.equal(manifest.quantized_artifact_count, 4);
assert.equal(manifest.float_control_count, 4);
assert.deepEqual(
  Object.fromEntries([...countBy(manifest.artifacts, (row) => row.baseline.artifact_class).entries()]),
  { float: 4, full_integer: 4 },
);
assert.deepEqual(
  Object.fromEntries([...countBy(manifest.artifacts, (row) => row.baseline.quantization_classification).entries()]),
  { full_integer: 3, integer_internal_float_io: 1, not_quantized_float: 4 },
);
assert.ok(manifest.corrected_proposals.some((row) => row.proposal.startsWith("MnasNet")
  && row.disposition === "retained_as_float_controls"));
assert.ok(manifest.corrected_proposals.some((row) => row.proposal === "SSD MobileNetV2 quant"
  && row.disposition === "pending"));

assert.equal(cohorts.schema, "deepbom.converter_cohorts.v1");
assert.equal(cohorts.cohorts.length, 4);
assert.equal(cohorts.cohorts.find((row) => row.id === "litert_torch")?.package?.version, "0.9.2");
assert.equal(
  cohorts.cohorts.find((row) => row.id === "litert_torch")?.package?.wheel_sha256,
  "e6688f08ded25dc4b8d7018260f3722b03cabc4a022c3f02cdfd9dadf500e3b3",
);
assert.equal(cohorts.cohorts.find((row) => row.id === "ai_edge_quantizer")?.package?.version, "0.8.0");
assert.equal(cohorts.runtime_validation.litert_js.project_pin, "@litertjs/core 2.5.2");
assert.equal(packageJson.dependencies["@litertjs/core"], "^2.5.2");
assert.match(applicability, /"mixed_integer"/);
assert.match(applicability, /source_quantization_classification/);
assert.equal(modern.artifact_count, 16);
assert.equal(modernResolved.length, 16);
assert.equal(new Set(modern.artifacts.map((row) => row.sha256)).size, 16);
assert.equal(modern.artifacts.filter((row) => row.variant === "static_wi8_ai8_channelwise").length, 3);
assert.equal(modern.artifacts.filter((row) => row.variant === "dynamic_wi8_afp32").length, 6);
assert.equal(measurementBaseline.schema, "deepbom.google_converter_measurement_baseline.v1");
assert.equal(measurementBaseline.rows.length, manifest.artifact_count + modern.artifact_count);
assert.equal(measurementBaseline.rows.every((row) => row.deterministic), true);
assert.deepEqual(
  new Set(measurementBaseline.rows.map((row) => row.id)),
  new Set([...manifest.artifacts, ...modern.artifacts].map((row) => row.id)),
);
assert.match(measurementBaseline.analyzer.git_commit, /^[0-9a-f]{40}$/);
assert.equal(measurementBaseline.analyzer.working_tree_dirty, false);
const recordedAnalyzer = spawnSync(
  "git",
  ["show", `${measurementBaseline.analyzer.git_commit}:pkg/tflite_wasm_audit_bg.wasm`],
  { encoding: null, maxBuffer: 8 * 1024 * 1024 },
);
assert.equal(
  recordedAnalyzer.status,
  0,
  `Cannot read the analyzer WASM from recorded commit ${measurementBaseline.analyzer.git_commit}.`,
);
assert.equal(measurementBaseline.analyzer.wasm_size_bytes, recordedAnalyzer.stdout.byteLength);
assert.equal(
  measurementBaseline.analyzer.wasm_sha256,
  createHash("sha256").update(recordedAnalyzer.stdout).digest("hex"),
);
const baselineIsAncestor = spawnSync(
  "git",
  ["merge-base", "--is-ancestor", measurementBaseline.analyzer.git_commit, "HEAD"],
  { encoding: "utf8" },
);
assert.equal(baselineIsAncestor.status, 0, "Measurement baseline analyzer must be an ancestor of HEAD.");

for (const invalid of ["", "../model.tflite", "/model.tflite", "a\\model.tflite", "a/../model.tflite"]) {
  assert.throws(() => normalizeMemberPath(invalid), /Unsafe archive member path/);
}
assert.equal(normalizeMemberPath("./model.tflite"), "model.tflite");
assert.equal(normalizeMemberPath("folder/model.tflite"), "folder/model.tflite");

const repositoryFiles = await listFiles("corpus/google_legacy");
assert.equal(repositoryFiles.filter((filename) => /\.(?:tflite|tgz|zip)$/i.test(filename)).length, 0);
for (const required of [
  "corpus/google_legacy/README.md",
  GOOGLE_LEGACY_MANIFEST_PATH,
  GOOGLE_LEGACY_COHORTS_PATH,
  "scripts/download-google-legacy-corpus.mjs",
  "scripts/run-google-legacy-corpus-sweep.mjs",
]) {
  if (required.startsWith("scripts/")) await readFile(required, "utf8");
  else assert.ok(repositoryFiles.includes(required), `${required} is missing.`);
}
const directRequirements = await readFile(
  "corpus/google_legacy/conversion/requirements.direct.txt",
  "utf8",
);
const torchHarness = await readFile(
  "corpus/google_legacy/conversion/convert_litert_torch.py",
  "utf8",
);
const quantizerHarness = await readFile(
  "corpus/google_legacy/conversion/quantize_ai_edge.py",
  "utf8",
);
assert.match(directRequirements, /^litert-torch==0\.9\.2$/m);
assert.match(directRequirements, /^ai-edge-quantizer==0\.8\.0$/m);
assert.match(torchHarness, /deepbom\.litert_torch_recipe\.v1/);
assert.match(quantizerHarness, /deepbom\.ai_edge_quantizer_recipe\.v1/);

console.log("Google legacy corpus metadata, cohort policy, and safe extraction contract passed.");

function countBy(rows, selector) {
  return new Map([...rows.reduce((counts, row) => {
    const key = selector(row);
    counts.set(key, (counts.get(key) || 0) + 1);
    return counts;
  }, new Map()).entries()].sort(([left], [right]) => left.localeCompare(right)));
}

async function listFiles(root) {
  const output = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const filename = path.join(root, entry.name).replaceAll("\\", "/");
    if (entry.isDirectory()) output.push(...await listFiles(filename));
    else output.push(filename);
  }
  return output;
}
