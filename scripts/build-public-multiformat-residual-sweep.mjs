import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { gunzipSync, gzipSync } from "node:zlib";

import { coverageResiduals } from "./corpus-coverage-residuals.mjs";

const args = parseArgs(process.argv.slice(2));
const sweep = JSON.parse(readFileSync(args.sweep, "utf8"));
const manifest = JSON.parse(readFileSync(args.manifest, "utf8"));
if (sweep.schema !== "deepbom.public_multiformat_corpus_sweep.v1") throw new Error("Public multiformat sweep schema is invalid.");
if (manifest.schema !== "deepbom.public_multiformat_corpus.v1") throw new Error("Public multiformat manifest schema is invalid.");
const artifacts = new Map(manifest.artifacts.map((artifact) => [artifact.id, artifact]));
const analysisDirectory = path.resolve(path.dirname(args.sweep), "analysis");
const rows = [];
for (const sweepRow of sweep.rows.filter((row) => row.format === "onnx")) {
  if (sweepRow.status !== "passed") throw new Error(`${sweepRow.artifact_id}: residual input sweep did not pass.`);
  const artifact = artifacts.get(sweepRow.artifact_id);
  if (!artifact || artifact.format !== "onnx") throw new Error(`${sweepRow.artifact_id}: manifest binding is unavailable.`);
  const analysisPath = path.join(analysisDirectory, `${artifact.id}.analysis.json.gz`);
  const analysisBytes = gunzipSync(readFileSync(analysisPath));
  const analysis = JSON.parse(analysisBytes);
  const observedAnalysisSha256 = sha256(JSON.stringify(analysis));
  if (observedAnalysisSha256 !== sweepRow.analysis_sha256) throw new Error(`${artifact.id}: analysis SHA-256 mismatch.`);
  const primary = artifact.files.find((file) => file.role === "model")
    || artifact.files.find((file) => file.path.toLowerCase().endsWith(".onnx"));
  if (!primary || primary.sha256 !== sweepRow.primary_artifact_sha256) throw new Error(`${artifact.id}: primary artifact binding mismatch.`);
  rows.push({
    schema: "deepbom.huggingface_corpus_artifact_result.v1",
    repository_id: artifact.source.repository,
    revision: artifact.source.revision,
    tier: artifact.stratum?.architecture_class || "public_multiformat",
    path: primary.path,
    format: "onnx",
    status: "passed",
    error: "",
    artifact_size_bytes: primary.size_bytes,
    artifact_sha256: primary.sha256,
    analysis_sha256: observedAnalysisSha256,
    operator_count: Number(analysis.operator_count || 0),
    tensor_count: Number(analysis.tensor_count || 0),
    coverage_residuals: coverageResiduals(analysis),
  });
}
if (!rows.length) throw new Error("Public multiformat sweep contains no ONNX rows.");
const output = {
  schema: "deepbom.public_multiformat_residual_sweep.v1",
  source_manifest: args.manifest.replaceAll("\\", "/"),
  source_manifest_sha256: sha256(readFileSync(args.manifest)),
  source_sweep: args.sweep.replaceAll("\\", "/"),
  source_sweep_sha256: sha256(readFileSync(args.sweep)),
  selected_artifact_count: rows.length,
  rows,
};
const serializedOutput = Buffer.from(`${JSON.stringify(output, null, 2)}\n`);
writeFileSync(args.output, args.output.endsWith(".gz")
  ? gzipSync(serializedOutput, { level: 9, mtime: 0 })
  : serializedOutput);
console.log(`Wrote ${args.output}: ${rows.length} hash-bound ONNX residual rows.`);

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv) {
  const output = {
    sweep: ".local-validation/public-multiformat-corpus-v1/public-multiformat-corpus-sweep.json",
    manifest: "corpus/public-multiformat-corpus.v1.json",
    output: ".local-validation/residual-coverage/public-multiformat-onnx-v1.23.json",
  };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--sweep") output.sweep = required(argv, ++index, "--sweep");
    else if (argv[index] === "--manifest") output.manifest = required(argv, ++index, "--manifest");
    else if (argv[index] === "--output") output.output = required(argv, ++index, "--output");
    else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  return output;
}

function required(argv, index, key) {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`);
  return value;
}
