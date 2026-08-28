import AdmZip from "adm-zip";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  publicArtifactFilePath,
  publicMultiformatCacheDir,
  readPublicMultiformatCorpus,
} from "./public-multiformat-corpus-lib.mjs";
import { fileIdentity } from "./hash-bound-corpus-cache.mjs";
import { canonicalJson } from "../web/lib/report-utils.js";

const args = parseArgs(process.argv.slice(2));
const manifest = await readPublicMultiformatCorpus(args.manifest);
const availableArtifactIds = new Set(manifest.artifacts.map((artifact) => artifact.id));
const availableFormats = new Set(manifest.artifacts.map((artifact) => artifact.format));
for (const artifactId of args.artifactIds) {
  if (!availableArtifactIds.has(artifactId)) throw new Error(`Unknown corpus artifact selector: ${artifactId}`);
}
for (const format of args.formats) {
  if (!availableFormats.has(format)) throw new Error(`Unknown corpus format selector: ${format}`);
}
const selected = manifest.artifacts.filter((artifact) => (!args.formats.length || args.formats.includes(artifact.format))
  && (!args.artifactIds.length || args.artifactIds.includes(artifact.id)));
if (!selected.length) throw new Error("Corpus selectors matched no artifacts.");
const cacheDir = path.resolve(args.cacheDir || publicMultiformatCacheDir());
const outputDir = path.resolve(args.outputDir);
const workDir = path.join(outputDir, "work");
const cli = path.resolve(args.cli);
await assertRegularFile(cli, "CLI entrypoint");
await mkdir(workDir, { recursive: true });

const rows = [];
for (let index = 0; index < selected.length; index += 1) {
  const artifact = selected[index];
  try {
    await assertCachedArtifact(cacheDir, artifact);
    const input = await resolveCliInput(cacheDir, artifact, workDir);
    const resultPath = path.join(workDir, `${artifact.id}.analysis.json`);
    const human = runCli(["audit", input], args.timeoutMs);
    if (!/^DEEPBOM \S+ deployment-artifact audit/m.test(human.stdout)) throw new Error("default output is not the bounded human summary");
    if (Buffer.byteLength(human.stdout, "utf8") >= 8192) throw new Error("default human output exceeds the 8192-byte terminal bound");
    runCli(["audit", input, "--compact", "--output", resultPath], args.timeoutMs);
    const analysisBytes = await readFile(resultPath);
    const analysis = JSON.parse(analysisBytes.toString("utf8"));
    await rm(resultPath, { force: true });
    if (analysis.format !== artifact.format) throw new Error(`format ${analysis.format || "missing"} differs from ${artifact.format}`);
    if (!/^[a-f0-9]{64}$/.test(String(analysis.model_sha256 || ""))) throw new Error("analysis artifact SHA-256 is missing");
    if (!(Number(analysis.file_size_bytes) > 0)) throw new Error("analysis file-size identity is missing");
    rows.push({
      artifact_id: artifact.id,
      format: artifact.format,
      status: "passed",
      source_primary_sha256: primarySourceSha256(artifact),
      analysis_artifact_sha256: analysis.model_sha256,
      analysis_output_sha256: sha256(analysisBytes),
      human_output_sha256: sha256(Buffer.from(human.stdout, "utf8")),
      human_output_byte_length: Buffer.byteLength(human.stdout, "utf8"),
      operator_count: nullableInteger(analysis.operator_count),
      tensor_count: nullableInteger(analysis.tensor_count),
    });
    console.log(`[${index + 1}/${selected.length}] ${artifact.id}: passed`);
  } catch (error) {
    rows.push({ artifact_id: artifact.id, format: artifact.format, status: "failed", error: error?.stack || String(error) });
    console.error(`[${index + 1}/${selected.length}] ${artifact.id}: FAILED ${error?.message || error}`);
  }
}

const body = {
  schema: "deepbom.public_cli_corpus_sweep.v1",
  corpus_schema: manifest.schema,
  corpus_generated_at: manifest.generated_at,
  corpus_manifest_sha256: sha256(await readFile(args.manifest)),
  cli: {
    filename: path.basename(cli),
    sha256: sha256(await readFile(cli)),
    runtime: process.version,
  },
  method: {
    original_bytes: "verified_user_local_hash_bound_cache",
    execution_isolation: "fresh_node_process_per_output_mode",
    output_modes_per_artifact: ["bounded_human_summary", "compact_analysis_json"],
    package_archives: "Core ML package ZIPs are materialized under the ignored local work directory after safe-path validation.",
    claim_boundary: "This sweep validates the CLI routing and output surface over the declared corpus. It does not create a new prevalence sample or replace the independent analyzer repeat sweep.",
  },
  path_record_count: rows.length,
  format_counts: countBy(rows, (row) => row.format),
  status_counts: countBy(rows, (row) => row.status),
  rows,
};
const result = { ...body, sweep_sha256: sha256(Buffer.from(canonicalJson(body), "utf8")) };
await mkdir(outputDir, { recursive: true });
await writeFile(path.join(outputDir, "public-cli-corpus-sweep.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
if (!args.keepWork) await rm(workDir, { recursive: true, force: true });
console.log(`Wrote ${path.join(outputDir, "public-cli-corpus-sweep.json")}: ${rows.length} records.`);
if (rows.some((row) => row.status !== "passed")) process.exitCode = 1;

async function assertCachedArtifact(root, artifact) {
  for (const file of artifact.files) {
    const filename = publicArtifactFilePath(root, artifact, file);
    const observed = await fileIdentity(filename);
    if (observed?.size !== file.size_bytes) throw new Error(`${artifact.id}/${file.path}: cached byte length differs from the manifest`);
    if (observed.sha256 !== file.sha256) throw new Error(`${artifact.id}/${file.path}: cached SHA-256 differs from the manifest`);
  }
}

async function resolveCliInput(root, artifact, workingDirectory) {
  const artifactRoot = path.join(root, artifact.format, artifact.id);
  if (artifact.format === "onnx") {
    const primary = artifact.files.find((file) => file.role === "model")
      || artifact.files.find((file) => file.path.toLowerCase().endsWith(".onnx"));
    if (!primary) throw new Error("ONNX primary model path is missing");
    return publicArtifactFilePath(root, artifact, primary);
  }
  if (artifact.format === "gguf") return publicArtifactFilePath(root, artifact, artifact.files[0]);
  if (artifact.format === "safetensors") return artifactRoot;
  const primary = artifact.files[0];
  const source = publicArtifactFilePath(root, artifact, primary);
  if (!primary.path.toLowerCase().endsWith(".zip")) return source;
  const destination = path.join(workingDirectory, `${artifact.id}.mlpackage`);
  await rm(destination, { recursive: true, force: true });
  await mkdir(destination, { recursive: true });
  const zip = new AdmZip(source);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (!entries.length || entries.length > 20_000) throw new Error(`Core ML ZIP entry count ${entries.length} is invalid`);
  const totalUncompressedBytes = entries.reduce((sum, entry) => sum + Number(entry.header?.size || 0), 0);
  if (!Number.isSafeInteger(totalUncompressedBytes) || totalUncompressedBytes > 8 * 1024 ** 3) {
    throw new Error(`Core ML ZIP uncompressed byte total ${totalUncompressedBytes} is outside the bounded materialization limit`);
  }
  for (const entry of entries) {
    const unixMode = (Number(entry.header?.attr || 0) >>> 16) & 0xffff;
    if ((unixMode & 0o170000) === 0o120000) throw new Error(`Core ML ZIP contains a symbolic link: ${entry.entryName}`);
    const relative = safeRelativePath(entry.entryName);
    if (!relative) throw new Error(`Core ML ZIP contains unsafe path ${entry.entryName}`);
    const output = path.join(destination, ...relative.split("/"));
    await mkdir(path.dirname(output), { recursive: true });
    await writeFile(output, entry.getData(), { flag: "wx" });
  }
  return destination;
}

function runCli(argv, timeoutMs) {
  const result = spawnSync(process.execPath, [cli, ...argv], {
    cwd: process.cwd(),
    encoding: "utf8",
    timeout: timeoutMs,
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.error || result.status !== 0) throw new Error(result.error?.message || result.stderr.trim() || `CLI exited ${result.status}`);
  return result;
}

async function assertRegularFile(filename, label) {
  const metadata = await stat(filename);
  if (!metadata.isFile()) throw new Error(`${label} must be a regular file`);
}

function primarySourceSha256(artifact) {
  const primary = artifact.files.find((file) => file.role === "model") || artifact.files[0];
  return primary.sha256;
}

function safeRelativePath(value) {
  const normalized = String(value || "").replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)) return "";
  if (normalized.split("/").some((part) => !part || part === "." || part === "..")) return "";
  return normalized;
}

function nullableInteger(value) {
  return Number.isSafeInteger(Number(value)) && Number(value) >= 0 ? Number(value) : null;
}

function countBy(rows, selector) {
  const counts = new Map();
  for (const row of rows) {
    const key = selector(row) || "unknown";
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right)));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseArgs(argv) {
  const output = {
    manifest: "corpus/public-multiformat-corpus.v1.json",
    cacheDir: "",
    outputDir: ".local-validation/public-cli-corpus-v1",
    cli: "bin/deepbom.mjs",
    formats: [],
    artifactIds: [],
    timeoutMs: 600_000,
    keepWork: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output-dir") output.outputDir = required(argv, ++index, key);
    else if (key === "--cli") output.cli = required(argv, ++index, key);
    else if (key === "--format") output.formats.push(required(argv, ++index, key));
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
    else if (key === "--timeout-ms") output.timeoutMs = boundedInteger(required(argv, ++index, key), 10_000, 3_600_000, key);
    else if (key === "--keep-work") output.keepWork = true;
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
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < minimum || number > maximum) throw new Error(`${key} must be ${minimum}..${maximum}.`);
  return number;
}
