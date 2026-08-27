import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import { basename, resolve } from "node:path";
import { spawnSync } from "node:child_process";

import { GGUF_BACKEND_PROFILES, GGUF_BACKEND_SOURCE, GGUF_RUNTIME_INSTRUMENTATION } from "../web/lib/gguf-backend-contract.generated.js";
import {
  GGUF_RUNTIME_ENVIRONMENT_SCHEMA,
  parseGgmlSchedulerTrace,
  validateLlamaCppBuildAttestation,
} from "../web/lib/gguf-runtime-environment.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const args = parseArgs(process.argv.slice(2));
const modelPath = resolve(required(args, "model"));
const binaryPath = resolve(required(args, "binary"));
const cmakeCachePath = resolve(required(args, "cmake-cache"));
const buildAttestationPath = resolve(required(args, "build-attestation"));
const outputPath = resolve(required(args, "output"));
const sourceCommit = normalizedSha(required(args, "source-commit"), "source commit");
if (sourceCommit !== GGUF_BACKEND_SOURCE.source_commit) fail("GGUF runtime collection requires the generated contract's pinned llama.cpp commit.");
const requestedBackend = String(args.backend || "cpu").toLowerCase();
const requestedProfile = GGUF_BACKEND_PROFILES.find((profile) => profile.id === requestedBackend);
if (!requestedProfile) fail(`Unknown backend profile: ${requestedBackend}.`);

const contextSize = positiveInteger(args.context || 2048, "context");
const batchSize = positiveInteger(args.batch || 512, "batch");
const ubatchSize = positiveInteger(args.ubatch || Math.min(batchSize, 128), "ubatch");
if (ubatchSize > batchSize) fail("ubatch cannot exceed batch.");
const gpuLayers = nonNegativeInteger(args["gpu-layers"] || 0, "gpu-layers");

const [artifactSha256, binarySha256, cmakeCacheBytes, buildAttestationBytes] = await Promise.all([
  hashFile(modelPath),
  hashFile(binaryPath),
  readFile(cmakeCachePath),
  readFile(buildAttestationPath),
]);
const cmakeCacheSha256 = sha256(cmakeCacheBytes);
const rawBuildAttestation = parseJson(buildAttestationBytes.toString("utf8"), "instrumented llama.cpp build attestation");
const buildAttestation = validateLlamaCppBuildAttestation({
  file_sha256: sha256(buildAttestationBytes),
  canonical_sha256: sha256TextHex(canonicalJson(rawBuildAttestation)),
  document: rawBuildAttestation,
});
if (buildAttestation.document.binary.sha256 !== binarySha256) fail("Selected runtime binary does not match the instrumented build attestation.");
if (buildAttestation.document.build.cmake_cache_sha256 !== cmakeCacheSha256) fail("Selected CMake cache does not match the instrumented build attestation.");
const options = parseCmakeOptions(cmakeCacheBytes.toString("utf8"));
const compiledProfiles = GGUF_BACKEND_PROFILES.filter((profile) => options[profile.cmake_option] === true);
if (!compiledProfiles.some((profile) => profile.id === requestedBackend)) {
  fail(`Requested backend ${requestedBackend} is not enabled in ${cmakeCachePath}.`);
}

const version = run(binaryPath, ["--version"]);
const inventory = run(binaryPath, ["--list-devices"]);
const runSmoke = args["run-smoke"] === true;
const smokeArgs = runSmoke ? [
  "-m", modelPath,
  "-p", "DeepBOM deterministic runtime evidence probe.",
  "-n", "1",
  "-c", String(contextSize),
  "-b", String(batchSize),
  "-ub", String(ubatchSize),
  "-ngl", String(gpuLayers),
  "--seed", "1",
  "--single-turn",
  "--simple-io",
] : [];
const tracePath = resolve(`${outputPath}.ggml-trace-${randomUUID()}.log`);
let smoke = null;
let computeGraph = null;
if (runSmoke) {
  smoke = run(binaryPath, smokeArgs, {
    timeoutMs: positiveInteger(args.timeout || 120000, "timeout"),
    env: { ...process.env, DEEPBOM_GGML_TRACE: tracePath },
  });
  let traceBytes;
  try {
    traceBytes = await readFile(tracePath);
  } catch {
    fail("Instrumented llama.cpp did not emit the required GGML scheduler trace.");
  } finally {
    await unlink(tracePath).catch(() => {});
  }
  computeGraph = parseGgmlSchedulerTrace(traceBytes.toString("utf8"), { traceSha256: sha256(traceBytes) });
  if (smoke.exitCode === 0 && (computeGraph.successful_dispatch_count < 1 || computeGraph.failed_dispatch_count > 0)) {
    fail("Successful llama.cpp execution did not produce an exclusively successful scheduler-dispatch ledger.");
  }
}
const modelLoadObserved = Boolean(computeGraph?.successful_dispatch_count > 0);
const inferenceObserved = Boolean(
  smoke?.exitCode === 0
  && computeGraph?.successful_dispatch_count > 0
  && computeGraph?.failed_dispatch_count === 0
);
const combinedSmokeOutput = smoke ? `${smoke.stdout}\n${smoke.stderr}` : "";
const backendNamed = smoke && new RegExp(`\\b${escapeRegExp(requestedProfile.label)}\\b`, "i").test(combinedSmokeOutput);
const schedulerBackendNamed = computeGraph?.graphs.some((graph) => graph.backends.some((backend) => {
  const name = backend.name.toLowerCase();
  return name.includes(requestedProfile.label.toLowerCase()) || name.includes(requestedBackend);
}));
const collectedAt = new Date().toISOString();
const document = {
  schema: GGUF_RUNTIME_ENVIRONMENT_SCHEMA,
  evidence_class: "RUNTIME_ENVIRONMENT_MANIFEST",
  artifact: { format: "gguf", filename: basename(modelPath), sha256: artifactSha256 },
  runtime: {
    repository: GGUF_BACKEND_SOURCE.repository,
    source_commit: sourceCommit,
    binary_sha256: binarySha256,
    version_output: bounded(`${version.stdout}\n${version.stderr}`.trim(), 8192),
  },
  build: {
    cmake_cache_sha256: cmakeCacheSha256,
    options,
    compiled_backend_profile_ids: compiledProfiles.map((profile) => profile.id),
    attestation: buildAttestation,
  },
  selection: {
    requested_backend_profile_id: requestedBackend,
    context_size: contextSize,
    batch_size: batchSize,
    ubatch_size: ubatchSize,
    gpu_layers: gpuLayers,
  },
  device: {
    platform: `${os.platform()} ${os.release()}`,
    architecture: os.arch(),
    hostname_sha256: sha256(Buffer.from(os.hostname(), "utf8")),
    cpu_features: [],
    accelerator_inventory: inventory.exitCode === 0
      ? inventory.stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).slice(0, 256)
      : [],
  },
  capture: {
    capture_id: randomUUID(),
    collected_at: collectedAt,
    collector: { name: "deepbom-gguf-runtime-collector", version: "2" },
  },
  observations: {
    backend_inventory_status: inventory.exitCode === 0 ? "observed_success" : "observed_failure",
    model_load_status: smoke ? modelLoadObserved ? "observed_success" : "observed_failure" : "not_run",
    inference_status: smoke ? inferenceObserved ? "observed_success" : "observed_failure" : "not_run",
    selected_backend_observation: schedulerBackendNamed ? "scheduler_graph_named_backend" : backendNamed ? "runtime_output_named_backend" : "not_observed",
    elapsed_ms: smoke?.elapsedMs ?? null,
    process_exit_code: smoke?.exitCode ?? null,
    stdout_sha256: smoke ? sha256(Buffer.from(smoke.stdout, "utf8")) : null,
    stderr_sha256: smoke ? sha256(Buffer.from(smoke.stderr, "utf8")) : null,
  },
  instrumentation: { ...GGUF_RUNTIME_INSTRUMENTATION, build_attestation_sha256: buildAttestation.file_sha256 },
  compute_graph: computeGraph,
  boundary: "Collector output binds model, pinned source, canonical scheduler patch, selected build, binary, CMake cache, requested configuration, host identity, and bounded process observations. Captured scheduler graphs establish split and backend assignment for those calls only; microkernel identity and uncaptured paths remain unobserved.",
};

await writeFile(outputPath, `${JSON.stringify(document, null, 2)}\n`, "utf8");
console.log(`GGUF runtime environment written to ${outputPath}`);
console.log(`artifact ${artifactSha256}`);
console.log(`binary ${binarySha256}`);
console.log(`compiled backends ${compiledProfiles.map((profile) => profile.id).join(", ")}`);
console.log(`smoke ${document.observations.inference_status}`);
console.log(`scheduler graphs ${computeGraph?.graph_count || 0}; dispatched ${computeGraph?.dispatched_graph_count || 0}`);

function parseArgs(tokens) {
  const parsed = {};
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token.startsWith("--")) fail(`Unexpected argument: ${token}.`);
    const [rawKey, inline] = token.slice(2).split("=", 2);
    if (!rawKey) fail("Empty argument name.");
    if (inline != null) parsed[rawKey] = inline;
    else if (tokens[index + 1] && !tokens[index + 1].startsWith("--")) parsed[rawKey] = tokens[++index];
    else parsed[rawKey] = true;
  }
  return parsed;
}

function parseCmakeOptions(text) {
  const cache = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (!line || line.startsWith("//") || line.startsWith("#")) continue;
    const match = /^([^:=]+):BOOL=(ON|OFF|TRUE|FALSE|1|0)$/i.exec(line.trim());
    if (match) cache.set(match[1], ["ON", "TRUE", "1"].includes(match[2].toUpperCase()));
  }
  return Object.fromEntries(GGUF_BACKEND_PROFILES.map((profile) => {
    if (!cache.has(profile.cmake_option)) fail(`CMake cache does not declare ${profile.cmake_option}.`);
    return [profile.cmake_option, cache.get(profile.cmake_option)];
  }));
}

function run(binary, commandArgs, { timeoutMs = 30000, env = process.env } = {}) {
  const started = performance.now();
  const result = spawnSync(binary, commandArgs, { encoding: "utf8", windowsHide: true, timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024, env });
  const elapsedMs = performance.now() - started;
  if (result.error && result.error.code !== "ETIMEDOUT") fail(`Cannot execute ${binary}: ${result.error.message}`);
  return {
    exitCode: Number.isInteger(result.status) ? result.status : -1,
    elapsedMs,
    stdout: bounded(result.stdout || "", 16 * 1024 * 1024),
    stderr: bounded(result.stderr || (result.error?.message || ""), 16 * 1024 * 1024),
  };
}

async function hashFile(path) {
  const hash = createHash("sha256");
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function parseJson(text, label) { try { const value = JSON.parse(text); if (!value || Array.isArray(value) || typeof value !== "object") fail(`${label} must be a JSON object.`); return value; } catch (error) { fail(`${label} is invalid: ${error.message}`); } }
function required(value, key) { if (!value[key]) fail(`--${key} is required.`); return value[key]; }
function normalizedSha(value, label) { const text = String(value).toLowerCase(); if (!/^[a-f0-9]{40,64}$/.test(text)) fail(`${label} must be a hexadecimal commit ID.`); return text; }
function positiveInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number <= 0) fail(`${label} must be a positive integer.`); return number; }
function nonNegativeInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 0) fail(`${label} must be a non-negative integer.`); return number; }
function bounded(value, limit) { const text = String(value); return text.length <= limit ? text : text.slice(0, limit); }
function escapeRegExp(value) { return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function fail(message) { throw new Error(message); }
