import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  CAPTURE_RUN_SCHEMA,
  TENSORFLOW_SOURCE_COMMIT,
  XNNPACK_SOURCE_COMMIT,
  runNativeCapture,
  stableJson,
  verifyNativeCapturePackage,
} from "./native-capture-lib.mjs";
import { defaultNativeRuntimeCacheDir } from "./native-runtime-paths.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelPath = requiredPath("--model");
const outputDir = requiredPath("--output-dir");
const targetProfileId = required("--target-profile-id");
const targetProfileSha256 = required("--target-profile-sha256").toLowerCase();
if (!/^[a-f0-9]{64}$/.test(targetProfileSha256)) throw new Error("--target-profile-sha256 must be 64 lowercase hex characters");
const inputs = argumentsAll("--input").map(parseInput);
if (!inputs.length) throw new Error("At least one --input <tensor-index:name:dimxdim...> is required to bind the capture invocation");
inputs.sort((left, right) => left.tensor_index - right.tensor_index);
if (new Set(inputs.map((input) => input.tensor_index)).size !== inputs.length) throw new Error("--input tensor indices must be unique");

const cacheDir = path.resolve(argument("--cache-dir") || defaultNativeRuntimeCacheDir());
const bootstrapArgs = [path.join(ROOT, "scripts", "bootstrap-instrumented-runtime.mjs"), "--cache-dir", cacheDir, "--build"];
const bazel = argument("--bazel");
if (bazel) bootstrapArgs.push("--bazel", bazel);
await run(process.execPath, bootstrapArgs, ROOT, "pinned instrumented runtime bootstrap/build");

const benchmarkModel = path.join(
  cacheDir, "sources", `tensorflow-${TENSORFLOW_SOURCE_COMMIT}`, "bazel-bin", "tensorflow", "lite", "tools", "benchmark",
  process.platform === "win32" ? "benchmark_model.exe" : "benchmark_model",
);
const buildIdentity = path.join(cacheDir, "instrumented-runtime-build-identity.json");
const debugSymbols = process.platform === "win32" ? path.join(path.dirname(benchmarkModel), "benchmark_model.pdb") : null;
await access(benchmarkModel);
await access(buildIdentity);
if (debugSymbols) await access(debugSymbols);
await mkdir(cacheDir, { recursive: true });

const threads = positiveInteger(argument("--threads") || "1", "--threads");
const runs = positiveInteger(argument("--runs") || "20", "--runs");
const warmups = nonNegativeInteger(argument("--warmups") || "5", "--warmups");
const requestedCpuIds = argument("--cpu-set") ? parseCpuSet(argument("--cpu-set")) : null;
const isolationExpectation = argument("--isolation-expectation") || "affinity_only";
if (requestedCpuIds && !["affinity_only", "exclusive_cpuset"].includes(isolationExpectation)) {
  throw new Error("--isolation-expectation must be affinity_only or exclusive_cpuset");
}
if (requestedCpuIds && threads > requestedCpuIds.length) {
  throw new Error("--threads must not exceed the number of CPUs in --cpu-set");
}
const collectedAt = new Date().toISOString();
const captureId = argument("--capture-id") || `pinned-host-${collectedAt.replace(/[:.]/g, "-")}`;
const configPath = path.join(cacheDir, `${captureId}.capture-run.json`);
const config = {
  schema: CAPTURE_RUN_SCHEMA,
  capture_mode: "runtime_capture",
  artifact_path: modelPath,
  output_dir: outputDir,
  target_profile: { id: targetProfileId, sha256: targetProfileSha256 },
  runtime: {
    name: "TensorFlow Lite benchmark_model",
    version: `tensorflow/tensorflow@${TENSORFLOW_SOURCE_COMMIT}`,
    backend: "Pinned instrumented XNNPACK delegate",
    build: "Bazel opt monolithic; XNNPACK assembly enabled; DeepBOM placement/lowering/dispatch and ArenaPlanner memory trace enabled",
    binary_path: benchmarkModel,
    arguments: [
      "--graph=${artifact_path}",
      `--num_runs=${runs}`,
      `--warmup_runs=${warmups}`,
      `--num_threads=${threads}`,
      "--use_xnnpack=true",
    ],
    environment: { DEEPBOM_XNN_NO_OPERATOR_FUSION: "1" },
  },
  source: { collected_at: collectedAt, capture_id: captureId },
  device: { identity: argument("--host-identity") || `${process.platform}-${process.arch}-host-capture` },
  build: {
    source: { tensorflow_commit: TENSORFLOW_SOURCE_COMMIT, validation_kind: "pinned_instrumented_host_capture" },
    xnnpack_source_commit: XNNPACK_SOURCE_COMMIT,
    microkernel_build_identifier_path: buildIdentity,
    ...(debugSymbols ? { debug_symbols_path: debugSymbols } : {}),
    compile_definitions: [
      { name: "DEEPBOM_RUNTIME_INSTRUMENTATION", value: "1" },
      { name: "XNN_BUILD_ALL_MICROKERNELS", value: "0" },
      { name: "XNN_ENABLE_ASSEMBLY", value: "1" },
    ],
    toolchain: { frontend: bazel || "auto-detected Bazel/Bazelisk", mode: "opt monolithic" },
  },
  invocation: {
    inputs,
    thread_count: threads,
    ...(requestedCpuIds ? {
      resource_partition: {
        requested_cpu_ids: requestedCpuIds,
        affinity_mode: "taskset_process_and_descendants",
        isolation_expectation: isolationExpectation,
      },
    } : {}),
    runtime_options: {
      DEEPBOM_XNN_NO_OPERATOR_FUSION: "1",
      num_runs: runs,
      warmup_runs: warmups,
      use_xnnpack: true,
    },
  },
  instrumentation: { lowering_ids: true, microkernel_ids: true, arena_allocations: true },
};
await writeFile(configPath, stableJson(config));
const result = await runNativeCapture(configPath, { outputDir });
await verifyNativeCapturePackage(result.outputDir);
console.log(`Verified pinned host capture package: ${result.outputDir}`);

function parseInput(value) {
  const match = String(value).match(/^(\d+):([^:]+):(\d+(?:x\d+)*)$/);
  if (!match) throw new Error(`Invalid --input ${value}; expected tensor-index:name:dimxdim...`);
  const shape = match[3].split("x").map(Number);
  if (shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)) throw new Error(`Invalid positive shape in --input ${value}`);
  return { tensor_index: Number(match[1]), name: match[2], shape };
}

function parseCpuSet(value) {
  const ids = new Set();
  for (const token of String(value).split(",").map((item) => item.trim()).filter(Boolean)) {
    const match = token.match(/^(\d+)(?:-(\d+))?$/);
    if (!match) throw new Error(`Invalid --cpu-set ${value}; expected an ascending Linux CPU list such as 2,3 or 2-3`);
    const first = Number(match[1]);
    const last = match[2] == null ? first : Number(match[2]);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last < first || last - first > 1024) {
      throw new Error(`Invalid --cpu-set range ${token}`);
    }
    for (let cpu = first; cpu <= last; cpu += 1) ids.add(cpu);
  }
  const sorted = [...ids].sort((left, right) => left - right);
  if (!sorted.length) throw new Error("--cpu-set requires at least one CPU");
  return sorted;
}

async function run(command, args, cwd, label) {
  const { spawn } = await import("node:child_process");
  console.log(`\n> ${[command, ...args].join(" ")}`);
  const code = await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: "inherit", shell: false });
    child.on("error", reject);
    child.on("exit", resolve);
  });
  if (code !== 0) throw new Error(`${label} failed with exit code ${code}`);
}

function argumentsAll(name) {
  return process.argv.flatMap((value, index) => value === name && process.argv[index + 1] ? [process.argv[index + 1]] : []);
}

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || null : null;
}

function required(name) {
  const value = argument(name);
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function requiredPath(name) { return path.resolve(required(name)); }
function positiveInteger(value, name) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${name} must be a positive integer`); return number; }
function nonNegativeInteger(value, name) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${name} must be a non-negative integer`); return number; }
