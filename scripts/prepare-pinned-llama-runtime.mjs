import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

import {
  GGUF_BACKEND_SOURCE,
  GGUF_RUNTIME_INSTRUMENTATION,
} from "../web/lib/gguf-backend-contract.generated.js";

const ROOT = path.resolve(import.meta.dirname, "..");
const args = parseArgs(process.argv.slice(2));
const sourceDir = path.resolve(args.source || path.join(ROOT, ".local-validation", "llama.cpp-runtime-source"));
const buildDir = path.resolve(args.build || path.join(ROOT, ".local-validation", "llama.cpp-runtime-build"));
const outputPath = path.resolve(args.output || path.join(buildDir, "deepbom-llama-build-attestation.json"));
const patchPath = path.join(ROOT, GGUF_RUNTIME_INSTRUMENTATION.patch_path);
const cmake = String(args.cmake || "cmake");
const buildJobs = boundedPositiveInteger(args.jobs ?? Math.min(4, availableParallelism()), "build jobs", 64);
const buildTimeoutMs = boundedPositiveInteger(args["build-timeout-ms"] ?? 1_800_000, "build timeout", 3_600_000);

await ensurePinnedSource(sourceDir);
const schedulerPath = path.join(sourceDir, GGUF_BACKEND_SOURCE.files.scheduler.path);
const patchSha256 = await hashFile(patchPath);
if (patchSha256 !== GGUF_RUNTIME_INSTRUMENTATION.patch_sha256) fail("GGML scheduler trace patch digest does not match the generated contract.");
const beforeSha256 = await hashFile(schedulerPath);
if (beforeSha256 === GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_original_sha256) {
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "apply", "--check", patchPath], { cwd: sourceDir });
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "apply", patchPath], { cwd: sourceDir });
} else if (beforeSha256 !== GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_patched_sha256) {
  fail(`Pinned scheduler source has an unexpected digest: ${beforeSha256}.`);
}
const patchedSha256 = await hashFile(schedulerPath);
if (patchedSha256 !== GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_patched_sha256) fail("Patched GGML scheduler source digest is not canonical.");
const dirtyRows = lines(run("git", ["-c", "safe.directory=*", "status", "--porcelain"], { cwd: sourceDir }).stdout);
if (dirtyRows.length !== 1 || !dirtyRows[0].endsWith(GGUF_BACKEND_SOURCE.files.scheduler.path.replaceAll("\\", "/"))) {
  fail(`Pinned source tree contains changes outside the canonical scheduler patch: ${dirtyRows.join(" / ") || "none"}.`);
}

await mkdir(buildDir, { recursive: true });
const configureArgs = [
  "-S", sourceDir,
  "-B", buildDir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DGGML_CPU=ON",
  "-DGGML_NATIVE=ON",
  "-DLLAMA_BUILD_TESTS=OFF",
  "-DLLAMA_BUILD_EXAMPLES=OFF",
  "-DLLAMA_BUILD_TOOLS=ON",
  "-DLLAMA_BUILD_SERVER=ON",
  "-DLLAMA_BUILD_APP=OFF",
  "-DLLAMA_BUILD_UI=OFF",
  "-DLLAMA_USE_PREBUILT_UI=OFF",
  "-DLLAMA_CURL=OFF",
];
const configure = run(cmake, configureArgs);
const buildArgs = ["--build", buildDir, "--config", "Release", "--target", "llama-cli", "--parallel", String(buildJobs)];
const build = run(cmake, buildArgs, { timeoutMs: buildTimeoutMs });
const binaryPath = findBinary(buildDir);
const cmakeCachePath = path.join(buildDir, "CMakeCache.txt");
if (!existsSync(cmakeCachePath)) fail("CMakeCache.txt was not produced.");
const cacheText = await readFile(cmakeCachePath, "utf8");
const version = run(binaryPath, ["--version"]);
const attestation = {
  schema: "deepbom.llama_cpp_instrumented_build_attestation.v1",
  evidence_class: "REPRODUCIBLE_BUILD_ATTESTATION",
  source: {
    repository: GGUF_BACKEND_SOURCE.repository,
    commit: GGUF_BACKEND_SOURCE.source_commit,
    scheduler_path: GGUF_BACKEND_SOURCE.files.scheduler.path,
    scheduler_original_sha256: GGUF_RUNTIME_INSTRUMENTATION.scheduler_source_original_sha256,
    scheduler_patched_sha256: patchedSha256,
  },
  instrumentation: {
    patch_id: GGUF_RUNTIME_INSTRUMENTATION.patch_id,
    patch_path: GGUF_RUNTIME_INSTRUMENTATION.patch_path,
    patch_sha256: patchSha256,
    trace_protocol: GGUF_RUNTIME_INSTRUMENTATION.trace_protocol,
  },
  build: {
    configure_arguments: configureArgs,
    build_arguments: buildArgs,
    parallel_jobs: buildJobs,
    timeout_ms: buildTimeoutMs,
    configure_stdout_sha256: sha256(configure.stdout),
    configure_stderr_sha256: sha256(configure.stderr),
    build_stdout_sha256: sha256(build.stdout),
    build_stderr_sha256: sha256(build.stderr),
    cmake_cache_sha256: await hashFile(cmakeCachePath),
    cmake_generator: cacheValue(cacheText, "CMAKE_GENERATOR"),
    c_compiler: cacheValue(cacheText, "CMAKE_C_COMPILER"),
    cxx_compiler: cacheValue(cacheText, "CMAKE_CXX_COMPILER"),
    build_type: cacheValue(cacheText, "CMAKE_BUILD_TYPE") || "Release",
  },
  binary: {
    filename: path.basename(binaryPath),
    sha256: await hashFile(binaryPath),
    version_output: bounded(`${version.stdout}\n${version.stderr}`.trim(), 8192),
  },
  boundary: "This attestation identifies a locally reproduced, source-pinned llama.cpp binary with the canonical DeepBOM scheduler trace patch. It does not assert bit-reproducible compilation across toolchains or runtime execution on any artifact.",
};
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(attestation, null, 2)}\n`, "utf8");
console.log(`Instrumented llama.cpp build attestation written to ${outputPath}`);
console.log(`binary ${binaryPath}`);
console.log(`binary SHA-256 ${attestation.binary.sha256}`);

async function ensurePinnedSource(destination) {
  if (!existsSync(path.join(destination, ".git"))) {
    await mkdir(path.dirname(destination), { recursive: true });
    run("git", ["-c", "core.autocrlf=false", "clone", "--filter=blob:none", "--no-checkout", `https://github.com/${GGUF_BACKEND_SOURCE.repository}.git`, destination], { cwd: ROOT, timeoutMs: 600000 });
  }
  run("git", ["-c", "safe.directory=*", "config", "core.autocrlf", "false"], { cwd: destination });
  run("git", ["-c", "safe.directory=*", "config", "core.eol", "lf"], { cwd: destination });
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "fetch", "--depth", "1", "origin", GGUF_BACKEND_SOURCE.source_commit], { cwd: destination, timeoutMs: 600000 });
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "checkout", "--detach", "--force", GGUF_BACKEND_SOURCE.source_commit], { cwd: destination, timeoutMs: 600000 });
  const head = run("git", ["-c", "safe.directory=*", "rev-parse", "HEAD"], { cwd: destination }).stdout.trim();
  if (head !== GGUF_BACKEND_SOURCE.source_commit) fail(`llama.cpp checkout is ${head}, not the pinned commit.`);
}

function findBinary(buildRoot) {
  const name = process.platform === "win32" ? "llama-cli.exe" : "llama-cli";
  const candidates = [path.join(buildRoot, "bin", name), path.join(buildRoot, "bin", "Release", name), path.join(buildRoot, "Release", name)];
  const found = candidates.find(existsSync);
  if (!found) fail(`Built ${name} was not found in the bounded candidate paths.`);
  return found;
}

function run(command, commandArgs, { cwd = ROOT, timeoutMs = 300000 } = {}) {
  const result = spawnSync(command, commandArgs, { cwd, encoding: "utf8", windowsHide: true, timeout: timeoutMs, maxBuffer: 64 * 1024 * 1024 });
  if (result.error) fail(`${command} could not execute: ${result.error.message}`);
  if (result.status !== 0) fail(`${command} ${commandArgs.join(" ")} failed (${result.status}): ${bounded(result.stderr || result.stdout, 8192)}`);
  return { stdout: result.stdout || "", stderr: result.stderr || "" };
}

async function hashFile(filePath) {
  const hash = createHash("sha256");
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}

function cacheValue(text, key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return text.match(new RegExp(`^${escaped}:[^=]*=(.*)$`, "m"))?.[1]?.trim() || null;
}
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function lines(value) { return String(value).split(/\r?\n/).map((line) => line.trim()).filter(Boolean); }
function bounded(value, limit) { const text = String(value || ""); return text.length <= limit ? text : text.slice(0, limit); }
function boundedPositiveInteger(value, label, maximum) { const number = Number(value); if (!Number.isSafeInteger(number) || number <= 0 || number > maximum) fail(`${label} must be an integer from 1 to ${maximum}.`); return number; }
function parseArgs(tokens) { const parsed = {}; for (let i = 0; i < tokens.length; i += 1) { const token = tokens[i]; if (!token.startsWith("--")) fail(`Unexpected argument: ${token}.`); const [key, inline] = token.slice(2).split("=", 2); parsed[key] = inline ?? (tokens[i + 1] && !tokens[i + 1].startsWith("--") ? tokens[++i] : true); } return parsed; }
function fail(message) { throw new Error(message); }
