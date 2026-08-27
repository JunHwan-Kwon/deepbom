import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

import { ORT_BUILD_ATTESTATION_SCHEMA, ORT_BUILD_SOURCE, validateOrtBuildAttestation } from "../web/lib/ort-build-attestation.js";
import { parseOrtReducedOperatorConfig } from "../web/lib/ort-reduced-operator-config.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const ROOT = path.resolve(import.meta.dirname, "..");
const args = parseArgs(process.argv.slice(2));
const sourceDir = path.resolve(args.source || path.join(ROOT, ".local-validation", "onnxruntime-source-build"));
const buildDir = path.resolve(args.build || path.join(ROOT, ".local-validation", "onnxruntime-native-build"));
const outputPath = path.resolve(args.output || path.join(buildDir, "deepbom-ort-build-attestation.json"));
const reducedConfigPath = args["reduced-op-config"] ? path.resolve(args["reduced-op-config"]) : null;
const enableTypeReduction = args["enable-reduced-operator-type-support"] === true;

await ensurePinnedSource(sourceDir);
const pristine = lines(run("git", ["-c", "safe.directory=*", "status", "--porcelain", "--untracked-files=no"], { cwd: sourceDir }).stdout);
if (pristine.length) fail(`Pinned ORT source must be pristine before build: ${pristine.join(" / ")}`);
const pinnedFiles = [];
for (const source of Object.values(ORT_BUILD_SOURCE.files)) {
  const observed = await hashFile(path.join(sourceDir, source.path));
  if (observed !== source.sha256) fail(`Pinned ORT build file ${source.path} has digest ${observed}, expected ${source.sha256}.`);
  pinnedFiles.push({ path: source.path, sha256: observed });
}
pinnedFiles.sort((left, right) => left.path.localeCompare(right.path));
const submoduleStatus = run("git", ["-c", "safe.directory=*", "submodule", "status", "--recursive"], { cwd: sourceDir, timeoutMs: 600000 }).stdout;
if (lines(submoduleStatus).some((line) => line.startsWith("-") || line.startsWith("+"))) fail("Pinned ORT submodules are missing or differ from the committed revisions.");

const reducedOperatorConfig = reducedConfigPath ? await buildReducedConfigInput(reducedConfigPath, enableTypeReduction) : null;
if (enableTypeReduction && !reducedOperatorConfig) fail("Type-reduced ORT build requires --reduced-op-config.");
await mkdir(buildDir, { recursive: true });
const driver = path.join(sourceDir, process.platform === "win32" ? "build.bat" : "build.sh");
const buildArguments = [
  "--config", "Release",
  "--build_dir", buildDir,
  "--build_shared_lib",
  "--parallel",
  "--skip_tests",
];
if (reducedOperatorConfig) buildArguments.push("--include_ops_by_config", reducedConfigPath);
if (enableTypeReduction) buildArguments.push("--enable_reduced_operator_type_support");
const nativeBuild = run(driver, buildArguments, { cwd: sourceDir, timeoutMs: 7_200_000, shell: process.platform === "win32" });
const nativeOutputDir = await findOrtNativeOutput(buildDir);
const cmakeCachePath = await findNamedFile(buildDir, "CMakeCache.txt");

const jsRoot = path.join(sourceDir, "js");
const commonRoot = path.join(jsRoot, "common");
const nodeRoot = path.join(jsRoot, "node");
const installRuns = [
  run(npmCommand(), ["ci", "--ignore-scripts=false"], { cwd: jsRoot, timeoutMs: 1_800_000, shell: process.platform === "win32" }),
  run(npmCommand(), ["ci", "--ignore-scripts=false"], { cwd: commonRoot, timeoutMs: 1_800_000, shell: process.platform === "win32" }),
  run(npmCommand(), ["ci", "--ignore-scripts=false"], { cwd: nodeRoot, timeoutMs: 1_800_000, shell: process.platform === "win32" }),
];
const nodeBuildArguments = ["run", "build", "--", "--rebuild", "--config=Release", `--onnxruntime-build-dir=${nativeOutputDir}`];
const nodeBuild = run(npmCommand(), nodeBuildArguments, { cwd: nodeRoot, timeoutMs: 3_600_000, shell: process.platform === "win32" });
const runtimeBinRoot = path.join(nodeRoot, "bin", "napi-v6", process.platform, process.arch);
const binaryInventory = await collectBinaryInventory(runtimeBinRoot, nodeRoot);
const primary = binaryInventory.find((item) => /(?:^|\/)(?:onnxruntime\.dll|libonnxruntime\.so(?:\.\d+)*|libonnxruntime\.dylib)$/.test(item.path))
  || binaryInventory.find((item) => item.path.endsWith("onnxruntime_binding.node"));
if (!primary) fail("Source-built ORT package does not contain a bounded primary runtime binary.");
const postBuildDiff = run("git", ["-c", "safe.directory=*", "diff", "--binary", "--no-ext-diff"], { cwd: sourceDir, timeoutMs: 300000 }).stdout;
const packageManifestPath = path.join(nodeRoot, "package.json");
const body = {
  schema: ORT_BUILD_ATTESTATION_SCHEMA,
  evidence_class: "REPRODUCIBLE_SOURCE_BUILD_ATTESTATION",
  source: {
    repository: ORT_BUILD_SOURCE.repository,
    commit: ORT_BUILD_SOURCE.commit,
    pristine_before_build: true,
    submodule_status_sha256: sha256TextHex(submoduleStatus),
    post_build_diff_sha256: sha256TextHex(postBuildDiff),
    pinned_files: pinnedFiles,
  },
  reduced_operator_config: reducedOperatorConfig,
  build: {
    configuration: "Release",
    build_arguments: buildArguments,
    build_stdout_sha256: sha256TextHex(nativeBuild.stdout),
    build_stderr_sha256: sha256TextHex(nativeBuild.stderr),
    cmake_cache_sha256: await hashFile(cmakeCachePath),
    node_build_arguments: nodeBuildArguments,
    node_install_stdout_sha256: sha256TextHex(installRuns.map((item) => item.stdout).join("\n")),
    node_install_stderr_sha256: sha256TextHex(installRuns.map((item) => item.stderr).join("\n")),
    node_build_stdout_sha256: sha256TextHex(nodeBuild.stdout),
    node_build_stderr_sha256: sha256TextHex(nodeBuild.stderr),
  },
  runtime_package: {
    package_name: "onnxruntime-node",
    version: ORT_BUILD_SOURCE.runtime_version,
    package_manifest_sha256: await hashFile(packageManifestPath),
    platform: process.platform,
    arch: process.arch,
    node_napi: "napi-v6",
    binary_inventory: binaryInventory,
    binary_inventory_sha256: sha256TextHex(canonicalJson(binaryInventory)),
    primary_binary_path: primary.path,
    primary_binary_sha256: primary.sha256,
  },
  boundary: "This attestation is produced by a single wrapper that verifies the exact ORT checkout and pinned build drivers, executes the native shared-library and Node-binding builds, and hashes the resulting selected package. It does not assert bit-reproducibility across compilers, device execution, or provider acceptance for an artifact.",
};
const attestation = validateOrtBuildAttestation({ ...body, attestation_sha256: sha256TextHex(canonicalJson(body)) });
await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(attestation, null, 2)}\n`, "utf8");
console.log(`Pinned ORT Node runtime attestation written to ${outputPath}`);
console.log(`Runtime module ${path.join(nodeRoot, "dist", "index.js")}`);
console.log(`Binary inventory ${attestation.runtime_package.binary_inventory_sha256}`);

async function ensurePinnedSource(destination) {
  if (!existsSync(path.join(destination, ".git"))) {
    await mkdir(path.dirname(destination), { recursive: true });
    run("git", ["-c", "core.autocrlf=false", "clone", "--filter=blob:none", "--no-checkout", `https://github.com/${ORT_BUILD_SOURCE.repository}.git`, destination], { cwd: ROOT, timeoutMs: 1_200_000 });
  }
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "fetch", "--depth", "1", "origin", ORT_BUILD_SOURCE.commit], { cwd: destination, timeoutMs: 1_200_000 });
  run("git", ["-c", "safe.directory=*", "-c", "core.autocrlf=false", "checkout", "--detach", "--force", ORT_BUILD_SOURCE.commit], { cwd: destination, timeoutMs: 1_200_000 });
  run("git", ["-c", "safe.directory=*", "submodule", "sync", "--recursive"], { cwd: destination, timeoutMs: 300000 });
  run("git", ["-c", "safe.directory=*", "submodule", "update", "--init", "--recursive", "--depth", "1"], { cwd: destination, timeoutMs: 3_600_000 });
  const head = run("git", ["-c", "safe.directory=*", "rev-parse", "HEAD"], { cwd: destination }).stdout.trim();
  if (head !== ORT_BUILD_SOURCE.commit) fail(`ORT checkout is ${head}, not the pinned commit.`);
}

async function buildReducedConfigInput(configPath, typeReductionEnabled) {
  const sourceText = await readFile(configPath, "utf8");
  const normalized = parseOrtReducedOperatorConfig(sourceText);
  return {
    schema: "deepbom.ort_reduced_operator_build_input.v1",
    source_name: path.basename(configPath),
    source_text: sourceText,
    source_sha256: sha256TextHex(sourceText),
    normalized_sha256: sha256TextHex(canonicalJson(normalized)),
    normalized_config: normalized,
    type_reduction_enabled: typeReductionEnabled,
    binary_binding_status: "BOUND_AS_OBSERVED_BUILD_INPUT",
  };
}

async function findOrtNativeOutput(root) {
  const files = await recursiveFiles(root, 6);
  const library = files.find((item) => /(?:^|[\\/])(?:onnxruntime\.dll|libonnxruntime\.so|libonnxruntime\.dylib)$/.test(item));
  if (!library) fail("ORT shared-library build output was not found in the bounded build tree.");
  return path.dirname(library);
}

async function findNamedFile(root, name) {
  const files = await recursiveFiles(root, 6);
  const found = files.find((item) => path.basename(item) === name);
  if (!found) fail(`${name} was not found in the bounded ORT build tree.`);
  return found;
}

async function collectBinaryInventory(root, packageRoot) {
  if (!existsSync(root)) fail(`ORT Node binary directory was not produced: ${root}`);
  const files = await recursiveFiles(root, 4);
  if (!files.length || files.length > 256) fail("ORT Node binary inventory is empty or oversized.");
  const rows = [];
  for (const file of files) {
    const info = await stat(file);
    rows.push({ path: path.relative(packageRoot, file).replaceAll("\\", "/"), byte_length: info.size, sha256: await hashFile(file) });
  }
  return rows.sort((left, right) => left.path.localeCompare(right.path));
}

async function recursiveFiles(root, maxDepth, depth = 0) {
  if (depth > maxDepth) return [];
  const entries = await readdir(root, { withFileTypes: true });
  const output = [];
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) output.push(...await recursiveFiles(full, maxDepth, depth + 1));
    else if (entry.isFile()) output.push(full);
  }
  return output;
}

function run(command, commandArgs, { cwd = ROOT, timeoutMs = 300000, shell = false } = {}) {
  const result = spawnSync(command, commandArgs, { cwd, encoding: "utf8", windowsHide: true, timeout: timeoutMs, maxBuffer: 256 * 1024 * 1024, shell });
  if (result.error) fail(`${command} could not execute: ${result.error.message}`);
  if (result.status !== 0) fail(`${command} ${commandArgs.join(" ")} failed (${result.status}): ${bounded(result.stderr || result.stdout, 8192)}`);
  return { stdout: result.stdout || "", stderr: result.stderr || "" };
}

async function hashFile(filePath) { const hash = createHash("sha256"); await new Promise((resolvePromise, reject) => { const stream = createReadStream(filePath); stream.on("data", (chunk) => hash.update(chunk)); stream.on("error", reject); stream.on("end", resolvePromise); }); return hash.digest("hex"); }
function npmCommand() { return process.platform === "win32" ? "npm.cmd" : "npm"; }
function lines(value) { return String(value).split(/\r?\n/).map((line) => line.trim()).filter(Boolean); }
function bounded(value, limit) { const text = String(value || ""); return text.length <= limit ? text : text.slice(0, limit); }
function parseArgs(tokens) { const result = {}; for (let index = 0; index < tokens.length; index += 1) { const token = tokens[index]; if (!token.startsWith("--")) fail(`Unexpected argument: ${token}.`); const [key, inline] = token.slice(2).split("=", 2); result[key] = inline ?? (tokens[index + 1] && !tokens[index + 1].startsWith("--") ? tokens[++index] : true); } return result; }
function fail(message) { throw new Error(message); }
