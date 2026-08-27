import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { access, chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { runCommand } from "./run-utils.mjs";
import { defaultNativeRuntimeCacheDir } from "./native-runtime-paths.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TF_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const XNN_COMMIT = "23a67314f7afdbb76191589ae090d82bf55afbfa";
const BAZELISK_VERSION = "1.29.0";
const BAZELISK_ASSETS = Object.freeze({
  "darwin-x64": Object.freeze({ name: "bazelisk-darwin-amd64", sha256: "16c3d7aa15323a9fb69f56c7ec5733ed18bedb786680d0ba13bb12a3c8083007" }),
  "darwin-arm64": Object.freeze({ name: "bazelisk-darwin-arm64", sha256: "cee851f726789227d5561004e9904a52be45c3efb56f8b38b6993d6adbaa0409" }),
  "linux-x64": Object.freeze({ name: "bazelisk-linux-amd64", sha256: "5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992" }),
  "linux-arm64": Object.freeze({ name: "bazelisk-linux-arm64", sha256: "e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010" }),
  "win32-x64": Object.freeze({ name: "bazelisk-windows-amd64.exe", sha256: "092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b" }),
  "win32-arm64": Object.freeze({ name: "bazelisk-windows-arm64.exe", sha256: "8bc42bd5d7857f18a21440b906469bb6c7cf91a7c72364d4b1e5ec56a76fe94f" }),
});
const SOURCES = Object.freeze({
  tensorflow: Object.freeze({
    url: `https://github.com/tensorflow/tensorflow/archive/${TF_COMMIT}.zip`,
    sha256: "081edc42742db04d154f1d793816d384469dabe1ec95696de7bd866b3f0902c5",
    archive: "tensorflow.zip",
    directory: `tensorflow-${TF_COMMIT}`,
    sentinels: Object.freeze([
      "WORKSPACE",
      "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc",
      "tensorflow/lite/tools/benchmark/BUILD",
      "tensorflow/core/framework/tensor.cc",
    ]),
  }),
  xnnpack: Object.freeze({
    url: `https://github.com/google/XNNPACK/archive/${XNN_COMMIT}.zip`,
    sha256: "b1ac2fcb6ed85623430a4ac05ddb08432e3ca87ccf77596ea2b4bc7d5ebad00a",
    archive: "xnnpack.zip",
    directory: `XNNPACK-${XNN_COMMIT}`,
    sentinels: Object.freeze([
      "BUILD.bazel",
      "CMakeLists.txt",
      "include/xnnpack.h",
      "src/operator-run.c",
      "src/runtime.c",
      "src/xnnpack/subgraph.h",
    ]),
  }),
});

const cacheDir = path.resolve(argument("--cache-dir") || defaultNativeRuntimeCacheDir());
const archiveDir = path.join(cacheDir, "archives");
const sourceDir = path.join(cacheDir, "sources");
await mkdir(archiveDir, { recursive: true });
await mkdir(sourceDir, { recursive: true });
const roots = {};
const archives = {};

for (const [name, source] of Object.entries(SOURCES)) {
  const archivePath = path.join(archiveDir, source.archive);
  await ensureArchive(source, archivePath);
  const sourceRoot = path.join(sourceDir, source.directory);
  await ensureExtracted(source, archivePath, sourceRoot);
  roots[name] = sourceRoot;
  archives[name] = archivePath;
}

const manifestPath = path.join(cacheDir, "deepbom-instrumentation-manifest.json");
if (!(await preparedManifestIsCurrent(manifestPath))) {
  for (const [name, source] of Object.entries(SOURCES)) {
    const sourceRoot = roots[name];
    const relative = path.relative(sourceDir, sourceRoot);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) throw new Error(`Refusing to refresh source outside the native runtime cache: ${sourceRoot}`);
    await rm(sourceRoot, { recursive: true, force: true });
    await ensureExtracted(source, archives[name], sourceRoot);
  }
  await runCommand(process.execPath, [
    path.join(ROOT, "scripts", "prepare-instrumented-runtime.mjs"),
    "--tensorflow", roots.tensorflow,
    "--xnnpack", roots.xnnpack,
    "--tensorflow-archive", archives.tensorflow,
    "--xnnpack-archive", archives.xnnpack,
    "--manifest", manifestPath,
  ]);
}

let binaryPath = null;
let buildIdentityPath = null;
let debugSymbolsPath = null;
if (process.argv.includes("--build")) {
  const bazel = argument("--bazel") || await provisionBazelisk();
  const python = await findPython();
  const bash = process.platform === "win32" ? await findBash() : null;
  const args = [];
  if (process.platform === "win32") args.push(`--output_user_root=${path.join(process.env.SystemDrive || "C:", "dbb").replaceAll("\\", "/")}`);
  args.push(
    "build", "-c", "opt", "--config=monolithic",
    "--define=xnn_enable_assembly=true",
    "--define=protobuf_allow_msvc=true",
    `--python_path=${python.executable.replaceAll("\\", "/")}`,
    `--repo_env=HERMETIC_PYTHON_VERSION=${python.version}`,
    `--override_repository=XNNPACK=${roots.xnnpack.replaceAll("\\", "/")}`,
  );
  if (process.platform === "win32") {
    args.push(
      "--copt=/utf-8",
      "--copt=/DDEEPBOM_RUNTIME_INSTRUMENTATION=1",
      "--host_copt=/utf-8",
      "--conlyopt=/std:c11",
      "--conlyopt=/experimental:c11atomics",
    );
  }
  if (process.platform !== "win32") args.push("--linkopt=-Wl,--export-dynamic");
  if (process.platform === "linux") args.push("--linkopt=-ldl");
  args.push("//tensorflow/lite/tools/benchmark:benchmark_model");
  await runWithCwd(bazel, args, roots.tensorflow, bash ? { ...process.env, BAZEL_SH: bash } : process.env);
  binaryPath = path.join(roots.tensorflow, "bazel-bin", "tensorflow", "lite", "tools", "benchmark", process.platform === "win32" ? "benchmark_model.exe" : "benchmark_model");
  await access(binaryPath);
  if (process.platform === "win32") {
    debugSymbolsPath = path.join(path.dirname(binaryPath), "benchmark_model.pdb");
    await access(debugSymbolsPath);
  }
  buildIdentityPath = path.join(cacheDir, "instrumented-runtime-build-identity.json");
  await writeFile(buildIdentityPath, `${JSON.stringify({
    schema: "deepbom.instrumented_runtime_build_identity.v1",
    tensorflow_commit: TF_COMMIT,
    xnnpack_commit: XNN_COMMIT,
    instrumentation_manifest_sha256: await sha256File(manifestPath),
    runtime_binary_sha256: await sha256File(binaryPath),
    runtime_debug_symbols: debugSymbolsPath ? {
      format: "pdb",
      path: debugSymbolsPath,
      sha256: await sha256File(debugSymbolsPath),
    } : null,
    bazel_frontend_sha256: await sha256File(bazel).catch(() => null),
    bazelisk_version: argument("--bazel") ? null : BAZELISK_VERSION,
    python,
    bash: bash ? { executable: bash, sha256: await sha256File(bash) } : null,
    compile_definitions: [
      { name: "DEEPBOM_RUNTIME_INSTRUMENTATION", value: "1" },
      { name: "XNN_BUILD_ALL_MICROKERNELS", value: "0" },
      { name: "XNN_ENABLE_ASSEMBLY", value: "1" },
    ],
    build_command: { command: bazel, arguments: args },
  }, null, 2)}\n`);
}

console.log(JSON.stringify({
  schema: "deepbom.instrumented_runtime_bootstrap.v1",
  tensorflow_root: roots.tensorflow,
  xnnpack_root: roots.xnnpack,
  instrumentation_manifest: manifestPath,
  benchmark_model: binaryPath,
  runtime_debug_symbols: debugSymbolsPath,
  build_identity: buildIdentityPath,
}, null, 2));

async function ensureArchive(source, archivePath) {
  const current = await sha256File(archivePath).catch(() => "");
  if (current === source.sha256) return;
  if (current) throw new Error(`${archivePath} exists but does not match the pinned SHA-256`);
  const temporary = `${archivePath}.download`;
  await rm(temporary, { force: true });
  const response = await fetch(source.url, { redirect: "follow" });
  if (!response.ok || !response.body) throw new Error(`Failed to download ${source.url}: ${response.status}`);
  await pipeline(response.body, createWriteStream(temporary, { flags: "wx" }));
  const digest = await sha256File(temporary);
  if (digest !== source.sha256) {
    await rm(temporary, { force: true });
    throw new Error(`Downloaded archive SHA-256 mismatch: expected ${source.sha256}, received ${digest}`);
  }
  await rename(temporary, archivePath);
}

async function ensureExtracted(source, archivePath, sourceRoot) {
  const completionMarker = path.join(sourceRoot, ".deepbom-extraction-complete.json");
  const marker = JSON.parse(await readFile(completionMarker, "utf8").catch(() => "null"));
  if (marker?.archive_sha256 === source.sha256 && await allSentinelsExist(sourceRoot, source.sentinels)) return;
  const resolvedRoot = path.resolve(sourceRoot);
  if (!resolvedRoot.startsWith(`${sourceDir}${path.sep}`)) throw new Error(`Refusing to clean source outside cache: ${resolvedRoot}`);
  await rm(resolvedRoot, { recursive: true, force: true });
  const result = await runAllowingTimestampWarning("tar", ["-xf", archivePath, "-C", sourceDir]);
  const missing = await missingSentinels(sourceRoot, source.sentinels);
  if (missing.length > 0) {
    throw new Error(`Archive extraction is incomplete; missing ${missing.join(", ")}${result ? `: ${result}` : ""}`);
  }
  await writeFile(completionMarker, `${JSON.stringify({
    schema: "deepbom.source_extraction.v1",
    archive_sha256: source.sha256,
    sentinels: source.sentinels,
  }, null, 2)}\n`);
}

async function allSentinelsExist(root, sentinels) {
  return (await missingSentinels(root, sentinels)).length === 0;
}

async function missingSentinels(root, sentinels) {
  const missing = [];
  for (const relative of sentinels) {
    if (!(await stat(path.join(root, relative)).catch(() => null))?.isFile()) missing.push(relative);
  }
  return missing;
}

async function preparedManifestIsCurrent(manifestPath) {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8").catch(() => "null"));
  if (manifest?.schema !== "deepbom.instrumented_runtime_patch.v1.3"
    || manifest.tensorflow_commit !== TF_COMMIT || manifest.xnnpack_commit !== XNN_COMMIT
    || manifest.source_verification?.mode !== "pinned_archive_and_file_hashes") return false;
  const requiredArenaPaths = [
    path.resolve(sourceDir, SOURCES.tensorflow.directory, "tensorflow", "lite", "arena_planner.cc"),
    path.resolve(sourceDir, SOURCES.tensorflow.directory, "tensorflow", "lite", "simple_memory_arena.cc"),
  ];
  if (manifest.patch_counts?.tensorflow_arena_memory_instrumentation !== 2
    || manifest.patch_counts?.tensorflow_arena_debug_msvc_compatibility !== 1
    || requiredArenaPaths.some((requiredPath) => !(manifest.files || []).some(
      (file) => path.resolve(file.path) === requiredPath,
    ))) return false;
  for (const file of manifest.files || []) {
    if (await sha256File(file.path) !== file.sha256) throw new Error(`Prepared runtime file changed after manifest creation: ${file.path}`);
  }
  return (manifest.files || []).length >= 10;
}

async function provisionBazelisk() {
  const asset = BAZELISK_ASSETS[`${process.platform}-${process.arch}`];
  if (!asset) throw new Error(`No pinned Bazelisk binary is declared for ${process.platform}-${process.arch}; pass --bazel explicitly`);
  const toolDir = path.join(cacheDir, "tools");
  const toolPath = path.join(toolDir, asset.name);
  await mkdir(toolDir, { recursive: true });
  const current = await sha256File(toolPath).catch(() => "");
  if (current && current !== asset.sha256) throw new Error(`Cached Bazelisk SHA-256 mismatch at ${toolPath}`);
  if (!current) {
    const temporary = `${toolPath}.download`;
    await rm(temporary, { force: true });
    const url = `https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/${asset.name}`;
    const response = await fetch(url, { redirect: "follow" });
    if (!response.ok || !response.body) throw new Error(`Failed to download ${url}: ${response.status}`);
    await pipeline(response.body, createWriteStream(temporary, { flags: "wx" }));
    const digest = await sha256File(temporary);
    if (digest !== asset.sha256) {
      await rm(temporary, { force: true });
      throw new Error(`Bazelisk SHA-256 mismatch: expected ${asset.sha256}, received ${digest}`);
    }
    await rename(temporary, toolPath);
  }
  if (process.platform !== "win32") await chmod(toolPath, 0o755);
  await runWithCwd(toolPath, ["--version"], roots.tensorflow);
  return toolPath;
}

async function findPython() {
  const candidates = [process.env.PYTHON_BIN_PATH, process.platform === "win32" ? "python" : "python3", "python"]
    .filter(Boolean);
  for (const command of [...new Set(candidates)]) {
    try {
      const output = await captureCommand(command, ["-c", "import sys; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}')"]);
      const [executable, version] = output.trim().split(/\r?\n/);
      if (executable && /^3\.\d+$/.test(version || "")) return { executable: path.resolve(executable), version };
    } catch { /* try next candidate */ }
  }
  throw new Error("Python 3 is required by the pinned TensorFlow build and no executable interpreter was found");
}

async function findBash() {
  const candidates = [
    process.env.BAZEL_SH,
    process.env.ProgramFiles ? path.join(process.env.ProgramFiles, "Git", "bin", "bash.exe") : null,
    process.env.ProgramFiles ? path.join(process.env.ProgramFiles, "Git", "usr", "bin", "bash.exe") : null,
  ].filter(Boolean);
  for (const candidate of [...new Set(candidates)]) {
    if ((await stat(candidate).catch(() => null))?.isFile()) return path.resolve(candidate);
  }
  throw new Error("A valid Bash executable is required for Bazel genrules on Windows; install Git for Windows or set BAZEL_SH");
}

function captureCommand(command, args) {
  return import("node:child_process").then(({ execFile }) => new Promise((resolve, reject) => {
    execFile(command, args, { encoding: "utf8", windowsHide: true }, (error, stdout, stderr) => {
      if (error) reject(new Error(stderr || error.message));
      else resolve(stdout);
    });
  }));
}

async function runWithCwd(command, args, cwd, env = process.env) {
  const { spawn } = await import("node:child_process");
  console.log(`\n> ${[command, ...args].join(" ")}`);
  const code = await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: "inherit", shell: false });
    child.on("error", reject);
    child.on("exit", resolve);
  });
  if (code !== 0) throw new Error(`${command} failed with exit code ${code}`);
}

async function runAllowingTimestampWarning(command, args) {
  const { spawn } = await import("node:child_process");
  return new Promise((resolve, reject) => {
    let stderr = "";
    const child = spawn(command, args, { cwd: ROOT, stdio: ["ignore", "inherit", "pipe"], shell: false });
    child.stderr.on("data", (chunk) => { stderr += chunk; process.stderr.write(chunk); });
    child.on("error", reject);
    child.on("exit", (code) => resolve(code === 0 ? "" : stderr.trim()));
  });
}

async function sha256File(filePath) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(filePath), hash);
  return hash.digest("hex");
}

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || null : null;
}
