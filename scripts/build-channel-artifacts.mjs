import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { access, chmod, copyFile, cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { writeBuildMetadata } from "./write-build-metadata.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const output = resolveOutput(process.argv.slice(2));
const withDist = process.argv.includes("--with-dist");
const publicLicense = path.join(root, "channels", "LICENSE");
const wasmSource = withDist
  ? path.join(root, "dist", "pkg", "tflite_wasm_audit_bg.wasm")
  : path.join(root, "pkg", "tflite_wasm_audit_bg.wasm");
const gitCommit = runCapture("git", ["rev-parse", "HEAD"]).trim();
const gitState = runCapture("git", ["status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":!.local-validation", ":!dist", ":!web/lib/build-metadata.js"]).trim() ? "dirty" : "clean";
const wasmSha256 = createHash("sha256").update(await readFile(wasmSource)).digest("hex");
writeBuildMetadata({ publicDistribution: true });
await assertChannelVersions(packageDocument.version);
assertLocalOutput(output);
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

const npmRoot = path.join(output, "npm", "package");
await mkdir(path.join(npmRoot, "bin"), { recursive: true });
await mkdir(path.join(npmRoot, "pkg"), { recursive: true });
const npmBuildResult = await build({
  entryPoints: [path.join(root, "bin", "deepbom.mjs")],
  outfile: path.join(npmRoot, "bin", "deepbom.mjs"),
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node20",
  minify: true,
  sourcemap: false,
  legalComments: "none",
  metafile: true,
  define: {
    __DEEPBOM_RELEASE_VERSION__: JSON.stringify(packageDocument.version),
    __DEEPBOM_TFLITE_WASM_SHA256__: JSON.stringify(wasmSha256),
  },
});
assertPublicBundleInputs(npmBuildResult.metafile, "npm");
await copyFile(wasmSource, path.join(npmRoot, "pkg", "tflite_wasm_audit_bg.wasm"));
await copyFile(path.join(root, "channels", "npm", "README.md"), path.join(npmRoot, "README.md"));
await copyFile(publicLicense, path.join(npmRoot, "LICENSE"));
await writeFile(path.join(npmRoot, "package.json"), `${JSON.stringify({
  name: "deepbom",
  version: packageDocument.version,
  description: "Local multi-format deployment-artifact analysis for on-device AI models",
  type: "module",
  bin: { deepbom: "bin/deepbom.mjs" },
  files: ["bin/", "pkg/", "README.md", "LICENSE"],
  engines: { node: ">=20" },
  license: "Apache-2.0",
  homepage: "https://deepbom.org",
  repository: { type: "git", url: "git+https://github.com/JunHwan-Kwon/deepbom.git" },
  author: "Jun-Hwan Kwon",
  publishConfig: { access: "public" },
  keywords: ["tflite", "onnx", "gguf", "safetensors", "coreml", "tensorrt", "quantization", "on-device", "ml-bom"],
}, null, 2)}\n`);
await writeFile(path.join(npmRoot, "pkg", "release-manifest.json"), `${JSON.stringify({
  schema: "deepbom.npm_release.v1",
  version: packageDocument.version,
  source: { git_commit: gitCommit, git_state: gitState, distribution: "public_channel" },
  runtime: { node: ">=20", tflite_wasm_sha256: wasmSha256 },
  license: { spdx: "Apache-2.0", file: "LICENSE" },
  public_bundle_input_count: Object.keys(npmBuildResult.metafile.inputs).length,
  supported_inputs: ["tflite", "onnx", "onnx_external_data", "gguf", "safetensors", "safetensors_sharded_repository", "coreml_mlmodel", "coreml_mlpackage", "tensorrt_evidence", "tensorrt_llm_contract"],
}, null, 2)}\n`);
runNpm(["pack", npmRoot, "--pack-destination", path.join(output, "npm")]);

const engineRoot = path.join(output, "engine", `${process.platform}-${process.arch}`);
await mkdir(path.join(engineRoot, "pkg"), { recursive: true });
const cjsBundle = path.join(engineRoot, "deepbom.cjs");
const engineBuildResult = await build({
  entryPoints: [path.join(root, "bin", "deepbom.mjs")],
  outfile: cjsBundle,
  bundle: true,
  platform: "node",
  format: "cjs",
  target: `node${process.versions.node.split(".")[0]}`,
  minify: true,
  sourcemap: false,
  legalComments: "none",
  metafile: true,
  define: {
    __DEEPBOM_RELEASE_VERSION__: JSON.stringify(packageDocument.version),
    __DEEPBOM_TFLITE_WASM_SHA256__: JSON.stringify(wasmSha256),
  },
});
assertPublicBundleInputs(engineBuildResult.metafile, "standalone engine");
await copyFile(wasmSource, path.join(engineRoot, "pkg", "tflite_wasm_audit_bg.wasm"));

const executableName = process.platform === "win32" ? "deepbom-core.exe" : "deepbom-core";
const executable = path.join(engineRoot, executableName);
const blob = path.join(engineRoot, "deepbom.blob");
const seaConfig = path.join(engineRoot, "sea-config.json");
await writeFile(seaConfig, `${JSON.stringify({
  main: cjsBundle,
  output: blob,
  disableExperimentalSEAWarning: true,
  useSnapshot: false,
  useCodeCache: false,
}, null, 2)}\n`);
run(process.execPath, ["--experimental-sea-config", seaConfig]);
await copyFile(process.execPath, executable);
if (process.platform !== "win32") await chmod(executable, 0o755);
if (process.platform === "win32") run(await resolveWindowsSignTool(), ["remove", "/s", executable]);
if (process.platform === "darwin") run("codesign", ["--remove-signature", executable]);
const postjectArguments = [
  path.join(root, "node_modules", "postject", "dist", "cli.js"),
  executable,
  "NODE_SEA_BLOB",
  blob,
  "--sentinel-fuse",
  "NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2",
  "--overwrite",
];
if (process.platform === "darwin") postjectArguments.push("--macho-segment-name", "NODE_SEA");
run(process.execPath, postjectArguments);
if (process.platform === "darwin") run("codesign", ["--force", "--sign", "-", executable]);
run(executable, ["--version"]);

const engineManifest = {
  schema: "deepbom.packaged_engine.v1",
  version: packageDocument.version,
  platform: process.platform,
  arch: process.arch,
  executable: { filename: executableName, ...(await fileRecord(engineRoot, executable)) },
  tflite_wasm: await fileRecord(engineRoot, path.join(engineRoot, "pkg", "tflite_wasm_audit_bg.wasm")),
  source: { git_commit: gitCommit, git_state: gitState },
};
await writeFile(path.join(engineRoot, "manifest.json"), `${JSON.stringify(engineManifest, null, 2)}\n`);

const pythonRoot = path.join(output, "python");
await cp(path.join(root, "channels", "python"), pythonRoot, { recursive: true });
await copyFile(publicLicense, path.join(pythonRoot, "LICENSE"));
const pythonEngineRoot = path.join(pythonRoot, "src", "deepbom", "_engine");
await mkdir(path.join(pythonEngineRoot, "pkg"), { recursive: true });
await copyFile(executable, path.join(pythonEngineRoot, executableName));
if (process.platform !== "win32") await chmod(path.join(pythonEngineRoot, executableName), 0o755);
await copyFile(path.join(engineRoot, "pkg", "tflite_wasm_audit_bg.wasm"), path.join(pythonEngineRoot, "pkg", "tflite_wasm_audit_bg.wasm"));
await copyFile(path.join(engineRoot, "manifest.json"), path.join(pythonEngineRoot, "manifest.json"));
const pythonDist = path.join(output, "python-dist");
await mkdir(pythonDist, { recursive: true });
if (!process.argv.includes("--skip-wheel")) {
  run("python", ["-m", "build", "--wheel", "--outdir", pythonDist, pythonRoot]);
}
const wheelName = (await readdir(pythonDist)).find((name) => name.endsWith(".whl")) || null;

const cargoRoot = path.join(output, "cargo");
await cp(path.join(root, "channels", "cargo"), cargoRoot, { recursive: true });

const hfRoot = path.join(output, "huggingface-space");
await cp(path.join(root, "channels", "huggingface"), hfRoot, { recursive: true });
if (withDist) {
  await cp(path.join(root, "dist"), path.join(hfRoot, "dist"), { recursive: true });
}

const manifest = {
  schema: "deepbom.channel_release.v1",
  version: packageDocument.version,
  source: { git_commit: gitCommit, git_state: gitState, tflite_wasm_source: relative(root, wasmSource) },
  license: { public_channels: "Apache-2.0", private_monorepo: "LicenseRef-DEEPBOM-Proprietary" },
  analysis_entrypoint: "bundled bin/deepbom.mjs",
  supported_formats: ["tflite", "onnx", "gguf", "safetensors", "coreml"],
  supported_packages: ["onnx_external_data", "safetensors_sharded_repository", "coreml_mlpackage"],
  accelerator_contracts: ["tensorrt_static_preflight", "tensorrt_parser_evidence", "tensorrt_llm_config_binding"],
  build_runtime: { node: process.versions.node, platform: process.platform, arch: process.arch },
  channels: {
    npm: { status: "built", runtime: "node>=20", package: relative(output, path.join(output, "npm", `deepbom-${packageDocument.version}.tgz`)) },
    python: { status: wheelName ? "platform_wheel_built" : "platform_wheel_source_built", runtime: "packaged_single_executable_engine", path: wheelName ? relative(output, path.join(pythonDist, wheelName)) : relative(output, pythonRoot) },
    cargo: { status: "launcher_built_publication_blocked_until_signed_engine_matrix", runtime: "verified_external_engine", path: relative(output, cargoRoot) },
    huggingface: { status: withDist ? "static_space_built" : "recipe_built", runtime: "browser", path: relative(output, hfRoot) },
  },
  artifacts: {
    npm_bundle: await fileRecord(output, path.join(npmRoot, "bin", "deepbom.mjs")),
    tflite_wasm: await fileRecord(output, path.join(engineRoot, "pkg", "tflite_wasm_audit_bg.wasm")),
    standalone_engine: await fileRecord(output, executable),
    ...(wheelName ? { python_wheel: await fileRecord(output, path.join(pythonDist, wheelName)) } : {}),
  },
  claim_boundary: "All channel adapters invoke the same bundled JavaScript and TFLite WASM analysis implementation. Static TensorRT and TensorRT-LLM contracts do not imply that NVIDIA runtime libraries are bundled or that runtime execution was observed. Publication readiness additionally requires a clean source state, a verified cross-platform wheel matrix, and registry authentication supplied outside the source tree.",
};
await writeFile(path.join(output, "channel-release-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Built DEEPBOM channel artifacts at ${output}`);

function resolveOutput(args) {
  const index = args.indexOf("--output");
  return path.resolve(index >= 0 ? args[index + 1] : path.join(root, ".local-validation", "channel-release"));
}

function assertLocalOutput(candidate) {
  const allowed = path.join(root, ".local-validation") + path.sep;
  if (!candidate.startsWith(allowed)) throw new Error(`Channel output must stay under ${allowed}`);
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", stdio: "pipe", maxBuffer: 128 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed\n${result.error?.message || ""}\n${result.stdout || ""}\n${result.stderr || ""}`);
  if (result.stdout.trim()) console.log(result.stdout.trim());
  if (result.stderr.trim()) console.error(result.stderr.trim());
}

function runNpm(args) {
  const npmCli = process.env.npm_execpath?.trim();
  if (npmCli) return run(process.execPath, [npmCli, ...args]);
  return run(process.platform === "win32" ? "npm.cmd" : "npm", args);
}

function runCapture(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", stdio: "pipe" });
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed: ${result.stderr || result.error?.message || "unknown error"}`);
  return result.stdout;
}

async function fileRecord(base, file) {
  const bytes = await readFile(file);
  return { path: relative(base, file), byte_length: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
}

async function assertChannelVersions(version) {
  const sources = [
    ["bin/deepbom.mjs", /const VERSION = [^\n]+: "([^"]+)";/],
    ["channels/python/pyproject.toml", /^version\s*=\s*"([^"]+)"/m],
    ["channels/python/src/deepbom/__init__.py", /^__version__\s*=\s*"([^"]+)"/m],
    ["channels/cargo/Cargo.toml", /^version\s*=\s*"([^"]+)"/m],
  ];
  for (const [relativePath, pattern] of sources) {
    const source = await readFile(path.join(root, relativePath), "utf8");
    const observed = source.match(pattern)?.[1];
    if (observed !== version) throw new Error(`Release version drift: package.json=${version}, ${relativePath}=${observed || "missing"}.`);
  }
}

async function resolveWindowsSignTool() {
  const configured = process.env.DEEPBOM_SIGNTOOL?.trim();
  if (configured) {
    await access(configured);
    return configured;
  }
  const located = spawnSync("where.exe", ["signtool.exe"], { encoding: "utf8", windowsHide: true });
  if (located.status === 0) return located.stdout.split(/\r?\n/).map((row) => row.trim()).find(Boolean);
  const kitsRoot = path.join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "Windows Kits", "10", "bin");
  const entries = await readdir(kitsRoot, { withFileTypes: true });
  const architecture = process.arch === "arm64" ? "arm64" : "x64";
  const versions = entries.filter((entry) => entry.isDirectory() && /^\d+\.\d+/.test(entry.name))
    .map((entry) => entry.name).sort((a, b) => b.localeCompare(a, undefined, { numeric: true }));
  for (const version of versions) {
    const candidate = path.join(kitsRoot, version, architecture, "signtool.exe");
    try { await access(candidate); return candidate; } catch { /* Continue through installed SDK versions. */ }
  }
  throw new Error("Windows SEA assembly requires signtool.exe to remove the invalidated Node Authenticode signature.");
}

function relative(base, file) {
  return path.relative(base, file).replaceAll(path.sep, "/");
}

function assertPublicBundleInputs(metafile, label) {
  const forbiddenPrefixes = [
    "protected/",
    "web/protected/",
    "docs/private/",
  ];
  const forbiddenFiles = new Set([
    "scripts/generate-ort-rulepack.mjs",
    "scripts/generate-tflite-delegate-rulepack.mjs",
    "scripts/generate-xnnpack-delegate-rulepack.mjs",
  ]);
  const inputs = Object.keys(metafile?.inputs || {}).map((input) => {
    const absolute = path.isAbsolute(input) ? input : path.resolve(root, input);
    return relative(root, absolute);
  });
  const escaped = inputs.filter((input) => input === ".." || input.startsWith("../"));
  if (escaped.length) throw new Error(`${label} bundle input escaped the repository root: ${escaped.join(", ")}`);
  const forbidden = inputs.filter((input) => forbiddenFiles.has(input)
    || forbiddenPrefixes.some((prefix) => input.startsWith(prefix)));
  if (forbidden.length) {
    throw new Error(`${label} bundle crossed the public/private source boundary:\n${forbidden.map((input) => `  - ${input}`).join("\n")}`);
  }
  const sourceMapOutputs = Object.keys(metafile?.outputs || {}).filter((output) => output.endsWith(".map"));
  if (sourceMapOutputs.length) throw new Error(`${label} bundle emitted source maps: ${sourceMapOutputs.join(", ")}`);
}
