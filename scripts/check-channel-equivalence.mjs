import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { resolveNpmCommand } from "./run-utils.mjs";

const root = process.cwd();
const releaseRoot = path.join(root, ".local-validation", "channel-release");
const manifestPath = path.join(releaseRoot, "channel-release-manifest.json");
if (!process.argv.includes("--no-build")) run(process.execPath, ["scripts/build-channel-artifacts.mjs"]);
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
assert.equal(manifest.schema, "deepbom.channel_release.v1");

const engine = path.join(releaseRoot, manifest.artifacts.standalone_engine.path);
const cargoManifest = path.join(releaseRoot, "cargo", "Cargo.toml");
const npmCli = await installNpmPackage(manifest);
const python = installPythonWheel(manifest);
const fixtures = await packageFixtures();
const cases = [
  { path: "web/samples/mobilenet_v2_1.0_224_quant.tflite", format: "tflite" },
  { path: "web/samples/tiny_decoder_llm.onnx", format: "onnx" },
  { path: "web/samples/tinymqa1m.Q4_0.gguf", format: "gguf" },
  { path: "web/samples/nanofable-1m-fp16.safetensors", format: "safetensors" },
  { path: "web/samples/MNISTClassifier.mlmodel", format: "coreml" },
  { path: fixtures.mlpackage, format: "coreml", bundle: "coreml_mlpackage" },
  { path: fixtures.sharded, format: "safetensors", bundle: "safetensors_sharded_repository" },
];

for (const item of cases) {
  const args = ["audit", item.path, "--compact"];
  const canonical = json(run(process.execPath, ["bin/deepbom.mjs", ...args]).stdout);
  const npm = json(run(process.execPath, [npmCli, ...args]).stdout);
  const standalone = json(run(engine, args).stdout);
  const pip = json(run(python, ["-m", "deepbom", ...args]).stdout);
  assert.equal(canonical.format, item.format, `${item.path}: canonical format`);
  if (item.bundle) assert.equal(canonical.artifact_bundle?.kind, item.bundle, `${item.path}: bundle kind`);
  assert.deepEqual(npm, canonical, `${item.path}: installed npm package diverged from canonical CLI`);
  assert.deepEqual(standalone, canonical, `${item.path}: standalone engine diverged from canonical CLI`);
  assert.deepEqual(pip, canonical, `${item.path}: installed Python wheel diverged from canonical CLI`);
}

const cargoResult = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", "audit", cases[1].path, "--compact"], {
  DEEPBOM_ENGINE: engine,
  DEEPBOM_ENGINE_SHA256: createHash("sha256").update(await readFile(engine)).digest("hex"),
  DEEPBOM_RUNTIME_ASSET_DIR: path.join(path.dirname(engine), "pkg"),
});
assert.deepEqual(json(cargoResult.stdout), json(run(process.execPath, ["bin/deepbom.mjs", "audit", cases[1].path, "--compact"]).stdout), "Cargo launcher diverged from canonical CLI");
const unboundCargo = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", "--version"], { DEEPBOM_ENGINE: engine }, false);
assert.notEqual(unboundCargo.status, 0);
assert.match(unboundCargo.stderr, /DEEPBOM_ENGINE_SHA256/);

const npmWasm = path.join(path.dirname(npmCli), "..", "pkg", "tflite_wasm_audit_bg.wasm");
await corruptLastByte(npmWasm);
const corruptNpm = run(process.execPath, [npmCli, "audit", cases[0].path, "--compact"], {}, false);
assert.notEqual(corruptNpm.status, 0);
assert.match(corruptNpm.stderr, /release SHA-256 check/);

const installedRoot = run(python, ["-c", "import pathlib,deepbom;print(pathlib.Path(deepbom.__file__).parent)"]).stdout.trim();
await corruptLastByte(path.join(installedRoot, "_engine", "pkg", "tflite_wasm_audit_bg.wasm"));
const corruptPip = run(python, ["-m", "deepbom", "--version"], {}, false);
assert.notEqual(corruptPip.status, 0);
assert.match(corruptPip.stderr, /failed its SHA-256 check/);

assert.equal(manifest.channels.cargo.status, "launcher_ready_for_immutable_engine_matrix");
console.log("Channel equivalence passed (installed npm tarball and Python wheel; five file formats, Core ML package, sharded SafeTensors, standalone/Cargo parity, and packaged-WASM tamper rejection)." );

async function installNpmPackage(release) {
  const directory = path.join(releaseRoot, "install-probe", "npm");
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "package.json"), '{"private":true}\n');
  const tarball = path.join(releaseRoot, release.channels.npm.package);
  const npmInstall = resolveNpmCommand(["install", "--ignore-scripts", "--no-audit", "--no-fund", tarball]);
  run(npmInstall.command, npmInstall.args, {}, true, directory);
  const cli = path.join(directory, "node_modules", "deepbom", "bin", "deepbom.mjs");
  assert.equal(run(process.execPath, [cli, "--version"]).stdout.trim(), release.version);
  return cli;
}

function installPythonWheel(release) {
  const directory = path.join(releaseRoot, "install-probe", "python");
  run("python", ["-m", "venv", directory]);
  const python = path.join(directory, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const wheel = path.join(releaseRoot, release.channels.python.path);
  run(python, ["-m", "pip", "install", "--disable-pip-version-check", "--no-deps", wheel]);
  assert.equal(run(python, ["-m", "deepbom", "--version"]).stdout.trim(), release.version);
  return python;
}

async function packageFixtures() {
  const root = path.join(releaseRoot, "install-probe", "fixtures");
  const mlpackage = path.join(root, "Fixture.mlpackage");
  await mkdir(path.join(mlpackage, "Data", "com.apple.CoreML"), { recursive: true });
  await writeFile(path.join(mlpackage, "Manifest.json"), `${JSON.stringify({
    fileFormatVersion: "1.0.0",
    rootModelIdentifier: "model-id",
    itemInfoEntries: {
      "model-id": { path: "com.apple.CoreML/MNISTClassifier.mlmodel", name: "MNISTClassifier.mlmodel", author: "com.apple.CoreML", description: "Apple public MNIST classifier" },
    },
  })}\n`);
  await copyFile("web/samples/MNISTClassifier.mlmodel", path.join(mlpackage, "Data", "com.apple.CoreML", "MNISTClassifier.mlmodel"));

  const sharded = path.join(root, "ShardedSafeTensors");
  await mkdir(sharded, { recursive: true });
  const shardName = "model-00001-of-00001.safetensors";
  const source = new Uint8Array(await readFile("web/samples/nanofable-1m-fp16.safetensors"));
  const headerLength = Number(new DataView(source.buffer, source.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(source.subarray(8, 8 + headerLength)).trim());
  const tensorNames = Object.keys(header).filter((name) => name !== "__metadata__");
  assert(tensorNames.length > 0, "SafeTensors fixture has no tensors");
  await copyFile("web/samples/nanofable-1m-fp16.safetensors", path.join(sharded, shardName));
  await writeFile(path.join(sharded, "model.safetensors.index.json"), `${JSON.stringify({
    metadata: { total_size: source.byteLength - 8 - headerLength },
    weight_map: Object.fromEntries(tensorNames.map((name) => [name, shardName])),
  })}\n`);
  return { mlpackage, sharded };
}

async function corruptLastByte(file) {
  const bytes = new Uint8Array(await readFile(file));
  assert(bytes.byteLength > 0, `${file}: cannot corrupt empty file`);
  bytes[bytes.length - 1] ^= 0xff;
  await writeFile(file, bytes);
}

function run(command, args, environment = {}, expectSuccess = true, cwd = root) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    env: { ...process.env, ...environment },
    maxBuffer: 256 * 1024 * 1024,
  });
  if (expectSuccess && result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed\n${result.stdout}\n${result.stderr}`);
  return result;
}

function json(source) {
  return JSON.parse(source.replace(/^\uFEFF/, ""));
}
