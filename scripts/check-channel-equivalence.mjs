import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { resolveNpmCommand } from "./run-utils.mjs";
import { writeBuildMetadata } from "./write-build-metadata.mjs";
import { buildInterfaceQuantizationContractLedger } from "../web/lib/quantization-contract-summary.js";

const root = process.cwd();
const releaseRoot = path.join(root, ".local-validation", "channel-release");
const manifestPath = path.join(releaseRoot, "channel-release-manifest.json");
const platformSmoke = process.argv.includes("--platform-smoke");
const releaseContract = process.argv.includes("--release-contract");
if (platformSmoke && releaseContract) throw new Error("--platform-smoke and --release-contract are mutually exclusive.");
if (!process.argv.includes("--no-build")) run(process.execPath, ["scripts/build-channel-artifacts.mjs"]);
const buildMetadataPath = path.join(root, "web", "lib", "build-metadata.js");
const priorBuildMetadata = existsSync(buildMetadataPath) ? readFileSync(buildMetadataPath) : null;
let buildMetadataRestored = false;
const restoreBuildMetadata = () => {
  if (buildMetadataRestored) return;
  if (priorBuildMetadata) writeFileSync(buildMetadataPath, priorBuildMetadata);
  else rmSync(buildMetadataPath, { force: true });
  buildMetadataRestored = true;
};
process.on("exit", restoreBuildMetadata);
// Every packaged channel is built from the public-distribution provenance
// contract. Compare it with the source CLI under that same contract, while
// preserving the private monorepo metadata outside this check.
writeBuildMetadata({ publicDistribution: true });
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
assert.equal(manifest.schema, "deepbom.channel_release.v1");

const engine = path.join(releaseRoot, manifest.artifacts.standalone_engine.path);
const cargoManifest = path.join(releaseRoot, "cargo", "Cargo.toml");
const npmCli = platformSmoke ? null : await installNpmPackage(manifest);
const python = platformSmoke || releaseContract ? await installPythonWheel(manifest) : null;
const fixtures = platformSmoke ? null : await packageFixtures();
const fileCases = [
  { path: "web/samples/mobilenet_v2_1.0_224_quant.tflite", format: "tflite" },
  { path: "web/samples/tiny_decoder_llm.onnx", format: "onnx" },
  { path: "web/samples/tinymqa1m.Q4_0.gguf", format: "gguf" },
  { path: "web/samples/nanofable-1m-fp16.safetensors", format: "safetensors" },
  { path: "web/samples/MNISTClassifier.mlmodel", format: "coreml" },
];
const cases = platformSmoke ? fileCases.slice(0, 2) : [
  ...fileCases,
  { path: fixtures.mlpackage, format: "coreml", bundle: "coreml_mlpackage" },
  { path: fixtures.sharded, format: "safetensors", bundle: "safetensors_sharded_repository" },
];
const canonicalByPath = new Map();
const capabilityArgs = ["capabilities", "--compact"];
const canonicalCapabilities = json(run(process.execPath, ["bin/deepbom.mjs", ...capabilityArgs]).stdout);
if (npmCli) {
  assert.deepEqual(json(run(process.execPath, [npmCli, ...capabilityArgs]).stdout), canonicalCapabilities,
    "installed npm capability discovery diverged from canonical CLI");
  const npmSelfTest = json(runNpmExecutable(npmCli, ["self-test", "--compact"]).stdout);
  assert.equal(npmSelfTest.status, "pass", "installed npm executable self-test failed");
}
if (platformSmoke || releaseContract) {
  assert.deepEqual(json(run(engine, capabilityArgs).stdout), canonicalCapabilities,
    "standalone engine capability discovery diverged from canonical CLI");
  assert.deepEqual(json(run(python, ["-m", "deepbom", ...capabilityArgs]).stdout), canonicalCapabilities,
    "installed Python capability discovery diverged from canonical CLI");
}

for (const [caseIndex, item] of cases.entries()) {
  const args = ["audit", item.path, "--compact"];
  const canonical = json(run(process.execPath, ["bin/deepbom.mjs", ...args]).stdout);
  canonicalByPath.set(item.path, canonical);
  assert.equal(canonical.format, item.format, `${item.path}: canonical format`);
  if (item.bundle) assert.equal(canonical.artifact_bundle?.kind, item.bundle, `${item.path}: bundle kind`);
  if (npmCli) {
    const npm = json(run(process.execPath, [npmCli, ...args]).stdout);
    assert.deepEqual(npm, canonical, `${item.path}: installed npm package diverged from canonical CLI`);
    if (item.format === "tflite") {
      const installedCommand = json(runNpmExecutable(npmCli, ["audit", path.resolve(item.path), "--compact"]).stdout);
      assert.deepEqual(installedCommand, canonical, `${item.path}: npm executable shim diverged from canonical CLI`);
    }
  }
  // The standalone engine and Python adapter both forward to the same immutable
  // engine. Two format-diverse executions prove that boundary without repeating
  // every expensive parser case in every wrapper channel.
  if ((platformSmoke || releaseContract) && caseIndex < 2) {
    assert.deepEqual(json(run(engine, args).stdout), canonical, `${item.path}: standalone engine diverged from canonical CLI`);
    assert.deepEqual(json(run(python, ["-m", "deepbom", ...args]).stdout), canonical, `${item.path}: installed Python wheel diverged from canonical CLI`);
  }
}

if (!platformSmoke) {
  const tfliteAnalysis = canonicalByPath.get(fileCases[0].path);
  const interfaceLedger = buildInterfaceQuantizationContractLedger(tfliteAnalysis);
  const interfaceContract = path.join(releaseRoot, "install-probe", "interface-contract.json");
  await writeFile(interfaceContract, `${JSON.stringify({
    schema: "deepbom.production_interface_contract.v1",
    artifact_sha256: tfliteAnalysis.model_sha256,
    implementation_sha256: "a".repeat(64),
    parameters: interfaceLedger.parameters,
  })}\n`);
  const lightweightTflite = "web/samples/mobilenet_v1_025_224_float.tflite";
  for (const [label, args] of [
    ["verify", ["verify", fileCases[0].path, "--contract", interfaceContract, "--compact"]],
    ["diff", ["diff", lightweightTflite, lightweightTflite, "--compact"]],
    ["explore", ["explore", lightweightTflite, "--compact"]],
  ]) {
    const canonical = json(run(process.execPath, ["bin/deepbom.mjs", ...args]).stdout);
    assert.deepEqual(json(run(process.execPath, [npmCli, ...args]).stdout), canonical, `${label}: installed npm package diverged`);
  }
  for (const outputFormat of ["envelope", "sarif"]) {
    const args = ["audit", fileCases[1].path, "--format", outputFormat, "--compact"];
    const canonical = json(run(process.execPath, ["bin/deepbom.mjs", ...args]).stdout);
    assert.deepEqual(json(run(process.execPath, [npmCli, ...args]).stdout), canonical,
      `${outputFormat}: installed npm automation output diverged`);
  }
  const policyArgs = ["audit", fileCases[1].path, "--fail-on", "high", "--compact"];
  const canonicalPolicy = run(process.execPath, ["bin/deepbom.mjs", ...policyArgs], {}, false);
  const npmPolicy = run(process.execPath, [npmCli, ...policyArgs], {}, false);
  assert.equal(canonicalPolicy.status, 2, "canonical finding policy must block the fixture");
  assert.equal(npmPolicy.status, canonicalPolicy.status, "installed npm finding-policy exit code diverged");
  assert.deepEqual(json(npmPolicy.stdout), json(canonicalPolicy.stdout), "installed npm finding-policy evidence diverged");
}

if (platformSmoke) {
  console.log("Platform channel smoke passed (native standalone engine and installed Python wheel; TFLite/WASM and ONNX execution parity)." );
} else {
  if (releaseContract) {
    const cargoResult = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", "audit", cases[1].path, "--compact"], {
      DEEPBOM_ENGINE: engine,
      DEEPBOM_ENGINE_SHA256: createHash("sha256").update(await readFile(engine)).digest("hex"),
      DEEPBOM_RUNTIME_ASSET_DIR: path.join(path.dirname(engine), "pkg"),
    });
    assert.deepEqual(json(cargoResult.stdout), canonicalByPath.get(cases[1].path), "Cargo launcher diverged from canonical CLI");
    const cargoCapabilities = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", ...capabilityArgs], {
      DEEPBOM_ENGINE: engine,
      DEEPBOM_ENGINE_SHA256: createHash("sha256").update(await readFile(engine)).digest("hex"),
      DEEPBOM_RUNTIME_ASSET_DIR: path.join(path.dirname(engine), "pkg"),
    });
    assert.deepEqual(json(cargoCapabilities.stdout), canonicalCapabilities, "Cargo capability discovery diverged");
    const unboundCargo = run(
      "cargo",
      ["run", "--quiet", "--manifest-path", cargoManifest, "--", "audit", cases[1].path, "--compact"],
      { DEEPBOM_ENGINE: engine },
      false,
    );
    assert.notEqual(unboundCargo.status, 0);
    assert.match(unboundCargo.stderr, /DEEPBOM_ENGINE_SHA256/);
  }

  const npmWasm = path.join(path.dirname(npmCli), "..", "pkg", "tflite_wasm_audit_bg.wasm");
  await corruptLastByte(npmWasm);
  const corruptNpm = run(process.execPath, [npmCli, "audit", cases[0].path, "--compact"], {}, false);
  assert.notEqual(corruptNpm.status, 0);
  assert.match(corruptNpm.stderr, /release SHA-256 check/);

  if (python) {
    const installedRoot = run(python, ["-c", "import pathlib,deepbom;print(pathlib.Path(deepbom.__file__).parent)"]).stdout.trim();
    await corruptLastByte(path.join(installedRoot, "_engine", "pkg", "tflite_wasm_audit_bg.wasm"));
    const corruptPip = run(python, ["-m", "deepbom", "--version"], {}, false);
    assert.notEqual(corruptPip.status, 0);
    assert.match(corruptPip.stderr, /failed its SHA-256 check/);
  }

  assert.equal(manifest.channels.cargo.status, "launcher_ready_for_immutable_engine_matrix");
  const cargoStatus = releaseContract ? "Cargo execution and unbound-engine rejection" : "Cargo execution reserved for --release-contract";
  const nativeStatus = releaseContract ? "standalone/Python TFLite and ONNX execution parity" : "native/Python execution reserved for platform release smoke";
  console.log(`Channel equivalence passed (installed npm tarball across five formats and two package forms; ${nativeStatus}; capability/envelope/SARIF/policy and verify/diff/explore npm parity; ${cargoStatus}; packaged-WASM tamper rejection).`);
}

async function installNpmPackage(release) {
  const directory = path.join(releaseRoot, "install-probe", "npm");
  await rm(directory, { recursive: true, force: true });
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "package.json"), '{"private":true}\n');
  const tarball = path.join(releaseRoot, release.channels.npm.package);
  const npmInstall = resolveNpmCommand(["install", "--ignore-scripts", "--no-audit", "--no-fund", tarball]);
  run(npmInstall.command, npmInstall.args, {}, true, directory);
  const cli = path.join(directory, "node_modules", "deepbom", "bin", "deepbom.mjs");
  assert.equal(run(process.execPath, [cli, "--version"]).stdout.trim(), release.version);
  return cli;
}

async function installPythonWheel(release) {
  const directory = path.join(releaseRoot, "install-probe", "python");
  await rm(directory, { recursive: true, force: true });
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

function runNpmExecutable(npmCli, args, expectSuccess = true) {
  const installRoot = path.resolve(path.dirname(npmCli), "..", "..", "..");
  const invocation = resolveNpmCommand(["exec", "--", "deepbom", ...args]);
  return run(invocation.command, invocation.args, {}, expectSuccess, installRoot);
}

function json(source) {
  return JSON.parse(source.replace(/^\uFEFF/, ""));
}
