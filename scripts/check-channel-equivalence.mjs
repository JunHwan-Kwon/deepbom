import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
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
const commandTimeoutMs = boundedPositiveInteger(process.env.DEEPBOM_CHANNEL_COMMAND_TIMEOUT_MS, 5 * 60_000);
const mcpTimeoutMs = boundedPositiveInteger(process.env.DEEPBOM_CHANNEL_MCP_TIMEOUT_MS, 5 * 60_000);
const installProbeRunId = `${process.pid}-${Date.now()}`;
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
const fixtures = await packageFixtures();
const spacedOnnx = path.join(releaseRoot, "install-probe", "artifact path with spaces", "tiny decoder model.onnx");
await mkdir(path.dirname(spacedOnnx), { recursive: true });
await copyFile("web/samples/tiny_decoder_llm.onnx", spacedOnnx);
const fileCases = [
  { path: "web/samples/mobilenet_v2_1.0_224_quant.tflite", format: "tflite" },
  { path: spacedOnnx, format: "onnx" },
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
const agentCapabilityArgs = ["capabilities", "--format", "agent-json"];
const canonicalAgentCapabilities = json(run(process.execPath, ["bin/deepbom.mjs", ...agentCapabilityArgs]).stdout);
if (npmCli) {
  assert.deepEqual(json(run(process.execPath, [npmCli, ...capabilityArgs]).stdout), canonicalCapabilities,
    "installed npm capability discovery diverged from canonical CLI");
  assert.deepEqual(json(run(process.execPath, [npmCli, ...agentCapabilityArgs]).stdout), canonicalAgentCapabilities,
    "installed npm agent capability discovery diverged from canonical CLI");
  await verifyInstalledAgentIntegration(npmCli, manifest.version);
  const npmSelfTest = json(runNpmExecutable(npmCli, ["self-test", "--compact"]).stdout);
  assert.equal(npmSelfTest.status, "pass", "installed npm executable self-test failed");
  const mcpFrames = exchangeMcp(process.execPath, [npmCli, "mcp"], [
    { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "channel-check", version: "0" } } },
    { jsonrpc: "2.0", method: "notifications/initialized" },
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: path.resolve(fileCases[1].path), output_format: "envelope" } } },
    { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: path.resolve(fileCases[0].path), output_format: "envelope" } } },
    { jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: path.resolve(fileCases[2].path), output_format: "envelope", scan: "structure" } } },
  ]);
  assert.equal(mcpFrames.length, 4, "installed npm MCP transport must not answer notifications");
  const mcpById = new Map(mcpFrames.map((frame) => [frame.id, frame]));
  assert.equal(mcpById.get(1).result.serverInfo.name, "deepbom", "installed npm MCP initialization failed");
  assert.equal(mcpById.get(1).result.protocolVersion, "2025-11-25", "installed npm MCP protocol negotiation drifted");
  const mcpAudit = json(mcpById.get(2).result.content[0].text);
  assert.equal(mcpAudit.schema, "deepbom.artifact_evidence_envelope.v1", "installed npm MCP audit contract diverged");
  assert.equal(mcpAudit.identity.sha256, createHash("sha256").update(readFileSync(fileCases[1].path)).digest("hex"),
    "installed npm MCP audit identity diverged");
  const mcpTflite = json(mcpById.get(3).result.content[0].text);
  assert.equal(mcpTflite.identity.format, "tflite", "installed npm MCP could not reach its packaged TFLite WASM runtime");
  assert.equal(mcpTflite.identity.sha256, createHash("sha256").update(readFileSync(fileCases[0].path)).digest("hex"),
    "installed npm MCP TFLite identity diverged");
  const mcpGguf = json(mcpById.get(4).result.content[0].text);
  assert.equal(mcpGguf.identity.format, "gguf", "installed npm MCP bounded GGUF scan failed");
  assert.equal(mcpGguf.identity.sha256, createHash("sha256").update(readFileSync(fileCases[2].path)).digest("hex"),
    "installed npm MCP GGUF identity diverged");
}
if (platformSmoke || releaseContract) {
  assert.deepEqual(json(run(engine, capabilityArgs).stdout), canonicalCapabilities,
    "standalone engine capability discovery diverged from canonical CLI");
  assert.deepEqual(json(run(python, ["-m", "deepbom", ...capabilityArgs]).stdout), canonicalCapabilities,
    "installed Python capability discovery diverged from canonical CLI");
  assert.deepEqual(json(run(python, ["-c", "import json, deepbom; print(json.dumps(deepbom.capabilities(), separators=(',', ':')))"]).stdout), canonicalCapabilities,
    "installed Python facade capability discovery diverged from canonical CLI");
  const pythonTensorInventory = json(run(python, ["-c", "import json, sys, deepbom; print(json.dumps(deepbom.tensor_inventory(sys.argv[1]), separators=(',', ':')))", fileCases[2].path]).stdout);
  assert.equal(pythonTensorInventory.schema, "deepbom.tensor_table.v1", "installed Python tensor facade schema drifted");
  assert.equal(pythonTensorInventory.scan_policy?.effective_mode, "structure", "installed Python tensor facade must remain bounded");
  assert.equal(run(python, ["-c", "import sys, deepbom; from decimal import Decimal; rows=deepbom.tensors(sys.argv[1]); assert isinstance(rows[0]['element_count'], int); assert isinstance(rows[0]['effective_bits_per_element'], Decimal); print('typed')", fileCases[2].path]).stdout.trim(), "typed",
    "installed Python tensors facade must expose native exact numeric types");
  assert.equal(json(run(python, ["-c", "import json, sys, deepbom; print(json.dumps(deepbom.audit(sys.argv[1], sections=['summary']), separators=(',', ':')))", fileCases[1].path]).stdout).schema,
    "deepbom.analysis_selection.v1", "installed Python sections must select analysis when output is omitted");
  assert.deepEqual(json(run(python, ["-c", [
    "import json, sys, deepbom",
    "try:",
    "    deepbom.audit(sys.argv[1], gate='defects')",
    "except deepbom.DeepBomPolicyBlocked as error:",
    "    print(json.dumps({'exit_code': error.exit_code, 'schema': error.document.get('schema'), 'finding': any(row.get('id') == 'EA-SER-0001' for row in error.document.get('findings', []))}, separators=(',', ':')))",
    "else:",
    "    raise AssertionError('defect gate did not block')",
  ].join("\n"), fixtures.nan]).stdout), {
    exit_code: 2, schema: "deepbom.artifact_evidence_envelope.v1", finding: true,
  }, "installed Python defect gate must raise DeepBomPolicyBlocked with the completed document");
  assert.deepEqual(json(run(engine, agentCapabilityArgs).stdout), canonicalAgentCapabilities,
    "standalone engine agent capability discovery diverged from canonical CLI");
  assert.deepEqual(json(run(python, ["-m", "deepbom", ...agentCapabilityArgs]).stdout), canonicalAgentCapabilities,
    "installed Python agent capability discovery diverged from canonical CLI");
  assert.equal(json(run(engine, ["self-test", "--compact"]).stdout).status, "pass",
    "standalone engine self-test failed");
  assert.equal(json(run(python, ["-m", "deepbom", "self-test", "--compact"]).stdout).status, "pass",
    "installed Python wheel self-test failed");
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
    const cargoAgentCapabilities = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", ...agentCapabilityArgs], {
      DEEPBOM_ENGINE: engine,
      DEEPBOM_ENGINE_SHA256: createHash("sha256").update(await readFile(engine)).digest("hex"),
      DEEPBOM_RUNTIME_ASSET_DIR: path.join(path.dirname(engine), "pkg"),
    });
    assert.deepEqual(json(cargoAgentCapabilities.stdout), canonicalAgentCapabilities, "Cargo agent capability discovery diverged");
    const cargoSelfTest = run("cargo", ["run", "--quiet", "--manifest-path", cargoManifest, "--", "self-test", "--compact"], {
      DEEPBOM_ENGINE: engine,
      DEEPBOM_ENGINE_SHA256: createHash("sha256").update(await readFile(engine)).digest("hex"),
      DEEPBOM_RUNTIME_ASSET_DIR: path.join(path.dirname(engine), "pkg"),
    });
    assert.equal(json(cargoSelfTest.stdout).status, "pass", "Cargo launcher self-test failed");
    const unboundCargo = run(
      "cargo",
      ["run", "--quiet", "--manifest-path", cargoManifest, "--", "audit", cases[1].path, "--compact"],
      { DEEPBOM_ENGINE: engine },
      false,
    );
    assert.notEqual(unboundCargo.status, 0);
    assert.match(unboundCargo.stderr, /DEEPBOM_ENGINE_SHA256/);
  }

  const npmSelfTest = path.join(path.dirname(npmCli), "deepbom-self-test.onnx");
  const originalNpmSelfTest = await readFile(npmSelfTest);
  await corruptLastByte(npmSelfTest);
  const corruptNpmSelfTest = runNpmExecutable(npmCli, ["self-test", "--compact"], false);
  assert.notEqual(corruptNpmSelfTest.status, 0);
  assert.match(corruptNpmSelfTest.stderr, /self-test probe failed its release SHA-256 check/);
  await writeFile(npmSelfTest, originalNpmSelfTest);

  const npmWasm = path.join(path.dirname(npmCli), "..", "pkg", "tflite_wasm_audit_bg.wasm");
  await corruptLastByte(npmWasm);
  const corruptNpm = run(process.execPath, [npmCli, "audit", cases[0].path, "--compact"], {}, false);
  assert.notEqual(corruptNpm.status, 0);
  assert.match(corruptNpm.stderr, /release SHA-256 check/);

  if (python) {
    const installedRoot = run(python, ["-c", "import pathlib,deepbom;print(pathlib.Path(deepbom.__file__).parent)"]).stdout.trim();
    const installedWasm = path.join(installedRoot, "_engine", "pkg", "tflite_wasm_audit_bg.wasm");
    const originalWasm = await readFile(installedWasm);
    await corruptLastByte(installedWasm);
    const corruptPip = run(python, ["-m", "deepbom", "--version"], {}, false);
    assert.notEqual(corruptPip.status, 0);
    assert.match(corruptPip.stderr, /failed its SHA-256 check/);
    await writeFile(installedWasm, originalWasm);
    await corruptLastByte(path.join(installedRoot, "_engine", "deepbom-self-test.onnx"));
    const corruptPipSelfTest = run(python, ["-m", "deepbom", "--version"], {}, false);
    assert.notEqual(corruptPipSelfTest.status, 0);
    assert.match(corruptPipSelfTest.stderr, /failed its SHA-256 check/);
  }

  assert.equal(manifest.channels.cargo.status, "launcher_ready_for_immutable_engine_matrix");
  const cargoStatus = releaseContract ? "Cargo execution and unbound-engine rejection" : "Cargo execution reserved for --release-contract";
  const nativeStatus = releaseContract ? "standalone/Python TFLite and ONNX execution parity" : "native/Python execution reserved for platform release smoke";
  console.log(`Channel equivalence passed (installed npm tarball across five formats and two package forms; installed MCP audit; clean-agent integration; ${nativeStatus}; CLI and agent capability/envelope/SARIF/policy and verify/diff/explore parity; ${cargoStatus}; packaged self-test/WASM tamper rejection).`);
}

async function verifyInstalledAgentIntegration(npmCli, version) {
  const directory = path.join(releaseRoot, "install-probe", `agent-${installProbeRunId}`);
  await mkdir(directory, { recursive: true });
  const preview = json(run(process.execPath, [npmCli, "integrate", "codex", "--compact"], {}, true, directory).stdout);
  assert.equal(preview.schema, "deepbom.agent_integration.v1");
  assert.equal(preview.status, "changes_pending");
  assert.equal(preview.applied, false);
  const installed = json(run(process.execPath, [npmCli, "integrate", "codex", "--apply", "--compact"], {}, true, directory).stdout);
  assert.equal(installed.status, "installed");
  assert.equal(installed.applied, true);
  const skill = await readFile(path.join(directory, ".agents", "skills", "deepbom", "SKILL.md"), "utf8");
  assert.match(skill, new RegExp(`deepbom@${version.replaceAll(".", "\\.")}`));
  const status = json(run(process.execPath, [npmCli, "integrate", "status", "codex", "--compact"], {}, true, directory).stdout);
  assert.equal(status.integrations[0].status, "current");
  const removal = json(run(process.execPath, [npmCli, "integrate", "remove", "codex", "--apply", "--compact"], {}, true, directory).stdout);
  assert.equal(removal.status, "removed");
  assert.equal(existsSync(path.join(directory, ".agents", "skills", "deepbom")), false);
}

async function installNpmPackage(release) {
  const directory = path.join(releaseRoot, "install-probe", `npm-${installProbeRunId}`);
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
  const directory = path.join(releaseRoot, "install-probe", `python-${installProbeRunId}`);
  run("python", ["-m", "venv", directory]);
  const python = path.join(directory, process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const wheel = path.join(releaseRoot, release.channels.python.path);
  run(python, ["-m", "pip", "install", "--disable-pip-version-check", "--no-deps", wheel]);
  const moduleVersion = run(python, ["-m", "deepbom", "--version"]);
  assert.equal(moduleVersion.stdout.trim(), release.version);
  assert.equal(moduleVersion.stderr, "", "python -m deepbom must not emit an import-cycle RuntimeWarning");
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
  const nan = path.join(root, "nan-weight.safetensors");
  const headerSource = Buffer.from(JSON.stringify({ weight: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }), "utf8");
  const padding = (8 - (headerSource.length % 8)) % 8;
  const nanHeader = Buffer.concat([headerSource, Buffer.alloc(padding, 0x20)]);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(nanHeader.length));
  const payload = Buffer.alloc(8);
  payload.writeFloatLE(Number.NaN, 0);
  payload.writeFloatLE(1, 4);
  await writeFile(nan, Buffer.concat([prefix, nanHeader, payload]));
  return { mlpackage, sharded, nan };
}

async function corruptLastByte(file) {
  const bytes = new Uint8Array(await readFile(file));
  assert(bytes.byteLength > 0, `${file}: cannot corrupt empty file`);
  bytes[bytes.length - 1] ^= 0xff;
  await writeFile(file, bytes);
}

function run(command, args, environment = {}, expectSuccess = true, cwd = root) {
  const label = `${path.basename(command)} ${args.slice(0, 4).join(" ")}`;
  const startedAt = Date.now();
  console.log(`[channel-check] start ${label}`);
  let result;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    result = spawnSync(command, args, {
      cwd,
      encoding: "utf8",
      env: { ...process.env, ...environment },
      maxBuffer: 256 * 1024 * 1024,
      timeout: commandTimeoutMs,
    });
    if (!expectSuccess || result.status === 0 || !isTransientWindowsModuleReadFailure(result) || attempt === 1) break;
    console.log(`[channel-check] retry ${label} after transient Windows module read failure`);
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 500);
  }
  const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1);
  console.log(`[channel-check] end ${label} (${elapsedSeconds}s, status=${result.status ?? result.signal ?? "unknown"})`);
  if (result.error?.code === "ETIMEDOUT") {
    throw new Error(`${command} ${args.join(" ")} exceeded ${commandTimeoutMs} ms`);
  }
  if (expectSuccess && result.status !== 0) {
    const executionError = result.error ? `\n${result.error.stack || result.error.message}` : "";
    const signal = result.signal ? `\nsignal=${result.signal}` : "";
    throw new Error(`${command} ${args.join(" ")} failed\n${result.stdout}\n${result.stderr}${executionError}${signal}`);
  }
  return result;
}

function runNpmExecutable(npmCli, args, expectSuccess = true) {
  const installRoot = path.resolve(path.dirname(npmCli), "..", "..", "..");
  const invocation = resolveNpmCommand(["exec", "--", "deepbom", ...args]);
  return run(invocation.command, invocation.args, {}, expectSuccess, installRoot);
}

function exchangeMcp(command, args, messages) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    env: process.env,
    input: `${messages.map((message) => JSON.stringify(message)).join("\n")}\n`,
    maxBuffer: 256 * 1024 * 1024,
    timeout: mcpTimeoutMs,
  });
  if (result.error?.code === "ETIMEDOUT") throw new Error(`installed npm MCP exchange exceeded ${mcpTimeoutMs} ms`);
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed\n${result.stdout}\n${result.stderr}`);
  if (result.stderr.trim()) throw new Error(`installed npm MCP wrote to stderr: ${result.stderr.trim()}`);
  return result.stdout.split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function json(source) {
  return JSON.parse(source.replace(/^\uFEFF/, ""));
}

function boundedPositiveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function isTransientWindowsModuleReadFailure(result) {
  if (process.platform !== "win32") return false;
  const diagnostic = `${result.error?.message || ""}\n${result.stderr || ""}`;
  return /ERR_MODULE_NOT_FOUND|UNKNOWN: unknown error, (?:open|stat)/.test(diagnostic);
}
