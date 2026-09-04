import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const fixtureRoot = path.join(root, ".local-validation", "cargo-engine-release-check");
const input = path.join(fixtureRoot, "input");
const output = path.join(fixtureRoot, "output");
const version = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;
const identities = [
  ["windows-x64", "win32", "x64", "deepbom-core.exe"],
  ["windows-arm64", "win32", "arm64", "deepbom-core.exe"],
  ["linux-x64", "linux", "x64", "deepbom-core"],
  ["linux-arm64", "linux", "arm64", "deepbom-core"],
  ["macos-x64", "darwin", "x64", "deepbom-core"],
  ["macos-arm64", "darwin", "arm64", "deepbom-core"],
];

await rm(fixtureRoot, { recursive: true, force: true });
const wasm = Buffer.from("deterministic-wasm-fixture");
const selfTest = Buffer.from("deterministic-self-test-probe");
for (const [id, platform, arch, executableName] of identities) {
  const directory = path.join(input, id);
  await mkdir(path.join(directory, "pkg"), { recursive: true });
  const executable = Buffer.from(`deterministic-engine-${id}`);
  await writeFile(path.join(directory, executableName), executable);
  await writeFile(path.join(directory, "pkg", "tflite_wasm_audit_bg.wasm"), wasm);
  await writeFile(path.join(directory, "deepbom-self-test.onnx"), selfTest);
  await writeFile(path.join(directory, "manifest.json"), `${JSON.stringify({
    schema: "deepbom.packaged_engine.v1",
    version,
    platform,
    arch,
    executable: record(executableName, executable),
    tflite_wasm: record("pkg/tflite_wasm_audit_bg.wasm", wasm),
    self_test: record("deepbom-self-test.onnx", selfTest),
    source: { git_commit: "c".repeat(40), git_state: "clean" },
  }, null, 2)}\n`);
}

const successful = runBuilder();
assert.equal(successful.status, 0, successful.stderr || successful.stdout);
const matrix = JSON.parse(await readFile(path.join(output, "engine-matrix.v1.json"), "utf8"));
assert.equal(matrix.schema, "deepbom.engine_matrix.v1");
assert.equal(matrix.version, version);
assert.equal(matrix.source.tag, `channels-v${version}`);
assert.deepEqual(matrix.targets.map((target) => target.id).sort(), identities.map(([id]) => id).sort());
assert.equal(matrix.self_test.filename, `deepbom-self-test-${version}.onnx`);
assert.equal(matrix.self_test.byte_length, selfTest.byteLength);
assert.equal(matrix.self_test.sha256, createHash("sha256").update(selfTest).digest("hex"));
assert.equal((await readdir(output)).length, 10);
assert.equal((await readFile(path.join(output, "SHA256SUMS"), "utf8")).trim().split(/\r?\n/).length, 9);

const tamperedManifest = path.join(input, "linux-x64", "manifest.json");
const tampered = JSON.parse(await readFile(tamperedManifest, "utf8"));
tampered.executable.sha256 = "0".repeat(64);
await writeFile(tamperedManifest, `${JSON.stringify(tampered, null, 2)}\n`);
const rejected = runBuilder();
assert.notEqual(rejected.status, 0, "A tampered engine manifest must be rejected.");
tampered.executable.sha256 = createHash("sha256").update(Buffer.from("deterministic-engine-linux-x64")).digest("hex");
tampered.self_test.sha256 = "0".repeat(64);
await writeFile(tamperedManifest, `${JSON.stringify(tampered, null, 2)}\n`);
const rejectedSelfTest = runBuilder();
assert.notEqual(rejectedSelfTest.status, 0, "A tampered self-test manifest must be rejected.");
console.log("Cargo engine release matrix passed (six targets, deterministic inventory, checksums, and engine/self-test tamper rejection).");

function runBuilder() {
  return spawnSync(process.execPath, ["scripts/build-cargo-engine-release.mjs", "--input", input, "--output", output, "--version", version], {
    cwd: root,
    encoding: "utf8",
  });
}

function record(file, bytes) {
  return { path: file, byte_length: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
}
