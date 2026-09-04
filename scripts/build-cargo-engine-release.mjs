import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const input = path.resolve(argument("--input") || path.join(root, ".local-validation", "channel-engines"));
const output = path.resolve(argument("--output") || path.join(root, ".local-validation", "cargo-engine-release"));
const expectedVersion = argument("--version") || JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;
const expectedTag = `channels-v${expectedVersion}`;
const expectedTargets = ["windows-x64", "windows-arm64", "linux-x64", "linux-arm64", "macos-x64", "macos-arm64"];
const manifests = await findFiles(input, "manifest.json");

assert.equal(manifests.length, expectedTargets.length, `Expected ${expectedTargets.length} packaged-engine manifests, found ${manifests.length}.`);
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

const targets = [];
let sourceIdentity = null;
let wasmRecord = null;
let wasmSource = null;
let selfTestRecord = null;
let selfTestSource = null;
for (const manifestPath of manifests) {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  assert.equal(manifest.schema, "deepbom.packaged_engine.v1");
  assert.equal(manifest.version, expectedVersion);
  assert.equal(manifest.source?.git_state, "clean", `${manifestPath}: engine source must be clean`);
  assert.match(manifest.source?.git_commit || "", /^[0-9a-f]{40}$/);
  sourceIdentity ||= manifest.source;
  assert.deepEqual(manifest.source, sourceIdentity, `${manifestPath}: source identity drift`);

  const identity = releaseIdentity(manifest.platform, manifest.arch);
  const directory = path.dirname(manifestPath);
  const executableSource = path.join(directory, manifest.executable?.path || "");
  const currentWasmSource = path.join(directory, manifest.tflite_wasm?.path || "");
  const currentSelfTestSource = path.join(directory, manifest.self_test?.path || "");
  await verifyRecord(executableSource, manifest.executable);
  await verifyRecord(currentWasmSource, manifest.tflite_wasm);
  await verifyRecord(currentSelfTestSource, manifest.self_test);

  const executableName = `deepbom-core-${identity}${identity.startsWith("windows-") ? ".exe" : ""}`;
  await copyFile(executableSource, path.join(output, executableName));
  targets.push({
    id: identity,
    platform: identity.split("-")[0],
    arch: identity.split("-")[1],
    executable: await fileRecord(path.join(output, executableName), executableName),
  });

  if (!wasmRecord) {
    wasmRecord = manifest.tflite_wasm;
    wasmSource = currentWasmSource;
  } else {
    assert.equal(manifest.tflite_wasm.sha256, wasmRecord.sha256, `${identity}: TFLite WASM digest drift`);
    assert.equal(manifest.tflite_wasm.byte_length, wasmRecord.byte_length, `${identity}: TFLite WASM size drift`);
  }
  if (!selfTestRecord) {
    selfTestRecord = manifest.self_test;
    selfTestSource = currentSelfTestSource;
  } else {
    assert.equal(manifest.self_test.sha256, selfTestRecord.sha256, `${identity}: self-test probe digest drift`);
    assert.equal(manifest.self_test.byte_length, selfTestRecord.byte_length, `${identity}: self-test probe size drift`);
  }
}

targets.sort((left, right) => left.id.localeCompare(right.id));
assert.deepEqual(targets.map((target) => target.id).sort(), [...expectedTargets].sort(), "Engine target matrix is incomplete.");
const wasmName = `tflite_wasm_audit_bg-${expectedVersion}.wasm`;
await copyFile(wasmSource, path.join(output, wasmName));
const selfTestName = `deepbom-self-test-${expectedVersion}.onnx`;
await copyFile(selfTestSource, path.join(output, selfTestName));
const matrix = {
  schema: "deepbom.engine_matrix.v1",
  version: expectedVersion,
  source: { git_commit: sourceIdentity.git_commit, git_state: sourceIdentity.git_state, tag: expectedTag },
  wasm: await fileRecord(path.join(output, wasmName), wasmName),
  self_test: await fileRecord(path.join(output, selfTestName), selfTestName),
  targets,
};
await writeFile(path.join(output, "engine-matrix.v1.json"), `${JSON.stringify(matrix, null, 2)}\n`);

const assetNames = (await readdir(output)).sort();
const checksums = [];
for (const name of assetNames) {
  const record = await fileRecord(path.join(output, name), name);
  checksums.push(`${record.sha256}  ${name}`);
}
await writeFile(path.join(output, "SHA256SUMS"), `${checksums.join("\n")}\n`);

const finalNames = (await readdir(output)).sort();
assert.equal(finalNames.length, 10, "Cargo engine release must contain six engines, one WASM, one self-test probe, one matrix, and SHA256SUMS.");
console.log(`Built ${expectedVersion} Cargo engine release (${targets.length} targets) at ${output}`);

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

async function findFiles(directory, filename) {
  const results = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) results.push(...await findFiles(candidate, filename));
    else if (entry.isFile() && entry.name === filename) results.push(candidate);
  }
  return results;
}

function releaseIdentity(platform, architecture) {
  const system = { win32: "windows", linux: "linux", darwin: "macos" }[platform];
  const arch = { x64: "x64", arm64: "arm64" }[architecture];
  assert(system && arch, `Unsupported engine identity ${platform}/${architecture}.`);
  return `${system}-${arch}`;
}

async function verifyRecord(file, record) {
  assert(record && typeof record === "object");
  const observed = await fileRecord(file, path.basename(file));
  assert.equal(observed.byte_length, record.byte_length, `${file}: byte length mismatch`);
  assert.equal(observed.sha256, record.sha256, `${file}: SHA-256 mismatch`);
}

async function fileRecord(file, filename) {
  const metadata = await stat(file);
  assert(metadata.isFile(), `${file}: expected a regular file`);
  const bytes = await readFile(file);
  return { filename, byte_length: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
}
