import assert from "node:assert/strict";
import { mkdir, readFile, writeFile, copyFile, rm } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { resolveNpmCommand } from "./run-utils.mjs";

const root = process.cwd();
const index = process.argv.indexOf("--python");
if (index < 0 || !process.argv[index + 1]) throw new Error("Pass --python <dedicated-venv-python>; example dependencies will be installed there.");
const python = path.resolve(process.argv[index + 1]);
const run = (command, args, options = {}) => {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", timeout: 300_000, maxBuffer: 16 * 1024 * 1024, ...options });
  if (result.status !== 0) throw new Error(`${command} failed: ${result.error?.message || ""}\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
};
assert.equal(run(python, ["-c", "import sys; print(sys.prefix != sys.base_prefix)"]).trim(), "True", "Use a dedicated Python virtual environment");
const release = path.join(root, ".local-validation/channel-release");
const manifest = JSON.parse(await readFile(path.join(release, "channel-release-manifest.json"), "utf8"));
const consumer = path.join(root, ".local-validation/sdk-consumer");
const output = path.join(root, ".local-validation/sdk-examples");
await mkdir(consumer, { recursive: true });
await writeFile(path.join(consumer, "package.json"), JSON.stringify({ private: true, type: "module" }));
const artifacts = manifest.artifacts;
// Build metadata is the authority for generated platform-specific filenames.
const npmTarball = manifest.channels?.npm?.package;
const pythonArtifact = artifacts.python_wheel;
assert(npmTarball && pythonArtifact?.path, "Build npm and Python wheel artifacts first");
const npm = resolveNpmCommand(["install", "--ignore-scripts", "--prefix", consumer, path.join(release, npmTarball)]);
// Rebuilt development tarballs can retain the same version and filename.
// Do not let a prior installed copy stand in for the artifact under test.
await rm(path.join(consumer, "node_modules"), { recursive: true, force: true });
await rm(path.join(consumer, "package-lock.json"), { force: true });
run(npm.command, npm.args);
for (const file of ["index.mjs", "index.d.mts"]) {
  assert.deepEqual(await readFile(path.join(consumer, "node_modules/deepbom/sdk", file)),
    await readFile(path.join(release, "npm/package/sdk", file)), "Installed SDK must match the current artifact");
}
run(python, ["-m", "pip", "install", "--force-reinstall", path.join(release, pythonArtifact.path)]);
run(python, ["-m", "pip", "install", "-r", path.join(root, "examples/integrations/requirements.txt")]);
for (const name of ["after_export.py", "mlflow_evidence.py", "release_review.mjs"]) {
  await copyFile(path.join(root, "examples/integrations", name), path.join(consumer, name));
}
console.log(run(process.execPath, ["scripts/check-sdk.mjs", "--package", path.join(consumer, "node_modules/deepbom"), "--consumer", consumer, "--python", python]).trim());
for (const [command, name] of [[python, "after_export.py"], [python, "mlflow_evidence.py"], [process.execPath, "release_review.mjs"]]) {
  console.log(run(command, [path.join(consumer, name), output], { cwd: consumer }).trim());
}
console.log(`Three integration examples verified. Results: ${output}`);
