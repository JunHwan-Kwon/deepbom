import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import AdmZip from "adm-zip";
import { resolveNpmCommand } from "./run-utils.mjs";

const root = process.cwd();
const releaseRoot = path.join(root, ".local-validation", "channel-release");
const releaseManifest = JSON.parse(await readFile(path.join(releaseRoot, "channel-release-manifest.json"), "utf8"));
const packageRoot = path.join(releaseRoot, "mcpb", "package");
const bundlePath = path.join(releaseRoot, releaseManifest.channels.mcpb.path);
assert(existsSync(bundlePath), `Missing MCPB bundle: ${bundlePath}`);

const manifestPath = path.join(packageRoot, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
assert.equal(manifest.manifest_version, "0.4");
assert.equal(manifest.name, "deepbom");
assert.equal(manifest.version, releaseManifest.version);
assert.equal(manifest.server.type, "node");
assert.deepEqual(manifest.server.mcp_config.args, ["${__dirname}/server/bin/deepbom.mjs", "mcp"]);
assert.equal(manifest.server.mcp_config.env.DEEPBOM_MCP_ALLOWED_ROOTS, "${user_config.allowed_root}");
assert.deepEqual(manifest.tools.map((tool) => tool.name), [
  "deepbom_capabilities",
  "deepbom_audit",
  "deepbom_diff",
  "deepbom_explain_rule",
]);

const zip = new AdmZip(bundlePath);
const entries = zip.getEntries().filter((entry) => !entry.isDirectory).map((entry) => entry.entryName).sort();
assert.deepEqual(entries, [
  "LICENSE",
  "README.md",
  "manifest.json",
  "server/bin/deepbom-self-test.onnx",
  "server/bin/deepbom.mjs",
  "server/pkg/release-manifest.json",
  "server/pkg/tflite_wasm_audit_bg.wasm",
]);
for (const relative of [
  "bin/deepbom.mjs",
  "bin/deepbom-self-test.onnx",
  "pkg/release-manifest.json",
  "pkg/tflite_wasm_audit_bg.wasm",
]) {
  const npmBytes = await readFile(path.join(releaseRoot, "npm", "package", relative));
  const bundleBytes = zip.getEntry(`server/${relative}`).getData();
  assert.equal(sha256(bundleBytes), sha256(npmBytes), `MCPB member diverged from npm: ${relative}`);
}
assert.equal(sha256(await readFile(bundlePath)), releaseManifest.artifacts.mcpb.sha256);

const selfTest = spawnSync(process.execPath, [path.join(packageRoot, "server", "bin", "deepbom.mjs"), "self-test", "--compact"], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 32 * 1024 * 1024,
});
assert.equal(selfTest.status, 0, selfTest.stderr || selfTest.stdout);
assert.equal(JSON.parse(selfTest.stdout).status, "pass");

if (!process.argv.includes("--offline")) {
  const toolVersion = process.env.DEEPBOM_MCPB_TOOL_VERSION || "2.1.2";
  const invocation = resolveNpmCommand(["exec", "--yes", `--package=@anthropic-ai/mcpb@${toolVersion}`, "--", "mcpb", "validate", manifestPath]);
  const validation = spawnSync(invocation.command, invocation.args, { cwd: root, encoding: "utf8", timeout: 120_000 });
  assert.equal(validation.status, 0, `${validation.stdout}\n${validation.stderr}`);
}

console.log("MCPB bundle passed (official manifest validation, exact npm-member hashes, local self-test, and four-tool declaration).");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
