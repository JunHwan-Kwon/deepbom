import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { buildAgentCapabilities } from "../bin/deepbom-agent-contract.mjs";
import { manageAgentIntegration } from "../bin/deepbom-agent-integration.mjs";
import { buildAgentSkillFiles } from "../bin/deepbom-agent-skill.mjs";
import { buildCliCapabilities } from "../bin/deepbom-automation.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const version = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version;
const skillFiles = buildAgentSkillFiles(version);
const forbiddenProductText = /CycloneDX\s*2\.0|cyclonedx-20|WG activity|pullrequestreview-|(?:issues|pull)\/\d{2,}/i;

for (const [relative, expected] of skillFiles) {
  const observed = await readFile(path.join(root, "skills", "deepbom", relative), "utf8");
  assert.equal(observed, expected, `public Agent Skill drifted from its bundled template: ${relative}`);
  assert.doesNotMatch(observed, forbiddenProductText, `${relative} exposes standards-development material`);
}

const cliCapabilities = buildCliCapabilities(version, {
  defaultTarget: "android_mid_a55",
  deltaTargets: ["android_mid_a55", "rpi4_a72"],
});
const agentCapabilities = buildAgentCapabilities(cliCapabilities);
assert.equal(agentCapabilities.schema, "deepbom.agent_capabilities.v1");
assert.equal(agentCapabilities.version, version);
assert.equal(agentCapabilities.execution_boundary.analysis_location, "local_process");
assert.equal(agentCapabilities.execution_boundary.hosted_analysis_endpoint, false);
assert.equal(agentCapabilities.outputs.cyclonedx.spec_version, "1.7");
assert.match(agentCapabilities.invocation.discovery, new RegExp(`deepbom@${escapeRegExp(version)}`));
assert.match(agentCapabilities.local_integrations.claude_desktop.release_asset, new RegExp(`channels-v${escapeRegExp(version)}/deepbom-${escapeRegExp(version)}\\.mcpb$`));
assert.doesNotMatch(JSON.stringify(agentCapabilities), forbiddenProductText);

const cli = path.join(root, "bin", "deepbom.mjs");
const capabilityRun = runNode([cli, "capabilities", "--format", "agent-json", "--compact"]);
assert.deepEqual(JSON.parse(capabilityRun.stdout), agentCapabilities, "CLI agent capabilities drifted from the shared builder");

const temporary = await mkdtemp(path.join(os.tmpdir(), "deepbom-agent-native-"));
try {
  for (const target of ["codex", "claude-code", "generic"]) {
    const preview = await manageAgentIntegration({ action: "install", target, apply: false, root: temporary, version });
    assert.equal(preview.schema, "deepbom.agent_integration.v1");
    assert.equal(preview.status, "changes_pending");
    assert.equal(await exists(path.join(temporary, preview.destination)), false, `${target} preview wrote files`);

    const installed = await manageAgentIntegration({ action: "install", target, apply: true, root: temporary, version });
    assert.equal(installed.status, "installed");
    const status = await manageAgentIntegration({ action: "status", target, root: temporary, version });
    assert.equal(status.integrations[0].status, "current");
    assert.equal(status.integrations[0].installed_version, version);

    const skillPath = path.join(temporary, installed.destination, "SKILL.md");
    const canonical = await readFile(skillPath, "utf8");
    await writeFile(skillPath, `${canonical}\nuser change\n`);
    await assert.rejects(
      manageAgentIntegration({ action: "install", target, apply: true, root: temporary, version }),
      /overwrite unmanaged changes/,
    );
    await assert.rejects(
      manageAgentIntegration({ action: "remove", target, apply: true, root: temporary, version }),
      /Refusing to remove modified/,
    );
    await writeFile(skillPath, canonical);

    const removePreview = await manageAgentIntegration({ action: "remove", target, apply: false, root: temporary, version });
    assert.equal(removePreview.status, "changes_pending");
    assert.equal(await exists(skillPath), true, `${target} removal preview changed files`);
    const removed = await manageAgentIntegration({ action: "remove", target, apply: true, root: temporary, version });
    assert.equal(removed.status, "removed");
    assert.equal(await exists(skillPath), false, `${target} managed Skill survived removal`);
  }

  const cliPreview = runNode([cli, "integrate", "codex", "--compact"], { cwd: temporary });
  const previewDocument = JSON.parse(cliPreview.stdout);
  assert.equal(previewDocument.status, "changes_pending");
  assert.equal(await exists(path.join(temporary, ".agents", "skills", "deepbom")), false);
} finally {
  await rm(temporary, { recursive: true, force: true });
}

console.log(`Agent-native contract passed (${skillFiles.size} bundled Skill files; capability parity; preview, apply, conflict, status, and managed removal across 3 targets).`);

function runNode(args, options = {}) {
  const result = spawnSync(process.execPath, args, { cwd: options.cwd || root, encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result;
}

async function exists(file) {
  try { await access(file); return true; } catch { return false; }
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
