import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicSurfaces = [
  "web/index.html",
  "web/evaluate/regulatory/index.html",
  "web/evaluate/quality/index.html",
  "web/evaluate/engineering/index.html",
  "web/for-agents/index.html",
  "bin/deepbom.mjs",
  "bin/deepbom-automation.mjs",
  "bin/deepbom-agent-contract.mjs",
  "bin/deepbom-agent-integration.mjs",
  "bin/deepbom-agent-skill.mjs",
  "docs/PUBLIC_README.md",
  "docs/CLI_REFERENCE.md",
  "channels/npm/README.md",
  "channels/python/README.md",
  "channels/cargo/README.md",
  "skills/deepbom/SKILL.md",
  "skills/deepbom/references/capability-selection.md",
  "skills/deepbom/references/evidence-semantics.md",
  "skills/deepbom/references/failure-recovery.md",
];
const forbidden = [
  /CycloneDX\s+2\.0/i,
  /deepbom\s+perspective\b/i,
  /cyclonedx[_-]perspective/i,
  /working group|pull request/i,
];

for (const relative of publicSurfaces) {
  const source = await readFile(path.join(root, relative), "utf8");
  for (const pattern of forbidden) {
    assert.equal(pattern.test(source), false, `${relative} exposes private standards work through ${pattern}`);
  }
}

const pageBuilder = await readFile(path.join(root, "scripts/build-pages.mjs"), "utf8");
assert.match(pageBuilder, /deploymentExcludedWebPatterns/);
assert.match(pageBuilder, /cyclonedx-\(\?:20\|draft\|perspective\)/);
assert.match(pageBuilder, /jsonpath-rfc95\\d\+/);

const result = spawnSync(process.execPath, [
  "bin/deepbom.mjs",
  "audit",
  "web/samples/gpu_partition_probe.onnx",
  "--output-format",
  "cyclonedx",
  "--compact",
], { cwd: root, encoding: "utf8", maxBuffer: 128 * 1024 * 1024 });
assert.equal(result.status, 0, result.stderr || result.stdout);
const document = JSON.parse(result.stdout);
assert.equal(document.bomFormat, "CycloneDX");
assert.equal(document.specVersion, "1.7");
assert.equal(JSON.stringify(document).includes("cyclonedx_2_0"), false);

console.log("Public product boundary passed (no draft/PR activity in product surfaces; CycloneDX output fixed to 1.7).");
