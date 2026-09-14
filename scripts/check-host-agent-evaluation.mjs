import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const catalogBytes = await readFile("docs/agent-evaluation/host-evaluation-cases.v1.json");
const catalog = JSON.parse(catalogBytes);
const template = JSON.parse(await readFile("docs/agent-evaluation/host-evaluation-run.template.json", "utf8"));
const readme = await readFile("docs/agent-evaluation/README.md", "utf8");
const operations = await readFile("docs/AGENT_DISTRIBUTION_OPERATIONS.md", "utf8");

assert.equal(catalog.schema, "deepbom.host_agent_evaluation_cases.v1");
assert.equal(catalog.version, packageDocument.version);
assert(catalog.cases.length >= 12);
assert.equal(new Set(catalog.cases.map((row) => row.id)).size, catalog.cases.length);
assert.deepEqual(new Set(catalog.cases.map((row) => row.class)), new Set(["direct", "indirect", "negative"]));
for (const surface of ["chatgpt_remote_attachment", "claude_remote_browser_mcp_app", "claude_desktop_local_mcpb", "codex_local_skill"]) {
  assert(catalog.cases.some((row) => row.surface === surface && row.class === "indirect"), `Missing indirect case for ${surface}`);
  assert(catalog.cases.some((row) => row.surface === surface && row.class === "negative"), `Missing negative case for ${surface}`);
}
assert(catalog.cases.filter((row) => row.class === "indirect").every((row) => !/deepbom/i.test(row.prompt)));
assert(catalog.answer_invariants.length >= 5);
const chatGptCases = catalog.cases.filter((row) => row.surface === "chatgpt_remote_attachment");
assert(chatGptCases.filter((row) => row.class !== "negative").length >= 5, "ChatGPT submission requires at least five positive cases.");
assert(chatGptCases.filter((row) => row.class === "negative").length >= 3, "ChatGPT submission requires at least three negative cases.");
const claudeRemoteCases = catalog.cases.filter((row) => row.surface === "claude_remote_browser_mcp_app");
assert(claudeRemoteCases.filter((row) => row.class !== "negative").length >= 5, "Claude remote qualification requires at least five positive cases in the local catalog.");
assert(claudeRemoteCases.filter((row) => row.class === "negative").length >= 3, "Claude remote qualification requires at least three negative cases in the local catalog.");

assert.equal(template.schema, "deepbom.host_agent_evaluation_run.v1");
assert.equal(template.status, "not_run");
assert.equal(template.deepbom_version, packageDocument.version);
assert.deepEqual(template.observations, []);
assert.equal(template.case_catalog_sha256, null);
assert.match(template.interpretation_boundary, /not execution evidence/);
assert.match(readme, /not evidence that ChatGPT/);
assert.match(readme, /Do not replace the template with invented/);
assert.match(operations, /ChatGPT developer-mode validation/);
assert.match(operations, /Claude Desktop validation/);
assert.match(operations, /Search discovery readback/);
assert.match(operations, /not a legal clearance determination/);
assert.match(operations, new RegExp(`deepbom@${packageDocument.version}`));

console.log(`Host-agent evaluation package passed (${catalog.cases.length} direct, brandless indirect, and negative cases; blank non-evidence template; catalog SHA-256 ${sha256(catalogBytes)}).\n`);

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
