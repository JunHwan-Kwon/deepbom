import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { ANALYZER_SEMANTIC_VERSION } from "../web/lib/app-config.js";
import { AGENT_CONTRACT, EVIDENCE_CONTRACT } from "../bin/deepbom-public-contract-versions.mjs";
import { CHATGPT_MCP_CONTRACT } from "../worker/chatgpt-mcp.js";

const page = await readFile("web/chatgpt/index.html", "utf8");
const privacy = await readFile("web/legal/privacy.html", "utf8");
const terms = await readFile("web/legal/terms.html", "utf8");
const support = await readFile("web/legal/support.html", "utf8");
const icon = await readFile("web/chatgpt/deepbom-app-icon.svg", "utf8");
const directoryIcon = await readFile("docs/chatgpt-app/assets/deepbom-directory-icon-512.png");
const composerIcon = await readFile("docs/chatgpt-app/assets/deepbom-composer-icon-128.png");
const readme = await readFile("docs/chatgpt-app/README.md", "utf8");
const profile = JSON.parse(await readFile("docs/chatgpt-app/submission-profile.json", "utf8"));
const evals = JSON.parse(await readFile("docs/chatgpt-app/evals.json", "utf8"));
const portalSubmission = JSON.parse(await readFile("chatgpt-app-submission.json", "utf8"));
const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const forbiddenStandardsText = /CycloneDX\s*2\.0|cyclonedx-20|WG activity|pullrequestreview-|(?:issues|pull)\/\d{2,}/i;

assert.equal(ANALYZER_SEMANTIC_VERSION, packageDocument.version);
assert.match(page, /rel="canonical" href="https:\/\/deepbom\.org\/chatgpt\/"/);
assert.match(page, /application\/ld\+json/);
assert.match(page, /https:\/\/deepbom\.org\/mcp/);
assert.match(page, /deepbom-app-icon\.svg/);
assert.match(page, /Public discovery in ChatGPT begins only after OpenAI plugin review and listing/);
for (const prompt of ["attached ONNX model", "tensor encodings", "runtime latency or accuracy"]) assert.ok(page.includes(prompt));

for (const [source, label] of [[privacy, "privacy"], [terms, "terms"], [support, "support"]]) {
  assert.match(source, new RegExp(`<title>[^<]+\\| DEEPBOM</title>`), `${label} page lacks product identity`);
  assert.doesNotMatch(source, forbiddenStandardsText, `${label} page exposes standards-development material`);
}
assert.match(privacy, /does not fetch or retain the file/);
assert.match(privacy, /bounded result to the conversation/);
for (const disclosure of ["Data used and purposes", "Sharing and sale", "Retention", "User choices and controls"]) {
  assert.match(privacy, new RegExp(disclosure), `privacy page lacks ${disclosure}`);
}
assert.match(privacy, /no more than seven days/);
assert.match(privacy, /does not sell personal data or analysis results/);
assert.match(terms, /do not establish measured runtime behavior/);
assert.match(support, /Tool discovery alone is not a completed analysis/);

assert.equal(profile.schema, "deepbom.chatgpt_plugin_submission.v1");
assert.equal(profile.name, "DEEPBOM Artifact Evidence");
assert.equal(profile.package_manifest, "https://github.com/JunHwan-Kwon/deepbom/blob/main/plugin.json");
assert.equal(profile.endpoint, "https://deepbom.org/mcp");
assert.equal(profile.engine_version_at_validation, packageDocument.version);
assert.deepEqual(profile.agent_contract, AGENT_CONTRACT);
assert.deepEqual(profile.evidence_contract, EVIDENCE_CONTRACT);
assert.equal(profile.listing_status, "not_submitted_or_approved_by_this_file");
assert.equal(profile.icon_url, "https://deepbom.org/web/chatgpt/deepbom-app-icon.svg");
assert([...profile.short_description].length <= 30);
assert.match(profile.data_flow.model_bytes, /not fetched or retained/);
assert.equal(evals.schema, "deepbom.chatgpt_plugin_evals.v1");
assert(evals.positive.length >= 5);
assert(evals.negative.length >= 4);
assert(evals.answer_invariants.length >= 5);
assert(evals.positive.some((row) => row.expected_tool === "deepbom_capabilities"));
assert(evals.positive.filter((row) => row.expected_tool === "deepbom_analyze_file").length >= 4);
assert(evals.positive.filter((row) => row.class === "indirect").length >= 3);
assert(evals.positive.filter((row) => row.class === "indirect").every((row) => !/deepbom/i.test(row.prompt)));
assert(evals.negative.every((row) => row.class === "negative"));
assert.doesNotMatch(`${readme}\n${JSON.stringify(profile)}\n${JSON.stringify(evals)}`, forbiddenStandardsText);
assert.match(icon, /^<svg/);
assert.doesNotMatch(icon, /<script|javascript:|(?:href|xlink:href)=["']https?:/i, "App icon must remain a self-contained passive SVG.");
assertPng(directoryIcon, 512, "directory icon");
assertPng(composerIcon, 128, "composer icon");

const toolNames = CHATGPT_MCP_CONTRACT.tools.map((tool) => tool.name);
assert.deepEqual(toolNames, ["deepbom_analyze_file", "deepbom_capabilities", "deepbom_publish_analysis", "deepbom_publish_error"]);
for (const tool of CHATGPT_MCP_CONTRACT.tools.slice(2)) {
  assert.deepEqual(tool._meta.ui.visibility, ["app"]);
  assert.equal(tool._meta["openai/visibility"], "private");
}

assert.equal(portalSubmission.$schema, "https://developers.openai.com/plugins/schemas/chatgpt-app-submission.v1.json");
assert.equal(portalSubmission.schema_version, 1);
assert.equal(portalSubmission.app_info.display_name, profile.name);
assert.equal(portalSubmission.app_info.subtitle, profile.short_description);
assert([...portalSubmission.app_info.subtitle].length <= 30);
assert(portalSubmission.app_info.description.length <= 4000);
assert.equal(portalSubmission.app_info.category, "DEVELOPER_TOOLS");
assert.equal(portalSubmission.test_cases.length, 5);
assert.equal(portalSubmission.negative_test_cases.length, 3);
assert.deepEqual(Object.keys(portalSubmission.tools), toolNames);
for (const tool of CHATGPT_MCP_CONTRACT.tools) {
  const imported = portalSubmission.tools[tool.name];
  assert(imported, `portal import is missing ${tool.name}`);
  assert.deepEqual(imported.annotations, {
    readOnlyHint: tool.annotations.readOnlyHint,
    openWorldHint: tool.annotations.openWorldHint,
    destructiveHint: tool.annotations.destructiveHint,
  });
  for (const value of Object.values(imported.justifications)) assert.match(value, /\S/);
}
for (const testCase of portalSubmission.test_cases) {
  assert.match(testCase.description, /\S/);
  assert.match(testCase.user_prompt, /\S/);
  assert.match(testCase.tools_triggered, /\S/);
  assert.match(testCase.expected_output, /\S/);
  for (const url of testCase.file_attachment_urls || []) {
    assert.match(url, /^https:\/\/raw\.githubusercontent\.com\/JunHwan-Kwon\/deepbom\/[0-9a-f]{40}\//);
  }
}
for (const testCase of portalSubmission.negative_test_cases) {
  assert.match(testCase.description, /\S/);
  assert.match(testCase.user_prompt, /\S/);
  assert.equal(testCase.tools_triggered, null);
  assert.match(testCase.expected_output, /\S/);
}
assert.doesNotMatch(JSON.stringify(portalSubmission), forbiddenStandardsText);

console.log("ChatGPT app metadata passed (listing copy, legal/support pages, eval prompts, discovery boundary, and public/private tool visibility).\n");

function assertPng(bytes, expectedSize, label) {
  assert.deepEqual(bytes.subarray(0, 8), Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), `${label} is not PNG`);
  assert.equal(bytes.readUInt32BE(16), expectedSize, `${label} width mismatch`);
  assert.equal(bytes.readUInt32BE(20), expectedSize, `${label} height mismatch`);
  assert(bytes.length <= 5 * 1024 * 1024, `${label} exceeds 5 MiB`);
}
