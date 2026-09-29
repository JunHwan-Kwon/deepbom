import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { build } from "esbuild";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { guideWorkflows, DEEPBOM_WORKFLOWS, workflowUrl } from "../web/lib/workflow-catalog.js";
import { parseWorkflowLink } from "../web/lib/workflow-continuation.js";
import { EVIDENCE_QUERY_SECTIONS } from "../web/lib/evidence-query-contract.js";
import { buildCliCapabilities } from "../bin/deepbom-automation.mjs";

const html = await readFile("web/index.html", "utf8");
const help = await readFile("docs/CLI_REFERENCE.md", "utf8");
const capabilities = buildCliCapabilities("2.0.0", {});
assert.equal(new Set(DEEPBOM_WORKFLOWS.map(row => row.id)).size, DEEPBOM_WORKFLOWS.length);
for (const row of DEEPBOM_WORKFLOWS) {
  const guide = guideWorkflows({ workflow: row.id });
  assert.equal(guide.workflows.length, 1);
  assert.equal(guide.status, "guidance_ready");
  if (row.chat.section) assert(EVIDENCE_QUERY_SECTIONS.includes(row.chat.section));
  for (const command of row.cli) {
    assert(capabilities.commands.some(item => item.name.split(" ")[0] === command.args[0]), `Missing CLI command: ${row.id}`);
    for (const arg of command.args.filter(arg => arg.startsWith("--"))) assert(help.includes(arg), `Undocumented CLI option: ${row.id}/${arg}`);
  }
  const url = new URL(workflowUrl(row.id));
  assert.equal(url.origin, "https://deepbom.org"); assert.equal(url.search, "");
  if (row.web) {
    assert.equal(parseWorkflowLink(url.hash)?.id, row.id);
    if (row.web.audit_tab) assert(html.includes(`data-audit-tab="${row.web.audit_tab}"`));
    assert(["input", "audit", "findings", "output", "graph", "weight", "metadata", "redesign", "runtime"].includes(row.web.workspace));
  }
}
assert.equal(guideWorkflows({ workflow: "redesign", format: "safetensors" }).status, "not_applicable");
for (const invalid of [{ workflow: "invented" }, { workflow: "weights", file: "private" }, { format: "checkpoint" }]) assert.throws(() => guideWorkflows(invalid));
for (const invalid of ["#workflow=weights&file=https://example.com/model", "#workflow=../evil", "#workflow=verify", "#workflow=WEIGHTS", "#workflow=weights%26secret"]) assert.equal(parseWorkflowLink(invalid), null);

if (existsSync("worker/chatgpt-mcp.js")) {
  const { routeChatGptMcp } = await import("../worker/chatgpt-mcp.js");
  const response = await routeChatGptMcp(new Request("https://deepbom.org/mcp", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "deepbom_workflow_guide", arguments: { workflow: "verify" } } }) }));
  const result = (await response.json()).result.structuredContent;
  assert.equal(result.workflows[0].chat.continuation, "local");
  assert.match(result.boundary, /no local command was executed/i);
}

const bundle = await build({ stdin: { contents: 'import {installWorkflowContinuation} from "./web/lib/workflow-continuation.js"; window.installContinuation=installWorkflowContinuation;', resolveDir: process.cwd() }, bundle: true, write: false, format: "iife" });
const browser = await launchChromium(chromium);
try {
  const page = await browser.newPage();
  await page.route("https://deepbom.org/**", route => route.fulfill({ contentType: "text/html", body: '<div id="host"></div>' }));
  await page.goto("https://deepbom.org/#workflow=weights");
  await page.addScriptTag({ content: bundle.outputFiles[0].text });
  await page.evaluate(() => {
    window.selectedModel = null; window.navigation = []; window.allowNavigation = true;
    window.continuation = window.installContinuation({ host: document.getElementById("host"), getAnalysis: () => window.selectedModel, navigate: id => { if (!window.allowNavigation) return false; window.navigation.push(id); return true; }, selectAuditTab: id => window.navigation.push(id) });
  });
  assert.match(await page.locator("#host").innerText(), /No|not transferred/);
  assert.equal(await page.evaluate(() => window.continuation.apply()), false);
  await page.evaluate(() => { window.selectedModel = { format: "onnx" }; window.allowNavigation = false; });
  assert.equal(await page.evaluate(() => window.continuation.apply()), false, "Respect existing navigation gates");
  await page.evaluate(() => { window.allowNavigation = true; });
  assert.equal(await page.evaluate(() => window.continuation.apply()), true);
  assert.deepEqual(await page.evaluate(() => window.navigation), ["weight"]);
  assert.equal(await page.evaluate(() => window.continuation.apply()), false, "Do not reroute subsequent audits");
  await page.evaluate(() => { window.location.hash = "workflow=redesign"; });
  await page.waitForFunction(() => document.body.textContent.includes("unavailable for the selected format"));
  assert.deepEqual(await page.evaluate(() => window.navigation), ["weight"]);
  await page.evaluate(() => { window.location.hash = "workflow=placement"; });
  await page.waitForFunction(() => window.navigation.includes("accelerator"));
  assert.deepEqual(await page.evaluate(() => window.navigation), ["weight", "audit", "accelerator"]);
} finally { await browser.close(); }
console.log("Workflow guide passed: CLI/schema route closure, fixed public links, no artifact transfer, browser continuation, format/access gates and one-shot navigation.");
