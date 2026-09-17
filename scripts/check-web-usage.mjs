import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, "http://127.0.0.1").pathname;
    if (pathname === "/favicon.ico") return response.writeHead(204).end();
    const relative = pathname === "/" ? "/web/index.html" : pathname;
    const resolved = path.resolve(ROOT, `.${decodeURIComponent(relative)}`);
    if (!resolved.startsWith(`${ROOT}${path.sep}`)) throw new Error("outside root");
    const bytes = await readFile(resolved);
    const type = relative.endsWith(".html") ? "text/html; charset=utf-8"
      : relative.endsWith(".css") ? "text/css; charset=utf-8"
        : relative.endsWith(".js") || relative.endsWith(".mjs") ? "text/javascript; charset=utf-8"
          : relative.endsWith(".json") ? "application/json; charset=utf-8"
            : relative.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
    response.writeHead(200, { "content-type": type });
    response.end(bytes);
  } catch {
    response.writeHead(404).end("not found");
  }
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const context = await browser.newContext({ acceptDownloads: true, serviceWorkers: "block" });
  const page = await context.newPage();
  const requests = [], errors = [];
  let outage = false, sessions = 0;
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/usage/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON();
    requests.push({ path, body });
    if (outage) return route.fulfill({ status: 503, json: { error: "test outage" } });
    await route.fulfill({ status: 200, json: path.endsWith("/session") ? { token: `test-session-${++sessions}` } : { ok: true } });
  });
  const origin = process.env.DEEPBOM_WEB_USAGE_ORIGIN || `http://127.0.0.1:${server.address().port}`;
  await page.goto(`${origin}/web/index.html`, { waitUntil: "domcontentloaded" });
  await page.locator("#dropzone").dispatchEvent("pointerdown");
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.childElementCount > 0);
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
    await page.locator("#agreementBackdrop").waitFor({ state: "hidden" });
  }
  const waitForAudit = async () => {
    await page.waitForFunction(() => /audit run complete|audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
    assert.doesNotMatch(await page.locator("#status").innerText(), /audit failed/i);
  };
  await page.locator("#sampleModelSelect").selectOption("gguf-tinymqa-q4");
  await page.locator("#trySampleModel").click();
  await waitForAudit();
  assert.equal(requests.length, 0, "No usage request before consent, even for a complete analysis");
  assert.equal(await page.evaluate(() => localStorage.getItem("deepbom.web.usage.v1")), null);
  const controls = page.locator("#webUsageControls");
  await controls.locator("summary").click();
  await controls.locator('[data-action="usage-test"]').check();
  await controls.locator('[data-action="usage-consent"]').check();
  await page.waitForFunction(() => /Sharing is on/.test(document.querySelector("#webUsageControls [role=status]")?.textContent || ""));
  const events = () => requests.filter(r => r.path.endsWith("/events")).flatMap(r => r.body.events.map(e => ({ ...e, token: r.body.token })));
  const waitFor = async (predicate) => { for (let n=0;n<100;n++) { if (predicate()) return; await new Promise(r => setTimeout(r, 50)); } assert(predicate(), "Expected usage event"); };
  await waitFor(() => events().some(e => e.event === "analysis_completed"));
  assert.equal(sessions, 1);
  assert.equal(requests.find(r => r.path.endsWith("/session")).body.channel, "web");
  assert.equal(requests.find(r => r.path.endsWith("/session")).body.cohort, "test");
  assert.equal(await page.evaluate(() => localStorage.getItem("deepbom.chatgpt.usage.v1")), null, "Channels have separate consent and keys");
  await page.locator('[data-workflow-step="output"]').click();
  await page.locator('[data-module-tab="export_contracts"]').click();
  await page.locator("#downloadCycloneDxEvidence").waitFor({ state: "visible" });
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 120_000 }), page.locator("#downloadCycloneDxEvidence").click()]);
  const bom = JSON.parse(await readFile(await download.path(), "utf8"));
  assert.equal(bom.bomFormat, "CycloneDX");
  await waitFor(() => events().some(e => e.detail === "cyclonedx"));
  await page.evaluate(async () => { window.delayedUsageExport = (await import("/web/lib/web-usage.js")).webUsage.observeExport("svg"); });
  await page.locator('[data-workflow-step="input"]').click();
  await page.locator("#runAudit").click();
  await waitForAudit();
  await waitFor(() => events().filter(e => e.event === "analysis_completed").length === 2);
  await page.evaluate(() => window.delayedUsageExport());
  assert(!events().some(e => e.detail === "svg"), "Late export is not attributed to a new analysis");
  assert.equal(sessions, 2, "A rerun creates a new analysis session");
  assert.deepEqual(events().filter(e => e.event === "analysis_completed").map(e => e.token), ["test-session-1", "test-session-2"]);
  assert.equal(events().find(e => e.detail === "cyclonedx").token, "test-session-1");
  assert.doesNotMatch(JSON.stringify(requests), /gguf-tinymqa|filename|sha256|model_bytes|download_url|email/);
  for (const event of events()) assert.deepEqual(Object.keys(event).sort(), ["detail", "event", "format", "token"]);
  await controls.locator('[data-action="usage-forget"]').click();
  await page.waitForFunction(() => localStorage.getItem("deepbom.web.usage.v1") === null);
  assert(requests.some(r => r.path.endsWith("/forget")));
  outage = true;
  await controls.locator('[data-action="usage-consent"]').check();
  await page.locator("#runAudit").click();
  await waitForAudit();
  await page.waitForFunction(() => /Analysis is unaffected/.test(document.querySelector("#webUsageControls [role=status]")?.textContent || ""));
  assert.deepEqual(errors, []);
  console.log("Web usage passed: real GGUF analysis and CycloneDX download, default-off consent, channel keys, rerun attribution, deletion and collector outage isolation.");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
