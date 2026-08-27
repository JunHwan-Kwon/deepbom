import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import process from "node:process";

import { chromium } from "playwright";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const serveRoot = argument("--root") ? path.resolve(argument("--root")) : "";
const server = serveRoot ? createStaticServer(serveRoot) : null;
if (server) await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const baseUrl = server
  ? new URL(`http://127.0.0.1:${server.address().port}/web/`)
  : new URL(argument("--base-url") || "https://deepbom.org/web/");
const outputDir = path.resolve(argument("--output-dir") || ".local-validation/live-product");
const auditTimeoutMs = Number(argument("--audit-timeout-ms") || 180_000);
assert(Number.isFinite(auditTimeoutMs) && auditTimeoutMs > 0, "--audit-timeout-ms must be a positive number");
await mkdir(outputDir, { recursive: true });

let browser;
try {
  const appResponse = await fetch(new URL("app.js", baseUrl), { cache: "no-store" });
  assert.equal(appResponse.status, 200);
  if (!new Set(["127.0.0.1", "localhost", "::1"]).has(baseUrl.hostname)) {
    assert.match(appResponse.headers.get("content-type") || "", /charset=utf-8/i);
  }
  const swResponse = await fetch(new URL("sw.js", baseUrl), { cache: "no-store" });
  assert.equal(swResponse.status, 200);
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const browserErrors = [];
  const badResponses = [];
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) {
      browserErrors.push(`console: ${message.text()}`);
    }
  });
  page.on("response", (response) => {
    if (response.status() >= 400) badResponses.push({ status: response.status(), url: response.url() });
  });
  await page.goto(baseUrl.href, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await acceptPrivacy(page);
  for (const width of [1440, 1180, 1024, 821, 820, 768, 620, 390, 360]) {
    await page.setViewportSize({ width, height: 1000 });
    await assertPrimaryControlsFit(page, `${width}px`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.selectOption("#sampleModelSelect", { index: 0 });
  await page.click("#trySampleModel");
  try {
    await page.waitForFunction(() => /\.tflite$/i.test(document.querySelector("#selectedModelName")?.textContent?.trim() || ""), null, { timeout: 60_000 });
  } catch (error) {
    const state = await productState(page);
    throw new Error(`Verified example was not staged: ${JSON.stringify(state)}; browser errors: ${browserErrors.join(" | ") || "none"}`, { cause: error });
  }
  await page.waitForSelector("#auditProgress:not([hidden])", { timeout: 30_000 });
  try {
    await page.waitForFunction(() => {
      const progress = document.querySelector("#auditProgressLabel")?.textContent || "";
      const status = document.querySelector("#status")?.textContent || "";
      return progress.includes("Complete") && !/failed/i.test(status) && !document.querySelector("#runAudit")?.disabled;
    }, null, { timeout: auditTimeoutMs });
  } catch (error) {
    const state = await productState(page);
    await page.screenshot({ path: path.join(outputDir, "audit-failure.png"), fullPage: true });
    throw new Error(`Audit did not complete: ${JSON.stringify(state)}; browser errors: ${browserErrors.join(" | ") || "none"}`, { cause: error });
  }
  const unexpectedResponses = badResponses.filter(({ status, url }) => {
    const pathname = new URL(url).pathname;
    if (pathname === "/favicon.ico") return false;
    return !(server && status === 404 && pathname.startsWith("/api/"));
  });
  assert.deepEqual(browserErrors, [], `Browser errors: ${browserErrors.join(" | ")}`);
  assert.deepEqual(unexpectedResponses, [], `Unexpected HTTP responses: ${JSON.stringify(unexpectedResponses)}`);

  assert.match(await page.locator("[data-capability-summary]").innerText(), /TFLite:/);
  assert.equal(await page.locator("html").getAttribute("data-analysis-depth"), "deep");
  await clickAfterScroll(page, '[data-workflow-step="graph"]');
  await clickAfterScroll(page, '[data-explorer-tab="node"]');
  assert((await page.locator("#nodeViewPanel [data-op-index], #nodeViewPanel .node-view-node").count()) > 0);
  await clickAfterScroll(page, '[data-explorer-tab="kernels"]');
  assert.equal(await page.locator("#runtimeEvidenceClosure").isVisible(), true);

  await clickAfterScroll(page, '[data-workflow-step="output"]');
  await clickAfterScroll(page, ".individual-export-menu > summary");
  const verificationButton = page.locator("#downloadPublicVerificationManifest");
  assert.equal(
    await verificationButton.isDisabled(),
    false,
    `Public verification manifest should be enabled after the audit: ${JSON.stringify(await productState(page))}`,
  );
  let download;
  try {
    [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 30_000 }),
      verificationButton.click(),
    ]);
  } catch (error) {
    const state = await productState(page);
    throw new Error(`Public verification manifest did not download: ${JSON.stringify(state)}; browser errors: ${browserErrors.join(" | ") || "none"}`, { cause: error });
  }
  const destination = path.join(outputDir, "public_report_verification_manifest.json");
  await download.saveAs(destination);
  const manifest = JSON.parse(await readFile(destination, "utf8"));
  const body = { ...manifest };
  delete body.manifest_sha256;
  assert.equal(manifest.manifest_sha256, sha256TextHex(canonicalJson(body)));
  assert.match(manifest.artifact.sha256, /^[a-f0-9]{64}$/);

  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const mobilePage = await mobile.newPage();
  await mobilePage.goto(baseUrl.href, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await acceptPrivacy(mobilePage);
  await assertPrimaryControlsFit(mobilePage, "mobile");
  await mobilePage.selectOption("#sampleModelSelect", { index: 0 });
  await mobilePage.click("#trySampleModel");
  await mobilePage.waitForFunction(() => /\.tflite$/i.test(document.querySelector("#selectedModelName")?.textContent?.trim() || ""), null, { timeout: 60_000 });
  await mobilePage.waitForSelector("#auditProgress:not([hidden])", { timeout: 30_000 });
  await mobilePage.waitForFunction(() => {
    const progress = document.querySelector("#auditProgressLabel")?.textContent || "";
    const status = document.querySelector("#status")?.textContent || "";
    return progress.includes("Complete") && !/failed/i.test(status) && !document.querySelector("#runAudit")?.disabled;
  }, null, { timeout: auditTimeoutMs });
  const overflow = await mobilePage.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert(overflow <= 1, `Mobile horizontal overflow is ${overflow}px.`);
  await mobile.close();
  console.log(`Live product check passed (${baseUrl.href}; manifest ${destination}; mobile overflow ${overflow}px).`);
} finally {
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}

async function acceptPrivacy(page) {
  await page.locator("#agreementBackdrop:not([hidden])").waitFor({ state: "visible", timeout: 10_000 }).catch(() => {});
  if (await page.locator("#agreementBackdrop").isVisible().catch(() => false)) {
    await page.check("#privacyAgree");
    await page.click("#acceptAgreement");
  }
}

async function assertPrimaryControlsFit(page, viewportLabel) {
  const layout = await page.evaluate(() => {
    const container = document.querySelector(".upload-controls");
    const sample = document.querySelector(".sample-model-control");
    const select = document.querySelector("#sampleModelSelect");
    const button = document.querySelector("#trySampleModel");
    const run = document.querySelector("#runAudit");
    if (!container || !sample || !select || !button || !run) return null;
    const bounds = (node) => {
      const rect = node.getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width };
    };
    return {
      container: bounds(container),
      sample: bounds(sample),
      select: bounds(select),
      button: bounds(button),
      run: bounds(run),
      selectClientWidth: select.clientWidth,
      selectScrollWidth: select.scrollWidth,
    };
  });
  assert(layout, `${viewportLabel}: primary upload controls were not rendered.`);
  const tolerance = 1;
  assert(layout.sample.left >= layout.container.left - tolerance && layout.sample.right <= layout.container.right + tolerance,
    `${viewportLabel}: verified-example control exceeds its container: ${JSON.stringify(layout)}`);
  assert(layout.select.left >= layout.sample.left - tolerance && layout.select.right <= layout.sample.right + tolerance,
    `${viewportLabel}: verified-example select is clipped: ${JSON.stringify(layout)}`);
  assert(layout.button.left >= layout.sample.left - tolerance && layout.button.right <= layout.sample.right + tolerance,
    `${viewportLabel}: verified-example button is clipped: ${JSON.stringify(layout)}`);
  assert(layout.selectScrollWidth <= layout.selectClientWidth + tolerance,
    `${viewportLabel}: selected example text is truncated (${layout.selectClientWidth}px < ${layout.selectScrollWidth}px).`);
}

async function clickAfterScroll(page, selector) {
  const control = page.locator(selector);
  await control.evaluate((node) => node.scrollIntoView({ block: "center", inline: "center" }));
  await control.click();
}

async function productState(page) {
  return page.evaluate(() => ({
    selectedModel: document.querySelector("#selectedModelName")?.textContent || "",
    progress: document.querySelector("#auditProgressLabel")?.textContent || "",
    status: document.querySelector("#status")?.textContent || "",
    nextStep: document.querySelector("#analysisPlanStatus")?.textContent || "",
    runDisabled: Boolean(document.querySelector("#runAudit")?.disabled),
    verificationDisabled: Boolean(document.querySelector("#downloadPublicVerificationManifest")?.disabled),
    verificationTitle: document.querySelector("#downloadPublicVerificationManifest")?.title || "",
    reportTarget: document.querySelector("#reportTargetSelect")?.value || "",
    reportTargetStatus: document.querySelector("#reportTargetStatus")?.textContent || "",
  }));
}

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || "" : "";
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
    const file = path.resolve(root, relative);
    if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain; charset=utf-8", "forbidden");
    try {
      send(response, 200, mimeType(file), await readFile(file));
    } catch {
      send(response, 404, "text/plain; charset=utf-8", "not found");
    }
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
