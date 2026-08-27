import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT
  ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT)
  : ROOT;
const BASELINE = path.join(ROOT, "web", "samples", "mobilenet_v1_025_224_float.tflite");
const CANDIDATE = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-delta-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }

  await auditFile(page, BASELINE);
  await page.locator("#pinDeploymentBaseline").click();
  await page.waitForFunction(() => document.querySelector("#deploymentDeltaStatus")?.textContent?.includes("baseline pinned"));
  const pinned = await page.locator("#deploymentDeltaPanel").textContent();
  if (!pinned.includes("Audit a different TFLite artifact") || !pinned.includes("Replace baseline")) throw new Error(`Baseline pin state is incomplete: ${pinned}`);

  await auditFile(page, CANDIDATE);
  await page.waitForFunction(() => document.querySelector("#deploymentDeltaStatus")?.textContent?.includes("4 targets / DERIVED"), null, { timeout: 30_000 });
  const overview = await panelState(page);
  if (!overview.text.includes("29 matched")
    || !overview.text.includes("36 added / 2 removed")
    || !overview.text.includes("different_artifacts_lineage_unproven")
    || !overview.text.includes("UINT8[1x224x224x3]")
    || overview.summaryMetrics !== 4
    || overview.downloadDisabled) throw new Error(`Delta overview is incomplete: ${JSON.stringify(overview)}`);

  await page.locator('[data-delta-view="targets"]').click();
  const targets = await page.locator("#deploymentDeltaPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll(".delta-target-table tbody tr").length,
    columns: panel.querySelectorAll(".delta-target-table thead th").length,
    text: panel.textContent || "",
  }));
  if (targets.rows !== 4 || targets.columns !== 9 || !targets.text.includes("WASM SIMD") || !targets.text.includes("Fallback")) throw new Error(`Delta target ledger is incomplete: ${JSON.stringify(targets)}`);

  await page.locator('[data-delta-view="drivers"]').click();
  const drivers = await page.locator("#deploymentDeltaPanel").evaluate((panel) => ({
    targetButtons: panel.querySelectorAll("[data-delta-target]").length,
    rows: panel.querySelectorAll(".delta-driver-table tbody tr").length,
    crossRows: panel.querySelectorAll(".delta-cross-row").length,
    text: panel.textContent || "",
  }));
  if (drivers.targetButtons !== 4 || drivers.rows !== 18 || drivers.crossRows < 1 || !drivers.text.includes("4/4 regressions")) throw new Error(`Delta driver view is incomplete: ${JSON.stringify(drivers)}`);
  await page.locator("[data-delta-target]").first().click();
  if (await page.locator("[data-delta-target].active").count() !== 1) throw new Error("Delta target selector should retain exactly one active target.");

  await page.locator('[data-delta-view="alignment"]').click();
  const alignment = await page.locator("#deploymentDeltaPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll(".delta-alignment-table tbody tr").length,
    filters: panel.querySelectorAll("[data-delta-filter]").length,
    text: panel.textContent || "",
  }));
  if (alignment.rows !== 67 || alignment.filters !== 4 || !alignment.text.includes("NOT_CONCLUDED")) throw new Error(`Delta alignment ledger is incomplete: ${JSON.stringify(alignment)}`);
  await page.locator('[data-delta-filter="added"]').click();
  if (await page.locator(".delta-alignment-table tbody tr").count() !== 36) throw new Error("Added-op filter should expose exactly 36 candidate additions.");

  const desktopPath = path.join(output, "deployment-delta-desktop.png");
  await page.locator("#deploymentDeltaPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#deploymentDeltaPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#deploymentDeltaPanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".delta-summary")).gridTemplateColumns.split(" ").length,
    tableScrollable: panel.querySelector(".delta-table-wrap").scrollWidth > panel.querySelector(".delta-table-wrap").clientWidth,
    tabsScrollable: panel.querySelector(".delta-tabs").scrollWidth >= panel.querySelector(".delta-tabs").clientWidth,
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || !mobile.tableScrollable || !mobile.tabsScrollable) throw new Error(`Delta mobile layout is invalid: ${JSON.stringify(mobile)}`);
  const mobilePath = path.join(output, "deployment-delta-mobile.png");
  await page.locator("#deploymentDeltaPanel").screenshot({ path: mobilePath });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Deployment Delta viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; in-memory baseline, actual two-model audit, four targets, 67 alignment entities, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    delta: document.querySelector("#deploymentDeltaPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function auditFile(browserPage, filename) {
  await browserPage.locator("#fileInput").setInputFiles(filename);
  await browserPage.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await browserPage.locator("#runAudit").click();
  await browserPage.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
  const status = await browserPage.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`Audit failed for ${path.basename(filename)}: ${status}`);
}

async function panelState(browserPage) {
  return browserPage.locator("#deploymentDeltaPanel").evaluate((panel) => ({
    status: panel.querySelector("#deploymentDeltaStatus")?.textContent || "",
    text: panel.textContent || "",
    summaryMetrics: panel.querySelectorAll(".delta-summary .delta-metric").length,
    downloadDisabled: panel.querySelector("#downloadDeploymentDelta")?.disabled,
  }));
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      if (url.pathname.startsWith("/api/")) return send(response, 404, "application/json", '{"error":"not_found"}');
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(root, relative);
      if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      send(response, 200, mimeType(file), await readFile(file));
    } catch {
      send(response, 404, "text/plain", "not found");
    }
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".tflite": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
