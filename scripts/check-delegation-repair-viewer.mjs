import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const LAB_RENDER_TIMEOUT_MS = 120_000;
const output = await mkdtemp(path.join(tmpdir(), "deepbom-repair-viewer-"));
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
  await page.locator("#fileInput").focus();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="xnnpack"]').click();
  await page.locator("#delegationRepairPanel").waitFor({ state: "visible" });
  await page.waitForFunction(
    () => document.querySelector("#delegationRepairStatus")?.textContent?.includes("1 repair / 0 group-only / 62 fragility"),
    null,
    { timeout: LAB_RENDER_TIMEOUT_MS },
  );
  const portfolio = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    text: panel.textContent || "",
    cards: panel.querySelectorAll(".repair-scenario-card").length,
  }));
  if (!portfolio.cards || !portfolio.text.includes("Runtime build configuration is not artifact-bound")
    || !portfolio.text.includes("tflite_with_xnnpack_qu8")
    || !portfolio.text.includes("Conditionally delegatable ops affected")
    || !portfolio.text.includes("If the build condition is absent: remaining conditionally delegatable ops")
    || /Affected predicted delegated ops|If absent: remaining delegated ops/.test(portfolio.text)) {
    throw new Error(`Repair scenario view is incomplete or uses observed-placement wording for a prediction: ${JSON.stringify(portfolio)}`);
  }
  await page.locator('[data-repair-view="repair"]').click();

  const merge = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    status: panel.querySelector("#delegationRepairStatus")?.textContent || "",
    summaryMetrics: panel.querySelectorAll(".repair-summary .repair-metric").length,
    rows: panel.querySelectorAll(".repair-table tbody tr").length,
    text: panel.textContent || "",
    downloadDisabled: panel.querySelector("#downloadDelegationRepair")?.disabled,
  }));
  if (merge.status !== "1 repair / 0 group-only / 62 fragility" || merge.summaryMetrics !== 4 || merge.rows !== 1
    || !merge.text.includes("#062 AVERAGE_POOL_2D") || !merge.text.includes("Bridge Merges Delegate Segments")
    || merge.downloadDisabled) throw new Error(`Repair merge view is incomplete: ${JSON.stringify(merge)}`);

  await page.locator('[data-repair-view="islands"]').click();
  const islands = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll(".repair-island-table tbody tr").length,
    edgeRows: panel.querySelectorAll(".repair-island-edge-wrap tbody tr").length,
    text: panel.textContent || "",
  }));
  if (islands.rows !== 1 || islands.edgeRows !== 2 || !islands.text.includes("#062")
    || !islands.text.includes("Eliminates CPU Island And Merges Delegate Segments")
    || !islands.text.includes("62.5 KiB") || !islands.text.includes("Beyond best single")) {
    throw new Error(`CPU-island portfolio is incomplete: ${JSON.stringify(islands)}`);
  }
  const islandsDesktopPath = path.join(output, "delegation-repair-islands-desktop.png");
  await page.locator("#delegationRepairPanel").screenshot({ path: islandsDesktopPath });

  await page.locator('[data-repair-view="fragility"]').click();
  const fragility = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    groupRows: panel.querySelectorAll(".repair-fragility-group-table tbody tr").length,
    rawRows: panel.querySelectorAll(".repair-method-details .repair-table tbody tr").length,
    text: panel.textContent || "",
  }));
  if (!fragility.groupRows || fragility.rawRows !== 62 || !fragility.text.includes("#004 DEPTHWISE_CONV_2D") || !fragility.text.includes("1.4 MiB")) {
    throw new Error(`Repair fragility view is incomplete: ${JSON.stringify(fragility)}`);
  }

  await page.locator('[data-repair-view="repair"]').click();
  await page.locator("[data-repair-open-edges]").click();
  const repairEdges = await edgeState(page);
  if (repairEdges.rows !== 2 || !repairEdges.text.includes("T4 MobilenetV2/Conv_1/Relu6") || !repairEdges.text.includes("61.3 KiB")
    || !repairEdges.text.includes("T6 MobilenetV2/Logits/AvgPool") || !repairEdges.text.includes("1.3 KiB") || !repairEdges.text.includes("delegate_to_cpu")) {
    throw new Error(`Repair edge ledger is incomplete: ${JSON.stringify(repairEdges)}`);
  }

  await page.locator('[data-repair-op="4"]').click();
  const fragilityEdges = await edgeState(page);
  if (fragilityEdges.rows !== 2 || !fragilityEdges.text.includes("#004 DEPTHWISE_CONV_2D")
    || !fragilityEdges.text.includes("+1.4 MiB") || !fragilityEdges.text.includes("added")) {
    throw new Error(`Fragmentation edge ledger is incomplete: ${JSON.stringify(fragilityEdges)}`);
  }

  const desktopPath = path.join(output, "delegation-repair-desktop.png");
  await page.locator("#delegationRepairPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#delegationRepairPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".repair-summary")).gridTemplateColumns.split(" ").length,
    contextColumns: getComputedStyle(panel.querySelector(".repair-context")).gridTemplateColumns.split(" ").length,
    selectorScrollable: panel.querySelector(".repair-op-selector").scrollWidth > panel.querySelector(".repair-op-selector").clientWidth,
    tableScrollable: panel.querySelector(".repair-table-wrap").scrollWidth > panel.querySelector(".repair-table-wrap").clientWidth,
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || mobile.contextColumns !== 2
    || !mobile.selectorScrollable || !mobile.tableScrollable) throw new Error(`Repair mobile layout is invalid: ${JSON.stringify(mobile)}`);
  const mobilePath = path.join(output, "delegation-repair-mobile.png");
  await page.locator("#delegationRepairPanel").screenshot({ path: mobilePath });
  await page.locator('[data-repair-view="islands"]').click();
  const islandMobile = await page.locator("#delegationRepairPanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    contextColumns: getComputedStyle(panel.querySelector(".repair-island-context")).gridTemplateColumns.split(" ").length,
    portfolioScrollable: panel.querySelector(".repair-island-table").scrollWidth > panel.querySelector(".repair-table-wrap").clientWidth,
    edgeScrollable: panel.querySelector(".repair-island-edge-wrap").scrollWidth > panel.querySelector(".repair-island-edge-wrap").clientWidth,
  }));
  if (islandMobile.bodyOverflow > 1 || islandMobile.panelOverflow > 1 || islandMobile.contextColumns !== 2
    || !islandMobile.portfolioScrollable || !islandMobile.edgeScrollable) throw new Error(`CPU-island mobile layout is invalid: ${JSON.stringify(islandMobile)}`);
  const islandsMobilePath = path.join(output, "delegation-repair-islands-mobile.png");
  await page.locator("#delegationRepairPanel").screenshot({ path: islandsMobilePath });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Delegation Repair viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; 65 single-op and 1 complete-island intervention, 62 fragility rows, exact edge ledgers, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`islands_desktop=${islandsDesktopPath}`);
  console.log(`mobile=${mobilePath}`);
  console.log(`islands_mobile=${islandsMobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    repair: document.querySelector("#delegationRepairPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function edgeState(browserPage) {
  return browserPage.locator("#delegationRepairPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll(".repair-edge-table tbody tr").length,
    text: panel.textContent || "",
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
