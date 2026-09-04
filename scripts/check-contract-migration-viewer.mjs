import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-contract-migration-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, deviceScaleFactor: 1 });
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
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="residual-contract"]').click();
  await page.locator("#contractMigrationPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#contractMigrationStatus")?.textContent === "independently verified", null, { timeout: 30_000 });

  const state = await panelState(page);
  if (state.status !== "independently verified" || state.metrics !== 4 || state.selectors !== 10 || state.designs !== 2
    || state.consumers !== 2 || state.biasRows !== 8 || state.addRows !== 3 || state.canvases !== 2
    || state.canvasPixels.some((pixels) => pixels < 300) || state.downloadDisabled
    || !state.text.includes("#027 output contract") || !state.text.includes("0.943574 old steps")
    || !state.text.includes("#028 CONV_2D") || !state.text.includes("#031 ADD")
    || !state.text.includes("counterfactual re-export impact analysis")) {
    throw new Error(`Migration viewer is incomplete: ${JSON.stringify(state)}`);
  }
  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/contract_migration_impact.png");
    if (!spec) return null;
    const canvas = spec[1]();
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    for (let index = 0; index < data.length; index += 128) {
      if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
    }
    return { width: canvas.width, height: canvas.height, colored };
  });
  if (!visualExport || visualExport.width !== 2360 || visualExport.height !== 1520 || visualExport.colored < 1_000) {
    throw new Error(`Contract-migration Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }
  const globalSignature = await canvasSignatures(page);
  await page.locator('[data-migration-design="fixed_zero_point_minimum_containment"]').click();
  const fixedState = await panelState(page);
  const fixedSignature = await canvasSignatures(page);
  if (!fixedState.text.includes("0.369338") || !fixedState.text.includes("zp 122")
    || globalSignature[0] === fixedSignature[0]) throw new Error("Candidate switch did not update exact contract evidence and channel pixels.");

  await page.locator('[data-migration-add="9"]').click();
  const selected = await panelState(page);
  if (!selected.activeSelector.includes("#009") || selected.consumers !== 1
    || !selected.text.includes("no direct ADD consumer") || !selected.text.includes("55 reachable ops")) {
    throw new Error(`Residual selection is inconsistent: ${JSON.stringify(selected)}`);
  }
  const desktopPath = path.join(output, "contract-migration-desktop.png");
  await page.locator("#contractMigrationPanel").screenshot({ path: desktopPath });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#contractMigrationPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#contractMigrationPanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".migration-summary")).gridTemplateColumns.split(" ").length,
    designColumns: getComputedStyle(panel.querySelector(".migration-design-tabs")).gridTemplateColumns.split(" ").length,
    consumerColumns: getComputedStyle(panel.querySelector(".migration-consumer-row")).gridTemplateColumns.split(" ").length,
    selectorScrollable: panel.querySelector(".migration-selector").scrollWidth > panel.querySelector(".migration-selector").clientWidth,
    tableScrollable: panel.querySelector(".migration-table").scrollWidth > panel.querySelector(".migration-table-wrap").clientWidth,
    canvasWidths: [...panel.querySelectorAll("canvas")].map((canvas) => canvas.getBoundingClientRect().width),
    overflowers: [...panel.querySelectorAll("*")].filter((node) => node.scrollWidth > node.clientWidth + 1)
      .slice(0, 12).map((node) => ({ className: node.className, tag: node.tagName, client: node.clientWidth, scroll: node.scrollWidth })),
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.designColumns !== 2 || mobile.consumerColumns !== 2 || !mobile.selectorScrollable
    || !mobile.tableScrollable || mobile.canvasWidths.some((width) => width < 280 || width > 360)) {
    throw new Error(`Migration mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "contract-migration-mobile.png");
  await page.locator("#contractMigrationPanel").screenshot({ path: mobilePath });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Contract Migration viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; candidate switching, two nonblank plots, direct-consumer ledgers, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    migration: document.querySelector("#contractMigrationPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#contractMigrationPanel").evaluate((panel) => ({
    status: panel.querySelector("#contractMigrationStatus")?.textContent || "",
    metrics: panel.querySelectorAll(".migration-summary .migration-metric").length,
    selectors: panel.querySelectorAll(".migration-selector button").length,
    activeSelector: panel.querySelector(".migration-selector button.active")?.textContent || "",
    designs: panel.querySelectorAll(".migration-design-tabs button").length,
    consumers: panel.querySelectorAll(".migration-consumer-row").length,
    biasRows: panel.querySelectorAll(".migration-table-section")[0]?.querySelectorAll("tbody tr").length || 0,
    addRows: panel.querySelectorAll(".migration-table-section")[1]?.querySelectorAll("tbody tr").length || 0,
    canvases: panel.querySelectorAll("canvas").length,
    canvasPixels: [...panel.querySelectorAll("canvas")].map((canvas) => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let colored = 0;
      for (let index = 0; index < data.length; index += 128) {
        if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
      }
      return colored;
    }),
    text: panel.textContent || "",
    downloadDisabled: panel.querySelector("#downloadContractMigration")?.disabled,
  }));
}

async function canvasSignatures(browserPage) {
  return browserPage.locator("#contractMigrationPanel canvas").evaluateAll((canvases) => canvases.map((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return hash;
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
