import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-step-response-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
const timingStart = Date.now();
const timings = {};
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
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 180_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  timings.auditCompleteMs = Date.now() - timingStart;
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="residual-contract"]').click();
  await page.locator("#residualStepResponsePanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#residualStepResponseStatus")?.textContent === "independently verified", null, { timeout: 60_000 });
  timings.independentlyVerifiedMs = Date.now() - timingStart;

  const state = await panelState(page);
  if (state.status !== "independently verified" || state.metrics !== 4 || state.selectors !== 10
    || state.designs !== 3 || state.canvases !== 2 || state.branchRows !== 2 || state.contractRows !== 3
    || state.portfolioRows !== 10 || state.canvasPixels.some((pixels) => pixels < 200) || state.downloadDisabled
    || !state.text.includes("#027 ADD") || !state.text.includes("65,375") || !state.text.includes("+30,391")
    || !state.text.includes("48.145%") || !state.text.includes("48.291%")
    || !state.text.includes("uniform legal-code-domain local distinguishability")) {
    throw new Error(`Step-response viewer is incomplete: ${JSON.stringify(state)}`);
  }
  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/residual_step_response.png");
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
    throw new Error(`Step-response Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }
  timings.visualExportMs = Date.now() - timingStart;

  const globalSignature = await canvasSignatures(page);
  await page.locator('[data-step-response-design="current_artifact_contract"]').click();
  const current = await panelState(page);
  const currentSignature = await canvasSignatures(page);
  if (!current.text.includes("14,792") || !current.text.includes("34,984") || globalSignature[0] === currentSignature[0]) {
    throw new Error("Current-contract switch did not update exact counts and influence-field pixels.");
  }
  await page.locator('[data-step-response-design="fixed_zero_point_minimum_containment"]').click();
  const fixed = await panelState(page);
  if (!fixed.text.includes("67,278") || !fixed.text.includes("+32,294")) throw new Error("Fixed-zero-point contract switch is inconsistent.");
  await page.locator('[data-step-response-op="9"]').click();
  const selected = await panelState(page);
  if (!selected.activeSelector.includes("#009") || !selected.text.includes("#009 ADD")) throw new Error("Residual selector did not update the workspace.");

  const desktopPath = path.join(output, "residual-step-response-desktop.png");
  await page.locator("#residualStepResponsePanel").screenshot({ path: desktopPath });
  timings.desktopScreenshotMs = Date.now() - timingStart;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#residualStepResponsePanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#residualStepResponsePanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".step-response-summary")).gridTemplateColumns.split(" ").length,
    canvasColumns: getComputedStyle(panel.querySelector(".step-response-canvas-grid")).gridTemplateColumns.split(" ").length,
    tableScrollable: [...panel.querySelectorAll(".step-response-table-scroll")].every((wrap) => wrap.scrollWidth > wrap.clientWidth),
    canvasWidths: [...panel.querySelectorAll("canvas")].map((canvas) => canvas.getBoundingClientRect().width),
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.canvasColumns !== 1 || !mobile.tableScrollable
    || mobile.canvasWidths.some((width) => width < 250 || width > 360)) throw new Error(`Step-response mobile layout is invalid: ${JSON.stringify(mobile)} timings=${JSON.stringify(timings)}`);
  const mobilePath = path.join(output, "residual-step-response-mobile.png");
  await page.locator("#residualStepResponsePanel").screenshot({ path: mobilePath });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Residual Step Response viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; 30 contracts, candidate switching, two nonblank plots, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
  console.log(`timings=${JSON.stringify({ ...timings, completeMs: Date.now() - timingStart })}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    response: document.querySelector("#residualStepResponsePanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\ntimings=${JSON.stringify({ ...timings, failedMs: Date.now() - timingStart })}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#residualStepResponsePanel").evaluate((panel) => ({
    status: panel.querySelector("#residualStepResponseStatus")?.textContent || "",
    metrics: panel.querySelectorAll(".step-response-summary .step-response-metric").length,
    selectors: panel.querySelectorAll(".step-response-op-selector button").length,
    activeSelector: panel.querySelector(".step-response-op-selector button.active")?.textContent || "",
    designs: panel.querySelectorAll(".step-response-design-tabs button").length,
    canvases: panel.querySelectorAll("canvas").length,
    branchRows: panel.querySelectorAll(".step-response-table-section")[0]?.querySelectorAll("tbody tr").length || 0,
    contractRows: panel.querySelectorAll(".step-response-table-section")[1]?.querySelectorAll("tbody tr").length || 0,
    portfolioRows: panel.querySelectorAll(".step-response-table-section")[2]?.querySelectorAll("tbody tr").length || 0,
    canvasPixels: [...panel.querySelectorAll("canvas")].map((canvas) => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let colored = 0;
      for (let index = 0; index < data.length; index += 128) {
        if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
      }
      return colored;
    }),
    text: panel.textContent || "",
    downloadDisabled: panel.querySelector("#downloadResidualStepResponse")?.disabled,
  }));
}

async function canvasSignatures(browserPage) {
  return browserPage.locator("#residualStepResponsePanel canvas").evaluateAll((canvases) => canvases.map((canvas) => {
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
