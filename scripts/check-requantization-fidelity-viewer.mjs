import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-requant-viewer-"));
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
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="numerical-abi"]').click();
  await page.locator("#requantizationFidelityPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#requantizationFidelityStatus")?.textContent === "independently verified", null, { timeout: 30_000 });

  const state = await page.locator("#requantizationFidelityPanel").evaluate((panel) => ({
    status: panel.querySelector("#requantizationFidelityStatus")?.textContent || "",
    metrics: panel.querySelectorAll(".requant-summary-metric").length,
    options: panel.querySelectorAll(".requant-op-select option").length,
    rows: panel.querySelectorAll(".requant-ranking-table tbody tr").length,
    graphButtons: panel.querySelectorAll(".requant-graph-button").length,
    sourceLinks: panel.querySelectorAll(".requant-definition a").length,
    sourceHref: panel.querySelector(".requant-definition a")?.href || "",
    text: panel.textContent || "",
    canvas: panel.querySelector("canvas")?.toDataURL() || "",
    downloadDisabled: panel.querySelector("#downloadRequantizationFidelity")?.disabled,
  }));
  if (state.status !== "independently verified" || state.metrics !== 4 || state.options !== 53 || state.rows !== 53
    || state.graphButtons !== 53 || state.sourceLinks !== 3 || !state.sourceHref.includes("87bbf65b8d23d3f06912b1b2183587e1884bc45c")
    || !state.text.includes("18,057") || !state.text.includes("4.497e-6") || !state.text.includes("0.750000 / 0.500004")
    || !state.text.includes("-11..-1") || !state.text.includes("Worst encoding witness") || !state.text.includes("1764866200")
    || state.canvas.length < 2_000 || state.downloadDisabled) {
    throw new Error(`Requantization viewer is incomplete: ${JSON.stringify({ ...state, text: state.text.slice(0, 700), canvas: state.canvas.length })}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/requantization_fidelity.png");
    if (!spec) return null;
    const canvas = spec[1]();
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    let opaque = 0;
    for (let index = 0; index < pixels.length; index += 128) {
      if (pixels[index] !== pixels[index + 1] || pixels[index + 1] !== pixels[index + 2]) colored += 1;
      if (pixels[index + 3] > 0) opaque += 1;
    }
    return { width: canvas.width, height: canvas.height, colored, opaque };
  });
  if (!visualExport || visualExport.width < 1180 || visualExport.height < 760
    || Math.abs(visualExport.width / visualExport.height - 1180 / 760) > 1e-9
    || visualExport.colored < 1_000 || visualExport.opaque < 2_000) {
    throw new Error(`Requantization Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const encodingSignature = await canvasSignature(page, "encoding");
  await page.locator('[data-requant-mode="rounding"]').click();
  const roundingSignature = await canvasSignature(page, "rounding");
  await page.locator('[data-requant-mode="shift"]').click();
  const shiftSignature = await canvasSignature(page, "shift");
  if (new Set([encodingSignature, roundingSignature, shiftSignature]).size !== 3) throw new Error("Requantization canvas modes did not render distinct pixels.");

  const canvas = page.locator("#requantizationFidelityPanel canvas");
  await canvas.scrollIntoViewIfNeeded();
  const channelPoint = await canvas.evaluate((node) => {
    const geometry = node._requantGeometry;
    const rect = node.getBoundingClientRect();
    return {
      x: (geometry.pad + geometry.cellWidth / 2) * rect.width / geometry.width,
      y: (geometry.pad + geometry.cellHeight / 2) * rect.height / geometry.height,
    };
  });
  await canvas.hover({ position: channelPoint });
  await page.locator(".requant-tooltip:not([hidden])").waitFor({ state: "visible" });
  const tooltip = await page.locator(".requant-tooltip").textContent();
  if (!tooltip.includes("ch ") || !tooltip.includes("shift") || !tooltip.includes("encoding") || !tooltip.includes("default")) {
    throw new Error(`Requantization tooltip is incomplete: ${tooltip}`);
  }

  await page.locator(".requant-op-select").selectOption("58");
  await canvasSignature(page, "shift", 58);
  const selected = await page.locator("#requantizationFidelityPanel").evaluate((panel) => ({
    option: panel.querySelector(".requant-op-select")?.value || "",
    selectedRow: panel.querySelector(".requant-ranking-table tr.selected")?.textContent || "",
    canvasLabel: panel.querySelector("canvas")?.getAttribute("aria-label") || "",
    signature: panel.querySelector("canvas")?.dataset.pixelSignature || "",
  }));
  if (selected.option !== "58" || !selected.selectedRow.includes("#58 CONV_2D") || !selected.canvasLabel.includes("operator 58") || !selected.signature.includes(":58:")) {
    throw new Error(`Requantization selection is inconsistent: ${JSON.stringify(selected)}`);
  }
  await page.locator('[data-requant-mode="encoding"]').click();
  await canvasSignature(page, "encoding", 58);

  const desktopPath = path.join(output, "requantization-fidelity-desktop.png");
  await page.locator("#requantizationFidelityPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#requantizationFidelityPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#requantizationFidelityPanel").evaluate((panel) => {
    const select = panel.querySelector(".requant-op-select");
    const tableWrap = panel.querySelector(".requant-ranking-wrap");
    const table = panel.querySelector(".requant-ranking-table");
    const canvasNode = panel.querySelector("canvas");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(panel.querySelector(".requant-summary")).gridTemplateColumns.split(" ").length,
      mainColumns: getComputedStyle(panel.querySelector(".requant-main")).gridTemplateColumns.split(" ").length,
      selectFits: select.getBoundingClientRect().width <= panel.getBoundingClientRect().width,
      tableScrollable: table.scrollWidth > tableWrap.clientWidth,
      tableVerticallyScrollable: table.scrollHeight > tableWrap.clientHeight,
      canvasWidth: canvasNode.getBoundingClientRect().width,
      canvasSignature: canvasNode.dataset.pixelSignature || "",
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || mobile.mainColumns !== 1
    || !mobile.selectFits || !mobile.tableScrollable || !mobile.tableVerticallyScrollable
    || mobile.canvasWidth < 280 || mobile.canvasWidth > 360 || !mobile.canvasSignature.includes(":58:")) {
    throw new Error(`Requantization mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "requantization-fidelity-mobile.png");
  await page.locator("#requantizationFidelityPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Requantization Fidelity viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; 53 ops, 18,057 independently verified channels, three canvas modes, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    requantizationStatus: document.querySelector("#requantizationFidelityStatus")?.textContent || null,
    requantization: document.querySelector("#requantizationFidelityPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function canvasSignature(browserPage, mode, opIndex = null) {
  await browserPage.waitForFunction(({ expectedMode, expectedOpIndex }) => {
    const signature = document.querySelector("#requantizationFidelityPanel canvas")?.dataset.pixelSignature || "";
    return signature.startsWith(`${expectedMode}:`)
      && (expectedOpIndex == null || signature.startsWith(`${expectedMode}:${expectedOpIndex}:`));
  }, { expectedMode: mode, expectedOpIndex: opIndex });
  await waitForAnimationFrames(browserPage);
  return browserPage.locator("#requantizationFidelityPanel canvas").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    let nonBlank = 0;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
      if (data[index] !== 16 || data[index + 1] !== 23 || data[index + 2] !== 26) nonBlank += 1;
    }
    if (nonBlank < 200) throw new Error(`Requantization canvas is effectively blank (${nonBlank} sampled pixels).`);
    return `${hash}:${nonBlank}`;
  });
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
