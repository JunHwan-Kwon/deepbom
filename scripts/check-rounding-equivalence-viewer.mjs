import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-rounding-equivalence-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1120 }, deviceScaleFactor: 1 });
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
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 240_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="numerical-abi"]').click();
  await page.locator("#roundingEquivalencePanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#roundingEquivalenceStatus")?.textContent === "independently verified", null, { timeout: 240_000 });

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.options !== 53
    || initial.tabs !== 3 || initial.tables !== 2 || initial.histogramRows !== 7 || initial.rankingRows !== 16
    || initial.heatPixels < 300 || initial.tracePixels < 300 || initial.downloadDisabled
    || initial.selectedOption !== "7" || initial.channel !== "37"
    || !initial.text.includes("13,933,008,957") || !initial.text.includes("2,874,544")
    || !initial.text.includes("17,083") || !initial.text.includes("0.0206%")
    || !initial.text.includes("191 divergent states") || !initial.text.includes("default 3, single 2")
    || !initial.text.includes("exact certificate over every integer")) {
    throw new Error(`Rounding-equivalence viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 1800) })}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/rounding_equivalence.png");
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
  if (!visualExport || visualExport.width !== 2360 || visualExport.height !== 1600
    || visualExport.colored < 1_000 || visualExport.opaque < 2_000) {
    throw new Error(`Rounding-equivalence Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const exposure = await canvasSignature(page, ".rounding-equivalence-canvas");
  await page.getByRole("tab", { name: "Max delta", exact: true }).click();
  const delta = await canvasSignature(page, ".rounding-equivalence-canvas");
  await page.getByRole("tab", { name: "Regions", exact: true }).click();
  const regions = await canvasSignature(page, ".rounding-equivalence-canvas");
  if (new Set([exposure, delta, regions]).size !== 3) throw new Error("Rounding-equivalence heatmap fields did not render distinct pixel states.");

  const heat = page.locator(".rounding-equivalence-canvas");
  const heatPoint = await heat.evaluate((canvas) => {
    const rect = canvas.getBoundingClientRect();
    const cell = Number(canvas.dataset.cell);
    return {
      x: rect.left + Number(canvas.dataset.offsetX) + cell / 2,
      y: rect.top + Number(canvas.dataset.offsetY) + cell / 2,
    };
  });
  await page.mouse.move(heatPoint.x, heatPoint.y);
  await page.locator(".rounding-equivalence-tooltip:not([hidden])").first().waitFor({ state: "visible" });
  const heatTooltip = await page.locator(".rounding-equivalence-tooltip:not([hidden])").first().textContent();
  if (!heatTooltip.includes("ch 0") || !heatTooltip.includes("regions") || !heatTooltip.includes("of")) throw new Error(`Heatmap tooltip is incomplete: ${heatTooltip}`);

  const trace = page.locator(".rounding-equivalence-trace");
  await trace.scrollIntoViewIfNeeded();
  const traceRect = await trace.boundingBox();
  if (!traceRect) throw new Error("Rounding-equivalence trace is not visible.");
  await trace.hover({
    position: {
      x: traceRect.width * 0.62,
      y: traceRect.height * 0.5,
    },
  });
  await page.locator(".rounding-equivalence-tooltip.trace:not([hidden])").waitFor({ state: "visible" });
  const traceTooltip = await page.locator(".rounding-equivalence-tooltip.trace").textContent();
  if (!traceTooltip.includes("states") || !traceTooltip.includes("default") || !traceTooltip.includes("single")) throw new Error(`Trace tooltip is incomplete: ${traceTooltip}`);

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Certificate JSON", exact: true }).click(),
  ]);
  const downloadedPath = path.join(output, download.suggestedFilename());
  await download.saveAs(downloadedPath);
  const certificate = JSON.parse(await readFile(downloadedPath, "utf8"));
  if (certificate.schema !== "deepbom.rounding_equivalence_selected_channel.v1" || certificate.op_index !== 7
    || certificate.channel_index !== 37 || certificate.interval_state_count_decimal !== "2041"
    || certificate.divergent_state_count_decimal !== "191" || certificate.pair_segment_count !== 447
    || certificate.divergent_region_count !== 191 || certificate.first_divergent_accumulator_decimal !== "14"
    || certificate.first_default_output_code !== 3 || certificate.first_single_output_code !== 2
    || certificate.last_divergent_accumulator_decimal !== "1498" || certificate.segments.length !== 447
    || certificate.generated_from.equivalence_ledger_sha256 !== "6b42280ab896789a75ce996634eb5251c01c8fbd554216f1cdaddbf3ee62e9ab") {
    throw new Error(`Selected equivalence certificate is inconsistent: ${JSON.stringify(certificate).slice(0, 1800)}`);
  }

  await page.locator(".rounding-equivalence-op-select").selectOption("51");
  await waitForAnimationFrames(page);
  const equivalent = await panelState(page);
  if (equivalent.channel !== "0" || !equivalent.text.includes("bit-exact equivalent over the complete interval hull")
    || !equivalent.text.includes("none; outputs are equal for the complete interval hull")) {
    throw new Error(`Equivalent-channel certificate is not visible: ${JSON.stringify({ ...equivalent, text: equivalent.text.slice(0, 1400) })}`);
  }

  const desktopPath = path.join(output, "rounding-equivalence-desktop.png");
  await page.locator("#roundingEquivalencePanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#roundingEquivalencePanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#roundingEquivalencePanel").evaluate((panel) => {
    const summary = panel.querySelector(".rounding-equivalence-summary");
    const main = panel.querySelector(".rounding-equivalence-main");
    const toolbar = panel.querySelector(".rounding-equivalence-toolbar");
    const select = panel.querySelector(".rounding-equivalence-op-select");
    const canvases = [...panel.querySelectorAll("canvas")];
    const scrolls = [...panel.querySelectorAll(".rounding-equivalence-table-scroll")];
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      mainColumns: getComputedStyle(main).gridTemplateColumns.split(" ").length,
      toolbarColumns: getComputedStyle(toolbar).gridTemplateColumns.split(" ").length,
      selectFits: select.getBoundingClientRect().width <= panel.getBoundingClientRect().width,
      tablesScrollable: scrolls.every((scroll) => scroll.scrollWidth > scroll.clientWidth),
      canvasWidths: canvases.map((canvas) => canvas.getBoundingClientRect().width),
      borderTopWidth: getComputedStyle(panel).borderTopWidth,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.mainColumns !== 1 || mobile.toolbarColumns !== 1 || !mobile.selectFits || !mobile.tablesScrollable
    || mobile.canvasWidths.some((width) => width < 280 || width > 360) || mobile.borderTopWidth !== "2px") {
    throw new Error(`Rounding-equivalence mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "rounding-equivalence-mobile.png");
  await page.locator("#roundingEquivalencePanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Fixed-Point Rounding Equivalence viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; exact worker verification, three heatmaps, exact trace, selected certificate, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    equivalenceStatus: document.querySelector("#roundingEquivalenceStatus")?.textContent || null,
    panel: document.querySelector("#roundingEquivalencePanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#roundingEquivalencePanel").evaluate((panel) => {
    const pixelCount = (canvas) => {
      const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
      let colored = 0;
      for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
      return colored;
    };
    const canvases = panel.querySelectorAll("canvas");
    const tables = panel.querySelectorAll("table");
    return {
      status: panel.querySelector("#roundingEquivalenceStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".rounding-equivalence-metric").length,
      options: panel.querySelectorAll(".rounding-equivalence-op-select option").length,
      selectedOption: panel.querySelector(".rounding-equivalence-op-select")?.value || "",
      channel: panel.querySelector(".rounding-equivalence-channel-control input")?.value || "",
      tabs: panel.querySelectorAll('.rounding-equivalence-segments [role="tab"]').length,
      tables: tables.length,
      histogramRows: tables[0]?.querySelectorAll("tbody tr").length || 0,
      rankingRows: tables[1]?.querySelectorAll("tbody tr").length || 0,
      heatPixels: pixelCount(canvases[0]),
      tracePixels: pixelCount(canvases[1]),
      downloadDisabled: panel.querySelector("#downloadRoundingEquivalence")?.disabled,
      text: panel.textContent || "",
    };
  });
}

async function canvasSignature(browserPage, selector) {
  await waitForAnimationFrames(browserPage);
  return browserPage.locator(selector).evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    let nonBlank = 0;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
      if (data[index] !== 17 || data[index + 1] !== 24 || data[index + 2] !== 32) nonBlank += 1;
    }
    if (nonBlank < 200) throw new Error(`Rounding-equivalence canvas is effectively blank (${nonBlank} sampled pixels).`);
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
