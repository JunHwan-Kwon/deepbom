import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-accumulator-reachability-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1120 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => { if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`); });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 90_000 });
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
  await page.locator("#accumulatorReachabilityPanel").waitFor({ state: "visible" });
  await page.locator("#accumulatorReachabilityPanel").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => document.querySelector("#accumulatorReachabilityStatus")?.textContent === "independently verified", null, { timeout: 300_000 });

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.options !== 53
    || initial.modes !== 4 || initial.tables !== 3 || initial.rankingRows !== 16
    || initial.heatPixels < 300 || initial.tracePixels < 200 || initial.downloadDisabled
    || initial.selectedOption !== "58" || initial.channel !== "0"
    || !initial.text.includes("2,239,435") || !initial.text.includes("3,585") || !initial.text.includes("631,524")
    || !initial.text.includes("complete integer interval") || !initial.text.includes("acc 27 -> default 1, single 0")
    || !initial.text.includes("kernel-local bounded-sum reachability")) {
    throw new Error(`Accumulator-reachability viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 1800) })}`);
  }

  await page.locator(".reachability-op-select").selectOption("1");
  await page.locator(".reachability-channel-control input").fill("5");
  await page.locator(".reachability-channel-control input").press("Enter");
  await page.locator(".reachability-channel-control input").blur();
  await page.waitForFunction(() => document.querySelector(".reachability-channel-control input")?.value === "5");
  const signatures = [await canvasSignature(page, ".reachability-heatmap")];
  for (const label of ["Excluded", "Unresolved", "GCD"]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    signatures.push(await canvasSignature(page, ".reachability-heatmap"));
  }
  if (new Set(signatures).size !== 4) throw new Error(`Reachability heatmap modes did not render distinct pixel states: ${signatures.join(",")}`);
  const modular = await panelState(page);
  if (!modular.text.includes("complete modular lattice") || !modular.text.includes("accumulator = minimum + 8k")
    || !modular.text.includes("23 exact / 163 excluded / 0 unresolved")) {
    throw new Error(`Modular lattice certificate is incomplete: ${modular.text.slice(0, 1600)}`);
  }

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#accumulatorReachabilityPanel").getByRole("button", { name: "Channel JSON", exact: true }).click(),
  ]);
  const downloadedPath = path.join(output, download.suggestedFilename());
  await download.saveAs(downloadedPath);
  const certificate = JSON.parse(await readFile(downloadedPath, "utf8"));
  if (certificate.schema !== "deepbom.accumulator_reachability_selected_channel.v1" || certificate.op_index !== 1
    || certificate.channel_index !== 5 || certificate.proof_status !== "complete_modular_lattice" || certificate.lattice_gcd !== 8
    || certificate.exact_reachable_divergent_state_count_decimal !== "23"
    || certificate.provably_unreachable_divergent_state_count_decimal !== "163"
    || certificate.unresolved_divergent_state_count_decimal !== "0"
    || !certificate.denomination_coverage_steps.length || !certificate.first_exact_reachable_aggregate_coefficient_witness.length
    || !/^[a-f0-9]{64}$/.test(certificate.generated_from.reachability_ledger_sha256)) {
    throw new Error(`Selected reachability certificate is inconsistent: ${JSON.stringify(certificate).slice(0, 1800)}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/accumulator_reachability.png");
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
  if (!visualExport || visualExport.width !== 2360 || visualExport.height !== 1640 || visualExport.colored < 1_000 || visualExport.opaque < 2_000) {
    throw new Error(`Accumulator-reachability Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const desktopPath = path.join(output, "accumulator-reachability-desktop.png");
  await page.locator("#accumulatorReachabilityPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#accumulatorReachabilityPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#accumulatorReachabilityPanel").evaluate((panel) => {
    const summary = panel.querySelector(".reachability-summary");
    const main = panel.querySelector(".reachability-main");
    const toolbar = panel.querySelector(".reachability-toolbar");
    const select = panel.querySelector(".reachability-op-select");
    const canvases = [...panel.querySelectorAll("canvas")];
    const scrolls = [...panel.querySelectorAll(".reachability-table-scroll")];
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
    throw new Error(`Accumulator-reachability mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "accumulator-reachability-mobile.png");
  await page.locator("#accumulatorReachabilityPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Accumulator Reachability viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; lazy independent verification, four heatmaps, exact trace, selected certificate, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    reachabilityStatus: document.querySelector("#accumulatorReachabilityStatus")?.textContent || null,
    panel: document.querySelector("#accumulatorReachabilityPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#accumulatorReachabilityPanel").evaluate((panel) => {
    const pixelCount = (canvas) => {
      const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
      let colored = 0;
      for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
      return colored;
    };
    const canvases = panel.querySelectorAll("canvas");
    const tables = panel.querySelectorAll("table");
    return {
      status: panel.querySelector("#accumulatorReachabilityStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".reachability-metric").length,
      options: panel.querySelectorAll(".reachability-op-select option").length,
      selectedOption: panel.querySelector(".reachability-op-select")?.value || "",
      channel: panel.querySelector(".reachability-channel-control input")?.value || "",
      modes: panel.querySelectorAll(".reachability-modes button").length,
      tables: tables.length,
      rankingRows: tables[tables.length - 1]?.querySelectorAll("tbody tr").length || 0,
      heatPixels: pixelCount(canvases[0]),
      tracePixels: pixelCount(canvases[1]),
      downloadDisabled: panel.querySelector("#downloadAccumulatorReachability")?.disabled,
      text: panel.textContent || "",
    };
  });
}

async function canvasSignature(browserPage, selector) {
  await waitForAnimationFrames(browserPage);
  return browserPage.locator(selector).evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    for (let index = 0; index < data.length; index += 97) hash = Math.imul(hash ^ data[index], 16777619) >>> 0;
    return hash.toString(16);
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

function send(response, status, type, body) { response.writeHead(status, { "content-type": type, "cache-control": "no-store" }); response.end(body); }
function mimeType(file) { if (file.endsWith(".html")) return "text/html; charset=utf-8"; if (file.endsWith(".js") || file.endsWith(".mjs")) return "text/javascript; charset=utf-8"; if (file.endsWith(".css")) return "text/css; charset=utf-8"; if (file.endsWith(".wasm")) return "application/wasm"; if (file.endsWith(".json")) return "application/json; charset=utf-8"; return "application/octet-stream"; }
