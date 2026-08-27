import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-channel-vitality-viewer-"));
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
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="integer-safety"]').click();
  await page.locator("#channelVitalityPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#channelVitalityStatus")?.textContent === "independently verified", null, { timeout: 120_000 });

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.options !== 53
    || initial.segments !== 5 || initial.tables !== 2 || initial.histogramRows !== 7 || initial.rankingRows !== 16
    || initial.canvasPixels < 300 || initial.downloadDisabled || initial.channel !== "3"
    || initial.selectedOption !== "1" || !initial.text.includes("18,057") || !initial.text.includes("Variable but constant")
    || !initial.text.includes("-320,299") || !initial.text.includes("lower code clamp") || !initial.text.includes("interval-hull upper bound")) {
    throw new Error(`Channel-vitality viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 1200) })}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/channel_vitality.png");
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
    throw new Error(`Channel-vitality Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const defaultSpan = await canvasSignature(page);
  await page.getByRole("tab", { name: "Cause", exact: true }).click();
  const defaultCause = await canvasSignature(page);
  await page.getByRole("tab", { name: "Sign", exact: true }).click();
  const sign = await canvasSignature(page);
  await page.getByRole("tab", { name: "Code span", exact: true }).click();
  await page.getByRole("tab", { name: "Single rounding", exact: true }).click();
  const singleSpan = await canvasSignature(page);
  if (new Set([defaultSpan, defaultCause, sign, singleSpan]).size !== 4) {
    throw new Error("Channel-vitality field and build modes did not render four distinct pixel states.");
  }

  const canvas = page.locator("#channelVitalityPanel canvas");
  await canvas.scrollIntoViewIfNeeded();
  const firstPoint = await canvas.evaluate((node) => {
    const geometry = node.__channelVitalityState;
    const rect = node.getBoundingClientRect();
    return {
      x: rect.left + (geometry.padX + geometry.cellWidth / 2) * rect.width / geometry.width,
      y: rect.top + (geometry.padTop + geometry.cellHeight / 2) * rect.height / geometry.height,
    };
  });
  await page.mouse.move(firstPoint.x, firstPoint.y);
  await page.locator(".channel-vitality-tooltip:not([hidden])").waitFor({ state: "visible" });
  const tooltip = await page.locator(".channel-vitality-tooltip").textContent();
  if (!tooltip.includes("channel 0") || !tooltip.includes("output") || !tooltip.includes("inclusive span") || !tooltip.includes("post bias")) {
    throw new Error(`Channel-vitality tooltip is incomplete: ${tooltip}`);
  }

  const channelInput = page.locator(".channel-vitality-channel-control input");
  await channelInput.fill("26");
  await channelInput.press("Enter");
  await waitForAnimationFrames(page);
  const modeDependent = await panelState(page);
  if (modeDependent.channel !== "26" || !modeDependent.text.includes("-254 .. 1")
    || !modeDependent.text.includes("preclamp -88 .. 1 -> output 0 .. 1; span 2; nonconstant")
    || !modeDependent.text.includes("preclamp -87 .. 0 -> output 0 .. 0; span 1; lower code clamp")
    || !modeDependent.text.includes("constant classification changes by build flag")) {
    throw new Error(`Mode-dependent channel selection is inconsistent: ${JSON.stringify({ ...modeDependent, text: modeDependent.text.slice(0, 1200) })}`);
  }

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#channelVitalityPanel").getByRole("button", { name: "Channel JSON", exact: true }).click(),
  ]);
  const downloadedPath = path.join(output, download.suggestedFilename());
  await download.saveAs(downloadedPath);
  const exported = JSON.parse(await readFile(downloadedPath, "utf8"));
  if (exported.schema !== "deepbom.channel_vitality_selected_channel.v1" || exported.op_index !== 1
    || exported.output_channel !== 26 || exported.post_bias_minimum_decimal !== "-254"
    || exported.post_bias_maximum_decimal !== "1" || exported.default.inclusive_code_span !== 2
    || exported.default.minimum_preclamp_code !== -88 || exported.default.maximum_preclamp_code !== 1
    || exported.single_rounding.inclusive_code_span !== 1 || exported.single_rounding.minimum_preclamp_code !== -87
    || exported.single_rounding.maximum_preclamp_code !== 0 || !exported.mode_dependent_constant
    || exported.vitality_ledger_sha256 !== "d4b09fabf80bb0b6ac30e0686097e220720d918f59e1bec1cf6e84a5c2e3498d") {
    throw new Error(`Selected Channel JSON is incomplete or inconsistent: ${JSON.stringify(exported)}`);
  }

  const desktopPath = path.join(output, "channel-vitality-desktop.png");
  await page.locator("#channelVitalityPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#channelVitalityPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#channelVitalityPanel").evaluate((panel) => {
    const summary = panel.querySelector(".channel-vitality-summary");
    const main = panel.querySelector(".channel-vitality-main");
    const toolbar = panel.querySelector(".channel-vitality-toolbar");
    const select = panel.querySelector(".channel-vitality-op-select");
    const canvasNode = panel.querySelector("canvas");
    const wraps = [...panel.querySelectorAll(".channel-vitality-table-wrap")];
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      mainColumns: getComputedStyle(main).gridTemplateColumns.split(" ").length,
      toolbarColumns: getComputedStyle(toolbar).gridTemplateColumns.split(" ").length,
      selectFits: select.getBoundingClientRect().width <= panel.getBoundingClientRect().width,
      tablesScrollable: wraps.every((wrap) => wrap.scrollWidth > wrap.clientWidth),
      canvasWidth: canvasNode.getBoundingClientRect().width,
      borderTopWidth: getComputedStyle(panel).borderTopWidth,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || mobile.mainColumns !== 1
    || mobile.toolbarColumns !== 1 || !mobile.selectFits || !mobile.tablesScrollable
    || mobile.canvasWidth < 280 || mobile.canvasWidth > 360 || mobile.borderTopWidth !== "2px") {
    throw new Error(`Channel-vitality mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "channel-vitality-mobile.png");
  await page.locator("#channelVitalityPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Quantized Channel Vitality viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; exact worker verification, four canvas states, selected JSON, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    vitalityStatus: document.querySelector("#channelVitalityStatus")?.textContent || null,
    panel: document.querySelector("#channelVitalityPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#channelVitalityPanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
    let colored = 0;
    for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
    return {
      status: panel.querySelector("#channelVitalityStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".channel-vitality-metric").length,
      options: panel.querySelectorAll(".channel-vitality-op-select option").length,
      selectedOption: panel.querySelector(".channel-vitality-op-select")?.value || "",
      channel: panel.querySelector(".channel-vitality-channel-control input")?.value || "",
      segments: panel.querySelectorAll('.channel-vitality-segments [role="tab"]').length,
      tables: panel.querySelectorAll(".channel-vitality-table").length,
      histogramRows: panel.querySelectorAll(".channel-vitality-table")[0]?.querySelectorAll("tbody tr").length || 0,
      rankingRows: panel.querySelectorAll(".channel-vitality-table")[1]?.querySelectorAll("tbody tr").length || 0,
      canvasPixels: colored,
      downloadDisabled: panel.querySelector("#downloadChannelVitality")?.disabled,
      text: panel.textContent || "",
    };
  });
}

async function canvasSignature(browserPage) {
  await waitForAnimationFrames(browserPage);
  return browserPage.locator("#channelVitalityPanel canvas").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    let nonBlank = 0;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
      if (data[index] !== 12 || data[index + 1] !== 21 || data[index + 2] !== 24) nonBlank += 1;
    }
    if (nonBlank < 200) throw new Error(`Channel-vitality canvas is effectively blank (${nonBlank} sampled pixels).`);
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
