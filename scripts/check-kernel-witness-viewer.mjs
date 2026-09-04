import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-kernel-witness-viewer-"));
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
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 180_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="integer-safety"]').click();
  await page.locator("#kernelWitnessPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#kernelWitnessStatus")?.textContent === "independently verified", null, { timeout: 120_000 });

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.options !== 53
    || initial.segments !== 4 || initial.tables !== 2 || initial.endpointRows !== 2 || initial.rankingRows !== 16
    || initial.canvasPixels < 300 || initial.downloadDisabled || initial.channel !== "767"
    || !initial.selectedOption.startsWith("55") || !initial.text.includes("6,942,080") || !initial.text.includes("72,228")
    || !initial.text.includes("137") || !initial.text.includes("138") || !initial.text.includes("full-model input")) {
    throw new Error(`Kernel-witness viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 900) })}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/kernel_extremum_witness.png");
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
    throw new Error(`Kernel-witness Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const maximumPattern = await canvasSignature(page);
  await page.getByRole("tab", { name: "Minimum", exact: true }).click();
  const minimumPattern = await canvasSignature(page);
  await page.getByRole("tab", { name: "Contribution", exact: true }).click();
  const minimumContribution = await canvasSignature(page);
  await page.getByRole("tab", { name: "Maximum", exact: true }).click();
  const maximumContribution = await canvasSignature(page);
  if (new Set([maximumPattern, minimumPattern, minimumContribution, maximumContribution]).size !== 4) {
    throw new Error("Kernel-witness endpoint and field modes did not render four distinct pixel states.");
  }

  const canvas = page.locator("#kernelWitnessPanel canvas");
  await canvas.scrollIntoViewIfNeeded();
  const termPoint = await canvas.evaluate((node) => {
    const geometry = node.__kernelWitnessState;
    const rect = node.getBoundingClientRect();
    return {
      x: rect.left + (geometry.padX + geometry.cellWidth / 2) * rect.width / geometry.width,
      y: rect.top + (geometry.padTop + geometry.cellHeight / 2) * rect.height / geometry.height,
    };
  });
  await page.mouse.move(termPoint.x, termPoint.y);
  await page.locator(".kernel-witness-tooltip:not([hidden])").waitFor({ state: "visible" });
  const tooltip = await page.locator(".kernel-witness-tooltip").textContent();
  if (!tooltip.includes("term 0") || !tooltip.includes("input code") || !tooltip.includes("centered weight") || !tooltip.includes("contribution")) {
    throw new Error(`Kernel-witness tooltip is incomplete: ${tooltip}`);
  }

  await page.getByRole("tab", { name: "Input pattern", exact: true }).click();
  const channelInput = page.locator(".kernel-witness-channel-control input");
  await channelInput.fill("1");
  await channelInput.press("Enter");
  await waitForRenderedChannel(page, 1);
  const channelOne = await panelState(page);
  if (channelOne.channel !== "1" || !channelOne.text.includes("output channel 1") || channelOne.canvasPixels < 300) {
    throw new Error(`Arbitrary witness channel selection is inconsistent: ${JSON.stringify(channelOne)}`);
  }

  await channelInput.fill("767");
  await channelInput.press("Enter");
  await waitForRenderedChannel(page, 767);
  await page.getByRole("tab", { name: "Maximum", exact: true }).click();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Witness JSON", exact: true }).click(),
  ]);
  const downloadedPath = path.join(output, download.suggestedFilename());
  await download.saveAs(downloadedPath);
  const exported = JSON.parse(await readFile(downloadedPath, "utf8"));
  if (exported.schema !== "deepbom.kernel_extremum_selected_witness.v1" || exported.op_index !== 55
    || exported.output_channel !== 767 || exported.endpoint !== "maximum" || exported.terms.length !== 9
    || exported.pattern_sha256 !== "35445b96d727a11cf2bdeab9dc0df29210496ffb5cac395e59bff80c97f7e31c"
    || exported.projection.ideal_output_code !== 137 || exported.projection.default_output_code !== 138
    || exported.projection.single_output_code !== 137 || exported.terms.some((term) => term.input_code !== 0)) {
    throw new Error(`Selected Witness JSON is incomplete or inconsistent: ${JSON.stringify(exported)}`);
  }

  const desktopPath = path.join(output, "kernel-witness-desktop.png");
  await page.locator("#kernelWitnessPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#kernelWitnessPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#kernelWitnessPanel").evaluate((panel) => {
    const summary = panel.querySelector(".kernel-witness-summary");
    const main = panel.querySelector(".kernel-witness-main");
    const toolbar = panel.querySelector(".kernel-witness-toolbar");
    const select = panel.querySelector(".kernel-witness-op-select");
    const canvasNode = panel.querySelector("canvas");
    const wraps = [...panel.querySelectorAll(".kernel-witness-table-wrap")];
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
      borderTopColor: getComputedStyle(panel).borderTopColor,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || mobile.mainColumns !== 1
    || mobile.toolbarColumns !== 1 || !mobile.selectFits || !mobile.tablesScrollable
    || mobile.canvasWidth < 280 || mobile.canvasWidth > 360 || mobile.borderTopWidth !== "2px") {
    throw new Error(`Kernel-witness mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "kernel-witness-mobile.png");
  await page.locator("#kernelWitnessPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Quantized Kernel Witness viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; exact worker verification, four canvas states, selected JSON, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    witnessStatus: document.querySelector("#kernelWitnessStatus")?.textContent || null,
    panel: document.querySelector("#kernelWitnessPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#kernelWitnessPanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
    let colored = 0;
    for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
    return {
      status: panel.querySelector("#kernelWitnessStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".kernel-witness-metric").length,
      options: panel.querySelectorAll(".kernel-witness-op-select option").length,
      selectedOption: panel.querySelector(".kernel-witness-op-select")?.value || "",
      channel: panel.querySelector(".kernel-witness-channel-control input")?.value || "",
      segments: panel.querySelectorAll('.kernel-witness-segments [role="tab"]').length,
      tables: panel.querySelectorAll(".kernel-witness-table").length,
      endpointRows: panel.querySelectorAll(".kernel-witness-table")[0]?.querySelectorAll("tbody tr").length || 0,
      rankingRows: panel.querySelectorAll(".kernel-witness-table")[1]?.querySelectorAll("tbody tr").length || 0,
      canvasPixels: colored,
      downloadDisabled: panel.querySelector("#downloadKernelWitness")?.disabled,
      text: panel.textContent || "",
    };
  });
}

async function canvasSignature(browserPage) {
  await waitForAnimationFrames(browserPage);
  return browserPage.locator("#kernelWitnessPanel canvas").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    let nonBlank = 0;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
      if (data[index] !== 13 || data[index + 1] !== 21 || data[index + 2] !== 24) nonBlank += 1;
    }
    if (nonBlank < 200) throw new Error(`Kernel-witness canvas is effectively blank (${nonBlank} sampled pixels).`);
    return `${hash}:${nonBlank}`;
  });
}

async function waitForRenderedChannel(browserPage, expectedChannel) {
  await browserPage.waitForFunction((channel) => {
    const input = document.querySelector(".kernel-witness-channel-control input");
    const canvas = document.querySelector("#kernelWitnessPanel canvas");
    if (input?.value !== String(channel) || Number(canvas?.__kernelWitnessState?.selected?.channel_index) !== channel) return false;
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    for (let index = 0; index < data.length; index += 128) {
      if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
    }
    return colored >= 300;
  }, expectedChannel, { timeout: 10_000 });
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
