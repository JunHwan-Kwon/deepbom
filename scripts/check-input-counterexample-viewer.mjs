import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-input-counterexample-viewer-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1120 }, deviceScaleFactor: 1, acceptDownloads: true });
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
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 240_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="numerical-abi"]').click();
  await page.locator("#inputCounterexamplePanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#inputCounterexampleStatus")?.textContent === "independently verified", null, { timeout: 240_000 });
  await page.waitForTimeout(150);

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.canvases !== 1
    || initial.tables !== 2 || initial.termRows !== 27 || initial.sourceRows !== 52
    || initial.canvasPixels < 600 || initial.portfolioDisabled || initial.rawDisabled
    || initial.actionButtons.join("|") !== "Input witness JSON|Input tensor|Graph source"
    || !initial.text.includes("1 / 52") || !initial.text.includes("2,918 realizable divergent accumulator states")
    || !initial.text.includes("150,528") || !initial.text.includes("26 sparse overrides")
    || !initial.text.includes("default 1 / single-rounding 0 / delta +1")
    || !initial.text.includes("1,024 graph routes; declared-output effect not proven")
    || !initial.text.includes("-13,115 + 13,159 = 44")
    || !initial.text.includes("89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035")
    || !initial.text.includes("45cdbb1087d79e088d7830c0e1840daa39b420b8f29e7c78fbb7e7ba702d0ce5")
    || !initial.text.includes("exact at the model tensor ABI")
    || !initial.text.includes("not necessarily realizable through an application's image/audio preprocessing contract")
    || !initial.text.includes("does not prove a declared model output changes")) {
    throw new Error(`Input witness viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 2600) })}`);
  }

  const [witnessDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#inputCounterexamplePanel").getByRole("button", { name: "Input witness JSON", exact: true }).click(),
  ]);
  const witnessPath = path.join(output, witnessDownload.suggestedFilename());
  await witnessDownload.saveAs(witnessPath);
  const witness = JSON.parse(await readFile(witnessPath, "utf8"));
  if (witness.source_op_index !== 0 || witness.source_channel_index !== 4
    || witness.model_input_element_count !== 150_528 || witness.sparse_override_count !== 26
    || witness.terms?.length !== 27 || witness.dot_product_decimal !== "-13115"
    || witness.post_bias_accumulator_decimal !== "44"
    || witness.full_model_input_tensor_sha256 !== "89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035") {
    throw new Error(`Witness JSON is inconsistent: ${JSON.stringify(witness).slice(0, 1800)}`);
  }

  const [rawDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#inputCounterexamplePanel").getByRole("button", { name: "Input tensor", exact: true }).click(),
  ]);
  const rawPath = path.join(output, rawDownload.suggestedFilename());
  await rawDownload.saveAs(rawPath);
  const raw = await readFile(rawPath);
  if (raw.byteLength !== 150_528 || createHash("sha256").update(raw).digest("hex") !== witness.full_model_input_tensor_sha256) {
    throw new Error(`Raw input tensor is inconsistent: ${raw.byteLength} bytes.`);
  }

  const [portfolioDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadInputCounterexample").click(),
  ]);
  const portfolioPath = path.join(output, portfolioDownload.suggestedFilename());
  await portfolioDownload.saveAs(portfolioPath);
  const portfolio = JSON.parse(await readFile(portfolioPath, "utf8"));
  if (portfolio.schema !== "deepbom.input_counterexample.v1" || portfolio.sources?.length !== 52
    || portfolio.witnesses?.length !== 1 || portfolio.tensor_abi_constructive_source_op_count !== 1
    || portfolio.upstream_activation_unresolved_source_op_count !== 51
    || portfolio.portfolio_ledger_sha256 !== "e3dbddcfe7445128e2c763fb43acee45e0fba43f243a92f289618d1a43242ac9") {
    throw new Error(`Portfolio JSON is inconsistent: ${JSON.stringify(portfolio).slice(0, 1200)}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/input_counterexample.png");
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
    throw new Error(`Input witness Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const desktopPath = path.join(output, "input-counterexample-desktop.png");
  await page.locator("#inputCounterexamplePanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#inputCounterexamplePanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#inputCounterexamplePanel").evaluate((panel) => {
    const summary = panel.querySelector(".input-witness-summary");
    const actions = panel.querySelector(".input-witness-actions");
    const canvas = panel.querySelector("canvas");
    const scrolls = [...panel.querySelectorAll(".input-witness-table-wrap")];
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      actionColumns: getComputedStyle(actions).gridTemplateColumns.split(" ").length,
      canvasWidth: canvas.getBoundingClientRect().width,
      canvasHeight: canvas.getBoundingClientRect().height,
      tablesScrollable: scrolls.every((scroll) => scroll.scrollWidth > scroll.clientWidth),
      borderTopWidth: getComputedStyle(panel).borderTopWidth,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.actionColumns !== 2 || mobile.canvasWidth < 280 || mobile.canvasWidth > 360
    || mobile.canvasHeight < 210 || !mobile.tablesScrollable || mobile.borderTopWidth !== "2px") {
    throw new Error(`Input witness mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "input-counterexample-mobile.png");
  await page.locator("#inputCounterexamplePanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Model Input Tensor ABI Witness viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; independent verification, JSON/raw downloads, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    witnessStatus: document.querySelector("#inputCounterexampleStatus")?.textContent || null,
    panel: document.querySelector("#inputCounterexamplePanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#inputCounterexamplePanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
    let canvasPixels = 0;
    for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) canvasPixels += 1;
    const tables = panel.querySelectorAll("table");
    return {
      status: panel.querySelector("#inputCounterexampleStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".input-witness-metric").length,
      canvases: panel.querySelectorAll("canvas").length,
      tables: tables.length,
      termRows: tables[0]?.querySelectorAll("tbody tr").length || 0,
      sourceRows: tables[1]?.querySelectorAll("tbody tr").length || 0,
      canvasPixels,
      portfolioDisabled: panel.querySelector("#downloadInputCounterexample")?.disabled,
      rawDisabled: [...panel.querySelectorAll("button")].find((button) => button.textContent === "Input tensor")?.disabled,
      actionButtons: [...panel.querySelectorAll(".input-witness-actions button")].map((button) => button.textContent || ""),
      text: panel.textContent || "",
    };
  });
}

function createStaticServer(root) {
  const mime = new Map([
    [".html", "text/html; charset=utf-8"], [".js", "text/javascript; charset=utf-8"], [".mjs", "text/javascript; charset=utf-8"],
    [".css", "text/css; charset=utf-8"], [".wasm", "application/wasm"],
    [".json", "application/json; charset=utf-8"], [".tflite", "application/octet-stream"],
    [".png", "image/png"], [".svg", "image/svg+xml"],
  ]);
  return createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      let target = path.join(root, pathname.replace(/^\/+/, ""));
      if (pathname.endsWith("/")) target = path.join(target, "index.html");
      if (!path.resolve(target).startsWith(path.resolve(root)) || !existsSync(target) || !(await stat(target)).isFile()) {
        response.writeHead(404).end("not found");
        return;
      }
      response.writeHead(200, { "Content-Type": mime.get(path.extname(target).toLowerCase()) || "application/octet-stream", "Cache-Control": "no-store" });
      createReadStream(target).pipe(response);
    } catch (error) {
      response.writeHead(500).end(error.message);
    }
  });
}
