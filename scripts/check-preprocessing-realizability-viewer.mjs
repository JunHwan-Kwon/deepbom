import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";
import { decodeStoredRgbPng } from "../web/lib/rgb-png.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-preprocessing-viewer-"));
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
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 60_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 240_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="preprocessing"]').click();
  await page.locator("#preprocessingRealizabilityPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#preprocessingRealizabilityStatus")?.textContent === "independently verified", null, { timeout: 240_000 });
  await page.waitForTimeout(150);

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.canvases !== 1
    || initial.tables !== 2 || initial.inverseRows !== 9 || initial.candidateRows !== 8
    || initial.options !== 8 || initial.selected !== "0" || initial.canvasPixels < 500
    || initial.portfolioDisabled || initial.pngDisabled || initial.rgbDisabled
    || initial.actionButtons.join("|") !== "Candidate JSON|RGB fixture PNG|RGB bytes|Graph source"
    || !initial.text.includes("Exact RGB realizations4") || !initial.text.includes("raw storage rgb / raw storage bgr / artifact affine rgb / center 128 div 128 rgb")
    || !initial.text.includes("EXACT COMPLETE TENSOR REALIZATION") || !initial.text.includes("150,528 exact / 0 unrealizable")
    || !initial.text.includes("256/256") || !initial.text.includes("89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035")
    || !initial.text.includes("explicit counterfactual contracts, not observations of the production application")) {
    throw new Error(`Preprocessing viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 3200) })}`);
  }

  const [candidateDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#preprocessingRealizabilityPanel").getByRole("button", { name: "Candidate JSON", exact: true }).click(),
  ]);
  const candidatePath = path.join(output, candidateDownload.suggestedFilename());
  await candidateDownload.saveAs(candidatePath);
  const candidate = JSON.parse(await readFile(candidatePath, "utf8"));
  if (candidate.contract_id !== "raw_storage_rgb" || !candidate.exact_tensor_realization
    || candidate.exact_tensor_element_count !== 150_528 || candidate.channel_maps?.length !== 3
    || candidate.channel_maps.some((row) => row.pixel_to_tensor_codes?.length !== 256)
    || candidate.candidate_ledger_sha256 !== "538dd18a64cbc4fe77305c56f1274670da6ec2bd5745596daa780ae21f0b7468") {
    throw new Error(`Candidate JSON is inconsistent: ${JSON.stringify(candidate).slice(0, 1800)}`);
  }

  const [pngDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#preprocessingRealizabilityPanel").getByRole("button", { name: "RGB fixture PNG", exact: true }).click(),
  ]);
  const pngPath = path.join(output, pngDownload.suggestedFilename());
  await pngDownload.saveAs(pngPath);
  const png = new Uint8Array(await readFile(pngPath));
  const decoded = decodeStoredRgbPng(png);
  if (png.byteLength !== 150_830 || decoded.width !== 224 || decoded.height !== 224
    || createHash("sha256").update(decoded.rgb).digest("hex") !== candidate.nearest_rgb_fixture_sha256) {
    throw new Error(`Downloaded RGB fixture PNG is inconsistent: ${png.byteLength} bytes.`);
  }

  const [rgbDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#preprocessingRealizabilityPanel").getByRole("button", { name: "RGB bytes", exact: true }).click(),
  ]);
  const rgbPath = path.join(output, rgbDownload.suggestedFilename());
  await rgbDownload.saveAs(rgbPath);
  const rgb = await readFile(rgbPath);
  if (rgb.byteLength !== 150_528 || createHash("sha256").update(rgb).digest("hex") !== candidate.nearest_rgb_fixture_sha256) {
    throw new Error(`Downloaded RGB fixture bytes are inconsistent: ${rgb.byteLength} bytes.`);
  }

  const [portfolioDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadPreprocessingRealizability").click(),
  ]);
  const portfolioPath = path.join(output, portfolioDownload.suggestedFilename());
  await portfolioDownload.saveAs(portfolioPath);
  const portfolio = JSON.parse(await readFile(portfolioPath, "utf8"));
  if (portfolio.schema !== "deepbom.preprocessing_realizability.v1" || portfolio.candidates?.length !== 8
    || portfolio.exact_tensor_realization_candidate_count !== 4 || portfolio.non_exact_candidate_count !== 4
    || portfolio.portfolio_ledger_sha256 !== "35a1ca877c09dd440ed75e7cfbe1789c6b3693ea6572da5069cf6a01aadd5bb5") {
    throw new Error(`Preprocessing portfolio JSON is inconsistent: ${JSON.stringify(portfolio).slice(0, 1400)}`);
  }

  await page.locator(".preprocess-contract-select select").selectOption("4");
  await waitForAnimationFrames(page);
  const minusOne = await panelState(page);
  if (minusOne.selected !== "4" || !minusOne.text.includes("NON-EXACT COUNTERFACTUAL")
    || !minusOne.text.includes("26 exact / 150,502 unrealizable")
    || !minusOne.text.includes("150,502 total / 1 maximum")
    || !minusOne.text.includes("target 128; source pixel 127 round-trips to 127")
    || !minusOne.text.includes("255/256")) {
    throw new Error(`[-1,1] counterfactual view is inconsistent: ${JSON.stringify({ ...minusOne, text: minusOne.text.slice(0, 2800) })}`);
  }
  await page.locator(".preprocess-contract-select select").selectOption("5");
  await waitForAnimationFrames(page);
  const unit = await panelState(page);
  if (unit.selected !== "5" || !unit.text.includes("150,510 exact / 18 unrealizable")
    || !unit.text.includes("2,304 total / 128 maximum") || !unit.text.includes("target 0; source pixel 0 round-trips to 128")) {
    throw new Error(`[0,1] counterfactual view is inconsistent: ${JSON.stringify({ ...unit, text: unit.text.slice(0, 2800) })}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/preprocessing_realizability.png");
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
    || visualExport.colored < 800 || visualExport.opaque < 2_000) {
    throw new Error(`Preprocessing Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  await page.locator(".preprocess-contract-select select").selectOption("0");
  await waitForAnimationFrames(page);
  const desktopPath = path.join(output, "preprocessing-realizability-desktop.png");
  await page.locator("#preprocessingRealizabilityPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#preprocessingRealizabilityPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#preprocessingRealizabilityPanel").evaluate((panel) => {
    const summary = panel.querySelector(".preprocess-lab-summary");
    const actions = panel.querySelector(".preprocess-lab-actions");
    const canvas = panel.querySelector("canvas");
    const scrolls = [...panel.querySelectorAll(".preprocess-lab-table-wrap")];
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
    || mobile.canvasHeight < 600 || !mobile.tablesScrollable || mobile.borderTopWidth !== "2px") {
    throw new Error(`Preprocessing mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "preprocessing-realizability-mobile.png");
  await page.locator("#preprocessingRealizabilityPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Pixel-to-Tensor Contract Lab viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; eight contracts, independently verified PNG/RGB downloads, contract switching, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    preprocessingStatus: document.querySelector("#preprocessingRealizabilityStatus")?.textContent || null,
    panel: document.querySelector("#preprocessingRealizabilityPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#preprocessingRealizabilityPanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
    let canvasPixels = 0;
    for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) canvasPixels += 1;
    const tables = panel.querySelectorAll("table");
    const selector = panel.querySelector(".preprocess-contract-select select");
    return {
      status: panel.querySelector("#preprocessingRealizabilityStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".preprocess-lab-metric").length,
      canvases: panel.querySelectorAll("canvas").length,
      tables: tables.length,
      inverseRows: tables[0]?.querySelectorAll("tbody tr").length || 0,
      candidateRows: tables[1]?.querySelectorAll("tbody tr").length || 0,
      options: selector?.options.length || 0,
      selected: selector?.value || "",
      canvasPixels,
      portfolioDisabled: panel.querySelector("#downloadPreprocessingRealizability")?.disabled,
      pngDisabled: [...panel.querySelectorAll("button")].find((button) => button.textContent === "RGB fixture PNG")?.disabled,
      rgbDisabled: [...panel.querySelectorAll("button")].find((button) => button.textContent === "RGB bytes")?.disabled,
      actionButtons: [...panel.querySelectorAll(".preprocess-lab-actions button")].map((button) => button.textContent.trim()),
      text: panel.textContent.replace(/\s+/g, " ").trim(),
    };
  });
}

function createStaticServer(root) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".tflite": "application/octet-stream", ".png": "image/png", ".json": "application/json" };
  return createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    let file = path.resolve(root, relative || "web/index.html");
    if (!file.startsWith(path.resolve(root))) {
      response.writeHead(403).end();
      return;
    }
    if (url.pathname.endsWith("/")) file = path.join(file, "index.html");
    const stream = createReadStream(file);
    stream.on("error", () => response.writeHead(404).end());
    response.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream");
    stream.pipe(response);
  });
}
