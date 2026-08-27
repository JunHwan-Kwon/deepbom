import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-numerical-abi-propagation-viewer-"));
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
  await page.locator("#numericalAbiPropagationPanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#numericalAbiPropagationStatus")?.textContent === "independently verified", null, { timeout: 240_000 });
  await page.waitForTimeout(150);

  const initial = await panelState(page);
  if (initial.status !== "independently verified" || initial.metrics !== 5 || initial.options !== 52
    || initial.selectedOption !== "0" || initial.canvases !== 2 || initial.tables !== 4
    || initial.pathRows !== 1 || initial.mergeRows !== 10 || initial.boundaryRows !== 2 || initial.rankingRows !== 20
    || initial.matrixPixels < 500 || initial.corridorPixels < 300 || initial.downloadDisabled
    || initial.facetButtons.join("|") !== "Exact local 52|Unresolved 32|Residue 9|All interval 52"
    || initial.selectedFacet !== "Exact local 52"
    || !initial.text.includes("52 / 52") || !initial.text.includes("2,239,435")
    || !initial.text.includes("77.905748% of interval divergence") || !initial.text.includes("2,021 repeated source-corridor edge instances")
    || !initial.text.includes("62.5 KiB / max 1,024 routes") || !initial.text.includes("1,024")
    || !initial.text.includes("10 reconvergences") || !initial.text.includes("2 edges; 62.5 KiB")
    || !initial.text.includes("2,918 exact / 0 residue-excluded / 1,062 unresolved states")
    || !initial.text.includes("67743dd329cabb5d6019924a46020b1ec59b6b69bb68f0136178184dfd1c4505")
    || !initial.text.includes("exact lexicographic order, no synthetic score")
    || !initial.text.includes("Exact-local qualification proves at least one bounded-sum reachable kernel-local accumulator")
    || !initial.text.includes("Downstream corridors remain tensor-level structural potential")
    || !initial.text.includes("not observed copies, latency, runtime assignment")) {
    throw new Error(`Numerical ABI viewer is incomplete: ${JSON.stringify({ ...initial, text: initial.text.slice(0, 2200) })}`);
  }

  const matrixTooltip = await hoverCanvasGeometry(page, ".abi-propagation-matrix", "__abiMatrix", (geometry) => ({
    x: geometry.left + geometry.cellWidth * 0.5,
    y: geometry.top + geometry.cellHeight * 0.5,
  }));
  if (!matrixTooltip.includes("source #0 -> op #0: source")
    || !matrixTooltip.includes("2,918 exact / 0 excluded / 1,062 unresolved")
    || !matrixTooltip.includes("1,024 output routes")) {
    throw new Error(`Matrix tooltip is incomplete: ${matrixTooltip}`);
  }
  const corridorTooltip = await hoverCanvasGeometry(page, ".abi-propagation-corridor", "__abiCorridor", (geometry) => ({
    x: geometry.nodes[0].x,
    y: geometry.nodes[0].y,
  }));
  if (!corridorTooltip.includes("#0 XNNPACK:C0")) throw new Error(`Corridor tooltip is incomplete: ${corridorTooltip}`);

  const [certificateDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Source certificate", exact: true }).click(),
  ]);
  const certificatePath = path.join(output, certificateDownload.suggestedFilename());
  await certificateDownload.saveAs(certificatePath);
  const certificate = JSON.parse(await readFile(certificatePath, "utf8"));
  if (certificate.schema !== "deepbom.numerical_abi_propagation_source.v1.1"
    || certificate.graph_ledger_sha256 !== "32f61ca0603b0b1da6749600ecde950293feffc2c74b56ededbb3f62d07e93aa"
    || certificate.source?.op_index !== 0 || certificate.source?.corridor_edge_count !== 74
    || certificate.source?.local_reachability_status !== "exact_local_counterexample"
    || certificate.source?.exact_reachable_divergent_state_count_decimal !== "2918"
    || certificate.source?.provably_unreachable_divergent_state_count_decimal !== "0"
    || certificate.source?.unresolved_divergent_state_count_decimal !== "1062"
    || certificate.source?.source_reachability_ledger_sha256 !== "67743dd329cabb5d6019924a46020b1ec59b6b69bb68f0136178184dfd1c4505"
    || certificate.source?.exact_model_output_graph_route_count_decimal !== "1024"
    || certificate.source?.propagation_ledger_sha256 !== "0318b9ba433a62b8543aa3286130052cccea90769e0fe23d43106c54e2365229"
    || certificate.referenced_graph_edges?.length !== 74) {
    throw new Error(`Selected source certificate is inconsistent: ${JSON.stringify(certificate).slice(0, 2000)}`);
  }

  const [portfolioDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadNumericalAbiPropagation").click(),
  ]);
  const portfolioPath = path.join(output, portfolioDownload.suggestedFilename());
  await portfolioDownload.saveAs(portfolioPath);
  const portfolio = JSON.parse(await readFile(portfolioPath, "utf8"));
  if (portfolio.schema !== "deepbom.numerical_abi_propagation.v1.1" || portfolio.sources?.length !== 53
    || portfolio.graph_edges?.length !== 74 || portfolio.maximum_model_output_graph_route_count_decimal !== "1024"
    || portfolio.exact_local_counterexample_source_op_count !== 52
    || portfolio.residue_excluded_divergence_source_op_count !== 9
    || portfolio.unresolved_divergence_source_op_count !== 32
    || portfolio.exact_local_divergent_state_count_decimal !== "2239435"
    || portfolio.residue_excluded_divergent_state_count_decimal !== "3585"
    || portfolio.unresolved_divergent_state_count_decimal !== "631524") {
    throw new Error(`Portfolio download is inconsistent: ${JSON.stringify(portfolio).slice(0, 1200)}`);
  }

  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/numerical_abi_propagation.png");
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
    throw new Error(`Numerical ABI Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  await page.getByRole("button", { name: "Unresolved 32", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".abi-propagation-facet.selected")?.textContent === "Unresolved 32");
  const unresolved = await panelState(page);
  if (unresolved.options !== 32 || unresolved.selectedOption !== "0" || unresolved.rankingRows !== 20
    || unresolved.selectedFacet !== "Unresolved 32" || !unresolved.text.includes("unresolved evidence facet x artifact ops")) {
    throw new Error(`Unresolved source facet is inconsistent: ${JSON.stringify({ ...unresolved, text: unresolved.text.slice(0, 1200) })}`);
  }

  await page.getByRole("button", { name: "Residue 9", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".abi-propagation-facet.selected")?.textContent === "Residue 9");
  const residue = await panelState(page);
  if (residue.options !== 9 || residue.selectedOption !== "1" || residue.rankingRows !== 9
    || residue.selectedFacet !== "Residue 9" || !residue.text.includes("excluded evidence facet x artifact ops")
    || !residue.text.includes("3,729 exact / 322 residue-excluded / 930 unresolved states")) {
    throw new Error(`Residue source facet is inconsistent: ${JSON.stringify({ ...residue, text: residue.text.slice(0, 1200) })}`);
  }

  await page.getByRole("button", { name: "All interval 52", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".abi-propagation-facet.selected")?.textContent === "All interval 52");
  const allInterval = await panelState(page);
  if (allInterval.options !== 52 || allInterval.rankingRows !== 20 || allInterval.selectedFacet !== "All interval 52"
    || !allInterval.text.includes("all evidence facet x artifact ops")) {
    throw new Error(`All-interval source facet is inconsistent: ${JSON.stringify({ ...allInterval, text: allInterval.text.slice(0, 1200) })}`);
  }

  await page.getByRole("button", { name: "Exact local 52", exact: true }).click();
  await page.waitForFunction(() => document.querySelector(".abi-propagation-facet.selected")?.textContent === "Exact local 52");

  await page.locator(".abi-propagation-source-select").selectOption("63");
  await waitForAnimationFrames(page);
  const classifier = await panelState(page);
  if (!classifier.text.includes("1 distinct graph routes") || !classifier.text.includes("1 ops, 2 tensors, 1 unique graph edges")
    || !classifier.text.includes("0 reconvergences; 0 single-branch merges")
    || !classifier.text.includes("The selected shortest downstream corridor remains within one predicted execution domain.")) {
    throw new Error(`Classifier source corridor is incomplete: ${JSON.stringify({ ...classifier, text: classifier.text.slice(0, 1800) })}`);
  }

  const desktopPath = path.join(output, "numerical-abi-propagation-desktop.png");
  await page.locator("#numericalAbiPropagationPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#numericalAbiPropagationPanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#numericalAbiPropagationPanel").evaluate((panel) => {
    const summary = panel.querySelector(".abi-propagation-summary");
    const toolbar = panel.querySelector(".abi-propagation-toolbar");
    const actions = panel.querySelector(".abi-propagation-actions");
    const facets = panel.querySelector(".abi-propagation-facets");
    const select = panel.querySelector(".abi-propagation-source-select");
    const canvases = [...panel.querySelectorAll("canvas")];
    const scrolls = [...panel.querySelectorAll(".abi-propagation-table-scroll")];
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      toolbarColumns: getComputedStyle(toolbar).gridTemplateColumns.split(" ").length,
      actionColumns: getComputedStyle(actions).gridTemplateColumns.split(" ").length,
      facetColumns: getComputedStyle(facets).gridTemplateColumns.split(" ").length,
      facetOverflow: Math.max(0, facets.scrollWidth - facets.clientWidth),
      selectFits: select.getBoundingClientRect().width <= panel.getBoundingClientRect().width,
      tablesScrollable: scrolls.every((scroll) => scroll.scrollWidth > scroll.clientWidth),
      canvasWidths: canvases.map((canvas) => canvas.getBoundingClientRect().width),
      borderTopWidth: getComputedStyle(panel).borderTopWidth,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.toolbarColumns !== 1 || mobile.actionColumns !== 2 || mobile.facetColumns !== 2 || mobile.facetOverflow > 1
    || !mobile.selectFits || !mobile.tablesScrollable
    || mobile.canvasWidths.some((width) => width < 280 || width > 360) || mobile.borderTopWidth !== "2px") {
    throw new Error(`Numerical ABI mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "numerical-abi-propagation-mobile.png");
  await page.locator("#numericalAbiPropagationPanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Numerical ABI Propagation viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; exact/unresolved/residue/all facets, independent verification, matrix/corridor tooltips, two JSON contracts, Visual PNG, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    propagationStatus: document.querySelector("#numericalAbiPropagationStatus")?.textContent || null,
    panel: document.querySelector("#numericalAbiPropagationPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#numericalAbiPropagationPanel").evaluate((panel) => {
    const pixelCount = (canvas) => {
      const data = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
      let colored = 0;
      for (let index = 0; index < data.length; index += 128) if (data[index] !== data[index + 1] || data[index + 1] !== data[index + 2]) colored += 1;
      return colored;
    };
    const canvases = panel.querySelectorAll("canvas");
    const tables = panel.querySelectorAll("table");
    return {
      status: panel.querySelector("#numericalAbiPropagationStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".abi-propagation-metric").length,
      options: panel.querySelectorAll(".abi-propagation-source-select option").length,
      selectedOption: panel.querySelector(".abi-propagation-source-select")?.value || "",
      facetButtons: [...panel.querySelectorAll(".abi-propagation-facet")].map((button) => button.textContent || ""),
      selectedFacet: panel.querySelector(".abi-propagation-facet.selected")?.textContent || "",
      canvases: canvases.length,
      tables: tables.length,
      pathRows: tables[0]?.querySelectorAll("tbody tr").length || 0,
      mergeRows: tables[1]?.querySelectorAll("tbody tr").length || 0,
      boundaryRows: tables[2]?.querySelectorAll("tbody tr").length || 0,
      rankingRows: tables[3]?.querySelectorAll("tbody tr").length || 0,
      matrixPixels: pixelCount(canvases[0]),
      corridorPixels: pixelCount(canvases[1]),
      downloadDisabled: panel.querySelector("#downloadNumericalAbiPropagation")?.disabled,
      text: panel.textContent || "",
    };
  });
}

async function hoverCanvasGeometry(browserPage, selector, geometryKey, coordinate) {
  await browserPage.locator(selector).scrollIntoViewIfNeeded();
  const point = await browserPage.locator(selector).evaluate((canvas, { geometryKey }) => {
    const geometry = canvas[geometryKey];
    const rect = canvas.getBoundingClientRect();
    return { rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height }, width: canvas.width, height: canvas.height, geometry };
  }, { geometryKey });
  const local = coordinate(point.geometry);
  await browserPage.mouse.move(point.rect.x + local.x * point.rect.width / point.width, point.rect.y + local.y * point.rect.height / point.height);
  await browserPage.waitForTimeout(80);
  return browserPage.locator(`${selector} + .abi-propagation-tooltip`).textContent();
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
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".js") || file.endsWith(".mjs")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".css")) return "text/css; charset=utf-8";
  if (file.endsWith(".wasm")) return "application/wasm";
  if (file.endsWith(".json")) return "application/json; charset=utf-8";
  return "application/octet-stream";
}
