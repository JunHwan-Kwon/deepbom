import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-lattice-viewer-"));
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
  await page.waitForFunction(() => document.documentElement.dataset.analysisDepth === "deep");
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="residual-contract"]').click();
  await page.locator("#quantizationLatticePanel").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#quantizationLatticeStatus")?.textContent === "10/10 ops / 655,360 binary pairs / 0 concat codes");

  const familyState = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    title: panel.querySelector("h3")?.textContent || "",
    tabs: [...panel.querySelectorAll("[data-lattice-family]")].map((tab) => ({
      family: tab.dataset.latticeFamily,
      label: tab.querySelector("strong")?.textContent || "",
      detail: tab.querySelector("small")?.textContent || "",
      selected: tab.getAttribute("aria-selected"),
    })),
  }));
  if (familyState.title !== "Quantization Lattice Lab"
    || familyState.tabs.map((tab) => tab.family).join(",") !== "ADD,SUB,MUL,MAXIMUM,MINIMUM,CONCATENATION"
    || familyState.tabs.some((tab) => tab.label !== tab.family)
    || familyState.tabs[0].detail !== "10/10 assessed" || familyState.tabs[0].selected !== "true"
    || familyState.tabs.slice(1).some((tab) => tab.detail !== "0 in model")) {
    throw new Error(`Lattice family navigation is incomplete: ${JSON.stringify(familyState)}`);
  }
  await page.click("#quantizationLatticePanel [data-lattice-family='SUB']");
  const absentFamily = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    selected: panel.querySelector("[data-lattice-family][aria-selected='true']")?.dataset.latticeFamily || "",
    text: panel.textContent || "",
    tables: panel.querySelectorAll(".lattice-portfolio-table").length,
  }));
  if (absentFamily.selected !== "SUB" || absentFamily.tables !== 0
    || !absentFamily.text.includes("SUB contract domain")
    || !absentFamily.text.includes("This artifact serializes 0 SUB operators")
    || !absentFamily.text.includes("real=(q0-zp0)*s0-(q1-zp1)*s1")) {
    throw new Error(`Absent Lattice family state is dishonest or incomplete: ${JSON.stringify(absentFamily)}`);
  }
  await page.click("#quantizationLatticePanel [data-lattice-family='ADD']");

  const atlasState = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    rows: panel.querySelectorAll(".lattice-atlas-table tbody tr").length,
    bins: panel.querySelectorAll(".lattice-atlas-table tbody tr:first-child td.atlas-bin").length,
    text: panel.textContent || "",
  }));
  if (atlasState.rows !== 10 || atlasState.bins !== 16
    || !atlasState.text.includes("Layerwise quantization margin atlas")
    || !atlasState.text.includes("complete projection of the lattice, not a sample")) {
    throw new Error(`Lattice atlas is incomplete: ${JSON.stringify(atlasState)}`);
  }
  await page.click("#quantizationLatticePanel [data-lattice-mode='projection']");
  const projectionText = await page.locator("#quantizationLatticePanel").textContent();
  if (!projectionText.includes("margin projection") || !projectionText.includes("Co-activation path slope 1.0740")
    || !projectionText.includes("steepest-increase slope 0.9311")) {
    throw new Error("Margin projection view is incomplete.");
  }
  await page.click("#quantizationLatticePanel [data-lattice-mode='escape']");
  const state = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    status: panel.querySelector("#quantizationLatticeStatus")?.textContent || "",
    metrics: panel.querySelectorAll(".lattice-summary .lattice-metric").length,
    ops: panel.querySelectorAll(".lattice-op-selector button").length,
    rows: panel.querySelectorAll(".lattice-portfolio-table tbody tr").length,
    text: panel.textContent || "",
    canvas: panel.querySelector("canvas")?.toDataURL() || "",
    downloadDisabled: panel.querySelector("#downloadQuantizationLattice")?.disabled,
  }));
  if (state.status !== "10/10 ops / 655,360 binary pairs / 0 concat codes" || state.metrics !== 4 || state.ops !== 10 || state.rows !== 10
    || !state.text.includes("#027 ADD") || !state.text.includes("22.77%") || !state.text.includes("9.124 steps")
    || !state.text.includes("Globally finest containment") || !state.text.includes("1.896x") || !state.text.includes("zp -4")
    || state.canvas.length < 2_000 || state.downloadDisabled) throw new Error(`Lattice viewer is incomplete: ${JSON.stringify({ ...state, canvas: state.canvas.length })}`);
  const visualExport = await page.evaluate(async () => {
    const wasm = await import("/pkg/tflite_wasm_audit.js");
    const visuals = await import("/web/lib/visual-export.js");
    const bytes = new Uint8Array(await (await fetch("/web/samples/mobilenet_v2_1.0_224_quant.tflite")).arrayBuffer());
    const analysis = wasm.analyze_tflite_for_target(bytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
    const spec = visuals.visualPngSpecs({ analysis, filename: analysis.filename }).find(([name]) => name === "visuals/residual_quantization_lattice.png");
    if (!spec) return null;
    const exportedCanvas = spec[1]();
    const pixels = exportedCanvas.getContext("2d").getImageData(0, 0, exportedCanvas.width, exportedCanvas.height).data;
    let colored = 0;
    for (let index = 0; index < pixels.length; index += 128) {
      if (pixels[index] !== pixels[index + 1] || pixels[index + 1] !== pixels[index + 2]) colored += 1;
    }
    return { width: exportedCanvas.width, height: exportedCanvas.height, colored };
  });
  if (!visualExport || visualExport.width < 1180 || visualExport.height < 760
    || Math.abs(visualExport.width / visualExport.height - 1180 / 760) > 1e-9 || visualExport.colored < 1_000) {
    throw new Error(`Residual-lattice Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }
  const escapeSignature = await canvasSignature(page);
  await page.locator('[data-lattice-mode="error"]').click();
  const errorSignature = await canvasSignature(page);
  await page.locator('[data-lattice-mode="histogram"]').click();
  const histogramSignature = await canvasSignature(page);
  await page.locator('[data-lattice-mode="design"]').click();
  const designSignature = await canvasSignature(page);
  if (new Set([escapeSignature, errorSignature, histogramSignature, designSignature]).size !== 4) throw new Error("Lattice canvas modes did not render distinct pixels.");

  const canvas = page.locator("#quantizationLatticePanel canvas");
  const bounds = await canvas.boundingBox();
  const designPoint = await canvas.evaluate((node) => node._containmentDesignPoints?.find((point) => point.label.includes("zp 118")) || null);
  if (!designPoint) throw new Error("Global containment design point is not exposed for pointer inspection.");
  const designHover = {
    x: Math.max(38, Math.min(472, designPoint.x)) / 512 * bounds.width,
    y: Math.max(38, Math.min(472, designPoint.y)) / 512 * bounds.height,
  };
  await canvas.hover({ position: designHover });
  await page.locator(".lattice-tooltip:not([hidden])").waitFor({ state: "visible" });
  const designTooltip = await page.locator(".lattice-tooltip").textContent();
  if (!designTooltip.includes("zp 118") || !designTooltip.includes("1.896x")) throw new Error(`Contract-design tooltip is incomplete: ${designTooltip}`);
  const designDesktopPath = path.join(output, "residual-contract-design-desktop.png");
  await page.locator("#quantizationLatticePanel").screenshot({ path: designDesktopPath });
  await page.locator('[data-lattice-mode="histogram"]').click();
  await canvas.hover({ position: { x: bounds.width * 0.8, y: bounds.height * 0.5 } });
  await page.locator(".lattice-tooltip:not([hidden])").waitFor({ state: "visible" });
  const tooltip = await page.locator(".lattice-tooltip").textContent();
  if (!tooltip.includes("qout") || !tooltip.includes("pairs")) throw new Error(`Histogram tooltip is incomplete: ${tooltip}`);
  await page.locator('[data-lattice-mode="escape"]').click();
  await page.locator('[data-lattice-op="9"]').first().click();
  const selected = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    active: panel.querySelector(".lattice-op-selector button.active")?.textContent || "",
    selectedRow: panel.querySelector(".lattice-portfolio-table tr.selected")?.textContent || "",
    title: panel.querySelector(".lattice-detail-head strong")?.textContent || "",
  }));
  if (!selected.active.includes("#009") || !selected.selectedRow.includes("#009") || selected.title !== "#009 ADD") throw new Error(`Lattice selection is inconsistent: ${JSON.stringify(selected)}`);

  const desktopPath = path.join(output, "residual-lattice-desktop.png");
  await page.locator("#quantizationLatticePanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-lattice-mode="design"]').click();
  await page.locator("#quantizationLatticePanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#quantizationLatticePanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".lattice-summary")).gridTemplateColumns.split(" ").length,
    familyColumns: getComputedStyle(panel.querySelector(".lattice-family-tabs")).gridTemplateColumns.split(" ").length,
    familyOverflow: Math.max(0, panel.querySelector(".lattice-family-tabs").scrollWidth - panel.querySelector(".lattice-family-tabs").clientWidth),
    workspaceColumns: getComputedStyle(panel.querySelector(".lattice-workspace")).gridTemplateColumns.split(" ").length,
    modeColumns: getComputedStyle(panel.querySelector(".lattice-mode-tabs")).gridTemplateColumns.split(" ").length,
    modeOverflow: Math.max(0, panel.querySelector(".lattice-mode-tabs").scrollWidth - panel.querySelector(".lattice-mode-tabs").clientWidth),
    activeMode: panel.querySelector(".lattice-mode-tabs button.active")?.textContent || "",
    selectorScrollable: panel.querySelector(".lattice-op-selector").scrollWidth > panel.querySelector(".lattice-op-selector").clientWidth,
    tableScrollable: panel.querySelector(".lattice-portfolio-table").scrollWidth > panel.querySelector(".lattice-table-wrap").clientWidth,
    canvasWidth: panel.querySelector("canvas").getBoundingClientRect().width,
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.familyColumns !== 2 || mobile.familyOverflow > 1 || mobile.workspaceColumns !== 1
    || mobile.modeColumns !== 2 || mobile.modeOverflow > 1 || mobile.activeMode !== "Contract design"
    || !mobile.selectorScrollable || !mobile.tableScrollable || mobile.canvasWidth < 280 || mobile.canvasWidth > 360) {
    throw new Error(`Lattice mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "residual-lattice-mobile.png");
  await page.locator("#quantizationLatticePanel").screenshot({ path: mobilePath });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Quantization Lattice viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; six visible operator families, 10 ADDs, honest empty states, four distinct canvas modes, exact tooltips, desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`design_desktop=${designDesktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    lattice: document.querySelector("#quantizationLatticePanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function canvasSignature(browserPage) {
  return browserPage.locator("#quantizationLatticePanel canvas").evaluate((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    let nonWhite = 0;
    for (let index = 0; index < data.length; index += 64) {
      hash ^= data[index];
      hash = Math.imul(hash, 16777619) >>> 0;
      if (data[index] < 245 || data[index + 1] < 245 || data[index + 2] < 245) nonWhite += 1;
    }
    if (nonWhite < 200) throw new Error(`Lattice canvas is effectively blank (${nonWhite} sampled pixels).`);
    return `${hash}:${nonWhite}`;
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
