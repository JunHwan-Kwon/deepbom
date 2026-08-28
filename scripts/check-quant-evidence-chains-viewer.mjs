import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const REGRESSION_MODEL = "C:/Users/junhw/Downloads/main_0604_v119_4_ckpt902087_int8.tflite";
const DYNAMIC_RANGE_MODEL = "C:/Users/junhw/Downloads/uw20_volume_network_0528.tflite";
const MODEL = existsSync(REGRESSION_MODEL)
  ? REGRESSION_MODEL
  : path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const THEME_EVIDENCE_DIR = path.join(ROOT, ".local-validation", "quant-lab-theme");
const server = createStaticServer(ROOT);
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
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator(".quant-lab-workbench").waitFor({ state: "visible" });
  await page.locator('[data-quant-evidence-chain="integer-safety"]').waitFor({ state: "attached" });

  const state = await page.locator("#perfVisuals").evaluate((root) => {
    const chains = [...root.querySelectorAll("[data-quant-evidence-chain]")];
    const panels = chains.flatMap((chain) => [...chain.querySelectorAll(".perf-panel")]);
    const synthesis = root.querySelector('[data-quant-chain-synthesis="integer-safety"]');
    const reachability = root.querySelector("#accumulatorReachabilityPanel");
    const summaryPanels = [
      "quantRiskHeatmapPanel",
      "quantScaleDistributionPanel",
      "quantRiskDetailPanel",
      "quantHolesPanel",
    ].map((id) => root.querySelector(`#${id}`));
    return {
      chainCount: chains.length,
      chainIds: chains.map((chain) => chain.dataset.quantEvidenceChain),
      visibleChainCount: chains.filter((chain) => !chain.hidden).length,
      panelCount: panels.length,
      uniquePanelCount: new Set(panels.map((panel) => panel.id)).size,
      summaryPanelCount: summaryPanels.filter(Boolean).length,
      summaryPanelsOutsideLabs: summaryPanels.every((panel) => panel && !panel.closest(".quant-lab-workbench")),
      tabCount: root.querySelectorAll("[data-quant-lab-tab]").length,
      activeTab: root.querySelector("[data-quant-lab-tab].active")?.dataset.quantLabTab || "",
      qdqText: root.querySelector("[data-quant-qdq-action]")?.textContent || "",
      qdqCells: root.querySelectorAll(".qdq-cell").length,
      graphOps: document.querySelectorAll("#graphOpBody tr").length,
      actionCards: root.querySelectorAll(".quant-action-card").length,
      synthesisHidden: synthesis?.hidden,
      synthesisText: synthesis?.textContent || "",
      synthesisMetrics: synthesis?.querySelectorAll(".quant-chain-synthesis-metric").length || 0,
      synthesisActions: synthesis?.querySelectorAll(".quant-chain-synthesis-actions button").length || 0,
      reachabilityText: reachability?.textContent || "",
    };
  });
  if (state.chainCount !== 4
    || state.chainIds.join(",") !== "residual-contract,integer-safety,numerical-abi,preprocessing"
    || state.visibleChainCount !== 0 || state.panelCount !== 14 || state.uniquePanelCount !== 14
    || state.summaryPanelCount !== 4 || !state.summaryPanelsOutsideLabs
    || state.tabCount !== 6 || state.activeTab !== "qdq-action" || state.qdqCells < 1
    || state.qdqCells !== state.graphOps || state.actionCards !== 3
    || !state.qdqText.includes("QUANTIZE ops0") || !state.qdqText.includes("DEQUANTIZE ops0")
    || !state.qdqText.includes("Mid-graph 8-bit/FP32 boundaries0")
    || !state.qdqText.includes("Constant precision conversions0")
    || !state.qdqText.includes("PTQ experiment") || !state.qdqText.includes("NOT INDICATED")
    || !state.qdqText.includes("QAT / source review") || !state.qdqText.includes("REVIEW")) {
    throw new Error(`Quant evidence chain layout is incomplete: ${JSON.stringify(state)}`);
  }
  await verifyQuantLabThemes(page);
  if (existsSync(REGRESSION_MODEL)) {
    if (state.synthesisHidden || state.synthesisMetrics !== 5 || state.synthesisActions !== 4
      || !state.synthesisText.includes("#000 CONV_2D, channel 19")
      || !state.synthesisText.includes("97.56%")
      || !state.synthesisText.includes("27/27 weights are zero")
      || !state.synthesisText.includes("shift -30")
      || !state.synthesisText.includes("output code -128")
      || !state.reachabilityText.includes("46 exact-local source ops = 1 full model-input constructive + 45 upstream-activation unresolved")) {
      throw new Error(`Exact-channel synthesis is incomplete: ${JSON.stringify(state)}`);
    }
    await page.locator('[data-quant-lab-tab="integer-safety"]').click();
    await page.locator(".quant-chain-synthesis-actions button").filter({ hasText: "Open Requantization" }).click();
    await page.locator("#requantizationFidelityPanel").waitFor({ state: "visible" });
    const activeAfterAction = await page.locator("[data-quant-lab-tab].active").getAttribute("data-quant-lab-tab");
    if (activeAfterAction !== "numerical-abi") throw new Error(`Cross-ledger action opened ${activeAfterAction}.`);
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-quant-lab-tab="integer-safety"]').click();
  await page.locator('[data-quant-evidence-chain="integer-safety"]').scrollIntoViewIfNeeded();
  const mobile = await page.locator('[data-quant-evidence-chain="integer-safety"]').evaluate((chain) => ({
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    chainOverflow: Math.max(0, chain.scrollWidth - chain.clientWidth),
    tabsOverflowViewport: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    gridColumns: getComputedStyle(chain.querySelector(".quant-chain-grid")).gridTemplateColumns.split(" ").length,
    synthesisColumns: chain.querySelector(".quant-chain-synthesis-metrics")
      ? getComputedStyle(chain.querySelector(".quant-chain-synthesis-metrics")).gridTemplateColumns.split(" ").length
      : 0,
  }));
  if (mobile.documentOverflow > 1 || mobile.chainOverflow > 1 || mobile.tabsOverflowViewport > 1 || mobile.gridColumns !== 1
    || (existsSync(REGRESSION_MODEL) && mobile.synthesisColumns !== 2)) {
    throw new Error(`Quant evidence mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  await setTheme(page, "light");
  await page.locator('[data-quant-evidence-chain="integer-safety"]').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(THEME_EVIDENCE_DIR, "mobile-light.png"), fullPage: false });
  await setTheme(page, "dark");
  await page.locator('[data-quant-evidence-chain="integer-safety"]').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(THEME_EVIDENCE_DIR, "mobile-dark.png"), fullPage: false });
  if (existsSync(DYNAMIC_RANGE_MODEL)) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#fileInput").setInputFiles(DYNAMIC_RANGE_MODEL);
    await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
    await page.locator("#runAudit").click();
    await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 180_000 });
    const dynamicStatus = await page.locator("#status").textContent();
    if (!dynamicStatus.includes("audit run complete")) throw new Error(dynamicStatus);
    await page.locator('[data-audit-tab="quant-labs"]').click();
    await page.locator('[data-quant-lab-tab="coverage"]').click();
    const dynamic = await page.locator("#perfVisuals").evaluate((root) => ({
      coverage: root.querySelector("[data-quant-research-coverage]")?.textContent || "",
      gated: [...root.querySelectorAll("[data-quant-lab-applicability]")].map((panel) => ({
        status: panel.dataset.quantLabApplicability,
        reason: panel.dataset.quantLabReasonCode,
      })),
      text: root.textContent || "",
    }));
    if (!dynamic.coverage.includes("Dynamic-range quantized")
      || !dynamic.coverage.includes("1/15 class-supported")
      || dynamic.gated.length !== 14
      || dynamic.gated.some((row) => row.status !== "not_applicable" || row.reason !== "QR-CLASS-DYNAMIC-RANGE")
      || /evidence rejected|#undefined ADD|#000 ADD|\bNaN\b/i.test(dynamic.text)) {
      throw new Error(`Dynamic-range applicability rendering is invalid: ${JSON.stringify(dynamic).slice(0, 4000)}`);
    }
  }
  const inferredHole = await page.evaluate(async () => {
    const { renderQuantEvidenceChains } = await import("/web/lib/quant-evidence-chains.js");
    renderQuantEvidenceChains({
      format: "onnx",
      quantization_status: { classification: "mixed_quantization", label: "Mixed quantization", quantized_compute_mac_percent: 0.5 },
      quant_holes: [{ op_index: 7, adjacent_mac_percent: 0.2 }],
      ops: [{ index: 7, name: "CONV", quant_hole: true, quant_hole_detail: "serialized 8-bit/FP32 boundary operator" }],
      tensors: [],
      metadata_presence: { converter_optimization_modes: [] },
    }, document);
    const row = document.querySelector(".qdq-boundary-row.risk");
    return { rows: document.querySelectorAll(".qdq-boundary-row").length, text: row?.textContent || "" };
  });
  if (inferredHole.rows !== 1 || !inferredHole.text.includes("#007 CONV") || !inferredHole.text.includes("serialized 8-bit/FP32 boundary operator")) {
    throw new Error(`Serialized activation-boundary detail is not visible: ${JSON.stringify(inferredHole)}`);
  }
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Quant evidence chains viewer passed (4 chains, 18 unique labs, exact-channel synthesis ${existsSync(REGRESSION_MODEL) ? "verified" : "not available"}, dynamic-range gate ${existsSync(DYNAMIC_RANGE_MODEL) ? "verified" : "not available"}, mobile overflow 0).`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || "",
    body: document.body?.textContent?.slice(0, 500) || "",
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
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
  return ({
    ".css": "text/css",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".mjs": "text/javascript; charset=utf-8",
    ".tflite": "application/octet-stream",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}

async function verifyQuantLabThemes(page) {
  await mkdir(THEME_EVIDENCE_DIR, { recursive: true });
  const tabs = ["qdq-action", "coverage", "residual-contract", "integer-safety", "numerical-abi", "preprocessing"];
  for (const theme of ["light", "dark"]) {
    await setTheme(page, theme);
    for (const tab of tabs) {
      await page.locator(`[data-quant-lab-tab="${tab}"]`).click();
      await page.waitForTimeout(180);
      const violations = await page.locator(".quant-lab-workbench").evaluate(contrastViolations);
      if (violations.length) {
        throw new Error(`Quant Lab ${theme}/${tab} contrast violations: ${JSON.stringify(violations.slice(0, 12))}`);
      }
    }
    await page.locator('[data-quant-lab-tab="integer-safety"]').click();
    await page.screenshot({ path: path.join(THEME_EVIDENCE_DIR, `desktop-${theme}.png`), fullPage: false });
  }
  await setTheme(page, "light");
}

async function setTheme(page, theme) {
  const current = await page.locator("html").getAttribute("data-theme");
  if (current !== theme) await page.locator("#themeToggle").click();
  await page.waitForFunction((expected) => document.documentElement.dataset.theme === expected, theme);
  await page.waitForTimeout(180);
}

function contrastViolations(root) {
  const parse = (value) => {
    const match = String(value || "").match(/rgba?\(([^)]+)\)/i);
    if (!match) return null;
    const parts = match[1].split(/[\s,\/]+/).filter(Boolean).map(Number);
    return { r: parts[0], g: parts[1], b: parts[2], a: Number.isFinite(parts[3]) ? parts[3] : 1 };
  };
  const blend = (front, back) => ({
    r: front.r * front.a + back.r * (1 - front.a),
    g: front.g * front.a + back.g * (1 - front.a),
    b: front.b * front.a + back.b * (1 - front.a),
    a: 1,
  });
  const effectiveBackground = (element) => {
    const layers = [];
    for (let node = element; node instanceof Element; node = node.parentElement) {
      const color = parse(getComputedStyle(node).backgroundColor);
      if (color?.a > 0) layers.push(color);
    }
    let result = parse(getComputedStyle(document.documentElement).backgroundColor) || { r: 255, g: 255, b: 255, a: 1 };
    for (const layer of layers.reverse()) result = blend(layer, result);
    return result;
  };
  const luminance = ({ r, g, b }) => {
    const channel = (value) => {
      const normalized = value / 255;
      return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
    };
    return channel(r) * 0.2126 + channel(g) * 0.7152 + channel(b) * 0.0722;
  };
  const ratio = (left, right) => {
    const a = luminance(left);
    const b = luminance(right);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  };
  return [...root.querySelectorAll("*")]
    .filter((element) => {
      const style = getComputedStyle(element);
      const directText = [...element.childNodes].some((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
      return directText && style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0
        && element.getClientRects().length > 0 && !element.closest('[hidden]');
    })
    .map((element) => {
      const style = getComputedStyle(element);
      const foreground = parse(style.color);
      const background = effectiveBackground(element);
      const fontSize = Number.parseFloat(style.fontSize);
      const fontWeight = Number.parseInt(style.fontWeight, 10) || 400;
      const minimum = fontSize >= 24 || (fontSize >= 18.66 && fontWeight >= 700) ? 3 : 4.5;
      const actual = foreground ? ratio(blend(foreground, background), background) : 0;
      return {
        selector: `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ""}${element.classList.length ? `.${[...element.classList].join(".")}` : ""}`,
        parent: element.parentElement ? `${element.parentElement.tagName.toLowerCase()}.${[...element.parentElement.classList].join(".")}` : "",
        text: element.textContent.trim().replace(/\s+/g, " ").slice(0, 80),
        ratio: Number(actual.toFixed(2)),
        minimum,
        color: style.color,
        background: style.backgroundColor,
        effectiveBackground: `rgb(${Math.round(background.r)}, ${Math.round(background.g)}, ${Math.round(background.b)})`,
      };
    })
    .filter((entry) => entry.ratio + 0.01 < entry.minimum);
}
