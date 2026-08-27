import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SERVE_ROOT = process.env.DEEPBOM_VIEWER_ROOT
  ? path.resolve(ROOT, process.env.DEEPBOM_VIEWER_ROOT)
  : ROOT;
const sourceIndex = await readFile(path.join(SERVE_ROOT, "web", "index.html"), "utf8");
const panelMarkup = extractPanel(sourceIndex);
const output = await mkdtemp(path.join(tmpdir(), "deepbom-format-capability-"));
const server = createStaticServer();
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/fixture.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__capabilityReady === true);

  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: width <= 390 ? 844 : 1000 });
      for (const format of ["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"]) {
        await page.evaluate((value) => window.renderCapability(value), format);
        const panel = page.locator("#formatCapabilityPanel");
        await panel.evaluate((node) => { node.open = false; });
        const summary = panel.locator("summary");
        const collapsedText = (await summary.innerText()).trim();
        if (collapsedText.length > 260 || !collapsedText.toLowerCase().startsWith("evidence capability")) {
          throw new Error(`Capability summary is not compact for ${format}/${width}/${theme}: ${collapsedText}`);
        }
        const backgroundBefore = await summary.evaluate((node) => getComputedStyle(node).backgroundColor);
        await summary.hover();
        await page.waitForTimeout(150);
        const backgroundHover = await summary.evaluate((node) => getComputedStyle(node).backgroundColor);
        if (backgroundBefore === backgroundHover) {
          const hoverDiagnostic = await summary.evaluate((node) => ({
            matches: node.matches(":hover"),
            variable: getComputedStyle(document.documentElement).getPropertyValue("--surface-hover"),
          }));
          throw new Error(`Capability summary has no visible hover state for ${format}/${width}/${theme}: ${backgroundBefore} -> ${backgroundHover}; ${JSON.stringify(hoverDiagnostic)}.`);
        }
        await summary.click();
        if (!(await panel.evaluate((node) => node.open))) throw new Error(`Pointer did not open ${format} capability panel.`);
        await summary.focus();
        await page.keyboard.press("Enter");
        if (await panel.evaluate((node) => node.open)) throw new Error(`Enter did not close ${format} capability panel.`);
        await page.keyboard.press("Space");
        if (!(await panel.evaluate((node) => node.open))) throw new Error(`Space did not open ${format} capability panel.`);

        const state = await panel.evaluate((node) => {
          const visible = (element) => Boolean(element?.getClientRects().length);
          const scopeWrap = node.querySelector("[data-scope-capability-table]")?.parentElement;
          const currentWrap = node.querySelector("[data-current-capability-wrap]");
          const badges = [...node.querySelectorAll(".capability-state")].filter(visible);
          const details = [...node.querySelectorAll(".capability-cell-detail")].filter(visible);
          const buttons = [...node.querySelectorAll("button")].filter(visible);
          return {
            documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
            panelOverflow: Math.max(0, node.scrollWidth - node.clientWidth),
            scopeOverflow: Math.max(0, (scopeWrap?.scrollWidth || 0) - (scopeWrap?.clientWidth || 0)),
            currentOverflow: currentWrap && visible(currentWrap)
              ? Math.max(0, currentWrap.scrollWidth - currentWrap.clientWidth)
              : 0,
            scopeRows: node.querySelectorAll("[data-scope-capability-table] tbody tr").length,
            scopeStates: node.querySelectorAll("[data-scope-capability-table] .capability-state").length,
            scopeDetailsVisible: [...node.querySelectorAll("[data-scope-capability-table] .capability-cell-detail")].filter(visible).length,
            badgeContracts: badges.map((badge) => ({
              title: badge.title,
              aria: badge.getAttribute("aria-label"),
              tabIndex: badge.tabIndex,
              clipped: badge.scrollWidth - badge.clientWidth > 1 || badge.scrollHeight - badge.clientHeight > 1,
            })),
            buttonContracts: buttons.map((button) => ({
              id: button.id,
              width: button.getBoundingClientRect().width,
              height: button.getBoundingClientRect().height,
              clipped: button.scrollWidth - button.clientWidth > 1 || button.scrollHeight - button.clientHeight > 1,
              disabled: button.disabled,
            })),
            detailsClipped: details.some((detail) => detail.scrollWidth - detail.clientWidth > 1 || detail.scrollHeight - detail.clientHeight > 1),
          };
        });
        const mobile = width <= 840;
        if (state.documentOverflow > 1 || state.panelOverflow > 1 || state.currentOverflow > 1
          || (mobile && state.scopeOverflow > 1) || state.scopeRows !== 6 || state.scopeStates !== 30
          || (mobile && state.scopeDetailsVisible !== 30) || (!mobile && state.scopeDetailsVisible !== 0)
          || state.badgeContracts.some((item) => !item.title || !item.aria || item.tabIndex !== 0 || item.clipped)
          || state.buttonContracts.some((item) => item.clipped || item.width <= 0 || (mobile && item.height < 43.5))
          || state.detailsClipped) {
          throw new Error(`Capability geometry failed for ${format}/${width}/${theme}: ${JSON.stringify(state)}`);
        }

        const firstBadge = panel.locator(".capability-state").first();
        const badgeBackground = await firstBadge.evaluate((node) => getComputedStyle(node).backgroundColor);
        await firstBadge.hover();
        await page.waitForTimeout(150);
        const badgeHoverBackground = await firstBadge.evaluate((node) => getComputedStyle(node).backgroundColor);
        if (badgeBackground === badgeHoverBackground) {
          throw new Error(`Capability badge has no visible hover state for ${format}/${width}/${theme}.`);
        }
        await firstBadge.focus();
        const focusOutline = await firstBadge.evaluate((node) => getComputedStyle(node).outlineStyle);
        if (focusOutline === "none") throw new Error(`Capability badge has no keyboard focus indicator for ${format}/${width}/${theme}.`);

        await page.evaluate(() => window.scrollTo(0, 0));
        await panel.locator("[data-scope-capability-table]").hover();
        await page.mouse.wheel(0, 420);
        await page.waitForTimeout(30);
        if (await page.evaluate(() => window.scrollY) <= 0) {
          throw new Error(`Vertical wheel was trapped by capability content for ${format}/${width}/${theme}.`);
        }

        if (format === "gguf") await verifyRuntimeButtons(page, mobile);
      }
      await page.evaluate(() => window.renderCapability("gguf"));
      const screenshot = path.join(output, `capability-${theme}-${width}.png`);
      await page.locator("#formatCapabilityPanel").screenshot({ path: screenshot });
      console.log(`${theme}_${width}=${screenshot}`);
    }
  }
  if (errors.length) throw new Error(`Browser errors: ${errors.join(" | ")}`);
  console.log("Evidence capability viewer check passed (6 formats; 4 viewports; light/dark; pointer, keyboard, hover, focus, wheel, and runtime actions). ");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

async function verifyRuntimeButtons(page, mobile) {
  await page.evaluate(() => { window.__capabilityClicks = {}; });
  const ids = [
    "importFormatRuntimeEvidence",
    "downloadFormatRuntimeTemplate",
    "downloadFormatRuntimeCapturePlan",
  ];
  for (const id of ids) {
    const control = page.locator(`#${id}`);
    if (!(await control.isVisible()) || await control.isDisabled()) {
      throw new Error(`${id} should be an enabled GGUF capability action.`);
    }
    const before = await control.evaluate((node) => ({
      background: getComputedStyle(node).backgroundColor,
      border: getComputedStyle(node).borderColor,
      color: getComputedStyle(node).color,
      height: node.getBoundingClientRect().height,
    }));
    await control.hover();
    await page.waitForTimeout(150);
    const after = await control.evaluate((node) => ({
      background: getComputedStyle(node).backgroundColor,
      border: getComputedStyle(node).borderColor,
      color: getComputedStyle(node).color,
    }));
    if (before.background === after.background && before.border === after.border && before.color === after.color) {
      throw new Error(`${id} has no visible hover state.`);
    }
    if (mobile && before.height < 43.5) throw new Error(`${id} mobile touch target is ${before.height}px.`);
    await control.click();
  }
  const clicks = await page.evaluate(() => window.__capabilityClicks);
  for (const id of ids) {
    if (clicks[id] !== 1) throw new Error(`${id} click contract failed: ${JSON.stringify(clicks)}`);
  }
}

function extractPanel(html) {
  const start = html.indexOf('<details class="format-capability-panel"');
  const end = html.indexOf("</details>", start);
  if (start < 0 || end < 0) throw new Error("Could not extract Evidence capability markup from web/index.html.");
  return html.slice(start, end + "</details>".length);
}

function createStaticServer() {
  return createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/fixture.html") {
      response.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
      response.end(`<!doctype html><html data-theme="light"><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><link rel="stylesheet" href="/web/styles.css"><link rel="stylesheet" href="/web/research-theme.css"></head><body><main style="max-width:1400px;margin:0 auto;padding:16px">${panelMarkup}<div style="height:1200px" aria-hidden="true"></div></main><script type="module" src="/fixture.js"></script></body></html>`);
      return;
    }
    if (url.pathname === "/fixture.js") {
      response.writeHead(200, { "content-type": "text/javascript; charset=utf-8", "cache-control": "no-store" });
      response.end(fixtureScript());
      return;
    }
    const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    const file = path.resolve(SERVE_ROOT, relative);
    if (!file.startsWith(`${SERVE_ROOT}${path.sep}`)) return send(response, 403, "text/plain; charset=utf-8", "forbidden");
    try {
      send(response, 200, mimeType(file), await readFile(file));
    } catch {
      send(response, 404, "text/plain; charset=utf-8", "not found");
    }
  });
}

function fixtureScript() {
  return `
    import { renderFormatCapabilityMatrix } from "/web/lib/format-capability-view.js";
    const sha = "a".repeat(64);
    const analyses = {
      tflite: { format: "tflite", filename: "mobilenet.tflite", model_sha256: sha, ops: [], inputs: [], target_profile: null },
      onnx: { format: "onnx", filename: "model.onnx", model_sha256: sha, ops: [], onnx_shape_inference: {}, mac_assessment: {}, weight_integrity: {} },
      gguf: { format: "gguf", filename: "model.gguf", model_sha256: sha, ops: [], tensor_inventory: { status: "assessed" }, tensor_numerical_integrity: { status: "assessed", assessed_tensor_count: 1, tensor_count: 1, assessed_tensor_bytes: 16, declared_tensor_bytes: 16, nonfinite_value_count: 0, exact_zero_value_count: 0, byte_conservation_status: "exact" }, gguf: { backend_compatibility: { status: "source_candidate" } } },
      safetensors: { format: "safetensors", filename: "model.safetensors", model_sha256: sha, ops: [], tensor_inventory: { status: "assessed" }, tensor_numerical_integrity: { status: "assessed", assessed_tensor_count: 1, tensor_count: 1, assessed_tensor_bytes: 16, declared_tensor_bytes: 16, nonfinite_value_count: 0, exact_zero_value_count: 0, byte_conservation_status: "exact" }, safetensors: {} },
      coreml: { format: "coreml", filename: "model.mlmodel", model_sha256: sha, ops: [{ index: 0, name: "convolution" }], mac_assessment: { assessed_compute_ops: 1, compute_ops: 1 }, tensor_liveness: { status: "assessed" }, weight_integrity: { status: "assessed", assessed_parameter_count: 1, parameter_count: 1, assessed_payload_bytes: 16, payload_bytes: 16, nonfinite_value_count: 0 }, coreml: { model_type: "mlProgram", deployment_floor: { status: "assessed" } } },
      executorch: { format: "executorch", executorch_container: "pte", filename: "model.pte", model_sha256: sha, subgraphs: 1, operator_count: 1, tensor_count: 1, ops: [{ index: 0, name: "DELEGATE:XnnpackBackend", instruction_kind: "DelegateCall" }], tensors: [{ index: 0, name: "forward/value_0", dtype: "FLOAT32", shape: [1], buffer_data_length_decimal: "4" }], weight_integrity: { assessed_tensors: 1 }, executorch_program: { delegate_instruction_count: 1, delegates: [{ backend_id: "XnnpackBackend" }] } },
    };
    window.__capabilityClicks = {};
    for (const button of document.querySelectorAll("#formatCapabilityPanel button")) {
      button.addEventListener("click", () => {
        window.__capabilityClicks[button.id] = (window.__capabilityClicks[button.id] || 0) + 1;
      });
    }
    window.renderCapability = (format) => renderFormatCapabilityMatrix(
      document.getElementById("formatCapabilityPanel"),
      format,
      { analysis: analyses[format], runtimeEvidence: null },
    );
    window.renderCapability("tflite");
    window.__capabilityReady = true;
  `;
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
