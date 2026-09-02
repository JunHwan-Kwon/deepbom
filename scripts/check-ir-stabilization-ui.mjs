import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = path.join(ROOT, ".local-validation", "1.96-stabilization", "ui");
const EVIDENCE_CLASSES = ["OBSERVED", "SOURCE_BACKED", "DERIVED", "DERIVED_WITH_HEURISTIC_THRESHOLD", "PREDICTED", "ESTIMATED", "DECLARED_UNVERIFIED", "MEASURED", "NOT_ASSESSABLE", "NOT_APPLICABLE"];
const server = createStaticServer();
const rows = [];
const diagnostics = [];
let browser;

try {
  await mkdir(OUTPUT, { recursive: true });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", (error) => diagnostics.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) diagnostics.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent !== "Waiting", null, { timeout: 30_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await runVerifiedExample(page);
  await page.locator('[data-workflow-step="audit"]').click();

  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    for (const viewport of [{ width: 1440, height: 1000, id: "desktop" }, { width: 390, height: 844, id: "mobile" }]) {
      await page.setViewportSize(viewport);
      const contrasts = await evidenceContrast(page, EVIDENCE_CLASSES);
      const failingContrast = contrasts.filter((row) => row.ratio < 4.5);
      if (failingContrast.length) throw new Error(`${theme}/${viewport.id} evidence contrast failed: ${JSON.stringify(failingContrast)}`);
      const state = await page.evaluate(() => {
        const visible = (node) => Boolean(node?.getClientRects().length);
        const auditTabs = [...document.querySelectorAll("[data-audit-tab]")];
        const workbench = document.querySelector("#auditWorkbench");
        return {
          button_count: [...document.querySelectorAll("button")].filter(visible).length,
          audit_tab_count: auditTabs.length,
          hidden_audit_tabs: auditTabs.filter((tab) => tab.hidden).map((tab) => tab.dataset.auditTab),
          primary_domain_count: auditTabs.filter((tab) => tab.dataset.auditPrimary === "true").length,
          specialized_lens_count: auditTabs.filter((tab) => tab.dataset.auditLens === "true").length,
          svg_count: [...document.querySelectorAll("svg")].filter(visible).length,
          canvas_count: [...document.querySelectorAll("canvas")].filter(visible).length,
          evidence_element_count: [...document.querySelectorAll("[data-evidence-class]")].filter(visible).length,
          document_overflow_px: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
          workbench_overflow_px: Math.max(0, (workbench?.scrollWidth || 0) - (workbench?.clientWidth || 0)),
          undersized_audit_controls: [...workbench.querySelectorAll("button,select")].filter(visible)
            .filter((node) => node.getBoundingClientRect().height < 43.5).map((node) => node.id || node.dataset.auditTab || node.textContent.trim().slice(0, 40)),
        };
      });
      if (state.audit_tab_count !== 8 || state.hidden_audit_tabs.length || state.primary_domain_count !== 5 || state.specialized_lens_count !== 3
        || state.document_overflow_px > 1 || state.workbench_overflow_px > 1 || (viewport.id === "mobile" && state.undersized_audit_controls.length)) {
        throw new Error(`${theme}/${viewport.id} audit geometry failed: ${JSON.stringify(state)}`);
      }
      const screenshot = path.join(OUTPUT, `tflite-${theme}-${viewport.id}.png`);
      await page.locator("#auditWorkbench").screenshot({ path: screenshot });
      rows.push({ artifact_format: "tflite", theme, viewport: viewport.id, ...state, minimum_evidence_contrast: Math.min(...contrasts.map((row) => row.ratio)), screenshot: path.relative(ROOT, screenshot).replaceAll("\\", "/") });
    }
  }

  await page.setViewportSize({ width: 1440, height: 1000 });
  await runAudit(page, path.join(ROOT, "web", "samples", "tinymqa1m.Q4_0.gguf"), "tinymqa1m.Q4_0.gguf");
  await page.locator('[data-workflow-step="audit"]').click();
  await page.locator('[data-audit-tab="xnnpack"]').click();
  await requireApplicabilityBoundary(page, "not_applicable", "TFLite");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#mobileAuditView").selectOption("xnnpack");
  await requireApplicabilityBoundary(page, "not_applicable", "TFLite");

  await page.setViewportSize({ width: 1440, height: 1000 });
  const heapBefore = await usedHeap(page);
  for (let round = 0; round < 3; round += 1) {
    for (const tab of ["overview", "quant", "accelerator", "roofline", "stage", "xnnpack", "quant-labs", "llm"]) {
      await page.locator(`[data-audit-tab="${tab}"]`).click();
    }
  }
  await page.requestGC().catch(() => {});
  const heapAfter = await usedHeap(page);
  if (heapBefore != null && heapAfter != null && heapAfter - heapBefore > 64 * 1024 * 1024) {
    throw new Error(`Repeated audit-tab navigation retained ${(heapAfter - heapBefore) / (1024 * 1024)} MiB.`);
  }
  const ggufState = await page.evaluate(() => ({
    hidden_audit_tabs: [...document.querySelectorAll("[data-audit-tab][hidden]")].map((tab) => tab.dataset.auditTab),
    active_tab: document.querySelector("[data-audit-tab].active")?.dataset.auditTab || null,
    document_overflow_px: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    heap_before: null,
    heap_after: null,
  }));
  ggufState.heap_before = heapBefore;
  ggufState.heap_after = heapAfter;
  if (ggufState.hidden_audit_tabs.length || ggufState.document_overflow_px > 1) throw new Error(`GGUF lens regression: ${JSON.stringify(ggufState)}`);
  rows.push({ artifact_format: "gguf", theme: "dark", viewport: "desktop", ...ggufState });
  if (diagnostics.length) throw new Error(`Browser diagnostics:\n${diagnostics.join("\n")}`);

  const manifest = {
    schema: "deepbom.ir_stabilization_ui_baseline.v1",
    source_commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim(),
    generated_at: new Date().toISOString(),
    evidence_classes: EVIDENCE_CLASSES,
    rows,
  };
  await writeFile(path.join(OUTPUT, "ui-regression-baseline.v1.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`IR stabilization UI passed (${rows.length} baseline rows; 10 evidence classes; fixed 5-domain/3-lens navigation; desktop/mobile; light/dark).`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}

async function runVerifiedExample(page) {
  await page.locator("#sampleModelSelect").selectOption("tflite-mobilenet-v2-int8");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`Verified TFLite example failed: ${status}`);
}

async function runAudit(page, modelPath, name) {
  await page.locator("#fileInput").setInputFiles({ name, mimeType: "application/octet-stream", buffer: await readFile(modelPath) });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`${name}: ${status}`);
}

async function requireApplicabilityBoundary(page, expectedStatus, reasonFragment) {
  const state = await page.locator("#auditApplicabilityBoundary").evaluate((node) => ({
    hidden: node.hidden,
    status: document.querySelector('[data-audit-lens="true"].active')?.dataset.applicabilityStatus,
    text: node.textContent || "",
    active: document.querySelector('[data-audit-lens="true"].active')?.dataset.auditTab,
  }));
  if (state.hidden || state.status !== expectedStatus || state.active !== "xnnpack" || !state.text.includes(reasonFragment)) {
    throw new Error(`Applicability boundary did not preserve the selected lens: ${JSON.stringify(state)}`);
  }
}

async function evidenceContrast(page, classes) {
  return page.evaluate((values) => {
    const host = document.createElement("div");
    host.style.cssText = "position:fixed;left:8px;top:8px;z-index:2147483647;background:var(--surface);color:var(--ink);padding:8px";
    for (const value of values) {
      const node = document.createElement("span");
      node.dataset.evidenceClass = value;
      node.textContent = value;
      node.style.cssText = "display:block;padding:6px 8px;font-size:14px;font-weight:600";
      host.append(node);
    }
    document.body.append(host);
    const parse = (value) => {
      const numbers = (value.match(/[\d.]+/g) || []).map(Number);
      return value.startsWith("color(srgb") ? numbers.slice(0, 3).map((item) => item * 255) : numbers;
    };
    const opaqueBackground = (node) => {
      let current = node;
      while (current) {
        const values = parse(getComputedStyle(current).backgroundColor);
        if (values.length >= 3 && (values[3] ?? 1) > 0.99) return values.slice(0, 3);
        current = current.parentElement;
      }
      return [255, 255, 255];
    };
    const luminance = (rgb) => {
      const channels = rgb.map((value) => { const normalized = value / 255; return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4; });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    };
    const rows = [...host.children].map((node) => {
      const foreground = parse(getComputedStyle(node).color).slice(0, 3);
      const background = opaqueBackground(node);
      const high = Math.max(luminance(foreground), luminance(background));
      const low = Math.min(luminance(foreground), luminance(background));
      return { evidence_class: node.dataset.evidenceClass, ratio: Number(((high + 0.05) / (low + 0.05)).toFixed(3)) };
    });
    host.remove();
    return rows;
  }, classes);
}

async function usedHeap(page) {
  return page.evaluate(() => Number.isFinite(performance?.memory?.usedJSHeapSize) ? performance.memory.usedJSHeapSize : null);
}

function createStaticServer() {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(ROOT, relative);
      if (!file.startsWith(`${ROOT}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      send(response, 200, mimeType(file), await readFile(file));
    } catch {
      send(response, 404, "text/plain", "not found");
    }
  });
}

function send(response, status, type, body) { response.writeHead(status, { "content-type": type, "cache-control": "no-store" }); response.end(body); }
function mimeType(file) { return ({ ".css": "text/css", ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm" })[path.extname(file).toLowerCase()] || "application/octet-stream"; }
