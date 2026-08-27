import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const index = await readFile(path.join(ROOT, "web", "index.html"), "utf8");
const panel = extractPanel(index);
const server = createServer(async (request, response) => {
  if (request.url === "/fixture.html") {
    response.setHeader("content-type", "text/html; charset=utf-8");
    response.end(fixture(panel));
    return;
  }
  const filePath = path.join(ROOT, String(request.url || "").replace(/^\/+/, ""));
  try {
    const bytes = await readFile(filePath);
    response.setHeader("content-type", contentType(filePath));
    response.end(bytes);
  } catch {
    response.statusCode = 404;
    response.end("not found");
  }
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) errors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}/fixture.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__coreIsolationReady === true);

  const initial = await state(page);
  if (initial.system !== "4" || initial.active !== "2" || initial.rows !== 4
    || !initial.text.includes("2 / 2") || !initial.text.includes("STATIC SCENARIO")
    || !initial.text.includes("Shared BW ceiling") || initial.overflow > 1) {
    throw new Error(`Initial core-allocation state is invalid: ${JSON.stringify(initial)}`);
  }

  await page.locator('[data-ai-cores="3"]').click();
  if (await page.locator(".core-isolation-segments button.active").textContent() !== "3") {
    throw new Error("Pointer selection did not activate the three-core scenario.");
  }
  await page.locator("#coreIsolationBody tr").last().focus();
  await page.keyboard.press("Enter");
  if (await page.locator(".core-isolation-segments button.active").textContent() !== "4") {
    throw new Error("Keyboard row selection did not activate the full-set baseline.");
  }
  if (!(await page.locator("#coreIsolationSummary").innerText()).includes("Full-set baseline")) {
    throw new Error("The all-core scenario was mislabeled as isolated.");
  }

  await page.locator(".core-isolation-system-select").selectOption("2");
  const dual = await state(page);
  if (dual.rows !== 2 || dual.active !== "1" || !dual.text.includes("1 / 1")) {
    throw new Error(`Dual-core variant did not recompute its available allocation set: ${JSON.stringify(dual)}`);
  }

  await page.evaluate(() => window.renderObservedPartition());
  const observed = await page.locator(".core-isolation-evidence").innerText();
  if (!observed.includes("OBSERVED ISOLATED CPUSET") || !observed.includes("CPU set 2, 3")) {
    throw new Error(`Imported cpuset evidence is not visible: ${observed}`);
  }
  await page.evaluate(() => window.renderMismatchedPartition());
  const mismatch = await page.locator(".core-isolation-evidence").innerText();
  if (!mismatch.includes("DOES NOT MATCH SCENARIO") || !mismatch.includes("imported mask has 1")) {
    throw new Error(`A runtime/static core-count mismatch is not visible: ${mismatch}`);
  }

  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => { document.documentElement.dataset.theme = value; }, theme);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 1000 });
      await page.evaluate(() => window.renderFourCore());
      const geometry = await page.locator("#coreIsolationPanel").evaluate((node) => ({
        panelOverflow: Math.max(0, node.scrollWidth - node.clientWidth),
        documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
        clipped: [...node.querySelectorAll("button, strong, td")].some((item) => item.scrollWidth - item.clientWidth > 2 && getComputedStyle(item).overflowWrap !== "anywhere"),
        buttonHeights: [...node.querySelectorAll(".core-isolation-segments button")].map((button) => button.getBoundingClientRect().height),
        cards: node.querySelectorAll("#coreIsolationBody tr").length,
      }));
      if (geometry.panelOverflow > 1 || geometry.documentOverflow > 1 || geometry.clipped || geometry.cards !== 4
        || (width < 500 && geometry.buttonHeights.some((height) => height < 43.5))) {
        throw new Error(`Core-allocation geometry failed for ${theme}/${width}: ${JSON.stringify(geometry)}`);
      }
      const button = page.locator('[data-ai-cores="3"]');
      await page.mouse.move(1, 1);
      const before = await button.evaluate((node) => ({ background: getComputedStyle(node).backgroundColor, color: getComputedStyle(node).color, shadow: getComputedStyle(node).boxShadow }));
      await button.hover();
      const after = await button.evaluate((node) => ({ background: getComputedStyle(node).backgroundColor, color: getComputedStyle(node).color, shadow: getComputedStyle(node).boxShadow }));
      if (JSON.stringify(before) === JSON.stringify(after)) throw new Error(`Core selector has no visible hover state for ${theme}/${width}.`);
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/index.html`, { waitUntil: "domcontentloaded" });
  await page.locator("#sampleModelSelect").selectOption("tflite-mobilenet-v2-int8");
  await page.locator("#trySampleModel").click();
  try {
    await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
    await page.waitForFunction(() => document.querySelectorAll("#targetSelect option").length >= 4, null, { timeout: 10_000 });
  } catch (error) {
    const boot = await page.evaluate(() => ({
      url: location.href,
      title: document.title,
      targetOptions: document.querySelectorAll("#targetSelect option").length,
      body: document.body.innerText.slice(0, 500),
    }));
    throw new Error(`Full-app boot failed: ${JSON.stringify(boot)}; browser errors: ${errors.join(" | ")}`, { cause: error });
  }
  await page.locator("#targetSelect").evaluate((select) => {
    select.value = "rpi4_a72";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(() => (
    document.querySelectorAll("#coreIsolationBody tr").length === 4
      && document.querySelector("#coreIsolationSummary")?.textContent?.includes("AI / housekeeping")
  ), null, { timeout: 120_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator('[data-audit-tab="roofline"]').click();
  await page.locator("#coreIsolationPanel:not([hidden])").waitFor({ timeout: 10_000 });
  const integrated = await page.locator("#coreIsolationPanel").evaluate((node) => ({
    hidden: node.hidden || getComputedStyle(node).display === "none",
    rows: node.querySelectorAll("#coreIsolationBody tr").length,
    active: node.querySelector(".core-isolation-segments button.active")?.textContent,
    target: document.querySelector("#targetSelect")?.value,
    overflow: Math.max(0, node.scrollWidth - node.clientWidth),
    text: node.textContent || "",
  }));
  if (integrated.hidden || integrated.rows !== 4 || integrated.active !== "2" || integrated.target !== "rpi4_a72"
    || integrated.overflow > 1 || !integrated.text.includes("192") || !integrated.text.includes("STATIC SCENARIO")) {
    throw new Error(`Full-app RPi4 integration is invalid: ${JSON.stringify(integrated)}`);
  }
  const evidenceDir = path.join(ROOT, ".local-validation", "core-isolation");
  await mkdir(evidenceDir, { recursive: true });
  await page.locator("#coreIsolationPanel").screenshot({ path: path.join(evidenceDir, "rpi4-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  const mobile = await page.locator("#coreIsolationPanel").evaluate((node) => ({
    hidden: node.hidden || getComputedStyle(node).display === "none",
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    rowDisplay: getComputedStyle(node.querySelector("#coreIsolationBody tr")).display,
    tableClipped: (() => {
      const wrap = node.querySelector(".core-isolation-table-wrap");
      return wrap.scrollHeight - wrap.clientHeight > 1;
    })(),
    segmentHeights: [...node.querySelectorAll(".core-isolation-segments button")].map((button) => button.getBoundingClientRect().height),
  }));
  if (mobile.hidden || mobile.documentOverflow > 1 || mobile.rowDisplay !== "grid" || mobile.tableClipped
    || mobile.segmentHeights.some((height) => height < 43.5)) {
    throw new Error(`Full-app mobile RPi4 integration is invalid: ${JSON.stringify(mobile)}`);
  }
  await page.locator("#coreIsolationPanel").screenshot({ path: path.join(evidenceDir, "rpi4-mobile.png") });
  if (errors.length) throw new Error(`Browser errors: ${errors.join(" | ")}`);
  console.log("Core isolation viewer check passed (full-app RPi4 audit, variants, pointer, keyboard, observed cpuset overlay, light/dark, desktop/mobile). ");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

async function state(page) {
  return page.locator("#coreIsolationPanel").evaluate((node) => ({
    system: node.querySelector(".core-isolation-system-select")?.value,
    active: node.querySelector(".core-isolation-segments button.active")?.textContent,
    rows: node.querySelectorAll("#coreIsolationBody tr").length,
    text: node.textContent || "",
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
}

function extractPanel(html) {
  const start = html.indexOf('<article class="perf-panel wide core-isolation-panel"');
  if (start < 0) throw new Error("Core isolation panel markup is missing.");
  const end = html.indexOf("</article>", start);
  if (end < 0) throw new Error("Core isolation panel markup is unterminated.");
  return html.slice(start, end + "</article>".length);
}

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js") || filePath.endsWith(".mjs")) return "text/javascript; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  if (filePath.endsWith(".svg")) return "image/svg+xml";
  if (filePath.endsWith(".png")) return "image/png";
  return "application/octet-stream";
}

function fixture(panelMarkup) {
  return `<!doctype html><html data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/web/styles.css"><link rel="stylesheet" href="/web/research-theme.css"></head><body><main style="padding:12px;max-width:1500px;margin:auto">${panelMarkup}</main><script type="module">
import { createCoreIsolationController } from "/web/lib/core-isolation-view.js";
const row = (system, ai) => ({ scenario_id: \`system-\${system}-ai-\${ai}\`, system_core_count: system, ai_assigned_core_count: ai, housekeeping_core_count: system-ai, allocation_class: ai < system ? "exclusive_isolation_candidate" : "full_set_non_isolated_baseline", int8_issue_ceiling_gops: 48*ai, fp32_issue_ceiling_gops: 12*ai, shared_memory_bandwidth_ceiling_gbps: 9.6, theoretical_roofline_floor_us: 900/ai+100, utilization_adjusted_roofline_estimate_us: 1000/ai+100, predicted_runtime_overhead_us: 20, steady_state_estimate_us: 1000/ai+120, one_time_packing_estimate_us: 50, cold_start_estimate_us: 1000/ai+170, estimate_ratio_vs_one_ai_core: 1120/(1000/ai+120) });
const context = { analysis: { core_isolation_analysis: { schema: "deepbom.core_isolation_roofline.v1", status: "assessed", evidence_class: "ESTIMATED_STATIC_RESOURCE_PARTITION", system_core_count_options: [2,4], scenarios: [row(2,1),row(2,2),row(4,1),row(4,2),row(4,3),row(4,4)], method: "Deterministic test formula.", isolation_evidence_boundary: "No scheduler or latency observation." } }, runtimeEvidence: null };
document.querySelector("#coreIsolationPanel").removeAttribute("data-visual-scope");
document.querySelector("#coreIsolationPanel").removeAttribute("data-format-scope");
document.querySelector("#coreIsolationPanel").style.display = "block";
const controller = createCoreIsolationController({ elements: { status: document.querySelector("#coreIsolationStatus"), controls: document.querySelector("#coreIsolationControls"), summary: document.querySelector("#coreIsolationSummary"), chart: document.querySelector("#coreIsolationChart"), body: document.querySelector("#coreIsolationBody"), boundary: document.querySelector("#coreIsolationBoundary") }, getContext: () => context });
window.renderObservedPartition = () => { context.runtimeEvidence = { selector_context: { invocation: { resource_partition: { requested_cpu_ids:[2,3], observed_effective_cpu_ids:[2,3], online_cpu_ids:[0,1,2,3], affinity_status:"observed_all_sampled_threads_within_requested_set", exclusive_isolation_status:"observed_cgroup_v2_isolated_partition" } } } }; controller.render(); const select = document.querySelector(".core-isolation-system-select"); if (select.value !== "4") { select.value = "4"; select.dispatchEvent(new Event("change", { bubbles:true })); } };
window.renderMismatchedPartition = () => { context.runtimeEvidence = { selector_context: { invocation: { resource_partition: { requested_cpu_ids:[3], observed_effective_cpu_ids:[3], online_cpu_ids:[0,1,2,3], affinity_status:"observed_all_sampled_threads_within_requested_set", exclusive_isolation_status:"observed_cgroup_v2_isolated_partition" } } } }; controller.render(); };
window.renderFourCore = () => { context.runtimeEvidence = null; controller.render(); const select = document.querySelector(".core-isolation-system-select"); if (select.value !== "4") { select.value = "4"; select.dispatchEvent(new Event("change", { bubbles:true })); } };
controller.render(); window.renderFourCore(); window.__coreIsolationReady = true;
</script></body></html>`;
}
