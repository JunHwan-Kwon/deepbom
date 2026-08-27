import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v1_025_224_float.tflite");
const LIVE_URL = readArgument("--url");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-runtime-viewer-"));
const server = LIVE_URL ? null : createStaticServer(ROOT);
const browserErrors = [];
let browser;
let page;

try {
  if (server) await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const pageUrl = LIVE_URL ? new URL("/", LIVE_URL).href : `http://127.0.0.1:${server.address().port}/web/`;
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const pathname = new URL(response.url()).pathname;
    if (!pathname.startsWith("/api/") && pathname !== "/favicon.ico") browserErrors.push(`http ${response.status()}: ${pathname}`);
  });
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#targetSelect").evaluate((select) => {
    select.value = "android_mid_a55";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.locator("#runAudit").waitFor({ state: "visible" });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadRuntimeAssignmentTemplate").click(),
  ]);
  const template = JSON.parse(await readFile(await download.path(), "utf8"));
  if (!template.graph_ops?.length || template.assignments?.length) throw new Error("Runtime template op reference contract failed.");
  template.runtime = {
    name: "TensorFlow Lite",
    version: "2.20.0-viewer-contract",
    backend: "XNNPACK delegate",
    build: "browser-viewer-contract-fixture",
    binary_sha256: null,
  };
  let flipped = false;
  let partition = -1;
  let previousDelegated = false;
  template.assignments = template.graph_ops.map((op) => {
    let delegated = op.predicted_delegated;
    if (!flipped && delegated) { delegated = false; flipped = true; }
    if (delegated && !previousDelegated) partition += 1;
    previousDelegated = delegated;
    return {
      op_index: op.op_index,
      op_name: op.op_name,
      provider: delegated ? "XNNPACK" : "TFLite CPU",
      delegated,
      partition_id: delegated ? String(partition) : null,
      kernel: null,
      duration_us: null,
    };
  });
  template.source.kind = "viewer_contract_fixture";
  template.source.collected_at = "2026-07-16T00:00:00.000Z";
  template.source.duration_semantics = "not_collected";
  await page.locator("#runtimeAssignmentInput").setInputFiles({
    name: "runtime_assignment.fixture.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(template)),
  });
  await page.waitForFunction(() => {
    const panel = document.querySelector("#runtimeAssignmentComparison");
    return panel && !panel.hidden && panel.querySelectorAll(".runtime-mismatch-chip").length > 0;
  }, null, { timeout: 30_000 });

  await page.locator('[data-kernel-filter="mismatch"]').click();
  const mismatchRows = await page.locator("#kernelInspectorBody tr").count();
  if (mismatchRows !== 1) throw new Error(`Mismatch filter rendered ${mismatchRows} rows instead of 1.`);
  if (!await page.locator("#kernelInspectorBody tr").first().evaluate((row) => row.classList.contains("runtime-assignment-mismatch"))) {
    throw new Error("Mismatch filter row is missing its visual classification.");
  }
  await page.locator('[data-kernel-filter="boundary"]').click();
  const boundaryText = await page.locator("#kernelBoundaryInventory").textContent();
  if (!boundaryText?.includes("Predicted internal execution-domain edges") || !boundaryText.includes("Observed internal execution-domain edges") || !boundaryText.includes("Prediction boundary deltas")) {
    throw new Error("Boundary filter did not expose predicted, observed, and delta inventories.");
  }
  await page.locator('[data-kernel-filter="selector"]').click();
  const selectorRows = await page.locator("#kernelInspectorBody tr").count();
  if (selectorRows !== 1 || !await page.locator("#kernelInspectorBody tr").first().evaluate((row) => row.classList.contains("kernel-empty-row"))) {
    throw new Error(`Public selector filter must render one explicit not-loaded row, got ${selectorRows}.`);
  }
  const selectorEmptyText = await page.locator("#kernelInspectorBody tr").first().textContent();
  if (!selectorEmptyText?.includes("Source-backed selector evidence is not loaded")) {
    throw new Error("Open static selector row did not explain the controlled evidence boundary.");
  }
  await page.locator('[data-kernel-filter="all"]').click();

  const desktop = await inspectViewer(page, template.graph_ops.length);
  const desktopPath = path.join(output, "runtime-assignment-desktop.png");
  await page.locator("#kernelInspectorPanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#runtimeAssignmentComparison").scrollIntoViewIfNeeded();
  const mobile = await inspectViewer(page, template.graph_ops.length);
  const mobilePath = path.join(output, "runtime-assignment-mobile.png");
  await page.screenshot({ path: mobilePath, fullPage: true });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  if (!desktop.visible || !mobile.visible || desktop.trackCells !== template.graph_ops.length * 2 || mobile.trackCells !== template.graph_ops.length * 2) {
    throw new Error(`Runtime viewer DOM contract failed: ${JSON.stringify({ desktop, mobile, ops: template.graph_ops.length })}`);
  }
  if (desktop.bodyOverflow > 1 || mobile.bodyOverflow > 1 || desktop.statusClip > 1 || mobile.statusClip > 1) {
    throw new Error(`Runtime viewer overflow: desktop body/status=${desktop.bodyOverflow}/${desktop.statusClip}px mobile body/status=${mobile.bodyOverflow}/${mobile.statusClip}px.`);
  }
  console.log(`Runtime assignment viewer passed (${template.graph_ops.length} ops, ${selectorRows} selector-gap row(s), ${desktop.mismatches} mismatch chip(s), desktop/mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    title: document.title,
    status: document.querySelector("#status")?.textContent || null,
    fileName: document.querySelector("#fileInput")?.files?.[0]?.name || null,
    bodyText: document.body?.textContent?.slice(0, 500) || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  if (server) await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function inspectViewer(page, opCount) {
  return page.evaluate((expectedOps) => {
    const panel = document.querySelector("#runtimeAssignmentComparison");
    const status = document.querySelector("#runtimeAssignmentStatus");
    const text = panel?.textContent || "";
    return {
      visible: Boolean(panel && !panel.hidden && text.includes("Placement match") && text.includes("Boundary match") && text.includes("Interface logical payload, predicted / observed")),
      trackCells: panel?.querySelectorAll(".runtime-assignment-cell").length || 0,
      mismatches: panel?.querySelectorAll(".runtime-mismatch-chip").length || 0,
      expectedOps,
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      statusClip: Math.max(0, (status?.scrollHeight || 0) - (status?.clientHeight || 0)),
    };
  }, opCount);
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      if (url.pathname.startsWith("/api/")) return send(response, 404, "application/json", '{"error":"not_found"}');
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(root, relative);
      if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      const bytes = await readFile(file);
      send(response, 200, mimeType(file), bytes);
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
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".tflite": "application/octet-stream", ".svg": "image/svg+xml" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}

function readArgument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || null : null;
}
