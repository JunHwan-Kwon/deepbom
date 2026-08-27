import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "sample_cnn_float.onnx");
const MOCK_USER = { email: "ort-contract@deepbom.test", role: "admin", name: "ORT Contract", email_verified: true };
const MOCK_ALLOWED = { report: true, export: true, raw_export: true, regulatory_report: true, deepbom: true, perturbation: true, runtime_basin: true, deployment_sensitivity: true };
const LIVE_URL = readArgument("--url");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-ort-profile-viewer-"));
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
  await page.route("**/api/auth/config", (route) => jsonRoute(route, { enabled: true }));
  await page.route("**/api/auth/me", (route) => jsonRoute(route, { user: MOCK_USER }));
  await page.route("**/api/access/status", (route) => jsonRoute(route, { user: MOCK_USER, allowed: MOCK_ALLOWED }));
  await page.route("**/api/access/check", (route) => jsonRoute(route, { user: MOCK_USER, allowed: MOCK_ALLOWED }));
  await page.route("**/api/analysis-module/deepbom/manifest", (route) => jsonRoute(route, {
    ok: true,
    capability: "deepbom",
    version: "2026-07-16.2",
    module_url: "/web/protected/deepbom/pkg/deepbom_wasm.js",
    wasm_url: "/web/protected/deepbom/pkg/deepbom_wasm_bg.wasm",
    cache: "no-store",
  }));
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#targetSelect").evaluate((select) => {
    select.value = "wasm_simd";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  await page.waitForFunction(() => document.documentElement.dataset.analysisDepth === "deep");
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();
  if (await page.locator('[data-explorer-tab="kernels"]').textContent() !== "Execution Placement") {
    throw new Error("ONNX kernel explorer did not expose the common execution-placement label.");
  }
  await page.getByRole("button", { name: "Load source-backed EP analysis" }).click();
  await page.waitForFunction(() => {
    const text = document.querySelector("#kernelInspectorSummary")?.textContent || "";
    return text.includes("8 pinned source EP profile(s)") && text.includes("ORT source compatibility ledger");
  }, null, { timeout: 30_000 });
  const sourcePlacement = await page.locator("#executionPlacementPanel").textContent();
  if (!sourcePlacement?.includes("8 EP PROFILES") || !sourcePlacement.includes("Per-EP source eligibility portfolios")
    || !sourcePlacement.includes("provider-priority partition is intentionally not inferred")) {
    throw new Error(`ONNX source eligibility was not separated from joint provider placement: ${sourcePlacement}`);
  }

  const opNames = ["Conv", "Relu", "MaxPool", "Conv", "Relu", "GlobalAveragePool", "Flatten", "Gemm", "Softmax"];
  const events = opNames.flatMap((opName, index) => [
    nodeEvent(`${opName}_${index}_kernel_time`, index, opName, index % 2 ? "CPUExecutionProvider" : "XnnpackExecutionProvider", index + 1),
    nodeEvent(`${opName}_${index}_kernel_time`, index, opName, index % 2 ? "CPUExecutionProvider" : "XnnpackExecutionProvider", index + 3),
  ]);
  await page.locator("#runtimeAssignmentInput").setInputFiles({
    name: "onnxruntime_profile_1.26.0.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(events)),
  });
  await page.locator("#runtimeProfileBackdrop").waitFor({ state: "visible" });
  if (!await page.locator("#runtimeProfileBackend").evaluate((input) => input.readOnly)) {
    throw new Error("Observed execution-provider identity must be read-only in the ORT import modal.");
  }
  await page.locator("#runtimeProfileClose").focus();
  await page.keyboard.press("Shift+Tab");
  if (await page.evaluate(() => document.activeElement?.id) !== "runtimeProfileCancel") {
    throw new Error("ORT profile modal did not trap reverse keyboard focus.");
  }
  const unknownPreview = await profilePreview(page);
  if (unknownPreview.mapped !== "0/9" || !unknownPreview.importDisabled) {
    throw new Error(`Unknown optimization mode must reject unnamed index mapping: ${JSON.stringify(unknownPreview)}`);
  }
  await page.locator("#runtimeProfileOptimization").selectOption("disabled");
  await page.locator("#runtimeProfileExecutionMode").selectOption("sequential");
  const strictPreview = await profilePreview(page);
  if (strictPreview.mapped !== "9/9" || strictPreview.events !== "18/18" || strictPreview.importDisabled) {
    throw new Error(`Strict ORT mapping preview failed: ${JSON.stringify(strictPreview)}`);
  }

  const desktopModalPath = path.join(output, "ort-profile-modal-desktop.png");
  await page.locator("#runtimeProfileBackdrop").screenshot({ path: desktopModalPath });
  await page.setViewportSize({ width: 390, height: 844 });
  const mobileModal = await modalGeometry(page);
  const mobileModalPath = path.join(output, "ort-profile-modal-mobile.png");
  await page.locator("#runtimeProfileBackdrop").screenshot({ path: mobileModalPath });
  if (mobileModal.bodyOverflow > 1 || mobileModal.modalOverflow > 1 || mobileModal.actionOverflow > 1) {
    throw new Error(`ORT profile modal overflows mobile viewport: ${JSON.stringify(mobileModal)}`);
  }

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator("#runtimeProfileVersion").fill("1.26.0");
  await page.locator("#runtimeProfileBuild").fill("release; graph_optimization_level=disabled; execution_mode=sequential");
  await page.locator("#runtimeProfileCollectedAt").fill("2026-07-16T12:00");
  await page.locator("#runtimeProfileImport").click();
  await page.waitForFunction(() => {
    const status = document.querySelector("#runtimeAssignmentStatus")?.textContent || "";
    return status.includes("ONNX Runtime 1.26.0") && status.includes("mapped 18/18 kernel event(s)");
  }, null, { timeout: 30_000 });
  if (await page.locator("#runtimeProfileBackdrop").isVisible()) throw new Error("ORT profile modal did not close after import.");
  const observedPlacement = await page.locator("#executionPlacementPanel").textContent();
  if (!observedPlacement?.includes("RUNTIME OBSERVED") || !observedPlacement.includes("Observed EP assignment flow")
    || !observedPlacement.includes("CPUExecutionProvider") || !observedPlacement.includes("XnnpackExecutionProvider")) {
    throw new Error(`ONNX observed provider flow did not replace the static portfolio-only state: ${observedPlacement}`);
  }
  await page.waitForFunction(() => {
    const backdrop = document.querySelector("#runtimeProfileBackdrop");
    return document.activeElement !== document.body && !backdrop?.contains(document.activeElement);
  }, null, { timeout: 5_000 });
  const rows = await page.locator("#kernelInspectorBody tr").count();
  if (rows !== 9) throw new Error(`ORT profile import rendered ${rows} kernel rows instead of 9.`);
  await page.locator("#kernelInspectorBody tr").first().click();
  const detail = await page.locator("#opDetail").textContent();
  if (!detail?.includes("Runtime Provider Evidence") || !detail.includes("exact") && !detail.includes("optimization_disabled_unnamed_node_index_and_op_type") || !detail.includes("not exposed by the imported ORT node event")) {
    throw new Error("ONNX op detail did not expose provider mapping, timing identity, and partition/microkernel boundary.");
  }
  if (!detail.includes("Static EP assignment") || detail.includes("XNNPACK reason") || detail.includes("Weight packing")) {
    throw new Error("ONNX op detail retained TFLite-only labels or omitted the explicit static-EP assessment boundary.");
  }
  const comparison = await page.locator("#runtimeAssignmentComparison").textContent();
  if (!comparison?.includes("Provider coverage") || !comparison.includes("Observed relation edges") || comparison.includes("Placement match") || comparison.includes("Predicted")) {
    throw new Error("ORT profile import did not render provider-only evidence metrics.");
  }
  if (!comparison.includes("Selected runtime backend evidence") || !["QNN", "NNAPI", "Core ML", "WebGPU", "WebNN"].every((label) => comparison.includes(label))
    || !comparison.includes("Build") || !comparison.includes("Capability") || !comparison.includes("Assignment") || !comparison.includes("Execution")) {
    throw new Error("ORT runtime backend ledger did not render all five providers and four independent evidence layers.");
  }
  const trackCells = await page.locator("#runtimeAssignmentComparison .runtime-assignment-cell").count();
  if (trackCells !== 9) throw new Error(`ORT provider view rendered ${trackCells} track cells instead of one observed track with 9 cells.`);
  await page.locator('[data-kernel-filter="mismatch"]').click();
  const mismatchRows = await page.locator("#kernelInspectorBody tr").count();
  const mismatchText = await page.locator("#kernelInspectorBody").textContent();
  if (mismatchRows !== 1 || !mismatchText?.includes("mismatch classification is not applicable")) {
    throw new Error(`ORT mismatch filter must render one explicit not-applicable row, got ${mismatchRows}: ${mismatchText}`);
  }
  await page.locator('[data-kernel-filter="boundary"]').click();
  const boundaryText = await page.locator("#kernelBoundaryInventory").textContent();
  if (!boundaryText?.includes("Observed internal execution-domain edges") || boundaryText.includes("Predicted internal") || boundaryText.includes("Prediction boundary deltas")) {
    throw new Error("ORT boundary view must expose observed provider transitions without TFLite prediction sections.");
  }
  await page.locator('[data-kernel-filter="all"]').click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#kernelInspectorPanel").scrollIntoViewIfNeeded();
  const mobileViewer = await page.evaluate(() => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    statusClip: Math.max(0, (document.querySelector("#runtimeAssignmentStatus")?.scrollHeight || 0) - (document.querySelector("#runtimeAssignmentStatus")?.clientHeight || 0)),
  }));
  const mobileViewerPath = path.join(output, "ort-profile-viewer-mobile.png");
  await page.screenshot({ path: mobileViewerPath, fullPage: true });
  if (mobileViewer.bodyOverflow > 1 || mobileViewer.statusClip > 1) throw new Error(`ORT viewer mobile overflow: ${JSON.stringify(mobileViewer)}`);
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log("ORT profile viewer passed (9/9 ops, 18/18 events, provider-only semantics, no inferred partitions/mismatches, desktop/mobile overflow 0).");
  console.log(`desktop_modal=${desktopModalPath}`);
  console.log(`mobile_modal=${mobileModalPath}`);
  console.log(`mobile_viewer=${mobileViewerPath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    runtimeStatus: document.querySelector("#runtimeProfileStatus")?.textContent || null,
    assignmentStatus: document.querySelector("#runtimeAssignmentStatus")?.textContent || null,
    activeElement: {
      id: document.activeElement?.id || null,
      tag: document.activeElement?.tagName || null,
      connected: Boolean(document.activeElement?.isConnected),
      inBackdrop: Boolean(document.querySelector("#runtimeProfileBackdrop")?.contains(document.activeElement)),
    },
    fallbackFocus: (() => {
      const element = document.querySelector("#kernelInspectorSearch");
      return {
        connected: Boolean(element?.isConnected),
        disabled: Boolean(element?.disabled),
        rects: element?.getClientRects().length || 0,
        hiddenAncestor: Boolean(element?.closest("[hidden], [inert]")),
        hiddenPath: [...function* () {
          let node = element;
          while (node) {
            if (node.hidden || node.inert) yield `${node.tagName}#${node.id || ""}.${node.className || ""}`;
            node = node.parentElement;
          }
        }()],
      };
    })(),
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  if (server) await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function profilePreview(page) {
  return page.evaluate(() => {
    const values = [...document.querySelectorAll(".runtime-profile-preview-item strong")].map((item) => item.textContent);
    return {
      mapped: values[0] || "",
      events: values[1] || "",
      importDisabled: Boolean(document.querySelector("#runtimeProfileImport")?.disabled),
    };
  });
}

async function modalGeometry(page) {
  return page.evaluate(() => {
    const modal = document.querySelector(".runtime-profile-modal");
    const actions = document.querySelector(".runtime-profile-actions");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      modalOverflow: Math.max(0, (modal?.scrollWidth || 0) - (modal?.clientWidth || 0)),
      actionOverflow: Math.max(0, (actions?.scrollWidth || 0) - (actions?.clientWidth || 0)),
    };
  });
}

function nodeEvent(name, nodeIndex, opName, provider, durationUs) {
  return { cat: "Node", name, dur: durationUs, args: { node_index: String(nodeIndex), op_name: opName, provider } };
}

function jsonRoute(route, value) {
  return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(value) });
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
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}

function readArgument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || null : null;
}
