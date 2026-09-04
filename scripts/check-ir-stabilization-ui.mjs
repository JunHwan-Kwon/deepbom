import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = path.join(ROOT, ".local-validation", "1.96-stabilization", "ui");
const EVIDENCE_CLASSES = ["OBSERVED", "SOURCE_BACKED", "DERIVED", "DERIVED_WITH_HEURISTIC_THRESHOLD", "PREDICTED", "ESTIMATED", "DECLARED_UNVERIFIED", "MEASURED", "NOT_ASSESSABLE", "NOT_APPLICABLE"];
const AUDIT_TABS = ["overview", "quant", "accelerator", "roofline", "stage", "xnnpack", "quant-labs", "llm"];
const FORMAT_CASES = [
  { format: "onnx", file: "web/samples/sample_cnn_float.onnx", name: "sample_cnn_float.onnx" },
  { format: "coreml", file: "web/samples/MNISTClassifier.mlmodel", name: "MNISTClassifier.mlmodel" },
  { format: "gguf", file: "web/samples/tinymqa1m.Q4_0.gguf", name: "tinymqa1m.Q4_0.gguf" },
  { format: "safetensors", file: "web/samples/nanofable-1m-fp16.safetensors", name: "nanofable-1m-fp16.safetensors" },
  { format: "executorch", name: "add.pte", buffer: Buffer.from(decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64)) },
];
const EXPECTED_APPLICABILITY = Object.freeze({
  tflite: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "applicable", stage: "applicable", xnnpack: "applicable", "quant-labs": "applicable", llm: "not_applicable" },
  onnx: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "applicable", stage: "applicable", xnnpack: "not_applicable", "quant-labs": "applicable", llm: "not_applicable" },
  coreml: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "not_applicable", stage: "applicable", xnnpack: "not_applicable", "quant-labs": "not_applicable", llm: "not_applicable" },
  gguf: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "not_applicable", stage: "not_applicable", xnnpack: "not_applicable", "quant-labs": "not_applicable", llm: "applicable" },
  safetensors: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "not_applicable", stage: "not_applicable", xnnpack: "not_applicable", "quant-labs": "not_applicable", llm: "applicable" },
  executorch: { overview: "applicable", quant: "applicable", accelerator: "applicable", roofline: "not_applicable", stage: "applicable", xnnpack: "not_applicable", "quant-labs": "not_applicable", llm: "not_applicable" },
});
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
  await page.locator("#dropzone").dispatchEvent("pointerdown");
  await page.locator("#fileInput").focus();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent === "Ready", null, { timeout: 30_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  const pendingNavigation = await navigationState(page);
  if (pendingNavigation.statuses.some((row) => row.status !== "not_assessed_yet") || pendingNavigation.hiddenTabs.length
    || pendingNavigation.hiddenOptions.length || pendingNavigation.disabledOptions.length) {
    throw new Error(`Pre-audit applicability drift: ${JSON.stringify(pendingNavigation)}`);
  }
  await runVerifiedExample(page);
  await verifyPrimaryWorkflowNavigation(page);
  const tfliteNavigation = await verifyFormatNavigation(page, "tflite", "desktop");
  tfliteNavigation.web_cli_semantic_digest = await verifyWebCliSemanticDigest(
    page,
    path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite"),
  );
  rows.push(tfliteNavigation);
  await verifyWhyDrawerFocusRestoration(page);

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
        throw new Error(`${theme}/${viewport.id} geometry failed: ${JSON.stringify(state)}`);
      }
      const screenshot = path.join(OUTPUT, `tflite-${theme}-${viewport.id}.png`);
      await page.locator("#auditWorkbench").screenshot({ path: screenshot });
      rows.push({ artifact_format: "tflite", theme, viewport: viewport.id, ...state, minimum_evidence_contrast: Math.min(...contrasts.map((row) => row.ratio)), screenshot: path.relative(ROOT, screenshot).replaceAll("\\", "/") });
    }
  }

  for (const entry of FORMAT_CASES) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    const artifactPath = entry.file ? path.join(ROOT, entry.file) : path.join(OUTPUT, "fixtures", entry.name);
    if (entry.buffer) {
      await mkdir(path.dirname(artifactPath), { recursive: true });
      await writeFile(artifactPath, entry.buffer);
    }
    await runAudit(page, artifactPath, entry.name, entry.buffer || null);
    await page.locator('[data-workflow-step="audit"]').click();
    const desktopNavigation = await verifyFormatNavigation(page, entry.format, "desktop");
    desktopNavigation.web_cli_semantic_digest = await verifyWebCliSemanticDigest(page, artifactPath);
    rows.push(desktopNavigation);
    await page.setViewportSize({ width: 390, height: 844 });
    rows.push(await verifyFormatNavigation(page, entry.format, "mobile"));
  }

  await page.setViewportSize({ width: 1440, height: 1000 });
  await runAudit(page, path.join(ROOT, "web", "samples", "tinymqa1m.Q4_0.gguf"), "tinymqa1m.Q4_0.gguf");
  await page.locator('[data-workflow-step="audit"]').click();
  const heapBefore = await usedHeap(page);
  for (let round = 0; round < 3; round += 1) {
    for (const tab of AUDIT_TABS) {
      await page.locator(`[data-audit-tab="${tab}"]`).click();
    }
  }
  await page.requestGC().catch(() => {});
  const heapAfter = await usedHeap(page);
  if (heapBefore != null && heapAfter != null && heapAfter - heapBefore > 64 * 1024 * 1024) {
    throw new Error(`Tab cycle retained ${(heapAfter - heapBefore) / (1024 * 1024)} MiB.`);
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
  ggufState.detached_dom = await detachedDomNodes(page);
  if (ggufState.hidden_audit_tabs.length || ggufState.document_overflow_px > 1) throw new Error(`GGUF lens regression: ${JSON.stringify(ggufState)}`);
  if (ggufState.detached_dom.count !== 0) throw new Error(`Detached DOM after tab cycle: ${JSON.stringify(ggufState.detached_dom)}`);
  rows.push({ artifact_format: "gguf", theme: "dark", viewport: "desktop", ...ggufState });
  if (diagnostics.length) throw new Error(`Errors:\n${diagnostics.join("\n")}`);

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
  if (!status.includes("audit run complete")) throw new Error(`Sample failed: ${status}${diagnostics.length ? `\n${diagnostics.join("\n")}` : ""}`);
}

async function verifyPrimaryWorkflowNavigation(page) {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const [workspace, selector] of [["input", "#artifactDossier"], ["audit", "#auditWorkbench"]]) {
      await page.locator(`[data-workflow-step="${workspace}"]`).click();
      await page.waitForTimeout(1300);
      const visible = await page.locator(selector).evaluate((node, workspaceId) => {
        const top = node.getBoundingClientRect().top;
        const bound = workspaceId !== "input" || /^[a-f\d]{64}$/.test(node.querySelector("code")?.textContent || "") && /operator|Not serialized/.test(node.textContent);
        return bound && document.body.classList.contains(`workspace-${workspaceId}`) && !node.hidden && top >= -16 && top < innerHeight;
      }, workspace);
      if (!visible) throw new Error(`${workspace}/${viewport.width}px nav failed`);
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
}

async function runAudit(page, modelPath, name, suppliedBuffer = null) {
  const buffer = suppliedBuffer || await readFile(modelPath);
  await page.locator("#fileInput").setInputFiles({ name, mimeType: "application/octet-stream", buffer });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`${name}: ${status}${diagnostics.length ? `\n${diagnostics.join("\n")}` : ""}`);
}

async function verifyFormatNavigation(page, format, viewport) {
  const expected = EXPECTED_APPLICABILITY[format];
  const before = await navigationState(page);
  const actual = Object.fromEntries(before.statuses.map((row) => [row.tab, row.status]));
  if (JSON.stringify(actual) !== JSON.stringify(expected) || before.hiddenTabs.length || before.hiddenOptions.length
    || before.disabledOptions.length || before.documentOverflowPx > 1) {
    throw new Error(`${format}/${viewport} navigation failed: ${JSON.stringify({ expected, ...before })}`);
  }
  for (const tab of AUDIT_TABS) {
    if (viewport === "mobile") await page.locator("#mobileAuditView").selectOption(tab);
    else await page.locator(`[data-audit-tab="${tab}"]`).click();
    const selected = await page.evaluate(() => {
      const selectedTab = document.querySelector("#mobileAuditView")?.value || "overview";
      const active = document.querySelector(`[data-audit-tab="${CSS.escape(selectedTab)}"]`);
      const boundary = document.querySelector("#auditApplicabilityBoundary");
      return {
        tab: active?.dataset.auditTab || null,
        status: active?.dataset.applicabilityStatus || null,
        reasonCode: active?.dataset.applicabilityReasonCode || null,
        reason: active?.dataset.applicabilityReason || "",
        boundaryHidden: boundary?.hidden ?? true,
        boundaryText: boundary?.textContent || "",
      };
    });
    if (selected.tab !== tab || selected.status !== expected[tab]) throw new Error(`${format}/${viewport}/${tab} selection drift: ${JSON.stringify(selected)}`);
    if (selected.status === "applicable") {
      if (!selected.boundaryHidden) throw new Error(`${format}/${viewport}/${tab} exposed a stale applicability boundary.`);
    } else if (selected.boundaryHidden || !selected.reasonCode || !selected.reason || !selected.boundaryText.includes(selected.reason)) {
      throw new Error(`${format}/${viewport}/${tab} missing applicability reason: ${JSON.stringify(selected)}`);
    }
  }
  const after = await navigationState(page);
  if (viewport === "mobile") {
    const undersized = await mobileTouchTargetFailures(page);
    if (undersized.length) throw new Error(`${format}/${viewport} undersized touch targets: ${JSON.stringify(undersized)}`);
  }
  return { artifact_format: format, theme: "current", viewport, navigation_status: "pass", ...after };
}

async function navigationState(page) {
  return page.evaluate(() => ({
    statuses: [...document.querySelectorAll("[data-audit-tab]")].map((tab) => ({ tab: tab.dataset.auditTab, status: tab.dataset.applicabilityStatus })),
    hiddenTabs: [...document.querySelectorAll("[data-audit-tab][hidden]")].map((tab) => tab.dataset.auditTab),
    hiddenOptions: [...document.querySelectorAll("#mobileAuditView option")].filter((option) => option.hidden).map((option) => option.value),
    disabledOptions: [...document.querySelectorAll("#mobileAuditView option")].filter((option) => option.disabled).map((option) => option.value),
    documentOverflowPx: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
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

async function detachedDomNodes(page) {
  const session = await page.context().newCDPSession(page);
  try {
    await session.send("DOM.enable");
    await session.send("HeapProfiler.enable");
    await session.send("HeapProfiler.collectGarbage");
    const result = await session.send("DOM.getDetachedDomNodes");
    const rows = [];
    for (const row of result.detachedNodes || []) {
      const backendNodeId = row.treeNode?.backendNodeId || null;
      let outerHtml = null;
      if (backendNodeId) {
        try {
          outerHtml = (await session.send("DOM.getOuterHTML", { backendNodeId })).outerHTML?.slice(0, 800) || null;
        } catch {
          // A node can disappear between the detached-node query and diagnostics.
        }
      }
      rows.push({
        node_name: row.treeNode?.nodeName || null,
        local_name: row.treeNode?.localName || null,
        child_node_count: row.treeNode?.childNodeCount || 0,
        attributes: row.treeNode?.attributes || [],
        retained_node_count: row.retainedNodeIds?.length || 0,
        outer_html: outerHtml,
      });
    }
    return {
      count: result.detachedNodes?.length || 0,
      rows,
    };
  } finally {
    await session.detach();
  }
}

async function verifyWebCliSemanticDigest(page, artifactPath) {
  await page.locator('[data-workflow-step="output"]').click();
  await page.waitForFunction(() => !document.querySelector("#downloadReviewHtml")?.disabled, null, { timeout: 30_000 });
  await page.evaluate(() => {
    const original = URL.createObjectURL.bind(URL);
    const originalAnchorClick = HTMLAnchorElement.prototype.click;
    globalThis.__deepbomCapturedReviewDownload = { original, originalAnchorClick, text: null, type: null };
    URL.createObjectURL = (blob) => {
      Promise.resolve(blob.text()).then((text) => {
        globalThis.__deepbomCapturedReviewDownload.text = text;
        globalThis.__deepbomCapturedReviewDownload.type = blob.type;
      });
      return original(blob);
    };
    HTMLAnchorElement.prototype.click = () => {};
  });
  await page.evaluate(() => document.querySelector("#downloadReviewHtml")?.click());
  await page.waitForFunction(() => Boolean(globalThis.__deepbomCapturedReviewDownload?.text), null, { timeout: 30_000 });
  const reviewHtml = await page.evaluate(() => {
    const captured = globalThis.__deepbomCapturedReviewDownload;
    URL.createObjectURL = captured.original;
    HTMLAnchorElement.prototype.click = captured.originalAnchorClick;
    delete globalThis.__deepbomCapturedReviewDownload;
    return captured.text;
  });
  const embedded = JSON.parse(reviewHtml.match(/<script id="deepbom-review-state" type="application\/json">([^<]+)<\/script>/)?.[1] || "null");
  const webSha = embedded?.artifact_ir_identity?.sha256 || null;
  if (!/^[a-f0-9]{64}$/.test(webSha || "")) throw new Error(`review.html IR SHA missing: ${JSON.stringify(embedded?.artifact_ir_identity || null)}`);
  const cliArguments = [
    path.join(ROOT, "bin", "deepbom.mjs"),
    "graph",
    artifactPath,
    "--format",
    "json",
    "--compact",
  ];
  if (embedded?.format === "tflite" && embedded?.cpu_cost_target_binding?.profile_id) {
    cliArguments.push("--target", embedded.cpu_cost_target_binding.profile_id);
  }
  const cli = JSON.parse(execFileSync(process.execPath, cliArguments, {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  }));
  const webIr = await page.evaluate(() => globalThis.__deepbomIrStabilizationTestHook?.() || null);
  if (webIr?.artifact_ir_sha256 !== webSha) {
    throw new Error(`Browser review.html and in-memory Artifact IR diverged: ${webSha} != ${webIr?.artifact_ir_sha256 || "missing"}`);
  }
  const cliSha = cli?.artifact_ir?.artifact_ir_sha256 || null;
  if (webSha !== cliSha) {
    throw new Error(`Web/CLI Artifact IR semantic digest mismatch for ${path.basename(artifactPath)}: ${webSha} != ${cliSha}; ${JSON.stringify(summarizeIrDifference(webIr, cli?.artifact_ir))}`);
  }
  await page.locator('[data-workflow-step="audit"]').click();
  return { artifact_ir_sha256: webSha, status: "equal" };
}

function summarizeIrDifference(webIr, cliIr) {
  return {
    artifact: { web: webIr?.artifact, cli: cliIr?.artifact },
    graph_totals_equal: JSON.stringify(webIr?.graph?.totals) === JSON.stringify(cliIr?.graph?.totals),
    storage_totals_equal: JSON.stringify(webIr?.storage_topology?.totals) === JSON.stringify(cliIr?.storage_topology?.totals),
    quantization_totals_equal: JSON.stringify(webIr?.quantization_contracts?.totals) === JSON.stringify(cliIr?.quantization_contracts?.totals),
    static_overlay_equal: JSON.stringify(webIr?.overlays?.static) === JSON.stringify(cliIr?.overlays?.static),
    completeness_equal: JSON.stringify(webIr?.completeness) === JSON.stringify(cliIr?.completeness),
  };
}

async function mobileTouchTargetFailures(page) {
  return page.evaluate(() => {
    const visible = (node) => Boolean(node?.getClientRects().length);
    const candidates = [...document.querySelectorAll('button, select, input[type="checkbox"], summary, [role="button"]')]
      .filter(visible)
      .filter((node) => !node.disabled && node.getAttribute("aria-hidden") !== "true");
    return candidates.map((node) => {
      const effective = node.matches('input[type="checkbox"]') && node.closest("label") || node;
      const rect = effective.getBoundingClientRect();
      return {
        id: node.id || node.dataset.auditTab || node.textContent?.trim().slice(0, 32) || node.tagName,
        width: rect.width,
        height: rect.height,
        minWidth: getComputedStyle(effective).minWidth,
        minHeight: getComputedStyle(effective).minHeight,
      };
    }).filter((row) => row.width < 43.5 || row.height < 43.5);
  });
}

async function verifyWhyDrawerFocusRestoration(page) {
  const trigger = page.locator('[data-audit-tab="overview"]');
  await trigger.focus();
  await page.evaluate(() => globalThis.dispatchEvent(new CustomEvent("deepbom:evidence-explain", { detail: {
    title: "Focus restoration fixture",
    value: "observed",
    evidence_class: "OBSERVED",
  } })));
  await page.waitForFunction(() => !document.querySelector("#evidenceWhyDrawer")?.hidden);
  if (await page.evaluate(() => document.activeElement?.id) !== "evidenceWhyDrawer") throw new Error("Evidence explanation drawer did not receive focus.");
  await page.locator("#closeEvidenceWhy").click();
  if (await page.evaluate(() => document.activeElement?.dataset?.auditTab) !== "overview") throw new Error("Evidence explanation drawer did not restore trigger focus.");
}

function createStaticServer() {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(ROOT, relative);
      if (!file.startsWith(`${ROOT}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      let body = await readFile(file);
      if (relative === "web/app.js") {
        body = Buffer.concat([
          body,
          Buffer.from("\nglobalThis.__deepbomIrStabilizationTestHook = () => currentArtifactIrContext?.artifact_ir || null;\n", "utf8"),
        ]);
      }
      send(response, 200, mimeType(file), body);
    } catch {
      send(response, 404, "text/plain", "not found");
    }
  });
}

function send(response, status, type, body) { response.writeHead(status, { "content-type": type, "cache-control": "no-store" }); response.end(body); }
function mimeType(file) { return ({ ".css": "text/css", ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm" })[path.extname(file).toLowerCase()] || "application/octet-stream"; }
