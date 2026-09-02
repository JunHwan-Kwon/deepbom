import { createServer } from "node:http";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v1_025_224_float.tflite");
const MOCK_USER = { email: "selector-contract@deepbom.test", role: "admin", name: "Selector Contract", email_verified: true };
const MOCK_ALLOWED = { report: true, export: true, raw_export: true, regulatory_report: true, deepbom: true, perturbation: true, runtime_basin: true, deployment_sensitivity: true };
const output = await mkdtemp(path.join(tmpdir(), "deepbom-protected-selector-"));
const server = createStaticServer(ROOT);
const errors = [];
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) errors.push(`console: ${message.text()}`);
  });
  const base = `http://127.0.0.1:${server.address().port}`;
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
  await page.goto(`${base}/web/`, { waitUntil: "domcontentloaded" });
  const result = await page.evaluate(async () => {
    const modelResponse = await fetch("/web/samples/mobilenet_v1_025_224_float.tflite");
    if (!modelResponse.ok) throw new Error(`sample fetch failed: ${modelResponse.status}`);
    const bytes = new Uint8Array(await modelResponse.arrayBuffer());

    const analyzer = await import("/pkg/tflite_wasm_audit.js");
    await analyzer.default({ module_or_path: "/pkg/tflite_wasm_audit_bg.wasm" });
    const analysis = analyzer.analyze_tflite_for_target(bytes, "mobilenet_v1_025_224_float.tflite", "android_mid_a55");
    const initialStatus = analysis.xnnpack_selector_assessment_status;

    const protectedModule = await import("/web/protected/deepbom/pkg/deepbom_wasm.js");
    await protectedModule.default({ module_or_path: "/web/protected/deepbom/pkg/deepbom_wasm_bg.wasm" });
    const protectedResult = protectedModule.analyze_deepbom(bytes, JSON.stringify(analysis));
    const evidence = protectedResult.xnnpack_selector_evidence;
    const merge = await import("/web/lib/xnnpack-selector-evidence.js");
    merge.applyProtectedXnnpackSelectorEvidence(analysis, evidence);
    const delegateMerge = await import("/web/lib/tflite-delegate-compatibility.js");
    delegateMerge.applyProtectedTfliteDelegateCompatibilityEvidence(
      analysis,
      protectedResult.tflite_delegate_compatibility_evidence,
    );

    const candidateOp = analysis.ops.find((op) => (op.xnnpack_kernel_candidates || []).length > 1)
      || analysis.ops.find((op) => (op.xnnpack_kernel_candidates || []).length === 1);
    if (!candidateOp) throw new Error("protected selector returned no source candidate op");

    const inspector = await import("/web/lib/kernel-inspector.js");
    const body = document.createElement("tbody");
    const summary = document.createElement("div");
    const comparisonPanel = document.createElement("div");
    const boundaryList = document.createElement("div");
    const status = document.createElement("div");
    inspector.renderKernelInspector({
      analysis,
      body,
      summary,
      comparisonPanel,
      boundaryList,
      status,
      filter: "selector",
    });

    const graph = await import("/web/lib/graph-ui.js");
    const detail = document.createElement("div");
    graph.renderOpDetailPanel(detail, analysis, candidateOp.index);
    const selectorDetail = detail.querySelector(".kernel-selector-details");
    const artifactSha = "b".repeat(64);
    analysis.model_sha256 = artifactSha;
    const identity = { filename: analysis.filename, format: "tflite", sha256: artifactSha, target_label: analysis.target_profile.label };
    const reportModule = await import("/web/lib/report-engineering.js");
    const mlBomModule = await import("/web/lib/report-mlbom.js");
    const evidenceModule = await import("/web/lib/report-evidence.js");
    const reportContext = { identity, deepBomResult: protectedResult };
    const engineeringReport = reportModule.buildEngineeringReport(analysis, reportContext);
    const mlBom = mlBomModule.buildMlBomDocument(analysis, { hash: artifactSha, fileSizeBytes: bytes.byteLength, target: analysis.target_profile, targetId: analysis.target_profile.id });
    const engineeringEvidence = evidenceModule.buildEngineeringEvidenceDocument(analysis, { reportContext, rawEvidenceContext: { identity }, mlBomDocument: mlBom });
    return {
      initialStatus,
      mergedStatus: analysis.xnnpack_selector_assessment_status,
      mergedSchema: analysis.xnnpack_selector_evidence_schema,
      access: analysis.xnnpack_selector_evidence_access,
      assessedOps: evidence.assessed_op_count,
      evidenceOps: evidence.ops.length,
      eligibleOps: analysis.ops.filter((op) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name)).length,
      candidateCount: candidateOp.xnnpack_kernel_candidates.length,
      inspectorRows: body.querySelectorAll("tr").length,
      inspectorText: body.textContent,
      inspectorSummary: summary.textContent,
      ledgerMetricCount: summary.querySelectorAll(".kernel-selector-ledger-metric").length,
      ledgerHotspotCount: summary.querySelectorAll(".kernel-selector-hotspot").length,
      inspectorStatus: status.textContent,
      opDetailText: detail.textContent,
      selectorDetailText: selectorDetail?.textContent || "",
      provenance: analysis.xnnpack_selector_evidence_provenance,
      sourceCommit: evidence.xnnpack_source_commit,
      sourceSha: candidateOp.xnnpack_kernel_candidates[0]?.source_file_sha256 || "",
      delegateSchema: analysis.tflite_delegate_compatibility_evidence?.schema,
      delegateProfiles: analysis.tflite_delegate_compatibility_evidence?.profiles?.map((profile) => ({
        id: profile.id,
        assessed: profile.assessed_graph_op_count,
        candidates: profile.source_candidate_after_artifact_precheck_count,
        exclusions: profile.definite_exclusion_count,
      })) || [],
      reportHasLedger: engineeringReport.includes("Decision-ledger metric") && engineeringReport.includes("Deterministic output-channel tail") && engineeringReport.includes("-> padded") && engineeringReport.includes("Unresolved selector dimensions"),
      reportHasAlternateDelegates: engineeringReport.includes("## TFLite GPU and NNAPI Source Compatibility (SOURCE+ARTIFACT_PRECHECK/NOT_OBSERVED)")
        && engineeringReport.includes("TFLITE_ENABLE_GPU")
        && engineeringReport.includes("TFLITE_ENABLE_NNAPI"),
      evidenceConformance: engineeringEvidence.evidence?.conformance_report?.status,
      mlBomHasEvidencePointer: mlBom.metadata.component.properties.some((item) => item.name === "deepbom:compatibility:profile" && item.value === "deepbom.compact_mlbom_compatibility.v2")
        && mlBom.metadata.component.properties.some((item) => item.name === "deepbom:compatibility:detailLocation" && item.value === "engineering_evidence.json#/evidence/static_analysis"),
    };
  });

  if (errors.length) throw new Error(`Browser errors:\n${errors.join("\n")}`);
  if (result.initialStatus !== "not_loaded") throw new Error(`Public analyzer selector status was ${result.initialStatus}.`);
  if (result.mergedStatus !== "complete" || result.mergedSchema !== "deepbom.xnnpack_selector_evidence.v2" || result.access !== "research") {
    throw new Error(`Protected selector identity was not rendered from a complete evidence merge: ${JSON.stringify(result)}.`);
  }
  if (result.assessedOps !== result.eligibleOps || result.evidenceOps !== result.eligibleOps) {
    throw new Error(`Protected selector coverage mismatch: ${JSON.stringify(result)}.`);
  }
  if (result.delegateSchema !== "deepbom.tflite_delegate_source_rulepack.v1"
    || result.delegateProfiles.length !== 2
    || result.delegateProfiles.some((profile) => profile.assessed !== result.delegateProfiles[0].assessed
      || profile.candidates + profile.exclusions !== profile.assessed)
    || !result.reportHasAlternateDelegates) {
    throw new Error(`Protected TFLite GPU/NNAPI evidence was not conserved through the report: ${JSON.stringify(result)}.`);
  }
  if (result.inspectorRows < 1 || !result.inspectorStatus.includes("Protected pinned-source selector evidence is loaded")) {
    throw new Error(`Kernel Inspector did not render protected selector state: ${JSON.stringify(result)}.`);
  }
  if (!result.inspectorText.includes("configuration") || !result.inspectorText.includes("SOURCE_ENUMERATED")) {
    throw new Error("Kernel Inspector did not expose protected candidate/evidence labels.");
  }
  if (!result.inspectorSummary.includes("Source selector decision ledger")
    || !result.inspectorSummary.includes("TFLite GPU / NNAPI source candidates")
    || !result.inspectorSummary.includes("Selected build: not imported")
    || !result.inspectorSummary.includes("Worst tail")
    || result.ledgerMetricCount !== 8
    || result.ledgerHotspotCount < 1
    || result.provenance.tail_assessed_op_count < 1
    || !Array.isArray(result.provenance.worst_case_tail_op_indices)) {
    throw new Error(`Kernel Inspector did not expose the verified selector decision ledger: ${JSON.stringify(result)}.`);
  }
  if (!result.reportHasLedger || result.evidenceConformance !== "pass" || !result.mlBomHasEvidencePointer) {
    throw new Error(`Protected selector evidence was not preserved in the report/conformance outputs or linked from the compact ML-BOM projection: ${JSON.stringify(result)}.`);
  }
  if (!result.opDetailText.includes("Artifact selector facts")
    || !result.opDetailText.includes("Unresolved selector dimensions")
    || !result.opDetailText.includes("Selector reason code")
    || !result.selectorDetailText.includes("Pinned selector matrix")
    || !result.selectorDetailText.includes("-> padded")
    || !result.selectorDetailText.includes("inactive")
    || !result.selectorDetailText.includes("Architecture:")
    || !result.selectorDetailText.includes("Compile:")
    || !result.selectorDetailText.includes("Runtime:")
    || !result.selectorDetailText.includes(result.sourceCommit)
    || !result.selectorDetailText.includes(result.sourceSha)
    || !/^[a-f0-9]{64}$/.test(result.sourceSha)) {
    throw new Error(`Op detail did not expose the complete protected selector matrix: ${JSON.stringify(result)}.`);
  }

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
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  const analysisPalette = page.locator("#analysisPalette");
  if (!await analysisPalette.isVisible()) throw new Error("Analysis palette is not visible after audit completion.");
  const deepBomStep = page.locator('[data-workflow-step="deepbom"]');
  await deepBomStep.evaluate((step) => step.scrollIntoView({ block: "nearest", inline: "center" }));
  await deepBomStep.click();
  await page.locator("#runDeepBom").click();
  await page.waitForFunction(() => {
    const appStatus = document.querySelector("#status")?.textContent || "";
    const moduleStatus = document.querySelector("#deepBomStatus")?.textContent || "";
    return appStatus.includes("DEEPBOM complete")
      || appStatus.includes("DEEPBOM failed")
      || moduleStatus === "Failed";
  }, null, { timeout: 60_000 });
  const appResult = await page.evaluate(() => {
    const selectorSection = [...document.querySelectorAll("#deepBomGrid > div")]
      .find((section) => section.querySelector(".deepbom-section-title")?.textContent === "XNNPACK Selector Evidence");
    return {
      metricText: selectorSection?.textContent || "",
      notes: document.querySelector("#deepBomNotes")?.textContent || "",
      moduleStatus: document.querySelector("#deepBomStatus")?.textContent || "",
      appStatus: document.querySelector("#status")?.textContent || "",
      commandCount: selectorSection?.querySelectorAll("button").length || 0,
    };
  });
  if (!appResult.metricText.includes("Selector Coverage")
    || !appResult.metricText.includes("Source Configurations")
    || !appResult.metricText.includes("Candidate Ops")
    || !appResult.metricText.includes("Unique / Ambiguous")
    || !appResult.metricText.includes("No-match Ops")
    || !appResult.metricText.includes("Worst Candidate Tail")
    || !appResult.metricText.includes("Unresolved Selector Gates")
    || appResult.commandCount !== 1
    || !appResult.notes.includes("selector 28 ops")) {
    throw new Error(`Advanced result did not render a complete selector summary: ${JSON.stringify(appResult)}. Browser errors: ${errors.join(" | ") || "none"}`);
  }
  await page.waitForFunction(() => !document.querySelector("#deepBomGrid .deepbom-metric.filling"), null, { timeout: 5_000 });
  const selectorSurface = page.getByRole("heading", { name: "XNNPACK Selector Evidence", exact: true }).last().locator("..");
  const desktopPath = path.join(output, "protected-selector-desktop.png");
  await selectorSurface.screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await selectorSurface.scrollIntoViewIfNeeded();
  const mobileLayout = await page.evaluate(() => {
    const button = document.querySelector("#deepBomGrid .deepbom-selector-actions button");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      buttonClip: Math.max(0, (button?.scrollWidth || 0) - (button?.clientWidth || 0)),
      overflowElements: [...document.querySelectorAll("body *")]
        .map((element) => ({
          tag: element.tagName,
          id: element.id,
          className: typeof element.className === "string" ? element.className : "",
          right: Math.round(element.getBoundingClientRect().right),
          width: Math.round(element.getBoundingClientRect().width),
        }))
        .filter((item) => item.right > document.documentElement.clientWidth + 1 && item.right < document.documentElement.clientWidth + 120)
        .sort((left, right) => right.right - left.right)
        .slice(0, 16),
      wideContainers: [...document.querySelectorAll("body *")]
        .filter((element) => element.scrollWidth > element.clientWidth + 1)
        .map((element) => ({
          tag: element.tagName,
          id: element.id,
          className: typeof element.className === "string" ? element.className : "",
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
          overflowX: getComputedStyle(element).overflowX,
        }))
        .slice(0, 16),
    };
  });
  if (mobileLayout.bodyOverflow > 1 || mobileLayout.buttonClip > 1) {
    throw new Error(`Advanced selector mobile overflow: ${JSON.stringify(mobileLayout)}.`);
  }
  const mobilePath = path.join(output, "protected-selector-mobile.png");
  await selectorSurface.screenshot({ path: mobilePath });
  await page.getByRole("button", { name: "Open Kernel Inspector" }).click();
  await page.waitForFunction(() => {
    const filter = document.querySelector('[data-kernel-filter="selector"]');
    return filter?.classList.contains("active")
      && document.querySelectorAll("#kernelInspectorBody tr:not(.kernel-empty-row)").length > 0
      && document.querySelectorAll("#kernelInspectorSummary .kernel-selector-ledger-metric").length === 8
      && document.querySelectorAll("#kernelInspectorSummary .kernel-selector-hotspot").length > 0;
  }, null, { timeout: 10_000 });
  const mobileLedgerPath = path.join(output, "selector-ledger-mobile.png");
  const ledgerSummary = page.locator("#kernelInspectorSummary");
  const inspectorPanel = page.locator("#kernelInspectorPanel");
  await inspectorPanel.evaluate((element) => {
    element.scrollIntoView({ block: "start", behavior: "instant" });
    element.scrollTop = 0;
  });
  await page.waitForTimeout(100);
  const mobileLedgerGeometry = await page.evaluate(() => {
    const panel = document.querySelector("#kernelInspectorPanel");
    const toolbar = panel?.querySelector(".kernel-inspector-toolbar");
    const summary = document.querySelector("#kernelInspectorSummary");
    const ledger = summary?.querySelector(".kernel-selector-ledger");
    const rect = (element) => element ? Object.fromEntries(["top", "right", "bottom", "left", "width", "height"].map((key) => [key, Math.round(element.getBoundingClientRect()[key])])) : null;
    return {
      panel: rect(panel),
      toolbar: rect(toolbar),
      summary: rect(summary),
      ledger: rect(ledger),
      panelScrollTop: panel?.scrollTop,
      viewportHeight: window.innerHeight,
      windowScrollY: Math.round(window.scrollY),
      documentScrollHeight: document.documentElement.scrollHeight,
      panelOffsetTop: panel?.offsetTop,
      parent: rect(panel?.parentElement),
      parentOverflowY: panel?.parentElement ? getComputedStyle(panel.parentElement).overflowY : null,
      summaryDisplay: summary ? getComputedStyle(summary).display : null,
      summaryTextLength: summary?.textContent?.length || 0,
    };
  });
  if (!mobileLedgerGeometry.summary || mobileLedgerGeometry.summaryDisplay === "none"
    || mobileLedgerGeometry.summaryTextLength < 100
    || mobileLedgerGeometry.panel.bottom <= 0
    || mobileLedgerGeometry.panel.top >= mobileLedgerGeometry.viewportHeight
    || mobileLedgerGeometry.summary.top >= mobileLedgerGeometry.viewportHeight
    || mobileLedgerGeometry.summary.top >= mobileLedgerGeometry.panel.bottom) {
    throw new Error(`Selector decision ledger is outside the mobile inspector viewport: ${JSON.stringify(mobileLedgerGeometry)}.`);
  }
  await ledgerSummary.screenshot({ path: mobileLedgerPath });
  const ledgerMobileLayout = await ledgerSummary.evaluate((element) => ({
    overflow: Math.max(0, element.scrollWidth - element.clientWidth),
    clippedButtons: [...element.querySelectorAll("button")].filter((button) => button.scrollWidth > button.clientWidth + 1).length,
  }));
  if (ledgerMobileLayout.overflow > 1 || ledgerMobileLayout.clippedButtons > 0) {
    throw new Error(`Selector decision ledger mobile overflow: ${JSON.stringify(ledgerMobileLayout)}.`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await inspectorPanel.scrollIntoViewIfNeeded();
  await inspectorPanel.evaluate((element) => { element.scrollTop = 0; });
  const desktopLedgerPath = path.join(output, "selector-ledger-desktop.png");
  await inspectorPanel.screenshot({ path: desktopLedgerPath });
  console.log(`Protected selector viewer passed (${result.assessedOps} assessed op(s), ${result.inspectorRows} unresolved selector row(s), ${result.candidateCount} candidate(s) in inspected op; full app command path and mobile overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
  console.log(`ledger_desktop=${desktopLedgerPath}`);
  console.log(`ledger_mobile=${mobileLedgerPath}`);
  console.log(`ledger_mobile_geometry=${JSON.stringify(mobileLedgerGeometry)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
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
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".wasm": "application/wasm", ".tflite": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}

function jsonRoute(route, value) {
  return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(value) });
}
