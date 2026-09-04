import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { readVersionContract } from "./version-contract.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VERSION = await readVersionContract(ROOT);
const SERVE_ROOT = process.env.DEEPBOM_VIEWER_ROOT
  ? path.resolve(ROOT, process.env.DEEPBOM_VIEWER_ROOT)
  : ROOT;
const TFLITE = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const ONNX = path.join(ROOT, "web", "samples", "sample_cnn_float.onnx");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-explorer-redesign-"));
const graphScreenshot = path.join(output, "top-down-model-graph.png");
const graphDarkScreenshot = path.join(output, "top-down-model-graph-dark.png");
const graphMobileScreenshot = path.join(output, "top-down-model-graph-mobile.png");
const firstVisitMobileScreenshot = path.join(output, "first-visit-mobile.png");
const blocksScreenshot = path.join(output, "blocks-light.png");
const resourceMapScreenshot = path.join(output, "resource-map-light.png");
const resourceMapDarkScreenshot = path.join(output, "resource-map-dark.png");
const operatorScreenshot = path.join(output, "operator-stage-light.png");
const tensorsScreenshot = path.join(output, "tensors-light.png");
const blocksDarkScreenshot = path.join(output, "blocks-dark.png");
const redesignScenarioScreenshot = path.join(output, "redesign-scenarios.png");
const redesignGraphScreenshot = path.join(output, "redesign-full-graph.png");
const server = createStaticServer(SERVE_ROOT);
const errors = [];
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) {
      errors.push(`console: ${message.text()}`);
    }
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 180_000 });
  await page.setViewportSize({ width: 320, height: 568 });
  await page.waitForFunction(() => !document.querySelector("#agreementBackdrop")?.hidden);
  const compactPrivacy = await page.evaluate(() => {
    const modal = document.querySelector(".agreement-modal");
    const scroll = document.querySelector(".agreement-scroll");
    const button = document.querySelector("#acceptAgreement");
    const modalRect = modal?.getBoundingClientRect();
    const buttonRect = button?.getBoundingClientRect();
    return {
      modalLeft: modalRect?.left || 0,
      modalRight: modalRect?.right || 0,
      modalTop: modalRect?.top || 0,
      modalBottom: modalRect?.bottom || 0,
      buttonHeight: buttonRect?.height || 0,
      buttonBottom: buttonRect?.bottom || 0,
      scrollable: Number(scroll?.scrollHeight || 0) > Number(scroll?.clientHeight || 0),
      documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  });
  if (compactPrivacy.modalLeft < 7 || compactPrivacy.modalRight > 313 || compactPrivacy.modalTop < 7
    || compactPrivacy.modalBottom > 561 || compactPrivacy.buttonHeight < 44 || compactPrivacy.buttonBottom > 561
    || !compactPrivacy.scrollable || compactPrivacy.documentOverflow > 1) {
    throw new Error(`Compact privacy acknowledgement is not viewport-safe: ${JSON.stringify(compactPrivacy)}`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await acceptAgreement(page);
  const desktopProvenance = await page.evaluate(() => ({
    citationVisible: Boolean(document.querySelector("#copyCitationBtn")?.getClientRects().length),
    buildVisible: Boolean(document.querySelector("#applicationBuild")?.getClientRects().length),
    doiVisible: Boolean(document.querySelector(".citation-doi")?.getClientRects().length),
    affiliationVisible: Boolean(document.querySelector(".author-affiliation")?.getClientRects().length),
    affiliation: document.querySelector(".author-affiliation")?.textContent || "",
    linkedinVisible: Boolean(document.querySelector(".author-social-link")?.getClientRects().length),
    linkedinHref: document.querySelector(".author-social-link")?.href || "",
    summaryPresent: Boolean(document.querySelector(".author-details > summary")),
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!desktopProvenance.citationVisible || !desktopProvenance.buildVisible || !desktopProvenance.doiVisible
    || !desktopProvenance.affiliationVisible || !desktopProvenance.affiliation.includes("Yonsei University College of Medicine")
    || !desktopProvenance.linkedinVisible || desktopProvenance.linkedinHref !== "https://www.linkedin.com/in/jun-hwan-kwon"
    || desktopProvenance.summaryPresent || desktopProvenance.overflow > 1) {
    throw new Error(`Desktop provenance must remain visible and viewport-safe: ${JSON.stringify(desktopProvenance)}`);
  }
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.textContent?.includes("65 ops"));
  await page.locator("#sampleModelSelect").selectOption("onnx-mnist-8");
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.textContent?.includes("12 ops / 21 tensors / 786,560 MACs"));
  await page.locator("#sampleModelSelect").selectOption("tflite-mobilenet-v2-int8");
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.textContent?.includes("65 ops / 173 tensors"));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  const firstVisit = await page.evaluate(() => {
    const intro = document.querySelector(".artifact-intro");
    const glance = document.querySelector("#sampleEvidenceGlance");
    const context = document.querySelector(".first-visit-context");
    const guide = document.querySelector("#preAuditReference");
    const meta = document.querySelector(".topbar-meta");
    const build = document.querySelector("#applicationBuild");
    const authorDetails = document.querySelector(".author-details");
    const evidenceDetails = document.querySelector(".evidence-class-disclosure");
    const medical = document.querySelector("#medicalEvidenceTitle")?.closest("section");
    const medicalDetail = document.querySelector(".medical-context-detail");
    const controls = document.querySelector(".upload-controls");
    const primaryAction = document.querySelector(".file-button");
    const rect = (node) => node?.getBoundingClientRect();
    return {
      intro: intro?.textContent || "",
      glance: glance?.textContent || "",
      context: context?.textContent || "",
      build: build?.textContent || "",
      buildVisible: Boolean(build?.getClientRects().length),
      authorSummaryPresent: Boolean(authorDetails?.querySelector("summary")),
      citationVisible: Boolean(document.querySelector("#copyCitationBtn")?.getClientRects().length),
      doiVisible: Boolean(document.querySelector(".citation-doi")?.getClientRects().length),
      affiliationVisible: Boolean(document.querySelector(".author-affiliation")?.getClientRects().length),
      linkedinVisible: Boolean(document.querySelector(".author-social-link")?.getClientRects().length),
      evidenceSummaryVisible: Boolean(evidenceDetails?.querySelector("summary")?.getClientRects().length),
      evidenceLegendVisible: Boolean(evidenceDetails?.querySelector(".evidence-class-strip")?.getClientRects().length),
      guideOpen: Boolean(guide?.open),
      guideSummaryVisible: Boolean(guide?.querySelector(":scope > summary")?.getClientRects().length),
      guideBodyVisible: Number(guide?.querySelector(".pre-audit-reference-body")?.getBoundingClientRect().height || 0) > 1,
      introTop: rect(intro)?.top,
      medicalTop: rect(medical)?.top,
      controlsTop: rect(controls)?.top,
      guideTop: rect(guide)?.top,
      primaryActionTop: rect(primaryAction)?.top,
      medicalSummary: medical?.querySelector(".medical-context-summary")?.textContent || "",
      medicalDetailVisible: Boolean(medicalDetail?.querySelector("div")?.getClientRects().length),
      introRight: rect(intro)?.right,
      controlsRight: rect(controls)?.right,
      guideRight: rect(guide)?.right,
      metaHeight: rect(meta)?.height,
      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  });
  if (!firstVisit.intro.includes("Inspect the artifact that will actually run")
    || !firstVisit.intro.includes("what remains unassessed") || firstVisit.introTop >= 844
    || !firstVisit.glance.includes("65 ops / 173 tensors / 300,775,552 MACs")
    || !firstVisit.glance.includes("0 / 0") || !firstVisit.glance.includes("Predicted break ops1")
    || !firstVisit.context.includes("Why this matters for medical AI")
    || !firstVisit.context.includes("zero-weight, shape/op/dtype/quantization-equivalent synthetic reconstruction")
    || !firstVisit.build.includes(`Application ${VERSION.displayVersion} · source`)
    || !firstVisit.build.includes("· content")
    || !firstVisit.buildVisible || firstVisit.authorSummaryPresent || !firstVisit.citationVisible || !firstVisit.doiVisible
    || !firstVisit.affiliationVisible || !firstVisit.linkedinVisible
    || !firstVisit.evidenceSummaryVisible
    || firstVisit.evidenceLegendVisible || firstVisit.guideOpen || !firstVisit.guideSummaryVisible || firstVisit.guideBodyVisible
    || firstVisit.controlsTop >= firstVisit.guideTop || firstVisit.primaryActionTop >= 844
    || !firstVisit.medicalSummary.includes("Model evidence alone") || firstVisit.medicalDetailVisible
    || firstVisit.metaHeight > 72 || firstVisit.overflow > 1
    || Math.max(firstVisit.introRight, firstVisit.controlsRight, firstVisit.guideRight) > 391) {
    throw new Error(`Mobile first-visit evidence contract failed: ${JSON.stringify(firstVisit)}`);
  }
  await page.screenshot({ path: firstVisitMobileScreenshot });
  await page.locator("#preAuditReference > summary").click();
  const expandedGuide = await page.evaluate(() => ({
    bodyVisible: Number(document.querySelector("#preAuditReference .pre-audit-reference-body")?.getBoundingClientRect().height || 0) > 1,
    baselineVisible: Number(document.querySelector("#sampleEvidenceGlance")?.getBoundingClientRect().height || 0) > 1,
    medicalVisible: Number(document.querySelector(".first-visit-context")?.getBoundingClientRect().height || 0) > 1,
    librarySummaryVisible: Boolean(document.querySelector("#sampleLibrary > summary")?.getClientRects().length),
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!expandedGuide.bodyVisible || !expandedGuide.baselineVisible || !expandedGuide.medicalVisible
    || !expandedGuide.librarySummaryVisible || expandedGuide.overflow > 1) {
    throw new Error(`Mobile evidence guide did not preserve its complete content: ${JSON.stringify(expandedGuide)}`);
  }
  await page.locator("#preAuditReference > summary").click();
  const visibleProvenance = await page.evaluate(() => ({
    buildVisible: Boolean(document.querySelector("#applicationBuild")?.getClientRects().length),
    citationVisible: Boolean(document.querySelector("#copyCitationBtn")?.getClientRects().length),
    doiVisible: Boolean(document.querySelector(".citation-doi")?.getClientRects().length),
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!visibleProvenance.buildVisible || !visibleProvenance.citationVisible || !visibleProvenance.doiVisible
    || visibleProvenance.overflow > 1) {
    throw new Error(`Mobile provenance must remain visible and viewport-safe: ${JSON.stringify(visibleProvenance)}`);
  }
  await page.setViewportSize({ width: 320, height: 568 });
  const narrowFirstVisit = await page.evaluate(() => {
    const controls = document.querySelector(".upload-controls")?.getBoundingClientRect();
    const primary = document.querySelector(".file-button")?.getBoundingClientRect();
    return {
      controlsTop: controls?.top || 0,
      primaryWidth: primary?.width || 0,
      primaryHeight: primary?.height || 0,
      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  });
  if (narrowFirstVisit.controlsTop <= 0 || narrowFirstVisit.controlsTop >= 680
    || narrowFirstVisit.primaryWidth > 288 || narrowFirstVisit.primaryHeight < 44 || narrowFirstVisit.overflow > 1) {
    throw new Error(`320px first-visit task path is not usable: ${JSON.stringify(narrowFirstVisit)}`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await runAudit(page, TFLITE, "mobilenet_v2_1.0_224_quant.tflite");

  const fullDepth = await page.evaluate(() => ({
    mode: document.documentElement.dataset.analysisDepth,
    graphVisible: Boolean(document.querySelector('[data-workflow-step="graph"]')?.getClientRects().length),
    quantLabsVisible: Boolean(document.querySelector('[data-audit-tab="quant-labs"]')?.getClientRects().length),
    stageOptionDisabled: document.querySelector('#mobileAuditView option[value="stage"]')?.disabled,
    releaseControlPresent: Boolean(document.querySelector('[data-analysis-depth-mode="release"]')),
    fullIndicatorVisible: Boolean(document.querySelector('[data-analysis-depth-mode="deep"]')?.getClientRects().length),
    fullIndicatorText: document.querySelector('[data-analysis-depth-mode="deep"]')?.textContent?.trim(),
    legacyPreference: sessionStorage.getItem("deepbom.analysis-depth.v1"),
  }));
  if (fullDepth.mode !== "deep" || !fullDepth.graphVisible || !fullDepth.quantLabsVisible
    || fullDepth.stageOptionDisabled || fullDepth.releaseControlPresent || !fullDepth.fullIndicatorVisible
    || fullDepth.fullIndicatorText !== "Full analysis" || fullDepth.legacyPreference !== null) {
    throw new Error(`Full analysis surface is not the only visible depth: ${JSON.stringify(fullDepth)}`);
  }

  await page.locator('[data-workflow-step="audit"]').click();
  await page.locator('[data-audit-tab="quant"]').click();
  const quantSummary = await page.locator("#perfVisuals").evaluate((root) => ({
    tiles: root.querySelectorAll("#quantHeatmap .quant-tile").length,
    warnTiles: root.querySelectorAll("#quantHeatmap .quant-tile.signal-warn").length,
    count: root.querySelector("#quantHeatmapCount")?.textContent || "",
    states: root.querySelector("#quantStateBreakdown")?.textContent || "",
    riskCount: root.querySelector("#quantRiskCount")?.textContent || "",
    risk: root.querySelector("#quantRiskTable")?.textContent || "",
    scales: root.querySelector("#quantScaleScatter")?.textContent || "",
    holeCount: root.querySelector("#quantHoleCount")?.textContent || "",
    holes: root.querySelector("#quantHoleList")?.textContent || "",
    overflow: Math.max(0, root.scrollWidth - root.clientWidth),
    scalePanelRatio: root.querySelector("#quantScaleDistributionPanel")?.getBoundingClientRect().width
      / root.querySelector(".perf-visual-grid")?.getBoundingClientRect().width,
    scaleOverflowY: getComputedStyle(root.querySelector("#quantScaleScatter")).overflowY,
  }));
  if (quantSummary.tiles !== 65 || quantSummary.warnTiles !== 53 || quantSummary.count !== "65 graph ops"
    || !quantSummary.states.includes("53/53 ops; 300,775,552/300,775,552 MACs (100%)")
    || !quantSummary.states.includes("64/65 graph ops (98.5%)") || !quantSummary.states.includes("1/65 graph ops (1.5%)")
    || quantSummary.riskCount !== "0 risk · 4 review categories"
    || !quantSummary.risk.includes("53 kernel op(s), non-zero weight zero-point")
    || !quantSummary.risk.includes("17 op(s), one weight scale across channels")
    || !quantSummary.risk.includes("11 exact-zero / 11 near-zero") || !quantSummary.risk.includes("3/53 below heuristic 25%")
    || !quantSummary.scales.includes("all 53 decoded 8-bit constants")
    || quantSummary.holeCount !== "0 mid-graph 8-bit/FP32 boundaries" || /path is clean/i.test(quantSummary.holes)
    || quantSummary.scalePanelRatio < 0.95 || quantSummary.scaleOverflowY === "auto"
    || quantSummary.overflow > 1) throw new Error(`Quant summary scope reconciliation failed: ${JSON.stringify(quantSummary)}`);

  await page.setViewportSize({ width: 390, height: 844 });
  const firstQuantTile = page.locator("#quantHeatmap .quant-tile").first();
  await firstQuantTile.scrollIntoViewIfNeeded();
  await page.waitForTimeout(100);
  const firstQuantTileBox = await firstQuantTile.boundingBox();
  await page.mouse.move(firstQuantTileBox.x + firstQuantTileBox.width / 2, firstQuantTileBox.y + firstQuantTileBox.height / 2);
  const mobileTip = await page.locator(".flame-tooltip").evaluate((tip) => {
    const rect = tip.getBoundingClientRect();
    return { hidden: tip.hidden, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
  });
  if (mobileTip.hidden || mobileTip.left < 7 || mobileTip.right > 383 || mobileTip.top < 7 || mobileTip.bottom > 837) {
    throw new Error(`Mobile quant tooltip escaped the viewport: ${JSON.stringify(mobileTip)}`);
  }
  await firstQuantTile.click();
  await page.locator("#quantHeatmap").locator("xpath=ancestor::*[contains(@class,'perf-panel')][1]").locator(".panel-detail-card").waitFor();
  await page.waitForTimeout(350);
  const mobileQuantDetail = await page.evaluate(() => ({
    tipHidden: document.querySelector(".flame-tooltip")?.hidden,
    detailVisible: Boolean(document.querySelector("#quantHeatmap")?.closest(".perf-panel")?.querySelector(".panel-detail-card")),
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!mobileQuantDetail.tipHidden || !mobileQuantDetail.detailVisible || mobileQuantDetail.overflow > 1) {
    throw new Error(`Mobile quant detail did not replace hover evidence cleanly: ${JSON.stringify(mobileQuantDetail)}`);
  }
  await page.locator("#mobileAuditView").selectOption("stage");
  await page.waitForTimeout(50);
  const mobileAuditTab = await page.evaluate(() => ({
    value: document.querySelector("#mobileAuditView")?.value || "",
    selectorVisible: Boolean(document.querySelector(".mobile-audit-view")?.getClientRects().length),
    railVisible: Boolean(document.querySelector(".audit-tabs")?.getClientRects().length),
    stageVisible: !document.querySelector("#diagramSection")?.hidden,
  }));
  if (mobileAuditTab.value !== "stage" || !mobileAuditTab.selectorVisible || mobileAuditTab.railVisible || !mobileAuditTab.stageVisible) {
    throw new Error(`Mobile audit selector did not activate the requested view: ${JSON.stringify(mobileAuditTab)}`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });

  await page.locator('[data-audit-tab="stage"]').click();
  const stageGeometry = await page.locator("#perfVisuals").evaluate((root) => ({
    panelRatio: root.querySelector("#performanceMacPanel")?.getBoundingClientRect().width
      / root.querySelector(".perf-visual-grid")?.getBoundingClientRect().width,
    overflow: Math.max(0, root.scrollWidth - root.clientWidth),
  }));
  if (stageGeometry.panelRatio < 0.95 || stageGeometry.overflow > 1) {
    throw new Error(`Stage MAC panel does not consume its available row: ${JSON.stringify(stageGeometry)}`);
  }

  await page.locator('[data-workflow-step="findings"]').click();
  const findingsGeometry = await page.locator("#findingsPanel").evaluate((panel) => {
    const rows = [...panel.querySelectorAll(".finding-evidence tr")];
    return {
      panelRatio: panel.getBoundingClientRect().width / panel.closest(".layout").getBoundingClientRect().width,
      crossingRows: rows.filter((row) => {
        const source = row.querySelector(".ev-source")?.getBoundingClientRect();
        const value = row.querySelector(".ev-text")?.getBoundingClientRect();
        return source && value && source.right > value.left + 1;
      }).length,
      overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    };
  });
  if (findingsGeometry.panelRatio < 0.9 || findingsGeometry.crossingRows || findingsGeometry.overflow > 1) {
    throw new Error(`Findings evidence geometry is not collision-safe: ${JSON.stringify(findingsGeometry)}`);
  }

  await page.setViewportSize({ width: 390, height: 844 });
  const compactFindings = await page.evaluate(async () => {
    const { renderFindings } = await import("./lib/findings-viewer.js");
    const root = document.createElement("div");
    document.body.append(root);
    const findings = Array.from({ length: 3 }, (_, index) => ({
      id: `mobile-${index}`,
      severity: index === 0 ? "high" : "medium",
      category: "quant",
      title: `Finding ${index + 1}`,
      evidence: [{ source: "Observed", text: `complete mobile evidence ${index + 1}` }],
      impact: `Impact ${index + 1}`,
      actions: [`Action ${index + 1}`],
      confidence: "static",
    }));
    renderFindings(root, { findings });
    const cards = [...root.querySelectorAll("details.finding-card")];
    const before = cards.filter((card) => card.open).length;
    root.querySelector(".findings-mobile-actions button")?.click();
    const expanded = cards.filter((card) => card.open).length;
    root.querySelector(".findings-mobile-actions button:last-child")?.click();
    const collapsed = cards.filter((card) => card.open).length;
    const result = {
      cards: cards.length,
      before,
      expanded,
      collapsed,
      evidencePreserved: root.textContent.includes("complete mobile evidence 3"),
      controlsVisible: getComputedStyle(root.querySelector(".findings-mobile-actions")).display !== "none",
      evidenceColumns: getComputedStyle(root.querySelector(".finding-evidence tr")).gridTemplateColumns.split(" ").length,
    };
    root.remove();
    return result;
  });
  if (compactFindings.cards !== 3 || compactFindings.before !== 1 || compactFindings.expanded !== 3
    || compactFindings.collapsed !== 0 || !compactFindings.evidencePreserved || !compactFindings.controlsVisible
    || compactFindings.evidenceColumns !== 1) {
    throw new Error(`Mobile findings disclosure contract failed: ${JSON.stringify(compactFindings)}`);
  }
  await page.locator('[data-workflow-step="redesign"]').click();
  await page.waitForTimeout(50);
  const mobileWorkflowTab = await page.locator(".analysis-tool-rail").evaluate((rail) => {
    const active = rail.querySelector(".analysis-tool.active");
    const railRect = rail.getBoundingClientRect();
    const activeRect = active?.getBoundingClientRect();
    return {
      scrollLeft: rail.scrollLeft,
      visible: Boolean(activeRect && activeRect.left >= railRect.left - 1 && activeRect.right <= railRect.right + 1),
    };
  });
  if (!mobileWorkflowTab.visible) {
    throw new Error(`Mobile analysis palette did not reveal the active tool: ${JSON.stringify(mobileWorkflowTab)}`);
  }
  await page.locator('[data-workflow-step="output"]').click();
  const mobileReportTarget = await page.locator('[data-module-panel="engineering_report"]').evaluate((panel) => {
    const label = panel.querySelector(".report-target-bar label")?.getBoundingClientRect();
    const select = panel.querySelector("#reportTargetSelect")?.getBoundingClientRect();
    return {
      labelHeight: label?.height || 0,
      selectHeight: select?.height || 0,
      overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    };
  });
  if (mobileReportTarget.labelHeight > 90 || mobileReportTarget.selectHeight < 36
    || mobileReportTarget.selectHeight > 52 || mobileReportTarget.overflow > 1) {
    throw new Error(`Mobile report target selector is not compact: ${JSON.stringify(mobileReportTarget)}`);
  }

  for (const [physicalWidth, zoom] of [[1280, 1], [1280, 1.25], [1280, 1.5], [1440, 1], [1440, 1.25], [1440, 1.5], [1920, 1], [1920, 1.25], [1920, 1.5], [2048, 1], [2048, 1.25], [2048, 1.5], [2560, 1], [2560, 1.25], [2560, 1.5]]) {
    const effectiveWidth = Math.floor(physicalWidth / zoom);
    await page.setViewportSize({ width: effectiveWidth, height: 1000 });
    const geometry = await page.evaluate(() => {
      const shell = document.querySelector(".layout")?.getBoundingClientRect();
      return {
        overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
        shellRatio: shell ? shell.width / innerWidth : 0,
      };
    });
    if (geometry.overflow > 1 || (zoom === 1 && physicalWidth <= 2048 && physicalWidth >= 1920 && geometry.shellRatio < 0.7)) {
      throw new Error(`Workspace geometry failed at ${physicalWidth}px/${zoom}x: ${JSON.stringify(geometry)}`);
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });

  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator("#nodeViewPanel .nv-node").first().waitFor();
  const defaultGraph = await page.evaluate(() => ({
    activeTab: document.querySelector("[data-explorer-tab].active")?.dataset.explorerTab || "",
    graphVisible: !document.getElementById("nodeViewPanel")?.hidden,
    resourceHidden: Boolean(document.getElementById("resourceMapPanel")?.hidden),
    blocksHidden: Boolean(document.getElementById("blocksExplorerPanel")?.hidden),
  }));
  if (defaultGraph.activeTab !== "node" || !defaultGraph.graphVisible || !defaultGraph.resourceHidden || !defaultGraph.blocksHidden) {
    throw new Error(`Graph and Tensor Explorer is not the default Explorer view: ${JSON.stringify(defaultGraph)}`);
  }
  const acceleratorSelector = await page.locator("#acceleratorSwitcherBar").evaluate((bar) => ({
    hidden: bar.hidden,
    labels: [...bar.querySelectorAll("button")].map((button) => button.textContent.trim()),
    boundary: bar.querySelector(".accelerator-switcher-boundary")?.textContent || "",
    overflow: Math.max(0, bar.scrollWidth - bar.clientWidth),
  }));
  if (acceleratorSelector.hidden || !acceleratorSelector.labels.some((label) => /GPU|NNAPI/i.test(label))
    || !acceleratorSelector.boundary.includes("does not change the selected CPU roofline")
    || acceleratorSelector.overflow > 1) {
    throw new Error(`Accelerator eligibility selector is absent or semantically mixed with CPU cost profiles: ${JSON.stringify(acceleratorSelector)}`);
  }

  await page.locator('[data-explorer-tab="resource"]').click();
  await page.locator("#resourceMapPanel .evidence-treemap-tile").first().waitFor();
  const resourceMap = await page.locator("#resourceMapPanel").evaluate((panel) => ({
    title: panel.querySelector("h3")?.textContent || "",
    summary: panel.querySelector(".evidence-map-summary")?.textContent || "",
    tileCount: panel.querySelectorAll(".evidence-treemap-tile").length,
    groupCount: panel.querySelectorAll(".evidence-treemap-group").length,
    metric: panel.querySelector('[data-treemap-control="metric"]')?.value || "",
    mobileListDisplay: getComputedStyle(panel.querySelector(".evidence-treemap-mobile-list")).display,
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (resourceMap.title !== "Explorer Resource Map" || resourceMap.tileCount !== 53 || resourceMap.groupCount !== 6
    || resourceMap.metric !== "macs" || !resourceMap.summary.includes("300,775,552 MACs")
    || !resourceMap.summary.includes("53 / 65") || !resourceMap.summary.includes("Groups mapped6 / 7") || !resourceMap.summary.includes("Exact zero12")
    || !resourceMap.summary.includes("Conservationexact") || resourceMap.mobileListDisplay !== "none" || resourceMap.overflow > 1) {
    throw new Error(`Explorer Resource Map is incomplete or non-conserving: ${JSON.stringify(resourceMap)}`);
  }
  await page.locator("#resourceMapPanel .evidence-treemap-tile").first().hover();
  await assertReadableState(page, "#resourceMapPanel .evidence-treemap-tile", "strong", "light resource-map hover");
  await page.locator("#resourceMapPanel").screenshot({ path: resourceMapScreenshot });
  await page.locator('#resourceMapPanel [data-treemap-control="metric"]').selectOption("traffic");
  await page.waitForFunction(() => document.querySelector('#resourceMapPanel [data-treemap-control="metric"]')?.value === "traffic");
  const trafficMap = await page.locator("#resourceMapPanel").evaluate((panel) => ({
    tiles: panel.querySelectorAll(".evidence-treemap-tile").length,
    summary: panel.querySelector(".evidence-map-summary")?.textContent || "",
  }));
  if (trafficMap.tiles !== 65 || !trafficMap.summary.includes("65 / 65") || !trafficMap.summary.includes("Conservationexact")) {
    throw new Error(`Explorer Resource Map traffic denominator is incomplete: ${JSON.stringify(trafficMap)}`);
  }
  await page.locator("#resourceMapPanel .evidence-treemap-tile").first().click();
  if (!(await page.locator('[data-explorer-tab="node"]').evaluate((button) => button.classList.contains("active")))) {
    throw new Error("Explorer Resource Map tile did not open its evidence-linked Node inspector.");
  }
  const resourceWhy = await page.locator("#evidenceWhyDrawer").evaluate((drawer) => ({
    visible: !drawer.hidden,
    title: drawer.querySelector("h2")?.textContent || "",
    content: drawer.textContent || "",
  }));
  if (!resourceWhy.visible || !resourceWhy.title.includes("Logical traffic") || !resourceWhy.content.includes("Formula / denominator")) {
    throw new Error(`Resource Map did not synchronize the Why drawer: ${JSON.stringify(resourceWhy)}`);
  }
  await page.locator("#closeEvidenceWhy").click();

  await page.locator('[data-explorer-tab="blocks"]').click();
  await page.locator("#blocksExplorerPanel .xr-block-layout").waitFor();
  const blocks = await page.locator("#blocksExplorerPanel").evaluate((panel) => ({
    stages: panel.querySelectorAll(".xr-stage").length,
    blocks: panel.querySelectorAll(".xr-block-row").length,
    metrics: panel.querySelector(".xr-head-metrics")?.textContent || "",
    detail: panel.querySelector(".xr-block-detail")?.textContent || "",
    text: panel.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (blocks.stages !== 7
    || blocks.blocks !== 27
    || !blocks.metrics.includes("Semantic17")
    || !blocks.metrics.includes("Named7")
    || !blocks.text.includes("graph_pattern_matched")
    || !blocks.detail.includes("Logical traffic")
    || blocks.overflow > 1) {
    throw new Error(`Block Inventory viewer is incomplete: ${JSON.stringify(blocks)}`);
  }
  const firstStage = page.locator("#blocksExplorerPanel .xr-stage").first();
  await firstStage.locator(".xr-stage-head").click();
  if (!(await firstStage.locator(".xr-block-list").evaluate((list) => list.hidden))) {
    throw new Error("Block stage collapse did not hide the stage block list.");
  }
  await firstStage.locator(".xr-stage-head").click();
  if (await firstStage.locator(".xr-block-list").evaluate((list) => list.hidden)) {
    throw new Error("Block stage expansion did not restore the stage block list.");
  }
  await page.locator("#blocksExplorerPanel button", { hasText: "All blocks" }).click();
  const warningBlockCount = await page.locator("#blocksExplorerPanel .xr-block-row").count();
  if (warningBlockCount < 1 || warningBlockCount > blocks.blocks) {
    throw new Error(`Warning-only block filter returned an invalid count: ${warningBlockCount}/${blocks.blocks}`);
  }
  await page.locator("#blocksExplorerPanel button", { hasText: "Warnings only" }).click();
  if (await page.locator("#blocksExplorerPanel .xr-block-row").count() !== blocks.blocks) {
    throw new Error("Clearing the block warning filter did not restore all blocks.");
  }

  const firstBlockRow = page.locator("#blocksExplorerPanel .xr-block-row").first();
  await firstBlockRow.hover();
  await assertReadableState(page, "#blocksExplorerPanel .xr-block-row", "strong", "light block hover");
  await page.locator("#blocksExplorerPanel").screenshot({ path: blocksScreenshot });

  await page.locator('[data-explorer-tab="ops"]').click();
  await page.locator("#graphOpBody tr").first().waitFor();
  const operatorRows = page.locator("#graphOpBody tr");
  if (await operatorRows.count() !== 65) {
    throw new Error(`Operator Table row count is not conserved: ${await operatorRows.count()}/65`);
  }
  await page.locator("#graphSearch").fill("AVERAGE_POOL_2D");
  await page.waitForFunction(() => document.querySelectorAll("#graphOpBody tr").length === 1);
  const searchedOps = await page.locator("#graphOpsView").evaluate((panel) => ({
    rows: [...document.querySelectorAll("#graphOpBody tr")].map((row) => ({
      index: row.dataset.opIndex,
      name: row.children[1]?.textContent?.trim() || "",
    })),
    count: panel.querySelector("#opFilterCount")?.textContent || "",
  }));
  if (searchedOps.rows.length !== 1
    || searchedOps.rows[0]?.name !== "AVERAGE_POOL_2D"
    || searchedOps.count !== "1/65 shown") {
    throw new Error(`Operator search leaked a non-match or reported the wrong denominator: ${JSON.stringify(searchedOps)}`);
  }
  await page.locator("#graphSearch").fill("");
  await page.waitForFunction(() => document.querySelectorAll("#graphOpBody tr").length === 65);
  await page.locator('[data-filter-group="bound"][data-filter-value="memory-bound"]').click();
  const memoryFilteredOps = await page.locator("#graphOpsView").evaluate((panel) => ({
    rows: [...document.querySelectorAll("#graphOpBody tr")].map((row) => row.children[6]?.textContent?.trim() || ""),
    count: panel.querySelector("#opFilterCount")?.textContent || "",
  }));
  if (!memoryFilteredOps.rows.length
    || memoryFilteredOps.rows.some((value) => value !== "memory-bound")
    || memoryFilteredOps.count !== `${memoryFilteredOps.rows.length}/65 shown`) {
    throw new Error(`Operator bound filter is not exact: ${JSON.stringify(memoryFilteredOps)}`);
  }
  await page.locator('[data-filter-group="bound"][data-filter-value=""]').click();
  await page.locator('#graphOpHead th[data-sort-key="macs"]').click();
  const sortedMacs = await page.locator("#graphOpBody tr").evaluateAll((rows) => rows.map((row) => Number((row.children[4]?.textContent || "0").replaceAll(",", ""))));
  if (sortedMacs.some((value, index) => index > 0 && sortedMacs[index - 1] < value)) {
    throw new Error(`Operator MAC sort is not descending: ${JSON.stringify(sortedMacs.slice(0, 12))}`);
  }
  await page.locator('#graphOpHead th[data-sort-key="index"]').click();
  await page.locator("#graphDepth").selectOption("1");
  await page.waitForFunction(() => /1-hop$/.test(document.querySelector("#graphMapStatus")?.textContent || ""));
  const oneHopNodes = await page.locator("#graphMapSvg .graph-node").count();
  await page.locator("#graphDepth").selectOption("all");
  await page.waitForFunction(() => /full graph$/.test(document.querySelector("#graphMapStatus")?.textContent || ""));
  const fullGraphNodes = await page.locator("#graphMapSvg .graph-node").count();
  if (oneHopNodes < 2 || oneHopNodes >= 65 || fullGraphNodes !== 65) {
    throw new Error(`Operator graph depth control is not reflected in the rendered node set: ${JSON.stringify({ oneHopNodes, fullGraphNodes })}`);
  }
  const graphViewBoxBefore = (await page.locator("#graphMapSvg").getAttribute("viewBox")).split(" ").map(Number);
  await page.locator("#graphZoomOut").click();
  const graphViewBoxOut = (await page.locator("#graphMapSvg").getAttribute("viewBox")).split(" ").map(Number);
  await page.locator("#graphZoomIn").click();
  const graphViewBoxIn = (await page.locator("#graphMapSvg").getAttribute("viewBox")).split(" ").map(Number);
  if (!(graphViewBoxOut[2] > graphViewBoxBefore[2] && graphViewBoxOut[3] > graphViewBoxBefore[3])
    || !(graphViewBoxIn[2] < graphViewBoxOut[2] && graphViewBoxIn[3] < graphViewBoxOut[3])) {
    throw new Error(`Operator graph zoom controls did not update the SVG viewBox: ${JSON.stringify({ graphViewBoxBefore, graphViewBoxOut, graphViewBoxIn })}`);
  }
  await page.locator("#graphDepth").selectOption("2");
  await operatorRows.first().hover();
  await assertReadableState(page, "#graphOpBody tr", "td:nth-child(2)", "light operator hover");
  await operatorRows.nth(1).focus();
  await operatorRows.nth(1).press("Enter");
  if (await operatorRows.nth(1).getAttribute("aria-selected") !== "true") {
    throw new Error("Operator Table keyboard selection did not expose aria-selected state.");
  }
  await page.locator('[data-graph-mode="stage"]').click();
  await page.waitForFunction(() => document.querySelector("#graphMapSvg .graph-node .subtext")?.textContent?.startsWith("Stage "));
  const stageMode = await page.locator("#graphMapSvg").evaluate((svg) => ({
    labels: [...svg.querySelectorAll(".graph-node .subtext")].map((node) => node.textContent || ""),
    tinted: svg.querySelectorAll(".graph-node-stage-bg").length,
  }));
  if (!stageMode.labels.length
    || stageMode.labels.some((label) => /^Stage \?\b/.test(label) || /not emitted/i.test(label))
    || stageMode.tinted !== stageMode.labels.length / 2) {
    throw new Error(`Stage-mode operator binding is incomplete: ${JSON.stringify(stageMode)}`);
  }
  await assertExplorerVisualizationTheme(page, "light");
  await page.locator("#graphExplorer").screenshot({ path: operatorScreenshot });

  await page.locator('[data-explorer-tab="tensors"]').click();
  await page.locator("#tensorBody tr").first().waitFor();
  const tensorState = await page.locator("#tensorExplorerPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll("#tensorBody tr").length,
    stats: panel.querySelector("#tensorStatsBar")?.textContent || "",
    quantCells: [...panel.querySelectorAll("#tensorBody .tensor-quant")].map((cell) => cell.textContent || ""),
  }));
  if (tensorState.rows !== 173
    || !tensorState.stats.includes("172 quantized")
    || tensorState.quantCells.filter((value) => value !== "none").length !== 172
    || !tensorState.quantCells.some((value) => value.startsWith("per-tensor"))) {
    throw new Error(`Tensor inventory UI is not conserved: ${JSON.stringify(tensorState)}`);
  }
  await page.locator("#tensorSearch").fill("t171");
  await page.waitForFunction(() => document.querySelectorAll("#tensorBody tr").length === 1);
  const exactTensorSearch = await page.locator("#tensorBody tr").first().evaluate((row) => ({
    id: row.children[0]?.textContent?.trim() || "",
    rows: row.parentElement?.children.length || 0,
  }));
  if (exactTensorSearch.id !== "T171" || exactTensorSearch.rows !== 1) {
    throw new Error(`Tensor exact-index search did not isolate T171: ${JSON.stringify(exactTensorSearch)}`);
  }
  await page.locator("#tensorSearch").fill("");
  await page.waitForFunction(() => document.querySelectorAll("#tensorBody tr").length === 173);
  const kernelStat = await page.locator("#tensorStatsBar .tstat", { hasText: "kernels" }).first().evaluate((node) => Number(node.querySelector("b")?.textContent || 0));
  await page.locator('[data-tfilter="kernel"]').click();
  const kernelTensorRows = await page.locator("#tensorBody tr").evaluateAll((rows) => rows.map((row) => row.children[4]?.textContent?.trim() || ""));
  if (kernelTensorRows.length !== kernelStat || kernelTensorRows.some((role) => !role.startsWith("kernel"))) {
    throw new Error(`Tensor kernel filter disagrees with the unfiltered role ledger: ${JSON.stringify({ kernelStat, rows: kernelTensorRows.length, roles: [...new Set(kernelTensorRows)] })}`);
  }
  await page.locator('[data-tfilter="fanout"]').click();
  const fanoutValues = await page.locator("#tensorBody tr").evaluateAll((rows) => rows.map((row) => Number(row.children[6]?.textContent || 0)));
  if (!fanoutValues.length || fanoutValues.some((value) => value <= 1)) {
    throw new Error(`Tensor fan-out filter admitted an invalid row: ${JSON.stringify(fanoutValues)}`);
  }
  await page.locator('[data-tfilter="quant"]').click();
  const quantizedTensorRows = await page.locator("#tensorBody tr").evaluateAll((rows) => rows.map((row) => row.children[8]?.textContent?.trim() || ""));
  if (quantizedTensorRows.length !== 172 || quantizedTensorRows.some((value) => value === "none")) {
    throw new Error(`Tensor quantized filter disagrees with the 172-tensor inventory: ${JSON.stringify({ rows: quantizedTensorRows.length })}`);
  }
  await page.locator('[data-tfilter="all"]').click();
  await page.waitForFunction(() => document.querySelectorAll("#tensorBody tr").length === 173);
  const tensorRow = page.locator("#tensorBody tr").first();
  await tensorRow.hover();
  await assertReadableState(page, "#tensorBody tr", "td:nth-child(2)", "light tensor hover");
  await page.locator("#tensorExplorerPanel").screenshot({ path: tensorsScreenshot });
  const consumerChip = page.locator("#tensorBody .consumer-chip").first();
  const consumerIndex = Number((await consumerChip.textContent()).replace("#", ""));
  await consumerChip.click();
  await page.waitForFunction((index) => document.querySelector(`#graphOpBody tr[data-op-index="${index}"]`)?.getAttribute("aria-selected") === "true", consumerIndex);
  if (!(await page.locator('[data-explorer-tab="ops"]').evaluate((button) => button.classList.contains("active")))) {
    throw new Error("Tensor consumer navigation did not open the Operator Table.");
  }
  await page.locator('[data-explorer-tab="tensors"]').click();
  const tensorWithProducer = page.locator('#tensorBody tr[aria-label*="open producer op"]').first();
  await tensorWithProducer.focus();
  await tensorWithProducer.press("Enter");
  if (!(await page.locator('[data-explorer-tab="ops"]').evaluate((button) => button.classList.contains("active")))) {
    throw new Error("Tensor keyboard navigation did not open its producer operator.");
  }

  await page.locator('[data-explorer-tab="node"]').click();
  await page.locator("#nodeViewPanel .nv-node").first().waitFor();
  const nodeView = await page.locator("#nodeViewPanel").evaluate((panel) => ({
    nodes: panel.querySelectorAll(".nv-node").length,
    edges: panel.querySelectorAll(".nv-edge, .nv-edge-risk").length,
    edgeLabels: panel.querySelectorAll(".nv-edge-label").length,
    inputs: panel.querySelectorAll(".nv-interface-input").length,
    outputs: panel.querySelectorAll(".nv-interface-output").length,
    downwardEdges: [...panel.querySelectorAll(".nv-edge, .nv-edge-risk")].filter((edge) => {
      const from = panel.querySelector(`.nv-node[data-op-index="${edge.dataset.fromOp}"]`)?.transform.baseVal.consolidate()?.matrix;
      const to = panel.querySelector(`.nv-node[data-op-index="${edge.dataset.toOp}"]`)?.transform.baseVal.consolidate()?.matrix;
      return from && to && to.f > from.f;
    }).length,
    viewBox: panel.querySelector(".nv-graph")?.getAttribute("viewBox") || "",
    minimapNodes: panel.querySelectorAll(".nv-minimap-nodes rect").length,
    minimapWindow: Boolean(panel.querySelector(".nv-minimap-window")),
    controls: [...panel.querySelectorAll(".nv-view-controls button")].map((button) => button.textContent.trim()),
    evidenceLedger: Boolean(panel.querySelector(".nv-evidence-ledger")),
    firstInputPorts: panel.querySelectorAll('.nv-node[data-op-index="0"] .nv-port-input').length,
    firstOutputPorts: panel.querySelectorAll('.nv-node[data-op-index="0"] .nv-port-output').length,
    text: panel.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (nodeView.nodes !== 65
    || nodeView.edges < 50
    || nodeView.edgeLabels !== nodeView.edges
    || nodeView.downwardEdges !== nodeView.edges
    || nodeView.inputs !== 1
    || nodeView.outputs !== 1
    || nodeView.minimapNodes !== nodeView.nodes
    || !nodeView.minimapWindow
    || !["Start", "Overview", "Selected", "Inspector", "Expand", "-", "+"].every((label) => nodeView.controls.includes(label))
    || !nodeView.evidenceLedger
    || nodeView.firstInputPorts !== 1
    || nodeView.firstOutputPorts !== 1
    || !nodeView.text.includes("Operators, tensors, and model flow")
    || !nodeView.text.includes("model input T171")
    || nodeView.overflow > 1) {
    throw new Error(`Node View graph contract failed: ${JSON.stringify(nodeView)}`);
  }
  const graph = page.locator("#nodeViewPanel .nv-graph");
  const initialViewBox = (await graph.getAttribute("viewBox")).split(" ").map(Number);
  const graphBox = await graph.boundingBox();
  await graph.dispatchEvent("wheel", {
    deltaY: 240,
    deltaX: 0,
    clientX: graphBox.x + graphBox.width * 0.35,
    clientY: graphBox.y + graphBox.height * 0.4,
  });
  const zoomedOutViewBox = (await graph.getAttribute("viewBox")).split(" ").map(Number);
  await graph.dispatchEvent("wheel", {
    deltaY: -180,
    deltaX: 0,
    clientX: graphBox.x + graphBox.width * 0.35,
    clientY: graphBox.y + graphBox.height * 0.4,
  });
  const zoomedInViewBox = (await graph.getAttribute("viewBox")).split(" ").map(Number);
  const zoomLabel = await page.locator("#nodeViewPanel .nv-zoom-level").textContent();
  if (!(zoomedOutViewBox[2] > initialViewBox[2] && zoomedOutViewBox[3] > initialViewBox[3])
    || !(zoomedInViewBox[2] < zoomedOutViewBox[2] && zoomedInViewBox[3] < zoomedOutViewBox[3])
    || !/^\d+%$/.test(zoomLabel.trim())) {
    throw new Error(`Top-down graph wheel contract failed: ${JSON.stringify({ initialViewBox, zoomedOutViewBox, zoomedInViewBox, zoomLabel })}`);
  }
  await page.locator("#nodeViewPanel").getByRole("button", { name: "Zoom out", exact: true }).click();
  const buttonZoomedOutViewBox = (await graph.getAttribute("viewBox")).split(" ").map(Number);
  await page.locator("#nodeViewPanel").getByRole("button", { name: "Zoom in", exact: true }).click();
  const buttonZoomedInViewBox = (await graph.getAttribute("viewBox")).split(" ").map(Number);
  if (!(buttonZoomedOutViewBox[2] > zoomedInViewBox[2] && buttonZoomedOutViewBox[3] > zoomedInViewBox[3])
    || !(buttonZoomedInViewBox[2] < buttonZoomedOutViewBox[2] && buttonZoomedInViewBox[3] < buttonZoomedOutViewBox[3])) {
    throw new Error(`Top-down graph button zoom contract failed: ${JSON.stringify({ zoomedInViewBox, buttonZoomedOutViewBox, buttonZoomedInViewBox })}`);
  }
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Overview" }).click();
  const overviewState = await page.locator("#nodeViewPanel").evaluate((panel) => ({
    overviewScale: panel.querySelector(".nv-graph")?.classList.contains("overview-scale"),
    minimapWindowWidth: Number(panel.querySelector(".nv-minimap-window")?.getAttribute("width") || 0),
    labelsVisible: [...panel.querySelectorAll(".nv-edge-label")].some((label) => getComputedStyle(label).display !== "none"),
  }));
  if (!overviewState.overviewScale || overviewState.minimapWindowWidth <= 0 || overviewState.labelsVisible) {
    throw new Error(`Graph overview did not simplify labels or synchronize the minimap: ${JSON.stringify(overviewState)}`);
  }
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Start" }).click();
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Inspector" }).click();
  const collapsedInspector = await page.locator("#nodeViewPanel").evaluate((panel) => ({
    collapsed: Boolean(panel.querySelector(".nv-shell.inspector-collapsed")),
    detailDisplay: getComputedStyle(panel.querySelector(".nv-detail")).display,
    columns: getComputedStyle(panel.querySelector(".nv-layout")).gridTemplateColumns.split(" ").length,
  }));
  if (!collapsedInspector.collapsed || collapsedInspector.detailDisplay !== "none" || collapsedInspector.columns !== 1) {
    throw new Error(`Graph inspector did not collapse into a full-width canvas: ${JSON.stringify(collapsedInspector)}`);
  }
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Inspector" }).click();
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Expand" }).click();
  const expandedGraph = await page.locator("#nodeViewPanel .nv-shell").evaluate((shell) => ({
    expanded: shell.classList.contains("expanded"),
    position: getComputedStyle(shell).position,
    right: shell.getBoundingClientRect().right,
    viewport: innerWidth,
  }));
  if (!expandedGraph.expanded || expandedGraph.position !== "fixed" || expandedGraph.right > expandedGraph.viewport) {
    throw new Error(`Expanded graph workspace is not viewport-bound: ${JSON.stringify(expandedGraph)}`);
  }
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Exit" }).click();
  await page.locator("#nodeViewPanel").screenshot({ path: graphScreenshot });
  await assertThemeContrast(page, "light");
  if (await page.locator("html").getAttribute("data-theme") !== "dark") {
    await page.locator("#themeToggle").click();
  }
  await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
  await assertThemeContrast(page, "dark");
  await page.locator('[data-explorer-tab="resource"]').click();
  await page.locator("#resourceMapPanel .evidence-treemap-tile").first().hover();
  await assertReadableState(page, "#resourceMapPanel .evidence-treemap-tile", "strong", "dark resource-map hover");
  await page.locator("#resourceMapPanel").screenshot({ path: resourceMapDarkScreenshot });
  await page.locator('[data-explorer-tab="ops"]').click();
  await assertExplorerVisualizationTheme(page, "dark");
  await page.locator('[data-explorer-tab="blocks"]').click();
  await page.locator("#blocksExplorerPanel .xr-block-row").first().hover();
  await assertReadableState(page, "#blocksExplorerPanel .xr-block-row", "strong", "dark block hover");
  await page.locator("#blocksExplorerPanel").screenshot({ path: blocksDarkScreenshot });
  await page.locator('[data-explorer-tab="node"]').click();
  await page.locator("#nodeViewPanel").screenshot({ path: graphDarkScreenshot });
  await page.locator("#themeToggle").click();
  await page.waitForFunction(() => document.documentElement.dataset.theme === "light");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#nodeViewPanel .nv-view-controls button", { hasText: "Start" }).click();
  const graphMobile = await page.locator("#nodeViewPanel").evaluate((panel) => ({
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    svgWidth: panel.querySelector(".nv-graph")?.getBoundingClientRect().width || 0,
    nodeWidth: panel.querySelector(".nv-node rect")?.getBoundingClientRect().width || 0,
    layoutColumns: getComputedStyle(panel.querySelector(".nv-layout")).gridTemplateColumns.split(" ").length,
    detailVisible: getComputedStyle(panel.querySelector(".nv-detail")).display !== "none",
    viewportTop: panel.querySelector(".nv-viewport")?.getBoundingClientRect().top || 0,
    returnVisible: getComputedStyle(panel.querySelector(".nv-return-graph")).display !== "none",
  }));
  if (graphMobile.documentOverflow > 1
    || graphMobile.panelOverflow > 1
    || graphMobile.svgWidth > 390
    || graphMobile.nodeWidth < 100
    || graphMobile.layoutColumns !== 1
    || graphMobile.detailVisible
    || !graphMobile.returnVisible) {
    throw new Error(`Top-down Graph mobile layout overflows or collapses: ${JSON.stringify(graphMobile)}`);
  }
  await page.locator("#nodeViewPanel .nv-viewport").scrollIntoViewIfNeeded();
  await page.locator("#nodeViewPanel .nv-node").nth(3).click();
  await page.waitForTimeout(450);
  const mobileSelection = await page.locator("#nodeViewPanel").evaluate((panel) => ({
    detail: panel.querySelector(".nv-detail")?.textContent || "",
    detailTop: panel.querySelector(".nv-detail")?.getBoundingClientRect().top || 0,
    detailBottom: panel.querySelector(".nv-detail")?.getBoundingClientRect().bottom || 0,
    closeVisible: getComputedStyle(panel.querySelector(".nv-detail-close")).display !== "none",
  }));
  if (!mobileSelection.detail.includes("OP #003") || !mobileSelection.detail.includes("INT8 compute")
    || mobileSelection.detailTop < 160 || mobileSelection.detailBottom > 836 || !mobileSelection.closeVisible) {
    throw new Error(`Mobile node selection did not surface its evidence: ${JSON.stringify(mobileSelection)}`);
  }
  await page.locator("#nodeViewPanel .nv-detail-close").click();
  await page.waitForTimeout(350);
  if (await page.locator("#nodeViewPanel .nv-detail").isVisible()) throw new Error("Mobile bottom-sheet inspector did not close without moving the graph viewport.");
  await page.locator("#nodeViewPanel").screenshot({ path: graphMobileScreenshot });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator("#nodeViewPanel .nv-node").nth(3).click();
  const publicQuantAction = await page.locator("#nodeViewPanel .nv-detail-actions button", { hasText: "Open Quant Evidence" }).evaluate((button) => ({
    disabled: button.disabled,
    title: button.title,
  }));
  if (!publicQuantAction.disabled || !publicQuantAction.title.includes("No per-axis kernel scale evidence")) {
    throw new Error(`Node View exposed unavailable per-axis evidence: ${JSON.stringify(publicQuantAction)}`);
  }
  await page.locator('[data-explorer-tab="quant"]').click();
  const publicQuantPanel = await page.locator("#quantEvidencePanel").textContent();
  if (!publicQuantPanel.includes("No per-axis INT8 kernel scale vectors")) {
    throw new Error(`Per-tensor sample did not expose an explicit Quant Evidence applicability boundary: ${publicQuantPanel}`);
  }
  await page.locator('[data-explorer-tab="node"]').click();

  const syntheticQuant = await page.evaluate(async () => {
    const { createQuantEvidenceController } = await import("./lib/quant-evidence-view.js");
    const root = document.createElement("div");
    document.body.append(root);
    const tensors = Array.from({ length: 10 }, (_, index) => ({
      index,
      name: `T${index}`,
      shape: [1],
      dtype: "INT8",
      constant_buffer: false,
      quant_scales: 0,
      scale_sample: [],
      zero_point_sample: [],
    }));
    tensors[0] = { ...tensors[0], name: "input", scale_sample: [0.1], zero_point_sample: [0], quant_scales: 1 };
    tensors[8] = {
      ...tensors[8],
      name: "T8 kernel",
      shape: [3, 1, 1, 1],
      constant_buffer: true,
      quant_scales: 3,
      quantized_dimension: 0,
      scale_sample: [1, 1e-9, 0.5],
      zero_point_sample: [0, 0, 0],
      scale_ratio_meaningful: true,
      scale_ratio: 1e9,
      scale_min: 1e-9,
      scale_max: 1,
    };
    tensors[9] = {
      ...tensors[9],
      name: "T9 bias",
      dtype: "INT32",
      shape: [3],
      constant_buffer: true,
      quant_scales: 3,
      scale_sample: [0.1, 1e-10, 0.05],
      zero_point_sample: [0, 0, 0],
    };
    const analysis = {
      format: "tflite",
      tensors,
      ops: [{
        index: 0,
        name: "CONV_2D",
        inputs: [0, 8, 9],
        outputs: [1],
        low_norm_filter_count: 2,
        low_norm_filter_total: 3,
      }],
    };
    const controller = createQuantEvidenceController({ root });
    controller.setAnalysis(analysis);
    const linked = controller.hasEvidenceForOp(0);
    const selected = controller.selectOp(0);
    const result = {
      text: root.textContent || "",
      chart: root.querySelectorAll(".qe-scale-chart polyline").length,
      rows: root.querySelectorAll(".qe-channel-table tbody tr").length,
      flags: root.querySelectorAll(".qe-flagged-row").length,
      linked,
      selected,
    };
    root.remove();
    return result;
  });
  if (syntheticQuant.chart !== 1
    || syntheticQuant.rows !== 3
    || syntheticQuant.flags !== 1
    || !syntheticQuant.linked
    || !syntheticQuant.selected
    || !syntheticQuant.text.includes("1 near-zero")
    || !syntheticQuant.text.includes("2 / 3")) {
    throw new Error(`Quant Evidence scale-vector contract failed: ${JSON.stringify(syntheticQuant)}`);
  }

  const projectedContractState = await page.evaluate(async () => {
    const { getRedesignContractState } = await import("./lib/node-view.js");
    const analysis = {
      ops: [{
        index: 0,
        name: "CONV_2D",
        channel_alignment_status: "misaligned",
        channel_alignment_multiple: 16,
        row_working_set_ratio: 1.2,
      }],
    };
    const projection = {
      constraints: [],
      op_projections: [{
        op_index: 0,
        shape_rule_status: "exact",
        source_outputs: [{ constant: false, shape: [1, 8, 8, 24] }],
        projected_outputs: [{ constant: false, shape: [1, 8, 8, 32] }],
        source_l1_ratio: 1.2,
        projected_l1_ratio: 0.6,
      }],
      propagation_edges: [],
    };
    return getRedesignContractState(analysis, projection, 0);
  });
  if (projectedContractState.id !== "satisfied"
    || projectedContractState.resolved.length !== 2
    || !projectedContractState.resolved.some((item) => item.includes("L1D"))
    || !projectedContractState.resolved.some((item) => item.includes("alignment"))) {
    throw new Error(`Projected contract fill did not reflect deterministically cleared conditions: ${JSON.stringify(projectedContractState)}`);
  }

  const selectedBlockLabel = (await page.locator("#blocksExplorerPanel .xr-block-detail h3").textContent())?.trim();
  await page.locator('[data-explorer-tab="blocks"]').click();
  await page.locator("#blocksExplorerPanel .xr-block-detail button", { hasText: "Clone to Redesign" }).click();
  await page.locator("#redesignPanel .xr-redesign-status-dock").waitFor({ timeout: 30_000 });
  await page.locator("#redesignPanel .xr-projection-table").waitFor({ state: "attached", timeout: 30_000 });
  const redesign = await page.locator("#redesignPanel").evaluate((panel) => ({
    banner: panel.querySelector(".xr-redesign-banner")?.textContent || "",
    selectedBlock: [...panel.querySelectorAll(".xr-redesign-editor label")]
      .find((label) => label.textContent?.includes("Block editor"))
      ?.querySelector("select")?.selectedOptions?.[0]?.textContent || "",
    hash: [...panel.querySelectorAll(".xr-source-binding .xr-ledger-row")]
      .find((row) => row.textContent?.includes("Source SHA-256"))?.textContent || "",
    rows: panel.querySelectorAll(".xr-projection-table tbody tr").length,
    statusCells: panel.querySelectorAll(".xr-redesign-status-dock .xr-status-cell").length,
    sourceCollapsed: !panel.querySelector(".xr-source-binding")?.open,
    workbenchColumns: getComputedStyle(panel.querySelector(".xr-redesign-workbench")).gridTemplateColumns,
    technicalLedger: [...panel.querySelectorAll(".xr-technical-ledger > summary")]
      .find((summary) => summary.textContent?.includes("Full metric"))?.textContent || "",
    footer: panel.querySelector(".xr-redesign-result > .xr-note")?.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    selectedContext: panel.querySelector(".xr-edit-context")?.textContent || "",
    selectedKicker: panel.querySelector(".xr-redesign-editor .xr-kicker")?.textContent || "",
    runDisabled: panel.querySelector('[data-redesign-action="run"]')?.disabled,
    resetAllDisabled: panel.querySelector('[data-redesign-action="reset-all"]')?.disabled,
    dockPosition: getComputedStyle(panel.querySelector(".xr-redesign-status-dock")).position,
    dockNotAssessed: (panel.querySelector(".xr-redesign-status-dock")?.textContent.match(/N\/A/g) || []).length,
    implementationNodes: panel.querySelectorAll(".xr-implementation-table tbody tr").length,
    implementationSummary: panel.querySelector(".xr-implementation-handoff")?.textContent || "",
    scenarioLab: panel.querySelector(".xr-scenario-lab")?.textContent || "",
  }));
  if (!redesign.banner.includes("PROJECTED_UNTRAINED")
    || !redesign.selectedBlock.includes(selectedBlockLabel)
    || !/[a-f0-9]{64}/i.test(redesign.hash)
    || redesign.rows !== 10
    || redesign.statusCells !== 6
    || !redesign.sourceCollapsed
    || redesign.workbenchColumns.split(" ").length < 2
    || !redesign.technicalLedger.includes("Full metric")
    || !redesign.footer.includes("verified in session")
    || !redesign.selectedKicker.includes("SELECTED OPERATOR")
    || !redesign.selectedContext.includes("semantic block")
    || !redesign.runDisabled || !redesign.resetAllDisabled
    || redesign.dockPosition === "sticky" || redesign.dockPosition === "fixed"
    || redesign.dockNotAssessed > 0
    || redesign.implementationNodes !== 65
    || !redesign.implementationSummary.includes("Unsupported / repeat0 / 0")
    || !redesign.scenarioLab.includes("Save, compare, and materialize")
    || redesign.overflow > 1) {
    throw new Error(`Redesign source binding or no-op projection failed: ${JSON.stringify(redesign)}`);
  }
  await page.locator("#redesignPanel .xr-scenario-lab button", { hasText: "Explore Pareto" }).click();
  await page.waitForFunction(() => {
    const panel = document.querySelector("#redesignPanel .xr-scenario-lab");
    return panel?.querySelectorAll(".xr-pareto-table tbody tr").length > 0
      && !panel.textContent?.includes("Exploring...");
  }, null, { timeout: 30_000 });
  const paretoUi = await page.locator("#redesignPanel .xr-scenario-lab").evaluate((panel) => ({
    rows: panel.querySelectorAll(".xr-pareto-table tbody tr").length,
    summary: [...panel.querySelectorAll(".xr-subsection-head span")]
      .find((span) => span.textContent?.includes("frontier"))?.textContent || "",
    text: panel.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (paretoUi.rows < 1
    || !/\d+ frontier \/ 25 evaluated/.test(paretoUi.summary)
    || !paretoUi.text.includes("structural proxy")
    || paretoUi.overflow > 1) {
    throw new Error(`Redesign Pareto workbench is incomplete: ${JSON.stringify(paretoUi)}`);
  }
  await page.locator("#redesignPanel .xr-scenario-lab button", { hasText: "Save current" }).click();
  if (!(await page.locator("#redesignPanel .xr-scenario-section", { hasText: "Scenario comparison" }).textContent())?.includes("1 saved")) {
    throw new Error("Redesign scenario save did not update the bound session ledger.");
  }
  const [scenarioDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#redesignPanel .xr-scenario-lab button", { hasText: "Export scenarios" }).click(),
  ]);
  const scenarioDownloadPath = await scenarioDownload.path();
  const scenarioExport = JSON.parse(await readFile(scenarioDownloadPath, "utf8"));
  if (scenarioExport.schema !== "deepbom.redesign_scenario_set.v1.1"
    || scenarioExport.scenario_count !== 1
    || scenarioExport.scenarios?.[0]?.source_mapping?.mapped_source_layer_count !== 65
    || scenarioExport.scenarios?.[0]?.regeneration?.unsupported_codegen_op_count !== 0
    || scenarioExport.pareto_search?.evaluated_candidate_count !== 25
    || !/^[a-f0-9]{64}$/.test(scenarioExport.ledger_sha256 || "")) {
    throw new Error(`Redesign scenario handoff export is incomplete: ${JSON.stringify(scenarioExport)}`);
  }
  const paretoChoice = page.locator("#redesignPanel .xr-pareto-table tbody tr").nth(1);
  const paretoChoiceInput = (await paretoChoice.locator("td").nth(0).textContent())?.trim();
  await paretoChoice.locator("button", { hasText: "Apply" }).click();
  await page.waitForFunction((input) => {
    const row = document.querySelector("#redesignPanel .xr-scenario-table tbody tr.current");
    const dock = document.querySelector("#redesignPanel .xr-redesign-status-dock");
    return row?.textContent?.includes(input) && !dock?.classList.contains("pending");
  }, paretoChoiceInput, { timeout: 30_000 });
  const appliedScenarioRows = await page.locator("#redesignPanel .xr-scenario-section", { hasText: "Scenario comparison" })
    .locator(".xr-scenario-table tbody tr")
    .count();
  if (appliedScenarioRows !== 2) {
    throw new Error(`Applying a Pareto candidate did not retain the saved baseline for comparison: ${appliedScenarioRows}`);
  }
  await page.locator("#redesignPanel .xr-scenario-lab").screenshot({ path: redesignScenarioScreenshot });

  const redesignNodeInitial = await page.locator("#redesignPanel").evaluate((panel) => ({
    nodes: panel.querySelectorAll(".xr-redesign-node-host .nv-node").length,
    visibleNodes: [...panel.querySelectorAll(".xr-redesign-node-host .nv-node")]
      .filter((node) => getComputedStyle(node).display !== "none").length,
    text: panel.querySelector(".xr-redesign-node-host")?.textContent || "",
    legend: panel.querySelector(".xr-redesign-node-legend")?.textContent || "",
    viewControls: [...panel.querySelectorAll(".nv-view-controls button")].map((button) => button.textContent?.trim()),
    fullActive: panel.querySelector('[data-view-scope="full"]')?.getAttribute("aria-pressed") === "true",
    selectedNodeWidth: panel.querySelector(".nv-node.selected rect")?.getBoundingClientRect().width || 0,
    direct: panel.querySelectorAll(".nv-node-redesign-direct").length,
    propagated: panel.querySelectorAll(".nv-node-redesign-propagated").length,
    issue: panel.querySelectorAll(".nv-contract-issue").length,
    watch: panel.querySelectorAll(".nv-contract-watch").length,
    satisfied: panel.querySelectorAll(".nv-contract-satisfied").length,
    conditional: panel.querySelectorAll(".nv-contract-conditional").length,
    unassessed: panel.querySelectorAll(".nv-contract-unassessed").length,
    blocked: panel.querySelectorAll(".nv-contract-blocked").length,
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (redesignNodeInitial.nodes !== 65
    || redesignNodeInitial.visibleNodes !== 65
    || !redesignNodeInitial.text.includes("Edit the graph contract in place")
    || !redesignNodeInitial.legend.includes("Border / change scope")
    || !redesignNodeInitial.legend.includes("Fill / projected status")
    || !redesignNodeInitial.legend.includes("Issue remains")
    || !redesignNodeInitial.viewControls.includes("Selection")
    || !redesignNodeInitial.viewControls.includes("Changes")
    || !redesignNodeInitial.viewControls.includes("Full")
    || !redesignNodeInitial.viewControls.includes("Inspector")
    || !redesignNodeInitial.viewControls.includes("Expand")
    || !redesignNodeInitial.fullActive
    || redesignNodeInitial.selectedNodeWidth < 8
    || redesignNodeInitial.issue + redesignNodeInitial.watch < 1
    || redesignNodeInitial.direct !== 0
    || redesignNodeInitial.propagated !== 0
    || redesignNodeInitial.issue + redesignNodeInitial.watch
      + redesignNodeInitial.satisfied + redesignNodeInitial.conditional
      + redesignNodeInitial.unassessed + redesignNodeInitial.blocked !== 65
    || redesignNodeInitial.overflow > 1) {
    throw new Error(`Node-first Redesign no-op state is incomplete: ${JSON.stringify(redesignNodeInitial)}`);
  }
  const redesignGraph = page.locator("#redesignPanel .nv-graph");
  const redesignViewBoxInitial = (await redesignGraph.getAttribute("viewBox")).split(" ").map(Number);
  await redesignGraph.dispatchEvent("wheel", {
    deltaY: 180,
    clientX: 420,
    clientY: 320,
  });
  const redesignViewBoxWheelOut = (await redesignGraph.getAttribute("viewBox")).split(" ").map(Number);
  await redesignGraph.dispatchEvent("wheel", {
    deltaY: -180,
    clientX: 420,
    clientY: 320,
  });
  const redesignViewBoxWheelIn = (await redesignGraph.getAttribute("viewBox")).split(" ").map(Number);
  await page.locator('#redesignPanel .nv-view-controls button[aria-label="Zoom out"]').click();
  const redesignViewBoxButtonOut = (await redesignGraph.getAttribute("viewBox")).split(" ").map(Number);
  await page.locator('#redesignPanel .nv-view-controls button[aria-label="Zoom in"]').click();
  const redesignViewBoxButtonIn = (await redesignGraph.getAttribute("viewBox")).split(" ").map(Number);
  if (!(redesignViewBoxWheelOut[2] > redesignViewBoxInitial[2])
    || !(redesignViewBoxWheelIn[2] < redesignViewBoxWheelOut[2])
    || !(redesignViewBoxButtonOut[2] > redesignViewBoxWheelIn[2])
    || !(redesignViewBoxButtonIn[2] < redesignViewBoxButtonOut[2])) {
    throw new Error(`Redesign zoom controls are not functional: ${JSON.stringify({
      redesignViewBoxInitial,
      redesignViewBoxWheelOut,
      redesignViewBoxWheelIn,
      redesignViewBoxButtonOut,
      redesignViewBoxButtonIn,
    })}`);
  }
  await page.locator('#redesignPanel [data-view-scope="selection"]').click();
  const redesignSelectionState = await page.locator("#redesignPanel").evaluate((panel) => ({
    visibleNodes: [...panel.querySelectorAll(".xr-redesign-node-host .nv-node")]
      .filter((node) => getComputedStyle(node).display !== "none").length,
    selectedNodeWidth: panel.querySelector(".nv-node.selected rect")?.getBoundingClientRect().width || 0,
  }));
  await page.locator('#redesignPanel [data-view-scope="full"]').click();
  const redesignFullState = await page.locator("#redesignPanel").evaluate((panel) => ({
    visibleNodes: [...panel.querySelectorAll(".xr-redesign-node-host .nv-node")]
      .filter((node) => getComputedStyle(node).display !== "none").length,
    active: panel.querySelector('[data-view-scope="full"]')?.getAttribute("aria-pressed"),
    stageHeight: panel.querySelector(".xr-redesign-node-stage")?.getBoundingClientRect().height || 0,
    viewportHeight: panel.querySelector(".xr-redesign-node-host .nv-viewport")?.getBoundingClientRect().height || 0,
    graphHeight: panel.querySelector(".xr-redesign-node-host .nv-graph")?.getBoundingClientRect().height || 0,
  }));
  if (!(redesignSelectionState.visibleNodes > 0 && redesignSelectionState.visibleNodes < 65)
    || redesignSelectionState.selectedNodeWidth < 40
    || redesignFullState.visibleNodes !== 65
    || redesignFullState.active !== "true"
    || redesignFullState.stageHeight < 590
    || redesignFullState.stageHeight > 1200
    || redesignFullState.viewportHeight < 500
    || redesignFullState.viewportHeight > 780
    || Math.abs(redesignFullState.graphHeight - redesignFullState.viewportHeight) > 1) {
    throw new Error(`Redesign scope controls do not restore the complete graph: ${JSON.stringify({ redesignSelectionState, redesignFullState })}`);
  }
  await page.locator("#redesignPanel .xr-redesign-node-stage").screenshot({ path: redesignGraphScreenshot });
  await page.locator("#redesignPanel .nv-flag-controls button", { hasText: "Next flag" }).click();
  const baselineIssueDetail = await page.locator("#redesignPanel .nv-detail").textContent();
  if (!baselineIssueDetail?.includes("Remaining signals")
    || baselineIssueDetail.includes("No flagged projected or retained condition")) {
    throw new Error(`Redesign baseline issue fill is not linked to its evidence: ${baselineIssueDetail}`);
  }
  const outputChannels = page.locator("#redesignPanel label", { hasText: "Output channels" }).locator("input");
  const sourceChannels = Number(await outputChannels.inputValue());
  await outputChannels.fill(String(sourceChannels + 8));
  await outputChannels.evaluate((input) => input.dispatchEvent(new Event("input", { bubbles: true })));
  await page.waitForFunction(() =>
    document.querySelectorAll("#redesignPanel .nv-node-redesign-direct").length > 0
    && document.querySelectorAll("#redesignPanel .nv-node-redesign-propagated").length > 0
    && document.querySelectorAll("#redesignPanel .nv-edge-redesign-direct, #redesignPanel .nv-edge-redesign-propagated").length > 0,
  null, { timeout: 30_000 });
  await page.waitForFunction(
    () => document.activeElement?.dataset?.redesignField === "output_channels",
    null,
    { timeout: 5_000 },
  );
  const activeField = await page.evaluate(() => document.activeElement?.dataset?.redesignField || "");
  if (activeField !== "output_channels") {
    throw new Error(`Redesign automatic projection did not preserve the active numeric control: ${activeField}`);
  }
  const redesignNodeEdited = await page.locator("#redesignPanel").evaluate((panel) => ({
    direct: panel.querySelectorAll(".nv-node-redesign-direct").length,
    propagated: panel.querySelectorAll(".nv-node-redesign-propagated").length,
    issue: panel.querySelectorAll(".nv-contract-issue").length,
    watch: panel.querySelectorAll(".nv-contract-watch").length,
    satisfied: panel.querySelectorAll(".nv-contract-satisfied").length,
    conditional: panel.querySelectorAll(".nv-contract-conditional").length,
    unassessed: panel.querySelectorAll(".nv-contract-unassessed").length,
    blocked: panel.querySelectorAll(".nv-contract-blocked").length,
    changedEdges: panel.querySelectorAll(".nv-edge-redesign-direct, .nv-edge-redesign-propagated").length,
    count: panel.querySelector(".xr-redesign-node-host .nv-count")?.textContent || "",
    sourceUnchanged: panel.textContent?.includes("Loaded source bytes unchanged: verified in session"),
  }));
  if (redesignNodeEdited.direct < 1
    || redesignNodeEdited.propagated < 1
    || redesignNodeEdited.issue + redesignNodeEdited.watch + redesignNodeEdited.satisfied
      + redesignNodeEdited.conditional + redesignNodeEdited.unassessed + redesignNodeEdited.blocked !== 65
    || redesignNodeEdited.changedEdges < 1
    || !redesignNodeEdited.count.includes("auto")
    || !redesignNodeEdited.sourceUnchanged) {
    throw new Error(`Node-first Redesign propagation overlay failed: ${JSON.stringify(redesignNodeEdited)}`);
  }
  await page.locator("#redesignPanel .nv-view-controls button", { hasText: "Changes" }).click();
  await page.locator("#redesignPanel .nv-node-redesign-propagated").first().click();
  const propagatedDetail = await page.locator("#redesignPanel .nv-detail").textContent();
  if (!propagatedDetail?.includes("Auto-adjusted")
    || !propagatedDetail.includes("propagated contract")
    || !propagatedDetail.includes("Contract")
    || !propagatedDetail.includes("Shape evidence")) {
    throw new Error(`Redesign related-layer selection did not expose the propagation ledger: ${propagatedDetail}`);
  }

  const inputWidth = page.locator("#redesignPanel label", { hasText: "Input width" }).locator("input");
  const previousProjectedMac = await page.locator("#redesignPanel .xr-projection-table tbody tr", { hasText: "MACs" })
    .locator("td")
    .nth(1)
    .textContent();
  await inputWidth.fill("112");
  await page.waitForFunction((previous) => {
    const rows = [...document.querySelectorAll("#redesignPanel .xr-projection-table tbody tr")];
    const mac = rows.find((row) => row.querySelector("th")?.textContent === "MACs");
    const projected = mac?.children[2]?.textContent || "";
    return projected && projected !== previous
      && !document.querySelector("#redesignPanel .xr-redesign-status-dock")?.classList.contains("pending");
  }, previousProjectedMac, { timeout: 30_000 });
  const projection = await page.locator("#redesignPanel").evaluate((panel) => ({
    status: panel.querySelector(".xr-redesign-result .xr-kicker")?.textContent || "",
    boundary: panel.querySelector(".xr-redesign-banner strong")?.textContent || "",
    constraints: panel.querySelectorAll(".xr-constraint").length,
    sourceUnchanged: panel.textContent?.includes("Loaded source bytes unchanged: verified in session"),
  }));
  if (projection.boundary !== "PROJECTED_UNTRAINED"
    || !projection.status.includes("assessed_with_serialized_shape_scaling")
    || projection.constraints < 1
    || !projection.sourceUnchanged) {
    throw new Error(`Edited projection did not preserve its evidence boundary: ${JSON.stringify(projection)}`);
  }

  await page.locator("#redesignPanel .nv-view-controls button", { hasText: "Selection" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  const redesignMobile = await page.locator("#redesignPanel").evaluate((panel) => ({
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    graphWidth: panel.querySelector(".xr-redesign-node-host svg")?.getBoundingClientRect().width || 0,
    selectedNodeWidth: panel.querySelector(".nv-node.selected rect")?.getBoundingClientRect().width || 0,
    workbenchColumns: getComputedStyle(panel.querySelector(".xr-redesign-workbench")).gridTemplateColumns,
  }));
  if (redesignMobile.documentOverflow > 1
    || redesignMobile.panelOverflow > 1
    || redesignMobile.graphWidth > 390
    || redesignMobile.selectedNodeWidth < 34
    || redesignMobile.workbenchColumns.split(" ").length !== 1) {
    throw new Error(`Node-first Redesign mobile layout overflows: ${JSON.stringify(redesignMobile)}`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });

  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();
  await page.locator("#kernelInspectorBody tr").first().waitFor();
  const kernelInitial = await page.locator("#kernelInspectorPanel").evaluate((panel) => ({
    rows: panel.querySelectorAll("#kernelInspectorBody tr").length,
    placement: panel.querySelector("#explorerExecutionPlacementPanel")?.textContent || "",
    summary: panel.querySelector("#kernelInspectorSummary")?.textContent || "",
    internalOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    overflowX: getComputedStyle(panel).overflowX,
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (kernelInitial.rows !== 65
    || !kernelInitial.placement.includes("Deployment evidence topology")
    || !kernelInitial.summary.includes("65/65 ops")
    || (kernelInitial.internalOverflow > 1 && !["auto", "scroll"].includes(kernelInitial.overflowX))
    || kernelInitial.documentOverflow > 1) {
    throw new Error(`Kernel & Runtime did not preserve the TFLite operator ledger: ${JSON.stringify(kernelInitial)}`);
  }
  await page.locator("#kernelInspectorSearch").fill("DEPTHWISE_CONV_2D");
  await page.waitForFunction(() => {
    const rows = [...document.querySelectorAll("#kernelInspectorBody tr")];
    return rows.length > 0 && rows.every((row) => row.children[1]?.textContent?.includes("DEPTHWISE_CONV_2D"));
  });
  const kernelSearchRows = await page.locator("#kernelInspectorBody tr").count();
  if (kernelSearchRows !== 17) {
    throw new Error(`Kernel search returned ${kernelSearchRows} rows instead of the 17 depthwise operators.`);
  }
  await page.locator("#kernelInspectorSearch").fill("");
  await page.waitForFunction(() => document.querySelectorAll("#kernelInspectorBody tr").length === 65);
  const kernelFirstIndex = Number((await page.locator("#kernelInspectorBody tr").first().locator("td").first().textContent()).replace("#", ""));
  const kernelFirstName = ((await page.locator("#kernelInspectorBody tr").first().locator("td").nth(1).textContent()) || "").split("|")[0].trim();
  await page.locator("#kernelInspectorBody tr").first().click();
  const kernelSelection = await page.locator("#graphExplorer").evaluate((panel) => ({
    active: panel.querySelector("[data-explorer-tab].active")?.dataset.explorerTab || "",
    detail: panel.querySelector("#opDetail")?.textContent || "",
  }));
  if (kernelSelection.active !== "kernels" || !kernelSelection.detail.includes(kernelFirstName)) {
    throw new Error(`Kernel row selection did not update the shared operator detail in place: ${JSON.stringify(kernelSelection)}`);
  }
  await page.locator('[data-explorer-tab="ops"]').click();
  if (await page.locator(`#graphOpBody tr[data-op-index="${kernelFirstIndex}"]`).getAttribute("aria-selected") !== "true") {
    throw new Error("Kernel row selection was not preserved when the Operator Table was opened.");
  }

  await page.locator('[data-explorer-tab="layered"]').click();
  await page.locator("#layeredViewPanel .layered-canvas").waitFor();
  const layeredSelection = await page.locator("#layeredViewPanel").evaluate((panel) => {
    const canvas = panel.querySelector(".layered-canvas");
    const tooltip = panel.querySelector(".layered-tooltip");
    const rect = canvas.getBoundingClientRect();
    let selected = null;
    for (let y = 8; y < Math.min(rect.height - 4, 180) && !selected; y += 8) {
      for (let x = 8; x < Math.min(rect.width - 4, 420); x += 8) {
        canvas.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: rect.left + x, clientY: rect.top + y }));
        if (!tooltip.hidden && /^#\d+\s/.test(tooltip.textContent || "")) {
          selected = { index: Number((tooltip.textContent.match(/^#(\d+)/) || [])[1]), text: tooltip.textContent, x, y };
          canvas.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: rect.left + x, clientY: rect.top + y }));
          break;
        }
      }
    }
    return selected;
  });
  if (!layeredSelection || !Number.isInteger(layeredSelection.index)) {
    throw new Error(`Architecture Runs hover/click hit-testing did not resolve an operator: ${JSON.stringify(layeredSelection)}`);
  }
  await page.locator('[data-explorer-tab="ops"]').click();
  if (await page.locator(`#graphOpBody tr[data-op-index="${layeredSelection.index}"]`).getAttribute("aria-selected") !== "true") {
    throw new Error(`Architecture Runs selected #${layeredSelection.index}, but Operator Table did not retain that selection.`);
  }

  await page.locator('[data-explorer-tab="cache"]').click();
  await page.locator("#cacheExplorerPanel .xr-scatter-wrap svg").waitFor();
  const cache = await page.locator("#cacheExplorerPanel").evaluate((panel) => ({
    points: panel.querySelectorAll(".xr-cache-point").length,
    projected: panel.querySelectorAll(".xr-projected-point").length,
    vectors: panel.querySelectorAll(".xr-projection-vector").length,
    guides: panel.querySelectorAll(".xr-cache-guide").length,
    timeline: panel.querySelectorAll(".xr-timeline-row").length,
    payloadBars: panel.querySelectorAll(".xr-payload-bars .xr-payload-row").length,
    text: panel.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (cache.points < 40
    || cache.projected < 40
    || cache.vectors < 20
    || cache.guides < 2
    || cache.timeline < 40
    || cache.payloadBars !== 4
    || !cache.text.includes("This is not a cache hit-rate or residency claim")
    || cache.overflow > 1) {
    throw new Error(`Cache surface or WASM projection overlay is incomplete: ${JSON.stringify(cache)}`);
  }
  const cacheTimeline = page.locator("#cacheExplorerPanel .xr-timeline-row").first();
  const cacheTimelineIndex = Number(((await cacheTimeline.getAttribute("title"))?.match(/^#(\d+)/) || [])[1]);
  await cacheTimeline.click();
  const cacheSelection = await page.locator("#cacheExplorerPanel").evaluate((panel) => ({
    active: document.querySelector("[data-explorer-tab].active")?.dataset.explorerTab || "",
    heading: panel.querySelector(".xr-cache-detail .xr-panel-heading")?.textContent || "",
  }));
  if (cacheSelection.active !== "cache" || !cacheSelection.heading.includes(`#${String(cacheTimelineIndex).padStart(3, "0")}`)) {
    throw new Error(`Cache timeline did not update its in-place payload ledger: ${JSON.stringify(cacheSelection)}`);
  }
  await page.locator("#cacheExplorerPanel .xr-cache-detail button", { hasText: "Open in Ops" }).click();
  await page.waitForFunction((index) => document.querySelector(`#graphOpBody tr[data-op-index="${index}"]`)?.getAttribute("aria-selected") === "true", cacheTimelineIndex);
  if (!(await page.locator('[data-explorer-tab="ops"]').evaluate((button) => button.classList.contains("active")))) {
    throw new Error("Cache Open in Ops command did not open the corresponding Operator Table row.");
  }
  await page.locator('[data-explorer-tab="cache"]').click();

  await page.setViewportSize({ width: 390, height: 844 });
  await assertMobileExplorerTabs(page);
  await page.locator('[data-explorer-tab="cache"]').click();
  const mobile = await page.evaluate(() => ({
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, document.querySelector("#cacheExplorerPanel").scrollWidth - document.querySelector("#cacheExplorerPanel").clientWidth),
    svgWidth: document.querySelector("#cacheExplorerPanel svg")?.getBoundingClientRect().width || 0,
    offenders: [...document.querySelectorAll("body *")]
      .filter((node) => node.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 8)
      .map((node) => {
        const rect = node.getBoundingClientRect();
        return `${node.tagName.toLowerCase()}#${node.id}.${node.className}[${Math.round(rect.left)},${Math.round(rect.right)},w${Math.round(rect.width)},s${node.scrollWidth},c${node.clientWidth}]`;
      }),
    wide: [...document.querySelectorAll("body *")]
      .filter((node) => node.getBoundingClientRect().width > window.innerWidth + 1)
      .slice(0, 8)
      .map((node) => {
        const rect = node.getBoundingClientRect();
        return `${node.tagName.toLowerCase()}#${node.id}.${node.className}[${Math.round(rect.left)},${Math.round(rect.right)},w${Math.round(rect.width)},s${node.scrollWidth},c${node.clientWidth}]`;
      }),
    containers: [document.body, document.querySelector(".layout"), document.querySelector(".session-anchor"), document.querySelector(".workflow-console"), document.querySelector(".workflow-rail")]
      .filter(Boolean)
      .map((node) => {
        const rect = node.getBoundingClientRect();
        const style = getComputedStyle(node);
        return `${node.tagName.toLowerCase()}.${node.className}[${Math.round(rect.left)},${Math.round(rect.right)},w${Math.round(rect.width)},s${node.scrollWidth},c${node.clientWidth},ox=${style.overflowX}]`;
      }),
    nonRailOffenders: [...document.querySelectorAll("body *")]
      .filter((node) => !node.closest(".workflow-rail") && node.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 12)
      .map((node) => {
        const rect = node.getBoundingClientRect();
        return `${node.tagName.toLowerCase()}#${node.id}.${node.className}[${Math.round(rect.left)},${Math.round(rect.right)},w${Math.round(rect.width)},s${node.scrollWidth},c${node.clientWidth}]`;
      }),
  }));
  if (mobile.documentOverflow > 1 || mobile.panelOverflow > 1 || mobile.svgWidth > 390) {
    throw new Error(`Explorer mobile layout overflows: ${JSON.stringify(mobile)}`);
  }

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('[data-explorer-tab="ops"]').click();
  await page.locator("#graphSearch").fill("AVERAGE_POOL_2D");
  await page.locator('[data-filter-group="bound"][data-filter-value="memory-bound"]').click();
  await page.locator("#graphDepth").selectOption("all");
  await page.locator('[data-graph-mode="stage"]').click();
  await page.locator('[data-explorer-tab="tensors"]').click();
  await page.locator("#tensorSearch").fill("t171");
  await page.locator('[data-tfilter="kernel"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();
  await page.locator("#kernelInspectorSearch").fill("DEPTHWISE_CONV_2D");
  await page.locator('[data-kernel-filter="tail"]').click();
  await page.locator('[data-explorer-tab="node"]').click();
  await page.locator('#nodeViewPanel input[aria-label="Search graph nodes"]').fill("CONV_2D");
  await page.locator('[data-workflow-step="input"]').click();
  await runAudit(page, ONNX, "sample_cnn_float.onnx");
  await page.locator('[data-workflow-step="audit"]').click();
  await page.locator('[data-audit-tab="quant-labs"]').click();
  const onnxQuantLabs = await page.locator(".quant-lab-workbench").evaluate((root) => ({
    active: root.querySelector("[data-quant-lab-tab].active")?.dataset.quantLabTab || "",
    hiddenTfliteTabs: [...root.querySelectorAll("[data-quant-lab-tab][data-tflite-only='true']")].filter((tab) => tab.hidden).length,
    tfliteTabs: root.querySelectorAll("[data-quant-lab-tab][data-tflite-only='true']").length,
    qdqCells: root.querySelectorAll(".qdq-cell").length,
    text: root.querySelector("[data-quant-qdq-action]")?.textContent || "",
    overflow: Math.max(0, root.scrollWidth - root.clientWidth),
  }));
  if (onnxQuantLabs.active !== "qdq-action"
    || onnxQuantLabs.hiddenTfliteTabs !== onnxQuantLabs.tfliteTabs
    || onnxQuantLabs.qdqCells < 1
    || !onnxQuantLabs.text.includes("Training path unbound")
    || onnxQuantLabs.overflow > 1) {
    throw new Error(`ONNX Quant Labs format isolation failed: ${JSON.stringify(onnxQuantLabs)}`);
  }
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="ops"]').click();
  const onnxExplorerReset = await page.locator("#graphExplorer").evaluate((root) => {
    const graphStats = root.querySelector("#graphStats")?.textContent || "";
    const expectedOps = Number((graphStats.match(/^(\d+) ops/) || [])[1]);
    return {
      expectedOps,
      operatorRows: root.querySelectorAll("#graphOpBody tr").length,
      graphSearch: root.querySelector("#graphSearch")?.value,
      graphDepth: root.querySelector("#graphDepth")?.value,
      graphMode: root.querySelector("[data-graph-mode].active")?.dataset.graphMode || "",
      activeBound: root.querySelector('[data-filter-group="bound"].active')?.dataset.filterValue,
      activeXnn: root.querySelector('[data-filter-group="xnn"].active')?.dataset.filterValue,
      activeQuant: root.querySelector('[data-filter-group="quant"].active')?.dataset.filterValue,
      filterCount: root.querySelector("#opFilterCount")?.textContent || "",
    };
  });
  if (!Number.isInteger(onnxExplorerReset.expectedOps)
    || onnxExplorerReset.operatorRows !== onnxExplorerReset.expectedOps
    || onnxExplorerReset.graphSearch !== ""
    || onnxExplorerReset.graphDepth !== "2"
    || onnxExplorerReset.graphMode !== "deploy"
    || onnxExplorerReset.activeBound !== ""
    || onnxExplorerReset.activeXnn !== ""
    || onnxExplorerReset.activeQuant !== ""
    || onnxExplorerReset.filterCount !== "") {
    throw new Error(`A new artifact inherited stale Operator Explorer state: ${JSON.stringify(onnxExplorerReset)}`);
  }
  await page.locator('[data-explorer-tab="tensors"]').click();
  const onnxTensorReset = await page.locator("#tensorExplorerPanel").evaluate((panel) => ({
    search: panel.querySelector("#tensorSearch")?.value,
    filter: panel.querySelector("[data-tfilter].active")?.dataset.tfilter || "",
    rows: panel.querySelectorAll("#tensorBody tr").length,
    expected: Number((panel.querySelector(".panel-count")?.textContent?.match(/^(\d+)/) || [])[1]),
  }));
  if (onnxTensorReset.search !== ""
    || onnxTensorReset.filter !== "all"
    || !onnxTensorReset.rows
    || (Number.isInteger(onnxTensorReset.expected) && onnxTensorReset.rows !== onnxTensorReset.expected)) {
    throw new Error(`A new artifact inherited stale Tensor Explorer state: ${JSON.stringify(onnxTensorReset)}`);
  }
  await page.locator('[data-explorer-tab="kernels"]').click();
  const onnxKernelReset = await page.locator("#kernelInspectorPanel").evaluate((panel) => ({
    search: panel.querySelector("#kernelInspectorSearch")?.value,
    filter: panel.querySelector("[data-kernel-filter].active")?.dataset.kernelFilter || "",
  }));
  if (onnxKernelReset.search !== "" || onnxKernelReset.filter !== "all") {
    throw new Error(`A new artifact inherited stale Kernel Explorer state: ${JSON.stringify(onnxKernelReset)}`);
  }
  await page.locator('[data-explorer-tab="node"]').click();
  const onnxNodeSearch = await page.locator('#nodeViewPanel input[aria-label="Search graph nodes"]').inputValue();
  if (onnxNodeSearch !== "") {
    throw new Error(`A new artifact inherited stale Node View search state: ${onnxNodeSearch}`);
  }
  await page.locator('[data-explorer-tab="blocks"]').click();
  const onnxBlocks = await page.locator("#blocksExplorerPanel").textContent();
  if (!onnxBlocks.includes("Block Inventory not assessed") || !onnxBlocks.includes("intentionally suppressed")) {
    throw new Error(`ONNX block semantics did not fail closed: ${onnxBlocks}`);
  }
  await page.locator('[data-explorer-tab="cache"]').click();
  const onnxCache = await page.locator("#cacheExplorerPanel").evaluate((panel) => ({
    points: panel.querySelectorAll(".xr-cache-point").length,
    text: panel.textContent || "",
  }));
  if (onnxCache.points !== 2 || !onnxCache.text.includes("Logical row payload")) {
    throw new Error(`ONNX deterministic cache payload is missing: ${JSON.stringify(onnxCache)}`);
  }
  const onnxRedesign = await page.locator('[data-workflow-step="redesign"]').evaluate((step) => ({
    hidden: step.hidden,
    applicable: step.dataset.formatApplicable,
    status: step.dataset.applicabilityStatus,
    reason: step.dataset.applicabilityReason,
  }));
  if (onnxRedesign.hidden || onnxRedesign.applicable !== "false"
    || onnxRedesign.status !== "not_applicable" || !onnxRedesign.reason) {
    throw new Error(`ONNX Redesign applicability is not isolated: ${JSON.stringify(onnxRedesign)}`);
  }
  if (errors.length) throw new Error(`Browser errors:\n${errors.join("\n")}`);
  console.log(`Explorer + Redesign viewer check passed (${path.relative(ROOT, SERVE_ROOT) || "source"}; TFLite desktop/mobile projection overlay; ONNX fail-closed behavior).`);
  console.log(`graph=${graphScreenshot}`);
  console.log(`graph_dark=${graphDarkScreenshot}`);
  console.log(`graph_mobile=${graphMobileScreenshot}`);
  console.log(`first_visit_mobile=${firstVisitMobileScreenshot}`);
  console.log(`blocks=${blocksScreenshot}`);
  console.log(`resource_map=${resourceMapScreenshot}`);
  console.log(`resource_map_dark=${resourceMapDarkScreenshot}`);
  console.log(`operator=${operatorScreenshot}`);
  console.log(`tensors=${tensorsScreenshot}`);
  console.log(`blocks_dark=${blocksDarkScreenshot}`);
  console.log(`redesign_scenarios=${redesignScenarioScreenshot}`);
  console.log(`redesign_graph=${redesignGraphScreenshot}`);
} catch (error) {
  const diagnostics = errors.length ? `\nBrowser diagnostics:\n${errors.join("\n")}` : "";
  throw new Error(`${error instanceof Error ? error.message : String(error)}${diagnostics}`, { cause: error });
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

async function runAudit(page, modelPath, name) {
  await page.locator("#fileInput").setInputFiles({
    name,
    mimeType: "application/octet-stream",
    buffer: readFileSync(modelPath),
  });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  const controlsBefore = await artifactControlGeometry(page);
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`${name}: ${status}`);
  const controlsAfter = await artifactControlGeometry(page);
  for (const key of Object.keys(controlsBefore)) {
    for (const metric of ["left", "top", "width"]) {
      if (Math.abs(controlsBefore[key][metric] - controlsAfter[key][metric]) > 1) {
        throw new Error(`${name}: artifact control moved after Run Static Audit (${key}.${metric}): ${JSON.stringify({ before: controlsBefore, after: controlsAfter })}`);
      }
    }
  }
  const publicExports = await page.evaluate(() => ({
    cdx17: document.querySelector("#downloadCycloneDxEvidence")?.disabled,
    cdx20: document.querySelector("#downloadCycloneDx20DraftStatus")?.disabled,
    companion: document.querySelector("#downloadRuntimeRequirements")?.disabled,
    pack: document.querySelector("#downloadContractPack")?.disabled,
  }));
  if (publicExports.cdx17 || publicExports.cdx20 || !publicExports.companion || !publicExports.pack) {
    throw new Error(`${name}: guest CycloneDX/companion access split is incorrect: ${JSON.stringify(publicExports)}`);
  }
  await page.evaluate(() => {
    const consoleNode = document.querySelector("#workflowConsole");
    const top = consoleNode ? consoleNode.getBoundingClientRect().top + window.scrollY - 8 : 0;
    window.scrollTo({ top, behavior: "instant" });
  });
  await page.waitForFunction(() => document.querySelector("#workflowConsole")?.getBoundingClientRect().top <= 16,
    null, { timeout: 5_000 });
  const scrollContract = await page.evaluate(() => {
    const node = document.querySelector(".workflow-rail");
    const rect = node.getBoundingClientRect();
    return {
      scrollPaddingTop: Number.parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop),
      top: rect.top,
      bottom: rect.bottom,
      viewportHeight: innerHeight,
      scrollY,
      scrollHeight: document.documentElement.scrollHeight,
      bodyScrollHeight: document.body.scrollHeight,
      railOverflowX: getComputedStyle(node).overflowX,
      railOverflowY: getComputedStyle(node).overflowY,
      consoleTop: document.querySelector("#workflowConsole")?.getBoundingClientRect().top,
    };
  });
  if (scrollContract.scrollPaddingTop > 16 || scrollContract.top < 0
    || scrollContract.bottom > scrollContract.viewportHeight) {
    throw new Error(`${name}: workflow navigation cannot enter the viewport: ${JSON.stringify(scrollContract)}`);
  }
}

async function assertThemeContrast(page, expectedTheme) {
  const snapshot = await page.evaluate(() => {
    const parse = (value) => {
      const text = String(value).trim();
      if (/^#[\da-f]{6}$/i.test(text)) {
        return [1, 3, 5].map((offset) => Number.parseInt(text.slice(offset, offset + 2), 16) / 255);
      }
      return [...text.matchAll(/[\d.]+/g)].slice(0, 3).map((match) => Number(match[0]) / 255);
    };
    const luminance = (value) => parse(value).reduce((sum, channel, index) => {
      const linear = channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      return sum + linear * [0.2126, 0.7152, 0.0722][index];
    }, 0);
    const ratio = (first, second) => {
      const [high, low] = [luminance(first), luminance(second)].sort((a, b) => b - a);
      return (high + 0.05) / (low + 0.05);
    };
    const root = getComputedStyle(document.documentElement);
    const token = (name) => root.getPropertyValue(name).trim();
    const selected = document.querySelector(".explorer-tab.active");
    const selectedStyle = getComputedStyle(selected);
    return {
      theme: document.documentElement.dataset.theme,
      body: ratio(token("--ink"), token("--bg")),
      selected: ratio(selectedStyle.color, selectedStyle.backgroundColor),
      selectedColors: `${selectedStyle.color} on ${selectedStyle.backgroundColor}`,
      nodeTitle: ratio(token("--viz-title"), token("--viz-node")),
      nodeMeta: ratio(token("--viz-meta"), token("--viz-node")),
      onAccent: ratio(token("--on-accent"), token("--accent")),
      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  });
  const failed = Object.entries(snapshot)
    .filter(([key, value]) => !["theme", "overflow", "selectedColors"].includes(key) && value < 4.5)
    .map(([key, value]) => `${key}=${value.toFixed(2)}`);
  if (snapshot.theme !== expectedTheme || snapshot.overflow > 1 || failed.length) {
    throw new Error(`${expectedTheme} theme contrast contract failed: ${JSON.stringify(snapshot)}; ${failed.join(", ")}`);
  }
}

async function assertExplorerVisualizationTheme(page, expectedTheme) {
  const lane = page.locator("#opTimeline .profile-lane-stack");
  await lane.waitFor();
  await lane.scrollIntoViewIfNeeded();
  const box = await lane.boundingBox();
  if (!box) throw new Error(`${expectedTheme} Explorer timeline has no rendered geometry.`);
  const hoverX = box.x + Math.min(box.width - 4, Math.max(4, box.width * 0.42));
  const hoverY = box.y + Math.min(box.height - 4, 12);
  await page.mouse.move(hoverX, hoverY);
  await page.waitForFunction(() => {
    const tooltip = document.querySelector("#opTimeline .layered-tooltip");
    return tooltip && !tooltip.hidden && tooltip.textContent?.trim();
  });

  const snapshot = await page.evaluate(() => {
    const channels = (value) => {
      const text = String(value).trim();
      if (/^#[\da-f]{6}$/i.test(text)) {
        return [1, 3, 5].map((offset) => Number.parseInt(text.slice(offset, offset + 2), 16));
      }
      return [...text.matchAll(/[\d.]+/g)].slice(0, 3).map((match) => Number(match[0]));
    };
    const luminance = (value) => channels(value).reduce((sum, channel, index) => {
      const normalized = channel / 255;
      const linear = normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      return sum + linear * [0.2126, 0.7152, 0.0722][index];
    }, 0);
    const ratio = (first, second) => {
      const [high, low] = [luminance(first), luminance(second)].sort((a, b) => b - a);
      return (high + 0.05) / (low + 0.05);
    };
    const sameColor = (first, second) => channels(first).every((value, index) => Math.abs(value - channels(second)[index]) <= 1);
    const root = getComputedStyle(document.documentElement);
    const graph = getComputedStyle(document.querySelector(".graph-map-viewport"));
    const timeline = getComputedStyle(document.querySelector("#opTimeline .memory-timeline-canvas"));
    const tooltipNode = document.querySelector("#opTimeline .layered-tooltip");
    const tooltip = getComputedStyle(tooltipNode);
    const canvasToken = root.getPropertyValue("--viz-canvas").trim();
    return {
      theme: document.documentElement.dataset.theme,
      graphBackground: graph.backgroundColor,
      timelineBackground: timeline.backgroundColor,
      canvasToken,
      graphUsesCanvas: sameColor(graph.backgroundColor, canvasToken),
      timelineUsesCanvas: sameColor(timeline.backgroundColor, canvasToken),
      tooltipText: tooltipNode.textContent?.trim() || "",
      tooltipForeground: tooltip.color,
      tooltipBackground: tooltip.backgroundColor,
      tooltipContrast: ratio(tooltip.color, tooltip.backgroundColor),
    };
  });
  if (snapshot.theme !== expectedTheme
    || !snapshot.graphUsesCanvas
    || !snapshot.timelineUsesCanvas
    || !snapshot.tooltipText
    || snapshot.tooltipContrast < 4.5) {
    throw new Error(`${expectedTheme} Explorer visualization theme contract failed: ${JSON.stringify(snapshot)}`);
  }
  const opIndex = Number((snapshot.tooltipText.match(/^#(\d+)/) || [])[1]);
  if (!Number.isInteger(opIndex)) {
    throw new Error(`${expectedTheme} Explorer timeline tooltip did not identify an operator: ${snapshot.tooltipText}`);
  }
  await page.mouse.click(hoverX, hoverY);
  await page.waitForFunction((index) => document.querySelector(`#graphOpBody tr[data-op-index="${index}"]`)?.getAttribute("aria-selected") === "true", opIndex);
}

async function assertMobileExplorerTabs(page) {
  const targets = {
    node: "#nodeViewPanel",
    resource: "#resourceMapPanel",
    blocks: "#blocksExplorerPanel",
    ops: "#graphOpsView",
    tensors: "#tensorExplorerPanel",
    quant: "#quantEvidencePanel",
    kernels: "#kernelInspectorPanel",
    cache: "#cacheExplorerPanel",
    layered: "#layeredViewPanel",
  };
  for (const [tab, selector] of Object.entries(targets)) {
    await page.locator(`[data-explorer-tab="${tab}"]`).click();
    await page.waitForTimeout(30);
    const snapshot = await page.evaluate(({ tab, selector, targets }) => {
      const visible = (query) => {
        const node = document.querySelector(query);
        return Boolean(node && !node.hidden && node.getClientRects().length && getComputedStyle(node).display !== "none");
      };
      return {
        active: document.querySelector("[data-explorer-tab].active")?.dataset.explorerTab || "",
        selected: document.querySelector(`[data-explorer-tab="${tab}"]`)?.getAttribute("aria-selected"),
        intendedVisible: visible(selector),
        visibleTargets: Object.entries(targets).filter(([, query]) => visible(query)).map(([key]) => key),
        documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      };
    }, { tab, selector, targets });
    if (snapshot.active !== tab
      || snapshot.selected !== "true"
      || !snapshot.intendedVisible
      || snapshot.visibleTargets.length !== 1
      || snapshot.visibleTargets[0] !== tab
      || snapshot.documentOverflow > 1) {
      throw new Error(`Mobile Explorer tab ${tab} is not isolated or viewport-safe: ${JSON.stringify(snapshot)}`);
    }
  }
  await page.locator('[data-explorer-tab="node"]').focus();
  await page.locator('[data-explorer-tab="node"]').press("ArrowRight");
  if (!(await page.locator('[data-explorer-tab="resource"]').evaluate((button) => button.classList.contains("active") && button.getAttribute("aria-selected") === "true"))) {
    throw new Error("Explorer roving keyboard navigation did not activate the next tab.");
  }
}

async function assertReadableState(page, containerSelector, textSelector, label) {
  const snapshot = await page.evaluate(({ containerSelector, textSelector }) => {
    const container = document.querySelector(containerSelector);
    const textNode = container?.querySelector(textSelector);
    if (!container || !textNode) return { missing: true };
    const rgba = (value) => {
      const values = [...String(value).matchAll(/[\d.]+/g)].map((match) => Number(match[0]));
      return [values[0] || 0, values[1] || 0, values[2] || 0, values.length > 3 ? values[3] : 1];
    };
    const over = (top, bottom) => {
      const alpha = top[3] + bottom[3] * (1 - top[3]);
      if (alpha <= 0) return [0, 0, 0, 0];
      return [0, 1, 2].map((index) => (
        (top[index] * top[3] + bottom[index] * bottom[3] * (1 - top[3])) / alpha
      )).concat(alpha);
    };
    const chain = [];
    for (let node = container; node instanceof Element; node = node.parentElement) chain.push(node);
    let background = [255, 255, 255, 1];
    for (const node of chain.reverse()) background = over(rgba(getComputedStyle(node).backgroundColor), background);
    const foreground = rgba(getComputedStyle(textNode).color);
    const displayedForeground = over(foreground, background);
    const linear = (channel) => {
      const normalized = channel / 255;
      return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
    };
    const luminance = (color) => 0.2126 * linear(color[0]) + 0.7152 * linear(color[1]) + 0.0722 * linear(color[2]);
    const first = luminance(displayedForeground);
    const second = luminance(background);
    return {
      missing: false,
      ratio: (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05),
      color: getComputedStyle(textNode).color,
      background: getComputedStyle(container).backgroundColor,
    };
  }, { containerSelector, textSelector });
  if (snapshot.missing || snapshot.ratio < 4.5) {
    throw new Error(`${label} contrast failed: ${JSON.stringify(snapshot)}`);
  }
}

async function artifactControlGeometry(page) {
  return page.evaluate(() => Object.fromEntries([
    ["artifact", document.querySelector(".upload-controls > .file-button:nth-of-type(1)")],
    ["package", document.querySelector(".upload-controls > .file-button:nth-of-type(2)")],
    ["run", document.querySelector("#runAudit")],
    ["sample", document.querySelector(".sample-model-control")],
  ].map(([key, node]) => {
    const { left, top, width } = node.getBoundingClientRect();
    return [key, { left: left + window.scrollX, top: top + window.scrollY, width }];
  })));
}

async function acceptAgreement(page) {
  if (!(await page.locator("#agreementBackdrop").isVisible())) return;
  await page.locator("#privacyAgree").check();
  await page.locator("#acceptAgreement").click();
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
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".onnx": "application/octet-stream",
    ".tflite": "application/octet-stream",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
