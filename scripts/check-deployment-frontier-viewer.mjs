import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { initSync as initDeepBom, analyze_deepbom } from "../web/protected/deepbom/pkg/deepbom_wasm.js";
import { applyProtectedOrtCompatibilityEvidence } from "../web/lib/ort-compatibility-evidence.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { analyzeOnnxModel } from "../web/onnx.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const ONNX_MODEL = path.join(ROOT, "web", "samples", "sample_cnn_float.onnx");
const LONG_MODEL_NAME = `${"production_research_candidate_".repeat(6)}mobilenet_v2_1.0_224_quant.tflite`;
const DOM_BINDING_IDS = [...readFileSync(path.join(ROOT, "web", "lib", "elements.js"), "utf8")
  .matchAll(/getElementById\("([^"]+)"\)/g)]
  .map((match) => match[1]);
const output = await mkdtemp(path.join(tmpdir(), "deepbom-frontier-viewer-"));
initDeepBom({ module: readFileSync(path.join(ROOT, "web", "protected", "deepbom", "pkg", "deepbom_wasm_bg.wasm")) });
const onnxAnalysis = analyzeOnnxModel(new Uint8Array(readFileSync(ONNX_MODEL)), path.basename(ONNX_MODEL));
const protectedOnnx = analyze_deepbom(new Uint8Array(readFileSync(ONNX_MODEL)), JSON.stringify(onnxAnalysis));
applyProtectedOrtCompatibilityEvidence(onnxAnalysis, protectedOnnx.ort_compatibility_evidence);
const decisionCoverageReport = buildEngineeringReport(onnxAnalysis, {
  identity: {
    filename: onnxAnalysis.filename,
    format: "onnx",
    sha256: onnxAnalysis.model_sha256,
    target_label: onnxAnalysis.target_profile?.label || "ONNX static target posture",
  },
  generatedAt: "2026-07-22T00:00:00.000Z",
});
const server = createStaticServer(ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  await page.addInitScript(({ ids }) => {
    for (const id of ids) {
      if (Object.prototype.hasOwnProperty.call(window, id)) continue;
      Object.defineProperty(window, id, {
        configurable: true,
        get() {
          throw new Error(`DOM named-global access is forbidden for ${id}`);
        },
      });
    }
  }, { ids: DOM_BINDING_IDS });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  const targetPlacement = await page.evaluate(() => ({
    insideInput: Boolean(document.querySelector("#dropzone > #targetSwitcherBar")),
    insideResults: Boolean(document.querySelector("#perfVisuals #targetSwitcherBar")),
    targetCount: document.querySelectorAll("#targetSwitcherBar .target-pill").length,
    hidden: document.querySelector("#targetSwitcherBar")?.hidden,
  }));
  if (!targetPlacement.insideInput || targetPlacement.insideResults || targetPlacement.targetCount !== 0 || !targetPlacement.hidden) {
    throw new Error(`Target conditions must remain unresolved until an artifact format is known: ${JSON.stringify(targetPlacement)}`);
  }
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.setViewportSize({ width: 320, height: 780 });
    const agreementLayout = await page.locator(".agreement-modal").evaluate((modal) => ({
      fitsViewport: modal.getBoundingClientRect().height <= window.innerHeight,
      scrollable: getComputedStyle(modal.querySelector(".agreement-scroll")).overflowY === "auto",
      actionVisible: modal.querySelector("#acceptAgreement").getBoundingClientRect().bottom <= window.innerHeight,
    }));
    if (!agreementLayout.fitsViewport || !agreementLayout.scrollable || !agreementLayout.actionVisible) {
      throw new Error(`Privacy agreement is not usable at 320px: ${JSON.stringify(agreementLayout)}`);
    }
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
    await page.setViewportSize({ width: 1440, height: 1000 });
  }
  await page.locator("#fileInput").setInputFiles({
    name: LONG_MODEL_NAME,
    mimeType: "application/octet-stream",
    buffer: readFileSync(MODEL),
  });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.waitForFunction(() => document.querySelectorAll("#targetSwitcherBar .target-pill").length >= 7, null, { timeout: 30_000 });
  const resolvedTargets = await page.locator("#targetSwitcherBar").evaluate((bar) => ({
    hidden: bar.hidden,
    targetCount: bar.querySelectorAll(".target-pill").length,
    ariaLabel: bar.getAttribute("aria-label"),
    label: bar.querySelector(".target-switcher-label")?.textContent?.trim() || "",
  }));
  const stagedContext = await page.locator("#artifactContextCopy").textContent();
  if (resolvedTargets.hidden
    || resolvedTargets.targetCount < 7
    || resolvedTargets.ariaLabel !== "TFLite CPU cost profiles"
    || resolvedTargets.label !== "CPU cost profile:"
    || !stagedContext.includes("GPU and NNAPI source eligibility is evaluated separately in Accelerator")) {
    throw new Error(`TFLite staging did not resolve the static target conditions: ${JSON.stringify(resolvedTargets)}`);
  }
  await page.locator('[data-target-id="android_mid_a55"]').click();
  const stagedEstimate = await page.locator("#analysisEstimate").textContent();
  if (stagedEstimate.includes("<1 sec") || !stagedEstimate.includes("sec")) {
    throw new Error(`TFLite full-audit estimate is not realistic: ${stagedEstimate}`);
  }
  await page.evaluate(() => {
    globalThis.__auditProgressStates = [];
    const progress = document.querySelector("#auditProgress");
    const capture = () => globalThis.__auditProgressStates.push({
      value: Number(progress?.getAttribute("aria-valuenow") || 0),
      phase: progress?.getAttribute("aria-valuetext") || "",
    });
    new MutationObserver(capture).observe(progress, { attributes: true, attributeFilter: ["aria-valuenow", "aria-valuetext"] });
    capture();
  });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-audit-tab="accelerator"]').click();
  await page.waitForFunction(() => {
    const panel = document.querySelector("#executionPlacementPanel");
    return panel && !panel.hidden && panel.getClientRects().length > 0;
  }, null, { timeout: 30_000 });
  const acceleratorSurface = await page.locator("#executionPlacementPanel").evaluate((panel) => ({
    text: panel.textContent || "",
    acceleratorProfiles: [...panel.querySelectorAll('.placement-profile-comparison-card[data-profile-class="accelerator"]')]
      .map((card) => card.dataset.profileId || ""),
    cpuProfiles: [...panel.querySelectorAll('.placement-profile-comparison-card[data-profile-class="cpu"]')]
      .map((card) => card.dataset.profileId || ""),
    detailProfile: panel.querySelector(".placement-profile-detail-select select")?.value || "",
    loadSourceAction: [...panel.querySelectorAll("button")].some((button) => button.textContent?.includes("Load GPU / NNAPI source ledger")),
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  const sourceLedgerLoaded = acceleratorSurface.acceleratorProfiles.includes("tflite_gpu");
  const expectedDetailProfile = sourceLedgerLoaded ? "tflite_gpu" : "litert_qualcomm_qnn";
  if (acceleratorSurface.detailProfile !== expectedDetailProfile
    || !acceleratorSurface.acceleratorProfiles.includes(expectedDetailProfile)
    || !acceleratorSurface.cpuProfiles.includes("xnnpack_cpu")
    || !acceleratorSurface.text.includes("N-way execution-path comparison")
    || sourceLedgerLoaded && acceleratorSurface.loadSourceAction
    || !sourceLedgerLoaded && (!acceleratorSurface.loadSourceAction
      || !acceleratorSurface.text.includes("GPU/NNAPI profiles are not loaded in this run"))
    || !acceleratorSurface.text.includes("GPU roofline")
    || acceleratorSurface.overflow > 1) {
    throw new Error(`Accelerator evidence is not integrated into the audited web surface: ${JSON.stringify(acceleratorSurface)}`);
  }
  const acceleratorDesktopPath = path.join(output, "accelerator-evidence-desktop.png");
  await page.locator("#executionPlacementPanel").screenshot({ path: acceleratorDesktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  const acceleratorMobile = await page.locator("#executionPlacementPanel").evaluate((panel) => ({
    documentOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    backendSelectHeight: panel.querySelector(".placement-profile-detail-select select")?.getBoundingClientRect().height || 0,
    topologyColumns: getComputedStyle(panel.querySelector(".execution-placement-levels")).gridTemplateColumns.split(" ").length,
  }));
  if (acceleratorMobile.documentOverflow > 1
    || acceleratorMobile.panelOverflow > 1
    || acceleratorMobile.backendSelectHeight < 43.5
    || acceleratorMobile.topologyColumns !== 1) {
    throw new Error(`Accelerator evidence mobile layout is invalid: ${JSON.stringify(acceleratorMobile)}`);
  }
  const acceleratorMobilePath = path.join(output, "accelerator-evidence-mobile.png");
  await page.locator("#executionPlacementPanel").screenshot({ path: acceleratorMobilePath });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('[data-audit-tab="overview"]').click();
  const overview = await page.evaluate(() => {
    const cards = [...document.querySelectorAll("#summary .metric")];
    const byLabel = Object.fromEntries(cards.map((card) => [
      card.querySelector("span")?.textContent || "",
      {
        value: card.querySelector("strong")?.textContent || "",
        detail: card.querySelector(".metric-detail")?.textContent || "",
      },
    ]));
    const targetByLabel = Object.fromEntries(
      [...document.querySelectorAll("#targetConditionSummary .target-condition-card")].map((card) => [
        card.querySelector("span")?.textContent || "",
        {
          value: card.querySelector("strong")?.textContent || "",
          detail: card.querySelector("small")?.textContent || "",
        },
      ]),
    );
    const artifactPanels = Object.fromEntries(
      [...document.querySelectorAll("#summary [data-artifact-panel]")].map((panel) => [
        panel.dataset.artifactPanel,
        {
          text: panel.textContent || "",
          overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
          bars: panel.querySelectorAll(".artifact-bar-row").length,
        },
      ]),
    );
    return {
      byLabel,
      targetByLabel,
      artifactPanels,
      overviewHeading: document.querySelector(".artifact-overview-head h2")?.textContent || "",
      fullAuditLabel: document.querySelector("#modelPlan > div:nth-child(2) > span")?.textContent || "",
      fullAuditValue: document.querySelector("#analysisEstimate")?.textContent || "",
      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    };
  });
  if (overview.fullAuditLabel !== "Analysis Time"
    || !overview.fullAuditValue.includes("measured")
    || overview.overviewHeading !== "Serialized model identity and workload"
    || ["CPU cores", "Architecture", "ISA / NEON", "Cache conditions", "Roofline ridge", "Audit processing time"].some((label) => label in overview.byLabel)
    || !overview.targetByLabel["CPU / Architecture"]?.value.includes("AArch64")
    || overview.targetByLabel.ISA?.value !== "NEON 128-bit"
    || !overview.targetByLabel["Cache Conditions"]?.value.includes("32 KiB L1D")
    || overview.targetByLabel["Roofline Parameters"]?.value !== "300 GOPS / 12.5 GB/s"
    || !overview.targetByLabel["Roofline Parameters"]?.detail.includes("Low/high thresholds 6 / 24 ops/B")
    || !overview.targetByLabel["Roofline Parameters"]?.detail.includes("compute utilization factor 1")
    || !overview.targetByLabel["Roofline Parameters"]?.detail.includes("uncalibrated")
    || overview.byLabel["Operators (layer count)"]?.value !== "65"
    || !overview.byLabel.Tensors?.detail.includes("Kernel 53 (UINT8 53)")
    || !overview.byLabel.Tensors?.detail.includes("Activation 66 (UINT8 66)")
    || !overview.byLabel["Quantized tensors"]?.detail.includes("Bias 53 (INT32 53)")
    || overview.byLabel["Per-channel"]?.value !== "0"
    || overview.byLabel["Per-tensor"]?.value !== "172"
    || !overview.byLabel["Per-tensor"]?.detail.includes("Activation 66 (UINT8 66)")
    || !overview.byLabel.Format?.detail.includes("converter version not embedded/determinable")
    || !overview.byLabel["Artifact schema evidence"]?.value.includes("FlatBuffer schema v3")
    || !overview.byLabel.Model?.detail.includes("SHA-256")
    || overview.artifactPanels["operator-composition"]?.bars < 4
    || !overview.artifactPanels["operator-composition"]?.text.includes("CONV_2D")
    || !overview.artifactPanels["model-interface"]?.text.includes("UINT8")
    || !overview.artifactPanels["model-interface"]?.text.includes("scale")
    || !overview.artifactPanels["model-interface"]?.text.includes("zero-point")
    || overview.artifactPanels["largest-kernels"]?.bars !== 5
    || !overview.artifactPanels["scale-spread"]?.text.includes("No compute kernel carries multiple quantization scales")
    || !overview.artifactPanels["graph-topology"]?.text.includes("ADD graph merges")
    || !overview.artifactPanels["metadata-signatures"]?.text.includes("SignatureDefs")
    || Object.values(overview.artifactPanels).some((panel) => panel.text.includes("NaN"))
    || Object.values(overview.artifactPanels).some((panel) => panel.overflow > 1)
    || overview.overflow > 1) {
    throw new Error(`Artifact overview or target dashboard evidence is incomplete or overflowing: ${JSON.stringify(overview)}`);
  }
  const prioritizedActions = await page.locator("#recommendationList .action-item").allTextContents();
  if (!prioritizedActions[0]?.includes("Weight packing watchlist")
    || !prioritizedActions[0]?.includes("Axis: cold-start only")
    || !prioritizedActions[0]?.includes("Cold-start component")
    || !prioritizedActions[1]?.includes("XNNPACK predicted partition breaks")
    || !prioritizedActions[1]?.includes("Axis: cold-start setup only")
    || !prioritizedActions[1]?.includes("of cold total")
    || prioritizedActions.some((text) => /quant numerical contract/i.test(text) && /Modeled exposure/i.test(text))) {
    throw new Error(`Priority Actions do not follow modeled impact or decision-axis boundaries: ${JSON.stringify(prioritizedActions)}`);
  }
  const artifactOverviewDesktopPath = path.join(output, "artifact-overview-desktop.png");
  await page.locator("#summary").screenshot({ path: artifactOverviewDesktopPath });
  const inputContractOverview = await page.evaluate(() => {
    const panel = document.querySelector('[data-artifact-panel="model-interface"]');
    return { text: panel?.textContent || "", overflow: panel ? Math.max(0, panel.scrollWidth - panel.clientWidth) : -1 };
  });
  if (!inputContractOverview.text.includes("NHWC (DERIVED)") || inputContractOverview.overflow > 1) {
    throw new Error(`Artifact interface panel does not expose the source-derived layout without overflow: ${JSON.stringify(inputContractOverview)}`);
  }
  const readinessText = await page.locator("#readinessLabel").textContent();
  const readinessRationaleText = await page.locator("#readinessRationale").textContent();
  if (!readinessText?.includes("11 exact-zero stored kernel channel(s)")
    || !readinessRationaleText?.includes("stored centered-code space")) {
    throw new Error(`Triage headline does not surface the scheme-independent exact-zero kernel evidence: ${JSON.stringify({ readinessText, readinessRationaleText })}`);
  }
  const glance = await page.locator("#modelGlancePanel").evaluate((panel) => ({
    schema: panel.dataset.glanceSchema,
    l1WatchCount: Number(panel.dataset.l1WatchCount),
    maxL1Ratio: Number(panel.dataset.maxL1Ratio),
    cacheState: panel.dataset.cacheState,
    latencyConservation: panel.dataset.latencyConservation,
    hotspotRows: panel.querySelectorAll(".glance-hotspot-row").length,
    penaltyRows: panel.querySelectorAll(".glance-penalty-row").length,
    ledgerLabels: [...panel.querySelectorAll(".glance-ledger-label")].map((node) => node.textContent || ""),
    text: panel.textContent || "",
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  if (glance.schema !== "deepbom.model_at_a_glance.v1.3"
    || glance.l1WatchCount !== 4
    || Math.abs(glance.maxL1Ratio - 1.1484375) > 1e-12
    || glance.cacheState !== "profile_bound"
    || glance.latencyConservation !== "conserved"
    || glance.hotspotRows !== 6
    || glance.penaltyRows < 1
    || !glance.text.includes("Target At A Glance")
    || glance.ledgerLabels.includes("Artifact")
    || glance.ledgerLabels.includes("Quantization")
    || !glance.text.includes("High 34 / mixed 14 / low 17")
    || !glance.text.includes("Steady-state modeled composition")
    || !(glance.text.includes("Cold start") || glance.text.includes("Cold range"))
    || !glance.text.includes("Conservation: conserved")
    || glance.overflow > 1) {
    throw new Error(`Model At A Glance is incomplete or inconsistent: ${JSON.stringify(glance)}`);
  }
  const modelGlanceDesktopPath = path.join(output, "model-at-a-glance-desktop.png");
  await page.locator("#modelGlancePanel").screenshot({ path: modelGlanceDesktopPath });
  await page.locator('#modelGlancePanel [data-glance-view="cache"]').click();
  const boundWatchCount = Number(await page.locator("#modelGlancePanel").getAttribute("data-l1-watch-count"));
  await page.locator('#modelGlancePanel [data-cache-bytes="16384"]').click();
  const cacheWhatIf = await page.locator("#modelGlancePanel").evaluate((panel) => ({
    state: panel.dataset.cacheState,
    watchCount: Number(panel.dataset.l1WatchCount),
    text: panel.textContent || "",
    rows: panel.querySelectorAll(".glance-cache-row").length,
  }));
  if (cacheWhatIf.state !== "unbound_what_if"
    || cacheWhatIf.watchCount <= boundWatchCount
    || cacheWhatIf.rows !== 12
    || !cacheWhatIf.text.includes("Viewer-only denominator")
    || !cacheWhatIf.text.includes("1.00x L1D")) {
    throw new Error(`Cache what-if did not update deterministically: ${JSON.stringify(cacheWhatIf)}`);
  }
  await page.locator('#modelGlancePanel [data-glance-view="actions"]').click();
  const actionState = await page.locator("#modelGlancePanel").evaluate((panel) => ({
    cards: panel.querySelectorAll(".glance-action-card").length,
    text: panel.textContent || "",
  }));
  if (actionState.cards < 2
    || !actionState.text.includes("Repair CPU island 1")
    || !actionState.text.includes("2 boundaries / 62.5 KiB removable")
    || !actionState.text.includes("DERIVED BOUNDARY COUNTERFACTUAL")) {
    throw new Error(`Decision action cards are incomplete: ${JSON.stringify(actionState)}`);
  }
  await page.locator("#modelGlancePanel button", { hasText: "Preview in graph" }).click();
  await page.locator("#graphScenarioBanner").waitFor({ state: "visible" });
  await page.locator("#graphExplorer").scrollIntoViewIfNeeded();
  const graphScenario = await page.evaluate(() => ({
    label: document.querySelector("#graphScenarioLabel")?.textContent || "",
    detail: document.querySelector("#graphScenarioDetail")?.textContent || "",
    resolvedIslands: document.querySelectorAll("#graphMapSvg .graph-cpu-island.scenario-resolved").length,
    scenarioNodes: document.querySelectorAll("#graphMapSvg .graph-node.scenario-delegated").length,
    removedEdges: document.querySelectorAll("#graphMapSvg .graph-edge--scenario-removed").length,
    removedLabels: [...document.querySelectorAll("#graphMapSvg .scenario-removed-label")].map((node) => node.textContent),
    renderedNodeWidth: document.querySelector("#graphMapSvg .graph-node.scenario-delegated .graph-node-card")?.getBoundingClientRect().width || 0,
    graphNodeCount: document.querySelectorAll("#graphMapSvg .graph-node").length,
    scenarioNodeVisible: (() => {
      const node = document.querySelector("#graphMapSvg .graph-node.scenario-delegated .graph-node-card")?.getBoundingClientRect();
      const svg = document.querySelector("#graphMapSvg")?.getBoundingClientRect();
      return Boolean(node && svg && node.right > svg.left && node.left < svg.right && node.bottom > svg.top && node.top < svg.bottom);
    })(),
    visualDiagnostics: (() => {
      const card = document.querySelector("#graphMapSvg .graph-node.scenario-delegated .graph-node-card");
      const node = card?.getBoundingClientRect();
      const svgNode = document.querySelector("#graphMapSvg");
      const svg = svgNode?.getBoundingClientRect();
      const style = card ? getComputedStyle(card) : null;
      const top = node ? document.elementFromPoint(node.left + node.width / 2, node.top + node.height / 2) : null;
      return {
        viewBox: svgNode?.getAttribute("viewBox") || "",
        svg: svg ? { left: svg.left, top: svg.top, width: svg.width, height: svg.height } : null,
        node: node ? { left: node.left, top: node.top, width: node.width, height: node.height } : null,
        fill: style?.fill || "",
        stroke: style?.stroke || "",
        opacity: style?.opacity || "",
        topElement: top ? `${top.tagName}.${top.getAttribute("class") || ""}` : "",
      };
    })(),
  }));
  if (!graphScenario.label.includes("CPU island 1")
    || !graphScenario.detail.includes("62.5 KiB")
    || graphScenario.resolvedIslands !== 1
    || graphScenario.scenarioNodes !== 1
    || graphScenario.removedEdges !== 2
    || graphScenario.renderedNodeWidth < 70
    || graphScenario.graphNodeCount < 1
    || !graphScenario.scenarioNodeVisible
    || graphScenario.visualDiagnostics.topElement !== "rect.graph-node-card"
    || !graphScenario.removedLabels.every((label) => label.includes("removed"))) {
    throw new Error(`Graph repair preview is incomplete: ${JSON.stringify(graphScenario)}`);
  }
  const graphScenarioDesktopPath = path.join(output, "graph-repair-scenario-desktop.png");
  await page.locator("#graphExplorer").screenshot({ path: graphScenarioDesktopPath });
  await page.locator("#graphDepth").selectOption("all");
  await page.waitForFunction(() => document.querySelector("#graphMapStatus")?.textContent.includes("full graph"));
  const fullGraphState = await page.evaluate(() => ({
    nodeCount: document.querySelectorAll("#graphMapSvg.full-graph .graph-node").length,
    renderedNodeWidth: document.querySelector("#graphMapSvg.full-graph .graph-node-card")?.getBoundingClientRect().width || 0,
    distinctRows: new Set([...document.querySelectorAll("#graphMapSvg.full-graph .graph-node")]
      .map((node) => node.getAttribute("transform")?.match(/,\s*([0-9.]+)\)/)?.[1])
      .filter(Boolean)).size,
    boundaryLabels: document.querySelectorAll("#graphMapSvg.full-graph .boundary-label").length,
  }));
  if (fullGraphState.nodeCount !== 65
    || fullGraphState.renderedNodeWidth < 60
    || fullGraphState.distinctRows < 8
    || fullGraphState.boundaryLabels !== 2) {
    throw new Error(`Folded full graph is not readable or complete: ${JSON.stringify(fullGraphState)}`);
  }
  const graphFullDesktopPath = path.join(output, "graph-full-scenario-desktop.png");
  await page.locator(".graph-map-panel").screenshot({ path: graphFullDesktopPath });
  await page.locator("#graphDepth").selectOption("2");
  await page.locator("#clearGraphScenario").click();
  if (await page.locator("#graphScenarioBanner").isVisible()) throw new Error("Graph scenario did not clear.");
  await page.locator('[data-workflow-step="audit"]').click();
  await page.locator("#modelGlancePanel").waitFor({ state: "visible" });
  const auditProgressState = await page.evaluate(() => ({
    hidden: document.querySelector("#auditProgress")?.hidden,
    state: document.querySelector("#auditProgress")?.dataset.state,
    value: Number(document.querySelector("#auditProgress")?.getAttribute("aria-valuenow") || 0),
    label: document.querySelector("#auditProgressLabel")?.textContent || "",
    estimate: document.querySelector("#analysisEstimate")?.textContent || "",
    phases: globalThis.__auditProgressStates || [],
    timingRecords: JSON.parse(localStorage.getItem("deepbom-static-audit-timing-v1") || "[]"),
  }));
  for (const phase of ["Building 4-target frontier", "Evaluating delegation repair", "Rendering audit overview", "Rendering numerical labs", "Rendering graph explorer", "Complete"]) {
    if (!auditProgressState.phases.some((entry) => entry.phase.includes(phase))) {
      throw new Error(`Audit progress did not expose phase '${phase}': ${JSON.stringify(auditProgressState)}`);
    }
  }
  const coreProgressValues = [...new Set(auditProgressState.phases
    .filter((entry) => entry.phase.includes("Decoding FlatBuffer graph"))
    .map((entry) => entry.value))];
  const progressValues = auditProgressState.phases.map((entry) => entry.value);
  if (coreProgressValues.length < 5
    || auditProgressState.phases.filter((entry) => /Step 03\/12.*\d+\.\d+s/.test(entry.phase)).length < 5
    || progressValues.some((value, index) => index > 0 && value < progressValues[index - 1])) {
    throw new Error(`Audit progress did not remain responsive and monotonic during isolated core analysis: ${JSON.stringify(auditProgressState.phases)}`);
  }
  if (auditProgressState.hidden || auditProgressState.state !== "complete" || auditProgressState.value !== 100
    || !/^Step 12\/12 · 100% · Complete · \d+\.\d+s total$/.test(auditProgressState.label)
    || !auditProgressState.estimate.includes("measured")) {
    throw new Error(`Audit progress completion is invalid: ${JSON.stringify(auditProgressState)}`);
  }
  const timingRecord = auditProgressState.timingRecords.at(-1);
  if (timingRecord?.format !== "TFLITE" || timingRecord?.sizeBytes !== 3577760
    || timingRecord?.comparisonTargetCount !== 4 || !(timingRecord?.durationMs >= 50)) {
    throw new Error(`Full-audit timing was not retained as privacy-minimized local calibration: ${JSON.stringify(timingRecord)}`);
  }
  const auditProgressDesktopPath = path.join(output, "audit-progress-desktop.png");
  await page.locator("#modelPlan").screenshot({ path: auditProgressDesktopPath });
  await page.locator("#deploymentFrontierPanel").waitFor({ state: "visible" });
  const longContentDesktop = await page.evaluate(() => {
    const name = document.querySelector("#selectedModelName");
    const subtitle = document.querySelector("#insightSubtitle");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      nameTitle: name?.title || "",
      nameText: name?.textContent || "",
      subtitleWidth: subtitle?.getBoundingClientRect().width || 0,
      subtitleParentWidth: subtitle?.parentElement?.getBoundingClientRect().width || 0,
    };
  });
  if (longContentDesktop.bodyOverflow > 1
    || longContentDesktop.nameTitle !== longContentDesktop.nameText
    || longContentDesktop.subtitleWidth > longContentDesktop.subtitleParentWidth + 1) {
    throw new Error(`Long artifact identity is not contained on desktop: ${JSON.stringify(longContentDesktop)}`);
  }

  const frontier = await panelState(page);
  if (!frontier.status.includes("4 targets / HEURISTIC COST MODEL")
    || !frontier.text.includes("Steady-state robust set")
    || !frontier.text.includes("Largest steady lever")
    || !frontier.text.includes("Cold-start packing lever")
    || !frontier.text.includes("Cold-start setup lever")
    || !frontier.text.includes("maximum across profiles")
    || !frontier.text.includes("unmeasured profile constant")
    || frontier.rows !== 12
    || frontier.cacheRows !== 4
    || !frontier.text.includes("1.15x / 0.29x")
    || !/Show all \d+ robust hotspots/.test(frontier.text)
    || frontier.targetColumns !== 6) {
    throw new Error(`Frontier view is incomplete: ${JSON.stringify(frontier)}`);
  }
  const comparisonCards = await page.locator("#targetCompareGrid .target-card").evaluateAll((cards) => cards.map((card) => {
    const metrics = [...card.querySelectorAll(".target-metrics > div")].map((row) => ({
      label: row.querySelector("span")?.textContent || "",
      value: row.querySelector("strong")?.textContent || "",
      title: row.title || "",
    }));
    return {
      target: card.querySelector("strong")?.textContent || "",
      steadyCold: metrics.find((metric) => metric.label === "Steady / cold") || null,
    };
  }));
  if (comparisonCards.length !== 4 || comparisonCards.some((card) => (
    !card.steadyCold
    || /\/\s*0(?:\.0+)?\s*(?:us|ms)$/i.test(card.steadyCold.value)
    || !card.steadyCold.title.includes("Steady range")
    || !card.steadyCold.title.includes("cold range")
  ))) {
    throw new Error(`Target Comparison steady/cold binding regressed: ${JSON.stringify(comparisonCards)}`);
  }
  const desktopPath = path.join(output, "deployment-frontier-desktop.png");
  await page.locator("[data-frontier-expand]").click();
  const expandedRows = await page.locator("#deploymentFrontierPanel .frontier-matrix:not(.frontier-cache-table) tbody tr").count();
  if (expandedRows <= 12 || !frontier.text.includes(`${expandedRows} / 65`)) throw new Error(`Expanded frontier should render the complete declared robust set, found ${expandedRows}.`);
  await page.locator("[data-frontier-expand]").click();

  await page.locator('[data-frontier-view="levers"]').click();
  const levers = await page.locator("#deploymentFrontierPanel").evaluate((panel) => ({
    leverCount: panel.querySelectorAll(".frontier-lever-row").length,
    evidenceCount: panel.querySelectorAll(".frontier-evidence-row").length,
    text: panel.textContent || "",
  }));
  if (levers.leverCount !== 3 || levers.evidenceCount !== 4
    || !levers.text.includes("Prepacked weights retained")
    || !levers.text.includes("ESTIMATED_COUNTERFACTUAL_UPPER_BOUND")) {
    throw new Error(`Frontier levers are incomplete: ${JSON.stringify(levers)}`);
  }

  await page.locator('[data-frontier-view="divergence"]').click();
  const divergence = await page.locator("#deploymentFrontierPanel").evaluate((panel) => ({
    pairCount: panel.querySelectorAll(".frontier-pair-selector button").length,
    driverRows: panel.querySelectorAll(".frontier-driver-table tbody tr").length,
    activePair: panel.querySelector(".frontier-pair-selector button.active")?.textContent || "",
    text: panel.textContent || "",
  }));
  if (divergence.pairCount !== 6
    || divergence.driverRows !== 12
    || !divergence.activePair.includes("x86 / AVX2 / WASM SIMD")
    || !divergence.text.includes("Attribution conservation")
    || !divergence.text.includes("80% explanation")) {
    throw new Error(`Frontier divergence is incomplete: ${JSON.stringify(divergence)}`);
  }
  await page.locator("#deploymentFrontierPanel .frontier-pair-selector button").first().click();
  const switchedPair = await page.locator("#deploymentFrontierPanel .frontier-driver-context").textContent();
  if (!switchedPair.includes("Cortex-A55 (L1D 32 KiB default) / RPi4 / Cortex-A72")
    || !/normalized JSD 0\.\d{4}/.test(switchedPair)
    || !switchedPair.includes("80% explanation")) throw new Error(`Frontier pair selection did not update attribution context: ${switchedPair}`);
  await page.locator("#deploymentFrontierPanel .frontier-pair-selector button", { hasText: "x86 / AVX2 / WASM SIMD" }).click();
  const divergenceDesktopPath = path.join(output, "deployment-frontier-divergence-desktop.png");
  await page.evaluate(() => {
    const clone = document.querySelector("#deploymentFrontierPanel").cloneNode(true);
    clone.id = "frontierDivergenceFixture";
    document.body.prepend(clone);
  });
  await page.locator("#frontierDivergenceFixture").screenshot({ path: divergenceDesktopPath });
  await page.locator("#frontierDivergenceFixture").evaluate((node) => node.remove());

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#modelGlancePanel [data-glance-view="summary"]').click();
  const artifactOverviewMobilePath = path.join(output, "artifact-overview-mobile.png");
  await page.locator("#summary").screenshot({ path: artifactOverviewMobilePath });
  const glanceMobile = await page.locator("#modelGlancePanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    panelWidth: panel.clientWidth,
    panelScrollWidth: panel.scrollWidth,
    summaryColumns: getComputedStyle(panel.querySelector(".glance-summary-layout")).gridTemplateColumns.split(" ").length,
    clippedButtons: [...panel.querySelectorAll("button")].filter((button) => button.scrollWidth > button.clientWidth + 1).length,
    widestChildren: [...panel.querySelectorAll("*")]
      .map((node) => ({
        selector: `${node.tagName.toLowerCase()}.${[...node.classList].join(".")}`,
        clientWidth: node.clientWidth,
        scrollWidth: node.scrollWidth,
      }))
      .filter((entry) => entry.scrollWidth > entry.clientWidth + 1)
      .sort((a, b) => (b.scrollWidth - b.clientWidth) - (a.scrollWidth - a.clientWidth))
      .slice(0, 5),
  }));
  if (glanceMobile.bodyOverflow > 1 || glanceMobile.panelOverflow > 1 || glanceMobile.summaryColumns !== 1 || glanceMobile.clippedButtons > 0) {
    throw new Error(`Model At A Glance mobile layout is invalid: ${JSON.stringify(glanceMobile)}`);
  }
  const modelGlanceMobilePath = path.join(output, "model-at-a-glance-mobile.png");
  await page.locator("#modelGlancePanel").screenshot({ path: modelGlanceMobilePath });
  const auditProgressMobilePath = path.join(output, "audit-progress-mobile.png");
  await page.locator("#modelPlan").screenshot({ path: auditProgressMobilePath });
  const auditProgressMobile = await page.locator("#modelPlan").evaluate((plan) => {
    const progress = plan.querySelector("#auditProgress");
    const track = progress?.querySelector(".audit-progress-track");
    const label = plan.querySelector("#auditProgressLabel");
    return {
      columns: getComputedStyle(plan).gridTemplateColumns.split(" ").length,
      progressWidth: progress?.clientWidth || 0,
      trackWidth: track?.clientWidth || 0,
      labelClipped: (label?.scrollWidth || 0) > (label?.clientWidth || 0) + 1,
    };
  });
  if (auditProgressMobile.columns !== 1 || auditProgressMobile.progressWidth < 300
    || auditProgressMobile.trackWidth < 180
    || auditProgressMobile.trackWidth > auditProgressMobile.progressWidth
    || auditProgressMobile.labelClipped) {
    throw new Error(`Audit progress mobile layout is invalid: ${JSON.stringify(auditProgressMobile)}`);
  }
  await page.locator("#deploymentFrontierPanel").scrollIntoViewIfNeeded();
  const divergenceMobile = await page.locator("#deploymentFrontierPanel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    contextColumns: getComputedStyle(panel.querySelector(".frontier-driver-context")).gridTemplateColumns.split(" ").length,
    pairScrollable: panel.querySelector(".frontier-pair-selector").scrollWidth > panel.querySelector(".frontier-pair-selector").clientWidth,
    tableScrollable: panel.querySelector(".frontier-table-wrap").scrollWidth > panel.querySelector(".frontier-table-wrap").clientWidth,
  }));
  if (divergenceMobile.bodyOverflow > 1 || divergenceMobile.panelOverflow > 1 || divergenceMobile.contextColumns !== 2 || !divergenceMobile.pairScrollable || !divergenceMobile.tableScrollable) {
    throw new Error(`Frontier divergence mobile layout is invalid: ${JSON.stringify(divergenceMobile)}`);
  }
  const divergenceMobilePath = path.join(output, "deployment-frontier-divergence-mobile.png");
  await page.evaluate(() => {
    const clone = document.querySelector("#deploymentFrontierPanel").cloneNode(true);
    clone.id = "frontierDivergenceMobileFixture";
    document.body.prepend(clone);
  });
  await page.locator("#frontierDivergenceMobileFixture").screenshot({ path: divergenceMobilePath });
  await page.locator("#frontierDivergenceMobileFixture").evaluate((node) => node.remove());
  await page.locator('[data-frontier-view="frontier"]').click();
  const mobile = await page.evaluate(() => {
    const panel = document.querySelector("#deploymentFrontierPanel");
    const summary = document.querySelector("#deploymentFrontierSummary");
    const tableWrap = panel?.querySelector(".frontier-table-wrap");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, (panel?.scrollWidth || 0) - (panel?.clientWidth || 0)),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      tableScrollable: (tableWrap?.scrollWidth || 0) > (tableWrap?.clientWidth || 0),
    };
  });
  const mobilePath = path.join(output, "deployment-frontier-mobile.png");
  await page.locator("#deploymentFrontierPanel").screenshot({ path: mobilePath });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2 || !mobile.tableScrollable) {
    throw new Error(`Frontier mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }

  await page.locator("#mobileAuditView").selectOption("roofline");
  const rooflineMobile = await page.evaluate(() => {
    const tables = document.querySelector("#tables");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      tablesOverflow: Math.max(0, (tables?.scrollWidth || 0) - (tables?.clientWidth || 0)),
      childrenContained: [...(tables?.children || [])].every((child) => child.getBoundingClientRect().width <= (tables?.clientWidth || 0) + 1),
    };
  });
  if (rooflineMobile.bodyOverflow > 1 || rooflineMobile.tablesOverflow > 1 || !rooflineMobile.childrenContained) {
    throw new Error(`Roofline tables escape the mobile content column: ${JSON.stringify(rooflineMobile)}`);
  }

  await page.locator('[data-workflow-step="output"]').click();
  await page.evaluate(() => {
    for (const panel of document.querySelectorAll("[data-module-panel]")) {
      panel.classList.toggle("active", panel.dataset.modulePanel === "engineering_report");
    }
  });
  await page.locator("#reportPreview").evaluate((node, report) => {
    node.textContent = report;
    node.classList.remove("report-protected");
  }, decisionCoverageReport);
  const reportsMobile = await page.evaluate(() => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    previewOverflow: Math.max(0, document.querySelector("#reportPreview").scrollWidth - document.querySelector("#reportPreview").clientWidth),
    previewScrollable: getComputedStyle(document.querySelector("#reportPreview")).overflowY === "auto"
      && document.querySelector("#reportPreview").scrollHeight > document.querySelector("#reportPreview").clientHeight,
    decisionCoverageVisible: document.querySelector("#reportPreview").textContent.includes("## Decision Coverage At A Glance (DERIVED)"),
    clippedScopeDescriptions: [...document.querySelectorAll("#engineeringBundleScope small")]
      .filter((node) => node.scrollHeight > node.clientHeight + 1).length,
    scopeTextTransform: getComputedStyle(document.querySelector("#engineeringBundleScope small")).textTransform,
    exportButtonsContained: [...document.querySelectorAll(".report-export-panel .export-buttons button")]
      .every((button) => button.scrollWidth <= button.clientWidth + 1),
    exportBadgeWhiteSpace: getComputedStyle(document.querySelector("#downloadCycloneDxEvidence"), "::after").whiteSpace,
    localActionsOverflow: Math.max(0, document.querySelector(".local-reports-actions").scrollWidth
      - document.querySelector(".local-reports-actions").clientWidth),
    localControlsContained: [...document.querySelectorAll(".local-reports-actions > *")]
      .every((control) => control.getBoundingClientRect().right <= document.documentElement.clientWidth + 1),
  }));
  if (reportsMobile.bodyOverflow > 1 || reportsMobile.previewOverflow > 1 || !reportsMobile.previewScrollable || !reportsMobile.decisionCoverageVisible || reportsMobile.clippedScopeDescriptions
    || reportsMobile.scopeTextTransform !== "none" || !reportsMobile.exportButtonsContained
    || reportsMobile.exportBadgeWhiteSpace !== "nowrap" || reportsMobile.localActionsOverflow > 1
    || !reportsMobile.localControlsContained) {
    throw new Error(`Report controls do not preserve long evidence labels on mobile: ${JSON.stringify(reportsMobile)}`);
  }

  await page.locator('[data-target-id="zynq_ultrascale_plus_a53"]').evaluate((button) => button.click());
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const switchedTargetAuditStatus = await page.locator("#status").textContent();
  const zynqTargetActive = await page.locator('[data-target-id="zynq_ultrascale_plus_a53"]').getAttribute("class");
  if (!switchedTargetAuditStatus.includes("audit run complete") || !zynqTargetActive?.includes("active")) {
    throw new Error(`Post-audit target switch did not complete on the requested target: ${JSON.stringify({ switchedTargetAuditStatus, zynqTargetActive })}`);
  }
  const zynqTargetFrontier = await panelState(page);
  await page.locator('[data-workflow-step="audit"]').click();
  await page.locator("#mobileAuditView").selectOption("overview");
  await page.locator('#modelGlancePanel [data-glance-view="cache"]').click();
  const zynqCacheEvidence = await page.locator("#modelGlancePanel").evaluate((panel) => ({
    text: panel.textContent || "",
    l1WatchCount: Number(panel.dataset.l1WatchCount),
    maxL1Ratio: Number(panel.dataset.maxL1Ratio),
    overflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
  }));
  const zynqTargetConditions = await page.evaluate(() => Object.fromEntries(
    [...document.querySelectorAll("#targetConditionSummary .target-condition-card")].map((card) => [
      card.querySelector("span")?.textContent || "",
      {
        value: card.querySelector("strong")?.textContent || "",
        detail: card.querySelector("small")?.textContent || "",
      },
    ]),
  ));
  if (!zynqTargetFrontier.status.includes("5 targets / HEURISTIC COST MODEL")
    || zynqTargetFrontier.targetColumns !== 7
    || zynqTargetFrontier.cacheRows !== 5
    || zynqCacheEvidence.l1WatchCount !== 4
    || Math.abs(zynqCacheEvidence.maxL1Ratio - 1.1484375) > 1e-12
    || !zynqCacheEvidence.text.includes("SOURCE_BACKED_PRODUCT")
    || !zynqCacheEvidence.text.includes("DS891 v1.11.1")
    || !zynqCacheEvidence.text.includes("1badf7142690c573987f3eacd788620ff8a8392425f13124f928aaed152265e9")
    || !zynqCacheEvidence.text.includes("52b19d733bdacfbd1cffd108b277bfbc115839aab0a9f5d51f43b6dfa7c33369")
    || !zynqTargetConditions["CPU / Architecture"]?.value.includes("2-4 cores")
    || !zynqTargetConditions["CPU / Architecture"]?.value.includes("Cortex-A53")
    || zynqTargetConditions.ISA?.value !== "NEON 128-bit"
    || zynqTargetConditions["Roofline Parameters"]?.value !== "48 GOPS / 19.2 GB/s"
    || !zynqTargetConditions["Roofline Parameters"]?.detail.includes("Low/high thresholds 0.7 / 2.5 ops/B")
    || zynqTargetConditions["Modeled Execution Scope"]?.value !== "Cortex-A53 APU CPU only"
    || !zynqTargetConditions["Modeled Execution Scope"]?.detail.includes("DPU/NPU accelerators")
    || zynqCacheEvidence.overflow > 1) {
    throw new Error(`Zynq source-bound target conditions are incomplete: ${JSON.stringify({ zynqTargetFrontier, zynqCacheEvidence, zynqTargetConditions })}`);
  }
  const savedTargetId = await page.evaluate(() => localStorage.getItem("ondevice-audit-target-v1"));
  if (savedTargetId !== "zynq_ultrascale_plus_a53") {
    throw new Error(`Post-audit target selection was not persisted: ${savedTargetId}`);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(() => {
    const panel = document.querySelector("#deploymentFrontierPanel");
    document.body.replaceChildren(panel.cloneNode(true));
  });
  await page.locator("#deploymentFrontierPanel").screenshot({ path: desktopPath });
  await page.evaluate(async (analysis) => {
    const { artifactOverviewHeader, artifactOverviewPanels } = await import("./lib/artifact-overview.js");
    const { summaryMetricCards, targetConditionCards } = await import("./lib/audit-ui.js");
    const main = document.createElement("main");
    main.style.maxWidth = "1180px";
    main.style.margin = "0 auto";
    const overview = document.createElement("section");
    overview.id = "onnxArtifactOverviewFixture";
    overview.className = "summary-grid";
    const metrics = document.createElement("div");
    metrics.className = "artifact-metric-grid";
    metrics.append(...summaryMetricCards(analysis));
    const evidence = document.createElement("div");
    evidence.className = "artifact-evidence-grid";
    evidence.append(...artifactOverviewPanels(analysis));
    overview.append(artifactOverviewHeader(analysis), metrics, evidence);
    const target = document.createElement("section");
    target.id = "onnxTargetConditionsFixture";
    target.className = "target-condition-summary";
    target.append(...targetConditionCards(analysis));
    main.append(overview, target);
    document.body.replaceChildren(main);
  }, onnxAnalysis);
  const onnxArtifactOverview = await page.locator("#onnxArtifactOverviewFixture").evaluate((root) => ({
    text: root.textContent || "",
    labels: [...root.querySelectorAll(".metric > span")].map((node) => node.textContent || ""),
    panels: [...root.querySelectorAll("[data-artifact-panel]")].map((panel) => panel.dataset.artifactPanel),
    largestKernelRows: root.querySelectorAll('[data-artifact-panel="largest-kernels"] .artifact-bar-row').length,
    overflow: Math.max(0, root.scrollWidth - root.clientWidth),
  }));
  const onnxTargetConditions = await page.locator("#onnxTargetConditionsFixture").textContent();
  if (!onnxArtifactOverview.text.includes("ONNX Graph Metadata")
    || !onnxArtifactOverview.text.includes("Conv")
    || !onnxArtifactOverview.text.includes("Graph Topology")
    || onnxArtifactOverview.text.includes("SignatureDefs")
    || !onnxArtifactOverview.labels.includes("IR / opset evidence")
    || onnxArtifactOverview.largestKernelRows !== 3
    || onnxArtifactOverview.panels.length !== 6
    || !onnxTargetConditions.includes("Cache Reference Target")
    || !onnxTargetConditions.includes("Throughput Model")
    || !onnxTargetConditions.includes("Not applied")
    || onnxArtifactOverview.overflow > 1) {
    throw new Error(`ONNX artifact overview is incomplete or format-leaking: ${JSON.stringify({ onnxArtifactOverview, onnxTargetConditions })}`);
  }
  await page.setViewportSize({ width: 320, height: 780 });
  const onnxArtifactMobileOverflow = await page.locator("#onnxArtifactOverviewFixture").evaluate((root) => Math.max(0, root.scrollWidth - root.clientWidth));
  if (onnxArtifactMobileOverflow > 1) throw new Error(`ONNX artifact overview overflows at 320px by ${onnxArtifactMobileOverflow}px.`);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(async (analysis) => {
    const { createDeploymentFrontierController } = await import("./lib/deployment-frontier.js");
    const fixture = document.createElement("article");
    fixture.id = "onnxFrontierFixture";
    fixture.className = "perf-panel wide deployment-frontier-panel";
    fixture.innerHTML = `
      <div class="perf-panel-head"><h3>Deployment Frontier Lab</h3><div class="frontier-head-actions"><span data-status></span></div></div>
      <div class="frontier-summary" data-summary></div>
      <div class="frontier-tabs" role="tablist">
        <button type="button" class="active" data-frontier-view="frontier">Frontier</button>
        <button type="button" data-frontier-view="levers">Levers</button>
        <button type="button" data-frontier-view="divergence">Divergence</button>
      </div>
      <div class="frontier-body" data-body></div>`;
    document.body.replaceChildren(fixture);
    const controller = createDeploymentFrontierController({
      root: fixture,
      status: fixture.querySelector("[data-status]"),
      summary: fixture.querySelector("[data-summary]"),
      body: fixture.querySelector("[data-body]"),
      downloadButton: null,
      getContext: () => ({ analysis, runtimeEvidence: null }),
      jumpToGraphOp: () => {},
      onDownload: () => {},
    });
    controller.render(analysis);
  }, onnxAnalysis);
  const onnxFixture = page.locator("#onnxFrontierFixture");
  const onnxFrontier = await onnxFixture.evaluate((panel) => ({
    text: panel.textContent || "",
    rows: panel.querySelectorAll(".frontier-matrix tbody tr").length,
    columns: panel.querySelectorAll(".frontier-matrix thead th").length,
  }));
  const expectedOnnxCandidates = onnxAnalysis.ort_ep_portability_frontier.all_ep_artifact_precheck_candidate_op_count;
  const expectedOnnxProviderCount = onnxAnalysis.ort_ep_portability_frontier.execution_provider_count;
  if (!onnxFrontier.text.includes(`Narrowed common candidates${expectedOnnxCandidates} / ${onnxAnalysis.ops.length}`)
    || !onnxFrontier.text.includes("UNRESOLVED")
    || !onnxFrontier.text.includes("still not support or assignment")
    || !onnxFrontier.text.includes("assignment unobserved")
    || onnxFrontier.rows !== 9
    || onnxFrontier.columns !== expectedOnnxProviderCount + 2) {
    throw new Error(`ONNX frontier matrix is incomplete: ${JSON.stringify(onnxFrontier)}`);
  }
  await onnxFixture.locator('[data-frontier-view="levers"]').click();
  const onnxGaps = await onnxFixture.textContent();
  if (!onnxGaps.toLowerCase().includes("webnn") || !onnxGaps.includes("GlobalAveragePool") || !onnxGaps.includes("SOURCE_SCHEMA_OR_KERNEL_VERSION_GAP") || !onnxGaps.includes("SOURCE_KERNEL_VERSION_NO_MATCH")) {
    throw new Error(`ONNX source gap view is incomplete: ${onnxGaps}`);
  }
  await onnxFixture.locator('[data-frontier-view="divergence"]').click();
  const onnxPairCount = await onnxFixture.locator(".frontier-bar-row").count();
  const expectedOnnxPairCount = expectedOnnxProviderCount * (expectedOnnxProviderCount - 1) / 2;
  if (onnxPairCount !== expectedOnnxPairCount) {
    throw new Error(`ONNX frontier should render ${expectedOnnxPairCount} pairs for ${expectedOnnxProviderCount} EP profiles, found ${onnxPairCount}.`);
  }
  await onnxFixture.locator('[data-frontier-view="frontier"]').click();
  const onnxDesktopPath = path.join(output, "deployment-frontier-onnx-desktop.png");
  await onnxFixture.screenshot({ path: onnxDesktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  const onnxMobile = await onnxFixture.evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    summaryColumns: getComputedStyle(panel.querySelector(".frontier-summary")).gridTemplateColumns.split(" ").length,
    tableScrollable: panel.querySelector(".frontier-table-wrap").scrollWidth > panel.querySelector(".frontier-table-wrap").clientWidth,
  }));
  if (onnxMobile.bodyOverflow > 1 || onnxMobile.panelOverflow > 1 || onnxMobile.summaryColumns !== 2 || !onnxMobile.tableScrollable) {
    throw new Error(`ONNX frontier mobile layout is invalid: ${JSON.stringify(onnxMobile)}`);
  }
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  const restoredTarget = await page.evaluate(() => ({
    saved: localStorage.getItem("ondevice-audit-target-v1"),
    hidden: document.querySelector("#targetSwitcherBar")?.hidden,
    targetCount: document.querySelectorAll("#targetSwitcherBar .target-pill").length,
  }));
  if (restoredTarget.saved !== "zynq_ultrascale_plus_a53" || !restoredTarget.hidden || restoredTarget.targetCount !== 0) {
    throw new Error(`Saved target preference must persist without exposing target conditions before format resolution: ${JSON.stringify(restoredTarget)}`);
  }
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log("Deployment Frontier viewer passed (canonical 4-target and selected 5-target TFLite frontiers; ONNX 9 EP source portfolios; desktop/mobile overflow 0).");
  console.log(`desktop=${desktopPath}`);
  console.log(`artifact_overview_desktop=${artifactOverviewDesktopPath}`);
  console.log(`artifact_overview_mobile=${artifactOverviewMobilePath}`);
  console.log(`model_glance_desktop=${modelGlanceDesktopPath}`);
  console.log(`model_glance_mobile=${modelGlanceMobilePath}`);
  console.log(`graph_scenario_desktop=${graphScenarioDesktopPath}`);
  console.log(`graph_full_desktop=${graphFullDesktopPath}`);
  console.log(`audit_progress_desktop=${auditProgressDesktopPath}`);
  console.log(`audit_progress_mobile=${auditProgressMobilePath}`);
  console.log(`accelerator_desktop=${acceleratorDesktopPath}`);
  console.log(`accelerator_mobile=${acceleratorMobilePath}`);
  console.log(`divergence_desktop=${divergenceDesktopPath}`);
  console.log(`divergence_mobile=${divergenceMobilePath}`);
  console.log(`mobile=${mobilePath}`);
  console.log(`onnx_desktop=${onnxDesktopPath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    frontier: document.querySelector("#deploymentFrontierPanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

async function panelState(browserPage) {
  return browserPage.locator("#deploymentFrontierPanel").evaluate((panel) => ({
    status: panel.querySelector("#deploymentFrontierStatus")?.textContent || "",
    text: panel.textContent || "",
    rows: panel.querySelectorAll(".frontier-matrix:not(.frontier-cache-table) tbody tr").length,
    targetColumns: panel.querySelectorAll(".frontier-matrix:not(.frontier-cache-table) thead th").length,
    cacheRows: panel.querySelectorAll(".frontier-cache-table tbody tr").length,
  }));
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
