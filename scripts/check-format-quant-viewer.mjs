import { createServer } from "node:http";
import { readFile, readFileSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(".");
const SERVE_ROOT = process.env.DEEPBOM_VIEWER_ROOT ? path.resolve(process.env.DEEPBOM_VIEWER_ROOT) : ROOT;
const FIXTURES = [
  ["tflite", "mobilenet_v2_1.0_224_quant.tflite"],
  ["gguf", "tinymqa1m.Q4_0.gguf"],
  ["safetensors", "nanofable-1m-fp16.safetensors"],
  ["coreml", "MNISTClassifier.mlmodel"],
];
const server = createStaticServer(SERVE_ROOT);
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) errors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.locator("#fileInput").focus();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  const bootStatus = await page.locator("#status").textContent();
  if (!bootStatus || bootStatus === "Waiting") {
    throw new Error(`Viewer did not initialize: status=${JSON.stringify(bootStatus)} diagnostics=${JSON.stringify(errors)}`);
  }
  if (await page.locator("#agreementBackdrop").isVisible()) {
    const openAgreement = await page.locator("#agreementBackdrop").evaluate((dialog) => ({ inert: dialog.inert, ariaHidden: dialog.getAttribute("aria-hidden") }));
    if (openAgreement.inert || openAgreement.ariaHidden === "true") throw new Error(`Open privacy dialog is incorrectly hidden from accessibility APIs: ${JSON.stringify(openAgreement)}`);
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  if (await page.locator("#agreementBackdrop").count()) {
    const closedAgreement = await page.locator("#agreementBackdrop").evaluate((dialog) => ({ hidden: dialog.hidden, inert: dialog.inert, ariaHidden: dialog.getAttribute("aria-hidden") }));
    if (!closedAgreement.hidden || !closedAgreement.inert || closedAgreement.ariaHidden !== "true") throw new Error(`Closed privacy dialog remains exposed: ${JSON.stringify(closedAgreement)}`);
  }
  const initial = await page.evaluate(() => ({
    hiddenPanels: ["auditWorkbench", "summary", "outputModuleSelector", "moduleRunConsole", "insightDashboard", "perfVisuals", "findingsPanel", "graphExplorer", "redesignPanel", "inferencePanel"]
      .filter((id) => document.getElementById(id)?.hidden),
    account: (() => {
      const node = document.getElementById("authUser");
      return { hidden: node?.hidden, inert: node?.inert, ariaHidden: node?.getAttribute("aria-hidden") };
    })(),
    claimHidden: document.getElementById("auditClaimBoundary")?.hidden,
    workflowHidden: document.getElementById("workflowConsole")?.hidden,
    targetSwitcherHidden: document.getElementById("targetSwitcherBar")?.hidden,
    sampleLabels: [...document.querySelectorAll("#sampleModelSelect option")].map((option) => option.getAttribute("aria-label") || ""),
    graphLegendLabels: [...document.querySelectorAll(".graph-mode-legend [role='img']")].map((item) => item.getAttribute("aria-label") || ""),
    evidenceCurrent: document.querySelector("[data-evidence-stage] [aria-current='step']")?.closest("[data-evidence-stage]")?.dataset.evidenceStage || "",
  }));
  if (initial.hiddenPanels.length !== 10 || !initial.claimHidden || !initial.workflowHidden || !initial.targetSwitcherHidden
    || !initial.account.hidden || !initial.account.inert || initial.account.ariaHidden !== "true"
    || initial.sampleLabels.length < 8 || initial.sampleLabels.some((label) => !label.includes(","))
    || initial.graphLegendLabels.length !== 7 || initial.graphLegendLabels.some((label) => !label)
    || initial.evidenceCurrent !== "artifact") {
    throw new Error(`Initial progressive-disclosure or accessibility contract failed: ${JSON.stringify(initial)}`);
  }
  await page.locator("#authOpen").click();
  const openAuth = await page.locator("#authBackdrop").evaluate((dialog) => ({ hidden: dialog.hidden, inert: dialog.inert, ariaHidden: dialog.getAttribute("aria-hidden") }));
  if (openAuth.hidden || openAuth.inert || openAuth.ariaHidden === "true") throw new Error(`Open account dialog is inaccessible: ${JSON.stringify(openAuth)}`);
  await page.keyboard.press("Escape");
  const closedAuth = await page.locator("#authBackdrop").evaluate((dialog) => ({ hidden: dialog.hidden, inert: dialog.inert, ariaHidden: dialog.getAttribute("aria-hidden") }));
  if (!closedAuth.hidden || !closedAuth.inert || closedAuth.ariaHidden !== "true") throw new Error(`Closed account dialog remains exposed: ${JSON.stringify(closedAuth)}`);
  if (await page.locator("body").evaluate((body) => body.classList.contains("modal-open"))) {
    throw new Error("Closing the final dialog must release the page scroll lock.");
  }
  const exportedMaps = await page.evaluate(async () => {
    const { visualPngSpecs } = await import("./lib/visual-export.js");
    const analysis = {
      format: "tflite",
      filename: "treemap-contract.tflite",
      ops: [
        { index: 0, name: "CONV_2D", macs: 100, estimated_bytes: 400, quantization_state: "quantized_compute", quant_risk: "ok", static_bound_guess: "compute-bound", stage_index: 0 },
        { index: 1, name: "RESHAPE", macs: 0, estimated_bytes: 120, quantization_state: "quantized_data_movement", quant_risk: "warn", static_bound_guess: "memory-bound", stage_index: 0 },
      ],
      tensors: [],
      block_inventory: { status: "assessed", blocks: [{ block_id: "b0", display_name: "Stem", op_indices: [0, 1] }], stages: [{ stage_id: "s0", index: 0, display_name: "Stem", op_indices: [0, 1] }] },
      target_profile: {},
    };
    const wanted = new Set(["visuals/explorer_resource_map.png", "visuals/quantization_exposure_map.png"]);
    return visualPngSpecs({ analysis, filename: analysis.filename })
      .filter(([name]) => wanted.has(name))
      .map(([name, render]) => {
        const canvas = render();
        const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let opaque = 0;
        for (let index = 3; index < pixels.length; index += 4) if (pixels[index] > 0) opaque += 1;
        return { name, width: canvas.width, height: canvas.height, opaque };
      });
  });
  if (exportedMaps.length !== 2 || exportedMaps.some((row) => row.width < 1180 || Math.abs(row.width / row.height - 1180 / 760) > 1e-6 || row.opaque < row.width * row.height * 0.5)) {
    throw new Error(`Evidence-map PNG export is missing or blank: ${JSON.stringify(exportedMaps)}`);
  }
  for (const [format, filename] of FIXTURES) {
    if (format === "tflite") await runVerifiedExample(page);
    else await runAudit(page, path.join(ROOT, "web", "samples", filename), filename);
    if (await page.locator("html").getAttribute("data-analysis-depth") !== "deep") {
    }
    if (await page.locator("html").getAttribute("data-analysis-depth") !== "deep") {
      throw new Error("Format viewer did not initialize the Full analysis surface before graph and quantization checks.");
    }
    await page.locator('[data-workflow-step="audit"]').click();
    await page.locator('[data-audit-tab="quant"]').click();
    const state = await page.locator("#perfVisuals").evaluate((root) => ({
      title: root.querySelector("#perfVisualTitle")?.textContent || "",
      heading: root.querySelector("#quantRiskHeatmapPanel h3")?.textContent || "",
      count: root.querySelector("#quantHeatmapCount")?.textContent || "",
      tileCount: root.querySelectorAll("#quantHeatmap .quant-tile").length,
      map: root.querySelector("#quantHeatmap")?.textContent || "",
      states: root.querySelector("#quantStateBreakdown")?.textContent || "",
      exposure: root.querySelector("#quantExposureMap")?.textContent || "",
      exposureTiles: root.querySelectorAll("#quantExposureMap .evidence-treemap-tile").length,
      exposureMobileDisplay: getComputedStyle(root.querySelector("#quantExposureMap .evidence-treemap-mobile-list")).display,
      scales: root.querySelector("#quantScaleScatter")?.textContent || "",
      risks: root.querySelector("#quantRiskTable")?.textContent || "",
      holes: `${root.querySelector("#quantHoleCount")?.textContent || ""} ${root.querySelector("#quantHoleList")?.textContent || ""}`,
      hiddenTabs: [...document.querySelectorAll("[data-audit-tab][hidden]")].map((tab) => tab.dataset.auditTab),
      hiddenWorkflow: [...document.querySelectorAll("[data-workflow-step][hidden]")].map((step) => step.dataset.workflowStep),
      notApplicableTabs: [...document.querySelectorAll("[data-audit-tab][data-applicability-status='not_applicable']")].map((tab) => tab.dataset.auditTab),
      notApplicableWorkflow: [...document.querySelectorAll("[data-workflow-step][data-applicability-status='not_applicable']")].map((step) => step.dataset.workflowStep),
      applicabilityReasonsMissing: [...document.querySelectorAll("[data-applicability-status='not_applicable']")]
        .filter((node) => !node.dataset.applicabilityReason).length,
      claimBoundary: document.querySelector("#auditClaimBoundary")?.textContent || "",
      nextProof: document.querySelector("#auditClaimNextProof")?.textContent || "",
      evidenceSpine: document.querySelector("#evidenceSpine")?.textContent || "",
      evidenceRuntimeDisabled: document.querySelector('[data-evidence-stage="runtime"] button')?.disabled ?? true,
      sampleVerification: document.querySelector("#sampleVerificationPanel")?.textContent || "",
      sampleVerificationHidden: document.querySelector("#sampleVerificationPanel")?.hidden ?? true,
      publicPrintDisabled: document.querySelector("#printPublicReport")?.disabled ?? null,
      xnnpackControlsHidden: [...document.querySelectorAll(".format-tflite")].every((node) => getComputedStyle(node).display === "none"),
      auditTabLabels: [...document.querySelectorAll("[data-audit-tab]:not([hidden])")].map((tab) => tab.getAttribute("aria-label") || ""),
      evidenceCurrent: document.querySelector("[data-evidence-stage] [aria-current='step']")?.closest("[data-evidence-stage]")?.dataset.evidenceStage || "",
      placement: document.querySelector("#executionPlacementPanel")?.textContent || "",
      placementFormat: document.querySelector("#executionPlacementPanel")?.dataset.placementFormat || "",
      overflow: Math.max(0, root.scrollWidth - root.clientWidth),
    }));
    validate(format, state);

    if (format === "tflite" && (state.sampleVerificationHidden
      || !state.sampleVerification.includes("passed")
      || !state.sampleVerification.includes("Artifact SHA-256")
      || !state.sampleVerification.includes("Assessed MACs"))) {
      throw new Error(`Verified example result is not reviewable: ${JSON.stringify(state)}`);
    }
    if (!state.evidenceSpine.includes("ArtifactAssessed") || !state.claimBoundary.includes("Next proof") || !state.nextProof
      || !state.evidenceSpine.includes("DerivationDerived")
      || !state.evidenceSpine.includes("Runtime observationNot imported")
      || !state.evidenceSpine.includes("Release decisionNot assessed")) {
      throw new Error(`${format} evidence progression is incomplete: ${JSON.stringify(state)}`);
    }
    if (state.evidenceCurrent !== "derivation" || !state.auditTabLabels.length || state.auditTabLabels.some((label) => !label)) {
      throw new Error(`${format} audit navigation is not exposed with a current evidence stage and unique accessible names: ${JSON.stringify(state)}`);
    }
    if (state.xnnpackControlsHidden !== (format !== "tflite")) {
      throw new Error(`${format} TFLite-only Explorer controls have incorrect format visibility: ${JSON.stringify(state)}`);
    }

    if (["gguf", "safetensors"].includes(format)) {
      await page.locator('[data-audit-tab="llm"]').click();
      const llmState = await page.locator("#perfVisuals").evaluate((root) => ({
        title: root.querySelector("#perfVisualTitle")?.textContent || "",
        panel: root.querySelector("#llmEvidencePanel")?.textContent || "",
        panelHidden: root.querySelector("#llmEvidencePanel")?.hidden ?? true,
        overflow: Math.max(0, root.scrollWidth - root.clientWidth),
      }));
      if (llmState.title !== "On-device LLM Evidence" || llmState.panelHidden || llmState.overflow > 1
        || !llmState.panel.includes("Architecture, state, and deployment contract")
        || !llmState.panel.includes("Runtime evidence still required")
        || !llmState.panel.includes("Medical AI evidence still required")
        || !llmState.panel.includes("Not established by this artifact")) {
        throw new Error(`${format} on-device LLM evidence view is incomplete: ${JSON.stringify(llmState)}`);
      }
    }

    if (format === "tflite") {
      if (state.evidenceRuntimeDisabled) throw new Error("TFLite runtime evidence navigation remained disabled after analysis.");
      await page.locator('[data-evidence-stage="runtime"] button').click();
      if (await page.locator("body").getAttribute("data-workspace") !== "runtime") throw new Error("Evidence Spine did not navigate to Runtime.");
      if (await page.locator("[data-evidence-stage] [aria-current='step']").evaluate((node) => node.closest("[data-evidence-stage]")?.dataset.evidenceStage) !== "runtime") throw new Error("Runtime evidence stage did not become current.");
      await page.locator('[data-evidence-stage="deployment"] button').click();
      if (await page.locator("body").getAttribute("data-workspace") !== "audit") throw new Error("Evidence Spine did not navigate to Deployment evidence.");
      if (await page.locator("[data-evidence-stage] [aria-current='step']").evaluate((node) => node.closest("[data-evidence-stage]")?.dataset.evidenceStage) !== "deployment") throw new Error("Deployment evidence stage did not become current.");
      const xnnpackPlacement = await page.evaluate(() => ({
        title: document.querySelector("#perfVisualTitle")?.textContent || "",
        flow: document.querySelector("#chainFlow")?.textContent || "",
        map: document.querySelector("#xnnpackFallbackCount")?.textContent || "",
        titles: [...document.querySelectorAll("#chainFlow [title]")].map((node) => node.getAttribute("title") || "").join(" "),
      }));
      if (!xnnpackPlacement.title.includes("Predicted Partition Flow")
        || !xnnpackPlacement.flow.includes("Conditional assignment")
        || !xnnpackPlacement.flow.includes("high-exposure break")
        || !xnnpackPlacement.titles.includes("High adjacent-MAC exposure")
        || !xnnpackPlacement.map.toLowerCase().includes("conditionally delegatable under the stated xnnpack build condition")
        || !xnnpackPlacement.map.includes("OPERATOR ANATOMY")
        || /high-compute break|predicted delegated/i.test(`${xnnpackPlacement.flow} ${xnnpackPlacement.map}`)) {
        throw new Error(`TFLite XNNPACK placement terminology or build boundary is inaccurate: ${JSON.stringify(xnnpackPlacement)}`);
      }
      await page.locator('[data-workflow-step="output"]').click();
      if (await page.locator("[data-evidence-stage] [aria-current='step']").evaluate((node) => node.closest("[data-evidence-stage]")?.dataset.evidenceStage) !== "release") throw new Error("Release evidence stage did not become current.");
      const exportCount = await page.locator('[data-module-tab="export_contracts"] em').textContent();
      if (exportCount !== "7 docs + pack") throw new Error(`Machine export count is not bound to the seven generated documents: ${exportCount}`);
      const exportInventory = await page.locator(".export-document-inventory li").allTextContents();
      if (exportInventory.length !== 7 || !exportInventory.some((item) => item.includes("Interface contract ledger"))) {
        throw new Error(`Machine export inventory does not expose the exact seven-document set: ${JSON.stringify(exportInventory)}`);
      }
      await page.locator('[data-module-tab="engineering_report"]').click();
      if (await page.locator("#printPublicReport").isVisible()) {
        throw new Error("Individual report controls should remain secondary until the download menu is expanded.");
      }
      await page.locator(".report-export-panel > summary").click();
      await page.locator("#printPublicReport").waitFor({ state: "visible", timeout: 10_000 });
      await page.locator("#printPublicReport").click();
      await page.waitForTimeout(1_000);
      const popup = page.context().pages().find((candidate) => candidate !== page);
      if (!popup) {
        const status = await page.locator("#status").textContent();
        throw new Error(`Login-free public report did not open: ${status}`);
      }
      await popup.waitForFunction(() => document.body?.textContent?.includes("DEEPBOM PUBLIC STATIC EVIDENCE SUMMARY"), null, { timeout: 20_000 });
      const publicCopy = await popup.evaluate(() => ({
        text: document.body?.textContent || "",
        watermark: getComputedStyle(document.querySelector("main"), "::before").content,
      }));
      if (!publicCopy.watermark.includes("DEEPBOM PUBLIC COPY")
        || !/Report-body SHA-256: [a-f0-9]{64}/i.test(publicCopy.text)
        || !publicCopy.text.includes("TFLite static deployment audit run complete")
        || !publicCopy.text.includes("not the complete Engineering Report or raw evidence ledger")) {
        throw new Error(`Login-free public report is incomplete: ${JSON.stringify(publicCopy).slice(0, 2000)}`);
      }
      await popup.close();
    }

    if (format === "coreml") {
      await page.locator('[data-workflow-step="graph"]').click();
      await page.locator("#nodeViewPanel .nv-node").first().waitFor({ timeout: 20_000 });
      const graph = await page.locator("#nodeViewPanel").evaluate((root) => ({
        nodes: root.querySelectorAll(".nv-node").length,
        edges: root.querySelectorAll(".nv-edge").length,
        text: root.textContent || "",
      }));
      if (graph.nodes !== 14 || graph.edges !== 13 || !graph.text.includes("CONVOLUTION") || !graph.text.includes("Core ML runtime unobserved")) {
        throw new Error(`Core ML serialized graph is incomplete: ${JSON.stringify(graph)}`);
      }
    }
    await page.locator('[data-workflow-step="input"]').click();
  }
  await runMobileVerifiedExample(page);
  if (errors.length) throw new Error(`Browser diagnostics:\n${errors.join("\n")}`);
  console.log(`Format-aware quant viewer passed (${path.relative(ROOT, SERVE_ROOT) || "source"}; TFLite -> GGUF -> SafeTensors -> Core ML stale-state isolation).`);
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

function validate(format, state) {
  if (state.overflow > 1) throw new Error(`${format} quant viewer overflows: ${JSON.stringify(state)}`);
  if (state.hiddenTabs.length || state.hiddenWorkflow.some((step) => step !== "offline_test") || state.applicabilityReasonsMissing) {
    throw new Error(`${format} applicability is hidden or lacks an explicit reason: ${JSON.stringify(state)}`);
  }
  if (!state.exposure.includes("Quantization Exposure Map") || !state.exposure.includes("Conservationexact")
    || state.exposureTiles < 1 || state.exposureMobileDisplay !== "none") {
    throw new Error(`${format} quantization exposure map is missing, empty, or non-conserving: ${JSON.stringify(state)}`);
  }
  if (state.publicPrintDisabled) throw new Error(`${format} public print is incorrectly login-gated: ${JSON.stringify(state)}`);
  if (state.placementFormat !== format || !state.placement.includes("Artifact observed")
    || !state.placement.includes("Source-pinned eligibility") || !state.placement.includes("Configuration-bound")
    || !state.placement.includes("Runtime evidence") || !state.placement.includes("Deployment evidence topology")
    || !state.placement.includes("Claim progression, not a physical-routing observation")) {
    throw new Error(`${format} execution-placement evidence ladder is incomplete or stale: ${JSON.stringify(state)}`);
  }
  const staleTflite = format !== "tflite" && (/53\/53 ops|300,775,552|64\/65 graph ops/.test(state.states));
  if (staleTflite) throw new Error(`${format} retained TFLite quantization evidence: ${JSON.stringify(state)}`);
  if (format === "tflite") {
    if (state.tileCount !== 65 || !state.states.includes("53/53 ops") || !state.states.includes("64/65 graph ops")
      || state.exposureTiles !== 53 || !state.exposure.includes("300,775,552 MACs") || !state.exposure.includes("Exact zero12")
      || !state.placement.includes("TFLite Execution Placement") || !state.placement.includes("Conditional XNNPACK partition flow")
      || !state.claimBoundary.includes("Deep graph and deployment-model audit")
      || state.notApplicableWorkflow.includes("runtime") || state.notApplicableWorkflow.includes("graph")) throw new Error(`TFLite quant baseline changed: ${JSON.stringify(state)}`);
  } else if (format === "gguf") {
    if (state.title !== "GGUF Tensor Encoding" || state.heading !== "GGUF Tensor Encoding & Storage Map" || state.tileCount !== 39
      || state.exposureTiles !== 39 || !state.exposure.includes("Serialized payload")
      || !state.states.includes("30/39 tensors") || !state.states.includes("9/39 tensors")
      || !state.risks.includes("general.quantization_version = 2") || !state.scales.includes("GGUF block encodings")
      || !state.holes.includes("does not serialize an operator graph") || !state.notApplicableTabs.includes("quant-labs")
      || !state.claimBoundary.includes("Container and tensor-payload audit") || !state.evidenceRuntimeDisabled
      || !state.placement.includes("GGUF Execution Placement") || !state.placement.includes("EXECUTION GRAPH EXTERNAL")
      || !["graph", "redesign", "runtime", "deepbom", "runtime_basin", "deployment_sensitivity"].every((step) => state.notApplicableWorkflow.includes(step))) {
      throw new Error(`GGUF encoding evidence is incomplete: ${JSON.stringify(state)}`);
    }
  } else if (format === "safetensors") {
    if (state.title !== "SafeTensors Storage Contract" || state.heading !== "SafeTensors Dtype & Payload Map" || state.tileCount !== 38
      || state.exposureTiles !== 38 || !state.exposure.includes("2.63 MiB")
      || !state.states.includes("F16 storage") || !state.states.includes("2,754,816/2,754,816 B")
      || !state.risks.includes("complete_without_gaps_or_overlaps") || !state.scales.includes("no standardized affine scale")
      || !state.holes.includes("does not serialize an operator graph") || !state.notApplicableTabs.includes("quant-labs")
      || !state.claimBoundary.includes("Checkpoint and shard-integrity audit") || !state.evidenceRuntimeDisabled
      || !state.placement.includes("NOT ASSESSABLE FROM CONTAINER") || !state.placement.includes("delegation map would be fabricated evidence")
      || !["graph", "redesign", "runtime", "deepbom", "runtime_basin", "deployment_sensitivity"].every((step) => state.notApplicableWorkflow.includes(step))) {
      throw new Error(`SafeTensors storage evidence is incomplete: ${JSON.stringify(state)}`);
    }
  } else if (format === "coreml") {
    if (state.title !== "Core ML Numerical Contract" || state.count !== "14 graph ops" || state.tileCount !== 0
      || !state.states.includes("0/10 decoded WeightParams") || !state.states.includes("14/14 layer WeightParams field scans complete")
      || !state.risks.includes("0/10 decoded WeightParams") || !state.scales.includes("0/10 decoded Core ML WeightParams")
      || !state.holes.includes("WeightParams rather than explicit graph Q/DQ") || !state.notApplicableTabs.includes("quant-labs") || state.notApplicableTabs.includes("stage")
      || !state.claimBoundary.includes("Model/package contract and serialized-program audit") || !state.evidenceRuntimeDisabled
      || !state.placement.includes("Core ML Execution Placement") || !state.placement.includes("RUNTIME PLAN REQUIRED")
      || !["redesign", "runtime", "deepbom", "runtime_basin", "deployment_sensitivity"].every((step) => state.notApplicableWorkflow.includes(step))
      || state.notApplicableWorkflow.includes("graph")) {
      throw new Error(`Core ML numerical evidence is incomplete: ${JSON.stringify(state)}`);
    }
  }
}

async function runAudit(page, modelPath, name) {
  await page.locator("#fileInput").setInputFiles({ name, mimeType: "application/octet-stream", buffer: readFileSync(modelPath) });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  if (await page.locator("#workflowConsole").evaluate((node) => node.hidden)) {
    throw new Error(`${name}: workflow navigation remained hidden after artifact staging.`);
  }
  const stagedFormat = name.toLowerCase().endsWith(".tflite") ? "tflite" : "other";
  const targetSwitcherHidden = await page.locator("#targetSwitcherBar").evaluate((node) => node.hidden);
  if (targetSwitcherHidden !== (stagedFormat !== "tflite")) {
    throw new Error(`${name}: target context visibility does not match the staged artifact format.`);
  }
  const stagedSpine = await page.locator("#evidenceSpine").textContent();
  if (!stagedSpine.includes("ArtifactSelected, not audited") || !stagedSpine.includes("DerivationPending audit")
    || !stagedSpine.includes("Deployment modelPending audit")) {
    throw new Error(`${name}: selected-but-not-audited Evidence Spine is inaccurate: ${stagedSpine}`);
  }
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`${name}: ${status}`);
}

async function runVerifiedExample(page) {
  await page.locator("#sampleModelSelect").selectOption("tflite-mobilenet-v2-int8");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  const status = await page.locator("#status").textContent();
  if (!status.includes("audit run complete")) throw new Error(`Verified TFLite example: ${status}`);
  if (await page.locator("#targetSwitcherBar").evaluate((node) => node.hidden)) {
    throw new Error("Verified TFLite example did not reveal its static target context.");
  }
  await page.locator("#sampleVerificationPanel:not([hidden])").waitFor({ timeout: 10_000 });
}

async function runMobileVerifiedExample(page) {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#sampleModelSelect").selectOption("tflite-mobilenet-v2-int8");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => Number(document.querySelector("#auditProgress")?.getAttribute("aria-valuenow") || 0) > 0, null, { timeout: 10_000 });
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  await page.waitForTimeout(1_000);
  const state = await page.evaluate(() => {
    const selector = document.querySelector(".mobile-audit-view");
    const workbench = document.querySelector("#auditWorkbench");
    return {
      status: document.querySelector("#status")?.textContent || "",
      progress: document.querySelector("#auditProgress")?.getAttribute("aria-valuenow"),
      workflowHidden: document.querySelector("#workflowConsole")?.hidden ?? true,
      workbenchHidden: workbench?.hidden ?? true,
      selectorDisplay: selector ? getComputedStyle(selector).display : "none",
      selectorFontSize: getComputedStyle(document.querySelector("#mobileAuditView")).fontSize,
      sampleColumns: getComputedStyle(document.querySelector(".sample-model-control")).gridTemplateColumns,
      workbenchTop: workbench?.getBoundingClientRect().top ?? Number.POSITIVE_INFINITY,
      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      scale: window.visualViewport?.scale || 1,
    };
  });
  if (!state.status.includes("audit run complete") || state.progress !== "100" || state.workflowHidden || state.workbenchHidden
    || state.selectorDisplay === "none" || state.selectorFontSize !== "16px" || state.sampleColumns.split(" ").length !== 1
    || state.workbenchTop < -1 || state.workbenchTop > 844 || state.overflow > 1 || state.scale !== 1) {
    throw new Error(`Mobile verified example workflow is not directly reviewable: ${JSON.stringify(state)}`);
  }
}

function createStaticServer(root) {
  return createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
    const file = path.resolve(root, relative);
    if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
    readFile(file, (error, body) => {
      if (error) return send(response, 404, "text/plain", "not found");
      send(response, 200, mimeType(file), body);
    });
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({ ".css": "text/css", ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
