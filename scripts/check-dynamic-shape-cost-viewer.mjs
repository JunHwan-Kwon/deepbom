import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const fixture = new Uint8Array(readFileSync(path.join(ROOT, "scripts", "fixtures", "onnx_dynamic_conv.onnx")));
const analysis = analyzeOnnxModel(fixture, "onnx_dynamic_conv.onnx");
const stressTerms = Array.from({ length: 225 }, (_, index) => ({
  coefficient_decimal: String(1_692_900 + index * 1024),
  factors: [{ symbol_id: `D${index}`, exponent: 1 }],
}));
const stressExpression = stressTerms
  .map((term, index) => `${term.coefficient_decimal}*D${index}`)
  .join(" + ");
analysis.dynamic_shape_cost_contract.symbol_count = 225;
analysis.dynamic_shape_cost_contract.total_macs_formula.expression = stressExpression;
analysis.dynamic_shape_cost_contract.total_macs_formula.terms = stressTerms;
analysis.dynamic_shape_cost_contract.total_macs_formula.symbol_ids = stressTerms.map((_, index) => `D${index}`);
analysis.dynamic_shape_cost_contract.op_formulas[0].macs_formula.expression = stressExpression;
analysis.dynamic_shape_cost_contract.op_formulas[0].macs_formula.terms = stressTerms;
analysis.dynamic_shape_cost_contract.op_formulas[0].macs_formula.symbol_ids = stressTerms.map((_, index) => `D${index}`);
const server = createStaticServer(ROOT);
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.evaluate(async (model) => {
    const { insightDashboardCards, insightDashboardRecommendationItems } = await import("/web/lib/audit-ui.js");
    const { artifactOverviewPanels } = await import("/web/lib/artifact-overview.js");
    const { renderOpDetailPanel } = await import("/web/lib/graph-ui.js");
    const { humanizeStageKey } = await import("/web/lib/format.js");
    const { createPerformanceVisualController } = await import("/web/lib/performance-visuals.js");
    const cards = document.createElement("section");
    cards.id = "dynamicCards";
    cards.className = "insight-grid";
    const insights = {
      targetLabel: "ONNX static reference",
      targetL1Bytes: 65536,
      quantStatus: { label: "Float" },
      perChannelRatio: 0,
      quantRiskOps: 0,
      topQuantRisk: null,
      maxL1Ratio: null,
      l1Watch: [],
      maxRowWorkingSet: 0,
      l1AssessedCount: 0,
      inputSummary: "batchx3x32x32",
      inputLayoutSummary: "NCHW (DERIVED)",
      dynamicInputs: [0],
      inputLayoutUnassessedCount: 0,
    };
    cards.append(...insightDashboardCards(model, insights));
    const artifact = document.createElement("section");
    artifact.id = "dynamicArtifactOverview";
    artifact.className = "summary-grid";
    const evidence = document.createElement("div");
    evidence.className = "artifact-evidence-grid";
    evidence.append(...artifactOverviewPanels(model));
    artifact.append(evidence);
    const detail = document.createElement("section");
    detail.id = "dynamicOpDetail";
    detail.className = "op-detail panel";
    renderOpDetailPanel(detail, model, 0);
    const flamePanel = document.createElement("article");
    flamePanel.className = "perf-panel";
    const flameSubtitle = document.createElement("span");
    const flame = document.createElement("div");
    flame.id = "flameStructure";
    flame.className = "fc-container";
    flamePanel.append(flameSubtitle, flame);
    const controller = createPerformanceVisualController({
      elements: { perfTimeline: flame, perfTimelineSubtitle: flameSubtitle },
      getContext: () => ({}),
      analyzeForTarget: async () => null,
      jumpToGraphOp: () => {},
    });
    controller.renderPerfTimeline({
      target_profile: { id: "test", label: "Test target" },
      ops: [
        { index: 0, name: "CONV_2D", xnnpack_supported: true, xnnpack_chain_id: 0, bottleneck_compute_us: 90, bottleneck_memory_us: 10, bottleneck_total_us: 90, static_bound_guess: "compute-bound" },
        { index: 1, name: "ADD", xnnpack_supported: true, xnnpack_chain_id: 0, bottleneck_compute_us: 2, bottleneck_memory_us: 1, bottleneck_total_us: 2, static_bound_guess: "compute-bound" },
        { index: 2, name: "RSQRT", xnnpack_supported: false, xnnpack_chain_id: -1, bottleneck_compute_us: 1, bottleneck_memory_us: 2, bottleneck_fallback_us: 1, bottleneck_total_us: 3, static_bound_guess: "memory-bound" },
        { index: 3, name: "PAD", xnnpack_supported: true, xnnpack_chain_id: 1, bottleneck_compute_us: 1, bottleneck_memory_us: 5, bottleneck_total_us: 5, static_bound_guess: "memory-bound" },
      ],
    });
    const actions = document.createElement("ul");
    actions.id = "modeledActions";
    actions.className = "signal-list";
    actions.append(...insightDashboardRecommendationItems({
      format: "tflite",
      target_profile: { id: "selected-target" },
      recommendations: [{
        priority: 1,
        tone: "risk",
        title: "Fallback tensor traffic 12.0% of static bytes",
        detail: "Prioritize fallback traffic by family.",
        op_index: -1,
      }],
      deployment_frontier: {
        interventions: [{
          id: "predicted_fallback_removed",
          per_target: [{
            target_id: "selected-target",
            recoverable_us: 1200,
            recoverable_share: 0.1,
            upper_bound_speedup: 1.111,
          }],
        }],
      },
    }));
    const main = document.createElement("main");
    main.style.maxWidth = "1180px";
    main.style.margin = "0 auto";
    main.dataset.humanStage = humanizeStageKey("330x570xCbucket<=32");
    main.append(artifact, cards, detail, flamePanel, actions);
    document.body.replaceChildren(main);
  }, analysis);

  const desktop = await page.evaluate(() => ({
    cards: document.querySelector("#dynamicCards")?.textContent || "",
    artifact: document.querySelector("#dynamicArtifactOverview")?.textContent || "",
    detail: document.querySelector("#dynamicOpDetail")?.textContent || "",
    artifactDisclosureClosed: document.querySelector('[data-artifact-panel="dynamic-shape-cost"] details')?.open === false,
    detailDisclosureClosed: document.querySelector("#dynamicOpDetail details")?.open === false,
    primaryCards: document.querySelectorAll("#dynamicCards .insight-card-primary").length,
    stageLabel: document.querySelector("main")?.dataset.humanStage || "",
    flameLevels: [...document.querySelectorAll("#flameStructure .fc-level-label strong")].map((node) => node.textContent),
    flameOpLabels: [...document.querySelectorAll("#flameStructure .fc-row-op .fc-block")].map((node) => node.textContent),
    accessibleFlameOps: document.querySelectorAll("#flameStructure .fc-row-op [role='button'][aria-label]").length,
    actionText: document.querySelector("#modeledActions")?.textContent || "",
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!desktop.artifact.includes("Dynamic Shape Cost") || !desktop.artifact.includes(stressExpression)
    || !desktop.detail.includes("Dynamic Shape Cost") || !desktop.detail.includes(stressExpression)
    || !desktop.artifact.includes("225 internal symbols / 225 terms")
    || !desktop.artifact.includes("Internal symbols225")
    || !desktop.artifactDisclosureClosed || !desktop.detailDisclosureClosed
    || desktop.primaryCards !== 2
    || desktop.stageLabel !== "High-resolution spatial stage (330x570, channels <=32)"
    || desktop.flameLevels.join("|") !== "Total|Partitions|Op families|Operators"
    || desktop.flameOpLabels.some((label) => label.length > 0 && !label.startsWith("#"))
    || desktop.accessibleFlameOps !== 4
    || !desktop.actionText.includes("Upper bound: 1.20 ms / 10% / 1.11x")
    || !desktop.actionText.includes("Effort (heuristic): High")
    || !desktop.detail.includes("not_embedded_in_artifact") || desktop.overflow > 1) {
    throw new Error(`Dynamic-shape viewer evidence is incomplete or overflowing on desktop: ${JSON.stringify(desktop)}`);
  }

  await page.setViewportSize({ width: 360, height: 800 });
  const mobile = await page.evaluate(() => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    artifactOverflow: Math.max(0, document.querySelector("#dynamicArtifactOverview")?.scrollWidth - document.querySelector("#dynamicArtifactOverview")?.clientWidth),
    cardOverflow: Math.max(0, document.querySelector("#dynamicCards")?.scrollWidth - document.querySelector("#dynamicCards")?.clientWidth),
    detailOverflow: Math.max(0, document.querySelector("#dynamicOpDetail")?.scrollWidth - document.querySelector("#dynamicOpDetail")?.clientWidth),
    flameOverflow: Math.max(0, document.querySelector("#flameStructure")?.scrollWidth - document.querySelector("#flameStructure")?.clientWidth),
    visibleFlameOpLabels: [...document.querySelectorAll("#flameStructure .fc-row-op .fc-block")]
      .filter((node) => node.textContent && getComputedStyle(node).fontSize !== "0px").length,
  }));
  if (mobile.bodyOverflow > 1 || mobile.artifactOverflow > 1 || mobile.cardOverflow > 1 || mobile.detailOverflow > 1
    || mobile.flameOverflow > 1 || mobile.visibleFlameOpLabels > 0) {
    throw new Error(`Dynamic-shape viewer overflows at 360px: ${JSON.stringify(mobile)}`);
  }
  console.log("Dynamic shape cost viewer checks passed.");
} finally {
  await browser?.close();
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
  return ({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
