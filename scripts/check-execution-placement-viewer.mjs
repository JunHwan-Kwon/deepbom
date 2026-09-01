import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SCREENSHOTS = path.join(ROOT, ".local-validation", "execution-placement-viewer");
const SHA = "a".repeat(64);
const ops = (names) => names.map((name, index) => ({ index, name }));
const graphOps = (names) => names.map((name, index) => ({ index, name, inputs: [index], outputs: [index + 1] }));
const tensors = (count, dtype = "FLOAT32") => Array.from({ length: count }, (_, index) => ({ index, name: `t${index}`, dtype, shape: [1, 4] }));

const cases = {
  tflite: {
    format: "tflite",
    model_sha256: SHA,
    ops: graphOps(["CONV_2D", "RELU", "AVERAGE_POOL_2D", "CONV_2D", "ADD"]).map((op, index) => ({
      ...op,
      xnnpack_chain_id: [0, 0, -1, 1, 1][index],
      xnnpack_break_class: index === 2 ? "high-adjacent-mac-exposure" : null,
      xnnpack_build_requirement: index === 1 || index === 3 ? "xnnpack-enabled" : null,
    })),
    tensors: tensors(6, "UINT8"),
    delegation_repair: {
      runtime_build_risks: [{ baseline_conditionally_delegatable_op_count: 4, required_build_configuration: "XNNPACK enabled" }],
    },
    tflite_delegate_compatibility_evidence: {
      profiles: [
        {
          id: "tflite_gpu", label: "GPU delegate", source_candidate_after_artifact_precheck_count: 3,
          rows: [
            { op_index: 0, artifact_precheck_status: "source_candidate_partial", unresolved_predicates: ["build"] },
            { op_index: 1, artifact_precheck_status: "source_candidate_partial", unresolved_predicates: ["build"] },
            { op_index: 2, artifact_precheck_status: "definite_exclusion", definite_exclusion_reasons: ["unsupported"] },
            { op_index: 3, artifact_precheck_status: "source_candidate_partial", unresolved_predicates: ["build"] },
            { op_index: 4, artifact_precheck_status: "definite_exclusion", definite_exclusion_reasons: ["unsupported"] },
          ],
        },
        { id: "nnapi", label: "NNAPI", source_candidate_after_artifact_precheck_count: 2 },
      ],
    },
  },
  onnx: {
    format: "onnx",
    model_sha256: SHA,
    ops: graphOps(["Conv", "Relu", "Add"]),
    tensors: tensors(4),
    ort_compatibility_evidence: {
      execution_providers: [
        { execution_provider: "CPUExecutionProvider", label: "CPU EP", source_candidate_after_artifact_precheck_count: 3 },
        { execution_provider: "XnnpackExecutionProvider", label: "XNNPACK EP", source_candidate_after_artifact_precheck_count: 2 },
        {
          execution_provider: "directml", label: "DirectML EP", source_candidate_after_artifact_precheck_count: 2,
          ops: [
            { op_index: 0, status: "SOURCE_KERNEL_VERSION_MATCH", source_candidate_after_artifact_precheck: true, artifact_precheck_status: "ARTIFACT_PRECHECK_PASS" },
            { op_index: 1, status: "SOURCE_KERNEL_VERSION_MATCH", source_candidate_after_artifact_precheck: true, artifact_precheck_status: "ARTIFACT_PRECHECK_UNRESOLVED" },
            { op_index: 2, status: "SOURCE_RULE_NOT_FOUND", source_candidate_after_artifact_precheck: false, artifact_precheck_status: "NOT_APPLICABLE_SOURCE_VERSION_GAP" },
          ],
        },
      ],
    },
  },
  coreml: {
    format: "coreml",
    model_sha256: SHA,
    ops: ops(["convolution", "activation", "add"]),
    coreml: { deployment_floor: { status: "assessed" } },
  },
  gguf: {
    format: "gguf",
    model_sha256: SHA,
    tensor_count: 4,
    gguf: { backend_compatibility: { status: "source_candidate" } },
  },
  safetensors: {
    format: "safetensors",
    model_sha256: SHA,
    tensor_count: 7,
  },
  executorch: {
    format: "executorch",
    executorch_container: "pte",
    model_sha256: SHA,
    ops: [
      { index: 0, name: "aten::add.out", instruction_kind: "KernelCall" },
      { index: 1, name: "QnnBackend", instruction_kind: "DelegateCall" },
      { index: 2, name: "aten::relu.out", instruction_kind: "KernelCall" },
    ],
  },
};

const html = `<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="/web/styles.css">
  <link rel="stylesheet" href="/web/execution-placement.css">
  <link rel="stylesheet" href="/web/explorer-redesign.css">
  <link rel="stylesheet" href="/web/research-theme.css">
  <style>body{margin:0;padding:16px}.mini-panel{max-width:1440px;margin:auto}</style>
</head>
<body>
  <section id="root" class="mini-panel explorer-execution-placement"></section>
  <button id="runDeepBom">source</button>
  <input id="runtimeAssignmentInput" type="file">
  <div data-evidence-stage="deployment"><button id="deploymentAction">deployment</button></div>
  <button data-workflow-step="graph" id="graphAction">graph</button>
  <button data-explorer-tab="kernels" id="kernelAction">kernels</button>
  <details id="formatCapabilityPanel"><summary>capability</summary></details>
</body>
</html>`;

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, "http://127.0.0.1").pathname;
    if (pathname === "/placement-test") {
      response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      response.end(html);
      return;
    }
    if (pathname === "/favicon.ico") {
      response.writeHead(204).end();
      return;
    }
    const resolved = path.resolve(ROOT, `.${decodeURIComponent(pathname)}`);
    if (!resolved.startsWith(ROOT + path.sep)) throw new Error("path outside test root");
    const data = await readFile(resolved);
    const contentType = pathname.endsWith(".css") ? "text/css" : pathname.endsWith(".js") ? "text/javascript" : "application/octet-stream";
    response.writeHead(200, { "content-type": contentType });
    response.end(data);
  } catch {
    response.writeHead(404).end();
  }
});

await mkdir(SCREENSHOTS, { recursive: true });
let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/placement-test`, { waitUntil: "domcontentloaded" });
  await page.evaluate(async () => {
    const module = await import("/web/lib/execution-placement-view.js");
    globalThis.renderPlacementCase = (analysis, runtimeEvidence = null) => module.renderExecutionPlacementView(
      document.querySelector("#root"), analysis, runtimeEvidence, { doc: document },
    );
    globalThis.actionCounts = {};
    globalThis.evidenceSelections = [];
    globalThis.addEventListener("deepbom:evidence-select", (event) => globalThis.evidenceSelections.push(event.detail));
    for (const id of ["runDeepBom", "runtimeAssignmentInput", "deploymentAction", "graphAction", "kernelAction"]) {
      document.getElementById(id).addEventListener("click", () => {
        globalThis.actionCounts[id] = (globalThis.actionCounts[id] || 0) + 1;
      });
    }
  });

  for (const [format, analysis] of Object.entries(cases)) {
    await page.evaluate(({ analysis }) => globalThis.renderPlacementCase(analysis), { analysis });
    await waitForAnimationFrames(page);
    const state = await placementState(page);
    assert.equal(state.format, format, `${format} artifact binding`);
    assert.equal(state.levels.length, 4, `${format} four-level evidence ladder`);
    assert.deepEqual(state.levels.map((level) => level.id), [
      "artifact_observed", "source_pinned_eligibility", "configuration_bound", "runtime_evidence",
    ]);
    assert.equal(state.relations, 3, `${format} relation count`);
    assert.equal(state.externalRelation, 2, `${format} external runtime boundary`);
    assert.equal(state.documentOverflow, 0, `${format} desktop viewport overflow`);
    assert(state.relationFontPx >= 10, `${format} relation label is too small`);
    assert(state.textContrast >= 7, `${format} primary text contrast ${state.textContrast}`);
    assert(state.mutedContrast >= 4.5, `${format} secondary text contrast ${state.mutedContrast}`);
    assert(state.text.includes("Claim progression, not a physical-routing observation"), `${format} interpretation boundary`);
    if (format === "tflite") {
      assert.deepEqual(state.segmentRanges, ["2 items / #0-#1", "1 item / #2", "2 items / #3-#4"]);
      assert.equal(state.flowRole, "region");
      assert.equal(state.flowTabIndex, 0);
    }
    if (format === "onnx") assert(/Each EP is assessed independently against pinned registrations/.test(state.text));
    if (format === "coreml") assert(state.text.includes("RUNTIME PLAN REQUIRED"));
    if (format === "gguf") assert(state.text.includes("EXECUTION GRAPH EXTERNAL"));
    if (format === "safetensors") assert(state.text.includes("NOT ASSESSABLE FROM CONTAINER"));
    if (format === "executorch") {
      assert(state.text.includes("SERIALIZED DELEGATE CALLS"));
      assert(state.text.includes("not an execution trace"));
      assert.deepEqual(state.segmentRanges, ["1 item / #0", "1 item / #1", "1 item / #2"]);
    }
  }

  await page.evaluate(({ analysis }) => globalThis.renderPlacementCase(analysis), { analysis: cases.tflite });
  assert.equal(await page.locator(".placement-profile-comparison-card").count(), 2, "TFLite should begin with one accelerator and one CPU projection.");
  assert.deepEqual((await page.locator(".placement-profile-comparison-card").evaluateAll((nodes) => nodes.map((node) => node.dataset.profileId))).sort(),
    ["tflite_gpu", "xnnpack_cpu"], "Default comparison should bind TFLite GPU and XNNPACK by profile identity.");
  await page.locator('.placement-profile-comparison-toggle input[value="tflite_coreml_delegate"]').check();
  assert.equal(await page.locator(".placement-profile-comparison-card").count(), 3, "N-way comparison should add a third source profile without replacing either baseline.");
  assert.equal(await page.locator('[data-profile-id="tflite_coreml_delegate"]').count(), 1, "Core ML source profile should render by exact profile ID.");
  assert.equal(await page.locator(".placement-profile-detail-select select").inputValue(), "tflite_gpu", "Detailed projection should retain the preferred accelerator profile.");
  const comparisonText = await page.locator(".placement-profile-comparison").innerText();
  assert.match(comparisonText, /Boundary exposure/i);
  assert.match(comparisonText, /GPU roofline[\s\S]*NOT ASSESSED/i);
  assert.match(comparisonText, /Artifact output dtypes[\s\S]*UINT8 5/i);
  assert.match(comparisonText, /Unresolved condition types[\s\S]*1/i);
  assert.match(await page.locator(".placement-profile-detail").innerText(), /Conditionally eligible[\s\S]*3[\s\S]*Definite exclusion[\s\S]*2/i);
  await page.locator(".placement-condition-ledger summary").click();
  const conditionRows = await page.locator(".placement-condition-ledger tbody tr").allInnerTexts();
  assert.deepEqual(conditionRows.map((row) => row.replace(/\s+/g, " ").trim()), [
    "Unresolved condition build 3 #0, #1, #3",
    "Definite exclusion unsupported 2 #2, #4",
  ], "GPU condition ledger must preserve condition role, affected count, and op identity.");
  await page.locator(".placement-boundary-ledger summary").click();
  assert.equal(await page.locator(".placement-boundary-ledger .placement-boundary-table-wrap tbody tr").count(), 3, "GPU profile should expose three cross-state tensor edges.");
  await page.locator(".execution-placement-level summary").first().focus();
  await page.keyboard.press("Enter");
  assert.equal(await page.locator(".execution-placement-level").first().getAttribute("open"), null, "keyboard collapse");
  await page.keyboard.press("Enter");
  assert.notEqual(await page.locator(".execution-placement-level").first().getAttribute("open"), null, "keyboard reopen");
  await page.locator('[data-placement-action="runtime"]').click();
  await page.locator('[data-placement-action="tflite-detail"]').click();
  const actionCounts = await page.evaluate(() => globalThis.actionCounts);
  assert.equal(actionCounts.runtimeAssignmentInput, 1, "runtime import action wiring");
  assert.equal(actionCounts.deploymentAction, 1, "detail action wiring");

  const runtimeFixture = {
    artifact_sha256: SHA,
    runtime_identity_status: "bound",
    assignments: [{ op_index: 0, provider: "XNNPACK", runtime_node_id: "delegate-node-0", mapping_method: "explicit op_index" }],
    runtime_graph: { nodes: [{ id: "generated-copy-1", provider: "CPU" }] },
  };
  await page.evaluate(({ analysis, runtime }) => globalThis.renderPlacementCase(analysis, runtime), { analysis: cases.tflite, runtime: runtimeFixture });
  await page.locator('.runtime-source-reconciliation tr[data-source-op-index="0"]').click();
  await page.locator('.runtime-source-reconciliation tr[data-runtime-node-id="generated-copy-1"]').focus();
  await page.keyboard.press("Enter");
  const evidenceSelections = await page.evaluate(() => globalThis.evidenceSelections);
  assert.deepEqual(evidenceSelections.slice(-2), [
    { op_index: 0, runtime_node_id: "delegate-node-0", source: "runtime-reconciliation" },
    { op_index: null, runtime_node_id: "generated-copy-1", source: "runtime-reconciliation" },
  ], "runtime/source rows must update the shared evidence cursor by exact imported identity.");

  await page.locator("#root").screenshot({ path: path.join(SCREENSHOTS, "light-desktop.png") });
  await page.evaluate(() => document.documentElement.dataset.theme = "dark");
  await waitForAnimationFrames(page);
  const darkState = await placementState(page);
  assert(darkState.textContrast >= 7 && darkState.mutedContrast >= 4.5, `dark contrast ${JSON.stringify(darkState)}`);

  await page.setViewportSize({ width: 390, height: 844 });
  await waitForAnimationFrames(page);
  const mobile = await placementState(page);
  assert.equal(mobile.documentOverflow, 0, "mobile viewport overflow");
  assert.equal(new Set(mobile.levels.map((level) => Math.round(level.left))).size, 1, "mobile vertical level alignment");
  assert(mobile.levels.every((level, index) => !index || level.top > mobile.levels[index - 1].bottom), "mobile level order");
  assert(mobile.actionHeights.every((height) => height >= 44), `mobile action target heights ${mobile.actionHeights}`);
  assert(mobile.comparisonToggleHeights.every((height) => height >= 44), `mobile comparison target heights ${mobile.comparisonToggleHeights}`);
  assert.equal(mobile.comparisonColumns, 1, "mobile accelerator/CPU comparison should stack into one column");
  assert(mobile.comparisonCards.every((card) => card.width <= mobile.rootWidth + 0.5), `mobile comparison card overflow ${JSON.stringify(mobile.comparisonCards)}`);
  assert.equal(mobile.scrollHintDisplay, "block", "mobile segment-scroll instruction");
  assert.equal(mobile.flowOverflowX, "auto", "mobile segment flow must own any horizontal overflow");
  await page.locator("#root").screenshot({ path: path.join(SCREENSHOTS, "dark-mobile.png") });

  assert.deepEqual(browserErrors, [], `browser diagnostics: ${browserErrors.join("; ")}`);
  console.log("Execution placement viewer passed (6 formats; light/dark contrast; desktop/mobile geometry; keyboard, scroll, and action wiring).");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

async function placementState(page) {
  return page.locator("#root").evaluate((root) => {
    const rgb = (value) => (value.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    const luminance = (color) => rgb(color).map((channel) => {
      const value = channel / 255;
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    }).reduce((sum, channel, index) => sum + channel * [0.2126, 0.7152, 0.0722][index], 0);
    const contrast = (a, b) => {
      const [lighter, darker] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (lighter + 0.05) / (darker + 0.05);
    };
    const rootStyle = getComputedStyle(root);
    const muted = root.querySelector(".execution-placement-level p");
    const mutedStyle = getComputedStyle(muted);
    const levels = [...root.querySelectorAll(".execution-placement-level")].map((node) => {
      const rect = node.getBoundingClientRect();
      return { id: node.dataset.placementLevel, top: rect.top, bottom: rect.bottom, left: rect.left };
    });
    const flow = root.querySelector(".execution-placement-flow");
    const comparisonGrid = root.querySelector(".placement-profile-comparison-grid");
    return {
      format: root.dataset.placementFormat,
      text: root.textContent || "",
      levels,
      relations: root.querySelectorAll(".execution-placement-relation").length,
      externalRelation: [...root.querySelectorAll(".execution-placement-relation")].findIndex((node) => node.classList.contains("external")),
      relationFontPx: parseFloat(getComputedStyle(root.querySelector(".execution-placement-relation span")).fontSize),
      segmentRanges: [...(root.querySelector(".execution-placement-body > .execution-placement-flow")?.querySelectorAll(".execution-placement-segment span") || [])].map((node) => node.textContent),
      flowRole: flow?.getAttribute("role") || "",
      flowTabIndex: flow?.tabIndex ?? -1,
      flowScrollWidth: flow?.scrollWidth || 0,
      flowClientWidth: flow?.clientWidth || 0,
      flowOverflowX: flow ? getComputedStyle(flow).overflowX : "",
      scrollHintDisplay: root.querySelector(".execution-placement-scroll-hint")
        ? getComputedStyle(root.querySelector(".execution-placement-scroll-hint")).display
        : "absent",
      actionHeights: [...root.querySelectorAll(".execution-placement-actions button")].map((node) => node.getBoundingClientRect().height),
      comparisonToggleHeights: [...root.querySelectorAll(".placement-profile-comparison-toggle")].map((node) => node.getBoundingClientRect().height),
      comparisonColumns: comparisonGrid ? getComputedStyle(comparisonGrid).gridTemplateColumns.split(" ").filter(Boolean).length : 0,
      comparisonCards: [...root.querySelectorAll(".placement-profile-comparison-card")].map((node) => {
        const rect = node.getBoundingClientRect();
        return { left: rect.left, right: rect.right, width: rect.width };
      }),
      rootWidth: root.getBoundingClientRect().width,
      documentOverflow: Math.max(0, document.documentElement.scrollWidth - innerWidth),
      textContrast: contrast(rootStyle.color, rootStyle.backgroundColor),
      mutedContrast: contrast(mutedStyle.color, mutedStyle.backgroundColor === "rgba(0, 0, 0, 0)" ? rootStyle.backgroundColor : mutedStyle.backgroundColor),
    };
  });
}
