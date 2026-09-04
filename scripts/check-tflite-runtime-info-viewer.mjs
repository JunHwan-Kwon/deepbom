import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v1_025_224_float.tflite");
const LIVE_URL = readArgument("--url");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-tflite-runtime-info-viewer-"));
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
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  await page.locator("#fileInput").focus();
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
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadRuntimeAssignmentTemplate").click(),
  ]);
  const template = JSON.parse(await readFile(await download.path(), "utf8"));
  const partitions = partitionInventory(template.graph_ops);
  if (!template.graph_ops.length || !partitions.length) throw new Error("TFLite fixture requires parsed ops and at least one predicted delegate partition.");
  const proto = modelRuntimeDetails(template.graph_ops, partitions);
  await page.locator("#runtimeAssignmentInput").setInputFiles({
    name: "model_runtime_details.pb",
    mimeType: "application/octet-stream",
    buffer: Buffer.from(proto),
  });
  await page.locator("#runtimeProfileBackdrop").waitFor({ state: "visible" });
  const modal = await page.evaluate(() => ({
    title: document.querySelector("#runtimeProfileTitle")?.textContent || "",
    optimizationHidden: Boolean(document.querySelector("#runtimeProfileOptimization")?.closest("label")?.hidden),
    executionHidden: Boolean(document.querySelector("#runtimeProfileExecutionMode")?.closest("label")?.hidden),
    backendReadOnly: Boolean(document.querySelector("#runtimeProfileBackend")?.readOnly),
    preview: Object.fromEntries([...document.querySelectorAll(".runtime-profile-preview-item")].map((item) => [item.querySelector("span")?.textContent, item.querySelector("strong")?.textContent])),
  }));
  if (modal.title !== "TFLite runtime plan" || !modal.optimizationHidden || !modal.executionHidden || !modal.backendReadOnly) {
    throw new Error(`TFLite runtime-plan modal semantics failed: ${JSON.stringify(modal)}`);
  }
  if (modal.preview["Original ops"] !== `${template.graph_ops.length}/${template.graph_ops.length}`
    || modal.preview["Delegate partitions"] !== String(partitions.length)
    || modal.preview.Binding !== "Exact topology") {
    throw new Error(`TFLite runtime-plan preview failed: ${JSON.stringify(modal.preview)}`);
  }
  const desktopModalPath = path.join(output, "tflite-runtime-plan-modal-desktop.png");
  await page.locator("#runtimeProfileBackdrop").screenshot({ path: desktopModalPath });
  await page.locator("#runtimeProfileVersion").fill("tensorflow/tensorflow@runtime-fixture");
  await page.locator("#runtimeProfileBuild").fill("benchmark_model release; XNNPACK enabled; fixture execution plan");
  await page.locator("#runtimeProfileCollectedAt").fill("2026-07-16T12:00");
  await page.locator("#runtimeProfileCapture").fill("viewer-capture-001");
  await page.locator("#runtimeProfileImport").click();
  await page.waitForFunction(() => {
    const status = document.querySelector("#runtimeAssignmentStatus")?.textContent || "";
    return status.includes("TFLite runtime plan") && status.includes("exact topology") && status.includes("source artifact SHA-256 absent");
  }, null, { timeout: 30_000 });
  const statusText = await page.locator("#runtimeAssignmentStatus").textContent();
  if (statusText?.includes("undefined") || !statusText?.includes("executed microkernel not exposed")) {
    throw new Error(`TFLite runtime evidence status omitted an evidence boundary: ${statusText}`);
  }
  const rows = await page.locator("#kernelInspectorBody tr").count();
  if (rows !== template.graph_ops.length) throw new Error(`TFLite runtime plan rendered ${rows}/${template.graph_ops.length} op rows.`);
  const comparisonText = await page.locator("#runtimeAssignmentComparison").textContent();
  if (!comparisonText?.includes("Placement match") || !comparisonText.includes("100%") || !comparisonText.includes(`Observed partitions${partitions.length}`)) {
    throw new Error(`TFLite runtime comparison did not render exact observed placement: ${comparisonText}`);
  }
  const firstDelegated = template.graph_ops.find((op) => op.predicted_delegated);
  await page.locator(`#kernelInspectorBody tr[data-op-index="${firstDelegated.op_index}"]`).click().catch(async () => {
    await page.locator("#kernelInspectorBody tr").nth(firstDelegated.op_index).click();
  });
  const detailText = await page.locator("#opDetail").textContent();
  if (!detailText?.includes("runtime_info_original_node_id_and_symmetric_delegate_map")
    || !detailText.includes("source artifact SHA-256 not embedded")
    || !detailText.includes("not exposed by imported TFLite evidence")) {
    throw new Error("TFLite op detail did not expose exact mapping, artifact binding, and microkernel evidence limits.");
  }
  const timingProto = benchmarkProfile(template.graph_ops, partitions);
  await page.locator("#runtimeAssignmentInput").setInputFiles({
    name: "benchmark_profiling_data.pb",
    mimeType: "application/octet-stream",
    buffer: Buffer.from(timingProto),
  });
  await page.locator("#runtimeProfileBackdrop").waitFor({ state: "visible" });
  const timingModal = await page.evaluate(() => ({
    title: document.querySelector("#runtimeProfileTitle")?.textContent || "",
    captureVisible: !document.querySelector("#runtimeProfileCaptureLabel")?.hidden,
    versionHidden: Boolean(document.querySelector("#runtimeProfileVersion")?.closest("label")?.hidden),
    preview: Object.fromEntries([...document.querySelectorAll(".runtime-profile-preview-item")].map((item) => [item.querySelector("span")?.textContent, item.querySelector("strong")?.textContent])),
  }));
  if (timingModal.title !== "TFLite execution profile" || !timingModal.captureVisible || !timingModal.versionHidden
    || timingModal.preview["Execution nodes"] !== `${executionPlan(template.graph_ops, partitions).length}/${executionPlan(template.graph_ops, partitions).length}`
    || timingModal.preview["Graph total"] === "Withheld") {
    throw new Error(`TFLite timing modal semantics failed: ${JSON.stringify(timingModal)}`);
  }
  await page.locator("#runtimeProfileCapture").fill("viewer-capture-001");
  await page.locator("#runtimeProfileCollectedAt").fill("2026-07-16T12:00:02");
  await page.locator("#runtimeProfileImport").click();
  await page.waitForFunction(() => {
    const text = document.querySelector("#runtimeAssignmentComparison")?.textContent || "";
    return text.includes("Execution-plan total") && text.includes("Observed latency hotspots")
      && text.includes("Primary delegate-profiled subtotal") && text.includes("Nested delegate-internal subtotal");
  }, null, { timeout: 30_000 });
  const timingText = await page.locator("#runtimeAssignmentComparison").textContent();
  if (!timingText?.includes("non-additive") || timingText.includes("Execution-plan totalWithheld")) {
    throw new Error(`TFLite timing viewer omitted total or delegate-internal isolation: ${timingText}`);
  }
  const placementText = await page.locator("#executionPlacementPanel").textContent();
  if (!placementText?.includes("RUNTIME OBSERVED") || !placementText.includes("Observed assignment flow")
    || !placementText.includes("TFLite non-delegated kernel")) {
    throw new Error(`TFLite observed execution placement did not replace the conditional-only state: ${placementText}`);
  }
  await page.locator("#kernelInspectorBody tr").nth(firstDelegated.op_index).click();
  const timedDetail = await page.locator("#opDetail").textContent();
  if (!timedDetail?.includes("partition total") || !timedDetail.includes("not attributed to this original op")) {
    throw new Error("Delegated op detail did not preserve partition-total attribution boundaries.");
  }
  await page.locator(".kernel-inspector-wrap").evaluate((element) => { element.scrollTop = 0; });
  const desktopViewerPath = path.join(output, "tflite-runtime-plan-viewer-desktop.png");
  await page.locator("#kernelInspectorPanel").screenshot({ path: desktopViewerPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(100);
  const mobile = await page.evaluate(() => {
    const status = document.querySelector("#runtimeAssignmentStatus");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      statusClip: Math.max(0, (status?.scrollHeight || 0) - (status?.clientHeight || 0)),
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.statusClip > 1) throw new Error(`TFLite runtime viewer mobile overflow: ${JSON.stringify(mobile)}`);
  await page.evaluate(() => {
    const comparison = document.querySelector("#runtimeAssignmentComparison");
    if (!comparison) throw new Error("Runtime comparison specimen is missing.");
    document.body.replaceChildren(comparison);
    document.body.style.margin = "0";
    comparison.hidden = false;
    comparison.style.width = "100%";
  });
  const mobilePath = path.join(output, "tflite-runtime-plan-viewer-mobile.png");
  await page.screenshot({ path: mobilePath, fullPage: true });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`TFLite runtime-info viewer passed (${template.graph_ops.length}/${template.graph_ops.length} exact ops, ${partitions.length} delegate partition(s), full execution timing, desktop/mobile overflow 0).`);
  console.log(`desktop_modal=${desktopModalPath}`);
  console.log(`desktop_viewer=${desktopViewerPath}`);
  console.log(`mobile_viewer=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    runtimeStatus: document.querySelector("#runtimeProfileStatus")?.textContent || null,
    assignmentStatus: document.querySelector("#runtimeAssignmentStatus")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  if (server) await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

function partitionInventory(ops) {
  const groups = new Map();
  for (const op of ops) {
    if (!op.predicted_delegated) continue;
    const key = String(op.predicted_chain_id);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(op.op_index);
  }
  return [...groups.values()].map((replaced, index) => ({ delegateNodeId: ops.length + index, replaced }));
}

function modelRuntimeDetails(ops, partitions) {
  const partitionByOp = new Map(partitions.flatMap((partition) => partition.replaced.map((opIndex) => [opIndex, partition])));
  const originalNodes = ops.map((op) => node({
    id: op.op_index,
    name: op.op_name,
    type: "0",
    inputs: op.input_tensor_ids,
    outputs: op.output_tensor_ids,
    delegatedTo: partitionByOp.get(op.op_index)?.delegateNodeId ?? null,
  }));
  const delegateNodes = partitions.map((partition) => node({
    id: partition.delegateNodeId,
    name: "TfLiteXNNPackDelegate",
    type: "Delegate/TfLiteXNNPackDelegate",
    inputs: [],
    outputs: [],
    delegateName: "TfLiteXNNPackDelegate",
    replacedIds: partition.replaced,
  }));
  const plan = executionPlan(ops, partitions);
  const subgraph = message([
    intField(1, 0),
    ...[...originalNodes, ...delegateNodes].map((value) => bytesField(3, value)),
    packedIntField(4, plan),
    intField(5, 1),
    stringField(6, "main"),
  ]);
  return message([stringField(1, "viewer fixture"), bytesField(2, subgraph)]);
}

function executionPlan(ops, partitions) {
  const partitionByOp = new Map(partitions.flatMap((partition) => partition.replaced.map((opIndex) => [opIndex, partition])));
  const plan = [];
  const emittedPartitions = new Set();
  for (const op of ops) {
    const partition = partitionByOp.get(op.op_index);
    if (!partition) plan.push(op.op_index);
    else if (!emittedPartitions.has(partition.delegateNodeId)) {
      emittedPartitions.add(partition.delegateNodeId);
      plan.push(partition.delegateNodeId);
    }
  }
  return plan;
}

function benchmarkProfile(ops, partitions) {
  const byOp = new Map(ops.map((op) => [op.op_index, op]));
  const partitionIds = new Set(partitions.map((item) => item.delegateNodeId));
  const profiles = executionPlan(ops, partitions).map((nodeId, runOrder) => opProfile(
    partitionIds.has(nodeId) ? "TfLiteXNNPackDelegate" : byOp.get(nodeId).op_name,
    `[node-${nodeId}]:${nodeId}`,
    runOrder,
    partitionIds.has(nodeId) ? 100 + runOrder : 5 + runOrder,
  ));
  const primary = message([stringField(1, "main"), intField(2, 0), ...profiles.map((value) => bytesField(3, value))]);
  const internal = message([stringField(1, "TfLiteXNNPackDelegate"), bytesField(2, opProfile("xnn_f32_gemm", "Delegate/xnn_f32_gemm:0", 0, 40))]);
  return message([stringField(1, "viewer fixture"), bytesField(3, message([bytesField(1, primary), bytesField(2, internal)]))]);
}

function opProfile(nodeType, name, runOrder, duration) {
  const stat = message([intField(1, duration), intField(2, duration), intField(3, duration), floatField(4, 0), floatField(5, 0), intField(6, duration), intField(7, duration), intField(8, duration * 2), intField(9, 2)]);
  return message([stringField(1, nodeType), bytesField(2, stat), intField(4, 1), stringField(5, name), intField(6, runOrder)]);
}

function node({ id, name, type, inputs, outputs, delegatedTo = null, delegateName = null, replacedIds = [] }) {
  const fields = [intField(1, id), stringField(2, name), stringField(3, type), packedIntField(4, inputs), packedIntField(5, outputs)];
  if (delegateName != null) fields.push(bytesField(8, message([stringField(1, delegateName), packedIntField(2, replacedIds)])));
  else if (delegatedTo != null) fields.push(intField(9, delegatedTo));
  return message(fields);
}

function intField(field, value) { return concat(varint((field << 3) | 0), varint(value)); }
function bytesField(field, value) { return concat(varint((field << 3) | 2), varint(value.length), value); }
function stringField(field, value) { return bytesField(field, new TextEncoder().encode(value)); }
function packedIntField(field, values) { return bytesField(field, concat(...values.map(varint))); }
function floatField(field, value) { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setFloat32(0, value, true); return concat(varint((field << 3) | 5), bytes); }
function message(fields) { return concat(...fields); }
function varint(value) {
  let remaining = BigInt(value);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return Uint8Array.from(bytes);
}
function concat(...arrays) {
  const output = new Uint8Array(arrays.reduce((sum, value) => sum + value.length, 0));
  let offset = 0;
  for (const value of arrays) { output.set(value, offset); offset += value.length; }
  return output;
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
function send(response, status, type, body) { response.writeHead(status, { "content-type": type, "cache-control": "no-store" }); response.end(body); }
function mimeType(file) { return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".tflite": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream"; }
function readArgument(name) { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] || null : null; }
