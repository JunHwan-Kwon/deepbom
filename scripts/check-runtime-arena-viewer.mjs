import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { initSync, analyze_tflite_for_target } from "../pkg/tflite_wasm_audit.js";
import { deriveArenaRuntimeReconciliation } from "../web/lib/runtime-memory.js";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-runtime-arena-viewer-"));
const server = createStaticServer(ROOT);
const browserErrors = [];
let browser;

initSync({ module: readFileSync(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
const analysis = analyze_tflite_for_target(
  new Uint8Array(readFileSync(MODEL)),
  path.basename(MODEL),
  "x86_avx2",
);
const runtimeMemory = actualCaptureRuntimeMemory();
const runtimeEvidence = {
  runtime_memory: runtimeMemory,
  arena_reconciliation: deriveArenaRuntimeReconciliation(analysis, runtimeMemory),
};

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/arena-fixture`, { waitUntil: "domcontentloaded" });
  await page.evaluate(async ({ staticAnalysis, observedEvidence }) => {
    const { renderTensorArenaViewer } = await import("/web/lib/arena-viewer.js");
    renderTensorArenaViewer(document.querySelector("#fixture"), staticAnalysis, {
      runtimeEvidence: observedEvidence,
      formatBytes(value) {
        const bytes = Number(value || 0);
        if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
        if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
        return `${bytes} B`;
      },
    });
  }, { staticAnalysis: analysis, observedEvidence: runtimeEvidence });
  await waitForAnimationFrames(page, 2);

  const desktop = await inspect(page);
  const desktopPath = path.join(output, "runtime-arena-desktop.png");
  await page.locator("#fixture").screenshot({ path: desktopPath });
  requireViewerContract(desktop, false);

  await page.setViewportSize({ width: 390, height: 844 });
  await waitForAnimationFrames(page, 2);
  const mobile = await inspect(page);
  const mobilePath = path.join(output, "runtime-arena-mobile.png");
  await page.locator("#fixture").screenshot({ path: mobilePath });
  requireViewerContract(mobile, true);

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Runtime arena viewer passed (projected ${runtimeEvidence.arena_reconciliation.projected_combined_arena_bytes} B, observed ${runtimeMemory.peak_combined_arena_bytes} B, ${desktop.differenceRows} visible differences, desktop/mobile body overflow 0).`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

function actualCaptureRuntimeMemory() {
  const allocations = [
    { tensor_index: 4, arena: "kTfLiteArenaRw", offset_bytes: 150528, size_bytes: 62720, first_node: 0, last_node: 1 },
    { tensor_index: 6, arena: "kTfLiteArenaRw", offset_bytes: 213248, size_bytes: 1280, first_node: 1, last_node: 2 },
    { tensor_index: 171, arena: "kTfLiteArenaRw", offset_bytes: 0, size_bytes: 150528, first_node: 0, last_node: null },
    { tensor_index: 172, arena: "kTfLiteArenaRw", offset_bytes: 150528, size_bytes: 1001, first_node: 2, last_node: null },
  ];
  const snapshot = {
    memory_snapshot_id: 0,
    non_persistent_arena_bytes: 214528,
    persistent_arena_bytes: 0,
    combined_arena_bytes: 214528,
    tensor_count: 173,
    execution_node_count: 3,
    allocation_count: allocations.length,
    alias_count: 0,
    allocated_interval_bytes: allocations.reduce((sum, item) => sum + item.size_bytes, 0),
    allocations,
    aliases: [],
  };
  return {
    schema: "deepbom.runtime_memory.v1",
    status: "assessed",
    evidence_class: "OBSERVED_RUNTIME",
    tensorflow_source_commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c",
    snapshot_count: 1,
    peak_non_persistent_arena_bytes: snapshot.non_persistent_arena_bytes,
    peak_persistent_arena_bytes: snapshot.persistent_arena_bytes,
    peak_combined_arena_bytes: snapshot.combined_arena_bytes,
    final_non_persistent_arena_bytes: snapshot.non_persistent_arena_bytes,
    final_persistent_arena_bytes: snapshot.persistent_arena_bytes,
    final_combined_arena_bytes: snapshot.combined_arena_bytes,
    allocation_ledger_sha256: "544e8f6ab987f17821835c6897dd9119e361a8708119ac35b52c9f7ccb609ddd",
    snapshots: [snapshot],
    interpretation_boundary: "Observed TFLite arena buffers; delegate-owned buffers and process RSS are excluded.",
  };
}

async function inspect(page) {
  return page.locator("#fixture").evaluate((fixture) => {
    const wrap = fixture.querySelector(".arena-runtime-table-wrap");
    const canvas = fixture.querySelector(".arena-map-canvas");
    const pixels = canvas?.getContext("2d")?.getImageData(0, 0, canvas.width, canvas.height).data || [];
    let nonBlankPixels = 0;
    for (let index = 3; index < pixels.length; index += 4) if (pixels[index] > 0) nonBlankPixels += 1;
    return {
      text: fixture.textContent || "",
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      fixtureOverflow: Math.max(0, fixture.scrollWidth - fixture.clientWidth),
      tableScrollable: Boolean(wrap && wrap.scrollWidth > wrap.clientWidth),
      differenceRows: fixture.querySelectorAll(".arena-runtime-table tbody tr").length,
      note: fixture.querySelector(".arena-runtime-table-note")?.textContent || "",
      canvasWidth: canvas?.clientWidth || 0,
      nonBlankPixels,
    };
  });
}

function requireViewerContract(state, mobile) {
  if (!state.text.includes("Observed TFLite arena allocation")
    || !state.text.includes("Peak observed209.5 KiB")
    || !state.text.includes("Projection delta-1.37 MiB")
    || state.differenceRows !== 40
    || !state.note.includes("40 / 53 difference rows shown")
    || state.bodyOverflow > 1
    || state.fixtureOverflow > 1
    || state.canvasWidth < (mobile ? 320 : 900)
    || state.nonBlankPixels < 1_000
    || (mobile && !state.tableScrollable)) {
    throw new Error(`Runtime arena ${mobile ? "mobile" : "desktop"} contract failed: ${JSON.stringify(state)}`);
  }
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      if (url.pathname === "/arena-fixture") {
        return send(response, 200, "text/html; charset=utf-8", `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/web/styles.css"></head><body><main id="fixture" style="width:min(1180px,calc(100% - 24px));margin:16px auto;padding:12px;box-sizing:border-box;background:var(--surface);border:1px solid var(--line);border-radius:6px"></main></body></html>`);
      }
      const relative = decodeURIComponent(url.pathname).replace(/^\/+/, "");
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
  return ({ ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".wasm": "application/wasm" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
