import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { routeClaudeMcp } from "../worker/claude-mcp.js";

// This exercises a simulated MCP App host, not an authenticated Claude account.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const temporary = await mkdtemp(path.join(tmpdir(), "deepbom-claude-widget-"));
const endpoint = process.env.DEEPBOM_CLAUDE_MCP_ENDPOINT;
const published = [];
const contexts = [];
const network = [];
let id = 0;
async function rpc(method, params = {}) {
  const request = new Request(endpoint || "https://deepbom.org/mcp/claude", {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id: ++id, method, params }),
  });
  const response = endpoint ? await fetch(request) : await routeClaudeMcp(request);
  assert.equal(response.status, 200);
  const document = await response.json();
  if (document.error) throw new Error(document.error.message);
  return document.result;
}

const initialized = await rpc("initialize", {
  protocolVersion: "2025-11-25", capabilities: {},
  clientInfo: { name: "deepbom-claude-simulated-host-test", version: "1" },
});
const capabilities = await rpc("tools/call", { name: "deepbom_capabilities", arguments: {} });
assert.equal(capabilities.structuredContent.version, initialized.serverInfo.version);
const opened = await rpc("tools/call", { name: "deepbom_open_local_analyzer", arguments: {} });
assert.equal(opened.structuredContent.status, "awaiting_user_file_selection");
const resource = await rpc("resources/read", { uri: "ui://deepbom/claude-local-analyzer.html" });
const scriptUrl = new URL(resource.contents[0].text.match(/src="([^"]+)"/)[1]);

await build({
  absWorkingDir: root, entryPoints: ["web/claude/deepbom-widget.js"],
  outfile: path.join(temporary, "widget.js"), bundle: true, format: "esm",
  minify: true, platform: "browser", target: "es2022", logLevel: "silent",
});
await build({
  absWorkingDir: root, entryPoints: ["web/workers/static-audit-worker.js"],
  outfile: path.join(temporary, "worker.js"), bundle: true, format: "iife",
  define: { "import.meta.url": "globalThis.__deepbomWorkerModuleUrl" },
  minify: true, platform: "browser", target: "es2022", logLevel: "silent",
});
const assets = new Map([
  ["/claude/deepbom-widget.js", ["text/javascript", await readFile(path.join(temporary, "widget.js"))]],
  ["/chatgpt/static-audit-worker.js", ["text/javascript", await readFile(path.join(temporary, "worker.js"))]],
  ["/pkg/tflite_wasm_audit_bg.wasm", ["application/wasm", await readFile(path.join(root, "pkg/tflite_wasm_audit_bg.wasm"))]],
]);
let assetOrigin;
const server = createServer((request, response) => {
  const url = new URL(request.url, "http://localhost");
  response.setHeader("access-control-allow-origin", "*");
  response.setHeader("cross-origin-resource-policy", "cross-origin");
  const asset = assets.get(url.pathname);
  if (asset) {
    response.writeHead(200, { "content-type": asset[0] }).end(asset[1]);
  } else if (url.pathname === "/host") {
    response.writeHead(200, { "content-type": "text/html" }).end(`<!doctype html><iframe name="widget" sandbox="allow-scripts allow-same-origin" src="/widget" style="width:1050px;height:850px"></iframe>`);
  } else if (url.pathname === "/widget") {
    response.setHeader("content-security-policy", `default-src 'none'; script-src ${assetOrigin} 'wasm-unsafe-eval'; worker-src blob:; connect-src ${assetOrigin}; style-src 'unsafe-inline'; img-src data: blob:`);
    response.writeHead(200, { "content-type": "text/html" }).end(`<!doctype html><main id="deepbom-claude-root"></main><script type="module" src="${assetOrigin}${scriptUrl.pathname}${scriptUrl.search}"></script>`);
  } else response.writeHead(404).end();
});
let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://localhost:${server.address().port}`;
  assetOrigin = process.env.DEEPBOM_CLAUDE_ASSET_ORIGIN || `http://127.0.0.1:${server.address().port}`;
  assert.notEqual(assetOrigin, origin);
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1200, height: 950 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => network.push({ method: request.method(), url: request.url() }));
  await page.exposeFunction("hostRequest", async (message) => {
    if (message.method === "ui/initialize") return {
      protocolVersion: "2026-01-26", hostInfo: { name: "simulated-claude-host", version: "1" },
      hostCapabilities: { serverTools: {}, updateModelContext: {} }, hostContext: { theme: "light" },
    };
    if (message.method === "tools/call") {
      published.push(message.params);
      return rpc("tools/call", message.params);
    }
    if (message.method === "ui/update-model-context") {
      contexts.push(message.params);
      return {};
    }
    throw new Error(`Unexpected bridge method: ${message.method}`);
  });
  await page.addInitScript(() => {
    if (window !== window.top) return;
    window.addEventListener("message", async (event) => {
      const message = event.data;
      if (event.source !== document.querySelector("iframe")?.contentWindow || message?.jsonrpc !== "2.0") return;
      if (message.method === "ui/notifications/initialized") {
        event.source.postMessage({ jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: { analysis_depth: "structure" } } }, event.origin);
      }
      if (!message.id || !message.method) return;
      try {
        const result = await window.hostRequest(message);
        event.source.postMessage({ jsonrpc: "2.0", id: message.id, result }, event.origin);
      } catch (error) {
        event.source.postMessage({ jsonrpc: "2.0", id: message.id, error: { code: -32603, message: error.message } }, event.origin);
      }
    });
  });
  await page.goto(`${origin}/host`);
  const frame = page.frame({ name: "widget" });
  await frame.waitForFunction(() => !document.querySelector("#analyze")?.disabled);
  await frame.locator("#analyze").click();
  assert.match(await frame.locator(".error").textContent(), /file_selection_missing/);
  assert.equal(published.length, 0, "No model result is invented before file selection");

  for (const [filename, format] of [["sample_cnn_float.onnx", "onnx"], ["mobilenet_v2_1.0_224_quant.tflite", "tflite"]]) {
    const artifact = path.join(root, "web/samples", filename);
    await frame.locator("#file").setInputFiles(artifact);
    await frame.locator("#analyze").click();
    await frame.waitForFunction(() => !document.querySelector("#analyze").disabled
      && (document.querySelector("#status").textContent === "Static evidence ready" || document.querySelector(".error")), null, { timeout: 120_000 });
    assert.equal(await frame.locator("#status").textContent(), "Static evidence ready", await frame.locator("#result").textContent());
    const result = published.at(-1).arguments.result;
    assert.equal(result.artifact.format, format);
    assert.equal(result.artifact.sha256, createHash("sha256").update(await readFile(artifact)).digest("hex"));
    assert.equal(result.transfer_boundary, "model_bytes_not_sent_to_deepbom_service");
    assert.equal(contexts.at(-1).structuredContent.artifact.sha256, result.artifact.sha256);
    assert.ok(result.model_summary.row_count > 0);
    assert.ok(JSON.stringify(result).length < 96 * 1024);
    console.log(`${format}: SHA-256, bounded result, model summary, and context handoff passed`);
  }
  await frame.locator("#file").setInputFiles({ name: "unsafe.py", mimeType: "text/plain", buffer: Buffer.from("raise RuntimeError('must not execute')") });
  await frame.locator("#analyze").click();
  await frame.waitForFunction(() => document.querySelector(".error") && !document.querySelector("#analyze").disabled);
  assert.equal(published.at(-1).name, "deepbom_publish_browser_error");
  assert.equal(contexts.at(-1).structuredContent.schema, "deepbom.browser_error.v1");
  assert.equal(await frame.locator(".model-summary").count(), 0);
  assert.ok(network.some(({ url }) => url.includes("/chatgpt/static-audit-worker.js")));
  assert.ok(network.some(({ url }) => url.includes("/pkg/tflite_wasm_audit_bg.wasm")));
  assert.ok(network.every(({ method }) => method === "GET"), "Model bytes must not leave the browser through HTTP");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: "pass", environment: "simulated_mcp_app_host", engine_version: initialized.serverInfo.version, asset_origin: assetOrigin, real_claude_host_validation: "not_run", tools_exercised: 4 }));
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
  await rm(temporary, { recursive: true, force: true });
}
