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

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifactPath = path.join(root, "web", "samples", "sample_cnn_float.onnx");
const artifact = await readFile(artifactPath);
const expectedSha256 = createHash("sha256").update(artifact).digest("hex");
const tfliteArtifactPath = path.join(root, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const tfliteArtifact = await readFile(tfliteArtifactPath);
const expectedTfliteSha256 = createHash("sha256").update(tfliteArtifact).digest("hex");
const temporary = await mkdtemp(path.join(tmpdir(), "deepbom-chatgpt-widget-"));
const bundlePath = path.join(temporary, "deepbom-widget.js");
const workerBundlePath = path.join(temporary, "static-audit-worker.js");

await build({
  absWorkingDir: root,
  entryPoints: ["web/chatgpt/deepbom-widget.js"],
  outfile: bundlePath,
  bundle: true,
  charset: "ascii",
  format: "esm",
  legalComments: "none",
  minify: true,
  platform: "browser",
  sourcemap: false,
  target: "es2022",
});
await build({
  absWorkingDir: root,
  entryPoints: ["web/workers/static-audit-worker.js"],
  outfile: workerBundlePath,
  bundle: true,
  charset: "ascii",
  format: "esm",
  legalComments: "none",
  minify: true,
  platform: "browser",
  sourcemap: false,
  target: "es2022",
});
const [bundle, workerBundle, tfliteWasm] = await Promise.all([
  readFile(bundlePath),
  readFile(workerBundlePath),
  readFile(path.join(root, "pkg", "tflite_wasm_audit_bg.wasm")),
]);
const requests = [];
const server = createServer((request, response) => {
  const url = new URL(request.url || "/", "http://127.0.0.1");
  if (url.pathname === "/test.html") {
    response.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
    response.end(`<!doctype html><html><body><main id="deepbom-chatgpt-root"></main><script type="module" src="/chatgpt/deepbom-widget.js"></script></body></html>`);
    return;
  }
  if (url.pathname === "/chatgpt/deepbom-widget.js") {
    response.writeHead(200, { "content-type": "text/javascript; charset=utf-8", "cache-control": "no-store" });
    response.end(bundle);
    return;
  }
  if (url.pathname === "/workers/static-audit-worker.js") {
    response.writeHead(200, { "content-type": "text/javascript; charset=utf-8", "cache-control": "no-store" });
    response.end(workerBundle);
    return;
  }
  if (url.pathname === "/pkg/tflite_wasm_audit_bg.wasm") {
    response.writeHead(200, { "content-type": "application/wasm", "content-length": tfliteWasm.byteLength, "cache-control": "no-store" });
    response.end(tfliteWasm);
    return;
  }
  if (url.pathname === "/artifact.onnx") {
    requests.push({ method: request.method, range: request.headers.range || null });
    // Exercise the widget's CORS/attachment fallback: not every attachment
    // origin exposes HEAD, while byte-range GET remains available.
    if (request.method === "HEAD") {
      response.writeHead(405, { allow: "GET" });
      response.end();
      return;
    }
    const match = String(request.headers.range || "").match(/^bytes=(\d+)-(\d+)$/);
    if (match) {
      const start = Number(match[1]);
      const end = Math.min(Number(match[2]), artifact.byteLength - 1);
      const body = artifact.subarray(start, end + 1);
      response.writeHead(206, {
        "content-type": "application/octet-stream",
        "content-length": body.byteLength,
        "content-range": `bytes ${start}-${end}/${artifact.byteLength}`,
        "accept-ranges": "bytes",
        "cache-control": "no-store",
      });
      response.end(body);
      return;
    }
    response.writeHead(200, {
      "content-type": "application/octet-stream",
      "content-length": artifact.byteLength,
      "cache-control": "no-store",
    });
    response.end(artifact);
    return;
  }
  response.writeHead(404).end();
});

let browser;
try {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert(address && typeof address === "object");
  const origin = `http://127.0.0.1:${address.port}`;
  const fileUrl = "https://chatgpt-files.example/artifact.onnx";
  browser = await launchChromium(chromium);
  const page = await browser.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(`console: ${message.text()}`);
  });
  await page.route(fileUrl, async (route, request) => {
    const range = request.headers().range || null;
    requests.push({ method: request.method(), range });
    if (request.method() === "HEAD") {
      await route.fulfill({ status: 405, headers: { allow: "GET" }, body: "" });
      return;
    }
    const match = String(range || "").match(/^bytes=(\d+)-(\d+)$/);
    if (match) {
      const start = Number(match[1]);
      const end = Math.min(Number(match[2]), artifact.byteLength - 1);
      const body = artifact.subarray(start, end + 1);
      await route.fulfill({
        status: 206,
        headers: {
          "content-type": "application/octet-stream",
          "content-length": String(body.byteLength),
          "content-range": `bytes ${start}-${end}/${artifact.byteLength}`,
          "accept-ranges": "bytes",
          "cache-control": "no-store",
          "access-control-allow-origin": "*",
          "access-control-expose-headers": "content-length, content-range, accept-ranges",
        },
        body,
      });
      return;
    }
    await route.fulfill({
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(artifact.byteLength),
        "cache-control": "no-store",
        "access-control-allow-origin": "*",
      },
      body: artifact,
    });
  });
  await page.addInitScript(({ fileUrl }) => {
    window.__deepbomToolCalls = [];
    window.__deepbomFollowUps = [];
    window.openai = {
      toolInput: {
        file: {
          file_id: "file_test_onnx",
          file_name: "sample_cnn_float.onnx",
          mime_type: "application/octet-stream",
          download_url: fileUrl,
        },
        analysis_depth: "structure",
      },
      async callTool(name, args) {
        window.__deepbomToolCalls.push({ name, args });
        return { structuredContent: args.result };
      },
      async sendFollowUpMessage(message) {
        window.__deepbomFollowUps.push(message);
      },
    };
  }, { fileUrl });
  await page.goto(`${origin}/test.html`, { waitUntil: "networkidle" });
  try {
    await page.waitForFunction(() => document.querySelector("#status")?.textContent === "Static evidence ready", null, { timeout: 60_000 });
  } catch (error) {
    const state = await page.evaluate(() => ({
      status: document.querySelector("#status")?.textContent || null,
      detail: document.querySelector("#detail")?.textContent || null,
      result: document.querySelector("#result")?.textContent || null,
    }));
    throw new Error(`ChatGPT widget did not finish: ${JSON.stringify({ state, browserErrors, requests })}`, { cause: error });
  }

  const observed = await page.evaluate(() => ({
    calls: window.__deepbomToolCalls,
    followUps: window.__deepbomFollowUps,
    status: document.querySelector("#status")?.textContent,
    detail: document.querySelector("#detail")?.textContent,
  }));
  assert.equal(observed.status, "Static evidence ready");
  assert.equal(observed.calls.length, 1);
  assert.equal(observed.calls[0].name, "deepbom_publish_analysis");
  const result = observed.calls[0].args.result;
  assert.equal(result.schema, "deepbom.chatgpt_analysis_result.v1");
  assert.equal(result.analysis_location, "chatgpt_browser_sandbox");
  assert.equal(result.transfer_boundary, "model_bytes_not_sent_to_deepbom_service");
  assert.equal(result.artifact.filename, "sample_cnn_float.onnx");
  assert.equal(result.artifact.format, "onnx");
  assert.equal(result.artifact.byte_length, artifact.byteLength);
  assert.equal(result.artifact.sha256, expectedSha256);
  assert(Number.isInteger(result.graph.operator_count) && result.graph.operator_count > 0);
  assert(Number.isInteger(result.graph.tensor_count) && result.graph.tensor_count > 0);
  assert.equal(observed.followUps.length, 1);
  assert.match(observed.followUps[0].prompt, /artifact defects, cautions, and evidence gaps separate/);
  assert(requests.some((row) => row.method === "HEAD"), "widget should attempt a non-body size probe");
  assert(requests.some((row) => row.range === "bytes=0-0"), "widget should fall back to a one-byte range request");
  assert(requests.filter((row) => row.range).length >= 3, "analysis should use explicit bounded range reads");

  const tflitePage = await browser.newPage();
  const tfliteUrl = "https://chatgpt-files.example/artifact.tflite";
  const tfliteRequests = [];
  await tflitePage.route(tfliteUrl, (route, request) => fulfillAttachmentRoute(route, request, tfliteArtifact, tfliteRequests));
  await tflitePage.addInitScript(({ tfliteUrl }) => {
    window.__deepbomToolCalls = [];
    window.__deepbomFollowUps = [];
    window.openai = {
      toolInput: {
        file: {
          file_id: "file_test_tflite",
          file_name: "mobilenet_v2_1.0_224_quant.tflite",
          mime_type: "application/octet-stream",
          download_url: tfliteUrl,
        },
        analysis_depth: "structure",
      },
      async callTool(name, args) {
        window.__deepbomToolCalls.push({ name, args });
        return { structuredContent: args.result };
      },
      async sendFollowUpMessage(message) { window.__deepbomFollowUps.push(message); },
    };
  }, { tfliteUrl });
  await tflitePage.goto(`${origin}/test.html`, { waitUntil: "networkidle" });
  await tflitePage.waitForFunction(() => document.querySelector("#status")?.textContent === "Static evidence ready", null, { timeout: 90_000 });
  const tfliteObserved = await tflitePage.evaluate(() => ({
    calls: window.__deepbomToolCalls,
    followUps: window.__deepbomFollowUps,
    status: document.querySelector("#status")?.textContent,
  }));
  assert.equal(tfliteObserved.calls.length, 1);
  assert.equal(tfliteObserved.calls[0].name, "deepbom_publish_analysis");
  const tfliteResult = tfliteObserved.calls[0].args.result;
  assert.equal(tfliteResult.artifact.format, "tflite");
  assert.equal(tfliteResult.artifact.sha256, expectedTfliteSha256);
  assert.equal(tfliteResult.artifact.byte_length, tfliteArtifact.byteLength);
  assert.equal(tfliteResult.graph.operator_count, 65);
  assert.equal(tfliteResult.graph.tensor_count, 173);
  assert.equal(tfliteObserved.followUps.length, 1);
  assert(tfliteRequests.some((row) => row.method === "HEAD"));
  assert(tfliteRequests.some((row) => row.range === "bytes=0-0"));
  await tflitePage.close();

  const badPage = await browser.newPage();
  const badUrl = "https://chatgpt-files.example/unsupported.bin";
  const badBytes = Buffer.from([0xde, 0xad, 0xbe, 0xef]);
  await badPage.route(badUrl, async (route, request) => {
    if (request.method() === "HEAD") {
      await route.fulfill({ status: 200, headers: { "content-length": String(badBytes.byteLength), "access-control-allow-origin": "*", "access-control-expose-headers": "content-length" }, body: "" });
      return;
    }
    await route.fulfill({ status: 206, headers: { "content-length": String(badBytes.byteLength), "content-range": `bytes 0-3/${badBytes.byteLength}`, "access-control-allow-origin": "*", "access-control-expose-headers": "content-length, content-range" }, body: badBytes });
  });
  await badPage.addInitScript(({ badUrl }) => {
    window.__deepbomToolCalls = [];
    window.__deepbomFollowUps = [];
    window.openai = {
      toolInput: { file: { file_id: "file_bad", file_name: "unsupported.bin", download_url: badUrl } },
      async callTool(name, args) {
        window.__deepbomToolCalls.push({ name, args });
        return { structuredContent: args.error };
      },
      async sendFollowUpMessage(message) { window.__deepbomFollowUps.push(message); },
    };
  }, { badUrl });
  await badPage.goto(`${origin}/test.html`, { waitUntil: "networkidle" });
  await badPage.waitForFunction(() => window.__deepbomToolCalls?.length === 1, null, { timeout: 30_000 });
  const failed = await badPage.evaluate(() => ({ calls: window.__deepbomToolCalls, followUps: window.__deepbomFollowUps }));
  assert.equal(failed.calls[0].name, "deepbom_publish_error");
  assert.equal(failed.calls[0].args.error.schema, "deepbom.chatgpt_error.v1");
  assert.equal(failed.calls[0].args.error.code, "unsupported_format");
  assert.equal(failed.calls[0].args.error.transfer_boundary, "model_bytes_not_sent_to_deepbom_service");
  assert.equal(failed.followUps.length, 1);
  await badPage.close();
  console.log(`ChatGPT widget E2E passed (ONNX and isolated-worker TFLite, independent SHA-256, HEAD fallback, range reads, success/error bridges, and follow-ups).`);
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
  await rm(temporary, { recursive: true, force: true });
}

async function fulfillAttachmentRoute(route, request, bytes, ledger) {
  const range = request.headers().range || null;
  ledger.push({ method: request.method(), range });
  if (request.method() === "HEAD") {
    await route.fulfill({ status: 405, headers: { allow: "GET" }, body: "" });
    return;
  }
  const match = String(range || "").match(/^bytes=(\d+)-(\d+)$/);
  if (match) {
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), bytes.byteLength - 1);
    const body = bytes.subarray(start, end + 1);
    await route.fulfill({
      status: 206,
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(body.byteLength),
        "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
        "accept-ranges": "bytes",
        "cache-control": "no-store",
        "access-control-allow-origin": "*",
        "access-control-expose-headers": "content-length, content-range, accept-ranges",
      },
      body,
    });
    return;
  }
  await route.fulfill({
    status: 200,
    headers: {
      "content-type": "application/octet-stream",
      "content-length": String(bytes.byteLength),
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
    },
    body: bytes,
  });
}
