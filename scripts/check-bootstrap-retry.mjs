import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const root = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const appRequests = [];
let appFailureBudget = 0;
let documentRequests = 0;
const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/web/") documentRequests += 1;
    if (url.pathname === "/web/app.js") {
      appRequests.push(`${url.pathname}${url.search}`);
      if (appFailureBudget > 0) {
        appFailureBudget -= 1;
        return send(response, 503, "text/plain", "transient module fetch failure");
      }
    }
    const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
    const file = path.resolve(root, relative);
    if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
    return send(response, 200, mimeType(file), await readFile(file));
  } catch {
    return send(response, 404, "text/plain", "not found");
  }
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  appFailureBudget = 1;
  const first = await runScenario(browser, server.address().port);
  assert(first.status.includes("Ready"), `Application did not recover after a transient module fetch failure: ${first.status}`);
  assert(appRequests.length >= 2, `Expected an app-module retry, observed ${appRequests.length}: ${appRequests.join(", ")}`);
  assert(appRequests[0] === "/web/app.js", `Unexpected initial app-module request: ${appRequests[0]}`);
  assert(appRequests[1] === "/web/app.js?bootstrap-retry=1", `Retry did not use a distinct module-map identity: ${appRequests[1]}`);
  assert(appRequests.filter((request) => request.startsWith("/web/app.js?bootstrap-retry=")).length === 1,
    `Expected exactly one cache-busted bootstrap retry: ${appRequests.join(", ")}`);
  assert(first.bootstrapErrors.length === 0, `A recovered transient fetch was reported as a terminal bootstrap error: ${first.bootstrapErrors.join(" | ")}`);

  appRequests.length = 0;
  documentRequests = 0;
  appFailureBudget = 3;
  const reloaded = await runScenario(browser, server.address().port);
  assert(reloaded.status.includes("Ready"), `Application did not recover after resetting the failed module map: ${reloaded.status}`);
  assert(documentRequests === 2, `Expected one bounded document reload, observed ${documentRequests} document requests.`);
  assert(appRequests.slice(0, 4).join("|") === [
    "/web/app.js",
    "/web/app.js?bootstrap-retry=1",
    "/web/app.js?bootstrap-retry=2",
    "/web/app.js",
  ].join("|"), `Document reload did not reset the failed module map: ${appRequests.join(", ")}`);
  assert(reloaded.bootstrapErrors.length === 0, `A recovered document reload was reported as terminal: ${reloaded.bootstrapErrors.join(" | ")}`);

  appRequests.length = 0;
  documentRequests = 0;
  appFailureBudget = 6;
  const persistent = await runScenario(browser, server.address().port);
  assert(persistent.status.includes("Application failed to initialize"),
    `Persistent module failure did not reach a terminal state: ${persistent.status}`);
  assert(documentRequests === 2, `Persistent failure exceeded one bounded document reload: ${documentRequests} requests.`);
  assert(appRequests.length === 6, `Persistent failure did not stop after six bounded imports: ${appRequests.join(", ")}`);
  assert(persistent.bootstrapErrors.length === 1,
    `Persistent failure should emit one terminal bootstrap error: ${persistent.bootstrapErrors.join(" | ")}`);
  console.log("Application bootstrap recovery passed (cache-busted retry, one module-map reset, and bounded terminal failure).");
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}

async function runScenario(runningBrowser, port) {
  const context = await runningBrowser.newContext({ serviceWorkers: "block", viewport: { width: 1000, height: 760 } });
  const page = await context.newPage();
  const bootstrapErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error" && message.text().includes("[bootstrap]")) bootstrapErrors.push(message.text());
  });
  try {
    await page.goto(`http://127.0.0.1:${port}/web/`, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => /Ready|Application failed to initialize/.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 90_000 });
    return { status: await page.locator("#status").textContent(), bootstrapErrors };
  } finally {
    await context.close();
  }
}

function mimeType(file) {
  const extension = path.extname(file).toLowerCase();
  return ({
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
  })[extension] || "application/octet-stream";
}

function send(response, status, contentType, body) {
  response.writeHead(status, { "content-type": contentType, "cache-control": "no-store" });
  response.end(body);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
