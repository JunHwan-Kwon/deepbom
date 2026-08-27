import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const server = createStaticServer(ROOT);
const errors = [];
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await launchChromium(chromium);

  const accessPage = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  watchErrors(accessPage, errors);
  await mockJson(accessPage, "**/api/auth/me", { user: null });
  let releaseActivation;
  const activationGate = new Promise((resolve) => { releaseActivation = resolve; });
  let activationObserved;
  const activationRequest = new Promise((resolve) => { activationObserved = resolve; });
  await accessPage.route("**/api/test/activate", async (route) => {
    const body = JSON.parse(route.request().postData() || "{}");
    activationObserved(body);
    await activationGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true, redirect_to: "/test?activated=mock" }),
    });
  });
  await accessPage.goto(`${origin}/test#access=fixture-access-token`, { waitUntil: "domcontentloaded" });
  const activationBody = await activationRequest;
  const accessState = await accessPage.evaluate(() => ({
    pathname: location.pathname,
    hash: location.hash,
    title: document.querySelector("#testAccessStatusTitle")?.textContent || "",
    identity: document.querySelector("#testAccessIdentity")?.textContent || "",
    stored: sessionStorage.getItem("deepbom.external-test.link.v2"),
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (activationBody.access !== "fixture-access-token" || accessState.pathname !== "/test" || accessState.hash
    || accessState.title !== "Activating test access" || accessState.stored !== "fixture-access-token"
    || accessState.overflow > 1) {
    throw new Error(`Automatic access gateway state is invalid: ${JSON.stringify({ activationBody, accessState })}`);
  }
  releaseActivation();
  await accessPage.waitForURL("**/test?activated=mock");

  const activePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  watchErrors(activePage, errors);
  await mockJson(activePage, "**/api/auth/me", { user: testerUser() });
  await activePage.goto(`${origin}/test`, { waitUntil: "domcontentloaded" });
  await activePage.locator("#testOpenWorkbench").waitFor({ state: "visible" });
  const activeState = await activePage.evaluate(() => ({
    title: document.querySelector("#testAccessStatusTitle")?.textContent || "",
    status: document.querySelector("#testAccessStatus")?.textContent || "",
    deactivateVisible: !document.querySelector("#testDeactivate")?.hidden,
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (activeState.title !== "Test access active" || !activeState.status.includes("Admin operations remain unavailable")
    || !activeState.deactivateVisible || activeState.overflow > 1) {
    throw new Error(`Active test gateway state is invalid: ${JSON.stringify(activeState)}`);
  }

  const adminPage = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  watchErrors(adminPage, errors);
  await mockJson(adminPage, "**/api/auth/me", { user: { ...testerUser(), role: "admin", access_profile: "admin" } });
  await mockJson(adminPage, "**/api/admin/requests", { requests: [] });
  await mockJson(adminPage, "**/api/admin/users", { users: [] });
  await mockJson(adminPage, "**/api/admin/benchmarks", { runs: [] });
  await mockJson(adminPage, "**/api/admin/model-structures?limit=50", { structures: [] });
  await adminPage.route("**/api/admin/test-links", async (route) => {
    const body = JSON.parse(route.request().postData() || "{}");
    if (Object.keys(body).length) throw new Error("24-hour link issuance must not accept identity or duration overrides.");
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({
        access_url: `${origin}/test#access=signed-fixture`,
        expires_at: "2026-08-09T02:00:00.000Z",
      }),
    });
  });
  await adminPage.goto(`${origin}/web/admin.html`, { waitUntil: "domcontentloaded" });
  await adminPage.locator("#testLinkCreate").click();
  await adminPage.locator("#testLinkResult").waitFor({ state: "visible" });
  const adminState = await adminPage.evaluate(() => ({
    url: document.querySelector("#testLinkUrl")?.value || "",
    meta: document.querySelector("#testLinkMeta")?.textContent || "",
    warning: document.querySelector(".external-test-link-form p")?.textContent || "",
    overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
  }));
  if (!adminState.url.includes("/test#access=signed-fixture") || !adminState.meta.includes("automatic access")
    || !adminState.meta.includes("no Admin access") || !adminState.warning.includes("Anyone who receives it")
    || adminState.overflow > 1) {
    throw new Error(`Admin link UI is invalid: ${JSON.stringify(adminState)}`);
  }

  if (errors.length) throw new Error(errors.join("\n"));
  console.log("External test pages passed (fragment hygiene, automatic account-free activation, active-session boundary, Admin link issuance, and responsive overflow). ");
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

function testerUser() {
  return {
    id: "external-test:fixture",
    email: "",
    name: "External Tester",
    provider: "access_link",
    role: "user",
    access_profile: "medical_ai",
    access_status: "active",
    email_verified: true,
    test_access: {
      active: true,
      expires_at: "2026-08-09T02:00:00.000Z",
      account_bound: false,
      admin_access: false,
    },
  };
}

function watchErrors(page, output) {
  page.on("pageerror", (error) => output.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) output.push(`console: ${message.text()}`);
  });
}

async function mockJson(page, pattern, body, status = 200) {
  await page.route(pattern, (route) => route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  }));
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      let pathname = url.pathname;
      if (pathname === "/test" || pathname === "/test/") pathname = "/web/test.html";
      if (pathname === "/web/" || pathname === "/web") pathname = "/web/index.html";
      const file = path.join(root, pathname.replace(/^\//, ""));
      const bytes = await readFile(file);
      response.writeHead(200, { "content-type": contentType(file), "cache-control": "no-store" });
      response.end(bytes);
    } catch {
      response.writeHead(404);
      response.end("not found");
    }
  });
}

function contentType(file) {
  if (file.endsWith(".html")) return "text/html; charset=utf-8";
  if (file.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (file.endsWith(".css")) return "text/css; charset=utf-8";
  return "application/octet-stream";
}
