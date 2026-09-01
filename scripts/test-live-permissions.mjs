import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import process from "node:process";

const DEFAULT_URL = "https://deepbom.org/";
const DEFAULT_MODEL = "F:/consistency/models/main_0604_v119_4_ckpt902087_int8.tflite";
const DEFAULT_CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const SESSION_COOKIE = "audit_session";
const D1_DATABASE = "deepbom_auth";
const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const ROLE_CASES = [
  { id: "public", label: "Public", accessProfile: "anonymous" },
  { id: "verified", label: "Verified", accessProfile: "verified" },
  { id: "research", label: "Research", accessProfile: "research" },
  { id: "medical_ai", label: "Medical AI", accessProfile: "medical_ai", path: "/medical" },
  { id: "admin", label: "Admin", accessProfile: "admin", role: "admin" },
];

const options = parseArgs(process.argv.slice(2));
const baseUrl = new URL(options.url || DEFAULT_URL);
const modelPath = resolve(options.model || DEFAULT_MODEL);
const chromePath = resolve(options.chrome || DEFAULT_CHROME);
const testRunId = `codex-${Date.now().toString(36)}`;
const users = [];
const selectedRoleCases = options.caseId
  ? ROLE_CASES.filter((testCase) => testCase.id === options.caseId)
  : ROLE_CASES;
if (!selectedRoleCases.length) throw new Error(`Unknown --case value: ${options.caseId}`);

async function main() {
  try {
    const accounts = await installTemporaryAccounts();
    for (const testCase of selectedRoleCases) {
      const account = accounts.get(testCase.id) || null;
      const result = await runBrowserCase(testCase, account);
      console.log(
        [
          `case=${testCase.id}`,
          `audit=${result.status}`,
          `summary=${result.summaryCount}`,
          `engineering=${result.moduleTabs.engineering_report?.badge || "n/a"}`,
          `deepbom=${result.moduleTabs.deepbom?.badge || "n/a"}`,
          `perturbation=${result.moduleTabs.perturbation?.badge || "n/a"}`,
          `runtime_basin=${result.moduleTabs.runtime_basin?.badge || "n/a"}`,
          `deploy=${result.moduleTabs.deployment_sensitivity?.badge || "n/a"}`,
        ].join(" "),
      );
    }
    console.log("status=ok");
  } finally {
    if (!options.keepAccounts) {
      await cleanupTemporaryAccounts().catch((error) => {
        console.error(`cleanup_failed=${error.message}`);
        process.exitCode = 1;
      });
    } else {
      console.log(`temporary_accounts_kept=${users.map((user) => user.email).join(",")}`);
    }
  }
}

function parseArgs(args) {
  const parsed = { url: "", model: "", chrome: "", caseId: "", keepAccounts: false };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--url") {
      parsed.url = args[index + 1] || "";
      index += 1;
    } else if (arg === "--model") {
      parsed.model = args[index + 1] || "";
      index += 1;
    } else if (arg === "--chrome") {
      parsed.chrome = args[index + 1] || "";
      index += 1;
    } else if (arg === "--case") {
      parsed.caseId = args[index + 1] || "";
      index += 1;
    } else if (arg === "--keep-accounts") {
      parsed.keepAccounts = true;
    }
  }
  return parsed;
}

async function installTemporaryAccounts() {
  const accounts = new Map();
  const now = new Date().toISOString();
  const expiresAt = new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString();
  for (const item of ROLE_CASES.filter((entry) => entry.id !== "public")) {
    const user = {
      id: randomUUID(),
      role: item.role || "user",
      accessProfile: item.accessProfile,
      email: `${testRunId}-${item.id}@deepbom.org`,
      name: `Codex ${item.label}`,
      sessionToken: randomBytes(32).toString("hex"),
      createdAt: now,
      expiresAt,
    };
    user.sessionHash = sha256Hex(user.sessionToken);
    users.push(user);
    accounts.set(item.id, user);
  }
  const statements = [];
  for (const user of users) {
    statements.push(
      `INSERT INTO users (id, email, name, avatar_url, google_sub, password_hash, password_salt, provider, created_at, updated_at, role, access_profile, access_status, access_expires_at, email_verified_at) VALUES (${sql(user.id)}, ${sql(user.email)}, ${sql(user.name)}, '', NULL, NULL, NULL, 'password', ${sql(now)}, ${sql(now)}, ${sql(user.role)}, ${sql(user.accessProfile)}, 'active', NULL, ${sql(now)})`,
      `INSERT INTO sessions (id_hash, user_id, created_at, expires_at) VALUES (${sql(user.sessionHash)}, ${sql(user.id)}, ${sql(now)}, ${sql(expiresAt)})`,
    );
  }
  await d1Batch(statements, "install");
  return accounts;
}

async function cleanupTemporaryAccounts() {
  if (!users.length) return;
  const idList = users.map((user) => sql(user.id)).join(", ");
  await d1Batch([
    `DELETE FROM sessions WHERE user_id IN (${idList})`,
    `DELETE FROM email_verification_tokens WHERE user_id IN (${idList})`,
    `DELETE FROM access_requests WHERE user_id IN (${idList})`,
    `DELETE FROM consent WHERE user_id IN (${idList})`,
    `DELETE FROM benchmark_runs WHERE user_id IN (${idList})`,
    `DELETE FROM users WHERE id IN (${idList})`,
  ], "cleanup");
}

async function d1Batch(statements, label) {
  if (!statements.length) return;
  const dir = await mkdtemp(join(tmpdir(), `deepbom-d1-${label}-`));
  const file = join(dir, `${label}.sql`);
  await writeFile(file, `${statements.join(";\n")};\n`, "utf8");
  try {
    await d1File(file);
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

async function d1File(file) {
  const args = ["wrangler", "d1", "execute", D1_DATABASE, "--remote", "--file", file];
  const bin = process.platform === "win32" ? "cmd.exe" : "npx";
  const finalArgs = process.platform === "win32" ? ["/d", "/s", "/c", "npx", ...args] : args;
  let lastError = null;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const result = spawnSync(bin, finalArgs, {
      cwd: ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.status === 0) return;
    lastError = new Error(`D1 command failed (${result.status}): ${result.error?.message || result.stderr || result.stdout}`);
    if (attempt < 3 && /fetch failed|connectivity issue|timed out|ETIMEDOUT|ECONNRESET/i.test(lastError.message)) {
      await sleep(1500 * attempt);
      continue;
    }
    break;
  }
  throw lastError;
}

function sql(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function sha256Hex(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function runBrowserCase(testCase, account) {
  const port = await freeDebugPort();
  const userDataDir = await mkdtemp(join(tmpdir(), `deepbom-${testCase.id}-`));
  const chrome = spawn(chromePath, [
    "--headless=new",
    `--remote-debugging-port=${port}`,
    "--remote-debugging-address=127.0.0.1",
    "--remote-allow-origins=*",
    `--user-data-dir=${userDataDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-sandbox",
    "--window-size=1440,1100",
    "about:blank",
  ], {
    stdio: ["ignore", "ignore", "pipe"],
  });
  const chromeErrors = [];
  chrome.stderr.on("data", (chunk) => {
    const text = String(chunk).trim();
    if (text) chromeErrors.push(text);
  });

  let page = null;
  try {
    const pageUrl = await createDebugPage(port);
    page = new CdpPage(pageUrl);
    await page.ready();
    await page.enable();
    await page.setViewport();
    if (account) await page.setSessionCookie(baseUrl.hostname, account.sessionToken);
    await page.goto(urlForCase(testCase));
    if (account) await page.waitForAuthEmail(account.email);
    await page.acceptPrivacyAgreement();
    await page.uploadModel(modelPath);
    await page.runStaticAudit();
    const snapshot = await page.snapshot();
    assert(snapshot.status.includes("audit run complete"), `${testCase.id}: audit status should complete; got ${snapshot.status}`);
    assert(snapshot.summaryCount > 0, `${testCase.id}: summary cards should render.`);
    assert(snapshot.insightVisible, `${testCase.id}: insight dashboard should be visible.`);
    await assertCaseAccess(testCase, page);
    page.assertNoSeriousErrors(testCase.id);
    return await page.snapshot();
  } finally {
    await page?.shutdownBrowser().catch(() => {});
    await page?.close().catch(() => {});
    chrome.kill();
    await waitForExit(chrome, 3000).catch(() => chrome.kill("SIGKILL"));
    await rm(userDataDir, { recursive: true, force: true }).catch(() => {});
    const seriousChromeErrors = chromeErrors.filter((line) => !isBenignChromeLine(line));
    if (seriousChromeErrors.length) {
      console.warn(`${testCase.id}: chrome_stderr=${seriousChromeErrors.slice(0, 3).join(" | ")}`);
    }
  }
}

async function assertCaseAccess(testCase, page) {
  await page.clickWorkflow("output");
  await page.waitFor(() => `!document.getElementById("outputModuleSelector").hidden`, 5000);
  await page.waitForAccessStatus(testCase.id);
  const base = await page.snapshot();
  if (testCase.id === "public") {
    assert(base.moduleTabs.engineering_report.badge === "Report", "public: report workspace should expose the complete watermarked report.");
    assert(base.moduleTabs.deepbom.badge === "Sign in", "public: DEEPBOM should require sign in.");
    assert(!base.buttons.downloadMarkdownDisabled, "public: Engineering Report download should be enabled without sign in.");
    assert(!base.buttons.publicPrintDisabled, "public: watermarked PDF print should remain available without sign in.");
    assert(base.buttons.rawDisabled, "public: raw data should be disabled.");
    return;
  }
  if (testCase.id === "verified") {
    await page.waitFor(() => `document.getElementById("reportPreview").textContent.includes("# DEEPBOM Static Artifact Engineering Audit")`, 10000);
    const snapshot = await page.snapshot();
    assert(snapshot.moduleTabs.engineering_report.badge === "Report", "verified: Engineering Report should remain available.");
    assert(snapshot.moduleTabs.deepbom.badge === "Module access", "account: DEEPBOM should require module authorization.");
    assert(!snapshot.buttons.downloadMarkdownDisabled, "verified: Engineering Report download should be enabled.");
    assert(!snapshot.buttons.rawDisabled, "verified: raw data should be enabled.");
    assert(!snapshot.buttons.engineeringBundleDisabled, "verified: Engineering Bundle should be enabled.");
    return;
  }
  if (testCase.id === "research") {
    assert(base.moduleTabs.deepbom.badge === "Not run", "research: DEEPBOM should be runnable.");
    assert(base.moduleTabs.runtime_basin.badge === "Not run", "research: Runtime Basin should be runnable.");
    assert(base.moduleTabs.deployment_sensitivity.badge === "Not run", "research: Deployment Sensitivity should be runnable.");
    assert(!base.buttons.rawDisabled, "controlled module account: account-bound exports should remain enabled.");
    await page.runDeepBom();
    await page.runResearchModule("runtime_basin", "#perturbationPanelAction", "#runtimeBasinStatus");
    await page.runResearchModule("deployment_sensitivity", "#deploymentSensitivityPanelAction", "#deploymentSensitivityStatus");
    const snapshot = await page.snapshot();
    assert(snapshot.moduleTabs.deepbom.badge === "Complete", "research: DEEPBOM should complete.");
    assert(/Complete|Rank changed/.test(snapshot.perturbationStatus), `research: Perturbation should finish, got ${snapshot.perturbationStatus}. ${snapshot.perturbationDetail}`);
    assert(snapshot.runtimeBasinStatus === "Complete", `research: Runtime Basin should finish, got ${snapshot.runtimeBasinStatus}.`);
    assert(snapshot.moduleTabs.runtime_basin.badge === "Complete", "research: Runtime Basin should complete.");
    assert(snapshot.moduleTabs.deployment_sensitivity.badge === "Complete", "research: Deployment Sensitivity should complete.");
    return;
  }
  if (testCase.id === "medical_ai") {
    assert(base.moduleTabs.regulatory_report?.badge === "Report", "medical_ai: Regulatory Support Report should export on /medical.");
    assert(!base.buttons.downloadRegulatoryDisabled, "medical_ai: Regulatory Report download should be enabled.");
    assert(!base.buttons.evidenceBundleDisabled, "medical_ai: Regulatory Bundle should be enabled.");
    return;
  }
  if (testCase.id === "admin") {
    assert(base.auth.adminVisible, "admin: admin button should be visible.");
    assert(base.moduleTabs.deepbom.badge === "Not run", "admin: DEEPBOM should be runnable.");
    assert(base.moduleTabs.runtime_basin.badge === "Not run", "admin: Perturbation and Runtime Basin should be runnable.");
    assert(base.accessStatus.includes("Admin access"), "admin: module access status should show admin access.");
  }
}

function urlForCase(testCase) {
  const url = new URL(testCase.path || "/", baseUrl);
  url.searchParams.set("permission_e2e", `${testRunId}-${testCase.id}`);
  return url.href;
}

async function freeDebugPort() {
  const { createServer } = await import("node:net");
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

async function createDebugPage(port) {
  await waitForJson(`http://127.0.0.1:${port}/json/version`, 30000);
  const response = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent("about:blank")}`, { method: "PUT" });
  if (!response.ok) throw new Error(`Chrome /json/new failed: ${response.status}`);
  const target = await response.json();
  return target.webSocketDebuggerUrl;
}

async function waitForJson(url, timeoutMs) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
    } catch (error) {
      lastError = error;
    }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError?.message || "no response"}`);
}

class CdpPage {
  constructor(webSocketUrl) {
    this.webSocketUrl = webSocketUrl;
    this.socket = new WebSocket(webSocketUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.errors = [];
    this.loaded = false;
    this.socket.addEventListener("message", (event) => this.onMessage(event));
  }

  ready() {
    if (this.socket.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
  }

  async enable() {
    await this.send("Page.enable");
    await this.send("Runtime.enable");
    await this.send("DOM.enable");
    await this.send("Network.enable");
    await this.send("Log.enable");
  }

  async setViewport() {
    await this.send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1100,
      deviceScaleFactor: 1,
      mobile: false,
    });
  }

  async setSessionCookie(hostname, token) {
    await this.send("Network.setCookie", {
      name: SESSION_COOKIE,
      value: token,
      domain: hostname,
      path: "/",
      secure: true,
      httpOnly: true,
      sameSite: "Lax",
      url: `https://${hostname}/`,
    });
  }

  async goto(url) {
    this.loaded = false;
    await this.send("Page.navigate", { url });
    await this.waitFor(() => `document.readyState === "complete"`, 20000);
    await this.waitFor(() => `Boolean(document.getElementById("fileInput"))`, 15000);
  }

  async acceptPrivacyAgreement() {
    await this.eval(`
      (() => {
        const backdrop = document.getElementById("agreementBackdrop");
        const agree = document.getElementById("privacyAgree");
        const accept = document.getElementById("acceptAgreement");
        if (backdrop && !backdrop.hidden) {
          if (!agree.checked) agree.click();
          accept.click();
        }
        return true;
      })()
    `);
    await this.waitFor(() => `document.getElementById("agreementBackdrop")?.hidden !== false`, 5000);
  }

  async waitForAuthEmail(email) {
    await this.waitFor(() => `
      (document.getElementById("authEmail")?.textContent || "").includes(${JSON.stringify(email)})
    `, 15000);
  }

  async uploadModel(path) {
    const { root } = await this.send("DOM.getDocument", { depth: 1 });
    const { nodeId } = await this.send("DOM.querySelector", { nodeId: root.nodeId, selector: "#fileInput" });
    assert(nodeId, "Could not find #fileInput.");
    await this.send("DOM.setFileInputFiles", { nodeId, files: [path] });
    await this.eval(`document.getElementById("fileInput").dispatchEvent(new Event("change", { bubbles: true }))`);
    await this.waitFor(() => `!document.getElementById("runAudit").disabled`, 10000);
  }

  async runStaticAudit() {
    await this.click("#runAudit");
    await this.waitFor(() => `
      (() => {
        const plan = document.getElementById("analysisPlanStatus")?.textContent || "";
        const status = document.getElementById("status")?.textContent || "";
        if (/failed/i.test(plan) || /failed/i.test(status)) throw new Error("audit failed: " + plan + " / " + status);
        return /audit run complete/i.test(plan) || /audit run complete/i.test(status);
      })()
    `, 30000);
  }

  async clickWorkflow(step) {
    await this.click(`[data-workflow-step="${step}"]`);
    await sleep(150);
  }

  async runDeepBom() {
    await this.clickWorkflow("deepbom");
    await this.waitFor(() => `!document.getElementById("moduleRunConsole").hidden`, 5000);
    await this.click("#runDeepBom");
    await this.waitFor(() => `
      (() => {
        const value = document.getElementById("deepBomStatus")?.textContent || "";
        if (/failed/i.test(value)) throw new Error("DEEPBOM failed: " + value);
        const badge = document.querySelector('[data-module-tab="deepbom"] em')?.textContent || "";
        return badge === "Complete" && value && !/Not loaded|Ready|Authorizing|Loading|Analyzing/i.test(value);
      })()
    `, 30000);
    await this.clickWorkflow("output");
  }

  async waitForAccessStatus(caseId) {
    const expected = {
      verified: "Raw exports enabled",
      research: "Advanced modules enabled",
      medical_ai: "Regulatory workspace enabled",
      admin: "Admin access",
    }[caseId];
    if (!expected) return;
    await this.waitFor(() => `
      (document.getElementById("moduleAccessStatus")?.textContent || "").includes(${JSON.stringify(expected)})
    `, 10000);
  }

  async runResearchModule(step, buttonSelector, statusSelector) {
    await this.clickWorkflow(step);
    await this.waitFor(() => `!document.querySelector(${JSON.stringify(buttonSelector)}).disabled`, 10000);
    await this.click(buttonSelector);
    await this.waitFor(() => `
      (() => {
        const statusNode = document.querySelector(${JSON.stringify(statusSelector)});
        const value = statusNode?.textContent || "";
        if (/failed/i.test(value)) {
          const detail = statusNode?.closest(".module-panel")?.textContent?.replace(/\s+/g, " ").trim().slice(0, 700) || value;
          throw new Error(${JSON.stringify(step)} + " failed: " + detail);
        }
        return value && !/Not run|Ready|Authorizing|Waiting|Running locally/i.test(value);
      })()
    `, 60000);
    await this.clickWorkflow("output");
  }

  async click(selector) {
    await this.eval(`
      (() => {
        const node = document.querySelector(${JSON.stringify(selector)});
        if (!node) throw new Error(${JSON.stringify(`Missing selector: ${selector}`)});
        node.click();
        return true;
      })()
    `);
  }

  async waitFor(expressionFactory, timeoutMs) {
    const started = Date.now();
    let lastError = null;
    while (Date.now() - started < timeoutMs) {
      try {
        const value = await this.eval(expressionFactory(), true);
        if (value) return value;
      } catch (error) {
        lastError = error;
        if (/failed|Missing selector|Could not/i.test(error.message)) throw error;
      }
      await sleep(200);
    }
    throw new Error(`Timed out waiting for expression. Last error: ${lastError?.message || "none"}`);
  }

  async snapshot() {
    return this.eval(`
      (() => {
        const text = (selector) => document.querySelector(selector)?.textContent?.trim() || "";
        const button = (selector) => document.querySelector(selector);
        const moduleTabs = {};
        for (const tab of document.querySelectorAll("[data-module-tab]")) {
          moduleTabs[tab.dataset.moduleTab] = {
            badge: tab.querySelector("em")?.textContent?.trim() || "",
            disabled: tab.getAttribute("aria-disabled") === "true",
          };
        }
        return {
          status: text("#analysisPlanStatus") || text("#status"),
          topStatus: text("#status"),
          workflow: text("#workflowMode"),
          summaryCount: document.getElementById("summary")?.children.length || 0,
          insightVisible: document.getElementById("insightDashboard")?.hidden === false,
          accessStatus: text("#moduleAccessStatus"),
          moduleTabs,
          auth: {
            name: text("#authName"),
            email: text("#authEmail"),
            role: text("#authRole"),
            adminVisible: document.getElementById("adminOpen")?.hidden === false,
          },
          buttons: {
            publicPrintDisabled: button("#printPublicReport")?.disabled ?? null,
            downloadMarkdownDisabled: button("#downloadMarkdown")?.disabled ?? null,
            rawDisabled: button("#downloadRawData")?.disabled ?? null,
            engineeringBundleDisabled: button("#downloadEngineeringBundle")?.disabled ?? null,
            downloadRegulatoryDisabled: button("#downloadRegulatoryReport")?.disabled ?? null,
            evidenceBundleDisabled: button("#downloadEvidenceBundle")?.disabled ?? null,
          },
          reportPreview: text("#reportPreview").slice(0, 200),
          regulatoryPreview: text("#regulatoryReportPreview").slice(0, 200),
          deepBomStatus: text("#deepBomStatus"),
          perturbationStatus: text("#perturbationStatus"),
          perturbationDetail: text("#perturbationResultPanel").replace(/\s+/g, " ").slice(0, 900),
          runtimeBasinStatus: text("#runtimeBasinStatus"),
          deploymentSensitivityStatus: text("#deploymentSensitivityStatus"),
        };
      })()
    `, true);
  }

  async eval(expression, awaitPromise = false) {
    const response = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise,
      returnByValue: true,
      userGesture: true,
    });
    if (response.exceptionDetails) {
      const detail = response.exceptionDetails.exception?.description || response.exceptionDetails.text || "Runtime evaluation failed";
      throw new Error(detail);
    }
    return response.result?.value;
  }

  send(method, params = {}) {
    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params });
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.socket.send(payload);
    return promise;
  }

  onMessage(event) {
    const message = JSON.parse(event.data);
    if (message.id) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message || JSON.stringify(message.error)));
      else pending.resolve(message.result || {});
      return;
    }
    if (message.method === "Runtime.exceptionThrown") {
      const text = message.params?.exceptionDetails?.exception?.description || message.params?.exceptionDetails?.text || "exception";
      this.errors.push({ kind: "exception", text });
    } else if (message.method === "Runtime.consoleAPICalled" && message.params?.type === "error") {
      const text = (message.params.args || []).map((arg) => arg.value || arg.description || "").join(" ");
      if (!isBenignBrowserLog(text)) this.errors.push({ kind: "console", text });
    } else if (message.method === "Log.entryAdded" && message.params?.entry?.level === "error") {
      const entry = message.params.entry;
      const text = [entry.text || "", entry.url || ""].filter(Boolean).join(" ");
      if (!isBenignBrowserLog(text)) this.errors.push({ kind: "log", text });
    }
  }

  assertNoSeriousErrors(caseId) {
    if (this.errors.length) {
      throw new Error(`${caseId}: browser errors: ${this.errors.map((item) => `${item.kind}:${item.text}`).join(" | ")}`);
    }
  }

  async shutdownBrowser() {
    await this.send("Browser.close").catch(() => {});
  }

  close() {
    return new Promise((resolve) => {
      if (this.socket.readyState === WebSocket.CLOSED) return resolve();
      this.socket.addEventListener("close", resolve, { once: true });
      this.socket.close();
      setTimeout(resolve, 500);
    });
  }
}

function isBenignBrowserLog(text) {
  return (
    !text ||
    /favicon\.ico/i.test(text) ||
    /Failed to load resource: the server responded with a status of 404/i.test(text) ||
    (/static\.cloudflareinsights\.com\/beacon\.min\.js/i.test(text) && /violates the following Content Security Policy directive/i.test(text) && /has been blocked/i.test(text)) ||
    /^INFO:\s*\[/i.test(text) ||
    /^WARNING:\s*\[/i.test(text) ||
    /Created TensorFlow Lite XNNPACK delegate for CPU/i.test(text)
  );
}

function isBenignChromeLine(text) {
  return /DevTools listening|usb_service|Bluetooth|policy|ERROR:device_event_log|ERROR:gpu/i.test(text);
}

function waitForExit(child, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("process exit timeout")), timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

await main();
