import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const server = createStaticServer(ROOT);
const browserErrors = [];
const networkErrors = [];
let browser;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("requestfailed", (request) => networkErrors.push(`request failed: ${request.url()} / ${request.failure()?.errorText || "unknown"}`));
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (response.status() >= 400 && !url.pathname.startsWith("/api/")) networkErrors.push(`HTTP ${response.status()}: ${url.pathname}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) {
      browserErrors.push(`console: ${message.text()}`);
    }
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  try {
    await page.locator("#fileInput").focus();
    await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  } catch (error) {
    const status = await page.locator("#status").textContent().catch(() => "missing #status");
    throw new Error(`App readiness failed (${status}); ${[...browserErrors, ...networkErrors].join("; ") || error.message}`);
  }
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }

  await page.evaluate(() => {
    document.querySelector("#outputModuleSelector").hidden = false;
    document.querySelector("#actions").hidden = false;
    document.querySelector('[data-module-tab="export_contracts"]').click();
    document.querySelector("#exportContractModel").textContent =
      "segmentation_release_candidate_with_a_deliberately_long_reproducible_artifact_name_int8.tflite · sha256:0123456789abcdef0123456789abcdef";
    document.querySelector("#exportContractTarget").textContent =
      "Zynq UltraScale+ Cortex-A53 · profile sha256:fedcba9876543210";
    const status = document.querySelector("#exportContractStatus");
    status.textContent = "Artifact and active target are bound";
    status.classList.add("ready");
  });

  const desktop = await geometry(page);
  assertGeometry(desktop, "desktop");
  if (desktop.rows !== 4 || desktop.buttons !== 6 || desktop.perspectiveButtons !== 0 || !desktop.panelActive) {
    throw new Error(`Export workspace content contract failed: ${JSON.stringify(desktop)}`);
  }

  await page.setViewportSize({ width: 390, height: 844 });
  const mobile = await geometry(page);
  assertGeometry(mobile, "mobile");
  if (!mobile.singleColumnRows || !mobile.fullWidthButtons || !mobile.bindingWraps) {
    throw new Error(`Export workspace mobile layout contract failed: ${JSON.stringify(mobile)}`);
  }
  await verifyContractImport(page);
  if (browserErrors.length) throw new Error(browserErrors.join("\n"));
  console.log("Export contract viewer passed (desktop/mobile hierarchy, binding, overflow, and action layout).");
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

async function verifyContractImport(page) {
  await page.evaluate(async () => {
    const { createExportContractController } = await import("/web/lib/export-contract-view.js");
    const { buildInterfaceQuantizationContractLedger } = await import("/web/lib/quantization-contract-summary.js");
    const { compareInterfaceContracts } = await import("/web/lib/interface-contract.js");
    const root = document.createElement("div");
    root.hidden = true;
    root.innerHTML = `
      <input id="testProductionContractInput" type="file">
      <button id="testClearProductionContract" type="button">Clear</button>
      <strong id="testProductionContractStatus"></strong>
      <span id="testProductionContractSummary"></span>
      <table><tbody id="testProductionContractDiffBody"></tbody></table>
      <div id="testTrustBoundaryBody"></div>`;
    document.body.append(root);
    const analysis = {
      model_sha256: "a".repeat(64),
      target_profile: { profile_sha256: "c".repeat(64) },
      inputs: [{ index: 0, name: "image", dtype: "UINT8", shape: [1, 4], shape_signature: [], quant_scales: 1, quant_zero_points: 1, scale_sample: [0.5], zero_point_sample: [128] }],
      outputs: [{ index: 1, name: "scores", dtype: "FLOAT32", shape: [1, 2], shape_signature: [], quant_scales: 0, quant_zero_points: 0, scale_sample: [], zero_point_sample: [] }],
    };
    const ledger = buildInterfaceQuantizationContractLedger(analysis);
    let declaration = null;
    createExportContractController({
      elements: {
        productionContractInput: root.querySelector("#testProductionContractInput"),
        clearProductionContract: root.querySelector("#testClearProductionContract"),
        productionContractStatus: root.querySelector("#testProductionContractStatus"),
        productionContractSummary: root.querySelector("#testProductionContractSummary"),
        productionContractDiffBody: root.querySelector("#testProductionContractDiffBody"),
        trustBoundaryBody: root.querySelector("#testTrustBoundaryBody"),
      },
      getContext: () => ({ analysis, modelBytes: new Uint8Array([1]), runtimeEvidence: null }),
      getDocuments: async () => ({ documents: { runtime_requirement_manifest: {
        interface_contract_requirement: {
          ledger_sha256: ledger.ledger_sha256,
          comparison: compareInterfaceContracts(ledger, declaration, analysis.model_sha256),
        },
        preprocessing_contract_requirement: { production_binding: { status: "pending" } },
      } } }),
      getAccess: () => ({ rawExportAllowed: true }),
      onProductionContractChange: (value) => { declaration = value; },
    });
  });
  await page.locator("#testProductionContractInput").setInputFiles({
    name: "production-contract.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({
      artifact_sha256: "d".repeat(64),
      implementation_sha256: "b".repeat(64),
      parameters: [],
    })),
  });
  await page.waitForFunction(() => document.querySelector("#testProductionContractStatus")?.textContent === "Artifact mismatch");
  const state = await page.locator("#testProductionContractStatus").getAttribute("data-state");
  const fields = await page.locator("#testProductionContractDiffBody td:nth-child(2)").allTextContents();
  if (state !== "block" || !fields.includes("artifact_sha256") || !fields.includes("parameter")) {
    throw new Error(`Production contract import did not render fail-closed field diffs: ${JSON.stringify({ state, fields })}`);
  }
  await page.locator("#testClearProductionContract").evaluate((button) => button.click());
  await page.waitForFunction(() => document.querySelector("#testProductionContractStatus")?.textContent === "Unbound");
}

async function geometry(page) {
  return page.locator('[data-module-panel="export_contracts"]').evaluate((panel) => {
    const rows = [...panel.querySelectorAll(".export-contract-row")];
    const buttons = [...panel.querySelectorAll("button")];
    const exportButtons = [...panel.querySelectorAll(".export-contract-row button, .export-contract-pack button")];
    const binding = panel.querySelector(".export-contract-binding");
    const rowColumns = rows.map((row) => getComputedStyle(row).gridTemplateColumns.split(" ").filter(Boolean).length);
    return {
      panelActive: panel.classList.contains("active"),
      rows: rows.length,
      buttons: buttons.length,
      perspectiveButtons: panel.querySelectorAll(".perspective-audit-actions button").length,
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      viewportOverflow: Math.max(0, document.documentElement.scrollWidth - innerWidth),
      rowOverflow: Math.max(0, ...rows.map((row) => row.scrollWidth - row.clientWidth)),
      singleColumnRows: rowColumns.every((count) => count === 1),
      fullWidthButtons: exportButtons.every((button) => button.getBoundingClientRect().width >= panel.clientWidth - 60
        || innerWidth > 820),
      bindingWraps: innerWidth > 820 || getComputedStyle(binding).gridTemplateColumns.split(" ").filter(Boolean).length === 1,
    };
  });
}

function assertGeometry(result, label) {
  if (result.panelOverflow > 1 || result.rowOverflow > 1 || result.viewportOverflow > 1) {
    throw new Error(`Export workspace overflows ${label}: ${JSON.stringify(result)}`);
  }
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
    } catch (error) {
      send(response, 404, "text/plain", `not found: ${error?.code || error?.message || "unknown"}`);
    }
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({
    ".css": "text/css",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
