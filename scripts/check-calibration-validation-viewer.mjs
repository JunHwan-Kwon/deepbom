import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-calibration-validation-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  const modelBytes = new Uint8Array(readFileSync(MODEL));
  initSync({ module: readFileSync(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
  const analysis = analyze_tflite_for_target(modelBytes, path.basename(MODEL), "android_mid_a55");
  const artifactSha256 = createHash("sha256").update(modelBytes).digest("hex");
  const input = analysis.inputs[0];
  const outputTensor = analysis.outputs[0];
  const inputCount = input.shape.reduce((product, dim) => product * dim, 1);
  const outputCount = outputTensor.shape.reduce((product, dim) => product * dim, 1);
  const inputValues = new Array(inputCount).fill(128);
  inputValues[0] = 0;
  inputValues[inputValues.length - 1] = 255;
  const referenceValues = new Array(outputCount).fill(58);
  const changedValues = [...referenceValues];
  changedValues[Math.min(7, changedValues.length - 1)] = 59;
  const tensor = (source, values) => ({
    tensor_index: source.index,
    name: source.name,
    dtype: source.dtype,
    shape: source.shape,
    values,
  });
  const capture = {
    schema: "deepbom.representative_dataset_capture.v1",
    artifact_sha256: artifactSha256,
    dataset: {
      id: "viewer-fixture",
      version: "1.0.0",
      manifest_sha256: "b".repeat(64),
      preprocessing_contract_sha256: "c".repeat(64),
      representativeness_claim: "test_fixture_only",
    },
    runtime: { name: "fixture-runtime", version: "1.0.0", backend: "cpu", binary_sha256: "d".repeat(64) },
    samples: [{
      sample_id: "sample-000",
      inputs: [tensor(input, inputValues)],
      reference_outputs: [tensor(outputTensor, referenceValues)],
      runs: [
        { run_index: 0, outputs: [tensor(outputTensor, referenceValues)] },
        { run_index: 1, outputs: [tensor(outputTensor, changedValues)] },
      ],
    }],
  };
  const capturePath = path.join(output, "representative-capture.json");
  await writeFile(capturePath, JSON.stringify(capture));
  const wrongCapturePath = path.join(output, "wrong-artifact-capture.json");
  await writeFile(wrongCapturePath, JSON.stringify({ ...capture, artifact_sha256: "0".repeat(64) }));

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, acceptDownloads: true });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource|Creating LiteRT|accelerator|Flatbuffer model|XNNPACK delegate/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 240_000 });
  const auditStatus = await page.locator("#status").textContent();
  if (!auditStatus.includes("audit run complete")) throw new Error(auditStatus);
  await page.locator('[data-workflow-step="output"]').click();
  await page.locator('[data-module-tab="engineering_report"]').waitFor({ state: "visible" });
  await page.locator("#calibrationValidationInput").setInputFiles(capturePath);
  await page.waitForFunction(() => document.querySelector("#calibrationValidationStatus")?.dataset.state === "ok", null, { timeout: 30_000 });
  const state = await panelState(page);
  const viewerChecks = {
    metrics: state.metrics === 4,
    rows: state.rows === 1,
    download: !state.downloadDisabled,
    endpoints: /2\s*\/\s*150,528/.test(state.text),
    reference: /1\s*\/\s*2,002/.test(state.text),
    runtime: state.text.includes("fixture-runtime@1.0.0/cpu"),
    boundary: state.text.includes("do not establish dataset representativeness"),
  };
  if (Object.values(viewerChecks).some((value) => !value)) {
    throw new Error(`Representative dataset viewer state is incomplete: ${JSON.stringify({ viewerChecks, state })}`);
  }

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#calibrationValidationDownload").click(),
  ]);
  const ledgerPath = path.join(output, download.suggestedFilename());
  await download.saveAs(ledgerPath);
  const ledger = JSON.parse(await readFile(ledgerPath, "utf8"));
  if (ledger.artifact_sha256 !== artifactSha256 || ledger.interface_binding?.status !== "matched_to_static_audit_external_interface"
    || ledger.input_endpoint_saturation?.endpoint_count !== 2 || ledger.input_endpoint_saturation?.assessed_value_count !== inputCount
    || ledger.reference_output_drift?.changed_value_count !== 1 || ledger.repeat_nondeterminism?.changed_value_count !== 1
    || !/^[a-f0-9]{64}$/.test(ledger.ledger_sha256 || "")) {
    throw new Error(`Downloaded representative dataset ledger is inconsistent: ${JSON.stringify(ledger).slice(0, 3000)}`);
  }
  analysis.model_sha256 = artifactSha256;
  const identity = {
    filename: analysis.filename,
    sha256: artifactSha256,
    file_size_bytes: modelBytes.byteLength,
    format: analysis.format,
    target_profile_id: analysis.target_profile.id,
    target_label: analysis.target_profile.label,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs,
  };
  const runtimeEvidence = { calibrationValidationResult: ledger };
  const report = buildEngineeringReport(analysis, { identity, runtimeEvidence });
  const mlBom = buildMlBomDocument(analysis, { hash: artifactSha256, fileSizeBytes: modelBytes.byteLength, target: analysis.target_profile });
  const evidenceDocument = buildEngineeringEvidenceDocument(analysis, {
    reportContext: { identity, runtimeEvidence },
    rawEvidenceContext: { identity, runtimeEvidence },
    mlBomDocument: mlBom,
  });
  if (!report.includes(ledger.ledger_sha256)
    || evidenceDocument.evidence?.runtime_results?.representative_dataset_validation?.ledger_sha256 !== ledger.ledger_sha256
    || evidenceDocument.evidence?.conformance_report?.status !== "pass") {
    throw new Error("Representative dataset evidence did not conserve through report, engineering evidence, and conformance validation.");
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#calibrationValidationResult").scrollIntoViewIfNeeded();
  const mobile = await page.locator(".calibration-validation-panel").evaluate((panel) => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
    metricColumns: getComputedStyle(panel.querySelector(".calibration-validation-metrics")).gridTemplateColumns.split(" ").length,
    tableScrollable: panel.querySelector(".calibration-validation-table-wrap").scrollWidth > panel.querySelector(".calibration-validation-table-wrap").clientWidth,
  }));
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.metricColumns !== 1 || !mobile.tableScrollable) {
    throw new Error(`Representative dataset viewer mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }

  await page.locator("#calibrationValidationInput").setInputFiles(wrongCapturePath);
  await page.waitForFunction(() => document.querySelector("#calibrationValidationStatus")?.dataset.state === "error", null, { timeout: 10_000 });
  const rejected = await panelState(page);
  if (!rejected.text.includes("different artifact SHA-256") || !rejected.downloadDisabled) throw new Error(`Wrong-artifact capture did not fail closed: ${JSON.stringify(rejected)}`);
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Representative dataset validation viewer passed (artifact + external I/O binding, endpoint ${ledger.input_endpoint_saturation.endpoint_count}/${inputCount}, reference/repeat drift, wrong-artifact rejection, desktop/mobile overflow 0).`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    validationStatus: document.querySelector("#calibrationValidationStatus")?.textContent || null,
    validation: document.querySelector(".calibration-validation-panel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

function panelState(browserPage) {
  return browserPage.locator(".calibration-validation-panel").evaluate((panel) => ({
    metrics: panel.querySelectorAll(".calibration-validation-metric").length,
    rows: panel.querySelectorAll("tbody tr").length,
    downloadDisabled: panel.querySelector("#calibrationValidationDownload")?.disabled,
    text: panel.textContent.replace(/\s+/g, " ").trim(),
  }));
}

function createStaticServer(root) {
  const mime = new Map([
    [".html", "text/html; charset=utf-8"], [".js", "text/javascript; charset=utf-8"], [".mjs", "text/javascript; charset=utf-8"],
    [".css", "text/css; charset=utf-8"], [".wasm", "application/wasm"], [".json", "application/json; charset=utf-8"],
    [".tflite", "application/octet-stream"], [".png", "image/png"], [".svg", "image/svg+xml"],
  ]);
  return createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      let target = path.join(root, pathname.replace(/^\/+/, ""));
      if (pathname.endsWith("/")) target = path.join(target, "index.html");
      if (!path.resolve(target).startsWith(path.resolve(root)) || !existsSync(target) || !(await stat(target)).isFile()) {
        response.writeHead(404).end("not found");
        return;
      }
      response.writeHead(200, { "Content-Type": mime.get(path.extname(target).toLowerCase()) || "application/octet-stream", "Cache-Control": "no-store" });
      createReadStream(target).pipe(response);
    } catch (error) {
      response.writeHead(500).end(error.message);
    }
  });
}
