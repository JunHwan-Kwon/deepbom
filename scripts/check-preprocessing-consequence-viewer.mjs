import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium, waitForAnimationFrames } from "./browser-launch.mjs";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import {
  buildEngineeringEvidenceDocument,
  buildRuntimeEvidence,
  buildStaticAnalysisExport,
  buildQuantizationEvidence,
} from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { visualPngSpecs } from "../web/lib/visual-export.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const SERVE_ROOT = process.env.DEEPBOM_E2E_SERVE_ROOT ? path.resolve(process.env.DEEPBOM_E2E_SERVE_ROOT) : ROOT;
const MODEL = path.join(ROOT, "web", "samples", "mobilenet_v2_1.0_224_quant.tflite");
const output = await mkdtemp(path.join(tmpdir(), "deepbom-preprocessing-consequence-"));
const server = createStaticServer(SERVE_ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1120 }, deviceScaleFactor: 1, acceptDownloads: true });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource|Creating LiteRT|NPU accelerator|RegisterAccelerator|GpuEnvironment|GPU accelerator registered|CPU accelerator registered|Adding options|Flatbuffer model|XNNPACK delegate/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.locator("#fileInput").focus();
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
  await page.locator('[data-audit-tab="quant-labs"]').click();
  await page.locator('[data-quant-lab-tab="preprocessing"]').click();
  await page.locator("#preprocessingConsequencePanel").waitFor({ state: "visible" });
  const ready = await panelState(page);
  if (ready.status !== "runtime replay not run" || ready.metrics !== 5 || ready.runDisabled || !ready.downloadDisabled
    || !ready.text.includes("Candidate contracts8") || !ready.text.includes("Exact tensor aliases4")
    || !ready.text.includes("No output consequence has been measured")) {
    throw new Error(`Preprocessing consequence ready state is incomplete: ${JSON.stringify(ready)}`);
  }

  await page.locator("#runPreprocessingConsequence").click();
  await page.waitForFunction(() => ["independently verified", "replay rejected"].includes(document.querySelector("#preprocessingConsequenceStatus")?.textContent || ""), null, { timeout: 240_000 });
  const replayStatus = await page.locator("#preprocessingConsequenceStatus").textContent();
  if (replayStatus !== "independently verified") throw new Error(`Preprocessing replay failed: ${replayStatus}`);
  await waitForAnimationFrames(page);

  const state = await panelState(page);
  if (state.status !== "independently verified" || state.metrics !== 5 || state.canvases !== 1 || state.tables !== 2
    || state.matrixRows !== 8 || state.options !== 8 || state.runDisabled || state.downloadDisabled
    || state.canvasPixels < 800 || !state.text.includes("2/2")
    || !state.text.includes("Input equivalence classes") || !state.text.includes("Output equivalence classes")
    || !state.text.includes("does not observe the production decoder")) {
    throw new Error(`Preprocessing consequence result is incomplete: ${JSON.stringify({ ...state, text: state.text.slice(0, 5000) })}`);
  }

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#downloadPreprocessingConsequence").click(),
  ]);
  const evidencePath = path.join(output, download.suggestedFilename());
  await download.saveAs(evidencePath);
  const evidence = JSON.parse(await readFile(evidencePath, "utf8"));
  validateEvidence(evidence);

  const selectedIndex = Math.max(0, evidence.candidates.findIndex((row) => row.contract_id === evidence.most_output_sensitive_contract_id));
  await page.locator(".consequence-select select").selectOption(String(selectedIndex));
  await waitForAnimationFrames(page);
  const selected = evidence.candidates[selectedIndex];
  const selectedState = await panelState(page);
  if (!selectedState.text.includes(selected.contract_label) || !selectedState.text.includes(selected.output_tensor_set_sha256)) {
    throw new Error(`Selected consequence row did not render: ${JSON.stringify(selectedState).slice(0, 3000)}`);
  }

  const [candidateDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#preprocessingConsequencePanel").getByRole("button", { name: "Candidate JSON", exact: true }).click(),
  ]);
  const candidatePath = path.join(output, candidateDownload.suggestedFilename());
  await candidateDownload.saveAs(candidatePath);
  const candidate = JSON.parse(await readFile(candidatePath, "utf8"));
  if (candidate.contract_id !== selected.contract_id || candidate.candidate_ledger_sha256 !== selected.candidate_ledger_sha256) {
    throw new Error("Selected preprocessing consequence JSON is detached from the portfolio.");
  }

  const [outputDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#preprocessingConsequencePanel").getByRole("button", { name: "Output tensor", exact: true }).click(),
  ]);
  const outputPath = path.join(output, outputDownload.suggestedFilename());
  await outputDownload.saveAs(outputPath);
  const outputBytes = await readFile(outputPath);
  if (outputBytes.byteLength !== selected.output_tensors[0].element_count * dtypeBytes(selected.output_tensors[0].output_dtype)) {
    throw new Error(`Selected output byte length is inconsistent: ${outputBytes.byteLength}.`);
  }

  const reportCheck = buildReportEvidence(evidence);
  if (!reportCheck.conformancePassed || !reportCheck.findingPassed || !reportCheck.mlBomPassed || !reportCheck.runtimePassed || !reportCheck.visualPassed) {
    throw new Error(`Preprocessing consequence report integration failed: ${JSON.stringify(reportCheck)}`);
  }

  const visualExport = await page.evaluate(async (runtimeEvidence) => {
    const viewer = await import("/web/lib/preprocessing-consequence-viewer.js");
    const canvas = viewer.renderPreprocessingConsequenceCanvas(runtimeEvidence, null, "mobilenet_v2_1.0_224_quant.tflite");
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    for (let index = 0; index < pixels.length; index += 128) if (pixels[index] !== pixels[index + 1] || pixels[index + 1] !== pixels[index + 2]) colored += 1;
    return { width: canvas.width, height: canvas.height, colored };
  }, evidence);
  if (visualExport.width !== 2360 || visualExport.height !== 1540 || visualExport.colored < 1_000) {
    throw new Error(`Preprocessing consequence Visual PNG renderer is invalid: ${JSON.stringify(visualExport)}`);
  }

  const desktopPath = path.join(output, "preprocessing-consequence-desktop.png");
  await page.locator("#preprocessingConsequencePanel").screenshot({ path: desktopPath });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#preprocessingConsequencePanel").scrollIntoViewIfNeeded();
  const mobile = await page.locator("#preprocessingConsequencePanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const summary = panel.querySelector(".consequence-summary");
    return {
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: Math.max(0, panel.scrollWidth - panel.clientWidth),
      summaryColumns: getComputedStyle(summary).gridTemplateColumns.split(" ").length,
      canvasWidth: canvas.getBoundingClientRect().width,
      canvasHeight: canvas.getBoundingClientRect().height,
      matrixScrollable: panel.querySelector(".consequence-band.table-band .consequence-table-wrap")?.scrollWidth > panel.querySelector(".consequence-band.table-band .consequence-table-wrap")?.clientWidth,
    };
  });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.summaryColumns !== 2
    || mobile.canvasWidth < 280 || mobile.canvasWidth > 360 || mobile.canvasHeight < 700 || !mobile.matrixScrollable) {
    throw new Error(`Preprocessing consequence mobile layout is invalid: ${JSON.stringify(mobile)}`);
  }
  const mobilePath = path.join(output, "preprocessing-consequence-mobile.png");
  await page.locator("#preprocessingConsequencePanel").screenshot({ path: mobilePath });

  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log(`Preprocessing Consequence Atlas viewer passed (${SERVE_ROOT === ROOT ? "source" : "dist"}; ${evidence.candidate_count} contracts -> ${evidence.unique_input_tensor_count} input / ${evidence.unique_output_tensor_set_count} output classes, ${evidence.output_changed_candidate_count} output-changing, ${evidence.top1_changed_candidate_count} raw top-1-changing, desktop/mobile overflow 0).`);
  console.log(`portfolio=${evidence.portfolio_ledger_sha256}`);
  console.log(`desktop=${desktopPath}`);
  console.log(`mobile=${mobilePath}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    consequenceStatus: document.querySelector("#preprocessingConsequenceStatus")?.textContent || null,
    panel: document.querySelector("#preprocessingConsequencePanel")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

function validateEvidence(evidence) {
  const expectedCandidates = new Map([
    ["raw_storage_rgb", [0, "0", 0, 0, "0", 0]],
    ["raw_storage_bgr", [0, "0", 0, 0, "0", 0]],
    ["artifact_affine_rgb", [0, "0", 0, 0, "0", 0]],
    ["center_128_div_128_rgb", [0, "0", 0, 0, "0", 0]],
    ["minus_one_to_one_rgb", [150_502, "150502", 1, 751, "1216", 5]],
    ["unit_interval_rgb", [18, "2304", 128, 250, "250", 1]],
    ["imagenet_mean_std_rgb", [150_503, "150503", 1, 769, "1217", 5]],
    ["imagenet_mean_std_bgr", [150_503, "150503", 1, 769, "1217", 5]],
  ]);
  if (evidence.schema !== "deepbom.preprocessing_consequence_atlas.v1" || evidence.method_version !== "2026-07-18.1"
    || evidence.evidence_class !== "MEASURED_SYNTHETIC" || evidence.status !== "assessed"
    || evidence.runtime?.version !== "2.5.2" || evidence.runtime?.backend !== "wasm"
    || evidence.candidate_count !== 8 || evidence.candidates?.length !== 8
    || evidence.execution_contract?.captured_repetitions_per_input !== 2
    || evidence.exact_source_contract_count !== 4 || evidence.non_exact_source_contract_count !== 4
    || evidence.unique_input_tensor_count !== 4 || evidence.unique_output_tensor_set_count !== 4
    || evidence.output_changed_candidate_count !== 4 || evidence.non_exact_output_changed_candidate_count !== 4
    || evidence.top1_changed_candidate_count !== 0 || evidence.maximum_output_changed_element_count !== 769
    || evidence.maximum_output_absolute_difference !== 5 || evidence.most_output_sensitive_contract_id !== "imagenet_mean_std_bgr"
    || evidence.baseline?.input_tensor_sha256 !== "89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035"
    || evidence.baseline?.output_tensor_set_sha256 !== "1abf2cca9690baec589df1f4e8120dc68bd1ea1839a24143d6b1f136a14ae5e6"
    || evidence.baseline?.first_output_top1_index !== 535
    || evidence.portfolio_ledger_sha256 !== "d51906882f2c5011c22b8883e3524b7e5c88d3c19e86a0a5171627d4558813e3"
    || !evidence.exact_contract_output_conservation
    || evidence.candidates.some((row) => {
      const expected = expectedCandidates.get(row.contract_id);
      return !expected || !row.deterministic_replay || row.first_output_top1_index !== 535
        || !/^[a-f0-9]{64}$/.test(row.candidate_ledger_sha256 || "")
        || row.input_changed_element_count !== expected[0]
        || row.input_total_absolute_code_difference_decimal !== expected[1]
        || row.input_maximum_absolute_code_difference !== expected[2]
        || row.output_changed_element_count !== expected[3]
        || row.output_total_absolute_difference_decimal !== expected[4]
        || row.output_maximum_absolute_difference !== expected[5];
    })) {
    throw new Error(`Preprocessing consequence evidence is inconsistent: ${JSON.stringify(evidence).slice(0, 5000)}`);
  }
  const classSizes = (rows) => rows.map((row) => row.candidate_count).sort((left, right) => right - left).join(",");
  if (classSizes(evidence.input_equivalence_classes) !== "4,2,1,1"
    || classSizes(evidence.output_equivalence_classes) !== "4,2,1,1") {
    throw new Error(`Preprocessing consequence equivalence classes changed: ${JSON.stringify({ input: evidence.input_equivalence_classes, output: evidence.output_equivalence_classes })}`);
  }
}

function buildReportEvidence(evidence) {
  initSync({ module: readFileSync(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
  const modelBytes = new Uint8Array(readFileSync(MODEL));
  const analysis = analyze_tflite_for_target(modelBytes, path.basename(MODEL), "android_mid_a55");
  const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
  if (modelSha256 !== evidence.artifact_sha256) throw new Error("Browser replay evidence is detached from the sample artifact bytes.");
  analysis.model_sha256 = modelSha256;
  const identity = {
    filename: analysis.filename,
    sha256: analysis.model_sha256,
    file_size_bytes: modelBytes.byteLength,
    format: analysis.format,
    target_profile_id: analysis.target_profile.id,
    target_label: analysis.target_profile.label,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs,
  };
  const sessionRuntime = { preprocessingConsequenceResult: evidence };
  const runtimeResults = buildRuntimeEvidence({ analysis, preprocessingConsequenceResult: evidence });
  const report = buildEngineeringReport(analysis, { identity, runtimeEvidence: sessionRuntime });
  const findings = buildFindingsRegister(analysis, { runtimeEvidence: sessionRuntime });
  const mlBom = buildMlBomDocument(analysis, {
    hash: analysis.model_sha256,
    fileSizeBytes: modelBytes.byteLength,
    target: analysis.target_profile,
    preprocessingConsequenceResult: evidence,
  });
  const staticAnalysis = buildStaticAnalysisExport(analysis);
  const quantization = buildQuantizationEvidence(analysis, identity);
  const conformance = buildConformanceReport({
    analysis,
    staticAnalysis,
    quantization,
    findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings },
    runtimeResults,
    securityPosture: { execution_integrity: {} },
    mlBomDocument: mlBom,
    engineeringReport: report,
  });
  const consequenceChecks = conformance.checks.filter((row) => row.id.startsWith("CF-PC-"));
  const properties = new Map(mlBom.metadata.component.properties.map((item) => [item.name, item.value]));
  const finding = findings.find((row) => row.finding_id === "EA-RUN-0003");
  const visual = visualPngSpecs({ analysis, preprocessingConsequenceResult: evidence }).some(([name]) => name === "visuals/preprocessing_consequence_atlas.png");
  const engineeringEvidence = buildEngineeringEvidenceDocument(analysis, {
    reportContext: { identity, runtimeEvidence: sessionRuntime },
    rawEvidenceContext: { identity, runtimeEvidence: sessionRuntime, findingsContext: { runtimeEvidence: sessionRuntime } },
    mlBomDocument: mlBom,
  });
  return {
    conformancePassed: consequenceChecks.length === 4 && consequenceChecks.every((row) => row.status === "pass"),
    findingPassed: evidence.output_changed_candidate_count
      ? finding?.evidence_class === "MEASURED_SYNTHETIC" && finding?.technical_priority === "Medium"
      : !finding,
    mlBomPassed: properties.get("deepbom:compatibility:detailLocation") === "engineering_evidence.json#/evidence/static_analysis"
      && !properties.has("deepbom:runtime:preprocessingConsequenceSchema")
      && !properties.has("deepbom:runtime:preprocessingConsequencePortfolioLedgerSha256")
      && (mlBom.metadata.component.externalReferences || []).some((item) => item.type === "evidence" && item.url === "engineering_evidence.json"),
    runtimePassed: runtimeResults.preprocessing_consequence_atlas?.portfolio_ledger_sha256 === evidence.portfolio_ledger_sha256
      && engineeringEvidence.evidence.runtime_results.preprocessing_consequence_atlas?.portfolio_ledger_sha256 === evidence.portfolio_ledger_sha256
      && report.includes("## Preprocessing Consequence Atlas (MEASURED_SYNTHETIC)")
      && report.includes(evidence.portfolio_ledger_sha256),
    visualPassed: visual,
  };
}

function panelState(browserPage) {
  return browserPage.locator("#preprocessingConsequencePanel").evaluate((panel) => {
    const canvas = panel.querySelector("canvas");
    const pixels = canvas?.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data || [];
    let canvasPixels = 0;
    for (let index = 0; index < pixels.length; index += 128) if (pixels[index] !== pixels[index + 1] || pixels[index + 1] !== pixels[index + 2]) canvasPixels += 1;
    const tables = panel.querySelectorAll("table");
    const selector = panel.querySelector(".consequence-select select");
    return {
      status: panel.querySelector("#preprocessingConsequenceStatus")?.textContent || "",
      metrics: panel.querySelectorAll(".consequence-metric").length,
      canvases: panel.querySelectorAll("canvas").length,
      canvasPixels,
      tables: tables.length,
      matrixRows: tables[1]?.querySelectorAll("tbody tr").length || 0,
      options: selector?.options.length || 0,
      runDisabled: panel.querySelector("#runPreprocessingConsequence")?.disabled,
      downloadDisabled: panel.querySelector("#downloadPreprocessingConsequence")?.disabled,
      text: panel.textContent.replace(/\s+/g, " ").trim(),
    };
  });
}

function dtypeBytes(dtype) {
  const value = String(dtype || "").toLowerCase();
  if (value.includes("64")) return 8;
  if (value.includes("32")) return 4;
  if (value.includes("16")) return 2;
  return 1;
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
