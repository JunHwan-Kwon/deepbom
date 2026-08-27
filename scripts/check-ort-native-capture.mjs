import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildOnnxRuntimeShapeBinding } from "../web/lib/onnx-runtime-shape-binding.js";
import { buildRuntimeDataMovementEvidence } from "../web/lib/runtime-data-movement-evidence.js";
import { initSync, analyze_deepbom } from "../web/protected/deepbom/pkg/deepbom_wasm.js";
import { applyProtectedOrtCompatibilityEvidence } from "../web/lib/ort-compatibility-evidence.js";
import {
  buildOrtRuntimeAssignmentDocument,
  parseRuntimeProfileSource,
  previewOrtProfileMapping,
  verifyOrtNativeCaptureProfile,
} from "../web/lib/runtime-profile-adapter.js";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";
import { capturePinnedOrtProfiles, verifyOrtNativeCapturePackage } from "./ort-native-capture-lib.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
initSync({ module: readFileSync(path.join(ROOT, "web", "protected", "deepbom", "pkg", "deepbom_wasm_bg.wasm")) });
const MODEL = path.join(ROOT, "web", "samples", "sample_cnn_float.onnx");
const EXTERNAL_MODEL_BASE64 = "CA0SGGRlZXBib21fZXh0ZXJuYWxfZml4dHVyZTqRAQoTCgF4CgF3EgF5GgNhZGQiA0FkZBIQZXh0ZXJuYWxfZml4dHVyZSo+CAEIAhABQgF3ahcKCGxvY2F0aW9uEgt3ZWlnaHRzLmJpbmoLCgZvZmZzZXQSATBqCwoGbGVuZ3RoEgE4cAFaEwoBeBIOCgwIARIICgIIAQoCCAJiEwoBeRIOCgwIARIICgIIAQoCCAJCBAoAEA0=";
const EXTERNAL_WEIGHTS_BASE64 = "AACgPwAAIMA=";
const output = await mkdtemp(path.join(tmpdir(), "deepbom-ort-native-e2e-"));
const captureDir = path.join(output, "capture");
const externalModelDir = path.join(output, "external-model");
const externalModelPath = path.join(externalModelDir, "external.onnx");
const externalWeightsPath = path.join(externalModelDir, "weights.bin");
const externalCaptureDir = path.join(output, "external-capture");
const reducedConfigPath = path.join(output, "required_operators.config");
const server = createStaticServer(ROOT);
const browserErrors = [];
let browser;
let page;

try {
  await mkdir(externalModelDir, { recursive: true });
  const externalModelBytes = Buffer.from(EXTERNAL_MODEL_BASE64, "base64");
  const externalWeightsBytes = Buffer.from(EXTERNAL_WEIGHTS_BASE64, "base64");
  await writeFile(externalModelPath, externalModelBytes);
  await writeFile(externalWeightsPath, externalWeightsBytes);
  await writeFile(reducedConfigPath, "ai.onnx;13;Add\n", "utf8");

  const capture = await capturePinnedOrtProfiles({ artifactPath: MODEL, outputDir: captureDir, providers: ["cpu"], runs: 2, warmupRuns: 0 });
  await verifyOrtNativeCapturePackage(captureDir, { artifactPath: MODEL });
  assert.equal(capture.index.profiles.length, 2);
  assert.equal(capture.index.runtime.provider_inventory_status, "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS");
  assert.equal(capture.index.runtime.reduced_operator_inventory_status, "NOT_EXPOSED_BY_ONNXRUNTIME_NODE_API_NOT_INFERRED");
  assert.match(capture.index.runtime.supported_backends_sha256, /^[a-f0-9]{64}$/);
  assert(capture.index.runtime.supported_backends.some((backend) => backend.name === "cpu" && backend.bundled), "Pinned native package must expose its bundled CPU backend.");
  assert.equal(capture.index.paired_profile_runtime_graph.schema, "deepbom.ort_paired_runtime_graph.v1.1");
  assert.equal(capture.index.paired_profile_runtime_graph.profiles.find((item) => item.role === "identity")?.runtime_node_count, 9);
  assert.equal(capture.index.paired_profile_runtime_graph.profiles.find((item) => item.role === "production")?.runtime_node_count, 8);
  assert.equal(capture.index.paired_profile_output_comparison.status, "assessed");

  const modelBytes = await readFile(MODEL);
  const analysis = analyzeOnnxModel(new Uint8Array(modelBytes), path.basename(MODEL));
  applyProtectedOrtCompatibilityEvidence(analysis, analyze_deepbom(new Uint8Array(modelBytes), JSON.stringify(analysis)).ort_compatibility_evidence);
  analysis.model_sha256 = sha256(modelBytes);
  analysis.target_profile = { id: "wasm_simd", profile_sha256: "b".repeat(64) };
  const identityPath = path.join(captureDir, "identity.deepbom-ort-profile.json");
  const productionPath = path.join(captureDir, "production.deepbom-ort-profile.json");
  const identityBytes = await readFile(identityPath);
  const identityProfile = parseRuntimeProfileSource(identityBytes.toString("utf8"), analysis);
  const verifiedIdentity = await verifyOrtNativeCaptureProfile(identityProfile, analysis);
  const identityPreview = previewOrtProfileMapping(identityProfile, analysis, verifiedIdentity.metadata);
  assert.equal(identityPreview.assignment_count, 9);
  assert.equal(identityPreview.mapped_kernel_event_count, 18);
  assert.equal(identityPreview.unresolved_runtime_node_count, 0);
  assert.equal(identityPreview.runtime_tensor_observation_count, 9);
  const assignment = buildOrtRuntimeAssignmentDocument(identityProfile, analysis, {
    ...verifiedIdentity.metadata,
    profileSha256: verifiedIdentity.profileSha256,
    nativeCaptureEvidence: verifiedIdentity.evidence,
  });
  const normalized = parseRuntimeAssignmentDocument(JSON.stringify(assignment), analysis, { fileSha256: sha256(identityBytes) });
  assert.equal(normalized.source.adapter.schema, "deepbom.ort_profile_adapter.v2.2");
  assert.equal(normalized.source.adapter.runtime_tensor_observation_count, 9);
  assert.equal(normalized.source.adapter.native_capture.profile_role, "identity");
  assert.equal(normalized.source.adapter.native_capture.selected_build_provider_binding.provider_inventory_status, "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS");
  const directmlBinding = normalized.source.adapter.native_capture.selected_build_provider_binding.bindings.find((row) => row.backend_name === "dml");
  if (directmlBinding) assert.equal(directmlBinding.source_profile, "directml");
  assert.equal(normalized.source.adapter.native_capture.paired_profile_runtime_graph.profiles.find((item) => item.role === "production")?.runtime_node_count, 8);
  const shapeBinding = buildOnnxRuntimeShapeBinding(analysis, normalized);
  assert.equal(shapeBinding.conflict_count, 0);
  assert.ok(shapeBinding.observed_internal_tensor_count > 0);
  const movementEvidence = buildRuntimeDataMovementEvidence(normalized);
  assert.equal(movementEvidence.status, "observed_no_profiled_copy_nodes_for_captured_configuration");
  assert.equal(movementEvidence.observed_copy_event_payload_bytes.decimal, "0");
  assert.equal(movementEvidence.physical_transfer_bytes, null);
  const nativeRuntimeReport = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: normalized }, [], analysis);
  assert.match(nativeRuntimeReport, new RegExp(`artifact content-set SHA-256 ${capture.index.artifact.content_set_sha256}`));
  assert.match(nativeRuntimeReport, /external files 0, tensor ranges 0/);
  assert.match(nativeRuntimeReport, /ONNX Runtime Selected-Build Provider Binding/);
  assert.match(nativeRuntimeReport, /ONNX Runtime Internal Shape And Cost Binding/);
  assert.match(nativeRuntimeReport, /Runtime Copy-Node Data Movement/);
  assert.match(nativeRuntimeReport, /not_exposed_by_ort_profile/);

  const productionProfile = parseRuntimeProfileSource((await readFile(productionPath)).toString("utf8"), analysis);
  const verifiedProduction = await verifyOrtNativeCaptureProfile(productionProfile, analysis);
  const productionPreview = previewOrtProfileMapping(productionProfile, analysis, verifiedProduction.metadata);
  assert.equal(productionPreview.assignment_count, 0, "optimized unnamed runtime nodes must not be assigned to original ops");
  assert.equal(productionPreview.unresolved_runtime_node_count, 8);
  assert.throws(
    () => buildOrtRuntimeAssignmentDocument(productionProfile, analysis, { ...verifiedProduction.metadata, profileSha256: verifiedProduction.profileSha256, nativeCaptureEvidence: verifiedProduction.evidence }),
    /No ONNX Runtime profile node can be bound deterministically/,
  );

  const tamperedEnvelope = JSON.parse(identityBytes.toString("utf8"));
  tamperedEnvelope.profile.json += " ";
  const tamperedProfile = parseRuntimeProfileSource(JSON.stringify(tamperedEnvelope), analysis);
  await assert.rejects(verifyOrtNativeCaptureProfile(tamperedProfile, analysis), /envelope content SHA-256 is invalid/);
  const wrongArtifact = { ...analysis, model_sha256: "a".repeat(64) };
  await assert.rejects(verifyOrtNativeCaptureProfile(identityProfile, wrongArtifact), /not bound to the active ONNX artifact/);

  const externalCapture = await capturePinnedOrtProfiles({ artifactPath: externalModelPath, outputDir: externalCaptureDir, providers: ["cpu"], runs: 1, warmupRuns: 0, reducedOperatorConfigPath: reducedConfigPath });
  await verifyOrtNativeCapturePackage(externalCaptureDir, { artifactPath: externalModelPath });
  assert.equal(externalCapture.index.artifact.external_data.status, "verified_payloads");
  assert.equal(externalCapture.index.artifact.external_data.tensor_count, 1);
  assert.equal(externalCapture.index.artifact.external_data.verified_payload_bytes, 8);
  assert.equal(externalCapture.index.artifact.external_data.files[0].path, "weights.bin");
  assert.equal(externalCapture.index.artifact.external_data.files[0].sha256, sha256(externalWeightsBytes));
  assert.match(externalCapture.index.artifact.external_data.ledger_sha256, /^[a-f0-9]{64}$/);
  assert.match(externalCapture.index.artifact.content_set_sha256, /^[a-f0-9]{64}$/);
  assert.equal(externalCapture.index.runtime.reduced_operator_inventory_status, "IMPORTED_CONFIG_NOT_BINARY_ATTESTED");
  assert.equal(externalCapture.index.runtime.reduced_operator_config.normalized_config.operator_identity_count, 1);
  assert.equal(externalCapture.index.runtime.reduced_operator_config.binary_binding_status, "NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY");
  await assert.rejects(readFile(path.join(externalCaptureDir, ".artifact-snapshot", "external.onnx")), /ENOENT/);

  const externalAnalysis = analyzeOnnxModel(new Uint8Array(externalModelBytes), path.basename(externalModelPath), null, {
    externalDataFiles: [{
      path: "weights.bin",
      bytes: new Uint8Array(externalWeightsBytes),
      sha256: sha256(externalWeightsBytes),
      sha1: sha1(externalWeightsBytes),
    }],
  });
  applyProtectedOrtCompatibilityEvidence(externalAnalysis, analyze_deepbom(new Uint8Array(externalModelBytes), JSON.stringify(externalAnalysis)).ort_compatibility_evidence);
  externalAnalysis.model_sha256 = sha256(externalModelBytes);
  externalAnalysis.target_profile = { id: "wasm_simd", profile_sha256: "b".repeat(64) };
  const externalIdentityPath = path.join(externalCaptureDir, "identity.deepbom-ort-profile.json");
  const externalIdentityBytes = await readFile(externalIdentityPath);
  const externalIdentityProfile = parseRuntimeProfileSource(externalIdentityBytes.toString("utf8"), externalAnalysis);
  const verifiedExternalIdentity = await verifyOrtNativeCaptureProfile(externalIdentityProfile, externalAnalysis);
  assert.equal(verifiedExternalIdentity.evidence.artifact_binding, "BROWSER_VERIFIED_ACTIVE_ONNX_AND_EXTERNAL_DATA_CONTENT_SET_SHA256");
  assert.equal(verifiedExternalIdentity.evidence.artifact.external_data.files[0].sha256, sha256(externalWeightsBytes));
  assert.equal(verifiedExternalIdentity.evidence.selected_build_provider_binding.reduced_operator_assessment.status, "compatible_operator_identity");
  assert.equal(verifiedExternalIdentity.evidence.selected_build_provider_binding.reduced_operator_assessment.included_node_count, 1);
  const externalAssignment = buildOrtRuntimeAssignmentDocument(externalIdentityProfile, externalAnalysis, {
    ...verifiedExternalIdentity.metadata,
    profileSha256: verifiedExternalIdentity.profileSha256,
    nativeCaptureEvidence: verifiedExternalIdentity.evidence,
  });
  const parsedExternalAssignment = parseRuntimeAssignmentDocument(JSON.stringify(externalAssignment), externalAnalysis, { fileSha256: sha256(Buffer.from(JSON.stringify(externalAssignment))) });
  const externalRuntimeReport = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: parsedExternalAssignment });
  assert.match(externalRuntimeReport, new RegExp(`artifact content-set SHA-256 ${externalCapture.index.artifact.content_set_sha256}`));
  assert.match(externalRuntimeReport, /external files 1, tensor ranges 1/);
  assert.match(externalRuntimeReport, /IMPORTED_CONFIG_NOT_BINARY_ATTESTED/);
  assert.match(externalRuntimeReport, /NOT_ATTESTED_CONFIG_INPUT_NOT_OBSERVED_FROM_SELECTED_BINARY/);
  const tamperedExternalAssignment = structuredClone(externalAssignment);
  tamperedExternalAssignment.source.adapter.native_capture.artifact.external_data.files[0].sha256 = "d".repeat(64);
  assert.throws(() => parseRuntimeAssignmentDocument(JSON.stringify(tamperedExternalAssignment), externalAnalysis, { fileSha256: "e".repeat(64) }), /external-data ledger does not match/);
  const wrongSidecarAnalysis = structuredClone(externalAnalysis);
  wrongSidecarAnalysis.onnx_external_data.supplied_files[0].sha256 = "c".repeat(64);
  await assert.rejects(verifyOrtNativeCaptureProfile(externalIdentityProfile, wrongSidecarAnalysis), /file ledger is invalid|tensor-range ledger is invalid|file\/range ledger does not match/);
  const mutatedWeights = Buffer.from(externalWeightsBytes);
  mutatedWeights[0] ^= 0xff;
  await writeFile(externalWeightsPath, mutatedWeights);
  await assert.rejects(verifyOrtNativeCapturePackage(externalCaptureDir, { artifactPath: externalModelPath }), /content-set identity does not match/);
  await writeFile(externalWeightsPath, externalWeightsBytes);
  await verifyOrtNativeCapturePackage(externalCaptureDir, { artifactPath: externalModelPath });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) browserErrors.push(`console: ${message.text()}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 60_000 });
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
  }
  await page.locator("#targetSelect").evaluate((select) => {
    select.value = "wasm_simd";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.locator("#fileInput").setInputFiles(MODEL);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();
  await page.locator("#runtimeAssignmentInput").setInputFiles(identityPath);
  await page.locator("#runtimeProfileBackdrop").waitFor({ state: "visible" });
  const modal = await page.evaluate(() => ({
    title: document.querySelector("#runtimeProfileTitle")?.textContent,
    version: document.querySelector("#runtimeProfileVersion")?.value,
    versionReadOnly: document.querySelector("#runtimeProfileVersion")?.readOnly,
    backend: document.querySelector("#runtimeProfileBackend")?.value,
    optimization: document.querySelector("#runtimeProfileOptimization")?.value,
    optimizationDisabled: document.querySelector("#runtimeProfileOptimization")?.disabled,
    execution: document.querySelector("#runtimeProfileExecutionMode")?.value,
    executionDisabled: document.querySelector("#runtimeProfileExecutionMode")?.disabled,
    binarySha: document.querySelector("#runtimeProfileBinarySha")?.value,
    build: document.querySelector("#runtimeProfileBuild")?.value,
    capture: document.querySelector("#runtimeProfileCapture")?.value,
    captureHidden: document.querySelector("#runtimeProfileCapture")?.closest("label")?.hidden,
    preview: [...document.querySelectorAll(".runtime-profile-preview-item strong")].map((item) => item.textContent),
    importDisabled: document.querySelector("#runtimeProfileImport")?.disabled,
  }));
  if (modal.title !== "Pinned native ONNX Runtime profile" || modal.version !== "1.26.0" || !modal.versionReadOnly
    || modal.backend !== "CPUExecutionProvider" || modal.optimization !== "disabled" || !modal.optimizationDisabled
    || modal.execution !== "sequential" || !modal.executionDisabled || !/^[a-f0-9]{64}$/.test(modal.binarySha)
    || !modal.build.includes("onnxruntime-node@1.26.0") || modal.capture !== capture.index.capture_id || modal.captureHidden
    || modal.preview[0] !== "9/9" || modal.preview[1] !== "18/18" || modal.importDisabled) {
    throw new Error(`Pinned native ORT modal binding failed: ${JSON.stringify(modal)}`);
  }
  const modalDesktop = path.join(output, "ort-native-modal-desktop.png");
  await page.locator("#runtimeProfileBackdrop").screenshot({ path: modalDesktop });
  await page.locator("#runtimeProfileImport").click();
  await page.waitForFunction(() => document.querySelector("#runtimeAssignmentStatus")?.textContent?.includes("production transformed graph 8 observed node(s)"), null, { timeout: 30_000 });
  const viewerText = await page.locator("#runtimeAssignmentComparison").textContent();
  if (!viewerText?.includes("Observed production ORT runtime graph") || !viewerText.includes("r1_nchwc")
    || !viewerText.includes("ReorderOutput") || !viewerText.includes("relative L2") || !viewerText.includes("original-op mapping is not inferred")) {
    throw new Error("Pinned native ORT production graph/output comparison is missing from the viewer.");
  }
  const viewerDesktop = path.join(output, "ort-native-viewer-desktop.png");
  await page.locator("#runtimeAssignmentComparison").screenshot({ path: viewerDesktop });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#runtimeAssignmentComparison").scrollIntoViewIfNeeded();
  const mobile = await page.evaluate(() => ({
    bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
    panelOverflow: Math.max(0, (document.querySelector("#runtimeAssignmentComparison")?.scrollWidth || 0) - (document.querySelector("#runtimeAssignmentComparison")?.clientWidth || 0)),
  }));
  const viewerMobile = path.join(output, "ort-native-viewer-mobile.png");
  await page.locator("#runtimeAssignmentComparison").screenshot({ path: viewerMobile });
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1) throw new Error(`Pinned native ORT viewer mobile overflow: ${JSON.stringify(mobile)}`);

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.locator('[data-workflow-step="input"]').click();
  await page.locator("#fileInput").setInputFiles(externalModelPath);
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
  await page.locator("#onnxExternalDataInput").setInputFiles(externalWeightsPath);
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("ONNX external data ready"), null, { timeout: 30_000 });
  await page.locator("#runAudit").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("audit run complete"), null, { timeout: 60_000 });
  const externalStatus = await page.locator("#onnxExternalDataStatus").textContent();
  if (!externalStatus?.includes("1/1 range(s) verified")) throw new Error(`External-data audit status is incomplete: ${externalStatus}`);
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator('[data-explorer-tab="kernels"]').click();
  await page.locator("#runtimeAssignmentInput").setInputFiles(externalIdentityPath);
  await page.locator("#runtimeProfileBackdrop").waitFor({ state: "visible" });
  const externalModal = await page.evaluate(() => ({
    build: document.querySelector("#runtimeProfileBuild")?.value,
    preview: [...document.querySelectorAll(".runtime-profile-preview-item strong")].map((item) => item.textContent),
    importDisabled: document.querySelector("#runtimeProfileImport")?.disabled,
  }));
  if (!externalModal.build?.includes(`artifact-content-set ${externalCapture.index.artifact.content_set_sha256}`)
    || externalModal.preview[0] !== "1/1" || externalModal.preview[1] !== "1/1" || externalModal.importDisabled) {
    throw new Error(`External-data native ORT modal binding failed: ${JSON.stringify(externalModal)}`);
  }
  await page.locator("#runtimeProfileImport").click();
  await page.waitForFunction(() => document.querySelector("#runtimeAssignmentStatus")?.textContent?.includes("1/1 rows"), null, { timeout: 30_000 });
  if (browserErrors.length) throw new Error(`Browser errors:\n${browserErrors.join("\n")}`);
  console.log("Pinned native ORT capture passed (dual profiles, package/binary hashes, immutable external-data snapshot, full content-set tamper rejection, 9/9 and 1/1 identity mapping, production ledger, paired output deltas, desktop/mobile overflow 0).");
  console.log(`desktop_modal=${modalDesktop}`);
  console.log(`desktop_viewer=${viewerDesktop}`);
  console.log(`mobile_viewer=${viewerMobile}`);
} catch (error) {
  const state = await page?.evaluate(() => ({
    status: document.querySelector("#status")?.textContent || null,
    runtimeStatus: document.querySelector("#runtimeAssignmentStatus")?.textContent || null,
    modalStatus: document.querySelector("#runtimeProfileStatus")?.textContent || null,
  })).catch(() => null);
  throw new Error(`${error.message}\nstate=${JSON.stringify(state)}\nbrowser_errors=${JSON.stringify(browserErrors)}`);
} finally {
  await browser?.close().catch(() => {});
  if (server.listening) await new Promise((resolve) => server.close(resolve));
  const resolvedCapture = path.resolve(captureDir);
  if (!resolvedCapture.startsWith(`${path.resolve(output)}${path.sep}`)) throw new Error(`Refusing to remove unexpected capture path ${resolvedCapture}`);
  await rm(captureDir, { recursive: true, force: true });
  if (process.exitCode) await rm(output, { recursive: true, force: true });
}

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function sha1(value) { return createHash("sha1").update(value).digest("hex"); }

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

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({ ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" })[path.extname(file).toLowerCase()] || "application/octet-stream";
}
