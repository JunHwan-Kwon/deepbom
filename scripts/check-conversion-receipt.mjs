import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";

import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildArtifactEvidenceEnvelope, validateArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import {
  bindConversionReceipt,
  buildConversionReceipt,
  validateBoundConversionReceipt,
  validateConversionReceipt,
} from "../web/lib/conversion-receipt.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";

const artifactPath = path.resolve("web/samples/sample_cnn_float.onnx");
const artifactBytes = await readFile(artifactPath);
const artifactSha256 = sha256(artifactBytes);
const activeArtifact = {
  filename: path.basename(artifactPath),
  format: "onnx",
  sha256: artifactSha256,
  byte_length_decimal: String(artifactBytes.length),
};
const receipt = buildConversionReceipt({
  created_at: "2026-09-04T00:00:00.000Z",
  source_artifacts: [{
    filename: "checkpoint.pth",
    format: "pytorch_state_dict",
    sha256: "1".repeat(64),
    byte_length_decimal: "1048576",
  }],
  converter: {
    name: "torch.onnx.export",
    version: "2.8.0",
    executable_sha256: "2".repeat(64),
    invocation: { argv: ["python", "export.py", "checkpoint.pth", "sample_cnn_float.onnx"] },
    environment: {
      manifest_sha256: "3".repeat(64),
      os: "linux",
      architecture: "x86_64",
      runtime: "python-3.13",
      container_image_digest: `sha256:${"4".repeat(64)}`,
    },
  },
  supporting_inputs: [{
    role: "export_script",
    filename: "export.py",
    sha256: "5".repeat(64),
    byte_length_decimal: "2048",
  }],
  output_artifact: activeArtifact,
  interpretation_boundary: "The receipt records declared conversion inputs and an output digest. It does not establish source-code safety, training provenance, converter execution, model accuracy, runtime assignment, or release readiness.",
});

assert.deepEqual(validateConversionReceipt(receipt), receipt, "receipt semantic and digest validation");
const bound = bindConversionReceipt(receipt, activeArtifact, { receiptFileSha256: "6".repeat(64) });
assert.deepEqual(validateBoundConversionReceipt(bound, activeArtifact), bound, "bound receipt semantic and digest validation");
assert.equal(bound.evidence_class, "DECLARED_UNVERIFIED", "converter claim evidence class");
assert.equal(bound.output_binding_evidence_class, "OBSERVED", "output binding evidence class");

const tampered = structuredClone(receipt);
tampered.converter.version = "tampered";
assert.throws(() => validateConversionReceipt(tampered), /SHA-256 is invalid/, "tampered receipt must fail closed");
assert.throws(() => bindConversionReceipt(receipt, { ...activeArtifact, sha256: "f".repeat(64) }), /not bound to the active artifact/, "wrong output digest must fail closed");
assert.throws(() => buildConversionReceipt({
  ...receipt,
  converter: { ...receipt.converter, invocation: { argv: ["converter", "--api-key=secret"] } },
}), /credential-bearing argument/, "credential-bearing invocation must be rejected");

const analysis = analyzeOnnxModel(new Uint8Array(artifactBytes), activeArtifact.filename);
analysis.model_sha256 = artifactSha256;
analysis.file_size_bytes = artifactBytes.length;
analysis.conversion_receipt = bound;
const context = getArtifactIrContext(analysis, {
  filename: activeArtifact.filename,
  format: activeArtifact.format,
  sha256: activeArtifact.sha256,
  size: artifactBytes.length,
});
assert.equal(context.artifact_ir.method_version, "2.2.0", "Artifact IR conversion-lineage method version");
assert.equal(context.artifact_ir.lineage_evidence.status, "output_bound_source_declared", "Artifact IR conversion lineage status");
assert.equal(context.artifact_ir.lineage_evidence.conversion.binding_sha256, bound.binding_sha256, "Artifact IR receipt binding conservation");

const envelope = buildArtifactEvidenceEnvelope(context.primary_view, {
  hash: artifactSha256,
  fileSizeBytes: artifactBytes.length,
  filename: activeArtifact.filename,
});
assert.equal(validateArtifactEvidenceEnvelope(envelope).valid, true, "evidence envelope conversion receipt validation");
assert.equal(envelope.conversion_receipt.binding_sha256, bound.binding_sha256, "evidence envelope receipt binding conservation");
const mlbom = buildMlBomDocument(context.primary_view, {
  hash: artifactSha256,
  fileSizeBytes: artifactBytes.length,
  artifactIr: context.artifact_ir,
  timestamp: "2026-09-04T00:00:00.000Z",
});
const properties = new Map(mlbom.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(properties.get("deepbom:model:conversionReceiptSha256"), receipt.receipt_sha256, "CycloneDX receipt identity");
assert.equal(properties.get("deepbom:model:conversionReceiptBindingSha256"), bound.binding_sha256, "CycloneDX receipt binding identity");
assert.equal(properties.get("deepbom:model:conversionEvidenceClass"), "DECLARED_UNVERIFIED", "CycloneDX converter evidence boundary");
assert.equal(properties.get("deepbom:model:conversionOutputBindingEvidenceClass"), "OBSERVED", "CycloneDX output binding evidence class");

const temporaryDirectory = await mkdtemp(path.join(tmpdir(), "deepbom-conversion-receipt-"));
try {
  const receiptPath = path.join(temporaryDirectory, "conversion-receipt.json");
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
  const cliAnalysis = runCli(["audit", artifactPath, "--conversion-receipt", receiptPath, "--compact"]);
  assert.equal(cliAnalysis.conversion_receipt.receipt_sha256, receipt.receipt_sha256, "CLI analysis receipt identity");
  assert.equal(cliAnalysis.conversion_receipt.active_artifact.sha256, artifactSha256, "CLI observed output binding");
  const cliEnvelope = runCli(["audit", artifactPath, "--conversion-receipt", receiptPath, "--format", "envelope", "--compact"]);
  assert.equal(cliEnvelope.conversion_receipt.binding_sha256, cliAnalysis.conversion_receipt.binding_sha256, "CLI envelope receipt binding conservation");
  const cliCycloneDx = runCli(["audit", artifactPath, "--conversion-receipt", receiptPath, "--format", "cyclonedx", "--timestamp", "2026-09-04T00:00:00.000Z", "--compact"]);
  const cliProperties = new Map(cliCycloneDx.metadata.component.properties.map((row) => [row.name, row.value]));
  assert.equal(cliProperties.get("deepbom:model:conversionReceiptBindingSha256"), cliAnalysis.conversion_receipt.binding_sha256, "CLI CycloneDX receipt binding conservation");
  const cliGraph = runCli(["graph", artifactPath, "--conversion-receipt", receiptPath, "--format", "json"]);
  assert.equal(cliGraph.artifact_ir.lineage_evidence.conversion.binding_sha256, cliAnalysis.conversion_receipt.binding_sha256, "CLI Graph IR receipt binding conservation");

  const wrongReceipt = buildConversionReceipt({ ...receipt, output_artifact: { ...receipt.output_artifact, sha256: "e".repeat(64) } });
  const wrongPath = path.join(temporaryDirectory, "wrong-receipt.json");
  await writeFile(wrongPath, JSON.stringify(wrongReceipt), "utf8");
  const rejected = spawnSync(process.execPath, ["bin/deepbom.mjs", "audit", artifactPath, "--conversion-receipt", wrongPath, "--compact"], { encoding: "utf8" });
  assert.notEqual(rejected.status, 0, "CLI must reject a receipt for another output artifact");
  assert.match(rejected.stderr, /not bound to the active artifact/, "CLI wrong-output rejection reason");
  await verifyBrowserReceipt(receiptPath, wrongPath);
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}

console.log("Conversion receipt contract passed (module, IR, envelope, CycloneDX, CLI/browser binding, tamper rejection).");

function runCli(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  assert.equal(result.status, 0, `CLI failed: ${result.stderr}`);
  return JSON.parse(result.stdout);
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function verifyBrowserReceipt(receiptPath, wrongPath) {
  const server = createStaticServer(path.resolve("."));
  const browserErrors = [];
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    browser = await launchChromium(chromium);
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
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
    await page.locator("#fileInput").setInputFiles(artifactPath);
    await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled, null, { timeout: 30_000 });
    await page.locator("#runAudit").click();
    await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
    assert.match(await page.locator("#status").textContent(), /audit run complete/i, "browser audit completion");

    await page.locator("#conversionReceiptInput").setInputFiles(receiptPath);
    await page.waitForFunction(() => document.querySelector("#conversionReceiptStatus")?.textContent?.startsWith("Output-bound receipt:"), null, { timeout: 30_000 });
    assert.match(await page.locator("#conversionReceiptStatus").textContent(), /Output-bound receipt:/, "browser receipt binding");

    await page.locator("#conversionReceiptInput").setInputFiles(wrongPath);
    await page.waitForFunction(() => document.querySelector("#conversionReceiptStatus")?.textContent?.startsWith("Receipt rejected:"), null, { timeout: 30_000 });
    assert.match(await page.locator("#conversionReceiptStatus").textContent(), /not bound to the active artifact/, "browser wrong-output rejection");

    await page.locator("#conversionReceiptInput").setInputFiles(receiptPath);
    await page.waitForFunction(() => document.querySelector("#conversionReceiptStatus")?.textContent?.startsWith("Output-bound receipt:"), null, { timeout: 30_000 });
    assert.equal(browserErrors.length, 0, `browser conversion receipt diagnostics: ${browserErrors.join("\n")}`);
  } finally {
    await browser?.close().catch(() => {});
    await new Promise((resolve) => server.close(resolve));
  }
}

function createStaticServer(root) {
  const mime = new Map([
    [".html", "text/html; charset=utf-8"], [".js", "text/javascript; charset=utf-8"], [".mjs", "text/javascript; charset=utf-8"],
    [".css", "text/css; charset=utf-8"], [".wasm", "application/wasm"], [".json", "application/json; charset=utf-8"],
    [".onnx", "application/octet-stream"], [".png", "image/png"], [".svg", "image/svg+xml"],
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
