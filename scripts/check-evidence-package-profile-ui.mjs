import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

import { launchChromium } from "./browser-launch.mjs";
import { parseStoredZip } from "./verify-package.mjs";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const decoder = new TextDecoder();
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, "http://127.0.0.1").pathname;
    if (pathname === "/favicon.ico") return response.writeHead(204).end();
    const relative = pathname === "/" ? "/web/index.html" : pathname;
    const resolved = path.resolve(ROOT, `.${decodeURIComponent(relative)}`);
    if (!resolved.startsWith(`${ROOT}${path.sep}`)) throw new Error("outside root");
    const bytes = await readFile(resolved);
    const type = relative.endsWith(".html") ? "text/html; charset=utf-8"
      : relative.endsWith(".css") ? "text/css; charset=utf-8"
        : relative.endsWith(".js") || relative.endsWith(".mjs") ? "text/javascript; charset=utf-8"
          : relative.endsWith(".json") ? "application/json; charset=utf-8"
            : relative.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
    response.writeHead(200, { "content-type": type });
    response.end(bytes);
  } catch {
    response.writeHead(404).end("not found");
  }
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) errors.push(message.text());
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const pathname = new URL(response.url()).pathname;
    if (!pathname.startsWith("/api/")) errors.push(`HTTP ${response.status()} ${pathname}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/index.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.childElementCount > 0);
  if (await page.locator("#agreementBackdrop").isVisible()) {
    await page.locator("#privacyAgree").check();
    await page.locator("#acceptAgreement").click();
    await page.locator("#agreementBackdrop").waitFor({ state: "hidden" });
  }
  await page.locator("#sampleModelSelect").selectOption("gguf-tinymqa-q4");
  await page.locator("#trySampleModel").click();
  await page.waitForFunction(() => document.body.dataset.modelFormat === "gguf"
    && /audit run complete|audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  assert.doesNotMatch(await page.locator("#status").innerText(), /audit failed/i);
  await page.locator('[data-workflow-step="output"]').click();
  await page.locator("#downloadPublicBundle:not([disabled])").waitFor({ timeout: 20_000 });

  await page.locator(".report-export-panel > summary").click();
  await page.locator("#downloadMarkdown:not([disabled])").waitFor({ timeout: 20_000 });
  assert.match(await page.locator("#reportPreview").innerText(), /^# DEEPBOM .+ Audit/m);
  const [engineeringReportDownload] = await Promise.all([
    page.waitForEvent("download", { timeout: 120_000 }),
    page.locator("#downloadMarkdown").click(),
  ]);
  const engineeringReportHtml = await readFile(await engineeringReportDownload.path(), "utf8");
  assert.match(engineeringReportHtml, /DEEPBOM ENGINEERING COPY/);
  assert.match(engineeringReportHtml, /Report-body SHA-256: [a-f0-9]{64}/i);
  assert.match(engineeringReportHtml, /without an independently trusted key/i);
  assert.equal(await page.locator("#authBackdrop").isVisible(), false, "Engineering Report must not open authentication");
  assert.equal(await page.locator("#downloadRawData").isDisabled(), true, "Raw evidence must remain separately controlled");

  for (const profile of ["public", "engineering", "regulatory", "machine_readable"]) {
    await page.locator("#evidencePackageProfile").selectOption(profile);
    for (const level of ["artifact_facts", "deterministic", "planning", "all_available"]) {
      await page.locator("#evidencePackageLevel").selectOption(level);
      const note = await page.locator("#engineeringBundleNote").innerText();
      assert.match(note, /No sign-in is required/i, `${profile}/${level} login-free note`);
      const [download] = await Promise.all([
        page.waitForEvent("download", { timeout: 120_000 }),
        page.locator("#downloadPublicBundle").click(),
      ]);
      const zipBytes = await readFile(await download.path());
      const members = parseStoredZip(zipBytes);
      assert(members.has("deepbom_public_key_signature.json"), `${profile}/${level} detached package signature`);
      assert(members.has("RIGHTS.txt"), `${profile}/${level} rights notice`);
      assert(members.has("evidence_level_manifest.json"), `${profile}/${level} level manifest`);
      assert.match(decoder.decode(members.get("RIGHTS.txt")), /All rights reserved/i, `${profile}/${level} copyright notice`);
      assert.match(decoder.decode(members.get("RIGHTS.txt")), /removal or concealment of attribution/i, `${profile}/${level} provenance concealment boundary`);
      assert([...members.keys()].every((name) => !/\.(tflite|onnx|gguf|safetensors|mlmodel)$/i.test(name)), `${profile}/${level} excludes model payload`);
      const scope = JSON.parse(decoder.decode(members.get("package_scope.json")));
      const levelManifest = JSON.parse(decoder.decode(members.get("evidence_level_manifest.json")));
      assert.equal(scope.profile, profile);
      assert.equal(scope.evidence_level, level);
      assert.equal(levelManifest.requested_level, level);
      assert.equal([...members.keys()].filter((name) => name.endsWith(".cdx.json")).length, level === "all_available" ? 2 : 0);
      assert.equal(scope.login_required, false);
      assert.match(scope.integrity_boundary, /changes detectable/i);
      assert.match(scope.integrity_boundary, /do not prevent copying or editing/i);
      if (profile === "public") assert(members.has("public_report_verification_manifest.json"));
      assert.match(download.suggestedFilename(), new RegExp(`deepbom_${profile}_${level}_evidence_package`, "i"));
      assert.equal(await page.locator("#authBackdrop").isVisible(), false, `${profile}/${level} must not open authentication`);
    }
  }

  await page.setViewportSize({ width: 390, height: 844 });
  const mobile = await page.locator(".bundle-panel").evaluate((panel) => ({
    overflow: Math.max(0, document.documentElement.scrollWidth - innerWidth),
    selectHeight: panel.querySelector("#evidencePackageProfile")?.getBoundingClientRect().height || 0,
    levelHeight: panel.querySelector("#evidencePackageLevel")?.getBoundingClientRect().height || 0,
    buttonHeight: panel.querySelector("#downloadPublicBundle")?.getBoundingClientRect().height || 0,
    panelWidth: panel.getBoundingClientRect().width,
    viewport: innerWidth,
  }));
  assert.equal(mobile.overflow, 0, `mobile Evidence Package overflow ${JSON.stringify(mobile)}`);
  assert(mobile.selectHeight >= 40 && mobile.levelHeight >= 40 && mobile.buttonHeight >= 44, `mobile Evidence Package targets ${JSON.stringify(mobile)}`);
  assert(mobile.panelWidth <= mobile.viewport + 0.5, `mobile Evidence Package panel width ${JSON.stringify(mobile)}`);

  const medicalPage = await browser.newPage({ viewport: { width: 390, height: 844 }, acceptDownloads: true });
  medicalPage.on("pageerror", (error) => errors.push(`medical: ${error.message}`));
  await medicalPage.goto(`http://127.0.0.1:${server.address().port}/web/index.html?surface=medical`, { waitUntil: "domcontentloaded" });
  await medicalPage.waitForFunction(() => document.querySelector("#sampleEvidenceGlance")?.childElementCount > 0);
  if (await medicalPage.locator("#agreementBackdrop").isVisible()) {
    await medicalPage.locator("#privacyAgree").check();
    await medicalPage.locator("#acceptAgreement").click();
    await medicalPage.locator("#agreementBackdrop").waitFor({ state: "hidden" });
  }
  await medicalPage.locator("#sampleModelSelect").selectOption("gguf-tinymqa-q4");
  await medicalPage.locator("#trySampleModel").click();
  await medicalPage.waitForFunction(() => document.body.dataset.modelFormat === "gguf"
    && /audit run complete|audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120_000 });
  assert.doesNotMatch(await medicalPage.locator("#status").innerText(), /audit failed/i);
  await medicalPage.locator('[data-workflow-step="output"]').click();
  await medicalPage.locator('[data-module-tab="regulatory_report"]').click();
  await medicalPage.locator("#downloadRegulatoryReport:not([disabled])").waitFor({ timeout: 20_000 });
  const [regulatoryReportDownload] = await Promise.all([
    medicalPage.waitForEvent("download", { timeout: 120_000 }),
    medicalPage.locator("#downloadRegulatoryReport").click(),
  ]);
  const regulatoryReportHtml = await readFile(await regulatoryReportDownload.path(), "utf8");
  assert.match(regulatoryReportHtml, /DEEPBOM REGULATORY SUPPORT/);
  assert.match(regulatoryReportHtml, /Report-body SHA-256: [a-f0-9]{64}/i);
  assert.match(regulatoryReportHtml, /not a regulatory submission/i);
  assert.equal(await medicalPage.locator("#authBackdrop").isVisible(), false, "Regulatory Support Report must not open authentication");
  assert.equal(await medicalPage.locator("#downloadEvidenceBundle").isDisabled(), true, "Full regulatory bundle must remain separately controlled");
  const medicalOverflow = await medicalPage.evaluate(() => Math.max(0, document.documentElement.scrollWidth - innerWidth));
  assert.equal(medicalOverflow, 0, "mobile Regulatory Support Report workspace must not overflow");
  await medicalPage.close();
  assert.deepEqual(errors, [], `Evidence Package browser diagnostics: ${errors.join(" | ")}`);
  console.log("Evidence Package UI passed (four profiles x four evidence levels, scoped manifests, raw gates, signatures, and mobile geometry)." );
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}
