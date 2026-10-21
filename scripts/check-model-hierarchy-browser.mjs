import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
import { createStaticTestServer } from "./static-test-server.mjs";
import { hierarchyFixture } from "./fixtures/source-hierarchy.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";

const bytes = hierarchyFixture(), analysis = analyzeOnnxModel(bytes, "source.onnx");
const ctx = getArtifactIrContext(analysis, { sha256: sha256BytesHex(bytes), filename: "source.onnx", format: "onnx", size: bytes.length });
const data = { model: ctx.model_ir, hierarchy: ctx.model_hierarchy };
const output = path.resolve(".local-validation/model-hierarchy"); await mkdir(output, { recursive: true });
const server = createStaticTestServer(process.cwd()); await new Promise(r => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`, browser = await launchChromium(chromium);
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const errors = []; page.on("pageerror", e => errors.push(e.message));
  await page.route(`${origin}/hierarchy`, route => route.fulfill({ contentType: "text/html", body: '<!doctype html><html><body style="margin:12px"><svg id="other"><text fill="red">Unaffected</text></svg><div id="host"></div><div id="editor"></div></body></html>' }));
  await page.goto(`${origin}/hierarchy`);
  await page.evaluate(async data => {
    const { initializeNumericalWasm } = await import("/web/lib/numerical-wasm.js");
    await initializeNumericalWasm();
    const { mountModelHierarchy } = await import("/web/lib/model-hierarchy-view.js");
    window.hierarchyMount = mountModelHierarchy; window.sourceData = data;
    window.current = mountModelHierarchy(document.getElementById("host"), data.model, data.hierarchy, { onSelect: op => { window.selectedOp = op.id; }, onSelection: selection => { window.selectedSource = selection; } });
  }, data);
  const host = page.locator("#host");
  assert.equal(await host.locator("[data-subject-ref]").count(), 2);
  await host.locator("[data-subject-ref]").first().focus(); await page.keyboard.press("Enter");
  assert.match(await host.locator(".mh-detail").textContent(), /enc\.0/);
  assert.equal(await page.evaluate(() => window.selectedSource.kind), "group");
  assert.equal(await page.evaluate(() => window.selectedOp), undefined, "Group selection must not invoke the legacy operation callback");
  await host.getByRole("button", { name: "Expand group", exact: true }).click();
  assert.equal(await host.locator("[data-subject-ref]").count(), 3);
  await host.locator("[data-subject-ref]").first().click();
  assert(await page.evaluate(() => !!window.selectedOp));
  assert.equal(await page.evaluate(() => window.selectedSource.kind), "operation");
  assert.equal(await page.evaluate(() => window.selectedSource.subject_ref === window.selectedOp), true);
  await host.getByRole("button", { name: "Trace upstream data paths" }).click();
  await host.getByRole("button", { name: "Whole model", exact: true }).click();
  await host.getByRole("searchbox").fill("dec.0");
  assert.equal(await host.locator("[data-subject-ref]").count(), 1);
  assert.match(await host.locator(".mh-status").textContent(), /1 represented · 2 outside/);
  assert.equal(await host.locator(".mh-table tr").count(), 2);
  const downloadPromise = page.waitForEvent("download"); await host.getByRole("button", { name: "Export SVG", exact: true }).click();
  const download = await downloadPromise; await download.saveAs(path.join(output, "browser-selected.svg"));
  const pngPromise = page.waitForEvent("download"); await host.getByRole("button", { name: "Export PNG", exact: true }).click();
  const png = await pngPromise; await png.saveAs(path.join(output, "browser-selected.png"));
  assert.deepEqual([...(await readFile(path.join(output, "browser-selected.png"))).subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  for (const theme of ["dark", "monochrome", "light"]) {
    await host.getByRole("combobox", { name: "Hierarchy theme" }).selectOption(theme);
    assert.equal(await page.locator("#other text").evaluate(el => getComputedStyle(el).fill), "rgb(255, 0, 0)", "Inline SVG styles must not leak into other diagrams");
    await page.screenshot({ path: path.join(output, `hierarchy-${theme}.png`) });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await host.evaluate(el => el.scrollWidth <= el.clientWidth + 2), "Mobile hierarchy must scroll inside the canvas");
  assert(await host.locator("svg text.secondary").first().evaluate(el => parseFloat(getComputedStyle(el).fontSize) >= 12), "Mobile glyphs must not be scaled to fit the graph");
  await page.screenshot({ path: path.join(output, "hierarchy-mobile.png"), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(async data => {
    const { createModelStructureEditor } = await import("/web/lib/model-structure-editor.js");
    const editor = createModelStructureEditor({ hierarchyContext: () => data });
    editor.sync(document.getElementById("editor"), data.model);
  }, data);
  await page.locator("#editor summary").filter({ hasText: "Original source hierarchy" }).click();
  await page.locator("#editor .model-hierarchy-panel [data-subject-ref]").first().waitFor();
  assert.equal(await page.locator("#editor .model-hierarchy-panel [data-subject-ref]").count(), 2);
  assert.equal(await page.locator("#editor .se-node").count(), 3, "Source grouping must not merge editable operation nodes");
  await page.evaluate(() => {
    const old = window.current;
    window.current = window.hierarchyMount(document.getElementById("host"), window.sourceData.model, window.sourceData.hierarchy);
    old.destroy(); // A stale owner must not clear a newer mount.
  });
  assert.equal(await host.locator("[data-subject-ref]").count(), 1, "Same-source remount preserves search state");
  await page.evaluate(() => { window.current.destroy(); });
  assert.equal(await host.locator("svg").count(), 0);
  // The optional host handoff preserves bytes and reports rejected delivery.
  await page.evaluate(() => {
    window.exported = [];
    window.hierarchyMount(document.getElementById("host"), window.sourceData.model, window.sourceData.hierarchy, {
      async offerDownload(blob, name) { if (window.rejectExport) throw new Error("host delivery denied"); window.exported.push({ name, type: blob.type, text: await blob.text() }); },
    });
  });
  await host.getByRole("button", { name: "Export SVG", exact: true }).click();
  await page.waitForFunction(() => window.exported.length === 1);
  assert.match(await page.evaluate(() => window.exported[0].text), /view_sha256/);
  await page.evaluate(() => { window.rejectExport = true; });
  await host.getByRole("button", { name: "Export SVG", exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.mh-status').textContent.includes("host delivery denied"));
  // Exercise the real file → worker → common context → Explorer path as well.
  await page.goto(`${origin}/web/`);
  await page.locator("#dropzone").dispatchEvent("pointerdown"); await page.locator("#fileInput").focus();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent === "Ready", null, { timeout: 60000 });
  if (await page.locator("#agreementBackdrop").isVisible()) { await page.locator("#privacyAgree").check(); await page.locator("#acceptAgreement").click(); }
  await page.locator("#fileInput").setInputFiles({ name: "source.onnx", mimeType: "application/octet-stream", buffer: Buffer.from(bytes) });
  await page.waitForFunction(() => !document.querySelector("#runAudit")?.disabled); await page.locator("#runAudit").click();
  await page.waitForFunction(() => /audit run complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 120000 });
  assert.match(await page.locator("#status").textContent(), /audit run complete/);
  assert.equal(await page.locator("[data-model-hierarchy] [data-subject-ref]").count(), 2);
  assert(await page.locator("[data-model-hierarchy]").evaluate(el => el.open), "Hierarchy is the initial open structure view");
  await page.locator('[data-workflow-step="graph"]').click();
  await page.locator("#showModelSummary").click();
  assert.match(await page.locator("#modelSummaryText").textContent(), /enc\.0/);
  assert.match(await page.locator("#modelSummaryText").textContent(), /dec\.0/);
  const exported = await page.evaluate(async () => {
    const { buildBrowserModelIrVisualizationArchive } = await import("/web/lib/model-ir-browser-export.js");
    const { analyzeOnnxModel } = await import("/web/onnx.js");
    const { getArtifactIrContext } = await import("/web/lib/artifact-ir-context.js");
    const { sha256BytesHex } = await import("/web/lib/sha256-sync.js");
    const bytes = new Uint8Array(await (await fetch("/.local-validation/model-hierarchy/source.onnx")).arrayBuffer());
    const context = getArtifactIrContext(analyzeOnnxModel(bytes, "source.onnx"), { sha256: sha256BytesHex(bytes), filename: "source.onnx", format: "onnx", size: bytes.length });
    const archive = await buildBrowserModelIrVisualizationArchive(context.model_ir, { hierarchy: context.model_hierarchy });
    return { rows: archive.bundle.hierarchy_view.rows.map(row => row.label), source: archive.bundle.manifest.architecture_source, pngs: archive.png_manifest.files.length };
  });
  assert.deepEqual(exported.rows, ["enc.0", "dec.0"]);
  assert(exported.pngs > 0 && exported.source.basis === "source_annotated");
  assert.deepEqual(errors, []);
  console.log("Hierarchy browser passed: real audit, source grouping, exact operation selection, independent IDE, filtering/export, theme isolation, mobile readability and cleanup.");
} finally { await browser.close(); await new Promise(r => server.close(r)); }
