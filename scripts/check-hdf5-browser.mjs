// Called by check-hdf5-source.py with independently generated real fixtures.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";

const root = path.resolve(process.env.DEEPBOM_HDF5_DIST_ROOT || "dist");
const fixtures = process.argv.slice(2);
assert.equal(fixtures.length, 2, "Provide HDF5 and Keras fixtures from check-hdf5-source.py");
const server = createServer(async (req, res) => {
  try {
    let name = new URL(req.url, "http://localhost").pathname;
    if (name.startsWith("/project/")) name = name.slice("/project".length);
    if (name === "/web/") name += "index.html";
    const file = path.resolve(root, "." + name);
    if (!file.startsWith(root + path.sep)) throw Error("outside root");
    const content = await readFile(file);
    const type = { ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".css": "text/css", ".wasm": "application/wasm", ".json": "application/json" }[path.extname(file)] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": type }); res.end(content);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await launchChromium(chromium);
  for (const prefix of ["", "/project"]) {
    const page = await browser.newPage(); const errors = [], requests = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => { if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) errors.push(message.text()); });
    page.on("request", request => requests.push(new URL(request.url()).pathname));
    await page.goto(`http://127.0.0.1:${server.address().port}${prefix}/web/`);
    await page.locator("#dropzone").dispatchEvent("pointerdown");
    await page.locator("#fileInput").focus();
    try {
      await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 180_000 });
    } catch (error) { console.error({prefix, errors, status:await page.locator("#status").textContent()}); throw error; }
    for (const file of fixtures) {
      await page.locator("#fileInput").setInputFiles(file);
      await page.locator("#runAudit:not(:disabled)").click();
      try {
        await page.waitForFunction(() => /audit(?: run)? complete|Audit failed/i.test(document.querySelector("#status")?.textContent || ""), null, { timeout: 60_000 });
      } catch (error) {
        console.error({file, prefix, errors, status:await page.locator("#status").textContent(), recentRequests:requests.slice(-10)});
        throw error;
      }
      assert.match(await page.locator("#status").textContent(), /audit(?: run)? complete/i);
      assert(await page.locator("#runInference").isDisabled(), "static files must not enable browser execution");
      assert.match(await page.locator("#runtimeNotes").textContent(), /browser execution is unavailable/);
      await page.locator('[data-workflow-step="weight"]').click();
      await page.locator('#weightWorkspace [data-action="weights"]').click();
      await page.locator('#weightWorkspace [data-action="save"]:not(:disabled)').waitFor({ timeout: 60_000 });
      const download = page.waitForEvent("download");
      await page.locator('#weightWorkspace [data-action="save"]').click();
      const saved = JSON.parse(await readFile(await (await download).path(), "utf8"));
      const rows = saved.weight_ir.tensors;
      if (file.endsWith(".h5")) {
        assert.equal(rows.find(row => row.name === "/float").statistics.minimum, -2);
        assert.equal(rows.find(row => row.name === "/text").status, "not_assessed");
      } else {
        assert.equal(rows.length, 2); assert(rows.every(row => row.status === "assessed"));
      }
    }
    // Exercise the unbundled shared adapter too; the audit worker deliberately
    // has its own self-contained decoder, while browser modules share vendor/.
    const chunks = requests.filter(name => name.startsWith(prefix + "/web/chunks/"));
    assert(chunks.length > 0, "production application chunks were loaded");
    const inventory = await page.evaluate(async ({prefix, bytes}) => {
      const { inspectHdf5Objects } = await import(prefix + "/web/lib/hdf5-source.js");
      return inspectHdf5Objects(Uint8Array.from(bytes));
    }, {prefix, bytes:[...await readFile(fixtures[0])]});
    assert(inventory.objects.some(row => row.path === "/float"));
    assert(requests.includes(prefix + "/vendor/h5wasm/dist/esm/hdf5_hl.js"));
    assert.deepEqual(errors, []);
    // No TFLite guard has been loaded on this page. A later parser failure must
    // still become a visible error, rather than staying at 'Rendering audit'.
    await page.locator("#fileInput").setInputFiles({name:"broken.onnx", mimeType:"application/octet-stream", buffer:Buffer.from([0x08])});
    await page.locator("#runAudit:not(:disabled)").click();
    await page.waitForFunction(() => document.querySelector("#status")?.textContent?.startsWith("Audit failed:"), null, {timeout:30_000});
    assert(errors.length > 0 && errors.every(message => message.startsWith("[analyzeFile]")), "only the deliberately malformed artifact may emit an error");
    console.log(`HDF5/Keras production UI, Weight IR export and subdirectory hosting passed: ${prefix || "/"}`);
    await page.close();
  }
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
