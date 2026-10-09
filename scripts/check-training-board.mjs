import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { createServer } from "node:http";
import { chromium } from "playwright";
import { launchChromium } from "./browser-launch.mjs";
const file = process.argv[2] || ".local-validation/trainable/board.html";
const candidate = process.argv[3];
const html = await readFile(file);
const server = createServer(async (req, res) => {
  const name = {
    "/candidate/report.html": "report.html",
    "/candidate/model.py": "model.py",
    "/candidate/model-code.md": "model-code.md",
  }[req.url];
  try {
    res.setHeader(
      "Content-Type",
      name && name !== "report.html"
        ? "text/plain; charset=utf-8"
        : "text/html; charset=utf-8",
    );
    res.end(candidate && name ? await readFile(join(candidate, name)) : html);
  } catch (error) {
    res.statusCode = 500;
    res.end(error.message);
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
let browser;
try {
  browser = await launchChromium(chromium);
  const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  assert((await page.locator("#weights button").count()) > 0);
  await page.locator("#weights button").first().click();
  assert((await page.locator("#hist rect").count()) > 0);
  for (const value of [
    "channels",
    "spectrum",
    "similarity",
    "projection",
    "sparsity",
  ]) {
    await page.locator("#view").selectOption(value);
    assert(
      (await page.locator("#hist rect,#hist polyline").count()) > 0,
      value,
    );
  }
  await page.locator("#view").selectOption("histogram");
  for (const format of ["svg", "png"]) {
    const downloaded = page.waitForEvent("download");
    await page.locator("#" + format).click();
    const download = await downloaded;
    const bytes = await readFile(await download.path());
    if (format === "png")
      assert.equal(bytes.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
    else
      assert(
        bytes.toString().includes("snapshot") ||
          bytes.toString().includes("histogram"),
      );
  }
  await page.locator("#step").fill("0");
  await page.locator("#step").dispatchEvent("input");
  assert.equal(await page.locator("#weights button").count(), 0);
  await page.locator("#step").evaluate((input) => {
    input.value = input.max;
    input.dispatchEvent(new Event("input"));
  });
  assert((await page.locator("#activations button").count()) > 0);
  await page.locator("#activations button").first().click();
  assert.match(await page.locator("#tensor").textContent(), /before_forward/);
  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
    "narrow viewport must not overflow",
  );
  await mkdir(".local-validation/trainable", { recursive: true });
  await page.screenshot({
    path: ".local-validation/trainable/board-mobile.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: ".local-validation/trainable/board-desktop.png",
    fullPage: true,
  });
  if (candidate) {
    const origin = `http://127.0.0.1:${server.address().port}`;
    await page
      .context()
      .grantPermissions(["clipboard-read", "clipboard-write"], { origin });
    await page.goto(`${origin}/candidate/report.html`);
    assert.match(
      await page.locator("h1").textContent(),
      /Model Optimization Report/,
    );
    assert((await page.locator("details").count()) >= 2);
    await page.locator("summary").first().click();
    assert(
      (await page.locator("details").first().getAttribute("open")) !== null,
    );
    await page.locator("#copy").click();
    await page.waitForFunction(
      () => document.getElementById("notice").textContent === "Copied",
    );
    assert.equal(
      await page.evaluate(() => navigator.clipboard.readText()),
      await readFile(join(candidate, "model.py"), "utf8"),
    );
    for (const name of ["model.py", "model-code.md"]) {
      const received = page.waitForEvent("download");
      await page.locator(`a[href="${name}"]`).click();
      assert.deepEqual(
        await readFile(await (await received).path()),
        await readFile(join(candidate, name)),
      );
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "candidate mobile report must not overflow",
    );
    console.log(
      "Candidate report: expandable structures, copy code, exact .py/Markdown downloads and narrow layout passed.",
    );
  }
  assert.deepEqual(errors, []);
  console.log(
    "DeepBoard: timeline, state/capture binding, six views, SVG/PNG, dark/narrow layout passed.",
  );
} finally {
  await browser?.close();
  await new Promise((r) => server.close(r));
}
