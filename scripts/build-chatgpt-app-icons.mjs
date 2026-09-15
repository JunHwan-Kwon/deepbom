import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const sourcePath = path.join(root, "web", "chatgpt", "deepbom-app-icon.svg");
const outputRoot = path.join(root, "docs", "chatgpt-app", "assets");
const source = await readFile(sourcePath, "utf8");

const outputs = [
  [512, "deepbom-directory-icon-512.png"],
  [128, "deepbom-composer-icon-128.png"],
];

await mkdir(outputRoot, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  for (const [size, filename] of outputs) {
    const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
    await page.setContent(`<!doctype html><style>html,body,svg{width:100%;height:100%;margin:0;display:block}</style>${source}`);
    await page.screenshot({ path: path.join(outputRoot, filename), type: "png", omitBackground: false });
    await page.close();
  }
} finally {
  await browser.close();
}

console.log("Built ChatGPT directory (512px) and composer (128px) icons from the canonical SVG mark.");
