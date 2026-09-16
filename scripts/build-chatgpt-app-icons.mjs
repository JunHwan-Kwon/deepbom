import { deflateSync } from "node:zlib";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const root = process.cwd();
const sourcePath = path.join(root, "web", "chatgpt", "deepbom-app-icon.svg");
const outputRoot = path.join(root, "docs", "chatgpt-app", "assets");
const source = await readFile(sourcePath, "utf8");

await mkdir(outputRoot, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const directoryPage = await browser.newPage({ viewport: { width: 512, height: 512 }, deviceScaleFactor: 1 });
  await directoryPage.setContent(`<!doctype html><style>html,body,svg{width:100%;height:100%;margin:0;display:block}</style>${source}`);
  await directoryPage.screenshot({ path: path.join(outputRoot, "deepbom-directory-icon-512.png"), type: "png", omitBackground: false });
  await directoryPage.close();

  // The ChatGPT composer upload requires a square PNG of at least 256 px and
  // no more than 10 KB. An indexed palette keeps the canonical gradient mark
  // recognizable while making that transport constraint deterministic.
  const composerPixels = await renderRgba(browser, source, 256);
  const composerPng = encodeIndexedPng(composerPixels, 256, 256);
  if (composerPng.length > 10_000) {
    throw new Error(`Composer icon is ${composerPng.length} bytes; the portal limit is 10,000 bytes.`);
  }
  await writeFile(path.join(outputRoot, "deepbom-composer-icon-256.png"), composerPng);
} finally {
  await browser.close();
}

console.log("Built ChatGPT directory (512px) and portal-safe composer (256px, <=10 KB) icons from the canonical SVG mark.");

async function renderRgba(browser, svg, size) {
  const page = await browser.newPage({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
  try {
    return Buffer.from(await page.evaluate(async ({ svg, size }) => {
      const blob = new Blob([svg], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const context = canvas.getContext("2d", { alpha: true });
        context.drawImage(image, 0, 0, size, size);
        return Array.from(context.getImageData(0, 0, size, size).data);
      } finally {
        URL.revokeObjectURL(url);
      }
    }, { svg, size }));
  } finally {
    await page.close();
  }
}

function encodeIndexedPng(rgba, width, height) {
  // Index 0 is transparent. The other 128 entries form a deterministic
  // 2/3/2-bit RGB palette, retaining more green resolution for this mark.
  const palette = Buffer.alloc(129 * 3);
  const alpha = Buffer.alloc(129, 255);
  alpha[0] = 0;
  for (let red = 0; red < 4; red += 1) {
    for (let green = 0; green < 8; green += 1) {
      for (let blue = 0; blue < 4; blue += 1) {
        const index = 1 + (red << 5) + (green << 2) + blue;
        palette[index * 3] = Math.round(red * 255 / 3);
        palette[index * 3 + 1] = Math.round(green * 255 / 7);
        palette[index * 3 + 2] = Math.round(blue * 255 / 3);
      }
    }
  }

  const scanlines = Buffer.alloc(height * (width + 1));
  for (let y = 0; y < height; y += 1) {
    const row = y * (width + 1);
    scanlines[row] = 0;
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = (y * width + x) * 4;
      const a = rgba[sourceIndex + 3];
      if (a < 128) {
        scanlines[row + x + 1] = 0;
        continue;
      }
      const red = Math.round(rgba[sourceIndex] * 3 / 255);
      const green = Math.round(rgba[sourceIndex + 1] * 7 / 255);
      const blue = Math.round(rgba[sourceIndex + 2] * 3 / 255);
      scanlines[row + x + 1] = 1 + (red << 5) + (green << 2) + blue;
    }
  }

  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 3;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", header),
    pngChunk("PLTE", palette),
    pngChunk("tRNS", alpha),
    pngChunk("IDAT", deflateSync(scanlines, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function pngChunk(type, data) {
  const name = Buffer.from(type, "ascii");
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  name.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([name, data])), 8 + data.length);
  return chunk;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}
