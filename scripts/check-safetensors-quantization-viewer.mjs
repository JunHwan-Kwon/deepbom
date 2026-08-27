import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

import { launchChromium } from "./browser-launch.mjs";
import { createStaticTestServer } from "./static-test-server.mjs";
import { buildSafeTensorsQuantizationContract } from "../web/lib/safetensors-quantization-contract.js";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (value) => value.slice(1)));
const OUTPUT = path.join(ROOT, ".local-validation", "safetensors-quantization-view");
const contract = buildSafeTensorsQuantizationContract({
  quantization_config: { quant_method: "awq", bits: 4, group_size: 128, zero_point: true, version: "gemm" },
}, packedTensors("model.layers.0.self_attn.q_proj"));
const analysis = {
  format: "safetensors",
  filename: "viewer-awq.safetensors",
  file_size: 1_024_000,
  file_size_bytes: 1_024_000,
  tensor_count: 3,
  operator_count: 0,
  total_macs: null,
  tensors: [],
  ops: [],
  inputs: [],
  outputs: [],
  safetensors: {
    tensor_count: 3,
    payload_byte_length: contract.packed_tensor_bytes,
    payload_coverage_status: "complete_without_gaps_or_overlaps",
    duplicate_key_validation: "complete",
    parser_scope: "viewer fixture",
    quantization_contract: contract,
  },
  tensor_numerical_integrity: { status: "not_assessed_fixture", tensor_count: 3 },
};
const server = createStaticTestServer(ROOT);
let browser;

try {
  await mkdir(OUTPUT, { recursive: true });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await launchChromium(chromium);
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
  await page.evaluate(async (model) => {
    const { artifactOverviewPanels } = await import("/web/lib/artifact-overview.js");
    document.documentElement.dataset.theme = "dark";
    const grid = document.createElement("main");
    grid.id = "safeQuantGrid";
    grid.className = "artifact-evidence-grid";
    grid.style.maxWidth = "1180px";
    grid.style.margin = "20px auto";
    grid.append(...artifactOverviewPanels(model));
    document.body.replaceChildren(grid);
  }, analysis);

  const desktop = await inspect(page);
  if (!desktop.text.includes("Packed Weight Quantization")
    || !desktop.text.includes("AWQ / WQLinear_GEMM")
    || !desktop.text.includes("2,048 / 2,048")
    || !desktop.text.includes(contract.source.sha256)
    || !desktop.text.includes("does not reconstruct floating weights")
    || desktop.disclosureOpen
    || desktop.moduleLedgerCount !== 1
    || desktop.bodyOverflow > 1
    || desktop.panelOverflow > 1
    || desktop.titleContrast < 4.5
    || desktop.valueContrast < 4.5) {
    throw new Error(`SafeTensors quantization viewer desktop contract failed: ${JSON.stringify(desktop)}`);
  }
  await page.locator('[data-artifact-panel="safetensors-quantization"] summary').click();
  const opened = await page.locator('[data-artifact-panel="safetensors-quantization"] details').evaluate((node) => node.open);
  if (!opened) throw new Error("Complete packed-module ledger disclosure did not open.");
  await page.screenshot({ path: path.join(OUTPUT, "desktop-dark.png"), fullPage: true });

  await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  const light = await inspect(page);
  if (light.titleContrast < 4.5 || light.valueContrast < 4.5) throw new Error(`SafeTensors light-theme contrast failed: ${JSON.stringify(light)}`);

  await page.setViewportSize({ width: 360, height: 800 });
  const mobile = await inspect(page);
  if (mobile.bodyOverflow > 1 || mobile.panelOverflow > 1 || mobile.titleContrast < 4.5 || mobile.valueContrast < 4.5) {
    throw new Error(`SafeTensors quantization viewer mobile contract failed: ${JSON.stringify(mobile)}`);
  }
  await page.screenshot({ path: path.join(OUTPUT, "mobile-light.png"), fullPage: true });
  console.log(`SafeTensors quantization viewer checks passed; screenshots: ${path.relative(ROOT, OUTPUT)}`);
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
}

async function inspect(page) {
  return page.evaluate(() => {
    const panel = document.querySelector('[data-artifact-panel="safetensors-quantization"]');
    const title = panel?.querySelector("h3");
    const value = panel?.querySelector("dd");
    const details = panel?.querySelector("details");
    const contrast = (node) => {
      if (!node) return 0;
      const foreground = rgb(getComputedStyle(node).color);
      let owner = node;
      let background = null;
      while (owner && !background) {
        const candidate = rgb(getComputedStyle(owner).backgroundColor);
        if (candidate?.[3] > 0) background = candidate;
        owner = owner.parentElement;
      }
      if (!foreground || !background) return 0;
      const lum = ([r, g, b]) => {
        const channels = [r, g, b].map((value) => {
          const normalized = value / 255;
          return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
      };
      const left = lum(foreground);
      const right = lum(background);
      return (Math.max(left, right) + 0.05) / (Math.min(left, right) + 0.05);
    };
    const rgb = (value) => {
      const match = String(value).match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
      return match ? [Number(match[1]), Number(match[2]), Number(match[3]), match[4] == null ? 1 : Number(match[4])] : null;
    };
    return {
      text: panel?.textContent || "",
      disclosureOpen: details?.open ?? null,
      moduleLedgerCount: (details?.querySelector("code")?.textContent.match(/"name":/g) || []).length,
      bodyOverflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
      panelOverflow: panel ? Math.max(0, panel.scrollWidth - panel.clientWidth) : -1,
      titleContrast: contrast(title),
      valueContrast: contrast(value),
    };
  });
}

function packedTensors(base) {
  const fields = {
    qweight: ["I32", [128, 2]],
    qzeros: ["I32", [1, 2]],
    scales: ["F16", [1, 16]],
  };
  let offset = 0;
  return Object.entries(fields).map(([suffix, [dtype, shape]], index) => {
    const bytes = shape.reduce((product, value) => product * value, 1) * (dtype === "F16" ? 2 : 4);
    const row = { index, name: `${base}.${suffix}`, dtype, shape, byte_length: bytes, data_offset: offset, data_end: offset + bytes };
    offset += bytes;
    return row;
  });
}
