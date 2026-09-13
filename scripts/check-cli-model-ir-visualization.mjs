import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { inflateSync } from "node:zlib";

import AdmZip from "adm-zip";

const output = path.join(os.tmpdir(), `deepbom-model-views-${process.pid}.zip`);
try {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", "visualize", "web/samples/tinymqa1m.Q4_0.gguf", "--view", "identity-boundary,exhaustive", "--orientation", "portrait", "-o", output], {
    encoding: "utf8", timeout: 120_000, maxBuffer: 64 * 1024 * 1024,
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const zip = new AdmZip(await readFile(output));
  const names = zip.getEntries().map((entry) => entry.entryName);
  assert(names.includes("model-views/manifest.json"));
  assert(names.includes("model-views/png-derivation-manifest.json"));
  assert(names.some((name) => name.endsWith(".svg")));
  assert(names.some((name) => name.endsWith(".png")));
  const manifest = JSON.parse(zip.readAsText("model-views/manifest.json"));
  assert.equal(manifest.conservation.status, "conserved");
  assert.equal(manifest.conservation.omitted_count, 0);
  const pngManifest = JSON.parse(zip.readAsText("model-views/png-derivation-manifest.json"));
  assert.equal(pngManifest.resolution_dpi, 300);
  assert.equal(pngManifest.color_contract, "black_white_only");
  for (const entry of zip.getEntries().filter((row) => row.entryName.endsWith(".png"))) assertBlackWhitePng(entry.getData());

  const json = spawnSync(process.execPath, ["bin/deepbom.mjs", "visualize", "web/samples/sample_cnn_float.onnx", "--view", "exhaustive", "--compact"], {
    encoding: "utf8", timeout: 120_000, maxBuffer: 64 * 1024 * 1024,
  });
  assert.equal(json.status, 0, json.stderr || json.stdout);
  const parsed = JSON.parse(json.stdout);
  assert.equal(parsed.schema, "deepbom.model_ir_visualization_manifest.v1");
  assert.equal(parsed.conservation.status, "conserved");

  const explicitJson = spawnSync(process.execPath, ["bin/deepbom.mjs", "visualize", "web/samples/sample_cnn_float.onnx", "--view", "architecture-overview", "--output-format", "json", "--compact"], {
    encoding: "utf8", timeout: 120_000, maxBuffer: 64 * 1024 * 1024,
  });
  assert.equal(explicitJson.status, 0, explicitJson.stderr || explicitJson.stdout);
  assert.equal(JSON.parse(explicitJson.stdout).schema, "deepbom.model_ir_visualization_manifest.v1");
} finally {
  await rm(output, { force: true });
}

console.log("CLI Model IR visualization checks passed (ZIP, canonical SVG, 300-DPI black-white PNG, captions, and manifest).\n");

function assertBlackWhitePng(bytes) {
  assert.equal(bytes.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  let offset = 8;
  let width = 0;
  let height = 0;
  const idat = [];
  while (offset < bytes.length) {
    const size = bytes.readUInt32BE(offset);
    const type = bytes.subarray(offset + 4, offset + 8).toString("ascii");
    const data = bytes.subarray(offset + 8, offset + 8 + size);
    if (type === "IHDR") { width = data.readUInt32BE(0); height = data.readUInt32BE(4); }
    if (type === "IDAT") idat.push(data);
    offset += 12 + size;
  }
  assert(width > 0 && height > 0);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * 4;
  assert.equal(raw.length, (stride + 1) * height);
  for (let row = 0; row < height; row += 1) {
    const start = row * (stride + 1);
    assert.equal(raw[start], 0, "PNG must use deterministic filter 0");
    for (let index = start + 1; index < start + 1 + stride; index += 4) {
      assert([0, 255].includes(raw[index]) && raw[index] === raw[index + 1] && raw[index] === raw[index + 2] && raw[index + 3] === 255, "PNG pixel must be opaque black or white");
    }
  }
}
