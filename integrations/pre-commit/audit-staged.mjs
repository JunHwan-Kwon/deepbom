#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const supported = /\.(?:tflite|onnx|gguf|safetensors|mlmodel|pte|ptd)$/i;
const artifacts = process.argv.slice(2).filter((file) => supported.test(file));
for (const artifact of artifacts) {
  const result = spawnSync(process.execPath, [path.join(root, "bin", "deepbom.mjs"), "audit", artifact, "--summary", "--gate", "defects"], {
    cwd: process.cwd(),
    encoding: "utf8",
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
