import { readFileSync } from "node:fs";

import { readSwRuntimeCacheableSuffixes } from "./sw-utils.mjs";

const loaderSource = readFileSync("web/lib/runtime-module-loader.js", "utf8");
const packageSource = readFileSync("node_modules/@litertjs/core/dist/index.js", "utf8");
const buildPagesSource = readFileSync("scripts/build-pages.mjs", "utf8");
const runtimeAssets = new Set(readSwRuntimeCacheableSuffixes());
const prefix = "/node_modules/@litertjs/core/wasm/";

const requiredVariants = Object.freeze({
  compat: "litert_wasm_compat_internal",
  default: "litert_wasm_internal",
  jspi: "litert_wasm_jspi_internal",
});
const forbiddenVariants = Object.freeze({
  threaded: "litert_wasm_threaded_internal",
});

for (const [mode, basename] of Object.entries(requiredVariants)) {
  for (const extension of ["js", "wasm"]) {
    const asset = `${prefix}${basename}.${extension}`;
    if (!runtimeAssets.has(asset)) {
      throw new Error(`LiteRT ${mode} mode requires missing deployment asset ${asset}.`);
    }
  }
}

for (const [mode, basename] of Object.entries(forbiddenVariants)) {
  for (const extension of ["js", "wasm"]) {
    const asset = `${prefix}${basename}.${extension}`;
    if (runtimeAssets.has(asset)) {
      throw new Error(`LiteRT ${mode} asset is deployed although the application never requests that mode: ${asset}.`);
    }
  }
}

if (!loaderSource.includes('backend === "webnn" || backend === "webgpu" ? "jspi" : "default"')) {
  throw new Error("LiteRT runtime loader must bind WebNN/WebGPU to JSPI and all other backends to the default mode.");
}
if (!loaderSource.includes('requestedMode === "jspi" ? { jspi: true } : undefined')) {
  throw new Error("LiteRT runtime loader must explicitly request JSPI only for the JSPI mode.");
}
if (/\bthreads\s*:\s*true\b/.test(loaderSource)) {
  throw new Error("LiteRT threaded mode was introduced without adding its deployment assets and isolation preconditions.");
}
if (!buildPagesSource.includes('"web/lib/runtime-module-loader.js"')) {
  throw new Error("Production assembly must rewrite runtime-module-loader.js from node_modules paths to deployed vendor paths.");
}

for (const sourceFragment of [
  'var WASM_JS_COMPAT_FILE_NAME = "litert_wasm_compat_internal.js"',
  'var WASM_JS_FILE_NAME = "litert_wasm_internal.js"',
  'var WASM_JS_JSPI_FILE_NAME = "litert_wasm_jspi_internal.js"',
  'if (options?.threads)',
  'else if (options?.jspi)',
]) {
  if (!packageSource.includes(sourceFragment)) {
    throw new Error(`Pinned @litertjs/core loader contract changed: ${sourceFragment}`);
  }
}

console.log("LiteRT runtime asset contract passed (compat + default + JSPI deployed; unused threaded variant excluded).");
