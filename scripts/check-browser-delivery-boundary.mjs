import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile("web/app.js", "utf8");
const worker = await readFile("web/workers/static-audit-worker.js", "utf8");
const serviceWorker = await readFile("web/sw.js", "utf8");
const bootstrap = await readFile("web/bootstrap.js", "utf8");

assert.doesNotMatch(app, /^import .*tflite_wasm_audit/m, "The application shell must not statically import the TFLite WASM wrapper.");
assert.match(app, /import\("\.\.\/pkg\/tflite_wasm_audit\.js"\)/);
assert.match(app, /BROWSER_TARGET_PROFILES/);

assert.doesNotMatch(worker, /^import .*?(?:tflite_wasm_audit|\.\.\/onnx\.js|\.\.\/executorch\.js)/m);
assert.match(worker, /import\("\.\.\/\.\.\/pkg\/tflite_wasm_audit\.js"\)/);
assert.match(worker, /import\("\.\.\/onnx\.js"\)/);
assert.match(worker, /import\("\.\.\/executorch\.js"\)/);

const shellBlock = /const APP_SHELL_ASSETS = \[([\s\S]*?)\];/.exec(serviceWorker)?.[1] || "";
assert.doesNotMatch(shellBlock, /tflite_wasm_audit|app\.js/, "Pre-interaction service-worker install must not fetch the analyzer or application graph.");
assert.match(shellBlock, /bootstrap\.js/);
assert.doesNotMatch(bootstrap, /requestIdleCallback|schedule\(\(\) => loadApp/);

console.log("Browser delivery boundary passed (interaction-loaded app; format analyzers and TFLite WASM are not preloaded).\n");
