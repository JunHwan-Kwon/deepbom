import { readFileSync } from "node:fs";
import { collectFileSizes } from "./size-utils.mjs";
import { STATIC_AUDIT_OPERATION } from "../web/lib/static-audit-worker-protocol.js";

const WASM_MODULE = "tflite_wasm_audit.js";
const MAIN_THREAD_ALLOWED = new Set(["default", "runtime_guard", "target_profiles"]);
const HEAVY_EXPORTS = new Set([
  "analyze_tflite_for_target",
  "compute_activation_haar",
  "compute_delegation_repair",
  "compute_deployment_delta",
  "compute_deployment_frontier",
  "compute_input_influence",
  "compute_kernel_haar_decomposition",
  "compute_model_tomography",
  "compute_output_influence",
  "compute_static_runtime_calibration",
  "compute_weight_histogram",
  "explore_tflite_redesign_pareto",
  "landscape_directions",
  "landscape_tomography",
  "layer_landscape_grid",
  "project_tflite_redesign",
  "synthetic_landscape_grid",
]);
const failures = [];
const browserFiles = (await collectFileSizes("web", {
  relativeRoot: ".",
  extensions: new Set([".js"]),
}))
  .map((item) => item.path)
  .filter((file) => !file.includes("/vendor/") && !file.includes("/protected/"));

for (const file of browserFiles) {
  const source = readFileSync(file, "utf8");
  const imported = wasmImports(source);
  if (file !== "web/workers/static-audit-worker.js") {
    for (const name of imported) {
      if (!MAIN_THREAD_ALLOWED.has(name)) failures.push(`${file}: heavy or unreviewed WASM import ${name}`);
    }
  }
  if (file === "web/workers/static-audit-worker.js") {
    if (!source.includes('import("../../pkg/tflite_wasm_audit.js")')) {
      failures.push(`${file}: Worker RPC must load the TFLite WASM module inside the worker`);
    }
    for (const name of HEAVY_EXPORTS) {
      if (!source.includes(`tflite.${name}`)) failures.push(`${file}: Worker RPC does not own heavy WASM export ${name}`);
    }
  }
}

const appSource = readFileSync("web/app.js", "utf8");
const workerSource = readFileSync("web/workers/static-audit-worker.js", "utf8");
if (!appSource.includes('import("../pkg/tflite_wasm_audit.js")')
  || !appSource.includes("tfliteAnalyzerModule.runtime_guard()")) {
  failures.push("web/app.js: main-thread TFLite module access must remain limited to lazy initialization and runtime_guard");
}
for (const name of HEAVY_EXPORTS) {
  if (appSource.includes(`tfliteAnalyzerModule.${name}`)) {
    failures.push(`web/app.js: heavy WASM export ${name} must remain Worker-bound`);
  }
}
for (const operation of Object.keys(STATIC_AUDIT_OPERATION)) {
  if (!workerSource.includes(`STATIC_AUDIT_OPERATION.${operation}`)) {
    failures.push(`web/workers/static-audit-worker.js: protocol operation ${operation} has no handler`);
  }
}
for (const name of HEAVY_EXPORTS) {
  if (name === "compute_deployment_delta") continue;
  if (!new RegExp(`${name}\\(\\s*requestModelBytes`).test(workerSource)) {
    failures.push(`web/workers/static-audit-worker.js: ${name} is not bound to the request-local model snapshot`);
  }
}
if (!/compute_deployment_delta\([\s\S]*?requestModelBytes,[\s\S]*?requestFilename,[\s\S]*?payload\.targetIdsJson/.test(workerSource)) {
  failures.push("web/workers/static-audit-worker.js: deployment delta is not bound to the request-local candidate snapshot");
}
if (!/TFLITE_DEPLOYMENT_DELTA,[\s\S]*?targetIdsJson:\s*JSON\.stringify\(targetIds\)/.test(appSource)) {
  failures.push("web/app.js: deployment delta must preserve the Rust API's string target-ID array contract");
}
for (const name of ["analyzeOnnxModel", "analyzeExecuTorchModel"]) {
  if (!new RegExp(`${name}\\(requestModelBytes, requestFilename`).test(workerSource)) {
    failures.push(`web/workers/static-audit-worker.js: ${name} is not bound to the request-local model snapshot`);
  }
}
for (const forbidden of ["readMetadataModelFile", "readCoreMlModelFile", "readArtifactBundle"]) {
  if (new RegExp(`import[\\s\\S]*?\\b${forbidden}\\b[\\s\\S]*?from`).test(appSource)) {
    failures.push(`web/app.js: main thread imports range analyzer ${forbidden}`);
  }
}
for (const operation of ["ARTIFACT_BUNDLE_ANALYZE", "METADATA_ANALYZE", "COREML_ANALYZE"]) {
  if (!appSource.includes(`STATIC_AUDIT_OPERATION.${operation}`)) {
    failures.push(`web/app.js: missing metadata Worker RPC ${operation}`);
  }
}

if (failures.length) {
  console.error("Main-thread heavy WASM boundary check failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Main-thread heavy WASM boundary passed (${HEAVY_EXPORTS.size} heavy exports isolated; metadata range analyzers Worker-bound).`);

function wasmImports(source) {
  const match = [...source.matchAll(/import\s+([\s\S]*?)\s+from\s+["'][^"']*tflite_wasm_audit\.js["']/g)][0];
  if (!match) return new Set();
  const clause = match[1].trim();
  const names = new Set();
  const named = clause.match(/\{([\s\S]*?)\}/);
  if (named) {
    for (const item of named[1].split(",")) {
      const name = item.trim().split(/\s+as\s+/)[0];
      if (name) names.add(name);
    }
  }
  const beforeNamed = named ? clause.slice(0, named.index).replace(/,$/, "").trim() : clause;
  if (beforeNamed && !beforeNamed.startsWith("*")) names.add("default");
  return names;
}
