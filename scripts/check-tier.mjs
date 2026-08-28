import process from "node:process";
import { existsSync } from "node:fs";

import { runNode } from "./run-utils.mjs";

const SMOKE = [
  "scripts/check-js.mjs",
  "scripts/check-web-imports.mjs",
  "scripts/check-browser-import-contract.mjs",
  "scripts/check-dom-contract.mjs",
  "scripts/check-workflow-contract.mjs",
  "scripts/check-hash-contract.mjs",
  "scripts/check-parser-robustness.mjs",
  "scripts/check-analysis-invariants.mjs",
  "scripts/check-evidence-treemap.mjs",
  "scripts/check-product-guidance-contracts.mjs",
  "scripts/check-litert-runtime-assets.mjs",
  "scripts/check-sw-assets.mjs",
  "scripts/check-service-worker-lifecycle.mjs",
  "scripts/check-offline-device-controller.mjs",
  "scripts/check-worker-config.mjs",
];

const FORMATS = [
  ...(existsSync("web/protected/deepbom/pkg/deepbom_wasm.js")
    ? ["scripts/check-model-file-contract.mjs"]
    : []),
  "scripts/check-format-routing.mjs",
  "scripts/check-metadata-model-adapters.mjs",
  "scripts/check-coreml-legacy-analysis.mjs",
  "scripts/check-coreml-mlprogram-analysis.mjs",
  "scripts/check-executorch-analysis.mjs",
  "scripts/check-format-quantization-isolation.mjs",
  "scripts/check-onnx-shape-inference.mjs",
  "scripts/check-onnx-operation-cost.mjs",
  "scripts/check-dynamic-shape-cost.mjs",
  "scripts/check-onnx-quantization-contract.mjs",
  "scripts/check-safetensors-quantization-contract.mjs",
  "scripts/check-tensor-numerical-integrity.mjs",
  "scripts/check-tensorrt-static-preflight.mjs",
  "scripts/check-execution-placement-evidence.mjs",
  "scripts/check-format-quant-viewer.mjs",
  "scripts/check-format-capability-viewer.mjs",
];

const FORMAT_UI = new Set([
  "scripts/check-format-quant-viewer.mjs",
  "scripts/check-format-capability-viewer.mjs",
]);
const FORMATS_CORE = FORMATS.filter((script) => !FORMAT_UI.has(script));

if (!existsSync("web/protected/deepbom/pkg/deepbom_wasm.js")) {
  console.log("Public source boundary: protected model/rulepack integration check omitted; public CLI and format contracts remain enabled.");
}

const FORMAT_EVIDENCE = [
  "scripts/write-build-metadata.mjs",
  "scripts/validate-supported-formats.mjs",
  "scripts/verify-supported-format-outputs.mjs",
];

const RELEASE = [
  ...SMOKE,
  ...FORMATS,
  ...FORMAT_EVIDENCE,
  "scripts/write-build-metadata.mjs",
  "scripts/check-build-metadata.mjs",
  "scripts/check-wasm-hardening.mjs",
  "scripts/check-access-hierarchy.mjs",
  "scripts/check-export-artifact-contract.mjs",
  "scripts/check-export-contract-documents.mjs",
  "scripts/check-runtime-contract.mjs",
  "scripts/check-runtime-evidence-sidecar.mjs",
  "scripts/check-cyclonedx-20-quantization-evidence.mjs",
  "scripts/check-standardization-evidence.mjs",
  "scripts/check-public-multiformat-corpus.mjs",
  "scripts/check-core-isolation-roofline.mjs",
  "scripts/check-core-isolation-viewer.mjs",
  "scripts/check-native-capture-pipeline.mjs",
  "scripts/check-private-wasm-contract.mjs",
  "scripts/check-channel-equivalence.mjs",
  "scripts/check-public-package-boundary.mjs",
  "scripts/check-source-budget.mjs",
  "scripts/check-ci-deploy-contract.mjs",
  "scripts/check-git-privacy.mjs",
];

const TIERS = {
  smoke: SMOKE,
  "formats-core": FORMATS_CORE,
  formats: FORMATS,
  "format-evidence": FORMAT_EVIDENCE,
  release: [...new Set(RELEASE)],
};
const tier = process.argv[2] || "smoke";
const checks = TIERS[tier];
if (!checks) throw new Error(`Unknown check tier ${tier}; expected ${Object.keys(TIERS).join(", ")}.`);
const fromValue = process.argv.find((value) => value.startsWith("--from="))?.slice(7)
  || (process.argv.includes("--from") ? process.argv[process.argv.indexOf("--from") + 1] : "");
const startIndex = fromValue ? checks.indexOf(fromValue) : 0;
if (startIndex < 0) throw new Error(`Unknown ${tier} --from check: ${fromValue}`);
const selected = checks.slice(startIndex);
const timeoutMs = Number.parseInt(process.env.DEEPBOM_CHECK_TIMEOUT_MS || "600000", 10);
for (const [offset, script] of selected.entries()) {
  const index = startIndex + offset;
  const started = performance.now();
  console.log(`\n[${tier} ${index + 1}/${checks.length}] ${script}`);
  await runNode(script, [], { timeoutMs });
  console.log(`[${tier} ${index + 1}/${checks.length}] completed in ${((performance.now() - started) / 1000).toFixed(1)}s`);
}
console.log(`DeepBOM ${tier} tier passed (${startIndex + 1}-${checks.length}/${checks.length}; ${selected.length} checks).`);
