import process from "node:process";
import { runNativeCapture } from "./native-capture-lib.mjs";

const args = process.argv.slice(2);
const configIndex = args.indexOf("--config");
if (configIndex < 0 || !args[configIndex + 1]) {
  throw new Error("usage: node scripts/run-native-capture.mjs --config <capture-run.json> [--output-dir <dir>] [--skip-collector-build]");
}
const outputIndex = args.indexOf("--output-dir");
const result = await runNativeCapture(args[configIndex + 1], {
  outputDir: outputIndex >= 0 ? args[outputIndex + 1] : undefined,
  skipCollectorBuild: args.includes("--skip-collector-build"),
});
console.log(`Native capture package written to ${result.outputDir}`);
console.log(`Mode: ${result.index.capture_mode}; importable runtime evidence: ${result.index.importable_runtime_evidence}`);
