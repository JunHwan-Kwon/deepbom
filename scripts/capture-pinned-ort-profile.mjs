import os from "node:os";
import path from "node:path";
import process from "node:process";
import { capturePinnedOrtProfiles } from "./ort-native-capture-lib.mjs";

const options = parseArguments(process.argv.slice(2));
if (!options.artifactPath) {
  throw new Error("usage: npm run capture:pinned-ort -- <model.onnx> [output-dir] [--providers=cpu] [--runs=3] [--warmup-runs=1] [--shape=input=1,3,224,224] [--reduced-op-config=required_operators.config] [--runtime-module=ORT_SOURCE/js/node/dist/index.js --build-attestation=deepbom-ort-build-attestation.json]");
}
const outputDir = options.outputDir || defaultOutputDir(options.artifactPath);
const result = await capturePinnedOrtProfiles({ ...options, outputDir });
console.log(`Pinned native ORT capture verified: ${result.outputDir}`);
console.log(`Capture ID: ${result.index.capture_id}`);
for (const profile of result.index.profiles) {
  console.log(`${profile.role}: ${profile.filename}; ${profile.kernel_event_count} kernel event(s); providers ${profile.observed_providers.join(" + ")}`);
}

function parseArguments(values) {
  const positional = [];
  const result = { providers: ["cpu"], runs: 3, warmupRuns: 1, inputShapes: new Map(), intraOpNumThreads: 1, interOpNumThreads: 1 };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) { positional.push(value); continue; }
    const [rawName, inlineValue] = value.slice(2).split(/=(.*)/s, 2);
    const readValue = () => inlineValue ?? values[++index];
    if (rawName === "providers") result.providers = String(readValue() || "").split(",");
    else if (rawName === "runs") result.runs = Number(readValue());
    else if (rawName === "warmup-runs") result.warmupRuns = Number(readValue());
    else if (rawName === "intra-op-threads") result.intraOpNumThreads = Number(readValue());
    else if (rawName === "inter-op-threads") result.interOpNumThreads = Number(readValue());
    else if (rawName === "reduced-op-config") result.reducedOperatorConfigPath = String(readValue() || "");
    else if (rawName === "runtime-module") result.runtimeModulePath = String(readValue() || "");
    else if (rawName === "build-attestation") result.buildAttestationPath = String(readValue() || "");
    else if (rawName === "shape") {
      const [name, dims] = String(readValue() || "").split(/=(.*)/s, 2);
      if (!name || !dims) throw new Error("--shape requires input_name=d0,d1,...");
      result.inputShapes.set(name, dims.split(",").map(Number));
    } else throw new Error(`Unknown ORT capture option --${rawName}.`);
  }
  [result.artifactPath, result.outputDir] = positional;
  return result;
}

function defaultOutputDir(artifactPath) {
  const root = process.env.LOCALAPPDATA
    ? path.join(process.env.LOCALAPPDATA, "DeepBOM", "ort-native-captures")
    : path.join(os.homedir(), ".cache", "DeepBOM", "ort-native-captures");
  const stem = path.basename(artifactPath, path.extname(artifactPath)).replaceAll(/[^A-Za-z0-9._-]+/g, "-").slice(0, 80) || "model";
  const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
  return path.join(root, `${stem}-${stamp}`);
}
