import os from "node:os";
import path from "node:path";
import process from "node:process";
import { readFile } from "node:fs/promises";
import { parseStrictJson } from "../web/lib/strict-json.js";
import { captureTensorRtEngineInspector } from "./tensorrt-engine-inspector-capture-lib.mjs";

const options = parseArguments(process.argv.slice(2));
if (!options.modelPath || !options.profilePath || !options.enginePath || !options.inspectorPath) {
  throw new Error("usage: npm run capture:tensorrt-inspector -- model.onnx profile.json engine.plan layer-info.json [evidence.json] --tensorrt-version X --cuda-version X --device-id N --device-identity TEXT --tool-binary trtexec --invocation TEXT");
}
const profile = parseStrictJson(await readFile(path.resolve(options.profilePath), "utf8"), "TensorRT build profile");
const parserObservation = options.parserObservationPath
  ? parseStrictJson(await readFile(path.resolve(options.parserObservationPath), "utf8"), "TensorRT parser observation") : null;
const outputPath = path.resolve(options.outputPath || defaultOutputPath(options.modelPath));
const evidence = await captureTensorRtEngineInspector({ ...options, profile, parserObservation, outputPath });
console.log(`TensorRT engine-inspector evidence captured: ${outputPath}`);
console.log(`${evidence.inspector.schema_generation}; ${evidence.inspector.profiling_verbosity}; ${evidence.engine.byte_length} engine bytes; declared build binding.`);

function parseArguments(values) {
  const positional = [];
  const result = { source: "trtexec_exportLayerInfo", profilingVerbosity: "detailed", executionContextBound: false };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) { positional.push(value); continue; }
    const [name, inline] = value.slice(2).split(/=(.*)/s, 2);
    const read = () => inline ?? values[++index];
    if (name === "parser-observation") result.parserObservationPath = read();
    else if (name === "tensorrt-version") result.tensorrtVersion = read();
    else if (name === "cuda-version") result.cudaVersion = read();
    else if (name === "device-id") result.deviceId = Number(read());
    else if (name === "device-compute-capability") result.deviceComputeCapability = read();
    else if (name === "device-identity") result.deviceIdentity = read();
    else if (name === "source") result.source = read();
    else if (name === "profiling-verbosity") result.profilingVerbosity = read();
    else if (name === "execution-context-bound") result.executionContextBound = true;
    else if (name === "tool-binary") result.toolBinaryPath = read();
    else if (name === "invocation") result.invocation = read();
    else throw new Error(`Unknown TensorRT inspector capture option --${name}.`);
  }
  [result.modelPath, result.profilePath, result.enginePath, result.inspectorPath, result.outputPath] = positional;
  return result;
}

function defaultOutputPath(modelPath) {
  const root = process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "DeepBOM", "tensorrt-captures") : path.join(os.homedir(), ".cache", "DeepBOM", "tensorrt-captures");
  const stem = path.basename(modelPath, path.extname(modelPath)).replaceAll(/[^A-Za-z0-9._-]+/g, "-").slice(0, 80) || "model";
  return path.join(root, `${stem}-engine-${new Date().toISOString().replaceAll(/[:.]/g, "-")}.json`);
}
