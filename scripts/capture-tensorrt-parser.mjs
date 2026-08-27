import os from "node:os";
import path from "node:path";
import process from "node:process";
import { readFile } from "node:fs/promises";
import { buildTensorRtCollector, captureTensorRtParser } from "./tensorrt-capture-lib.mjs";

const options = parseArguments(process.argv.slice(2));
if (!options.modelPath || !options.profilePath) {
  throw new Error("usage: npm run capture:tensorrt -- model.onnx profile.json [evidence.json] [--collector path] [--tensorrt-root path] [--external relative/path=file]");
}
const profile = JSON.parse(await readFile(path.resolve(options.profilePath), "utf8"));
const collectorPath = options.collectorPath || await buildTensorRtCollector({ tensorRtRoot: options.tensorRtRoot, buildDir: options.buildDir });
const outputPath = path.resolve(options.outputPath || defaultOutputPath(options.modelPath));
const observation = await captureTensorRtParser({
  modelPath: options.modelPath,
  profile,
  collectorPath,
  outputPath,
  externalComponents: options.externalComponents,
});
console.log(`TensorRT parser observation verified: ${outputPath}`);
console.log(`${observation.subgraphs.length} parser subgraph(s); parser returned ${observation.parser_returned}; ${observation.errors.length} parser error(s).`);

function parseArguments(values) {
  const positional = [];
  const result = { externalComponents: [] };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) { positional.push(value); continue; }
    const [name, inline] = value.slice(2).split(/=(.*)/s, 2);
    const read = () => inline ?? values[++index];
    if (name === "collector") result.collectorPath = read();
    else if (name === "tensorrt-root") result.tensorRtRoot = read();
    else if (name === "build-dir") result.buildDir = read();
    else if (name === "external") {
      const [relativePath, sourcePath] = String(read() || "").split(/=(.*)/s, 2);
      if (!relativePath || !sourcePath) throw new Error("--external requires model/relative/path=source-file");
      result.externalComponents.push({ relative_path: relativePath, source_path: sourcePath });
    } else throw new Error(`Unknown TensorRT capture option --${name}.`);
  }
  [result.modelPath, result.profilePath, result.outputPath] = positional;
  return result;
}

function defaultOutputPath(modelPath) {
  const root = process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "DeepBOM", "tensorrt-captures") : path.join(os.homedir(), ".cache", "DeepBOM", "tensorrt-captures");
  const stem = path.basename(modelPath, path.extname(modelPath)).replaceAll(/[^A-Za-z0-9._-]+/g, "-").slice(0, 80) || "model";
  return path.join(root, `${stem}-${new Date().toISOString().replaceAll(/[:.]/g, "-")}.json`);
}
