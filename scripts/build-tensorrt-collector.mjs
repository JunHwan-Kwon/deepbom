import process from "node:process";
import { buildTensorRtCollector } from "./tensorrt-capture-lib.mjs";

const args = process.argv.slice(2);
const readOption = (name) => {
  const prefix = `--${name}=`;
  const inline = args.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : null;
};
const collector = await buildTensorRtCollector({
  tensorRtRoot: readOption("tensorrt-root"),
  buildDir: readOption("build-dir"),
  configuration: readOption("configuration") || "Release",
});
console.log(`TensorRT parser collector built: ${collector}`);
