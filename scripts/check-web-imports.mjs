import { existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { collectFileSizes } from "./size-utils.mjs";

const libFiles = (await collectFileSizes("web/lib", {
  relativeRoot: "web/lib",
  extensions: new Set([".js"]),
}))
  .map((file) => file.path)
  .filter((name) => !name.includes("/"))
  .map((name) => path.join("web", "lib", name))
  .sort();

const importTargets = [
  ...libFiles,
  "web/onnx.js",
  "pkg/tflite_wasm_audit.js",
];
const protectedTarget = "web/protected/deepbom/pkg/deepbom_wasm.js";
if (existsSync(protectedTarget)) importTargets.push(protectedTarget);

for (const target of importTargets) {
  await import(pathToFileURL(path.resolve(target)).href);
}

console.log(`Web module import smoke passed (${importTargets.length} modules; protected integration ${existsSync(protectedTarget) ? "included" : "omitted at the public boundary"}).`);
