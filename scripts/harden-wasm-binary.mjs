import { resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { hardenWasmFile } from "./wasm-binary-hardening.mjs";

export function hardenWasmArtifacts(filePaths) {
  return filePaths.map((filePath) => ({ filePath, ...hardenWasmFile(filePath) }));
}

function main() {
  const filePaths = process.argv.slice(2);
  if (!filePaths.length) throw new Error("Usage: node scripts/harden-wasm-binary.mjs <file.wasm> [...]");
  for (const report of hardenWasmArtifacts(filePaths)) {
    console.log(
      `Hardened ${report.filePath}: ${report.beforeBytes} -> ${report.afterBytes} bytes; custom=${report.strippedCustomSections.length}; paths=${report.redactedPathCount}; exports=${report.exports}.`,
    );
  }
}

if (process.argv[1] && resolve(fileURLToPath(import.meta.url)) === resolve(process.argv[1])) main();
