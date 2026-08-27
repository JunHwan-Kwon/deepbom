import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  analyzeSafeTensorsCorpusArtifact,
  baselineFromReceipt,
  readSafeTensorsArchitectureCorpus,
  receiptMatchesBaseline,
  safeTensorsCorpusCacheDir,
} from "./safetensors-architecture-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readSafeTensorsArchitectureCorpus();
const cacheDir = args.cacheDir || safeTensorsCorpusCacheDir();
const outputDir = path.resolve(args.outputDir);
await mkdir(outputDir, { recursive: true });

const receipts = [];
for (const artifact of manifest.artifacts) {
  const receipt = await analyzeSafeTensorsCorpusArtifact(cacheDir, artifact, { offline: args.offline });
  receipts.push(receipt);
  const state = artifact.baseline ? (receiptMatchesBaseline(receipt, artifact) ? "baseline matched" : "BASELINE MISMATCH") : "baseline pending";
  console.log(`${artifact.id}: ${receipt.status}; ${receipt.tensor_count} tensors; ${receipt.payload_byte_length} payload bytes; ${state}`);
  if (artifact.baseline && !receiptMatchesBaseline(receipt, artifact)) {
    console.error(JSON.stringify({ expected: artifact.baseline, actual: baselineFromReceipt(receipt) }, null, 2));
    process.exitCode = 1;
  }
}

const sweep = {
  schema: "deepbom.safetensors_architecture_corpus_sweep.v1",
  generated_at: new Date().toISOString(),
  population_scope: manifest.population_scope,
  redistribution_boundary: manifest.redistribution_boundary,
  artifact_count: receipts.length,
  receipts,
};
await writeFile(path.join(outputDir, "safetensors-architecture-corpus-sweep.json"), `${JSON.stringify(sweep, null, 2)}\n`, "utf8");

function parseArgs(argv) {
  const output = { offline: false, cacheDir: "", outputDir: ".local-validation/safetensors-architecture-corpus" };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--offline") output.offline = true;
    else if (value === "--cache-dir") output.cacheDir = argv[++index] || "";
    else if (value === "--output-dir") output.outputDir = argv[++index] || "";
    else throw new Error(`Unknown argument ${value}.`);
  }
  if (!output.outputDir) throw new Error("--output-dir requires a path.");
  return output;
}
