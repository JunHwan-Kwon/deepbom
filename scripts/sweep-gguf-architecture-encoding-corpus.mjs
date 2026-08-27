import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  analyzeGgufCorpusArtifact,
  baselineFromReceipt,
  ggufCorpusCacheDir,
  readGgufArchitectureEncodingCorpus,
  receiptMatchesBaseline,
} from "./gguf-architecture-encoding-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readGgufArchitectureEncodingCorpus();
const cacheDir = args.cacheDir || ggufCorpusCacheDir();
const outputDir = path.resolve(args.outputDir);
await mkdir(outputDir, { recursive: true });
const selectedArtifacts = args.artifactIds.length
  ? args.artifactIds.map((id) => manifest.artifacts.find((row) => row.id === id) || (() => { throw new Error(`Unknown artifact id ${id}.`); })())
  : manifest.artifacts;
const receipts = [];
for (const artifact of selectedArtifacts) {
  const receipt = await analyzeGgufCorpusArtifact(cacheDir, artifact, { offline: args.offline });
  receipts.push(receipt);
  const matched = artifact.baseline ? receiptMatchesBaseline(receipt, artifact) : null;
  console.log(`${artifact.id}: ${receipt.architecture}; ${receipt.tensor_count} tensors; ${receipt.encoding_histogram.map((row) => `${row.encoding}:${row.tensor_count}`).join(", ")}; ${matched == null ? "baseline pending" : matched ? "baseline matched" : "BASELINE MISMATCH"}`);
  if (matched === false) {
    console.error(JSON.stringify({ expected: artifact.baseline, actual: baselineFromReceipt(receipt) }, null, 2));
    process.exitCode = 1;
  }
}
await writeFile(path.join(outputDir, "gguf-architecture-encoding-corpus-sweep.json"), `${JSON.stringify({
  schema: "deepbom.gguf_architecture_encoding_corpus_sweep.v1",
  generated_at: new Date().toISOString(),
  population_scope: manifest.population_scope,
  redistribution_boundary: manifest.redistribution_boundary,
  artifact_count: receipts.length,
  receipts,
}, null, 2)}\n`, "utf8");

function parseArgs(argv) {
  const output = { offline: false, cacheDir: "", outputDir: ".local-validation/gguf-architecture-encoding-corpus", artifactIds: [] };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--offline") output.offline = true;
    else if (argv[index] === "--cache-dir") output.cacheDir = argv[++index] || "";
    else if (argv[index] === "--output-dir") output.outputDir = argv[++index] || "";
    else if (argv[index] === "--artifact") output.artifactIds.push(argv[++index] || "");
    else throw new Error(`Unknown argument ${argv[index]}.`);
  }
  if (!output.outputDir) throw new Error("--output-dir requires a path.");
  return output;
}
