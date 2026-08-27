import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  analyzeCoreMlProgramCorpusArtifact,
  baselineFromReceipt,
  COREML_MLPROGRAM_CORPUS_PATH,
  readCoreMlProgramCorpus,
  receiptMatchesBaseline,
} from "./coreml-mlprogram-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readCoreMlProgramCorpus();
await mkdir(args.outputDir, { recursive: true });
const selected = args.ids.length
  ? args.ids.map((id) => manifest.artifacts.find((row) => row.id === id) || (() => { throw new Error(`Unknown artifact id ${id}.`); })())
  : manifest.artifacts;
const receipts = [];
let baselineUpdated = false;
for (const artifact of selected) {
  const receipt = await analyzeCoreMlProgramCorpusArtifact(artifact);
  receipts.push(receipt);
  const matched = artifact.baseline ? receiptMatchesBaseline(receipt, artifact) : null;
  console.log(`${artifact.id}: ${receipt.model_type}; ${receipt.operator_count} op(s); ${receipt.input_flexibility.kind}; ${matched == null ? "baseline pending" : matched ? "baseline matched" : "BASELINE MISMATCH"}`);
  if (matched === false) {
    if (args.updateBaselines) {
      artifact.baseline = baselineFromReceipt(receipt);
      baselineUpdated = true;
    } else {
      console.error(JSON.stringify({ expected: artifact.baseline, actual: baselineFromReceipt(receipt) }, null, 2));
      process.exitCode = 1;
    }
  }
}
if (baselineUpdated) {
  await writeFile(COREML_MLPROGRAM_CORPUS_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Updated deterministic baselines in ${COREML_MLPROGRAM_CORPUS_PATH}.`);
}
await writeFile(path.join(args.outputDir, "coreml-mlprogram-contract-corpus-sweep.json"), `${JSON.stringify({
  schema: "deepbom.coreml_mlprogram_contract_corpus_sweep.v1",
  generated_at: new Date().toISOString(),
  population_scope: manifest.population_scope,
  artifact_count: receipts.length,
  receipts,
}, null, 2)}\n`, "utf8");

function parseArgs(argv) {
  const output = { outputDir: ".local-validation/coreml-mlprogram-contract-corpus", ids: [], updateBaselines: false };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--output-dir") output.outputDir = path.resolve(argv[++index] || "");
    else if (argv[index] === "--artifact") output.ids.push(argv[++index] || "");
    else if (argv[index] === "--update-baselines") output.updateBaselines = true;
    else throw new Error(`Unknown argument ${argv[index]}.`);
  }
  if (!output.outputDir) throw new Error("--output-dir requires a path.");
  return output;
}
