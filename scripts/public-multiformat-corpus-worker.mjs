import { writeFile } from "node:fs/promises";
import process from "node:process";

import { analyzePublicMultiformatArtifact, publicMultiformatCacheDir, readPublicMultiformatCorpus } from "./public-multiformat-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readPublicMultiformatCorpus(args.manifest);
const artifact = manifest.artifacts.find((row) => row.id === args.artifactId);
if (!artifact) throw new Error(`Unknown public multiformat artifact ${args.artifactId}.`);
const result = await analyzePublicMultiformatArtifact(args.cacheDir || publicMultiformatCacheDir(), artifact, { offline: args.offline });
await writeFile(args.output, `${JSON.stringify(result)}\n`, "utf8");

function parseArgs(argv) {
  const output = { manifest: "corpus/public-multiformat-corpus.v1.json", artifactId: "", cacheDir: "", output: "", offline: false };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--artifact") output.artifactId = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--output") output.output = required(argv, ++index, key);
    else if (key === "--offline") output.offline = true;
    else throw new Error(`Unknown argument: ${key}`);
  }
  if (!output.artifactId || !output.output) throw new Error("--artifact and --output are required.");
  return output;
}
function required(argv, index, key) { const value = argv[index]; if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`); return value; }
