import process from "node:process";

import {
  ensurePublicArtifactFiles,
  publicMultiformatCacheDir,
  readPublicMultiformatCorpus,
} from "./public-multiformat-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readPublicMultiformatCorpus(args.manifest);
const selected = manifest.artifacts.filter((artifact) => (!args.formats.length || args.formats.includes(artifact.format))
  && (!args.artifactIds.length || args.artifactIds.includes(artifact.id)));
const totalBytes = selected.flatMap((artifact) => artifact.files).reduce((sum, file) => sum + file.size_bytes, 0);
if (totalBytes > args.maxTotalGib * 1024 ** 3) {
  throw new Error(`Selected corpus is ${formatBytes(totalBytes)}, above the explicit ${args.maxTotalGib} GiB download limit.`);
}
const cacheDir = args.cacheDir || publicMultiformatCacheDir();
let downloaded = 0;
for (let index = 0; index < selected.length; index += 1) {
  const artifact = selected[index];
  const files = await ensurePublicArtifactFiles(cacheDir, artifact, { offline: args.offline });
  downloaded += files.filter((file) => file.downloaded).length;
  console.log(`[${index + 1}/${selected.length}] ${artifact.id}: ${files.some((file) => file.downloaded) ? "downloaded and verified" : "cache verified"}`);
}
console.log(`Verified ${selected.length} artifact records / ${formatBytes(totalBytes)}; downloaded ${downloaded} file(s).`);

function parseArgs(argv) {
  const output = { manifest: "corpus/public-multiformat-corpus.v1.json", cacheDir: "", formats: [], artifactIds: [], maxTotalGib: 4, offline: false };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--format") output.formats.push(required(argv, ++index, key));
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
    else if (key === "--max-total-gib") output.maxTotalGib = positiveNumber(required(argv, ++index, key), key);
    else if (key === "--offline") output.offline = true;
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}

function required(argv, index, key) { const value = argv[index]; if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`); return value; }
function positiveNumber(value, key) { const number = Number(value); if (!(number > 0)) throw new Error(`${key} must be positive.`); return number; }
function formatBytes(value) { return `${(value / 1024 ** 2).toFixed(1)} MiB`; }
