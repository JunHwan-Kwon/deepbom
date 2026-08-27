import path from "node:path";
import process from "node:process";

import {
  hfCorpusCacheDir,
} from "./huggingface-community-corpus-lib.mjs";
import {
  GOOGLE_MODERN_COMPARATOR_PATH,
  ensureGoogleModernComparator,
  loadResolvedGoogleModernComparators,
  readGoogleModernComparators,
} from "./google-modern-comparator-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readGoogleModernComparators(args.manifest);
const resolved = await loadResolvedGoogleModernComparators(manifest);
const selected = args.artifactIds.length
  ? resolved.filter((row) => args.artifactIds.includes(row.artifact.id))
  : resolved;
if (selected.length !== (args.artifactIds.length || resolved.length)) {
  throw new Error("One or more requested modern comparator ids are absent.");
}
const cacheDir = path.resolve(args.cacheDir || hfCorpusCacheDir());
for (let index = 0; index < selected.length; index += 1) {
  const row = selected[index];
  process.stdout.write(`[${index + 1}/${selected.length}] ${row.artifact.id}: `);
  const result = await ensureGoogleModernComparator(row, cacheDir, {
    offline: args.offline,
    onProgress: ({ received, total }) => {
      if (process.stdout.isTTY) {
        process.stdout.write(`\r[${index + 1}/${selected.length}] ${row.artifact.id}: ${formatBytes(received)} / ${formatBytes(total)}`);
      }
    },
  });
  console.log(result.downloaded ? "downloaded and verified" : "cached and verified");
}
console.log(`Verified cache: ${cacheDir}`);

function parseArgs(argv) {
  const output = {
    manifest: GOOGLE_MODERN_COMPARATOR_PATH,
    cacheDir: "",
    offline: false,
    artifactIds: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--offline") output.offline = true;
    else if (key === "--manifest") output.manifest = required(argv, ++index, key);
    else if (key === "--cache-dir") output.cacheDir = required(argv, ++index, key);
    else if (key === "--artifact") output.artifactIds.push(required(argv, ++index, key));
    else throw new Error(`Unknown argument: ${key}`);
  }
  return output;
}

function required(argv, index, key) {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${key} requires a value.`);
  return value;
}

function formatBytes(value) {
  return `${(Number(value || 0) / 1024 ** 2).toFixed(2)} MiB`;
}
