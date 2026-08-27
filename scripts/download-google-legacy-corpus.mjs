import path from "node:path";
import process from "node:process";

import {
  GOOGLE_LEGACY_MANIFEST_PATH,
  ensureGoogleLegacyArtifact,
  googleLegacyCacheDir,
  readGoogleLegacyManifest,
} from "./google-legacy-corpus-lib.mjs";

const args = parseArgs(process.argv.slice(2));
const manifest = await readGoogleLegacyManifest(args.manifest);
const cacheDir = path.resolve(args.cacheDir || googleLegacyCacheDir());
const selected = args.artifactIds.length
  ? manifest.artifacts.filter((artifact) => args.artifactIds.includes(artifact.id))
  : manifest.artifacts;
if (selected.length !== (args.artifactIds.length || manifest.artifacts.length)) {
  throw new Error("One or more requested Google legacy artifact ids are absent from the manifest.");
}

for (let index = 0; index < selected.length; index += 1) {
  const artifact = selected[index];
  process.stdout.write(`[${index + 1}/${selected.length}] ${artifact.id}: verify`);
  const result = await ensureGoogleLegacyArtifact(artifact, cacheDir, {
    offline: args.offline,
    onProgress: ({ received, total }) => {
      if (process.stdout.isTTY) {
        process.stdout.write(`\r[${index + 1}/${selected.length}] ${artifact.id}: ${formatBytes(received)} / ${formatBytes(total)}`);
      }
    },
  });
  console.log(`${result.downloaded ? " downloaded" : ""}${result.extracted ? " extracted" : ""} ready`);
}
console.log(`Verified cache: ${cacheDir}`);

function parseArgs(argv) {
  const output = {
    manifest: GOOGLE_LEGACY_MANIFEST_PATH,
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
