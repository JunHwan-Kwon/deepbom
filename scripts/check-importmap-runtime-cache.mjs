import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";
import { readSwRuntimeCacheableSuffixes } from "./sw-utils.mjs";

const htmlPath = "web/index.html";
const html = readFileSync(htmlPath, "utf8");
const runtimeSuffixes = new Set(readSwRuntimeCacheableSuffixes());
const errors = [];

const importMaps = [...html.matchAll(/<script\s+type="importmap"\s*>([\s\S]*?)<\/script>/g)].map((match) => match[1]);
if (!importMaps.length) {
  errors.push(`${htmlPath} must define an importmap for browser runtime dependencies.`);
}

let checkedImports = 0;
for (const source of importMaps) {
  const parsed = parseImportMap(source);
  for (const [name, specifier] of Object.entries(parsed.imports || {})) {
    if (!specifier.startsWith("../node_modules/")) {
      continue;
    }
    checkedImports += 1;
    const repoPath = normalizePath(path.join(path.dirname(htmlPath), specifier));
    const suffix = `/${repoPath}`;
    if (!existsSync(repoPath)) {
      errors.push(`importmap entry ${name} points to missing file ${repoPath}.`);
    }
    if (!runtimeSuffixes.has(suffix)) {
      errors.push(`importmap entry ${name} (${specifier}) is not listed in RUNTIME_CACHEABLE_SUFFIXES as ${suffix}.`);
    }
  }
}

if (!checkedImports) {
  errors.push(`${htmlPath} importmap did not include any ../node_modules runtime imports.`);
}

if (errors.length) {
  console.error("Import map runtime cache check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(`Import map runtime cache check passed (${checkedImports} runtime imports).`);

function parseImportMap(source) {
  try {
    return JSON.parse(source);
  } catch (error) {
    errors.push(`Invalid importmap JSON: ${error.message}`);
    return {};
  }
}
