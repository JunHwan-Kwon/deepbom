import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { readHtmlModuleEntrypoints } from "./html-utils.mjs";
import { normalizePath } from "./path-utils.mjs";
import {
  appAssetToRepoPath,
  parseStringArrayConst,
  readSwAppAssets,
  readSwSource,
} from "./sw-utils.mjs";

const entryPoints = readHtmlModuleEntrypoints();
const swAppAssets = readSwAppAssets().sort();
const swSource = readSwSource();
const swShellAssets = parseStringArrayConst(swSource, "APP_SHELL_ASSETS");
const authenticatedPrefixes = parseStringArrayConst(swSource, "AUTHENTICATED_ASSET_PREFIXES");
const authenticatedAssetPaths = new Set(parseStringArrayConst(swSource, "AUTHENTICATED_ASSET_PATHS"));
const swRepoPathToAsset = new Map(swAppAssets.map((asset) => [normalizePath(appAssetToRepoPath(asset)), asset]));
const protectedUncachedAssets = new Set([
  "./lib/report-raw-entry.js",
]);

const appLocalImports = collectLocalImports(entryPoints);

const missing = appLocalImports.filter((asset) => !swAppAssets.includes(asset) && !protectedUncachedAssets.has(asset));
if (missing.length) {
  throw new Error(`web/sw.js APP_ASSETS is missing app imports: ${missing.join(", ")}`);
}

const protectedCached = swAppAssets.filter((asset) => authenticatedAssetPaths.has(assetToPathSuffix(asset))
  || asset.startsWith("./protected/"));
if (protectedCached.length) {
  throw new Error(`web/sw.js APP_ASSETS must not cache gated report formatter assets: ${protectedCached.join(", ")}`);
}
if (!authenticatedPrefixes.includes("/web/protected/")
  || !swSource.includes("APP_SHELL_ASSETS.filter((asset) => !isAuthenticatedAssetPath(assetToPathSuffix(asset)))")
  || !swSource.includes("if (isAuthenticatedAssetPath(url.pathname))")) {
  throw new Error("web/sw.js must fail closed before installing or fetching authenticated module assets.");
}

const unknownShellAssets = swShellAssets.filter((asset) => !swAppAssets.includes(asset));
if (!swShellAssets.length || unknownShellAssets.length) {
  throw new Error(`web/sw.js APP_SHELL_ASSETS must be a non-empty APP_ASSETS subset: ${unknownShellAssets.join(", ")}`);
}
if (swShellAssets.length >= swAppAssets.length / 2) {
  throw new Error(`web/sw.js app shell is not bounded (${swShellAssets.length}/${swAppAssets.length} assets).`);
}

const missingFiles = swAppAssets
  .map((asset) => [asset, appAssetToRepoPath(asset)])
  .filter(([, path]) => path && !existsSync(path));
if (missingFiles.length) {
  throw new Error(`web/sw.js APP_ASSETS references missing files: ${missingFiles.map(([asset]) => asset).join(", ")}`);
}

console.log(`Service worker app import coverage passed (${appLocalImports.length} imports, ${swShellAssets.length} install assets, ${swAppAssets.length} on-demand cacheable assets).`);

function assetToPathSuffix(asset) {
  if (asset === "./") return "/web/";
  if (asset.startsWith("./")) return `/web/${asset.slice(2)}`;
  if (asset.startsWith("../")) return `/${asset.slice(3)}`;
  return asset;
}

function collectLocalImports(entries) {
  const visited = new Set();
  const imports = new Set();
  const queue = entries.map(normalizePath);

  while (queue.length) {
    const filePath = queue.shift();
    if (!filePath || visited.has(filePath) || !existsSync(filePath)) {
      continue;
    }
    visited.add(filePath);

    for (const specifier of parseStaticImportSpecifiers(readFileSync(filePath, "utf8"))) {
      if (!isCacheableLocalSpecifier(specifier)) {
        continue;
      }
      const resolved = resolveImportPath(filePath, specifier);
      const asset = swRepoPathToAsset.get(resolved);
      if (asset) {
        imports.add(asset);
      } else {
        imports.add(importSpecifierToAppAsset(filePath, specifier));
      }
      if (resolved.startsWith("web/")) {
        queue.push(resolved);
      }
    }
  }

  return [...imports].sort();
}

function parseStaticImportSpecifiers(source) {
  const specs = [];
  for (const match of source.matchAll(/import\s+(?:[^"'()]+?\s+from\s+)?["']([^"']+)["']/g)) {
    specs.push(match[1]);
  }
  for (const match of source.matchAll(/import\(\s*["']([^"']+)["']\s*\)/g)) {
    specs.push(match[1]);
  }
  return specs;
}

function isCacheableLocalSpecifier(specifier) {
  return specifier.startsWith("./") || specifier.startsWith("../pkg/");
}

function resolveImportPath(fromFile, specifier) {
  return normalizePath(path.join(path.dirname(fromFile), specifier));
}

function importSpecifierToAppAsset(fromFile, specifier) {
  const resolved = resolveImportPath(fromFile, specifier);
  if (resolved.startsWith("web/")) {
    return `./${resolved.slice(4)}`;
  }
  if (resolved.startsWith("pkg/")) {
    return `../${resolved}`;
  }
  return specifier;
}
