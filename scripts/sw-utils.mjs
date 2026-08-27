import { readFileSync } from "node:fs";

export function readSwSource(path = "web/sw.js") {
  return readFileSync(path, "utf8");
}

export function parseStringArrayConst(swSource, constName) {
  const block = new RegExp(`const ${constName} = \\[([\\s\\S]*?)\\];`).exec(swSource || "")?.[1] || "";
  return [...block.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
}

export function parseAppAssets(swSource) {
  return parseStringArrayConst(swSource, "APP_ASSETS");
}

export function parseRuntimeCacheableSuffixes(swSource) {
  return parseStringArrayConst(swSource, "RUNTIME_CACHEABLE_SUFFIXES");
}

export function appAssetToRepoPath(asset) {
  if (asset === "./") return "web/index.html";
  if (asset.startsWith("./")) return `web/${asset.slice(2)}`;
  if (asset.startsWith("../")) return asset.slice(3);
  return "";
}

export function readSwAppAssets(path = "web/sw.js") {
  return parseAppAssets(readSwSource(path));
}

export function readSwAppAssetPaths(path = "web/sw.js") {
  return readSwAppAssets(path)
    .map(appAssetToRepoPath)
    .filter(Boolean);
}

export function readSwRuntimeCacheableSuffixes(path = "web/sw.js") {
  return parseRuntimeCacheableSuffixes(readSwSource(path));
}
