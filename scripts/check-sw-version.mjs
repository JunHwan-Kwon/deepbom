import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { normalizePath } from "./path-utils.mjs";
import { parseAppAssets, readSwAppAssetPaths } from "./sw-utils.mjs";

const cacheVersionPattern = /tflite-wasm-static-audit-v(\d+)/;

function git(args) {
  return execFileSync("git", args, { encoding: "utf8" });
}

function cacheVersion(source, label) {
  const match = cacheVersionPattern.exec(source || "");
  if (!match) throw new Error(`Could not find service worker cache version in ${label}.`);
  return Number(match[1]);
}

function appAssetListChanged(currentSource, headSource) {
  return parseAppAssets(currentSource).join("\n") !== parseAppAssets(headSource).join("\n");
}

const changedPaths = git(["diff", "--name-only", "HEAD", "--"])
  .split(/\r?\n/)
  .filter(Boolean)
  .map(normalizePath);

const currentSwSource = readFileSync("web/sw.js", "utf8");
const headSwSource = git(["show", "HEAD:web/sw.js"]);
const cachedAssetPaths = new Set(readSwAppAssetPaths());
const changedCachedAssets = changedPaths.filter((path) => cachedAssetPaths.has(path) && path !== "web/sw.js");
const cacheManifestChanged = changedPaths.includes("web/sw.js") && appAssetListChanged(currentSwSource, headSwSource);

if (!changedCachedAssets.length && !cacheManifestChanged) {
  console.log("Service worker version bump check passed (no cached app asset changes).");
  process.exit(0);
}

if (!changedPaths.includes("web/sw.js")) {
  throw new Error(`Cached app assets changed without web/sw.js update: ${changedCachedAssets.join(", ")}`);
}

const currentVersion = cacheVersion(currentSwSource, "web/sw.js");
const headVersion = cacheVersion(headSwSource, "HEAD:web/sw.js");
if (currentVersion <= headVersion) {
  throw new Error(`web/sw.js cache version must increase when cached app assets change: HEAD v${headVersion}, current v${currentVersion}`);
}

console.log(`Service worker version bump check passed (v${headVersion} -> v${currentVersion}).`);
