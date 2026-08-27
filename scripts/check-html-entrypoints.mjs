import { existsSync } from "node:fs";
import { HTML_ENTRYPOINTS, readHtmlLocalAssets, readHtmlModuleEntrypoints, repoPathToAppAsset } from "./html-utils.mjs";
import { readSwAppAssets } from "./sw-utils.mjs";

const swAppAssets = new Set(readSwAppAssets());
const htmlAssets = readHtmlLocalAssets();
const moduleEntrypoints = readHtmlModuleEntrypoints();
const errors = [];

for (const htmlPath of HTML_ENTRYPOINTS) {
  if (!existsSync(htmlPath)) {
    errors.push(`HTML entrypoint is missing: ${htmlPath}`);
    continue;
  }
  const asset = repoPathToAppAsset(htmlPath);
  if (asset && !swAppAssets.has(asset)) {
    errors.push(`web/sw.js APP_ASSETS is missing HTML entrypoint ${asset} (${htmlPath}).`);
  }
}

for (const { htmlPath, specifier, repoPath } of htmlAssets) {
  const asset = repoPathToAppAsset(repoPath);
  if (!existsSync(repoPath)) {
    errors.push(`${htmlPath} references missing local asset ${specifier} (${repoPath}).`);
  }
  if (asset && !swAppAssets.has(asset)) {
    errors.push(`${htmlPath} references ${specifier}, but web/sw.js APP_ASSETS is missing ${asset}.`);
  }
}

if (!moduleEntrypoints.length) {
  errors.push("No module script entrypoints were found in HTML.");
}

if (errors.length) {
  console.error("HTML entrypoint contract check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(
  `HTML entrypoint contract passed (${HTML_ENTRYPOINTS.length} HTML files, ${moduleEntrypoints.length} module scripts, ${htmlAssets.length} local assets).`,
);
