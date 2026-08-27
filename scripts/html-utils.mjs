import { readFileSync } from "node:fs";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";

export const HTML_ENTRYPOINTS = ["web/index.html", "web/medical.html", "web/admin.html", "web/auth-complete.html", "web/test.html"];

export function readHtmlModuleEntrypoints(htmlPaths = HTML_ENTRYPOINTS) {
  return uniqueSorted(readHtmlLocalAssets(htmlPaths, { moduleScriptsOnly: true }).map((asset) => asset.repoPath));
}

export function readHtmlLocalAssets(htmlPaths = HTML_ENTRYPOINTS, options = {}) {
  const assets = [];
  for (const htmlPath of htmlPaths) {
    const html = readFileSync(htmlPath, "utf8");
    for (const tag of parseHtmlTags(html)) {
      const attrs = parseAttributes(tag.attrs);
      const localRefs = localAssetRefs(tag.name, attrs, options);
      for (const ref of localRefs) {
        assets.push({
          htmlPath,
          specifier: ref,
          repoPath: resolveHtmlAsset(htmlPath, ref),
        });
      }
    }
  }
  return assets;
}

export function repoPathToAppAsset(repoPath) {
  const normalized = normalizePath(repoPath);
  if (normalized.startsWith("web/")) {
    return `./${normalized.slice(4)}`;
  }
  if (normalized.startsWith("pkg/") || normalized.startsWith("node_modules/")) {
    return `../${normalized}`;
  }
  return "";
}

function parseHtmlTags(html) {
  return [...html.matchAll(/<([a-zA-Z][\w-]*)\b([^>]*)>/g)].map((match) => ({
    name: match[1].toLowerCase(),
    attrs: match[2] || "",
  }));
}

function parseAttributes(source) {
  const attrs = new Map();
  for (const match of source.matchAll(/([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
    attrs.set(match[1].toLowerCase(), match[2] ?? match[3] ?? match[4] ?? "");
  }
  return attrs;
}

function localAssetRefs(tagName, attrs, options) {
  const refs = [];
  if (tagName === "script") {
    const src = attrs.get("src");
    const isModule = (attrs.get("type") || "").toLowerCase() === "module";
    if (src && (!options.moduleScriptsOnly || isModule)) {
      refs.push(src);
    }
  }
  if (!options.moduleScriptsOnly && tagName === "link") {
    const rel = (attrs.get("rel") || "").toLowerCase();
    const href = attrs.get("href");
    if (href && (rel.includes("stylesheet") || rel.includes("manifest"))) {
      refs.push(href);
    }
  }
  return refs.filter(isLocalAsset);
}

function resolveHtmlAsset(htmlPath, specifier) {
  const bare = specifier.split("?")[0];
  if (bare.startsWith("/web/") || bare.startsWith("/pkg/") || bare.startsWith("/node_modules/")) {
    return normalizePath(bare.slice(1));
  }
  return normalizePath(path.join(path.dirname(htmlPath), bare));
}

function isLocalAsset(specifier) {
  return specifier.startsWith("./") || specifier.startsWith("../") || specifier.startsWith("/web/") || specifier.startsWith("/pkg/") || specifier.startsWith("/node_modules/");
}

function uniqueSorted(values) {
  return [...new Set(values.map(normalizePath))].sort();
}
