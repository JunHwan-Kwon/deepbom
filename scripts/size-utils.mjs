import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";

export function kibToBytes(value) {
  return Math.max(1, value) * 1024;
}

export function mibToBytes(value) {
  return Math.max(1, value) * 1024 * 1024;
}

export function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

export async function collectFileSizes(root, { relativeRoot = root, extensions = null, ignoredDirs = new Set() } = {}) {
  const rootPath = path.resolve(root);
  const relativeRootPath = path.resolve(relativeRoot);
  const files = await walkFileSizes(rootPath, { relativeRootPath, extensions, ignoredDirs });
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

async function walkFileSizes(dir, options) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (options.ignoredDirs.has(entry.name)) continue;
      files.push(...await walkFileSizes(fullPath, options));
      continue;
    }
    if (!entry.isFile()) continue;
    if (options.extensions && !options.extensions.has(path.extname(entry.name).toLowerCase())) continue;
    const info = await stat(fullPath);
    files.push({
      path: normalizePath(path.relative(options.relativeRootPath, fullPath)),
      bytes: info.size,
    });
  }
  return files;
}
