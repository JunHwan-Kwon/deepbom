import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";
import { collectFileSizes } from "./size-utils.mjs";

const roots = ["web"];
const errors = [];
const exportCache = new Map();
let checkedImports = 0;
let checkedUsage = 0;

const files = (await Promise.all(roots.map((root) => collectFileSizes(root, { relativeRoot: "." }))))
  .flat()
  .map((file) => file.path)
  .filter((file) => file.endsWith(".js"))
  .filter((file) => !file.includes("/protected/deepbom/pkg/"));

for (const file of files) {
  const source = readFileSync(file, "utf8");
  const body = sourceWithoutStaticImports(source);
  for (const item of parseStaticImports(source)) {
    for (const localName of item.localNames) {
      checkedUsage += 1;
      if (!identifierUsed(body, localName)) {
        errors.push(`${file}: imports ${localName} from ${item.specifier}, but does not use it.`);
      }
    }

    const resolved = normalizePath(path.join(path.dirname(file), item.specifier));
    if (!isLocalManagedImport(item.specifier, resolved)) {
      continue;
    }
    if (!existsSync(resolved)) {
      errors.push(`${file}: local import target is missing: ${item.specifier}`);
      continue;
    }
    const exports = exportedNames(resolved);
    for (const name of item.names) {
      checkedImports += 1;
      if (!exports.has(name)) {
        errors.push(`${file}: imports ${name} from ${item.specifier}, but ${resolved} does not export it.`);
      }
    }
  }
}

if (errors.length) {
  console.error("Browser import contract check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(`Browser import contract check passed (${checkedImports} named/default exports and ${checkedUsage} local import usages checked across ${files.length} files).`);

function parseStaticImports(source) {
  const imports = [];
  for (const match of source.matchAll(/import\s+([\s\S]*?)\s+from\s+["']([^"']+)["']/g)) {
    const bindings = importBindings(match[1]);
    imports.push({
      specifier: match[2],
      names: bindings.map((item) => item.imported).filter(Boolean),
      localNames: bindings.map((item) => item.local).filter(Boolean),
    });
  }
  return imports;
}

function importBindings(clause) {
  const bindings = [];
  const trimmed = clause.trim();
  const namedBlock = trimmed.match(/\{([\s\S]*?)\}/);
  if (namedBlock) {
    for (const item of namedBlock[1].split(",")) {
      const part = item.trim();
      if (!part) continue;
      const [imported, local = imported] = part.split(/\s+as\s+/).map((value) => value.trim());
      bindings.push({ imported, local });
    }
  }

  const beforeNamed = namedBlock ? trimmed.slice(0, namedBlock.index).trim().replace(/,$/, "").trim() : trimmed;
  if (beforeNamed && !beforeNamed.startsWith("*")) {
    bindings.push({ imported: "default", local: beforeNamed });
  } else if (beforeNamed.startsWith("*")) {
    const namespace = beforeNamed.match(/\*\s+as\s+([a-zA-Z_$][\w$]*)/);
    if (namespace) bindings.push({ imported: "", local: namespace[1] });
  }

  return bindings;
}

function isLocalManagedImport(specifier, resolved) {
  if (!specifier.startsWith(".") && !specifier.startsWith("..")) {
    return false;
  }
  return resolved.startsWith("web/") || resolved.startsWith("pkg/");
}

function exportedNames(file) {
  if (exportCache.has(file)) {
    return exportCache.get(file);
  }
  const source = readFileSync(file, "utf8");
  const names = new Set();

  for (const match of source.matchAll(/export\s+(?:async\s+)?function\s+([a-zA-Z_$][\w$]*)/g)) {
    names.add(match[1]);
  }
  for (const match of source.matchAll(/export\s+(?:const|let|var|class)\s+([a-zA-Z_$][\w$]*)/g)) {
    names.add(match[1]);
  }
  for (const match of source.matchAll(/export\s*\{([\s\S]*?)\}/g)) {
    for (const item of match[1].split(",")) {
      const part = item.trim();
      if (!part) continue;
      const pieces = part.split(/\s+as\s+/).map((value) => value.trim());
      names.add(pieces[1] || pieces[0]);
    }
  }
  if (/export\s+default\b/.test(source)) {
    names.add("default");
  }

  exportCache.set(file, names);
  return names;
}

function sourceWithoutStaticImports(source) {
  return source.replace(/import\s+[\s\S]*?\s+from\s+["'][^"']+["'];?/g, "");
}

function identifierUsed(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`).test(source);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
