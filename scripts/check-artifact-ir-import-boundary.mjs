import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const policyPath = path.join(root, "config", "artifact-ir-import-policy.v1.json");
const policy = JSON.parse(await readFile(policyPath, "utf8"));
if (policy.schema !== "deepbom.artifact_ir_import_policy.v1") throw new Error("Artifact IR import policy schema is invalid.");
const roots = policy.source_roots;
const violations = [];

for (const directory of roots) {
  for (const file of await javascriptFiles(path.join(root, directory))) {
    const relative = path.normalize(path.relative(root, file));
    const portableRelative = portable(relative);
    const source = await readFile(file, "utf8");
    for (const [symbol, allowedImporters] of Object.entries(policy.protected_symbols)) {
      if (source.includes(symbol) && !allowedImporters.some((allowed) => matchesPolicyPath(portableRelative, allowed))) {
        violations.push(`${portableRelative}: imports or calls protected Artifact IR symbol ${symbol}`);
      }
    }
    for (const symbol of policy.removed_symbols) {
      if (new RegExp(`\\b${escapeRegExp(symbol)}\\b`).test(source)) violations.push(`${portableRelative}: uses removed Artifact IR symbol ${symbol}`);
    }
    for (const specifier of staticModuleSpecifiers(source)) {
      if (!specifier.startsWith(".")) continue;
      const resolved = portable(path.relative(root, resolveModuleSpecifier(file, specifier)));
      if (!resolved.startsWith(policy.internal_module_prefix)) continue;
      if (!policy.internal_module_importers.some((allowed) => matchesPolicyPath(portableRelative, allowed))) {
        violations.push(`${portableRelative}: imports internal Artifact IR module ${resolved}`);
      }
    }
  }
}

if (violations.length) throw new Error(`Artifact IR construction boundary violated:\n${violations.join("\n")}`);
console.log("Artifact IR construction boundary passed (shared context is the only public construction path).");

async function javascriptFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (["node_modules", "dist", "public-source", ".local-validation"].includes(entry.name)) continue;
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await javascriptFiles(target));
    else if (/\.(?:js|mjs)$/.test(entry.name)) files.push(target);
  }
  return files;
}

function staticModuleSpecifiers(source) {
  const values = [];
  const pattern = /(?:^|[;\n]\s*)(?:import|export)\s+(?:[^;"']*?\s+from\s+)?["']([^"']+)["']/gm;
  for (const match of source.matchAll(pattern)) values.push(match[1]);
  return values;
}

function resolveModuleSpecifier(importer, specifier) {
  const resolved = path.resolve(path.dirname(importer), specifier);
  return path.extname(resolved) ? resolved : `${resolved}.js`;
}

function matchesPolicyPath(candidate, allowed) {
  return allowed.endsWith("/") ? candidate.startsWith(allowed) : candidate === allowed;
}

function portable(value) { return value.split(path.sep).join("/"); }
function escapeRegExp(value) { return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
