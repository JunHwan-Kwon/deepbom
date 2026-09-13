import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { init, parse } from "es-module-lexer";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const policy = JSON.parse(await readFile(path.join(root, "config/model-ir-import-policy.v1.json"), "utf8"));
if (policy.schema !== "deepbom.model_ir_import_policy.v1") throw new Error("Model IR import policy schema is invalid.");
const violations = [];
const importEdges = [];
await init;

for (const directory of policy.source_roots) {
  for (const file of await javascriptFiles(path.join(root, directory))) {
    const portableRelative = portable(path.relative(root, file));
    const source = await readFile(file, "utf8");
    for (const [symbol, allowedImporters] of Object.entries(policy.protected_symbols)) {
      if (source.includes(symbol) && !allowedImporters.includes(portableRelative)) violations.push(`${portableRelative}: imports or calls protected Model IR symbol ${symbol}`);
    }
    for (const specifier of staticModuleSpecifiers(source, portableRelative)) {
      if (!specifier.startsWith(".")) continue;
      const resolved = portable(path.relative(root, resolveModuleSpecifier(file, specifier)));
      importEdges.push({ importer: portableRelative, imported: resolved });
      if (!resolved.startsWith(policy.internal_module_prefix)) continue;
      const allowed = policy.internal_module_importers[resolved];
      if (!allowed || !allowed.includes(portableRelative)) violations.push(`${portableRelative}: imports unregistered or restricted internal Model IR module ${resolved}`);
    }
  }
}

const internalFiles = new Set((await javascriptFiles(path.join(root, policy.internal_module_prefix))).map((file) => portable(path.relative(root, file))));
const registered = new Set(Object.keys(policy.internal_module_importers));
for (const file of internalFiles) if (!registered.has(file)) violations.push(`${file}: internal Model IR module is absent from the import policy`);
for (const file of registered) {
  if (!internalFiles.has(file)) violations.push(`${file}: import policy names a missing internal Model IR module`);
  for (const importer of policy.internal_module_importers[file]) if (!await fileExists(path.join(root, importer))) violations.push(`${file}: allowed importer does not exist: ${importer}`);
}
if (violations.length) throw new Error(`Model IR construction boundary violated:\n${violations.join("\n")}`);
const edgeCount = importEdges.filter((row) => row.imported.startsWith(policy.internal_module_prefix)).length;
console.log(`Model IR construction boundary passed (${internalFiles.size} registered internal modules; ${edgeCount} internal ESM edges; artifact-ir-context is the canonical construction path).`);

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
function staticModuleSpecifiers(source, file) {
  try { const [imports] = parse(source, file); return imports.filter((row) => row.d === -1).map((row) => source.slice(row.s, row.e)); }
  catch (error) { throw new Error(`${file}: ESM parse failed: ${error.message}`); }
}
function resolveModuleSpecifier(importer, specifier) { const resolved = path.resolve(path.dirname(importer), specifier); return path.extname(resolved) ? resolved : `${resolved}.js`; }
function portable(value) { return value.split(path.sep).join("/"); }
async function fileExists(file) { try { await readFile(file); return true; } catch { return false; } }
