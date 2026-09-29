import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { parse } from "acorn";

const roots = ["web", "src", "bin", "worker", "native", "channels", "protected"];
const excluded = new Set(["node_modules", "target", "vendor", "pkg", ".git", ".local-validation", "__pycache__"]);
const files = [];
function walk(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isSymbolicLink() || excluded.has(entry.name)) continue;
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(file);
    else if (/\.(js|mjs|rs|py|cpp|c|h)$/.test(file)) files.push(file);
  }
}
roots.forEach(walk);
if (fs.existsSync("index.js")) files.push("index.js");
files.sort();
const hash = text => createHash("sha256").update(text).digest("hex");
const registry = JSON.parse(fs.readFileSync("config/scalar-types.v1.json", "utf8"));
const names = new Set(registry.types.flatMap(row => [row.name, ...row.aliases]));
const exceptions = JSON.parse(fs.readFileSync("config/common-rule-exceptions.v1.json", "utf8"));
const duplicateCandidates = new Map(), widthTables = [], nativeScalarTables = [], inventory = [], errors = [];
const generated = file => /(?:[._-]generated|rulepack[._-])/.test(path.basename(file));
for (const file of files) {
  const source = fs.readFileSync(file, "utf8");
  const row = { file, sha256: hash(source), language: path.extname(file).slice(1),
    role: generated(file) ? "generated_source" : file.includes("conformance") ? "independent_reconstruction" : "runtime",
    functions: 0, calculation_candidates: 0 };
  inventory.push(row);
  if (!/\.(js|mjs)$/.test(file)) {
    row.functions = (source.match(/\b(?:fn|def)\s+\w+\s*\(/g) || []).length;
    row.calculation_candidates = (source.match(/\b(?:fn|def)\s+\w*(?:bytes|bits|count|macs|shape|quant|round|product|sum)\w*\s*\(/gi) || []).length;
    if (file.endsWith(".rs") && !generated(file)) {
      const widths = [...source.matchAll(/"([A-Z][A-Z0-9_]*)"(?:\s*\|\s*"[A-Z][A-Z0-9_]*")*\s*=>\s*(?:Some\()?\d+(?:\.\d+)?/g)]
        .filter(match => names.has(match[1]));
      if (widths.length >= 3) {
        const body_sha256 = hash(widths.map(match => match[0].replace(/\s+/g, " ")).join("\n"));
        const reason = exceptions.non_width_tables?.find(row => row.file === file && row.body_sha256 === body_sha256)?.reason;
        nativeScalarTables.push({ file, body_sha256, disposition: reason || "unexpected_duplicate" });
        if (!reason) {
          errors.push(`${file}: unreviewed Rust scalar lookup (${body_sha256}); generate storage widths from config/scalar-types.v1.json.`);
        }
      }
    }
    continue;
  }
  // Syntax failures are failures, not silently skipped files.
  let ast;
  try { ast = parse(source, { ecmaVersion: "latest", sourceType: "module", locations: true }); }
  catch (error) { errors.push(`${file}: ${error.message}`); continue; }
  function visit(node, parent) {
    if (!node || typeof node !== "object") return;
    if (/Function/.test(node.type) && node.body) {
      row.functions++;
      const name = node.id?.name || parent?.id?.name || "<anonymous>";
      if (/(?:bytes|bits|count|macs|shape|quant|round|product|sum|exact|dtype|float|axis)/i.test(name)) {
        row.calculation_candidates++;
        const body = source.slice(node.body.start, node.body.end).replace(/\s+/g, " ");
        if (body.length >= 80 && !generated(file)) {
          const key = hash(body), entries = duplicateCandidates.get(key) || [];
          entries.push({ file, name, line: node.loc.start.line }); duplicateCandidates.set(key, entries);
        }
      }
    }
    if (node.type === "ObjectExpression" || node.type === "ArrayExpression") {
      const pairs = node.type === "ObjectExpression"
        ? node.properties.map(p => [p.key?.name ?? p.key?.value, p.value])
        : node.elements.filter(p => p?.type === "ArrayExpression").map(p => [p.elements[0]?.value, p.elements[1]]);
      const widths = pairs.filter(([name, value]) => names.has(name) && value?.type === "Literal" && Number.isInteger(value.value));
      if (widths.length >= 3) {
        const body_sha256 = hash(source.slice(node.start, node.end).replace(/\s+/g, " "));
        const purpose = exceptions.scalar_width_oracles?.find(row => row.file === file && row.body_sha256 === body_sha256)?.reason;
        widthTables.push({ file, line: node.loc.start.line, body_sha256, names: widths.map(([name]) => name), disposition: purpose || "unexpected_duplicate" });
        if (!purpose) errors.push(`${file}:${node.loc.start.line}: duplicated scalar width/range table; use tensor-size.js or document a distinct native contract.`);
      }
    }
    for (const [key, value] of Object.entries(node)) {
      if (["loc", "start", "end"].includes(key)) continue;
      if (Array.isArray(value)) value.forEach(child => visit(child, node));
      else if (value && typeof value === "object") visit(value, node);
    }
  }
  visit(ast, null);
}
const duplicateGroups = [...duplicateCandidates.entries()].filter(([, entries]) => new Set(entries.map(row => row.file)).size > 1)
  .map(([body_sha256, occurrences]) => ({ body_sha256, occurrences })).sort((a, b) => a.body_sha256.localeCompare(b.body_sha256));
const reviewed = exceptions.exceptions;
for (const row of exceptions.non_width_tables || []) if (!nativeScalarTables.some(table => table.file === row.file && table.body_sha256 === row.body_sha256)) {
  errors.push(`Stale native scalar-table exception: ${row.file}`);
}
for (const oracle of exceptions.scalar_width_oracles || []) if (widthTables.filter(row => row.file === oracle.file && row.body_sha256 === oracle.body_sha256).length !== 1) {
  errors.push(`Missing or duplicated width oracle: ${oracle.file}`);
}
const occurrencesKey = rows => rows.map(row => `${row.file}:${row.name}`).sort().join("\n");
for (const group of duplicateGroups) {
  const exception = reviewed.find(row => row.body_sha256 === group.body_sha256);
  if (!exception?.reason || occurrencesKey(exception.occurrences) !== occurrencesKey(group.occurrences)) {
    errors.push(`Unreviewed duplicate calculation: ${occurrencesKey(group.occurrences)}`);
  } else group.disposition = { category: exception.category, reason: exception.reason };
}
for (const exception of reviewed) if (!duplicateGroups.some(group => group.body_sha256 === exception.body_sha256)) {
  errors.push(`Stale common-rule exception: ${exception.body_sha256}`);
}
const report = {
  schema: "deepbom.common_rule_inventory.v1",
  scope: { roots, excluded_directories: [...excluded], boundary: "All application source files in these roots are enumerated. JS functions/tables are parsed; Rust/Python/C++ candidates also require semantic review. Textual equality is a candidate, not proof of equivalent contracts." },
  source_file_count: inventory.length, files: inventory, scalar_width_tables: widthTables,
  native_scalar_tables: nativeScalarTables,
  remaining_duplicate_calculation_candidates: duplicateGroups,
};
if (process.argv.includes("--write")) {
  fs.mkdirSync("docs/reviews", { recursive: true });
  fs.writeFileSync("docs/reviews/common-calculation-inventory.json", JSON.stringify(report, null, 2) + "\n");
}
assert.deepEqual(errors, [], errors.join("\n"));
console.log(`Common rule inventory: ${files.length} source files; ${widthTables.length} explicitly retained width oracle; ${duplicateGroups.length} duplicate calculation candidates recorded for semantic review; no unowned scalar-width table.`);
