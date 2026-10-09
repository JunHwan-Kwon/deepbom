import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { directAnalysisLedgerReads } from "./artifact-consumer-analysis.mjs";

const root = path.resolve(".");
const policy = JSON.parse(await readFile(path.join(root, "config", "artifact-ir-consumer-policy.v1.json"), "utf8"));
if (policy.schema !== "deepbom.artifact_ir_consumer_policy.v1") throw new Error("Artifact IR consumer policy schema is invalid.");

const categories = ["native_ledger_readers", "canonical_ir_modules", "compatibility_facade_modules", "compatibility_surface_readers"];
const classified = new Map();
for (const category of categories) {
  for (const file of policy[category] || []) {
    if (classified.has(file)) throw new Error(`${file} is classified as both ${classified.get(file)} and ${category}.`);
    classified.set(file, category);
  }
}

const compatibilityBudget = Number(policy.compatibility_surface_budget);
if (!Number.isSafeInteger(compatibilityBudget) || compatibilityBudget < 0) {
  throw new Error("Artifact IR compatibility-surface budget is invalid.");
}
if (policy.compatibility_surface_readers.length > compatibilityBudget) {
  throw new Error(`Artifact IR compatibility-surface debt grew to ${policy.compatibility_surface_readers.length}; budget is ${compatibilityBudget}.`);
}
const groupedCompatibilityReaders = Object.values(policy.compatibility_surface_groups || {}).flat();
if (new Set(groupedCompatibilityReaders).size !== groupedCompatibilityReaders.length) {
  throw new Error("Artifact IR compatibility-surface groups contain duplicate readers.");
}
const compatibilitySet = new Set(policy.compatibility_surface_readers);
const groupedSet = new Set(groupedCompatibilityReaders);
const ungroupedCompatibilityReaders = policy.compatibility_surface_readers.filter((file) => !groupedSet.has(file));
const unknownGroupedReaders = groupedCompatibilityReaders.filter((file) => !compatibilitySet.has(file));
if (ungroupedCompatibilityReaders.length || unknownGroupedReaders.length) {
  throw new Error([
    "Artifact IR compatibility-surface grouping drifted.",
    ungroupedCompatibilityReaders.length ? `Ungrouped compatibility readers: ${ungroupedCompatibilityReaders.join(", ")}` : "",
    unknownGroupedReaders.length ? `Grouped readers absent from compatibility policy: ${unknownGroupedReaders.join(", ")}` : "",
  ].filter(Boolean).join("\n"));
}
const retiredCompatibilityReaders = Object.values(policy.retired_compatibility_surface_groups || {}).flat();
if (new Set(retiredCompatibilityReaders).size !== retiredCompatibilityReaders.length) {
  throw new Error("Retired Artifact IR compatibility-surface groups contain duplicate modules.");
}
if (retiredCompatibilityReaders.some((file) => compatibilitySet.has(file))) {
  throw new Error("An Artifact IR surface cannot be both active compatibility debt and retired migration evidence.");
}

for (const source of [
  "analysis.ops", "const x=analysis; x['tensors']", "const a=analysis; const b=a; b?.ops",
  "const {tensors: values}=analysis", "let a; a=analysis; const {ops}=a || {}",
]) assert(directAnalysisLedgerReads(source).length, `Missed native read: ${source}`);
assert.equal(directAnalysisLedgerReads("const ops = model.program.operations").length, 0);
const directReaders = [];
for (const file of await javascriptFiles(path.join(root, "web"))) {
  const source = await readFile(file, "utf8");
  if (/\b(?:compute_weight_histogram|weightHistogram|TFLITE_WEIGHT_HISTOGRAM)\b/.test(source))
    throw new Error(`${file}: retired weight statistics bypass; use common Weight IR.`);
  if (directAnalysisLedgerReads(source, file).length) {
    directReaders.push(portable(path.relative(root, file)));
  }
}
directReaders.sort();
const unclassified = directReaders.filter((file) => !classified.has(file));
const stale = [...classified.keys()].filter((file) => !directReaders.includes(file));
if (unclassified.length || stale.length) {
  throw new Error([
    "Artifact IR native-consumer classification drifted.",
    unclassified.length ? `Unclassified direct readers: ${unclassified.join(", ")}` : "",
    stale.length ? `Stale classified readers: ${stale.join(", ")}` : "",
  ].filter(Boolean).join("\n"));
}
const retiredDirectReaders = retiredCompatibilityReaders.filter((file) => directReaders.includes(file));
if (retiredDirectReaders.length) {
  throw new Error(`Retired Artifact IR surface modules regained direct native reads: ${retiredDirectReaders.join(", ")}`);
}

for (const [file, contract] of Object.entries(policy.orchestration_contracts || {})) {
  const source = await readFile(path.join(root, file), "utf8");
  for (const fragment of contract.required_fragments || []) {
    if (!source.includes(fragment)) throw new Error(`${file} no longer satisfies required Artifact IR routing fragment: ${fragment}`);
  }
  for (const pattern of contract.forbidden_patterns || []) {
    if (new RegExp(pattern, "m").test(source)) throw new Error(`${file} contains forbidden raw-analysis routing: ${pattern}`);
  }
}

const counts = Object.fromEntries(categories.map((category) => [category, policy[category].length]));
const groupSummary = Object.entries(policy.compatibility_surface_groups || {}).map(([name, files]) => `${name}=${files.length}`).join(", ");
console.log(`Artifact IR consumer policy passed (${directReaders.length} classified readers: ${counts.native_ledger_readers} native producers, ${counts.canonical_ir_modules} canonical modules, ${counts.compatibility_facade_modules} compatibility facade, ${counts.compatibility_surface_readers}/${compatibilityBudget} direct surface readers; ${retiredCompatibilityReaders.length} migrated surfaces${groupSummary ? `; ${groupSummary}` : ""}).`);

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

function portable(value) { return value.split(path.sep).join("/"); }
