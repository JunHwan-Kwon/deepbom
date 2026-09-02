import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(".");
const policy = JSON.parse(await readFile(path.join(root, "config", "artifact-ir-consumer-policy.v1.json"), "utf8"));
if (policy.schema !== "deepbom.artifact_ir_consumer_policy.v1") throw new Error("Artifact IR consumer policy schema is invalid.");

const categories = ["native_ledger_readers", "canonical_ir_modules", "compatibility_surface_readers"];
const classified = new Map();
for (const category of categories) {
  for (const file of policy[category] || []) {
    if (classified.has(file)) throw new Error(`${file} is classified as both ${classified.get(file)} and ${category}.`);
    classified.set(file, category);
  }
}

const directReaders = [];
for (const file of await javascriptFiles(path.join(root, "web"))) {
  const source = await readFile(file, "utf8");
  if (/\banalysis\s*(?:\?\.)?\.\s*(?:ops|tensors)\b/.test(source)) {
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
console.log(`Artifact IR consumer policy passed (${directReaders.length} classified readers: ${counts.native_ledger_readers} native producers, ${counts.canonical_ir_modules} canonical modules, ${counts.compatibility_surface_readers} IR-backed compatibility surfaces).`);

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
