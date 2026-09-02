import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(".");
const allowedUncheckedImport = path.normalize("web/lib/artifact-ir-context.js");
const boundaryChecker = path.normalize("scripts/check-artifact-ir-import-boundary.mjs");
const roots = ["web", "bin", "scripts"];
const violations = [];

for (const directory of roots) {
  for (const file of await javascriptFiles(path.join(root, directory))) {
    const relative = path.normalize(path.relative(root, file));
    const source = await readFile(file, "utf8");
    if (source.includes("buildArtifactEvidenceIrUnchecked") && relative !== allowedUncheckedImport && relative !== boundaryChecker
      && relative !== path.normalize("web/lib/artifact-ir.js")) {
      violations.push(`${relative}: imports or calls the unchecked Artifact IR builder`);
    }
    if (/\bbuildArtifactEvidenceIr\b/.test(source)) {
      violations.push(`${relative}: uses the removed direct Artifact IR builder`);
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
