import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { compatibilityDefinition } from "../config/evidence-compatibility.mjs";

export const DIRECTORY = "docs/evidence-ir/compatibility";
export const sha256 = value => createHash("sha256").update(value).digest("hex");
export const encode = value => JSON.stringify(value) + "\n";
export function compareVersions(a, b) {
  const parse = value => { if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value)) throw Error(`Invalid catalog version: ${value}`); return value.split(".").map(BigInt); };
  const left = parse(a), right = parse(b);
  for (let i = 0; i < 3; i++) if (left[i] !== right[i]) return left[i] < right[i] ? -1 : 1;
  return 0;
}
export async function buildSnapshot() {
  const definition = compatibilityDefinition();
  const files = new Set(["config/evidence-compatibility.mjs", "docs/schemas/deepbom-evidence-compatibility-v1.schema.json", "web/lib/evidence-ir.js", "web/lib/public-product-contracts.js", "web/lib/audit-output-contracts.js"]);
  for (const row of definition.mappings) for (const file of row.implementation) files.add(file);
  for (const node of definition.nodes) if (node.json_schema) files.add(`docs/schemas/${new URL(node.json_schema).pathname.split("/").pop()}`);
  const source_files = [];
  for (const path of [...files].sort()) source_files.push({ path, sha256: sha256(await readFile(path)) });
  return { ...definition, source_files };
}

const esc = value => String(value).replaceAll("|", "\\|").replaceAll("\n", " ");
export function renderMarkdown(snapshot) {
  const byId = new Map(snapshot.nodes.map(node => [node.id, node]));
  const lines = ["# Evidence IR compatibility catalog", "", `Catalog **${snapshot.catalog_version}** · ${snapshot.status}. Generated from [one mapping definition](../../../config/evidence-compatibility.mjs); do not edit this table.`, "", snapshot.scope, "", snapshot.validation_boundary, "", "[Browse and search on DEEPBOM](https://deepbom.org/guides/evidence-ir/compatibility/) · [Version index](index.json) · [Version policy and review](../COMPATIBILITY.md)", "", "## Input schemas → common IR → output schemas", "", "Mapping status describes an implementation path. It is not a per-artifact pass result. Source and destination paths are grouped selectors; this is not an executable transformation specification.", "", "| Mapping | From → to | Status | Field groups | Boundary | Implementation / checks |", "| --- | --- | --- | --- | --- | --- |"];
  for (const row of snapshot.mappings) lines.push(`| ${row.id} | ${byId.get(row.from).name} → ${byId.get(row.to).name} | ${row.status} | ${row.fields.map(f => `${esc(f.source)} → ${esc(f.target)}`).join("<br>") || "No implementation"} | ${esc(row.boundary)} | ${[...row.implementation, ...row.checks].map(file => `[${file.split("/").pop()}](../../../${file})`).join(" · ")} |`);
  lines.push("", "## Schema inventory", "", "| Endpoint | Kind | Contract | Schema |", "| --- | --- | --- | --- |");
  for (const node of snapshot.nodes) lines.push(`| ${node.name} | ${node.kind} | ${esc(node.contract)} | ${node.json_schema ? `[JSON Schema](${node.json_schema})` : "Native contract / no standalone JSON Schema"} |`);
  return lines.join("\n") + "\n";
}

export async function generate({ check = false } = {}) {
  const snapshot = await buildSnapshot(), bytes = encode(snapshot), file = `${snapshot.catalog_version}.json`;
  const snapshots = [];
  try {
    for (const name of (await readdir(DIRECTORY)).filter(name => /^\d+\.\d+\.\d+\.json$/.test(name)).sort()) {
      const previous = await readFile(`${DIRECTORY}/${name}`, "utf8"), data = JSON.parse(previous);
      if (name !== `${data.catalog_version}.json`) throw Error(`Invalid archived version: ${name}`);
      if (name === file && previous !== bytes) throw Error(`Compatibility ${snapshot.catalog_version} is immutable. Increment CATALOG_VERSION; do not rewrite a snapshot.`);
      snapshots.push({ version: data.catalog_version, status: data.status, file: name, sha256: sha256(previous) });
    }
  } catch (error) { if (error.code !== "ENOENT") throw error; }
  if (!snapshots.some(row => row.file === file)) snapshots.push({ version: snapshot.catalog_version, status: snapshot.status, file, sha256: sha256(bytes) });
  snapshots.sort((a, b) => compareVersions(a.version, b.version));
  const index = { schema: "deepbom.evidence_compatibility_index.v1", current: snapshot.catalog_version, snapshots };
  // Existing entries are append-only, including their byte digests.
  try {
    const prior = JSON.parse(await readFile(`${DIRECTORY}/index.json`, "utf8"));
    for (const entry of prior.snapshots) if (!snapshots.some(row => encode(row) === encode(entry))) throw Error(`Archived catalog ${entry.version} was removed or altered.`);
    if (compareVersions(snapshot.catalog_version, prior.current) < 0) throw Error("Current catalog version cannot move backwards.");
  } catch (error) { if (error.code !== "ENOENT") throw error; }
  const outputs = [[file, bytes], ["index.json", encode(index)], ["README.md", renderMarkdown(snapshot)]];
  for (const [name, content] of outputs) {
    const target = `${DIRECTORY}/${name}`;
    if (check) { if (await readFile(target, "utf8") !== content) throw Error(`Stale compatibility output: ${target}`); }
    else { await mkdir(DIRECTORY, { recursive: true }); await writeFile(target, content); }
  }
  console.log(`Compatibility ${snapshot.catalog_version}: ${snapshot.nodes.length} endpoints, ${snapshot.mappings.length} mappings; immutable snapshots and source digests checked.`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await generate({ check: process.argv.includes("--check") });
