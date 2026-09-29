import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { EVIDENCE_IR_NAME, EVIDENCE_IR_LAYERS } from "../web/lib/evidence-ir.js";

const base = "https://deepbom.org/schemas/";
const family = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: `${base}deepbom-evidence-ir-v1.schema.json`,
  title: EVIDENCE_IR_NAME,
  description: "Draft DEEPBOM-maintained family entry point for one IR document. This is not an aggregate bundle and does not authenticate evidence. Member semantic validators and exact source bindings remain required.",
  oneOf: EVIDENCE_IR_LAYERS.map(layer => ({ $ref: layer.json_schema })),
};
const catalog = {
  schema: "deepbom.evidence_ir_catalog.v1", name: EVIDENCE_IR_NAME,
  status: "draft", catalog_version: "1.0.0", json_schema: family.$id,
  layers: EVIDENCE_IR_LAYERS,
  compatibility_catalog: { index: "https://deepbom.org/schemas/compatibility/index.json", schema: "https://deepbom.org/schemas/deepbom-evidence-compatibility-v1.schema.json", explorer: "https://deepbom.org/guides/evidence-ir/compatibility/" },
  compatibility: "Current contracts only; no retired schema, hash-field or selector aliases. Preserve archived evidence separately and regenerate new documents from original inputs.",
  validation: "JSON Schema checks structure. Member validators additionally check digests, source bindings, references, exact counts and semantic consistency. Authenticity and external standards conformance are separate.",
};
for (const [file, value] of [
  ["docs/schemas/deepbom-evidence-ir-v1.schema.json", family],
  ["docs/evidence-ir/catalog.json", catalog],
]) {
  const content = JSON.stringify(value, null, 2) + "\n";
  if (process.argv.includes("--check")) {
    if (await readFile(file, "utf8") !== content) throw new Error(`Stale Evidence IR catalog/schema: ${file}`);
  } else {
    await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, content);
  }
}
console.log("Evidence IR catalog and schema entry points: five layers; canonical Provenance IR contract.");
