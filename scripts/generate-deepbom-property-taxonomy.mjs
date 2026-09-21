import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

import {
  DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES,
  DEEPBOM_STRUCTURED_EVIDENCE_PROPERTIES,
  componentPropertyReconciliationKind,
} from "../web/lib/deepbom-property-taxonomy.js";

const sourceFiles = ["web/lib/report-export-contracts.js", "web/lib/public-cyclonedx-export.js", "web/lib/report-mlbom-compat.js", "web/lib/safetensors-quantization-export.js", "web/lib/tensorrt-cyclonedx-properties.js"];
const names = new Set();
for (const sourceFile of sourceFiles) {
  const source = await readFile(sourceFile, "utf8");
  for (const match of source.matchAll(/(?:\[|property\()"((?:deepbom|mlbom):[^"]+)"/g)) names.add(match[1]);
}
const stable = new Set(DEEPBOM_RECONCILABLE_COMPONENT_PROPERTIES);
const deprecated = new Set(DEEPBOM_STRUCTURED_EVIDENCE_PROPERTIES);
const properties = [...names].sort().map((name) => ({
  name,
  status: stable.has(name) ? "stable" : deprecated.has(name) ? "deprecated" : "experimental",
  reconciliation: componentPropertyReconciliationKind(name),
}));
for (const name of stable) assert(names.has(name), `Stable property is not emitted by the exporter: ${name}`);
const document = {
  schema: "deepbom.property_taxonomy.v1",
  namespace: "deepbom",
  compatibility_namespaces: ["mlbom"],
  ownership: "DeepBOM vendor extension; no external registration or endorsement is claimed",
  source_files: sourceFiles,
  counts: {
    total: properties.length,
    stable: properties.filter((row) => row.status === "stable").length,
    experimental: properties.filter((row) => row.status === "experimental").length,
    deprecated: properties.filter((row) => row.status === "deprecated").length,
  },
  properties,
};
const output = `${JSON.stringify(document, null, 2)}\n`;
const outputPath = "docs/DEEPBOM_PROPERTY_TAXONOMY.v1.json";
if (process.argv.includes("--check")) {
  assert.equal(await readFile(outputPath, "utf8"), output, `${outputPath} is stale; run npm run generate:property-taxonomy.`);
  console.log(`DeepBOM property taxonomy is current (${properties.length} exact names).`);
} else {
  await writeFile(outputPath, output, "utf8");
  console.log(`Generated ${outputPath} (${properties.length} exact names).`);
}
