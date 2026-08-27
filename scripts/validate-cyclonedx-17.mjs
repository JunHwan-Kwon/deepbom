import { readFileSync } from "node:fs";

import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";

const files = process.argv.slice(2);
if (!files.length) {
  console.error("Usage: npm run validate:cyclonedx -- <bom.cdx.json> [...]");
  process.exit(2);
}
for (const file of files) {
  assertCycloneDx17(JSON.parse(readFileSync(file, "utf8")), file);
  console.log(`${file}: valid CycloneDX 1.7 JSON`);
}
