import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";

const briefs = ["regulatory", "quality", "engineering"];
const sources = new Map();

for (const brief of briefs) {
  const relative = path.join("web", "evaluate", brief, "index.html");
  const source = await readFile(relative, "utf8");
  sources.set(brief, source);
  assert.match(source, new RegExp(`<title>${capitalize(brief)} Evaluation \\| DEEPBOM</title>`));
  assert.match(source, new RegExp(`https://deepbom\\.org/evaluate/${brief}/`));
  assert.match(source, /href="\/"/);
  for (const peer of briefs) assert.match(source, new RegExp(`>${capitalize(peer)}<`));
  assert.doesNotMatch(source, /CycloneDX\s+2\.0|#(?:990|175|1067|1075)|perspective audit/i);
  assert.doesNotMatch(source, /guarantee(?:s|d)?\s+(?:compliance|approval|safety)|submission-ready/i);
}

const regulatory = sources.get("regulatory");
for (const identifier of ["IEC 62304", "ISO 13485", "ISO 14971"]) assert.match(regulatory, new RegExp(identifier));
assert.match(regulatory, /does not reproduce licensed requirements or claim conformity/i);
assert.match(regulatory, /does not replace clinical validation/i);

const quality = sources.get("quality");
assert.match(quality, /npx deepbom self-test/);
assert.match(quality, /does not validate every model, target, or intended use/i);

const engineering = sources.get("engineering");
assert.match(engineering, /Rust analyzer compiled to WebAssembly/);
assert.match(engineering, /one primary maintainer/i);
assert.match(engineering, /--output-format sarif --gate defects/);

const buildSource = await readFile("scripts/build-pages.mjs", "utf8");
assert.match(buildSource, /copyDir\(path\.join\(root, "web", "evaluate"\), path\.join\(dist, "evaluate"\)\)/);
for (const brief of briefs) assert.match(buildSource, new RegExp(`evaluate/\\$\\{brief\\}/`));

console.log("Evaluation brief contract passed (bounded regulatory, quality, and engineering decision surfaces).\n");

function capitalize(value) {
  return value[0].toUpperCase() + value.slice(1);
}
