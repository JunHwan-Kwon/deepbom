import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import Ajv from "ajv";
import addFormats from "ajv-formats";
import addDraft2019Formats from "ajv-formats-draft2019";

const schemaDir = fileURLToPath(new URL("../reference/cyclonedx/1.7/", import.meta.url));
const lock = JSON.parse(readFileSync(`${schemaDir}/schema-lock.json`, "utf8"));

function loadPinnedSchema(filename) {
  const bytes = readFileSync(`${schemaDir}/${filename}`);
  const actual = createHash("sha256").update(bytes).digest("hex");
  const expected = lock.files[filename];
  if (!expected || actual !== expected) throw new Error(`CycloneDX schema pin mismatch for ${filename}: ${actual}`);
  return JSON.parse(bytes.toString("utf8"));
}

const ajv = new Ajv({ allErrors: true, strict: true, strictRequired: false });
addFormats(ajv);
addDraft2019Formats(ajv);
ajv.addKeyword({ keyword: "meta:enum", schemaType: "object", valid: true });
for (const filename of ["spdx.schema.json", "jsf-0.82.schema.json", "cryptography-defs.schema.json"]) {
  ajv.addSchema(loadPinnedSchema(filename));
}
const validate = ajv.compile(loadPinnedSchema("bom-1.7.schema.json"));
const schemaIdentifier = "http://cyclonedx.org/schema/bom-1.7.schema.json";

export function cycloneDx17Validation(document) {
  validate(document);
  const errors = structuredClone(validate.errors || []);
  if (document?.$schema !== schemaIdentifier) errors.push({ instancePath: "/$schema", message: `must equal ${schemaIdentifier}` });
  if (document?.specVersion !== "1.7") errors.push({ instancePath: "/specVersion", message: "must equal 1.7" });
  return { valid: errors.length === 0, errors };
}

export function assertCycloneDx17(document, label = "CycloneDX document") {
  const result = cycloneDx17Validation(document);
  if (!result.valid) {
    const detail = result.errors.map((item) => `${item.instancePath || "/"} ${item.message}`).join("; ");
    throw new Error(`${label} does not conform to the pinned official CycloneDX 1.7 schema: ${detail}`);
  }
}
