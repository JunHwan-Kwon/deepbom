import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import Ajv from "ajv";

const root = new URL("../reference/spdx/2.3/", import.meta.url);
const lock = JSON.parse(readFileSync(new URL("schema-lock.json", root), "utf8"));
const bytes = readFileSync(new URL("spdx-schema.json", root));
assert.equal(createHash("sha256").update(bytes).digest("hex"), lock.sha256, "SPDX schema pin mismatch");
const validate = new Ajv({ allErrors: true, strict: false }).compile(JSON.parse(bytes));

export function assertSpdx23(document) {
  assert(validate(document), JSON.stringify(validate.errors));
  assert.equal(document.spdxVersion, "SPDX-2.3");
  assert.equal(document.SPDXID, "SPDXRef-DOCUMENT");
  assert.equal(document.dataLicense, "CC0-1.0");
  assert.match(document.creationInfo.created, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
  const ids = new Set(document.packages.map((item) => item.SPDXID));
  assert.equal(ids.size, document.packages.length, "SPDX identities must be unique");
  for (const id of document.documentDescribes) assert(ids.has(id));
  for (const relation of document.relationships) {
    assert(ids.has(relation.spdxElementId));
    assert(ids.has(relation.relatedSpdxElement));
  }
}
