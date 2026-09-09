import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import {
  buildCycloneDxComponentInventory,
  validateCycloneDxComponentInventory,
} from "../web/lib/cyclonedx-component-inventory.js";

const record = JSON.parse(await readFile("corpus/external-review/cyclonedx-17-consumer-compatibility.v1.json", "utf8"));
assert.equal(record.schema, "deepbom.external_review_cyclonedx_consumer_compatibility.v1");
const sourcePath = path.resolve(record.source.path);
const bytes = await readFile(sourcePath);
assert.equal(createHash("sha256").update(bytes).digest("hex"), record.source.sha256);
const bom = JSON.parse(bytes.toString("utf8"));
assertCycloneDx17(bom, record.source.path);
assert.equal(bom.specVersion, record.expected.spec_version);
assert.equal(bom.metadata.component["bom-ref"], record.expected.root_bom_ref);
assert.equal(bom.metadata.component.type, record.expected.root_type);
assert.equal(bom.metadata.component.properties.length, record.expected.root_property_count);
assert.equal((bom.components || []).length, record.expected.components_array_count);

const inventory = validateCycloneDxComponentInventory(buildCycloneDxComponentInventory(bom));
assert.equal(inventory.visible_component_count, record.expected.schema_aware_visible_component_count);
assert.equal(inventory.components[0].pointer, "metadata.component");
assert.equal(inventory.components[0].root, true);
assert.equal((bom.components || []).length, record.expected.components_only_visible_component_count,
  "The negative control must reproduce omission by a components-only consumer.");

const withMember = structuredClone(bom);
withMember.components = [{ type: "library", name: "fixture-member", version: "1", "bom-ref": "fixture-member@1" }];
const two = validateCycloneDxComponentInventory(buildCycloneDxComponentInventory(withMember));
assert.equal(two.visible_component_count, 2);
assert.deepEqual(two.components.map((row) => row.pointer), ["metadata.component", "components[0]"]);
const duplicate = structuredClone(withMember);
duplicate.components[0]["bom-ref"] = duplicate.metadata.component["bom-ref"];
assert.throws(() => buildCycloneDxComponentInventory(duplicate), /bom-ref .* is duplicated/);
const tampered = structuredClone(inventory);
tampered.visible_component_count = 0;
assert.throws(() => validateCycloneDxComponentInventory(tampered), /count is invalid/);

console.log("CycloneDX 1.7 root-component consumer compatibility verified: metadata.component is visible without duplicating it in components[].");
