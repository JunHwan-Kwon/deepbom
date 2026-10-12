import assert from "node:assert/strict";
import { collectAnalysisFieldPatterns } from "../web/lib/metric-coverage.js";

// Every array element must contribute its field kinds, including a late unknown
// or non-finite value. Sharing an object must not hide a different pointer.
const shared = { value: 0 };
const source = { rows: [shared, shared, { value: null }, { value: NaN, extra: "late" }], another: shared };
const fields = new Map(collectAnalysisFieldPatterns(source).fields.map(row => [row.field_path, row.value_kinds]));
assert.deepEqual(fields.get("/rows/[]/value"), ["non_finite_number", "null", "number"]);
assert.deepEqual(fields.get("/rows/[]/extra"), ["string"]);
assert.deepEqual(fields.get("/another/value"), ["number"]);

// Composition does not spend one document's budget on another complete IR.
// Inspection still visits every scalar; unknown schema tags gain no allowance.
const payload = Array(1_000_001).fill(0);
const composite = { artifact_ir: { schema: "deepbom.artifact_ir.v2", payload }, model_ir: { schema: "deepbom.model_ir.v1", payload } };
const result = collectAnalysisFieldPatterns(composite);
assert.equal(result.status, "complete");
assert.equal(result.visit_count, 2_000_009);
assert.equal(result.visit_limit_per_document, 2_000_000);
assert.equal(result.visit_limit, 6_000_000);
assert.equal(result.document_visits["/artifact_ir"], 1_000_004);
assert.equal(collectAnalysisFieldPatterns({ unknown: composite }).status, "incomplete_visit_limit", "Nested arbitrary objects do not allocate document budgets");
assert.equal(collectAnalysisFieldPatterns({ artifact_ir: { schema: "unknown", payload }, model_ir: { schema: "unknown", payload } }).status, "incomplete_visit_limit");
assert.equal(collectAnalysisFieldPatterns({ model_ir: { schema: "deepbom.model_ir.v1", payload: Array(2_000_001).fill(0) } }).status, "incomplete_visit_limit", "One oversized document still fails closed");
console.log("Metric field discovery: every field kind retained; composed IR budgets isolated; unknown schemas and oversized documents remain bounded.");
