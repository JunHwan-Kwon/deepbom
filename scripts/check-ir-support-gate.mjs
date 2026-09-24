import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { ARTIFACT_IR_METHOD_VERSION } from "../web/lib/artifact-ir.js";
import { MODEL_IR_METHOD_VERSION } from "../web/lib/model-ir.js";

const raw = readFileSync("config/model-ir-upstream-sources.v1.json"), sources = JSON.parse(raw);
const gate = JSON.parse(readFileSync("config/ir-support-gate.v1.json", "utf8"));
assert.equal(gate.schema, "deepbom.ir_support_gate.v1");
assert.equal(gate.reviewed_source_manifest_sha256, createHash("sha256").update(raw).digest("hex"),
  "Upstream sources changed: review the schema diff, adapter impact, native fixtures and support claims before updating the reviewed digest.");
assert.equal(gate.artifact_ir_method, ARTIFACT_IR_METHOD_VERSION); assert.equal(gate.model_ir_method, MODEL_IR_METHOD_VERSION);
assert(gate.reviewed_change.length > 40, "A concrete change assessment is required.");
const allChecks = readFileSync("scripts/check-all.mjs", "utf8");
for (const check of gate.required_checks) { assert(existsSync(check), `Missing required fixture runner ${check}`); assert(allChecks.includes(`"${check}"`), `Required runner omitted from full checks: ${check}`); }
assert.equal(new Set(gate.support.map(row => row.source_id)).size, sources.sources.length);
for (const source of sources.sources) {
  const row = gate.support.find(row => row.source_id === source.id);
  assert(row, `Missing source coverage ${source.id}`); assert.equal(row.reviewed_commit, source.commit);
  assert.equal(row.source_verifier, source.verifier); assert(existsSync(row.source_verifier));
  assert.equal(row.upstream_head_implies_support, false); assert.equal(row.support_claim, "bounded_adapter_projection");
  assert.equal(row.semantic_coverage, "operator_and_format_specific_tests_required");
}
console.log(`IR support gate passed (${gate.support.length} reviewed source roles; ${gate.required_checks.length} mandatory regression runners).`);
