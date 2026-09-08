import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const ledger = JSON.parse(await readFile("corpus/external-review/external-review-cases.v1.json", "utf8"));
const triage = await readFile("docs/EXTERNAL_REVIEW_TRIAGE.md", "utf8");

assert.equal(ledger.schema, "deepbom.external_review_cases.v1");
assert.match(ledger.baseline.source_commit, /^[a-f0-9]{7,40}$/);
assert.match(ledger.baseline.published_version, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/);
assert.deepEqual(ledger.status_lifecycle, ["reported", "reproduced", "fixed", "verified", "deferred"]);
assert(Array.isArray(ledger.cases) && ledger.cases.length >= 20);

const ids = new Set();
const allowedClasses = new Set([
  "correctness", "export_contract", "policy_design", "documentation", "semantic_label",
  "format_coverage", "evidence_gap", "product_contract", "summary_contract",
  "cli_experience", "feature_gap", "feature_scope", "explainability",
  "interoperability_risk", "test_coverage", "compatibility_policy", "release_policy",
]);
for (const row of ledger.cases) {
  assert.match(row.id, /^ER-[A-Z]\d+$/, `Invalid external-review id: ${row.id}`);
  assert.equal(ids.has(row.id), false, `Duplicate external-review id: ${row.id}`);
  ids.add(row.id);
  assert.equal(typeof row.title, "string");
  assert(row.title.length >= 12, `${row.id} title is too short.`);
  assert(allowedClasses.has(row.class), `${row.id} has an unknown class.`);
  assert(["P0", "P1", "P2"].includes(row.priority), `${row.id} has an invalid priority.`);
  assert(ledger.status_lifecycle.includes(row.status), `${row.id} has an invalid status.`);
  assert(Array.isArray(row.formats) && row.formats.length > 0, `${row.id} must declare formats.`);
  assert(triage.includes(`### ${row.id}.`), `${row.id} is missing from EXTERNAL_REVIEW_TRIAGE.md.`);
  if (["fixed", "verified"].includes(row.status)) {
    assert.equal(typeof row.reproducer, "string", `${row.id} must bind a reproducer before ${row.status}.`);
  }
}

const documentedIds = [...triage.matchAll(/^### (ER-[A-Z]\d+)\./gm)].map((match) => match[1]);
assert.deepEqual(new Set(documentedIds), ids, "The Markdown and JSON external-review case sets differ.");
console.log(`External-review ledger verified: ${ledger.cases.length} unique cases, Markdown parity, lifecycle and evidence bindings valid.`);
