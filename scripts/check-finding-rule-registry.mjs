import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";

import { parse } from "acorn";

import { explainFindingRule, listFindingRuleExplanations } from "../web/lib/finding-rule-catalog.js";
import { buildFindingEvidenceExplanation } from "../web/lib/findings-viewer.js";
import { listAnalysisRuleExplanations } from "../web/lib/rule-explanations.js";

const policy = JSON.parse(await readFile("config/finding-rule-registry-policy.v1.json", "utf8"));
assert.equal(policy.schema, "deepbom.finding_rule_registry_policy.v1");

const canonical = await canonicalFindingIds(policy.canonical_report_findings.producer);
assert.equal(canonical.length, policy.canonical_report_findings.count, "Canonical report finding count changed; classify and register the change explicitly.");
assert.equal(digest(canonical), policy.canonical_report_findings.sorted_ids_sha256, "Canonical report finding inventory changed; new IDs must not enter as unexplained debt.");

const auxiliary = sortedUnique(policy.auxiliary_ui_findings);
const retired = sortedUnique(policy.retired_ids);
assert.equal(auxiliary.length, policy.auxiliary_ui_findings.length, "Auxiliary finding policy contains duplicates.");
assert.equal(retired.length, policy.retired_ids.length, "Retired finding policy contains duplicates.");

const productionLiterals = await productionEaLiterals();
const classified = new Set([...canonical, ...auxiliary, ...retired]);
assert.deepEqual(productionLiterals.filter((id) => !classified.has(id)), [], "Production source contains unclassified EA identifiers.");
for (const id of [...auxiliary, ...retired]) assert(productionLiterals.includes(id), `${id} is classified but absent from production source.`);

const analysisRules = listAnalysisRuleExplanations().map((row) => row.rule_id).sort();
assert.deepEqual(analysisRules, [...policy.analysis_rule_ids].sort(), "Analysis-rule namespace changed without a policy update.");

const catalog = listFindingRuleExplanations();
const catalogIds = catalog.map((row) => row.rule_id).sort();
assert.equal(new Set(catalogIds).size, catalogIds.length, "Finding-rule catalog contains duplicate IDs.");
assert.deepEqual(catalogIds.filter((id) => !canonical.includes(id)), [], "Finding-rule catalog contains a non-canonical report ID.");
const unexplained = canonical.filter((id) => !catalogIds.includes(id));
assert.equal(policy.legacy_unexplained.count, 0);
assert.equal(policy.legacy_unexplained.sorted_ids_sha256, null);
assert.deepEqual(unexplained, [], "Every canonical finding ID must have a generated explanation before it can be emitted.");
for (const row of catalog) {
  assert.equal(row.rule_kind, "finding_rule");
  assert.match(row.rule_id, /^EA-[A-Z][A-Z0-9_]{1,15}-\d{4}$/);
  assert.equal(typeof row.title, "string");
  assert(row.title.length >= 8, `${row.rule_id} title is too short.`);
  const explanation = explainFindingRule(row.rule_id);
  assert.equal(explanation.finding_id, row.rule_id);
  assert(Array.isArray(explanation.applicable_formats) && explanation.applicable_formats.length > 0);
  assert(["artifact_defect", "caution", "evidence_gap", "conditional"].includes(explanation.finding_kind));
  assert(Array.isArray(explanation.possible_finding_kinds) && explanation.possible_finding_kinds.every((value) => ["artifact_defect", "caution", "evidence_gap"].includes(value)));
  assert(Array.isArray(explanation.evidence_requirements) && explanation.evidence_requirements.every((value) => value.startsWith("/")));
  for (const key of ["trigger_contract", "false_positive_boundary", "remediation", "method_reference"]) assert.match(explanation[key], /\S/, `${row.rule_id} ${key} is empty.`);
  assert.equal(explanation.source_reference.path, policy.canonical_report_findings.producer);
  const lines = explanation.source_reference.lines || [explanation.source_reference.line];
  assert(lines.every((line) => Number.isSafeInteger(line) && line > 0), `${row.rule_id} source lines are invalid.`);
}

const generated = execFileSync(process.execPath, ["scripts/generate-finding-rule-catalog.mjs", "--check"], { encoding: "utf8" });
assert.match(generated, /119 rules/);
const webExplanation = await buildFindingEvidenceExplanation({
  id: "EA-CML-0001",
  title: "Core ML serialized constants contain non-finite values",
  severity: "high",
  category: "integrity",
  evidence_class: "OBSERVED",
  evidence: [{ source: "coreml.constants", text: "non-finite value" }],
});
assert.equal(webExplanation.evidence_class, "OBSERVED");
assert(webExplanation.source_pointers.includes("/evidence/static_analysis/weight_integrity/parameters"));
assert.match(webExplanation.method, /deepbom\.report_findings\.EA-CML-0001\.v1/);
assert.match(webExplanation.limitations.join(" "), /does not establish task, clinical, or runtime impact/);
console.log(`Finding-rule boundary passed: ${canonical.length} canonical report IDs, ${auxiliary.length} auxiliary, ${retired.length} retired, ${catalogIds.length} explained, zero unexplained debt.`);

async function canonicalFindingIds(file) {
  const source = await readFile(file, "utf8");
  const tree = parse(source, { ecmaVersion: "latest", sourceType: "module" });
  const ids = new Set();
  walk(tree, (node) => {
    if (node.type !== "CallExpression" || node.callee?.type !== "Identifier" || node.callee.name !== "finding") return;
    collectFindingArgument(node.arguments[0], ids);
  });
  return sortedUnique(ids);
}

function collectFindingArgument(node, ids) {
  if (node?.type === "ObjectExpression") {
    const property = node.properties.find((item) => item.type === "Property" && !item.computed && (item.key.name || item.key.value) === "id");
    const id = staticString(property?.value);
    assert.match(id || "", /^EA-[A-Z][A-Z0-9_]{1,15}-\d{4}$/, "Every finding() branch must bind one static canonical EA identifier.");
    ids.add(id);
    return;
  }
  if (node?.type === "ConditionalExpression") {
    collectFindingArgument(node.consequent, ids);
    collectFindingArgument(node.alternate, ids);
    return;
  }
  assert.fail("Every finding() call must use an object or a conditional whose branches are objects.");
}

async function productionEaLiterals() {
  const files = execFileSync("git", ["ls-files", "web", "src", "bin"], { encoding: "utf8" })
    .split(/\r?\n/).filter((file) => /\.(?:js|mjs|rs)$/.test(file));
  const ids = new Set();
  for (const file of files) {
    const source = await readFile(file, "utf8");
    for (const match of source.matchAll(/EA-[A-Z][A-Z0-9_]{1,15}-\d{4}/g)) ids.add(match[0]);
  }
  return sortedUnique(ids);
}

function walk(node, visit) {
  if (!node || typeof node !== "object") return;
  visit(node);
  for (const [key, value] of Object.entries(node)) {
    if (["start", "end", "loc"].includes(key)) continue;
    if (Array.isArray(value)) value.forEach((item) => walk(item, visit));
    else if (value?.type) walk(value, visit);
  }
}

function staticString(node) {
  if (node?.type === "Literal" && typeof node.value === "string") return node.value;
  if (node?.type === "TemplateLiteral" && node.expressions.length === 0) return node.quasis[0].value.cooked;
  return null;
}

function sortedUnique(values) {
  return [...new Set(values)].sort();
}

function digest(ids) {
  return createHash("sha256").update(`${ids.join("\n")}\n`).digest("hex");
}
