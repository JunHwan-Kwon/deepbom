import { GENERATED_FINDING_RULES } from "./finding-rule-catalog.generated.js";

const FINDING_RULES = GENERATED_FINDING_RULES;

export function findFindingRule(findingId) {
  return FINDING_RULES[String(findingId || "").trim().toUpperCase()] || null;
}

export function explainFindingRule(findingId) {
  const id = String(findingId || "").trim().toUpperCase();
  const rule = findFindingRule(id);
  if (!rule) throw new Error(`Unknown finding rule: ${findingId}. Use deepbom explain-rule --list.`);
  return {
    schema: "deepbom.finding_rule_explanation.v1",
    rule_kind: "finding_rule",
    rule_id: id,
    ...rule,
  };
}

export function listFindingRuleExplanations() {
  return Object.entries(FINDING_RULES).map(([rule_id, rule]) => ({
    rule_kind: "finding_rule",
    rule_id,
    title: rule.title,
  }));
}
