export const AGENT_CONTRACT = deepFreeze({
  id: "deepbom.agent_contract.v1",
  version: "1.0.1",
  compatibility: "backward_compatible_within_v1",
});

export const EVIDENCE_CONTRACT = deepFreeze({
  id: "deepbom.artifact_evidence_envelope.v1",
  version: "1.0.0",
  compatibility: "additive_fields_only_within_v1",
  required_top_level_fields: [
    "schema",
    "generated_at",
    "identity",
    "artifact_set",
    "conversion_receipt",
    "cpu_cost_target_binding",
    "accelerator_profile_binding",
    "accelerator_bindings",
    "policy_identity",
    "capabilities",
    "interfaces",
    "graph",
    "external_files",
    "metadata",
    "findings",
    "format_extensions",
    "provenance",
    "evidence_boundary",
    "envelope_sha256",
  ],
  conditional_top_level_fields: ["llm_token_budget_scenario", "structured_details"],
  semantic_boundaries: [
    "artifact_defects_cautions_and_evidence_gaps_remain_distinct",
    "static_predictions_are_not_runtime_observations",
    "unknown_and_not_assessable_are_not_coerced_to_absence_or_zero",
  ],
});

export const AGENT_PLUGIN_VERSION = AGENT_CONTRACT.version;

export function cloneContractIdentity(contract) {
  return JSON.parse(JSON.stringify(contract));
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
