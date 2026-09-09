export const GATE_POLICY_PROFILE_SCHEMA = "deepbom.gate_policy_profile.v1";
export const GATE_POLICY_RESULT_SCHEMA = "deepbom.gate_policy_profile_result.v1";

const PROFILES = Object.freeze({
  engineering: Object.freeze({
    id: "engineering",
    label: "Engineering artifact gate",
    blocking_finding_kinds: Object.freeze(["artifact_defect"]),
    purpose: "Block defects observed in the deployment artifact while retaining cautions and external evidence gaps for review.",
  }),
  regulatory: Object.freeze({
    id: "regulatory",
    label: "Regulatory evidence-completeness gate",
    blocking_finding_kinds: Object.freeze(["artifact_defect", "evidence_gap"]),
    purpose: "Block observed artifact defects and unresolved evidence gaps before a controlled release review.",
  }),
});

export function listGatePolicyProfiles() {
  return Object.values(PROFILES).map((profile) => ({
    schema: GATE_POLICY_PROFILE_SCHEMA,
    id: profile.id,
    label: profile.label,
    blocking_finding_kinds: [...profile.blocking_finding_kinds],
    purpose: profile.purpose,
    boundary: policyBoundary(profile.id),
  }));
}

export function evaluateGatePolicyProfile(envelope, profileId) {
  const id = String(profileId || "").trim().toLowerCase();
  const profile = PROFILES[id];
  if (!profile) throw new Error("--policy must be engineering or regulatory.");
  const findings = Array.isArray(envelope?.findings) ? envelope.findings : [];
  const counts = { artifact_defect: 0, caution: 0, evidence_gap: 0 };
  for (const finding of findings) counts[finding.finding_kind] = (counts[finding.finding_kind] || 0) + 1;
  const blocking = findings.filter((finding) => profile.blocking_finding_kinds.includes(finding.finding_kind));
  return {
    schema: GATE_POLICY_RESULT_SCHEMA,
    status: blocking.length ? "block" : "pass",
    profile: id,
    profile_schema: GATE_POLICY_PROFILE_SCHEMA,
    blocking_finding_kinds: [...profile.blocking_finding_kinds],
    finding_count: findings.length,
    finding_kind_counts: counts,
    blocking_finding_count: blocking.length,
    blocking_finding_ids: blocking.map((finding) => finding.id),
    evidence_envelope_sha256: envelope?.envelope_sha256 || null,
    interpretation_boundary: policyBoundary(id),
  };
}

function policyBoundary(id) {
  return id === "regulatory"
    ? "This profile checks evidence completeness for review workflow entry. It does not determine legal compliance, safety, clinical performance, or regulatory acceptance."
    : "This profile checks artifact defects only. Passing does not resolve cautions, external evidence gaps, runtime behavior, model quality, or task safety.";
}
