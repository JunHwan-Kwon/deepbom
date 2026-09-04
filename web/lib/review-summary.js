import { auditTabApplicability } from "./evidence-applicability.js";

export const REVIEW_SUMMARY_SCHEMA = "deepbom.review_summary.v1";
const FINDING_KINDS = Object.freeze(["artifact_defect", "caution", "evidence_gap"]);

export function buildReviewSummary({ analysis = {}, envelope, artifactIrContext = null } = {}) {
  if (envelope?.schema !== "deepbom.artifact_evidence_envelope.v1") {
    throw new Error("Review summary requires a canonical artifact evidence envelope.");
  }
  const findings = Array.isArray(envelope.findings) ? envelope.findings : [];
  const grouped = Object.fromEntries(FINDING_KINDS.map((kind) => [kind, findings.filter((row) => row.finding_kind === kind)]));
  const applicability = auditTabApplicability(envelope.identity?.format, analysis);
  const applicabilityCounts = { applicable: 0, not_applicable: 0, not_assessable: 0, not_assessed_yet: 0 };
  for (const row of Object.values(applicability)) {
    applicabilityCounts[row.applicability_status] = (applicabilityCounts[row.applicability_status] || 0) + 1;
  }
  const capabilities = envelope.capabilities || {};
  const result = {
    schema: REVIEW_SUMMARY_SCHEMA,
    artifact: {
      filename: envelope.identity?.filename || null,
      format: envelope.identity?.format || null,
      sha256: envelope.identity?.sha256 || null,
      byte_length: envelope.identity?.byte_length ?? null,
      artifact_ir_sha256: artifactIrContext?.artifact_ir?.artifact_ir_sha256 || analysis?.artifact_ir?.artifact_ir_sha256 || null,
    },
    verdict: {
      artifact_defect_count: grouped.artifact_defect.length,
      caution_count: grouped.caution.length,
      evidence_needed_count: grouped.evidence_gap.length,
      status: grouped.artifact_defect.length ? "artifact_defects_observed" : "no_artifact_defect_observed",
      scope: "This verdict covers only the applicable static checks and separately identifies evidence that the artifact cannot supply.",
    },
    findings: {
      artifact_defects: compactFindings(grouped.artifact_defect),
      cautions: compactFindings(grouped.caution),
      evidence_needed: compactFindings(grouped.evidence_gap),
    },
    graph: {
      operator_count: envelope.graph?.operator_count ?? null,
      tensor_count: envelope.graph?.tensor_count ?? null,
      total_macs: envelope.graph?.total_macs ?? null,
      mac_assessment_status: envelope.graph?.mac_assessment_status || null,
    },
    coverage: {
      assessed: Array.isArray(capabilities.assessed) ? capabilities.assessed.length : 0,
      partial: Array.isArray(capabilities.partial) ? capabilities.partial.length : 0,
      needs_external_evidence: Array.isArray(capabilities.unavailable) ? capabilities.unavailable.length : 0,
      declared: Number(capabilities?.conservation?.declared || 0),
      applicability: applicabilityCounts,
    },
    target: {
      id: analysis?.target_profile?.id || analysis?.cpu_cost_target_binding?.profile_id || null,
      label: analysis?.target_profile?.label || null,
      binding_source: analysis?.cpu_cost_target_binding?.binding_source || analysis?.target_binding_source || null,
    },
    rulepack: {
      version: analysis?.rulepack_version || analysis?.target_profile?.source_rulepack_version || null,
      sha256: analysis?.rulepack_sha256 || analysis?.target_profile?.source_rulepack_sha256 || null,
    },
    applicability,
    next_actions: nextActions(grouped),
    evidence_envelope_sha256: envelope.envelope_sha256,
  };
  validateReviewSummary(result);
  return result;
}

export function validateReviewSummary(summary) {
  const errors = [];
  if (summary?.schema !== REVIEW_SUMMARY_SCHEMA) errors.push("schema_mismatch");
  if (!summary?.artifact?.sha256) errors.push("artifact_sha256_missing");
  if (!summary?.artifact?.artifact_ir_sha256) errors.push("artifact_ir_sha256_missing");
  const count = (key) => Number(summary?.verdict?.[key]);
  if (count("artifact_defect_count") !== (summary?.findings?.artifact_defects || []).length) errors.push("artifact_defect_count_mismatch");
  if (count("caution_count") !== (summary?.findings?.cautions || []).length) errors.push("caution_count_mismatch");
  if (count("evidence_needed_count") !== (summary?.findings?.evidence_needed || []).length) errors.push("evidence_needed_count_mismatch");
  if (summary?.graph?.total_macs != null && !Number.isFinite(Number(summary.graph.total_macs))) errors.push("graph_total_macs_invalid");
  if (errors.length) throw new Error(`Invalid review summary: ${errors.join(", ")}`);
  return { valid: true, errors: [] };
}

function compactFindings(findings) {
  return findings.map((row) => ({
    id: row.id,
    title: row.title,
    severity: row.severity,
    evidence_class: row.evidence_class,
    source_pointers: row.source_pointers || [],
    recommendation: row.recommendation || null,
  }));
}

function nextActions(grouped) {
  const actions = [];
  if (grouped.artifact_defect.length) actions.push({ action: "review_artifact_defects", command_hint: "--section findings" });
  if (grouped.caution.length) actions.push({ action: "review_cautions", command_hint: "--section findings" });
  if (grouped.evidence_gap.length) actions.push({ action: "supply_external_evidence", command_hint: "Use the relevant runtime, lineage, or build-evidence import option." });
  actions.push({ action: "open_complete_evidence", command_hint: "--output-format json" });
  return actions;
}
