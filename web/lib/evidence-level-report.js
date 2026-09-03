import { artifactIrValues } from "./artifact-ir-selectors.js";
export const EVIDENCE_LEVEL_PROFILE_SCHEMA = "deepbom.evidence_level_profile.v1";

export const EVIDENCE_LEVEL_PROFILES = Object.freeze([
  Object.freeze({
    id: "artifact_facts",
    label: "Artifact facts",
    rank: 1,
    detail: "Serialized/session observations plus explicit unavailable boundaries",
  }),
  Object.freeze({
    id: "deterministic",
    label: "Deterministic",
    rank: 2,
    detail: "Artifact facts plus source-pinned and deterministic derivations",
  }),
  Object.freeze({
    id: "planning",
    label: "Planning",
    rank: 3,
    detail: "Deterministic evidence plus predicted, estimated, heuristic, and declared planning evidence",
  }),
  Object.freeze({
    id: "all_available",
    label: "All available",
    rank: 4,
    detail: "Every available class, including imported runtime and measured evidence",
  }),
]);

const PROFILE_BY_ID = new Map(EVIDENCE_LEVEL_PROFILES.map((profile) => [profile.id, profile]));
const GAP_PATTERN = /NOT[_ -]?(?:ASSESSED|ASSESSABLE|APPLICABLE|OBSERVED)|UNAVAILABLE|SUPPRESSED|OUT_OF_SCOPE/i;
const RUNTIME_PATTERN = /MEASURED|OBSERVED_RUNTIME|BOUND_RUNTIME|RUNTIME_CAPTURE|EXECUTED|PROFILE_EVENTS/i;
const PLANNING_PATTERN = /PREDICTED|ESTIMATED|HEURISTIC|PROXY|COUNTERFACTUAL|DECLARED|CONFIGURATION|PREFLIGHT|REQUIREMENT|ANTICIPATED|COMPUTE_PLAN/i;
const DETERMINISTIC_PATTERN = /DERIVED|SOURCE(?:_|-)?(?:PINNED|BACKED|MATCH)|IMPLEMENTATION_TESTED|NECESSARY_MINIMUM|STATIC_RESOURCE|CONTRACT_BOUND/i;
const OBSERVED_PATTERN = /OBSERVED/i;

export function evidenceLevelProfile(profileId) {
  return PROFILE_BY_ID.get(String(profileId || "")) || PROFILE_BY_ID.get("all_available");
}

export function classifyEvidenceClass(evidenceClass) {
  const value = String(evidenceClass || "").trim();
  if (!value) return { rank: 5, category: "unclassified", status: "unclassified" };
  if (GAP_PATTERN.test(value)) return { rank: 0, category: "boundary", status: "classified" };
  if (RUNTIME_PATTERN.test(value)) return { rank: 4, category: "runtime", status: "classified" };
  if (PLANNING_PATTERN.test(value)) return { rank: 3, category: "planning", status: "classified" };
  if (DETERMINISTIC_PATTERN.test(value)) return { rank: 2, category: "deterministic", status: "classified" };
  if (OBSERVED_PATTERN.test(value)) return { rank: 1, category: "artifact", status: "classified" };
  return { rank: 5, category: "unclassified", status: "unclassified" };
}

export function buildEvidenceLevelManifest({
  levelId = "all_available",
  analysis,
  metricCoverage,
  findings = null,
  runtimeEvidence = null,
  generatedAt,
} = {}) {
  if (!analysis || typeof analysis !== "object") throw new Error("Evidence-level export requires completed analysis.");
  if (!generatedAt) throw new Error("Evidence-level export requires a frozen generatedAt timestamp.");
  if (!Array.isArray(metricCoverage?.entries)) throw new Error("Evidence-level export requires the metric-coverage ledger.");
  const profile = evidenceLevelProfile(levelId);
  const applicable = metricCoverage.entries.filter((entry) => !["not_applicable", "suppressed"].includes(String(entry?.status || "")));
  const included = [];
  const excluded = [];
  const boundaries = [];
  const unclassified = [];
  for (const entry of applicable) {
    const evidenceClass = String(entry?.evidence_class || "");
    const classification = classifyEvidenceClass(evidenceClass);
    const row = {
      metric_id: String(entry?.metric_id || ""),
      label: String(entry?.label || entry?.metric_id || "unnamed metric"),
      status: String(entry?.status || "not_assessed"),
      evidence_class: evidenceClass || "UNCLASSIFIED",
      evidence_category: classification.category,
      required_level_rank: classification.rank,
      report_section: entry?.report_section || null,
    };
    if (!row.metric_id) throw new Error("Evidence-level export encountered a metric without an ID.");
    if (classification.rank === 0 || row.status === "not_assessed") boundaries.push(row);
    else if (classification.status !== "classified") {
      unclassified.push(row);
      excluded.push({ ...row, exclusion_reason: "unclassified_evidence_class_fail_closed" });
    } else if (classification.rank <= profile.rank) included.push(row);
    else excluded.push({ ...row, exclusion_reason: `requires_level_rank_${classification.rank}` });
  }
  const findingRows = Array.isArray(findings)
    ? findings
    : Array.isArray(analysis.findings) ? analysis.findings : [];
  const includedFindings = [];
  const excludedFindings = [];
  for (const finding of findingRows) {
    const evidenceClass = String(finding?.evidence_class || "").trim();
    const classification = classifyEvidenceClass(evidenceClass);
    const row = {
      finding_id: String(finding?.id || finding?.finding_id || "unidentified_finding"),
      title: String(finding?.title || "Untitled finding"),
      priority: String(finding?.priority || finding?.severity || "Informational"),
      evidence_class: evidenceClass || "UNCLASSIFIED",
      evidence_category: classification.category,
      required_level_rank: classification.rank,
      observation: String(finding?.observation || ""),
      actions: Array.isArray(finding?.actions) ? finding.actions.map(String) : [],
    };
    if (classification.rank === 0 || classification.rank <= profile.rank) includedFindings.push(row);
    else excludedFindings.push({ ...row, exclusion_reason: classification.status === "classified"
      ? `requires_level_rank_${classification.rank}`
      : "unclassified_evidence_class_fail_closed" });
  }
  const facts = buildScopedFacts(analysis, runtimeEvidence).filter((fact) => {
    const classification = classifyEvidenceClass(fact.evidence_class);
    return classification.rank === 0 || classification.rank <= profile.rank;
  });
  return {
    schema: EVIDENCE_LEVEL_PROFILE_SCHEMA,
    generated_at: generatedAt,
    requested_level: profile.id,
    requested_level_label: profile.label,
    requested_level_rank: profile.rank,
    cumulative_selection: true,
    selection_rule: "Include classified evidence whose required level rank is less than or equal to the requested rank. Preserve explicit not-assessed boundaries without treating them as claims. Exclude unclassified evidence fail-closed.",
    artifact: {
      filename: String(analysis.filename || "model"),
      format: String(analysis.format || "unknown"),
      sha256: String(analysis.model_sha256 || ""),
      byte_length: finiteInteger(analysis.file_size_bytes ?? analysis.file_size),
    },
    metric_coverage_schema: metricCoverage.schema || null,
    metric_counts: {
      applicable: applicable.length,
      included: included.length,
      boundary: boundaries.length,
      excluded: excluded.length,
      unclassified: unclassified.length,
    },
    included_metrics: included,
    boundary_metrics: boundaries,
    excluded_metrics: excluded,
    included_findings: includedFindings,
    excluded_findings: excludedFindings.map((row) => ({
      finding_id: row.finding_id,
      evidence_class: row.evidence_class,
      required_level_rank: row.required_level_rank,
      exclusion_reason: row.exclusion_reason,
    })),
    excluded_finding_ids: excludedFindings.map((row) => row.finding_id),
    summary_facts: facts,
    full_report_included: profile.id === "all_available",
    unscoped_cyclonedx_included: profile.id === "all_available",
    interpretation_boundary: "This manifest controls report/package membership, not the underlying audit. Excluded classes remain available in the local full audit. A lower level never promotes absence to zero and never converts static eligibility into runtime assignment.",
  };
}

export function buildEvidenceLevelReport(manifest, { scope = null } = {}) {
  if (!validateEvidenceLevelManifest(manifest)) throw new Error("Evidence-level manifest is invalid.");
  const facts = manifest.summary_facts.length
    ? manifest.summary_facts.map((row) => `| ${cell(row.label)} | ${cell(row.value)} | ${cell(row.evidence_class)} |`)
    : ["| No selected summary fact | - | - |"];
  const findings = manifest.included_findings.length
    ? manifest.included_findings.map((row) => `| ${cell(row.finding_id)} | ${cell(row.priority)} | ${cell(row.evidence_class)} | ${cell(row.title)} |`)
    : ["| - | - | - | No finding belongs to the selected evidence level. |"];
  const metrics = manifest.included_metrics.length
    ? manifest.included_metrics.map((row) => `| ${cell(row.metric_id)} | ${cell(row.status)} | ${cell(row.evidence_class)} | ${cell(row.label)} |`)
    : ["| - | - | - | No assessed metric belongs to the selected evidence level. |"];
  const boundaries = manifest.boundary_metrics.length
    ? manifest.boundary_metrics.map((row) => `| ${cell(row.metric_id)} | ${cell(row.status)} | ${cell(row.label)} |`)
    : ["| - | - | No explicit evidence gap was emitted. |"];
  return [
    "# DEEPBOM Scoped Evidence Report",
    "",
    `**Artifact:** \`${manifest.artifact.filename}\``,
    `**Artifact SHA-256:** \`${manifest.artifact.sha256 || "not embedded"}\``,
    `**Evidence level:** ${manifest.requested_level_label} (rank ${manifest.requested_level_rank})`,
    `**Generated:** ${manifest.generated_at}`,
    "",
    "## Selection Contract",
    manifest.selection_rule,
    "",
    `Included metric families: ${manifest.metric_counts.included}; explicit boundaries: ${manifest.metric_counts.boundary}; excluded higher/unclassified families: ${manifest.metric_counts.excluded}.`,
    "",
    "## Selected Summary Values",
    "| Metric | Value | Evidence |",
    "| --- | --- | --- |",
    ...facts,
    "",
    "## Findings In Scope",
    "| ID | Priority | Evidence | Finding |",
    "| --- | --- | --- | --- |",
    ...findings,
    "",
    "## Metric Families In Scope",
    "| Metric ID | Status | Evidence | Scope |",
    "| --- | --- | --- | --- |",
    ...metrics,
    "",
    "## Explicit Evidence Boundaries",
    "| Metric ID | Status | Boundary |",
    "| --- | --- | --- |",
    ...boundaries,
    "",
    "## Excluded From This Report",
    `- Higher or unclassified metric families: ${manifest.excluded_metrics.map((row) => row.metric_id).join(", ") || "none"}`,
    `- Findings above the selected level: ${manifest.excluded_finding_ids.join(", ") || "none"}`,
    manifest.unscoped_cyclonedx_included
      ? "- Full standalone CycloneDX documents are included because All available was selected."
      : "- Standalone CycloneDX documents are omitted because they are not field-filtered by this scoped report.",
    "",
    "## Claim Boundary",
    `- ${scope?.runtimeBoundary || "Actual runtime assignment, copies, kernels, and device latency require bound runtime evidence."}`,
    `- ${scope?.releaseStatus || "Release readiness is not established by artifact evidence alone."}`,
    `- ${manifest.interpretation_boundary}`,
    "",
  ].join("\n");
}

export function validateEvidenceLevelManifest(manifest) {
  const profile = evidenceLevelProfile(manifest?.requested_level);
  if (manifest?.schema !== EVIDENCE_LEVEL_PROFILE_SCHEMA || profile.id !== manifest.requested_level) return false;
  if (!Array.isArray(manifest.included_metrics) || !Array.isArray(manifest.boundary_metrics)
    || !Array.isArray(manifest.excluded_metrics) || !Array.isArray(manifest.included_findings)
    || !Array.isArray(manifest.excluded_findings) || !Array.isArray(manifest.excluded_finding_ids)) return false;
  if (manifest.included_metrics.some((row) => classifyEvidenceClass(row.evidence_class).rank > profile.rank)) return false;
  if (manifest.boundary_metrics.some((row) => classifyEvidenceClass(row.evidence_class).rank !== 0 && row.status !== "not_assessed")) return false;
  if (manifest.included_findings.some((row) => classifyEvidenceClass(row.evidence_class).rank > profile.rank)) return false;
  if (manifest.excluded_metrics.some((row) => !row.exclusion_reason)) return false;
  if (manifest.excluded_findings.some((row) => !row.exclusion_reason
    || classifyEvidenceClass(row.evidence_class).rank <= profile.rank)) return false;
  const metricIds = [...manifest.included_metrics, ...manifest.boundary_metrics, ...manifest.excluded_metrics].map((row) => row.metric_id);
  const findingIds = [...manifest.included_findings, ...manifest.excluded_findings].map((row) => row.finding_id);
  return metricIds.length === new Set(metricIds).size
    && findingIds.length === new Set(findingIds).size
    && manifest.excluded_finding_ids.join("\u0000") === manifest.excluded_findings.map((row) => row.finding_id).join("\u0000")
    && manifest.metric_counts.applicable === metricIds.length
    && manifest.metric_counts.included === manifest.included_metrics.length
    && manifest.metric_counts.boundary === manifest.boundary_metrics.length
    && manifest.metric_counts.excluded === manifest.excluded_metrics.length;
}

function buildScopedFacts(analysis, runtimeEvidence) {
  const facts = [];
  addFact(facts, "Format", analysis.format, "OBSERVED");
  addFact(facts, "Serialized bytes", finiteInteger(analysis.file_size_bytes ?? analysis.file_size), "OBSERVED");
  addFact(facts, "Operators", finiteInteger(analysis.operator_count ?? analysis.total_ops), "OBSERVED");
  addFact(facts, "Tensors", finiteInteger(analysis.tensor_count ?? artifactIrValues(analysis)?.length), "OBSERVED");
  addFact(facts, "Inputs / outputs", Array.isArray(analysis.inputs) && Array.isArray(analysis.outputs)
    ? `${analysis.inputs.length} / ${analysis.outputs.length}` : null, "OBSERVED");
  addFact(facts, "Assessed MACs", finiteInteger(analysis.total_macs), "DERIVED");
  addFact(facts, "Quantization classification", analysis.quantization_status?.classification || analysis.quantization_status?.label, "OBSERVED/DERIVED");
  addFact(facts, "Peak live activation bytes", finiteInteger(analysis.tensor_liveness?.peak_live_bytes), "DERIVED");
  addFact(facts, "Predicted conditionally delegatable ops", finiteInteger(analysis.xnnpack_delegated_ops ?? analysis.predicted_partition_boundaries?.delegated_op_count), "PREDICTED/DERIVED");
  addFact(facts, "Estimated INT8 opportunity", analysis.estimated_int8_speedup_detail || finiteNumberText(analysis.estimated_int8_speedup), "HEURISTIC/ESTIMATED");
  const assignments = runtimeEvidence?.assignments || runtimeEvidence?.runtime_assignment?.assignments;
  addFact(facts, "Imported runtime assignments", Array.isArray(assignments) ? assignments.length : null, "OBSERVED_RUNTIME");
  addFact(facts, "Serialized LLM weight bytes", finiteInteger(analysis.on_device_llm?.storage?.serialized_tensor_bytes), "OBSERVED_SERIALIZED_STORAGE");
  const memory = analysis.on_device_llm?.memory_feasibility;
  addFact(facts, "LLM memory feasibility status", memory?.status, memory?.evidence_class || "OBSERVED/DERIVED");
  return facts;
}

function addFact(rows, label, value, evidenceClass) {
  if (value == null || value === "") return;
  rows.push({ label, value: String(value), evidence_class: evidenceClass });
}

function finiteInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function finiteNumberText(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(number) : null;
}

function cell(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
