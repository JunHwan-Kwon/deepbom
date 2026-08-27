import { formatNumber, padOp } from "./format.js";
import { markdownTable } from "./report-utils.js";

export function tfliteAlternateDelegateCompatibilityMarkdown(analysis) {
  const evidence = analysis?.tflite_delegate_compatibility_evidence;
  if (!evidence) return [
    "## TFLite GPU and NNAPI Source Compatibility (NOT_ASSESSED)",
    "The protected pinned-source delegate rulepack was not loaded for this export. No GPU or NNAPI support, build availability, device acceptance, partitioning, or runtime assignment is inferred.",
  ].join("\n\n");
  const sources = evidence.source_files || [];
  const profiles = evidence.profiles || [];
  const requirements = evidence.build_requirements || [];
  const sections = [
    "## TFLite GPU and NNAPI Source Compatibility (SOURCE+ARTIFACT_PRECHECK/NOT_OBSERVED)",
    "> Registration in pinned TensorFlow source plus passed artifact-visible common checks produces a candidate, never a support or assignment claim. Parser-specific predicates, selected build flags, runtime/device capability, partition policy, and observed placement remain explicit requirements.",
    markdownTable(["Rulepack field", "Value"], [
      ["Assessment / evidence class", `${evidence.assessment_status} / ${evidence.evidence_class}`],
      ["Schema / rulepack SHA-256", `${evidence.schema} / ${evidence.rulepack_sha256}`],
      ["Pinned TensorFlow commit", evidence.tensorflow_source_commit],
      ["Graph op binding", `${formatNumber(evidence.graph_op_count)} op(s)`],
      ["Interpretation boundary", evidence.interpretation_boundary],
    ]),
    markdownTable(["Pinned source", "Scope", "SHA-256", "Reference"], sources.map((source) => [
      source.id,
      source.scope,
      source.sha256,
      source.source_ref,
    ])),
    markdownTable(["Delegate profile", "Registered source ops", "Graph ops assessed", "Source candidates", "Definite exclusions", "Build binding", "Runtime assignment"], profiles.map((profile) => [
      profile.label,
      formatNumber(profile.registered_source_op_count),
      formatNumber(profile.assessed_graph_op_count),
      formatNumber(profile.source_candidate_after_artifact_precheck_count),
      formatNumber(profile.definite_exclusion_count),
      profile.selected_build_status,
      profile.runtime_assignment_status,
    ])),
    markdownTable(["Required runtime/build evidence", "Profile", "Binding", "Affected candidates", "Configuration"], requirements.map((requirement) => [
      requirement.id,
      requirement.profile,
      requirement.binding_status,
      formatNumber(requirement.affected_source_candidate_op_count),
      requirement.required_configuration,
    ])),
  ];
  for (const profile of profiles) {
    sections.push(
      `### ${profile.label} Complete Operator Ledger`,
      markdownTable(["Op", "Version", "Source registration", "Artifact precheck", "Definite exclusion", "Unresolved predicates", "Source version limits", "Source feature tokens", "Pinned source"], (profile.rows || []).map((row) => [
        `#${padOp(row.op_index)} ${row.op_name}`,
        row.op_version,
        row.source_registration_status,
        row.artifact_precheck_status,
        (row.definite_exclusion_reasons || []).join("; ") || "none",
        (row.unresolved_predicates || []).join("; ") || "none",
        row.source_maximum_op_version_candidates?.length
          ? `${row.source_maximum_op_version_candidates.join("/")}; definite max ${row.source_definite_maximum_op_version ?? "branch-dependent"}`
          : "not emitted",
        (row.source_feature_level_tokens || []).map((token) => `${token.name}=${token.value}`).join("; ") || "none",
        row.source_text_sha256 ? `${row.source_ref}; fragment SHA-256 ${row.source_text_sha256}` : row.source_ref,
      ])),
    );
  }
  sections.push(`> ${evidence.interpretation_boundary}`);
  return sections.join("\n\n");
}
