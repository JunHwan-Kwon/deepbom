export function buildChangeAnalysis(analysis, { priorSnapshot = null, identity = {} } = {}) {
  const notPerformed = (reasonCode, reason) => ({
    schema: "deepbom.change_analysis.v1",
    status: "not_performed",
    comparison_basis: null,
    reason_code: reasonCode,
    reason,
    prior_artifact_sha256: priorSnapshot?.sha256 || null,
    current_artifact_sha256: analysis?.model_sha256 || identity.sha256 || null,
  });
  if (!priorSnapshot) return notPerformed("NO_PRIOR_SELECTED", "No prior artifact was selected.");
  const currentSha = analysis?.model_sha256 || identity.sha256 || "";
  if (!priorSnapshot.sha256 || priorSnapshot.sha256 === currentSha) return notPerformed("INVALID_PRIOR_IDENTITY", "The selected snapshot has no distinct artifact SHA-256.");
  if (String(priorSnapshot.format || "").toLowerCase() !== String(analysis?.format || identity.format || "").toLowerCase()) return notPerformed("FORMAT_MISMATCH", "Current and prior artifacts use different formats.");
  if ((priorSnapshot.target || "") !== (analysis?.target_profile?.id || "")) return notPerformed("TARGET_MISMATCH", "Current and prior audits use different target profiles.");
  let basis = null;
  if (priorSnapshot.modelLineageId && analysis?.model_lineage_id && priorSnapshot.modelLineageId === analysis.model_lineage_id) basis = "matching_model_lineage_id";
  else if (analysis?.previous_artifact_sha256 && priorSnapshot.sha256 === analysis.previous_artifact_sha256) basis = "current_previous_artifact_sha256";
  else if (priorSnapshot.derivationManifestId && analysis?.derivation_manifest_id && priorSnapshot.derivationManifestId === analysis.derivation_manifest_id) basis = "matching_derivation_manifest_id";
  else if (priorSnapshot.explicitlySelectedForComparison === true) basis = "explicit_user_selection";
  if (!basis) return notPerformed("LINEAGE_NOT_ESTABLISHED", "Filename similarity is not lineage evidence. Supply a model lineage ID, previous artifact SHA-256, derivation manifest ID, or explicit user selection.");
  const currentTotalMacs = analysis?.total_macs == null ? null : Number(analysis.total_macs);
  const priorTotalMacs = priorSnapshot.totalMacs == null ? null : Number(priorSnapshot.totalMacs);
  return {
    schema: "deepbom.change_analysis.v1",
    status: "assessed",
    comparison_basis: basis,
    reason_code: null,
    reason: "The prior relationship is established by an explicit lineage signal.",
    prior_artifact_sha256: priorSnapshot.sha256,
    current_artifact_sha256: currentSha,
    prior_snapshot: {
      audited_at: priorSnapshot.updatedAt || priorSnapshot.createdAt || null,
      analyzer_version: priorSnapshot.analyzerVersion || null,
      rulepack_version: priorSnapshot.rulepackVersion || null,
    },
    deltas: {
      operator_count: Number(analysis?.operator_count || 0) - Number(priorSnapshot.operatorCount || 0),
      total_macs: currentTotalMacs == null || priorTotalMacs == null ? null : currentTotalMacs - priorTotalMacs,
      predicted_chain_breaks: Number(analysis?.xnnpack_effective_chain_breaks || 0) - Number(priorSnapshot.delegation?.effectiveChainBreaks || 0),
      quantized_compute_mac_ratio: analysis?.quantization_status?.quantized_compute_mac_percent == null || priorSnapshot.quant?.quantComputeMacPercent == null
        ? null
        : Number(analysis.quantization_status.quantized_compute_mac_percent) - Number(priorSnapshot.quant.quantComputeMacPercent),
    },
  };
}
