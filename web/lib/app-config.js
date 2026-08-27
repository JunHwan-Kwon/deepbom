// Canonical analyzer/rulepack identity for both ungated runtime code and gated
// report formatters. Release checks bind these values to package and HTML metadata.
export const ANALYZER_VERSION = "2026-08-03";
export const ANALYZER_SEMANTIC_VERSION = "1.94.6";
export const RULEPACK_VERSION = "deepbom.rulepack.2026-07-24.63";
export const DEEPBOM_CITATION = "Kwon, J. (2026). DEEPBOM: Browser-Native Static Analysis of On-Device Neural Network Deployment Artifacts (Version 1.94.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21834509";

export const WORKFLOW_ORDER = ["input", "audit", "findings", "graph", "redesign", "runtime", "deepbom", "runtime_basin", "offline_test", "deployment_sensitivity", "output"];

export const MODULE_WORKSPACES = new Set(["deepbom", "runtime_basin", "offline_test", "deployment_sensitivity"]);

export const DEFAULT_REPORT_WORKSPACES = new Set(["engineering_report", "export_contracts"]);

export const REPORT_WORKSPACES = new Set(["engineering_report", "export_contracts", "regulatory_report"]);

export const TARGET_COMPARISON_IDS = ["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"];

export function deploymentFrontierTargetIds(targetProfiles = [], selectedTargetId = "") {
  const availableIds = new Set(targetProfiles.map((profile) => profile?.id).filter(Boolean));
  const targetIds = TARGET_COMPARISON_IDS.filter((id) => availableIds.has(id));
  if (selectedTargetId && availableIds.has(selectedTargetId) && !targetIds.includes(selectedTargetId)) {
    targetIds.push(selectedTargetId);
  }
  return targetIds;
}

export function deploymentFrontierMatchesTargetIds(frontier, targetIds) {
  const observedIds = frontier?.targets?.map((target) => target?.target_id);
  return Array.isArray(observedIds)
    && observedIds.length === targetIds.length
    && observedIds.every((id, index) => id === targetIds[index]);
}

export function resolveReportTargetBinding({
  requestedTargetId = "",
  activeTargetId = "",
  cachedTargetIds = [],
  hasArtifact = false,
  artifactOnly = false,
} = {}) {
  const targetId = requestedTargetId || activeTargetId || "";
  const result = (state, analyzed = false, active = false, bindingScope = "target") => ({
    targetId,
    state,
    analyzed,
    active,
    canCopy: active,
    bindingScope,
  });
  if (!hasArtifact) {
    return result("unavailable");
  }
  if (artifactOnly) {
    return {
      targetId: "",
      state: "artifact",
      analyzed: true,
      active: true,
      canCopy: true,
      bindingScope: "artifact",
    };
  }
  if (!targetId) return result("unavailable");
  if (targetId === activeTargetId) {
    return result("active", true, true);
  }
  const cached = new Set(cachedTargetIds);
  if (cached.has(targetId)) {
    return result("cached", true);
  }
  return result("required");
}

export function resolveReportTargetId({
  targetProfileApplicable = true,
  availableTargetIds = [],
  requestedTargetId = "",
  activeTargetId = "",
  selectedTargetId = "",
  fallbackTargetId = "",
} = {}) {
  if (!targetProfileApplicable) return "";
  const available = new Set(availableTargetIds);
  return [requestedTargetId, activeTargetId, selectedTargetId, fallbackTargetId]
    .find((id) => id && available.has(id)) || "";
}

export function reportTargetLabel({ targetProfileApplicable = true, targetId = "", targetProfiles = [] } = {}) {
  if (!targetProfileApplicable) return "Artifact-only / no execution target";
  return targetProfiles.find((profile) => profile.id === targetId)?.label || targetId || "No target";
}

export function reportTargetControlCopy(binding = {}) {
  const statusText = ({
    unavailable: "Not analyzed",
    required: "Analysis required",
    cached: "Analyzed \u00b7 ready to load",
    active: "Analyzed \u00b7 active",
    artifact: "Artifact-bound \u00b7 no target assumptions",
  })[binding.state] || "Not analyzed";
  const analyzeText = ({
    artifact: "Artifact-bound",
    active: "Analyzed",
    cached: "Use analyzed",
    required: "Analyze target",
    unavailable: "Run audit first",
  })[binding.state] || "Run audit first";
  return {
    statusText,
    analyzeText,
    analyzeDisabled: ["artifact", "active", "unavailable"].includes(binding.state),
  };
}

export function populateReportTargetSelect(select, {
  artifactOnly = false,
  targetProfiles = [],
  selectedTargetId = "",
} = {}) {
  if (!select) return;
  const profiles = artifactOnly
    ? [{ id: "", label: "Artifact-only / no execution target" }]
    : targetProfiles;
  select.replaceChildren(...profiles.map((profile) => new Option(profile.label, profile.id)));
  select.value = artifactOnly ? "" : selectedTargetId;
}
