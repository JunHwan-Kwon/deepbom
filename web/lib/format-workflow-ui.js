import { formatEvidenceScope, formatWorkflowApplicability } from "./format-evidence-scope.js";
import { formatNumber } from "./format.js";
import {
  insightDashboardCards,
  insightDashboardFallbackTrafficItems,
  insightDashboardRecommendationItems,
  insightDashboardSignalItems,
  targetConditionCards,
} from "./audit-ui.js";
import { rooflineBar } from "./dom.js";
import { renderOnnxDomainViewer } from "./onnx-domain-viewer.js";
import { renderExecutionPlacementView } from "./execution-placement-view.js";

function elementSet(doc, ids, supplied = {}) {
  return Object.fromEntries(ids.map((entry) => {
    const [key, id] = Array.isArray(entry) ? entry : [entry, entry];
    return [key, supplied[key] || doc?.getElementById(id)];
  }));
}

export function renderFormatInsightFrame({ analysis, insights, runtimeEvidence = null, elements = {}, doc = globalThis.document } = {}) {
  const format = String(analysis?.format || "tflite").toLowerCase();
  const tflite = format === "tflite";
  const scope = formatEvidenceScope(format, { analysis, runtimeEvidence });
  const {
    title,
    description,
    subtitle,
    readinessRing,
    readinessScore,
    readinessLabel,
    readinessRationale,
    rooflineBars,
    rooflineTotal,
    fallbackTrafficList,
  } = {
    ...elementSet(doc, [["title", "insightTitle"], ["description", "insightDescription"], ["subtitle", "insightSubtitle"], "readinessRing", "readinessScore", "readinessLabel", "readinessRationale", "rooflineBars", "rooflineTotal", "fallbackTrafficList"]),
    ...elements,
  };
  if (title) title.textContent = scope.dashboardTitle;
  if (description) description.textContent = scope.dashboardCopy;
  subtitle.textContent = `static audit / ${analysis.filename}`;
  subtitle.title = `Audited artifact: ${analysis.filename}`;
  const readinessCard = readinessRing.closest(".readiness-card");
  if (readinessCard) readinessCard.hidden = !tflite;
  readinessCard?.parentElement?.classList.toggle("artifact-only", !tflite);
  const componentSignalCount = Array.isArray(insights.signals) ? insights.signals.length : 0;
  readinessScore.textContent = tflite
    ? `${componentSignalCount} component signal${componentSignalCount === 1 ? "" : "s"} listed separately. No context-free composite score is reported.`
    : "Use the format-specific assessed evidence cards; no composite score is reported.";
  readinessLabel.textContent = tflite ? insights.label : "TFLite-only heuristic not computed";
  readinessRationale.textContent = tflite ? insights.rationale : "Use the format-specific assessed evidence cards and imported runtime evidence.";
  readinessRing.className = `triage-mark ${tflite ? insights.tone : "neutral"}`;
  readinessRing.title = tflite
    ? "HEURISTIC component signals shown independently. Not model quality, measured performance, or release readiness."
    : `Not applicable to ${scope.label}.`;
  readinessRing.tabIndex = tflite ? 0 : -1;
  readinessRing.setAttribute("aria-label", readinessRing.title);
  const rooflinePanel = rooflineBars.closest(".mini-panel");
  if (rooflinePanel) rooflinePanel.hidden = !tflite;
  const fallbackPanel = fallbackTrafficList.closest(".mini-panel");
  if (fallbackPanel) fallbackPanel.hidden = !tflite;
  rooflineTotal.textContent = tflite ? `${formatNumber(insights.totalOps)} ops` : "Not applicable";
  return { tflite, onnx: format === "onnx", scope };
}

export function renderInsightDashboardView({
  analysis,
  insights,
  runtimeEvidence = null,
  elements = {},
  doc = globalThis.document,
  setActiveWorkspace,
  setActiveAuditTab,
} = {}) {
  const resolved = {
    ...elementSet(doc, ["insightTitle", "insightDescription", "insightSubtitle", "readinessRing", "readinessScore", "readinessLabel", "readinessRationale", "rooflineBars", "rooflineTotal", "fallbackTrafficList", "targetConditionSummary", "insightGrid", "opFilterBar", "topSignals", "recommendationList", "onnxDomainPanel"]),
    ...elements,
  };
  const { tflite } = renderFormatInsightFrame({
    analysis,
    insights,
    runtimeEvidence,
    doc,
    elements: {
      title: resolved.insightTitle,
      description: resolved.insightDescription,
      subtitle: resolved.insightSubtitle,
      ...resolved,
    },
  });
  const {
    targetConditionSummary,
    insightGrid,
    opFilterBar,
    rooflineBars,
    topSignals,
    recommendationList,
    fallbackTrafficList,
    onnxDomainPanel,
  } = resolved;
  targetConditionSummary.replaceChildren(...targetConditionCards(analysis));
  insightGrid.replaceChildren(...insightDashboardCards(analysis, insights));
  insightGrid.onclick = (event) => {
    const card = event.target.closest("[data-jump-tab]");
    if (!card) return;
    const tab = card.dataset.jumpTab;
    const filter = card.dataset.jumpFilter;
    if (tab === "graph") {
      if (filter && opFilterBar) {
        const [group, value = ""] = filter.split(":");
        const chip = opFilterBar.querySelector(`[data-filter-group="${group}"][data-filter-value="${value}"]`);
        if (chip && !chip.classList.contains("active")) chip.click();
      }
      setActiveWorkspace("graph", { force: true });
    } else {
      setActiveAuditTab(tab);
    }
  };
  rooflineBars.replaceChildren(
    rooflineBar("Compute-bound", insights.boundCounts["compute-bound"], insights.totalOps, "compute-bound"),
    rooflineBar("Mixed", insights.boundCounts.mixed, insights.totalOps, "mixed"),
    rooflineBar("Low-intensity", insights.boundCounts["memory-bound"], insights.totalOps, "memory-bound"),
  );
  topSignals.replaceChildren(...insightDashboardSignalItems(insights));
  recommendationList.replaceChildren(...insightDashboardRecommendationItems(analysis));
  fallbackTrafficList.replaceChildren(...insightDashboardFallbackTrafficItems(analysis));
  renderOnnxDomainViewer(onnxDomainPanel, analysis);
  return { tflite };
}

export function renderAuditClaimBoundaryView({
  format,
  analysis = null,
  runtimeEvidence = null,
  elements = {},
  doc = globalThis.document,
} = {}) {
  const {
    boundary,
    status,
    evidence,
    depth,
    assessed,
    runtime,
    nextProof,
    release,
    reportScopeCopy,
  } = {
    ...elementSet(doc, [["boundary", "auditClaimBoundary"], ["status", "auditClaimStatus"], ["evidence", "auditClaimEvidence"], ["depth", "auditClaimDepth"], ["assessed", "auditClaimAssessed"], ["runtime", "auditClaimRuntime"], ["nextProof", "auditClaimNextProof"], ["release", "auditClaimRelease"], ["reportScopeCopy", "engineeringReportScopeCopy"]]),
    ...elements,
  };
  if (!boundary) return;
  if (!format) {
    boundary.hidden = true;
    renderExecutionPlacementView(doc?.getElementById("executionPlacementPanel"), null, null, { doc });
    return;
  }
  const scope = formatEvidenceScope(format, { analysis, runtimeEvidence });
  boundary.hidden = false;
  status.textContent = analysis ? scope.completion : `${scope.label} audit scope selected`;
  evidence.textContent = analysis ? scope.evidenceClass : "EXPECTED STATIC SCOPE";
  depth.textContent = scope.depth;
  assessed.textContent = analysis ? scope.assessed : `Will assess: ${scope.assessed}.`;
  runtime.textContent = analysis ? `${scope.runtimeStatus}. ${scope.runtimeBoundary}.` : scope.runtimeBoundary;
  nextProof.textContent = analysis ? `${scope.nextProof}.` : `After the static audit: ${scope.nextProof}.`;
  release.textContent = analysis
    ? scope.releaseStatus
    : "Completion of the future static run will not establish task accuracy, runtime placement, or release readiness.";
  renderEvidenceSpine(boundary, scope, Boolean(analysis));
  renderExecutionPlacementView(doc?.getElementById("executionPlacementPanel"), analysis, runtimeEvidence, { doc });
  if (reportScopeCopy) {
    reportScopeCopy.textContent = `Public summary: ${scope.assessed}. ${scope.runtimeBoundary}. The login-free Engineering Report adds the complete synthesized evidence ledger and action queue; reusable raw derivatives retain a separate access boundary.`;
  }
}

function renderEvidenceSpine(boundary, scope, analyzed) {
  const runtimeBound = scope.runtimeObserved;
  const deployment = ({
    tflite: ["Source-predicted", "partial"],
    onnx: ["Source eligibility", "partial"],
    gguf: ["Source prerequisites", "partial"],
    coreml: [scope.placementEstimateBound ? "Plan anticipated" : "OS floor only", "partial"],
    safetensors: ["Not serialized", "missing"],
  })[scope.id] || ["Runtime contract required", "missing"];
  const runtime = runtimeBound
    ? ["Observed import", "complete"]
    : scope.placementEstimateBound
      ? ["Plan, not execution", "partial"]
      : scope.runtimeConfigurationBound
        ? ["Configuration bound", "partial"]
        : ["Not imported", "missing"];
  const stages = {
    artifact: [analyzed ? "Assessed" : "Selected, not audited", analyzed ? "complete" : "partial"],
    derivation: [analyzed ? "Derived" : "Pending audit", analyzed ? "complete" : "pending"],
    deployment: analyzed ? deployment : ["Pending audit", "pending"],
    runtime,
    release: ["Not assessed", "missing"],
  };
  for (const item of boundary.querySelectorAll("[data-evidence-stage]")) {
    const [label, tone] = stages[item.dataset.evidenceStage] || ["Not assessed", "missing"];
    const value = item.querySelector("strong");
    const stage = item.querySelector("span")?.textContent?.trim() || item.dataset.evidenceStage;
    const navigation = item.querySelector("button");
    if (value) value.textContent = label;
    if (navigation) {
      if (item.dataset.evidenceStage === "deployment") {
        navigation.dataset.evidenceAuditTab = scope.id === "tflite" ? "xnnpack" : "overview";
      }
      const runtimeNavigable = ["tflite", "onnx"].includes(scope.id);
      navigation.disabled = item.dataset.evidenceStage !== "artifact"
        && (!analyzed || item.dataset.evidenceStage === "runtime" && !runtimeNavigable);
      navigation.setAttribute("aria-label", `${stage}: ${label}. ${navigation.disabled ? "Unavailable until the required evidence is present." : "Open the corresponding evidence workspace."}`);
    }
    item.dataset.status = tone;
  }
}

export function syncFormatWorkflowVisibilityView({
  format,
  analysis = null,
  workflowSteps = [],
  moduleTabs = [],
  currentUser = null,
  activeModule = "",
  setActiveModule = () => {},
  doc = globalThis.document,
} = {}) {
  if (!format) return;
  const applicability = formatWorkflowApplicability(format, analysis);
  const visibleSteps = {
    input: true,
    audit: true,
    findings: true,
    graph: applicability.graph,
    redesign: applicability.redesign,
    runtime: applicability.runtime,
    deepbom: applicability.protectedSourceAnalysis,
    runtime_basin: applicability.tfliteResearch,
    offline_test: applicability.tfliteResearch && currentUser?.role === "admin",
    deployment_sensitivity: applicability.tfliteResearch,
    output: true,
  };
  for (const step of workflowSteps) {
    const applicable = visibleSteps[step.dataset.workflowStep] !== false;
    step.dataset.formatApplicable = String(applicable);
    step.hidden = !applicable;
  }
  for (const tab of moduleTabs) {
    const module = tab.dataset.moduleTab;
    const research = ["deepbom", "runtime_basin", "offline_test", "deployment_sensitivity"].includes(module);
    const applicable = !research || (module === "deepbom" ? applicability.protectedSourceAnalysis : applicability.tfliteResearch);
    tab.dataset.formatApplicable = String(applicable);
    tab.hidden = !applicable || tab.dataset.moduleTab === "offline_test" && currentUser?.role !== "admin";
  }
  const researchGroup = doc?.getElementById("workflowResearchGroup");
  if (researchGroup) {
    researchGroup.hidden = !applicability.protectedSourceAnalysis;
    if (!applicability.protectedSourceAnalysis) researchGroup.open = false;
  }
  const activeTab = [...moduleTabs].find((tab) => tab.dataset.moduleTab === activeModule);
  if (activeTab?.hidden) setActiveModule("engineering_report");
  const quantLabsTab = doc?.querySelector('[data-audit-tab="quant-labs"]');
  if (quantLabsTab) {
    quantLabsTab.querySelector("strong").textContent = format === "onnx" ? "Q/DQ Contracts" : "Quant Labs";
    quantLabsTab.querySelector("em").textContent = format === "onnx" ? "QOperator" : "Q/DQ";
  }
  const onnx = format === "onnx";
  const deepStep = doc?.querySelector('[data-workflow-step="deepbom"]');
  const deepTab = doc?.querySelector('[data-module-tab="deepbom"]');
  const deepPanel = doc?.querySelector('[data-module-run-panel="deepbom"]');
  const kernelTab = doc?.querySelector('[data-explorer-tab="kernels"]');
  if (kernelTab) kernelTab.textContent = "Execution Placement";
  if (deepStep) {
    deepStep.querySelector("span").textContent = onnx ? "Deployment" : "Artifact";
    deepStep.querySelector("strong").textContent = onnx ? "ORT EP Compatibility" : "Geometry Proxy";
    deepStep.querySelector("p").textContent = onnx ? "Pinned source candidates" : "Weight/topology descriptors";
  }
  if (deepTab) {
    deepTab.querySelector("span").textContent = onnx ? "Deployment" : "Research";
    deepTab.querySelector("strong").textContent = onnx ? "ORT EP Compatibility" : "Artifact Geometry";
  }
  if (deepPanel) {
    deepPanel.querySelector("h3").textContent = onnx
      ? "ORT EP Compatibility: pinned source registrations and artifact-visible eligibility."
      : "Artifact Geometry: deploy-artifact weight and topology descriptors.";
    deepPanel.querySelector("#runDeepBom").textContent = onnx ? "Load ORT EP Compatibility" : "Run Artifact Geometry";
  }
}
