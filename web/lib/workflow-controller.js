import {
  auditFocusCopyFor,
  moduleWorkspaceIdFor,
  workflowActionCopyFor,
  workflowConfigForState,
  workflowStepIndexFor,
} from "./workflow-copy.js";

export function createWorkflowController({
  elements,
  order,
  moduleWorkspaces,
  reportWorkspaces,
  getSelectedArtifactName,
  getAnalysis,
  getFormat,
  getActiveModule,
  setActiveModule,
  syncSelection,
  updatePerformanceVisibility,
}) {
  const {
    body,
    workflowMode,
    workflowNextAction,
    workflowNextDetail,
    workflowSteps,
    auditTabs,
    auditFocusTitle,
    auditFocusCopy,
    auditApplicabilityBoundary,
    auditApplicabilityStatus,
    auditApplicabilityTitle,
    auditApplicabilityReason,
    auditApplicabilityRequired,
    dropzone,
    modelPlan,
    workflowConsole,
    auditWorkbench,
    summary,
    insightDashboard,
    perfVisuals,
    tables,
    diagramSection,
    findingsPanel,
    graphExplorer,
    redesignPanel,
    inferencePanel,
    outputModuleSelector,
    moduleRunConsole,
    actions,
  } = elements;

  let state = "idle";
  let activeWorkspace = "input";
  let activeAuditTab = "overview";
  const mobileAuditView = body.querySelector("#mobileAuditView");
  const evidenceSteps = [...body.querySelectorAll("[data-evidence-workflow]")];
  const primaryWorkflowSteps = workflowSteps.filter((step) => step.dataset.navigationRole === "workflow");
  const analysisTools = workflowSteps.filter((step) => step.dataset.navigationRole === "tool");
  const primaryOrder = primaryWorkflowSteps.map((step) => step.dataset.workflowStep);

  function stepIndex(step) {
    return workflowStepIndexFor(order, step);
  }

  function configFor(nextState = state, detail = {}) {
    return workflowConfigForState(nextState, {
      selected: getSelectedArtifactName() || "selected artifact",
      detail,
      activeWorkspace,
      order,
      format: getFormat() || "tflite",
    });
  }

  function moduleWorkspaceId(workspace) {
    return moduleWorkspaceIdFor(workspace, moduleWorkspaces);
  }

  function syncWorkspaceClass() {
    body.dataset.workspace = activeWorkspace;
    for (const className of [...body.classList]) {
      if (className.startsWith("workspace-")) body.classList.remove(className);
    }
    body.classList.add(`workspace-${activeWorkspace}`);
    body.classList.toggle("workspace-module", moduleWorkspaces.has(activeWorkspace));
  }

  function revealSelectedTab(tabs, predicate) {
    const selected = tabs.find((tab) => !tab.hidden && predicate(tab));
    const rail = selected?.closest(".workflow-rail, .audit-tabs, .module-tabs");
    if (!selected || !rail || rail.scrollWidth <= rail.clientWidth + 1) return;
    requestAnimationFrame(() => {
      const tabRect = selected.getBoundingClientRect();
      const railRect = rail.getBoundingClientRect();
      const centered = rail.scrollLeft + tabRect.left - railRect.left
        - (rail.clientWidth - tabRect.width) / 2;
      rail.scrollTo({ left: Math.max(0, centered), behavior: "auto" });
    });
  }

  function syncActiveModule(workspace) {
    const moduleId = moduleWorkspaceId(workspace);
    if (moduleId) {
      setActiveModule(moduleId);
    } else if (workspace === "output" && !reportWorkspaces.has(getActiveModule())) {
      setActiveModule("engineering_report");
    }
  }

  function syncWorkflowTabs(config = configFor()) {
    const primaryWorkspace = primaryOrder.includes(activeWorkspace) ? activeWorkspace : "audit";
    const activeIndex = Math.max(0, primaryOrder.indexOf(primaryWorkspace));
    const availableIndex = ["idle", "selected", "locked"].includes(state) ? 0 : state === "running" ? 1 : primaryOrder.length - 1;
    for (const step of primaryWorkflowSteps) {
      const index = primaryOrder.indexOf(step.dataset.workflowStep);
      step.dataset.workflowIndex = String(index);
      step.classList.toggle("active", step.dataset.workflowStep === primaryWorkspace);
      step.classList.toggle("complete", index < activeIndex);
      step.classList.toggle("available", index <= availableIndex);
      step.disabled = index > availableIndex;
      step.removeAttribute("aria-disabled");
    }
    const hasAnalysis = Boolean(getAnalysis());
    for (const step of analysisTools) {
      const applicable = step.dataset.formatApplicable !== "false";
      const available = hasAnalysis && applicable;
      step.classList.toggle("active", step.dataset.workflowStep === activeWorkspace);
      step.classList.remove("complete");
      step.classList.toggle("available", available);
      step.disabled = !available;
      step.setAttribute("aria-disabled", String(!available));
    }
    syncSelection(primaryWorkflowSteps, (step) => step.dataset.workflowStep === primaryWorkspace);
    revealSelectedTab(primaryWorkflowSteps, (step) => step.dataset.workflowStep === primaryWorkspace);
    revealSelectedTab(analysisTools, (step) => step.dataset.workflowStep === activeWorkspace);
    syncEvidenceNavigation();
  }

  function syncEvidenceNavigation() {
    const stage = activeWorkspace === "input"
      ? "artifact"
      : ["runtime", "runtime_basin", "offline_test", "deployment_sensitivity"].includes(activeWorkspace)
        ? "runtime"
        : ["output", "findings"].includes(activeWorkspace)
          ? "release"
          : activeWorkspace === "redesign" || activeWorkspace === "audit" && ["xnnpack", "roofline"].includes(activeAuditTab)
            ? "deployment"
            : "derivation";
    for (const navigation of evidenceSteps) {
      const item = navigation.closest("[data-evidence-stage]");
      const active = item?.dataset.evidenceStage === stage;
      item?.toggleAttribute("data-active", active);
      if (active) navigation.setAttribute("aria-current", "step");
      else navigation.removeAttribute("aria-current");
    }
  }

  function updateAction(workspace, fallbackConfig = configFor()) {
    const copy = workflowActionCopyFor(workspace, fallbackConfig, getFormat() || "tflite");
    workflowNextAction.textContent = copy.action;
    workflowNextDetail.textContent = copy.detail;
  }

  function updateVisibility() {
    syncWorkspaceClass();
    const hasAnalysis = Boolean(getAnalysis());
    const moduleWorkspace = moduleWorkspaceId(activeWorkspace);
    dropzone.hidden = false;
    modelPlan.hidden = false;
    workflowConsole.hidden = state === "idle";
    auditWorkbench.hidden = !(hasAnalysis && activeWorkspace === "audit");
    const activeTab = auditTabs.find((tab) => tab.dataset.auditTab === activeAuditTab);
    const applicable = activeTab?.dataset.applicabilityStatus !== "not_applicable";
    if (auditApplicabilityBoundary) auditApplicabilityBoundary.hidden = !(hasAnalysis && activeWorkspace === "audit" && !applicable);
    summary.hidden = !(hasAnalysis && activeWorkspace === "audit" && applicable && activeAuditTab === "overview");
    insightDashboard.hidden = !(hasAnalysis && activeWorkspace === "audit" && applicable && activeAuditTab === "overview");
    perfVisuals.hidden = !(hasAnalysis && activeWorkspace === "audit" && applicable);
    if (!perfVisuals.hidden) updatePerformanceVisibility();
    tables.hidden = !(hasAnalysis && activeWorkspace === "audit" && applicable && activeAuditTab === "roofline");
    diagramSection.hidden = !(hasAnalysis && activeWorkspace === "audit" && applicable && activeAuditTab === "stage");
    if (findingsPanel) findingsPanel.hidden = !(hasAnalysis && activeWorkspace === "findings");
    graphExplorer.hidden = !(hasAnalysis && activeWorkspace === "graph");
    if (redesignPanel) redesignPanel.hidden = !(hasAnalysis && activeWorkspace === "redesign");
    inferencePanel.hidden = !(hasAnalysis && activeWorkspace === "runtime");
    outputModuleSelector.hidden = !(hasAnalysis && activeWorkspace === "output");
    moduleRunConsole.hidden = !(hasAnalysis && moduleWorkspace);
    actions.hidden = !(hasAnalysis && (moduleWorkspace || activeWorkspace === "output"));
  }

  function setWorkspace(workspace = "input", options = {}) {
    const config = configFor();
    const target = stepIndex(workspace) >= 0 ? workspace : "input";
    const index = stepIndex(target);
    const step = workflowSteps.find((item) => item.dataset.workflowStep === target);
    if (!options.force && (step?.disabled || step?.dataset.formatApplicable === "false" || index > config.availableIndex)) return false;
    activeWorkspace = target;
    syncActiveModule(target);
    updateAction(target, config);
    syncWorkflowTabs(config);
    updateVisibility();
    return true;
  }

  function setAuditTab(tabId = "overview") {
    activeAuditTab = auditTabs.some((tab) => tab.dataset.auditTab === tabId) ? tabId : "overview";
    const selected = auditTabs.find((tab) => tab.dataset.auditTab === activeAuditTab);
    const activeDomain = selected?.dataset.auditDomain || activeAuditTab;
    const primaryTabs = auditTabs.filter((tab) => tab.dataset.auditPrimary === "true");
    const lensTabs = auditTabs.filter((tab) => tab.dataset.auditLens === "true");
    for (const tab of primaryTabs) tab.classList.toggle("active", tab.dataset.auditDomain === activeDomain);
    for (const tab of lensTabs) {
      const active = tab.dataset.auditTab === activeAuditTab;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-pressed", String(active));
    }
    if (mobileAuditView) mobileAuditView.value = activeAuditTab;
    syncSelection(primaryTabs, (tab) => tab.dataset.auditDomain === activeDomain);
    revealSelectedTab(auditTabs, (tab) => tab.dataset.auditTab === activeAuditTab);
    const copy = auditFocusCopyFor(activeAuditTab, getFormat() || "tflite");
    if (auditFocusTitle) auditFocusTitle.textContent = copy.title;
    if (auditFocusCopy) auditFocusCopy.textContent = copy.detail;
    if (auditApplicabilityBoundary && selected?.dataset.applicabilityStatus === "not_applicable") {
      if (auditApplicabilityStatus) auditApplicabilityStatus.textContent = "Not applicable";
      if (auditApplicabilityTitle) auditApplicabilityTitle.textContent = `${copy.title} is not applicable`;
      if (auditApplicabilityReason) auditApplicabilityReason.textContent = selected.dataset.applicabilityReason || "This evidence domain is not applicable to the selected artifact class.";
      if (auditApplicabilityRequired) auditApplicabilityRequired.textContent = selected.dataset.applicabilityRequired
        ? `Required evidence: ${selected.dataset.applicabilityRequired}`
        : "No missing file is implied by this state.";
    }
    syncEvidenceNavigation();
    updateVisibility();
  }

  mobileAuditView?.addEventListener("change", () => setAuditTab(mobileAuditView.value));

  function updateState(nextState, detail = {}) {
    state = nextState;
    const config = configFor(nextState, detail);
    workflowMode.textContent = config.mode;
    workflowNextAction.textContent = config.action;
    workflowNextDetail.textContent = config.detail;
    activeWorkspace = config.active;
    syncActiveModule(activeWorkspace);
    syncWorkflowTabs(config);
    updateVisibility();
  }

  return {
    get state() { return state; },
    get activeWorkspace() { return activeWorkspace; },
    get activeAuditTab() { return activeAuditTab; },
    setWorkspace,
    setAuditTab,
    updateState,
    updateVisibility,
  };
}
