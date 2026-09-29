export function syncTabSelection(tabs, isActive) {
  for (const tab of tabs) {
    const active = Boolean(isActive(tab));
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
}

export function installWorkspaceNavigation({
  workflowSteps,
  evidenceSteps,
  auditTabs,
  moduleTabs,
  explorerTabs,
  onWorkspace,
  onAudit,
  onModule,
  onLockedModule,
  onExplorer,
}) {
  for (const step of [...workflowSteps, ...evidenceSteps]) {
    step.addEventListener("click", () => {
      onWorkspace(step.dataset.workflowStep || step.dataset.evidenceWorkflow);
      if (step.dataset.evidenceAuditTab) onAudit(step.dataset.evidenceAuditTab);
    });
  }
  for (const tab of auditTabs) tab.addEventListener("click", () => onAudit(tab.dataset.auditTab));
  for (const tab of moduleTabs) {
    tab.addEventListener("click", () => tab.getAttribute("aria-disabled") === "true"
      ? onLockedModule(tab.dataset.featureId || "advanced")
      : onModule(tab.dataset.moduleTab));
  }
  installRovingTablist(workflowSteps.filter((tab) => tab.dataset.navigationRole === "workflow"), (tab) => onWorkspace(tab.dataset.workflowStep));
  for (const group of groupedTablists(auditTabs)) installRovingTablist(group, (tab) => onAudit(tab.dataset.auditTab));
  installRovingTablist(moduleTabs, (tab) => tab.click());
  installRovingTablist(explorerTabs, (tab) => onExplorer(tab.dataset.explorerTab));
}

function groupedTablists(tabs) {
  const groups = new Map();
  for (const tab of tabs) {
    const owner = tab.closest('[role="tablist"]');
    if (!owner) continue;
    if (!groups.has(owner)) groups.set(owner, []);
    groups.get(owner).push(tab);
  }
  return [...groups.values()];
}

function installRovingTablist(tabs, activate) {
  for (const tab of tabs) {
    tab.setAttribute("role", "tab");
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const available = tabs.filter((item) => !item.hidden && item.getClientRects().length > 0 && !item.disabled && item.getAttribute("aria-disabled") !== "true");
      if (!available.length) return;
      const current = Math.max(0, available.indexOf(tab));
      const next = event.key === "Home" ? 0 : event.key === "End" ? available.length - 1
        : (current + (event.key === "ArrowRight" ? 1 : -1) + available.length) % available.length;
      event.preventDefault();
      available[next].focus();
      activate(available[next]);
    });
  }
}

export function initPinnedSessionOffset({ topbar, sessionAnchor }) {
  const root = document.documentElement;
  const update = () => {
    const stickyHeight = (element) => {
      const position = element ? getComputedStyle(element).position : "static";
      return ["fixed", "sticky"].includes(position)
        ? Math.round(element.getBoundingClientRect().height || 0)
        : 0;
    };
    const topbarH = stickyHeight(topbar);
    const anchorH = stickyHeight(sessionAnchor);
    const staticMobileChrome = window.matchMedia("(max-width: 820px)").matches;
    root.style.setProperty("--session-sticky-top", `${staticMobileChrome ? 0 : topbarH}px`);
    const coverH = staticMobileChrome ? 0 : topbarH + anchorH + 8;
    root.style.setProperty("--sticky-cover-height", `${coverH}px`);
    root.style.scrollPaddingTop = `${coverH}px`;
  };
  update();
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(update);
    if (topbar) ro.observe(topbar);
    if (sessionAnchor) ro.observe(sessionAnchor);
  }
  window.addEventListener("resize", update);
}
