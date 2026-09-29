import { DEEPBOM_WORKFLOWS } from "./workflow-catalog.js";

export function parseWorkflowLink(hash) {
  const id = /^#workflow=([a-z]+)$/.exec(hash || "")?.[1];
  return DEEPBOM_WORKFLOWS.find(row => row.id === id && row.web) ?? null;
}

export function installWorkflowContinuation({ host, getAnalysis, navigate, selectAuditTab }) {
  let pending = parseWorkflowLink(window.location.hash);
  const banner = document.createElement("aside");
  banner.className = "workflow-continuation";
  banner.style.cssText = "margin:16px 0;padding:16px;border:1px solid currentColor;border-radius:10px;text-align:left;font:inherit";
  banner.setAttribute("aria-live", "polite");
  const title = document.createElement("strong"), description = document.createElement("p"), status = document.createElement("p");
  const dismiss = document.createElement("button"); dismiss.type = "button"; dismiss.textContent = "Dismiss continuation";
  const open = document.createElement("button"); open.type = "button"; open.textContent = "Open requested workspace";
  open.style.marginRight = "10px";
  banner.append(title, description, status, open, dismiss); host.prepend(banner);
  const render = () => {
    banner.hidden = !pending;
    if (!pending) return;
    title.textContent = `Continue: ${pending.title}`;
    description.textContent = `${pending.description}. Select your model and run its static audit. The file and result are not transferred from ChatGPT. Compare the resulting SHA-256 with your conversation before treating this as the same artifact.`;
    status.textContent = pending.limits;
    open.disabled = !getAnalysis();
  };
  const apply = () => {
    if (!pending || !getAnalysis()) return false;
    open.disabled = false;
    if (!pending.formats.includes(getAnalysis().format)) { status.textContent = "This workflow is unavailable for the selected format. Select a supported artifact or choose another analysis view."; return false; }
    // The normal navigation controller retains format and authorization gates.
    if (!navigate(pending.web.workspace)) { status.textContent = "This workspace is currently unavailable. Complete the audit and check its access and format requirements."; return false; }
    if (pending.web.audit_tab) selectAuditTab(pending.web.audit_tab);
    status.textContent = `Opened ${pending.title}. ${pending.limits} Verify the artifact SHA-256 against your conversation; model identity was not transferred in this link.`;
    pending = null; open.disabled = true;
    return true;
  };
  open.addEventListener("click", apply);
  dismiss.addEventListener("click", () => { pending = null; banner.hidden = true; });
  window.addEventListener("hashchange", () => { pending = parseWorkflowLink(window.location.hash); render(); if (getAnalysis()) apply(); });
  render();
  return { apply };
}
