import { readFileSync } from "node:fs";
import {
  DEFAULT_REPORT_WORKSPACES,
  MODULE_WORKSPACES,
  REPORT_WORKSPACES,
  resolveReportTargetBinding,
  WORKFLOW_ORDER,
} from "../web/lib/app-config.js";
import { REGULATORY_BUNDLE_MODULE_SPECS } from "../web/lib/bundle.js";
import { workflowActionCopyFor } from "../web/lib/workflow-copy.js";

const errors = [];
const appSource = readFileSync("web/app.js", "utf8");
const primaryStyleSource = readFileSync("web/styles.css", "utf8");
const styleSource = [
  readFileSync("web/app-shell.css", "utf8"),
  readFileSync("web/device-workspace.css", "utf8"),
  primaryStyleSource,
  readFileSync("web/report-workspace.css", "utf8"),
].join("\n");
const medicalSurfaceSource = readFileSync("web/lib/app-surface.js", "utf8");
const privacySource = readFileSync("web/lib/privacy-ui.js", "utf8");
const performanceVisualSource = readFileSync("web/lib/performance-visuals.js", "utf8");
const workspaceNavigationSource = readFileSync("web/lib/workspace-navigation.js", "utf8");
const offlineDeviceSource = readFileSync("web/lib/offline-device-controller.js", "utf8");
const adminSource = readFileSync("web/admin.js", "utf8");
const workerSource = readFileSync("worker/index.js", "utf8");
const webServiceWorkerSource = readFileSync("web/sw.js", "utf8");
const syntheticModelSource = readFileSync("scripts/reconstruct_tflite.py", "utf8");
const moduleWorkspaces = [...MODULE_WORKSPACES];
const regulatoryBundleIds = REGULATORY_BUNDLE_MODULE_SPECS.map((item) => item.id);
const regulatoryBundleWorkspaces = REGULATORY_BUNDLE_MODULE_SPECS.map((item) => item.workspace);

checkDefaultPage();
checkMedicalSurface();
checkAccessStateContract();
checkResponsiveWorkspaceCss();
checkPrivacyAndAccessibilityContract();
checkReportTargetBindingContract();
checkDomInjectionContract();
expectUnique("regulatory bundle module ids", regulatoryBundleIds);

if (errors.length) {
  console.error("Workflow contract check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(
  `Workflow contract passed (1 HTML shell, medical surface augmentation, ${WORKFLOW_ORDER.length} workflow steps, ${moduleWorkspaces.length} modules, ${regulatoryBundleIds.length} bundle modules).`,
);

function checkDefaultPage() {
  const label = "main app";
  const html = readFileSync("web/index.html", "utf8");
  const reportWorkspaces = [...DEFAULT_REPORT_WORKSPACES];
  const workflowSteps = attrValues(html, "data-workflow-step");
  const workflowModules = attrValues(html, "data-workflow-module");
  const moduleTabs = attrValues(html, "data-module-tab");
  const moduleRunPanels = attrValues(html, "data-module-run-panel");
  const moduleResultPanels = attrValues(html, "data-module-panel");
  const outputWorkspaces = [...moduleWorkspaces, ...reportWorkspaces];

  expectExactOrder(`${label} workflow rail`, workflowSteps, WORKFLOW_ORDER);
  expectExactSet(`${label} workflow module markers`, workflowModules, moduleWorkspaces);
  expectContainsAll(`${label} workflow rail`, workflowSteps, moduleWorkspaces);
  expectExactSet(`${label} output module tabs`, moduleTabs, outputWorkspaces);
  expectExactSet(`${label} module run panels`, moduleRunPanels, outputWorkspaces);
  expectExactSet(`${label} module result panels`, moduleResultPanels, outputWorkspaces);

  for (const report of reportWorkspaces) {
    if (workflowSteps.includes(report)) {
      errors.push(`${label}: report workspace ${report} must stay under Output, not the main workflow rail.`);
    }
  }

  expectAbsent(`${label} public regulatory tab`, moduleTabs, "regulatory_report");
  expectAbsent(`${label} public regulatory run panel`, moduleRunPanels, "regulatory_report");
  expectAbsent(`${label} public regulatory result panel`, moduleResultPanels, "regulatory_report");

  const auditedInputCopy = workflowActionCopyFor("input", { availableIndex: 10 });
  if (auditedInputCopy.action !== "Replace artifact or target" || !auditedInputCopy.detail.includes("current audit remains available")) {
    errors.push("audited Artifact workspace must describe replacement without claiming that no artifact is selected.");
  }
  const outputCopy = workflowActionCopyFor("output");
  if (!outputCopy.detail.includes("login-free") || !outputCopy.detail.includes("raw derivatives") || outputCopy.detail.includes("sign in for the complete editable report")) {
    errors.push("Output workspace copy must expose watermarked reports without login while preserving the separate raw-derivative access boundary.");
  }

  for (const copy of ["1. Artifact", "2. Derive", "3. Model", "4. Observe", "5. Release", "7 docs + pack", "7-document inventory", "Next proof"]) {
    if (!html.includes(copy)) errors.push(`${label}: evidence workflow or exact export-set copy is missing ${copy}.`);
  }
  for (const copy of ["Not selected", "Pending artifact", "Pending format", "Not imported", "Not assessed"]) {
    if (!html.includes(`<strong>${copy}</strong>`)) errors.push(`${label}: initial Evidence Spine state is missing ${copy}.`);
  }
  if (!/id="workflowConsole"[^>]*\bhidden\b/.test(html)) {
    errors.push(`${label}: workflow navigation must remain progressively disclosed until an artifact is staged.`);
  }
  if (!/id="summary"[^>]*\bhidden\b/.test(html)) {
    errors.push(`${label}: the static summary must start hidden before application hydration.`);
  }
  for (const copy of ["Synthetic Device Benchmark", "zero-weight reconstruction", "backend consistency checks"]) {
    if (!html.includes(copy)) errors.push(`${label}: bounded device/runtime copy is missing ${copy}.`);
  }
  for (const stale of ["Offline Test", "queue the current artifact for a real device benchmark", "runtime backend stability", "build identity loading"]) {
    if (html.includes(stale)) errors.push(`${label}: stale or over-broad product copy remains: ${stale}.`);
  }
  if (!/id="applicationBuild"\s+hidden/.test(html)) {
    errors.push(`${label}: build provenance must stay hidden until concrete identity values are hydrated.`);
  }
  if (!html.includes('class="author-details"') || !html.includes('class="evidence-class-disclosure"')) {
    errors.push(`${label}: mobile provenance and evidence-class disclosures must preserve their complete content.`);
  }
  if (!html.includes('class="medical-context-summary"') || !html.includes('class="medical-context-detail"')) {
    errors.push(`${label}: mobile clinical context must keep a compact summary and the complete source text.`);
  }
  const mobileAuditSelect = html.match(/<select id="mobileAuditView"[^>]*>([\s\S]*?)<\/select>/)?.[1] || "";
  const mobileAuditOptions = attrValues(mobileAuditSelect, "value");
  const auditTabIds = attrValues(html, "data-audit-tab");
  const expectedAuditViews = ["overview", "xnnpack", "accelerator", "quant", "quant-labs", "llm", "roofline", "stage"];
  if (!sameSet(mobileAuditOptions, expectedAuditViews) || mobileAuditOptions.length !== expectedAuditViews.length) {
    errors.push(`${label}: mobile audit selector must preserve every static analysis view exactly once.`);
  }
  if (!sameSet(auditTabIds, expectedAuditViews) || auditTabIds.length !== expectedAuditViews.length) {
    errors.push(`${label}: desktop audit tabs must preserve every static analysis view exactly once.`);
  }
  const auditTabLabels = [...html.matchAll(/data-audit-tab="[^"]+"[^>]*aria-label="([^"]+)"/g)].map((match) => match[1]);
  if (auditTabLabels.length !== expectedAuditViews.length || new Set(auditTabLabels).size !== auditTabLabels.length) {
    errors.push(`${label}: every static audit tab needs a unique explicit accessible name.`);
  }
  const documentInventory = html.match(/<details class="export-document-inventory">[\s\S]*?<\/details>/)?.[0] || "";
  if ((documentInventory.match(/<li>/g) || []).length !== 7) {
    errors.push(`${label}: machine export inventory must enumerate exactly seven evidence documents.`);
  }
}

function checkPrivacyAndAccessibilityContract() {
  const html = readFileSync("web/index.html", "utf8");
  for (const [condition, message] of [
    [html.includes('id="researchConsent" type="checkbox"'), "research telemetry consent must be a visible optional checkbox"],
    [/id="agreementBackdrop"[^>]*\bhidden\b/.test(html), "privacy acknowledgement must start hidden so accepted sessions do not flash the modal before JavaScript hydration"],
    [!html.includes('id="consentGate"'), "static analysis must not be gated on optional research telemetry"],
    [privacySource.includes("readAgreementAccepted()"), "accepted session privacy acknowledgement should suppress repeat modal display"],
    [privacySource.includes("openModal(agreementBackdrop"), "unaccepted sessions must explicitly reveal the initially hidden privacy acknowledgement through the shared accessible modal controller"],
    [privacySource.includes("writeResearchConsent(researchConsent.checked)"), "privacy acknowledgement must preserve the user's explicit telemetry choice"],
    [!privacySource.includes("writeResearchConsent(true)"), "privacy acknowledgement must not force-enable telemetry"],
    [html.includes("Inspect the artifact that will actually run."), "first-visit surface must state the deployment-artifact problem before controls"],
    [html.includes('id="preAuditReference"') && html.includes('id="sampleEvidenceGlance"'), "first-visit evidence guide must preserve the selected hash-pinned baseline without forcing it into the task path"],
    [html.includes("Why this matters for medical AI"), "medical-AI relevance must remain visible without changing the general product scope"],
    [html.includes("zero-weight, shape/op/dtype/quantization-equivalent synthetic reconstruction"), "offline device path must distinguish its synthetic reconstruction from the original artifact"],
    [appSource.includes('body: JSON.stringify({ fingerprint, target })') && offlineDeviceSource.includes("await queueTarget(fingerprint, target)"), "offline queue action must send only the structure fingerprint and selected target"],
    [workerSource.includes("No model bytes or weights were uploaded"), "offline queue response must preserve its no-artifact-upload contract"],
    [syntheticModelSource.includes("Synthetic zero-weight reconstruction for benchmark_model"), "device benchmark builder must identify the generated artifact as a zero-weight reconstruction"],
    [appSource.includes("installWorkspaceNavigation") && workspaceNavigationSource.includes("installRovingTablist") && workspaceNavigationSource.includes('setAttribute("aria-selected"'), "workflow, audit, module, and explorer tabs need keyboard and ARIA state"],
    [styleSource.includes(".topbar > div:first-child,\n  .topbar-meta") && styleSource.includes("flex: 0 1 auto"), "mobile topbar must reset inherited desktop flex bases"],
    [styleSource.includes(".session-anchor .mobile-audit-view") && styleSource.includes(".session-anchor .audit-tabs {\n    display: none"), "mobile audit navigation must use the compact all-view selector"],
    [styleSource.includes(".upload-controls { order: 2; }") && styleSource.includes(".pre-audit-reference { order: 3; }"), "mobile artifact controls must precede the complete collapsible evidence guide"],
    [styleSource.includes("body:not(.workspace-input) .session-anchor .pre-audit-reference"), "pre-audit reference material must leave the workspace after an artifact is loaded"],
  ]) {
    if (!condition) errors.push(message);
  }
}

function checkMedicalSurface() {
  const reportWorkspaces = [...REPORT_WORKSPACES];
  const tabWorkspaces = new Set([
    ...attrValues(readFileSync("web/index.html", "utf8"), "data-module-tab"),
    ...attrValues(medicalSurfaceSource, "data-module-tab"),
  ]);
  const runWorkspaces = new Set([
    ...attrValues(readFileSync("web/index.html", "utf8"), "data-module-run-panel"),
    ...attrValues(medicalSurfaceSource, "data-module-run-panel"),
  ]);
  const panelWorkspaces = new Set([
    ...attrValues(readFileSync("web/index.html", "utf8"), "data-module-panel"),
    ...attrValues(medicalSurfaceSource, "data-module-panel"),
  ]);
  const outputWorkspaces = [...moduleWorkspaces, ...reportWorkspaces];

  expectExactSet("medical surface output module tabs", [...tabWorkspaces], outputWorkspaces);
  expectExactSet("medical surface module run panels", [...runWorkspaces], outputWorkspaces);
  expectExactSet("medical surface module result panels", [...panelWorkspaces], outputWorkspaces);
  expectContainsAll("medical surface regulatory bundle module tabs", [...tabWorkspaces], regulatoryBundleWorkspaces);
  expectContainsAll("medical surface regulatory bundle run panels", [...runWorkspaces], regulatoryBundleWorkspaces);
  expectContainsAll("medical surface regulatory bundle result panels", [...panelWorkspaces], regulatoryBundleWorkspaces);
  expectSourceOrder(
    appSource,
    "prepareAppSurface(appSurface, document)",
    "bindAppElements()",
    "app surface must be prepared before DOM elements are bound.",
  );
  for (const snippet of [
    "downloadRegulatoryReport",
    "downloadEvidenceBundle",
    "Regulatory Report",
    "Regulatory Bundle ZIP",
  ]) {
    if (!medicalSurfaceSource.includes(snippet)) {
      errors.push(`medical surface source is missing ${snippet}.`);
    }
  }
}

function checkResponsiveWorkspaceCss() {
  const mediumBlock = cssBetween("@media (max-width: 1180px)", "@media (max-width: 820px)");
  const lowHeightBlock = cssBetween("@media (max-height: 760px) and (min-width: 821px)", "body.privacy-locked");
  expectSourceContains(mediumBlock, ".topbar", "medium-width topbar wrapping selector");
  expectSourceContains(mediumBlock, "flex-wrap: wrap", "medium-width topbar wrapping behavior");
  expectSourceContains(mediumBlock, ".module-run-head", "medium-width module execution head wrapping selector");
  expectSourceContains(lowHeightBlock, ".session-anchor", "low-height session sticky release selector");
  expectSourceContains(lowHeightBlock, "position: static", "low-height sticky release behavior");
  expectSourceContains(lowHeightBlock, ".compact-graph-tools", "low-height graph tools sticky release selector");
  expectSourceContains(styleSource, "body.workspace-output .bundle-action-wrap", "output bundle action wrapping selector");
  expectSourceContains(styleSource, "body.workspace-output .evidence-bundle-note", "output bundle note wrapping selector");
  expectSourceOrder(primaryStyleSource, '@import url("./research-labs.css");', '@import url("./app-shell.css");', "research styles must precede application-shell styles");
  expectSourceOrder(primaryStyleSource, '@import url("./app-shell.css");', '@import url("./device-workspace.css");', "application-shell styles must precede device-workspace styles");
  for (const asset of ["./app-shell.css", "./device-workspace.css"]) {
    const occurrences = webServiceWorkerSource.split(`"${asset}"`).length - 1;
    if (occurrences !== 2) errors.push(`${asset} must appear once in APP_ASSETS and once in APP_SHELL_ASSETS.`);
  }
}

function checkAccessStateContract() {
  expectSourceOrder(
    appSource,
    "const verified = Boolean(currentAuthUser?.email_verified || currentAuthUser?.role === \"admin\");",
    "? \"Open reports + requests\"",
    "module access state must define verified before using it for account copy.",
  );
}

function checkReportTargetBindingContract() {
  const html = readFileSync("web/index.html", "utf8");
  for (const id of ["reportTargetSelect", "reportTargetStatus", "reportTargetAnalyzeBtn"]) {
    if (!html.includes(`id="${id}"`)) errors.push(`report target binding control is missing ${id}.`);
  }
  const active = resolveReportTargetBinding({
    requestedTargetId: "rpi4_a72",
    activeTargetId: "rpi4_a72",
    cachedTargetIds: ["rpi4_a72"],
    hasArtifact: true,
  });
  const cached = resolveReportTargetBinding({
    requestedTargetId: "wasm_simd",
    activeTargetId: "rpi4_a72",
    cachedTargetIds: ["rpi4_a72", "wasm_simd"],
    hasArtifact: true,
  });
  const required = resolveReportTargetBinding({
    requestedTargetId: "x86_sse4",
    activeTargetId: "rpi4_a72",
    cachedTargetIds: ["rpi4_a72"],
    hasArtifact: true,
  });
  const unavailable = resolveReportTargetBinding({
    requestedTargetId: "x86_sse4",
    activeTargetId: "",
    cachedTargetIds: [],
    hasArtifact: false,
  });
  const artifact = resolveReportTargetBinding({
    requestedTargetId: "rpi4_a72",
    activeTargetId: "",
    cachedTargetIds: [],
    hasArtifact: true,
    artifactOnly: true,
  });
  for (const [condition, message] of [
    [active.state === "active" && active.canCopy, "active analyzed report target must be copyable"],
    [cached.state === "cached" && cached.analyzed && !cached.canCopy, "cached report target must require explicit activation before copy"],
    [required.state === "required" && !required.analyzed && !required.canCopy, "uncached report target must require analysis"],
    [unavailable.state === "unavailable" && !unavailable.canCopy, "report target must remain unavailable before an artifact audit"],
    [artifact.state === "artifact" && artifact.bindingScope === "artifact" && artifact.targetId === "" && artifact.canCopy, "metadata formats must export from an artifact-only binding without inheriting a UI target"],
    [appSource.includes("analyzeLoadedModel(currentFilename, requestedTargetId"), "top-level target changes must pass the newly selected target explicitly"],
    [appSource.includes("keepModule: preserveReportWorkspace") && appSource.includes('setActiveWorkspace("output", { force: true })'), "report-target analysis must preserve the Reports workspace"],
    [appSource.includes("reportBindingMatchesAnalysis(binding, analysis)"), "report copy must enforce target-bound or artifact-only binding semantics"],
    [appSource.includes("formatter.buildEngineeringReport(analysis, currentReportContext())"), "report copy must regenerate from the target-bound analysis object"],
    [!appSource.includes('const text = reportPreview?.textContent || "";'), "report copy must not reuse a possibly stale preview string"],
    [styleSource.includes(".report-target-bar") && styleSource.includes(".report-target-actions"), "report target controls need responsive layout rules"],
  ]) {
    if (!condition) errors.push(message);
  }
}

function attrValues(html, attr) {
  return [...html.matchAll(new RegExp(`\\b${attr}="([^"]+)"`, "g"))].map((match) => match[1]);
}

function sameSet(actual, expected) {
  return actual.length === expected.length
    && actual.every((value) => expected.includes(value));
}

function expectExactOrder(label, actual, expected) {
  if (actual.join("|") !== expected.join("|")) {
    errors.push(`${label} order mismatch: expected ${expected.join(" -> ")}, got ${actual.join(" -> ")}.`);
  }
}

function expectExactSet(label, actual, expected) {
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  for (const value of expectedSet) {
    if (!actualSet.has(value)) {
      errors.push(`${label} is missing ${value}.`);
    }
  }
  for (const value of actualSet) {
    if (!expectedSet.has(value)) {
      errors.push(`${label} has unexpected value ${value}.`);
    }
  }
}

function expectContainsAll(label, actual, expected) {
  const actualSet = new Set(actual);
  for (const value of expected) {
    if (!actualSet.has(value)) {
      errors.push(`${label} is missing ${value}.`);
    }
  }
}

function expectUnique(label, values) {
  const seen = new Set();
  for (const value of values) {
    if (seen.has(value)) {
      errors.push(`${label} contains duplicate ${value}.`);
    }
    seen.add(value);
  }
}

function expectAbsent(label, values, value) {
  if (values.includes(value)) {
    errors.push(`${label} should not include ${value}.`);
  }
}

function checkDomInjectionContract() {
  for (const [source, snippet, label] of [
    [appSource, "infoBar.innerHTML", "model explorer must not inject model-controlled op names through innerHTML"],
    [performanceVisualSource, "<td>${op.name}</td>", "performance tables must not inject model-controlled op names through innerHTML"],
    [performanceVisualSource, "tip.innerHTML = lines.map", "performance tooltips must render model-controlled text with DOM text nodes"],
    [adminSource, "${s.user_email ||", "admin structure rows must not inject account data through innerHTML"],
    [adminSource, "${run.status}</span>", "admin status values must not be injected through innerHTML"],
  ]) {
    if (source.includes(snippet)) errors.push(label);
  }
}

function cssBetween(start, end) {
  const startIndex = styleSource.indexOf(start);
  const endIndex = styleSource.indexOf(end, Math.max(0, startIndex + start.length));
  if (startIndex < 0 || endIndex < 0 || endIndex <= startIndex) return "";
  return styleSource.slice(startIndex, endIndex);
}

function expectSourceContains(source, snippet, label) {
  if (!source.includes(snippet)) {
    errors.push(`CSS is missing ${label}: ${snippet}`);
  }
}

function expectSourceOrder(source, before, after, label) {
  const beforeIndex = source.indexOf(before);
  const afterIndex = source.indexOf(after);
  if (beforeIndex < 0 || afterIndex < 0 || beforeIndex > afterIndex) {
    errors.push(label);
  }
}
