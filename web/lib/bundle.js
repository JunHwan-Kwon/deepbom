import { jsonForDownload, zipTextFile } from "./report-utils.js";

const BUNDLE_PROGRESS_STATUSES = new Set(["queued", "running", "done", "failed", "skipped"]);

export const REGULATORY_BUNDLE_MODULE_SPECS = [
  {
    capability: "deepbom",
    id: "deepbom",
    workspace: "deepbom",
    label: "DEEPBOM",
    path: "research/deepbom.json",
    detail: "Deploy-artifact weight/topology proxy JSON.",
    queuedDetail: "Waiting to run DEEPBOM locally.",
  },
  {
    capability: "perturbation",
    id: "perturbation",
    workspace: "deepbom",
    label: "Perturbation",
    path: "research/perturbation_analysis.json",
    detail: "Research-stage input/output drift plus artifact-supported weight and layer sensitivity JSON.",
    queuedDetail: "Waiting to run local perturbation probes.",
  },
  {
    capability: "runtime_basin",
    id: "runtime_basin",
    workspace: "runtime_basin",
    label: "Backend Consistency",
    path: "research/runtime_basin.json",
    detail: "Research-stage local backend availability and output-drift JSON.",
    queuedDetail: "Waiting to compare detected browser runtime paths.",
  },
  {
    capability: "deployment_sensitivity",
    id: "deploy_curvature",
    workspace: "deployment_sensitivity",
    label: "Deployment Sensitivity Proxy",
    path: "research/deploy_curvature_basin.json",
    detail: "Research-stage finite-difference deployment-function sensitivity proxy JSON.",
    queuedDetail: "Waiting to run the deployment-sensitivity proxy.",
  },
];

export function engineeringBundleItems(hasAnalysis) {
  return [
    {
      id: "engineering_static",
      label: "Engineering audit package",
      detail: "Compact package: one Engineering Report and one conformance-checked evidence JSON containing static analysis, normalized execution-placement evidence, CSV/graph sources, ML-BOM, and technical evidence; a raw constructive input tensor is added only when certified.",
      enabled: Boolean(hasAnalysis),
      status: hasAnalysis ? "included" : "run audit first",
    },
  ];
}

export function regulatoryBundleItems(capabilities, hasAnalysis) {
  return [
    {
      id: "engineering_bundle",
      label: "Engineering Bundle",
      detail: "The complete Engineering Bundle is included first.",
      enabled: Boolean(hasAnalysis),
      status: hasAnalysis ? "included" : "run audit first",
    },
    {
      id: "regulatory_report",
      label: "Regulatory Report",
      detail: "Regulatory-facing report with evidence classes, findings register, integrity posture, and Engineering Report appendix.",
      enabled: Boolean(hasAnalysis && capabilities.regulatory_report),
      status: capabilities.regulatory_report ? "included" : "locked",
    },
    ...REGULATORY_BUNDLE_MODULE_SPECS.map((item) => ({
      id: item.id,
      label: item.label,
      detail: item.detail,
      enabled: Boolean(hasAnalysis && capabilities[item.capability]),
      status: capabilities[item.capability] ? "will run" : "locked",
    })),
  ];
}

export function initialEvidenceBundleProgress(capabilities) {
  const progress = {
    engineering_bundle: {
      status: "queued",
      detail: "Waiting to package the complete Engineering Bundle.",
    },
    regulatory_report: {
      status: "queued",
      detail: "Waiting to add the Regulatory Report.",
    },
  };
  for (const item of REGULATORY_BUNDLE_MODULE_SPECS) {
    progress[item.id] = capabilities[item.capability]
      ? { status: "queued", detail: item.queuedDetail }
      : { status: "locked", detail: "Not included for this account." };
  }
  return progress;
}

export function applyBundleProgress(item, progress) {
  if (!progress) return item;
  return {
    ...item,
    detail: progress.detail || item.detail,
    enabled: item.enabled || BUNDLE_PROGRESS_STATUSES.has(progress.status),
    status: progress.status,
  };
}

export function renderBundleScope(scopeNode, items, progressState) {
  if (!scopeNode) return;
  scopeNode.replaceChildren(...items.map((item) => bundleScopeItem(applyBundleProgress(item, progressState?.[item.id]))));
}

function bundleScopeItem(item) {
  const li = document.createElement("li");
  const statusClass = bundleStatusClass(item.status);
  li.className = `${item.enabled ? "enabled" : "locked"} status-${statusClass}`;
  li.dataset.bundleScope = item.id || "";
  const text = document.createElement("span");
  const strong = document.createElement("strong");
  strong.textContent = item.label;
  const small = document.createElement("small");
  small.textContent = item.detail;
  text.append(strong, small);
  const badge = document.createElement("em");
  badge.textContent = item.status;
  li.append(text, badge);
  return li;
}

export function bundleStatusClass(status = "") {
  return String(status)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "idle";
}

export function skippedBundleModule(id, label, reason) {
  return {
    id,
    label,
    status: "skipped",
    included: false,
    reason,
    generated_at: new Date().toISOString(),
  };
}

export function bundleStepProgressForResult(label, result) {
  const resultStatus = result?.status || "complete";
  const error = result?.error || result?.reason || "";
  const progressStatus = resultStatus === "failed"
    ? "failed"
    : resultStatus === "blocked"
      ? "skipped"
      : result
        ? "done"
        : "skipped";
  const detail = resultStatus === "failed"
    ? `${label} generated a failure report: ${error || "see module JSON"}`
    : resultStatus === "blocked"
      ? `${label} was blocked and the status report is included.`
      : result
        ? `${label} result is included in the bundle.`
        : `${label} completed without an export payload.`;
  return { progressStatus, detail, resultStatus, error };
}

export function bundleStepLogItem({
  id,
  label,
  status,
  included,
  error = "",
  generatedAt = new Date().toISOString(),
}) {
  const item = {
    id,
    label,
    status,
    included: Boolean(included),
    generated_at: generatedAt,
  };
  if (error) item.error = error;
  return item;
}

export function buildBundleEnvelope(files, {
  buildSummary,
  buildManifest,
  includeSummary = true,
} = {}) {
  const packaged = [...(files || [])];
  if (includeSummary) {
    packaged.unshift({
      name: "summary.md",
      data: String(buildSummary?.(packaged) ?? ""),
    });
  }
  packaged.push({
    name: "manifest.json",
    data: String(buildManifest?.(packaged) ?? ""),
  });
  return packaged;
}

export async function executeBundleStep({
  id,
  label,
  runner,
  setProgress,
  setStatus,
  beforeRun,
  formatError = (error) => error?.message || String(error || "Unknown error"),
}) {
  setProgress(id, "running", `${label} is running locally in this browser.`);
  setStatus(`Running ${label}`);
  await beforeRun?.();
  try {
    const result = await runner();
    const progress = bundleStepProgressForResult(label, result);
    setProgress(id, progress.progressStatus, progress.detail);
    return {
      result,
      logItem: bundleStepLogItem({
        id,
        label,
        status: progress.resultStatus,
        included: Boolean(result),
        error: progress.error,
      }),
    };
  } catch (error) {
    const message = formatError(error);
    setProgress(id, "failed", message);
    return {
      result: null,
      error,
      logItem: bundleStepLogItem({
        id,
        label,
        status: "failed",
        included: false,
        error: message,
      }),
    };
  }
}

export async function appendRegulatoryBundleModule({
  files,
  moduleLog,
  capabilities,
  bundleModule,
  runners,
  setProgress,
  setStatus,
  beforeRun,
  formatError,
  onError,
}) {
  if (!capabilities[bundleModule.capability]) {
    setProgress(bundleModule.id, "skipped", "Skipped because this account does not include this module.");
    moduleLog.push(skippedBundleModule(bundleModule.id, bundleModule.label, "capability_not_enabled"));
    return null;
  }
  const runner = runners[bundleModule.id];
  if (!runner) throw new Error(`No regulatory bundle runner configured for ${bundleModule.id}`);
  const { result, logItem, error } = await executeBundleStep({
    id: bundleModule.id,
    label: bundleModule.label,
    runner,
    setProgress,
    setStatus,
    beforeRun,
    formatError,
  });
  moduleLog.push(logItem);
  if (error) onError?.(error, bundleModule);
  if (result) files.push(zipTextFile(bundleModule.path, jsonForDownload(result)));
  return result;
}
