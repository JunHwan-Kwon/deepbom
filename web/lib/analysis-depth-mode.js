const LEGACY_STORAGE_KEY = "deepbom.analysis-depth.v1";

export function createAnalysisDepthMode({
  doc = document,
} = {}) {
  const root = doc.documentElement;
  root.dataset.analysisDepth = "deep";
  for (const indicator of doc.querySelectorAll('[data-analysis-depth-mode="deep"]')) {
    indicator.setAttribute("aria-label", "Full analysis surface active");
  }
  for (const option of doc.querySelectorAll('#mobileAuditView option[data-analysis-depth="deep"]')) {
    option.disabled = option.hidden;
  }
  try { sessionStorage.removeItem(LEGACY_STORAGE_KEY); } catch { /* Legacy preference cleanup is optional. */ }

  return {
    get mode() { return "deep"; },
    setMode() {
      root.dataset.analysisDepth = "deep";
      return "deep";
    },
  };
}
