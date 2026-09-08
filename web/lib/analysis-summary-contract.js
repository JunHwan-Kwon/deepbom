import { coverageComplete, deriveMacCoverage } from "./mac-coverage.js";

export const MAC_CONFIDENCE_VALUES = Object.freeze([
  "exact",
  "symbolic",
  "partial",
  "not_applicable",
]);

const WEIGHT_CONTAINERS = new Set(["gguf", "safetensors"]);

export function normalizeAnalysisSummaryContract(analysis) {
  if (!analysis || typeof analysis !== "object" || Array.isArray(analysis)) return analysis;
  if (analysis.total_macs === undefined) analysis.total_macs = null;
  if (analysis.total_macs_decimal === undefined) analysis.total_macs_decimal = null;
  analysis.mac_confidence = deriveMacConfidence(analysis);
  return analysis;
}

export function deriveMacConfidence(analysis = {}) {
  const explicit = String(analysis?.mac_confidence || analysis?.mac_assessment?.confidence || "")
    .trim().toLowerCase().replaceAll("-", "_");
  if (MAC_CONFIDENCE_VALUES.includes(explicit)) return explicit;

  const format = String(analysis?.format || "unknown").toLowerCase();
  if (WEIGHT_CONTAINERS.has(format)) return "not_applicable";

  const coverage = deriveMacCoverage(analysis);
  const status = String(coverage.status || "").toLowerCase();
  if (status.includes("not_applicable")) return "not_applicable";
  if (coverage.compute_ops === 0 && coverage.assessed_compute_ops === 0) return "not_applicable";
  if (coverageComplete(coverage) && finite(analysis.total_macs) != null) return "exact";

  const formulaStatus = String(
    analysis?.dynamic_shape_cost_contract?.total_macs_formula_status
      || analysis?.mac_assessment?.formula_status
      || "",
  ).toLowerCase();
  const formula = analysis?.dynamic_shape_cost_contract?.total_macs_formula
    ?? analysis?.mac_assessment?.total_macs_formula
    ?? null;
  if (formula && /exact_symbolic|symbolic_exact/.test(formulaStatus)) return "symbolic";
  return "partial";
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
