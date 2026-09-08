import { normalizeAnalysisSummaryContract } from "./analysis-summary-contract.js";

export function normalizeTfliteAnalysisContract(analysis) {
  if (!analysis || analysis.format !== "tflite") return analysis;
  for (const op of Array.isArray(analysis.ops) ? analysis.ops : []) {
    const macs = op?.macs;
    if (macs === undefined || macs === null || !Number.isFinite(Number(macs)) || Number(macs) < 0) {
      op.macs = null;
      if (!op.macs_status) op.macs_status = "not_assessed";
      if (!op.macs_reason) {
        op.macs_reason = "The serialized operator contract does not close a finite non-negative nominal MAC value.";
      }
    }
  }
  return normalizeAnalysisSummaryContract(analysis);
}
