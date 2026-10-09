import { histogramSvg, escapeXml } from "./weight-visuals.js";
import { renderWeightAnalysisView } from "./weight-analysis-view.js";
import { validateWeightAnalysis } from "./weight-analysis.js";

const palette = Object.freeze({ surface: "#ffffff", ink: "#182d29", muted: "#53625f", line: "#d2ddd9", accent: "#216b59", blue: "#426eac" });
// Render the complete in-browser tensor feature, never the bounded/truncated
// conversation projection. Use the same visual encoders as the web workbench.
export function buildWeightQueryVisual(evidence, ref, view = "distribution", model) {
  if (!evidence) return null;
  validateWeightAnalysis(evidence.weight_analysis, evidence.weight_ir, model);
  const row = evidence?.weight_analysis.tensors.find(item => item.weight_ref === ref);
  if (!row || row.status !== "assessed") return null;
  const plot = view === "distribution" ? histogramSvg(row, palette) : renderWeightAnalysisView(view, row, null, palette).svg;
  if (!plot) return null;
  const source = evidence.weight_analysis.source;
  const height = view === "distribution" ? 298 : 340;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="${height + 155}" viewBox="0 0 1000 ${height + 155}" role="img" aria-label="Weight ${escapeXml(view)}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#182d29"><text x="18" y="26" font-size="18">DEEPBOM · ${escapeXml(view)} · ${escapeXml(String(row.name || ref).slice(0, 65))}</text><text x="18" y="50" font-size="12">Artifact SHA-256: ${source.artifact_sha256}</text><text x="18" y="69" font-size="12">Weight analysis: ${evidence.weight_analysis.weight_analysis_sha256}</text><text x="18" y="89" font-size="12">${escapeXml(ref)} · ${escapeXml(row.representation)}</text></g><g transform="translate(120 103)">${plot.replace('<svg ', '<svg width="760" height="' + height + '" ')}</g><text x="18" y="${height + 138}" font-family="Arial,sans-serif" font-size="12">Payload evidence; feature coverage and axis contracts apply. Not an activation or measured optimization gain.</text></svg>`;
  return { svg, render_model: { page: { width_mm: 210, height_mm: (height + 155) / 1000 * 210 } } };
}
