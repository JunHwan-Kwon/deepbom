// computeInsights has been migrated to Rust/WASM.
// Results are now in analysis.insights (computed by compute_insights() in src/lib.rs).
// app.js uses adaptInsightsForUI(analysis) to map those fields to the UI shape.
// This file is intentionally empty — kept to avoid breaking any stale import.
export function computeInsights(_analysis, _fallbackTarget = {}) {
  return null;
}
