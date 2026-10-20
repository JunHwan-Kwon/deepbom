import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
const irContract = (operation, input) => portableContract("ir_semantics", canonicalJsonInput({ operation, ...input }));

// Counts and sums have one WASM owner; partial totals remain unknown.
export function integerMetric({ value, unit, sourceRefs = [], assessed = 0, eligible = 0, reason = null, assumptions = [] }) {
  return irContract("new_metric", {value, unit, source_refs:sourceRefs, assessed, eligible, reason, assumptions});
}
export function aggregateIntegerMetrics(rows, unit) { return irContract("aggregate_metrics", {rows,unit}); }
export function validateMetric(value) { irContract("metric", {value}); return value; }
