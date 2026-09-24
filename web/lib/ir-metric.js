import { canonicalJson } from "./report-utils.js";
import { exactInteger, validateExactInteger } from "./exact-integer.js";

// Counts and sums use decimal integers. Null is never a zero contribution to a complete total.
export function integerMetric({ value, unit, sourceRefs = [], assessed = 0, eligible = 0, reason = null, assumptions = [] }) {
  validateExactInteger(value);
  const metric = { schema: "deepbom.metric.v1", unit, value, exactness: value === null ? "not_assessable" : "exact_integer",
    denominator: null, coverage: { assessed, eligible }, source_refs: sourceRefs, assumptions,
    reason: value === null ? reason || "source_did_not_establish_a_complete_value" : null, evidence_class: "DERIVED" };
  validateMetric(metric); return metric;
}

export function aggregateIntegerMetrics(rows, unit) {
  rows.forEach(validateMetric);
  if (rows.some(row => row.unit !== unit || row.denominator !== null)) throw new Error("IR metric aggregation mixes units or ratios.");
  const refs = rows.flatMap(row => row.source_refs);
  if (new Set(refs).size !== refs.length) throw new Error("IR metric aggregation would count a source twice.");
  const assessed = rows.filter(row => row.value !== null).length;
  const subtotal = exactInteger(rows.reduce((sum, row) => sum + BigInt(row.value?.decimal || "0"), 0n));
  return { total: integerMetric({ value: assessed === rows.length ? subtotal : null, unit, sourceRefs: refs, assessed, eligible: rows.length }), assessed_subtotal: subtotal };
}

export function validateMetric(row) {
  if (row?.schema !== "deepbom.metric.v1" || !["MAC", "byte", "element", "count"].includes(row.unit) || row.denominator !== null) throw new Error("IR metric unit or denominator is invalid.");
  validateExactInteger(row.value);
  if (![row.coverage?.assessed, row.coverage?.eligible].every(n => Number.isSafeInteger(n) && n >= 0) || row.coverage.assessed > row.coverage.eligible) throw new Error("IR metric coverage is invalid.");
  if (row.exactness !== (row.value === null ? "not_assessable" : "exact_integer") || (row.value === null && !row.reason)) throw new Error("IR metric exactness is inconsistent.");
  if (!Array.isArray(row.source_refs) || row.source_refs.some(ref => typeof ref !== "string" || !ref) || new Set(row.source_refs).size !== row.source_refs.length) throw new Error("IR metric source references are invalid.");
  canonicalJson(row); return row;
}
