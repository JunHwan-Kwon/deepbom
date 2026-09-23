import { decimalCount, exactKeys, parseNumeric, requireCondition } from "./common.js";
import { ExactMoments } from "./exact-moments.js";

// Fixed log2 bins permit exact-count streaming and merging without a second
// payload read. The bin definition is identical for every format and tensor.
const EXPONENTS = [-1074, -32, -24, -16, -12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024];
export const HISTOGRAM_EDGES = Object.freeze([-Number.MAX_VALUE, ...EXPONENTS.slice(0, -1).map(e => -(2 ** e)).reverse(), 0, ...EXPONENTS.slice(0, -1).map(e => 2 ** e), Number.MAX_VALUE]);
export class TensorStatistics {
  constructor() {
    this.count = 0n; this.finite = 0n; this.nan = 0n; this.positiveInf = 0n; this.negativeInf = 0n; this.zero = 0n; this.negativeZero = 0n;
    this.min = null; this.max = null; this.moments = new ExactMoments(); this.unsafeInteger = 0n;
    this.integerMin = null; this.integerMax = null;
    this.bins = HISTOGRAM_EDGES.slice(1).map(() => 0n);
  }
  add(raw) {
    let value = parseNumeric(raw); this.count++;
    if (typeof value === "bigint") {
      this.integerMin = this.integerMin == null || value < this.integerMin ? value : this.integerMin;
      this.integerMax = this.integerMax == null || value > this.integerMax ? value : this.integerMax;
      if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) { this.unsafeInteger++; return; }
      value = Number(value);
    }
    if (Number.isNaN(value)) { this.nan++; return; }
    if (value === Infinity) { this.positiveInf++; return; }
    if (value === -Infinity) { this.negativeInf++; return; }
    this.finite++;
    requireCondition(this.finite <= BigInt(Number.MAX_SAFE_INTEGER), "floating statistics exceed exact count budget");
    this.min = this.min == null ? value : Math.min(this.min, value); this.max = this.max == null ? value : Math.max(this.max, value);
    if (value === 0) { this.zero++; if (Object.is(value, -0)) this.negativeZero++; }
    this.moments.add(value);
    let lo = 0, hi = this.bins.length - 1;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (value < HISTOGRAM_EDGES[mid + 1]) hi = mid; else lo = mid + 1; }
    this.bins[lo]++;
  }
  finish() {
    const complete = this.unsafeInteger === 0n;
    const rounded = value => Number.isFinite(value) ? value : null;
    const n = Number(this.finite);
    const moments = this.moments.finish(n);
    const mean = rounded(moments.mean), std = rounded(moments.std), rms = rounded(moments.rms), l2 = rounded(moments.l2);
    return {
      value_count: String(this.count), finite_count: String(this.finite + this.unsafeInteger), nonfinite: { nan: String(this.nan), positive_infinity: String(this.positiveInf), negative_infinity: String(this.negativeInf) },
      zero_count: String(this.zero), negative_zero_count: String(this.negativeZero),
      precision: complete ? "binary64_rounded_statistics_exact_counts" : "not_assessed_unsafe_integer_arithmetic",
      unsafe_integer_count: String(this.unsafeInteger), integer_minimum: this.integerMin?.toString() ?? null, integer_maximum: this.integerMax?.toString() ?? null,
      minimum: complete ? this.min : null, maximum: complete ? this.max : null, mean: complete ? mean : null, population_stddev: complete ? std : null, rms: complete ? rms : null, l2_norm: complete ? l2 : null,
      overflowed_metrics: complete && n ? [["mean",mean],["population_stddev",std],["rms",rms],["l2_norm",l2]].filter(([,v])=>v==null).map(([k])=>k) : [],
      histogram: complete ? { method: "fixed_log2_edges_v1", interval: "left_closed_right_open_last_closed", edges: [...HISTOGRAM_EDGES], counts: this.bins.map(String) } : null,
    };
  }
}
export function statisticsOf(values) { const s = new TensorStatistics(); for (const value of values) s.add(value); return s.finish(); }
export function validateStatistics(s) {
  exactKeys(s, ["value_count","finite_count","nonfinite","zero_count","negative_zero_count","precision","unsafe_integer_count","integer_minimum","integer_maximum","minimum","maximum","mean","population_stddev","rms","l2_norm","overflowed_metrics","histogram"], "statistics");
  exactKeys(s.nonfinite, ["nan","positive_infinity","negative_infinity"], "nonfinite counts");
  const total = decimalCount(s.value_count), finite = decimalCount(s.finite_count);
  requireCondition(total === finite + Object.values(s.nonfinite).reduce((a,b)=>a+decimalCount(b),0n), "value counts do not conserve");
  requireCondition(decimalCount(s.negative_zero_count) <= decimalCount(s.zero_count) && decimalCount(s.zero_count) <= finite, "zero counts do not conserve");
  const unsafe = decimalCount(s.unsafe_integer_count);
  requireCondition(unsafe <= finite, "unsafe integer count exceeds finite count");
  requireCondition(s.precision === (unsafe ? "not_assessed_unsafe_integer_arithmetic" : "binary64_rounded_statistics_exact_counts"), "precision status contradicts counts");
  for (const key of ["minimum","maximum","mean","population_stddev","rms","l2_norm"]) requireCondition(s[key] === null || (typeof s[key] === "number" && Number.isFinite(s[key])), `invalid ${key}`);
  requireCondition(Array.isArray(s.overflowed_metrics) && s.overflowed_metrics.every(k=>["mean","population_stddev","rms","l2_norm"].includes(k) && s[k]===null), "invalid overflow ledger");
  if (unsafe || !finite) for (const key of ["minimum","maximum","mean","population_stddev","rms","l2_norm"]) requireCondition(s[key]===null, "unassessed statistic is not null");
  requireCondition(decimalCount(s.zero_count) <= finite - unsafe, "zero count includes undecoded unsafe integers");
  const integerToken = value => typeof value === "string" && /^-?(0|[1-9][0-9]*)$/.test(value) && value !== "-0";
  requireCondition((s.integer_minimum===null && s.integer_maximum===null && !unsafe) || (integerToken(s.integer_minimum) && integerToken(s.integer_maximum) && BigInt(s.integer_minimum)<=BigInt(s.integer_maximum)), "invalid exact integer extrema");
  requireCondition(new Set(s.overflowed_metrics).size===s.overflowed_metrics.length,"duplicate overflow metric");
  if(!unsafe && finite) {
    requireCondition(s.minimum!==null && s.maximum!==null && s.minimum<=s.maximum,"invalid numeric extrema");
    for(const key of ["mean","population_stddev","rms","l2_norm"]) requireCondition(s[key]!==null || s.overflowed_metrics.includes(key),"missing metric without overflow reason");
    for(const key of ["population_stddev","rms","l2_norm"]) requireCondition(s[key]===null || s[key]>=0,"negative magnitude statistic");
  }
  if (s.histogram) {
    exactKeys(s.histogram,["method","interval","edges","counts"],"histogram");
    requireCondition(!unsafe && s.histogram.method === "fixed_log2_edges_v1" && s.histogram.interval === "left_closed_right_open_last_closed", "histogram method mismatch");
    requireCondition(JSON.stringify(s.histogram.edges)===JSON.stringify(HISTOGRAM_EDGES), "histogram edges mismatch");
    requireCondition(s.histogram.counts.length===HISTOGRAM_EDGES.length-1 && s.histogram.counts.reduce((a,b)=>a+decimalCount(b),0n)===finite,"histogram count conservation failed");
  } else requireCondition(unsafe>0n,"missing histogram");
  return s;
}
