import { validateStatistics } from "./statistics.js";
import { exactCountRatio } from "./exact-moments.js";
import { requireCondition } from "./common.js";

const QUANTILES = [1, 5, 25, 50, 75, 95, 99];
const ratio = (n, d) => ({ numerator: String(n), denominator: String(d), value: exactCountRatio(n, d) });

// This projection uses the existing exact counts. It never interpolates raw
// percentiles from the deliberately coarse fixed-log2 histogram.
export function describeDistribution(statistics) {
  const s = validateStatistics(statistics), total = BigInt(s.value_count), finite = BigInt(s.finite_count);
  const zero = BigInt(s.zero_count), unsafe = BigInt(s.unsafe_integer_count);
  const h = s.histogram;
  let negative = null, positive = null;
  if (h) {
    negative = h.counts.reduce((n, c, i) => n + (h.edges[i + 1] <= 0 ? BigInt(c) : 0n), 0n);
    positive = finite - negative - zero;
    requireCondition(positive >= 0n, "histogram signs contradict zero count");
    const zeroBin = h.edges.indexOf(0);
    requireCondition(zero <= BigInt(h.counts[zeroBin]), "zero count exceeds zero histogram bin");
  }
  const quantiles = [];
  if (h && finite) {
    for (const percent of QUANTILES) {
      const rank = (BigInt(percent) * finite + 99n) / 100n;
      let before = 0n, index = 0;
      while (index < h.counts.length - 1 && before + BigInt(h.counts[index]) < rank) before += BigInt(h.counts[index++]);
      let lower = Math.max(s.minimum, h.edges[index]), upper = Math.min(s.maximum, h.edges[index + 1]);
      let upperClosed = index === h.counts.length - 1 || s.maximum < h.edges[index + 1];
      if (rank === 1n) { lower = upper = s.minimum; upperClosed = true; }
      else if (rank === finite) { lower = upper = s.maximum; upperClosed = true; }
      else if (rank > negative && rank <= negative + zero) { lower = upper = 0; upperClosed = true; }
      else if (s.minimum === s.maximum) { lower = upper = s.minimum; upperClosed = true; }
      requireCondition(lower <= upper, "quantile interval contradicts extrema");
      quantiles.push({ percent, rank: String(rank), lower, upper, upper_closed: upperClosed, exact: lower === upper && upperClosed });
    }
  }
  return {
    finite_fraction: ratio(finite, total), nonfinite_fraction: ratio(total - finite, total),
    zero_fraction_of_finite: ratio(zero, finite),
    sign_counts: { negative: negative?.toString() ?? null, zero: String(zero), positive: positive?.toString() ?? null },
    constant_finite_value: finite && !unsafe && s.minimum === s.maximum ? s.minimum : null,
    all_values_zero: total ? zero === total : null,
    all_values_finite: total ? total === finite : null,
    quantiles: { method: "finite_nearest_rank_bounds_from_fixed_bins_v1", status: quantiles.length ? "assessed" : "not_assessed", reason: quantiles.length ? null : unsafe ? "unsafe_integer_histogram_unavailable" : "no_finite_values", intervals: quantiles },
  };
}

// Total variation is computed as an exact rational on normalized histogram
// counts, then rounded once. Different sample counts remain comparable.
export function compareDistributions(baseline, candidate) {
  validateStatistics(baseline); validateStatistics(candidate);
  if (!baseline.histogram || !candidate.histogram) return { status: "not_assessed", reason: "histogram_unavailable" };
  const a = BigInt(baseline.finite_count), b = BigInt(candidate.finite_count);
  if (!a || !b) return { status: "not_assessed", reason: "no_finite_values" };
  requireCondition(JSON.stringify(baseline.histogram.edges) === JSON.stringify(candidate.histogram.edges), "histogram comparison edges differ");
  let distance = 0n;
  for (let i = 0; i < baseline.histogram.counts.length; i++) {
    const delta = BigInt(baseline.histogram.counts[i]) * b - BigInt(candidate.histogram.counts[i]) * a;
    distance += delta < 0n ? -delta : delta;
  }
  const delta = key => {
    if (baseline[key] === null || candidate[key] === null) return null;
    const n = candidate[key] - baseline[key];
    return Number.isFinite(n) ? n : null;
  };
  return {
    status: "assessed", reason: null,
    histogram_total_variation: ratio(distance, 2n * a * b),
    baseline_finite_count: String(a), candidate_finite_count: String(b),
    mean_delta: delta("mean"), rms_delta: delta("rms"), population_stddev_delta: delta("population_stddev"),
    boundary: "Finite-value histogram mass only; differences inside the same bin are unresolved. Zero distance does not establish equal values. Metric deltas are candidate minus baseline; null means unavailable or overflow. Not a paired-value error or a causal effect.",
  };
}
