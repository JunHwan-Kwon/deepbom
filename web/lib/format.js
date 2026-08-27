export function formatShapes(shapes) {
  if (!Array.isArray(shapes) || !shapes.length) return "-";
  return shapes.map((shape) => `[${shape.join(",")}]`).join(" ");
}

export function shapeText(shape) {
  return Array.isArray(shape) && shape.length ? `[${shape.join("x")}]` : "[]";
}

export function tensorShapeText(tensor) {
  const staticShape = shapeText(tensor?.shape);
  const signature = tensor?.shape_signature;
  if (!Array.isArray(signature) || !signature.length) return staticShape;
  const signatureText = shapeText(signature);
  return signatureText === staticShape ? staticShape : `${staticShape} sig=${signatureText}`;
}

export function formatNumber(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Number(value || 0));
}

export function formatExactInteger(decimalValue, numericValue = null, unavailable = "Not assessed") {
  const decimal = decimalValue == null ? "" : String(decimalValue).trim();
  if (/^-?(?:0|[1-9]\d*)$/.test(decimal) && decimal !== "-0") {
    try {
      return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(BigInt(decimal));
    } catch {
      // Fall through to the safe numeric mirror.
    }
  }
  const numeric = Number(numericValue);
  return Number.isSafeInteger(numeric) ? formatNumber(numeric) : unavailable;
}

export function formatScientific(value) {
  const number = Number(value || 0);
  if (!number) return "0";
  return number.toExponential(2);
}

export function formatPercent(value) {
  const scaled = Number(value || 0) * 100;
  const normalized = Object.is(Math.round(scaled * 10) / 10, -0) ? 0 : scaled;
  if (normalized !== 0 && Math.abs(normalized) < 0.0001) return `${normalized.toExponential(2)}%`;
  const maximumFractionDigits = normalized !== 0 && Math.abs(normalized) < 0.1 ? 4 : 1;
  return `${new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(normalized)}%`;
}

export function formatPercentRange(minimum, maximum) {
  const low = Number(minimum);
  const high = Number(maximum);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return "Not assessed";
  const lowText = formatPercent(low);
  const highText = formatPercent(high);
  return Math.abs(high - low) <= 1e-12 || lowText === highText ? lowText : `${lowText} to ${highText}`;
}

export function formatPercent1(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export function score100(value) {
  return `${(Number(value || 0) * 100).toFixed(1)} / 100`;
}

export function formatDateTime(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function humanStatusLabel(status) {
  return String(status || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function humanizeStageKey(value) {
  const key = String(value || "").trim();
  const spatialResolution = key.match(/^spatial\/(\d+)x(\d+)$/i);
  if (spatialResolution) {
    const area = Number(spatialResolution[1]) * Number(spatialResolution[2]);
    const scale = area >= 100_000 ? "High-resolution" : area >= 16_384 ? "Mid-resolution" : "Compact";
    return `${scale} spatial stage (${spatialResolution[1]}x${spatialResolution[2]}; channel range shown separately)`;
  }
  const spatial = key.match(/^(\d+)x(\d+)xCbucket<=(\d+)$/i);
  if (spatial) {
    const area = Number(spatial[1]) * Number(spatial[2]);
    const scale = area >= 100_000 ? "High-resolution" : area >= 16_384 ? "Mid-resolution" : "Compact";
    return `${scale} spatial stage (${spatial[1]}x${spatial[2]}, channels <=${spatial[3]})`;
  }
  const sequence = key.match(/^seq\/(.+)$/i);
  if (sequence) return `Sequence stage (length ${sequence[1]})`;
  if (key === "vector/head") return "Vector or prediction head";
  if (key === "shape/vector") return "Shape or vector operations";
  if (key === "scalar/shape") return "Scalar or shape-control operations";
  if (key === "other") return "Other tensor-rank stage";
  return key || "Unclassified stage";
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) {
    return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(bytes / (1024 * 1024))} MiB`;
  }
  if (bytes >= 1024) {
    return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(bytes / 1024)} KiB`;
  }
  return `${formatNumber(bytes)} B`;
}

export function formatUs(value) {
  if (value == null || value === "") return "Not assessed";
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return "Not assessed";
  if (number >= 1000) return `${(number / 1000).toFixed(2)} ms`;
  if (number >= 10) return `${number.toFixed(1)} us`;
  return `${number.toFixed(2)} us`;
}

export function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, Number(value || 0)));
}

export function clampInteger(value, min, max) {
  const rounded = Math.round(Number(value || 0));
  return Math.min(max, Math.max(min, rounded));
}

export function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

export function cloneTypedArray(value) {
  return new value.constructor(value);
}

export function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

export function maxNumber(values) {
  return values.reduce((max, value) => Math.max(max, Number(value || 0)), 0);
}

export function maxBy(items, picker) {
  let best = null;
  let bestValue = -Infinity;
  for (const item of items) {
    const value = Number(picker(item) || 0);
    if (value > bestValue) {
      best = item;
      bestValue = value;
    }
  }
  return best;
}

export function sumCountItems(items) {
  if (!Array.isArray(items)) return 0;
  return items.reduce((sum, item) => sum + Number(item.count || 0), 0);
}

export function sumNumbers(values) {
  return values.reduce((sum, value) => sum + Number(value || 0), 0);
}

export function countByArray(values) {
  const counts = new Map();
  for (const value of values || []) counts.set(value, (counts.get(value) || 0) + 1);
  return [...counts.entries()].sort(([a], [b]) => String(a).localeCompare(String(b))).map(([name, count]) => ({ name, count }));
}

export function argmax(values) {
  if (!values?.length) return -1;
  let best = 0;
  for (let index = 1; index < values.length; index++) {
    if (Number(values[index]) > Number(values[best])) {
      best = index;
    }
  }
  return best;
}

export function latencyStats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) {
    return { min: 0, max: 0, p50: 0, p90: 0, p95: 0, p99: 0, mean: 0, stddev: 0, cv: 0 };
  }
  const mean = sorted.reduce((acc, value) => acc + value, 0) / sorted.length;
  const variance = sorted.reduce((acc, value) => acc + (value - mean) ** 2, 0) / sorted.length;
  const stddev = Math.sqrt(variance);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    p50: percentile(sorted, 0.5),
    p90: percentile(sorted, 0.9),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    mean,
    stddev,
    cv: mean ? stddev / mean : 0,
  };
}

function percentile(sorted, q) {
  if (!sorted.length) return 0;
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * q) - 1);
  return sorted[index];
}

// Sliding window mean for smoothing a run-time series
export function movingAverage(values, window = 5) {
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(values.length, i + half + 1);
    const slice = values.slice(start, end);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

// Noise analysis on a raw timings array
export function benchmarkNoise(timings) {
  const n = timings.length;
  if (n < 3) {
    return { outlierCount: 0, gcSpikeCount: 0, trendSlope: 0, trendLabel: "—", trimmedP50: timings[0] ?? 0, trimmedMean: timings[0] ?? 0 };
  }
  const mean = timings.reduce((a, b) => a + b, 0) / n;
  const variance = timings.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const stddev = Math.sqrt(variance);
  const limit = mean + 2.5 * stddev;

  // Any run above 2.5σ
  const outlierCount = timings.filter(v => v > limit).length;

  // Isolated spike: outlier whose immediate neighbors are both normal
  const gcSpikeCount = timings.reduce((count, v, i) => {
    if (v <= limit) return count;
    const prevNormal = i === 0 || timings[i - 1] <= limit;
    const nextNormal = i === n - 1 || timings[i + 1] <= limit;
    return count + (prevNormal && nextNormal ? 1 : 0);
  }, 0);

  // Linear regression slope (ms per run) — positive slope = getting slower
  const xMean = (n - 1) / 2;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (timings[i] - mean);
    den += (i - xMean) ** 2;
  }
  const trendSlope = den > 0 ? num / den : 0;

  let trendLabel;
  if (n < 10) {
    trendLabel = "—";
  } else if (Math.abs(trendSlope) < 0.03) {
    trendLabel = "stable";
  } else if (trendSlope > 0) {
    trendLabel = `↑${trendSlope.toFixed(2)}ms/run`;
  } else {
    trendLabel = `↓${Math.abs(trendSlope).toFixed(2)}ms/run`;
  }

  // Trimmed p50: remove top+bottom 10% of runs, then compute median
  const sorted = [...timings].sort((a, b) => a - b);
  const trimCount = Math.max(1, Math.floor(n * 0.1));
  const trimmed = n > trimCount * 2 + 1 ? sorted.slice(trimCount, n - trimCount) : sorted;
  const trimmedMean = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  const trimmedP50 = trimmed[Math.floor(trimmed.length / 2)] ?? 0;

  return { outlierCount, gcSpikeCount, trendSlope, trendLabel, trimmedP50, trimmedMean };
}

export function compactText(value, maxLength) {
  const text = String(value || "");
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

export function shortError(error) {
  const message = error?.message || String(error);
  return message.length > 220 ? `${message.slice(0, 220)}...` : message;
}

export function padOp(index) {
  return String(index).padStart(3, "0");
}

export function safeStem(filename) {
  return filename.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "_");
}

export function formatDrift(value) {
  const number = Number(value || 0);
  if (Math.abs(number) < 0.001 && number !== 0) return number.toExponential(2);
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 }).format(number);
}

export function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
