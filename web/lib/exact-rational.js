export const EXACT_RATIO_DECIMAL_DIGITS = 15;

function nonnegativeInteger(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  const text = value == null ? "" : String(value);
  return /^(?:0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
}

export function exactNonnegativeRatio(numeratorValue, denominatorValue, digits = EXACT_RATIO_DECIMAL_DIGITS) {
  const numerator = nonnegativeInteger(numeratorValue);
  const denominator = nonnegativeInteger(denominatorValue);
  if (numerator == null || denominator == null || denominator === 0n
    || !Number.isSafeInteger(digits) || digits < 0 || digits > 15) return null;
  const scale = 10n ** BigInt(digits);
  const scaled = (numerator * scale + denominator / 2n) / denominator;
  const value = Number(scaled) / (10 ** digits);
  return Number.isFinite(value) ? value : null;
}
