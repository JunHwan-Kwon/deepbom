export function staticValuesWithSignedZeros(tensor) {
  if (tensor?.static_values_complete !== true || !Array.isArray(tensor.static_values)) return null;
  const indices = tensor.static_values_negative_zero_indices;
  const count = tensor.static_values_negative_zero_count;
  if (!Array.isArray(indices) || !Number.isSafeInteger(count) || count < 0 || count !== indices.length) return null;
  if (tensor.static_values.some((value) => Object.is(value, -0) || !Number.isFinite(value))) return null;
  const values = [...tensor.static_values];
  const unique = new Set();
  for (const index of indices) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= values.length || unique.has(index) || values[index] !== 0) return null;
    unique.add(index);
    values[index] = -0;
  }
  return values;
}

export function numericArraysExactlyEqual(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

export function canonicalFloatText(value) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

export function parseCanonicalFloatText(text, { float32 = false } = {}) {
  if (typeof text !== "string" || !text.length) return { ok: false, value: null };
  let value;
  if (text === "NaN") value = Number.NaN;
  else if (text === "Infinity") value = Number.POSITIVE_INFINITY;
  else if (text === "-Infinity") value = Number.NEGATIVE_INFINITY;
  else if (text === "-0") value = -0;
  else value = Number(text);
  if (Number.isNaN(value) && text !== "NaN") return { ok: false, value: null };
  if (float32) value = Math.fround(value);
  return canonicalFloatText(value) === text ? { ok: true, value } : { ok: false, value: null };
}
