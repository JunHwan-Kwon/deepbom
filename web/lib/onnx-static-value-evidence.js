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

export function staticNumericInput(input, dtype) {
  if (dtype === "INT64" && input?.initializerIntegerValuesExactComplete === true
    && Array.isArray(input.initializerIntegerValuesExactDecimals)) {
    try { return input.initializerIntegerValuesExactDecimals.map(value => BigInt(value)); } catch { return null; }
  }
  if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) return input.staticValues;
  return null;
}

export function floatListAttribute(attribute) {
  if (attribute?.type !== 6 || !Array.isArray(attribute.floats)
    || !Array.isArray(attribute.valueTypesPresent) || attribute.valueTypesPresent.length !== 1
    || attribute.valueTypesPresent[0] !== 6) return null;
  return attribute.floats.map(value => Math.fround(value));
}

export function numericTokenValue(value) {
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  if (value === "-0") return -0;
  return Number(value);
}

export function duplicateValueCount(values) {
  return values.length - new Set(values.map(value => typeof value === "bigint" ? `i:${value}` : `s:${value}`)).size;
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
