// IEEE binary16/bfloat16 interpretation, shared by native-format adapters.
// Keep NaN/Inf and signed zero; finite-value policy belongs to the consumer.
export function float16ToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 31, fraction = bits & 1023;
  if (!exponent) return fraction ? sign * fraction * 2 ** -24 : sign < 0 ? -0 : 0;
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

const FLOAT32_SCRATCH = new DataView(new ArrayBuffer(4));
export function bfloat16ToNumber(bits) {
  FLOAT32_SCRATCH.setUint32(0, bits << 16, false);
  return FLOAT32_SCRATCH.getFloat32(0, false);
}

// Direct integer -> binary32 round-to-nearest, ties-to-even. Converting through
// binary64 first can double-round large integers at a binary32 midpoint.
export function bigintToFloat32(value) {
  if (value === 0n) return 0;
  const negative = value < 0n, magnitude = negative ? -value : value;
  let exponent = magnitude.toString(2).length - 1;
  if (exponent <= 23) return Math.fround(Number(value));
  const shift = BigInt(exponent - 23);
  let significand = magnitude >> shift;
  const remainder = magnitude - (significand << shift), half = 1n << (shift - 1n);
  if (remainder > half || remainder === half && (significand & 1n) === 1n) significand += 1n;
  if (significand === 1n << 24n) { significand >>= 1n; exponent += 1; }
  return Math.fround((negative ? -1 : 1) * Number(significand) * 2 ** (exponent - 23));
}

export function numericToFloat32(value) {
  return typeof value === "bigint" ? bigintToFloat32(value) : Math.fround(Number(value));
}

export function typedNumericToFloat32(value, dtype) {
  if (dtype === "INT64" && typeof value === "bigint") return bigintToFloat32(value);
  return Math.fround(Number(value));
}

export function normalizeAxis(axis, rank) {
  if (!Number.isSafeInteger(axis) || !Number.isSafeInteger(rank) || rank <= 0) return null;
  const value = axis < 0 ? axis + rank : axis;
  return value >= 0 && value < rank ? value : null;
}

export function float64BitsHex(value) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  return view.getBigUint64(0, false).toString(16).padStart(16, "0");
}

// Contiguous little-endian bit packing, including values crossing byte boundaries.
// Callers retain native dtype, padding and total-cardinality validation.
export function readPackedUnsigned(bytes, index, bits) {
  if (!Number.isSafeInteger(index) || index < 0 || !Number.isInteger(bits) || bits < 1 || bits > 8
    || index >= Math.floor(bytes.length * 8 / bits)) return null;
  const bitOffset = index * bits, offset = Math.floor(bitOffset / 8), shift = bitOffset % 8;
  const word = bytes[offset] | ((bytes[offset + 1] ?? 0) << 8);
  return word >>> shift & (2 ** bits - 1);
}
