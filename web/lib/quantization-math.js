const Q31_SCALE = 2 ** 31;

export function quantizeMultiplier(realMultiplier, singleRounding = false) {
  if (realMultiplier === 0) return { multiplier: 0, shift: 0, represented: 0 };
  let [fraction, shift] = frexpPositive(realMultiplier);
  let multiplier = Math.round(fraction * Q31_SCALE);
  if (multiplier === Q31_SCALE) {
    multiplier /= 2;
    shift += 1;
  }
  if (shift < -31) {
    multiplier = 0;
    shift = 0;
  }
  if (singleRounding && shift > 30) {
    multiplier = Q31_SCALE - 1;
    shift = 30;
  }
  return { multiplier, shift, represented: multiplier * (2 ** (shift - 31)) };
}

function frexpPositive(value) {
  if (!(Number.isFinite(value) && value > 0)) throw new Error("frexp input must be positive and finite.");
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  const bits = view.getBigUint64(0, false);
  const exponent = Number((bits >> 52n) & 0x7ffn);
  if (exponent === 0) {
    const [fraction, shift] = frexpPositive(value * (2 ** 54));
    return [fraction, shift - 54];
  }
  const fractionBits = (bits & ((1n << 52n) - 1n)) | (1022n << 52n);
  view.setBigUint64(0, fractionBits, false);
  return [view.getFloat64(0, false), exponent - 1022];
}

export function roundTiesAway(value) {
  return value >= 0 ? Math.floor(value + 0.5) : Math.ceil(value - 0.5);
}

export function multiplyByQuantizedMultiplierDefault(value, multiplier, shift) {
  const x = checkedInt32(value, "Fixed-point input");
  const q = checkedNonnegativeInt32(multiplier, "Quantized multiplier");
  const exponent = checkedShift(shift);
  const leftShift = Math.max(exponent, 0);
  const rightShift = Math.max(-exponent, 0);
  const shifted = BigInt(x) * (1n << BigInt(leftShift));
  if (shifted < -2147483648n || shifted > 2147483647n) return null;
  const high = saturatingRoundingDoublingHighMul(Number(shifted), q);
  return roundingDivideByPowerOfTwo(high, rightShift);
}

export function multiplyByQuantizedMultiplierSingleRounding(value, multiplier, shift) {
  const x = checkedInt32(value, "Fixed-point input");
  const q = checkedNonnegativeInt32(multiplier, "Quantized multiplier");
  const exponent = checkedShift(shift);
  const totalShift = 31 - exponent;
  const rounded = BigInt(x) * BigInt(q) + (1n << BigInt(totalShift - 1));
  const result = rounded >> BigInt(totalShift);
  if (result < -2147483648n || result > 2147483647n) return null;
  return Number(result);
}

function saturatingRoundingDoublingHighMul(left, right) {
  if (left === -2147483648 && right === -2147483648) return 2147483647;
  const product = BigInt(left) * BigInt(right);
  const nudge = product >= 0n ? 1n << 30n : 1n - (1n << 30n);
  return Number((product + nudge) / (1n << 31n));
}

function roundingDivideByPowerOfTwo(value, exponent) {
  if (exponent === 0) return value;
  const x = BigInt(value);
  const mask = (1n << BigInt(exponent)) - 1n;
  const remainder = x & mask;
  const threshold = (mask >> 1n) + (x < 0n ? 1n : 0n);
  return Number((x >> BigInt(exponent)) + (remainder > threshold ? 1n : 0n));
}

function checkedInt32(value, label) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < -2147483648 || number > 2147483647) {
    throw new Error(`${label} must be a signed 32-bit integer.`);
  }
  return number;
}

function checkedNonnegativeInt32(value, label) {
  const number = checkedInt32(value, label);
  if (number < 0) throw new Error(`${label} must be non-negative.`);
  return number;
}

function checkedShift(value) {
  const shift = Number(value);
  if (!Number.isInteger(shift) || shift < -31 || shift > 30) {
    throw new Error("Fixed-point shift must be in [-31, 30].");
  }
  return shift;
}
