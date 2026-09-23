// A finite binary64 value is an integer multiple of 2^-1074. Accumulate
// integer significands by exponent; shift only when finishing, not per value.
// This preserves cancellation, subnormals and variance near a large offset.
// Counts and both moments are exact; only the final result is rounded.
const bits = new DataView(new ArrayBuffer(8));
const bitLength = value => value.toString(2).length;
function ratioExponent(numerator, denominator) {
  let exponent = bitLength(numerator) - bitLength(denominator);
  if (exponent >= 0 ? numerator < (denominator << BigInt(exponent)) : (numerator << BigInt(-exponent)) < denominator) exponent--;
  return exponent;
}
function scaledRatio(numerator, denominator, shift) {
  return shift >= 0 ? [numerator << BigInt(shift), denominator] : [numerator, denominator << BigInt(-shift)];
}
function roundedRatio(numerator, denominator, power) {
  if (!numerator) return 0;
  const negative = numerator < 0n;
  if (negative) numerator = -numerator;
  const quantum = Math.max(-1074, ratioExponent(numerator, denominator) + power - 52);
  const [a, b] = scaledRatio(numerator, denominator, power - quantum);
  let q = a / b;
  const remainder = 2n * (a % b);
  if (remainder > b || remainder === b && (q & 1n)) q++;
  const result = Number(q) * 2 ** quantum;
  return negative ? -result : result;
}
export function exactCountRatio(numerator, denominator) {
  const n=BigInt(numerator),d=BigInt(denominator);
  if(n<0n||d<0n)throw new Error("Count ratios require nonnegative counts.");
  return d ? roundedRatio(n,d,0) : null;
}
function integerSqrt(value) {
  if (value < 2n) return value;
  let x = 1n << BigInt(Math.ceil(bitLength(value) / 2));
  for (;;) { const next = (x + value / x) >> 1n; if (next >= x) return x; x = next; }
}
function roundedSqrt(numerator, denominator, power) {
  if (!numerator) return 0;
  const quantum = Math.max(-1074, Math.floor((ratioExponent(numerator, denominator) + power) / 2) - 52);
  const [a, b] = scaledRatio(numerator, denominator, power - 2 * quantum);
  let q = integerSqrt(a / b);
  // Compare the exact radicand against the square of the rounding midpoint.
  const midpoint = b * (2n * q + 1n) ** 2n;
  if (4n * a > midpoint || 4n * a === midpoint && (q & 1n)) q++;
  return Number(q) * 2 ** quantum;
}
function components(value) {
  bits.setFloat64(0, value);
  const high = bits.getUint32(0), low = bits.getUint32(4), exponent = (high >>> 20) & 2047;
  if (exponent === 2047) throw new Error("Exact moments require finite values.");
  const significand = BigInt((high & 0xfffff) * 4294967296 + low + (exponent ? 4503599627370496 : 0));
  return { shift: exponent ? exponent - 1 : 0, significand: high >>> 31 ? -significand : significand };
}
function addProduct(bins, shift, value) { if (value) bins.set(shift, (bins.get(shift) || 0n) + value); }
function productTotal(bins) { let sum=0n; for(const [shift,value] of bins)sum+=value<<BigInt(shift); return sum; }
function vectorProducts(a, b) {
  if (a.length !== b.length) throw new Error("Vector products require equal value counts.");
  const aa=new Map(), bb=new Map(), ab=new Map();
  for(let i=0;i<a.length;i++) {
    const x=components(a[i]),y=components(b[i]);
    addProduct(aa,2*x.shift,x.significand*x.significand);
    addProduct(bb,2*y.shift,y.significand*y.significand);
    addProduct(ab,x.shift+y.shift,x.significand*y.significand);
  }
  return {left:productTotal(aa),right:productTotal(bb),dot:productTotal(ab)};
}
function cosineDistanceFromProducts(left, right, dot) {
  if (!left || !right) return null;
  const product = left * right;
  if (!dot) return 1;
  if (dot * dot === product) return dot < 0n ? 2 : 0;
  // Rationalize 1-cos(theta) so near-parallel vectors retain small distances.
  const scale = 1n << 128n, root = integerSqrt(product << 256n);
  let value = dot > 0n
    ? roundedRatio((product - dot * dot) * scale, product * scale + dot * root, 0)
    : roundedRatio(root - dot * scale, root, 0);
  const units = x => { const c = components(x); return c.significand << BigInt(c.shift); };
  const denominator = 1n << 1075n;
  const compare = numerator => {
    const complement = dot > 0n ? denominator - numerator : numerator - denominator;
    if (complement < 0n) return dot > 0n ? -1 : 1;
    const difference = dot * dot * denominator * denominator - product * complement * complement;
    const sign = difference < 0n ? -1 : difference > 0n ? 1 : 0;
    return dot > 0n ? -sign : sign;
  };
  // Correct the approximate quotient by comparing the *exact* irrational
  // distance against binary64 rounding midpoints (including ties-to-even).
  for (;;) {
    bits.setFloat64(0, value);
    const code = bits.getBigUint64(0), odd = Boolean(code & 1n);
    if (code > 0n) {
      bits.setBigUint64(0, code - 1n); const previous = bits.getFloat64(0);
      const cmp = compare(units(previous) + units(value));
      if (cmp < 0 || cmp === 0 && odd) { value = previous; continue; }
    }
    bits.setBigUint64(0, code + 1n); const next = bits.getFloat64(0);
    const cmp = compare(units(value) + units(next));
    if (cmp > 0 || cmp === 0 && odd) { value = next; continue; }
    return value;
  }
}
export function exactRegressionGain(reference, observed) {
  const {left,dot}=vectorProducts(reference,observed);
  const value=left ? roundedRatio(dot,left,0) : null;
  return Number.isFinite(value) ? value : null;
}
export function exactCosine(a, b) {
  const {left,right,dot}=vectorProducts(a,b);
  if (!left || !right) return null;
  const absolute=roundedSqrt(dot*dot,left*right,0);
  return dot<0n ? -absolute : absolute;
}
export class ExactMoments {
  constructor({ squares = true } = {}) { this.bins = new Map(); this.withSquares = squares; }
  add(value) {
    if (value === 0) return;
    const {shift,significand}=components(value);
    let bin = this.bins.get(shift);
    if (!bin) { bin = { sum: 0n, squares: 0n }; this.bins.set(shift, bin); }
    bin.sum += significand;
    if (this.withSquares) bin.squares += significand * significand;
  }
  totals() {
    let sum = 0n, squares = 0n;
    for (const [shift, bin] of this.bins) { sum += bin.sum << BigInt(shift); squares += bin.squares << BigInt(shift * 2); }
    return { sum, squares };
  }
  mean(count) { return count ? roundedRatio(this.totals().sum, BigInt(count), -1074) : null; }
  dividedBy(value, factor = 1) {
    const {shift,significand}=components(value);
    if(significand<=0n)throw new Error("Exact sum divisor must be positive.");
    return roundedRatio(this.totals().sum,significand*BigInt(factor),-shift);
  }
  coefficientOfVariation(count) {
    if (!count) return null;
    const { sum, squares } = this.totals(), n = BigInt(count);
    return sum ? roundedSqrt(squares * n - sum * sum, sum * sum, 0) : null;
  }
  finish(count) {
    if (!count) return { mean: null, std: null, rms: null, l2: null };
    const n = BigInt(count), { sum, squares } = this.totals();
    return {
      mean: roundedRatio(sum, n, -1074),
      std: roundedSqrt(squares * n - sum * sum, n * n, -2148),
      rms: roundedSqrt(squares, n, -2148),
      l2: roundedSqrt(squares, 1n, -2148),
    };
  }
}

// Retain exact products across tensors and samples. In particular, neither
// subtraction nor squaring is rounded before computing a normalized metric.
export class ExactComparison {
  constructor() {
    this.count = 0;
    this.absolute = new Map(); this.differenceSquares = new Map();
    this.leftSquares = new Map(); this.rightSquares = new Map(); this.dot = new Map();
    this.maximum = 0n;
  }
  add(left, right) {
    const x = components(left), y = components(right);
    const shift = Math.min(x.shift, y.shift);
    const delta = (y.significand << BigInt(y.shift - shift)) - (x.significand << BigInt(x.shift - shift));
    const absolute = delta < 0n ? -delta : delta;
    addProduct(this.absolute, shift, absolute);
    addProduct(this.differenceSquares, 2 * shift, delta * delta);
    addProduct(this.leftSquares, 2 * x.shift, x.significand * x.significand);
    addProduct(this.rightSquares, 2 * y.shift, y.significand * y.significand);
    addProduct(this.dot, x.shift + y.shift, x.significand * y.significand);
    const maximum = absolute << BigInt(shift);
    if (maximum > this.maximum) this.maximum = maximum;
    this.count++;
  }
  merge(other) {
    for (const name of ["absolute", "differenceSquares", "leftSquares", "rightSquares", "dot"]) {
      for (const [shift, value] of other[name]) addProduct(this[name], shift, value);
    }
    this.count += other.count;
    if (other.maximum > this.maximum) this.maximum = other.maximum;
    return this;
  }
  finish() {
    const n = BigInt(this.count || 1), absolute = productTotal(this.absolute);
    const squares = productTotal(this.differenceSquares), left = productTotal(this.leftSquares);
    const right = productTotal(this.rightSquares), dot = productTotal(this.dot);
    const metrics = {
      total_absolute_difference: roundedRatio(absolute, 1n, -1074),
      mean_absolute_difference: roundedRatio(absolute, n, -1074),
      root_mean_square_difference: roundedSqrt(squares, n, -2148),
      maximum_absolute_difference: roundedRatio(this.maximum, 1n, -1074),
      relative_l2_difference: left ? roundedSqrt(squares, left, 0) : null,
      cosine_distance: cosineDistanceFromProducts(left, right, dot),
    };
    const unrepresentable = [];
    for (const [name, value] of Object.entries(metrics)) {
      if (value != null && !Number.isFinite(value)) { metrics[name] = null; unrepresentable.push(name); }
    }
    return { ...metrics, unrepresentable_metrics: unrepresentable };
  }
}
