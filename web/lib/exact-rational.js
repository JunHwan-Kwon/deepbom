import { portableContract } from "./portable-contracts.js";
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

// Arithmetic belongs to the common WASM owner. BigInt is only host transport.
const encodeDecimal = value => ({coefficient:String(value.coefficient), exponent:value.exponent});
const decodeDecimal = value => ({coefficient:BigInt(value.coefficient), exponent:value.exponent});
function decimalOperation(operation,a,b) {
  return portableContract("decimal", {operation,a:encodeDecimal(a),...(b === undefined ? {} : {b:encodeDecimal(b)})});
}
export function reportedDecimal(value) {
  if(typeof value !== "number" || !Number.isFinite(value))throw Error("Reported decimal requires a finite number");
  return decodeDecimal(portableContract("decimal", {operation:"reported",value}));
}
export function addExactDecimals(a,b){return decodeDecimal(decimalOperation("add",a,b));}
export function negateExactDecimal(a){return decodeDecimal(decimalOperation("negate",a));}
export function subtractExactDecimals(a,b){return decodeDecimal(decimalOperation("subtract",a,b));}
export function halveExactDecimal(a){return decodeDecimal(decimalOperation("halve",a));}
export function compareExactDecimals(a,b){return decimalOperation("compare",a,b);}
export function exactDecimalText(a){return decimalOperation("text",a);}
export function exactDecimalNumber(a){return decimalOperation("number",a);}
