// Exact arithmetic and mirror validation are owned by the common WASM engine.
import { portableContract } from "./portable-contracts.js";
import { canonicalJsonInput } from "./report-utils.js";
const call = (operation, value) => portableContract("ir_semantics", {operation,value});
function scalar(value) {
  if (typeof value === "bigint") return value.toString();
  return typeof value === "string" || typeof value === "number" && Number.isFinite(value) ? value : null;
}
export function exactInteger(value) { return call("exact_integer", scalar(value)); }
export function safeBigIntNumber(value) { return typeof value === "bigint" ? call("safe_bigint_number", value.toString()) : null; }
// Legacy public ledgers use `value`; common IR uses `number`.
export function exactIntegerWithValue(value) {
  const result = exactInteger(value);
  return result ? {value:result.number,decimal:result.decimal} : null;
}
export function exactCountValue(value) {
  let input;
  if (value && typeof value === "object") {
    input = {decimal:scalar(value.decimal)};
    for (const key of ["number","value"]) if (Object.hasOwn(value,key)) {
      if (value[key] !== null && (typeof value[key] !== "number" || !Number.isFinite(value[key]))) return null;
      input[key]=value[key];
    }
  } else input=scalar(value);
  const result=call("exact_count_value",input);
  return result === null ? null : BigInt(result);
}
export function validateExactInteger(value) {
  if (value == null) return;
  call("validate_exact_integer",canonicalJsonInput(value));
}
