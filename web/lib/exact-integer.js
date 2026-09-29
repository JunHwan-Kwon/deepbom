// Exact nonnegative counts shared by evidence contracts. Numeric mirrors are optional only above the safe integer range.
export function exactInteger(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "string" && /^\d+$/.test(value)) return exact(BigInt(value));
  if (typeof value === "bigint" && value >= 0n) return exact(value);
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) return exact(BigInt(value));
  return null;
}

function exact(value) { return { decimal: value.toString(), number: safeBigIntNumber(value) }; }

export function safeBigIntNumber(value) {
  return typeof value === "bigint" && value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

// Legacy public ledgers use `value`, while common IR uses `number`. Share the
// count/mirror rule without changing either wire contract.
export function exactIntegerWithValue(value) {
  const result = exactInteger(value);
  return result ? { value: result.number, decimal: result.decimal } : null;
}

export function exactCountValue(value) {
  const record = value && typeof value === "object" ? value : null;
  const raw = record ? record.decimal : value;
  if (typeof raw === "string" && !/^(0|[1-9][0-9]*)$/.test(raw)) return null;
  const count = exactInteger(raw);
  if (!count) return null;
  if (record && ["number", "value"].some(key => Object.hasOwn(record, key) && record[key] !== count.number)) return null;
  return BigInt(count.decimal);
}

export function validateExactInteger(value) {
  if (value == null) return;
  if (typeof value !== "object" || typeof value.decimal !== "string" || !/^(0|[1-9][0-9]*)$/.test(value.decimal)) throw new Error("IR exact integer decimal is invalid.");
  const integer = BigInt(value.decimal), expected = integer <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(integer) : null;
  if (value.number !== expected) throw new Error("IR exact integer numeric mirror contradicts its decimal value.");
}
