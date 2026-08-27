const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

function exactInteger(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  if (typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value)) return BigInt(value);
  return null;
}

export function sumCoreMlExactIntegers(values) {
  let total = 0n;
  for (const value of values) {
    const parsed = exactInteger(value);
    if (parsed == null) throw new Error("Core ML exact-integer ledger received an invalid nonnegative value");
    total += parsed;
  }
  return { decimal: total.toString(), number: total <= MAX_SAFE ? Number(total) : null };
}

export function multiplyCoreMlExactIntegers(values) {
  let product = 1n;
  for (const value of values) {
    const parsed = exactInteger(value);
    if (parsed == null) return null;
    product *= parsed;
  }
  return { decimal: product.toString(), number: product <= MAX_SAFE ? Number(product) : null };
}

export function coreMlExactLedger(values, complete) {
  const assessed = sumCoreMlExactIntegers(values);
  return {
    assessed_value: assessed.number,
    assessed_value_decimal: assessed.decimal,
    complete_value: complete ? assessed.number : null,
    complete_value_decimal: complete ? assessed.decimal : null,
    safe_number_mirror_status: assessed.number == null ? "exact_decimal_only" : "safe_integer_mirror_available",
  };
}
