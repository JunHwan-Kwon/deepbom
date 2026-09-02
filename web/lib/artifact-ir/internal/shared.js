import { SHA256 } from "./constants.js";

export function logicalTensorBytes(tensor) {
  return exactInteger(tensor.logical_payload_bytes ?? tensor.byte_length ?? tensor.buffer_data_length ?? tensor.initializer_available_bytes);
}

export function positiveStorageBytes(tensor, format = "") {
  const candidates = [
    ...(["gguf", "safetensors"].includes(format) ? [tensor.byte_length] : []),
    tensor.buffer_data_length,
    tensor.initializer_available_bytes,
    tensor.initializer_bytes,
    tensor.serialized_payload_bytes,
  ];
  for (const value of candidates) {
    const number = nonNegativeInteger(value);
    if (number && number > 0) return number;
  }
  return 0;
}

export function safeExactSum(start, decimalLength) {
  const result = BigInt(start) + BigInt(decimalLength);
  return result <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(result) : result.toString();
}

export function uniqueIds(rows, label) {
  const ids = new Set();
  for (const row of list(rows)) {
    if (!text(row?.id, 600) || ids.has(row.id)) throw new Error(`Artifact IR ${label} identity is invalid or duplicated.`);
    ids.add(row.id);
  }
  return ids;
}

export function storageId(index) { return `storage:tensor:${index}`; }

export function scopedStorageId(scopeId, index) { return `storage:${scopeId}:tensor:${index}`; }

export function tensorIndex(tensor, fallback) { return Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : fallback; }

export function normalizeFormat(value) { return String(value || "unknown").trim().toLowerCase().replace(".mlmodel", "coreml"); }

export function normalizeSha256(value) { const normalized = String(value || "").trim().toLowerCase(); return SHA256.test(normalized) ? normalized : null; }

export function list(value) { return Array.isArray(value) ? value : []; }

export function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }

export function dimensions(value) { return list(value).map((item) => Number.isSafeInteger(Number(item)) ? Number(item) : String(item)); }

export function integerArray(value) { return list(value).map(Number).filter(Number.isSafeInteger); }

export function finiteNumberArray(value) { return list(value).map(Number).filter(Number.isFinite); }

export function compactStrings(value) { return list(Array.isArray(value) ? value : [value]).map((item) => String(item || "").trim()).filter(Boolean); }

export function optionalText(value) { const normalized = String(value ?? "").trim(); return normalized || null; }

export function text(value, maximum) { const normalized = String(value ?? "").trim(); return normalized.length > 0 && normalized.length <= maximum; }

export function nonNegativeInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : null; }

export function positiveInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number > 0 ? number : null; }

export function optionalInteger(value) { if (value === null || value === undefined || value === "") return null; const number = Number(value); return Number.isSafeInteger(number) ? number : null; }

export function exactInteger(value) {
  if (typeof value === "string" && /^\d+$/.test(value)) return exact(BigInt(value));
  if (typeof value === "bigint" && value >= 0n) return exact(value);
  if (Number.isSafeInteger(Number(value)) && Number(value) >= 0) return exact(BigInt(Number(value)));
  return null;
}

export function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }

export function deepFreeze(value) { if (!value || typeof value !== "object" || Object.isFrozen(value)) return value; Object.freeze(value); for (const child of Object.values(value)) deepFreeze(child); return value; }
