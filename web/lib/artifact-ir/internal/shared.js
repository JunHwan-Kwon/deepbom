import { buildValueType, valueLogicalBytes } from "../../ir-value-type.js";
import { exactInteger } from "../../exact-integer.js";
export { exactInteger, validateExactInteger } from "../../exact-integer.js";
import { SHA256 } from "./constants.js";

export function logicalTensorBytes(tensor, format = "") {
  if (Object.hasOwn(tensor, "logical_payload_bytes")) return exactInteger(tensor.logical_payload_bytes);
  return valueLogicalBytes(buildValueType(tensor, format, "logical-byte-count"));
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

export function tensorIndex(tensor, fallback) { return nonNegativeInteger(tensor?.index) ?? fallback; }

export function normalizeFormat(value) { return String(value || "unknown").trim().toLowerCase().replace(".mlmodel", "coreml"); }

export function normalizeSha256(value) { const normalized = String(value || "").trim().toLowerCase(); return SHA256.test(normalized) ? normalized : null; }

export function list(value) { return Array.isArray(value) ? value : []; }

export function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }

export function dimensions(value) {
  return list(value).map(item => {
    if (item == null) return "?";
    if (typeof item === "string" || (typeof item === "number" && Number.isSafeInteger(item))) return item;
    if (typeof item === "bigint") return item.toString();
    throw new Error("Artifact IR dimension must be an exact integer, symbolic string, or unknown value.");
  });
}
export function shapeRankStatus(tensor, format) {
  if (!Array.isArray(tensor.shape) || tensor.shape_declared === false) return "unknown_rank";
  return tensor.shape.length || tensor.shape_declared === true || tensor.has_rank === true || ["onnx", "safetensors", "gguf"].includes(format) ? "ranked" : "unknown_rank";
}

export function integerArray(value) { return list(value).map(optionalInteger).filter(item => item !== null); }

export function finiteNumberArray(value) { return list(value).filter(item => typeof item === "number" && Number.isFinite(item)); }

export function compactStrings(value) { return list(Array.isArray(value) ? value : [value]).map((item) => String(item || "").trim()).filter(Boolean); }

export function optionalText(value) { const normalized = String(value ?? "").trim(); return normalized || null; }

export function text(value, maximum) { const normalized = String(value ?? "").trim(); return normalized.length > 0 && normalized.length <= maximum; }

export function nonNegativeInteger(value) { const number = optionalInteger(value); return number !== null && number >= 0 ? number : null; }

export function positiveInteger(value) { const number = optionalInteger(value); return number !== null && number > 0 ? number : null; }

export function optionalInteger(value) { if (typeof value !== "number" && !(typeof value === "string" && /^-?(0|[1-9][0-9]*)$/.test(value))) return null; const number = Number(value); return Number.isSafeInteger(number) ? number : null; }


export function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }


export function deepFreeze(value) { if (!value || typeof value !== "object" || Object.isFrozen(value)) return value; Object.freeze(value); for (const child of Object.values(value)) deepFreeze(child); return value; }
