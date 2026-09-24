import { exactInteger } from "../../exact-integer.js";
import { compareCanonicalText } from "../../report-utils.js";
export function list(value) { return Array.isArray(value) ? value : []; }
export function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }
export function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
export function unique(values) { return [...new Set(list(values))]; }
export function text(value) { const result = String(value ?? "").trim(); return result || null; }
export function exact(value) {
  if (value == null) return null;
  const decimal = typeof value === "object" ? value.decimal : value;
  return exactInteger(decimal);
}
export function evidenceClass(value, fallback = "DERIVED") {
  const normalized = String(value || "").toUpperCase();
  if (normalized.includes("RUNTIME")) return "OBSERVED_RUNTIME";
  if (normalized.includes("OBSERVED")) return "OBSERVED_SERIALIZED_ARTIFACT";
  if (normalized.includes("DECLARED")) return "DECLARED_UNVERIFIED";
  if (normalized.includes("PREDICT")) return "PREDICTED";
  if (normalized.includes("NOT_ASSESSABLE") || normalized.includes("NOT ASSESSABLE")) return "NOT_ASSESSABLE";
  return fallback;
}
export function sortById(rows) { return [...list(rows)].sort((left, right) => compareCanonicalText(left.id, right.id)); }
