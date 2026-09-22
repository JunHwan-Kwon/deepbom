import { canonicalJson } from "../report-utils.js";
import { sha256TextHex } from "../sha256-sync.js";
import { validateModelIr } from "../model-ir.js";

export const METHOD_VERSION = "1.0.0";
export const SHA256 = /^[a-f0-9]{64}$/;
export function requireCondition(ok, message) { if (!ok) throw new Error(`Numerical IR: ${message}`); }
export function exactKeys(value, keys, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const allowed = new Set(keys);
  requireCondition(Object.keys(value).every(key => allowed.has(key)), `${label} contains an unknown field`);
  requireCondition(keys.every(key => Object.hasOwn(value, key)), `${label} is missing a field`);
}
export function sourceContract(input) {
  const model = validateModelIr(input);
  return { model, source: { model_ir_sha256: model.model_ir_sha256, artifact_sha256: model.artifact.sha256, artifact_set_sha256: model.artifact_set.artifact_set_sha256 } };
}
export function bindSource(source, model) {
  exactKeys(source, ["model_ir_sha256", "artifact_sha256", "artifact_set_sha256"], "source");
  const expected = sourceContract(model).source;
  requireCondition(canonicalJson(source) === canonicalJson(expected), "source does not match Model IR and artifact identity");
}
export function seal(body, field) { return { ...body, [field]: sha256TextHex(canonicalJson(body)) }; }
export function checkDigest(document, field) {
  const { [field]: digest, ...body } = document;
  requireCondition(SHA256.test(digest || "") && digest === sha256TextHex(canonicalJson(body)), `${field} mismatch`);
}
export function decimalCount(value, label = "count") {
  requireCondition(typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value), `${label} must be a canonical nonnegative decimal string`);
  return BigInt(value);
}
export function shapeCount(shape) {
  requireCondition(Array.isArray(shape) && shape.every(v => Number.isSafeInteger(v) && v >= 0), "numeric shape must have exact nonnegative dimensions");
  return shape.reduce((n, v) => n * BigInt(v), 1n);
}
export function canonicalValue(value) {
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number") return Number.isNaN(value) ? "NaN" : value === Infinity ? "+Infinity" : value === -Infinity ? "-Infinity" : Object.is(value, -0) ? "-0" : value;
  requireCondition(["NaN", "+Infinity", "-Infinity", "-0"].includes(value), "unsupported numeric value");
  return value;
}
export function parseNumeric(value) {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return value;
  if (value === "NaN") return NaN;
  if (value === "+Infinity") return Infinity;
  if (value === "-Infinity") return -Infinity;
  if (value === "-0") return -0;
  if (typeof value === "string" && /^-?(0|[1-9][0-9]*)$/.test(value)) return BigInt(value);
  throw new Error("Numerical IR: value is not a number or canonical numeric token");
}
