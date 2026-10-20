import { shapeElementCount } from "../tensor-size.js";
import { canonicalJsonInput } from "../report-utils.js";
import { canonicalEvidenceDigest } from "../evidence-identity.js";
import { portableContract } from "../portable-contracts.js";
import { modelEvidenceSource } from "../ir-evidence-contract.js";

export const METHOD_VERSION = "1.0.0";
export const SHA256 = /^[a-f0-9]{64}$/;
export function requireCondition(ok, message) { if (!ok) throw new Error(`Numerical IR: ${message}`); }
export function exactKeys(value, keys, label) {
  const error = portableContract("evidence_contract", {operation:"exact_keys",value:canonicalJsonInput(value ?? null),keys,label});
  requireCondition(!error,error);
}
export function sourceContract(input) {
  return modelEvidenceSource(input);
}
export function bindSource(source, model) {
  sourceContract(model);
  portableContract("semantic_validation",{operation:"numerical_source",value:source,model});
}
export function seal(body, field) { return { ...body, [field]: canonicalEvidenceDigest(body) }; }
export function checkDigest(document, field) {
  const { [field]: digest, ...body } = document;
  requireCondition(SHA256.test(digest || "") && digest === canonicalEvidenceDigest(body), `${field} mismatch`);
}
export function decimalCount(value, label = "count") {
  requireCondition(portableContract("evidence_contract", {operation:"decimal_count",value:value ?? null}), `${label} must be a canonical nonnegative decimal string`);
  return BigInt(value);
}
export function shapeCount(shape) {
  const count = shapeElementCount(shape);
  requireCondition(count != null, "numeric shape must have exact nonnegative dimensions");
  return BigInt(count.decimal);
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
