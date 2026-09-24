import { canonicalJson } from "./report-utils.js";
import { Sha256Accumulator } from "./sha256-sync.js";

// Canonical JSON array digest, streamed without constructing a second full array/string.
export function parameterVectorEvidence(values, kind, { inlineLimit = 32 } = {}) {
  if (!Array.isArray(values) || !["scale", "zero_point"].includes(kind)) throw new Error("Invalid quantization parameter vector.");
  const valid = values.every(value => typeof value === "number" && (kind === "scale" ? Number.isFinite(value) : Number.isSafeInteger(value)));
  if (!valid) return { count: values.length, sha256: null, inline_values: null, inline_status: "not_assessed_invalid_or_inexact_parameter", collection_status: "invalid_or_inexact_values", all_zero: null };
  const hash = new Sha256Accumulator(), encoder = new TextEncoder();
  hash.update(encoder.encode("["));
  let chunk = "";
  for (let i = 0; i < values.length; i++) {
    chunk += `${i ? "," : ""}${canonicalJson(values[i])}`;
    if (chunk.length >= 8192) { hash.update(encoder.encode(chunk)); chunk = ""; }
  }
  hash.update(encoder.encode(`${chunk}]`));
  return { count: values.length, sha256: hash.digestHex(), inline_values: values.length <= inlineLimit ? [...values] : null,
    inline_status: values.length <= inlineLimit ? "complete" : "digest_only_large_vector", collection_status: "complete_vector", all_zero: values.every(value => value === 0) };
}
