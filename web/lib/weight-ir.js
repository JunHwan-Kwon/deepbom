import { portableContract } from "./portable-contracts.js";
import { initializeNumericalWasm } from "./numerical-wasm.js";
import { METHOD_VERSION, requireCondition, seal, shapeCount, sourceContract } from "./numerical-ir/common.js";
import { TensorStatistics, validateStatistics } from "./numerical-ir/statistics.js";
import { hashSource, numericSources, verifySourceAfterRead } from "./numerical-ir/weight-sources.js";
import { canonicalJson } from "./report-utils.js";

import { WEIGHT_IR_SCHEMA } from "./evidence-ir.js";
export { WEIGHT_IR_SCHEMA };
function boundary() { return portableContract("semantic_validation", {operation:"weight_boundary"}); }
function coverage(rows, model) {
  return portableContract("semantic_validation", {operation:"weight_coverage",tensors:rows,model});
}
export async function buildWeightIr(modelIr, analysis, sourceBytes, { maxValues = 100_000_000, signal, onProgress } = {}) {
  await initializeNumericalWasm();
  const { model, source } = sourceContract(modelIr);
  requireCondition(Number.isSafeInteger(maxValues) && maxValues >= 0, "invalid numeric value budget");
  requireCondition(await hashSource(sourceBytes, { signal, onProgress }) === source.artifact_sha256, "weight source bytes do not match artifact SHA-256");
  const candidates = await numericSources(sourceBytes, analysis, model), rows = [];
  let remaining = BigInt(maxValues);
  for (const [index, candidate] of candidates.entries()) {
    signal?.throwIfAborted();
    const exactShape = Array.isArray(candidate.shape) && candidate.shape.every(v => Number.isSafeInteger(v) && v >= 0);
    const expected = exactShape ? shapeCount(candidate.shape) : null, stats = new TensorStatistics();
    let result;
    try {
    if (expected === null) result = { status: "not_assessed", reason: "shape_not_exact" };
    else if (expected > remaining) result = { status: "not_assessed", reason: "value_budget_exceeded" };
    else {
      result = await candidate.visit(v => { requireCondition(stats.count < expected, "decoder emitted more values than the declared shape"); stats.add(v); }, { maxValues: Number(remaining), signal });
      if (result.status === "assessed") {
        result.statistics ||= stats.finish(); validateStatistics(result.statistics);
        requireCondition(BigInt(result.statistics.value_count) === expected, `payload ${candidate.locator} does not conserve declared value count`);
        remaining -= expected;
      }
    }
    } finally {stats.free();}
    const assessed = result.status === "assessed";
    rows.push({ id: `weight:${index}`, storage_ref: candidate.storage_ref, native_locator: candidate.locator, name: String(candidate.name || ""), dtype: String(candidate.dtype || "UNKNOWN"),
      shape: exactShape ? candidate.shape : null, status: assessed ? "assessed" : "not_assessed", reason: assessed ? null : String(result.reason || "decoder_unavailable"),
      representation: assessed ? result.representation : null, payload_sha256: result.payload_sha256 || null,
      invalid_encoding_count: assessed ? String(result.invalid_encoding_count || 0) : null,
      binding_refs: model.weight_bindings.bindings.filter(b => b.storage_ref === candidate.storage_ref).map(b => b.id),
      statistics: assessed ? result.statistics : null });
    onProgress?.({ phase: "weights", completed: index + 1, total: candidates.length });
  }
  const document = seal({ schema: WEIGHT_IR_SCHEMA, method_version: METHOD_VERSION, source, budget: { maximum_decoded_values: String(maxValues) }, coverage: coverage(rows, model), tensors: rows, interpretation_boundary: boundary() }, "weight_ir_sha256");
  await verifySourceAfterRead(sourceBytes, source.artifact_sha256, {signal});
  return validateWeightIr(document, model);
}
export function validateWeightIr(document, model) {
  const validated = sourceContract(model).model;
  canonicalJson(document);
  portableContract("semantic_validation", {operation:"weight",value:document,model:validated});
  return document;
}
