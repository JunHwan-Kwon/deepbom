import { METHOD_VERSION, bindSource, checkDigest, decimalCount, exactKeys, requireCondition, seal, shapeCount, sourceContract, SHA256 } from "./numerical-ir/common.js";
import { TensorStatistics, validateStatistics } from "./numerical-ir/statistics.js";
import { hashSource, numericSources } from "./numerical-ir/weight-sources.js";
import { canonicalJson } from "./report-utils.js";

export const WEIGHT_IR_SCHEMA = "deepbom.weight_ir.v1";
const BOUNDARY = "Statistics describe serialized numeric payloads, not trainability, accuracy or execution. Integer codes are not dequantized unless representation explicitly says dequantized_real. Shared storage may have multiple uses; tensor views are not unique physical allocations. Unmapped payloads do not establish layers. Floating moments are rounded binary64; counts are exact decimal integers. No sampling is performed.";
function coverage(rows, model) {
  return { model_storage_count: model.tensors_and_storage.storage_objects.length, inventory_count: rows.length,
    assessed_count: rows.filter(r => r.status === "assessed").length, not_assessed_count: rows.filter(r => r.status === "not_assessed").length,
    unlinked_count: rows.filter(r => r.storage_ref === null).length,
    decoded_value_count: String(rows.reduce((n,r)=>n+BigInt(r.statistics?.value_count || "0"),0n)),
    scope: "all_model_storage_objects_plus_decoder_exposed_payloads", completeness: rows.every(r => r.status === "assessed") ? "complete_within_inventory" : "partial_within_inventory" };
}
export async function buildWeightIr(modelIr, analysis, sourceBytes, { maxValues = 100_000_000, signal, onProgress } = {}) {
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
    const assessed = result.status === "assessed";
    rows.push({ id: `weight:${index}`, storage_ref: candidate.storage_ref, native_locator: candidate.locator, name: String(candidate.name || ""), dtype: String(candidate.dtype || "UNKNOWN"),
      shape: exactShape ? candidate.shape : null, status: assessed ? "assessed" : "not_assessed", reason: assessed ? null : String(result.reason || "decoder_unavailable"),
      representation: assessed ? result.representation : null, payload_sha256: result.payload_sha256 || null,
      invalid_encoding_count: assessed ? String(result.invalid_encoding_count || 0) : null,
      binding_refs: model.weight_bindings.bindings.filter(b => b.storage_ref === candidate.storage_ref).map(b => b.id),
      statistics: assessed ? result.statistics : null });
    onProgress?.({ phase: "weights", completed: index + 1, total: candidates.length });
  }
  const document = seal({ schema: WEIGHT_IR_SCHEMA, method_version: METHOD_VERSION, source, budget: { maximum_decoded_values: String(maxValues) }, coverage: coverage(rows, model), tensors: rows, interpretation_boundary: BOUNDARY }, "weight_ir_sha256");
  return validateWeightIr(document, model);
}
export function validateWeightIr(document, model) {
  exactKeys(document,["schema","method_version","source","budget","coverage","tensors","interpretation_boundary","weight_ir_sha256"],"Weight IR");
  requireCondition(document.schema===WEIGHT_IR_SCHEMA && document.method_version===METHOD_VERSION && document.interpretation_boundary===BOUNDARY,"unsupported Weight IR contract");
  bindSource(document.source,model); checkDigest(document,"weight_ir_sha256");
  exactKeys(document.budget,["maximum_decoded_values"],"weight budget"); const budget=decimalCount(document.budget.maximum_decoded_values);
  requireCondition(Array.isArray(document.tensors),"missing weight inventory");
  const stores=new Map(model.tensors_and_storage.storage_objects.map(s=>[s.id,s])), refs=new Set();
  for (const [index,row] of document.tensors.entries()) {
    exactKeys(row,["id","storage_ref","native_locator","name","dtype","shape","status","reason","representation","payload_sha256","invalid_encoding_count","binding_refs","statistics"],"weight tensor");
    requireCondition(row.id===`weight:${index}` && typeof row.native_locator==="string" && row.native_locator.length>0 && typeof row.name==="string" && typeof row.dtype==="string","invalid weight identity");
    requireCondition(row.storage_ref===null || (stores.has(row.storage_ref) && !refs.has(row.storage_ref)),"invalid or duplicate storage reference");
    if(row.storage_ref) {
      refs.add(row.storage_ref); const stored=stores.get(row.storage_ref);
      requireCondition(row.dtype===stored.dtype,"weight dtype contradicts Model IR");
      if(stored.shape.length) requireCondition(canonicalJson(row.shape)===canonicalJson(stored.shape),"weight shape contradicts Model IR");
    }
    requireCondition(canonicalJson(row.binding_refs)===canonicalJson(model.weight_bindings.bindings.filter(b=>b.storage_ref===row.storage_ref).map(b=>b.id)),"weight bindings do not match Model IR");
    requireCondition(row.payload_sha256===null || SHA256.test(row.payload_sha256),"invalid payload hash");
    const expected=row.shape===null ? null : shapeCount(row.shape);
    if(row.status==="assessed") {
      requireCondition(row.reason===null && ["stored_scalar","dequantized_real","complex_magnitude"].includes(row.representation),"invalid assessed representation");
      validateStatistics(row.statistics); requireCondition(BigInt(row.statistics.value_count)===expected,"weight shape/count mismatch"); decimalCount(row.invalid_encoding_count);
    } else requireCondition(row.status==="not_assessed" && typeof row.reason==="string" && row.reason.length>0 && row.statistics===null && row.representation===null && row.invalid_encoding_count===null,"unassessed weight has unsupported statistics");
  }
  requireCondition(refs.size===stores.size,"Weight IR omitted Model IR storage objects");
  requireCondition(canonicalJson(document.coverage)===canonicalJson(coverage(document.tensors,model)),"weight coverage ledger mismatch");
  requireCondition(decimalCount(document.coverage.decoded_value_count)<=budget,"decoded values exceed budget"); return document;
}
