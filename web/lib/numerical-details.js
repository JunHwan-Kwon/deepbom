import { validateWeightIr } from "./weight-ir.js";
import { validateActivationIr } from "./activation-ir.js";
import { describeDistribution, compareDistributions } from "./numerical-ir/distribution.js";
import { sourceContract, seal, requireCondition } from "./numerical-ir/common.js";
import { canonicalJson } from "./report-utils.js";

export const NUMERICAL_DETAILS_SCHEMA = "deepbom.numerical_details.v1";
const BOUNDARY = "Derived from bound Weight IR and Activation IR. Quantiles are finite nearest-rank intervals, not interpolated percentiles. Flow lists serialized ports and capture coverage, not runtime scheduling or causality. Uncaptured values are not zeros. Run comparison describes imported observations, not training progress or task quality. Integer activation values are runtime codes; no affine dequantization is implied.";

// Validated by the builder; exported for the UI to show inputs and explicit
// gaps using exactly the same inventory as the downloadable projection.
export function activationInventory(model, activation) {
  if (!activation) return [];
  const values = new Map(model.program.values.map(v => [v.id, v]));
  const captured = (row, kind) => ({ ...row, name: values.get(row.value_ref)?.name || row.native_locator, status: "assessed", reason: null, capture_status: kind });
  const gap = (ref, kind, reason) => {
    const value = values.get(ref);
    return { id: `${kind}:${ref}`, value_ref: ref, native_locator: value?.name || ref, name: value?.name || ref, dtype: value?.dtype || "UNKNOWN", shape: value?.shape || null, statistics: null, status: "not_assessed", capture_status: kind, reason };
  };
  return [
    ...activation.tensors.map(row => captured(row, row.value_ref === null ? "unmapped_capture" : "captured")),
    ...activation.inputs.map(row => captured(row, "input")),
    ...activation.missing.map(row => gap(row.value_ref, "missing", row.reason)),
    ...activation.coverage.not_requested_value_refs.map(ref => gap(ref, "not_requested", "value_not_requested")),
  ];
}

function compareRuns(baseline, candidate) {
  const before = new Map(baseline.tensors.filter(r => r.value_ref !== null).map(r => [r.value_ref, r]));
  const after = new Map(candidate.tensors.filter(r => r.value_ref !== null).map(r => [r.value_ref, r]));
  const refs = [...new Set([...baseline.requested_value_refs, ...candidate.requested_value_refs])];
  const inputIdentity = doc => doc.inputs.map(row => ({ value_ref: row.value_ref, native_locator: row.value_ref === null ? row.native_locator : null, values_sha256: row.values_sha256 })).sort((a, b) => { const x = canonicalJson(a), y = canonicalJson(b); return x < y ? -1 : x > y ? 1 : 0; });
  const rows = refs.map(value_ref => {
    const a = before.get(value_ref), b = after.get(value_ref);
    const reason = !a ? "baseline_value_not_captured" : !b ? "candidate_value_not_captured" : a.dtype !== b.dtype ? "dtype_mismatch" : canonicalJson(a.shape) !== canonicalJson(b.shape) ? "shape_mismatch" : null;
    return { value_ref, baseline_ref: a?.id || null, candidate_ref: b?.id || null, ...(reason ? { status: "not_assessed", reason } : compareDistributions(a.statistics, b.statistics)) };
  });
  const assessed = rows.filter(row => row.status === "assessed").length;
  return {
    baseline_activation_ir_sha256: baseline.activation_ir_sha256, candidate_activation_ir_sha256: candidate.activation_ir_sha256,
    baseline_run_id: baseline.run.id, candidate_run_id: candidate.run.id,
    same_inputs: canonicalJson(inputIdentity(baseline)) === canonicalJson(inputIdentity(candidate)),
    same_runtime: canonicalJson(baseline.run.runtime) === canonicalJson(candidate.run.runtime),
    same_collector: canonicalJson(baseline.run.collector) === canonicalJson(candidate.run.collector),
    same_configuration: baseline.run.execution.configuration_sha256 === candidate.run.execution.configuration_sha256,
    same_instrumentation: baseline.run.execution.instrumented_artifact_sha256 === candidate.run.execution.instrumented_artifact_sha256,
    same_probe: canonicalJson(baseline.run.probe) === canonicalJson(candidate.run.probe),
    same_entry_region: baseline.run.entry_region_ref === candidate.run.entry_region_ref,
    coverage: { union_requested_count: refs.length, compared_count: assessed, not_assessed_count: refs.length - assessed, baseline_unmapped_count: baseline.coverage.unmapped_capture_count, candidate_unmapped_count: candidate.coverage.unmapped_capture_count },
    tensors: rows,
  };
}

export function buildNumericalDetails(modelIr, { weightIr = null, activationIr = null, baselineActivationIr = null } = {}) {
  const { model, source } = sourceContract(modelIr);
  requireCondition(weightIr || activationIr, "numerical details require weight or activation evidence");
  if (weightIr) validateWeightIr(weightIr, model);
  if (activationIr) validateActivationIr(activationIr, model);
  if (baselineActivationIr) { requireCondition(activationIr, "baseline capture requires a candidate capture"); validateActivationIr(baselineActivationIr, model); }
  const inventory = activationInventory(model, activationIr);
  const rowsByValue = new Map(inventory.filter(row => row.value_ref !== null).map(row => [row.value_ref, row]));
  const values = new Map(model.program.values.map(v => [v.id, v]));
  const portsByOp = new Map();
  for (const port of model.program.ports) {
    if (!portsByOp.has(port.operation_ref)) portsByOp.set(port.operation_ref, []);
    portsByOp.get(port.operation_ref).push(port);
  }
  const flow = activationIr ? model.program.operations.map(op => ({
    operation_ref: op.id, name: op.name || op.native_op?.name || op.id, native_index: op.native_index ?? null,
    ports: (portsByOp.get(op.id) || []).map(port => {
      const row = rowsByValue.get(port.value_ref), value = values.get(port.value_ref);
      return { port_ref: port.id, direction: port.direction, position: port.position, value_ref: port.value_ref, tensor_ref: row?.id || null, capture_status: row?.capture_status || (value?.storage_refs.length ? "stored_value" : "not_captured"), reason: row?.reason || null };
    }),
  })) : [];
  const profile = row => row.statistics ? describeDistribution(row.statistics) : null;
  return seal({
    schema: NUMERICAL_DETAILS_SCHEMA, method_version: "1.0.0", source,
    weight_ir_sha256: weightIr?.weight_ir_sha256 || null, activation_ir_sha256: activationIr?.activation_ir_sha256 || null,
    weights: (weightIr?.tensors || []).map(row => ({ weight_ref: row.id, storage_ref: row.storage_ref, status: row.status, reason: row.reason, representation: row.representation, distribution: profile(row) })),
    activation: activationIr ? {
      run: structuredClone(activationIr.run), coverage: structuredClone(activationIr.coverage),
      tensors: inventory.map(row => ({ tensor_ref: row.id, value_ref: row.value_ref, capture_status: row.capture_status, status: row.status, reason: row.reason, distribution: profile(row) })),
      flow,
    } : null,
    activation_comparison: baselineActivationIr ? compareRuns(baselineActivationIr, activationIr) : null,
    interpretation_boundary: BOUNDARY,
  }, "numerical_details_sha256");
}

export function validateNumericalDetails(document, model, evidence) {
  const expected = buildNumericalDetails(model, evidence);
  requireCondition(canonicalJson(document) === canonicalJson(expected), "numerical details differ from bound source evidence");
  return document;
}
