// A consumer of Model/Weight IR, not a tensor decoder or another statistics engine.
import { validateModelIr } from "./model-ir.js";
import { validateWeightIr } from "./weight-ir.js";
import { validateWeightAnalysis } from "./weight-analysis.js";

export function operatorWeightSelection(model, nativeIndex) {
  validateModelIr(model);
  const entries = new Set(model.program.programs.flatMap(program => program.entry_region_refs));
  const matches = model.program.operations.filter(op => entries.has(op.region_ref) && op.native_index === nativeIndex);
  if (matches.length !== 1) throw Error("No unambiguous Model IR operation for this selection.");
  const operation = matches[0];
  const storage = new Set(model.weight_bindings.bindings
    .filter(binding => binding.operation_ref === operation.id).map(binding => binding.storage_ref));
  return { operation, storage };
}

export function operatorWeightView(model, nativeIndex, evidence) {
  const { operation, storage } = operatorWeightSelection(model, nativeIndex);
  const weight = validateWeightIr(evidence.weight_ir, model);
  const advanced = evidence.weight_analysis
    ? validateWeightAnalysis(evidence.weight_analysis, weight, model) : null;
  const byRef = new Map(advanced?.tensors.map(row => [row.weight_ref, row]) || []);
  return {
    source: weight.source, operation_ref: operation.id,
    weight_ir_sha256: weight.weight_ir_sha256,
    weight_analysis_sha256: advanced?.weight_analysis_sha256 ?? null,
    coverage: weight.coverage,
    rows: weight.tensors.filter(row => storage.has(row.storage_ref))
      .map(tensor => ({ tensor, advanced: byRef.get(tensor.id) ?? null })),
  };
}

// Keep one artifact's basic evidence and at most eight operation projections.
// The worker still validates the source bytes and IR for every advanced request.
export function createOperatorWeightReader({ getContext, runWeight }) {
  let current = null;
  return async nativeIndex => {
    const { model, analysis, source } = getContext();
    if (!model || !source) throw Error("Model IR and the original payload are required for weight evidence.");
    const selection = operatorWeightSelection(model, nativeIndex);
    if (!selection.storage.size) return { operation_ref: selection.operation.id, rows: [] };
    if (!current || current.model !== model || current.source !== source) {
      const file = source instanceof Uint8Array ? new Blob([source]) : source;
      const entry = { model, source, file, views: new Map() };
      entry.weight = Promise.resolve().then(() => runWeight({ model, analysis, file, advanced: false }))
        .then(weight => validateWeightIr(weight, model)).catch(error => {
          if (current === entry) current = null;
          throw error;
        });
      current = entry;
    }
    const entry = current, key = selection.operation.id;
    if (!entry.views.has(key)) {
      const result = entry.weight.then(async weight => {
        const tensor_ids = weight.tensors.filter(row => selection.storage.has(row.storage_ref)).map(row => row.id);
        if (!tensor_ids.length) return operatorWeightView(model, nativeIndex, { weight_ir: weight });
        const evidence = await runWeight({ model, analysis, file: entry.file, weightIr: weight,
          advanced: true, options: { tensor_ids } });
        return operatorWeightView(model, nativeIndex, evidence);
      }).catch(error => { if (entry.views.get(key) === result) entry.views.delete(key); throw error; });
      entry.views.set(key, result);
      if (entry.views.size > 8) entry.views.delete(entry.views.keys().next().value);
    }
    return entry.views.get(key);
  };
}
