import { collectWeightAnalysis } from "../lib/weight-analysis.js";
import { prepareMagnitudePruning, simulateMagnitudePruning } from "../lib/weight-pruning.js";

let prepared = null;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "initialize") {
      const { model, analysis, source, weight, weightRef, options = {} } = data;
      const result = await collectWeightAnalysis(model, analysis, source, {
        weightIr: weight,
        options: { ...options, tensor_ids: [weightRef], max_tensor_values: Math.min(options.max_tensor_values ?? 2_000_000, 2_000_000), max_channels: 0, max_pairwise_channels: 0, max_svd_work: 0 },
      });
      const decoded = result.decoded.get(weightRef);
      if (!decoded) throw new Error(result.weight_analysis.tensors.find(t => t.weight_ref === weightRef)?.reason || "Complete tensor values unavailable");
      prepared = prepareMagnitudePruning({ ...decoded, source: result.weight_ir.source, weight_ir_sha256: result.weight_ir.weight_ir_sha256 });
      self.postMessage({ type: "ready" });
    } else if (data.type === "preview") {
      if (!prepared) throw new Error("Pruning values have not been prepared");
      self.postMessage({ type: "preview", request: data.request, result: simulateMagnitudePruning(prepared, data.settings) });
    }
  } catch (error) { self.postMessage({ type: "error", request: data.request, error: String(error?.message || error) }); }
};
