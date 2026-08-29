import init, {
  analyze_tflite_for_target,
  compute_delegation_repair,
  compute_deployment_frontier,
} from "../../pkg/tflite_wasm_audit.js";
import { analyzeOnnxModel } from "../onnx.js";
import { analyzeExecuTorchModel } from "../executorch.js";
import { resolveExecuTorchSelectedBuildAttestation } from "../lib/executorch-build-binding.js";

let modelBytes = null;
let filename = "";
let wasmReady = null;

function status(id, phase) {
  self.postMessage({ id, type: "status", phase });
}

function ensureWasm(id) {
  if (!wasmReady) {
    status(id, "Initializing isolated TFLite analyzer");
    wasmReady = init({ module_or_path: new URL("../../pkg/tflite_wasm_audit_bg.wasm", import.meta.url) });
  }
  return wasmReady;
}

self.addEventListener("message", async ({ data }) => {
  const { id, operation, payload = {} } = data || {};
  try {
    if (operation === "load_model") {
      modelBytes = payload.bytes;
      filename = payload.filename || "model";
      self.postMessage({ id, type: "result", result: true });
      return;
    }
    if (!(modelBytes instanceof Uint8Array)) throw new Error("Static audit worker has no loaded model.");
    let result;
    if (operation === "tflite_analyze") {
      await ensureWasm(id);
      status(id, "Decoding FlatBuffer graph, tensors, and numerical contracts");
      result = analyze_tflite_for_target(modelBytes, filename, payload.targetId);
    } else if (operation === "tflite_frontier") {
      await ensureWasm(id);
      const count = JSON.parse(payload.targetIdsJson || "[]").length;
      status(id, `Modeling latency, memory, and cache across ${count} targets`);
      result = compute_deployment_frontier(modelBytes, filename, payload.targetIdsJson);
    } else if (operation === "tflite_delegation_repair") {
      await ensureWasm(id);
      status(id, "Enumerating delegate islands and repair scenarios");
      result = compute_delegation_repair(modelBytes, filename, payload.targetId);
    } else if (operation === "onnx_analyze") {
      status(id, "Decoding protobuf graph, initializers, and interface contracts");
      result = analyzeOnnxModel(modelBytes, filename, payload.targetProfile, {
        externalDataFiles: payload.externalDataFiles || [],
      });
    } else if (operation === "executorch_analyze") {
      status(id, "Validating ET12/FT01 tables, segments, tensors, source identities, and selected-build evidence");
      const selectedBuild = resolveExecuTorchSelectedBuildAttestation(payload.externalDataFiles || []);
      result = analyzeExecuTorchModel(modelBytes, filename, {
        externalDataFiles: payload.externalDataFiles || [],
        selectedBuildAttestation: selectedBuild?.attestation || null,
        selectedBuildInput: selectedBuild?.input || null,
      });
    } else {
      throw new Error(`Unknown static audit worker operation: ${operation}`);
    }
    self.postMessage({ id, type: "result", result });
  } catch (error) {
    self.postMessage({
      id,
      type: "error",
      error: { name: error?.name || "Error", message: error?.message || String(error) },
    });
  }
});
