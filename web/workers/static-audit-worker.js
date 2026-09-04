import init, {
  analyze_tflite_for_target,
  compute_activation_haar,
  compute_delegation_repair,
  compute_deployment_delta,
  compute_deployment_frontier,
  compute_input_influence,
  compute_kernel_haar_decomposition,
  compute_model_tomography,
  compute_output_influence,
  compute_static_runtime_calibration,
  compute_weight_histogram,
  explore_tflite_redesign_pareto,
  landscape_directions,
  landscape_tomography,
  layer_landscape_grid,
  project_tflite_redesign,
  synthetic_landscape_grid,
} from "../../pkg/tflite_wasm_audit.js";
import { analyzeOnnxModel } from "../onnx.js";
import { analyzeExecuTorchModel } from "../executorch.js";
import { resolveExecuTorchSelectedBuildAttestation } from "../lib/executorch-build-binding.js";
import {
  FILE_SCOPED_STATIC_AUDIT_OPERATIONS,
  STATIC_AUDIT_OPERATION,
} from "../lib/static-audit-worker-protocol.js";

let modelBytes = null;
let filename = "";
let wasmReady = null;

function status(id, phase, progress = null) {
  self.postMessage({ id, type: "status", phase, ...(progress ? { progress } : {}) });
}

function ensureWasm(id) {
  if (!wasmReady) {
    status(id, "Initializing isolated TFLite analyzer");
    wasmReady = init({ module_or_path: new URL("../../pkg/tflite_wasm_audit_bg.wasm", import.meta.url) });
  }
  return wasmReady;
}

function metadataProgress(id, progress = {}) {
  const index = Number(progress.index);
  const count = Number(progress.count);
  const position = Number.isSafeInteger(index) && Number.isSafeInteger(count) && count > 0
    ? ` ${index + 1}/${count}`
    : "";
  status(id, `${progress.phase || "Reading metadata payload"}${position}`, progress);
}

async function runFileScopedOperation(id, operation, payload) {
  if (operation === STATIC_AUDIT_OPERATION.ARTIFACT_BUNDLE_ANALYZE) {
    const { readArtifactBundle } = await import("../lib/artifact-bundle.js");
    status(id, "Resolving artifact package and range-bound payloads");
    return readArtifactBundle(payload.files, {
      scanMode: payload.scanMode || "full",
      onProgress: (progress) => metadataProgress(id, progress),
    });
  }
  if (operation === STATIC_AUDIT_OPERATION.METADATA_ANALYZE) {
    const { readMetadataModelFile } = await import("../lib/metadata-model-adapters.js");
    status(id, "Reading container header and tensor directory");
    return readMetadataModelFile(payload.file, payload.format, {
      scanMode: payload.scanMode || "full",
      onProgress: (progress) => metadataProgress(id, progress),
    });
  }
  if (operation === STATIC_AUDIT_OPERATION.COREML_ANALYZE) {
    const { readCoreMlModelFile } = await import("../lib/coreml-metadata-adapter.js");
    status(id, "Range-decoding Core ML protobuf records");
    return readCoreMlModelFile(payload.file);
  }
  return null;
}

self.addEventListener("message", async ({ data }) => {
  const { id, operation, payload = {} } = data || {};
  try {
    if (operation === STATIC_AUDIT_OPERATION.LOAD_MODEL) {
      modelBytes = payload.bytes;
      filename = payload.filename || "model";
      self.postMessage({ id, type: "result", result: true });
      return;
    }
    if (FILE_SCOPED_STATIC_AUDIT_OPERATIONS.has(operation)) {
      const result = await runFileScopedOperation(id, operation, payload);
      self.postMessage({ id, type: "result", result });
      return;
    }
    if (!(modelBytes instanceof Uint8Array)) throw new Error("Static audit worker has no loaded model.");
    const requestModelBytes = modelBytes;
    const requestFilename = filename;
    let result;
    if (operation === STATIC_AUDIT_OPERATION.TFLITE_ANALYZE) {
      await ensureWasm(id);
      status(id, "Decoding FlatBuffer graph, tensors, and numerical contracts");
      result = analyze_tflite_for_target(requestModelBytes, requestFilename, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_FRONTIER) {
      await ensureWasm(id);
      const count = JSON.parse(payload.targetIdsJson || "[]").length;
      status(id, `Modeling latency, memory, and cache across ${count} targets`);
      result = compute_deployment_frontier(requestModelBytes, requestFilename, payload.targetIdsJson);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_DELEGATION_REPAIR) {
      await ensureWasm(id);
      status(id, "Enumerating delegate islands and repair scenarios");
      result = compute_delegation_repair(requestModelBytes, requestFilename, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_DEPLOYMENT_DELTA) {
      await ensureWasm(id);
      status(id, "Comparing baseline and candidate deployment projections");
      result = compute_deployment_delta(
        payload.baselineBytes,
        payload.baselineFilename,
        requestModelBytes,
        requestFilename,
        payload.targetIdsJson,
      );
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_INPUT_INFLUENCE) {
      await ensureWasm(id);
      status(id, "Tracing input influence through the selected operator");
      result = compute_input_influence(requestModelBytes, requestFilename, payload.opIndex, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_OUTPUT_INFLUENCE) {
      await ensureWasm(id);
      status(id, "Tracing selected-operator output influence");
      result = compute_output_influence(requestModelBytes, requestFilename, payload.opIndex, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_RUNTIME_CALIBRATION) {
      await ensureWasm(id);
      status(id, "Binding measured runtime to the static projection");
      result = compute_static_runtime_calibration(requestModelBytes, requestFilename, payload.targetId, payload.measuredMs);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_WEIGHT_HISTOGRAM) {
      await ensureWasm(id);
      status(id, "Decoding the selected constant tensor");
      result = compute_weight_histogram(requestModelBytes, requestFilename, payload.tensorIndex, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PROJECT) {
      await ensureWasm(id);
      status(id, "Projecting the requested artifact redesign");
      result = project_tflite_redesign(requestModelBytes, requestFilename, payload.targetId, payload.request);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PARETO) {
      await ensureWasm(id);
      status(id, "Exploring the deterministic redesign frontier");
      result = explore_tflite_redesign_pareto(requestModelBytes, requestFilename, payload.targetId, payload.request);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_MODEL_TOMOGRAPHY) {
      await ensureWasm(id);
      status(id, "Computing model tomography");
      result = compute_model_tomography(requestModelBytes, requestFilename, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_LAYER_LANDSCAPE) {
      await ensureWasm(id);
      status(id, "Computing the selected layer landscape");
      result = layer_landscape_grid(
        requestModelBytes,
        payload.opIndex,
        payload.seed1,
        payload.seed2,
        payload.gridSize,
        payload.radius,
      );
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_LANDSCAPE_TOMOGRAPHY) {
      await ensureWasm(id);
      status(id, "Computing landscape tomography");
      result = landscape_tomography(requestModelBytes, payload.numProjections, payload.gridSize, payload.radius);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_SYNTHETIC_LANDSCAPE) {
      await ensureWasm(id);
      status(id, "Computing the synthetic float landscape");
      result = synthetic_landscape_grid(requestModelBytes, payload.seed1, payload.seed2, payload.gridSize, payload.radius);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_LANDSCAPE_DIRECTIONS) {
      await ensureWasm(id);
      status(id, "Deriving deterministic landscape directions");
      result = landscape_directions(requestModelBytes, payload.seed1, payload.seed2);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_KERNEL_HAAR) {
      await ensureWasm(id);
      status(id, "Computing static kernel Haar evidence");
      result = compute_kernel_haar_decomposition(requestModelBytes, requestFilename, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.TFLITE_ACTIVATION_HAAR) {
      await ensureWasm(id);
      status(id, "Computing synthetic activation Haar evidence");
      result = compute_activation_haar(requestModelBytes, requestFilename, payload.targetId);
    } else if (operation === STATIC_AUDIT_OPERATION.ONNX_ANALYZE) {
      status(id, "Decoding protobuf graph, initializers, and interface contracts");
      result = analyzeOnnxModel(requestModelBytes, requestFilename, payload.targetProfile, {
        externalDataFiles: payload.externalDataFiles || [],
      });
    } else if (operation === STATIC_AUDIT_OPERATION.EXECUTORCH_ANALYZE) {
      status(id, "Validating ET12/FT01 tables, segments, tensors, source identities, and selected-build evidence");
      const selectedBuild = resolveExecuTorchSelectedBuildAttestation(payload.externalDataFiles || []);
      result = analyzeExecuTorchModel(requestModelBytes, requestFilename, {
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
