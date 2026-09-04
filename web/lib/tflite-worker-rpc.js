import { STATIC_AUDIT_OPERATION } from "./static-audit-worker-protocol.js";

export function createTfliteWorkerRpc(client, { resolveTarget, getCurrentModel = () => ({}) }) {
  const run = (operation, {
    bytes = null,
    filename = "",
    targetId = null,
    onStatus = null,
    ...payload
  }) => {
    const current = getCurrentModel() || {};
    return client.run(operation, {
      bytes: bytes || current.bytes,
      filename: filename || current.filename || "model.tflite",
      ...(targetId == null ? {} : { targetId: resolveTarget(targetId) }),
      onStatus,
      ...payload,
    });
  };
  const bound = (operation, map) => (...args) => run(operation, map(...args));

  return Object.freeze({
    run,
    analyzeForTarget: bound(STATIC_AUDIT_OPERATION.TFLITE_ANALYZE,
      (bytes, filename, targetId) => ({ bytes, filename, targetId })),
    projectRedesign: bound(STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PROJECT,
      (bytes, filename, targetId, request) => ({ bytes, filename, targetId, request })),
    exploreRedesignPareto: bound(STATIC_AUDIT_OPERATION.TFLITE_REDESIGN_PARETO,
      (bytes, filename, targetId, request) => ({ bytes, filename, targetId, request })),
    inputInfluence: bound(STATIC_AUDIT_OPERATION.TFLITE_INPUT_INFLUENCE,
      (bytes, filename, opIndex, targetId) => ({ bytes, filename, targetId, opIndex })),
    outputInfluence: bound(STATIC_AUDIT_OPERATION.TFLITE_OUTPUT_INFLUENCE,
      (bytes, filename, opIndex, targetId) => ({ bytes, filename, targetId, opIndex })),
    runtimeCalibration: bound(STATIC_AUDIT_OPERATION.TFLITE_RUNTIME_CALIBRATION,
      (bytes, filename, targetId, measuredMs) => ({ bytes, filename, targetId, measuredMs })),
    weightHistogram: bound(STATIC_AUDIT_OPERATION.TFLITE_WEIGHT_HISTOGRAM,
      (bytes, filename, tensorIndex, targetId) => ({ bytes, filename, targetId, tensorIndex })),
    modelTomography: bound(STATIC_AUDIT_OPERATION.TFLITE_MODEL_TOMOGRAPHY,
      (bytes, filename, targetId) => ({ bytes, filename, targetId })),
    layerLandscape: bound(STATIC_AUDIT_OPERATION.TFLITE_LAYER_LANDSCAPE,
      (bytes, opIndex, seed1, seed2, gridSize, radius) => ({
        bytes,
        filename: "",
        opIndex,
        seed1,
        seed2,
        gridSize,
        radius,
      })),
  });
}
