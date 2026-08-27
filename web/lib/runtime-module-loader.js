const LITERT_WASM_BASE_URL = "../node_modules/@litertjs/core/wasm/";

let liteRtCorePromise = null;
let onnxBenchmarkPromise = null;
let tfliteBenchmarkPromise = null;

export function loadLiteRtCore() {
  liteRtCorePromise ||= import("@litertjs/core");
  return liteRtCorePromise;
}

export function loadOnnxBenchmark() {
  onnxBenchmarkPromise ||= import("./onnx-benchmark.js").then((module) => {
    module.configureOnnxRuntime({
      mjs: new URL("../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs", import.meta.url),
      wasm: new URL("../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm", import.meta.url),
    });
    return module;
  });
  return onnxBenchmarkPromise;
}

export function loadTfliteBenchmark() {
  tfliteBenchmarkPromise ||= import("./tflite-benchmark.js");
  return tfliteBenchmarkPromise;
}

export function createLiteRtRuntimeLoader({
  onStatus = () => {},
  wasmBaseUrl = LITERT_WASM_BASE_URL,
} = {}) {
  let loadedMode = null;
  let loadPromise = null;

  async function ensure(backend) {
    const requestedMode = backend === "webnn" || backend === "webgpu" ? "jspi" : "default";
    if (loadPromise) {
      await loadPromise;
      return ensure(backend);
    }
    if (loadedMode === "jspi" || loadedMode === requestedMode) return;

    const operation = (async () => {
      const { loadLiteRt, unloadLiteRt } = await loadLiteRtCore();
      if (loadedMode) {
        unloadLiteRt();
        loadedMode = null;
      }
      onStatus(requestedMode === "jspi" ? "Loading LiteRT.js JSPI/Asyncify" : "Loading LiteRT.js");
      await loadLiteRt(wasmBaseUrl, requestedMode === "jspi" ? { jspi: true } : undefined);
      loadedMode = requestedMode;
    })();
    loadPromise = operation;
    try {
      await operation;
    } finally {
      if (loadPromise === operation) loadPromise = null;
    }
  }

  return Object.freeze({ ensure, loadCore: loadLiteRtCore });
}
