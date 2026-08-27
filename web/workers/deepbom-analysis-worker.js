let loadedModule = null;
let loadedKey = "";

function status(phase) {
  self.postMessage({ type: "status", phase });
}

self.addEventListener("message", async ({ data }) => {
  try {
    const { moduleUrl, wasmUrl, version, modelBytes, analysisJson } = data || {};
    if (!(modelBytes instanceof Uint8Array)) throw new Error("Worker model bytes are unavailable.");
    if (!moduleUrl || !wasmUrl) throw new Error("Protected module manifest is incomplete.");
    const key = `${moduleUrl}|${wasmUrl}|${version || "1"}`;
    if (!loadedModule || loadedKey !== key) {
      status("Initializing isolated deployment-artifact analyzer");
      const module = await import(`${moduleUrl}${moduleUrl.includes("?") ? "&" : "?"}v=${encodeURIComponent(version || "1")}`);
      await module.default({ module_or_path: `${wasmUrl}${wasmUrl.includes("?") ? "&" : "?"}v=${encodeURIComponent(version || "1")}` });
      loadedModule = module;
      loadedKey = key;
    }
    status("Decoding weight, quantization, and topology evidence off the main thread");
    const result = loadedModule.analyze_deepbom(modelBytes, analysisJson);
    self.postMessage({ type: "result", result });
  } catch (error) {
    self.postMessage({
      type: "error",
      error: { name: error?.name || "Error", message: error?.message || String(error) },
    });
  }
});
