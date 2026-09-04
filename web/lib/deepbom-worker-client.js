import { browserAssetUrl } from "./browser-asset-url.js";

const WORKER_URL = browserAssetUrl("./workers/deepbom-analysis-worker.js", "../workers/deepbom-analysis-worker.js", import.meta.url);

export function analyzeDeepBomInWorker({ manifest, modelBytes, analysis, onStatus = null }) {
  if (typeof Worker !== "function") {
    return Promise.reject(new Error("This browser does not provide the Worker API required for isolated analysis."));
  }
  const moduleUrl = new URL(manifest.module_url, globalThis.location?.href || import.meta.url).href;
  const wasmUrl = new URL(manifest.wasm_url, globalThis.location?.href || import.meta.url).href;
  const transferableBytes = new Uint8Array(modelBytes);
  const worker = new Worker(WORKER_URL, { type: "module", name: "deepbom-protected-analysis" });
  return new Promise((resolve, reject) => {
    const finish = (callback, value) => {
      worker.terminate();
      callback(value);
    };
    worker.addEventListener("message", ({ data }) => {
      if (data?.type === "status") {
        onStatus?.(data.phase || "Analyzing deployment artifact");
        return;
      }
      if (data?.type === "result") {
        finish(resolve, data.result);
        return;
      }
      finish(reject, new Error(data?.error?.message || "Isolated DEEPBOM analysis failed."));
    });
    worker.addEventListener("error", (event) => {
      finish(reject, new Error(event.message || "Isolated DEEPBOM analysis worker failed."));
    }, { once: true });
    worker.addEventListener("messageerror", () => {
      finish(reject, new Error("Isolated DEEPBOM analysis returned an unreadable result."));
    }, { once: true });
    worker.postMessage({
      moduleUrl,
      wasmUrl,
      version: manifest.version || "1",
      modelBytes: transferableBytes,
      analysisJson: JSON.stringify(analysis),
    }, [transferableBytes.buffer]);
  });
}
