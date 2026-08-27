const WORKER_URL = new URL("../workers/static-audit-worker.js", import.meta.url);

export function createStaticAuditWorkerClient({ createWorker = null } = {}) {
  let worker = null;
  let sequence = 0;
  let loadedBytes = null;
  let loadedFilename = "";
  const pending = new Map();

  function ensureWorker() {
    if (worker) return worker;
    const factory = createWorker || (() => new Worker(WORKER_URL, { type: "module", name: "deepbom-static-audit" }));
    worker = factory();
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", (event) => failWorker(event.message || "Static audit worker failed."));
    worker.addEventListener("messageerror", () => failWorker("Static audit worker returned an unreadable message."));
    return worker;
  }

  function handleMessage({ data }) {
    const request = pending.get(data?.id);
    if (!request) return;
    if (data.type === "status") {
      request.onStatus?.(data.phase || "Analyzing artifact");
      return;
    }
    pending.delete(data.id);
    if (data.type === "result") request.resolve(data.result);
    else request.reject(new Error(data.error?.message || "Static audit worker failed."));
  }

  function failWorker(message) {
    for (const request of pending.values()) request.reject(new Error(message));
    pending.clear();
    worker?.terminate();
    worker = null;
    loadedBytes = null;
    loadedFilename = "";
  }

  function request(operation, payload = {}, onStatus = null) {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, onStatus });
      ensureWorker().postMessage({ id, operation, payload });
    });
  }

  async function ensureModel(bytes, filename, onStatus) {
    if (bytes === loadedBytes && filename === loadedFilename) return;
    await request("load_model", { bytes, filename }, onStatus);
    loadedBytes = bytes;
    loadedFilename = filename;
  }

  async function run(operation, { bytes, filename, onStatus = null, ...payload }) {
    await ensureModel(bytes, filename, onStatus);
    return request(operation, payload, onStatus);
  }

  function reset() {
    failWorker("Static audit worker was reset.");
  }

  return { reset, run };
}
