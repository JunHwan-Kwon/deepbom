import { STATIC_AUDIT_OPERATION } from "./static-audit-worker-protocol.js";

const WORKER_URL = new URL("../workers/static-audit-worker.js", import.meta.url);

export function createStaticAuditWorkerClient({ createWorker = null, inactivityTimeoutMs = 300_000 } = {}) {
  if (!Number.isSafeInteger(inactivityTimeoutMs) || inactivityTimeoutMs <= 0) {
    throw new Error("Static audit Worker inactivity timeout must be a positive safe integer.");
  }
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
      armTimeout(data.id, request);
      request.onStatus?.(data.phase || "Analyzing artifact", data.progress || null);
      return;
    }
    pending.delete(data.id);
    clearTimeout(request.timeoutId);
    if (data.type === "result") request.resolve(data.result);
    else request.reject(new Error(data.error?.message || "Static audit worker failed."));
  }

  function failWorker(message) {
    for (const request of pending.values()) {
      clearTimeout(request.timeoutId);
      request.reject(new Error(message));
    }
    pending.clear();
    worker?.terminate();
    worker = null;
    loadedBytes = null;
    loadedFilename = "";
  }

  function request(operation, payload = {}, onStatus = null) {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const state = { resolve, reject, onStatus, operation, timeoutId: null };
      pending.set(id, state);
      armTimeout(id, state);
      ensureWorker().postMessage({ id, operation, payload });
    });
  }

  function armTimeout(id, state) {
    clearTimeout(state.timeoutId);
    state.timeoutId = setTimeout(() => {
      if (!pending.has(id)) return;
      failWorker(`Static audit worker timed out after ${inactivityTimeoutMs} ms without progress during ${state.operation}.`);
    }, inactivityTimeoutMs);
  }

  async function ensureModel(bytes, filename, onStatus) {
    if (bytes === loadedBytes && filename === loadedFilename) return;
    await request(STATIC_AUDIT_OPERATION.LOAD_MODEL, { bytes, filename }, onStatus);
    loadedBytes = bytes;
    loadedFilename = filename;
  }

  async function run(operation, { bytes, filename, onStatus = null, ...payload }) {
    if (!(bytes instanceof Uint8Array)) {
      throw new Error("Static audit Worker model operations require Uint8Array bytes.");
    }
    await ensureModel(bytes, filename, onStatus);
    return request(operation, payload, onStatus);
  }

  function runFile(operation, { file = null, files = null, onStatus = null, ...payload }) {
    const hasFile = Boolean(file && typeof file.slice === "function" && Number.isFinite(file.size));
    const hasFiles = Array.isArray(files) && files.length > 0
      && files.every((item) => item && typeof item.slice === "function" && Number.isFinite(item.size));
    if (hasFile === hasFiles) {
      throw new Error("Static audit Worker file operations require exactly one file or one non-empty file array.");
    }
    const filePayload = hasFile
      ? { file }
      : {
          files: files.map((item) => ({
            file: item,
            path: String(item.webkitRelativePath || item.name || ""),
          })),
        };
    return request(operation, { ...payload, ...filePayload }, onStatus);
  }

  function reset() {
    failWorker("Static audit worker was reset.");
  }

  return { reset, run, runFile };
}
