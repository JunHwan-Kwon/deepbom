import assert from "node:assert/strict";

import { createStaticAuditWorkerClient } from "../web/lib/static-audit-worker-client.js";

class FakeWorker {
  constructor(mode) {
    this.mode = mode;
    this.listeners = new Map();
    this.messages = [];
    this.terminated = false;
  }

  addEventListener(type, listener) { this.listeners.set(type, listener); }

  postMessage(message) {
    this.messages.push(message);
    if (this.mode === "hang") return;
    const emit = (data, delay) => setTimeout(() => {
      if (!this.terminated) this.listeners.get("message")?.({ data });
    }, delay);
    if (message.operation === "load_model") {
      emit({ id: message.id, type: "status", phase: "loaded" }, 2);
      emit({ id: message.id, type: "result", result: { loaded: true } }, 7);
    } else {
      emit({ id: message.id, type: "status", phase: "analyzing" }, 2);
      emit({ id: message.id, type: "result", result: { status: "ok" } }, 7);
    }
  }

  terminate() { this.terminated = true; }
}

const workers = [];
const createWorker = () => {
  const worker = new FakeWorker(workers.length === 0 ? "hang" : "success");
  workers.push(worker);
  return worker;
};
const client = createStaticAuditWorkerClient({ createWorker, inactivityTimeoutMs: 25 });

await assert.rejects(
  client.run("analyze", { bytes: new Uint8Array([1]), filename: "fixture.tflite" }),
  /timed out.*load_model/i,
  "a silent Worker must fail with an operation-specific timeout",
);
assert.equal(workers[0].terminated, true, "the timed-out Worker must be terminated");

const statuses = [];
const result = await client.run("analyze", {
  bytes: new Uint8Array([1]),
  filename: "fixture.tflite",
  onStatus: (status) => statuses.push(status),
});
assert.deepEqual(result, { status: "ok" }, "the next run must use a clean Worker and complete");
assert.deepEqual(statuses, ["loaded", "analyzing"], "progress messages must remain observable");
assert.equal(workers.length, 2, "timeout recovery must create exactly one replacement Worker");

client.reset();
assert.equal(workers[1].terminated, true, "reset must terminate the active Worker");

const fileResult = await client.runFile("metadata_analyze", {
  file: { name: "fixture.gguf", size: 16, slice() {} },
  format: "gguf",
});
assert.deepEqual(fileResult, { status: "ok" }, "file-scoped analysis must not require model state");
assert.equal(workers.length, 3, "file-scoped analysis must recover on a fresh Worker");
const packageFiles = [
  { name: "model.mlmodel", size: 12, slice() {}, webkitRelativePath: "Fixture.mlpackage/Data/model.mlmodel" },
  { name: "weights.bin", size: 24, slice() {}, webkitRelativePath: "Fixture.mlpackage/Data/weights/weights.bin" },
];
await client.runFile("artifact_bundle_analyze", { files: packageFiles });
assert.deepEqual(
  workers[2].messages.at(-1).payload.files.map(({ path }) => path),
  packageFiles.map(({ webkitRelativePath }) => webkitRelativePath),
  "package-relative paths must survive the Worker RPC descriptor boundary",
);
assert.throws(
  () => client.runFile("metadata_analyze", {}),
  /exactly one file or one non-empty file array/,
);
await assert.rejects(
  client.run("analyze", { bytes: new ArrayBuffer(1), filename: "fixture.tflite" }),
  /Uint8Array bytes/,
);
client.reset();
assert.throws(() => createStaticAuditWorkerClient({ inactivityTimeoutMs: 0 }), /positive safe integer/);

console.log("Static audit Worker client checks passed: model/file isolation, validation, timeout termination, progress reset, and clean retry.");
