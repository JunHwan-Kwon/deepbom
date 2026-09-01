import assert from "node:assert/strict";

import { createStaticAuditWorkerClient } from "../web/lib/static-audit-worker-client.js";

class FakeWorker {
  constructor(mode) {
    this.mode = mode;
    this.listeners = new Map();
    this.terminated = false;
  }

  addEventListener(type, listener) { this.listeners.set(type, listener); }

  postMessage(message) {
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
assert.throws(() => createStaticAuditWorkerClient({ inactivityTimeoutMs: 0 }), /positive safe integer/);

console.log("Static audit Worker client checks passed: inactivity timeout, termination, progress reset, and clean retry.");
