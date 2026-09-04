import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { performance } from "node:perf_hooks";

import { Sha256Accumulator } from "../web/lib/sha256-sync.js";
import { scanSerializedTensorPayloads } from "../web/lib/tensor-numerical-integrity.js";

const mib = integerArgument("--mib", 16, 1, 256);
const iterations = integerArgument("--iterations", 3, 1, 10);
const writeTarget = stringArgument("--write");
const payloadBytes = mib * 1024 * 1024;
const chunkBytes = 1024 * 1024;
const payload = new Uint8Array(payloadBytes);
for (let index = 0; index < payload.length; index += 1) payload[index] = (index * 31 + 17) & 0xff;

const nativeHash = measured(iterations, () => {
  const hash = createHash("sha256");
  for (let offset = 0; offset < payload.length; offset += chunkBytes) {
    hash.update(payload.subarray(offset, offset + chunkBytes));
  }
  return hash.digest("hex");
});
const javascriptHash = measured(iterations, () => {
  const hash = new Sha256Accumulator();
  for (let offset = 0; offset < payload.length; offset += chunkBytes) {
    hash.update(payload.subarray(offset, offset + chunkBytes));
  }
  return hash.digestHex();
});
assert.equal(javascriptHash.value, nativeHash.value, "incremental JavaScript SHA-256 must match the platform reference");

const headerPrefix = new Uint8Array(8);
const source = new Blob([headerPrefix, payload]);
const analysis = {
  format: "safetensors",
  filename: "streaming-benchmark.safetensors",
  file_size_bytes: source.size,
  safetensors: { header_byte_length: 0 },
  tensors: [{
    index: 0,
    name: "benchmark.weight",
    dtype: "F32",
    shape: [payloadBytes / 4],
    data_offset: 0,
    byte_length: payloadBytes,
  }],
};
const decodeRuns = [];
let decodeResult = null;
for (let iteration = 0; iteration < iterations; iteration += 1) {
  const started = performance.now();
  decodeResult = await scanSerializedTensorPayloads(source, analysis, { chunkBytes });
  decodeRuns.push(performance.now() - started);
}
assert.equal(decodeResult.assessed_tensor_bytes, payloadBytes, "decoder benchmark must assess the complete declared payload");
assert.equal(decodeResult.tensor_records[0].payload_sha256, nativeHash.value, "decoder payload digest must match the platform reference");

const result = {
  schema: "deepbom.streaming_wasm_candidate_benchmark.v1",
  purpose: "Screen current Worker-bound JavaScript hot paths before any streaming WASM port is proposed.",
  environment: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
  },
  input: { payload_bytes: payloadBytes, chunk_bytes: chunkBytes, iterations },
  incremental_sha256: metric(javascriptHash.samples, payloadBytes, {
    reference: "node:crypto SHA-256",
    reference_median_ms: median(nativeHash.samples),
    digest_match: true,
  }),
  safetensors_f32_decode_and_sha256: metric(decodeRuns, payloadBytes, {
    value_count: decodeResult.decoded_value_count,
    byte_conservation_status: decodeResult.byte_conservation_status,
    digest_match: true,
  }),
  decision: {
    current_execution_boundary: "dedicated metadata Worker",
    port_status: "not_selected_by_screening_alone",
    requirement: "A Rust/WASM prototype must preserve byte and value conservation, match all source-pinned decoder fixtures, and show repeatable browser throughput or memory improvement before replacing JavaScript.",
    independent_javascript_validator: "required",
  },
};

if (writeTarget) {
  const target = path.resolve(writeTarget);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, `${JSON.stringify({ ...result, generated_at: new Date().toISOString() }, null, 2)}\n`);
}
console.log(JSON.stringify(result, null, 2));

function measured(count, callback) {
  const samples = [];
  let value = null;
  for (let iteration = 0; iteration < count; iteration += 1) {
    const started = performance.now();
    value = callback();
    samples.push(performance.now() - started);
  }
  return { samples, value };
}

function metric(samples, bytes, extra = {}) {
  const medianMs = median(samples);
  return {
    median_ms: Number(medianMs.toFixed(3)),
    throughput_mib_per_second: Number(((bytes / 1024 / 1024) / (medianMs / 1000)).toFixed(3)),
    samples_ms: samples.map((value) => Number(value.toFixed(3))),
    ...extra,
  };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function integerArgument(name, fallback, minimum, maximum) {
  const raw = stringArgument(name);
  if (raw == null) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`);
  }
  return value;
}

function stringArgument(name) {
  const exact = process.argv.find((value) => value.startsWith(`${name}=`));
  if (exact) return exact.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}
