#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const output = path.resolve(process.argv[2] || "corpus/external-review/fixtures");
const TYPES = Object.freeze([
  { name: "F16", code: 1, elements: 32, bytes: 64, quantized: false },
  { name: "Q4_0", code: 2, elements: 32, bytes: 18, quantized: true },
  { name: "Q8_0", code: 8, elements: 32, bytes: 34, quantized: true },
]);

class Writer {
  constructor() { this.bytes = []; }
  raw(values) { this.bytes.push(...values); }
  u32(value) { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setUint32(0, value, true); this.raw(bytes); }
  u64(value) { const bytes = new Uint8Array(8); new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true); this.raw(bytes); }
  text(value) { const bytes = new TextEncoder().encode(value); this.u64(bytes.length); this.raw(bytes); }
  finish() { return Uint8Array.from(this.bytes); }
}

await mkdir(output, { recursive: true });
const fixtures = [];
for (const type of TYPES) {
  const bytes = buildGguf(type);
  const filename = `gguf-${type.name.toLowerCase().replaceAll("_", "-")}.gguf`;
  await writeFile(path.join(output, filename), bytes);
  fixtures.push({
    path: filename,
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    expected: {
      dtype: type.name,
      tensor_count: 1,
      block_quantized_tensor_count: type.quantized ? 1 : 0,
      quantization_classification: type.quantized ? "format_defined_block_quantized_weights" : "not_quantized_storage_encoding",
      storage_precision_classification: type.quantized ? "block_quantized_storage" : "reduced_precision_floating_point_storage",
    },
  });
}

const manifest = {
  schema: "deepbom.external_review_gguf_fixtures.v1",
  generator: "deterministic_node_binary_writer",
  source: {
    repository: "ggml-org/llama.cpp",
    commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
    path: "ggml/src/ggml.c",
    sha256: "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4",
  },
  fixtures,
};
await writeFile(path.join(output, "gguf-storage-classes.manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);

function buildGguf(type) {
  const writer = new Writer();
  writer.raw([0x47, 0x47, 0x55, 0x46]);
  writer.u32(3);
  writer.u64(1);
  writer.u64(1);
  writer.text("general.architecture");
  writer.u32(8);
  writer.text("test");
  writer.text("weight");
  writer.u32(1);
  writer.u64(type.elements);
  writer.u32(type.code);
  writer.u64(0);
  const header = writer.finish();
  const payloadOffset = Math.ceil(header.length / 32) * 32;
  const result = new Uint8Array(payloadOffset + type.bytes);
  result.set(header);
  return result;
}
