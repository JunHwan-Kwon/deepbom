// Negative / error-path test suite for every advertised parser. Each case feeds
// a malformed or hostile
// artifact and classifies the observed behaviour.
//
//   rejected  - threw a domain Error (desired fail-closed behaviour)
//   accepted  - returned an analysis (suspicious when the input is invalid)
//   crash     - threw TypeError/RangeError/non-Error (unhandled internal fault)
//   timeout   - exceeded the per-case time budget
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { parseMetadataModel, readMetadataModelFile, parseStrictJson } = await import(pathToFileURL(path.join(ROOT, "web/lib/metadata-model-adapters.js")).href);
const { parseCoreMlModel, readCoreMlModelFile } = await import(pathToFileURL(path.join(ROOT, "web/lib/coreml-metadata-adapter.js")).href);
const { readArtifactBundle, inspectArtifactBundle } = await import(pathToFileURL(path.join(ROOT, "web/lib/artifact-bundle.js")).href);

const CASE_TIMEOUT_MS = 8000;
const results = [];
const descriptors = [];
const isolationMode = String(process.env.DEEPBOM_PARSER_ROBUSTNESS_MODE || "standalone");
const selectedCaseId = String(process.env.DEEPBOM_PARSER_ROBUSTNESS_CASE || "");

function caseId(group, name) {
  return `${group}:${name}`;
}

function selected(group, name, expectation) {
  const id = caseId(group, name);
  descriptors.push({ id, group, name, expectation, timeout_ms: group === "fuzz" ? 120000 : 15000 });
  if (isolationMode === "list") return false;
  if (isolationMode === "case") return id === selectedCaseId;
  return true;
}

function file(name, bytes, relativePath) {
  const blob = new File([bytes], name);
  if (relativePath) Object.defineProperty(blob, "webkitRelativePath", { value: relativePath });
  return blob;
}

async function run(group, name, expectation, thunk) {
  if (!selected(group, name, expectation)) return null;
  let timer;
  const started = performance.now();
  let outcome;
  try {
    const value = await Promise.race([
      Promise.resolve().then(thunk),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("__TIMEOUT__")), CASE_TIMEOUT_MS); }),
    ]);
    outcome = { status: "accepted", detail: summarize(value) };
  } catch (error) {
    if (error?.message === "__TIMEOUT__") outcome = { status: "timeout", detail: `>${CASE_TIMEOUT_MS}ms` };
    else if (typeof error === "string") outcome = { status: "rejected", detail: `[thrown as string, not Error] ${error}` };
    else if (!(error instanceof Error)) outcome = { status: "crash", detail: `non-Error thrown: ${String(error)}` };
    else if (error instanceof TypeError || error instanceof RangeError || error instanceof ReferenceError) {
      outcome = { status: "crash", detail: `${error.constructor.name}: ${error.message}` };
    } else outcome = { status: "rejected", detail: error.message };
  } finally { clearTimeout(timer); }
  const row = { group, name, expectation, ...outcome, duration_ms: Math.round(performance.now() - started) };
  row.verdict = expectation === "reject" ? (row.status === "rejected" ? "ok" : "FAIL") : (row.status === "accepted" ? "ok" : "FAIL");
  results.push(row);
  return row;
}

function summarize(value) {
  if (value == null) return String(value);
  if (typeof value !== "object") return String(value);
  const a = value.analysis || value;
  const bits = [];
  if (a.format) bits.push(`format=${a.format}`);
  if (a.tensor_count != null) bits.push(`tensors=${a.tensor_count}`);
  if (a.gguf) bits.push(`align=${a.gguf.alignment} badKeys=${a.gguf.invalid_metadata_key_count} badOff=${a.gguf.invalid_tensor_offset_count} status=${a.tensor_inventory?.status}`);
  if (a.safetensors) bits.push(`hdr=${a.safetensors.header_byte_length} payload=${a.safetensors.payload_byte_length}`);
  if (a.coreml) bits.push(`type=${a.coreml.model_type} spec=${a.coreml.specification_version} in=${a.inputs?.length} out=${a.outputs?.length}`);
  return bits.join(" ") || "<object>";
}

// ---------------------------------------------------------------- GGUF -----
class Writer {
  constructor(littleEndian = true) { this.parts = []; this.le = littleEndian; }
  raw(bytes) { this.parts.push(Uint8Array.from(bytes)); return this; }
  ascii(text) { return this.raw([...text].map((c) => c.charCodeAt(0))); }
  u32(value) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, value, this.le); this.parts.push(b); return this; }
  u64(value) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(value), this.le); this.parts.push(b); return this; }
  f32(value) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, value, this.le); this.parts.push(b); return this; }
  str(text) { const e = new TextEncoder().encode(text); this.u64(e.length); this.parts.push(e); return this; }
  strBytes(bytes) { this.u64(bytes.length); this.parts.push(Uint8Array.from(bytes)); return this; }
  finish() {
    const total = this.parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(total);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  }
}

function ggufBase({ version = 3, tensorCount = 1, kv = [], tensors = [{ name: "w", shape: [4], type: 0, offset: 0 }], le = true } = {}) {
  const w = new Writer(le);
  w.ascii("GGUF").u32(version).u64(tensorCount).u64(kv.length);
  for (const entry of kv) {
    w.str(entry.key).u32(entry.type);
    if (entry.type === 4) w.u32(entry.value);
    else if (entry.type === 8) w.str(entry.value);
    else if (entry.type === 7) w.raw([entry.value]);
    else if (entry.type === 10) w.u64(entry.value);
    else if (entry.type === 6) w.f32(entry.value);
  }
  for (const t of tensors) {
    w.str(t.name).u32(t.shape.length);
    for (const d of t.shape) w.u64(d);
    w.u32(t.type).u64(t.offset);
  }
  return w.finish();
}

const goodGguf = ggufBase({ kv: [{ key: "general.architecture", type: 8, value: "llama" }] });
await run("gguf", "control: valid v3 container", "accept", () => parseMetadataModel(goodGguf, "ok.gguf", 4096, "gguf"));
await run("gguf", "bad magic", "reject", () => parseMetadataModel(Uint8Array.of(0x47, 0x47, 0x55, 0x46 ^ 1, ...goodGguf.subarray(4)), "x.gguf", 4096, "gguf"));
await run("gguf", "unsupported version 1", "reject", () => parseMetadataModel(ggufBase({ version: 1 }), "x.gguf", 4096, "gguf"));
await run("gguf", "truncated to 12 bytes", "reject", () => parseMetadataModel(goodGguf.subarray(0, 12), "x.gguf", 4096, "gguf"));
await run("gguf", "tensor count 10_000_001", "reject", () => parseMetadataModel(ggufBase({ tensorCount: 10_000_001 }), "x.gguf", 4096, "gguf"));
await run("gguf", "tensor count 9_999_999 (under limit, header truncated)", "reject", () => parseMetadataModel(ggufBase({ tensorCount: 9_999_999 }), "x.gguf", 1 << 30, "gguf"));
await run("gguf", "rank 17 exceeds limit", "reject", () => parseMetadataModel(ggufBase({ tensors: [{ name: "w", shape: Array(17).fill(1), type: 0, offset: 0 }] }), "x.gguf", 1 << 20, "gguf"));
await run("gguf", "alignment = 0", "reject", () => parseMetadataModel(ggufBase({ kv: [{ key: "general.alignment", type: 4, value: 0 }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "alignment = 3 (non power of two)", "reject", () => parseMetadataModel(ggufBase({ kv: [{ key: "general.alignment", type: 4, value: 3 }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "alignment declared as STRING \"64\"", "reject", () => parseMetadataModel(ggufBase({ kv: [{ key: "general.alignment", type: 8, value: "64" }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "alignment declared as BOOL true", "reject", () => parseMetadataModel(ggufBase({ kv: [{ key: "general.alignment", type: 7, value: 1 }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "alignment = 2^53 via UINT64", "reject", () => parseMetadataModel(ggufBase({ kv: [{ key: "general.alignment", type: 10, value: 2 ** 53 }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "metadata key is invalid UTF-8", "reject", () => {
  const w = new Writer();
  w.ascii("GGUF").u32(3).u64(0).u64(1).strBytes([0xff, 0xfe]).u32(4).u32(1);
  return parseMetadataModel(w.finish(), "x.gguf", 4096, "gguf");
});
await run("gguf", "tensor data offset beyond declared file size", "reject", () => parseMetadataModel(goodGguf, "x.gguf", 8, "gguf"));
await run("gguf", "tensor offset far beyond file size (flagged not thrown)", "accept", () => parseMetadataModel(ggufBase({ tensors: [{ name: "w", shape: [4], type: 0, offset: 1 << 30 }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "prototype key __proto__ in metadata", "accept", () => parseMetadataModel(ggufBase({ kv: [{ key: "__proto__", type: 8, value: "polluted" }] }), "x.gguf", 4096, "gguf"));
await run("gguf", "10M-element packed BOOL array (CPU budget)", "accept", () => {
  const w = new Writer();
  w.ascii("GGUF").u32(3).u64(0).u64(1).str("a.b").u32(9).u32(7).u64(10_000_000);
  const payload = new Uint8Array(10_000_000);
  return parseMetadataModel(new Uint8Array([...w.finish(), ...payload]), "x.gguf", 1 << 30, "gguf");
});
await run("gguf", "File path: 0-byte file", "reject", () => readMetadataModelFile(file("empty.gguf", new Uint8Array()), "gguf"));
await run("gguf", "File path: valid container", "accept", () => readMetadataModelFile(file("ok.gguf", new Uint8Array([...goodGguf, ...new Uint8Array(4096)])), "gguf"));
await run("gguf", "File path: truncated mid-header", "reject", () => readMetadataModelFile(file("t.gguf", goodGguf.subarray(0, 20)), "gguf"));

// --------------------------------------------------------- SafeTensors -----
function safetensors(headerText, payloadBytes = 0, { headerLengthOverride = null } = {}) {
  const encoded = new TextEncoder().encode(headerText);
  const bytes = new Uint8Array(8 + encoded.length + payloadBytes);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerLengthOverride ?? encoded.length), true);
  bytes.set(encoded, 8);
  return bytes;
}

const goodSt = safetensors(JSON.stringify({ w: { dtype: "U8", shape: [4], data_offsets: [0, 4] } }), 4);
await run("safetensors", "control: valid container", "accept", () => parseMetadataModel(goodSt, "ok.safetensors", goodSt.length, "safetensors"));
await run("safetensors", "header length 0", "reject", () => parseMetadataModel(safetensors("", 0), "x.safetensors", 8, "safetensors"));
await run("safetensors", "header length 2^60", "reject", () => parseMetadataModel(safetensors("{}", 0, { headerLengthOverride: 2n ** 60n }), "x.safetensors", 10, "safetensors"));
await run("safetensors", "header does not start with '{'", "reject", () => parseMetadataModel(safetensors("[1,2]"), "x.safetensors", 13, "safetensors"));
await run("safetensors", "duplicate tensor key", "reject", () => {
  const b = safetensors('{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "negative data_offsets", "reject", () => {
  const b = safetensors('{"w":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}}', 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "overlapping tensor ranges", "reject", () => {
  const b = safetensors('{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}', 6);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "unknown dtype", "reject", () => {
  const b = safetensors('{"w":{"dtype":"FP8","shape":[1],"data_offsets":[0,1]}}', 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "shape product overflows 2^53", "reject", () => {
  const b = safetensors('{"w":{"dtype":"F32","shape":[9007199254740991,9007199254740991],"data_offsets":[0,1]}}', 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "__metadata__ with non-string value", "reject", () => {
  const b = safetensors('{"__metadata__":{"n":1},"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "payload shorter than declared coverage", "reject", () => {
  const b = safetensors('{"w":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}', 0);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "header NUL-padded (non-reference producer padding)", "reject", () => {
  const json = '{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}';
  const b = safetensors(json + "\0\0\0\0", 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "header space-padded (reference impl pads with spaces)", "accept", () => {
  const json = '{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}';
  const b = safetensors(json + "    ", 1);
  return parseMetadataModel(b, "x.safetensors", b.length, "safetensors");
});
await run("safetensors", "header is invalid UTF-8", "reject", () => {
  const bytes = new Uint8Array(8 + 3);
  new DataView(bytes.buffer).setBigUint64(0, 3n, true);
  bytes.set([0x7b, 0xff, 0xfe], 8);
  return parseMetadataModel(bytes, "x.safetensors", bytes.length, "safetensors");
});
await run("safetensors", "File path: header length lies (larger than file)", "reject", () => {
  const b = safetensors('{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1, { headerLengthOverride: 100000 });
  return readMetadataModelFile(file("x.safetensors", b), "safetensors");
});
await run("safetensors", "File path: 0-byte file", "reject", () => readMetadataModelFile(file("empty.safetensors", new Uint8Array()), "safetensors"));

// ------------------------------------------------------------- Core ML -----
class Proto {
  constructor() { this.parts = []; }
  varint(value) {
    let v = BigInt(value);
    const out = [];
    do { let byte = Number(v & 0x7fn); v >>= 7n; if (v > 0n) byte |= 0x80; out.push(byte); } while (v > 0n);
    this.parts.push(Uint8Array.from(out));
    return this;
  }
  key(field, wire) { return this.varint(field * 8 + wire); }
  int(field, value) { return this.key(field, 0).varint(value); }
  f32(field, value) {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    this.key(field, 5);
    this.parts.push(bytes);
    return this;
  }
  bytes(field, payload) { this.key(field, 2).varint(payload.length); this.parts.push(Uint8Array.from(payload)); return this; }
  str(field, text) { return this.bytes(field, new TextEncoder().encode(text)); }
  finish() {
    const total = this.parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(total);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  }
}

function featureType() { return new Proto().bytes(5, new Proto().int(2, 65568).finish()).finish(); }
function feature(name) { return new Proto().str(1, name).bytes(3, featureType()).finish(); }
function description({ inputs = ["image"], outputs = ["out"] } = {}) {
  const p = new Proto();
  for (const n of inputs) p.bytes(1, feature(n));
  for (const n of outputs) p.bytes(10, feature(n));
  return p.finish();
}
function coreml({ spec = 4, desc = description(), typeField = 500, typePayload = new Proto().finish(), extra = null } = {}) {
  const p = new Proto().int(1, spec).bytes(2, desc).bytes(typeField, typePayload);
  if (extra) extra(p);
  return p.finish();
}

function coreMlLayer({ name = "conv", input = "image", output = "out", typeField = 100, params = new Proto().finish() } = {}) {
  return new Proto().str(1, name).str(2, input).str(3, output).bytes(typeField, params).finish();
}

function coreMlNetwork(layers, preprocessing = []) {
  const network = new Proto();
  for (const layer of layers) network.bytes(1, layer);
  for (const item of preprocessing) network.bytes(2, item);
  return network.finish();
}

function coreMlImageScaler(featureName = "image", scaler = new Proto().f32(10, 1 / 255).finish()) {
  return new Proto().str(1, featureName).bytes(10, scaler).finish();
}

function packedFloat32(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytes;
}

const goodCoreMl = coreml();
await run("coreml", "control: valid model (sync bytes)", "accept", () => parseCoreMlModel(goodCoreMl, "ok.mlmodel"));
await run("coreml", "control: valid model (File path)", "accept", () => readCoreMlModelFile(file("ok.mlmodel", goodCoreMl)));
await run("coreml", "empty file", "reject", () => readCoreMlModelFile(file("empty.mlmodel", new Uint8Array())));
await run("coreml", "truncated mid-message", "reject", () => parseCoreMlModel(goodCoreMl.subarray(0, goodCoreMl.length - 3), "x.mlmodel"));
await run("coreml", "missing specificationVersion", "reject", () => parseCoreMlModel(new Proto().bytes(2, description()).bytes(500, new Proto().finish()).finish(), "x.mlmodel"));
await run("coreml", "missing ModelDescription", "reject", () => parseCoreMlModel(new Proto().int(1, 4).bytes(500, new Proto().finish()).finish(), "x.mlmodel"));
await run("coreml", "unknown model type field 999", "reject", () => parseCoreMlModel(coreml({ typeField: 999 }), "x.mlmodel"));
await run("coreml", "two model types declared", "reject", () => parseCoreMlModel(coreml({ extra: (p) => p.bytes(502, new Proto().finish()) }), "x.mlmodel"));
await run("coreml", "DUPLICATE ModelDescription field (sync bytes path)", "reject", () => parseCoreMlModel(coreml({ extra: (p) => p.bytes(2, description({ inputs: ["evil"], outputs: ["evil_out"] })) }), "x.mlmodel"));
await run("coreml", "DUPLICATE ModelDescription field (File path)", "reject", () => readCoreMlModelFile(file("x.mlmodel", coreml({ extra: (p) => p.bytes(2, description({ inputs: ["evil"], outputs: ["evil_out"] })) }))));
await run("coreml", "DUPLICATE specificationVersion (sync bytes path)", "reject", () => parseCoreMlModel(coreml({ extra: (p) => p.int(1, 99) }), "x.mlmodel"));
await run("coreml", "DUPLICATE specificationVersion (File path)", "reject", () => readCoreMlModelFile(file("x.mlmodel", coreml({ extra: (p) => p.int(1, 99) }))));
await run("coreml", "feature with no name", "reject", () => parseCoreMlModel(coreml({ desc: new Proto().bytes(1, new Proto().bytes(3, featureType()).finish()).bytes(10, feature("o")).finish() }), "x.mlmodel"));
await run("coreml", "FeatureType with two oneof values", "reject", () => {
  const ft = new Proto().bytes(5, new Proto().int(2, 65568).finish()).bytes(4, new Proto().int(1, 8).int(2, 8).finish()).finish();
  const desc = new Proto().bytes(1, new Proto().str(1, "i").bytes(3, ft).finish()).bytes(10, feature("o")).finish();
  return parseCoreMlModel(coreml({ desc }), "x.mlmodel");
});
await run("coreml", "defaultFunctionName that resolves to nothing", "reject", () => {
  const desc = new Proto().bytes(20, new Proto().str(1, "main").bytes(2, feature("i")).bytes(3, feature("o")).finish()).str(21, "missing").finish();
  return parseCoreMlModel(coreml({ desc }), "x.mlmodel");
});
await run("coreml", "mixes function-level and model-level interfaces", "reject", () => {
  const desc = new Proto().bytes(1, feature("i")).bytes(20, new Proto().str(1, "main").bytes(2, feature("i2")).finish()).finish();
  return parseCoreMlModel(coreml({ desc }), "x.mlmodel");
});
await run("coreml", "wire-type 2 on specificationVersion (File path)", "reject", () => readCoreMlModelFile(file("x.mlmodel", new Proto().str(1, "four").bytes(2, description()).bytes(500, new Proto().finish()).finish())));
await run("coreml", "length-delimited field claiming 2^40 bytes", "reject", () => parseCoreMlModel(new Proto().int(1, 4).key(2, 2).varint(2 ** 40).finish(), "x.mlmodel"));
await run("coreml", "1MB of unknown-field padding", "accept", () => {
  const p = new Proto().int(1, 4).bytes(2, description()).bytes(500, new Proto().finish());
  p.bytes(9999, new Uint8Array(1024 * 1024));
  return parseCoreMlModel(p.finish(), "x.mlmodel");
});
await run("coreml", "metadata map entry missing value", "reject", () => {
  const meta = new Proto().bytes(100, new Proto().str(1, "k").finish()).finish();
  const desc = new Proto().bytes(1, feature("i")).bytes(10, feature("o")).bytes(100, meta).finish();
  return parseCoreMlModel(coreml({ desc }), "x.mlmodel");
});
const floatWeight = new Proto().bytes(1, packedFloat32([0.25, -0.5, 1])).finish();
function coreMlConvParams(weight, cardinality) {
  return new Proto().int(1, cardinality).int(2, 1).int(10, 1).int(20, 1).int(20, 1).bytes(90, weight).finish();
}
const floatConv = coreMlConvParams(floatWeight, 3);
await run("coreml", "legacy NeuralNetwork layer and FP32 WeightParams", "accept", () => parseCoreMlModel(coreml({ typePayload: coreMlNetwork([
  coreMlLayer({ params: floatConv }),
]) }), "nn.mlmodel"));
await run("coreml", "legacy NeuralNetwork repeats an output blob", "reject", () => parseCoreMlModel(coreml({ typePayload: coreMlNetwork([
  coreMlLayer({ name: "a", output: "same", params: floatConv }),
  coreMlLayer({ name: "b", input: "same", output: "same", params: floatConv }),
]) }), "nn.mlmodel"));
await run("coreml", "WeightParams declares conflicting FP32 and FP16 storage", "reject", () => {
  const weight = new Proto().f32(1, 1).bytes(2, Uint8Array.of(0, 0)).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: new Proto().bytes(90, weight).finish() })]) }), "nn.mlmodel");
});
await run("coreml", "quantized WeightParams omits QuantizationParams", "reject", () => {
  const weight = new Proto().bytes(30, Uint8Array.of(1, 2, 3, 4)).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: new Proto().bytes(90, weight).finish() })]) }), "nn.mlmodel");
});
await run("coreml", "linear quantization contains non-finite scale", "reject", () => {
  const linear = new Proto().bytes(1, packedFloat32([Number.NaN])).finish();
  const quant = new Proto().int(1, 8).bytes(101, linear).finish();
  const weight = new Proto().bytes(30, Uint8Array.of(1)).bytes(40, quant).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: new Proto().bytes(90, weight).finish() })]) }), "nn.mlmodel");
});
await run("coreml", "packed 4-bit WeightParams binds exact parent cardinality and code utilization", "accept", () => {
  const linear = new Proto()
    .bytes(1, packedFloat32([0.5, 0.5, 0.5, 0.5]))
    .bytes(2, packedFloat32([0, 0, 0, 0]))
    .finish();
  const quant = new Proto().int(1, 4).bytes(101, linear).finish();
  const weight = new Proto().bytes(30, Uint8Array.of(0x12, 0x34)).bytes(40, quant).finish();
  const parsed = parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: coreMlConvParams(weight, 4) })]) }), "nn.mlmodel");
  const evidence = parsed.ops?.[0]?.coreml_weights?.[0];
  if (evidence?.value_count !== 4
    || evidence?.value_count_status !== "exact_from_parent_parameter_contract"
    || evidence?.packed_code_capacity !== 4
    || evidence?.packed_code_capacity_status !== "validated_against_parent_parameter_cardinality"
    || evidence?.quantization?.packed_bit_order !== "MSB-first within each byte"
    || evidence?.numerical_integrity?.quant_code_levels_used !== 4
    || evidence?.numerical_integrity?.finite_min !== 0.5
    || evidence?.numerical_integrity?.finite_max !== 2) {
    throw new Error(`packed WeightParams cardinality boundary is inconsistent: ${JSON.stringify(evidence)}`);
  }
  return parsed;
});
await run("coreml", "linear quantization scale/bias cardinality mismatches parent axis", "reject", () => {
  const linear = new Proto().bytes(1, packedFloat32([0.5])).bytes(2, packedFloat32([0])).finish();
  const quant = new Proto().int(1, 4).bytes(101, linear).finish();
  const weight = new Proto().bytes(30, Uint8Array.of(0x12, 0x34)).bytes(40, quant).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: coreMlConvParams(weight, 4) })]) }), "nn.mlmodel");
});
await run("coreml", "lookup-table quantization cardinality mismatches bit width", "reject", () => {
  const table = new Proto().bytes(1, packedFloat32([0, 1, 2])).finish();
  const quant = new Proto().int(1, 2).bytes(102, table).finish();
  const weight = new Proto().bytes(30, Uint8Array.of(1)).bytes(40, quant).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: new Proto().bytes(90, weight).finish() })]) }), "nn.mlmodel");
});
await run("coreml", "large packed FP32 WeightParams streams without an artificial value cap", "accept", () => {
  const weight = new Proto().bytes(1, packedFloat32(new Float32Array(100_001).fill(0.125))).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: coreMlConvParams(weight, 100_001) })]) }), "nn.mlmodel");
});
await run("coreml", "image-scaler preprocessing binds to a declared image input", "accept", () => {
  const imageType = new Proto().bytes(4, new Proto().int(1, 28).int(2, 28).int(3, 10).finish()).finish();
  const desc = new Proto().bytes(1, new Proto().str(1, "image").bytes(3, imageType).finish()).bytes(10, feature("out")).finish();
  return parseCoreMlModel(coreml({ desc, typePayload: coreMlNetwork([coreMlLayer({ params: floatConv })], [coreMlImageScaler()]) }), "nn.mlmodel");
});
await run("coreml", "preprocessing feature does not resolve to an input", "reject", () => parseCoreMlModel(coreml({
  typePayload: coreMlNetwork([coreMlLayer({ params: floatConv })], [coreMlImageScaler("missing")]),
}), "nn.mlmodel"));
await run("coreml", "image scaler contains a non-finite scalar", "reject", () => {
  const scaler = new Proto().f32(10, Number.POSITIVE_INFINITY).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: floatConv })], [coreMlImageScaler("image", scaler)]) }), "nn.mlmodel");
});
await run("coreml", "preprocessing contains multiple oneof values", "reject", () => {
  const invalid = new Proto().str(1, "image").bytes(10, new Proto().f32(10, 1).finish()).bytes(11, new Proto().f32(1, 0).finish()).finish();
  return parseCoreMlModel(coreml({ typePayload: coreMlNetwork([coreMlLayer({ params: floatConv })], [invalid]) }), "nn.mlmodel");
});

// ------------------------------------------------------ Artifact bundle -----
const manifestOf = (entries, rootId = "model-id") => new TextEncoder().encode(JSON.stringify({ fileFormatVersion: "1.0.0", rootModelIdentifier: rootId, itemInfoEntries: entries }));
const modelItem = { path: "com.apple.CoreML/model.mlmodel", name: "model.mlmodel", author: "com.apple.CoreML", description: "spec" };

function mlpackage(manifestBytes, files = {}) {
  const rows = [file("Manifest.json", manifestBytes, "Pkg.mlpackage/Manifest.json")];
  for (const [rel, bytes] of Object.entries(files)) rows.push(file(path.basename(rel), bytes, `Pkg.mlpackage/${rel}`));
  return rows;
}

await run("bundle", "control: valid mlpackage", "accept", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": modelItem }), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "empty selection", "reject", () => readArtifactBundle([]));
await run("bundle", "no manifest and no index", "reject", () => readArtifactBundle([file("a.bin", new Uint8Array(4), "Blob/a.bin")]));
await run("bundle", "manifest item path traverses out (../)", "reject", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": { ...modelItem, path: "../../etc/passwd" } }), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "manifest item path is absolute", "reject", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": { ...modelItem, path: "/etc/passwd" } }), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "manifest item path is a Windows drive path", "reject", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": { ...modelItem, path: "C:/Windows/system.ini" } }), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "rootModelIdentifier = __proto__", "reject", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": modelItem }, "__proto__"), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "rootModelIdentifier = toString (prototype lookup)", "reject", () => readArtifactBundle(mlpackage(manifestOf({ "model-id": modelItem }, "toString"), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl })));
await run("bundle", "unsupported fileFormatVersion", "reject", () => {
  const bad = new TextEncoder().encode(JSON.stringify({ fileFormatVersion: "2.0.0", rootModelIdentifier: "model-id", itemInfoEntries: { "model-id": modelItem } }));
  return readArtifactBundle(mlpackage(bad, { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl }));
});
await run("bundle", "case-collision between selected files", "reject", () => readArtifactBundle([
  ...mlpackage(manifestOf({ "model-id": modelItem }), { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl }),
  file("MODEL.mlmodel", goodCoreMl, "Pkg.mlpackage/Data/com.apple.CoreML/MODEL.MLMODEL"),
]));
await run("bundle", "root item resolves to two .mlmodel files", "reject", () => readArtifactBundle(mlpackage(
  manifestOf({ "model-id": { ...modelItem, path: "com.apple.CoreML" } }),
  { "Data/com.apple.CoreML/model.mlmodel": goodCoreMl, "Data/com.apple.CoreML/other.mlmodel": goodCoreMl },
)));

const shardIndex = (weightMap) => new TextEncoder().encode(JSON.stringify({ metadata: { total_size: 2 }, weight_map: weightMap }));
function shardSet(indexBytes, shards) {
  const rows = [file("model.safetensors.index.json", indexBytes, "Sharded/model.safetensors.index.json")];
  for (const [name, bytes] of Object.entries(shards)) rows.push(file(name, bytes, `Sharded/${name}`));
  return rows;
}
const shardA = safetensors('{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);
const shardB = safetensors('{"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);

await run("bundle", "control: valid sharded safetensors", "accept", () => readArtifactBundle(shardSet(shardIndex({ a: "s1.safetensors", b: "s2.safetensors" }), { "s1.safetensors": shardA, "s2.safetensors": shardB })));
await run("bundle", "index references a missing shard", "reject", () => readArtifactBundle(shardSet(shardIndex({ a: "s1.safetensors", b: "gone.safetensors" }), { "s1.safetensors": shardA })));
await run("bundle", "shard contains a tensor not in the index", "reject", () => readArtifactBundle(shardSet(shardIndex({ a: "s1.safetensors" }), { "s1.safetensors": shardB })));
await run("bundle", "weight_map path traverses out (../)", "reject", () => readArtifactBundle(shardSet(shardIndex({ a: "../../x.safetensors" }), { "s1.safetensors": shardA })));
await run("bundle", "weight_map entry is not a string", "reject", () => readArtifactBundle(shardSet(new TextEncoder().encode('{"weight_map":{"a":42}}'), { "s1.safetensors": shardA })));
await run("bundle", "tensor named toString (Object.prototype lookup)", "reject", () => {
  const shard = safetensors('{"toString":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', 1);
  return readArtifactBundle(shardSet(shardIndex({ a: "s1.safetensors" }), { "s1.safetensors": shard }));
});
await run("bundle", "index is not valid JSON", "reject", () => readArtifactBundle(shardSet(new TextEncoder().encode("{not json"), { "s1.safetensors": shardA })));
await run("bundle", "index weight_map is empty", "reject", () => readArtifactBundle(shardSet(new TextEncoder().encode('{"weight_map":{}}'), { "s1.safetensors": shardA })));

// ---------------------------------------------------- strict JSON parser ----
for (const [label, source, expectation] of [
  ["duplicate keys", '{"a":1,"a":2}', "reject"],
  ["trailing comma", '{"a":1,}', "reject"],
  ["trailing content", '{"a":1} junk', "reject"],
  ["NaN literal", '{"a":NaN}', "reject"],
  ["single quotes", "{'a':1}", "reject"],
  ["leading zero number", '{"a":01}', "reject"],
  ["deep nesting x200", `${"[".repeat(200)}1${"]".repeat(200)}`, "reject"],
  ["nested duplicate keys", '{"a":{"b":1,"b":2}}', "reject"],
  ["unterminated object", '{"a":1', "reject"],
  ["valid document", '{"a":[1,2,{"b":null}]}', "accept"],
  ["escaped quote in key", '{"a\\"b":1}', "accept"],
  ["duplicate key after escape", '{"a\\"b":1,"a\\"b":2}', "reject"],
]) {
  await run("strict-json", label, expectation, () => parseStrictJson(source, "doc"));
}

// ---------------------------------------------------------------- ONNX -----
const { analyzeOnnxModel } = await import(pathToFileURL(path.join(ROOT, "web/onnx.js")).href);
const onnxSample = await readFile(path.join(ROOT, "web/samples/sample_cnn_float.onnx"));
await run("onnx", "control: valid sample", "accept", () => analyzeOnnxModel(new Uint8Array(onnxSample), "ok.onnx"));
await run("onnx", "truncated to half", "reject", () => analyzeOnnxModel(new Uint8Array(onnxSample.subarray(0, onnxSample.length >> 1)), "x.onnx"));
await run("onnx", "empty buffer", "reject", () => analyzeOnnxModel(new Uint8Array(), "x.onnx"));
await run("onnx", "random bytes", "reject", () => analyzeOnnxModel(Uint8Array.from({ length: 4096 }, (_, i) => (i * 137 + 29) & 255), "x.onnx"));
await run("onnx", "single-byte flip at offset 64 (inside a doc_string)", "accept", () => {
  const bytes = new Uint8Array(onnxSample);
  bytes[64] ^= 0xff;
  return analyzeOnnxModel(bytes, "x.onnx");
});
await run("onnx", "all-zero buffer", "reject", () => analyzeOnnxModel(new Uint8Array(4096), "x.onnx"));

// --------------------------------------------------------------- TFLite -----
const { analyze_tflite_for_target, initSync } = await import(pathToFileURL(path.join(ROOT, "pkg/tflite_wasm_audit.js")).href);
initSync({ module: await readFile(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
const tflSample = new Uint8Array(await readFile(path.join(ROOT, "web/samples/mobilenet_v2_1.0_224_quant.tflite")));
const tfl = (bytes, target = "android_mid_a55", name = "x.tflite") => analyze_tflite_for_target(bytes, name, target);
await run("tflite", "control: valid MobileNetV2", "accept", () => tfl(tflSample, "android_mid_a55", "mobilenet_v2_1.0_224_quant.tflite").format);
await run("tflite", "empty buffer", "reject", () => tfl(new Uint8Array()));
await run("tflite", "random bytes", "reject", () => tfl(Uint8Array.from({ length: 8192 }, (_, i) => (i * 91 + 7) & 255)));
await run("tflite", "truncated to 1 KiB", "reject", () => tfl(tflSample.subarray(0, 1024)));
await run("tflite", "truncated to half", "reject", () => tfl(tflSample.subarray(0, tflSample.length >> 1)));
await run("tflite", "truncated to 99%", "reject", () => tfl(tflSample.subarray(0, Math.floor(tflSample.length * 0.99))));
await run("tflite", "valid magic, garbage body", "reject", () => {
  const bytes = new Uint8Array(4096);
  bytes.set(tflSample.subarray(0, 8));
  return tfl(bytes);
});
await run("tflite", "root offset points past end", "reject", () => {
  const bytes = tflSample.slice(0, 65536);
  new DataView(bytes.buffer).setUint32(0, 0x7fffffff, true);
  return tfl(bytes);
});
await run("tflite", "root offset = 0xFFFFFFFC", "reject", () => {
  const bytes = tflSample.slice(0, 65536);
  new DataView(bytes.buffer).setUint32(0, 0xfffffffc, true);
  return tfl(bytes);
});
await run("tflite", "unknown target id", "reject", () => tfl(tflSample, "not_a_real_target"));
await run("tflite", "empty target id", "reject", () => tfl(tflSample, ""));
await run("tflite", "filename with NUL and control chars", "accept", () => tfl(tflSample, "android_mid_a55", "a\u0000b\u001f<script>.tflite").format);
await run("tflite", "16 MiB of zeros", "reject", () => tfl(new Uint8Array(16 * 1024 * 1024)));

// ------------------------------------------------------- mutation fuzz -----
// Deterministic single/multi-byte corruption of the real fixtures. Any TypeError,
// RangeError, non-Error throw, or hang here is an unhandled parser fault.
function mulberry32(seed) {
  return () => { seed = (seed + 0x6d2b79f5) >>> 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

async function fuzz(label, base, invoke, iterations, headerBias) {
  const name = `${label} x${iterations}`;
  if (!selected("fuzz", name, "no-fault")) return;
  const random = mulberry32(0x51ed4a);
  const faults = [];
  let rejected = 0;
  let accepted = 0;
  for (let round = 0; round < iterations; round += 1) {
    const bytes = base.slice();
    const flips = 1 + Math.floor(random() * 4);
    const seeds = [];
    for (let i = 0; i < flips; i += 1) {
      const limit = random() < headerBias ? Math.min(bytes.length, 4096) : bytes.length;
      const at = Math.floor(random() * limit);
      bytes[at] = Math.floor(random() * 256);
      seeds.push(`${at}=${bytes[at]}`);
    }
    const started = performance.now();
    try {
      const value = await Promise.race([
        Promise.resolve().then(() => invoke(bytes)),
        new Promise((_, reject) => setTimeout(() => reject(new Error("__TIMEOUT__")), CASE_TIMEOUT_MS)),
      ]);
      accepted += 1;
      void value;
    } catch (error) {
      if (error?.message === "__TIMEOUT__") faults.push({ kind: "hang", seeds, detail: `>${CASE_TIMEOUT_MS}ms` });
      else if (typeof error === "string") rejected += 1;
      else if (!(error instanceof Error)) faults.push({ kind: "non-error-throw", seeds, detail: String(error).slice(0, 160) });
      else if (error instanceof TypeError || error instanceof RangeError) faults.push({ kind: error.constructor.name, seeds, detail: error.message.slice(0, 160) });
      else rejected += 1;
    }
    if (performance.now() - started > 4000) faults.push({ kind: "slow", seeds, detail: `${Math.round(performance.now() - started)}ms` });
  }
  results.push({
    group: "fuzz", name, expectation: "no-fault",
    status: faults.length ? "crash" : "rejected",
    detail: faults.length ? `${faults.length} faults; first: ${faults[0].kind} @ ${faults[0].seeds.join(",")} -> ${faults[0].detail}` : `${accepted} accepted / ${rejected} cleanly rejected`,
    duration_ms: 0,
    verdict: faults.length ? "FAIL" : "ok",
  });
  if (faults.length) {
    const kinds = new Map();
    for (const fault of faults) kinds.set(`${fault.kind}: ${fault.detail}`, (kinds.get(`${fault.kind}: ${fault.detail}`) || 0) + 1);
    console.log(`\n--- fuzz faults for ${label} (${faults.length}/${iterations}) ---`);
    for (const [key, count] of [...kinds].sort((a, b) => b[1] - a[1]).slice(0, 12)) console.log(`  x${count} ${key}`);
  }
}

await fuzz("onnx sample", new Uint8Array(onnxSample), (bytes) => analyzeOnnxModel(bytes, "fuzz.onnx"), 400, 0.5);
await fuzz("gguf container", goodGguf, (bytes) => parseMetadataModel(bytes, "fuzz.gguf", 1 << 20, "gguf"), 400, 1);
await fuzz("safetensors container", goodSt, (bytes) => parseMetadataModel(bytes, "fuzz.safetensors", bytes.length, "safetensors"), 400, 1);
await fuzz("coreml model", goodCoreMl, (bytes) => parseCoreMlModel(bytes, "fuzz.mlmodel"), 400, 1);
await fuzz("tflite header region", tflSample.slice(0, 262144), (bytes) => tfl(bytes, "android_mid_a55", "fuzz.tflite"), 250, 1);

// ---------------------------------------------------------------- report ----
const failures = results.filter((row) => row.verdict !== "ok");
const byGroup = new Map();
for (const row of results) {
  if (!byGroup.has(row.group)) byGroup.set(row.group, []);
  byGroup.get(row.group).push(row);
}
for (const [group, rows] of byGroup) {
  console.log(`\n=== ${group} (${rows.filter((r) => r.verdict === "ok").length}/${rows.length} as expected) ===`);
  for (const row of rows) {
    const mark = row.verdict === "ok" ? "  ok " : ">>FAIL";
    console.log(`${mark} [${row.status.padEnd(8)}] ${row.name}`);
    if (row.verdict !== "ok" || row.status === "crash" || row.status === "timeout") console.log(`         -> ${row.detail}`);
  }
}
console.log(`\nTOTAL ${results.length} cases, ${failures.length} unexpected`);
if (failures.length) process.exitCode = 1;
if (failures.length) {
  console.log("\nUNEXPECTED BEHAVIOUR:");
  for (const row of failures) console.log(`  [${row.group}] ${row.name}\n     status=${row.status} (${row.duration_ms}ms) detail=${row.detail}`);
}

if (isolationMode === "list") {
  console.log(`DEEPBOM_PARSER_CASES_JSON=${JSON.stringify(descriptors)}`);
} else if (isolationMode === "case") {
  const row = results[0] || null;
  console.log(`DEEPBOM_PARSER_RESULT_JSON=${JSON.stringify(row)}`);
  if (!row || results.length !== 1 || row.verdict !== "ok") process.exitCode = 1;
}
