// Format-gate, extension/magic conflict, malformed-input, and ONNX external-data routing tests.
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const load = (rel) => import(pathToFileURL(path.join(ROOT, rel)).href);
const { detectModelFormat, modelFormatGate, MODEL_FORMAT_ADAPTERS } = await load("web/lib/model-file.js");
const { analyzeOnnxModel } = await load("web/onnx.js");
const { analyzeExecuTorchModel } = await load("web/executorch.js");
const { parseMetadataModel } = await load("web/lib/metadata-model-adapters.js");
const { parseCoreMlModel } = await load("web/lib/coreml-metadata-adapter.js");

const results = [];
const record = (group, name, ok, detail) => { results.push({ group, name, ok, detail }); return ok; };

// ------------------------------------------------- advertised extension map --
const ADVERTISED = [];
for (const adapter of Object.values(MODEL_FORMAT_ADAPTERS)) {
  for (const extension of adapter.extensions) ADVERTISED.push({ extension, expect: adapter.id });
}
record("inventory", "every advertised extension is enumerable", ADVERTISED.length === 11,
  `${ADVERTISED.length} extensions: ${ADVERTISED.map((e) => e.extension).join(" ")}`);

for (const { extension, expect } of ADVERTISED) {
  const got = detectModelFormat(`model${extension}`, new Uint8Array());
  record("routing", `${extension} routes to ${expect}`, got === expect, `got ${got}`);
  const upper = detectModelFormat(`MODEL${extension.toUpperCase()}`, new Uint8Array());
  record("routing", `${extension.toUpperCase()} (uppercase) routes to ${expect}`, upper === expect, `got ${upper}`);
}

// ------------------------------------------------------------- gate policy --
for (const extension of [".pt", ".pth", ".ckpt"]) {
  const gate = modelFormatGate(detectModelFormat(`weights${extension}`, new Uint8Array()));
  record("gate", `${extension} is blocked as unsafe serialized code`, gate.blocked && gate.reason === "unsafe_serialized_code", JSON.stringify(gate));
}
for (const extension of [".pte", ".ptd"]) {
  const gate = modelFormatGate(detectModelFormat(`model${extension}`, new Uint8Array()));
  record("gate", `${extension} is admitted to the bounded ExecuTorch analyzer`, !gate.blocked && gate.adapter.id === "executorch", JSON.stringify(gate));
}
for (const name of ["model.bin", "model.h5", "model.pb", "model.npz", "archive.zip", "noextension"]) {
  const gate = modelFormatGate(detectModelFormat(name, new Uint8Array()));
  record("gate", `${name} (unknown, no magic) is blocked`, gate.blocked && gate.reason === "unsupported_format", JSON.stringify(gate));
}

// ----------------------------------------- content vs extension conflicts ---
// torch.save produces either a raw pickle (0x80 proto) or a ZIP container.
const PICKLE = Uint8Array.from([0x80, 0x02, 0x63, 0x5f, 0x5f, 0x6d, 0x61, 0x69, 0x6e, 0x5f, 0x5f, 0x0a]);
const TORCH_ZIP = Uint8Array.from([0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00]);

for (const [label, bytes] of [["raw pickle", PICKLE], ["torch zip", TORCH_ZIP]]) {
  for (const name of ["evil.pt", "evil.pth", "evil.ckpt"]) {
    const gate = modelFormatGate(detectModelFormat(name, bytes));
    record("conflict", `${label} named ${name} stays blocked`, gate.blocked && gate.reason === "unsafe_serialized_code", JSON.stringify(gate));
  }
  // Renaming a pickle to a supported extension must never yield an analysis.
  for (const [name, run] of [
    ["evil.gguf", () => parseMetadataModel(bytes, name, bytes.length, "gguf")],
    ["evil.safetensors", () => parseMetadataModel(bytes, name, bytes.length, "safetensors")],
    ["evil.mlmodel", () => parseCoreMlModel(bytes, name)],
    ["evil.onnx", () => analyzeOnnxModel(bytes, name)],
    ["evil.pte", () => analyzeExecuTorchModel(bytes, name)],
  ]) {
    let outcome;
    try { run(); outcome = "ACCEPTED"; }
    catch (error) {
      outcome = error instanceof TypeError || error instanceof RangeError ? `crash:${error.constructor.name}` : "rejected";
    }
    record("conflict", `${label} renamed to ${name} is rejected by that parser`, outcome === "rejected", outcome);
  }
  // A pickle with no recognisable extension must not be sniffed into a format.
  const sniffed = detectModelFormat("pytorch_model.bin", bytes);
  record("conflict", `${label} as pytorch_model.bin is not sniffed into a parser`, modelFormatGate(sniffed).blocked, `detected ${sniffed}`);
}

// Double extensions: the real (last) extension must win.
for (const [name, expect] of [
  ["model.pt.tflite", "tflite"],
  ["model.tflite.pt", "pytorch_pickle"],
  ["model.onnx.pth", "pytorch_pickle"],
  ["model.pth.onnx", "onnx"],
  ["model.safetensors.index.json", "unsupported"],
]) {
  const got = detectModelFormat(name, new Uint8Array());
  record("conflict", `${name} resolves to ${expect}`, got === expect, `got ${got}`);
}

// ------------------------------------------------------- sniffing fallback --
const sniffCases = [
  ["TFL3 magic, unknown extension", Uint8Array.from([0, 0, 0, 0, 0x54, 0x46, 0x4c, 0x33]), "tflite"],
  ["GGUF magic, unknown extension", Uint8Array.from([0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0]), "gguf"],
  ["ET12 magic, unknown extension", Uint8Array.from([0, 0, 0, 0, 0x45, 0x54, 0x31, 0x32]), "executorch"],
  ["FT01 magic, unknown extension", Uint8Array.from([0, 0, 0, 0, 0x46, 0x54, 0x30, 0x31]), "executorch"],
  ["all zeros", new Uint8Array(64), "unsupported"],
  ["empty", new Uint8Array(), "unsupported"],
];
for (const [label, bytes, expect] of sniffCases) {
  const got = detectModelFormat("payload.dat", bytes);
  record("sniff", label, got === expect, `got ${got}`);
}
// A single 0x08 lead byte is the entire ONNX signature; anything starting that
// way is routed to the protobuf parser, which must then reject it cleanly.
{
  const got = detectModelFormat("payload.dat", Uint8Array.from([0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]));
  record("sniff", "lone 0x08 lead byte is claimed as ONNX", got === "onnx", `got ${got}`);
  let outcome;
  try { analyzeOnnxModel(Uint8Array.from([0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]), "payload.dat"); outcome = "ACCEPTED"; }
  catch (error) { outcome = error instanceof TypeError || error instanceof RangeError ? `crash:${error.constructor.name}` : "rejected"; }
  record("sniff", "…and the ONNX parser then rejects it cleanly", outcome === "rejected", outcome);
}

// ------------------------------------------------ ONNX external data guards --
const onnxSample = new Uint8Array(await readFile(path.join(ROOT, "web/samples/sample_cnn_float.onnx")));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sidecar = (overrides = {}) => {
  const bytes = Uint8Array.from({ length: 64 }, (_, i) => i);
  return { path: "weights.bin", bytes, sha256: sha256(bytes), ...overrides };
};

const externalCases = [
  ["control: one well-formed sidecar", [sidecar()], "accept"],
  ["path traverses out of the model directory", [sidecar({ path: "../../etc/passwd" })], "reject"],
  ["absolute POSIX path", [sidecar({ path: "/etc/passwd" })], "reject"],
  ["Windows drive path", [sidecar({ path: "C:/Windows/system.ini" })], "reject"],
  ["backslash traversal", [sidecar({ path: "..\\..\\secret.bin" })], "reject"],
  ["file:// URI", [sidecar({ path: "file:///etc/passwd" })], "reject"],
  ["http:// URI", [sidecar({ path: "http://evil.example/x.bin" })], "reject"],
  ["NUL byte in path", [sidecar({ path: "weights\0.bin" })], "reject"],
  ["empty path", [sidecar({ path: "" })], "reject"],
  ["duplicate supplied paths", [sidecar(), sidecar()], "reject"],
  ["missing SHA-256", [sidecar({ sha256: "" })], "reject"],
  ["malformed SHA-256", [sidecar({ sha256: "not-a-hash" })], "reject"],
  ["malformed SHA-1", [sidecar({ sha1: "zz" })], "reject"],
  ["no byte payload", [sidecar({ bytes: null })], "reject"],
  ["externalDataFiles is not an array", "not-an-array", "reject"],
  ["file count over the 1,024 limit", Array.from({ length: 1025 }, (_, i) => sidecar({ path: `w${i}.bin` })), "reject"],
];

for (const [label, files, expectation] of externalCases) {
  let outcome;
  try {
    analyzeOnnxModel(onnxSample, "sample.onnx", null, { externalDataFiles: files });
    outcome = "accepted";
  } catch (error) {
    outcome = error instanceof TypeError || error instanceof RangeError ? `crash:${error.constructor.name}` : "rejected";
  }
  record("onnx-external", label, outcome === (expectation === "accept" ? "accepted" : "rejected"), outcome);
}

// ------------------------------------------------------------------ report --
const byGroup = new Map();
for (const row of results) {
  if (!byGroup.has(row.group)) byGroup.set(row.group, []);
  byGroup.get(row.group).push(row);
}
for (const [group, rows] of byGroup) {
  console.log(`\n=== ${group} (${rows.filter((r) => r.ok).length}/${rows.length}) ===`);
  for (const row of rows) console.log(`${row.ok ? "  ok " : ">>FAIL"} ${row.name}${row.ok ? "" : ` — ${row.detail}`}`);
}
const failures = results.filter((r) => !r.ok);
console.log(`\nTOTAL ${results.length} checks, ${failures.length} failing`);
if (failures.length) process.exitCode = 1;
if (failures.length) {
  console.log("\nFAILURES:");
  for (const row of failures) console.log(`  [${row.group}] ${row.name} — ${row.detail}`);
}
