// Logic review harness: properties that must hold for ANY input, checked
// against real analyses rather than against the producing code's own claims.
//
//   A. determinism      - same bytes twice must give the same result
//   B. target invariance - structural facts must not depend on the target profile
//   C. conservation      - every declared decomposition must sum to its total
//   D. value sanity      - no NaN/-0/Infinity, percentages in range, counts integral
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const load = (rel) => import(pathToFileURL(path.join(ROOT, rel)).href);
const wasm = await load("pkg/tflite_wasm_audit.js");
wasm.initSync({ module: await readFile(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
const { analyzeOnnxModel } = await load("web/onnx.js");

const problems = [];
const notes = [];
const check = (ok, label, detail) => { (ok ? notes : problems).push({ ok, label, detail }); return ok; };

const TARGETS = ["rpi4_a72", "android_mid_a55", "android_mid_a55_l1_16k", "android_mid_a55_l1_64k",
  "zynq_ultrascale_plus_a53", "android_flagship_x3_a715", "x86_avx2", "x86_sse4", "wasm_simd"];

const samples = [];
const missingSamples = [];
for (const [name, kind] of [
  ["mobilenet_v2_1.0_224_quant.tflite", "tflite"],
  ["mobilenet_v1_025_224_float.tflite", "tflite"],
  ["sample_cnn_float.onnx", "onnx"],
  ["mnist-8.onnx", "onnx"],
]) {
  try { samples.push({ name, kind, bytes: new Uint8Array(await readFile(path.join(ROOT, "web/samples", name))) }); }
  catch (error) { missingSamples.push(`${name}: ${error?.message || error}`); }
}
if (missingSamples.length) {
  throw new Error(`Required invariant fixtures are missing:\n${missingSamples.join("\n")}`);
}
notes.push({ ok: true, label: `samples loaded: ${samples.map((s) => s.name).join(", ")}` });

const analyze = (sample, target = "android_mid_a55") => sample.kind === "tflite"
  ? wasm.analyze_tflite_for_target(sample.bytes, sample.name, target)
  : analyzeOnnxModel(sample.bytes, sample.name);

// ------------------------------------------------------------ A. determinism --
for (const sample of samples) {
  const first = JSON.stringify(analyze(sample));
  const second = JSON.stringify(analyze(sample));
  check(first === second, `${sample.name}: analysis is deterministic`,
    first === second ? "" : `outputs differ (${first.length} vs ${second.length} chars)`);
}

// ------------------------------------------------------ B. target invariance --
// Structure comes from the artifact; only cost/placement models may vary.
const STRUCTURAL = ["operator_count", "tensor_count", "total_macs", "total_ops", "model_sha256",
  "file_size", "quantized_tensors", "per_channel_tensors", "version"];
for (const sample of samples.filter((s) => s.kind === "tflite")) {
  const base = analyze(sample, TARGETS[0]);
  for (const target of TARGETS.slice(1)) {
    const other = analyze(sample, target);
    for (const key of STRUCTURAL) {
      check(JSON.stringify(base[key]) === JSON.stringify(other[key]),
        `${sample.name}: ${key} is target-invariant`,
        `${TARGETS[0]}=${JSON.stringify(base[key])} ${target}=${JSON.stringify(other[key])}`);
    }
    const baseOps = base.ops.map((op) => [op.name, op.macs, op.ops].join("|")).join(",");
    const otherOps = other.ops.map((op) => [op.name, op.macs, op.ops].join("|")).join(",");
    check(baseOps === otherOps, `${sample.name}: per-op MAC/op counts are target-invariant`,
      `differs under ${target}`);
    const baseQuant = JSON.stringify(base.quantization_status?.classification);
    check(baseQuant === JSON.stringify(other.quantization_status?.classification),
      `${sample.name}: quantization classification is target-invariant`, `differs under ${target}`);
  }
}

// -------------------------------------------------------- C/D. per-analysis --
function scanNumbers(value, pathLabel, sink, depth = 0) {
  if (depth > 12) return;
  if (typeof value === "number") {
    if (Number.isNaN(value)) sink.push(`${pathLabel} is NaN`);
    else if (!Number.isFinite(value)) sink.push(`${pathLabel} is ${value}`);
    else if (Object.is(value, -0)) sink.push(`${pathLabel} is negative zero`);
    return;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < Math.min(value.length, 400); i += 1) scanNumbers(value[i], `${pathLabel}[${i}]`, sink, depth + 1);
    return;
  }
  if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) scanNumbers(item, `${pathLabel}.${key}`, sink, depth + 1);
  }
}

function percentFields(value, pathLabel, sink, depth = 0) {
  if (depth > 12 || !value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (let i = 0; i < Math.min(value.length, 400); i += 1) percentFields(value[i], `${pathLabel}[${i}]`, sink, depth + 1);
    return;
  }
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "number" && /_percent$|_pct$|_ratio$/.test(key) && Number.isFinite(item)) {
      const limit = /_percent$|_pct$/.test(key) ? 100 : Infinity;
      if (item < 0 || item > limit) sink.push(`${pathLabel}.${key} = ${item} (outside 0..${limit})`);
    }
    percentFields(item, `${pathLabel}.${key}`, sink, depth + 1);
  }
}

for (const sample of samples) {
  const a = analyze(sample);
  const label = sample.name;

  const numeric = [];
  scanNumbers(a, "analysis", numeric);
  check(numeric.length === 0, `${label}: no NaN / Infinity / -0 in the analysis`,
    `${numeric.length} found, e.g. ${numeric.slice(0, 4).join("; ")}`);

  const ranges = [];
  percentFields(a, "analysis", ranges);
  check(ranges.length === 0, `${label}: every *_percent field is within 0..100`,
    `${ranges.length} found, e.g. ${ranges.slice(0, 4).join("; ")}`);

  if (Array.isArray(a.ops) && a.total_macs != null) {
    const sum = a.ops.reduce((s, op) => s + Number(op.macs || 0), 0);
    check(sum === a.total_macs, `${label}: sum(ops[].macs) == total_macs`, `${sum} vs ${a.total_macs}`);
  }
  if (a.delegated_macs != null) {
    check(Math.abs(a.delegated_macs + a.fallback_macs - a.total_macs) < 1e-6,
      `${label}: delegated + fallback == total MACs`, `${a.delegated_macs}+${a.fallback_macs} vs ${a.total_macs}`);
    check(a.delegated_macs >= 0 && a.fallback_macs >= 0, `${label}: delegated/fallback MACs are non-negative`,
      `${a.delegated_macs} / ${a.fallback_macs}`);
  }
  if (a.size_breakdown) {
    const sb = a.size_breakdown;
    check(sb.constant_bytes + sb.metadata_bytes + sb.structure_overhead_bytes === sb.file_size,
      `${label}: size_breakdown conserves file size`,
      `${sb.constant_bytes}+${sb.metadata_bytes}+${sb.structure_overhead_bytes} vs ${sb.file_size}`);
    check(sb.unique_constant_bytes + sb.duplicate_constant_bytes === sb.constant_bytes,
      `${label}: unique + duplicate == constant bytes`,
      `${sb.unique_constant_bytes}+${sb.duplicate_constant_bytes} vs ${sb.constant_bytes}`);
    check(sb.structure_overhead_bytes >= 0, `${label}: structure overhead is non-negative`, String(sb.structure_overhead_bytes));
  }
  if (Array.isArray(a.tensors) && a.tensor_count != null) {
    check(a.tensors.length === a.tensor_count, `${label}: tensors[].length == tensor_count`, `${a.tensors.length} vs ${a.tensor_count}`);
    const past = a.tensors.filter((t) => t.constant_buffer
      && Number(t.buffer_data_offset) + Number(t.buffer_data_length) > Number(a.file_size));
    check(past.length === 0, `${label}: no constant buffer extends past EOF`, `${past.length} tensors`);
  }
  if (Array.isArray(a.tensor_types) && a.tensor_count != null) {
    const total = a.tensor_types.reduce((s, e) => s + Number(e.count || 0), 0);
    check(total === a.tensor_count, `${label}: tensor_types histogram totals tensor_count`, `${total} vs ${a.tensor_count}`);
  }
  if (a.quantized_tensors != null && a.tensor_count != null) {
    check(a.quantized_tensors <= a.tensor_count, `${label}: quantized_tensors <= tensor_count`,
      `${a.quantized_tensors} vs ${a.tensor_count}`);
    check(Number(a.per_channel_tensors || 0) <= Number(a.quantized_tensors || 0),
      `${label}: per_channel_tensors <= quantized_tensors`, `${a.per_channel_tensors} vs ${a.quantized_tensors}`);
  }
  // Every op must reference tensors that exist.
  if (Array.isArray(a.ops) && Array.isArray(a.tensors)) {
    const bad = [];
    for (const op of a.ops) {
      for (const id of [...(op.inputs || []), ...(op.outputs || [])]) {
        if (id >= 0 && id >= a.tensors.length) bad.push(`${op.name || op.index}->${id}`);
      }
    }
    check(bad.length === 0, `${label}: every op tensor index is in range`, `${bad.length}, e.g. ${bad.slice(0, 4).join(",")}`);
  }
  // Declared outputs must be produced, declared inputs must exist.
  if (Array.isArray(a.input_tensor_indices) && Array.isArray(a.tensors)) {
    const bad = [...a.input_tensor_indices, ...(a.output_tensor_indices || [])].filter((i) => i < 0 || i >= a.tensors.length);
    check(bad.length === 0, `${label}: graph IO tensor indices are in range`, JSON.stringify(bad));
  }
}

// ---------------------------------------------------- ONNX-specific checks ---
for (const sample of samples.filter((s) => s.kind === "onnx")) {
  const a = analyze(sample);
  if (Array.isArray(a.ops) && a.operator_count != null) {
    check(a.ops.length === a.operator_count, `${sample.name}: ops[].length == operator_count`, `${a.ops.length} vs ${a.operator_count}`);
  }
  if (a.mac_assessment) {
    const m = a.mac_assessment;
    if (m.assessed_compute_ops != null && m.compute_ops != null) {
      check(m.assessed_compute_ops <= m.compute_ops, `${sample.name}: assessed compute ops <= compute ops`,
        `${m.assessed_compute_ops} vs ${m.compute_ops}`);
    }
  }
}

// ------------------------------------------------------------------ report ---
console.log(`checks run: ${problems.length + notes.filter((n) => n.label.includes(":")).length}`);
console.log(`\n================ ${problems.length} PROBLEM(S) ================`);
for (const row of problems) console.log(`BAD  ${row.label}${row.detail ? ` — ${row.detail}` : ""}`);
if (!problems.length) console.log("(none)");
else process.exitCode = 1;
