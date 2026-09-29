import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildValueType, valueLogicalBytes } from "../web/lib/ir-value-type.js";
import { scalarDtypeBits, scalarDtypeBytes, logicalBytesForElements, logicalBytesForShape, shapeElementCount,
  safeShapeElementCount, staticTensorPayloadBytes, tensorPayloadAssessment, broadcastStaticShapes,
  safeIntegerCodeRange, safePositiveShapeElementCount } from "../web/lib/tensor-size.js";
import { exactCountValue, exactIntegerWithValue } from "../web/lib/exact-integer.js";
import { float16ToNumber, bfloat16ToNumber, bigintToFloat32, normalizeAxis, readPackedUnsigned } from "../web/lib/scalar-numeric.js";
import { onnxTensorPayloadBytes, deterministicTensorPayloadAssessment } from "../web/lib/report-conformance-runtime-helpers.js";
import { collectFullGraph } from "../web/lib/graph-layout.js";
import { perTensor8BitContract } from "../web/lib/quantization-math.js";
import { arraysEqual } from "../web/lib/array-contract.js";
import { buildBackendPlacementProjection } from "../web/lib/backend-placement-projection.js";

execFileSync(process.execPath, ["scripts/generate-scalar-types.mjs", "--check"], { stdio: "inherit" });
const registry = JSON.parse(readFileSync("config/scalar-types.v1.json", "utf8"));
for (const name of ["F8_E8M0", "FLOAT8E8M0"]) {
  assert.equal(registry.types.find(row => row.name === name).signed, false, "E8M0 has no sign bit.");
  assert.equal(safeIntegerCodeRange(name), null, "Unsigned float is not an unsigned integer code domain.");
}
let checks = 0;
// Independent quotient/remainder oracle avoids the production ceil expression.
for (const type of registry.types) for (const n of [0n, 1n, 3n, 7n, 8n, 31n, 256n, 2n ** 53n - 1n, 2n ** 63n + 1n]) {
  const bits = n * BigInt(type.bits);
  const expected = bits / 8n + (bits % 8n === 0n ? 0n : 1n);
  const actual = logicalBytesForElements(type.name, n);
  assert.equal(actual.decimal, expected.toString());
  assert.equal(actual.number, expected <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(expected) : null);
  assert.equal(scalarDtypeBytes(type.name), type.bits % 8 ? null : type.bits / 8);
  checks++;
}
for (const dtype of ["Q4_0", "Q4_K", "STRING", "RESOURCE", "constructor", "__proto__", ""]) {
  assert.equal(scalarDtypeBits(dtype), null); assert.equal(logicalBytesForShape(dtype, [1]), null);
}
assert.deepEqual(safeIntegerCodeRange("INT2"), [-2, 1]);
assert.deepEqual(safeIntegerCodeRange("UINT32"), [0, 4294967295]);
assert.equal(safeIntegerCodeRange("INT64"), null);
for (const shape of [null, undefined, [null], ["2"], [NaN], [-1], [Infinity], [0, -1], [2 ** 53], new Array(2), [0, , 3]]) assert.equal(shapeElementCount(shape), null);
assert.equal(arraysEqual(new Array(2), [2, 3]), false);
assert.equal(arraysEqual([2, 3], new Array(2)), false);
assert.equal(arraysEqual([2, 3], [2, 3]), true);
assert.equal(shapeElementCount([]).decimal, "1");
assert.equal(safePositiveShapeElementCount([]), null);
assert.equal(safePositiveShapeElementCount([], { allowScalar: true }), 1);
assert.equal(safePositiveShapeElementCount([0]), null);
assert.equal(safePositiveShapeElementCount(["2"]), null);
assert.equal(safeShapeElementCount([Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, 0]), 0);
assert.equal(safeShapeElementCount([Number.MAX_SAFE_INTEGER, 2]), null);
assert.equal(logicalBytesForElements("INT2", Number.MAX_SAFE_INTEGER).number, 2251799813685248);
for (const a of [[], [0], [1], [2], [0, 1], [1, 2], [2, 1]]) for (const b of [[], [0], [1], [2], [1, 0], [2, 1]]) {
  const actual = broadcastStaticShapes(a, b);
  if (actual) {
    for (const input of [a, b]) input.forEach((d, i) => assert.ok(d === 1 || d === actual[actual.length - input.length + i]));
  }
}
assert.deepEqual(broadcastStaticShapes([0, 3], [1, 3]), [0, 3]);
assert.equal(broadcastStaticShapes([0, 3], [2, 3]), null);
assert.equal(normalizeAxis(null, 3), null); assert.equal(normalizeAxis(-1, 3), 2);
assert.equal(normalizeAxis(0, 0), null); assert.equal(normalizeAxis(1, 1.5), null);
assert.equal(exactCountValue({ decimal: "3", number: 2 }), null);
assert.equal(exactCountValue({ decimal: "3", value: 2 }), null);
assert.equal(exactCountValue({ decimal: "3", number: 3 }), 3n);
assert.deepEqual(exactIntegerWithValue(2n ** 63n), { value: null, decimal: "9223372036854775808" });

for (const dtype of ["FLOAT32", "BFLOAT16", "FLOAT4E2M1", "INT2", "UINT1", "UINT3", "UINT6"]) {
  for (const shape of [[], [0, 3], [3], [2, 3], [Number.MAX_SAFE_INTEGER]]) {
    const tensor = { index: 0, dtype, shape, shape_declared: true };
    const expected = logicalBytesForShape(dtype, shape);
    assert.deepEqual(valueLogicalBytes(buildValueType(tensor, "tflite", "main")), expected);
    assert.equal(staticTensorPayloadBytes(tensor), expected.number);
    assert.equal(onnxTensorPayloadBytes(dtype, shapeElementCount(shape).decimal), expected.number);
    assert.equal(deterministicTensorPayloadAssessment(tensor).payload_bytes, expected.number);
    const analysis = { format: "tflite", tensors: [tensor], ops: [{ index: 0, name: "A", inputs: [], outputs: [0] }, { index: 1, name: "B", inputs: [0], outputs: [] }] };
    const graph = collectFullGraph(analysis, { consumers: new Map([[0, [1]]]) });
    assert.equal(graph.edges[0].bytes, expected.number);
    const placement = buildBackendPlacementProjection({ analysis, profileId: "test", label: "test", evidenceClass: "DERIVED", rows: analysis.ops.map(op => ({ op_index: op.index, state: "UNRESOLVED", reason_codes: [], unresolved_predicates: [] })) });
    assert.equal(placement.graph_edges[0].logical_payload_bytes, expected.number);
    checks++;
  }
}
const dynamic = { dtype: "FLOAT32", shape: [1, 3], shape_signature: [-1, 3] };
assert.equal(staticTensorPayloadBytes(dynamic), null);
assert.deepEqual(tensorPayloadAssessment(dynamic, { allowSerializedBatchOne: true }), { bytes: 12, status: "assessed_serialized_batch1", binding: "serialized_batch1_projection", reason: null });
for (const shape of [undefined, []]) assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape, shape_declared: false }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [3], shape_signature: [4] }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [] }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [], has_rank: false }), null);
assert.equal(staticTensorPayloadBytes({ dtype: "FLOAT32", shape: [], has_rank: true }), 4);
const quantized = { index: 0, dtype: "INT8", scale_sample: [0.25], zero_point_sample: [0] };
assert.equal(perTensor8BitContract([quantized], 0).qmin, -128);
for (const value of [null, false, "0", NaN, Infinity, -Infinity, 0.5, 128]) assert.throws(() => perTensor8BitContract([{ ...quantized, zero_point_sample: [value] }], 0));
assert.throws(() => perTensor8BitContract([{ ...quantized, quant_scales: 2 }], 0));

const oracle = JSON.parse(execFileSync(process.env.PYTHON || "python3", ["scripts/oracles/scalar-numeric-oracle.py"], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
const decoded = token => token === "NaN" ? NaN : token === "Infinity" ? Infinity : token === "-Infinity" ? -Infinity : token === "-0" ? -0 : token;
for (let bits = 1; bits <= 8; bits++) {
  const values = Array.from({ length: 65 }, (_, i) => (i * 53 + 7) % 2 ** bits);
  const packed = values.reduce((sum, value, index) => sum | BigInt(value) << BigInt(index * bits), 0n);
  const bytes = Array.from({ length: Math.ceil(values.length * bits / 8) }, (_, i) => Number(packed >> BigInt(i * 8) & 255n));
  values.forEach((value, i) => assert.equal(readPackedUnsigned(bytes, i, bits), value));
  assert.equal(readPackedUnsigned(bytes, bytes.length * 8, bits), null);
}
for (const [key, convert] of [["binary16", float16ToNumber], ["bfloat16", bfloat16ToNumber]]) {
  oracle[key].forEach((expected, bits) => assert.ok(Object.is(convert(bits), decoded(expected)), `${key} code ${bits}`));
}
for (const [integer, expected] of oracle.integer_to_binary32) assert.ok(Object.is(bigintToFloat32(BigInt(integer)), decoded(expected)), `integer -> binary32 ${integer}`);

// Exercise the guard with deliberately broken isolated source trees. Do not
// inject files into the application being tested by concurrent browser checks.
const isolated = mkdtempSync(path.join(tmpdir(), "deepbom-common-rule-guard-"));
try {
  for (const name of ["config", "web", "src"]) mkdirSync(path.join(isolated, name));
  writeFileSync(path.join(isolated, "config/scalar-types.v1.json"), JSON.stringify(registry));
  writeFileSync(path.join(isolated, "config/common-rule-exceptions.v1.json"), JSON.stringify({ exceptions: [] }));
  const audit = () => spawnSync(process.execPath, [fileURLToPath(new URL("./audit-common-rules.mjs", import.meta.url))], { cwd: isolated, encoding: "utf8" });
  assert.equal(audit().status, 0);
  writeFileSync(path.join(isolated, "web/copy.js"), 'export const WIDTHS = { FLOAT32: 4, FLOAT16: 2, INT8: 1 };');
  let rejected = audit(); assert.notEqual(rejected.status, 0); assert.match(rejected.stderr, /duplicated scalar width/);
  rmSync(path.join(isolated, "web/copy.js"));
  writeFileSync(path.join(isolated, "src/copy.rs"), 'fn bytes(dtype: &str) -> Option<usize> { match dtype { "FLOAT32" => Some(4), "FLOAT16" => Some(2), "INT8" => Some(1), _ => None } }');
  rejected = audit(); assert.notEqual(rejected.status, 0); assert.match(rejected.stderr, /unreviewed Rust scalar lookup/);
  rmSync(path.join(isolated, "src/copy.rs"));
  const copy = 'export function shapeCount(shape) { if (!Array.isArray(shape) || !shape.every(Number.isSafeInteger)) return null; return shape.reduce((total, dimension) => total * dimension, 1); }';
  for (const name of ["a", "b"]) writeFileSync(path.join(isolated, `web/${name}.js`), copy);
  rejected = audit(); assert.notEqual(rejected.status, 0); assert.match(rejected.stderr, /Unreviewed duplicate calculation/);
} finally { rmSync(isolated, { recursive: true, force: true }); }
console.log(`Common calculations: ${checks} exact size/consumer comparisons; 131072 IEEE encodings; ${oracle.integer_to_binary32.length} independently rounded integers; unknown/zero/rank/projection/broadcast guards passed.`);
