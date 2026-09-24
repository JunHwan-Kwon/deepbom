import assert from "node:assert/strict";
import Ajv2020 from "ajv/dist/2020.js";
import { readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { analyzeOnnxModel } from "../web/onnx.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { validateArtifactEvidenceIr } from "../web/lib/artifact-ir.js";
import { validateModelIr } from "../web/lib/model-ir.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import { exactInteger } from "../web/lib/exact-integer.js";
import { model, node, tensor, externalTensor, valueInfo, float32 } from "./onnx-proto-fixture.mjs";

const rows = [];
function check(name, run) {
  try { run(); rows.push({ name, status: "pass" }); }
  catch (error) { rows.push({ name, status: "fail", error: error.message }); }
}
function context(bytes, analysis, filename) {
  return getArtifactIrContext(analysis, { filename, format: analysis.format, sha256: sha256BytesHex(bytes), size: bytes.length });
}
function synthetic(tensorRow) {
  return getArtifactIrContext({ format: "safetensors", filename: "adapter-boundary.safetensors", model_sha256: "c".repeat(64), file_size_bytes: 100,
    ops: [], tensors: [{ index: 0, name: "w", dtype: "F32", shape: [3], byte_length: 12, data_offset: 0, ...tensorRow }] });
}
function resign(document, field) {
  const body = structuredClone(document); delete body[field];
  return { ...body, [field]: sha256TextHex(canonicalJson(body)) };
}
const bytes = new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx"));
const base = context(bytes, analyzeOnnxModel(bytes, "cnn.onnx"), "cnn.onnx");
check("JCS object reordering preserves Artifact IR and Model IR validity", () => {
  for (const [ir, validate] of [[base.artifact_ir, validateArtifactEvidenceIr], [base.model_ir, validateModelIr]])
    assert.deepEqual(validate(JSON.parse(canonicalJson(ir))), ir);
});
check("Unicode namespace hashes are independent of the process locale", () => {
  const source = `import {getArtifactIrContext} from './web/lib/artifact-ir-context.js'; const a={format:'safetensors',filename:'unicode',model_sha256:'b'.repeat(64),file_size_bytes:24,ops:[],tensors:['ä.w','z.w','a.w'].map((name,index)=>({index,name,shape:[1],dtype:'F32',byte_length:4}))};for(const options of [{},{runtimeEvidence:{artifact_sha256:'b'.repeat(64),runtime_nodes:['ä.node','z.node','a.node'].map(runtime_node_ref=>({runtime_node_ref}))}}]) {const c=getArtifactIrContext(a,{},options);console.log(c.artifact_ir.artifact_ir_sha256,c.model_ir.model_ir_sha256,c.model_summary.model_summary_sha256);}`;
  const results = ["en_US.UTF-8", "sv_SE.UTF-8", "tr_TR.UTF-8"].map(locale => {
    const result = spawnSync(process.execPath, ["--input-type=module", "-e", source], { encoding: "utf8", env: { ...process.env, LANG: locale, LC_ALL: locale } });
    assert.equal(result.status, 0, result.stderr); return result.stdout;
  });
  assert.equal(new Set(results).size, 1);
});
check("JCS rejects sparse arrays and preserves valid zero/unicode serialization", () => {
  assert.throws(() => canonicalJson(new Array(2)), /undefined|sparse|JSON/);
  assert.equal(canonicalJson({ z: -0, a: [null, "한글"] }), '{"a":[null,"한글"],"z":0}');
});
check("IR validators reject non-JSON values before a lossy clone", () => {
  const artifact = structuredClone(base.artifact_ir); artifact.extra = undefined;
  assert.throws(() => validateArtifactEvidenceIr(artifact), /undefined|JSON/);
  const modelIr = structuredClone(base.model_ir); modelIr.program.operations[0].runtime_order = NaN;
  assert.throws(() => validateModelIr(modelIr), /non-finite|JSON/);
});
check("exact counts reject booleans, whitespace, arrays and unsafe numeric inputs", () => {
  for (const value of [true, false, " ", [], [2], {}, Number.MAX_SAFE_INTEGER + 1]) assert.equal(exactInteger(value), null);
  assert.deepEqual(exactInteger("9007199254740993"), { decimal: "9007199254740993", number: null });
});
check("unknown dimensions are not observed empty dimensions", () => {
  const unknown = synthetic({ shape: [null, 3] }).model_ir.tensors_and_storage.storage_objects[0];
  assert.notEqual(unknown.shape[0], 0); assert.equal(unknown.element_count, null);
  assert.deepEqual(synthetic({ shape: [0, 3] }).model_ir.tensors_and_storage.storage_objects[0].element_count, { decimal: "0", number: 0 });
});
check("canonical decimal dimensions preserve cardinality above 2^53", () => {
  assert.deepEqual(synthetic({ shape: ["9007199254740993", 2] }).model_ir.tensors_and_storage.storage_objects[0].element_count,
    { decimal: "18014398509481986", number: null });
});
check("a real SafeTensors rank-zero tensor has one element", () => {
  const header = new TextEncoder().encode(JSON.stringify({ scalar: { dtype: "F32", shape: [], data_offsets: [0, 4] } }));
  const file = new Uint8Array(8 + header.length + 4); new DataView(file.buffer).setBigUint64(0, BigInt(header.length), true); file.set(header, 8);
  const ir = context(file, parseMetadataModel(file, "scalar.safetensors", file.length, "safetensors"), "scalar.safetensors").model_ir;
  assert.deepEqual(ir.tensors_and_storage.storage_objects[0].element_count, { decimal: "1", number: 1 });
  assert.equal(synthetic({ shape: undefined }).model_ir.tensors_and_storage.storage_objects[0].element_count, null);
});
check("rehashed contradictory exact-number mirrors fail semantic validation", () => {
  for (const [ir, field, validate] of [[base.artifact_ir, "artifact_ir_sha256", validateArtifactEvidenceIr], [base.model_ir, "model_ir_sha256", validateModelIr]]) {
    const bad = structuredClone(ir); bad.artifact.byte_length.number += 1;
    assert.throws(() => validate(resign(bad, field)), /exact|mirror|integer/i);
  }
});
check("Model IR rejects duplicate storage identities and contradictory storage totals", () => {
  const duplicate = structuredClone(base.model_ir);
  duplicate.tensors_and_storage.storage_objects.push(structuredClone(duplicate.tensors_and_storage.storage_objects[0]));
  assert.throws(() => validateModelIr(resign(duplicate, "model_ir_sha256")), /storage|duplicat|conservation/i);
  const totals = structuredClone(base.model_ir); totals.tensors_and_storage.totals.serialized_object_bytes_sum = { decimal: "1", number: 1 };
  assert.throws(() => validateModelIr(resign(totals, "model_ir_sha256")), /storage|conservation/i);
  const cardinality = structuredClone(base.model_ir); cardinality.tensors_and_storage.storage_objects[0].element_count = { decimal: "999", number: 999 };
  assert.throws(() => validateModelIr(resign(cardinality, "model_ir_sha256")), /element count|shape/i);
});
check("external file digest is not mislabeled as a tensor payload digest", () => {
  const file = float32([10, 20, 30, 40]);
  const onnx = model([node("Add", "add", ["x", "w"], ["y"])],
    [externalTensor("w", 1, [2], [["location", "weights.bin"], ["offset", "4"], ["length", "8"]])],
    [valueInfo("x", 1, [2])], [valueInfo("y", 1, [2])], 13);
  const analysis = analyzeOnnxModel(onnx, "external.onnx", null, { externalDataFiles: [{ path: "weights.bin", bytes: file, sha256: sha256BytesHex(file) }] });
  const c = context(onnx, analysis, "external.onnx");
  for (const [schemaFile, document] of [["deepbom-artifact-ir-v2.schema.json", c.artifact_ir], ["deepbom-model-ir-v1.schema.json", c.model_ir]]) {
    const validate = new Ajv2020({ allErrors: true, strict: true }).compile(JSON.parse(readFileSync(`docs/schemas/${schemaFile}`, "utf8")));
    assert.equal(validate(document), true, JSON.stringify(validate.errors));
  }
  const storage = c.model_ir.tensors_and_storage.storage_objects[0];
  assert.notEqual(storage.payload_sha256, sha256BytesHex(file));
  assert.equal(storage.external_storage.status, "external_bound");
  assert.equal(storage.external_storage.source.file_sha256, sha256BytesHex(file));
});
check("partial native quantization samples do not claim a complete vector digest", () => {
  const c = synthetic({ dtype: "INT8", shape: [300], byte_length: 300, quant_scales: 300, quant_zero_points: 300,
    scale_sample: [0.1, 0.2], zero_point_sample: [0, 0], quantized_dimension: 0 });
  const record = c.artifact_ir.quantization_contracts.records[0];
  assert.equal(record.parameters.scale.count, 300); assert.equal(record.parameters.scale.sha256, null);
  assert.equal(record.completeness, "partial_serialized_contract");
});
check("declared multi-channel quantization is retained without vector samples", () => {
  const record = synthetic({ dtype: "INT8", quant_scales: 300, quant_zero_points: 300, quantized_dimension: null }).artifact_ir.quantization_contracts.records[0];
  assert.equal(record.parameterization.kind, "per_axis"); assert.deepEqual(record.parameterization.axes, []);
  assert.equal(record.parameters.scale.count, 300); assert.equal(record.parameters.scale.sha256, null);
  assert.equal(record.parameters.zero_point.inline_status, "partial_sample_not_complete_vector");
});
check("complete interface vectors take precedence over truncated samples", () => {
  const scale = Array.from({ length: 300 }, (_, i) => (i + 1) / 1024), zero = Array(300).fill(0);
  const c = synthetic({ dtype: "INT8", shape: [300], byte_length: 300, quant_scales: 300, quant_zero_points: 300,
    scale_sample: scale.slice(0, 2), zero_point_sample: zero.slice(0, 2), interface_scale_values: scale, interface_zero_point_values: zero, quantized_dimension: 0 });
  const record = c.artifact_ir.quantization_contracts.records[0];
  assert.equal(record.parameters.scale.count, 300); assert.equal(record.parameters.scale.sha256, sha256TextHex(canonicalJson(scale)));
  assert.equal(record.completeness, "complete_for_serialized_contract");
});
check("a real 300-channel ONNX quantization contract retains full parameter coverage", () => {
  const scales = Array.from({ length: 300 }, (_, i) => (i + 1) / 1024);
  const file = model([node("QuantizeLinear", "quantize", ["x", "scale", "zero"], ["y"])],
    [tensor("scale", 1, [300], float32(scales)), tensor("zero", 2, [300], new Uint8Array(300))],
    [valueInfo("x", 1, [1, 300])], [valueInfo("y", 2, [1, 300])], 13);
  const c = context(file, analyzeOnnxModel(file, "per-axis.onnx"), "per-axis.onnx");
  const value = c.artifact_ir.graph.values.find(row => row.name === "y");
  const record = c.artifact_ir.quantization_contracts.records.find(row => row.subject_ref === value.id);
  assert.equal(record.parameters.scale.count, 300); assert.equal(record.parameters.scale.sha256, sha256TextHex(canonicalJson(scales)));
});

for (const row of rows) console.log(`${row.status}: ${row.name}${row.error ? `: ${row.error}` : ""}`);
const baselinePath = process.argv.find(arg => arg.startsWith("--record-baseline="))?.split("=")[1];
if (baselinePath) writeFileSync(baselinePath, JSON.stringify(rows, null, 2));
else assert.equal(rows.filter(row => row.status === "fail").length, 0, "Common IR semantic regressions failed");
console.log(`Common IR semantics: ${rows.filter(row => row.status === "pass").length}/${rows.length} passed.`);
