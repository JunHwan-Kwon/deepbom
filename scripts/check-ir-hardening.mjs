import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { validateArtifactEvidenceIr } from "../web/lib/artifact-ir.js";
import { validateModelIr, validateModelIrAgainstSource } from "../web/lib/model-ir.js";
import { buildValueType, valueElementCount } from "../web/lib/ir-value-type.js";
import { integerMetric, aggregateIntegerMetrics } from "../web/lib/ir-metric.js";
import { parameterVectorEvidence } from "../web/lib/ir-parameter-vector.js";
import { buildModelSummary, validateModelSummary, validateModelSummaryAgainstSource, renderModelSummaryTable } from "../web/lib/model-summary.js";
import { buildActivationIr } from "../web/lib/activation-ir.js";
import { inputManifestSha256, validateRuntimeProvenance } from "../web/lib/ir-runtime-provenance.js";
import { buildSingleFileArtifactSet, validateArtifactSet } from "../web/lib/artifact-set.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex, sha256BytesHex } from "../web/lib/sha256-sync.js";
import { model, node, tensor, valueInfo, valueInfoWithoutShape, float32, message, bytesField, stringField, varintField } from "./onnx-proto-fixture.mjs";

let count = 0;
function check(name, run) { run(); count++; console.log(`pass: ${name}`); }
function seal(value, field) { const body = structuredClone(value); delete body[field]; return { ...body, [field]: sha256TextHex(canonicalJson(body)) }; }
function context(bytes, name = "fixture.onnx", analysis = analyzeOnnxModel(bytes, name)) {
  return getArtifactIrContext(analysis, { filename: name, format: analysis.format, sha256: sha256BytesHex(bytes), size: bytes.length });
}
const base = context(new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx")));

check("zero-byte SafeTensors remains a logical tensor without fabricated storage or graph", () => {
  const header = new TextEncoder().encode(JSON.stringify({ empty: { dtype: "F32", shape: [0, 3], data_offsets: [0, 0] } }));
  const bytes = new Uint8Array(header.length + 8); new DataView(bytes.buffer).setBigUint64(0, BigInt(header.length), true); bytes.set(header, 8);
  const c = context(bytes, "empty.safetensors", parseMetadataModel(bytes, "empty.safetensors", bytes.length, "safetensors"));
  assert.equal(c.artifact_ir.logical_inventory.count, 1); assert.equal(c.model_ir.tensors_and_storage.logical_values.length, 1);
  assert.equal(c.model_ir.logical_inventory.values[0].element_count.decimal, "0");
  assert.equal(c.model_summary.totals.logical_value_count, 1);
  assert.equal(c.model_summary.totals.macs.decimal, null); assert.equal(c.model_summary.totals.macs.status, "not_assessable");
  assert.equal(c.model_ir.program.values.length, 0); assert.equal(c.model_ir.tensors_and_storage.storage_objects.length, 0);
});
check("real ONNX scalar and unknown rank retain different type contracts", () => {
  for (const [info, status, elements] of [[valueInfo("x", 1, []), "ranked", "1"], [valueInfoWithoutShape("x", 1), "unknown_rank", null]]) {
    const c = context(model([], [], [info], [], 13));
    const row = c.model_ir.logical_inventory.values[0]; assert.equal(row.type_contract.root.rank_status, status);
    assert.equal(row.element_count?.decimal ?? null, elements);
  }
});
check("serialized ONNX sequence, optional, map and sparse type trees survive both IR layers", () => {
  const dense = message([bytesField(1, message([varintField(1, 1), bytesField(2, message([]))]))]);
  const variants = [
    ["sequence", message([bytesField(4, message([bytesField(1, dense)]))])],
    ["optional", message([bytesField(9, message([bytesField(1, dense)]))])],
    ["map", message([bytesField(5, message([varintField(1, 8), bytesField(2, dense)]))])],
    ["sparse_tensor", message([bytesField(8, message([varintField(1, 1), bytesField(2, message([]))]))])],
  ];
  for (const [kind, encoded] of variants) {
    const info = message([stringField(1, "x"), bytesField(2, encoded)]), c = context(model([], [], [info], [], 16));
    assert.equal(c.model_ir.program.values[0].type_contract.root.kind, kind);
    assert.deepEqual(c.model_ir.program.values[0].type_contract, c.artifact_ir.graph.values[0].type_contract);
    validateModelIrAgainstSource(c.model_ir, c.artifact_ir);
  }
});
check("numeric-looking ONNX symbols are scoped symbols, and zero times symbolic is zero", () => {
  const type = buildValueType({ type_proto: { kind: "tensor", dtype: "FLOAT32", shapeDeclared: true, shape: ["123"], shapeDimensions: [{ kind: "symbolic", parameter: "123" }] } }, "onnx", "scope:a");
  assert.equal(type.root.dimensions[0].scope_ref, "symbol-scope:onnx:model");
  assert.equal(type.root.dimensions[0].kind, "symbol"); assert.equal(valueElementCount(type), null);
  assert.equal(valueElementCount(buildValueType({ shape: [0, "N"], dtype: "F32" }, "safetensors", "scope:a")).decimal, "0");
});
check("TFLite scalar rank depends on has_rank", () => {
  for (const has_rank of [false, true]) assert.equal(buildValueType({ shape: [], has_rank, dtype: "FLOAT32" }, "tflite", "scope:0").root.rank_status, has_rank ? "ranked" : "unknown_rank");
});
check("ONNX internal 300-channel vectors retain full digest beyond UI samples", () => {
  const scales = Array.from({ length: 300 }, (_, i) => Math.fround((i + 1) / 1024));
  const bytes = model([node("QuantizeLinear", "q", ["x", "s", "z"], ["internal"]), node("DequantizeLinear", "dq", ["internal", "s", "z"], ["y"])],
    [tensor("s", 1, [300], float32(scales)), tensor("z", 2, [300], new Uint8Array(300))], [valueInfo("x", 1, [1, 300])], [valueInfo("y", 1, [1, 300])], 13);
  const c = context(bytes), subject = c.artifact_ir.graph.values.find(row => row.name === "internal");
  const record = c.artifact_ir.quantization_contracts.records.find(row => row.subject_ref === subject.id);
  assert.equal(record.parameters.scale.sha256, sha256TextHex(canonicalJson(scales)));
  assert.equal(record.parameters.scale.count, 300); assert.equal(record.completeness, "complete_for_serialized_contract");
});
check("streamed vector hashes equal an independent canonical JSON digest", () => {
  for (const n of [0, 1, 32, 257, 10000]) {
    const values = Array.from({ length: n }, (_, i) => i % 2 ? -i / 16 : i / 32);
    assert.equal(parameterVectorEvidence(values, "scale").sha256, sha256TextHex(canonicalJson(values)));
  }
  assert.equal(parameterVectorEvidence([Number.MAX_SAFE_INTEGER + 1], "zero_point").sha256, null);
});
check("logical activation bytes derive from shape, not absent initializer storage", () => {
  const c = context(model([node("Identity", "id", ["x"], ["y"])], [], [valueInfo("x", 1, [2, 3])], [valueInfo("y", 1, [2, 3])], 13));
  for (const row of c.artifact_ir.graph.values) assert.equal(row.logical_byte_length.decimal, "24");
  const unknown = context(model([], [], [valueInfoWithoutShape("x", 1)], [], 13));
  assert.equal(unknown.artifact_ir.graph.values[0].logical_byte_length, null);
});
check("repeated native tensor indices in nested scopes cannot redirect primary quantization", () => {
  const analysis = { format: "tflite", filename: "scopes.tflite", model_sha256: "a".repeat(64), file_size_bytes: 100,
    tensors: [{ index: 0, name: "primary", shape: [1], dtype: "INT8", quant_scales: 1, quant_zero_points: 1, scale_sample: [0.5], zero_point_sample: [0] }], ops: [],
    tflite_subgraph_inventory: { rows: [{ subgraph_index: 1, tensor_count: 1, operator_count: 0,
      tensor_intrinsics: [{ tensor_index: 0, name: "nested", shape: [1], dtype: "INT8", buffer_data_length: 0, quant_scales: 1, quant_zero_points: 1, scale_sample: [0.75], zero_point_sample: [1] }], operator_intrinsics: [] }] } };
  const c = getArtifactIrContext(analysis), q = c.artifact_ir.quantization_contracts.records[0];
  assert.equal(q.subject_ref, c.artifact_ir.graph.values.find(row => row.name === "primary").id);
  const nested = c.artifact_ir.quantization_contracts.records[1];
  assert.equal(nested.subject_ref, c.artifact_ir.graph.values.find(row => row.name === "nested").id);
  assert.deepEqual(nested.parameters.scale.inline_values, [0.75]);
});
check("missing operator MACs never become a complete model total", () => {
  const c = getArtifactIrContext({ format: "onnx", filename: "unknown.onnx", model_sha256: "a".repeat(64), file_size_bytes: 1,
    tensors: [], ops: [{ index: 0, name: "Custom", inputs: [], outputs: [], macs: null }] });
  assert.equal(c.model_summary.totals.macs.decimal, null);
  assert.equal(c.artifact_ir.graph.totals.macs, null); assert.equal(c.artifact_ir.graph.totals.assessed_macs.decimal, "0");
});
check("metric aggregation rejects duplicate sources and mixed units; preserves partial totals", () => {
  const a = integerMetric({ value: { decimal: "9007199254740993", number: null }, unit: "MAC", sourceRefs: ["a"], assessed: 1, eligible: 1 });
  const b = integerMetric({ value: null, unit: "MAC", sourceRefs: ["b"], assessed: 0, eligible: 1 });
  assert.equal(aggregateIntegerMetrics([a, b], "MAC").total.value, null);
  assert.equal(aggregateIntegerMetrics([a, b], "MAC").assessed_subtotal.decimal, "9007199254740993");
  assert.throws(() => aggregateIntegerMetrics([a, a], "MAC"), /twice/);
  assert.throws(() => aggregateIntegerMetrics([a], "byte"), /unit/);
});
check("rehashed graph consumer and producer contradictions are rejected", () => {
  for (const key of ["consumers", "producer"]) {
    const copy = structuredClone(base.artifact_ir), row = copy.graph.values.find(row => key === "consumers" ? row.consumers.length : row.producer);
    if (key === "consumers") row.consumers[0].port += 100; else row.producer.port += 100;
    assert.throws(() => validateArtifactEvidenceIr(seal(copy, "artifact_ir_sha256")), /port|producer|consumer/);
  }
});
check("source rederivation catches self-consistent fabricated derived explanations", () => {
  assert.equal(validateModelIrAgainstSource(base.model_ir, base.artifact_ir).status, "source_projection_verified");
  const bad = structuredClone(base.model_ir); bad.interpretation_boundary = "changed after derivation";
  const signed = seal(bad, "model_ir_sha256"); validateModelIr(signed);
  assert.throws(() => validateModelIrAgainstSource(signed, base.artifact_ir), /rederived/);
});
check("summary source validation and numeric mirrors reject rehashed contradictions", () => {
  validateModelSummaryAgainstSource(base.model_summary, base.model_ir);
  const bad = structuredClone(base.model_summary); bad.totals.macs.number++;
  assert.throws(() => validateModelSummary(seal(bad, "model_summary_sha256")), /mirror/);
  const changed = structuredClone(base.model_summary); changed.rows[0].name = "invented";
  assert.throws(() => validateModelSummaryAgainstSource(seal(changed, "model_summary_sha256"), base.model_ir), /source/);
});
check("summary MAC total excludes nested branch bodies and retains scalar contracts", () => {
  const c = context(new Uint8Array(readFileSync("scripts/fixtures/onnx_recursive_scope.onnx")));
  const refs = new Set(c.model_ir.program.programs.flatMap(row => row.entry_region_refs));
  const ops = c.model_ir.program.operations.filter(row => refs.has(row.region_ref));
  assert(ops.length < c.model_ir.program.operations.length);
  assert.equal(c.model_summary.totals.macs.total_count, ops.length);
  const scalar = context(model([node("Identity", "id", ["x"], ["y"])], [], [valueInfo("x", 1, [])], [valueInfo("y", 1, [])], 13));
  assert.match(renderModelSummaryTable(scalar.model_summary), /FLOAT32\[\]/);
});
check("out-of-bounds observed storage remains reportable with an explicit failed bounds assessment", () => {
  const c = getArtifactIrContext({ format: "tflite", filename: "bad.tflite", file_size: 128, model_sha256: "c".repeat(64),
    tensors: [{ index: 0, name: "bad", shape: [4], dtype: "FLOAT32", buffer_data_offset: 120, buffer_data_length: 16 }], ops: [] });
  assert.equal(c.artifact_ir.artifact_members.storage_bindings[0].bounds_assessment, "outside_artifact_member");
  const bad = structuredClone(c.artifact_ir); bad.artifact_members.storage_bindings[0].bounds_assessment = "within_artifact_member";
  assert.throws(() => validateArtifactEvidenceIr(seal(bad, "artifact_ir_sha256")), /bounds assessment/);
});
check("artifact member bindings cannot reference missing files or silently omit storage", () => {
  const bad = structuredClone(base.artifact_ir); bad.artifact_members.storage_bindings[0].member_ref = "missing";
  assert.throws(() => validateArtifactEvidenceIr(seal(bad, "artifact_ir_sha256")), /member/);
  bad.artifact_members.storage_bindings = [];
  assert.throws(() => validateArtifactEvidenceIr(seal(bad, "artifact_ir_sha256")), /omitted/);
});
check("shared storage ranges are reported without double-counting an overlap defect", () => {
  const c = getArtifactIrContext({ format: "safetensors", filename: "shared.safetensors", model_sha256: "a".repeat(64), file_size_bytes: 16, ops: [],
    tensors: [0,1].map(index => ({ index, name: `w${index}`, dtype: "F32", shape: [1], byte_length: 4, data_offset: index * 2 })) });
  assert.equal(c.artifact_ir.artifact_members.overlap_groups.length, 1);
  assert.equal(c.artifact_ir.storage_topology.totals.serialized_object_bytes_sum.decimal, "8");
  const bad = structuredClone(c.artifact_ir); bad.artifact_members.overlap_groups = [];
  assert.throws(() => validateArtifactEvidenceIr(seal(bad, "artifact_ir_sha256")), /overlap/);
});
check("artifact-set exact sizes reject string mirrors and non-JSON metadata", () => {
  const set = buildSingleFileArtifactSet({ filename: "test.onnx", format: "onnx", sha256: "a".repeat(64), byteLength: 10 });
  const bad = structuredClone(set); bad.files[0].byte_length.number = "10";
  assert.throws(() => validateArtifactSet(seal(bad, "artifact_set_sha256")), /differ/);
  assert.throws(() => validateArtifactSet({ ...set, missing: undefined }), /undefined|JSON/);
});

function captureFor(ir, shape) {
  const input = ir.program.values.find(row => row.roles.includes("graph_input")), config = {};
  return { schema: "deepbom.activation_capture.v1", source: { model_ir_sha256: ir.model_ir_sha256, artifact_sha256: ir.artifact.sha256, artifact_set_sha256: ir.artifact_set.artifact_set_sha256 },
    run: { id: "run", started_at: "2026-09-24T00:00:00Z", entry_region_ref: input.region_ref, runtime: { name: "fixture", version: "1", configured_providers: [], device: null },
      collector: { name: "fixture", version: "1", sha256: "b".repeat(64) }, execution: { artifact_sha256: ir.artifact.sha256, instrumented_artifact_sha256: null, configuration: config, configuration_sha256: sha256TextHex(canonicalJson(config)) },
      probe: { kind: "synthetic_ones", description: "test" }, runtime_evidence: null },
    inputs: [{ value_ref: input.id, native_locator: "x", dtype: "FLOAT32", shape, values: Array(shape.reduce((a, b) => a * b, 1)).fill(1) }], requested_value_refs: [], captures: [], missing: [] };
}
check("execution captures bind unknown-rank input without treating it as a scalar", () => {
  const ir = context(model([], [], [valueInfoWithoutShape("x", 1)], [], 13)).model_ir;
  assert.equal(buildActivationIr(ir, captureFor(ir, [2, 3])).inputs[0].statistics.value_count, "6");
});
check("execution rejects conflicting dimensions bound to one ONNX symbol", () => {
  const symbol = name => message([stringField(1, name), bytesField(2, message([bytesField(1, message([varintField(1, 1), bytesField(2, message([bytesField(1, message([stringField(2, "N")]))]))]))]))]);
  const ir = context(model([], [], [symbol("x"), symbol("y")], [], 13)).model_ir, capture = captureFor(ir, [2]);
  const second = ir.program.values.find(row => row.name === "y");
  capture.inputs.push({ value_ref: second.id, native_locator: "y", dtype: "FLOAT32", shape: [3], values: [1,1,1] });
  assert.throws(() => buildActivationIr(ir, capture), /symbolic dimension/);
  capture.inputs[1].shape = [2]; capture.inputs[1].values = [1,1]; buildActivationIr(ir, capture);
});
check("execution provenance binds input values, environment and trace without attestation claims", () => {
  const ir = context(model([], [], [valueInfo("x", 1, [2])], [], 13)).model_ir;
  const capture = captureFor(ir, [2]), initial = buildActivationIr(ir, capture), environment = { device: "test CPU" };
  capture.run.provenance = { schema: "deepbom.runtime_provenance.v1", environment, environment_sha256: sha256TextHex(canonicalJson(environment)),
    input_manifest_sha256: inputManifestSha256(initial.inputs), code_sha256: "c".repeat(64), trace_sha256: null, signature: null, trust: "declared_not_attested" };
  buildActivationIr(ir, capture);
  capture.inputs[0].values[0] = 2; assert.throws(() => buildActivationIr(ir, capture), /input digest/);
  const bad = { ...capture.run.provenance, trust: "verified" }; assert.throws(() => validateRuntimeProvenance(bad, initial.inputs, capture.run), /contract/);
});
console.log(`IR hardening: ${count}/${count} regression groups passed.`);
