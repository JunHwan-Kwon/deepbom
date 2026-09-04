import { readFileSync } from "node:fs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { initSync, analyze_deepbom } from "../web/protected/deepbom/pkg/deepbom_wasm.js";
import { applyProtectedOrtCompatibilityEvidence } from "../web/lib/ort-compatibility-evidence.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("ORT artifact precheck");

initSync({ module: readFileSync("web/protected/deepbom/pkg/deepbom_wasm_bg.wasm") });
const protectedOrtEvidence = (analysis) => analyze_deepbom(Uint8Array.of(0), JSON.stringify(analysis)).ort_compatibility_evidence;

const boolAdd = analyzeOnnxModel(model({
  nodes: [node("Add", "bool_add", ["a", "b"], ["c"])],
  inputs: [valueInfo("a", 9, [1, 4]), valueInfo("b", 9, [1, 4])],
  outputs: [valueInfo("c", 9, [1, 4])],
  opset: 14,
}), "bool_add.onnx");
const boolEvidence = protectedOrtEvidence(boolAdd);
const boolCpu = provider(boolEvidence, "wasm_cpu");
const boolCuda = provider(boolEvidence, "cuda");
const boolDirectml = provider(boolEvidence, "directml");
expectEqual(boolEvidence.source_condition_inventory?.source_rule_count, 1167, "Protected evidence must bind the complete pinned EP/operator rule inventory.");
expectEqual(boolEvidence.source_condition_inventory?.cpu_registration_variant_count, 611, "Protected evidence must bind every parsed CPU registration variant.");
expectEqual(boolEvidence.source_condition_inventory?.machine_condition_count, 540, "Protected evidence must bind every machine-evaluated Web and native EP condition.");
expectEqual(boolEvidence.source_condition_inventory?.unresolved_source_fragment_count, 432, "Protected evidence must preserve every uncompiled source fragment as unresolved.");
expectEqual(boolCpu.ops[0].status, "SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL", "BOOL Add must fail the pinned CPU type registration despite an op/version match.");
expectEqual(boolCpu.ops[0].artifact_conditions[0].condition_id, "cpu_kernel_type_contract", "CPU type failure must retain its condition identity.");
expect(boolCpu.ops[0].artifact_conditions[0].observed.includes("BOOL"), "CPU type failure must disclose the observed artifact dtype.");
expectEqual(boolCuda.ops[0].status, "SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL", "BOOL Add must fail the pinned CUDA type registration despite an op/version match.");
expectEqual(boolCuda.ops[0].artifact_conditions[0].condition_id, "cpu_kernel_type_contract", "CUDA type failure must retain the shared kernel type-contract identity.");
expect(boolCuda.ops[0].artifact_conditions[0].observed.includes("BOOL"), "CUDA type failure must disclose the observed artifact dtype.");
expectEqual(boolEvidence.portability_frontier.all_ep_source_match_op_count, 0, "The nine-profile common source intersection must remain empty when XNNPACK has no Add registration.");
const boolFrontierRow = boolEvidence.portability_frontier.ops[0];
expect(boolFrontierRow.source_match_eps.includes("wasm_cpu"), "BOOL Add should retain the CPU source-version match before artifact precheck.");
expect(!boolFrontierRow.artifact_precheck_candidate_eps.includes("wasm_cpu"), "A definite CPU type failure must remove BOOL Add from the CPU artifact-precheck candidate set.");
expectEqual(boolDirectml.ops[0].artifact_conditions.find((condition) => condition.condition_kind === "schema_version_in")?.status, "PASS", "DirectML Add must bind its resolved schema version through the pinned OperatorVersions header.");
expect(boolDirectml.source_sha256 === "82ba5f2f391978bd2fc36a821fd53ad9157afac56f0f16e6830b880966d9daa0"
  && boolDirectml.ops[0].artifact_conditions.some((condition) => condition.source_sha256 === "d6aac0145556dc3c6e3688c6fa26bfebf6f6e2e965fa1e58aba32860e26333d7"), "DirectML registration and schema-version evidence must preserve both pinned source hashes.");

const rank3Pool = analyzeOnnxModel(model({
  nodes: [node("MaxPool", "rank3_pool", ["x"], ["y"], "", [
    intsAttribute("kernel_shape", [2, 2]),
    intAttribute("storage_order", 0),
  ])],
  inputs: [valueInfo("x", 1, [1, 4, 8])],
  outputs: [valueInfo("y", 1, [1, 4, 4])],
  opset: 13,
}), "rank3_pool.onnx");
const rankEvidence = protectedOrtEvidence(rank3Pool);
const rankWebnn = provider(rankEvidence, "webnn");
expectEqual(rankWebnn.ops[0].artifact_precheck_status, "ARTIFACT_PRECHECK_DEFINITE_FAIL", "A 3-D MaxPool must fail the pinned WebNN 4-D condition.");
expect(rankWebnn.ops[0].artifact_conditions.some((condition) => condition.condition_id === "input_0_rank_4" && condition.status === "DEFINITE_FAIL"), "WebNN rank exclusion must retain the exact failing clause.");
expectEqual(rank3Pool.ops[0].onnx_attributes.find((attribute) => attribute.name === "kernel_shape")?.int_values.length, 2, "Repeated integer attributes must remain available to protected condition evaluation.");

const shapeBytes = int64([1n, 0n]);
const zeroShape = analyzeOnnxModel(model({
  nodes: [node("Reshape", "zero_shape", ["x", "shape"], ["y"])],
  initializers: [tensor("shape", 7, [2], shapeBytes)],
  inputs: [valueInfo("x", 1, [1, 4])],
  outputs: [valueInfo("y", 1, [1, 4])],
  opset: 13,
}), "zero_shape.onnx");
const shapeTensor = zeroShape.tensors.find((tensorRow) => tensorRow.name === "shape");
expectEqual(shapeTensor?.initializer_integer_values_status, "complete", "A bounded INT64 initializer must expose complete exact condition values.");
expectEqual(JSON.stringify(shapeTensor?.initializer_integer_values), JSON.stringify([1, 0]), "INT64 condition values must preserve the exact signed integer sequence.");
const shapeWebnn = provider(protectedOrtEvidence(zeroShape), "webnn");
expect(shapeWebnn.ops[0].artifact_conditions.some((condition) => condition.condition_id === "input_shape_contains_no_zero" && condition.status === "DEFINITE_FAIL"), "A zero in the Reshape shape initializer must trigger the pinned WebNN exclusion.");

const allowzeroShape = analyzeOnnxModel(model({
  nodes: [node("Reshape", "allowzero_shape", ["x", "shape"], ["y"], "", [intAttribute("allowzero", 1)])],
  initializers: [tensor("shape", 7, [2], shapeBytes)],
  inputs: [valueInfo("x", 1, [1, 4])],
  outputs: [valueInfo("y", 1, [1, 4])],
  opset: 14,
}), "allowzero_shape.onnx");
const allowzeroQnn = provider(protectedOrtEvidence(allowzeroShape), "qnn");
const allowzeroCondition = allowzeroQnn.ops[0].artifact_conditions.find((condition) => condition.condition_id === "qnn_reshape_allowzero_contract");
expectEqual(allowzeroCondition?.status, "DEFINITE_FAIL", "QNN Reshape allowzero=1 must reject an initializer shape containing zero.");
expect(allowzeroCondition?.source_ref.endsWith("reshape_op_builder.cc") && /^[a-f0-9]{64}$/.test(allowzeroCondition?.source_sha256 || ""), "QNN Reshape exclusion must carry its exact predicate-source identity.");

const dynamicQnnAnalysis = structuredClone(boolAdd);
dynamicQnnAnalysis.tensors.find((row) => row.name === "a").shape_signature = [-1, 4];
const dynamicQnn = provider(protectedOrtEvidence(dynamicQnnAnalysis), "qnn");
expect(dynamicQnn.ops[0].artifact_conditions.some((condition) => condition.condition_id === "qnn_all_io_shapes_static" && condition.status === "DEFINITE_FAIL"), "QNN static-shape gate must exclude an artifact-visible symbolic dimension.");

const rank6Add = analyzeOnnxModel(model({
  nodes: [node("Add", "rank6_add", ["a", "b"], ["c"])],
  inputs: [valueInfo("a", 1, [1, 1, 1, 1, 1, 1]), valueInfo("b", 1, [1, 1, 1, 1, 1, 1])],
  outputs: [valueInfo("c", 1, [1, 1, 1, 1, 1, 1])],
  opset: 14,
}), "rank6_add.onnx");
const rank6Coreml = provider(protectedOrtEvidence(rank6Add), "coreml");
expect(rank6Coreml.ops[0].artifact_conditions.some((condition) => condition.condition_id === "coreml_all_present_input_rank_at_most_five" && condition.status === "DEFINITE_FAIL"), "CoreML common builder gate must exclude an artifact-visible input rank above five.");

const dynamicNnapiAnalysis = structuredClone(rank6Add);
dynamicNnapiAnalysis.tensors.find((row) => row.name === "a").shape_signature = [-1, 1, 1, 1, 1, 1];
const dynamicNnapi = provider(protectedOrtEvidence(dynamicNnapiAnalysis), "nnapi");
expect(dynamicNnapi.ops[0].artifact_conditions.some((condition) => condition.condition_id === "nnapi_all_present_input_shapes_static" && condition.status === "DEFINITE_FAIL"), "NNAPI common builder gate must exclude an artifact-visible symbolic input dimension.");

const runtimeGemm = analyzeOnnxModel(model({
  nodes: [node("Gemm", "runtime_weight_gemm", ["a", "b"], ["y"])],
  inputs: [valueInfo("a", 1, [1, 4]), valueInfo("b", 1, [4, 4])],
  outputs: [valueInfo("y", 1, [1, 4])],
  opset: 13,
}), "runtime_weight_gemm.onnx");
const runtimeGemmXnnpack = provider(protectedOrtEvidence(runtimeGemm), "xnnpack");
expect(runtimeGemmXnnpack.ops[0].artifact_conditions.some((condition) => condition.condition_id === "xnnpack_gemm_weight_constant" && condition.status === "DEFINITE_FAIL"), "XNNPACK Gemm must exclude a runtime-provided B matrix when the pinned source requires a constant initializer.");

const indexedMaxPool = analyzeOnnxModel(model({
  nodes: [node("MaxPool", "indexed_pool", ["x"], ["y", "indices"], "", [intsAttribute("kernel_shape", [2, 2])])],
  inputs: [valueInfo("x", 1, [1, 1, 4, 4])],
  outputs: [valueInfo("y", 1, [1, 1, 2, 2]), valueInfo("indices", 7, [1, 1, 2, 2])],
  opset: 13,
}), "indexed_maxpool.onnx");
const indexedMaxPoolXnnpack = provider(protectedOrtEvidence(indexedMaxPool), "xnnpack");
expect(indexedMaxPoolXnnpack.ops[0].artifact_conditions.some((condition) => condition.condition_id === "xnnpack_maxpool_output_count" && condition.status === "DEFINITE_FAIL"), "XNNPACK MaxPool must exclude the optional indices output unsupported by the pinned source.");

const legacyConv = analyzeOnnxModel(model({
  nodes: [node("Conv", "legacy_conv", ["x", "w"], ["y"])],
  initializers: [tensor("w", 1, [1, 1, 1, 1], float32([1]))],
  inputs: [valueInfo("x", 1, [1, 1, 4, 4])],
  outputs: [valueInfo("y", 1, [1, 1, 4, 4])],
  opset: 10,
}), "legacy_conv.onnx");
const legacyConvXnnpack = provider(protectedOrtEvidence(legacyConv), "xnnpack");
expect(legacyConvXnnpack.ops[0].artifact_conditions.some((condition) => condition.condition_id === "xnnpack_conv_schema_minimum" && condition.status === "DEFINITE_FAIL"), "XNNPACK Conv must exclude a schema revision below the pinned source minimum even when a kernel range is registered.");

const rank3Conv = analyzeOnnxModel(model({
  nodes: [node("Conv", "rank3_conv", ["x", "w"], ["y"])],
  initializers: [tensor("w", 1, [1, 1, 3], float32([1, 1, 1]))],
  inputs: [valueInfo("x", 1, [1, 1, 8])],
  outputs: [valueInfo("y", 1, [1, 1, 6])],
  opset: 13,
}), "rank3_conv.onnx");
const rank3ConvXnnpack = provider(protectedOrtEvidence(rank3Conv), "xnnpack");
expectEqual(rank3ConvXnnpack.ops[0].artifact_conditions.find((condition) => condition.condition_id === "xnnpack_conv_required_dimensions_static")?.status, "PASS", "XNNPACK rank-3 Conv must require only C/W, not a nonexistent fourth dimension.");

const nearestResize = analyzeOnnxModel(model({
  nodes: [node("Resize", "nearest_resize", ["x", "", "scales"], ["y"])],
  initializers: [tensor("scales", 1, [4], float32([1, 1, 2, 2]))],
  inputs: [valueInfo("x", 1, [1, 1, 4, 4])],
  outputs: [valueInfo("y", 1, [1, 1, 8, 8])],
  opset: 13,
}), "nearest_resize.onnx");
const nearestResizeXnnpack = provider(protectedOrtEvidence(nearestResize), "xnnpack");
expect(nearestResizeXnnpack.ops[0].artifact_conditions.some((condition) => condition.condition_id === "xnnpack_resize_mode" && condition.status === "DEFINITE_FAIL"), "XNNPACK Resize must evaluate the schema-default nearest mode and exclude it when the pinned source requires linear mode.");

const signedAttribute = analyzeOnnxModel(model({
  nodes: [node("Flatten", "negative_axis", ["x"], ["y"], "", [intAttribute("axis", -1)])],
  inputs: [valueInfo("x", 1, [1, 2, 3])],
  outputs: [valueInfo("y", 1, [1, 6])],
  opset: 13,
}), "negative_axis.onnx");
const axis = signedAttribute.ops[0].onnx_attributes.find((attribute) => attribute.name === "axis");
expectEqual(axis?.int_value, -1, "AttributeProto INT values must decode as signed int64 rather than saturating negative varints.");
expectEqual(axis?.int_value_exact_decimal, "-1", "Signed attribute evidence must preserve its exact decimal representation.");

const modernTypeNames = new Map([
  [14, "COMPLEX64"], [15, "COMPLEX128"], [16, "BFLOAT16"],
  [17, "FLOAT8E4M3FN"], [18, "FLOAT8E4M3FNUZ"], [19, "FLOAT8E5M2"], [20, "FLOAT8E5M2FNUZ"],
  [21, "UINT4"], [22, "INT4"], [23, "FLOAT4E2M1"], [24, "FLOAT8E8M0"], [25, "UINT2"], [26, "INT2"],
]);
const modernTypes = analyzeOnnxModel(model({
  nodes: [], inputs: [],
  outputs: [...modernTypeNames].map(([type, name]) => valueInfo(`type_${type}_${name}`, type, [1])),
  opset: 26,
}), "modern_types.onnx");
for (const [type, name] of modernTypeNames) {
  expectEqual(modernTypes.tensors.find((tensorRow) => tensorRow.name === `type_${type}_${name}`)?.dtype, name, `TensorProto dtype ${type} must retain its pinned ONNX 1.21 name.`);
}

const lowBit = analyzeOnnxModel(model({
  nodes: [], inputs: [],
  initializers: [
    tensor("int4_raw", 22, [3], new Uint8Array([0x78, 0x0f])),
    tensor("uint2_raw", 25, [5], new Uint8Array([0xe4, 0x01])),
    tensor("fp8_raw", 17, [3], new Uint8Array([0x38, 0x7e, 0x7f])),
    tensor("fp4_raw", 23, [3], new Uint8Array([0x61, 0x07])),
    tensor("e8m0_raw", 24, [3], new Uint8Array([0x7f, 0x80, 0xff])),
    tensor("scalar_f32", 1, [], float32([2.5])),
  ],
  outputs: [
    valueInfo("int4_raw", 22, [3]), valueInfo("uint2_raw", 25, [5]), valueInfo("fp8_raw", 17, [3]),
    valueInfo("fp4_raw", 23, [3]), valueInfo("e8m0_raw", 24, [3]), valueInfo("scalar_f32", 1, []),
  ],
  opset: 26,
}), "low_bit_raw.onnx");
const int4Raw = lowBit.tensors.find((tensorRow) => tensorRow.name === "int4_raw");
const uint2Raw = lowBit.tensors.find((tensorRow) => tensorRow.name === "uint2_raw");
expectEqual(int4Raw?.initializer_bytes, 2, "Three INT4 values must occupy ceil(3*4/8)=2 raw bytes.");
expectEqual(int4Raw?.initializer_elements, 3, "Packed INT4 padding must not become a fourth logical element.");
expectEqual(JSON.stringify(int4Raw?.initializer_integer_values), JSON.stringify([-8, 7, -1]), "INT4 raw_data must decode low nibble first with signed two's-complement values.");
expectEqual(uint2Raw?.initializer_bytes, 2, "Five UINT2 values must occupy ceil(5*2/8)=2 raw bytes.");
expectEqual(JSON.stringify(uint2Raw?.initializer_integer_values), JSON.stringify([0, 1, 2, 3, 1]), "UINT2 raw_data must decode four LSB-first values per byte and discard only shape padding.");
expectEqual(lowBit.tensors.find((tensorRow) => tensorRow.name === "scalar_f32")?.initializer_elements, 1, "A rank-0 TensorProto initializer must count as one scalar element.");
expectEqual(lowBit.size_breakdown.constant_bytes, 16, "Mixed packed, FP8, FP4, E8M0, and scalar raw initializer bytes must conserve exactly.");
expectEqual(lowBit.weight_integrity.nan_tensors, 2, "Pinned FP8 and E8M0 NaN encodings must be decoded into numerical-integrity evidence.");
expectEqual(lowBit.weight_integrity.tensor_results.find((row) => row.tensor_name === "fp8_raw")?.max_abs_value, 448, "FLOAT8E4M3FN code 0x7e must decode to the exact finite maximum 448.");
expectEqual(lowBit.weight_integrity.max_abs_weight, null, "Unbound low-bit constants must not be promoted into the learned-weight magnitude aggregate.");
const lowBitBundle = buildEngineeringBundleArtifactFiles(lowBit, {
  reportContext: { identity: { filename: lowBit.filename, format: "onnx" }, generatedAt: "2026-07-21T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: lowBit.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(lowBit, { hash: "" }),
});
expectEqual(JSON.parse(lowBitBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}").evidence?.conformance_report?.status, "pass", "Modern ONNX dtype evidence must remain report/ML-BOM/conformance complete.");

const typedPacked = analyzeOnnxModel(model({
  nodes: [], inputs: [],
  initializers: [typedInt32Tensor("int4_typed", 22, [3], [0x78, 0x0f])],
  outputs: [valueInfo("int4_typed", 22, [3])],
  opset: 26,
}), "low_bit_typed.onnx");
const int4Typed = typedPacked.tensors.find((tensorRow) => tensorRow.name === "int4_typed");
expectEqual(int4Typed?.initializer_typed_data_bytes, 2, "Typed INT4 logical payload must use packed storage bytes rather than four bytes per int32_data slot.");
expectEqual(int4Typed?.initializer_elements, 3, "Typed INT4 shape padding must not inflate scalar cardinality.");
expectEqual(JSON.stringify(int4Typed?.initializer_integer_values), JSON.stringify([-8, 7, -1]), "Typed INT4 packed slots must decode identically to raw_data.");

const packedExternal = analyzeOnnxModel(model({
  nodes: [], inputs: [],
  initializers: [externalTensor("int4_external", 22, [3], "weights.bin", 2)],
  outputs: [valueInfo("int4_external", 22, [3])],
  opset: 26,
}), "low_bit_external.onnx");
expectEqual(packedExternal.onnx_external_data.tensors[0].expected_payload_bytes, 2, "External INT4 cardinality must use ceil(elements*bits/8), including odd element counts.");

const malformedPacked = analyzeOnnxModel(model({
  nodes: [], inputs: [],
  initializers: [tensor("short_int4", 22, [3], new Uint8Array([0x78]))],
  outputs: [valueInfo("short_int4", 22, [3])],
  opset: 26,
}), "short_int4.onnx");
expectEqual(malformedPacked.weight_integrity.initializer_tensors_unassessed, 1, "A packed payload shorter than ceil(elements*bits/8) must remain unassessed rather than decoding a partial tensor.");
expect(malformedPacked.weight_integrity.tensor_results[0].reason.includes("require 2 byte"), "Packed cardinality failure must disclose the exact required byte count.");

applyProtectedOrtCompatibilityEvidence(boolAdd, boolEvidence);
const tamperedConditionCount = structuredClone(boolEvidence);
provider(tamperedConditionCount, "wasm_cpu").artifact_condition_fail_count = 0;
expectThrows(() => applyProtectedOrtCompatibilityEvidence(boolAdd, tamperedConditionCount), "summary does not reproduce", "Browser verification must reject a tampered artifact-condition count.");
const tamperedCandidate = structuredClone(boolEvidence);
tamperedCandidate.portability_frontier.all_ep_artifact_precheck_candidate_op_count = 1;
expectThrows(() => applyProtectedOrtCompatibilityEvidence(boolAdd, tamperedCandidate), "narrowed candidate op count", "Browser verification must reject a tampered narrowed-candidate intersection.");
const tamperedInventory = structuredClone(boolEvidence);
tamperedInventory.source_condition_inventory.machine_condition_count = 539;
expectThrows(() => applyProtectedOrtCompatibilityEvidence(boolAdd, tamperedInventory), "source artifact-condition inventory", "Browser verification must reject a tampered pinned-source condition inventory.");

const bundle = buildEngineeringBundleArtifactFiles(boolAdd, {
  reportContext: { identity: { filename: boolAdd.filename, format: "onnx" }, generatedAt: "2026-07-21T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: boolAdd.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(boolAdd, { hash: "" }),
});
const report = bundle.find((file) => file.name === "engineering_report.md")?.data || "";
const evidenceDocument = JSON.parse(bundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(report.includes("SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL") && report.includes("cpu_kernel_type_contract") && report.includes("BOOL"), "Engineering Report must surface the exact EP condition exclusion, identifier, and observed dtype.");
expect(report.includes("All-EP narrowed artifact-precheck candidates"), "Engineering Report must separate source-version and narrowed candidate intersections.");
expectEqual(evidenceDocument.evidence?.conformance_report?.status, "pass", "Artifact-precheck Engineering Evidence must pass semantic conformance.");

const recursiveScope = analyzeOnnxModel(
  new Uint8Array(readFileSync("scripts/fixtures/onnx_recursive_scope.onnx")),
  "onnx_recursive_scope.onnx",
);
applyProtectedOrtCompatibilityEvidence(recursiveScope, protectedOrtEvidence(recursiveScope));
const recursiveScopeBundle = buildEngineeringBundleArtifactFiles(recursiveScope, {
  reportContext: { identity: { filename: recursiveScope.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: recursiveScope.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(recursiveScope, { hash: "" }),
});
expectEqual(JSON.parse(recursiveScopeBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}").evidence?.conformance_report?.status, "pass", "FunctionProto and control-flow models should preserve complete protected ORT provenance in the Engineering Bundle.");

done();

function provider(evidence, id) {
  const row = evidence.execution_providers.find((candidate) => candidate.execution_provider === id);
  if (!row) throw new Error(`Missing execution provider ${id}.`);
  return row;
}

function model({ nodes, initializers = [], inputs, outputs, opset }) {
  const graph = message([
    ...nodes.map((value) => bytesField(1, value)),
    stringField(2, "deepbom_ort_precheck_fixture"),
    ...initializers.map((value) => bytesField(5, value)),
    ...inputs.map((value) => bytesField(11, value)),
    ...outputs.map((value) => bytesField(12, value)),
  ]);
  const opsetImport = message([stringField(1, ""), varintField(2, opset)]);
  return message([varintField(1, 10), stringField(2, "deepbom_ort_precheck_fixture"), bytesField(7, graph), bytesField(8, opsetImport)]);
}

function node(opType, name, inputs, outputs, domain = "", attributes = []) {
  return message([
    ...inputs.map((value) => stringField(1, value)),
    ...outputs.map((value) => stringField(2, value)),
    stringField(3, name), stringField(4, opType),
    ...attributes.map((value) => bytesField(5, value)),
    ...(domain ? [stringField(7, domain)] : []),
  ]);
}

function intAttribute(name, value) {
  return message([stringField(1, name), varintField(3, value), varintField(20, 2)]);
}

function intsAttribute(name, values) {
  return message([stringField(1, name), ...values.map((value) => varintField(8, value)), varintField(20, 7)]);
}

function tensor(name, dtype, dims, raw) {
  return message([...dims.map((dim) => varintField(1, dim)), varintField(2, dtype), stringField(8, name), bytesField(9, raw)]);
}

function typedInt32Tensor(name, dtype, dims, packedValues) {
  return message([
    ...dims.map((dim) => varintField(1, dim)), varintField(2, dtype),
    bytesField(5, message(packedValues.map((value) => varint(value)))), stringField(8, name),
  ]);
}

function externalTensor(name, dtype, dims, location, length) {
  return message([
    ...dims.map((dim) => varintField(1, dim)), varintField(2, dtype), stringField(8, name),
    bytesField(13, stringEntry("location", location)), bytesField(13, stringEntry("length", String(length))),
    varintField(14, 1),
  ]);
}

function stringEntry(key, value) {
  return message([stringField(1, key), stringField(2, value)]);
}

function valueInfo(name, dtype, dims) {
  const shape = message(dims.map((dim) => bytesField(1, message([varintField(1, dim)]))));
  const tensorType = message([varintField(1, dtype), bytesField(2, shape)]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

function int64(values) {
  const out = new Uint8Array(values.length * 8);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, value, true));
  return out;
}

function float32(values) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return out;
}

function stringField(field, value) {
  return bytesField(field, new TextEncoder().encode(value));
}

function bytesField(field, value) {
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  return message([varint((field << 3) | 2), varint(bytes.length), bytes]);
}

function varintField(field, value) {
  return message([varint(field << 3), varint(value)]);
}

function varint(value) {
  let remaining = BigInt.asUintN(64, BigInt(value));
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return new Uint8Array(bytes);
}

function message(parts) {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}
