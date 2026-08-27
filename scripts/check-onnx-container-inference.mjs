import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { inferOnnxShapes } from "../web/lib/onnx-shape-inference.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import {
  canonicalOnnxTypeProto,
  makeOnnxOptionalType,
  makeOnnxSequenceType,
  makeOnnxTensorType,
} from "../web/lib/onnx-type-proto.js";

const { done, expect, expectEqual } = createCheck("ONNX container-value inference");

const tensorMap = new Map([
  ["a", tensor("a", "FLOAT32", [2, 3])],
  ["b", tensor("b", "FLOAT32", [2, 5])],
  ["position", constant("position", "INT64", [], [1])],
  ["split", constant("split", "INT64", [2], [2, 3])],
]);
const nodes = [
  node("SequenceConstruct", ["a", "b"], ["seq"]),
  node("SequenceLength", ["seq"], ["length"]),
  node("SequenceAt", ["seq", "position"], ["picked"]),
  node("SequenceInsert", ["seq", "a", "position"], ["seq3"]),
  node("SequenceErase", ["seq3", "position"], ["seq2"]),
  node("ConcatFromSequence", ["seq2"], ["concat"], { axis: intAttribute(1) }),
  node("SplitToSequence", ["b", "split"], ["parts"], { axis: intAttribute(1) }),
  node("ConcatFromSequence", ["parts"], ["joined"], { axis: intAttribute(1) }),
  node("Optional", ["a"], ["present"]),
  node("OptionalHasElement", ["present"], ["has"]),
  node("OptionalGetElement", ["present"], ["unwrapped"]),
  node("Optional", [], ["empty"], { type: typeAttribute(makeOnnxTensorType("FLOAT32", [2, 3], true)) }),
  node("OptionalHasElement", ["empty"], ["has_empty"]),
  node("SequenceEmpty", [], ["empty_seq"], { dtype: intAttribute(1) }),
  node("Identity", ["seq2"], ["seq_identity"]),
];
for (const item of nodes) for (const name of [...item.inputs, ...item.outputs]) if (name && !tensorMap.has(name)) tensorMap.set(name, { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false });

const evidence = inferOnnxShapes({ nodes, inputs: [], outputs: [], valueInfo: [], initializers: [], sparseInitializers: [] }, tensorMap, [{ domain: "", version: 24 }], typeName);
const container = evidence.container_value_inference;

expectEqual(container.status, "assessed", "All deterministic Sequence/Optional fixture rows should be assessed.");
expectEqual(container.assessed_node_count, 15, "Every direct container node, including non-dense Identity, should enter the ledger.");
expectEqual(container.failed_node_count, 0, "The valid fixture should have no container contract failure.");
expectEqual(container.exact_sequence_length_output_count, 6, "Construct, insert, erase, split, empty, and identity sequence outputs should carry exact lengths.");
expectEqual(tensorMap.get("seq").sequenceLength, 2, "SequenceConstruct length should equal variadic input count.");
expectEqual(tensorMap.get("length").staticValues?.[0], 2, "SequenceLength should emit an exact INT64 scalar when length is known.");
expectEqual(JSON.stringify(tensorMap.get("picked").shape), JSON.stringify([2, 5]), "SequenceAt should select the exact per-element type at a constant position.");
expectEqual(tensorMap.get("seq3").sequenceLength, 3, "SequenceInsert should increment an exact length.");
expectEqual(tensorMap.get("seq2").sequenceLength, 2, "SequenceErase should decrement an exact length.");
expectEqual(JSON.stringify(tensorMap.get("concat").shape), JSON.stringify([2, 8]), "ConcatFromSequence should sum exact per-element axis extents.");
expectEqual(tensorMap.get("parts").sequenceLength, 2, "SplitToSequence vector split should emit exact part count.");
expectEqual(JSON.stringify(tensorMap.get("joined").shape), JSON.stringify([2, 5]), "Split and concat should conserve the exact source shape.");
expectEqual(tensorMap.get("has").staticValues?.[0], 1, "OptionalHasElement should be statically true for Optional(input).");
expectEqual(tensorMap.get("has_empty").staticValues?.[0], 0, "OptionalHasElement should be statically false for Optional(type).");
expectEqual(JSON.stringify(tensorMap.get("unwrapped").shape), JSON.stringify([2, 3]), "OptionalGetElement should recover the exact contained tensor contract.");
expectEqual(tensorMap.get("empty_seq").sequenceLength, 0, "SequenceEmpty should emit exact length zero.");
expectEqual(tensorMap.get("seq_identity").sequenceLength, 2, "Identity should preserve a non-dense sequence state.");
expectEqual(evidence.known_non_dense_node_output_count, 8, "Every sequence/optional node output should be known separately from dense outputs.");
expectEqual(evidence.unresolved_non_dense_node_output_count, 0, "The valid fixture should leave no non-dense output unresolved.");
expectEqual(evidence.unknown_node_output_count, 0, "Dense outputs derived from container operations should be known.");
expectEqual(evidence.status, "assessed", "Known non-dense outputs should no longer force shape coverage partial.");
expect(canonicalOnnxTypeProto(tensorMap.get("seq").typeProto).startsWith("sequence<tensor<FLOAT32"), "Sequence output should retain a canonical recursive TypeProto.");

const invalid = runInvalid([
  node("SequenceConstruct", ["a", "i"], ["bad_seq"]),
], new Map([
  ["a", tensor("a", "FLOAT32", [2])],
  ["i", tensor("i", "INT32", [2])],
]));
expectEqual(invalid.container_value_inference.status, "fail", "Mixed element dtypes must fail SequenceConstruct.");
expect(invalid.container_value_inference.rows[0].reason_codes.some((reason) => reason.includes("element_type_mismatch")), "The exact dtype mismatch reason should be retained.");

const emptyGetMap = new Map();
const emptyGetNodes = [
  node("Optional", [], ["empty"], { type: typeAttribute(makeOnnxTensorType("FLOAT32", [1], true)) }),
  node("OptionalGetElement", ["empty"], ["bad"]),
];
for (const item of emptyGetNodes) for (const name of item.outputs) emptyGetMap.set(name, { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false });
const emptyGet = inferOnnxShapes({ nodes: emptyGetNodes, inputs: [], outputs: [], valueInfo: [], initializers: [], sparseInitializers: [] }, emptyGetMap, [{ domain: "", version: 18 }], typeName);
expectEqual(emptyGet.container_value_inference.status, "fail", "OptionalGetElement on a provably empty optional must fail before runtime.");
expectEqual(emptyGet.container_value_inference.rows[1].reason_codes[0], "optional_get_element_provably_empty", "Empty optional failure should use a stable reason code.");

const emptyConcat = runInvalid([
  node("SequenceEmpty", [], ["empty_seq"], { dtype: intAttribute(1) }),
  node("ConcatFromSequence", ["empty_seq"], ["bad"], { axis: intAttribute(0) }),
], new Map());
expectEqual(emptyConcat.container_value_inference.status, "fail", "ConcatFromSequence on a provably empty sequence must fail before runtime.");
expectEqual(emptyConcat.container_value_inference.failed_rows[0].reason_codes[0], "concat_from_sequence_provably_empty", "Empty concat failure should retain a stable reason code.");

const exactInventorySequence = descriptorFromType("inventory_seq", makeOnnxSequenceType(makeOnnxTensorType("FLOAT32")));
Object.assign(exactInventorySequence, {
  sequenceLengthStatus: "assessed_exact",
  sequenceLength: 2,
  sequenceElementInventoryStatus: "assessed_exact",
  sequenceElementTypes: [makeOnnxTensorType("FLOAT32", [1], true), makeOnnxTensorType("FLOAT32", [2], true)],
});
const exactInventoryConcat = runInvalid([
  node("ConcatFromSequence", ["inventory_seq"], ["inventory_concat"], { axis: intAttribute(0) }),
], new Map([["inventory_seq", exactInventorySequence]]));
expectEqual(JSON.stringify(exactInventoryConcat.container_value_inference.rows[0]?.canonical_output_types), JSON.stringify(["tensor<FLOAT32[3]>"]), "Exact sequence inventory must recover ConcatFromSequence shape even when its aggregate element rank is unknown.");

const mismatchedInventorySequence = descriptorFromType("mismatched_inventory_seq", makeOnnxSequenceType(makeOnnxTensorType("FLOAT32")));
Object.assign(mismatchedInventorySequence, {
  sequenceLengthStatus: "assessed_exact",
  sequenceLength: 2,
  sequenceElementInventoryStatus: "assessed_exact",
  sequenceElementTypes: [makeOnnxTensorType("FLOAT32", [1], true), makeOnnxTensorType("INT64", [1], true)],
});
const mismatchedInventoryConcat = runInvalid([
  node("ConcatFromSequence", ["mismatched_inventory_seq"], ["bad_inventory_concat"], { axis: intAttribute(0) }),
], new Map([["mismatched_inventory_seq", mismatchedInventorySequence]]));
expectEqual(mismatchedInventoryConcat.container_value_inference.failed_rows[0]?.reason_codes[0], "concat_from_sequence_inventory_dtype_mismatch", "Exact sequence inventory with mixed element dtypes must fail closed.");

const zeroSplit = runInvalid([
  node("SplitToSequence", ["a", "zero_split"], ["bad"]),
], new Map([
  ["a", tensor("a", "FLOAT32", [2])],
  ["zero_split", constant("zero_split", "INT64", [1], [0])],
]));
expectEqual(zeroSplit.container_value_inference.failed_rows[0].reason_codes[0], "split_to_sequence_split_values_not_positive", "SplitToSequence must reject non-positive split values under the pinned operator contract.");

const invalidSequenceType = makeOnnxSequenceType(makeOnnxOptionalType(makeOnnxTensorType("FLOAT32", [1], true)));
const invalidSequence = runInvalid([
  node("SequenceLength", ["bad_sequence"], ["length"]),
], new Map([["bad_sequence", descriptorFromType("bad_sequence", invalidSequenceType)]]));
expectEqual(invalidSequence.container_value_inference.failed_rows[0].reason_codes[0], "sequence_length_input_not_typed_tensor_sequence", "Sequence operators must reject a non-tensor element sequence.");

const nestedOptionalType = makeOnnxOptionalType(makeOnnxOptionalType(makeOnnxTensorType("FLOAT32", [1], true)));
const invalidOptional = runInvalid([
  node("OptionalGetElement", ["nested_optional"], ["bad"]),
], new Map([["nested_optional", descriptorFromType("nested_optional", nestedOptionalType)]]));
expectEqual(invalidOptional.container_value_inference.failed_rows[0].reason_codes[0], "optional_get_element_input_type_invalid", "Optional operators must reject nested Optional values outside the pinned schema type set.");

const mapType = { kind: "map", keyTypeName: "INT64", valueType: makeOnnxTensorType("FLOAT32", [1], true), valueFieldsPresent: [5] };
const invalidIdentity = runInvalid([
  node("Identity", ["map_value"], ["bad"]),
], new Map([["map_value", descriptorFromType("map_value", mapType)]]));
expectEqual(invalidIdentity.container_value_inference.failed_rows[0].reason_codes[0], "identity_non_dense_type_not_supported", "Identity must not promote map or sparse values beyond its pinned type constraint.");

const parsed = analyzeOnnxModel(serializedContainerModel(), "serialized_container.onnx");
const parsedContainer = parsed.onnx_shape_inference.container_value_inference;
expectEqual(parsedContainer.status, "assessed", "Serialized ModelProto container contracts should be fully assessed through the public parser.");
expectEqual(parsedContainer.assessed_node_count, 11, "Every serialized direct Sequence/Optional node should enter the evidence ledger.");
expectEqual(parsed.onnx_shape_inference.extended_scope_inference.sequence_map_pass_count, 1, "Serialized SequenceMap should execute its pinned body contract.");
expectEqual(parsed.tensors.find((item) => item.name === "length")?.static_values?.[0], 2, "Serialized SequenceLength should preserve its exact scalar in public tensor evidence.");
expectEqual(parsed.tensors.find((item) => item.name === "has")?.static_values?.[0], 1, "Serialized OptionalHasElement(present) should preserve true in public evidence.");
expectEqual(parsed.tensors.find((item) => item.name === "has_empty")?.static_values?.[0], 0, "Serialized OptionalHasElement(empty) should preserve false in public evidence.");
expectEqual(JSON.stringify(parsed.tensors.find((item) => item.name === "joined")?.shape), JSON.stringify([2, 5]), "Serialized SplitToSequence/ConcatFromSequence should recover the exact dense shape.");
expectEqual(parsed.tensors.find((item) => item.name === "seq_identity")?.sequence_length, 2, "Public evidence should preserve the exact non-dense Identity sequence length.");
expectEqual(parsed.tensors.find((item) => item.name === "mapped_seq")?.sequence_length, 2, "Serialized SequenceMap should preserve exact input length.");
expectEqual(parsed.tensors.find((item) => item.name === "conditional_seq")?.sequence_length, 2, "Serialized non-dense If should preserve an equal exact branch length.");
expect(!buildFindingsRegister(parsed).some((item) => item.finding_id === "EA-ONX-0012"), "A valid serialized container model must not emit EA-ONX-0012.");
const parsedBundle = bundle(parsed);
expectEqual(parsedBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized container analysis should pass report, ML-BOM, finding, and evidence conformance.");
expect(parsedBundle.report.includes("Sequence / Optional Value Contracts")
  && parsedBundle.report.includes("ConcatFromSequence")
  && parsedBundle.report.includes(`${parsedContainer.exact_sequence_length_output_count} sequence length output(s)`), "Engineering Report should expose deterministic container rows and exact-fact counts.");
assertCompactMlBomProjection(parsedBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxExactSequenceLengthOutputs"],
  label: "ONNX container compact ML-BOM",
});

const parsedInvalid = analyzeOnnxModel(serializedEmptyOptionalGetModel(), "serialized_empty_optional_get.onnx");
const invalidContainerFinding = buildFindingsRegister(parsedInvalid).find((item) => item.finding_id === "EA-ONX-0012");
expectEqual(parsedInvalid.onnx_shape_inference.container_value_inference.failed_rows[0].reason_codes[0], "optional_get_element_provably_empty", "Serialized empty OptionalGetElement must fail deterministically.");
expect(Boolean(invalidContainerFinding) && invalidContainerFinding.technical_priority === "High", "Serialized container failure must enter the High action queue.");
expectEqual(bundle(parsedInvalid).evidence.evidence?.conformance_report?.status, "pass", "A correctly reported serialized container failure should still pass export conformance.");

done("pinned Sequence/Optional TypeProto, length, presence, scalar, bounds, split/concat, serialized parser/export, and fail-closed contracts");

function runInvalid(nodes, rows) {
  for (const item of nodes) for (const name of item.outputs) if (!rows.has(name)) rows.set(name, { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false });
  return inferOnnxShapes({ nodes, inputs: [], outputs: [], valueInfo: [], initializers: [], sparseInitializers: [] }, rows, [{ domain: "", version: 24 }], typeName);
}

function tensor(name, dtype, shape) {
  return { name, dtype, shape: [...shape], shapeDeclared: true, valueKind: "tensor", typeProto: makeOnnxTensorType(dtype, shape, true) };
}

function constant(name, dtype, shape, values) {
  return { ...tensor(name, dtype, shape), staticValuesStatus: "assessed_exact_static_data", staticValuesComplete: true, staticValues: [...values], staticValuesSource: "fixture" };
}

function descriptorFromType(name, typeProto) {
  return { name, valueKind: typeProto.kind, typeProto, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
}

function node(opType, inputs, outputs, attributes = {}) {
  return { name: opType, opType, domain: "", overload: "", inputs, outputs, attributes: new Map(Object.entries(attributes)), duplicateAttributeNames: [] };
}

function intAttribute(i) { return { type: 2, i, ints: [], valueTypesPresent: [2], duplicateValueTypes: [] }; }
function typeAttribute(typeProto) { return { type: 13, typeProto, valueTypesPresent: [13], duplicateValueTypes: [] }; }
function typeName(id) {
  return ["UNDEFINED", "FLOAT32", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL", "FLOAT16", "FLOAT64", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT4", "FLOAT4E2M1", "FLOAT8E8M0", "UINT2", "INT2"][id] || `TYPE_${id}`;
}

function bundle(analysis) {
  const mlBom = buildMlBomDocument(analysis, { hash: "" });
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
    rawEvidenceContext: { identity: { filename: analysis.filename, format: "onnx" } },
    mlBomDocument: mlBom,
  });
  return {
    report: files.find((file) => file.name === "engineering_report.md")?.data || "",
    evidence: JSON.parse(files.find((file) => file.name === "engineering_evidence.json")?.data || "{}"),
    mlBom,
  };
}

function serializedContainerModel() {
  const float23 = tensorTypeProto(1, [2, 3]);
  const float25 = tensorTypeProto(1, [2, 5]);
  const sequenceFloatUnknown = sequenceTypeProto(tensorTypeProto(1, [2, -1]));
  const mapBody = graphProto({
    nodes: [nodeProto("Identity", ["item"], ["mapped_item"])],
    inputs: [valueInfoProto("item", tensorTypeProto(1, [2, -1]))],
    outputs: [valueInfoProto("mapped_item", tensorTypeProto(1, [2, -1]))],
  });
  const thenBranch = graphProto({
    nodes: [nodeProto("Identity", ["mapped_seq"], ["then_seq"])],
    outputs: [valueInfoProto("then_seq", sequenceFloatUnknown)],
  });
  const elseBranch = graphProto({
    nodes: [nodeProto("Identity", ["seq_identity"], ["else_seq"])],
    outputs: [valueInfoProto("else_seq", sequenceFloatUnknown)],
  });
  const graph = graphProto({
    nodes: [
      nodeProto("SequenceConstruct", ["a", "b"], ["seq"]),
      nodeProto("SequenceLength", ["seq"], ["length"]),
      nodeProto("SequenceAt", ["seq", "position"], ["picked"]),
      nodeProto("SplitToSequence", ["b", "split"], ["parts"], [intAttributeProto("axis", 1)]),
      nodeProto("ConcatFromSequence", ["parts"], ["joined"], [intAttributeProto("axis", 1)]),
      nodeProto("Optional", ["a"], ["present"]),
      nodeProto("OptionalHasElement", ["present"], ["has"]),
      nodeProto("OptionalGetElement", ["present"], ["unwrapped"]),
      nodeProto("Optional", [], ["empty"], [typeAttributeProto("type", float23)]),
      nodeProto("OptionalHasElement", ["empty"], ["has_empty"]),
      nodeProto("Identity", ["seq"], ["seq_identity"]),
      nodeProto("SequenceMap", ["seq_identity"], ["mapped_seq"], [graphAttributeProto("body", mapBody)]),
      nodeProto("If", ["cond"], ["conditional_seq"], [graphAttributeProto("then_branch", thenBranch), graphAttributeProto("else_branch", elseBranch)]),
    ],
    initializers: [
      tensorProto("position", 7, [], int64Bytes([1n])),
      tensorProto("split", 7, [2], int64Bytes([2n, 3n])),
    ],
    inputs: [valueInfoProto("a", float23), valueInfoProto("b", float25), valueInfoProto("cond", tensorTypeProto(9, []))],
    outputs: [
      valueInfoProto("length", tensorTypeProto(7, [])),
      valueInfoProto("picked", float25),
      valueInfoProto("joined", float25),
      valueInfoProto("has", tensorTypeProto(9, [])),
      valueInfoProto("unwrapped", float23),
      valueInfoProto("has_empty", tensorTypeProto(9, [])),
      valueInfoProto("mapped_seq", sequenceFloatUnknown),
      valueInfoProto("conditional_seq", sequenceFloatUnknown),
    ],
  });
  return modelProto(graph, 18, "deepbom_container_fixture");
}

function serializedEmptyOptionalGetModel() {
  const float1 = tensorTypeProto(1, [1]);
  const graph = graphProto({
    nodes: [
      nodeProto("Optional", [], ["empty"], [typeAttributeProto("type", float1)]),
      nodeProto("OptionalGetElement", ["empty"], ["bad"]),
    ],
    outputs: [valueInfoProto("bad", float1)],
  });
  return modelProto(graph, 18, "deepbom_empty_optional_get_fixture");
}

function modelProto(graph, opset, producer) {
  const importProto = protoMessage([protoString(1, ""), protoVarintField(2, opset)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, producer), protoBytes(7, graph), protoBytes(8, importProto)]);
}

function graphProto({ nodes = [], initializers = [], inputs = [], outputs = [] }) {
  return protoMessage([
    ...nodes.map((value) => protoBytes(1, value)),
    protoString(2, "deepbom_container_graph"),
    ...initializers.map((value) => protoBytes(5, value)),
    ...inputs.map((value) => protoBytes(11, value)),
    ...outputs.map((value) => protoBytes(12, value)),
  ]);
}

function nodeProto(opType, inputs, outputs, attributes = []) {
  return protoMessage([
    ...inputs.map((value) => protoString(1, value)),
    ...outputs.map((value) => protoString(2, value)),
    protoString(3, `${opType}_fixture`), protoString(4, opType),
    ...attributes.map((value) => protoBytes(5, value)),
  ]);
}

function valueInfoProto(name, type) { return protoMessage([protoString(1, name), protoBytes(2, type)]); }
function tensorTypeProto(dtype, dims) {
  const shape = protoMessage(dims.map((dim) => protoBytes(1, dim >= 0 ? protoMessage([protoVarintField(1, dim)]) : protoMessage([]))));
  return protoMessage([protoBytes(1, protoMessage([protoVarintField(1, dtype), protoBytes(2, shape)]))]);
}
function sequenceTypeProto(elementType) { return protoMessage([protoBytes(4, protoMessage([protoBytes(1, elementType)]))]); }
function tensorProto(name, dtype, dims, raw) {
  return protoMessage([...dims.map((dim) => protoVarintField(1, dim)), protoVarintField(2, dtype), protoString(8, name), protoBytes(9, raw)]);
}
function intAttributeProto(name, value) { return protoMessage([protoString(1, name), protoVarintField(3, value), protoVarintField(20, 2)]); }
function typeAttributeProto(name, value) { return protoMessage([protoString(1, name), protoBytes(14, value), protoVarintField(20, 13)]); }
function graphAttributeProto(name, value) { return protoMessage([protoString(1, name), protoBytes(6, value), protoVarintField(20, 5)]); }
function int64Bytes(values) {
  const out = new Uint8Array(values.length * 8);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, value, true));
  return out;
}
function protoString(field, value) { return protoBytes(field, new TextEncoder().encode(value)); }
function protoBytes(field, value) {
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  return protoMessage([protoVarint((field << 3) | 2), protoVarint(bytes.length), bytes]);
}
function protoVarintField(field, value) { return protoMessage([protoVarint(field << 3), protoVarint(value)]); }
function protoVarint(value) {
  let remaining = BigInt(value);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return new Uint8Array(bytes);
}
function protoMessage(parts) {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
}
