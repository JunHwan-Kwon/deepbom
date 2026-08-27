import { readFileSync } from "node:fs";
import { Builder } from "flatbuffers";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";

const expect = (condition, message) => {
  if (!condition) throw new Error(message);
};
const expectEqual = (actual, expected, message) => {
  if (actual !== expected) throw new Error(`${message}: expected ${expected}, got ${actual}`);
};

function int32Vector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index -= 1) builder.addInt32(values[index]);
  return builder.endVector();
}

function uint8Vector(builder, values) {
  builder.startVector(1, values.length, 1);
  for (let index = values.length - 1; index >= 0; index -= 1) builder.addInt8(values[index]);
  return builder.endVector();
}

function offsetVector(builder, values) {
  builder.startVector(4, values.length, 4);
  for (let index = values.length - 1; index >= 0; index -= 1) builder.addOffset(values[index]);
  return builder.endVector();
}

function indexVector(builder, values) {
  const vector = int32Vector(builder, values);
  builder.startObject(1);
  builder.addFieldOffset(0, vector, 0);
  return builder.endObject();
}

function denseDimension(builder, size) {
  builder.startObject(6);
  builder.addFieldInt32(1, size, 0);
  return builder.endObject();
}

function csrDimension(builder, segments, indices) {
  const segmentTable = indexVector(builder, segments);
  const indexTable = indexVector(builder, indices);
  builder.startObject(6);
  builder.addFieldInt8(0, 1, 0); // DimensionType.SPARSE_CSR
  builder.addFieldInt8(2, 1, 0); // SparseIndexVector.Int32Vector
  builder.addFieldOffset(3, segmentTable, 0);
  builder.addFieldInt8(4, 1, 0);
  builder.addFieldOffset(5, indexTable, 0);
  return builder.endObject();
}

function makeSparseFixture({ indices = [0, 2, 1], values = [7, 8, 9] } = {}) {
  const builder = new Builder(1024);
  const name = builder.createString("sparse_constant");
  const shape = int32Vector(builder, [2, 3]);
  const traversal = int32Vector(builder, [0, 1]);
  const dimensions = offsetVector(builder, [denseDimension(builder, 2), csrDimension(builder, [0, 2, 3], indices)]);

  builder.startObject(3);
  builder.addFieldOffset(0, traversal, 0);
  builder.addFieldOffset(2, dimensions, 0);
  const sparsity = builder.endObject();

  builder.startObject(10);
  builder.addFieldOffset(0, shape, 0);
  builder.addFieldInt8(1, 9, 0); // TensorType.INT8
  builder.addFieldInt32(2, 1, 0);
  builder.addFieldOffset(3, name, 0);
  builder.addFieldOffset(6, sparsity, 0);
  const tensor = builder.endObject();
  const tensors = offsetVector(builder, [tensor]);
  const outputs = int32Vector(builder, [0]);

  builder.startObject(5);
  builder.addFieldOffset(0, tensors, 0);
  builder.addFieldOffset(2, outputs, 0);
  const subgraph = builder.endObject();
  const subgraphs = offsetVector(builder, [subgraph]);

  builder.startObject(3);
  const emptyBuffer = builder.endObject();
  const data = uint8Vector(builder, values);
  builder.startObject(3);
  builder.addFieldOffset(0, data, 0);
  const valueBuffer = builder.endObject();
  const buffers = offsetVector(builder, [emptyBuffer, valueBuffer]);

  builder.startObject(8);
  builder.addFieldInt32(0, 3, 0);
  builder.addFieldOffset(2, subgraphs, 0);
  builder.addFieldOffset(4, buffers, 0);
  const model = builder.endObject();
  builder.finish(model, "TFL3");
  return builder.asUint8Array();
}

function expectRejected(bytes, label) {
  try {
    analyze_tflite_for_target(bytes, `${label}.tflite`, "android_mid_a55");
  } catch (error) {
    expect(String(error).includes("sparse") || String(error).includes("Sparse"), `${label} should fail in the sparse contract: ${error}`);
    return;
  }
  throw new Error(`${label} was accepted`);
}

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });

const result = analyze_tflite_for_target(makeSparseFixture(), "sparse-valid.tflite", "android_mid_a55");
result.model_sha256 = "a".repeat(64);
const contract = result.tflite_sparse_storage_contract;
expectEqual(contract.schema, "deepbom.tflite_sparse_storage_contract.v1", "Sparse contract schema");
expectEqual(contract.status, "assessed", "Sparse contract status");
expectEqual(contract.sparse_tensor_count, 1, "Sparse tensor count");
expectEqual(contract.fully_decoded_tensor_count, 1, "Decoded sparse tensor count");
expectEqual(contract.logical_element_count, 6, "Logical sparse elements");
expectEqual(contract.stored_element_count, 3, "Stored sparse elements");
expectEqual(contract.implicit_zero_element_count, 3, "Implicit sparse zero elements");
expectEqual(contract.serialized_value_bytes, 3, "Sparse physical value bytes");
expectEqual(contract.logical_element_count, contract.stored_element_count + contract.implicit_zero_element_count, "Sparse element conservation");
expectEqual(contract.rows[0].encoding.canonical_metadata_sha256.length, 64, "Canonical metadata digest length");
expectEqual(result.weight_integrity.sparse_constant_tensors_decoded, 1, "Weight-integrity sparse decode count");
expectEqual(result.weight_integrity.sparse_logical_elements, 6, "Weight-integrity logical element count");
expectEqual(result.weight_integrity.sparse_stored_elements, 3, "Weight-integrity stored element count");
expectEqual(result.weight_integrity.sparse_implicit_zero_elements, 3, "Weight-integrity implicit zero count");
expectEqual(result.size_breakdown.constant_bytes, 3, "Size ledger must retain physical sparse bytes");

const mlBomDocument = buildMlBomDocument(result, { hash: result.model_sha256, targetId: result.target_profile.id });
const bundle = buildEngineeringBundleArtifactFiles(result, {
  reportContext: { identity: { filename: result.filename, format: "tflite", sha256: result.model_sha256 } },
  rawEvidenceContext: { identity: { filename: result.filename, format: "tflite", sha256: result.model_sha256 } },
  mlBomDocument,
});
const report = bundle.find((file) => file.name === "engineering_report.md")?.data || "";
const evidence = JSON.parse(bundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(report.includes("## TFLite Sparse Storage Contract (DERIVED)"), "Engineering Report sparse section");
expect(report.includes(contract.schema_source_sha256), "Engineering Report sparse schema source digest");
expect(report.includes(contract.converter_source_sha256), "Engineering Report sparse converter source digest");
expect(report.includes("6 logical = 3 stored + 3 implicit raw-zero"), "Engineering Report sparse aggregate conservation");
expectEqual(evidence.evidence?.static_analysis?.tflite_sparse_storage_contract?.logical_element_count, 6, "Raw evidence sparse logical elements");
expectEqual(evidence.evidence?.conformance_report?.status, "pass", "Sparse bundle conformance");
const componentProperties = mlBomDocument.metadata?.component?.properties || [];
const componentProperty = (name) => componentProperties.find((item) => item.name === name)?.value;
expectEqual(componentProperty("deepbom:model:tfliteSparseTensorCount"), "1", "CycloneDX sparse tensor count");
expectEqual(componentProperty("deepbom:model:tfliteSparseLogicalElements"), "6", "CycloneDX sparse logical elements");
expectEqual(componentProperty("deepbom:model:tfliteSparseStoredElements"), "3", "CycloneDX sparse stored elements");
expectEqual(componentProperty("deepbom:model:tfliteSparseImplicitZeroElements"), "3", "CycloneDX sparse implicit zeros");
expectEqual(componentProperty("deepbom:model:tfliteSparseSerializedValueBytes"), "3", "CycloneDX sparse bytes");

expectRejected(makeSparseFixture({ indices: [0, 0, 1] }), "sparse-duplicate-coordinate");
expectRejected(makeSparseFixture({ indices: [0, 3, 1] }), "sparse-out-of-range-coordinate");
expectRejected(makeSparseFixture({ values: [7, 8] }), "sparse-value-byte-mismatch");

console.log("TFLite sparse storage checks passed (1 valid fixture, 3 fail-closed fixtures).");
