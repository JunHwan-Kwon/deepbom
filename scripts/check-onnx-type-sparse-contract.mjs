import { analyzeOnnxModel } from "../web/onnx.js";
import { prepareOnnxExternalDataFiles } from "../web/lib/onnx-external-data.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("ONNX TypeProto and SparseTensorProto contract check");

const valid = analyzeOnnxModel(sparseModel({ indices: [1n, 6n] }), "valid_sparse.onnx");
const sparseInitializerTensor = valid.tensors.find((tensor) => tensor.name === "w");
expectEqual(valid.onnx_type_proto_contract.status, "assessed", "Recursive tensor/sequence/map/optional/sparse TypeProto declarations should pass the pinned contract.");
expectEqual(valid.onnx_type_proto_contract.sequence_value_count, 1, "Sequence TypeProto should remain a non-dense value kind.");
expectEqual(valid.onnx_type_proto_contract.map_value_count, 1, "Map TypeProto should remain a non-dense value kind.");
expectEqual(valid.onnx_type_proto_contract.optional_value_count, 1, "Optional TypeProto should remain a non-dense value kind.");
expectEqual(valid.onnx_type_proto_contract.sparse_tensor_value_count, 1, "Sparse-tensor TypeProto should remain distinct from a sparse-format initializer.");
expectEqual(valid.onnx_type_proto_contract.non_dense_value_count, 4, "Every declared non-dense ValueInfo should be conserved separately from dense tensors.");
expectEqual(valid.onnx_type_proto_contract.symbolic_dimension_count, 1, "Symbolic dimensions nested in a sequence element should be preserved.");
expectEqual(valid.onnx_shape_inference.non_dense_value_count, 4, "Tensor shape inference should exclude declared non-dense values from unknown dense-tensor counts.");
expectEqual(valid.onnx_sparse_tensor_contract.status, "assessed", "A complete ascending unique in-bounds sparse index payload should be fully assessed.");
expectEqual(valid.onnx_sparse_tensor_contract.assessed_index_count, 2, "Both sparse index rows should be decoded and checked.");
expectEqual(valid.onnx_sparse_tensor_contract.out_of_bounds_index_count, 0, "Valid sparse indices should have no bounds violation.");
expectEqual(valid.onnx_sparse_tensor_contract.duplicate_index_count, 0, "Valid sparse indices should have no duplicate.");
expectEqual(valid.onnx_sparse_tensor_contract.unsorted_index_count, 0, "Valid sparse indices should be ascending.");
expectEqual(sparseInitializerTensor?.value_kind, "tensor", "Graph sparse_initializer is a logical dense tensor initializer stored in sparse format.");
expectEqual(sparseInitializerTensor?.initializer_storage_kind, "sparse_tensor_proto", "Sparse storage kind should remain explicit.");
expectEqual(sparseInitializerTensor?.initializer_bytes, 24, "Sparse storage should count 8 values bytes plus 16 index bytes.");
expectEqual(sparseInitializerTensor?.initializer_elements, 8, "Sparse initializer logical cardinality should use the underlying dense shape.");
expectEqual(sparseInitializerTensor?.initializer_stored_elements, 4, "Stored-element cardinality should count values plus index elements.");
expectEqual(valid.size_breakdown.constant_tensor_count, 1, "One sparse initializer should count as one logical initializer declaration.");
expectEqual(valid.size_breakdown.dense_initializer_count, 0, "The fixture should contain no dense TensorProto initializer.");
expectEqual(valid.size_breakdown.sparse_initializer_count, 1, "The fixture should contain one SparseTensorProto initializer.");
expectEqual(valid.size_breakdown.constant_bytes, 24, "Embedded sparse component bytes should conserve exactly.");
expectEqual(valid.size_breakdown.stored_scalar_elements, 4, "Embedded sparse values and indices should conserve stored element counts.");
expectEqual(valid.size_breakdown.logical_initializer_elements, 8, "Dense-semantic sparse initializer elements should remain separate from stored elements.");
expectEqual(valid.size_breakdown.theoretical_fp16_constant_bytes, 20, "Sparse FLOAT value projection should change only NNZ values while preserving INT64 indices.");
expectEqual(valid.size_breakdown.theoretical_int8_constant_bytes, 18, "Sparse INT8 value projection should change only NNZ values while preserving INT64 indices.");
expectEqual(valid.ops[0].estimated_bytes, 96, "Operator traffic should use the sparse initializer's 32-byte logical dense tensor payload, not 24-byte storage encoding.");
expectEqual(valid.weight_integrity.initializer_tensors_present, 1, "Sparse initializer declarations must enter the weight-integrity inventory.");
expectEqual(valid.weight_integrity.constant_tensors_scanned, 1, "A structurally valid complete sparse initializer should be value-assessed.");
expectEqual(valid.weight_integrity.weight_tensors_scanned, 0, "An Add operand must not be promoted to a learned parameter without a learned-weight slot contract.");
expectEqual(valid.weight_integrity.constant_role_counts.unknown_or_mixed, 1, "An Add constant should retain its unresolved semantic role.");
expectEqual(valid.weight_integrity.logical_elements_assessed, 8, "Sparse integrity coverage should use the logical dense cardinality.");
expectEqual(valid.weight_integrity.stored_weight_values_decoded, 2, "Sparse integrity coverage should expose only the physically stored values as decoded.");
expectEqual(valid.weight_integrity.implicit_zero_elements, 6, "Every absent sparse logical element should be counted as an exact implicit zero.");
expectEqual(valid.weight_integrity.tensor_results.find((row) => row.tensor_name === "w")?.sparsity, 0.75, "Logical sparse-tensor sparsity should include specification-defined implicit zeros.");
expectEqual(valid.weight_integrity.mean_sparsity, null, "Learned-weight sparsity must remain unassessed when the only constant has an unresolved semantic role.");
expect(!buildFindingsRegister(valid).some((finding) => ["EA-ONX-0010", "EA-ONX-0011"].includes(finding.finding_id)), "Valid type and sparse contracts should not emit integrity findings.");
const validBundle = bundle(valid);
expectEqual(validBundle.evidence.evidence?.conformance_report?.status, "pass", "Valid type/sparse evidence should pass report, finding, and ML-BOM conformance.");
expect(validBundle.report.includes("## ONNX TypeProto Contract") && validBundle.report.includes("## ONNX SparseTensorProto Contract"), "Engineering Report should expose both contracts.");

const coordinate = analyzeOnnxModel(sparseModel({ indices: [0n, 1n, 1n, 2n], coordinateIndices: true }), "valid_coordinate_sparse.onnx");
expectEqual(coordinate.onnx_sparse_tensor_contract.rows[0].index_encoding, "coordinate_indices", "Rank-two [NNZ, rank] indices should use coordinate encoding.");
expectEqual(coordinate.onnx_sparse_tensor_contract.status, "assessed", "Lexicographically ascending coordinate indices should pass exact content validation.");

const unsorted = analyzeOnnxModel(sparseModel({ indices: [6n, 1n] }), "unsorted_sparse.onnx");
expectEqual(unsorted.onnx_sparse_tensor_contract.unsorted_index_count, 1, "One descending linear-index transition should be counted exactly.");
expect(unsorted.onnx_sparse_tensor_contract.invalid_rows[0].reason_codes.includes("sparse_indices_not_ascending"), "Descending sparse indices should retain their pinned reason code.");
const outOfBounds = analyzeOnnxModel(sparseModel({ indices: [1n, 8n] }), "out_of_bounds_sparse.onnx");
expectEqual(outOfBounds.onnx_sparse_tensor_contract.out_of_bounds_index_count, 1, "A linear index equal to dense cardinality should be rejected exactly once.");
expect(outOfBounds.onnx_sparse_tensor_contract.invalid_rows[0].reason_codes.includes("sparse_indices_out_of_bounds"), "Out-of-bounds sparse indices should retain their pinned reason code.");

const sparseKernel = analyzeOnnxModel(sparseModel({ indices: [1n, 6n], matmul: true }), "sparse_matmul_kernel.onnx");
expectEqual(sparseKernel.weight_integrity.output_channels_evaluated, 4, "Sparse MatMul kernels should evaluate every logical output-axis slice.");
expectEqual(sparseKernel.weight_integrity.zero_kernel_slice_count, 2, "Implicit sparse zeros should identify the two exact all-zero MatMul output columns.");
expectEqual(sparseKernel.weight_integrity.zero_kernel_slice_details[0].channels.join(","), "0,3", "Sparse output-slice witnesses should preserve exact logical channel indices.");

const asymmetricSparseKernel = analyzeOnnxModel(sparseQuantizedMatMulModel(), "sparse_quantized_matmul.onnx");
const asymmetricSparseResult = asymmetricSparseKernel.weight_integrity.tensor_results.find((row) => row.tensor_name === "wq");
expectEqual(asymmetricSparseResult.sparsity, 0.25, "Sparse integer code 0 must dequantize through scale*(0-zp), not be assumed to be real zero.");
expectEqual(asymmetricSparseResult.zero_kernel_slice_count, 0, "A non-zero asymmetric dequantized implicit code must prevent false all-zero output-slice findings.");
expectEqual(asymmetricSparseKernel.weight_integrity.quantized_constant_tensors_scanned, 1, "Validated sparse 8-bit kernels should enter exact logical quantization-grid analysis.");
expectEqual(asymmetricSparseKernel.weight_integrity.min_grid_utilization, 2 / 256, "Sparse grid utilization should include stored code 128 and implicit code 0 exactly once each.");
expectEqual(asymmetricSparseKernel.weight_integrity.max_saturation_percent, 0.75, "UINT8 implicit code 0 is qmin and must contribute exactly to endpoint saturation.");
const asymmetricSparseBundle = bundle(asymmetricSparseKernel);
expect(asymmetricSparseBundle.report.includes("### Quantized Kernel Grid Ledger") && asymmetricSparseBundle.report.includes("sparse_tensor_proto"), "Engineering Report should expose sparse logical quantization-grid arithmetic per kernel tensor.");
expectEqual(asymmetricSparseBundle.evidence.evidence?.conformance_report?.status, "pass", "Sparse quantization-grid evidence should pass report and finding conformance.");

const duplicateSparse = analyzeOnnxModel(duplicateSparseInitializerModel(), "duplicate_sparse.onnx");
expectEqual(duplicateSparse.size_breakdown.duplicate_initializer_analysis.status, "assessed", "Complete canonical sparse initializer duplicates should no longer force a partial result.");
expectEqual(duplicateSparse.size_breakdown.constant_bytes, 64, "Linear and coordinate sparse encodings should conserve their distinct stored byte counts.");
expectEqual(duplicateSparse.size_breakdown.unique_constant_bytes, 24, "Canonical sparse deduplication should retain the smallest equivalent storage representation.");
expectEqual(duplicateSparse.size_breakdown.duplicate_constant_bytes, 40, "Canonical sparse duplicate bytes should equal total equivalent storage minus the smallest representative.");

const nonDenseFlow = analyzeOnnxModel(nonDenseValueFlowModel(), "non_dense_value_flow.onnx");
expectEqual(nonDenseFlow.onnx_shape_inference.non_dense_node_output_count, 1, "Declared sparse-tensor node outputs should remain outside dense tensor shape coverage.");
expectEqual(nonDenseFlow.tensor_liveness.non_dense_value_count, 1, "Liveness should ledger a live non-dense value separately from unknown dense activations.");
expectEqual(nonDenseFlow.tensor_liveness.status, "partial", "A known dense subtotal plus a live non-dense value should be labeled as a partial lower bound.");
expect(nonDenseFlow.ops.every((op) => op.estimated_bytes === null && op.estimated_bytes_status === "not_assessed"), "Operators crossing a non-dense value must not receive dense payload-byte totals.");
expectEqual(nonDenseFlow.quantization_status.non_dense_value_count, 1, "Quantization coverage should disclose the non-dense denominator exclusion.");
expectEqual(nonDenseFlow.quantization_status.int8_tensors, 0, "An INT8 sparse-tensor TypeProto must not be counted as a dense INT8 tensor.");
const nonDenseBundle = bundle(nonDenseFlow);
expect(nonDenseBundle.report.includes("Non-dense runtime values excluded") && nonDenseBundle.report.includes("1 non-dense value(s) excluded"), "Engineering Report should disclose liveness and quantization denominator exclusions for non-dense values.");
expectEqual(nonDenseBundle.evidence.evidence?.conformance_report?.status, "pass", "Non-dense fail-closed evidence should pass export conformance.");

const invalid = analyzeOnnxModel(sparseModel({ indices: [1n, 1n], invalidMapKey: true }), "invalid_sparse.onnx");
expectEqual(invalid.onnx_type_proto_contract.status, "fail", "A BOOL map key should fail the pinned integral-or-string map-key contract.");
expect(invalid.onnx_type_proto_contract.invalid_rows.some((row) => row.reason_codes.includes("map_key_type_invalid_or_duplicate")), "Invalid TypeProto evidence should retain the exact map-key reason.");
expectEqual(invalid.onnx_sparse_tensor_contract.status, "fail", "Duplicate sparse indices should fail the pinned ascending-without-duplication contract.");
expectEqual(invalid.onnx_sparse_tensor_contract.duplicate_index_count, 1, "Exactly one adjacent duplicate index should be counted.");
const invalidFindings = buildFindingsRegister(invalid);
expect(invalidFindings.some((finding) => finding.finding_id === "EA-ONX-0010" && finding.technical_priority === "High"), "Invalid TypeProto should enter the High action queue.");
expect(invalidFindings.some((finding) => finding.finding_id === "EA-ONX-0011" && finding.technical_priority === "High"), "Invalid SparseTensorProto should enter the High action queue.");
const invalidBundle = bundle(invalid);
expectEqual(invalidBundle.evidence.evidence?.conformance_report?.status, "pass", "A correctly reported failing artifact contract should still pass evidence conformance.");
expect(invalidBundle.report.includes("Invalid TypeProto Declarations") && invalidBundle.report.includes("Invalid SparseTensorProto Records"), "Engineering Report should render both invalid ledgers.");

const indexSidecar = int64([1n, 6n]);
const externalModel = sparseModel({ indicesExternal: true });
const missingExternal = analyzeOnnxModel(externalModel, "mixed_sparse_external.onnx");
expectEqual(missingExternal.onnx_external_data.tensor_count, 1, "Sparse indices external_data should enter the all-scope TensorProto ledger.");
expectEqual(missingExternal.onnx_external_data.tensors[0].tensor_role, "graph_sparse_initializer_indices", "External evidence should preserve the sparse component role.");
expectEqual(missingExternal.onnx_sparse_tensor_contract.status, "partial", "Unavailable sparse index bytes should remain partial rather than pass or fail.");
expectEqual(missingExternal.size_breakdown.constant_bytes, 8, "Embedded sparse values bytes should remain counted when indices are external and unavailable.");
expectEqual(missingExternal.size_breakdown.stored_scalar_elements, 2, "Embedded sparse values cardinality should remain counted in mixed storage.");
expectEqual(missingExternal.size_breakdown.verified_external_payload_bytes, 0, "Unavailable sparse external indices must not be promoted to verified bytes.");
expectEqual(missingExternal.size_breakdown.available_initializer_bytes, 8, "Available payload should preserve only the embedded sparse values component.");
expectEqual(missingExternal.size_breakdown.logical_initializer_elements, 8, "Logical dense cardinality remains known even when sparse index payload is unavailable.");

const prepared = await prepareOnnxExternalDataFiles([{
  name: "indices.bin",
  size: indexSidecar.byteLength,
  arrayBuffer: async () => indexSidecar.buffer.slice(indexSidecar.byteOffset, indexSidecar.byteOffset + indexSidecar.byteLength),
}]);
const verifiedExternal = analyzeOnnxModel(externalModel, "mixed_sparse_external.onnx", null, { externalDataFiles: prepared });
expectEqual(verifiedExternal.onnx_external_data.status, "verified_payloads", "Sparse external indices should use the same path/range/cardinality/hash verification contract.");
expectEqual(verifiedExternal.onnx_sparse_tensor_contract.status, "assessed", "Verified sparse indices should close index-content assessment.");
expectEqual(verifiedExternal.onnx_sparse_tensor_contract.assessed_index_count, 2, "Verified external sparse index rows should be decoded exactly.");
expectEqual(verifiedExternal.size_breakdown.constant_bytes, 8, "External sparse indices must not be misclassified as ModelProto-embedded bytes.");
expectEqual(verifiedExternal.size_breakdown.verified_external_payload_bytes, 16, "Verified sparse index component bytes should be reported separately.");
expectEqual(verifiedExternal.size_breakdown.available_initializer_bytes, 24, "Available sparse storage should sum embedded values and verified external indices.");
expectEqual(verifiedExternal.size_breakdown.available_initializer_scalar_elements, 4, "Available sparse stored elements should sum NNZ values and index elements.");
expectEqual(verifiedExternal.weight_integrity.logical_elements_assessed, 8, "Verified external sparse indices should close logical initializer integrity assessment.");
expectEqual(bundle(verifiedExternal).evidence.evidence?.conformance_report?.status, "pass", "Mixed embedded/external sparse evidence should pass full export conformance.");

done("ONNX TypeProto and SparseTensorProto contracts passed (recursive kinds, dense/non-dense conservation, exact sparse indices, mixed external storage, findings, report, and ML-BOM).");

function bundle(analysis) {
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
    rawEvidenceContext: { identity: { filename: analysis.filename, format: "onnx" } },
    mlBomDocument: buildMlBomDocument(analysis, { hash: "" }),
  });
  return {
    report: files.find((file) => file.name === "engineering_report.md")?.data || "",
    evidence: JSON.parse(files.find((file) => file.name === "engineering_evidence.json")?.data || "{}"),
  };
}

function sparseModel({ indices = [1n, 6n], invalidMapKey = false, indicesExternal = false, coordinateIndices = false, matmul = false } = {}) {
  const values = tensor("w", 1, [2], float32([1.5, -2.25]));
  const indexTensor = indicesExternal
    ? externalTensor("", 7, [2], [["location", "indices.bin"], ["offset", "0"], ["length", "16"]])
    : tensor("", 7, coordinateIndices ? [2, 2] : [2], int64(indices));
  const sparse = sparseTensor(values, indexTensor, [2, 4]);
  const graph = graphProto({
    nodes: [node(matmul ? "MatMul" : "Add", matmul ? "matmul_sparse" : "add_sparse", ["x", "w"], ["y"])],
    sparseInitializers: [sparse],
    inputs: [valueInfo("x", tensorType(1, matmul ? [1, 2] : [2, 4]))],
    outputs: [valueInfo("y", tensorType(1, matmul ? [1, 4] : [2, 4]))],
    valueInfoRows: [
      valueInfo("seq", sequenceType(tensorType(1, [symbolicDimension("N"), 4]))),
      valueInfo("map", mapType(invalidMapKey ? 9 : 7, tensorType(1, [2]))),
      valueInfo("optional", optionalType(tensorType(6, []))),
      valueInfo("sparse_value", sparseValueType(1, [2, 4])),
    ],
  });
  const opset = message([stringField(1, ""), varintField(2, 13)]);
  return message([varintField(1, 8), stringField(2, "deepbom_type_sparse_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function sparseQuantizedMatMulModel() {
  const values = tensor("wq", 2, [1], new Uint8Array([128]));
  const indices = tensor("", 7, [1], int64([0n]));
  const sparse = sparseTensor(values, indices, [2, 2]);
  const graph = graphProto({
    nodes: [
      node("DequantizeLinear", "dq_sparse", ["wq", "scale", "zp"], ["w"]),
      node("MatMul", "matmul_sparse_q", ["x", "w"], ["y"]),
    ],
    initializers: [
      tensor("scale", 1, [], float32([0.5])),
      tensor("zp", 2, [], new Uint8Array([128])),
    ],
    sparseInitializers: [sparse],
    inputs: [valueInfo("x", tensorType(1, [1, 2]))],
    outputs: [valueInfo("y", tensorType(1, [1, 2]))],
    valueInfoRows: [valueInfo("w", tensorType(1, [2, 2]))],
  });
  const opset = message([stringField(1, ""), varintField(2, 13)]);
  return message([varintField(1, 8), stringField(2, "deepbom_sparse_quant_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function duplicateSparseInitializerModel() {
  const first = sparseTensor(
    tensor("w1", 1, [2], float32([1.5, -2.25])),
    tensor("", 7, [2], int64([1n, 6n])),
    [2, 4],
  );
  const second = sparseTensor(
    tensor("w2", 1, [2], float32([1.5, -2.25])),
    tensor("", 7, [2, 2], int64([0n, 1n, 1n, 2n])),
    [2, 4],
  );
  const graph = graphProto({
    nodes: [
      node("Add", "add_sparse_1", ["x", "w1"], ["z"]),
      node("Add", "add_sparse_2", ["z", "w2"], ["y"]),
    ],
    sparseInitializers: [first, second],
    inputs: [valueInfo("x", tensorType(1, [2, 4]))],
    outputs: [valueInfo("y", tensorType(1, [2, 4]))],
  });
  const opset = message([stringField(1, ""), varintField(2, 13)]);
  return message([varintField(1, 8), stringField(2, "deepbom_sparse_duplicate_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function nonDenseValueFlowModel() {
  const graph = graphProto({
    nodes: [
      nodeWithDomain("MakeSparse", "make_sparse", "com.acme", ["x"], ["s"]),
      nodeWithDomain("ConsumeSparse", "consume_sparse", "com.acme", ["s"], ["y"]),
    ],
    inputs: [valueInfo("x", tensorType(1, []))],
    outputs: [valueInfo("y", tensorType(1, []))],
    valueInfoRows: [valueInfo("s", sparseValueType(3, [4]))],
  });
  const standardOpset = message([stringField(1, ""), varintField(2, 13)]);
  const customOpset = message([stringField(1, "com.acme"), varintField(2, 1)]);
  return message([varintField(1, 8), stringField(2, "deepbom_non_dense_flow_fixture"), bytesField(7, graph), bytesField(8, standardOpset), bytesField(8, customOpset)]);
}

function graphProto({ nodes = [], initializers = [], sparseInitializers = [], inputs = [], outputs = [], valueInfoRows = [] }) {
  return message([
    ...nodes.map((value) => bytesField(1, value)),
    stringField(2, "deepbom_type_sparse_graph"),
    ...initializers.map((value) => bytesField(5, value)),
    ...inputs.map((value) => bytesField(11, value)),
    ...outputs.map((value) => bytesField(12, value)),
    ...valueInfoRows.map((value) => bytesField(13, value)),
    ...sparseInitializers.map((value) => bytesField(15, value)),
  ]);
}

function node(opType, name, inputs, outputs) {
  return message([
    ...inputs.map((value) => stringField(1, value)),
    ...outputs.map((value) => stringField(2, value)),
    stringField(3, name), stringField(4, opType),
  ]);
}

function nodeWithDomain(opType, name, domain, inputs, outputs) {
  return message([
    ...inputs.map((value) => stringField(1, value)),
    ...outputs.map((value) => stringField(2, value)),
    stringField(3, name), stringField(4, opType), stringField(7, domain),
  ]);
}

function valueInfo(name, type) {
  return message([stringField(1, name), bytesField(2, type)]);
}

function tensorType(dtype, dims) {
  const shape = message(dims.map((dim) => bytesField(1, typeof dim === "object" ? dim.encoded : message([varintField(1, dim)]))));
  return message([bytesField(1, message([varintField(1, dtype), bytesField(2, shape)]))]);
}

function sparseValueType(dtype, dims) {
  const shape = message(dims.map((dim) => bytesField(1, message([varintField(1, dim)]))));
  return message([bytesField(8, message([varintField(1, dtype), bytesField(2, shape)]))]);
}

function sequenceType(elementType) {
  return message([bytesField(4, message([bytesField(1, elementType)]))]);
}

function mapType(keyType, valueType) {
  return message([bytesField(5, message([varintField(1, keyType), bytesField(2, valueType)]))]);
}

function optionalType(elementType) {
  return message([bytesField(9, message([bytesField(1, elementType)]))]);
}

function symbolicDimension(name) {
  return { encoded: message([stringField(2, name)]) };
}

function sparseTensor(values, indices, dims) {
  return message([bytesField(1, values), bytesField(2, indices), ...dims.map((dim) => varintField(3, dim))]);
}

function tensor(name, dtype, dims, raw) {
  return message([...dims.map((dim) => varintField(1, dim)), varintField(2, dtype), ...(name ? [stringField(8, name)] : []), bytesField(9, raw)]);
}

function externalTensor(name, dtype, dims, entries) {
  return message([
    ...dims.map((dim) => varintField(1, dim)),
    varintField(2, dtype),
    ...(name ? [stringField(8, name)] : []),
    ...entries.map(([key, value]) => bytesField(13, message([stringField(1, key), stringField(2, value)]))),
    varintField(14, 1),
  ]);
}

function float32(values) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return out;
}

function int64(values) {
  const out = new Uint8Array(values.length * 8);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, BigInt(value), true));
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
