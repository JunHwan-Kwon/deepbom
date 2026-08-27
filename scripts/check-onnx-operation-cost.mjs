import { estimateOnnxMacs, summarizeOnnxAssessedMacs } from "../web/onnx.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("ONNX operation cost check");

function tensor(name, shape) {
  return { name, dtype: "FLOAT32", shape: [...shape], shapeDeclared: true, valueKind: "tensor" };
}

function node(opType, inputs, outputs, attributes = {}, domain = "") {
  return {
    opType,
    inputs,
    outputs,
    domain,
    attributes: new Map(Object.entries(attributes).map(([name, value]) => [name, {
      i: Number.isSafeInteger(value) ? value : null,
      ints: Array.isArray(value) ? value : [],
      s: typeof value === "string" ? value : "",
    }])),
  };
}

function assess(op, rows) {
  return estimateOnnxMacs(op, new Map(rows.map((row) => [row.name, row])));
}

const conv1d = assess(node("Conv", ["x", "w"], ["y"], { group: 2 }), [
  tensor("x", [2, 4, 10]), tensor("w", [6, 2, 3]), tensor("y", [2, 6, 8]),
]);
expectEqual(conv1d.status, "assessed", "Conv1D should be assessed from the ONNX rank-N contract.");
expectEqual(conv1d.value, 576, "Conv1D nominal MACs should equal product(Y)*product(W[1:]).");

const conv3d = assess(node("Conv", ["x", "w"], ["y"]), [
  tensor("x", [1, 2, 5, 6, 7]), tensor("w", [4, 2, 3, 2, 2]), tensor("y", [1, 4, 3, 5, 6]),
]);
expectEqual(conv3d.status, "assessed", "Conv3D should be assessed from the same rank-N contract.");
expectEqual(conv3d.value, 8_640, "Conv3D nominal MACs should preserve every spatial axis.");

const convTranspose = assess(node("ConvTranspose", ["x", "w"], ["y"]), [
  tensor("x", [1, 8, 16, 16]), tensor("w", [8, 4, 3, 3]), tensor("y", [1, 4, 18, 18]),
]);
expectEqual(convTranspose.status, "assessed", "A compatible static ConvTranspose should be assessed.");
expectEqual(convTranspose.value, 73_728, "Uncropped ConvTranspose should count every contributing input/kernel pair.");

const croppedConvTranspose = assess(node("ConvTranspose", ["x", "w"], ["y"], { pads: [1, 1] }), [
  tensor("x", [1, 2, 3]), tensor("w", [2, 4, 3]), tensor("y", [1, 4, 3]),
]);
expectEqual(croppedConvTranspose.value, 56, "ConvTranspose padding should exclude products cropped outside the serialized output.");

const groupedConvTranspose = assess(node("ConvTranspose", ["x", "w"], ["y"], { group: 2, strides: [2], dilations: [2], pads: [1, 2] }), [
  tensor("x", [1, 4, 3]), tensor("w", [4, 3, 3]), tensor("y", [1, 6, 6]),
]);
expectEqual(groupedConvTranspose.value, 84, "Grouped ConvTranspose should retain stride, dilation, crop, and M/group semantics.");

const shapedConvTranspose = assess(node("ConvTranspose", ["x", "w"], ["y"], { strides: [2], output_shape: [4] }), [
  tensor("x", [1, 1, 3]), tensor("w", [1, 1, 3]), tensor("y", [1, 1, 4]),
]);
expectEqual(shapedConvTranspose.value, 6, "ConvTranspose output_shape should generate the source-defined asymmetric pads before MAC counting.");

const invalidConvTranspose = assess(node("ConvTranspose", ["x", "w"], ["y"], { strides: [2], pads: [0, 0] }), [
  tensor("x", [1, 1, 3]), tensor("w", [1, 1, 3]), tensor("y", [1, 1, 99]),
]);
expectEqual(invalidConvTranspose.status, "not_assessed", "An inconsistent ConvTranspose output contract must fail closed.");

const lstm = assess(node("LSTM", ["x", "w", "r"], ["y"], { hidden_size: 4 }), [
  tensor("x", [5, 2, 3]), tensor("w", [1, 16, 3]), tensor("r", [1, 16, 4]), tensor("y", [5, 1, 2, 4]),
]);
expectEqual(lstm.value, 1_120, "LSTM should count four input and recurrent gate contractions per step.");

const bidirectionalGru = assess(node("GRU", ["x", "w", "r"], ["y"], { hidden_size: 4, direction: "bidirectional", layout: 1 }), [
  tensor("x", [2, 5, 3]), tensor("w", [2, 12, 3]), tensor("r", [2, 12, 4]), tensor("y", [2, 5, 2, 4]),
]);
expectEqual(bidirectionalGru.value, 1_680, "Bidirectional layout-1 GRU should count three gate contractions in both directions.");

const invalidRecurrent = assess(node("RNN", ["x", "w", "r"], ["y"], { hidden_size: 4 }), [
  tensor("x", [5, 2, 3]), tensor("w", [1, 4, 3]), tensor("r", [1, 4, 5]), tensor("y", [5, 1, 2, 4]),
]);
expectEqual(invalidRecurrent.status, "not_assessed", "A recurrent-state width mismatch must fail closed.");

const attention = assess(node("Attention", ["q", "k", "v", "", "pk", "pv"], ["y", "present_k", "present_v", "qk"]), [
  tensor("q", [2, 8, 16, 64]), tensor("k", [2, 4, 20, 64]), tensor("v", [2, 4, 20, 80]),
  tensor("pk", [2, 4, 5, 64]), tensor("pv", [2, 4, 5, 80]), tensor("y", [2, 8, 16, 80]),
]);
expectEqual(attention.status, "assessed", "Static Attention contractions should be assessed from the source-defined Q/K/V/cache contract.");
expectEqual(attention.value, 921_600, "Attention should sum the QK and attention-value contraction terms over total KV sequence length.");

const deformConv = assess(node("DeformConv", ["x", "w", "offset", "bias", "mask"], ["y"], { group: 2, offset_group: 2 }), [
  tensor("x", [2, 8, 16, 16]), tensor("w", [12, 4, 3, 3]), tensor("offset", [2, 36, 14, 14]),
  tensor("bias", [12]), tensor("mask", [2, 18, 14, 14]), tensor("y", [2, 12, 14, 14]),
]);
expectEqual(deformConv.status, "assessed", "Static DeformConv sampled-value contractions should be assessed after auxiliary-shape validation.");
expectEqual(deformConv.value, 169_344, "DeformConv nominal MACs should count each output/channel-kernel contraction once.");

const einsum = assess(node("Einsum", ["a", "b"], ["y"], { equation: "bij,bjk->bik" }), [
  tensor("a", [2, 3, 4]), tensor("b", [2, 4, 5]), tensor("y", [2, 3, 5]),
]);
expectEqual(einsum.status, "assessed", "A static two-input Einsum should have an order-independent nominal contraction count.");
expectEqual(einsum.value, 120, "Einsum should count one contraction term for each complete Einstein index assignment.");

const scalarEinsum = assess(node("Einsum", ["scalar", "vector"], ["scaled"], { equation: ",i->i" }), [
  tensor("scalar", []), tensor("vector", [7]), tensor("scaled", [7]),
]);
expectEqual(scalarEinsum.value, 7, "Einsum must accept an empty scalar subscript and count scalar-vector products.");

const orderDependentEinsum = assess(node("Einsum", ["a", "b", "c"], ["y"], { equation: "ab,bc,cd->ad" }), [
  tensor("a", [2, 3]), tensor("b", [3, 4]), tensor("c", [4, 5]), tensor("y", [2, 5]),
]);
expectEqual(orderDependentEinsum.status, "not_assessed", "A three-input Einsum must remain unassessed without a serialized contraction order.");

for (const opType of ["DFT", "STFT"]) {
  const result = assess(node(opType, ["x", "w"], ["y"]), [tensor("x", [1]), tensor("w", [1]), tensor("y", [1])]);
  expectEqual(result.status, "not_assessed", `${opType} must not be silently classified as zero MAC.`);
  expectEqual(result.value, null, `${opType} must retain a null MAC value until its rule is implemented.`);
}
for (const opType of ["DFT", "STFT"]) {
  const result = assess(node(opType, ["x", "w"], ["y"]), [tensor("x", [1]), tensor("w", [1]), tensor("y", [1])]);
  expect(result.reason.includes("algorithm-dependent"), `${opType} must explain why an implementation-independent MAC count does not exist.`);
}

expectEqual(assess(node("Relu", ["x"], ["y"]), [tensor("x", [1]), tensor("y", [1])]).status, "not_applicable", "A source-classified non-MAC op should remain not applicable.");
expectEqual(assess(node("FutureTensorProduct", ["x", "w"], ["y"]), [tensor("x", [1]), tensor("w", [1]), tensor("y", [1])]).status, "not_assessed", "An operator absent from the pinned source registry must fail closed.");

const zeroConv = assess(node("Conv", ["x", "w"], ["y"]), [
  tensor("x", [0, 2, 5]), tensor("w", [4, 2, 3]), tensor("y", [0, 4, 3]),
]);
expectEqual(zeroConv.status, "assessed", "A legal zero-cardinality output is known, not dynamic.");
expectEqual(zeroConv.value, 0, "A zero-cardinality Conv output should have zero nominal MACs.");

const invalidGroup = assess(node("Conv", ["x", "w"], ["y"], { group: 0 }), [
  tensor("x", [1, 2, 5]), tensor("w", [4, 2, 3]), tensor("y", [1, 4, 3]),
]);
expectEqual(invalidGroup.status, "not_assessed", "An invalid group must not be coerced to one.");

const vectorDot = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [3]), tensor("b", [3]), tensor("y", []),
]);
expectEqual(vectorDot.value, 3, "Vector-vector MatMul should retain the rank-promotion dot-product cost.");

const vectorMatrix = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [3]), tensor("b", [3, 4]), tensor("y", [4]),
]);
expectEqual(vectorMatrix.value, 12, "Vector-matrix MatMul should remove only the promoted A axis.");

const matrixVector = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [2, 3]), tensor("b", [3]), tensor("y", [2]),
]);
expectEqual(matrixVector.value, 6, "Matrix-vector MatMul should remove only the promoted B axis.");

const emptyBatch = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [0, 2, 3]), tensor("b", [1, 3, 4]), tensor("y", [0, 2, 4]),
]);
expectEqual(emptyBatch.status, "assessed", "A zero batch dimension broadcast against one is valid.");
expectEqual(emptyBatch.value, 0, "A zero batch dimension should produce zero nominal MACs.");

const wrongOutput = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [2, 3]), tensor("b", [3, 4]), tensor("y", [2, 5]),
]);
expectEqual(wrongOutput.status, "not_assessed", "MatMul must validate the serialized output shape before emitting MACs.");

const overflow = assess(node("MatMul", ["a", "b"], ["y"]), [
  tensor("a", [Number.MAX_SAFE_INTEGER, 2]), tensor("b", [2, 2]), tensor("y", [Number.MAX_SAFE_INTEGER, 2]),
]);
expectEqual(overflow.status, "assessed", "A static MAC product above the safe Number range should remain exactly assessed.");
expectEqual(overflow.value, null, "An unsafe per-op Number mirror must be withheld.");
expectEqual(overflow.value_decimal, "36028797018963964", "An unsafe per-op MAC product must remain available as an exact decimal.");

const emptyGemm = assess(node("Gemm", ["a", "b"], ["y"]), [
  tensor("a", [0, 3]), tensor("b", [3, 4]), tensor("y", [0, 4]),
]);
expectEqual(emptyGemm.status, "assessed", "A zero-row Gemm is a known zero-cardinality operation.");
expectEqual(emptyGemm.value, 0, "A zero-row Gemm should have zero nominal MACs.");

const invalidTranspose = assess(node("Gemm", ["a", "b"], ["y"], { transA: 2 }), [
  tensor("a", [2, 3]), tensor("b", [3, 4]), tensor("y", [2, 4]),
]);
expectEqual(invalidTranspose.status, "not_assessed", "Invalid Gemm transpose values must fail closed.");

const customCollision = assess(node("Conv", ["x", "w"], ["y"], {}, "com.acme"), [
  tensor("x", [1, 2, 5]), tensor("w", [4, 2, 3]), tensor("y", [1, 4, 3]),
]);
expectEqual(customCollision.status, "not_assessed", "A custom-domain name collision must not inherit ai.onnx MAC semantics.");

const unsafeAggregate = summarizeOnnxAssessedMacs([Number.MAX_SAFE_INTEGER, 1]);
expectEqual(unsafeAggregate.total_assessed_macs_decimal, "9007199254740992", "Aggregate MACs should retain an exact decimal above the safe Number range.");
expectEqual(unsafeAggregate.total_assessed_macs, null, "An unsafe aggregate MAC Number mirror must be withheld.");
expectEqual(unsafeAggregate.total_assessed_ops_decimal, "18014398509481984", "Aggregate arithmetic operations should derive exactly from the BigInt MAC total.");
expectEqual(unsafeAggregate.total_assessed_ops, null, "An unsafe aggregate operation Number mirror must be withheld.");
expectEqual(unsafeAggregate.safe_number_mirror_status, "exact_decimal_only", "Unsafe aggregate status should disclose decimal-only representation.");

done("source-classified MAC coverage, exact rank-N ConvTranspose, rank-1 MatMul promotion, zero-cardinality tensors, and exact aggregate ledgers");
