import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { analyzeExecuTorchModel, assessExecuTorchPortableKernelMac, compareExecuTorchExternalTensorContract, deriveExecuTorchTensorShapeContract, EXECUTORCH_SCHEMA_SOURCE } from "../web/executorch.js";
import { EXECUTORCH_OPERATOR_SIGNATURE_SOURCE, EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES } from "../web/lib/executorch-operator-signatures.generated.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import {
  decodeFixtureBase64,
  EXECUTORCH_ADD_PTE_BASE64,
  EXECUTORCH_EMPTY_PTD_BASE64,
  EXECUTORCH_LEGACY_SEGMENT_PTD_BASE64,
} from "./fixtures/executorch-fixtures.mjs";

const addBytes = decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64);
assert.equal(sha256(addBytes), "e6dcfc0a685450c409a2ea924730742b86987c8432a754744604715601c3bf23");
const add = analyzeExecuTorchModel(addBytes, "add.pte");
assert.equal(add.schema, "deepbom.static_analysis.executorch.v1.1");
assert.equal(add.executorch_container, "pte");
assert.equal(add.subgraphs, 1);
assert.equal(add.operator_count, 1);
assert.equal(add.tensor_count, 3);
assert.deepEqual(add.inputs.map((tensor) => tensor.shape), [[1], [1]]);
assert.deepEqual(add.outputs.map((tensor) => tensor.shape), [[1]]);
assert.equal(add.ops[0].name, "aten::add.out");
assert.deepEqual(add.ops[0].evalue_args, [0, 1, 3, 2, 2]);
assert.deepEqual(add.ops[0].inputs, [0, 1]);
assert.deepEqual(add.ops[0].outputs, [2]);
assert.equal(add.ops[0].signature_status, "source_bound");
assert.equal(add.ops[0].signature_source_schema, "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
assert.equal(add.ops[0].macs, 0);
assert.equal(add.ops[0].macs_status, "assessed_zero_nominal_tensor_contraction_macs");
assert.equal(add.total_macs, 0);
assert.equal(add.total_macs_decimal, "0");
assert.equal(add.mac_assessment.complete, true);
assert.equal(add.mac_assessment.unknown_compute_instruction_count, 0);
assert.equal(add.mac_assessment.source_bound_kernel_instruction_count, 1);
assert.equal(add.mac_assessment.non_compute_instruction_count, 0);
assert.equal(add.stages[0].mac_assessed_ops, 1);
assert.equal(add.stages[0].mac_not_assessed_ops, 0);
assert.equal(add.tensor_liveness.planned_non_const_memory_decimal, "48");
assert.equal(add.tensor_liveness.peak_planned_live_allocation_decimal, "12");
assert.equal(add.tensor_liveness.peak_planned_live_allocation_status, "derived_exact_aot_static_address_liveness");
assert.equal(add.tensor_liveness.peak_live_payload_bytes, null, "AOT address occupancy must not be mislabeled as observed runtime payload liveness.");
assert.equal(add.tensors.every((tensor) => tensor.dtype === "FLOAT32" && tensor.buffer_data_length_decimal === "4"), true);
assert.equal(JSON.stringify(analyzeExecuTorchModel(addBytes, "add.pte")), JSON.stringify(add), "PTE analysis must be deterministic.");
const addReport = buildEngineeringReport(add, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(addReport.includes("1 kernel + 0 delegate + 0 move/control/free")
  && addReport.includes("0/0 exact name/dtype/shape/logical-byte/layout contracts")
  && addReport.includes("Matching portable KernelCall direction is source-bound")
  && addReport.includes("1/1 KernelCall signatures source-bound"));

const emptyPtd = analyzeExecuTorchModel(decodeFixtureBase64(EXECUTORCH_EMPTY_PTD_BASE64), "weights.ptd");
assert.equal(emptyPtd.executorch_container, "ptd");
assert.equal(emptyPtd.version, 1);
assert.equal(emptyPtd.operator_count, 0);
assert.equal(emptyPtd.tensor_count, 1);
assert.equal(emptyPtd.tensors[0].name, "weights.bin");
assert.equal(emptyPtd.tensors[0].layout_status, "not_applicable_blob");

const legacyPtdBytes = decodeFixtureBase64(EXECUTORCH_LEGACY_SEGMENT_PTD_BASE64);
const legacyPtd = analyzeExecuTorchModel(legacyPtdBytes, "legacy.ptd");
assert.equal(legacyPtd.executorch_flat_tensor.extended_header.present, true);
assert.equal(legacyPtd.executorch_flat_tensor.segments[0].size_wire_bits, 32);
assert.equal(legacyPtd.executorch_flat_tensor.segments[0].size, "4");
assert.equal(legacyPtd.executorch_flat_tensor.segments[0].absolute_end, String(legacyPtdBytes.length));
assert.equal(legacyPtd.size_breakdown.appended_segment_bytes_decimal, "4");

const externalExpected = {
  shape_status: "assessed",
  shape: [2, 2],
  dtype: "FLOAT32",
  buffer_data_length_decimal: "16",
};
const externalSupplied = {
  shape_declared: true,
  shape: [2, 2],
  dtype: "FLOAT32",
  buffer_data_length_decimal: "16",
  serialized_storage_span_decimal: "24",
  layout_status: "assessed",
};
assert.deepEqual(compareExecuTorchExternalTensorContract(externalExpected, externalSupplied), {
  status: "matched",
  reasons: [],
  dtype: "FLOAT32",
  shape: [2, 2],
  logical_bytes_decimal: "16",
  serialized_span_bytes_decimal: "24",
});
const externalMismatch = compareExecuTorchExternalTensorContract(externalExpected, {
  ...externalSupplied,
  shape: [4],
  dtype: "FLOAT16",
  buffer_data_length_decimal: "8",
  layout_status: "not_applicable_blob",
});
assert.deepEqual(externalMismatch.reasons, [
  "shape_mismatch",
  "dtype_mismatch",
  "logical_byte_length_mismatch",
  "supplied_ptd_tensor_layout_not_assessed",
]);

assert.equal(EXECUTORCH_SCHEMA_SOURCE.commit, "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f");
for (const key of ["program_schema_sha256", "scalar_type_schema_sha256", "flat_tensor_schema_sha256", "extended_header_sha256"]) {
  assert.match(EXECUTORCH_SCHEMA_SOURCE[key], /^[0-9a-f]{64}$/);
}
assert.equal(EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.portable_operator_count, 209);
assert.equal(EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.exact_pytorch_out_schema_count, 139);
assert.equal(EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.functional_out_derivation_count, 61);
assert.equal(EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.executorch_custom_schema_count, 9);
assert.equal(EXECUTORCH_OPERATOR_SIGNATURE_SOURCE.pytorch.release, "v2.13.0");
assert.equal(Object.keys(EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES).length, 209);
assert.deepEqual(EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES["aten::add.out"].tensor_input_argument_positions, [0, 1]);
assert.deepEqual(EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES["aten::add.out"].tensor_output_argument_positions, [3]);

const boundedShape = deriveExecuTorchTensorShapeContract([8, 16], "DYNAMIC_BOUND", 4);
assert.equal(boundedShape.shape_status, "assessed_upper_bound");
assert.deepEqual(boundedShape.shape_signature, [-1, -1]);
assert.deepEqual(boundedShape.shape_upper_bound, [8, 16]);
assert.equal(boundedShape.logical_elements, 128n);
assert.equal(boundedShape.logical_bytes, 64n);
assert.equal(boundedShape.logical_bytes_status, "exact_serialized_upper_bound");
const unboundedShape = deriveExecuTorchTensorShapeContract([1, 1], "DYNAMIC_UNBOUND", 32);
assert.equal(unboundedShape.logical_bytes, null);
assert.equal(unboundedShape.logical_bytes_status, "not_assessed_dynamic_unbound");
assert.throws(() => deriveExecuTorchTensorShapeContract([-1, 4], "DYNAMIC_BOUND", 8), /negative or non-integer/);

const tensorValues = (count) => Array.from({ length: count }, (_, index) => ({ kind: "Tensor", tensor_index: index }));
const tensor = (shape) => ({ shape_status: "assessed", shape });
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::mm.out", [0, 1, 2], tensorValues(3), [tensor([2, 3]), tensor([3, 4]), tensor([2, 4])]),
  { macs: 24, decimal: "24", status: "assessed_source_bound_mm" },
);
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::bmm.out", [0, 1, 2], tensorValues(3), [tensor([5, 2, 3]), tensor([5, 3, 4]), tensor([5, 2, 4])]),
  { macs: 120, decimal: "120", status: "assessed_source_bound_bmm" },
);
const convolutionValues = tensorValues(10);
convolutionValues[6] = { kind: "Bool", value: false };
convolutionValues[8] = { kind: "Int", value: 2, value_decimal: "2" };
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::convolution.out", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], convolutionValues, [
    tensor([1, 4, 16, 16]), tensor([8, 2, 3, 3]), tensor([8]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([1, 8, 14, 14]),
  ]),
  { macs: 28224, decimal: "28224", status: "assessed_source_bound_convolution" },
);
const transposedValues = convolutionValues.map((value) => ({ ...value }));
transposedValues[6] = { kind: "Bool", value: true };
transposedValues[3] = { kind: "IntList", values: [2], values_decimal: ["2"] };
transposedValues[4] = { kind: "IntList", values: [1], values_decimal: ["1"] };
transposedValues[5] = { kind: "IntList", values: [1], values_decimal: ["1"] };
transposedValues[7] = { kind: "IntList", values: [0], values_decimal: ["0"] };
transposedValues[8] = { kind: "Int", value: 1, value_decimal: "1" };
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::convolution.out", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], transposedValues, [
    tensor([1, 2, 3]), tensor([2, 3, 3]), tensor([3]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([]), tensor([1, 3, 5]),
  ]),
  { macs: 42, decimal: "42", status: "assessed_source_bound_transposed_convolution_overlap" },
);
const dynamicBoundTensors = [tensor([2, 3]), tensor([3, 4]), tensor([2, 4])];
dynamicBoundTensors[0].shape_status = "assessed_upper_bound";
assert.match(
  assessExecuTorchPortableKernelMac("aten::mm.out", [0, 1, 2], tensorValues(3), dynamicBoundTensors).status,
  /^not_assessed_shape_contract_conflict:/,
  "DYNAMIC_BOUND extents must remain upper bounds and must not be promoted to exact nominal MACs.",
);

for (const length of [0, 1, 7, 8, 16, 63, 127, 255, 511, addBytes.length - 1]) {
  assert.throws(() => analyzeExecuTorchModel(addBytes.subarray(0, length), "truncated.pte"), /ExecuTorch|FlatBuffer|ET12|FT01|range|payload|identifier/i);
}
const badIdentifier = addBytes.slice();
badIdentifier[4] = 0x58;
assert.throws(() => analyzeExecuTorchModel(badIdentifier, "bad.pte"), /identifier/);
const badRoot = addBytes.slice();
badRoot.fill(0xff, 0, 4);
assert.throws(() => analyzeExecuTorchModel(badRoot, "bad-root.pte"), /root|range|payload/);
const truncatedSegment = legacyPtdBytes.subarray(0, legacyPtdBytes.length - 1);
assert.throws(() => analyzeExecuTorchModel(truncatedSegment, "truncated.ptd"), /segment_data_size|beyond file|file boundary/);

console.log("ExecuTorch ET12/FT01 analysis checks passed.");

function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
