import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { analyzeExecuTorchModel, assessExecuTorchPortableKernelMac, compareExecuTorchExternalTensorContract, deriveExecuTorchTensorShapeContract, EXECUTORCH_SCHEMA_SOURCE } from "../web/executorch.js";
import { EXECUTORCH_OPERATOR_SIGNATURE_SOURCE, EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES } from "../web/lib/executorch-operator-signatures.generated.js";
import {
  assessExecuTorchProcessedPayload,
  EXECUTORCH_BACKEND_REGISTRY_SOURCE,
  EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA,
  EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA,
  resolveExecuTorchSelectedBuildAttestation,
  validateExecuTorchSelectedBuildAttestation,
} from "../web/lib/executorch-build-binding.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
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
assert.equal(add.runtime_compat.min_runtime_version, null);
assert.equal(add.runtime_compat.min_runtime_version_status, "NOT_DERIVABLE_SCHEMA_VERSION_NOT_RELEASE_MONOTONIC");
assert.equal(add.runtime_compat.schema_version_status, "MATCHES_PINNED_EXPORTER_SCHEMA_VERSION");
assert.equal(add.executorch_program.selected_build_binding.status, "SOURCE_ONLY_SELECTED_BUILD_UNBOUND");
assert.equal(add.executorch_program.selected_build_binding.kernel_bindings[0].selected_build_status, "SELECTED_BUILD_NOT_BOUND");
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
assert(addReport.includes(add.executorch_program.graph_boundary)
  && addReport.includes(add.executorch_program.selected_build_binding.interpretation_boundary),
"ExecuTorch reports must preserve the artifact graph and selected-build interpretation boundaries verbatim.");

const selectedBuild = buildSelectedBuildAttestation({ portableOperatorNames: ["aten::add.out"] });
assert.deepEqual(validateExecuTorchSelectedBuildAttestation(selectedBuild), selectedBuild);
const selectedBuildBytes = new TextEncoder().encode(JSON.stringify(selectedBuild));
const resolvedSelectedBuild = resolveExecuTorchSelectedBuildAttestation([{
  path: "evidence/deepbom.executorch-build.json",
  bytes: selectedBuildBytes,
  sha256: sha256(selectedBuildBytes),
}]);
assert.equal(resolvedSelectedBuild.input.schema, EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA);
assert.equal(resolvedSelectedBuild.input.duplicate_key_validation, "complete");
assert.equal(resolvedSelectedBuild.input.file_sha256, sha256(selectedBuildBytes));
const buildBoundAdd = analyzeExecuTorchModel(addBytes, "add.pte", {
  selectedBuildAttestation: resolvedSelectedBuild.attestation,
  selectedBuildInput: resolvedSelectedBuild.input,
});
assert.equal(buildBoundAdd.executorch_program.selected_build_binding.status, "SELECTED_BUILD_INVENTORY_SATISFIES_SERIALIZED_IDENTITIES");
assert.equal(buildBoundAdd.executorch_program.selected_build_binding.kernel_bindings[0].selected_build_status, "ATTESTED_PORTABLE_OPERATOR_INCLUDED");
assert.equal(buildBoundAdd.executorch_program.selected_build_binding.selected_build_input.file_sha256, sha256(selectedBuildBytes));
assert.equal(buildBoundAdd.weight_integrity.status, "pass");
const buildBoundReport = buildEngineeringReport(buildBoundAdd, { generatedAt: "2026-08-25T00:00:00.000Z" });
assert(buildBoundReport.includes(buildBoundAdd.executorch_program.graph_boundary)
  && buildBoundReport.includes(buildBoundAdd.executorch_program.selected_build_binding.interpretation_boundary),
"Build-bound ExecuTorch reports must preserve both reconstructed interpretation boundaries verbatim.");
assert(buildBoundReport.includes(selectedBuild.attestation_sha256)
  && buildBoundReport.includes("SELECTED_BUILD_INVENTORY_SATISFIES_SERIALIZED_IDENTITIES")
  && buildBoundReport.includes(sha256(selectedBuildBytes)));
assert.throws(() => resolveExecuTorchSelectedBuildAttestation([
  { path: "a/deepbom.executorch-build.json", bytes: selectedBuildBytes },
  { path: "b/deepbom.executorch-build.json", bytes: selectedBuildBytes },
]), /more than one/);
const duplicateKeyBytes = new TextEncoder().encode('{"schema":"deepbom.executorch_selected_build_attestation.v1","schema":"duplicate"}');
assert.throws(() => resolveExecuTorchSelectedBuildAttestation([{
  path: "deepbom.executorch-build.json",
  bytes: duplicateKeyBytes,
}]), /duplicate JSON key schema/);

const missingOperatorBuild = buildSelectedBuildAttestation({ portableOperatorNames: [], portableOpsEnabled: false });
const contradictedAdd = analyzeExecuTorchModel(addBytes, "add.pte", { selectedBuildAttestation: missingOperatorBuild });
assert.equal(contradictedAdd.executorch_program.selected_build_binding.status, "CONTRADICTION_SELECTED_BUILD_CANNOT_SATISFY_SERIALIZED_PROGRAM");
assert.equal(contradictedAdd.executorch_program.selected_build_binding.kernel_contradiction_count, 1);
assert(contradictedAdd.weight_integrity.issues.some((row) => row.code === "EXECUTORCH_SELECTED_BUILD_BINDING_CONTRADICTION"));
assert.throws(() => validateExecuTorchSelectedBuildAttestation({ ...selectedBuild, attestation_sha256: "0".repeat(64) }), /does not reconstruct/);

const bareFlatBuffer = Uint8Array.from([8, 0, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0]);
const xnnPayload = assessExecuTorchProcessedPayload("XnnpackBackend", bareFlatBuffer);
assert.equal(xnnPayload.structural_status, "OBSERVED_BOUNDED_FLATBUFFER_ROOT_ENVELOPE");
assert.equal(xnnPayload.root_type, "XNNGraph");
assert.equal(xnnPayload.byte_length, bareFlatBuffer.length);
assert.match(xnnPayload.sha256, /^[a-f0-9]{64}$/);
const invalidXnnPayload = assessExecuTorchProcessedPayload("XnnpackBackend", Uint8Array.from([0, 1, 2]));
assert.equal(invalidXnnPayload.structural_status, "CONTRADICTION_SOURCE_DECLARED_FLATBUFFER_ENVELOPE_INVALID");
const opaqueCoreMlPayload = assessExecuTorchProcessedPayload("CoreMLBackend", Uint8Array.from([0, 1, 2]));
assert.equal(opaqueCoreMlPayload.structural_status, "NOT_ASSESSED_SOURCE_BOUND_NON_FLATBUFFER_PAYLOAD");
assert.equal(EXECUTORCH_BACKEND_REGISTRY_SOURCE.backend_count, 7);
assert.equal(EXECUTORCH_BACKEND_REGISTRY_SOURCE.commit, "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f");
assert.equal(EXECUTORCH_BACKEND_REGISTRY_SOURCE.registry_sha256, "75538913e6cd07fa90450c82f7193688f84c78dd785224f7299949e6b0c20d43");
assert.deepEqual(Object.fromEntries(Object.entries(EXECUTORCH_BACKEND_REGISTRY_SOURCE.files).map(([key, row]) => [key, row.sha256])), {
  cmake_options: "d718e91fea803271f3febeb000bbbc1bba6c0305f4f6852db9bedf93a74c1c9b",
  schema_version: "d1853272c0ed0cf026ecec49f2ad6932d924cbca7b03a46d2ed16e73227a2047",
  runtime_loader: "d38be8eeec0fac0cea8f25d61820bc6f6d2bac4f07a89f3cb9ce175649260ca9",
});

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
const convolutionBackwardValues = tensorValues(14);
convolutionBackwardValues[3] = { kind: "IntList", values: [8], values_decimal: ["8"] };
convolutionBackwardValues[4] = { kind: "IntList", values: [1, 1], values_decimal: ["1", "1"] };
convolutionBackwardValues[5] = { kind: "IntList", values: [0, 0], values_decimal: ["0", "0"] };
convolutionBackwardValues[6] = { kind: "IntList", values: [1, 1], values_decimal: ["1", "1"] };
convolutionBackwardValues[7] = { kind: "Bool", value: false };
convolutionBackwardValues[8] = { kind: "IntList", values: [0, 0], values_decimal: ["0", "0"] };
convolutionBackwardValues[9] = { kind: "Int", value: 2, value_decimal: "2" };
convolutionBackwardValues[10] = { kind: "BoolList", values: [true, true, true] };
const convolutionBackwardTensors = Array.from({ length: 14 }, () => tensor([]));
convolutionBackwardTensors[0] = tensor([1, 8, 14, 14]);
convolutionBackwardTensors[1] = tensor([1, 4, 16, 16]);
convolutionBackwardTensors[2] = tensor([8, 2, 3, 3]);
convolutionBackwardTensors[11] = tensor([1, 4, 16, 16]);
convolutionBackwardTensors[12] = tensor([8, 2, 3, 3]);
convolutionBackwardTensors[13] = tensor([8]);
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::convolution_backward.out", Array.from({ length: 14 }, (_, index) => index), convolutionBackwardValues, convolutionBackwardTensors),
  { macs: 56448, decimal: "56448", status: "assessed_source_bound_convolution_backward" },
  "ExecuTorch convolution_backward must independently count requested grad-input and grad-weight contractions.",
);
const biasOnlyValues = convolutionBackwardValues.map((value) => ({ ...value }));
biasOnlyValues[10] = { kind: "BoolList", values: [false, false, true] };
assert.deepEqual(
  assessExecuTorchPortableKernelMac("aten::convolution_backward.out", Array.from({ length: 14 }, (_, index) => index), biasOnlyValues, convolutionBackwardTensors),
  { macs: 0, decimal: "0", status: "assessed_source_bound_convolution_backward_bias_only" },
  "ExecuTorch bias-only convolution backward must remain zero in the nominal tensor-contraction MAC metric.",
);
const invalidBackwardTensors = convolutionBackwardTensors.map((value) => ({ ...value, shape: [...value.shape] }));
invalidBackwardTensors[11] = tensor([1, 4, 15, 16]);
assert.match(
  assessExecuTorchPortableKernelMac("aten::convolution_backward.out", Array.from({ length: 14 }, (_, index) => index), convolutionBackwardValues, invalidBackwardTensors).status,
  /^not_assessed_shape_contract_conflict:grad_input_shape_mismatch$/,
  "ExecuTorch convolution_backward output-shape contradiction must fail closed.",
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

function buildSelectedBuildAttestation({ portableOperatorNames, portableOpsEnabled = true }) {
  const hex = (character) => character.repeat(64);
  const cmakeOptions = {
    EXECUTORCH_BUILD_COREML: false,
    EXECUTORCH_BUILD_CUDA: false,
    EXECUTORCH_BUILD_METAL: false,
    EXECUTORCH_BUILD_MPS: false,
    EXECUTORCH_BUILD_PORTABLE_OPS: portableOpsEnabled,
    EXECUTORCH_BUILD_QNN: false,
    EXECUTORCH_BUILD_VULKAN: false,
    EXECUTORCH_BUILD_XNNPACK: false,
  };
  const binaryInventory = [{ path: "bin/executorch_runner", byte_length: 4096, sha256: hex("a") }];
  const normalized = {
    schema: EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA,
    evidence_class: "REPRODUCIBLE_SELECTED_BUILD_ATTESTATION",
    source: {
      repository: "pytorch/executorch",
      release: "v1.4.1",
      commit: "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f",
      pristine_before_build: true,
      submodule_status_sha256: hex("b"),
      post_build_diff_sha256: hex("c"),
      backend_registry_sha256: EXECUTORCH_BACKEND_REGISTRY_SOURCE.registry_sha256,
    },
    build: {
      configuration: "test-release",
      cmake_options: cmakeOptions,
      linked_backend_ids: [],
      custom_backend_sources: [],
      portable_operator_names: [...portableOperatorNames].sort(),
      custom_operator_names: [],
      cmake_cache_sha256: hex("d"),
      build_stdout_sha256: hex("e"),
      build_stderr_sha256: hex("f"),
    },
    runtime: {
      platform: "linux",
      arch: "x86_64",
      binary_inventory: binaryInventory,
      binary_inventory_sha256: sha256TextHex(canonicalJson(binaryInventory)),
      primary_binary_path: "bin/executorch_runner",
      primary_binary_sha256: hex("a"),
    },
    boundary: null,
  };
  return { ...normalized, attestation_sha256: sha256TextHex(canonicalJson(normalized)) };
}
