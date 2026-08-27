import { createHash } from "node:crypto";
import { analyzeOnnxModel } from "../web/onnx.js";
import { prepareOnnxExternalDataFiles } from "../web/lib/onnx-external-data.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildInterfaceQuantizationContractLedger } from "../web/lib/quantization-contract-summary.js";
import { createCheck } from "./check-assert.mjs";
import {
  bytesField,
  externalTensor,
  float32,
  graphProto,
  int32,
  message,
  model,
  node,
  nodeWithGraphAttribute,
  nodeWithIntegerAttributes,
  stringField,
  tensor,
  tensorAnnotation,
  valueInfo,
  valueInfoWithoutShape,
  varintField,
} from "./onnx-proto-fixture.mjs";

const { done, expect, expectDeepEqual, expectEqual, expectThrows } = createCheck("ONNX quantization contract check");

const validBytes = qlinearConvModel();
const valid = analyzeOnnxModel(validBytes, "qlinearconv_valid.onnx");
const contracts = buildQuantizationContractChecks(valid);
expectEqual(valid.onnx_quantization_binding.status, "pass", "Valid QLinearConv parameters should bind completely.");
expectEqual(valid.onnx_quantization_binding.binding_count, 3, "QLinearConv should bind input, weight, and output contracts.");
expectEqual(valid.ops[0].macs, 8, "QLinearConv MACs should use x/w/y signature positions and inferred NCHW output shape.");
expectEqual(valid.size_breakdown.stored_scalar_elements, 12, "Raw-data scalar quantization parameters should count as one ONNX initializer element each.");
expectEqual(valid.size_breakdown.theoretical_fp16_constant_bytes, 22, "Mixed-dtype FP16 projection should include scalar scale parameters and preserve non-float payload widths.");
expectEqual(valid.size_breakdown.theoretical_int8_constant_bytes, 18, "Mixed-dtype INT8 projection should include scalar scale parameters and preserve non-float payload widths.");
expect(valid.tensors.some((tensor) => tensor.constant_buffer && Number(tensor.buffer_data_length || 0) > 0), "Embedded initializer tensors should expose their exact stored payload length to common viewer inventory.");
expect(valid.tensors.filter((tensor) => tensor.constant_buffer).every((tensor) => tensor.buffer_data_status === "observed_embedded_initializer_payload"), "Embedded initializer byte evidence should retain an explicit observed status.");
expectEqual(valid.per_channel_tensors, 1, "Two QLinearConv weight scales should produce one per-axis tensor.");
expectDeepEqual(valid.onnx_quantization_binding.bindings.find((item) => item.tensor_name === "w")?.scale_values, [0.25, 0.5], "Weight scales should retain complete TensorProto values in the canonical binding record.");
expectEqual(contracts.contract_integrity_status, "pass", "Valid ONNX quantization contracts should pass.");
expectEqual(contracts.bias_scale.contract_kind, "onnx_implicit_qlinearconv_bias_scale", "QLinearConv bias should use its implicit ONNX scale contract.");
expectDeepEqual(contracts.bias_scale.details[0].expected_bias_scales, [0.125, 0.25], "Implicit QLinearConv bias scales should equal x_scale times each weight scale.");
expectEqual(contracts.accumulator_bound.checked_ops, 1, "QLinearConv should receive a deterministic INT32 accumulator bound.");
expectEqual(valid.weight_integrity.quantized_constant_tensors_scanned, 1, "Bound quantized kernel grid should be scanned.");
expectEqual(valid.weight_integrity.min_grid_utilization, 2 / 256, "Grid utilization should count exact stored INT8 levels.");
expectEqual(valid.weight_integrity.max_saturation_percent, 1, "Both endpoint-valued weights should produce exact 100% endpoint saturation.");
expectEqual(valid.input_contracts?.[0]?.layout, "NCHW", "QLinearConv input layout should derive from standard-domain Conv semantics.");
expectEqual(valid.input_contracts?.[0]?.layout_evidence_class, "DERIVED", "QLinearConv input layout should not be a convention-based prediction.");
expectEqual(valid.input_contracts?.[0]?.tensor_numerical_contract_status, "known_from_artifact_quantization_metadata", "Bound scalar QLinearConv input quantization should produce an exact tensor numerical contract.");
expectEqual(valid.input_contracts?.[0]?.expected_range_low, -64, "QLinearConv input lower real bound should equal scale*(qmin-zp).");
expectEqual(valid.input_contracts?.[0]?.expected_range_high, 63.5, "QLinearConv input upper real bound should equal scale*(qmax-zp).");
const mlBom = buildMlBomDocument(valid, { hash: "" });
const bundleFiles = buildEngineeringBundleArtifactFiles(valid, {
  reportContext: { identity: { filename: valid.filename, format: "onnx" }, generatedAt: "2026-07-16T00:00:00.000Z" },
  rawEvidenceContext: { identity: { filename: valid.filename, format: "onnx" } },
  mlBomDocument: mlBom,
});
const report = bundleFiles.find((file) => file.name === "engineering_report.md")?.data || "";
const evidence = JSON.parse(bundleFiles.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expect(report.includes("3 binding(s); 0 invalid, 0 unresolved"), "Engineering report should render ONNX parameter-binding coverage.");
expect(report.includes("minimum level utilization 0.8%; maximum qmin/qmax saturation 100%"), "Engineering report should render exact ONNX quantized-weight grid metrics.");
expect(report.includes("0 out-of-range/type violation(s)"), "ONNX zero-point reporting should use the ONNX dtype/range contract.");
expect(!report.includes("invalid/symmetric-INT8 violation(s)"), "ONNX reporting must not apply TFLite's symmetric INT8 weight expectation.");
expectEqual(evidence.evidence?.quantization?.quantization_contract_checks?.bias_scale?.expected_scale_count, undefined, "Bias scale counts should remain in per-op details rather than an invented aggregate field.");
expectEqual(evidence.evidence?.conformance_report?.status, "pass", "Quantized ONNX engineering evidence should pass semantic conformance.");

const invalid = analyzeOnnxModel(qlinearConvModel({ weightScales: [0.25, 0.5, 0.75] }), "qlinearconv_bad_axis.onnx");
const invalidContracts = buildQuantizationContractChecks(invalid);
expectEqual(invalid.onnx_quantization_binding.status, "fail", "Per-axis scale cardinality mismatch should fail closed.");
expectEqual(invalidContracts.parameter_integrity.invalid_bindings, 1, "Exactly the weight binding should fail cardinality validation.");
expect((invalid.onnx_quantization_binding.bindings.find((item) => item.role === "kernel_weight")?.reasons || []).some((reason) => reason.includes("does_not_match_axis_0_dimension_2")), "Failure should identify exact scale and axis cardinalities.");
expect(buildFindingsRegister(invalid).some((item) => item.finding_id === "EA-QNT-0105" && item.technical_priority === "High"), "Invalid ONNX quantization parameters should enter the High action queue.");

const qdq = analyzeOnnxModel(qdqModel(), "qdq_roundtrip.onnx");
const qdqContracts = buildQuantizationContractChecks(qdq);
expectEqual(qdq.onnx_quantization_binding.explicit_qdq_boundary_count, 2, "QuantizeLinear and DequantizeLinear boundaries should both be inventoried.");
expectEqual(qdqContracts.qdq_boundaries.runtime_materialization_status, "not_assessed_static_graph_only", "Static Q/DQ syntax must not claim runtime materialization.");
expectEqual(qdq.tensors.find((tensor) => tensor.name === "q")?.quantization_binding_status, "pass", "Matching Q and DQ parameters should bind to the same integer tensor without conflict.");
expectDeepEqual(focusedEvidence(analyzeOnnxModel(validBytes, "qlinearconv_valid.onnx")), focusedEvidence(valid), "The synthetic corpus should reproduce identical focused evidence from identical bytes.");

const annotated = analyzeOnnxModel(model(
  [node("Identity", "annotated_identity", ["q"], ["y"])],
  [tensor("ann_scale", 1, [], float32([0.125])), tensor("ann_zp", 2, [], new Uint8Array([128]))],
  [valueInfo("q", 2, [1, 4])], [valueInfo("y", 2, [1, 4])], 13,
  [tensorAnnotation("q", [["SCALE_TENSOR", "ann_scale"], ["ZERO_POINT_TENSOR", "ann_zp"]])],
), "tensor_annotation_valid.onnx");
const annotationBinding = annotated.onnx_quantization_binding.bindings.find((item) => item.binding_source === "graph_quantization_annotation");
expectEqual(annotated.onnx_quantization_binding.schema, "deepbom.onnx_quantization_binding.v1.1", "TensorAnnotation support must version the ONNX binding schema.");
expectEqual(annotated.onnx_quantization_binding.main_graph_annotation_count, 1, "Main GraphProto TensorAnnotation count must be exact.");
expectEqual(annotationBinding?.status, "pass", "A complete scalar TensorAnnotation mapping must bind without an operator witness.");
expectEqual(annotationBinding?.operator_cross_check_status, "not_applicable_no_operator_binding", "Annotation-only contracts must not invent an operator cross-check.");
expectEqual(annotated.tensors.find((row) => row.name === "q")?.scale_sample?.[0], 0.125, "A valid TensorAnnotation must populate the annotated tensor numerical contract.");
expect(annotationBinding?.source_ref.includes("/onnx/onnx.in.proto") && /^[a-f0-9]{64}$/.test(annotationBinding?.source_sha256 || ""), "TensorAnnotation evidence must bind the pinned ONNX protobuf source and hash.");
const annotatedBundle = buildEngineeringBundleArtifactFiles(annotated, {
  reportContext: { identity: { filename: annotated.filename, format: "onnx" } },
  rawEvidenceContext: { identity: { filename: annotated.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(annotated, { hash: "" }),
});
const annotatedReport = annotatedBundle.find((file) => file.name === "engineering_report.md")?.data || "";
expect(annotatedReport.includes("ONNX GraphProto TensorAnnotation Bindings") && annotatedReport.includes("ann_scale") && annotatedReport.includes(annotationBinding.source_sha256), "Engineering Report must render the annotation mapping and pinned source identity.");
expectEqual(JSON.parse(annotatedBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}").evidence?.conformance_report?.status, "pass", "TensorAnnotation evidence must pass independent conformance.");

const duplicateAnnotation = analyzeOnnxModel(model(
  [], [tensor("s", 1, [], float32([0.25])), tensor("zp", 2, [], new Uint8Array([0]))],
  [valueInfo("q", 2, [1])], [valueInfo("q", 2, [1])], 13,
  [tensorAnnotation("q", [["SCALE_TENSOR", "s"], ["SCALE_TENSOR", "s"], ["ZERO_POINT_TENSOR", "zp"]])],
), "tensor_annotation_duplicate_key.onnx");
expectEqual(duplicateAnnotation.onnx_quantization_binding.status, "fail", "Duplicate TensorAnnotation parameter keys must fail closed.");
expect(duplicateAnnotation.onnx_quantization_binding.bindings[0].reasons.includes("duplicate_quant_parameter_key_SCALE_TENSOR"), "Duplicate-key failure must retain the exact key identity.");

const perAxisAnnotation = analyzeOnnxModel(model(
  [], [tensor("s", 1, [2], float32([0.25, 0.5])), tensor("zp", 3, [2], new Uint8Array([0, 0]))],
  [valueInfo("q", 3, [2, 4])], [valueInfo("q", 3, [2, 4])], 13,
  [tensorAnnotation("q", [["SCALE_TENSOR", "s"], ["ZERO_POINT_TENSOR", "zp"]])],
), "tensor_annotation_axis_unbound.onnx");
expectEqual(perAxisAnnotation.onnx_quantization_binding.status, "partial", "A vector TensorAnnotation without an operator axis witness must remain partial.");
expectEqual(perAxisAnnotation.onnx_quantization_binding.unresolved_annotation_count, 1, "Axis-unbound annotation count must be exact.");

const annotationConflict = analyzeOnnxModel(model(
  [node("QuantizeLinear", "quant", ["x", "s", "zp"], ["q"])],
  [
    tensor("s", 1, [], float32([0.25])), tensor("s_other", 1, [], float32([0.5])),
    tensor("zp", 2, [], new Uint8Array([7])),
  ],
  [valueInfo("x", 1, [1, 4])], [valueInfo("q", 2, [1, 4])], 13,
  [tensorAnnotation("q", [["SCALE_TENSOR", "s_other"], ["ZERO_POINT_TENSOR", "zp"]])],
), "tensor_annotation_conflict.onnx");
const conflictingAnnotation = annotationConflict.onnx_quantization_binding.bindings.find((item) => item.binding_source === "graph_quantization_annotation");
expectEqual(conflictingAnnotation?.status, "fail", "A TensorAnnotation numerical mapping that contradicts Q/DQ must fail.");
expect(conflictingAnnotation?.reasons.includes("annotation_numerical_contract_conflicts_with_operator_binding"), "Q/DQ conflict must retain its deterministic reason.");
expectEqual(annotationConflict.tensors.find((row) => row.name === "q")?.quantization_binding_status, "conflict", "The annotated tensor must expose the cross-source conflict.");

const nestedAnnotatedGraph = graphProto([], [tensor("nested_s", 1, [], float32([0.25])), tensor("nested_zp", 2, [], new Uint8Array([0]))], [valueInfo("nested_q", 2, [1])], [valueInfo("nested_q", 2, [1])], "nested_quant", [tensorAnnotation("nested_q", [["SCALE_TENSOR", "nested_s"], ["ZERO_POINT_TENSOR", "nested_zp"]])]);
const nestedAnnotated = analyzeOnnxModel(model([nodeWithGraphAttribute("ScopedBody", "scoped_quant", "body", nestedAnnotatedGraph, "com.acme")], [], [], [], 13), "nested_tensor_annotation.onnx");
expectEqual(nestedAnnotated.onnx_quantization_binding.nested_graph_annotation_count, 1, "Nested GraphProto annotations must be inventoried instead of silently dropped.");
expectEqual(nestedAnnotated.onnx_quantization_binding.status, "partial", "Nested annotations must keep the binding assessment partial until scope-local binding is reconstructed.");
expectEqual(nestedAnnotated.onnx_quantization_binding.annotation_scope_status, "main_graph_bound_nested_graph_annotations_inventoried_not_bound", "Nested annotation scope boundary must be explicit.");

const longPerAxisScales = Array.from({ length: 300 }, (_, index) => Math.fround(0.001 + index / 100_000));
const longPerAxis = analyzeOnnxModel(model(
  [node("QuantizeLinear", "quant_long_axis", ["x", "s", "zp"], ["q"])],
  [tensor("s", 1, [300], float32(longPerAxisScales)), tensor("zp", 2, [300], new Uint8Array(300))],
  [valueInfo("x", 1, [1, 300])],
  [valueInfo("q", 2, [1, 300])],
  13,
), "qdq_external_per_axis_300.onnx");
const longPerAxisTensor = longPerAxis.outputs[0];
const longPerAxisLedger = buildInterfaceQuantizationContractLedger(longPerAxis);
const longPerAxisContract = longPerAxisLedger.parameters.find((row) => row.direction === "output");
expectEqual(longPerAxisTensor.scale_sample.length, 256, "General ONNX tensor preview remains bounded to 256 scale values.");
expectEqual(longPerAxisTensor.interface_scale_values.length, 300, "External interface contract must retain every per-axis scale value.");
expectEqual(longPerAxisContract.quantization.status, "complete", "A 300-value external per-axis contract must not fail because the viewer sample is bounded.");
expectEqual(longPerAxisContract.quantization.scale_count, 300, "External per-axis ledger scale cardinality must remain exact.");
expectDeepEqual(longPerAxisContract.quantization.scales, longPerAxisScales, "External per-axis ledger must preserve all decoded scales in order.");

const runtimePerAxis = analyzeOnnxModel(model(
  [node("QuantizeLinear", "quant_runtime_axis", ["x", "s", "zp"], ["q"])],
  [],
  [valueInfo("x", 1, [1, 3, 3, 2]), valueInfo("s", 1, [3]), valueInfo("zp", 2, [3])],
  [valueInfo("q", 2, [1, 3, 3, 2])],
  25,
), "qdq_runtime_per_axis.onnx");
const runtimePerAxisBinding = runtimePerAxis.onnx_quantization_binding.bindings[0];
expectEqual(runtimePerAxis.onnx_quantization_binding.status, "partial", "Runtime-supplied affine values must keep the numerical assessment partial.");
expectEqual(runtimePerAxisBinding.status, "not_assessed_runtime_parameter_values", "Present runtime parameters must not be mislabeled as missing parameters.");
expectEqual(runtimePerAxisBinding.parameterization, "per_axis", "A declared three-element scale parameter must retain per-axis structure.");
expectEqual(runtimePerAxisBinding.axis, 1, "QuantizeLinear opset 25 must retain the schema-default axis 1.");
expectEqual(runtimePerAxisBinding.axis_source, "schema_default_axis_1", "Default-axis provenance must remain explicit.");
expectEqual(runtimePerAxisBinding.scale_count, 3, "Declared scale cardinality must be derived from the serialized parameter shape.");
expectEqual(runtimePerAxisBinding.scale_value_count, 0, "Runtime scale values must not be invented from shape metadata.");
expectEqual(runtimePerAxisBinding.scale_cardinality_source, "declared_tensor_shape", "Scale cardinality and scale values must have separate provenance.");
expectEqual(runtimePerAxisBinding.cardinality_status, "pass", "The declared scale shape must be checked against the selected tensor axis.");
expectEqual(runtimePerAxisBinding.value_evidence_class, "RUNTIME_REQUIRED", "Unserialized affine values must remain runtime-required evidence.");
expectEqual(runtimePerAxis.outputs[0].quantization_parameterization, "per_axis", "Per-axis structure must reach the common tensor inventory.");
expectEqual(runtimePerAxis.outputs[0].quant_scales, 0, "The tensor inventory must not report unobserved scale values as decoded scales.");
expectEqual(runtimePerAxis.per_channel_tensors, 1, "Structurally declared per-axis tensors must be counted separately from decoded-value coverage.");

const blocked = analyzeOnnxModel(model(
  [nodeWithIntegerAttributes("QuantizeLinear", "quant_blocked", ["x", "s", "zp"], ["q"], { axis: 1, block_size: 10 })],
  [tensor("s", 1, [1, 30], float32(Array.from({ length: 30 }, (_, index) => 0.01 + index / 10_000))), tensor("zp", 2, [1, 30], new Uint8Array(30))],
  [valueInfo("x", 1, [1, 300])],
  [valueInfo("q", 2, [1, 300])],
  21,
), "qdq_external_blocked.onnx");
const blockedLedger = buildInterfaceQuantizationContractLedger(blocked);
const blockedContract = blockedLedger.parameters.find((row) => row.direction === "output");
expectEqual(blocked.outputs[0].quantization_parameterization, "blocked", "ONNX blocked parameterization must survive tensor serialization.");
expectEqual(blocked.outputs[0].quantization_block_size, 10, "ONNX block_size must survive tensor serialization.");
expectEqual(blockedContract.quantization.granularity, "blocked", "External contract must classify blocked quantization separately from per-axis.");
expectEqual(blockedContract.quantization.block_size, 10, "External blocked contract must retain block_size.");
expectEqual(blockedContract.quantization.cardinality_status, "valid", "Valid blocked scale geometry must pass the external interface contract.");

const customCollision = analyzeOnnxModel(model(
  [node("Conv", "custom_conv", ["x", "w"], ["y"], "com.acme")],
  [tensor("w", 1, [2, 1, 1, 1], float32([1, 1]))],
  [valueInfo("x", 1, [1, 1, 2, 2])],
  [valueInfo("y", 1, [1, 2, 2, 2])],
  13,
), "custom_domain_name_collision.onnx");
expectEqual(customCollision.ops[0].domain, "com.acme", "Custom-domain identity should survive parsing.");
expectEqual(customCollision.ops[0].standard_domain, false, "A custom Conv name must not be classified as ai.onnx Conv.");
expectEqual(customCollision.ops[0].macs_status, "not_assessed", "Custom-domain Conv MAC semantics must remain unassessed.");
expectEqual(customCollision.ops[0].macs, null, "Custom-domain Conv MACs must be null rather than a standard-Conv value or zero.");
expectEqual(customCollision.ops[0].row_working_set_bytes, null, "Custom-domain Conv row working set must remain unassessed.");
expectEqual(customCollision.onnx_shape_inference.rule_supported_nodes, 0, "Custom-domain Conv must not consume the ai.onnx shape rule.");
expectEqual(customCollision.onnx_shape_inference.rule_unsupported_nodes, 1, "Custom-domain Conv should be counted in unsupported shape semantics.");
expectEqual(customCollision.stages[0].key, "com.acme:Conv", "Stage grouping must preserve custom domain identity.");
expectEqual(customCollision.input_contracts?.[0]?.layout, null, "A custom-domain Conv name must not inherit the ai.onnx NCHW contract.");
expectEqual(customCollision.input_contracts?.[0]?.layout_evidence_class, "NOT_ASSESSABLE", "Custom-domain layout must remain explicitly unassessed.");
const partialPayload = analyzeOnnxModel(model(
  [node("CustomTransform", "custom_dynamic", ["x"], ["y"], "com.acme")],
  [],
  [valueInfo("x", 1, [1, 4])],
  [valueInfoWithoutShape("y", 1)],
  13,
), "custom_domain_unknown_payload.onnx");
expectEqual(partialPayload.ops[0].estimated_bytes_status, "not_assessed", "Unknown ONNX tensor payload must not be substituted with zero bytes.");
expectEqual(partialPayload.ops[0].estimated_bytes, null, "Unknown ONNX op traffic must serialize as null.");
expectEqual(partialPayload.tensor_liveness.status, "partial", "Known and unknown activation payloads should produce a partial liveness result.");
expectEqual(partialPayload.tensor_liveness.peak_bytes_status, "assessed_tensor_lower_bound", "Partial ONNX liveness must be labeled as a lower bound.");
expectEqual(partialPayload.tensor_liveness.unassessed_tensor_count, 1, "The unknown custom output should remain in the liveness exclusion ledger.");
const scalarPayload = analyzeOnnxModel(model(
  [node("Identity", "scalar_identity", ["x"], ["y"])],
  [],
  [valueInfo("x", 1, [])],
  [valueInfo("y", 1, [])],
  13,
), "scalar_payload.onnx");
expectEqual(scalarPayload.inputs[0].shape_declared, true, "An explicit empty ONNX TensorShapeProto must be preserved as a declared scalar shape.");
expectEqual(scalarPayload.ops[0].estimated_bytes, 8, "Scalar FLOAT32 identity traffic should include one input and one output scalar.");
expectEqual(scalarPayload.tensor_liveness.status, "assessed", "Declared scalar tensors should participate in complete liveness assessment.");

const externalPayloadFile = new Uint8Array(24);
const externalPayloadView = new DataView(externalPayloadFile.buffer);
externalPayloadView.setFloat32(16, 1.25, true);
externalPayloadView.setFloat32(20, -2.5, true);
const preparedExternalFiles = await prepareOnnxExternalDataFiles([{
  name: "weights.bin",
  size: externalPayloadFile.byteLength,
  arrayBuffer: async () => externalPayloadFile.buffer.slice(0),
}]);
const expectedExternalSha256 = createHash("sha256").update(externalPayloadFile).digest("hex");
const expectedExternalSha1 = createHash("sha1").update(externalPayloadFile).digest("hex");
expectEqual(preparedExternalFiles[0].sha256, expectedExternalSha256, "Browser sidecar preparation should hash the exact selected bytes with SHA-256.");
expectEqual(preparedExternalFiles[0].sha1, expectedExternalSha1, "Browser sidecar preparation should hash the exact selected bytes with standard SHA-1.");
const preparedDirectoryFiles = await prepareOnnxExternalDataFiles([{
  name: "weights.bin",
  webkitRelativePath: "model_bundle/data/weights.bin",
  size: externalPayloadFile.byteLength,
  arrayBuffer: async () => externalPayloadFile.buffer.slice(0),
}]);
expectEqual(preparedDirectoryFiles[0].path, "data/weights.bin", "Directory selection should remove only the picker root and retain the model-relative nested sidecar path.");

const externalModel = model(
  [node("Add", "external_add", ["x", "w"], ["y"])],
  [externalTensor("w", 1, [1, 2], [["location", "weights.bin"], ["offset", "16"], ["length", "8"], ["checksum", expectedExternalSha1]])],
  [valueInfo("x", 1, [1, 2])],
  [valueInfo("y", 1, [1, 2])],
  13,
);
const external = analyzeOnnxModel(externalModel, "external_data_valid.onnx");
expectEqual(external.onnx_external_data.schema, "deepbom.onnx_external_data.v1.3", "External-data evidence should be versioned.");
expectEqual(external.onnx_external_data.status, "not_assessed_payload_not_supplied", "A valid sidecar reference must remain unassessed until payload bytes are supplied.");
expectEqual(external.onnx_external_data.tensor_count, 1, "Exactly one external initializer should be inventoried.");
expectEqual(external.onnx_external_data.entry_count, 4, "Every external_data key/value record should be preserved.");
expectEqual(external.onnx_external_data.declared_payload_bytes, 8, "Declared external byte ranges should sum exactly when every length is present.");
expectEqual(external.onnx_external_data.invalid_checksum_count, 0, "A standard hexadecimal SHA-1 declaration should pass syntax validation.");
expectEqual(external.onnx_external_data.tensors[0].location_status, "safe_relative_path", "A simple relative sidecar path should pass path syntax validation.");
expectEqual(external.tensors.find((tensor) => tensor.name === "w")?.external_data?.[0]?.value, "weights.bin", "Public tensor evidence should preserve exact external_data entries.");
expectEqual(external.tensors.find((tensor) => tensor.name === "w")?.buffer_data_length, null, "Unsupplied external initializer bytes must remain null rather than zero.");
expectEqual(external.tensors.find((tensor) => tensor.name === "w")?.buffer_data_status, "not_assessed_external_payload_unavailable", "Unsupplied external initializer bytes should retain an explicit not-assessed reason.");
expectEqual(external.size_breakdown.constant_tensor_count, 1, "External initializer declarations must remain inventoried.");
expectEqual(external.size_breakdown.embedded_constant_tensor_count, 0, "Unsupplied external initializer payloads must not be counted as embedded constants.");
expectEqual(external.size_breakdown.stored_scalar_elements, 0, "Unsupplied external initializer scalar cardinality must not be presented as stored payload coverage.");
expectEqual(external.size_breakdown.metrics.zero_constant_byte_ratio.status, "not_assessed_external_data", "An external-only model must not emit an assessed null zero-byte ratio.");
expect(buildFindingsRegister(external).some((item) => item.finding_id === "EA-ONX-0004" && item.technical_priority === "High"), "Unsupplied external initializer payloads should enter the High action queue.");
const externalBundle = buildEngineeringBundleArtifactFiles(external, {
  reportContext: { identity: { filename: external.filename, format: "onnx" } },
  rawEvidenceContext: { identity: { filename: external.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(external, { hash: "" }),
});
const externalReport = externalBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const externalEvidence = JSON.parse(externalBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expectEqual(externalEvidence.evidence?.conformance_report?.status, "pass", "External-data reference evidence should pass independent conservation while payload coverage remains unassessed.");
expect(externalReport.includes("weights.bin") && externalReport.includes("not_assessed_payload_not_supplied") && externalReport.includes("8 B"), "Engineering report should render exact sidecar location, status, and declared range length.");

const largeUnboundExternal = analyzeOnnxModel(model(
  [],
  [externalTensor("large_w", 1, [100_000_001], [["location", "large-weights.bin"], ["offset", "0"], ["length", "400000004"]])],
  [],
  [],
  13,
), "large_external_data_unbound.onnx");
expectEqual(largeUnboundExternal.onnx_external_data.status, "not_assessed_payload_not_supplied", "A large logical external initializer must remain analyzable without materializing its absent payload.");
expectEqual(largeUnboundExternal.onnx_external_data.declared_payload_bytes, 400_000_004, "Large external-data cardinality should remain exactly derived from dtype and shape.");
expectEqual(largeUnboundExternal.size_breakdown.stored_scalar_elements, 0, "An unbound large external initializer must not be counted as decoded or stored payload coverage.");

const verifiedExternal = analyzeOnnxModel(externalModel, "external_data_verified.onnx", null, { externalDataFiles: preparedExternalFiles });
const verifiedTensor = verifiedExternal.tensors.find((tensor) => tensor.name === "w");
expectEqual(verifiedExternal.onnx_external_data.status, "verified_payloads", "A path/range/cardinality/checksum-valid sidecar should produce complete external payload coverage.");
expectEqual(verifiedExternal.onnx_external_data.supplied_file_count, 1, "The selected sidecar file ledger should preserve exact file cardinality.");
expectEqual(verifiedExternal.onnx_external_data.used_file_count, 1, "A referenced selected sidecar should be marked used exactly once at file level.");
expectEqual(verifiedExternal.onnx_external_data.verified_payload_count, 1, "Exactly one external tensor range should be verified.");
expectEqual(verifiedExternal.onnx_external_data.verified_payload_bytes, 8, "Verified external payload bytes should equal the selected tensor range.");
expectEqual(verifiedTensor?.external_payload_verified, true, "Public tensor evidence should expose verified external payload state.");
expectEqual(verifiedTensor?.external_sidecar_sha256, expectedExternalSha256, "Public tensor evidence should bind the selected sidecar SHA-256.");
expectEqual(verifiedTensor?.initializer_bytes, 8, "Verified external raw_data should hydrate the initializer payload without becoming embedded bytes.");
expectEqual(verifiedTensor?.buffer_data_length, 8, "Verified external initializer bytes should enter the common stored-payload inventory.");
expectEqual(verifiedTensor?.buffer_data_status, "verified_external_initializer_payload", "Verified external initializer bytes should retain their verification basis.");
expectEqual(verifiedExternal.size_breakdown.constant_bytes, 0, "External payload bytes must not be misclassified as bytes embedded in ModelProto.");
expectEqual(verifiedExternal.size_breakdown.verified_external_payload_bytes, 8, "Size evidence should count verified external payload bytes separately.");
expectEqual(verifiedExternal.size_breakdown.available_initializer_bytes, 8, "Available initializer bytes should add embedded and verified external payloads.");
expectEqual(verifiedExternal.size_breakdown.available_initializer_scalar_elements, 2, "Verified FLOAT32 sidecar cardinality should be decoded from dtype and shape.");
expectEqual(verifiedExternal.weight_integrity.elements_scanned, 2, "Verified external FLOAT32 values should enter initializer integrity scanning.");
expectEqual(verifiedExternal.size_breakdown.metrics.zero_constant_byte_ratio.status, "assessed", "Complete raw_data coverage should make the raw zero-byte ratio assessable.");
expect(!buildFindingsRegister(verifiedExternal).some((item) => item.finding_id === "EA-ONX-0004"), "Complete external payload coverage should suppress the incomplete-payload finding.");
expectEqual(buildArtifactEvidenceEnvelope(verifiedExternal).external_files[0]?.verification_status, "payload_range_cardinality_and_hash_verified", "A fully verified ONNX sidecar should retain its exact successful verification state in the canonical envelope.");
const verifiedExternalBundle = buildEngineeringBundleArtifactFiles(verifiedExternal, {
  reportContext: { identity: { filename: verifiedExternal.filename, format: "onnx" } },
  rawEvidenceContext: { identity: { filename: verifiedExternal.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(verifiedExternal, { hash: "" }),
});
const verifiedExternalReport = verifiedExternalBundle.find((file) => file.name === "engineering_report.md")?.data || "";
const verifiedExternalEvidence = JSON.parse(verifiedExternalBundle.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
expectEqual(verifiedExternalEvidence.evidence?.conformance_report?.status, "pass", "Verified external payload evidence should pass independent report conformance.");
expect(verifiedExternalReport.includes(expectedExternalSha256) && verifiedExternalReport.includes("verified_payloads"), "Engineering report should render the exact verified sidecar SHA-256 and coverage state.");

const nestedExternalGraph = graphProto([], [
  externalTensor("nested_w", 1, [1, 2], [["location", "weights.bin"], ["offset", "16"], ["length", "8"], ["checksum", expectedExternalSha1]]),
], [], [], "nested_external_graph");
const functionDefault = functionProtoWithExternalDefault(
  "WithExternalDefault",
  externalTensor("function_default_w", 1, [1, 2], [["location", "weights.bin"], ["offset", "16"], ["length", "8"], ["checksum", expectedExternalSha1]]),
);
const scopedExternal = analyzeOnnxModel(modelWithFunctions([
  nodeWithGraphAttribute("ScopedBody", "scoped_body", "body", nestedExternalGraph, "com.acme"),
], [functionDefault], 13), "external_data_scoped.onnx", null, { externalDataFiles: preparedExternalFiles });
expectEqual(scopedExternal.onnx_external_data.status, "verified_payloads", "Every parsed graph/function TensorProto sidecar range should be verified, not only main-graph initializers.");
expectEqual(scopedExternal.onnx_external_data.tensor_count, 2, "Nested GraphProto and FunctionProto default-attribute external tensors should both be inventoried.");
expectEqual(scopedExternal.onnx_external_data.verified_payload_count, 2, "Both all-scope external TensorProto payloads should be verified.");
expectEqual(scopedExternal.onnx_external_data.verified_payload_bytes, 16, "All-scope payload arithmetic should count each referenced tensor range exactly once.");
expectEqual(scopedExternal.onnx_external_data.used_file_count, 1, "Two ranges may bind one sidecar without duplicating the file ledger.");
expect(scopedExternal.onnx_external_data.tensors.some((row) => row.scope === "main_graph/node:0/attribute:body" && row.tensor_role === "graph_initializer"), "Nested GraphProto initializer evidence should preserve its exact scope and role.");
expect(scopedExternal.onnx_external_data.tensors.some((row) => row.scope.includes("default_attribute:weight") && row.tensor_role === "function_default_attribute_tensor"), "FunctionProto default TensorProto evidence should preserve its exact scope and role.");
const scopedExternalBundle = buildEngineeringBundleArtifactFiles(scopedExternal, {
  reportContext: { identity: { filename: scopedExternal.filename, format: "onnx" } },
  rawEvidenceContext: { identity: { filename: scopedExternal.filename, format: "onnx" } },
  mlBomDocument: buildMlBomDocument(scopedExternal, { hash: "" }),
});
const scopedExternalReport = scopedExternalBundle.find((file) => file.name === "engineering_report.md")?.data || "";
expect(scopedExternalReport.includes("All-scope ONNX external data") && scopedExternalReport.includes("function_default_attribute_tensor"), "Engineering Report should distinguish main-graph size accounting from all-scope external TensorProto coverage.");

const checksumMismatchModel = model(
  [node("Add", "external_add", ["x", "w"], ["y"])],
  [externalTensor("w", 1, [1, 2], [["location", "weights.bin"], ["offset", "16"], ["length", "8"], ["checksum", "f".repeat(40)]])],
  [valueInfo("x", 1, [1, 2])], [valueInfo("y", 1, [1, 2])], 13,
);
const checksumMismatch = analyzeOnnxModel(checksumMismatchModel, "external_checksum_mismatch.onnx", null, { externalDataFiles: preparedExternalFiles });
expectEqual(checksumMismatch.onnx_external_data.status, "payload_verification_failed", "A mismatched declared whole-file SHA-1 must fail payload verification.");
expectEqual(checksumMismatch.onnx_external_data.checksum_mismatch_count, 1, "Checksum mismatch count should identify the exact failed range.");
expectEqual(buildArtifactEvidenceEnvelope(checksumMismatch).external_files[0]?.verification_status, "payload_verification_incomplete:checksum_mismatch", "A hash-identified but checksum-rejected ONNX sidecar must remain a dependency without being mislabeled verified.");
expectEqual(checksumMismatch.tensors.find((tensor) => tensor.name === "w")?.initializer_bytes, 0, "A checksum-failed sidecar must never hydrate initializer values.");

const rangeFailure = analyzeOnnxModel(model(
  [], [externalTensor("w", 1, [1, 2], [["location", "weights.bin"], ["offset", "20"], ["length", "8"]])], [], [], 13,
), "external_range_failure.onnx", null, { externalDataFiles: preparedExternalFiles });
expectEqual(rangeFailure.onnx_external_data.range_out_of_bounds_count, 1, "A sidecar range beyond EOF should fail before value decoding.");

const cardinalityFailure = analyzeOnnxModel(model(
  [], [externalTensor("w", 1, [1, 2], [["location", "weights.bin"], ["offset", "16"], ["length", "4"]])], [], [], 13,
), "external_cardinality_failure.onnx", null, { externalDataFiles: preparedExternalFiles });
expectEqual(cardinalityFailure.onnx_external_data.payload_size_mismatch_count, 1, "A sidecar range that disagrees with dtype x shape cardinality should fail closed.");

const pathMismatch = analyzeOnnxModel(externalModel, "external_path_mismatch.onnx", null, {
  externalDataFiles: [{ ...preparedExternalFiles[0], path: "other.bin" }],
});
expectEqual(pathMismatch.onnx_external_data.status, "not_assessed_payload_not_supplied", "Sidecar matching must use the exact normalized relative location rather than basename guessing.");
expectEqual(pathMismatch.onnx_external_data.unused_file_count, 1, "A selected but unreferenced sidecar should be reported as unused.");
const dotPrefixedExternalModel = model(
  [node("Add", "external_add", ["x", "w"], ["y"])],
  [externalTensor("w", 1, [1, 2], [["location", "./weights.bin"], ["offset", "16"], ["length", "8"], ["checksum", expectedExternalSha1]])],
  [valueInfo("x", 1, [1, 2])], [valueInfo("y", 1, [1, 2])], 13,
);
const dotPrefixedExternal = analyzeOnnxModel(dotPrefixedExternalModel, "external_dot_prefix.onnx", null, { externalDataFiles: preparedExternalFiles });
expectEqual(dotPrefixedExternal.onnx_external_data.status, "verified_payloads", "A leading ./ in an ONNX location should resolve through one canonical model-relative path.");
expectEqual(dotPrefixedExternal.onnx_external_data.tensors[0].normalized_location, "weights.bin", "External-data evidence should preserve a canonical model-relative location separately from the declaration.");
const emptySegmentExternal = analyzeOnnxModel(model(
  [], [externalTensor("w", 1, [1, 2], [["location", "weights//part.bin"], ["offset", "0"], ["length", "8"]])], [], [], 13,
), "external_empty_segment.onnx");
expectEqual(emptySegmentExternal.onnx_external_data.status, "malformed_reference", "Empty external-data path segments should fail closed instead of being normalized ambiguously.");
expectEqual(emptySegmentExternal.onnx_external_data.tensors[0].location_status, "unsafe_noncanonical_segment", "Non-canonical path segments should have an explicit reason code.");
expectThrows(() => analyzeOnnxModel(externalModel, "external_duplicate_selection.onnx", null, {
  externalDataFiles: [preparedExternalFiles[0], { ...preparedExternalFiles[0] }],
}), "Duplicate supplied ONNX external data path", "Duplicate selected sidecar paths should be rejected before graph analysis.");

const malformedExternal = analyzeOnnxModel(model(
  [],
  [externalTensor("w", 1, [1], [["location", "../weights.bin"], ["location", "duplicate.bin"], ["offset", "-1"]])],
  [],
  [],
  13,
), "external_data_malformed.onnx");
expectEqual(malformedExternal.onnx_external_data.status, "malformed_reference", "Unsafe, duplicate, or invalid external_data fields should fail closed.");
expectEqual(malformedExternal.onnx_external_data.malformed_reference_count, 1, "Malformed external reference count should be exact.");
expectEqual(malformedExternal.onnx_external_data.unsafe_location_count, 1, "Path traversal should be counted as one unsafe location.");
expectEqual(malformedExternal.onnx_external_data.duplicate_key_count, 1, "Duplicate external_data keys should be counted exactly.");
expectEqual(malformedExternal.onnx_external_data.invalid_range_count, 1, "Invalid decimal offsets should be counted exactly.");

done("ONNX quantization contracts passed (QLinearConv, Q/DQ, GraphProto TensorAnnotation, external-data, domain isolation, scalar, grid, bias, and accumulator cases).");

function focusedEvidence(analysis) {
  return {
    total_macs: analysis.total_macs,
    binding: analysis.onnx_quantization_binding,
    tensors: analysis.tensors.map((tensor) => ({ name: tensor.name, dtype: tensor.dtype, scales: tensor.scale_sample, zero_points: tensor.zero_point_sample })),
    grid: analysis.weight_integrity.quant_grid_details,
  };
}

function qlinearConvModel({ weightScales = [0.25, 0.5] } = {}) {
  const initializers = [
    tensor("x_scale", 1, [], float32([0.5])), tensor("x_zp", 2, [], new Uint8Array([128])),
    tensor("w", 3, [2, 1, 1, 1], new Uint8Array([128, 127])),
    tensor("w_scale", 1, [weightScales.length], float32(weightScales)), tensor("w_zp", 3, [weightScales.length], new Uint8Array(weightScales.length)),
    tensor("y_scale", 1, [], float32([0.75])), tensor("y_zp", 2, [], new Uint8Array([100])),
    tensor("bias", 6, [2], int32([4, -4])),
  ];
  const nodeBytes = node("QLinearConv", "qconv", ["x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "bias"], ["y"]);
  return model([nodeBytes], initializers, [valueInfo("x", 2, [1, 1, 2, 2])], [valueInfo("y", 2, [1, 2, 2, 2])], 13);
}

function qdqModel() {
  const initializers = [tensor("s", 1, [], float32([0.25])), tensor("zp", 2, [], new Uint8Array([7]))];
  return model([
    node("QuantizeLinear", "quant", ["x", "s", "zp"], ["q"]),
    node("DequantizeLinear", "dequant", ["q", "s", "zp"], ["y"]),
  ], initializers, [valueInfo("x", 1, [1, 4])], [valueInfo("y", 1, [1, 4])], 13);
}

function modelWithFunctions(nodes, functions, opset) {
  const graph = graphProto(nodes, [], [], [], "deepbom_scoped_external_fixture");
  const opsetImport = message([stringField(1, ""), varintField(2, opset)]);
  return message([
    varintField(1, 8), stringField(2, "deepbom_scoped_external_fixture"), bytesField(7, graph), bytesField(8, opsetImport),
    ...functions.map((value) => bytesField(25, value)),
  ]);
}

function functionProtoWithExternalDefault(name, tensorValue) {
  const opsetImport = message([stringField(1, ""), varintField(2, 13)]);
  const defaultAttribute = message([stringField(1, "weight"), bytesField(5, tensorValue), varintField(20, 4)]);
  return message([
    stringField(1, name), stringField(4, "X"), stringField(5, "Y"), stringField(6, "weight"),
    bytesField(7, node("Identity", "identity", ["X"], ["Y"])), bytesField(9, opsetImport),
    stringField(10, "local.deepbom"), bytesField(11, defaultAttribute),
  ]);
}
