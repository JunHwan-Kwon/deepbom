import * as ort from "onnxruntime-node";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { validateTreeEnsembleRowAgainstEvidence } from "../web/lib/onnx-ml-tree-ensemble-conformance.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectDeepEqual, expectThrows } = createCheck("ONNX-ML TreeEnsemble check");
const cpuOptions = { executionProviders: ["cpu"], graphOptimizationLevel: "disabled", logSeverityLevel: 4 };

const regressorBytes = legacyRegressorModel();
const regressor = analyzeOnnxModel(regressorBytes, "tree_regressor_v3.onnx");
const regressorRow = treeRow(regressor);
expect(validateTreeEnsembleRowAgainstEvidence(regressorRow, regressor.tensors, regressor.ops), "Legacy regressor row must independently reconstruct from public tensor/op evidence.");
expectEqual(regressorRow.op_name, "TreeEnsembleRegressor", "Legacy regressor must enter the ONNX-ML value engine.");
expectEqual(regressorRow.resolved_schema_version, 3, "Legacy regressor must resolve the greatest schema not exceeding opset 3.");
expectEqual(regressorRow.tree_exact_tree_count, 1, "Legacy regressor tree count must be exact.");
expectEqual(regressorRow.tree_exact_node_count, 3, "Legacy regressor node count must be exact.");
expectEqual(regressorRow.tree_reference_assessment_status, "assessed_scalar_reference", "Static legacy regressor input must produce a bounded source-order reference.");
expectDeepEqual(regressorRow.tree_reference_output_score_preview, ["2", "5"], "Legacy regressor leaf scores must match the serialized tree.");
const regressorNative = await runNative(regressorBytes);
expectDeepEqual([...regressorNative.predictions.data], [2, 5], "Pinned native ORT CPU must agree with the legacy regressor fixture.");
const regressorBundle = bundle(regressor);
expectEqual(regressorBundle.conformance.status, "pass", "Legacy regressor evidence must pass independent bundle conformance.");
expect(regressorBundle.report.includes("TreeEnsemble topology/runtime contracts")
  && regressorBundle.report.includes("weight conservation")
  && regressorBundle.report.includes("reference values are not claimed runtime-bit-exact"), "Engineering Report must expose TreeEnsemble topology, weight conservation, and reference boundaries.");
assertCompactMlBomProjection(regressorBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxMlExactTreeEnsembleUsedWeights"],
  label: "TreeEnsemble compact ML-BOM",
});

const classifierBytes = legacyClassifierModel();
const classifier = analyzeOnnxModel(classifierBytes, "tree_classifier_v3.onnx");
const classifierRow = treeRow(classifier);
expect(validateTreeEnsembleRowAgainstEvidence(classifierRow, classifier.tensors, classifier.ops), "Legacy classifier row must independently reconstruct from public tensor/op evidence.");
expectEqual(classifierRow.tree_class_label_count, 2, "Legacy classifier label cardinality must be exact.");
expectEqual(classifierRow.tree_weight_tuple_count, 2, "Legacy classifier vote cardinality must be exact.");
expect(classifierRow.risk_codes.includes("tree_classifier_pinned_ort_binary_label_index_semantics"), "Custom INT64 labels with two represented classes must expose pinned ORT binary index semantics.");
const classifierNative = await runNative(classifierBytes);
expectDeepEqual(classifierRow.tree_reference_label_preview, [...classifierNative.labels.data].map(String), "Classifier reference labels must reproduce pinned ORT CPU semantics.");
expectDeepEqual(classifierRow.tree_reference_output_score_preview, [...classifierNative.scores.data].map(String), "Classifier reference scores must reproduce the fixture's pinned ORT CPU result.");
expectEqual(buildFindingsRegister(classifier).find((finding) => finding.finding_id === "EA-ONX-0064")?.technical_priority, "High", "Binary classifier label semantics must enter the High action queue.");
expectEqual(bundle(classifier).conformance.status, "pass", "Classifier semantic-hazard evidence must pass bundle conformance.");

const genericBytes = genericTreeModel();
const generic = analyzeOnnxModel(genericBytes, "tree_ensemble_v5_member.onnx");
const genericRow = treeRow(generic);
expect(validateTreeEnsembleRowAgainstEvidence(genericRow, generic.tensors, generic.ops), "Generic TreeEnsemble row must independently reconstruct from public tensor/op evidence.");
expectEqual(genericRow.op_name, "TreeEnsemble", "Generic TreeEnsemble-5 must enter the ONNX-ML value engine.");
expectEqual(genericRow.tree_membership_node_count, 1, "BRANCH_MEMBER node count must be exact.");
expectEqual(genericRow.tree_membership_value_count, 2, "Membership values must exclude the NaN delimiter.");
expectEqual(genericRow.tree_membership_separator_count, 1, "Membership delimiter count must be exact.");
expect(genericRow.risk_codes.includes("tree_ensemble_v5_zero_member_differs_from_pinned_onnx_reference_parser"), "A zero member must expose the pinned ONNX-reference parser divergence.");
expectDeepEqual(genericRow.tree_reference_output_score_preview, ["10", "-1", "10"], "Generic membership paths must be evaluated exactly in source order.");
const genericNative = await runNative(genericBytes);
expectDeepEqual([...genericNative.Y.data], [10, -1, 10], "Pinned native ORT CPU must agree with generic membership evaluation.");
expectEqual(buildFindingsRegister(generic).find((finding) => finding.finding_id === "EA-ONX-0063")?.technical_priority, "Medium", "Zero-member source divergence must enter the action queue.");
expectEqual(bundle(generic).conformance.status, "pass", "Generic MEMBER evidence must pass bundle conformance.");

const malformed = analyzeOnnxModel(genericTreeModel({ badChild: true }), "tree_ensemble_v5_bad_child.onnx");
const malformedRow = treeRow(malformed);
expect(validateTreeEnsembleRowAgainstEvidence(malformedRow, malformed.tensors, malformed.ops), "Malformed generic evidence must still independently reconstruct as a faithful failure.");
expectEqual(malformedRow.status, "fail", "Out-of-range generic child indices must fail closed.");
expect(malformedRow.reason_codes.includes("tree_ensemble_v5_invalid_child_reference"), "Malformed child evidence must retain the exact failure reason.");
expectEqual(buildFindingsRegister(malformed).find((finding) => finding.finding_id === "EA-ONX-0059")?.technical_priority, "High", "Malformed TreeEnsemble topology must enter the High action queue.");
expectEqual(bundle(malformed).conformance.status, "pass", "Faithfully reported malformed TreeEnsemble evidence must pass bundle conformance.");

const legacyV1 = analyzeOnnxModel(legacyRegressorModel({ opset: 1, inputShape: [1], inputValues: [2] }), "tree_regressor_v1_rank1.onnx");
expectEqual(treeRow(legacyV1).resolved_schema_version, 1, "Legacy opset 1 must resolve TreeEnsembleRegressor-1.");
expectDeepEqual(treeRow(legacyV1).exact_output_shape, [1, 1], "Version 1 rank-one input must deterministically produce one batch row.");
expectEqual(bundle(legacyV1).conformance.status, "pass", "Version 1 rank-one evidence must pass bundle conformance.");

const legacyV5 = analyzeOnnxModel(legacyRegressorModel({ opset: 5 }), "tree_regressor_v5_deprecated.onnx");
expectEqual(treeRow(legacyV5).tree_deprecated_operator, true, "Legacy TreeEnsembleRegressor-5 must retain the source-pinned deprecation marker.");
expectEqual(buildFindingsRegister(legacyV5).find((finding) => finding.finding_id === "EA-ONX-0061")?.technical_priority, "Medium", "Deprecated legacy trees must enter the action queue.");
expectEqual(bundle(legacyV5).conformance.status, "pass", "Deprecated-but-valid legacy evidence must pass bundle conformance.");

const legacyTensor64Bytes = legacyRegressorModel({ inputDtype: 11, tensorAttributes: true });
const legacyTensor64 = analyzeOnnxModel(legacyTensor64Bytes, "tree_regressor_v3_tensor_attrs_float64.onnx");
expectEqual(treeRow(legacyTensor64).status, "pass", "Version 3 FLOAT64 tensor attributes must satisfy the legacy schema/runtime contract.");
expectDeepEqual(treeRow(legacyTensor64).tree_reference_output_score_preview, ["2", "5"], "FLOAT64 tensor attributes must preserve exact tree path scores.");
expectDeepEqual([...(await runNative(legacyTensor64Bytes)).predictions.data], [2, 5], "Pinned native ORT must execute version 3 FLOAT64 tensor attributes.");
expectEqual(bundle(legacyTensor64).conformance.status, "pass", "FLOAT64 tensor-attribute evidence must pass bundle conformance.");

const legacyInt32Bytes = legacyRegressorModel({ inputDtype: 6 });
const legacyInt32 = analyzeOnnxModel(legacyInt32Bytes, "tree_regressor_v3_int32_cpu_gap.onnx");
expectEqual(treeRow(legacyInt32).tree_onnx_contract_status, "pass", "INT32 legacy regressor must remain schema-valid.");
expectEqual(treeRow(legacyInt32).tree_pinned_cpu_dtype_gap, true, "INT32 legacy regressor must expose the pinned ORT CPU registration gap.");
expectEqual(buildFindingsRegister(legacyInt32).find((finding) => finding.finding_id === "EA-ONX-0060")?.technical_priority, "High", "Legacy INT32 CPU gaps must enter the High action queue.");
expect(await nativeRejects(legacyInt32Bytes), "Pinned native ORT CPU must reject the schema-valid INT32 legacy regressor.");

const genericFloat16Bytes = genericTreeModel({ inputDtype: 10 });
const genericFloat16 = analyzeOnnxModel(genericFloat16Bytes, "tree_ensemble_v5_float16_cpu_gap.onnx");
expectEqual(treeRow(genericFloat16).tree_onnx_contract_status, "pass", "FLOAT16 generic TreeEnsemble-5 must remain schema-valid.");
expectEqual(treeRow(genericFloat16).tree_pinned_cpu_dtype_gap, true, "FLOAT16 generic tree must expose the pinned ORT CPU registration gap.");
expectEqual(bundle(genericFloat16).conformance.status, "pass", "FLOAT16 CPU-gap evidence must pass bundle conformance.");
expect(await nativeRejects(genericFloat16Bytes), "Pinned native ORT CPU must reject generic FLOAT16 TreeEnsemble-5.");

const cycle = analyzeOnnxModel(genericTreeModel({ cycle: true }), "tree_ensemble_v5_cycle.onnx");
expectEqual(treeRow(cycle).status, "fail", "A reachable generic tree cycle must fail closed.");
expect(treeRow(cycle).reason_codes.includes("tree_ensemble_v5_cycle_detected"), "Cycle evidence must retain its exact failure reason.");
expectEqual(bundle(cycle).conformance.status, "pass", "Faithfully reported cycle evidence must pass bundle conformance.");

const orphan = analyzeOnnxModel(genericTreeModel({ orphanLeaf: true }), "tree_ensemble_v5_orphan_leaf.onnx");
expectEqual(treeRow(orphan).tree_orphan_node_or_leaf_count, 1, "An unreferenced generic leaf must be counted exactly.");
expectEqual(treeRow(orphan).tree_unused_weight_count, 1, "An unreferenced generic leaf weight must be counted as unused.");
expectEqual(buildFindingsRegister(orphan).find((finding) => finding.finding_id === "EA-ONX-0062")?.technical_priority, "Medium", "Unreachable serialized tree state must enter the action queue.");
expectEqual(bundle(orphan).conformance.status, "pass", "Orphan-leaf evidence must pass bundle conformance.");

const missingDelimiter = analyzeOnnxModel(genericTreeModel({ missingMembershipTerminator: true }), "tree_ensemble_v5_missing_member_delimiter.onnx");
expectEqual(treeRow(missingDelimiter).status, "fail", "An unterminated MEMBER set must fail closed.");
expect(treeRow(missingDelimiter).reason_codes.includes("tree_ensemble_v5_membership_values_missing_nan_terminator"), "MEMBER delimiter failure must retain its exact reason.");
expectEqual(bundle(missingDelimiter).conformance.status, "pass", "Faithfully reported MEMBER delimiter failure must pass bundle conformance.");

const tampered = structuredClone(regressor);
const tamperedMl = tampered.onnx_shape_inference.ml_value_inference;
const tamperedRow = treeRow(tampered);
tamperedRow.tree_used_weight_count = 1;
tamperedRow.tree_unused_weight_count = 1;
tamperedMl.exact_tree_ensemble_used_weight_count = 1;
tamperedMl.exact_tree_ensemble_unused_weight_count = 1;
expectThrows(() => bundle(tampered), "CF-SHAPE-ML-ROW-002", "Independent evidence reconstruction must reject forged but aggregate-conserving TreeEnsemble weight accounting.");

done("ONNX-ML TreeEnsemble legacy/generic topology, arithmetic, membership, malformed, and native ORT checks passed.");

function treeRow(analysis) {
  return analysis.onnx_shape_inference.ml_value_inference.rows[0];
}

async function runNative(bytes) {
  const session = await ort.InferenceSession.create(bytes.slice(), cpuOptions);
  return session.run({});
}

async function nativeRejects(bytes) {
  try {
    await runNative(bytes);
    return false;
  } catch {
    return true;
  }
}

function bundle(analysis) {
  const mlBom = buildMlBomDocument(analysis, { hash: "" });
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-23T00:00:00.000Z" },
    rawEvidenceContext: { identity: { filename: analysis.filename, format: "onnx" } },
    mlBomDocument: mlBom,
  });
  const report = files.find((file) => file.name === "engineering_report.md")?.data || "";
  const evidence = JSON.parse(files.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
  return { report, mlBom, conformance: evidence.evidence?.conformance_report || {} };
}

function property(document, name) {
  return document.metadata.component.properties.find((item) => item.name === name)?.value;
}

function legacyRegressorModel({ opset = 3, inputDtype = 1, inputShape = [2, 1], inputValues = [0, 2], tensorAttributes = false } = {}) {
  const input = numericTensor("X", inputDtype, inputShape, inputValues);
  const attributes = legacyNodeAttributes("target", [2, 5], { tensorAttributes });
  attributes.push(intAttribute("n_targets", 1));
  const node = nodeProto("TreeEnsembleRegressor", ["X"], ["predictions"], attributes);
  const batch = inputShape.length === 1 ? 1 : inputShape[0];
  const graph = graphProto("tree_regressor", node, input, [valueInfo("predictions", 1, [batch, 1])]);
  return modelProto(graph, opset);
}

function legacyClassifierModel() {
  const input = numericTensor("X", 1, [2, 1], [0, 2]);
  const attributes = legacyNodeAttributes("class", [2, 3]);
  attributes.push(intListAttribute("classlabels_int64s", [10, 20]));
  const node = nodeProto("TreeEnsembleClassifier", ["X"], ["labels", "scores"], attributes);
  const graph = graphProto("tree_classifier", node, input, [
    valueInfo("labels", 7, [2]), valueInfo("scores", 1, [2, 2]),
  ]);
  return modelProto(graph, 3);
}

function legacyNodeAttributes(prefix, weights, { tensorAttributes = false } = {}) {
  return [
    intListAttribute("nodes_treeids", [0, 0, 0]),
    intListAttribute("nodes_nodeids", [0, 1, 2]),
    intListAttribute("nodes_featureids", [0, 0, 0]),
    tensorAttributes ? tensorAttribute("nodes_values_as_tensor", 11, [1, 0, 0]) : floatListAttribute("nodes_values", [1, 0, 0]),
    stringListAttribute("nodes_modes", ["BRANCH_LEQ", "LEAF", "LEAF"]),
    intListAttribute("nodes_truenodeids", [1, 0, 0]),
    intListAttribute("nodes_falsenodeids", [2, 0, 0]),
    intListAttribute(`${prefix}_treeids`, [0, 0]),
    intListAttribute(`${prefix}_nodeids`, [1, 2]),
    intListAttribute(`${prefix}_ids`, [0, prefix === "class" ? 1 : 0]),
    tensorAttributes ? tensorAttribute(`${prefix}_weights_as_tensor`, 11, weights) : floatListAttribute(`${prefix}_weights`, weights),
  ];
}

function genericTreeModel({ badChild = false, inputDtype = 1, cycle = false, orphanLeaf = false,
  missingMembershipTerminator = false } = {}) {
  const input = numericTensor("X", inputDtype, [3, 1], [0, 1, 2]);
  const membership = missingMembershipTerminator ? [0, 2] : [0, 2, Number.NaN];
  const leafTargets = orphanLeaf ? [0, 0, 0] : [0, 0];
  const leafWeights = orphanLeaf ? [10, -1, 99] : [10, -1];
  const attributes = [
    intListAttribute("nodes_featureids", [0]),
    tensorAttribute("nodes_splits", inputDtype, [0]),
    tensorAttribute("nodes_modes", 2, [6]),
    intListAttribute("nodes_truenodeids", [0]),
    intListAttribute("nodes_falsenodeids", [badChild ? 2 : 1]),
    intListAttribute("nodes_trueleafs", [cycle ? 0 : 1]),
    intListAttribute("nodes_falseleafs", [1]),
    intListAttribute("tree_roots", [0]),
    tensorAttribute("membership_values", inputDtype, membership),
    intListAttribute("leaf_targetids", leafTargets),
    tensorAttribute("leaf_weights", inputDtype, leafWeights),
    intAttribute("n_targets", 1),
  ];
  const node = nodeProto("TreeEnsemble", ["X"], ["Y"], attributes);
  const graph = graphProto("tree_ensemble", node, input, [valueInfo("Y", inputDtype, [3, 1])]);
  return modelProto(graph, 5);
}

function nodeProto(opType, inputs, outputs, attributes) {
  return message([
    ...inputs.map((name) => stringField(1, name)), ...outputs.map((name) => stringField(2, name)),
    stringField(3, opType.toLowerCase()), stringField(4, opType),
    ...attributes.map((attribute) => bytesField(5, attribute)), stringField(7, "ai.onnx.ml"),
  ]);
}

function graphProto(name, node, initializer, outputs) {
  return message([bytesField(1, node), stringField(2, name), bytesField(5, initializer), ...outputs.map((output) => bytesField(12, output))]);
}

function modelProto(graph, mlOpset) {
  const opset = message([stringField(1, "ai.onnx.ml"), varintField(2, mlOpset)]);
  return message([varintField(1, 8), stringField(2, "deepbom_tree_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function valueInfo(name, dtype, shape) {
  const dimensions = shape.map((dimension) => bytesField(1, message([varintField(1, dimension)])));
  const tensorType = message([varintField(1, dtype), bytesField(2, message(dimensions))]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

function numericTensor(name, dtype, shape, values) {
  return message([...shape.map((dimension) => varintField(1, dimension)), varintField(2, dtype), stringField(8, name), bytesField(9, numericBytes(dtype, values))]);
}

function tensorAttribute(name, dtype, values) {
  const tensor = message([varintField(1, values.length), varintField(2, dtype), bytesField(9, numericBytes(dtype, values))]);
  return message([stringField(1, name), bytesField(5, tensor), varintField(20, 4)]);
}

function numericBytes(dtype, values) {
  if (dtype === 2) return Uint8Array.from(values);
  if (dtype === 10) {
    const bytes = new Uint8Array(values.length * 2);
    const view = new DataView(bytes.buffer);
    values.forEach((value, index) => view.setUint16(index * 2, float16Bits(Number(value)), true));
    return bytes;
  }
  if (dtype === 6) {
    const bytes = new Uint8Array(values.length * 4);
    const view = new DataView(bytes.buffer);
    values.forEach((value, index) => view.setInt32(index * 4, Number(value), true));
    return bytes;
  }
  const width = dtype === 11 ? 8 : 4;
  const bytes = new Uint8Array(values.length * width);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => dtype === 11
    ? view.setFloat64(index * width, Number(value), true) : view.setFloat32(index * width, Number(value), true));
  return bytes;
}

function float16Bits(value) {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) { mantissa = 0; exponent += 1; }
  return exponent >= 31 ? sign | 0x7c00 : sign | (exponent << 10) | (mantissa >>> 13);
}

function intAttribute(name, value) {
  return message([stringField(1, name), varintField(3, value), varintField(20, 2)]);
}

function intListAttribute(name, values) {
  return message([stringField(1, name), ...values.map((value) => varintField(8, value)), varintField(20, 7)]);
}

function floatListAttribute(name, values) {
  return message([stringField(1, name), ...values.map((value) => floatField(7, value)), varintField(20, 6)]);
}

function stringListAttribute(name, values) {
  return message([stringField(1, name), ...values.map((value) => stringField(9, value)), varintField(20, 8)]);
}

function stringField(field, value) {
  return bytesField(field, new TextEncoder().encode(value));
}

function bytesField(field, value) {
  return message([varint((field << 3) | 2), varint(value.length), value]);
}

function varintField(field, value) {
  return message([varint(field << 3), varint(value)]);
}

function floatField(field, value) {
  const bytes = new Uint8Array(5);
  bytes[0] = (field << 3) | 5;
  new DataView(bytes.buffer).setFloat32(1, Number(value), true);
  return bytes;
}

function varint(value) {
  let remaining = BigInt(value);
  if (remaining < 0n) remaining = BigInt.asUintN(64, remaining);
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
  const output = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { output.set(part, offset); offset += part.length; }
  return output;
}
