import * as ort from "onnxruntime-node";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { validateSvmRowAgainstEvidence } from "../web/lib/onnx-ml-svm-conformance.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("ONNX-ML SVM check");
const cpuOptions = { executionProviders: ["cpu"], graphOptimizationLevel: "disabled", logSeverityLevel: 4 };

const linearClassifierBytes = svmClassifierModel({
  input: numericTensor("scores", 1, [2, 2], [1, 2, 3, 1]),
  labels: [10n, 20n],
  attributes: [
    floatListAttribute("coefficients", [1, 0, 0, 1]),
    floatListAttribute("rho", [0]),
  ],
  scoreShape: [2, 2],
});
const linearClassifier = analyzeOnnxModel(linearClassifierBytes, "svm_classifier_linear.onnx");
const linearClassifierRow = svmRow(linearClassifier);
expect(validateSvmRowAgainstEvidence(linearClassifierRow, linearClassifier.tensors, linearClassifier.ops), "Linear SVMClassifier row must independently reconstruct from public tensor/op evidence.");
expectEqual(linearClassifierRow.status, "pass", "Complete linear SVMClassifier must pass the static contract.");
expectEqual(linearClassifierRow.svm_mode, "linear", "Empty vectors_per_class must select pinned ORT linear mode.");
expectEqual(linearClassifierRow.svm_used_coefficient_count, 4, "Linear classifier coefficient use must equal classes multiplied by features.");
expectEqual(JSON.stringify(linearClassifierRow.svm_reference_raw_score_preview), '["1","2","3","1"]', "Scalar reference must preserve linear raw-score order.");
expectEqual(JSON.stringify(linearClassifierRow.svm_reference_label_preview), '["20","20"]', "Pinned all-positive binary selection must use the positive label at weight >= 0.5.");
const linearClassifierNative = await runNative(linearClassifierBytes);
expectEqual(JSON.stringify([...linearClassifierNative.labels.data].map(String)), '["20","20"]', "Native ORT must confirm linear classifier labels.");
expectEqual(JSON.stringify([...linearClassifierNative.scores_out.data]), "[1,2,3,1]", "Native ORT must confirm linear classifier scores.");
expect(validateSvmRowAgainstEvidence(linearClassifierRow, linearClassifier.tensors, linearClassifier.ops), "Native confirmation must not mutate the separately parsed static evidence.");
const linearBundle = bundle(linearClassifier);
expectEqual(linearBundle.conformance.status, "pass", "Linear SVMClassifier evidence must pass independent bundle conformance.");
expect(linearBundle.report.includes("SVM support-vector/runtime contracts")
  && linearBundle.report.includes("support/coefficients/rho expected/used/serialized"), "Engineering Report must expose SVM aggregate and row-level parameter conservation.");
assertCompactMlBomProjection(linearBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxMlExactSvmUsedCoefficients"],
  label: "SVM compact ML-BOM",
});

const pairwiseBytes = svmClassifierModel({
  input: numericTensor("scores", 1, [1, 1], [1]),
  labels: [0n, 1n, 2n, 3n],
  attributes: [
    intListAttribute("vectors_per_class", [1n, 1n, 1n, 1n]),
    floatListAttribute("support_vectors", [1, 2, 3, 4]),
    floatListAttribute("coefficients", new Array(12).fill(1)),
    floatListAttribute("rho", new Array(6).fill(0)),
  ],
  scoreShape: [1, 6],
});
const pairwise = analyzeOnnxModel(pairwiseBytes, "svm_classifier_pairwise_width.onnx");
const pairwiseRow = svmRow(pairwise);
expectEqual(pairwiseRow.svm_mode, "svc", "Positive vectors_per_class sum must select SVC mode.");
expectEqual(pairwiseRow.svm_schema_score_width, 4, "ONNX class-score width must equal class count.");
expectEqual(pairwiseRow.svm_pinned_ort_score_width, 6, "Pinned ORT raw SVC score width must equal pair count.");
expectEqual(pairwiseRow.svm_schema_runtime_score_width_mismatch, true, "The 4-class raw-score width mismatch must be explicit.");
expectEqual(pairwiseRow.status, "fail", "Conflicting schema/runtime score shapes must suppress inferred output propagation.");
const pairwiseNative = await runNative(pairwiseBytes);
expectEqual(pairwiseNative.scores_out.dims[1], 6, "Native ORT must confirm pairwise score width six.");
expectEqual(buildFindingsRegister(pairwise).find((finding) => finding.finding_id === "EA-ONX-0053")?.technical_priority, "High", "Schema/runtime score-width conflicts must enter the High action queue.");
expectEqual(bundle(pairwise).conformance.status, "pass", "Faithfully reported score-width conflicts must pass independent conformance.");

const regressorBytes = svmRegressorModel({
  input: numericTensor("scores", 1, [2, 2], [1, 2, 3, 1]),
  attributes: [
    floatListAttribute("coefficients", [1, 2]),
    floatListAttribute("rho", [0.5]),
    stringAttribute("post_transform", "LOGISTIC"),
  ],
  outputShape: [2, 1],
});
const regressor = analyzeOnnxModel(regressorBytes, "svm_regressor_ignored_transform.onnx");
const regressorRow = svmRow(regressor);
expectEqual(regressorRow.status, "pass", "Complete FLOAT32 SVMRegressor must pass the static contract.");
expectEqual(regressorRow.svm_post_transform_applied_by_pinned_ort, false, "Pinned ORT regressor must disclose that post_transform is not applied.");
expect(regressorRow.risk_codes.includes("svm_regressor_post_transform_ignored_by_pinned_ort"), "Ignored regressor post_transform must retain a risk code.");
expectEqual(JSON.stringify(regressorRow.svm_reference_output_score_preview), '["5.5","5.5"]', "Regressor reference must preserve raw output instead of applying LOGISTIC.");
const regressorNative = await runNative(regressorBytes);
expectEqual(JSON.stringify([...regressorNative.predictions.data]), "[5.5,5.5]", "Native ORT must confirm ignored LOGISTIC transform.");
expectEqual(buildFindingsRegister(regressor).find((finding) => finding.finding_id === "EA-ONX-0055")?.technical_priority, "High", "Ignored SVMRegressor transforms must enter the High action queue.");
const regressorBundle = bundle(regressor);
expectEqual(regressorBundle.conformance.status, "pass", "Ignored-transform evidence must pass independent conformance.");
assertCompactMlBomProjection(regressorBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxMlSvmIgnoredPostTransforms"],
  label: "SVMRegressor compact ML-BOM",
});

const float64Bytes = svmRegressorModel({
  input: numericTensor("scores", 11, [1, 2], [1, 2]),
  attributes: [floatListAttribute("coefficients", [1, 2]), floatListAttribute("rho", [0])],
  outputShape: [1, 1],
});
const float64 = analyzeOnnxModel(float64Bytes, "svm_regressor_float64_gap.onnx");
const float64Row = svmRow(float64);
expectEqual(float64Row.status, "partial", "Schema-valid FLOAT64 regressor must retain a separate runtime gap rather than an ONNX failure.");
expect(float64Row.risk_codes.includes("svm_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"), "FLOAT64 CPU kernel gap must be explicit.");
expectEqual(buildFindingsRegister(float64).find((finding) => finding.finding_id === "EA-ONX-0054")?.technical_priority, "High", "SVMRegressor CPU dtype gaps must enter the High action queue.");
expectEqual(bundle(float64).conformance.status, "pass", "Faithfully separated SVMRegressor dtype gaps must pass independent conformance.");
expect(await nativeRejects(float64Bytes), "Native ORT CPU must reject schema-valid FLOAT64 SVMRegressor.");

const malformedBytes = svmRegressorModel({
  input: numericTensor("scores", 1, [1, 2], [1, 2]),
  attributes: [
    intAttribute("n_supports", 2n), floatListAttribute("support_vectors", [1, 2, 3]),
    floatListAttribute("coefficients", [1, 1]), floatListAttribute("rho", [0]),
  ],
  outputShape: [1, 1],
});
const malformed = analyzeOnnxModel(malformedBytes, "svm_regressor_bad_support_layout.onnx");
expectEqual(svmRow(malformed).status, "fail", "Non-divisible support-vector layout must fail deterministically.");
expectEqual(buildFindingsRegister(malformed).find((finding) => finding.finding_id === "EA-ONX-0052")?.technical_priority, "High", "Malformed executable SVM layouts must enter the High action queue.");
expectEqual(bundle(malformed).conformance.status, "pass", "Faithfully reported malformed SVM evidence must pass independent conformance.");
expect(await nativeRejects(malformedBytes), "Native ORT must reject the same non-divisible support-vector layout.");

const omittedKernelParamsBytes = svmClassifierModel({
  input: numericTensor("scores", 1, [1, 2], [1, 2]),
  labels: [0n, 1n],
  attributes: [floatListAttribute("coefficients", [1, 0, 0, 1]), floatListAttribute("rho", [0])],
  scoreShape: [1, 2],
  includeKernelParams: false,
});
const omittedKernelParams = analyzeOnnxModel(omittedKernelParamsBytes, "svm_classifier_omitted_kernel_params.onnx");
const omittedRow = svmRow(omittedKernelParams);
expectEqual(omittedRow.svm_onnx_contract_status, "pass", "ONNX schema must permit omitted optional kernel_params.");
expectEqual(omittedRow.svm_pinned_ort_contract_status, "fail", "Pinned ORT constructor must separately reject omitted kernel_params.");
expectEqual(omittedRow.svm_pinned_ort_contract_reason, "kernel_params_attribute_missing_in_pinned_ort_constructor", "Omitted kernel_params rejection must retain its exact source-backed reason.");
expectEqual(bundle(omittedKernelParams).conformance.status, "pass", "The ONNX/ORT optional-attribute gap must pass faithful export conformance.");
expect(await nativeRejects(omittedKernelParamsBytes), "Native ORT must reject omitted kernel_params at initialization.");

const tampered = structuredClone(linearClassifier);
const tamperedMl = tampered.onnx_shape_inference.ml_value_inference;
const tamperedRow = svmRow(tampered);
tamperedRow.svm_used_coefficient_count = 3;
tamperedRow.svm_unused_coefficient_count = 1;
tamperedRow.risk_codes.push("svm_classifier_serialized_parameters_ignored_by_pinned_ort");
tamperedMl.exact_svm_used_coefficient_count = 3;
tamperedMl.exact_svm_unused_coefficient_count = 1;
tamperedMl.svm_ignored_parameter_node_count = 1;
expectThrows(() => bundle(tampered), "CF-SHAPE-ML-ROW-002", "Independent evidence reconstruction must reject forged but aggregate-conserving SVM parameter-use accounting.");

done("ONNX-ML SVM schema/runtime shape, arithmetic reference, findings, report, ML-BOM, malformed, native ORT, and tamper checks passed.");

function svmRow(analysis) {
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

function svmClassifierModel({ input, labels, attributes, scoreShape, includeKernelParams = true }) {
  const node = message([
    stringField(1, "scores"), stringField(2, "labels"), stringField(2, "scores_out"),
    stringField(3, "svm_classifier"), stringField(4, "SVMClassifier"),
    ...[intListAttribute("classlabels_ints", labels), ...(includeKernelParams ? [floatListAttribute("kernel_params", [])] : []), ...attributes].map((attribute) => bytesField(5, attribute)),
    stringField(7, "ai.onnx.ml"),
  ]);
  const graph = message([
    bytesField(1, node), stringField(2, "svm_classifier_graph"), bytesField(5, input),
    bytesField(12, valueInfo("labels", 7, [input._shape.length === 1 ? 1 : input._shape[0]])),
    bytesField(12, valueInfo("scores_out", 1, scoreShape)),
  ]);
  return model(graph);
}

function svmRegressorModel({ input, attributes, outputShape }) {
  const node = message([
    stringField(1, "scores"), stringField(2, "predictions"), stringField(3, "svm_regressor"),
    stringField(4, "SVMRegressor"), ...[floatListAttribute("kernel_params", []), ...attributes].map((attribute) => bytesField(5, attribute)),
    stringField(7, "ai.onnx.ml"),
  ]);
  const graph = message([
    bytesField(1, node), stringField(2, "svm_regressor_graph"), bytesField(5, input),
    bytesField(12, valueInfo("predictions", 1, outputShape)),
  ]);
  return model(graph);
}

function model(graph) {
  const opset = message([stringField(1, "ai.onnx.ml"), varintField(2, 1)]);
  return message([varintField(1, 8), stringField(2, "deepbom_svm_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function bundle(analysis) {
  const mlBom = buildMlBomDocument(analysis, { hash: "" });
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
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

function valueInfo(name, dtype, shape) {
  const dimensions = shape.map((dimension) => bytesField(1, message([varintField(1, dimension)])));
  const tensorType = message([varintField(1, dtype), bytesField(2, message(dimensions))]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

function numericTensor(name, dtype, shape, values) {
  const raw = numericBytes(dtype, values);
  const tensor = message([
    ...shape.map((dimension) => varintField(1, dimension)), varintField(2, dtype), stringField(8, name), bytesField(9, raw),
  ]);
  Object.defineProperty(tensor, "_shape", { value: shape, enumerable: false });
  return tensor;
}

function numericBytes(dtype, values) {
  const width = dtype === 11 ? 8 : 4;
  const bytes = new Uint8Array(values.length * width);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    if (dtype === 11) view.setFloat64(index * width, Number(value), true);
    else view.setFloat32(index * width, Number(value), true);
  });
  return bytes;
}

function stringAttribute(name, value) {
  return message([stringField(1, name), stringField(4, value), varintField(20, 3)]);
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
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}
