import { File } from "node:buffer";
import { createHash } from "node:crypto";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { deriveCurrentArtifactCapabilityRow } from "../web/lib/format-capability-view.js";
import { runtimeCapturePlanAvailability } from "../web/lib/runtime-evidence-closure.js";

function assert(value, message) { if (!value) throw new Error(message); }
function varint(value) {
  let current = BigInt(value); const bytes = [];
  while (current > 0x7fn) { bytes.push(Number(current & 0x7fn) | 0x80); current >>= 7n; }
  bytes.push(Number(current)); return Buffer.from(bytes);
}
function concat(...values) { return Buffer.concat(values.flat(Infinity).filter((value) => value != null)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function double(field, value) { const body = Buffer.alloc(8); body.writeDoubleLE(value); return concat(key(field, 1), body); }
function packedDouble(field, values) { const body = Buffer.alloc(values.length * 8); values.forEach((value, index) => body.writeDoubleLE(value, index * 8)); return bytes(field, body); }
function packedUint(field, values) { const body = concat(values.map(varint)); return bytes(field, body); }

function arrayType(shape) { return message(5, concat(packedUint(1, shape), uint(2, 65600))); }
function doubleType() { return message(2, Buffer.alloc(0)); }
function stringType() { return message(3, Buffer.alloc(0)); }
function stringDictionaryType() { return message(6, message(2, Buffer.alloc(0))); }
function feature(name, type) { return concat(string(1, name), message(3, type)); }
function description({ classifier = false } = {}) {
  return concat(
    message(1, feature("features", arrayType([3]))),
    classifier
      ? [message(10, feature("classLabel", stringType())), message(10, feature("classProbability", stringDictionaryType())), string(11, "classLabel"), string(12, "classProbability")]
      : message(10, feature("prediction", doubleType())),
  );
}
function model(field, payload, options = {}) { return concat(uint(1, 1), message(2, description(options)), message(field, payload)); }
function modelWithDescription(field, payload, modelDescription, updatable = false) {
  return concat(uint(1, 1), message(2, modelDescription), updatable ? uint(10, 1) : null, message(field, payload));
}
function regressionDescription(inputName, inputType, outputName) {
  return concat(message(1, feature(inputName, inputType)), message(10, feature(outputName, doubleType())));
}
function doubleArray(values) { return packedDouble(1, values); }
function stringLabels(values) { return message(100, concat(values.map((value) => string(1, value)))); }
function denseSupportVectors(rows) { return concat(rows.map((row) => message(1, packedDouble(1, row)))); }
function coefficients(values) { return packedDouble(1, values); }

async function analyze(payload, name) {
  const analysis = (await readCoreMlModelFile(new File([payload], name))).analysis;
  analysis.model_sha256 = createHash("sha256").update(payload).digest("hex");
  return analysis;
}
async function rejects(payload, pattern, label) {
  try { await analyze(payload, `${label}.mlmodel`); }
  catch (error) { if (pattern.test(String(error?.message))) return; throw new Error(`${label}: unexpected error ${error?.message}`); }
  throw new Error(`${label}: expected rejection`);
}

function releaseConformance(analysis) {
  const identity = {
    filename: analysis.filename,
    format: "coreml",
    sha256: analysis.model_sha256,
  };
  const generatedAt = "2026-08-12T00:00:00.000Z";
  return buildEngineeringEvidenceDocument(analysis, {
    reportContext: { identity, generatedAt },
    rawEvidenceContext: { identity, generatedAt },
  }).evidence.conformance_report;
}

const glmRegressorPayload = concat(message(1, doubleArray([1, 2, 3])), packedDouble(2, [0.5]), uint(3, 0));
const glmRegressor = await analyze(model(300, glmRegressorPayload), "glm-regressor.mlmodel");
assert(glmRegressor.coreml.classical_model.kind === "glmRegressor" && glmRegressor.total_macs === 3, "GLM regressor graph or exact dot-product MACs are incorrect");
assert(glmRegressor.weight_integrity.status === "assessed" && glmRegressor.weight_integrity.payload_bytes === 32, "GLM regressor numerical byte conservation is incorrect");

const glmClassifierPayload = concat(message(1, doubleArray([1, 2, 3])), packedDouble(2, [0.25]), uint(3, 0), uint(4, 0), stringLabels(["negative", "positive"]));
const glmClassifier = await analyze(model(400, glmClassifierPayload, { classifier: true }), "glm-classifier.mlmodel");
assert(glmClassifier.total_macs === 3 && glmClassifier.coreml.classical_model.class_labels.values.length === 2, "GLM classifier class binding or MACs are incorrect");

const linearKernel = message(1, Buffer.alloc(0));
const denseVectors = denseSupportVectors([[1, 0, 0], [0, 1, 0]]);
const svmRegressorPayload = concat(message(1, linearKernel), message(3, denseVectors), message(4, coefficients([0.5, -0.5])), double(5, 0.1));
const svmRegressor = await analyze(model(301, svmRegressorPayload), "svm-regressor.mlmodel");
assert(svmRegressor.total_macs === 8 && svmRegressor.coreml.classical_model.support_vectors.count === 2, "SVM regressor support-vector arithmetic is incorrect");

const svmClassifierPayload = concat(
  message(1, linearKernel), packedUint(2, [1, 1]), message(4, denseVectors), message(5, coefficients([0.5, -0.5])),
  double(6, 0.1), stringLabels(["negative", "positive"]),
);
const svmClassifier = await analyze(model(401, svmClassifierPayload, { classifier: true }), "svm-classifier.mlmodel");
assert(svmClassifier.total_macs === 8 && svmClassifier.coreml.classical_model.coefficient_row_count === 1, "SVM classifier class-pair arithmetic is incorrect");

function evaluation(index, value) { return concat(uint(1, index), double(2, value)); }
function branchNode({ trueChild = 1, falseChild = 2 } = {}) { return concat(uint(3, 0), uint(10, 0), double(11, 0.5), uint(12, trueChild), uint(13, falseChild)); }
function leafNode(id, value) { return concat(uint(2, id), uint(3, 6), message(20, evaluation(0, value))); }
function treeParameters(nodes = [branchNode(), leafNode(1, -1), leafNode(2, 1)]) {
  return concat(nodes.map((node) => message(1, node)), uint(2, 1), packedDouble(3, [0]));
}
const treeRegressorPayload = concat(message(1, treeParameters()), uint(2, 0));
const treeRegressor = await analyze(model(302, treeRegressorPayload), "tree-regressor.mlmodel");
assert(treeRegressor.total_macs === 0 && treeRegressor.coreml.classical_model.tree_count === 1
  && treeRegressor.coreml.classical_model.maximum_depth === 1, "Tree regressor structure ledger is incorrect");

const treeClassifierPayload = concat(message(1, treeParameters()), uint(2, 1), stringLabels(["negative", "positive"]));
const treeClassifier = await analyze(model(402, treeClassifierPayload, { classifier: true }), "tree-classifier.mlmodel");
assert(treeClassifier.coreml.classical_model.branch_node_count === 1 && treeClassifier.coreml.classical_model.leaf_node_count === 2,
  "Tree classifier node ledger is incorrect");
const treeReport = buildEngineeringReport(treeClassifier, { generatedAt: "2026-08-12T00:00:00.000Z" });
assert(treeReport.includes("Classical Model Contract") && treeReport.includes("1 tree(s); 1 branch / 2 leaf node(s)")
  && treeReport.includes("Pinned classical/pipeline sources"), "Core ML classical report omits structure or source evidence");
const treeConformance = releaseConformance(treeClassifier);
assert(treeConformance.release_export_allowed && treeConformance.checks.some((row) => row.id === "CF-COREML-CLASSICAL-001" && row.status === "pass"),
  `Core ML classical release conformance failed: ${JSON.stringify(treeConformance.failures)} ${JSON.stringify(treeConformance.finding_evidence_pointer_validation)}`);
assert(deriveCurrentArtifactCapabilityRow("coreml", treeClassifier).cells[4].id === "external",
  "Core ML classical models must not advertise an operation-level MLComputePlan import");
assert(!runtimeCapturePlanAvailability(treeClassifier).available,
  "Core ML classical models must fail closed for the NeuralNetwork/ML Program compute-plan collector");

await rejects(model(300, concat(message(1, doubleArray([1, 2, 3])), packedDouble(2, [0, 1]))), /weights and offsets must have equal cardinality/, "GLM offset mismatch");
await rejects(model(301, concat(message(1, linearKernel), message(3, denseVectors), message(4, coefficients([1])), double(5, 0))), /coefficient width does not match/, "SVM coefficient mismatch");
await rejects(model(302, concat(message(1, treeParameters([branchNode({ trueChild: 0, falseChild: 0 })])), uint(2, 0))), /multiple parents|exactly one root|cycle/, "Tree cycle");

const nonfinite = await analyze(model(300, concat(message(1, doubleArray([Number.NaN, 2, 3])), packedDouble(2, [0]))), "glm-nan.mlmodel");
assert(nonfinite.weight_integrity.nonfinite_value_count === 1, "Classical Core ML non-finite coefficient was not preserved as evidence");

const firstStage = modelWithDescription(300, glmRegressorPayload, regressionDescription("features", arrayType([3]), "mid"));
const secondStagePayload = concat(message(1, doubleArray([2])), packedDouble(2, [1]));
const secondStage = modelWithDescription(300, secondStagePayload, regressionDescription("mid", doubleType(), "prediction"));
const pipelineBody = concat(message(1, firstStage), message(1, secondStage), string(2, "project"), string(2, "calibrate"));
const pipelineDescription = regressionDescription("features", arrayType([3]), "prediction");
const pipeline = await analyze(modelWithDescription(202, pipelineBody, pipelineDescription), "pipeline.mlmodel");
assert(pipeline.operator_count === 2 && pipeline.tensor_count === 3 && pipeline.total_macs === 4, "Core ML pipeline graph merge or exact MAC conservation is incorrect");
assert(pipeline.weight_integrity.status === "assessed" && pipeline.weight_integrity.payload_bytes === 48,
  "Core ML pipeline nested numerical payload conservation is incorrect");
assert(pipeline.ops.map((op) => op.pipeline_model_name).join(",") === "project,calibrate", "Core ML pipeline model identity was not retained");
const pipelineReport = buildEngineeringReport(pipeline, { generatedAt: "2026-08-12T00:00:00.000Z" });
assert(pipelineReport.includes("Pipeline Stage Contract") && pipelineReport.includes("project") && pipelineReport.includes("calibrate")
  && pipelineReport.includes("2 pipeline operation(s)"), "Core ML pipeline report omits stage identity or graph semantics");
const pipelineConformance = releaseConformance(pipeline);
assert(pipelineConformance.release_export_allowed && pipelineConformance.checks.some((row) => row.id === "CF-COREML-CLASSICAL-001" && row.status === "pass"),
  `Core ML pipeline release conformance failed: ${JSON.stringify(pipelineConformance.failures)}`);

const tamperedPipeline = structuredClone(pipeline);
tamperedPipeline.weight_integrity.payload_bytes += 8;
let tamperedRejected = false;
try { releaseConformance(tamperedPipeline); }
catch (error) { tamperedRejected = /CF-COREML-CLASSICAL-001/.test(String(error?.message)); }
assert(tamperedRejected, "Core ML pipeline payload-byte tampering must fail release conformance");

const wrappedPipeline = await analyze(modelWithDescription(201, message(1, pipelineBody), pipelineDescription), "pipeline-regressor.mlmodel");
assert(wrappedPipeline.operator_count === 2 && wrappedPipeline.coreml.model_type === "pipelineRegressor", "Core ML PipelineRegressor wrapper was not decoded");
const mismatchedStage = modelWithDescription(300, secondStagePayload, regressionDescription("missing", doubleType(), "prediction"));
await rejects(modelWithDescription(202, concat(message(1, firstStage), message(1, mismatchedStage)), pipelineDescription), /input missing is not produced/, "Pipeline missing feature");
await rejects(modelWithDescription(202, concat(message(1, firstStage), message(1, secondStage), string(2, "duplicate")), pipelineDescription), /model-name count/, "Pipeline name cardinality");

console.log("Core ML GLM, SVM, TreeEnsemble, Pipeline cardinality, structure, arithmetic, numerical-integrity, and fail-closed checks passed.");
