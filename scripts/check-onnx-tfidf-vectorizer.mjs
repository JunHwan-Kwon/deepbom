import * as ort from "onnxruntime-node";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { validateTfIdfVectorizerRowAgainstEvidence } from "../web/lib/onnx-tfidf-vectorizer-conformance.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectDeepEqual, expectThrows } = createCheck("ONNX TfIdfVectorizer check");
const cpuOptions = { executionProviders: ["cpu"], graphOptimizationLevel: "disabled", logSeverityLevel: 4 };

const skipBytes = tfidfModel({ mode: "TF", minimum: 2, maximum: 2, skip: 5 });
const skip = analyzeOnnxModel(skipBytes, "tfidf_skip5.onnx");
const skipRow = row(skip);
expect(validateTfIdfVectorizerRowAgainstEvidence(skipRow, skip.tensors, skip.ops), "Exact TF row must independently reconstruct from public tensor/op evidence.");
expectEqual(skipRow.status, "pass", "Complete integer input must close the TfIdf contract and static execution.");
expectDeepEqual(skipRow.exact_output_shape, [7], "Rank-one TfIdf output width must be max(ngram_indexes)+1.");
expectDeepEqual(skipRow.exact_frequency_values, [0, 0, 0, 0, 1, 3, 1], "All skip distances through max_skip_count must be counted exactly.");
expectDeepEqual([...((await runNative(skipBytes)).Y.data)], skipRow.exact_output_values, "Pinned ORT CPU must agree with exact TF skip enumeration.");
expectEqual(skip.onnx_shape_inference.tfidf_vectorizer_inference.exact_match_count, 5, "Aggregate match arithmetic must conserve the node row.");

const batchBytes = tfidfModel({ inputShape: [2, 6], mode: "TF", minimum: 1, maximum: 2, skip: 5 });
const batchRow = row(analyzeOnnxModel(batchBytes, "tfidf_batch.onnx"));
expectDeepEqual(batchRow.exact_output_shape, [2, 7], "Rank-two output must preserve the batch dimension.");
expectDeepEqual(batchRow.exact_output_values, [0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1], "N-gram windows must not cross row boundaries.");
expectDeepEqual([...((await runNative(batchBytes)).Y.data)], batchRow.exact_output_values, "Pinned ORT CPU must agree with row-isolated enumeration.");

const stringBytes = tfidfModel({
  inputType: "string", inputValues: ["seven", "eight", "six", "seven"], inputShape: [4], mode: "TF",
  minimum: 2, maximum: 2, skip: 0, counts: [0, 0], indexes: [0, 1],
  poolType: "string", pool: ["seven", "eight", "six", "seven"], outputShape: [2],
});
const stringRow = row(analyzeOnnxModel(stringBytes, "tfidf_string.onnx"));
expectEqual(stringRow.static_input_status, "assessed_exact_string_values", "STRING initializer tokens must remain exact UTF-8 values.");
expectDeepEqual(stringRow.exact_output_values, [1, 1], "String bigrams must use the same exact skip enumeration.");
expectDeepEqual([...((await runNative(stringBytes)).Y.data)], stringRow.exact_output_values, "Pinned ORT CPU must agree with STRING token matching.");

const largeInt64 = 9_007_199_254_740_993n;
const int64Bytes = tfidfModel({
  inputType: "int64", inputValues: [largeInt64, 7n], inputShape: [2], mode: "TF",
  minimum: 1, maximum: 1, skip: 0, counts: [0], indexes: [0, 1], pool: [largeInt64, 7n], outputShape: [2],
});
const int64Analysis = analyzeOnnxModel(int64Bytes, "tfidf_exact_int64.onnx");
const int64Row = row(int64Analysis);
expectEqual(int64Row.static_input_status, "assessed_exact_int64_decimals", "Unsafe INT64 tokens must be compared as exact decimal identities.");
expectDeepEqual(int64Row.exact_output_values, [1, 1], "Unsafe INT64 tokens must not be rounded through JavaScript Number.");
expectDeepEqual([...((await runNative(int64Bytes)).Y.data)], int64Row.exact_output_values, "Pinned ORT CPU must confirm exact INT64 token matching.");

const mappedBytes = tfidfModel({
  inputValues: [5, 6], inputShape: [2], mode: "TFIDF", minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [1, 0], pool: [5, 6], weights: [2, 3], outputShape: [2],
});
const mappedAnalysis = analyzeOnnxModel(mappedBytes, "tfidf_weight_mapping.onnx");
const mappedRow = row(mappedAnalysis);
expectDeepEqual(mappedRow.exact_output_values, [2, 3], "Pinned runtime coordinate-indexed weights must be reproduced exactly.");
expectEqual(mappedRow.exact_weight_coordinate_value_disagreement_count, 2, "Both permuted n-grams must expose the schema-prose/runtime weight disagreement.");
expect(mappedRow.risk_codes.includes("tfidf_weight_coordinate_semantics_divergence"), "Coordinate-indexed weighting divergence must remain explicit evidence.");
expectDeepEqual([...((await runNative(mappedBytes)).Y.data)], mappedRow.exact_output_values, "Native ORT must confirm coordinate-indexed weighting semantics.");
expect(validateTfIdfVectorizerRowAgainstEvidence(mappedRow, mappedAnalysis.tensors, mappedAnalysis.ops), "Mapped weighted row must independently reconstruct.");

const repeatedBytes = tfidfModel({
  inputValues: new Array(20).fill(5), inputShape: [20], mode: "TFIDF", minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [0], pool: [5], weights: [0.1], outputShape: [1],
});
const repeatedAnalysis = analyzeOnnxModel(repeatedBytes, "tfidf_repeated_float32.onnx");
const repeatedRow = row(repeatedAnalysis);
expectEqual(repeatedRow.exact_match_count, 20, "Every repeated unigram hit must be counted.");
expectEqual(repeatedRow.exact_ort_reference_divergent_output_count, 1, "ORT repeated FLOAT32 addition must be distinguished from ONNX reference multiply-once arithmetic.");
expectDeepEqual([...((await runNative(repeatedBytes)).Y.data)], repeatedRow.exact_output_values, "Native ORT must confirm repeated FLOAT32 accumulation order.");

const negativeZeroBytes = tfidfModel({
  inputValues: [5], inputShape: [1], mode: "IDF", minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [0], pool: [5], weights: [-0], outputShape: [1],
});
const negativeZeroAnalysis = analyzeOnnxModel(negativeZeroBytes, "tfidf_negative_zero.onnx");
expectEqual(row(negativeZeroAnalysis).exact_negative_zero_output_count, 1, "IDF must preserve a matched negative-zero weight.");
expectEqual(negativeZeroAnalysis.tensors.find((tensor) => tensor.name === "Y")?.static_values_negative_zero_count, 1, "Propagated tensor evidence must preserve the negative-zero index.");
expect(Object.is((await runNative(negativeZeroBytes)).Y.data[0], -0), "Pinned ORT CPU must confirm the negative-zero output bit sign.");

const duplicate = analyzeOnnxModel(tfidfModel({
  inputValues: [5], inputShape: [1], minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [0, 1], pool: [5, 5], outputShape: [2],
}), "tfidf_duplicate.onnx");
expectEqual(row(duplicate).status, "fail", "A duplicate active n-gram must fail the pinned ORT constructor contract.");
expect(row(duplicate).reason_codes.includes("tfidf_duplicate_active_ngram_rejected_by_pinned_ort"), "Duplicate failure must retain its exact reason.");

const mismatch = analyzeOnnxModel(tfidfModel({
  inputValues: [5], inputShape: [1], minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [0, 1], pool: [5], outputShape: [2],
}), "tfidf_cardinality_mismatch.onnx");
expectEqual(row(mismatch).status, "fail", "Pool definition and index cardinality mismatch must fail closed.");
expect(row(mismatch).reason_codes.includes("tfidf_definition_index_cardinality_mismatch"), "Cardinality mismatch must remain machine-readable.");

const aliased = analyzeOnnxModel(tfidfModel({
  inputValues: [5, 6], inputShape: [2], mode: "TF", minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [0, 0], pool: [5, 6], outputShape: [1],
}), "tfidf_coordinate_alias.onnx");
expectDeepEqual(row(aliased).exact_output_values, [2], "Shared output coordinates must aggregate distinct n-gram frequencies exactly.");

const bounded = analyzeOnnxModel(tfidfModel({
  inputValues: [5], inputShape: [1], mode: "TF", minimum: 1, maximum: 1, skip: 0,
  counts: [0], indexes: [65_536], pool: [5], outputShape: [65_537],
}), "tfidf_bounded_output.onnx");
expectEqual(row(bounded).status, "partial", "Static output beyond the declared materialization limit must remain partial.");
expectEqual(row(bounded).static_execution_status, "not_assessed_output_element_limit", "Bounded residual must retain its exact limit reason without inventing zeros.");

const skipBundle = bundle(skip);
expectEqual(skipBundle.conformance.status, "pass", "Exact TfIdf evidence must pass independent bundle conformance.");
expect(skipBundle.report.includes("### TfIdfVectorizer-9 Contracts")
  && skipBundle.report.includes("Exact TfIdfVectorizer facts")
  && skipBundle.report.includes("1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02"), "Engineering Report must preserve exact TfIdf arithmetic and pinned source identity.");
assertCompactMlBomProjection(skipBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxTfIdfVectorizerExactMatches"],
  label: "TfIdfVectorizer compact ML-BOM",
});
expectEqual(buildFindingsRegister(mappedAnalysis).find((finding) => finding.finding_id === "EA-ONX-0067")?.technical_priority, "High", "Weight-coordinate semantics must enter the High action queue.");
expectEqual(buildFindingsRegister(repeatedAnalysis).find((finding) => finding.finding_id === "EA-ONX-0068")?.technical_priority, "Medium", "ORT/reference FLOAT32 divergence must enter the action queue.");
expectEqual(buildFindingsRegister(duplicate).find((finding) => finding.finding_id === "EA-ONX-0066")?.technical_priority, "High", "Runtime-invalid duplicate n-grams must enter the High action queue.");
expectEqual(buildFindingsRegister(aliased).find((finding) => finding.finding_id === "EA-ONX-0069")?.technical_priority, "Medium", "Coordinate aliasing must enter the action queue.");
expectEqual(buildFindingsRegister(bounded).find((finding) => finding.finding_id === "EA-ONX-0070")?.technical_priority, "Informational", "Bounded exact residuals must remain explicit informational findings.");
expectEqual(bundle(mappedAnalysis).conformance.status, "pass", "Weight-semantic divergence evidence must pass conformance when faithfully reported.");
expectEqual(bundle(repeatedAnalysis).conformance.status, "pass", "Reference-divergence evidence must pass conformance when faithfully reported.");
expectEqual(bundle(duplicate).conformance.status, "pass", "Faithfully reported runtime-invalid TfIdf evidence must pass conformance.");
expectEqual(bundle(aliased).conformance.status, "pass", "Faithfully reported coordinate aliases must pass conformance.");
expectEqual(bundle(bounded).conformance.status, "pass", "Faithfully bounded static residuals must pass conformance.");

const tampered = structuredClone(mappedAnalysis);
tampered.onnx_shape_inference.tfidf_vectorizer_inference.rows[0].exact_output_values[0] = 99;
expectThrows(() => bundle(tampered), "CF-SHAPE-TFIDF-002", "Independent public-evidence reconstruction must reject a forged exact feature vector.");

done("ONNX TfIdfVectorizer shape, skip, batch, weighting, native ORT, and malformed checks passed.");

function row(analysis) {
  return analysis.onnx_shape_inference.tfidf_vectorizer_inference.rows[0];
}

function bundle(analysis) {
  const mlBom = buildMlBomDocument(analysis, { hash: "" });
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-23T00:00:00.000Z" },
    rawEvidenceContext: { identity: { filename: analysis.filename, format: "onnx" } },
    mlBomDocument: mlBom,
  });
  const evidence = JSON.parse(files.find((file) => file.name === "engineering_evidence.json")?.data || "{}");
  return {
    mlBom,
    report: files.find((file) => file.name === "engineering_report.md")?.data || "",
    conformance: evidence.evidence?.conformance_report || {},
  };
}

function property(document, name) {
  return [...(document.properties || []), ...(document.metadata?.component?.properties || [])]
    .find((item) => item.name === name)?.value;
}

async function runNative(bytes) {
  const session = await ort.InferenceSession.create(bytes.slice(), cpuOptions);
  try {
    return await session.run({});
  } finally {
    await session.release();
  }
}

function tfidfModel({
  inputValues = [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8], inputShape = [12], mode = "TF",
  minimum = 2, maximum = 2, skip = 5, counts = [0, 4], indexes = [0, 1, 2, 3, 4, 5, 6],
  pool = [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], weights = null, outputShape = null,
  inputType = "int32", poolType = "int64",
} = {}) {
  const input = inputType === "string" ? stringTensor("X", inputShape, inputValues)
    : inputType === "int64" ? int64Tensor("X", inputShape, inputValues) : int32Tensor("X", inputShape, inputValues);
  const attributes = [
    stringAttribute("mode", mode), intAttribute("min_gram_length", minimum),
    intAttribute("max_gram_length", maximum), intAttribute("max_skip_count", skip),
    intListAttribute("ngram_counts", counts), intListAttribute("ngram_indexes", indexes),
    poolType === "string" ? stringListAttribute("pool_strings", pool) : intListAttribute("pool_int64s", pool),
  ];
  if (weights) attributes.push(floatListAttribute("weights", weights));
  const node = message([
    stringField(1, "X"), stringField(2, "Y"), stringField(3, "tfidf"), stringField(4, "TfIdfVectorizer"),
    ...attributes.map((attribute) => bytesField(5, attribute)),
  ]);
  const width = Math.max(...indexes) + 1;
  const shape = outputShape || (inputShape.length === 1 ? [width] : [inputShape[0], width]);
  const graph = message([bytesField(1, node), stringField(2, "tfidf_fixture"), bytesField(5, input), bytesField(12, valueInfo("Y", 1, shape))]);
  const opset = message([varintField(2, 9)]);
  return message([varintField(1, 8), stringField(2, "deepbom_tfidf_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function int64Tensor(name, shape, values) {
  return message([...shape.map((dimension) => varintField(1, dimension)), varintField(2, 7), ...values.map((value) => varintField(7, value)), stringField(8, name)]);
}

function stringTensor(name, shape, values) {
  return message([...shape.map((dimension) => varintField(1, dimension)), varintField(2, 8), ...values.map((value) => stringField(6, value)), stringField(8, name)]);
}

function int32Tensor(name, shape, values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  return message([...shape.map((dimension) => varintField(1, dimension)), varintField(2, 6), stringField(8, name), bytesField(9, bytes)]);
}

function valueInfo(name, dtype, shape) {
  const dimensions = shape.map((dimension) => bytesField(1, message([varintField(1, dimension)])));
  const tensorType = message([varintField(1, dtype), bytesField(2, message(dimensions))]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

function intAttribute(name, value) { return message([stringField(1, name), varintField(3, value), varintField(20, 2)]); }
function stringAttribute(name, value) { return message([stringField(1, name), stringField(4, value), varintField(20, 3)]); }
function intListAttribute(name, values) { return message([stringField(1, name), ...values.map((value) => varintField(8, value)), varintField(20, 7)]); }
function stringListAttribute(name, values) { return message([stringField(1, name), ...values.map((value) => stringField(9, value)), varintField(20, 8)]); }
function floatListAttribute(name, values) { return message([stringField(1, name), ...values.map((value) => fixed32Field(7, value)), varintField(20, 6)]); }
function stringField(no, value) { return bytesField(no, new TextEncoder().encode(value)); }
function bytesField(no, bytes) { return Uint8Array.from([...encodeVarint(BigInt((no << 3) | 2)), ...encodeVarint(BigInt(bytes.length)), ...bytes]); }
function varintField(no, value) { return Uint8Array.from([...encodeVarint(BigInt(no << 3)), ...encodeVarint(BigInt(value))]); }
function fixed32Field(no, value) {
  const bytes = new Uint8Array(5);
  bytes[0] = (no << 3) | 5;
  new DataView(bytes.buffer).setFloat32(1, value, true);
  return bytes;
}
function message(fields) { return Uint8Array.from(fields.flatMap((field) => [...field])); }
function encodeVarint(value) {
  const bytes = [];
  let current = value;
  do {
    let byte = Number(current & 0x7fn);
    current >>= 7n;
    if (current) byte |= 0x80;
    bytes.push(byte);
  } while (current);
  return bytes;
}
