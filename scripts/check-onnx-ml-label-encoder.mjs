import * as ort from "onnxruntime-node";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("ONNX-ML LabelEncoder check");
const cpuOptions = { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" };

const v1Bytes = labelEncoderModel({
  version: 1,
  input: stringTensor("scores", [2], ["dog", "missing"]),
  outputDtype: 7,
  attributes: [stringListAttribute("classes_strings", ["cat", "dog"]), intAttribute("default_int64", -7)],
});
const v1 = analyzeOnnxModel(v1Bytes, "label_encoder_v1.onnx");
const v1Row = row(v1);
expectEqual(v1Row.resolved_schema_version, 1, "Opset 1 must select LabelEncoder-1.");
expectEqual(v1Row.label_encoder_exact_match_count, 1, "Version 1 exact class matches must be counted.");
expectEqual(v1Row.label_encoder_exact_default_count, 1, "Version 1 default-path uses must be counted.");
expectEqual(JSON.stringify(v1Row.label_encoder_runtime_output_preview), '["1","-7"]', "Version 1 output preview must preserve exact INT64 text.");
expectEqual(JSON.stringify(v1.tensors.find((tensor) => tensor.name === "labels")?.initializer_integer_values_exact_decimals), '["1","-7"]', "Version 1 propagated INT64 values must retain exact decimal evidence.");
const v1Native = await runNative(v1Bytes);
expectEqual(JSON.stringify([...v1Native.labels.data].map(String)), '["1","-7"]', "Native ORT must match the version 1 reference output.");
const v1Bundle = bundle(v1);
expectEqual(v1Bundle.conformance.status, "pass", "Version 1 evidence must pass independent conformance.");
expect(v1Bundle.report.includes("LabelEncoder exact mapping effects"), "Engineering Report must expose LabelEncoder mapping conservation.");
assertCompactMlBomProjection(v1Bundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxMlExactLabelEncoderDefaults"],
  label: "LabelEncoder v1 compact ML-BOM",
});

const duplicateBytes = labelEncoderModel({
  version: 4,
  input: stringTensor("scores", [3], ["a", "b", "x"]),
  outputDtype: 5,
  attributes: [
    tensorAttribute("keys_tensor", stringTensor("", [3], ["a", "a", "b"])),
    tensorAttribute("values_tensor", numericTensor("", 5, [3], [1, 2, 3])),
    tensorAttribute("default_tensor", numericTensor("", 5, [1], [-1])),
  ],
});
const duplicate = analyzeOnnxModel(duplicateBytes, "label_encoder_v4_duplicate.onnx");
const duplicateRow = row(duplicate);
expectEqual(duplicateRow.label_encoder_duplicate_key_count, 1, "Version 4 duplicate keys must be counted exactly.");
expectEqual(duplicateRow.label_encoder_exact_duplicate_key_hit_count, 1, "Artifact-known duplicate-key hits must be counted exactly.");
expectEqual(duplicateRow.label_encoder_schema_runtime_mismatch_count, 1, "First-key/last-key output divergence must be counted exactly.");
expectEqual(JSON.stringify(duplicateRow.label_encoder_runtime_output_preview), '["1","3","-1"]', "Pinned ORT first-key output must be explicit.");
expectEqual(JSON.stringify(duplicateRow.label_encoder_schema_output_preview), '["2","3","-1"]', "ONNX-4 last-key output must be explicit.");
expectEqual(duplicateRow.label_encoder_output_materialized, false, "Conflicting exact semantics must suppress downstream exact values.");
const duplicateOutput = duplicate.tensors.find((tensor) => tensor.name === "labels");
expectEqual(duplicateOutput?.dtype, "INT16", "Schema-derived output dtype must survive a semantic conflict.");
expectEqual(JSON.stringify(duplicateOutput?.shape), "[3]", "Schema-derived output shape must survive a semantic conflict.");
expectEqual(duplicateOutput?.static_values_complete, false, "A source conflict must not leak one side as an unqualified static output.");
const duplicateAttrs = duplicate.ops[0].onnx_attributes;
expectEqual(duplicateAttrs.find((attribute) => attribute.name === "keys_tensor")?.tensor_value?.exact_values_text.join("|"), "a|a|b", "Serialized TensorProto string keys must remain in public evidence.");
expectEqual(duplicateAttrs.find((attribute) => attribute.name === "values_tensor")?.tensor_value?.exact_values_text.join("|"), "1|2|3", "Serialized TensorProto INT16 values must remain in public evidence.");
const duplicateNative = await runNative(duplicateBytes);
expectEqual(JSON.stringify([...duplicateNative.labels.data]), "[1,3,-1]", "Native ORT must confirm first-key ownership for duplicate version 4 keys.");
const duplicateFindings = buildFindingsRegister(duplicate);
expectEqual(duplicateFindings.find((finding) => finding.finding_id === "EA-ONX-0047")?.technical_priority, "High", "Duplicate ownership divergence must enter the High action queue.");
expectEqual(duplicateFindings.find((finding) => finding.finding_id === "EA-ONX-0049")?.technical_priority, "Medium", "Artifact-known default use must enter the Medium action queue.");
const duplicateBundle = bundle(duplicate);
expectEqual(duplicateBundle.conformance.status, "pass", "Version 4 duplicate evidence must pass independent reconstruction.");
expect(duplicateBundle.report.includes("first-key / last-key") && duplicateBundle.report.includes("schema/runtime mismatches"), "Engineering Report must state the ownership boundary and exact mismatch count.");
assertCompactMlBomProjection(duplicateBundle.mlBom, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:model:onnxMlExactLabelEncoderSchemaRuntimeMismatches"],
  label: "LabelEncoder v4 compact ML-BOM",
});

const nanBytes = labelEncoderModel({
  version: 2,
  input: numericTensor("scores", 1, [2], [Number.NaN, 1]),
  outputDtype: 8,
  attributes: [
    floatListAttribute("keys_floats", [Number.NaN]),
    stringListAttribute("values_strings", ["mapped"]),
    stringAttribute("default_string", "default"),
  ],
});
const nan = analyzeOnnxModel(nanBytes, "label_encoder_v2_nan.onnx");
const nanRow = row(nan);
expectEqual(nanRow.label_encoder_nan_key_count, 1, "Version 2 NaN keys must be counted without JSON null ambiguity.");
expectEqual(nanRow.label_encoder_exact_match_count, 0, "Pinned ORT version 2 equality must not report a NaN match.");
expectEqual(nanRow.label_encoder_exact_default_count, 2, "Both exact inputs must reach the default path under pinned ORT semantics.");
expectEqual(JSON.stringify(nanRow.label_encoder_runtime_output_preview), '["default","default"]', "Pinned runtime output must expose NaN miss behavior.");
const nanNative = await runNative(nanBytes);
expectEqual(JSON.stringify([...nanNative.labels.data]), '["default","default"]', "Native ORT must confirm version 2 NaN lookup misses.");
const nanFindings = buildFindingsRegister(nan);
expectEqual(nanFindings.find((finding) => finding.finding_id === "EA-ONX-0048")?.technical_priority, "High", "Version 2 NaN equality divergence must enter the High action queue.");
expectEqual(nanFindings.find((finding) => finding.finding_id === "EA-ONX-0050")?.technical_priority, "High", "Non-finite mapping state must enter the High action queue.");
expectEqual(bundle(nan).conformance.status, "pass", "Version 2 NaN evidence must pass independent reconstruction.");

const dtypeGapBytes = labelEncoderModel({
  version: 4,
  input: numericTensor("scores", 6, [1], [7]),
  outputDtype: 7,
  attributes: [
    tensorAttribute("keys_tensor", numericTensor("", 6, [1], [7])),
    tensorAttribute("values_tensor", numericTensor("", 7, [1], [99n])),
    tensorAttribute("default_tensor", numericTensor("", 7, [1], [-1n])),
  ],
});
const dtypeGap = analyzeOnnxModel(dtypeGapBytes, "label_encoder_v4_int32_int64_gap.onnx");
const dtypeGapRow = row(dtypeGap);
expectEqual(dtypeGapRow.status, "pass", "A schema-valid LabelEncoder dtype pair must retain an ONNX pass.");
expectEqual(dtypeGapRow.label_encoder_pinned_ort_contract_status, "fail", "Missing pinned CPU pair must retain a separate runtime failure.");
expectEqual(dtypeGap.tensors.find((tensor) => tensor.name === "labels")?.dtype, "INT64", "Runtime kernel absence must not erase ONNX-derived output type.");
expectEqual(JSON.stringify(dtypeGap.tensors.find((tensor) => tensor.name === "labels")?.shape), "[1]", "Runtime kernel absence must not erase ONNX-derived output shape.");
expectEqual(buildFindingsRegister(dtypeGap).find((finding) => finding.finding_id === "EA-ONX-0046")?.technical_priority, "High", "Pinned CPU dtype-pair gaps must enter the High action queue.");
expectEqual(bundle(dtypeGap).conformance.status, "pass", "Dtype-pair gap evidence must pass independent conformance.");
expect(await nativeRejects(dtypeGapBytes), "Native ORT CPU must reject a schema-valid pair absent from its version 4 registration ledger.");

const countGapBytes = labelEncoderModel({
  version: 2,
  input: stringTensor("scores", [1], ["a"]),
  outputDtype: 7,
  attributes: [stringListAttribute("keys_strings", ["a", "b"]), intListAttribute("values_int64s", [1])],
});
const countGap = analyzeOnnxModel(countGapBytes, "label_encoder_v2_count_gap.onnx");
const countGapRow = row(countGap);
expectEqual(countGapRow.status, "pass", "Version 2 schema inference must not invent a key/value count rejection absent from the schema.");
expectEqual(countGapRow.label_encoder_pinned_ort_contract_reason, "pinned_ort_key_value_count_mismatch", "Pinned runtime count rejection must be classified exactly.");
expect(countGapRow.risk_codes.includes("label_encoder_pinned_ort_runtime_contract_invalid"), "Runtime cardinality failure must not be mislabeled as a dtype-pair gap.");
expect(!countGapRow.risk_codes.includes("label_encoder_schema_dtype_pair_missing_pinned_ort_cpu_kernel"), "Runtime cardinality failure must remain distinct from kernel registration coverage.");
expectEqual(buildFindingsRegister(countGap).find((finding) => finding.finding_id === "EA-ONX-0051")?.technical_priority, "High", "Pinned runtime cardinality failures must enter the High action queue.");
expectEqual(bundle(countGap).conformance.status, "pass", "Runtime count-failure evidence must pass independent conformance.");
expect(await nativeRejects(countGapBytes), "Native ORT CPU must reject unequal LabelEncoder key/value cardinality.");

const malformedBytes = labelEncoderModel({
  version: 4,
  input: stringTensor("scores", [1], ["a"]),
  outputDtype: 5,
  attributes: [
    tensorAttribute("keys_tensor", stringTensor("", [1], ["a"])),
    tensorAttribute("values_tensor", numericTensor("", 5, [1, 1], [1])),
    tensorAttribute("default_tensor", numericTensor("", 5, [1], [-1])),
  ],
});
const malformed = analyzeOnnxModel(malformedBytes, "label_encoder_v4_malformed_tensor.onnx");
expectEqual(row(malformed).status, "fail", "Rank-2 LabelEncoder mapping attributes must fail the one-dimensional TensorProto contract.");
expectEqual(malformed.onnx_shape_inference.ml_value_inference.failed_node_count, 1, "Malformed LabelEncoder rows must enter failure conservation.");
expectEqual(bundle(malformed).conformance.status, "pass", "Faithfully reported malformed LabelEncoder evidence must still pass export conformance.");

const tampered = structuredClone(v1);
row(tampered).label_encoder_exact_default_count = 0;
expectThrows(() => bundle(tampered), "Evidence conformance failed", "Independent conformance must reject a tampered LabelEncoder mapping count.");

done("ONNX-ML LabelEncoder schema/runtime parity, report, ML-BOM, malformed, and tamper checks passed.");

function row(analysis) {
  return analysis.onnx_shape_inference.ml_value_inference.rows[0];
}

async function runNative(bytes) {
  const session = await ort.InferenceSession.create(bytes, cpuOptions);
  return session.run({});
}

async function nativeRejects(bytes) {
  try {
    const session = await ort.InferenceSession.create(bytes, cpuOptions);
    await session.run({});
    return false;
  } catch {
    return true;
  }
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

function labelEncoderModel({ version, input, outputDtype, attributes }) {
  const node = message([
    stringField(1, "scores"), stringField(2, "labels"), stringField(3, `label_encoder_v${version}`),
    stringField(4, "LabelEncoder"), ...attributes.map((attribute) => bytesField(5, attribute)), stringField(7, "ai.onnx.ml"),
  ]);
  const inputShape = tensorShape(input);
  const graph = message([
    bytesField(1, node), stringField(2, `label_encoder_v${version}_graph`), bytesField(5, input),
    bytesField(12, valueInfo("labels", outputDtype, inputShape)),
  ]);
  const opset = message([stringField(1, "ai.onnx.ml"), varintField(2, version)]);
  return message([varintField(1, 8), stringField(2, "deepbom_label_encoder_fixture"), bytesField(7, graph), bytesField(8, opset)]);
}

function tensorShape(tensor) {
  return tensor._shape;
}

function valueInfo(name, dtype, shape) {
  const dimensions = shape.map((dimension) => bytesField(1, message([varintField(1, dimension)])));
  const tensorType = message([varintField(1, dtype), bytesField(2, message(dimensions))]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

function numericTensor(name, dtype, shape, values) {
  const raw = numericBytes(dtype, values);
  const tensor = message([
    ...shape.map((dimension) => varintField(1, dimension)), varintField(2, dtype),
    ...(name ? [stringField(8, name)] : []), bytesField(9, raw),
  ]);
  Object.defineProperty(tensor, "_shape", { value: shape, enumerable: false });
  return tensor;
}

function stringTensor(name, shape, values) {
  const tensor = message([
    ...shape.map((dimension) => varintField(1, dimension)), varintField(2, 8),
    ...values.map((value) => stringField(6, value)), ...(name ? [stringField(8, name)] : []),
  ]);
  Object.defineProperty(tensor, "_shape", { value: shape, enumerable: false });
  return tensor;
}

function numericBytes(dtype, values) {
  const widths = { 1: 4, 5: 2, 6: 4, 7: 8, 11: 8 };
  const width = widths[dtype];
  const bytes = new Uint8Array(values.length * width);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    if (dtype === 1) view.setFloat32(index * width, Number(value), true);
    else if (dtype === 5) view.setInt16(index * width, Number(value), true);
    else if (dtype === 6) view.setInt32(index * width, Number(value), true);
    else if (dtype === 7) view.setBigInt64(index * width, BigInt(value), true);
    else if (dtype === 11) view.setFloat64(index * width, Number(value), true);
  });
  return bytes;
}

function tensorAttribute(name, tensor) {
  return message([stringField(1, name), bytesField(5, tensor), varintField(20, 4)]);
}

function stringAttribute(name, value) {
  return message([stringField(1, name), stringField(4, value), varintField(20, 3)]);
}

function intAttribute(name, value) {
  return message([stringField(1, name), varintField(3, value), varintField(20, 2)]);
}

function stringListAttribute(name, values) {
  return message([stringField(1, name), ...values.map((value) => stringField(9, value)), varintField(20, 8)]);
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
