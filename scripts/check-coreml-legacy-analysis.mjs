import { createHash } from "node:crypto";
import { File } from "node:buffer";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildCoreMlPerChannelLinearFixture } from "./coreml-legacy-quantization-corpus-lib.mjs";

function assert(value, message) { if (!value) throw new Error(message); }

function varint(value) {
  let current = BigInt(value);
  const bytes = [];
  while (current > 0x7fn) { bytes.push(Number(current & 0x7fn) | 0x80); current >>= 7n; }
  bytes.push(Number(current));
  return Buffer.from(bytes);
}

function concat(...values) { return Buffer.concat(values.flat().filter(Boolean)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function float32(field, value) { const body = Buffer.alloc(4); body.writeFloatLE(value); return concat(key(field, 5), body); }
function packedUint(field, values) { const body = concat(values.map(varint)); return concat(key(field, 2), varint(body.length), body); }
function sint(field, value) { return concat(key(field, 0), varint(BigInt.asUintN(64, BigInt(value)))); }
function packedInt(field, values) {
  const body = concat(values.map((value) => varint(BigInt.asUintN(64, BigInt(value)))));
  return concat(key(field, 2), varint(body.length), body);
}
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function packedFloat(field, values) {
  const body = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => body.writeFloatLE(value, index * 4));
  return bytes(field, body);
}

function borderAmounts(pairs) {
  return concat(pairs.map(([start, end]) => message(10, concat(uint(1, start), uint(2, end)))));
}

function weight(values) { return packedFloat(1, values); }
function quantizedWeight(codes, { bits = 4, scales = [0.25], biases = [0] } = {}) {
  let accumulator = 0;
  let occupied = 0;
  const packed = [];
  for (const code of codes) {
    if (!Number.isSafeInteger(code) || code < 0 || code >= 2 ** bits) throw new Error("invalid fixture quantized code");
    accumulator = (accumulator << bits) | code;
    occupied += bits;
    while (occupied >= 8) {
      occupied -= 8;
      packed.push((accumulator >>> occupied) & 0xff);
      accumulator &= 2 ** occupied - 1;
    }
  }
  if (occupied) packed.push((accumulator << (8 - occupied)) & 0xff);
  const quantization = concat(uint(1, bits), message(101, concat(packedFloat(1, scales), packedFloat(2, biases))));
  return concat(bytes(30, packed), message(40, quantization));
}
function layer(name, inputs, outputs, typeField, params) {
  return concat(string(1, name), inputs.map((value) => string(2, value)), outputs.map((value) => string(3, value)), message(typeField, params));
}

function feature(name, type) { return concat(string(1, name), message(3, type)); }
function imageType(width, height) { return message(4, concat(uint(1, width), uint(2, height), uint(3, 10))); }
function enumeratedImageType(width, height, sizes) {
  const enumerated = concat(sizes.map(([candidateWidth, candidateHeight]) => message(1, concat(uint(1, candidateWidth), uint(2, candidateHeight)))));
  return message(4, concat(uint(1, width), uint(2, height), uint(3, 10), message(21, enumerated)));
}
function arrayType(shape) { return message(5, concat(packedUint(1, shape), uint(2, 65568))); }
function rangedArrayType(shape, ranges) {
  const shapeRange = concat(ranges.map(({ lower, upper, omitLower = false }) => message(1, concat(
    omitLower ? null : uint(1, lower),
    sint(2, upper),
  ))));
  return message(5, concat(packedUint(1, shape), uint(2, 65568), message(31, shapeRange)));
}

function rangedInterfaceFixture() {
  const description = concat(
    message(1, feature("input", rangedArrayType([1, 4], [
      { lower: 0, upper: -2, omitLower: true },
      { lower: 2, upper: 8 },
    ]))),
    message(10, feature("output", arrayType([1, 4]))),
  );
  return concat(uint(1, 5), message(2, description), message(500, Buffer.alloc(0)));
}

function flexibleImageConvFixture() {
  const convolution = concat(
    uint(1, 2), uint(2, 1), packedUint(20, [3, 3]), packedUint(30, [1, 1]), message(50, Buffer.alloc(0)),
    message(90, weight(Array.from({ length: 18 }, (_, index) => (index + 1) / 32))),
  );
  const network = concat(message(1, layer("conv", ["image"], ["output"], 100, convolution)));
  const description = concat(
    message(1, feature("image", enumeratedImageType(4, 4, [[4, 4], [8, 8]]))),
    message(10, feature("output", arrayType([1, 1, 2, 2, 2]))),
  );
  return concat(uint(1, 3), message(2, description), message(500, network));
}

function fixture({ corruptWeights = false, nonfinite = false, unnamedPreprocessing = false } = {}) {
  const convValues = Array.from({ length: corruptWeights ? 17 : 18 }, (_, index) => nonfinite && index === 0 ? Number.NaN : (index + 1) / 32);
  const conv = concat(
    uint(1, 2), uint(2, 1), packedUint(20, [3, 3]), packedUint(30, [1, 1]), message(51, Buffer.alloc(0)),
    uint(70, 1), message(90, weight(convValues)), message(91, weight([0, 0])),
  );
  const pool = concat(uint(1, 0), packedUint(10, [2, 2]), packedUint(20, [2, 2]), message(30, Buffer.alloc(0)));
  const dense = concat(uint(1, 8), uint(2, 3), uint(10, 1), message(20, weight(Array.from({ length: 24 }, (_, index) => index / 24))), message(21, weight([0, 0, 0])));
  const network = concat(
    message(1, layer("conv", ["image"], ["conv_out"], 100, conv)),
    message(1, layer("pool", ["conv_out"], ["pool_out"], 120, pool)),
    message(1, layer("flatten", ["pool_out"], ["flat_out"], 301, Buffer.alloc(0))),
    message(1, layer("dense", ["flat_out"], ["scores"], 140, dense)),
    message(1, layer("softmax", ["scores"], ["probabilities"], 175, Buffer.alloc(0))),
    unnamedPreprocessing ? message(2, message(10, concat(float32(10, 1), float32(20, -103.939), float32(21, -116.779), float32(22, -123.68)))) : null,
  );
  const description = concat(message(1, feature("image", imageType(4, 4))), message(10, feature("probabilities", arrayType([3]))));
  return concat(uint(1, 1), message(2, description), message(500, network));
}


function imagePipelineFixture() {
  const nested = fixture();
  const pipeline = message(1, nested);
  const description = concat(message(1, feature("image", imageType(4, 4))), message(10, feature("probabilities", arrayType([3]))));
  return concat(uint(1, 1), message(2, description), message(202, pipeline));
}

function singleLayerFixture({ inputShape, outputShape, typeField, params, specificationVersion = 5 }) {
  const network = concat(
    message(1, layer("subject", ["input"], ["output"], typeField, params)),
    uint(5, 1),
  );
  const description = concat(
    message(1, feature("input", arrayType(inputShape))),
    message(10, feature("output", arrayType(outputShape))),
  );
  return concat(uint(1, specificationVersion), message(2, description), message(500, network));
}

function operationFixture({ inputShapes, outputShapes, typeField, params = Buffer.alloc(0), specificationVersion = 5 }) {
  const inputNames = inputShapes.map((_, index) => `input_${index}`);
  const outputNames = outputShapes.map((_, index) => `output_${index}`);
  const network = concat(message(1, layer("subject", inputNames, outputNames, typeField, params)), uint(5, 1));
  const description = concat(
    inputShapes.map((shape, index) => message(1, feature(inputNames[index], arrayType(shape)))),
    outputShapes.map((shape, index) => message(10, feature(outputNames[index], arrayType(shape)))),
  );
  return concat(uint(1, specificationVersion), message(2, description), message(500, network));
}

async function expectRejected(payload, pattern, message) {
  let rejected = false;
  try { await analyze(payload); } catch (error) { rejected = pattern.test(String(error?.message)); }
  assert(rejected, message);
}

function recurrentWeights(kind, inputSize, outputSize, { bias = true, peephole = false } = {}) {
  const matrix = (rows, columns, seed) => weight(Array.from({ length: rows * columns }, (_, index) => (seed + index + 1) / 1024));
  const vector = (seed) => weight(Array.from({ length: outputSize }, (_, index) => (seed + index + 1) / 1024));
  if (kind === "simple") return concat(
    message(30, matrix(outputSize, inputSize, 0)),
    message(31, matrix(outputSize, outputSize, 100)),
    bias ? message(32, vector(200)) : null,
  );
  if (kind === "gru") {
    return concat(
      [30, 31, 32].map((field, index) => message(field, matrix(outputSize, inputSize, index * 100))),
      [50, 51, 52].map((field, index) => message(field, matrix(outputSize, outputSize, 500 + index * 100))),
      bias ? [70, 71, 72].map((field, index) => message(field, vector(900 + index * 100))) : null,
    );
  }
  return concat(
    [1, 2, 3, 4].map((field, index) => message(field, matrix(outputSize, inputSize, index * 100))),
    [20, 21, 22, 23].map((field, index) => message(field, matrix(outputSize, outputSize, 500 + index * 100))),
    bias ? [40, 41, 42, 43].map((field, index) => message(field, vector(900 + index * 100))) : null,
    peephole ? [60, 61, 62].map((field, index) => message(field, vector(1400 + index * 100))) : null,
  );
}

function recurrentFixture(kind) {
  const inputSize = 4;
  const outputSize = 5;
  const inputShape = [3, 2, inputSize, 1, 1];
  const typeField = { simple: 400, gru: 410, lstm: 420, bilstm: 430 }[kind];
  let params = concat(uint(1, inputSize), uint(2, outputSize));
  if (kind === "simple") params = concat(params, message(10, Buffer.alloc(0)), uint(15, 1), uint(20, 1), recurrentWeights(kind, inputSize, outputSize));
  else if (kind === "gru") params = concat(params, message(10, Buffer.alloc(0)), message(10, Buffer.alloc(0)), uint(15, 1), uint(20, 1), recurrentWeights(kind, inputSize, outputSize));
  else {
    const lstmParams = concat(uint(10, 1), uint(20, 1), uint(40, 1), float32(60, 50));
    const weights = recurrentWeights("lstm", inputSize, outputSize, { peephole: true });
    params = concat(params,
      [10, 10, 10].map((field) => message(field, Buffer.alloc(0))),
      kind === "bilstm" ? [11, 11, 11].map((field) => message(field, Buffer.alloc(0))) : null,
      message(15, lstmParams), message(20, weights), kind === "bilstm" ? message(20, weights) : null,
    );
  }
  const directions = kind === "bilstm" ? 2 : 1;
  return singleLayerFixture({ inputShape, outputShape: [3, 2, outputSize * directions, 1, 1], typeField, params });
}

async function analyze(payload) {
  return (await readCoreMlModelFile(new File([payload], "fixture.mlmodel"))).analysis;
}

const bytesOk = fixture();
const analysis = await analyze(bytesOk);
analysis.model_sha256 = createHash("sha256").update(bytesOk).digest("hex");
assert(analysis.operator_count === 5 && analysis.tensor_count === 6, "Core ML layer/tensor conservation failed");
assert(analysis.total_macs === 312, `Core ML exact MAC total is ${analysis.total_macs}; expected 312`);
assert(analysis.mac_assessment?.status === "assessed_all_decoded_compute_ops", "Core ML MAC assessment is not complete");
assert(JSON.stringify(analysis.ops.map((op) => op.output_shapes[0])) === JSON.stringify([[1, 1, 2, 4, 4], [1, 1, 2, 2, 2], [1, 1, 8, 1, 1], [1, 1, 3, 1, 1], [1, 1, 3, 1, 1]]), "Core ML shape propagation is incorrect");
assert(analysis.weight_integrity?.status === "assessed" && analysis.weight_integrity.parameter_count === 4, "Core ML WeightParams coverage is incomplete");
assert(analysis.weight_integrity.payload_bytes === 188 && analysis.weight_integrity.assessed_payload_bytes === 188, "Core ML WeightParams byte conservation failed");
assert(analysis.weight_integrity.nonfinite_value_count === 0, "Finite Core ML fixture produced a non-finite finding");
const rangedInterface = await analyze(rangedInterfaceFixture());
assert(rangedInterface.inputs[0].constraints?.flexibility?.dimensions?.[0]?.lower_bound === 0
  && rangedInterface.inputs[0].constraints.flexibility.dimensions[0].upper_bound === -1
  && rangedInterface.inputs[0].constraints.flexibility.dimensions[0].upper_bound_unbounded === true
  && rangedInterface.inputs[0].constraints.flexibility.dimensions[0].serialized_upper_bound === -2,
"Core ML SizeRange omitted lowerBound or negative unbounded upperBound was not normalized from the pinned protobuf contract");
const flexibleImageBytes = flexibleImageConvFixture();
const flexibleImage = await analyze(flexibleImageBytes);
flexibleImage.model_sha256 = createHash("sha256").update(flexibleImageBytes).digest("hex");
const flexibleScenarios = flexibleImage.coreml?.flexible_input_scenarios;
assert(flexibleScenarios?.status === "assessed_all_serialized_cases" && flexibleScenarios.scenario_count === 2,
"Core ML enumerated input cases were not exhaustively evaluated");
assert(JSON.stringify(flexibleScenarios.scenarios.map((row) => [row.total_macs, row.input_logical_payload_bytes, row.output_logical_payload_bytes, row.peak_live_logical_payload_bytes]))
  === JSON.stringify([[72, 64, 32, 96], [648, 256, 288, 544]]),
`Core ML flexible-shape scenario arithmetic is inconsistent: ${JSON.stringify(flexibleScenarios.scenarios)}`);
const flexibleReport = buildEngineeringReport(flexibleImage, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(flexibleReport.includes("Core ML Flexible Input Scenarios") && flexibleReport.includes("648 MACs")
  && flexibleReport.includes("not a proof that an interior point cannot have a larger cost or payload"),
"Core ML flexible-shape scenarios are not projected with their evidence boundary");
const flexibleEvidence = buildEngineeringEvidenceDocument(flexibleImage, {
  reportContext: { generatedAt: "2026-08-07T00:00:00.000Z", identity: { filename: "flexible.mlmodel", format: "coreml", sha256: flexibleImage.model_sha256 } },
  rawEvidenceContext: { identity: { filename: "flexible.mlmodel", format: "coreml", sha256: flexibleImage.model_sha256 } },
  mlBomDocument: buildMlBomDocument(flexibleImage, { hash: flexibleImage.model_sha256 }),
});
assert(flexibleEvidence.evidence?.conformance_report?.status === "pass", "Core ML flexible-shape report/export conformance failed");
const unnamedPreprocessingAnalysis = await analyze(fixture({ unnamedPreprocessing: true }));
assert(unnamedPreprocessingAnalysis.coreml?.preprocessing_binding?.status === "partial_missing_required_feature_name"
  && unnamedPreprocessingAnalysis.coreml.preprocessing_binding.unbound_entry_count === 1
  && unnamedPreprocessingAnalysis.inputs[0].coreml_preprocessing == null,
"Core ML preprocessing without the required featureName was not retained as an explicit unbound contract");
assert(analysis.size_breakdown?.status === "assessed" && analysis.size_breakdown.constant_bytes === 188
  && analysis.size_breakdown.constant_bytes + analysis.size_breakdown.structure_overhead_bytes === bytesOk.length, "Core ML model/container byte conservation failed");
assert(analysis.tensor_liveness?.status === "assessed" && analysis.tensor_liveness.peak_bytes === 192, `Core ML exact graph-liveness peak is ${analysis.tensor_liveness?.peak_bytes}; expected 192 B`);
const convDigest = analysis.ops[0].coreml_weights.find((item) => item.role === "weights").numerical_integrity.payload_sha256;
const expectedDigest = createHash("sha256").update(Buffer.from(new Float32Array(Array.from({ length: 18 }, (_, index) => (index + 1) / 32)).buffer)).digest("hex");
assert(convDigest === expectedDigest, "Core ML WeightParams payload digest is not exact");
const report = buildEngineeringReport(analysis, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(report.includes("Output shape(s) / contract") && report.includes("derived_coreml_convolution_contract"), "Core ML report omits per-layer shape evidence");
const evidence = buildEngineeringEvidenceDocument(analysis, {
  reportContext: { generatedAt: "2026-08-07T00:00:00.000Z", identity: { filename: "fixture.mlmodel", format: "coreml", sha256: analysis.model_sha256 } },
  rawEvidenceContext: { identity: { filename: "fixture.mlmodel", format: "coreml", sha256: analysis.model_sha256 } },
  mlBomDocument: buildMlBomDocument(analysis, { hash: analysis.model_sha256 }),
});
assert(evidence.evidence?.conformance_report?.status === "pass", "Core ML legacy report/export conformance failed");

const perAxisPayload = buildCoreMlPerChannelLinearFixture();
const perAxisQuantized = await analyze(perAxisPayload);
perAxisQuantized.model_sha256 = createHash("sha256").update(perAxisPayload).digest("hex");
const perAxisWeight = perAxisQuantized.ops[0].coreml_weights.find((item) => item.role === "weights");
assert(perAxisWeight?.storage === "raw_quantized"
  && perAxisWeight.value_count === 18
  && perAxisWeight.quantization?.number_of_bits === 4
  && perAxisWeight.quantization?.scheme === "linear"
  && perAxisWeight.quantization?.granularity === "per_axis"
  && perAxisWeight.quantization?.axis === 0
  && perAxisWeight.quantization?.channel_count === 2
  && perAxisWeight.quantization?.scale_count === 2
  && perAxisWeight.quantization?.bias_count === 2,
"Core ML per-axis linear WeightParams contract was not retained exactly");
assert(perAxisWeight.numerical_integrity?.status === "assessed_dequantized_quantized_codes"
  && perAxisWeight.numerical_integrity.decoded_value_count === 18
  && perAxisWeight.numerical_integrity.finite_count === 18
  && perAxisWeight.numerical_integrity.nonfinite_count === 0
  && perAxisWeight.numerical_integrity.quant_code_levels_used === 16,
"Core ML per-axis packed-code numerical integrity was not derived exactly");
assert(perAxisQuantized.quantization_status?.assessment_status === "assessed"
  && perAxisQuantized.quantization_status.per_axis_quantized_weight_parameter_count === 1,
"Core ML per-axis quantization summary did not conserve the decoded WeightParams ledger");
const perAxisReport = buildEngineeringReport(perAxisQuantized, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(perAxisReport.includes("Legacy quantization granularity")
  && perAxisReport.includes("1 per-axis linear / 0 single-scale or LUT among 1 quantized WeightParams"),
"Core ML engineering report omits the per-axis quantization denominator");
const perAxisEvidence = buildEngineeringEvidenceDocument(perAxisQuantized, {
  reportContext: { generatedAt: "2026-08-07T00:00:00.000Z", identity: { filename: "per-output-channel-linear-int4.mlmodel", format: "coreml", sha256: perAxisQuantized.model_sha256 } },
  rawEvidenceContext: { identity: { filename: "per-output-channel-linear-int4.mlmodel", format: "coreml", sha256: perAxisQuantized.model_sha256 } },
  mlBomDocument: buildMlBomDocument(perAxisQuantized, { hash: perAxisQuantized.model_sha256 }),
});
assert(perAxisEvidence.evidence?.conformance_report?.status === "pass", "Core ML per-axis report/export conformance failed");
await expectRejected(buildCoreMlPerChannelLinearFixture({ scaleCount: 1 }), /scale\/bias cardinality does not match quantization axis 0 \(2\)/,
  "Core ML per-axis scale cardinality mismatch did not fail closed");

const quantizedBatchnorm = await analyze(singleLayerFixture({
  inputShape: [1, 3, 2, 2], outputShape: [1, 3, 2, 2], typeField: 160,
  params: concat(
    uint(1, 3),
    message(15, quantizedWeight([1, 2, 3])),
    message(16, quantizedWeight([4, 5, 6])),
    message(17, quantizedWeight([7, 8, 9])),
    message(18, quantizedWeight([10, 11, 12])),
  ),
}));
assert(quantizedBatchnorm.ops[0].coreml_weights.every((row) => row.value_count === 3
  && row.quantization?.granularity === "per_tensor"
  && row.quantization?.axis === null
  && row.numerical_integrity?.status === "assessed_dequantized_quantized_codes"),
"Core ML BatchNorm quantized WeightParams were not bound to the source-declared channel cardinality");

const quantizedScale = await analyze(singleLayerFixture({
  inputShape: [1, 3, 2, 2], outputShape: [1, 3, 2, 2], typeField: 245,
  params: concat(
    packedUint(1, [3]), message(2, quantizedWeight([1, 2, 3])), uint(3, 1),
    packedUint(4, [3, 1, 1]), message(5, quantizedWeight([4, 5, 6], { scales: [0.25, 0.5, 1], biases: [0, 0, 0] })),
  ),
}));
const scaleWeights = Object.fromEntries(quantizedScale.ops[0].coreml_weights.map((row) => [row.role, row]));
assert(scaleWeights.scale?.value_count === 3 && scaleWeights.scale.quantization?.granularity === "per_tensor"
  && scaleWeights.bias?.value_count === 3 && scaleWeights.bias.quantization?.granularity === "per_axis"
  && scaleWeights.bias.quantization?.channel_count === 3,
"Core ML Scale quantization channels do not follow the pinned validator's shape rules");

const quantizedBias = await analyze(singleLayerFixture({
  inputShape: [1, 2, 2, 2], outputShape: [1, 2, 2, 2], typeField: 250,
  params: concat(packedUint(1, [2, 1, 1]), message(2, quantizedWeight([1, 2], { scales: [0.25, 0.5], biases: [0, 0] }))),
}));
assert(quantizedBias.ops[0].coreml_weights[0]?.value_count === 2
  && quantizedBias.ops[0].coreml_weights[0].quantization?.granularity === "per_axis",
"Core ML Bias quantized WeightParams were not closed from the serialized shape contract");

const imagePipeline = await analyze(imagePipelineFixture());
assert(imagePipeline.operator_count === 5 && imagePipeline.total_macs === 312, "Core ML image pipeline must preserve nested graph arithmetic.");
assert(imagePipeline.coreml.pipeline.feature_adapter_count >= 1, "Core ML image-to-neural-network layout adaptation should be retained as an explicit pipeline binding.");
const imageBinding = imagePipeline.coreml.pipeline.feature_bindings.find((binding) => binding.feature_name === "image");
assert(imageBinding?.binding === "serialized_feature_adapter"
  && JSON.stringify(imageBinding.feature_shape) !== JSON.stringify(imageBinding.nested_shape),
"Core ML image feature and nested neural-network tensor layouts must not be silently merged or rejected.");

const nonfiniteAnalysis = await analyze(fixture({ nonfinite: true }));
assert(nonfiniteAnalysis.weight_integrity.nonfinite_value_count === 1, "Core ML NaN payload was not observed");

let rejected = false;
try { await analyze(fixture({ corruptWeights: true })); } catch (error) { rejected = /contains 17 values; expected 18/.test(String(error?.message)); }
assert(rejected, "Core ML parent cardinality mismatch did not fail closed");

const layerNorm = await analyze(singleLayerFixture({
  inputShape: [2, 3], outputShape: [2, 3], typeField: 1350,
  params: concat(packedUint(1, [2, 3]), float32(2, 1e-5), message(3, weight([1, 1, 1, 1, 1, 1])), message(4, weight([0, 0, 0, 0, 0, 0]))),
}));
assert(layerNorm.ops[0].output_shapes[0].join("x") === "2x3", "Core ML LayerNorm output shape is not conserved");
assert(layerNorm.weight_integrity.status === "assessed" && layerNorm.weight_integrity.parameter_count === 2
  && layerNorm.weight_integrity.payload_bytes === 48, "Core ML LayerNorm WeightParams were not fully assessed");

const conv3dValues = Array.from({ length: 4 * 2 * 3 * 3 * 3 }, (_, index) => (index + 1) / 2048);
const conv3d = await analyze(singleLayerFixture({
  inputShape: [1, 2, 4, 4, 4], outputShape: [1, 4, 4, 4, 4], typeField: 1471,
  params: concat(
    uint(1, 4), uint(2, 2), uint(10, 1), uint(20, 3), uint(21, 3), uint(22, 3),
    uint(31, 1), uint(32, 1), uint(33, 1), uint(40, 1), uint(41, 1), uint(42, 1),
    uint(50, 1), message(60, weight(conv3dValues)), message(61, weight([0, 0, 0, 0])), uint(70, 2),
  ),
}));
assert(conv3d.total_macs === 13_824 && conv3d.ops[0].macs_status === "derived_exact_coreml_convolution_3d", "Core ML Conv3D exact MAC derivation is incorrect");
assert(conv3d.ops[0].output_shapes[0].join("x") === "1x4x4x4x4" && conv3d.weight_integrity.payload_bytes === 880,
  "Core ML Conv3D shape or WeightParams byte conservation is incorrect");

for (const [name, inputShape, outputShape, typeField, params, status] of [
  ["Permute", [2, 3, 4, 5, 6], [2, 3, 6, 4, 5], 310, packedUint(1, [0, 3, 1, 2]), "derived_coreml_permute_contract"],
  ["Transpose", [2, 3, 4], [4, 2, 3], 985, packedUint(1, [2, 0, 1]), "derived_coreml_transpose_contract"],
  ["Squeeze", [1, 2, 1, 3], [2, 3], 1120, packedInt(1, [0, -2]), "derived_coreml_squeeze_contract"],
  ["ExpandDims", [2, 3], [1, 2, 1, 3], 1125, packedInt(1, [0, 2]), "derived_coreml_expand_dims_contract"],
  ["FlattenTo2D", [2, 3, 4], [6, 4], 1130, sint(1, -1), "derived_coreml_flatten_to_2d_contract"],
  ["ReshapeStatic", [2, 3, 4], [4, 6], 1140, packedInt(1, [4, -1]), "derived_coreml_reshape_static_contract"],
  ["Pooling3D", [1, 2, 4, 5, 6], [1, 2, 2, 3, 3], 1465,
    concat(uint(1, 0), uint(2, 2), uint(3, 3), uint(4, 2), uint(5, 2), uint(6, 1), uint(7, 2), uint(15, 1)),
    "derived_coreml_pooling_3d_contract"],
  ["GlobalPooling3D", [1, 2, 4, 5, 6], [1, 2, 1, 1, 1], 1466, uint(1, 1), "derived_coreml_global_pooling_3d_contract"],
]) {
  const shaped = await analyze(operationFixture({ inputShapes: [inputShape], outputShapes: [outputShape], typeField, params }));
  assert(shaped.ops[0].output_shapes[0].join("x") === outputShape.join("x") && shaped.ops[0].shape_status === status,
    `Core ML ${name} shape contract is incorrect`);
  assert(shaped.ops[0].macs === 0 && shaped.ops[0].macs_status === "derived_non_mac_operation", `Core ML ${name} MAC classification is incorrect`);
}

const concatNd = await analyze(operationFixture({
  inputShapes: [[2, 3], [2, 5]], outputShapes: [[2, 8]], typeField: 980, params: sint(1, -1),
}));
assert(concatNd.ops[0].shape_status === "derived_coreml_concat_nd_contract" && concatNd.ops[0].output_shapes[0].join("x") === "2x8",
  "Core ML ConcatND axis conservation is incorrect");

const splitNd = await analyze(operationFixture({
  inputShapes: [[2, 8]], outputShapes: [[2, 3], [2, 5]], typeField: 975,
  params: concat(sint(1, -1), uint(2, 99), packedUint(3, [3, 5])),
}));
assert(splitNd.ops[0].shape_status === "derived_coreml_split_nd_contract"
  && splitNd.ops[0].output_shapes.map((shape) => shape.join("x")).join("/") === "2x3/2x5", "Core ML SplitND unequal split is incorrect");

const broadcastMatmul = await analyze(operationFixture({
  inputShapes: [[2, 3, 4], [1, 4, 5]], outputShapes: [[2, 3, 5]], typeField: 1045,
}));
assert(broadcastMatmul.total_macs === 120 && broadcastMatmul.ops[0].shape_status === "derived_coreml_batched_matmul_input_contract",
  "Core ML BatchedMatMul broadcast shape or MAC count is incorrect");

const embedding = await analyze(operationFixture({
  inputShapes: [[2, 1, 1, 1]], outputShapes: [[2, 3, 1, 1]], typeField: 150,
  params: concat(uint(1, 5), uint(2, 3), message(20, weight(Array(15).fill(0.25)))),
}));
assert(embedding.ops[0].shape_status === "derived_coreml_embedding_contract" && embedding.total_macs === 0,
  "Core ML Embedding lookup shape or MAC classification is incorrect");

const embeddingNd = await analyze(operationFixture({
  inputShapes: [[2, 4, 1]], outputShapes: [[2, 4, 3]], typeField: 1040,
  params: concat(uint(1, 5), uint(2, 3), message(20, weight(Array(15).fill(0.25)))),
}));
assert(embeddingNd.ops[0].shape_status === "derived_coreml_embedding_nd_contract" && embeddingNd.total_macs === 0,
  "Core ML EmbeddingND lookup shape or MAC classification is incorrect");

for (const [name, inputShapes, outputShapes, typeField, params, status] of [
  ["Broadcast", [[2, 1, 4], [1, 3, 4]], [[2, 3, 4]], 880, Buffer.alloc(0), "derived_coreml_broadcast_contract"],
  ["WhereBroadcast", [[2, 1], [1, 3], [2, 3]], [[2, 3]], 1330, Buffer.alloc(0), "derived_coreml_broadcast_contract"],
  ["Concat", [[2, 3, 4, 5, 6], [7, 3, 4, 5, 6]], [[9, 3, 4, 5, 6]], 320, uint(100, 1), "derived_coreml_concat_contract"],
  ["Split", [[6, 3, 4]], [[3, 3, 4], [3, 3, 4]], 330, uint(1, 2), "derived_coreml_split_contract"],
  ["GetShape", [[2, 3, 4]], [[3]], 1065, Buffer.alloc(0), "derived_coreml_get_shape_contract"],
  ["FillLike", [[2, 3, 4]], [[2, 3, 4]], 1080, Buffer.alloc(0), "derived_coreml_like_shape_contract"],
  ["ReshapeLike", [[2, 3], [3, 2]], [[3, 2]], 1135, Buffer.alloc(0), "derived_coreml_like_shape_contract"],
  ["BroadcastToLike", [[1, 3], [2, 3]], [[2, 3]], 1100, Buffer.alloc(0), "derived_coreml_like_shape_contract"],
  ["FillStatic", [], [[2, 3]], 1085, packedUint(2, [2, 3]), "derived_coreml_static_shape_contract"],
  ["BroadcastToStatic", [[1, 3]], [[2, 3]], 1105, packedUint(1, [2, 3]), "derived_coreml_static_shape_contract"],
  ["RandomNormalStatic", [], [[2, 3]], 1175, packedUint(4, [2, 3]), "derived_coreml_static_shape_contract"],
]) {
  const shaped = await analyze(operationFixture({ inputShapes, outputShapes, typeField, params }));
  assert(shaped.ops[0].shape_status === status
    && shaped.ops[0].output_shapes.map((shape) => shape.join("x")).join("/") === outputShapes.map((shape) => shape.join("x")).join("/"),
  `Core ML ${name} source-backed shape contract is incorrect`);
  assert(shaped.ops[0].macs === 0, `Core ML ${name} must not enter the matrix MAC ledger`);
}

for (const [name, inputShapes, outputShapes, typeField, params, status] of [
  ["Padding", [[2, 3, 4]], [[2, 6, 11]], 200, message(10, borderAmounts([[1, 2], [3, 4]])), "derived_coreml_padding_contract"],
  ["Upsample", [[2, 3, 4]], [[2, 6, 12]], 210, packedUint(1, [2, 3]), "derived_coreml_upsample_contract"],
  ["FractionalUpsample", [[2, 4, 3]], [[2, 6, 6]], 210, packedFloat(7, [1.5, 2]), "derived_coreml_upsample_contract"],
  ["ResizeBilinear", [[2, 3, 4]], [[2, 5, 6]], 211, packedUint(1, [5, 6]), "derived_coreml_resize_bilinear_contract"],
  ["CropResize", [[1, 2, 3, 4, 5], [7, 1, 4, 1, 1]], [[7, 2, 3, 6, 8]], 212, packedUint(1, [6, 8]), "derived_coreml_crop_resize_contract"],
  ["LegacyReduce", [[2, 3, 4, 5]], [[2, 3, 1, 1]], 280, uint(3, 1), "derived_coreml_reduce_contract"],
  ["Tile", [[2, 3]], [[2, 12]], 920, packedUint(1, [4]), "derived_coreml_tile_contract"],
  ["Stack", [[2, 3], [2, 3]], [[2, 2, 3]], 925, sint(1, 1), "derived_coreml_stack_contract"],
  ["Gather", [[2, 3, 4], [5, 6]], [[2, 5, 6, 4]], 930, sint(1, 1), "derived_coreml_gather_contract"],
  ["Scatter", [[2, 3, 4], [5, 6], [2, 5, 6, 4]], [[2, 3, 4]], 935, sint(1, 1), "derived_coreml_scatter_contract"],
  ["GatherND", [[4, 2, 3, 4], [6, 2]], [[6, 3, 4]], 940, Buffer.alloc(0), "derived_coreml_gather_nd_contract"],
  ["ScatterND", [[4, 2, 3, 4], [6, 2], [6, 3, 4]], [[4, 2, 3, 4]], 945, Buffer.alloc(0), "derived_coreml_scatter_nd_contract"],
  ["GatherAlongAxis", [[4, 4, 7], [4, 5, 7]], [[4, 5, 7]], 952, sint(1, 1), "derived_coreml_gather_along_axis_contract"],
  ["ScatterAlongAxis", [[2, 5, 6], [2, 2, 6], [2, 2, 6]], [[2, 5, 6]], 954, sint(1, -2), "derived_coreml_scatter_along_axis_contract"],
  ["RankPreservingReshape", [[20, 10, 5]], [[20, 2, 25]], 1150, packedInt(1, [0, 2, -1]), "derived_coreml_rank_preserving_reshape_contract"],
  ["ConstantPadding", [[2, 3]], [[3, 7]], 1155, packedUint(2, [0, 1, 4, 0]), "derived_coreml_constant_padding_contract"],
  ["ConstantPaddingOutputSize", [[20, 10]], [[21, 14]], 1155, concat(packedUint(2, [0, 21, 14, 0]), uint(3, 1)), "derived_coreml_constant_padding_contract"],
  ["ArgMax", [[2, 3, 4]], [[2, 4]], 1025, concat(sint(1, 1), uint(2, 1)), "derived_coreml_arg_max_contract"],
  ["ReduceND", [[2, 3, 4]], [[2, 4]], 1270, concat(packedInt(1, [-2]), uint(2, 0)), "derived_coreml_reduce_nd_contract"],
]) {
  const shaped = await analyze(operationFixture({ inputShapes, outputShapes, typeField, params }));
  assert(shaped.ops[0].shape_status === status
    && shaped.ops[0].output_shapes.map((shape) => shape.join("x")).join("/") === outputShapes.map((shape) => shape.join("x")).join("/"),
  `Core ML ${name} deterministic shape contract is incorrect`);
  assert(shaped.ops[0].macs === 0, `Core ML ${name} must stay outside the matrix MAC ledger`);
}

const topK = await analyze(operationFixture({
  inputShapes: [[2, 7, 4]], outputShapes: [[2, 3, 4], [2, 3, 4]], typeField: 1015,
  params: concat(sint(1, 1), uint(2, 3)),
}));
assert(topK.ops[0].shape_status === "derived_coreml_top_k_contract" && topK.ops[0].output_shapes.length === 2,
  "Core ML TopK static K output contracts are incorrect");

const dynamicTopK = await analyze(operationFixture({
  inputShapes: [[2, 7, 4], [1]], outputShapes: [[2, 3, 4], [2, 3, 4]], typeField: 1015,
  params: concat(sint(1, 1), uint(2, 6)),
}));
assert(dynamicTopK.ops[0].shape_status === "not_assessed", "Core ML dynamic TopK must not reuse the ignored serialized K attribute");

await expectRejected(operationFixture({ inputShapes: [[2, 3, 4]], outputShapes: [[2, 3, 4]], typeField: 985, params: packedUint(1, [0, 0, 2]) }),
  /axes must be a non-empty permutation/, "Core ML duplicate Transpose axes did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 8]], outputShapes: [[2, 3], [2, 4]], typeField: 975, params: concat(uint(2, 2), packedUint(3, [3, 4])) }),
  /splitSizes do not conserve/, "Core ML non-conserving SplitND did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3, 4]], outputShapes: [[5, 5]], typeField: 1140, params: packedInt(1, [5, 5]) }),
  /does not conserve input cardinality/, "Core ML non-conserving ReshapeStatic did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3, 4], [2, 6, 5]], outputShapes: [[2, 3, 5]], typeField: 1045 }),
  /incompatible contracted or broadcast dimensions/, "Core ML incompatible BatchedMatMul did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3], [4, 3]], outputShapes: [[2, 3]], typeField: 880 }),
  /not broadcast-compatible/, "Core ML incompatible broadcast did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[5, 3, 4]], outputShapes: [[2, 3, 4], [3, 3, 4]], typeField: 330, params: uint(1, 2) }),
  /does not uniformly divide/, "Core ML non-uniform legacy Split did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3], [4, 2]], outputShapes: [[4, 2]], typeField: 1135 }),
  /does not conserve input cardinality/, "Core ML ReshapeLike cardinality mismatch did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3, 4], [5], [2, 5, 3]], outputShapes: [[2, 3, 4]], typeField: 935, params: sint(1, 1) }),
  /updates do not match/, "Core ML invalid Scatter updates did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3, 4]], outputShapes: [[2]], typeField: 1270, params: packedInt(1, [1, -2]) }),
  /axes are invalid or repeated/, "Core ML duplicate reduction axes did not fail closed");
await expectRejected(operationFixture({ inputShapes: [[2, 3, 3]], outputShapes: [[2, 4, 6]], typeField: 210, params: packedFloat(7, [1.5, 2]) }),
  /does not derive integral positive spatial dimensions/, "Core ML fractional upsample with a non-integral output shape did not fail closed");

for (const [kind, expectedMacs, expectedParams] of [
  ["simple", 270, 3], ["gru", 810, 9], ["lstm", 1080, 15], ["bilstm", 2160, 30],
]) {
  const recurrent = await analyze(recurrentFixture(kind));
  assert(recurrent.total_macs === expectedMacs, `Core ML ${kind} exact recurrent matrix MACs are incorrect`);
  assert(recurrent.weight_integrity.status === "assessed" && recurrent.weight_integrity.parameter_count === expectedParams,
    `Core ML ${kind} WeightParams scan is incomplete`);
  assert(recurrent.ops[0].coreml_macs_definition?.includes("gate elementwise arithmetic"), `Core ML ${kind} MAC definition boundary is missing`);
}

console.log("Core ML legacy source-pinned shape/MAC, broadcast, split/reshape conservation, weights, and report conformance passed.");
