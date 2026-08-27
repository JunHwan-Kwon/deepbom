import { createHash } from "node:crypto";
import { File } from "node:buffer";
import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { scanCoreMlBlobFile } from "../web/lib/coreml-blob.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";

function assert(value, message) { if (!value) throw new Error(message); }
function varint(value) { let x = BigInt(value); const out = []; while (x > 127n) { out.push(Number(x & 127n) | 128); x >>= 7n; } out.push(Number(x)); return Buffer.from(out); }
function concat(...rows) { return Buffer.concat(rows.flat().filter(Boolean)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function packedUint(field, values) { return bytes(field, concat(values.map(varint))); }

function tensorType(dtype, shape, { omitDefaultRank = false } = {}) {
  const tensor = concat(uint(1, dtype), omitDefaultRank && shape.length === 0 ? null : uint(2, shape.length), shape.map((size) => message(3, message(1, uint(1, size)))));
  return message(1, tensor);
}
function named(name, dtype, shape, options) { return concat(string(1, name), message(2, tensorType(dtype, shape, options))); }
function nameBinding(name) { return message(1, concat(string(1, name), uint(99, 1))); }
function blobValue(dtype, shape, fileName, offset) { return concat(message(2, tensorType(dtype, shape)), message(5, concat(string(1, fileName), uint(2, offset)))); }
function immediateBytesValue(dtype, shape, payload) {
  const tensorValue = concat(message(7, bytes(1, payload)), uint(98, 1));
  const immediate = concat(message(1, tensorValue), uint(99, 1));
  return concat(message(2, tensorType(dtype, shape)), message(3, immediate));
}
function immediateStringValue(value) { return concat(message(2, tensorType(2, [], { omitDefaultRank: true })), message(3, message(1, message(4, string(1, value))))); }
function immediateStringListValue(values) {
  const listType = message(2, concat(message(1, tensorType(2, [], { omitDefaultRank: true })), message(2, message(1, uint(1, values.length)))));
  return concat(message(2, listType), message(3, message(3, concat(values.map((value) => message(1, immediateStringValue(value)))))));
}
function boolValue(value) { return concat(message(2, tensorType(1, [])), message(3, message(1, message(3, uint(1, value ? 1 : 0))))); }
function int32Value(value) { return concat(message(2, tensorType(23, [])), message(3, message(1, message(2, uint(1, value))))); }
function int32ListValue(values) { return concat(message(2, tensorType(23, [values.length])), message(3, message(1, message(2, packedUint(1, values))))); }
function valueBinding(value) { return message(1, concat(message(2, value), uint(99, 1))); }
function argument(binding) { return message(2, concat(string(1, binding[0]), message(2, binding[1]))); }
function operation(type, inputs, outputs, blocks = []) { return concat(string(1, type), inputs.map(argument), outputs.map((output) => message(3, named(...output))), blocks.map((block) => message(4, block))); }

function mlProgramModel({ forwardReference = false } = {}) {
  const weightShape = [2, 1, 3, 3];
  const constOp = operation("const", [["val", valueBinding(blobValue(11, weightShape, "@model_path/weights/weight.bin", 64))]], [["weight", 11, weightShape]]);
  const convOp = operation("conv", [["x", nameBinding("image")], ["weight", nameBinding("weight")]], [["output", 11, [1, 2, 2, 2]]]);
  const block = concat(string(2, "output"), [forwardReference ? convOp : constOp, forwardReference ? constOp : convOp].map((op) => message(3, op)));
  const fn = concat(message(1, named("image", 11, [1, 1, 4, 4])), string(2, "CoreML7"), message(3, concat(string(1, "CoreML7"), message(2, block))));
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 65568)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = concat(message(1, feature("image", [1, 1, 4, 4])), message(10, feature("output", [1, 2, 2, 2])));
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function matmulProgramModel(outputShape = [2, 4], { xShape = [2, 3], yShape = [4, 3] } = {}) {
  const matmul = operation("matmul", [
    ["x", nameBinding("x")],
    ["y", nameBinding("y")],
    ["transpose_y", valueBinding(boolValue(true))],
  ], [["output", 11, outputShape]]);
  const block = concat(string(2, "output"), message(3, matmul));
  const fn = concat(
    message(1, named("x", 11, xShape)),
    message(1, named("y", 11, yShape)),
    string(2, "CoreML7"),
    message(3, concat(string(1, "CoreML7"), message(2, block))),
  );
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 65568)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = concat(message(1, feature("x", xShape)), message(1, feature("y", yShape)), message(10, feature("output", outputShape)));
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function milCostProgramModel({ opType, functionInputs, opInputs, outputs, opset = "CoreML8" }) {
  const op = operation(opType, opInputs, outputs);
  const block = concat(outputs.map((output) => string(2, output[0])), message(3, op));
  const fn = concat(
    functionInputs.map((input) => message(1, named(...input))),
    string(2, opset),
    message(3, concat(string(1, opset), message(2, block))),
  );
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 65568)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = concat(
    functionInputs.map(([name, , shape]) => message(1, feature(name, shape))),
    outputs.map(([name, , shape]) => message(10, feature(name, shape))),
  );
  return concat(uint(1, opset === "CoreML8" ? 9 : 8), message(2, description), message(502, program));
}

function nestedBlockProgramModel() {
  const nestedMatmul = operation("matmul", [["x", nameBinding("x")], ["y", nameBinding("y")]], [["inner", 11, [2, 4]]]);
  const nestedBlock = concat(string(2, "inner"), message(3, nestedMatmul));
  const outer = operation("identity", [["x", nameBinding("x")]], [["output", 11, [2, 3]]], [nestedBlock]);
  const block = concat(string(2, "output"), message(3, outer));
  const fn = concat(message(1, named("x", 11, [2, 3])), message(1, named("y", 11, [3, 4])), string(2, "CoreML7"), message(3, concat(string(1, "CoreML7"), message(2, block))));
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 65568)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = concat(message(1, feature("x", [2, 3])), message(1, feature("y", [3, 4])), message(10, feature("output", [2, 3])));
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function condProgramModel() {
  const trueMatmul = operation("matmul", [["x", nameBinding("x")], ["y", nameBinding("y")]], [["true_value", 11, [2, 4]]]);
  const falseMatmul = operation("matmul", [["x", nameBinding("x")], ["y", nameBinding("y")]], [["false_value", 11, [2, 4]]]);
  const trueBlock = concat(string(2, "true_value"), message(3, trueMatmul));
  const falseBlock = concat(string(2, "false_value"), message(3, falseMatmul));
  const conditional = operation("cond", [["pred", nameBinding("pred")]], [["output", 11, [2, 4]]], [trueBlock, falseBlock]);
  const block = concat(string(2, "output"), message(3, conditional));
  const fn = concat(
    message(1, named("pred", 1, [])), message(1, named("x", 11, [2, 3])), message(1, named("y", 11, [3, 4])),
    string(2, "CoreML7"), message(3, concat(string(1, "CoreML7"), message(2, block))),
  );
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape, dtype = 65568) => message(5, concat(packedUint(1, shape), uint(2, dtype)));
  const feature = (name, shape, dtype) => concat(string(1, name), message(3, array(shape, dtype)));
  const description = concat(
    message(1, feature("pred", [], 131104)), message(1, feature("x", [2, 3])), message(1, feature("y", [3, 4])),
    message(10, feature("output", [2, 4])),
  );
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function scalarRankProgramModel() {
  const identity = operation("identity", [["x", nameBinding("scalar")]], [["output", 23, [], { omitDefaultRank: true }]]);
  const block = concat(string(2, "output"), message(3, identity));
  const fn = concat(
    message(1, named("scalar", 23, [], { omitDefaultRank: true })),
    string(2, "CoreML7"),
    message(3, concat(string(1, "CoreML7"), message(2, block))),
  );
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 131104)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = concat(message(1, feature("scalar", [])), message(10, feature("output", [])));
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function immediateBytesProgramModel() {
  const constOp = operation("const", [["val", valueBinding(immediateBytesValue(10, [3], Buffer.from([0, 0, 0, 60, 0, 64])))]], [["output", 10, [3]]]);
  const block = concat(string(2, "output"), message(3, constOp));
  const fn = concat(string(2, "CoreML7"), message(3, concat(string(1, "CoreML7"), message(2, block))));
  const labels = concat(string(1, "labels"), message(2, immediateStringListValue(["alpha", "beta"])));
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))), message(4, labels));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, 65552)));
  const feature = (name, shape) => concat(string(1, name), message(3, array(shape)));
  const description = message(10, feature("output", [3]));
  return concat(uint(1, 8), message(2, description), message(502, program));
}

function compressionProgramModel(kind, { invalidBlock = false, invalidPalette = false, vector = false, invalidSparse = false, invalidPadding = false } = {}) {
  let op;
  let outputShape;
  let outputDtype;
  if (kind === "blockwise") {
    const scaleShape = invalidBlock ? [3, 1] : [2, 1];
    op = operation("constexpr_blockwise_shift_scale", [
      ["data", valueBinding(immediateBytesValue(35, [4, 4], Buffer.alloc(8, 0x21)))],
      ["scale", valueBinding(immediateBytesValue(11, scaleShape, Buffer.alloc(scaleShape[0] * 4)))],
    ], [["output", 11, [4, 4]]]);
    outputShape = [4, 4];
    outputDtype = 11;
  } else if (kind === "lut") {
    const vectorSize = vector ? 2 : 1;
    const paletteCount = invalidPalette ? 2 : 4;
    const lutShape = [1, 1, paletteCount, vectorSize];
    const outputAxis = vector ? 1 : null;
    outputShape = vector ? [2, 4] : [2, 2];
    outputDtype = 10;
    op = operation("constexpr_lut_to_dense", [
      ["indices", valueBinding(immediateBytesValue(36, [2, 2], Buffer.from([0x1b])))],
      ["lut", valueBinding(immediateBytesValue(10, lutShape, Buffer.alloc(paletteCount * vectorSize * 2)))],
      ...(outputAxis == null ? [] : [["vector_axis", valueBinding(int32Value(outputAxis))]]),
    ], [["output", outputDtype, outputShape]]);
  } else {
    outputShape = [2, 3];
    outputDtype = 21;
    const mask = invalidPadding ? 0xa5 : invalidSparse ? 0x05 : 0x25;
    op = operation("constexpr_sparse_to_dense", [
      ["nonzero_data", valueBinding(immediateBytesValue(21, [3], Buffer.from([1, 2, 3])))],
      ["mask", valueBinding(immediateBytesValue(37, outputShape, Buffer.from([mask])))],
    ], [["output", outputDtype, outputShape]]);
  }
  const block = concat(string(2, "output"), message(3, op));
  const fn = concat(string(2, "CoreML8"), message(3, concat(string(1, "CoreML8"), message(2, block))));
  const program = concat(uint(1, 1), message(2, concat(string(1, "main"), message(2, fn))));
  const array = (shape) => message(5, concat(packedUint(1, shape), uint(2, outputDtype === 10 ? 65552 : 65568)));
  const description = message(10, concat(string(1, "output"), message(3, array(outputShape))));
  return concat(uint(1, 9), message(2, description), message(502, program));
}

function blobFile(dtype, payload, paddingBits = 0, { sentinel = 0xdeadbeef } = {}) {
  const data = Buffer.from(payload);
  const result = Buffer.alloc(128 + data.length);
  result.writeUInt32LE(1, 0);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(sentinel, 64);
  result.writeUInt32LE(dtype, 68);
  result.writeBigUInt64LE(BigInt(data.length), 72);
  result.writeBigUInt64LE(128n, 80);
  result.writeBigUInt64LE(BigInt(paddingBits), 88);
  data.copy(result, 128);
  return result;
}

function floatPayload(values) { const result = Buffer.alloc(values.length * 4); values.forEach((value, index) => result.writeFloatLE(value, index * 4)); return result; }
function packageFile(payload, name, relativePath) { const file = new File([payload], name); Object.defineProperty(file, "webkitRelativePath", { value: relativePath }); return file; }

const weights = floatPayload(Array.from({ length: 18 }, (_, index) => (index + 1) / 16));
const model = mlProgramModel();
const parsed = (await readCoreMlModelFile(new File([model], "model.mlmodel"))).analysis;
assert(parsed.coreml?.model_type === "mlProgram", "Core ML ML Program model type was not retained");
assert(parsed.operator_count === 2 && parsed.tensor_count === 3, "Core ML MIL SSA graph inventory is incorrect");
assert(parsed.total_macs === 72 && parsed.mac_assessment?.status === "assessed_all_decoded_compute_ops", "Core ML MIL conv MAC derivation is incorrect");
assert(parsed.coreml_blob_references?.length === 1 && parsed.weight_integrity?.status.includes("package_blob_binding_required"), "Core ML MIL external blob boundary is not explicit");

const convTransposeParsed = (await readCoreMlModelFile(new File([milCostProgramModel({
  opType: "conv_transpose",
  functionInputs: [["x", 11, [1, 2, 3]], ["weight", 11, [2, 3, 3]]],
  opInputs: [
    ["x", nameBinding("x")], ["weight", nameBinding("weight")],
    ["strides", valueBinding(int32ListValue([2]))], ["dilations", valueBinding(int32ListValue([1]))],
    ["pad", valueBinding(int32ListValue([1, 1]))], ["pad_type", valueBinding(immediateStringValue("custom"))],
    ["groups", valueBinding(int32Value(1))],
  ],
  outputs: [["output", 11, [1, 3, 5]]],
})], "conv-transpose.mlmodel"))).analysis;
assert(convTransposeParsed.total_macs === 42
  && convTransposeParsed.ops[0].macs_status === "derived_exact_mil_conv_transpose_overlap",
"Core ML MIL conv_transpose must count exact retained input/kernel pairs after serialized cropping");

const einsumParsed = (await readCoreMlModelFile(new File([milCostProgramModel({
  opType: "einsum",
  functionInputs: [["x", 11, [2, 3, 4]], ["y", 11, [4, 3, 5]]],
  opInputs: [
    ["values", concat(nameBinding("x"), nameBinding("y"))],
    ["equation", valueBinding(immediateStringValue("chw,whr->chr"))],
  ],
  outputs: [["output", 11, [2, 3, 5]]],
})], "einsum.mlmodel"))).analysis;
assert(einsumParsed.total_macs === 120 && einsumParsed.ops[0].macs_status === "derived_exact_mil_einsum",
"Core ML MIL einsum must use the source-supported equation and exact contracted width");

const gruParsed = (await readCoreMlModelFile(new File([milCostProgramModel({
  opType: "gru",
  functionInputs: [["x", 11, [4, 2, 3]], ["initial_h", 11, [2, 5]], ["weight_ih", 11, [15, 3]], ["weight_hh", 11, [15, 5]]],
  opInputs: [["x", nameBinding("x")], ["initial_h", nameBinding("initial_h")], ["weight_ih", nameBinding("weight_ih")], ["weight_hh", nameBinding("weight_hh")]],
  outputs: [["sequence", 11, [1, 2, 5]], ["hidden", 11, [2, 5]]],
})], "gru.mlmodel"))).analysis;
assert(gruParsed.total_macs === 960 && gruParsed.ops[0].macs_status === "derived_exact_mil_gru",
"Core ML MIL GRU must count every source-defined input and recurrent gate contraction across sequence and batch");

const attentionParsed = (await readCoreMlModelFile(new File([milCostProgramModel({
  opType: "scaled_dot_product_attention",
  functionInputs: [["query", 11, [2, 4, 3, 8]], ["key", 11, [2, 4, 5, 8]], ["value", 11, [2, 4, 5, 6]]],
  opInputs: [["query", nameBinding("query")], ["key", nameBinding("key")], ["value", nameBinding("value")]],
  outputs: [["output", 11, [2, 4, 3, 6]]],
})], "attention.mlmodel"))).analysis;
assert(attentionParsed.total_macs === 1_680
  && attentionParsed.ops[0].macs_status === "derived_exact_mil_scaled_dot_product_attention",
"Core ML MIL scaled_dot_product_attention must count QK and attention-value contractions with exact batch and sequence extents");
const scalarRankParsed = (await readCoreMlModelFile(new File([scalarRankProgramModel()], "scalar-rank.mlmodel"))).analysis;
assert(scalarRankParsed.tensors.length === 2 && scalarRankParsed.tensors.every((tensor) => tensor.shape.length === 0), "Omitted proto3 MIL rank=0 did not retain the scalar tensor contract");
const immediateBytesParsed = (await readCoreMlModelFile(new File([immediateBytesProgramModel()], "immediate-bytes.mlmodel"))).analysis;
assert(immediateBytesParsed.tensors[0].immediate_value?.byte_length === 6
  && immediateBytesParsed.tensors[0].immediate_value.logical_count === 3
  && immediateBytesParsed.weight_integrity?.status === "partial"
  && immediateBytesParsed.coreml?.ml_program?.attributes?.labels?.immediate?.kind === "list"
  && immediateBytesParsed.coreml.ml_program.attributes.labels.immediate.count === 2,
"MIL byte-backed FLOAT16 immediate cardinality or forward-compatible unknown-field handling is incorrect");

const blockwiseCompressionModel = compressionProgramModel("blockwise");
const blockwiseCompression = (await readCoreMlModelFile(new File([blockwiseCompressionModel], "blockwise.mlmodel"))).analysis;
const blockwiseLedger = blockwiseCompression.coreml?.mil_compression_contract;
const blockwiseRow = blockwiseLedger?.transforms?.[0];
assert(blockwiseLedger?.status === "assessed_exact_serialized_contracts" && blockwiseLedger.exact_contract_count === 1
  && blockwiseRow?.representation === "blockwise_affine" && blockwiseRow.logical_output_elements === 16
  && blockwiseRow.scale_elements === 2 && JSON.stringify(blockwiseRow.block_shape) === "[2,4]",
"Core ML iOS 18 blockwise affine shape/cardinality contract is incorrect");

const lutCompressionModel = compressionProgramModel("lut", { vector: true });
const lutCompression = (await readCoreMlModelFile(new File([lutCompressionModel], "lut.mlmodel"))).analysis;
const lutRow = lutCompression.coreml?.mil_compression_contract?.transforms?.[0];
assert(lutRow?.representation === "blockwise_lut_palettization" && lutRow.index_bits === 2
  && lutRow.palette_count === 4 && lutRow.vector_size === 2 && lutRow.vector_axis === 1
  && lutRow.logical_index_elements === 4 && lutRow.logical_output_elements === 8,
"Core ML iOS 18 LUT palette/vector cardinality contract is incorrect");

const sparseCompressionModel = compressionProgramModel("sparse");
const sparseCompression = (await readCoreMlModelFile(new File([sparseCompressionModel], "sparse.mlmodel"))).analysis;
const sparseRow = sparseCompression.coreml?.mil_compression_contract?.transforms?.[0];
assert(sparseRow?.status === "assessed_exact_serialized_contract"
  && sparseRow.mask_population === 3 && sparseRow.stored_nonzero_elements === 3
  && sparseRow.logical_output_elements === 6 && sparseRow.mask_population_status === "assessed_exact_immediate_payload",
"Core ML iOS 18 sparse mask population and nonzero-data cardinality contract is incorrect");

blockwiseCompression.model_sha256 = createHash("sha256").update(blockwiseCompressionModel).digest("hex");
const blockwiseReport = buildEngineeringReport(blockwiseCompression, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(blockwiseReport.includes("Core ML Serialized Compression Contracts")
  && blockwiseReport.includes("blockwise_affine") && blockwiseReport.includes("[2, 4]"),
"Core ML serialized compression contract is missing from the engineering report");
const blockwiseEvidence = buildEngineeringEvidenceDocument(blockwiseCompression, {
  reportContext: { generatedAt: "2026-08-07T00:00:00.000Z", identity: { filename: "blockwise.mlmodel", format: "coreml", sha256: blockwiseCompression.model_sha256 } },
  rawEvidenceContext: { identity: { filename: "blockwise.mlmodel", format: "coreml", sha256: blockwiseCompression.model_sha256 } },
  mlBomDocument: buildMlBomDocument(blockwiseCompression, { hash: blockwiseCompression.model_sha256 }),
});
assert(blockwiseEvidence.evidence?.conformance_report?.status === "pass", "Core ML compression evidence failed report/export conformance");

let invalidBlockRejected = false;
try { await readCoreMlModelFile(new File([compressionProgramModel("blockwise", { invalidBlock: true })], "invalid-blockwise.mlmodel")); }
catch (error) { invalidBlockRejected = /block divisibility constraints/.test(String(error?.message)); }
assert(invalidBlockRejected, "Core ML non-divisible blockwise scale shape did not fail closed");
let invalidPaletteRejected = false;
try { await readCoreMlModelFile(new File([compressionProgramModel("lut", { invalidPalette: true })], "invalid-lut.mlmodel")); }
catch (error) { invalidPaletteRejected = /palette cardinality/.test(String(error?.message)); }
assert(invalidPaletteRejected, "Core ML LUT/index bit-width contradiction did not fail closed");
let invalidSparseRejected = false;
try { await readCoreMlModelFile(new File([compressionProgramModel("sparse", { invalidSparse: true })], "invalid-sparse.mlmodel")); }
catch (error) { invalidSparseRejected = /mask population does not equal/.test(String(error?.message)); }
assert(invalidSparseRejected, "Core ML sparse mask/nonzero-data population contradiction did not fail closed");
let invalidSparsePaddingRejected = false;
try { await readCoreMlModelFile(new File([compressionProgramModel("sparse", { invalidPadding: true })], "invalid-sparse-padding.mlmodel")); }
catch (error) { invalidSparsePaddingRejected = /non-zero padding bits/.test(String(error?.message)); }
assert(invalidSparsePaddingRejected, "Core ML sparse UINT1 immediate padding corruption did not fail closed");

const matmulParsed = (await readCoreMlModelFile(new File([matmulProgramModel()], "matmul.mlmodel"))).analysis;
assert(matmulParsed.total_macs === 24 && matmulParsed.ops[0].macs_status === "derived_exact_mil_matmul", "Core ML MIL transposed matmul MAC derivation is incorrect");
const hugeWidth = 134_217_728;
const hugeMatmul = (await readCoreMlModelFile(new File([matmulProgramModel([hugeWidth, hugeWidth], {
  xShape: [hugeWidth, 1], yShape: [hugeWidth, 1],
})], "huge-matmul.mlmodel"))).analysis;
assert(hugeMatmul.total_macs === null
  && hugeMatmul.ops[0].macs_decimal === "18014398509481984"
  && hugeMatmul.ops[0].macs_status === "derived_exact_decimal_only_mil_matmul"
  && hugeMatmul.mac_assessment?.complete_macs_decimal === "18014398509481984"
  && hugeMatmul.mac_assessment?.safe_number_mirror_status === "exact_decimal_only",
"Core ML MIL MAC products above 2^53 must remain exact decimal evidence instead of becoming unassessed.");
let matmulShapeRejected = false;
try { await readCoreMlModelFile(new File([matmulProgramModel([2, 3])], "bad-matmul.mlmodel")); } catch (error) { matmulShapeRejected = /matmul output ValueType contradicts/.test(String(error?.message)); }
assert(matmulShapeRejected, "Core ML MIL matmul output-shape contradiction did not fail closed");
const nestedModel = nestedBlockProgramModel();
const nestedParsed = (await readCoreMlModelFile(new File([nestedModel], "nested.mlmodel"))).analysis;
nestedParsed.model_sha256 = createHash("sha256").update(nestedModel).digest("hex");
assert(nestedParsed.total_macs === null && nestedParsed.mac_assessment?.status === "partial_control_flow_execution_count_not_reconstructed"
  && nestedParsed.mac_assessment?.nested_block_operation_count === 1, "Nested MIL block MACs were incorrectly flattened into an exact total");
assert(nestedParsed.tensor_liveness?.peak_bytes === null && nestedParsed.tensor_liveness?.status === "not_assessed_unsupported_nested_operation", "Unsupported nested MIL operation emitted a false exact peak");
const nestedScopes = nestedParsed.coreml?.mil_scope_intrinsic_cost;
assert(nestedScopes?.schema === "deepbom.coreml.mil_scope_intrinsic_cost.v1" && nestedScopes.scope_count === 2 && nestedScopes.nested_scope_count === 1, "Nested MIL scope inventory is incomplete");
const rootScope = nestedScopes.scope_rows.find((row) => row.scope_class === "function_root");
const bodyScope = nestedScopes.scope_rows.find((row) => row.scope_class === "nested_block");
assert(rootScope?.complete_nominal_macs_decimal === "0" && rootScope.complete_output_payload_bytes_decimal === "24"
  && rootScope.scope_local_liveness?.peak_bytes === 48, "MIL root one-invocation cost or scope-local liveness is incorrect");
assert(bodyScope?.complete_nominal_macs_decimal === "24" && bodyScope.complete_output_payload_bytes_decimal === "32"
  && bodyScope.scope_local_liveness?.peak_bytes === 104, "MIL nested-block one-invocation cost or scope-local liveness is incorrect");
const nestedReport = buildEngineeringReport(nestedParsed, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(nestedReport.includes("no total is emitted without branch/loop execution counts")
  && nestedReport.includes("not_assessed_unsupported_nested_operation")
  && nestedReport.includes("MIL One-Invocation Scope Ledger") && nestedReport.includes("24 complete"), "Core ML report hides the nested-block MAC, liveness, or no-double-count boundary");
const nestedEvidence = buildEngineeringEvidenceDocument(nestedParsed, {
  reportContext: { generatedAt: "2026-08-07T00:00:00.000Z", identity: { filename: "nested.mlmodel", format: "coreml", sha256: nestedParsed.model_sha256 } },
  rawEvidenceContext: { identity: { filename: "nested.mlmodel", format: "coreml", sha256: nestedParsed.model_sha256 } },
  mlBomDocument: buildMlBomDocument(nestedParsed, { hash: nestedParsed.model_sha256 }),
});
assert(nestedEvidence.evidence?.conformance_report?.status === "pass", "Nested MIL scope evidence failed full report/export conformance");

const condModel = condProgramModel();
const condParsed = (await readCoreMlModelFile(new File([condModel], "cond.mlmodel"))).analysis;
assert(condParsed.tensor_liveness?.status === "assessed_static_control_flow_peak_envelope"
  && condParsed.tensor_liveness.peak_bytes === 137
  && condParsed.tensor_liveness.control_flow_scope_count === 3,
`Core ML cond branch-max liveness envelope is ${condParsed.tensor_liveness?.peak_bytes} B (${condParsed.tensor_liveness?.status}); expected 137 B: ${JSON.stringify(condParsed.tensor_liveness)}`);
assert(condParsed.coreml?.mil_scope_intrinsic_cost?.scope_rows.filter((row) => row.scope_class === "nested_block").every((row) => row.scope_local_liveness?.peak_bytes === 104),
"Core ML cond branch-local liveness is not independently conserved");

let forwardRejected = false;
try { await readCoreMlModelFile(new File([mlProgramModel({ forwardReference: true })], "invalid.mlmodel")); } catch (error) { forwardRejected = /not defined before use/.test(String(error?.message)); }
assert(forwardRejected, "Core ML MIL forward SSA reference did not fail closed");

const directBlob = new File([blobFile(2, weights)], "weight.bin");
const scan = await scanCoreMlBlobFile(directBlob, [{ metadata_offset: 64, tensor_index: 1, tensor_name: "weight", dtype: "FLOAT32", shape: [2, 1, 3, 3] }]);
assert(scan.status === "assessed" && scan.records[0].value_count === 18, "Core ML blob F32 cardinality is incorrect");
assert(scan.records[0].numerical_integrity.payload_sha256 === createHash("sha256").update(weights).digest("hex"), "Core ML blob payload SHA-256 is incorrect");
assert(scan.nonfinite_value_count === 0 && scan.records[0].numerical_integrity.finite_min === 1 / 16, "Core ML blob F32 numerical statistics are incorrect");

const int4 = await scanCoreMlBlobFile(new File([blobFile(8, Buffer.from([0x08, 0x07]), 4)], "int4.bin"), [{ metadata_offset: 64, tensor_index: 0, tensor_name: "int4", dtype: "INT4", shape: [3] }]);
const int4Stats = int4.records[0].numerical_integrity;
assert(int4Stats.decoded_value_count === 3 && int4Stats.finite_min === -8 && int4Stats.finite_max === 7 && int4Stats.zero_count === 1, "Core ML signed INT4 little-endian bit decoding is incorrect");

let sentinelRejected = false;
try { await scanCoreMlBlobFile(new File([blobFile(2, weights, 0, { sentinel: 0 })], "bad.bin")); } catch (error) { sentinelRejected = /invalid sentinel/.test(String(error?.message)); }
assert(sentinelRejected, "Core ML blob sentinel corruption did not fail closed");

const manifest = Buffer.from(JSON.stringify({
  fileFormatVersion: "1.0.0",
  rootModelIdentifier: "root",
  itemInfoEntries: { root: { path: "com.example", name: "Model", author: "DeepBOM", description: "Synthetic ML Program" } },
}));
const files = [
  packageFile(manifest, "Manifest.json", "Fixture.mlpackage/Manifest.json"),
  packageFile(model, "model.mlmodel", "Fixture.mlpackage/Data/com.example/model.mlmodel"),
  packageFile(blobFile(2, weights), "weight.bin", "Fixture.mlpackage/Data/com.example/weights/weight.bin"),
];
const bundled = (await readArtifactBundle(files)).analysis;
assert(bundled.artifact_bundle?.kind === "coreml_mlpackage", "Core ML package identity was not retained");
assert(bundled.weight_integrity?.status === "assessed" && bundled.weight_integrity.assessed_parameter_count === 1, "Core ML package blob did not bind to the MIL tensor");
assert(bundled.weight_integrity.payload_bytes === weights.length && bundled.weight_integrity.nonfinite_value_count === 0, "Core ML package blob aggregate is incorrect");
assert(bundled.size_breakdown?.status === "assessed" && bundled.size_breakdown.constant_bytes === weights.length
  && bundled.size_breakdown.constant_bytes + bundled.size_breakdown.structure_overhead_bytes === bundled.file_size, "Core ML package byte conservation is incorrect");
assert(bundled.tensor_liveness?.status === "assessed" && bundled.tensor_liveness.peak_bytes === 96, "Core ML MIL static activation liveness is incorrect");
assert(bundled.tensors[1].numerical_integrity?.payload_sha256 === createHash("sha256").update(weights).digest("hex"), "Core ML package tensor did not retain blob integrity evidence");
const report = buildEngineeringReport(bundled, { generatedAt: "2026-08-07T00:00:00.000Z" });
assert(report.includes("72 MACs") && report.includes("Core ML Serialized Constant Numerical Integrity") && report.includes("Pinned MIL.proto")
  && report.includes("Core ML Static Resource Accounting") && report.includes("96 B"), `Core ML ML Program report omits exact graph, payload, memory, or source evidence: mac=${report.includes("72 MACs")} integrity=${report.includes("Core ML Serialized Constant Numerical Integrity")} source=${report.includes("Pinned MIL.proto")} memory=${report.includes("96 B")}`);
assert(!buildFindingsRegister(bundled).some((item) => item.id === "EA-CML-0003"), "Fully bound Core ML package emitted a false partial-coverage finding");

console.log("Core ML ML Program SSA, shape/MAC, package blob, F32/INT4, digest, and fail-closed checks passed.");
