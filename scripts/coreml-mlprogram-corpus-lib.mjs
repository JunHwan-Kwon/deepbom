import { createHash } from "node:crypto";
import { File } from "node:buffer";
import { readFile } from "node:fs/promises";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { COREML_FORMAT_SOURCE } from "../web/lib/coreml-metadata-adapter.js";
import { canonicalJson } from "../web/lib/report-utils.js";

export const COREML_MLPROGRAM_CORPUS_PATH = "corpus/coreml-mlprogram-contract-corpus.v1.json";
export const COREML_MLPROGRAM_CORPUS_SCHEMA = "deepbom.coreml_mlprogram_contract_corpus.v1";
export const COREML_MLPROGRAM_RECEIPT_SCHEMA = "deepbom.coreml_mlprogram_contract_corpus_receipt.v1";

export async function readCoreMlProgramCorpus(filename = COREML_MLPROGRAM_CORPUS_PATH) {
  return validateCoreMlProgramCorpus(JSON.parse(await readFile(filename, "utf8")));
}

export function validateCoreMlProgramCorpus(manifest, { requireBaselines = false } = {}) {
  if (manifest?.schema !== COREML_MLPROGRAM_CORPUS_SCHEMA || manifest.format !== "coreml_mlpackage") {
    throw new Error(`Unsupported Core ML MLProgram corpus schema ${manifest?.schema || "missing"}.`);
  }
  if (manifest.generator_source?.repository !== COREML_FORMAT_SOURCE.repository
    || !/^[a-f0-9]{40}$/.test(String(manifest.generator_source?.revision || ""))
    || manifest.generator_source?.model_proto_sha256 !== COREML_FORMAT_SOURCE.model_proto_sha256
    || manifest.generator_source?.feature_types_proto_sha256 !== COREML_FORMAT_SOURCE.feature_types_proto_sha256
    || manifest.generator_source?.compression_ios18_definition_sha256 !== COREML_FORMAT_SOURCE.mil_compression_ios18_definition_sha256) {
    throw new Error("Core ML corpus generator provenance is not immutable or its contract-defining source bytes differ from the analyzer pin.");
  }
  if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length !== 5 || manifest.artifact_count !== 5) {
    throw new Error("Core ML MLProgram corpus must contain exactly five contract fixtures.");
  }
  const expected = new Set(["static_external_blob", "enumerated_shape", "bounded_shape_range", "blockwise_affine_compression", "lut_palettization_compression"]);
  const ids = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) throw new Error(`Invalid or duplicate Core ML fixture id ${artifact.id || "missing"}.`);
    if (!expected.delete(artifact.contract_class)) throw new Error(`${artifact.id}: contract class is invalid or duplicated.`);
    if (artifact.provenance_class !== "source_pinned_generated_contract_fixture" || artifact.ecosystem_prevalence_claim !== false) {
      throw new Error(`${artifact.id}: generated-fixture evidence boundary is not explicit.`);
    }
    if (requireBaselines) validateBaseline(artifact);
    ids.add(artifact.id);
  }
  if (expected.size) throw new Error(`Core ML corpus is missing ${[...expected].join(", ")}.`);
  return manifest;
}

function validateBaseline(artifact) {
  const baseline = artifact.baseline;
  if (!baseline || baseline.status !== "assessed" || baseline.model_type !== "mlProgram" || baseline.contract_class !== artifact.contract_class) {
    throw new Error(`${artifact.id}: assessed MLProgram baseline is missing or mismatched.`);
  }
  if (!Array.isArray(baseline.files) || baseline.files.length < 2 || baseline.files.some((row) => !row.path || !Number.isSafeInteger(row.size_bytes) || row.size_bytes < 1 || !isSha256(row.sha256))) {
    throw new Error(`${artifact.id}: generated package file identity is invalid.`);
  }
  for (const key of ["bundle_sha256", "analysis_receipt_sha256"]) if (!isSha256(baseline[key])) throw new Error(`${artifact.id}: ${key} is invalid.`);
}

export async function analyzeCoreMlProgramCorpusArtifact(artifact) {
  const generated = generateFixture(artifact);
  const browserFiles = generated.files.map((row) => packageFile(row.bytes, row.path));
  const { analysis } = await readArtifactBundle(browserFiles);
  const files = generated.files.map((row) => ({ path: row.path, size_bytes: row.bytes.length, sha256: sha256(row.bytes) }));
  const inputFlexibility = summarizeFlexibility(analysis.inputs?.[0]?.constraints?.flexibility);
  const outputFlexibility = summarizeFlexibility(analysis.outputs?.[0]?.constraints?.flexibility);
  const receiptBody = {
    artifact_id: artifact.id,
    contract_class: artifact.contract_class,
    provenance_class: artifact.provenance_class,
    source_revision: COREML_FORMAT_SOURCE.source_commit,
    files,
    bundle_sha256: analysis.artifact_bundle?.bundle_sha256 || null,
    status: analysis.artifact_bundle?.kind === "coreml_mlpackage" ? "assessed" : "invalid_bundle_kind",
    model_type: analysis.coreml?.model_type || null,
    specification_version: analysis.coreml?.specification_version ?? null,
    operator_count: Number(analysis.operator_count || 0),
    tensor_count: Number(analysis.tensor_count || 0),
    total_macs: analysis.total_macs ?? null,
    mac_assessment_status: analysis.mac_assessment?.status || null,
    tensor_liveness_status: analysis.tensor_liveness?.status || null,
    tensor_liveness_peak_bytes: analysis.tensor_liveness?.peak_bytes ?? null,
    input_flexibility: inputFlexibility,
    output_flexibility: outputFlexibility,
    unknown_mil_dimension_count: (analysis.tensors || []).flatMap((row) => row.shape || []).filter((value) => value == null).length,
    external_blob_reference_count: Number(analysis.coreml_blob_references?.length || 0),
    weight_integrity_status: analysis.weight_integrity?.status || null,
    weight_payload_bytes: analysis.weight_integrity?.payload_bytes ?? null,
    compression_contract_status: analysis.coreml?.mil_compression_contract?.status || null,
    compression_transform_count: Number(analysis.coreml?.mil_compression_contract?.transform_count || 0),
    compression_exact_contract_count: Number(analysis.coreml?.mil_compression_contract?.exact_contract_count || 0),
    byte_conservation: analysis.size_breakdown?.constant_bytes != null
      && analysis.size_breakdown.constant_bytes + analysis.size_breakdown.structure_overhead_bytes === analysis.file_size,
  };
  return { schema: COREML_MLPROGRAM_RECEIPT_SCHEMA, ...receiptBody, analysis_receipt_sha256: sha256(canonicalJson(receiptBody)) };
}

export function baselineFromReceipt(receipt) {
  return {
    status: receipt.status,
    contract_class: receipt.contract_class,
    model_type: receipt.model_type,
    specification_version: receipt.specification_version,
    files: receipt.files,
    bundle_sha256: receipt.bundle_sha256,
    operator_count: receipt.operator_count,
    tensor_count: receipt.tensor_count,
    total_macs: receipt.total_macs,
    mac_assessment_status: receipt.mac_assessment_status,
    tensor_liveness_status: receipt.tensor_liveness_status,
    tensor_liveness_peak_bytes: receipt.tensor_liveness_peak_bytes,
    input_flexibility: receipt.input_flexibility,
    output_flexibility: receipt.output_flexibility,
    unknown_mil_dimension_count: receipt.unknown_mil_dimension_count,
    external_blob_reference_count: receipt.external_blob_reference_count,
    weight_integrity_status: receipt.weight_integrity_status,
    weight_payload_bytes: receipt.weight_payload_bytes,
    compression_contract_status: receipt.compression_contract_status,
    compression_transform_count: receipt.compression_transform_count,
    compression_exact_contract_count: receipt.compression_exact_contract_count,
    byte_conservation: receipt.byte_conservation,
    analysis_receipt_sha256: receipt.analysis_receipt_sha256,
  };
}

export function receiptMatchesBaseline(receipt, artifact) {
  return canonicalJson(baselineFromReceipt(receipt)) === canonicalJson(artifact.baseline);
}

function generateFixture(artifact) {
  if (artifact.contract_class === "static_external_blob") return staticBlobPackage(artifact.id);
  if (artifact.contract_class === "enumerated_shape") return flexiblePackage(artifact.id, "enumerated");
  if (artifact.contract_class === "bounded_shape_range") return flexiblePackage(artifact.id, "range");
  if (artifact.contract_class === "blockwise_affine_compression") return compressionPackage(artifact.id, "blockwise");
  if (artifact.contract_class === "lut_palettization_compression") return compressionPackage(artifact.id, "lut");
  throw new Error(`Unsupported Core ML fixture class ${artifact.contract_class}.`);
}

function compressionPackage(id, kind) {
  let op;
  let outputShape;
  let outputDtype;
  if (kind === "blockwise") {
    outputShape = [4, 4];
    outputDtype = 11;
    op = operation("constexpr_blockwise_shift_scale", [
      ["data", valueBinding(immediateBytesValue(35, [4, 4], Buffer.alloc(8, 0x21)))],
      ["scale", valueBinding(immediateBytesValue(11, [2, 1], Buffer.alloc(8)))],
    ], [["output", outputDtype, outputShape]]);
  } else {
    outputShape = [2, 4];
    outputDtype = 10;
    op = operation("constexpr_lut_to_dense", [
      ["indices", valueBinding(immediateBytesValue(36, [2, 2], Buffer.from([0x1b])))],
      ["lut", valueBinding(immediateBytesValue(10, [1, 1, 4, 2], Buffer.alloc(16)))],
      ["vector_axis", valueBinding(immediateInt32Value(1))],
    ], [["output", outputDtype, outputShape]]);
  }
  const block = concat(string(2, "output"), message(3, op));
  const fn = concat(string(2, "CoreML8"), message(3, mapEntry("CoreML8", block)));
  const program = concat(uint(1, 1), message(2, mapEntry("main", fn)));
  const description = concat(message(10, feature("output", arrayType(outputShape, { dtype: outputDtype === 10 ? 65552 : 65568 }))), string(14, "main"));
  return packageRows(id, concat(uint(1, 9), message(2, description), message(502, program)), null);
}

function staticBlobPackage(id) {
  const weightShape = [2, 1, 3, 3];
  const constOp = operation("const", [["val", valueBinding(blobValue(11, weightShape, "@model_path/weights/weight.bin", 64))]], [["weight", 11, weightShape]]);
  const convOp = operation("conv", [["x", nameBinding("image")], ["weight", nameBinding("weight")]], [["output", 11, [1, 2, 2, 2]]]);
  const block = concat(string(2, "output"), message(3, constOp), message(3, convOp));
  const fn = concat(message(1, named("image", 11, [1, 1, 4, 4])), string(2, "CoreML7"), message(3, mapEntry("CoreML7", block)));
  const program = concat(uint(1, 1), message(2, mapEntry("main", fn)));
  const description = concat(message(1, feature("image", arrayType([1, 1, 4, 4]))), message(10, feature("output", arrayType([1, 2, 2, 2]))), string(14, "main"));
  const model = concat(uint(1, 8), message(2, description), message(502, program));
  const weights = floatPayload(Array.from({ length: 18 }, (_, index) => (index + 1) / 16));
  return packageRows(id, model, blobFile(2, weights));
}

function flexiblePackage(id, flexibility) {
  const milShape = [1, 3, null, null];
  const identity = operation("identity", [["x", nameBinding("input")]], [["output", 11, milShape]]);
  const block = concat(string(2, "output"), message(3, identity));
  const fn = concat(message(1, named("input", 11, milShape)), string(2, "CoreML7"), message(3, mapEntry("CoreML7", block)));
  const program = concat(uint(1, 1), message(2, mapEntry("main", fn)));
  const defaultShape = [1, 3, 224, 224];
  const interfaceType = flexibility === "enumerated"
    ? arrayType(defaultShape, { enumerated: [defaultShape, [1, 3, 256, 256]] })
    : arrayType(defaultShape, { ranges: [[1, 1], [3, 3], [128, 512], [128, 512]] });
  const description = concat(message(1, feature("input", interfaceType)), message(10, feature("output", interfaceType)), string(14, "main"));
  return packageRows(id, concat(uint(1, 8), message(2, description), message(502, program)), null);
}

function packageRows(id, model, weights) {
  const root = `${id}.mlpackage`;
  const itemPath = "com.deepbom.fixture";
  const manifest = Buffer.from(`${JSON.stringify({
    fileFormatVersion: "1.0.0",
    rootModelIdentifier: "root",
    itemInfoEntries: { root: { path: itemPath, name: "Model", author: "DEEPBOM", description: "Source-pinned Core ML contract fixture" } },
  }, null, 2)}\n`, "utf8");
  const files = [
    { path: `${root}/Manifest.json`, bytes: manifest },
    { path: `${root}/Data/${itemPath}/model.mlmodel`, bytes: model },
  ];
  if (weights) files.push({ path: `${root}/Data/${itemPath}/weights/weight.bin`, bytes: weights });
  return { files };
}

function arrayType(shape, { enumerated = null, ranges = null, dtype = 65568 } = {}) {
  let body = concat(packedInt(1, shape), uint(2, dtype));
  if (enumerated) body = concat(body, message(21, concat(enumerated.map((candidate) => message(1, packedInt(1, candidate))))));
  if (ranges) body = concat(body, message(31, concat(ranges.map(([lower, upper]) => message(1, concat(uint(1, lower), uint(2, upper)))))));
  return message(5, body);
}

function tensorType(dtype, shape) {
  const dimensions = shape.map((size) => message(3, size == null ? message(2, Buffer.alloc(0)) : message(1, uint(1, size))));
  return message(1, concat(uint(1, dtype), uint(2, shape.length), dimensions));
}

function named(name, dtype, shape) { return concat(string(1, name), message(2, tensorType(dtype, shape))); }
function nameBinding(name) { return message(1, string(1, name)); }
function blobValue(dtype, shape, fileName, offset) { return concat(message(2, tensorType(dtype, shape)), message(5, concat(string(1, fileName), uint(2, offset)))); }
function immediateBytesValue(dtype, shape, payload) {
  return concat(message(2, tensorType(dtype, shape)), message(3, message(1, message(7, bytes(1, payload)))));
}
function immediateInt32Value(value) {
  return concat(message(2, tensorType(23, [])), message(3, message(1, message(2, uint(1, value)))));
}
function valueBinding(value) { return message(1, message(2, value)); }
function argument(binding) { return message(2, concat(string(1, binding[0]), message(2, binding[1]))); }
function operation(type, inputs, outputs) { return concat(string(1, type), inputs.map(argument), outputs.map((output) => message(3, named(...output)))); }
function mapEntry(name, value) { return concat(string(1, name), message(2, value)); }
function feature(name, type) { return concat(string(1, name), message(3, type)); }

function blobFile(dtype, payload) {
  const data = Buffer.from(payload);
  const result = Buffer.alloc(128 + data.length);
  result.writeUInt32LE(1, 0);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(0xdeadbeef, 64);
  result.writeUInt32LE(dtype, 68);
  result.writeBigUInt64LE(BigInt(data.length), 72);
  result.writeBigUInt64LE(128n, 80);
  data.copy(result, 128);
  return result;
}

function floatPayload(values) {
  const result = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => result.writeFloatLE(value, index * 4));
  return result;
}

function packageFile(payload, relativePath) {
  const file = new File([payload], relativePath.split("/").at(-1));
  Object.defineProperty(file, "webkitRelativePath", { value: relativePath });
  return file;
}

function summarizeFlexibility(value) {
  if (!value) return { kind: "fixed" };
  if (value.kind === "enumerated") return { kind: value.kind, shapes: value.shapes || value.sizes || [] };
  return { kind: value.kind, dimensions: value.dimensions || null, width: value.width || null, height: value.height || null };
}

function varint(value) {
  let remaining = BigInt.asUintN(64, BigInt(value));
  const output = [];
  while (remaining > 127n) { output.push(Number(remaining & 127n) | 128); remaining >>= 7n; }
  output.push(Number(remaining));
  return Buffer.from(output);
}
function concat(...values) { return Buffer.concat(values.flat(Infinity).filter((value) => value != null)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function packedInt(field, values) { return bytes(field, concat(values.map(varint))); }
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function isSha256(value) { return /^[0-9a-f]{64}$/.test(String(value || "")); }
