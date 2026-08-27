import { createHash } from "node:crypto";
import { File } from "node:buffer";
import { readFile } from "node:fs/promises";

import { COREML_NEURAL_NETWORK_SOURCE } from "../web/lib/coreml-neural-network.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";

export const COREML_LEGACY_QUANTIZATION_CORPUS_PATH = "corpus/coreml-legacy-quantization-corpus.v1.json";
export const COREML_LEGACY_QUANTIZATION_MODEL_PATH = "corpus/coreml-legacy-quantization-corpus/per-output-channel-linear-int4.mlmodel";

export function buildCoreMlPerChannelLinearFixture({ scaleCount = 2 } = {}) {
  const packedCodes = Buffer.from([0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01]);
  const scales = [0.5, 0.25].slice(0, scaleCount);
  const biases = [-1, 1].slice(0, scaleCount);
  const quantization = concat(uint(1, 4), message(101, concat(packedFloat(1, scales), packedFloat(2, biases))));
  const weights = concat(bytes(30, packedCodes), message(40, quantization));
  const convolution = concat(
    uint(1, 2), uint(2, 1), packedUint(20, [3, 3]), packedUint(30, [1, 1]),
    message(51, Buffer.alloc(0)), message(90, weights),
  );
  const network = concat(message(1, layer("per_channel_conv", ["input"], ["output"], 100, convolution)), uint(5, 1));
  const description = concat(
    message(1, feature("input", arrayType([1, 1, 4, 4]))),
    message(10, feature("output", arrayType([1, 2, 2, 2]))),
  );
  return concat(uint(1, 5), message(2, description), message(500, network));
}

export async function analyzeCoreMlPerChannelLinearFixture(payload = buildCoreMlPerChannelLinearFixture()) {
  const data = Buffer.from(payload);
  const analysis = await readCoreMlPerChannelLinearAnalysis(data);
  const weight = analysis.ops?.[0]?.coreml_weights?.find((item) => item.role === "weights");
  return {
    schema: "deepbom.coreml_legacy_quantization_corpus_receipt.v1",
    artifact_sha256: createHash("sha256").update(data).digest("hex"),
    artifact_size_bytes: data.length,
    model_type: analysis.coreml?.model_type || null,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs,
    weight_storage: weight?.storage || null,
    number_of_bits: weight?.quantization?.number_of_bits ?? null,
    scheme: weight?.quantization?.scheme || null,
    granularity: weight?.quantization?.granularity || null,
    axis: weight?.quantization?.axis ?? null,
    channel_count: weight?.quantization?.channel_count ?? null,
    scale_count: weight?.quantization?.scale_count ?? null,
    bias_count: weight?.quantization?.bias_count ?? null,
    value_count: weight?.value_count ?? null,
    decoded_value_count: weight?.numerical_integrity?.decoded_value_count ?? null,
    finite_count: weight?.numerical_integrity?.finite_count ?? null,
    nonfinite_count: weight?.numerical_integrity?.nonfinite_count ?? null,
    quant_code_levels_used: weight?.numerical_integrity?.quant_code_levels_used ?? null,
    payload_sha256: weight?.numerical_integrity?.payload_sha256 || null,
    quantization_assessment_status: analysis.quantization_status?.assessment_status || null,
    per_axis_quantized_weight_parameter_count: analysis.quantization_status?.per_axis_quantized_weight_parameter_count ?? null,
  };
}

export async function readCoreMlPerChannelLinearAnalysis(payload = buildCoreMlPerChannelLinearFixture()) {
  const data = Buffer.from(payload);
  const analysis = (await readCoreMlModelFile(new File([data], "per-output-channel-linear-int4.mlmodel"))).analysis;
  analysis.model_sha256 = createHash("sha256").update(data).digest("hex");
  return analysis;
}

export async function readCoreMlLegacyQuantizationCorpus(path = COREML_LEGACY_QUANTIZATION_CORPUS_PATH) {
  return JSON.parse(await readFile(path, "utf8"));
}

export function coreMlLegacyQuantizationSource() { return { ...COREML_NEURAL_NETWORK_SOURCE }; }

function varint(value) {
  let current = BigInt(value);
  const result = [];
  while (current > 0x7fn) { result.push(Number(current & 0x7fn) | 0x80); current >>= 7n; }
  result.push(Number(current));
  return Buffer.from(result);
}
function concat(...values) { return Buffer.concat(values.flat(Infinity).filter((value) => value != null)); }
function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { const body = Buffer.from(value); return concat(key(field, 2), varint(body.length), body); }
function message(field, value) { return bytes(field, value); }
function string(field, value) { return bytes(field, Buffer.from(value, "utf8")); }
function packedUint(field, values) { return bytes(field, concat(values.map(varint))); }
function packedFloat(field, values) {
  const body = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => body.writeFloatLE(value, index * 4));
  return bytes(field, body);
}
function arrayType(shape) { return message(5, concat(packedUint(1, shape), uint(2, 65568))); }
function feature(name, type) { return concat(string(1, name), message(3, type)); }
function layer(name, inputs, outputs, typeField, params) {
  return concat(string(1, name), inputs.map((value) => string(2, value)), outputs.map((value) => string(3, value)), message(typeField, params));
}
