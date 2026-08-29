import { File } from "node:buffer";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";

function assert(value, message) { if (!value) throw new Error(message); }

function varint(value) {
  let current = BigInt(value);
  const bytes = [];
  while (current > 0x7fn) { bytes.push(Number(current & 0x7fn) | 0x80); current >>= 7n; }
  bytes.push(Number(current));
  return Uint8Array.from(bytes);
}

function concat(...parts) {
  const rows = parts.flat().filter(Boolean).map((value) => value instanceof Uint8Array ? value : new Uint8Array(value));
  const result = new Uint8Array(rows.reduce((sum, row) => sum + row.length, 0));
  let offset = 0;
  for (const row of rows) { result.set(row, offset); offset += row.length; }
  return result;
}

function key(field, wire) { return varint(field * 8 + wire); }
function uint(field, value) { return concat(key(field, 0), varint(value)); }
function bytes(field, value) { return concat(key(field, 2), varint(value.length), value); }
function text(field, value) { return bytes(field, new TextEncoder().encode(value)); }
function packed(field, values) { const payload = concat(values.map(varint)); return bytes(field, payload); }
function messagePrefix(field, length) { return concat(key(field, 2), varint(length)); }

function feature(name) {
  const arrayType = bytes(5, concat(packed(1, [1]), uint(2, 65568)));
  return concat(text(1, name), bytes(3, arrayType));
}

const description = concat(bytes(1, feature("input")), bytes(10, feature("output")));
const copyLayer = concat(text(1, "copy"), text(2, "input"), text(3, "output"), bytes(600, new Uint8Array()));
const layerRecord = bytes(1, copyLayer);
const oneMiB = new Blob([new Uint8Array(1024 * 1024)]);
const paddingBytes = 65 * 1024 * 1024;
const padding = new Blob(Array.from({ length: 65 }, () => oneMiB));
const unknownPrefix = messagePrefix(999, paddingBytes);
const neuralNetworkLength = unknownPrefix.length + padding.size + layerRecord.length;
const neuralNetwork = new Blob([unknownPrefix, padding, layerRecord]);
const nestedModel = new Blob([
  uint(1, 1), bytes(2, description), messagePrefix(500, neuralNetworkLength), neuralNetwork,
]);
const pipelineBody = new Blob([messagePrefix(1, nestedModel.size), nestedModel, text(2, "large_nested")]);
const model = new File([
  uint(1, 1), bytes(2, description), messagePrefix(202, pipelineBody.size), pipelineBody,
], "large-pipeline.mlmodel");

const parsed = await readCoreMlModelFile(model);
assert(model.size > 64 * 1024 * 1024, "Core ML range-reader fixture does not cross the former payload boundary");
assert(parsed.analysis?.coreml?.payload_read_strategy === "range_streamed_top_level_records", "Core ML range-read strategy is not disclosed");
assert(parsed.analysis?.coreml?.pipeline?.models?.length === 1, "Core ML nested pipeline model was not decoded");
assert(parsed.analysis?.operator_count === 1 && parsed.analysis?.ops?.[0]?.name === "COPY", "Core ML nested range graph was not reconstructed");
assert(parsed.retainedBytes?.length === 0 && parsed.payloadLoaded === false && parsed.payloadScanned === true, "Core ML range reader retained the original payload unexpectedly");

console.log(`Core ML range reader passed: ${(model.size / 1048576).toFixed(1)} MiB nested pipeline decoded without whole-model retention.`);
