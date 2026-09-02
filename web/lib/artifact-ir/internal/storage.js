import { SHA256 } from "./constants.js";
import {
  dimensions, exact, exactInteger, list, nonNegativeInteger, normalizeSha256, positiveInteger, positiveStorageBytes,
  safeExactSum, scopedStorageId, storageId, tensorIndex,
} from "./shared.js";

export function buildStorageTopology(analysis, format, tensors) {
  const tensorObjects = tensors.map((tensor, position) => storageObject(format, tensor, tensorIndex(tensor, position)))
    .filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n);
  const nestedTfliteObjects = format === "tflite" ? list(analysis?.tflite_subgraph_inventory?.rows)
    .filter((row) => Number(row.subgraph_index) !== 0)
    .flatMap((row) => list(row.tensor_intrinsics).map((tensor, position) => {
      const index = nonNegativeInteger(tensor.tensor_index) ?? position;
      const byteLength = exactInteger(tensor.buffer_data_length);
      const start = nonNegativeInteger(tensor.buffer_data_offset);
      return {
        id: scopedStorageId(`scope:tflite:subgraph:${row.subgraph_index}`, index),
        native_index: index,
        scope_ref: `scope:tflite:subgraph:${row.subgraph_index}`,
        name: String(tensor.name || `tensor_${index}`),
        dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        shape: dimensions(tensor.shape),
        serialized_byte_length: byteLength,
        byte_range: start != null && byteLength ? { status: "exact", offset_basis: "artifact_absolute", start, end_exclusive: safeExactSum(start, byteLength.decimal) }
          : { status: byteLength ? "length_only" : "not_assessed", offset_basis: null, start: null, end_exclusive: null },
        payload_sha256: null,
        encoding: { family: "scalar_or_declared_tensor_encoding", name: String(tensor.dtype || "UNKNOWN").toUpperCase(), block_elements: null, block_bytes: null },
        native_source: { format: "tflite", path: `SubGraph[${row.subgraph_index}].tensors[${index}].buffer` },
      };
    }).filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n)) : [];
  const parameterObjects = (["coreml", "mlmodel"].includes(format) ? list(analysis?.weight_integrity?.parameters) : []).map((parameter, index) => ({
    id: `storage:parameter:${index}`,
    source_parameter_index: index,
    native_index: index,
    name: String(parameter.layer_name ? `${parameter.layer_name}/${parameter.role || index}` : parameter.name || `parameter_${index}`),
    dtype: String(parameter.storage || parameter.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(parameter.shape),
    serialized_byte_length: exactInteger(parameter.byte_length),
    byte_range: { status: "length_only", offset_basis: null, start: null, end_exclusive: null },
    payload_sha256: normalizeSha256(parameter?.numerical_integrity?.payload_sha256 || parameter.payload_sha256),
    encoding: { family: parameter.quantization ? "coreml_declared_quantized_weight" : "scalar_or_declared_parameter_encoding", name: String(parameter.storage || parameter.dtype || "UNKNOWN"), block_elements: null, block_bytes: null },
    native_source: { format, path: `weight_integrity.parameters[${index}]` },
  })).filter((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n);
  const objects = [...tensorObjects, ...nestedTfliteObjects, ...parameterObjects];
  const bytes = objects.reduce((sum, row) => sum + BigInt(row.serialized_byte_length?.decimal || "0"), 0n);
  return {
    status: objects.length ? "assessed_serialized_objects" : "not_applicable_no_serialized_tensor_payload_ledger",
    objects,
    totals: {
      object_count: objects.length,
      serialized_object_bytes_sum: exact(bytes),
      exact_range_count: objects.filter((row) => row.byte_range?.status === "exact").length,
      payload_digest_count: objects.filter((row) => SHA256.test(String(row.payload_sha256 || ""))).length,
    },
    interpretation_boundary: "Storage objects preserve serialized payload identity and ranges where the parser exposes them. Summed object bytes are not file size, runtime allocation, residency, repacking, or transfer volume.",
  };
}

function storageObject(format, tensor, index) {
  const byteLength = exactInteger(positiveStorageBytes(tensor, format));
  const absoluteOffset = nonNegativeInteger(tensor?.numerical_integrity?.byte_offset_absolute ?? tensor.buffer_data_offset);
  const relativeStart = nonNegativeInteger(tensor.data_offset);
  const start = absoluteOffset ?? relativeStart;
  const end = start != null && byteLength ? safeExactSum(start, byteLength.decimal) : null;
  return {
    id: storageId(index),
    native_index: index,
    name: String(tensor.name || `tensor_${index}`),
    dtype: String(tensor.dtype || "UNKNOWN").toUpperCase(),
    shape: dimensions(tensor.shape),
    serialized_byte_length: byteLength,
    byte_range: start != null && end != null
      ? { status: "exact", offset_basis: absoluteOffset != null ? "artifact_absolute" : "format_payload_relative", start, end_exclusive: end }
      : { status: byteLength ? "length_only" : "not_assessed", offset_basis: null, start: null, end_exclusive: null },
    payload_sha256: normalizeSha256(tensor?.numerical_integrity?.payload_sha256 || tensor.external_sidecar_sha256),
    encoding: storageEncoding(format, tensor),
    native_source: nativeStorageLocator(format, tensor, index),
  };
}

function nativeStorageLocator(format, tensor, index) {
  if (format === "gguf") return { format, path: `tensor_infos[${index}]`, payload_offset_basis: "tensor_data_section" };
  if (format === "safetensors") return { format, path: `header[${JSON.stringify(String(tensor.name || ""))}]`, payload_offset_basis: "data_section" };
  if (format === "tflite") return { format, path: `SubGraph[0].tensors[${index}].buffer` };
  if (format === "onnx") return { format, path: `ModelProto.graph.initializer[name=${JSON.stringify(String(tensor.name || ""))}]` };
  return { format, path: `serialized_parameters[${index}]` };
}

function storageEncoding(format, tensor) {
  if (format === "gguf") return { family: "ggml_block_encoding", name: String(tensor.dtype || "UNKNOWN"), block_elements: positiveInteger(tensor.block_elements), block_bytes: positiveInteger(tensor.block_bytes) };
  return { family: "scalar_or_declared_tensor_encoding", name: String(tensor.dtype || "UNKNOWN").toUpperCase(), block_elements: null, block_bytes: null };
}
