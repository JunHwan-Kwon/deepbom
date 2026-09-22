import { Sha256Accumulator, sha256BytesHex } from "../sha256-sync.js";
import { TensorStatistics } from "./statistics.js";
import { requireCondition, shapeCount } from "./common.js";

export const MAX_IN_MEMORY_SOURCE = 512 * 1024 * 1024;
export const sourceSize = source => source instanceof Uint8Array ? source.byteLength : source?.size;
export async function readBytes(source, start = 0, end = sourceSize(source)) {
  requireCondition(Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end >= start && end <= sourceSize(source), "payload range is outside artifact");
  const bytes = source instanceof Uint8Array ? source.subarray(start, end) : new Uint8Array(await source.slice(start, end).arrayBuffer());
  requireCondition(bytes.length === end - start, "payload read was truncated"); return bytes;
}
export async function hashSource(source, { signal, onProgress } = {}) {
  const digest = new Sha256Accumulator(), size = sourceSize(source);
  requireCondition(Number.isSafeInteger(size) && size >= 0, "missing artifact source");
  for (let offset = 0; offset < size; offset += 4 * 1024 * 1024) {
    signal?.throwIfAborted(); const end = Math.min(size, offset + 4 * 1024 * 1024);
    digest.update(await readBytes(source, offset, end)); onProgress?.({ phase: "identity", bytes: end, total: size });
  }
  return digest.digestHex();
}
const unavailable = reason => ({ status: "not_assessed", reason });
const dtypeAlias = dtype => ({ F32: "FLOAT32", F16: "FLOAT16", F64: "FLOAT64", BF16: "BFLOAT16", FLOAT: "FLOAT32", DOUBLE: "FLOAT64", HALF: "FLOAT16", BYTE: "UINT8", CHAR: "INT8", SHORT: "INT16", INT: "INT32", LONG: "INT64" }[dtype] || dtype);
export async function numericSources(source, analysis, model, { captureValues = false, maxCapturedValues = 8_000_000 } = {}) {
  const stores = model.tensors_and_storage.storage_objects, candidates = [], format = String(analysis.format || model.artifact.format).toLowerCase();
  const add = (store, entry) => candidates.push({ storage_ref: store?.id || null, ...entry });
  if (["gguf", "safetensors"].includes(format)) {
    const { visitSerializedTensorValues } = await import("../tensor-numerical-integrity.js");
    for (const tensor of analysis.tensors || []) {
      const store = stores.find(s => s.native_subject_ref === `storage:tensor:${tensor.index}`);
      add(store, { locator: `${format}:tensor:${tensor.index}`, name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
        visit: (visitor, options) => visitSerializedTensorValues(source, analysis, tensor, visitor, options) });
    }
  } else if (format === "onnx" && sourceSize(source) <= MAX_IN_MEMORY_SOURCE) {
    const { onnxNumericTensorSources } = await import("../../onnx.js");
    for (const entry of onnxNumericTensorSources(await readBytes(source))) {
      const matches = entry.scope === "main_graph" && entry.role === "graph_initializer" ? stores.filter(s => s.native_source?.path === `ModelProto.graph.initializer[name=${JSON.stringify(entry.name)}]`) : [];
      add(matches.length === 1 ? matches[0] : null, { locator: `onnx:${entry.scope}:${entry.role}:${candidates.length}`, name: entry.name, dtype: entry.dtype, shape: entry.shape,
        visit: visitor => { const result = entry.visit(visitor); return result.ok ? { status: "assessed", representation: /^COMPLEX/.test(entry.dtype) ? "complex_magnitude" : "stored_scalar", payload_sha256: null, invalid_encoding_count: 0 } : unavailable(result.reason); } });
    }
  } else if (["coreml", "mlmodel"].includes(format) && sourceSize(source) <= MAX_IN_MEMORY_SOURCE) {
    const { parseCoreMlModel } = await import("../coreml-metadata-adapter.js");
    const captures = new WeakMap(); let captured = 0;
    const parsed = parseCoreMlModel(await readBytes(source), model.artifact.filename, sourceSize(source), { numericalFactory: () => {
      const stats = new TensorStatistics(); let values = [];
      return { add(value) { stats.add(value); if (captureValues && values) { if (captured < maxCapturedValues) {values.push(value); captured++;} else values = null; } }, finish() { const result = stats.finish(); if (captureValues) captures.set(result, values); return result; } };
    } });
    for (const [index, parameter] of (parsed.weight_integrity?.parameters || []).entries()) {
      const store = stores.find(s => s.native_source?.path === `weight_integrity.parameters[${index}]`);
      add(store, { locator: `coreml:parameter:${index}`, name: store?.name || parameter.role, dtype: store?.dtype || parameter.storage,
        shape: store?.shape?.length ? store.shape : Number.isSafeInteger(parameter.value_count) ? [parameter.value_count] : null,
        visit: visitor => {
          if (!parameter.optional_statistics) return unavailable("parameter_values_not_exposed_by_decoder");
          if (captureValues) { const values = captures.get(parameter.optional_statistics); if (!values) return unavailable("coreml_capture_budget_exceeded"); for (const value of values) visitor(value); }
          return { status: "assessed", representation: /quantized|int8_dynamic/.test(parameter.storage) ? "dequantized_real" : "stored_scalar", statistics: parameter.optional_statistics, payload_sha256: parameter.numerical_integrity?.payload_sha256 || null, invalid_encoding_count: 0 };
        } });
    }
    // Complete immediate values can be inspected without interpreting names as graph edges.
    for (const tensor of parsed.tensors || []) {
      const immediate = tensor.immediate_value;
      if (!tensor.constant_buffer || !immediate) continue;
      const values = immediate.values;
      const store = stores.find(s => s.native_subject_ref === `storage:tensor:${tensor.index}`);
      if (store && candidates.some(c => c.storage_ref === store.id)) continue;
      add(store, { locator: `coreml:immediate:${tensor.index}`, name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
        visit: visitor => visitMilImmediateValues(tensor, visitor) });
    }
  }
  if (format === "tflite" || format === "executorch") {
    const { visitDeclaredNumericBytes } = await import("../../onnx.js");
    let execRanges = new Map();
    if (format === "executorch" && sourceSize(source) <= MAX_IN_MEMORY_SOURCE) {
      const { execuTorchNumericRanges } = await import("../../executorch.js");
      execRanges = execuTorchNumericRanges(await readBytes(source), analysis);
    }
    for (const store of stores) {
      let range = store.byte_range?.offset_basis === "artifact_absolute" && store.byte_range.status === "exact" ? store.byte_range : null;
      if (format === "executorch") range = execRanges.get(store.native_subject_ref) || null;
      add(store, { locator: store.native_source?.path || store.id, name: store.name, dtype: store.dtype, shape: store.shape,
        visit: async visitor => {
          if (!range) return unavailable("serialized_payload_range_unavailable");
          if (range.end_exclusive - range.start > MAX_IN_MEMORY_SOURCE) return unavailable("payload_memory_budget_exceeded");
          const bytes = await readBytes(source, range.start, range.end_exclusive);
          const result = visitDeclaredNumericBytes(bytes, dtypeAlias(store.dtype), store.shape, visitor);
          return result.ok ? { status: "assessed", representation: /^COMPLEX/.test(store.dtype) ? "complex_magnitude" : "stored_scalar", payload_sha256: sha256BytesHex(bytes), invalid_encoding_count: 0 } : unavailable(result.reason);
        } });
    }
  }
  for (const store of stores) if (!candidates.some(c => c.storage_ref === store.id)) add(store, { locator: store.native_source?.path || store.id, name: store.name, dtype: store.dtype, shape: store.shape, visit: () => unavailable(sourceSize(source) > MAX_IN_MEMORY_SOURCE ? "source_memory_budget_exceeded" : "serialized_payload_decoder_unavailable") });
  return candidates;
}

export async function visitMilImmediateValues(tensor, visitor) {
  const immediate=tensor.immediate_value, values=immediate?.values;
  if(!Array.isArray(values)||immediate.truncated||values.length!==immediate.count) return unavailable("immediate_values_incomplete");
  if(immediate.kind==="bytes") {
    const { visitDeclaredNumericBytes }=await import("../../onnx.js");
    if(values.some(v=>!Number.isInteger(v)||v<0||v>255)) return unavailable("invalid_immediate_byte_value");
    const result=visitDeclaredNumericBytes(Uint8Array.from(values),dtypeAlias(tensor.dtype),tensor.shape,visitor);
    if(!result.ok) return unavailable(result.reason);
  } else {
    const kindDtype={float32:"FLOAT32",float64:"FLOAT64",int32:"INT32",int64:"INT64",bool:"BOOL"}[immediate.kind];
    if(kindDtype!==tensor.dtype || BigInt(values.length)!==shapeCount(tensor.shape)) return unavailable("immediate_kind_or_shape_mismatch");
    for(const value of values) visitor(immediate.kind==="bool"?Number(value):value);
  }
  return {status:"assessed",representation:"stored_scalar",payload_sha256:null,invalid_encoding_count:0};
}
