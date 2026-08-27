export const LLM_LAYER_STORAGE_SCHEMA = "deepbom.llm_layer_storage.v1";

function exactDecimal(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  const text = String(value ?? "");
  return /^(?:0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: value.toString() };
}

function namespaceContract(format, architecture) {
  if (format === "gguf") return {
    id: "gguf_block_namespace",
    expression: "^blk\\.([0-9]+)\\.",
    regex: /^blk\.([0-9]+)\./,
    evidence: "OBSERVED_SERIALIZED_GGUF_TENSOR_NAMESPACE",
  };
  if (format !== "safetensors") return null;
  const recurrent = architecture?.kind === "ssm_recurrent";
  return {
    id: recurrent ? "hf_mamba_backbone_layer_namespace" : "hf_decoder_model_layer_namespace",
    expression: recurrent ? "^backbone\\.layers\\.([0-9]+)\\." : "^model\\.layers\\.([0-9]+)\\.",
    regex: recurrent ? /^backbone\.layers\.([0-9]+)\./ : /^model\.layers\.([0-9]+)\./,
    evidence: "SOURCE_REGISTERED_HF_CANONICAL_TENSOR_NAMESPACE",
  };
}

export function buildLlmLayerStorageLedger(analysis, architecture, storage) {
  const format = String(analysis?.format || "").toLowerCase();
  const contract = namespaceContract(format, architecture);
  const boundary = "Exact serialized tensor bytes grouped by a source-registered or format-canonical layer namespace. These bytes are not runtime packing, replicas, allocator usage, residency, offload capacity, engine workspace, or physical transfer.";
  if (!contract) return { schema: LLM_LAYER_STORAGE_SCHEMA, status: "not_applicable_format", evidence_class: "NOT_APPLICABLE", format, boundary };
  const expectedLayerCount = Number(architecture?.layer_count);
  if (!Number.isSafeInteger(expectedLayerCount) || expectedLayerCount <= 0) return {
    schema: LLM_LAYER_STORAGE_SCHEMA,
    status: "not_assessable_layer_count_unbound",
    evidence_class: "NOT_ASSESSABLE",
    format,
    namespace: { id: contract.id, expression: contract.expression, evidence: contract.evidence },
    boundary,
  };
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const rows = [];
  const invalidByteTensors = [];
  for (const [position, tensor] of tensors.entries()) {
    const bytes = exactDecimal(tensor?.byte_length_decimal ?? tensor?.byte_length);
    if (bytes == null) invalidByteTensors.push({ tensor_index: Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : position, tensor_name: String(tensor?.name || "") });
    rows.push({ tensor, position, bytes });
  }
  if (invalidByteTensors.length) return {
    schema: LLM_LAYER_STORAGE_SCHEMA,
    status: "invalid_serialized_tensor_byte_contract",
    evidence_class: "OBSERVED_INVALID",
    format,
    expected_layer_count: expectedLayerCount,
    invalid_byte_tensor_count: invalidByteTensors.length,
    invalid_byte_tensors: invalidByteTensors.slice(0, 256),
    namespace: { id: contract.id, expression: contract.expression, evidence: contract.evidence },
    boundary,
  };
  const layers = Array.from({ length: expectedLayerCount }, (_, layerIndex) => ({ layer_index: layerIndex, tensor_count: 0, bytes: 0n }));
  let nonLayerBytes = 0n;
  let nonLayerTensorCount = 0;
  const unexpected = [];
  for (const row of rows) {
    const name = String(row.tensor?.name || "");
    const match = contract.regex.exec(name);
    if (!match) {
      nonLayerBytes += row.bytes;
      nonLayerTensorCount += 1;
      continue;
    }
    const layerIndex = Number(match[1]);
    if (!Number.isSafeInteger(layerIndex) || layerIndex < 0 || layerIndex >= expectedLayerCount) {
      unexpected.push({ tensor_index: Number.isSafeInteger(Number(row.tensor?.index)) ? Number(row.tensor.index) : row.position, tensor_name: name, layer_index: layerIndex });
      continue;
    }
    layers[layerIndex].tensor_count += 1;
    layers[layerIndex].bytes += row.bytes;
  }
  const allBytes = rows.reduce((sum, row) => sum + row.bytes, 0n);
  const layerBytes = layers.reduce((sum, row) => sum + row.bytes, 0n);
  const expectedBytes = exactDecimal(storage?.byte_length_decimal ?? storage?.byte_length);
  const missing = layers.filter((row) => row.tensor_count === 0).map((row) => row.layer_index);
  const conservation = layerBytes + nonLayerBytes === allBytes && expectedBytes != null && allBytes === expectedBytes;
  const status = unexpected.length ? "invalid_layer_index_out_of_range"
    : !conservation ? "invalid_storage_conservation"
      : missing.length ? "partial_layer_namespace_coverage" : "assessed_exact_serialized_layer_storage";
  const populated = layers.filter((row) => row.tensor_count > 0);
  const sortedBytes = populated.map((row) => row.bytes).sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
  return {
    schema: LLM_LAYER_STORAGE_SCHEMA,
    status,
    evidence_class: status.startsWith("invalid") ? "OBSERVED_INVALID" : status.startsWith("partial") ? `${contract.evidence}/PARTIAL` : `${contract.evidence}/DERIVED_EXACT_INTEGER`,
    format,
    namespace: { id: contract.id, expression: contract.expression, evidence: contract.evidence },
    expected_layer_count: expectedLayerCount,
    observed_layer_count: populated.length,
    missing_layer_indices: missing,
    unexpected_layer_tensor_count: unexpected.length,
    unexpected_layer_tensors: unexpected.slice(0, 256),
    serialized_tensor_count: rows.length,
    layer_tensor_count: layers.reduce((sum, row) => sum + row.tensor_count, 0),
    non_layer_tensor_count: nonLayerTensorCount,
    serialized_tensor_bytes: exact(allBytes),
    expected_serialized_tensor_bytes: expectedBytes == null ? null : exact(expectedBytes),
    layer_bytes: exact(layerBytes),
    non_layer_bytes: exact(nonLayerBytes),
    minimum_populated_layer_bytes: sortedBytes.length ? exact(sortedBytes[0]) : null,
    maximum_populated_layer_bytes: sortedBytes.length ? exact(sortedBytes.at(-1)) : null,
    conservation: {
      status: conservation ? "pass" : "fail",
      equation: "sum(layer.serialized_bytes) + non_layer_bytes = sum(tensor.byte_length) = tensor_storage_summary.byte_length",
    },
    layers: layers.map((row) => ({ layer_index: row.layer_index, tensor_count: row.tensor_count, serialized_bytes: exact(row.bytes) })),
    boundary,
  };
}
