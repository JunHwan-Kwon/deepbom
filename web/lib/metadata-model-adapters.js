import { GGUF_DEQUANTIZATION_SOURCE, SAFETENSORS_NUMERICAL_SOURCE, safeTensorsNumericalSourceEvidence, scanSerializedTensorPayloads } from "./tensor-numerical-integrity.js";
import { buildGgufBackendCompatibility } from "./gguf-backend-contract.js";
import { buildOnDeviceLlmContract } from "./on-device-llm-contract.js";
import { buildSafeTensorsQuantizationContract } from "./safetensors-quantization-contract.js";
import { sha256TextHex } from "./sha256-sync.js";
import { buildCanonicalGatedDecoderProjection, buildKvStateProjection } from "./transformer-architecture-projection.js";
import { parseStrictJson } from "./strict-json.js";

export { parseStrictJson } from "./strict-json.js";

const MIB = 1024 * 1024;
const MAX_SAFE_TENSORS_HEADER = 100 * MIB;
const MAX_GGUF_HEADER = 64 * MIB;
const GGUF_VALUE_TYPES = ["UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32", "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64"];
export const SAFETENSORS_REFERENCE = SAFETENSORS_NUMERICAL_SOURCE;
const SAFE_TENSOR_BITS = Object.freeze({
  F4: 4, F6_E2M3: 6, F6_E3M2: 6,
  BOOL: 8, U8: 8, I8: 8,
  F8_E4M3: 8, F8_E5M2: 8, F8_E8M0: 8, F8_E4M3FNUZ: 8, F8_E5M2FNUZ: 8,
  U16: 16, I16: 16, F16: 16, BF16: 16,
  U32: 32, I32: 32, F32: 32,
  U64: 64, I64: 64, F64: 64, C64: 64,
});
export const GGML_TYPE_TRAITS_SOURCE = Object.freeze({
  repository: "ggml-org/llama.cpp",
  source_commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
  type_traits_source: "ggml/src/ggml.c",
  type_traits_source_sha256: "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4",
  block_layout_source: "ggml/src/ggml-common.h",
  block_layout_source_sha256: "af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0",
});
// Exact serialized block cardinalities for every non-removed type in the
// pinned GGML enum. Payload dequantization coverage is tracked independently;
// knowing a source-pinned block size never implies that its values were decoded.
const GGML_TYPE_TRAITS = Object.freeze({
  0: { name: "F32", block_elements: 1, block_bytes: 4 },
  1: { name: "F16", block_elements: 1, block_bytes: 2 },
  2: { name: "Q4_0", block_elements: 32, block_bytes: 18 },
  3: { name: "Q4_1", block_elements: 32, block_bytes: 20 },
  6: { name: "Q5_0", block_elements: 32, block_bytes: 22 },
  7: { name: "Q5_1", block_elements: 32, block_bytes: 24 },
  8: { name: "Q8_0", block_elements: 32, block_bytes: 34 },
  9: { name: "Q8_1", block_elements: 32, block_bytes: 36 },
  10: { name: "Q2_K", block_elements: 256, block_bytes: 84 },
  11: { name: "Q3_K", block_elements: 256, block_bytes: 110 },
  12: { name: "Q4_K", block_elements: 256, block_bytes: 144 },
  13: { name: "Q5_K", block_elements: 256, block_bytes: 176 },
  14: { name: "Q6_K", block_elements: 256, block_bytes: 210 },
  15: { name: "Q8_K", block_elements: 256, block_bytes: 292 },
  16: { name: "IQ2_XXS", block_elements: 256, block_bytes: 66 },
  17: { name: "IQ2_XS", block_elements: 256, block_bytes: 74 },
  18: { name: "IQ3_XXS", block_elements: 256, block_bytes: 98 },
  19: { name: "IQ1_S", block_elements: 256, block_bytes: 50 },
  20: { name: "IQ4_NL", block_elements: 32, block_bytes: 18 },
  21: { name: "IQ3_S", block_elements: 256, block_bytes: 110 },
  22: { name: "IQ2_S", block_elements: 256, block_bytes: 82 },
  23: { name: "IQ4_XS", block_elements: 256, block_bytes: 136 },
  24: { name: "I8", block_elements: 1, block_bytes: 1 },
  25: { name: "I16", block_elements: 1, block_bytes: 2 },
  26: { name: "I32", block_elements: 1, block_bytes: 4 },
  27: { name: "I64", block_elements: 1, block_bytes: 8 },
  28: { name: "F64", block_elements: 1, block_bytes: 8 },
  29: { name: "IQ1_M", block_elements: 256, block_bytes: 56 },
  30: { name: "BF16", block_elements: 1, block_bytes: 2 },
  34: { name: "TQ1_0", block_elements: 256, block_bytes: 54 },
  35: { name: "TQ2_0", block_elements: 256, block_bytes: 66 },
  39: { name: "MXFP4", block_elements: 32, block_bytes: 17 },
  40: { name: "NVFP4", block_elements: 64, block_bytes: 36 },
  41: { name: "Q1_0", block_elements: 128, block_bytes: 18 },
  42: { name: "Q2_0", block_elements: 64, block_bytes: 18 },
});

const GGUF_CANONICAL_GATED_DECODER_ARCHITECTURES = new Set(["llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2"]);
export function isCanonicalGgufDecoderArchitecture(value) { return GGUF_CANONICAL_GATED_DECODER_ARCHITECTURES.has(value); }

class NeedMoreData extends Error {
  constructor(requiredBytes) {
    super(`metadata header requires at least ${requiredBytes} bytes`);
    this.requiredBytes = requiredBytes;
  }
}

// TextDecoder throws a raw TypeError in fatal mode. Every decode site in this
// codebase reports a domain error instead so callers never have to distinguish
// a malformed artifact from an internal fault.
function decodeUtf8(bytes, label) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`${label} is not valid UTF-8`);
  }
}

class Reader {
  constructor(bytes, littleEndian = true) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.offset = 0;
    this.littleEndian = littleEndian;
  }

  need(length) {
    if (!Number.isSafeInteger(length) || length < 0) throw new Error("invalid read length");
    if (this.offset + length > this.bytes.length) throw new NeedMoreData(this.offset + length);
  }

  u8() { this.need(1); return this.bytes[this.offset++]; }
  i8() { this.need(1); return this.view.getInt8(this.offset++); }
  u16() { this.need(2); const value = this.view.getUint16(this.offset, this.littleEndian); this.offset += 2; return value; }
  i16() { this.need(2); const value = this.view.getInt16(this.offset, this.littleEndian); this.offset += 2; return value; }
  u32() { this.need(4); const value = this.view.getUint32(this.offset, this.littleEndian); this.offset += 4; return value; }
  i32() { this.need(4); const value = this.view.getInt32(this.offset, this.littleEndian); this.offset += 4; return value; }
  f32() { this.need(4); const value = this.view.getFloat32(this.offset, this.littleEndian); this.offset += 4; return value; }
  f64() { this.need(8); const value = this.view.getFloat64(this.offset, this.littleEndian); this.offset += 8; return value; }
  u64() { return this.safeInteger(this.big64(false)); }
  i64() { return this.safeInteger(this.big64(true)); }
  big64(signed) {
    this.need(8);
    const value = signed
      ? this.view.getBigInt64(this.offset, this.littleEndian)
      : this.view.getBigUint64(this.offset, this.littleEndian);
    this.offset += 8;
    return value;
  }
  safeInteger(value) {
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) throw new Error("64-bit value exceeds JavaScript exact integer range");
    return Number(value);
  }
  string(maxLength = 65535) {
    const length = this.u64();
    if (length > maxLength) throw new Error(`string length ${length} exceeds ${maxLength}`);
    this.need(length);
    const value = decodeUtf8(this.bytes.subarray(this.offset, this.offset + length), "metadata string");
    this.offset += length;
    return value;
  }
}

function parseGgufValue(reader, type, depth = 0) {
  if (depth > 4) throw new Error("GGUF metadata array nesting exceeds 4");
  switch (type) {
    case 0: return reader.u8();
    case 1: return reader.i8();
    case 2: return reader.u16();
    case 3: return reader.i16();
    case 4: return reader.u32();
    case 5: return reader.i32();
    case 6: return reader.f32();
    case 7: {
      const value = reader.u8();
      if (value > 1) throw new Error(`invalid GGUF boolean ${value}`);
      return Boolean(value);
    }
    case 8: return reader.string(Number.MAX_SAFE_INTEGER);
    case 9: {
      const elementType = reader.u32();
      if (!GGUF_VALUE_TYPES[elementType]) throw new Error(`unknown GGUF array value type ${elementType}`);
      const count = reader.u64();
      if (count > 10_000_000) throw new Error(`GGUF metadata array count ${count} exceeds safety limit`);
      const sample = [];
      for (let index = 0; index < count; index += 1) {
        const value = parseGgufValue(reader, elementType, depth + 1);
        if (index < 8) sample.push(value);
      }
      return { kind: "array", element_type: GGUF_VALUE_TYPES[elementType], count, sample };
    }
    case 10: return reader.u64();
    case 11: return reader.i64();
    case 12: return reader.f64();
    default: throw new Error(`unknown GGUF metadata value type ${type}`);
  }
}

function ggmlTensorStorage(shape, ggmlType) {
  const traits = GGML_TYPE_TRAITS[ggmlType];
  if (!traits) return { status: "unsupported_type", traits: null, byteLength: null };
  const rowElements = BigInt(shape[0] ?? 1);
  const blockElements = BigInt(traits.block_elements);
  if (rowElements % blockElements !== 0n) return { status: "invalid_row_cardinality", traits, byteLength: null };
  const rows = shape.slice(1).reduce((product, dimension) => product * BigInt(dimension), 1n);
  const byteLength = rowElements / blockElements * BigInt(traits.block_bytes) * rows;
  if (byteLength > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("GGUF tensor storage exceeds JavaScript exact integer range");
  return { status: "assessed", traits, byteLength: Number(byteLength) };
}

function exactElementCount(shape) {
  return (shape || []).reduce((product, dimension) => product * BigInt(dimension), 1n);
}

function safeBigInt(value) {
  return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function decimalRatio(numerator, denominator, places = 6) {
  if (denominator === 0n) return null;
  const scale = 10n ** BigInt(places);
  const rounded = (numerator * scale * 2n / denominator + 1n) / 2n;
  const whole = rounded / scale;
  const fraction = String(rounded % scale).padStart(places, "0").replace(/0+$/, "");
  return fraction ? `${whole}.${fraction}` : String(whole);
}

function duplicatePayloadGroups(tensors, numericalIntegrity) {
  const records = new Map((numericalIntegrity?.tensor_records || []).map((record) => [record.tensor_index, record]));
  const groups = new Map();
  for (const tensor of tensors) {
    const record = records.get(tensor.index);
    if (record?.status !== "assessed_full_payload" || !record.payload_sha256) continue;
    const key = `${tensor.dtype}\0${JSON.stringify(tensor.shape || [])}\0${tensor.byte_length}\0${record.payload_sha256}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(tensor);
  }
  return [...groups.values()].filter((group) => group.length > 1).map((group) => ({
    dtype: group[0].dtype,
    shape: [...(group[0].shape || [])],
    byte_length_each: Number(group[0].byte_length || 0),
    tensor_count: group.length,
    duplicate_bytes_after_first: Number(group[0].byte_length || 0) * (group.length - 1),
    payload_sha256: records.get(group[0].index).payload_sha256,
    tensor_indices: group.map((tensor) => tensor.index),
    tensor_names: group.map((tensor) => tensor.name),
    interpretation: "Content-addressed duplicate candidate: identical dtype, shape, byte length, and payload SHA-256. Runtime aliasing or semantic weight tying is not inferred.",
  })).sort((left, right) => right.duplicate_bytes_after_first - left.duplicate_bytes_after_first
    || left.tensor_names[0].localeCompare(right.tensor_names[0]));
}

export function buildTensorStorageSummary(format, tensors, numericalIntegrity = null) {
  const rows = new Map();
  let totalElements = 0n;
  let totalBytes = 0n;
  let unknownByteLengthTensorCount = 0;
  for (const tensor of tensors || []) {
    const elements = exactElementCount(tensor.shape);
    const byteLengthKnown = Number.isSafeInteger(tensor.byte_length) && tensor.byte_length >= 0;
    const bytes = BigInt(byteLengthKnown ? tensor.byte_length : 0);
    if (!byteLengthKnown) unknownByteLengthTensorCount += 1;
    totalElements += elements;
    totalBytes += bytes;
    const key = String(tensor.dtype || "UNKNOWN");
    const row = rows.get(key) || { dtype: key, tensor_count: 0, element_count: 0n, byte_length: 0n, unknown_byte_length_tensor_count: 0 };
    row.tensor_count += 1;
    row.element_count += elements;
    row.byte_length += bytes;
    if (!byteLengthKnown) row.unknown_byte_length_tensor_count += 1;
    rows.set(key, row);
  }
  const duplicateGroups = duplicatePayloadGroups(tensors || [], numericalIntegrity);
  const duplicateBytes = duplicateGroups.reduce((sum, group) => sum + group.duplicate_bytes_after_first, 0);
  const encodingRows = [...rows.values()].map((row) => ({
    dtype: row.dtype,
    tensor_count: row.tensor_count,
    element_count: safeBigInt(row.element_count),
    element_count_decimal: String(row.element_count),
    byte_length: safeBigInt(row.byte_length),
    byte_length_decimal: String(row.byte_length),
    unknown_byte_length_tensor_count: row.unknown_byte_length_tensor_count,
    effective_bits_per_element: row.unknown_byte_length_tensor_count ? null : decimalRatio(row.byte_length * 8n, row.element_count),
  })).sort((left, right) => {
    const leftBytes = BigInt(left.byte_length_decimal);
    const rightBytes = BigInt(right.byte_length_decimal);
    return leftBytes === rightBytes ? left.dtype.localeCompare(right.dtype) : leftBytes > rightBytes ? -1 : 1;
  });
  return {
    schema: "deepbom.tensor_storage_summary.v1",
    status: unknownByteLengthTensorCount ? "partial_unknown_serialized_byte_cardinality" : "assessed",
    evidence_class: "OBSERVED/DERIVED",
    format,
    tensor_count: (tensors || []).length,
    element_count: safeBigInt(totalElements),
    element_count_decimal: String(totalElements),
    byte_length: safeBigInt(totalBytes),
    byte_length_decimal: String(totalBytes),
    unknown_byte_length_tensor_count: unknownByteLengthTensorCount,
    effective_bits_per_element: unknownByteLengthTensorCount ? null : decimalRatio(totalBytes * 8n, totalElements),
    encoding_count: encodingRows.length,
    encodings: encodingRows,
    content_addressed_duplicate_group_count: duplicateGroups.length,
    content_addressed_duplicate_tensor_count: duplicateGroups.reduce((sum, group) => sum + group.tensor_count, 0),
    content_addressed_duplicate_bytes_after_first: duplicateBytes,
    duplicate_groups: duplicateGroups,
    duplicate_method: "Group only full-payload records with identical dtype, shape, byte length, and SHA-256; no payload rescan and no semantic alias claim.",
  };
}

function ggufArrayEntry(entries, key) {
  const row = entries.find((entry) => entry.key === key);
  return row?.value?.kind === "array" ? row.value : null;
}

function buildGgufSemanticContract(entries, metadata, tensors) {
  const architecture = typeof metadata["general.architecture"] === "string" ? metadata["general.architecture"] : null;
  const issues = [];
  const architectureFields = {};
  const recordPositiveInteger = (suffix) => {
    if (!architecture) return null;
    const key = `${architecture}.${suffix}`;
    if (!Object.hasOwn(metadata, key)) return null;
    const value = metadata[key];
    architectureFields[suffix] = value;
    if (!Number.isSafeInteger(value) || value <= 0) issues.push({ code: "GGUF_ARCHITECTURE_INTEGER_INVALID", key, value });
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  };
  const recordPositiveNumber = (suffix) => {
    if (!architecture) return null;
    const key = `${architecture}.${suffix}`;
    if (!Object.hasOwn(metadata, key)) return null;
    const value = metadata[key];
    architectureFields[suffix] = value;
    if (!Number.isFinite(value) || value <= 0) issues.push({ code: "GGUF_ARCHITECTURE_NUMBER_INVALID", key, value });
    return Number.isFinite(value) && value > 0 ? value : null;
  };
  const recordString = (suffix) => {
    if (!architecture) return null;
    const key = `${architecture}.${suffix}`;
    if (!Object.hasOwn(metadata, key)) return null;
    const value = metadata[key];
    architectureFields[suffix] = value;
    if (typeof value !== "string" || !value.trim()) issues.push({ code: "GGUF_ARCHITECTURE_STRING_INVALID", key, value });
    return typeof value === "string" && value.trim() ? value.trim() : null;
  };
  const contextLength = recordPositiveInteger("context_length");
  const embeddingLength = recordPositiveInteger("embedding_length");
  const blockCount = recordPositiveInteger("block_count");
  const feedForwardLength = recordPositiveInteger("feed_forward_length");
  const headCount = recordPositiveInteger("attention.head_count");
  const headCountKv = recordPositiveInteger("attention.head_count_kv");
  const keyLength = recordPositiveInteger("attention.key_length");
  const valueLength = recordPositiveInteger("attention.value_length");
  const ropeDimensionCount = recordPositiveInteger("rope.dimension_count");
  const ropeFrequencyBase = recordPositiveNumber("rope.freq_base");
  const ropeScalingType = recordString("rope.scaling.type");
  const ropeScalingFactor = recordPositiveNumber("rope.scaling.factor");
  if (embeddingLength && headCount && embeddingLength % headCount !== 0) {
    issues.push({ code: "GGUF_ATTENTION_HEAD_WIDTH_NONINTEGRAL", embedding_length: embeddingLength, head_count: headCount });
  }
  if (headCount && headCountKv && headCount % headCountKv !== 0) {
    issues.push({ code: "GGUF_ATTENTION_GQA_GROUP_NONINTEGRAL", head_count: headCount, head_count_kv: headCountKv });
  }
  const tokenArray = ggufArrayEntry(entries, "tokenizer.ggml.tokens");
  const scoreArray = ggufArrayEntry(entries, "tokenizer.ggml.scores");
  const typeArray = ggufArrayEntry(entries, "tokenizer.ggml.token_type");
  if (tokenArray && scoreArray && tokenArray.count !== scoreArray.count) {
    issues.push({ code: "GGUF_TOKENIZER_SCORE_CARDINALITY_MISMATCH", token_count: tokenArray.count, score_count: scoreArray.count });
  }
  if (tokenArray && typeArray && tokenArray.count !== typeArray.count) {
    issues.push({ code: "GGUF_TOKENIZER_TYPE_CARDINALITY_MISMATCH", token_count: tokenArray.count, token_type_count: typeArray.count });
  }
  const specialTokenRows = entries.filter((entry) => /^tokenizer\.ggml\..+_token_id$/.test(entry.key) && entry.value?.kind !== "array").map((entry) => ({
    key: entry.key,
    value: entry.value,
    in_vocabulary: Number.isSafeInteger(entry.value) && entry.value >= 0 && (!tokenArray || entry.value < tokenArray.count),
  }));
  for (const row of specialTokenRows) if (!row.in_vocabulary) {
    issues.push({ code: "GGUF_SPECIAL_TOKEN_ID_OUT_OF_RANGE", key: row.key, value: row.value, vocabulary_count: tokenArray?.count ?? null });
  }
  const assessedFieldCount = Object.keys(architectureFields).length + (tokenArray ? 1 : 0) + (scoreArray ? 1 : 0) + (typeArray ? 1 : 0) + specialTokenRows.length;
  const chatTemplate = typeof metadata["tokenizer.chat_template"] === "string" && metadata["tokenizer.chat_template"].length
    ? metadata["tokenizer.chat_template"] : null;
  const derivedHeadWidth = embeddingLength && headCount && embeddingLength % headCount === 0 ? embeddingLength / headCount : null;
  const effectiveKeyLength = keyLength || derivedHeadWidth;
  const effectiveValueLength = valueLength || derivedHeadWidth;
  const kvStateProjection = blockCount && headCountKv && effectiveKeyLength && effectiveValueLength && contextLength
    ? buildKvStateProjection({
        layerCount: blockCount,
        kvHeadCount: headCountKv,
        keyHeadWidth: effectiveKeyLength,
        valueHeadWidth: effectiveValueLength,
        contextLength,
      })
    : null;
  const vocabularyCount = tokenArray?.count ?? null;
  const computeProjectionEligible = isCanonicalGgufDecoderArchitecture(architecture)
    && vocabularyCount && embeddingLength && feedForwardLength && blockCount && headCount && headCountKv
    && effectiveKeyLength && effectiveKeyLength === effectiveValueLength && contextLength;
  const computeProjection = computeProjectionEligible
    ? buildCanonicalGatedDecoderProjection({
        vocabularySize: vocabularyCount,
        hiddenSize: embeddingLength,
        intermediateSize: feedForwardLength,
        layerCount: blockCount,
        attentionHeadCount: headCount,
        kvHeadCount: headCountKv,
        headWidth: effectiveKeyLength,
        contextLength,
      })
    : null;
  return {
    schema: "deepbom.gguf_semantic_contract.v1",
    status: issues.length ? "invalid" : assessedFieldCount ? "assessed" : "not_declared",
    evidence_class: "OBSERVED/DERIVED",
    architecture,
    architecture_fields: architectureFields,
    context_length: contextLength,
    embedding_length: embeddingLength,
    block_count: blockCount,
    feed_forward_length: feedForwardLength,
    attention_head_count: headCount,
    attention_head_count_kv: headCountKv,
    attention_key_length: keyLength,
    attention_value_length: valueLength,
    derived_attention_head_width: derivedHeadWidth,
    derived_gqa_query_heads_per_kv_head: headCount && headCountKv && headCount % headCountKv === 0 ? headCount / headCountKv : null,
    position_encoding: {
      status: ropeDimensionCount || ropeFrequencyBase || ropeScalingType || ropeScalingFactor ? "declared" : "not_declared",
      rope_dimension_count: ropeDimensionCount,
      rope_frequency_base: ropeFrequencyBase,
      rope_scaling_type: ropeScalingType,
      rope_scaling_factor: ropeScalingFactor,
    },
    kv_state_projection: kvStateProjection,
    compute_projection: computeProjection,
    compute_projection_status: computeProjection ? "assessed_registered_canonical_decoder_scenario"
      : !isCanonicalGgufDecoderArchitecture(architecture) ? "not_assessed_architecture_not_registered_for_canonical_decoder_projection"
        : "not_assessed_incomplete_or_nonuniform_attention_metadata",
    tokenizer: {
      status: tokenArray || specialTokenRows.length ? issues.some((issue) => issue.code.startsWith("GGUF_TOKENIZER") || issue.code.startsWith("GGUF_SPECIAL")) ? "invalid" : "assessed" : "not_declared",
      model: metadata["tokenizer.ggml.model"] || null,
      vocabulary_count: vocabularyCount,
      score_count: scoreArray?.count ?? null,
      token_type_count: typeArray?.count ?? null,
      merge_count: ggufArrayEntry(entries, "tokenizer.ggml.merges")?.count ?? null,
      special_token_count: specialTokenRows.length,
      special_tokens: specialTokenRows,
      chat_template: chatTemplate ? {
        status: "embedded_in_gguf",
        sha256: sha256TextHex(chatTemplate),
        utf8_byte_length: new TextEncoder().encode(chatTemplate).length,
      } : { status: "not_declared" },
      complete_array_values_retained: false,
      array_evidence: "Full array cardinalities are parsed; only bounded previews are retained in browser memory.",
    },
    serialized_tensor_count: tensors.length,
    assessed_field_count: assessedFieldCount,
    issue_count: issues.length,
    issues,
    boundary: "Internal metadata consistency, declared attention-state cardinality, and an explicit canonical decoder compute scenario where registered only; an execution DAG, architecture implementation, tokenizer behavior, cache storage dtype/layout, backend support, and runtime allocation are not inferred.",
  };
}

function ggufAnalysis(bytes, filename, fileSize) {
  if (bytes.length < 24) throw new NeedMoreData(24);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "GGUF") throw new Error("invalid GGUF magic");
  const versionLittle = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, true);
  const littleEndian = versionLittle === 2 || versionLittle === 3;
  const versionBig = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, false);
  const version = littleEndian ? versionLittle : versionBig;
  if (version !== 2 && version !== 3) throw new Error(`unsupported GGUF version ${version}`);
  if (!littleEndian && version !== 3) throw new Error("big-endian GGUF requires format version 3");
  const reader = new Reader(bytes, littleEndian);
  reader.offset = 8;
  const tensorCount = reader.u64();
  const metadataCount = reader.u64();
  if (tensorCount > 10_000_000 || metadataCount > 1_000_000) throw new Error("GGUF header count exceeds safety limit");
  const entries = [];
  const metadata = {};
  let invalidKeyCount = 0;
  for (let index = 0; index < metadataCount; index += 1) {
    const key = reader.string();
    if (!/^[a-z0-9_]+(?:\.[a-z0-9_]+)*$/.test(key)) invalidKeyCount += 1;
    const type = reader.u32();
    if (!GGUF_VALUE_TYPES[type]) throw new Error(`unknown GGUF metadata value type ${type}`);
    const value = parseGgufValue(reader, type);
    const summarized = value?.kind === "array" ? value : value;
    entries.push({ key, type: GGUF_VALUE_TYPES[type], value: summarized });
    if (value?.kind !== "array" && typeof value !== "object") metadata[key] = value;
  }
  // general.alignment is a UINT32 in the GGUF specification. Coercing the raw
  // metadata value would let a declared 0 fall through to the default and would
  // accept string or boolean declarations, so the type is checked first.
  const declaredAlignment = Object.hasOwn(metadata, "general.alignment") ? metadata["general.alignment"] : undefined;
  if (declaredAlignment !== undefined && typeof declaredAlignment !== "number") {
    throw new Error(`GGUF general.alignment must be a numeric metadata value, received ${typeof declaredAlignment}`);
  }
  const alignment = declaredAlignment === undefined ? 32 : declaredAlignment;
  if (!Number.isInteger(alignment) || alignment <= 0 || alignment > MIB || (alignment & (alignment - 1)) !== 0) throw new Error(`invalid GGUF alignment ${alignment}`);
  const tensors = [];
  let invalidOffsetCount = 0;
  for (let index = 0; index < tensorCount; index += 1) {
    const name = reader.string(64);
    const rank = reader.u32();
    if (rank > 16) throw new Error(`GGUF tensor rank ${rank} exceeds safety limit`);
    const shape = Array.from({ length: rank }, () => reader.u64());
    const ggmlType = reader.u32();
    const offset = reader.u64();
    if (offset % alignment !== 0) invalidOffsetCount += 1;
    const storage = ggmlTensorStorage(shape, ggmlType);
    tensors.push({
      index,
      name,
      shape,
      dtype: storage.traits?.name || `GGML_TYPE_${ggmlType}`,
      ggml_type: ggmlType,
      data_offset: offset,
      data_end: storage.byteLength == null ? null : offset + storage.byteLength,
      byte_length: storage.byteLength,
      block_elements: storage.traits?.block_elements || null,
      block_bytes: storage.traits?.block_bytes || null,
      storage_status: storage.status,
    });
  }
  const tensorDataOffset = Math.ceil(reader.offset / alignment) * alignment;
  if (tensorDataOffset > fileSize) throw new Error("GGUF tensor data offset exceeds file size");
  const payloadLength = fileSize - tensorDataOffset;
  let unsupportedTypeCount = 0;
  let invalidCardinalityCount = 0;
  let outOfBoundsRangeCount = 0;
  for (const tensor of tensors) {
    if (tensor.storage_status === "unsupported_type") unsupportedTypeCount += 1;
    else if (tensor.storage_status === "invalid_row_cardinality") invalidCardinalityCount += 1;
    if (tensor.data_offset > payloadLength || tensor.data_end != null && tensor.data_end > payloadLength) {
      invalidOffsetCount += 1;
      outOfBoundsRangeCount += 1;
    }
  }
  const knownRanges = tensors
    .filter((tensor) => tensor.data_end != null)
    .sort((left, right) => left.data_offset - right.data_offset || left.data_end - right.data_end || left.index - right.index);
  let rangeCursor = 0;
  let alignmentPaddingBytes = 0;
  let overlappingRangeCount = 0;
  for (const tensor of knownRanges) {
    if (tensor.data_offset < rangeCursor) overlappingRangeCount += 1;
    else alignmentPaddingBytes += tensor.data_offset - rangeCursor;
    rangeCursor = Math.max(rangeCursor, tensor.data_end);
  }
  const trailingBytes = Math.max(0, payloadLength - rangeCursor);
  alignmentPaddingBytes += trailingBytes;
  const declaredTensorBytes = tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
  const invalidStorage = invalidCardinalityCount + overlappingRangeCount + outOfBoundsRangeCount;
  const payloadCoverageStatus = invalidStorage
    ? "invalid_tensor_storage_ranges"
    : unsupportedTypeCount
      ? "partial_unsupported_ggml_types"
      : alignmentPaddingBytes
        ? "complete_with_alignment_padding"
        : "complete_without_gaps_or_overlaps";
  const architecture = String(metadata["general.architecture"] || "not_declared");
  const semanticContract = buildGgufSemanticContract(entries, metadata, tensors);
  const backendCompatibility = buildGgufBackendCompatibility({ architecture }, tensors);
  const analysis = containerAnalysis("gguf", filename, fileSize, tensors, {
    status: invalidKeyCount || invalidOffsetCount || invalidCardinalityCount || overlappingRangeCount ? "invalid" : unsupportedTypeCount ? "partial" : "assessed",
    metadata,
    metadata_entries: entries,
    gguf: {
      schema: "deepbom.gguf_metadata.v1",
      version,
      endianness: littleEndian ? "little" : "big",
      architecture,
      quantization_version: Number.isSafeInteger(metadata["general.quantization_version"])
        ? metadata["general.quantization_version"] : null,
      general_file_type: Number.isSafeInteger(metadata["general.file_type"])
        ? metadata["general.file_type"] : null,
      tensor_count: tensorCount,
      metadata_kv_count: metadataCount,
      alignment,
      tensor_data_offset: tensorDataOffset,
      payload_byte_length: payloadLength,
      declared_tensor_byte_length: declaredTensorBytes,
      alignment_padding_byte_length: alignmentPaddingBytes,
      payload_coverage_status: payloadCoverageStatus,
      unsupported_ggml_type_count: unsupportedTypeCount,
      invalid_tensor_cardinality_count: invalidCardinalityCount,
      overlapping_tensor_range_count: overlappingRangeCount,
      out_of_bounds_tensor_range_count: outOfBoundsRangeCount,
      invalid_metadata_key_count: invalidKeyCount,
      invalid_tensor_offset_count: invalidOffsetCount,
      type_traits_source: GGML_TYPE_TRAITS_SOURCE,
      semantic_contract: semanticContract,
      backend_compatibility: backendCompatibility,
      parser_scope: "GGUF metadata, tensor descriptors, source-pinned serialized byte ranges, and full-payload numerical integrity for supported scalar and GGML block encodings.",
    },
  });
  semanticContract.serialized_parameter_count = analysis.tensor_storage_summary.element_count;
  semanticContract.serialized_parameter_count_decimal = analysis.tensor_storage_summary.element_count_decimal;
  semanticContract.serialized_tensor_bytes = analysis.tensor_storage_summary.byte_length;
  semanticContract.serialized_tensor_bytes_decimal = analysis.tensor_storage_summary.byte_length_decimal;
  semanticContract.effective_bits_per_parameter = analysis.tensor_storage_summary.effective_bits_per_element;
  semanticContract.parameter_storage_basis = "Reused from deepbom.tensor_storage_summary.v1; tensor shape products and serialized payload bytes are not recomputed.";
  analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
  return analysis;
}

function safeTensorsAnalysis(bytes, filename, fileSize) {
  if (bytes.length < 9) throw new NeedMoreData(9);
  const reader = new Reader(bytes, true);
  const headerLength = reader.u64();
  if (headerLength < 2 || headerLength > MAX_SAFE_TENSORS_HEADER) throw new Error(`SafeTensors header length ${headerLength} is outside the supported range`);
  const headerEnd = 8 + headerLength;
  if (bytes.length < headerEnd) throw new NeedMoreData(headerEnd);
  if (bytes[8] !== 0x7b) throw new Error("SafeTensors header must begin with '{'");
  const headerText = decodeUtf8(bytes.subarray(8, headerEnd), "SafeTensors header").trimEnd();
  const header = parseStrictJson(headerText, "SafeTensors header");
  if (!header || typeof header !== "object" || Array.isArray(header)) throw new Error("SafeTensors header must be an object");
  const metadata = header.__metadata__ || {};
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata) || Object.values(metadata).some((value) => typeof value !== "string")) {
    throw new Error("SafeTensors __metadata__ must contain only string values");
  }
  const payloadLength = fileSize - headerEnd;
  if (payloadLength < 0) throw new Error("SafeTensors header exceeds file size");
  const tensors = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`invalid SafeTensors descriptor ${name}`);
    const bits = SAFE_TENSOR_BITS[value.dtype];
    if (!bits) throw new Error(`unsupported SafeTensors dtype ${value.dtype}`);
    const shape = value.shape;
    const offsets = value.data_offsets;
    if (!Array.isArray(shape) || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)) throw new Error(`invalid SafeTensors shape ${name}`);
    if (!Array.isArray(offsets) || offsets.length !== 2 || offsets.some((offset) => !Number.isSafeInteger(offset) || offset < 0) || offsets[1] < offsets[0]) {
      throw new Error(`invalid SafeTensors offsets ${name}`);
    }
    const elements = shape.reduce((product, dimension) => product * BigInt(dimension), 1n);
    const expectedBits = elements * BigInt(bits);
    if (expectedBits % 8n !== 0n) throw new Error(`SafeTensors sub-byte tensor ${name} is not byte-aligned`);
    const expectedBytes = Number(expectedBits / 8n);
    if (!Number.isSafeInteger(expectedBytes) || offsets[1] - offsets[0] !== expectedBytes) throw new Error(`SafeTensors byte cardinality mismatch ${name}`);
    tensors.push({ index: tensors.length, name, shape, dtype: value.dtype, data_offset: offsets[0], data_end: offsets[1], byte_length: expectedBytes });
  }
  const ranges = [...tensors].sort((left, right) => left.data_offset - right.data_offset || left.data_end - right.data_end);
  let cursor = 0;
  for (const tensor of ranges) {
    if (tensor.data_offset !== cursor) throw new Error(`SafeTensors payload gap or overlap before ${tensor.name}`);
    cursor = tensor.data_end;
  }
  if (cursor !== payloadLength) throw new Error(`SafeTensors payload coverage ${cursor}/${payloadLength}`);
  const analysis = containerAnalysis("safetensors", filename, fileSize, tensors, {
    status: "assessed",
    metadata,
    safetensors: {
      schema: "deepbom.safetensors_metadata.v1",
      header_byte_length: headerLength,
      tensor_count: tensors.length,
      payload_byte_length: payloadLength,
      payload_coverage_status: "complete_without_gaps_or_overlaps",
      duplicate_key_validation: "complete",
      reference_implementation: safeTensorsNumericalSourceEvidence(),
      quantization_contract: buildSafeTensorsQuantizationContract(null, tensors),
      parser_scope: "Header, exact tensor byte ranges, and full-payload numerical integrity for source-bound scalar, FP8, and PyTorch-packed F4 dtypes. F6 remains explicitly unassessed because SafeTensors does not define its packed-bit mapping.",
    },
  });
  analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
  return analysis;
}

function containerAnalysis(format, filename, fileSize, tensors, details) {
  const totalTensorBytes = tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
  const blockQuantizedTensorCount = format === "gguf"
    ? tensors.filter((tensor) => Number(tensor.block_elements || 0) > 1).length
    : 0;
  const scalarEncodedTensorCount = format === "gguf"
    ? tensors.filter((tensor) => tensor.storage_status === "assessed" && Number(tensor.block_elements || 0) === 1).length
    : 0;
  const unsupportedEncodingTensorCount = format === "gguf"
    ? tensors.filter((tensor) => tensor.storage_status !== "assessed").length
    : 0;
  return {
    schema: "deepbom.static_analysis.container.v1",
    format,
    filename,
    file_size: fileSize,
    file_size_bytes: fileSize,
    operator_count: null,
    tensor_count: tensors.length,
    quantized_tensors: blockQuantizedTensorCount,
    per_channel_tensors: 0,
    per_tensor_tensors: 0,
    total_macs: null,
    mac_assessment: { status: "not_applicable_weight_container", compute_ops: 0, assessed_compute_ops: 0 },
    ops: [],
    inputs: [],
    outputs: [],
    tensors,
    tensor_inventory: { status: details.status, tensor_count: tensors.length, total_declared_tensor_bytes: totalTensorBytes, tensors },
    tensor_storage_summary: buildTensorStorageSummary(format, tensors),
    metadata_presence: {
      status: "assessed",
      producer_name: details.metadata?.["general.name"] || null,
      producer_version: details.metadata?.["general.version"] || null,
      metadata_author: details.metadata?.["general.author"] || null,
      metadata_license: details.metadata?.["general.license"] || null,
      description: details.metadata?.["general.description"] || null,
      preprocessing_contract_status: "not_applicable_weight_container",
      output_semantics_documented: false,
    },
    quantization_status: {
      classification: format === "gguf" ? "block_or_tensor_encoded_weights" : "weight_container",
      label: format === "gguf" ? "GGUF tensor encodings" : "SafeTensors weight container",
      summary: format === "gguf"
        ? `${blockQuantizedTensorCount}/${tensors.length} tensors use source-pinned block-quantized GGML encodings.`
        : "Tensor payload dtypes are declared, but no execution-graph quantization contract is present.",
      detail: format === "gguf"
        ? `${blockQuantizedTensorCount} block-quantized, ${scalarEncodedTensorCount} scalar-encoded, ${unsupportedEncodingTensorCount} unsupported or invalid encoding tensor(s). GGML block encoding is not TFLite/ONNX affine per-tensor or per-axis quantization.`
        : "SafeTensors dtype and byte ranges are container metadata, not an executable quantization graph contract.",
      full_integer: false,
      compute_ops: 0,
      quantized_compute_ops: 0,
      quantized_compute_mac_percent: null,
      encoded_tensor_count: format === "gguf" ? tensors.length - unsupportedEncodingTensorCount : tensors.length,
      block_quantized_tensor_count: blockQuantizedTensorCount,
      scalar_encoded_tensor_count: scalarEncodedTensorCount,
      unsupported_encoding_tensor_count: unsupportedEncodingTensorCount,
      op_state_counts: [],
    },
    format_extensions: { [format]: details[format] },
    [format]: details[format],
  };
}

export function parseMetadataModel(bytes, filename, fileSize = bytes.length, format = "") {
  const normalized = String(format || "").toLowerCase();
  if (normalized === "gguf") return ggufAnalysis(bytes, filename, fileSize);
  if (normalized === "safetensors") return safeTensorsAnalysis(bytes, filename, fileSize);
  throw new Error(`metadata adapter is not implemented for ${normalized || "unknown"}`);
}

function attachNumericalIntegrity(analysis, numericalIntegrity) {
  analysis.tensor_numerical_integrity = numericalIntegrity;
  const byIndex = new Map(numericalIntegrity.tensor_records.map((record) => [record.tensor_index, record]));
  for (const tensor of analysis.tensors) tensor.numerical_integrity = byIndex.get(tensor.index) || null;
  analysis.tensor_inventory.numerical_integrity_status = numericalIntegrity.status;
  analysis.tensor_inventory.assessed_payload_bytes = numericalIntegrity.assessed_tensor_bytes;
  analysis.tensor_inventory.unassessed_payload_bytes = numericalIntegrity.unassessed_tensor_bytes;
  analysis.tensor_inventory.decoded_value_count = numericalIntegrity.decoded_value_count;
  analysis.tensor_storage_summary = buildTensorStorageSummary(analysis.format, analysis.tensors, numericalIntegrity);
  if (analysis.format === "safetensors") {
    analysis.safetensors.quantization_contract = buildSafeTensorsQuantizationContract(null, analysis.tensors);
    analysis.format_extensions.safetensors = analysis.safetensors;
  }
  return analysis;
}

export async function readMetadataModelFile(file, format, { onProgress, scanMode = "full" } = {}) {
  const normalized = String(format || "").toLowerCase();
  if (!["structure", "integrity", "full"].includes(scanMode)) throw new Error(`unsupported metadata scan mode ${scanMode}`);
  if (normalized === "safetensors") {
    const prefix = new Uint8Array(await file.slice(0, 8).arrayBuffer());
    if (prefix.length < 8) throw new Error("truncated SafeTensors length prefix");
    const headerLength = new Reader(prefix).u64();
    if (headerLength > MAX_SAFE_TENSORS_HEADER) throw new Error(`SafeTensors header exceeds ${MAX_SAFE_TENSORS_HEADER} bytes`);
    const bytes = new Uint8Array(await file.slice(0, 8 + headerLength).arrayBuffer());
    const analysis = safeTensorsAnalysis(bytes, file.name, file.size);
    if (scanMode === "structure") attachNumericalIntegrity(analysis, skippedNumericalIntegrity(analysis, file.size));
    else attachNumericalIntegrity(analysis, await scanSerializedTensorPayloads(file, analysis, { onProgress }));
    return { analysis, retainedBytes: bytes, payloadLoaded: false, payloadScanned: scanMode !== "structure" };
  }
  if (normalized === "gguf") {
    let length = Math.min(file.size, 64 * 1024);
    while (length <= Math.min(file.size, MAX_GGUF_HEADER)) {
      const bytes = new Uint8Array(await file.slice(0, length).arrayBuffer());
      try {
        const analysis = ggufAnalysis(bytes, file.name, file.size);
        if (scanMode === "structure") attachNumericalIntegrity(analysis, skippedNumericalIntegrity(analysis, file.size));
        else attachNumericalIntegrity(analysis, await scanSerializedTensorPayloads(file, analysis, { onProgress }));
        return { analysis, retainedBytes: bytes, payloadLoaded: false, payloadScanned: scanMode !== "structure" };
      } catch (error) {
        if (!(error instanceof NeedMoreData)) throw error;
        const next = Math.max(length * 2, error.requiredBytes);
        if (next <= length || next > MAX_GGUF_HEADER || length === file.size) throw new Error(`GGUF metadata header exceeds the ${MAX_GGUF_HEADER}-byte inspection limit`);
        length = Math.min(file.size, next);
      }
    }
  }
  throw new Error(`metadata adapter is not implemented for ${normalized || "unknown"}`);
}

function skippedNumericalIntegrity(analysis, sourceBytes) {
  const records = analysis.tensors.map((tensor) => ({
    schema: "deepbom.tensor_numerical_integrity.tensor.v1",
    status: "not_assessed_scan_policy_structure",
    evidence_class: "NOT_ASSESSABLE",
    tensor_index: tensor.index,
    tensor_name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    byte_length: Number.isSafeInteger(tensor.byte_length) ? tensor.byte_length : null,
    reason: "Payload numerical decoding was intentionally skipped by the structure scan policy.",
  }));
  const declaredBytes = analysis.tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0);
  return {
    schema: "deepbom.tensor_numerical_integrity.v1",
    status: "not_assessed_scan_policy_structure",
    evidence_class: "NOT_ASSESSABLE",
    scan_scope: "metadata_and_tensor_directory_only",
    tensor_count: records.length,
    assessed_tensor_count: 0,
    unassessed_tensor_count: records.length,
    declared_tensor_bytes: declaredBytes,
    assessed_tensor_bytes: 0,
    unassessed_tensor_bytes: declaredBytes,
    byte_conservation_status: "complete_descriptor_cardinality_payload_not_scanned",
    source_file_bytes: sourceBytes,
    decoded_value_count: 0,
    nonfinite_value_count: null,
    exact_zero_value_count: null,
    all_zero_tensor_count: null,
    constant_tensor_count: null,
    invalid_encoding_value_count: null,
    nonfinite_scale_block_count: null,
    tensor_records: records,
    decoder_source: analysis.format === "gguf" ? GGUF_DEQUANTIZATION_SOURCE : safeTensorsNumericalSourceEvidence(),
    limitations: [{ reason: "Payload numerical integrity requires --scan integrity or --scan full." }],
  };
}
