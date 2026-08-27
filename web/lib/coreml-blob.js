import { Sha256Accumulator } from "./sha256-sync.js";

const ALIGNMENT = 64;
const HEADER_BYTES = 64;
const METADATA_BYTES = 64;
const SENTINEL = 0xdeadbeef;
const CHUNK_BYTES = 1024 * 1024;

export const COREML_BLOB_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  storage_format: "mlmodel/src/MILBlob/Blob/StorageFormat.hpp",
  storage_format_sha256: "68b6ce0553a42176ed6847caf4d7605f374c028928cb8e255d53dcf431b944b3",
  dtype_source: "mlmodel/src/MILBlob/Blob/BlobDataType.hpp",
  dtype_source_sha256: "98ef66b4e25cb570c97bba816fee63054dc309353f679474eaa7278c3af833b3",
  reader_source: "mlmodel/src/MILBlob/Blob/StorageReader.cpp",
  reader_source_sha256: "9e62bbe03dc5cab71fc17c43a668c97d1cb4ef8d6702a66a9824fa457441dc6f",
  subbyte_source: "mlmodel/src/MILBlob/SubByteTypes.cpp",
  subbyte_source_sha256: "382df15c01e7a1dd0e48e1f3897c5a267e57769c484145ab76988e1c3bd2f4f4",
  fp8_source: "mlmodel/src/MILBlob/Fp8.cpp",
  fp8_source_sha256: "279c8ab85617289d844c4b63bef42cecdc177c77ba0066ea04529952ae90e35a",
});

const TYPES = Object.freeze({
  1: { dtype: "FLOAT16", bits: 16, kind: "float16" },
  2: { dtype: "FLOAT32", bits: 32, kind: "float32" },
  3: { dtype: "UINT8", bits: 8, kind: "uint" },
  4: { dtype: "INT8", bits: 8, kind: "int" },
  5: { dtype: "BFLOAT16", bits: 16, kind: "bfloat16" },
  6: { dtype: "INT16", bits: 16, kind: "int" },
  7: { dtype: "UINT16", bits: 16, kind: "uint" },
  8: { dtype: "INT4", bits: 4, kind: "int" },
  9: { dtype: "UINT1", bits: 1, kind: "uint" },
  10: { dtype: "UINT2", bits: 2, kind: "uint" },
  11: { dtype: "UINT4", bits: 4, kind: "uint" },
  12: { dtype: "UINT3", bits: 3, kind: "uint" },
  13: { dtype: "UINT6", bits: 6, kind: "uint" },
  14: { dtype: "INT32", bits: 32, kind: "int" },
  15: { dtype: "UINT32", bits: 32, kind: "uint" },
  16: { dtype: "FLOAT8E4M3FN", bits: 8, kind: "float8e4m3fn" },
  17: { dtype: "FLOAT8E5M2", bits: 8, kind: "float8e5m2" },
});

function align64(value) { return Math.ceil(value / ALIGNMENT) * ALIGNMENT; }

function safeNumber(value, label) {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range`);
  return Number(value);
}

function product(shape) {
  if (!Array.isArray(shape) || shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) return null;
  let result = 1;
  for (const value of shape) { if (result > Math.floor(Number.MAX_SAFE_INTEGER / value)) return null; result *= value; }
  return result;
}

function halfToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 31;
  const fraction = bits & 1023;
  if (!exponent) return fraction ? sign * fraction * 2 ** -24 : sign < 0 ? -0 : 0;
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function bfloatToNumber(bits) {
  const bytes = new ArrayBuffer(4);
  new DataView(bytes).setUint32(0, bits << 16, true);
  return new DataView(bytes).getFloat32(0, true);
}

function float8ToNumber(code, exponentBits, mantissaBits, bias, finiteOnly) {
  const sign = code & 0x80 ? -1 : 1;
  const exponentMask = (1 << exponentBits) - 1;
  const mantissaMask = (1 << mantissaBits) - 1;
  const exponent = code >>> mantissaBits & exponentMask;
  const mantissa = code & mantissaMask;
  if (finiteOnly && exponent === exponentMask && mantissa === mantissaMask) return Number.NaN;
  if (!finiteOnly && exponent === exponentMask) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  if (!exponent) return mantissa ? sign * (mantissa / 2 ** mantissaBits) * 2 ** (1 - bias) : sign < 0 ? -0 : 0;
  return sign * (1 + mantissa / 2 ** mantissaBits) * 2 ** (exponent - bias);
}

function statsState(type) {
  return {
    type, decoded_value_count: 0, finite_count: 0, zero_count: 0, negative_zero_count: 0,
    nan_count: 0, positive_infinity_count: 0, negative_infinity_count: 0,
    finite_min: null, finite_max: null, code_histogram: type.bits <= 8 && ["int", "uint"].includes(type.kind) ? new Array(2 ** type.bits).fill(0) : null,
  };
}

function observe(state, value, rawCode = null) {
  state.decoded_value_count += 1;
  if (rawCode != null && state.code_histogram) state.code_histogram[rawCode] += 1;
  if (Number.isNaN(value)) { state.nan_count += 1; return; }
  if (value === Number.POSITIVE_INFINITY) { state.positive_infinity_count += 1; return; }
  if (value === Number.NEGATIVE_INFINITY) { state.negative_infinity_count += 1; return; }
  state.finite_count += 1;
  if (value === 0) { state.zero_count += 1; if (Object.is(value, -0)) state.negative_zero_count += 1; }
  state.finite_min = state.finite_min == null ? value : Math.min(state.finite_min, value);
  state.finite_max = state.finite_max == null ? value : Math.max(state.finite_max, value);
}

function decodeByteAligned(state, bytes) {
  const type = state.type;
  const width = type.bits / 8;
  if (bytes.length % width) throw new Error(`Core ML blob ${type.dtype} chunk is not element aligned`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset < bytes.length; offset += width) {
    let value;
    if (type.kind === "float16") value = halfToNumber(view.getUint16(offset, true));
    else if (type.kind === "bfloat16") value = bfloatToNumber(view.getUint16(offset, true));
    else if (type.kind === "float32") value = view.getFloat32(offset, true);
    else if (type.kind === "float8e4m3fn") value = float8ToNumber(bytes[offset], 4, 3, 7, true);
    else if (type.kind === "float8e5m2") value = float8ToNumber(bytes[offset], 5, 2, 15, false);
    else if (type.kind === "int") value = width === 1 ? view.getInt8(offset) : width === 2 ? view.getInt16(offset, true) : view.getInt32(offset, true);
    else value = width === 1 ? view.getUint8(offset) : width === 2 ? view.getUint16(offset, true) : view.getUint32(offset, true);
    observe(state, value, type.bits <= 8 && ["int", "uint"].includes(type.kind) ? bytes[offset] : null);
  }
}

async function scanPayload(file, metadata, type, onProgress) {
  const digest = new Sha256Accumulator();
  const state = statsState(type);
  const expectedCount = (metadata.size_in_bytes * 8 - (type.bits < 8 ? metadata.padding_bits : 0)) / type.bits;
  let processed = 0;
  if (type.bits >= 8) {
    const width = type.bits / 8;
    const step = Math.max(width, Math.floor(CHUNK_BYTES / width) * width);
    for (let offset = 0; offset < metadata.size_in_bytes; offset += step) {
      const length = Math.min(step, metadata.size_in_bytes - offset);
      const bytes = new Uint8Array(await file.slice(metadata.data_offset + offset, metadata.data_offset + offset + length).arrayBuffer());
      digest.update(bytes);
      decodeByteAligned(state, bytes);
      processed += bytes.length;
      onProgress?.({ phase: "coreml_blob_payload", processed_bytes: processed, total_bytes: metadata.size_in_bytes, metadata_offset: metadata.metadata_offset });
    }
  } else {
    let bitBuffer = 0n;
    let bitCount = 0;
    const mask = (1n << BigInt(type.bits)) - 1n;
    let finalByte = null;
    for (let offset = 0; offset < metadata.size_in_bytes; offset += CHUNK_BYTES) {
      const length = Math.min(CHUNK_BYTES, metadata.size_in_bytes - offset);
      const bytes = new Uint8Array(await file.slice(metadata.data_offset + offset, metadata.data_offset + offset + length).arrayBuffer());
      digest.update(bytes);
      for (const byte of bytes) {
        finalByte = byte;
        bitBuffer |= BigInt(byte) << BigInt(bitCount);
        bitCount += 8;
        while (bitCount >= type.bits && state.decoded_value_count < expectedCount) {
          const code = Number(bitBuffer & mask);
          bitBuffer >>= BigInt(type.bits);
          bitCount -= type.bits;
          const value = type.kind === "int" && code & (1 << (type.bits - 1)) ? code - 2 ** type.bits : code;
          observe(state, value, code);
        }
      }
      processed += bytes.length;
      onProgress?.({ phase: "coreml_blob_payload", processed_bytes: processed, total_bytes: metadata.size_in_bytes, metadata_offset: metadata.metadata_offset });
    }
    if (metadata.padding_bits > 0) {
      const usedLowBits = 8 - metadata.padding_bits;
      const paddingMask = 0xff ^ (2 ** usedLowBits - 1);
      if ((finalByte & paddingMask) !== 0) throw new Error("Core ML blob has non-zero sub-byte padding bits");
    }
  }
  if (state.decoded_value_count !== expectedCount) throw new Error(`Core ML blob decoded ${state.decoded_value_count} values; expected ${expectedCount}`);
  const nonfinite = state.nan_count + state.positive_infinity_count + state.negative_infinity_count;
  const levels = state.code_histogram?.filter((count) => count > 0).length ?? null;
  delete state.type;
  delete state.code_histogram;
  return {
    status: "assessed_full_payload", byte_length: metadata.size_in_bytes, payload_sha256: digest.digestHex(), ...state,
    nonfinite_count: nonfinite, all_zero: state.decoded_value_count > 0 && state.zero_count === state.decoded_value_count,
    constant_finite: state.decoded_value_count > 0 && nonfinite === 0 && state.finite_min === state.finite_max,
    quant_code_levels_used: levels, quant_code_level_capacity: levels == null ? null : 2 ** type.bits,
    quant_code_utilization_ratio: levels == null ? null : levels / 2 ** type.bits,
  };
}

async function parseInventory(file) {
  if (file.size < HEADER_BYTES) throw new Error("Core ML blob storage file is shorter than its 64-byte header");
  const headerBytes = new Uint8Array(await file.slice(0, HEADER_BYTES).arrayBuffer());
  const header = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
  const count = header.getUint32(0, true);
  const version = header.getUint32(4, true);
  if (version !== 2) throw new Error(`Core ML blob storage version ${version} is unsupported; expected 2`);
  if (count > 1_000_000) throw new Error("Core ML blob storage count exceeds 1,000,000 entries");
  const entries = [];
  let metadataOffset = HEADER_BYTES;
  for (let index = 0; index < count; index += 1) {
    if (metadataOffset % ALIGNMENT || metadataOffset + METADATA_BYTES > file.size) throw new Error("Core ML blob metadata is unaligned or out of bounds");
    const bytes = new Uint8Array(await file.slice(metadataOffset, metadataOffset + METADATA_BYTES).arrayBuffer());
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const sentinel = view.getUint32(0, true);
    const typeId = view.getUint32(4, true);
    const size = safeNumber(view.getBigUint64(8, true), "Core ML blob size");
    const dataOffset = safeNumber(view.getBigUint64(16, true), "Core ML blob data offset");
    const paddingBits = safeNumber(view.getBigUint64(24, true), "Core ML blob padding bits");
    const type = TYPES[typeId];
    if (sentinel !== SENTINEL) throw new Error(`Core ML blob metadata at ${metadataOffset} has invalid sentinel`);
    if (!type) throw new Error(`Core ML blob metadata at ${metadataOffset} uses unknown dtype ${typeId}`);
    if (dataOffset % ALIGNMENT || dataOffset < metadataOffset + METADATA_BYTES || dataOffset + size > file.size) throw new Error(`Core ML blob data at metadata ${metadataOffset} is unaligned, overlapping, or out of bounds`);
    if (type.bits < 8) {
      if (paddingBits >= 8 || (size * 8 - paddingBits) % type.bits) throw new Error(`Core ML blob metadata at ${metadataOffset} has invalid sub-byte padding`);
    } else if (size * 8 % type.bits) throw new Error(`Core ML blob metadata at ${metadataOffset} has a non-integral ${type.dtype} cardinality`);
    const valueCount = (size * 8 - (type.bits < 8 ? paddingBits : 0)) / type.bits;
    if (!Number.isSafeInteger(valueCount)) throw new Error(`Core ML blob value cardinality at ${metadataOffset} is not lossless`);
    entries.push({ index, metadata_offset: metadataOffset, dtype_id: typeId, dtype: type.dtype, bits_per_value: type.bits, size_in_bytes: size, data_offset: dataOffset, padding_bits: paddingBits, value_count: valueCount, type });
    metadataOffset = align64(dataOffset + size);
  }
  return { version, declared_blob_count: count, entries, next_aligned_offset: metadataOffset };
}

export async function scanCoreMlBlobFile(file, bindings = [], { onProgress } = {}) {
  const inventory = await parseInventory(file);
  const bindingByOffset = new Map();
  for (const binding of bindings) {
    const offset = Number(binding.metadata_offset);
    if (!Number.isSafeInteger(offset)) throw new Error("Core ML blob binding offset is invalid");
    if (!bindingByOffset.has(offset)) bindingByOffset.set(offset, []);
    bindingByOffset.get(offset).push(binding);
  }
  const knownOffsets = new Set(inventory.entries.map((entry) => entry.metadata_offset));
  for (const offset of bindingByOffset.keys()) if (!knownOffsets.has(offset)) throw new Error(`Core ML MIL blob reference ${offset} does not resolve to storage metadata`);
  const records = [];
  for (let index = 0; index < inventory.entries.length; index += 1) {
    const entry = inventory.entries[index];
    const refs = bindingByOffset.get(entry.metadata_offset) || [];
    for (const ref of refs) {
      if (ref.dtype && ref.dtype !== entry.dtype) throw new Error(`Core ML blob ${entry.metadata_offset} dtype ${entry.dtype} does not match MIL tensor dtype ${ref.dtype}`);
      const expected = product(ref.shape);
      if (expected != null && expected !== entry.value_count) throw new Error(`Core ML blob ${entry.metadata_offset} has ${entry.value_count} values; MIL tensor ${ref.tensor_name || ref.tensor_index} expects ${expected}`);
    }
    onProgress?.({ phase: "coreml_blob", index, count: inventory.entries.length, metadata_offset: entry.metadata_offset });
    const numerical = await scanPayload(file, entry, entry.type, onProgress);
    records.push({
      ...entry, type: undefined, reference_count: refs.length,
      tensor_indices: [...new Set(refs.map((ref) => ref.tensor_index).filter(Number.isSafeInteger))],
      tensor_names: [...new Set(refs.map((ref) => ref.tensor_name).filter(Boolean))],
      numerical_integrity: numerical,
    });
  }
  const unreferenced = records.filter((record) => record.reference_count === 0);
  const totalPayload = records.reduce((sum, record) => sum + record.size_in_bytes, 0);
  const totalValues = records.reduce((sum, record) => sum + record.value_count, 0);
  const nonfinite = records.reduce((sum, record) => sum + record.numerical_integrity.nonfinite_count, 0);
  return {
    schema: "deepbom.coreml.blob_storage_integrity.v1",
    status: "assessed",
    evidence_class: "OBSERVED/DERIVED",
    file_name: file.name,
    file_size_bytes: file.size,
    storage_version: inventory.version,
    declared_blob_count: inventory.declared_blob_count,
    assessed_blob_count: records.length,
    referenced_blob_count: records.length - unreferenced.length,
    unreferenced_blob_count: unreferenced.length,
    payload_bytes: totalPayload,
    decoded_value_count: totalValues,
    nonfinite_value_count: nonfinite,
    all_zero_blob_count: records.filter((record) => record.numerical_integrity.all_zero).length,
    records,
    source_basis: COREML_BLOB_SOURCE,
  };
}
