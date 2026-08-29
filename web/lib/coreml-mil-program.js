import { coreMlExactLedger, multiplyCoreMlExactIntegers } from "./coreml-exact-integer.js";
import { convTransposeAxisPairs } from "./guarded-integer-expression.js";

const MAX_ITEMS = 1_000_000;
const MAX_RANK = 64;
const MAX_INLINE_SAMPLE = 64;

export const COREML_MIL_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  mil_proto: "mlmodel/format/MIL.proto",
  mil_proto_sha256: "eba3c58155319ffc9532aa9723c849e1b57eedc92942516e194a0779baa4fe6c",
  conv_definition: "coremltools/converters/mil/mil/ops/defs/iOS15/conv.py",
  conv_definition_sha256: "e024ee10b7fe3639bac1e5c829fd05e386d73f37e8e693ea08357b170b7803ab",
  linear_definition: "coremltools/converters/mil/mil/ops/defs/iOS15/linear.py",
  linear_definition_sha256: "71a34a2401481eb09837dd9baa9b904927b61142f9ee97ea535dc6725e7de08e",
  recurrent_definition: "coremltools/converters/mil/mil/ops/defs/iOS15/recurrent.py",
  recurrent_definition_sha256: "4e8f476c5fb39f0b94c3c5024beb3efc92967820612d5af00e6fe6f8bf6da97c",
  transformer_ios18_definition: "coremltools/converters/mil/mil/ops/defs/iOS18/transformers.py",
  transformer_ios18_definition_sha256: "1ea0654a6ea246a12e51723ccfe7770acac2711ca91a6e38d691267c3569dead",
  control_flow_definition: "coremltools/converters/mil/mil/ops/defs/iOS15/control_flow.py",
  control_flow_definition_sha256: "0a6b241f4decd1dea3b467b7966b18467a0e461c4014b8688049633c25797967",
  compression_ios18_definition: "coremltools/converters/mil/mil/ops/defs/iOS18/compression.py",
  compression_ios18_definition_sha256: "a48dfcb3d24f77bfbf1a5c3d1c27e813de344064b38afa9ea143386b9d8ca804",
  constexpr_ios16_definition: "coremltools/converters/mil/mil/ops/defs/iOS16/constexpr_ops.py",
  constexpr_ios16_definition_sha256: "2575e57c32f99c18e8e9d32af152ee19f9414835cb1c141ab19a92ec0b659a6d",
});

const DTYPES = Object.freeze({
  0: "UNUSED", 1: "BOOL", 2: "STRING", 10: "FLOAT16", 11: "FLOAT32", 12: "FLOAT64", 13: "BFLOAT16",
  21: "INT8", 22: "INT16", 23: "INT32", 24: "INT64", 25: "INT4", 31: "UINT8", 32: "UINT16",
  33: "UINT32", 34: "UINT64", 35: "UINT4", 36: "UINT2", 37: "UINT1", 38: "UINT6", 39: "UINT3",
  40: "FLOAT8E4M3FN", 41: "FLOAT8E5M2",
});

const DTYPE_BYTES = Object.freeze({ BOOL: 1, FLOAT16: 2, FLOAT32: 4, FLOAT64: 8, BFLOAT16: 2, INT8: 1, INT16: 2, INT32: 4, INT64: 8, UINT8: 1, UINT16: 2, UINT32: 4, UINT64: 8 });
const DTYPE_BITS = Object.freeze({
  BOOL: 8, STRING: null, FLOAT16: 16, FLOAT32: 32, FLOAT64: 64, BFLOAT16: 16,
  FLOAT8E4M3FN: 8, FLOAT8E5M2: 8, INT8: 8, INT16: 16, INT32: 32, INT64: 64,
  INT4: 4, UINT8: 8, UINT16: 16, UINT32: 32, UINT64: 64, UINT4: 4, UINT2: 2,
  UINT1: 1, UINT6: 6, UINT3: 3,
});

function boundedPush(rows, value, label) {
  if (rows.length >= MAX_ITEMS) throw new Error(`${label} exceeds ${MAX_ITEMS} entries`);
  rows.push(value);
}

function safeString(reader, wire, label) {
  const bytes = reader.bytesField(wire, label);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`${label} is not valid UTF-8`); }
}

function safeUint64(reader, wire, singular, field, label) {
  reader.requireWire(wire, 0, label);
  if (singular.has(field)) throw new Error(`${label} is repeated`);
  singular.add(field);
  const value = reader.rawVarint();
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range`);
  return Number(value);
}

function parseDimension(reader) {
  let value = null;
  let kind = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (kind) throw new Error("Core ML MIL Dimension contains multiple oneof values");
      kind = "constant";
      const constant = reader.message(wire, "MIL.Dimension.ConstantDimension");
      const singular = new Set();
      while (!constant.done) {
        const item = constant.key();
        if (item.field === 1) value = safeUint64(constant, item.wire, singular, 1, "MIL constant dimension size");
        else constant.skip(item.wire);
      }
      if (!Number.isSafeInteger(value) || value <= 0) throw new Error("Core ML MIL constant dimension is not positive");
    } else if (field === 2) {
      if (kind) throw new Error("Core ML MIL Dimension contains multiple oneof values");
      kind = "unknown";
      const unknown = reader.message(wire, "MIL.Dimension.UnknownDimension");
      let variadic = false;
      const singular = new Set();
      while (!unknown.done) {
        const item = unknown.key();
        if (item.field === 1) {
          const raw = unknown.intField(item.wire, singular, 1, "variadic");
          if (raw > 1) throw new Error("Core ML MIL variadic dimension flag is invalid");
          variadic = raw !== 0;
        } else unknown.skip(item.wire);
      }
      value = variadic ? "variadic" : null;
    } else reader.skip(wire);
  }
  if (!kind) throw new Error("Core ML MIL Dimension is missing its oneof value");
  return { kind, value };
}

function parseTensorType(reader) {
  let dtype = null;
  // TensorType.rank is a proto3 int64. Rank zero is therefore omitted by
  // conforming serializers for scalar tensors and must retain its default.
  let rank = 0;
  const dimensions = [];
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      const value = reader.intField(wire, singular, 1, "dataType");
      dtype = DTYPES[value] || `MIL_DTYPE_${value}`;
    } else if (field === 2) rank = reader.int64Field(wire, singular, 2, "rank");
    else if (field === 3) boundedPush(dimensions, parseDimension(reader.message(wire, "MIL.Dimension")), "Core ML MIL tensor rank");
    else reader.skip(wire);
  }
  if (!dtype || dtype === "UNUSED" || rank < -1 || rank > MAX_RANK) throw new Error("Core ML MIL TensorType has an invalid dtype or rank");
  if (rank >= 0 && dimensions.length !== rank) throw new Error("Core ML MIL TensorType rank does not match dimensions");
  if (rank === -1 && dimensions.length) throw new Error("Core ML MIL variable-rank TensorType must not declare fixed dimensions");
  return { kind: "tensor", dtype, rank, shape: dimensions.map((item) => item.kind === "constant" ? item.value : null), dimensions };
}

function parseValueType(reader) {
  let type = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (type) throw new Error("Core ML MIL ValueType contains multiple oneof values");
    if (field === 1) type = parseTensorType(reader.message(wire, "MIL.TensorType"));
    else if ([2, 3, 4, 5].includes(field)) {
      type = { kind: { 2: "list", 3: "tuple", 4: "dictionary", 5: "state" }[field], dtype: "NON_TENSOR", rank: null, shape: [] };
      reader.skip(wire);
    } else { reader.skip(wire); type = null; }
  }
  if (!type) throw new Error("Core ML MIL ValueType is missing a recognized type");
  return type;
}

function parseNamedValueType(reader) {
  let name = null;
  let type = null;
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) name = reader.stringField(wire, singular, 1, "name");
    else if (field === 2) {
      if (type) throw new Error("Core ML MIL NamedValueType repeats type");
      type = parseValueType(reader.message(wire, "MIL.ValueType"));
    } else reader.skip(wire);
  }
  if (!name || !type) throw new Error("Core ML MIL NamedValueType is incomplete");
  return { name, type };
}

function readPacked(reader, wire, width, label, decode) {
  const bytes = reader.bytesField(wire, label);
  if (bytes.length % width !== 0) throw new Error(`${label} is not ${width}-byte aligned`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  for (let offset = 0; offset < bytes.length; offset += width) if (values.length < MAX_INLINE_SAMPLE) values.push(decode(view, offset));
  return { count: bytes.length / width, values, truncated: bytes.length / width > values.length };
}

function parseIntegerValues(reader, wire, bits, signed, label, validate = null) {
  const values = [];
  let count = 0;
  const read = (source) => {
    const raw = source.rawVarint();
    const value = signed ? BigInt.asIntN(bits, raw) : BigInt.asUintN(bits, raw);
    if (validate && !validate(value)) throw new Error(`${label} contains an invalid value`);
    if (values.length < MAX_INLINE_SAMPLE) values.push(value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString());
    count += 1;
  };
  if (wire === 0) read(reader);
  else if (wire === 2) { const packed = reader.message(wire, label); while (!packed.done) read(packed); }
  else throw new Error(`${label} has unsupported wire type ${wire}`);
  return { count, values, truncated: count > values.length };
}

function parseTensorValue(reader) {
  let result = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.RepeatedFloats");
      result = { kind: "float32", count: 0, values: [], truncated: false };
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { let row; if (item.wire === 2) row = readPacked(payload, item.wire, 4, "MIL float values", (view, offset) => view.getFloat32(offset, true)); else if (item.wire === 5) { const start = payload.position; payload.advance(4); row = { count: 1, values: [new DataView(payload.bytes.buffer, payload.bytes.byteOffset + start, 4).getFloat32(0, true)] }; } else throw new Error("MIL float values have invalid wire type"); result.count += row.count; result.values.push(...row.values.slice(0, MAX_INLINE_SAMPLE - result.values.length)); } else payload.skip(item.wire); }
      result.truncated = result.count > result.values.length;
    } else if (field === 2 || field === 5) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, field === 2 ? "MIL.RepeatedInts" : "MIL.RepeatedLongInts");
      result = { kind: field === 2 ? "int32" : "int64", count: 0, values: [], truncated: false };
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { const row = parseIntegerValues(payload, item.wire, field === 2 ? 32 : 64, true, `MIL ${result.kind} values`); result.count += row.count; result.values.push(...row.values.slice(0, MAX_INLINE_SAMPLE - result.values.length)); } else payload.skip(item.wire); }
      result.truncated = result.count > result.values.length;
    } else if (field === 3) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.RepeatedBools");
      result = { kind: "bool", count: 0, values: [], truncated: false };
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { const row = parseIntegerValues(payload, item.wire, 64, false, "MIL bool values", (value) => value === 0n || value === 1n); result.count += row.count; result.values.push(...row.values.map(Boolean).slice(0, MAX_INLINE_SAMPLE - result.values.length)); } else payload.skip(item.wire); }
      result.truncated = result.count > result.values.length;
    } else if (field === 4) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.RepeatedStrings");
      result = { kind: "string", count: 0, values: [], truncated: false };
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { const value = safeString(payload, item.wire, "MIL string value"); if (result.values.length < MAX_INLINE_SAMPLE) result.values.push(value); result.count += 1; } else payload.skip(item.wire); }
      result.truncated = result.count > result.values.length;
    } else if (field === 6) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.RepeatedDoubles");
      result = { kind: "float64", count: 0, values: [], truncated: false };
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { let row; if (item.wire === 2) row = readPacked(payload, item.wire, 8, "MIL double values", (view, offset) => view.getFloat64(offset, true)); else if (item.wire === 1) { const start = payload.position; payload.advance(8); row = { count: 1, values: [new DataView(payload.bytes.buffer, payload.bytes.byteOffset + start, 8).getFloat64(0, true)] }; } else throw new Error("MIL double values have invalid wire type"); result.count += row.count; result.values.push(...row.values.slice(0, MAX_INLINE_SAMPLE - result.values.length)); } else payload.skip(item.wire); }
      result.truncated = result.count > result.values.length;
    } else if (field === 7) {
      if (result) throw new Error("Core ML MIL TensorValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.RepeatedBytes");
      result = { kind: "bytes", count: 0, values: [], byte_length: 0, byte_popcount: 0, last_byte: null, truncated: false };
      const singular = new Set();
      while (!payload.done) { const item = payload.key(); if (item.field === 1) { if (singular.has(1)) throw new Error("MIL RepeatedBytes repeats values"); singular.add(1); const bytes = payload.bytesField(item.wire, "MIL byte values"); result.count = bytes.length; result.byte_length = bytes.length; result.values = [...bytes.subarray(0, MAX_INLINE_SAMPLE)]; result.byte_popcount = bytes.reduce((sum, value) => sum + value.toString(2).replaceAll("0", "").length, 0); result.last_byte = bytes.length ? bytes.at(-1) : null; result.truncated = bytes.length > result.values.length; } else payload.skip(item.wire); }
    } else reader.skip(wire);
  }
  if (!result) throw new Error("Core ML MIL TensorValue is missing a recognized value");
  return result;
}

function staticCardinality(type) {
  if (type?.kind !== "tensor" || type.shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) return null;
  let count = 1;
  for (const value of type.shape) { if (count > Math.floor(Number.MAX_SAFE_INTEGER / value)) return null; count *= value; }
  return count;
}

function parseImmediateValue(reader, depth) {
  if (depth > 64) throw new Error("Core ML MIL immediate-value nesting exceeds 64");
  let result = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (result) throw new Error("Core ML MIL ImmediateValue contains multiple oneof values");
      result = parseTensorValue(reader.message(wire, "MIL.TensorValue"));
    }
    else if (field === 2 || field === 3) {
      if (result) throw new Error("Core ML MIL ImmediateValue contains multiple oneof values");
      const payload = reader.message(wire, field === 2 ? "MIL.TupleValue" : "MIL.ListValue");
      const values = [];
      let count = 0;
      while (!payload.done) {
        const item = payload.key();
        if (item.field === 1) {
          const value = parseValue(payload.message(item.wire, "MIL.Value"), depth + 1);
          if (values.length < MAX_INLINE_SAMPLE) values.push(value);
          count += 1;
          if (count > MAX_ITEMS) throw new Error("Core ML MIL immediate collection exceeds the bounded item limit");
        } else payload.skip(item.wire);
      }
      result = { kind: field === 2 ? "tuple" : "list", count, values, truncated: count > values.length };
    } else if (field === 4) {
      if (result) throw new Error("Core ML MIL ImmediateValue contains multiple oneof values");
      const payload = reader.message(wire, "MIL.DictionaryValue");
      const entries = [];
      let count = 0;
      while (!payload.done) {
        const item = payload.key();
        if (item.field !== 1) { payload.skip(item.wire); continue; }
        const pair = payload.message(item.wire, "MIL.DictionaryValue.KeyValuePair");
        let keyValue = null;
        let mappedValue = null;
        while (!pair.done) {
          const member = pair.key();
          if (member.field === 1) keyValue = parseValue(pair.message(member.wire, "MIL.Value"), depth + 1);
          else if (member.field === 2) mappedValue = parseValue(pair.message(member.wire, "MIL.Value"), depth + 1);
          else pair.skip(member.wire);
        }
        if (!keyValue || !mappedValue) throw new Error("Core ML MIL immediate dictionary entry is incomplete");
        if (entries.length < MAX_INLINE_SAMPLE) entries.push({ key: keyValue, value: mappedValue });
        count += 1;
        if (count > MAX_ITEMS) throw new Error("Core ML MIL immediate dictionary exceeds the bounded item limit");
      }
      result = { kind: "dictionary", count, entries, truncated: count > entries.length };
    } else reader.skip(wire);
  }
  if (!result) throw new Error("Core ML MIL immediate value has no supported oneof value");
  return result;
}

function parseValue(reader, depth = 0) {
  if (depth > 64) throw new Error("Core ML MIL value nesting exceeds 64");
  const result = { type: null, storage: null, immediate: null, blob: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) reader.stringField(wire, singular, 1, "docString");
    else if (field === 2) {
      if (result.type) throw new Error("Core ML MIL Value repeats type");
      result.type = parseValueType(reader.message(wire, "MIL.ValueType"));
    } else if (field === 3) {
      if (result.storage) throw new Error("Core ML MIL Value contains multiple storage oneof values");
      result.storage = "immediate";
      result.immediate = parseImmediateValue(reader.message(wire, "MIL.ImmediateValue"), depth + 1);
    } else if (field === 5) {
      if (result.storage) throw new Error("Core ML MIL Value contains multiple storage oneof values");
      result.storage = "blob";
      const blob = reader.message(wire, "MIL.BlobFileValue");
      let file_name = null;
      let offset = null;
      const fields = new Set();
      while (!blob.done) {
        const item = blob.key();
        if (item.field === 1) file_name = blob.stringField(item.wire, fields, 1, "fileName");
        else if (item.field === 2) offset = safeUint64(blob, item.wire, fields, 2, "MIL blob metadata offset");
        else blob.skip(item.wire);
      }
      if (!file_name || offset == null) throw new Error("Core ML MIL blob reference is incomplete");
      result.blob = { file_name, metadata_offset: offset };
    } else reader.skip(wire);
  }
  if (!result.type || !result.storage) throw new Error("Core ML MIL Value is missing type or storage");
  const expected = staticCardinality(result.type);
  if (expected != null && result.immediate?.kind === "bytes") {
    const bits = DTYPE_BITS[result.type.dtype];
    if (!Number.isSafeInteger(bits) || bits <= 0) throw new Error(`Core ML MIL byte-backed immediate tensor has unsupported dtype ${result.type.dtype}`);
    const expectedBits = expected * bits;
    const expectedBytes = Number.isSafeInteger(expectedBits) ? Math.ceil(expectedBits / 8) : null;
    if (expectedBytes == null || result.immediate.byte_length !== expectedBytes) {
      throw new Error(`Core ML MIL byte-backed immediate tensor has ${result.immediate.byte_length} bytes; expected ${expectedBytes ?? "an exact bounded cardinality"}`);
    }
    const paddingBits = expectedBytes * 8 - expectedBits;
    if (paddingBits > 0) {
      const usedLowBits = 8 - paddingBits;
      const paddingMask = 0xff ^ (2 ** usedLowBits - 1);
      if ((result.immediate.last_byte & paddingMask) !== 0) throw new Error("Core ML MIL sub-byte immediate tensor has non-zero padding bits");
    }
    result.immediate.logical_count = expected;
    result.immediate.bits_per_value = bits;
    result.immediate.padding_bits = paddingBits;
    result.immediate.packing = bits < 8 ? "little_endian_contiguous_lsb_first" : "byte_aligned";
    // Stored bytes are not semantic scalar samples until their dtype-specific
    // decoder has been applied, so arithmetic rules must not consume them.
    result.immediate.truncated = true;
  } else if (result.type.kind === "tensor" && expected != null && result.immediate?.count != null && expected !== result.immediate.count) {
    throw new Error(`Core ML MIL immediate tensor has ${result.immediate.count} values; expected ${expected}`);
  }
  return result;
}

function parseBinding(reader) {
  let result = null;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (result) throw new Error("Core ML MIL Binding contains multiple oneof values");
      result = { kind: "name", name: safeString(reader, wire, "MIL binding name") };
    } else if (field === 2) {
      if (result) throw new Error("Core ML MIL Binding contains multiple oneof values");
      result = { kind: "value", value: parseValue(reader.message(wire, "MIL.Value")) };
    }
    else reader.skip(wire);
  }
  if (!result) throw new Error("Core ML MIL Binding is empty");
  return result;
}

function parseArgument(reader) {
  const bindings = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(bindings, parseBinding(reader.message(wire, "MIL.Argument.Binding")), "Core ML MIL argument bindings");
    else reader.skip(wire);
  }
  return bindings;
}

function parseMapEntry(reader, label, parse) {
  let key = null;
  let value = null;
  const singular = new Set();
  while (!reader.done) {
    const item = reader.key();
    if (item.field === 1) key = reader.stringField(item.wire, singular, 1, "key");
    else if (item.field === 2) { if (value != null) throw new Error(`${label} repeats value`); value = parse(reader.message(item.wire, `${label}.value`)); }
    else reader.skip(item.wire);
  }
  if (!key || value == null) throw new Error(`${label} is incomplete`);
  return [key, value];
}

function parseOperation(reader, depth) {
  if (depth > 64) throw new Error("Core ML MIL nested block depth exceeds 64");
  const result = { type: null, inputs: {}, outputs: [], blocks: [], attributes: {} };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.type = reader.stringField(wire, singular, 1, "type");
    else if (field === 2) {
      const [name, value] = parseMapEntry(reader.message(wire, "MIL.Operation.inputs"), "MIL operation input map", parseArgument);
      if (Object.hasOwn(result.inputs, name)) throw new Error(`Core ML MIL operation repeats input ${name}`);
      result.inputs[name] = value;
    } else if (field === 3) boundedPush(result.outputs, parseNamedValueType(reader.message(wire, "MIL.NamedValueType")), "Core ML MIL operation outputs");
    else if (field === 4) boundedPush(result.blocks, parseBlock(reader.message(wire, "MIL.Block"), depth + 1), "Core ML MIL nested blocks");
    else if (field === 5) {
      const [name, value] = parseMapEntry(reader.message(wire, "MIL.Operation.attributes"), "MIL operation attribute map", parseValue);
      if (Object.hasOwn(result.attributes, name)) throw new Error(`Core ML MIL operation repeats attribute ${name}`);
      result.attributes[name] = value;
    } else reader.skip(wire);
  }
  if (!result.type || !result.outputs.length) throw new Error("Core ML MIL Operation is missing type or outputs");
  return result;
}

function parseBlock(reader, depth = 0) {
  const result = { inputs: [], outputs: [], operations: [], attributes: {} };
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(result.inputs, parseNamedValueType(reader.message(wire, "MIL.NamedValueType")), "Core ML MIL block inputs");
    else if (field === 2) boundedPush(result.outputs, safeString(reader, wire, "MIL block output"), "Core ML MIL block outputs");
    else if (field === 3) boundedPush(result.operations, parseOperation(reader.message(wire, "MIL.Operation"), depth), "Core ML MIL operations");
    else if (field === 4) {
      const [name, value] = parseMapEntry(reader.message(wire, "MIL.Block.attributes"), "MIL block attribute map", parseValue);
      if (Object.hasOwn(result.attributes, name)) throw new Error(`Core ML MIL block repeats attribute ${name}`);
      result.attributes[name] = value;
    } else reader.skip(wire);
  }
  return result;
}

function parseFunction(reader) {
  const result = { inputs: [], opset: null, block_specializations: {}, attributes: {} };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) boundedPush(result.inputs, parseNamedValueType(reader.message(wire, "MIL.NamedValueType")), "Core ML MIL function inputs");
    else if (field === 2) result.opset = reader.stringField(wire, singular, 2, "opset");
    else if (field === 3) {
      const [name, block] = parseMapEntry(reader.message(wire, "MIL.Function.blockSpecializations"), "MIL block specialization map", (value) => parseBlock(value));
      if (Object.hasOwn(result.block_specializations, name)) throw new Error(`Core ML MIL function repeats specialization ${name}`);
      result.block_specializations[name] = block;
    } else if (field === 4) {
      const [name, value] = parseMapEntry(reader.message(wire, "MIL.Function.attributes"), "MIL function attribute map", parseValue);
      if (Object.hasOwn(result.attributes, name)) throw new Error(`Core ML MIL function repeats attribute ${name}`);
      result.attributes[name] = value;
    } else reader.skip(wire);
  }
  if (!result.opset || !Object.hasOwn(result.block_specializations, result.opset)) throw new Error("Core ML MIL function opset does not resolve to a block specialization");
  result.active_block = result.block_specializations[result.opset];
  return result;
}

export function parseCoreMlMilFunctionEntry(reader) {
  return parseMapEntry(reader, "MIL function map", parseFunction);
}

export function parseCoreMlMilAttributeEntry(reader) {
  return parseMapEntry(reader, "MIL program attribute map", parseValue);
}

export function finalizeCoreMlMilProgram(result) {
  if (result.version == null || !Object.keys(result.functions || {}).length) throw new Error("Core ML MIL Program is missing version or functions");
  result.function_count = Object.keys(result.functions).length;
  result.source_basis = COREML_MIL_SOURCE;
  return result;
}

export function parseCoreMlMilProgram(reader) {
  const result = { version: null, functions: {}, attributes: {}, source_basis: COREML_MIL_SOURCE };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.version = reader.int64Field(wire, singular, 1, "version");
    else if (field === 2) {
      const [name, value] = parseCoreMlMilFunctionEntry(reader.message(wire, "MIL.Program.functions"));
      if (Object.hasOwn(result.functions, name)) throw new Error(`Core ML MIL program repeats function ${name}`);
      result.functions[name] = value;
    } else if (field === 3) reader.stringField(wire, singular, 3, "docString");
    else if (field === 4) {
      const [name, value] = parseCoreMlMilAttributeEntry(reader.message(wire, "MIL.Program.attributes"));
      if (Object.hasOwn(result.attributes, name)) throw new Error(`Core ML MIL program repeats attribute ${name}`);
      result.attributes[name] = value;
    } else reader.skip(wire);
  }
  return finalizeCoreMlMilProgram(result);
}

function safeProduct(shape) {
  if (!shape?.length || shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) return null;
  let product = 1;
  for (const value of shape) { if (product > Math.floor(Number.MAX_SAFE_INTEGER / value)) return null; product *= value; }
  return product;
}

const MIL_KNOWN_NON_MAC = new Set([
  "const", "identity", "cast", "reshape", "reshape_like", "squeeze", "expand_dims", "flatten2d", "transpose",
  "slice_by_index", "slice_by_size", "split", "concat", "stack", "tile", "pad", "crop", "gather", "gather_nd",
  "scatter", "scatter_nd", "select", "shape", "rank", "range_1d", "fill", "non_zero", "argsort", "topk",
  "add", "sub", "mul", "real_div", "floor_div", "mod", "pow", "maximum", "minimum", "avg_pool", "max_pool",
  "reduce_argmax", "reduce_argmin", "reduce_l1_norm", "reduce_l2_norm", "reduce_log_sum", "reduce_log_sum_exp",
  "reduce_max", "reduce_mean", "reduce_min", "reduce_prod", "reduce_sum", "reduce_sum_square",
  "relu", "relu6", "leaky_relu", "prelu", "elu", "gelu", "sigmoid", "softmax", "log_softmax", "tanh",
  "abs", "acos", "asin", "atan", "atanh", "ceil", "clip", "cos", "cosh", "erf", "exp", "exp2", "floor",
  "inverse", "log", "logical_and", "logical_not", "logical_or", "logical_xor", "round", "rsqrt", "sign", "sin",
  "sinh", "sqrt", "square", "threshold", "quantize", "dequantize",
]);

function sameStaticShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

const IOS18_BLOCKWISE_DATA_DTYPES = new Set(["INT4", "UINT4", "INT8", "UINT8", "FLOAT16", "FLOAT32"]);
const IOS18_LUT_INDEX_DTYPES = new Set(["UINT1", "UINT2", "UINT3", "UINT4", "UINT6", "UINT8"]);

function exactBinding(inputBindings, tensors, argument) {
  const bindings = inputBindings?.[argument];
  if (!Array.isArray(bindings) || bindings.length !== 1) return null;
  const binding = bindings[0];
  if (Number.isSafeInteger(binding.tensor_index)) return tensors[binding.tensor_index] || null;
  if (binding.kind !== "value") return null;
  return tensorFromMilValue(binding.value);
}

function tensorFromMilValue(value) {
  if (value?.type?.kind !== "tensor") return null;
  return {
    dtype: value.type.dtype, shape: value.type.shape,
    immediate_value: value.immediate, blob_reference: value.blob,
  };
}

function exactAttributeBinding(attributes, argument) {
  return tensorFromMilValue(attributes?.[argument]);
}

function exactScalarBinding(inputBindings, tensors, argument) {
  const tensor = exactBinding(inputBindings, tensors, argument);
  const immediate = tensor?.immediate_value;
  return tensor?.shape?.length === 0 && immediate && !immediate.truncated && immediate.count === 1
    ? immediate.values[0] : null;
}

function coreMlOpsetGeneration(opset) {
  const match = /^CoreML(\d+)$/.exec(String(opset || ""));
  return match ? Number(match[1]) : null;
}

function exactImmediateShape(tensor) {
  const immediate = tensor?.immediate_value;
  if (tensor?.dtype !== "UINT32" || tensor.shape?.length !== 1 || !immediate) return null;
  const count = tensor.shape[0];
  let values = null;
  if (immediate.kind === "bytes" && immediate.byte_length === count * 4 && immediate.values.length === immediate.byte_length) {
    const bytes = Uint8Array.from(immediate.values);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    values = Array.from({ length: count }, (_, index) => view.getUint32(index * 4, true));
  } else if (!immediate.truncated && immediate.count === count && immediate.values.length === count) {
    values = [...immediate.values];
  }
  return values?.every((value) => Number.isSafeInteger(value) && value > 0) ? values : null;
}

function exactLegacyLutContract(outputShapes, outputIds, inputBindings, tensors, attributes) {
  const indices = exactBinding(inputBindings, tensors, "indices") || exactAttributeBinding(attributes, "indices");
  const lut = exactBinding(inputBindings, tensors, "lut") || exactAttributeBinding(attributes, "lut");
  const shapeTensor = exactBinding(inputBindings, tensors, "shape") || exactAttributeBinding(attributes, "shape");
  if (!indices || !lut || !shapeTensor) {
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense is missing indices, LUT, or shape binding; serialized arguments: ${Object.keys(inputBindings || {}).sort().join(", ") || "none"}; attributes: ${Object.keys(attributes || {}).sort().join(", ") || "none"}`);
  }
  const outputShape = exactImmediateShape(shapeTensor);
  if (indices.dtype !== "UINT8" || indices.shape.length !== 1) {
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense indices must be rank-1 UINT8, observed ${indices.dtype} ${JSON.stringify(indices.shape)}`);
  }
  if (!["INT8", "UINT8", "FLOAT16", "FLOAT32"].includes(lut.dtype) || lut.shape.length !== 1) {
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense LUT must be a supported rank-1 tensor, observed ${lut.dtype} ${JSON.stringify(lut.shape)}`);
  }
  if (!outputShape) {
    const immediate = shapeTensor.immediate_value;
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense shape must be a complete positive UINT32 vector; observed ${shapeTensor.dtype} ${JSON.stringify(shapeTensor.shape)}, immediate ${immediate ? `${immediate.count}/${immediate.values?.length || 0}${immediate.truncated ? " truncated" : ""}` : "absent"}`);
  }
  const paletteCount = lut.shape[0];
  if (![2, 4, 16, 64, 256].includes(paletteCount)) {
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense palette cardinality ${paletteCount} is outside the pinned set`);
  }
  const outputElements = safeProduct(outputShape);
  const indexBits = Math.log2(paletteCount);
  const packedIndexBytes = Math.ceil(indexBits * outputElements / 8);
  if (indices.shape[0] !== packedIndexBytes) {
    throw new Error(`Core ML MIL CoreML6 constexpr_lut_to_dense packed index length ${indices.shape[0]} contradicts ${packedIndexBytes} bytes derived from shape and palette`);
  }
  if (!sameStaticShape(outputShapes[0], outputShape) || tensors[outputIds[0]]?.dtype !== lut.dtype) {
    throw new Error("Core ML MIL CoreML6 constexpr_lut_to_dense output ValueType contradicts pinned type inference");
  }
  return {
    schema: "deepbom.coreml.mil_compression_transform.v1", status: "assessed_exact_serialized_contract",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: "constexpr_lut_to_dense", representation: "packed_index_lut_palettization",
    logical_index_elements: outputElements, logical_output_elements: outputElements,
    serialized_index_bytes: packedIndexBytes, index_bits: indexBits, palette_count: paletteCount,
    vector_size: 1, vector_axis: null, index_shape: [...indices.shape], lut_shape: [...lut.shape], output_shape: outputShape,
    indices_tensor_index: Number.isSafeInteger(indices.index) ? indices.index : null,
    lut_tensor_index: Number.isSafeInteger(lut.index) ? lut.index : null,
    source_file: COREML_MIL_SOURCE.constexpr_ios16_definition,
    source_sha256: COREML_MIL_SOURCE.constexpr_ios16_definition_sha256,
    boundary: "Serialized iOS 16 packed-index palette cardinality and output type/shape contract only. LUT usage distribution, decompressed numerical quality, physical runtime storage, device placement, and timing are not inferred.",
  };
}

function exactLegacySparseContract(outputShapes, outputIds, inputBindings, tensors, attributes) {
  const data = exactBinding(inputBindings, tensors, "nonzero_data") || exactAttributeBinding(attributes, "nonzero_data");
  const mask = exactBinding(inputBindings, tensors, "mask") || exactAttributeBinding(attributes, "mask");
  const shapeTensor = exactBinding(inputBindings, tensors, "shape") || exactAttributeBinding(attributes, "shape");
  if (!data || !mask || !shapeTensor) {
    throw new Error("Core ML MIL CoreML6 constexpr_sparse_to_dense is missing nonzero_data, mask, or shape binding");
  }
  const outputShape = exactImmediateShape(shapeTensor);
  if (!outputShape || data.shape.length !== 1 || mask.dtype !== "UINT8" || mask.shape.length !== 1
    || !["INT8", "UINT8", "FLOAT16", "FLOAT32"].includes(data.dtype)) {
    throw new Error("Core ML MIL CoreML6 constexpr_sparse_to_dense violates pinned rank, dtype, mask, or shape constraints");
  }
  const outputElements = safeProduct(outputShape);
  const packedMaskBytes = Math.ceil(outputElements / 8);
  if (mask.shape[0] !== packedMaskBytes || !sameStaticShape(outputShapes[0], outputShape)
    || tensors[outputIds[0]]?.dtype !== data.dtype) {
    throw new Error("Core ML MIL CoreML6 constexpr_sparse_to_dense packed mask or output ValueType contradicts pinned type inference");
  }
  const immediate = mask.immediate_value;
  const maskPopulation = immediate?.kind === "bytes" && immediate.byte_length === packedMaskBytes
    && Number.isSafeInteger(immediate.byte_popcount) ? immediate.byte_popcount : null;
  if (maskPopulation != null && maskPopulation !== data.shape[0]) {
    throw new Error("Core ML MIL CoreML6 constexpr_sparse_to_dense mask population does not equal nonzero_data cardinality");
  }
  const paddingBits = packedMaskBytes * 8 - outputElements;
  if (immediate?.kind === "bytes" && immediate.byte_length === packedMaskBytes && paddingBits > 0) {
    const usedLowBits = 8 - paddingBits;
    const paddingMask = 0xff ^ (2 ** usedLowBits - 1);
    if ((immediate.last_byte & paddingMask) !== 0) {
      throw new Error("Core ML MIL CoreML6 constexpr_sparse_to_dense has non-zero padding bits");
    }
  }
  return {
    schema: "deepbom.coreml.mil_compression_transform.v1",
    status: maskPopulation == null ? "assessed_shape_contract_mask_population_unresolved" : "assessed_exact_serialized_contract",
    evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: "constexpr_sparse_to_dense",
    representation: "packed_unstructured_sparse_bitmask", stored_nonzero_elements: data.shape[0],
    logical_output_elements: outputElements, serialized_mask_bytes: packedMaskBytes, mask_shape: [...mask.shape],
    output_shape: outputShape, mask_population: maskPopulation, padding_bits: paddingBits,
    nonzero_data_tensor_index: Number.isSafeInteger(data.index) ? data.index : null,
    mask_tensor_index: Number.isSafeInteger(mask.index) ? mask.index : null,
    mask_population_status: maskPopulation == null ? "not_decoded_at_mil_contract_layer" : "assessed_exact_immediate_payload",
    source_file: COREML_MIL_SOURCE.constexpr_ios16_definition,
    source_sha256: COREML_MIL_SOURCE.constexpr_ios16_definition_sha256,
    boundary: maskPopulation == null
      ? "Serialized iOS 16 packed-mask cardinality and output type/shape are exact. Mask population remains unresolved until its payload is decoded."
      : "Serialized iOS 16 packed-mask cardinality, little-endian mask population, zero padding, and output type/shape are exact. Runtime storage, placement, and timing are not inferred.",
  };
}

function exactCompressionContract(type, outputShapes, outputIds, inputBindings, tensors, opset, attributes) {
  const name = String(type || "").toLowerCase();
  const opsetGeneration = coreMlOpsetGeneration(opset);
  if (name.startsWith("constexpr_") && opsetGeneration == null) {
    throw new Error(`Core ML MIL ${name} has no recognized CoreML<N> opset binding`);
  }
  if (name === "constexpr_affine_dequantize") {
    if (opsetGeneration < 6) {
      throw new Error("Core ML MIL constexpr_affine_dequantize is unavailable before the pinned CoreML6 opset");
    }
    const data = exactBinding(inputBindings, tensors, "quantized_data");
    const zeroPoint = exactBinding(inputBindings, tensors, "zero_point");
    const scale = exactBinding(inputBindings, tensors, "scale");
    const axisTensor = exactBinding(inputBindings, tensors, "axis");
    const axisRaw = exactScalarBinding(inputBindings, tensors, "axis");
    if (!data || !zeroPoint || !scale || !axisTensor || axisRaw == null) {
      throw new Error("Core ML MIL constexpr_affine_dequantize is missing a serialized quantized_data, zero_point, scale, or scalar axis binding");
    }
    if (data.shape.length < 1 || !["INT8", "UINT8"].includes(data.dtype)
      || !["INT8", "UINT8", "FLOAT32"].includes(zeroPoint.dtype)
      || !["FLOAT16", "FLOAT32"].includes(scale.dtype)
      || axisTensor.dtype !== "INT32" || axisTensor.shape.length !== 0
      || !Number.isSafeInteger(axisRaw)) {
      throw new Error("Core ML MIL constexpr_affine_dequantize violates the pinned rank or dtype domains");
    }
    if (![0, 1].includes(scale.shape.length) || ![0, 1].includes(zeroPoint.shape.length)) {
      throw new Error("Core ML MIL constexpr_affine_dequantize scale and zero_point must each be a scalar or vector");
    }
    if (axisRaw < -data.shape.length || axisRaw >= data.shape.length) {
      throw new Error("Core ML MIL constexpr_affine_dequantize axis is outside the pinned quantized_data rank range");
    }
    const normalizedAxis = axisRaw < 0 ? axisRaw + data.shape.length : axisRaw;
    const axisExtent = data.shape[normalizedAxis];
    if (!Number.isSafeInteger(axisExtent) || axisExtent <= 0
      || scale.shape.length === 1 && scale.shape[0] !== axisExtent
      || zeroPoint.shape.length === 1 && zeroPoint.shape[0] !== axisExtent) {
      throw new Error("Core ML MIL constexpr_affine_dequantize vector cardinality contradicts the selected quantized_data axis");
    }
    if (!sameStaticShape(outputShapes[0], data.shape) || tensors[outputIds[0]]?.dtype !== scale.dtype) {
      throw new Error("Core ML MIL constexpr_affine_dequantize output ValueType contradicts pinned type inference");
    }
    const scaleElements = scale.shape.length ? scale.shape[0] : 1;
    const zeroPointElements = zeroPoint.shape.length ? zeroPoint.shape[0] : 1;
    return {
      schema: "deepbom.coreml.mil_compression_transform.v1", status: "assessed_exact_serialized_contract",
      evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: name,
      representation: "affine_constant_dequantization",
      granularity: scale.shape.length === 1 || zeroPoint.shape.length === 1 ? "per_axis" : "per_tensor",
      logical_output_elements: safeProduct(data.shape), quantized_data_elements: safeProduct(data.shape),
      scale_elements: scaleElements, zero_point_elements: zeroPointElements,
      serialized_axis: axisRaw, normalized_axis: normalizedAxis, axis_extent: axisExtent,
      quantized_data_dtype: data.dtype, zero_point_dtype: zeroPoint.dtype,
      scale_dtype: scale.dtype, output_dtype: scale.dtype, output_shape: [...data.shape],
      quantized_data_tensor_index: Number.isSafeInteger(data.index) ? data.index : null,
      zero_point_tensor_index: Number.isSafeInteger(zeroPoint.index) ? zeroPoint.index : null,
      scale_tensor_index: Number.isSafeInteger(scale.index) ? scale.index : null,
      axis_tensor_index: Number.isSafeInteger(axisTensor.index) ? axisTensor.index : null,
      source_file: COREML_MIL_SOURCE.constexpr_ios16_definition,
      source_sha256: COREML_MIL_SOURCE.constexpr_ios16_definition_sha256,
      boundary: "The serialized affine mapping, scalar/per-axis cardinality, normalized axis, and output type/shape are exact. Decompressed values, runtime materialization, device placement, timing, and task accuracy are not inferred.",
    };
  }
  if (name === "constexpr_blockwise_shift_scale") {
    if (opsetGeneration < 8) {
      throw new Error("Core ML MIL constexpr_blockwise_shift_scale is unavailable before the pinned CoreML8 opset");
    }
    const data = exactBinding(inputBindings, tensors, "data");
    const scale = exactBinding(inputBindings, tensors, "scale");
    const offset = exactBinding(inputBindings, tensors, "offset");
    if (!data || !scale || data.shape.length < 1 || data.shape.length !== scale.shape.length
      || !IOS18_BLOCKWISE_DATA_DTYPES.has(data.dtype) || !["FLOAT16", "FLOAT32"].includes(scale.dtype)
      || data.shape.some((value, index) => !Number.isSafeInteger(value) || value <= 0
        || !Number.isSafeInteger(scale.shape[index]) || scale.shape[index] <= 0 || value % scale.shape[index] !== 0)) {
      throw new Error("Core ML MIL constexpr_blockwise_shift_scale violates pinned rank, dtype, or block divisibility constraints");
    }
    if (offset && (!sameStaticShape(offset.shape, scale.shape)
      || !IOS18_BLOCKWISE_DATA_DTYPES.has(offset.dtype)
      || (!["FLOAT16", "FLOAT32"].includes(offset.dtype) && offset.dtype !== data.dtype))) {
      throw new Error("Core ML MIL constexpr_blockwise_shift_scale offset contradicts the pinned scale shape or dtype contract");
    }
    if (!sameStaticShape(outputShapes[0], data.shape) || tensors[outputIds[0]]?.dtype !== scale.dtype) {
      throw new Error("Core ML MIL constexpr_blockwise_shift_scale output ValueType contradicts pinned type inference");
    }
    return {
      schema: "deepbom.coreml.mil_compression_transform.v1", status: "assessed_exact_serialized_contract",
      evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: name, representation: "blockwise_affine",
      logical_output_elements: safeProduct(data.shape), scale_elements: safeProduct(scale.shape),
      offset_present: Boolean(offset), block_shape: data.shape.map((value, index) => value / scale.shape[index]),
      data_dtype: data.dtype, scale_dtype: scale.dtype, output_dtype: scale.dtype, output_shape: [...data.shape],
      data_tensor_index: Number.isSafeInteger(data.index) ? data.index : null,
      scale_tensor_index: Number.isSafeInteger(scale.index) ? scale.index : null,
      offset_tensor_index: Number.isSafeInteger(offset?.index) ? offset.index : null,
      source_file: COREML_MIL_SOURCE.compression_ios18_definition,
      source_sha256: COREML_MIL_SOURCE.compression_ios18_definition_sha256,
      boundary: "Serialized block cardinality and type/shape contract only. Decompressed values, physical runtime storage, device placement, and decompression timing are not inferred.",
    };
  }
  if (name === "constexpr_lut_to_dense") {
    if (opsetGeneration < 8) return exactLegacyLutContract(outputShapes, outputIds, inputBindings, tensors, attributes);
    const indices = exactBinding(inputBindings, tensors, "indices");
    const lut = exactBinding(inputBindings, tensors, "lut");
    const vectorAxisRaw = exactScalarBinding(inputBindings, tensors, "vector_axis");
    if (!indices || !lut) throw new Error("Core ML MIL constexpr_lut_to_dense is missing a serialized indices or LUT binding");
    if (indices.shape.length < 1 || lut.shape.length !== indices.shape.length + 2) {
      throw new Error(`Core ML MIL constexpr_lut_to_dense rank contract failed: indices rank ${indices.shape.length}, LUT rank ${lut.shape.length}`);
    }
    if (!IOS18_LUT_INDEX_DTYPES.has(indices.dtype)) {
      throw new Error(`Core ML MIL constexpr_lut_to_dense index dtype ${indices.dtype} is outside the pinned iOS 18 type domain`);
    }
    if (!["INT8", "UINT8", "FLOAT16", "FLOAT32"].includes(lut.dtype)) {
      throw new Error(`Core ML MIL constexpr_lut_to_dense LUT dtype ${lut.dtype} is outside the pinned iOS 18 type domain`);
    }
    for (let index = 0; index < indices.shape.length; index += 1) {
      const indexDimension = indices.shape[index];
      const lutDimension = lut.shape[index];
      if (!Number.isSafeInteger(indexDimension) || indexDimension <= 0
        || !Number.isSafeInteger(lutDimension) || lutDimension <= 0 || indexDimension % lutDimension !== 0) {
        throw new Error(`Core ML MIL constexpr_lut_to_dense block grid failed at dimension ${index}: indices ${indexDimension}, LUT ${lutDimension}`);
      }
    }
    const paletteCount = lut.shape.at(-2);
    const vectorSize = lut.shape.at(-1);
    const indexBits = DTYPE_BITS[indices.dtype];
    if (!Number.isSafeInteger(paletteCount) || paletteCount <= 0 || (paletteCount & (paletteCount - 1)) !== 0
      || 2 ** indexBits !== paletteCount || !Number.isSafeInteger(vectorSize) || vectorSize <= 0) {
      throw new Error("Core ML MIL constexpr_lut_to_dense palette cardinality contradicts the serialized index dtype");
    }
    let vectorAxis = null;
    if (vectorAxisRaw != null) {
      if (!Number.isSafeInteger(vectorAxisRaw) || vectorAxisRaw < -indices.shape.length || vectorAxisRaw >= indices.shape.length) {
        throw new Error("Core ML MIL constexpr_lut_to_dense vector_axis is outside the pinned rank range");
      }
      vectorAxis = vectorAxisRaw < 0 ? vectorAxisRaw + indices.shape.length : vectorAxisRaw;
    } else if (vectorSize > 1) throw new Error("Core ML MIL constexpr_lut_to_dense vector palettization is missing vector_axis");
    const expected = [...indices.shape];
    if (vectorSize > 1) expected[vectorAxis] *= vectorSize;
    if (!sameStaticShape(outputShapes[0], expected) || tensors[outputIds[0]]?.dtype !== lut.dtype) {
      throw new Error("Core ML MIL constexpr_lut_to_dense output ValueType contradicts pinned type inference");
    }
    return {
      schema: "deepbom.coreml.mil_compression_transform.v1", status: "assessed_exact_serialized_contract",
      evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: name, representation: "blockwise_lut_palettization",
      logical_index_elements: safeProduct(indices.shape), logical_output_elements: safeProduct(expected),
      index_bits: indexBits, palette_count: paletteCount, vector_size: vectorSize, vector_axis: vectorAxis,
      group_grid_shape: lut.shape.slice(0, -2), index_shape: [...indices.shape], lut_shape: [...lut.shape], output_shape: expected,
      indices_tensor_index: Number.isSafeInteger(indices.index) ? indices.index : null,
      lut_tensor_index: Number.isSafeInteger(lut.index) ? lut.index : null,
      source_file: COREML_MIL_SOURCE.compression_ios18_definition,
      source_sha256: COREML_MIL_SOURCE.compression_ios18_definition_sha256,
      boundary: "Serialized palette/index cardinality and type/shape contract only. LUT usage distribution, decompressed numerical quality, physical runtime storage, device placement, and timing are not inferred.",
    };
  }
  if (name === "constexpr_sparse_to_dense") {
    if (opsetGeneration < 8) return exactLegacySparseContract(outputShapes, outputIds, inputBindings, tensors, attributes);
    const data = exactBinding(inputBindings, tensors, "nonzero_data");
    const mask = exactBinding(inputBindings, tensors, "mask");
    if (!data || !mask || data.shape.length !== 1 || mask.shape.length < 1 || mask.dtype !== "UINT1"
      || !IOS18_BLOCKWISE_DATA_DTYPES.has(data.dtype) || !sameStaticShape(outputShapes[0], mask.shape)
      || tensors[outputIds[0]]?.dtype !== data.dtype) {
      throw new Error("Core ML MIL constexpr_sparse_to_dense violates pinned rank, mask, dtype, or output constraints");
    }
    const maskPopulation = mask.immediate_value?.kind === "bytes"
      && Number.isSafeInteger(mask.immediate_value.byte_popcount) ? mask.immediate_value.byte_popcount : null;
    if (maskPopulation != null && maskPopulation !== data.shape[0]) {
      throw new Error("Core ML MIL constexpr_sparse_to_dense mask population does not equal nonzero_data cardinality");
    }
    return {
      schema: "deepbom.coreml.mil_compression_transform.v1",
      status: maskPopulation == null ? "assessed_shape_contract_mask_population_unresolved" : "assessed_exact_serialized_contract",
      evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", transform: name, representation: "unstructured_sparse_bitmask",
      stored_nonzero_elements: data.shape[0], logical_output_elements: safeProduct(mask.shape), mask_shape: [...mask.shape], output_shape: [...mask.shape],
      nonzero_data_tensor_index: Number.isSafeInteger(data.index) ? data.index : null,
      mask_tensor_index: Number.isSafeInteger(mask.index) ? mask.index : null,
      mask_population: maskPopulation,
      mask_population_status: maskPopulation == null ? "not_decoded_at_mil_contract_layer" : "assessed_exact_immediate_payload",
      source_file: COREML_MIL_SOURCE.compression_ios18_definition,
      source_sha256: COREML_MIL_SOURCE.compression_ios18_definition_sha256,
      boundary: maskPopulation == null
        ? "Rank, dtype, mask, and dense output cardinality are exact. The source-required equality between mask population and nonzero-data length remains unassessed until the referenced mask payload is decoded and bound."
        : "Rank, dtype, dense output cardinality, little-endian packed mask population, and source-required equality with nonzero-data length are exact from the serialized immediate payload.",
    };
  }
  if (name.startsWith("constexpr_") && /(affine|lut|sparse|shift|palett)/.test(name)) return {
    schema: "deepbom.coreml.mil_compression_transform.v1", status: "not_assessed_source_semantics_not_implemented",
    evidence_class: "OBSERVED/NOT_ASSESSABLE", transform: name, representation: "serialized_compression_transform",
    source_file: name === "constexpr_affine_dequantize" ? COREML_MIL_SOURCE.constexpr_ios16_definition : null,
    source_sha256: name === "constexpr_affine_dequantize" ? COREML_MIL_SOURCE.constexpr_ios16_definition_sha256 : null,
    boundary: "The serialized transform is inventoried, but no numeric compression contract is emitted without an implemented source-pinned rule for this exact operation version.",
  };
  return null;
}

function sparseMaskPopulation(tensor) {
  const immediate = tensor?.immediate_value;
  if (immediate?.kind === "bytes" && Number.isSafeInteger(immediate.byte_popcount)) {
    return { value: immediate.byte_popcount, status: "assessed_exact_immediate_payload" };
  }
  const numerical = tensor?.numerical_integrity;
  if (numerical?.status?.startsWith("assessed") && Number.isSafeInteger(numerical.decoded_value_count)
    && Number.isSafeInteger(numerical.zero_count) && numerical.zero_count <= numerical.decoded_value_count) {
    return { value: numerical.decoded_value_count - numerical.zero_count, status: "assessed_exact_bound_blob_payload" };
  }
  return null;
}

function halfToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 31;
  const fraction = bits & 1023;
  if (!exponent) return fraction ? sign * fraction * 2 ** -24 : sign < 0 ? -0 : 0;
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function decodeMilImmediateBytes(tensor) {
  const immediate = tensor?.immediate_value;
  if (immediate?.kind !== "bytes" || immediate.values.length !== immediate.byte_length) return null;
  const bytes = Uint8Array.from(immediate.values);
  const count = tensor.shape?.length === 0 ? 1 : safeProduct(tensor.shape);
  if (!Number.isSafeInteger(count)) return null;
  const bits = DTYPE_BITS[tensor.dtype];
  if (Number.isSafeInteger(bits) && bits < 8) {
    const mask = 2 ** bits - 1;
    const values = [];
    for (const byte of bytes) for (let shift = 0; shift < 8 && values.length < count; shift += bits) {
      const code = byte >>> shift & mask;
      values.push(tensor.dtype.startsWith("INT") && code & 1 << (bits - 1) ? code - 2 ** bits : code);
    }
    return values.length === count ? values : null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const widths = { INT8: 1, UINT8: 1, INT16: 2, UINT16: 2, INT32: 4, UINT32: 4, FLOAT16: 2, BFLOAT16: 2, FLOAT32: 4, FLOAT64: 8 };
  const width = widths[tensor.dtype];
  if (!width || bytes.length !== count * width) return null;
  const values = [];
  for (let offset = 0; offset < bytes.length; offset += width) {
    if (tensor.dtype === "INT8") values.push(view.getInt8(offset));
    else if (tensor.dtype === "UINT8") values.push(view.getUint8(offset));
    else if (tensor.dtype === "INT16") values.push(view.getInt16(offset, true));
    else if (tensor.dtype === "UINT16") values.push(view.getUint16(offset, true));
    else if (tensor.dtype === "INT32") values.push(view.getInt32(offset, true));
    else if (tensor.dtype === "UINT32") values.push(view.getUint32(offset, true));
    else if (tensor.dtype === "FLOAT16") values.push(halfToNumber(view.getUint16(offset, true)));
    else if (tensor.dtype === "BFLOAT16") {
      const buffer = new ArrayBuffer(4);
      new DataView(buffer).setUint32(0, view.getUint16(offset, true) << 16, true);
      values.push(new DataView(buffer).getFloat32(0, true));
    } else if (tensor.dtype === "FLOAT32") values.push(view.getFloat32(offset, true));
    else values.push(view.getFloat64(offset, true));
  }
  return values;
}

function retainedTensorValues(tensor) {
  const numerical = tensor?.numerical_integrity;
  if (Array.isArray(numerical?.decoded_values) && numerical.decoded_values.length
    && String(numerical.decoded_values_status || "").startsWith("complete")) {
    const values = numerical.decoded_values.map(Number);
    return values.every(Number.isFinite) ? values : null;
  }
  const immediate = tensor?.immediate_value;
  if (immediate && !immediate.truncated && Array.isArray(immediate.values)
    && immediate.kind !== "bytes" && immediate.values.length === immediate.count) {
    const values = immediate.values.map(Number);
    return values.every(Number.isFinite) ? values : null;
  }
  const decoded = decodeMilImmediateBytes(tensor);
  return decoded?.every(Number.isFinite) ? decoded : null;
}

function tensorCodeHistogram(tensor, contract) {
  const numerical = tensor?.numerical_integrity;
  if (Array.isArray(numerical?.quant_code_histogram)) return [...numerical.quant_code_histogram];
  const immediate = tensor?.immediate_value;
  if (immediate?.kind !== "bytes" || immediate.values.length !== immediate.byte_length) return null;
  const bytes = immediate.values;
  const tensorBits = DTYPE_BITS[tensor?.dtype];
  const packedLogicalCount = Number.isSafeInteger(tensorBits) && tensorBits < 8 ? safeProduct(tensor.shape) : null;
  if ((packedLogicalCount != null || (contract.index_shape?.length === 1 && contract.serialized_index_bytes === bytes.length))
    && contract.index_bits < 8) {
    const histogram = new Array(contract.palette_count).fill(0);
    let remaining = packedLogicalCount ?? contract.logical_index_elements;
    const mask = 2 ** contract.index_bits - 1;
    for (const byte of bytes) for (let shift = 0; shift < 8 && remaining > 0; shift += contract.index_bits) {
      histogram[byte >>> shift & mask] += 1;
      remaining -= 1;
    }
    return remaining === 0 ? histogram : null;
  }
  const capacity = contract.palette_count || 2 ** (DTYPE_BITS[tensor.dtype] || 8);
  const histogram = new Array(capacity).fill(0);
  for (const value of bytes) {
    if (value >= histogram.length) return null;
    histogram[value] += 1;
  }
  return histogram;
}

function numericalDigest(tensor) {
  const row = tensor?.numerical_integrity;
  const retained = retainedTensorValues(tensor);
  if ((!row || String(row.status || "").startsWith("not_assessed")) && retained) {
    const stats = reconstructedStats(retained);
    return {
      status: "assessed_exact_retained_serialized_payload",
      decoded_value_count: stats.value_count,
      nonfinite_count: 0,
      zero_count: stats.zero_count,
      negative_count: retained.filter((value) => value < 0).length,
      finite_min: stats.finite_min,
      finite_max: stats.finite_max,
      payload_sha256: row?.payload_sha256 || null,
    };
  }
  if (!row) return { status: "not_assessed_payload_not_bound" };
  return {
    status: row.status,
    decoded_value_count: row.decoded_value_count ?? row.value_count ?? null,
    nonfinite_count: row.nonfinite_count ?? ((row.nan_value_count || 0) + (row.positive_infinity_value_count || 0) + (row.negative_infinity_value_count || 0)),
    zero_count: row.zero_count ?? row.zero_value_count ?? null,
    negative_count: row.negative_count ?? row.negative_value_count ?? null,
    finite_min: row.finite_min ?? row.minimum_finite ?? null,
    finite_max: row.finite_max ?? row.maximum_finite ?? null,
    payload_sha256: row.payload_sha256 || null,
  };
}

function reconstructedStats(values) {
  if (!values?.length || values.some((value) => !Number.isFinite(value))) return null;
  let minimum = values[0];
  let maximum = values[0];
  let zeroCount = 0;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (value === 0) zeroCount += 1;
  }
  return { value_count: values.length, finite_min: minimum, finite_max: maximum, zero_count: zeroCount };
}

function lutReconstructedStats(histogram, values) {
  let valueCount = 0;
  let zeroCount = 0;
  let minimum = null;
  let maximum = null;
  for (let index = 0; index < histogram.length; index += 1) {
    const count = histogram[index];
    if (!count) continue;
    const value = values[index];
    if (!Number.isFinite(value)) return null;
    valueCount += count;
    if (value === 0) zeroCount += count;
    minimum = minimum == null ? value : Math.min(minimum, value);
    maximum = maximum == null ? value : Math.max(maximum, value);
  }
  return valueCount ? { value_count: valueCount, finite_min: minimum, finite_max: maximum, zero_count: zeroCount } : null;
}

function sparseReconstructedStats(values, logicalCount) {
  if (!values || !Number.isSafeInteger(logicalCount) || logicalCount < values.length || values.some((value) => !Number.isFinite(value))) return null;
  let minimum = 0;
  let maximum = 0;
  let zeroCount = logicalCount - values.length;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (value === 0) zeroCount += 1;
  }
  return { value_count: logicalCount, finite_min: minimum, finite_max: maximum, zero_count: zeroCount };
}

function blockwiseReconstruction(contract, tensors) {
  const data = tensors?.[contract.data_tensor_index];
  const scale = tensors?.[contract.scale_tensor_index];
  const offset = Number.isSafeInteger(contract.offset_tensor_index) ? tensors?.[contract.offset_tensor_index] : null;
  const dataValues = retainedTensorValues(data);
  const scaleValues = retainedTensorValues(scale);
  const offsetValues = offset ? retainedTensorValues(offset) : null;
  if (!dataValues || !scaleValues || (offset && !offsetValues)) return null;
  const dataShape = data.shape;
  const scaleShape = scale.shape;
  if (dataValues.length !== safeProduct(dataShape) || scaleValues.length !== safeProduct(scaleShape)
    || (offsetValues && offsetValues.length !== scaleValues.length)) return null;
  const scaleStrides = scaleShape.map((_, index) => safeProduct(scaleShape.slice(index + 1)) || 1);
  const dataStrides = dataShape.map((_, index) => safeProduct(dataShape.slice(index + 1)) || 1);
  const output = new Array(dataValues.length);
  for (let linear = 0; linear < dataValues.length; linear += 1) {
    let scaleLinear = 0;
    for (let axis = 0; axis < dataShape.length; axis += 1) {
      const coordinate = Math.floor(linear / dataStrides[axis]) % dataShape[axis];
      scaleLinear += Math.floor(coordinate / contract.block_shape[axis]) * scaleStrides[axis];
    }
    output[linear] = dataValues[linear] * scaleValues[scaleLinear] + (offsetValues?.[scaleLinear] || 0);
  }
  return reconstructedStats(output);
}

function affineReconstruction(contract, tensors) {
  const dataValues = retainedTensorValues(tensors?.[contract.quantized_data_tensor_index]);
  const zeroPointValues = retainedTensorValues(tensors?.[contract.zero_point_tensor_index]);
  const scaleValues = retainedTensorValues(tensors?.[contract.scale_tensor_index]);
  if (!dataValues || !zeroPointValues || !scaleValues
    || dataValues.length !== contract.quantized_data_elements
    || ![1, contract.axis_extent].includes(zeroPointValues.length)
    || ![1, contract.axis_extent].includes(scaleValues.length)) return null;
  const trailing = safeProduct(contract.output_shape.slice(contract.normalized_axis + 1)) || 1;
  const output = dataValues.map((value, index) => {
    const coordinate = Math.floor(index / trailing) % contract.axis_extent;
    const scale = scaleValues.length === 1 ? scaleValues[0] : scaleValues[coordinate];
    const zeroPoint = zeroPointValues.length === 1 ? zeroPointValues[0] : zeroPointValues[coordinate];
    return scale * (value - zeroPoint);
  });
  return reconstructedStats(output);
}

function annotateCompressionPayloadEvidence(contract, tensors) {
  if (contract.transform === "constexpr_affine_dequantize") {
    contract.payload_integrity = {
      quantized_data: numericalDigest(tensors?.[contract.quantized_data_tensor_index]),
      zero_point: numericalDigest(tensors?.[contract.zero_point_tensor_index]),
      scale: numericalDigest(tensors?.[contract.scale_tensor_index]),
      axis: numericalDigest(tensors?.[contract.axis_tensor_index]),
    };
    const reconstruction = affineReconstruction(contract, tensors);
    contract.reconstruction = reconstruction
      ? { status: "assessed_exact_retained_payload", evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", ...reconstruction }
      : { status: "not_materialized_large_payload", evidence_class: "NOT_ASSESSED", reason: "The affine contract is exact, but dense value extrema require retained quantized data, scale, and zero-point values." };
  } else if (contract.transform === "constexpr_blockwise_shift_scale") {
    contract.payload_integrity = {
      data: numericalDigest(tensors?.[contract.data_tensor_index]),
      scale: numericalDigest(tensors?.[contract.scale_tensor_index]),
      offset: Number.isSafeInteger(contract.offset_tensor_index) ? numericalDigest(tensors?.[contract.offset_tensor_index]) : { status: "not_applicable_absent" },
    };
    const reconstruction = blockwiseReconstruction(contract, tensors);
    contract.reconstruction = reconstruction
      ? { status: "assessed_exact_retained_payload", evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", ...reconstruction }
      : { status: "not_materialized_large_payload", evidence_class: "NOT_ASSESSED", reason: "All source tensors are fully scanned, but element-correlated dense reconstruction is retained only when every operand has at most 256 decoded values." };
    contract.boundary = "Block cardinality, operand payload integrity, and source-defined affine mapping are assessed. Exact reconstructed extrema are emitted only for bounded retained operands; runtime buffer residency, placement, and timing are not inferred.";
  } else if (contract.transform === "constexpr_lut_to_dense") {
    const indices = tensors?.[contract.indices_tensor_index];
    const lut = tensors?.[contract.lut_tensor_index];
    const histogram = tensorCodeHistogram(indices, contract);
    const used = histogram?.filter((count) => count > 0).length ?? null;
    contract.payload_integrity = { indices: numericalDigest(indices), lut: numericalDigest(lut) };
    contract.lut_usage = histogram ? {
      status: "assessed_exact_full_index_payload", palette_entries_used: used,
      palette_entries_total: contract.palette_count, utilization_ratio: used / contract.palette_count,
      index_count: histogram.reduce((sum, count) => sum + count, 0), code_histogram: histogram,
    } : { status: "not_assessed_index_payload_not_bound_or_not_decoded" };
    const lutValues = retainedTensorValues(lut);
    const singleGrid = Array.isArray(contract.group_grid_shape) ? contract.group_grid_shape.every((value) => value === 1) : true;
    const reconstruction = histogram && lutValues && singleGrid && contract.vector_size === 1 && lutValues.length === contract.palette_count
      ? lutReconstructedStats(histogram, lutValues) : null;
    if (reconstruction) {
      contract.reconstruction = { status: "assessed_exact_retained_payload", evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", ...reconstruction };
    } else contract.reconstruction = {
      status: "not_materialized_grouped_or_large_payload", evidence_class: "NOT_ASSESSED",
      reason: "Exact LUT usage is retained when index codes are decoded; dense value extrema additionally require one retained scalar palette shared by all groups.",
    };
    contract.boundary = "Palette/index cardinality and full-payload code utilization are assessed when index payloads are bound. Exact reconstructed extrema require a bounded retained scalar palette; runtime storage, placement, and timing are not inferred.";
  } else if (contract.transform === "constexpr_sparse_to_dense") {
    const data = tensors?.[contract.nonzero_data_tensor_index];
    const mask = tensors?.[contract.mask_tensor_index];
    contract.payload_integrity = { nonzero_data: numericalDigest(data), mask: numericalDigest(mask) };
    const values = retainedTensorValues(data);
    const reconstructed = sparseReconstructedStats(values, contract.logical_output_elements);
    contract.reconstruction = reconstructed
      ? { status: "assessed_exact_retained_payload", evidence_class: "OBSERVED/SOURCE_PINNED/DERIVED", ...reconstructed }
      : { status: "not_materialized_large_payload", evidence_class: "NOT_ASSESSED", reason: "Mask population is exact, but dense numerical extrema require retained nonzero values." };
    contract.boundary = "Sparse mask population, nonzero cardinality, padding, and payload integrity are assessed. Exact reconstructed extrema are emitted only for bounded retained nonzero values; runtime storage, placement, and timing are not inferred.";
  }
}

export function refreshCoreMlMilCompressionEvidence(analysis) {
  const ops = analysis?.ops || [];
  for (const op of ops) {
    const contract = op?.compression_contract;
    if (contract) annotateCompressionPayloadEvidence(contract, analysis.tensors || []);
    if (contract?.transform !== "constexpr_sparse_to_dense" || !Number.isSafeInteger(contract.mask_tensor_index)) continue;
    const population = sparseMaskPopulation(analysis.tensors?.[contract.mask_tensor_index]);
    if (!population) continue;
    if (population.value !== contract.stored_nonzero_elements) {
      throw new Error("Core ML MIL constexpr_sparse_to_dense mask population does not equal nonzero_data cardinality");
    }
    contract.status = "assessed_exact_serialized_contract";
    contract.mask_population = population.value;
    contract.mask_population_status = population.status;
    contract.boundary = "Rank, dtype, dense output cardinality, packed mask population, and source-required equality with nonzero-data length are exact from the bound serialized payload.";
  }
  const ledger = analysis?.coreml?.mil_compression_contract;
  if (!ledger) return analysis;
  ledger.transforms = ops.filter((op) => op.compression_contract).map((op) => ({
    op_index: op.index, op_type: op.mil_operation_type, ...op.compression_contract,
  }));
  ledger.transform_count = ledger.transforms.length;
  ledger.exact_contract_count = ledger.transforms.filter((row) => row.status.startsWith("assessed_exact")).length;
  ledger.partial_contract_count = ledger.transform_count - ledger.exact_contract_count;
  ledger.status = !ledger.transform_count ? "not_applicable_no_serialized_compression_transform"
    : ledger.exact_contract_count === ledger.transform_count ? "assessed_exact_serialized_contracts"
      : ledger.transforms.some((row) => row.status.startsWith("not_assessed")) ? "partial" : "assessed_with_explicit_payload_boundary";
  return analysis;
}

function broadcastStatic(left, right) {
  const rank = Math.max(left.length, right.length);
  const result = [];
  for (let offset = rank; offset > 0; offset -= 1) {
    const a = left[left.length - offset] ?? 1;
    const b = right[right.length - offset] ?? 1;
    if (a !== b && a !== 1 && b !== 1) return null;
    result.push(Math.max(a, b));
  }
  return result;
}

function exactConvTransposePairs(input, kernel, stride, dilation, padStart, output) {
  const result = convTransposeAxisPairs(
    BigInt(input), BigInt(kernel), BigInt(stride), BigInt(dilation), BigInt(padStart), BigInt(output),
  );
  return result == null ? null : result.toString();
}

function normalizedEinsumEquation(equation) {
  const compact = String(equation || "").replace(/\s+/g, "");
  const match = /^([^,]+),([^>]+)->(.+)$/.exec(compact);
  if (!match || match.slice(1).some((part) => part.includes("..."))) return null;
  const labels = new Map();
  const encode = (part) => [...part].map((label) => {
    if (!labels.has(label)) labels.set(label, labels.size);
    return labels.get(label);
  });
  return [encode(match[1]), encode(match[2]), encode(match[3])];
}

function sameIntegerArray(left, right) {
  return Array.isArray(left) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function macsForMil(type, inputShapes, outputShapes, inputBindings, tensors) {
  const name = String(type || "").toLowerCase();
  const outputProduct = multiplyCoreMlExactIntegers(outputShapes[0] || []);
  const argumentShape = (argument) => {
    const id = inputBindings?.[argument]?.find((item) => Number.isSafeInteger(item.tensor_index))?.tensor_index;
    return Number.isSafeInteger(id) ? tensors[id]?.shape : null;
  };
  const argumentImmediate = (argument, fallback) => {
    const bindings = inputBindings?.[argument];
    if (!bindings?.length) return fallback;
    if (bindings.length !== 1) return null;
    const binding = bindings[0];
    const immediate = binding.kind === "value" ? binding.value?.immediate : tensors[binding.tensor_index]?.immediate_value;
    return immediate && !immediate.truncated && immediate.count === immediate.values.length ? [...immediate.values] : null;
  };
  const scalar = (argument, fallback) => {
    const values = argumentImmediate(argument, [fallback]);
    return values?.length === 1 ? values[0] : null;
  };
  if (name === "conv" && outputProduct) {
    const x = argumentShape("x") || inputShapes[0];
    const weight = argumentShape("weight");
    if (!x?.length || !weight?.length || x.length !== weight.length || x.length < 3 || x.length > 5) return { macs: null, status: "not_assessed_invalid_or_dynamic_mil_conv_rank", reason: "conv x/weight rank is not a matching static 1D-3D convolution contract" };
    const spatialRank = x.length - 2;
    const groups = scalar("groups", 1);
    const strides = argumentImmediate("strides", new Array(spatialRank).fill(1));
    const dilations = argumentImmediate("dilations", new Array(spatialRank).fill(1));
    const padType = scalar("pad_type", "valid");
    const pad = argumentImmediate("pad", new Array(spatialRank * 2).fill(0));
    if (!Number.isSafeInteger(groups) || groups <= 0 || !strides || !dilations || !pad || ![strides, dilations].every((row) => row.length === spatialRank && row.every((value) => Number.isSafeInteger(value) && value > 0))
      || pad.length !== spatialRank * 2 || pad.some((value) => !Number.isSafeInteger(value) || value < 0) || !["valid", "custom", "same", "same_lower"].includes(padType)) {
      return { macs: null, status: "not_assessed_dynamic_or_invalid_mil_conv_parameters", reason: "conv groups/stride/dilation/padding constants are unavailable or invalid" };
    }
    if (x[1] % groups !== 0 || x[1] / groups !== weight[1]) throw new Error("Core ML MIL conv input channels/groups contradict the weight shape");
    const expected = [x[0], weight[0]];
    for (let index = 0; index < spatialRank; index += 1) {
      const effective = (weight[index + 2] - 1) * dilations[index] + 1;
      const custom = padType === "custom" ? pad[index * 2] + pad[index * 2 + 1] : 0;
      expected.push(padType === "same" || padType === "same_lower"
        ? Math.ceil(x[index + 2] / strides[index])
        : Math.floor((x[index + 2] + custom - effective) / strides[index]) + 1);
    }
    if (expected.some((value) => !Number.isSafeInteger(value) || value <= 0) || !sameStaticShape(expected, outputShapes[0])) throw new Error("Core ML MIL conv output ValueType contradicts source-defined shape inference");
    const product = multiplyCoreMlExactIntegers([outputProduct.decimal, ...weight.slice(1)]);
    if (product) return exactMacResult(product, "derived_exact_mil_conv");
  }
  if (name === "conv_transpose" && outputProduct) {
    const x = argumentShape("x") || inputShapes[0];
    const weight = argumentShape("weight");
    if (!x?.length || !weight?.length || x.length !== weight.length || x.length < 3 || x.length > 5) return { macs: null, status: "not_assessed_invalid_or_dynamic_mil_conv_transpose_rank", reason: "conv_transpose x/weight rank is not a matching static 1D-3D convolution contract" };
    const spatialRank = x.length - 2;
    const groups = scalar("groups", 1);
    const strides = argumentImmediate("strides", new Array(spatialRank).fill(1));
    const dilations = argumentImmediate("dilations", new Array(spatialRank).fill(1));
    const padType = scalar("pad_type", "valid");
    const pad = argumentImmediate("pad", new Array(spatialRank * 2).fill(0));
    if (!Number.isSafeInteger(groups) || groups <= 0 || !strides || !dilations || !pad
      || ![strides, dilations].every((row) => row.length === spatialRank && row.every((value) => Number.isSafeInteger(value) && value > 0))
      || pad.length !== spatialRank * 2 || pad.some((value) => !Number.isSafeInteger(value) || value < 0)
      || !["valid", "custom", "same"].includes(padType)) {
      return { macs: null, status: "not_assessed_dynamic_or_invalid_mil_conv_transpose_parameters", reason: "conv_transpose groups/stride/dilation/padding constants are unavailable or invalid" };
    }
    if (x[1] !== weight[0] || x[1] % groups !== 0) throw new Error("Core ML MIL conv_transpose input channels/groups contradict the weight shape");
    const outputChannels = weight[1] * groups;
    if (outputShapes[0]?.[0] !== x[0] || outputShapes[0]?.[1] !== outputChannels) throw new Error("Core ML MIL conv_transpose output batch/channels contradict source-defined type inference");
    const pairs = [];
    for (let index = 0; index < spatialRank; index += 1) {
      const inputSize = x[index + 2];
      const kernel = weight[index + 2];
      const outputSize = outputShapes[0][index + 2];
      const effective = (kernel - 1) * dilations[index] + 1;
      if (![inputSize, kernel, outputSize, effective].every((value) => Number.isSafeInteger(value) && value > 0)) return { macs: null, status: "not_assessed_incomplete_static_shape", reason: "conv_transpose spatial cardinality is not exact" };
      const totalPad = (inputSize - 1) * strides[index] + effective - outputSize;
      if (totalPad < 0) throw new Error("Core ML MIL conv_transpose output spatial extent exceeds the source-defined contract");
      let padStart = 0;
      if (padType === "custom") {
        if (totalPad !== pad[index * 2] + pad[index * 2 + 1]) throw new Error("Core ML MIL conv_transpose custom padding contradicts the output ValueType");
        padStart = pad[index * 2];
      } else if (padType === "valid") {
        if (totalPad !== 0) return { macs: null, status: "not_assessed_output_shape_override_changes_conv_transpose_placement", reason: "output_shape overrides valid-padding placement without a serialized crop origin" };
      } else {
        const lower = Math.floor(totalPad / 2);
        const upper = totalPad - lower;
        const lowerPairs = exactConvTransposePairs(inputSize, kernel, strides[index], dilations[index], lower, outputSize);
        const upperPairs = exactConvTransposePairs(inputSize, kernel, strides[index], dilations[index], upper, outputSize);
        if (lowerPairs == null || upperPairs == null || lowerPairs !== upperPairs) return { macs: null, status: "not_assessed_same_padding_origin_not_serialized", reason: "same padding does not uniquely serialize the leading crop and candidate pair counts differ" };
        pairs.push(lowerPairs);
        continue;
      }
      const count = exactConvTransposePairs(inputSize, kernel, strides[index], dilations[index], padStart, outputSize);
      if (count == null) return { macs: null, status: "not_assessed_conv_transpose_pair_count_exceeds_bound", reason: "conv_transpose overlap enumeration exceeds the bounded exact path" };
      pairs.push(count);
    }
    return exactMacResult(multiplyCoreMlExactIntegers([x[0], x[1], weight[1], ...pairs]), "derived_exact_mil_conv_transpose_overlap");
  }
  if (name === "linear" && outputProduct) {
    const x = argumentShape("x") || inputShapes[0];
    const weight = argumentShape("weight");
    if (!x?.length || weight?.length !== 2 || x.at(-1) !== weight[1]) throw new Error("Core ML MIL linear input width contradicts the weight shape");
    const expected = [...x.slice(0, -1), weight[0]];
    if (!sameStaticShape(expected, outputShapes[0])) throw new Error("Core ML MIL linear output ValueType contradicts source-defined shape inference");
    return exactMacResult(multiplyCoreMlExactIntegers([outputProduct.decimal, weight[1]]), "derived_exact_mil_linear");
  }
  if (name === "matmul" && outputProduct) {
    const originalX = argumentShape("x") || inputShapes[0];
    const originalY = argumentShape("y") || inputShapes[1];
    const transposeX = scalar("transpose_x", false);
    const transposeY = scalar("transpose_y", false);
    if (!originalX?.length || !originalY?.length || typeof transposeX !== "boolean" || typeof transposeY !== "boolean") return { macs: null, status: "not_assessed_dynamic_mil_matmul_contract", reason: "matmul shape or transpose constant is unavailable" };
    if ((originalX.length === 1 && transposeX) || (originalY.length === 1 && transposeY)) throw new Error("Core ML MIL matmul transposes a rank-1 operand");
    const x = [...originalX];
    const y = [...originalY];
    if (transposeX) [x[x.length - 2], x[x.length - 1]] = [x[x.length - 1], x[x.length - 2]];
    if (transposeY) [y[y.length - 2], y[y.length - 1]] = [y[y.length - 1], y[y.length - 2]];
    const xVector = x.length === 1;
    const yVector = y.length === 1;
    if (xVector) x.unshift(1);
    if (yVector) y.push(1);
    if (x.at(-1) !== y.at(-2)) throw new Error("Core ML MIL matmul contracted dimensions disagree");
    const batch = broadcastStatic(x.slice(0, -2), y.slice(0, -2));
    if (!batch) throw new Error("Core ML MIL matmul batch dimensions are not broadcast-compatible");
    const expected = [...batch, x.at(-2), y.at(-1)];
    if (xVector) expected.splice(expected.length - 2, 1);
    if (yVector) expected.pop();
    if (!sameStaticShape(expected, outputShapes[0])) throw new Error("Core ML MIL matmul output ValueType contradicts source-defined shape inference");
    return exactMacResult(multiplyCoreMlExactIntegers([outputProduct.decimal, x.at(-1)]), "derived_exact_mil_matmul");
  }
  if (name === "einsum" && outputProduct) {
    const values = (inputBindings?.values || []).map((binding) => Number.isSafeInteger(binding.tensor_index) ? tensors[binding.tensor_index] : null);
    const equation = scalar("equation", null);
    const pattern = normalizedEinsumEquation(equation);
    if (values.length !== 2 || values.some((tensor) => !tensor?.shape?.length) || !pattern) return { macs: null, status: "not_assessed_incomplete_mil_einsum_contract", reason: "einsum requires two static input tensors and a constant supported equation" };
    const x = values[0].shape;
    const y = values[1].shape;
    const rank4 = sameIntegerArray(pattern[0], [0, 1, 2, 3]) && sameIntegerArray(pattern[1], [0, 3, 2, 4]) && sameIntegerArray(pattern[2], [0, 1, 2, 4]);
    const rank3 = sameIntegerArray(pattern[0], [0, 1, 2]) && sameIntegerArray(pattern[1], [2, 1, 3]) && sameIntegerArray(pattern[2], [0, 1, 3]);
    if ((!rank4 && !rank3) || x.length !== y.length || x.length !== (rank4 ? 4 : 3)) return { macs: null, status: "not_assessed_unsupported_mil_einsum_equation", reason: "equation is outside the two source-supported Core ML einsum contraction patterns" };
    if (x.at(-1) !== y.at(-3) || (x.at(-2) !== 1 && y.at(-2) !== 1 && x.at(-2) !== y.at(-2))
      || (rank4 && x[0] !== 1 && y[0] !== 1 && x[0] !== y[0])) throw new Error("Core ML MIL einsum input dimensions contradict source-defined broadcasting");
    const expected = rank4
      ? [Math.max(x[0], y[0]), x[1], Math.max(x[2], y[2]), y[3]]
      : [x[0], Math.max(x[1], y[1]), y[2]];
    if (!sameStaticShape(expected, outputShapes[0])) throw new Error("Core ML MIL einsum output ValueType contradicts source-defined type inference");
    return exactMacResult(multiplyCoreMlExactIntegers([outputProduct.decimal, x.at(-1)]), "derived_exact_mil_einsum");
  }
  if (["rnn", "gru", "lstm"].includes(name)) {
    const x = argumentShape("x") || inputShapes[0];
    const weightIh = argumentShape("weight_ih");
    const weightHh = argumentShape("weight_hh");
    const direction = scalar("direction", "forward");
    if (x?.length !== 3 || weightIh?.length !== 2 || weightHh?.length !== 2 || !["forward", "reverse", "bidirectional"].includes(direction)) return { macs: null, status: "not_assessed_incomplete_mil_recurrent_contract", reason: "recurrent input, weights, or direction are unavailable" };
    const [sequence, batch, inputSize] = x;
    const gates = name === "lstm" ? 4 : name === "gru" ? 3 : 1;
    const hidden = weightHh[1];
    const directions = direction === "bidirectional" ? 2 : 1;
    if ((name !== "lstm" && directions !== 1) || weightIh[0] !== gates * hidden || weightIh[1] !== inputSize
      || weightHh[0] !== gates * hidden || weightHh[1] !== hidden) throw new Error(`Core ML MIL ${name} weights contradict the source-defined gate dimensions`);
    if (directions === 2) {
      const backIh = argumentShape("weight_ih_back");
      const backHh = argumentShape("weight_hh_back");
      if (!sameStaticShape(backIh, weightIh) || !sameStaticShape(backHh, weightHh)) throw new Error("Core ML MIL bidirectional lstm lacks matching backward weights");
    }
    const outputSequence = scalar("output_sequence", false);
    if (typeof outputSequence !== "boolean") return { macs: null, status: "not_assessed_incomplete_mil_recurrent_contract", reason: "output_sequence is not a constant boolean" };
    const expectedY = [outputSequence ? sequence : 1, batch, directions * hidden];
    if (!sameStaticShape(expectedY, outputShapes[0]) || !sameStaticShape([batch, directions * hidden], outputShapes[1])) throw new Error(`Core ML MIL ${name} output ValueType contradicts source-defined type inference`);
    if (name === "lstm" && !sameStaticShape([batch, directions * hidden], outputShapes[2])) throw new Error("Core ML MIL lstm cell-state output contradicts source-defined type inference");
    return exactMacResult(multiplyCoreMlExactIntegers([sequence, batch, directions, gates, hidden, inputSize + hidden]), `derived_exact_mil_${name}`);
  }
  if (name === "scaled_dot_product_attention" && outputProduct) {
    const query = argumentShape("query");
    const key = argumentShape("key");
    const value = argumentShape("value");
    if (!query || !key || !value || query.length < 3 || query.length !== key.length || query.length !== value.length) return { macs: null, status: "not_assessed_incomplete_mil_attention_contract", reason: "attention query/key/value ranks or shapes are unavailable" };
    if (!sameStaticShape(query.slice(0, -2), key.slice(0, -2)) || !sameStaticShape(query.slice(0, -2), value.slice(0, -2))
      || query.at(-1) !== key.at(-1) || key.at(-2) !== value.at(-2)) throw new Error("Core ML MIL scaled_dot_product_attention Q/K/V dimensions contradict source validation");
    const expected = [...query.slice(0, -1), value.at(-1)];
    if (!sameStaticShape(expected, outputShapes[0])) throw new Error("Core ML MIL scaled_dot_product_attention output ValueType contradicts source-defined type inference");
    const batch = safeProduct(query.slice(0, -2));
    return exactMacResult(multiplyCoreMlExactIntegers([batch, query.at(-2), key.at(-2), query.at(-1) + value.at(-1)]), "derived_exact_mil_scaled_dot_product_attention");
  }
  if (["conv", "conv_transpose", "linear", "matmul", "einsum", "rnn", "gru", "lstm", "scaled_dot_product_attention"].includes(name)) return { macs: null, status: "not_assessed_incomplete_static_shape", reason: "output or contracted input cardinality exceeds the exact static-analysis range" };
  if (MIL_KNOWN_NON_MAC.has(name) || name.startsWith("constexpr_")) return { macs: 0, macs_decimal: "0", status: "derived_non_mac_operation", reason: "source operation is not a multiply-accumulate kernel" };
  return { macs: null, status: "not_assessed_operation_cost_rule_not_implemented", reason: `MIL operation ${type} has no implemented MAC classification` };
}

function exactMacResult(product, status) {
  if (!product) return { macs: null, macs_decimal: null, status: "not_assessed_invalid_exact_cardinality", reason: "MAC factors are not exact nonnegative integers" };
  return {
    macs: product.number,
    macs_decimal: product.decimal,
    status: product.number == null ? status.replace("derived_exact_", "derived_exact_decimal_only_") : status,
    reason: null,
  };
}

export function graphFromCoreMlMilProgram(program, preferredFunction = null) {
  const functionNames = Object.keys(program.functions);
  const functionName = preferredFunction && Object.hasOwn(program.functions, preferredFunction)
    ? preferredFunction : Object.hasOwn(program.functions, "main") ? "main" : functionNames[0];
  const fn = program.functions[functionName];
  const tensors = [];
  const ops = [];
  const blobReferences = [];
  const scopeContracts = new Map();
  const rootScope = new Map();
  const addTensor = (name, type, extra = {}) => {
    const index = tensors.length;
    tensors.push({ index, name, dtype: type?.dtype || "UNKNOWN", rank: Number.isSafeInteger(type?.rank) ? type.rank : null, shape: type?.shape || [], constant_buffer: false, quant_scales: 0, quant_zero_points: 0, ...extra });
    return index;
  };
  const inputTensorIndices = fn.inputs.map((item) => {
    if (rootScope.has(item.name)) throw new Error(`Core ML MIL function repeats input ${item.name}`);
    const id = addTensor(item.name, item.type, { role: "function_input", shape_source: "mil_value_type" });
    rootScope.set(item.name, id);
    return id;
  });
  const resolve = (scopes, name) => {
    for (let index = scopes.length - 1; index >= 0; index -= 1) if (scopes[index].has(name)) return scopes[index].get(name);
    throw new Error(`Core ML MIL SSA reference ${name} is not defined before use`);
  };
  const walkBlock = (block, scopes, path) => {
    const local = new Map();
    const chain = [...scopes, local];
    const blockInputTensorIndices = [];
    for (const item of block.inputs) {
      if (local.has(item.name)) throw new Error(`Core ML MIL block repeats input ${item.name}`);
      const id = addTensor(`${path}/${item.name}`, item.type, { role: "block_input", mil_ssa_name: item.name, shape_source: "mil_value_type" });
      local.set(item.name, id);
      blockInputTensorIndices.push(id);
    }
    for (let operationIndex = 0; operationIndex < block.operations.length; operationIndex += 1) {
      const operation = block.operations[operationIndex];
      const opIndex = ops.length;
      const inputIds = [];
      const inputBindings = {};
      for (const [argumentName, bindings] of Object.entries(operation.inputs)) {
        inputBindings[argumentName] = [];
        for (let bindingIndex = 0; bindingIndex < bindings.length; bindingIndex += 1) {
          const binding = bindings[bindingIndex];
          if (binding.kind === "name") {
            const id = resolve(chain, binding.name);
            inputIds.push(id);
            inputBindings[argumentName].push({ kind: "name", tensor_index: id, name: binding.name });
          } else {
            if (String(operation.type).toLowerCase() === "const") {
              inputBindings[argumentName].push({ kind: "value", tensor_index: null, value: binding.value });
            } else {
              const id = addTensor(`${path}/op${opIndex}/${argumentName}:${bindingIndex}`, binding.value.type, {
                role: "inline_constant", constant_buffer: true, value_storage: binding.value.storage, immediate_value: binding.value.immediate, blob_reference: binding.value.blob, shape_source: "mil_value_type",
              });
              inputIds.push(id);
              inputBindings[argumentName].push({ kind: "value", tensor_index: id, value: binding.value });
              if (binding.value.blob) blobReferences.push({ ...binding.value.blob, tensor_index: id, op_index: opIndex, argument_name: argumentName });
            }
          }
        }
      }
      const outputIds = [];
      for (const output of operation.outputs) {
        if (local.has(output.name)) throw new Error(`Core ML MIL scope repeats SSA value ${output.name}`);
        const constValue = String(operation.type).toLowerCase() === "const"
          ? Object.values(inputBindings).flat().find((item) => item.kind === "value")?.value || null : null;
        const id = addTensor(`${path}/${output.name}`, output.type, {
          role: constValue ? "constant" : "activation", mil_ssa_name: output.name, constant_buffer: Boolean(constValue),
          value_storage: constValue?.storage || null, immediate_value: constValue?.immediate || null, blob_reference: constValue?.blob || null, shape_source: "mil_value_type",
        });
        if (constValue?.blob) blobReferences.push({ ...constValue.blob, tensor_index: id, op_index: opIndex, argument_name: "const_output" });
        local.set(output.name, id);
        outputIds.push(id);
      }
      const inputShapes = inputIds.map((id) => tensors[id].shape);
      const outputShapes = outputIds.map((id) => tensors[id].shape);
      const arithmetic = macsForMil(operation.type, inputShapes, outputShapes, inputBindings, tensors);
      const compression = exactCompressionContract(operation.type, outputShapes, outputIds, inputBindings, tensors, fn.opset, operation.attributes);
      const outputBytes = outputIds.reduce((sum, id) => {
        if (sum == null) return null;
        const tensor = tensors[id];
        const count = safeProduct(tensor.shape);
        const width = DTYPE_BYTES[tensor.dtype];
        return count != null && width && count <= Math.floor((Number.MAX_SAFE_INTEGER - sum) / width) ? sum + count * width : null;
      }, 0);
      const opRow = {
        index: opIndex, name: String(operation.type).toUpperCase(), mil_operation_type: operation.type, coreml_layer_name: operation.outputs.map((item) => item.name).join(", "),
        inputs: inputIds, outputs: outputIds, input_shapes: inputShapes, output_shapes: outputShapes, mil_input_bindings: inputBindings,
        macs: arithmetic.macs, macs_decimal: arithmetic.macs_decimal ?? (arithmetic.macs == null ? null : String(arithmetic.macs)), macs_status: arithmetic.status, macs_reason: arithmetic.reason,
        estimated_bytes: outputBytes, estimated_bytes_reason: outputBytes == null ? "MIL output shape or dtype is dynamic/unsupported" : "MIL ValueType cardinality multiplied by element width",
        quantization_state: compression || /quant|dequant|constexpr.*(affine|lut|shift|sparse|palett)/i.test(operation.type) ? "serialized_quantization_transform" : "none",
        quantization: compression || /quant|dequant|constexpr.*(affine|lut|shift|sparse|palett)/i.test(operation.type) ? "serialized_quantization_transform" : "none",
        compression_contract: compression,
        quant_risk: "none", stage_index: opIndex, stage_key: String(operation.type).toUpperCase(), mil_scope: path,
        mil_nested_scopes: operation.blocks.map((_, nestedIndex) => `${path}/op${opIndex}/block${nestedIndex}`),
      };
      if (["cond", "while_loop"].includes(String(operation.type).toLowerCase()) && operation.blocks.length !== 2) {
        throw new Error(`Core ML MIL ${operation.type} must contain exactly two serialized blocks`);
      }
      ops.push(opRow);
      for (let nestedIndex = 0; nestedIndex < operation.blocks.length; nestedIndex += 1) {
        walkBlock(operation.blocks[nestedIndex], chain, opRow.mil_nested_scopes[nestedIndex]);
      }
    }
    const blockOutputTensorIndices = block.outputs.map((name) => resolve(chain, name));
    scopeContracts.set(path, {
      scope: path,
      block_input_tensor_indices: blockInputTensorIndices,
      block_output_tensor_indices: blockOutputTensorIndices,
    });
    return blockOutputTensorIndices;
  };
  const outputTensorIndices = walkBlock(fn.active_block, [rootScope], functionName);
  const nestedBlockOperationCount = ops.filter((op) => op.mil_scope !== functionName).length;
  const computeOps = ops.filter((op) => op.macs_status !== "derived_non_mac_operation");
  const assessed = computeOps.filter((op) => op.macs_decimal != null);
  const completeTotal = assessed.length === computeOps.length && nestedBlockOperationCount === 0;
  const macLedger = coreMlExactLedger(assessed.map((op) => op.macs_decimal), completeTotal);
  const scopes = new Map([...scopeContracts].map(([scope]) => [scope, []]));
  for (const op of ops) {
    const rows = scopes.get(op.mil_scope) || [];
    rows.push(op);
    scopes.set(op.mil_scope, rows);
  }
  const scopeRows = [...scopes].map(([scope, scopeOps]) => {
    const scopeCompute = scopeOps.filter((op) => op.macs_status !== "derived_non_mac_operation");
    const scopeAssessed = scopeCompute.filter((op) => op.macs_decimal != null);
    const scopeMacs = coreMlExactLedger(scopeAssessed.map((op) => op.macs_decimal), scopeAssessed.length === scopeCompute.length);
    const payloadAssessed = scopeOps.filter((op) => op.estimated_bytes != null);
    const payload = coreMlExactLedger(payloadAssessed.map((op) => op.estimated_bytes), payloadAssessed.length === scopeOps.length);
    return {
      scope, scope_class: scope === functionName ? "function_root" : "nested_block", operator_count: scopeOps.length,
      block_input_tensor_indices: scopeContracts.get(scope)?.block_input_tensor_indices || [],
      block_output_tensor_indices: scopeContracts.get(scope)?.block_output_tensor_indices || [],
      mac_compute_operator_count: scopeCompute.length, assessed_mac_operator_count: scopeAssessed.length,
      residual_mac_operator_count: scopeCompute.length - scopeAssessed.length, assessed_nominal_macs: scopeMacs.assessed_value,
      assessed_nominal_macs_decimal: scopeMacs.assessed_value_decimal, complete_nominal_macs: scopeMacs.complete_value,
      complete_nominal_macs_decimal: scopeMacs.complete_value_decimal, assessed_output_payload_operator_count: payloadAssessed.length,
      residual_output_payload_operator_count: scopeOps.length - payloadAssessed.length, assessed_output_payload_bytes: payload.assessed_value,
      assessed_output_payload_bytes_decimal: payload.assessed_value_decimal, complete_output_payload_bytes: payload.complete_value,
      complete_output_payload_bytes_decimal: payload.complete_value_decimal,
      status: scopeAssessed.length === scopeCompute.length && payloadAssessed.length === scopeOps.length ? "assessed" : "partial",
    };
  });
  const totalMacs = macLedger.complete_value;
  const compressionRows = ops.filter((op) => op.compression_contract).map((op) => ({
    op_index: op.index, op_type: op.mil_operation_type, ...op.compression_contract,
  }));
  return {
    function_name: functionName, opset: fn.opset, ops, tensors, input_tensor_indices: inputTensorIndices, output_tensor_indices: outputTensorIndices,
    blob_references: blobReferences, total_macs: totalMacs,
    compression_contract: {
      schema: "deepbom.coreml.mil_compression_ledger.v1",
      status: !compressionRows.length ? "not_applicable_no_serialized_compression_transform"
        : compressionRows.every((row) => row.status.startsWith("assessed_exact")) ? "assessed_exact_serialized_contracts"
          : compressionRows.some((row) => row.status.startsWith("not_assessed")) ? "partial" : "assessed_with_explicit_payload_boundary",
      evidence_class: compressionRows.length ? "OBSERVED/SOURCE_PINNED/DERIVED" : "NOT_APPLICABLE",
      transform_count: compressionRows.length,
      exact_contract_count: compressionRows.filter((row) => row.status.startsWith("assessed_exact")).length,
      partial_contract_count: compressionRows.filter((row) => !row.status.startsWith("assessed_exact")).length,
      transforms: compressionRows,
      source: {
        repository: COREML_MIL_SOURCE.repository, source_commit: COREML_MIL_SOURCE.source_commit,
        ios18_definition: COREML_MIL_SOURCE.compression_ios18_definition,
        ios18_definition_sha256: COREML_MIL_SOURCE.compression_ios18_definition_sha256,
        ios16_definition: COREML_MIL_SOURCE.constexpr_ios16_definition,
        ios16_definition_sha256: COREML_MIL_SOURCE.constexpr_ios16_definition_sha256,
      },
      boundary: "The ledger validates source-defined serialized compression contracts. It does not reconstruct runtime materialization, compressed buffer residency, device placement, latency, or task accuracy.",
    },
    mac_assessment: {
      status: completeTotal
        ? totalMacs == null ? "assessed_all_decoded_compute_ops_exact_decimal_only" : "assessed_all_decoded_compute_ops"
        : nestedBlockOperationCount ? "partial_control_flow_execution_count_not_reconstructed" : "partial_operation_shape_or_safe_integer_range",
      compute_ops: computeOps.length,
      assessed_compute_ops: assessed.length,
      nested_block_operation_count: nestedBlockOperationCount,
      total_macs: totalMacs,
      assessed_macs: macLedger.assessed_value,
      assessed_macs_decimal: macLedger.assessed_value_decimal,
      complete_macs_decimal: macLedger.complete_value_decimal,
      safe_number_mirror_status: macLedger.safe_number_mirror_status,
    },
    scope_intrinsic_cost: {
      schema: "deepbom.coreml.mil_scope_intrinsic_cost.v1", evidence_class: "SOURCE_PINNED_AND_DERIVED",
      status: scopeRows.some((row) => row.status === "partial") ? "partial" : "assessed",
      root_scope: functionName, scope_count: scopeRows.length, nested_scope_count: scopeRows.filter((row) => row.scope_class === "nested_block").length,
      global_execution_total_status: nestedBlockOperationCount ? "not_assessed_control_flow_execution_count" : "complete_root_scope",
      scope_rows: scopeRows,
      method: "Group decoded MIL SSA operations by serialized block scope; sum exact one-invocation nominal MACs and logical output ValueType bytes within each scope.",
      interpretation_boundary: "Nested scope rows are not multiplied or summed into a model total without artifact-known branch selection and loop iteration counts. Output payload is not physical traffic, allocator memory, or runtime scratch.",
    },
  };
}
