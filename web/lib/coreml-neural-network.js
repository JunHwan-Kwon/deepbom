import { Sha256Accumulator } from "./sha256-sync.js";

const MAX_LAYERS = 1_000_000;
const MAX_BLOBS_PER_LAYER = 100_000;

export const COREML_NEURAL_NETWORK_SOURCE = Object.freeze({
  repository: "apple/coremltools",
  release: "9.0",
  source_commit: "428d4b2658dfc44194f27f4f36870751be402ff7",
  neural_network_proto: "mlmodel/format/NeuralNetwork.proto",
  neural_network_proto_sha256: "ca97f083461e61f1cf708c0adb6c28f2110bad63fcea5648235a956b8e7b2497",
  neural_network_validator: "mlmodel/src/Validation/NeuralNetwork/NeuralNetworkLayerValidator.cpp",
  neural_network_validator_sha256: "c4e0d565368627b8bdd9dd46b7d132449d8277295c4ea7a978709493e4e9fc7e",
  neural_network_validator_utils: "mlmodel/src/Validation/NeuralNetwork/NeuralNetworkValidatorUtils.hpp",
  neural_network_validator_utils_sha256: "ad315304ff4d44623e9aa3ca3d3853995980114b47fc3a7a34ae7ff62148bcf8",
  quantization_implementation: "coremltools/models/neural_network/quantization_utils.py",
  quantization_implementation_sha256: "51af467b3b8bd483cd7e39eb13fb2f8814b0221cac556b6f6a68f1a38a544d2d",
});

const LAYER_TYPES = Object.freeze({
  100: "CONVOLUTION", 120: "POOLING", 130: "ACTIVATION", 140: "INNER_PRODUCT", 150: "EMBEDDING",
  160: "BATCHNORM", 165: "MEAN_VARIANCE_NORMALIZE", 170: "L2_NORMALIZE", 175: "SOFTMAX", 180: "LRN",
  190: "CROP", 200: "PADDING", 210: "UPSAMPLE", 211: "RESIZE_BILINEAR", 212: "CROP_RESIZE", 220: "UNARY",
  230: "ADD", 231: "MULTIPLY", 240: "AVERAGE", 245: "SCALE", 250: "BIAS", 260: "MAX", 261: "MIN",
  270: "DOT", 280: "REDUCE", 290: "LOAD_CONSTANT", 300: "RESHAPE", 301: "FLATTEN", 310: "PERMUTE",
  320: "CONCAT", 330: "SPLIT", 340: "SEQUENCE_REPEAT", 345: "REORGANIZE_DATA", 350: "SLICE",
  400: "SIMPLE_RECURRENT", 410: "GRU", 420: "UNIDIRECTIONAL_LSTM", 430: "BIDIRECTIONAL_LSTM", 500: "CUSTOM",
  600: "COPY", 605: "BRANCH", 615: "LOOP", 620: "LOOP_BREAK", 625: "LOOP_CONTINUE", 635: "RANGE_STATIC",
  640: "RANGE_DYNAMIC", 660: "CLIP", 665: "CEIL", 670: "FLOOR", 680: "SIGN", 685: "ROUND", 700: "EXP2",
  710: "SIN", 715: "COS", 720: "TAN", 730: "ASIN", 735: "ACOS", 740: "ATAN", 750: "SINH", 755: "COSH",
  760: "TANH", 770: "ASINH", 775: "ACOSH", 780: "ATANH", 790: "ERF", 795: "GELU", 815: "EQUAL",
  820: "NOT_EQUAL", 825: "LESS_THAN", 827: "LESS_EQUAL", 830: "GREATER_THAN", 832: "GREATER_EQUAL",
  840: "LOGICAL_OR", 845: "LOGICAL_XOR", 850: "LOGICAL_NOT", 855: "LOGICAL_AND", 865: "MOD_BROADCASTABLE",
  870: "MIN_BROADCASTABLE", 875: "MAX_BROADCASTABLE", 880: "ADD_BROADCASTABLE", 885: "POW_BROADCASTABLE",
  890: "DIVIDE_BROADCASTABLE", 895: "FLOOR_DIV_BROADCASTABLE", 900: "MULTIPLY_BROADCASTABLE",
  905: "SUBTRACT_BROADCASTABLE", 920: "TILE", 925: "STACK", 930: "GATHER", 935: "SCATTER", 940: "GATHER_ND",
  945: "SCATTER_ND", 950: "SOFTMAX_ND", 952: "GATHER_ALONG_AXIS", 954: "SCATTER_ALONG_AXIS",
  960: "REVERSE", 965: "REVERSE_SEQUENCE", 975: "SPLIT_ND", 980: "CONCAT_ND", 985: "TRANSPOSE",
  995: "SLICE_STATIC", 1000: "SLICE_DYNAMIC", 1005: "SLIDING_WINDOWS", 1015: "TOP_K", 1020: "ARG_MIN",
  1025: "ARG_MAX", 1040: "EMBEDDING_ND", 1045: "BATCHED_MATMUL", 1065: "GET_SHAPE",
  1070: "LOAD_CONSTANT_ND", 1080: "FILL_LIKE", 1085: "FILL_STATIC", 1090: "FILL_DYNAMIC",
  1100: "BROADCAST_TO_LIKE", 1105: "BROADCAST_TO_STATIC", 1110: "BROADCAST_TO_DYNAMIC", 1120: "SQUEEZE",
  1125: "EXPAND_DIMS", 1130: "FLATTEN_TO_2D", 1135: "RESHAPE_LIKE", 1140: "RESHAPE_STATIC",
  1145: "RESHAPE_DYNAMIC", 1150: "RANK_PRESERVING_RESHAPE", 1155: "CONSTANT_PADDING",
  1170: "RANDOM_NORMAL_LIKE", 1175: "RANDOM_NORMAL_STATIC", 1180: "RANDOM_NORMAL_DYNAMIC",
  1190: "RANDOM_UNIFORM_LIKE", 1195: "RANDOM_UNIFORM_STATIC", 1200: "RANDOM_UNIFORM_DYNAMIC",
  1210: "RANDOM_BERNOULLI_LIKE", 1215: "RANDOM_BERNOULLI_STATIC", 1220: "RANDOM_BERNOULLI_DYNAMIC",
  1230: "CATEGORICAL_DISTRIBUTION", 1250: "REDUCE_L1", 1255: "REDUCE_L2", 1260: "REDUCE_MAX",
  1265: "REDUCE_MIN", 1270: "REDUCE_SUM", 1275: "REDUCE_PROD", 1280: "REDUCE_MEAN", 1285: "REDUCE_LOG_SUM",
  1290: "REDUCE_SUM_SQUARE", 1295: "REDUCE_LOG_SUM_EXP", 1313: "WHERE_NON_ZERO", 1315: "MATRIX_BAND_PART",
  1320: "LOWER_TRIANGULAR", 1325: "UPPER_TRIANGULAR", 1330: "WHERE_BROADCASTABLE",
  1350: "LAYER_NORMALIZATION", 1400: "NON_MAXIMUM_SUPPRESSION", 1450: "ONE_HOT", 1455: "CUM_SUM",
  1460: "CLAMPED_RELU", 1461: "ARG_SORT", 1465: "POOLING_3D", 1466: "GLOBAL_POOLING_3D",
  1470: "SLICE_BY_SIZE", 1471: "CONVOLUTION_3D",
});

const DIRECT_WEIGHT_FIELDS = new Map([
  [100, new Map([[90, "weights"], [91, "bias"]])],
  [140, new Map([[20, "weights"], [21, "bias"]])],
  [150, new Map([[20, "weights"], [21, "bias"]])],
  [160, new Map([[15, "gamma"], [16, "beta"], [17, "mean"], [18, "variance"]])],
  [245, new Map([[2, "scale"], [5, "bias"]])],
  [250, new Map([[2, "bias"]])],
  [290, new Map([[2, "data"]])],
  [1040, new Map([[20, "weights"], [21, "bias"]])],
  [1045, new Map([[8, "weights"], [9, "bias"]])],
  [1070, new Map([[2, "data"]])],
  [1350, new Map([[3, "gamma"], [4, "beta"]])],
  [1471, new Map([[60, "weights"], [61, "bias"]])],
]);

const UNSCANNED_WEIGHT_CAPABLE = new Set([500]);
const AXIS_LAYER_FIELDS = new Set([925, 930, 935, 952, 954, 1015, 1020, 1025]);
const REDUCE_ND_LAYER_FIELDS = new Set([1250, 1255, 1260, 1265, 1270, 1275, 1280, 1285, 1290, 1295]);

function pushBounded(rows, value, label) {
  if (rows.length >= MAX_BLOBS_PER_LAYER) throw new Error(`${label} exceeds ${MAX_BLOBS_PER_LAYER} entries`);
  rows.push(value);
}

function readString(reader, wire, label) {
  const bytes = reader.bytesField(wire, label);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`${label} is not valid UTF-8`);
  }
}

function readFloatStats(reader, wire, label, positive = false, allowNonfinite = false, digest = null, valueSink = null) {
  const result = { count: 0, finite_count: 0, zero_count: 0, negative_zero_count: 0, nan_count: 0, positive_infinity_count: 0, negative_infinity_count: 0, min: null, max: null };
  const read = (bytes) => {
    if (bytes.length % 4 !== 0) throw new Error(`${label} packed float payload is not 4-byte aligned`);
    digest?.update(bytes);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let offset = 0; offset < bytes.length; offset += 4) {
      const value = view.getFloat32(offset, true);
      valueSink?.push(value);
      result.count += 1;
      if (Number.isNaN(value)) { result.nan_count += 1; if (!allowNonfinite) throw new Error(`${label} contains an invalid floating-point value`); continue; }
      if (value === Number.POSITIVE_INFINITY) { result.positive_infinity_count += 1; if (!allowNonfinite) throw new Error(`${label} contains an invalid floating-point value`); continue; }
      if (value === Number.NEGATIVE_INFINITY) { result.negative_infinity_count += 1; if (!allowNonfinite) throw new Error(`${label} contains an invalid floating-point value`); continue; }
      if (positive && value <= 0) throw new Error(`${label} contains an invalid floating-point value`);
      result.finite_count += 1;
      if (value === 0) { result.zero_count += 1; if (Object.is(value, -0)) result.negative_zero_count += 1; }
      result.min = result.min == null ? value : Math.min(result.min, value);
      result.max = result.max == null ? value : Math.max(result.max, value);
    }
  };
  if (wire === 2) read(reader.bytesField(wire, label));
  else if (wire === 5) {
    const start = reader.position;
    reader.advance(4);
    read(reader.bytes.subarray(start, start + 4));
  } else throw new Error(`${label} has unsupported wire type ${wire}`);
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

function byteValueStats(bytes, kind) {
  const result = { count: 0, finite_count: 0, zero_count: 0, negative_zero_count: 0, nan_count: 0, positive_infinity_count: 0, negative_infinity_count: 0, min: null, max: null };
  const add = (value) => {
    result.count += 1;
    if (Number.isNaN(value)) { result.nan_count += 1; return; }
    if (value === Number.POSITIVE_INFINITY) { result.positive_infinity_count += 1; return; }
    if (value === Number.NEGATIVE_INFINITY) { result.negative_infinity_count += 1; return; }
    result.finite_count += 1;
    if (value === 0) { result.zero_count += 1; if (Object.is(value, -0)) result.negative_zero_count += 1; }
    result.min = result.min == null ? value : Math.min(result.min, value);
    result.max = result.max == null ? value : Math.max(result.max, value);
  };
  if (kind === "float16") {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let offset = 0; offset < bytes.length; offset += 2) add(halfToNumber(view.getUint16(offset, true)));
  } else for (const value of bytes) add(kind === "int8_dynamic" && value > 127 ? value - 256 : value);
  return result;
}

function mergeValueStats(target, source) {
  for (const key of ["count", "finite_count", "zero_count", "negative_zero_count", "nan_count", "positive_infinity_count", "negative_infinity_count"]) target[key] += Number(source[key] || 0);
  if (source.min != null) target.min = target.min == null ? source.min : Math.min(target.min, source.min);
  if (source.max != null) target.max = target.max == null ? source.max : Math.max(target.max, source.max);
}

function parseQuantization(reader) {
  const result = { number_of_bits: null, scheme: null, scale_count: 0, scale_min: null, scale_max: null, bias_count: 0, lookup_table_count: 0 };
  const scales = [];
  const biases = [];
  const lookupTable = [];
  const scaleDigest = new Sha256Accumulator();
  const biasDigest = new Sha256Accumulator();
  const lookupDigest = new Sha256Accumulator();
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.number_of_bits = reader.int64Field(wire, singular, 1, "numberOfBits");
    else if (field === 101 || field === 102) {
      if (result.scheme) throw new Error("Core ML QuantizationParams contains multiple quantization schemes");
      const params = reader.message(wire, `CoreML.QuantizationParams.${field}`);
      result.scheme = field === 101 ? "linear" : "lookup_table";
      while (!params.done) {
        const item = params.key();
        if (field === 101 && item.field === 1) {
          const stats = readFloatStats(params, item.wire, "Core ML linear scale", true, false, scaleDigest, scales);
          result.scale_count += stats.count;
          if (stats.count) {
            result.scale_min = result.scale_min == null ? stats.min : Math.min(result.scale_min, stats.min);
            result.scale_max = result.scale_max == null ? stats.max : Math.max(result.scale_max, stats.max);
          }
        }
        else if (field === 101 && item.field === 2) result.bias_count += readFloatStats(params, item.wire, "Core ML linear bias", false, false, biasDigest, biases).count;
        else if (field === 102 && item.field === 1) result.lookup_table_count += readFloatStats(params, item.wire, "Core ML lookup table", false, false, lookupDigest, lookupTable).count;
        else params.skip(item.wire);
      }
    } else reader.skip(wire);
  }
  if (!Number.isSafeInteger(result.number_of_bits) || result.number_of_bits < 1 || result.number_of_bits > 8 || !result.scheme) {
    throw new Error("Core ML QuantizationParams is missing a valid 1-8 bit scheme");
  }
  if (result.scheme === "linear" && !result.scale_count) throw new Error("Core ML linear quantization has no positive scale");
  if (result.scheme === "lookup_table" && result.lookup_table_count !== 2 ** result.number_of_bits) {
    throw new Error("Core ML lookup-table cardinality does not match numberOfBits");
  }
  result.scale_payload_sha256 = scales.length ? scaleDigest.digestHex() : null;
  result.bias_payload_sha256 = biases.length ? biasDigest.digestHex() : null;
  result.lookup_table_payload_sha256 = lookupTable.length ? lookupDigest.digestHex() : null;
  Object.defineProperties(result, {
    _scales: { value: scales, enumerable: false },
    _biases: { value: biases, enumerable: false },
    _lookup_table: { value: lookupTable, enumerable: false },
  });
  return result;
}

function parseWeightParams(reader, role) {
  const result = {
    role, storage: null, value_count: 0, value_count_status: "exact", byte_length: 0, quantization: null, is_updatable: false,
  };
  const digest = new Sha256Accumulator();
  const values = byteValueStats(new Uint8Array(), "uint8");
  const rawPayloads = [];
  const storageFields = new Set();
  const byteStorageFields = new Set();
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      storageFields.add("float32");
      const decoded = readFloatStats(reader, wire, `Core ML ${role} floatValue`, false, true, digest);
      mergeValueStats(values, decoded);
      result.value_count += decoded.count;
      result.byte_length += decoded.count * 4;
    } else if (field === 2 || field === 30 || field === 31) {
      if (byteStorageFields.has(field)) throw new Error(`Core ML ${role} repeats singular weight payload field ${field}`);
      byteStorageFields.add(field);
      const storage = field === 2 ? "float16" : field === 30 ? "raw_quantized" : "int8_dynamic";
      storageFields.add(storage);
      const bytes = reader.bytesField(wire, `Core ML ${role} ${storage}`);
      digest.update(bytes);
      result.byte_length += bytes.length;
      if (field === 2 && bytes.length % 2 !== 0) throw new Error(`Core ML ${role} float16 payload has odd byte cardinality`);
      if (field === 2) {
        const decoded = byteValueStats(bytes, "float16");
        mergeValueStats(values, decoded);
        result.value_count += decoded.count;
      } else if (field === 31) {
        const decoded = byteValueStats(bytes, "int8_dynamic");
        mergeValueStats(values, decoded);
        result.value_count += decoded.count;
      }
      rawPayloads.push(bytes);
    } else if (field === 40) {
      if (result.quantization) throw new Error(`Core ML ${role} repeats quantization metadata`);
      result.quantization = parseQuantization(reader.message(wire, `CoreML.WeightParams.${role}.quantization`));
    } else if (field === 50) result.is_updatable = reader.intField(wire, singular, 50, "isUpdatable") !== 0;
    else reader.skip(wire);
  }
  if (storageFields.size > 1) throw new Error(`Core ML ${role} declares conflicting weight storage fields`);
  result.storage = [...storageFields][0] || "empty";
  if (["raw_quantized", "int8_dynamic"].includes(result.storage) && !result.quantization) throw new Error(`Core ML ${role} quantized bytes have no QuantizationParams`);
  if (!["raw_quantized", "int8_dynamic"].includes(result.storage) && result.quantization) throw new Error(`Core ML ${role} quantization metadata is not bound to quantized bytes`);
  if (result.storage === "int8_dynamic" && (result.quantization.number_of_bits !== 8
    || result.quantization.scheme !== "linear" || result.quantization.scale_count !== 1 || result.quantization.bias_count !== 0)) {
    throw new Error(`Core ML ${role} int8 dynamic quantization contract is invalid`);
  }
  if (result.storage === "raw_quantized") {
    result.value_count = null;
    result.value_count_status = "not_assessed_without_parent_parameter_cardinality";
    result.packed_code_capacity = Math.floor((result.byte_length * 8) / result.quantization.number_of_bits);
    result.packed_code_capacity_status = "upper_bound_including_possible_terminal_padding_bits";
  }
  const nonfinite = values.nan_count + values.positive_infinity_count + values.negative_infinity_count;
  result.numerical_integrity = {
    schema: "deepbom.coreml.weight_integrity.v1",
    status: result.storage === "raw_quantized" ? "pending_parent_cardinality" : "assessed",
    payload_sha256: result.byte_length ? digest.digestHex() : null,
    decoded_value_count: result.storage === "raw_quantized" ? null : values.count,
    finite_count: result.storage === "raw_quantized" ? null : values.finite_count,
    zero_count: result.storage === "raw_quantized" ? null : values.zero_count,
    negative_zero_count: result.storage === "raw_quantized" ? null : values.negative_zero_count,
    nan_count: result.storage === "raw_quantized" ? null : values.nan_count,
    positive_infinity_count: result.storage === "raw_quantized" ? null : values.positive_infinity_count,
    negative_infinity_count: result.storage === "raw_quantized" ? null : values.negative_infinity_count,
    nonfinite_count: result.storage === "raw_quantized" ? null : nonfinite,
    finite_min: result.storage === "raw_quantized" ? null : values.min,
    finite_max: result.storage === "raw_quantized" ? null : values.max,
    all_zero: result.storage === "raw_quantized" ? null : values.count > 0 && values.zero_count === values.count,
    constant: result.storage === "raw_quantized" ? null : values.count > 0 && nonfinite === 0 && values.min === values.max,
  };
  Object.defineProperty(result, "_raw_payloads", { value: rawPayloads, enumerable: false, configurable: true });
  return result;
}

function repeatedUint64(reader, wire, label) {
  const values = [];
  const read = (source) => {
    const value = source.rawVarint();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range`);
    values.push(Number(value));
  };
  if (wire === 0) read(reader);
  else if (wire === 2) {
    const packed = reader.message(wire, label);
    while (!packed.done) read(packed);
  } else throw new Error(`${label} has unsupported wire type ${wire}`);
  return values;
}

function repeatedInt64(reader, wire, label) {
  const values = [];
  const read = (source) => {
    const value = BigInt.asIntN(64, source.rawVarint());
    if (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds the safe integer range`);
    values.push(Number(value));
  };
  if (wire === 0) read(reader);
  else if (wire === 2) {
    const packed = reader.message(wire, label);
    while (!packed.done) read(packed);
  } else throw new Error(`${label} has unsupported wire type ${wire}`);
  return values;
}

function parseBorderAmounts(reader) {
  const result = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field !== 10) { reader.skip(wire); continue; }
    const edge = reader.message(wire, "CoreML.BorderAmounts.EdgeSizes");
    let start = 0;
    let end = 0;
    const singular = new Set();
    while (!edge.done) {
      const item = edge.key();
      if (item.field === 1) start = edge.int64Field(item.wire, singular, 1, "startEdgeSize");
      else if (item.field === 2) end = edge.int64Field(item.wire, singular, 2, "endEdgeSize");
      else edge.skip(item.wire);
    }
    result.push([start, end]);
  }
  if (result.length && result.length !== 2) throw new Error("Core ML 2-D padding must contain exactly H and W edge amounts");
  return result.length ? result : [[0, 0], [0, 0]];
}

function parseValidPadding(reader) {
  let amounts = [[0, 0], [0, 0]];
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (singular.has(field)) throw new Error("Core ML ValidPadding repeats paddingAmounts");
      singular.add(field);
      amounts = parseBorderAmounts(reader.message(wire, "CoreML.BorderAmounts"));
    } else reader.skip(wire);
  }
  return { kind: "valid", amounts };
}

function addDecodedValue(stats, value) {
  stats.count += 1;
  if (Number.isNaN(value)) { stats.nan_count += 1; return; }
  if (value === Number.POSITIVE_INFINITY) { stats.positive_infinity_count += 1; return; }
  if (value === Number.NEGATIVE_INFINITY) { stats.negative_infinity_count += 1; return; }
  stats.finite_count += 1;
  if (value === 0) { stats.zero_count += 1; if (Object.is(value, -0)) stats.negative_zero_count += 1; }
  stats.min = stats.min == null ? value : Math.min(stats.min, value);
  stats.max = stats.max == null ? value : Math.max(stats.max, value);
}

function packedMsbCode(bytes, bitOffset, bitWidth) {
  let code = 0;
  for (let lane = 0; lane < bitWidth; lane += 1) {
    const position = bitOffset + lane;
    code = code * 2 + ((bytes[position >>> 3] >>> (7 - (position & 7))) & 1);
  }
  return code;
}

function bindWeightCardinality(weight, shape, axis, label, expectedQuantizationChannels = null) {
  const cardinality = exactCardinality(shape, label);
  if (!Number.isSafeInteger(cardinality) || cardinality < 0) throw new Error(`${label} has an invalid declared cardinality`);
  if (weight.storage === "raw_quantized") {
    const bits = weight.quantization.number_of_bits;
    const expectedBytes = Math.ceil(cardinality * bits / 8);
    if (weight.byte_length !== expectedBytes) throw new Error(`${label} packed payload is ${weight.byte_length} bytes; expected ${expectedBytes}`);
    const histogram = new Array(2 ** bits).fill(0);
    const bytes = weight._raw_payloads?.[0];
    if (!bytes) throw new Error(`${label} packed payload is missing`);
    const quantization = weight.quantization;
    const channelCount = expectedQuantizationChannels ?? shape[axis];
    if (!Number.isSafeInteger(channelCount) || channelCount <= 0) throw new Error(`${label} has an invalid quantization axis cardinality`);
    if (quantization.scheme === "linear" && (quantization.scale_count !== channelCount || quantization.bias_count !== channelCount)) {
      throw new Error(`${label} linear scale/bias cardinality does not match quantization axis ${axis} (${channelCount})`);
    }
    const channelStride = exactCardinality(shape.slice(axis + 1), `${label} quantization channel stride`);
    const decoded = byteValueStats(new Uint8Array(), "uint8");
    for (let valueIndex = 0; valueIndex < cardinality; valueIndex += 1) {
      const code = packedMsbCode(bytes, valueIndex * bits, bits);
      histogram[code] += 1;
      const channel = Math.floor(valueIndex / channelStride) % channelCount;
      const value = quantization.scheme === "linear"
        ? code * quantization._scales[channel] + quantization._biases[channel]
        : quantization._lookup_table[code];
      addDecodedValue(decoded, value);
    }
    const nonfinite = decoded.nan_count + decoded.positive_infinity_count + decoded.negative_infinity_count;
    weight.value_count = cardinality;
    weight.value_count_status = "exact_from_parent_parameter_contract";
    weight.packed_code_capacity_status = "validated_against_parent_parameter_cardinality";
    Object.assign(weight.quantization, {
      axis: channelCount === 1 ? null : axis,
      channel_count: channelCount,
      granularity: channelCount === 1 ? "per_tensor" : "per_axis",
      dequantization_formula: quantization.scheme === "linear" ? "real = stored_unsigned_code * scale[channel] + bias[channel]" : "real = lookup_table[stored_unsigned_code]",
      packed_bit_order: "MSB-first within each byte",
    });
    weight.numerical_integrity = {
      ...weight.numerical_integrity,
      status: "assessed_dequantized_quantized_codes",
      decoded_value_count: cardinality,
      finite_count: decoded.finite_count,
      zero_count: decoded.zero_count,
      negative_zero_count: decoded.negative_zero_count,
      nan_count: decoded.nan_count,
      positive_infinity_count: decoded.positive_infinity_count,
      negative_infinity_count: decoded.negative_infinity_count,
      nonfinite_count: nonfinite,
      finite_min: decoded.min,
      finite_max: decoded.max,
      all_zero: cardinality > 0 && decoded.zero_count === cardinality,
      constant: cardinality > 0 && nonfinite === 0 && decoded.min === decoded.max,
      quant_code_levels_used: histogram.filter((count) => count > 0).length,
      quant_code_level_capacity: histogram.length,
      quant_code_utilization_ratio: histogram.filter((count) => count > 0).length / histogram.length,
      zero_code_count: histogram[0],
    };
  } else if (weight.storage === "int8_dynamic") {
    if (weight.value_count !== cardinality) throw new Error(`${label} contains ${weight.value_count} values; expected ${cardinality}`);
    const bytes = weight._raw_payloads?.[0];
    if (!bytes || bytes.length !== cardinality) throw new Error(`${label} INT8 payload cardinality is inconsistent`);
    const scale = weight.quantization._scales[0];
    const decoded = byteValueStats(new Uint8Array(), "uint8");
    const codes = new Set();
    for (const raw of bytes) {
      const code = raw > 127 ? raw - 256 : raw;
      codes.add(code);
      addDecodedValue(decoded, code * scale);
    }
    const nonfinite = decoded.nan_count + decoded.positive_infinity_count + decoded.negative_infinity_count;
    Object.assign(weight.quantization, {
      axis: null,
      channel_count: 1,
      granularity: "per_tensor",
      dequantization_formula: "real = stored_signed_int8_code * scale",
      packed_bit_order: "not_applicable_byte_aligned",
    });
    weight.numerical_integrity = {
      ...weight.numerical_integrity,
      status: "assessed_dequantized_int8_dynamic_codes",
      finite_count: decoded.finite_count,
      zero_count: decoded.zero_count,
      negative_zero_count: decoded.negative_zero_count,
      nan_count: decoded.nan_count,
      positive_infinity_count: decoded.positive_infinity_count,
      negative_infinity_count: decoded.negative_infinity_count,
      nonfinite_count: nonfinite,
      finite_min: decoded.min,
      finite_max: decoded.max,
      all_zero: cardinality > 0 && decoded.zero_count === cardinality,
      constant: cardinality > 0 && nonfinite === 0 && decoded.min === decoded.max,
      quant_code_levels_used: codes.size,
      quant_code_level_capacity: 256,
      quant_code_utilization_ratio: codes.size / 256,
    };
  } else if (weight.value_count !== cardinality) {
    throw new Error(`${label} contains ${weight.value_count} values; expected ${cardinality}`);
  }
  delete weight._raw_payloads;
}

function parseSamePadding(reader) {
  let asymmetryMode = 0;
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) asymmetryMode = reader.intField(wire, singular, 1, "asymmetryMode");
    else reader.skip(wire);
  }
  if (asymmetryMode > 1) throw new Error("Core ML SamePadding has an unknown asymmetry mode");
  return { kind: "same", asymmetry_mode: asymmetryMode };
}

function appendRepeated(target, values, label, expected = null) {
  target.push(...values);
  if (target.length > 16) throw new Error(`${label} exceeds the bounded rank`);
  if (expected != null && target.length > expected) throw new Error(`${label} has more than ${expected} values`);
}

function roleWeight(weights, role) {
  const matches = weights.filter((item) => item.role === role);
  if (matches.length > 1) throw new Error(`Core ML layer repeats ${role} WeightParams`);
  return matches[0] || null;
}

function exactCardinality(values, label) {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || (value && product > Math.floor(Number.MAX_SAFE_INTEGER / value))) {
      throw new Error(`${label} cardinality exceeds the safe integer range`);
    }
    product *= value;
  }
  return product;
}

function bindOptionalWeight(weights, role, shape, required, label, axis = 0, expectedQuantizationChannels = null) {
  const weight = roleWeight(weights, role);
  if (!weight) {
    if (required) throw new Error(`${label} is missing ${role} WeightParams`);
    return;
  }
  if (!required && weight.storage === "empty") { delete weight._raw_payloads; return; }
  bindWeightCardinality(weight, shape, axis, `${label} ${role}`, expectedQuantizationChannels);
}

const SIMPLE_RECURRENT_WEIGHT_FIELDS = new Map([
  [30, "input_weight"], [31, "recursion_weight"], [32, "bias"],
]);
const GRU_WEIGHT_FIELDS = new Map([
  [30, "update_input_weight"], [31, "reset_input_weight"], [32, "output_input_weight"],
  [50, "update_recursion_weight"], [51, "reset_recursion_weight"], [52, "output_recursion_weight"],
  [70, "update_bias"], [71, "reset_bias"], [72, "output_bias"],
]);
const LSTM_WEIGHT_FIELDS = new Map([
  [1, "input_gate_input_weight"], [2, "forget_gate_input_weight"], [3, "block_input_weight"], [4, "output_gate_input_weight"],
  [20, "input_gate_recursion_weight"], [21, "forget_gate_recursion_weight"], [22, "block_recursion_weight"], [23, "output_gate_recursion_weight"],
  [40, "input_gate_bias"], [41, "forget_gate_bias"], [42, "block_bias"], [43, "output_gate_bias"],
  [60, "input_gate_peephole"], [61, "forget_gate_peephole"], [62, "output_gate_peephole"],
]);
const LSTM_BOOL_FIELDS = new Map([
  [10, "sequenceOutput"], [20, "hasBiasVectors"], [30, "forgetBias"],
  [40, "hasPeepholeVectors"], [50, "coupledInputAndForgetGate"],
]);

function readBoolField(reader, wire, singular, field, name) {
  const value = reader.intField(wire, singular, field, name);
  if (value > 1) throw new Error(`Core ML ${name} is not a valid bool`);
  return value !== 0;
}

function parseLstmParams(reader) {
  const result = {
    sequence_output: false,
    has_bias_vectors: false,
    forget_bias: false,
    has_peephole_vectors: false,
    coupled_input_and_forget_gate: false,
    cell_clip_threshold: 0,
  };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (LSTM_BOOL_FIELDS.has(field)) {
      const value = readBoolField(reader, wire, singular, field, LSTM_BOOL_FIELDS.get(field));
      if (field === 10) result.sequence_output = value;
      else if (field === 20) result.has_bias_vectors = value;
      else if (field === 30) result.forget_bias = value;
      else if (field === 40) result.has_peephole_vectors = value;
      else result.coupled_input_and_forget_gate = value;
    } else if (field === 60) {
      const value = reader.floatField(wire, singular, field, "cellClipThreshold");
      if (value < 0) throw new Error("Core ML cellClipThreshold must be non-negative");
      result.cell_clip_threshold = value;
    } else reader.skip(wire);
  }
  return result;
}

function parseLstmWeightParams(reader, prefix) {
  const weights = [];
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (LSTM_WEIGHT_FIELDS.has(field)) {
      const role = `${prefix}${LSTM_WEIGHT_FIELDS.get(field)}`;
      weights.push(parseWeightParams(reader.message(wire, `CoreML.LSTMWeightParams.${role}`), role));
    } else reader.skip(wire);
  }
  return weights;
}

function bindRecurrentMatrixSet(weights, prefix, inputSize, outputSize, options, label) {
  const inputRoles = options.kind === "gru"
    ? ["update_input_weight", "reset_input_weight", "output_input_weight"]
    : ["input_gate_input_weight", "forget_gate_input_weight", "block_input_weight", "output_gate_input_weight"];
  const recursionRoles = options.kind === "gru"
    ? ["update_recursion_weight", "reset_recursion_weight", "output_recursion_weight"]
    : ["input_gate_recursion_weight", "forget_gate_recursion_weight", "block_recursion_weight", "output_gate_recursion_weight"];
  const biasRoles = options.kind === "gru"
    ? ["update_bias", "reset_bias", "output_bias"]
    : ["input_gate_bias", "forget_gate_bias", "block_bias", "output_gate_bias"];
  for (const role of inputRoles) bindOptionalWeight(weights, `${prefix}${role}`, [outputSize, inputSize], true, label);
  for (const role of recursionRoles) bindOptionalWeight(weights, `${prefix}${role}`, [outputSize, outputSize], true, label);
  for (const role of biasRoles) bindOptionalWeight(weights, `${prefix}${role}`, [outputSize], options.hasBias, label);
  if (options.kind === "lstm") {
    for (const role of ["input_gate_peephole", "forget_gate_peephole", "output_gate_peephole"]) {
      bindOptionalWeight(weights, `${prefix}${role}`, [outputSize], options.hasPeephole, label);
    }
  }
}

function finalizeLayerParameters(layerField, attributes, weights) {
  const label = `Core ML ${LAYER_TYPES[layerField] || `layer field ${layerField}`}`;
  if (layerField === 100) {
    attributes.n_groups ||= 1;
    if (!attributes.kernel_size.length) attributes.kernel_size = [3, 3];
    if (!attributes.stride.length) attributes.stride = [1, 1];
    if (!attributes.dilation.length) attributes.dilation = [1, 1];
    attributes.padding ||= { kind: "valid", amounts: [[0, 0], [0, 0]] };
    if (![attributes.output_channels, attributes.kernel_channels, attributes.n_groups, ...attributes.kernel_size, ...attributes.stride, ...attributes.dilation]
      .every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has an invalid channel, group, kernel, stride, or dilation value`);
    if (attributes.kernel_size.length !== 2 || attributes.stride.length !== 2 || attributes.dilation.length !== 2) throw new Error(`${label} expects two spatial values`);
    if (attributes.output_channels % attributes.n_groups !== 0) throw new Error(`${label} outputChannels is not divisible by nGroups`);
    const weightCount = exactCardinality(attributes.is_deconvolution
      ? [attributes.kernel_channels, attributes.output_channels / attributes.n_groups, ...attributes.kernel_size]
      : [attributes.output_channels, attributes.kernel_channels, ...attributes.kernel_size], `${label} weights`);
    const weightShape = attributes.is_deconvolution
      ? [attributes.kernel_channels, attributes.output_channels / attributes.n_groups, ...attributes.kernel_size]
      : [attributes.output_channels, attributes.kernel_channels, ...attributes.kernel_size];
    if (exactCardinality(weightShape, `${label} weights`) !== weightCount) throw new Error(`${label} weight cardinality derivation is inconsistent`);
    bindOptionalWeight(weights, "weights", weightShape, true, label, attributes.is_deconvolution ? 1 : 0);
    bindOptionalWeight(weights, "bias", [attributes.output_channels], attributes.has_bias, label);
  } else if (layerField === 140) {
    if (![attributes.input_channels, attributes.output_channels].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid channel cardinality`);
    bindOptionalWeight(weights, "weights", [attributes.output_channels, attributes.input_channels], true, label);
    bindOptionalWeight(weights, "bias", [attributes.output_channels], attributes.has_bias, label);
  } else if (layerField === 150) {
    if (![attributes.input_dim, attributes.output_channels].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid dictionary or output cardinality`);
    bindOptionalWeight(weights, "weights", [attributes.output_channels, attributes.input_dim], true, label);
    bindOptionalWeight(weights, "bias", [attributes.output_channels], attributes.has_bias, label);
  } else if (layerField === 160) {
    if (!Number.isSafeInteger(attributes.channels) || attributes.channels <= 0) throw new Error(`${label} has an invalid channel cardinality`);
    bindOptionalWeight(weights, "gamma", [attributes.channels], true, label, 0, 1);
    bindOptionalWeight(weights, "beta", [attributes.channels], true, label, 0, 1);
    bindOptionalWeight(weights, "mean", [attributes.channels], !attributes.compute_mean_var, label, 0, 1);
    bindOptionalWeight(weights, "variance", [attributes.channels], !attributes.compute_mean_var, label, 0, 1);
  } else if (layerField === 245) {
    for (const [name, shape] of [["shapeScale", attributes.shape_scale], ["shapeBias", attributes.shape_bias]]) {
      if (name === "shapeBias" && !attributes.has_bias) continue;
      if (![1, 3].includes(shape.length) || shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
        throw new Error(`${label} ${name} must contain one or three positive dimensions`);
      }
    }
    const scaleChannels = attributes.shape_scale.length === 3 && attributes.shape_scale[0] > 1 ? attributes.shape_scale[0] : 1;
    bindOptionalWeight(weights, "scale", attributes.shape_scale, true, label, 0, scaleChannels);
    if (attributes.has_bias) {
      const biasChannels = attributes.shape_bias.length === 3 && attributes.shape_bias[0] > 1 ? attributes.shape_bias[0] : 1;
      bindOptionalWeight(weights, "bias", attributes.shape_bias, true, label, 0, biasChannels);
    } else bindOptionalWeight(weights, "bias", [0], false, label, 0, 1);
  } else if (layerField === 250) {
    if (![1, 3].includes(attributes.shape.length) || attributes.shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
      throw new Error(`${label} shape must contain one or three positive dimensions`);
    }
    const biasChannels = attributes.shape.length === 3 && attributes.shape[0] > 1 ? attributes.shape[0] : 1;
    bindOptionalWeight(weights, "bias", attributes.shape, true, label, 0, biasChannels);
  } else if (layerField === 1040) {
    if (![attributes.vocab_size, attributes.embedding_size].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid vocabulary or embedding cardinality`);
    bindOptionalWeight(weights, "weights", [attributes.embedding_size, attributes.vocab_size], true, label);
    bindOptionalWeight(weights, "bias", [attributes.embedding_size], attributes.has_bias, label);
  } else if (layerField === 1045 && roleWeight(weights, "weights")) {
    if (![attributes.weight_first_dimension, attributes.weight_second_dimension].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid weight dimensions`);
    bindOptionalWeight(weights, "weights", [attributes.weight_second_dimension, attributes.weight_first_dimension], true, label);
    bindOptionalWeight(weights, "bias", [attributes.weight_second_dimension], attributes.has_bias, label);
  } else if (layerField === 290 || layerField === 1070) {
    if (!attributes.shape.length || attributes.shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`${label} has an invalid constant shape`);
    const count = exactCardinality(attributes.shape, `${label} constant`);
    bindOptionalWeight(weights, "data", [count], true, label);
  } else if (layerField === 400) {
    if (![attributes.input_vector_size, attributes.output_vector_size].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid vector cardinality`);
    bindOptionalWeight(weights, "input_weight", [attributes.output_vector_size, attributes.input_vector_size], true, label);
    bindOptionalWeight(weights, "recursion_weight", [attributes.output_vector_size, attributes.output_vector_size], true, label);
    bindOptionalWeight(weights, "bias", [attributes.output_vector_size], attributes.has_bias_vectors, label);
  } else if (layerField === 410) {
    if (![attributes.input_vector_size, attributes.output_vector_size].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid vector cardinality`);
    bindRecurrentMatrixSet(weights, "", attributes.input_vector_size, attributes.output_vector_size, {
      kind: "gru", hasBias: attributes.has_bias_vectors, hasPeephole: false,
    }, label);
  } else if (layerField === 420 || layerField === 430) {
    if (![attributes.input_vector_size, attributes.output_vector_size].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has invalid vector cardinality`);
    const directions = layerField === 430 ? ["forward_", "backward_"] : [""];
    if (layerField === 430 && attributes.weight_set_count !== 2) throw new Error(`${label} must contain exactly two LSTMWeightParams messages`);
    for (const prefix of directions) bindRecurrentMatrixSet(weights, prefix, attributes.input_vector_size, attributes.output_vector_size, {
      kind: "lstm",
      hasBias: attributes.lstm_params.has_bias_vectors,
      hasPeephole: attributes.lstm_params.has_peephole_vectors,
    }, label);
  } else if (layerField === 1350) {
    if (!attributes.normalized_shape.length || attributes.normalized_shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`${label} has an invalid normalizedShape`);
    if (!Number.isFinite(attributes.epsilon) || attributes.epsilon <= 0) throw new Error(`${label} epsilon must be finite and positive`);
    bindOptionalWeight(weights, "gamma", attributes.normalized_shape, true, label);
    bindOptionalWeight(weights, "beta", attributes.normalized_shape, true, label);
  } else if (layerField === 310) {
    if (!attributes.axes.length) attributes.axes = [0, 1, 2, 3];
    if (attributes.axes.length !== 4 || new Set(attributes.axes).size !== 4
      || attributes.axes.some((value) => value < 0 || value > 3)) throw new Error(`${label} axes must be a permutation of [0,1,2,3]`);
  } else if (layerField === 985) {
    if (!attributes.axes.length || new Set(attributes.axes).size !== attributes.axes.length) throw new Error(`${label} axes must be a non-empty permutation`);
  } else if (layerField === 975) {
    if (attributes.split_sizes.some((value) => !Number.isSafeInteger(value) || value <= 0)) throw new Error(`${label} splitSizes must be positive integers`);
    if (!attributes.split_sizes.length && (!Number.isSafeInteger(attributes.num_splits) || attributes.num_splits < 2)) throw new Error(`${label} must declare at least two splits`);
  } else if (layerField === 1120) {
    if (!attributes.squeeze_all && !attributes.axes.length) throw new Error(`${label} must declare axes or squeezeAll`);
  } else if (layerField === 1125) {
    if (!attributes.axes.length) throw new Error(`${label} must declare at least one axis`);
  } else if (layerField === 1140) {
    if (!attributes.target_shape.length || attributes.target_shape.some((value) => !Number.isSafeInteger(value) || value === 0 || value < -1)
      || attributes.target_shape.filter((value) => value === -1).length > 1) throw new Error(`${label} targetShape must contain positive dimensions and at most one -1`);
  } else if (layerField === 1465) {
    if (![0, 1].includes(attributes.pooling_type) || ![0, 1, 2].includes(attributes.padding_type)
      || [...attributes.kernel_size, ...attributes.stride].some((value) => !Number.isSafeInteger(value) || value <= 0)
      || attributes.custom_padding.flat().some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`${label} has an invalid type, kernel, stride, or padding contract`);
  } else if (layerField === 1466 && ![0, 1].includes(attributes.pooling_type)) {
    throw new Error(`${label} has an unknown pooling type`);
  } else if (layerField === 1471) {
    const values = [attributes.output_channels, attributes.input_channels, attributes.n_groups,
      ...attributes.kernel_size, ...attributes.stride, ...attributes.dilation];
    if (!values.every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error(`${label} has an invalid channel, group, kernel, stride, or dilation value`);
    if (attributes.input_channels % attributes.n_groups !== 0 || attributes.output_channels % attributes.n_groups !== 0
      || attributes.n_groups > attributes.input_channels) throw new Error(`${label} has an invalid grouped-convolution contract`);
    if (![0, 1, 2].includes(attributes.padding_type)) throw new Error(`${label} has an unknown paddingType`);
    if (attributes.output_shape.length && (!attributes.is_deconvolution || attributes.output_shape.length !== 3
      || attributes.output_shape.some((value) => !Number.isSafeInteger(value) || value <= 0))) {
      throw new Error(`${label} has an invalid deconvolution outputShape`);
    }
    const weightShape = attributes.is_deconvolution
      ? [attributes.output_channels / attributes.n_groups, attributes.input_channels, ...attributes.kernel_size]
      : [attributes.output_channels, attributes.input_channels / attributes.n_groups, ...attributes.kernel_size];
    bindOptionalWeight(weights, "weights", weightShape, true, label);
    bindOptionalWeight(weights, "bias", [attributes.output_channels], attributes.has_bias, label);
  }
  for (const weight of weights) {
    if (weight._raw_payloads && weight.numerical_integrity.status === "pending_parent_cardinality") {
      delete weight._raw_payloads;
      weight.numerical_integrity.status = "not_assessed_parent_parameter_cardinality_unimplemented";
    }
  }
}

function parseLayerParameters(layerField, reader) {
  const weights = [];
  const attributes = {};
  const direct = DIRECT_WEIGHT_FIELDS.get(layerField);
  const singular = new Set();
  const setPadding = (value) => {
    if (attributes.padding) throw new Error(`Core ML ${LAYER_TYPES[layerField]} contains multiple padding oneof values`);
    attributes.padding = value;
  };
  if (layerField === 100) Object.assign(attributes, { output_channels: null, kernel_channels: null, n_groups: 1, kernel_size: [], stride: [], dilation: [], padding: null, is_deconvolution: false, has_bias: false, output_shape: [] });
  else if (layerField === 120) Object.assign(attributes, { pooling_type: 0, kernel_size: [], stride: [], padding: null, exclude_padding: false, global_pooling: false });
  else if (layerField === 140) Object.assign(attributes, { input_channels: null, output_channels: null, has_bias: false, int8_dynamic_quantize: false });
  else if (layerField === 150) Object.assign(attributes, { input_dim: null, output_channels: null, has_bias: false });
  else if (layerField === 160) Object.assign(attributes, { channels: null, compute_mean_var: false, instance_normalization: false, epsilon: 1e-5 });
  else if (layerField === 245) Object.assign(attributes, { shape_scale: [], has_bias: false, shape_bias: [] });
  else if (layerField === 250) attributes.shape = [];
  else if (layerField === 1040) Object.assign(attributes, { vocab_size: null, embedding_size: null, has_bias: false });
  else if (layerField === 1045) Object.assign(attributes, { transpose_a: false, transpose_b: false, weight_first_dimension: null, weight_second_dimension: null, has_bias: false, int8_dynamic_quantize: false });
  else if (layerField === 290 || layerField === 1070) attributes.shape = [];
  else if (layerField === 300) Object.assign(attributes, { target_shape: [], mode: 0 });
  else if (layerField === 301) attributes.mode = 0;
  else if (layerField === 310 || layerField === 985) attributes.axes = [];
  else if (layerField === 320) attributes.sequence_concat = false;
  else if (layerField === 330) attributes.n_outputs = null;
  else if (layerField === 200) attributes.padding_amounts = [[0, 0], [0, 0]];
  else if (layerField === 210) Object.assign(attributes, { scaling_factor: [], fractional_scaling_factor: [] });
  else if (layerField === 211 || layerField === 212) attributes.target_size = [];
  else if (layerField === 280) Object.assign(attributes, { mode: 0, axis: 0 });
  else if (layerField === 920) attributes.reps = [];
  else if (AXIS_LAYER_FIELDS.has(layerField)) Object.assign(attributes, { axis: 0, k: 0, remove_dim: false });
  else if (layerField === 1150) attributes.target_shape = [];
  else if (layerField === 1155) Object.assign(attributes, { pad_amounts: [], pad_to_output_size: false });
  else if (REDUCE_ND_LAYER_FIELDS.has(layerField)) Object.assign(attributes, { axes: [], keep_dims: false, reduce_all: false });
  else if (layerField === 975) Object.assign(attributes, { axis: 0, num_splits: 0, split_sizes: [] });
  else if (layerField === 980) Object.assign(attributes, { axis: 0, interleave: false });
  else if (layerField === 1085 || layerField === 1105) attributes.target_shape = [];
  else if ([1175, 1195, 1215].includes(layerField)) attributes.output_shape = [];
  else if (layerField === 1120) Object.assign(attributes, { axes: [], squeeze_all: false });
  else if (layerField === 1125) attributes.axes = [];
  else if (layerField === 1130) attributes.axis = 0;
  else if (layerField === 1140) attributes.target_shape = [];
  else if (layerField === 400 || layerField === 410) Object.assign(attributes, {
    input_vector_size: null, output_vector_size: null, activation_count: 0,
    sequence_output: false, has_bias_vectors: false, reverse_input: false,
  });
  else if (layerField === 420 || layerField === 430) Object.assign(attributes, {
    input_vector_size: null, output_vector_size: null,
    activation_count: 0, backward_activation_count: 0,
    reverse_input: false, weight_set_count: 0,
    lstm_params: {
      sequence_output: false, has_bias_vectors: false, forget_bias: false,
      has_peephole_vectors: false, coupled_input_and_forget_gate: false, cell_clip_threshold: 0,
    },
  });
  else if (layerField === 1350) Object.assign(attributes, { normalized_shape: [], epsilon: 1e-5 });
  else if (layerField === 1471) Object.assign(attributes, {
    output_channels: null, input_channels: null, n_groups: 1,
    kernel_size: [null, null, null], stride: [1, 1, 1], dilation: [1, 1, 1],
    has_bias: false, padding_type: 0, custom_padding: [[0, 0], [0, 0], [0, 0]],
    is_deconvolution: false, output_shape: [],
  });
  else if (layerField === 1465) Object.assign(attributes, {
    pooling_type: 0, kernel_size: [null, null, null], stride: [null, null, null],
    padding_type: 0, custom_padding: [[0, 0], [0, 0], [0, 0]], exclude_padding: false,
  });
  else if (layerField === 1466) attributes.pooling_type = 0;
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (direct?.has(field)) {
      weights.push(parseWeightParams(reader.message(wire, `CoreML.${LAYER_TYPES[layerField]}.${direct.get(field)}`), direct.get(field)));
    } else if ((layerField === 400 || layerField === 410 || layerField === 420 || layerField === 430) && (field === 1 || field === 2)) {
      const name = field === 1 ? "input_vector_size" : "output_vector_size";
      attributes[name] = reader.int64Field(wire, singular, field, name);
    } else if (layerField === 400 && field === 10) {
      if (singular.has(field)) throw new Error("Core ML SIMPLE_RECURRENT repeats activation");
      singular.add(field);
      reader.bytesField(wire, "CoreML.SIMPLE_RECURRENT.activation");
      attributes.activation_count = 1;
    } else if ((layerField === 410 || layerField === 420 || layerField === 430) && (field === 10 || layerField === 430 && field === 11)) {
      reader.bytesField(wire, `CoreML.${LAYER_TYPES[layerField]}.activation`);
      if (field === 11) attributes.backward_activation_count += 1;
      else attributes.activation_count += 1;
      if (attributes.activation_count > 3 || attributes.backward_activation_count > 3) throw new Error(`Core ML ${LAYER_TYPES[layerField]} has too many activation parameters`);
    } else if ((layerField === 400 || layerField === 410) && [15, 20, 100].includes(field)) {
      const names = { 15: "sequence_output", 20: "has_bias_vectors", 100: "reverse_input" };
      attributes[names[field]] = readBoolField(reader, wire, singular, field, names[field]);
    } else if ((layerField === 400 || layerField === 410) && (SIMPLE_RECURRENT_WEIGHT_FIELDS.has(field) || GRU_WEIGHT_FIELDS.has(field))) {
      const roleMap = layerField === 400 ? SIMPLE_RECURRENT_WEIGHT_FIELDS : GRU_WEIGHT_FIELDS;
      if (!roleMap.has(field)) reader.skip(wire);
      else {
        const role = roleMap.get(field);
        weights.push(parseWeightParams(reader.message(wire, `CoreML.${LAYER_TYPES[layerField]}.${role}`), role));
      }
    } else if ((layerField === 420 || layerField === 430) && field === 15) {
      if (singular.has(field)) throw new Error(`Core ML ${LAYER_TYPES[layerField]} repeats LSTMParams`);
      singular.add(field);
      attributes.lstm_params = parseLstmParams(reader.message(wire, `CoreML.${LAYER_TYPES[layerField]}.params`));
    } else if ((layerField === 420 || layerField === 430) && field === 20) {
      if (layerField === 420 && attributes.weight_set_count) throw new Error("Core ML UNIDIRECTIONAL_LSTM repeats LSTMWeightParams");
      if (attributes.weight_set_count >= (layerField === 430 ? 2 : 1)) throw new Error(`Core ML ${LAYER_TYPES[layerField]} has too many LSTMWeightParams messages`);
      const prefix = layerField === 430 ? attributes.weight_set_count === 0 ? "forward_" : "backward_" : "";
      weights.push(...parseLstmWeightParams(reader.message(wire, `CoreML.${LAYER_TYPES[layerField]}.weightParams`), prefix));
      attributes.weight_set_count += 1;
    } else if (layerField === 420 && field === 100) {
      attributes.reverse_input = readBoolField(reader, wire, singular, field, "reverse_input");
    } else if (layerField === 160 && field === 1) {
      attributes.channels = reader.int64Field(wire, singular, field, "channels");
    } else if (layerField === 160 && (field === 5 || field === 6)) {
      attributes[field === 5 ? "compute_mean_var" : "instance_normalization"] = readBoolField(reader, wire, singular, field, field === 5 ? "computeMeanVar" : "instanceNormalization");
    } else if (layerField === 160 && field === 10) {
      const epsilon = reader.floatField(wire, singular, field, "epsilon");
      attributes.epsilon = epsilon === 0 ? 1e-5 : epsilon;
    } else if (layerField === 245 && (field === 1 || field === 4)) {
      appendRepeated(field === 1 ? attributes.shape_scale : attributes.shape_bias, repeatedUint64(reader, wire, `CoreML.SCALE.${field === 1 ? "shapeScale" : "shapeBias"}`), "Core ML scale shape", 3);
    } else if (layerField === 245 && field === 3) {
      attributes.has_bias = readBoolField(reader, wire, singular, field, "hasBias");
    } else if (layerField === 250 && field === 1) {
      appendRepeated(attributes.shape, repeatedUint64(reader, wire, "CoreML.BIAS.shape"), "Core ML bias shape", 3);
    } else if (layerField === 1350 && field === 1) {
      appendRepeated(attributes.normalized_shape, repeatedInt64(reader, wire, "CoreML.LAYER_NORMALIZATION.normalizedShape"), "Core ML normalizedShape");
    } else if (layerField === 1350 && field === 2) {
      attributes.epsilon = reader.floatField(wire, singular, field, "epsilon");
    } else if (layerField === 1471 && [1, 2, 10, 20, 21, 22, 31, 32, 33, 40, 41, 42, 50, 70, 80, 81, 82, 83, 84, 85, 86].includes(field)) {
      const value = reader.intField(wire, singular, field, `CoreML.CONVOLUTION_3D.${field}`);
      if ([50, 86].includes(field) && value > 1) throw new Error(`Core ML CONVOLUTION_3D field ${field} is not a valid bool`);
      if (field === 1) attributes.output_channels = value;
      else if (field === 2) attributes.input_channels = value;
      else if (field === 10) attributes.n_groups = value;
      else if (field >= 20 && field <= 22) attributes.kernel_size[field - 20] = value;
      else if (field >= 31 && field <= 33) attributes.stride[field - 31] = value;
      else if (field >= 40 && field <= 42) attributes.dilation[field - 40] = value;
      else if (field === 50) attributes.has_bias = value !== 0;
      else if (field === 70) attributes.padding_type = value;
      else if (field >= 80 && field <= 85) attributes.custom_padding[Math.floor((field - 80) / 2)][(field - 80) % 2] = value;
      else attributes.is_deconvolution = value !== 0;
    } else if (layerField === 1471 && field === 87) {
      appendRepeated(attributes.output_shape, repeatedUint64(reader, wire, "CoreML.CONVOLUTION_3D.outputShape"), "Core ML Conv3D outputShape", 3);
    } else if (layerField === 100 && [1, 2, 10, 60, 70].includes(field)) {
      const names = { 1: "output_channels", 2: "kernel_channels", 10: "n_groups", 60: "is_deconvolution", 70: "has_bias" };
      const value = reader.int64Field(wire, singular, field, names[field]);
      if (field >= 60 && value > 1) throw new Error(`Core ML CONVOLUTION ${names[field]} is not a valid bool`);
      attributes[names[field]] = field >= 60 ? value !== 0 : value;
    } else if (layerField === 100 && [20, 30, 40, 100].includes(field)) {
      const target = field === 20 ? attributes.kernel_size : field === 30 ? attributes.stride : field === 40 ? attributes.dilation : attributes.output_shape;
      appendRepeated(target, repeatedUint64(reader, wire, `CoreML.CONVOLUTION.${field}`), "Core ML convolution spatial values", 2);
    } else if (layerField === 100 && field === 50) setPadding(parseValidPadding(reader.message(wire, "CoreML.ValidPadding")));
    else if (layerField === 100 && field === 51) setPadding(parseSamePadding(reader.message(wire, "CoreML.SamePadding")));
    else if (layerField === 120 && [1, 50, 60].includes(field)) {
      const names = { 1: "pooling_type", 50: "exclude_padding", 60: "global_pooling" };
      const value = reader.intField(wire, singular, field, names[field]);
      if (field !== 1 && value > 1) throw new Error(`Core ML POOLING ${names[field]} is not a valid bool`);
      attributes[names[field]] = field === 1 ? value : value !== 0;
    } else if (layerField === 120 && [10, 20].includes(field)) {
      appendRepeated(field === 10 ? attributes.kernel_size : attributes.stride, repeatedUint64(reader, wire, `CoreML.POOLING.${field}`), "Core ML pooling spatial values", 2);
    } else if (layerField === 120 && field === 30) setPadding(parseValidPadding(reader.message(wire, "CoreML.ValidPadding")));
    else if (layerField === 120 && field === 31) setPadding(parseSamePadding(reader.message(wire, "CoreML.SamePadding")));
    else if (layerField === 120 && field === 32) {
      const complete = reader.message(wire, "CoreML.ValidCompletePadding");
      const amounts = [];
      while (!complete.done) {
        const item = complete.key();
        if (item.field === 10) appendRepeated(amounts, repeatedUint64(complete, item.wire, "CoreML.ValidCompletePadding.paddingAmounts"), "Core ML include-last-pixel padding", 2);
        else complete.skip(item.wire);
      }
      setPadding({ kind: "include_last_pixel", amounts: amounts.length ? amounts : [0, 0] });
    } else if (layerField === 140 && [1, 2, 10, 22].includes(field)) {
      const names = { 1: "input_channels", 2: "output_channels", 10: "has_bias", 22: "int8_dynamic_quantize" };
      const value = reader.int64Field(wire, singular, field, names[field]);
      if (field >= 10 && value > 1) throw new Error(`Core ML INNER_PRODUCT ${names[field]} is not a valid bool`);
      attributes[names[field]] = field >= 10 ? value !== 0 : value;
    } else if (layerField === 150 && [1, 2, 10].includes(field)) {
      const names = { 1: "input_dim", 2: "output_channels", 10: "has_bias" };
      const value = reader.int64Field(wire, singular, field, names[field]);
      if (field === 10 && value > 1) throw new Error(`Core ML EMBEDDING ${names[field]} is not a valid bool`);
      attributes[names[field]] = field === 10 ? value !== 0 : value;
    } else if (layerField === 1040 && [1, 2, 3].includes(field)) {
      const names = { 1: "vocab_size", 2: "embedding_size", 3: "has_bias" };
      const value = reader.int64Field(wire, singular, field, names[field]);
      if (field === 3 && value > 1) throw new Error(`Core ML EMBEDDING_ND ${names[field]} is not a valid bool`);
      attributes[names[field]] = field === 3 ? value !== 0 : value;
    } else if (layerField === 1045 && [1, 2, 5, 6, 7, 10].includes(field)) {
      const names = { 1: "transpose_a", 2: "transpose_b", 5: "weight_first_dimension", 6: "weight_second_dimension", 7: "has_bias", 10: "int8_dynamic_quantize" };
      const value = reader.int64Field(wire, singular, field, names[field]);
      if ([1, 2, 7, 10].includes(field) && value > 1) throw new Error(`Core ML BATCHED_MATMUL ${names[field]} is not a valid bool`);
      attributes[names[field]] = [1, 2, 7, 10].includes(field) ? value !== 0 : value;
    } else if ((layerField === 290 || layerField === 1070) && field === 1) {
      appendRepeated(attributes.shape, repeatedUint64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.shape`), "Core ML constant shape");
    } else if (layerField === 200 && field === 10) {
      if (singular.has(field)) throw new Error("Core ML PADDING repeats paddingAmounts");
      singular.add(field);
      attributes.padding_amounts = parseBorderAmounts(reader.message(wire, "CoreML.PADDING.paddingAmounts"));
    } else if (layerField === 210 && field === 1) {
      appendRepeated(attributes.scaling_factor, repeatedUint64(reader, wire, "CoreML.UPSAMPLE.scalingFactor"), "Core ML upsample scalingFactor", 2);
    } else if (layerField === 210 && field === 7) {
      const values = [];
      readFloatStats(reader, wire, "CoreML.UPSAMPLE.fractionalScalingFactor", true, false, null, values);
      appendRepeated(attributes.fractional_scaling_factor, values, "Core ML upsample fractionalScalingFactor", 2);
    } else if ((layerField === 211 || layerField === 212) && field === 1) {
      appendRepeated(attributes.target_size, repeatedUint64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.targetSize`), `Core ML ${LAYER_TYPES[layerField]} targetSize`, 2);
    } else if (layerField === 280 && (field === 1 || field === 3)) {
      attributes[field === 1 ? "mode" : "axis"] = reader.intField(wire, singular, field, field === 1 ? "mode" : "axis");
    } else if (layerField === 920 && field === 1) {
      appendRepeated(attributes.reps, repeatedUint64(reader, wire, "CoreML.TILE.reps"), "Core ML tile reps");
    } else if (AXIS_LAYER_FIELDS.has(layerField) && field === 1) {
      attributes.axis = reader.int64Field(wire, singular, field, "axis");
    } else if (layerField === 1015 && field === 2) attributes.k = reader.int64Field(wire, singular, field, "K");
    else if ((layerField === 1020 || layerField === 1025) && field === 2) attributes.remove_dim = readBoolField(reader, wire, singular, field, "removeDim");
    else if (layerField === 1150 && field === 1) {
      appendRepeated(attributes.target_shape, repeatedInt64(reader, wire, "CoreML.RANK_PRESERVING_RESHAPE.targetShape"), "Core ML rank-preserving reshape target");
    } else if (layerField === 1155 && field === 2) {
      appendRepeated(attributes.pad_amounts, repeatedUint64(reader, wire, "CoreML.CONSTANT_PADDING.padAmounts"), "Core ML constant padding amounts");
    } else if (layerField === 1155 && field === 3) attributes.pad_to_output_size = readBoolField(reader, wire, singular, field, "padToGivenOutputSizeMode");
    else if (REDUCE_ND_LAYER_FIELDS.has(layerField) && field === 1) {
      appendRepeated(attributes.axes, repeatedInt64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.axes`), `Core ML ${LAYER_TYPES[layerField]} axes`);
    } else if (REDUCE_ND_LAYER_FIELDS.has(layerField) && field === 2) attributes.keep_dims = readBoolField(reader, wire, singular, field, "keepDims");
    else if (REDUCE_ND_LAYER_FIELDS.has(layerField) && field === 3) attributes.reduce_all = readBoolField(reader, wire, singular, field, "reduceAll");
    else if (layerField === 300 && field === 1) {
      appendRepeated(attributes.target_shape, repeatedInt64(reader, wire, "CoreML.RESHAPE.targetShape"), "Core ML reshape target");
    } else if ((layerField === 300 && field === 2) || (layerField === 301 && field === 1)) attributes.mode = reader.intField(wire, singular, field, "mode");
    else if ((layerField === 310 || layerField === 985) && field === 1) {
      appendRepeated(attributes.axes, repeatedUint64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.axes`), `Core ML ${LAYER_TYPES[layerField]} axes`);
    } else if (layerField === 320 && field === 100) attributes.sequence_concat = readBoolField(reader, wire, singular, field, "sequenceConcat");
    else if (layerField === 330 && field === 1) attributes.n_outputs = reader.int64Field(wire, singular, field, "nOutputs");
    else if ((layerField === 975 || layerField === 980 || layerField === 1130) && field === 1) {
      attributes.axis = reader.int64Field(wire, singular, field, "axis");
    } else if (layerField === 975 && field === 2) attributes.num_splits = reader.int64Field(wire, singular, field, "numSplits");
    else if (layerField === 975 && field === 3) {
      appendRepeated(attributes.split_sizes, repeatedUint64(reader, wire, "CoreML.SPLIT_ND.splitSizes"), "Core ML split sizes");
    } else if (layerField === 980 && field === 2) attributes.interleave = readBoolField(reader, wire, singular, field, "interleave");
    else if ((layerField === 1120 || layerField === 1125) && field === 1) {
      appendRepeated(attributes.axes, repeatedInt64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.axes`), `Core ML ${LAYER_TYPES[layerField]} axes`);
    } else if (layerField === 1120 && field === 2) attributes.squeeze_all = readBoolField(reader, wire, singular, field, "squeezeAll");
    else if (layerField === 1140 && field === 1) {
      appendRepeated(attributes.target_shape, repeatedInt64(reader, wire, "CoreML.RESHAPE_STATIC.targetShape"), "Core ML static reshape target");
    } else if ((layerField === 1085 && field === 2) || (layerField === 1105 && field === 1)) {
      appendRepeated(attributes.target_shape, repeatedUint64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.targetShape`), `Core ML ${LAYER_TYPES[layerField]} target shape`);
    } else if ([1175, 1195, 1215].includes(layerField) && field === 4) {
      appendRepeated(attributes.output_shape, repeatedUint64(reader, wire, `CoreML.${LAYER_TYPES[layerField]}.outputShape`), `Core ML ${LAYER_TYPES[layerField]} output shape`);
    } else if (layerField === 1465 && field >= 1 && field <= 15) {
      const value = reader.intField(wire, singular, field, `CoreML.POOLING_3D.${field}`);
      if (field === 1) attributes.pooling_type = value;
      else if (field >= 2 && field <= 4) attributes.kernel_size[field - 2] = value;
      else if (field >= 5 && field <= 7) attributes.stride[field - 5] = value;
      else if (field >= 8 && field <= 13) attributes.custom_padding[Math.floor((field - 8) / 2)][(field - 8) % 2] = value;
      else if (field === 14) {
        if (value > 1) throw new Error("Core ML POOLING_3D countExcludePadding is not a valid bool");
        attributes.exclude_padding = value !== 0;
      } else attributes.padding_type = value;
    } else if (layerField === 1466 && field === 1) attributes.pooling_type = reader.intField(wire, singular, field, "type");
    else if (layerField === 130 && (field === 25 || field === 71)) {
      const activation = reader.message(wire, `CoreML.ACTIVATION.${field}`);
      const roles = field === 25 ? new Map([[1, "alpha"]]) : new Map([[1, "alpha"], [2, "beta"]]);
      while (!activation.done) {
        const item = activation.key();
        if (roles.has(item.field)) weights.push(parseWeightParams(activation.message(item.wire, `CoreML.ACTIVATION.${roles.get(item.field)}`), roles.get(item.field)));
        else activation.skip(item.wire);
      }
    } else reader.skip(wire);
  }
  finalizeLayerParameters(layerField, attributes, weights);
  const complete = !UNSCANNED_WEIGHT_CAPABLE.has(layerField);
  return { weights, attributes, weight_scan_status: complete ? "assessed" : "partial_layer_type_not_implemented" };
}

function parseTensorDescriptor(reader) {
  let rank = null;
  const dimensions = [];
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) rank = reader.intField(wire, singular, 1, "rank");
    else if (field === 2) {
      if (wire !== 0 && wire !== 2) throw new Error(`Core ML tensor dimension has unsupported wire type ${wire}`);
      const source = wire === 2 ? reader.message(wire, "CoreML.Tensor.dimValue") : reader;
      do {
        const value = BigInt.asIntN(64, source.rawVarint());
        if (value < -1n || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Core ML tensor dimension is outside the supported range");
        dimensions.push(Number(value));
      } while (wire === 2 && !source.done);
    } else reader.skip(wire);
  }
  if (rank == null || rank !== dimensions.length) throw new Error("Core ML tensor rank does not match dimValue cardinality");
  return dimensions;
}

export function parseCoreMlNeuralNetworkLayer(reader, index) {
  const layer = { index, name: "", inputs: [], outputs: [], input_shapes: [], output_shapes: [], type: null, type_field: null, weights: [], attributes: {}, weight_scan_status: "not_assessed" };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) layer.name = reader.stringField(wire, singular, 1, "name");
    else if (field === 2 || field === 3) pushBounded(field === 2 ? layer.inputs : layer.outputs, readString(reader, wire, `Core ML layer ${field === 2 ? "input" : "output"}`), "Core ML layer blobs");
    else if (field === 4 || field === 5) pushBounded(field === 4 ? layer.input_shapes : layer.output_shapes, parseTensorDescriptor(reader.message(wire, "CoreML.Tensor")), "Core ML layer tensor descriptors");
    else if (field >= 100) {
      if (layer.type_field != null) throw new Error("Core ML neural-network layer contains multiple layer oneof values");
      layer.type_field = field;
      layer.type = LAYER_TYPES[field] || `COREML_LAYER_FIELD_${field}`;
      const parsed = parseLayerParameters(field, reader.message(wire, `CoreML.${layer.type}`));
      layer.weights = parsed.weights;
      layer.attributes = parsed.attributes;
      layer.weight_scan_status = parsed.weight_scan_status;
    } else reader.skip(wire);
  }
  if (!layer.name || layer.type_field == null || !layer.outputs.length) throw new Error("Core ML neural-network layer is missing name, type, or outputs");
  if (layer.input_shapes.length && layer.input_shapes.length !== layer.inputs.length) throw new Error(`Core ML layer ${layer.name} input tensor descriptors do not match inputs`);
  if (layer.output_shapes.length && layer.output_shapes.length !== layer.outputs.length) throw new Error(`Core ML layer ${layer.name} output tensor descriptors do not match outputs`);
  return layer;
}

export function parseCoreMlNeuralNetworkPreprocessing(reader) {
  const result = { feature_name: "", kind: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.feature_name = reader.stringField(wire, singular, 1, "featureName");
    else if (field === 10 || field === 11) {
      if (result.kind) throw new Error("Core ML preprocessing contains multiple preprocessor oneof values");
      const payload = reader.message(wire, `CoreML.NeuralNetworkPreprocessing.${field}`);
      if (field === 10) {
        result.kind = "image_scaler";
        result.serialized_values = {
          channel_scale: null,
          blue_bias: null,
          green_bias: null,
          red_bias: null,
          gray_bias: null,
        };
        const fields = new Map([[10, "channel_scale"], [20, "blue_bias"], [21, "green_bias"], [22, "red_bias"], [30, "gray_bias"]]);
        const seen = new Set();
        while (!payload.done) {
          const item = payload.key();
          if (fields.has(item.field)) {
            if (seen.has(item.field)) throw new Error(`Core ML image scaler repeats field ${item.field}`);
            seen.add(item.field);
            const value = readFloatStats(payload, item.wire, `Core ML image scaler ${fields.get(item.field)}`);
            if (value.count !== 1) throw new Error("Core ML image scaler scalar field has invalid cardinality");
            result.serialized_values[fields.get(item.field)] = value.min;
          } else payload.skip(item.wire);
        }
        result.serialized_field_count = seen.size;
      } else {
        result.kind = "mean_image";
        let count = 0;
        let minimum = null;
        let maximum = null;
        while (!payload.done) {
          const item = payload.key();
          if (item.field === 1) {
            const values = readFloatStats(payload, item.wire, "Core ML mean image");
            count += values.count;
            if (values.count) {
              minimum = minimum == null ? values.min : Math.min(minimum, values.min);
              maximum = maximum == null ? values.max : Math.max(maximum, values.max);
            }
          } else payload.skip(item.wire);
        }
        if (!count) throw new Error("Core ML mean-image preprocessing contains no values");
        result.value_count = count;
        result.value_min = minimum;
        result.value_max = maximum;
        result.byte_length = count * 4;
      }
    } else reader.skip(wire);
  }
  if (!result.kind) throw new Error("Core ML preprocessing is missing its preprocessor type");
  result.feature_name = result.feature_name || null;
  result.feature_name_status = result.feature_name ? "serialized" : "missing_required_feature_name";
  return result;
}

export function finalizeCoreMlNeuralNetwork({
  layers = [], preprocessing = [], array_input_shape_mapping = 0, image_input_shape_mapping = 0,
}) {
  const names = new Set();
  const producers = new Set();
  for (const layer of layers) {
    if (names.has(layer.name)) throw new Error(`Core ML neural network repeats layer name ${layer.name}`);
    names.add(layer.name);
    for (const output of layer.outputs) {
      if (producers.has(output)) throw new Error(`Core ML neural network repeats output blob ${output}`);
      producers.add(output);
    }
  }
  const preprocessingFeatures = new Set();
  for (const item of preprocessing) {
    if (item.feature_name && preprocessingFeatures.has(item.feature_name)) throw new Error(`Core ML neural network repeats preprocessing for ${item.feature_name}`);
    if (item.feature_name) preprocessingFeatures.add(item.feature_name);
  }
  return { layers, preprocessing, preprocessing_count: preprocessing.length, array_input_shape_mapping, image_input_shape_mapping };
}

export function parseCoreMlNeuralNetwork(reader) {
  const state = { layers: [], preprocessing: [], array_input_shape_mapping: 0, image_input_shape_mapping: 0 };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) {
      if (state.layers.length >= MAX_LAYERS) throw new Error(`Core ML neural network exceeds ${MAX_LAYERS} layers`);
      state.layers.push(parseCoreMlNeuralNetworkLayer(reader.message(wire, "CoreML.NeuralNetworkLayer"), state.layers.length));
    } else if (field === 2) {
      pushBounded(state.preprocessing, parseCoreMlNeuralNetworkPreprocessing(reader.message(wire, "CoreML.NeuralNetworkPreprocessing")), "Core ML preprocessing entries");
    } else if (field === 5) state.array_input_shape_mapping = reader.intField(wire, singular, 5, "arrayInputShapeMapping");
    else if (field === 6) state.image_input_shape_mapping = reader.intField(wire, singular, 6, "imageInputShapeMapping");
    else reader.skip(wire);
  }
  return finalizeCoreMlNeuralNetwork(state);
}
