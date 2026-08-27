import { scanSerializedTensorPayloads, ggufDequantizationSource } from "../web/lib/tensor-numerical-integrity.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";

function expect(value, message) {
  if (!value) throw new Error(message);
}

function ggufBlock(dtype, bytes, elements, configure = () => {}) {
  const payload = new Uint8Array(bytes);
  configure(payload, new DataView(payload.buffer));
  return {
    payload,
    analysis: {
      format: "gguf",
      filename: `${dtype}.gguf`,
      file_size_bytes: bytes,
      gguf: { tensor_data_offset: 0, endianness: "little" },
      tensors: [{ index: 0, name: "weight", dtype, shape: [elements], data_offset: 0, byte_length: bytes, storage_status: "assessed" }],
    },
  };
}

const halfOne = (view, offset) => view.setUint16(offset, 0x3c00, true);
const endianUnits = (offset, count, width) => Array.from({ length: count }, (_, index) => [offset + index * width, width]);
const GGUF_BIG_ENDIAN_UNITS = Object.freeze({
  Q1_0: [[0, 2]], Q2_0: [[0, 2]], Q4_0: [[0, 2]], Q4_1: [[0, 2], [2, 2]],
  Q5_0: [[0, 2]], Q5_1: [[0, 2], [2, 2]],
  Q8_0: [[0, 2]], Q8_1: [[0, 2], [2, 2]],
  Q2_K: [[80, 2], [82, 2]],
  Q3_K: [[108, 2]],
  Q4_K: [[0, 2], [2, 2]], Q5_K: [[0, 2], [2, 2]], Q6_K: [[208, 2]],
  Q8_K: [[0, 4], ...endianUnits(260, 16, 2)],
  IQ2_XXS: [[0, 2], ...endianUnits(2, 32, 2)],
  IQ2_XS: [[0, 2], ...endianUnits(2, 32, 2)],
  IQ3_XXS: [[0, 2]],
  IQ1_S: [[0, 2], ...endianUnits(34, 8, 2)],
  IQ4_NL: [[0, 2]], IQ3_S: [[0, 2]], IQ2_S: [[0, 2]],
  IQ4_XS: [[0, 2], [2, 2]], IQ1_M: [],
  TQ1_0: [[52, 2]], TQ2_0: [[64, 2]], MXFP4: [], NVFP4: [],
});

function bigEndianFixture(fixture) {
  const payload = fixture.payload.slice();
  const dtype = fixture.analysis.tensors[0].dtype;
  for (const [offset, width] of GGUF_BIG_ENDIAN_UNITS[dtype] || []) {
    payload.subarray(offset, offset + width).reverse();
  }
  const analysis = structuredClone(fixture.analysis);
  analysis.filename = `${dtype}.big.gguf`;
  analysis.gguf.endianness = "big";
  return { payload, analysis };
}

function comparableRecord(record) {
  const copy = structuredClone(record);
  delete copy.payload_sha256;
  delete copy.serialized_endianness;
  return copy;
}
const zeroBlocks = [
  ggufBlock("Q4_0", 18, 32, (bytes, view) => { halfOne(view, 0); bytes.fill(0x88, 2); }),
  ggufBlock("Q4_1", 20, 32, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q5_1", 24, 32, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q8_0", 34, 32, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q8_1", 36, 32, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q2_K", 84, 256, (bytes, view) => { bytes.fill(1, 0, 16); halfOne(view, 80); }),
  ggufBlock("Q3_K", 110, 256, (bytes, view) => { bytes.fill(0xff, 0, 32); halfOne(view, 108); }),
  ggufBlock("Q4_K", 144, 256, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q5_K", 176, 256, (bytes, view) => { halfOne(view, 0); }),
  ggufBlock("Q6_K", 210, 256, (bytes, view) => { bytes.fill(0xaa, 128, 192); bytes.fill(1, 192, 208); halfOne(view, 208); }),
  ggufBlock("Q8_K", 292, 256, (bytes, view) => { view.setFloat32(0, 1, true); }),
];

for (const fixture of zeroBlocks) {
  const integrity = await scanSerializedTensorPayloads(fixture.payload, fixture.analysis, { chunkBytes: 17 });
  const row = integrity.tensor_records[0];
  expect(integrity.status === "assessed", `${row.dtype} full assessment`);
  expect(row.value_count === fixture.analysis.tensors[0].shape[0], `${row.dtype} decoded cardinality`);
  expect(row.all_zero && row.zero_value_count === row.value_count, `${row.dtype} all-zero source-pinned decode`);
  expect(row.nonfinite_scale_block_count === 0, `${row.dtype} finite block-scale classification`);
  expect(row.payload_sha256?.length === 64, `${row.dtype} payload digest`);
}

const q50 = ggufBlock("Q5_0", 22, 32, (bytes, view) => { halfOne(view, 0); });
const q50Record = (await scanSerializedTensorPayloads(q50.payload, q50.analysis)).tensor_records[0];
expect(q50Record.minimum_finite === -16 && q50Record.maximum_finite === -16, "Q5_0 signed code reconstruction");
expect(q50Record.quantization_code_levels_used === 1 && q50Record.quantization_code_levels_legal === 32, "Q5_0 code utilization");

const modernBlocks = [
  {
    fixture: ggufBlock("Q1_0", 18, 128, (bytes, view) => { halfOne(view, 0); bytes.fill(0xaa, 2); }),
    expected: { min: -1, max: 1, mean: 0, zeros: 0, codes: 2 },
  },
  {
    fixture: ggufBlock("Q2_0", 18, 64, (bytes, view) => { halfOne(view, 0); bytes.fill(0xe4, 2); }),
    expected: { min: -1, max: 2, mean: 0.5, zeros: 16, codes: 4 },
  },
  {
    fixture: ggufBlock("IQ2_XXS", 66, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: 1, max: 1, mean: 1, zeros: 0, codebook: ["iq2xxs_grid", 1, 256] },
  },
  {
    fixture: ggufBlock("IQ2_XS", 74, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: 1, max: 1, mean: 1, zeros: 0, codebook: ["iq2xs_grid", 1, 512] },
  },
  {
    fixture: ggufBlock("IQ3_XXS", 98, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: 1, max: 1, mean: 1, zeros: 0, codebook: ["iq3xxs_grid", 1, 256] },
  },
  {
    fixture: ggufBlock("IQ1_S", 50, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: -0.875, max: -0.875, mean: -0.875, zeros: 0, codebook: ["iq1s_grid", 1, 2048] },
  },
  {
    fixture: ggufBlock("IQ4_NL", 18, 32, (bytes, view) => { halfOne(view, 0); bytes.fill(0xf0, 2); }),
    expected: { min: -127, max: 113, mean: -7, zeros: 0, codes: 2 },
  },
  {
    fixture: ggufBlock("IQ3_S", 110, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: 1, max: 1, mean: 1, zeros: 0, codebook: ["iq3s_grid", 1, 512] },
  },
  {
    fixture: ggufBlock("IQ2_S", 82, 256, (bytes, view) => { halfOne(view, 0); }),
    expected: { min: 1, max: 1, mean: 1, zeros: 0, codebook: ["iq2s_grid", 1, 1024] },
  },
  {
    fixture: ggufBlock("IQ4_XS", 136, 256, (bytes, view) => {
      halfOne(view, 0);
      view.setUint16(2, 0xaaaa, true);
      bytes.fill(0x11, 4, 8);
      bytes.fill(0xf0, 8);
    }),
    expected: { min: -127, max: 113, mean: -7, zeros: 0, codes: 2 },
  },
  {
    fixture: ggufBlock("IQ1_M", 56, 256, (bytes, view) => {
      view.setUint16(52, 0xc000, true);
      view.setUint16(54, 0x3000, true);
    }),
    expected: { min: -0.875, max: -0.875, mean: -0.875, zeros: 0, codebook: ["iq1s_grid", 1, 2048] },
  },
  {
    fixture: ggufBlock("TQ1_0", 54, 256, (bytes, view) => { halfOne(view, 52); }),
    expected: { min: -1, max: -1, mean: -1, zeros: 0, codes: 1 },
  },
  {
    fixture: ggufBlock("TQ2_0", 66, 256, (bytes, view) => { bytes.fill(0xe4, 0, 64); halfOne(view, 64); }),
    expected: { min: -1, max: 2, mean: 0.5, zeros: 64, codes: 4 },
  },
  {
    fixture: ggufBlock("MXFP4", 17, 32, (bytes) => { bytes[0] = 128; bytes.fill(0xf0, 1); }),
    expected: { min: -12, max: 0, mean: -6, zeros: 16, codes: 2 },
  },
  {
    fixture: ggufBlock("NVFP4", 36, 64, (bytes) => { bytes.fill(64, 0, 4); bytes.fill(0xf0, 4); }),
    expected: { min: -12, max: 0, mean: -6, zeros: 32, codes: 2 },
  },
];

for (const { fixture, expected } of modernBlocks) {
  const row = (await scanSerializedTensorPayloads(fixture.payload, fixture.analysis, { chunkBytes: 17 })).tensor_records[0];
  expect(row.status === "assessed_full_payload", `${row.dtype} modern block assessment`);
  expect(row.value_count === fixture.analysis.tensors[0].shape[0], `${row.dtype} modern decoded cardinality`);
  expect(row.minimum_finite === expected.min && row.maximum_finite === expected.max, `${row.dtype} modern extrema`);
  expect(row.arithmetic_mean_finite === expected.mean, `${row.dtype} modern arithmetic mean`);
  expect(row.zero_value_count === expected.zeros, `${row.dtype} modern zero count`);
  if (expected.codebook) {
    const [name, used, legal] = expected.codebook;
    expect(row.encoded_codebook_name === name && row.encoded_codebook_entries_used === used
      && row.encoded_codebook_entries_legal === legal, `${row.dtype} modern codebook utilization`);
    expect(row.quantization_code_levels_used == null, `${row.dtype} codebook must not masquerade as scalar quantization levels`);
  } else expect(row.quantization_code_levels_used === expected.codes, `${row.dtype} modern code utilization`);
  expect(row.encoded_block_count === 1 && row.nonfinite_scale_block_count === 0, `${row.dtype} modern scale ledger`);
}

const endianSensitiveBlocks = [
  ggufBlock("Q5_0", 22, 32, (bytes, view) => {
    halfOne(view, 0); view.setUint32(2, 0x80000001, true); bytes.fill(0x21, 6);
  }),
  ggufBlock("Q3_K", 110, 256, (bytes, view) => {
    bytes.fill(0xff, 0, 32); bytes.fill(0x55, 32, 96);
    view.setUint32(96, 0x01234567, true); view.setUint32(100, 0x89abcdef, true);
    view.setUint32(104, 0x13579bdf, true); halfOne(view, 108);
  }),
  ggufBlock("IQ2_XXS", 66, 256, (bytes, view) => {
    halfOne(view, 0);
    for (let group = 0; group < 8; group += 1) {
      bytes.set([group, group + 1, group + 2, group + 3], 2 + group * 8);
      view.setUint32(6 + group * 8, ((group + 1) << 28) | 0x0123456, true);
    }
  }),
  ggufBlock("IQ2_XS", 74, 256, (bytes, view) => {
    halfOne(view, 0);
    for (let index = 0; index < 32; index += 1) view.setUint16(2 + index * 2, (index & 511) | ((index * 3 & 127) << 9), true);
    bytes.fill(0x43, 66);
  }),
  ggufBlock("IQ3_XXS", 98, 256, (bytes, view) => {
    halfOne(view, 0); bytes.fill(0x17, 2, 66);
    for (let group = 0; group < 8; group += 1) view.setUint32(66 + group * 4, ((group + 1) << 28) | 0x0123456, true);
  }),
  ggufBlock("IQ1_S", 50, 256, (bytes, view) => {
    halfOne(view, 0); bytes.fill(0x21, 2, 34);
    for (let group = 0; group < 8; group += 1) view.setUint16(34 + group * 2, 0x9000 | group * 0x49, true);
  }),
  ggufBlock("IQ4_XS", 136, 256, (bytes, view) => {
    halfOne(view, 0); view.setUint16(2, 0x1234, true); bytes.fill(0x65, 4, 8); bytes.fill(0xf0, 8);
  }),
  ggufBlock("Q8_K", 292, 256, (bytes, view) => {
    view.setFloat32(0, 0.5, true);
    for (let group = 0; group < 16; group += 1) {
      let sum = 0;
      for (let lane = 0; lane < 16; lane += 1) {
        const value = group - lane;
        bytes[4 + group * 16 + lane] = value & 255;
        sum += value;
      }
      view.setInt16(260 + group * 2, sum, true);
    }
  }),
];

for (const fixture of [...zeroBlocks, q50, ...modernBlocks.map(({ fixture }) => fixture), ...endianSensitiveBlocks]) {
  const little = (await scanSerializedTensorPayloads(fixture.payload, fixture.analysis, { chunkBytes: 17 })).tensor_records[0];
  const bigFixture = bigEndianFixture(fixture);
  const big = (await scanSerializedTensorPayloads(bigFixture.payload, bigFixture.analysis, { chunkBytes: 17 })).tensor_records[0];
  expect(JSON.stringify(comparableRecord(big)) === JSON.stringify(comparableRecord(little)), `${little.dtype} little/big-endian numerical equivalence`);
  expect(big.serialized_endianness === "big", `${little.dtype} serialized endian identity`);
  if (GGUF_BIG_ENDIAN_UNITS[little.dtype]?.length) {
    expect(big.payload_sha256 !== little.payload_sha256, `${little.dtype} raw-byte endian digest separation`);
  }
}

const invalidQ8K = endianSensitiveBlocks.at(-1);
invalidQ8K.payload[260] ^= 1;
const invalidQ8KRow = (await scanSerializedTensorPayloads(invalidQ8K.payload, invalidQ8K.analysis)).tensor_records[0];
expect(invalidQ8KRow.invalid_encoding_value_count === 1, "Q8_K redundant block sums are validated against serialized quants");

function scalarFixture(dtype, width, values, write, endianness) {
  const payload = new Uint8Array(width * values.length);
  const view = new DataView(payload.buffer);
  const littleEndian = endianness === "little";
  values.forEach((value, index) => write(view, index * width, value, littleEndian));
  return {
    payload,
    analysis: {
      format: "gguf", filename: `${dtype}.${endianness}.gguf`, file_size_bytes: payload.length,
      gguf: { tensor_data_offset: 0, endianness },
      tensors: [{ index: 0, name: "value", dtype, shape: [values.length], data_offset: 0, byte_length: payload.length, storage_status: "assessed" }],
    },
  };
}

const scalarCases = [
  ["F32", 4, [1, -2], (view, offset, value, little) => view.setFloat32(offset, value, little)],
  ["F16", 2, [0x3c00, 0xc000], (view, offset, value, little) => view.setUint16(offset, value, little)],
  ["I8", 1, [1, -2], (view, offset, value) => view.setInt8(offset, value)],
  ["I16", 2, [1, -2], (view, offset, value, little) => view.setInt16(offset, value, little)],
  ["I32", 4, [1, -2], (view, offset, value, little) => view.setInt32(offset, value, little)],
  ["I64", 8, [1n, -2n], (view, offset, value, little) => view.setBigInt64(offset, value, little)],
  ["F64", 8, [1, -2], (view, offset, value, little) => view.setFloat64(offset, value, little)],
  ["BF16", 2, [0x3f80, 0xc000], (view, offset, value, little) => view.setUint16(offset, value, little)],
];
for (const [dtype, width, values, write] of scalarCases) {
  const littleFixture = scalarFixture(dtype, width, values, write, "little");
  const bigFixture = scalarFixture(dtype, width, values, write, "big");
  const little = (await scanSerializedTensorPayloads(littleFixture.payload, littleFixture.analysis)).tensor_records[0];
  const big = (await scanSerializedTensorPayloads(bigFixture.payload, bigFixture.analysis)).tensor_records[0];
  expect(JSON.stringify(comparableRecord(big)) === JSON.stringify(comparableRecord(little)), `${dtype} scalar little/big-endian numerical equivalence`);
  if (width > 1) expect(big.payload_sha256 !== little.payload_sha256, `${dtype} scalar raw-byte endian digest separation`);
}

const nonfiniteQ1 = ggufBlock("Q1_0", 18, 128, (bytes, view) => { view.setUint16(0, 0x7c00, true); bytes.fill(0xff, 2); });
const nonfiniteQ1Record = (await scanSerializedTensorPayloads(nonfiniteQ1.payload, nonfiniteQ1.analysis)).tensor_records[0];
expect(nonfiniteQ1Record.positive_infinity_value_count === 128 && nonfiniteQ1Record.nonfinite_scale_block_count === 1, "Q1_0 non-finite scale propagation");

function safeFile(header, payload) {
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + encoded.length + payload.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(encoded.length), true);
  bytes.set(encoded, 8);
  bytes.set(payload, 8 + encoded.length);
  return bytes;
}

const safeBytes = safeFile({
  byte_values: { dtype: "U8", shape: [4], data_offsets: [0, 4] },
  half_values: { dtype: "F16", shape: [4], data_offsets: [4, 12] },
  wide_values: { dtype: "I64", shape: [2], data_offsets: [12, 28] },
}, Uint8Array.from([
  0, 1, 1, 255,
  0x00, 0x00, 0x00, 0x3c, 0x00, 0x7c, 0x01, 0x7c,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
]));
const safeAnalysis = parseMetadataModel(safeBytes, "integrity.safetensors", safeBytes.length, "safetensors");
const safeIntegrity = await scanSerializedTensorPayloads(safeBytes, safeAnalysis, { chunkBytes: 9 });
expect(safeIntegrity.status === "assessed" && safeIntegrity.decoded_value_count === 10, "SafeTensors full payload cardinality");
const safeRows = new Map(safeIntegrity.tensor_records.map((row) => [row.tensor_name, row]));
expect(safeRows.get("byte_values").zero_value_count === 1 && safeRows.get("byte_values").distinct_finite_values === 3, "SafeTensors U8 statistics");
expect(safeRows.get("half_values").positive_infinity_value_count === 1 && safeRows.get("half_values").nan_value_count === 1, "SafeTensors F16 non-finite classification");
expect(safeRows.get("wide_values").minimum_finite_decimal === "-9223372036854775808" && safeRows.get("wide_values").maximum_finite_decimal === "9223372036854775807", "SafeTensors I64 exact decimal extrema");
expect(safeRows.get("wide_values").moment_status.includes("avoid_binary64_rounding"), "SafeTensors I64 refuses lossy moments");
expect(safeRows.get("byte_values").decoded_values_status === "complete_binary_value_decoding"
  && JSON.stringify(safeRows.get("byte_values").decoded_values) === JSON.stringify([0, 1, 1, 255]), "Small scalar SafeTensors payloads retain complete decoded values for metadata contracts");
expect(safeRows.get("wide_values").decoded_values_status === "complete_exact_decimal_integer_decoding"
  && JSON.stringify(safeRows.get("wide_values").decoded_values) === JSON.stringify(["9223372036854775807", "-9223372036854775808"]), "Small 64-bit integer metadata remains exact decimal text");

async function scanSafeLowPrecision(dtype, payload, logicalValueCount) {
  const bytes = safeFile({ values: { dtype, shape: [logicalValueCount], data_offsets: [0, payload.length] } }, payload);
  const analysis = parseMetadataModel(bytes, `${dtype.toLowerCase()}.safetensors`, bytes.length, "safetensors");
  const integrity = await scanSerializedTensorPayloads(bytes, analysis, { chunkBytes: 17 });
  return { integrity, row: integrity.tensor_records[0] };
}

const f4Payload = Uint8Array.from({ length: 8 }, (_, index) => (index * 2) | ((index * 2 + 1) << 4));
const f4 = await scanSafeLowPrecision("F4", f4Payload, 16);
expect(f4.integrity.status === "assessed" && f4.row.value_count === 16, "SafeTensors F4 full code-space cardinality");
expect(f4.row.minimum_finite === -6 && f4.row.maximum_finite === 6 && f4.row.arithmetic_mean_finite === 0, "SafeTensors F4 exact extrema and mean");
expect(f4.row.zero_value_count === 2 && f4.row.negative_zero_value_count === 1 && f4.row.subnormal_value_count === 2, "SafeTensors F4 zero and subnormal classification");
expect(f4.row.quantization_code_levels_used === 16 && f4.row.quantization_code_levels_legal === 16, "SafeTensors F4 full code utilization");
expect(f4.row.decoder === "source_pinned_safetensors_pytorch_packed_float" && f4.row.value_count_conservation_status === "complete", "SafeTensors F4 source and logical-cardinality binding");

const allByteCodes = Uint8Array.from({ length: 256 }, (_, index) => index);
const fp8Expectations = [
  ["F8_E4M3", { nan: 2, inf: 0, zero: 2, negativeZero: 1, subnormal: 14, min: -448, max: 448 }],
  ["F8_E5M2", { nan: 6, inf: 2, zero: 2, negativeZero: 1, subnormal: 6, min: -57344, max: 57344 }],
  ["F8_E4M3FNUZ", { nan: 1, inf: 0, zero: 1, negativeZero: 0, subnormal: 14, min: -240, max: 240 }],
  ["F8_E5M2FNUZ", { nan: 1, inf: 0, zero: 1, negativeZero: 0, subnormal: 6, min: -57344, max: 57344 }],
];
for (const [dtype, expected] of fp8Expectations) {
  const { integrity, row } = await scanSafeLowPrecision(dtype, allByteCodes, 256);
  expect(integrity.status === "assessed" && row.value_count === 256, `${dtype} full code-space cardinality`);
  expect(row.nan_value_count === expected.nan, `${dtype} NaN classification`);
  expect(row.positive_infinity_value_count + row.negative_infinity_value_count === expected.inf, `${dtype} infinity classification`);
  expect(row.zero_value_count === expected.zero && row.negative_zero_value_count === expected.negativeZero, `${dtype} signed-zero classification`);
  expect(row.subnormal_value_count === expected.subnormal, `${dtype} subnormal classification`);
  expect(row.minimum_finite === expected.min && row.maximum_finite === expected.max, `${dtype} finite extrema`);
  expect(row.quantization_code_levels_used === 256 && row.quantization_code_levels_legal === 256, `${dtype} complete code utilization`);
  expect(row.decoder === "source_pinned_safetensors_pytorch_float8" && row.value_count_conservation_status === "complete", `${dtype} source and logical-cardinality binding`);
  expect(row.decoded_values_status === "not_retained_above_small_tensor_limit" && row.decoded_values.length === 0, `${dtype} large payload values are not duplicated in evidence`);
}

const e8m0 = await scanSafeLowPrecision("F8_E8M0", allByteCodes, 256);
expect(e8m0.integrity.status === "assessed" && e8m0.row.nan_value_count === 1, "SafeTensors E8M0 finite/NaN cardinality");
expect(e8m0.row.positive_infinity_value_count === 0 && e8m0.row.negative_infinity_value_count === 0 && e8m0.row.zero_value_count === 0, "SafeTensors E8M0 has no infinity or zero code");
expect(e8m0.row.minimum_finite === 2 ** -127 && e8m0.row.maximum_finite === 2 ** 127, "SafeTensors E8M0 exact exponent extrema");
expect(e8m0.row.quantization_code_levels_used === 256 && e8m0.row.value_count_conservation_status === "complete", "SafeTensors E8M0 full code utilization and cardinality");

for (const dtype of ["F6_E2M3", "F6_E3M2"]) {
  const f6 = await scanSafeLowPrecision(dtype, Uint8Array.of(0, 0, 0), 4);
  expect(f6.integrity.status === "not_assessed" && f6.integrity.unassessed_tensor_bytes === 3, `${dtype} packing ambiguity remains explicit`);
  expect(f6.integrity.byte_conservation_status === "complete" && /producer packing contract/.test(f6.row.reason), `${dtype} byte conservation and evidence boundary`);
}

expect(ggufDequantizationSource().source_sha256 === "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d", "GGUF decoder source digest pin");
expect(ggufDequantizationSource().numeric_format_source_sha256 === "2ed56e264202906d107e26d08eabb242d3107b026ebfb78096fa1e5f94bdbbb8", "GGUF numerical-format source digest pin");
expect(ggufDequantizationSource().format_specification_source_sha256 === "1dead27b6a522709f0127d194e58c21dbbbf00ba1c64fe37c54d1a9048b05020", "GGUF endian-format source digest pin");

console.log("Tensor numerical integrity passed (LE/BE scalar and GGML blocks, Q8_K redundant sums, 64-bit exactness, exhaustive SafeTensors F4/FP8 codes, explicit F6 boundary, non-finite classification, and byte conservation).");
