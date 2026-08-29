import { Sha256Accumulator } from "./sha256-sync.js";
import { ggufCodebookByte, ggufCodebookEntry } from "./gguf-codebooks.generated.js";

const DEFAULT_CHUNK_BYTES = 1024 * 1024;
const MAX_DISTINCT_VALUES = 4096;
export const GGUF_DEQUANTIZATION_SOURCE = Object.freeze({
  repository: "ggml-org/llama.cpp",
  source_commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
  source: "ggml/src/ggml-quants.c",
  source_sha256: "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d",
  block_layout_source: "ggml/src/ggml-common.h",
  block_layout_source_sha256: "af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0",
  numeric_format_source: "ggml/src/ggml-impl.h",
  numeric_format_source_sha256: "2ed56e264202906d107e26d08eabb242d3107b026ebfb78096fa1e5f94bdbbb8",
  format_specification_repository: "ggml-org/ggml",
  format_specification_commit: "6af560d55df03ad92116e3c0a697779584477e85",
  format_specification_source: "docs/gguf.md",
  format_specification_source_sha256: "1dead27b6a522709f0127d194e58c21dbbbf00ba1c64fe37c54d1a9048b05020",
});

export const SAFETENSORS_NUMERICAL_SOURCE = Object.freeze({
  repository: "huggingface/safetensors",
  commit: "6eb4dc9a28ebce297606e0f4836bbf28839cacef",
  tensor_source: "safetensors/src/tensor.rs",
  tensor_rs_sha256: "d7033872125a58f2fca600a30646f69f585e6699991556186eb144e8ff17ffc6",
  torch_binding_source: "bindings/python/py_src/safetensors/torch.py",
  torch_binding_source_sha256: "f3f476d1f8c04fe65fa3797426556a0b7afa43f8c4db9db6b799c7cf84748f3d",
  pytorch_repository: "pytorch/pytorch",
  pytorch_commit: "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
  pytorch_sources: Object.freeze([
    Object.freeze({ path: "torch/headeronly/util/Float4_e2m1fn_x2.h", sha256: "1bbdf8c345b8a0e7b7e4a113f536a9ef8e845b24deec6e926cc9a8347bf2b748" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_e4m3fn.h", sha256: "e4899a29cccf024866ae55318d1598e325e4fae16054c18665693109aa9528fc" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_e5m2.h", sha256: "602101d6f81fe6b478daa61073a5685f3e3ac1e701b657c697945b5d246b05e3" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_e4m3fnuz.h", sha256: "933c2e775119a8961c00ee23113dc53fea9f0780993892d1fd39af1add5a6ffc" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_e5m2fnuz.h", sha256: "f3b3ed913de17ede4179d83048bdfdfb8280e3f2913ff74c156b39d571349e27" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_e8m0fnu.h", sha256: "52d17d7786d2d9f9d350503f5071caed8de91b58adf7d5caa37662680ebb3ac1" }),
    Object.freeze({ path: "torch/headeronly/util/Float8_fnuz_cvt.h", sha256: "6952629a80a66607da8419a4451e345121b57927809e0e61ce17256536b94c33" }),
  ]),
  ocp_mx_specification: "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf",
  ocp_mx_version: "1.0",
  subbyte_boundary: "SafeTensors fixes bit cardinality, little-endian byte order, and C-order but does not define an F6 packed-bit mapping. F6 numerical decoding therefore requires a separately bound producer packing contract.",
});

export function safeTensorsNumericalSourceEvidence() {
  return {
    ...SAFETENSORS_NUMERICAL_SOURCE,
    pytorch_sources: SAFETENSORS_NUMERICAL_SOURCE.pytorch_sources.map((row) => ({ ...row })),
  };
}

const SAFE_SCALAR_LAYOUTS = Object.freeze({
  BOOL: { bytes: 1, kind: "bool" },
  U8: { bytes: 1, kind: "uint" }, I8: { bytes: 1, kind: "int" },
  U16: { bytes: 2, kind: "uint" }, I16: { bytes: 2, kind: "int" },
  U32: { bytes: 4, kind: "uint" }, I32: { bytes: 4, kind: "int" },
  U64: { bytes: 8, kind: "biguint" }, I64: { bytes: 8, kind: "bigint" },
  F16: { bytes: 2, kind: "f16" }, BF16: { bytes: 2, kind: "bf16" },
  F32: { bytes: 4, kind: "f32" }, F64: { bytes: 8, kind: "f64" },
  C64: { bytes: 8, kind: "c64" },
  F8_E4M3: { bytes: 1, kind: "f8_e4m3fn", levels: 256 },
  F8_E5M2: { bytes: 1, kind: "f8_e5m2", levels: 256 },
  F8_E8M0: { bytes: 1, kind: "f8_e8m0fnu", levels: 256 },
  F8_E4M3FNUZ: { bytes: 1, kind: "f8_e4m3fnuz", levels: 256 },
  F8_E5M2FNUZ: { bytes: 1, kind: "f8_e5m2fnuz", levels: 256 },
});

const SAFE_PACKED_LAYOUTS = Object.freeze({
  F4: { bytes: 1, elements: 2, kind: "f4_e2m1fn_x2", levels: 16 },
});

const GGUF_SCALAR_LAYOUTS = Object.freeze({
  F32: { bytes: 4, kind: "f32" }, F16: { bytes: 2, kind: "f16" },
  I8: { bytes: 1, kind: "int" }, I16: { bytes: 2, kind: "int" },
  I32: { bytes: 4, kind: "int" }, I64: { bytes: 8, kind: "bigint" },
  F64: { bytes: 8, kind: "f64" }, BF16: { bytes: 2, kind: "bf16" },
});

const GGUF_BLOCK_LAYOUTS = Object.freeze({
  Q1_0: { bytes: 18, elements: 128, levels: 2 },
  Q2_0: { bytes: 18, elements: 64, levels: 4 },
  Q4_0: { bytes: 18, elements: 32, levels: 16 },
  Q4_1: { bytes: 20, elements: 32, levels: 16 },
  Q5_0: { bytes: 22, elements: 32, levels: 32 },
  Q5_1: { bytes: 24, elements: 32, levels: 32 },
  Q8_0: { bytes: 34, elements: 32, levels: 256 },
  Q8_1: { bytes: 36, elements: 32, levels: 256 },
  Q2_K: { bytes: 84, elements: 256, levels: 4 },
  Q3_K: { bytes: 110, elements: 256, levels: 8 },
  Q4_K: { bytes: 144, elements: 256, levels: 16 },
  Q5_K: { bytes: 176, elements: 256, levels: 32 },
  Q6_K: { bytes: 210, elements: 256, levels: 64 },
  Q8_K: { bytes: 292, elements: 256, levels: 256 },
  IQ2_XXS: { bytes: 66, elements: 256, codebook: { name: "iq2xxs_grid", entries: 256 } },
  IQ2_XS: { bytes: 74, elements: 256, codebook: { name: "iq2xs_grid", entries: 512 } },
  IQ3_XXS: { bytes: 98, elements: 256, codebook: { name: "iq3xxs_grid", entries: 256 } },
  IQ1_S: { bytes: 50, elements: 256, codebook: { name: "iq1s_grid", entries: 2048 } },
  IQ4_NL: { bytes: 18, elements: 32, levels: 16 },
  IQ3_S: { bytes: 110, elements: 256, codebook: { name: "iq3s_grid", entries: 512 } },
  IQ2_S: { bytes: 82, elements: 256, codebook: { name: "iq2s_grid", entries: 1024 } },
  IQ4_XS: { bytes: 136, elements: 256, levels: 16 },
  IQ1_M: { bytes: 56, elements: 256, codebook: { name: "iq1s_grid", entries: 2048 } },
  TQ1_0: { bytes: 54, elements: 256, levels: 3 },
  TQ2_0: { bytes: 66, elements: 256, levels: 4 },
  MXFP4: { bytes: 17, elements: 32, levels: 16 },
  NVFP4: { bytes: 36, elements: 64, levels: 16 },
});

const IQ4_NL_VALUES = Object.freeze([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]);
const FP4_E2M1_DOUBLED_VALUES = Object.freeze([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]);

function float16(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = bits >>> 10 & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return fraction ? sign * fraction * 2 ** -24 : sign < 0 ? -0 : 0;
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

const FLOAT32_SCRATCH = new DataView(new ArrayBuffer(4));
function bfloat16(bits) {
  FLOAT32_SCRATCH.setUint32(0, bits << 16, false);
  return FLOAT32_SCRATCH.getFloat32(0, false);
}

function classifyFloatBits(kind, view, offset, littleEndian) {
  if (kind === "f16") {
    const bits = view.getUint16(offset, littleEndian);
    return { value: float16(bits), subnormal: (bits & 0x7c00) === 0 && (bits & 0x03ff) !== 0 };
  }
  if (kind === "bf16") {
    const bits = view.getUint16(offset, littleEndian);
    return { value: bfloat16(bits), subnormal: (bits & 0x7f80) === 0 && (bits & 0x007f) !== 0 };
  }
  if (kind === "f32") {
    const bits = view.getUint32(offset, littleEndian);
    return { value: view.getFloat32(offset, littleEndian), subnormal: (bits & 0x7f800000) === 0 && (bits & 0x007fffff) !== 0 };
  }
  const low = view.getUint32(offset + (littleEndian ? 0 : 4), littleEndian);
  const high = view.getUint32(offset + (littleEndian ? 4 : 0), littleEndian);
  return { value: view.getFloat64(offset, littleEndian), subnormal: (high & 0x7ff00000) === 0 && ((high & 0x000fffff) !== 0 || low !== 0) };
}

function decodeSignedMinifloat(code, {
  exponentBits,
  mantissaBits,
  bias,
  nanMagnitude = null,
  nanCode = null,
  ieeeSpecial = false,
}) {
  const signMask = 1 << (exponentBits + mantissaBits);
  const magnitudeMask = signMask - 1;
  const magnitude = code & magnitudeMask;
  if ((nanCode != null && code === nanCode) || (nanMagnitude != null && magnitude === nanMagnitude)) {
    return { value: Number.NaN, code };
  }
  const mantissaMask = (1 << mantissaBits) - 1;
  const exponentMask = (1 << exponentBits) - 1;
  const mantissa = magnitude & mantissaMask;
  const exponent = magnitude >>> mantissaBits & exponentMask;
  const sign = code & signMask ? -1 : 1;
  if (ieeeSpecial && exponent === exponentMask) {
    return { value: mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN, code };
  }
  if (exponent === 0) {
    if (mantissa === 0) return { value: sign < 0 ? -0 : 0, code };
    return {
      value: sign * 2 ** (1 - bias) * (mantissa / 2 ** mantissaBits),
      subnormal: true,
      code,
    };
  }
  return {
    value: sign * 2 ** (exponent - bias) * (1 + mantissa / 2 ** mantissaBits),
    code,
  };
}

function safeLowPrecisionValue(kind, code) {
  switch (kind) {
    case "f4_e2m1fn": return decodeSignedMinifloat(code, { exponentBits: 2, mantissaBits: 1, bias: 1 });
    case "f8_e4m3fn": return decodeSignedMinifloat(code, { exponentBits: 4, mantissaBits: 3, bias: 7, nanMagnitude: 0x7f });
    case "f8_e5m2": return decodeSignedMinifloat(code, { exponentBits: 5, mantissaBits: 2, bias: 15, ieeeSpecial: true });
    case "f8_e4m3fnuz": return decodeSignedMinifloat(code, { exponentBits: 4, mantissaBits: 3, bias: 8, nanCode: 0x80 });
    case "f8_e5m2fnuz": return decodeSignedMinifloat(code, { exponentBits: 5, mantissaBits: 2, bias: 16, nanCode: 0x80 });
    case "f8_e8m0fnu": return {
      value: code === 0xff ? Number.NaN : code === 0 ? 2 ** -127 : 2 ** (code - 127),
      code,
    };
    default: throw new Error(`SafeTensors low-precision decoder is not implemented for ${kind}`);
  }
}

class NumericAccumulator {
  constructor({ representation = "real", legalLevels = null, codebook = null, collectValues = false, semanticTracker = null } = {}) {
    this.representation = representation;
    this.legalLevels = legalLevels;
    this.count = 0;
    this.finiteCount = 0;
    this.nanCount = 0;
    this.positiveInfCount = 0;
    this.negativeInfCount = 0;
    this.zeroCount = 0;
    this.negativeZeroCount = 0;
    this.subnormalCount = 0;
    this.negativeCount = 0;
    this.minimum = Number.POSITIVE_INFINITY;
    this.maximum = Number.NEGATIVE_INFINITY;
    this.sum = 0;
    this.sumCompensation = 0;
    this.squareScale = 0;
    this.scaledSquareSum = 1;
    this.firstValue = undefined;
    this.constant = true;
    this.distinct = new Set();
    this.distinctOverflow = false;
    this.codes = legalLevels ? new Set() : null;
    this.codebook = codebook;
    this.codebookEntries = codebook ? new Set() : null;
    this.invalidEncodingCount = 0;
    this.blockCount = 0;
    this.nonfiniteScaleBlockCount = 0;
    this.collectValues = collectValues;
    this.decodedValues = [];
    this.semanticTracker = semanticTracker;
  }

  add(value, { subnormal = false, code = null, codebookEntry = null } = {}) {
    this.semanticTracker?.add(value, this.count);
    this.count += 1;
    if (this.collectValues) this.decodedValues.push(Number.isNaN(value) ? "NaN"
      : value === Number.POSITIVE_INFINITY ? "+Infinity"
        : value === Number.NEGATIVE_INFINITY ? "-Infinity"
          : Object.is(value, -0) ? "-0" : value);
    if (code != null && this.codes) this.codes.add(code);
    if (codebookEntry != null && this.codebookEntries) this.codebookEntries.add(codebookEntry);
    if (Number.isNaN(value)) { this.nanCount += 1; this.constant = false; return; }
    if (value === Number.POSITIVE_INFINITY) { this.positiveInfCount += 1; this.constant = false; return; }
    if (value === Number.NEGATIVE_INFINITY) { this.negativeInfCount += 1; this.constant = false; return; }
    this.finiteCount += 1;
    if (value === 0) {
      this.zeroCount += 1;
      if (Object.is(value, -0)) this.negativeZeroCount += 1;
    }
    if (value < 0) this.negativeCount += 1;
    if (subnormal) this.subnormalCount += 1;
    this.minimum = Math.min(this.minimum, value);
    this.maximum = Math.max(this.maximum, value);
    if (this.firstValue === undefined) this.firstValue = value;
    else if (!Object.is(this.firstValue, value)) this.constant = false;
    if (!this.distinctOverflow) {
      this.distinct.add(Object.is(value, -0) ? "-0" : value);
      if (this.distinct.size > MAX_DISTINCT_VALUES) { this.distinctOverflow = true; this.distinct.clear(); }
    }
    const adjusted = value - this.sumCompensation;
    const next = this.sum + adjusted;
    this.sumCompensation = next - this.sum - adjusted;
    this.sum = next;
    const absolute = Math.abs(value);
    if (absolute !== 0) {
      if (this.squareScale < absolute) {
        this.scaledSquareSum = 1 + this.scaledSquareSum * (this.squareScale / absolute) ** 2;
        this.squareScale = absolute;
      } else {
        this.scaledSquareSum += (absolute / this.squareScale) ** 2;
      }
    }
  }

  finish() {
    const allFinite = this.count === this.finiteCount;
    const l2Norm = this.squareScale === 0 ? 0 : this.squareScale * Math.sqrt(this.scaledSquareSum);
    return {
      representation: this.representation,
      value_count: this.count,
      finite_value_count: this.finiteCount,
      nan_value_count: this.nanCount,
      positive_infinity_value_count: this.positiveInfCount,
      negative_infinity_value_count: this.negativeInfCount,
      zero_value_count: this.zeroCount,
      negative_zero_value_count: this.negativeZeroCount,
      subnormal_value_count: this.subnormalCount,
      negative_value_count: this.negativeCount,
      minimum_finite: this.finiteCount ? this.minimum : null,
      maximum_finite: this.finiteCount ? this.maximum : null,
      arithmetic_mean_finite: this.finiteCount ? this.sum / this.finiteCount : null,
      l2_norm_finite: this.finiteCount ? l2Norm : null,
      rms_finite: this.finiteCount ? l2Norm / Math.sqrt(this.finiteCount) : null,
      exact_zero_fraction: this.count ? this.zeroCount / this.count : null,
      all_zero: this.count > 0 && allFinite && this.zeroCount === this.count,
      constant_finite: this.count > 0 && allFinite && this.constant,
      distinct_finite_values: this.distinctOverflow ? null : this.distinct.size,
      distinct_value_status: this.distinctOverflow ? `more_than_${MAX_DISTINCT_VALUES}` : "exact",
      invalid_encoding_value_count: this.invalidEncodingCount,
      quantization_code_levels_used: this.codes?.size ?? null,
      quantization_code_levels_legal: this.legalLevels,
      quantization_code_utilization: this.codes && this.legalLevels ? this.codes.size / this.legalLevels : null,
      encoded_codebook_name: this.codebook?.name || null,
      encoded_codebook_entries_used: this.codebookEntries?.size ?? null,
      encoded_codebook_entries_legal: this.codebook?.entries ?? null,
      encoded_codebook_utilization: this.codebookEntries && this.codebook?.entries ? this.codebookEntries.size / this.codebook.entries : null,
      encoded_block_count: this.blockCount || null,
      nonfinite_scale_block_count: this.blockCount ? this.nonfiniteScaleBlockCount : null,
      decoded_values_status: this.collectValues ? "complete_binary_value_decoding" : "not_retained_above_small_tensor_limit",
      decoded_values: this.collectValues ? this.decodedValues : [],
      ...(this.semanticTracker ? { quantization_metadata_integrity: this.semanticTracker.finish() } : {}),
      aggregation: "full_payload; compensated_sum; scaled_sum_of_squares; IEEE-754 round-to-nearest JS Number",
    };
  }
}

function safeTensorsQuantMetadataTracker(tensor, format) {
  if (format !== "safetensors") return null;
  const name = String(tensor?.name || "");
  const role = /(?:^|[._])(g_idx)$/.test(name) ? "group_index"
    : /(?:qzeros|zero[_\-.]?points?|zeros)(?:$|[._])/.test(name) ? "packed_zero_point"
      : /(?:scales?|k_scale|v_scale)(?:$|[._])/.test(name) ? "scale" : null;
  if (!role) return null;
  let integer = true;
  let nondecreasing = true;
  let previous = null;
  let transitions = 0;
  let nonpositive = 0;
  let nonfinite = 0;
  const packedProfiles = role === "packed_zero_point" && tensor.dtype === "I32"
    ? Object.fromEntries([2, 4, 8].map((bits) => [bits, new Array(2 ** bits).fill(0)])) : null;
  return {
    add(value) {
      const numeric = typeof value === "bigint" ? Number(value) : value;
      if (!Number.isFinite(numeric)) nonfinite += 1;
      if (role === "scale" && Number.isFinite(numeric) && numeric <= 0) nonpositive += 1;
      if (role === "group_index") {
        if (!Number.isSafeInteger(numeric)) integer = false;
        if (previous != null && numeric < previous) nondecreasing = false;
        if (previous != null && numeric !== previous) transitions += 1;
        previous = numeric;
      }
      if (packedProfiles && Number.isSafeInteger(numeric)) {
        const word = numeric >>> 0;
        for (const [bitsText, histogram] of Object.entries(packedProfiles)) {
          const bits = Number(bitsText);
          const mask = 2 ** bits - 1;
          for (let shift = 0; shift < 32; shift += bits) histogram[word >>> shift & mask] += 1;
        }
      }
    },
    finish() {
      return {
        schema: "deepbom.safetensors.quantization_metadata_integrity.v1",
        status: nonfinite || (role === "scale" && nonpositive) || (role === "group_index" && !integer) ? "fail" : "assessed_full_payload",
        role,
        nonfinite_value_count: nonfinite,
        nonpositive_scale_count: role === "scale" ? nonpositive : null,
        integer_value_contract: role === "group_index" ? integer : null,
        nondecreasing: role === "group_index" ? nondecreasing : null,
        transition_count: role === "group_index" ? transitions : null,
        packed_lane_profiles: packedProfiles ? Object.entries(packedProfiles).map(([bits, histogram]) => ({
          bits: Number(bits), lane_count: histogram.reduce((sum, count) => sum + count, 0),
          levels_used: histogram.filter((count) => count > 0).length, histogram,
        })) : [],
      };
    },
  };
}

class ExactIntegerAccumulator {
  constructor({ signed, collectValues = false }) {
    this.signed = signed;
    this.count = 0;
    this.zeroCount = 0;
    this.negativeCount = 0;
    this.minimum = null;
    this.maximum = null;
    this.first = null;
    this.constant = true;
    this.distinct = new Set();
    this.distinctOverflow = false;
    this.collectValues = collectValues;
    this.decodedValues = [];
  }

  add(value) {
    this.count += 1;
    if (this.collectValues) this.decodedValues.push(value.toString());
    if (value === 0n) this.zeroCount += 1;
    if (value < 0n) this.negativeCount += 1;
    if (this.minimum == null || value < this.minimum) this.minimum = value;
    if (this.maximum == null || value > this.maximum) this.maximum = value;
    if (this.first == null) this.first = value;
    else if (this.first !== value) this.constant = false;
    if (!this.distinctOverflow) {
      this.distinct.add(value.toString());
      if (this.distinct.size > MAX_DISTINCT_VALUES) { this.distinctOverflow = true; this.distinct.clear(); }
    }
  }

  finish() {
    return {
      representation: this.signed ? "signed_integer_64" : "unsigned_integer_64",
      value_count: this.count,
      finite_value_count: this.count,
      nan_value_count: 0,
      positive_infinity_value_count: 0,
      negative_infinity_value_count: 0,
      zero_value_count: this.zeroCount,
      negative_zero_value_count: 0,
      subnormal_value_count: 0,
      negative_value_count: this.negativeCount,
      minimum_finite_decimal: this.minimum?.toString() ?? null,
      maximum_finite_decimal: this.maximum?.toString() ?? null,
      arithmetic_mean_finite: null,
      l2_norm_finite: null,
      rms_finite: null,
      exact_zero_fraction: this.count ? this.zeroCount / this.count : null,
      all_zero: this.count > 0 && this.zeroCount === this.count,
      constant_finite: this.count > 0 && this.constant,
      distinct_finite_values: this.distinctOverflow ? null : this.distinct.size,
      distinct_value_status: this.distinctOverflow ? `more_than_${MAX_DISTINCT_VALUES}` : "exact",
      integer_extrema_status: "exact_decimal",
      decoded_values_status: this.collectValues ? "complete_exact_decimal_integer_decoding" : "not_retained_above_small_tensor_limit",
      decoded_values: this.collectValues ? this.decodedValues : [],
      moment_status: "not_assessed_to_avoid_binary64_rounding_of_64_bit_integers",
    };
  }
}

function scalarValue(view, offset, layout, littleEndian) {
  switch (layout.kind) {
    case "bool": {
      const value = view.getUint8(offset);
      return { value, invalid: value > 1 };
    }
    case "uint": return { value: layout.bytes === 1 ? view.getUint8(offset) : layout.bytes === 2 ? view.getUint16(offset, littleEndian) : view.getUint32(offset, littleEndian) };
    case "int": return { value: layout.bytes === 1 ? view.getInt8(offset) : layout.bytes === 2 ? view.getInt16(offset, littleEndian) : view.getInt32(offset, littleEndian) };
    case "biguint": return { value: view.getBigUint64(offset, littleEndian) };
    case "bigint": return { value: view.getBigInt64(offset, littleEndian) };
    case "f8_e4m3fn":
    case "f8_e5m2":
    case "f8_e8m0fnu":
    case "f8_e4m3fnuz":
    case "f8_e5m2fnuz": return safeLowPrecisionValue(layout.kind, view.getUint8(offset));
    default: return classifyFloatBits(layout.kind, view, offset, littleEndian);
  }
}

function decodeScalarChunk(bytes, layout, accumulator, littleEndian) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (layout.kind === "c64") {
    for (let offset = 0; offset < bytes.length; offset += 8) {
      const real = classifyFloatBits("f32", view, offset, littleEndian);
      const imaginary = classifyFloatBits("f32", view, offset + 4, littleEndian);
      accumulator.add(Math.hypot(real.value, imaginary.value), { subnormal: real.subnormal || imaginary.subnormal });
    }
    return;
  }
  for (let offset = 0; offset < bytes.length; offset += layout.bytes) {
    const decoded = scalarValue(view, offset, layout, littleEndian);
    if (decoded.invalid) accumulator.invalidEncodingCount += 1;
    accumulator.add(decoded.value, decoded);
  }
}

function decodeSafePackedChunk(bytes, layout, accumulator) {
  if (layout.kind !== "f4_e2m1fn_x2") throw new Error(`unsupported SafeTensors packed layout ${layout.kind}`);
  for (const packed of bytes) {
    const low = packed & 0x0f;
    const high = packed >>> 4;
    const decodedLow = safeLowPrecisionValue("f4_e2m1fn", low);
    const decodedHigh = safeLowPrecisionValue("f4_e2m1fn", high);
    accumulator.add(decodedLow.value, decodedLow);
    accumulator.add(decodedHigh.value, decodedHigh);
  }
}

function readHalf(view, offset, littleEndian) { return float16(view.getUint16(offset, littleEndian)); }
function signedByte(value) { return value & 0x80 ? value - 0x100 : value; }

function addBlockValue(accumulator, value, code) { accumulator.add(value, { code }); }
function addCodebookValue(accumulator, value, codebookEntry) { accumulator.add(value, { codebookEntry }); }
function recordBlockScales(accumulator, ...scales) {
  accumulator.blockCount += 1;
  if (scales.some((value) => !Number.isFinite(value))) accumulator.nonfiniteScaleBlockCount += 1;
}

function scaleMinK4(index, scales) {
  return index < 4
    ? [scales[index] & 63, scales[index + 4] & 63]
    : [(scales[index + 4] & 0x0f) | (scales[index - 4] >>> 6) << 4,
      (scales[index + 4] >>> 4) | (scales[index] >>> 6) << 4];
}

function decodeQ4(block, accumulator, affine, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const m = affine ? readHalf(view, 2, littleEndian) : 0;
  const qOffset = affine ? 4 : 2;
  recordBlockScales(accumulator, d, m);
  for (let index = 0; index < 16; index += 1) {
    const packed = block[qOffset + index];
    const low = packed & 15;
    const high = packed >>> 4;
    addBlockValue(accumulator, (affine ? low : low - 8) * d + m, low);
    addBlockValue(accumulator, (affine ? high : high - 8) * d + m, high);
  }
}

function decodeQ5(block, accumulator, affine, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const m = affine ? readHalf(view, 2, littleEndian) : 0;
  const hOffset = affine ? 4 : 2;
  const qOffset = affine ? 8 : 6;
  // qh is a four-byte packed array, not a serialized uint32 field.
  const highBits = view.getUint32(hOffset, true);
  recordBlockScales(accumulator, d, m);
  for (let index = 0; index < 16; index += 1) {
    const packed = block[qOffset + index];
    const low = (packed & 15) | ((highBits >>> index & 1) << 4);
    const high = (packed >>> 4) | ((highBits >>> (index + 16) & 1) << 4);
    addBlockValue(accumulator, (affine ? low : low - 16) * d + m, low);
    addBlockValue(accumulator, (affine ? high : high - 16) * d + m, high);
  }
}

function decodeQ8(block, accumulator, affine, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const secondary = affine ? readHalf(view, 2, littleEndian) : null;
  if (affine) recordBlockScales(accumulator, d, secondary);
  else recordBlockScales(accumulator, d);
  const qOffset = affine ? 4 : 2;
  for (let index = 0; index < 32; index += 1) {
    const q = signedByte(block[qOffset + index]);
    addBlockValue(accumulator, q * d, q + 128);
  }
}

function decodeQ2K(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 80, littleEndian);
  const dmin = readHalf(view, 82, littleEndian);
  recordBlockScales(accumulator, d, dmin);
  let qOffset = 16;
  let scaleIndex = 0;
  for (let group = 0; group < 2; group += 1) {
    let shift = 0;
    for (let lane = 0; lane < 4; lane += 1) {
      for (let half = 0; half < 2; half += 1) {
        const scale = block[scaleIndex++];
        const dl = d * (scale & 15);
        const ml = dmin * (scale >>> 4);
        for (let index = 0; index < 16; index += 1) {
          const code = block[qOffset + half * 16 + index] >>> shift & 3;
          addBlockValue(accumulator, dl * code - ml, code);
        }
      }
      shift += 2;
    }
    qOffset += 32;
  }
}

function decodeQ3K(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const auxBytes = new Uint8Array(16);
  auxBytes.set(block.subarray(96, 108));
  const aux = new DataView(auxBytes.buffer);
  const mask1 = 0x03030303;
  const mask2 = 0x0f0f0f0f;
  // scales is a byte-packed array. These 32-bit operations reproduce the
  // pinned byte-lane algebra and are intentionally independent of file endian.
  const a0 = aux.getUint32(0, true);
  const a1 = aux.getUint32(4, true);
  const tmp = aux.getUint32(8, true);
  aux.setUint32(8, ((a0 >>> 4 & mask2) | ((tmp >>> 4 & mask1) << 4)) >>> 0, true);
  aux.setUint32(12, ((a1 >>> 4 & mask2) | ((tmp >>> 6 & mask1) << 4)) >>> 0, true);
  aux.setUint32(0, ((a0 & mask2) | ((tmp & mask1) << 4)) >>> 0, true);
  aux.setUint32(4, ((a1 & mask2) | ((tmp >>> 2 & mask1) << 4)) >>> 0, true);
  const d = readHalf(view, 108, littleEndian);
  recordBlockScales(accumulator, d);
  let qOffset = 32;
  let scaleIndex = 0;
  let highMask = 1;
  for (let group = 0; group < 2; group += 1) {
    let shift = 0;
    for (let lane = 0; lane < 4; lane += 1) {
      for (let half = 0; half < 2; half += 1) {
        const dl = d * (signedByte(auxBytes[scaleIndex++]) - 32);
        for (let index = 0; index < 16; index += 1) {
          const low = block[qOffset + half * 16 + index] >>> shift & 3;
          const code = low - (block[half * 16 + index] & highMask ? 0 : 4);
          addBlockValue(accumulator, dl * code, code + 4);
        }
      }
      shift += 2;
      highMask <<= 1;
    }
    qOffset += 32;
  }
}

function decodeQ4K(block, accumulator, fiveBit, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const dmin = readHalf(view, 2, littleEndian);
  const scales = block.subarray(4, 16);
  const high = fiveBit ? block.subarray(16, 48) : null;
  let qOffset = fiveBit ? 48 : 16;
  let scaleIndex = 0;
  let highLowMask = 1;
  let highHighMask = 2;
  recordBlockScales(accumulator, d, dmin);
  for (let group = 0; group < 4; group += 1) {
    const [scaleLow, minLow] = scaleMinK4(scaleIndex, scales);
    const [scaleHigh, minHigh] = scaleMinK4(scaleIndex + 1, scales);
    const dLow = d * scaleLow;
    const dHigh = d * scaleHigh;
    const mLow = dmin * minLow;
    const mHigh = dmin * minHigh;
    for (let index = 0; index < 32; index += 1) {
      const low = (block[qOffset + index] & 15) + (fiveBit && high[index] & highLowMask ? 16 : 0);
      const upper = (block[qOffset + index] >>> 4) + (fiveBit && high[index] & highHighMask ? 16 : 0);
      addBlockValue(accumulator, dLow * low - mLow, low);
      addBlockValue(accumulator, dHigh * upper - mHigh, upper);
    }
    qOffset += 32;
    scaleIndex += 2;
    highLowMask <<= 2;
    highHighMask <<= 2;
  }
}

function decodeQ6K(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 208, littleEndian);
  recordBlockScales(accumulator, d);
  let qLow = 0;
  let qHigh = 128;
  let scales = 192;
  for (let group = 0; group < 2; group += 1) {
    for (let index = 0; index < 32; index += 1) {
      const scaleIndex = Math.floor(index / 16);
      const q1 = ((block[qLow + index] & 15) | ((block[qHigh + index] & 3) << 4)) - 32;
      const q2 = ((block[qLow + 32 + index] & 15) | ((block[qHigh + index] >>> 2 & 3) << 4)) - 32;
      const q3 = ((block[qLow + index] >>> 4) | ((block[qHigh + index] >>> 4 & 3) << 4)) - 32;
      const q4 = ((block[qLow + 32 + index] >>> 4) | ((block[qHigh + index] >>> 6 & 3) << 4)) - 32;
      addBlockValue(accumulator, d * signedByte(block[scales + scaleIndex]) * q1, q1 + 32);
      addBlockValue(accumulator, d * signedByte(block[scales + scaleIndex + 2]) * q2, q2 + 32);
      addBlockValue(accumulator, d * signedByte(block[scales + scaleIndex + 4]) * q3, q3 + 32);
      addBlockValue(accumulator, d * signedByte(block[scales + scaleIndex + 6]) * q4, q4 + 32);
    }
    qLow += 64;
    qHigh += 32;
    scales += 8;
  }
}

function decodeQ8K(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = view.getFloat32(0, littleEndian);
  recordBlockScales(accumulator, d);
  for (let index = 0; index < 256; index += 1) {
    const q = signedByte(block[4 + index]);
    addBlockValue(accumulator, d * q, q + 128);
  }
  for (let group = 0; group < 16; group += 1) {
    let sum = 0;
    for (let lane = 0; lane < 16; lane += 1) sum += signedByte(block[4 + group * 16 + lane]);
    if (view.getInt16(260 + group * 2, littleEndian) !== sum) accumulator.invalidEncodingCount += 1;
  }
}

function decodeQ1(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  recordBlockScales(accumulator, d);
  for (let index = 0; index < 128; index += 1) {
    const code = block[2 + Math.floor(index / 8)] >>> (index % 8) & 1;
    addBlockValue(accumulator, code ? d : -d, code);
  }
}

function decodeQ2(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  recordBlockScales(accumulator, d);
  for (let index = 0; index < 64; index += 1) {
    const code = block[2 + Math.floor(index / 4)] >>> ((index % 4) * 2) & 3;
    addBlockValue(accumulator, (code - 1) * d, code);
  }
}

function iqSignMask(index) { return ggufCodebookByte("ksigns_iq2xs", index); }
function iqSigned(value, mask, lane) { return mask & 1 << lane ? -value : value; }

function decodeIQ2XXS(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const base = 2 + group * 8;
    const words = Array.from({ length: 4 }, (_, index) => view.getUint16(base + index * 2, littleEndian));
    const gridIndices = [words[0] & 255, words[0] >>> 8, words[1] & 255, words[1] >>> 8];
    const packedHigh = (words[2] | words[3] << 16) >>> 0;
    const scale = d * (0.5 + (packedHigh >>> 28)) * 0.25;
    scales.push(scale);
    for (let vector = 0; vector < 4; vector += 1) {
      const gridIndex = gridIndices[vector];
      const grid = ggufCodebookEntry("iq2xxs_grid", gridIndex);
      const signs = iqSignMask(packedHigh >>> (7 * vector) & 127);
      for (let lane = 0; lane < 8; lane += 1) {
        addCodebookValue(accumulator, scale * iqSigned(grid[lane], signs, lane), gridIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ2XS(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const packedScale = block[66 + group];
    const groupScales = [d * (0.5 + (packedScale & 15)) * 0.25, d * (0.5 + (packedScale >>> 4)) * 0.25];
    scales.push(...groupScales);
    for (let vector = 0; vector < 4; vector += 1) {
      const encoded = view.getUint16(2 + 2 * (group * 4 + vector), littleEndian);
      const gridIndex = encoded & 511;
      const grid = ggufCodebookEntry("iq2xs_grid", gridIndex);
      const signs = iqSignMask(encoded >>> 9);
      const scale = groupScales[Math.floor(vector / 2)];
      for (let lane = 0; lane < 8; lane += 1) {
        addCodebookValue(accumulator, scale * iqSigned(grid[lane], signs, lane), gridIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ2S(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const packedScale = block[74 + group];
    const groupScales = [d * (0.5 + (packedScale & 15)) * 0.25, d * (0.5 + (packedScale >>> 4)) * 0.25];
    scales.push(...groupScales);
    const high = block[66 + group];
    for (let vector = 0; vector < 4; vector += 1) {
      const gridIndex = block[2 + group * 4 + vector] | (high << (8 - 2 * vector) & 0x300);
      const grid = ggufCodebookEntry("iq2s_grid", gridIndex);
      const signs = block[34 + group * 4 + vector];
      const scale = groupScales[Math.floor(vector / 2)];
      for (let lane = 0; lane < 8; lane += 1) {
        addCodebookValue(accumulator, scale * iqSigned(grid[lane], signs, lane), gridIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ3XXS(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    // scales_and_signs is a byte-packed array, not a uint32 field.
    const packed = view.getUint32(66 + group * 4, true);
    const scale = d * (0.5 + (packed >>> 28)) * 0.5;
    scales.push(scale);
    for (let vector = 0; vector < 4; vector += 1) {
      const signs = iqSignMask(packed >>> (7 * vector) & 127);
      const firstIndex = block[2 + group * 8 + 2 * vector];
      const secondIndex = block[3 + group * 8 + 2 * vector];
      const first = ggufCodebookEntry("iq3xxs_grid", firstIndex);
      const second = ggufCodebookEntry("iq3xxs_grid", secondIndex);
      for (let lane = 0; lane < 4; lane += 1) {
        addCodebookValue(accumulator, scale * iqSigned(first[lane], signs, lane), firstIndex);
        addCodebookValue(accumulator, scale * iqSigned(second[lane], signs, lane + 4), secondIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ3S(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const packedScale = block[106 + Math.floor(group / 2)];
    const nibble = group % 2 ? packedScale >>> 4 : packedScale & 15;
    const scale = d * (1 + 2 * nibble);
    scales.push(scale);
    const high = block[66 + group];
    for (let vector = 0; vector < 4; vector += 1) {
      const firstIndex = block[2 + group * 8 + 2 * vector] | (high << (8 - 2 * vector) & 256);
      const secondIndex = block[3 + group * 8 + 2 * vector] | (high << (7 - 2 * vector) & 256);
      const first = ggufCodebookEntry("iq3s_grid", firstIndex);
      const second = ggufCodebookEntry("iq3s_grid", secondIndex);
      const signs = block[74 + group * 4 + vector];
      for (let lane = 0; lane < 4; lane += 1) {
        addCodebookValue(accumulator, scale * iqSigned(first[lane], signs, lane), firstIndex);
        addCodebookValue(accumulator, scale * iqSigned(second[lane], signs, lane + 4), secondIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ1S(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const high = view.getUint16(34 + group * 2, littleEndian);
    const scale = d * (2 * (high >>> 12 & 7) + 1);
    const delta = high & 0x8000 ? -0.125 : 0.125;
    scales.push(scale);
    for (let vector = 0; vector < 4; vector += 1) {
      const gridIndex = block[2 + group * 4 + vector] | (high >>> (3 * vector) & 7) << 8;
      const grid = ggufCodebookEntry("iq1s_grid", gridIndex);
      for (let lane = 0; lane < 8; lane += 1) {
        addCodebookValue(accumulator, scale * (signedByte(grid[lane]) + delta), gridIndex);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ1M(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  // IQ1_M stores these lanes in a byte array; the packed lane order is fixed.
  const scaleWords = Array.from({ length: 4 }, (_, index) => view.getUint16(48 + index * 2, true));
  const scaleBits = scaleWords[0] >>> 12 | (scaleWords[1] >>> 8 & 0x00f0)
    | (scaleWords[2] >>> 4 & 0x0f00) | (scaleWords[3] & 0xf000);
  const d = float16(scaleBits);
  const scales = [d];
  for (let group = 0; group < 8; group += 1) {
    const word = scaleWords[Math.floor(group / 2)];
    const shift = 6 * (group % 2);
    const groupScales = [d * (2 * (word >>> shift & 7) + 1), d * (2 * (word >>> (shift + 3) & 7) + 1)];
    scales.push(...groupScales);
    const high0 = block[32 + group * 2];
    const high1 = block[33 + group * 2];
    const indices = [
      block[group * 4] | (high0 << 8 & 0x700),
      block[group * 4 + 1] | (high0 << 4 & 0x700),
      block[group * 4 + 2] | (high1 << 8 & 0x700),
      block[group * 4 + 3] | (high1 << 4 & 0x700),
    ];
    const deltas = [high0 & 0x08, high0 & 0x80, high1 & 0x08, high1 & 0x80].map((negative) => negative ? -0.125 : 0.125);
    for (let vector = 0; vector < 4; vector += 1) {
      const grid = ggufCodebookEntry("iq1s_grid", indices[vector]);
      const scale = groupScales[vector < 2 ? 0 : 1];
      for (let lane = 0; lane < 8; lane += 1) {
        addCodebookValue(accumulator, scale * (signedByte(grid[lane]) + deltas[vector]), indices[vector]);
      }
    }
  }
  recordBlockScales(accumulator, ...scales);
}

function decodeIQ4NL(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  recordBlockScales(accumulator, d);
  for (let index = 0; index < 16; index += 1) {
    const packed = block[2 + index];
    const low = packed & 15;
    const high = packed >>> 4;
    addBlockValue(accumulator, d * IQ4_NL_VALUES[low], low);
    addBlockValue(accumulator, d * IQ4_NL_VALUES[high], high);
  }
}

function decodeIQ4XS(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 0, littleEndian);
  const scalesHigh = view.getUint16(2, littleEndian);
  const scalesLow = 4;
  const quants = 8;
  const scales = [];
  for (let subBlock = 0; subBlock < 8; subBlock += 1) {
    const encodedScale = (block[scalesLow + Math.floor(subBlock / 2)] >>> (4 * (subBlock % 2)) & 15)
      | (scalesHigh >>> (2 * subBlock) & 3) << 4;
    const scale = d * (encodedScale - 32);
    scales.push(scale);
    for (let index = 0; index < 16; index += 1) {
      const packed = block[quants + subBlock * 16 + index];
      const low = packed & 15;
      const high = packed >>> 4;
      addBlockValue(accumulator, scale * IQ4_NL_VALUES[low], low);
      addBlockValue(accumulator, scale * IQ4_NL_VALUES[high], high);
    }
  }
  recordBlockScales(accumulator, d, ...scales);
}

function decodeTQ1(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 52, littleEndian);
  const powersOfThree = [1, 3, 9, 27, 81];
  recordBlockScales(accumulator, d);
  for (let group = 0; group < 32; group += 32) {
    for (let trit = 0; trit < 5; trit += 1) {
      for (let index = 0; index < 32; index += 1) {
        const wrapped = block[group + index] * powersOfThree[trit] & 255;
        const code = wrapped * 3 >>> 8;
        addBlockValue(accumulator, (code - 1) * d, code);
      }
    }
  }
  for (let group = 32; group < 48; group += 16) {
    for (let trit = 0; trit < 5; trit += 1) {
      for (let index = 0; index < 16; index += 1) {
        const wrapped = block[group + index] * powersOfThree[trit] & 255;
        const code = wrapped * 3 >>> 8;
        addBlockValue(accumulator, (code - 1) * d, code);
      }
    }
  }
  for (let trit = 0; trit < 4; trit += 1) {
    for (let index = 0; index < 4; index += 1) {
      const wrapped = block[48 + index] * powersOfThree[trit] & 255;
      const code = wrapped * 3 >>> 8;
      addBlockValue(accumulator, (code - 1) * d, code);
    }
  }
}

function decodeTQ2(block, accumulator, littleEndian) {
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const d = readHalf(view, 64, littleEndian);
  recordBlockScales(accumulator, d);
  for (let group = 0; group < 64; group += 32) {
    for (let plane = 0; plane < 4; plane += 1) {
      for (let index = 0; index < 32; index += 1) {
        const code = block[group + index] >>> (plane * 2) & 3;
        addBlockValue(accumulator, (code - 1) * d, code);
      }
    }
  }
}

function e8m0Half(value) {
  if (value < 2) return value === 0 ? 2 ** -128 : 2 ** -127;
  return 2 ** (value - 128);
}

function ue4m3Half(value) {
  if (value === 0 || value === 0x7f) return 0;
  const exponent = value >>> 3 & 15;
  const mantissa = value & 7;
  const raw = exponent === 0 ? mantissa * 2 ** -9 : (1 + mantissa / 8) * 2 ** (exponent - 7);
  return raw * 0.5;
}

function decodeMXFP4(block, accumulator) {
  const d = e8m0Half(block[0]);
  recordBlockScales(accumulator, d);
  for (let index = 0; index < 16; index += 1) {
    const packed = block[1 + index];
    const low = packed & 15;
    const high = packed >>> 4;
    addBlockValue(accumulator, FP4_E2M1_DOUBLED_VALUES[low] * d, low);
    addBlockValue(accumulator, FP4_E2M1_DOUBLED_VALUES[high] * d, high);
  }
}

function decodeNVFP4(block, accumulator) {
  const scales = Array.from(block.subarray(0, 4), ue4m3Half);
  recordBlockScales(accumulator, ...scales);
  for (let subBlock = 0; subBlock < 4; subBlock += 1) {
    const d = scales[subBlock];
    for (let index = 0; index < 8; index += 1) {
      const packed = block[4 + subBlock * 8 + index];
      const low = packed & 15;
      const high = packed >>> 4;
      addBlockValue(accumulator, FP4_E2M1_DOUBLED_VALUES[low] * d, low);
      addBlockValue(accumulator, FP4_E2M1_DOUBLED_VALUES[high] * d, high);
    }
  }
}

function decodeGgufBlock(block, dtype, accumulator, littleEndian) {
  switch (dtype) {
    case "Q1_0": return decodeQ1(block, accumulator, littleEndian);
    case "Q2_0": return decodeQ2(block, accumulator, littleEndian);
    case "Q4_0": return decodeQ4(block, accumulator, false, littleEndian);
    case "Q4_1": return decodeQ4(block, accumulator, true, littleEndian);
    case "Q5_0": return decodeQ5(block, accumulator, false, littleEndian);
    case "Q5_1": return decodeQ5(block, accumulator, true, littleEndian);
    case "Q8_0": return decodeQ8(block, accumulator, false, littleEndian);
    case "Q8_1": return decodeQ8(block, accumulator, true, littleEndian);
    case "Q2_K": return decodeQ2K(block, accumulator, littleEndian);
    case "Q3_K": return decodeQ3K(block, accumulator, littleEndian);
    case "Q4_K": return decodeQ4K(block, accumulator, false, littleEndian);
    case "Q5_K": return decodeQ4K(block, accumulator, true, littleEndian);
    case "Q6_K": return decodeQ6K(block, accumulator, littleEndian);
    case "Q8_K": return decodeQ8K(block, accumulator, littleEndian);
    case "IQ2_XXS": return decodeIQ2XXS(block, accumulator, littleEndian);
    case "IQ2_XS": return decodeIQ2XS(block, accumulator, littleEndian);
    case "IQ3_XXS": return decodeIQ3XXS(block, accumulator, littleEndian);
    case "IQ1_S": return decodeIQ1S(block, accumulator, littleEndian);
    case "IQ4_NL": return decodeIQ4NL(block, accumulator, littleEndian);
    case "IQ3_S": return decodeIQ3S(block, accumulator, littleEndian);
    case "IQ2_S": return decodeIQ2S(block, accumulator, littleEndian);
    case "IQ4_XS": return decodeIQ4XS(block, accumulator, littleEndian);
    case "IQ1_M": return decodeIQ1M(block, accumulator, littleEndian);
    case "TQ1_0": return decodeTQ1(block, accumulator, littleEndian);
    case "TQ2_0": return decodeTQ2(block, accumulator, littleEndian);
    case "MXFP4": return decodeMXFP4(block, accumulator);
    case "NVFP4": return decodeNVFP4(block, accumulator);
    default: throw new Error(`GGUF payload decoder is not implemented for ${dtype}`);
  }
}

async function readRange(source, start, end) {
  if (source instanceof Uint8Array) return source.subarray(start, end);
  if (source && typeof source.slice === "function") return new Uint8Array(await source.slice(start, end).arrayBuffer());
  throw new Error("tensor payload source must be a Uint8Array or Blob-compatible object");
}

function sourceLength(source) {
  if (source instanceof Uint8Array) return source.length;
  return Number(source?.size);
}

function tensorRange(analysis, tensor) {
  if (analysis.format === "gguf") {
    const start = Number(analysis.gguf?.tensor_data_offset) + Number(tensor.data_offset);
    return { start, end: start + Number(tensor.byte_length) };
  }
  const start = 8 + Number(analysis.safetensors?.header_byte_length) + Number(tensor.data_offset);
  return { start, end: start + Number(tensor.byte_length) };
}

function tensorDecoder(analysis, tensor) {
  if (analysis.format === "safetensors") {
    const layout = SAFE_SCALAR_LAYOUTS[tensor.dtype];
    if (layout) return { status: "assessed", kind: "scalar", layout, littleEndian: true, representation: layout.kind === "c64" ? "complex_magnitude" : "real" };
    const packed = SAFE_PACKED_LAYOUTS[tensor.dtype];
    if (packed) return { status: "assessed", kind: "safe_packed", layout: packed, representation: "real" };
    if (["F6_E2M3", "F6_E3M2"].includes(tensor.dtype)) {
      return {
        status: "not_assessed",
        reason: `SafeTensors ${tensor.dtype} defines six-bit element semantics and exact byte cardinality but not a canonical packed-bit mapping; a producer packing contract is required before bytes can be decoded`,
      };
    }
    return { status: "not_assessed", reason: `SafeTensors ${tensor.dtype} payload semantics are not source-bound in this analyzer` };
  }
  if (!["little", "big"].includes(analysis.gguf?.endianness)) return { status: "not_assessed", reason: "GGUF tensor payload endianness is not bound" };
  const littleEndian = analysis.gguf.endianness === "little";
  const scalar = GGUF_SCALAR_LAYOUTS[tensor.dtype];
  if (scalar) return { status: "assessed", kind: "scalar", layout: scalar, littleEndian, representation: "real" };
  const block = GGUF_BLOCK_LAYOUTS[tensor.dtype];
  if (block) return { status: "assessed", kind: "gguf_block", layout: block, littleEndian, representation: "dequantized_real" };
  return { status: "not_assessed", reason: `GGUF ${tensor.dtype} dequantization is not implemented from the pinned source` };
}

function expectedTensorValueCount(tensor) {
  const count = (tensor.shape || []).reduce((product, dimension) => product * BigInt(dimension), 1n);
  if (count > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`tensor ${tensor.name} value count exceeds the JavaScript exact integer range`);
  return Number(count);
}

async function scanTensor(source, analysis, tensor, decoder, { chunkBytes, onProgress, index, count }) {
  const range = tensorRange(analysis, tensor);
  if (!Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.end) || range.start < 0 || range.end < range.start || range.end > sourceLength(source)) {
    throw new Error(`tensor ${tensor.name} payload range ${range.start}:${range.end} exceeds source length ${sourceLength(source)}`);
  }
  const unitBytes = decoder.layout.bytes;
  if ((range.end - range.start) % unitBytes !== 0) throw new Error(`tensor ${tensor.name} byte length is not aligned to decoder unit ${unitBytes}`);
  const expectedValueCount = expectedTensorValueCount(tensor);
  const collectValues = decoder.kind === "scalar" && expectedValueCount <= 64;
  const exactInteger = decoder.kind === "scalar" && ["bigint", "biguint"].includes(decoder.layout.kind);
  const semanticTracker = safeTensorsQuantMetadataTracker(tensor, analysis.format);
  const accumulator = exactInteger
    ? new ExactIntegerAccumulator({ signed: decoder.layout.kind === "bigint", collectValues })
    : new NumericAccumulator({
      representation: decoder.representation,
      legalLevels: decoder.layout.levels || null,
      codebook: decoder.layout.codebook || null,
      collectValues,
      semanticTracker,
    });
  const digest = new Sha256Accumulator();
  const alignedChunk = Math.max(unitBytes, Math.floor(chunkBytes / unitBytes) * unitBytes);
  let cursor = range.start;
  while (cursor < range.end) {
    const end = Math.min(range.end, cursor + alignedChunk);
    const bytes = await readRange(source, cursor, end);
    if (bytes.length !== end - cursor) throw new Error(`tensor ${tensor.name} payload read was truncated`);
    digest.update(bytes);
    if (decoder.kind === "scalar") decodeScalarChunk(bytes, decoder.layout, accumulator, decoder.littleEndian);
    else if (decoder.kind === "safe_packed") decodeSafePackedChunk(bytes, decoder.layout, accumulator);
    else for (let offset = 0; offset < bytes.length; offset += unitBytes) decodeGgufBlock(bytes.subarray(offset, offset + unitBytes), tensor.dtype, accumulator, decoder.littleEndian);
    cursor = end;
    onProgress?.({ phase: "tensor_payload", index, count, tensor: tensor.name, bytes_read: cursor - range.start, tensor_bytes: range.end - range.start });
  }
  const metrics = accumulator.finish();
  if (metrics.value_count !== expectedValueCount) {
    throw new Error(`tensor ${tensor.name} decoder emitted ${metrics.value_count}/${expectedValueCount} logical values`);
  }
  const decoderName = decoder.kind === "gguf_block"
    ? "source_pinned_ggml_dequantization"
    : decoder.kind === "safe_packed"
      ? "source_pinned_safetensors_pytorch_packed_float"
      : analysis.format === "safetensors" && String(decoder.layout.kind).startsWith("f8_")
        ? "source_pinned_safetensors_pytorch_float8"
        : "declared_scalar_dtype";
  return {
    schema: "deepbom.tensor_numerical_integrity.tensor.v1",
    status: "assessed_full_payload",
    evidence_class: "OBSERVED/DERIVED",
    tensor_index: tensor.index,
    tensor_name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    byte_offset_absolute: range.start,
    byte_length: range.end - range.start,
    payload_sha256: digest.digestHex(),
    serialized_endianness: analysis.format === "gguf" ? analysis.gguf.endianness : "little",
    decoder: decoderName,
    expected_logical_value_count: expectedValueCount,
    value_count_conservation_status: "complete",
    ...metrics,
  };
}

function summary(records, analysis, sourceBytes) {
  const assessed = records.filter((record) => record.status === "assessed_full_payload");
  const unassessed = records.filter((record) => record.status !== "assessed_full_payload");
  const sum = (key) => assessed.reduce((total, record) => total + Number(record[key] || 0), 0);
  const assessedBytes = sum("byte_length");
  const storageCardinalityComplete = analysis.tensors.every((tensor) => Number.isSafeInteger(tensor.byte_length) && tensor.byte_length >= 0);
  const declaredBytes = analysis.tensors.reduce((total, tensor) => total + (Number.isSafeInteger(tensor.byte_length) ? tensor.byte_length : 0), 0);
  const unassessedKnownBytes = unassessed.reduce((total, record) => total + (Number.isSafeInteger(record.byte_length) ? record.byte_length : 0), 0);
  return {
    schema: "deepbom.tensor_numerical_integrity.v1",
    status: unassessed.length ? assessed.length ? "partial" : "not_assessed" : "assessed",
    evidence_class: "OBSERVED/DERIVED",
    scan_scope: "full_declared_tensor_payloads",
    tensor_count: records.length,
    assessed_tensor_count: assessed.length,
    unassessed_tensor_count: unassessed.length,
    declared_tensor_bytes: declaredBytes,
    assessed_tensor_bytes: assessedBytes,
    unassessed_tensor_bytes: storageCardinalityComplete ? declaredBytes - assessedBytes : null,
    byte_conservation_status: !storageCardinalityComplete
      ? "not_assessed_unknown_tensor_storage_cardinality"
      : assessedBytes + unassessedKnownBytes === declaredBytes ? "complete" : "invalid",
    source_file_bytes: sourceBytes,
    decoded_value_count: sum("value_count"),
    nonfinite_value_count: sum("nan_value_count") + sum("positive_infinity_value_count") + sum("negative_infinity_value_count"),
    exact_zero_value_count: sum("zero_value_count"),
    all_zero_tensor_count: assessed.filter((record) => record.all_zero).length,
    constant_tensor_count: assessed.filter((record) => record.constant_finite).length,
    invalid_encoding_value_count: sum("invalid_encoding_value_count"),
    nonfinite_scale_block_count: sum("nonfinite_scale_block_count"),
    tensor_records: records,
    decoder_source: analysis.format === "gguf" ? GGUF_DEQUANTIZATION_SOURCE : safeTensorsNumericalSourceEvidence(),
    limitations: unassessed.map((record) => ({ tensor_name: record.tensor_name, dtype: record.dtype, reason: record.reason })),
  };
}

export async function scanSerializedTensorPayloads(source, analysis, { chunkBytes = DEFAULT_CHUNK_BYTES, onProgress } = {}) {
  if (!analysis || !["gguf", "safetensors"].includes(analysis.format)) throw new Error("serialized tensor scan requires a GGUF or SafeTensors analysis");
  if (!Number.isSafeInteger(sourceLength(source)) || sourceLength(source) !== Number(analysis.file_size_bytes)) {
    throw new Error(`serialized tensor source length ${sourceLength(source)} does not match analysis file size ${analysis.file_size_bytes}`);
  }
  const records = [];
  for (let index = 0; index < analysis.tensors.length; index += 1) {
    const tensor = analysis.tensors[index];
    if (analysis.format === "gguf" && tensor.storage_status !== "assessed") {
      records.push({
        schema: "deepbom.tensor_numerical_integrity.tensor.v1",
        status: "not_assessed",
        evidence_class: "OBSERVED",
        tensor_index: tensor.index,
        tensor_name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        byte_length: Number.isSafeInteger(tensor.byte_length) ? tensor.byte_length : null,
        reason: `GGUF tensor storage cardinality is ${tensor.storage_status}; payload decoding is not attempted without a source-valid byte range`,
      });
      continue;
    }
    const decoder = tensorDecoder(analysis, tensor);
    if (decoder.status !== "assessed") {
      records.push({
        schema: "deepbom.tensor_numerical_integrity.tensor.v1",
        status: "not_assessed",
        evidence_class: "OBSERVED",
        tensor_index: tensor.index,
        tensor_name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        byte_length: Number(tensor.byte_length || 0),
        reason: decoder.reason,
      });
      continue;
    }
    records.push(await scanTensor(source, analysis, tensor, decoder, { chunkBytes, onProgress, index, count: analysis.tensors.length }));
  }
  return summary(records, analysis, sourceLength(source));
}

export function ggufDequantizationSource() { return GGUF_DEQUANTIZATION_SOURCE; }
