import { scalarDtypeBytes } from "./tensor-size.js";

// Native block layout is distinct from scalar width. Decoder availability is
// separately enumerated by tensor-numerical-integrity.js.
export const GGML_TYPE_TRAITS_SOURCE = Object.freeze({
  repository: "ggml-org/llama.cpp",
  source_commit: "7bd8282c37fcd9c4d7236106d664761a23318f18",
  type_traits_source: "ggml/src/ggml.c",
  type_traits_source_sha256: "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4",
  block_layout_source: "ggml/src/ggml-common.h",
  block_layout_source_sha256: "af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0",
});

export const GGML_TYPE_TRAITS = Object.freeze({
  0: { name: "F32", block_elements: 1, block_bytes: scalarDtypeBytes("F32") },
  1: { name: "F16", block_elements: 1, block_bytes: scalarDtypeBytes("F16") },
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
  24: { name: "I8", block_elements: 1, block_bytes: scalarDtypeBytes("I8") },
  25: { name: "I16", block_elements: 1, block_bytes: scalarDtypeBytes("I16") },
  26: { name: "I32", block_elements: 1, block_bytes: scalarDtypeBytes("I32") },
  27: { name: "I64", block_elements: 1, block_bytes: scalarDtypeBytes("I64") },
  28: { name: "F64", block_elements: 1, block_bytes: scalarDtypeBytes("F64") },
  29: { name: "IQ1_M", block_elements: 256, block_bytes: 56 },
  30: { name: "BF16", block_elements: 1, block_bytes: scalarDtypeBytes("BF16") },
  34: { name: "TQ1_0", block_elements: 256, block_bytes: 54 },
  35: { name: "TQ2_0", block_elements: 256, block_bytes: 66 },
  39: { name: "MXFP4", block_elements: 32, block_bytes: 17 },
  40: { name: "NVFP4", block_elements: 64, block_bytes: 36 },
  41: { name: "Q1_0", block_elements: 128, block_bytes: 18 },
  42: { name: "Q2_0", block_elements: 64, block_bytes: 18 },
});

export const GGML_TYPES_BY_NAME = Object.freeze(Object.fromEntries(Object.values(GGML_TYPE_TRAITS).map(row => [row.name, Object.freeze(row)])));
