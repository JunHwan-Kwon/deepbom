import { scalarDtypeBits } from "./tensor-size.js";

// ONNX TensorProto enum identity, pinned to the source declared in onnx.js.
export const ONNX_TENSOR_TYPES = Object.fromEntries(Object.entries({
  0: "UNDEFINED",
  1: "FLOAT32",
  2: "UINT8",
  3: "INT8",
  4: "UINT16",
  5: "INT16",
  6: "INT32",
  7: "INT64",
  8: "STRING",
  9: "BOOL",
  10: "FLOAT16",
  11: "FLOAT64",
  12: "UINT32",
  13: "UINT64",
  14: "COMPLEX64",
  15: "COMPLEX128",
  16: "BFLOAT16",
  17: "FLOAT8E4M3FN",
  18: "FLOAT8E4M3FNUZ",
  19: "FLOAT8E5M2",
  20: "FLOAT8E5M2FNUZ",
  21: "UINT4",
  22: "INT4",
  23: "FLOAT4E2M1",
  24: "FLOAT8E8M0",
  25: "UINT2",
  26: "INT2",
}).map(([id, name]) => [id, { name, bits: scalarDtypeBits(name) ?? 0 }]));
export const ONNX_TENSOR_TYPE_IDS = new Map(Object.entries(ONNX_TENSOR_TYPES).filter(([code]) => Number(code) > 0).map(([code, row]) => [row.name, Number(code)]));
