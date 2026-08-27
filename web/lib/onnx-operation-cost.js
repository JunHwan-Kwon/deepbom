import { ONNX_SHAPE_SCHEMA_FORMS } from "./onnx-shape-schema-generated.js";

const IMPLEMENTED = new Set([
  "Attention", "Conv", "ConvInteger", "ConvTranspose", "DeformConv", "Einsum", "Gemm", "GRU", "LSTM", "MatMul", "MatMulInteger", "QLinearConv", "QLinearMatMul", "RNN",
]);
const UNIMPLEMENTED = new Set([]);
const ALGORITHM_DEPENDENT = new Set(["DFT", "STFT"]);

export function classifyOnnxMacOperation(opName, standardDomain = true) {
  if (!standardDomain) return "non_standard_domain";
  if (IMPLEMENTED.has(opName)) return "implemented";
  if (UNIMPLEMENTED.has(opName)) return "known_mac_bearing_unimplemented";
  if (ALGORITHM_DEPENDENT.has(opName)) return "algorithm_dependent_arithmetic";
  return ONNX_SHAPE_SCHEMA_FORMS.has(opName) ? "source_classified_non_mac" : "unclassified";
}

export function isOnnxMacBearingOperation(opName, standardDomain = true) {
  const value = classifyOnnxMacOperation(opName, standardDomain);
  return value === "implemented" || value === "known_mac_bearing_unimplemented";
}

export function isOnnxAlgorithmDependentArithmetic(opName, standardDomain = true) {
  return classifyOnnxMacOperation(opName, standardDomain) === "algorithm_dependent_arithmetic";
}
