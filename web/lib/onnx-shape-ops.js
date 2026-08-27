import { ONNX_CONTAINER_VALUE_OPS } from "./onnx-container-inference.js";

export const ONNX_SAME_SHAPE_OPS = new Set([
  "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "BatchNormalization",
  "Ceil", "Celu", "Clip", "Cos", "Cosh", "Dropout", "Elu", "Erf", "Exp", "Floor",
  "Gelu", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization", "LeakyRelu", "Log",
  "LRN", "MeanVarianceNormalization", "Mish", "Neg", "Reciprocal", "Relu", "Round",
  "Selu", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh", "Softmax", "Softplus", "Softsign",
  "Sqrt", "Tan", "Tanh", "ThresholdedRelu", "Trilu",
]);

export const ONNX_BOOL_SAME_SHAPE_OPS = new Set(["IsInf", "IsNaN", "Not"]);

export const ONNX_BROADCAST_SAME_TYPE_OPS = new Set([
  "Add", "And", "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Div", "Max", "Mean",
  "Min", "Mod", "Mul", "Or", "Pow", "PRelu", "Sub", "Sum", "Xor",
]);

export const ONNX_BROADCAST_BOOL_OPS = new Set([
  "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
]);

export const ONNX_REDUCE_OPS = new Set([
  "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
  "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
]);

export const ONNX_SHAPE_INFERENCE_OPS = new Set([
  ...ONNX_SAME_SHAPE_OPS,
  ...ONNX_BOOL_SAME_SHAPE_OPS,
  ...ONNX_BROADCAST_SAME_TYPE_OPS,
  ...ONNX_BROADCAST_BOOL_OPS,
  ...ONNX_REDUCE_OPS,
  "ArgMax", "ArgMin", "Attention", "Cast", "CastLike", "Concat", "Constant", "ConstantOfShape", "Conv", "CumSum", "DeformConv", "Einsum",
  "ConvInteger", "ConvTranspose", "DequantizeLinear", "DepthToSpace", "DynamicQuantizeLinear", "Expand", "Flatten", "Gather", "GatherElements", "GatherND",
  "Gemm", "GlobalAveragePool", "GlobalMaxPool", "If", "Loop", "LSTM", "MatMul", "MatMulInteger", "MaxPool", "AveragePool",
  "LayerNormalization",
  "NonZero", "Pad", "QLinearConv", "QLinearMatMul", "QuantizeLinear", "RandomNormalLike", "RandomUniformLike", "Reshape", "Resize", "Shape", "Size",
  "Range", "Scan", "ScatterElements", "ScatterND", "Slice", "Split", "Squeeze", "STFT", "Tile", "TopK", "Transpose", "Unsqueeze", "Where",
  "SequenceMap", "TfIdfVectorizer",
  ...ONNX_CONTAINER_VALUE_OPS,
]);
