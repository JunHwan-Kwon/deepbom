import { onnxMlSemantic } from './onnx-portable.js';
export const SCALER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;
export function inferOnnxMlScaler(context) { return onnxMlSemantic(context); }
