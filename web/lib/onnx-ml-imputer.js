import { onnxMlSemantic } from './onnx-portable.js';
export const IMPUTER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;
export function inferOnnxMlImputer(context) { return onnxMlSemantic(context); }
