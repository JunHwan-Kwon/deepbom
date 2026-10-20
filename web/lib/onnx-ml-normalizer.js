import { onnxMlSemantic } from './onnx-portable.js';
export const NORMALIZER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;
export function inferOnnxMlNormalizer(context) { return onnxMlSemantic(context); }
