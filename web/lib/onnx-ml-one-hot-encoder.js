import { onnxMlSemantic } from './onnx-portable.js';
export const ONE_HOT_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;
export function inferOnnxMlOneHotEncoder(context) { return onnxMlSemantic(context); }
