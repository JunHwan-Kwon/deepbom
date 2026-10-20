// Transport only; versioned contracts and exact mapping live in WASM.
import { onnxMlSemantic, onnxSemantic } from './onnx-portable.js';
export const LABEL_ENCODER_MAX_PROPAGATED_STATIC_VALUES = 1_000_000;
export function resolveOnnxMlLabelEncoderVersion(importedOpset) {
 return onnxSemantic('ml_version',{node:{opType:'LabelEncoder'},imported_opset:importedOpset});
}
export function inferOnnxMlLabelEncoder(context) { return onnxMlSemantic(context); }
