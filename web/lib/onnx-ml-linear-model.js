// Transport only; source-pinned contracts and reference arithmetic live in WASM.
import { onnxMlSemantic } from './onnx-portable.js';
export function inferOnnxMlLinearClassifier(context) { return onnxMlSemantic(context); }
export function inferOnnxMlLinearRegressor(context) { return onnxMlSemantic(context); }
