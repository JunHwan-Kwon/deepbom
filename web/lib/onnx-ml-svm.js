// Transport only. Contracts and bounded reference arithmetic share the WASM core.
import { onnxMlSemantic } from './onnx-portable.js';
export function inferOnnxMlSvmClassifier(context) { return onnxMlSemantic(context); }
export function inferOnnxMlSvmRegressor(context) { return onnxMlSemantic(context); }
