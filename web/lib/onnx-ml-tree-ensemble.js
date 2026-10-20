// Transport only: shared WASM owns schema, topology and bounded scalar references.
import { onnxMlSemantic, onnxSemantic } from './onnx-portable.js';
export function resolveOnnxMlTreeEnsembleVersion(opName, importedOpset) {
  if (!["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(opName)) return null;
  return onnxSemantic('ml_version', { node: { opType: opName }, imported_opset: importedOpset });
}
export function inferOnnxMlTreeEnsemble(context) { return onnxMlSemantic(context); }
export function inferOnnxMlTreeEnsembleClassifier(context) { return onnxMlSemantic(context); }
export function inferOnnxMlTreeEnsembleRegressor(context) { return onnxMlSemantic(context); }
