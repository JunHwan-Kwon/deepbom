// Host transport and pinned source citations only; contracts and arithmetic run in WASM.
import { onnxNodeSemantic } from "./onnx-portable.js";
const ONNX_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const ORT_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";


export const ONNX_TFIDF_VECTORIZER_SOURCE = Object.freeze({
  schema: "deepbom.onnx_tfidf_vectorizer_source.v1",
  onnx_release: "v1.21.0",
  onnx_commit: ONNX_COMMIT,
  ort_release: "v1.26.0",
  ort_commit: ORT_COMMIT,
  documents: Object.freeze([
    source("onnx_schema_and_shape", "onnx/defs/nn/defs.cc", "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02", "onnx"),
    source("onnx_reference", "onnx/reference/ops/op_tfidf_vectorizer.py", "ac8e8495a50a0b85fb0f4adf5de2284efa2f5e14a999d4e1ac12e20c1079e69d", "onnx"),
    source("onnx_backend_tests", "onnx/backend/test/case/node/tfidfvectorizer.py", "9b8e1d00174a66727864baf40964c6f3da5d4e3a72858f0ee34589990b14e9dc", "onnx"),
    source("ort_cpu_kernel_header", "onnxruntime/core/providers/cpu/nn/tfidfvectorizer.h", "76308c5d7cf403eec02d9819fa919f24cd5a8567a42b8e4cf9695592d25f5645", "ort"),
    source("ort_cpu_kernel", "onnxruntime/core/providers/cpu/nn/tfidfvectorizer.cc", "8d3494b5d9344d49d97fae6a0ca2ed41cb97be2a5bfa26d598c34c96de7321b8", "ort"),
    source("ort_cpu_tests", "onnxruntime/test/providers/cpu/nn/tfidfvectorizer_test.cc", "21e16e4382809769a9cd8857b9a57c16a1e37348248378fc5f8cb28c5ae46687", "ort"),
  ]),
});

export function canInferOnnxTfIdfVectorizer(node) {
  return (!node?.domain || node.domain === "ai.onnx") && node?.opType === "TfIdfVectorizer";
}
export function inferOnnxTfIdfVectorizerNode({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  return onnxNodeSemantic("tfidf", node, tensorMap, {
    node_index: nodeIndex, node_name: node.name || "", imported_opset: importedOpset, scope,
  });
}
function source(role, path, sha256, repo) {
  const root = repo === "onnx" ? `https://raw.githubusercontent.com/onnx/onnx/${ONNX_COMMIT}` : `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}`;
  return Object.freeze({ role, source_ref: `${root}/${path}`, sha256 });
}
