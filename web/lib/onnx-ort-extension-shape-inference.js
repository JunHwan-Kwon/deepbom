import { onnxNodeSemantic } from "./onnx-portable.js";
const ORT_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
export const ONNX_ORT_EXTENSION_SHAPE_SOURCE = Object.freeze({
  release: "v1.23.0-source-pin",
  commit: ORT_COMMIT,
  documents: Object.freeze([
    Object.freeze({
      role: "ort_standard_domain_extension_schemas",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/contrib_defs.cc`,
      sha256: "e313a9ec5b8c11620445961c1f36da5ed894f70765ad94771e141de13b3e45ca",
    }),
    Object.freeze({
      role: "ort_transformer_contrib_schemas",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/contrib_ops/transformers/bert/bert_defs.cc`,
      sha256: "5783df1b2d56a6af0ba8a8b228ce165526fc3fcc5d162b00aa047709da621258",
    }),
    Object.freeze({
      role: "ort_transformer_shared_shape_functions",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/shape_inference_functions.cc`,
      sha256: "de6b66a75150dd390f52f9eaf5affe8a846ff5ff8b6641cf353d25e89fdd59db",
    }),
    Object.freeze({
      role: "ort_quantized_recurrent_contrib_schema",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_COMMIT}/onnxruntime/core/graph/contrib_ops/quantization_defs.cc`,
      sha256: "de215366d115e5b49fbc7ac0bbd09af19738cb05b9278469dd8a0a2537d28d6c",
    }),
  ]),
  interpretation_boundary: "These rules cover ORT contrib schemas registered in ai.onnx at the pinned ORT source commit. They do not promote those extensions to ONNX-standard operators or prove runtime kernel assignment.",
});

export function isOnnxOrtStandardDomainExtension(node, importedOpset) { return onnxNodeSemantic("ort_standard", node, null, {imported_opset:importedOpset}); }
export function canInferOnnxOrtExtensionNode(node, importedOpset) {return onnxNodeSemantic("ort_can",node,null,{family:"extension",imported_opset:importedOpset});}
export function assessOnnxOrtExtensionSchemaForm(node, importedOpset) {return onnxNodeSemantic("ort_schema",node,null,{family:"extension",imported_opset:importedOpset});}
export function inferOnnxOrtExtensionNode(node, tensors, tensorTypeName) {return onnxNodeSemantic("ort_infer",node,tensors,{family:"extension",stash_dtype:tensorTypeName?.(Number.isSafeInteger(node.attributes?.get("stash_type")?.i)?node.attributes.get("stash_type").i:1)});}
