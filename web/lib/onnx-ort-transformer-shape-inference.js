import { onnxNodeSemantic } from "./onnx-portable.js";
export const ORT_TRANSFORMER_SHAPE_OPS = Object.freeze(new Set(["FastGelu","RotaryEmbedding","SkipLayerNormalization","SkipSimplifiedLayerNormalization","MultiHeadAttention","GroupQueryAttention"]));
export function canInferOnnxOrtTransformerNode(node, importedOpset) {return onnxNodeSemantic("ort_can",node,null,{family:"transformer",imported_opset:importedOpset});}
export function assessOnnxOrtTransformerSchemaForm(node, importedOpset) {return onnxNodeSemantic("ort_schema",node,null,{family:"transformer",imported_opset:importedOpset});}
export function inferOnnxOrtTransformerNode(node, tensors, tensorTypeName) {return onnxNodeSemantic("ort_infer",node,tensors,{family:"transformer",stash_dtype:tensorTypeName?.(Number.isSafeInteger(node.attributes?.get("stash_type")?.i)?node.attributes.get("stash_type").i:1)});}
