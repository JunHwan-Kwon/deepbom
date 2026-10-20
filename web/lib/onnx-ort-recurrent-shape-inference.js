import { onnxNodeSemantic } from "./onnx-portable.js";
export function canInferOnnxOrtRecurrentNode(node, importedOpset) {return onnxNodeSemantic("ort_can",node,null,{family:"recurrent",imported_opset:importedOpset});}
export function assessOnnxOrtRecurrentSchemaForm(node, importedOpset) {return onnxNodeSemantic("ort_schema",node,null,{family:"recurrent",imported_opset:importedOpset});}
export function inferOnnxOrtRecurrentNode(node, tensors, tensorTypeName) {return onnxNodeSemantic("ort_infer",node,tensors,{family:"recurrent",stash_dtype:tensorTypeName?.(Number.isSafeInteger(node.attributes?.get("stash_type")?.i)?node.attributes.get("stash_type").i:1)});}
