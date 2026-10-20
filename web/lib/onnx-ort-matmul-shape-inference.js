import { onnxNodeSemantic } from "./onnx-portable.js";
export function canInferOnnxOrtMatMulNode(node, importedOpset) {return onnxNodeSemantic("ort_can",node,null,{family:"matmul",imported_opset:importedOpset});}
export function assessOnnxOrtMatMulSchemaForm(node, importedOpset) {return onnxNodeSemantic("ort_schema",node,null,{family:"matmul",imported_opset:importedOpset});}
export function inferOnnxOrtMatMulNode(node, tensors, tensorTypeName) {return onnxNodeSemantic("ort_infer",node,tensors,{family:"matmul",stash_dtype:tensorTypeName?.(Number.isSafeInteger(node.attributes?.get("stash_type")?.i)?node.attributes.get("stash_type").i:1)});}
