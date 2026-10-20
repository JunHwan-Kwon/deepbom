import {
  ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES,
  ONNX_SHAPE_SCHEMA_FORMS,
  ONNX_SHAPE_SCHEMA_SOURCE,
} from "./onnx-shape-schema-generated.js";

import { onnxNodeRecord, onnxAttributeRecord, onnxSemantic } from "./onnx-portable.js";
export { ONNX_SHAPE_SCHEMA_SOURCE };
export function resolveOnnxSchemaSinceVersion(opName, importedOpset) {
  return onnxSemantic("schema", {node:{opType:opName},imported_opset:importedOpset,forms:ONNX_SHAPE_SCHEMA_FORMS.get(opName)||[],attribute_names:ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES}).schema_since_version;
}
export function assessOnnxAttributeProto(attribute,{allowReference=false}={}) {
  return onnxSemantic("attribute",{value:onnxAttributeRecord(attribute),allow_reference:allowReference});
}
export function assessOnnxNodeSchemaForm(node,importedOpset) {
  return onnxSemantic("schema",{node:onnxNodeRecord(node),imported_opset:importedOpset,forms:ONNX_SHAPE_SCHEMA_FORMS.get(node?.opType)||[],attribute_names:ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES});
}
