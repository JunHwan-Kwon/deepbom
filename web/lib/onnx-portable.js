// Transport only. Preserve parsed descriptors and bounded attribute tensor payloads; no full model buffer.
import { canonicalFloatText } from './onnx-static-value-evidence.js';
import { portableContract } from './portable-contracts.js';
import { ONNX_TENSOR_TYPE_IDS } from './onnx-tensor-types.js';
const TYPE_IDS=Object.fromEntries(ONNX_TENSOR_TYPE_IDS);
export function onnxAttributeRecord(value) {
  const record={};
  for(const key of ['type','refAttrName','valueTypesPresent','typeProto','i','iExactDecimal','f','s','ints','intExactDecimals','floats','strings']) {
    if(value?.[key] !== undefined) record[key]=value[key];
  }
  if(typeof value?.f === 'number')record.fText=canonicalFloatText(value.f);
  if(Array.isArray(value?.floats))record.floatsText=value.floats.map(canonicalFloatText);
  if(value?.tensor){
    record.tensor=onnxTensorRecord(value.tensor);
    for(const key of ['externalDataEntries','dataLocation','stringValues'])if(value.tensor[key]!==undefined)record.tensor[key]=value.tensor[key];
    if(value.tensor.rawData instanceof Uint8Array)record.tensor.rawData=Array.from(value.tensor.rawData);
    if(Array.isArray(value.tensor.typedValues))record.tensor.typedValues=value.tensor.typedValues.map(canonicalFloatText);
  }
  if(value?.graph)record.graph={};
  for(const key of ['tensors','graphs']) if(value?.[key]?.length)record[key]=Array.from({length:value[key].length},()=>({}));
  return record;
}
export function onnxNodeRecord(node) {
  return {domain:node?.domain,opType:node?.opType,inputs:node?.inputs||[],outputs:node?.outputs||[],duplicateAttributeNames:node?.duplicateAttributeNames||[],attributes:[...(node?.attributes||[])].map(([name,value])=>[name,onnxAttributeRecord(value)])};
}
export function onnxTensorRecord(tensor) {
  if(!tensor)return null;
  const record={};
  for(const key of ['name','role','mapKeysExact','mapKeyCount','mapKeyType','mapDuplicateKeyCount','initializerIntegerValuesComplete','initializerIntegerValues','contractStatus','contractConflict','conditionalShapeContract','conditionalShapeVariants','value_kind','shape_declared','dtype','shape','shapeDeclared','valueKind','typeProto','type_proto','initializerIntegerValuesExactComplete','initializerIntegerValuesExactDecimals','staticValues','staticValuesComplete','staticExactDecimals','staticDimensionValues','staticDimensionValuesComplete','staticDimensionValuesStatus','staticDimensionValuesSource','runtimeDimensionBounds','staticValuesCanonicalTextComplete','staticValuesCanonicalTexts','staticValuesStatus','staticValuesSource','sequenceLength','sequenceLengthStatus','sequenceElementTypes','sequenceElementInventoryStatus','optionalPresence','optionalPresenceStatus'])if(tensor[key]!==undefined)record[key]=tensor[key];
  if(Array.isArray(tensor.staticValues))record.staticValuesText=tensor.staticValues.map(v=>canonicalFloatText(typeof v==='boolean'?Number(v):v));
  return record;
}
export function onnxSemantic(operation,input) {return portableContract('onnx_semantics',{operation,...input,tensor_type_ids:TYPE_IDS});}
export function onnxNodeSemantic(operation,node,tensors,extra={}) {
  return onnxSemantic(operation,{node:onnxNodeRecord(node),tensors:Object.fromEntries((node?.inputs||[]).map(name=>[name,onnxTensorRecord(tensors?.get(name))])),...extra});
}
export function onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope='main_graph'}) {
  return onnxNodeSemantic('ml',node,tensorMap,{node_index:nodeIndex,imported_opset:importedOpset,scope});
}
