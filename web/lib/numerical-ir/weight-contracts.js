import { BoundedFlatBufferReader } from "../flatbuffer-reader.js";
import { readBytes, sourceSize, MAX_IN_MEMORY_SOURCE } from "./weight-sources.js";
import { notAssessed } from "./weight-math.js";
import { numericalWasm } from "../numerical-wasm.js";
import { canonicalJson } from "../report-utils.js";

export async function quantizationContracts(source, analysis) {
  const result = new Map();
  if (analysis.format === "onnx") {
    for (const tensor of analysis.tensors || []) {
      const bindings = (analysis.onnx_quantization_binding?.bindings || []).filter(b=>b.tensor_name===tensor.name && b.status==="pass");
      const contracts = bindings.map(b=>({scales:b.scale_values,zero_points:b.zero_point_values,axis:b.axis,parameterization:b.parameterization,block_size:b.block_size,scale_shape:b.scale_tensor_shape}));
      if (contracts.length && contracts.every(c=>canonicalJson(c)===canonicalJson(contracts[0])) && tensor.quantization_binding_status==="pass") result.set(tensor.name,contracts[0]);
    }
  } else if (analysis.format === "tflite" && sourceSize(source)<=MAX_IN_MEMORY_SOURCE) {
    const fb = new BoundedFlatBufferReader(await readBytes(source));
    for (const [sg,graph] of fb.tableVector(fb.root("TFL3"),2).entries()) {
      for (const [ti,tensor] of fb.tableVector(graph,0).entries()) {
        const q=fb.tableField(tensor,4);if(!q)continue;
        const scales=fb.scalarVector(q,2,"f32"),zero_points=fb.scalarVector(q,3,"i64").map(Number);
        if(!scales.length)continue;
        result.set("tflite:"+sg+":"+ti,{scales,zero_points,axis:fb.scalar(q,6,"i32"),parameterization:scales.length===1?"per_tensor":"per_axis",block_size:null,scale_shape:[scales.length]});
      }
    }
  }
  return result;
}
export function tensorQuantization(row, store, analysis, contracts) {
  if(analysis.format==="onnx")return store && row.native_locator.startsWith("onnx:main_graph:graph_initializer:") ? contracts.get(row.name) || null : null;
  if(analysis.format==="tflite") {
    const path=store?.native_source?.path || "",match=/subgraphs\[(\d+)\].*tensors\[(\d+)\]/.exec(path);
    if(match)return contracts.get("tflite:"+match[1]+":"+match[2]) || null;
    const index=/^storage:tensor:(\d+)$/.exec(store?.native_subject_ref || "");
    return index?contracts.get("tflite:0:"+index[1]) || null:null;
  }
  return null;
}
export function axisContract(row, model, override = {}) {
  const shape=row.shape,format=model.artifact.format.toLowerCase();
  const order=format==="gguf"?"first_axis_fastest":"last_axis_fastest";
  let axis=format==="gguf"?Math.max(0,shape.length-1):0,meaning="generic_axis_slices",basis="declared_shape_only";
  const operations=new Map(model.program.operations.map(op=>[op.id,op]));
  const bindings=model.weight_bindings.bindings.filter(b=>b.storage_ref===row.storage_ref);
  const ports=new Map(model.program.ports.map(p=>[p.id,p]));
  const inferred=[], reductionAxes=[];
  for(const binding of bindings) {
    const op=operations.get(binding.operation_ref),port=ports.get(binding.port_ref),name=op?.native_op?.name;
    if (["coreml", "mlmodel"].includes(format)) {
      const contract = op?.attributes?.entries?.find(row=>row.name==="deepbom.parameter_contracts")?.value?.find(row=>row.port===port?.position);
      if (contract?.basis === "serialized_parent_parameter_contract" && Number.isInteger(contract.channel_axis)
        && contract.channel_axis >= 0 && contract.channel_axis < shape.length
        && canonicalJson(contract.shape) === canonicalJson(shape)) {
        inferred.push(contract.channel_axis);
        continue;
      }
    }
    if(format==="onnx" && op?.native_op.domain==="ai.onnx" && ["Conv","QLinearConv"].includes(name) && port?.position===(name==="QLinearConv"?3:1) && shape.length>=3) {inferred.push(0);reductionAxes.push(1);continue;}
    if(port?.position!==1)continue;
    if(format==="onnx" && op.native_op.domain==="ai.onnx" && name==="MatMul" && shape.length>=2){inferred.push(shape.length-1);reductionAxes.push(shape.length-2);}
    if(format==="tflite" && ["CONV_2D","FULLY_CONNECTED","DEPTHWISE_CONV_2D"].includes(name)){inferred.push(name==="DEPTHWISE_CONV_2D"?3:0);if(name!=="DEPTHWISE_CONV_2D")reductionAxes.push(shape.length-1);}
  }
  if(inferred.length && inferred.every(x=>x===inferred[0]) && inferred[0]<shape.length){axis=inferred[0];meaning="output_channel";basis="serialized_operator_weight_contract";}
  if(override.channel_axis!==undefined){axis=override.channel_axis;meaning="user_selected_axis_slices";basis="explicit_analysis_option";}
  const inferredReduction=reductionAxes.length&&reductionAxes.every(x=>x===reductionAxes[0])?reductionAxes[0]:null;
  const sparsityAxis=override.sparsity_axis ?? inferredReduction ?? (format==="gguf"?0:Math.max(0,shape.length-1));
  return {channel_axis:axis,channel_meaning:meaning,basis,storage_order:order,sparsity_axis:sparsityAxis,sparsity_axis_basis:override.sparsity_axis!==undefined?"explicit_analysis_option":inferredReduction!==null?"serialized_operator_reduction_axis":"innermost_serialized_axis"};
}
export function dequantize(values, row, contract, order) {
  if(!contract)return {values,report:notAssessed(row.representation==="dequantized_real"?"decoder_already_dequantized_block_parameters_not_exposed":"complete_affine_quantization_contract_unavailable"),representation:row.representation};
  const result=numericalWasm().numerical_affine(values,row.shape,row.dtype,contract,order);
  return {values:result.values??values,report:result.report,representation:result.report.status==="assessed"?"dequantized_real":row.representation};
}
