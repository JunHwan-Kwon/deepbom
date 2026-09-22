import { BoundedFlatBufferReader } from "../flatbuffer-reader.js";
import { readBytes, sourceSize, MAX_IN_MEMORY_SOURCE } from "./weight-sources.js";
import { assessed, notAssessed } from "./weight-math.js";
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
  const {scales,zero_points:zeros,axis,parameterization,block_size:blockSize,scale_shape:scaleShape}=contract;
  const integer=/^(U?INT)(2|4|8|16|32)$/.exec(row.dtype);
  if(!integer || !Array.isArray(scales) || !scales.length || !scales.every(s=>Number.isFinite(s)&&s>0) || !Array.isArray(zeros) || !(zeros.length===0||zeros.length===scales.length) || !zeros.every(Number.isSafeInteger))return {values,report:notAssessed("invalid_or_unsupported_affine_contract"),representation:row.representation};
  const minimum=integer[1]==="UINT"?0:-(2**(Number(integer[2])-1)),maximum=integer[1]==="UINT"?2**Number(integer[2])-1:2**(Number(integer[2])-1)-1;
  if(!zeros.every(z=>z>=minimum&&z<=maximum))return {values,report:notAssessed("zero_point_outside_encoding_range"),representation:row.representation};
  const shape=row.shape;
  if(parameterization!=="per_tensor" && (!Number.isInteger(axis)||axis<0||axis>=shape.length))return {values,report:notAssessed("invalid_quantization_axis"),representation:row.representation};
  if(parameterization==="per_axis"&&scales.length!==shape[axis] || parameterization==="per_tensor"&&scales.length!==1 || !["per_tensor","per_axis","blocked"].includes(parameterization))return {values,report:notAssessed("quantization_cardinality_mismatch"),representation:row.representation};
  if(parameterization==="blocked" && (!Number.isSafeInteger(blockSize)||blockSize<=0||!Array.isArray(scaleShape)||scaleShape.length!==shape.length||!scaleShape.every((d,i)=>d===(i===axis?Math.ceil(shape[i]/blockSize):shape[i]))||scaleShape.reduce((a,b)=>a*b,1)!==scales.length))return {values,report:notAssessed("blocked_quantization_cardinality_mismatch"),representation:row.representation};
  const strides=shape.map((_,i)=>(order==="first_axis_fastest"?shape.slice(0,i):shape.slice(i+1)).reduce((a,b)=>a*b,1));
  const output=new Float64Array(values.length);let min=Infinity,max=-Infinity,lower=0,upper=0;
  for(let i=0;i<values.length;i++) {
    const code=values[i];if(!Number.isInteger(code)||code<minimum||code>maximum)return {values,report:notAssessed("stored_code_outside_encoding_range"),representation:row.representation};
    let index=0;
    if(parameterization==="per_axis")index=Math.floor(i/strides[axis])%shape[axis];
    else if(parameterization==="blocked") for(let dim=0;dim<shape.length;dim++){let coordinate=Math.floor(i/strides[dim])%shape[dim];if(dim===axis)coordinate=Math.floor(coordinate/blockSize);index=index*scaleShape[dim]+coordinate;}
    output[i]=(code-(zeros[index]??0))*scales[index];if(!Number.isFinite(output[i]))return {values,report:notAssessed("dequantization_overflow"),representation:row.representation};
    min=Math.min(min,code);max=Math.max(max,code);lower+=Number(code===minimum);upper+=Number(code===maximum);
  }
  return {values:output,representation:"dequantized_real",report:assessed({method:"affine_scale_times_code_minus_zero_point",parameterization,axis:parameterization==="per_tensor"?null:axis,block_size:blockSize??null,scales,zero_points:zeros.length?zeros:scales.map(()=>0),encoding_minimum:minimum,encoding_maximum:maximum,observed_code_minimum:values.length?min:null,observed_code_maximum:values.length?max:null,range_utilization:values.length?(max-min)/(maximum-minimum):null,lower_endpoint_count:lower,upper_endpoint_count:upper,endpoint_interpretation:"observed_endpoint_occupancy_not_proof_of_clipping",reconstruction_error:notAssessed("original_reference_required")})};
}
