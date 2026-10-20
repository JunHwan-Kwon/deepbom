import assert from 'node:assert/strict';
import { initializeNumericalWasm } from '../web/lib/numerical-wasm.js';
import { portableContract } from '../web/lib/portable-contracts.js';
import { inferOnnxShapes, mergeOnnxInferredTensor } from '../web/lib/onnx-shape-inference.js';
import { parseOnnxDimensionExpression, evaluateOnnxDimensionExpression } from '../web/lib/onnx-dimension-expression.js';
await initializeNumericalWasm();
const shape=(operation,input)=>portableContract('shape',{operation,...input});
const symbolic=parameter=>({kind:'symbolic',parameter});
const values=rows=>rows?.map(d=>d.kind==='value'?d.value:d.kind==='symbolic'?d.parameter:null);
assert.equal(shape('add',{left:Number.MAX_SAFE_INTEGER,right:1}).kind,'unknown');
assert.equal(shape('factor',{left:Number.MAX_SAFE_INTEGER,factor:2}).kind,'unknown');
assert.equal(shape('add',{left:null,right:3}).kind,'unknown');
assert.deepEqual(values(shape('broadcast',{shapes:[[0,3],[1,3]]})),[0,3]);
assert.equal(shape('broadcast',{shapes:[[2,3],[4,3]]}),null);
assert.deepEqual(values(shape('matmul',{left:[2,3,4],right:[1,4,5]})),[2,3,5]);
assert.deepEqual(values(shape('matmul',{left:[4],right:[4]})),[]);
assert.equal(shape('matmul',{left:[3],right:[4]}),null);
assert.deepEqual(values(shape('reshape',{input:[2,3,4],requested:[0,-1]}).shape),[2,12]);
assert.equal(shape('reshape',{input:[0,3],requested:[0,-1],allow_zero:true}).reason,'reshape_allowzero_one_cannot_mix_zero_and_minus_one');
assert.equal(shape('reshape',{input:[Number.MAX_SAFE_INTEGER,2],requested:[-1]}).reason,'reshape_inferred_dimension_outside_exact_static_integer_range');
assert.deepEqual(values(shape('reshape',{input:[symbolic('N'),3,4],requested:[-1,3]}).shape),['deepbom_expr:mul(v:4,s:N)',3]);
// Enumerate valid start positions independently from the closed-form Pool rule.
let poolCases=0;
for(let n=1;n<=12;n++)for(let k=1;k<=5;k++)for(let stride=1;stride<=4;stride++)for(let pad=0;pad<k;pad++)for(const ceil of [0,1]){
  const result=shape('spatial',{input:[n],kernels:[k],attributes:{strides:[stride],pads:[pad,pad],ceil_mode:ceil}});
  const positions=[];
  for(let start=-pad;start<n+pad;start+=stride){
    if(ceil ? start<n && start<n+pad-k+stride : start+k<=n+pad)positions.push(start);
  }
  if(result && result[0].value>=0)assert.equal(result[0].value,positions.length,JSON.stringify({n,k,stride,pad,ceil}));
  poolCases++;
}
// Symbolic and concrete shape rules must agree under the same assignment,
// including the final ceil-mode window that starts in right padding.
let symbolicCases=0;
for(let n=1;n<=12;n++)for(let k=1;k<=4;k++)for(let stride=1;stride<=3;stride++)for(let pad=0;pad<k;pad++)for(const ceil of [0,1]){
  const attributes={strides:[stride],pads:[pad,pad],ceil_mode:ceil};
  const concrete=shape('spatial',{input:[n],kernels:[k],attributes});
  const symbolicResult=shape('spatial',{input:[symbolic('N')],kernels:[k],attributes});
  const d=symbolicResult[0];
  const evaluated=d.kind==='value'?BigInt(d.value):evaluateOnnxDimensionExpression(parseOnnxDimensionExpression(d.parameter),{N:n});
  assert.equal(evaluated,concrete?BigInt(concrete[0].value):null,JSON.stringify({n,k,stride,pad,ceil}));
  symbolicCases++;
}
// Explicit Slice indexes: check both strides without repeating the length formula.
let slices=0;
for(let size=0;size<=8;size++)for(let start=-12;start<=12;start++)for(let end=-12;end<=12;end++)for(const step of [-3,-1,1,2]){
  const r=shape('slice',{size,start,end,step});
  let a=start<0?start+size:start,b=end<0?end+size:end;
  a=Math.max(step>0?0:-1,Math.min(step>0?size:size-1,a));b=Math.max(step>0?0:-1,Math.min(step>0?size:size-1,b));
  let count=0;for(let i=a;step>0?i<b:i>b;i+=step)count++;
  assert.equal(r.length,count);slices++;
}
assert.equal(shape('slice',{size:5,start:'+inf',end:'-9223372036854775808',step:-1}).length,5);
assert.equal(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression('deepbom_expr:add(s:N,1)'),{N:null}),null);
assert.equal(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression('deepbom_expr:mul(s:N,s:N)'),{N:9007199254740993n}),81129638414606699710187514626049n);
// Pad pairs follow the serialized order of axes, including negative axes.
const tensor=(name,shape,staticValues)=>({name,shape,dtype:staticValues?'INT64':'FLOAT32',shapeDeclared:true,
  ...(staticValues?{staticValues,staticValuesComplete:true,staticValuesStatus:'complete'}:{})});
for(const axes of [[2,0],[-1,0]]){
 const tensors=new Map([tensor('x',[10,20,30]),tensor('pads',[4],[1,2,3,4]),tensor('axes',[2],axes)].map(t=>[t.name,t]));
 inferOnnxShapes({nodes:[{opType:'Pad',inputs:['x','pads','','axes'],outputs:['y'],attributes:new Map(),domain:''}]},tensors,[{domain:'',version:18}],()=> 'FLOAT32');
 assert.deepEqual(tensors.get('y').shape,[16,20,34]);
}
console.log(`Portable shapes: ${poolCases} Pool window cases, ${symbolicCases} symbolic assignments, ${slices} Slice enumerations; exact overflow, Reshape, MatMul, unknown and permuted Pad axes passed.`);

// IR leaf contracts preserve the existing timestamp/reference API and digests.
const { evidenceTimestamp, subjectReference, uniqueEvidenceIds, canonicalEvidenceDigest } = await import('../web/lib/evidence-identity.js');
const { createHash } = await import('node:crypto');
for (const stamp of ['0000-02-29T23:59:59.999-23:59','0099-01-01T00:00:00Z','1900-03-01T12:00:00Z','2000-02-29T00:00:00.1+09:00','9999-12-31T23:59:59.999Z']) assert.equal(evidenceTimestamp(stamp),Date.parse(stamp));
for (const stamp of ['1900-02-29T00:00:00Z','2026-10-14T24:00:00Z','2026-10-14T00:00:60Z','2026-10-14T00:00:00.0000Z','2026-10-14T00:00:00+24:00',null]) assert.throws(()=>evidenceTimestamp(stamp),/Evidence:/);
assert.equal(subjectReference('artifact_file','a'.repeat(64)).digest_scope,'file_bytes');
assert.throws(()=>subjectReference('snapshot','a'.repeat(64),{schema:'wrong'}),/snapshot schema mismatch/);
assert.throws(()=>uniqueEvidenceIds([{id:'same'},{id:'same'}]),/IDs must be unique/);
assert.throws(()=>uniqueEvidenceIds([{id:'\ufeff'}]),/nonempty bounded text/);
assert.throws(()=>uniqueEvidenceIds([{id:'😀'.repeat(2049)}]),/nonempty bounded text/);
// A pre-existing canonical-byte identity must not change in this migration.
assert.equal(canonicalEvidenceDigest({b:2,a:1}),createHash('sha256').update('{"a":1,"b":2}').digest('hex'));
const {validateRedesignImplementationPlan} = await import('../web/lib/redesign-codegen.js');
const plan={schema:'deepbom.redesign_implementation_plan.v1',model_inputs:[{tensor_index:0,projected_shape:[1,2]}],nodes:[{op_index:0,generated_symbol:'op_0',module_kind:'relu',activation_inputs:[0],input_shapes:[[1,2]],outputs:[1],output_shapes:[[1,2]],codegen_status:'supported'}],model_outputs:[{tensor_index:1,projected_shape:[1,2]}]};
assert.equal(validateRedesignImplementationPlan(plan,{executable:true}),true);
const duplicate=structuredClone(plan);duplicate.nodes.push({...duplicate.nodes[0],op_index:1,outputs:[2]});assert.throws(()=>validateRedesignImplementationPlan(duplicate),/operation identity/);
const bad=structuredClone(plan);Object.assign(bad.nodes[0],{module_kind:'add',activation_inputs:[0,0],input_shapes:[[1,2],[1,2]],output_shapes:[[1,3]]});bad.model_outputs[0].projected_shape=[1,3];assert.throws(()=>validateRedesignImplementationPlan(bad,{executable:true}),/broadcast disagrees/);
assert.deepEqual(values(shape('pad',{input:[Number.MAX_SAFE_INTEGER],axes:[0],pads:[Number.MAX_SAFE_INTEGER,-Number.MAX_SAFE_INTEGER]})),[Number.MAX_SAFE_INTEGER]);
console.log('Common redesign admission and evidence timestamp/reference/identity contracts passed.');
const ternary=structuredClone(plan);Object.assign(ternary.nodes[0],{module_kind:'add',activation_inputs:[0,0,0],input_shapes:[[1,2],[1,2],[1,2]]});assert.throws(()=>validateRedesignImplementationPlan(ternary,{executable:true}),/input arity/);
const concat=structuredClone(plan);Object.assign(concat.nodes[0],{module_kind:'concatenate',activation_inputs:[0,0],input_shapes:[[1,2],[1,2]],concatenation_axis_nhwc:-1,output_shapes:[[1,4]]});concat.model_outputs[0].projected_shape=[1,4];assert.equal(validateRedesignImplementationPlan(concat,{executable:true}),true);concat.nodes[0].concatenation_axis_nhwc=0;assert.throws(()=>validateRedesignImplementationPlan(concat,{executable:true}),/Concatenation/);
console.log('Framework arity and negative-axis concatenation contracts passed.');

// Unknown dimensions must survive host decoding, Size inference and type merging.
const { makeOnnxTensorType, makeOnnxTensorTypeFromDimensions } = await import('../web/lib/onnx-type-proto.js');
for (const unknown of [null, false, true, '', ' ', [], {}]) {
  assert.equal(makeOnnxTensorType('FLOAT32', [unknown, 2], true).shapeDimensions[0].kind, 'unknown');
  assert.equal(makeOnnxTensorTypeFromDimensions('FLOAT32', [{kind:'value',value:unknown}], true).shapeDimensions[0].kind, 'unknown');
  const ts=new Map([['x',tensor('x',[unknown,2])]]);
  inferOnnxShapes({nodes:[{opType:'Size',inputs:['x'],outputs:['y'],attributes:new Map(),domain:''}]},ts,[{domain:'',version:18}],()=> 'FLOAT32');
  assert.notEqual(ts.get('y')?.staticValuesComplete,true, 'unknown shape must not imply a measured zero size');
}
const unresolved=new Map([['x',tensor('x',[null,2])]]);
assert.equal(mergeOnnxInferredTensor(unresolved,'x',tensor('x',[null,2]),0,'Identity'),null);
assert.equal(unresolved.get('x').shapeDimensions?.[0]?.kind ?? unresolved.get('x').typeProto.shapeDimensions[0].kind,'unknown');
assert.equal(mergeOnnxInferredTensor(unresolved,'x',tensor('x',[3,2]),0,'Identity'),null);
assert.deepEqual(unresolved.get('x').shape,[3,2]);
const zero=new Map([['x',tensor('x',[0,2])]]);
inferOnnxShapes({nodes:[{opType:'Size',inputs:['x'],outputs:['y'],attributes:new Map(),domain:''}]},zero,[{domain:'',version:18}],()=> 'FLOAT32');
assert.deepEqual(zero.get('y').staticValues,[0]);
console.log('Unknown/invalid ONNX dimensions stay distinct from zero in Size and merged type contracts.');
