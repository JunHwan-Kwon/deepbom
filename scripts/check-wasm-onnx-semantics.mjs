import assert from 'node:assert/strict';
import {makeOnnxTensorTypeFromDimensions} from '../web/lib/onnx-type-proto.js';
import {portableContract} from '../web/lib/portable-contracts.js';
import {onnxNodeSemantic,onnxMlSemantic,onnxSemantic} from '../web/lib/onnx-portable.js';
const tensor=(dtype,shape,values)=>({dtype,shape,shapeDeclared:true,...(values?{staticValuesComplete:true,staticValues:values}:{})});
const node=(opType,inputs=['x'],outputs=['y'],attributes={})=>({opType,inputs,outputs,attributes:new Map(Object.entries(attributes))});
const infer=(n,values)=>onnxNodeSemantic('conditional_node',n,new Map(Object.entries(values)));
for(const op of ['Relu','Sigmoid','Abs','Neg','Dropout']) {
 const r=infer(node(op),{x:tensor('INT32',[2],[-2,5])});
 assert.deepEqual(r.outputs[0][1].shape,[2]);
 assert.equal(r.outputs[0][1].staticValuesComplete,undefined,`${op} is not a value identity`);
}
assert.deepEqual(infer(node('Identity'),{x:tensor('INT32',[2],[-2,5])}).outputs[0][1].staticValues,[-2,5]);
assert.equal(infer(node('Cast',['x'],['y'],{to:{type:2,i:3,valueTypesPresent:[2]}}),{x:tensor('INT32',[2],[1000,-2])}).outputs[0][1].staticValuesComplete,undefined);
assert.match(infer(node('ReduceSum',['x','axes']),{x:tensor('FLOAT32',[2,3]),axes:tensor('INT64',[1])}).reason,/runtime|unknown/);
assert.match(infer(node('Split',['x','sizes'],['a','b']),{x:tensor('FLOAT32',[6]),sizes:tensor('INT64',[2])}).reason,/runtime|unknown/);
const split=onnxNodeSemantic('container',node('SplitToSequence',['x','split']),new Map(Object.entries({x:tensor('FLOAT32',[Number.MAX_SAFE_INTEGER]),split:tensor('INT64',[],[1])})),{node_index:0,imported_opset:18,scope:'main_graph',empty_dtype:'FLOAT32'});
assert.equal(split.result.outputs[0][1].sequenceLength,Number.MAX_SAFE_INTEGER);
assert.deepEqual(split.result.outputs[0][1].sequenceElementTypes,[]);
const normal=onnxMlSemantic({node:{...node('Normalizer',['x'],['y'],{norm:{type:3,s:'L1',valueTypesPresent:[3]}}),domain:'ai.onnx.ml'},tensorMap:new Map([['x',tensor('FLOAT32',[Number.MAX_SAFE_INTEGER,0],[])]]),nodeIndex:0,importedOpset:1});
assert.equal(normal.row.normalizer_zero_divisor_row_count,Number.MAX_SAFE_INTEGER);
assert.deepEqual(normal.result.outputs[0][1].staticValues,[]);
const deform=infer(node('DeformConv',['x','w','offset'],['y'],{offset_group:{i:2}}),{x:tensor('FLOAT32',[1,4,8,8]),w:tensor('FLOAT32',[6,4,3,3]),offset:tensor('FLOAT32',[1,18,6,6])});
assert.equal(deform.reason,'deform_conv_offset_shape_mismatch');
console.log('Shared ONNX: value identity, cast boundaries, runtime controls, bounded sequence allocation, empty-row work, and DeformConv cardinality');

for (const axis of [2**32,2**32+1,Number.MAX_SAFE_INTEGER]) {
 const a=infer(node('Gather',['x','indices'],['y'],{axis:{i:axis}}),{x:tensor('FLOAT32',[2,3]),indices:tensor('INT64',[1],[0])});
 assert.equal(a.reason,'gather_axis_out_of_range');
 assert.equal(portableContract('shape',{operation:'flatten',input:[2,3],axis}),null);
 assert.equal(portableContract('shape',{operation:'concat',shapes:[[2,3],[2,3]],axis}),null);
}
const hugeConstant=infer(node('Constant',[],['y'],{value_int:{type:2,i:null,iExactDecimal:'9223372036854775807',valueTypesPresent:[2]}}),{}).outputs[0][1];
assert.equal(hugeConstant.staticValuesComplete,false);
assert.deepEqual(hugeConstant.initializerIntegerValuesExactDecimals,['9223372036854775807']);
assert.deepEqual(infer(node('Identity'),{x:hugeConstant}).outputs[0][1].initializerIntegerValuesExactDecimals,['9223372036854775807']);
const badBlock=onnxNodeSemantic('ort_schema',{...node('MatMulNBits',['a','b','scale'],['y'],{K:{type:2,i:32,valueTypesPresent:[2]},N:{type:2,i:16,valueTypesPresent:[2]},block_size:{type:2,i:2**32+16,valueTypesPresent:[2]}}),domain:'com.microsoft'},null,{family:'extension',imported_opset:1});
assert(badBlock.reason_codes.some(reason=>reason.includes('block_size')));
// Missing branch flags must produce a contract failure, never a WASM trap.
const ints=values=>({type:7,valueTypesPresent:[7],ints:values,intExactDecimals:values.map(String)});
const tensorAttr=(dtype,values)=>({type:4,valueTypesPresent:[4],tensor:tensor(dtype,[values.length],values)});
const malformedTree=onnxMlSemantic({node:{...node('TreeEnsemble',['x'],['y'],{
 nodes_featureids:ints([0]),nodes_splits:tensorAttr('FLOAT32',[0]),nodes_modes:tensorAttr('UINT8',[0]),
 nodes_truenodeids:ints([0]),nodes_falsenodeids:ints([1]),nodes_falseleafs:ints([1]),
 tree_roots:ints([0]),leaf_targetids:ints([0,0]),leaf_weights:tensorAttr('FLOAT32',[2,3]),n_targets:{type:2,valueTypesPresent:[2],i:1,iExactDecimal:'1'}
}),domain:'ai.onnx.ml'},tensorMap:new Map([['x',tensor('FLOAT32',[1,1],[0])]]),nodeIndex:0,importedOpset:5});
assert.equal(malformedTree.status,'fail');
assert(malformedTree.row.reason_codes.includes('tree_ensemble_v5_node_tuple_cardinality_mismatch'));
// A zero-length token sequence must finish even for the largest safe skip count.
const tfidfNode=node('TfIdfVectorizer',['x'],['y'],{
 mode:{s:'TF'},min_gram_length:{i:1},max_gram_length:{i:1},max_skip_count:{i:Number.MAX_SAFE_INTEGER},
 ngram_counts:ints([0]),ngram_indexes:ints([0]),pool_strings:{strings:['a']},
});
const emptyText=onnxNodeSemantic('tfidf',tfidfNode,new Map([['x',tensor('STRING',[0],[])]]),{scope:'main_graph',node_index:0,imported_opset:9});
assert.equal(emptyText.status,'pass');
assert.deepEqual(emptyText.row.exact_output_values,[0]);
assert.equal(emptyText.row.exact_static_work_units,0);
console.log('TfIdf empty input with a huge skip bound terminates with exact zero output.');

// If (symbolic dimension == 1) discharges only the matching Squeeze failure.
const symbolic={...tensor('FLOAT32',[-1,3]),typeProto:makeOnnxTensorTypeFromDimensions('FLOAT32',[{kind:'symbolic',parameter:'batch'},{kind:'value',value:3}],true)};
const squeezed=infer(node('Squeeze',['x','axes']),{x:symbolic,axes:tensor('INT64',[1],[0])}).outputs[0][1];
assert.equal(squeezed.conditionalShapeContract.status,'assessed_partial');
const dimValue={...tensor('INT64',[1]),staticDimensionValuesComplete:true,staticDimensionValues:[{kind:'symbolic',parameter:'batch',value:null}]};
const guard=onnxNodeSemantic('region_condition',node('Equal',['dim','one']),new Map([['dim',dimValue],['one',tensor('INT64',[1],[1])]]));
assert.deepEqual(guard,{key:'onnx:dimension:batch:equals:1',thenValue:'true',elseValue:'false'});
const guarded=onnxSemantic('union_branch',{left:squeezed,right:symbolic,condition:guard}).patch;
assert.equal(guarded.conditionalShapeContract.status,'assessed_complete');
assert.equal(guarded.conditionalShapeVariants.length,2);
assert.equal(guarded.conditionalShapeContract.variant_failures,undefined);
const unrelated=onnxSemantic('union_branch',{left:squeezed,right:symbolic,condition:{key:'unrelated',thenValue:'true',elseValue:'false'}}).patch;
assert.equal(unrelated.conditionalShapeContract.status,'assessed_partial');
assert.equal(unrelated.conditionalShapeContract.variant_failures.length,1);
assert.equal(unrelated.conditionalShapeContract.variant_failures[0].reason,'squeeze_axis_dimension_not_one');
assert.equal(onnxNodeSemantic('region_condition',node('Equal',['dim','one']),new Map([['dim',dimValue],['one',tensor('INT64',[1],[2])]])),null);
assert.equal(onnxNodeSemantic('region_condition',{...node('Equal',['dim','one']),domain:'custom'},new Map([['dim',dimValue],['one',tensor('INT64',[1],[1])]])),null);
console.log('Shared If/Squeeze predicates: matching guards discharge failures; unrelated guards retain them.');
