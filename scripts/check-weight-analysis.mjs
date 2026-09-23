import Ajv2020 from "ajv/dist/2020.js";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { prepareMagnitudePruning, simulateMagnitudePruning } from "../web/lib/weight-pruning.js";
import { collectWeightAnalysis, compareWeightAnalyses, validateWeightAnalysis } from "../web/lib/weight-analysis.js";
import { matrixView, spectrumAnalysis, similarityAnalysis, sparsityAnalysis, compareValues } from "../web/lib/numerical-ir/weight-math.js";
import { dequantize } from "../web/lib/numerical-ir/weight-contracts.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { parseCoreMlModel } from "../web/lib/coreml-metadata-adapter.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";
import { seal } from "../web/lib/numerical-ir/common.js";
import { buildCoreMlPerChannelLinearFixture } from "./coreml-legacy-quantization-corpus-lib.mjs";
import { model, node, nodeWithIntegerAttributes, tensor, valueInfo, float32 } from "./onnx-proto-fixture.mjs";
import { initSync, analyze_tflite } from "../pkg/tflite_wasm_audit.js";

const near=(actual,expected,rtol=1e-10,atol=1e-13)=>assert(Math.abs(actual-expected)<=Math.max(atol,Math.abs(expected)*rtol),actual+" != "+expected);
const oracle=JSON.parse(await readFile(new URL('./fixtures/weight-svd-oracles.json',import.meta.url),'utf8'));
for(const fixture of oracle.cases) {
  const result=spectrumAnalysis({data:Float64Array.from(fixture.values),rows:fixture.rows,columns:fixture.columns});
  assert.equal(result.status,'assessed',fixture.name);assert.equal(result.numerical_rank,fixture.rank,fixture.name);
  result.singular_values.forEach((x,i)=>near(x,fixture.singular_values[i],fixture.name==='ill_conditioned'?2e-6:1e-10));
  assert(result.energy_relative_residual<1e-12);
}
assert.equal(spectrumAnalysis(matrixView([1,2,3,4],[2,2],0,'last_axis_fastest'),0).status,'not_assessed');
const sim=similarityAnalysis(matrixView([1,0,1,0,-1,0,0,0],[4,2],0,'last_axis_fastest'),10,0.95);
assert.equal(sim.values[1],1);assert.equal(sim.values[2],-1);assert.equal(sim.values[3],null);assert.deepEqual(sim.clusters,[[0,1]]);
assert.equal(similarityAnalysis(matrixView([1,2,3,4],[2,2],0,'last_axis_fastest'),1,0.95).status,'not_assessed');
assert.deepEqual([...matrixView([1,2,3,4,5,6],[2,3],0,'first_axis_fastest').data],[1,3,5,2,4,6]);
const sp=sparsityAnalysis([0,0,1,2,0,3,0,4],[2,4],'last_axis_fastest',1);assert.equal(sp.structured_2_4.pattern_satisfied,true);assert.equal(sp.structured_2_4.groups,2);
assert.equal(sparsityAnalysis([1,2,3,0],[1,4],'last_axis_fastest',1).structured_2_4.pattern_satisfied,false);
assert.equal(sparsityAnalysis([0,0,0],[1,3],'last_axis_fastest',1).structured_2_4.pattern_satisfied,null);
const affine={scales:[0.5,2],zero_points:[1,2],axis:0,parameterization:'per_axis',block_size:null,scale_shape:[2]};
const dq=dequantize([1,3,2,4],{dtype:'INT8',shape:[2,2],representation:'stored_scalar'},affine,'last_axis_fastest');
assert.deepEqual([...dq.values],[0,1,0,4]);assert.equal(dq.report.status,'assessed');
const cmp=compareValues([0,1,0,2],[0,1,0,4]);near(cmp.metrics.rmse,1);near(cmp.metrics.relative_l2,2/Math.sqrt(5));assert.equal(cmp.metrics.changed_count,'1');

function context(bytes,analysis,name){analysis.model_sha256=sha256BytesHex(bytes);return getArtifactIrContext(analysis,{filename:name,format:analysis.format,sha256:analysis.model_sha256,size:bytes.length}).model_ir;}
async function collect(bytes,analysis,name,options={}){const ir=context(bytes,analysis,name);return {ir,...await collectWeightAnalysis(ir,analysis,bytes,{options})};}
const onnx=new Uint8Array(model([node('MatMul','mm',['x','w'],['y'])],[tensor('w',1,[2,2],float32([1,2,3,4]))],[valueInfo('x',1,[1,2])],[valueInfo('y',1,[1,2])],13));
const a=await collect(onnx,analyzeOnnxModel(onnx,'a.onnx'),'a.onnx');assert.equal(a.weight_analysis.coverage.assessed_count,1);near(a.weight_analysis.tensors[0].spectrum.singular_values[0],5.464985704219043);
const qdq=new Uint8Array(model([nodeWithIntegerAttributes('DequantizeLinear','dq',['w','scale','zp'],['y'],{axis:0})],[tensor('w',3,[2,2],new Uint8Array([1,3,2,4])),tensor('scale',1,[2],float32([0.5,2])),tensor('zp',3,[2],new Uint8Array([1,2]))],[],[valueInfo('y',1,[2,2])],13));
const quantized=await collect(qdq,analyzeOnnxModel(qdq,'q.onnx'),'q.onnx');
const quantRow=quantized.weight_analysis.tensors.find(t=>t.name==='w');
assert.equal(quantRow.quantization.status,'assessed');assert.equal(quantRow.statistics.maximum,4);assert.equal(quantRow.statistics.zero_count,'2');
const blocked=dequantize([1,2,3,4,5,6,7,8],{dtype:'INT8',shape:[2,4],representation:'stored_scalar'},{scales:[1,2,3,4],zero_points:[0,0,0,0],axis:1,parameterization:'blocked',block_size:2,scale_shape:[2,2]},'last_axis_fastest');
assert.deepEqual([...blocked.values],[1,2,6,8,15,18,28,32]);
const identical=compareWeightAnalyses(a,a);assert.equal(identical.coverage.compared_count,1);assert.equal(identical.tensors[0].metrics.rmse,0);
const restricted=await collect(onnx,analyzeOnnxModel(onnx,'a.onnx'),'a.onnx',{max_values:0});assert.equal(restricted.weight_analysis.coverage.not_assessed_count,1);
await assert.rejects(()=>collect(onnx,analyzeOnnxModel(onnx,'a.onnx'),'a.onnx',{axes:{'weight:0':{channel_axis:5}}}),/axis/);
await assert.rejects(()=>collect(onnx,analyzeOnnxModel(onnx,'a.onnx'),'a.onnx',{tensor_ids:['weight:900']}),/unknown weight reference/);
const changed=structuredClone(a.weight_analysis);changed.coverage.assessed_count=99;delete changed.weight_analysis_sha256;
assert.throws(()=>validateWeightAnalysis(seal(changed,'weight_analysis_sha256'),a.weight_ir,a.ir),/coverage/);
function safetensors(shape,values){const payload=Buffer.from(float32(values));const header=Buffer.from(JSON.stringify({w:{dtype:'F32',shape,data_offsets:[0,payload.length]}})),length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(header.length));return new Uint8Array(Buffer.concat([length,header,payload]));}
const st=safetensors([2,2],[1,2,3,4]),safe=await collect(st,parseMetadataModel(st,'a.safetensors',st.length,'safetensors'),'a.safetensors');
assert.equal(compareWeightAnalyses(a,safe).tensors[0].metrics.rmse,0,'cross-format decoded values');
const transposed=safetensors([2,2],[1,3,2,4]),trans=await collect(transposed,parseMetadataModel(transposed,'t.safetensors',transposed.length,'safetensors'),'t.safetensors');
assert.equal(compareWeightAnalyses(a,trans,{mapping:[{baseline:'weight:0',candidate:'weight:0',candidate_axis_permutation:[1,0]}]}).tensors[0].metrics.rmse,0);
assert.throws(()=>compareWeightAnalyses(a,trans,{mapping:[{baseline:'weight:0',candidate:'weight:0',candidate_axis_permutation:[0,0]}]}),/permutation/);
const core=buildCoreMlPerChannelLinearFixture(),c=await collect(core,parseCoreMlModel(core,'a.mlmodel'),'a.mlmodel');assert(c.weight_analysis.coverage.assessed_count>0);assert.equal(c.weight_analysis.tensors[0].representation,'dequantized_real');
const coreBias=buildCoreMlPerChannelLinearFixture({biasValues:[2,3]});
for(const options of [{tensor_ids:['weight:0'],max_values:18},{tensor_ids:['weight:1'],max_values:2},{max_tensor_values:2,max_values:2}]) {
  const selected=await collect(coreBias,parseCoreMlModel(coreBias,'selected.mlmodel'),'selected.mlmodel',options);
  const index=options.tensor_ids?.[0]==='weight:0'?0:1;
  assert.equal(selected.weight_analysis.tensors[index].status,'assessed','excluded Core ML payloads must not consume capture budget');
  assert.equal(selected.weight_analysis.coverage.assessed_count,1);
  assert.equal(selected.weight_analysis.tensors[index].statistics.value_count,index===0?'18':'2');
}
const ptd=new Uint8Array(Buffer.from(await readFile(new URL('./fixtures/numerical-tensor.ptd.base64.txt',import.meta.url),'utf8'),'base64')),p=await collect(ptd,analyzeExecuTorchModel(ptd,'a.ptd'),'a.ptd');assert(p.weight_analysis.coverage.assessed_count>0);
initSync({module:await readFile('pkg/tflite_wasm_audit_bg.wasm')});
const tflite=new Uint8Array(await readFile('web/samples/mobilenet_v2_1.0_224_quant.tflite')),tf=await collect(tflite,analyze_tflite(tflite,'a.tflite'),'a.tflite');
assert(tf.weight_analysis.coverage.feature_counts.quantization>0,'TFLite affine contracts');
const gguf=new Uint8Array(await readFile('corpus/external-review/fixtures/gguf-f16.gguf')),g=await collect(gguf,parseMetadataModel(gguf,'a.gguf',gguf.length,'gguf'),'a.gguf');assert(g.weight_analysis.coverage.assessed_count>0);assert.equal(g.weight_analysis.tensors[0].axes.storage_order,'first_axis_fastest');
console.log('Advanced Weight IR passed: NumPy SVD oracles, signed similarity, exact axes, zero-point restoration, 2:4 tails, cross-format and transpose comparison, coverage/tamper/budget rejection, ONNX/SafeTensors/CoreML/ExecuTorch/TFLite/GGUF actual decoders.');
console.log(JSON.stringify({tflite:tf.weight_analysis.coverage,gguf:g.weight_analysis.coverage}));

const adjacent=compareValues([1e100,0],[1e100,1e-200]);
assert(adjacent.metrics.rmse>0,'small absolute error must not vanish next to a large unchanged value');
assert.equal(adjacent.metrics.changed_count,'1');assert(Number.isFinite(adjacent.metrics.sqnr_db));
const overflow=compareValues([1e308,0],[-1e308,0]);assert.equal(overflow.metrics.maximum_absolute_error,null);assert(Number.isFinite(overflow.metrics.rmse));
const ajv=new Ajv2020({strict:true,allErrors:true});
ajv.addSchema(JSON.parse(await readFile('docs/schemas/deepbom-numerical-ir-v1.schema.json','utf8')));
const schema=ajv.compile(JSON.parse(await readFile('docs/schemas/deepbom-weight-analysis-v1.schema.json','utf8')));
for(const result of [a,safe,trans,c,p,tf,g,restricted,quantized])assert(schema(result.weight_analysis),JSON.stringify(schema.errors));
assert(schema(identical),JSON.stringify(schema.errors));
console.log('Advanced analysis/comparison JSON Schemas passed.');
for (const result of [a, safe, c, p, tf, g, quantized]) {
  const decoded = [...result.decoded.values()].find(d => d.matrix.data.length > 0 && d.matrix.rows <= 4096 && (d.row.representation === 'dequantized_real' || /^(?:B?FLOAT|BF|F)(?:8|16|32|64)/.test(d.row.dtype)));
  assert(decoded, result.ir.artifact.format + ': expected a real-valued pruning fixture');
  const prepared = prepareMagnitudePruning({...decoded,source:result.weight_ir.source,weight_ir_sha256:result.weight_ir.weight_ir_sha256});
  const preview = simulateMagnitudePruning(prepared,{target_percent:50});
  assert.equal(preview.candidate_zero_count,Math.max(preview.original_zero_count,Math.floor(preview.value_count/2)));
  assert.equal(preview.metrics.changed_count,String(preview.newly_zeroed_count));
  assert.equal(preview.channels.reduce((n,c)=>n+c.newly_zeroed_count,0),preview.newly_zeroed_count);
}
console.log('Pruning simulation passed through six actual format decoders and affine ONNX quantization.');
