import { visitMilImmediateValues } from "../web/lib/numerical-ir/weight-sources.js";
import Ajv2020 from "ajv/dist/2020.js";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { analyzeOnnxModel } from "../web/onnx.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { parseCoreMlModel } from "../web/lib/coreml-metadata-adapter.js";
import { parseMetadataModel } from "../web/lib/metadata-model-adapters.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { TensorStatistics, statisticsOf, validateStatistics, HISTOGRAM_EDGES } from "../web/lib/numerical-ir/statistics.js";
import { seal } from "../web/lib/numerical-ir/common.js";
import { buildWeightIr, validateWeightIr } from "../web/lib/weight-ir.js";
import { buildActivationIr, validateActivationIr, buildNumericalEvidenceBundle } from "../web/lib/activation-ir.js";
import { buildCoreMlPerChannelLinearFixture } from "./coreml-legacy-quantization-corpus-lib.mjs";
import { model, node, tensor, valueInfo, float32 } from "./onnx-proto-fixture.mjs";

const near=(actual,expected)=>assert(Math.abs(actual-expected)<=Math.max(1e-14,Math.abs(expected)*1e-12),`${actual} != ${expected}`);
const s=statisticsOf([-2,0,1,3]);validateStatistics(s);assert.equal(s.value_count,"4");near(s.mean,0.5);near(s.population_stddev,Math.sqrt(3.25));near(s.rms,Math.sqrt(3.5));near(s.l2_norm,Math.sqrt(14));
const extremes=statisticsOf([-1e308,1e308]);near(extremes.mean,0);near(extremes.population_stddev,1e308);assert.equal(extremes.l2_norm,Math.SQRT2*1e308);
const overflow=statisticsOf([Number.MAX_VALUE,Number.MAX_VALUE]);assert.equal(overflow.l2_norm,null);assert.deepEqual(overflow.overflowed_metrics,["l2_norm"]);
const nonfinite=statisticsOf([0,-0,NaN,Infinity,-Infinity]);assert.equal(nonfinite.finite_count,"2");assert.equal(nonfinite.negative_zero_count,"1");validateStatistics(nonfinite);
const exact=statisticsOf([9007199254740993n,-9223372036854775808n]);assert.equal(exact.minimum,null);assert.equal(exact.mean,null);assert.equal(exact.histogram,null);assert.equal(exact.integer_minimum,"-9223372036854775808");validateStatistics(exact);
validateStatistics(statisticsOf([]));validateStatistics(statisticsOf([Number.MIN_VALUE]));
const bins=statisticsOf(HISTOGRAM_EDGES);assert.equal(bins.histogram.counts.reduce((n,v)=>n+BigInt(v),0n),BigInt(HISTOGRAM_EDGES.length));
assert.throws(()=>validateStatistics({...s,finite_count:"3"}));assert.throws(()=>validateStatistics({...s,histogram:{...s.histogram,counts:s.histogram.counts.map(()=>"0")}}));
function context(bytes,analysis,name) { analysis.model_sha256=sha256BytesHex(bytes);return getArtifactIrContext(analysis,{filename:name,format:analysis.format,sha256:analysis.model_sha256,size:bytes.length}).model_ir; }
const bytes=new Uint8Array(model([node("Add","first",["x","w"],["middle"]),node("Add","second",["middle","w"],["y"])],[tensor("w",1,[4],float32([-2,0,1,3]))],[valueInfo("x",1,[4])],[valueInfo("y",1,[4])],13));
const analysis=analyzeOnnxModel(bytes,"shared.onnx"),ir=context(bytes,analysis,"shared.onnx"),original=canonicalJson(ir);
const w=await buildWeightIr(ir,analysis,bytes);assert.equal(w.coverage.assessed_count,1);assert.equal(w.tensors[0].binding_refs.length,2);assert.equal(w.tensors[0].statistics.mean,0.5);assert.equal(canonicalJson(ir),original);
assert.equal((await buildWeightIr(ir,analysis,new Blob([bytes]))).weight_ir_sha256,w.weight_ir_sha256);
await assert.rejects(()=>buildWeightIr(ir,analysis,new Uint8Array([1,2])));
const budget=await buildWeightIr(ir,analysis,bytes,{maxValues:0});assert.equal(budget.coverage.not_assessed_count,1);assert.equal(budget.tensors[0].statistics,null);
function mutateWeight(change) { const copy=structuredClone(w);change(copy);delete copy.weight_ir_sha256;return seal(copy,"weight_ir_sha256"); }
assert.throws(()=>validateWeightIr(mutateWeight(v=>v.tensors.pop()),ir));assert.throws(()=>validateWeightIr(mutateWeight(v=>v.coverage.assessed_count=99),ir));assert.throws(()=>validateWeightIr(mutateWeight(v=>v.tensors[0].binding_refs=[]),ir));assert.throws(()=>validateWeightIr(mutateWeight(v=>v.tensors[0].statistics.value_count="9"),ir));
const core=buildCoreMlPerChannelLinearFixture(),ca=parseCoreMlModel(core,"linear.mlmodel"),ci=context(core,ca,"linear.mlmodel"),cw=await buildWeightIr(ci,ca,core);
const decoded=Array.from({length:18},(_,i)=>i<9?(i%16)*0.5-1:(i%16)*0.25+1);assert.equal(cw.tensors[0].representation,"dequantized_real");assert.equal(cw.tensors[0].statistics.value_count,"18");near(cw.tensors[0].statistics.mean,decoded.reduce((a,b)=>a+b,0)/18);
const ptd=new Uint8Array(Buffer.from(await readFile(new URL("./fixtures/numerical-tensor.ptd.base64.txt",import.meta.url),"utf8"),"base64")),pa=analyzeExecuTorchModel(ptd,"weight.ptd"),pi=context(ptd,pa,"weight.ptd"),pw=await buildWeightIr(pi,pa,ptd);assert.equal(pw.coverage.assessed_count,1);assert.equal(pw.tensors[0].statistics.mean,0.5);
function safeTensor(dtype,shape,payload) { const header=Buffer.from(JSON.stringify({weight:{dtype,shape,data_offsets:[0,payload.length]}}));const size=Buffer.alloc(8);size.writeBigUInt64LE(BigInt(header.length));return new Uint8Array(Buffer.concat([size,header,payload])); }
const u64=Buffer.alloc(16);u64.writeBigUInt64LE(9007199254740993n);u64.writeBigUInt64LE(18446744073709551615n,8);const sb=safeTensor("U64",[2],u64),sa=parseMetadataModel(sb,"exact.safetensors",sb.length,"safetensors"),si=context(sb,sa,"exact.safetensors"),sw=await buildWeightIr(si,sa,sb);assert.equal(sw.tensors[0].statistics.integer_maximum,"18446744073709551615");assert.equal(sw.tensors[0].statistics.mean,null);
const empty=safeTensor("F32",[0],Buffer.alloc(0)),ea=parseMetadataModel(empty,"empty.safetensors",empty.length,"safetensors"),ei=context(empty,ea,"empty.safetensors"),ew=await buildWeightIr(ei,ea,empty);assert.equal(ew.tensors.length,1);assert.equal(ew.tensors[0].storage_ref,null);assert.equal(ew.tensors[0].statistics.value_count,"0");
const vals=ir.program.values,input=vals.find(v=>v.name==="x"),outputs=vals.filter(v=>["middle","y"].includes(v.name));
const config={optimization:"disabled"};
const capture={schema:"deepbom.activation_capture.v1",source:w.source,run:{id:"run1",entry_region_ref:input.region_ref,started_at:"2026-09-22T00:00:00Z",runtime:{name:"fixture",version:"1",configured_providers:["CPU"],device:null},collector:{name:"test",version:"1",sha256:"a".repeat(64)},execution:{artifact_sha256:w.source.artifact_sha256,instrumented_artifact_sha256:null,configuration:config,configuration_sha256:sha256TextHex(canonicalJson(config))},probe:{kind:"synthetic_ones",description:"one-valued tensor"},runtime_evidence:null},inputs:[{value_ref:input.id,native_locator:"x",dtype:"FLOAT32",shape:[4],values:[1,1,1,1]}],requested_value_refs:outputs.map(v=>v.id),captures:[{value_ref:outputs[0].id,native_locator:outputs[0].name,dtype:"FLOAT32",shape:[4],values:[-1,1,2,4]}],missing:[{value_ref:outputs[1].id,reason:"not_preserved"}]};
const a=buildActivationIr(ir,capture);assert.equal(a.coverage.completeness,"partial_requested_capture");assert.equal(a.coverage.missing_count,1);assert.equal(a.tensors[0].statistics.mean,1.5);validateActivationIr(a,ir);assert(!JSON.stringify(a).includes('"values":'));
for(const change of [c=>c.source.artifact_sha256="0".repeat(64),c=>c.missing=[],c=>c.inputs=[],c=>c.captures[0].shape=[3],c=>c.captures[0].dtype="INT8",c=>c.captures[0].values=[1e100,1,2,4],c=>c.requested_value_refs.push(c.requested_value_refs[0]),c=>c.run.execution.configuration.optimization="enabled"]) { const copy=structuredClone(capture);change(copy);assert.throws(()=>buildActivationIr(ir,copy)); }
const bundle=buildNumericalEvidenceBundle(ir,w,a);assert.equal(bundle.weight_ir_sha256,w.weight_ir_sha256);assert.equal(bundle.activation_ir_sha256,a.activation_ir_sha256);
console.log("Numerical IR: exact counts, stable moments, unsafe integers, all inventory rows, shared bindings, ONNX/Core ML/ExecuTorch/SafeTensors decoders, runtime identity, capture coverage and tamper checks passed.");

const schema=JSON.parse(await readFile(new URL("../docs/schemas/deepbom-numerical-ir-v1.schema.json",import.meta.url),"utf8"));
const validateSchema=new Ajv2020({strict:true,allErrors:true}).compile(schema);
for(const document of [w,cw,pw,sw,ew,a,capture,bundle]) assert(validateSchema(document),JSON.stringify(validateSchema.errors));
assert(!validateSchema({...w,unexpected:true}));console.log("Numerical IR JSON Schema contracts passed.");

for(const [dtype,shape,values,expected] of [["INT8",[2],[255,128],[-1,-128]],["UINT8",[2],[255,128],[255,128]],["FLOAT16",[1],[0,60],[1]]]) {
 const decoded=[];const result=await visitMilImmediateValues({dtype,shape,immediate_value:{kind:"bytes",values,count:values.length,truncated:false}},v=>decoded.push(v));assert.equal(result.status,"assessed");assert.deepEqual(decoded,expected);
}
const incomplete=await visitMilImmediateValues({dtype:"INT8",shape:[2],immediate_value:{kind:"bytes",values:[255],count:2,truncated:true}},()=>assert.fail("partial immediate must not emit values"));assert.equal(incomplete.status,"not_assessed");

const integerCapture=structuredClone(capture);integerCapture.captures[0].value_ref=null;integerCapture.captures[0].dtype="INT8";integerCapture.captures[0].values=["-0",1,2,3];assert.throws(()=>buildActivationIr(ir,integerCapture),/negative zero/);
