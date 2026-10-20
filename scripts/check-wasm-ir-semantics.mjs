import assert from "node:assert/strict";
import { portableContract } from "../web/lib/portable-contracts.js";
import { integerMetric, aggregateIntegerMetrics } from "../web/lib/ir-metric.js";
import { exactInteger, exactCountValue, safeBigIntNumber, validateExactInteger } from "../web/lib/exact-integer.js";
import { buildValueType, valueElementCount, validateValueType } from "../web/lib/ir-value-type.js";
import { buildArtifactMembers, validateArtifactMembers } from "../web/lib/ir-artifact-members.js";
import { validateLogicalInventory } from "../web/lib/ir-logical-inventory.js";
import { buildCoreMlDeploymentContract } from "../web/lib/coreml-deployment-contract.js";
import { buildOnnxContractConflictCapsule } from "../web/lib/onnx-contract-conflict.js";
import { inputManifestSha256, validateRuntimeProvenance } from "../web/lib/ir-runtime-provenance.js";
import { canonicalEvidenceDigest } from "../web/lib/evidence-identity.js";
const ir=(operation,input)=>portableContract("ir_semantics",{operation,...input});
const identity={filename:"m",sha256:"a".repeat(64),byte_length:exactInteger(100)};
const object=(id,a,b)=>({id,native_source:null,byte_range:{status:"exact",offset_basis:"artifact_absolute",start:String(a),end_exclusive:String(b)},serialized_byte_length:exactInteger(BigInt(b)-BigInt(a))});
const storage={objects:[object("a",0,10),object("empty",5,5),object("b",10,20)]};
const members=buildArtifactMembers({},identity,storage);assert.deepEqual(members.overlap_groups,[]);validateArtifactMembers(members,identity,storage);
assert.throws(()=>validateArtifactMembers({...members,completeness:"partial_member_binding"},identity,storage),/completeness/);
assert.throws(()=>validateArtifactMembers(members,identity,{objects:[...storage.objects,storage.objects[0]]}),/identit/);
const huge=9007199254740993n;
const largeStorage={objects:[object("a",huge,huge+10n),object("b",huge+9n,huge+20n)]};
const large=buildArtifactMembers({},identity,largeStorage);assert.deepEqual(large.overlap_groups[0].storage_refs,["a","b"]);validateArtifactMembers(large,identity,largeStorage);
assert.equal(large.storage_bindings[0].bounds_assessment,"outside_artifact_member");
// Independent all-pairs connectivity oracle, including empty, touching and transitive intervals.
let seed=1471;const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
for(let trial=0;trial<200;trial++){
 const intervals=Array.from({length:20},(_,i)=>{const a=random(50);return {id:`s${String(i).padStart(2,"0")}`,a,b:a+random(10)};});
 const adjacency=intervals.map((a,i)=>intervals.flatMap((b,j)=>i!==j&&a.a<a.b&&b.a<b.b&&a.a<b.b&&b.a<a.b?[j]:[]));
 const visited=new Set(),expected=[];
 for(let i=0;i<intervals.length;i++){if(visited.has(i))continue;const stack=[i],component=[];while(stack.length){const j=stack.pop();if(visited.has(j))continue;visited.add(j);component.push(intervals[j].id);stack.push(...adjacency[j]);}if(component.length>1)expected.push(component.sort());}
 const rows=intervals.map(x=>({storage_ref:x.id,member_ref:"f",range:{status:"exact",offset_basis:"artifact_absolute",start:String(x.a),end_exclusive:String(x.b)}}));
 const actual=ir("member_overlaps",{bindings:rows}).map(x=>x.storage_refs.sort());
 assert.deepEqual(actual.sort((a,b)=>a[0].localeCompare(b[0])),expected.sort((a,b)=>a[0].localeCompare(b[0])));
}
const unknown=integerMetric({value:null,unit:"MAC",sourceRefs:["u"],assessed:0,eligible:1});
const known=integerMetric({value:exactInteger(huge),unit:"MAC",sourceRefs:["k"],assessed:1,eligible:1});
assert.equal(aggregateIntegerMetrics([unknown,known],"MAC").total.value,null);
assert.equal(aggregateIntegerMetrics([unknown,known],"MAC").assessed_subtotal.decimal,String(huge));
assert.throws(()=>aggregateIntegerMetrics([known,known],"MAC"),/twice/);
assert.throws(()=>aggregateIntegerMetrics([known],"byte"),/units/);
for(const shape of [[],[0,"N"],[2,3],["9007199254740993",2],["N",3]]){
 const t=buildValueType({dtype:"FLOAT32",shape,shape_declared:true},"onnx","s");validateValueType(t);
 const expected=shape.includes(0)?0n:shape.includes("N")?null:shape.reduce((a,b)=>a*BigInt(b),1n);
 assert.equal(valueElementCount(t)?.decimal??null,expected===null?null:String(expected));
}
const invalid=buildValueType({dtype:"FLOAT32",shape:[0,2],shape_declared:true},"onnx","s");invalid.root.dimensions[1].value.number=3;assert.throws(()=>valueElementCount(invalid));
const scalar=buildValueType({dtype:"FLOAT32",shape:[],shape_declared:true},"onnx","s");
const g={id:"v",name:"v",dtype:"FLOAT32",shape:[],type_contract:scalar,storage_refs:[]};
const inventory={schema:"deepbom.logical_inventory.v1",count:1,values:[{...g,graph_value_ref:"v",element_count:exactInteger(1)}]};
validateLogicalInventory(inventory,[g],[]);assert.throws(()=>validateLogicalInventory(inventory,[g,g],[]),/identit/);
const future=buildCoreMlDeploymentContract({specificationVersion:9,modelType:"mlProgram",mlProgram:{functions:{main:{opset:"FutureCoreML"}}}});
assert.equal(future.status,"not_assessed_unmapped_feature_floor");assert.equal(future.observed_feature_floor,null);
const invalidFloor=buildCoreMlDeploymentContract({specificationVersion:6,modelType:"mlProgram",description:{inputs:[{dtype:"INT8"}]}});assert.equal(invalidFloor.status,"invalid_declared_version_below_observed_feature_floor");
// Fallback positions must come from the original tensor array, not the filtered array.
const analysis={format:"onnx",model_sha256:"a".repeat(64),onnx_shape_inference:{schema:"s",declaration_conflicts:[{node_index:null,tensor_name:"bad"}]},dynamic_shape_cost_contract:{schema:"d"},ops:[{outputs:[2]}],tensors:[{name:"a"},{name:"b"},{name:"bad",contract_status:"invalid"}]};
const capsule=buildOnnxContractConflictCapsule(analysis);assert.equal(capsule.affected_values[0].tensor_index,2);assert.match(capsule.affected_values[0].subject_ref,/:2$/);assert.equal(capsule.root_conflicts[0].subject_ref,null);
const inputs=[{value_ref:"v",native_locator:null,dtype:"F32",shape:[1],values_sha256:"b".repeat(64)}];
const environment={runtime:"test"};
const p={schema:"deepbom.runtime_provenance.v1",environment,environment_sha256:canonicalEvidenceDigest(environment),input_manifest_sha256:inputManifestSha256(inputs),code_sha256:null,trace_sha256:null,signature:null,trust:"declared_not_attested"};
validateRuntimeProvenance(p,inputs,{});assert.throws(()=>validateRuntimeProvenance(p,[{...inputs[0],shape:[2]}],{}),/input digest/);
assert.throws(()=>validateRuntimeProvenance({...p,environment:{runtime:"other"}},inputs,{}),/environment digest/);
console.log("WASM IR semantics: 200 independent overlap graphs, exact/unknown counts, member and logical bindings, Core ML unknown floor, ONNX indices and runtime identity passed.");

for (const value of [0,1,"001",9007199254740991,"9007199254740993",9007199254740993n,-1,1.5,null,NaN,Infinity,{},true]) {
  const valid=typeof value === "bigint" ? value>=0n : typeof value === "number" ? Number.isSafeInteger(value)&&value>=0 : typeof value === "string" && /^\d+$/.test(value);
  const n=valid?BigInt(value):null;
  assert.deepEqual(exactInteger(value),n===null?null:{decimal:String(n),number:n<=9007199254740991n?Number(n):null});
}
assert.equal(exactCountValue({decimal:"2",number:2,value:2}),2n);assert.equal(exactCountValue({decimal:"2",value:3}),null);assert.equal(exactCountValue("01"),null);
assert.equal(safeBigIntNumber(-9007199254740991n),-9007199254740991);assert.equal(safeBigIntNumber(-9007199254740992n),null);
assert.throws(()=>validateExactInteger({decimal:"3",number:4}),/mirror/);

assert.throws(()=>ir("member_overlaps",{bindings:[{storage_ref:"x",member_ref:"f",range:{status:"exact",offset_basis:"artifact_absolute",start:{decimal:"0"},end_exclusive:1}}]}),/range/);
for (const input of [{description:{inputs:[null]}},{description:{inputs:{}}},{mlProgram:{functions:{main:null}}},{mlProgram:{functions:{main:{opset:9}}}}]) {
 assert.throws(()=>buildCoreMlDeploymentContract({specificationVersion:9,modelType:"mlProgram",...input}),/Core ML/);
}
