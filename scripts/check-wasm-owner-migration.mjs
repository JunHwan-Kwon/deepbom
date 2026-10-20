import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFileSync} from "node:fs";
import {numericalWasm} from "../web/lib/numerical-wasm.js";
import {portableContract} from "../web/lib/portable-contracts.js";
import {decodeNativeTensor} from "../web/lib/native-evidence.js";
import {seal,checkDigest,exactKeys,decimalCount} from "../web/lib/numerical-ir/common.js";
import {canonicalJson} from "../web/lib/report-utils.js";
import {makeOnnxTensorType as tensor,makeOnnxSequenceType as sequence,makeOnnxMapType as map,makeOnnxOptionalType as optional,unionOnnxTypeProtos as union,unifyOnnxTypeProtos as unify} from "../web/lib/onnx-type-proto.js";
import {validateReferenceProducer} from "./reference-contract-provenance.mjs";
const sha=v=>createHash("sha256").update(v).digest("hex");
for(const bytes of [Buffer.alloc(0),Buffer.from("abc"),Buffer.from(Array.from({length:65537},(_,i)=>i%256))]) assert.equal(numericalWasm().evidence_sha256(bytes),sha(bytes));
for(const value of [{"😀":1,"\ue000":2},[5e-324,1e21,-0,1e-7],{a:null,b:false}]) {
 const sealed=seal(value,"sha256"); // arrays are not document bodies; test canonical operation separately
 assert.equal(portableContract("evidence_identity",{operation:"digest",value}),sha(canonicalJson(value)));
 if(!Array.isArray(value)){checkDigest(sealed,"sha256");assert.throws(()=>checkDigest({...sealed,a:"tampered"},"sha256"));}
}
assert.throws(()=>exactKeys({a:1,toJSON(){return {a:1};}},["a"],"row"));
assert.throws(()=>exactKeys({a:1,b:2},["a"],"row"));assert.throws(()=>exactKeys({},["a"],"row"));exactKeys({a:null},["a"],"row");
for(const invalid of ["01","+1","-1",1,"",null])assert.throws(()=>decimalCount(invalid));
assert.equal(decimalCount("900719925474099312345"),900719925474099312345n);
const base={id:"w",name:"w",role:"parameter",trainable:true,aliases:[],dtype:"F32",shape:[2],byte_order:"little",data_base64:Buffer.alloc(8).toString("base64")};
const decoded=decodeNativeTensor(base);assert.equal(decoded.identity.payload_sha256,sha(Buffer.alloc(8)));assert.equal(decoded.count,2n);
for(const changes of [{aliases:["w"]},{shape:[0,null]},{shape:[3]},{byte_order:"big"},{dtype:"UNKNOWN"},{trainable:1}])assert.throws(()=>decodeNativeTensor({...base,...changes}));
const identity=decoded.identity;portableContract("native_tensor",{operation:"identity",value:identity});
for(const changes of [{byte_length:"7"},{element_count:"02"},{payload_sha256:"g".repeat(64)},{aliases:["shared","shared"]}])assert.throws(()=>portableContract("native_tensor",{operation:"identity",value:{...identity,...changes}}));
for(const wrap of [x=>x,sequence,optional,x=>map("INT64",x)]){
 const a=wrap(tensor("FLOAT32",[2,3],true)),b=wrap(tensor("FLOAT32",[2,4],true));
 assert.equal(union([a,b]).status,"pass");assert.equal(unify(a,b).status,"fail");
 assert.deepEqual(unify(a,a).type,a);
}
const a=tensor("FLOAT32",[0,3],true);assert.deepEqual(unify(a,tensor("UNKNOWN",[0,3],true)).type,a);
assert.equal(union([a,tensor("INT8",[0,3],true)]).status,"fail");
const source=readFileSync(new URL("../web/lib/onnx-type-proto.js",import.meta.url),"utf8");assert.ok(!source.includes("function unifyDimension("));
const expected={analyzer:{buildSourceState:"dirty",rulepackSha256:"a".repeat(64)},wasm_sha256:"b".repeat(64)};
const record={schema:"deepbom.export_reference_set.v1",status:"development_reference_not_released",generated_at:"2026-10-20T00:00:00.000Z",directory:"mobilenet-v2-quant-"+"c".repeat(64),producer:expected};
validateReferenceProducer(record,expected);
const otherCommit={...expected,analyzer:{...expected.analyzer,buildCommit:"d".repeat(40),buildSourceState:"clean"}};
assert.throws(()=>validateReferenceProducer(record,otherCommit));
validateReferenceProducer(record,otherCommit,{allowBuildContext:true});
assert.throws(()=>validateReferenceProducer(record,{...otherCommit,wasm_sha256:"e".repeat(64)},{allowBuildContext:true}));
assert.throws(()=>validateReferenceProducer({...record,status:"clean_source_reference"},expected));
assert.throws(()=>validateReferenceProducer({...record,producer:{...expected,wasm_sha256:"d".repeat(64)}},expected));
console.log("WASM owner migration: native identity, exact counts, JCS/SHA, ONNX nested joins and producer mismatch guards passed.");
