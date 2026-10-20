import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {initializeNumericalWasm} from '../web/lib/numerical-wasm.js';
import {portableContract} from '../web/lib/portable-contracts.js';
import {canonicalJson,canonicalJsonInput} from '../web/lib/report-utils.js';
import {canonicalEvidenceDigest,canonicalEvidenceJson,sealDocument,subjectReference} from '../web/lib/evidence-identity.js';
import {syntheticScenario} from '../examples/evidence-workflow/synthetic.mjs';
import {validateSnapshotIr,snapshotFromManifest} from '../web/lib/snapshot-ir.js';
import {validateEvaluationProtocol,validateEvaluationRecord,validateMeasurementAgainstProtocol} from '../web/lib/evaluation-evidence.js';
import {buildDeploymentEvidence} from '../web/lib/deployment-evidence.js';
import {reportedDecimal,addExactDecimals,subtractExactDecimals,negateExactDecimal,halveExactDecimal,exactDecimalText,exactDecimalNumber,compareExactDecimals} from '../web/lib/exact-rational.js';
await initializeNumericalWasm();
const digest=text=>createHash('sha256').update(text).digest('hex');
// ECMAScript itself is the independent JCS number spelling oracle. Probe all
// exponent ranges, including subnormals; preserve binary64 across JSON transport.
let seed=0x243f6a8885a308d3n;
const random=()=>{seed^=seed<<13n;seed^=seed>>7n;seed^=seed<<17n;seed=BigInt.asUintN(64,seed);return seed;};
const view=new DataView(new ArrayBuffer(8)),numbers=[-0,Number.MIN_VALUE,Number.MAX_VALUE,1e-7,1e-6,1e20,1e21,333333333.33333329];
for(let i=0;i<16384;i++){view.setBigUint64(0,random());const n=view.getFloat64(0);if(Number.isFinite(n))numbers.push(n);}
for(let i=0;i<numbers.length;i+=256){const a=numbers.slice(i,i+256);assert.equal(canonicalEvidenceJson(a),JSON.stringify(a));assert.equal(canonicalEvidenceDigest(a),digest(JSON.stringify(a)));}
const unicode=Object.fromEntries(['\r','1','€','😀','\u0080','ö','דּ','\ue000','10','2','\u0000'].map((k,i)=>[k,i]));
const tricky=[unicode,JSON.parse('{"__proto__":{"x":1},"constructor":true}'),{slash:'\\/\b\t\n\f\r"\u0000',text:'é\u2028\u2029'},Object.assign(Object.create(null),{a:1}),{a:[null,true,false,{},[]]}];
for(const input of tricky){assert.equal(canonicalEvidenceJson(input),canonicalJson(input));assert.equal(canonicalEvidenceDigest(input),digest(canonicalJson(input)));}
const array=[1,2];array.toJSON=()=>['not called'];assert.equal(canonicalEvidenceJson(array),'[1,2]');
let reads=0;assert.equal(canonicalEvidenceJson({get x(){reads++;return 1;}}),'{"x":1}');assert.equal(reads,1);
const cyclic={};cyclic.self=cyclic;
for(const input of [NaN,Infinity,undefined,1n,Symbol('x'),()=>0,'\ud800',{'\udfff':1},new Date(),new Uint8Array([1]),cyclic,new Array(1),{a:undefined},{toJSON(){return 1;}}])assert.throws(()=>canonicalEvidenceDigest(input),/RFC 8785/);
assert.equal(Object.getPrototypeOf(canonicalJsonInput(JSON.parse('{"__proto__":1}'))),null);
// Valid old schemas are unchanged. Every dropped required member is rejected
// even after a new valid content hash is supplied; hashes alone are insufficient.
const mutable={z:{items:[1]},a:2};
const sealed=sealDocument(mutable,'sha256');
assert.equal(JSON.stringify(sealed),JSON.stringify({...mutable,sha256:canonicalEvidenceDigest(mutable)}));
mutable.z.items.push(2);assert.deepEqual(sealed.z.items,[1]);
assert.equal(sealed.sha256,canonicalEvidenceDigest({z:{items:[1]},a:2}));
let sealReads=0;
const once=sealDocument({get nested(){sealReads++;return {value:sealReads};}},'sha256');
assert.equal(sealReads,1);assert.equal(once.sha256,canonicalEvidenceDigest({nested:{value:1}}));
const s=syntheticScenario();
for(const [base,key,validate] of [[s.protocol,'protocol_sha256',validateEvaluationProtocol],[s.records[0],'evaluation_sha256',validateEvaluationRecord],[s.releases[0],'snapshot_ir_sha256',validateSnapshotIr]]){
  validate(base);
  for(const field of Object.keys(base)){
    if(field===key)continue;
    const {[key]:ignored,...body}=structuredClone(base);delete body[field];
    assert.throws(()=>validate(sealDocument(body,key)),undefined,`removed ${field}`);
  }
  const {[key]:ignored,...body}=base;assert.throws(()=>validate(sealDocument({...body,extra:true},key)));
}
const reseal=(doc,key)=>{const {[key]:ignored,...body}=doc;return sealDocument(body,key);};
for(const denominator of ['9007199254740993','9'.repeat(100)]){const r=structuredClone(s.records[0]);r.metrics[0].denominator=denominator;validateEvaluationRecord(reseal(r,'evaluation_sha256'));}
for(const denominator of ['01','-1','1.5','0','9'.repeat(101),1,null]){const r=structuredClone(s.records[0]);r.metrics[0].denominator=denominator;assert.throws(()=>validateEvaluationRecord(reseal(r,'evaluation_sha256')));}
const changed=structuredClone(s.records[0]);changed.metrics[0].unit='other';assert.throws(()=>validateMeasurementAgainstProtocol(reseal(changed,'evaluation_sha256'),s.protocol),/unit/);
for(const path of ['/root','../escape','x/../escape','x//y','C:escape','a\\b','a\0b']){const r=structuredClone(s.releases[0]);r.contents[0].path=path;assert.throws(()=>validateSnapshotIr(reseal(r,'snapshot_ir_sha256')),/path/);}
const config=snapshotFromManifest({schema:'deepbom.configuration_input.v1',identity:{namespace:'test',name:'runtime',version:'1'},scope:{fixed_aspects:['runtime'],excluded_aspects:['use'],consistency:'manifest_declared_scope'},contents:[{id:'bytes',role:'settings',ref:s.files[0].ref}]});
const ref=subjectReference('snapshot',config.snapshot_ir_sha256,{schema:config.schema});
const input={schema:'deepbom.deployment_context_input.v1',evaluated_configuration:ref,deployments:[{id:'d',configuration:ref,site:'site',start:'1970-01-01T00:00:01Z',end:'1970-01-01T00:00:02Z',source:s.files[0].ref}],execution:{event_id:'e',site:'site',at:'1970-01-01T00:00:01.500Z',clock_uncertainty_ms:0,configuration:ref,output_state:'generated',source:s.files[0].ref}};
const context={document:()=>config,resolve:ref=>({ref,status:'matched'})};
// Independent enumerated half-open [start,end) interval oracle. Uncertainty
// must be fully inside a unique deployment; touching its end is ambiguous.
let intervals=0;
for(let ms=900;ms<=2100;ms+=25)for(const u of [0,1,50,100,500]){
 const x=structuredClone(input);x.execution.at=new Date(ms).toISOString();x.execution.clock_uncertainty_ms=u;
 const result=buildDeploymentEvidence(x,context),candidate=1000<=ms+u&&2000>ms-u,exact=candidate&&1000<=ms-u&&2000>ms+u;
 assert.deepEqual(result.candidate_deployments,candidate?['d']:[]);assert.equal(result.status,exact?'consistent_declared_execution':'ambiguous_or_conflicting_deployment');intervals++;
}
const missing=buildDeploymentEvidence(input,{...context,resolve:ref=>({ref,status:'unresolved'})});assert.equal(missing.status,'unresolved_evidence');
const duplicate=structuredClone(input);duplicate.deployments.push({...duplicate.deployments[0],id:'other'});assert.equal(buildDeploymentEvidence(duplicate,context).status,'ambiguous_or_conflicting_deployment');
const noExecution=structuredClone(input);noExecution.execution=null;assert.equal(buildDeploymentEvidence(noExecution,context).status,'execution_not_provided');
const wide=structuredClone(input);wide.execution.clock_uncertainty_ms=Number.MAX_SAFE_INTEGER;assert.throws(()=>buildDeploymentEvidence(wide,context),/exact time/);
// Independent scaled-integer decimal oracle (bounded decimal inputs), plus
// limits that cannot be represented as ordinary finite JSON numbers.
for(let i=-100;i<=100;i++)for(let j=-7;j<=7;j++){
 const a=reportedDecimal(i/10),b=reportedDecimal(j/100),delta=subtractExactDecimals(a,b);
 assert.equal(compareExactDecimals(delta,reportedDecimal((i*10-j)/100)),0);
 assert.equal(exactDecimalNumber(delta),(i*10-j)/100||0);
 assert.equal(compareExactDecimals(addExactDecimals(delta,b),a),0);
 assert.equal(compareExactDecimals(negateExactDecimal(delta),subtractExactDecimals(b,a)),0);
 assert.equal(compareExactDecimals(addExactDecimals(halveExactDecimal(a),halveExactDecimal(a)),a),0);
}
assert.equal(exactDecimalText(subtractExactDecimals(reportedDecimal(.4),reportedDecimal(.1))),'0.3');
assert.equal(exactDecimalText(halveExactDecimal(reportedDecimal(5e-324))),'0.'+'0'.repeat(323)+'25');
assert.equal(exactDecimalNumber(halveExactDecimal(reportedDecimal(5e-324))),5e-324);
assert.equal(exactDecimalNumber(subtractExactDecimals(reportedDecimal(1e308),reportedDecimal(-1e308))),null);
assert.equal(exactDecimalNumber({coefficient:1n,exponent:-400}),null);
assert.throws(()=>portableContract('decimal',{operation:'text',a:{coefficient:'1',exponent:1000000}}),/budget/);
console.log(`Portable evidence: ${numbers.length} binary64 JCS vectors; strict host admission; semantic resealing; ${intervals} deployment intervals; 3015 exact decimal pairs passed.`);
